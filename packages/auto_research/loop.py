from __future__ import annotations

import time
import uuid
from pathlib import Path

from auto_research.llm.proposer import Proposal, Proposer
from auto_research.logging import trial_logger
from auto_research.runner.base import Runner
from auto_research.spec import Spec
from auto_research.states.decide import decide_round
from auto_research.states.evaluate import evaluate
from auto_research.states.train import train
from auto_research.store.base import Store
from auto_research.types import LoopState, Trial


def run(
    spec: Spec,
    spec_path: str | Path,
    proposer: Proposer,
    runner: Runner,
    store: Store,
    initial_source: str,
) -> LoopState:
    """Execute the Karpathy loop until budget or max_iterations is exhausted.

    With spec.parallelism=1 (the default), this is the original sequential loop:
    one proposal, one trial, decide, repeat. With spec.parallelism=K>1, each round
    asks the proposer for K diverse proposals (one LLM call), runs them sequentially
    in MVP-1, and picks at most one winner per round (the cohort tournament). MVP-2
    runs the K trials in parallel via Step Functions Map but is otherwise identical.
    """
    spec_path = Path(spec_path).resolve()
    log = trial_logger()

    state = LoopState(spec_path=str(spec_path), workdir=str(spec.workdir))
    for t in store.read_history():
        state.history.append(t)
        if t.kept and t.metric is not None:
            state.best_metric = t.metric
            state.best_trial_id = t.trial_id
        state.usd_spent += t.usd

    best_source = store.read_best_source() or initial_source
    state.best_source = best_source

    data_path = spec.resolve(spec_path, "data_path")
    eval_path = spec.resolve(spec_path, "eval_script")

    while state.iteration < spec.max_iterations:
        if state.remaining_budget(spec.daily_budget_usd) <= 0:
            log.info("budget_exhausted", extra={"usd_spent": state.usd_spent})
            break

        k = min(spec.parallelism, spec.max_iterations - state.iteration)
        round_id = uuid.uuid4().hex[:8]

        t0 = time.monotonic()
        proposals = proposer.propose_batch(
            k=k,
            objective=spec.objective,
            current_source=best_source,
            history=state.history,
            best_metric=state.best_metric,
            metric_direction=spec.metric.direction,
        )
        propose_ms_per_trial = int((time.monotonic() - t0) * 1000) // max(1, len(proposals))

        cohort_size = len(proposals)
        cohort_usd = sum(p.usd for p in proposals)
        state.usd_spent += cohort_usd

        if state.usd_spent > spec.daily_budget_usd:
            for proposal in proposals:
                trial = _trial_from_proposal(
                    proposal=proposal,
                    parent_id=state.best_trial_id,
                    round_id=round_id,
                    cohort_size=cohort_size,
                    propose_ms=propose_ms_per_trial,
                )
                trial.status = "failed"
                trial.error = "budget exceeded after proposal"
                trial.reason = "budget exceeded"
                state.history.append(trial)
                store.append_trial(trial)
                log.log_trial(trial)
                state.iteration += 1
            break

        trials: list[Trial] = []
        for proposal in proposals:
            trial = _trial_from_proposal(
                proposal=proposal,
                parent_id=state.best_trial_id,
                round_id=round_id,
                cohort_size=cohort_size,
                propose_ms=propose_ms_per_trial,
            )
            trial = train(
                runner=runner,
                store=store,
                candidate_source=proposal.new_source,
                data_path=data_path,
                trial=trial,
            )
            trial = evaluate(
                runner=runner,
                store=store,
                eval_script_path=eval_path,
                data_path=data_path,
                trial=trial,
            )
            trials.append(trial)

        decided = decide_round(
            trials=trials,
            best_metric=state.best_metric,
            metric_direction=spec.metric.direction,
        )

        winner: Trial | None = None
        winner_source: str | None = None
        for trial, proposal in zip(decided, proposals):
            state.history.append(trial)
            store.append_trial(trial)
            log.log_trial(trial)
            state.iteration += 1
            if trial.kept and trial.metric is not None and winner is None:
                winner = trial
                winner_source = proposal.new_source

        if winner is not None and winner_source is not None:
            state.best_metric = winner.metric
            state.best_trial_id = winner.trial_id
            best_source = winner_source
            state.best_source = best_source
            store.promote_candidate(winner.trial_id, best_source)

        state.round += 1

    return state


def _trial_from_proposal(
    *,
    proposal: Proposal,
    parent_id: str | None,
    round_id: str,
    cohort_size: int,
    propose_ms: int,
) -> Trial:
    return Trial(
        trial_id=uuid.uuid4().hex[:12],
        parent_id=parent_id,
        round_id=round_id,
        cohort_size=cohort_size,
        diff=proposal.diff,
        tokens_in=proposal.tokens_in,
        tokens_out=proposal.tokens_out,
        usd=proposal.usd,
        duration_ms=propose_ms,
        status="proposed",
    )
