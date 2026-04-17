from __future__ import annotations

import uuid
from pathlib import Path

from auto_research.llm.proposer import Proposer
from auto_research.logging import trial_logger
from auto_research.runner.base import Runner
from auto_research.spec import Spec
from auto_research.states.decide import decide
from auto_research.states.evaluate import evaluate
from auto_research.states.propose import propose
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
    """Execute the Karpathy loop until budget or max_iterations is exhausted."""
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

        state.iteration += 1
        trial = Trial(trial_id=uuid.uuid4().hex[:12], parent_id=state.best_trial_id)

        trial, proposal = propose(
            proposer=proposer,
            objective=spec.objective,
            current_source=best_source,
            history=state.history,
            best_metric=state.best_metric,
            metric_direction=spec.metric.direction,
            trial=trial,
        )

        state.usd_spent += proposal.usd
        if state.usd_spent > spec.daily_budget_usd:
            trial.status = "failed"
            trial.error = "budget exceeded after proposal"
            trial.reason = "budget exceeded"
            store.append_trial(trial)
            log.log_trial(trial)
            break

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
        trial = decide(
            trial=trial,
            best_metric=state.best_metric,
            metric_direction=spec.metric.direction,
        )

        state.history.append(trial)
        store.append_trial(trial)
        log.log_trial(trial)

        if trial.kept and trial.metric is not None:
            state.best_metric = trial.metric
            state.best_trial_id = trial.trial_id
            best_source = proposal.new_source
            state.best_source = best_source
            store.promote_candidate(trial.trial_id, best_source)

    return state
