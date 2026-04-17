from __future__ import annotations

import time

from auto_research.llm.proposer import Proposal, Proposer
from auto_research.types import Trial


def propose(
    proposer: Proposer,
    objective: str,
    current_source: str,
    history: list[Trial],
    best_metric: float | None,
    metric_direction: str,
    trial: Trial,
) -> tuple[Trial, Proposal]:
    t0 = time.monotonic()
    proposal = proposer.propose(
        objective=objective,
        current_source=current_source,
        history=history,
        best_metric=best_metric,
        metric_direction=metric_direction,
    )
    trial.diff = proposal.diff
    trial.tokens_in = proposal.tokens_in
    trial.tokens_out = proposal.tokens_out
    trial.usd += proposal.usd
    trial.duration_ms += int((time.monotonic() - t0) * 1000)
    trial.status = "proposed"
    return trial, proposal
