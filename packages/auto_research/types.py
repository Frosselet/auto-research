from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Trial(BaseModel):
    """One iteration of the Karpathy loop. Written as a single line to the ledger."""

    trial_id: str
    parent_id: str | None = Field(
        default=None, description="trial_id of the best-so-far this trial was proposed against."
    )
    round_id: str | None = Field(
        default=None,
        description="Identifier of the round this trial was proposed in. Sibling trials in the "
        "same cohort share a round_id. None for legacy ledger entries.",
    )
    cohort_size: int = Field(
        default=1,
        description="Number of sibling trials proposed in this round. 1 = sequential Karpathy loop.",
    )
    diff: str = ""
    metric: float | None = None
    best_metric_before: float | None = None
    delta: float | None = None
    kept: bool = False
    reason: str = ""
    duration_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    usd_per_bp: float | None = Field(
        default=None,
        description="$ spent on this trial divided by improvement in bps. None if not kept or delta <= 0.",
    )
    status: Literal["proposed", "trained", "evaluated", "decided", "failed"] = "proposed"
    error: str | None = None


class LoopState(BaseModel):
    """The mutable state threaded through the loop across trials."""

    spec_path: str
    workdir: str
    best_trial_id: str | None = None
    best_metric: float | None = None
    best_source: str = ""
    history: list[Trial] = Field(default_factory=list)
    usd_spent: float = 0.0
    iteration: int = Field(
        default=0, description="Total trials run (kept or not). Equals sum of cohort sizes across rounds."
    )
    round: int = Field(default=0, description="Number of completed rounds. Equals iteration when parallelism=1.")

    def remaining_budget(self, daily_budget_usd: float) -> float:
        return max(0.0, daily_budget_usd - self.usd_spent)
