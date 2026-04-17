from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from auto_research.types import Trial


@dataclass
class Proposal:
    new_source: str
    diff: str
    tokens_in: int
    tokens_out: int
    usd: float


class Proposer(ABC):
    """LLM-backed proposer of the next train.py variant.

    Returns a full replacement `new_source` plus a human-readable diff for the ledger.
    We intentionally accept full-source returns (not just diffs) because diff-application
    reliability across LLM providers is uneven; the diff is recorded for humans, the
    full source is what gets executed.
    """

    @abstractmethod
    def propose(
        self,
        objective: str,
        current_source: str,
        history: list[Trial],
        best_metric: float | None,
        metric_direction: str,
    ) -> Proposal: ...
