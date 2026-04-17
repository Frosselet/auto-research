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

    def propose_batch(
        self,
        *,
        k: int,
        objective: str,
        current_source: str,
        history: list[Trial],
        best_metric: float | None,
        metric_direction: str,
    ) -> list[Proposal]:
        """Propose K diverse candidate edits to train.py for one round.

        Default implementation calls propose() k times sequentially — works for any
        Proposer but issues K separate LLM calls. Subclasses (e.g. OpenAIProposer)
        should override with a single LLM call returning K proposals to halve cost
        and naturally avoid duplicate proposals.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        return [
            self.propose(
                objective=objective,
                current_source=current_source,
                history=history,
                best_metric=best_metric,
                metric_direction=metric_direction,
            )
            for _ in range(k)
        ]
