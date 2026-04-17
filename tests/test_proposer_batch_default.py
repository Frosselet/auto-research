"""The Proposer ABC default propose_batch loops propose() k times."""
from __future__ import annotations

import pytest

from auto_research.llm.proposer import Proposal, Proposer


class CountingProposer(Proposer):
    def __init__(self) -> None:
        self.calls = 0

    def propose(self, *args, **kwargs) -> Proposal:
        self.calls += 1
        return Proposal(
            new_source=f"# call {self.calls}\n",
            diff="",
            tokens_in=1,
            tokens_out=2,
            usd=0.0001,
        )


def test_propose_batch_default_calls_propose_k_times() -> None:
    proposer = CountingProposer()
    out = proposer.propose_batch(
        k=4,
        objective="x",
        current_source="",
        history=[],
        best_metric=None,
        metric_direction="maximize",
    )
    assert proposer.calls == 4
    assert len(out) == 4
    assert [p.new_source for p in out] == [
        "# call 1\n",
        "# call 2\n",
        "# call 3\n",
        "# call 4\n",
    ]


def test_propose_batch_k1_returns_single_proposal() -> None:
    proposer = CountingProposer()
    out = proposer.propose_batch(
        k=1,
        objective="x",
        current_source="",
        history=[],
        best_metric=None,
        metric_direction="maximize",
    )
    assert len(out) == 1
    assert proposer.calls == 1


def test_propose_batch_rejects_k_below_one() -> None:
    proposer = CountingProposer()
    with pytest.raises(ValueError):
        proposer.propose_batch(
            k=0,
            objective="x",
            current_source="",
            history=[],
            best_metric=None,
            metric_direction="maximize",
        )
