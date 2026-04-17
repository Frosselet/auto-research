"""Parallel-cohort variant of the Karpathy loop. parallelism=K runs K trials per round
and at most one is kept (the cohort tournament).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml

from auto_research.llm.proposer import Proposal, Proposer
from auto_research.loop import run as run_loop
from auto_research.runner.local import LocalRunner
from auto_research.spec import Spec
from auto_research.store.local import LocalStore

TRAIN_PY_TEMPLATE = """
import json, os
from pathlib import Path
artifact = Path(os.environ["AUTORESEARCH_ARTIFACT_DIR"])
artifact.mkdir(parents=True, exist_ok=True)
(artifact / "value.json").write_text(json.dumps({{"v": {v}}}))
"""

EVAL_PY = """
import json, os
from pathlib import Path
artifact = Path(os.environ["AUTORESEARCH_ARTIFACT_DIR"])
v = json.loads((artifact / "value.json").read_text())["v"]
(artifact / "metric.json").write_text(json.dumps({"metric": float(v)}))
"""


class CohortProposer(Proposer):
    """Returns scripted cohorts of K proposals per call."""

    def __init__(self, cohorts: list[list[float]]):
        self._cohorts = list(cohorts)
        self._cur = 0

    def propose(self, *args, **kwargs) -> Proposal:  # pragma: no cover
        raise AssertionError("test should call propose_batch, not propose")

    def propose_batch(self, *, k: int, **_kwargs) -> list[Proposal]:
        cohort = self._cohorts[self._cur]
        self._cur += 1
        assert len(cohort) == k, f"scripted cohort size {len(cohort)} != requested k={k}"
        return [
            Proposal(
                new_source=TRAIN_PY_TEMPLATE.format(v=v),
                diff=f"# scripted v={v}\n",
                tokens_in=10,
                tokens_out=20,
                usd=0.001,
            )
            for v in cohort
        ]


def _setup_recipe(tmp_path: Path, parallelism: int, max_iter: int, budget: float = 1.0) -> Path:
    (tmp_path / "train.py").write_text(TRAIN_PY_TEMPLATE.format(v=0.0))
    (tmp_path / "eval.py").write_text(EVAL_PY)
    (tmp_path / "data.csv").write_text("dummy\n1\n")
    spec = {
        "objective": "maximize v",
        "data_path": "data.csv",
        "metric": {"name": "v", "direction": "maximize"},
        "daily_budget_usd": budget,
        "max_iterations": max_iter,
        "parallelism": parallelism,
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    return spec_path


def test_parallel_k3_runs_three_per_round_and_keeps_only_winner(tmp_path: Path) -> None:
    spec_path = _setup_recipe(tmp_path, parallelism=3, max_iter=6)
    spec = Spec.load(spec_path)
    store = LocalStore(tmp_path / ".auto-research")
    proposer = CohortProposer([[0.1, 0.4, 0.2], [0.5, 0.3, 0.7]])
    state = run_loop(
        spec=spec,
        spec_path=spec_path,
        proposer=proposer,
        runner=LocalRunner(),
        store=store,
        initial_source=(tmp_path / "train.py").read_text(),
    )
    assert state.iteration == 6
    assert state.round == 2
    assert state.best_metric == 0.7

    history = store.read_history()
    assert len(history) == 6

    # Two rounds, each cohort_size=3.
    by_round: dict[str, list] = defaultdict(list)
    for t in history:
        assert t.cohort_size == 3
        by_round[t.round_id].append(t)
    assert len(by_round) == 2
    for cohort in by_round.values():
        assert len(cohort) == 3
        kept = [t for t in cohort if t.kept]
        assert len(kept) == 1, "cohort tournament must keep at most one trial per round"

    # Round 1 winner = 0.4 (largest, becomes baseline).
    # Round 2 winner = 0.7 (largest beating 0.4).
    kept_metrics = sorted(t.metric for t in history if t.kept)
    assert kept_metrics == [0.4, 0.7]


def test_parallel_loser_reason_names_winner(tmp_path: Path) -> None:
    """Sibling improvers in the same round that lose the tournament should explain why."""
    spec_path = _setup_recipe(tmp_path, parallelism=3, max_iter=3)
    spec = Spec.load(spec_path)
    store = LocalStore(tmp_path / ".auto-research")
    # All three improve over None (first round). Largest (0.5) wins; 0.3 and 0.4 lose.
    proposer = CohortProposer([[0.3, 0.5, 0.4]])
    run_loop(
        spec=spec,
        spec_path=spec_path,
        proposer=proposer,
        runner=LocalRunner(),
        store=store,
        initial_source=(tmp_path / "train.py").read_text(),
    )
    history = store.read_history()
    winner = next(t for t in history if t.metric == 0.5)
    losers = [t for t in history if t.metric in (0.3, 0.4)]
    assert winner.kept
    for t in losers:
        assert not t.kept
        assert "lost cohort tournament" in t.reason
        assert winner.trial_id in t.reason
