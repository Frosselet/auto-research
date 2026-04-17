"""End-to-end loop test with a mock proposer — no network required."""
from __future__ import annotations

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


class ScriptedProposer(Proposer):
    """Returns a sequence of scripted train.py sources so we can control the metric trajectory."""

    def __init__(self, values: list[float]):
        self._values = list(values)
        self._cur = 0

    def propose(
        self, objective, current_source, history, best_metric, metric_direction
    ) -> Proposal:
        v = self._values[self._cur % len(self._values)]
        self._cur += 1
        src = TRAIN_PY_TEMPLATE.format(v=v)
        return Proposal(
            new_source=src,
            diff=f"# scripted v={v}\n",
            tokens_in=10,
            tokens_out=20,
            usd=0.001,
        )


def _setup_recipe(tmp_path: Path, budget: float = 1.0, max_iter: int = 5) -> Path:
    (tmp_path / "train.py").write_text(TRAIN_PY_TEMPLATE.format(v=0.0))
    (tmp_path / "eval.py").write_text(EVAL_PY)
    (tmp_path / "data.csv").write_text("dummy\n1\n")
    spec = {
        "objective": "maximize v",
        "data_path": "data.csv",
        "metric": {"name": "v", "direction": "maximize"},
        "daily_budget_usd": budget,
        "max_iterations": max_iter,
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    return spec_path


def test_loop_improves_and_keeps_best(tmp_path: Path) -> None:
    spec_path = _setup_recipe(tmp_path, max_iter=5)
    spec = Spec.load(spec_path)
    store = LocalStore(tmp_path / ".auto-research")
    proposer = ScriptedProposer([0.1, 0.5, 0.3, 0.7, 0.2])
    state = run_loop(
        spec=spec,
        spec_path=spec_path,
        proposer=proposer,
        runner=LocalRunner(),
        store=store,
        initial_source=(tmp_path / "train.py").read_text(),
    )
    assert state.iteration == 5
    assert state.best_metric == 0.7
    history = store.read_history()
    assert len(history) == 5
    kept = [t for t in history if t.kept]
    assert [t.metric for t in kept] == [0.1, 0.5, 0.7]
    # usd_per_bp is recorded for improvements
    assert kept[1].usd_per_bp is not None


def test_loop_respects_budget(tmp_path: Path) -> None:
    # Each trial costs ~0.001. Budget 0.0025 → ~2 trials then budget stop.
    spec_path = _setup_recipe(tmp_path, budget=0.0025, max_iter=10)
    spec = Spec.load(spec_path)
    store = LocalStore(tmp_path / ".auto-research")
    proposer = ScriptedProposer([0.1, 0.2, 0.3, 0.4])
    state = run_loop(
        spec=spec,
        spec_path=spec_path,
        proposer=proposer,
        runner=LocalRunner(),
        store=store,
        initial_source=(tmp_path / "train.py").read_text(),
    )
    assert state.iteration < 10
    assert state.usd_spent >= 0.0025 or state.remaining_budget(spec.daily_budget_usd) == 0


def test_loop_resumes_from_existing_ledger(tmp_path: Path) -> None:
    spec_path = _setup_recipe(tmp_path, max_iter=2)
    spec = Spec.load(spec_path)
    store = LocalStore(tmp_path / ".auto-research")
    proposer = ScriptedProposer([0.1, 0.5])
    run_loop(
        spec=spec,
        spec_path=spec_path,
        proposer=proposer,
        runner=LocalRunner(),
        store=store,
        initial_source=(tmp_path / "train.py").read_text(),
    )
    # New session: best_source should be read back from the store.
    store2 = LocalStore(tmp_path / ".auto-research")
    proposer2 = ScriptedProposer([0.8])
    spec2 = Spec.model_validate({**spec.model_dump(), "max_iterations": 1})
    state2 = run_loop(
        spec=spec2,
        spec_path=spec_path,
        proposer=proposer2,
        runner=LocalRunner(),
        store=store2,
        initial_source="# ignored — best source is in the store\n",
    )
    assert state2.best_metric == 0.8
    assert len(store2.read_history()) == 3
