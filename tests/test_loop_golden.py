"""Golden snapshot for the K=1 (sequential Karpathy) loop.

Locks the wire shape of the ledger so accidental refactors of loop.py / decide.py /
spec.py don't silently drift. To regenerate after an intentional ledger change:

    REGOLD=1 uv run pytest tests/test_loop_golden.py
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import pytest
import yaml

from auto_research.llm.proposer import Proposal, Proposer
from auto_research.loop import run as run_loop
from auto_research.runner.local import LocalRunner
from auto_research.spec import Spec
from auto_research.store.local import LocalStore

GOLDEN = Path(__file__).parent / "golden" / "k1_ledger.json"

TRAIN_PY = """
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


class _ScriptedProposer(Proposer):
    def __init__(self, values):
        self._values = list(values)
        self._cur = 0

    def propose(self, *_args, **_kwargs) -> Proposal:
        v = self._values[self._cur % len(self._values)]
        self._cur += 1
        return Proposal(
            new_source=TRAIN_PY.format(v=v),
            diff=f"# scripted v={v}\n",
            tokens_in=10,
            tokens_out=20,
            usd=0.001,
        )


def _normalize(trial_dict: dict) -> dict:
    # Strip timing-dependent fields so the snapshot is reproducible.
    out = dict(trial_dict)
    out["duration_ms"] = 0
    return out


def _make_deterministic_uuid(monkeypatch):
    counter = itertools.count(1)

    class _FakeUUID:
        def __init__(self, n: int) -> None:
            # Pad on the right so loop.py's hex[:12] / hex[:8] slices reveal the counter.
            self._hex = f"{n:x}".ljust(32, "0")

        @property
        def hex(self) -> str:
            return self._hex

    def fake_uuid4():
        return _FakeUUID(next(counter))

    monkeypatch.setattr("auto_research.loop.uuid.uuid4", fake_uuid4)


def test_k1_ledger_matches_golden(tmp_path: Path, monkeypatch) -> None:
    _make_deterministic_uuid(monkeypatch)

    (tmp_path / "train.py").write_text(TRAIN_PY.format(v=0.0))
    (tmp_path / "eval.py").write_text(EVAL_PY)
    (tmp_path / "data.csv").write_text("dummy\n1\n")
    spec_dict = {
        "objective": "maximize v",
        "data_path": "data.csv",
        "metric": {"name": "v", "direction": "maximize"},
        "daily_budget_usd": 1.0,
        "max_iterations": 5,
        "parallelism": 1,
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_dict))
    spec = Spec.load(spec_path)
    store = LocalStore(tmp_path / ".auto-research")
    proposer = _ScriptedProposer([0.1, 0.5, 0.3, 0.7, 0.2])

    run_loop(
        spec=spec,
        spec_path=spec_path,
        proposer=proposer,
        runner=LocalRunner(),
        store=store,
        initial_source=(tmp_path / "train.py").read_text(),
    )

    history = store.read_history()
    snapshot = [_normalize(t.model_dump()) for t in history]

    if os.environ.get("REGOLD"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(snapshot, indent=2) + "\n")
        pytest.skip(f"wrote golden snapshot to {GOLDEN}")

    assert GOLDEN.exists(), (
        f"golden snapshot missing at {GOLDEN}; "
        "regenerate with: REGOLD=1 uv run pytest tests/test_loop_golden.py"
    )
    expected = json.loads(GOLDEN.read_text())
    assert snapshot == expected, (
        "K=1 ledger drifted from golden snapshot. If the change is intentional, "
        "regenerate with: REGOLD=1 uv run pytest tests/test_loop_golden.py"
    )
