from __future__ import annotations

from pathlib import Path

from auto_research.store.local import LocalStore
from auto_research.types import Trial


def test_append_and_read_roundtrip(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    t1 = Trial(trial_id="a", metric=0.5, kept=True, status="decided", usd=0.01)
    t2 = Trial(trial_id="b", metric=0.4, kept=False, status="decided", usd=0.02)
    store.append_trial(t1)
    store.append_trial(t2)
    history = store.read_history()
    assert [t.trial_id for t in history] == ["a", "b"]
    assert history[0].kept and not history[1].kept


def test_promote_candidate_writes_best_source_and_artifact(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    cand_dir = store.candidate_artifact_dir("a")
    (cand_dir / "model.pkl").write_bytes(b"hello")
    store.promote_candidate("a", new_source="# best source\n")
    assert (tmp_path / "best" / "train.py").read_text() == "# best source\n"
    assert (tmp_path / "best" / "artifact" / "model.pkl").read_bytes() == b"hello"


def test_read_best_source_none_before_first_keep(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    assert store.read_best_source() is None
