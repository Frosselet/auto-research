from __future__ import annotations

import json
import shutil
from pathlib import Path

from auto_research.store.base import Store
from auto_research.types import Trial


class LocalStore(Store):
    """Filesystem-backed store.

    Layout under workdir/:
        ledger.jsonl                 # one Trial JSON per line
        best/train.py                # current best-so-far source
        best/artifact/               # current best-so-far model artifact
        candidates/<trial_id>/
            train.py                 # candidate source for that trial
            artifact/                # candidate model artifact
        iter/train.py                # the working copy the Runner executes each trial
    """

    def __init__(self, workdir: Path):
        self._workdir = workdir
        self._workdir.mkdir(parents=True, exist_ok=True)
        (self._workdir / "best" / "artifact").mkdir(parents=True, exist_ok=True)
        (self._workdir / "candidates").mkdir(parents=True, exist_ok=True)
        (self._workdir / "iter").mkdir(parents=True, exist_ok=True)
        self._ledger = self._workdir / "ledger.jsonl"
        self._ledger.touch(exist_ok=True)

    @property
    def workdir(self) -> Path:
        return self._workdir

    def working_train_path(self) -> Path:
        return self._workdir / "iter" / "train.py"

    def candidate_artifact_dir(self, trial_id: str) -> Path:
        d = self._workdir / "candidates" / trial_id / "artifact"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def best_artifact_dir(self) -> Path:
        return self._workdir / "best" / "artifact"

    def promote_candidate(self, trial_id: str, new_source: str) -> None:
        (self._workdir / "best" / "train.py").write_text(new_source)
        cand = self._workdir / "candidates" / trial_id / "artifact"
        best = self.best_artifact_dir()
        if cand.exists():
            if best.exists():
                shutil.rmtree(best)
            shutil.copytree(cand, best)

    def read_best_source(self) -> str | None:
        p = self._workdir / "best" / "train.py"
        return p.read_text() if p.exists() else None

    def append_trial(self, trial: Trial) -> None:
        with self._ledger.open("a") as f:
            f.write(trial.model_dump_json() + "\n")

    def read_history(self) -> list[Trial]:
        if not self._ledger.exists():
            return []
        out: list[Trial] = []
        for line in self._ledger.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(Trial.model_validate_json(line))
        return out
