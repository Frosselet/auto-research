"""Run the XGBoost reference recipe's train.py + eval.py directly on the data.

This proves the scaffold works independently of the loop/proposer.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

RECIPE = Path(__file__).parent.parent / "examples" / "gbdt-ohlcv"


@pytest.mark.skipif(not (RECIPE / "data.csv").exists(), reason="run gen_data.py first")
def test_train_and_eval_produce_sharpe(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    env = {
        **os.environ,
        "AUTORESEARCH_DATA_PATH": str(RECIPE / "data.csv"),
        "AUTORESEARCH_ARTIFACT_DIR": str(artifact),
    }

    r1 = subprocess.run([sys.executable, str(RECIPE / "train.py")], env=env, capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr

    r2 = subprocess.run([sys.executable, str(RECIPE / "eval.py")], env=env, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr

    payload = json.loads((artifact / "metric.json").read_text())
    assert "metric" in payload
    assert isinstance(payload["metric"], float)
