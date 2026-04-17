from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from auto_research.spec import Spec


def _spec_dict() -> dict:
    return {
        "objective": "maximise sharpe",
        "data_path": "data.csv",
        "metric": {"name": "sharpe", "direction": "maximize"},
        "daily_budget_usd": 1.0,
    }


def test_spec_defaults(tmp_path: Path) -> None:
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(_spec_dict()))
    spec = Spec.load(p)
    assert spec.train_script == "train.py"
    assert spec.eval_script == "eval.py"
    assert spec.max_iterations == 50
    assert spec.metric.direction == "maximize"


def test_spec_rejects_absolute_data_path(tmp_path: Path) -> None:
    d = _spec_dict()
    d["data_path"] = "/absolute/path/data.csv"
    with pytest.raises(ValueError):
        Spec.model_validate(d)


def test_spec_resolve_is_relative_to_spec(tmp_path: Path) -> None:
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(_spec_dict()))
    spec = Spec.load(p)
    assert spec.resolve(p, "data_path") == (tmp_path / "data.csv").resolve()


def test_budget_must_be_positive() -> None:
    d = _spec_dict()
    d["daily_budget_usd"] = 0
    with pytest.raises(ValueError):
        Spec.model_validate(d)
