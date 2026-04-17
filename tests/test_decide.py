from __future__ import annotations

import pytest

from auto_research.states.decide import decide
from auto_research.types import Trial


def _trial(metric: float | None = None, status: str = "evaluated", usd: float = 0.01) -> Trial:
    return Trial(trial_id="t1", metric=metric, status=status, usd=usd)


def test_first_trial_becomes_baseline() -> None:
    t = decide(_trial(metric=1.2), best_metric=None, metric_direction="maximize")
    assert t.kept
    assert t.delta is None
    assert "baseline" in t.reason


def test_maximize_keep_if_higher() -> None:
    t = decide(_trial(metric=1.5), best_metric=1.2, metric_direction="maximize")
    assert t.kept
    assert t.delta == pytest.approx(0.3)
    assert t.usd_per_bp is not None


def test_maximize_discard_if_lower() -> None:
    t = decide(_trial(metric=1.0), best_metric=1.2, metric_direction="maximize")
    assert not t.kept
    assert t.delta < 0


def test_minimize_keep_if_lower() -> None:
    t = decide(_trial(metric=0.8), best_metric=1.0, metric_direction="minimize")
    assert t.kept


def test_failed_trial_is_not_kept() -> None:
    t = _trial(metric=None, status="failed")
    t.error = "train.py crashed"
    t = decide(t, best_metric=1.0, metric_direction="maximize")
    assert not t.kept
    assert "crashed" in t.reason
