from __future__ import annotations

from auto_research.types import Trial


def decide(
    trial: Trial,
    best_metric: float | None,
    metric_direction: str,
) -> Trial:
    trial.best_metric_before = best_metric
    if trial.status == "failed":
        trial.kept = False
        trial.reason = trial.error or "trial failed before evaluation"
        return trial
    if trial.metric is None:
        trial.kept = False
        trial.reason = "no metric produced"
        return trial

    if best_metric is None:
        trial.kept = True
        trial.delta = None
        trial.reason = "first successful trial becomes baseline"
    else:
        if metric_direction == "maximize":
            trial.delta = trial.metric - best_metric
            trial.kept = trial.delta > 0
        else:
            trial.delta = best_metric - trial.metric
            trial.kept = trial.delta > 0
        trial.reason = (
            f"delta={trial.delta:+.6f} vs best={best_metric:.6f}"
            + (" (kept)" if trial.kept else " (discarded)")
        )

    if trial.kept and trial.delta is not None and trial.delta > 0 and trial.usd > 0:
        bps = trial.delta * 10000
        if bps > 0:
            trial.usd_per_bp = trial.usd / bps
    trial.status = "decided"
    return trial
