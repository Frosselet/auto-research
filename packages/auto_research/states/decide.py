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


def decide_round(
    trials: list[Trial],
    best_metric: float | None,
    metric_direction: str,
) -> list[Trial]:
    """Decide a cohort of K sibling trials proposed in the same round.

    Each trial is first decided independently against the round's incoming best_metric
    (so kept=True for any improver). Then we run a tournament: at most one trial in the
    cohort is promoted (the largest improvement); losers are reverted to kept=False with
    a reason indicating which sibling beat them.

    For K=1 the tournament is a no-op — the single trial's decide() result is returned
    unchanged, byte-identical to the original sequential loop.
    """
    decided = [decide(t, best_metric=best_metric, metric_direction=metric_direction) for t in trials]
    winners = [t for t in decided if t.kept]
    if len(winners) <= 1:
        return decided

    if metric_direction == "maximize":
        champ = max(winners, key=lambda t: t.metric)  # type: ignore[arg-type]
    else:
        champ = min(winners, key=lambda t: t.metric)  # type: ignore[arg-type]
    for t in winners:
        if t.trial_id == champ.trial_id:
            continue
        t.kept = False
        t.reason = f"lost cohort tournament to {champ.trial_id} (its metric={champ.metric})"
        t.usd_per_bp = None
    return decided
