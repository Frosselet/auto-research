"""Decide handler — collects the Map output, runs the cohort tournament, persists,
and emits the loop-back signal.

Event input (the Map state's output is at event["map_results"], with shared context):
    {
      "run_id": ..., "s3_bucket": ..., "ddb_table": ..., "region": ...,
      "openai_secret_id": ..., "data_basename": ...,
      "round_id": ..., "round_usd": float, "trials_done_after": int,
      "best_metric": float | None, "best_trial_id": str | None,
      "usd_spent": float,
      "map_results": [ {trial: {...}, ...}, ... ]
    }

Output:
    {
      ...propagated context...,
      "best_metric": new best (or unchanged),
      "best_trial_id": ...,
      "usd_spent": cumulative,
      "trials_done": new count,
      "continue": bool,        # Choice state inspects this
    }
"""
from __future__ import annotations

from auto_research.spec import Spec
from auto_research.states.decide import decide_round
from auto_research.types import Trial
from auto_research_aws.handlers._common import load_spec, store_from_event


def handle(event: dict, _context=None) -> dict:
    store = store_from_event(event)
    spec = load_spec(store)

    incoming_best = event.get("best_metric")
    trials = [Trial.model_validate(item["trial"]) for item in event["map_results"]]

    decided = decide_round(
        trials=trials,
        best_metric=incoming_best,
        metric_direction=spec.metric.direction,
    )

    new_best_metric = incoming_best
    new_best_trial_id = event.get("best_trial_id")
    winner_idx: int | None = None

    for i, trial in enumerate(decided):
        store.append_trial(trial)
        if trial.kept and trial.metric is not None and winner_idx is None:
            winner_idx = i
            new_best_metric = trial.metric
            new_best_trial_id = trial.trial_id

    if winner_idx is not None:
        winner = decided[winner_idx]
        # Pull the winner's source from the original proposal in S3 and promote.
        proposal = store.download_proposal(event["round_id"], event["map_results"][winner_idx]["idx"])
        store.promote_candidate(winner.trial_id, proposal["new_source"])

    usd_spent = float(event.get("usd_spent", 0.0)) + float(event.get("round_usd", 0.0))
    trials_done = int(event.get("trials_done_after", event.get("trials_done", 0)))
    keep_going = (trials_done < spec.max_iterations) and (usd_spent < spec.daily_budget_usd)

    propagate = {
        k: event[k]
        for k in (
            "run_id", "s3_bucket", "ddb_table", "region",
            "openai_secret_id", "data_basename",
        )
    }
    return {
        **propagate,
        "best_metric": new_best_metric,
        "best_trial_id": new_best_trial_id,
        "usd_spent": usd_spent,
        "trials_done": trials_done,
        "continue": bool(keep_going),
    }
