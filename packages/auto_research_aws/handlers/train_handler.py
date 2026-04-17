"""Train handler — runs the candidate train.py for one Map item.

Event input (one item from propose_handler's proposals list, augmented by Map's ItemSelector):
    {
      "run_id": str, "s3_bucket": str, "ddb_table": str, "region": str | None,
      "openai_secret_id": str, "data_basename": str,
      "round_id": str, "trial_id": str, "idx": int, "parent_id": str | None,
    }

Output: a Trial dict (status="trained" or "failed") plus the propagated context.
"""
from __future__ import annotations

from auto_research.runner.local import LocalRunner
from auto_research.states.train import train
from auto_research.types import Trial
from auto_research_aws.handlers._common import load_spec, store_from_event


def handle(event: dict, _context=None) -> dict:
    store = store_from_event(event)
    spec = load_spec(store)

    proposal = store.download_proposal(event["round_id"], event["idx"])
    data_path = store.download_input(event["data_basename"])

    trial = Trial(
        trial_id=event["trial_id"],
        parent_id=event.get("parent_id"),
        round_id=event["round_id"],
        cohort_size=int(proposal.get("cohort_size", 1)),
        diff=proposal.get("diff", ""),
        tokens_in=int(proposal.get("tokens_in", 0)),
        tokens_out=int(proposal.get("tokens_out", 0)),
        usd=float(proposal.get("usd", 0.0)),
        status="proposed",
    )

    trial = train(
        runner=LocalRunner(),
        store=store,
        candidate_source=proposal["new_source"],
        data_path=data_path,
        trial=trial,
    )

    # Push the candidate artifact to S3 so the (likely-different) evaluate Lambda can read it.
    store.upload_candidate_artifact(trial.trial_id)

    return {**_propagate(event), "trial": trial.model_dump()}


def _propagate(event: dict) -> dict:
    return {
        k: event[k]
        for k in (
            "run_id", "s3_bucket", "ddb_table", "region",
            "openai_secret_id", "data_basename",
            "round_id", "trial_id", "idx", "parent_id",
        )
    }
