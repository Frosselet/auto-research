"""Propose handler — one LLM call returns K diverse proposals for a round.

Event input (Step Functions state input):
    {
      "run_id": str,
      "s3_bucket": str,
      "ddb_table": str,
      "region": str | None,
      "openai_secret_id": str,
      "trials_done": int,           # count of trials completed before this round
      "best_metric": float | None,
      "best_trial_id": str | None,
      "usd_spent": float,
      "data_basename": str,         # filename of the data file in inputs/
    }

Output (passed to the Map state):
    {
      ...all input fields propagated...,
      "round_id": str,
      "cohort_size": int,
      "round_usd": float,
      "trials_done_after": int,
      "proposals": [
        {"trial_id": str, "idx": int, "parent_id": str | None, ...input fields...}
      ]
    }
"""
from __future__ import annotations

import uuid

from auto_research.llm.openai_proposer import OpenAIProposer
from auto_research_aws.handlers._common import load_spec, openai_key, store_from_event


def handle(event: dict, _context=None) -> dict:
    store = store_from_event(event)
    spec = load_spec(store)

    # Recover history & seed best_source from S3 (handles cold-start of new Lambda container).
    history = store.read_history()
    best_source = store.read_best_source()
    if best_source is None:
        # Bootstrap: download the user's initial train.py from inputs/.
        best_source = store.download_input("train.py").read_text()
        store.upload_text_input(best_source, "train.py")  # idempotent — keeps a known seed

    trials_done = int(event.get("trials_done", 0))
    remaining = spec.max_iterations - trials_done
    k = max(1, min(spec.parallelism, remaining))

    proposer = OpenAIProposer(api_key=openai_key(event), model=spec.openai_model)
    proposals = proposer.propose_batch(
        k=k,
        objective=spec.objective,
        current_source=best_source,
        history=history,
        best_metric=event.get("best_metric"),
        metric_direction=spec.metric.direction,
    )

    round_id = uuid.uuid4().hex[:8]
    cohort_size = len(proposals)
    round_usd = sum(p.usd for p in proposals)

    parent_id = event.get("best_trial_id")

    propagate = {k_: event[k_] for k_ in (
        "run_id", "s3_bucket", "ddb_table", "region", "openai_secret_id", "data_basename",
    )}

    items = []
    for idx, proposal in enumerate(proposals):
        trial_id = uuid.uuid4().hex[:12]
        store.upload_proposal(
            round_id=round_id,
            idx=idx,
            payload={
                "trial_id": trial_id,
                "parent_id": parent_id,
                "round_id": round_id,
                "cohort_size": cohort_size,
                "idx": idx,
                "diff": proposal.diff,
                "tokens_in": proposal.tokens_in,
                "tokens_out": proposal.tokens_out,
                "usd": proposal.usd,
                "new_source": proposal.new_source,
            },
        )
        items.append({
            **propagate,
            "round_id": round_id,
            "trial_id": trial_id,
            "idx": idx,
            "parent_id": parent_id,
        })

    return {
        **event,
        "round_id": round_id,
        "cohort_size": cohort_size,
        "round_usd": round_usd,
        "trials_done_after": trials_done + cohort_size,
        "proposals": items,
    }
