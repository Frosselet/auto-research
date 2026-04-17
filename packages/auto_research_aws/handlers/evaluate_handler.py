"""Evaluate handler — runs eval.py against the candidate artifact.

Event input: train_handler's output (or a passthrough if train failed).
Output: same shape with the trial's status updated to "evaluated" (or "failed") and metric set.
"""
from __future__ import annotations

from auto_research.runner.local import LocalRunner
from auto_research.states.evaluate import evaluate
from auto_research.types import Trial
from auto_research_aws.handlers._common import load_spec, store_from_event


def handle(event: dict, _context=None) -> dict:
    store = store_from_event(event)
    spec = load_spec(store)

    trial = Trial.model_validate(event["trial"])
    if trial.status == "failed":
        return {**event, "trial": trial.model_dump()}

    eval_path = store.download_input(spec.eval_script)
    data_path = store.download_input(event["data_basename"])
    # Candidate artifact was uploaded by train_handler; pull it back into this Lambda's /tmp.
    store.download_candidate_artifact(trial.trial_id)

    trial = evaluate(
        runner=LocalRunner(),
        store=store,
        eval_script_path=eval_path,
        data_path=data_path,
        trial=trial,
    )
    # eval.py may write metric.json; push the post-eval artifact dir back to S3 so the
    # decide handler (different Lambda again) sees it when promoting the winner.
    store.upload_candidate_artifact(trial.trial_id)

    return {**event, "trial": trial.model_dump()}
