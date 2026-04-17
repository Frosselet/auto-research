"""Async entry points for the AWS flavour: submit / results / watch.

submit() uploads the analyst's spec + scripts + data to S3 under a fresh run_id and
starts a Step Functions execution. It does not block — returns a Handle the analyst
uses with results() and watch().
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from auto_research.spec import Spec
from auto_research_aws.store import S3DynamoStore


@dataclass
class Handle:
    """Identifies a submitted cloud run."""

    run_id: str
    execution_arn: str
    state_machine_arn: str
    s3_bucket: str
    ddb_table: str
    region: str | None
    openai_secret_id: str
    data_basename: str

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "execution_arn": self.execution_arn,
            "state_machine_arn": self.state_machine_arn,
            "s3_bucket": self.s3_bucket,
            "ddb_table": self.ddb_table,
            "region": self.region,
            "openai_secret_id": self.openai_secret_id,
            "data_basename": self.data_basename,
        }


def submit(
    spec_path: str | Path,
    *,
    state_machine_arn: str,
    s3_bucket: str,
    ddb_table: str,
    openai_secret_id: str,
    region: str | None = None,
    sfn_client=None,
    s3_client=None,
    ddb_resource=None,
) -> Handle:
    spec_path = Path(spec_path).resolve()
    spec = Spec.load(spec_path)

    if spec.parallelism > 40:
        raise ValueError(
            "spec.parallelism > 40 is not supported by the Standard Map state machine; "
            "lower it or wait for MVP-3's Distributed Map support"
        )

    run_id = uuid.uuid4().hex[:16]
    store = S3DynamoStore(
        s3_bucket=s3_bucket,
        ddb_table=ddb_table,
        run_id=run_id,
        region_name=region,
        s3_client=s3_client,
        ddb_resource=ddb_resource,
    )

    # Upload inputs/ from the spec's directory.
    store.upload_input(spec_path, "spec.yaml")
    train_path = spec.resolve(spec_path, "train_script")
    eval_path = spec.resolve(spec_path, "eval_script")
    data_path = spec.resolve(spec_path, "data_path")
    store.upload_input(train_path, "train.py")
    store.upload_input(eval_path, "eval.py")
    store.upload_input(data_path, data_path.name)

    if sfn_client is None:
        import boto3

        sfn_client = boto3.client("stepfunctions", region_name=region)

    payload = {
        "run_id": run_id,
        "s3_bucket": s3_bucket,
        "ddb_table": ddb_table,
        "region": region,
        "openai_secret_id": openai_secret_id,
        "data_basename": data_path.name,
        "trials_done": 0,
        "best_metric": None,
        "best_trial_id": None,
        "usd_spent": 0.0,
    }
    resp = sfn_client.start_execution(
        stateMachineArn=state_machine_arn,
        name=f"auto-research-{run_id}",
        input=json.dumps(payload),
    )

    return Handle(
        run_id=run_id,
        execution_arn=resp["executionArn"],
        state_machine_arn=state_machine_arn,
        s3_bucket=s3_bucket,
        ddb_table=ddb_table,
        region=region,
        openai_secret_id=openai_secret_id,
        data_basename=data_path.name,
    )


def results(
    handle: Handle,
    *,
    s3_client=None,
    ddb_resource=None,
) -> dict:
    """Read the cloud ledger for a submitted run. Same shape as the local results()."""
    store = S3DynamoStore(
        s3_bucket=handle.s3_bucket,
        ddb_table=handle.ddb_table,
        run_id=handle.run_id,
        region_name=handle.region,
        s3_client=s3_client,
        ddb_resource=ddb_resource,
    )
    history = store.read_history()
    kept = [t for t in history if t.kept and t.metric is not None]
    best = None
    if kept:
        # Direction unknown without the spec; fetch it.
        try:
            spec_local = store.download_input("spec.yaml")
            spec = Spec.load(spec_local)
            direction = spec.metric.direction
        except Exception:
            direction = "maximize"
        if direction == "maximize":
            best = max(kept, key=lambda t: t.metric)  # type: ignore[arg-type]
        else:
            best = min(kept, key=lambda t: t.metric)  # type: ignore[arg-type]
    return {
        "best": best.model_dump() if best else None,
        "history": [t.model_dump() for t in history],
        "trials": len(history),
        "kept": len(kept),
        "usd_spent": sum(t.usd for t in history),
    }


def watch(
    handle: Handle,
    poll_s: int = 15,
    sfn_client=None,
    s3_client=None,
    ddb_resource=None,
) -> Iterator[dict]:
    """Yield {status, trials_done, best_metric, usd_spent} until the execution terminates."""
    if sfn_client is None:
        import boto3

        sfn_client = boto3.client("stepfunctions", region_name=handle.region)

    last_count = -1
    while True:
        desc = sfn_client.describe_execution(executionArn=handle.execution_arn)
        status = desc["status"]
        snap = results(handle, s3_client=s3_client, ddb_resource=ddb_resource)
        if snap["trials"] != last_count:
            last_count = snap["trials"]
            yield {
                "status": status,
                "trials": snap["trials"],
                "kept": snap["kept"],
                "usd_spent": snap["usd_spent"],
                "best_metric": (snap["best"] or {}).get("metric"),
            }
        if status in ("SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"):
            return
        time.sleep(poll_s)
