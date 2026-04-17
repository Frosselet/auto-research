"""submit() uploads inputs to S3 and starts a Step Functions execution (mocked).

We don't exercise the actual state machine (that's in test_handlers.py) — here we
verify submit()'s contract: inputs land in s3://.../inputs/, start_execution is
called with a well-formed JSON payload, and the returned Handle points at the run.
"""
from __future__ import annotations

import json
from pathlib import Path

import boto3
import pytest
import yaml

from auto_research_aws.submit import Handle, submit, results


def _write_recipe(tmp_path: Path, parallelism: int = 1) -> Path:
    (tmp_path / "train.py").write_text("print('train')\n")
    (tmp_path / "eval.py").write_text("print('eval')\n")
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump({
        "objective": "maximize v",
        "data_path": "data.csv",
        "metric": {"name": "v", "direction": "maximize"},
        "daily_budget_usd": 1.0,
        "max_iterations": 2,
        "parallelism": parallelism,
    }))
    return spec_path


def _fake_sfn_client(started):
    class _FakeSfn:
        def start_execution(self, **kwargs):
            started["kwargs"] = kwargs
            return {"executionArn": "arn:aws:states:eu-west-1:000:execution:sm:run1"}

    return _FakeSfn()


def test_submit_uploads_inputs_and_starts_execution(aws, tmp_path: Path) -> None:
    spec_path = _write_recipe(tmp_path, parallelism=3)
    started: dict = {}

    handle = submit(
        spec_path,
        state_machine_arn="arn:aws:states:eu-west-1:000:stateMachine:sm",
        s3_bucket=aws["bucket"],
        ddb_table=aws["table"],
        openai_secret_id=aws["secret_id"],
        region=aws["region"],
        sfn_client=_fake_sfn_client(started),
    )

    assert isinstance(handle, Handle)
    assert handle.execution_arn.endswith(":run1")

    # Inputs made it to S3.
    def _get(key):
        return aws["s3"].get_object(Bucket=aws["bucket"], Key=key)["Body"].read()

    assert _get(f"runs/{handle.run_id}/inputs/spec.yaml")
    assert _get(f"runs/{handle.run_id}/inputs/train.py") == b"print('train')\n"
    assert _get(f"runs/{handle.run_id}/inputs/eval.py") == b"print('eval')\n"
    assert _get(f"runs/{handle.run_id}/inputs/data.csv") == b"a,b\n1,2\n"

    # start_execution got a well-formed JSON payload.
    payload = json.loads(started["kwargs"]["input"])
    assert payload["run_id"] == handle.run_id
    assert payload["trials_done"] == 0
    assert payload["best_metric"] is None
    assert payload["data_basename"] == "data.csv"
    assert payload["openai_secret_id"] == aws["secret_id"]


def test_submit_rejects_parallelism_over_40(tmp_path: Path, aws) -> None:
    spec_path = _write_recipe(tmp_path, parallelism=41)
    started: dict = {}
    with pytest.raises(ValueError, match="parallelism"):
        submit(
            spec_path,
            state_machine_arn="arn:aws:states:eu-west-1:000:stateMachine:sm",
            s3_bucket=aws["bucket"],
            ddb_table=aws["table"],
            openai_secret_id=aws["secret_id"],
            region=aws["region"],
            sfn_client=_fake_sfn_client(started),
        )
    assert "kwargs" not in started  # no execution started


def test_results_reads_empty_ledger_before_any_trials(aws, tmp_path: Path) -> None:
    spec_path = _write_recipe(tmp_path)
    started: dict = {}
    handle = submit(
        spec_path,
        state_machine_arn="arn:aws:states:eu-west-1:000:stateMachine:sm",
        s3_bucket=aws["bucket"],
        ddb_table=aws["table"],
        openai_secret_id=aws["secret_id"],
        region=aws["region"],
        sfn_client=_fake_sfn_client(started),
    )
    r = results(handle)
    assert r == {"best": None, "history": [], "trials": 0, "kept": 0, "usd_spent": 0}
