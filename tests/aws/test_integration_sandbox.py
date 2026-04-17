"""Integration test hitting a real AWS sandbox account. Skipped by default.

Run with:
    AWS_AUTORESEARCH_INTEGRATION=1 \
    AUTO_RESEARCH_STATE_MACHINE_ARN=arn:... \
    AUTO_RESEARCH_S3_BUCKET=... \
    AUTO_RESEARCH_DDB_TABLE=... \
    AUTO_RESEARCH_OPENAI_SECRET_ID=... \
    uv run pytest tests/aws/test_integration_sandbox.py -m integration

The enablement team runs this after `terraform apply` to verify the stack end-to-end.
Expect ~0.05 USD spend and ~5 minutes wall-clock with parallelism=2, max_iter=4.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _gate():
    if not os.environ.get("AWS_AUTORESEARCH_INTEGRATION"):
        pytest.skip("set AWS_AUTORESEARCH_INTEGRATION=1 to run")


def test_end_to_end_short_run(tmp_path: Path) -> None:
    from auto_research_aws.submit import submit, watch, results

    state_machine_arn = os.environ["AUTO_RESEARCH_STATE_MACHINE_ARN"]
    bucket = os.environ["AUTO_RESEARCH_S3_BUCKET"]
    table = os.environ["AUTO_RESEARCH_DDB_TABLE"]
    secret_id = os.environ["AUTO_RESEARCH_OPENAI_SECRET_ID"]
    region = os.environ.get("AWS_DEFAULT_REGION")

    # Tiny recipe: counts up by one feature, maximize dummy value.
    (tmp_path / "train.py").write_text(
        "import json,os\n"
        "from pathlib import Path\n"
        "art = Path(os.environ['AUTORESEARCH_ARTIFACT_DIR']); art.mkdir(parents=True, exist_ok=True)\n"
        "(art/'value.json').write_text(json.dumps({'v': 0.1}))\n"
    )
    (tmp_path / "eval.py").write_text(
        "import json,os\n"
        "from pathlib import Path\n"
        "art = Path(os.environ['AUTORESEARCH_ARTIFACT_DIR'])\n"
        "v = json.loads((art/'value.json').read_text())['v']\n"
        "(art/'metric.json').write_text(json.dumps({'metric': float(v)}))\n"
    )
    (tmp_path / "data.csv").write_text("dummy\n1\n")
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump({
        "objective": "maximize v (integration test)",
        "data_path": "data.csv",
        "metric": {"name": "v", "direction": "maximize"},
        "daily_budget_usd": 0.05,
        "max_iterations": 4,
        "parallelism": 2,
    }))

    handle = submit(
        spec_path,
        state_machine_arn=state_machine_arn,
        s3_bucket=bucket,
        ddb_table=table,
        openai_secret_id=secret_id,
        region=region,
    )

    for snapshot in watch(handle, poll_s=10):
        print(snapshot)

    r = results(handle)
    assert r["trials"] > 0
    assert r["usd_spent"] > 0
