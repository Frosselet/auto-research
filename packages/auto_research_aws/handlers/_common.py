"""Shared utilities for the AWS Lambda handlers."""
from __future__ import annotations

from auto_research.spec import Spec
from auto_research_aws.secrets import SecretsAWS
from auto_research_aws.store import S3DynamoStore


def store_from_event(event: dict) -> S3DynamoStore:
    return S3DynamoStore(
        s3_bucket=event["s3_bucket"],
        ddb_table=event["ddb_table"],
        run_id=event["run_id"],
        region_name=event.get("region"),
    )


def load_spec(store: S3DynamoStore) -> Spec:
    spec_path = store.download_input("spec.yaml")
    return Spec.load(spec_path)


def openai_key(event: dict) -> str:
    return SecretsAWS(region_name=event.get("region")).get(event["openai_secret_id"])
