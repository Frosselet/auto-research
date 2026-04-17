"""Shared fixtures for AWS-flavour tests.

All tests in this directory mock AWS via moto's `@mock_aws` and never touch a real
account. The integration test (test_integration_sandbox.py) is the lone exception
and is skipped unless AWS_AUTORESEARCH_INTEGRATION=1.
"""
from __future__ import annotations

import os

import boto3
import pytest
from moto import mock_aws

REGION = "eu-west-1"
BUCKET = "auto-research-test"
TABLE = "auto-research-ledger-test"


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def aws():
    """Yield a context with a mocked S3 bucket + DynamoDB table + SecretsManager secret."""
    with mock_aws():
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": REGION},
        )

        ddb = boto3.resource("dynamodb", region_name=REGION)
        ddb.create_table(
            TableName=TABLE,
            KeySchema=[
                {"AttributeName": "run_id", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "run_id", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        ddb.meta.client.get_waiter("table_exists").wait(TableName=TABLE)

        sm = boto3.client("secretsmanager", region_name=REGION)
        sm.create_secret(Name="openai/test", SecretString="sk-test")

        yield {
            "region": REGION,
            "bucket": BUCKET,
            "table": TABLE,
            "secret_id": "openai/test",
            "s3": s3,
            "ddb": ddb,
            "sm": sm,
        }
