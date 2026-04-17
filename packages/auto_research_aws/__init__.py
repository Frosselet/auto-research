"""AWS adapters for auto-research (MVP-2).

Provides:
    SecretsAWS       — AWS Secrets Manager backed Secrets impl
    S3DynamoStore    — S3-and-DynamoDB backed Store impl with /tmp bridging for the
                       Path-returning ABC methods.
    submit / results / watch — async entry points for the cloud loop. submit() uploads
                       inputs to S3 and starts a Step Functions execution.
    build_state_machine_definition — Step Functions ASL builder for the Map+Choice loop.

The handlers/ subpackage holds the per-state Lambda entry points.
"""
from __future__ import annotations

from auto_research_aws.orchestrator import build_state_machine_definition
from auto_research_aws.secrets import SecretsAWS
from auto_research_aws.store import S3DynamoStore
from auto_research_aws.submit import Handle, results, submit, watch

__all__ = [
    "Handle",
    "S3DynamoStore",
    "SecretsAWS",
    "build_state_machine_definition",
    "results",
    "submit",
    "watch",
]
