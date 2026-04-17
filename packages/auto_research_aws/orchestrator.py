"""Step Functions ASL builder for the parallel hill-climb topology.

The state machine is parametric on Lambda ARNs (substituted by Terraform) and on the
Spec.parallelism (used as MaxConcurrency on the Map state).

Topology:
    Propose                          # K diverse proposals via one OpenAI call
      → MapTrials (Map, MaxConcurrency=K)
            Train → Evaluate         # one branch per proposal
      → Decide                       # cohort tournament + ledger writes + promote
      → Choice (continue?)
            yes → loop back to Propose
            no  → End
"""
from __future__ import annotations

from typing import Any


def build_state_machine_definition(
    *,
    propose_lambda_arn: str,
    train_lambda_arn: str,
    evaluate_lambda_arn: str,
    decide_lambda_arn: str,
    max_concurrency: int = 10,
    comment: str = "auto-research parallel Karpathy loop",
) -> dict[str, Any]:
    """Return the ASL JSON dict for the state machine."""
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    if max_concurrency > 40:
        # Standard Map default ceiling. Distributed Map allows more but is a different model.
        raise ValueError(
            "max_concurrency > 40 not supported by Standard Map; "
            "use Distributed Map (MVP-3) or lower spec.parallelism"
        )

    return {
        "Comment": comment,
        "StartAt": "Propose",
        "States": {
            "Propose": {
                "Type": "Task",
                "Resource": propose_lambda_arn,
                "Next": "MapTrials",
            },
            "MapTrials": {
                "Type": "Map",
                "ItemsPath": "$.proposals",
                "MaxConcurrency": max_concurrency,
                "ItemSelector": {
                    "run_id.$": "$$.Map.Item.Value.run_id",
                    "s3_bucket.$": "$$.Map.Item.Value.s3_bucket",
                    "ddb_table.$": "$$.Map.Item.Value.ddb_table",
                    "region.$": "$$.Map.Item.Value.region",
                    "openai_secret_id.$": "$$.Map.Item.Value.openai_secret_id",
                    "data_basename.$": "$$.Map.Item.Value.data_basename",
                    "round_id.$": "$$.Map.Item.Value.round_id",
                    "trial_id.$": "$$.Map.Item.Value.trial_id",
                    "idx.$": "$$.Map.Item.Value.idx",
                    "parent_id.$": "$$.Map.Item.Value.parent_id",
                },
                "ItemProcessor": {
                    "ProcessorConfig": {"Mode": "INLINE"},
                    "StartAt": "Train",
                    "States": {
                        "Train": {
                            "Type": "Task",
                            "Resource": train_lambda_arn,
                            "Next": "Evaluate",
                        },
                        "Evaluate": {
                            "Type": "Task",
                            "Resource": evaluate_lambda_arn,
                            "End": True,
                        },
                    },
                },
                "ResultPath": "$.map_results",
                "Next": "Decide",
            },
            "Decide": {
                "Type": "Task",
                "Resource": decide_lambda_arn,
                "Next": "ContinueChoice",
            },
            "ContinueChoice": {
                "Type": "Choice",
                "Choices": [
                    {
                        "Variable": "$.continue",
                        "BooleanEquals": True,
                        "Next": "Propose",
                    }
                ],
                "Default": "Done",
            },
            "Done": {"Type": "Succeed"},
        },
    }
