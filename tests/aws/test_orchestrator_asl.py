"""ASL builder snapshot — locks the Step Functions topology so accidental edits
to orchestrator.py are reviewed.

Regenerate the snapshot intentionally with:
    REGOLD=1 uv run pytest tests/aws/test_orchestrator_asl.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from auto_research_aws.orchestrator import build_state_machine_definition

SNAPSHOT = Path(__file__).parent / "golden" / "asl_default.json"


def _definition():
    return build_state_machine_definition(
        propose_lambda_arn="arn:aws:lambda:eu-west-1:000000000000:function:auto-research-propose",
        train_lambda_arn="arn:aws:lambda:eu-west-1:000000000000:function:auto-research-train",
        evaluate_lambda_arn="arn:aws:lambda:eu-west-1:000000000000:function:auto-research-evaluate",
        decide_lambda_arn="arn:aws:lambda:eu-west-1:000000000000:function:auto-research-decide",
        max_concurrency=10,
    )


def test_asl_snapshot() -> None:
    actual = _definition()
    if os.environ.get("REGOLD"):
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(actual, indent=2) + "\n")
        pytest.skip(f"wrote snapshot to {SNAPSHOT}")
    assert SNAPSHOT.exists(), (
        f"missing {SNAPSHOT}; regenerate with REGOLD=1"
    )
    expected = json.loads(SNAPSHOT.read_text())
    assert actual == expected, (
        "ASL drifted from snapshot. If intentional, regenerate with: "
        "REGOLD=1 uv run pytest tests/aws/test_orchestrator_asl.py"
    )


def test_max_concurrency_validation() -> None:
    with pytest.raises(ValueError):
        build_state_machine_definition(
            propose_lambda_arn="x", train_lambda_arn="x",
            evaluate_lambda_arn="x", decide_lambda_arn="x",
            max_concurrency=0,
        )
    with pytest.raises(ValueError):
        build_state_machine_definition(
            propose_lambda_arn="x", train_lambda_arn="x",
            evaluate_lambda_arn="x", decide_lambda_arn="x",
            max_concurrency=41,
        )


def test_choice_loops_back_to_propose() -> None:
    d = _definition()
    choice = d["States"]["ContinueChoice"]
    assert choice["Type"] == "Choice"
    assert choice["Choices"][0]["Next"] == "Propose"
    assert choice["Default"] == "Done"


def test_map_state_max_concurrency_threads_through() -> None:
    d = build_state_machine_definition(
        propose_lambda_arn="x", train_lambda_arn="x",
        evaluate_lambda_arn="x", decide_lambda_arn="x",
        max_concurrency=7,
    )
    assert d["States"]["MapTrials"]["MaxConcurrency"] == 7
