"""End-to-end test of the propose → train → evaluate → decide handler chain
without Step Functions, using moto for S3/DDB/SecretsManager and a monkeypatched
OpenAIProposer.

The orchestration that Step Functions would do — Map fan-out, ResultPath insertion,
Choice loop-back — is simulated in-process here.
"""
from __future__ import annotations

from pathlib import Path

import boto3
import pytest
import yaml

from auto_research.llm.proposer import Proposal, Proposer
from auto_research_aws.handlers import (
    decide_handler,
    evaluate_handler,
    propose_handler,
    train_handler,
)
from auto_research_aws.store import S3DynamoStore

TRAIN_PY = """
import json, os
from pathlib import Path
artifact = Path(os.environ["AUTORESEARCH_ARTIFACT_DIR"])
artifact.mkdir(parents=True, exist_ok=True)
(artifact / "value.json").write_text(json.dumps({{"v": {v}}}))
"""

EVAL_PY = """
import json, os
from pathlib import Path
artifact = Path(os.environ["AUTORESEARCH_ARTIFACT_DIR"])
v = json.loads((artifact / "value.json").read_text())["v"]
(artifact / "metric.json").write_text(json.dumps({"metric": float(v)}))
"""


class _FakeProposer(Proposer):
    """Returns scripted cohorts so the chain has deterministic outputs."""

    def __init__(self, cohorts):
        self._cohorts = list(cohorts)
        self._cur = 0

    def propose(self, *_args, **_kwargs) -> Proposal:  # pragma: no cover
        raise AssertionError("test calls propose_batch")

    def propose_batch(self, *, k, **_kwargs):
        cohort = self._cohorts[self._cur]
        self._cur += 1
        assert len(cohort) == k
        return [
            Proposal(
                new_source=TRAIN_PY.format(v=v),
                diff=f"# scripted v={v}\n",
                tokens_in=10,
                tokens_out=20,
                usd=0.001,
            )
            for v in cohort
        ]


def _stage_inputs(aws, run_id: str, tmp_path: Path, parallelism: int, max_iter: int) -> str:
    """Upload spec.yaml, train.py, eval.py, data.csv to S3 inputs/. Returns data basename."""
    (tmp_path / "train.py").write_text(TRAIN_PY.format(v=0.0))
    (tmp_path / "eval.py").write_text(EVAL_PY)
    (tmp_path / "data.csv").write_text("dummy\n1\n")
    (tmp_path / "spec.yaml").write_text(
        yaml.safe_dump({
            "objective": "maximize v",
            "data_path": "data.csv",
            "metric": {"name": "v", "direction": "maximize"},
            "daily_budget_usd": 1.0,
            "max_iterations": max_iter,
            "parallelism": parallelism,
        })
    )
    s = S3DynamoStore(
        s3_bucket=aws["bucket"],
        ddb_table=aws["table"],
        run_id=run_id,
        region_name=aws["region"],
        tmp_root=str(tmp_path / "_seed_tmp"),
    )
    s.upload_input(tmp_path / "spec.yaml", "spec.yaml")
    s.upload_input(tmp_path / "train.py", "train.py")
    s.upload_input(tmp_path / "eval.py", "eval.py")
    s.upload_input(tmp_path / "data.csv", "data.csv")
    return "data.csv"


def _seed_event(aws, run_id: str, data_basename: str) -> dict:
    return {
        "run_id": run_id,
        "s3_bucket": aws["bucket"],
        "ddb_table": aws["table"],
        "region": aws["region"],
        "openai_secret_id": aws["secret_id"],
        "data_basename": data_basename,
        "trials_done": 0,
        "best_metric": None,
        "best_trial_id": None,
        "usd_spent": 0.0,
    }


def _patch_openai_proposer(monkeypatch, fake: _FakeProposer) -> None:
    """Make OpenAIProposer(..) return our fake proposer regardless of constructor args."""
    def _factory(*_args, **_kwargs):
        return fake
    monkeypatch.setattr("auto_research_aws.handlers.propose_handler.OpenAIProposer", _factory)


def _simulate_map(event: dict, tmp_root: Path) -> list[dict]:
    """Run train then evaluate sequentially for each item in event['proposals']
    — the Step Functions Map state's job."""
    map_results: list[dict] = []
    for i, item in enumerate(event["proposals"]):
        # In real SFn each Map branch is an independent Lambda with its own /tmp.
        # Simulate that by giving each a separate tmp_root.
        item_with_tmp = dict(item)
        # Trick: train_handler uses store_from_event which spins up a fresh store
        # under tmp_root="/tmp" by default. We can't override that without changing
        # the API; for this in-process test we monkeypatch tmp_root via a per-item shim.
        train_out = _run_handler_with_tmp(train_handler.handle, item_with_tmp, tmp_root / f"train_{i}")
        eval_out = _run_handler_with_tmp(
            evaluate_handler.handle, train_out, tmp_root / f"eval_{i}"
        )
        map_results.append(eval_out)
    return map_results


def _run_handler_with_tmp(handle_fn, event: dict, tmp_root: Path) -> dict:
    """Invoke a handler with S3DynamoStore.tmp_root forced to a per-call dir."""
    import auto_research_aws.handlers._common as _common
    real = _common.store_from_event

    def fake(ev: dict) -> S3DynamoStore:
        return S3DynamoStore(
            s3_bucket=ev["s3_bucket"],
            ddb_table=ev["ddb_table"],
            run_id=ev["run_id"],
            region_name=ev.get("region"),
            tmp_root=str(tmp_root),
        )

    _common.store_from_event = fake
    try:
        return handle_fn(event)
    finally:
        _common.store_from_event = real


def test_one_round_k2_writes_two_trials_and_promotes_winner(aws, tmp_path, monkeypatch) -> None:
    run_id = "r1"
    data = _stage_inputs(aws, run_id, tmp_path, parallelism=2, max_iter=2)
    _patch_openai_proposer(monkeypatch, _FakeProposer([[0.3, 0.7]]))

    event = _seed_event(aws, run_id, data)

    # Simulate one full Step Functions execution of: Propose → Map[Train→Eval] → Decide.
    propose_out = _run_handler_with_tmp(propose_handler.handle, event, tmp_path / "propose")
    assert propose_out["cohort_size"] == 2
    assert len(propose_out["proposals"]) == 2

    map_results = _simulate_map(propose_out, tmp_path / "map")
    assert all(item["trial"]["status"] == "evaluated" for item in map_results)
    metrics = [item["trial"]["metric"] for item in map_results]
    assert metrics == [0.3, 0.7]

    decide_in = {**propose_out, "map_results": map_results}
    decide_out = _run_handler_with_tmp(decide_handler.handle, decide_in, tmp_path / "decide")

    # Cohort tournament: 0.7 wins, 0.3 loses.
    assert decide_out["best_metric"] == 0.7
    assert decide_out["trials_done"] == 2
    assert decide_out["continue"] is False  # max_iter=2 hit

    # Ledger has both trials.
    s = S3DynamoStore(
        s3_bucket=aws["bucket"],
        ddb_table=aws["table"],
        run_id=run_id,
        region_name=aws["region"],
        tmp_root=str(tmp_path / "verify"),
    )
    history = s.read_history()
    assert len(history) == 2
    kept = [t for t in history if t.kept]
    assert len(kept) == 1
    assert kept[0].metric == 0.7

    # The promoted source landed in S3 best/.
    obj = aws["s3"].get_object(Bucket=aws["bucket"], Key=f"runs/{run_id}/best/train.py")
    assert b'"v": 0.7' in obj["Body"].read()


def test_two_rounds_loop_back_then_terminate(aws, tmp_path, monkeypatch) -> None:
    """parallelism=1, max_iter=2 — two sequential rounds; second wins, then terminates."""
    run_id = "r2"
    data = _stage_inputs(aws, run_id, tmp_path, parallelism=1, max_iter=2)
    _patch_openai_proposer(monkeypatch, _FakeProposer([[0.4], [0.9]]))

    state = _seed_event(aws, run_id, data)

    for round_idx in range(2):
        propose_out = _run_handler_with_tmp(
            propose_handler.handle, state, tmp_path / f"propose_{round_idx}"
        )
        map_results = _simulate_map(propose_out, tmp_path / f"map_{round_idx}")
        decide_in = {**propose_out, "map_results": map_results}
        state = _run_handler_with_tmp(
            decide_handler.handle, decide_in, tmp_path / f"decide_{round_idx}"
        )

    assert state["best_metric"] == 0.9
    assert state["trials_done"] == 2
    assert state["continue"] is False
