"""S3DynamoStore — round-trip ledger ops + promote_candidate via S3."""
from __future__ import annotations

from pathlib import Path

from auto_research.types import Trial
from auto_research_aws.store import S3DynamoStore


def _store(aws, run_id: str, tmp_root: Path) -> S3DynamoStore:
    return S3DynamoStore(
        s3_bucket=aws["bucket"],
        ddb_table=aws["table"],
        run_id=run_id,
        region_name=aws["region"],
        tmp_root=str(tmp_root),
    )


def test_append_and_read_history_preserves_order(aws, tmp_path: Path) -> None:
    s = _store(aws, "run1", tmp_path)
    t1 = Trial(trial_id="a", metric=0.1, kept=True, status="decided", usd=0.001)
    t2 = Trial(trial_id="b", metric=0.2, kept=True, status="decided", usd=0.002)
    t3 = Trial(trial_id="c", metric=0.05, kept=False, status="decided", usd=0.001)
    s.append_trial(t1)
    s.append_trial(t2)
    s.append_trial(t3)

    history = s.read_history()
    assert [t.trial_id for t in history] == ["a", "b", "c"]
    assert [t.metric for t in history] == [0.1, 0.2, 0.05]


def test_history_is_scoped_to_run_id(aws, tmp_path: Path) -> None:
    s1 = _store(aws, "run1", tmp_path / "1")
    s2 = _store(aws, "run2", tmp_path / "2")
    s1.append_trial(Trial(trial_id="x", metric=0.1, kept=True, status="decided"))
    s2.append_trial(Trial(trial_id="y", metric=0.2, kept=True, status="decided"))
    assert [t.trial_id for t in s1.read_history()] == ["x"]
    assert [t.trial_id for t in s2.read_history()] == ["y"]


def test_promote_candidate_uploads_source_and_artifact_to_s3(aws, tmp_path: Path) -> None:
    s = _store(aws, "promo", tmp_path)
    cand = s.candidate_artifact_dir("trial-promo")
    (cand / "model.pkl").write_bytes(b"weights")
    (cand / "metric.json").write_text('{"metric": 1.5}')

    s.promote_candidate("trial-promo", "# best source\n")

    # S3: best/train.py + best/artifact/{model.pkl, metric.json}
    obj = aws["s3"].get_object(Bucket=aws["bucket"], Key="runs/promo/best/train.py")
    assert obj["Body"].read() == b"# best source\n"
    obj = aws["s3"].get_object(Bucket=aws["bucket"], Key="runs/promo/best/artifact/model.pkl")
    assert obj["Body"].read() == b"weights"

    # read_best_source returns the new source.
    assert s.read_best_source() == "# best source\n"


def test_read_best_source_none_before_first_promote(aws, tmp_path: Path) -> None:
    s = _store(aws, "fresh", tmp_path)
    assert s.read_best_source() is None


def test_proposal_round_trip(aws, tmp_path: Path) -> None:
    s = _store(aws, "p", tmp_path)
    payload = {"trial_id": "x", "new_source": "print('hi')\n", "diff": "+ ...", "usd": 0.001}
    key = s.upload_proposal(round_id="r1", idx=2, payload=payload)
    assert key == "runs/p/rounds/r1/proposals/2.json"
    got = s.download_proposal(round_id="r1", idx=2)
    assert got == payload


def test_input_round_trip(aws, tmp_path: Path) -> None:
    s = _store(aws, "i", tmp_path)
    src = tmp_path / "spec.yaml"
    src.write_text("hello: world\n")
    s.upload_input(src, "spec.yaml")
    fetched = s.download_input("spec.yaml")
    assert fetched.read_text() == "hello: world\n"


def test_candidate_artifact_round_trip_through_s3(aws, tmp_path: Path) -> None:
    """Simulates train_handler upload then evaluate_handler download in a fresh container."""
    upload = _store(aws, "art", tmp_path / "u")
    cand = upload.candidate_artifact_dir("t1")
    (cand / "value.json").write_text('{"v": 0.42}')
    upload.upload_candidate_artifact("t1")

    # Brand-new "Lambda" with empty /tmp.
    download = _store(aws, "art", tmp_path / "d")
    local_dir = download.download_candidate_artifact("t1")
    assert (local_dir / "value.json").read_text() == '{"v": 0.42}'
