"""S3 + DynamoDB backed Store for the AWS flavour of auto-research.

S3 layout (per run):
    runs/<run_id>/inputs/{spec.yaml, train.py, eval.py, <data_basename>}
    runs/<run_id>/best/{train.py, artifact/...}
    runs/<run_id>/candidates/<trial_id>/{train.py, artifact/...}
    runs/<run_id>/rounds/<round_id>/proposals/<idx>.json

DynamoDB (one item per trial, append-only):
    pk: run_id              (string, hash key)
    sk: <ts_us>#<trial_id>  (string, range key — sorts in insertion order)
    payload: JSON Trial     (string)
    trial_id, created_at    (string, for convenience)

Important: `working_train_path()` / `candidate_artifact_dir()` / `best_artifact_dir()`
all return paths under `tmp_root` (default `/tmp` for Lambda). `/tmp` is per-execution-
environment and is NOT shared across sibling Lambda invocations (e.g. across Step Functions
Map branches). Truth lives in S3/DDB; `/tmp` is a per-handler scratch cache. Handlers must
explicitly call `download_*` before reading and `upload_*` after writing.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from auto_research.store.base import Store
from auto_research.types import Trial


class S3DynamoStore(Store):
    def __init__(
        self,
        *,
        s3_bucket: str,
        ddb_table: str,
        run_id: str,
        region_name: str | None = None,
        tmp_root: str | Path = "/tmp",
        s3_client=None,
        ddb_resource=None,
    ) -> None:
        self._bucket = s3_bucket
        self._table_name = ddb_table
        self._run_id = run_id
        self._root = Path(tmp_root) / run_id
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / "iter").mkdir(parents=True, exist_ok=True)
        (self._root / "best" / "artifact").mkdir(parents=True, exist_ok=True)
        (self._root / "candidates").mkdir(parents=True, exist_ok=True)

        if s3_client is None:
            import boto3

            self._s3 = boto3.client("s3", region_name=region_name)
        else:
            self._s3 = s3_client
        if ddb_resource is None:
            import boto3

            self._ddb = boto3.resource("dynamodb", region_name=region_name)
        else:
            self._ddb = ddb_resource
        self._table = self._ddb.Table(self._table_name)

    # ── Store ABC ─────────────────────────────────────────────────────────────

    def working_train_path(self) -> Path:
        return self._root / "iter" / "train.py"

    def candidate_artifact_dir(self, trial_id: str) -> Path:
        d = self._root / "candidates" / trial_id / "artifact"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def best_artifact_dir(self) -> Path:
        d = self._root / "best" / "artifact"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def promote_candidate(self, trial_id: str, new_source: str) -> None:
        """Promote a kept candidate: write source + copy artifact to S3 best/, sync /tmp.

        If the candidate artifact exists in /tmp (e.g. when a single-process caller uses
        this store directly) but not yet in S3, push it up first so the S3 best/ copy
        finds something to mirror. In the typical multi-Lambda flow the artifact is
        already in S3 (uploaded by evaluate_handler) and the local upload is a no-op.
        """
        # /tmp side: same shape as LocalStore so eval.py running in the same Lambda sees the
        # newly promoted artifact at best_artifact_dir().
        (self._root / "best" / "train.py").write_text(new_source)
        cand = self._root / "candidates" / trial_id / "artifact"
        best_local = self.best_artifact_dir()
        if cand.exists():
            if best_local.exists():
                shutil.rmtree(best_local)
            shutil.copytree(cand, best_local)
            self.upload_candidate_artifact(trial_id)

        # S3 side: source as a single object, artifacts mirrored under best/artifact/.
        self._s3.put_object(
            Bucket=self._bucket,
            Key=self._run_key("best/train.py"),
            Body=new_source.encode("utf-8"),
        )
        self._copy_prefix_in_s3(
            src_prefix=self._run_key(f"candidates/{trial_id}/artifact/"),
            dst_prefix=self._run_key("best/artifact/"),
        )

    def read_best_source(self) -> str | None:
        from botocore.exceptions import ClientError

        try:
            obj = self._s3.get_object(Bucket=self._bucket, Key=self._run_key("best/train.py"))
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise
        return obj["Body"].read().decode("utf-8")

    def append_trial(self, trial: Trial) -> None:
        ts = _iso_microsecond_now()
        sk = f"{ts}#{trial.trial_id}"
        item = {
            "run_id": self._run_id,
            "sk": sk,
            "trial_id": trial.trial_id,
            "created_at": ts,
            "payload": trial.model_dump_json(),
        }
        self._table.put_item(Item=item)

    def read_history(self) -> list[Trial]:
        from boto3.dynamodb.conditions import Key

        out: list[Trial] = []
        kwargs: dict = {
            "KeyConditionExpression": Key("run_id").eq(self._run_id),
            "ScanIndexForward": True,  # ascending sort key = insertion order
        }
        while True:
            resp = self._table.query(**kwargs)
            for item in resp.get("Items", []):
                out.append(Trial.model_validate_json(item["payload"]))
            if "LastEvaluatedKey" not in resp:
                break
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        return out

    # ── Cloud-only helpers used by handlers around S3/`/tmp` boundaries ───────

    def download_input(self, basename: str) -> Path:
        """Download runs/<run_id>/inputs/<basename> to /tmp; return the local path."""
        local = self._root / "inputs" / basename
        local.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(self._bucket, self._run_key(f"inputs/{basename}"), str(local))
        return local

    def upload_input(self, local_path: Path, basename: str | None = None) -> str:
        """Upload a file as runs/<run_id>/inputs/<basename>. Returns the S3 key."""
        name = basename or local_path.name
        key = self._run_key(f"inputs/{name}")
        self._s3.upload_file(str(local_path), self._bucket, key)
        return key

    def upload_text_input(self, text: str, basename: str) -> str:
        key = self._run_key(f"inputs/{basename}")
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=text.encode("utf-8"))
        return key

    def download_best_to_tmp(self) -> Path | None:
        """Populate /tmp/.../best/ from S3 (train.py + artifact/). Returns best/train.py path
        if present, else None."""
        from botocore.exceptions import ClientError

        try:
            obj = self._s3.get_object(Bucket=self._bucket, Key=self._run_key("best/train.py"))
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise
        local_train = self._root / "best" / "train.py"
        local_train.write_bytes(obj["Body"].read())
        self._download_prefix_to_tmp(
            s3_prefix=self._run_key("best/artifact/"),
            local_dir=self._root / "best" / "artifact",
        )
        return local_train

    def download_candidate_artifact(self, trial_id: str) -> Path:
        """Pull candidate artifact dir from S3 to /tmp (e.g. for evaluate_handler).
        Returns the local artifact dir."""
        local_dir = self.candidate_artifact_dir(trial_id)
        self._download_prefix_to_tmp(
            s3_prefix=self._run_key(f"candidates/{trial_id}/artifact/"),
            local_dir=local_dir,
        )
        return local_dir

    def upload_candidate_artifact(self, trial_id: str) -> None:
        """Push the local candidate artifact dir to S3."""
        local_dir = self._root / "candidates" / trial_id / "artifact"
        if not local_dir.exists():
            return
        self._upload_dir_to_s3(
            local_dir=local_dir,
            s3_prefix=self._run_key(f"candidates/{trial_id}/artifact/"),
        )

    def upload_proposal(self, round_id: str, idx: int, payload: dict) -> str:
        key = self._run_key(f"rounds/{round_id}/proposals/{idx}.json")
        self._s3.put_object(
            Bucket=self._bucket, Key=key, Body=json.dumps(payload).encode("utf-8")
        )
        return key

    def download_proposal(self, round_id: str, idx: int) -> dict:
        key = self._run_key(f"rounds/{round_id}/proposals/{idx}.json")
        obj = self._s3.get_object(Bucket=self._bucket, Key=key)
        return json.loads(obj["Body"].read())

    # ── internals ─────────────────────────────────────────────────────────────

    def _run_key(self, suffix: str) -> str:
        return f"runs/{self._run_id}/{suffix}"

    def _download_prefix_to_tmp(self, *, s3_prefix: str, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(s3_prefix):]
                if not rel:
                    continue
                local_file = local_dir / rel
                local_file.parent.mkdir(parents=True, exist_ok=True)
                self._s3.download_file(self._bucket, key, str(local_file))

    def _upload_dir_to_s3(self, *, local_dir: Path, s3_prefix: str) -> None:
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir).as_posix()
            self._s3.upload_file(str(path), self._bucket, f"{s3_prefix}{rel}")

    def _copy_prefix_in_s3(self, *, src_prefix: str, dst_prefix: str) -> None:
        # Delete existing dst objects, then copy src → dst.
        existing = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=dst_prefix):
            existing.extend([o["Key"] for o in page.get("Contents", [])])
        for key in existing:
            self._s3.delete_object(Bucket=self._bucket, Key=key)
        for page in paginator.paginate(Bucket=self._bucket, Prefix=src_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(src_prefix):]
                if not rel:
                    continue
                self._s3.copy_object(
                    Bucket=self._bucket,
                    Key=f"{dst_prefix}{rel}",
                    CopySource={"Bucket": self._bucket, "Key": key},
                )


def _iso_microsecond_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
