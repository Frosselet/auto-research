from __future__ import annotations

import json
from pathlib import Path

from auto_research.runner.base import Runner
from auto_research.store.base import Store
from auto_research.types import Trial

EVAL_TIMEOUT_S = 60 * 5


def evaluate(
    runner: Runner,
    store: Store,
    eval_script_path: Path,
    data_path: Path,
    trial: Trial,
    timeout_s: int = EVAL_TIMEOUT_S,
) -> Trial:
    """Invoke eval.py against the candidate artifact; read metric.json back.

    Contract for the analyst's eval.py:
        - reads AUTORESEARCH_DATA_PATH and AUTORESEARCH_ARTIFACT_DIR from env
        - writes {"metric": <float>} to AUTORESEARCH_ARTIFACT_DIR/metric.json
    """
    if trial.status == "failed":
        return trial
    artifact_dir = store.candidate_artifact_dir(trial.trial_id)
    result = runner.run(
        script_path=eval_script_path,
        args=[],
        workdir=eval_script_path.parent,
        env={
            "AUTORESEARCH_DATA_PATH": str(data_path),
            "AUTORESEARCH_ARTIFACT_DIR": str(artifact_dir),
        },
        timeout_s=timeout_s,
    )
    trial.duration_ms += result.duration_ms
    if result.exit_code != 0:
        trial.status = "failed"
        trial.error = f"eval.py exit={result.exit_code}\nstderr tail:\n{result.stderr[-1000:]}"
        return trial

    metric_file = artifact_dir / "metric.json"
    if not metric_file.exists():
        trial.status = "failed"
        trial.error = "eval.py did not write metric.json"
        return trial
    try:
        payload = json.loads(metric_file.read_text())
        trial.metric = float(payload["metric"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        trial.status = "failed"
        trial.error = f"metric.json invalid: {e}"
        return trial
    trial.status = "evaluated"
    return trial
