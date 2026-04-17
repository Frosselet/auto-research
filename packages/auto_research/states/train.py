from __future__ import annotations

from pathlib import Path

from auto_research.runner.base import Runner
from auto_research.store.base import Store
from auto_research.types import Trial

TRAIN_TIMEOUT_S = 60 * 14


def train(
    runner: Runner,
    store: Store,
    candidate_source: str,
    data_path: Path,
    trial: Trial,
    timeout_s: int = TRAIN_TIMEOUT_S,
) -> Trial:
    """Write the candidate source to the iter working copy and invoke train.py.

    Contract for the analyst's train.py:
        - reads AUTORESEARCH_DATA_PATH from env
        - writes its artifact into AUTORESEARCH_ARTIFACT_DIR
    """
    script = store.working_train_path()
    script.write_text(candidate_source)
    artifact_dir = store.candidate_artifact_dir(trial.trial_id)

    result = runner.run(
        script_path=script,
        args=[],
        workdir=script.parent,
        env={
            "AUTORESEARCH_DATA_PATH": str(data_path),
            "AUTORESEARCH_ARTIFACT_DIR": str(artifact_dir),
        },
        timeout_s=timeout_s,
    )
    trial.duration_ms += result.duration_ms
    if result.exit_code != 0:
        trial.status = "failed"
        trial.error = f"train.py exit={result.exit_code}\nstderr tail:\n{result.stderr[-1000:]}"
        return trial
    trial.status = "trained"
    return trial
