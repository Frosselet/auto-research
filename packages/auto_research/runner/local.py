from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from auto_research.runner.base import Runner, ScriptResult


class LocalRunner(Runner):
    """Runs a Python script as a subprocess in a given workdir with a timeout."""

    def run(
        self,
        script_path: Path,
        args: list[str],
        workdir: Path,
        env: dict[str, str],
        timeout_s: int,
    ) -> ScriptResult:
        merged_env = {**os.environ, **env}
        cmd = [sys.executable, str(script_path), *args]
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                env=merged_env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            return ScriptResult(
                exit_code=124,
                stdout=e.stdout if isinstance(e.stdout, str) else "",
                stderr=f"timeout after {timeout_s}s",
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
        return ScriptResult(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_ms=int((time.monotonic() - t0) * 1000),
        )
