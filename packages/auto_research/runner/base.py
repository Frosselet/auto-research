from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScriptResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int


class Runner(ABC):
    """Executes a Python script in isolation and returns its stdout/stderr/exit code.

    MVP-1: LocalRunner subprocesses the script with a timeout.
    MVP-2: an AWS Runner will invoke a Lambda or submit a Batch job with the same contract.
    """

    @abstractmethod
    def run(
        self,
        script_path: Path,
        args: list[str],
        workdir: Path,
        env: dict[str, str],
        timeout_s: int,
    ) -> ScriptResult: ...
