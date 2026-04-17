from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from auto_research.types import Trial


class Store(ABC):
    """Persists artifacts, working copies of train.py, and the trial ledger.

    MVP-1: LocalStore uses the filesystem + a JSONL ledger.
    MVP-2: an AWS Store will use S3 + DynamoDB behind the same contract.
    """

    @abstractmethod
    def working_train_path(self) -> Path:
        """Filesystem path where the current candidate train.py is staged for this iteration."""

    @abstractmethod
    def candidate_artifact_dir(self, trial_id: str) -> Path:
        """Directory into which the candidate train.py must write its artifact(s)."""

    @abstractmethod
    def best_artifact_dir(self) -> Path:
        """Directory where the best-so-far artifact lives; read by eval.py and end-users."""

    @abstractmethod
    def promote_candidate(self, trial_id: str, new_source: str) -> None:
        """Promote a kept candidate: persist its train.py as the new best-so-far and move its artifact."""

    @abstractmethod
    def read_best_source(self) -> str | None:
        """Return the current best-so-far train.py source, or None if no trial has been kept yet."""

    @abstractmethod
    def append_trial(self, trial: Trial) -> None:
        """Append one trial line to the ledger."""

    @abstractmethod
    def read_history(self) -> list[Trial]:
        """Return all previously recorded trials in chronological order."""
