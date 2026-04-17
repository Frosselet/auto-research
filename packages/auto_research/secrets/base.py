from __future__ import annotations

from abc import ABC, abstractmethod


class Secrets(ABC):
    """Fetches secret values by key. MVP-1: env vars. MVP-2: AWS Secrets Manager."""

    @abstractmethod
    def get(self, key: str) -> str: ...
