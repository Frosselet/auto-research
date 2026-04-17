from __future__ import annotations

import os

from auto_research.secrets.base import Secrets


class EnvSecrets(Secrets):
    def get(self, key: str) -> str:
        value = os.environ.get(key)
        if not value:
            raise RuntimeError(f"secret {key} not set in environment")
        return value
