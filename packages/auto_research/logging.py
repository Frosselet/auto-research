from __future__ import annotations

import json
import logging
import sys
import time

from auto_research.types import Trial

_LOGGER_NAME = "auto_research"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(record.created)),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        extra = getattr(record, "_auto_research_extra", None)
        if extra:
            base.update(extra)
        return json.dumps(base, default=str)


class TrialLogger:
    def __init__(self, logger: logging.Logger):
        self._log = logger

    def info(self, event: str, extra: dict | None = None) -> None:
        self._log.info(event, extra={"_auto_research_extra": extra or {}})

    def log_trial(self, trial: Trial) -> None:
        payload = trial.model_dump()
        payload["event"] = "trial"
        self._log.info(
            "trial",
            extra={"_auto_research_extra": payload},
        )


def trial_logger() -> TrialLogger:
    log = logging.getLogger(_LOGGER_NAME)
    if not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    return TrialLogger(log)
