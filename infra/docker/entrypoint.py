"""Handler dispatcher for the shared Step Functions Lambda image.

Terraform configures each Lambda function with one of:
    HANDLER = "propose" | "train" | "evaluate" | "decide"

The Lambda runtime calls entrypoint.dispatch(event, context); we route to the
right handler module. One image + four env-var-keyed Lambdas keeps ECR simple
and guarantees all four states share exactly the same auto_research code.
"""
from __future__ import annotations

import os

from auto_research_aws.handlers import (
    decide_handler,
    evaluate_handler,
    propose_handler,
    train_handler,
)

_HANDLERS = {
    "propose": propose_handler.handle,
    "train": train_handler.handle,
    "evaluate": evaluate_handler.handle,
    "decide": decide_handler.handle,
}


def dispatch(event, context):
    name = os.environ.get("HANDLER")
    if name is None:
        raise RuntimeError(
            "HANDLER env var not set; expected one of " + ", ".join(sorted(_HANDLERS))
        )
    try:
        handler = _HANDLERS[name]
    except KeyError:
        raise RuntimeError(
            f"HANDLER={name!r} is not a known handler; expected one of "
            + ", ".join(sorted(_HANDLERS))
        ) from None
    return handler(event, context)
