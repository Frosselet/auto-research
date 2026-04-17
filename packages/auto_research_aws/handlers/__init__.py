"""Per-state Lambda entry points for the Step Functions Map+Choice topology.

Each handler exposes `handle(event, context)` — events are the JSON shapes documented
in each module. Handlers share one container image and are dispatched by the entrypoint
script using the HANDLER env var.
"""
from auto_research_aws.handlers import (
    decide_handler,
    evaluate_handler,
    propose_handler,
    train_handler,
)

__all__ = ["decide_handler", "evaluate_handler", "propose_handler", "train_handler"]
