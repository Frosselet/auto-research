"""auto_research — enterprise-grade Karpathy autoresearch for quant analysts.

Public API (MVP-1, local):
    onboard()        — interactive interview that emits spec.yaml + train.py + eval.py
    run_local(spec)  — execute the Karpathy loop locally against an OpenAI key
    results(spec)    — read the ledger, return the best trial and full history

Public API (MVP-2, AWS — requires `pip install auto-research[aws]`):
    submit(spec)     — upload inputs to S3, start a Step Functions execution, return Handle
    aws_results(handle) — read the cloud ledger from DynamoDB
    watch(handle)    — generator yielding round summaries until the execution finishes
"""
from __future__ import annotations

from pathlib import Path

from auto_research.llm.openai_proposer import OpenAIProposer
from auto_research.loop import run as _run_loop
from auto_research.onboard import onboard
from auto_research.runner.local import LocalRunner
from auto_research.secrets.env import EnvSecrets
from auto_research.spec import Spec
from auto_research.store.local import LocalStore
from auto_research.types import LoopState, Trial

__all__ = [
    "onboard",
    "run_local",
    "results",
    "submit",
    "aws_results",
    "watch",
    "Spec",
    "Trial",
    "LoopState",
]


def _workdir_for(spec: Spec, spec_path: Path) -> Path:
    wd = Path(spec.workdir)
    if not wd.is_absolute():
        wd = spec_path.parent / wd
    return wd


def run_local(spec_path: str | Path, api_key: str | None = None) -> LoopState:
    """Execute the full autoresearch loop locally until budget or max_iterations is hit."""
    spec_path = Path(spec_path).resolve()
    spec = Spec.load(spec_path)
    key = api_key or EnvSecrets().get("OPENAI_API_KEY")
    proposer = OpenAIProposer(api_key=key, model=spec.openai_model)
    runner = LocalRunner()
    store = LocalStore(_workdir_for(spec, spec_path))
    initial_source = spec.resolve(spec_path, "train_script").read_text()
    return _run_loop(
        spec=spec,
        spec_path=spec_path,
        proposer=proposer,
        runner=runner,
        store=store,
        initial_source=initial_source,
    )


def results(spec_path: str | Path) -> dict:
    """Read the ledger and return best trial + full history."""
    spec_path = Path(spec_path).resolve()
    spec = Spec.load(spec_path)
    store = LocalStore(_workdir_for(spec, spec_path))
    history = store.read_history()
    kept = [t for t in history if t.kept and t.metric is not None]
    best = None
    if kept:
        if spec.metric.direction == "maximize":
            best = max(kept, key=lambda t: t.metric)
        else:
            best = min(kept, key=lambda t: t.metric)
    total_usd = sum(t.usd for t in history)
    return {
        "best": best.model_dump() if best else None,
        "history": [t.model_dump() for t in history],
        "trials": len(history),
        "kept": len(kept),
        "usd_spent": total_usd,
    }


def submit(spec_path: str | Path, **kwargs):
    """Submit a run to AWS (MVP-2). Requires `pip install auto-research[aws]`.

    Uploads spec.yaml + train.py + eval.py + data file to S3, starts a Step Functions
    execution, and returns a Handle. The execution runs asynchronously; use
    `aws_results(handle)` and `watch(handle)` to follow it.
    """
    from auto_research_aws.submit import submit as _submit

    return _submit(spec_path, **kwargs)


def aws_results(handle, **kwargs) -> dict:
    """Read the cloud ledger for a previously submitted run from DynamoDB."""
    from auto_research_aws.submit import results as _aws_results

    return _aws_results(handle, **kwargs)


def watch(handle, **kwargs):
    """Yield round summaries for a running cloud execution until it terminates."""
    from auto_research_aws.submit import watch as _watch

    return _watch(handle, **kwargs)
