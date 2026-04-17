from __future__ import annotations

import difflib
import json
from dataclasses import dataclass

from openai import OpenAI

from auto_research.llm.proposer import Proposal, Proposer
from auto_research.types import Trial


@dataclass(frozen=True)
class ModelRates:
    usd_per_1k_input: float
    usd_per_1k_output: float


_RATES: dict[str, ModelRates] = {
    "gpt-4o-mini": ModelRates(0.00015, 0.0006),
    "gpt-4o": ModelRates(0.0025, 0.01),
    "gpt-4.1-mini": ModelRates(0.00040, 0.0016),
    "gpt-4.1": ModelRates(0.002, 0.008),
    "o4-mini": ModelRates(0.00110, 0.00440),
}
_DEFAULT_RATES = ModelRates(0.001, 0.004)


_SYSTEM = """You are an autoresearch proposer for a quantitative-research training script.

Your job: propose ONE small, targeted edit to train.py that is likely to improve the primary
metric. Bias toward changes with known-good signal-to-noise: hyperparameters, feature
engineering tweaks, regularization, train/test split details. Avoid sweeping rewrites.

You MUST return a JSON object with exactly these fields:
  - "summary": one sentence describing the change and the expected reason it helps.
  - "new_source": the FULL new train.py source code, not a diff.

Never invent imports that aren't already present unless the stdlib provides them.
Keep train.py self-contained: it must read data from the path passed as sys.argv[1],
write its artifact into sys.argv[2], and not touch anything else.
"""


def _format_history(history: list[Trial], limit: int = 10) -> str:
    if not history:
        return "(no prior trials)"
    recent = history[-limit:]
    lines = []
    for t in recent:
        tag = "KEPT" if t.kept else "DISCARDED"
        metric = f"{t.metric:.6f}" if t.metric is not None else "n/a"
        lines.append(f"- [{tag}] trial={t.trial_id} metric={metric} — {t.reason}")
    return "\n".join(lines)


class OpenAIProposer(Proposer):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def propose(
        self,
        objective: str,
        current_source: str,
        history: list[Trial],
        best_metric: float | None,
        metric_direction: str,
    ) -> Proposal:
        user = (
            f"Objective: {objective}\n"
            f"Metric direction: {metric_direction}\n"
            f"Current best metric: {best_metric if best_metric is not None else 'none yet'}\n\n"
            f"Recent trials:\n{_format_history(history)}\n\n"
            f"Current train.py:\n```python\n{current_source}\n```\n\n"
            "Return the JSON object as specified."
        )

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        content = resp.choices[0].message.content or "{}"
        payload = json.loads(content)
        new_source = payload.get("new_source", "").strip()
        if not new_source:
            raise RuntimeError("proposer returned empty new_source")

        diff = "".join(
            difflib.unified_diff(
                current_source.splitlines(keepends=True),
                new_source.splitlines(keepends=True),
                fromfile="train.py (before)",
                tofile="train.py (after)",
                n=3,
            )
        )

        usage = resp.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        rates = _RATES.get(self._model, _DEFAULT_RATES)
        usd = tokens_in * rates.usd_per_1k_input / 1000 + tokens_out * rates.usd_per_1k_output / 1000

        return Proposal(
            new_source=new_source,
            diff=diff,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            usd=usd,
        )
