from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class Metric(BaseModel):
    name: str
    direction: Literal["maximize", "minimize"] = "maximize"


class Spec(BaseModel):
    objective: str = Field(description="One-sentence research goal stated by the analyst.")
    data_path: str = Field(description="Path (relative to spec file) to the training data file.")
    train_script: str = Field(default="train.py")
    eval_script: str = Field(default="eval.py")
    metric: Metric
    daily_budget_usd: float = Field(
        gt=0,
        description=(
            "Hard stop for LLM + compute spend per run. Checked at round boundaries; "
            "when parallelism > 1, K proposals are charged together up front and the round "
            "may overshoot mid-flight."
        ),
    )
    max_iterations: int = Field(
        default=50,
        gt=0,
        description="Total trial cap (not round cap). With parallelism=K, runs ~ ceil(max_iterations/K) rounds.",
    )
    parallelism: int = Field(
        default=1,
        ge=1,
        description=(
            "Trials proposed per round. parallelism=1 reproduces the original sequential "
            "Karpathy loop exactly. parallelism>1 batches K proposals per round; in MVP-1 they "
            "run sequentially, in MVP-2 (AWS) they run in parallel via Step Functions Map."
        ),
    )
    openai_model: str = Field(default="gpt-4o-mini")
    workdir: str = Field(
        default=".auto-research",
        description="Where ledger, artifacts, and working copies of train.py live, relative to spec file.",
    )

    @field_validator("train_script", "eval_script", "data_path")
    @classmethod
    def _no_absolute_paths(cls, v: str) -> str:
        if Path(v).is_absolute():
            raise ValueError(f"{v!r} must be relative to the spec file, not absolute")
        return v

    @classmethod
    def load(cls, path: str | Path) -> "Spec":
        path = Path(path)
        with path.open() as f:
            return cls.model_validate(yaml.safe_load(f))

    def resolve(self, spec_path: str | Path, attr: str) -> Path:
        return (Path(spec_path).parent / getattr(self, attr)).resolve()
