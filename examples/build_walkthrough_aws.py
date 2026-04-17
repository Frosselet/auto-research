"""Generate examples/walkthrough_aws.ipynb — the MVP-2 (AWS) counterpart of
examples/walkthrough.ipynb.

Same gbdt-ohlcv recipe, same LLM loop, but submitted to Step Functions via
auto_research.submit() and observed via auto_research.watch() / aws_results().

Assumes the enablement team has already deployed the stack per infra/DEPLOY.md
and exported the four ARNs/names as env vars.

Run once to materialize the notebook:
    uv run python examples/build_walkthrough_aws.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB = nbf.v4.new_notebook()
NB.metadata["kernelspec"] = {
    "name": "auto-research",
    "display_name": "auto-research (Python 3.12)",
    "language": "python",
}
NB.metadata["language_info"] = {"name": "python", "version": "3.12"}


def md(text: str) -> None:
    NB.cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    NB.cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(r"""
# auto-research on AWS — end-to-end walkthrough (MVP-2)

This notebook is the cloud counterpart of [`walkthrough.ipynb`](walkthrough.ipynb). Same
`train.py`, `eval.py`, `spec.yaml`. Same Karpathy loop. Three things are different:

1. **Execution environment is standardized.** Every quant running the loop gets the same
   frozen container image (scipy + sklearn + pandas + auto_research) regardless of their
   laptop. No "works on my Mac, fails on Pierre's PC."
2. **Trials run in parallel.** `spec.parallelism: K` fans out K trial branches per round
   via a Step Functions Map state. A 100-trial study that took ~25 minutes locally drops
   to ~3 minutes with K=8.
3. **Your notebook stays interactive.** `submit()` returns a handle immediately; `watch()`
   streams live progress; `aws_results()` fetches the final ledger from DynamoDB.

Prerequisite: your enablement team has deployed the stack per
[`infra/DEPLOY.md`](../infra/DEPLOY.md) and handed you four values — the state-machine
ARN, the S3 bucket, the DynamoDB table, and the OpenAI secret ID.
""")

md(r"""
## Step 0 — Configure

Fill in the four values from your enablement team, and make sure your AWS credentials are
active (`aws sso login --profile <team>-dev` or equivalent).
""")

code(r"""
import os
import pathlib
import shutil
import yaml

# ── Enablement-team output values ─────────────────────────────────────────────
STATE_MACHINE_ARN = os.environ.get("AUTO_RESEARCH_STATE_MACHINE_ARN") or "arn:aws:states:eu-west-1:...:stateMachine:auto-research-<team>-dev-loop"
S3_BUCKET         = os.environ.get("AUTO_RESEARCH_S3_BUCKET")         or "auto-research-<team>-dev"
DDB_TABLE         = os.environ.get("AUTO_RESEARCH_DDB_TABLE")         or "auto-research-<team>-dev-ledger"
OPENAI_SECRET_ID  = os.environ.get("AUTO_RESEARCH_OPENAI_SECRET_ID")  or "auto-research-<team>-dev/openai"
REGION            = os.environ.get("AWS_DEFAULT_REGION")              or "eu-west-1"

import auto_research
print("auto_research", auto_research.__all__)
""")

md(r"""
## Step 1 — Stage a fresh recipe folder

We copy the shipped `examples/gbdt-ohlcv/` reference recipe into a scratch folder and
tweak the spec for a short cloud demo: small budget, 8 trials, **`parallelism: 4`**.
""")

code(r"""
def _find_recipe() -> pathlib.Path:
    cwd = pathlib.Path.cwd()
    for candidate in (
        cwd / "gbdt-ohlcv",
        cwd / "examples" / "gbdt-ohlcv",
        cwd.parent / "examples" / "gbdt-ohlcv",
    ):
        if (candidate / "train.py").exists():
            return candidate
    raise FileNotFoundError("could not find examples/gbdt-ohlcv relative to cwd")

RECIPE = _find_recipe()
WORK = pathlib.Path.cwd() / "_walkthrough_aws_work"
if WORK.exists(): shutil.rmtree(WORK)
WORK.mkdir()
for name in ("train.py", "eval.py", "spec.yaml", "data.csv"):
    shutil.copy(RECIPE / name, WORK / name)

spec = yaml.safe_load((WORK / "spec.yaml").read_text())
spec["daily_budget_usd"] = 0.10
spec["max_iterations"]   = 8
spec["parallelism"]      = 4    # 4 parallel trial branches per round → 2 rounds total
(WORK / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))

print("Working folder:", WORK)
print("Spec:")
print((WORK / "spec.yaml").read_text())
""")

md(r"""
## Step 2 — Submit the run

`submit()` uploads `spec.yaml` + `train.py` + `eval.py` + `data.csv` to
`s3://$S3_BUCKET/runs/<run_id>/inputs/`, calls `StartExecution` on the state machine, and
returns a `Handle`. It does **not** block — the cloud loop runs asynchronously.
""")

code(r"""
os.chdir(WORK)
handle = auto_research.submit(
    "./spec.yaml",
    state_machine_arn=STATE_MACHINE_ARN,
    s3_bucket=S3_BUCKET,
    ddb_table=DDB_TABLE,
    openai_secret_id=OPENAI_SECRET_ID,
    region=REGION,
)
print("run_id:       ", handle.run_id)
print("execution ARN:", handle.execution_arn)
""")

md(r"""
## Step 3 — Watch it run

`watch(handle)` is a generator that polls Step Functions + DynamoDB and yields one
snapshot per *new* trial committed to the ledger. Each snapshot has `status`, `trials`,
`kept`, `usd_spent`, `best_metric`. The generator returns when the execution terminates
(`SUCCEEDED`, `FAILED`, `TIMED_OUT`, or `ABORTED`).

Expect ~2 minutes wall-clock for 8 trials at `parallelism=4` (two rounds of ~1 minute each).
""")

code(r"""
for snap in auto_research.watch(handle, poll_s=10):
    print(snap)
""")

md(r"""
## Step 4 — Read the ledger

`aws_results(handle)` returns the same shape as MVP-1's `results()` — best trial, full
history, totals — but reads from DynamoDB rather than the local JSONL file.

> 🧮 **For quants** — the ledger has all 8 trials, not just the 2 kept (one per round).
> Look at the losing siblings' diffs to understand what the LLM considered but rejected.
>
> 💻 **For developers** — each Trial now has a `round_id` and `cohort_size=4`; losers
> share the round_id with the round's winner but have `kept=false` and
> `reason="lost cohort tournament to <winner_id>"`.
""")

code(r"""
import json
r = auto_research.aws_results(handle)
print(json.dumps({k: v for k, v in r.items() if k != "history"}, indent=2, default=str))
""")

md(r"""
### Sharpe trajectory

Same chart as the local walkthrough — but here we plot all K×rounds trials. The grey dots
form two vertical clusters (one per round); the red line is the running best (kept trials
only) and only moves up.
""")

code(r"""
import matplotlib.pyplot as plt
import pandas as pd

hist = pd.DataFrame(r["history"])
hist_kept = hist[hist["kept"]]
hist_kept = hist_kept.assign(running_best=hist_kept["metric"].cummax())

fig, ax = plt.subplots(figsize=(8, 4))
hist.reset_index().plot(
    x="index", y="metric", kind="scatter",
    ax=ax, color="gray", label="trial metric",
)
hist_kept.reset_index().plot(
    x="index", y="running_best",
    ax=ax, color="crimson", label="running best (kept only)",
)
ax.set_xlabel("trial #")
ax.set_ylabel("Sharpe (out-of-sample)")
ax.set_title("Karpathy loop on AWS — parallelism=4, two rounds")
ax.grid(True, alpha=0.3)
plt.show()
""")

md(r"""
## What changes between this notebook and `walkthrough.ipynb`?

Three lines, one concept:

```diff
- auto_research.run_local("./spec.yaml")
+ handle = auto_research.submit("./spec.yaml", state_machine_arn=..., ...)
+ for snap in auto_research.watch(handle): print(snap)
- auto_research.results("./spec.yaml")
+ auto_research.aws_results(handle)
```

And in the spec:

```diff
+ parallelism: 4
```

Everything else — `train.py`, `eval.py`, the metric, the ledger shape — is identical.
See [`docs/parallelism.md`](../docs/parallelism.md) for when to pick K=1 vs K=4 vs K=20.
""")


def main() -> None:
    out = Path(__file__).parent / "walkthrough_aws.ipynb"
    nbf.write(NB, out)
    print(f"wrote {out} ({len(NB.cells)} cells)")


if __name__ == "__main__":
    main()
