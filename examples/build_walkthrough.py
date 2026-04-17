"""Generate examples/walkthrough.ipynb — a dual-audience end-to-end tutorial.

Two audiences are addressed throughout:
  - Quants who know the theory (Sharpe, OHLCV, overfitting) but not the implementation.
  - Developers who know the implementation (Python, state machines, LLMs) but not the theory.

Run once to materialize the notebook:
    uv run python examples/build_walkthrough.py
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


# ──────────────────────────────────────────────────────────────────────────────
# 1. Title + dual intro
md(r"""
# auto-research — end-to-end walkthrough

This notebook runs a full **Karpathy-style autoresearch loop** on synthetic OHLCV data,
end to end, on your laptop, against an OpenAI key. No cloud, no AWS.

It is written for **two audiences**:

> 🧮 **For quants** *(green boxes)* — you know Sharpe, OHLCV, overfitting, train/test splits.
>   You don't know how the loop, the ledger, or the LLM fit together. We'll fill that in.
>
> 💻 **For developers** *(blue boxes)* — you know Python packages, subprocesses, state machines,
>   LLM APIs. You don't know what a Sharpe ratio is or why a direction classifier is interesting.
>   We'll fill that in too.

Expect ~30 seconds of OpenAI calls + ~15 seconds of local CPU training. Total cost: about a
fifth of a cent at `gpt-4o-mini` rates.
""")

# ──────────────────────────────────────────────────────────────────────────────
# 2. What is Karpathy's autoresearch?
md(r"""
## What is Karpathy's autoresearch?

In **March 2026**, Andrej Karpathy released
[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) — a tiny program that
lets an LLM **do research autonomously**. The loop is deceptively simple:

```
repeat:
  1. read the current training script
  2. propose a small edit
  3. run a short training
  4. measure the result
  5. keep the edit if it improved the metric, otherwise discard
```

Left running overnight, the LLM ran 700 experiments and found 20 optimizations that materially
sped up nanochat training. Shopify's CEO reported a 19% gain from one overnight run.

> 🧮 **For quants** — this is the thing you wish you had when you iterate on a model at 9pm and
> wake up with nothing. It does exactly the thing you would do — change a hyperparameter, re-fit,
> check OOS — but it does it overnight, in a reproducible log, at LLM speed.
>
> 💻 **For developers** — think of it as a hill-climb search where the *mutation operator* is an
> LLM instead of random perturbation. The LLM has read a lot of code and papers, so its proposals
> are smarter than random; but evaluation is the ground truth, not the LLM's opinion.

This notebook is the **local flavour** of an enterprise productization of that idea —
same loop, wrapped as a pip package so any quant can call it from a Jupyter notebook without
owning a server.
""")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Developer primer on the trading problem
md(r"""
## 💻 Developer primer: what's the trading problem we're solving?

Don't skip this if you don't trade. One paragraph each.

**OHLCV** is a table with one row per trading day and columns `open, high, low, close, volume` —
the price at market open and close, the daily range, and how much was traded. It's the most
common raw input in systematic trading.

**A direction classifier** predicts "will the price go up or down tomorrow?" — a binary
label `{0, 1}`. You train on yesterday's features (recent returns, volatility, moving averages)
and the known next-day outcome.

**Sharpe ratio** is the metric. If you turn your predicted direction into a trade
(`+1` = buy, `-1` = sell) and multiply by the actual return, you get a daily P&L. The Sharpe ratio
is `mean(P&L) / std(P&L) * sqrt(252)`. **Higher is better.** A Sharpe above 1 annualized is
considered decent; above 2 is very good; above 4 is suspicious.

**Why it's hard**: markets are ~50.5% predictable. Any training run can hit a local optimum
that looks great on the training window and terrible out-of-sample. The whole sport is
fighting overfitting. This is exactly the kind of problem where a thousand small, reversible
experiments beats one heroic big-model attempt — i.e. where autoresearch shines.
""")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Quant primer on the implementation
md(r"""
## 🧮 Quant primer: what's actually running inside the loop?

Two files on disk: `train.py` and `eval.py`. They are normal Python scripts, no framework.

- `train.py` — reads `data.csv`, trains a model, saves it as `model.pkl`. The LLM edits this file.
- `eval.py` — loads `model.pkl`, computes your metric (Sharpe), writes it as `metric.json`.
  The LLM **never edits** this — it's your stable scoring rubric.

The loop does this, in-process, without touching your notebook:

1. Ask OpenAI for a small edit to `train.py`, given a history of previous attempts.
2. Save the candidate script, run it as a subprocess, wait for it to finish.
3. Run `eval.py` against the produced model artifact, read the metric back.
4. Compare to the best metric so far. Promote or discard.
5. Repeat until budget (\$) or iteration cap is reached.

Every trial gets one JSON line in `.auto-research/ledger.jsonl`:
trial id, parent, diff, metric, kept-or-not, tokens used, dollars spent, **\$ per basis-point
of improvement** — the FinOps KPI.
""")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Setup
md(r"""
---

## Step 0 — Setup

The cell below assumes the repo is already pip-installed (`uv sync --extra examples`) and that
`OPENAI_API_KEY` is exported in your shell. If either fails, fix it and re-run.
""")

code(r"""
import os, pathlib, shutil, json
assert os.environ.get("OPENAI_API_KEY"), "set OPENAI_API_KEY before running this notebook"
import auto_research
print("auto_research loaded. Public API:", [name for name in dir(auto_research) if not name.startswith("_")][:10])
""")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Stage a working directory
md(r"""
## Step 1 — Stage a working folder

In real life you would call `auto_research.onboard()` — an OpenAI-backed interview that
generates `spec.yaml`, `train.py`, and `eval.py` tailored to your objective. For a
runnable tutorial we'll skip the interactive input and copy the shipped reference recipe
(`examples/gbdt-ohlcv/`) into a scratch folder.

> 💻 **Developer note** — the reference recipe is a gradient-boosted direction classifier
> (`sklearn.HistGradientBoostingClassifier`) on ~1200 days of synthetic OHLCV.
>
> 🧮 **Quant note** — the metric is annualized daily Sharpe of the long/short signal on the
> last 30% of the history (pure OOS). The 70/30 split is inside `train.py` and is visible in
> `features.json` so `eval.py` can align its scoring window.
""")

code(r"""
def _find_recipe() -> pathlib.Path:
    cwd = pathlib.Path.cwd()
    for candidate in (
        cwd / "gbdt-ohlcv",                        # cwd == examples/
        cwd / "examples" / "gbdt-ohlcv",           # cwd == project root
        cwd.parent / "examples" / "gbdt-ohlcv",    # cwd == examples/<sub>/
    ):
        if (candidate / "train.py").exists():
            return candidate
    raise FileNotFoundError("could not find examples/gbdt-ohlcv relative to cwd")

RECIPE = _find_recipe()
WORK = pathlib.Path.cwd() / "_walkthrough_work"
if WORK.exists(): shutil.rmtree(WORK)
WORK.mkdir()
for name in ("train.py", "eval.py", "spec.yaml", "data.csv"):
    shutil.copy(RECIPE / name, WORK / name)

# Trim the spec to a tiny budget for a fast tutorial.
import yaml
spec = yaml.safe_load((WORK / "spec.yaml").read_text())
spec["daily_budget_usd"] = 0.05
spec["max_iterations"]  = 4
(WORK / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))

print("Working folder:", WORK)
for p in sorted(WORK.iterdir()): print(" -", p.name, f"({p.stat().st_size} bytes)")
""")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Inspect the train.py
md(r"""
## Step 2 — Inspect `train.py`

The LLM will edit this file. Read it.

> 🧮 **For quants** — features are a few canonical technical indicators (1/5/20-day returns,
> rolling vol, 10-day momentum, daily range). The target is next-day direction. The train
> window is the first 70% of rows.
>
> 💻 **For developers** — the contract is two env vars: `AUTORESEARCH_DATA_PATH` and
> `AUTORESEARCH_ARTIFACT_DIR`. The script is self-contained and must exit 0.
""")

code(r"""
print((WORK / "train.py").read_text())
""")

# ──────────────────────────────────────────────────────────────────────────────
# 8. Inspect the eval.py
md(r"""
## Step 3 — Inspect `eval.py`

The scoring rubric. The LLM does **not** edit this; otherwise it could game the metric.

> 🧮 **For quants** — the rule is OOS-only (last 30% of rows), `signal = +1 if p > 0.5 else -1`,
> P&L = `signal * next_day_return`, Sharpe = annualized.
>
> 💻 **For developers** — the contract is to write `{"metric": <float>}` into
> `AUTORESEARCH_ARTIFACT_DIR/metric.json`. That's it.
""")

code(r"""
print((WORK / "eval.py").read_text())
""")

# ──────────────────────────────────────────────────────────────────────────────
# 9. The spec
md(r"""
## Step 4 — Inspect `spec.yaml`

The knobs the analyst sets. `daily_budget_usd` is a hard stop — the loop halts as soon as
the cumulative OpenAI + compute spend crosses it.

> 🧮 **For quants** — `metric.direction = maximize` tells the loop "bigger Sharpe is better."
> Set it to `minimize` for losses/errors.
>
> 💻 **For developers** — the spec is a pydantic model (`auto_research.Spec`). Relative paths
> are resolved against the spec file's directory. Absolute paths are rejected by a validator.
""")

code(r"""
print((WORK / "spec.yaml").read_text())
""")

# ──────────────────────────────────────────────────────────────────────────────
# 10. Run the loop
md(r"""
## Step 5 — Run the Karpathy loop locally

One call. It will stream **one JSON log line per trial** (diff, metric, kept/discarded, \$ spent).
Expect 4 trials here (that's the budget). Watch the `metric` and `kept` fields.
""")

code(r"""
os.chdir(WORK)
state = auto_research.run_local("./spec.yaml")
""")

# ──────────────────────────────────────────────────────────────────────────────
# 11. Narrate what just happened
md(r"""
## Step 6 — What just happened?

Each log line is one **trial** — one iteration of the loop. Look at a few fields:

| Field | Meaning |
|---|---|
| `trial_id` / `parent_id` | the trial's id; which "best-so-far" it branched from |
| `diff` | unified diff of the LLM's edit to `train.py` |
| `metric` | Sharpe produced by `eval.py` for this candidate |
| `best_metric_before` / `delta` | the best Sharpe going in and how this one compared |
| `kept` | whether the candidate was promoted to the new best-so-far |
| `tokens_in` / `tokens_out` / `usd` | OpenAI usage and approximate cost |
| `usd_per_bp` | dollars spent **per basis point of improvement** — the FinOps KPI |
| `status` | `decided` on success, `failed` if `train.py` or `eval.py` crashed |

> 🧮 **For quants** — the ledger is the research diary. Every attempt, successful or not,
> is recorded with its diff, so a rejected idea is never lost. When the LLM proposes something
> you tried last week, it'll see it in the history and won't repeat itself (this model of
> "institutional memory" is exactly what the original Karpathy setup lacks).
>
> 💻 **For developers** — when a trial crashes (bad LLM edit, missing import, wrong kwarg),
> the error is captured into `reason` and the loop keeps going. No trial can break the loop.
""")

# ──────────────────────────────────────────────────────────────────────────────
# 12. Results and plot
md(r"""
## Step 7 — Inspect the ledger

`auto_research.results()` reads the JSONL ledger back into a dict. Everything is on disk —
re-open this notebook tomorrow and you'll see the same numbers.
""")

code(r"""
r = auto_research.results("./spec.yaml")
print(json.dumps({k: v for k, v in r.items() if k != "history"}, indent=2, default=str))
""")

md(r"""
### Metric trajectory

The chart below plots the metric for every trial (grey dots) and the running best (red line).
The running best only ever goes up (for `maximize`) — that's the whole point of the loop.
""")

code(r"""
import matplotlib.pyplot as plt
import pandas as pd

hist = pd.DataFrame(r["history"])
hist["running_best"] = hist["metric"].cummax()

fig, ax = plt.subplots(figsize=(8, 4))
hist.reset_index().plot(
    x="index", y="metric", kind="scatter",
    ax=ax, color="gray", label="trial metric",
)
hist.reset_index().plot(
    x="index", y="running_best",
    ax=ax, color="crimson", label="running best",
)
ax.set_xlabel("trial #")
ax.set_ylabel("Sharpe (out-of-sample)")
ax.set_title("Karpathy loop — metric per trial")
ax.grid(True, alpha=0.3)
plt.show()
""")

md(r"""
### The winning diff

This is the edit the LLM made that produced the best Sharpe. Read it as a code review —
would you have shipped this change manually?
""")

code(r"""
best_id = r["best"]["trial_id"]
best = next(t for t in r["history"] if t["trial_id"] == best_id)
print(f"best trial: {best_id}  metric: {best['metric']:.4f}  usd_per_bp: {best['usd_per_bp']}")
print()
print(best["diff"])
""")

# ──────────────────────────────────────────────────────────────────────────────
# 13. FinOps KPI
md(r"""
## Step 8 — The FinOps KPI: `$/bp gained`

For every kept trial we record **dollars spent on the proposer call** divided by
**basis points of metric improvement**. Flipping it: how expensive was the Sharpe bump?

> 🧮 **For quants** — when this number flatlines across many trials, you've hit diminishing
> returns: the LLM can't find more juice in *this* `train.py` / `eval.py` pair. That's when you,
> the human, step in — add a new feature, swap the model family, review the data.
>
> 💻 **For developers** — at team scale you'd aggregate `$/bp` across jobs and flag ones that
> consume budget without improving. In MVP-2 that aggregation is a CloudWatch query; here
> it's a pandas groupby.
""")

code(r"""
kept = hist[hist["kept"] & hist["usd_per_bp"].notna()]
if kept.empty:
    print("no kept trials with measurable $/bp this run (often the case on such a short budget)")
else:
    print(kept[["trial_id", "metric", "usd", "usd_per_bp"]].to_string(index=False))
""")

# ──────────────────────────────────────────────────────────────────────────────
# 14. What changes in the cloud flavour
md(r"""
## Step 9 — What changes when you move this to AWS (MVP-2)

Exactly one line of this notebook changes:

```python
# laptop:
state = auto_research.run_local("./spec.yaml")

# cloud:
state = auto_research.submit("./spec.yaml")          # starts a Step Functions execution
```

Same package. Same `spec.yaml`. Same `train.py` / `eval.py`. Same ledger shape.

Under the hood the `Runner` / `Store` / `Secrets` seams swap to Lambda / S3+DynamoDB /
Secrets Manager. Your notebook code is backend-agnostic by design — that's what the ABCs in
`packages/auto_research/{runner,store,secrets}/base.py` are for.

> 🧮 **For quants** — you won't feel a difference except that: (a) your laptop is free to do
> other things, (b) the loop runs on the enablement team's AWS account so FinOps is per-team,
> and (c) the loop can run on a daily schedule without your laptop being awake.
>
> 💻 **For developers** — the interesting work left is wiring Step Functions + Lambda +
> Terraform, not rewriting the loop. The critical invariant is that nothing in `loop.py` or
> `states/*.py` imports boto3.
""")

# ──────────────────────────────────────────────────────────────────────────────
# 15. Wrap-up
md(r"""
## Wrap-up

You ran a full Karpathy autoresearch loop, on your laptop, against a real LLM, for less than a
cent, using a pip-installable package that can be lifted to AWS by changing one line.

Things to try next:

- Replace the synthetic OHLCV with your own data (drop in a CSV with a `close` column).
- Swap the model family inside `train.py` — try `sklearn.linear_model.LogisticRegression` as a
  baseline, let the loop evolve features instead.
- Change `metric.direction` to `minimize` and target a forecasting MSE.
- Bump `daily_budget_usd` to `\$1.00` and let it run longer — watch the running-best curve flatten.

Every run appends to `.auto-research/ledger.jsonl`. Nothing is ever lost; delete that file
to start fresh.
""")


def main() -> None:
    out = Path(__file__).parent / "walkthrough.ipynb"
    nbf.write(NB, out)
    print(f"wrote {out} ({len(NB.cells)} cells)")


if __name__ == "__main__":
    main()
