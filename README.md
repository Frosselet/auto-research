# auto-research

**Enterprise productization of Karpathy's autoresearch pattern for quant analysts** —
a pip-installable Python package that runs an LLM-driven research loop on your laptop today,
and (roadmap) on a managed AWS backend tomorrow, without changing your notebook code.

---

## What is Karpathy's autoresearch?

In **March 2026**, Andrej Karpathy released
[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) — a tiny program that
lets an LLM **do research autonomously**. The loop is deceptively simple:

```text
repeat:
  1. read the current training script
  2. propose a small edit
  3. run a short training
  4. measure the result
  5. keep the edit if it improved the metric, otherwise discard
```

Left running overnight, the LLM ran 700 experiments and found 20 optimizations that materially
sped up nanochat training. Shopify's CEO reported a 19% gain from one overnight run.

It is the thing a quant wishes they had when they iterate on a model at 9pm and wake up with
nothing — same actions a human would take (change a hyperparameter, re-fit, check OOS), done
overnight, in a reproducible log, at LLM speed. Mechanically it is a hill-climb where the
*mutation operator* is an LLM instead of random perturbation, and *evaluation* (not the LLM's
opinion) is ground truth.

## What this repo adds on top

Karpathy's reference implementation is a great starting point but not something a quant team
can adopt as-is — it assumes you own a GPU box, manage your own OpenAI key, hand-write the
training script, and are happy reading raw logs. This repo wraps the same loop into a product:

- **A pip-installable package** (`auto_research`) you `import` from any Jupyter notebook.
- **A typed spec** (`spec.yaml`) for budgets, metric direction, models, paths — validated by
  pydantic before a single dollar is spent.
- **An onboarding interview** (`auto_research.onboard()`) that generates `spec.yaml` +
  `train.py` + `eval.py` scaffolding from a short OpenAI-backed conversation.
- **A persistent ledger** at `.auto-research/ledger.jsonl` — every trial (kept or rejected),
  with diff, metric, tokens, dollars, and **`$/bp gained`** as the FinOps KPI.
- **Backend-agnostic seams** — `Runner` / `Store` / `Secrets` are abstract base classes, so the
  same loop runs locally today and on AWS tomorrow with no change to the loop itself.

## The two milestones

The codebase ships in two stages with a deliberately identical surface area:

| Axis | **MVP-1 — local** *(this milestone)* | **MVP-2 — cloud** *(roadmap)* |
| --- | --- | --- |
| **Where it runs** | Your laptop, in-process from a notebook | Step Functions + Lambda on AWS |
| **Storage** | `.auto-research/ledger.jsonl` on disk | S3 + DynamoDB |
| **Secrets** | `OPENAI_API_KEY` from the shell | AWS Secrets Manager |
| **Deployment** | `uv sync` | Terraform, by the enablement team |
| **Notebook call** | `auto_research.run_local("./spec.yaml")` | `auto_research.submit("./spec.yaml")` |
| **Status** | Working, validated end-to-end via the walkthrough notebook | Not yet implemented |

The critical design rule: **exactly one line of notebook code differs between the two.**
Same package, same `spec.yaml`, same `train.py` / `eval.py`, same ledger shape. The cloud
flavour swaps the `Runner` / `Store` / `Secrets` implementations behind the seams; nothing in
`loop.py` or `states/*.py` imports `boto3`.

For a quant: MVP-2 means your laptop is free to do other things, the loop runs on the
enablement team's AWS account (so FinOps is per-team), and the loop can run on a daily schedule
without your laptop being awake. For a developer: the work left for MVP-2 is wiring AWS
infrastructure, not rewriting the loop.

---

## Install (MVP-1)

Python 3.12 is the baseline (matches the office environment).

```bash
uv sync --extra examples --extra dev
uv run python -m ipykernel install --user --name auto-research --display-name "auto-research (Python 3.12)"
```

The second command registers a Jupyter kernel named `auto-research` pointing at this venv,
so notebooks pick up the correct Python and packages without a per-notebook activate step.

## Quick start (from a Jupyter notebook)

```python
import auto_research

# Interactive onboarding — emits spec.yaml + train.py + eval.py scaffold
auto_research.onboard()

# Run the Karpathy loop locally
auto_research.run_local("./spec.yaml")

# Inspect results
print(auto_research.results("./spec.yaml"))
```

## Reference recipe

`examples/gbdt-ohlcv/` — a gradient-boosted direction classifier
(`sklearn.HistGradientBoostingClassifier`) on ~1200 days of synthetic OHLCV. Metric is
annualized daily Sharpe of the long/short signal on the last 30% of the history (pure OOS).

```bash
uv run python examples/gbdt-ohlcv/gen_data.py   # one-time: materialize data.csv
export OPENAI_API_KEY=sk-...
uv run python -c "import auto_research; auto_research.run_local('examples/gbdt-ohlcv/spec.yaml')"
```

## End-to-end walkthrough

`examples/walkthrough.ipynb` runs the full loop on synthetic OHLCV from a Jupyter notebook,
end to end, against an OpenAI key. Expect ~30 seconds of OpenAI calls + ~15 seconds of local
CPU training. Total cost: about a fifth of a cent at `gpt-4o-mini` rates. The notebook is
written for two audiences side-by-side — quants who know the theory but not the implementation,
and developers who know the implementation but not the theory.

## Public API (MVP-1)

| Symbol | Purpose |
| --- | --- |
| `auto_research.onboard()` | Interactive interview that emits `spec.yaml` + `train.py` + `eval.py` |
| `auto_research.run_local(spec)` | Execute the Karpathy loop locally against an OpenAI key |
| `auto_research.results(spec)` | Read the ledger, return the best trial and full history |
| `auto_research.Spec` | Pydantic model for `spec.yaml` (validated paths, budgets, metric) |
| `auto_research.Trial`, `LoopState` | Types for trial records and loop state |

## The ledger and the FinOps KPI

Every trial appends one JSON line to `.auto-research/ledger.jsonl`: trial id, parent, diff,
metric, kept-or-not, tokens used, dollars spent, and **`$/bp`** — dollars per basis point of
metric improvement. Nothing is ever lost; delete the file to start fresh. When `$/bp`
flatlines across many trials you have hit diminishing returns on *this* `train.py` /
`eval.py` pair — that is when the human steps back in (new feature, new model family, fresh
data review).
