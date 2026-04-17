# auto-research

Enterprise productization of Karpathy's autoresearch pattern for quant analysts.

Two milestones:

- **MVP-1** — local Karpathy loop runnable from any Jupyter notebook against an OpenAI key, no AWS required.
- **MVP-2** — same loop hosted on AWS (Step Functions + Lambda + S3 + DynamoDB), deployed via Terraform by an enablement team.

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

`examples/gbdt-ohlcv/` — gradient-boosted direction classifier on synthetic OHLCV, Sharpe of out-of-sample signal.

```bash
uv run python examples/gbdt-ohlcv/gen_data.py   # one-time: materialize data.csv
export OPENAI_API_KEY=sk-...
uv run python -c "import auto_research; auto_research.run_local('examples/gbdt-ohlcv/spec.yaml')"
```
