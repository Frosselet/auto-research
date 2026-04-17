from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _ask(prompt: str, default: str | None = None) -> str:
    hint = f" [{default}]" if default else ""
    while True:
        value = input(f"{prompt}{hint}: ").strip()
        if value:
            return value
        if default is not None:
            return default


_TRAIN_BLOCK = re.compile(r"```python\s*#\s*train\.py\s*\n(.*?)```", re.DOTALL)
_EVAL_BLOCK = re.compile(r"```python\s*#\s*eval\.py\s*\n(.*?)```", re.DOTALL)


def _looks_like_real_python(src: str) -> bool:
    """Reject outputs that are one-line JSON-escaped strings."""
    if not src.strip():
        return False
    if "\n" not in src and len(src) > 80:
        return False
    return True


def _tailor_scripts(objective: str, metric: str, direction: str, api_key: str, model: str) -> dict[str, str]:
    """Ask OpenAI to rewrite the train.py/eval.py scaffolds for this objective.

    Uses fenced markdown code blocks (not JSON-mode) to avoid escape-leaking that turns
    multiline Python source into a single literal line with backslash-n sequences.
    """
    from openai import OpenAI

    train_base = (_TEMPLATE_DIR / "train.py.template").read_text()
    eval_base = (_TEMPLATE_DIR / "eval.py.template").read_text()

    system = (
        "You adapt two scaffold scripts for a quant-research autoresearch loop. "
        "Respond with EXACTLY two fenced markdown code blocks, in this order:\n"
        "```python\\n# train.py\\n<full source>\\n```\n"
        "```python\\n# eval.py\\n<full source>\\n```\n"
        "Each block must contain literal Python source with real newlines — no JSON, "
        "no escaping, no surrounding prose. Honour the env-var contracts "
        "(AUTORESEARCH_DATA_PATH, AUTORESEARCH_ARTIFACT_DIR). eval.py MUST write "
        "metric.json with a single 'metric' float into AUTORESEARCH_ARTIFACT_DIR."
    )
    user = (
        f"Objective: {objective}\n"
        f"Primary metric: {metric} ({direction})\n\n"
        f"Starter train.py:\n```python\n{train_base}\n```\n\n"
        f"Starter eval.py:\n```python\n{eval_base}\n```\n\n"
        "Adapt these to a reasonable first attempt for this metric. "
        "Keep it simple — the loop will improve it."
    )

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
        )
        content = resp.choices[0].message.content or ""
        t = _TRAIN_BLOCK.search(content)
        e = _EVAL_BLOCK.search(content)
        train_src = t.group(1).rstrip() + "\n" if t else ""
        eval_src = e.group(1).rstrip() + "\n" if e else ""
        if _looks_like_real_python(train_src) and _looks_like_real_python(eval_src):
            return {"train_py": train_src, "eval_py": eval_src}
        print("[onboard] LLM output did not contain two valid python blocks; using raw templates.")
    except Exception as e:
        print(f"[onboard] LLM tailoring failed ({e}); falling back to raw templates.")
    return {"train_py": train_base, "eval_py": eval_base}


def onboard(
    target_dir: str | Path | None = None,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Interactive onboarding: interview the analyst, emit spec.yaml + train.py + eval.py.

    Safe to run from any Jupyter notebook or a terminal. Uses input() for prompts.
    """
    import os

    target = Path(target_dir) if target_dir else Path.cwd()
    target.mkdir(parents=True, exist_ok=True)

    print("=== auto-research onboarding ===")
    print("I'll ask a few questions, then generate spec.yaml, train.py, eval.py in:")
    print(f"  {target}\n")

    objective = _ask("Research objective in one sentence")
    data_path = _ask("Path to training data file (relative to this folder)", default="data.csv")
    metric_name = _ask("Primary metric name (e.g. sharpe, auc, mse)", default="sharpe")
    direction = _ask("Maximize or minimize this metric?", default="maximize")
    if direction not in ("maximize", "minimize"):
        direction = "maximize"
    budget = float(_ask("Daily budget in USD", default="1.00"))
    max_iter = int(_ask("Max iterations per run", default="20"))

    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if resolved_key:
        print("\nCalling OpenAI to tailor train.py / eval.py to your objective...")
        scripts = _tailor_scripts(objective, metric_name, direction, resolved_key, model)
    else:
        print("\nNo OPENAI_API_KEY — writing raw scaffold templates. Fill them in before running.")
        scripts = {
            "train_py": (_TEMPLATE_DIR / "train.py.template").read_text(),
            "eval_py": (_TEMPLATE_DIR / "eval.py.template").read_text(),
        }

    (target / "train.py").write_text(scripts["train_py"])
    (target / "eval.py").write_text(scripts["eval_py"])

    spec = {
        "objective": objective,
        "data_path": data_path,
        "train_script": "train.py",
        "eval_script": "eval.py",
        "metric": {"name": metric_name, "direction": direction},
        "daily_budget_usd": budget,
        "max_iterations": max_iter,
        "openai_model": model,
        "workdir": ".auto-research",
    }
    spec_path = target / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))

    print(f"\n✓ Wrote {spec_path}")
    print(f"✓ Wrote {target / 'train.py'}")
    print(f"✓ Wrote {target / 'eval.py'}")
    print("\nNext:")
    print("  1. Check the generated train.py and eval.py match your intent.")
    print("  2. Make sure your data file exists at the path above.")
    print("  3. Run:  import auto_research; auto_research.run_local('./spec.yaml')")

    return {"spec_path": str(spec_path), "spec": spec}
