"""Run daily_stock_analysis on US stocks, in English, with the API keys taken from your environment (never written to a file).

Put this file next to your clone of github.com/ZhuLinsen/daily_stock_analysis, copy .env.us.example into the clone as .env,
then set your keys in the shell before running, for example in PowerShell:

    $env:ANTHROPIC_API_KEY = "..."          # Claude Haiku 5.5 (default)
    $env:OPENROUTER_API_KEY = "..."         # only for --model deepseek
    $env:TAVILY_API_KEYS = "..."            # optional: news search (free tier about 1,000 searches a month)

    python run.py --stocks NVDA --no-notify --no-market-review --force-run
    python run.py --model deepseek --stocks NVDA,AAPL,KO --no-notify --no-market-review --force-run

Any argument after --model goes to the repo's main.py. --force-run is needed on weekends and holidays (the repo skips closed days).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("DSA_REPO", HERE / "daily_stock_analysis"))

MODELS = {
    "haiku": ("anthropic/claude-haiku-5-5", "ANTHROPIC_API_KEY"),             # about 0.4 cents a stock in the video's runs
    "deepseek": ("openrouter/deepseek/deepseek-v4.1-flash", "OPENROUTER_API_KEY"),   # about 0.15 cents a stock
}

args = sys.argv[1:]
model = "haiku"
if "--model" in args:
    i = args.index("--model")
    model = args[i + 1]
    del args[i:i + 2]
name, key = MODELS[model]
if not os.environ.get(key):
    sys.exit(f"{key} is not set in this shell")

env = dict(os.environ, PYTHONIOENCODING="utf-8", LITELLM_MODEL=name)
if not env.get("TAVILY_API_KEYS"):
    print("TAVILY_API_KEYS not set: running without news search")
py = REPO / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
sys.exit(subprocess.call([str(py if py.exists() else sys.executable), "main.py", *args], cwd=REPO, env=env))
