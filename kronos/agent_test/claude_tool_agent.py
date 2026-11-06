"""1 Claude agent that may call Kronos as a tool, printed live for a terminal take.

    python claude_tool_agent.py NVDA

2 tools: price summary and Kronos forecast (runs kronos_tool.py on the GPU). Claude decides whether to call them. Key read from ANTHROPIC_API_KEY, never printed, and only held by this process.
KRONOS_PYTHON names the Python that has Kronos installed, default is the Python running this script.
Pays from the Anthropic API credit: Sonnet 5.5, a few cents a run. Paper thinking only, places no orders.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import anthropic
import numpy as np
import yfinance as yf

HERE = Path(__file__).parent
KRONOS_PY = os.environ.get("KRONOS_PYTHON") or sys.executable
# Only what the forecast needs, so the API key stays in this process.
TOOL_ENV = {k: os.environ[k] for k in ("SYSTEMROOT", "USERPROFILE", "HOME", "TEMP", "TMP", "HF_HOME", "KRONOS_REPO") if k in os.environ}
TOOL_ENV.update({"PATH": str(Path(KRONOS_PY).parent), "PYTHONUTF8": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "PYTHONWARNINGS": "ignore"})
MODEL = "claude-sonnet-5-5"

TOOLS = [
    {"name": "get_price_summary", "description": "Recent price facts for a US ticker: last close, returns over 5, 20 and 60 sessions, and 20 session daily volatility.",
     "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}},
    {"name": "kronos_forecast", "description": "Runs Kronos, an open-source foundation model trained on candlesticks, on the last 100 daily bars. Samples 30 paths for the next 5 sessions. Returns the share of paths ending higher, the 10th, 50th and 90th percentile moves, and predicted volatility against recent volatility. Takes several seconds.",
     "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}},
]
SYSTEM = ("You decide a paper position in one US stock for the next 5 trading sessions: long, flat or short. Use the tools you find useful, weigh how far each one can be trusted, and "
          "end with one line that starts POSITION: followed by your call, then two sentences of reasoning. Keep every message short.")


def api_key(name):
    """Read an API key from the environment. Never printed."""
    key = os.environ.get(name, "").strip()
    if not key:
        raise SystemExit(f"{name} is not set. Set it in the environment before running this script, see kronos/.env.example.")
    return key


def price_summary(ticker):
    c = yf.Ticker(ticker).history(period="1y", auto_adjust=True)["Close"]
    ret = lambda n: round(float(c.iloc[-1] / c.iloc[-1 - n] - 1) * 100, 1)
    return {"ticker": ticker, "as_of": str(c.index[-1].date()), "last_close": round(float(c.iloc[-1]), 2), "return_5_pct": ret(5), "return_20_pct": ret(20), "return_60_pct": ret(60),
            "daily_vol_20_pct": round(float(np.log(c).diff().iloc[-20:].std()) * 100, 2)}


def kronos(ticker):
    r = subprocess.run([KRONOS_PY, str(HERE.parent / "kronos_tool.py"), ticker, "--horizon", "5", "--lookback", "100", "--paths", "30"], capture_output=True, text=True, timeout=300, cwd=HERE.parent,
                       env=TOOL_ENV)
    if r.returncode:
        return {"error": r.stderr[-300:]}
    out = json.loads(r.stdout)
    for k in ("training_mode", "device", "seconds"):  # plumbing, not forecast. The first take showed Claude using training_mode as a reason to distrust the numbers.
        out.pop(k, None)
    return out


def show(label, text):
    print(f"\n[{label}]")
    for line in str(text).splitlines() or [""]:
        print(textwrap.fill(line, width=118) if line else "")
    sys.stdout.flush()


def main(ticker):
    client = anthropic.Anthropic(api_key=api_key("ANTHROPIC_API_KEY"))
    messages = [{"role": "user", "content": f"What position do you take in {ticker} for the next 5 sessions?"}]
    show("you", messages[0]["content"])
    spent = 0.0
    for _ in range(6):
        r = client.messages.create(model=MODEL, max_tokens=1200, system=SYSTEM, tools=TOOLS, messages=messages)
        spent += (r.usage.input_tokens * 2 + r.usage.output_tokens * 10) / 1e6
        results = []
        for b in r.content:
            if b.type == "text" and b.text.strip():
                show("claude", b.text)
            elif b.type == "tool_use":
                show(f"tool call: {b.name}", json.dumps(b.input))
                out = price_summary(b.input["ticker"]) if b.name == "get_price_summary" else kronos(b.input["ticker"])
                show(f"tool result: {b.name}", json.dumps(out, indent=2))
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out)})
        if not results:
            break
        messages += [{"role": "assistant", "content": r.content}, {"role": "user", "content": results}]
    print(f"\ncost about ${spent:.3f}, {MODEL}\nAGENT DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "NVDA")
