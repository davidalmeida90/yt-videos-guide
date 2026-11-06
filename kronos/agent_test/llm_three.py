"""Terminal take: same decision 3 times, live on DeepSeek. No forecast, the real Kronos forecast, a fake one.

    python llm_three.py 2026-03-13 JPM

Each case is 1 example, picked to make the point visible. Measured result across all decisions is printed at the end, from llm_calls.jsonl, and that is the number that counts.
Costs a fraction of a cent. Key read from DEEPSEEK_API_KEY, never printed. Needs data/, see a0_data.py.
"""
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import a2_llm as m

HERE = Path(__file__).parent


def main(day, ticker):
    key = m.api_key("DEEPSEEK_API_KEY")
    jobs = {j["id"]: j for j in m.plan()}
    print(f"One anonymous stock, one date. Names and dates are hidden from the model.\n", flush=True)
    for arm, label in (("none", "NO FORECAST"), ("kronos", "WITH THE KRONOS FORECAST"), ("placebo", "WITH A FAKE FORECAST (another stock, another date)")):
        j = jobs[f"{ticker}|{day}|{arm}"]
        print("=" * 100 + f"\n{label}")
        if j["shown_up_share"] is not None:
            print("\n".join(line for line in j["prompt"].splitlines()[-5:]))
        r = m.call(key, j["prompt"])
        text = json.loads(r["choices"][0]["message"]["content"])
        print(f"\n  -> {text['position'].upper()}")
        print(textwrap.fill(text.get("reason", ""), width=96, initial_indent="     ", subsequent_indent="     ") + "\n", flush=True)
    calls = pd.DataFrame([json.loads(s) for s in (HERE / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()])
    print("=" * 100 + "\nACROSS ALL 1,404 TEST DECISIONS: how often the call matched the forecast it was shown")
    for arm, label in (("kronos", "Real Kronos"), ("placebo", "Fake forecast")):
        c = calls[(calls.arm == arm) & (calls.period == "test") & calls.shown_up_share.notna() & (calls.shown_up_share != 0.5)]
        fc = np.where(c.shown_up_share > 0.5, "long", "short")
        print(f"  {label:14} {(c.position.values == fc).mean() * 100:.1f}%")
    print("THREE DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
