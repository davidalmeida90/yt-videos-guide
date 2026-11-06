"""Prints the exact text the LLM receives for 1 decision, from saved data: no model is called and nothing is spent.

    python show_prompt.py 2026-10-23 XOM [kronos|placebo|none]

Names and dates never reach the model: the arguments here only pick the saved case. Needs data/, see a0_data.py.
"""
import sys

import a2_llm as m

day, ticker = sys.argv[1], sys.argv[2]
arm = sys.argv[3] if len(sys.argv) > 3 else "kronos"
job = {j["id"]: j for j in m.plan()}[f"{ticker}|{day}|{arm}"]
print("--- system\n" + m.SYSTEM + "\n\n--- user\n" + job["prompt"], flush=True)
print("\nEND OF PROMPT", flush=True)
