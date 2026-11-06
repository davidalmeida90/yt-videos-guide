"""Noise check: the plain Haiku arm run a second time. How often does the rating change with no forecast at all? Without this, "the rating changed in 42% of cases" means nothing.

    python run_repeat.py
"""
import json

from run_trio import BUDGET, HERE, MODEL, TICKERS, call, restart, wait


def main():
    trio = json.loads((HERE / "trio.json").read_text())
    if "repeat" not in trio:
        restart()
        dates = sorted({i["date"] for i in call(f"/api/backtests/{trio['stock']}")["items"]})
        trio["repeat"] = call("/api/kronos_repeat", {"tickers": TICKERS, "dates": dates, "budget_usd": BUDGET, "model": MODEL})["id"]
        (HERE / "trio.json").write_text(json.dumps(trio))
    if call(f"/api/backtests/{trio['repeat']}")["status"] != "finished":
        call(f"/api/backtests/{trio['repeat']}/start", {})
    wait(trio["repeat"], "haiku repeat")
    print("REPEAT DONE", flush=True)


if __name__ == "__main__":
    main()
