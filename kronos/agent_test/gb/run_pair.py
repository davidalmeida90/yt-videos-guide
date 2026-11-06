"""Run the matched pair on the scratch Glassbench: same 12 cells, TradingAgents as released against TradingAgents with a Kronos forecast in the price tool.

    python run_pair.py
Cells are fixed by rule before any run: 4 names across sectors, and the decision dates nearest 50%, 75% and 100% of the way through the test period. 1 arm at a time so the 2 never share a budget.
"""
import csv
import json
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
API = "http://127.0.0.1:8767"
TICKERS, BUDGET = ["NVDA", "AAPL", "JPM", "XOM"], 0.65
STATE = HERE / "pair.json"


def call(path, body=None):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


rows = list(csv.DictReader((HERE.parent / "forecasts.csv").open(encoding="utf-8")))
days = sorted({r["date"] for r in rows if r["period"] == "test"})
dates = [days[int((len(days) - 1) * f)] for f in (0.5, 0.75, 1.0)]
print("cells", TICKERS, dates, flush=True)
if STATE.exists():
    ids = json.loads(STATE.read_text())
else:
    ids = call("/api/kronos_pair", {"tickers": TICKERS, "dates": dates, "budget_usd": BUDGET})
    STATE.write_text(json.dumps(ids))
print(ids, flush=True)
for arm in ("stock", "kronos"):
    bid = ids[arm]
    d = call(f"/api/backtests/{bid}")
    if d.get("status") != "finished":
        call(f"/api/backtests/{bid}/start", {})
    t0 = time.time()
    while True:
        d = call(f"/api/backtests/{bid}")
        print(f"{arm} {d.get('status')} finished {d.get('finished')} of {d.get('total') or d.get('planned')} spent {d.get('spent_usd') or d.get('cost_usd')} {time.time() - t0:.0f}s", flush=True)
        if d.get("status") in ("finished", "paused", "cancelled"):
            break
        time.sleep(60)
print("done", flush=True)
