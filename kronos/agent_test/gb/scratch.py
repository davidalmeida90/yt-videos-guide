"""A second Glassbench on 127.0.0.1:8767 with its own copy of the database, plus 1 extra variant: "kronos".

This variant appends a saved Kronos forecast (agent_test/forecasts.csv, Kronos-small, lookback 100) to the price tool's output, so the market analyst sees it and the rest of the team sees what
it writes. Same wording as the LLM test. No GPU here. Glassbench's own data/desk.db is only read once, to copy it. Keys come from DEEPSEEK_API_KEY and ANTHROPIC_API_KEY, never printed.

    python scratch.py        with the Python of the Glassbench venv

GLASSBENCH_REPO names the Glassbench clone, GLASSBENCH_SCRATCH the folder for the copy. Defaults: glassbench and glassbench-kronos-scratch beside this repository.
"""
from __future__ import annotations

import csv
import os
import random
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if not (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
    raise SystemExit("Set DEEPSEEK_API_KEY (run_pair.py) or ANTHROPIC_API_KEY (run_trio.py, run_repeat.py) before starting this server, see kronos/.env.example.")
GLASSBENCH = Path(os.environ.get("GLASSBENCH_REPO") or HERE.parents[3] / "glassbench")
SCRATCH = Path(os.environ.get("GLASSBENCH_SCRATCH") or HERE.parents[3] / "glassbench-kronos-scratch")
if not (GLASSBENCH / "backend" / "deskapp").is_dir():
    raise SystemExit(f"Glassbench clone not found at {GLASSBENCH}. Clone github.com/davidalmeida90/glassbench there or set GLASSBENCH_REPO.")
PORT = 8767

data = SCRATCH / "data"
(data / "runs").mkdir(parents=True, exist_ok=True)
if not (data / "desk.db").exists():
    with sqlite3.connect(f"file:{GLASSBENCH / 'data' / 'desk.db'}?mode=ro", uri=True) as src, sqlite3.connect(data / "desk.db") as dst:
        src.backup(dst)

os.environ.update(GLASSBENCH_STATE_DIR=str(SCRATCH / "state"), GLASSBENCH_ENV=str(SCRATCH / "no-keys.env"), PYTHONUTF8="1")
os.environ.setdefault("DEEPSEEK_API_KEY", "")
sys.path.insert(0, str(GLASSBENCH / "backend"))

from deskapp import settings  # noqa: E402

settings.DATA_DIR, settings.RUNS_DIR, settings.BACKTESTS_DIR, settings.DB_PATH = data, data / "runs", data / "backtests", data / "desk.db"

from deskapp import api, variants, vibe  # noqa: E402
from deskapp.capture import CURRENT_CAPTURE  # noqa: E402
from deskapp.runner import RunManager  # noqa: E402
from pydantic import BaseModel  # noqa: E402

vibe.RUNS_DIR = settings.RUNS_DIR
variants.VARIANTS["kronos"] = {"label": "+ Kronos forecast", "detail": "Appends a Kronos-small forecast for the run date to the price tool's output"}

FORECASTS = {(r["ticker"], r["date"]): r for r in csv.DictReader((HERE.parent / "forecasts.csv").open(encoding="utf-8"))}
TEST_ROWS = [r for r in FORECASTS.values() if r["period"] == "test"]
# Second variant: the same wording with the numbers of another stock and date, fixed by a seed from the cell.
variants.VARIANTS["kronos_fake"] = {"label": "+ fake forecast", "detail": "Same wording as the Kronos variant, with the numbers of another stock and date"}
settings.PRICING["claude-haiku-5-5"] = (0.10, 0.50)
os.environ.setdefault("ANTHROPIC_API_KEY", "")


def fake_for(ticker, day):
    rng = random.Random(f"{ticker}|{day}|fake")
    return rng.choice([r for r in TEST_ROWS if r["ticker"] != ticker and r["date"] != day])


def block(r):
    return ("\n\nForecast from a candlestick foundation model, 30 sampled paths for the next 5 sessions:\n"
            f"Share of paths ending higher: {float(r['up_share']) * 100:.0f}%\n"
            f"Median move: {float(r['p50']):+.1f}%\n"
            f"10th to 90th percentile of moves: {float(r['p10']):+.1f}% to {float(r['p90']):+.1f}%\n"
            f"Predicted volatility against the last 20 sessions: {float(r['vol_ratio']):.2f}x")


def with_forecast(fn):
    def get_stock_data(symbol, start_date=None, end_date=None, *args, **kwargs):
        out = fn(symbol, start_date, end_date, *args, **kwargs)
        cap = CURRENT_CAPTURE.get()
        variant = getattr(cap, "variant", "") if cap is not None else ""
        if variant not in ("kronos", "kronos_fake"):
            return out
        r = FORECASTS.get((str(symbol).upper(), str(end_date)))
        shown = fake_for(str(symbol).upper(), str(end_date)) if (r and variant == "kronos_fake") else r
        cap.bus.emit(cap.run_id, "variant.applied", "market", variant=variant, found=r is not None, up_share=(shown or {}).get("up_share"), detail=f"forecast for {symbol} {end_date}: {'added' if r else 'none on file'}")
        return f"{out}{block(shown)}" if shown else out
    get_stock_data.__name__ = getattr(fn, "__name__", "get_stock_data")
    get_stock_data.__doc__ = fn.__doc__
    return get_stock_data


from tradingagents.dataflows import router  # noqa: E402

for vendor, fn in list(router.VENDOR_METHODS["get_stock_data"].items()):
    router.VENDOR_METHODS["get_stock_data"][vendor] = with_forecast(fn)


_submit = RunManager.submit


def submit_any_provider(self, ticker, trade_date, analysts, depth, deep_model, quick_model, **kw):
    """Glassbench's batch loop never passes a provider, so Claude models are routed here."""
    if str(deep_model).startswith("claude"):
        kw["provider"] = "anthropic"
    return _submit(self, ticker, trade_date, analysts, depth, deep_model, quick_model, **kw)


RunManager.submit = submit_any_provider


class Pair(BaseModel):
    tickers: list[str]
    dates: list[str]
    budget_usd: float


class Trio(BaseModel):
    tickers: list[str]
    dates: list[str]
    budget_usd: float
    model: str


class TopUp(BaseModel):
    id: str
    budget_usd: float


# Noise check: the engine exactly as released, run again. It only has its own label so the batch does not reuse the first plain runs.
variants.VARIANTS["repeat"] = {"label": "Stock, second run", "detail": "TradingAgents as released, run again to measure how much ratings change by chance"}


@api.app.post("/api/kronos_repeat")
def kronos_repeat(req: Trio):
    bt = api.backtests.create(f"Kronos noise check {req.model}", req.tickers, req.dates, ["market", "fundamentals"], 1, req.model, req.model, req.budget_usd,
                              grid={"frequency": "kronos dates", "start": req.dates[0], "end": req.dates[-1], "grid_size": len(req.dates), "sample": None}, variant="repeat")
    return {"id": bt["id"]}


@api.app.post("/api/kronos_topup")
def kronos_topup(req: TopUp):
    api.backtests.store.update_backtest(req.id, budget_usd=req.budget_usd, status="paused")
    return {"ok": True}


@api.app.post("/api/kronos_trio")
def kronos_trio(req: Trio):
    ids = {}
    for arm, variant in (("stock", ""), ("kronos", "kronos"), ("fake", "kronos_fake")):
        bt = api.backtests.create(f"Kronos trio {req.model}: {arm}", req.tickers, req.dates, ["market", "fundamentals"], 1, req.model, req.model, req.budget_usd,
                                  grid={"frequency": "kronos dates", "start": req.dates[0], "end": req.dates[-1], "grid_size": len(req.dates), "sample": None}, variant=variant)
        ids[arm] = bt["id"]
    return ids


@api.app.post("/api/kronos_pair")
def kronos_pair(req: Pair):
    """Glassbench's own route only takes weekly dates; these cells must be dates with a saved forecast."""
    ids = {}
    for arm, variant in (("stock", ""), ("kronos", "kronos")):
        bt = api.backtests.create(f"Kronos pair: {arm}", req.tickers, req.dates, ["market", "fundamentals"], 1, "deepseek-v4-pro", "deepseek-v4-flash", req.budget_usd,
                                  grid={"frequency": "kronos dates", "start": req.dates[0], "end": req.dates[-1], "grid_size": len(req.dates), "sample": None}, variant=variant)
        ids[arm] = bt["id"]
    return ids


if __name__ == "__main__":
    import uvicorn

    assert str(api.DB_PATH).startswith(str(SCRATCH)) and str(vibe.RUNS_DIR).startswith(str(SCRATCH))
    print(f"Glassbench scratch with Kronos at http://127.0.0.1:{PORT}  data {api.DB_PATH}", flush=True)
    uvicorn.run(api.app, host="127.0.0.1", port=PORT, log_level="warning")
