"""Kronos desk: candles, the next candle as a distribution, paths drawn as they are generated, replay against what happened, dated forecast ledger.

    python server.py        serves http://127.0.0.1:7072 and never opens a browser
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import kr  # noqa: E402
from engine import Engine  # noqa: E402

app = Flask(__name__, static_folder=str(HERE / "static"))
LOOKBACK = 100
LEDGER = HERE / "ledger.jsonl"
ENGINE = None
CACHE = {}


def engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = Engine("small")
    return ENGINE


def get_bars(ticker, interval):
    key = (ticker, interval)
    now = time.time()
    if key not in CACHE or now - CACHE[key][0] > 300:
        d = kr.bars(ticker, interval, start="2023-01-01") if interval == "1d" else kr.bars(ticker, interval, period="700d")
        CACHE[key] = (now, d)
    return CACHE[key][1]


def stamp(t, interval):
    return t.strftime("%Y-%m-%d") if interval == "1d" else int(t.timestamp())


def candle(t, r, interval):
    return {"time": stamp(t, interval), "open": round(float(r[0]), 4), "high": round(float(r[1]), 4), "low": round(float(r[2]), 4), "close": round(float(r[3]), 4)}


def split(ticker, interval, as_of):
    d = get_bars(ticker, interval)
    if as_of:
        cut = d.index.searchsorted(pd.Timestamp(as_of), side="right")
    else:
        cut = len(d)
    return d, cut


def future_dates(d, cut, interval, h):
    """Real dates when they exist, otherwise the next business days or hours."""
    known = list(d.index[cut : cut + h])
    last = known[-1] if known else d.index[cut - 1]
    while len(known) < h:
        last = last + (pd.tseries.offsets.BDay(1) if interval == "1d" else pd.Timedelta(hours=1))
        known.append(last)
    return pd.DatetimeIndex(known)


@app.get("/")
def index():
    return send_from_directory(HERE / "static", "index.html")


@app.get("/static/<path:p>")
def static_file(p):
    return send_from_directory(HERE / "static", p)


@app.get("/api/bars")
def bars():
    ticker, interval = request.args.get("ticker", "NVDA"), request.args.get("interval", "1d")
    d, cut = split(ticker, interval, request.args.get("as_of"))
    show = int(request.args.get("show", 80))
    hist = [candle(t, r, interval) for t, r in zip(d.index[max(0, cut - show) : cut], d[["open", "high", "low", "close"]].values[max(0, cut - show) : cut])]
    fut = [candle(t, r, interval) for t, r in zip(d.index[cut : cut + 60], d[["open", "high", "low", "close"]].values[cut : cut + 60])]
    return jsonify({"ticker": ticker, "interval": interval, "history": hist, "future": fut, "as_of": str(d.index[cut - 1]), "last_bar": str(d.index[-1])})


@app.get("/api/tokens")
def tokens():
    ticker, interval = request.args.get("ticker", "NVDA"), request.args.get("interval", "1d")
    d, cut = split(ticker, interval, request.args.get("as_of"))
    return jsonify(engine().tokens(d.iloc[cut - LOOKBACK : cut], k=int(request.args.get("k", 10))))


@app.get("/api/next")
def next_candle():
    ticker, interval = request.args.get("ticker", "NVDA"), request.args.get("interval", "1d")
    d, cut = split(ticker, interval, request.args.get("as_of"))
    r = engine().next_candle(d.iloc[cut - LOOKBACK : cut])
    if cut < len(d):
        a = d.iloc[cut]
        r["actual"] = {"open": float(a.open), "high": float(a.high), "low": float(a.low), "close": float(a.close), "move": float(a.close / d.close.iloc[cut - 1] - 1) * 100}
    return jsonify(r)


@app.get("/api/stream")
def stream():
    ticker, interval = request.args.get("ticker", "NVDA"), request.args.get("interval", "1d")
    h, n, seed = int(request.args.get("h", 15)), int(request.args.get("n", 30)), int(request.args.get("seed", 0))
    delay = float(request.args.get("delay", 0))
    d, cut = split(ticker, interval, request.args.get("as_of"))
    hist = d.iloc[cut - LOOKBACK : cut]
    fut = future_dates(d, cut, interval, h)
    last = float(hist.close.iloc[-1])
    recent = float(np.log(hist.close).diff().iloc[-20:].std())
    times = [stamp(t, interval) for t in fut]

    def gen():
        for step, c in engine().stream(hist, fut, n=n, seed=seed):
            if step == "final":
                close = c[:, :, 3]
                moves = (close[:, -1] / last - 1) * 100
                med = np.median(c, axis=0)
                pick = int(np.argsort(moves)[len(moves) // 2])
                vol = np.diff(np.log(np.c_[np.full(n, last), close]), axis=1).std(axis=1).mean() / recent
                msg = {"type": "final", "times": times, "median_path": [candle(t, r, interval) for t, r in zip(fut, c[pick])],
                       "close_paths": close.round(3).tolist(), "up_share": float((moves > 0).mean()), "p10": float(np.percentile(moves, 10)),
                       "p50": float(np.percentile(moves, 50)), "p90": float(np.percentile(moves, 90)), "vol_ratio": float(vol), "last": last,
                       "band_low": np.percentile(close, 10, axis=0).round(3).tolist(), "band_high": np.percentile(close, 90, axis=0).round(3).tolist(), "median": med[:, 3].round(3).tolist()}
            else:
                msg = {"type": "step", "step": step, "time": times[step], "close": c[:, 3].round(3).tolist(), "first_path": candle(fut[step], c[0], interval)}
            yield f"data: {json.dumps(msg)}\n\n"
            if delay:
                time.sleep(delay)

    return Response(gen(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/surprise")
def surprise():
    ticker, interval = request.args.get("ticker", "NVDA"), request.args.get("interval", "1d")
    d, cut = split(ticker, interval, request.args.get("as_of"))
    days = list(d.index[max(LOOKBACK, cut - int(request.args.get("days", 60))) : cut])
    return jsonify(engine().surprise(d, days, LOOKBACK))


def score_entry(e):
    d = get_bars(e["ticker"], e["interval"])
    day = pd.Timestamp(e["as_of"])
    if day not in d.index:
        return {**e, "status": "open"}
    i = d.index.get_loc(day)
    if i + e["h"] >= len(d):
        return {**e, "status": "open", "sessions_left": e["h"] - (len(d) - 1 - i)}
    act = float(d.close.iloc[i + e["h"]] / d.close.iloc[i] - 1) * 100
    return {**e, "status": "closed", "actual": act, "model_right": (e["p50"] > 0) == (act > 0), "always_up_right": act > 0, "abs_err": abs(e["p50"] - act), "abs_err_no_change": abs(act),
            "in_band": e["p10"] <= act <= e["p90"]}


@app.get("/api/ledger")
def ledger():
    if not LEDGER.exists():
        return jsonify({"entries": [], "summary": {}})
    rows = [score_entry(json.loads(s)) for s in LEDGER.read_text(encoding="utf-8").splitlines() if s.strip()]
    closed = [r for r in rows if r["status"] == "closed"]
    summary = {"forecasts": len(rows), "closed": len(closed)}
    if closed:
        summary.update(model_right=float(np.mean([r["model_right"] for r in closed]) * 100), always_up_right=float(np.mean([r["always_up_right"] for r in closed]) * 100),
                       mae_model=float(np.mean([r["abs_err"] for r in closed])), mae_no_change=float(np.mean([r["abs_err_no_change"] for r in closed])),
                       in_band=float(np.mean([r["in_band"] for r in closed]) * 100))
    return jsonify({"entries": rows[::-1][:200], "summary": summary})


if __name__ == "__main__":
    # 127.0.0.1 only, no reloader, no browser.
    app.run(host="127.0.0.1", port=7072, debug=False, use_reloader=False, threaded=True)
