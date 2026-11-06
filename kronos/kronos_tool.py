"""Kronos as a tool an agent can call: 1 ticker in, 1 JSON forecast out, nothing else on stdout.

    python kronos_tool.py NVDA --horizon 5 --lookback 100 --paths 30 [--interval 1d] [--model small] [--eval]

Fields: upside_prob = share of sampled paths that end above the last close; p10/p50/p90 = percent move at the horizon;
vol_ratio = predicted daily volatility over the volatility of the last 20 bars.
"""
import argparse
import contextlib
import io
import json
import time

import numpy as np
import pandas as pd

import kr


def forecast(ticker, horizon=5, lookback=100, n=30, interval="1d", model="small", eval_mode=False):
    t0 = time.time()
    pred = kr.load(model, eval_mode=eval_mode)
    d = kr.bars(ticker, interval, period="max" if interval == "1d" else "700d").iloc[-(lookback + 25):]
    hist = d.iloc[-lookback:]
    step = "B" if interval == "1d" else "h"
    fut = pd.date_range(hist.index[-1], periods=horizon + 1, freq=step)[1:]
    p = kr.paths(pred, hist, fut, n=n)
    close, last = p[:, :, 3], float(hist.close.iloc[-1])
    moves = (close[:, -1] / last - 1) * 100
    recent = float(np.log(d.close).diff().iloc[-20:].std())
    pred_vol = float(np.mean(np.diff(np.log(np.c_[np.full(n, last), close]), axis=1).std(axis=1)))
    return {"ticker": ticker, "model": f"Kronos-{model}", "as_of": str(hist.index[-1]), "interval": interval, "last_close": round(last, 2),
            "lookback": lookback, "horizon": horizon, "paths": n, "upside_prob": round(float((moves > 0).mean()), 3),
            "move_pct": {"p10": round(float(np.percentile(moves, 10)), 2), "p50": round(float(np.percentile(moves, 50)), 2),
                         "p90": round(float(np.percentile(moves, 90)), 2)},
            "vol_ratio": round(pred_vol / recent, 2), "training_mode": bool(pred.model.training), "device": kr.DEVICE,
            "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("ticker")
    a.add_argument("--horizon", type=int, default=5)
    a.add_argument("--lookback", type=int, default=100)
    a.add_argument("--paths", type=int, default=30)
    a.add_argument("--interval", default="1d")
    a.add_argument("--model", default="small")
    a.add_argument("--eval", action="store_true")
    o = a.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        r = forecast(o.ticker, o.horizon, o.lookback, o.paths, o.interval, o.model, o.eval)
    print(json.dumps(r, indent=2))
