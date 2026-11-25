"""Replay daily_stock_analysis's own technical engine (src/stock_analyzer.py, StockTrendAnalyzer) on 10 years of US stocks.

Before the AI writes a report, this Python code scores the stock 0 to 100 from the moving averages, the distance from MA5
("bias"), volume, MACD and RSI, and turns the score into a signal (strong buy, buy, wait, sell, strong sell). That score and
its reasons are pasted into the prompt. The engine only reads past rows, so it can be replayed day by day with no look-ahead.

Data: the Qlib project's Alpaca daily bars (split and dividend adjusted) for every S&P 500 member since 2016, plus the
membership spells from fja05680/sp500, so removed and bankrupt members are in the sample (no survivorship bias).
A random sample of members (fixed seed) is scored on every day it was in the index, from 2017 to 17 Nov 2026.

Timing: signal at the close of day t, trade at the open of t+1. Forward returns are open(t+1) to open(t+1+h).
Costs: 5 bps per side on the daily portfolio.

    .venv/Scripts/python.exe engine_backtest.py --n 100 --seed 7      (the video: seeds 7 and 11)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("DSA_REPO", HERE / "daily_stock_analysis"))   # your clone of the repo
DATA = Path(os.environ.get("US_DATA", HERE / "us_data"))   # daily bars per symbol (raw/<SYM>.csv) + sp500_ticker_start_end.csv
os.chdir(REPO)
sys.path.insert(0, str(REPO))
import logging  # noqa: E402

logging.disable(logging.WARNING)
from src.stock_analyzer import StockTrendAnalyzer  # noqa: E402

START, WINDOW, H = "2017-01-03", 250, 10
OUT = HERE / "engine_bt"


def spells():
    s = pd.read_csv(DATA / "sp500_ticker_start_end.csv", parse_dates=["start_date", "end_date"])
    s["end_date"] = s["end_date"].fillna(pd.Timestamp("2099-01-01"))
    return s


def load(sym):
    df = pd.read_csv(DATA / "raw" / f"{sym}.csv", parse_dates=["date"])
    return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)


def score_symbol(sym, sp):
    df = load(sym)
    an = StockTrendAnalyzer()
    member = np.zeros(len(df), bool)
    for _, r in sp[sp.ticker == sym].iterrows():
        member |= ((df.date >= r.start_date) & (df.date < r.end_date)).to_numpy()
    rows = []
    for i in range(WINDOW, len(df) - 1):
        if df.date[i] < pd.Timestamp(START) or not member[i]:
            continue
        res = an.analyze(df.iloc[i - WINDOW + 1:i + 1].copy(), sym)
        rows.append((df.date[i], sym, res.signal_score, res.buy_signal.name, res.trend_status.name,
                     round(res.bias_ma5, 2)))
    out = pd.DataFrame(rows, columns=["date", "sym", "score", "signal", "trend", "bias5"])
    o = df.set_index("date")["open"]
    nxt = o.shift(-1)                                    # open of t+1
    out["r1"] = out.date.map(o.shift(-2) / nxt - 1)      # open(t+1) -> open(t+2)
    out[f"r{H}"] = out.date.map(o.shift(-(H + 1)) / nxt - 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    sp = spells()
    have = {p.stem for p in (DATA / "raw").glob("*.csv")} - {"SPY", "QQQ", "RSP"}
    pool = sorted(t for t in sp.ticker.unique() if t in have and sp[sp.ticker == t].end_date.max() > pd.Timestamp(START))
    random.seed(a.seed)
    pick = random.sample(pool, a.n)
    parts = []
    t0 = time.time()
    for k, sym in enumerate(pick, 1):
        f = OUT / f"{sym}.pkl"
        if f.exists():
            parts.append(pd.read_pickle(f))
            continue
        d = score_symbol(sym, sp)
        d.to_pickle(f)
        parts.append(d)
        print(f"{k}/{len(pick)} {sym}: {len(d)} days, {time.time() - t0:.0f}s", flush=True)
    allr = pd.concat(parts, ignore_index=True)
    allr.to_pickle(OUT / f"_all_n{a.n}_s{a.seed}.pkl")

    # 1) does the signal predict the next 10 days?
    base = allr[f"r{H}"].mean()
    tab = allr.groupby("signal")[f"r{H}"].agg(days="count", mean="mean", hit=lambda r: (r > 0).mean())
    tab["vs_all_days"] = tab["mean"] - base
    print(f"\n10 day forward return by signal (all {len(allr)} stock days, average {base:.4f})")
    print(tab.reindex(["STRONG_BUY", "BUY", "WAIT", "SELL", "STRONG_SELL"]).round(4))

    # 2) daily portfolio: hold every sampled member flagged BUY or STRONG_BUY, equal weight; else cash
    allr["long"] = allr.signal.isin(["BUY", "STRONG_BUY"])
    g = allr.dropna(subset=["r1"]).groupby("date")
    port = g.apply(lambda d: d.loc[d.long, "r1"].mean() if d.long.any() else 0.0)
    held = g.apply(lambda d: d.long.mean())
    turn = allr.pivot_table(index="date", columns="sym", values="long", aggfunc="first").fillna(False).astype(float)
    turn = turn.div(turn.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).diff().abs().sum(axis=1)
    port_net = port - turn.reindex(port.index).fillna(0) * 0.0005
    ew = g["r1"].mean()                                   # equal weight, every sampled member, always invested
    spy = load("SPY").set_index("date")["open"]
    spy_r = (spy.shift(-2) / spy.shift(-1) - 1).reindex(port.index)

    def stats(r):
        r = r.dropna()
        yrs = len(r) / 252
        eq = (1 + r).cumprod()
        return {"cagr": eq.iloc[-1] ** (1 / yrs) - 1, "vol": r.std() * 252 ** 0.5,
                "sharpe": r.mean() / r.std() * 252 ** 0.5, "maxdd": (eq / eq.cummax() - 1).min()}

    res = {"engine_long_net": stats(port_net), "equal_weight_hold": stats(ew), "SPY": stats(spy_r),
           "avg_share_invested": float(held.mean()), "days": int(len(port)), "stocks": a.n, "seed": a.seed}
    print("\n" + json.dumps(res, indent=1, default=float))
    (OUT / f"_summary_n{a.n}_s{a.seed}.json").write_text(json.dumps({"signal_table": tab.round(5).to_dict(), **res}, indent=1, default=float))
    pd.DataFrame({"engine_long_net": (1 + port_net).cumprod(), "equal_weight_hold": (1 + ew).cumprod(),
                  "SPY": (1 + spy_r.fillna(0)).cumprod()}).to_csv(OUT / f"_curves_n{a.n}_s{a.seed}.csv")


if __name__ == "__main__":
    main()
