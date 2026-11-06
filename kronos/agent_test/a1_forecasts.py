"""Kronos forecasts for every decision date in PLAN.md, on the GPU. Resumable, 1 CSV row per window.

    python a1_forecasts.py

Free, GPU only. Downloads the daily bars into data/ when they are missing.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import kr  # noqa: E402

TICKERS = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMZN", "JPM", "XOM", "UNH", "WMT", "TSLA", "KO"]
START, TEST_START, END = "2023-01-03", "2024-07-01", "2026-10-23"
LOOKBACK, H, STEP, PATHS, WINDOWS_PER_BATCH = 100, 5, 5, 30, 8
DATA, OUT = HERE / "data", HERE / "forecasts.csv"
DATA.mkdir(exist_ok=True)


def decision_days(index):
    """Same calendar for every ticker: every 5th session from START, so all arms and assets line up."""
    days = index[(index >= START) & (index <= END)]
    return days[::STEP]


def main():
    bars = {}
    for t in TICKERS:
        f = DATA / f"{t}.csv"
        if f.exists():
            bars[t] = pd.read_csv(f, parse_dates=["timestamps"], index_col="timestamps")
        else:
            bars[t] = kr.bars(t, "1d", start="2021-06-01")
            bars[t].to_csv(f)
    days = decision_days(bars["SPY"].index)
    print(f"{len(days)} decision days, {days[0].date()} to {days[-1].date()}, {len(days) * len(TICKERS)} windows", flush=True)

    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(zip(prev.ticker, prev.date))
    pred = kr.load("small")
    print("training mode", pred.model.training, "device", kr.DEVICE, flush=True)
    t0, n = time.time(), 0
    for t in TICKERS:
        d = bars[t]
        todo = [day for day in days if (t, str(day.date())) not in done]
        for a in range(0, len(todo), WINDOWS_PER_BATCH):
            part = todo[a : a + WINDOWS_PER_BATCH]
            hists, futs = [], []
            for day in part:
                i = d.index.get_loc(day)
                hists.append(d.iloc[i - LOOKBACK + 1 : i + 1])
                futs.append(pd.Series(pd.bdate_range(day, periods=H + 1)[1:]))
            torch.manual_seed(int(part[0].strftime("%Y%m%d")) + TICKERS.index(t))
            out = pred.predict_batch(
                df_list=[h[kr.COLS] for h in hists for _ in range(PATHS)],
                x_timestamp_list=[pd.Series(h.index) for h in hists for _ in range(PATHS)],
                y_timestamp_list=[f for f in futs for _ in range(PATHS)],
                pred_len=H, T=1.0, top_p=0.9, sample_count=1, verbose=False)
            rows = []
            for k, (day, h) in enumerate(zip(part, hists)):
                paths = np.stack([o["close"].values for o in out[k * PATHS : (k + 1) * PATHS]])
                last = float(h.close.iloc[-1])
                moves = (paths[:, -1] / last - 1) * 100
                recent = float(np.log(h.close).diff().iloc[-20:].std())
                per_path = np.diff(np.log(np.c_[np.full(PATHS, last), paths]), axis=1).std(axis=1)
                rows.append({"ticker": t, "date": str(day.date()), "period": "test" if str(day.date()) >= TEST_START else "control",
                             "last": round(last, 4), "up_share": round(float((moves > 0).mean()), 4),
                             "p10": round(float(np.percentile(moves, 10)), 3), "p50": round(float(np.percentile(moves, 50)), 3),
                             "p90": round(float(np.percentile(moves, 90)), 3), "vol_ratio": round(float(per_path.mean() / recent), 3)})
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            n += len(rows)
        print(f"{t} done, {n} windows in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
