"""Test 3: reproduce the claims before anyone says them on camera.

    python t3_repro.py

A) Issue 415: with sampling switched off (top_k=1) the forecast must repeat exactly. If it changes between 2 runs,
   dropout is on at inference.
B) Issue 422 and an earlier pilot: on rising markets, how often does it call "down", by lookback, as shipped and in
   eval mode? Windows every 5 sessions from 1 Jul 2024 (after the training data ends) to Oct 2026, 5 sessions ahead.
Results in out/t3_results.csv.
"""
import numpy as np
import pandas as pd
import torch

import kr

TICKERS = ["SPY", "QQQ", "NVDA", "AAPL", "JPM", "XOM"]
START, H, STEP, SAMPLES, CHUNK = "2024-07-01", 5, 5, 5, 40


def determinism(pred):
    d = kr.bars("SPY", "1d", start="2023-01-01")
    hist, fut = d.iloc[-120:-5], d.index[-5:]
    x_ts, y_ts = pd.Series(hist.index), pd.Series(fut)
    runs = []
    for s in (1, 2):
        torch.manual_seed(s)
        runs.append(pred.predict(df=hist[kr.COLS], x_timestamp=x_ts, y_timestamp=y_ts, pred_len=5, T=1.0, top_k=1, top_p=1.0,
                                 sample_count=1, verbose=False)["close"].values)
    return float(np.abs(runs[0] - runs[1]).max())


def windows(d, lookback):
    idx = [i for i in range(lookback, len(d) - H + 1) if d.index[i] >= pd.Timestamp(START)][::STEP]
    return idx


def run(pred, d, lookback):
    idx = windows(d, lookback)
    rows = []
    for a in range(0, len(idx), CHUNK):
        part = idx[a : a + CHUNK]
        hs = [d.iloc[i - lookback : i] for i in part]
        out = pred.predict_batch(df_list=[h[kr.COLS] for h in hs], x_timestamp_list=[pd.Series(h.index) for h in hs],
                                 y_timestamp_list=[pd.Series(d.index[i : i + H]) for i in part], pred_len=H, T=1.0, top_p=0.9,
                                 sample_count=SAMPLES, verbose=False)
        for i, h, o in zip(part, hs, out):
            last = h.close.iloc[-1]
            rows.append({"date": d.index[i], "pred": o["close"].iloc[-1] / last - 1, "act": d.close.iloc[i + H - 1] / last - 1})
    return pd.DataFrame(rows)


def main():
    res = []
    data = {t: kr.bars(t, "1d", start="2021-06-01") for t in TICKERS}
    for mode in ("as shipped (training mode)", "eval mode"):
        pred = kr.load("small", eval_mode=mode == "eval mode")
        print(f"\n== {mode}: model.training = {pred.model.training}", flush=True)
        print(f"A) same input, sampling off, two runs, biggest difference in close: {determinism(pred):.4f}", flush=True)
        for lb in (400, 100):
            for t, d in data.items():
                r = kr.timed(f"{t} lookback {lb}", lambda: run(pred, d, lb))
                res.append({"mode": mode, "lookback": lb, "ticker": t, "n": len(r), "pct_down_calls": (r.pred < 0).mean() * 100,
                            "hit": ((r.pred > 0) == (r.act > 0)).mean() * 100, "always_up": (r.act > 0).mean() * 100,
                            "avg_pred": r.pred.mean() * 100, "avg_act": r.act.mean() * 100,
                            "mse_model": ((r.pred - r.act) ** 2).mean() * 1e4, "mse_no_change": (r.act**2).mean() * 1e4})
                pd.DataFrame(res).round(2).to_csv(kr.OUT / "t3_results.csv", index=False)
    df = pd.DataFrame(res)
    print(df.round(2).to_string(index=False))
    print(df.groupby(["mode", "lookback"])[["pct_down_calls", "hit", "always_up", "avg_pred", "avg_act", "mse_model", "mse_no_change"]]
          .mean().round(2).to_string())


if __name__ == "__main__":
    main()
