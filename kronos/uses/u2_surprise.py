"""Surprise score: how unlikely the model found each real candle, in bits. A generative model gives this and a forecast chart does not.

    python u2_surprise.py

Questions, on 12 names from Jul 2024:
1. Which candles surprised it most? Does that line up with real shocks?
2. Does surprise today say anything about volatility over the next 5 sessions, beyond the simple size of today's move and recent volatility?
3. As a baseline for scale: uniform guessing over 1024 tokens costs exactly 10 bits.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "desk"))
import kr  # noqa: E402
from engine import Engine  # noqa: E402

TICKERS = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMZN", "JPM", "XOM", "UNH", "WMT", "TSLA", "KO"]
OUT = Path(__file__).parent / "surprise.csv"


def main():
    e = Engine("small")
    rows = []
    done = set()
    if OUT.exists():
        p = pd.read_csv(OUT)
        rows = p.to_dict("records")
        done = set(zip(p.ticker, p.date))
    for t in TICKERS:
        d = kr.bars(t, "1d", start="2023-06-01")
        days = [x for x in d.index if x >= pd.Timestamp("2024-07-01") and (t, str(x.date())) not in done]
        if days:
            for day, bits in e.surprise(d, days, 100):
                rows.append({"ticker": t, "date": day, "bits": bits})
            pd.DataFrame(rows).to_csv(OUT, index=False)
        print(t, len(days), "new days", flush=True)
    s = pd.DataFrame(rows)
    panel = []
    for t, g in s.groupby("ticker"):
        d = kr.bars(t, "1d", start="2023-06-01")
        lr = np.log(d.close).diff()
        g = g.assign(date=pd.to_datetime(g.date)).set_index("date")
        x = pd.DataFrame({"bits": g.bits, "ret": lr.reindex(g.index), "abs_ret": lr.abs().reindex(g.index), "recent": lr.rolling(20).std().shift(1).reindex(g.index),
                          "next_vol": lr[::-1].rolling(5).std()[::-1].shift(-1).reindex(g.index), "range": ((d.high - d.low) / d.close).reindex(g.index)})
        panel.append(x.assign(ticker=t).reset_index())
    x = pd.concat(panel).dropna()
    x.to_csv(Path(__file__).parent / "surprise_panel.csv", index=False)
    print(f"\n{len(x)} candles. Mean {x.bits.mean():.2f} bits, median {x.bits.median():.2f}, 99th percentile {x.bits.quantile(.99):.2f}. Uniform guessing is 10 bits.")
    print(f"Share of candles worse than uniform guessing: {(x.bits > 10).mean() * 100:.1f}%")
    print("\nMost surprising candles:")
    print(x.sort_values("bits", ascending=False).head(14)[["date", "ticker", "bits", "ret"]].assign(ret=lambda v: (v.ret * 100).round(1), bits=lambda v: v.bits.round(1)).to_string(index=False))
    corr = lambda a, b: x.groupby("ticker").apply(lambda g: g[a].corr(g[b], method="spearman"), include_groups=False).mean()
    print("\nRank correlation within each name:")
    print(f"  surprise vs size of today's move  {corr('bits', 'abs_ret'):+.3f}")
    print(f"  surprise vs today's high-low range {corr('bits', 'range'):+.3f}")
    print(f"  surprise vs next 5 sessions' volatility {corr('bits', 'next_vol'):+.3f}")
    print(f"  size of today's move vs next 5 sessions' volatility {corr('abs_ret', 'next_vol'):+.3f}")
    print(f"  recent 20 session volatility vs next 5 sessions' volatility {corr('recent', 'next_vol'):+.3f}")
    # Adding surprise on top of the simple predictors, per name, in logs.
    gains = []
    for t, g in x.groupby("ticker"):
        y = np.log(g.next_vol.clip(lower=1e-5)).values
        base = np.c_[np.ones(len(g)), np.log(g.recent), np.log(g.abs_ret.clip(lower=1e-5))]
        full = np.c_[base, g.bits]
        r2 = lambda m: 1 - ((y - m @ np.linalg.lstsq(m, y, rcond=None)[0]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        gains.append((r2(base), r2(full)))
    b, f = np.mean(gains, axis=0)
    print(f"\nFit for next 5 sessions' log volatility, in sample, mean over names: simple predictors R2 {b:.3f}, with surprise added {f:.3f}")


if __name__ == "__main__":
    main()
