"""Synthetic market paths: the paper also claims realistic generation. Does a sampled path behave like a real one?

    python u3_synthetic.py

Takes real windows of daily SPY, NVDA, and KO from Jul 2024, generates 60 bars forward, and compares the textbook market features: size of moves, fat tails,
volatility clustering, autocorrelation, high low range. Reference: the real bars that followed, plus a simple bootstrap of past returns, which has fat tails but no memory.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "desk"))
import kr  # noqa: E402
from engine import Engine  # noqa: E402

H, LOOKBACK, N, STARTS = 60, 100, 16, 10


def facts(close, high=None, low=None):
    """close: paths x bars. Facts pooled across paths."""
    r = np.diff(np.log(close), axis=1)
    pool = r.ravel()
    d = pool - pool.mean()
    kurt = (d**4).mean() / (d**2).mean() ** 2 - 3
    ac = lambda x, k=1: np.nanmean([np.corrcoef(p[:-k], p[k:])[0, 1] for p in x if np.std(p) > 0])
    out = {"daily_vol_%": r.std() * 100, "excess_kurtosis": kurt, "autocorr_returns": ac(r), "autocorr_abs_returns": ac(np.abs(r)), "worst_day_%": pool.min() * 100, "best_day_%": pool.max() * 100,
           "drift_%_per_day": pool.mean() * 100}
    if high is not None:
        out["high_low_range_%"] = ((high - low) / close).mean() * 100
    return out


def main():
    e = Engine("small")
    res = []
    rng = np.random.default_rng(7)
    for t in ("SPY", "NVDA", "KO"):
        d = kr.bars(t, "1d", start="2023-01-01")
        idx = [i for i in range(len(d) - H) if d.index[i] >= pd.Timestamp("2024-07-01")]
        starts = idx[:: max(1, len(idx) // STARTS)][:STARTS]
        gen_c, gen_h, gen_l, real_c, real_h, real_l, boot = [], [], [], [], [], [], []
        for k, i in enumerate(starts):
            hist = d.iloc[i - LOOKBACK + 1 : i + 1]
            fut = d.index[i + 1 : i + 1 + H]
            for step, c in e.stream(hist, fut, n=N, seed=k):
                if step == "final":
                    last = hist.close.iloc[-1]
                    gen_c.append(np.c_[np.full(N, last), c[:, :, 3]])
                    gen_h.append(c[:, :, 1]); gen_l.append(c[:, :, 2])
            real_c.append(d.close.iloc[i : i + 1 + H].values[None, :])
            real_h.append(d.high.iloc[i + 1 : i + 1 + H].values[None, :]); real_l.append(d.low.iloc[i + 1 : i + 1 + H].values[None, :])
            past = np.log(d.close.iloc[: i + 1]).diff().dropna().values
            boot.append(np.c_[np.zeros(N), np.cumsum(rng.choice(past, (N, H)), axis=1)])
            print(t, k + 1, "of", len(starts), flush=True)
        gc = np.concatenate(gen_c)
        res.append({"ticker": t, "source": "Kronos paths", **facts(gc, np.concatenate(gen_h), np.concatenate(gen_l)[:, :])} if False else {"ticker": t, "source": "Kronos paths", **facts(gc)})
        res[-1]["high_low_range_%"] = ((np.concatenate(gen_h) - np.concatenate(gen_l)) / gc[:, 1:]).mean() * 100
        rc = np.concatenate(real_c)
        res.append({"ticker": t, "source": "Real bars after", **facts(rc)})
        res[-1]["high_low_range_%"] = ((np.concatenate(real_h) - np.concatenate(real_l)) / rc[:, 1:]).mean() * 100
        res.append({"ticker": t, "source": "Bootstrap", **facts(np.exp(np.concatenate(boot)))})
    r = pd.DataFrame(res).round(3)
    r.to_csv(Path(__file__).parent / "synthetic_facts.csv", index=False)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
