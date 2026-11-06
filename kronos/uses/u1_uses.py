"""Practical uses, measured on the same 12 names and dates as the agent test (agent_test/forecasts.csv, Kronos-small, lookback 100, as shipped).

    python u1_uses.py

1. Is the p10 to p90 range honest? It should hold the real 5 session move 80% of the time.
2. Is the volatility forecast useful for sizing? Rank correlation of predicted volatility with what followed, against simply using the last 20 sessions.
3. Does the spread of the paths say anything about risk?
Each runs on the control period too, the data Kronos may have seen. Needs agent_test/data, see agent_test/a0_data.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent / "agent_test"


def build():
    f = pd.read_csv(ROOT / "forecasts.csv")
    rows = []
    for t, g in f.groupby("ticker"):
        d = pd.read_csv(ROOT / "data" / f"{t}.csv", parse_dates=["timestamps"], index_col="timestamps")
        lr = np.log(d.close).diff()
        for _, r in g.iterrows():
            i = d.index.get_loc(pd.Timestamp(r.date))
            if i + 5 >= len(d):
                continue
            recent = lr.iloc[i - 19 : i + 1].std()
            ewma = np.sqrt((lr.iloc[: i + 1] ** 2).ewm(alpha=0.06).mean().iloc[-1])        # RiskMetrics, lambda 0.94
            rows.append({**r.to_dict(), "act": (d.close.iloc[i + 5] / d.close.iloc[i] - 1) * 100, "rv_next": lr.iloc[i + 1 : i + 6].std(), "recent": recent, "ewma": ewma,
                         "pred_vol": r.vol_ratio * recent, "spread": r.p90 - r.p10})
    return pd.DataFrame(rows)


def spearman_panel(x, a, b):
    """Mean of per ticker rank correlations, so a volatile name against a calm name does not drive the number."""
    return x.groupby("ticker").apply(lambda g: g[a].corr(g[b], method="spearman"), include_groups=False).mean()


def main():
    x = build()
    x.to_csv(Path(__file__).parent / "uses_panel.csv", index=False)
    for period, g in x.groupby("period"):
        print(f"\n===== {period}: {len(g)} forecasts, {g.date.min()} to {g.date.max()}")
        cover = ((g.act >= g.p10) & (g.act <= g.p90)).mean() * 100
        print(f"p10 to p90 holds the real move: {cover:.1f}% of the time (should be 80%). Above p90 {(g.act > g.p90).mean() * 100:.1f}%, below p10 {(g.act < g.p10).mean() * 100:.1f}%")
        print(f"Average width of that range {g.spread.mean():.1f} points. Width needed for 80%: about {np.percentile(np.abs(g.act - g.p50), 80) * 2:.1f}")
        print("Rank correlation with the next 5 sessions' realised volatility, within each name:")
        for name, col in (("Kronos predicted volatility", "pred_vol"), ("Width of the paths", "spread"), ("Last 20 sessions", "recent"), ("EWMA 0.94, the 1990s formula", "ewma")):
            print(f"  {name:32} {spearman_panel(g, col, 'rv_next'):+.3f}")
        err = lambda c: (np.log(g[c]) - np.log(g.rv_next.clip(lower=1e-5))).abs().mean()
        print(f"Mean absolute log error of the volatility forecast: Kronos {err('pred_vol'):.3f}, last 20 {err('recent'):.3f}, EWMA {err('ewma'):.3f}")
        print(f"Average predicted vs realised daily vol: {g.pred_vol.mean() * 100:.2f}% vs {g.rv_next.mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
