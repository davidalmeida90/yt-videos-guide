"""Sample for the video: Kronos-base on every 5th trading day of CSI300, against the 1 line rules on the same days, with 1 ruler.

    python p5_sample.py

Decided 6 Nov 2026: no full daily run, so this cannot reproduce the paper's +19.11% or its TopK backtest. It says "a sample", never "reproduced".
Ruler: on each sampled day, buy the top 50 by signal from the close, hold 5 sessions, compare with the equal weight of all members. Those holds do not overlap. Cost 0.25% a round trip, charged in full each period, which overstates turnover against the paper's dropout rule and is the same for every signal.
Also rank IC at 5 sessions, and how much Kronos overlaps with each rule. A prediction was written in PLAN.md before this ran.
Needs data_csi300.csv from p1_data.py (Qlib data, not included). signals_csi300_base.csv is included.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
WINDOWS = {"A paper window": ("2024-07-01", "2025-06-05"), "B new": ("2025-06-06", "2026-12-31")}
KRONOS = ["avg_return", "mean", "last"]
RULES = ["rev5", "rev20", "rev60", "dist90", "z90"]
H, TOP, COST, PER_YEAR = 5, 50, 0.0025, 252 / 5


def rules(raw):
    out = []
    for s, d in raw.groupby("instrument"):
        d = d.sort_values("datetime")
        c = d.close
        m, sd = c.rolling(90).mean(), c.rolling(90).std()
        out.append(pd.DataFrame({"instrument": s, "datetime": d.datetime.values, "fwd": (c.shift(-H) / c - 1).values, "rev5": (-(c / c.shift(5) - 1)).values, "rev20": (-(c / c.shift(20) - 1)).values,
                                 "rev60": (-(c / c.shift(60) - 1)).values, "dist90": (-(c / m - 1)).values, "z90": (-((c - m) / (sd + 1e-5))).values}))
    return pd.concat(out)


def newey_west_t(x, lag=1):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    e = x - x.mean()
    v = e @ e / len(x)
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / len(x)
    return x.mean() / np.sqrt(v / len(x))


def main():
    sig = pd.read_csv(HERE / "signals_csi300_base.csv", parse_dates=["datetime"]).drop_duplicates(["datetime", "instrument"])
    raw = pd.read_csv(HERE / "data_csi300.csv", parse_dates=["datetime"])
    days = sorted(sig.datetime.unique())
    counts = sig.groupby("datetime").size()
    full = counts[counts >= 0.9 * counts.median()].index          # a day still being written does not count
    m = sig[sig.datetime.isin(full)].merge(rules(raw), on=["instrument", "datetime"], how="left").dropna(subset=["fwd"])
    np.random.seed(1)
    m["random"] = np.random.random(len(m))
    print(f"{len(m)} stock days, {m.datetime.nunique()} sampled days, {m.datetime.min().date()} to {m.datetime.max().date()}")
    rows = []
    for w, (a, b) in WINDOWS.items():
        g = m[(m.datetime >= a) & (m.datetime <= b)]
        if g.empty:
            continue
        print(f"\n===== {w}: {g.datetime.nunique()} sampled days, {g.datetime.min().date()} to {g.datetime.max().date()}")
        for name in KRONOS + RULES + ["random"]:
            ic = g.groupby("datetime").apply(lambda d: d[name].corr(d.fwd, method="spearman"), include_groups=False)
            ex = g.groupby("datetime").apply(lambda d: d.nlargest(TOP, name).fwd.mean() - d.fwd.mean() - COST, include_groups=False)
            rows.append({"window": w, "signal": name, "periods": len(ex), "rank_ic": ic.mean(), "ic_t": newey_west_t(ic.values), "excess_per_year_%": ex.mean() * PER_YEAR * 100, "excess_t": ex.mean() / ex.std() * np.sqrt(len(ex)), "hit_%": (ex > 0).mean() * 100})
        t = pd.DataFrame([r for r in rows if r["window"] == w]).drop(columns="window").round(3)
        print(t.to_string(index=False))
        print("Daily rank correlation of Kronos avg_return with: " + ", ".join(f"{r} {g.groupby('datetime').apply(lambda d: d.avg_return.corr(d[r], method='spearman'), include_groups=False).mean():+.2f}" for r in RULES))
        best = "dist90"
        resid = g.groupby("datetime").apply(lambda d: (lambda x, y, f: pd.Series(y - np.polyval(np.polyfit(x, y, 1), x)).corr(pd.Series(f), method="spearman"))(d[best].rank().values, d.avg_return.rank().values, d.fwd.values), include_groups=False)
        print(f"Rank IC of Kronos avg_return with {best} removed: {resid.mean():+.4f} (t {newey_west_t(resid.values):+.2f})")
    pd.DataFrame(rows).round(4).to_csv(HERE / "sample_results.csv", index=False)


if __name__ == "__main__":
    main()
