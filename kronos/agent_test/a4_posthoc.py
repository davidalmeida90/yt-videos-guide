"""Post hoc, see PLAN.md: alpha and beta of each arm against buy and hold, and whether Kronos's calls are just reversal. Needs data/, see a0_data.py.

    python a4_posthoc.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import a3_score as s

HERE = Path(__file__).parent
PER_YEAR = 252 / 5
POS = {"long": 1, "flat": 0, "short": -1}


def alpha_beta(y, x):
    b = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    e = y - b * x
    a = e.mean()
    return a * PER_YEAR * 100, a / (e.std(ddof=2) / np.sqrt(len(y))), b


def main():
    df = s.load()
    bars = {t: pd.read_csv(HERE / "data" / f"{t}.csv", parse_dates=["timestamps"], index_col="timestamps") for t in df.ticker.unique()}
    rev, rev5, rev20 = [], [], []
    for _, r in df.iterrows():
        c = bars[r.ticker].close
        i = c.index.get_loc(pd.Timestamp(r.date))
        rev.append(-np.sign(c.iloc[i] / c.iloc[i - 99 : i + 1].mean() - 1))
        rev5.append(-np.sign(r.mom5)); rev20.append(-np.sign(r.mom20))
    df["rev100"], df["rev5"], df["rev20"] = rev, rev5, rev20
    calls = pd.DataFrame([json.loads(l) for l in (HERE / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()])
    for period, g in df.groupby("period"):
        g = g.reset_index(drop=True)
        arms = {"Buy and hold": pd.Series(1, index=g.index), "Kronos rule": pd.Series(np.sign(g.up_share - 0.5)).astype(int),
                "Reversal, 100 session average": g.rev100.astype(int), "Reversal, 5 sessions": g.rev5.astype(int), "Reversal, 20 sessions": g.rev20.astype(int)}
        for arm, label in (("none", "LLM, no forecast"), ("kronos", "LLM + Kronos"), ("placebo", "LLM + placebo")):
            c = calls[(calls.period == period) & (calls.arm == arm)]
            if len(c):
                arms[label] = g.merge(c[["ticker", "date", "position"]], on=["ticker", "date"], how="left").position.map(POS).fillna(0).astype(int)
        bh = s.book(g, arms["Buy and hold"])
        print(f"\n===== {period}")
        print(f"{'':32}{'total %':>9}{'Sharpe':>8}{'alpha %/y':>11}{'t':>7}{'beta':>7}{'avg net exposure':>18}")
        for label, pos in arms.items():
            w = s.book(g, pos)
            st = s.stats(w)
            a, t, b = alpha_beta(w.values, bh.values)
            print(f"{label:32}{st['total_%']:9.1f}{st['sharpe']:8.2f}{a:11.1f}{t:7.2f}{b:7.2f}{pos.mean():18.2f}")
        k = np.sign(g.up_share - 0.5)
        print("Kronos direction matches: " + ", ".join(f"{n} {(k == g[n]).mean() * 100:.0f}%" for n in ("rev100", "rev20", "rev5")))
        print(f"Rank correlation of up_share with 20 session momentum, within names: {g.assign(d=-g.rev100).groupby('ticker').apply(lambda x: x.up_share.corr(x.mom20, method='spearman'), include_groups=False).mean():+.3f}")


if __name__ == "__main__":
    main()
