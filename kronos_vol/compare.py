"""Compare saved TEST forecasts (runs/<run>/test_preds.parquet) with Diebold-Mariano tests against a reference.
Loss per stock-day: QLIKE on variance (y/f - log(y/f) - 1, f = exp(pred)), the same after each model's own best level
shift (QL cal), and the squared error on log variance.
DM: daily cross-sectional mean loss differential, Newey-West t-stat (5 lags). Negative = lower loss than the reference.
Groups: several runs given as a+b+c are averaged first (seed ensemble). HAR and HARPLUS are built in (OLS on TRAIN).
    py -3.11 compare.py --ref CTRL_s0+CTRL_s1+CTRL_s2 RUN_s0+RUN_s1+RUN_s2 HAR HARPLUS --subset holdout
"""
import argparse

import numpy as np
import pandas as pd

import evallib as E
from har import add_baselines

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="lrv1")
ap.add_argument("--ref", required=True)
ap.add_argument("--panel", default="intra_daily200.parquet")
ap.add_argument("--subset", default="all", choices=["all", "seen", "holdout"])
ap.add_argument("runs", nargs="+")
a = ap.parse_args()
TE = E.TEST

D = add_baselines(E.load_panel(a.panel).dropna(subset=[a.target]), a.target, gbm=False)
T = E.rows(D, TE)[["date", "symbol", a.target, "HAR", "HARPLUS"]]


def load(group):
    if group in T.columns:
        return T[["date", "symbol", group]].rename(columns={group: "p"})
    parts = []
    for r in group.split("+"):
        q = pd.read_parquet(E.RUNS / r / "test_preds.parquet")
        if a.subset != "all" and "holdout" in q:
            q = q[q.holdout == (a.subset == "holdout")]
        parts.append(q.set_index(["date", "symbol"]).pred)
    return pd.concat(parts, axis=1).mean(axis=1).rename("p").reset_index()


def losses(group):
    x = T[["date", "symbol", a.target]].merge(load(group), on=["date", "symbol"]).dropna()
    y, f = np.exp(x[a.target]), np.exp(x.p)
    x["ql"] = y / f - np.log(y / f) - 1
    c = np.log((y / f).mean())                     # each model's own best level shift (oracle, same for all): bias removed
    x["qlc"] = y / (f * np.exp(c)) - np.log(y / (f * np.exp(c))) - 1
    x["bias"] = x.p - x[a.target]
    x["se"] = (x[a.target] - x.p) ** 2
    return x


def nw_t(d, lags=5):
    d = d - d.mean(); n = len(d)
    g = [np.dot(d[k:], d[:n - k]) / n for k in range(lags + 1)]
    v = g[0] + 2 * sum((1 - k / (lags + 1)) * g[k] for k in range(1, lags + 1))
    return np.sqrt(v / n)


if a.subset != "all":                              # built-in baselines on the same stocks as the runs
    T = T.merge(load(a.ref)[["date", "symbol"]], on=["date", "symbol"])
R = losses(a.ref)
print(f"target {a.target}, TEST {TE[0]}..{TE[1]}, {a.subset} stocks; reference = {a.ref}")
print(f"{'model':60s} {'QLIKE':>7s} {'QL cal':>7s} {'bias':>6s} {'R2':>7s}   DM t vs ref (QLIKE, QLIKE cal, SE)   QLIKE by year")
for g in [a.ref] + a.runs:
    X = losses(g)
    m = X.merge(R[["date", "symbol", "ql", "se"]], on=["date", "symbol"], suffixes=("", "_r"))
    dq = m.groupby("date").apply(lambda z: (z.ql - z.ql_r).mean(), include_groups=False)
    ds = m.groupby("date").apply(lambda z: (z.se - z.se_r).mean(), include_groups=False)
    tq = dq.mean() / nw_t(dq.values) if g != a.ref else 0.0
    m2 = X.merge(R[["date", "symbol", "qlc"]], on=["date", "symbol"], suffixes=("", "_r"))
    dc = m2.groupby("date").apply(lambda z: (z.qlc - z.qlc_r).mean(), include_groups=False)
    tc = dc.mean() / nw_t(dc.values) if g != a.ref else 0.0
    ts = ds.mean() / nw_t(ds.values) if g != a.ref else 0.0
    r2 = 1 - X.se.sum() / ((X[a.target] - X[a.target].mean()) ** 2).sum()
    yr = {int(y): round(float(z.ql.mean()), 4) for y, z in X.groupby(X.date.dt.year)}
    print(f"{g[:60]:60s} {X.ql.mean():7.4f} {X.qlc.mean():7.4f} {X.bias.mean():+6.3f} {r2:7.4f}   {tq:+6.2f} {tc:+6.2f} {ts:+6.2f}   {yr}   n={len(X)}")
