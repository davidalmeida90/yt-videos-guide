"""Score the fine-tune against the simple rules and the control on TEST, by default on the held-out stocks only.
Arms: YDAY, HAR, HARPLUS, GBM (har.py) and the seed ensembles of the runs given (mean of their log forecasts).
    per arm   calibrated QLIKE (each arm's own level shift removed), lower loss than HAR in %, Diebold-Mariano t of
              calibrated QLIKE against HAR (daily cross-sectional mean loss, Newey-West 5 lags), R2 on log RV,
              mean daily rank IC, and the rank IC left after HAR (per session, forecast and realised log RV both
              regressed on HAR across stocks first)
    then      the fine-tune against the control (DM t), each TEST year, and stock by stock
    py -3.11 evaluate.py --kronos RUN_s0 RUN_s1 RUN_s2 --control CTRL_s0 CTRL_s1 CTRL_s2 [--subset holdout|seen|all]
"""
import argparse

import numpy as np
import pandas as pd

import evallib as E
from har import add_baselines

ap = argparse.ArgumentParser()
ap.add_argument("--kronos", nargs="+", required=True, help="fine-tune runs (folders in runs/), averaged")
ap.add_argument("--control", nargs="*", default=[], help="the same head without Kronos, averaged")
ap.add_argument("--panel", default="intra_daily200.parquet")
ap.add_argument("--subset", default="holdout", choices=["holdout", "seen", "all"])
a = ap.parse_args()


def ensemble(runs):
    q = [pd.read_parquet(E.RUNS / r / "test_preds.parquet").set_index(["date", "symbol"]).pred for r in runs]
    return pd.concat(q, axis=1).mean(axis=1)


R = E.rows(add_baselines(E.load_panel(a.panel)), E.TEST).rename(columns={"lrv1": "y"})
R = R[["date", "symbol", "y", "YDAY", "HAR", "HARPLUS", "GBM"]]
h = pd.read_parquet(E.RUNS / a.kronos[0] / "test_preds.parquet")[["date", "symbol", "holdout"]]
R = R.merge(h, on=["date", "symbol"], how="left")
R = R.merge(ensemble(a.kronos).rename("kronos").reset_index(), on=["date", "symbol"], how="left")
ARMS = ["YDAY", "HAR", "HARPLUS", "GBM"]
if a.control:
    R = R.merge(ensemble(a.control).rename("control").reset_index(), on=["date", "symbol"], how="left")
    ARMS.append("control")
ARMS.append("kronos")
if a.subset != "all":
    R = R[R.holdout == (a.subset == "holdout")]
x = R.dropna(subset=["y"] + ARMS)
if x.empty:
    raise SystemExit("no rows: check the run names and --subset")

out = []
_, har_c = E.qlike_parts(x.y.values, x.HAR.values)
for arm in ARMS:
    q, qc = E.qlike_parts(x.y.values, x[arm].values)
    r2 = 1 - ((x.y - x[arm]) ** 2).sum() / ((x.y - x.y.mean()) ** 2).sum()
    ic = x.groupby("date").apply(lambda g: g[arm].corr(g.y, method="spearman") if len(g) > 4 else np.nan, include_groups=False)
    d = pd.Series(qc - har_c, index=x.date.values).groupby(level=0).mean()
    left = np.nan
    if arm not in ("HAR", "YDAY"):
        vals = []
        for _, g in x.groupby("date"):
            if len(g) < 10:
                continue
            A = np.c_[np.ones(len(g)), g.HAR.values]
            ry = g.y.values - A @ np.linalg.lstsq(A, g.y.values, rcond=None)[0]
            rf = g[arm].values - A @ np.linalg.lstsq(A, g[arm].values, rcond=None)[0]
            vals.append(pd.Series(rf).corr(pd.Series(ry), method="spearman"))
        left = np.nanmean(vals)
    out.append({"arm": arm, "cal_QLIKE": qc.mean(), "vs_HAR_%": (har_c.mean() - qc.mean()) / har_c.mean() * 100,
                "t_vs_HAR": E.nw_t(d) if arm != "HAR" else np.nan, "R2": r2, "rank_IC": ic.mean(), "IC_after_HAR": left,
                "QLIKE": q.mean()})
print(f"TEST {a.subset} stocks: {len(x):,} stock-days, {x.symbol.nunique()} stocks, {x.date.nunique()} sessions "
      f"({x.date.min().date()} to {x.date.max().date()})")
print(pd.DataFrame(out).set_index("arm").round(4).to_string())

_, kc = E.qlike_parts(x.y.values, x.kronos.values)
if a.control:
    _, cc = E.qlike_parts(x.y.values, x.control.values)
    d = pd.Series(kc - cc, index=x.date.values).groupby(level=0).mean()
    print(f"\nfine-tune against the control: calibrated QLIKE {(cc.mean() - kc.mean()) / cc.mean() * 100:.1f}% lower, DM t {E.nw_t(d):+.2f}")
print("\ncalibrated QLIKE by TEST year:")
for yr, g in x.groupby(x.date.dt.year):
    line = "  ".join(f"{arm} {E.qlike_parts(g.y.values, g[arm].values)[1].mean():.4f}" for arm in ARMS[::-1] if arm != "YDAY")
    print(f"  {yr}: {line}")
wins = {arm: 0 for arm in ARMS if arm != "kronos"}
for _, g in x.groupby("symbol"):
    k = E.qlike_parts(g.y.values, g.kronos.values)[1].mean()
    for arm in wins:
        wins[arm] += k < E.qlike_parts(g.y.values, g[arm].values)[1].mean()
n = x.symbol.nunique()
print("\nstock by stock, the fine-tune has the lower calibrated QLIKE than: " + ", ".join(f"{arm} on {w} of {n}" for arm, w in wins.items()))
