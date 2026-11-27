"""The simple rules every fine-tune has to beat, fitted on the daily panel (CPU only):
    YDAY      today's log RV as tomorrow's
    HAR       HAR-RV: OLS of next-day log RV on log RV over 1, 5 and 22 sessions, fitted on TRAIN
    HARPLUS   the same OLS on all 9 features (adds the overnight gap, the last hour, the daily return and SPY's RV)
    GBM       LightGBM on the 9 features, early stopping on VAL (the settings ft_intra.py uses)
    py -3.11 har.py [--panel intra_daily200.parquet]     TEST scores of the rules alone, all stocks
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import evallib as E


def add_baselines(D, target="lrv1", gbm=True):
    """D: the panel. Returns a copy with the columns YDAY, HAR, HARPLUS (and GBM) for every row with all 9 features."""
    D = D.replace([np.inf, -np.inf], np.nan).dropna(subset=E.F).copy()
    tr = E.rows(D, E.TRAIN).dropna(subset=[target])
    D["YDAY"] = D.lrv_d
    for k, cols in {"HAR": E.HAR, "HARPLUS": E.F}.items():
        D[k] = LinearRegression().fit(tr[cols], tr[target]).predict(D[cols])
    if gbm:
        import lightgbm as lgb
        va = E.rows(D, E.VAL).dropna(subset=[target])
        g = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.02, num_leaves=15, min_child_samples=100, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        g.fit(tr[E.F], tr[target], eval_set=[(va[E.F], va[target])], callbacks=[lgb.early_stopping(100, verbose=False)])
        D["GBM"] = g.predict(D[E.F])
    return D


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="intra_daily200.parquet")
    a = ap.parse_args()
    T = E.rows(add_baselines(E.load_panel(a.panel)), E.TEST).dropna(subset=["lrv1"])
    _, har_c = E.qlike_parts(T.lrv1.values, T.HAR.values)
    out = []
    for arm in ["YDAY", "HAR", "HARPLUS", "GBM"]:
        s = E.vol_scores(T, arm)
        _, qc = E.qlike_parts(T.lrv1.values, T[arm].values)
        d = pd.Series(qc - har_c, index=T.date.values).groupby(level=0).mean()
        out.append({"rule": arm, "cal_QLIKE": qc.mean(), "QLIKE": s["qlike"], "R2": s["r2"], "rank_IC": s["cs_ic"],
                    "t_vs_HAR": E.nw_t(d) if arm != "HAR" else 0.0})
    print(f"TEST {E.TEST[0]} to {E.TEST[1]}: {len(T):,} stock-days, {T.symbol.nunique()} stocks")
    print(pd.DataFrame(out).set_index("rule").round(4).to_string())
