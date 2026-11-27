"""One ruler for every model in this folder: paths, splits and the volatility scores.
- R2 of log variance, QLIKE on variance, cross-sectional rank IC (vol_scores, used inside training)
- calibrated QLIKE: each forecast's own best constant level shift removed, so forecasting high earns nothing (qlike_parts)
- Diebold-Mariano t on a daily loss differential, Newey-West errors, 5 lags (nw_t)
Data and runs live in this folder (data/, runs/), or in the folder named by the KRONOS_VOL_DIR environment variable.
"""
import datetime
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("KRONOS_VOL_DIR") or Path(__file__).resolve().parent)
DATA = ROOT / "data"
RUNS = ROOT / "runs"
TRAIN = ("2021-07-01", "2023-12-31")
VAL = ("2024-01-01", "2024-06-30")
TEST = ("2024-07-01", "2026-11-10")      # after Kronos's pretraining data (about mid 2024)
F = ["lrv_d", "lrv_w", "lrv_m", "l_on", "l_last", "ret_d", "spy_lrv_d", "spy_lrv_w", "spy_lrv_m"]
HAR = ["lrv_d", "lrv_w", "lrv_m"]


def rows(df, span):
    return df[(df.date >= span[0]) & (df.date <= span[1])]


def vol_scores(df, col, target="lrv1", resid_var=None):
    """df rows with a log-variance forecast `col` and target `target` (log). Returns R2, MSE, QLIKE, rank IC."""
    x = df[[col, target, "date"]].replace([np.inf, -np.inf], np.nan).dropna()
    e = x[target] - x[col]
    r2 = 1 - (e ** 2).sum() / ((x[target] - x[target].mean()) ** 2).sum()
    s2 = resid_var if resid_var is not None else 0.0
    f = np.exp(x[col] + s2 / 2); y = np.exp(x[target])
    qlike = (y / f - np.log(y / f) - 1).mean()
    ic = x.groupby("date").apply(lambda g: g[col].corr(g[target], method="spearman"), include_groups=False).mean()
    return {"n": len(x), "r2": round(r2, 4), "mse": round((e ** 2).mean(), 4), "qlike": round(qlike, 4), "cs_ic": round(ic, 4)}


def by_year(df, col, target="lrv1"):
    out = {}
    for y, g in df.groupby(df.date.dt.year):
        out[int(y)] = vol_scores(g, col, target)["r2"]
    return out


def qlike_parts(y, p):
    """Per-row QLIKE and calibrated QLIKE of log forecasts p against log realised variance y."""
    r = np.exp(y - p)
    c = np.log(r.mean())
    rc = np.exp(y - p - c)
    return r - np.log(r) - 1, rc - np.log(rc) - 1


def nw_t(s, lags=5):
    """t-stat of the mean of a series, Newey-West errors."""
    s = np.asarray(s, float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 3:
        return np.nan
    e = s - s.mean()
    v = (e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return s.mean() / np.sqrt(v / n) if v > 0 else np.nan


def log_result(name, payload):
    with open(ROOT / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"name": name, "at": datetime.datetime.now().isoformat(timespec="seconds"), **payload}, default=float) + "\n")


def load_panel(name="intra_daily200.parquet"):
    return pd.read_parquet(DATA / name).replace([np.inf, -np.inf], np.nan)
