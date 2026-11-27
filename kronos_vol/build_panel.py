"""Daily realised variance from the 5-minute bars (regular session) and the HAR-RV features: one row per stock and session.
    RV_t      sum of squared 5-minute log returns in session t (first bar from its own open)
    on_t      squared overnight log return, open_t / close_{t-1}
    lrv1      target: log RV_{t+1};   lrv5: log mean RV over t+1..t+5
    features  lrv_d, lrv_w (5 days), lrv_m (22 days), l_on, l_last (last-hour RV), ret_d, and SPY's lrv_d / lrv_w / lrv_m
Sessions with fewer than 70 of the 78 bars are dropped (half days, gaps).
Splits: TRAIN 2021-07 to 2023-12, VAL 2024-01 to 2024-06, TEST 2024-07 to 2026-11 (after Kronos's pretraining data).
    py -3.11 build_panel.py 200     ->  data/intra_daily200.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import evallib as E

IN = E.DATA / "intraday"
TOP = int(sys.argv[1]) if len(sys.argv) > 1 else 200      # the first TOP names of symbols.txt (+ SPY)
OUTF = "intra_daily.parquet" if TOP == 60 else f"intra_daily{TOP}.parquet"
names = (Path(__file__).resolve().parent / "symbols.txt").read_text().split()
keep = set([x for x in names if x != "SPY"][:TOP]) | {"SPY"}
rows = []
for f in sorted(IN.glob("*.parquet")):
    if f.stem not in keep:
        continue
    d = pd.read_parquet(f).sort_values("ts")
    d["date"] = d.ts.dt.normalize()
    d["r"] = np.log(d.close / d.close.shift(1))
    first = d.date != d.date.shift(1)
    d.loc[first, "r"] = np.log(d.close / d.open)[first]
    d["late"] = (d.ts.dt.hour >= 15).astype(float)
    g = d.groupby("date")
    day = pd.DataFrame({"n": g.size(), "rv": g.r.apply(lambda s: (s ** 2).sum()), "o": g.open.first(), "c": g.close.last(),
                        "rv_last": g.apply(lambda x: ((x.r ** 2) * x.late).sum(), include_groups=False)})
    day = day[day.n >= 70].copy()
    day["on"] = np.log(day.o / day.c.shift(1)) ** 2
    rv = day.rv.clip(lower=1e-9)
    day["lrv_d"] = np.log(rv)
    day["lrv_w"] = np.log(rv.rolling(5).mean())
    day["lrv_m"] = np.log(rv.rolling(22).mean())
    day["l_on"] = np.log(day.on + 1e-8)
    day["l_last"] = np.log(day.rv_last + 1e-9)
    day["ret_d"] = np.log(day.c / day.c.shift(1))
    day["lrv1"] = np.log(rv.shift(-1))
    day["lrv5"] = np.log(sum(rv.shift(-k) for k in range(1, 6)) / 5)
    day["symbol"] = f.stem
    rows.append(day.reset_index())
if not rows:
    raise SystemExit(f"no bars in {IN}: run download_bars.py first")
P = pd.concat(rows, ignore_index=True)
spy = P[P.symbol == "SPY"].set_index("date")[["lrv_d", "lrv_w", "lrv_m"]].add_prefix("spy_")
P = P[P.symbol != "SPY"].join(spy, on="date")
P.to_parquet(E.DATA / OUTF, index=False)
print(f"{P.symbol.nunique()} stocks, {len(P):,} stock-days, {P.date.min().date()} to {P.date.max().date()}; "
      f"median bars {P.n.median():.0f}  ->  {E.DATA / OUTF}")
