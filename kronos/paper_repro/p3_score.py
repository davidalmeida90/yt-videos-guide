"""Score the signals with the repo's backtest settings: rank IC by horizon, TopK dropout backtest, the floor, the reversal check.

    python p3_score.py --model base --universe csi300 [--no-backtest]
Writes results_<universe>_<model>.csv and rank_ic_<universe>_<model>.csv. Needs data_<universe>.csv from p1_data.py and, for the backtest, pyqlib.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
WINDOWS = {"A paper window": ("2024-07-01", "2025-06-05"), "B new": ("2025-06-06", "2026-12-31")}
KRONOS = ["avg_return", "mean", "last", "max", "min"]
BENCH = {"csi300": "SH000300", "csi800": "SH000906"}


def panel(data):
    close = pd.DataFrame({s: d["close"] for s, d in data.items()}).sort_index()
    return close


def floor_signals(data):
    """1 line price signals on the same rows as Kronos. Positive = expect to outperform."""
    out = {}
    for s, d in data.items():
        c = d["close"]
        roll_m, roll_s = c.rolling(90).mean(), c.rolling(90).std()
        out[s] = pd.DataFrame({"rev5": -(c / c.shift(5) - 1), "rev10": -(c / c.shift(10) - 1), "rev20": -(c / c.shift(20) - 1),
                               "rev60": -(c / c.shift(60) - 1), "dist90": -(c / roll_m - 1), "z90": -((c - roll_m) / (roll_s + 1e-5))})
    f = pd.concat(out, names=["instrument", "datetime"]).reset_index()
    for k in ("rev5", "rev10", "rev20", "rev60", "dist90", "z90"):
        f["-" + k] = -f[k]
    return f


def rank_ic(sig_wide, ret_wide):
    a, b = sig_wide.align(ret_wide, join="inner")
    return a.rank(axis=1).corrwith(b.rank(axis=1), axis=1).dropna()


def newey_west_t(x, lag):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n, e = len(x), x - x.mean()
    v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(v / n)


def wide(df, col):
    return df.pivot_table(index="datetime", columns="instrument", values=col).sort_index()


def run_backtest(series, window, bench):
    codes = sorted(series.index.get_level_values("instrument").unique())   # the exchange otherwise loads all 6,000+ stocks
    from qlib.backtest import backtest, executor
    from qlib.contrib.evaluate import risk_analysis
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.utils.time import Freq

    strategy = TopkDropoutStrategy(topk=50, n_drop=5, hold_thresh=5, signal=series)
    pm, _ = backtest(strategy=strategy, start_time=window[0], end_time=window[1], account=100_000_000, benchmark=bench,
                     exchange_kwargs={"codes": codes, "freq": "day", "limit_threshold": 0.095, "deal_price": "open", "open_cost": 0.001, "close_cost": 0.0015, "min_cost": 5},
                     executor=executor.SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True, delay_execution=True))
    freq = "{0}{1}".format(*Freq.parse("day"))
    report, _ = pm.get(freq)
    r_gross = risk_analysis(report["return"] - report["bench"], freq=freq)["risk"]
    r_net = risk_analysis(report["return"] - report["bench"] - report["cost"], freq=freq)["risk"]
    return {"bench_%": report["bench"].sum() * 100, "AER_gross_%": r_gross["annualized_return"] * 100, "IR_gross": r_gross["information_ratio"],
            "AER_net_%": r_net["annualized_return"] * 100, "IR_net": r_net["information_ratio"], "maxDD_%": r_net["max_drawdown"] * 100,
            "turnover": report["turnover"].mean()}, (report["return"] - report["bench"] - report["cost"])


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="base")
    a.add_argument("--universe", default="csi300")
    a.add_argument("--no-backtest", action="store_true")
    o = a.parse_args()
    raw = pd.read_csv(HERE / f"data_{o.universe}.csv", parse_dates=["datetime"])
    data = {s: g.set_index("datetime").sort_index() for s, g in raw.groupby("instrument")}
    sig = pd.read_csv(HERE / f"signals_{o.universe}_{o.model}.csv", parse_dates=["datetime"]).drop_duplicates(["datetime", "instrument"])
    print(f"{len(sig)} windows, {sig.datetime.min().date()} to {sig.datetime.max().date()}", flush=True)
    close = panel(data)
    fl = floor_signals(data)
    floors = [c for c in fl.columns if c not in ("instrument", "datetime")]
    all_sig = sig.merge(fl, on=["instrument", "datetime"], how="left")

    ic_rows, res = [], []
    for wname, (s, e) in WINDOWS.items():
        g = all_sig[(all_sig.datetime >= s) & (all_sig.datetime <= e)]
        if g.empty:
            continue
        print(f"\n===== {wname}: {g.datetime.min().date()} to {g.datetime.max().date()}, {g.datetime.nunique()} days", flush=True)
        for name in KRONOS + floors:
            w = wide(g, name)
            row = {"window": wname, "signal": name}
            for h in (1, 2, 5, 10):
                fwd = close.shift(-h) / close - 1
                ic = rank_ic(w, fwd)
                row[f"ic{h}"], row[f"t{h}"] = ic.mean(), newey_west_t(ic.values, h - 1)
            ic_rows.append(row)
        ic_df = pd.DataFrame([r for r in ic_rows if r["window"] == wname]).round(4)
        print(ic_df.drop(columns="window").to_string(index=False), flush=True)

        # Reversal check: how much of Kronos is the best reversal signal?
        kw = wide(g, "avg_return")
        print("\nDaily rank correlation of Kronos avg_return with each floor signal", flush=True)
        for name in floors[:6]:
            c = rank_ic(kw, wide(g, name))
            print(f"  {name:8} {c.mean():+.3f}", flush=True)
        positive = [r for r in ic_rows if r["window"] == wname and r["signal"] in floors]
        best = max(positive, key=lambda r: r["ic1"])["signal"]
        print(f"Best floor by h=1 rank IC: {best}", flush=True)
        fwd = close.shift(-1) / close - 1
        for sname in ("avg_return", "mean"):
            kr_rank, fl_rank = wide(g, sname).rank(axis=1), wide(g, best).rank(axis=1)
            resid = {}
            for d in kr_rank.index:
                x, y = fl_rank.loc[d], kr_rank.loc[d]
                ok = x.notna() & y.notna()
                if ok.sum() > 30:
                    b = np.polyfit(x[ok], y[ok], 1)
                    resid[d] = y[ok] - np.polyval(b, x[ok])
            rw = pd.DataFrame(resid).T
            ic = rank_ic(rw, fwd)
            print(f"Rank IC of Kronos {sname} after removing {best}: {ic.mean():+.4f} (t {newey_west_t(ic.values, 0):+.2f})", flush=True)

        if not o.no_backtest:
            import qlib
            from qlib.config import REG_CN
            qlib.init(provider_uri=str(Path.home() / ".qlib" / "qlib_data" / "cn_data"), region=REG_CN, kernels=1)
            names = KRONOS + [best] + [f for f in ("rev5", "z90") if f != best]
            for name in names:
                ser = g.set_index(["datetime", "instrument"])[name].dropna().sort_index()
                ser.index.names = ["datetime", "instrument"]
                print(f"backtest {name} ...", flush=True)
                m, daily = run_backtest(ser, (str(g.datetime.min().date()), str(g.datetime.max().date())), BENCH[o.universe])
                res.append({"window": wname, "signal": name, **m})
                daily.to_csv(HERE / f"daily_{o.universe}_{o.model}_{wname[0]}_{name.replace('-', 'neg_')}.csv")
            print(pd.DataFrame([r for r in res if r["window"] == wname]).round(3).drop(columns="window").to_string(index=False), flush=True)
    pd.DataFrame(ic_rows).round(5).to_csv(HERE / f"rank_ic_{o.universe}_{o.model}.csv", index=False)
    if res:
        pd.DataFrame(res).round(4).to_csv(HERE / f"results_{o.universe}_{o.model}.csv", index=False)


if __name__ == "__main__":
    main()
