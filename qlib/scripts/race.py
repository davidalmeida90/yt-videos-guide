"""Model race on one dataset: Qlib model-zoo models against plain-rule baselines.

Every entry produces a daily score per stock for the test window and goes through the same
backtest: TopkDropoutStrategy (hold 50, swap at most 5 a day), trade at the close, same costs.
Features are computed once (Alpha158) and cached.

usage: python race.py --market us [--models lgb,linear,dens,xgb,mlp] [--seeds 5]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import qlib  # noqa: E402
from qlib.data import D  # noqa: E402
from qlib.data.dataset import DatasetH  # noqa: E402
from qlib.data.dataset.handler import DataHandlerLP  # noqa: E402
from qlib.contrib.data.handler import Alpha158  # noqa: E402
from qlib.contrib.strategy import TopkDropoutStrategy  # noqa: E402
from qlib.backtest import backtest  # noqa: E402
from qlib.backtest.executor import SimulatorExecutor  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURFACE, INK, INK2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#383835"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

MARKETS = {
    "us": dict(provider="~/.qlib/qlib_data/us_alpaca", region="us", market="sp500", bench="SPY",
               start="2016-01-04", end="2026-11-12", fit=("2016-04-01", "2021-12-31"),
               train=("2016-04-01", "2021-12-31"), valid=("2022-01-03", "2023-12-29"),
               test=("2024-01-02", "2026-11-12"), account=10_000_000, extra=["RSP", "QQQ"],
               exchange=dict(limit_threshold=None, deal_price="close", open_cost=0.0005,
                             close_cost=0.0005, min_cost=0)),
    "cn": dict(provider="~/.qlib/qlib_data/cn_data", region="cn", market="csi300", bench="SH000300",
               start="2008-01-01", end="2026-11-12", fit=("2008-01-01", "2021-12-31"),
               train=("2008-01-01", "2021-12-31"), valid=("2022-01-04", "2023-12-29"),
               test=("2024-01-02", "2026-11-12"), account=100_000_000, extra=[],
               exchange=dict(limit_threshold=0.095, deal_price="close", open_cost=0.0005,
                             close_cost=0.0015, min_cost=5)),
}
LGB = dict(loss="mse", colsample_bytree=0.8879, learning_rate=0.2, subsample=0.8789,
           lambda_l1=205.6999, lambda_l2=580.9768, max_depth=8, num_leaves=210, num_threads=14)
BASELINES = {   # plain rules written as Qlib expressions; higher score = buy
    "Momentum 12-1": "Ref($close, 21) / Ref($close, 252) - 1",
    "Reversal 5d": "Ref($close, 5) / $close - 1",
    "Low volatility": "1 / Std($close / Ref($close, 1) - 1, 60)",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def datasets(cfg, cache):
    """Alpha158 computed fresh, the same way qrun does it. A cached pickle (dump_all keeps raw, learn and
    infer copies) used too much memory on the 16 GB laptop next to the headless recorder, so no cache."""
    seg = {"train": cfg["train"], "valid": cfg["valid"], "test": cfg["test"]}
    t = time.time()
    h_raw = Alpha158(instruments=cfg["market"], start_time=cfg["start"], end_time=cfg["end"],
                     fit_start_time=cfg["fit"][0], fit_end_time=cfg["fit"][1])
    log(f"Alpha158 computed in {time.time() - t:.0f}s")
    ds = DatasetH(h_raw, seg)
    return ds, ds   # demo version: tree models only, so one raw Alpha158 dataset serves all


def make_model(name):
    if name == "lgb":
        from qlib.contrib.model.gbdt import LGBModel
        return "LightGBM", LGBModel(**LGB), "raw"
    if name == "linear":
        from qlib.contrib.model.linear import LinearModel
        return "Linear (OLS)", LinearModel(estimator="ols"), "norm"
    if name == "dens":
        from qlib.contrib.model.double_ensemble import DEnsembleModel
        kw = {k: v for k, v in LGB.items()}
        return "DoubleEnsemble", DEnsembleModel(
            base_model="gbm", num_models=3, enable_sr=True, enable_fs=True, alpha1=1, alpha2=1,
            bins_sr=10, bins_fs=5, decay=0.5, sample_ratios=[0.8, 0.7, 0.6, 0.5, 0.4],
            sub_weights=[1, 1, 1], epochs=28, verbosity=-1, **kw), "raw"
    if name == "xgb":
        from qlib.contrib.model.xgboost import XGBModel
        return "XGBoost", XGBModel(eval_metric="rmse", colsample_bytree=0.8879, eta=0.0421,
                                   max_depth=8, n_estimators=647, subsample=0.8789,
                                   nthread=14), "raw"
    if name == "mlp":
        from qlib.contrib.model.pytorch_nn import DNNModelPytorch
        return "MLP", DNNModelPytorch(loss="mse", lr=0.002, optimizer="adam", max_steps=8000,
                                      batch_size=8192, GPU=0, weight_decay=0.0002,
                                      pt_model_kwargs={"input_dim": 158}), "norm"
    raise ValueError(name)


def ic_stats(score, label):
    df = pd.concat([score.rename("s"), label.rename("y")], axis=1).dropna()
    g = df.groupby(level="datetime")
    ic = g.apply(lambda x: x["s"].corr(x["y"]))
    ric = g.apply(lambda x: x["s"].corr(x["y"], method="spearman"))
    return {"IC": float(ic.mean()), "IC_t": float(ic.mean() / ic.std() * np.sqrt(len(ic))),
            "RankIC": float(ric.mean())}


def perf(r):
    r = r.dropna()
    w = (1 + r).cumprod()
    return {"total_return": float(w.iloc[-1] - 1), "cagr": float(w.iloc[-1] ** (252 / len(r)) - 1),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)),
            "max_drawdown": float((w / w.cummax() - 1).min())}


def run_backtest(score, cfg):
    strat = TopkDropoutStrategy(signal=score, topk=50, n_drop=5)
    exe = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    pm, _ = backtest(start_time=cfg["test"][0], end_time=cfg["test"][1], strategy=strat, executor=exe,
                     benchmark=cfg["bench"], account=cfg["account"], exchange_kwargs=cfg["exchange"])
    rep = pm["1day"][0]
    return rep["return"] - rep["cost"], rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="us")
    ap.add_argument("--models", default="lgb")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    cfg = MARKETS[a.market]
    out = os.path.join(ROOT, "runs", f"race_{a.market}")
    cache = os.path.join(ROOT, "cache", a.market)
    os.makedirs(out, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    qlib.init(provider_uri=os.path.expanduser(cfg["provider"]), region=cfg["region"])

    ds_raw, ds_norm = datasets(cfg, cache)
    label = ds_raw.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R).iloc[:, 0]
    idx = label.index
    results, curves = {}, {}

    def record(name, score, kind, extra=None):
        t = time.time()
        net, rep = run_backtest(score, cfg)
        res = {"kind": kind, **ic_stats(score, label), **perf(net),
               "turnover": float(rep["turnover"].mean())}
        res.update(extra or {})
        results[name] = res
        curves[name] = net
        rep.to_pickle(os.path.join(out, f"report_{name.replace(' ', '_')}.pkl"))
        log(f"{name:<22} IC {res['IC']:+.4f}  CAGR {res['cagr']:+.1%}  Sharpe {res['sharpe']:.2f}"
            f"  MDD {res['max_drawdown']:.1%}  ({time.time() - t:.0f}s backtest)")

    for m in [x for x in a.models.split(",") if x]:
        if m == "lgb_trees":
            # early stopping on squared error stops at tree 1 on daily US returns, so instead
            # grow a fixed number of trees and pick that number on the 2022-2023 validation IC only
            # Qlib's LGBModel always early-stops and LightGBM then saves only the trees up to that
            # pick, so this variant calls LightGBM directly on the same Qlib train/valid/test data.
            import lightgbm as lgb
            tr = ds_raw.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
            xva = ds_raw.prepare("valid", col_set="feature", data_key=DataHandlerLP.DK_I)
            xte = ds_raw.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
            vlabel = ds_raw.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_R).iloc[:, 0]
            params = {k: v for k, v in LGB.items() if k != "loss"}
            params.update(objective="mse", verbosity=-1)
            dtrain = lgb.Dataset(tr["feature"].values, label=tr["label"].iloc[:, 0].values)
            best = None
            for n in (100, 300):
                t = time.time()
                booster = lgb.train(params, dtrain, num_boost_round=n)
                pva = pd.Series(booster.predict(xva.values), index=xva.index)
                v = ic_stats(pva.reindex(vlabel.index), vlabel)["IC"]
                log(f"  {n} trees: validation IC {v:+.4f} ({time.time() - t:.0f}s)")
                if best is None or v > best[0]:
                    best = (v, n, booster)
            v, n, booster = best
            score = pd.Series(booster.predict(xte.values), index=xte.index).reindex(idx)
            score.to_pickle(os.path.join(out, "pred_lgb_trees.pkl"))
            record(f"LightGBM, {n} trees", score, "model",
                   {"trees": n, "valid_IC": v,
                    "distinct_scores_per_day": float(score.groupby(level="datetime").nunique().median())})
            continue
        if m == "lgb_tuned":
            # one honest fix: slower learning rate, picked on the 2022-2023 validation IC only
            from qlib.contrib.model.gbdt import LGBModel
            vlabel = ds_raw.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_R).iloc[:, 0]
            best = None
            for lr in (0.05, 0.02, 0.01):
                t = time.time()
                mdl = LGBModel(**{**LGB, "learning_rate": lr}, early_stopping_rounds=100)
                mdl.fit(ds_raw)
                v = ic_stats(mdl.predict(ds_raw, segment="valid").reindex(vlabel.index), vlabel)["IC"]
                it = mdl.model.best_iteration
                log(f"  lr {lr}: best iteration {it}, validation IC {v:+.4f} ({time.time() - t:.0f}s)")
                if best is None or v > best[0]:
                    best = (v, lr, it, mdl)
            v, lr, it, mdl = best
            score = mdl.predict(ds_raw, segment="test").reindex(idx)
            score.to_pickle(os.path.join(out, "pred_lgb_tuned.pkl"))
            record(f"LightGBM, lr {lr}", score, "model",
                   {"learning_rate": lr, "best_iteration": it, "valid_IC": v})
            continue
        try:
            name, model, which = make_model(m)
        except ImportError as e:
            log(f"skip {m}: {e}")
            continue
        ds = ds_raw if which == "raw" else ds_norm
        t = time.time()
        model.fit(ds)
        score = model.predict(ds, segment="test").reindex(idx)
        fit_s = time.time() - t
        score.to_pickle(os.path.join(out, f"pred_{m}.pkl"))
        record(name, score, "model", {"fit_seconds": fit_s})

    pad = pd.Timestamp(cfg["test"][0]) - pd.Timedelta(days=400)
    for name, expr in BASELINES.items():
        log(f"rule {name}: {expr}")
        f = D.features(D.instruments(cfg["market"]), [expr], start_time=pad, end_time=cfg["test"][1])
        s = f.iloc[:, 0].swaplevel().sort_index().reindex(idx)
        record(name, s, "rule", {"expression": expr})

    rnd = []
    for seed in range(a.seeds):
        s = pd.Series(np.random.default_rng(seed).standard_normal(len(idx)), index=idx)
        net, _ = run_backtest(s, cfg)
        rnd.append((perf(net)["cagr"], seed, net))
        log(f"random seed {seed}: CAGR {rnd[-1][0]:+.1%}")
    rnd.sort(key=lambda x: x[0])
    mid = rnd[len(rnd) // 2]
    results["Random (median seed)"] = {"kind": "random", **perf(mid[2]), "seed": mid[1],
                                       "all_seed_cagr": [x[0] for x in rnd]}
    curves["Random (median seed)"] = mid[2]

    px = D.features([cfg["bench"]] + cfg["extra"], ["$close"], start_time=pad,
                    end_time=cfg["test"][1])["$close"].unstack(0)
    bret = px.pct_change().reindex(curves[next(iter(curves))].index)
    for b in [cfg["bench"]] + cfg["extra"]:
        results[f"{b} (buy and hold)"] = {"kind": "index", **perf(bret[b])}

    table = pd.DataFrame(results).T
    table.to_csv(os.path.join(out, "race.csv"))
    print()
    print(f"{'test 2024-01-02 to 2026-11-12, after costs':<34}{'IC':>8}{'per year':>10}{'Sharpe':>8}{'max drawdown':>14}")
    for k, r in table.iterrows():
        ic = f"{r['IC']:+.4f}" if pd.notna(r.get("IC")) else ""
        print(f"{k:<34}{ic:>8}{r['cagr']:>+10.1%}{r['sharpe']:>8.2f}{r['max_drawdown']:>14.1%}")
    print(flush=True)
    with open(os.path.join(out, "race.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)

    # growth of $1: models first, then the strongest rule, random, the benchmark (max 8 series)
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2, labelsize=10, length=0)
    order = [k for k in curves if results[k]["kind"] == "model"]
    rules = sorted([k for k in curves if results[k]["kind"] == "rule"],
                   key=lambda k: -results[k]["cagr"])
    order += rules[:2] + ["Random (median seed)"]
    lines = [(k, (1 + curves[k]).cumprod()) for k in order[:7]]
    lines.append((cfg["bench"], (1 + bret[cfg["bench"]].fillna(0)).cumprod()))
    for i, (k, w) in enumerate(lines):
        ax.plot(w.index, w.values, color=SERIES[i], linewidth=2, label=k)
        ax.annotate(f"{k}  {w.iloc[-1] - 1:+.0%}", (w.index[-1], w.iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", color=INK, fontsize=9, va="center")
    ax.set_title(f"{cfg['market'].upper()}: growth of $1 after costs, {cfg['test'][0]} to {cfg['test'][1]}",
                 color=INK, fontsize=15, loc="left", pad=14)
    ax.legend(frameon=False, labelcolor=INK2, loc="upper left", fontsize=9)
    ax.margins(x=0.22)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "race_growth.png"), facecolor=SURFACE)
    log("saved", out)


if __name__ == "__main__":
    main()
