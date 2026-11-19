"""Summarise one qrun experiment: signal quality, Qlib's own excess-return table, and plain
after-cost performance of the TopkDropout portfolio against index ETFs.

usage: python report.py --exp us_lgb_recent --provider ~/.qlib/qlib_data/us_alpaca --region us
                        --bench SPY,RSP,QQQ --mlruns runs/mlruns --out runs/us_lgb_recent
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import qlib  # noqa: E402
from qlib.data import D  # noqa: E402
from qlib.workflow import R  # noqa: E402

# validated default palette, dark mode, fixed categorical order
SURFACE, INK, INK2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#383835"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]


def perf(r):
    """Daily simple returns -> headline stats (rf = 0)."""
    r = r.dropna()
    wealth = (1 + r).cumprod()
    yrs = len(r) / 252
    dd = wealth / wealth.cummax() - 1
    return {"total_return": float(wealth.iloc[-1] - 1),
            "cagr": float(wealth.iloc[-1] ** (1 / yrs) - 1),
            "vol": float(r.std() * np.sqrt(252)),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)),
            "max_drawdown": float(dd.min())}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2, labelsize=10, length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--region", default="us")
    ap.add_argument("--bench", default="SPY")
    ap.add_argument("--mlruns", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="Qlib LightGBM, top 50")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    qlib.init(provider_uri=os.path.expanduser(a.provider), region=a.region,
              exp_manager={"class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
                           "kwargs": {"uri": "file:" + os.path.abspath(a.mlruns),
                                      "default_exp_name": "Experiment"}})
    exp = R.get_exp(experiment_name=a.exp, create=False)
    recs = [r for r in exp.list_recorders().values() if r.status == "FINISHED"]
    rec = sorted(recs, key=lambda r: r.info["start_time"])[-1]

    metrics = rec.list_metrics()
    rep = rec.load_object("portfolio_analysis/report_normal_1day.pkl")
    ic = rec.load_object("sig_analysis/ic.pkl")
    ric = rec.load_object("sig_analysis/ric.pkl")

    net = rep["return"] - rep["cost"]
    benches = [b for b in a.bench.split(",") if b]
    px = D.features(benches, ["$close"], start_time=rep.index.min() - pd.Timedelta(days=10),
                    end_time=rep.index.max())["$close"].unstack(0)
    bret = px.pct_change().reindex(rep.index)

    summary = {
        "experiment": a.exp, "recorder": rec.id, "test_start": str(rep.index.min().date()),
        "test_end": str(rep.index.max().date()), "days": int(len(rep)),
        "IC": float(ic.mean()), "ICIR": float(ic.mean() / ic.std()),
        "RankIC": float(ric.mean()), "RankICIR": float(ric.mean() / ric.std()),
        "IC_t": float(ic.mean() / ic.std() * np.sqrt(len(ic))),
        "qlib_metrics": {k: float(v) for k, v in metrics.items()},
        "avg_daily_turnover": float(rep["turnover"].mean()),
        "cost_drag_per_year": float(rep["cost"].mean() * 252),
        "strategy_net": perf(net),
        "strategy_gross": perf(rep["return"]),
        "benchmarks": {b: perf(bret[b]) for b in benches},
    }
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # chart 1: growth of $1, one axis, strategy first in the fixed order
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150, facecolor=SURFACE)
    style(ax)
    lines = [(a.label + " (after costs)", (1 + net).cumprod())] + \
            [(b, (1 + bret[b].fillna(0)).cumprod()) for b in benches]
    for i, (name, w) in enumerate(lines):
        ax.plot(w.index, w.values, color=SERIES[i], linewidth=2, label=name)
        ax.annotate(f"{name.split(' (')[0]}  {w.iloc[-1] - 1:+.0%}", (w.index[-1], w.iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", color=INK, fontsize=10, va="center")
    ax.set_title(f"Growth of $1, {summary['test_start']} to {summary['test_end']}",
                 color=INK, fontsize=15, loc="left", pad=14)
    ax.legend(frameon=False, labelcolor=INK2, loc="upper left")
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "growth.png"), facecolor=SURFACE)
    plt.close(fig)

    # chart 2: monthly mean daily IC, single series
    m = ic.groupby(pd.Grouper(freq="ME")).mean()
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150, facecolor=SURFACE)
    style(ax)
    ax.bar(m.index, m.values, width=20, color=SERIES[0])
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_title(f"Monthly average IC (mean daily IC {summary['IC']:.4f}, t = {summary['IC_t']:.1f})",
                 color=INK, fontsize=15, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "ic_monthly.png"), facecolor=SURFACE)
    plt.close(fig)

    print(json.dumps({k: summary[k] for k in ("IC", "ICIR", "RankIC", "RankICIR", "IC_t")}, indent=1))
    print("strategy net :", {k: round(v, 4) for k, v in summary["strategy_net"].items()})
    for b in benches:
        print(f"{b:<13}:", {k: round(v, 4) for k, v in summary["benchmarks"][b].items()})
    print("qlib excess (with cost):",
          {k.split(".")[-1]: round(v, 4) for k, v in summary["qlib_metrics"].items()
           if "excess_return_with_cost" in k})


if __name__ == "__main__":
    main()
