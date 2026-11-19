"""Qlib's own report charts for a finished qrun experiment (qlib.contrib.report), saved as offline HTML pages.

    python qlib_charts.py --exp us_lgb_recent --provider ~/.qlib/qlib_data/us_alpaca --region us

Writes runs/qlib_charts/<exp>_groups.html (5 groups by score, best to worst, cumulative return)
and <exp>_report.html (Qlib's backtest report: cumulative return vs benchmark, drawdown, turnover).
"""
import argparse
import os

import pandas as pd
import qlib
from qlib.workflow import R
from qlib.contrib.report import analysis_model, analysis_position

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def page(figs, title, path):
    parts = [f"<h2 style='font-family:Segoe UI,Arial;margin:18px 24px'>{title}</h2>"]
    for i, f in enumerate(figs):
        f.update_layout(width=1800, height=760 if i == 0 else 520)
        parts.append(f.to_html(full_html=False, include_plotlyjs="inline" if i == 0 else False))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("<html><body style='margin:0;background:#fff'>" + "".join(parts) + "</body></html>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--region", default="us")
    ap.add_argument("--mlruns", default="mlruns")
    a = ap.parse_args()
    out = os.path.join(ROOT, "runs", "qlib_charts")
    os.makedirs(out, exist_ok=True)

    qlib.init(provider_uri=os.path.expanduser(a.provider), region=a.region,
              exp_manager={"class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
                           "kwargs": {"uri": "file:" + os.path.abspath(a.mlruns), "default_exp_name": "Experiment"}})
    exp = R.get_exp(experiment_name=a.exp, create=False)
    rec = sorted([r for r in exp.list_recorders().values() if r.status == "FINISHED"],
                 key=lambda r: r.info["start_time"])[-1]

    pred = rec.load_object("pred.pkl")
    label = rec.load_object("label.pkl")
    pred_label = pd.concat([pred["score"], label.iloc[:, 0].rename("label")], axis=1).dropna()
    groups = analysis_model.model_performance_graph(pred_label, graph_names=["group_return"], show_notebook=False)
    page(groups, f"{a.exp}: stocks split into 5 groups by the model's score, cumulative return",
         os.path.join(out, f"{a.exp}_groups.html"))

    report = rec.load_object("portfolio_analysis/report_normal_1day.pkl")
    figs = analysis_position.report_graph(report, show_notebook=False)
    page(figs, f"{a.exp}: Qlib backtest report", os.path.join(out, f"{a.exp}_report.html"))
    print(f"saved {a.exp}_groups.html and {a.exp}_report.html to {out}")


if __name__ == "__main__":
    main()
