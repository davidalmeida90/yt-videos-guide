"""Look inside the qrun result: how many different scores the model gives each day,
and which stocks the top 50 ends up holding (a short take for the video).

usage: python inside.py [mlruns folder]
"""
import collections
import glob
import os
import sys

import pandas as pd


def main():
    mlruns = sys.argv[1] if len(sys.argv) > 1 else "mlruns"
    pred_path = sorted(glob.glob(os.path.join(mlruns, "*", "*", "artifacts", "pred.pkl")),
                       key=os.path.getmtime)[-1]
    run_dir = os.path.dirname(pred_path)
    # pickles here are Qlib's own run artifacts written by qrun on this machine, never downloaded
    pred = pd.read_pickle(pred_path)["score"]

    per_day = pred.groupby(level="datetime")
    print(f"stocks scored per day         : {int(per_day.size().median())}")
    print(f"different scores per day      : {int(per_day.nunique().median())} (median)")
    one_day = pred.xs(pred.index.get_level_values(0)[-1], level="datetime").sort_values(ascending=False)
    top_score = one_day.iloc[0]
    print(f"stocks tied at the top score  : {int((one_day == top_score).sum())} on {pred.index.get_level_values(0)[-1].date()}")

    top = pred.groupby(level="datetime", group_keys=False).apply(lambda x: x.nlargest(50))
    fav = top.index.get_level_values("instrument").value_counts().head(6)
    print("most days in its top 50       : " + ", ".join(f"{k} {v}" for k, v in fav.items()))

    pos = pd.read_pickle(os.path.join(run_dir, "portfolio_analysis", "positions_normal_1day.pkl"))
    held = collections.Counter(s[0] for p in pos.values() for s in p.get_stock_list())
    universe = collections.Counter(s[0] for s in one_day.index)
    n_held, n_uni = sum(held.values()), sum(universe.values())
    print()
    print("first letter   share of holdings   share of the S&P 500")
    for letter in "ABC":
        print(f"     {letter}              {held[letter] / n_held:6.1%}              {universe[letter] / n_uni:6.1%}")
    print("-" * 56)
    abc_h = sum(held[x] for x in "ABC") / n_held
    abc_u = sum(universe[x] for x in "ABC") / n_uni
    print(f"   A to C           {abc_h:6.1%}              {abc_u:6.1%}")


if __name__ == "__main__":
    main()
