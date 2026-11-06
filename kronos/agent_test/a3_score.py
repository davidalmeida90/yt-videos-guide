"""Score every arm in PLAN.md with 1 ruler. Writes results.csv and halfyears.csv, prints the tables. Needs data/, see a0_data.py.

    python a3_score.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
COST = 0.0005
PER_YEAR = 252 / 5
POS = {"long": 1, "flat": 0, "short": -1}


def load():
    f = pd.read_csv(HERE / "forecasts.csv")
    bars = {t: pd.read_csv(HERE / "data" / f"{t}.csv", parse_dates=["timestamps"], index_col="timestamps") for t in f.ticker.unique()}
    rows = []
    for _, r in f.iterrows():
        d = bars[r.ticker]
        i = d.index.get_loc(pd.Timestamp(r.date))
        if i + 6 >= len(d):
            continue
        c = d.close
        feats = {"mom5": c.iloc[i] / c.iloc[i - 5] - 1, "mom20": c.iloc[i] / c.iloc[i - 20] - 1, "mom60": c.iloc[i] / c.iloc[i - 60] - 1,
                 "mom120": c.iloc[i] / c.iloc[i - 120] - 1, "ma50": c.iloc[i] / c.iloc[i - 49 : i + 1].mean() - 1}
        rows.append({"ticker": r.ticker, "date": r.date, "period": r.period, "up_share": r.up_share,
                     "ret": d.open.iloc[i + 6] / d.open.iloc[i + 1] - 1, **feats})
    return pd.DataFrame(rows)


def book(df, pos):
    """Weekly equal weight return of a position panel. pos: Series indexed like df."""
    x = df.assign(pos=pos.values).sort_values(["ticker", "date"])
    x["trade"] = x.groupby("ticker").pos.diff().abs().fillna(x.pos.abs())
    x["net"] = x.pos * x.ret - COST * x.trade
    return x.groupby("date").net.mean()


def stats(w):
    eq = (1 + w).cumprod()
    return {"total_%": (eq.iloc[-1] - 1) * 100, "sharpe": w.mean() / w.std() * np.sqrt(PER_YEAR), "max_dd_%": (eq / eq.cummax() - 1).min() * 100,
            "weeks": len(w)}


def hit(df, pos):
    x = df.assign(pos=pos.values)
    x = x[x.pos != 0]
    return ((x.pos > 0) == (x.ret > 0)).mean() * 100, len(x)


def main():
    df = load()
    calls = pd.DataFrame([json.loads(s) for s in (HERE / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()])
    print("calls", len(calls), "unparsed", int((~calls.parsed).sum()), "cost upper bound $", round(calls.cost.sum(), 3), "tokens in/out",
          int(calls.tokens_in.sum()), int(calls.tokens_out.sum()), "cached", int(calls.cached.fillna(0).sum()))
    results, halves, books = [], [], {}
    for period, g in df.groupby("period"):
        g = g.reset_index(drop=True)
        arms = {"Buy and hold": pd.Series(1, index=g.index),
                "Kronos rule": pd.Series(np.sign(g.up_share - 0.5)).astype(int)}
        sweep = {k: pd.Series(np.sign(g[k])).astype(int) for k in ("mom5", "mom20", "mom60", "mom120", "ma50")}
        best = max(sweep, key=lambda k: stats(book(g, sweep[k]))["sharpe"])
        arms[f"Floor: {best}"] = sweep[best]
        for arm, label in (("none", "LLM, no forecast"), ("kronos", "LLM + Kronos"), ("placebo", "LLM + placebo")):
            c = calls[(calls.period == period) & (calls.arm == arm)]
            if len(c):
                m = g.merge(c[["ticker", "date", "position"]], on=["ticker", "date"], how="left")
                arms[label] = m.position.map(POS).fillna(0).astype(int)
        for label, pos in arms.items():
            w = book(g, pos)
            books[(period, label)] = w
            h, n = hit(g, pos)
            mix = pos.value_counts(normalize=True)
            results.append({"period": period, "arm": label, **stats(w), "hit_%": h, "calls": n, "long_%": mix.get(1, 0) * 100,
                            "flat_%": mix.get(0, 0) * 100, "short_%": mix.get(-1, 0) * 100})
            hy = w.groupby(lambda s: f"{s[:4]}H{1 if int(s[5:7]) <= 6 else 2}").sum() * 100
            halves.append({"period": period, "arm": label, **hy.round(1).to_dict()})
    res = pd.DataFrame(results).round(2)
    res.to_csv(HERE / "results.csv", index=False)
    print("\n" + res.to_string(index=False))
    hv = pd.DataFrame(halves)
    hv.to_csv(HERE / "halfyears.csv", index=False)
    print("\nSum of weekly returns by half year, %\n" + hv.to_string(index=False))

    print("\nPaired tests on weekly returns")
    for period, a, b in (("test", "LLM + Kronos", "LLM, no forecast"), ("test", "LLM + Kronos", "LLM + placebo"), ("test", "LLM + placebo", "LLM, no forecast"),
                         ("test", "LLM, no forecast", "Buy and hold"), ("test", "LLM + Kronos", "Buy and hold"), ("test", "Kronos rule", "Buy and hold"),
                         ("control", "LLM + Kronos", "LLM, no forecast"), ("control", "Kronos rule", "Buy and hold")):
        if (period, a) not in books or (period, b) not in books:
            continue
        d = books[(period, a)] - books[(period, b)]
        t = d.mean() / d.std() * np.sqrt(len(d))
        print(f"{period:8} {a:18} minus {b:18} {d.mean() * PER_YEAR * 100:+7.2f}% a year  t {t:+.2f}  n {len(d)}")

    print("\nHow often the LLM's call matches the direction of the forecast it was shown")
    for arm in ("kronos", "placebo"):
        for period in ("test", "control"):
            c = calls[(calls.arm == arm) & (calls.period == period) & calls.shown_up_share.notna() & (calls.shown_up_share != 0.5)]
            if len(c):
                fc = np.where(c.shown_up_share > 0.5, "long", "short")
                print(f"{period:8} {arm:8} agrees {(c.position.values == fc).mean() * 100:.1f}%, against {((c.position.values != fc) & (c.position.values != 'flat')).mean() * 100:.1f}%, "
                      f"flat {(c.position == 'flat').mean() * 100:.1f}%  n {len(c)}")
    t = calls[calls.period == "test"].pivot_table(index=["ticker", "date"], columns="arm", values="position", aggfunc="first")
    if {"none", "kronos"} <= set(t.columns):
        print(f"\nCalls that changed once the forecast was added: real {(t.none != t.kronos).mean() * 100:.1f}%, placebo {(t.none != t.placebo).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
