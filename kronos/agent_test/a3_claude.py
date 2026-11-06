"""Score the Claude follow up with the same ruler as a3_score.py, next to DeepSeek. Uses the last parsed reply for each decision. Needs data/, see a0_data.py.

    python a3_claude.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import a3_score as s

HERE = Path(__file__).parent
PER_YEAR = 252 / 5


def load_calls(name):
    rows = [json.loads(line) for line in (HERE / name).read_text(encoding="utf-8").splitlines()]
    c = pd.DataFrame(rows)
    good = c[c.parsed].drop_duplicates("id", keep="last")
    print(f"{name}: {len(c)} calls, {len(good)} decisions parsed, cost ${c.cost.sum():.2f}, tokens in {int(c.tokens_in.sum()):,} out {int(c.tokens_out.sum()):,}")
    return good[good.period == "test"]


def main():
    df = s.load()
    g = df[df.period == "test"].reset_index(drop=True)
    bh = s.book(g, pd.Series(1, index=g.index))
    print(f"Buy and hold: {s.stats(bh)}\n")
    books = {}
    for label, f in (("DeepSeek", "llm_calls.jsonl"), ("Claude", "llm_calls_claude.jsonl")):
        calls = load_calls(f)
        print(f"{'arm':14}{'decisions':>10}{'total %':>9}{'Sharpe':>8}{'max DD %':>10}{'hit %':>7}{'long %':>8}{'flat %':>8}{'short %':>9}")
        for arm in ("none", "kronos", "placebo"):
            c = calls[calls.arm == arm]
            m = g.merge(c[["ticker", "date", "position"]], on=["ticker", "date"], how="left")
            pos = m.position.map(s.POS)
            missing = pos.isna().sum()
            pos = pos.fillna(0).astype(int)
            w = s.book(g, pos)
            books[(label, arm)] = w
            st = s.stats(w)
            h, n = s.hit(g, pos)
            mix = pos.value_counts(normalize=True)
            print(f"{arm:14}{len(c):10}{st['total_%']:9.1f}{st['sharpe']:8.2f}{st['max_dd_%']:10.1f}{h:7.1f}{mix.get(1, 0) * 100:8.1f}{mix.get(0, 0) * 100:8.1f}{mix.get(-1, 0) * 100:9.1f}" + (f"  ({missing} missing)" if missing else ""))
        print("Matches the direction of the forecast it was shown:")
        for arm in ("kronos", "placebo"):
            c = calls[(calls.arm == arm) & calls.shown_up_share.notna() & (calls.shown_up_share != 0.5)]
            fc = np.where(c.shown_up_share > 0.5, "long", "short")
            strong = c.shown_up_share.isin([0, 1])
            print(f"  {arm:8} agrees {(c.position.values == fc).mean() * 100:.1f}%, against {((c.position.values != fc) & (c.position.values != 'flat')).mean() * 100:.1f}%, flat {(c.position == 'flat').mean() * 100:.1f}%, n {len(c)}"
                  f" | when every path agrees: agrees {(c.position.values[strong] == fc[strong]).mean() * 100:.1f}%, n {int(strong.sum())}")
        t = calls.pivot_table(index=["ticker", "date"], columns="arm", values="position", aggfunc="first")
        print(f"Calls changed once a forecast was added: real {(t.none != t.kronos).mean() * 100:.1f}%, fake {(t.none != t.placebo).mean() * 100:.1f}%")
        for a, b in (("kronos", "none"), ("kronos", "placebo"), ("placebo", "none")):
            d = books[(label, a)] - books[(label, b)]
            print(f"  {a} minus {b}: {d.mean() * PER_YEAR * 100:+.2f}% a year, t {d.mean() / d.std() * np.sqrt(len(d)):+.2f}")
        print()
    d = books[("Claude", "none")] - books[("DeepSeek", "none")]
    print(f"Claude minus DeepSeek, no forecast: {d.mean() * PER_YEAR * 100:+.2f}% a year, t {d.mean() / d.std() * np.sqrt(len(d)):+.2f}")
    a = books[("Claude", "kronos")] - books[("DeepSeek", "kronos")]
    print(f"Claude minus DeepSeek, with Kronos: {a.mean() * PER_YEAR * 100:+.2f}% a year, t {a.mean() / a.std() * np.sqrt(len(a)):+.2f}")


if __name__ == "__main__":
    main()
