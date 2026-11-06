"""Did the TradingAgents team follow the forecast? Reads the scratch Glassbench's finished batches. Measures behaviour, because 12 cells cannot judge returns.

    python compare.py

For each model: rating per cell by arm, how often the rating changed against the plain run, how often its direction matched the forecast shown, and whether the forecast reached the final decision text.
GLASSBENCH_SCRATCH names the scratch folder, same default as scratch.py.
"""
import json
import os
import re
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
API = "http://127.0.0.1:8767"
RUNS = Path(os.environ.get("GLASSBENCH_SCRATCH") or Path(__file__).resolve().parents[4] / "glassbench-kronos-scratch") / "data" / "runs"
SCORE = {"Buy": 2, "Overweight": 1, "Hold": 0, "Underweight": -1, "Sell": -2}
MENTION = re.compile(r"foundation model|sampled paths|paths ending higher|candlestick (?:foundation )?model|model forecast|forecast model", re.I)


def get(path):
    return json.loads(urllib.request.urlopen(API + path, timeout=30).read())


def shown(run_id):
    """Share of paths up that the run was shown, and where the forecast shows up in what the agents wrote."""
    f = RUNS / run_id / "events.jsonl"
    up, agents = None, set()
    if not f.exists():
        return up, agents
    for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "variant.applied" and e.get("payload", {}).get("up_share") is not None:
            up = float(e["payload"]["up_share"])
        text = json.dumps(e.get("payload", {}))
        if e.get("type") not in ("variant.applied",) and e.get("agent") and MENTION.search(text):
            agents.add(e["agent"])
    return up, agents


def load(ids, model):
    rows = []
    fc = pd.read_csv(HERE.parent / "forecasts.csv").set_index(["ticker", "date"])
    for arm, bid in ids.items():
        d = get(f"/api/backtests/{bid}")
        for it in d["items"]:
            up, agents = shown(it["run_id"]) if it.get("run_id") else (None, set())
            if arm == "kronos" and up is None and (it["ticker"], it["date"]) in fc.index:
                up = float(fc.loc[(it["ticker"], it["date"]), "up_share"])
            rows.append({"model": model, "arm": arm, "ticker": it["ticker"], "date": it["date"], "rating": it.get("rating"), "score": SCORE.get(it.get("rating")), "up_share": up, "status": it.get("status"),
                         "cost": it.get("cost_usd"), "agents_citing": sorted(agents)})
        print(f"{model} {arm}: {d['status']}, {d['counts']}, ${d['spent_usd']:.3f}")
    return pd.DataFrame(rows)


def report(t):
    model = t.model.iloc[0]
    print(f"\n===== {model}")
    p = t.pivot_table(index=["date", "ticker"], columns="arm", values="rating", aggfunc="first")
    u = t.pivot_table(index=["date", "ticker"], columns="arm", values="up_share", aggfunc="first")
    show = p.copy()
    for arm in u.columns:
        show[f"{arm} paths up"] = (u[arm] * 100).round(0)
    print(show.to_string())
    print("\nRating mix by arm:", {a: g.rating.value_counts().to_dict() for a, g in t.groupby("arm")})
    print("Mean rating score (Buy +2 to Sell -2):", t.groupby("arm").score.mean().round(2).to_dict())
    plain = t[t.arm == "stock"].set_index(["date", "ticker"])
    for arm in [a for a in ("kronos", "fake") if a in set(t.arm)]:
        g = t[t.arm == arm].set_index(["date", "ticker"])
        both = g.join(plain[["rating", "score"]], rsuffix="_plain").dropna(subset=["score", "score_plain"])
        directional = both[both.up_share != 0.5]
        toward = ((directional.score - directional.score_plain) * (directional.up_share - 0.5)) > 0
        away = ((directional.score - directional.score_plain) * (directional.up_share - 0.5)) < 0
        match = ((directional.score > 0) & (directional.up_share > 0.5)) | ((directional.score < 0) & (directional.up_share < 0.5))
        against = ((directional.score > 0) & (directional.up_share < 0.5)) | ((directional.score < 0) & (directional.up_share > 0.5))
        citing = g.agents_citing.apply(len).gt(0).mean() * 100
        print(f"{arm:7}: rating changed from plain {(both.rating != both.rating_plain).mean() * 100:.0f}% | moved toward the forecast {toward.sum()}, away {away.sum()}, unchanged {(~toward & ~away).sum()} of {len(directional)} | rating matches forecast direction {match.mean() * 100:.0f}%, against {against.mean() * 100:.0f}%, neutral {100 - match.mean() * 100 - against.mean() * 100:.0f}% | runs where an agent cites it {citing:.0f}%")
    if "repeat" in set(t.arm):
        r = t[t.arm == "repeat"].set_index(["date", "ticker"]).join(plain[["rating"]], rsuffix="_plain").dropna(subset=["rating", "rating_plain"])
        print(f"NOISE, plain run against a second plain run: rating changed {(r.rating != r.rating_plain).mean() * 100:.0f}% of {len(r)} cases, mean size {(r.rating.map(SCORE) - r.rating_plain.map(SCORE)).abs().mean():.2f} tiers")
        for arm in ("kronos", "fake"):
            k = t[t.arm == arm].set_index(["date", "ticker"]).join(plain[["rating"]], rsuffix="_plain")
            print(f"       {arm} against plain, for comparison: mean size {(k.rating.map(SCORE) - k.rating_plain.map(SCORE)).abs().mean():.2f} tiers")
    from collections import Counter
    for arm in [a for a in ("kronos", "fake") if a in set(t.arm)]:
        print(f"{arm} citing by agent:", Counter(a for x in t[t.arm == arm].agents_citing for a in x).most_common(9))


def main():
    frames = [load(json.loads((HERE / "pair.json").read_text()), "DeepSeek")]
    f = HERE / "trio.json"
    if f.exists():
        frames.append(load(json.loads(f.read_text()), "Haiku"))
    out = pd.concat(frames)
    out.drop(columns="agents_citing").to_csv(HERE / "pair_results.csv", index=False)
    for t in frames:
        report(t)


if __name__ == "__main__":
    main()
