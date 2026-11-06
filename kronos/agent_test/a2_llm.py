"""LLM arms from PLAN.md on DeepSeek Flash. Resumable, stops at the budget. Key read from DEEPSEEK_API_KEY, never printed. Needs data/, see a0_data.py.

    python a2_llm.py --test 3      # 3 calls, prints the prompt and the reply, spends about nothing
    python a2_llm.py               # everything
"""
import argparse
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
MODEL, BUDGET, WORKERS = "deepseek-flash", 1.60, 16
PRICE_IN, PRICE_OUT = 0.14, 0.28          # USD per million tokens, an older price table. See PLAN.md for the real spend
OUT = HERE / "llm_calls.jsonl"

SYSTEM = ("You manage one position in one anonymous asset for the next 5 trading sessions. Names and dates are hidden and prices "
          "are rebased to 100. Decide from the data given. Reply with JSON only: "
          '{"position": "long" | "flat" | "short", "reason": "one short sentence"}')


def api_key(name):
    """Read an API key from the environment. Never printed."""
    key = os.environ.get(name, "").strip()
    if not key:
        raise SystemExit(f"{name} is not set. Set it in the environment before running this script, see kronos/.env.example.")
    return key


def features(d, day):
    i = d.index.get_loc(day)
    h = d.iloc[: i + 1]
    c = h.close
    ret = lambda n: (c.iloc[-1] / c.iloc[-1 - n] - 1) * 100
    delta = c.diff().iloc[-14:]
    gain, loss = delta.clip(lower=0).mean(), -delta.clip(upper=0).mean()
    rsi = 100 - 100 / (1 + gain / loss) if loss > 0 else 100.0
    last = h.iloc[-20:]
    base = last.close.iloc[0]
    vol_rel = last.volume / h.volume.iloc[-20:].mean()
    rows = "\n".join(f"{o / base * 100:.2f},{hi / base * 100:.2f},{lo / base * 100:.2f},{cl / base * 100:.2f},{v:.2f}"
                     for o, hi, lo, cl, v in zip(last.open, last.high, last.low, last.close, vol_rel))
    return (f"Returns: 5 sessions {ret(5):+.1f}%, 20 sessions {ret(20):+.1f}%, 60 sessions {ret(60):+.1f}%, 120 sessions {ret(120):+.1f}%\n"
            f"Daily volatility over 20 sessions: {np.log(c).diff().iloc[-20:].std() * 100:.2f}%\n"
            f"Price against its 50 session average: {(c.iloc[-1] / c.iloc[-50:].mean() - 1) * 100:+.1f}%\n"
            f"RSI 14: {rsi:.0f}\n\n"
            f"Last 20 sessions, oldest first (open,high,low,close,volume relative to its 20 session average):\n{rows}")


def forecast_block(r):
    return ("\n\nForecast from a candlestick foundation model, 30 sampled paths for the next 5 sessions:\n"
            f"Share of paths ending higher: {r.up_share * 100:.0f}%\n"
            f"Median move: {r.p50:+.1f}%\n"
            f"10th to 90th percentile of moves: {r.p10:+.1f}% to {r.p90:+.1f}%\n"
            f"Predicted volatility against the last 20 sessions: {r.vol_ratio:.2f}x")


def plan():
    f = pd.read_csv(HERE / "forecasts.csv")
    rng = np.random.default_rng(20261105)
    jobs = []
    bars = {t: pd.read_csv(HERE / "data" / f"{t}.csv", parse_dates=["timestamps"], index_col="timestamps") for t in f.ticker.unique()}
    for period, g in f.groupby("period"):
        g = g.reset_index(drop=True)
        for _, r in g.iterrows():
            body = features(bars[r.ticker], pd.Timestamp(r.date))
            arms = {"none": (body, None)}
            arms["kronos"] = (body + forecast_block(r), float(r.up_share))
            if period == "test":
                others = g[(g.ticker != r.ticker) & (g.date != r.date)]
                o = others.iloc[int(rng.integers(len(others)))]
                arms["placebo"] = (body + forecast_block(o), float(o.up_share))
            for arm, (prompt, shown) in arms.items():
                jobs.append({"id": f"{r.ticker}|{r.date}|{arm}", "ticker": r.ticker, "date": r.date, "period": period, "arm": arm,
                             "shown_up_share": shown, "prompt": prompt})
    return jobs


def call(key, prompt):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 200, "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=120).read())
        except Exception as e:
            if attempt == 5:
                raise RuntimeError(type(e).__name__) from None
            time.sleep(2**attempt)


def parse(text):
    try:
        p = str(json.loads(text).get("position", "")).lower()
    except Exception:
        p = ""
    return p if p in ("long", "flat", "short") else "flat", p in ("long", "flat", "short")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--test", type=int, default=0)
    o = a.parse_args()
    key = api_key("DEEPSEEK_API_KEY")
    jobs = plan()
    print(len(jobs), "calls planned:", pd.Series([j["arm"] + "/" + j["period"] for j in jobs]).value_counts().to_dict(), flush=True)
    done = set()
    spent = 0.0
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            x = json.loads(line)
            done.add(x["id"])
            spent += x["cost"]
    todo = [j for j in jobs if j["id"] not in done]
    if o.test:
        todo = todo[:o.test]
        print("--- system\n" + SYSTEM + "\n--- user\n" + todo[1]["prompt"], flush=True)
    lock = threading.Lock()
    state = {"spent": spent, "n": 0, "bad": 0}

    def work(j):
        if state["spent"] >= BUDGET:
            return
        r = call(key, j["prompt"])
        u = r.get("usage", {})
        text = r["choices"][0]["message"].get("content") or ""
        pos, ok = parse(text)
        cost = (u.get("prompt_tokens", 0) * PRICE_IN + u.get("completion_tokens", 0) * PRICE_OUT) / 1e6
        rec = {k: j[k] for k in ("id", "ticker", "date", "period", "arm", "shown_up_share")}
        rec.update(position=pos, parsed=ok, reply=text, tokens_in=u.get("prompt_tokens"), tokens_out=u.get("completion_tokens"),
                   cached=u.get("prompt_cache_hit_tokens"), cost=cost)
        with lock:
            state["spent"] += cost
            state["n"] += 1
            state["bad"] += not ok
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            if state["n"] % 200 == 0 or o.test:
                print(f"{state['n']} done, ${state['spent']:.3f} spent (upper bound, ignores cache discount), {state['bad']} unparsed", flush=True)
                if o.test:
                    print(rec["arm"], text, u, flush=True)

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(work, todo))
    print(f"finished: {state['n']} new calls, ${state['spent']:.3f} total upper bound, {state['bad']} unparsed", flush=True)
    if state["spent"] >= BUDGET:
        print("STOPPED AT BUDGET", flush=True)


if __name__ == "__main__":
    main()
