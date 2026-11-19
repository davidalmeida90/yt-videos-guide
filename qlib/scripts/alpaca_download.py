"""Download daily US bars from Alpaca's market data API for every S&P 500 member since 2016.

Universe: fja05680/sp500 `sp500_ticker_start_end.csv` (point-in-time membership spells,
includes names that left the index). Prices: split and dividend adjusted (adjustment=all),
SIP feed. Alpaca's history starts in 2016.

Symbols that were renamed or delisted are requested with `asof` = the last day of their
spell, so Alpaca maps them to the company that carried the ticker at that time.

Keys come from the environment variables ALPACA_API_KEY and ALPACA_SECRET_KEY (free Alpaca account) and are never printed.
Output: us_data/raw/<SYMBOL>.csv with date, open, high, low, close, volume, vwap, factor.
"""
import os
import sys
import time
import json
import datetime as dt

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "us_data", "raw")
os.makedirs(OUT, exist_ok=True)


URL = "https://data.alpaca.markets/v2/stocks/bars"
START = "2015-12-01T00:00:00Z"
END = "2026-11-18T00:00:00Z"
EXTRA = ["SPY", "QQQ", "RSP"]          # benchmarks: cap weighted, Nasdaq 100, equal weight
SESSION = requests.Session()
SESSION.headers.update({
    "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"],
    "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"],
    "User-Agent": "qlib-test-research/1.0",
})


def get_bars(symbols, asof=None):
    """All daily bars for a list of symbols, following next_page_token."""
    params = {"symbols": ",".join(symbols), "timeframe": "1Day", "start": START, "end": END,
              "adjustment": "all", "feed": "sip", "limit": 10000, "sort": "asc"}
    if asof:
        params["asof"] = asof
    out = {}
    while True:
        r = None
        for attempt in range(8):
            try:
                r = SESSION.get(URL, params=params, timeout=90)
            except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.Timeout) as e:
                print(f"    network drop ({type(e).__name__}), retry {attempt + 1}", flush=True)
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            break
        if r is None:
            raise RuntimeError("network down after 8 retries")
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        js = r.json()
        for sym, bars in (js.get("bars") or {}).items():
            out.setdefault(sym, []).extend(bars)
        tok = js.get("next_page_token")
        time.sleep(0.35)                 # free plan: 200 requests per minute
        if not tok:
            return out
        params["page_token"] = tok


def save(sym, bars, out_dir=None):
    df = pd.DataFrame(bars)
    df["date"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York").dt.date
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap"})
    df = df[["date", "open", "high", "low", "close", "volume", "vwap"]].drop_duplicates("date")
    df["factor"] = 1.0                   # prices already adjusted
    df["symbol"] = sym
    df.to_csv(os.path.join(out_dir or OUT, f"{sym}.csv"), index=False)
    return len(df), str(df["date"].min()), str(df["date"].max())


def demo(only, out_dir):
    """Short version for the video: a few symbols into a separate folder, delisted ones with asof."""
    os.makedirs(out_dir, exist_ok=True)
    spells = pd.read_csv(os.path.join(ROOT, "us_data", "sp500_ticker_start_end.csv"), parse_dates=["end_date"])
    for sym in only:
        ends = spells[spells.ticker == sym].end_date
        asof = None if ends.isna().any() or ends.empty else (ends.max() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        bars = get_bars([sym], asof=asof)
        n, first, last = save(sym, bars[sym], out_dir)
        print(f"  {sym:<5} {n:>5} daily bars  {first} to {last}" + (f"   (asof {asof})" if asof else ""), flush=True)
    print(f"saved {len(only)} files to {out_dir}")


def main():
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
        out_dir = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(ROOT, "us_data", "demo")
        return demo(only, out_dir)
    spells = pd.read_csv(os.path.join(ROOT, "us_data", "sp500_ticker_start_end.csv"),
                         parse_dates=["start_date", "end_date"])
    spells = spells[spells.end_date.isna() | (spells.end_date >= "2016-01-01")]
    open_syms = sorted(set(spells[spells.end_date.isna()].ticker) | set(EXTRA))
    closed = (spells[spells.end_date.notna()].groupby("ticker").end_date.max()
              .reset_index().query("ticker not in @open_syms"))

    log = {}
    done = {f[:-4] for f in os.listdir(OUT) if f.endswith(".csv")}

    todo = [s for s in open_syms if s not in done]
    print(f"open spells: {len(open_syms)} symbols ({len(todo)} to fetch)", flush=True)
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        bars = get_bars(batch)
        for s in batch:
            log[s] = save(s, bars[s]) if bars.get(s) else "missing"
        print(f"  {i + len(batch)}/{len(todo)}", flush=True)

    rows = [r for r in closed.itertuples() if r.ticker not in done]
    print(f"closed spells: {len(closed)} symbols ({len(rows)} to fetch, one request each with asof)", flush=True)
    for j, r in enumerate(rows, 1):
        asof = (r.end_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            bars = get_bars([r.ticker], asof=asof)
            log[r.ticker] = save(r.ticker, bars[r.ticker]) if bars.get(r.ticker) else "missing"
        except RuntimeError as e:
            log[r.ticker] = f"error {e}"
        if j % 25 == 0:
            print(f"  {j}/{len(rows)}", flush=True)

    with open(os.path.join(ROOT, "us_data", "download_log.json"), "w") as f:
        json.dump({k: v for k, v in log.items()}, f, indent=1, default=str)
    miss = [k for k, v in log.items() if not isinstance(v, tuple)]
    print(f"saved {len(log) - len(miss)}, missing {len(miss)}: {miss[:40]}")


if __name__ == "__main__":
    main()
