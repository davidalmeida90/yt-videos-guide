"""5-minute bars from Alpaca (SIP feed, split and dividend adjusted), regular session only.
Universe: symbols.txt, the 200 S&P 500 stocks with the highest median dollar volume in 2021 among the names that were
members from June 2021 on (chosen on 2021 data only), plus SPY. One name, TIF, returns no bars and is skipped.
Keys come from the environment (ALPACA_API_KEY, ALPACA_SECRET_KEY), from a free Alpaca account; they are never printed.
    py -3.11 download_bars.py 200     ->  data/intraday/<SYM>.parquet (ts New York time, open, high, low, close, volume, vwap)
A symbol already on disk is skipped, so the script can be stopped and run again.
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://data.alpaca.markets/v2/stocks/bars"
_S = None


def session():
    global _S
    if _S is None:
        key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise SystemExit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in the environment first (see .env.example).")
        _S = requests.Session()
        _S.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret, "User-Agent": "kronos-vol-research/1.0"})
    return _S


def fetch(sym, start="2021-06-01T00:00:00Z", end="2026-11-18T00:00:00Z"):
    """Regular-session 5-minute bars of one symbol, as a DataFrame (empty when Alpaca has none)."""
    params = {"symbols": sym, "timeframe": "5Min", "start": start, "end": end, "adjustment": "all", "feed": "sip",
              "limit": 10000, "sort": "asc"}
    bars = []
    while True:
        for attempt in range(8):
            try:
                r = session().get(URL, params=params, timeout=90)
            except requests.exceptions.RequestException:
                time.sleep(10 * (attempt + 1)); continue
            if r.status_code == 429:
                time.sleep(15 * (attempt + 1)); continue
            break
        if r.status_code != 200:
            raise RuntimeError(f"{sym}: HTTP {r.status_code} {r.text[:120]}")
        js = r.json()
        bars += (js.get("bars") or {}).get(sym, [])
        time.sleep(0.32)
        if not js.get("next_page_token"):
            break
        params["page_token"] = js["next_page_token"]
    cols = ["ts", "open", "high", "low", "close", "volume", "vwap"]
    if not bars:
        return pd.DataFrame(columns=cols)
    d = pd.DataFrame(bars)
    d["ts"] = pd.to_datetime(d["t"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    d = d.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap"})
    tm = d.ts.dt.hour * 60 + d.ts.dt.minute
    d = d[(tm >= 570) & (tm < 960)]                       # 09:30 to 15:55 bar starts
    return d[cols].reset_index(drop=True)


if __name__ == "__main__":
    import evallib as E

    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=200, help="the first N names of symbols.txt (plus SPY)")
    ap.add_argument("--start", default="2021-06-01T00:00:00Z")
    ap.add_argument("--end", default="2026-11-18T00:00:00Z")
    a = ap.parse_args()
    names = [s for s in (Path(__file__).resolve().parent / "symbols.txt").read_text().split() if s != "SPY"]
    syms = names[:a.n] + ["SPY"]
    out = E.DATA / "intraday"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(syms)} symbols -> {out}", flush=True)
    t0 = time.time()
    for i, s in enumerate(syms, 1):
        if (out / f"{s}.parquet").exists():
            continue
        d = fetch(s, a.start, a.end)
        if d.empty:
            print(f"  {i}/{len(syms)} {s}: no bars, skipped", flush=True); continue
        d.to_parquet(out / f"{s}.parquet", index=False)
        print(f"  {i}/{len(syms)} {s}: {len(d):,} bars  ({time.time() - t0:.0f}s)", flush=True)
    print("done")
