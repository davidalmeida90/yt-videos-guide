"""Use the published model, Kronos-Vol (https://huggingface.co/ddedei/Kronos-Vol): a forecast of next session's volatility.
Downloads the 3 seed files and config.json from the Hugging Face Hub (about 300 MB, cached after the first run), takes
the last 60 days of 5-minute bars of each name and of SPY, and prints the forecast for the session after the last
complete one. Runs on CPU too (forecasts then differ from the GPU ones by about 0.01 in log variance).
    py -3.11 use_model.py AAPL NVDA JPM                   bars from Alpaca (ALPACA_API_KEY, ALPACA_SECRET_KEY)
    py -3.11 use_model.py AAPL --bars data/intraday       bars already on disk (download_bars.py)
"""
import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from kronos_rv import KronosRV

ap = argparse.ArgumentParser()
ap.add_argument("symbols", nargs="+")
ap.add_argument("--bars", default=None, help="folder of <SYM>.parquet 5-minute bars; default: pull from Alpaca")
ap.add_argument("--repo", default="ddedei/Kronos-Vol")
ap.add_argument("--device", default=None)
a = ap.parse_args()
syms = [s.upper() for s in a.symbols]

now = pd.Timestamp.now(tz="America/New_York")
if a.bars:
    bars = {s: pd.read_parquet(Path(a.bars) / f"{s}.parquet") for s in syms + ["SPY"]}
else:
    from download_bars import fetch
    start = (now - pd.Timedelta(days=60)).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (now - pd.Timedelta(minutes=20)).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    bars = {s: fetch(s, start, end) for s in syms + ["SPY"]}
    if now.hour < 16:                                     # today's session is not complete yet: forecast from yesterday's close
        bars = {s: b[b.ts.dt.normalize() < now.tz_localize(None).normalize()] for s, b in bars.items()}
spy = bars.pop("SPY")

m = KronosRV.from_dir(snapshot_download(a.repo, allow_patterns=["config.json", "*.safetensors"]), device=a.device)
out = m.forecast(bars, spy)
if out.empty:
    raise SystemExit("no forecast: each name needs at least 256 bars and 22 full sessions")
out["as_of_close"] = pd.to_datetime(out.date).dt.date
out["next_vol_annual_%"] = (out.next_vol_annual.astype(float) * 100).round(1)
print(out[["symbol", "as_of_close", "next_log_rv", "next_vol_annual_%"]].round(4).to_string(index=False))
