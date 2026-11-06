"""Download the daily bars that the agent test and its scoring scripts read. Free, from Yahoo Finance through yfinance.

    python a0_data.py

Writes data/<ticker>.csv for the 12 names, from 1 Jun 2021, with the same download and columns as kr.bars, which is what
a1_forecasts.py used. Existing files are kept. Prices are adjusted, so results scored on a later download can differ a little from the filmed ones.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
TICKERS = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMZN", "JPM", "XOM", "UNH", "WMT", "TSLA", "KO"]
START = "2021-06-01"


def bars(ticker):
    d = yf.download(ticker, start=START, interval="1d", progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    d.index = pd.to_datetime(d.index)
    if d.index.tz is not None:
        d.index = d.index.tz_convert("UTC").tz_localize(None)
    d["amount"] = d["volume"] * d[["open", "high", "low", "close"]].mean(axis=1)
    d.index.name = "timestamps"
    return d


def main():
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    for t in TICKERS:
        f = out / f"{t}.csv"
        if f.exists():
            print(t, "already there")
            continue
        d = bars(t)
        d.to_csv(f)
        print(t, len(d), "rows", d.index[0].date(), "to", d.index[-1].date(), flush=True)


if __name__ == "__main__":
    main()
