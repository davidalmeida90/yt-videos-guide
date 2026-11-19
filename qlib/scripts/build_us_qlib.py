"""Turn the Alpaca CSVs into a Qlib dataset with a point-in-time S&P 500 universe.

1. dump_bin.py dump_all  -> ~/.qlib/qlib_data/us_alpaca (calendar, features, instruments/all.txt)
2. instruments/sp500.txt  from fja05680 membership spells, only symbols that have data,
   each spell clipped to the data range. Renamed tickers keep separate spells (FB, then META).
"""
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "us_data", "raw")
QDIR = os.path.join(os.path.expanduser("~"), ".qlib", "qlib_data", "us_alpaca")
DUMP = os.path.join(ROOT, "qlib_files", "scripts_dump_bin.py")


def main():
    subprocess.run([sys.executable, DUMP, "dump_all", "--data_path", RAW, "--qlib_dir", QDIR,
                    "--include_fields", "open,close,high,low,volume,vwap,factor",
                    "--date_field_name", "date", "--symbol_field_name", "symbol"], check=True)

    have = {}
    for f in os.listdir(RAW):
        if f.endswith(".csv"):
            d = pd.read_csv(os.path.join(RAW, f), usecols=["date"])
            have[f[:-4]] = (d.date.min(), d.date.max())

    sp = pd.read_csv(os.path.join(ROOT, "us_data", "sp500_ticker_start_end.csv"))
    sp["end_date"] = sp["end_date"].fillna("2099-12-31")
    rows, dropped = [], []
    for r in sp.itertuples():
        if r.end_date < "2016-01-01":
            continue
        if r.ticker not in have:
            dropped.append(r.ticker)
            continue
        lo, hi = have[r.ticker]
        s, e = max(r.start_date, lo), min(r.end_date, hi)
        if s <= e:
            rows.append((r.ticker.upper(), s, e))
    out = os.path.join(QDIR, "instruments", "sp500.txt")
    pd.DataFrame(rows).to_csv(out, sep="\t", header=False, index=False)
    print(f"sp500.txt: {len(rows)} spells, {len({r[0] for r in rows})} symbols; "
          f"no data for {len(dropped)}: {sorted(set(dropped))}")


if __name__ == "__main__":
    main()
