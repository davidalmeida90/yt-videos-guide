"""Load csi300 from qlib exactly as finetune/qlib_data_preprocess.py does and save 1 table for inference.

    python p1_data.py csi300
Needs pyqlib, in its own venv, and the Qlib cn_data set in ~/.qlib/qlib_data/cn_data. Neither is included here.
Output: data_<universe>.csv with instrument, datetime, open, high, low, close, vol, amt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader

HERE = Path(__file__).parent
FEATURES = ["open", "high", "low", "close", "vol", "amt"]
FIELDS = ["open", "close", "high", "low", "volume", "vwap"]
BEGIN, LOOKBACK = "2024-01-01", 90


def main(universe):
    qlib.init(provider_uri=str(Path.home() / ".qlib" / "qlib_data" / "cn_data"), region=REG_CN)
    cal = D.calendar()
    print("calendar ends", cal[-1])
    start = cal[max(cal.searchsorted(pd.Timestamp(BEGIN)) - LOOKBACK, 0)]
    df = QlibDataLoader(config=["$" + f for f in FIELDS]).load(universe, start, cal[-1])
    df = df.stack().unstack(level=1)
    out = {}
    for symbol in df.columns:
        s = df[symbol].reset_index().rename(columns={"level_1": "field"})
        s = pd.pivot(s, index="datetime", columns="field", values=symbol)
        s = s.rename(columns={f"${f}": f for f in FIELDS})
        s["vol"] = s["volume"]
        s["amt"] = (s["open"] + s["high"] + s["low"] + s["close"]) / 4 * s["vol"]
        s = s[FEATURES].dropna()
        if len(s) >= LOOKBACK + 10 + 1:
            out[symbol] = s.astype(np.float32)
    days = sorted({d for s in out.values() for d in s.index})
    print(f"{universe}: {len(out)} symbols, {sum(len(s) for s in out.values())} rows, {days[0].date()} to {days[-1].date()}")
    # CSV, because this venv (pandas 3) and the Kronos venv (pandas 2.2) cannot share pickled DataFrames.
    pd.concat(out, names=["instrument", "datetime"]).to_csv(HERE / f"data_{universe}.csv")
    per_day = pd.Series([d for s in out.values() for d in s.index]).value_counts().sort_index()
    print("symbols per day: median", per_day.median(), "min", per_day.min(), "max", per_day.max())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "csi300")
