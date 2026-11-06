"""Shared helpers for the Kronos tests: load the model on the GPU, pull free bars, sample forecast paths.

KRONOS_REPO names a clone of shiyu-coder/Kronos. Default: a folder called Kronos beside this repository.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf

REPO = Path(os.environ.get("KRONOS_REPO") or Path(__file__).resolve().parents[2] / "Kronos")
if not (REPO / "model").is_dir():
    raise SystemExit(f"Kronos clone not found at {REPO}. Clone github.com/shiyu-coder/Kronos there or set KRONOS_REPO.")
sys.path.insert(0, str(REPO))
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
COLS = ["open", "high", "low", "close", "volume", "amount"]
IDS = {
    "mini": ("NeoQuasar/Kronos-Tokenizer-2k", "NeoQuasar/Kronos-mini", 2048),
    "small": ("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-small", 512),
    "base": ("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-base", 512),
}


def load(name="small", eval_mode=False):
    """eval_mode=False is the repo as shipped with its pinned huggingface_hub (issue 415: training mode at inference)."""
    t, m, ctx = IDS[name]
    tok, mod = KronosTokenizer.from_pretrained(t), Kronos.from_pretrained(m)
    pred = KronosPredictor(mod, tok, device=DEVICE, max_context=ctx)
    if eval_mode:
        pred.tokenizer.eval()
        pred.model.eval()
    return pred


def bars(ticker, interval="1d", start=None, period=None):
    d = yf.download(ticker, start=start, period=period, interval=interval, progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    d.index = pd.to_datetime(d.index)
    if d.index.tz is not None:
        d.index = d.index.tz_convert("UTC").tz_localize(None)
    d["amount"] = d["volume"] * d[["open", "high", "low", "close"]].mean(axis=1)
    d.index.name = "timestamps"
    return d


def paths(pred, hist, future_ts, n=30, T=1.0, top_p=0.9, seed=0):
    """n independent forecast paths for 1 window, as an array (n, len(future_ts), 6)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_ts = pd.Series(hist.index)
    y_ts = pd.Series(pd.DatetimeIndex(future_ts))
    out = pred.predict_batch(df_list=[hist[COLS]] * n, x_timestamp_list=[x_ts] * n, y_timestamp_list=[y_ts] * n,
                             pred_len=len(y_ts), T=T, top_p=top_p, sample_count=1, verbose=False)
    return np.stack([o[COLS].values for o in out])


def timed(label, fn):
    t = time.time()
    r = fn()
    print(f"{label}: {time.time() - t:.1f} s on {DEVICE}", flush=True)
    return r
