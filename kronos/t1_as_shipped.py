"""Test 1: run Kronos exactly as the README says, on the repo's own sample data, and report what the code really does.

    python t1_as_shipped.py

Checks issue #415 (from_pretrained leaves the modules in training mode with the pinned huggingface_hub) and prints
how long 1 forecast takes on the GPU. Output: out/t1_readme_example.png
"""
import os
import sys
import time
from pathlib import Path

import huggingface_hub
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

REPO = Path(os.environ.get("KRONOS_REPO") or Path(__file__).resolve().parents[2] / "Kronos")
if not (REPO / "model").is_dir():
    raise SystemExit(f"Kronos clone not found at {REPO}. Clone github.com/shiyu-coder/Kronos there or set KRONOS_REPO.")
sys.path.insert(0, str(REPO))
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("torch", torch.__version__, "| device", DEVICE, torch.cuda.get_device_name(0) if DEVICE != "cpu" else "")
print("huggingface_hub", huggingface_hub.__version__, "(the repo pins 0.33.1)")

t = time.time()
tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
print(f"loaded in {time.time() - t:.1f} s")
print("ISSUE 415: tokenizer.training =", tok.training, "| model.training =", model.training)

pred = KronosPredictor(model, tok, device=DEVICE, max_context=512)
print("after KronosPredictor(): tokenizer.training =", pred.tokenizer.training, "| model.training =", pred.model.training)

# The README reads ./data/XSHG_5min_600977.csv, which is not in the repo (examples/data does not exist), so the first
# command a new user runs crashes. This is the only sample the repo ships.
df = pd.read_csv(REPO / "finetune_csv" / "data" / "HK_ali_09988_kline_5min_all.csv")
df["timestamps"] = pd.to_datetime(df["timestamps"])
lookback, pred_len = 400, 120
cols = ["open", "high", "low", "close", "volume", "amount"]
x_df = df.loc[: lookback - 1, cols]
x_ts = df.loc[: lookback - 1, "timestamps"]
y_ts = df.loc[lookback : lookback + pred_len - 1, "timestamps"]

t = time.time()
torch.manual_seed(0)
p = pred.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1,
                 verbose=False)
print(f"one forecast of {pred_len} bars from {lookback} bars: {time.time() - t:.1f} s on {DEVICE}")
print(p.head(3).round(3).to_string())

actual = df.loc[lookback : lookback + pred_len - 1, "close"].values
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(range(lookback + pred_len), df["close"].values[: lookback + pred_len], color="#1f4fd8", label="Actual")
ax.plot(range(lookback, lookback + pred_len), p["close"].values, color="#d81f2a", label="Kronos-small")
ax.axvline(lookback, color="#888", ls=":")
ax.legend()
ax.set_title("README flow, as shipped: the repo's Alibaba HK sample, 5 minute bars")
fig.tight_layout()
fig.savefig(OUT / "t1_readme_example.png", dpi=110)
print("saved", OUT / "t1_readme_example.png")
