"""How much memory does Kronos really need, and how slow is the CPU? Numbers for the install part of the video.

Peak allocated by PyTorch for the forecasts this project actually runs. 1 job, small batches. Heaviest case tried
once (base, 400 bars, 30 paths) took 1.6 s a step, over the 1 s limit set after a GPU crash (Windows stop code 0x119), so it is left out of this list.

    python u4_vram.py            measure on the GPU
    python u4_vram.py --cap 1    same, with PyTorch held to 1 GB
    python u4_vram.py --cpu      time the same forecasts on the CPU
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import kr  # noqa: E402

# (model, lookback, paths, ahead). 400/1/120 is what the repo's web UI runs, 100/30/5 is what this project's tool runs.
CASES = [("mini", 100, 30, 5), ("small", 100, 30, 5), ("small", 400, 1, 120), ("base", 100, 30, 5), ("base", 400, 1, 120)]

a = argparse.ArgumentParser()
a.add_argument("--cap", type=float, help="hold PyTorch to this many GB of GPU memory")
a.add_argument("--cpu", action="store_true")
args = a.parse_args()
if args.cpu:
    kr.DEVICE = "cpu"
elif args.cap:
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(args.cap * 1e9 / total, 0)
    print(f"total GPU memory seen by PyTorch {total / 1e9:.1f} GB, capped at {args.cap} GB", flush=True)

d = kr.bars("NVDA", "1d", period="3y")
gpu = kr.DEVICE != "cpu"
print("device", kr.DEVICE, torch.cuda.get_device_name(0) if gpu else f"{torch.get_num_threads()} threads", flush=True)
pred, loaded = None, None
for name, lookback, n, h in CASES:
    if name != loaded:
        del pred
        if gpu:
            torch.cuda.empty_cache()
        pred, loaded = kr.load(name, eval_mode=True), name
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    hist = d.iloc[-lookback:]
    fut = pd.bdate_range(hist.index[-1], periods=h + 1)[1:]
    t = time.time()
    try:
        kr.paths(pred, hist, fut, n=n)
    except RuntimeError as e:
        print(f"{name} {lookback} bars, {n} paths, {h} ahead: FAILED {str(e)[:80]}", flush=True)
        continue
    s = time.time() - t
    mem = f"peak {torch.cuda.max_memory_allocated() / 1e6:.0f} MB, reserved {torch.cuda.max_memory_reserved() / 1e6:.0f} MB, " if gpu else ""
    print(f"{name} {lookback} bars, {n} paths, {h} ahead: {mem}{s:.1f} s, {s / h:.2f} s a step", flush=True)
