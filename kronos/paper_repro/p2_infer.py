"""Kronos ranking signals on the GPU, same steps as finetune/qlib_test.py, resumable by chunk.

    python p2_infer.py --model base --universe csi300 --bench 600     # timing only, writes nothing
    python p2_infer.py --model base --universe csi300 --batch 600
Signals per window: last, mean, max, min (normalised space, as qlib_test.py) and avg_return (paper text, raw close),
plus pred_1 ... pred_10, the mean predicted close return at each horizon.
Output: signals_<universe>_<model>.csv
Needs data_<universe>.csv from p1_data.py. Windows only as written, because of below_normal_priority.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(os.environ.get("KRONOS_REPO") or Path(__file__).resolve().parents[3] / "Kronos")
if not (REPO / "model").is_dir():
    raise SystemExit(f"Kronos clone not found at {REPO}. Clone github.com/shiyu-coder/Kronos there or set KRONOS_REPO.")
sys.path.insert(0, str(REPO))
from model import Kronos, KronosTokenizer  # noqa: E402
from model.kronos import auto_regressive_inference  # noqa: E402

HERE = Path(__file__).parent
LOOKBACK, PRED, CLIP, MAX_CONTEXT, T, TOP_P, TOP_K, SAMPLES = 90, 10, 5.0, 512, 0.6, 0.9, 0, 5
START = "2024-07-01"
IDS = {"small": "NeoQuasar/Kronos-small", "base": "NeoQuasar/Kronos-base"}
STAMP = ["minute", "hour", "weekday", "day", "month"]
MAX_STEP = 1.0  # seconds per generation step. Windows resets the GPU near 2 s, which is what returned zeros and then blue screened the PC on 5 Nov.


class GPUFault(RuntimeError):
    pass


def below_normal_priority():
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


def stamps(index):
    idx = pd.DatetimeIndex(index)
    return np.stack([idx.minute, idx.hour, idx.weekday, idx.day, idx.month], axis=1).astype(np.float32)


def windows(data):
    """(symbol, end position): the window's last context row is the signal date, as QlibTestDataset does."""
    out = []
    for symbol, df in data.items():
        idx = df.index
        for i in range(LOOKBACK - 1, len(idx) - PRED):
            if idx[i] >= pd.Timestamp(START):
                out.append((idx[i], symbol, i))
    out.sort(key=lambda r: (r[0], r[1]))        # by date, so a stopped run still covers a continuous stretch
    return out


def run(batch_items, data, tok, model, device):
    xs, xst, yst, means, stds, lasts = [], [], [], [], [], []
    for _, symbol, i in batch_items:
        df = data[symbol]
        ctx = df.iloc[i - LOOKBACK + 1 : i + 1]
        x = ctx.values.astype(np.float32)
        m, s = x.mean(axis=0), x.std(axis=0)
        xs.append(np.clip((x - m) / (s + 1e-5), -CLIP, CLIP))
        xst.append(stamps(ctx.index))
        yst.append(stamps(df.index[i + 1 : i + 1 + PRED]))
        means.append(m[3])
        stds.append(s[3])
        lasts.append(x[-1, 3])
    x = torch.from_numpy(np.stack(xs)).to(device)
    preds = auto_regressive_inference(tok, model, x, torch.from_numpy(np.stack(xst)).to(device), torch.from_numpy(np.stack(yst)).to(device),
                                      max_context=MAX_CONTEXT, pred_len=PRED, clip=CLIP, T=T, top_k=TOP_K, top_p=TOP_P, sample_count=SAMPLES)
    preds = preds[:, -PRED:, :]
    close = preds[:, :, 3]
    last_norm = x[:, -1, 3].cpu().numpy()
    # A GPU reset (Windows timeout on a long step) comes back as zeros, not as an error. Never write that as data, and stop before it escalates.
    if not np.isfinite(preds).all() or close.std() < 1e-3 or close.std(axis=1).mean() < 1e-3 or np.abs(last_norm).max() == 0:
        raise GPUFault(f"bad output: finite {np.isfinite(preds).all()}, std {close.std():.5f}, across horizons {close.std(axis=1).mean():.5f}, input max {np.abs(last_norm).max():.3f}")
    means, stds, lasts = np.array(means), np.array(stds), np.array(lasts)
    raw = close * (stds[:, None] + 1e-5) + means[:, None]
    rows = {"last": close[:, -1] - last_norm, "mean": close.mean(axis=1) - last_norm, "max": close.max(axis=1) - last_norm,
            "min": close.min(axis=1) - last_norm, "avg_return": raw.mean(axis=1) / lasts - 1}
    t = pd.DataFrame(rows)
    for h in range(PRED):
        t[f"pred_{h + 1}"] = raw[:, h] / lasts - 1
    t.insert(0, "datetime", [str(r[0].date()) for r in batch_items])
    t.insert(1, "instrument", [r[1] for r in batch_items])
    return t.round(6)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="base")
    a.add_argument("--universe", default="csi300")
    a.add_argument("--batch", type=int, default=24, help="windows per GPU batch; sequences = windows x 5. Small on purpose, see MAX_STEP")
    a.add_argument("--bench", action="store_true")
    a.add_argument("--device", default="")
    o = a.parse_args()
    below_normal_priority()
    raw = pd.read_csv(HERE / f"data_{o.universe}.csv", parse_dates=["datetime"])
    data = {s: g.set_index("datetime")[["open", "high", "low", "close", "vol", "amt"]].astype(np.float32).sort_index() for s, g in raw.groupby("instrument")}
    device = o.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").to(device).eval()
    model = Kronos.from_pretrained(IDS[o.model]).to(device).eval()
    all_w = windows(data)
    days = sorted({w[0] for w in all_w})
    print(f"{o.universe} {o.model}: {len(all_w)} windows, {len(days)} days, {days[0].date()} to {days[-1].date()}, device {device}", flush=True)
    if o.bench:
        for b in ((6, 12, 24, 48, 96) if device != "cpu" else (6, 24)):
            for rep in range(3):                        # the first run of a new shape compiles kernels, so only later ones count
                t0 = time.time()
                torch.manual_seed(1 + rep)
                run(all_w[rep * b : (rep + 1) * b], data, tok, model, device)
                dt = time.time() - t0
            step = dt / PRED
            print(f"batch {b} windows ({b * SAMPLES} sequences): last run {dt:.2f} s, {step:.3f} s a step, {b / dt:.1f} windows/s, whole universe {len(all_w) / (b / dt) / 3600:.1f} h", flush=True)
            if step > MAX_STEP / 2:
                print(f"stopping the ramp: the next size would risk passing {MAX_STEP} s a step", flush=True)
                break
        return
    out = HERE / f"signals_{o.universe}_{o.model}.csv"
    done = set()
    if out.exists():
        p = pd.read_csv(out, usecols=["datetime", "instrument"])
        done = set(zip(p.datetime, p.instrument))
    # Every 5th trading day first, so rank IC is usable within hours; the other days fill in after, which the daily backtest needs.
    fifth = {d for i, d in enumerate(days) if i % 5 == 0}
    todo = [w for w in all_w if (str(w[0].date()), w[1]) not in done]
    todo.sort(key=lambda w: (w[0] not in fifth, w[0], w[1]))
    print(f"every 5th day first: {sum(w[0] in fifth for w in todo)} windows", flush=True)
    print(f"{len(done)} already done, {len(todo)} to go", flush=True)
    t0 = time.time()
    for k in range(0, len(todo), o.batch):
        part = todo[k : k + o.batch]
        torch.manual_seed(20240701 + k)
        t1 = time.time()
        try:
            rows = run(part, data, tok, model, device)
        except GPUFault as e:
            print(f"STOPPED, nothing written for this batch: {e}", flush=True)
            sys.exit(3)
        step = (time.time() - t1) / PRED
        if step > MAX_STEP and k > 3 * o.batch:          # after the kernels are compiled, a step this long is too close to the Windows GPU timeout
            print(f"STOPPED: {step:.2f} s a step is over the {MAX_STEP} s limit", flush=True)
            sys.exit(4)
        rows.to_csv(out, mode="a", header=not out.exists(), index=False)
        time.sleep(0.05)                                 # room for the desktop between batches
        n = k + len(part)
        if (k // o.batch) % 200 == 0:
            rate = n / (time.time() - t0)
            print(f"{n}/{len(todo)} up to {part[-1][0].date()}, {rate:.1f} windows/s, {(len(todo) - n) / rate / 3600:.1f} h left", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
