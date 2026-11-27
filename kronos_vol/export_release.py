"""Seed checkpoints (runs/<run>/model.pt, written by ft_intra.py --save) -> a release folder in the Hugging Face format
of Kronos-Vol (https://huggingface.co/ddedei/Kronos-Vol):
    seed_<k>.safetensors   fine-tuned backbone ("backbone.*") and head ("net.*") of one seed, float32
    config.json            base model ids, context, pooling, feature scaling, target mean, the VAL variance level
    kronos_rv.py, kronos_model/   the inference module and the Kronos model code, copied next to the weights
Then checks the folder: kronos_rv.KronosRV must reproduce the runs' own TEST forecasts (ensemble) on a sample of rows.
    py -3.11 export_release.py RUN_s0 RUN_s1 RUN_s2 [--out release/Kronos-Vol] [--check 300] [--device cpu]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

import evallib as E

CODE = Path(__file__).resolve().parent                 # kronos_rv.py and the vendored kronos_model live here
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", default=str(E.ROOT / "release" / "Kronos-Vol"))
ap.add_argument("--check", type=int, default=300)
ap.add_argument("--device", default=None, help="cpu keeps the GPU free for a training job")
a = ap.parse_args()
REL = Path(a.out)
REL.mkdir(parents=True, exist_ok=True)

cks = [torch.load(E.RUNS / r / "model.pt", map_location="cpu", weights_only=True) for r in a.runs]
args0 = cks[0]["args"]
for c in cks[1:]:
    for k in ("L", "pool", "loss", "panel", "holdout", "train_stocks", "target", "model"):
        assert c["args"][k] == args0[k], (k, c["args"][k], args0[k])
    assert np.allclose(list(c["mu"].values()), list(cks[0]["mu"].values())) and abs(c["ymu"] - cks[0]["ymu"]) < 1e-9
assert args0["target"] == "lrv1" and args0["mode"] == "full"

files = []
for r, c in zip(a.runs, cks):
    w = {f"net.{k}": v.contiguous().clone() for k, v in c["head"].items()}
    w.update({f"backbone.{k}": v.contiguous().clone() for k, v in c["backbone"].items()})
    f = f"seed_{c['args']['seed']}.safetensors"
    save_file(w, str(REL / f), metadata={"run": r})
    files.append(f)

# variance level from VAL only: E[RV] / exp(forecast), on the ensemble's VAL rows (held-out stocks included when written)
D = pd.read_parquet(E.DATA / args0["panel"])[["date", "symbol", "lrv1"]]
vp = [pd.read_parquet(E.RUNS / r / "val_preds.parquet").set_index(["date", "symbol"]).pred for r in a.runs]
v = pd.concat(vp, axis=1).mean(axis=1).rename("p").reset_index().merge(D, on=["date", "symbol"]).dropna()
var_scale = float(np.mean(np.exp(v.lrv1 - v.p)))

cfg = {"name": "Kronos-Vol", "base_model": f"NeoQuasar/Kronos-{args0['model']}", "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
       "L": args0["L"], "pool": args0["pool"], "loss": args0["loss"], "features": list(cks[0]["mu"].keys()),
       "feature_mean": cks[0]["mu"], "feature_std": cks[0]["sd"], "target": "log realised variance of the next session (5-minute returns)",
       "target_mean": cks[0]["ymu"], "var_scale": var_scale, "var_scale_rows": int(len(v)), "seed_files": files, "runs": a.runs,
       "train": {"panel": args0["panel"], "train_period": "2021-07-01 to 2023-12-31", "val_period": "2024-01-01 to 2024-06-30",
                 "held_out_stocks": args0["holdout"], "epochs": args0["epochs"], "batch": args0["bs"], "lr_backbone": args0["lr_bb"],
                 "lr_head": args0["lr_head"]}}
(REL / "config.json").write_text(json.dumps(cfg, indent=1, default=float))
if REL.resolve() != CODE:
    shutil.copy2(CODE / "kronos_rv.py", REL / "kronos_rv.py")
    shutil.copytree(CODE / "kronos_model", REL / "kronos_model", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
print(f"wrote {len(files)} seeds + config.json to {REL}; var_scale {var_scale:.4f} from {len(v):,} VAL rows", flush=True)

# check: the module reproduces the ensemble TEST forecasts
if a.check:
    from kronos_rv import KronosRV  # noqa: E402
    m = KronosRV.from_dir(REL, device=a.device)
    tp = pd.concat([pd.read_parquet(E.RUNS / r / "test_preds.parquet").set_index(["date", "symbol"]).pred for r in a.runs], axis=1)
    tp = tp.mean(axis=1).rename("p_run").reset_index()
    smp = tp.sample(min(a.check, len(tp)), random_state=0)
    spy = pd.read_parquet(E.DATA / "intraday" / "SPY.parquet")
    got = []
    for sym, g in smp.groupby("symbol"):
        bars = pd.read_parquet(E.DATA / "intraday" / f"{sym}.parquet")
        got.append(m.forecast({sym: bars}, spy, dates=g.date))
    got = pd.concat(got).merge(smp, on=["symbol", "date"])
    d = (got.next_log_rv - got.p_run).abs()
    print(f"check on {len(got)} TEST rows: max |diff| {d.max():.4f}, mean {d.mean():.5f}, 99th pct {d.quantile(0.99):.4f}", flush=True)
    if str(m.device).startswith("cuda"):                 # same arithmetic as training (fp16 autocast): must match
        assert d.max() < 0.02, "the release module does not reproduce the runs"
    else:
        # fp32 on CPU against fp16 in training: the tokenizer quantises by sign, so a value near zero can flip one
        # token bit and move a few windows. A real bug moves them all.
        print("  (CPU fp32 against the runs' GPU fp16: a few windows differ through token flips; exact needs CUDA)")
        assert d.mean() < 0.02 and d.quantile(0.99) < 0.15, "the release module does not reproduce the runs"
    print("CHECK PASSED")
