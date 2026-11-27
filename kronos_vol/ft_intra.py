"""Fine-tune Kronos-small on 5-minute bars to forecast next-day realised variance.
Kronos sees the last L 5-minute bars at the close of day t, with their real intraday timestamps; a small head joins its
last hidden state with the 9 HAR-RV features of the same day. Target: next-day log realised variance (lrv1) or the
next week's (lrv5).
Modes: full (whole transformer trained) | none (the head alone, no Kronos: the control) | frozen | lora;
--random_init trains the same transformer from random weights (the untrained twin).
Splits: TRAIN 2021-07..2023-12, VAL 2024-01..2024-06 (every choice, and the checkpoint), TEST 2024-07..2026-11 (read once).
HAR, HAR with the extra features (OLS) and LightGBM are scored on the same rows, so every number has its baseline beside it.
Writes runs/<run>/: card.json (args, VAL history, train loss, scores), test_preds.parquet, val_preds.parquet,
live.json while training (read by the LooksFine dashboard) and, with --save, model.pt for export_release.py.
Final recipe (one seed, about 50 min on a 6 GB RTX 4050 laptop GPU):
    py -3.11 ft_intra.py --mode full --target lrv1 --panel intra_daily200.parquet --loss mix --holdout 40 --epochs 3 --seed 0
"""
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import evallib as E
import kx

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["none", "frozen", "lora", "full"], required=True)
ap.add_argument("--L", type=int, default=256)
ap.add_argument("--epochs", type=float, default=6.0)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr_head", type=float, default=1e-3)
ap.add_argument("--lr_bb", type=float, default=1e-5)
ap.add_argument("--lr_lora", type=float, default=2e-4)
ap.add_argument("--target", default="lrv1")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--evals", type=int, default=12)
ap.add_argument("--random_init", action="store_true")
ap.add_argument("--panel", default="intra_daily200.parquet")
ap.add_argument("--holdout", type=int, default=0)      # stocks kept out of TRAIN and VAL entirely, scored apart on TEST
ap.add_argument("--save", action="store_true")
ap.add_argument("--loss", default="mse", choices=["mse", "qlike", "mix"])   # on log variance; qlike = exp(e) - e - 1
ap.add_argument("--pool", default="last", choices=["last", "day", "both"])  # last bar, mean of the last 78 bars, or both
ap.add_argument("--model", default="small", choices=["small", "base"])
ap.add_argument("--train_stocks", type=int, default=0)  # train on only N of the non-held-out stocks (fixed nested order)
ap.add_argument("--tag", default="")
ap.add_argument("--show", type=int, default=0)          # filming only: a live progress line every N steps (print only, 0 = off)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
XTRA = (f"_{a.loss}" if a.loss != "mse" else "") + (f"_pool{a.pool}" if a.pool != "last" else "") + ("_base" if a.model == "base" else "")
RUN = f"intra_{a.target}_{a.mode}{'_randinit' if a.random_init else ''}_L{a.L}{XTRA}{a.tag}_s{a.seed}"
OUT = E.RUNS / RUN
OUT.mkdir(parents=True, exist_ok=True)
TR_, VA_, TE_ = ("2021-07-01", "2023-12-31"), ("2024-01-01", "2024-06-30"), ("2024-07-01", "2026-11-10")
F = ["lrv_d", "lrv_w", "lrv_m", "l_on", "l_last", "ret_d", "spy_lrv_d", "spy_lrv_w", "spy_lrv_m"]

# ---------------------------------------------------------------- data
D = pd.read_parquet(E.DATA / a.panel).replace([np.inf, -np.inf], np.nan).dropna(subset=F + [a.target])
HOLD = set(np.random.default_rng(123).choice(sorted(D.symbol.unique()), a.holdout, replace=False)) if a.holdout else set()
if a.train_stocks:                                   # data-efficiency runs: 20 is a subset of 60, held-out stocks unchanged
    pool = sorted(set(D.symbol.unique()) - HOLD)
    USE = set(np.random.default_rng(7).permutation(pool)[:a.train_stocks])
    D = D[D.symbol.isin(USE | HOLD)]
X5, ST5, END = {}, {}, {}
for s in D.symbol.unique():
    b = pd.read_parquet(E.DATA / "intraday" / f"{s}.parquet").sort_values("ts").reset_index(drop=True)
    b["amount"] = b.vwap.fillna((b.open + b.high + b.low + b.close) / 4) * b.volume
    X5[s] = b[kx.COLS].to_numpy(np.float32)
    t = b.ts.dt
    ST5[s] = np.stack([t.minute, t.hour, t.weekday, t.day, t.month], 1).astype(np.uint8)    # small ints: 4x less RAM
    END[s] = pd.Series(np.arange(len(b)), index=b.ts.dt.normalize()).groupby(level=0).max()


def rows(lo, hi):
    return D[(D.date >= lo) & (D.date <= hi)]


TR, VA, TE = rows(*TR_), rows(*VA_), rows(*TE_)
TR, VA = TR[~TR.symbol.isin(HOLD)], VA[~VA.symbol.isin(HOLD)]
mu, sd = TR[F].mean(), TR[F].std()


def build(R):
    xs, st, fs, ys, keep = [], [], [], [], []
    fz = ((R[F] - mu) / sd).fillna(0.0).to_numpy(np.float32)
    for i, r in enumerate(R.itertuples()):
        j = END[r.symbol].get(r.date)
        if j is None or j + 1 < a.L:
            continue
        x = X5[r.symbol][j + 1 - a.L:j + 1]
        if not np.isfinite(x).all():
            continue
        xs.append(x); st.append(ST5[r.symbol][j + 1 - a.L:j + 1]); fs.append(fz[i]); ys.append(getattr(r, a.target)); keep.append(i)
    return np.stack(xs), np.stack(st), np.stack(fs), np.array(ys, np.float32), R.iloc[keep]


Xtr, Xva, Xte = build(TR), build(VA), build(TE)
ymu = float(Xtr[3].mean())
print(f"{RUN}: train {len(Xtr[0]):,}  val {len(Xva[0]):,}  test {len(Xte[0]):,} windows", flush=True)

# baselines on exactly these rows
from sklearn.linear_model import LinearRegression  # noqa: E402
import lightgbm as lgb  # noqa: E402
base = {}
for name, cols in (("har", ["lrv_d", "lrv_w", "lrv_m"]), ("har_plus", F)):
    m = LinearRegression().fit(Xtr[4][cols], Xtr[3])
    base[name] = (m.predict(Xva[4][cols]), m.predict(Xte[4][cols]))
g = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.02, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, verbose=-1)
g.fit(Xtr[4][F], Xtr[3], eval_set=[(Xva[4][F], Xva[3])], callbacks=[lgb.early_stopping(100, verbose=False)])
base["gbm"] = (g.predict(Xva[4][F]), g.predict(Xte[4][F]))

# ---------------------------------------------------------------- model (same as ft_vol.py)
tok, mdl = kx.load(f"NeoQuasar/Kronos-{a.model}")
if a.random_init:
    mdl.apply(mdl._init_weights)
for q in tok.parameters():
    q.requires_grad = False


class LoRA(nn.Module):
    def __init__(self, base_, r=8, alpha=16):
        super().__init__()
        self.base, self.s = base_, alpha / r
        self.A = nn.Parameter(torch.randn(r, base_.in_features, device=base_.weight.device) * (1 / math.sqrt(base_.in_features)))
        self.B = nn.Parameter(torch.zeros(base_.out_features, r, device=base_.weight.device))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.s


if a.mode == "lora":
    for blk in mdl.transformer:
        for nm in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(blk.self_attn, nm, LoRA(getattr(blk.self_attn, nm)))
for nm, q in mdl.named_parameters():
    q.requires_grad = (a.mode == "full") or (a.mode == "lora" and (nm.endswith(".A") or nm.endswith(".B")))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        d = 0 if a.mode == "none" else mdl.d_model * (2 if a.pool == "both" else 1)
        self.head = nn.Sequential(nn.Linear(d + len(F), 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(self, x, st, f):
        if a.mode == "none":
            h = f
        else:
            with torch.no_grad():
                s1, s2 = tok.encode(x, half=True)
            z = mdl.token_drop(mdl.embedding([s1, s2]) + mdl.time_emb(st))
            for layer in mdl.transformer:
                z = layer(z)
            z = mdl.norm(z)
            hz = {"last": lambda: z[:, -1, :], "day": lambda: z[:, -78:, :].mean(1),
                  "both": lambda: torch.cat([z[:, -1, :], z[:, -78:, :].mean(1)], 1)}[a.pool]()
            h = torch.cat([hz.float(), f], 1)
        return self.head(h).squeeze(-1) + ymu


net = Net().to(kx.DEV)
groups = [{"params": net.head.parameters(), "lr": a.lr_head}]
bb = [q for q in mdl.parameters() if q.requires_grad]
if bb:
    groups.append({"params": bb, "lr": a.lr_lora if a.mode == "lora" else a.lr_bb})
opt = torch.optim.AdamW(groups, weight_decay=0.01)
n = len(Xtr[0]); steps = int(a.epochs * n / a.bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g_["lr"] for g_ in groups], total_steps=steps, pct_start=0.05)
scaler = torch.amp.GradScaler()


def to_dev(X, idx):
    xn, _, _ = kx.norm(X[0][idx])
    return (torch.from_numpy(xn).to(kx.DEV), torch.from_numpy(X[1][idx].astype(np.float32)).to(kx.DEV), torch.from_numpy(X[2][idx]).to(kx.DEV),
            torch.from_numpy(X[3][idx]).to(kx.DEV))


@torch.no_grad()
def predict(X):
    net.eval(); mdl.eval(); out = []
    pb = 256 if a.model == "small" and a.L <= 256 else 64
    for k in range(0, len(X[0]), pb):
        x, st, f, _ = to_dev(X, np.arange(k, min(k + pb, len(X[0]))))
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(net(x, st, f).float().cpu().numpy())
    net.train(); mdl.train(a.mode == "full")
    return np.concatenate(out)


def score(X, pred):
    return E.vol_scores(X[4].assign(pred=pred), "pred", a.target)


net.train(); mdl.train(a.mode == "full")
best, hist, t0 = None, [], time.time()
tl = []                                               # (step, train loss) every 25 steps, for loss charts
show_ema = None                                       # --show only
eval_at = set(int(steps * (k + 1) / a.evals) for k in range(a.evals))
perm = np.random.permutation(n); pos = 0
for step in range(1, steps + 1):
    if pos + a.bs > n:
        perm = np.random.permutation(n); pos = 0
    idx = perm[pos:pos + a.bs]; pos += a.bs
    x, st, f, y = to_dev(Xtr, idx)
    with torch.autocast("cuda", dtype=torch.float16):
        e = (y - net(x, st, f).float()).clamp(-10, 10)
        loss = {"mse": lambda: (e ** 2).mean(), "qlike": lambda: (torch.exp(e) - e - 1).mean(),
                "mix": lambda: (e ** 2 + torch.exp(e) - e - 1).mean()}[a.loss]()
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([q for g_ in groups for q in g_["params"]], 1.0)
    scaler.step(opt); scaler.update(); sched.step()
    if step % 25 == 0:
        tl.append((step, round(loss.item(), 5)))
    if a.show and step % a.show == 0:                     # one line rewritten in place: epoch, bar, smoothed loss, time left, GPU memory
        lv_ = loss.item(); show_ema = lv_ if show_ema is None else 0.9 * show_ema + 0.1 * lv_
        el_ = time.time() - t0; bar_ = int(20 * step / steps)
        print(f"\r  epoch {step / steps * a.epochs:4.2f}/{a.epochs:g} [{'#' * bar_}{'.' * (20 - bar_)}] step {step:>4}/{steps}"
              f"  loss {show_ema:.4f}  {el_ / step * (steps - step) / 60:3.0f} min left  GPU {torch.cuda.memory_reserved() / 2**30:.1f} GB ",
              end="", flush=True)
    if step % 100 == 0 or step in eval_at:                # live file for the LooksFine dashboard
        (OUT / "live.json").write_text(json.dumps({"step": step, "steps": steps, "secs": round(time.time() - t0, 1),
                                                   "train_loss": tl, "history": hist, "args": vars(a)}, default=float))
    if step in eval_at:
        v = score(Xva, predict(Xva))
        hist.append({"step": step, **{f"val_{k}": x_ for k, x_ in v.items()}})
        if a.show:
            print("\r" + " " * 96 + "\r", end="")
        print(f"  step {step}/{steps}  {time.time() - t0:.0f}s  val r2 {v['r2']}  qlike {v['qlike']}  ic {v['cs_ic']}", flush=True)
        if best is None or v["qlike"] < best[0]:
            best = (v["qlike"], step, {k: q.detach().clone() for k, q in net.state_dict().items()},
                    {k: q.detach().clone() for k, q in mdl.state_dict().items()} if bb else None)

net.load_state_dict(best[2])
if best[3] is not None:
    mdl.load_state_dict(best[3])
pv, pt = predict(Xva), predict(Xte)
res = {"model": (score(Xva, pv), score(Xte, pt))}
for k, (bv, bt) in base.items():
    res[k] = (score(Xva, bv), score(Xte, bt))
print(f"\n{RUN}   (R2 log | QLIKE | cross-sectional IC)        VAL                      TEST")
for k, (rv, rt) in res.items():
    print(f"  {k:9s} {rv['r2']:.4f} {rv['qlike']:.4f} {rv['cs_ic']:.4f}      {rt['r2']:.4f} {rt['qlike']:.4f} {rt['cs_ic']:.4f}")
card = {"run": RUN, "args": vars(a), "best_step": best[1], "results": {k: {"val": v[0], "test": v[1]} for k, v in res.items()},
        "test_by_year_r2": E.by_year(Xte[4].assign(pred=pt), "pred", a.target), "minutes": round((time.time() - t0) / 60, 1), "history": hist, "train_loss": tl}
(OUT / "card.json").write_text(json.dumps(card, indent=1, default=float))
Xte[4].assign(pred=pt, holdout=Xte[4].symbol.isin(HOLD))[["date", "symbol", "pred", "holdout"]].to_parquet(OUT / "test_preds.parquet", index=False)
try:                                                  # VAL forecasts, held-out stocks included, so a recipe can be chosen off TEST
    parts = [Xva[4].assign(pred=pv, holdout=False)]
    VH = rows(*VA_)
    VH = VH[VH.symbol.isin(HOLD)]
    if len(VH):
        Xvh = build(VH)
        parts.append(Xvh[4].assign(pred=predict(Xvh), holdout=True))
    pd.concat(parts)[["date", "symbol", "pred", "holdout"]].to_parquet(OUT / "val_preds.parquet", index=False)
except Exception as ex_:                              # never lose a finished run over this extra file
    print(f"  val_preds not written: {ex_!r}", flush=True)
if HOLD:
    for nm, mask in (("seen", ~Xte[4].symbol.isin(HOLD).values), ("holdout", Xte[4].symbol.isin(HOLD).values)):
        sub = (Xte[0][mask], Xte[1][mask], Xte[2][mask], Xte[3][mask], Xte[4][mask])
        print(f"  TEST {nm:8s} stocks {sub[4].symbol.nunique():3d}: model {score(sub, pt[mask])}  " +
              "  ".join(f"{k} {score(sub, bt[mask])['qlike']:.4f}" for k, (bv, bt) in base.items()))
if a.save:
    torch.save({"head": net.state_dict(), "backbone": mdl.state_dict() if bb else None, "mu": mu.to_dict(), "sd": sd.to_dict(), "ymu": ymu,
                "args": vars(a)}, OUT / "model.pt")
E.log_result(f"ft_{RUN}", {"val": res["model"][0], "test": res["model"][1], "baselines_test": {k: v[1] for k, v in res.items() if k != "model"}})
