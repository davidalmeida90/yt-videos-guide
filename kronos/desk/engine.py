"""Kronos internals for the desk: tokens per candle, the next candle as a probability distribution, paths generated step by step.

Uses the repo's modules in eval mode. Its step loop mirrors model.kronos.auto_regressive_inference, with a yield per step so the
page can draw each candle as the model writes it.
KRONOS_REPO names a clone of shiyu-coder/Kronos. Default: a folder called Kronos beside this repository.
"""
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(os.environ.get("KRONOS_REPO") or Path(__file__).resolve().parents[3] / "Kronos")
if not (REPO / "model").is_dir():
    raise SystemExit(f"Kronos clone not found at {REPO}. Clone github.com/shiyu-coder/Kronos there or set KRONOS_REPO.")
sys.path.insert(0, str(REPO))
from model import Kronos, KronosTokenizer  # noqa: E402
from model.kronos import calc_time_stamps, sample_from_logits  # noqa: E402

COLS = ["open", "high", "low", "close", "volume", "amount"]
CLIP = 5.0
IDS = {"mini": ("NeoQuasar/Kronos-Tokenizer-2k", "NeoQuasar/Kronos-mini"), "small": ("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-small"),
       "base": ("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-base")}


class Engine:
    def __init__(self, name="small"):
        self.name = name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        t, m = IDS[name]
        self.tok = KronosTokenizer.from_pretrained(t).to(self.device).eval()
        self.model = Kronos.from_pretrained(m).to(self.device).eval()
        self.lock = threading.Lock()
        self.s1_size = 2 ** self.model.s1_bits
        self.s2_size = 2 ** self.model.s2_bits

    def _prep(self, hist):
        x = hist[COLS].values.astype(np.float32)
        mean, std = x.mean(axis=0), x.std(axis=0)
        z = np.clip((x - mean) / (std + 1e-5), -CLIP, CLIP)
        st = calc_time_stamps(pd.Series(hist.index)).values.astype(np.float32)
        return torch.from_numpy(z).unsqueeze(0).to(self.device), torch.from_numpy(st).unsqueeze(0).to(self.device), mean, std

    def _raw(self, z, mean, std):
        return z * (std + 1e-5) + mean

    @torch.no_grad()
    def tokens(self, hist, k=12):
        with self.lock:
            x, _, _, _ = self._prep(hist)
            s1, s2 = self.tok.encode(x, half=True)
        out = []
        for i in range(-k, 0):
            r = hist.iloc[i]
            out.append({"time": hist.index[i].strftime("%Y-%m-%d"), "open": float(r.open), "high": float(r.high), "low": float(r.low), "close": float(r.close),
                        "s1": int(s1[0, i]), "s2": int(s2[0, i])})
        return out

    @torch.no_grad()
    def next_candle(self, hist, top=6):
        """Every possible next coarse token with its probability and the candle it decodes to."""
        with self.lock:
            x, st, mean, std = self._prep(hist)
            s1, s2 = self.tok.encode(x, half=True)
            logits, ctx = self.model.decode_s1(s1, s2, st)
            p = F.softmax(logits[:, -1, :], dim=-1)[0]
            cand = torch.arange(self.s1_size, device=self.device).unsqueeze(1)
            ctx_all = ctx[:, -1:, :].repeat(self.s1_size, 1, 1)
            s2_logits = self.model.decode_s2(ctx_all, cand)[:, -1, :]
            s2_best = s2_logits.argmax(dim=-1, keepdim=True)
            n = self.s1_size
            pre = torch.cat([s1.repeat(n, 1), cand], dim=1)
            post = torch.cat([s2.repeat(n, 1), s2_best], dim=1)
            z = torch.cat([self.tok.decode([pre[a : a + 256], post[a : a + 256]], half=True)[:, -1, :] for a in range(0, n, 256)])
        raw = self._raw(z.cpu().numpy(), mean, std)
        p = p.cpu().numpy()
        last = float(hist.close.iloc[-1])
        moves = raw[:, 3] / last - 1
        order = np.argsort(-p)[:top]
        hist_bins = np.linspace(-0.10, 0.10, 81)
        weights, _ = np.histogram(np.clip(moves, -0.0999, 0.0999), bins=hist_bins, weights=p)
        srt = np.argsort(moves)
        cdf = np.cumsum(p[srt])
        pct = lambda q: float(moves[srt][np.searchsorted(cdf, q)] * 100)
        return {"last": last, "up_prob": float(p[moves > 0].sum()), "expected_move": float((p * moves).sum() * 100), "p10": pct(0.1), "p50": pct(0.5), "p90": pct(0.9),
                "candidates": [{"s1": int(i), "prob": float(p[i]), "open": float(raw[i, 0]), "high": float(raw[i, 1]), "low": float(raw[i, 2]), "close": float(raw[i, 3]),
                                "move": float(moves[i] * 100)} for i in order],
                "hist": {"edges": [float(e * 100) for e in hist_bins], "weights": [float(w) for w in weights]}, "vocab": self.s1_size}

    @torch.no_grad()
    def surprise(self, bars, day_list, lookback=100):
        """-log2 p(the coarse token that happened), for each day, given the lookback bars before it. In bits."""
        out = []
        with self.lock:
            for day in day_list:
                i = bars.index.get_loc(day)
                if i < lookback:
                    continue
                x, st, _, _ = self._prep(bars.iloc[i - lookback : i + 1])
                s1, s2 = self.tok.encode(x, half=True)
                logits, _ = self.model.decode_s1(s1[:, :-1], s2[:, :-1], st[:, :-1, :])
                lp = F.log_softmax(logits[:, -1, :], dim=-1)[0, s1[0, -1]]
                out.append((str(pd.Timestamp(day).date()), float(-lp / np.log(2))))
        return out

    @torch.no_grad()
    def stream(self, hist, future_index, n=30, T=1.0, top_p=0.9, seed=0):
        """Yield (step, candles) with candles shaped (n, 6) in raw units, 1 step per future bar, then ('final', all candles)."""
        with self.lock:
            torch.manual_seed(seed)
            x, st, mean, std = self._prep(hist)
            y = torch.from_numpy(calc_time_stamps(pd.Series(future_index)).values.astype(np.float32)).unsqueeze(0).to(self.device)
            x_n, st_n, y_n = x.repeat(n, 1, 1), st.repeat(n, 1, 1), y.repeat(n, 1, 1)
            s1, s2 = self.tok.encode(x_n, half=True)
            stamp = torch.cat([st_n, y_n], dim=1)
            base = s1.size(1)
            for i in range(y_n.size(1)):
                logits, ctx = self.model.decode_s1(s1, s2, stamp[:, : base + i, :])
                a = sample_from_logits(logits[:, -1, :], temperature=T, top_k=0, top_p=top_p, sample_logits=True)
                b_logits = self.model.decode_s2(ctx, a)
                b = sample_from_logits(b_logits[:, -1, :], temperature=T, top_k=0, top_p=top_p, sample_logits=True)
                s1, s2 = torch.cat([s1, a], dim=1), torch.cat([s2, b], dim=1)
                z = self.tok.decode([s1, s2], half=True)[:, -1, :].cpu().numpy()
                yield i, self._raw(z, mean, std)
            z = self.tok.decode([s1, s2], half=True)[:, -y_n.size(1) :, :].cpu().numpy()
            yield "final", self._raw(z, mean, std)
