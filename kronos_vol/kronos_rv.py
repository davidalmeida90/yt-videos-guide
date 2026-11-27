"""Kronos-Vol: Kronos-small fine-tuned to forecast the next session's realised variance of a US stock from its 5-minute bars.

Input, per stock: regular-session 5-minute bars (New York time, bar start 09:30 to 15:55), columns
ts, open, high, low, close, volume and optionally vwap, with at least 256 bars and 22 full sessions of history.
SPY bars are needed too: three market features come from SPY's realised variance.

Output, per stock and session t: a forecast of log RV_{t+1}, where RV is the sum of squared 5-minute log returns of the
session (the first bar from its own open). The overnight gap is not part of the target.

    from kronos_rv import KronosRV
    m = KronosRV.from_dir(".")                       # the folder with config.json and the seed files
    out = m.forecast({"AAPL": aapl_bars, "JPM": jpm_bars}, spy_bars)

Three fine-tuned seeds are averaged in log space, as in the evaluation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

FEATURES = ["lrv_d", "lrv_w", "lrv_m", "l_on", "l_last", "ret_d", "spy_lrv_d", "spy_lrv_w", "spy_lrv_m"]
COLS = ["open", "high", "low", "close", "volume", "amount"]


def session_bars(bars):
    """Regular-session bars, sorted, with `amount` (vwap x volume; mid of OHLC when vwap is missing)."""
    b = bars.copy()
    b["ts"] = pd.to_datetime(b["ts"])
    tm = b.ts.dt.hour * 60 + b.ts.dt.minute
    b = b[(tm >= 570) & (tm < 960)].sort_values("ts").reset_index(drop=True)
    vw = b["vwap"] if "vwap" in b else pd.Series(np.nan, index=b.index)
    b["amount"] = vw.fillna((b.open + b.high + b.low + b.close) / 4) * b.volume
    return b


def daily_features(bars):
    """Per session: realised variance and the HAR features, exactly as in training. Sessions with < 70 bars are dropped."""
    d = session_bars(bars)
    d["date"] = d.ts.dt.normalize()
    d["r"] = np.log(d.close / d.close.shift(1))
    first = d.date != d.date.shift(1)
    d.loc[first, "r"] = np.log(d.close / d.open)[first]
    d["late"] = (d.ts.dt.hour >= 15).astype(float)
    g = d.groupby("date")
    day = pd.DataFrame({"n": g.size(), "rv": g.r.apply(lambda s: (s ** 2).sum()), "o": g.open.first(), "c": g.close.last(),
                        "rv_last": g.apply(lambda x: ((x.r ** 2) * x.late).sum(), include_groups=False)})
    day = day[day.n >= 70].copy()
    day["on"] = np.log(day.o / day.c.shift(1)) ** 2
    rv = day.rv.clip(lower=1e-9)
    day["lrv_d"] = np.log(rv)
    day["lrv_w"] = np.log(rv.rolling(5).mean())
    day["lrv_m"] = np.log(rv.rolling(22).mean())
    day["l_on"] = np.log(day.on + 1e-8)
    day["l_last"] = np.log(day.rv_last + 1e-9)
    day["ret_d"] = np.log(day.c / day.c.shift(1))
    return day


class _Head(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))


class KronosRV:
    def __init__(self, cfg, tok, members, device):
        self.cfg, self.tok, self.members, self.device = cfg, tok, members, device
        self.mu = pd.Series(cfg["feature_mean"])[FEATURES]
        self.sd = pd.Series(cfg["feature_std"])[FEATURES]

    @classmethod
    def from_dir(cls, path=".", device=None):
        from safetensors.torch import load_file
        from kronos_model import Kronos, KronosTokenizer
        path = Path(path)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = json.loads((path / "config.json").read_text())
        tok = KronosTokenizer.from_pretrained(cfg["tokenizer"]).to(device).eval()
        members = []
        for f in cfg["seed_files"]:
            w = load_file(str(path / f))
            mdl = Kronos.from_pretrained(cfg["base_model"])
            mdl.load_state_dict({k[len("backbone."):]: v for k, v in w.items() if k.startswith("backbone.")})
            d = mdl.d_model * (2 if cfg["pool"] == "both" else 1)
            head = _Head(d + len(FEATURES))
            head.load_state_dict({k[len("net."):]: v for k, v in w.items() if k.startswith("net.")})
            members.append((mdl.to(device).eval(), head.to(device).eval()))
        return cls(cfg, tok, members, device)

    def _rows(self, bars, spy_bars):
        spy = daily_features(spy_bars)[["lrv_d", "lrv_w", "lrv_m"]].add_prefix("spy_")
        out = {}
        for sym, b in bars.items():
            b = session_bars(b)
            day = daily_features(b).join(spy, how="left")
            end = pd.Series(np.arange(len(b)), index=b.ts.dt.normalize()).groupby(level=0).max()
            out[sym] = (b, day, end)
        return out

    @torch.no_grad()
    def forecast(self, bars, spy_bars, dates=None, batch=128):
        """bars: {symbol: DataFrame}. dates: sessions to forecast from (default: each symbol's last full session).
        Returns one row per (symbol, date): the forecast for the NEXT session as log RV, variance and annualised volatility."""
        L, pool = self.cfg["L"], self.cfg["pool"]
        xs, sts, fs, keys = [], [], [], []
        for sym, (b, day, end) in self._rows(bars, spy_bars).items():
            want = day.index[-1:] if dates is None else pd.DatetimeIndex(pd.to_datetime(list(dates))).intersection(day.index)
            stamp = np.stack([b.ts.dt.minute, b.ts.dt.hour, b.ts.dt.weekday, b.ts.dt.day, b.ts.dt.month], 1).astype(np.float32)
            X = b[COLS].to_numpy(np.float32)
            for dt in want:
                j = end.get(dt)
                f = day.loc[dt, FEATURES]
                if j is None or j + 1 < L or f.isna().any():
                    continue
                x = X[j + 1 - L:j + 1]
                if not np.isfinite(x).all():
                    continue
                xs.append(x); sts.append(stamp[j + 1 - L:j + 1])
                fs.append(((f - self.mu) / self.sd).fillna(0.0).to_numpy(np.float32)); keys.append((sym, dt))
        if not keys:
            return pd.DataFrame(columns=["symbol", "date", "next_log_rv", "next_var", "next_vol_annual"])
        xs, sts, fs = np.stack(xs), np.stack(sts), np.stack(fs)
        preds = []
        for mdl, head in self.members:
            out = []
            for k in range(0, len(xs), batch):
                x = xs[k:k + batch]
                mu_, sd_ = x.mean(1, keepdims=True), x.std(1, keepdims=True)
                xn = np.clip((x - mu_) / (sd_ + 1e-5), -5.0, 5.0)          # per-window z-score, as Kronos's predictor
                x_t = torch.from_numpy(xn).to(self.device)
                st_t = torch.from_numpy(sts[k:k + batch]).to(self.device)
                f_t = torch.from_numpy(fs[k:k + batch]).to(self.device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
                    s1, s2 = self.tok.encode(x_t, half=True)
                    z = mdl.token_drop(mdl.embedding([s1, s2]) + mdl.time_emb(st_t))
                    for layer in mdl.transformer:
                        z = layer(z)
                    z = mdl.norm(z)
                    hz = {"last": lambda: z[:, -1, :], "day": lambda: z[:, -78:, :].mean(1),
                          "both": lambda: torch.cat([z[:, -1, :], z[:, -78:, :].mean(1)], 1)}[pool]()
                    p = head.head(torch.cat([hz.float(), f_t], 1)).squeeze(-1) + self.cfg["target_mean"]
                out.append(p.float().cpu().numpy())
            preds.append(np.concatenate(out))
        p = np.mean(preds, 0)
        var = np.exp(p) * self.cfg["var_scale"]                       # level set on the validation period
        return pd.DataFrame({"symbol": [k[0] for k in keys], "date": [k[1] for k in keys], "next_log_rv": p,
                             "next_var": var, "next_vol_annual": np.sqrt(var * 252)})
