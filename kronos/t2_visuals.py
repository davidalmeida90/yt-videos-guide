"""Test 2: what Kronos looks like on real markets, the candles and the fan.

    python t2_visuals.py

a) NVDA daily: predicted candles against what happened, forecast made at the 30 Oct 2026 close, lookback 400 and 100.
b) BTC hourly: the official demo's fan and 2 probabilities, on today's data (the official page stopped on 4 Jul 2026).
Outputs in out/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import kr

BG, FG, MUTED, GRID = "#0B0F17", "#F2F4F8", "#8B93A7", "#1C2330"
UP, DOWN, AMBER = "#30CC5A", "#FF3B3B", "#FFB020"


def candles(ax, x, o, h, l, c, up=UP, down=DOWN, alpha=1.0, width=0.62):
    for xi, oi, hi, li, ci in zip(x, o, h, l, c):
        col = up if ci >= oi else down
        ax.plot([xi, xi], [li, hi], color=col, lw=1.1, alpha=alpha, zorder=2)
        ax.add_patch(plt.Rectangle((xi - width / 2, min(oi, ci)), width, max(abs(ci - oi), 1e-6), facecolor=col,
                                   edgecolor=col, alpha=alpha, zorder=3))


def style(ax):
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def nvda_panel(pred, name="NVDA"):
    d = kr.bars(name, "1d", start="2023-01-01")
    as_of = pd.Timestamp("2026-10-30")
    hist_all = d[d.index <= as_of]
    actual = d[d.index > as_of]
    fut = actual.index
    print(name, "as of", hist_all.index[-1].date(), "| next bars", [t.strftime("%d %b") for t in fut],
          "| last close", round(hist_all.close.iloc[-1], 2), "| actual closes", actual.close.round(2).tolist())

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=100, facecolor=BG, sharey=True)
    for ax, lb in zip(axes, (400, 100)):
        hist = hist_all.iloc[-lb:]
        p = kr.timed(f"{name} lookback {lb}, 30 paths of {len(fut)} bars", lambda: kr.paths(pred, hist, fut, n=30))
        mean = p.mean(axis=0)
        show = hist_all.iloc[-25:]
        style(ax)
        x0 = np.arange(len(show))
        candles(ax, x0, show.open, show.high, show.low, show.close, alpha=0.55)
        x1 = len(show) + np.arange(len(fut))
        # predicted candles, amber outline, next to the actual candle
        candles(ax, x1 - 0.2, actual.open, actual.high, actual.low, actual.close, width=0.34)
        for xi, row in zip(x1 + 0.2, mean):
            o, h, l, c = row[:4]
            ax.plot([xi, xi], [l, h], color=AMBER, lw=1.1, zorder=2)
            ax.add_patch(plt.Rectangle((xi - 0.17, min(o, c)), 0.34, max(abs(c - o), 1e-6), facecolor="none",
                                       edgecolor=AMBER, lw=1.5, zorder=3))
        ax.axvline(len(show) - 0.5, color="#3A4458", ls=(0, (3, 3)))
        ax.set_xticks([x0[-1], *x1])
        ax.set_xticklabels([show.index[-1].strftime("%d %b"), *[t.strftime("%d %b") for t in fut]], fontsize=11)
        last = hist_all.close.iloc[-1]
        up = float((p[:, -1, 3] > last).mean() * 100)
        chg_p, chg_a = (mean[-1, 3] / last - 1) * 100, (actual.close.iloc[-1] / last - 1) * 100
        ax.set_title(f"Lookback {lb}: predicted {chg_p:+.1f}%, actual {chg_a:+.1f}%, {up:.0f}% of paths up", color=FG,
                     fontsize=15, loc="left", pad=12)
        print(f"  lookback {lb}: mean path close change {chg_p:+.2f}% | actual {chg_a:+.2f}% | up paths {up:.0f}%")
    fig.suptitle(f"{name}: Kronos-small candles (amber outline) against what happened, forecast at the 30 Oct close",
                 color=FG, fontsize=20, x=0.03, ha="left", y=0.97)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.94))
    f = kr.OUT / f"t2_candles_{name}.png"
    fig.savefig(f, facecolor=BG)
    print("saved", f)


def btc_fan():
    pred = kr.load("mini")
    d = kr.bars("BTC-USD", "1h", period="60d")
    hist = d.iloc[-360:]
    fut = pd.date_range(hist.index[-1] + pd.Timedelta(hours=1), periods=24, freq="h")
    p = kr.timed("BTC mini, 30 paths of 24 hours", lambda: kr.paths(pred, hist, fut, n=30))
    last = hist.close.iloc[-1]
    close = p[:, :, 3]
    up_prob = (close[:, -1] > last).mean() * 100
    hist_vol = np.log(hist.close).diff().iloc[-24:].std()
    path_vol = np.diff(np.log(close), axis=1).std(axis=1)
    vol_amp = (path_vol > hist_vol).mean() * 100
    print(f"BTC at {hist.index[-1]} UTC, last {last:.0f}: upside {up_prob:.1f}% | vol amplification {vol_amp:.1f}%")

    fig, ax = plt.subplots(figsize=(16, 9), dpi=100, facecolor=BG)
    style(ax)
    show = hist.iloc[-120:]
    x0 = np.arange(len(show))
    ax.plot(x0, show.close, color="#5B8DEF", lw=2)
    x1 = len(show) - 1 + np.arange(1, 25)
    for path in close:
        ax.plot(x1, path, color=AMBER, lw=0.8, alpha=0.28)
    ax.fill_between(x1, close.min(axis=0), close.max(axis=0), color=AMBER, alpha=0.14, lw=0)
    ax.plot(x1, close.mean(axis=0), color=AMBER, lw=2.8)
    ax.axvline(len(show) - 1, color="#3A4458", ls=(0, (3, 3)))
    ax.set_title(f"BTC/USD next 24 hours: {up_prob:.0f}% of paths end higher, {vol_amp:.0f}% more volatile than the last day",
                 color=FG, fontsize=20, loc="left", pad=14)
    ax.set_xticks([x0[0], x0[-1], x1[-1]])
    ax.set_xticklabels([show.index[0].strftime("%d %b"), show.index[-1].strftime("%d %b %H:%M UTC"), "+24 h"], fontsize=12)
    fig.tight_layout()
    f = kr.OUT / "t2_btc_fan.png"
    fig.savefig(f, facecolor=BG)
    print("saved", f)


if __name__ == "__main__":
    pred = kr.load("small")
    print("training mode:", pred.model.training)
    nvda_panel(pred, "NVDA")
    btc_fan()
