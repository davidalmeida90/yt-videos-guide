"""Test 5: the boxes. 1 sampled Kronos path drawn as real candles, above the candles that actually printed.

    python t5_boxes.py

A mean of many paths smooths the candles into thin lines, so this draws single paths. Forecast made at the 9 Oct 2026
close for every session up to 5 Nov (5 Nov still trading when this ran).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import kr
from t2_visuals import AMBER, BG, DOWN, FG, MUTED, UP, candles, style

TICKER, AS_OF, LOOKBACK, SHOW = "NVDA", "2026-10-09", 100, 30


def main():
    d = kr.bars(TICKER, "1d", start="2024-01-01")
    as_of = pd.Timestamp(AS_OF)
    past, actual = d[d.index <= as_of], d[d.index > as_of]
    hist = past.iloc[-LOOKBACK:]
    pred = kr.load("small")
    p = kr.timed(f"{TICKER} 12 paths of {len(actual)} sessions", lambda: kr.paths(pred, hist, actual.index, n=12))
    last = hist.close.iloc[-1]
    ends = p[:, -1, 3] / last - 1
    print(f"as of {hist.index[-1].date()} close {last:.2f}; actual end {actual.close.iloc[-1]:.2f} "
          f"({(actual.close.iloc[-1] / last - 1) * 100:+.1f}%); path ends {np.round(np.sort(ends) * 100, 1).tolist()}")
    pick = int(np.argsort(ends)[len(ends) // 2])         # the median path, so the picture is not a cherry pick

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), dpi=100, facecolor=BG, sharex=True, sharey=True)
    show = past.iloc[-SHOW:]
    x0, x1 = np.arange(len(show)), len(show) + np.arange(len(actual))
    for ax, title in zip(axes, ("Kronos-small: one sampled path, the median of 12", "What happened")):
        style(ax)
        candles(ax, x0, show.open, show.high, show.low, show.close, alpha=0.45)
        ax.axvline(len(show) - 0.5, color="#3A4458", ls=(0, (3, 3)))
        ax.set_title(title, color=FG, fontsize=17, loc="left", pad=8)
    r = p[pick]
    candles(axes[0], x1, r[:, 0], r[:, 1], r[:, 2], r[:, 3], up=AMBER, down="#E07B00")
    candles(axes[1], x1, actual.open, actual.high, actual.low, actual.close, up=UP, down=DOWN)
    ticks = list(x0[::6]) + list(x1[::4])
    labels = [t.strftime("%d %b") for t in show.index[::6]] + [t.strftime("%d %b") for t in actual.index[::4]]
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels(labels, fontsize=11)
    fig.suptitle(f"{TICKER}: the candles Kronos draws when it only sees data up to 9 Oct", color=FG, fontsize=22, x=0.03,
                 ha="left", y=0.98)
    fig.text(0.03, 0.012, "Median of 12 sampled paths, lookback 100 sessions, repo as shipped. 5 Nov still trading.", color=MUTED, fontsize=12)
    fig.tight_layout(rect=(0.01, 0.03, 0.99, 0.94))
    f = kr.OUT / f"t5_boxes_{TICKER}.png"
    fig.savefig(f, facecolor=BG)
    print("saved", f)


if __name__ == "__main__":
    main()
