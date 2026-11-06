"""Was the filmed web UI run typical? Repo's own defaults (small, 400 bars in, 120 out, 1 sample, T 1, top_p 0.9, as shipped)
on every later non overlapping window of the same Nvidia hourly file, scored against repeating the last price.

Added after seeing the filmed run, as a check on 1 line of the script. Forecast windows start after June 2024 only.
Reads <KRONOS_REPO>/data/NVDA_1h.csv, written by t4_webui.py.

    python u5_webui_default.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import kr  # noqa: E402

LOOKBACK, AHEAD = 400, 120
d = pd.read_csv(kr.REPO / "data" / "NVDA_1h.csv", parse_dates=["timestamps"]).set_index("timestamps")
print("rows", len(d), d.index[0], d.index[-1], flush=True)
pred = kr.load("small", eval_mode=False)
print("as shipped, model.training =", pred.model.training, flush=True)


def one(end, seed):
    """Forecast the AHEAD bars after position end. Returns predicted move, actual move, both errors."""
    hist, real = d.iloc[end - LOOKBACK:end], d.iloc[end:end + AHEAD]
    p = kr.paths(pred, hist, real.index, n=1, seed=seed)[0]
    last = hist["close"].iloc[-1]
    close, actual = p[:, kr.COLS.index("close")], real["close"].values
    return dict(start=str(real.index[0]), end=str(real.index[-1]), last=last, window_mean=hist["close"].mean(),
                pred_move=100 * (close[-1] / last - 1), actual_move=100 * (actual[-1] / last - 1),
                mape_kronos=100 * np.mean(np.abs(close - actual) / actual), mape_flat=100 * np.mean(np.abs(last - actual) / actual))


film = int(np.searchsorted(d.index, pd.Timestamp("2024-07-29 18:30")))
print("\nfilmed window, 5 samples", flush=True)
for s in range(5):
    r = one(film, s)
    print(f"  sample {s}: last {r['last']:.2f}, window mean {r['window_mean']:.2f}, predicted {r['pred_move']:+.1f}%, actual {r['actual_move']:+.1f}%, "
          f"error {r['mape_kronos']:.2f}% vs repeat the price {r['mape_flat']:.2f}%", flush=True)

rows = []
for end in range(film % AHEAD + LOOKBACK, len(d) - AHEAD + 1, AHEAD):
    if d.index[end] < pd.Timestamp("2024-07-01"):
        continue
    rows.append(one(end, 0))
    r = rows[-1]
    print(f"{r['start'][:10]} predicted {r['pred_move']:+6.1f}% actual {r['actual_move']:+6.1f}% error {r['mape_kronos']:5.2f}% vs {r['mape_flat']:5.2f}%", flush=True)
t = pd.DataFrame(rows)
t.to_csv(kr.OUT / "u5_webui_default.csv", index=False)
beat = (t.mape_kronos < t.mape_flat).mean()
print(f"\n{len(t)} windows: Kronos beat repeating the price {beat:.0%} of the time | direction right {(np.sign(t.pred_move) == np.sign(t.actual_move)).mean():.0%} | "
      f"predicted down {(t.pred_move < 0).mean():.0%}, really down {(t.actual_move < 0).mean():.0%}")
print(f"average error Kronos {t.mape_kronos.mean():.2f}% vs repeat the price {t.mape_flat.mean():.2f}%, median {t.mape_kronos.median():.2f}% vs {t.mape_flat.median():.2f}%")
print(f"average predicted move {t.pred_move.mean():+.1f}%, average actual {t.actual_move.mean():+.1f}%, correlation {t.pred_move.corr(t.actual_move):+.2f}")
below = t["last"] < t.window_mean
print(f"when the price sat below its window mean ({below.sum()} windows) predicted up {(t.pred_move[below] > 0).mean():.0%}; above it ({(~below).sum()} windows) predicted up {(t.pred_move[~below] > 0).mean():.0%}")
