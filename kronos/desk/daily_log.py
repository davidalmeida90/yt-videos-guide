"""Forward test: 1 dated forecast per name after each US close, appended to ledger.jsonl. Scored later by the desk (/api/ledger).

    python daily_log.py

Rules fixed on 5 Nov 2026, before the first entry:
1. Same 12 names as the agent test. Kronos-small, lookback 100, 30 paths, T 1.0, top_p 0.9, 5 sessions ahead, seed from the date.
2. 2 rows per name: as shipped (training mode, what README users get) and eval mode.
3. Only completed daily bars. Before 16:15 New York time, today's bar is dropped.
4. Every line carries the hash of the line before it, so an edit to an old forecast breaks the chain. Never rewrite this file.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import sys
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import kr  # noqa: E402

TICKERS = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMZN", "JPM", "XOM", "UNH", "WMT", "TSLA", "KO"]
LOOKBACK, H, PATHS = 100, 5, 30
LEDGER = HERE / "ledger.jsonl"


def main():
    lines = LEDGER.read_text(encoding="utf-8").splitlines() if LEDGER.exists() else []
    prev = hashlib.sha256(lines[-1].encode()).hexdigest() if lines else "genesis"
    seen = {(e["ticker"], e["as_of"], e["mode"]) for e in map(json.loads, lines)}
    ny = datetime.now(ZoneInfo("America/New_York"))
    now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new = 0
    for mode, eval_mode in (("as_shipped", False), ("eval", True)):
        pred = kr.load("small", eval_mode=eval_mode)
        assert pred.model.training != eval_mode
        for t in TICKERS:
            d = kr.bars(t, "1d", period="2y")
            if d.index[-1].date() == ny.date() and (ny.hour, ny.minute) < (16, 15):
                d = d.iloc[:-1]
            as_of = str(d.index[-1].date())
            if (t, as_of, mode) in seen:
                continue
            hist = d.iloc[-LOOKBACK:]
            fut = pd.bdate_range(hist.index[-1], periods=H + 1)[1:]
            p = kr.paths(pred, hist, fut, n=PATHS, seed=int(as_of.replace("-", "")) + TICKERS.index(t))
            last = float(hist.close.iloc[-1])
            moves = (p[:, -1, 3] / last - 1) * 100
            row = {"logged_utc": now_utc, "ticker": t, "interval": "1d", "as_of": as_of, "h": H, "mode": mode, "model": "Kronos-small", "lookback": LOOKBACK, "paths": PATHS,
                   "last": round(last, 4), "up_share": round(float((moves > 0).mean()), 4), "p10": round(float(np.percentile(moves, 10)), 3), "p50": round(float(np.percentile(moves, 50)), 3),
                   "p90": round(float(np.percentile(moves, 90)), 3), "prev_hash": prev}
            line = json.dumps(row)
            with LEDGER.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            prev = hashlib.sha256(line.encode()).hexdigest()
            new += 1
            print(f"{mode:10} {t:5} as of {as_of}: {row['up_share'] * 100:3.0f}% of paths up, median {row['p50']:+.1f}%, range {row['p10']:+.1f}% to {row['p90']:+.1f}%", flush=True)
    print(f"{new} new rows, {len(lines) + new} in the ledger")


if __name__ == "__main__":
    main()
