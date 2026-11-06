"""Bar to beat: 1 line price signals through the repo's own backtest settings, both windows. No model, no GPU.
Run from this folder: it reads data_csi300.csv by relative path and adds rows to floor_csi300.csv.

Paper Table 10, CSI300: +19.11% annual excess return, IR 1.38, Jul 2024 to Jun 2025. Audit: +12.89%, IR 1.13.
kernels=1 and the main guard stop qlib spawning worker processes on Windows, which ran away on the first attempt.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    import qlib
    from qlib.config import REG_CN
    import p3_score as p

    uni, only = "csi300", sys.argv[1:] or None
    qlib.init(provider_uri=str(Path.home() / ".qlib" / "qlib_data" / "cn_data"), region=REG_CN, kernels=1)
    raw = pd.read_csv(f"data_{uni}.csv", parse_dates=["datetime"])
    data = {s: g.set_index("datetime").sort_index() for s, g in raw.groupby("instrument")}
    fl = p.floor_signals(data)
    fl["random"] = np.random.default_rng(1).random(len(fl))
    names = only or ["rev5", "rev10", "rev20", "rev60", "dist90", "z90", "-rev5", "-rev20", "random"]
    out = Path(f"floor_{uni}.csv")
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    for wname, (s, e) in p.WINDOWS.items():
        g = fl[(fl.datetime >= s) & (fl.datetime <= e)]
        end = min(pd.Timestamp(e), g.datetime.max())
        for name in names:
            ser = g.set_index(["datetime", "instrument"])[name].dropna().sort_index()
            m, _ = p.run_backtest(ser, (str(g.datetime.min().date()), str(end.date())), p.BENCH[uni])
            rows.append({"window": wname, "signal": name, **{k: round(float(v), 3) for k, v in m.items()}})
            print("RESULT", rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
