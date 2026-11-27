# Kronos-Vol: fine-tuning Kronos to forecast volatility

Code behind the YouTube video on fine-tuning [Kronos](https://github.com/shiyu-coder/Kronos), an open source candlestick
foundation model, to forecast how much a US stock will move in the next session. Video link: coming soon.

- Published model: [huggingface.co/ddedei/Kronos-Vol](https://huggingface.co/ddedei/Kronos-Vol) (MIT)
- Dashboard used in the video: [LooksFine](https://github.com/davidalmeida90/looksfine)

Kronos-small reads the last 256 five-minute candles of a stock; a small head joins its last hidden state with 9 classic
volatility features, and the whole network is trained on next-day realised variance. Everything here runs on 1 laptop GPU.

**Research only. It forecasts how much a stock moves, never which way, and nothing here is investment advice.**

## Results

40 stocks, drawn at random, were held out of training and validation entirely (39 have data in the test window).
Test window: 1 Jul 2024 to 10 Nov 2026, after Kronos's pretraining data. 22,637 stock-days. Seeds 0, 1 and 2 averaged.

| Forecast | Calibrated QLIKE | Lower loss than HAR-RV | DM t vs HAR-RV | R² (log RV) | Rank IC |
|---|---|---|---|---|---|
| **Kronos fine-tune (Kronos-Vol recipe)** | **0.1790** | **8.9%** | -4.85 | 0.562 | 0.733 |
| Same head without Kronos (control) | 0.1849 | 5.9% | -3.58 | 0.558 | 0.731 |
| LightGBM, same 9 features | 0.1889 | 3.8% | -3.80 | 0.552 | 0.727 |
| HAR + the extra features (OLS) | 0.1885 | 4.1% | -3.06 | 0.551 | 0.728 |
| HAR-RV | 0.1965 | 0% | | 0.533 | 0.722 |
| Yesterday's realised variance | 0.2405 | -22.4% | +7.34 | 0.370 | 0.654 |

- Against the control, the same network trained the same way without Kronos: 3.2% lower loss, DM t -4.71.
- Each test year: 2024 0.1874 against HAR-RV 0.2044, 2025 0.1946 against 0.2207, 2026 0.1543 against 0.1635.
- Stock by stock: lower loss than HAR-RV on 39 of 39 held-out stocks, than the control on 35 of 39.
- Pretraining buys data efficiency. With only 20 training stocks: Kronos 0.1810, the same transformer from random weights
  0.1839, no Kronos 0.1859.
- Kronos as shipped (zero-shot, 16 sampled paths of the next session) on 1,289 held-out stock-days: 0.2444, worse than
  yesterday's variance. On the same rows the fine-tune scores 0.1719 and HAR-RV 0.1870.
- In a portfolio sized by the forecast (12 large names), Sharpe 1.32 against 1.30 with HAR-RV sizing: steadier risk, and
  no significant difference in Sharpe.

Calibrated QLIKE removes each model's own constant level shift first, because raw QLIKE rewards forecasting high. Lower
is better. DM t: Diebold-Mariano on the daily cross-sectional mean loss, Newey-West errors, 5 lags; below -2 is a real gain.

## Hardware

- 1 NVIDIA RTX 4050 laptop GPU with 6 GB, 16 GB of RAM, Windows 11, Python 3.11.
- About 50 minutes per seed for the fine-tune (3 epochs, 4,507 steps), about 1 minute for the control.
- Run 1 training job at a time: 2 at once filled the RAM.
- A CUDA GPU is needed for training (fp16 autocast). `use_model.py` also runs on CPU.

## Install

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Kronos's model code is included in `kronos_model/` (MIT, commit 67b630e, 1 import made relative), so no clone of the
Kronos repository is needed. Its pretrained weights download from the Hugging Face Hub on first use.

Keys come from the environment only, see `.env.example` for the names. A free [Alpaca](https://alpaca.markets) account
gives the market data API:

```
$env:ALPACA_API_KEY = "..."
$env:ALPACA_SECRET_KEY = "..."
```

## Run, in order

**1. Data: 5-minute bars, then the daily panel.** About 660 MB of bars for 200 stocks and SPY, June 2021 to November 2026.

```
py -3.11 download_bars.py 200        # data/intraday/<SYM>.parquet; stop and rerun at will, finished names are skipped
py -3.11 build_panel.py 200          # data/intra_daily200.parquet: 199 stocks (TIF has no bars), 1 row per stock and session
py -3.11 har.py                      # the simple rules on TEST, CPU, seconds: the bar to clear
```

**2. Fine-tune, the final recipe, 3 seeds.** The recipe was chosen on the validation period by a rule fixed in advance.

```
py -3.11 ft_intra.py --mode full --target lrv1 --panel intra_daily200.parquet --loss mix --holdout 40 --epochs 3 --seed 0 --save --tag _u200
py -3.11 ft_intra.py --mode full --target lrv1 --panel intra_daily200.parquet --loss mix --holdout 40 --epochs 3 --seed 1 --save --tag _u200
py -3.11 ft_intra.py --mode full --target lrv1 --panel intra_daily200.parquet --loss mix --holdout 40 --epochs 3 --seed 2 --save --tag _u200
```

| Flag | Meaning |
|---|---|
| `--mode full` | train the whole Kronos-small transformer (24.7M parameters) and the head; the Kronos tokenizer stays frozen |
| `--target lrv1` | next session's log realised variance: the sum of squared 5-minute returns |
| `--panel intra_daily200.parquet` | the panel from `build_panel.py 200` |
| `--loss mix` | squared error plus QLIKE on the log variance error e: `e² + exp(e) - e - 1` |
| `--holdout 40` | 40 stocks drawn at random (generator seed 123) never enter TRAIN or VAL, and are scored on TEST alone |
| `--epochs 3` | 3 passes over the 96,154 training windows, 4,507 steps of 64 |
| `--seed 0` | torch and numpy seed; seeds 0, 1 and 2 are averaged |
| `--save` | keep the weights (`model.pt`) for `export_release.py` |
| `--tag _u200` | a suffix of the run name, `intra_lrv1_full_L256_mix_u200_s0`; LooksFine groups runs by it |

Fixed by default: 256 bars of context, batch 64, AdamW with a one-cycle schedule, learning rate 1e-5 for the transformer
and 1e-3 for the head, fp16 autocast, 12 validation checks, and the checkpoint with the lowest validation QLIKE is kept.
Splits: TRAIN July 2021 to December 2023, VAL January to June 2024, TEST July 2024 to November 2026.

Each run writes `runs/<run>/`: `card.json` (arguments, validation history, training loss, scores), `test_preds.parquet`,
`val_preds.parquet`, `live.json` while it trains, and `model.pt` with `--save`. It also prints HAR-RV, HAR with the extra
features and LightGBM on the same rows.

**3. Control and twins.**

```
py -3.11 ft_intra.py --mode none --target lrv1 --panel intra_daily200.parquet --loss mix --holdout 40 --epochs 3 --seed 0 --save --tag _u200
```

Seeds 1 and 2 the same way. Optional, the twins from the video:
- Random weights, same transformer: add `--random_init --lr_bb 1e-4` to the fine-tune command.
- Data efficiency, 20 training stocks: `--train_stocks 20 --epochs 8 --tag _u200n20` (the 20 are a fixed draw from the
  159 training stocks; the same 40 stay held out).

**4. Score the held-out stocks.**

```
py -3.11 evaluate.py --kronos intra_lrv1_full_L256_mix_u200_s0 intra_lrv1_full_L256_mix_u200_s1 intra_lrv1_full_L256_mix_u200_s2 --control intra_lrv1_none_L256_mix_u200_s0 intra_lrv1_none_L256_mix_u200_s1 intra_lrv1_none_L256_mix_u200_s2
```

This prints the results table above, the fine-tune against the control, each year and stock by stock. `--subset seen`
scores the training stocks on later dates. `compare.py` puts any runs or seed groups (joined with `+`) side by side with
Diebold-Mariano tests against a reference.

**5. Train the release model and export it.** The published model uses the same recipe on all 199 stocks:

```
py -3.11 ft_intra.py --mode full --target lrv1 --panel intra_daily200.parquet --loss mix --holdout 0 --epochs 3 --seed 0 --save --tag _all
```

Seeds 1 and 2 the same way, then:

```
py -3.11 export_release.py intra_lrv1_full_L256_mix_all_s0 intra_lrv1_full_L256_mix_all_s1 intra_lrv1_full_L256_mix_all_s2
```

`release/Kronos-Vol/` then holds the Hugging Face format: 1 safetensors file per seed, `config.json` (feature scaling,
target mean, a variance level set on the validation period only), `kronos_rv.py` and `kronos_model/`. A final check
confirms that the module reproduces the runs' own test forecasts.

**6. Use the published model.**

```
py -3.11 use_model.py AAPL NVDA JPM
```

It downloads Kronos-Vol from the Hugging Face Hub, takes the last 60 days of 5-minute bars of each name and SPY, and
prints the forecast for the session after the last complete one, as log variance and annualised volatility.
`--bars data/intraday` uses bars already on disk. In Python:

```python
from huggingface_hub import snapshot_download
from kronos_rv import KronosRV

m = KronosRV.from_dir(snapshot_download("ddedei/Kronos-Vol"))
print(m.forecast({"AAPL": aapl_bars}, spy_bars))     # 5-minute bars: ts (New York time), open, high, low, close, volume, vwap
```

**7. Watch it train.** [LooksFine](https://github.com/davidalmeida90/looksfine) reads `runs/` and `data/` from this folder:
`python server.py --root path/to/kronos_vol`, then http://127.0.0.1:8765.

## Files

| File | What it does |
|---|---|
| `download_bars.py` | 5-minute SIP bars from Alpaca, split and dividend adjusted, regular session only |
| `symbols.txt` | the universe: 200 S&P 500 stocks with the highest median dollar volume in 2021 among the names that were members from June 2021 on, plus SPY |
| `build_panel.py` | daily realised variance and the 9 features: log RV over 1, 5 and 22 sessions, the overnight gap, the last hour, the daily return, SPY's RV over 1, 5 and 22 sessions |
| `har.py` | the simple rules: yesterday's RV, HAR-RV, HAR with the extra features, LightGBM |
| `ft_intra.py` | the fine-tune, the control and the twins |
| `evaluate.py` | the held-out results table |
| `compare.py` | any runs side by side, Diebold-Mariano tests |
| `export_release.py` | seeds to the Hugging Face release format, with a reproduction check |
| `kronos_rv.py` | the inference module of Kronos-Vol (the same file as on the Hub) |
| `use_model.py` | forecasts from the published model |
| `evallib.py`, `kx.py` | paths, splits, the scores; Kronos loading and the input scaling of Kronos's own predictor |
| `kronos_model/` | Kronos model code, vendored from shiyu-coder/Kronos (MIT) |

Data and runs go to `data/` and `runs/` inside this folder, or under the folder named by `KRONOS_VOL_DIR`.

## Limitations

- Large, liquid US stocks only. Small caps, other markets, crypto and futures are untested.
- Stocks were picked among names that stayed large, a survivorship shortcut. It affects every model in the comparison
  the same way, but levels may be optimistic.
- Training variants (loss, pooling, context length) were screened on the held-out TEST stocks; the final recipe was then
  chosen on the validation period by a rule fixed in advance. Sessions after 10 Nov 2026 were never used for any choice.
- On SPY and QQQ, never trained on, the gain over HAR-RV comes from the head and its features: the control does as well.
- It forecasts the regular session only. Overnight gaps, earnings and macro releases are not in the target.
- A new download carries Alpaca's latest split and dividend adjustments, so a rebuilt panel can differ slightly from the
  one in the video (a later pull matched 98.4% of shared rows exactly; the rest differed by under 1% of log RV).
- Seeds on another GPU or driver can land a little apart; the export check is exact only on CUDA.

## Credits

Built on [Kronos](https://github.com/shiyu-coder/Kronos) by Shi et al. (2025), MIT license. `kronos_model/` is a copy of
its model code, commit 67b630e, with its license.

```bibtex
@misc{shi2025kronos,
  title={Kronos: A Foundation Model for the Language of Financial Markets},
  author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
  year={2025}, eprint={2508.02739}, archivePrefix={arXiv}, primaryClass={q-fin.ST},
  url={https://arxiv.org/abs/2508.02739}
}
```

Market data: Alpaca, SIP feed. Research and education only.
