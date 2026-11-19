# Microsoft Qlib on the S&P 500

Code for the video "Qlib: Microsoft's AI for Quant Research, Tested on Real Data": https://youtu.be/SxEduv5e_nE

Qlib 0.9.7 from pip, on Windows, with a point in time S&P 500 dataset built from free data, Microsoft's own LightGBM recipe, and a race against simple rules.

Research only. None of these strategies is proven to work, the dataset is about 10 years of one market, and nothing here is
investment advice.

## Install (Windows, CPU only)

Qlib's pip wheels stop at Python 3.12, so use Python 3.11.

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -c "import qlib; print(qlib.__version__)"
```

Two environment variables make the first `qrun` work on a fresh install:

```
$env:MLFLOW_ALLOW_FILE_STORE = "true"            # MLflow refuses its file store otherwise
$env:GIT_CEILING_DIRECTORIES = (Get-Location).Path   # Qlib logs a git diff of the working folder; this stops it
```

## Data: every S&P 500 member since 2016

1. Membership spells: download `sp500_ticker_start_end.csv` from https://github.com/fja05680/sp500 into `us_data/`.
2. Prices: a free Alpaca account gives the market data API. Set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`, then
   `python scripts/alpaca_download.py` (adjusted daily bars, SIP feed, renamed and delisted tickers requested with `asof`).
3. Qlib format: copy `scripts/dump_bin.py` from https://github.com/microsoft/qlib into `qlib_files/scripts_dump_bin.py`, then
   `python scripts/build_us_qlib.py`. Output goes to `~/.qlib/qlib_data/us_alpaca`, with `instruments/sp500.txt` holding the
   members of each day.

Membership spells in that file run to 18 Aug 2026; later index changes are not in the data.

## Run

```
mkdir runs; cd runs
qrun ..\configs\us_lgb_recent.yaml
```

`configs/us_lgb_recent.yaml` is Microsoft's LightGBM on Alpha158 benchmark file (`configs/cn_lgb_official.yaml`, as shipped)
with only the market and the dates changed: train 2016 to 2021, validation 2022 to 2023, test 2 Jan 2024 to 12 Nov 2026.
Strategy: TopkDropout, hold 50, swap at most 5 a day, trade at the close, 5 basis points each way.

## Scripts

| Script | What it does |
|---|---|
| `alpaca_download.py` | daily bars for every S&P 500 member since 2016, plus SPY, QQQ and RSP |
| `build_us_qlib.py` | CSVs to Qlib's binary format and a point in time `sp500` universe |
| `peek.py` | reads the data through Qlib's expression engine (close, 20 day average, the label) |
| `report.py` | IC, Qlib's excess return table, and after cost performance against SPY, RSP and QQQ |
| `qlib_charts.py` | Qlib's own report charts for a finished run, as offline HTML |
| `inside.py` | how many different scores the model gives per day, and what the top 50 ends up holding |
| `race.py` | 9 backtests through the same strategy: 3 versions of LightGBM, 3 one line rules, 3 random portfolios |

## Results (test 2 Jan 2024 to 12 Nov 2026, after costs)

| Strategy | IC | Per year | Sharpe | Max drawdown |
|---|---|---|---|---|
| LightGBM as released | 0.0037 | +15.0% | 0.96 | 20.9% |
| LightGBM, learning rate 0.05 | 0.0037 | +15.0% | 0.96 | 20.9% |
| LightGBM, 300 trees | 0.0018 | +17.6% | 0.98 | 22.7% |
| Momentum 12-1 | 0.0226 | +28.6% | 1.00 | 32.3% |
| Reversal 5 day | 0.0120 | +17.9% | 0.85 | 17.5% |
| Low volatility | (0.0091) | +10.8% | 1.00 | 8.9% |
| Random (median seed) | | +9.8% | 0.72 | 17.8% |
| SPY (buy and hold) | | +19.0% | 1.19 | 18.8% |
| RSP (buy and hold) | | +14.2% | 1.03 | 17.8% |
| QQQ (buy and hold) | | +20.5% | 0.98 | 22.8% |

LightGBM stopped at its first tree (best iteration 1): every later tree made the validation error worse. A one tree model
gives only about 18 different scores a day across 502 stocks, so the top 50 is full of ties, and the ties favoured names
early in the alphabet.
