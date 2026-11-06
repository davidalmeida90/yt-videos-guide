# Reproducing the Kronos paper's stock ranking result

Written 5 Nov 2026 before any inference on this data.

## What the paper claims
Table 10, Kronos-base, annualised excess return and information ratio: CSI300 +19.11% / 1.38, CSI800 +18.67% / 1.67. The audit (apmoore499/kronos-oos-edge-audit) got +12.89% / 1.13 and +13.46% / 1.43 with the paper-text signal, on the same window, and stopped there on purpose.

## Setup copied from the repo (finetune/config.py, qlib_data_preprocess.py, qlib_test.py)
- Data: qlib cn_data, chenditc build, release 2026-11-04.
- Universe: csi300 through QlibDataLoader, so membership spans come from qlib's instruments file.
- Features: open, high, low, close, vol, amt, with amt = mean(ohlc) * volume. Lookback 90, predict 10, clip 5.
- Normalisation on the context only. Time features: minute, hour, weekday, day, month.
- auto_regressive_inference with T 0.6, top_p 0.9, top_k 0, sample_count 5, max_context 512, modules in eval mode as qlib_test.py does.
- Signals as in qlib_test.py, in normalised space: last, mean, max, min. Plus the paper-text reading avg_return = mean(predicted raw close over 10 days) / last close - 1.
- Backtest: TopkDropoutStrategy topk 50, n_drop 5, hold_thresh 5; deal price open, delay execution, limit 9.5%, open cost 0.10%, close cost 0.15%, min cost 5; account 100M; benchmark SH000300.

## Windows
- A, the paper's: 1 Jul 2024 to 5 Jun 2025. Should come out near the audit's numbers. If not, the pipeline is suspect.
- B, new: 6 Jun 2025 to the last date with data. Neither the paper nor the audit tested this.
Window B is reported whatever it shows.

## Floor (one ruler, swept, never hand picked)
Same backtest and same rank IC for one-line signals built from prices only: 5, 10, 20 and 60 day reversal, and distance from the 90 day mean, raw and in standard deviations (the same normalised space Kronos sees), each with its opposite sign. The best one in each window is the floor.

## Hypothesis to check, stated before the numbers
Kronos pulls forecasts back toward the window mean (issue 422; seen in t3). Chinese large caps have strong short-term reversal. So part or all of the ranking edge could be reversal. Measured by: daily rank correlation between the Kronos signal and each floor signal, and the rank IC left in Kronos after removing the floor signal cross-sectionally each day.

## Reported
Rank IC by horizon 1 to 10 with Newey-West t-stat using lag h-1; AER and IR with and without cost; by window; by half year.

## Model order
Kronos-small first if base is too slow for the machine, labelled as small. The paper's number is for base, so a comparison with Table 10 only counts on base.

## Dated lines
- 5 Nov 2026, before any Kronos backtest on this data, after the floor backtest (floor_csi300.csv) and a first 25 days of real signals: floor rules in the paper window made rev60 +21.84% (IR 1.59) and dist90 +15.98% (IR 1.10) net of costs; random -2.68%. In window B: rev60 -11.34%, dist90 -12.89%, random +8.05%, opposite of rev20 +22.01%. Kronos avg_return has daily rank correlation +0.73 with z90 and +0.68 with dist90 over those 25 days. Prediction written down now: if Kronos is mostly reversal, it should do well in window A and badly in window B, and it should add little once the floor signal is removed. Reported whatever it shows.
- The first CSI300 run (signals_csi300_base.BAD_gpu_zeros.csv) returned zeros from GPU resets and was discarded. Its "rank correlation +1.000 with dist90" was that bug, not a finding.
- 6 Nov 2026, decision: no full daily run. The video talks about the paper and shows a sample: Kronos-base on every 5th trading day, scored by p5_sample.py with one ruler against the one-line rules on the same days. It can't reproduce Table 10 or the TopK backtest, so it is never called a reproduction. floor_csi300.csv stays valid for the one-line rules on their own.
- 6 Nov 2026: stopped by decision, using what was done. The sample ends where the file ends, so window B is shorter than the 16 months available.
