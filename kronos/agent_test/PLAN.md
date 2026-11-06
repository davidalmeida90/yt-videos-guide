# Does an LLM trader do better with Kronos? Pre-registered rules

Written on 5 Nov 2026 before any LLM call. Nothing here changes after seeing results; any later change gets its own dated line.

## Question
Same LLM, same data, same dates: does adding a Kronos forecast change results, compared with no forecast and with a fake forecast?

## Universe and dates
- 12 large US names picked today: SPY, QQQ, NVDA, AAPL, MSFT, AMZN, JPM, XOM, UNH, WMT, TSLA, KO. Survivorship bias is present and affects every arm the same way.
- Decision every 5 sessions, on the daily close.
- TEST period: 1 Jul 2024 to 23 Oct 2026, after Kronos's stated June 2024 training cutoff.
- CONTROL period: 3 Jan 2023 to 28 Jun 2024, data Kronos may have trained on. Same code, same prompts. No-forecast and Kronos arms only.

## Kronos settings
Kronos-small, lookback 100, as shipped, 30 sampled paths, T=1.0, top_p=0.9, 5 sessions ahead. Lookback 100 had the lowest error in t3, so this is the setting most favourable to Kronos.

## Arms
1. Buy and hold, equal weight.
2. Kronos rule, no LLM: long if over 50% of paths end up, short if under 50%, flat at exactly 50%.
3. Floor: best single price feature, swept over 5, 20, 60 and 120 session momentum and price above the 50 session average, long or short by sign. The sweep runs on the same period, which flatters the floor, so beating it means more.
4. LLM, no forecast.
5. LLM + Kronos.
6. LLM + placebo: same wording, numbers taken from a random other asset and date, fixed seed.

## LLM
- deepseek-flash, temperature 0, one call per decision, JSON reply: long, flat or short.
- Names and dates hidden, prices rebased to 100, so the model cannot recall what happened.
- The prompt never says to follow or ignore the forecast, and arms 5 and 6 cannot be told apart from the text.

## Scoring (one ruler for every arm)
- Enter at the next session's open, exit at the open 5 sessions later.
- 5 bps per side on each change in position.
- Equal weight across the 12 names, weekly portfolio return.
- Report: total return, annualised Sharpe, max drawdown, hit rate on non-flat calls, and results by half year.
- Main test: paired t-stat of weekly returns, arm 5 minus arm 4. Second test: arm 5 minus arm 6.
- Also report: how often the LLM's call matches the forecast direction, in arm 5 and in arm 6. If they match about equally, the LLM follows any number it is given.

## Budget
Hard stop at $1.60 of DeepSeek spend, tracked from usage and checked against the account balance.

## What would count as Kronos helping
Arm 5 beats arm 4 and arm 6 with |t| above 2 in the TEST period, and that gap is not larger only in the CONTROL period. Anything less is reported as no evidence of help.

## Later additions (dated, after the first results)
- 5 Nov 2026, after seeing results: the Kronos rule made +33% while mostly short in a rising market, so raw return against buy and hold is the wrong question for it. Added, labelled post hoc: its alpha and beta against buy and hold, and a one-line reversal rule (long if price is under its 100 session average, short if over) as the comparison, because Kronos pulls forecasts back toward that average. Results in a4_posthoc.py. None of this changes the pre-registered verdict.
- Actual DeepSeek cost: $0.77 from the account balance. The upper bound in the log said $0.60 because the price table in a2_llm.py is out of date.
- 5 Nov 2026, after seeing results: a live take showed Claude rejecting a Kronos forecast, so "LLMs follow any number" may be a DeepSeek habit. Follow-up, labelled as such: the same prompts and arms on Claude Sonnet 5.5, test period, in a2_claude.py. Reported whatever it shows.
