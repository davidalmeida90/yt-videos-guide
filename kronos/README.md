# Kronos: code behind the video

Code behind a YouTube video that tests [Kronos](https://github.com/shiyu-coder/Kronos), an open source candlestick foundation model. Everything written for the video is here: the desk, the tests and the agents. Video link: coming soon.

**Research only. Nothing here places an order, and nothing here is investment advice.**

## What the tests found

Kronos runs well as a small tool, and no test here shows a proven edge. On 12 US names it trailed buy and hold, and a 1 line rule matched its return. LLMs followed a fake forecast about as often as the real forecast. A team of agents changed its ratings about as often as a plain rerun did.

Test period: 1 Jul 2024 to 23 Oct 2026, after the June 2024 training cutoff stated in the paper. Parentheses mean negative.

### Backtest, Kronos alone

117 weeks, 1,404 decisions, 5 basis points a side, entry at the next open.

| Rule | Total return | Sharpe |
|---|---:|---:|
| Buy and hold | +60.53% | 1.16 |
| Kronos rule: long or short by most paths | +33.35% | 0.90 |
| Opposite of last week, added after seeing results | +34.96% | 0.94 |
| Opposite of last month, added after seeing results | +3.60% | 0.17 |

* Buy and hold won. Kronos made 55% of its return and was short on 55.41% of its calls.
* Kronos pointed the same way as a 20 session reversal 76% of the time, and as "opposite of last week" 66% of the time. Kronos and "opposite of last week" have about the same total return, with a weekly correlation of 0.51.
* Through June 2024, on dates it may have trained on, the Kronos rule returned (8.56)% against +87.97% for buy and hold. No sign that it simply remembers.
* Its 10th to 90th percentile range held the real 5 session move 34.6% of the time. It should hold 80%.

### Kronos as shipped

* With the pinned `huggingface_hub`, Kronos runs in training mode at inference ([issue 415](https://github.com/shiyu-coder/Kronos/issues/415)). Fixing that made the forecast worse. Mean squared error against a no change forecast went from 6.30x to 9.42x with 400 bars, and from 2.52x to 3.10x with 100 bars.
* On daily candles, 6 tickers, it called a fall 80.8% of the time with 400 bars and 63.6% with 100 bars. Real share of falls: 42.8%. On the 33 hourly windows below it called a fall 39.4% of the time, so this bias is a daily bar result.
* Repo's web UI, filmed run: predicted +16.52%, actual +14.47%, which looks great. On 33 windows with the same defaults, repeating the last price had the lower error 25 times, 75.8%. Average error 9.21% against 5.26%.

### Agent 1: 1 Claude agent with Kronos as a tool

On the filmed NVDA run Claude Sonnet 5.5 chose to call Kronos, saw every path going up with a 10th percentile of +8.6%, and answered `POSITION: FLAT`, with the words "so I don't trust it". Kronos works as a live tool. This is 1 demo.

### Agent 2: forecast attached to 1 analyst in TradingAgents

Market analyst reads the forecast in its price tool, the rest of the team reads what that analyst writes, and nobody can call Kronos. 12 cases, 4 names on 3 dates. Rating changed against the plain run:

| Model | Real forecast | Fake forecast | Plain run again |
|---|---:|---:|---:|
| Claude Haiku | 5 of 12 | 6 of 12 | 4 of 12 |
| DeepSeek | 7 of 12 | not run | not run |

Ratings changed about as often with no forecast at all, so Kronos made no measurable difference to the team.

### Agent 3: forecast pasted into a prompt, 1,404 decisions

1 LLM with no tools. Names and dates are hidden and prices are rebased to 100, so the model cannot recall what happened. Same 1,404 decisions in each arm.

| | DeepSeek Flash | Claude Sonnet 5.5 |
|---|---:|---:|
| No forecast: return, Sharpe | (9.49)%, (0.31) | +10.72%, 0.48 |
| Real Kronos: return, Sharpe | +23.22%, 0.67 | +35.96%, 1.01 |
| Fake forecast: return, Sharpe | +3.01%, 0.21 | +2.54%, 0.18 |
| Call matched the real forecast | 84.2% | 81.6% |
| Call matched the fake forecast | 82.7% | 80.6% |
| t statistic, real against no forecast | 0.99 | 0.93 |
| t statistic, real against fake | 0.86 | 1.36 |

Rules written before any LLM call ([agent_test/PLAN.md](agent_test/PLAN.md)) ask for a t statistic above 2 against both. Neither model reaches it, and buy and hold made +60.53% on the same dates. Claude is a follow up, added after seeing the DeepSeek results. Same stock and day, XOM on 23 Oct 2026, gave 3 different trades on DeepSeek: flat with no forecast, short with the real forecast, long with the fake. That case is 1 of the 1,404, picked by a fixed rule.

### Paper sample

Table 10 of the paper reports +19.11% a year over CSI300, information ratio 1.38. An independent audit got +12.89%, 1.13. Through the repo's own backtest settings, in the same window, 1 line "recent losers" rules made:

| Window | 5 days | 10 days | 20 days | 60 days |
|---|---:|---:|---:|---:|
| Paper window, net | +2.65% | (1.31)% | +2.97% | +21.84% |
| Later window, 6 Jun 2025 to 4 Nov 2026, net | (4.89)% | (7.05)% | (14.74)% | (11.34)% |

Kronos-base on a sample of every 5th trading day, 86 days:

| Window | Rank IC, t | With distance from the 90 day mean removed, t |
|---|---:|---:|
| Paper window, 45 days | 0.044, 1.47 | 0.002, 0.12 |
| Later window to 2 Apr 2026, 41 days | 0.005, 0.21 | 0.008, 0.57 |

Remove that reversal rule and almost nothing is left. Reversal was working in the paper's window, and both Kronos and the rules faded after it. This says nothing against the paper's numbers for its own period.

### Forward test

[desk/ledger.jsonl](desk/ledger.jsonl) holds 24 dated forecasts for 5 Nov 2026, 12 names in 2 modes. Each line carries the hash of the line before it. First scoring date: 12 Nov 2026.

## Honest limits

* Backtest and LLM tests ran Kronos as shipped, in training mode. Only the desk uses the fix.
* "Opposite of last week" was added after seeing the results.
* 12 stocks really means 10 stocks and 2 ETFs (SPY, QQQ), picked on 5 Nov 2026, so survivorship applies to every arm.
* 12 Agent 2 cases are Claude Haiku. On DeepSeek only the real forecast ran. Agent 2's models see the ticker and date, so only rating changes count, never returns. 12 cases are few.
* Video line "one line in your prompt" rests on 1 demo and was not tested across the 1,404. Agent 1's system prompt asks Claude to weigh how far each tool can be trusted. Agent 3's prompt does not.
* Agent 1's tool hides `training_mode`, `device` and `seconds` from Claude, because in an earlier take Claude used `training_mode` as a reason to distrust the numbers.
* Paper sample is not a reproduction of Table 10. It uses every 5th trading day and a simpler ruler, and it stops on 2 Apr 2026 because the run was stopped. Its 60 day rule is the best of 9, picked after the fact.
* 2 of the 33 web UI windows overlap the filmed run.
* Filmed web UI run: the page draws 120 hourly bars as 5 days. They really span 23 days.

## What costs money

Every paid step has its saved output in this folder, so scoring needs no key.

| Step | Script | Cost, from the code and logs |
|---|---|---|
| Tool, desk, forward test | `kronos_tool.py`, `desk/` | Free |
| Backtest and scoring | `a0_data.py`, `a3_score.py`, `a3_claude.py`, `a4_posthoc.py`, `uses/u1_uses.py` | Free |
| New forecasts | `a1_forecasts.py` | Free, GPU |
| Paper sample | `paper_repro/` | Free, needs Qlib data |
| Agent 1 | `claude_tool_agent.py` | About $0.009 on the filmed run, the script's own estimate is a few cents |
| Agent 3 demo | `llm_three.py` | A fraction of a cent |
| Agent 3, DeepSeek | `a2_llm.py` | $0.77 by account balance, $0.60 in its log, 6,012 calls, stops at $1.60 |
| Agent 3, Claude | `a2_claude.py` | $10.21, 4,387 calls, stops at $11 |
| Agent 2, DeepSeek | `gb/run_pair.py` | $1.23, capped at $0.65 per arm |
| Agent 2, Claude Haiku | `gb/run_trio.py`, `gb/run_repeat.py` | $1.29, capped at $1.50 per arm |

## How to run it

Tested on Windows 11 with an AMD GPU. PyTorch ROCm reports it as `cuda`. A few steps are Windows only, marked below. After the install, commands start from this `kronos` folder.

### Install Kronos

Clone Kronos and this repository side by side. Kronos's location comes from `KRONOS_REPO`, and its default is a folder called `Kronos` beside this repository.

```powershell
git clone https://github.com/shiyu-coder/Kronos.git
git clone https://github.com/davidalmeida90/yt-videos-guide.git
cd Kronos
py -3.12 -m venv .venv              # Mac or Linux: python3.12 -m venv .venv
.venv\Scripts\Activate.ps1          #               source .venv/bin/activate
```

Install [the PyTorch build for your card](https://pytorch.org/get-started/locally/) in that venv, then:

```powershell
pip install -r requirements.txt
cd ..\yt-videos-guide\kronos
pip install -r requirements.txt
```

You do not need a big GPU. Kronos-small peaked at 335 MB for 30 paths on 100 bars, about 0.5 s on the GPU against 33.3 s on the CPU. Longer jobs lean harder on the GPU: 400 bars in and 120 out took 5.3 s, and on the CPU it had not finished after about 4 minutes. `uses/u4_vram.py` measures this.

Keys are read from the environment, [.env.example](.env.example) lists the names:

```powershell
$env:ANTHROPIC_API_KEY = "paste here"     # Mac or Linux: export ANTHROPIC_API_KEY=...
$env:DEEPSEEK_API_KEY = "paste here"
```

### Command line tool

1 ticker in, 1 JSON forecast out. Free, about 10 s on a GPU.

```powershell
python kronos_tool.py NVDA --horizon 5 --lookback 100 --paths 30
```

`upside_prob` is the share of paths ending higher, and `move_pct` holds the 10th, 50th and 90th percentile moves. It runs as shipped by default, `--eval` applies the fix. It uses the latest bar Yahoo returns, which can be a session still trading.

### Desk

Candles, tokens, the next candle as a distribution, paths drawn as they are generated, then a reveal of what happened. Free. Runs in eval mode on 127.0.0.1 and never opens a browser.

```powershell
cd desk
python server.py
```

Open http://127.0.0.1:7072 yourself. A date in the address replays the past: `http://127.0.0.1:7072/?ticker=NVDA&as_of=2026-10-09`. Forward test page: `http://127.0.0.1:7072/static/ledger.html`. `python daily_log.py` adds forecasts to the ledger, never edit old lines.

Filmed runs on NVDA, 15 sessions ahead, forecast median against actual: 9 Oct 2026 (10.8)% and (10.7)%, 27 Aug (12.5)% and (2.4)%, 16 Jul (12.0)% and +5.6%, 3 Jun (16.0)% and (8.7)%. Those dates come from a fixed rule.

### Agent 1: 1 Claude agent with Kronos as a tool

```powershell
cd agent_test
python claude_tool_agent.py NVDA
```

Needs `ANTHROPIC_API_KEY`. Claude gets 2 tools, a price summary and Kronos, and decides whether to call them. If this runs from a Python without Kronos, point `KRONOS_PYTHON` at the Kronos venv. Your key stays in this process, the forecast runs with a reduced environment.

### Agent 2: Kronos forecast attached to 1 analyst in TradingAgents, run inside Glassbench

Install [Glassbench](https://github.com/davidalmeida90/glassbench) and its TradingAgents engine by its own quick start, in a folder called `glassbench` beside this repository or wherever `GLASSBENCH_REPO` points. `gb/scratch.py` starts a 2nd Glassbench on 127.0.0.1:8767 with its own copy of the database, and adds the saved forecast from `forecasts.csv` to the price tool's output. No GPU needed. Run all of these with the Python of the Glassbench venv, with your keys set.

```powershell
cd agent_test\gb
python scratch.py           # in its own terminal, leave it running
python run_pair.py          # DeepSeek, plain and real, about 80 minutes
python run_trio.py          # Claude Haiku, plain, real and fake, about 45 minutes
python run_repeat.py        # Claude Haiku, plain again, about 15 minutes
python compare.py           # reads the finished batches, rewrites pair_results.csv
```

Run them in that order. `run_trio.py` restarts the server through PowerShell, Windows only, and it force stops any Python whose command line contains `scratch.py`. Saved ratings are in [pair_results.csv](agent_test/gb/pair_results.csv), and the costs are in `pair.log`, `trio.log` and `repeat.log`. Filmed XOM run on 27 Mar 2026 used DeepSeek, cost about $0.039, and was chosen to show the mechanism. It is not the measured result.

### Agent 3: forecast pasted into a prompt, 1,404 decisions

```powershell
cd agent_test
python a0_data.py                            # free, daily bars from Yahoo into data/
python show_prompt.py 2026-10-23 XOM         # free, prints the exact prompt
python a3_score.py                           # free, scores the saved DeepSeek calls
python a3_claude.py                          # free, scores the saved Claude calls too
python llm_three.py 2026-10-23 XOM           # needs DEEPSEEK_API_KEY, a fraction of a cent
```

`a2_llm.py` and `a2_claude.py` make the paid calls. Both resume from their saved files, so with `llm_calls.jsonl` and `llm_calls_claude.jsonl` in place they find nothing left to do, and `a2_llm.py --test 3` stops with an index error. Move those files away to pay for a new run.

### Backtest

```powershell
cd agent_test
python a0_data.py
python a3_score.py          # every arm, writes results.csv and halfyears.csv
python a4_posthoc.py        # alpha, beta and the reversal rules
cd ..\uses
python u1_uses.py           # range and volatility
```

`a1_forecasts.py` rebuilds `forecasts.csv` on the GPU, 2,304 windows of 30 paths each, and skips rows already in that file. Sampled paths can differ on another machine. Prices from Yahoo are adjusted, so a later download can move results a little. On 6 Nov 2026 a new download gave the same numbers as the filmed run.

### Paper sample

`p5_sample.py` needs `data_csi300.csv`, which comes from Qlib and is not included. Its other input, `signals_csi300_base.csv`, is here.

```powershell
pip install pyqlib                # in a separate venv
```

Put Qlib's `cn_data` set in `~/.qlib/qlib_data/cn_data`, see [Qlib](https://github.com/microsoft/qlib). This test used the chenditc build, release 2026-11-04, and another release can change the numbers. Then:

```powershell
cd paper_repro
python p1_data.py csi300    # Qlib venv, writes data_csi300.csv
python p5_sample.py         # Kronos venv, the sample, free
python p4_floor.py          # Qlib venv, 1 line rules through the repo's backtest
```

`p4_floor.py` adds rows to `floor_csi300.csv`, so move that file away first. `p2_infer.py` rebuilds the signals on the GPU, free, Windows only as written. A full CSI300 run is 161,128 windows. Early in the run its log estimated 24.6 hours for the last 137,476, and it was stopped at the sample. Reasons and dated decisions are in [paper_repro/PLAN.md](paper_repro/PLAN.md).

### Other tests

| Script | What it does |
|---|---|
| `t1_as_shipped.py` | README flow on the repo's own sample, checks issue 415 |
| `t2_visuals.py`, `t5_boxes.py` | Candles and the fan as charts in `out/` |
| `t3_repro.py` | Issues 415 and 422 by lookback, writes `out/t3_results.csv` |
| `t4_webui.py` | Drives the repo's web UI headless, writes `NVDA_1h.csv` into the Kronos clone |
| `uses/u2_surprise.py` | How unlikely the model found each real candle, in bits |
| `uses/u3_synthetic.py` | Do sampled paths behave like real ones |
| `uses/u5_webui_default.py` | Web UI defaults on 33 windows, reads that `NVDA_1h.csv` |

## Credits

* [Kronos](https://github.com/shiyu-coder/Kronos), the model and repo, with [its paper](https://arxiv.org/abs/2508.02739) and [Kronos-small on Hugging Face](https://huggingface.co/NeoQuasar/Kronos-small).
* [Issue 415](https://github.com/shiyu-coder/Kronos/issues/415), training mode at inference, and [issue 422](https://github.com/shiyu-coder/Kronos/issues/422), forecasts pulled back toward the recent price level. Credit to their authors.
* [Independent audit by apmoore499](https://github.com/apmoore499/kronos-oos-edge-audit).
* [TradingAgents](https://github.com/TauricResearch/TradingAgents), the agent team in Agent 2.
* [Glassbench](https://github.com/davidalmeida90/glassbench), the app that runs and records Agent 2.
* [Microsoft Qlib](https://github.com/microsoft/qlib), data and backtest settings for the paper sample.
* `desk/static/lightweight-charts.js` is [TradingView](https://www.tradingview.com/) Lightweight Charts 4.2.3, under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

Code in this repository is under the MIT license, see [LICENSE](../LICENSE). Kronos has its own license in its repo.

Research only. Nothing here places an order, and nothing here is investment advice.
