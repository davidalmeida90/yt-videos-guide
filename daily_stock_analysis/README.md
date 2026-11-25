# daily_stock_analysis on US stocks, in English

Code from the video testing [ZhuLinsen/daily_stock_analysis](https://github.com/ZhuLinsen/daily_stock_analysis) (MIT), an AI
stock analyst built for Chinese A-shares, run here on US stocks with Claude Haiku 5.5 and DeepSeek V4.1 Flash. The repo's own
code is untouched: these are the settings and small scripts around it.

Nothing here is investment advice. These are tests of the tool.

| file | what it does |
|---|---|
| `.env.us.example` | the settings that turn it to US stocks and English (watchlist, language, market, model, sender name) |
| `run.py` | runs it on your PC with the API keys taken from your shell, never from a file; `--model deepseek` switches the model |
| `github_setup.ps1` | the intended setup: fork, secrets, settings, enable and run the GitHub Actions workflow, email report |
| `en_patch.py` | makes the email fully English: the subject, the number units (亿 / 万) and the punctuation are hard-coded in Chinese |
| `render_report.py` | rebuilds a saved report from the repo's database and renders it as the repo's email HTML |
| `engine_backtest.py` | replays the repo's Python score (src/stock_analyzer.py) day by day on point-in-time S&P 500 members |

## On your PC

```powershell
git clone https://github.com/ZhuLinsen/daily_stock_analysis.git
cd daily_stock_analysis
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env          # then apply the lines in .env.us.example
cd ..
$env:ANTHROPIC_API_KEY = "..."       # and TAVILY_API_KEYS for news, OPENROUTER_API_KEY for DeepSeek
python run.py --stocks NVDA --no-notify --no-market-review --force-run
```

About 1 GB, a few minutes to install. No GPU. In the video's runs a stock cost about 0.4 cents with Claude Haiku 5.5 and about
0.15 cents with DeepSeek V4.1 Flash through OpenRouter (list prices); the agent mode about 3 cents.

## On GitHub, every weekday, by email

See `github_setup.ps1`. Two things to change for US stocks:

1. **Run it after the US close.** The default schedule is 10:00 UTC, before the US open, and the repo then counts the
   previous close twice in the 5 day average (on 23 Nov 2026 that moved Apple from reduce to watch on the same prices).
2. **English email.** Run `en_patch.py` once in a clone of your fork and push it; set `EMAIL_SENDER_NAME`.

GitHub pauses scheduled workflows on a public fork after 60 days without activity.

## The 10 year check

`engine_backtest.py` scores random samples of 100 S&P 500 members (including the ones that left the index) every day from
2017 to Nov 2026 with the repo's own engine, signal at the close, trade at the next open, 5 bps a side. It needs daily bars in
`us_data/raw/<SYMBOL>.csv` and the membership file `us_data/sp500_ticker_start_end.csv` (from fja05680/sp500). The video used
`--n 100 --seed 7` and `--n 100 --seed 11`. Only the Python score is replayed: the AI part can't be backtested, the models already know what happened next.
