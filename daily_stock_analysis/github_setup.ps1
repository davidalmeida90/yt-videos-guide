# The way the repo is meant to run: your own fork on GitHub, run by GitHub Actions every weekday, report by email.
# Needs the GitHub CLI (gh auth login once). Your keys go in as GitHub secrets from your shell's environment, so they are
# never typed on screen or saved in a file. Replace YOUR_USER with your GitHub user name.

gh repo fork ZhuLinsen/daily_stock_analysis --clone=false --default-branch-only
$r = 'YOUR_USER/daily_stock_analysis'

# secrets: the model key, the news key (optional), the email that sends the report and its app password
gh secret set ANTHROPIC_API_KEY --repo $r --body $env:ANTHROPIC_API_KEY
gh secret set TAVILY_API_KEYS --repo $r --body $env:TAVILY_API_KEYS
gh secret set EMAIL_SENDER --repo $r --body $env:EMAIL_SENDER          # e.g. you@gmail.com
gh secret set EMAIL_PASSWORD --repo $r --body $env:EMAIL_PASSWORD      # a Gmail app password, not your login password

# settings for US stocks in English
gh variable set STOCK_LIST --repo $r --body 'NVDA,AAPL,KO'
gh variable set LITELLM_MODEL --repo $r --body 'anthropic/claude-haiku-5-5'
gh variable set LLM_TEMPERATURE --repo $r --body '1.0'
gh variable set REPORT_LANGUAGE --repo $r --body 'en'
gh variable set MARKET_REVIEW_REGION --repo $r --body 'us'
gh variable set EMAIL_SENDER_NAME --repo $r --body 'Daily Stock Analysis'

# GitHub runs no workflow on a fork until you allow it: open https://github.com/$r/actions once and click
# "I understand my workflows, go ahead and enable them" (or push any commit), then:
gh workflow enable 00-daily-analysis.yml --repo $r
gh workflow run 00-daily-analysis.yml --repo $r -f mode=stocks-only -f force_run=true
Start-Sleep 10
gh run watch (gh run list --repo $r --workflow 00-daily-analysis.yml --limit 1 --json databaseId --jq '.[0].databaseId') --repo $r --exit-status

# Fully English email: in a clone of your fork, run en_patch.py once, then commit and push
#   git clone https://github.com/$r; cd daily_stock_analysis; python ..\en_patch.py; git commit -qam "English email"; git push
#
# Run it AFTER the US close: the default schedule is 10:00 UTC (6 pm Beijing, 5 or 6 am New York), before the US open,
# when yesterday's close is counted twice in the 5 day average. In .github/workflows/00-daily-analysis.yml change
#   - cron: '0 10 * * 1-5'
# to, for example, 21:30 UTC (4:30 or 5:30 pm New York):
#   - cron: '30 21 * * 1-5'
