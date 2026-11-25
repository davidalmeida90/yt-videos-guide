"""Rebuild saved daily_stock_analysis reports and render them the way the repo's email notification does.

The daily report file is overwritten by every run on the same date, so each saved analysis is rebuilt from the database
with the repo's own HistoryService.get_markdown_report, then turned into HTML with its email formatter
(src/formatters.markdown_to_html_document). Output: report_html/<id>_<code>.html, ready for a page take.

    .venv/Scripts/python.exe render_report.py 1 3 7          # database ids
    .venv/Scripts/python.exe render_report.py --list
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("DSA_REPO", HERE / "daily_stock_analysis"))   # your clone of the repo
os.chdir(REPO)
sys.path.insert(0, str(REPO))
import logging  # noqa: E402

logging.disable(logging.WARNING)
from src.config import setup_env  # noqa: E402

setup_env()
from src.formatters import markdown_to_html_document, strip_hidden_markdown_metadata  # noqa: E402
from src.services.history_service import HistoryService  # noqa: E402

OUT = HERE / "report_html"


def main():
    args = sys.argv[1:]
    db = sqlite3.connect(REPO / "data" / "stock_analysis.db")
    if not args or args == ["--list"]:
        for r in db.execute("select id, code, sentiment_score, operation_advice, created_at from analysis_history order by id"):
            print(*r)
        return
    OUT.mkdir(exist_ok=True)
    hs = HistoryService()
    for rid in args:
        code = db.execute("select code from analysis_history where id=?", (int(rid),)).fetchone()[0]
        md = hs.get_markdown_report(str(rid))
        html = markdown_to_html_document(strip_hidden_markdown_metadata(md))
        f = OUT / f"{int(rid):03d}_{code}.html"
        f.write_text(html, encoding="utf-8")
        (OUT / f"{int(rid):03d}_{code}.md").write_text(md, encoding="utf-8")
        print(f"{f.name}: {len(md)} chars of markdown")


if __name__ == "__main__":
    main()
