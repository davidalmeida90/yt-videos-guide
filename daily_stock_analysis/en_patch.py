"""English email patch for daily_stock_analysis (ZhuLinsen/daily_stock_analysis, checked on commit 603a55e, 22 Nov 2026).

With REPORT_LANGUAGE=en the report body is already English, but a few strings are hard-coded in Chinese and reach the
email anyway. This script changes only those, so the daily email is fully English:

  1. the email subject "📈 股票智能分析报告" -> "📈 Stock Analysis Report"
  2. the market snapshot volume: it called the volume formatter without the report language, so it printed 亿股/万股
     even in English reports (the English branch already existed; this passes the language)
  3. money in the financial summary: 亿美元/万美元 -> B/M USD (English reports only)
  4. dividends per share: 美元 -> USD (English reports only)
  5. institutional net shares: 亿股/万股 -> B/M shares (English reports only)
  6. the full-width colon after "Data Sources", "Generated At" and "Model"
  7. the full-width semicolon joining the two confidence sentences (English reports only)

The sender name needs no code: set EMAIL_SENDER_NAME (a GitHub variable or .env line).
Run from the repo folder:   python en_patch.py      (safe to run twice: it skips what is already patched)
"""
from pathlib import Path

MARK = "en_patch.py"
NL = chr(10)
HELPER = '''

def _english_report() -> bool:  # en_patch.py: English units in English reports
    try:
        return str(getattr(get_config(), "report_language", "zh")).lower().startswith("en")
    except Exception:
        return False
'''

EDITS = {
    "src/notification_sender/email_sender.py": [
        ('subject = f"📈 股票智能分析报告 - {date_str}"', 'subject = f"📈 Stock Analysis Report - {date_str}"', 2),
    ],
    "src/analyzer.py": [
        ("\"volume\": self._format_volume(today.get('volume')),",
         "\"volume\": self._format_volume(today.get('volume'), getattr(self._get_runtime_config(), \"report_language\", \"zh\")),  # en_patch.py", 1),
        ("\"amount\": self._format_amount(today.get('amount')),",
         "\"amount\": self._format_amount(today.get('amount'), getattr(self._get_runtime_config(), \"report_language\", \"zh\")),  # en_patch.py", 1),
    ],
    "src/phase_decision_guardrail.py": [
        ('        phase_decision["confidence_reason"] = f"{existing}；{reason}"' + NL,
         '        sep = "；"  # en_patch.py: English reports join the two sentences with a space' + NL
         + '        try:' + NL
         + '            from src.config import get_config' + NL
         + '            if str(getattr(get_config(), "report_language", "zh")).lower().startswith("en"):' + NL
         + '                sep = " "' + NL
         + '        except Exception:' + NL
         + '            pass' + NL
         + '        phase_decision["confidence_reason"] = f"{existing}{sep}{reason}"' + NL, 1),
    ],
    "src/notification.py": [
        ("from src.config import Config, get_config\n", "from src.config import Config, get_config\n" + HELPER + "\n", 1),
        ("        # Compare the rounded figure too, so 99_999_999 reads 1.00 亿 rather than 10000.00 万.\n",
         "        if _english_report():  # en_patch.py\n"
         "            code = (currency or \"\").upper() or \"CNY\"\n"
         "            for div, unit in ((1e12, \"T\"), (1e9, \"B\"), (1e6, \"M\"), (1e3, \"K\")):\n"
         "                if abs_amount >= div:\n"
         "                    return f\"{sign}{abs_amount / div:.2f}{unit} {code}\"\n"
         "            return f\"{sign}{abs_amount:.0f} {code}\"\n"
         "        # Compare the rounded figure too, so 99_999_999 reads 1.00 亿 rather than 10000.00 万.\n", 1),
        ('        return f"{amount:.4f} {suffix}"\n',
         '        return f"{amount:.4f} {((currency or \'\').upper() or suffix) if _english_report() else suffix}"  # en_patch.py\n', 1),
        ("        # Compare the rounded figure too, so 99_999_999 reads 1.00 亿股 rather than 10000.00 万股.\n",
         "        if _english_report():  # en_patch.py\n"
         "            for div, unit in ((1e9, \"B\"), (1e6, \"M\"), (1e3, \"K\")):\n"
         "                if a >= div:\n"
         "                    return f\"{sign}{a / div:.2f}{unit} shares\"\n"
         "            return f\"{sign}{a:.0f} shares\"\n"
         "        # Compare the rounded figure too, so 99_999_999 reads 1.00 亿股 rather than 10000.00 万股.\n", 1),
        ("{labels['data_sources_label']}：{", "{labels['data_sources_label']}: {", 1),
        ("{labels['generated_at_label']}：{", "{labels['generated_at_label']}: {", 3),
        ("report_lines.append(f\"*{labels['analysis_model_label']}：{', '.join(models)}*\")",
         "report_lines.append(f\"*{labels['analysis_model_label']}: {', '.join(models)}*\")", 1),
    ],
}


def main():
    for rel, edits in EDITS.items():
        p = Path(rel)
        with open(p, encoding="utf-8", newline="") as fh:        # keep the file's own line endings
            text = fh.read()
        eol = "\r\n" if "\r\n" in text else "\n"
        changed = 0
        for old, new, count in edits:
            old, new = old.replace("\n", eol), new.replace("\n", eol)
            if new in text:
                continue                                        # already patched
            found = text.count(old)
            if found != count:
                raise SystemExit(f"{rel}: expected {count} of {old[:60]!r}, found {found}; the upstream code changed, check by hand")
            text = text.replace(old, new)
            changed += 1
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        print(f"{rel}: {changed} change(s)")


if __name__ == "__main__":
    main()
