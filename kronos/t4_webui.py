"""Test 4: drive the repo's own web UI headless and film it as screenshots. Never opens a window.

Starts webui/app.py on 127.0.0.1 only, with no reloader and no browser (the repo's run.py opens one). Needs a data file
in <repo>/data; this script writes NVDA hourly there. Run with a Python that has Playwright. KRONOS_PYTHON names the Python
of the Kronos venv for the server, default is the Python running this script.

    python t4_webui.py
"""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(os.environ.get("KRONOS_REPO") or Path(__file__).resolve().parents[2] / "Kronos")
if not (REPO / "model").is_dir():
    raise SystemExit(f"Kronos clone not found at {REPO}. Clone github.com/shiyu-coder/Kronos there or set KRONOS_REPO.")
PY = os.environ.get("KRONOS_PYTHON") or sys.executable
OUT = Path(__file__).parent / "out"
PORT = 7071

BOOT = f"""
import sys, os
sys.path.insert(0, r"{REPO}")
sys.path.insert(0, r"{REPO / 'webui'}")
os.chdir(r"{REPO / 'webui'}")
import app as a
a.app.run(debug=False, use_reloader=False, host="127.0.0.1", port={PORT})
"""

DATA = f"""
import sys
sys.path.insert(0, r"{Path(__file__).parent}")
import kr
from pathlib import Path
d = kr.bars("NVDA", "1h", period="700d")
Path(r"{REPO / 'data'}").mkdir(exist_ok=True)
d.reset_index().to_csv(r"{REPO / 'data' / 'NVDA_1h.csv'}", index=False)
print("rows", len(d), d.index[0], d.index[-1])
"""


def main():
    print(subprocess.run([str(PY), "-c", DATA], capture_output=True, text=True, timeout=120).stdout.strip())
    log = open(OUT / "t4_server.log", "w", encoding="utf-8")
    srv = subprocess.Popen([str(PY), "-c", BOOT], stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(90):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=2)
                break
            except Exception:
                time.sleep(1)
        else:
            raise SystemExit("server did not start, see out/t4_server.log")
        with sync_playwright() as p:
            b = p.chromium.launch(headless=True)
            pg = b.new_page(viewport={"width": 1600, "height": 1000})
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")
            pg.screenshot(path=str(OUT / "t4_1_open.png"))
            print("devices offered:", pg.locator("#device-select option").all_inner_texts())
            print("models offered:", pg.locator("#model-select option").all_inner_texts())
            print("data files:", pg.locator("#data-file-select option").all_inner_texts())
            pg.select_option("#model-select", value="kronos-small")
            opts = pg.locator("#device-select option").evaluate_all("o => o.map(x => x.value)")
            print("device values:", opts)
            pg.select_option("#device-select", value="cuda" if "cuda" in opts else opts[0])
            pg.click("#load-model-btn")
            pg.wait_for_function("document.querySelector('#model-status').innerText.length > 5", timeout=180000)
            time.sleep(1)
            print("model status:", pg.locator("#model-status").inner_text())
            pg.select_option("#data-file-select", index=1)
            pg.click("#load-data-btn")
            pg.wait_for_selector("#data-info", state="visible", timeout=60000)
            print("data:", pg.locator("#data-rows").inner_text(), "rows |", pg.locator("#data-time-range").inner_text(),
                  "|", pg.locator("#data-timeframe").inner_text())
            pg.screenshot(path=str(OUT / "t4_2_loaded.png"), full_page=True)
            pg.wait_for_selector("#predict-btn:not([disabled])", timeout=60000)
            t = time.time()
            pg.click("#predict-btn")
            pg.wait_for_function("document.querySelector('#chart') && document.querySelector('#chart').children.length > 0",
                                 timeout=300000)
            pg.wait_for_timeout(1500)
            print(f"prediction through the UI: {time.time() - t:.1f} s")
            pg.screenshot(path=str(OUT / "t4_3_predicted.png"), full_page=True)
            pg.locator("#chart").screenshot(path=str(OUT / "t4_4_chart.png"))
            if pg.locator("#comparison-section").is_visible():
                print("MAE", pg.locator("#mae").inner_text(), "RMSE", pg.locator("#rmse").inner_text(), "MAPE",
                      pg.locator("#mape").inner_text())
            print("page errors:", errs)
            b.close()
    finally:
        srv.terminate()
        srv.wait(timeout=15)
        log.close()


if __name__ == "__main__":
    main()
