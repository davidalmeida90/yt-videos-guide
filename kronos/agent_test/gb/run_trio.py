"""Once the DeepSeek pair is done: restart the scratch Glassbench with the fake variant and run plain, real and fake on Haiku, same 12 cells. Restarting earlier would mark live runs as failed.

    python run_trio.py

1 model across all 3 arms, so the arms compare cleanly. DeepSeek pair stays as a second model. Paid from the Anthropic API credit, capped per arm.
GLASSBENCH_PYTHON names the Python of the Glassbench venv, default is the Python running this script. Its restart step is Windows only.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
API = "http://127.0.0.1:8767"
PY = os.environ.get("GLASSBENCH_PYTHON") or sys.executable
MODEL, BUDGET = "claude-haiku-5-5", 1.5
TICKERS = ["NVDA", "AAPL", "JPM", "XOM"]


def call(path, body=None):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def wait(bid, label):
    t0 = time.time()
    while True:
        d = call(f"/api/backtests/{bid}")
        c = d.get("counts", {})
        print(f"{label} {d['status']} {c} spent {d['spent_usd']} {time.time() - t0:.0f}s", flush=True)
        if d["status"] in ("finished", "cancelled") or (d["status"] == "paused" and not c.get("running")):
            return d
        time.sleep(60)


def restart():
    ps = "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'scratch\\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -Confirm:$false }"
    subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], check=False)
    time.sleep(3)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore")
    log = open(HERE / "scratch.log", "a", encoding="utf-8")
    subprocess.Popen([PY, "scratch.py"], cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=0x00000008 | 0x08000000)  # detached, no window
    for _ in range(60):
        try:
            call("/api/backtests")
            print("scratch restarted", flush=True)
            return
        except Exception:
            time.sleep(2)
    raise SystemExit("scratch did not come back")


def main():
    pair = json.loads((HERE / "pair.json").read_text())
    d = wait(pair["kronos"], "deepseek kronos")
    restart()
    if d["status"] == "paused":
        print("deepseek kronos arm hit its budget, topping up to finish its cells:", d.get("note"), flush=True)
        call("/api/kronos_topup", {"id": pair["kronos"], "budget_usd": 0.85})
        call(f"/api/backtests/{pair['kronos']}/start", {})
        d = wait(pair["kronos"], "deepseek kronos")
    dates = sorted({i["date"] for i in d["items"]})
    state = HERE / "trio.json"
    ids = json.loads(state.read_text()) if state.exists() else call("/api/kronos_trio", {"tickers": TICKERS, "dates": dates, "budget_usd": BUDGET, "model": MODEL})
    state.write_text(json.dumps(ids))
    print("trio", ids, dates, flush=True)
    for arm in ("stock", "kronos", "fake"):
        if call(f"/api/backtests/{ids[arm]}")["status"] != "finished":
            call(f"/api/backtests/{ids[arm]}/start", {})
        r = wait(ids[arm], f"haiku {arm}")
        if r.get("counts", {}).get("failed", 0) > 2:
            raise SystemExit(f"stopping: {arm} has failures, see the runs. {r.get('note')}")
    print("TRIO DONE", flush=True)


if __name__ == "__main__":
    main()
