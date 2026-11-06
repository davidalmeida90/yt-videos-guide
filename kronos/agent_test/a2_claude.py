"""Same LLM test on Claude, because 1 live demo showed Claude rejecting a forecast that DeepSeek would have followed. Is following the forecast a DeepSeek habit or an LLM habit?

    python a2_claude.py --test 3
    python a2_claude.py

Identical prompts, data, dates and arms to a2_llm.py, test period only: no forecast, real Kronos, fake forecast. Sonnet 5.5, $2 / $10 per million tokens, paid from the Anthropic API credit, hard stop at $11. Some replies reason in prose before the JSON and were cut at 200 tokens, nearly all in the forecast arms, so they would have counted as flat; those are retried with 800 tokens, and the score uses the last parsed reply.
Added 5 Nov 2026 after seeing the DeepSeek results, so it is labelled as a follow up. Scored with a3_score's ruler by a3_claude.py. Key read from ANTHROPIC_API_KEY, never printed.
"""
import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

import a2_llm as m

HERE = Path(__file__).parent
MODEL, BUDGET, WORKERS = "claude-sonnet-5-5", 11.0, 8
OUT = HERE / "llm_calls_claude.jsonl"


def parse(text):
    found = re.search(r'"position"\s*:\s*"(long|flat|short)"', text, re.I)
    return (found.group(1).lower(), True) if found else ("flat", False)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--test", type=int, default=0)
    o = a.parse_args()
    client = anthropic.Anthropic(api_key=m.api_key("ANTHROPIC_API_KEY"), max_retries=6)
    jobs = [j for j in m.plan() if j["period"] == "test"]
    done, spent = set(), 0.0
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            x = json.loads(line)
            spent += x["cost"]
            if x["parsed"]:
                done.add(x["id"])
    todo = [j for j in jobs if j["id"] not in done]
    print(len(jobs), "planned,", len(todo), "to go", flush=True)
    if o.test:
        todo = todo[:o.test]
    lock = threading.Lock()
    state = {"spent": spent, "n": 0, "bad": 0}

    def work(j):
        if state["spent"] >= BUDGET:
            return
        r = client.messages.create(model=MODEL, max_tokens=800, system=m.SYSTEM, messages=[{"role": "user", "content": j["prompt"]}])
        text = "".join(b.text for b in r.content if b.type == "text")
        pos, ok = parse(text)
        cost = (r.usage.input_tokens * 2 + r.usage.output_tokens * 10) / 1e6
        rec = {k: j[k] for k in ("id", "ticker", "date", "period", "arm", "shown_up_share")}
        rec.update(position=pos, parsed=ok, reply=text, tokens_in=r.usage.input_tokens, tokens_out=r.usage.output_tokens, cached=0, cost=cost)
        with lock:
            state["spent"] += cost
            state["n"] += 1
            state["bad"] += not ok
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            if state["n"] % 300 == 0 or o.test:
                print(f"{state['n']} done, ${state['spent']:.3f}, {state['bad']} unparsed", flush=True)
                if o.test:
                    print(rec["arm"], text[:220].replace("\n", " "), flush=True)

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(work, todo))
    print(f"finished: {state['n']} new calls, ${state['spent']:.3f}, {state['bad']} unparsed" + (" STOPPED AT BUDGET" if state["spent"] >= BUDGET else ""), flush=True)


if __name__ == "__main__":
    main()
