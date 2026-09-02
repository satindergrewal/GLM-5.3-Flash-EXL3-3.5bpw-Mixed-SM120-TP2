#!/usr/bin/env python3
"""Thinking-ON single-stream decode rate: 3 runs, forced 1024.

Protocol parity with bench_conc.py but enable_thinking=true; the forced
tokens are consumed largely by reasoning, so the rate is the honest
thinking-on decode rate (non-stream wall, ignore_eos).
Output: eval/think-on.json
"""
import json
import time
import urllib.request
from pathlib import Path

BASE = "http://localhost:8012"
MODEL = "GLM-5.3-Flash-EXL3-3.5bpw"
OUT = Path("/mnt/t5evo/glm53-k35-work/eval/think-on.json")


def post(path, body, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def forced_gen(prompt, max_tokens=1024):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": True}}
    t0 = time.time()
    r = post("/v1/chat/completions", body)
    wall = time.time() - t0
    comp = r["usage"]["completion_tokens"]
    return {"wall_s": round(wall, 2), "completion_tokens": comp,
            "tok_s": round(comp / wall, 1)}


def main():
    runs = [forced_gen(
        "Solve step by step: a train leaves Auckland at 08:00 at 90 km/h, "
        "another leaves Hamilton at 08:30 at 110 km/h toward Auckland. "
        "When do they meet? Reason, then answer.") for _ in range(3)]
    out = {"time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
           "protocol": "non-stream, ignore_eos, forced 1024, thinking ON",
           "single_thinking_on": runs}
    OUT.write_text(json.dumps(out, indent=1))
    print("THINK-ON:", [r["tok_s"] for r in runs])


if __name__ == "__main__":
    main()
