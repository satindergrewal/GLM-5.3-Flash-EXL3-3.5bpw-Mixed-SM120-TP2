#!/usr/bin/env python3
"""Real-batching concurrency matrix for the 3.5bpw mixed serve.

Protocol parity with bench_final.py forced-gen numbers: non-stream,
ignore_eos, forced 1024 completion tokens, thinking off, temp 0.
Runs against a serve booted with max_num_seqs >= 8 so the streams
actually batch inside the engine (earlier matrix ran on seqs=1).

Sections: single x3 | c=2 | c=4 | c=6 (8K unique prompts)
Output: /mnt/t5evo/glm53-k35-work/eval/conc-real.json
"""
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path

BASE = "http://localhost:8012"
MODEL = "GLM-5.3-Flash-EXL3-3.5bpw"
OUT = Path("/mnt/t5evo/glm53-k35-work/eval/conc-real.json")
FILLER = ("The quick brown fox jumps over the lazy dog. "
          "Pack my box with five dozen liquor jugs. "
          "How vexingly quick daft zebras jump! ") * 40


def post(path, body, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def sized_prompt(seed, target=8000):
    p = f"[prompt {seed}] " + FILLER
    while True:
        n = len(post("/tokenize", {"model": MODEL, "prompt": p})["tokens"])
        if n >= target:
            return p, n
        p += FILLER


def forced_gen(prompt, max_tokens=1024):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    r = post("/v1/chat/completions", body)
    wall = time.time() - t0
    comp = r["usage"]["completion_tokens"]
    return {"wall_s": round(wall, 2), "completion_tokens": comp,
            "tok_s": round(comp / wall, 1)}


def main():
    out = {"time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
           "protocol": "non-stream, ignore_eos, forced 1024, thinking off, 8K prompts",
           "sections": {}}
    print("warmup...", flush=True)
    forced_gen("Warmup request. Answer with one word: ok.", 32)

    print("single x3...", flush=True)
    out["sections"]["single"] = [
        forced_gen(f"Count the words in this text and reply with the count only. {FILLER[:2000]} {i}")
        for i in range(3)]

    for n in (2, 4, 6):
        print(f"c={n}...", flush=True)
        prompts = [sized_prompt(i)[0] for i in range(n)]
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
            rs = list(ex.map(lambda p: forced_gen(
                f"Summarize this text in one sentence. {p}"), prompts))
        wall = time.time() - t0
        total = sum(r["completion_tokens"] for r in rs)
        out["sections"][f"concurrency_{n}"] = {
            "wall_s": round(wall, 2),
            "total_tokens": total,
            "aggregate_tok_s": round(total / wall, 1),
            "per_stream_tok_s": [r["tok_s"] for r in rs],
            "per_stream_wall_s": [r["wall_s"] for r in rs],
        }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
