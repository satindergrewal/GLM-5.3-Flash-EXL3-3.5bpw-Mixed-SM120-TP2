#!/usr/bin/env python3
"""Bench the running 3.5bpw serve: thinking-on singles, tool call, spec check.

  python3 bench/bench.py [--base http://localhost:8012] [--runs 3]
"""
import argparse, json, statistics, time, urllib.request

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8012")
    ap.add_argument("--model", default="GLM-5.3-Flash-EXL3-3.5bpw")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()
    url = args.base.rstrip("/") + "/v1/chat/completions"

    def call(body):
        t0 = time.time()
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.load(r)
        dt = time.time() - t0
        u = d.get("usage", {})
        comp = u.get("completion_tokens", 0)
        return {"tok": comp, "wall_s": round(dt, 2),
                "tps": round(comp / dt, 1) if dt else 0,
                "finish": d["choices"][0].get("finish_reason", "?")}

    prompts = [
        ("prose", "Explain three practical differences between TCP and UDP, one short paragraph each."),
        ("code", "Write a Python function that merges two sorted lists in O(n), with a test."),
        ("reason", "A shop sells pens at 3 for $2. How much do 18 pens cost? Think it through."),
    ]
    print(f"{'kind':8} {'tok':>6} {'wall_s':>7} {'tok/s':>7}  finish")
    for name, p in prompts:
        rates = []
        for _ in range(args.runs):
            r = call({"model": args.model, "max_tokens": args.max_tokens,
                      "messages": [{"role": "user", "content": p}], "temperature": 0})
            rates.append(r["tps"])
            print(f"{name:8} {r['tok']:>6} {r['wall_s']:>7} {r['tps']:>7}  {r['finish']}")
        if args.runs > 1:
            print(f"{name:8} median tok/s: {statistics.median(rates)}")

    tools = [{"type": "function", "function": {"name": "get_weather",
               "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
    r = call({"model": args.model, "max_tokens": 256, "temperature": 0,
              "messages": [{"role": "user", "content": "Weather in Auckland? Use the tool."}],
              "tools": tools, "tool_choice": "auto"})
    print(f"tools    finish={r['finish']} (tool_calls = parser working)")

if __name__ == "__main__":
    main()
