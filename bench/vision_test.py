#!/usr/bin/env python3
"""Vision qualification tests: describe (shapes/colors) + verbatim OCR.

Synthetic test image (repo charts/ or box /tmp/vision-test.png): red
triangle, blue oval, caption "GLM VISION TEST 42". Both tests must
pass at a rung for the rung to be qualified. Exit 0 = both pass and
engine alive after.
"""
import base64
import json
import sys
import time
import urllib.request

BASE = "http://localhost:8012"
MODEL = "GLM-5.3-Flash-EXL3-3.5bpw"
IMG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vision-test.png"


def ask(text, max_tokens=512):
    img = base64.b64encode(open(IMG, "rb").read()).decode()
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img}"}},
                {"type": "text", "text": text},
            ]}],
            "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    wall = time.time() - t0
    return wall, (d["choices"][0]["message"].get("content") or "")


def alive():
    try:
        urllib.request.urlopen(BASE + "/health", timeout=5)
        return True
    except Exception:
        return False


def main():
    mode = sys.argv[2] if len(sys.argv) > 2 else "single"
    if mode == "multi":
        img = base64.b64encode(open(IMG, "rb").read()).decode()
        body = {"model": MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img}"}},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img}"}},
                    {"type": "text",
                     "text": "Two images are attached. Quote the text "
                             "visible in each, one line per image."},
                ]}],
                "max_tokens": 512, "temperature": 0}
        req = urllib.request.Request(
            BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
        wall = time.time() - t0
        ocr = d["choices"][0]["message"].get("content") or ""
        ok = ocr.count("GLM VISION TEST 42") == 2
        print(f"multi wall={wall:.1f}s both-ocr={'PASS' if ok else 'FAIL'}")
        print(ocr[:200].replace("\n", " | "))
        print("engine alive after:", alive())
        return 0 if (ok and alive()) else 1

    w1, desc = ask("Describe this image exactly: shapes and colors.")
    ok_shapes = ("red" in desc.lower() and "triangle" in desc.lower()
                 and "blue" in desc.lower())
    print(f"describe wall={w1:.1f}s shapes/colors={'PASS' if ok_shapes else 'FAIL'}")
    print(desc[:300].replace("\n", " "))

    w2, ocr = ask("Quote the text in this image verbatim, nothing else.")
    ok_ocr = "GLM VISION TEST 42" in ocr
    print(f"ocr wall={w2:.1f}s verbatim={'PASS' if ok_ocr else 'FAIL'}")
    print(ocr[:200].replace("\n", " "))

    ok_alive = alive()
    print("engine alive after:", ok_alive)
    return 0 if (ok_shapes and ok_ocr and ok_alive) else 1


if __name__ == "__main__":
    raise SystemExit(main())
