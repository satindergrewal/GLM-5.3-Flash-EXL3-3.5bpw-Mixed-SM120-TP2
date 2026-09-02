#!/usr/bin/env python3
"""Video qualification test: static-caption clip, verbatim OCR from frames.

Sends a small mp4 (fixture on box: /tmp/vision-test.mp4, built from
/tmp/vision-test.png) and asks for the visible text. Pass = the caption
quoted and engine alive after.
"""
import base64
import json
import sys
import time
import urllib.request

BASE = "http://localhost:8012"
MODEL = "GLM-5.3-Flash-EXL3-3.5bpw"
VID = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vision-test.mp4"


def alive():
    try:
        urllib.request.urlopen(BASE + "/health", timeout=5)
        return True
    except Exception:
        return False


def main():
    vid = base64.b64encode(open(VID, "rb").read()).decode()
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "video_url",
                 "video_url": {"url": f"data:video/mp4;base64,{vid}"}},
                {"type": "text",
                 "text": "Quote the text visible in this video."},
            ]}],
            "max_tokens": 256, "temperature": 0}
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    wall = time.time() - t0
    ocr = d["choices"][0]["message"].get("content") or ""
    ok = "GLM VISION TEST 42" in ocr
    print(f"video wall={wall:.1f}s ocr={'PASS' if ok else 'FAIL'}")
    print(ocr[:200].replace("\n", " "))
    print("engine alive after:", alive())
    return 0 if (ok and alive()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
