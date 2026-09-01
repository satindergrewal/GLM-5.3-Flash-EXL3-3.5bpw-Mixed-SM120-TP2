#!/usr/bin/env bash
# Pull the checkpoint from Hugging Face into ./artifact-glm53-k35-mixed.
# aria2c multi-connection; resumable. ~146 GiB.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${MODEL_DEST:-$HERE/artifact-glm53-k35-mixed}"
REPO="satgeze/GLM-5.3-Flash-EXL3-TR3-3.5bpw"
command -v aria2c >/dev/null || { echo "install aria2 first"; exit 1; }
mkdir -p "$DEST" && cd "$DEST"
python3 - << 'PY' > aria2-input.txt
import json, urllib.request
url = "https://huggingface.co/api/models/satgeze/GLM-5.3-Flash-EXL3-TR3-3.5bpw"
req = urllib.request.Request(url, headers={"User-Agent": "recipe"})
for f in json.load(urllib.request.urlopen(req, timeout=30))["siblings"]:
    name = f["rfilename"]
    if name.startswith("."):
        continue
    print("https://huggingface.co/" + "satgeze/GLM-5.3-Flash-EXL3-TR3-3.5bpw" + "/resolve/main/" + name)
    print("  out=" + name)
PY
aria2c -i aria2-input.txt -j4 -x8 -s8 -c --console-log-level=warn --summary-interval=0
echo "done: $DEST"
