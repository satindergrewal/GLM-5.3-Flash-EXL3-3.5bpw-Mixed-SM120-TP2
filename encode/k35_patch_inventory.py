#!/usr/bin/env python3
"""Patch the sealed inventory with the checkpoint path field and re-seal.

The encode backend (Glm53BF16Source) and MTP capture both bind the BF16
source root via inventory["checkpoint"]; the original builder omitted it.
Re-seals from the existing shard/tensor hashes (no re-walk of the master).
"""
import json
from pathlib import Path

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes
from quant_pipeline.campaign import glm53_uniform_k4 as k4
from quant_pipeline.campaign import glm53_direct_k4 as k4d

WORK = Path("/mnt/t5evo/glm53-k35-work")
MASTER = "/mnt/t5evo/GLM-5.3-Flash-BF16"

body = json.loads((WORK / "inventory.json").read_text())
body.pop("inventory_sha256", None)
body["checkpoint"] = MASTER
body["inventory_sha256"] = sha256_bytes(canonical_json(body))

main_rows, mtp_rows, native_rows = k4._inventory_surfaces(body)
assert len(main_rows) == 36288 and len(mtp_rows) == 864, (len(main_rows), len(mtp_rows))
tensor_map = k4d.inventory_tensor_map(body)
assert len(tensor_map) == 38770, len(tensor_map)
src = k4d.Glm53BF16Source(body, MASTER, verify_shards=False)
assert src.rows is not None

(WORK / "inventory.json").write_text(json.dumps(body, indent=1))
print("INVENTORY_PATCHED", body["inventory_sha256"][:16])
