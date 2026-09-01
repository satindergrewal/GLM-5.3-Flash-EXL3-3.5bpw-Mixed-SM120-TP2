#!/usr/bin/env python3
"""Build the sealed GLM-5.3 release inventory in the campaign's vocabulary.

The r7 CLI emits GLM-5.2-flat tensor names (model.layers.*); the k35/K4
campaigns consume quant-pipeline.glm-release-inventory.v1 with
model.language_model.layers.* names, a full-shard SHA-256 closure, and a
geometry block (contract: glm53_uniform_k4._inventory_surfaces and
glm53_direct_k4.inventory_tensor_map). This builder walks the BF16 master's
own safetensors headers, so every tensor name is correct by construction.

Run inside the encode container with the campaign PYTHONPATH. Output:
inventory.json in the work root, self-validated by the real consumer.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file
from quant_pipeline.campaign import glm53_uniform_k4 as k4

MASTER = Path("/mnt/t5evo/GLM-5.3-Flash-BF16")
OUT = Path("/mnt/t5evo/glm53-k35-work/inventory.json")
REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"   # zai-org/GLM-5.3-Flash @ 2026-08-27

ROUTED_LAYERS = tuple(range(3, 45))
MTP_LAYERS = (45,)
_ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4,
             "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def shard_header_and_sha(path: Path) -> tuple[dict, str, dict[str, str]]:
    """One streaming pass: file sha256 + per-tensor payload sha256 + header.

    Tensor hashes are over the exact file byte range of each tensor
    ([8+header_len+start, 8+header_len+end) per the header's
    data_offsets), so no library semantics are involved.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        prefix = fh.read(8)
        h.update(prefix)
        header_len = struct.unpack("<Q", prefix)[0]
        header_bytes = fh.read(header_len)
        h.update(header_bytes)
        header = json.loads(header_bytes)
        spans = {name: (row["data_offsets"][0], row["data_offsets"][1])
                 for name, row in header.items() if name != "__metadata__"}
        tensor_hashes = {name: hashlib.sha256() for name in spans}
        base = 8 + header_len
        pos = 0
        total = path.stat().st_size - base
        chunk = 1 << 24
        while pos < total:
            block = fh.read(min(chunk, total - pos))
            if not block:
                raise RuntimeError(f"short read on {path}")
            h.update(block)
            lo, hi = pos, pos + len(block)
            for name, (start, end) in spans.items():
                if end <= lo or start >= hi:
                    continue
                tensor_hashes[name].update(block[max(0, start - lo):min(len(block), end - lo)])
            pos = hi
    return header, h.hexdigest(), {n: t.hexdigest() for n, t in tensor_hashes.items()}


def main() -> int:
    index = json.loads((MASTER / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    print(f"walking {len(shards)} shards, {len(weight_map)} tensors", flush=True)

    tensors: list[dict] = []
    shard_sha256: dict[str, str] = {}
    seen: set[str] = set()
    for n, shard in enumerate(shards):
        header, digest, payload_shas = shard_header_and_sha(MASTER / shard)
        shard_sha256[shard] = digest
        for name, row in header.items():
            if name == "__metadata__":
                continue
            if name in seen:
                raise RuntimeError(f"duplicate tensor across shards: {name}")
            seen.add(name)
            if weight_map.get(name) != shard:
                raise RuntimeError(f"index/header placement differs for {name}")
            dtype = row["dtype"]
            shape = [int(v) for v in row["shape"]]
            itemsize = _ITEMSIZE.get(dtype)
            if itemsize is None:
                raise RuntimeError(f"unmapped dtype {dtype} for {name}")
            scope = "native"
            parts = name.split(".")
            if (len(parts) > 6 and parts[0] == "model" and parts[1] == "language_model"
                    and parts[2] == "layers" and parts[4] == "mlp" and parts[5] == "experts"):
                layer = int(parts[3])
                if layer in ROUTED_LAYERS:
                    scope = "routed_expert"
                elif layer in MTP_LAYERS:
                    scope = "mtp_routed_expert"
            tensors.append({
                "tensor_name": name,
                "scope": scope,
                "dtype": dtype,
                "shape": shape,
                "source_bytes": int(math.prod(shape)) * itemsize,
                "source_payload_sha256": payload_shas[name],
                "shard": shard,
            })
        if (n + 1) % 20 == 0 or n + 1 == len(shards):
            print(f"  {n + 1}/{len(shards)} shards hashed", flush=True)

    if seen != set(weight_map):
        missing = set(weight_map) - seen
        extra = seen - set(weight_map)
        raise RuntimeError(f"index closure differs: missing={len(missing)} extra={len(extra)}")

    discovered = sorted({int(p.split(".")[3])
                         for p in weight_map
                         if p.startswith("model.language_model.layers.")})
    if discovered != list(range(46)):
        raise RuntimeError(f"discovered layer surface differs: {discovered[:3]}...{discovered[-3:]}")

    body = {
        "schema": "quant-pipeline.glm-release-inventory.v1",
        "seal_mode": "full-shard-sha256",
        "model_revision": REVISION,
        "config_sha256": sha256_file(MASTER / "config.json"),
        "index_sha256": sha256_file(MASTER / "model.safetensors.index.json"),
        "geometry": {
            "model_type": "glm5_next",
            "main_layers": 45,
            "mtp_layers": 1,
            "first_moe_layer": 3,
            "routed_experts": 288,
            "discovered_layers": discovered,
        },
        "shard_sha256": shard_sha256,
        "tensors": sorted(tensors, key=lambda r: r["tensor_name"]),
    }
    body["inventory_sha256"] = sha256_bytes(canonical_json(body))

    # Self-check with the real consumers before writing.
    main_rows, mtp_rows, native_rows = k4._inventory_surfaces(body)
    print(f"surfaces check OK: routed={len(main_rows)} mtp={len(mtp_rows)} "
          f"native={len(native_rows)}", flush=True)
    if len(main_rows) != 36288 or len(mtp_rows) != 864:
        raise RuntimeError("routed census differs from the sealed K4 surface")

    from quant_pipeline.campaign import glm53_direct_k4 as k4d
    tensor_map = k4d.inventory_tensor_map(body)
    if len(tensor_map) != len(weight_map):
        raise RuntimeError(f"encode-backend tensor map census differs: {len(tensor_map)}")
    print(f"encode-backend map OK over {len(tensor_map)} tensors", flush=True)

    OUT.write_text(json.dumps(body, indent=1))
    print("INVENTORY_OK", body["inventory_sha256"][:16], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
