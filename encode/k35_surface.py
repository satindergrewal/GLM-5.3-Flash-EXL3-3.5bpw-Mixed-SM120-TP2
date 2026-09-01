"""K35 mixed-rate packed surface: loads the sealed per-layer k35 payload
stores into the surface shape the GLM MCG materializer consumes.

The 4bpw lineage validated a uniform-K4 combined store through
evaluation.glm53_packed_k4_reader.load_complete_surface.  The k35 campaign
sealed per-layer stores (layers/L<NN>/payload-store) with honest per-choice
bits under K35_PACKED_CHOICE_SCHEMA, which the K4 verifier rejects by
design (k35_common.py NEW SURFACE warning).  This module is the downstream
reader that warning demands: same census discipline, mixed-rate honest.

Integrity rules:
- Every layer receipt must be sealed, complete, and bound to the readiness
  receipt's allocation hash for that layer.
- Choice census per layer = NUM_EXPERTS * len(PROJECTIONS) exactly.
- Rates are read from the choices themselves; the only accepted set is
  PER_TENSOR_ALLOWED_BITS.  No global rate is invented.
- The MTP adapter receipt carries payload-completeness attestations ONLY;
  qualification stays with the KLD gate downstream.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Mapping

from k35_common import K35_PACKED_CHOICE_SCHEMA, K35PackedPayloadStore, die

_HASH = re.compile(r"[0-9a-f]{64}")
_LAYER_DIR = re.compile(r"^L(\d{2})$")

# Populated by import of the campaign module (keeps one source of truth).
from quant_pipeline.campaign.glm53_direct_k4 import (  # noqa: E402
    MAIN_ROUTED_LAYERS,
    MTP_LAYER,
    NUM_EXPERTS,
    PROJECTIONS,
)
from quant_pipeline.campaign import glm53_uniform_k35 as _k35  # noqa: E402

MIXED_BITS_MARKER = "mixed_k34_per_tensor"
K35_LAYER_RECEIPT_SCHEMA = "quant-pipeline.glm53-k35-layer-receipt.v1"
K35_MTP_ADAPTER_SCHEMA = "quant-pipeline.glm53-k35-mtp-adapter-receipt.v1"


def _canonical(obj: Any) -> bytes:
    from quant_pipeline.core.artifacts import canonical_json

    return canonical_json(obj)


def _sha(data: bytes) -> str:
    from quant_pipeline.core.artifacts import sha256_bytes

    return sha256_bytes(data)


def _verify_layer_receipt(receipt: Mapping[str, Any], *, layer: int) -> str:
    if receipt.get("schema") != K35_LAYER_RECEIPT_SCHEMA:
        die(f"L{layer} receipt schema differs")
    digest = receipt.get("receipt_sha256")
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if (
        not isinstance(digest, str)
        or _HASH.fullmatch(digest) is None
        or _sha(_canonical(body)) != digest
    ):
        die(f"L{layer} receipt seal differs")
    if (
        receipt.get("layer") != layer
        or receipt.get("complete") is not True
        or receipt.get("bits") != MIXED_BITS_MARKER
    ):
        die(f"L{layer} receipt is not a complete mixed-rate seal")
    return digest


class K35ObjectIndex:
    """Digest-indexed view over every layer's payload-store objects."""

    def __init__(self, layers_root: Path):
        self.paths: dict[str, Path] = {}
        for store_dir in sorted(layers_root.glob("L*/payload-store/objects")):
            for bucket in sorted(store_dir.iterdir()):
                if not bucket.is_dir():
                    continue
                for obj in sorted(bucket.glob("*.bin")):
                    self.paths.setdefault(obj.stem, obj)
        if not self.paths:
            die("k35 object index is empty")

    def load_tensor(self, ref: Mapping[str, Any]):
        import torch

        digest = str(ref.get("sha256", ""))
        path = self.paths.get(digest)
        if path is None:
            die(f"k35 object absent from index: {digest[:12]}")
        expected_bytes = ref.get("bytes")
        if isinstance(expected_bytes, int) and path.stat().st_size != expected_bytes:
            die(f"k35 object byte count differs: {digest[:12]}")
        import numpy as np

        raw = np.fromfile(path, dtype=np.uint8)
        dtype = str(ref.get("dtype"))
        if dtype == "int16":
            viewed = raw.view(np.int16)
        elif dtype == "float16":
            viewed = raw.view(np.float16)
        elif dtype == "int32":
            viewed = raw.view(np.int32)
        else:
            die(f"k35 object dtype unsupported by index loader: {dtype}")
        shape = ref.get("shape")
        if isinstance(shape, list):
            viewed = viewed.reshape(shape)
        else:
            die(f"k35 object shape unsupported: {shape!r}")
        return torch.from_numpy(viewed.copy())


class K35MultiLayerStore:
    """The `store` face the materializer expects, over per-layer stores."""

    def __init__(self, layers_root: Path):
        self.layers_root = layers_root
        self.by_layer: dict[int, K35PackedPayloadStore] = {}
        for d in sorted(layers_root.iterdir()):
            m = _LAYER_DIR.match(d.name)
            if m and (d / "payload-store").is_dir():
                self.by_layer[int(m.group(1))] = K35PackedPayloadStore(
                    d / "payload-store"
                )
        if not self.by_layer:
            die("no k35 per-layer payload stores found")
        self.objects = K35ObjectIndex(layers_root)

    def verify_choice(self, choice: Mapping[str, Any]) -> dict[str, Any]:
        layer = int(choice["layer"])
        store = self.by_layer.get(layer)
        if store is None:
            die(f"choice references absent layer store: L{layer}")
        # Route through the per-layer verifier (seal + honest bits + dtype).
        return store.verify_choice(choice)


class K35Surface:
    """Surface shape consumed by the materializer (see _validate_plan)."""

    def __init__(
        self,
        *,
        root: Path,
        contract_sha256: str,
        choices: Mapping[tuple, Mapping[str, Any]],
        main_layer_receipt_sha256: tuple,
        mtp_layer_receipt_sha256: str,
        readiness_receipt_sha256: str,
        store: K35MultiLayerStore,
        packed_reader_abi_sha256: str = "",
    ):
        self.root = root
        self.contract_sha256 = contract_sha256
        self.choices = choices
        self.main_layer_receipt_sha256 = main_layer_receipt_sha256
        self.mtp_layer_receipt_sha256 = mtp_layer_receipt_sha256
        self.readiness_receipt_sha256 = readiness_receipt_sha256
        self.bits = MIXED_BITS_MARKER
        self.packed_reader_abi_sha256 = packed_reader_abi_sha256
        self._store = store

    @property
    def store(self) -> K35MultiLayerStore:
        return self._store

    def choice(self, layer: int, expert: int, projection: str):
        key = (layer, expert, projection)
        if key not in self.choices:
            die(f"k35 surface lacks choice L{layer} E{expert} {projection}")
        return self.choices[key]

    def rate_census(self) -> dict[str, int]:
        counts = {3: 0, 4: 0}
        for choice in self.choices.values():
            bits = int(choice["bits"])
            if bits not in _k35.PER_TENSOR_ALLOWED_BITS:
                die(f"choice rate outside allowed set: {bits}")
            counts[bits] += 1
        return {"k3_choice_count": counts[3], "k4_choice_count": counts[4]}


def load_k35_surface(
    layers_root: str | Path,
    *,
    readiness_receipt: Mapping[str, Any],
) -> K35Surface:
    root = Path(layers_root).resolve()
    readiness_sha = str(readiness_receipt["readiness_receipt_sha256"])
    alloc_by_layer = {
        int(entry["layer"]): str(entry["allocation_sha256"])
        for entry in readiness_receipt["allocations"]
    }
    store = K35MultiLayerStore(root)
    choices: dict[tuple, Mapping[str, Any]] = {}
    main_hashes: list[str] = []
    mtp_hash = ""
    expected_layers = (*MAIN_ROUTED_LAYERS, MTP_LAYER)
    for layer in expected_layers:
        receipt = json.loads((root / f"L{layer:02d}" / "layer-receipt.json").read_text())
        _verify_layer_receipt(receipt, layer=layer)
        if layer == MTP_LAYER:
            mtp_hash = receipt["receipt_sha256"]
        else:
            main_hashes.append(receipt["receipt_sha256"])
            if str(receipt.get("allocation_sha256")) != alloc_by_layer.get(layer):
                die(f"L{layer} receipt is not bound to the readiness allocation")
        n_k3 = int(receipt.get("k3_tensor_count", -1))
        n_k4 = int(receipt.get("k4_tensor_count", -1))
        if n_k3 + n_k4 != NUM_EXPERTS * len(PROJECTIONS):
            die(f"L{layer} rate census differs: {n_k3}+{n_k4}")
        for digest in receipt.get("choice_sha256", []):
            path = root / f"L{layer:02d}" / "payload-store" / "choices" / f"{digest}.json"
            row = json.loads(path.read_text())
            if row.get("schema") != K35_PACKED_CHOICE_SCHEMA:
                die(f"choice schema differs: {digest[:12]}")
            if row.get("choice_sha256") != digest:
                die(f"choice file/name hash differs: {digest[:12]}")
            key = (int(row["layer"]), int(row["expert"]), str(row["projection"]))
            if key in choices:
                die(f"duplicate choice key: {key}")
            choices[key] = row
    expected = len(expected_layers) * NUM_EXPERTS * len(PROJECTIONS)
    if len(choices) != expected:
        die(f"k35 choice census differs: {len(choices)} != {expected}")
    abis = {str(row["decoder"]["reader_abi_sha256"]) for row in choices.values()}
    if len(abis) != 1:
        die("k35 choices do not share one sealed MCG reader ABI")
    return K35Surface(
        root=root,
        contract_sha256=readiness_sha,
        choices=choices,
        main_layer_receipt_sha256=tuple(main_hashes),
        mtp_layer_receipt_sha256=mtp_hash,
        readiness_receipt_sha256=readiness_sha,
        store=store,
        packed_reader_abi_sha256=abis.pop(),
    )


def build_k35_mtp_adapter_receipt(
    *,
    surface: K35Surface,
    launch_plan_sha256: str,
    inventory_sha256: str,
) -> dict[str, Any]:
    """Honest MTP binding: sealed payloads complete and hash-bound.

    Deliberately does NOT claim runtime qualification; the KLD gate owns
    that verdict.  Fields mirror the 4bpw adapter receipt shape so the
    materializer's binding checks carry over unchanged.
    """
    body = {
        "schema": K35_MTP_ADAPTER_SCHEMA,
        "layer": MTP_LAYER,
        "launch_plan_sha256": launch_plan_sha256,
        "inventory_sha256": inventory_sha256,
        "expert_count": NUM_EXPERTS,
        "matrix_count": NUM_EXPERTS * len(PROJECTIONS),
        "bits": MIXED_BITS_MARKER,
        "packed_payload_receipt_sha256": surface.mtp_layer_receipt_sha256,
        "qualified": False,
        "qualification_owner": "k35 KLD gate (downstream); this receipt binds payloads only",
    }
    from quant_pipeline.core.artifacts import sha256_bytes, canonical_json

    body["receipt_sha256"] = sha256_bytes(canonical_json(body))
    return body
