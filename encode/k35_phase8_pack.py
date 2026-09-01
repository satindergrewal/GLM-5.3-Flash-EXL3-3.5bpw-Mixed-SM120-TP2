"""K35 phase-8 pack driver: assemble the sealed materialization plan and
materialize the mixed-rate checkpoint.

Replaces the 4bpw lineage's load_complete_surface call with the k35
surface (per-layer stores, honest per-choice bits) via an explicit
in-process override.  The override is loud and local to this driver; the
materializer module on disk keeps its uniform path intact for K4/K6
audits.

Usage (inside glm53-k35-encode, work root as cwd):
  python3 k35_phase8_pack.py --execute
Without --execute: validates surface + plan assembly and prints the plan
seal without writing the artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import k35_surface
from quant_pipeline.campaign.glm53_direct_k4 import (
    MAIN_ROUTED_LAYERS,
    MTP_LAYER,
    NUM_EXPERTS,
    PROJECTIONS,
)
from quant_pipeline.core.artifacts import canonical_json, sha256_bytes
from quant_pipeline.checkpoint import glm53_mcg_materializer as mat

WORK = Path("/mnt/t5evo/glm53-k35-work")
SOURCE_ROOT = Path("/mnt/t5evo/GLM-5.3-Flash-BF16")
OUTPUT_ROOT = WORK / "artifact-glm53-k35-mixed"


def seal(body: dict, field: str) -> dict:
    body = dict(body)
    body[field] = sha256_bytes(canonical_json(body))
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    readiness = json.loads((WORK / "gss" / "readiness-receipt.json").read_text())
    inventory = json.loads((WORK / "inventory.json").read_text())
    launch_plan_sha = str(readiness["launch_plan_sha256"])
    inventory_sha = str(inventory["inventory_sha256"])

    print("[1/4] loading k35 surface (verifies every layer + choice seal)...")
    surface = k35_surface.load_k35_surface(WORK / "layers", readiness_receipt=readiness)
    census = surface.rate_census()
    print(
        f"      surface: {len(surface.choices)} choices, "
        f"k3={census['k3_choice_count']} k4={census['k4_choice_count']}, "
        f"abi={surface.packed_reader_abi_sha256[:12]}"
    )

    mtp_adapter = k35_surface.build_k35_mtp_adapter_receipt(
        surface=surface,
        launch_plan_sha256=launch_plan_sha,
        inventory_sha256=inventory_sha,
    )
    print(f"[2/4] mtp adapter receipt: {mtp_adapter['receipt_sha256'][:12]} (payloads-bound, not qualified)")

    main_choice_count = len(MAIN_ROUTED_LAYERS) * NUM_EXPERTS * len(PROJECTIONS)
    mtp_choice_count = NUM_EXPERTS * len(PROJECTIONS)
    plan = seal(
        {
            "schema": mat.K35_MIXED_PLAN_SCHEMA,
            "bits": mat.MIXED_BITS_MARKER,
            "contract_sha256": surface.contract_sha256,
            "inventory_sha256": inventory_sha,
            "main_layer_receipt_sha256": list(surface.main_layer_receipt_sha256),
            "mtp_adapter_receipt_sha256": mtp_adapter["receipt_sha256"],
            "main_choice_count": main_choice_count,
            "mtp_choice_count": mtp_choice_count,
            "total_choice_count": main_choice_count + mtp_choice_count,
            "packed_root": str(WORK / "layers"),
            "source_root": str(SOURCE_ROOT),
            "output_root": str(OUTPUT_ROOT),
        },
        "plan_sha256",
    )
    print(f"[3/4] materialization plan sealed: {plan['plan_sha256'][:12]}")

    if not args.execute:
        print("[4/4] dry-run only (pass --execute to write the artifact)")
        return 0

    # Loud local override: k35 surface instead of the uniform-K4 loader.
    def _k35_surface_loader(*, root, contract, mtp_adapter_receipt):
        if mtp_adapter_receipt.get("receipt_sha256") != mtp_adapter["receipt_sha256"]:
            raise ValueError("k35 pack: mtp adapter receipt differs from the sealed plan")
        return surface

    mat.load_complete_surface = _k35_surface_loader

    print("[4/4] materializing (per-shard, resumable)...")
    receipt = mat.materialize_checkpoint(
        plan=plan,
        contract={
            "launch_plan_sha256": launch_plan_sha,
            "inventory_sha256": inventory_sha,
        },
        inventory=inventory,
        mtp_adapter_receipt=mtp_adapter,
        packed_root=WORK / "layers",
        source_root=SOURCE_ROOT,
        output_root=OUTPUT_ROOT,
    )
    print(
        "PACK COMPLETE:",
        json.dumps(
            {
                k: receipt.get(k)
                for k in (
                    "bits",
                    "routed_choice_count",
                    "output_tensor_count",
                    "output_logical_bytes",
                    "complete",
                )
            }
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
