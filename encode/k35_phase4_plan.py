#!/usr/bin/env python3
"""Phase 4: declared sm120 preflight + sealed launch plan (box venue).

Adapted from the runbook (section 8 phase 4): work root moved to
/mnt/t5evo/glm53-k35-work (t9 filled by archive pulls), GPUs are the box
pair (indices 0,1), baselines copied into baseline/.
"""
import json
from pathlib import Path

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes
from quant_pipeline.campaign import glm53_uniform_k35 as k35

WORK = Path("/mnt/t5evo/glm53-k35-work")


def seal(body: dict, field: str) -> dict:
    body[field] = sha256_bytes(canonical_json(body))
    return body


def main() -> int:
    inventory = json.loads((WORK / "inventory.json").read_text())
    release_bytes = (WORK / "baseline/v84-release-validation.json").read_bytes()

    preflight = seal({
        "schema": k35.SM120_DECLARED_PREFLIGHT_SCHEMA,
        "ready": True, "mode": "layer-streaming",
        "checkpoint_seal_mode": "full-shard-sha256",
        "checkpoint_inventory_sha256": inventory["inventory_sha256"],
        "workers": 2,
        "gpus": [
            {"index": 0, "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
             "compute_capability": "12.0"},
            {"index": 1, "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
             "compute_capability": "12.0"},
        ],
        "declaration": {
            "attested_by": "sm120-declared-variant",
            "rationale": "box encode: sealed four-B200 venue unavailable; "
                         "2x RTX PRO 6000 declared for k35 only",
            "runtime_receipt_sha256": sha256_bytes(release_bytes),
        },
    }, "preflight_sha256")
    (WORK / "preflight-sm120-declared.json").write_text(json.dumps(preflight, indent=2))

    baseline = {"five_cold_run_kld_receipt":
                json.loads((WORK / "baseline/k4-five-cold-run.json").read_text())}
    allocations = {L: k35.seal_layer_allocation(
        L, k35.build_provisional_allocation(L),
        provisional=True, basis="deterministic_provisional") for L in (*range(3, 45), 45)}
    for L, alloc in allocations.items():
        (WORK / "allocations" / f"L{L:02d}.json").write_text(json.dumps(alloc, indent=2))

    plan = k35.build_launch_plan(
        inventory, preflight,
        four_bpw_baseline=baseline, layer_allocations=allocations,
        allow_declared_sm120_preflight=True)
    k35.verify_launch_plan(plan)
    (WORK / "plan.json").write_text(json.dumps(plan, indent=2))
    state = k35.initial_state(plan)
    (WORK / "state" / "state-0000.json").write_text(json.dumps(state, indent=2))
    print("plan", plan["launch_plan_sha256"][:16], "variant", plan["preflight_variant"])
    print("PLAN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
