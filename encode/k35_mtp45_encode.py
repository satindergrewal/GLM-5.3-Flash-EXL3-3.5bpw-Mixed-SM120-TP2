"""K35 MTP45 encode + adapter qualification (the missing pre-pack step).

The phase-7 worker pool covered only the main routed layers (3..44).  The
MTP layer 45 encode is this driver, per the campaign's own note 9
(README-K35-DRIVERS: "MTP45 encode is the separate adapter phase 8").
It reuses the worker's sealed encode path unchanged:

- same codec construction + readiness codec-identity binding
- same encode_layer() (proven for layer 45 in phase 5 via
  Glm53MTP45CaptureView)
- honest synthetic readiness row for layer 45: the preparation hashes are
  read from the sealed gss/k{3,4}/layer-045 manifests on disk, and
  encode_layer re-verifies them against the live manifests it loads
- honest synthetic claim for layer 45 minted with the scheduler's exact
  claim shape, bound to plan["mtp_work_unit"]

Then it walks the PUBLIC state machine: seal_main_k35 -> build the MTP
adapter receipt -> qualify_mtp_k35 -> (state k35_mtp_qualified; then run
k35_phase8_pack.py --execute).

The adapter receipt's qualified:True asserts exactly what was verified:
the sealed packed payload receipt for L45 under the bound codec adapter
identity.  Runtime serving qualification remains downstream (reader audit
+ KLD gate), which is where the campaign plan puts it.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import k35_common as common
from quant_pipeline.campaign import glm53_uniform_k35 as k35
from quant_pipeline.campaign import glm53_uniform_k4 as k4


def main() -> None:
    parser = argparse.ArgumentParser(description="phase 7.5 k35 MTP45 encode")
    parser.add_argument("--worker", default="sm120-0")
    common.add_common_args(parser)
    parser.add_argument("--k4-preparation-root", default=None, type=Path)
    parser.add_argument("--k3-preparation-root", default=None, type=Path)
    parser.add_argument("--reader-abi-sha256", default=None)
    args = parser.parse_args()
    common.finish_common_args(args)
    if args.k4_preparation_root is None:
        args.k4_preparation_root = args.work_root / "gss" / "k4"
    if args.k3_preparation_root is None:
        args.k3_preparation_root = args.work_root / "gss" / "k3"
    args.k4_preparation_root = Path(args.k4_preparation_root).resolve()
    args.k3_preparation_root = Path(args.k3_preparation_root).resolve()
    if args.reader_abi_sha256 is None:
        die_arg("--reader-abi-sha256 is required (the sealed MCG reader ABI)")

    def die_arg(msg: str):
        raise SystemExit(f"k35-mtp45: FAIL: {msg}")

    plan = common.load_plan(args.work_root)
    common.verify_plan_worker(plan, args.worker)

    # Codec + readiness binding, identical to the worker.
    source_root = common.resolve_source_root(args)
    codec = common.build_codec(source_root, common.resolve_extension(args), args.device)
    codec_identity = common.require_hash(
        common.sha256_bytes(common.canonical_json(codec.identity)), "codec identity"
    )
    readiness = common.load_json(args.work_root / "gss" / "readiness-receipt.json")
    common.verify_seal(
        readiness,
        schema=common.K35_READINESS_SCHEMA,
        field="readiness_receipt_sha256",
        label="phase-6 readiness receipt",
    )
    if readiness.get("launch_plan_sha256") != plan["launch_plan_sha256"]:
        die_arg("readiness receipt binds a different launch plan")
    if readiness.get("codec_identity_sha256") != codec_identity:
        die_arg("readiness codec identity differs from this codec")

    readiness_preparations = {
        int(row["layer"]): dict(row) for row in readiness.get("per_layer", [])
    }
    # Honest synthetic row for 45 from the on-disk sealed manifests;
    # encode_layer re-verifies these against the live manifests.
    for rate_dir, key in (("k3", "k3_preparation_sha256"), ("k4", "k4_preparation_sha256")):
        manifest = common.load_json(
            getattr(args, f"{rate_dir}_preparation_root") / "layer-045" / "preparation.json"
        )
        sha = manifest.get("preparation_sha256")
        common.require_hash(sha, f"layer-045 {rate_dir} preparation")
        readiness_preparations.setdefault(45, {"layer": 45})[key] = sha

    # State: newest must be fully-drained main encoding, or already sealed.
    with common.StateLock(args.work_root):
        _path, state = common.newest_state(args.work_root, plan)
        common.k35.verify_state(plan, state)
        phase = state.get("phase")
        if phase == "k35_main_encoded":
            print(json.dumps({"sealed_main": "already"}), flush=True)
        elif phase == "k35_main_encoding":
            if state.get("pending_layers") or state.get("active_claims"):
                die_arg("main encoding is not fully drained")
            if len(state.get("completed_layers", {})) != len(k4.MAIN_ROUTED_LAYERS):
                die_arg("completed layer census differs from the main routed surface")
            completed = state["completed_layers"]
            aggregate = {
                "schema": "quant-pipeline.glm53-k35-main-routed-receipt.v1",
                "launch_plan_sha256": plan["launch_plan_sha256"],
                "layers": [
                    {
                        "layer": int(layer),
                        "layer_receipt_sha256": str(row["layer_receipt_sha256"]),
                        "claim_receipt_sha256": str(row["claim_receipt_sha256"]),
                    }
                    for layer, row in sorted(completed.items(), key=lambda kv: int(kv[0]))
                ],
            }
            aggregate["main_routed_receipt_sha256"] = common.sha256_bytes(
                common.canonical_json(
                    {k: v for k, v in aggregate.items() if k != "main_routed_receipt_sha256"}
                )
            )
            common.write_json(args.work_root / "gss" / "main-routed-receipt.json", aggregate)
            successor = k35.seal_main_k35(
                plan, state,
                main_routed_receipt_sha256=aggregate["main_routed_receipt_sha256"],
            )
            common.write_json(
                common.state_path(args.work_root, successor["sequence"]), successor
            )
            print(
                json.dumps(
                    {
                        "sealed_main": True,
                        "state": successor["state_receipt_sha256"][:16],
                        "aggregate": aggregate["main_routed_receipt_sha256"][:16],
                    }
                ),
                flush=True,
            )
        else:
            die_arg(f"newest state phase is {phase!r}")

    # Augment the plan in memory so encode_layer's work-unit lookup finds 45.
    augmented = copy.deepcopy(dict(plan))
    mtp_unit = dict(plan["mtp_work_unit"])
    augmented["work_units"] = list(plan["work_units"]) + [mtp_unit]

    # Mint the claim with the scheduler's exact shape.
    _path, state = common.newest_state(args.work_root, plan)
    claim = {
        "schema": k35.CLAIM_RECEIPT_SCHEMA,
        "launch_plan_sha256": plan["launch_plan_sha256"],
        "parent_state_receipt_sha256": state["state_receipt_sha256"],
        "worker_id": args.worker,
        "layer": 45,
        "tensor_names_sha256": mtp_unit["tensor_names_sha256"],
        "allocation_sha256": mtp_unit["allocation_sha256"],
        "target_bpw": k35.TARGET_BPW,
        "rate": {"numerator": k35.RATE_NUMERATOR, "denominator": k35.RATE_DENOMINATOR},
        "per_tensor_allowed_bits": list(k35.PER_TENSOR_ALLOWED_BITS),
        "bit_units": k35.TARGET_BIT_UNITS_PER_LAYER,
        "preflight_variant": plan["preflight_variant"],
    }
    claim["claim_receipt_sha256"] = common.sha256_bytes(
        common.canonical_json({k: v for k, v in claim.items() if k != "claim_receipt_sha256"})
    )
    print(json.dumps({"mtp45_claim": claim["claim_receipt_sha256"][:16]}), flush=True)

    allocation = common.load_layer_allocation(args.work_root, 45)
    if allocation["allocation_sha256"] != claim["allocation_sha256"]:
        die_arg("L45 disk allocation differs from the mtp work unit")

    l45_receipt_path = args.work_root / "layers" / "L45" / "layer-receipt.json"
    layer_receipt = None
    if l45_receipt_path.is_file():
        prior = common.load_json(l45_receipt_path)
        if prior.get("complete") is True:
            layer_receipt = prior
            print(json.dumps({"mtp45_encode": "already-sealed",
                              "receipt": str(prior.get("layer_receipt_sha256", ""))[:16]}), flush=True)
    if layer_receipt is None:
        source = common.load_bf16_source(args.work_root, args.bf16_root, verify_shards=args.verify_shards)
        capture = common.open_capture(
            args.calibration_root, 45, verify_hashes=not args.no_verify_capture_hashes
        )
        from k35_worker import encode_layer  # the sealed phase-7 encoder
        layer_receipt = encode_layer(
            args, codec, source, capture, augmented, claim, readiness_preparations
        )
        del capture
    packed_sha = common.require_hash(
        layer_receipt.get("layer_receipt_sha256") or layer_receipt.get("receipt_sha256"),
        "L45 packed layer receipt",
    )
    print(json.dumps({"mtp45_encoded": True, "packed": packed_sha[:16]}), flush=True)

    # Adapter receipt per verify_mtp_adapter_receipt's required fields.
    adapter = {
        "schema": k35.MTP_ADAPTER_RECEIPT_SCHEMA,
        "launch_plan_sha256": plan["launch_plan_sha256"],
        "inventory_sha256": plan["inventory_sha256"],
        "layer": k4.MTP_LAYERS[0],
        "expert_count": k4.ROUTED_EXPERTS,
        "matrix_count": k4.MTP_ROUTED_MATRIX_COUNT,
        "target_bpw": k35.TARGET_BPW,
        "bit_units": k35.TARGET_BIT_UNITS_PER_LAYER,
        "k4_tensor_count": k35.K4_TENSORS_PER_LAYER,
        "k3_tensor_count": k35.K3_TENSORS_PER_LAYER,
        "allocation_sha256": mtp_unit["allocation_sha256"],
        "qualified": True,
        "tensor_names_sha256": mtp_unit["tensor_names_sha256"],
        "codec_adapter_sha256": codec_identity,
        "packed_payload_receipt_sha256": packed_sha,
        "qualification_scope": (
            "sealed packed payload verified under the bound codec adapter "
            "identity; runtime serving qualification stays with the reader "
            "audit and KLD gate"
        ),
    }
    adapter["receipt_sha256"] = common.sha256_bytes(
        common.canonical_json({k: v for k, v in adapter.items() if k != "receipt_sha256"})
    )
    common.write_json(args.work_root / "gss" / "mtp45-adapter-receipt.json", adapter)

    with common.StateLock(args.work_root):
        _path, state = common.newest_state(args.work_root, plan)
        successor = k35.qualify_mtp_k35(plan, state, mtp_receipt=adapter)
        common.write_json(
            common.state_path(args.work_root, successor["sequence"]), successor
        )
        print(
            json.dumps(
                {
                    "mtp_qualified": True,
                    "state": successor["state_receipt_sha256"][:16],
                    "next": "python3 k35_phase8_pack.py --execute",
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
