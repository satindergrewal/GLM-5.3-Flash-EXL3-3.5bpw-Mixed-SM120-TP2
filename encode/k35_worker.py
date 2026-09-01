#!/usr/bin/env python3
"""Phase 7 k35 encode worker: one process per GPU, dynamic whole-layer
claims through the sealed glm53_uniform_k35 state machine.

Loop per invocation (runbook section 8 phase 7):

  while layers remain:
    load newest state/state-NNNN.json (under the state flock)
    (successor, claim) = k35.claim_next_layer(plan, state, worker_id=...)
    WRITE THE SUCCESSOR STATE FILE BEFORE ENCODING
    encode all 864 tensors at their allocated integer rates
    write layers/L{NN}/ artifacts (glm53_direct_k4 layout) + layer receipt
    state = k35.complete_layer(...)
    write the next state file
    repeat

Drain pause: before each claim the worker checks <work-root>/PAUSE; if the
file exists it takes no new claim and exits 0 (a layer already in flight
always completes first).  This is the preferred pause; the alternative is
`docker pause` freezing the processes mid-layer (runbook phase 7).

Claim recovery: if a worker process dies mid-layer, its active claim stays
in the newest state.  Restarting the SAME worker id resumes that layer
(per-expert receipts are skipped when they already verify, with every
pre-existing choice re-hashed against the payload store and every Hessian
artifact re-hashed on disk, mirroring the K4 resume verifier).  If the
worker is dead for good, run this driver once with --recover-worker <id>:
it drops the claim, QUARANTINES the layer's partial per-claim artifacts
(expert receipts, payload-store, stale layer receipt; Hessians are kept),
and re-queues the layer at the front of pending in a recovery successor.
A fresh claim mints a new claim_receipt_sha256, and every expert receipt
and choice embeds the dead claim's hash, so deletion under the state lock
is the only seal-safe recovery; the removal is recorded in the successor's
evidence block.

The worker re-loads plan.json every loop iteration and exits cleanly when
the launch plan changes on disk (a phase-5b re-plan), so a live old-plan
worker stops appending old-plan successors instead of deepening the wedge;
state selection itself only ever considers states bound to the current
plan (k35_common.newest_state).

WARN: the recovery successor is the ONE transition the public state machine
does not offer.  It is sealed through the module-private k35._successor
(the runbook phase-7 text "drops the stale active claim in a recovery
successor" names a transition no public function implements).  It requires
an explicit flag and is recorded in the successor's evidence block; never
run it while the target worker might still be alive.

Usage (inside the encode container, cwd <work>):

  CUDA_VISIBLE_DEVICES=0 python3 k35_worker.py --worker sm120-0 --work-root /mnt/t5evo/glm53-k35-work
  CUDA_VISIBLE_DEVICES=1 python3 k35_worker.py --worker sm120-1 --work-root /mnt/t5evo/glm53-k35-work

Worker id: the sm120-declared plan assigns f"sm120-{slot}" from the
preflight gpus enumeration (glm53_uniform_k35.py:293); with preflight gpu
indices [1, 3] the ids are sm120-0 and sm120-1 (NOT sm120-1/sm120-3).

Encode notes (documented mismatches, details in README-K35-DRIVERS.md):
  - the sealed Glm53PreparedMCGBackend accepts only uniform bits in (4, 6)
    (glm53_prepared_backend.py:35, :117-124), so the worker drives the same
    numeric path per expert through Exl3MCGCodec.encode_candidates at the
    allocated rate, with rate-specific preparation vectors (gss/k3, gss/k4);
  - the sealed PackedMCGPayloadStore hardcodes bits=4 per choice
    (checkpoint/packed_payload.py put_choice), so layer artifacts use
    k35_common.K35PackedPayloadStore (same layout, honest per-choice bits);
  - gate/up encode is per tensor (the backend's grouped lockstep batching is
    a uniform-rate surface); throughput can be revisited without touching
    the receipt formats;
  - the down Hessian is conditioned on gate/up decoded at the reference
    rates k35.FLOOR_BITS (R7 pair_at semantics), the SAME context the probe
    measured and the DP allocation ranked under; mixed-rate triplets add
    reference-rate conditioning encodes (deployed choices keep the allocated
    rates), and every receipt stamps the rates actually used.

ASCII only.  CODE ONLY: nothing here is executed by the author.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import k35_common as common
from k35_common import die

VECTOR_TOPOLOGY = {
    "gate_proj": {"suh": "layer_shared", "svh": "expert_private"},
    "up_proj": {"suh": "layer_shared", "svh": "expert_private"},
    "down_proj": {"suh": "expert_private", "svh": "layer_shared"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="phase 7 k35 encode worker")
    parser.add_argument("--worker", default=None, help="plan worker id, e.g. sm120-0")
    common.add_common_args(parser)
    parser.add_argument(
        "--k4-preparation-root", default=None, type=Path, help="default <work>/gss/k4"
    )
    parser.add_argument(
        "--k3-preparation-root", default=None, type=Path, help="default <work>/gss/k3"
    )
    parser.add_argument(
        "--reader-abi-sha256",
        default=None,
        help="sealed reader ABI hash recorded in every packed choice",
    )
    parser.add_argument(
        "--recover-worker",
        default=None,
        metavar="ID",
        help="maintenance: drop a stale active claim and re-queue its layer",
    )
    args = parser.parse_args()
    common.finish_common_args(args)
    if args.k4_preparation_root is None:
        args.k4_preparation_root = args.work_root / "gss" / "k4"
    if args.k3_preparation_root is None:
        args.k3_preparation_root = args.work_root / "gss" / "k3"
    args.k4_preparation_root = Path(args.k4_preparation_root).resolve()
    args.k3_preparation_root = Path(args.k3_preparation_root).resolve()
    if args.recover_worker is None and args.worker is None:
        die("pass --worker ID (or --recover-worker ID for maintenance)")
    return args


# ---------------------------------------------------------------------------
# Recovery transition (see module WARN)
# ---------------------------------------------------------------------------


def quarantine_partial_layer(work_root: Path, layer: int) -> dict[str, Any]:
    """Remove the partial per-claim artifacts of a recovered (crashed) layer.

    Every expert receipt, every packed choice, and the choice predecessor
    chain embed the dead claim's claim_receipt_sha256; the re-claim
    necessarily mints a new claim hash (the claim seals its
    parent_state_receipt_sha256, which the recovery successor moved), so the
    resume gate and build_layer_receipt would reject every surviving receipt
    forever and the layer could never complete.  Deletion is the only
    seal-safe recovery.  The payload store is per-layer, so removing the
    whole store removes exactly this layer's entries; Hessians are
    claim-independent and save_hessians tolerates byte-identical existing
    artifacts, so they are kept.
    """

    import shutil

    stem = common.probe_stem(layer)
    layer_root = work_root / "layers" / stem
    removed: list[str] = []
    expert_dir = layer_root / "experts" / common.layer_dir_name(layer)
    receipts_removed = 0
    if expert_dir.exists():
        receipts_removed = len(list(expert_dir.glob("expert-*.json")))
        shutil.rmtree(expert_dir)
        removed.append(f"layers/{stem}/experts/{common.layer_dir_name(layer)}")
    store_dir = layer_root / "payload-store"
    if store_dir.exists():
        shutil.rmtree(store_dir)
        removed.append(f"layers/{stem}/payload-store")
    layer_receipt_path = layer_root / "layer-receipt.json"
    if layer_receipt_path.exists():
        layer_receipt_path.unlink()
        removed.append(f"layers/{stem}/layer-receipt.json")
    return {
        "layer": int(layer),
        "removed": removed,
        "expert_receipts_removed": receipts_removed,
        "kept": ["hessians (claim-independent; byte-identical artifacts tolerated)"],
        "reason": (
            "receipts, choices, and the choice predecessor chain embed the dead "
            "claim's claim_receipt_sha256; the re-claim mints a new hash, so "
            "deletion is the only seal-safe recovery"
        ),
    }


def recover_worker(args: argparse.Namespace) -> None:
    plan = common.load_plan(args.work_root)
    with common.StateLock(args.work_root):
        _path, state = common.newest_state(args.work_root, plan)
        common.k35.verify_state(plan, state)
        active = dict(state["active_claims"])
        target = active.get(args.recover_worker)
        if target is None:
            die(
                f"worker {args.recover_worker} holds no active claim in the newest "
                "state; nothing to recover"
            )
        layer = int(target["layer"])
        pending = [layer] + [int(x) for x in state["pending_layers"]]
        del active[args.recover_worker]
        quarantine = quarantine_partial_layer(args.work_root, layer)
        evidence = dict(state.get("evidence", {}))
        evidence["k35_recovery_note"] = (
            f"stale claim of worker {args.recover_worker} on layer "
            f"{layer} dropped and re-queued by k35_worker --recover-worker; "
            "the layer's partial per-claim artifacts (expert receipts, "
            "payload-store, stale layer receipt) were quarantined under the "
            "state lock because they all embed the dead claim hash; "
            "transition via k35._successor because the public state machine "
            "offers no recovery transition"
        )
        evidence["k35_recovery_quarantine"] = quarantine
        successor = common.k35._successor(
            plan,
            state,
            pending_layers=pending,
            active_claims=active,
            evidence=evidence,
        )
        common.write_json(
            common.state_path(args.work_root, successor["sequence"]), successor
        )
    print(
        json.dumps(
            {
                "recovered_worker": args.recover_worker,
                "requeued_layer": layer,
                "quarantined": quarantine["removed"],
                "state": successor["state_receipt_sha256"],
            }
        )
    )


# ---------------------------------------------------------------------------
# Layer encode
# ---------------------------------------------------------------------------


def tensor_full_name(layer: int, expert: int, projection: str) -> str:
    return (
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
    )


def save_hessians(layer_root: Path, layer: int, expert: int, gate_up, down, evidence) -> dict:
    """Mirror of Glm53PreparedMCGBackend._save_hessians
    (glm53_prepared_backend.py:336-405): FP16 matrices plus exact FP32
    recomputation hashes in the metadata block."""

    import os

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    directory = layer_root / "hessians" / f"layer-{layer:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"expert-{expert:03d}.safetensors"
    metadata = {
        "schema": "quant-pipeline.glm53-routed-p2-hessian-pair.v1",
        "layer": str(layer),
        "expert": str(expert),
        "stored_dtype": "float16",
        "gate_up_exact_fp32_sha256": str(evidence["gate_up"]["matrix_sha256"]),
        "down_exact_fp32_sha256": str(evidence["down"]["matrix_sha256"]),
        "exact_recomputation": "sealed_capture_routes_plus_decoded_gate_up",
    }
    if path.exists():
        if path.is_symlink():
            die(f"Hessian artifact is a symlink: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            if (
                set(handle.keys()) != {"gate_up_hessian", "down_hessian"}
                or (handle.metadata() or {}) != metadata
                or tuple(handle.get_slice("gate_up_hessian").get_shape())
                != (common.HIDDEN_SIZE, common.HIDDEN_SIZE)
                or tuple(handle.get_slice("down_hessian").get_shape())
                != (common.INTERMEDIATE_SIZE, common.INTERMEDIATE_SIZE)
            ):
                die(f"existing Hessian artifact differs: {path}")
    else:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        save_file(
            {
                "gate_up_hessian": torch.as_tensor(gate_up)
                .detach()
                .to(device="cpu", dtype=torch.float16)
                .contiguous(),
                "down_hessian": torch.as_tensor(down)
                .detach()
                .to(device="cpu", dtype=torch.float16)
                .contiguous(),
            },
            str(temporary),
            metadata=metadata,
        )
        os.replace(temporary, path)
    return {
        "schema": metadata["schema"],
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
        "stored_dtype": "float16",
        "gate_up_shape": [common.HIDDEN_SIZE, common.HIDDEN_SIZE],
        "down_shape": [common.INTERMEDIATE_SIZE, common.INTERMEDIATE_SIZE],
        "gate_up_exact_fp32_sha256": metadata["gate_up_exact_fp32_sha256"],
        "down_exact_fp32_sha256": metadata["down_exact_fp32_sha256"],
        "exact_recomputation_inputs": (
            "sealed_raw_capture_routes_plus_decoded_gate_up_plus_numeric_core"
        ),
    }


def encode_layer(
    args: argparse.Namespace,
    codec,
    source,
    capture,
    plan: Mapping[str, Any],
    claim: Mapping[str, Any],
    readiness_preparations: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    import torch

    from quant_pipeline.normalization.prior_search import permute_expert_hf

    layer = int(claim["layer"])
    stem = common.probe_stem(layer)
    layer_root = args.work_root / "layers" / stem
    expert_root = layer_root / "experts"
    store = common.K35PackedPayloadStore(layer_root / "payload-store")

    unit = next(row for row in plan["work_units"] if row["layer"] == layer)
    allocation = common.load_layer_allocation(args.work_root, layer)
    if allocation["allocation_sha256"] != claim["allocation_sha256"]:
        die(
            f"layer {layer} allocation seal differs from the claim: claim "
            f"{claim['allocation_sha256']} disk {allocation['allocation_sha256']}"
        )
    if allocation["allocation_sha256"] != unit["allocation_sha256"]:
        die(
            f"layer {layer} allocation seal differs from the plan unit; the plan "
            "was not rebuilt after phase 5 (README phase 5b)"
        )
    bits_map: dict[str, int] = {
        name: int(bits) for name, bits in allocation["allocation"].items()
    }

    _manifest_k4, tensors_k4 = common.load_preparation(
        args.k4_preparation_root, layer, expected_bits=4
    )
    _manifest_k3, tensors_k3 = common.load_preparation(
        args.k3_preparation_root, layer, expected_bits=3
    )
    readiness_row = readiness_preparations.get(layer)
    if not isinstance(readiness_row, Mapping):
        die(
            f"layer {layer} is absent from the phase-6 readiness receipt "
            "per_layer list; re-run phase 6"
        )
    if _manifest_k4["preparation_sha256"] != readiness_row.get(
        "k4_preparation_sha256"
    ):
        die(
            f"layer {layer} K4 preparation seal differs from the phase-6 "
            "readiness receipt; the preparation tree is not the sealed one"
        )
    if _manifest_k3["preparation_sha256"] != readiness_row.get(
        "k3_preparation_sha256"
    ):
        die(
            f"layer {layer} K3 preparation seal differs from the phase-6 "
            "readiness receipt; the preparation tree is not the sealed one"
        )
    if not torch.equal(tensors_k4["permutations"], tensors_k3["permutations"]):
        die(
            f"layer {layer}: K3 and K4 preparation permutations differ; the "
            "permuted basis must be shared across the triplet"
        )

    codec_identity_sha256 = common.require_hash(
        common.sha256_bytes(common.canonical_json(codec.identity)), "codec identity"
    )
    reader_abi = args.reader_abi_sha256
    if reader_abi is None:
        die(
            "--reader-abi-sha256 is required: every packed choice binds the sealed "
            "reader ABI (mirror of encode_work_unit, glm53_direct_k4.py:962)"
        )
    common.require_hash(reader_abi, "reader ABI")

    claim_sha = claim["claim_receipt_sha256"]
    allocation_sha = allocation["allocation_sha256"]
    capture_binding = capture.binding()
    mcg_marker = torch.tensor([-877912083], dtype=torch.int32)
    started = time.monotonic()

    for expert in range(common.NUM_EXPERTS):
        receipt_path = expert_root / f"layer-{layer:03d}" / f"expert-{expert:03d}.json"
        bits_by_projection = {
            projection: bits_map[tensor_full_name(layer, expert, projection)]
            for projection in common.PROJECTIONS
        }
        if receipt_path.exists():
            receipt = common.verify_expert_receipt(common.load_json(receipt_path))
            if (
                receipt["claim_receipt_sha256"] != claim_sha
                or receipt["allocation_sha256"] != allocation_sha
                or receipt["bits"] != bits_by_projection
            ):
                die(
                    f"existing expert receipt binds a foreign claim, allocation, or "
                    f"rate: {receipt_path}"
                )
            # Mirror the K4 resume verifier (glm53_direct_k4.py:817-829):
            # re-verify every pre-existing choice against the store.
            # K35PackedPayloadStore.verify_choice loads every object through
            # ExactCodecPayloadStore.load_tensor, which re-hashes each
            # content-addressed object file and fails loudly on truncation,
            # corruption, or deletion (exact_payload.py:130-152); without
            # this the damage surfaces only at phase-8 materialization.
            for projection in common.PROJECTIONS:
                choice = receipt["choices"][projection]
                verified = store.verify_choice(choice)
                if (
                    verified.get("layer") != layer
                    or verified.get("expert") != expert
                    or verified.get("projection") != projection
                    or verified.get("bits") != receipt["bits"][projection]
                ):
                    die(
                        f"existing expert receipt choice binding differs: "
                        f"{receipt_path}"
                    )
            # Re-hash the Hessian artifact on disk like the K4 verifier
            # (glm53_direct_k4.py:845-860); save_hessians only guards the
            # fresh path.
            hessian_record = receipt.get("hessian_artifact")
            if not isinstance(hessian_record, Mapping):
                die(f"existing expert receipt lacks its Hessian artifact: {receipt_path}")
            hessian_path = Path(str(hessian_record.get("path", ""))).resolve()
            try:
                hessian_path.relative_to(layer_root.resolve())
            except ValueError:
                die(
                    f"existing expert receipt Hessian artifact escapes the layer "
                    f"root: {receipt_path}"
                )
            if (
                not hessian_path.is_file()
                or hessian_path.is_symlink()
                or hessian_path.stat().st_size != hessian_record.get("bytes")
                or common.sha256_file(hessian_path) != hessian_record.get("sha256")
            ):
                die(
                    f"existing expert receipt Hessian artifact differs on disk: "
                    f"{receipt_path}"
                )
            continue

        triplet = source.load_triplet(layer, expert, device=args.device)
        permutation = tensors_k4["permutations"][expert].tolist()
        gate_weight, up_weight, down_weight = permute_expert_hf(
            triplet["gate_proj"], triplet["up_proj"], triplet["down_proj"], permutation
        )

        gate_cov, gate_up_evidence = common.gate_covariance(
            codec, capture, expert, args.device, args.chunk_rows
        )
        gate_up_evidence["matrix_sha256"] = common.tensor_sha256(gate_cov)

        encoded = {}
        choices: dict[str, dict[str, Any]] = {}
        used_vectors: dict[str, tuple[Any, Any]] = {}
        predecessor = claim_sha
        for projection, weight in (
            ("gate_proj", gate_weight),
            ("up_proj", up_weight),
        ):
            bits = bits_by_projection[projection]
            vectors = tensors_k4 if bits == 4 else tensors_k3
            suh, svh = common.preparation_vectors(vectors, projection, expert, args.device)
            candidate = common.encode_one_rate(
                codec,
                layer=layer,
                expert=expert,
                projection=projection,
                weight_hf=weight,
                covariance=gate_cov,
                bits=bits,
                suh=suh,
                svh=svh,
                provenance={
                    "claim_receipt_sha256": claim_sha,
                    "allocation_sha256": allocation_sha,
                    "public_shapleymcg_mixed_rate": True,
                    "global_allocator": False,
                },
            )
            encoded[projection] = candidate
            used_vectors[projection] = (suh, svh)
            choices[projection] = store.put_choice(
                layer=layer,
                expert=expert,
                projection=projection,
                bits=bits,
                choice_id=f"L{layer:03d}.E{expert:03d}.{projection}.K{bits}",
                trellis=candidate.packed,
                suh=suh,
                svh=svh,
                mcg=mcg_marker,
                reconstruction=candidate.reconstructed.half().contiguous(),
                vector_topology=VECTOR_TOPOLOGY[projection],
                reader_abi_sha256=reader_abi,
                provenance={
                    "claim_receipt_sha256": claim_sha,
                    "allocation_sha256": allocation_sha,
                    "bits": bits,
                    "packed_sha256": candidate.packed_sha256,
                    "reconstruction_sha256": candidate.reconstruction_sha256,
                    "vector_rate": bits,
                },
                predecessor_state_hash=predecessor,
            )
            predecessor = choices[projection]["choice_sha256"]

        # Corrected operation order: a fresh factor domain for the
        # candidate-conditioned down covariance after exact gate/up decode
        # (mirror of glm53_prepared_backend.py:490-492).
        codec._codec().clear_caches()

        # ONE conditioning context for the whole down curve, identical to the
        # probe and the DP allocation (R7 pair_at semantics,
        # r7_encoder/layer.py:901-925, 974-994): the down Hessian is
        # conditioned on gate/up reconstructions decoded at the reference
        # rates k35.FLOOR_BITS regardless of the allocated gate/up rates.
        # When an allocated rate differs, the mismatched tensor is
        # additionally encoded at the reference rate for the conditioning
        # only; the deployed choice still carries the allocated rate, and
        # the receipt records both.
        reference_bits = common.k35.FLOOR_BITS
        conditioning: dict[str, Any] = {}
        for projection, weight in (
            ("gate_proj", gate_weight),
            ("up_proj", up_weight),
        ):
            if bits_by_projection[projection] == reference_bits:
                conditioning[projection] = encoded[projection]
            else:
                vectors = tensors_k4 if reference_bits == 4 else tensors_k3
                suh, svh = common.preparation_vectors(
                    vectors, projection, expert, args.device
                )
                conditioning[projection] = common.encode_one_rate(
                    codec,
                    layer=layer,
                    expert=expert,
                    projection=projection,
                    weight_hf=weight,
                    covariance=gate_cov,
                    bits=reference_bits,
                    suh=suh,
                    svh=svh,
                    provenance={
                        "claim_receipt_sha256": claim_sha,
                        "allocation_sha256": allocation_sha,
                        "conditioning_only": True,
                        "conditioning_reference_bits": reference_bits,
                        "public_shapleymcg_mixed_rate": True,
                        "global_allocator": False,
                    },
                )

        down_bits = bits_by_projection["down_proj"]
        down_cov, down_evidence = common.down_covariance(
            codec,
            capture,
            expert,
            conditioning["gate_proj"].reconstructed.t().contiguous(),
            conditioning["up_proj"].reconstructed.t().contiguous(),
            gate_bits=reference_bits,
            up_bits=reference_bits,
            device=args.device,
            chunk_rows=args.chunk_rows,
        )
        down_evidence["matrix_sha256"] = common.tensor_sha256(down_cov)
        down_evidence["gate_reconstruction_sha256"] = (
            conditioning["gate_proj"].reconstruction_sha256
        )
        down_evidence["up_reconstruction_sha256"] = (
            conditioning["up_proj"].reconstruction_sha256
        )

        vectors_down = tensors_k4 if down_bits == 4 else tensors_k3
        suh_down, svh_down = common.preparation_vectors(
            vectors_down, "down_proj", expert, args.device
        )
        down_candidate = common.encode_one_rate(
            codec,
            layer=layer,
            expert=expert,
            projection="down_proj",
            weight_hf=down_weight,
            covariance=down_cov,
            bits=down_bits,
            suh=suh_down,
            svh=svh_down,
            provenance={
                "claim_receipt_sha256": claim_sha,
                "allocation_sha256": allocation_sha,
                "public_shapleymcg_mixed_rate": True,
                "global_allocator": False,
            },
        )
        choices["down_proj"] = store.put_choice(
            layer=layer,
            expert=expert,
            projection="down_proj",
            bits=down_bits,
            choice_id=f"L{layer:03d}.E{expert:03d}.down_proj.K{down_bits}",
            trellis=down_candidate.packed,
            suh=suh_down,
            svh=svh_down,
            mcg=mcg_marker,
            reconstruction=down_candidate.reconstructed.half().contiguous(),
            vector_topology=VECTOR_TOPOLOGY["down_proj"],
            reader_abi_sha256=reader_abi,
            provenance={
                "claim_receipt_sha256": claim_sha,
                "allocation_sha256": allocation_sha,
                "bits": down_bits,
                "packed_sha256": down_candidate.packed_sha256,
                "reconstruction_sha256": down_candidate.reconstruction_sha256,
                "vector_rate": down_bits,
                "down_conditioning": {
                    "gate_bits": reference_bits,
                    "up_bits": reference_bits,
                    "semantics": "r7_pair_at_reference_rates_v1",
                },
                "gate_up_roundtrip_sha256": {
                    "gate": conditioning["gate_proj"].reconstruction_sha256,
                    "up": conditioning["up_proj"].reconstruction_sha256,
                },
            },
            predecessor_state_hash=predecessor,
        )

        hessian_artifact = save_hessians(
            layer_root,
            layer,
            expert,
            gate_cov,
            down_cov,
            {"gate_up": gate_up_evidence, "down": down_evidence},
        )
        receipt = common.build_expert_receipt(
            layer=layer,
            expert=expert,
            bits_by_projection=bits_by_projection,
            choices=choices,
            claim_receipt_sha256=claim_sha,
            allocation_sha256=allocation_sha,
            capture_binding=capture_binding,
            hessian_artifact=hessian_artifact,
            down_conditioning={
                "gate_rate": reference_bits,
                "up_rate": reference_bits,
                "down_rate": down_bits,
                "deployed_gate_rate": bits_by_projection["gate_proj"],
                "deployed_up_rate": bits_by_projection["up_proj"],
                "semantics": "r7_pair_at_reference_rates_v1",
                "note": (
                    "gate_rate/up_rate are the rates whose decoded "
                    "reconstructions conditioned the Hessian (the R7 pair_at "
                    "reference context shared with the probe and the DP "
                    "allocation); deployed rates are the shipped encodes"
                ),
                "evidence": down_evidence,
            },
            codec_identity_sha256=codec_identity_sha256,
        )
        common.write_json(receipt_path, receipt)
        common.verify_expert_receipt(receipt)
        for projection in common.PROJECTIONS:
            store.verify_choice(choices[projection])
        del (
            triplet,
            gate_weight,
            up_weight,
            down_weight,
            gate_cov,
            down_cov,
            encoded,
            conditioning,
        )
        if expert % 8 == 7:
            gc.collect()
            torch.cuda.empty_cache()

    expert_receipts = [
        common.verify_expert_receipt(
            common.load_json(expert_root / f"layer-{layer:03d}" / f"expert-{expert:03d}.json")
        )
        for expert in range(common.NUM_EXPERTS)
    ]
    layer_receipt = common.build_layer_receipt(
        layer=layer,
        worker_id=args.worker,
        claim_receipt_sha256=claim_sha,
        allocation_sha256=allocation_sha,
        expert_receipts=expert_receipts,
    )
    common.write_json(layer_root / "layer-receipt.json", layer_receipt)
    elapsed = time.monotonic() - started
    print(
        json.dumps(
            {
                "layer": layer,
                "worker": args.worker,
                "elapsed_seconds": round(elapsed, 1),
                "layer_receipt_sha256": layer_receipt["receipt_sha256"],
            }
        ),
        flush=True,
    )
    return layer_receipt


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.recover_worker is not None:
        recover_worker(args)
        return

    plan = common.load_plan(args.work_root)
    worker_row = common.verify_plan_worker(plan, args.worker)
    print(
        json.dumps(
            {
                "worker": args.worker,
                "preflight_variant": plan["preflight_variant"],
                "plan_cuda_visible_devices": worker_row.get("cuda_visible_devices"),
                "device": args.device,
                "note": (
                    "docker --gpus renumbers devices; verify CUDA_VISIBLE_DEVICES "
                    "against the enumeration visible inside this container"
                ),
            }
        ),
        flush=True,
    )

    source_root = common.resolve_source_root(args)
    codec = common.build_codec(source_root, common.resolve_extension(args), args.device)
    codec_identity_sha256 = common.require_hash(
        common.sha256_bytes(common.canonical_json(codec.identity)), "codec identity"
    )

    # Bind this worker to the phase-6 readiness receipt BEFORE any expert
    # receipt is sealed: the receipt's codec_identity_sha256 is derived from
    # the sealed rate-3 preparation manifests (checked identical across
    # layers and against the live codec by phase 6), so equality here means
    # the encode codec is the codec the preparations were built with, not an
    # unbound attestation.  The per_layer preparation seals are enforced per
    # layer inside encode_layer.
    readiness_path = args.work_root / "gss" / "readiness-receipt.json"
    if not readiness_path.is_file():
        die(f"the phase-6 readiness receipt is absent: {readiness_path}")
    readiness = common.load_json(readiness_path)
    common.verify_seal(
        readiness,
        schema=common.K35_READINESS_SCHEMA,
        field="readiness_receipt_sha256",
        label="phase-6 readiness receipt",
    )
    if readiness.get("launch_plan_sha256") != plan["launch_plan_sha256"]:
        die(
            "the phase-6 readiness receipt binds a different launch plan; "
            "re-run phase 6 after any phase-5b plan rebuild"
        )
    if readiness.get("codec_identity_sha256") != codec_identity_sha256:
        die(
            "the phase-6 readiness receipt codec identity differs from this "
            "worker's codec (extension/torch/environment differ); refusing to "
            "encode under a codec the preparations were not built with"
        )
    readiness_preparations: dict[int, dict[str, Any]] = {}
    for row in readiness.get("per_layer", []):
        if not isinstance(row, Mapping) or not isinstance(row.get("layer"), int):
            die("phase-6 readiness receipt per_layer list is malformed")
        readiness_preparations[int(row["layer"])] = dict(row)

    source = common.load_bf16_source(
        args.work_root, args.bf16_root, verify_shards=args.verify_shards
    )
    capture = None

    while True:
        active_claim = None
        with common.StateLock(args.work_root):
            current_plan = common.load_plan(args.work_root)
            if current_plan["launch_plan_sha256"] != plan["launch_plan_sha256"]:
                print(
                    json.dumps(
                        {
                            "worker": args.worker,
                            "plan_changed": True,
                            "note": (
                                "launch plan changed on disk (phase-5b "
                                "re-plan); exiting cleanly without new claims; "
                                "restart the worker on the new plan"
                            ),
                        }
                    ),
                    flush=True,
                )
                return
            _path, state = common.newest_state(args.work_root, plan)
            common.k35.verify_state(plan, state)
            phase = state["phase"]
            if phase == "planned":
                die(
                    "the state chain is still in phase 'planned'; run the phase-6 "
                    "readiness receipt and enter_k35_encoding first (README "
                    "phase 6b)"
                )
            if phase != "k35_main_encoding":
                print(
                    json.dumps(
                        {
                            "worker": args.worker,
                            "phase": phase,
                            "note": "encoding phase is closed; nothing to claim",
                        }
                    )
                )
                return
            mine = state["active_claims"].get(args.worker)
            if mine is not None:
                active_claim = dict(mine)
            elif (args.work_root / "PAUSE").exists():
                print(
                    json.dumps(
                        {
                            "worker": args.worker,
                            "drain_pause": True,
                            "note": "PAUSE file present; no new claim taken",
                        }
                    )
                )
                return
            elif not state["pending_layers"]:
                print(
                    json.dumps(
                        {
                            "worker": args.worker,
                            "pending": 0,
                            "note": "no unclaimed layers remain; ready for phase 8",
                        }
                    )
                )
                return
            else:
                successor, claim = common.k35.claim_next_layer(
                    plan, state, worker_id=args.worker
                )
                common.write_json(
                    common.state_path(args.work_root, successor["sequence"]), successor
                )
                active_claim = dict(claim)
                print(
                    json.dumps(
                        {
                            "worker": args.worker,
                            "claimed_layer": claim["layer"],
                            "state": successor["sequence"],
                        }
                    ),
                    flush=True,
                )

        layer = int(active_claim["layer"])
        capture = common.open_capture(
            args.calibration_root,
            layer,
            verify_hashes=not args.no_verify_capture_hashes,
        )
        layer_receipt = encode_layer(
            args, codec, source, capture, plan, active_claim, readiness_preparations
        )
        del capture

        with common.StateLock(args.work_root):
            _path, newest = common.newest_state(args.work_root, plan)
            common.k35.verify_state(plan, newest)
            claim_now = newest["active_claims"].get(args.worker)
            if (
                claim_now is None
                or claim_now.get("claim_receipt_sha256")
                != active_claim["claim_receipt_sha256"]
            ):
                die(
                    f"worker {args.worker} claim on layer {layer} vanished from the "
                    "newest state before completion; inspect the chain"
                )
            successor = common.k35.complete_layer(
                plan,
                newest,
                worker_id=args.worker,
                layer=layer,
                layer_receipt_sha256=layer_receipt["receipt_sha256"],
            )
            common.write_json(
                common.state_path(args.work_root, successor["sequence"]), successor
            )
            print(
                json.dumps(
                    {
                        "worker": args.worker,
                        "completed_layer": layer,
                        "completed_total": len(successor["completed_layers"]),
                        "state": successor["sequence"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
