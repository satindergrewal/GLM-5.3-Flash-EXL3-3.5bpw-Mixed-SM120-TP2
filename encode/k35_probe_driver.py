#!/usr/bin/env python3
"""Phase 5 probe driver: per-layer K3/K4 probe losses and the sealed
sensitivity-DP allocation for the GLM-5.3 mixed 3.5-bpw campaign.

One layer per invocation (layers 3..44 plus 45 = MTP):

  1. open the sealed calibration capture (main-ep4-full for 3..44,
     mtp45-ep4-full for 45) and the sealed BF16 source,
  2. per 288 experts x 3 projections encode BOTH rates (bits=(3, 4)) through
     Exl3MCGCodec.encode_candidates with the pinned campaign sigma_reg 0.025,
  3. score each candidate with the K4 campaign's covariance-proxy loss
     (k35_common.covariance_proxy_loss),
  4. write probes/L{NN}.json (NEW SURFACE ledger, sealed),
  5. DP-solve the exact 3024-bit-unit / 432-K4 allocation
     (k35_common.solve_layer_dp; see the WARN there),
  6. seal non-provisionally via k35.seal_layer_allocation(...,
     basis="sensitivity_dp_probe_v1") into allocations/L{NN}.json,
  7. print exactly one JSON line.

Loss definition (documented choice): the probe loss is the relative
covariance quadratic e^T C e / w^T C w, the loss the sealed K4 numeric
closure computes in Exl3TrellisCodec.encode (r7_encoder/trellis.py:383-396),
recomputed over the R10 candidate reconstruction.  The R10 path returns
proxy_loss=0.0 by design (r10_codec.py:512), so the driver evaluates the
identical formula; R10 documents byte-compatibility with the R7 audited
path, and the bridge is recorded in the ledger header.  Optional
cross-check (not run here): encode one tensor through the R7 audited path
and compare proxy_loss directly.

Down conditioning (documented choice): the down-projection curve is measured
under ONE conditioning context per expert, the R7 pair_at semantics
(r7_encoder/layer.py:901-925 memoizes one context per gate/up rate pair;
layer.py:974-994 runs every down bit width under pair_at(base_gate_bits,
base_up_bits)).  Both candidate rates (3 and 4) are conditioned on gate/up
decoded at the reference rates k35.FLOOR_BITS, so loss@3 and loss@4 are
same-denominator relative quadratics and the DP gain mass*(loss3-loss4)
subtracts one metric.  The encode worker conditions its encode-time down
Hessian on the identical context.  This is the fixed_point_iteration=0
probe; the lineage's iterate-to-fixed-point refinement
(r7_encoder/sensitivity.py ProbeLedger) is not ported here.

Probe-time vectors: the probe uses the K4-rate preparation vectors for BOTH
rates because the runbook sequences phase 5 (probe) before phase 6
(rate-specific GSS); the final encode uses rate-specific vectors.  The
preparation binding is recorded in the ledger.

Usage (inside the encode container, cwd <work>):

  python3 k35_probe_driver.py --layer 3 --work-root /mnt/t5evo/glm53-k35-work

ASCII only.  CODE ONLY: nothing here is executed by the author.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import k35_common as common
from k35_common import die


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="phase 5 k35 probe driver")
    parser.add_argument("--layer", required=True, type=int, help="3..44 plus 45 = MTP")
    common.add_common_args(parser)
    parser.add_argument(
        "--preparation-root",
        default=None,
        type=Path,
        help="K4-rate preparation root (default: <work-root>/gss/k4)",
    )
    args = parser.parse_args()
    common.finish_common_args(args)
    if args.layer not in common.ALL_PROBE_LAYERS:
        die(f"--layer must be one of {list(common.MAIN_LAYERS)} or {common.MTP_LAYER}")
    if args.preparation_root is None:
        args.preparation_root = args.work_root / "gss" / "k4"
    args.preparation_root = Path(args.preparation_root).resolve()
    return args


def role_row_counts(capture, expert: int) -> dict[str, int]:
    return {
        role: int(capture.routed_rows(expert, role).rows)
        for role in ("fit", "conditional-fit", "selection", "confirmation")
    }


def candidate_record(weight_hf, candidate, covariance) -> dict:
    loss = common.covariance_proxy_loss(weight_hf, candidate.reconstructed, covariance)
    return {
        "loss": format(loss, ".17g"),
        "packed_sha256": candidate.packed_sha256,
        "reconstruction_sha256": candidate.reconstruction_sha256,
        "stored_bytes": int(candidate.stored_bytes),
    }


def main() -> None:
    args = parse_args()
    layer = args.layer

    capture = common.open_capture(
        args.calibration_root, layer, verify_hashes=not args.no_verify_capture_hashes
    )
    source = common.load_bf16_source(
        args.work_root, args.bf16_root, verify_shards=args.verify_shards
    )
    preparation_manifest, preparation_tensors = common.load_preparation(
        args.preparation_root, layer, expected_bits=4
    )
    source_root = common.resolve_source_root(args)
    codec = common.build_codec(source_root, common.resolve_extension(args), args.device)
    codec_identity_sha256 = common.require_hash(
        common.sha256_bytes(common.canonical_json(codec.identity)), "codec identity"
    )

    from quant_pipeline.campaign.glm53_direct_k4 import tensor_name
    from quant_pipeline.normalization.prior_search import permute_expert_hf

    records = []
    masses = []
    for expert in range(common.NUM_EXPERTS):
        triplet = source.load_triplet(layer, expert, device=args.device)
        permutation = preparation_tensors["permutations"][expert].tolist()
        gate_weight, up_weight, down_weight = permute_expert_hf(
            triplet["gate_proj"], triplet["up_proj"], triplet["down_proj"], permutation
        )

        gate_cov, gate_cov_evidence = common.gate_covariance(
            codec, capture, expert, args.device, args.chunk_rows
        )
        gate_cov_evidence["matrix_sha256"] = common.tensor_sha256(gate_cov)

        # gate/up: encode BOTH rates per tensor, keeping the candidate
        # objects alive for the down conditioning decode.
        encoded_gate_up = {}
        projections_record = {}
        for projection, weight in (("gate_proj", gate_weight), ("up_proj", up_weight)):
            suh, svh = common.preparation_vectors(
                preparation_tensors, projection, expert, args.device
            )
            candidates = codec.encode_candidates(
                unit_id=f"L{layer}.E{expert}.{projection}",
                weight_hf=weight,
                covariance=gate_cov,
                bits=(3, 4),
                input_vector=suh,
                output_vector=svh,
                provenance={
                    "k35_probe": True,
                    "layer": layer,
                    "expert": expert,
                    "projection": projection,
                },
            )
            encoded_gate_up[projection] = candidates
            projections_record[projection] = {
                str(bits): candidate_record(weight, candidates[bits], gate_cov)
                for bits in (3, 4)
            }
            projections_record[projection]["covariance_matrix_sha256"] = (
                gate_cov_evidence["matrix_sha256"]
            )

        # Corrected operation order: a fresh factor domain for the
        # candidate-conditioned down covariance after exact gate/up decode
        # (mirror of glm53_prepared_backend.py:490-492).
        codec._codec().clear_caches()

        # ONE conditioning context for the whole down curve (R7 pair_at
        # semantics, r7_encoder/layer.py:901-925 and 974-994): gate/up decoded
        # at the reference rates k35.FLOOR_BITS for BOTH candidate rates, so
        # loss@3 and loss@4 share one Hessian and one denominator and the DP
        # gain mass*(loss3-loss4) subtracts same-metric losses.  The encode
        # worker conditions its encode-time Hessian on this exact context.
        reference_bits = common.k35.FLOOR_BITS
        gate_reference = encoded_gate_up["gate_proj"][reference_bits]
        up_reference = encoded_gate_up["up_proj"][reference_bits]
        # Candidate reconstructions are HF [N,K]
        # (codecs/exl3_mcg.py:197); down_inputs_from_roundtrip wants
        # [K,N], matching the backend's direct pass of reconstructed_kn
        # (glm53_prepared_backend.py:284-289).
        down_cov, down_cov_evidence = common.down_covariance(
            codec,
            capture,
            expert,
            gate_reference.reconstructed.t().contiguous(),
            up_reference.reconstructed.t().contiguous(),
            gate_bits=reference_bits,
            up_bits=reference_bits,
            device=args.device,
            chunk_rows=args.chunk_rows,
        )
        down_cov_evidence["matrix_sha256"] = common.tensor_sha256(down_cov)
        down_record = {
            "covariance_matrix_sha256": down_cov_evidence["matrix_sha256"],
            "covariance_evidence": down_cov_evidence,
            "conditioning": {
                "gate_bits": reference_bits,
                "up_bits": reference_bits,
                "semantics": "r7_pair_at_reference_rates_v1",
            },
        }
        for rate in (3, 4):
            suh, svh = common.preparation_vectors(
                preparation_tensors, "down_proj", expert, args.device
            )
            candidates = codec.encode_candidates(
                unit_id=f"L{layer}.E{expert}.down_proj",
                weight_hf=down_weight,
                covariance=down_cov,
                bits=(rate,),
                input_vector=suh,
                output_vector=svh,
                provenance={
                    "k35_probe": True,
                    "layer": layer,
                    "expert": expert,
                    "projection": "down_proj",
                    "conditioning_gate_bits": reference_bits,
                    "conditioning_up_bits": reference_bits,
                },
            )
            entry = candidate_record(down_weight, candidates[rate], down_cov)
            entry["gate_up_roundtrip_sha256"] = {
                "gate": gate_reference.reconstruction_sha256,
                "up": up_reference.reconstruction_sha256,
            }
            down_record[str(rate)] = entry
            del candidates
        projections_record["down_proj"] = down_record

        mass = common.expert_p2_mass(capture, expert)
        masses.append(Decimal(format(mass, ".17g")))
        records.append(
            {
                "expert": expert,
                "tensor_names": {
                    projection: tensor_name(layer, expert, projection)
                    for projection in common.PROJECTIONS
                },
                "p2_mass": format(mass, ".17g"),
                "row_counts": role_row_counts(capture, expert),
                "gate_up_covariance": gate_cov_evidence,
                "projections": projections_record,
            }
        )
        del (
            triplet,
            gate_weight,
            up_weight,
            down_weight,
            gate_cov,
            down_cov,
            encoded_gate_up,
        )
        if expert % 16 == 15:
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    ledger = common.seal(
        {
            "schema": common.K35_PROBE_LEDGER_SCHEMA,
            "layer": layer,
            "capture_binding": capture.binding(),
            "inventory_sha256": common.load_inventory(args.work_root)["inventory_sha256"],
            "codec_identity_sha256": codec_identity_sha256,
            "sigma_reg": common.SIGMA_REG,
            "loss_definition": (
                "relative covariance quadratic e^T C e / w^T C w; formula verbatim "
                "from r7_encoder/trellis.py:383-396 (Exl3TrellisCodec.encode), "
                "recomputed over the R10 candidate reconstruction "
                "(r10_codec.py:512 returns proxy_loss=0.0 by design); R10/R7 "
                "byte-compatibility is documented in the r10_codec.py module "
                "docstring"
            ),
            "probe_loss_bridge": "r10_candidate_reconstruction_plus_r7_quadratic_v1",
            "down_conditioning": (
                "conditional-fit Hessian conditioned on gate/up decoded at the "
                "reference rates k3/k3 (k35.FLOOR_BITS) for BOTH candidate "
                "rates; R7 pair_at semantics (r7_encoder/layer.py:901-925, "
                "974-994); the DP gain mass*(loss3-loss4) subtracts "
                "same-denominator ratios; the encode worker uses the identical "
                "context; fixed_point_iteration=0"
            ),
            "fixed_point_iteration": 0,
            "probe_vectors": (
                "K4-rate preparation vectors for both rates; phase 5 precedes "
                "phase 6 rate-specific GSS by runbook sequencing"
            ),
            "preparation_binding": {
                "root": str(args.preparation_root),
                "preparation_sha256": preparation_manifest["preparation_sha256"],
                "bits": 4,
            },
            "records": records,
        },
        "probe_sha256",
    )
    from quant_pipeline.core.artifacts import write_json

    write_json(args.work_root / "probes" / f"{common.probe_stem(layer)}.json", ledger)

    loss_by_bits = {}
    for record in records:
        for projection in common.PROJECTIONS:
            name = record["tensor_names"][projection]
            loss3 = Decimal(record["projections"][projection]["3"]["loss"])
            loss4 = Decimal(record["projections"][projection]["4"]["loss"])
            loss_by_bits[name] = (loss3, loss4)
    allocation_bits = common.solve_layer_dp(layer, loss_by_bits, masses)
    allocation = common.k35.seal_layer_allocation(
        layer, allocation_bits, provisional=False, basis="sensitivity_dp_probe_v1"
    )
    write_json(
        args.work_root / "allocations" / f"{common.probe_stem(layer)}.json", allocation
    )

    worst_k3_loss = max(
        float(record["projections"][projection]["3"]["loss"])
        for record in records
        for projection in common.PROJECTIONS
        if allocation_bits[record["tensor_names"][projection]] == 3
    )
    print(
        json.dumps(
            {
                "layer": layer,
                "k4_tensors": allocation["k4_tensor_count"],
                "worst_k3_loss": worst_k3_loss,
                "alloc_sha256": allocation["allocation_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
