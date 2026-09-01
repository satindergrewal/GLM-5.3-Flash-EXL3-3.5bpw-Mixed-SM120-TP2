#!/usr/bin/env python3
"""Phase 6: rate-specific GSS preparation at BOTH rates (K3 and K4, no
cross-rate reuse) plus the k35 readiness receipt.

Readiness contract search result (grep over src/, scripts/, results/,
docs/): the repo defines NO readiness schema and NO readiness builder.  The
only traces are the evidence hash field "k35_readiness_receipt_sha256"
(glm53_uniform_k35.py:856 during verify_state, :897-899 in
enter_k35_encoding; the K4 analogue "k4_readiness_receipt_sha256" at
glm53_uniform_k4.py:475, :516-517) and the plan's preparation_contract block
(glm53_uniform_k35.py:599-605): rate_specific_gss_required,
reuse_k4_gss_forbidden, k3_and_k4_gss_both_required,
candidate_conditioned_down_uses_decoded_gate_up_at_matching_rate.

WARN: the readiness receipt schema defined here
(quant-pipeline.glm53-k35-readiness-receipt.v1) is NEW SURFACE.  Nothing in
the sealed campaign validates it.  It is a minimal sealed binding document:
launch plan, both rate-specific GSS preparation sets, the shared
permutation identity, the transform seed, the codec identity, and the
phase-5 allocation set.  Its seal hash is what enter_k35_encoding accepts
as readiness_receipt_sha256 (any 64-hex hash passes the state machine; this
document is what the hash promises).

Rate halves:
  K4 half: verify existing sealed K4 preparations (schema
    quant-pipeline.glm53-public-shapleymcg-layer-preparation.v1, bits=4) at
    --k4-preparation-root, or build them through the SEALED builder
    glm53_mcg_preparation.build_layer_preparation (legal at bits=4) when
    --build-k4 and --profile-selection are given.
  K3 half: NEW SURFACE port.  The sealed builder hard-rejects bits=3
    (glm53_mcg_preparation.py:39 SUPPORTED_BITS=(4,6), :278-279) and its
    profile-selection verifier binds a sealed (4,6) receipt (:57-85, :456-
    458).  The GSS search itself is bits-agnostic at the core
    (CorrectedPinnedGSSProducer.search -> backend._quant_args(int(bits), ...)
    -> g_scale_gss; the R10 quant_args accept 3/4/5,
    r10_codec.py:392-393).  This driver therefore replays the sealed
    builder's own numeric path (its private helpers _p2_profile_statistics
    and _matrix_inputs_for_expert, the StreamingLayerFitter, and
    CorrectedPinnedGSSProducer) at bits=3 and stores the result under a k35
    schema.  The transform POLICY and scale FAMILY (energy_balanced,
    per128-grid) and the transform seed are borrowed from the K4-rate sealed
    preparation: the policy is rate-independent; what makes GSS
    rate-specific is the 13-point golden-section search run at bits=3.
    Reusing the K4-fitted VECTORS for K3 tensors is the forbidden thing and
    this port never does it; permutations are additionally checked identical
    across the two halves.

Usage (inside the encode container, cwd <work>):

  python3 k35_phase6_gss.py --work-root /mnt/t5evo/glm53-k35-work

ASCII only.  CODE ONLY: nothing here is executed by the author.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import k35_common as common
from k35_common import die

ZERO_HASH = "0" * 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="phase 6 k35 rate-specific GSS")
    parser.add_argument(
        "--layers",
        default="3-44",
        help="layer range or comma list, e.g. 3-44 (default) or 3-44,45",
    )
    common.add_common_args(parser)
    parser.add_argument(
        "--k4-preparation-root", default=None, type=Path, help="default <work>/gss/k4"
    )
    parser.add_argument(
        "--k3-output-root", default=None, type=Path, help="default <work>/gss/k3"
    )
    parser.add_argument(
        "--profile-selection",
        default=None,
        type=Path,
        help="sealed bits=4 profile-selection receipt; required only with --build-k4",
    )
    parser.add_argument(
        "--transform-seed-sha256",
        default=None,
        help="default: the K4 preparation's transform seed (recommended)",
    )
    parser.add_argument(
        "--build-k4",
        action="store_true",
        help="build missing K4 preparations through the sealed builder",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify both halves without building anything",
    )
    args = parser.parse_args()
    common.finish_common_args(args)
    if args.k4_preparation_root is None:
        args.k4_preparation_root = args.work_root / "gss" / "k4"
    if args.k3_output_root is None:
        args.k3_output_root = args.work_root / "gss" / "k3"
    args.k4_preparation_root = Path(args.k4_preparation_root).resolve()
    args.k3_output_root = Path(args.k3_output_root).resolve()

    layers: list[int] = []
    for token in args.layers.split(","):
        token = token.strip()
        if "-" in token:
            low, high = token.split("-", 1)
            layers.extend(range(int(low), int(high) + 1))
        else:
            layers.append(int(token))
    for layer in layers:
        if layer not in common.ALL_PROBE_LAYERS:
            die(f"layer {layer} outside 3..44 plus 45")
    if len(set(layers)) != len(layers):
        die("duplicate layer in --layers")
    args.layer_list = sorted(set(layers))
    if args.build_k4 and args.profile_selection is None:
        die("--build-k4 requires --profile-selection (sealed bits=4 receipt)")
    return args


def verify_k4_preparation(root: Path, layer: int) -> dict[str, Any]:
    manifest, _tensors = common.load_preparation(root, layer, expected_bits=4)
    return manifest


# ---------------------------------------------------------------------------
# Producer source closure over the released tree
#
# WARN (closure surface change): the sealed builder's _producer_closure
# (glm53_mcg_preparation.py:234-255) requires scripts/run_qwen_fast_encode.py
# plus src/quant_pipeline/{normalization/streaming_v31.py,codecs/exl3_mcg.py}
# under the passed source root, and resolve_source_root can only ever return
# the r10 bundle root (the only tree carrying r7_encoder/r10_codec.py).  The
# released deliverable contains scripts/run_qwen_fast_encode.py NOWHERE
# (repo-wide find), and the sealed profile-selection verifier pins its sha256
# (glm53_mcg_preparation.py:81-82), so the sealed two-predicate closure is
# jointly unsatisfiable over everything shipped: phase 6 would deterministically
# die at the closure AFTER the full 288-expert GPU pass, and the sealed
# --build-k4 subprocess path dies identically inside build_layer_preparation.
# This derivation keeps the sealed helper's hash framing over the producer
# sources that DO exist, records the missing file as a loud manifest warning,
# and is computed BEFORE the GPU pass so an unsatisfiable surface fails in
# seconds.  The sealed surface is restored by vendoring the genuine
# scripts/run_qwen_fast_encode.py at the pinned sha (plus the vendored copies
# below) under the r10 bundle root.
# ---------------------------------------------------------------------------

_RUN_QWEN_FAST_ENCODE_SHA256 = (
    "ceea8c64d63ffb60cdf95adee3ba7b488c54303d3a85502798b2c3fd0fcbb492"
)

_SEALED_CLOSURE_RELATIVE_PATHS = (
    "scripts/run_qwen_fast_encode.py",
    "src/quant_pipeline/normalization/streaming_v31.py",
    "src/quant_pipeline/codecs/exl3_mcg.py",
)


def vendored_producer_sources() -> dict[str, Path]:
    """Vendored copies of the producer sources that exist in the released repo."""

    vendored_root = Path(__file__).resolve().parent / "vendored" / "src" / "quant_pipeline"
    sources = {
        "src/quant_pipeline/normalization/streaming_v31.py": vendored_root
        / "normalization"
        / "streaming_v31.py",
        "src/quant_pipeline/codecs/exl3_mcg.py": vendored_root / "codecs" / "exl3_mcg.py",
    }
    for path in sources.values():
        if not path.is_file():
            die(f"vendored producer source is absent: {path}")
    return sources


def producer_closure_over_released_tree(
    source_root: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """_producer_closure derived over the files that DO exist (see WARN)."""

    import quant_pipeline.codecs.exl3_mcg as exl3_mcg_module
    import quant_pipeline.normalization.streaming_v31 as streaming_module
    from quant_pipeline.campaign import glm53_mcg_preparation as sealed_prep

    vendored = vendored_producer_sources()
    live = {
        "src/quant_pipeline/normalization/streaming_v31.py": Path(
            streaming_module.__file__
        ),
        "src/quant_pipeline/codecs/exl3_mcg.py": Path(exl3_mcg_module.__file__),
    }
    bundle = sorted((source_root / "r7_encoder").rglob("*.py"))
    if not bundle:
        die(f"the r7_encoder numeric closure is absent under {source_root}")
    records: list[dict[str, str]] = [
        {
            "path": path.relative_to(source_root).as_posix(),
            "sha256": common.sha256_file(path),
        }
        for path in bundle
    ]
    for name in sorted(vendored):
        vendored_sha = common.sha256_file(vendored[name])
        if vendored_sha != common.sha256_file(live[name]):
            die(
                f"vendored producer source {name} differs from the imported "
                "campaign module; re-vendor the exact campaign file"
            )
        records.append({"path": name, "sha256": vendored_sha})
    records.append(
        {
            "path": (
                "campaign-adapter/src/quant_pipeline/campaign/"
                "glm53_mcg_preparation.py"
            ),
            "sha256": common.sha256_file(Path(sealed_prep.__file__)),
        }
    )
    warning = {
        "surface_changed": True,
        "missing_producer_sources": ["scripts/run_qwen_fast_encode.py"],
        "missing_producer_note": (
            "absent from the released deliverable (repo-wide find); the "
            "sealed profile-selection verifier pins its sha256 at "
            f"{_RUN_QWEN_FAST_ENCODE_SHA256}; this closure is derived over "
            "the producer sources that exist, and the sealed "
            "_producer_closure surface cannot be satisfied until the genuine "
            "file is vendored under the source root"
        ),
    }
    return records, warning


def preflight_sealed_closure(source_root: Path) -> None:
    """Fail fast when the sealed builder's closure is unsatisfiable.

    --build-k4 reaches the sealed build_layer_preparation, whose own
    _producer_closure (glm53_mcg_preparation.py:405) would raise only AFTER
    the full GPU pass.  This pre-flight checks the exact sealed surface and
    dies in seconds instead; it never relaxes the sealed check.
    """

    missing = [
        relative
        for relative in _SEALED_CLOSURE_RELATIVE_PATHS
        if not (source_root / relative).is_file()
    ]
    if not missing and not list((source_root / "r7_encoder").glob("*.py")):
        missing.append("r7_encoder/*.py")
    if missing:
        die(
            f"the sealed K4 builder's producer closure is unsatisfiable under "
            f"{source_root} (missing {missing}); vendor the genuine files "
            "(scripts/run_qwen_fast_encode.py is pinned at sha256 "
            f"{_RUN_QWEN_FAST_ENCODE_SHA256}; streaming_v31.py and exl3_mcg.py "
            "ship in vendored/) under the source root BEFORE the GPU pass "
            "(README-K35-DRIVERS.md phase 6)"
        )


def sweep_abandoned_staging(output: Path, layer: int) -> None:
    """Remove staging directories whose owner process is gone.

    Staging directories are PID-named, so every crashed build leaves one
    behind and retries accumulate orphans.  A live owner means another
    phase-6 process is building this layer right now, which is not a
    supported mode: die loudly.
    """

    import shutil

    for candidate in sorted(output.glob(f".layer-{layer:03d}.staging-*")):
        token = candidate.name.rsplit("-", 1)[-1]
        if not token.isdigit() or int(token) <= 0:
            die(f"foreign staging entry in the rate-3 output root: {candidate}")
        owner = int(token)
        try:
            os.kill(owner, 0)
        except ProcessLookupError:
            shutil.rmtree(candidate)
            continue
        except OSError:
            die(
                f"staging directory {candidate.name} has a live or inaccessible "
                f"owner (pid {owner}); concurrent phase-6 builds are not supported"
            )
        die(
            f"staging directory {candidate.name} has a live owner (pid {owner}); "
            "concurrent phase-6 builds are not supported"
        )


_K4_BUILD_SUBPROCESS = r'''
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import k35_common as common
from quant_pipeline.campaign import glm53_mcg_preparation as sealed_prep

(
    _driver_dir, work_root, calibration_root, bf16_root, k4_root,
    profile_selection_path, source_root, extension, numeric_core, seed,
    device, layer_text, chunk_text,
) = sys.argv[1:]
layer = int(layer_text)
capture = common.open_capture(Path(calibration_root), layer)
source = common.load_bf16_source(Path(work_root), Path(bf16_root), verify_shards=False)
result = sealed_prep.build_layer_preparation(
    layer=layer,
    capture=capture,
    source=source,
    source_root=Path(source_root),
    numeric_core=Path(numeric_core),
    extension=Path(extension),
    output_root=Path(k4_root),
    transform_seed_sha256=seed,
    profile_selection=json.loads(Path(profile_selection_path).read_text()),
    device=device,
    chunk_rows=int(chunk_text),
    bits=4,
)
print(result["preparation_sha256"])
'''


def build_k4_preparation(args: argparse.Namespace, layer: int) -> dict[str, Any]:
    """K4 half through the SEALED builder (legal at bits=4), in a subprocess.

    The sealed builder constructs its own codec and triggers its own sealed
    r7_encoder import (glm53_mcg_preparation.py:297-304).  Running it in
    this process would collide with this driver's codec on the cached-module
    guard (codecs/exl3_mcg.py:127-129 refuses any cached r7_encoder) in one
    direction or the other.  A subprocess isolates the two constructions;
    the parent then re-verifies the manifest from disk.
    """

    import os
    import subprocess

    source_root = common.resolve_source_root(args)
    preflight_sealed_closure(source_root)
    extension = common.resolve_extension(args)
    driver_dir = Path(__file__).resolve().parent
    command = [
        sys.executable,
        "-c",
        _K4_BUILD_SUBPROCESS,
        str(driver_dir),
        str(args.work_root),
        str(args.calibration_root),
        str(args.bf16_root),
        str(args.k4_preparation_root),
        str(Path(args.profile_selection).resolve()),
        str(source_root),
        str(extension),
        str(common.numeric_core_path(source_root)),
        str(args.transform_seed_sha256_resolved),
        str(args.device),
        str(layer),
        str(args.chunk_rows),
    ]
    environment = os.environ.copy()
    completed = subprocess.run(command, env=environment, capture_output=True, text=True)
    if completed.returncode != 0:
        die(
            f"sealed K4 preparation build failed for layer {layer} "
            f"(exit {completed.returncode}): {completed.stderr.strip()[-2000:]}"
        )
    print(completed.stdout.strip(), flush=True)
    return verify_k4_preparation(args.k4_preparation_root, layer)


def build_rate3_preparation(
    args: argparse.Namespace,
    codec,
    capture,
    source,
    layer: int,
    policy: str,
    family: str,
) -> dict[str, Any]:
    """K3 half: NEW SURFACE port of build_layer_preparation at bits=3.

    Numeric path mirror of glm53_mcg_preparation.build_layer_preparation
    (lines 296-447): statistics, shared scales, streaming fit, per-matrix
    pinned GSS at bits=3, FP16 vectors, sealed manifest.  Differences from
    the sealed builder, all deliberate and loud:
      - bits=3 (the sealed builder would raise, :278-279);
      - no sealed profile-selection receipt exists at bits=3, so the policy
        and family are borrowed from the K4-rate sealed preparation and the
        manifest carries a k35 schema instead;
      - the manifest records the new-surface warning verbatim.
    """

    import torch
    from safetensors.torch import save_file

    from quant_pipeline.campaign import glm53_mcg_preparation as sealed_prep
    from quant_pipeline.campaign.glm53_mcg_preparation import (
        _block_values,
        _matrix_inputs_for_expert,
        _p2_profile_statistics,
    )
    from quant_pipeline.campaign.qwen_services import CorrectedPinnedGSSProducer
    from quant_pipeline.normalization.artifact_v31 import (
        PinnedGSSRequest,
        tensor_identity_sha256,
        tensor_sha256,
    )
    from quant_pipeline.normalization.prior_search import scale_family_candidates
    from quant_pipeline.normalization.streaming_v31 import (
        FitSamplePlan,
        FitSampleSpec,
        StreamingLayerFitter,
    )

    seed = args.transform_seed_sha256_resolved
    output = args.k3_output_root
    final_destination = output / f"layer-{layer:03d}"
    manifest_path = final_destination / "preparation.json"
    if manifest_path.exists():
        manifest, _tensors = common.load_preparation(output, layer, expected_bits=3)
        return manifest
    if args.verify_only:
        die(f"verify-only: rate-3 preparation is absent for layer {layer}")
    if final_destination.exists():
        die(f"incomplete final rate-3 preparation directory exists: {final_destination}")
    output.mkdir(parents=True, exist_ok=True)
    sweep_abandoned_staging(output, layer)
    staging = output / f".layer-{layer:03d}.staging-{os.getpid()}"
    staging.mkdir(exist_ok=False)

    # Compute the producer closure BEFORE the GPU pass: the sealed helper's
    # surface is unsatisfiable over the released tree (see the WARN above the
    # closure helpers), and this derivation fails in seconds instead of after
    # the 288-expert statistics/fit/GSS pass.
    source_root = common.resolve_source_root(args)
    closure, closure_warning = producer_closure_over_released_tree(source_root)

    backend = codec._codec()
    producer = CorrectedPinnedGSSProducer(codec)
    statistics = _p2_profile_statistics(
        capture=capture, source=source, layer=layer, device=args.device, chunk_rows=args.chunk_rows
    )
    shared_gate_scales = scale_family_candidates(
        _block_values(statistics["shared_gate_diagonal"].numpy())
    )[family]
    shared_down_scales = scale_family_candidates(
        _block_values(statistics["shared_down_output_energy"].numpy())
    )[family]

    specs: list[Any] = []
    permutations = torch.empty((common.NUM_EXPERTS, common.INTERMEDIATE_SIZE), dtype=torch.int64)
    for expert in range(common.NUM_EXPERTS):
        matrices, permutation = _matrix_inputs_for_expert(
            source=source,
            layer=layer,
            expert=expert,
            device=args.device,
            policy=policy,
            family=family,
            seed=seed,
            statistics=statistics,
            shared_gate_scales=shared_gate_scales,
            shared_down_scales=shared_down_scales,
            bits=3,
        )
        permutations[expert].copy_(torch.tensor(permutation, dtype=torch.int64))
        specs.extend(FitSampleSpec.from_input(matrix) for matrix in matrices)
        del matrices
    plan = FitSamplePlan.from_specs(specs, block=128)
    fitter = StreamingLayerFitter(
        backend.core,
        plan,
        codebook_scale=float(backend.codebook_scale),
        numeric_core_sha256=codec.identity["numeric_core_sha256"],
    )
    for expert in range(common.NUM_EXPERTS):
        matrices, _permutation = _matrix_inputs_for_expert(
            source=source,
            layer=layer,
            expert=expert,
            device=args.device,
            policy=policy,
            family=family,
            seed=seed,
            statistics=statistics,
            shared_gate_scales=shared_gate_scales,
            shared_down_scales=shared_down_scales,
            bits=3,
        )
        for matrix in matrices:
            fitter.add_fit_matrix(matrix)
        del matrices
    fit = fitter.finish()

    vectors = {
        "gate_suh": torch.empty((common.NUM_EXPERTS, common.HIDDEN_SIZE), dtype=torch.float16),
        "gate_svh": torch.empty((common.NUM_EXPERTS, common.INTERMEDIATE_SIZE), dtype=torch.float16),
        "up_suh": torch.empty((common.NUM_EXPERTS, common.HIDDEN_SIZE), dtype=torch.float16),
        "up_svh": torch.empty((common.NUM_EXPERTS, common.INTERMEDIATE_SIZE), dtype=torch.float16),
        "down_suh": torch.empty((common.NUM_EXPERTS, common.INTERMEDIATE_SIZE), dtype=torch.float16),
        "down_svh": torch.empty((common.NUM_EXPERTS, common.HIDDEN_SIZE), dtype=torch.float16),
    }
    gss_receipts: list[dict[str, Any]] = []
    for expert in range(common.NUM_EXPERTS):
        matrices, _permutation = _matrix_inputs_for_expert(
            source=source,
            layer=layer,
            expert=expert,
            device=args.device,
            policy=policy,
            family=family,
            seed=seed,
            statistics=statistics,
            shared_gate_scales=shared_gate_scales,
            shared_down_scales=shared_down_scales,
            bits=3,
        )
        for matrix in matrices:
            prepared = fit.prepare_matrix(matrix)
            target = prepared.gss_target()
            result = producer.search(
                PinnedGSSRequest(
                    matrix_key=matrix.key,
                    bits=3,
                    target=target,
                    target_sha256=tensor_sha256(target),
                    source_weight_identity_sha256=tensor_identity_sha256(matrix.weight_kn),
                    predecessor_checkpoint_hash=ZERO_HASH,
                )
            )
            finalized = prepared.finalize(
                prepared.bind_gss(result.scale), materialize_regularized=False
            )
            prefix = matrix.projection.removesuffix("_proj")
            vectors[f"{prefix}_suh"][expert].copy_(finalized.stored_suh.detach().cpu())
            vectors[f"{prefix}_svh"][expert].copy_(finalized.stored_svh.detach().cpu())
            gss_receipts.append(
                {
                    "expert": expert,
                    "projection": matrix.projection,
                    "scale": float(result.scale),
                    "receipt_sha256": result.receipt["receipt_sha256"],
                    "suh_sha256": finalized.suh_sha256,
                    "svh_sha256": finalized.svh_sha256,
                }
            )
        del matrices

    shard = staging / "preparation.safetensors"
    save_file(
        {"permutations": permutations, **vectors},
        str(shard),
        metadata={"schema": common.K35_RATE3_GSS_SCHEMA, "layer": str(layer), "bits": "3"},
    )
    decision_stats = staging / "profile-decision-statistics.safetensors"
    save_file(
        {
            "gate_p2_diagonal": statistics["gate_diagonal"],
            "source_down_p2_diagonal": statistics["down_diagonal"],
            "p2_mass": statistics["masses"],
            "source_down_output_energy": statistics["down_output_energy"],
        },
        str(decision_stats),
        metadata={
            "schema": common.K35_RATE3_GSS_SCHEMA,
            "purpose": "lossless-selected-transform-decision-statistics",
        },
    )
    body = {
        "schema": common.K35_RATE3_GSS_SCHEMA,
        "new_surface_warning": (
            "k35 rate-3 GSS preparation: the sealed builder rejects bits=3 "
            "(glm53_mcg_preparation.py:39,278-279) and no bits=3 profile-selection "
            "receipt exists; this manifest replays the sealed builder's numeric "
            "path at bits=3 with the policy, scale family, and transform seed "
            "borrowed from the K4-rate sealed preparation.  No sealed validator "
            "knows this schema."
        ),
        "complete": True,
        "layer": layer,
        "bits": 3,
        "codec_family": "exl3-mcg",
        "policy": policy,
        "scale_family": family,
        "policy_provenance": "borrowed_from_k4_rate_sealed_profile_selection",
        "rate_specific_gss": True,
        "reuse_k4_gss_forbidden": True,
        "k3_and_k4_gss_both_required": True,
        "transform_seed_sha256": seed,
        "streaming_fit_plan_sha256": plan.content_sha256,
        "shared_gate_up_suh_sha256": fit.shared_gate_up_suh_sha256,
        "shared_down_svh_sha256": fit.shared_down_svh_sha256,
        "permutation_set_sha256": common.tensor_sha256(permutations),
        "gss_receipts_sha256": common.sha256_bytes(common.canonical_json(gss_receipts)),
        "gss_receipt_count": len(gss_receipts),
        "profile_fit_row_evidence_sha256": common.sha256_bytes(
            common.canonical_json(statistics["row_evidence"])
        ),
        "producer_source_closure": closure,
        "producer_source_closure_sha256": common.sha256_bytes(
            common.canonical_json(closure)
        ),
        "producer_source_closure_surface_changed": closure_warning["surface_changed"],
        "producer_source_closure_missing": closure_warning["missing_producer_sources"],
        "producer_source_closure_note": closure_warning["missing_producer_note"],
        "codec_identity": codec.identity,
        "shard": shard.name,
        "shard_sha256": common.sha256_file(shard),
        "decision_statistics": decision_stats.name,
        "decision_statistics_sha256": common.sha256_file(decision_stats),
        "exact_production_hessians": (
            "recomputed_from_sealed_raw_capture_and_sealed_packed_gate_up"
        ),
    }
    result = common.seal(body, "preparation_sha256")
    common.write_json(staging / "preparation.json", result)
    os.replace(staging, final_destination)
    return result


def main() -> None:
    args = parse_args()
    plan = common.load_plan(args.work_root)

    # Resolve the transform seed and the policy/family: from the first
    # available K4 preparation when present, else (building K4 from
    # scratch) from the sealed profile-selection receipt plus a required
    # explicit seed.  They must be shared across layers and rates.
    reference_layer = args.layer_list[0]
    k4_reference_path = (
        args.k4_preparation_root / f"layer-{reference_layer:03d}" / "preparation.json"
    )
    if k4_reference_path.exists():
        k4_reference = verify_k4_preparation(args.k4_preparation_root, reference_layer)
        seed = args.transform_seed_sha256 or k4_reference["transform_seed_sha256"]
        policy = k4_reference["policy"]
        family = k4_reference["scale_family"]
    elif args.build_k4:
        selection = common.load_json(args.profile_selection)
        if selection.get("policy") != "energy_balanced" or (
            selection.get("scale_family") != "per128-grid"
        ):
            die("profile selection carries a foreign policy/family")
        if selection.get("bits") != 4:
            die("profile selection must be the sealed bits=4 receipt")
        policy = "energy_balanced"
        family = "per128-grid"
        if not args.transform_seed_sha256:
            die(
                "no K4 preparation exists yet; pass --transform-seed-sha256 (the "
                "one used by the K4 campaign) before building from scratch"
            )
        seed = args.transform_seed_sha256
    else:
        die(
            f"K4 preparation is absent for layer {reference_layer} at "
            f"{args.k4_preparation_root}; pass --build-k4 --profile-selection "
            "to build it through the sealed builder"
        )
    common.require_hash(seed, "transform seed")
    args.transform_seed_sha256_resolved = seed
    if policy != "energy_balanced" or family != "per128-grid":
        die(f"K4 preparation carries a foreign policy/family: {policy}/{family}")

    codec = None
    source = None
    per_layer = []
    k3_identity_shas: set[str] = set()
    k4_identity_shas: set[str] = set()
    for layer in args.layer_list:
        capture = common.open_capture(
            args.calibration_root, layer, verify_hashes=not args.no_verify_capture_hashes
        )
        if source is None:
            source = common.load_bf16_source(
                args.work_root, args.bf16_root, verify_shards=args.verify_shards
            )

        k4_manifest_path = args.k4_preparation_root / f"layer-{layer:03d}" / "preparation.json"
        if k4_manifest_path.exists():
            k4_manifest = verify_k4_preparation(args.k4_preparation_root, layer)
        elif args.build_k4:
            k4_manifest = build_k4_preparation(args, layer)
        else:
            die(
                f"K4 preparation is absent for layer {layer} at "
                f"{args.k4_preparation_root}; pass --build-k4 --profile-selection "
                "to build it through the sealed builder"
            )
        # Semantic binding on EVERY layer's K4 manifest, not just the
        # reference layer: load_preparation already enforces the sealed
        # per-manifest semantic fields; this pins the shared policy, scale
        # family, and transform seed across the whole set.
        if (
            k4_manifest.get("policy") != policy
            or k4_manifest.get("scale_family") != family
            or k4_manifest.get("transform_seed_sha256") != seed
        ):
            die(
                f"layer {layer} K4 preparation carries a foreign "
                f"policy/family/seed: {k4_manifest.get('policy')}/"
                f"{k4_manifest.get('scale_family')}"
            )
        k4_identity_shas.add(
            common.sha256_bytes(common.canonical_json(k4_manifest["codec_identity"]))
        )

        k3_manifest_path = args.k3_output_root / f"layer-{layer:03d}" / "preparation.json"
        if k3_manifest_path.exists():
            k3_manifest, _tensors = common.load_preparation(
                args.k3_output_root, layer, expected_bits=3
            )
        else:
            if codec is None:
                codec = common.build_codec(
                    common.resolve_source_root(args),
                    common.resolve_extension(args),
                    args.device,
                )
            k3_manifest = build_rate3_preparation(
                args, codec, capture, source, layer, policy, family
            )
        k3_identity_shas.add(
            common.sha256_bytes(common.canonical_json(k3_manifest["codec_identity"]))
        )

        import torch

        _m4, tensors_k4 = common.load_preparation(
            args.k4_preparation_root, layer, expected_bits=4
        )
        _m3, tensors_k3 = common.load_preparation(
            args.k3_output_root, layer, expected_bits=3
        )
        if not torch.equal(tensors_k4["permutations"], tensors_k3["permutations"]):
            die(f"layer {layer}: K3/K4 permutations differ across rate halves")
        if k3_manifest["transform_seed_sha256"] != seed:
            die(f"layer {layer}: rate-3 transform seed differs from the K4 reference")

        per_layer.append(
            {
                "layer": layer,
                "k4_preparation_sha256": k4_manifest["preparation_sha256"],
                "k3_preparation_sha256": k3_manifest["preparation_sha256"],
                "permutation_identity": True,
            }
        )
        print(
            json.dumps(
                {
                    "layer": layer,
                    "k4_prep": k4_manifest["preparation_sha256"][:16],
                    "k3_prep": k3_manifest["preparation_sha256"][:16],
                }
            ),
            flush=True,
        )

    allocations = []
    for layer in args.layer_list:
        receipt = common.load_layer_allocation(args.work_root, layer)
        allocations.append(
            {
                "layer": layer,
                "allocation_sha256": receipt["allocation_sha256"],
                "provisional": receipt["provisional"],
            }
        )
    if any(row["provisional"] for row in allocations):
        die("a layer allocation is still provisional; re-run phase 5 first")

    # Derive the receipt's codec identity from the artifacts being sealed,
    # not from incidental execution state: every rate-3 manifest's embedded
    # codec_identity must agree (the encode-venue identity the phase-7
    # worker must reproduce), and when a live codec was constructed this
    # invocation it must reproduce the same identity.  The K4 half's
    # identity (the original campaign venue) is recorded under its own
    # field instead of silently masquerading as the encode identity when no
    # rate-3 build ran.
    if len(k3_identity_shas) != 1:
        die(
            "rate-3 preparations do not share one codec identity: "
            f"{sorted(k3_identity_shas)}"
        )
    codec_identity_sha256 = next(iter(k3_identity_shas))
    if len(k4_identity_shas) != 1:
        die(
            "K4 preparations do not share one codec identity: "
            f"{sorted(k4_identity_shas)}"
        )
    k4_codec_identity_sha256 = next(iter(k4_identity_shas))
    if codec is not None:
        live_identity_sha256 = common.sha256_bytes(
            common.canonical_json(codec.identity)
        )
        if live_identity_sha256 != codec_identity_sha256:
            die(
                "the live codec identity differs from the sealed rate-3 "
                "preparation identities; this run's codec is not the codec "
                "the preparation set was built with"
            )
    body = {
        "schema": common.K35_READINESS_SCHEMA,
        "new_surface_warning": (
            "k35 readiness receipt: no readiness schema or builder exists in the "
            "sealed campaign (only the k35_readiness_receipt_sha256 evidence "
            "field, glm53_uniform_k35.py:856,897-899); this document is the "
            "binding the hash promises"
        ),
        "launch_plan_sha256": plan["launch_plan_sha256"],
        "preflight_variant": plan["preflight_variant"],
        "layers": [row["layer"] for row in per_layer],
        "per_layer": per_layer,
        "preparation_contract": {
            "rate_specific_gss_required": True,
            "reuse_k4_gss_forbidden": True,
            "k3_and_k4_gss_both_required": True,
            "candidate_conditioned_down_uses_decoded_gate_up_at_matching_rate": True,
            "reuse_raw_calibration_and_routes": True,
            "reuse_fixed_policy_and_permutations": True,
            "down_conditioning_context": {
                "gate_bits": common.k35.FLOOR_BITS,
                "up_bits": common.k35.FLOOR_BITS,
                "semantics": "r7_pair_at_reference_rates_v1",
                "one_context_for_whole_down_curve": True,
                "shared_by": [
                    "probe_loss_curve",
                    "dp_allocation",
                    "encode_time_hessian",
                ],
                "note": (
                    "matching rate in the R7 pair_at sense "
                    "(r7_encoder/layer.py:901-925, 974-994): the whole down "
                    "curve (probe rates 3 and 4 and the encode-time Hessian) "
                    "is conditioned on ONE gate/up roundtrip decoded at the "
                    "reference rates; every conditioning receipt stamps the "
                    "rates actually used, never an approximation"
                ),
            },
        },
        "transform_seed_sha256": seed,
        "policy": policy,
        "scale_family": family,
        "codec_identity_sha256": codec_identity_sha256,
        "k4_codec_identity_sha256": k4_codec_identity_sha256,
        "codec_identity_binding": (
            "codec_identity_sha256 is checked identical across every rate-3 "
            "preparation manifest and against the live codec when one was "
            "constructed, and the phase-7 worker dies unless it reproduces "
            "this exact identity; k4_codec_identity_sha256 records the K4 "
            "half's venue separately (the original campaign venue differs "
            "by design)"
        ),
        "sigma_reg": common.SIGMA_REG,
        "allocations": allocations,
    }
    receipt = common.seal(body, "readiness_receipt_sha256")
    common.write_json(args.work_root / "gss" / "readiness-receipt.json", receipt)
    print(
        json.dumps(
            {
                "readiness_receipt_sha256": receipt["readiness_receipt_sha256"],
                "layers": len(per_layer),
                "next": (
                    "enter_k35_encoding(plan, state, readiness_receipt_sha256="
                    f"'{receipt['readiness_receipt_sha256']}')"
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
