"""Shared bindings for the GLM-5.3 mixed K3/K4 (3.5 bpw) sm120 encode drivers.

Scope: CODE ONLY.  This module binds the existing campaign library and mirrors
the minimum numeric paths that the sealed uniform-K4 surfaces hard-gate to
bits in (4, 6).  Every mirror cites the file and lines it reproduces.  Every
new receipt schema defined here is NEW SURFACE and is marked with a WARN
comment: the sealed campaign offers no builder or validator for it.

Environment contract (runbook section 8, adapted to the task work root):
  PYTHONPATH=/mnt/t5evo/glm53-k35-work/src:/mnt/t5evo/glm53-k35-work/reproducibility/r10
  cwd       =/mnt/t5evo/glm53-k35-work
Never import these drivers by /tmp path; import quant_pipeline.* and
r7_encoder.* through PYTHONPATH.

ASCII only.  No em-dashes.  No network.  No writes outside --work-root.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import re
import sys
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_pipeline.core.artifacts import (
    canonical_json,
    sha256_bytes,
    sha256_file,
    write_json,
)
from quant_pipeline.campaign import glm53_uniform_k35 as k35
from quant_pipeline.campaign.glm53_direct_k4 import (
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    MTP_LAYER,
    NUM_EXPERTS,
    PROJECTIONS,
)
from quant_pipeline.calibration.glm53_capture import MAIN_ROUTED_LAYERS

# ---------------------------------------------------------------------------
# Pinned constants
# ---------------------------------------------------------------------------

# The K4 campaign pins sigma_reg = 0.025 at codec construction
# (glm53_prepared_backend.py:146, glm53_mcg_preparation.py:302).  It equals
# the adapter default (codecs/exl3_mcg.py:45) and r7 DEFAULT_SIGMA_REG
# (r7_encoder/constants.py:42).  Reuse it; never parameterize it per run.
SIGMA_REG = 0.025

DEFAULT_WORK_ROOT = Path("/mnt/t5evo/glm53-k35-work")
DEFAULT_CALIBRATION_ROOT = Path(
    "/mnt/t9/glm53-archive/brandonmusic_GLM-5.3-Flash-BF16-Teacher-Logits/calibration"
)
DEFAULT_BF16_ROOT = Path("/mnt/t5evo/GLM-5.3-Flash-BF16")
ENV_EXTENSION = "K35_EXLLAMAV3_EXT"
CHUNK_ROWS = 1024

MAIN_LAYERS = tuple(MAIN_ROUTED_LAYERS)  # 3..44
ALL_PROBE_LAYERS = tuple(MAIN_LAYERS) + (MTP_LAYER,)  # 3..44, 45 = MTP

# ---------------------------------------------------------------------------
# NEW SURFACE schemas.
#
# WARN: none of the schemas below exist in the sealed campaign.  The grep for
# a readiness/preparation contract finds only the hash field
# "k35_readiness_receipt_sha256" (glm53_uniform_k35.py:856, 898) and the
# "preparation_contract" block (glm53_uniform_k35.py:599-605); there is no
# builder and no validator for any k35 execution-phase receipt.  Each schema
# defined here fails loud, seals over canonical JSON with the campaign's own
# canonical_json/sha256_bytes, and must be registered in the derivation
# report before these artifacts are treated as evidence.
# ---------------------------------------------------------------------------

K35_PROBE_LEDGER_SCHEMA = "quant-pipeline.glm53-k35-probe-ledger.v1"
K35_RATE3_GSS_SCHEMA = "quant-pipeline.glm53-k35-rate3-gss-preparation.v1"
K35_READINESS_SCHEMA = "quant-pipeline.glm53-k35-readiness-receipt.v1"
K35_PACKED_CHOICE_SCHEMA = "quant-pipeline.glm53-k35-packed-choice.v1"
K35_EXPERT_RECEIPT_SCHEMA = "quant-pipeline.glm53-k35-expert-receipt.v1"
K35_LAYER_RECEIPT_SCHEMA = "quant-pipeline.glm53-k35-layer-receipt.v1"

_HASH = re.compile(r"[0-9a-f]{64}")

DP_SCORE_SCALE = 10**15  # mirrors r7_encoder/allocation.py:25


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def die(message: str) -> None:
    raise SystemExit(f"k35: FAIL: {message}")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        die(f"{label} must be a lowercase 64-hex SHA-256, got {value!r}")
    return value


def seal(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    """Campaign sealing discipline (glm53_uniform_k35.py:177-180, verbatim)."""

    result = copy.deepcopy(dict(body))
    result[field] = sha256_bytes(canonical_json(result))
    return result


def verify_seal(document: Mapping[str, Any], *, schema: str, field: str, label: str) -> str:
    if document.get("schema") != schema:
        die(f"{label} schema differs: expected {schema}, got {document.get('schema')!r}")
    digest = require_hash(document.get(field), f"{label}.{field}")
    body = copy.deepcopy(dict(document))
    del body[field]
    if sha256_bytes(canonical_json(body)) != digest:
        die(f"{label} seal differs")
    return digest


def probe_stem(layer: int) -> str:
    return f"L{layer:02d}"


def layer_dir_name(layer: int) -> str:
    return f"layer-{layer:03d}"


def resolve_extension(args: argparse.Namespace) -> Path:
    raw = getattr(args, "extension", None) or os.environ.get(ENV_EXTENSION)
    if not raw:
        die(
            "the compiled exllamav3_ext .so is required: pass --extension PATH or set "
            f"{ENV_EXTENSION}; the codec hash-binds it at construction "
            "(codecs/exl3_mcg.py:52-54)"
        )
    path = Path(raw).resolve()
    if not path.is_file():
        die(f"extension is not a file: {path}")
    return path


def resolve_source_root(args: argparse.Namespace) -> Path:
    """Locate the r10 bundle (source_root) from --repo-root or sys.path."""

    raw = getattr(args, "repo_root", None)
    if raw:
        root = Path(raw).resolve()
        if not (root / "r7_encoder" / "r10_codec.py").is_file():
            die(f"--repo-root lacks r7_encoder/r10_codec.py: {root}")
        return root
    for entry in sys.path:
        candidate = Path(entry or ".").resolve()
        if (candidate / "r7_encoder" / "r10_codec.py").is_file():
            return candidate
    die("cannot find the r10 bundle on sys.path; pass --repo-root")


def numeric_core_path(source_root: Path) -> Path:
    path = source_root / "lineage" / "encode_tr3_v31.py"
    if not path.is_file():
        die(f"numeric core is absent: {path}")
    return path


# ---------------------------------------------------------------------------
# Codec construction (import-order critical)
# ---------------------------------------------------------------------------


def build_codec(source_root: Path, extension: Path, device: str):
    """Construct Exl3MCGCodec and force the sealed import NOW.

    Exl3MCGCodec._codec() refuses to run when any r7_encoder module is
    already cached (codecs/exl3_mcg.py:127-129).  Therefore every driver
    must construct the codec and trigger its sealed import BEFORE any code
    imports r7_encoder.* from PYTHONPATH.  This helper enforces the order
    for all three drivers.
    """

    from quant_pipeline.codecs.exl3_mcg import Exl3MCGCodec

    codec = Exl3MCGCodec(
        source_root=source_root,
        numeric_core=numeric_core_path(source_root),
        extension=extension,
        device=device,
        sigma_reg=SIGMA_REG,
    )
    codec._codec()  # sealed import of r7_encoder from source_root
    return codec


def r7_hessian():
    """Lazy r7_encoder.hessian access; only legal after build_codec()."""

    import r7_encoder.hessian as hessian

    return hessian


# ---------------------------------------------------------------------------
# Capture and source
# ---------------------------------------------------------------------------


def open_capture(calibration_root: Path, layer: int, *, verify_hashes: bool = True):
    if layer in MAIN_LAYERS:
        from quant_pipeline.campaign.glm53_direct_k4 import Glm53CaptureView

        return Glm53CaptureView(
            calibration_root / "main-ep4-full", layer, verify_hashes=verify_hashes
        )
    if layer == MTP_LAYER:
        from quant_pipeline.campaign.glm53_mtp_k4 import Glm53MTP45CaptureView

        return Glm53MTP45CaptureView(
            calibration_root / "mtp45-ep4-full", verify_hashes=verify_hashes
        )
    die(f"layer {layer} is outside the main routed surface 3..44 plus MTP {MTP_LAYER}")


def load_inventory(work_root: Path) -> dict[str, Any]:
    inventory = load_json(work_root / "inventory.json")
    if not isinstance(inventory, Mapping):
        die("inventory.json is not an object")
    require_hash(inventory.get("inventory_sha256"), "inventory_sha256")
    return dict(inventory)


def load_bf16_source(work_root: Path, bf16_root: Path, *, verify_shards: bool):
    from quant_pipeline.campaign.glm53_direct_k4 import Glm53BF16Source

    return Glm53BF16Source(
        load_inventory(work_root), bf16_root, verify_shards=verify_shards
    )


# ---------------------------------------------------------------------------
# Preparation shards (vectors + permutations)
# ---------------------------------------------------------------------------

_PREPARATION_REQUIRED_KEYS = {
    "permutations",
    "gate_suh",
    "gate_svh",
    "up_suh",
    "up_svh",
    "down_suh",
    "down_svh",
}
_PREPARATION_SHAPES = {
    "permutations": (NUM_EXPERTS, INTERMEDIATE_SIZE),
    "gate_suh": (NUM_EXPERTS, HIDDEN_SIZE),
    "gate_svh": (NUM_EXPERTS, INTERMEDIATE_SIZE),
    "up_suh": (NUM_EXPERTS, HIDDEN_SIZE),
    "up_svh": (NUM_EXPERTS, INTERMEDIATE_SIZE),
    "down_suh": (NUM_EXPERTS, INTERMEDIATE_SIZE),
    "down_svh": (NUM_EXPERTS, HIDDEN_SIZE),
}


def load_preparation(root: Path, layer: int, *, expected_bits: int | None = None):
    """Load and verify one preparation shard.

    Accepts the sealed K4/K6 campaign schema
    (quant-pipeline.glm53-public-shapleymcg-layer-preparation.v1, validated
    field-for-field like glm53_prepared_backend._load_preparation:189-243,
    including the semantic binding fields: policy, scale_family,
    profile_source, the selection/confirmation flags, source_closure_sqg_free,
    and shapleymcg_process_structure) or this campaign's rate-3 schema
    defined above (NEW SURFACE).
    Returns (manifest, tensors) with tensors on CPU.
    """

    from safetensors import safe_open

    directory = Path(root) / f"layer-{layer:03d}"
    manifest_path = directory / "preparation.json"
    if not manifest_path.is_file():
        die(f"preparation manifest is absent: {manifest_path}")
    manifest = load_json(manifest_path)
    schema = manifest.get("schema")
    if schema == "quant-pipeline.glm53-public-shapleymcg-layer-preparation.v1":
        from quant_pipeline.campaign.glm53_mcg_preparation import _PROCESS_STRUCTURE

        body = copy.deepcopy(dict(manifest))
        digest = body.pop("preparation_sha256", None)
        if (
            manifest.get("complete") is not True
            or manifest.get("layer") != layer
            or manifest.get("codec_family") != "exl3-mcg"
            or manifest.get("shard") is None
            or not isinstance(digest, str)
            or digest != sha256_bytes(canonical_json(body))
            or manifest.get("policy") != "energy_balanced"
            or manifest.get("scale_family") != "per128-grid"
            or manifest.get("profile_source")
            != "public-run-qwen-fast-encode-defaults"
            or manifest.get("profile_fixed_before_encoding") is not True
            or manifest.get("selection_rows_used") is not False
            or manifest.get("selection_used_for_profile_choice") is not False
            or manifest.get("selection_used_for_final_encoding") is not False
            or manifest.get("confirmation_used_for_choice") is not False
            or manifest.get("confirmation_report_only") is not True
            or manifest.get("global_allocator_invoked") is not False
            or manifest.get("candidate_rate_grid_invoked") is not False
            or manifest.get("source_closure_sqg_free") is not True
            or manifest.get("shapleymcg_process_structure") != _PROCESS_STRUCTURE
        ):
            die(f"sealed preparation binding differs: {manifest_path}")
        bits = int(manifest["bits"])
        if bits not in (4, 6):
            die(f"sealed preparation carries unsupported bits {bits}")
    elif schema == K35_RATE3_GSS_SCHEMA:
        verify_seal(
            manifest,
            schema=K35_RATE3_GSS_SCHEMA,
            field="preparation_sha256",
            label=f"rate-3 GSS layer {layer}",
        )
        if manifest.get("layer") != layer or manifest.get("complete") is not True:
            die(f"rate-3 GSS manifest layer binding differs: {manifest_path}")
        bits = int(manifest["bits"])
        if bits != 3:
            die(f"rate-3 GSS manifest carries bits {bits}")
    else:
        die(f"unknown preparation schema {schema!r} at {manifest_path}")
    if expected_bits is not None and bits != expected_bits:
        die(
            f"preparation bits {bits} differs from expected {expected_bits}: {manifest_path}"
        )
    shard = directory / str(manifest["shard"])
    if not shard.is_file() or shard.is_symlink():
        die(f"preparation shard is absent or a symlink: {shard}")
    if sha256_file(shard) != manifest.get("shard_sha256"):
        die(f"preparation shard hash differs: {shard}")
    with safe_open(shard, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if keys != _PREPARATION_REQUIRED_KEYS:
            die(f"preparation tensor census differs: {sorted(keys)}")
        tensors = {name: handle.get_tensor(name).contiguous() for name in sorted(keys)}
    import torch

    for name, shape in _PREPARATION_SHAPES.items():
        if tuple(tensors[name].shape) != shape:
            die(
                f"preparation tensor {name} shape {tuple(tensors[name].shape)} != {shape}"
            )
    if tensors["permutations"].dtype != torch.int64 or any(
        tensors[name].dtype != torch.float16
        for name in _PREPARATION_REQUIRED_KEYS - {"permutations"}
    ):
        die("preparation dtypes differ (need int64 permutations, fp16 vectors)")
    return manifest, tensors


def preparation_vectors(tensors: Mapping[str, Any], projection: str, expert: int, device: str):
    """Move one expert's suh/svh rows to the encode device (fp16, like the
    K4 backend passes CPU fp16 rows straight into the codec request,
    glm53_prepared_backend.py:479-480,513-514)."""

    prefix = projection.removesuffix("_proj")
    return (
        tensors[f"{prefix}_suh"][expert].to(device),
        tensors[f"{prefix}_svh"][expert].to(device),
    )


# ---------------------------------------------------------------------------
# Covariances (mirrors of glm53_prepared_backend.py:253-305, same arithmetic)
# ---------------------------------------------------------------------------


def _hidden_chunks(capture: Any, row_indices, device: str, chunk_rows: int):
    import numpy as np
    import torch

    for begin in range(0, row_indices.size, chunk_rows):
        stop = min(row_indices.size, begin + chunk_rows)
        words = np.array(
            capture.hidden_u16[row_indices[begin:stop]], dtype=np.uint16, copy=True
        )
        yield (
            torch.from_numpy(words)
            .view(torch.bfloat16)
            .to(device, dtype=torch.float32)
            .contiguous()
        )


def _routed_evidence(routed) -> dict[str, Any]:
    import hashlib

    import numpy as np

    return {
        "rows": int(routed.rows),
        "documents": int(np.unique(routed.document_epochs).size),
        "row_indices_sha256": hashlib.sha256(routed.row_indices.tobytes()).hexdigest(),
        "route_weights_sha256": hashlib.sha256(
            np.asarray(routed.applied_weights, dtype="<f4").tobytes()
        ).hexdigest(),
    }


def expert_p2_mass(capture: Any, expert: int) -> float:
    """Per-expert p2 mass over fit rows (the DP gain weight).

    Same statistic as _p2_profile_statistics masses
    (glm53_mcg_preparation.py:142-144): sum of squared applied router
    weights across the expert's fit rows.
    """

    import numpy as np

    routed = capture.routed_rows(expert, "fit")
    if routed.rows <= 0:
        die(f"L{capture.layer} E{expert}: fit rows are empty; mass is undefined")
    weights = np.asarray(routed.applied_weights, dtype=np.float64)
    mass = float(np.square(weights).sum())
    if mass <= 0:
        die(f"L{capture.layer} E{expert}: degenerate p2 mass")
    return mass


def gate_covariance(codec, capture: Any, expert: int, device: str, chunk_rows: int):
    """Routed p2 uncentered full covariance over 'fit' rows.

    Mirror of glm53_prepared_backend._gate_covariance (lines 253-276): guided
    accumulator, squared applied router weights as row weights, finalize with
    the pinned sigma_reg and add_damping=False.
    """

    import numpy as np

    hessian = r7_hessian()
    routed = capture.routed_rows(expert, "fit")
    if routed.rows <= 0:
        die(f"L{capture.layer} E{expert}: empty fit rows")
    accumulator = hessian.FullCovarianceAccumulator(
        HIDDEN_SIZE, device=device, guided=True
    )
    cursor = 0
    for hidden in _hidden_chunks(capture, routed.row_indices, device, chunk_rows):
        stop = cursor + int(hidden.shape[0])
        weights = np.square(routed.applied_weights[cursor:stop], dtype=np.float32)
        accumulator.add(hidden, weights)
        cursor = stop
    value = accumulator.finalize(SIGMA_REG, add_damping=False)
    evidence = dict(_routed_evidence(routed))
    evidence.update(
        {
            "construction": "routed-p2-uncentered-second-moment-v1",
            "weight_sum": float(value.weight_sum),
        }
    )
    return value.matrix, evidence


def down_covariance(
    codec,
    capture: Any,
    expert: int,
    gate_kn: Any,
    up_kn: Any,
    *,
    gate_bits: int,
    up_bits: int,
    device: str,
    chunk_rows: int,
):
    """Candidate-conditioned down covariance over 'conditional-fit' rows.

    Mirror of glm53_prepared_backend._down_covariance (lines 278-305) with the
    conditioning made explicit and receipt-honest: ``gate_kn``/``up_kn`` are
    gate/up reconstructions decoded at exactly ``gate_bits``/``up_bits``, and
    the evidence stamps those rates both in the construction string and as
    numeric fields.  The k35 drivers condition the WHOLE down curve (probe
    rates 3 and 4 and the encode-time Hessian) on ONE reference context,
    gate/up decoded at k35.FLOOR_BITS, which is the R7 pair_at semantics
    (r7_encoder/layer.py:901-925 memoizes one context; layer.py:974-994 runs
    every down bit width under pair_at(base_gate_bits, base_up_bits)), so the
    DP gain mass*(loss3-loss4) subtracts same-denominator ratios and the
    encode-time Hessian equals the probe measurement's conditioning.
    """

    import numpy as np

    hessian = r7_hessian()
    routed = capture.routed_rows(expert, "conditional-fit")
    if routed.rows <= 0:
        die(f"L{capture.layer} E{expert}: empty conditional-fit rows")
    accumulator = hessian.FullCovarianceAccumulator(
        INTERMEDIATE_SIZE, device=device, guided=True
    )
    gate_rt = gate_kn.to(device)
    up_rt = up_kn.to(device)
    cursor = 0
    for hidden in _hidden_chunks(capture, routed.row_indices, device, chunk_rows):
        stop = cursor + int(hidden.shape[0])
        middle = hessian.down_inputs_from_roundtrip(hidden, gate_rt, up_rt)
        weights = np.square(routed.applied_weights[cursor:stop], dtype=np.float32)
        accumulator.add(middle, weights)
        cursor = stop
    value = accumulator.finalize(SIGMA_REG, add_damping=False)
    evidence = dict(_routed_evidence(routed))
    evidence.update(
        {
            "construction": (
                f"decoded-gate-k{int(gate_bits)}-up-k{int(up_bits)}-"
                "candidate-conditioned-routed-p2-uncentered-second-moment-v1"
            ),
            "conditioning_gate_bits": int(gate_bits),
            "conditioning_up_bits": int(up_bits),
            "weight_sum": float(value.weight_sum),
        }
    )
    return value.matrix, evidence


def tensor_sha256(value: Any) -> str:
    """Dtype+shape+bytes digest, verbatim glm53_prepared_backend.py:48-58."""

    import hashlib

    import torch

    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(map(str, tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def raw_payload_sha256(value: Any) -> str:
    """Raw-bytes digest, verbatim glm53_prepared_backend.py:61-65."""

    import hashlib

    import torch

    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def covariance_proxy_loss(weight_hf: Any, reconstructed_hf: Any, covariance: Any) -> float:
    """Per-rate probe loss: the relative covariance quadratic.

    This is the loss function of the sealed K4 numeric closure, computed by
    the caller because the R10 fast path deliberately returns proxy_loss=0.0
    and marks the encode-time audit not-run (r7_encoder/r10_codec.py:512 and
    :431).  The formula is verbatim Exl3TrellisCodec.encode
    (r7_encoder/trellis.py:383-396), evaluated in float64 on the
    reconstruction retained by the R10 candidate:

        error = reconstructed - weight
        loss  = (error^T C error) / (weight^T C weight),
                denominator clamped at 1e-30

    trellis.py evaluates it in [K,N] orientation with einsum "kn,kl,ln->";
    Exl3MCGCodec.encode_candidates hands back reconstructed in HF [N,K]
    orientation (codecs/exl3_mcg.py:197), so the transposed einsum
    "nk,kl,nl->" evaluates the identical sum over the same index set.

    Bridge (README mismatch 3): R10 documents byte-compatibility of the
    packed and reconstructed artifacts with the R7 audited path
    (r10_codec.py module docstring), so this recomputation over the R10
    candidate is the K4 loss of the same bytes.  Optional cross-check: encode
    one tensor through the R7 audited path (r7_encoder.trellis
    Exl3TrellisCodec.encode) and compare proxy_loss directly.
    """

    import torch

    weight = torch.as_tensor(weight_hf)
    recon = torch.as_tensor(reconstructed_hf)
    if tuple(weight.shape) != tuple(recon.shape):
        die(
            f"proxy-loss orientation differs: weight {tuple(weight.shape)} "
            f"reconstruction {tuple(recon.shape)}"
        )
    covariance_tensor = torch.as_tensor(
        covariance, dtype=torch.float64, device=weight.device
    )
    error = recon.double() - weight.double()
    numerator = torch.einsum("nk,kl,nl->", error, covariance_tensor, error)
    denominator = torch.einsum(
        "nk,kl,nl->", weight.double(), covariance_tensor, weight.double()
    ).clamp_min(1e-30)
    value = float((numerator / denominator).item())
    if value < 0:
        die("proxy loss is negative; covariance or reconstruction is malformed")
    return value


def encode_one_rate(
    codec,
    *,
    layer: int,
    expert: int,
    projection: str,
    weight_hf: Any,
    covariance: Any,
    bits: int,
    suh: Any,
    svh: Any,
    provenance: dict[str, Any] | None = None,
):
    """encode_candidates narrowed to one integer rate (worker path).

    The probe driver calls encode_candidates(bits=(3, 4)) directly per its
    task contract; this helper serves the worker, which encodes only the
    allocated rate.
    """

    if int(bits) not in k35.PER_TENSOR_ALLOWED_BITS:
        die(f"encode rate {bits} outside {k35.PER_TENSOR_ALLOWED_BITS}")
    return codec.encode_candidates(
        unit_id=f"L{layer}.E{expert}.{projection}",
        weight_hf=weight_hf,
        covariance=covariance,
        bits=(int(bits),),
        input_vector=suh,
        output_vector=svh,
        provenance=provenance,
    )[int(bits)]


# ---------------------------------------------------------------------------
# NEW SURFACE: bits-honest packed choice store.
#
# WARN: the sealed PackedMCGPayloadStore hardcodes "bits": 4 in the choice
# body (checkpoint/packed_payload.py, put_choice body key "bits") and its
# verify_choice rejects bits != 4.  A K3 choice stored through it would carry
# a false bits=4 field.  This store mirrors the layout exactly (objects/ plus
# choices/ directories under the store root, same per-object hashing, same
# packed and checkpoint hash framing) but writes an honest per-choice bits
# field under a k35 schema and checks the trellis byte count against bits.
# Downstream materializers must be taught this schema; they cannot read
# these choices through the K4 verifier.
# ---------------------------------------------------------------------------


class K35PackedPayloadStore:
    def __init__(self, root: str | Path) -> None:
        from quant_pipeline.checkpoint.exact_payload import ExactCodecPayloadStore

        self.root = Path(root)
        self.objects = ExactCodecPayloadStore(self.root)
        self.choices = self.root / "choices"
        self.choices.mkdir(parents=True, exist_ok=True)

    def put_choice(
        self,
        *,
        layer: int,
        expert: int,
        projection: str,
        bits: int,
        choice_id: str,
        trellis: Any,
        suh: Any,
        svh: Any,
        mcg: Any,
        reconstruction: Any,
        vector_topology: Mapping[str, str],
        reader_abi_sha256: str,
        provenance: Mapping[str, Any],
        predecessor_state_hash: str,
    ) -> dict[str, Any]:
        import torch

        from quant_pipeline.checkpoint.exact_payload import (
            packed_payload_sha256,
            tensor_sha256,
        )
        from quant_pipeline.checkpoint.packed_payload import checkpoint_payload_sha256

        if projection not in PROJECTIONS:
            die(f"unknown projection {projection}")
        if bits not in k35.PER_TENSOR_ALLOWED_BITS:
            die(f"choice bits {bits} outside {k35.PER_TENSOR_ALLOWED_BITS}")
        require_hash(predecessor_state_hash, "choice predecessor state")
        require_hash(reader_abi_sha256, "reader ABI")
        values = {}
        for name, value in (
            ("trellis", trellis),
            ("suh", suh),
            ("svh", svh),
            ("mcg", mcg),
            ("reconstruction", reconstruction),
        ):
            tensor = torch.as_tensor(value).detach().contiguous().cpu()
            values[name] = tensor.reshape(1) if tensor.ndim == 0 else tensor
        if values["trellis"].dtype != torch.int16:
            die("trellis must be int16")
        if any(
            values[name].dtype != torch.float16 for name in ("suh", "svh", "reconstruction")
        ):
            die("scales and closure reconstruction must be FP16")
        if (
            values["mcg"].dtype != torch.int32
            or values["mcg"].numel() != 1
            or int(values["mcg"].reshape(-1)[0]) != -877912083
        ):
            die("choice is not marked with MCG 0xCBAC1FED")
        reconstruction = values["reconstruction"]
        if reconstruction.ndim != 2 or values["suh"].ndim != 1 or values["svh"].ndim != 1:
            die("choice tensor ranks differ")
        n, k = map(int, reconstruction.shape)
        if values["suh"].numel() != k or values["svh"].numel() != n:
            die("vectors disagree with reconstruction geometry")
        expected_trellis_bytes = n * k * int(bits) // 8
        actual_trellis_bytes = values["trellis"].numel() * values["trellis"].element_size()
        if actual_trellis_bytes != expected_trellis_bytes:
            die(
                f"trellis bytes {actual_trellis_bytes} disagree with bits {bits} "
                f"geometry {n}x{k} (expected {expected_trellis_bytes})"
            )
        stored = {name: values[name] for name in ("trellis", "suh", "svh", "mcg")}
        refs = {
            name: self.objects.put_tensor(value).as_dict() for name, value in stored.items()
        }
        body = {
            "schema": K35_PACKED_CHOICE_SCHEMA,
            "layer": int(layer),
            "expert": int(expert),
            "projection": projection,
            "choice_id": str(choice_id),
            "bits": int(bits),
            "predecessor_state_hash": str(predecessor_state_hash),
            "objects": refs,
            "packed_sha256": packed_payload_sha256(
                {name: stored[name] for name in ("trellis", "suh", "svh")}
            ),
            "checkpoint_payload_sha256": checkpoint_payload_sha256(stored),
            "logical_payload_bytes": sum(int(ref["bytes"]) for ref in refs.values()),
            "param_count": n * k,
            "vector_topology": dict(vector_topology),
            "reconstruction_closure": {
                "schema": "quant-pipeline.exl3-mcg-fp16-closure.v1",
                "dtype": "float16",
                "shape": [n, k],
                "orientation": "huggingface_out_in",
                "payload_sha256": tensor_sha256(reconstruction),
                "persisted": False,
                "encoder_full_decode_closure": True,
            },
            "decoder": {
                "codec_family": "exl3-mcg",
                "mcg_multiplier_hex": "0xCBAC1FED",
                "mcg_marker_signed_int32": -877912083,
                "reader_abi_sha256": str(reader_abi_sha256),
            },
            "provenance": copy.deepcopy(dict(provenance)),
        }
        body["choice_sha256"] = sha256_bytes(canonical_json(body))
        path = self.choices / f"{body['choice_sha256']}.json"
        if path.exists():
            if load_json(path) != body:
                die("EXL3/MCG choice hash collision")
        else:
            write_json(path, body)
        return body

    def verify_choice(self, choice: str | Path | Mapping[str, Any]) -> dict[str, Any]:
        import torch

        row = (
            load_json(choice)
            if isinstance(choice, (str, Path))
            else copy.deepcopy(dict(choice))
        )
        expected = row.get("choice_sha256")
        unsigned = {key: value for key, value in row.items() if key != "choice_sha256"}
        if (
            row.get("schema") != K35_PACKED_CHOICE_SCHEMA
            or not isinstance(expected, str)
            or _HASH.fullmatch(expected) is None
            or sha256_bytes(canonical_json(unsigned)) != expected
            or row.get("bits") not in k35.PER_TENSOR_ALLOWED_BITS
            or row.get("projection") not in PROJECTIONS
        ):
            die("k35 packed-choice seal differs")
        objects = row.get("objects")
        if not isinstance(objects, Mapping) or set(objects) != {"trellis", "suh", "svh", "mcg"}:
            die("k35 packed-choice object census differs")
        values = {name: self.objects.load_tensor(ref) for name, ref in objects.items()}
        if values["trellis"].dtype != torch.int16:
            die("k35 packed-choice trellis dtype differs")
        return row


# ---------------------------------------------------------------------------
# NEW SURFACE: sensitivity DP solver for GLM-5.3 geometry.
#
# WARN: r7_encoder.allocation.solve_exact_allocation cannot serve this
# geometry.  Its constants module is owner-locked to GLM-5.2
# (r7_encoder/constants.py:15 NUM_EXPERTS=256, :35-40
# TENSORS_PER_LAYER=768, TARGET_BIT_UNITS_PER_LAYER=2688,
# UPGRADE_UNITS_PER_LAYER=384) and TensorId.__post_init__ rejects layers
# above 77 and experts at or above 256 (constants.py:56-62), so even
# constructing GLM-5.3 tensor ids fails.  This port keeps the algorithm
# byte-faithful: integer scores scaled by 10^15 with ROUND_HALF_EVEN
# (allocation.py:25, :114-118), the low-bits-first plus
# replace-only-on-strict-gain tie rule (allocation.py:144-157), the exact
# upgrade budget as the DP dimension (allocation.py:134-135), and
# backpointer reconstruction with the zero-budget return check
# (allocation.py:165-179).  The result is audited by the sealed campaign
# authority k35.audit_layer_allocation (glm53_uniform_k35.py:348-360),
# which enforces the 864-name census, integer bits in (3,4), the exact 3024
# sum, and the 432/432 split.
# ---------------------------------------------------------------------------


def _gain_integer(mass: Decimal, loss_floor: Decimal, loss_upgrade: Decimal) -> int:
    with localcontext() as context:
        context.prec = 50
        scaled = mass * (loss_floor - loss_upgrade) * Decimal(DP_SCORE_SCALE)
        return int(scaled.to_integral_value(rounding=ROUND_HALF_EVEN))


def solve_layer_dp(
    layer: int,
    loss_by_bits: Mapping[str, tuple[Decimal, Decimal]],
    mass_by_expert: Sequence[Decimal],
) -> dict[str, int]:
    """Exact 3024-bit-unit allocation with exactly 432 K4 tensors.

    ``loss_by_bits`` maps the full HF tensor name to (loss at 3, loss at 4)
    as Decimals parsed from their .17g probe strings.
    ``mass_by_expert`` is the per-expert p2 mass over fit rows.
    Tensor order is k35._layer_tensor_names (sorted), the same census the
    sealed audit enforces.
    """

    ordered_names = k35._layer_tensor_names(layer)
    if len(ordered_names) != k35.TENSORS_PER_LAYER:
        die("layer tensor census differs from 864")
    if set(loss_by_bits) != set(ordered_names):
        missing = sorted(set(ordered_names) - set(loss_by_bits))
        extra = sorted(set(loss_by_bits) - set(ordered_names))
        die(
            f"probe loss census differs: {len(missing)} missing "
            f"(first {missing[0] if missing else None}), {len(extra)} extra"
        )
    if len(mass_by_expert) != NUM_EXPERTS:
        die(f"expected {NUM_EXPERTS} expert masses, got {len(mass_by_expert)}")
    curves = []
    for name in ordered_names:
        loss3, loss4 = loss_by_bits[name]
        if loss3 < 0 or loss4 < 0:
            die(f"negative probe loss for {name}")
        expert = int(name.split(".experts.")[1].split(".")[0])
        curves.append((name, mass_by_expert[expert], loss3, loss4))

    budget = k35.K4_TENSORS_PER_LAYER  # 432 upgrades; each upgrade is 1 unit
    negative_infinity = None
    scores: list[int | None] = [0] + [negative_infinity] * budget
    parent_costs: list[list[int]] = []
    parent_bits: list[list[int]] = []

    for _name, mass, loss3, loss4 in curves:
        next_scores: list[int | None] = [negative_infinity] * (budget + 1)
        costs = [-1] * (budget + 1)
        choices = [0] * (budget + 1)
        gains = {
            3: _gain_integer(mass, loss3, loss3),
            4: _gain_integer(mass, loss3, loss4),
        }
        for prior_cost, prior_score in enumerate(scores):
            if prior_score is None:
                continue
            for bits in k35.PER_TENSOR_ALLOWED_BITS:  # low bits first (3, 4)
                extra_units = bits - k35.FLOOR_BITS
                new_cost = prior_cost + extra_units
                if new_cost > budget:
                    continue
                candidate_score = prior_score + gains[bits]
                incumbent = next_scores[new_cost]
                if incumbent is None or candidate_score > incumbent:
                    next_scores[new_cost] = candidate_score
                    costs[new_cost] = prior_cost
                    choices[new_cost] = bits
        scores = next_scores
        parent_costs.append(costs)
        parent_bits.append(choices)

    final_score = scores[budget]
    if final_score is None:
        die(f"layer {layer}: exact {k35.TARGET_BIT_UNITS_PER_LAYER}-bit-unit budget is unreachable")
    selected_reversed: list[int] = []
    cost = budget
    for item_index in range(len(curves) - 1, -1, -1):
        bits = int(parent_bits[item_index][cost])
        prior = int(parent_costs[item_index][cost])
        if bits not in k35.PER_TENSOR_ALLOWED_BITS or prior < 0:
            die(f"layer {layer}: DP backpointer corruption")
        selected_reversed.append(bits)
        cost = prior
    if cost != 0:
        die(f"layer {layer}: allocation did not return to zero budget")
    selected = list(reversed(selected_reversed))
    result = {curves[index][0]: selected[index] for index in range(len(curves))}
    k35.audit_layer_allocation(layer, result)
    return result


# ---------------------------------------------------------------------------
# State chain helpers
# ---------------------------------------------------------------------------


def state_dir(work_root: Path) -> Path:
    return Path(work_root) / "state"


def state_path(work_root: Path, sequence: int) -> Path:
    return state_dir(work_root) / f"state-{int(sequence):04d}.json"


def newest_state(work_root: Path, plan: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Newest state receipt bound to THIS launch plan.

    State files whose launch_plan_sha256 differs from the passed plan are
    ignored: verify_state raises "state receipt targets a different launch
    plan" on them (glm53_uniform_k35.py:781-782), so a superseded-plan file
    at the max sequence would wedge every worker and --recover-worker start.
    Superseded states belong in state/history/ (README phase 5b); files left
    in place are filtered out here instead of poisoning the selection.
    """

    directory = state_dir(work_root)
    plan_sha = plan["launch_plan_sha256"]
    paths = sorted(directory.glob("state-*.json"))
    if not paths:
        die(f"no state receipts under {directory}; run the phase-4 plan bootstrap first")
    parsed: list[tuple[int, Path]] = []
    states: dict[int, dict[str, Any]] = {}
    for path in paths:
        match = re.fullmatch(r"state-(\d+)\.json", path.name)
        if match is None:
            die(f"foreign file in the state directory: {path}")
        state = load_json(path)
        if not isinstance(state, Mapping):
            die(f"state file is not an object: {path}")
        if state.get("launch_plan_sha256") != plan_sha:
            continue
        sequence = int(match.group(1))
        if state.get("sequence") != sequence:
            die(f"state file sequence stamp differs: {path}")
        parsed.append((sequence, path))
        states[sequence] = dict(state)
    if not parsed:
        die(
            f"no state bound to the current plan {plan_sha[:16]} under "
            f"{directory}; run the phase-5b bootstrap (rebuild plan.json, "
            "rewrite state-0000.json, and move superseded state files into "
            "state/history/)"
        )
    parsed.sort()
    top = [item for item in parsed if item[0] == parsed[-1][0]]
    if len(top) != 1:
        die(f"multiple newest state receipts: {[str(item[1]) for item in top]}")
    sequence, path = top[0]
    return path, states[sequence]


class StateLock:
    """Cross-process lock serializing claim/complete transitions.

    The state machine is the lock (runbook phase 7): claim_next_layer refuses
    a worker that already owns a layer, and every successor is sealed.  With
    file-based state, the read-claim-write critical section must also be
    serialized between the two worker processes, hence this flock.
    """

    def __init__(self, work_root: Path) -> None:
        directory = state_dir(work_root)
        directory.mkdir(parents=True, exist_ok=True)
        self.handle = (directory / ".lock").open("a")

    def __enter__(self) -> "StateLock":
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *_exc: Any) -> None:
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def verify_plan_worker(plan: Mapping[str, Any], worker_id: str) -> dict[str, Any]:
    rows = plan.get("scheduler", {}).get("workers", [])
    for row in rows:
        if row.get("worker_id") == worker_id:
            return dict(row)
    die(
        f"worker id {worker_id!r} is not in the plan; accepted ids for this plan: "
        f"{[row.get('worker_id') for row in rows]}.  The sm120-declared venue assigns "
        'worker ids f"sm120-{{slot}}" from the preflight gpus enumeration '
        "(glm53_uniform_k35.py:293); the sealed four-B200 venue assigns "
        'f"b200-{{slot}}" (glm53_uniform_k4.py:211).  Note claim_next_layer error '
        'text says "unknown B200 worker" on every venue (glm53_uniform_k35.py:911).'
    )
    raise AssertionError("unreachable")


def load_plan(work_root: Path) -> dict[str, Any]:
    plan = load_json(work_root / "plan.json")
    k35.verify_launch_plan(plan)
    return plan


def load_layer_allocation(work_root: Path, layer: int) -> dict[str, Any]:
    path = work_root / "allocations" / f"{probe_stem(layer)}.json"
    if not path.is_file():
        die(f"sealed layer allocation is absent: {path} (run the phase-5 probe driver)")
    receipt = load_json(path)
    k35.verify_layer_allocation(receipt, layer=layer)
    if receipt.get("provisional") is not False:
        die(f"layer {layer} allocation is still provisional: {path}")
    return receipt


# ---------------------------------------------------------------------------
# NEW SURFACE: receipts
# ---------------------------------------------------------------------------


def build_expert_receipt(
    *,
    layer: int,
    expert: int,
    bits_by_projection: Mapping[str, int],
    choices: Mapping[str, Mapping[str, Any]],
    claim_receipt_sha256: str,
    allocation_sha256: str,
    capture_binding: Mapping[str, Any],
    hessian_artifact: Mapping[str, Any],
    down_conditioning: Mapping[str, Any],
    codec_identity_sha256: str,
) -> dict[str, Any]:
    """NEW SURFACE expert receipt (WARN: no sealed validator exists).

    Mirrors the K4 EXPERT_RECEIPT field structure
    (glm53_direct_k4.py:971-987) with the k35 per-projection rate binding.
    """

    if sorted(bits_by_projection) != sorted(PROJECTIONS):
        die("expert receipt does not close its triplet")
    body = {
        "schema": K35_EXPERT_RECEIPT_SCHEMA,
        "claim_receipt_sha256": require_hash(claim_receipt_sha256, "claim receipt"),
        "allocation_sha256": require_hash(allocation_sha256, "allocation receipt"),
        "layer": int(layer),
        "expert": int(expert),
        "projections": list(PROJECTIONS),
        "bits": dict(sorted(bits_by_projection.items())),
        "bit_units": int(sum(bits_by_projection.values())),
        "rate": {"numerator": k35.RATE_NUMERATOR, "denominator": k35.RATE_DENOMINATOR},
        "capture_binding": copy.deepcopy(dict(capture_binding)),
        "codec_identity_sha256": require_hash(codec_identity_sha256, "codec identity"),
        "sigma_reg": SIGMA_REG,
        "down_conditioning": copy.deepcopy(dict(down_conditioning)),
        "hessian_artifact": copy.deepcopy(dict(hessian_artifact)),
        "choices": copy.deepcopy({p: dict(choices[p]) for p in PROJECTIONS}),
    }
    return seal(body, "receipt_sha256")


def verify_expert_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    verify_seal(
        receipt,
        schema=K35_EXPERT_RECEIPT_SCHEMA,
        field="receipt_sha256",
        label=f"k35 expert receipt L{receipt.get('layer')}E{receipt.get('expert')}",
    )
    if sorted(receipt.get("choices", {})) != sorted(PROJECTIONS):
        die("k35 expert receipt choice census differs")
    bits = receipt.get("bits", {})
    if sorted(bits) != sorted(PROJECTIONS) or any(
        bits.get(p) not in k35.PER_TENSOR_ALLOWED_BITS for p in PROJECTIONS
    ):
        die("k35 expert receipt bits binding differs")
    conditioning = receipt.get("down_conditioning")
    if not isinstance(conditioning, Mapping):
        die("k35 expert receipt lacks its down-conditioning block")
    evidence = conditioning.get("evidence")
    if not isinstance(evidence, Mapping):
        die("k35 expert receipt down-conditioning evidence is absent")
    if (
        evidence.get("conditioning_gate_bits") != conditioning.get("gate_rate")
        or evidence.get("conditioning_up_bits") != conditioning.get("up_rate")
    ):
        die(
            "k35 expert receipt down-conditioning stamp disagrees with its "
            "recorded gate/up conditioning rates"
        )
    return dict(receipt)


def build_layer_receipt(
    *,
    layer: int,
    worker_id: str,
    claim_receipt_sha256: str,
    allocation_sha256: str,
    expert_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """NEW SURFACE layer receipt (WARN: no sealed validator exists).

    Mirrors glm53_direct_k4.seal_layer (glm53_direct_k4.py:1032-1045) plus
    the k35 rate census.  complete_layer binds only the seal hash
    (glm53_uniform_k35.py:949-977).
    """

    if len(expert_receipts) != NUM_EXPERTS:
        die(f"layer receipt census differs: {len(expert_receipts)} experts")
    expert_shas: list[str] = []
    choice_shas: list[str] = []
    k4_tensors = 0
    bit_units = 0
    for receipt in expert_receipts:
        verify_expert_receipt(receipt)
        if receipt["layer"] != layer:
            die("layer receipt expert targets a foreign layer")
        if receipt["claim_receipt_sha256"] != claim_receipt_sha256:
            die("layer receipt expert binds a foreign claim")
        if receipt["allocation_sha256"] != allocation_sha256:
            die("layer receipt expert binds a foreign allocation")
        expert_shas.append(receipt["receipt_sha256"])
        for projection in PROJECTIONS:
            choice = receipt["choices"][projection]
            require_hash(choice.get("choice_sha256"), "choice seal")
            bits = receipt["bits"][projection]
            k4_tensors += 1 if bits == 4 else 0
            bit_units += int(bits)
            choice_shas.append(choice["choice_sha256"])
    if k4_tensors != k35.K4_TENSORS_PER_LAYER or bit_units != k35.TARGET_BIT_UNITS_PER_LAYER:
        die(
            f"layer receipt rate census differs: k4 {k4_tensors} "
            f"(need {k35.K4_TENSORS_PER_LAYER}), bit units {bit_units} "
            f"(need {k35.TARGET_BIT_UNITS_PER_LAYER})"
        )
    body = {
        "schema": K35_LAYER_RECEIPT_SCHEMA,
        "layer": int(layer),
        "worker_id": str(worker_id),
        "claim_receipt_sha256": require_hash(claim_receipt_sha256, "claim receipt"),
        "allocation_sha256": require_hash(allocation_sha256, "allocation receipt"),
        "experts": NUM_EXPERTS,
        "matrix_count": k35.TENSORS_PER_LAYER,
        "bits": "mixed_k34_per_tensor",
        "bit_units": k35.TARGET_BIT_UNITS_PER_LAYER,
        "k4_tensor_count": k35.K4_TENSORS_PER_LAYER,
        "k3_tensor_count": k35.K3_TENSORS_PER_LAYER,
        "rate": {"numerator": k35.RATE_NUMERATOR, "denominator": k35.RATE_DENOMINATOR},
        "expert_receipt_sha256": expert_shas,
        "choice_sha256": choice_shas,
        "complete": True,
    }
    return seal(body, "receipt_sha256")


def decimal_from_float(value: float, label: str) -> Decimal:
    text = format(float(value), ".17g")
    parsed = Decimal(text)
    if not parsed.is_finite():
        die(f"{label} is not finite: {value!r}")
    return parsed


# ---------------------------------------------------------------------------
# Shared argument fragments
# ---------------------------------------------------------------------------


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT), type=Path)
    parser.add_argument("--device", default=None, help="default: cuda:0")
    parser.add_argument(
        "--extension",
        default=None,
        help=f"path to the compiled exllamav3_ext .so (or env {ENV_EXTENSION})",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="r10 bundle root containing r7_encoder/ (default: discovered on sys.path)",
    )
    parser.add_argument("--calibration-root", default=str(DEFAULT_CALIBRATION_ROOT), type=Path)
    parser.add_argument("--bf16-root", default=str(DEFAULT_BF16_ROOT), type=Path)
    parser.add_argument("--chunk-rows", default=CHUNK_ROWS, type=int)
    parser.add_argument(
        "--verify-shards",
        action="store_true",
        help="re-hash every BF16 shard against the sealed inventory before starting",
    )
    parser.add_argument(
        "--no-verify-capture-hashes",
        action="store_true",
        help="skip per-layer capture payload hashing (manifest seal still verified)",
    )


def finish_common_args(args: argparse.Namespace) -> tuple[Path, str]:
    args.work_root = Path(args.work_root).resolve()
    args.calibration_root = Path(args.calibration_root).resolve()
    args.bf16_root = Path(args.bf16_root).resolve()
    if args.device is None:
        args.device = "cuda:0"
    return args.work_root, args.device
