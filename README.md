# GLM-5.3-Flash EXL3 3.5bpw Mixed (K3/K4) - SM120 TP2 Recipe

First sub-4bpw serving of GLM-5.3-Flash on 2x RTX PRO 6000 (SM120, 96GB each):
**1M context + MTP3 speculative decoding + CUDA graphs + thinking mode, simultaneously.**

At 4bpw this combination does not fit: the drafter KV plus a 1M-token pool exceeds
192GB. The mixed-rate 3.5bpw checkpoint frees 9.07 GiB per GPU, which is exactly
the gap. The result is a single serve that does deep context AND fast decode,
instead of separate read-lane and fast-lane serves.

## Measured

All numbers from the production serve on 2x RTX PRO 6000, TP2, thinking mode ON.

| Metric | Value |
|---|---|
| Decode, engine-side, MTP3, thinking on | 126-150 tok/s |
| Decode, real agentic client session (Spock, LAN) | 51-109 tok/s client-side |
| MTP3 acceptance length | ~2.4 mean |
| KV pool | 1,985,915 tokens |
| max_model_len | 1,000,000 (1.99x concurrency at full 1M) |
| CUDA graphs | PIECEWISE x4 + FULL, captured |
| Cold boot to ready | ~13 min (shards + b12x compile + graph capture) |
| Checkpoint size | 145.4 GiB, 120 shards, 150,226 tensors |

Reference points on the same hardware: the 4bpw DFlash2 cell peaks at 78-91 tok/s
with 192K-230K context. This serve beats that decode rate at 5x the context.

Evidence: [docs/evidence/](docs/evidence/). Formal KLD and eval-suite numbers are
being added (see Status).

## The checkpoint

Mixed-rate per-tensor EXL3/MCG: every routed-expert tensor independently assigned
K3 or K4 (18,576 K3 + 18,576 K4 choices = 37,152 routed decisions), allocated by
energy-balanced dynamic programming over calibration activations. Exact average
7/2 bpw. Derived from zai-org GLM-5.3-Flash BF16 using brandonmusic's published
R10 encoder closure (hash-verified) and teacher-logits calibration set. Every
layer carries claim/packed/reconstruction provenance receipts; the pack plan and
all 120 shard receipts are SHA-sealed (see encode/ drivers).

Config-side, the artifact declares the mixed schema vLLM's EXL3 path consumes:

```json
"quantization_config": {
  "r7_routed_experts": {
    "schema": "r7-complete-v2-checkpoint-v1",
    "codebook": "mcg",
    "bits": "mixed_tensor",
    "moe_layers": [3, 45],
    "k_values": [3, 4]
  }
}
```

Per-tensor widths ride inside the trellis payload shapes; no kernel changes are
needed to read them. IMPORTANT: vLLM reads the quantization_config EMBEDDED in
config.json, not the standalone quantization_config.json. Details in
[docs/POST-PACK-CONFIG-SURGERY.md](docs/POST-PACK-CONFIG-SURGERY.md).

## Serve recipe

Launcher: [serve/serve-k35.sh](serve/serve-k35.sh) (bind-mounts the patched
quant layer; container name derives from CACHE/DCP env values).

```bash
MODEL=/path/to/artifact-glm53-k35-mixed \
IMAGE=glm53-exl3:dflash2-mixed \
CACHE=nvfp4_ds_mla DCP=1 PROFILE=long500k \
MAX_MODEL_LEN=1000000 \
VLLM_EXL3_PREFILL_BLOCK_M=64 \
GLM53_CACHE_PATH=$HOME/.cache/glm53-exl3-k4 \
PORT=8012 ./serve-k35.sh
```

- `VLLM_EXL3_PREFILL_BLOCK_M` MUST be 64 or 32. The image default 128 fails the
  FC2 route-subtile check (allowed routed sizes: 8, 16, 32, 48, 64).
- Tool calling: launcher passes `--enable-auto-tool-choice --tool-call-parser
  glm45` (the GLM family parser in this vLLM build). Verified end-to-end with
  OpenAI-format tools.
- Served name is env-overridable: `SERVED_NAME` (default
  `GLM-5.3-Flash-EXL3-3.5bpw`).

## Image build

`glm53-exl3:dflash2-mixed` = glm53-exl3:dflash2 (verdictai v75p2 lineage + the
tonyd2wild DFlash2 overlay, three calling-convention fixes applied) + four b12x
files taken from local-inference-lab/b12x master (commit 139e040) which carry
the trellis3 mixed-rate API the stock 1.2.6 release lacks:

- `b12x/moe/_shared/kernels/w4a16/host.py`
- `b12x/moe/_shared/kernels/w4a16/kernel.py`
- `b12x/moe/_shared/kernels/w4a16/mixed_trellis.py`
- `b12x/moe/fused_moe/_impl.py`

The exact files shipped from our image are in [b12x-master/](b12x-master/) and
the [Dockerfile](Dockerfile) applies them. The patched vLLM quant layer
([vllm-patches/exl3-mixed.py](vllm-patches/exl3-mixed.py)) is bind-mounted by
the launcher at runtime instead of baked, so it stays inspectable; bake it with
one COPY line if you prefer.

The base image is NOT stock vLLM: it is the brandonmusic SM120 EXL3 fork
(v75p2/v84 lineage, B12X MoE, calibrated NVFP4 MLA KV, DFlash2 support). See
Credits.

## The one line that bought 5x decode

The fork's `exl3.py` `_require_enforce_eager` exempts rank-sliced paths from
eager because they are "eagerly planned before graph capture". The
`r7_routed_experts` path is the same class but was not on the exemption list,
so mixed-rate serves silently booted with `enforce_eager=True`: 25-30 tok/s and
zero graph captures, with no error anywhere. The fix in exl3-mixed.py is one
condition:

```python
if self.rank_sliced_metadata is not None or self.r7_routed_experts is not None:
    return
```

With graphs captured: 126-150 tok/s. If a mixed serve ever looks slow, grep the
boot log for `enforce_eager` and count `Capturing CUDA graph` lines first.

## Repo layout

- [serve/](serve/) - launcher script
- [vllm-patches/](vllm-patches/) - patched EXL3 quant layer + DFlash2 overlay files
- [b12x-master/](b12x-master/) - the four b12x master files the image needs
- [encode/](encode/) - full derivation drivers: inventory, plan, GSS readiness,
  per-layer encode worker, MTP45 adapter encode, pack driver, patched
  materializer, driver README
- [docs/](docs/) - provenance, third-party notices, config surgery, evidence

## Status

- DONE: encode, MTP45 qualification, pack, mixed-rate vLLM serve, 1M pool,
  MTP3 + graphs + tools, live agentic session evidence.
- IN PROGRESS: KLD gate vs teacher logits, GSM8K/HumanEval/MATH-500 suite,
  deep-context decode curve (500K/750K/1M rungs).

## Credits

- zai-org: GLM-5.3-Flash base model
- brandonmusic: the SM120 EXL3 runtime fork (v44-v84), the R10 encoder closure
  and teacher-logits calibration set this derivation is built on, and the
  uniform-K4 lineage these drivers extend
- tonyd2wild: the DFlash2 vLLM overlay ported from the GB10 recipes
- local-inference-lab: b12x (four master files used under its license)
- incoai: the GLM-5.3 DFlash2 drafter weights

License: the checkpoint and scripts inherit the upstream stack's terms (see
[docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md)).
