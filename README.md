# GLM-5.3-Flash EXL3 3.5bpw Mixed (K3/K4) - SM120 TP2 Recipe

First sub-4bpw serving of GLM-5.3-Flash on 2x RTX PRO 6000 (SM120, 96 GB
each): **1M context + MTP3 speculative decoding + CUDA graphs + thinking
mode, simultaneously.**

At 4bpw this combination does not fit: the drafter KV plus a 1M-token pool
exceeds 192 GB. The mixed-rate 3.5bpw checkpoint frees 9.07 GiB per GPU,
which is exactly the gap. One serve does deep context AND fast decode.

Checkpoint on Hugging Face:
[satgeze/GLM-5.3-Flash-EXL3-TR3-3.5bpw](https://huggingface.co/satgeze/GLM-5.3-Flash-EXL3-TR3-3.5bpw)

**Benchmarks (GSM8K, HumanEval, MATH-500, deep-context decode curve) and
the five-run KLD fidelity receipt are being measured now and will be
published here and on the model card.**

## Quick start

```bash
./download.sh        # pull the checkpoint from HF next to the repo (aria2c)
./start.sh           # port 8012, 1M ctx, graphs, tools; waits for /health
./status.sh          # container + endpoint + GPU state at a glance
./stop.sh            # clean shutdown incl. the process-tree sweep
```

Boot is about 13 minutes (120 shards + b12x compile + CUDA graph capture).
Every knob is an env override; see [.env.example](.env.example). Bench the
running serve: `python3 bench/bench.py`.

## What runs (the stack, layer by layer)

| Layer | What | Where |
|---|---|---|
| API | OpenAI-compatible, tool calling (glm45 parser) | serve-k35.sh flags |
| Model | GLM-5.3-Flash 320B MoE, language profile | artifact config.json |
| Quant | EXL3/TR3 MCG, per-tensor mixed K3/K4 (37,152 routed decisions) | vllm-patches/exl3-mixed.py |
| MoE kernels | b12x trellis3 mixed-rate API (4 files from master 139e040) | b12x-master/ |
| Image | v75p2 lineage + DFlash2 overlay + b12x swap | Dockerfile |
| KV cache | calibrated NVFP4 MLA | CACHE=nvfp4_ds_mla |
| Spec decode | MTP3 built in | PROFILE=long500k |
| Parallelism | TP2, DCP1, A2A | serve-k35.sh |
| Context | 1,000,000 max, 1.99x concurrency at 1M | MAX_MODEL_LEN |

## Results (2026-09-01, this exact weight set)

Policy: if a number is not in a dated table, treat it as unverified.

| Metric | Value | Verified |
|---|---|---|
| Decode, engine-side, MTP3, thinking on | 126-150 tok/s | serving |
| Decode, real agentic client session (LAN, Spock) | 51-109 tok/s client-side | serving |
| MTP3 acceptance length | ~2.4 mean | serving |
| KV pool | 1,985,915 tokens | serving |
| max_model_len | 1,000,000 (1.99x concurrency) | serving |
| CUDA graphs | PIECEWISE x4 + FULL | boot log |
| Tool calling | OpenAI format roundtrip verified | serving |
| Boot to ready | ~13 min | boot log |
| Checkpoint | 145.4 GiB, 120 shards, 150,226 tensors | pack receipt |

Reference on the same hardware: the 4bpw DFlash2 cell peaks at 78-91 tok/s
at 192K-230K context. This beats that decode rate at 5x the context.

Evidence: [docs/evidence/](docs/evidence/) - includes a real multi-turn
agentic session with tool calls and long reasoning.

## Quality gates (this exact weight set)

| Gate | State | Number |
|---|---|---|
| GSM8K 5-shot (T=0.2, thinking on) | MEASURED (2026-09-02) | 96.89 (4bpw ref: 96.51) |
| MATH-500 (boxed+normalized, budget retry) | MEASURED (2026-09-02) | 77.8 raw (4bpw raw ref: 75.4) |
| HumanEval (canonical subprocess checks) | MEASURED (2026-09-02) | 82.32 raw, 28 thinking-budget empties (corrected class ~98.5) |
| Deep-context decode curve 500K/750K/950K | RUNNING | - |
| KLD vs BF16 teacher (5 cold runs, 51,175 positions, same 0.06 bar as 4bpw gate) | STAGED | 4bpw reference: 0.024555 |

Numbers will be published to the HF card and here when the runs complete.

## Why the patched image exists

Stock components cannot read this checkpoint. Three things were needed:

1. **Mixed rates were rejected.** The offline runtime and quant layer
   hard-bound uniform integer rates at multiple gates; the mixed marker,
   48-word (K3) trellises, and per-tensor rate censuses all had to become
   first-class. Twelve anchored edits in
   [vllm-patches/exl3-mixed.py](vllm-patches/exl3-mixed.py) plus the runtime
   widening documented in [encode/](encode/).
2. **b12x 1.2.6 lacks the trellis3 mixed API.** Four files from
   local-inference-lab/b12x master (commit 139e040) supply it; the stock
   release does not have them. The [Dockerfile](Dockerfile) applies the
   swap; the image's own 1,673-line SM120 patches are kept.
3. **The eager trap.** Without the one-line exemption in exl3-mixed.py
   (`r7_routed_experts` added to `_require_enforce_eager`'s exempt list),
   the serve silently boots eager: 25-30 tok/s, zero graph captures, no
   error anywhere. With it: 126-150 tok/s. See "The one line that bought
   5x decode" below.

## The one line that bought 5x decode

```python
if self.rank_sliced_metadata is not None or self.r7_routed_experts is not None:
    return
```

Mixed-rate paths are eagerly planned before graph capture, same as
rank-sliced paths; the exemption list just did not include them. If a mixed
serve ever looks slow: grep the boot log for `enforce_eager` and count
`Capturing CUDA graph` lines first.

## Do not

- Do not omit `VLLM_EXL3_PREFILL_BLOCK_M=64` (or 32). The image default 128
  fails the FC2 route-subtile check; allowed routed sizes are 8/16/32/48/64.
- Do not set DCP=2 on 2 discrete cards (A2A handshake wedge; DCP=1 is the
  qualified config on this hardware).
- Do not edit the standalone quantization_config.json expecting vLLM to see
  it: vLLM reads the quantization_config EMBEDDED in config.json.
- Do not use `docker rm` alone to stop the serve: the engine process tree
  can survive it and keep holding ~90 GB per GPU. Use ./stop.sh, which
  sweeps and verifies VRAM is actually free.
- Do not add the np0 fabric ports on DGX Spark pairs (they show link-local
  + PORT_ACTIVE with no reachable peer).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Boot dies: "FC2 route subtile must be an allowed divisor" | prefill block 128 | VLLM_EXL3_PREFILL_BLOCK_M=64 |
| Boot dies: "requires eager execution" | old quant layer | image built from this repo's Dockerfile |
| Serves but 25-30 tok/s | eager mode (exemption lost) | confirm exl3-mixed.py is the bind-mounted file; check graph captures |
| "requires the projection-tier B12X API" | stock b12x in image | rebuild image; four master files must land |
| Tool 400: 'auto' tool choice requires... | parser flags missing | serve-k35.sh from this repo passes them |
| Port free but serve "missing" | container name derives from CACHE/DCP | ./status.sh finds it by image |

## Status

| Item | State |
|---|---|
| Derivation (encode, MTP45, pack) | complete, sealed receipts |
| Mixed-rate vLLM serve, 1M pool, MTP3 + graphs + tools | complete, serving |
| Real-session evidence | complete |
| KLD five-run gate | staged; GPU window pending |
| Eval suite (GSM8K/HumanEval/MATH-500) + deep-context curve | pending |
| Vision profile | not exercised (language profile only) |

## Repo layout

- [start.sh](start.sh) / [stop.sh](stop.sh) / [status.sh](status.sh) /
  [download.sh](download.sh) - lifecycle
- [serve/serve-k35.sh](serve/serve-k35.sh) - raw launcher (env-driven)
- [vllm-patches/](vllm-patches/) - patched EXL3 quant layer + DFlash2 overlay
- [b12x-master/](b12x-master/) - the four b12x master files the image needs
- [encode/](encode/) - derivation drivers: inventory, plan, readiness,
  encode worker, MTP45 adapter, pack, patched materializer
- [bench/bench.py](bench/bench.py) - serve benchmark CLI
- [docs/](docs/) - provenance, third-party notices, config surgery, evidence

## Credits

- zai-org: GLM-5.3-Flash base model
- brandonmusic: the SM120 EXL3 runtime fork (v44-v84), the R10 encoder
  closure and teacher-logits calibration set this derivation is built on,
  and the uniform-K4 lineage these drivers extend
- tonyd2wild: the DFlash2 vLLM overlay ported from the GB10 recipes
- local-inference-lab: b12x (four master files used under its license)
- incoai: the GLM-5.3 DFlash2 drafter weights
- turboderp: EXL3/TR3 format and kernels

License: the checkpoint and scripts inherit the upstream stack's terms (see
[docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md)).
