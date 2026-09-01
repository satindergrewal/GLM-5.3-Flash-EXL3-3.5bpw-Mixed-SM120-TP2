# glm53-exl3:dflash2-mixed
#
# Base: glm53-exl3:dflash2
#   = verdictai/... v75p2 SM120 EXL3 runtime (brandonmusic v44-v84 lineage,
#     B12X MoE, calibrated NVFP4 MLA KV) + the tonyd2wild DFlash2 vLLM overlay
#     (with three calling-convention fixes for the fork's API).
# This layer swaps in four b12x files from local-inference-lab/b12x
# master (commit 139e040). Stock b12x 1.2.6 in the base image lacks the
# trellis3 mixed-rate API (bind/compile/run_bound_mixed_trellis3,
# build_projection_tiered_maps) that per-tensor K3/K4 checkpoints need.
# The files below are extracted verbatim from the running image.
#
# Build: docker build -t glm53-exl3:dflash2-mixed .
FROM glm53-exl3:dflash2

COPY b12x-master/w4a16/host.py          /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/w4a16/host.py
COPY b12x-master/w4a16/kernel.py        /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/w4a16/kernel.py
COPY b12x-master/w4a16/mixed_trellis.py /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/w4a16/mixed_trellis.py
COPY b12x-master/fused_moe/_impl.py     /opt/infernal-invocation/b12x/b12x/moe/fused_moe/_impl.py

# Optional: bake the patched quant layer instead of bind-mounting it at
# runtime (serve-k35.sh bind-mounts by default):
# COPY vllm-patches/exl3-mixed.py /opt/infernal-invocation/vllm/vllm/model_executor/layers/quantization/exl3.py
