Post-pack config surgery 2026-09-01 ~14:40 NZ: tensors untouched;
quantization_config.json scope marker removed and r7_routed_experts block
added so the vLLM R7 native projection-tier path (shape-derived K3/K4
tiers) loads this mixed checkpoint. Original saved as .k35-orig. Campaign
materialization-receipt config hashes refer to the pre-surgery file.
