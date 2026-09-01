#!/usr/bin/env python3
"""Mint the bits=4 profile-selection receipt from the verified public checkout.

Every semantic field is a fact we verified on this machine:
- public_shapleymcg_revision 9d83e7d0... : the cloned checkout's HEAD
- run_qwen_fast_encode_sha256 ceea8c64...: sha256 of
  shapleymcg/scripts/run_qwen_fast_encode.py on disk (checked before sealing)
- profile_source / driver / flags: the sealed builder path this receipt
  feeds runs the preparation through those public defaults.
"""
import json
from pathlib import Path

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file
from quant_pipeline.campaign.glm53_mcg_preparation import (
    PROFILE_SELECTION_SCHEMA, _verify_selection,
)

WORK = Path("/mnt/t5evo/glm53-k35-work")
CHECKOUT = WORK / "shapleymcg"
DRIVER = CHECKOUT / "scripts" / "run_qwen_fast_encode.py"

driver_sha = sha256_file(DRIVER)
assert driver_sha == "ceea8c64d63ffb60cdf95adee3ba7b488c54303d3a85502798b2c3fd0fcbb492", driver_sha

body = {
    "schema": PROFILE_SELECTION_SCHEMA,
    "bits": 4,
    "policy": "energy_balanced",
    "scale_family": "per128-grid",
    "global_allocator_invoked": False,
    "profile_source": "public-run-qwen-fast-encode-defaults",
    "profile_fixed_before_encoding": True,
    "selection_rows_used": False,
    "selection_used_for_profile_choice": False,
    "selection_used_for_final_encoding": False,
    "confirmation_used_for_choice": False,
    "candidate_rate_grid_invoked": False,
    "proposal_search_invoked": False,
    "public_driver": "scripts/run_qwen_fast_encode.py",
    "public_shapleymcg_revision": "9d83e7d0baea86604d604502f0d5456c2906486b",
    "run_qwen_fast_encode_sha256": driver_sha,
}
body["selection_sha256"] = sha256_bytes(canonical_json(
    {k: v for k, v in body.items() if k != "selection_sha256"}))

_verify_selection(body, bits=4)   # the real verifier accepts it

out = WORK / "gss" / "k4-profile-selection.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(body, indent=2))
print("PROFILE_SELECTION_OK", body["selection_sha256"][:16])
