# K35 box encode drivers (GLM-5.3-Flash mixed K3/K4 3.5 bpw, sm120 declared venue)

Files (copy them into the work root next to plan.json):

- `k35_common.py` - shared bindings: codec construction, capture/source opening,
  preparation loading, covariances, the probe loss, the bits-honest packed
  store, the GLM-5.3 DP solver, receipts, state helpers
- `k35_probe_driver.py` - phase 5: per-layer K3/K4 probe losses + sealed
  sensitivity-DP allocation
- `k35_phase6_gss.py` - phase 6: rate-specific GSS (K3 and K4, no reuse) +
  readiness receipt
- `k35_worker.py` - phase 7: one process per GPU, dynamic whole-layer encode
  through the k35 state machine
- `vendored/` - byte-exact copies of the two producer sources the sealed
  closure expects that exist in the repo (streaming_v31.py, exl3_mcg.py),
  plus `vendored/VENDORED-SHA256SUMS` documenting them and the
  unavailable `scripts/run_qwen_fast_encode.py` pinned sha; see phase 6

Everything here is CODE ONLY: it was written without touching any GPU or
remote box, and nothing was executed against the campaign data.

## Environment (inside container glm53-k35-encode)

```
export WORK=/mnt/t5evo/glm53-k35-work
export PYTHONPATH=$WORK/src:$WORK/reproducibility/r10
export K35_EXLLAMAV3_EXT=/path/to/exllamav3_ext.so   # from the container image
cd $WORK
```

The drivers never hardcode the roots: `--work-root` (default
`/mnt/t5evo/glm53-k35-work`), `--calibration-root` (default
`/mnt/t9/glm53-archive/brandonmusic_GLM-5.3-Flash-BF16-Teacher-Logits/calibration`),
`--bf16-root` (default `/mnt/t5evo/GLM-5.3-Flash-BF16`), `--extension` or
`K35_EXLLAMAV3_EXT`.  The r10 bundle is found on sys.path or via
`--repo-root`.

Prerequisite phases (runbook section 8): phase 2 container up, phase 3
sealed inventory at `$WORK/inventory.json`, phase 4 sm120 declared preflight
+ launch plan built with PROVISIONAL allocations + `state/state-0000.json`.

## Phase 5: probe pass, per layer (3..44, then 45 = MTP)

One layer per invocation; embarrassingly per-layer and safe to pause
between layers.

```
cd $WORK
for L in $(seq 3 45); do
  python3 k35_probe_driver.py --layer $L --work-root $WORK
done
```

Each invocation prints exactly one JSON line:

```
{"layer": 3, "k4_tensors": 432, "worst_k3_loss": 0.0113, "alloc_sha256": "..."}
```

Writes:
- `probes/L{NN}.json` - sealed probe ledger (NEW SURFACE schema
  `quant-pipeline.glm53-k35-probe-ledger.v1`)
- `allocations/L{NN}.json` - sealed NON-provisional allocation,
  `basis="sensitivity_dp_probe_v1"` (sealed by
  `k35.seal_layer_allocation`, glm53_uniform_k35.py:380-393)

The probe needs a K4-rate preparation for the layer (vectors and
permutations; `--preparation-root`, default `$WORK/gss/k4`).  If the K4
preparations have not been produced yet, either point `--preparation-root`
at the original K4 campaign preparation root or run phase 6 first with
`--build-k4`.  Probing with K4-rate vectors for both rates is a documented
choice recorded in the ledger (phase 5 precedes phase 6 in runbook order);
the final encode uses rate-specific vectors.

## Phase 5b: rebuild the plan with the DP allocations (MANDATORY)

`claim_next_layer` fail-closes on provisional allocations baked into the
plan at build time (glm53_uniform_k35.py:920-926); the allocation files on
disk are not consulted by the claim.  Rebuild the plan and RESTART the
state chain (states bind `launch_plan_sha256`,
glm53_uniform_k35.py:781-782, so every prior state receipt is dead weight
once the plan changes):

```
cd $WORK && python3 - <<'PY'
import json
import shutil
from pathlib import Path
from quant_pipeline.campaign import glm53_uniform_k35 as k35
work = Path(".")
history = work / "state" / "history"
history.mkdir(parents=True, exist_ok=True)
for old in sorted((work / "state").glob("state-*.json")):
    shutil.move(str(old), str(history / old.name))
inventory = json.load(open("inventory.json"))
preflight = json.load(open("preflight-sm120-declared.json"))
baseline = {"five_cold_run_kld_receipt": json.load(open("baseline/k4-five-cold-run.json"))}
allocations = {}
for L in [*range(3, 45), 45]:
    allocations[L] = json.load(open(f"allocations/L{L:02d}.json"))
    k35.verify_layer_allocation(allocations[L], layer=L)
    assert allocations[L]["provisional"] is False
plan = k35.build_launch_plan(
    inventory, preflight, four_bpw_baseline=baseline,
    layer_allocations=allocations, allow_declared_sm120_preflight=True)
k35.verify_launch_plan(plan)
json.dump(plan, open("plan.json", "w"), indent=2)
state = k35.initial_state(plan)
json.dump(state, open("state/state-0000.json", "w"), indent=2)
print("plan", plan["launch_plan_sha256"][:16], "state0", state["state_receipt_sha256"][:16])
PY
```

Superseded state files must move into `state/history/` BEFORE the new
`state-0000.json` is written (the snippet above does this).  History is
still the audit trail and is never deleted; the move only takes the old
files out of the selection glob.  As a second belt, `k35_common.newest_state`
filters state files by the CURRENT `launch_plan_sha256` and ignores
superseded-plan files left behind in place; if nothing matches it dies with
`no state bound to the current plan; run the phase-5b bootstrap` instead of
wedge-dying on verify_state's "state receipt targets a different launch
plan".  A worker that is alive across the re-plan re-loads plan.json every
loop iteration, exits cleanly at its next claim boundary, and never appends
old-plan successors.

## Phase 6: rate-specific GSS + readiness receipt

```
cd $WORK
python3 k35_phase6_gss.py --work-root $WORK                 # K4 prep already present
python3 k35_phase6_gss.py --work-root $WORK --build-k4 \
    --profile-selection $WORK/gss/k4-profile-selection.json # if K4 prep must be built
```

Writes `gss/k3/layer-{NNN}/` (NEW SURFACE schema
`quant-pipeline.glm53-k35-rate3-gss-preparation.v1`), verifies
`gss/k4/layer-{NNN}/` (sealed schema, bits=4), checks permutation identity
across the two halves, cross-checks policy/scale_family/transform seed on
EVERY layer's K4 manifest, derives the readiness receipt's
`codec_identity_sha256` from the sealed rate-3 manifests (checked identical
across layers and against the live codec, recorded separately from the K4
venue identity in `k4_codec_identity_sha256`), and seals
`gss/readiness-receipt.json`
(NEW SURFACE schema `quant-pipeline.glm53-k35-readiness-receipt.v1`).
The final line prints the readiness receipt hash.

Producer-source closure (CHANGED SURFACE, WARN): the sealed builder's
`_producer_closure` (glm53_mcg_preparation.py:234-255) requires
`scripts/run_qwen_fast_encode.py` plus
`src/quant_pipeline/{normalization/streaming_v31.py,codecs/exl3_mcg.py}`
under the source root, but the released deliverable contains
`run_qwen_fast_encode.py` nowhere (its sha256 is pinned at
`ceea8c64d63ffb60cdf95adee3ba7b488c54303d3a85502798b2c3fd0fcbb492` by the
sealed profile-selection verifier), so the sealed closure surface is
unsatisfiable over everything shipped.  The rate-3 build therefore derives
its closure over the sources that DO exist (the r7_encoder bundle, the two
vendored producer sources - verified byte-identical to the imported
campaign modules - and the campaign adapter), records the missing file as
loud manifest fields (`producer_source_closure_surface_changed`,
`_missing`, `_note`), and computes the closure BEFORE the GPU pass so it
fails in seconds.  `--build-k4` preflights the sealed closure surface and
dies immediately with the missing-file list instead of after a full GPU
pass inside the sealed builder; to use `--build-k4`, vendor the genuine
`run_qwen_fast_encode.py` at the pinned sha plus the two vendored files
under the r10 bundle root first (see `vendored/VENDORED-SHA256SUMS`).
Abandoned PID-named staging directories whose owner process is gone are
swept on entry.

## Phase 6b: enter the encoding phase

```
cd $WORK && python3 - <<'PY'
import json
from quant_pipeline.campaign import glm53_uniform_k35 as k35
plan = json.load(open("plan.json"))
state = json.load(open("state/state-0000.json"))
readiness = json.load(open("gss/readiness-receipt.json"))
successor = k35.enter_k35_encoding(
    plan, state, readiness_receipt_sha256=readiness["readiness_receipt_sha256"])
json.dump(successor, open(f"state/state-{successor['sequence']:04d}.json", "w"), indent=2)
print("state", successor["sequence"], successor["phase"])
PY
```

## Phase 7: two workers, one per GPU

Worker ids come from the PLAN, not the shell: the sm120-declared venue
assigns `sm120-0` and `sm120-1` (f-string `"sm120-{slot}"` over the
preflight gpus rows, glm53_uniform_k35.py:293).  With preflight gpu indices
[1, 3] the ids are still sm120-0/sm120-1, NOT sm120-1/sm120-3.

```
cd $WORK
CUDA_VISIBLE_DEVICES=0 python3 k35_worker.py --worker sm120-0 \
    --work-root $WORK --reader-abi-sha256 <sealed-reader-abi-sha> &
CUDA_VISIBLE_DEVICES=1 python3 k35_worker.py --worker sm120-1 \
    --work-root $WORK --reader-abi-sha256 <sealed-reader-abi-sha> &
```

DEVICE TRAP: `docker run --gpus '"device=1,3"'` renumbers the GPUs to 0 and
1 inside the container.  The runbook's `CUDA_VISIBLE_DEVICES=1/3` host
indices only work if the container exposes the host numbering; with the
shared container from runbook phase 2, use 0 and 1 (the worker prints its
plan row's `cuda_visible_devices` for cross-checking at startup).

Startup binding: before taking any claim, the worker verifies the phase-6
readiness receipt's seal, dies unless its `launch_plan_sha256` matches the
current plan and its `codec_identity_sha256` equals the worker's own
`sha256(canonical_json(codec.identity))` (so the encode codec is actually
bound to the receipt, not an unbound attestation), and enforces every
loaded K4/K3 preparation manifest's `preparation_sha256` against the
receipt's `per_layer` list.  Each loop iteration re-loads plan.json; if the
launch plan changed on disk (a phase-5b re-plan) the worker exits cleanly
at its next claim boundary instead of appending old-plan successors.

Per layer the worker writes `layers/L{NN}/`:
- `payload-store/objects/` + `payload-store/choices/` (glm53_direct_k4
  layout; choices carry honest per-tensor bits under the k35 schema)
- `experts/layer-{NNN}/expert-{EEE}.json` (NEW SURFACE expert receipts)
- `hessians/layer-{NNN}/expert-{EEE}.safetensors`
- `layer-receipt.json` (NEW SURFACE; its `receipt_sha256` is what
  `complete_layer` binds)

After the loop: phase 8 (`seal_main_k35`, MTP45 adapter through the
separate glm53_mtp_k4-style track, `seal_k35_packed`) is NOT part of these
drivers; the workers exit when the pending queue is empty.

## State-file conventions

- `state/state-{sequence:04d}.json`, one sealed receipt per transition:
  claim (+1), completion (+1), recovery (+1).  Claim successors are written
  BEFORE encoding starts; that is the crash checkpoint.
- Superseded-plan state files live in `state/history/` after a phase-5b
  re-plan (the selection glob is non-recursive, and selection additionally
  filters by the current `launch_plan_sha256`); history is the audit trail
  and is never deleted.
- `state/.lock` is an exclusive flock held during read-claim-write and
  read-complete-write; the state machine is the logical lock, the flock
  serializes the file critical section between the two workers.
- Never delete state history: the chain is the checkpoint and the audit
  trail.  Never hand-edit a state file: seals fail closed
  (glm53_uniform_k35.py:773-877).
- `PAUSE` (any content) in the work root is the drain-pause sentinel.

## Pause discipline

- DRAIN PAUSE (preferred): `touch $WORK/PAUSE`.  Each worker checks before
  its NEXT claim, finishes any in-flight layer completely (receipts + next
  state), then exits 0.  Then `docker pause glm53-k35-encode` if needed.
  Resume: remove PAUSE, `docker unpause`, restart both worker commands.
- HARD PAUSE: `docker pause glm53-k35-encode` mid-layer freezes both
  processes in place with GPU memory resident; `docker unpause` resumes the
  in-flight work exactly.  Safe on a single-purpose encode box.
- CRASH (container killed): completed layers are sealed and safe.  The
  newest state carries the dead worker's active claim:
  - restart the SAME worker id: it resumes its own claim and skips experts
    whose receipts already verify (per-expert resume; every pre-existing
    choice is re-verified against the payload store - which re-hashes each
    content-addressed object file - and each Hessian artifact is re-hashed
    on disk, mirroring the K4 resume verifier);
  - worker gone for good: `python3 k35_worker.py --recover-worker <id> --work-root $WORK`
    drops the claim, QUARANTINES the layer's partial per-claim artifacts,
    and re-queues the layer at the front of pending in a recovery
    successor.  Quarantine removes `layers/L{NN}/experts/layer-{NNN}/`
    (expert receipts), `layers/L{NN}/payload-store/` (per-layer store), and
    a stale `layers/L{NN}/layer-receipt.json`; the Hessians under
    `layers/L{NN}/hessians/` are kept (claim-independent, byte-identical
    artifacts tolerated).  This deletion is REQUIRED, not optional: every
    expert receipt, packed choice, and the choice predecessor chain embed
    the dead claim's `claim_receipt_sha256`, the re-claim necessarily mints
    a new hash, and surviving receipts would wedge the layer forever.  The
    removal happens under the state lock and is recorded in the recovery
    successor's evidence block (`k35_recovery_quarantine`).  WARN: the
    recovery transition is not offered by the public state machine; it is
    sealed through k35._successor and recorded in the successor evidence.
    Never run it while the worker might still be alive.

## Runbook-vs-code API mismatches (file:line)

1. Work root drift.  Runbook section 8: `/mnt/t9/glm53-k35-work` with the
   repo at `$WORK/repo`.  Task/container PYTHONPATH:
   `/mnt/t5evo/glm53-k35-work` with the repo directly under `$WORK`.
   The drivers default to the task layout and accept `--work-root`
   everywhere; calibration stays on `/mnt/t9`.
2. r7 DP solver is geometry-locked to GLM-5.2.  `r7_encoder/constants.py:15`
   (NUM_EXPERTS=256), `:35-40` (TENSORS_PER_LAYER=768,
   TARGET_BIT_UNITS_PER_LAYER=2688, UPGRADE_UNITS_PER_LAYER=384),
   `:56-62` (TensorId rejects layer>77 / expert>=256),
   `allocation.py:194-211` (audit demands the 2688 sum).  GLM-5.3 needs
   864 tensors / 3024 / 432 on layers 3..45.  Ported byte-faithfully in
   `k35_common.solve_layer_dp` (same 10^15 ROUND_HALF_EVEN integer scores,
   same low-bits-first / strict-gain tie rule, allocation.py:144-157);
   the sealed authority is `k35.audit_layer_allocation`
   (glm53_uniform_k35.py:348-360).  Note also the id-vocabulary difference:
   r7 `TensorId.key` is `L{layer:02d}/E{expert:03d}/{projection}`
   (constants.py:66) while the codec unit id is
   `L{L}.E{e}.{projection}` (codecs/exl3_mcg.py:154); the drivers use the
   codec vocabulary and full HF names in sealed documents.
3. Probe loss is not returned by the R10 codec.  The fast path sets
   `proxy_loss=0.0` and `covariance_proxy_loss_evaluated: False`
   (r10_codec.py:512, :431).  The K4-closure loss formula lives inline in
   the R7 audited path (trellis.py:383-396).  The probe driver recomputes
   that exact quadratic (float64, denominator clamp 1e-30) over the R10
   candidate reconstruction; the bridge is the documented R10/R7
   byte-compatibility (r10_codec.py module docstring).  Cross-check
   command (optional, one tensor): encode through
   `r7_encoder.trellis.Exl3TrellisCodec.encode` at bits 3 and 4 and compare
   proxy_loss.  The uniform-K4 campaign itself never ran a DP, so no K4
   campaign probe-loss code exists to reuse verbatim; this is the honest
   bridge and it is recorded in every probe ledger header.
4. The prepared backend rejects mixed rates.  `Glm53PreparedMCGBackend`:
   SUPPORTED_BITS=(4,6) (glm53_prepared_backend.py:35), contract rate gate
   (:117-124), per-request bits gate in `_encode_batch` (:453).  The worker
   mirrors the backend's operation order (gate/up covariance from fit rows,
   grouped encode, cache clear, candidate-conditioned down Hessian, down
   encode; :442-586) through `Exl3MCGCodec.encode_candidates` at the
   allocated rate.  Its grouped lockstep batching (:483) is a uniform-rate
   surface; the worker encodes per tensor, so wall-clock differs from K4
   (throughput follow-up, not a receipt change).
5. The preparation builder rejects bits=3.  `build_layer_preparation`
   raises outside (4,6) (glm53_mcg_preparation.py:39, :278-279);
   `_verify_selection` binds a sealed (4,6) profile-selection receipt
   (:57-85); `seal_campaign_preparation` enforces one uniform rate
   (:456-458).  Phase 6 replays the sealed builder's own numeric helpers at
   bits=3 under a NEW k35 schema with WARN banners (the GSS core itself is
   rate-agnostic: `CorrectedPinnedGSSProducer.search` ->
   `backend._quant_args(int(bits), ...)` accepts 3/4/5,
   qwen_services.py:287, r10_codec.py:392-393).
6. The packed-choice store hardcodes bits=4.  `PackedMCGPayloadStore
   .put_choice` writes `"bits": 4` in every choice body and
   `verify_choice` rejects bits != 4 (checkpoint/packed_payload.py; seal
   map section 1.9 cites lines 136 and 191).  A K3 choice stored through it
   would lie about its rate.  `k35_common.K35PackedPayloadStore` keeps the
   identical layout (objects/ + choices/, same hash framing) with honest
   per-choice bits and an added trellis-byte-count-vs-bits check.
7. No readiness contract exists.  Only the hash field
   `k35_readiness_receipt_sha256` (glm53_uniform_k35.py:856 verify,
   :897-899 enter; K4 analogue glm53_uniform_k4.py:475, :516-517) and the
   `preparation_contract` block (glm53_uniform_k35.py:599-605).  The
   readiness receipt schema is NEW SURFACE; the state machine accepts any
   64-hex hash, so the receipt document is what the hash promises.
8. Worker id vocabulary.  sm120-declared plans accept exactly `sm120-0`,
   `sm120-1` (f-string over the preflight gpus enumeration,
   glm53_uniform_k35.py:293); sealed four-B200 plans accept `b200-0..3`
   (glm53_uniform_k4.py:211).  The claim error text says "unknown B200
   worker" on every venue including sm120 (glm53_uniform_k35.py:911).
9. MTP45 capture needs the other adapter.  `Glm53CaptureView` rejects layer
   45 (glm53_direct_k4.py:207-208); the probe driver opens
   `Glm53MTP45CaptureView` (glm53_mtp_k4.py:54) for layer 45.  The worker
   only serves the main scheduler: `claim_next_layer` draws from the MAIN
   queue only (glm53_uniform_k35.py:915-918); MTP45 encode is the separate
   adapter phase 8 (verify_mtp_adapter_receipt, glm53_uniform_k35.py:998).
10. Plan rebuild + chain restart after phase 5 (see phase 5b above).  The
    claim reads `allocation_provisional` from the plan built at phase 4
    (glm53_uniform_k35.py:522, :920-926), not from disk; and every state
    binds the plan hash (glm53_uniform_k35.py:781-782).
11. Codec import order is enforced.  `Exl3MCGCodec._codec()` refuses to
    construct when any `r7_encoder` module is already cached
    (codecs/exl3_mcg.py:127-129).  All drivers construct the codec FIRST
    (`k35_common.build_codec` forces the sealed import) before anything
    imports `r7_encoder.hessian`.
12. No public recovery transition.  Runbook phase 7 says a crashed layer
    "re-queues after the restart drops the stale active claim in a recovery
    successor"; no public k35 function implements it.  The worker's
    `--recover-worker` seals it through the module-private
    `k35._successor` with an explicit WARN and an evidence note.
13. Canonical JSON vocabularies differ across the repo.  Campaign sealing
    uses `quant_pipeline.core.artifacts.canonical_json` (sort_keys, trailing
    newline, ensure_ascii=False; core/artifacts.py:15); the r7 bundle uses
    `r7_encoder.determinism.canonical_json_bytes` (no newline,
    ensure_ascii=True).  All k35 driver seals use the campaign functions,
    matching the state machine.  Never mix the two when hashing.
14. NEW SURFACE receipt schemas defined by these drivers (no sealed
    validators exist): probe ledger, rate-3 GSS preparation, readiness
    receipt, packed choice, expert receipt, layer receipt.  Each carries a
    WARN in its builder and must be registered in the derivation report
    before its artifacts count as campaign evidence.

## Operational notes

- `--verify-shards` re-hashes the full BF16 master (~643 GB) against the
  sealed inventory; default OFF because the per-layer drivers would repeat
  it per invocation.  Run it once (any driver, one layer) at campaign
  start.
- Per-layer capture hashing is ON by default (the sealed capture view
  verifies that layer's three payload files); `--no-verify-capture-hashes`
  trades that seal for speed.
- The probe driver encodes gate/up at BOTH rates per tensor and the down
  projection at both rates under ONE conditioning Hessian per expert; budget
  roughly 2.5x a uniform-K4 encode pass per layer.  At encode time, experts
  whose gate or up rate differs from the reference rate add one
  conditioning-only encode per mismatched tensor.
- Everything the drivers print is one JSON object per line; the probe
  driver's LAST line is exactly the task-contract summary line.

## Down-projection conditioning policy (ONE context: R7 pair_at semantics)

Probe, DP solve, and encode share ONE conditioning context per expert for
the whole down curve: the conditional-fit Hessian conditioned on gate/up
reconstructions decoded at the reference rates `k35.FLOOR_BITS` (k3/k3),
mirroring R7's memoized `pair_at(base_gate_bits, base_up_bits)` context
(r7_encoder/layer.py:901-925, 974-994).  Consequences:

- the probe's loss@3 and loss@4 are same-denominator relative quadratics,
  so the DP gain `mass*(loss3-loss4)` subtracts one metric (per-rate
  conditioning compared ratios under different matrices and could invert
  the ranking);
- the worker's encode-time down Hessian is conditioned on the identical
  context regardless of the expert's allocated gate/up rates; mismatched
  tensors get one extra conditioning-only encode at the reference rate
  (deployed choices keep the allocated rates);
- every receipt stamps what was ACTUALLY used: the evidence construction
  string carries the real rates
  (`decoded-gate-k3-up-k3-candidate-conditioned-...`), numeric
  `conditioning_gate_bits`/`conditioning_up_bits` fields accompany it, the
  expert receipt's `down_conditioning` records the conditioning rates plus
  the deployed rates, and `verify_expert_receipt` fails when the stamp and
  the recorded rates disagree.

## FIXES-2026-08-29

Applied review fixes (spec: `CONFIRMED-FINDINGS.json`, 10 findings; the
JSON's fix_hint fields for findings 4/9 and 7/10 were cross-wired, so each
fix follows the hint that matches its claim):

1. k35_worker.py - recovery wedge: `--recover-worker` now quarantines the
   re-queued layer's per-claim artifacts (receipts, payload-store, stale
   layer receipt) under the state lock before sealing the successor, so a
   fresh claim re-encodes cleanly instead of dying on foreign-claim
   receipts forever.
2. k35_common.py - `newest_state` takes the plan, filters state files by
   the current `launch_plan_sha256`, and dies with a clear 5b-bootstrap
   message when nothing matches (no more wedge on superseded-plan states).
3. k35_probe_driver.py - the down curve is probed under ONE conditioning
   context (gate/up decoded at the reference rates k3/k3, R7 pair_at
   semantics) for both candidate rates, so the DP gain subtracts
   same-denominator losses.
4. k35_worker.py - the encode-time down Hessian uses the identical
   reference-rate conditioning context (extra conditioning-only encodes for
   mismatched gate/up), matching the probe instead of conditioning on
   deployed rates; receipts record both conditioning and deployed rates.
5. k35_phase6_gss.py - producer closure: the unsatisfiable sealed surface
   (run_qwen_fast_encode.py absent repo-wide) is replaced by a derivation
   over the sources that DO exist (loud WARN, manifest surface-change
   fields), computed BEFORE the GPU pass; `--build-k4` preflights the
   sealed surface and dies in seconds; abandoned staging dirs are swept;
   the two obtainable producer sources are vendored in `vendored/`.
6. k35_worker.py + k35_common.py - conditioning receipts stamp what was
   actually used (construction string + numeric rates + deployed rates),
   and `verify_expert_receipt` cross-checks the stamp against the recorded
   rates; the readiness receipt's conditioning claim pins the actual
   construction.
7. k35_worker.py - same root cause as 1: recovery deletion is the seal-safe
   option (receipts single-claim); the relaxation alternative was rejected.
8. k35_phase6_gss.py - the readiness receipt's `codec_identity_sha256` is
   derived from the sealed rate-3 manifests (identical across layers,
   checked against the live codec), the K4 venue identity moves to a
   separate `k4_codec_identity_sha256` field, and the worker dies at
   startup unless its own codec identity equals the receipt's.
9. k35_common.py + k35_phase6_gss.py + k35_worker.py - `load_preparation`
   enforces the sealed loader's semantic fields (policy, scale family,
   profile source/flags, `shapleymcg_process_structure`), phase 6
   cross-checks policy/family/seed on EVERY K4 layer, and the worker binds
   each loaded K4/K3 manifest's seal to the readiness receipt's per_layer
   hashes.
10. k35_worker.py - the resume branch re-verifies every pre-existing choice
    against the payload store (re-hashing content-addressed objects) and
    re-hashes the Hessian artifact on disk, mirroring the K4 resume
    verifier, so post-receipt corruption fails at resume instead of phase 8.

