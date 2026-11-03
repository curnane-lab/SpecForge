# Ascend NPU capture baseline (N0) — status and SHA inventory

Date: 2026-11-02 (Beijing). Owner: NPU capture working branch
`npu/spec-capture-ascend` (this repo) and `npu/capture-forward-observer`
(curnane-lab/sglang).

This is the N0 deliverable of the Ascend capture plan: a reproducible record
of the NPU deployment baseline, the extension-point dependency closure, and
what has and has not been verified. It is not a hardware gate report; the
remaining N0 exit items are listed at the end.

## Component SHAs

| Component | SHA / ref | Notes |
| --- | --- | --- |
| SpecForge upstream main | `af1c3f3868e0ad1d1bc4a3bc4c577f2c9d3fa7b5` | base of `npu/spec-capture-ascend` |
| SGLang upstream main (fetched) | `4ae54e94588380c45565eb3edee73ad6435d7d9a` | later than the plan snapshot `0c29b0b9` |
| NPU SGLang fork base | `7fe5b127e4ac2a3b2c67e1bc1fb2a299daf0983e` | upstream commit, 2026-10-27 20:19 +0800 |
| NPU SGLang fork head | `88a0368750` (`feat/npu-gdn-compact-verification`) | 5 NPU commits on top of the base |
| #43450 cherry-pick | `b462f44827` (`npu/capture-forward-observer`) | applies cleanly, zero conflicts |
| Legacy rollback | `npu-legacy-capture-freeze` = `55c6711` (this repo) | hspec offline/file-protocol route |

## Extension-point dependency closure

Verified against the fork source, not just symbols:

- **#43449 (target-only aux capture): already in the fork** — it merged
  upstream ~8h before the fork point. `configure_aux_hidden_state_capture`
  dispatch exists in `model_runner.py`; the capture targets
  `qwen3_5.py` / `qwen3_5_text.py` already implement
  `set_eagle3_layers_to_capture` / `set_dflash_layers_to_capture`, and
  `qwen3_next.py` (GDN hybrid) has `layers_to_capture` wiring. E1 model
  enablement may turn out to be a no-op for these targets; confirm per model
  with a real aux + last-hidden forward.
- **#43450 (forward observer + defer_response): was missing**; cherry-picked
  as `b462f44827`. Its dependencies (`managers/auxiliary_output.py` with
  `append_auxiliary_output` / `CompositeHostAuxiliaryOutput`,
  `runtime_context.get_disagg`, `hardware_backend/mlx`, `_async_d2h`) were all
  already present in the fork.

## Verification log (this host: 8x 910B3, torch 2.10.0+cpu + torch_npu, venv-gdn-compact)

| Check | Result |
| --- | --- |
| sglang unit tests brought by #43450 (`test_deferred_output.py`, `test_forward_observer.py`) against the fork + cherry-pick | 33 passed |
| SpecForge plugin integration test (`test_sglang_capture_plugin_integration.py`, real scheduler/streaming code with a Mooncake stand-in) against the fork | 1 passed |
| Plugin unit + sink suites (`test_sglang_capture_plugin.py`, `test_sglang_capture_plugin_sink.py`, includes the N1 Ascend changes) under the NPU venv against the fork | 60 passed, 6 skipped (CUDA-only). Also 59 passed, 7 skipped with `TORCH_DEVICE_BACKEND_AUTOLOAD=0` (no torch_npu registered): NPU devices in the tests are stand-ins/local mocks, so the suites pass on plain CPU/CUDA hosts too |
| N0 exit 1: plain forward / minimal generation, Qwen3.5-4B (qwen3_5, GDN hybrid VL) on device 3, `disable_cuda_graph`, `mamba_ssm_dtype=bfloat16` | passed (`scripts/npu/smoke_plain_forward.py`) |
| N0 exit 2: boot with `enable_return_aux_hidden_states` + `speculative_algorithm=DFLASH` + `return_aux_hidden_state_layer_ids=[2,16,29]` + `return_hidden_states_mode=full` + `chunked_prefill_size=-1`; full-prefill `max_new_tokens=0` request | passed; `aux_hidden_state_capture` resolved to `dflash`, draft algorithm reset as designed |
| N0 exit 4 (partial): plugin capture request lifecycle on the real NPU scheduler (`scripts/npu/smoke_plugin_capture.py`) | passed up to the transport boundary: `SpecForge spec capture enabled (host publication, writer=True)` at startup; the spec_capture request was held, the observer found both `aux` and `last_hidden` rows (no missing-artifact error), and the sink's lazy Mooncake import failure surfaced as `meta_info["spec_capture"]=[{"sample_id":"s0","error":"No module named 'mooncake'}"]`; a normal request was unaffected |
| N2 entry: capture → Mooncake → client read-back (`scripts/npu/verify_mooncake_e2e.py`, public mooncake-transfer-engine 0.3.13.post1 wheel, local `mooncake_master -enable_http_metadata_server=true`, host publication + TCP) | passed (`VERIFY_E2E_OK`). The N1 hardened path ran end-to-end for real: device detection picked the NPU, the sink worker bound its NPU context, the Ascend CPU mount (`setup(0,0)` + `allocate_and_mount_segment(size, "tcp", "cpu")`) succeeded against the public wheel, publication returned feature refs (`hidden_states [1,9,7680]`, `target [1,9,2560]`, `input_ids [1,9]`, all bf16/int64), and a separate client process read every object back while the engine was still up: passthrough input_ids byte-exact, aux payload **bit-identical** to the engine's own response rows (rel_err 0.0), last_hidden finite with correct shape. The wheel exposes `allocate_and_mount_segment` and `ReplicateConfig.with_hard_pin` (hard-pin capable build) |
| #1070 final-head review follow-up: nonzero logical card and published-target numerics (`67bb423`; `ASCEND_RT_VISIBLE_DEVICES=3,4`, `base_gpu_id=1`; `verify_mooncake_e2e.py` + test-only `device_trace_sitecustomize.py`) | passed (`VERIFY_E2E_OK`). Trace: `model_runner.device='npu' model_runner.gpu_id=1 resolved=npu:1`, then `sink configured device=npu:1` and `sink bound current_device=1`; host publication + TCP read-back returned exact input_ids and aux (rel 0.0). The **actual Mooncake target**, not another engine's proxy, matched the capture-off post-norm reference on the same logical card for **9/9 rows bit-exactly** (`max_abs=0`, every per-row rel 0.0) |
| N2 trainer side: consume through SpecForge's own `MooncakeFeatureStore` (`scripts/npu/verify_trainer_consume.py`, same engine/master) | passed (`VERIFY_TRAINER_OK`). `get(ref, "cpu")` returns every feature with aux bit-identical to the engine rows; `get(ref, "npu:0")` — the pageable receive → H2D baseline the plan calls the N3 starting point — lands all tensors on the NPU with values intact; `release()` then frees the objects and a later `get()` raises `KeyError` (no use-after-free). Together with the E2E row above this closes the P0 chain on hardware: NPU hidden → async D2H → Mooncake host object → SpecForge data-plane receive → H2D |
| N0 exit 2 (numeric half): captured aux rows vs an HF fp32 CPU reference forward (`scripts/npu/verify_capture_numerics.py`, same prompt, capture-ON engine rows via `return_hidden_states=True` under `return_hidden_states_mode=full`) | passed (`VERIFY_OK`). Empirically established capture convention on this fork: **the aux chunk for requested layer L is that layer's completed *output* residual** (matches HF `hidden_states[L+1]`), not the layer input — chunk rel-err 5e-3..1e-2, cos 0.99994..0.99999 vs fp32 reference, and each neighbor-layer offset is 30-100x worse, so the layer order is unambiguous. Final hidden rows match HF post-norm (`cos=0.999957`; pre-norm neighbor 0.51), and the engine's greedy next token (id 11751) equals the HF top-1, so the no-softcap/no-muP lm_head contract reproduces |
| N0 exit 3: prefill graph lifecycle, plain hidden-state path (`scripts/npu/verify_capture_graph_lifecycle.py`, `cuda_graph_backend_prefill=breakable` + explicit `cuda_graph_bs_prefill=[64,128,512]`, radix cache off so the repeat capture really re-prefills) | passed (`VERIFY_GRAPH_OK`) for the **plain `return_hidden_states` path only** — no `spec_capture` item, so the plugin observer is not exercised; the plugin-chain counterpart is the next row. Boot log shows `Capturing prefill shape` and prefill batches report `npu graph: True`, so a replay really happened. Capture request A followed by an interleaved burst and a repeat A2: A's aux rows match the eager reference (rel 5.7e-3..1.2e-2, cos 0.99994+), and A2 is bit-identical to A — later replays do not corrupt a drained capture. Decode graphs also capture and run on this stack (bs 1..64) |
| N0 numerics at TP2 (`verify_capture_numerics.py ... 2`, cards 3+4, deferred-allreduce capture path) | passed (`VERIFY_OK`): same L+1 output-residual convention (cos 0.99993..0.99998), post-norm final hidden (cos 0.999951), greedy token matches HF; no duplicate-writer or row-mixing symptoms at TP2 |
| N2 review follow-up: prefill graph lifecycle through the **real plugin chain** (`scripts/npu/verify_plugin_graph_lifecycle.py` + a test-only `sitecustomize` shim delaying `put_samples` by 10s; real `spec_capture` custom_params, so observer snapshot → async D2H → sink publication → deferred response → Mooncake read-back is all exercised) | passed (`VERIFY_PLUGIN_GRAPH_OK`). Completion timestamps prove **other requests kept executing (including graph replays) while A's publication was still pending**, and A's read-back data is consistent: published aux is **bit-identical to the eager reference** (rel 0.0 on all 3 chunks) and the isolated repeat capture A2 is bit-identical to A; input_ids byte-exact. Scope note: the shim delays publication only — it does not force a D2H/replay race. A repeat capture *batched together with the burst* shows ordinary bf16 batch-arrangement noise (rel ≤ 1.8e-2, matching the capture numerics envelope) — the deterministic A/A2 comparison requires the same single-request graph path |
| N2 training-step smoke: one-command managed-local stack (`scripts/npu/smoke/qwen3.5-4b-dflash-npu-smoke.yaml`: mooncake master + capture server on physical card 1 with `SPECFORGE_SPEC_CAPTURE=1`, producer, single-rank DFlash trainer on physical card 2; pre-tokenized 8-sample dataset from `scripts/npu/smoke/gen_smoke_dataset.py`; random-init draft from `configs/qwen3.5-4b-dflash.json`) | passed (`EXIT=0`). Server logged `SpecForge spec capture enabled (host publication, writer=True)`; producer published 8/8 refs with no failures; trainer consumed in lease order (per-rank inbox `.consumed_count` = 3, one sample per step) and completed 3 DFlash steps with finite loss (12.69 / 17.13 / 12.44, the expected random-init scale ~ln(vocab)) and live teacher-agreement metrics; control plane closed cleanly (`.closed`, `.consumer_done`, no `.failed`). Functional smoke only — loss quality is not a draft-quality claim |
| N2 offline round-trip (#1064 semantics, plugin → Mooncake → CPU client → offline `.ckpt` files; `scripts/npu/offline_roundtrip.py`, layers `[1,8,15,22,29]` from the draft config) | passed (`VERIFY_OFFLINE_ROUNDTRIP_OK`). 8/8 samples captured through the real plugin path, read back over TCP, persisted as offline-schema files; reloading each file reproduces the Mooncake bytes exactly. Found operational requirement: Mooncake rejects overlapping registered regions (`-600`), so a raw client must `unregister_buffer` after every `get_into` — torch's caching allocator reuses freed addresses |
| N2 offline training from the round-trip files (`scripts/npu/smoke/qwen3.5-4b-dflash-npu-offline.yaml`, local_colocated, single NPU) | passed: 8 steps, finite loss, `.ckpt` schema accepted by the DFlash offline reader as-is |
| Offline route determinism | bit-identical rerun: two full 8-step offline runs produce exactly equal per-step losses |
| Checkpoint resume (offline route) | passed bit-exact: run to step 8 uninterrupted vs `training.resume_from` the step-4 checkpoint of a completed run — resumed steps 5–8 losses equal the uninterrupted run's to the last bit (optimizer/RNG/sample position fully restored; no repeated or corrupted samples) |
| Fixed-data loss equivalence, online route vs offline route | passed **bit-identical at every step** (8/8 steps, rel diff 0.0) once the consumption order is aligned. Two orderings exist upstream and they differ for the same seed: the online producer consumes in `random.Random(training.seed + epoch)` shuffle order, the offline trainer in `torch.randperm(seed + epoch)` (DistributedSampler) order. With the offline file order realigned so both paths consume `[3,4,6,7,2,5,0,1]`, and the features being the capture server's own Mooncake bytes (`scripts/npu/server_capture_to_files.py` replays the managed server's argv and the producer's shuffled batch), every per-step loss matches `0.0000%`. Draft init is also bit-identical across both paths (probe: SHA-256 over all named parameters equal, `81a08985…`; probe reverted after use) |
| Capture numerics envelope (same model/prompts, different capture arrangements) | server (launch_server, batched 8-seq prefill) vs in-process Engine: 4/8 samples bit-identical, the rest rel ≤ 9.8e-3 (bf16 intra-batch arrangement noise) with ≤ 0.02pp loss impact per step; serial vs batched prefill: rel ≈ 1.2e-2, ≤ 0.8% loss impact; `page_size` 64 vs 128: bit-identical. All far below semantic error (a layer permutation would leave loss flat at ~ln(vocab)) |
| N4 mixed-length slow-sink gate (`scripts/npu/measure_publication_memory.py` + `publication_memory_sitecustomize.py`, Qwen3.5-4B, layers `[2,16,29]`, lengths `4096,4096,4096,7,2048,129`, 10 s `put_samples` delay) | passed (`MEASURE_PUBLICATION_MEMORY_OK`). Without a byte budget, two publication batches held **282.5 MiB** in flight despite `MAX_PENDING_BATCHES=2`. With `SGLANG_SPEC_CAPTURE_MAX_PENDING_BYTES=128 MiB`, every capacity rejection reached D2H with **0 payload bytes**, the observed reservation peak was **128,614,400 bytes (122.65625 MiB)**, and all six samples eventually succeeded. Retry gating reduced the worst sample from 30 attempts in the first implementation to 10; it does not eliminate retries |

### N4 publication-byte measurement

The workload captures aux plus last hidden in BF16: 80 MiB for each
4096-token sample, 40 MiB for 2048 tokens, 2.5195 MiB for 129 tokens, and
0.1367 MiB for 7 tokens. The unbudgeted baseline overlapped an 80 MiB batch
with a 202.5 MiB batch. The final implementation reserves requested feature
bytes before row selection/clone and D2H, releases them after publication,
and lets an oversized batch run alone. The 128 MiB run admitted 80 MiB,
80 MiB, 120.15625 MiB, and finally a 2.5 MiB sample overlapping the previous
batch; its 122.65625 MiB peak stayed below the 128 MiB limit. This is payload
admission evidence, not an end-to-end speedup claim. It does not bound
Mooncake-resident objects or attribute the full NPU reserved-memory high-water
mark to capture, because that also includes prefill workspaces.

Platform notes:

- 910B3 (pre-arch35) rejects FP32 GDN `initial_state` in
  `torch.ops.npu.chunk_gated_delta_rule`; run with
  `mamba_ssm_dtype=bfloat16` until the kernel accepts or downcasts FP32
  states. This is a model-kernel constraint, independent of capture.
- The NPU smoke/verification scripts and dataset referenced in this log live
  in `scripts/npu/` on the full archive branch `npu/spec-capture-ascend`
  (curnane-lab/SpecForge); they are intentionally not part of the PR branch
  `pr/npu/spec-capture-ascend`.
- Graph paths (decode graph capture, prefill graph) import the CANN `acl`
  python module — source `/usr/local/Ascend/ascend-toolkit/set_env.sh` first,
  and prepend the fork's `python/` to the resulting `PYTHONPATH` (the script
  appends its own). Eager runs do not need this.
- The prefill graph stays off unless `cuda_graph_bs_prefill` is set
  explicitly; `cuda_graph_backend_prefill=breakable` alone resolves to
  "capture size is not set" and silently falls back to eager prefill.
- An HF eager reference forward does not run on the NPU here (torch_npu
  conv1d → GE/TBE init fails without the full toolkit runtime); the numerics
  script uses a CPU fp32 HF reference instead, which is also the cleaner
  reference for measuring engine bf16 divergence.
- Mooncake on this host: the public aarch64 wheel
  (`mooncake-transfer-engine 0.3.13.post1`, via a PyPI mirror) works for the
  TCP host path, but its native libs link `libcuda.so.1`/`libcudart.so.12`
  (66 cudart refs carry the `@libcudart.so.12` version tag) and
  `libibverbs.so.1`. `libibverbs1` installs from apt; the CUDA deps were
  satisfied with local stub libraries under `cuda_stub_lib/` (cudart stubs
  need a matching version script). This is a bring-up vehicle for the TCP
  host path only — production NPU deployments want a Mooncake build that
  links the Ascend runtime instead, and N5 device-direct needs real Ascend
  transport support, not stubs.
- Mooncake object lifetime: mounted segments live in the publisher process —
  read-backs must happen while the capture server is up (reading after
  `engine.shutdown()` fails with TRANSFER_FAIL -800; the status is also
  listed as retryable in the data plane).

## Branch series (archive map)

curnane-lab/sglang:

| Branch | Content | Plan item |
| --- | --- | --- |
| `npu/capture-forward-observer` | fork head `88a0368750` + #43450 cherry-pick `b462f44827` | N0(b) runtime baseline |

curnane-lab/SpecForge (stacked, bottom-up):

| Branch | Base | Content | Plan item |
| --- | --- | --- | --- |
| `npu-legacy-capture-freeze` | `55c6711` (hspec-debug) | frozen legacy offline/file-protocol route | §2.1 rollback |
| `npu/capture-baseline` | upstream/main `af1c3f3` | this document: SHA inventory + verification log | N0 record |
| `npu/spec-capture-ascend` | `npu/capture-baseline` | full work history incl. `scripts/npu/` smoke assets (archive) | N0+N1+N2 |
| `pr/npu/spec-capture-ascend` | upstream/main `af1c3f3` | PR-ready: N1 plugin hardening + unit tests + existing Ascend guide/plugin README updates, no smoke assets | N1+N2 PR |

N3/N4/N5 are measurement-gated; no placeholder branches until the N2
profiling data picks them.

## Remaining N0 exit items (need real NPU runs)

1. ~~Target model loads and runs a plain forward~~ — done for Qwen3.5-4B
   (see the verification log). Still open: any other target (e.g. 9B/35B
   variants) follows the same script.
2. ~~Aux-capture boot + full-prefill request~~ — done for Qwen3.5-4B,
   including the numeric check against an HF fp32 CPU reference (layer order,
   post-norm last hidden, head contract; see the verification log).
3. ~~Real prefill graph lifecycle~~ — done for Qwen3.5-4B
   (`scripts/npu/verify_capture_graph_lifecycle.py`): replay proven in the
   boot/batch logs, capture A survives later replays, repeat capture A2 is
   bit-identical. No S1 core defect surfaced on this stack.
4. ~~Plugin capture request lifecycle on the real stack
   (`scripts/npu/smoke_plugin_capture.py`)~~ — done, and extended past the
   transport boundary: `verify_mooncake_e2e.py` proves capture → Mooncake →
   client read-back on this host (bit-exact aux), and
   `verify_trainer_consume.py` proves the trainer-side
   `MooncakeFeatureStore` receive + H2D baseline + release lifecycle. Still
   open: a real training-step smoke needs draft weights (out of scope for
   N0); the migration doc and offline round-trip remain N2 work.
5. ~~Training-step smoke~~ — done as N2 work on this branch: the managed-local
   stack ran end-to-end (plugin capture server + producer + single-rank DFlash
   trainer on real NPU cards) with a random-init draft; 3 steps, finite loss,
   clean lifecycle (see the verification log and
   `docs/ascend_capture_migration.md`).
6. ~~Fixed-data loss/gradient comparison and offline round-trip~~ — done (N2):
   the offline (#1064-semantics) round-trip writes `.ckpt` features the DFlash
   offline reader consumes directly; offline training is deterministic
   (bit-identical rerun), checkpoint resume reproduces the uninterrupted
   trajectory bit-exactly, and with consumption order aligned the online and
   offline routes produce **bit-identical per-step losses** on the server's own
   captured bytes. Note the upstream asymmetry this surfaced: online consumes
   in `random.Random(seed + epoch)` order, offline in `torch.randperm(seed +
   epoch)` order — same seed, different order.

Still open (beyond N2 scope): EAGLE3/DSPARK route validation (DFlash only
here), a real draft quality run, TP2 end-to-end training, and the profiling
pass that decides N3/N4/N5.

## Rollback

- `npu-legacy-capture-freeze` (this repo, `55c6711`): hspec offline
  in-process capture + windowed file protocol. Do not follow upstream
  deletions (#1064/#1065) into the NPU maintenance branch until the N2 gate
  passes on the native-plugin route.
