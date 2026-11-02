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
| Plugin unit + sink suites (`test_sglang_capture_plugin.py`, `test_sglang_capture_plugin_sink.py`, includes the N1 Ascend changes) under the NPU venv against the fork | 51 passed, 6 skipped (CUDA-only) |
| N0 exit 1: plain forward / minimal generation, Qwen3.5-4B (qwen3_5, GDN hybrid VL) on device 3, `disable_cuda_graph`, `mamba_ssm_dtype=bfloat16` | passed (`scripts/npu/smoke_plain_forward.py`) |
| N0 exit 2: boot with `enable_return_aux_hidden_states` + `speculative_algorithm=DFLASH` + `return_aux_hidden_state_layer_ids=[2,16,29]` + `return_hidden_states_mode=full` + `chunked_prefill_size=-1`; full-prefill `max_new_tokens=0` request | passed; `aux_hidden_state_capture` resolved to `dflash`, draft algorithm reset as designed |
| N0 exit 4 (partial): plugin capture request lifecycle on the real NPU scheduler (`scripts/npu/smoke_plugin_capture.py`) | passed up to the transport boundary: `SpecForge spec capture enabled (host publication, writer=True)` at startup; the spec_capture request was held, the observer found both `aux` and `last_hidden` rows (no missing-artifact error), and the sink's lazy Mooncake import failure surfaced as `meta_info["spec_capture"]=[{"sample_id":"s0","error":"No module named 'mooncake'}"]`; a normal request was unaffected |
| N2 entry: capture → Mooncake → client read-back (`scripts/npu/verify_mooncake_e2e.py`, public mooncake-transfer-engine 0.3.13.post1 wheel, local `mooncake_master -enable_http_metadata_server=true`, host publication + TCP) | passed (`VERIFY_E2E_OK`). The N1 hardened path ran end-to-end for real: device detection picked the NPU, the sink worker bound its NPU context, the Ascend CPU mount (`setup(0,0)` + `allocate_and_mount_segment(size, "tcp", "cpu")`) succeeded against the public wheel, publication returned feature refs (`hidden_states [1,9,7680]`, `target [1,9,2560]`, `input_ids [1,9]`, all bf16/int64), and a separate client process read every object back while the engine was still up: passthrough input_ids byte-exact, aux payload **bit-identical** to the engine's own response rows (rel_err 0.0), last_hidden finite with correct shape. The wheel exposes `allocate_and_mount_segment` and `ReplicateConfig.with_hard_pin` (hard-pin capable build) |
| N0 exit 2 (numeric half): captured aux rows vs an HF fp32 CPU reference forward (`scripts/npu/verify_capture_numerics.py`, same prompt, capture-ON engine rows via `return_hidden_states=True` under `return_hidden_states_mode=full`) | passed (`VERIFY_OK`). Empirically established capture convention on this fork: **the aux chunk for requested layer L is that layer's completed *output* residual** (matches HF `hidden_states[L+1]`), not the layer input — chunk rel-err 5e-3..1e-2, cos 0.99994..0.99999 vs fp32 reference, and each neighbor-layer offset is 30-100x worse, so the layer order is unambiguous. Final hidden rows match HF post-norm (`cos=0.999957`; pre-norm neighbor 0.51), and the engine's greedy next token (id 11751) equals the HF top-1, so the no-softcap/no-muP lm_head contract reproduces |
| N0 exit 3: prefill graph lifecycle (`scripts/npu/verify_capture_graph_lifecycle.py`, `cuda_graph_backend_prefill=breakable` + explicit `cuda_graph_bs_prefill=[64,128,512]`, radix cache off so the repeat capture really re-prefills) | passed (`VERIFY_GRAPH_OK`). Boot log shows `Capturing prefill shape` and prefill batches report `npu graph: True`, so a replay really happened. Capture request A followed by an interleaved burst and a repeat A2: A's aux rows match the eager reference (rel 5.7e-3..1.2e-2, cos 0.99994+), and A2 is bit-identical to A — later replays do not corrupt a drained capture. Decode graphs also capture and run on this stack (bs 1..64) |
| N0 numerics at TP2 (`verify_capture_numerics.py ... 2`, cards 3+4, deferred-allreduce capture path) | passed (`VERIFY_OK`): same L+1 output-residual convention (cos 0.99993..0.99998), post-norm final hidden (cos 0.999951), greedy token matches HF; no duplicate-writer or row-mixing symptoms at TP2 |

Platform notes:

- 910B3 (pre-arch35) rejects FP32 GDN `initial_state` in
  `torch.ops.npu.chunk_gated_delta_rule`; run with
  `mamba_ssm_dtype=bfloat16` until the kernel accepts or downcasts FP32
  states. This is a model-kernel constraint, independent of capture.
- The NPU smoke scripts live in `scripts/npu/` on this branch.
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
| `npu/spec-capture-ascend` | `npu/capture-baseline` | plugin Ascend host-publication hardening | N1 |
| `npu/capture-gate` (planned) | series tip | real-hardware gate + migration docs | N2, after N0 exit items pass |

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
   client read-back on this host (bit-exact aux). Still open: the
   trainer-side consumer (`MooncakeFeatureStore` receive + an actual
   training-step smoke) — that is the remaining N2 gate.

## Rollback

- `npu-legacy-capture-freeze` (this repo, `55c6711`): hspec offline
  in-process capture + windowed file protocol. Do not follow upstream
  deletions (#1064/#1065) into the NPU maintenance branch until the N2 gate
  passes on the native-plugin route.
