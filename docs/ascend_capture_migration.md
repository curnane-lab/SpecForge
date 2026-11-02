# Ascend NPU capture migration guide — legacy patch route → native plugin route

Date: 2026-11-02 (Beijing). Companion to `ascend_capture_baseline.md`; that
document owns the SHA inventory and the N0 verification log, this one owns the
migration procedure and the rollback contract.

Audience: operators moving an existing NPU capture deployment (SGLang 0.5.18
source patch + `spec-capture-ascend-mount.patch`, or the legacy offline
in-process route) onto the native route: SGLang with the upstream capture
extension points plus the `specforge-sglang-capture` plugin.

Status: every step of this guide, including the section "Migration procedure"
step 5 training smoke, has been executed on the validated combination below
(see `ascend_capture_baseline.md` for the full verification log).

## Route mapping

| Concern | Legacy route | Native route |
| --- | --- | --- |
| SGLang server | 0.5.18 + `patches/sglang/v0.5.18/spec-capture.patch` + `spec-capture-ascend-mount.patch`, `--enable-spec-capture` | fork `npu/capture-forward-observer` (or any build with #43449 + #43450), plugin enabled by `SPECFORGE_SPEC_CAPTURE=1` |
| Aux layer selection | `--spec-capture-aux-layer-ids` (patch flag) | `--enable-return-aux-hidden-states --return-aux-hidden-state-layer-ids <ids> --return-hidden-states-mode full` (launcher emits these itself in plugin mode) |
| Ascend Mooncake registration | source patch on the server | built into the plugin sink (`setup(0,0)` + `allocate_and_mount_segment(size, protocol, "cpu")`); no patch is applied |
| Producer | in-process capture / file protocol | `specforge train --role producer`, control plane only; tensors never enter the producer |
| Offline capture | `offline_capture/sglang_backend` in-process ModelRunner (deleted upstream by #1064) | plugin → Mooncake → CPU client → file (`#1064` route); the legacy backend stays available on the freeze branch until N2 closes |
| Spec transport to server | top-level `spec_capture` request field | `sampling_params.custom_params`; `deployment.disaggregated.server_capture: plugin` tells the producer which wire to use |

The Ascend mount semantics (host segment, `location:`-free registration) moved
into the plugin verbatim. The old mount patch is **not** re-applied on the
native route; keep the two patch files only for legacy 0.5.18 deployments
pinned to the freeze branch.

## Validated combination (this host, 8x 910B3)

Record your own combination the same way before cutting over.

| Component | Value |
| --- | --- |
| SpecForge | `npu/spec-capture-ascend` @ `8d096e0` (worktree `specforge-npu-capture`) |
| SGLang | fork `npu/capture-forward-observer` @ `b462f44827` (`PYTHONPATH=…/sglang-gdn-compact-npu/python`) |
| Python env | `venv-gdn-compact`: torch 2.10.0+cpu, torch_npu 2.10.0, transformers 5.12.1 |
| Mooncake | public wheel `mooncake-transfer-engine 0.3.13.post1` + `libibverbs1`; `mooncake_master -enable_http_metadata_server=true` |
| CANN / toolkit | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` (CANN 9.0.0) |
| Publication / transport | host publication + TCP |
| Target model | `Qwen3.5-4B` (qwen3_5 GDN hybrid VL), bf16 |

### Environment pitfalls specific to this stack

- **`PYTHONPATH` must keep the Ascend toolkit paths.** `set_env.sh` adds the
  `acl` module path to `PYTHONPATH`; prepend the fork and SpecForge paths
  (`export PYTHONPATH=<fork>:<specforge>:$PYTHONPATH`), never overwrite, or
  decode-graph capture dies later with `ModuleNotFoundError: No module named
  'acl'` deep inside the NPU graph runner.
- **`SGLANG_MAMBA_SSM_DTYPE=bfloat16`** is required on 910B3 for GDN hybrid
  models (the NPU `chunk_gated_delta_rule` rejects an FP32 initial state). In
  managed-local configs pass it through `capture_servers[].env`.
- The mooncake wheel links `libcuda`/`libcudart`; on CUDA-less hosts preload
  the stub via `LD_LIBRARY_PATH=/data1/f00538480/cuda_stub_lib`. The stub is a
  bring-up shim for the TCP host path only — it reports zero CUDA devices and
  must not be used to judge any device-direct (N5) behavior.
- `managed_local` requires `mooncake_master` on `PATH` (it lives in the venv's
  `bin/`).
- `managed_local` refuses a non-empty `control_dir`; delete the previous
  run's output directory before relaunching.
- Raw Mooncake clients must pair every `register_buffer` with
  `unregister_buffer` after the transfer completes: the engine rejects
  overlapping registered regions (`-600`), and torch's caching allocator
  reuses freed buffer addresses, so a leaked registration breaks later reads
  on unrelated buffers. (`SpecForge`'s own data plane already does this; the
  pitfall bites hand-rolled readers.)

## Migration procedure

1. **Freeze the legacy deployment.** Pin the running SpecForge + SGLang
   commits and environment (branch `npu-legacy-capture-freeze` = `55c6711` is
   the reference). Do not uninstall or overwrite that environment; the native
   route runs in a separate, clean server environment.
2. **Stand up the native baseline** per `ascend_capture_baseline.md`:
   fork SGLang on `PYTHONPATH`, plugin installed editable
   (`pip install -e plugins/sglang-spec-capture --no-deps`), Mooncake master
   reachable.
3. **Smoke the server alone** with `scripts/npu/smoke_plugin_capture.py`, then
   the transport with `scripts/npu/verify_mooncake_e2e.py`, then the trainer
   read path with `scripts/npu/verify_trainer_consume.py`.
4. **Switch the SpecForge config**: add `server_capture: plugin` under
   `deployment.disaggregated` (or export `DISAGG_SERVER_CAPTURE=plugin`). For
   `managed_local`, the launcher then sets `SPECFORGE_SPEC_CAPTURE=1` and the
   capture flags on the server itself; remove any legacy
   `--enable-spec-capture` / `SGLANG_SPEC_CAPTURE_*` knobs from custom server
   args. Keep `mooncake_protocol: tcp` (or leave unset — the launcher resolves
   TCP on Ascend hosts).
5. **Run the training smoke** (`scripts/npu/smoke/qwen3.5-4b-dflash-npu-smoke.yaml`
   is the reference: one capture server, one trainer rank, 3 steps) and the
   fixed-data loss/gradient comparison before any real run. Expected smoke
   behavior, as observed on the validated combination: the server logs
   `SpecForge spec capture enabled (host publication, writer=True)`, the
   producer reports `produced=8 prompts_failed=0`, the trainer's per-rank
   inbox `.consumed_count` advances once per step, losses are finite (a
   random-init draft sits near `ln(vocab)` ≈ 12.4), and the control dir ends
   with `.closed` + `.consumer_done` and no `.failed`.
6. **Cut over** by changing the deployment entrypoint, not by mutating the old
   environment in place.

## Verification gate before declaring the migration done

All of the following must pass on the same pinned combination:

- `verify_capture_numerics.py` at TP1 and TP2 (aux = layer-L output residual,
  cos ≥ 0.9999 vs the HF fp32 reference; final hidden post-norm).
- `verify_capture_graph_lifecycle.py` with real prefill replays (capture A,
  interleave replay B, delayed read of A unchanged; repeat capture
  bit-identical).
- `verify_mooncake_e2e.py` and `verify_trainer_consume.py` (bit-identical aux
  through Mooncake; release → `KeyError`).
- Training smoke: ≥3 DFlash steps, finite loss, control-plane artifacts
  (`.consumed_count`, no `.failed`) consistent. On the validated combination
  this passes end-to-end (`scripts/npu/smoke/qwen3.5-4b-dflash-npu-smoke.yaml`).
- Offline round-trip (#1064 semantics): capture through the plugin server,
  read back over Mooncake, persist `.ckpt` files, train from them
  (`scripts/npu/offline_roundtrip.py` + `qwen3.5-4b-dflash-npu-offline.yaml`).
  On the validated combination this passes, the offline route is
  bit-deterministic across reruns, checkpoint resume reproduces the
  uninterrupted trajectory bit-exactly, and the online and offline routes give
  bit-identical per-step losses on identical bytes in identical order.
- Same-sample comparison between the legacy route's output and the native
  route's capture (ordered layers, token rows, shape/dtype, training-input
  semantics).

**Ordering asymmetry to know before comparing runs.** The online producer
consumes prompts in `random.Random(training.seed + epoch)` shuffle order; the
offline trainer consumes feature files in `torch.randperm(seed + epoch)`
(DistributedSampler) order. The same seed therefore yields *different*
per-step sample orders across the two routes; per-step loss comparisons must
realign order first (or compare loss multisets). This is an upstream behavior
note, not an NPU defect.

## Rollback

The freeze branch keeps the legacy route runnable. If any gate above fails,
stay on the freeze branch, file the failure with its server/consumer logs, and
do not partially upgrade (new plugin on a patched 0.5.18 server is not a
supported mix).

## Known limits of the native route today

- Host publication + TCP only. `GPU_PUT=1` / NVLink publication is rejected at
  install time on NPU (device-direct transport is N5 and needs an Ascend
  Mooncake transport probe first — the CUDA stub must not be used to infer
  device-direct capability).
- Trainer receive is pageable host memory + explicit H2D (N3 will add pinned
  receive/prefetch only if profiling shows the wait matters).
- No pending-bytes budget yet (N4); keep `max_length` and concurrency moderate
  on long-context workloads.
- PP=1, no PD disaggregation, full prefill (`--chunked-prefill-size -1`) only.
