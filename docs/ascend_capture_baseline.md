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

1. Target model loads and runs a plain (no-capture) forward / minimal
   generation on the fork: tiny model first, then the actual Qwen3.5 / GDN
   hybrid target.
2. Aux + last-hidden minimal forward on the real target: layer order, full
   token rows, dtype/shape, post-norm semantics, head-scaling contract.
3. Real prefill graph lifecycle: prove a replay happens, and a captured
   batch A survives a later replay B (S1 territory if it fails).
4. Launch the capture server with the plugin installed and run one
   capture → Mooncake → read-back → training-step smoke (this is the N2
   entry point).

## Rollback

- `npu-legacy-capture-freeze` (this repo, `55c6711`): hspec offline
  in-process capture + windowed file protocol. Do not follow upstream
  deletions (#1064/#1065) into the NPU maintenance branch until the N2 gate
  passes on the native-plugin route.
