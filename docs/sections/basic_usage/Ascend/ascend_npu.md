# Ascend NPU Tutorial

This is an end-to-end tutorial for running SpecForge on Ascend NPU hosts. It
walks through **installation → data preparation → online disaggregated training
with external services → the managed-local full stack → split multi-node
roles**, using Qwen3.5-4B DFlash as the running example. Validated on a 16-card
A3 (64GB) host.

---

## 1. Installation

You need an Ascend host with the driver, CANN, and a `torch_npu`-enabled
PyTorch already installed, plus an NPU-compatible SGLang `main` build with
the capture extension APIs described below. Then install
SpecForge without touching that stack:

```bash
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge
python -m pip install -e . --no-deps
```

`--no-deps` keeps pip from pulling CUDA wheels over the working NPU
torch/sglang. If a later step reports a missing lightweight dependency, install
just that package, also with `--no-deps`.

`liger-kernel` is not needed: on NPU, DFlash drafts keep the native
RMSNorm/SwiGLU (`model.use_liger_kernel` resolves to off). Liger's Ascend
backend is not validated in SpecForge; an explicit `model.use_liger_kernel:
true` tries it and logs a warning.

### Install the SGLang capture plugin (online and offline preparation)

Use an Ascend-compatible SGLang build with the generic extension APIs required
by [the plugin guide](../../../../plugins/sglang-spec-capture/README.md):

```bash
python -m pip install --no-deps plugins/sglang-spec-capture
export SPECFORGE_SPEC_CAPTURE=1
```

Release 0.5.23 lacks the required capture APIs. The plugin handles Ascend's CPU
segment mount directly. Offline training from
existing feature files needs no capture service.

### Device visibility on Ascend

Ascend selects devices through `ASCEND_RT_VISIBLE_DEVICES`, and the driver
rejects an *empty* value — hiding devices from a process means **unsetting**
the variable, not setting it to `""`. SpecForge handles this internally: device
ordinals from the config are injected through `CUDA_VISIBLE_DEVICES` on CUDA
hosts and `ASCEND_RT_VISIBLE_DEVICES` on Ascend hosts, and a "hide all devices"
role unsets the variable. You only need to export the visible set once for the
supervisor:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

Two more environment settings are recommended for long online runs:

```bash
export HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

### Attention backends on Ascend

Use `sdpa` for the **trainer**. Sequence parallelism (`sp_ulysses_size` /
`sp_ring_size` > 1, the `usp` backend) requires `yunchang`, whose import probes
the CUDA device and crashes NPU-only torch builds, so USP is CUDA-only for now.
SpecForge imports `yunchang` lazily only when SP sizes exceed 1, so the default
SP=1 path never touches it.

For the **capture server**, the recipes set `model.sglang_attention_backend:
ascend`. If a config leaves it at the flashinfer default, the launcher falls
back to `ascend` automatically on Ascend hosts (flashinfer does not exist
there).

#### DFlash attention in anchor groups

DFlash and DFlash2 training can bound SDPA attention intermediates with
`training.attention_chunk_blocks`. Use a nonzero tile size when attention memory
limits the workload, and measure training-step time and peak allocated memory.
The 910B3 measurements below provide initial tile choices for 8K and 32K context.

```yaml
training:
  attention_backend: sdpa
  attention_chunk_blocks: 64
```

The setting works with the existing DFlash recipes and the canonical
`draft`/`initialization` training entry. Canonical runs also accept the same
setting in `training.objective_options`. Conflicting values are rejected.
The default is zero. A nonzero value requires SDPA and a DFlash or DFlash2 draft.
Resume records a nonzero tile size because changing attention call shapes can
change floating-point rounding. A zero tile size uses the historical dense
resume contract.

Each layer projects context and draft K/V once. A tile attends to shared context
and its own group of draft blocks, preserving block isolation, layer causality
and sliding windows. By default each tile uses the full context. Non-reentrant
checkpointing regenerates tile masks and concatenated K/V during backward,
bounding these intermediates throughout the training step. It composes with
draft-layer activation checkpointing. Dropout replay preserves each tile's RNG
state. Numerical comparisons against dense SDPA use tolerances.

`training.attention_trim_context: true` additionally limits each tile to the
largest context prefix that its valid anchors can access. This option requires
a nonzero `attention_chunk_blocks`. Prefix limits include all samples in the
batch and are computed once when anchors are sampled, using one device-to-host
transfer of the reduced bounds. All layers and backward replay reuse these
Python slice bounds. The default is false. A full resume records this setting.

CPU and NPU output/gradient checks cover the prefix option. The reported
attention and complete-training measurements below show its workload-dependent
memory and throughput tradeoffs. Add `--trim-context` to the benchmark command
to include prefix-bound construction and its synchronization in timing.

Use this attention-only forward/backward benchmark to compare tile sizes.
It includes mask construction and recomputation. Full training throughput and
serving acceptance require separate measurements.

```bash
python -m specforge.benchmarks.benchmark_dflash_attention \
  --device npu --dtype bfloat16 --batch-size 2 \
  --context-length 8192 --num-anchors 512 --block-size 16 \
  --heads 32 --kv-heads 8 --head-dim 128 --chunk-blocks 16 32 64
```

##### Reported 910B3 validation

A validation run on Ascend 910B3 used the unmodified implementation at
[`f321b7a`](https://github.com/curnane-lab/SpecForge/commit/f321b7ac4eaddc27c4c7f83a910a48ed4129ff3d).
The supplied results report 89 tests passed and five skipped. All 48 BF16
causal/window/tile/GQA combinations passed with padding and invalid anchors.
Maximum absolute differences were zero for outputs and dQ, and 0.015625 for
dK/dV. Legacy and canonical small-model forward/backward checks had equal losses
and finite gradients, including samples with no valid anchors. The largest
parameter-gradient differences were 1.22e-4 and 3.91e-3, respectively.

The benchmark used batch size 2, 512 anchors, block size 16, 32 query heads,
eight KV heads, head dimension 128 and BF16. Times include attention forward,
backward, mask construction and tile recomputation, using two warmups and the
median of five measurements. Peak allocation comes from
`torch.npu.max_memory_allocated` and includes resident Q/K/V and gradients.

| Context | Tile blocks | Forward/backward ms | Latency increase | Peak allocated GiB | Peak reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K | 0 | 115.83 | baseline | 2.145 | baseline |
| 8K | 16 | 229.34 | 98.0% | 1.384 | 35.4% |
| 8K | 32 | 147.18 | 27.1% | 1.424 | 33.6% |
| 8K | 64 | 122.54 | 5.8% | 1.496 | 30.2% |
| 8K | 128 | 121.59 | 5.0% | 1.649 | 23.1% |
| 8K | 256 | 133.81 | 15.5% | 1.840 | 14.2% |
| 32K | 0 | 293.66 | baseline | 3.645 | baseline |
| 32K | 16 | 628.49 | 114.0% | 2.521 | 30.8% |
| 32K | 32 | 460.71 | 56.9% | 2.572 | 29.4% |
| 32K | 64 | 389.11 | 32.5% | 2.668 | 26.8% |
| 32K | 128 | 376.13 | 28.1% | 2.867 | 21.3% |

These measurements establish a memory/latency tradeoff on 910B3. At 8K, a tile
of 64 saved 30.2% of peak allocation with a 5.8% latency increase. At 32K, a tile
of 32 approached the smallest measured allocation while completing faster than
a tile of 16. The fastest measured 32K tile still increased latency by 28.1%.
Keep the default zero when throughput is the objective and the dense run fits.

##### Reported prefix-trimming attention validation

A subsequent 910B3 run of
[`bd7a7ac`](https://github.com/curnane-lab/SpecForge/commit/bd7a7aca66d7aa28b074186bce11584e0172bcc8)
reported 16 tests passed and one CUDA test skipped. All 96 NPU BF16 output and
Q/K/V-gradient comparisons passed across trim on/off, causal/window, tile and
GQA combinations, including padding and invalid anchors.

For the same uniform-anchor 32K attention workload, repeated measurements
included prefix-bound D2H and synchronization. The following table uses the
retest measurements. Allocations include resident Q/K/V and gradients.

| Configuration | Forward/backward ms | Throughput relative to dense | Peak allocated GiB |
| --- | ---: | ---: | ---: |
| Dense | 293.35 | 1.000 | 3.645 |
| 64 blocks, trim enabled | 265.54 | 1.105 | 2.535 |
| 128 blocks, trim enabled | 275.60 | 1.064 | 2.586 |

##### Reported GovReport complete-training validation

A single-device validation of the same implementation used Ascend 910B3,
CANN 9.1.0, PyTorch 2.10, torch_npu 2.10.post6 and Transformers 5.19. It trained
a six-layer DFlash draft with five sliding-attention layers, one full-attention
layer and 634,425,856 trainable parameters. Fixed teacher features came from a
local Qwen3.5-4B target using layers 1, 5, 9, 13, 17, 21, 25 and 29.

The workload used two actual GovReport report/summary pairs per context length,
with long prompts and suffix-summary supervision. Both arms used independent
models initialized to the same weights and FP32-master AdamW. Batch size was 2,
gradient accumulation was 4, scheduler total steps was 10,000 and warmup ratio
was 0.04. Ten optimizer steps warmed up each arm before 50 measured optimizer
steps. The 240 microbatches per arm and context had matching anchor hashes.

Single-step comparisons covered complete forward, backward and optimizer
updates. Gradient relative L2 differences were 0.006136 at 8K and 0.006158 at
32K, with gradient cosine 0.999981 at both lengths. Updated-parameter cosines
exceeded 0.9999997. Grad norms were finite and close across arms.

| Context | Dense median step s | Trim median step s | Trim throughput relative to dense | Dense peak GiB | Trim peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K | 6.720 | 7.616 | 0.882 | 43.36 | 38.67 |
| 32K | 11.799 | 17.476 | 0.675 | 54.63 | 44.33 |

Here trim used 64 anchor blocks per tile. The complete training loop saved
4.69 GiB (10.8%) at 8K and 10.30 GiB (18.9%) at 32K, with step-latency increases
of 13.3% and 48.1%. Effective supervised positions per second changed from
9,097 to 8,025 at 8K and from 5,170 to 3,491 at 32K. All peaks above are
allocator peaks.

The reported per-group prefix-length fractions had P50/P90 values of
0.9479/0.9994 at 8K and 0.98697/0.99988 at 32K. Every sample had 512 valid
anchors. Suffix-concentrated supervision left little prefix length to remove.
On this workload the complete-training benefit is reduced memory consumption,
with a throughput regression at both lengths.

Loss and grad-norm traces showed finite, consistent short-term progress. The
two examples were repeatedly trained, so these traces establish short-term
stability. Final quality and serving acceptance require held-out evaluation.
The uniform-anchor attention measurements and complete-training measurements
describe distinct workloads.

The supplied artifacts were reported under `/data/dflash-real-validation/`,
including `single-step-8192.json`, `single-step-32768.json`,
`ab-formal-8k-comparison.json`, `ab-formal-32k-comparison.json`, per-step
`steps.jsonl`, `environment.json`, `commands.sh`, scripts and fixed teacher
features. Multi-device training, actual checkpoint/resume comparison, profiler
attribution, whole-device HBM and maximum context measurements remain pending.
The following 64K run additionally tested microbatch capacity with an explicitly
checkpointed recipe.

##### Reported 64K matched-effective-batch validation

A further 910B3 validation of implementation `bd7a7ac` used real 64K suffix
supervision and explicitly enabled layer checkpointing in both arms. The
matched effective batch was four samples. Dense used microbatch 2 and two
accumulation steps, while trim used microbatch 4 and one accumulation step.
One optimizer step warmed up each arm before 50 measured optimizer steps.

| Metric | Dense B2, accumulation 2 | Trim B4, accumulation 1 |
| --- | ---: | ---: |
| Median optimizer step s | 11.914 | 19.960 |
| P90 optimizer step s | 11.951 | 20.003 |
| Samples/s | 0.3357 | 0.2004 |
| Effective supervised positions/s | 2559.35 | 1527.30 |
| Peak allocated GiB | 33.72 | 47.68 |

Trim increased median step latency by 67.5% and delivered approximately 0.597
times the dense throughput. Both losses and grad norms remained finite. Each
of the two fixed samples was processed 100 times during measurement. The mean
supervised positions per optimizer step differed by approximately 0.005%.
An additional 50-step dense B3 run reached 0.3348 samples/s.

Capacity probes used accumulation 1 with layer checkpointing enabled.

| Configuration | Largest measured microbatch | Next tested microbatch |
| --- | ---: | --- |
| Dense | 3 | B4 ran out of memory on optimizer step 2 |
| Trim | 4 | B5 ran out of memory on optimizer step 2 |

Trim B4 with accumulation 2 or 3 also ran out of memory on optimizer step 2.
These measured capacities therefore depend on the recorded accumulation and
checkpoint settings. The earlier dense 64K B2 OOM observation used a script
with layer checkpointing disabled. The checkpoint-enabled baseline above
completed the formal comparison.

At the same microbatch 3, two-step probes measured 44.16 GiB for dense and
38.13 GiB for trim, a 13.6% reduction. The formal effective-batch comparison
uses different microbatch sizes and reached the allocations reported in its
table. On this workload trim enabled a larger microbatch while its effective
training throughput remained below dense.

All memory figures in this section are allocator peaks. Fixed-sample training
supports capacity and runtime comparisons; held-out quality and serving
acceptance need their own evaluation. Supplied reproduction materials include
`dflash-real-validation/summary-64k-batch-capacity.json`,
`dflash-real-validation/capacity-64k-formal-comparison.json`, and
`dflash-real-validation/commands.sh` starting at line 61.

---

## 2. Data preparation

Data preparation is platform independent. From the repository root:

```bash
python scripts/prepare_data.py --dataset sharegpt
```

This writes `./cache/dataset/sharegpt_train.jsonl`. For custom datasets and
target-model regeneration, see the [Data Preparation](../data_preparation.md)
guide.

---

## 3. Online training (external capture server)

Online training is always **disaggregated**: a producer drives prompts through
a SGLang capture server, features stream through Mooncake, and a
consumer trains the draft. The checked-in
[`qwen3.5-4b-dflash-npu.yaml`](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-npu.yaml)
recipe targets an externally started capture server.

### Step 1: Start Mooncake and the capture server

```bash
mooncake_master --enable_http_metadata_server=true \
  --rpc_port=35551 --http_metadata_server_port=35880 \
  --metrics_port=35903 --enable_metric_reporting=false &
```

Start the plugin-enabled capture server on NPU 0. The `--return-aux-hidden-state-layer-ids`
must match the draft's `target_layer_ids` — for DFlash these come from
`configs/qwen3.5-4b-dflash.json`: `1 8 15 22 29`. A mismatch produces zero
features with no error:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 \
MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata \
MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551 \
MOONCAKE_PROTOCOL=tcp \
MOONCAKE_GLOBAL_SEGMENT_SIZE=$((32<<30)) \
SPECFORGE_SPEC_CAPTURE=1 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-4B \
  --trust-remote-code \
  --skip-tokenizer-init \
  --tp-size 1 \
  --mem-fraction-static 0.8 \
  --attention-backend ascend \
  --enable-return-aux-hidden-states --return-hidden-states-mode full --speculative-algorithm DFLASH \
  --return-aux-hidden-state-layer-ids 1 8 15 22 29 \
  --host 127.0.0.1 --port 30000 &
```

Wait for `curl --fail http://127.0.0.1:30000/health` to return 200.

### Step 2: Launch training

On the remaining NPUs:

```bash
ASCEND_RT_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 \
specforge train -c examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-npu.yaml
```

Before rerunning, clear stale control state:
`rm -rf outputs/qwen3.5-4b-dflash-npu-online`.

### Success criteria

- Producer log ends with `prompts_failed=0`.
- Consumer prints `step N: {...loss..., acc...}` lines and no
  `could not drain` error at teardown.
- If the capture producer dies with `ACL_ERROR_RT_CONTEXT_NULL` (107002), the
  installed SpecForge predates the NPU transport bind — upgrade past #722.

### Training DFlash 2 drafts

DFlash 2 reuses the DFlash strategy and capture path end to end, so the
external-server setup above carries over unchanged — including the
`--return-aux-hidden-state-layer-ids 1 8 15 22 29` flags, which match
`configs/qwen3.5-4b-dflash2.json`. Use the checked-in
[`qwen3.5-4b-dflash2-npu.yaml`](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/online/disaggregated/external/qwen3.5-4b-dflash2-npu.yaml)
recipe:

```bash
ASCEND_RT_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 \
specforge train -c examples/configs/online/disaggregated/external/qwen3.5-4b-dflash2-npu.yaml
```

Notes specific to DFlash 2 on Ascend:

- **`attention_backend: sdpa` is mandatory** (already set in the recipe). The
  schema default `flex_attention` is rejected at the first training forward
  with `flex_attention is not available on this device; use sdpa/eager`.
- **`sdpa` and `eager` share the explicit boolean-mask path**; both implement
  the same block-local draft visibility, and the grouped convolutions operate
  inside each proposal block independently of the attention mask. The checked-in
  4B draft uses `full_attention` layers only, so no sliding-window mask is
  involved.
- **Selector knobs keep their defaults** (`dflash2_selector_loss_alpha: 1.0`,
  zero warmup/ramp). The selector and convolutions are stock torch ops and need
  no NPU-specific settings.
- **Serving the export** still requires an SGLang build with DFlash 2 support
  (SGLang PR #35371); training-side capture is unchanged.

---

## 4. Managed-local full stack (one command)

Instead of starting Mooncake and capture servers by hand, the
[`qwen3.5-4b-dflash-npu.yaml`](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/online/disaggregated/managed-local/qwen3.5-4b-dflash-npu.yaml)
recipe lets a single `specforge train` command own the whole single-node
stack — Mooncake, capture server(s), and the trainer — and derives their
endpoints and device assignments:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

specforge train -c examples/configs/online/disaggregated/managed-local/qwen3.5-4b-dflash-npu.yaml
```

The checked-in layout parks the capture server on device 0 and runs a 14-rank
trainer on devices 2-15 (8-card hosts: trainer on devices 1-7 with
`deployment.trainer.nproc_per_node=7`).

### Scaling capture throughput

One capture server is enough for correctness but bounds the pipeline: the
trainer's data wait then dominates step time. To give capture more cards,
override the layout inline — e.g. a 6:10 split with six TP=1 capture servers:

```bash
specforge train -c examples/configs/online/disaggregated/managed-local/qwen3.5-4b-dflash-npu.yaml \
  'deployment.trainer.nproc_per_node=10' \
  'deployment.disaggregated.managed_local.trainer_cuda_visible_devices=["6","7","8","9","10","11","12","13","14","15"]' \
  'deployment.disaggregated.managed_local.capture_servers=[{port: 40000, cuda_visible_devices: ["0"], tp_size: 1}, {port: 40001, cuda_visible_devices: ["1"], tp_size: 1}, {port: 40002, cuda_visible_devices: ["2"], tp_size: 1}, {port: 40003, cuda_visible_devices: ["3"], tp_size: 1}, {port: 40004, cuda_visible_devices: ["4"], tp_size: 1}, {port: 40005, cuda_visible_devices: ["5"], tp_size: 1}]'
```

In the validated 16-card run this cut step time from ~20s to ~4.5s.

### Cleanup between runs

Managed children exit with the supervisor, but after an interrupted run check
for strays before relaunching:

```bash
pkill -9 -f specforge; pkill -9 -f mooncake_master; pkill -9 -f torch.distributed.run
rm -rf outputs/qwen3.5-4b-dflash-npu-managed
```

---

## 5. Split producer and consumer roles across nodes

The same configs split across nodes with an explicit `--role`:

```bash
specforge train -c run.yaml --role producer   # inference / capture pool
specforge train -c run.yaml --role consumer --node-rank 0   # trainer-0
specforge train -c run.yaml --role consumer --node-rank 1   # trainer-1
```

Every capture server must use the same target model, capture method, and aux
layer ids. For external-service prerequisites, freshness rules, and resume, see
the [Disaggregated training](../disaggregated_training.md) guide.

---

## 6. NPU notes and troubleshooting

- **Empty `ASCEND_RT_VISIBLE_DEVICES` is invalid** — the Ascend driver rejects
  it. SpecForge unsets the variable instead of emptying it; do not export
  `ASCEND_RT_VISIBLE_DEVICES=` by hand.
- **USP is CUDA-only for now** — `yunchang` probes CUDA at import. Keep
  `sp_ulysses_size` / `sp_ring_size` at 1 and use the `sdpa` trainer backend.
- **Ascend Mooncake rejects wildcard buffer registration** — the trainer side
  forces `local_buffer_size=0` automatically (SpecForge roles are pure
  zero-copy clients), and the capture plugin mounts its feature segment
  with `location="cpu"`.
- **`ACL_ERROR_RT_CONTEXT_NULL` in the capture producer** — the Mooncake
  transfer engine needs a bound device context; SpecForge binds the local NPU
  before `setup()`. Seeing this error means the installation predates #722.
- **Teardown drain** — the lifecycle drain window (~20s) covers Mooncake's
  read-lease TTL, and managed-local masters start with
  `default_kv_lease_ttl_ms=500`. A `could not drain pending removals` error
  after an otherwise successful run means one of these is missing — upgrade
  SpecForge.

---

## Reference results on 16x A3 (64GB)

End-to-end online run on the managed-local stack, training a **Qwen3.5-4B
DFlash** draft (`configs/qwen3.5-4b-dflash.json`) with 6 TP=1 capture servers
(devices 0-5) and 10 trainer ranks (devices 6-15): ~9.5k prompts x 10 epochs,
global batch 80/step, ~1.18k optimizer steps, ~1.5h total.

- Loss falls from ~6.0 to ~2.0 over the run with cosine LR (6e-4 -> 0); no
  NaN, no stall.
- ~4.5s/optimizer step with six capture servers (~20s/step with one).
- Memory: ~29GB per trainer rank, ~31-33GB per capture server; no OOM with
  `num_anchors: 512` (lower it on smaller cards).
- Clean terminal drain on shutdown.

![Qwen3.5-4B DFlash online training on 16x A3](https://github.com/user-attachments/assets/1b94b455-cb61-4374-a548-9a219a37ed64)

These numbers are a functional reference for the NPU managed-local stack, not
a tuned performance benchmark.
