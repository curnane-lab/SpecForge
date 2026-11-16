# DSpark teacher-position deduplication

Overlapping DSpark anchors can request the same teacher position multiple times.
With `deduplicate_teacher_positions`, the frozen teacher LM head projects each
`(sample, teacher position)` once **per objective chunk**. An inverse row map lets
each draft slot consume that teacher distribution independently. Draft projection,
hard labels, masks, loss weights, Markov head and confidence supervision keep their
existing meaning. There is no cross-step cache or distributed collective.

The loss consumes the unique teacher table directly. It does not gather a repeated
`[proposal slots, vocabulary]` teacher-logit/probability tensor. On CUDA and Ascend
with a compatible Triton backend, L1 uses the existing streaming loss kernels with
an indexed teacher-row load and the Ascend 4096-element tile cap. Without Triton,
the torch fallback tiles the vocabulary. The L1 subgradient at equal probabilities
is zero, matching the DSpark `abs()` reference. Teacher features and head weights
remain stop-gradient; repeated draft slots retain separate gradients.

## Enable

The option defaults to **false**. For the legacy DSpark recipe:

```yaml
training:
  strategy: dspark
  deduplicate_teacher_positions: true
  objective_chunk_blocks: 128
  objective_checkpointing: true
  fused_logits_loss: true
```

`fused_logits_loss: false` selects the tiled torch indexed-L1 fallback. The dense
NPU baseline on this revision uses eager CE/L1; the indexed path can use Triton L1.
These backend changes must be distinguished when attributing performance gains.

For canonical DSpark configurations:

```yaml
training:
  objective_options:
    deduplicate_teacher_positions: true
    objective_chunk_size: 128
```

The top-level `training.deduplicate_teacher_positions: true` is also accepted on
the canonical path. Conflicting settings fail validation. Canonical DSpark supports
this option with its default CE/L1 objective or explicit `ce_l1` / `ce` token loss;
other token losses fail validation when deduplication is enabled. CE-only token
loss can still require teacher L1 for confidence supervision.

Canonical `objective_chunk_size` counts flattened prediction units across the
batch; legacy `objective_chunk_blocks` counts anchors per sample. In both cases,
zero means one whole-objective chunk. Canonical positive chunk sizes checkpoint
the projection/loss when the chunk is smaller than the prediction set, following
the shared executor. Its legacy chunk/checkpoint/fusion switches remain separate.

The execution switches add no model or optimizer state, so existing checkpoints
can resume with deduplication toggled. Numerical results need not be bit-identical:
changing the teacher GEMM row count and the L1 reduction order can change rounding.

## Validate on NPU

Run in the normal SpecForge / torch_npu environment with triton-ascend installed:

```bash
OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_utils/test_teacher_projection.py \
  tests/test_algorithms/test_dspark_teacher_dedup.py
```

The core loss tests automatically include an available NPU/CUDA accelerator,
FP32/BF16, repeated rows, masked gradients, equal/saturated probabilities and
vocabularies up to 151936. They explicitly check that an available Triton backend
executes the indexed kernel. Model-entry and checkpoint tests always exercise
CPU; additional BF16 accelerator model checks are included when available.

Projection/loss microbenchmark matching the Qwen3-8B DSpark head dimensions:

```bash
python -m specforge.benchmarks.benchmark_dspark_teacher_dedup \
  --device npu:0 --dtype bfloat16 \
  --batch 1 --sequence 3072 --anchors 512 --block-size 7 \
  --hidden 4096 --vocab 151936 --chunk 128 \
  --checkpoint --fused --warmup 5 --repeats 20 \
  --pattern uniform --output results/dspark-dedup-uniform.json
```

Repeat with `--pattern suffix --suffix-fraction 0.25` and with `--no-checkpoint`.
The three arms are:

| Arm | Teacher projection | L1 backend |
| --- | --- | --- |
| `dense` | Every slot | Existing CE/L1 backend |
| `dense_indexed` | Every slot, identity map | Indexed L1 |
| `dedup_indexed` | Unique `(sample, position)` rows | Same indexed L1 |

`dedup_indexed` versus `dense_indexed` isolates row deduplication including its
unique/inverse overhead. Compare against `dense` for the total feature effect.
All arms include gather, projection, CE/L1, confidence loss, backward and enabled
checkpoint replay. Python row-count hooks run only in the untimed correctness
probe. JSON includes actual teacher rows projected during forward/replay, loss
and gradient errors, synchronized wall times and allocator peak/reserved memory.
Allocator peaks are not whole-card HBM. The benchmark fails on non-finite results
or loss/gradient errors beyond its numerical thresholds (absolute loss 0.005,
relative gradient L2 0.03).

This is a synthetic objective microbenchmark: it excludes the backbone, Markov
head, teacher inference, data loading, optimizer and distributed training. Check
real-data complete optimizer-step throughput and supervised positions/s before
claiming a training speedup. Deduplication overhead can outweigh smaller GEMMs
when anchors seldom overlap or chunks are small. No NPU performance result is
bundled with this implementation.
