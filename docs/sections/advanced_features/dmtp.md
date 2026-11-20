# DMTP training through the capture plugin

DMTP freezes the target and its native MTP, fuses the MTP predictor state with
the first candidate's embedding, and trains the canonical DMTP draft for the
remaining candidates. The default Qwen3.6 recipe uses seven model slots:
`[T1, MASK, MASK, MASK, MASK, MASK, MASK]`; T1 does not receive draft loss.

## Capture contract

The optional `mtp_hidden_states` artifact has shape
`[1, sequence_length, mtp_hidden_size]` and uses `anchor_post_norm_v1` alignment:

```text
mtp_hidden_states[a] = MTP(input_ids[a], target_last_hidden[a-1], causal history)
T1                   = input_ids[a+1]
```

The tensor is after MTP final normalization and before its LM head. Row zero is
padding and is never selected for training. Target context ends before A.
Requests must contain one text document; packed and multimodal capture are not
supported.

The plugin executes the frozen MTP after the target prefill through the existing
forward observer. It does not run speculative decoding or the SGLang draft
worker, and performs no vocabulary projection during capture. Only the writer
rank holds the replicated frozen MTP and shared embedding; reserve device memory
for those weights and activations. The initial implementation supports
unquantized, single-layer, dense Qwen3.5/Qwen3.6 native MTP checkpoints with
default RoPE. Unsupported families and missing weights fail explicitly.

## Server and online training

Use an SGLang environment with the forward-observer and deferred-response APIs
required by the capture plugin. Install this SpecForge checkout and the plugin
in that environment, or put both source directories on `PYTHONPATH`. The main
SpecForge package is needed only when MTP capture is enabled.

Configure Mooncake in both processes, then launch the server:

```bash
export MODEL=/path/to/Qwen3.6-27B
export SPECFORGE_SPEC_CAPTURE=1
export SPECFORGE_MTP_MODEL_PATH="$MODEL"
export SGLANG_PLUGINS=specforge_spec_capture
export SGLANG_SPEC_CAPTURE_GPU_PUT=0

python -m sglang.launch_server \
  --model-path "$MODEL" \
  --speculative-algorithm DFLASH \
  --enable-return-aux-hidden-states \
  --return-aux-hidden-state-layer-ids 1 16 31 46 61 \
  --return-hidden-states-mode full \
  --chunked-prefill-size -1 \
  --port 30000
```

`SPECFORGE_MTP_MODEL_PATH` must identify the same checkpoint as `--model-path`.
DFLASH names only the aux-feature convention; it does not enable an MTP rollout.
Requesting `mtp_hidden_states` without the frozen provider fails explicitly. The
response includes `mtp_hidden_alignment: anchor_post_norm_v1`, and `/server_info`
advertises the same capability for offline-client preflight.

Update paths and the server URL in
`examples/configs/online/disaggregated/external/qwen3.6-27b-dmtp.yaml`, then use
the normal training CLI. Canonical DMTP retains `dflash` as its data/rendezvous
runtime name; this does not change the draft architecture or its distinct MTP
fusion parameters. Checkpoints include those parameters and support the normal
save/resume lifecycle. DMTP speculative inference is not added.

## Offline features

The generic preparation client supports the third feature and independently
validates target and MTP widths:

```bash
python scripts/prepare_hidden_states.py \
  --server-urls http://127.0.0.1:30000 \
  --target-model-path "$MODEL" \
  --draft specforge/config/draft/presets/qwen3.6-27b-dmtp.yaml \
  --data-path ./cache/dataset/sharegpt_train.jsonl \
  --output-path ./cache/hidden_states/qwen3.6-27b-dmtp \
  --chat-template qwen3.5 --max-length 3072
```

Offline training uses `data.hidden_states_path` with the same draft preset.

## Validation

Unit tests cover frozen-model HF decoder parity, causal-prefix equivalence,
mixed-request plugin slicing, legacy three-artifact samples, transport alignment,
and canonical capture contracts. A CPU lifecycle test captures three samples,
trains for two steps, resumes from step one, and compares the final training
state with the uninterrupted run. A real Ascend validation loads Qwen3.5-4B
native MTP weights and checks finite BF16 outputs and causal-prefix consistency.
The Qwen3.6-27B online server/short-training recipe still requires a full
hardware validation before being claimed as verified.
