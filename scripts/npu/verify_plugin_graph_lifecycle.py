"""N2 review follow-up: prefill graph replay must not corrupt a *plugin* capture.

Unlike verify_capture_graph_lifecycle.py (plain return_hidden_states requests),
this drives the real plugin chain: spec_capture custom_params -> observer
snapshot -> async D2H -> sink publication -> deferred response -> Mooncake
read-back.

Design:
  Phase 1 (eager reference): capture-ON engine, graphs disabled; a plain
    return_hidden_states request records prompt A's aux rows.
  Phase 2 (graph): capture-ON engine with the prefill graph backend enabled.
    A test-only sitecustomize shim delays every sink put_samples by
    SPECFORGE_CAPTURE_TEST_PUT_DELAY_S seconds, so capture request A's
    publication is still pending while a burst of normal requests and a
    repeat capture A2 run later prefill replays. Completion timestamps must
    show every burst response arriving before A's (deferred) response.
  Pass iff:
    - every burst response completes while A is still unpublished, and A/A2
      return feature refs without error, and
    - A's published aux matches the eager reference within kernel tolerance,
      and A2's published aux is bit-identical to A's, and
    - passthrough input_ids are byte-exact and last_hidden is finite.

Radix cache is disabled so A2 really re-runs a full prefill.

Usage:
  cp scripts/npu/graph_delay_sitecustomize.py /tmp/npu_verify/sitecustomize.py
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ASCEND_RT_VISIBLE_DEVICES=1 PYTHONPATH=/tmp/npu_verify:<fork>/python:<specforge>:$PYTHONPATH \
      LD_LIBRARY_PATH=/data1/f00538480/cuda_stub_lib:$LD_LIBRARY_PATH \
      python scripts/npu/verify_plugin_graph_lifecycle.py [model_path]
  The shell wrapper should also grep the log for "Capturing prefill shape"
  and "npu graph: True" to prove replays happened.
"""

import asyncio
import json
import os
import sys
import time

import torch

LAYER_IDS = [2, 16, 29]
HIDDEN = 2560
INPUT_IDS = [151644, 872, 198, 6023, 374, 11914, 30, 151645, 198]
STORE_ID, GEN = "pgraph", 1
PUT_DELAY_S = "10"


def boot(model_path, *, graphs: bool):
    import sglang as sgl

    kwargs = dict(
        model_path=model_path,
        mem_fraction_static=0.5,
        page_size=64,
        mamba_ssm_dtype="bfloat16",
        log_level="info",
        enable_return_aux_hidden_states=True,
        speculative_algorithm="DFLASH",
        return_aux_hidden_state_layer_ids=LAYER_IDS,
        return_hidden_states_mode="full",
        chunked_prefill_size=-1,
        skip_tokenizer_init=True,
        disable_radix_cache=True,
    )
    if graphs:
        kwargs.update(
            cuda_graph_backend_prefill="breakable",
            cuda_graph_bs_prefill=[64, 128, 512],
        )
    else:
        kwargs["disable_cuda_graph"] = True
    return sgl.Engine(**kwargs)


def spec(sample_id):
    return {
        "store_id": STORE_ID,
        "sample_id": sample_id,
        "gen": GEN,
        "replace": False,
        "features": {"aux": "hidden_states", "last_hidden": "target"},
        "passthrough": [
            {
                "name": "input_ids",
                "data": INPUT_IDS,
                "shape": [1, len(INPUT_IDS)],
                "dtype": "int64",
            }
        ],
    }


def capture_params(sample_id):
    return {
        "max_new_tokens": 1,
        "temperature": 0.0,
        "custom_params": {"spec_capture": json.dumps(spec(sample_id))},
    }


def rel_err(a, b):
    return (a - b).norm().item() / b.norm().clamp_min(1e-6).item()


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def eager_reference(engine):
    ref = engine.generate(
        None,
        input_ids=INPUT_IDS,
        sampling_params={"max_new_tokens": 1, "temperature": 0.0},
        return_hidden_states=True,
    )
    n_prompt = ref["meta_info"]["prompt_tokens"]
    rows = [r for block in ref["meta_info"]["hidden_states"] for r in block]
    return torch.tensor(rows[:n_prompt], dtype=torch.float32)  # [n, 3H]


class Reader:
    def __init__(self):
        from mooncake.store import MooncakeDistributedStore

        self.store = MooncakeDistributedStore()
        rc = self.store.setup(
            local_hostname="localhost",
            metadata_server="http://localhost:8080/metadata",
            global_segment_size=0,
            local_buffer_size=0,
            protocol="tcp",
            rdma_devices="",
            master_server_addr="localhost:50051",
        )
        assert rc in (0, None), rc

    def read(self, sample_id, name, shape, dtype):
        key = f"{STORE_ID}/{sample_id}/g{GEN}/{name}"
        buf = torch.zeros(shape, dtype=dtype)
        ptr, nbytes = buf.data_ptr(), buf.numel() * buf.element_size()
        rc = self.store.register_buffer(ptr, nbytes)
        assert rc in (0, None), (key, rc)
        try:
            got = -800
            for attempt in range(4):  # TRANSFER_FAIL (-800) is retryable
                got = self.store.get_into(key, ptr, nbytes)
                if got == nbytes:
                    break
                time.sleep(1 + attempt)
            assert got == nbytes, (key, got, nbytes)
        finally:
            # Mooncake rejects overlapping registered regions (-600); torch's
            # caching allocator reuses freed addresses.
            self.store.unregister_buffer(ptr)
        return buf

    def close(self):
        self.store.close()


async def phase2(engine):
    done = {}

    async def fire(name, ids, params, hidden=False):
        out = await engine.async_generate(
            None,
            input_ids=ids,
            sampling_params=params,
            return_hidden_states=hidden,
        )
        done[name] = time.monotonic()
        return out

    tasks = [asyncio.create_task(fire("A", INPUT_IDS, capture_params("a")), name="A")]
    burst_tasks = []
    for j, length in enumerate([9, 20, 40, 70, 100, 200]):
        ids = [(1000 + 17 * j + i) % 150000 for i in range(length)]
        burst_tasks.append(
            asyncio.create_task(
                fire(
                    f"burst{j}",
                    ids,
                    {"max_new_tokens": 4, "temperature": 0.0},
                ),
                name=f"burst{j}",
            )
        )
    outs = await asyncio.gather(*tasks, *burst_tasks)
    # A2 runs alone after the burst: same single-request graph path as A, so
    # the repeat capture must be bit-identical (batching a capture with other
    # prompts introduces ordinary bf16 batch-arrangement noise instead).
    out_a2 = await fire("A2", INPUT_IDS, capture_params("a2"))
    return done, outs[0], out_a2


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/data1/f00538480/Qwen3.5-4B"
    os.environ["SPECFORGE_SPEC_CAPTURE"] = "1"
    failures = []

    print("=== phase 1: eager reference ===", flush=True)
    engine = boot(model_path, graphs=False)
    ref_aux = eager_reference(engine)
    n_prompt = ref_aux.shape[0]
    engine.shutdown()
    print(f"eager rows for A: {tuple(ref_aux.shape)}")

    print("=== phase 2: real capture requests + delayed publication + replays ===", flush=True)
    os.environ["SPECFORGE_CAPTURE_TEST_PUT_DELAY_S"] = PUT_DELAY_S
    engine = boot(model_path, graphs=True)
    done, out_a, out_a2 = asyncio.run(phase2(engine))
    os.environ.pop("SPECFORGE_CAPTURE_TEST_PUT_DELAY_S", None)

    print("\n--- completion timeline (monotonic seconds) ---")
    for name, ts in sorted(done.items(), key=lambda kv: kv[1]):
        print(f"  {name}: {ts:.3f}")
    for j in range(6):
        name = f"burst{j}"
        if done[name] >= done["A"]:
            failures.append(f"{name} did not complete while A was still pending")
    if not failures:
        print("overlap: every burst response completed before A's deferred response")

    for label, out in (("A", out_a), ("A2", out_a2)):
        sc = out.get("meta_info", {}).get("spec_capture")
        if not sc or "error" in sc[0]:
            failures.append(f"{label} capture result error: {sc}")
        else:
            print(f"{label}: published features {sorted(sc[0].get('features', {}))}")

    reader = Reader()
    ids = reader.read("a", "input_ids", (1, len(INPUT_IDS)), torch.int64)
    if ids[0].tolist() != INPUT_IDS:
        failures.append("input_ids mismatch")
    else:
        print("input_ids: byte-exact")

    aux_a = reader.read("a", "hidden_states", (1, n_prompt, len(LAYER_IDS) * HIDDEN), torch.bfloat16)[0].float()
    aux_a2 = reader.read("a2", "hidden_states", (1, n_prompt, len(LAYER_IDS) * HIDDEN), torch.bfloat16)[0].float()
    last = reader.read("a", "target", (1, n_prompt, HIDDEN), torch.bfloat16)[0].float()
    if not torch.isfinite(last).all():
        failures.append("last_hidden contains non-finite values")

    print("\n--- published A(graph) vs eager reference ---")
    for i, lid in enumerate(LAYER_IDS):
        a = aux_a[:, i * HIDDEN : (i + 1) * HIDDEN]
        r = ref_aux[:, i * HIDDEN : (i + 1) * HIDDEN]
        print(f"chunk{i} (layer {lid}): rel={rel_err(a, r):.4e} cos={cos_sim(a, r):.6f}")
        if not (rel_err(a, r) < 2e-2 and cos_sim(a, r) > 0.999):
            failures.append(f"published A chunk{i} diverges from eager under graph replay")

    print("--- published A2(graph, after burst) vs published A(graph) ---")
    if torch.equal(aux_a2, aux_a):
        print("A2 aux is bit-identical to A aux")
    else:
        for i, lid in enumerate(LAYER_IDS):
            a = aux_a[:, i * HIDDEN : (i + 1) * HIDDEN]
            b = aux_a2[:, i * HIDDEN : (i + 1) * HIDDEN]
            print(f"chunk{i} (layer {lid}): rel={rel_err(b, a):.4e}")
        failures.append("published A2 differs from A; replay may corrupt the capture")

    reader.close()
    engine.shutdown()
    print("\n=== RESULT ===")
    if failures:
        for f in failures:
            print("FAIL:", f)
        sys.exit(1)
    print("VERIFY_PLUGIN_GRAPH_OK")


if __name__ == "__main__":
    main()
