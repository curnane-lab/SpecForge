"""Gate publication backpressure through the real online producer on Ascend."""

import json
import os
import signal
import subprocess
import sys
import threading
import time

import requests

LAYER_IDS = [2, 16, 29]
HIDDEN = 2560
STORE_ID = "publication-producer"
LENGTHS = [4096, 4096, 4096, 7, 2048, 129]
PORT = int(os.environ.get("MEASURE_PORT", "30002"))
CHANNEL_PATH = f"/tmp/publication_producer_refs_{os.getpid()}.jsonl"
SERVER_LOG = "/tmp/publication_producer_server.log"


class _StaggeredPost:
    def __init__(self):
        self._seen = set()
        self._lock = threading.Lock()
        self.backpressure = 0

    def __call__(self, url, json_body, timeout):
        specs = [
            json.loads(params["custom_params"]["spec_capture"])
            for params in json_body["sampling_params"]
        ]
        sample_id = specs[0]["sample_id"]
        with self._lock:
            first_attempt = sample_id not in self._seen
            self._seen.add(sample_id)
        if first_attempt:
            index = int(sample_id.rsplit(":s", 1)[1])
            time.sleep(index * 0.75)
        response = requests.post(url, json=json_body, timeout=timeout)
        response.raise_for_status()
        rows = response.json()
        with self._lock:
            self.backpressure += sum(
                (row.get("meta_info", {}).get("spec_capture") or [{}])[0].get(
                    "backpressure"
                )
                == "publication_bytes"
                for row in rows
            )
        return rows


def _server_args(model_path):
    return [
        "--model-path",
        model_path,
        "--skip-tokenizer-init",
        "--mem-fraction-static",
        "0.5",
        "--page-size",
        "64",
        "--disable-cuda-graph",
        "--chunked-prefill-size",
        "-1",
        "--enable-return-aux-hidden-states",
        "--return-aux-hidden-state-layer-ids",
        *(str(layer) for layer in LAYER_IDS),
        "--return-hidden-states-mode",
        "full",
        "--speculative-algorithm",
        "DFLASH",
        "--disable-radix-cache",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
    ]


def _wait_ready(server):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"capture server exited early; see {SERVER_LOG}")
        try:
            response = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("capture server not ready")


def _prompts():
    prompts = []
    for index, length in enumerate(LENGTHS):
        input_ids = [(1000 + index * 17 + offset) % 150000 for offset in range(length)]
        prompts.append(
            {
                "task_id": f"s{index}",
                "payload": {
                    "input_ids": input_ids,
                    "loss_mask": [1] * length,
                },
            }
        )
    return prompts


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/data1/f00538480/Qwen3.5-4B"
    env = dict(os.environ)
    env["SPECFORGE_SPEC_CAPTURE"] = "1"
    env["SGLANG_MAMBA_SSM_DTYPE"] = "bfloat16"
    env["SGLANG_SPEC_CAPTURE_MAX_PENDING_BATCHES"] = "2"
    if os.path.exists(CHANNEL_PATH):
        os.remove(CHANNEL_PATH)
    with open(SERVER_LOG, "w") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "sglang.launch_server", *_server_args(model_path)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    store = None
    try:
        _wait_ready(server)
        from specforge.algorithms.builtin import builtin_algorithm_registry
        from specforge.inference.adapters.server_capture import (
            ServerCaptureSchema,
            SGLangServerCaptureAdapter,
        )
        from specforge.launch import build_disagg_online_producer
        from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
        from specforge.runtime.data_plane.streaming_ref_channel import (
            StreamingRefChannel,
        )

        store = MooncakeFeatureStore(
            store_id=STORE_ID,
            setup_kwargs={
                "local_hostname": "localhost",
                "metadata_server": "http://localhost:8080/metadata",
                "master_server_addr": "localhost:50051",
                "protocol": "tcp",
                "rdma_devices": "",
                "global_segment_size": 0,
                "local_buffer_size": 0,
            },
        )
        algorithm = builtin_algorithm_registry().resolve("dflash")
        layout = algorithm.providers.server_streaming_for("text").layout
        post = _StaggeredPost()
        adapter = SGLangServerCaptureAdapter(
            f"http://127.0.0.1:{PORT}",
            store,
            run_id=STORE_ID,
            algorithm=algorithm.name,
            schema=ServerCaptureSchema(
                aux_feature=layout.aux_feature,
                last_hidden_feature=layout.last_hidden_feature,
                passthrough=layout.passthrough,
                attention_mask_feature=layout.attention_mask_feature,
            ),
            server_capture="plugin",
            post_fn=post,
            backpressure_timeout_s=60.0,
            backpressure_poll_s=0.2,
        )
        channel = StreamingRefChannel(CHANNEL_PATH)
        channel.publish_consumer_quantum(1)
        started = time.monotonic()
        _workers, drive = build_disagg_online_producer(
            algorithm=algorithm,
            prompts=_prompts(),
            feature_store=store,
            channel=channel,
            feature_source=adapter,
            run_id=STORE_ID,
            target_hidden_size=HIDDEN,
            target_repr=None,
            aux_hidden_state_layer_ids=LAYER_IDS,
            lease=1,
            producer_concurrency=len(LENGTHS),
            max_prompt_attempts=5,
        )
        produced = drive(max_rounds=10_000)
        elapsed = time.monotonic() - started
        assert produced == len(LENGTHS), produced
        assert channel.is_closed()
        assert post.backpressure > 0, "workload did not exercise backpressure"

        refs = StreamingRefChannel(CHANNEL_PATH).poll()
        assert len(refs) == len(LENGTHS), len(refs)
        prompts = {f"s{index}": prompt for index, prompt in enumerate(_prompts())}
        for ref in refs:
            prompt = prompts[ref.source_task_id]
            expected_ids = prompt["payload"]["input_ids"]
            tensors, handle = store.get(ref, device="cpu")
            try:
                assert tensors["input_ids"][0].tolist() == expected_ids
                assert (
                    tensors["loss_mask"][0].tolist() == prompt["payload"]["loss_mask"]
                )
                assert tensors["hidden_states"].shape == (
                    1,
                    len(expected_ids),
                    len(LAYER_IDS) * HIDDEN,
                )
            finally:
                store.release(handle)
            print(
                f"MEASURE_PRODUCER sample={ref.source_task_id} "
                f"tokens={len(expected_ids)} read_back=True",
                flush=True,
            )
        print(
            f"MEASURE_PUBLICATION_PRODUCER_OK produced={produced} "
            f"backpressure={post.backpressure} elapsed_s={elapsed:.3f} "
            "max_prompt_attempts=5",
            flush=True,
        )
    finally:
        if store is not None:
            store.close()
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=15)


if __name__ == "__main__":
    main()
