"""Capture through a REAL sglang server (managed-local argv replica) -> .ckpt.

The earlier round-trip used an in-process Engine; the online route uses a
separate server process fed by the producer's batched HTTP request in
seed-shuffled order (random.Random(seed+epoch)). This script reproduces that
path exactly — same launch_server, same args, one batched /generate in the
shuffled order — and writes offline .ckpt files in that same order, so the
offline trainer replays the online run's consumption order with the server's
own feature bytes.

Run: card via ASCEND_RT_VISIBLE_DEVICES; background mooncake_master on
50051/8080; PYTHONPATH includes the sglang fork; LD_LIBRARY_PATH has the
CUDA stub dir.
"""

import json
import os
import signal
import subprocess
import sys
import time

import requests
import torch

HIDDEN = 2560
SMOKE_DIR = "/data1/f00538480/specforge-npu-capture/scripts/npu/smoke"
DATASET = f"{SMOKE_DIR}/smoke_dataset.jsonl"
OUT_DIR = f"{SMOKE_DIR}/offline_features_server"
PORT = 30001

SERVER_ARGS = [
    "--model-path", "/data1/f00538480/Qwen3.5-4B",
    "--tokenizer-path", "/data1/f00538480/Qwen3.5-4B",
    "--skip-tokenizer-init",
    "--trust-remote-code",
    "--context-length", "1031",
    "--dtype", "bfloat16",
    "--mem-fraction-static", "0.4",
    "--chunked-prefill-size", "-1",
    "--page-size", "128",
    "--attention-backend", "ascend",
    "--enable-return-aux-hidden-states",
    "--return-aux-hidden-state-layer-ids", "1", "8", "15", "22", "29",
    "--return-hidden-states-mode", "full",
    "--speculative-algorithm", "DFLASH",
    "--disable-radix-cache",
    "--host", "127.0.0.1",
    "--port", str(PORT),
]


def main():
    import random

    samples = []
    with open(DATASET) as f:
        for line in f:
            rec = json.loads(line)
            samples.append((rec["input_ids"], rec["loss_mask"]))

    perm = list(range(len(samples)))
    random.Random(42).shuffle(perm)  # producer order: seed=training.seed, epoch=0
    print("shuffled order:", perm, flush=True)

    env = dict(os.environ)
    env["SPECFORGE_SPEC_CAPTURE"] = "1"
    env["SGLANG_MAMBA_SSM_DTYPE"] = "bfloat16"
    server = subprocess.Popen(
        [sys.executable, "-m", "sglang.launch_server", *SERVER_ARGS],
        env=env,
        stdout=open("/tmp/server_capture_replica.log", "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError("server exited early; see /tmp/server_capture_replica.log")
            try:
                r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=5)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise TimeoutError("server not ready")

        batch_ids, batch_params = [], []
        for k, src in enumerate(perm):
            ids, mask = samples[src]
            spec = {
                "store_id": "server_rt",
                "sample_id": f"r{k}",
                "gen": 1,
                "replace": False,
                "features": {"aux": "hidden_states", "last_hidden": "target_last_hidden_states"},
                "passthrough": [
                    {"name": "input_ids", "data": ids, "shape": [1, len(ids)], "dtype": "int64"},
                    {"name": "loss_mask", "data": mask, "shape": [1, len(mask)], "dtype": "int64"},
                ],
            }
            batch_ids.append(ids)
            batch_params.append({
                "max_new_tokens": 0,
                "temperature": 0.0,
                "custom_params": {"spec_capture": json.dumps(spec)},
            })
        resp = requests.post(
            f"http://127.0.0.1:{PORT}/generate",
            json={"input_ids": batch_ids, "sampling_params": batch_params},
            timeout=300,
        )
        resp.raise_for_status()
        outs = resp.json()
        assert len(outs) == len(perm), len(outs)
        for k, out in enumerate(outs):
            result = (out.get("meta_info", {}).get("spec_capture") or [{}])[0]
            assert "error" not in result, (k, result)
        print(f"captured {len(outs)} samples via server batch", flush=True)

        from mooncake.store import MooncakeDistributedStore

        store = MooncakeDistributedStore()
        rc = store.setup(
            local_hostname="localhost",
            metadata_server="http://localhost:8080/metadata",
            global_segment_size=0,
            local_buffer_size=0,
            protocol="tcp",
            rdma_devices="",
            master_server_addr="localhost:50051",
        )
        assert rc in (0, None), rc

        def read_back(key, shape, dtype):
            buf = torch.zeros(shape, dtype=dtype)
            ptr, nbytes = buf.data_ptr(), buf.numel() * buf.element_size()
            rc = store.register_buffer(ptr, nbytes)
            assert rc in (0, None), (key, rc)
            try:
                got = -800
                for attempt in range(4):
                    got = store.get_into(key, ptr, nbytes)
                    if got == nbytes:
                        break
                    time.sleep(1 + attempt)
                assert got == nbytes, (key, got, nbytes)
                return buf
            finally:
                store.unregister_buffer(ptr)

        os.makedirs(OUT_DIR, exist_ok=True)
        failures = []
        for k, src in enumerate(perm):
            ids, mask = samples[src]
            length = len(ids)
            prefix = f"server_rt/r{k}/g1"
            input_ids = read_back(f"{prefix}/input_ids", (1, length), torch.int64)
            loss_mask = read_back(f"{prefix}/loss_mask", (1, length), torch.int64)
            hidden = read_back(f"{prefix}/hidden_states", (1, length, 5 * HIDDEN), torch.bfloat16)
            last = read_back(f"{prefix}/target_last_hidden_states", (1, length, HIDDEN), torch.bfloat16)
            if input_ids[0].tolist() != ids:
                failures.append(f"slot {k}: input_ids mismatch")
                continue
            path = os.path.join(OUT_DIR, f"sample_{k:05d}.ckpt")
            torch.save(
                {
                    "input_ids": input_ids[0].cpu(),
                    "loss_mask": loss_mask[0].cpu(),
                    "hidden_states": hidden.cpu(),
                    "target_last_hidden_states": last.cpu(),
                },
                path,
            )
            print(f"wrote {path} (dataset sample {src})", flush=True)
        store.close()
        if failures:
            for f in failures:
                print("FAIL:", f)
            return 1
        print("SERVER_CAPTURE_TO_FILES_OK")
        return 0
    finally:
        os.killpg(server.pid, signal.SIGTERM)
        time.sleep(3)
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGKILL)


if __name__ == "__main__":
    sys.exit(main())
