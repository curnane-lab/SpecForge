import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from specforge.inference.adapters.server_capture import (
    ServerCaptureSchema,
    SGLangServerCaptureAdapter,
)
from specforge.inference.capture import CaptureConfig, CaptureMismatchError
from specforge.runtime.contracts import PromptTask
from tests.test_runtime.test_server_capture import _FakeMooncakeStore

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "plugins/sglang-spec-capture")
)
from specforge_sglang_capture.capture import CaptureRuntime
from specforge_sglang_capture.sink import SpecCaptureSink


def capture_spec(*, mtp=True):
    return {
        "store_id": "store",
        "sample_id": "sample",
        "gen": 1,
        "features": {
            "aux": "hidden_states",
            **({"mtp_hidden": "mtp_hidden_states"} if mtp else {}),
        },
        "passthrough": [],
    }


def batch_and_logits():
    request = SimpleNamespace(
        rid="request",
        origin_input_ids=[1, 2, 3, 4],
        multimodal_inputs=None,
        sampling_params=SimpleNamespace(custom_params={"spec_capture": capture_spec()}),
        finished=lambda: True,
    )
    batch = SimpleNamespace(reqs=[request])
    forward = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        extend_seq_lens_cpu=[4],
        extend_prefix_lens_cpu=[0],
    )
    logits = SimpleNamespace(
        hidden_states=torch.randn(4, 6),
        last_hidden_states=torch.randn(4, 8),
        hidden_states_token_indices=None,
    )
    return batch, forward, logits


class DMTPCaptureTransportTest(unittest.TestCase):
    def test_mtp_rows_reach_host_publication_with_alignment(self):
        backend = _FakeMooncakeStore()
        sink = SpecCaptureSink(aux_layer_ids=[1])
        sink._store = backend
        self.addCleanup(sink._executor.shutdown)
        runtime = CaptureRuntime(
            sink=sink,
            is_writer=True,
            gpu_put=False,
            max_pending_batches=2,
            defer_response=lambda _request, completion: setattr(
                self, "completion", completion
            ),
        )
        runtime.mtp_capture = lambda ids, hidden: hidden + ids[:, None]
        batch, forward, logits = batch_and_logits()
        output = runtime.observer.after_forward(
            batch, forward, logits, can_run_graph=False
        )
        host = output.copy_to_host(lambda tensor: tensor.cpu().clone())
        host.consume(batch, [])
        result = self.completion.future.result()[0]
        self.assertEqual(result["mtp_hidden_alignment"], "anchor_post_norm_v1")
        expected = logits.last_hidden_states + torch.tensor([1, 2, 3, 4])[:, None]
        raw = backend._d["store/sample/g1/mtp_hidden_states"]
        stored = torch.frombuffer(bytearray(raw), dtype=expected.dtype).reshape(1, 4, 8)
        torch.testing.assert_close(stored[0], expected)
        self.assertEqual(result["features"]["mtp_hidden_states"]["shape"], [1, 4, 8])

    def test_missing_mtp_provider_fails_instead_of_using_hidden_states(self):
        sink = SpecCaptureSink(aux_layer_ids=[1])
        self.addCleanup(sink._executor.shutdown)
        runtime = CaptureRuntime(
            sink=sink,
            is_writer=True,
            gpu_put=False,
            max_pending_batches=2,
            defer_response=lambda *_: None,
        )
        batch, forward, logits = batch_and_logits()
        output = runtime.observer.after_forward(
            batch, forward, logits, can_run_graph=False
        )
        self.assertIn("SPECFORGE_MTP_MODEL_PATH", output.items[0].error)
        self.assertIsNone(output.mtp_hidden)

    def test_adapter_rejects_missing_alignment(self):
        adapter = SGLangServerCaptureAdapter(
            "http://capture",
            SimpleNamespace(
                store_id="store",
                adopt=lambda *args, **kwargs: None,
                discard_external_attempts=lambda *args, **kwargs: None,
                track_external_attempt=lambda *args, **kwargs: None,
            ),
            run_id="run",
            algorithm="dflash",
            schema=ServerCaptureSchema(
                aux_feature="hidden_states",
                last_hidden_feature=None,
                passthrough=(),
                mtp_hidden_feature="mtp_hidden_states",
            ),
            server_capture="plugin",
        )
        task = PromptTask(
            task_id="task",
            run_id="run",
            source_id="test",
            payload={"input_ids": [1, 2, 3, 4]},
            max_length=4,
        )
        capture = CaptureConfig.from_strategy(
            {"hidden_states", "mtp_hidden_states"},
            [1],
            target_repr="hidden_state",
            target_hidden_size=8,
        )
        result = {
            "sample_id": "sample",
            "store_id": "store",
            "gen": 1,
            "features": {
                name: {"shape": [1, 4, 8], "dtype": "float32"}
                for name in ("hidden_states", "mtp_hidden_states")
            },
        }
        with self.assertRaisesRegex(CaptureMismatchError, "mtp_hidden_alignment"):
            adapter._ref_from_result(task, result, capture)
        result["mtp_hidden_alignment"] = "anchor_post_norm_v1"
        ref = adapter._ref_from_result(task, result, capture)
        self.assertEqual(ref.metadata["mtp_hidden_alignment"], "anchor_post_norm_v1")
        self.assertIn("mtp_hidden_states", ref.feature_specs)


if __name__ == "__main__":
    unittest.main()
