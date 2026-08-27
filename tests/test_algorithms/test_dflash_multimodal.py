from __future__ import annotations

import unittest
from types import SimpleNamespace

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.common.vlm_input import VlmServerInputAdapter
from specforge.algorithms.contracts import FeatureMode
from specforge.data.vlm_preprocessing import _expand_image_region, _image_token_count

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch-free dev boxes
    torch = None
    TORCH_AVAILABLE = False


class MultimodalRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.registry = builtin_algorithm_registry()

    def test_dflash_registers_the_multimodal_streaming_contract(self):
        registration = self.registry.resolve("dflash")
        contract = registration.spec.feature_contract(
            FeatureMode.STREAMING, "multimodal"
        )
        self.assertEqual(
            set(contract.required_tensors),
            {"input_ids", "loss_mask", "hidden_states"},
        )
        provider = registration.providers.server_streaming_for("multimodal")
        self.assertEqual(provider.capture_method, "dflash")
        # position_ids capture is pending re-port onto the upstream
        # v0.5.14 capture-sink rewrite; plain-rope drafts do not consume it.
        self.assertIsNone(provider.layout.position_ids_feature)
        self.assertEqual(provider.layout.aux_feature, "hidden_states")

    def test_other_builtins_have_no_multimodal_contract(self):
        for name in ("domino", "dspark", "eagle3", "peagle"):
            with self.subTest(algorithm=name):
                registration = self.registry.resolve(name)
                with self.assertRaises(KeyError):
                    registration.spec.feature_contract(
                        FeatureMode.STREAMING, "multimodal"
                    )

    def test_input_adapter_factory_builds_a_valid_adapter(self):
        registration = self.registry.resolve("dflash")
        provider = registration.providers.server_streaming_for("multimodal")
        adapter = provider.create_input_adapter(
            SimpleNamespace(model=SimpleNamespace(), data=SimpleNamespace())
        )
        self.assertIsInstance(adapter, VlmServerInputAdapter)


class VlmExpansionMathTest(unittest.TestCase):
    def test_image_token_count_uses_merge_size(self):
        self.assertEqual(_image_token_count([[2, 4, 6]], merge_size=2), 12)
        self.assertEqual(_image_token_count([[1, 2, 2]], merge_size=1), 4)

    def test_expand_image_region_splices_ids_and_zero_mask(self):
        ids, mask = _expand_image_region(
            [10, 99, 20],
            [0, 0, 1],
            pad_token_id=99,
            count=4,
            source="test",
        )
        self.assertEqual(ids, [10, 99, 99, 99, 99, 20])
        self.assertEqual(mask, [0, 0, 0, 0, 0, 1])

    def test_expand_image_region_requires_exactly_one_placeholder(self):
        with self.assertRaises(ValueError):
            _expand_image_region([10, 20], [1, 1], pad_token_id=99, count=4, source="t")
        with self.assertRaises(ValueError):
            _expand_image_region(
                [99, 10, 99], [0, 0, 0], pad_token_id=99, count=4, source="t"
            )


class ExtractImageFieldTest(unittest.TestCase):
    def _extract(self, record):
        from specforge.data.vlm_preprocessing import _extract_image_field

        return _extract_image_field(record, source="t")

    def test_image_and_image_path_fields(self):
        self.assertEqual(self._extract({"image": "a.jpg"}), "a.jpg")
        self.assertEqual(self._extract({"image_path": "b.jpg"}), "b.jpg")
        self.assertIsNone(self._extract({}))

    def test_images_list_takes_the_single_element(self):
        self.assertEqual(self._extract({"images": ["a.jpg"]}), "a.jpg")
        self.assertIsNone(self._extract({"images": []}))

    def test_multi_image_sample_is_fatal(self):
        from specforge.data.vlm_preprocessing import ImageDataError

        with self.assertRaises(ImageDataError):
            self._extract({"images": ["a.jpg", "b.jpg"]})

    def test_non_list_images_and_non_string_element_are_fatal(self):
        from specforge.data.vlm_preprocessing import ImageDataError

        with self.assertRaises(ImageDataError):
            self._extract({"images": "a.jpg"})
        with self.assertRaises(ImageDataError):
            self._extract({"images": [123]})

    def test_conflicting_image_fields_are_fatal(self):
        from specforge.data.vlm_preprocessing import ImageDataError

        with self.assertRaises(ImageDataError):
            self._extract({"image": "a.jpg", "images": ["b.jpg"]})

    def test_unreadable_image_is_fatal_not_skipped(self):
        from specforge.data.vlm_preprocessing import ImageDataError, _load_image

        with self.assertRaises(ImageDataError):
            _load_image("/nonexistent/path/to/image.jpg", source="t")
        with self.assertRaises(ImageDataError):
            _load_image("not-valid-base64!!!", source="t")

    def test_load_image_returns_data_uri(self):
        import base64 as b64mod

        from specforge.data.vlm_preprocessing import _load_image

        # 1x1 white PNG
        png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4"
            "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
        )
        _, uri = _load_image(png_b64, source="t")
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        raw = b64mod.b64decode(uri.split(",", 1)[1])
        self.assertEqual(raw, b64mod.b64decode(png_b64))

    def test_load_image_preserves_input_media_type(self):
        from specforge.data.vlm_preprocessing import _load_image

        png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4"
            "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
        )
        _, uri = _load_image(f"data:image/png;base64,{png_b64}", source="t")
        self.assertTrue(uri.startswith("data:image/png;base64,"))


class VlmRequestInputsTest(unittest.TestCase):
    def test_build_request_inputs_uses_collapsed_ids_and_image_data(self):
        adapter = VlmServerInputAdapter(config=SimpleNamespace())
        tasks = [
            SimpleNamespace(
                payload={
                    "input_ids": [1, 2, 2, 2, 3],
                    "request_input_ids": [1, 2, 3],
                    "image_data": "aGVsbG8=",
                }
            ),
            SimpleNamespace(
                payload={
                    "input_ids": [7, 8],
                    "request_input_ids": [7, 8],
                    "image_data": None,
                }
            ),
        ]
        request = adapter.build_request_inputs(tasks)
        self.assertEqual(request["input_ids"], [[1, 2, 3], [7, 8]])
        self.assertEqual(request["image_data"], ["aGVsbG8=", None])


class ServerCapturePositionIdsTest(unittest.TestCase):
    def _adapter(self, position_ids_feature):
        from specforge.inference.adapters.server_capture import (
            ServerCaptureSchema,
            SGLangServerCaptureAdapter,
        )

        class _FakeStore:
            store_id = "store"

            def adopt(self, ref):
                pass

            def discard_external_attempts(self, *args, **kwargs):
                pass

            def track_external_attempt(self, *args, **kwargs):
                pass

        schema = ServerCaptureSchema(
            aux_feature="hidden_states",
            last_hidden_feature=None,
            passthrough=(
                ("input_ids", "input_ids", ()),
                ("loss_mask", "loss_mask", ()),
            ),
            position_ids_feature=position_ids_feature,
        )
        return SGLangServerCaptureAdapter(
            "http://localhost:1",
            _FakeStore(),
            run_id="run",
            algorithm="dflash",
            schema=schema,
        )

    def _task(self):
        return SimpleNamespace(
            task_id="t0",
            attempt=0,
            payload={"input_ids": [5, 6, 7], "loss_mask": [0, 1, 1]},
            metadata={},
        )

    def test_payload_requests_position_ids_artifact_when_configured(self):
        adapter = self._adapter("position_ids")
        payload = adapter._spec_capture_payload(self._task())
        self.assertEqual(
            payload["features"],
            {"aux": "hidden_states", "position_ids": "position_ids"},
        )

    def test_payload_omits_position_ids_artifact_when_unset(self):
        adapter = self._adapter(None)
        payload = adapter._spec_capture_payload(self._task())
        self.assertEqual(payload["features"], {"aux": "hidden_states"})


@unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
class VlmCollatorTest(unittest.TestCase):
    def test_collator_pads_position_ids_like_other_features(self):
        from specforge.algorithms.common.dflash_family_data import build_vlm_collator

        collate = build_vlm_collator()
        features = [
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "loss_mask": torch.tensor([[0, 1, 1]]),
                "hidden_states": torch.zeros(1, 3, 8),
                "position_ids": torch.arange(9).reshape(1, 3, 3),
            },
            {
                "input_ids": torch.tensor([[4]]),
                "loss_mask": torch.tensor([[1]]),
                "hidden_states": torch.zeros(1, 1, 8),
                "position_ids": torch.arange(3).reshape(1, 1, 3),
            },
        ]
        batch = collate(features)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 3))
        self.assertEqual(tuple(batch["position_ids"].shape), (2, 3, 3))
        # Padding is zeros on the sequence axis.
        self.assertTrue((batch["position_ids"][1, 1:] == 0).all())
        self.assertEqual(batch["position_ids"][0, 2].tolist(), [6, 7, 8])


@unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
class MropeDraftPositionsTest(unittest.TestCase):
    def test_gathered_draft_positions_follow_anchor_offsets(self):
        import torch as t

        from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel

        model = OnlineDFlashModel.__new__(OnlineDFlashModel)
        model.block_size = 2
        anchors = t.tensor([[1, 3]])
        stored = t.arange(5 * 3).reshape(1, 5, 3)
        offsets = t.arange(model.block_size).view(1, 1, -1)
        draft_indices = (anchors.unsqueeze(-1) + offsets).view(1, -1)
        gathered = t.gather(stored, 1, draft_indices.unsqueeze(-1).expand(-1, -1, 3))
        full = t.cat([stored, gathered], dim=1).permute(2, 0, 1)
        self.assertEqual(tuple(full.shape), (3, 1, 5 + 4))
        # Draft slot for anchor=1: positions of indices 1 and 2.
        self.assertEqual(full[:, 0, 5].tolist(), [3, 4, 5])
        self.assertEqual(full[:, 0, 6].tolist(), [6, 7, 8])


@unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
class PlainRopePositionFallbackTest(unittest.TestCase):
    """Plain-rope drafts ignore server mRoPE position_ids and train on the same
    1D convention as the text path; mRoPE drafts consume them."""

    def _build_model(self, use_interleaved_mrope):
        import torch as t
        from torch import nn

        from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel

        class _StubDraftModel(nn.Module):
            def __init__(self, flag):
                super().__init__()
                self.use_interleaved_mrope = flag
                self.recorded = {}

            def forward(
                self,
                position_ids=None,
                noise_embedding=None,
                target_hidden=None,
                attention_mask=None,
            ):
                self.recorded["position_ids"] = position_ids
                return t.zeros(1)

        return OnlineDFlashModel(
            draft_model=_StubDraftModel(use_interleaved_mrope),
            target_lm_head=nn.Linear(8, 32, bias=False),
            target_embed_tokens=nn.Embedding(32, 8),
            mask_token_id=31,
            block_size=2,
            attention_backend="sdpa",
            num_anchors=4,
        )

    def _inputs(self):
        import torch as t

        b, s = 2, 8
        input_ids = t.randint(0, 31, (b, s))
        hidden_states = t.randn(b, s, 16)
        loss_mask = t.ones(b, s)
        # Server-shaped mRoPE feature: (B, S, 3), offset by 100 so it is
        # distinguishable from the 1D arange convention.
        pos3 = (100 + t.arange(s)).view(1, s, 1).expand(b, s, 3).contiguous()
        return input_ids, hidden_states, loss_mask, pos3

    def test_plain_rope_draft_falls_back_to_1d_positions(self):
        import torch as t

        model = self._build_model(use_interleaved_mrope=False)
        input_ids, hidden_states, loss_mask, pos3 = self._inputs()
        t.manual_seed(0)
        model._forward_draft_blocks(
            input_ids, hidden_states, loss_mask, position_ids=pos3
        )
        got = model.draft_model.recorded["position_ids"]
        b, s = input_ids.shape
        self.assertEqual(got.ndim, 2)
        self.assertTrue(
            t.equal(got[:, :s], t.arange(s).unsqueeze(0).expand(b, -1))
        )

    def test_mrope_draft_consumes_server_positions(self):
        import torch as t

        model = self._build_model(use_interleaved_mrope=True)
        input_ids, hidden_states, loss_mask, pos3 = self._inputs()
        t.manual_seed(0)
        model._forward_draft_blocks(
            input_ids, hidden_states, loss_mask, position_ids=pos3
        )
        got = model.draft_model.recorded["position_ids"]
        b, s = input_ids.shape
        self.assertEqual(got.ndim, 3)
        self.assertEqual(got.shape[0], 3)
        self.assertTrue(t.equal(got[:, :, :s], pos3.permute(2, 0, 1)))

    def test_none_position_ids_keeps_text_1d_path(self):
        import torch as t

        model = self._build_model(use_interleaved_mrope=True)
        input_ids, hidden_states, loss_mask, _ = self._inputs()
        t.manual_seed(0)
        model._forward_draft_blocks(
            input_ids, hidden_states, loss_mask, position_ids=None
        )
        got = model.draft_model.recorded["position_ids"]
        b, s = input_ids.shape
        self.assertEqual(got.ndim, 2)
        self.assertTrue(
            t.equal(got[:, :s], t.arange(s).unsqueeze(0).expand(b, -1))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
