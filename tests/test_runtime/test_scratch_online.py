# coding=utf-8
"""Scratch-online data plane: index-driven refs + wait/read/delete store.

The reader must assemble instantly without touching the filesystem (the
producer may not have written anything yet), and the store must block until
the producer publishes a file, then delete it so the producer's window
advances.
"""

import os
import tempfile
import threading
import time
import unittest

import torch

from specforge.runtime.data_plane.offline_reader import OfflineManifestReader
from specforge.runtime.data_plane.scratch_online import (
    ScratchFileFeatureStore,
    ScratchOnlineReader,
)

_KEYS = ("input_ids", "loss_mask", "hidden_states")


def _base_reader(path: str) -> OfflineManifestReader:
    return OfflineManifestReader(
        path,
        run_id="scratch",
        strategy="hspec",
        feature_keys=_KEYS,
        validate_files=False,
    )


def _payload(index: int):
    return {
        "input_ids": torch.arange(8) + index,
        "loss_mask": torch.ones(8, dtype=torch.long),
        "hidden_states": torch.randn(8, 4).to(torch.bfloat16),
    }


class TestScratchOnlineReader(unittest.TestCase):
    def test_emits_expected_refs_without_touching_filesystem(self):
        with tempfile.TemporaryDirectory() as directory:
            # Nothing on disk; assembly must not look.
            missing = os.path.join(directory, "scratch")
            reader = ScratchOnlineReader(
                _base_reader(missing), expected_samples=5, file_group_size=3
            )
            refs = reader.read()
        self.assertEqual(len(refs), 5)
        self.assertEqual(
            [ref.feature_store_uri for ref in refs],
            [f"file://{reader.path_for(i)}" for i in range(5)],
        )

    def test_grouped_layout_matches_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            reader = ScratchOnlineReader(
                _base_reader(directory), expected_samples=1, file_group_size=2000
            )
            self.assertTrue(
                reader.path_for(2001).endswith(
                    os.path.join("rows_2000-4000", "data_2001.ckpt")
                )
            )

    def test_rejects_nonpositive_expected_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                ScratchOnlineReader(_base_reader(directory), expected_samples=0)


class TestScratchFileFeatureStore(unittest.TestCase):
    def _store(self, **kwargs) -> ScratchFileFeatureStore:
        defaults = dict(poll_interval_s=0.05, file_timeout_s=5.0)
        defaults.update(kwargs)
        return ScratchFileFeatureStore("scratch", **defaults)

    def test_get_waits_for_publish_and_deletes(self):
        with tempfile.TemporaryDirectory() as directory:
            reader = ScratchOnlineReader(_base_reader(directory), expected_samples=1)
            ref = reader.read()[0]
            path = ref.feature_store_uri[len("file://") :]

            def producer():
                time.sleep(0.3)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = path + ".tmp"
                torch.save(_payload(0), tmp)
                os.replace(tmp, path)

            thread = threading.Thread(target=producer)
            started = time.monotonic()
            thread.start()
            tensors, handle = self._store().get(ref)
            thread.join()
            self.assertGreaterEqual(time.monotonic() - started, 0.25)
            self.assertTrue(torch.equal(tensors["input_ids"], torch.arange(8)))
            self.assertFalse(os.path.exists(path), "consumed file must be deleted")
            self._store().release(handle)

    def test_get_keeps_file_when_delete_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            reader = ScratchOnlineReader(_base_reader(directory), expected_samples=1)
            ref = reader.read()[0]
            path = ref.feature_store_uri[len("file://") :]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(_payload(0), path)
            store = self._store(delete_after_read=False)
            tensors, handle = store.get(ref)
            store.release(handle)
            self.assertTrue(os.path.exists(path))

    def test_get_times_out_with_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            reader = ScratchOnlineReader(_base_reader(directory), expected_samples=1)
            ref = reader.read()[0]
            store = self._store(file_timeout_s=0.2)
            with self.assertRaises(TimeoutError) as ctx:
                store.get(ref)
            self.assertIn("producer dead", str(ctx.exception))

    def test_get_fails_fast_on_nan_skip_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            reader = ScratchOnlineReader(_base_reader(directory), expected_samples=1)
            ref = reader.read()[0]
            path = ref.feature_store_uri[len("file://") :]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path + ".nan_skip", "w").close()
            with self.assertRaises(RuntimeError) as ctx:
                self._store().get(ref)
            self.assertIn("NaN-skipped", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
