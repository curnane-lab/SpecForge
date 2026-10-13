# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Scratch-online data plane: file-mediated on-demand feature capture.

Instead of precomputing the whole dataset's hidden states up front (tens of
MB per sample — multi-hundred-GB datasets), a producer process
(``scripts/online_capture_producer.py``) fills a scratch directory with the
same offline feature files while training is already running, keeping only a
small sliding window of unconsumed files on disk:

* :class:`ScratchOnlineReader` emits the *expected* refs by index without
  touching the filesystem, so assembly is instant even when the directory is
  still empty. Paths follow ``prepare_hidden_states.py``'s grouped layout
  (``rows_<g>-<g+G>/data_<i>.ckpt``).
* :class:`ScratchFileFeatureStore` waits for each referenced file to appear
  (the producer publishes atomically via tmp+rename, so existence implies a
  complete file), then deletes it after reading.

Two operational constraints follow from the window protocol:

* Sampling must be sequential (``shuffle=False``): the producer writes indices
  in order and caps the number of *unconsumed* files, so an out-of-order read
  of a far-ahead index can deadlock the window. ``build_offline_runtime``
  forces sequential sampling when scratch mode is on.
* The trainer must actually consume every produced index. A dropped tail batch
  (``drop_last=True`` with ``batch_size > 1``) strands files in the window;
  prefer ``batch_size == 1`` or ``expected_samples % batch_size == 0``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Iterator, List, Optional

from specforge.runtime.contracts import SampleRef
from specforge.runtime.data_plane.feature_store import LocalFeatureStore

logger = logging.getLogger(__name__)

_FILE_GROUP_SIZE_DEFAULT = 2000


class ScratchOnlineReader:
    """Index-driven reader over a scratch dir filled by an external producer.

    Wraps any assembled offline reader (which owns the per-strategy feature
    keys and ref construction) and re-emits refs for indices
    ``0..expected_samples-1`` at the paths the producer will write, without
    checking the filesystem — files may not exist yet by construction.
    """

    def __init__(
        self,
        base_reader,
        *,
        expected_samples: int,
        file_group_size: int = _FILE_GROUP_SIZE_DEFAULT,
        file_extension: str = ".ckpt",
    ) -> None:
        if isinstance(expected_samples, bool) or int(expected_samples) <= 0:
            raise ValueError(
                f"scratch-online expected_samples must be a positive int, "
                f"got {expected_samples!r}"
            )
        self._base = base_reader
        self.expected_samples = int(expected_samples)
        self.file_group_size = int(file_group_size)
        self.file_extension = file_extension
        # Mirror the wrapped reader's public attributes for callers that
        # introspect the reader after assembly.
        self.hidden_states_path = base_reader.hidden_states_path
        self.run_id = base_reader.run_id
        self.strategy = base_reader.strategy

    def path_for(self, index: int) -> str:
        """The grouped-layout path ``prepare_hidden_states.py`` writes."""
        group = (index // self.file_group_size) * self.file_group_size
        return os.path.join(
            os.path.abspath(self.hidden_states_path),
            f"rows_{group}-{group + self.file_group_size}",
            f"data_{index}{self.file_extension}",
        )

    def __iter__(self) -> Iterator[SampleRef]:
        for index in range(self.expected_samples):
            yield self._base._ref_for(index, self.path_for(index))

    def read(self, limit: Optional[int] = None) -> List[SampleRef]:
        refs: List[SampleRef] = []
        for i, ref in enumerate(self):
            if limit is not None and i >= limit:
                break
            refs.append(ref)
        return refs


class ScratchFileFeatureStore(LocalFeatureStore):
    """LocalFeatureStore variant that waits for scratch files and deletes them.

    ``get()`` on a ``file://`` ref blocks (polling) until the producer has
    published the file, reads it, and — with ``delete_after_read`` — unlinks
    it so the producer's window can advance. ``torch.load`` materializes the
    full record before the unlink, so no reader can observe a partial file
    (the producer publishes atomically via tmp+rename).
    """

    def __init__(
        self,
        *args,
        poll_interval_s: float = 1.0,
        file_timeout_s: float = 600.0,
        delete_after_read: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be > 0")
        if file_timeout_s <= 0:
            raise ValueError("file_timeout_s must be > 0")
        self.poll_interval_s = float(poll_interval_s)
        self.file_timeout_s = float(file_timeout_s)
        self.delete_after_read = bool(delete_after_read)

    def _get_from_file(
        self, path: str, ref: SampleRef, wanted: List[str]
    ) -> Dict[str, "object"]:
        deadline = time.monotonic() + self.file_timeout_s
        waited = 0.0
        while not os.path.exists(path):
            if os.path.exists(path + ".nan_skip"):
                raise RuntimeError(
                    f"scratch feature file {path} was NaN-skipped by the "
                    "producer; exclude this sample from the dataset (it is "
                    "poison for training anyway)"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"scratch feature file {path} was not produced within "
                    f"{self.file_timeout_s:.0f}s — producer dead, NaN-skipped "
                    "sample, or window starved (see scratch_online docstring)"
                )
            time.sleep(min(self.poll_interval_s, remaining))
            waited += self.poll_interval_s
        if waited > 0:
            logger.debug("scratch file %s arrived after ~%.1fs", path, waited)
        tensors = super()._get_from_file(path, ref, wanted)
        if self.delete_after_read:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        return tensors


__all__ = ["ScratchOnlineReader", "ScratchFileFeatureStore"]
