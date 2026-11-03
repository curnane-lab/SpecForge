"""Copy as sitecustomize.py on PYTHONPATH to trace final-head NPU device binding."""

import os

from specforge_sglang_capture import capture, sink

_original_model_device = capture._model_device
_original_bind_worker_context = sink.SpecCaptureSink._bind_worker_context


def _traced_model_device(model_runner):
    device = _original_model_device(model_runner)
    print(
        "DEVICE_TRACE model_runner.device="
        f"{getattr(model_runner, 'device', None)!r} "
        f"model_runner.gpu_id={getattr(model_runner, 'gpu_id', None)!r} "
        f"resolved={device} visible={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!r}",
        flush=True,
    )
    return device


def _traced_bind_worker_context(self):
    print(f"DEVICE_TRACE sink configured device={self.device}", flush=True)
    result = _original_bind_worker_context(self)
    if self.device is not None and self.device.type == "npu":
        import torch

        print(
            f"DEVICE_TRACE sink bound current_device={torch.npu.current_device()}",
            flush=True,
        )
    return result


capture._model_device = _traced_model_device
sink.SpecCaptureSink._bind_worker_context = _traced_bind_worker_context
