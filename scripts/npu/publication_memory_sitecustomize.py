"""Copy as sitecustomize.py to trace publication memory and delay the sink."""

import os
import resource
import time

from specforge_sglang_capture import capture, sink


def _sample_bytes(samples):
    return sum(
        tensor.numel() * tensor.element_size()
        for _, aux, last_hidden in samples
        for tensor in (aux, last_hidden)
        if tensor is not None
    )


def _memory_fields():
    fields = {"rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    try:
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            fields["npu_allocated_mib"] = torch.npu.memory_allocated() / (1 << 20)
            fields["npu_reserved_mib"] = torch.npu.memory_reserved() / (1 << 20)
    except Exception:
        pass
    return fields


def _log(event, **fields):
    values = {**fields, **_memory_fields()}
    print(
        "PUBLICATION_MEMORY "
        + event
        + " "
        + " ".join(f"{key}={value}" for key, value in values.items()),
        flush=True,
    )


_original_copy_to_host = capture.CaptureDeviceOutput.copy_to_host
_original_submit_samples = sink.SpecCaptureSink.submit_samples
_original_put_samples = sink.SpecCaptureSink.put_samples


def _traced_copy_to_host(self, copy_tensor):
    nbytes = _sample_bytes([(None, self.aux, self.last_hidden)])
    _log("d2h_begin", payload_bytes=nbytes)
    output = _original_copy_to_host(self, copy_tensor)
    _log("d2h_end", payload_bytes=nbytes)
    return output


def _traced_submit_samples(self, samples, **kwargs):
    future = _original_submit_samples(self, samples, **kwargs)
    _log("submit", samples=len(samples), payload_bytes=_sample_bytes(samples))
    return future


def _traced_put_samples(self, samples):
    delay = float(os.environ.get("SPECFORGE_CAPTURE_TEST_PUT_DELAY_S", "0"))
    _log("put_begin", samples=len(samples), delay_s=delay)
    if delay:
        time.sleep(delay)
    result = _original_put_samples(self, samples)
    _log("put_end", samples=len(samples))
    return result


capture.CaptureDeviceOutput.copy_to_host = _traced_copy_to_host
sink.SpecCaptureSink.submit_samples = _traced_submit_samples
sink.SpecCaptureSink.put_samples = _traced_put_samples


print(f"PUBLICATION_MEMORY capture_file={capture.__file__}", flush=True)

if hasattr(capture.PendingCaptures, "try_reserve"):
    _original_try_reserve = capture.PendingCaptures.try_reserve

    def _traced_try_reserve(self, nbytes):
        lease = _original_try_reserve(self, nbytes)
        _log(
            "reserve",
            request_bytes=nbytes,
            pending_bytes=self.pending_bytes,
            admitted=lease is not None,
        )
        return lease

    capture.PendingCaptures.try_reserve = _traced_try_reserve
