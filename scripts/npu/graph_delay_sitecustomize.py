"""Test-only shim: delay the plugin sink's publication by a fixed time.

Active only when SPECFORGE_CAPTURE_TEST_PUT_DELAY_S is set in the environment
of the process (the capture server's scheduler subprocess inherits it).
Runs in every Python process on PYTHONPATH; a no-op otherwise.
"""

import os

_delay = os.environ.get("SPECFORGE_CAPTURE_TEST_PUT_DELAY_S")
if _delay:
    try:
        import time

        from specforge_sglang_capture import sink as _sink

        _orig_put = _sink.SpecCaptureSink.put_samples

        def _delayed_put(self, samples, **kwargs):
            print(
                f"TEST-DELAY: put_samples holding publication for {_delay}s",
                flush=True,
            )
            time.sleep(float(_delay))
            return _orig_put(self, samples, **kwargs)

        _sink.SpecCaptureSink.put_samples = _delayed_put
    except Exception as exc:  # never break an unrelated process
        print(f"TEST-DELAY shim inactive: {exc}", flush=True)
