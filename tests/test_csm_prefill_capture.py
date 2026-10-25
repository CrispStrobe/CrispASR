"""Model-free guards for hosted CSM stage capture and resolved HF cache paths."""

import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/ci-heavy"))
import csm_prefill_capture as capture


class CsmPrefillCaptureTests(unittest.TestCase):
    def test_local_invocation_refuses_before_trace_or_library_loading(self):
        argv = ["capture", "--library", "absent", "--model", "absent",
                "--trace-prefix", "absent", "--out", "absent"]
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}), patch.object(sys, "argv", argv), \
                patch.object(capture.ctypes, "CDLL") as load, \
                patch.object(capture.profile, "read_profile_trace") as trace:
            with self.assertRaisesRegex(RuntimeError, "requires a GitHub Actions"):
                capture.main()
        load.assert_not_called()
        trace.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "HF cache fixture uses a symlink")
    def test_resolved_cache_blob_reaches_physical_sha_guard(self):
        with tempfile.TemporaryDirectory(prefix="csm-prefill-guard-") as temp:
            root = Path(temp)
            scratch, out = root / "scratch", root / "out"
            cache = scratch / "hf/blobs"
            cache.mkdir(parents=True)
            out.mkdir()
            blob = cache / "abcdef0123456789"
            blob.write_bytes(b"wrong physical weights")
            model = scratch / "csm-1b-q4_k.gguf"
            model.symlink_to(blob)
            library = scratch / "libcrispasr.so"
            library.write_bytes(b"must never be loaded")
            argv = ["capture", "--library", str(library), "--model", str(model),
                    "--trace-prefix", str(out / "sampled"), "--out", str(out / "prefill")]
            env = {"GITHUB_ACTIONS": "true", "RUNNER_TEMP": str(root),
                   "HEAVY_SCRATCH": str(scratch), "HEAVY_OUT": str(out)}
            state = {"memory": {"MemAvailable": "16777216 kB"}, "loadavg": [0, 0, 0],
                     "disk_root": 20 * 1024**3, "disk_scratch": 20 * 1024**3}
            parsed = {"tokens": list(range(14)), "samples": [
                {"vocab": 2051, "topk": 50, "temperature": 0.9, "rng_before": 1234}]}
            with patch.dict(os.environ, env), patch.object(sys, "argv", argv), \
                    patch.object(capture.os, "uname", return_value=SimpleNamespace(machine="x86_64")), \
                    patch.object(capture.ab, "snapshot", return_value=state), \
                    patch.object(capture.profile, "read_profile_trace", return_value=parsed), \
                    patch.object(capture.ctypes, "CDLL") as load:
                with self.assertRaisesRegex(RuntimeError, "CSM GGUF bytes differ"):
                    capture.main()
            load.assert_not_called()
            self.assertFalse((out / "prefill").exists())


if __name__ == "__main__":
    unittest.main()
