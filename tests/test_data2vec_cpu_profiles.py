"""An unsuitable follow-up host must stop before downloads or compilation."""

from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cpu_profiles", ROOT / "tools/ci-heavy/data2vec_cpu_profiles.py")
profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiles)


class HardwarePreflightTest(unittest.TestCase):
    def exercise(self, flags, expect_skip):
        actual_read = Path.read_text

        def read(path, *args, **kwargs):
            return "flags : " + flags if str(path) == "/proc/cpuinfo" else actual_read(path, *args, **kwargs)

        def command(argv, *args, **kwargs):
            if str(argv[0]) == "bash":
                raise RuntimeError("package-boundary reached")
            return ""

        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, HEAVY_OUT=folder + "/out", HEAVY_SCRATCH=folder))
            stack.enter_context(patch.object(sys, "argv", ["worker", "--require-avx512"]))
            stack.enter_context(patch.object(Path, "read_text", read))
            stack.enter_context(patch.object(profiles, "headroom", return_value={}))
            calls = stack.enter_context(patch.object(profiles, "run", side_effect=command))
            download = stack.enter_context(patch.object(profiles.regression, "hf_download"))
            if expect_skip:
                profiles.main()
            else:
                with self.assertRaisesRegex(RuntimeError, "package-boundary reached"):
                    profiles.main()
            receipt = json.loads((Path(folder) / "out/receipt.json").read_text())
            self.assertFalse(receipt["complete"])
            self.assertFalse(receipt["profiles"])
            download.assert_not_called()
            if expect_skip:
                self.assertEqual(receipt["skipped"], "avx512-unavailable")
                self.assertEqual([call.args[2] for call in calls.call_args_list],
                                 ["cpu", "compiler", "submodules"])
            else:
                self.assertNotIn("skipped", receipt)
                self.assertIn("package-boundary", receipt["error"])

    def test_avx2_only_host_stops_before_any_package_or_model_work(self):
        self.exercise("avx2 fma f16c", True)

    def test_full_avx512_host_can_reach_package_setup(self):
        self.exercise("avx2 fma f16c avx512f avx512cd avx512bw avx512dq avx512vl", False)


if __name__ == "__main__":
    unittest.main()
