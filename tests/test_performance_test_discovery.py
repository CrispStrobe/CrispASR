#!/usr/bin/env python3
"""Reject missing or misregistered standalone performance regressions."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("discovery", ROOT / "tools/check_performance_test_discovery.py")
DISCOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DISCOVERY)

TARGET = "test-omnivoice-encoder"
TEST_NAME = "omnivoice-encoder-parity"


class StandaloneDiscoveryControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.build = Path(self.temp.name)
        (self.build / "bin").mkdir()
        catch_targets = ("test-arktts-cache", "test-meanvc2-conditioning", "test-voxcpm2-cpu-conv")
        for target in (TARGET, *catch_targets):
            (self.build / "bin" / target).write_bytes(b"fixture executable")
        names = {
            TEST_NAME,
            "core_lfr: stacking matches FunASR padding and striding contract",
            "prefix_beam_search: basic emissions and repeat collapsing",
            "istft-bounded-memory",
            "benchmark-mel-ab-controls",
            "performance-test-discovery-controls",
            *("regression::" + target for target in catch_targets),
        }
        self.inventories = {"unit": set(names), "performance-parity": set(names), "bench": set()}
        self.tests = [{"name": TEST_NAME, "command": [str(self.build / "bin" / TARGET)]}]
        self.catch_calls = []

    def check(self):
        def catch_names(binary, spec):
            self.assertNotEqual(binary.name, TARGET, "A standalone executable must not receive Catch arguments")
            self.catch_calls.append(binary.name)
            return ["regression::" + binary.name]

        # Isolate the standalone contract while retaining the real file check
        # and the complete guard's label/command validation.
        with patch.object(DISCOVERY, "AB_TARGETS", {}), \
             patch.object(DISCOVERY, "inventory", side_effect=lambda build, ctest, config, label: self.inventories[label]), \
             patch.object(DISCOVERY, "catch_names", side_effect=catch_names), \
             patch.object(DISCOVERY.subprocess, "run", return_value=subprocess.CompletedProcess(
                 [], 0, stdout=json.dumps({"tests": self.tests}))):
            return DISCOVERY.check(self.build)

    def test_standalone_passes_without_catch_inventory(self):
        result = self.check()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["executables"][TARGET], 1)
        self.assertNotIn(TARGET, self.catch_calls)

    def test_missing_binary_fails(self):
        (self.build / "bin" / TARGET).unlink()
        with self.assertRaisesRegex(RuntimeError, "Required test executable was not built"):
            self.check()

    def test_missing_unit_label_fails(self):
        self.inventories["unit"].remove(TEST_NAME)
        with self.assertRaisesRegex(RuntimeError, "missing from CTest's unit tier"):
            self.check()

    def test_missing_performance_parity_label_fails(self):
        self.inventories["performance-parity"].remove(TEST_NAME)
        with self.assertRaisesRegex(RuntimeError, "missing performance-parity label"):
            self.check()

    def test_wrong_registered_command_fails(self):
        self.tests[0]["command"] = [str(self.build / "bin" / "test-arktts-cache")]
        with self.assertRaisesRegex(RuntimeError, "does not invoke required standalone executable"):
            self.check()

    def test_missing_full_registration_fails(self):
        self.tests.clear()
        with self.assertRaisesRegex(RuntimeError, "does not invoke required standalone executable"):
            self.check()


if __name__ == "__main__":
    unittest.main()
