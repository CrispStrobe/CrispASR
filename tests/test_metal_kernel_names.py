#!/usr/bin/env python3
"""Every Metal kernel the host code names literally must exist in a .metal file.

Issue #568: ggml-metal-ops.cpp routes Q4_K/Q8_0 x F32 mul_mats that carry
GGML_PREC_F32 to bespoke kernels looked up by string. When the fork split
ggml-metal.metal into kernels/*.metal, four of those kernels were dropped.
Nothing failed at build time: the lookup returns a nil pipeline at run time,
only on GPUs with SIMD-group reduction (every M-series Mac). Hosted macOS
runners expose an Apple Paravirtual GPU without it, so MUL_MAT never reaches
Metal there and CI could not notice. This check runs anywhere, without a GPU.
"""
import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
METAL = ROOT / "ggml" / "src" / "ggml-metal"

LITERAL = re.compile(r'"(kernel_[A-Za-z0-9_]+)"')


def missing_kernels(host_sources, kernel_sources):
    """Return literal kernel names in host_sources not defined in kernel_sources."""
    defined = set(re.findall(r'host_name\("(kernel_[A-Za-z0-9_]+)"\)', kernel_sources))
    defined |= set(re.findall(r"\bkernel\s+void\s+(kernel_[A-Za-z0-9_]+)\s*\(", kernel_sources))
    return sorted(set(LITERAL.findall(host_sources)) - defined)


def read_tree():
    host = "".join(p.read_text(errors="replace") for p in sorted(METAL.glob("*")) if p.suffix in (".cpp", ".m"))
    kernels = "".join(p.read_text(errors="replace") for p in sorted(METAL.rglob("*.metal")))
    return host, kernels


class MetalKernelNames(unittest.TestCase):
    def setUp(self):
        if not METAL.is_dir():
            self.skipTest("ggml submodule not checked out")
        self.host, self.kernels = read_tree()

    def test_tree_is_scanned(self):
        # Guard against a silently empty scan making the main check vacuous.
        self.assertGreater(len(LITERAL.findall(self.host)), 4)
        self.assertIn("kernel_mul_mv_q4_K_q8_K", self.host)

    def test_every_literal_kernel_is_defined(self):
        self.assertEqual(missing_kernels(self.host, self.kernels), [])

    def test_check_detects_a_missing_kernel(self):
        # Positive control: the check must fire on exactly the #568 shape.
        host = self.host + '\nget_pipeline(lib, "kernel_issue568_not_defined");\n'
        self.assertEqual(missing_kernels(host, self.kernels), ["kernel_issue568_not_defined"])
        stripped = self.kernels.replace('host_name("kernel_mul_mv_q4_K_q8_K")', "").replace(
            "kernel void kernel_mul_mv_q4_K_q8_K(", "")
        self.assertEqual(missing_kernels(self.host, stripped), ["kernel_mul_mv_q4_K_q8_K"])


if __name__ == "__main__":
    sys.exit(unittest.main())
