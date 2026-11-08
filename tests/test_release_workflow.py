#!/usr/bin/env python3
"""Validate release.yml workflow targets (issue #131).

Checks:
  1. All Linux release jobs run on ubuntu-22.04 (glibc 2.35), not 24.04.
  2. A linux-x86_64-vulkan release job exists.
  3. YAML is syntactically valid.

Run:
  python tests/test_release_workflow.py
  pytest tests/test_release_workflow.py -v
"""

import os
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RELEASE_YML = REPO / ".github" / "workflows" / "release.yml"


def _load_yaml():
    """Load release.yml as a Python dict."""
    try:
        import yaml
    except ImportError:
        # Fallback: parse just what we need with a simple approach
        return None
    with open(RELEASE_YML) as f:
        return yaml.safe_load(f)


class TestReleaseWorkflow(unittest.TestCase):
    """Static tests for .github/workflows/release.yml."""

    @classmethod
    def setUpClass(cls):
        cls.data = _load_yaml()
        # Also load raw text for regex-based checks when pyyaml is unavailable
        cls.text = RELEASE_YML.read_text()

    def test_yaml_exists(self):
        self.assertTrue(RELEASE_YML.exists(), f"{RELEASE_YML} must exist")

    def test_yaml_parses(self):
        """release.yml must be valid YAML."""
        if self.data is None:
            self.skipTest("pyyaml not installed")
        self.assertIn("jobs", self.data)

    def test_linux_jobs_target_ubuntu_22_04(self):
        """All Linux x86_64 jobs must use ubuntu-22.04, not ubuntu-latest or 24.04."""
        linux_x86_jobs = [
            "build-linux",
            "build-linux-x86_64-avx512",
            "build-linux-x86_64-cuda",
            "build-python-linux",
            "build-libs-linux-x86_64",
            "build-libs-linux-x86_64-avx512",
            "build-libs-linux-x86_64-cuda",
        ]
        if self.data is None:
            self.skipTest("pyyaml not installed")
        jobs = self.data.get("jobs", {})
        for job_name in linux_x86_jobs:
            with self.subTest(job=job_name):
                self.assertIn(job_name, jobs, f"Job {job_name} must exist")
                runner = jobs[job_name].get("runs-on", "")
                self.assertEqual(
                    runner,
                    "ubuntu-22.04",
                    f"{job_name} must run on ubuntu-22.04, got '{runner}'",
                )

    def test_linux_arm64_jobs_target_ubuntu_22_04(self):
        """All Linux ARM64 jobs must use ubuntu-22.04-arm."""
        arm_jobs = [
            "build-linux-arm64",
            "build-python-linux-arm64",
            "build-libs-linux-arm64",
        ]
        if self.data is None:
            self.skipTest("pyyaml not installed")
        jobs = self.data.get("jobs", {})
        for job_name in arm_jobs:
            with self.subTest(job=job_name):
                self.assertIn(job_name, jobs, f"Job {job_name} must exist")
                runner = jobs[job_name].get("runs-on", "")
                self.assertEqual(
                    runner,
                    "ubuntu-22.04-arm",
                    f"{job_name} must run on ubuntu-22.04-arm, got '{runner}'",
                )

    def test_no_ubuntu_latest_in_linux_jobs(self):
        """No Linux release job should use 'ubuntu-latest' (pins to 24.04)."""
        if self.data is None:
            self.skipTest("pyyaml not installed")
        jobs = self.data.get("jobs", {})
        for name, job in jobs.items():
            if "linux" not in name.lower():
                continue
            runner = job.get("runs-on", "")
            with self.subTest(job=name):
                self.assertNotEqual(
                    runner,
                    "ubuntu-latest",
                    f"{name} must not use ubuntu-latest (glibc too new)",
                )

    def test_vulkan_linux_job_exists(self):
        """A linux-vulkan release job must exist."""
        if self.data is None:
            # Fallback: text search
            self.assertIn("build-linux-x86_64-vulkan", self.text)
            return
        jobs = self.data.get("jobs", {})
        self.assertIn(
            "build-linux-x86_64-vulkan",
            jobs,
            "build-linux-x86_64-vulkan job must exist in release.yml",
        )

    def test_vulkan_linux_job_enables_vulkan(self):
        """The Vulkan job must pass -DGGML_VULKAN=ON."""
        if self.data is None:
            self.skipTest("pyyaml not installed")
        jobs = self.data.get("jobs", {})
        job = jobs.get("build-linux-x86_64-vulkan", {})
        steps_text = str(job.get("steps", []))
        self.assertIn(
            "GGML_VULKAN=ON",
            steps_text,
            "Vulkan job must enable GGML_VULKAN",
        )

    def test_vulkan_linux_job_produces_tarball(self):
        """The Vulkan job must produce a .tar.gz artifact."""
        if self.data is None:
            self.skipTest("pyyaml not installed")
        jobs = self.data.get("jobs", {})
        job = jobs.get("build-linux-x86_64-vulkan", {})
        steps_text = str(job.get("steps", []))
        self.assertIn(
            "crispasr-linux-x86_64-vulkan.tar.gz",
            steps_text,
            "Vulkan job must produce crispasr-linux-x86_64-vulkan.tar.gz",
        )

    def test_windows_cuda_publishes_complete_and_split_assets(self):
        """Windows CUDA releases must support both install styles (#342)."""
        if self.data is None:
            self.skipTest("pyyaml not installed")
        job = self.data.get("jobs", {}).get("build-windows-cuda", {})
        self.assertTrue(job, "build-windows-cuda job must exist")
        rows = job.get("strategy", {}).get("matrix", {}).get("include", [])
        self.assertEqual({row["flavor"] for row in rows}, {"cuda", "cuda126"})
        library = self.data["jobs"]["build-libs-windows-x86_64-cuda"]
        self.assertEqual(rows, library["strategy"]["matrix"]["include"])
        steps = job.get("steps", [])
        attachments = [
            step["with"]["files"]
            for step in steps
            if step.get("uses", "").startswith("softprops/action-gh-release@")
        ]
        self.assertEqual(len(attachments), 1)
        published = set()
        for row in rows:
            flavor = row["flavor"]
            files = attachments[0].replace("${{ matrix.flavor }}", flavor).splitlines()
            expected = {
                f"release/crispasr-windows-x86_64-{flavor}.zip",
                f"release/crispasr-windows-x86_64-{flavor}-non-cuda.zip",
                f"release/{flavor}-runtime.zip",
                f"release/crispasr-windows-x86_64-{flavor}-runtime-sha256.txt",
            }
            with self.subTest(flavor=flavor):
                self.assertTrue(expected.issubset(files), expected - set(files))
                self.assertFalse(expected & published, "flavors must not overwrite assets")
            published.update(expected)
        steps_text = str(steps)
        for asset in ("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll"):
            with self.subTest(asset=asset):
                self.assertIn(asset, steps_text)
        self.assertIn(
            "Expected exactly three CUDA runtime DLLs",
            steps_text,
            "split packaging must fail if the CUDA runtime set is incomplete",
        )

    def test_gpu_artifacts_keep_optimized_cpu_and_legacy_stays_portable(self):
        """#302's atomic ggml fix must not slow every prebuilt GPU artifact."""
        gpu_jobs = (
            "build-linux-x86_64-cuda",
            "build-linux-x86_64-cuda13",
            "build-linux-x86_64-hip",
            "build-linux-x86_64-hip7",
            "build-libs-linux-x86_64-hip",
            "build-linux-x86_64-vulkan",
            "build-libs-windows-x86_64-vulkan",
            "build-libs-linux-x86_64-cuda",
            "build-libs-windows-x86_64-cuda",
            "build-windows-cuda",
            "build-windows-vulkan",
        )
        for job_name in gpu_jobs:
            with self.subTest(job=job_name):
                match = re.search(
                    rf"(?ms)^  {re.escape(job_name)}:\n"
                    rf"(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)",
                    self.text,
                )
                self.assertIsNotNone(match, f"release job {job_name} must exist")
                self.assertNotIn(
                    "-DCRISPASR_PORTABLE_CPU=ON",
                    match.group("body"),
                    f"{job_name} must retain its optimized CPU helper",
                )

        for job_name in (
            "build-libs-windows-x86_64-cpu-legacy",
            "build-windows-cpu-legacy",
        ):
            match = re.search(
                rf"(?ms)^  {re.escape(job_name)}:\n"
                rf"(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)",
                self.text,
            )
            self.assertIsNotNone(match, f"release job {job_name} must exist")
            self.assertIn("-DCRISPASR_PORTABLE_CPU=ON", match.group("body"))

        portable_docker = (REPO / ".devops" / "main-cuda-portable.Dockerfile").read_text()
        self.assertIn("-DCRISPASR_PORTABLE_CPU=ON", portable_docker)
        cuda_smoke = (REPO / ".github" / "workflows" / "win-cuda-smoke.yml").read_text()
        self.assertNotIn("-DCRISPASR_PORTABLE_CPU=ON", cuda_smoke)
        self.assertIn("ggml-cpu ^", cuda_smoke)

    def test_ue4m3_lut_is_lazy_with_explicit_eager_ab_switch(self):
        cpu_source = (REPO / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c").read_text()
        startup = cpu_source[cpu_source.index("void ggml_cpu_init(void)") :]
        self.assertIn('getenv("GGML_CPU_EAGER_UE4M3_LUT")', startup)
        self.assertIn("ggml_cpu_init_ue4m3_table_unlocked();", startup)

        for arch in ("x86", "arm"):
            quants = (REPO / "ggml" / "src" / "ggml-cpu" / "arch" / arch / "quants.c").read_text()
            nvfp4 = quants[quants.index("void ggml_vec_dot_nvfp4_q8_0") :]
            self.assertIn("ggml_cpu_ensure_ue4m3_table();", nvfp4)


if __name__ == "__main__":
    unittest.main()
