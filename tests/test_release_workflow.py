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


if __name__ == "__main__":
    unittest.main()
