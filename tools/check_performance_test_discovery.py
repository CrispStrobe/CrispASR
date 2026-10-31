#!/usr/bin/env python3
"""Require the actual optimization regression cases in CTest's unit tier.

Run after building the named executables. Compare Catch's own case inventory
with CTest's selected names; merely finding an executable/CTest placeholder
does not establish that its TEST_SPEC matched any tests.
Standalone tests must have a built executable and the expected CTest command.
"""

import argparse
import json
from pathlib import Path
import subprocess


AB_TARGETS = {
    "test-align-ab-bench": 1,
    "test-conformer-ab-bench": 1,
    "test-cpu-ops-ab-bench": 3,
    "test-firered-ab-bench": 4,
    "test-voxcpm2-ab-bench": 3,
    "test-kokoro-diarize-ab-bench": 3,
    "test-cqt-diarize-ab-bench": 3,
    "test-rvq-istft-ab-bench": 2,
}

STANDALONE_TESTS = {"test-omnivoice-encoder": "omnivoice-encoder-parity"}


def find_binary(build_dir, target, config):
    for parent in (build_dir / "bin" / config, build_dir / "bin"):
        for suffix in ("", ".exe"):
            candidate = parent / (target + suffix)
            if candidate.is_file():
                return candidate
    raise RuntimeError(f"Required test executable was not built: {target}")


def inventory(build_dir, ctest, config, label):
    result = subprocess.run([ctest, "--test-dir", str(build_dir), "-C", config,
                             "--show-only=json-v1", "-L", label, "-E", "live"],
                            text=True, encoding="utf-8", capture_output=True, check=True)
    return {test["name"] for test in json.loads(result.stdout)["tests"]}


def catch_names(binary, spec):
    listed = subprocess.run([str(binary), spec, "--list-tests", "--reporter", "json"],
                            text=True, encoding="utf-8", capture_output=True, check=True)
    return [test["name"] for test in json.loads(listed.stdout)["listings"]["tests"]]


def check(build_dir, ctest="ctest", config="Release", require_blas=False, require_openmp=False):
    build_dir = Path(build_dir).resolve()
    unit_names = inventory(build_dir, ctest, config, "unit")
    bench_names = inventory(build_dir, ctest, config, "bench")
    parity_names = inventory(build_dir, ctest, config, "performance-parity")
    required = {
        "core_lfr: stacking matches FunASR padding and striding contract",
        "prefix_beam_search: basic emissions and repeat collapsing",
        "istft-bounded-memory",
        "benchmark-mel-ab-controls",
        "performance-test-discovery-controls",
    }
    checked = {}
    standalone_binaries = {}
    for target, name in STANDALONE_TESTS.items():
        standalone_binaries[name] = find_binary(build_dir, target, config)
        required.add(name)
        checked[target] = 1
    for target, minimum in AB_TARGETS.items():
        binary = find_binary(build_dir, target, config)
        names = catch_names(binary, "[unit]")
        if len(names) < minimum:
            raise RuntimeError(f"{target}: only {len(names)} numerical cases; expected at least {minimum}")
        required.update("unit::" + name for name in names)
        checked[target] = len(names)
        timed_names = catch_names(binary, "[ab-bench]")
        if not timed_names:
            raise RuntimeError(f"{target}: no benchmark cases discovered")
        missing_timed = {"perf::" + name for name in timed_names} - bench_names
        if missing_timed:
            raise RuntimeError(f"{target}: missing benchmark registration: {sorted(missing_timed)}")

    for target, prefix in (("test-arktts-cache", ""), ("test-meanvc2-conditioning", ""),
                           ("test-voxcpm2-cpu-conv", ""),
                           *(([("test-core-istft-blas", "blas::")] if require_blas else [])),
                           *(([("test-parallel-for-openmp", "openmp::")] if require_openmp else []))):
        binary = find_binary(build_dir, target, config)
        spec = "[parallel]" if target == "test-parallel-for-openmp" else "[unit]"
        names = catch_names(binary, spec)
        if not names:
            raise RuntimeError(f"{target}: no required regression cases discovered")
        required.update(prefix + name for name in names)
        checked[target] = len(names)
    if require_blas:
        required.add("istft-bounded-memory-blas")
    missing = required - unit_names
    if missing:
        raise RuntimeError(f"Required numerical regressions missing from CTest's unit tier: {sorted(missing)}")
    missing = required - parity_names
    if missing:
        raise RuntimeError(f"Required regressions missing performance-parity label: {sorted(missing)}")
    if any(name.startswith("perf::") for name in unit_names):
        raise RuntimeError("Optional repeated timings unexpectedly selected by the unit tier")
    result = subprocess.run([ctest, "--test-dir", str(build_dir), "-C", config, "--show-only=json-v1"],
                            text=True, encoding="utf-8", capture_output=True, check=True)
    tests = {test["name"]: test for test in json.loads(result.stdout)["tests"]}
    for name, binary in standalone_binaries.items():
        command = tests.get(name, {}).get("command", [])
        if not command or Path(command[0]).resolve() != binary.resolve():
            raise RuntimeError(f"{name}: CTest does not invoke required standalone executable: {binary.name}")
    ab_names = {name for name in required if name.startswith("unit::")}
    for test in tests.values():
        name = test["name"]
        unit_equivalent = "unit::" + name.removeprefix("perf::") if name.startswith("perf::") else name
        if unit_equivalent in ab_names:
            expected = "CRISPASR_RUN_PERF=" + ("1" if name.startswith("perf::") else "0")
            properties = {prop["name"]: prop["value"] for prop in test.get("properties", [])}
            if expected not in properties.get("ENVIRONMENT", []):
                raise RuntimeError(f"{name}: required timing environment is missing: {expected}")
    return {"status": "passed", "required_unit_cases": len(required), "executables": checked,
            "blas_required": require_blas, "openmp_required": require_openmp}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--ctest", default="ctest")
    parser.add_argument("--config", default="Release")
    parser.add_argument("--require-blas", action="store_true")
    parser.add_argument("--require-openmp", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check(args.build_dir, args.ctest, args.config, args.require_blas, args.require_openmp), indent=2))


if __name__ == "__main__":
    main()
