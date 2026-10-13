#!/usr/bin/env python3
"""Verify the hardware wrapper against real ggml CPU positive/empty probes."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ["HEAVY_OUT"])
SCRATCH = Path(os.environ["HEAVY_SCRATCH"])
OUT.mkdir(parents=True, exist_ok=True)
build = SCRATCH / "ggml-ops-build"


def run(command, name):
    with (OUT / f"{name}.log").open("w") as log:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT, timeout=1800)
    print(name, result.returncode, flush=True)
    return result.returncode


assert run(["cmake", "-S", ROOT / "ggml", "-B", build, "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_BUILD_TESTS=ON", "-DGGML_BUILD_EXAMPLES=OFF", "-DGGML_NATIVE=OFF",
            "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DGGML_METAL=OFF", "-DGGML_BLAS=OFF"], "configure") == 0
assert run(["cmake", "--build", build, "--target", "test-backend-ops", "-j4"], "build") == 0
binary = build / "bin/test-backend-ops"
# Existing tiny F32 broadcast case in the pinned upstream test suite.
params = r"^type=f32,ne=\[1,1,1,1\],nr=\[32,1,1,1\],nf=1,perm1=0,src_overlap=0$"
controls = [
    ("positive", "CPU", "ADD", params, True),
    ("missing-backend", "CANN_MISSING_NEGATIVE_CONTROL", "ADD", params, False),
    ("zero-cases", "CPU", "ADD", "NO_PARAMETERS_CAN_MATCH_THIS", False),
    ("unknown-operator", "CPU", "UNKNOWN_OPERATOR_NEGATIVE_CONTROL", params, False),
]
results = []
for name, backend, operator, pattern, expected in controls:
    raw = run([binary, "test", "-b", backend, "-o", operator, "-p", pattern, "-j", "1"], f"{name}-upstream")
    # Real upstream false positives: all three empty probes return success.
    assert raw == 0, name
    code = run([sys.executable, ROOT / "tools/validate_backend_ops.py", "--binary", binary,
                "--backend", backend, "--ops", operator, "--params", pattern,
                "--out", OUT / name], f"{name}-wrapper")
    receipt = json.loads((OUT / name / "receipt.json").read_text())
    assert (code == 0) == expected and receipt["pass"] == expected, name
    if expected:
        assert receipt["runs"][0]["cases_run"] == 1, receipt
    results.append({"control": name, "upstream_exit": raw, "wrapper_exit": code,
                    "expected_acceptance": expected, "receipt": receipt})
(OUT / "controls.json").write_text(json.dumps(results, indent=2) + "\n")
print("BACKEND_OPS_WRAPPER_REAL_BINARY_PASS: 1 positive, 3 real false-success controls; no Ascend inference claim", flush=True)
