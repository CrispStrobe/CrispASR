#!/usr/bin/env python3
"""Run existing ggml CPU-reference operator tests with non-vacuous acceptance.

Intended for contributor Ascend hardware; --backend CPU also permits testing
the wrapper on hosted runners. This validates operators, not model inference,
speech quality, quantized-model acceptance or performance. No tests/tolerances
inside ggml are changed. Full logs and incremental receipts survive failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def source_revision(directory):
    # An uninitialized submodule directory otherwise resolves the parent's
    # checkout and falsely records the harness commit as the ggml revision.
    if not (Path(directory) / ".git").exists():
        return None
    result = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def assess(log, backend, operator, exit_code):
    """Require observed target execution, real case counts and upstream success."""
    clean = ANSI.sub("", log)
    summaries = re.findall(r"^\s+(\d+)/(\d+) tests passed\s*$", clean, re.MULTILINE)
    cases = re.findall(r"^\s+" + re.escape(operator) + r"\(.*\):\s+OK\s*$", clean, re.MULTILINE)
    executed = bool(re.search(r"^\s+Backend " + re.escape(backend) + r": OK\s*$", clean, re.MULTILINE))
    counts_valid = len(summaries) == 1 and int(summaries[0][1]) > 0
    counts_pass = counts_valid and summaries[0][0] == summaries[0][1]
    return {
        "exit": exit_code,
        "selected_backend_completed": executed,
        "cases_passed": int(summaries[0][0]) if len(summaries) == 1 else None,
        "cases_run": int(summaries[0][1]) if len(summaries) == 1 else None,
        "observed_operator_cases": len(cases),
        "unsupported_cases": len(re.findall(r"^\s+" + re.escape(operator) + r"\(.*\):.*not supported", clean, re.MULTILINE)),
        "pass": exit_code == 0 and executed and bool(cases) and bool(counts_pass),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--backend", default="CANN0")
    parser.add_argument("--ops", default="ADD,RMS_NORM,SOFT_MAX,ROPE,MUL_MAT")
    parser.add_argument("--params", help="Optional ggml parameter regex; zero matching cases fail")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", default=600, type=int, help="Seconds per operator")
    args = parser.parse_args()
    operators = args.ops.split(",")
    if (args.timeout <= 0 or len(set(operators)) != len(operators)
            or any(not re.fullmatch(r"[A-Z][A-Z0-9_]*", name) for name in operators)):
        parser.error("Use unique exact operator names and a positive timeout")
    binary = args.binary.resolve(strict=True)
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {"harness_source": source_revision(ROOT), "checkout_ggml_source": source_revision(ROOT / "ggml"),
               "binary_sha256": digest(binary), "system": platform.system(),
               "architecture": platform.machine(), "backend": args.backend,
               "scope": "Existing ggml operator correctness against CPU reference",
               "model_inference_validated": False, "quantized_model_quality_validated": False,
               "performance_validated": False, "runs": [], "pass": False}
    receipt_path = args.out / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    for operator in operators:
        command = [str(binary), "test", "-b", args.backend, "-o", operator, "-j", "1"]
        if args.params is not None:
            command += ["-p", args.params]
        path = args.out / f"{operator.lower()}.log"
        start = time.monotonic()
        with path.open("w") as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=args.timeout)
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = 124
                log.write(f"\nWRAPPER_TIMEOUT after {args.timeout} seconds\n")
            except OSError as error:
                code = 127
                log.write(f"\nWRAPPER_EXECUTION_ERROR: {error}\n")
        result = assess(path.read_text(errors="replace"), args.backend, operator, code)
        result.update(operator=operator, command=command, log_sha256=digest(path),
                      elapsed_seconds=time.monotonic() - start)
        receipt["runs"].append(result)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"{operator}: {'PASS' if result['pass'] else 'FAIL'}; "
              f"cases={result['cases_passed']}/{result['cases_run']}; exit={code}", flush=True)
    receipt["pass"] = all(result["pass"] for result in receipt["runs"])
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
