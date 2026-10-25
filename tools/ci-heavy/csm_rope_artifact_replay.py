#!/usr/bin/env python3
"""Recompile a bounded CSM rotary probe against retained, hashed CPU GGML.

No model weights, full builds or synthesis. Reuses retained AVX2 libraries by
default; native-library replay additionally requires an eligible AVX512 host.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import csm_cpu_profile_ab as profile
import csm_rope_component_run as component
import nightly_error_ab as ab


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-run", required=True, type=int)
    parser.add_argument("--producer-source", required=True)
    parser.add_argument("--include-native", action="store_true")
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("component replay belongs on hosted Actions")
    if not re.fullmatch("[0-9a-f]{40}", args.producer_source):
        raise RuntimeError("producer source must be immutable")
    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "csm-rope-replay"
    scratch.mkdir(parents=True, exist_ok=True)
    receipt = {"complete": False, "acceptance": False, "producer_run": args.producer_run,
               "producer_source": args.producer_source, "scope": __doc__, "profiles": {}}

    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    try:
        state = ab.snapshot()
        ab.require_headroom(state)
        meta = json.loads(subprocess.check_output([
            "gh", "api", f"repos/CrispStrobe/CrispASR/actions/runs/{args.producer_run}"], text=True))
        # Expected failed legacy speech cells can make the completed producer
        # red. Actual component success is checked separately below.
        if meta["status"] != "completed" or meta["head_sha"] != args.producer_source:
            raise RuntimeError("component producer has not completed at the pinned worker source")
        artifact = scratch / "producer"
        ab.required(["gh", "run", "download", args.producer_run, "--repo", "CrispStrobe/CrispASR",
                     "--name", f"heavy-{args.producer_run}", "--dir", artifact], ROOT, out, "download", 300)
        producer = json.loads((artifact / "receipt.json").read_text())
        if not re.fullmatch("[0-9a-f]{40}", producer["source"]):
            raise RuntimeError("retained native source is not immutable")
        ab.required(["git", "fetch", "--depth=1", "origin", producer["source"]], ROOT, out, "fetch-source", 300)
        features = Path("/proc/cpuinfo").read_bytes()
        if not all(re.search(rb"\b" + flag + rb"\b", features)
                   for flag in (b"avx2", b"fma", b"f16c", b"bmi2", b"sse4_2")):
            raise RuntimeError("retained AVX2 library requires AVX2/FMA/F16C/BMI2/SSE4.2")
        if args.include_native and not re.search(rb"\bavx512f\b", features):
            raise RuntimeError("native library replay requires AVX512 before compilation/execution")
        profiles = ("native", "avx2") if args.include_native else ("avx2",)
        receipt["requested_profiles"] = list(profiles)
        original = artifact / "original-rope-reference"
        if ab.sha256(original / "receipt.json") != profile.ROPE_REFERENCE_RECEIPT:
            raise RuntimeError("original reference artifact pin differs")
        reference = out / "original-rope-reference"
        shutil.copytree(original, reference)
        for filename in ("src/csm_rope.h", "tools/ci-heavy/csm_rope_probe.cpp"):
            content = subprocess.check_output(["git", "show", f"{producer['source']}:{filename}"], cwd=ROOT)
            if content != (ROOT / filename).read_bytes():
                raise RuntimeError(f"component source differs from original experiment: {filename}")
        gitlink = subprocess.check_output(["git", "ls-tree", producer["source"], "ggml"], cwd=ROOT, text=True)
        current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "ggml", text=True).strip()
        if current not in gitlink:
            raise RuntimeError("GGML source differs from retained shared libraries")
        receipt["producer_receipt_sha256"] = ab.sha256(artifact / "receipt.json")
        for name in profiles:
            before = json.loads((artifact / name / "rope-component/receipt.json").read_text())
            if before.get("complete") is not True or before.get("pass") is not True:
                raise RuntimeError(f"{name}: original component did not pass")
            for stem, entry in before["ggml_libraries"].items():
                if ab.sha256(artifact / name / "rope-component/libs" / f"{stem}.so.0") != entry["sha256"]:
                    raise RuntimeError(f"{name}: retained GGML bytes differ")
            after = component.run(ROOT, artifact / name, name, reference, out / name)
            for mode in component.MODES:
                for case in component.check.CASES:
                    for tensor in ("q_rotated", "k_rotated", "freq_factors"):
                        result = out / name / mode / case / f"{tensor}.bin"
                        previous = artifact / name / "rope-component" / mode / case / f"{tensor}.bin"
                        if result.read_bytes() != previous.read_bytes():
                            raise RuntimeError(f"{name}/{mode}/{case}/{tensor}: recompiled probe changed output")
            receipt["profiles"][name] = {"pass": after["pass"], "retained_outputs_byte_identical": True,
                                          "receipt_sha256": ab.sha256(out / name / "receipt.json")}
            save()
        receipt["complete"] = receipt["acceptance"] = True
        save()
        return 0
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    sys.exit(main())
