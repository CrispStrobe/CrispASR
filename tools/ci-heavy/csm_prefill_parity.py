#!/usr/bin/env python3
"""Convert -> actual crispasr-quantize -> original-reference/native prefill.

Hosted CPU only. Requires a successful original F32 producer run and its full
source SHA. Keeps weights/builds in scratch, all stage arrays and inventories
in the artifact. This never publishes a model or changes production defaults.
Dependencies: numpy torch==2.6.0 gguf==0.17.1 safetensors huggingface_hub.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

import csm_original_rope_reference as original
import csm_rope_component_check as metrics
import nightly_error_ab as ab


ROOT = Path(__file__).resolve().parents[2]
# Fixed before observing any candidate full-model output. These are F16 port
# gates, not relaxed Q4 speech gates and not a claim of F32 bit identity.
F16_GATES = {"rel_l2": 0.005, "norm_ratio_delta": 0.005, "cosine_min": 0.99999, "max_abs": 0.05}
MODEL_SHA = "1123f9e8ff2d9212700d1fface749b8ec4f9765569bb996667a15a9a9832ee13"
# Inventory of the existing release: 439/442 tensors already reproduce under
# bare Q4_K; its three embedding tables are Q8_0. Reproduce that published
# recipe exactly, without choosing precision from speech outcomes.
PUBLISHED_EMBEDDING_RULE = r"^(backbone\.(audio|text)_embd|depth\.token_embd)\.weight$=q8_0"


def inventory(path):
    from gguf import GGUFReader
    reader = GGUFReader(path)
    result = {}
    for tensor in reader.tensors:
        shape = list(map(int, tensor.shape))
        while len(shape) > 1 and shape[-1] == 1:
            shape.pop()
        result[tensor.name] = {"type": tensor.tensor_type.name, "shape": shape,
                               "sha256": hashlib.sha256(tensor.data).hexdigest()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-run", required=True, type=int)
    parser.add_argument("--producer-source", required=True)
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("model conversion, build and inference belong on hosted Actions")
    if len(args.producer_source) != 40 or any(c not in "0123456789abcdef" for c in args.producer_source):
        raise RuntimeError("producer source must be a full immutable commit")
    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "csm-prefill-parity"
    scratch.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    receipt = {"complete": False, "acceptance": False, "scope": __doc__,
               "source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "producer_run": args.producer_run, "producer_source": args.producer_source,
               "f16_gates": F16_GATES, "stages": {}, "models": {}, "comparisons": {}}

    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def run(argv, label, timeout=1800):
        state = ab.snapshot()
        receipt["stages"][label] = {"resources": state}
        save()
        ab.require_headroom(state)
        if int(state["memory"]["MemAvailable"].split()[0]) < 8 * 1024**2:
            raise RuntimeError(f"8 GiB available required before {label}")
        ab.required(argv, ROOT, out, label, timeout)
        receipt["stages"][label]["pass"] = True
        save()

    save()
    try:
        meta = json.loads(subprocess.check_output([
            "gh", "api", f"repos/CrispStrobe/CrispASR/actions/runs/{args.producer_run}"], text=True))
        if (meta["status"] != "completed" or meta["conclusion"] != "success" or
                meta["head_sha"] != args.producer_source or meta["event"] != "workflow_dispatch"):
            raise RuntimeError("original F32 producer must succeed at its exact pinned source")
        reference = out / "original-prefill"
        run(["gh", "run", "download", args.producer_run, "--repo", "CrispStrobe/CrispASR",
             "--name", f"heavy-{args.producer_run}", "--dir", reference], "download-reference", 300)
        ref = json.loads((reference / "receipt.json").read_text())
        if (ref.get("complete") is not True or ref.get("source_revision") != original.REVISION
                or ref.get("config_sha256") != original.CONFIG_SHA256):
            raise RuntimeError("original F32 reference is incomplete or pins differ")
        from huggingface_hub import snapshot_download
        ab.require_headroom(ab.snapshot())
        weights = list(ref["weight_files"])
        model_dir = Path(snapshot_download("sesame/csm-1b", revision=original.CONFIG_REVISION,
                                          allow_patterns=[*weights, "*.json", "tokenizer*", "*.model"]))
        for name, expected in ref["weight_files"].items():
            if ab.sha256(model_dir / name) != expected["sha256"]:
                raise RuntimeError(f"conversion coefficients differ from original reference: {name}")
        run(["bash", ROOT / "tools/ci-apt.sh", "update"], "apt-update", 600)
        run(["bash", ROOT / "tools/ci-apt.sh", "install", "-y", "libopenblas-dev", "libespeak-ng-dev",
             "espeak-ng", "pkg-config", "ccache"], "apt-install", 600)
        build = scratch / "build-avx2"
        from csm_cpu_profile_ab import PROFILES
        run(["cmake", "-S", ROOT, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
             "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
             "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache", "-DCRISPASR_BUILD_TESTS=OFF",
             "-DCRISPASR_BUILD_SERVER=OFF", *PROFILES["avx2"]], "configure")
        run(["cmake", "--build", build, "--target", "crispasr-cli", "crispasr-quantize", "-j4"], "build")
        shutil.copy2(build / "compile_commands.json", out / "compile_commands.json")
        f16 = scratch / "csm-original-f16.gguf"
        run([sys.executable, ROOT / "models/convert-csm-to-gguf.py", "--input", model_dir,
             "--output", f16, "--outtype", "f16"], "convert-f16")
        fresh_q4 = scratch / "csm-fresh-q4_k.gguf"
        receipt["quantize_recipe"] = {"body": "q4_k", "tensor_type": [PUBLISHED_EMBEDDING_RULE],
                                       "scope": "Exact reproduction of existing published tensor types; no precision tuning"}
        save()
        run([build / "bin/crispasr-quantize", f16, fresh_q4, "q4_k", "--tensor-type", PUBLISHED_EMBEDDING_RULE],
            "quantize-q4", 1200)
        published = ab.regression.hf_download("cstr/csm-1b-GGUF", "csm-1b-q4_k.gguf",
                                               "5abf936b9d17a839fca69e09baf31e0ea5068f97", scratch)
        if ab.sha256(published) != MODEL_SHA:
            raise RuntimeError("published Q4 bytes differ from retained failure model")
        inventories = {}
        for name, model in (("f16", f16), ("fresh-q4", fresh_q4), ("published-q4", published)):
            inventories[name] = inventory(model)
            (out / f"{name}-inventory.json").write_text(json.dumps(inventories[name], indent=2) + "\n")
            receipt["models"][name] = {"sha256": ab.sha256(model), "bytes": model.stat().st_size,
                                        "inventory_sha256": ab.sha256(out / f"{name}-inventory.json")}
        receipt["published_q4_payloads_match_fresh_quantization"] = inventories["fresh-q4"] == inventories["published-q4"]
        if not any(tensor["type"] == "Q4_K" for tensor in inventories["fresh-q4"].values()):
            raise RuntimeError("actual quantizer produced no Q4_K tensors")
        save()
        for kind, path in (("f16", f16), ("fresh-q4", fresh_q4), ("published-q4", published)):
            for rope, attention in (("legacy", "default"), ("original", "default"), ("original", "eager-f32")):
                label = f"{kind}-{rope}-{attention}"
                stage = out / label
                run([sys.executable, ROOT / "tools/ci-heavy/csm_native_prefill.py",
                     "--library", build / "src/libcrispasr.so", "--model", path,
                     "--model-sha256", receipt["models"][kind]["sha256"], "--reference", reference,
                     "--out", stage, "--rope", rope, "--attention", attention], f"prefill-{label}", 300)
                native = json.loads((stage / "receipt.json").read_text())
                comparisons = {}
                for name, entry in ref["files"].items():
                    expected = metrics._read_array(reference / f"{name}.bin", tuple(entry["shape"]),
                                                   "float32", digest=entry["raw_sha256"])
                    actual_entry = native["files"][name]
                    actual = metrics._read_array(stage / f"{name}.bin", tuple(entry["shape"]),
                                                 "float32", digest=actual_entry["raw_sha256"])
                    result = metrics._metrics(expected, actual)
                    result["f16_port_gate"] = (result["rel_l2"] <= F16_GATES["rel_l2"] and
                                                abs(result["norm_ratio"] - 1) <= F16_GATES["norm_ratio_delta"] and
                                                result["cosine"] >= F16_GATES["cosine_min"] and
                                                result["max_abs"] <= F16_GATES["max_abs"])
                    comparisons[name] = result
                receipt["comparisons"][label] = comparisons
                save()
        receipt["f16_original_parity"] = all(
            value["f16_port_gate"] for attention in ("default", "eager-f32")
            for value in receipt["comparisons"][f"f16-original-{attention}"].values())
        receipt["complete"] = True
        receipt["acceptance"] = receipt["f16_original_parity"]
        save()
        return 0 if receipt["acceptance"] else 1
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    sys.exit(main())
