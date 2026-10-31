#!/usr/bin/env python3
"""Opt-in, fail-closed qualification of supplied post-release model fixtures.

No downloads, reference rebakes, or accepted model results are supplied by this
tool. See docs/post-release-audit-validation.md for the adapter/fixture contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import unicodedata
import wave


REPO = Path(__file__).resolve().parents[2]
SCOPES = {
    "arktts": {
        "stages": {"embed_tokens", "slow_logits_last", "frame0_codebooks", "decoded_pcm"},
        "arrays": {"prefill_k", "prefill_v", "decode_logits", "decode_k", "decode_v", "waveform"},
        "texts": {"synthesized_text"}, "cuda": True,
    },
    "haas": {
        "stages": {"encoder_output_ref_mel", "decoder_sos", "joint_sos_t0"},
        "arrays": {"decoder_steps", "joint_steps"},
        "texts": {"cli_haas", "cli_haas_ja", "cli_haas_asr", "cabi_auto",
                  "cabi_haas", "cabi_haas_ja", "cabi_haas_asr"}, "cuda": True,
    },
    "meanvc2": {
        "stages": {"asr_bn", "gtm_k", "gtm_v", "dit_mel", "vocos_pcm"},
        "arrays": {"asr_bn", "asr_bn_interp", "dit_mel", "waveform", "speaker_embedding"},
        "texts": {"converted_text"}, "cuda": False,
    },
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Path escapes its declared directory: {relative}")
    return result


def require_revision(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"{label} must be an immutable, full 40-character revision")


def preflight(manifest_path, source_sha):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("Expected manifest version 1")
    require_revision(manifest.get("source_sha"), "source_sha")
    if manifest["source_sha"] != source_sha:
        raise ValueError("Manifest source_sha does not match the checked-out candidate")
    root = inside(manifest_path.parent, manifest.get("fixture_root", "."))
    files = {}
    if not manifest.get("files"):
        raise ValueError("A nonempty immutable file inventory is required")
    for key, spec in manifest["files"].items():
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", key):
            raise ValueError(f"Invalid file identifier: {key}")
        path = inside(root, spec["path"])
        expected = spec.get("sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or digest(path) != expected:
            raise ValueError(f"Missing or mismatched SHA-256: {key}")
        files[key] = path
    cases = manifest.get("cases", [])
    if {case.get("backend") for case in cases} != set(SCOPES):
        raise ValueError("Require nonempty ArkTTS, Haas, and MeanVC2 qualification cases")
    ids = set()
    for case in cases:
        name = case.get("id", "")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in ids:
            raise ValueError("Case IDs must be unique filesystem-safe names")
        ids.add(name)
        scope = SCOPES[case["backend"]]
        for key in ("model_revision", "reference_revision", "original_code_revision"):
            require_revision(case.get(key), name + "." + key)
        for key in ("model", "reference", "audio", "adapter", "reference_lock"):
            if case.get(key) not in files:
                raise ValueError(f"{name}: required pinned file missing: {key}")
        argv = case.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
            raise ValueError(f"{name}: argv must be a nonempty argument array, without shell expansion")
        adapter_arg = "{file:" + case["adapter"] + "}"
        if not (argv[0] == adapter_arg or argv[:2] == ["{python}", adapter_arg]):
            raise ValueError(f"{name}: argv must execute the pinned adapter")
        if "{out}" not in argv:
            raise ValueError(f"{name}: argv must explicitly receive the fresh output directory")
        if not scope["arrays"].issubset(case.get("arrays", {})):
            raise ValueError(f"{name}: required complete-pipeline array gates are missing")
        if not scope["texts"].issubset(case.get("texts", {})):
            raise ValueError(f"{name}: required intelligibility/dispatch transcripts are missing")
        if case["backend"] == "meanvc2" and case.get("speaker_embedding") not in files:
            raise ValueError(f"{name}: require a pinned embedding from the real speaker encoder")
        if case["backend"] == "meanvc2" and "{file:" + case["speaker_embedding"] + "}" not in argv:
            raise ValueError(f"{name}: adapter must receive the pinned speaker embedding")
        for role, spec in case["arrays"].items():
            if spec.get("reference") not in files:
                raise ValueError(f"{name}.{role}: reference array/waveform is not pinned")
            inside(root, spec["actual"])
            mode = spec.get("comparison", "numeric")
            keys = {"numeric": ("max_abs", "relative_l2_max", "cosine_min"),
                    "cosine": ("cosine_min",),
                    "waveform": ("duration_ratio_min", "duration_ratio_max",
                                 "rms_ratio_min", "rms_ratio_max", "peak_abs_max")}.get(mode)
            if keys is None or not all(isinstance(spec.get(k), (float, int)) and
                                       math.isfinite(spec[k]) for k in keys):
                raise ValueError(f"{name}.{role}: finite, explicit acceptance bounds are required")
            if mode == "waveform" and role != "waveform":
                raise ValueError("Waveform quality bounds cannot replace intermediate numerical gates")
            if role == "waveform" and mode == "cosine":
                raise ValueError("Waveforms require numerical parity or duration/amplitude bounds, not cosine alone")
            if role != "waveform" and role != "speaker_embedding" and mode != "numeric":
                raise ValueError("Intermediate tensors require all numerical bounds")
            if mode in {"numeric", "cosine"} and not 0 < spec["cosine_min"] <= 1:
                raise ValueError("A positive cosine bound in (0, 1] is required")
            if mode == "numeric" and min(spec["max_abs"], spec["relative_l2_max"]) < 0:
                raise ValueError("Numerical error bounds must be nonnegative")
            if mode == "waveform" and not (0 < spec["duration_ratio_min"] <= spec["duration_ratio_max"] and
                                            0 < spec["rms_ratio_min"] <= spec["rms_ratio_max"] and
                                            spec["peak_abs_max"] > 0):
                raise ValueError("Waveform ranges must be positive and ordered")
        for spec in case["texts"].values():
            if spec.get("reference") not in files:
                raise ValueError(f"{name}: expected transcript is not pinned")
            inside(root, spec["actual"])
    return manifest, files


def verify_diff(text, backend):
    rows = re.findall(r"^\[(PASS|FAIL|SKIP)\]\s+(\S+)", text, re.MULTILINE)
    passed = {name for state, name in rows if state == "PASS"}
    missing = SCOPES[backend]["stages"] - passed
    invalid = any(state == "FAIL" or (state == "SKIP" and name in SCOPES[backend]["stages"])
                  for state, name in rows)
    if missing or invalid:
        raise ValueError(f"{backend}: absent/failing required diff stages: {sorted(missing)}")
    return sorted(passed)


def cuda_kernel_instances(text):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        fields = next(csv.reader([line]), [])
        if "Instances" in fields and "Name" in fields:
            rows = list(csv.DictReader(io.StringIO("\n".join(lines[i:]))))
            count = sum(int(row["Instances"].replace(",", "")) for row in rows if row.get("Instances"))
            if count > 0:
                return count
    raise ValueError("CUDA qualification requires a nonempty nsys CUDA kernel trace")


def run_command(argv, out, stem, cuda=False):
    if cuda:
        if not shutil.which("nsys"):
            raise ValueError("nsys is required to prove actual CUDA kernel execution")
        argv = ["nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                "--force-overwrite=true", "--output=" + str(out / stem), *argv]
    start = time.monotonic()
    process = subprocess.Popen(argv, text=True, encoding="utf-8", errors="replace",
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=(os.name == "posix"))
    try:
        stdout, _ = process.communicate(timeout=3600)
    except subprocess.TimeoutExpired:
        # A timed-out profiler/adapter must not leave inference children using
        # the self-hosted worker's GPU after this job has failed.
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        stdout, _ = process.communicate()
        (out / (stem + ".log")).write_text(stdout, encoding="utf-8")
        raise RuntimeError(f"{stem} timed out after 3600 seconds")
    (out / (stem + ".log")).write_text(stdout, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"{stem} exited {process.returncode}; see {stem}.log")
    receipt = {"process_seconds": time.monotonic() - start, "exit_code": process.returncode}
    if cuda:
        report = subprocess.run(["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                                 str(out / (stem + ".nsys-rep"))], text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120, check=True)
        (out / (stem + "-cuda-kernels.csv")).write_text(report.stdout, encoding="utf-8")
        receipt["cuda_kernel_instances"] = cuda_kernel_instances(report.stdout)
    return stdout, receipt


def read_array(path):
    import numpy as np
    if Path(path).suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as audio:
            if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                raise ValueError("Waveform gates require mono PCM16 WAV or a float NPY array")
            data = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float64) / 32768
            rate = audio.getframerate()
    else:
        data, rate = np.load(path, allow_pickle=False), None
    if data.dtype.kind not in "fi" or not data.size or not np.isfinite(data).all():
        raise ValueError(f"Empty/non-numerical/non-finite capture: {Path(path).name}")
    return data.astype(np.float64), rate


def compare_array(actual_path, reference_path, spec):
    import numpy as np
    actual, rate = read_array(actual_path)
    expected, expected_rate = read_array(reference_path)
    mode = spec.get("comparison", "numeric")
    if mode == "waveform":
        if actual.ndim != 1 or expected.ndim != 1 or rate != expected_rate:
            raise ValueError("Waveform layout/sample rate mismatch")
        denom = float(np.sqrt(np.mean(expected * expected)))
        if denom == 0:
            raise ValueError("A silent reference cannot qualify generated speech")
        metrics = {"duration_ratio": actual.size / expected.size,
                   "rms_ratio": float(np.sqrt(np.mean(actual * actual))) / denom,
                   "peak_abs": float(np.abs(actual).max())}
        passed = (spec["duration_ratio_min"] <= metrics["duration_ratio"] <= spec["duration_ratio_max"] and
                  spec["rms_ratio_min"] <= metrics["rms_ratio"] <= spec["rms_ratio_max"] and
                  metrics["peak_abs"] <= spec["peak_abs_max"])
    else:
        if actual.shape != expected.shape or rate != expected_rate:
            raise ValueError("Reference/candidate shape or sample rate mismatch")
        a, b = actual.ravel(), expected.ravel()
        a_norm, b_norm = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if a_norm == 0 or b_norm == 0:
            if mode == "cosine":
                raise ValueError("Speaker similarity requires nonzero candidate and reference embeddings")
            cosine = 1.0 if np.array_equal(a, b) else 0.0
        else:
            cosine = float(np.dot(a, b) / (a_norm * b_norm))
        metrics = {"cosine": cosine, "max_abs": float(np.max(np.abs(a - b))),
                   "relative_l2": float(np.linalg.norm(a - b)) / max(b_norm, 1e-30)}
        passed = cosine >= spec["cosine_min"]
        if mode == "numeric":
            passed = passed and metrics["max_abs"] <= spec["max_abs"] and metrics["relative_l2"] <= spec["relative_l2_max"]
    if not passed:
        raise ValueError(f"Array quality gate failed: {metrics}")
    return {"shape": list(actual.shape), "comparison": mode, **metrics}


def normalized_text(path):
    return " ".join(unicodedata.normalize("NFC", Path(path).read_text(encoding="utf-8")).split())


def find_binary(build, name):
    for folder in (build / "bin", build / "bin" / "Release"):
        candidate = folder / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Required native executable was not built: {name}")


def find_library(build):
    for folder in (build / "bin", build / "src", build / "lib"):
        candidate = folder / "libcrispasr.so"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Required native shared library was not built: libcrispasr.so")


def run_cases(manifest, files, build, out):
    diff = find_binary(build, "crispasr-diff")
    receipts = []
    for case in manifest["cases"]:
        folder = out / case["id"]
        folder.mkdir()  # Existing output is an error: stale captures must not pass.
        backend = case["backend"]
        stdout, timing = run_command([str(diff), backend, str(files[case["model"]]),
                                      str(files[case["reference"]]), str(files[case["audio"]])],
                                     folder, "stages", SCOPES[backend]["cuda"])
        stages = verify_diff(stdout, backend)

        def expand(arg):
            if arg == "{python}":
                return sys.executable
            if arg == "{out}":
                return str(folder.resolve())
            if arg == "{library}":
                return str(find_library(build))
            match = re.fullmatch(r"\{(file|bin):([\w.-]+)\}", arg)
            if match:
                return str(files[match[2]] if match[1] == "file" else find_binary(build, match[2]))
            if "{" in arg or "}" in arg:
                raise ValueError("Unknown adapter placeholder")
            return arg

        _, native_timing = run_command([expand(arg) for arg in case["argv"]], folder, "pipeline",
                                       SCOPES[backend]["cuda"])
        arrays = {}
        for role, spec in case["arrays"].items():
            actual = inside(folder, spec["actual"])
            arrays[role] = compare_array(actual, files[spec["reference"]], spec)
            if role in {"decode_logits", "decode_k", "decode_v", "decoder_steps", "joint_steps"}:
                if len(arrays[role]["shape"]) < 2 or arrays[role]["shape"][0] < 3:
                    raise ValueError("Stateful qualification needs at least three actual decoded steps")
        if backend == "meanvc2":
            bn, interp = arrays["asr_bn"]["shape"], arrays["asr_bn_interp"]["shape"]
            if len(bn) != 2 or bn[1] != 256 or interp != [4 * bn[0], bn[1]]:
                raise ValueError("MeanVC2 requires native time-major BNF and its actual 4x interpolation")
        for spec in case["texts"].values():
            expected = normalized_text(files[spec["reference"]])
            if not expected or normalized_text(inside(folder, spec["actual"])) != expected:
                raise ValueError(f"{case['id']}: transcription/intelligibility gate failed")
        receipts.append({"id": case["id"], "backend": backend, "stages": stages,
                         "execution": "CUDA kernels traced" if SCOPES[backend]["cuda"] else "CPU host kernels",
                         "arrays": arrays, "transcript_gates": sorted(case["texts"]),
                         "stage_run": timing, "pipeline_run": native_timing})
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {"status": "failed", "cases": []}
    try:
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=REPO, check=True)
        receipt["source_sha"] = source
        manifest, files = preflight(args.manifest, source)
        receipt["manifest_sha256"] = digest(args.manifest)
        receipt["inputs"] = {key: manifest["files"][key]["sha256"] for key in files}
        if args.preflight_only:
            receipt["status"] = "inputs_verified_only"
        else:
            if args.build_dir is None:
                raise ValueError("--build-dir is required for model execution")
            import numpy as np
            receipt["comparison_environment"] = {"python": sys.version, "numpy": np.__version__}
            receipt["native_sha256"] = {name: digest(find_binary(args.build_dir, name))
                                        for name in ("crispasr", "crispasr-diff", "test-meanvc2-convert")}
            receipt["native_sha256"]["libcrispasr.so"] = digest(find_library(args.build_dir))
            receipt["cases"] = run_cases(manifest, files, args.build_dir.resolve(), args.out.resolve())
            for key, path in files.items():
                if digest(path) != manifest["files"][key]["sha256"]:
                    raise ValueError(f"Pinned input changed during qualification: {key}")
            receipt["status"] = "passed_supplied_fixture_gates"
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError, subprocess.SubprocessError) as error:
        receipt["error"] = str(error)
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 1 if receipt["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
