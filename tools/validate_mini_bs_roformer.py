#!/usr/bin/env python3
"""Quantize, diff and time Mini BS RoFormer against the pinned Python blueprint.

Correctness and performance are separate gates. A skipped benchmark never
counts as acceptance. The native model and Python model stay loaded for warm
timings; audio reading, conversion and loading are outside the timed region.
"""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np


STEMS = ("bass", "drums", "other", "vocals")
THRESHOLDS = {"f16": 0.999, "q8_0": 0.98, "q4_k": 0.94}
NORM_RATIO_BOUNDS = (0.8, 1.2)
REQUIRED_STAGES = (
    "input_audio", "stft_input", "stft_output", "band_split", "time_compressed",
    *(f"layer{i}_{axis}" for i in range(6) for axis in ("time", "freq")),
    "encoder_output", "freq_pooled", "stem_embeddings", "mask_raw",
    *(f"output_{stem}" for stem in STEMS),
)
# stft_output's contiguous rows are individual Re/Im pairs. Near-zero bins
# make their angle unstable; whole-tensor cosine plus magnitude still gates it.
F16_ROW_STAGES = set(REQUIRED_STAGES) - {"stft_output"}
OUTPUT_STAGES = ("input_audio", *(f"output_{stem}" for stem in STEMS))
PREFIX = "freq_domain_model."
PROTECTED = (PREFIX + "mask_model.band_split.", PREFIX + "mask_model.mask_estimator.")
FLOORED = (PREFIX + "freq_pooling.", PREFIX + "time_pooling.",
           PREFIX + "mask_model.time_conv.", PREFIX + "mask_model.time_deconv.")
TYPE_NAMES = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1",
              8: "q8_0", 10: "q2_k", 11: "q3_k", 12: "q4_k", 13: "q5_k", 14: "q6_k"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(path, value):
    """Publish complete JSON without colliding with another writer's temporary file."""
    path = Path(path)
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(encoded)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verify_existing_quant(regenerated, existing):
    """Reuse only byte-identical real-tool output; never remove the source model."""
    regenerated, existing = Path(regenerated), Path(existing)
    if regenerated.samefile(existing):
        raise ValueError("temporary quantizer output aliases the existing model")
    actual, expected = sha256(regenerated), sha256(existing)
    if actual != expected:
        raise ValueError(f"regenerated quant differs from existing artifact: {actual} vs {expected}")
    return actual


def runtime_provenance(library):
    """Record the observed host and adjacent build metadata, without guessing."""
    result = {"platform": platform.platform(), "machine": platform.machine(),
              "logical_cpus": os.cpu_count(), "library_sha256": sha256(library)}
    result["runtime_environment"] = {name: os.environ.get(name) for name in (
        "CRISPASR_MINI_BS_CPU_BF16", "CRISPASR_MINI_BS_PACKED_BF16", "CRISPASR_MINI_BS_SHARED_DECODER",
        "CRISPASR_MINI_BS_BLAS",
        "CRISPASR_MINI_BS_FLASH", "CRISPASR_MINI_BS_THREADS", "CRISPASR_MINI_BS_GPU",
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    for name, path in (("cpu_quota", "/sys/fs/cgroup/cpu.max"),
                       ("cpuset", "/sys/fs/cgroup/cpuset.cpus.effective"),
                       ("memory_limit_bytes", "/sys/fs/cgroup/memory.max")):
        try:
            result[name] = Path(path).read_text().strip()
        except OSError:
            pass
    try:
        first_cpu = Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0]
        fields = dict(line.split(":", 1) for line in first_cpu.splitlines() if ":" in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        result["cpu_model"] = fields.get("model name", fields.get("Processor"))
        result["cpu_flags"] = fields.get("flags", fields.get("Features", "")).split()
    except OSError:
        pass
    cache = Path(library).resolve().parent.parent / "CMakeCache.txt"
    if cache.is_file():
        settings = {}
        for line in cache.read_text().splitlines():
            match = re.match(r"([^:#/]+):[^=]+=(.*)", line)
            if match:
                settings[match[1]] = match[2]
        result["cmake_cache_sha256"] = sha256(cache)
        result["build_options"] = {key: value for key, value in settings.items()
                                   if key.startswith("GGML_") or key in
                                   ("CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "BLAS_openblas_LIBRARY")}
        source = settings.get("CMAKE_HOME_DIRECTORY")
        if source:
            result["source_files_sha256"] = {
                str(path.relative_to(source)): sha256(path)
                for name in ("mini_bs_roformer.cpp", "mini_bs_roformer.h", "mini_bs_roformer_fft.h",
                             "mini_bs_roformer_bf16.h")
                if (path := Path(source) / "src" / name).is_file()
            }
            git = subprocess.run(["git", "-C", str(Path(source) / "ggml"), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, check=False)
            if git.returncode == 0:
                result["ggml_checkout_revision"] = git.stdout.strip()
        blas = settings.get("BLAS_openblas_LIBRARY")
        if blas and Path(blas).is_file():
            result["openblas_sha256"] = sha256(blas)
            try:
                query = ctypes.CDLL(blas).openblas_get_config
                query.restype = ctypes.c_char_p
                result["openblas_config"] = query().decode()
            except (OSError, AttributeError):
                pass
    return result


def gguf_inventory(path):
    """Read metadata/type geometry without loading or dequantizing weights."""
    with Path(path).open("rb") as stream:
        def number(fmt):
            width = struct.calcsize("<" + fmt)
            data = stream.read(width)
            if len(data) != width:
                raise ValueError("truncated GGUF")
            return struct.unpack("<" + fmt, data)[0]

        def text():
            length = number("Q")
            if length > 16 * 1024 * 1024:
                raise ValueError("invalid GGUF string length")
            data = stream.read(length)
            if len(data) != length:
                raise ValueError("truncated GGUF string")
            return data.decode()

        def value(kind):
            if kind == 8:
                return text()
            if kind == 9:
                subtype, length = number("I"), number("Q")
                if length > 10_000_000:
                    raise ValueError("invalid GGUF metadata array length")
                return [value(subtype) for _ in range(length)]
            return number({0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
                           7: "B", 10: "Q", 11: "q", 12: "d"}[kind])

        if stream.read(4) != b"GGUF" or number("I") not in (2, 3):
            raise ValueError("expected a GGUF v2/v3 file")
        count, n_metadata = number("Q"), number("Q")
        if count > 100_000 or n_metadata > 100_000:
            raise ValueError("unreasonable GGUF header counts")
        metadata = {}
        for _ in range(n_metadata):
            key, kind = text(), number("I")
            metadata[key] = value(kind)
        tensors = {}
        for _ in range(count):
            name, rank = text(), number("I")
            if not 1 <= rank <= 4:
                raise ValueError(f"invalid rank for {name}")
            shape = [number("Q") for _ in range(rank)]
            kind, offset = number("I"), number("Q")
            if name in tensors:
                raise ValueError(f"duplicate tensor {name}")
            tensors[name] = {"shape_ggml": shape, "type": TYPE_NAMES.get(kind, f"type-{kind}"),
                             "elements": math.prod(shape), "offset": offset}
    counts = Counter(t["type"] for t in tensors.values())
    elements = Counter()
    for tensor in tensors.values():
        elements[tensor["type"]] += tensor["elements"]
    return {"file_bytes": Path(path).stat().st_size, "sha256": sha256(path),
            "metadata": metadata, "tensor_count_by_type": dict(counts),
            "elements_by_type": dict(elements), "tensors": tensors}


def check_inventory(inventory, precision):
    if inventory["metadata"].get("general.architecture") != "mini-bs-roformer":
        raise ValueError("GGUF architecture is not mini-bs-roformer")
    tensors = inventory["tensors"]
    if not tensors:
        raise ValueError("model has no tensors")
    body = [t for name, t in tensors.items() if ".encoder.layers." in name and name.endswith(".weight")]
    if not body or not any(t["type"] == precision for t in body):
        raise ValueError(f"no axial encoder weights stored as {precision}")
    for name, tensor in tensors.items():
        kind = tensor["type"]
        if precision == "f16" and kind not in ("f16", "f32"):
            raise ValueError(f"F16 base contains {kind}: {name}")
        if name.endswith((".scale", ".probe", ".register_tokens")) and kind != "f32":
            raise ValueError(f"critical arithmetic operand must remain F32: {name} ({kind})")
        if name.startswith(PROTECTED) and name.endswith(".weight") and kind not in ("f16", "f32"):
            raise ValueError(f"spectral projection or shared mask head lost precision: {name} ({kind})")
        if precision == "q4_k" and name.startswith(FLOORED) and kind not in ("q8_0", "f16", "f32"):
            raise ValueError(f"pooling/temporal projection is below Q8_0: {name} ({kind})")


def quantizer_command(executable, kind, source, destination, precision, imatrix=None):
    command = [str(executable), str(source), str(destination), precision]
    if kind == "crispembed":
        # CrispEmbed uses full-name globs (last match wins); CrispASR uses
        # partial regular expressions (first match wins). Do not interchange.
        rules = [*(prefix + "*.weight=f16" for prefix in PROTECTED),
                 "*.probe=f32", "*.register_tokens=f32"]
        if precision == "q4_k":
            rules.extend(prefix + "*.weight=q8_0" for prefix in FLOORED)
        for rule in rules:
            command.extend(("--tensor-type", rule))
    if imatrix is not None:
        command.extend(("--imatrix", str(imatrix)))
    return command


def parse_diff(log, precision, outputs_only=False):
    """Fail closed on missing/duplicate stages, skipped work, or boundary scores."""
    rows = {}
    expected_stages = OUTPUT_STAGES if outputs_only else REQUIRED_STAGES
    for line in log.splitlines():
        match = re.search(r"\[(PASS|FAIL|SKIP)\]\s+(\S+)\s+(.*)", line)
        if not match:
            continue
        status, stage, details = match.groups()
        if stage not in expected_stages:
            continue
        if stage in rows:
            raise ValueError(f"duplicate diff stage: {stage}")
        values = {}
        for key in ("cos_global", "cos_min", "norm_ratio", "n_nonfinite", "nonfinite"):
            item = re.search(rf"\b{key}=([^\s,]+)", details)
            if item:
                values[key] = float(item.group(1))
        if status != "PASS" or "cos_global" not in values:
            raise ValueError(f"stage not accepted: {line}")
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError(f"non-finite metric: {stage}")
        if values.get("n_nonfinite", values.get("nonfinite", 0)) != 0:
            raise ValueError(f"non-finite tensor: {stage}")
        if values["cos_global"] <= THRESHOLDS[precision]:
            raise ValueError(f"{stage} cosine must be > {THRESHOLDS[precision]}")
        if not NORM_RATIO_BOUNDS[0] <= values.get("norm_ratio", -1) <= NORM_RATIO_BOUNDS[1]:
            raise ValueError(f"{stage} norm ratio is outside {NORM_RATIO_BOUNDS}")
        if precision == "f16" and stage in F16_ROW_STAGES and values.get("cos_min", -1) <= THRESHOLDS[precision]:
            raise ValueError(f"{stage} F16 row cosine must be > {THRESHOLDS[precision]}")
        rows[stage] = values
    missing = set(expected_stages) - rows.keys()
    if missing:
        raise ValueError(f"missing diff stages: {sorted(missing)}")
    return rows


def waveform_metrics(actual, expected, threshold):
    if actual.shape != expected.shape or actual.ndim != 3 or actual.shape[:2] != (4, 2):
        raise ValueError(f"waveform shapes differ: {actual.shape} vs {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("waveform contains NaN or Inf")
    reports = {}
    for index, stem in enumerate(STEMS):
        a = actual[index].astype(np.float64, copy=False).reshape(-1)
        b = expected[index].astype(np.float64, copy=False).reshape(-1)
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        cosine = float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float(na == nb)
        ratio = na / nb if nb > 0 else (1.0 if na == 0 else None)
        report = {"cos_global": cosine, "norm_ratio": ratio,
                  "max_abs": float(np.max(np.abs(a - b))),
                  "rms_error": float(np.sqrt(np.mean((a - b) ** 2)))}
        # Cosine alone accepts scaled or silent replacements. Retain raw
        # magnitudes and require energy agreement independently of direction.
        report["acceptance"] = (cosine > threshold and ratio is not None and
                                NORM_RATIO_BOUNDS[0] <= ratio <= NORM_RATIO_BOUNDS[1])
        reports[stem] = report
    return reports


class NativeModel:
    class Params(ctypes.Structure):
        _fields_ = [("n_threads", ctypes.c_int), ("use_gpu", ctypes.c_bool),
                    ("gpu_device", ctypes.c_int), ("segment_seconds", ctypes.c_int),
                    ("no_segment", ctypes.c_bool)]

    class Result(ctypes.Structure):
        _fields_ = [("n_sources", ctypes.c_int), ("n_channels", ctypes.c_int),
                    ("n_samples", ctypes.c_int), ("sample_rate", ctypes.c_int),
                    ("sources", ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
                    ("source_names", ctypes.POINTER(ctypes.c_char_p))]

    def __init__(self, library, model, threads, device, mode):
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        self.lib.mini_bs_roformer_default_params.restype = self.Params
        self.lib.mini_bs_roformer_init_from_file.argtypes = [ctypes.c_char_p, self.Params]
        self.lib.mini_bs_roformer_init_from_file.restype = ctypes.c_void_p
        self.lib.mini_bs_roformer_separate.argtypes = [ctypes.c_void_p,
                                                      ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
        self.lib.mini_bs_roformer_separate.restype = ctypes.POINTER(self.Result)
        self.lib.mini_bs_roformer_result_free.argtypes = [ctypes.POINTER(self.Result)]
        self.lib.mini_bs_roformer_free.argtypes = [ctypes.c_void_p]
        params = self.lib.mini_bs_roformer_default_params()
        params.n_threads, params.use_gpu = threads, device != "cpu"
        params.no_segment = mode == "forward"
        self.ctx = self.lib.mini_bs_roformer_init_from_file(os.fsencode(model), params)
        if not self.ctx:
            raise RuntimeError(f"native model failed to initialize: {model}")
        self.actual_backend = "CPU" if device == "cpu" else "unverified"
        self.backend_identity_source = "explicit CPU parameters" if device == "cpu" else "unverified"
        query = getattr(self.lib, "mini_bs_roformer_backend_name", None)
        if query is not None:
            query.argtypes = [ctypes.c_void_p]
            query.restype = ctypes.c_char_p
            observed = query(self.ctx)
            self.actual_backend = observed.decode() if observed else "unverified"
            self.backend_identity_source = "native backend query"
        self.compute_precision = "unverified"
        query = getattr(self.lib, "mini_bs_roformer_compute_precision", None)
        if query is not None:
            query.argtypes = [ctypes.c_void_p]
            query.restype = ctypes.c_char_p
            observed = query(self.ctx)
            self.compute_precision = observed.decode() if observed else "unverified"

    def run(self, interleaved):
        result = self.lib.mini_bs_roformer_separate(
            self.ctx, interleaved.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(interleaved), 2)
        if not result:
            raise RuntimeError("native separation failed")
        try:
            r = result.contents
            if (r.n_sources, r.n_channels, r.n_samples, r.sample_rate) != (4, 2, len(interleaved), 44100):
                raise ValueError("native result has unexpected stem/channel/sample/rate metadata")
            names = tuple(r.source_names[i].decode() for i in range(r.n_sources))
            if names != STEMS:
                raise ValueError(f"native stem order differs: {names}")
            return np.stack([np.ctypeslib.as_array(r.sources[i], shape=(r.n_samples * 2,))
                             .reshape(-1, 2).T.copy() for i in range(4)])
        finally:
            self.lib.mini_bs_roformer_result_free(result)

    def close(self):
        if self.ctx:
            self.lib.mini_bs_roformer_free(self.ctx)
            self.ctx = None


def benchmark(args, models, expected_model_hashes=None, checkpoint_path=None):
    run_id = uuid.uuid4().hex
    if checkpoint_path is not None:
        atomic_json_write(checkpoint_path, {
            "run_id": run_id, "complete": False, "acceptance": False, "status": "initializing",
            "requested_models": {precision: str(path) for precision, path in models.items()},
            "expected_model_sha256": expected_model_hashes, "audio": str(args.audio),
            "library": str(args.library),
        })
    import gc
    import soundfile
    import torch
    import transformers
    from reference_backends.mini_bs_roformer import _load, _load_published

    model_hashes = {precision: sha256(path) for precision, path in models.items()}
    if expected_model_hashes is not None and model_hashes != expected_model_hashes:
        raise RuntimeError("model files differ from the artifacts that passed stage comparison")

    audio, rate = soundfile.read(args.audio, dtype="float32", always_2d=True)
    if rate != 44100 or audio.shape[1] != 2 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("benchmark audio must be nonempty finite stereo at 44100 Hz")
    audio = np.ascontiguousarray(audio)
    audio_digest = hashlib.sha256(audio.tobytes()).hexdigest()
    python_batch_size = getattr(args, "python_batch_size", 4)
    python_dtype = getattr(args, "python_dtype", "bfloat16")
    torch.set_num_threads(args.threads)
    blueprint = _load(args.model_dir, device=args.device, dtype="float32")
    waveform = torch.from_numpy(audio.T.copy()).to(args.device)

    def synchronize():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elif args.device == "mps":
            torch.mps.synchronize()

    def original():
        with torch.inference_mode():
            if args.mode == "forward":
                output = blueprint(waveform[None])[0]
            else:
                output = blueprint.separate(waveform, batch_size=python_batch_size, verbose=False)
            return output.float().cpu().numpy().copy()

    # The numerical oracle remains the pinned F32 original, independently of
    # the dtype used to measure the Python performance baseline. Compute it
    # before loading native models so its transient memory is not concurrent.
    reference_output = original()
    if python_dtype == "bfloat16":
        del blueprint
        gc.collect()
        blueprint = _load_published(args.model_dir, device=args.device)
        loader_description = "original class from_pretrained; published default dtype, no override"
    else:
        loader_description = "original class; strict checkpoint load and explicit float32"
    actual_python_dtype = str(next(blueprint.parameters()).dtype).removeprefix("torch.")
    if actual_python_dtype != python_dtype:
        raise ValueError(f"Python dtype differs from requested profile: {actual_python_dtype} vs {python_dtype}")
    print(f"Python timing profile: {actual_python_dtype}, batch_size={python_batch_size}; "
          f"quality oracle=float32; {loader_description}", file=sys.stderr, flush=True)
    python_buffers = {
        name: {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
               "sha256_raw_bytes": hashlib.sha256(
                   tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
        for name, tensor in blueprint.named_buffers()
    }

    def timed(fn):
        synchronize()
        start = time.perf_counter()
        output = fn()
        synchronize()
        return output, (time.perf_counter() - start) * 1000.0

    result = {"run_id": run_id, "mode": args.mode, "sample_rate": rate, "channels": 2, "samples": len(audio),
              "duration_seconds": len(audio) / rate, "threads": args.threads,
              "python_dtype": actual_python_dtype, "python_loader": loader_description,
              "quality_reference_dtype": "float32", "quality_reference": "untimed original on identical input and mode",
              "quality_reference_sha256": hashlib.sha256(reference_output.tobytes()).hexdigest(),
              "python_parameter_elements_by_dtype": {str(dtype): sum(
                  p.numel() for p in blueprint.parameters() if p.dtype == dtype)
                  for dtype in {p.dtype for p in blueprint.parameters()}},
              "python_buffers": python_buffers, "native_output_dtype": "float32",
              "requested_device": args.device, "python_device": str(next(blueprint.parameters()).device),
              "python_version": platform.python_version(), "torch_version": torch.__version__,
              "transformers_version": transformers.__version__,
              "python_intraop_threads": torch.get_num_threads(),
              "python_interop_threads": torch.get_num_interop_threads(),
              "python_mkl_available": torch.backends.mkl.is_available(),
              "python_mkldnn_enabled": torch.backends.mkldnn.enabled,
              "warmups": args.warmups, "repetitions": args.repetitions,
              "timing_boundary": "warm inference including result allocation/copy; excludes loading and audio read",
              "python_batch_size": python_batch_size,
              "input_pcm_sha256": audio_digest, "runtime": runtime_provenance(args.library), "models": {}}
    for precision, path in models.items():
        native = NativeModel(args.library, path, args.threads, args.device, args.mode)
        try:
            native_call = lambda: native.run(audio)
            for _ in range(args.warmups):
                original()
                native_call()
            print(f"{precision}: completed {args.warmups} warmup pair(s)", file=sys.stderr, flush=True)
            rows = []
            for iteration in range(args.repetitions):
                if iteration % 2:
                    actual, native_ms = timed(native_call)
                    expected, python_ms = timed(original)
                    order = ["native", "python"]
                else:
                    expected, python_ms = timed(original)
                    actual, native_ms = timed(native_call)
                    order = ["python", "native"]
                metrics = waveform_metrics(actual, reference_output, THRESHOLDS[precision])
                baseline_metrics = waveform_metrics(expected, reference_output, THRESHOLDS["f16"])
                timed_pair_metrics = waveform_metrics(actual, expected, THRESHOLDS[precision])
                rows.append({"iteration": iteration, "order": order, "native_ms": native_ms,
                             "python_ms": python_ms, "stems": metrics,
                             "timed_python_vs_f32_reference": baseline_metrics,
                             "native_vs_timed_python": timed_pair_metrics})
                print(f"{precision}: pair {iteration + 1}/{args.repetitions} "
                      f"native={native_ms:.3f} ms python={python_ms:.3f} ms "
                      f"quality={'PASS' if all(m['acceptance'] for m in metrics.values()) else 'FAIL'}",
                      file=sys.stderr, flush=True)
                if checkpoint_path is not None:
                    # Persist outside the timing boundary. A memory-limit kill
                    # must not erase earlier complete pairs or model profiles.
                    progress = {**result, "complete": False, "acceptance": False,
                                "active_precision": precision, "active_model_sha256": model_hashes[precision],
                                "active_native_backend": native.actual_backend,
                                "active_compute_precision": native.compute_precision, "active_runs": rows}
                    atomic_json_write(checkpoint_path, progress)
            if hashlib.sha256(audio.tobytes()).hexdigest() != audio_digest:
                raise ValueError("native inference modified the input waveform")
            if not np.array_equal(waveform.float().cpu().numpy(), audio.T):
                raise ValueError("original inference modified the input waveform")
            native_median = statistics.median(row["native_ms"] for row in rows)
            python_median = statistics.median(row["python_ms"] for row in rows)
            expected_backend = {"cpu": "CPU", "cuda": "CUDA", "mps": "METAL"}[args.device]
            backend_verified = native.actual_backend.upper().startswith(expected_backend)
            result["models"][precision] = {
                "model_sha256": model_hashes[precision],
                "native_backend": native.actual_backend, "backend_verified": backend_verified,
                "native_compute_precision": native.compute_precision,
                "backend_identity_source": native.backend_identity_source,
                "native_median_ms": native_median, "python_median_ms": python_median,
                "native_speedup": python_median / native_median,
                "native_realtime_factor": native_median / (len(audio) / rate * 1000.0),
                "quality_acceptance": all(item["acceptance"] for row in rows for item in row["stems"].values()),
                "native_vs_timed_python_acceptance": all(
                    item["acceptance"] for row in rows for item in row["native_vs_timed_python"].values()),
                "timed_python_matches_f32_port_tolerance": all(
                    item["acceptance"] for row in rows for item in row["timed_python_vs_f32_reference"].values()),
                "performance_acceptance": native_median <= python_median and backend_verified,
                "runs": rows}
        finally:
            native.close()
    result["profile_acceptance"] = all(item["quality_acceptance"] and item["performance_acceptance"]
                                       for item in result["models"].values())
    result["published_default_profile"] = (actual_python_dtype == "bfloat16" and
                                           args.mode == "separate" and python_batch_size == 4)
    result["acceptance"] = result["profile_acceptance"] and result["published_default_profile"]
    if sha256(args.library) != result["runtime"]["library_sha256"]:
        raise RuntimeError("native library changed during the paired benchmark")
    if {precision: sha256(path) for precision, path in models.items()} != model_hashes:
        raise RuntimeError("model files changed during the paired benchmark")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--f16", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--full-reference", type=Path,
                        help="additional complete-chunk reference containing input_audio and all four stems")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--quantizer", required=True, type=Path)
    parser.add_argument("--quantizer-kind", choices=("crispasr", "crispembed"), default="crispasr")
    parser.add_argument("--verify-existing-quants", type=Path,
                        help="reuse Q8/Q4 files in this directory only after byte-identical regeneration with the real quantizer")
    parser.add_argument("--imatrix", type=Path,
                        help="original-input column-energy calibration GGUF for weighted Q4_K")
    parser.add_argument("--diff", required=True, type=Path)
    parser.add_argument("--crispembed-diff", type=Path,
                        help="adapter compiled against the genuine external crispembed_diff.h")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--python-batch-size", type=int, default=4,
                        help="original separate() chunk batch size; 4 is the method default")
    parser.add_argument("--python-dtype", choices=("float32", "bfloat16"), default="bfloat16",
                        help="timed Python profile; bfloat16 uses the published default loader, while quality stays F32")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--mode", choices=("forward", "separate"), default="separate")
    parser.add_argument("--stages-only", action="store_true",
                        help="capture numerical evidence without claiming performance acceptance")
    parser.add_argument("--discard-native-captures", action="store_true",
                        help="remove large native activation archives after recording their hashes and comparison logs")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.threads < 1 or args.warmups < 1 or args.repetitions < 3 or args.python_batch_size < 1:
        parser.error("threads/warmups/python-batch-size must be positive and repetitions must be at least three")
    if not args.stages_only and not args.library:
        parser.error("--library is required for the warm in-process benchmark")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in ("model_dir", "f16", "reference", "full_reference", "audio", "quantizer", "imatrix", "diff", "crispembed_diff", "output_dir", "library", "verify_existing_quants"):
        path = getattr(args, key)
        if path is not None:
            setattr(args, key, path.resolve())
    receipt_path = args.output_dir / "validation.json"
    run_id = uuid.uuid4().hex
    atomic_json_write(receipt_path, {"run_id": run_id, "acceptance": False, "complete": False,
                                    "status": "initializing"})
    receipt = {"schema": 1, "run_id": run_id, "acceptance": False, "complete": False,
               "platform": platform.platform(), "thresholds_strict_gt": THRESHOLDS,
               "norm_ratio_bounds": NORM_RATIO_BOUNDS,
               "amplitude_contract": "native original 20% guard; earlier exploratory external 10% checks retained in logs",
               "python_timing_profile": args.python_dtype,
               "required_performance_profile": {"python_dtype": "bfloat16", "mode": "separate", "batch_size": 4,
                                                "loader": "published from_pretrained without a dtype override"},
               "genuine_crispembed_profile": args.quantizer_kind == "crispembed" and args.crispembed_diff is not None,
               "existing_quants_verified_by_regeneration": {},
               "reference_sha256": sha256(args.reference), "audio_sha256": sha256(args.audio),
               "quantizer_kind": args.quantizer_kind, "quantizer_sha256": sha256(args.quantizer),
               "diff_sha256": sha256(args.diff), "models": {}, "commands": []}
    if args.crispembed_diff:
        receipt["crispembed_diff_sha256"] = sha256(args.crispembed_diff)
    stage_inventory = gguf_inventory(args.reference)
    receipt["stage_reference"] = {"metadata": stage_inventory["metadata"],
                                   "input_shape_ggml": stage_inventory["tensors"].get("input_audio", {}).get("shape_ggml")}
    receipt["reference_geometry_acceptance"] = receipt["stage_reference"]["input_shape_ggml"] == [352800, 2]
    receipt["native_capture_retention"] = "discard-after-comparison" if args.discard_native_captures else "retain"
    if args.full_reference:
        full_inventory = gguf_inventory(args.full_reference)
        receipt["full_reference"] = {"sha256": full_inventory["sha256"], "metadata": full_inventory["metadata"],
                                     "input_shape_ggml": full_inventory["tensors"].get("input_audio", {}).get("shape_ggml")}
    importance_inventory = gguf_inventory(args.imatrix) if args.imatrix else None
    if importance_inventory:
        receipt["imatrix"] = {"sha256": importance_inventory["sha256"],
                              "tensor_count": len(importance_inventory["tensors"])}

    def command(argv, log_name, env=None):
        receipt["commands"].append(argv)
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=args.timeout, env=env)
        (args.output_dir / log_name).write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f"command failed ({result.returncode}); see {log_name}")
        return result.stdout

    try:
        if not args.stages_only and not receipt["reference_geometry_acceptance"]:
            raise ValueError("qualification requires all 25 stages for one complete 352800-frame stereo chunk; "
                             "use --stages-only for short raw-forward diagnostics")
        quant_directory = args.verify_existing_quants or args.output_dir
        models = {"f16": args.f16, **{q: quant_directory / f"mini-bs-roformer-{q}.gguf"
                                    for q in ("q8_0", "q4_k")}}
        source_hash = sha256(args.f16)
        reference_hash = sha256(args.reference)
        for precision, path in models.items():
            if precision != "f16":
                destination = path
                if args.verify_existing_quants:
                    if not path.is_file():
                        raise ValueError(f"existing quantized artifact is missing: {path}")
                    descriptor, temporary = tempfile.mkstemp(prefix=f"regenerated-{precision}-", suffix=".gguf",
                                                              dir=args.output_dir)
                    os.close(descriptor)
                    destination = Path(temporary)
                argv = quantizer_command(args.quantizer, args.quantizer_kind, args.f16, destination, precision, args.imatrix)
                command(argv, f"quantize-{precision}.log")
                if args.verify_existing_quants:
                    receipt["existing_quants_verified_by_regeneration"][precision] = verify_existing_quant(destination, path)
                    atomic_json_write(receipt_path, receipt)
                    destination.unlink()
            inventory = gguf_inventory(path)
            check_inventory(inventory, precision)
            if importance_inventory and precision == "q4_k":
                for name, tensor in inventory["tensors"].items():
                    if tensor["type"] != "q4_k":
                        continue
                    importance = importance_inventory["tensors"].get(name)
                    count = importance_inventory["metadata"].get("count." + name, 0)
                    if (not importance or importance["type"] != "f32" or count <= 0 or
                            importance["elements"] != tensor["shape_ggml"][0]):
                        raise ValueError(f"missing/invalid calibration for a Q4_K matrix: {name}")
            (args.output_dir / f"inventory-{precision}.json").write_text(json.dumps(inventory, indent=2) + "\n")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(args.threads)
            env["CRISPASR_MINI_BS_THREADS"] = str(args.threads)
            env["CRISPASR_MINI_BS_GPU"] = "0" if args.device == "cpu" else "1"
            env.pop("CRISPASR_MINI_BS_OUTPUTS_ONLY", None)
            capture_path = args.output_dir / f"native-{precision}.gguf"
            if args.crispembed_diff:
                env["CRISPASR_MINI_BS_CAPTURE_GGUF"] = str(capture_path)
            log = command([str(args.diff), "mini-bs-roformer", str(path), str(args.reference), str(args.audio)],
                          f"diff-{precision}.log", env)
            stages = parse_diff(log, precision)
            receipt["models"][precision] = {"sha256": inventory["sha256"], "file_bytes": inventory["file_bytes"],
                                            "tensor_count_by_type": inventory["tensor_count_by_type"],
                                            "elements_by_type": inventory["elements_by_type"], "stages": stages,
                                            "numerical_acceptance": True}
            if args.crispembed_diff:
                embed_log = command([str(args.crispembed_diff), str(args.reference), str(capture_path), precision],
                                    f"crispembed-diff-{precision}.log")
                receipt["models"][precision]["crispembed_stages"] = parse_diff(embed_log, precision)
                receipt["models"][precision]["native_captures_sha256"] = sha256(capture_path)
                if args.discard_native_captures:
                    capture_path.unlink()
            if args.full_reference:
                full_capture = args.output_dir / f"native-full-{precision}.gguf"
                env["CRISPASR_MINI_BS_OUTPUTS_ONLY"] = "1"
                if args.crispembed_diff:
                    env["CRISPASR_MINI_BS_CAPTURE_GGUF"] = str(full_capture)
                full_log = command([str(args.diff), "mini-bs-roformer", str(path), str(args.full_reference), str(args.audio)],
                                   f"diff-full-{precision}.log", env)
                receipt["models"][precision]["full_output_stages"] = parse_diff(full_log, precision, outputs_only=True)
                if args.crispembed_diff:
                    full_embed_log = command([str(args.crispembed_diff), str(args.full_reference), str(full_capture),
                                              precision, "--outputs-only"], f"crispembed-diff-full-{precision}.log")
                    receipt["models"][precision]["crispembed_full_output_stages"] = parse_diff(
                        full_embed_log, precision, outputs_only=True)
                    receipt["models"][precision]["native_full_captures_sha256"] = sha256(full_capture)
                    if args.discard_native_captures:
                        full_capture.unlink()
                if sha256(args.full_reference) != receipt["full_reference"]["sha256"]:
                    raise RuntimeError("full reference changed during the sweep")
            if sha256(args.f16) != source_hash or sha256(args.reference) != reference_hash:
                raise RuntimeError("base model or reference changed during the quantization sweep")
            if args.imatrix and sha256(args.imatrix) != receipt["imatrix"]["sha256"]:
                raise RuntimeError("importance matrix changed during the quantization sweep")
        receipt["numerical_acceptance"] = True
        if args.stages_only:
            receipt["benchmark"] = {"acceptance": False, "status": "not-run"}
        else:
            receipt["library_sha256"] = sha256(args.library)
            receipt["benchmark"] = benchmark(args, models,
                {precision: receipt["models"][precision]["sha256"] for precision in models},
                checkpoint_path=args.output_dir / "benchmark-progress.json")
        receipt["complete"] = not args.stages_only
        receipt["profile_acceptance"] = receipt["numerical_acceptance"] and receipt["benchmark"].get("profile_acceptance", False)
        receipt["acceptance"] = (receipt["numerical_acceptance"] and receipt["benchmark"]["acceptance"] and
                                 receipt["genuine_crispembed_profile"])
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        print(receipt["error"], file=sys.stderr)
    finally:
        atomic_json_write(receipt_path, receipt)
        print(receipt_path)
    return 0 if receipt["acceptance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
