#!/usr/bin/env python3
"""Qualify SCNet against its pinned original Python implementation on one CPU.

All 35 native stages and four complete stereo stems are checked independently
from the native log. The default inference recipe, a contiguous >=24-second
music excerpt, and three alternating warm timing pairs are required for a full
acceptance. A diagnostic/stages-only run never establishes that acceptance.

Run conversion and quantization first; this program records the exact artifacts
it tests. It never overwrites a previous run directory or discards completed
timings, failed logs, native captures, or model files.
"""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
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

# Set BLAS controls before NumPy (or later PyTorch/native libraries) can load a
# thread pool. Importing this module for its model-free tests has no side effect.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)
if __name__ == "__main__":
    _thread_parser = argparse.ArgumentParser(add_help=False)
    _thread_parser.add_argument("--threads", type=int, default=2)
    _thread_args, _ = _thread_parser.parse_known_args()
    if _thread_args.threads > 0:
        for _thread_variable in THREAD_VARIABLES:
            os.environ[_thread_variable] = str(_thread_args.threads)

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scnet_runtime_identity import LoadedDSOTracker, validate_snapshot


SOURCE_REVISION = "5d95bf96b19c3eede63248d171efeca8e3abb948"
STEMS = ("drums", "bass", "other", "vocals")
PRECISIONS = ("f16", "q8_0", "q4_k")
THRESHOLDS = {"f16": 0.999, "q8_0": 0.98, "q4_k": 0.94}
# Fixed before the first actual model run. Direction alone cannot establish
# amplitude or error fidelity. No fitted gain, alignment, or output trimming.
NORM_RATIO_BOUNDS = {"f16": (0.99, 1.01), "q8_0": (0.95, 1.05), "q4_k": (0.8, 1.2)}
RELATIVE_L2_MAX = {"f16": 0.05, "q8_0": 0.20, "q4_k": 0.40}
MIN_TARGET_MATRIX_FRACTION = 0.5
# Numerical-conditioning correction established after the initial raw probes.
# It applies to fresh reports; prior strict-minimum-cosine failures stay intact.
# Binary16 machine epsilon scales absolute errors from the ORIGINAL stage.
# This reference-scale engineering tolerance is not a forward-error theorem.
F16_ROW_POLICY = "cosine_or_binary16_reference_floor_v1"
F16_ROW_FLOOR_EPSILON = 2.0 ** -10
REQUIRED_STAGES = (
    "input_audio", "waveform_stats", "raw_input_audio", "stft", "stft_stats", "normalized",
    *(f"{stage}{i}" for i in range(3) for stage in ("skip", "encoder", "decoder")),
    *(f"{stage}{i}" for i in range(6) for stage in ("dp", "feature")),
    *(prefix + stem for stem in STEMS for prefix in ("raw_output_", "output_")),
)
TYPE_NAMES = {0: "f32", 1: "f16", 8: "q8_0", 12: "q4_k"}
TYPE_GEOMETRY = {0: (1, 4), 1: (1, 2), 8: (32, 34), 12: (256, 144)}
FILE_TYPES = {"f16": 1, "q8_0": 7, "q4_k": 12}
F16_ROW_STAGES = set(REQUIRED_STAGES) - {
    "input_audio", "raw_input_audio", "waveform_stats", "stft", "stft_stats", "normalized",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pcm_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f4").tobytes()).hexdigest()


def atomic_json_write(path, value):
    path = Path(path)
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def portable_receipt(value, args):
    """Copy evidence for JSON output, replacing machine paths only.

    The live receipt and subprocess arguments remain unchanged. More specific
    file/directory prefixes take precedence over their containing directories.
    """
    prefixes = {str(Path(__file__).resolve().parents[1]): "{repo}",
                str(args.output_dir): "{output}", str(args.source_dir): "{original-source}"}
    for name in ("checkpoint", "f16", "q8_0", "q4_k", "reference", "audio", "fixture_source"):
        path = getattr(args, name, None)
        if path is not None:
            path = Path(path)
            if path.parent != Path(path.anchor):
                prefixes.setdefault(str(path.parent), "{assets}")
            prefixes[str(path)] = "{assets}/" + path.name
    for name in ("diff", "library"):
        path = getattr(args, name, None)
        if path is not None:
            path = Path(path)
            build = path.parent.parent if path.parent.name in ("bin", "src", "lib") else path.parent
            if build != Path(build.anchor):
                prefixes.setdefault(str(build), "{build}")
            prefixes[str(path)] = "{build}/" + path.relative_to(build).as_posix()
    replacements = sorted(((prefix, alias) for prefix, alias in prefixes.items() if len(prefix) > 1),
                          key=lambda item: len(item[0]), reverse=True)

    def visit(item):
        if isinstance(item, dict):
            return {visit(key): visit(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(child) for child in item]
        if isinstance(item, (str, Path)):
            text = str(item)
            for prefix, alias in replacements:
                text = text.replace(prefix, alias)
            return text
        return item

    return visit(value)


def shape4(shape):
    if not 1 <= len(shape) <= 4 or any(not isinstance(x, int) or x <= 0 for x in shape):
        raise ValueError(f"invalid tensor shape: {shape}")
    return tuple(shape) + (1,) * (4 - len(shape))


class GGUFFile:
    """Bounded GGUF header reader, including payload/range validation.

    Captures are read as memory-mapped F32 arrays. The checker does not depend
    on native tensor loading, its stage scores, or its quantization flags.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.size = self.path.stat().st_size
        with self.path.open("rb") as stream:
            def number(fmt):
                width = struct.calcsize("<" + fmt)
                data = stream.read(width)
                if len(data) != width:
                    raise ValueError("truncated GGUF header")
                return struct.unpack("<" + fmt, data)[0]

            def text():
                length = number("Q")
                if length > 1024 * 1024:
                    raise ValueError("unreasonable GGUF string length")
                data = stream.read(length)
                if len(data) != length:
                    raise ValueError("truncated GGUF string")
                return data.decode("utf-8")

            def value(kind, in_array=False):
                if kind == 8:
                    return text()
                if kind == 9:
                    if in_array:
                        raise ValueError("nested GGUF metadata arrays are unsupported")
                    subtype, count = number("I"), number("Q")
                    if count > 250_000:
                        raise ValueError("unreasonable GGUF metadata array length")
                    return [value(subtype, True) for _ in range(count)]
                formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
                           7: "B", 10: "Q", 11: "q", 12: "d"}
                if kind not in formats:
                    raise ValueError(f"unsupported GGUF metadata type {kind}")
                result = number(formats[kind])
                if isinstance(result, float) and not math.isfinite(result):
                    raise ValueError("nonfinite GGUF metadata")
                return result

            if stream.read(4) != b"GGUF" or number("I") not in (2, 3):
                raise ValueError("expected GGUF v2 or v3")
            count, fields = number("Q"), number("Q")
            if count > 100_000 or fields > 100_000:
                raise ValueError("unreasonable GGUF header counts")
            self.metadata = {}
            for _ in range(fields):
                key, kind = text(), number("I")
                if key in self.metadata:
                    raise ValueError(f"duplicate GGUF metadata key: {key}")
                self.metadata[key] = value(kind)
            self.tensors = {}
            for _ in range(count):
                name, rank = text(), number("I")
                if name in self.tensors or not 1 <= rank <= 4:
                    raise ValueError(f"duplicate tensor or invalid rank: {name}")
                shape = [number("Q") for _ in range(rank)]
                shape4(shape)
                kind, offset = number("I"), number("Q")
                if kind not in TYPE_GEOMETRY:
                    raise ValueError(f"unsupported SCNet/capture tensor type {kind}: {name}")
                block, width = TYPE_GEOMETRY[kind]
                if shape[0] % block:
                    raise ValueError(f"misaligned quantized row: {name}")
                elements = math.prod(shape)
                self.tensors[name] = {"shape_ggml": shape, "kind": kind,
                                      "type": TYPE_NAMES[kind], "elements": elements,
                                      "offset": offset, "bytes": elements // block * width}
            alignment = self.metadata.get("general.alignment", 32)
            if not isinstance(alignment, int) or not 1 <= alignment <= 65536 or alignment & (alignment - 1):
                raise ValueError("invalid GGUF alignment")
            self.data_start = (stream.tell() + alignment - 1) // alignment * alignment
            ranges = []
            for name, tensor in self.tensors.items():
                start = self.data_start + tensor["offset"]
                end = start + tensor["bytes"]
                if tensor["offset"] % alignment or end > self.size:
                    raise ValueError(f"unaligned or truncated tensor payload: {name}")
                ranges.append((start, end, name))
            ranges.sort()
            for previous, current in zip(ranges, ranges[1:]):
                if current[0] < previous[1]:
                    raise ValueError(f"overlapping tensor payloads: {previous[2]} / {current[2]}")

    def floats(self, name):
        tensor = self.tensors[name]
        if tensor["kind"] != 0:
            raise ValueError(f"reference/native capture must be F32: {name}")
        return np.memmap(self.path, dtype="<f4", mode="r",
                         offset=self.data_start + tensor["offset"], shape=(tensor["elements"],))

    def tensor_hash(self, name):
        tensor = self.tensors[name]
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            stream.seek(self.data_start + tensor["offset"])
            remaining = tensor["bytes"]
            while remaining:
                data = stream.read(min(remaining, 1024 * 1024))
                if not data:
                    raise ValueError(f"truncated tensor payload: {name}")
                digest.update(data)
                remaining -= len(data)
        return digest.hexdigest()

    def inventory(self):
        counts, elements = Counter(), Counter()
        for tensor in self.tensors.values():
            counts[tensor["type"]] += 1
            elements[tensor["type"]] += tensor["elements"]
        return {"file_bytes": self.size, "sha256": sha256(self.path), "metadata": self.metadata,
                "tensor_count_by_type": dict(counts), "elements_by_type": dict(elements),
                "tensors": self.tensors}


def check_inventory(model, precision, base=None, checkpoint_sha256=None, source_hashes=None):
    metadata, tensors = model.metadata, model.tensors
    if metadata.get("general.architecture") != "scnet" or metadata.get("scnet.tensor_layout_version") != 1:
        raise ValueError("SCNet architecture/layout metadata is missing or incompatible")
    if metadata.get("general.file_type") != FILE_TYPES[precision]:
        raise ValueError(f"declared precision is not {precision}")
    if metadata.get("general.source.revision") != SOURCE_REVISION:
        raise ValueError("model source revision is not the pinned SCNet blueprint")
    if source_hashes is not None and json.loads(metadata.get("general.source.hashes_json", "null")) != source_hashes:
        raise ValueError("model source-file hashes differ from the pinned original blueprint")
    if checkpoint_sha256 is not None and metadata.get("scnet.checkpoint.sha256") != checkpoint_sha256:
        raise ValueError("model checkpoint hash differs from the original checkpoint")
    if not tensors:
        raise ValueError("model contains no tensors")
    allowed = {"f16", "f32"} if precision == "f16" else {"f16", "f32", "q8_0"}
    if precision == "q4_k":
        allowed.add("q4_k")
    matrix_elements = target_elements = 0
    for name, tensor in tensors.items():
        if tensor["type"] not in allowed:
            raise ValueError(f"unsupported precision fallback {tensor['type']}: {name}")
        dims = shape4(tensor["shape_ggml"])
        matrix = dims[1] > 1
        if not matrix and tensor["type"] != "f32":
            raise ValueError(f"scalar/norm/bias lost F32 precision: {name}")
        if matrix:
            matrix_elements += tensor["elements"]
            target_elements += tensor["elements"] if tensor["type"] == precision else 0
        if base is not None:
            original = base.tensors.get(name)
            if original is None or shape4(original["shape_ggml"]) != dims:
                raise ValueError(f"quantized model changed tensor identity/shape: {name}")
            if not matrix and model.tensor_hash(name) != base.tensor_hash(name):
                raise ValueError(f"quantization changed scalar/norm/bias bytes: {name}")
    if base is not None and set(tensors) != set(base.tensors):
        raise ValueError("quantized tensor set differs from F16 source")
    fraction = target_elements / matrix_elements if matrix_elements else 0.0
    if fraction < MIN_TARGET_MATRIX_FRACTION:
        raise ValueError(f"{precision} matrix coverage {fraction:.6f} is below {MIN_TARGET_MATRIX_FRACTION}")
    return {"matrix_elements": matrix_elements, "target_type_matrix_elements": target_elements,
            "target_type_matrix_fraction": fraction, "minimum_fraction": MIN_TARGET_MATRIX_FRACTION}


def tensor_metrics(actual, expected, precision):
    """Absolute-scale comparison with bounded temporary memory, no alignment."""
    if actual.shape != expected.shape or not actual.size:
        raise ValueError(f"tensor shapes differ or are empty: {actual.shape} / {expected.shape}")
    a, b = actual.reshape(-1), expected.reshape(-1)
    dot = aa = bb = difference = maximum = 0.0
    for offset in range(0, a.size, 262144):
        x, y = a[offset:offset + 262144].astype(np.float64), b[offset:offset + 262144].astype(np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("tensor contains NaN or Inf")
        delta = x - y
        dot += float(np.dot(x, y))
        aa += float(np.dot(x, x))
        bb += float(np.dot(y, y))
        difference += float(np.dot(delta, delta))
        maximum = max(maximum, float(np.max(np.abs(delta))))
    cosine = dot / math.sqrt(aa * bb) if aa and bb else float(aa == bb)
    ratio = math.sqrt(aa / bb) if bb else (1.0 if aa == 0 else None)
    relative = math.sqrt(difference / bb) if bb else (0.0 if difference == 0 else None)
    bounds = NORM_RATIO_BOUNDS[precision]
    return {"cos_global": cosine, "norm_ratio": ratio, "relative_l2": relative,
            "max_abs": maximum, "rms_error": math.sqrt(difference / a.size), "elements": a.size,
            "acceptance": bool(cosine > THRESHOLDS[precision] and ratio is not None and
                               bounds[0] <= ratio <= bounds[1] and relative is not None and
                               relative <= RELATIVE_L2_MAX[precision])}


def row_metrics(actual, expected, width):
    """Report raw angles plus the declared F16 rule for every logical row.

    A row passes on strict cosine, OR on BOTH RMS and maximum absolute error
    within binary16 machine epsilon times the original whole-stage RMS and
    peak. This is an engineering tolerance, not a network roundoff theorem.
    Every rescued row retains its raw low cosine in the report. No actual-
    output scale, fitted gain, or observed-error calibration sets the bounds.
    An entirely zero reference stage permits no added signal. Analytic stages
    report this policy but do not gate F16 on row angles or the absolute floor.
    """
    if width <= 0 or not actual.size or actual.size != expected.size or actual.size % width:
        raise ValueError("invalid logical row geometry")
    a, b = actual.reshape(-1, width), expected.reshape(-1, width)
    reference_energy = reference_peak = 0.0
    flat_reference = expected.reshape(-1)
    for offset in range(0, flat_reference.size, 262144):
        y = flat_reference[offset:offset + 262144].astype(np.float64)
        if not np.isfinite(y).all():
            raise ValueError("nonfinite logical row reference")
        reference_energy += float(np.dot(y, y))
        reference_peak = max(reference_peak, float(np.max(np.abs(y))))
    reference_rms = math.sqrt(reference_energy / expected.size)
    rms_limit = F16_ROW_FLOOR_EPSILON * reference_rms
    peak_limit = F16_ROW_FLOOR_EPSILON * reference_peak
    minimum, total, zero_pairs, zero_mismatches, failed_f16 = 1.0, 0.0, 0, 0, 0
    rescued = 0
    # Bound transient float64 arrays even when rows themselves are long audio.
    for offset in range(0, len(a), max(1, 262144 // width)):
        count = max(1, 262144 // width)
        x, y = a[offset:offset + count].astype(np.float64), b[offset:offset + count].astype(np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("nonfinite logical row")
        aa, bb = np.einsum("ij,ij->i", x, x), np.einsum("ij,ij->i", y, y)
        dots = np.einsum("ij,ij->i", x, y)
        positive = (aa > 0) & (bb > 0)
        scores = np.where(aa == bb, 1.0, 0.0)
        scores[positive] = dots[positive] / np.sqrt(aa[positive] * bb[positive])
        zero_pairs += int(np.count_nonzero((aa == 0) & (bb == 0)))
        zero_mismatches += int(np.count_nonzero((aa == 0) != (bb == 0)))
        minimum = min(minimum, float(np.min(scores)))
        total += float(np.sum(scores))
        failed_cosine = scores <= THRESHOLDS["f16"]
        failed_f16 += int(np.count_nonzero(failed_cosine))
        delta = x - y
        rms_errors = np.sqrt(np.einsum("ij,ij->i", delta, delta) / width)
        peak_errors = np.max(np.abs(delta), axis=1)
        rescued += int(np.count_nonzero(failed_cosine & (rms_errors <= rms_limit) &
                                       (peak_errors <= peak_limit)))
    return {"cos_min": minimum, "cos_mean": total / len(a), "rows": len(a), "row_width": width,
            "zero_row_pairs": zero_pairs, "zero_row_mismatches": zero_mismatches,
            "rows_failing_f16_cosine": failed_f16, "row_policy": F16_ROW_POLICY,
            "row_floor_epsilon": F16_ROW_FLOOR_EPSILON,
            "reference_rms": reference_rms, "reference_peak": reference_peak,
            "row_rms_error_limit": rms_limit, "row_max_abs_error_limit": peak_limit,
            "row_rescued_count": rescued, "row_failed_count": failed_f16 - rescued}


def compare_captures(actual, expected, precision):
    if set(actual.tensors) != set(REQUIRED_STAGES) or set(expected.tensors) != set(REQUIRED_STAGES):
        raise ValueError("captures must contain exactly all 35 required SCNet stages")
    result = {}
    for name in REQUIRED_STAGES:
        a, b = actual.tensors[name], expected.tensors[name]
        if shape4(a["shape_ggml"]) != shape4(b["shape_ggml"]):
            raise ValueError(f"capture shape mismatch: {name}")
        observed, truth = actual.floats(name), expected.floats(name)
        result[name] = tensor_metrics(observed, truth, precision)
        result[name].update(row_metrics(observed, truth, b["shape_ggml"][0]))
        result[name]["row_cosine_gated"] = precision == "f16" and name in F16_ROW_STAGES
        if result[name]["row_cosine_gated"]:
            result[name]["acceptance"] &= result[name]["row_failed_count"] == 0
    return result


def parse_diff(log, precision):
    rows = {}
    for line in log.splitlines():
        match = re.match(r"^\[(PASS|FAIL|SKIP)\]\s+(\S+)\s+(.*)$", line)
        if not match:
            continue
        status, name, rest = match.groups()
        if name not in REQUIRED_STAGES or name in rows or status != "PASS":
            raise ValueError(f"failed, duplicate, skipped or unexpected stage: {name}")
        fields = dict(re.findall(r"([a-z_][a-z_0-9]*)=([^\s]+)", rest))
        required = {"cos_global", "norm_ratio", "relative_l2", "max_abs", "nonfinite", "n"}
        if not required <= fields.keys():
            raise ValueError(f"missing native metrics: {name}")
        values = {key: float(fields[key]) for key in required}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"nonfinite native metrics: {name}")
        if (values["nonfinite"] != 0 or values["n"] <= 0 or
                values["cos_global"] <= THRESHOLDS[precision] or
                not NORM_RATIO_BOUNDS[precision][0] <= values["norm_ratio"] <= NORM_RATIO_BOUNDS[precision][1] or
                not 0 <= values["relative_l2"] <= RELATIVE_L2_MAX[precision] or values["max_abs"] < 0):
            raise ValueError(f"native metric gate failed: {name}")
        if precision == "f16":
            row_fields = {"cos_min", "rows", "row_width", "row_cosine_gated", "rows_failing_cosine",
                          "row_floor_epsilon", "reference_rms", "reference_peak", "row_rms_error_limit",
                          "row_max_abs_error_limit", "row_rescued_count", "row_failed_count"}
            if fields.get("row_policy") != F16_ROW_POLICY or not row_fields <= fields.keys():
                raise ValueError(f"missing or incompatible native row policy: {name}")
            row_values = {key: float(fields[key]) for key in row_fields}
            if not all(math.isfinite(value) for value in row_values.values()):
                raise ValueError(f"nonfinite native row metrics: {name}")
            count_fields = ("rows", "row_width", "rows_failing_cosine", "row_rescued_count", "row_failed_count")
            if (any(row_values[key] < 0 or not row_values[key].is_integer() for key in count_fields) or
                    row_values["rows"] <= 0 or row_values["row_width"] <= 0 or
                    row_values["rows"] * row_values["row_width"] != values["n"] or
                    row_values["rows_failing_cosine"] > row_values["rows"] or
                    row_values["row_rescued_count"] + row_values["row_failed_count"] !=
                    row_values["rows_failing_cosine"]):
                raise ValueError(f"inconsistent native row counts: {name}")
            if (row_values["row_floor_epsilon"] != F16_ROW_FLOOR_EPSILON or
                    row_values["reference_rms"] < 0 or row_values["reference_peak"] < 0 or
                    not math.isclose(row_values["row_rms_error_limit"],
                                     F16_ROW_FLOOR_EPSILON * row_values["reference_rms"], rel_tol=1e-8, abs_tol=0) or
                    not math.isclose(row_values["row_max_abs_error_limit"],
                                     F16_ROW_FLOOR_EPSILON * row_values["reference_peak"], rel_tol=1e-8, abs_tol=0)):
                raise ValueError(f"inconsistent native reference-derived row bounds: {name}")
            gated = name in F16_ROW_STAGES
            if row_values["row_cosine_gated"] != int(gated) or (gated and row_values["row_failed_count"] != 0):
                raise ValueError(f"native row policy gate failed: {name}")
            values.update(row_values, row_policy=F16_ROW_POLICY)
        rows[name] = values
    if set(rows) != set(REQUIRED_STAGES):
        raise ValueError("native log omitted required SCNet stages")
    expected = f"scnet diff: 35 required stages, 0 failed, strict cosine > {THRESHOLDS[precision]:.3f}"
    if log.splitlines().count(expected) != 1:
        raise ValueError("native complete-stage summary is missing, duplicated or inconsistent")
    if precision == "f16":
        policy_header = f"scnet row policy: {F16_ROW_POLICY} epsilon={F16_ROW_FLOOR_EPSILON}"
        if log.splitlines().count(policy_header) != 1:
            raise ValueError("native F16 row-policy declaration is missing, duplicated or inconsistent")
    return rows


def waveform_metrics(actual, expected, precision):
    if actual.shape != expected.shape or actual.ndim != 3 or actual.shape[:2] != (4, 2):
        raise ValueError(f"unexpected stereo stem geometry: {actual.shape} / {expected.shape}")
    return {stem: tensor_metrics(actual[i], expected[i], precision) for i, stem in enumerate(STEMS)}


def all_pass(rows):
    return bool(rows) and all(row["acceptance"] for row in rows.values())


def validate_reference(reference, audio, checkpoint_hash, diagnostic=False, source_hashes=None, variant=None, threads=None):
    metadata = reference.metadata
    if metadata.get("scnet.reference.source_revision") != SOURCE_REVISION:
        raise ValueError("reference was not captured from the pinned original source")
    if metadata.get("scnet.reference.checkpoint_sha256") != checkpoint_hash:
        raise ValueError("reference checkpoint hash differs from the original checkpoint")
    if source_hashes is not None:
        if json.loads(metadata.get("scnet.reference.source_hashes_json", "null")) != source_hashes:
            raise ValueError("reference source-file hashes differ from the pinned original blueprint")
        if metadata.get("scnet.reference.required_stage_count") != 35 or metadata.get("scnet.reference.sample_rate") != 44100:
            raise ValueError("reference stage-count/sample-rate provenance is missing or incompatible")
    if variant is not None and metadata.get("scnet.reference.variant") != variant:
        raise ValueError("reference variant differs from the selected original checkpoint")
    if source_hashes is not None and str(metadata.get("scnet.reference.torch_version", "")).split("+")[0] != "2.8.0":
        raise ValueError("reference does not use the upstream-required torch==2.8.0")
    if threads is not None and (metadata.get("scnet.reference.threads") != threads or
                                metadata.get("scnet.reference.device") != "cpu"):
        raise ValueError("reference CPU/thread profile differs from the qualified profile")
    if set(reference.tensors) != set(REQUIRED_STAGES):
        raise ValueError("reference does not contain exactly 35 required stages")
    profile = {"mode": metadata.get("scnet.reference.mode"),
               "segment_seconds": metadata.get("scnet.reference.segment_seconds"),
               "shifts": metadata.get("scnet.reference.shifts"),
               "shift_seed": metadata.get("scnet.reference.shift_seed"),
               "overlap": metadata.get("scnet.reference.overlap")}
    full = (profile == {"mode": "pipeline", "segment_seconds": 20, "shifts": 1,
                        "shift_seed": 0, "overlap": 0.5} and len(audio) >= 24 * 44100)
    if not diagnostic and not full:
        raise ValueError("full acceptance requires >=24s and original 20s/.5 overlap/one-shift/seed-zero pipeline")
    if profile["mode"] not in ("pipeline", "raw") or profile["overlap"] != 0.5:
        raise ValueError("unsupported reference inference profile")
    if (not isinstance(profile["segment_seconds"], int) or not 1 <= profile["segment_seconds"] <= 119 or
            not isinstance(profile["shifts"], int) or not 0 <= profile["shifts"] <= 16 or
            not isinstance(profile["shift_seed"], int) or not 0 <= profile["shift_seed"] < 2 ** 64):
        raise ValueError("invalid reference inference parameters")
    if shape4(reference.tensors["input_audio"]["shape_ggml"]) != (len(audio), 2, 1, 1):
        raise ValueError("reference input geometry differs from benchmark WAV")
    planar = np.ascontiguousarray(audio.T, dtype=np.float32)
    if not np.array_equal(reference.floats("input_audio"), planar.reshape(-1)):
        raise ValueError("reference PCM differs from benchmark WAV bytes")
    if metadata.get("scnet.reference.input_pcm_sha256") != pcm_sha256(planar):
        raise ValueError("reference PCM provenance hash is missing or inconsistent")
    return {**profile, "samples": len(audio), "sample_rate": 44100, "full_profile": full}


class NativeModel:
    class Params(ctypes.Structure):
        _fields_ = [("n_threads", ctypes.c_int), ("use_gpu", ctypes.c_bool),
                    ("gpu_device", ctypes.c_int), ("segment_seconds", ctypes.c_int),
                    ("no_segment", ctypes.c_bool), ("shifts", ctypes.c_int), ("shift_seed", ctypes.c_uint64)]

    class Result(ctypes.Structure):
        _fields_ = [("n_sources", ctypes.c_int), ("n_channels", ctypes.c_int),
                    ("n_samples", ctypes.c_int), ("sample_rate", ctypes.c_int),
                    ("sources", ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
                    ("source_names", ctypes.POINTER(ctypes.c_char_p))]

    def __init__(self, library, model, threads, profile):
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        self.lib.scnet_default_params.argtypes = []
        self.lib.scnet_default_params.restype = self.Params
        self.lib.scnet_init_from_file.argtypes = [ctypes.c_char_p, self.Params]
        self.lib.scnet_init_from_file.restype = ctypes.c_void_p
        self.lib.scnet_separate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
        self.lib.scnet_separate.restype = ctypes.POINTER(self.Result)
        self.lib.scnet_result_free.argtypes = [ctypes.POINTER(self.Result)]
        self.lib.scnet_result_free.restype = None
        self.lib.scnet_free.argtypes = [ctypes.c_void_p]
        self.lib.scnet_free.restype = None
        params = self.lib.scnet_default_params()
        params.n_threads, params.use_gpu = threads, False
        params.segment_seconds = profile["segment_seconds"]
        params.no_segment = profile["mode"] == "raw"
        params.shifts, params.shift_seed = profile["shifts"], profile["shift_seed"]
        self.ctx = self.lib.scnet_init_from_file(os.fsencode(model), params)
        if not self.ctx:
            raise RuntimeError(f"native model initialization failed: {model}")
        for function, attribute in (("scnet_backend_name", "backend"), ("scnet_compute_precision", "compute_precision")):
            query = getattr(self.lib, function)
            query.argtypes, query.restype = [ctypes.c_void_p], ctypes.c_char_p
            value = query(self.ctx)
            setattr(self, attribute, value.decode() if value else "unknown")
        if not self.backend.upper().startswith("CPU"):
            self.close()
            raise ValueError(f"requested CPU benchmark selected {self.backend}")
        self.blas_thread_queries = {}
        for symbol in ("openblas_get_num_threads", "MKL_Get_Max_Threads", "bli_thread_get_num_threads"):
            query = getattr(self.lib, symbol, None)
            if query is not None:
                query.argtypes, query.restype = [], ctypes.c_int
                observed = query()
                self.blas_thread_queries[symbol] = observed
                if observed != threads:
                    self.close()
                    raise ValueError(f"native BLAS thread budget {observed} from {symbol} differs from {threads}")

    def run(self, audio):
        result = self.lib.scnet_separate(self.ctx, audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(audio), 2)
        if not result:
            raise RuntimeError("native separation failed")
        try:
            value = result.contents
            if (value.n_sources, value.n_channels, value.n_samples, value.sample_rate) != (4, 2, len(audio), 44100):
                raise ValueError("native separation result geometry/rate is invalid")
            if not value.sources or not value.source_names:
                raise ValueError("native separation result arrays are null")
            names = tuple(value.source_names[i].decode() for i in range(4))
            if names != STEMS or any(not value.sources[i] for i in range(4)):
                raise ValueError("native separation source names/pointers are invalid")
            return np.stack([np.ctypeslib.as_array(value.sources[i], shape=(len(audio) * 2,))
                             .reshape(-1, 2).T.copy() for i in range(4)])
        finally:
            self.lib.scnet_result_free(result)

    def close(self):
        if self.ctx:
            self.lib.scnet_free(self.ctx)
            self.ctx = None


def runtime_provenance(library, loaded_snapshot=None):
    result = {"platform": platform.platform(), "machine": platform.machine(), "logical_cpus": os.cpu_count(),
              "library_sha256": sha256(library), "thread_environment": {key: os.environ.get(key) for key in THREAD_VARIABLES},
              "wait_policy_environment": {key: os.environ.get(key) for key in (
                  "OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "KMP_BLOCKTIME", "OPENBLAS_THREAD_TIMEOUT")},
              "compute_environment": {key: os.environ.get(key) for key in (
                  "CRISPASR_SCNET_GPU", "CRISPASR_SCNET_THREADS", "CRISPASR_SCNET_BENCH",
                  "CRISPASR_SCNET_CAPTURE_GGUF", "CRISPASR_SCNET_SCALAR_LSTM", "CRISPASR_SCNET_SCALAR_GATES",
                  "CRISPASR_SCNET_BLAS", "CRISPASR_CPU_GEMM")}}
    native = ctypes.CDLL(str(Path(library).resolve()))
    result["blas_runtime"] = {}
    for name, kind in (("openblas_get_config", ctypes.c_char_p), ("openblas_get_corename", ctypes.c_char_p),
                       ("openblas_get_parallel", ctypes.c_int), ("openblas_get_num_threads", ctypes.c_int)):
        query = getattr(native, name, None)
        if query is None:
            continue
        query.argtypes, query.restype = [], kind
        value = query()
        result["blas_runtime"][name] = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    torch = sys.modules.get("torch")
    if torch is not None:
        result["torch_build_config"] = torch.__config__.show()
        result["torch_parallel_info"] = torch.__config__.parallel_info()
        result["torch_cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    for key, filename in (("cpuinfo", "/proc/cpuinfo"), ("cpu_quota", "/sys/fs/cgroup/cpu.max"),
                          ("cpuset", "/sys/fs/cgroup/cpuset.cpus.effective"),
                          ("memory_limit_bytes", "/sys/fs/cgroup/memory.max")):
        try:
            value = Path(filename).read_text()
            result[key] = value.split("\n\n", 1)[0] if key == "cpuinfo" else value.strip()
        except OSError:
            pass
    cache = Path(library).resolve().parent.parent / "CMakeCache.txt"
    if cache.is_file():
        result["cmake_cache_sha256"] = sha256(cache)
        result["build_settings"] = [line for line in cache.read_text().splitlines()
                                     if re.match(r"(?:GGML_|CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER|BLAS_)[^=]*=", line)]
    if loaded_snapshot is not None:
        result["loaded_dso_initialization"] = validate_snapshot(loaded_snapshot)
    return result


def summarize_pairs(calls, repetitions):
    expected = [(iteration, implementation) for iteration in range(repetitions)
                for implementation in (("original", "native") if iteration % 2 == 0 else ("native", "original"))]
    if [(row.get("iteration"), row.get("implementation")) for row in calls] != expected:
        raise ValueError("timing calls are missing, duplicated, reordered, or selectively omitted")
    if any(not isinstance(row.get("seconds"), (int, float)) or not math.isfinite(row["seconds"]) or
           row["seconds"] <= 0 or not row.get("quality_acceptance") for row in calls):
        raise ValueError("timing calls contain an invalid duration or failed/missing waveform gate")
    medians = {name: statistics.median(row["seconds"] for row in calls if row["implementation"] == name)
               for name in ("original", "native")}
    return {"native_median_seconds": medians["native"], "original_median_seconds": medians["original"],
            "speedup": medians["original"] / medians["native"],
            "performance_acceptance": medians["native"] <= medians["original"]}


def benchmark(args, models, reference, audio, profile, receipt, save):
    import torch
    from reference_backends.scnet import load_original, separate_original

    if torch.__version__.split("+")[0] != "2.8.0":
        raise ValueError("original benchmark must use upstream-required torch==2.8.0")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    blueprint = load_original(args.source_dir, args.checkpoint, args.variant, "cpu",
                              expected_sha256=receipt["checkpoint_sha256"])
    if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32 for parameter in blueprint.parameters()):
        raise ValueError("original benchmark is not the published F32 CPU model")
    planar = np.ascontiguousarray(audio.T)
    expected = np.stack([reference.floats("output_" + stem).reshape(2, -1).copy() for stem in STEMS])
    native_input_digest, original_input_digest = pcm_sha256(audio), pcm_sha256(planar)
    details = receipt["benchmark"] = {
        "device": "cpu", "threads": args.threads, "torch_version": torch.__version__,
        "torch_intraop_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "original_dtype": "float32", "original_provenance": blueprint._scnet_provenance,
        "runtime": runtime_provenance(args.library), "repetitions": args.repetitions, "warmups": args.warmups,
        "timing_boundary": "public separation call, normalization/chunking and owned CPU result conversion; includes original helper printing to a local log; excludes model/audio load and quality metrics",
        "native_stage_timers": os.environ.get("CRISPASR_SCNET_BENCH"),
        "profile": profile, "models": {}, "acceptance": False,
    }
    save()

    def original():
        return separate_original(blueprint, planar, segment_seconds=profile["segment_seconds"],
                                 shifts=profile["shifts"], shift_seed=profile["shift_seed"],
                                 no_segment=profile["mode"] == "raw")

    with (args.output_dir / "original-inference.log").open("w", encoding="utf-8") as original_log:
        with contextlib.redirect_stdout(original_log):
            for precision, path in models.items():
                if not receipt["models"][precision].get("numerical_acceptance"):
                    details["models"][precision] = {"attempted": False, "acceptance": False, "reason": "stage fidelity failed"}
                    save()
                    continue
                native = NativeModel(args.library, path, args.threads, profile)
                row = details["models"][precision] = {"native_backend": native.backend,
                    "native_compute_precision": native.compute_precision, "model_sha256": sha256(path),
                    "native_blas_thread_queries": native.blas_thread_queries,
                    "warmups": [], "calls": [], "complete": False, "acceptance": False}
                save()
                try:
                    identity = LoadedDSOTracker(args.library)
                    row["loaded_dso_identity"] = identity.evidence
                    if "loaded_dso_initialization" not in details["runtime"]:
                        details["runtime"]["loaded_dso_initialization"] = identity.evidence["initialization"]
                    save()
                    for iteration in range(args.warmups):
                        for name, call in (("original", original), ("native", lambda: native.run(audio))):
                            output = call()
                            scores = waveform_metrics(output, expected, "f16" if name == "original" else precision)
                            row["warmups"].append({"iteration": iteration, "implementation": name, "stems": scores,
                                                    "quality_acceptance": all_pass(scores)})
                            save()
                            if not all_pass(scores):
                                raise ValueError(f"{precision} {name} warmup output failed quality")
                    # First-use libraries may appear during warmups; the actual
                    # timed process mapping must then remain stable for every pair.
                    identity.before_timing()
                    save()
                    for iteration in range(args.repetitions):
                        order = ("original", "native") if iteration % 2 == 0 else ("native", "original")
                        for name in order:
                            started = time.perf_counter()
                            output = original() if name == "original" else native.run(audio)
                            seconds = time.perf_counter() - started
                            measurement = {"iteration": iteration, "implementation": name, "seconds": seconds,
                                           "quality_acceptance": False}
                            row["calls"].append(measurement)
                            # Persist every completed duration before doing any untimed metrics.
                            save()
                            scores = waveform_metrics(output, expected, "f16" if name == "original" else precision)
                            measurement.update(stems=scores, quality_acceptance=all_pass(scores), pcm_sha256=pcm_sha256(output))
                            save()
                            print(f"{precision} pair {iteration + 1}: {name} {seconds:.6f}s quality={all_pass(scores)}",
                                  file=sys.stderr, flush=True)
                            if not all_pass(scores):
                                raise ValueError(f"{precision} {name} measured output failed quality")
                    identity.after_timing()
                    save()
                    row.update(summarize_pairs(row["calls"], args.repetitions))
                    row["complete"] = True
                    row["acceptance"] = row["performance_acceptance"]
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                finally:
                    native.close()
                    save()
                if pcm_sha256(audio) != native_input_digest or pcm_sha256(planar) != original_input_digest:
                    raise ValueError("inference modified its input waveform")
    details["acceptance"] = (set(details["models"]) == set(PRECISIONS) and
                               all(row.get("acceptance", False) for row in details["models"].values()))
    save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source-dir", "checkpoint", "f16", "q8-0", "q4-k", "reference", "audio", "diff", "output-dir"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--library", type=Path, help="actual native shared library exporting scnet_* APIs")
    parser.add_argument("--variant", choices=("scnet", "scnet-large"), default="scnet")
    parser.add_argument("--fixture-source", type=Path, help="uncut source audio, required for full qualification")
    parser.add_argument("--fixture-source-url", help="public source provenance URL, required for full qualification")
    parser.add_argument("--fixture-offset-samples", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=3600, help="seconds allowed for each native stage process")
    parser.add_argument("--stages-only", action="store_true", help="no speed claim or overall acceptance")
    parser.add_argument("--diagnostic", action="store_true", help="permit short/raw reference; never counts as full acceptance")
    args = parser.parse_args(argv)
    if min(args.threads, args.warmups, args.timeout) < 1 or args.repetitions < 3 or args.fixture_offset_samples < 0:
        parser.error("threads/warmups/timeout must be positive; repetitions>=3 and fixture offset>=0")
    if not args.stages_only and args.library is None:
        parser.error("--library is required for actual warm timings")
    if not args.diagnostic and (args.fixture_source is None or not args.fixture_source_url):
        parser.error("full qualification requires --fixture-source and --fixture-source-url")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = args.output_dir / "validation.json"
    receipt = {"schema": 1, "run_id": uuid.uuid4().hex, "complete": False, "acceptance": False,
               "status": "initializing", "variant": args.variant, "source_revision": SOURCE_REVISION,
               "thresholds_strict_gt": THRESHOLDS, "norm_ratio_bounds": NORM_RATIO_BOUNDS,
               "relative_l2_max": RELATIVE_L2_MAX, "f16_row_stages": sorted(F16_ROW_STAGES),
               "f16_row_policy": F16_ROW_POLICY, "f16_row_floor_epsilon": F16_ROW_FLOOR_EPSILON,
               "required_stages": list(REQUIRED_STAGES), "models": {}, "commands": [], "errors": [],
               "diagnostic": args.diagnostic, "stages_only": args.stages_only}
    save = lambda: atomic_json_write(receipt_path, portable_receipt(receipt, args))
    save()
    try:
        import soundfile
        from reference_backends.scnet import SOURCE_HASHES, verify_source_files

        for key in THREAD_VARIABLES:
            os.environ[key] = str(args.threads)
        os.environ["CRISPASR_SCNET_GPU"] = "0"
        os.environ["CRISPASR_SCNET_THREADS"] = str(args.threads)
        os.environ["CRISPASR_SCNET_BENCH"] = "0"
        if not args.diagnostic:
            os.environ.pop("CRISPASR_SCNET_SCALAR_GATES", None)
        os.environ["ACCELERATE_USE_CPU"] = "true"
        verify_source_files(args.source_dir)
        receipt["source_file_hashes"] = dict(SOURCE_HASHES)
        audio, rate = soundfile.read(args.audio, dtype="float32", always_2d=True)
        audio = np.ascontiguousarray(audio)
        if rate != 44100 or audio.shape[1] != 2 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError("fixture must be finite, nonempty, 44100Hz stereo audio")
        receipt.update(checkpoint_sha256=sha256(args.checkpoint), reference_sha256=sha256(args.reference),
                       audio_sha256=sha256(args.audio), input_pcm_interleaved_sha256=pcm_sha256(audio),
                       diff_sha256=sha256(args.diff), library_sha256=sha256(args.library) if args.library else None)
        if args.fixture_source is not None:
            with soundfile.SoundFile(args.fixture_source) as original:
                if original.samplerate != rate or original.channels != 2:
                    raise ValueError("source audio must have the same 44100Hz stereo geometry")
                original.seek(args.fixture_offset_samples)
                section = original.read(len(audio), dtype="float32", always_2d=True)
            if not np.array_equal(section, audio):
                raise ValueError("fixture is not the exact contiguous decoded source interval")
            receipt["fixture"] = {"source_url": args.fixture_source_url, "source_sha256": sha256(args.fixture_source),
                                  "offset_samples": args.fixture_offset_samples, "samples": len(audio),
                                  "duration_seconds": len(audio) / rate, "contiguous_exact_pcm": True}
        reference = GGUFFile(args.reference)
        profile = receipt["profile"] = validate_reference(reference, audio, receipt["checkpoint_sha256"],
                                                          args.diagnostic, SOURCE_HASHES, args.variant, args.threads)
        models = {"f16": args.f16, "q8_0": args.q8_0, "q4_k": args.q4_k}
        base = GGUFFile(args.f16)
        receipt["status"] = "stage_comparison"
        save()
        for precision, path in models.items():
            row = receipt["models"][precision] = {"numerical_acceptance": False}
            try:
                model = GGUFFile(path)
                row["coverage"] = check_inventory(model, precision, None if precision == "f16" else base,
                                                   receipt["checkpoint_sha256"], SOURCE_HASHES)
                inventory = model.inventory()
                row.update(sha256=inventory["sha256"], file_bytes=inventory["file_bytes"],
                           elements_by_type=inventory["elements_by_type"])
                atomic_json_write(args.output_dir / f"inventory-{precision}.json", portable_receipt(inventory, args))
                capture = args.output_dir / f"native-{precision}.gguf"
                command = [str(args.diff), "scnet", str(path), str(args.reference), str(args.audio)]
                environment = os.environ.copy()
                environment["CRISPASR_SCNET_CAPTURE_GGUF"] = str(capture)
                log_path = args.output_dir / f"diff-{precision}.log"
                event = {"argv": command, "log": log_path.name, "complete": False}
                receipt["commands"].append(event)
                save()
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                               timeout=args.timeout, env=environment, check=False)
                event.update(complete=True, returncode=completed.returncode)
                save()
                if capture.is_file():
                    row["native_capture_sha256"] = sha256(capture)
                    row["stages"] = compare_captures(GGUFFile(capture), reference, precision)
                    save()
                row["native_log_metrics"] = parse_diff(log_path.read_text(), precision)
                if completed.returncode:
                    raise RuntimeError(f"native stage comparison exited {completed.returncode}")
                if "stages" not in row:
                    raise ValueError("native did not write the required independently checked capture")
                row["numerical_acceptance"] = all_pass(row["stages"])
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            save()
        receipt["numerical_acceptance"] = all(row["numerical_acceptance"] for row in receipt["models"].values())
        if not args.stages_only:
            receipt["status"] = "paired_benchmark"
            save()
            benchmark(args, models, reference, audio, profile, receipt, save)
        for name, path in models.items():
            recorded = receipt["models"][name].get("sha256")
            if recorded is not None and sha256(path) != recorded:
                raise ValueError(f"{name} model changed during validation")
        for key, path in (("checkpoint", args.checkpoint), ("reference", args.reference), ("audio", args.audio),
                          ("diff", args.diff), ("library", args.library)):
            if path is not None and sha256(path) != receipt[key + "_sha256"]:
                raise ValueError(f"{key} artifact changed during validation")
        verify_source_files(args.source_dir)
        receipt["complete"] = True
        receipt["acceptance"] = bool(not args.diagnostic and not args.stages_only and profile["full_profile"] and
                                     receipt["numerical_acceptance"] and receipt.get("benchmark", {}).get("acceptance"))
        receipt["status"] = "passed" if receipt["acceptance"] else "not_qualified"
    except BaseException as error:
        receipt["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "error"
        receipt["errors"].append(f"{type(error).__name__}: {error}")
        receipt["acceptance"] = False
    finally:
        save()
    print(json.dumps({"receipt": str(receipt_path), "complete": receipt["complete"],
                      "acceptance": receipt["acceptance"], "status": receipt["status"]}))
    return 0 if receipt["acceptance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
