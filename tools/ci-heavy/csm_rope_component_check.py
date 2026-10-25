#!/usr/bin/env python3
"""Check candidate CSM RoPE operator arrays against pinned original fixtures.

This checks only one small rotary component, not synthesis or model acceptance.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


SOURCE_REVISION = "798f948e88fd0b93fc515ec6b96e0503b78ad6ba"
CONFIG_SHA256 = "b203c014cb5a2f7b4f98d2e945f091182aceb17fa530ce968e8c3437e01a9b70"
CASES = ("backbone", "depth")
REFERENCE_KEYS = ("positions", "q", "k", "q_rotated", "k_rotated",
                  "q_unscaled", "k_unscaled", "freq_factors")
ROTATION_GATES = {"rel_l2": 1e-4, "norm_ratio_delta": 1e-6,
                  "cosine_min": 1 - 1e-8, "max_abs": 1e-3}
FACTOR_GATES = {"rel_l2": 2e-6, "max_abs": 1e-4}


def _read_array(path: Path, shape: tuple[int, ...], dtype: str, *, digest=None):
    if dtype not in ("float32", "int32"):
        raise ValueError(f"{path}: unsupported dtype {dtype!r}")
    if not shape or len(shape) > 3 or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError(f"{path}: invalid shape {shape!r}")
    count = math.prod(shape)
    if count > 1_000_000:
        raise ValueError(f"{path}: array exceeds component size cap")
    if path.stat().st_size != count * 4:
        raise ValueError(f"{path}: expected {count * 4} bytes, found {path.stat().st_size}")
    data = path.read_bytes()
    if len(data) != count * 4:
        raise ValueError(f"{path}: expected {count * 4} bytes, found {len(data)}")
    if digest is not None and hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"{path}: reference hash mismatch")
    array = np.frombuffer(data, dtype="<f4" if dtype == "float32" else "<i4").reshape(shape)
    if dtype == "float32" and not np.isfinite(array).all():
        raise ValueError(f"{path}: nonfinite array")
    return array


def _reference_array(root: Path, files: dict, case: str, key: str):
    entry = files.get(key)
    if not isinstance(entry, dict) or not isinstance(entry.get("raw_sha256"), str):
        raise ValueError(f"{case}/{key}: missing reference metadata or hash")
    shape = entry.get("shape")
    if not isinstance(shape, list):
        raise ValueError(f"{case}/{key}: missing shape")
    return _read_array(root / case / f"{key}.bin", tuple(shape),
                       entry.get("dtype"), digest=entry["raw_sha256"])


def _metrics(expected, actual):
    a = expected.astype(np.float64).ravel()
    b = actual.astype(np.float64).ravel()
    anorm = float(np.linalg.norm(a))
    bnorm = float(np.linalg.norm(b))
    if anorm == 0 or bnorm == 0:
        raise ValueError("reference or candidate has zero norm")
    delta = b - a
    return {"rel_l2": float(np.linalg.norm(delta) / anorm),
            "norm_ratio": bnorm / anorm,
            "cosine": float(np.dot(a, b) / (anorm * bnorm)),
            "max_abs": float(np.max(np.abs(delta)))}


def _rotation_pass(metrics):
    return (metrics["rel_l2"] <= ROTATION_GATES["rel_l2"] and
            abs(metrics["norm_ratio"] - 1) <= ROTATION_GATES["norm_ratio_delta"] and
            metrics["cosine"] >= ROTATION_GATES["cosine_min"] and
            metrics["max_abs"] <= ROTATION_GATES["max_abs"])


def _factor_pass(metrics):
    return metrics["rel_l2"] <= FACTOR_GATES["rel_l2"] and metrics["max_abs"] <= FACTOR_GATES["max_abs"]


def compare(reference_dir: Path, candidate_dir: Path) -> dict:
    """Return a complete receipt; malformed inputs raise ValueError/OSError."""
    reference_dir, candidate_dir = Path(reference_dir), Path(candidate_dir)
    manifest = json.loads((reference_dir / "receipt.json").read_text())
    if manifest.get("complete") is not True or manifest.get("source_revision") != SOURCE_REVISION or \
            manifest.get("config_sha256") != CONFIG_SHA256:
        raise ValueError("reference incomplete or original source/config pins differ")
    if set(manifest.get("cases", {})) != set(CASES):
        raise ValueError("reference must contain exactly backbone and depth cases")
    receipt = {"pass": True, "source_revision": SOURCE_REVISION,
               "config_sha256": CONFIG_SHA256, "rotation_gates": ROTATION_GATES,
               "factor_gates": FACTOR_GATES, "cases": {}}
    for case in CASES:
        files = manifest["cases"][case].get("files", {})
        ref = {key: _reference_array(reference_dir, files, case, key) for key in REFERENCE_KEYS}
        positions = ref["positions"]
        if positions.dtype != np.dtype("int32") or positions.ndim != 1 or \
                np.any(positions < 0) or np.any(np.diff(positions.astype(np.int64)) <= 0):
            raise ValueError(f"{case}: invalid positions")
        for key in ("q", "k", "q_rotated", "k_rotated", "q_unscaled", "k_unscaled", "freq_factors"):
            if ref[key].dtype != np.dtype("float32"):
                raise ValueError(f"{case}/{key}: expected float32")
        for prefix in ("q", "k"):
            shape = ref[prefix].shape
            if len(shape) != 3 or shape[0] != positions.size or shape[2] % 2 or \
                    ref[f"{prefix}_rotated"].shape != shape or ref[f"{prefix}_unscaled"].shape != shape:
                raise ValueError(f"{case}: inconsistent {prefix} shapes")
        if ref["q"].shape[2] != ref["k"].shape[2] or \
                ref["freq_factors"].shape != (ref["q"].shape[2] // 2,) or \
                np.any(ref["freq_factors"] <= 0):
            raise ValueError(f"{case}: inconsistent frequency factors")
        result = {"candidate": {}, "unscaled_negative_control": {}}
        for prefix in ("q", "k"):
            key = f"{prefix}_rotated"
            old = _metrics(ref[key], ref[f"{prefix}_unscaled"])
            if _rotation_pass(old):
                raise ValueError(f"{case}/{prefix}: unscaled original passes positive gates")
            result["unscaled_negative_control"][prefix] = old
        for key in ("q_rotated", "k_rotated", "freq_factors"):
            actual = _read_array(candidate_dir / case / f"{key}.bin", ref[key].shape, "float32")
            metrics = _metrics(ref[key], actual)
            passed = _factor_pass(metrics) if key == "freq_factors" else _rotation_pass(metrics)
            result["candidate"][key] = {"pass": passed, **metrics}
            receipt["pass"] &= passed
        receipt["cases"][case] = result
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = compare(args.reference_dir, args.candidate_dir)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        receipt = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
