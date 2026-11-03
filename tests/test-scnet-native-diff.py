#!/usr/bin/env python3
"""Exercise the production SCNet comparator using a model's own native captures.

This is a comparator regression test, never an independent model-fidelity test.
It needs a built scnet-probe and a supported model; random full-schema weights
are sufficient. Only NumPy is required, not PyTorch or the Python gguf package.
Run once per F16/Q8_0/Q4_K artifact to exercise all three acceptance profiles.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "scnet_validation", Path(__file__).resolve().parents[1] / "tools" / "validate_scnet.py")
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


def string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def write_reference(path, tensors, metadata):
    """Independent small F32 GGUF emitter; dimensions are already in ggml order."""
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for name, value in metadata.items():
        header.extend(string(name))
        if isinstance(value, str):
            header.extend(struct.pack("<I", 8) + string(value))
        elif isinstance(value, float):
            header.extend(struct.pack("<If", 6, value))
        else:
            header.extend(struct.pack("<IQ", 10, value))
    payload = bytearray()
    for name, shape, values in tensors:
        header.extend(string(name) + struct.pack("<I", len(shape)))
        header.extend(struct.pack("<" + "Q" * len(shape), *shape))
        header.extend(struct.pack("<IQ", 0, len(payload)))
        payload.extend(np.ascontiguousarray(values, dtype="<f4").tobytes())
        payload.extend(b"\0" * (-len(payload) % 32))
    header.extend(b"\0" * (-len(header) % 32))
    path.write_bytes(header + payload)


def stage_metrics(log, stage, expected_status):
    matches = [line for line in log.splitlines() if line.startswith(f"[{expected_status}] {stage} ")]
    if len(matches) != 1:
        raise AssertionError(f"expected one numerical {expected_status} for {stage}, got {matches}")
    fields = dict(re.findall(r"([a-z_][a-z_0-9]*)=([^\s]+)", matches[0]))
    assert fields.pop("row_policy") == "cosine_or_binary16_reference_floor_v1"
    values = {key: float(value) for key, value in fields.items()}
    if not {"cos_global", "norm_ratio", "relative_l2", "max_abs", "nonfinite", "n", "cos_min"} <= values.keys():
        raise AssertionError(f"incomplete numerical stage log: {matches[0]}")
    diagnostic_fields = {"rows", "row_width", "ne1", "ne2", "ne3", "row_cosine_gated", "rows_failing_cosine",
                         "nonfinite_rows", "worst_row", "worst_i1", "worst_i2", "worst_i3",
                         "worst_actual_norm", "worst_reference_norm", "worst_rms_error", "worst_max_abs"}
    if not diagnostic_fields <= values.keys():
        raise AssertionError(f"incomplete row diagnostics: {matches[0]}")
    assert values["rows"] * values["row_width"] == values["n"]
    assert values["rows"] == values["ne1"] * values["ne2"] * values["ne3"]
    assert values["worst_row"] == ((values["worst_i3"] * values["ne2"] + values["worst_i2"]) *
                                   values["ne1"] + values["worst_i1"])
    assert values["row_floor_epsilon"] == 2**-10
    assert values["row_cosine_threshold"] == .999
    assert values["row_rescued_count"] + values["row_failed_count"] == values["rows_failing_cosine"]
    np.testing.assert_allclose(values["row_rms_error_limit"], 2**-10 * values["reference_rms"], rtol=1e-15, atol=0)
    np.testing.assert_allclose(values["row_max_abs_error_limit"], 2**-10 * values["reference_peak"], rtol=1e-15, atol=0)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="New directory for captures, logs, and report")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.probe, args.model = args.probe.resolve(), args.model.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    inventory = V.GGUFFile(args.model)
    precision = {value: key for key, value in V.FILE_TYPES.items()}[inventory.metadata["general.file_type"]]
    metadata = {
        "general.architecture": "crispasr.reference",
        "scnet.reference.source_revision": inventory.metadata["general.source.revision"],
        "scnet.reference.checkpoint_sha256": inventory.metadata["scnet.checkpoint.sha256"],
        "scnet.reference.mode": "raw",
        "scnet.reference.segment_seconds": 20,
        "scnet.reference.shifts": 0,
        "scnet.reference.shift_seed": 0,
        "scnet.reference.overlap": 0.5,
        "scnet.reference.sample_rate": 44100,
        "scnet.reference.required_stage_count": 35,
    }
    environment = dict(os.environ, CRISPASR_SCNET_GPU="0", CRISPASR_SCNET_THREADS=str(args.threads))
    environment.pop("CRISPASR_SCNET_CAPTURE_GGUF", None)
    for name in V.THREAD_VARIABLES:
        environment[name] = str(args.threads)
    report = {
        "scope": "Production comparator controls using native self-captures; not independent model fidelity",
        "precision": precision, "probe": str(args.probe), "probe_sha256": V.sha256(args.probe),
        "model": str(args.model), "model_sha256": V.sha256(args.model), "controls": {},
    }
    reference = args.work_dir / "control-reference.gguf"

    def run(name, tensors, fields=None, model=None, capture=None, expected=1, diagnostic=None):
        write_reference(reference, tensors, metadata if fields is None else fields)
        env = environment.copy()
        if capture is not None:
            env["CRISPASR_SCNET_CAPTURE_GGUF"] = str(capture)
        completed = subprocess.run([str(args.probe), "scnet", str(model or args.model), str(reference)],
                                   env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   timeout=180, check=False)
        (args.work_dir / f"{name}.log").write_text(completed.stdout)
        if completed.returncode != expected:
            raise AssertionError(f"{name}: expected exit {expected}, got {completed.returncode}; see log")
        if diagnostic is not None and diagnostic not in completed.stdout:
            raise AssertionError(f"{name}: did not reach intended rejection: {diagnostic}")
        report["controls"][name] = {"returncode": completed.returncode, "acceptance": True}
        return completed.stdout

    # First execute all actual native stages. Placeholder stage shapes make this
    # comparison fail, while its saved capture supplies the subsequent controls.
    frames = 5000
    time = np.arange(frames, dtype=np.float32) / 44100
    audio = np.ascontiguousarray(np.stack([.12 * np.sin(220 * 2 * np.pi * time),
                                          .07 * np.cos(659 * 2 * np.pi * time)]), dtype="<f4")
    placeholders = [(name, [frames, 2], audio) if name == "input_audio" else
                    (name, [1], np.zeros(1, dtype="<f4")) for name in V.REQUIRED_STAGES]
    capture = args.work_dir / "native-capture.gguf"
    run("bootstrap-capture", placeholders, capture=capture, diagnostic="shape mismatch")
    native = V.GGUFFile(capture)
    assert set(native.tensors) == set(V.REQUIRED_STAGES)
    tensors = [(name, native.tensors[name]["shape_ggml"], native.floats(name).copy())
               for name in V.REQUIRED_STAGES]
    indices = {name: index for index, (name, _, _) in enumerate(tensors)}
    log = run("valid-self-capture", tensors, expected=0)
    assert len(V.parse_diff(log, precision)) == 35
    assert all(stage_metrics(log, name, "PASS")["cos_min"] > .999 for name in V.REQUIRED_STAGES)

    def replaced(stage, transform):
        changed = list(tensors)
        index = indices[stage]
        name, shape, values = changed[index]
        changed[index] = name, shape, transform(values.copy())
        return changed

    stage = "output_vocals"
    gain = {"f16": 1.02, "q8_0": 1.06, "q4_k": 1.30}[precision]
    log = run("altered-gain", replaced(stage, lambda values: values * np.float32(gain)))
    metrics = stage_metrics(log, stage, "FAIL")
    assert metrics["cos_global"] > V.THRESHOLDS[precision]
    assert metrics["norm_ratio"] < V.NORM_RATIO_BOUNDS[precision][0]
    assert metrics["nonfinite"] == 0
    report["controls"]["altered-gain"]["metrics"] = metrics

    if precision != "f16":
        # A controlled rotation plus gain keeps the looser quantized cosine and
        # norm gates satisfied while independently violating relative L2.
        cosine, ratio = (.981, 1.049) if precision == "q8_0" else (.941, 1.199)

        def excessive_relative_error(values):
            actual = values.astype(np.float64)
            length = np.linalg.norm(actual)
            unit = actual / length
            orthogonal = np.where(np.arange(len(actual)) % 2, -1.0, 1.0)
            orthogonal -= unit * np.dot(orthogonal, unit)
            orthogonal /= np.linalg.norm(orthogonal)
            return (cosine * unit + np.sqrt(1 - cosine * cosine) * orthogonal) * (length / ratio)

        log = run("excessive-relative-error", replaced(stage, excessive_relative_error))
        metrics = stage_metrics(log, stage, "FAIL")
        lower, upper = V.NORM_RATIO_BOUNDS[precision]
        assert metrics["cos_global"] > V.THRESHOLDS[precision] and lower <= metrics["norm_ratio"] <= upper
        assert metrics["relative_l2"] > V.RELATIVE_L2_MAX[precision] and metrics["nonfinite"] == 0
        report["controls"]["excessive-relative-error"]["metrics"] = metrics

    if precision == "f16":
        # Reverse one very low-energy learned row. Whole-tensor direction, gain,
        # and relative error still pass; only the new per-row gate should fail.
        selected = None
        for name, shape, values in tensors:
            if name not in V.F16_ROW_STAGES:
                continue
            rows = values.astype(np.float64).reshape(-1, shape[0])
            energy = np.einsum("ij,ij->i", rows, rows)
            stage_rms = np.sqrt(energy.sum() / values.size)
            candidates = np.flatnonzero((energy > (2**-10 * stage_rms) ** 2 * shape[0]) &
                                         (energy < (.018 * stage_rms) ** 2 * shape[0]) &
                                         (energy < energy.sum() * .00025))
            if len(candidates):
                selected = name, shape[0], int(candidates[-1])
                break
        assert selected is not None, "model must expose a nonzero low-energy learned row"
        stage, width, row = selected

        def negate_row(values):
            values[row * width:(row + 1) * width] *= -1
            return values

        log = run("altered-learned-row", replaced(stage, negate_row))
        metrics = stage_metrics(log, stage, "FAIL")
        assert metrics["cos_global"] > .999 and .99 <= metrics["norm_ratio"] <= 1.01
        assert metrics["relative_l2"] <= .05 and metrics["cos_min"] < -.999
        assert metrics["worst_row"] == row and metrics["rows_failing_cosine"] == 1
        assert metrics["row_failed_count"] == 1 and metrics["row_rescued_count"] == 0
        assert metrics["row_cosine_gated"] == 1 and metrics["nonfinite_rows"] == 0
        actual = tensors[indices[stage]][2][row * width:(row + 1) * width].astype(np.float64)
        expected = -actual
        np.testing.assert_allclose([metrics["worst_actual_norm"], metrics["worst_reference_norm"],
                                    metrics["worst_rms_error"], metrics["worst_max_abs"]],
                                   [np.linalg.norm(actual), np.linalg.norm(expected),
                                    np.sqrt(np.mean((actual - expected) ** 2)), np.max(np.abs(actual - expected))],
                                   rtol=1e-7, atol=0)
        if width <= 32:
            assert f"[ROW] {stage} worst_row={row} actual=[" in log
        report["controls"]["altered-learned-row"].update(stage=stage, row=row, metrics=metrics)

        # The same quiet row receives a small orthogonal error that fails raw
        # cosine but fits both reference-only error bounds. A larger error must
        # still fail the RMS bound while global metrics and the peak bound pass.
        stage_values = tensors[indices[stage]][2].astype(np.float64)
        reference_rms = np.sqrt(np.mean(stage_values * stage_values))
        orthogonal = np.where(np.arange(width) % 2, -1.0, 1.0)
        orthogonal -= actual * (np.dot(orthogonal, actual) / np.dot(actual, actual))
        orthogonal /= np.sqrt(np.mean(orthogonal * orthogonal))

        def row_error(scale):
            def apply(values):
                values[row * width:(row + 1) * width] += (scale * 2**-10 * reference_rms * orthogonal).astype(np.float32)
                return values
            return apply

        log = run("reference-floor-rescues-quiet-row", replaced(stage, row_error(.9)), expected=0)
        metrics = stage_metrics(log, stage, "PASS")
        assert metrics["cos_min"] <= .999 and metrics["rows_failing_cosine"] == 1
        assert metrics["row_rescued_count"] == 1 and metrics["row_failed_count"] == 0
        assert metrics["worst_rms_error"] <= metrics["row_rms_error_limit"]
        assert metrics["worst_max_abs"] <= metrics["row_max_abs_error_limit"]
        report["controls"]["reference-floor-rescues-quiet-row"]["metrics"] = metrics
        log = run("reference-floor-rejects-excess-rms", replaced(stage, row_error(1.1)))
        metrics = stage_metrics(log, stage, "FAIL")
        assert metrics["cos_global"] > .999 and .99 <= metrics["norm_ratio"] <= 1.01
        assert metrics["relative_l2"] <= .05 and metrics["cos_min"] <= .999
        assert metrics["row_rescued_count"] == 0 and metrics["row_failed_count"] == 1
        assert metrics["worst_rms_error"] > metrics["row_rms_error_limit"]
        assert metrics["worst_max_abs"] <= metrics["row_max_abs_error_limit"]
        report["controls"]["reference-floor-rejects-excess-rms"]["metrics"] = metrics

        log = run("zero-reference-stage-rejects-error", replaced(stage, np.zeros_like))
        metrics = stage_metrics(log, stage, "FAIL")
        assert metrics["reference_rms"] == metrics["reference_peak"] == 0
        assert metrics["row_rms_error_limit"] == metrics["row_max_abs_error_limit"] == 0
        assert metrics["row_rescued_count"] == 0 and metrics["row_failed_count"] > 0
        if inventory.metadata.get("test.synthetic"):
            # The documented random fixture has zero biases: a raw all-zero
            # mixture produces exactly zero learned stages. Zero reference
            # stages therefore pass only their exact zero-error counterpart.
            zeros = [(name, shape, values if name == "waveform_stats" else np.zeros_like(values))
                     for name, shape, values in tensors]
            log = run("zero-reference-stage-accepts-zero-error", zeros, expected=0)
            metrics = stage_metrics(log, stage, "PASS")
            assert metrics["reference_rms"] == metrics["reference_peak"] == 0
            assert metrics["max_abs"] == metrics["worst_rms_error"] == 0
            assert metrics["row_rescued_count"] == metrics["row_failed_count"] == 0

    def add_nan(values):
        values[0] = np.nan
        return values

    log = run("nonfinite-output", replaced("output_drums", add_nan))
    metrics = stage_metrics(log, "output_drums", "FAIL")
    assert metrics["nonfinite"] == 1 and metrics["nonfinite_rows"] == 1
    wrong_shape = list(tensors)
    index = indices["output_drums"]
    name, shape, values = tensors[index]
    wrong_shape[index] = name, [shape[1], shape[0], *shape[2:]], values
    run("wrong-stage-shape", wrong_shape, diagnostic="[FAIL] output_drums missing required stage or shape mismatch")

    # These malformed headers must fail before model evaluation, with a precise
    # provenance/profile or stage-inventory diagnostic, not a metric failure.
    for key in metadata:
        if key == "general.architecture":
            continue
        changed = dict(metadata)
        del changed[key]
        run("missing-" + key.rsplit(".", 1)[-1], tensors, changed, expected=2, diagnostic=key)
    invalid = {
        "segment-zero": ("segment_seconds", 0), "segment-out-of-range": ("segment_seconds", 120),
        "shifts-out-of-range": ("shifts", 17), "sample-rate": ("sample_rate", 48000),
        "overlap": ("overlap", .25), "stage-count": ("required_stage_count", 34),
        "mode": ("mode", "unspecified"), "source-revision": ("source_revision", "0" * 40),
        "malformed-sha": ("checkpoint_sha256", "z" * 64),
    }
    for label, (field, value) in invalid.items():
        changed = dict(metadata, **{"scnet.reference." + field: value})
        run("invalid-" + label, tensors, changed, expected=2, diagnostic="incompatible original-reference")
    for field in ("segment_seconds", "shifts", "shift_seed", "sample_rate", "required_stage_count", "overlap"):
        key = "scnet.reference." + field
        changed = dict(metadata, **{key: str(metadata[key])})
        run("wrong-type-" + field, tensors, changed, expected=2, diagnostic=key)
    changed = dict(metadata, **{"scnet.reference.checkpoint_sha256": "0" * 64})
    run("mismatched-checkpoint", tensors, changed, expected=2, diagnostic="does not match the model")
    run("missing-stage", tensors[:-1], expected=2, diagnostic="exactly all 35")
    run("extra-stage", [*tensors, ("unexpected_stage", [1], np.zeros(1, dtype="<f4"))],
        expected=2, diagnostic="exactly all 35")
    renamed = list(tensors)
    renamed[-1] = "unexpected_stage", renamed[-1][1], renamed[-1][2]
    run("substituted-stage", renamed, expected=2, diagnostic="exactly all 35")

    # Mutate only each model identity string in a disposable byte-for-byte copy,
    # preserving all tensor bytes. Correct reference metadata must then fail.
    with tempfile.TemporaryDirectory(prefix="model-identity-", dir=args.work_dir) as directory:
        altered_model = Path(directory) / "model.gguf"
        for key in ("general.source.revision", "scnet.checkpoint.sha256"):
            original = inventory.metadata[key]
            replacement = ("0" if original[0] != "0" else "1") + original[1:]
            needle = string(key) + struct.pack("<I", 8) + string(original)
            data = args.model.read_bytes()
            assert data.count(needle) == 1
            altered_model.write_bytes(data.replace(needle, string(key) + struct.pack("<I", 8) + string(replacement), 1))
            run("model-mismatch-" + key.rsplit(".", 1)[-1], tensors, model=altered_model,
                expected=2, diagnostic="does not match the model")

    reference.unlink()
    report["acceptance"] = True
    report["control_count"] = len(report["controls"])
    (args.work_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"PASS {precision}: {len(report['controls'])} production comparator controls; "
          "native self-consistency only, not model fidelity")


if __name__ == "__main__":
    main()
