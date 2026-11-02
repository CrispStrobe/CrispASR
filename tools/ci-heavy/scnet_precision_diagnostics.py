#!/usr/bin/env python3
"""Diagnose SCNet F16 weight rounding separately from native arithmetic.

A is the immutable, independently captured original F32 reference. B executes
the same original model and inference methods in F32 after rounding only
rank>=2 parameters through F16. C is the native F16 capture; optional D is a
native F32 capture. B and C never become qualification oracles. Every report
is explicitly nonqualifying, regardless of any pairwise numerical agreement.

The B archive deliberately omits scnet.reference.* provenance, preventing its
accidental use by the qualification driver as an original reference. Capture
archives remain worker-local. Public JSON includes metrics and bounded samples
from failed internal learned rows, never waveform or STFT sample values.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import heapq
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
if __name__ == "__main__":
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--threads", type=int, default=2)
    early_args, _ = early.parse_known_args()
    for variable in THREAD_VARIABLES:
        os.environ[variable] = str(early_args.threads)
    os.environ["ACCELERATE_USE_CPU"] = "true"

import numpy as np  # noqa: E402

import validate_scnet as validation  # noqa: E402
from dump_scnet_reference import ReferenceArchive  # noqa: E402
from reference_backends.scnet import (  # noqa: E402
    SOURCE_HASHES, SOURCE_REVISION, capture_original, load_original, sha256, verify_source_files,
)


def learned_stage(name):
    return name.startswith(("skip", "encoder", "decoder", "dp", "feature"))


def row_coordinates(name, shape, index):
    ne = validation.shape4(shape)
    i1 = index % ne[1]
    i2 = (index // ne[1]) % ne[2]
    i3 = index // (ne[1] * ne[2])
    coordinates = {"flat_row": index, "ggml_i1": i1, "ggml_i2": i2, "ggml_i3": i3}
    if learned_stage(name) or name in ("stft", "normalized"):
        coordinates.update(batch=i3, channel=i2, frequency=i1)
    elif name.endswith("audio") or name.startswith(("output_", "raw_output_")):
        coordinates.update(channel=i1)
    return coordinates


def worst_rows(name, actual, expected, shape, count=5, max_values=32):
    """Find low-cosine rows with bounded working storage and explicit scales."""
    ne = validation.shape4(shape)
    width = ne[0]
    if actual.shape != expected.shape or actual.size % width:
        raise ValueError(f"invalid row geometry: {name}")
    a, b = actual.reshape(-1, width), expected.reshape(-1, width)
    retained = []
    nonfinite = failed = zero_pairs = zero_mismatches = 0
    reference_energy = 0.0
    for begin in range(0, len(a), max(1, 262144 // width)):
        stop = min(len(a), begin + max(1, 262144 // width))
        x, y = a[begin:stop].astype(np.float64), b[begin:stop].astype(np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            nonfinite += int(np.count_nonzero(~np.isfinite(x)) + np.count_nonzero(~np.isfinite(y)))
            raise ValueError(f"{name}: nonfinite row data ({nonfinite} values)")
        aa, bb = np.einsum("ij,ij->i", x, x), np.einsum("ij,ij->i", y, y)
        dots = np.einsum("ij,ij->i", x, y)
        valid = (aa > 0) & (bb > 0)
        cosine = np.where(aa == bb, 1.0, 0.0)
        cosine[valid] = dots[valid] / np.sqrt(aa[valid] * bb[valid])
        reference_energy += float(np.sum(bb))
        zero_pairs += int(np.count_nonzero((aa == 0) & (bb == 0)))
        zero_mismatches += int(np.count_nonzero((aa == 0) != (bb == 0)))
        failed += int(np.count_nonzero(cosine <= validation.THRESHOLDS["f16"]))
        # Keep the smallest cosines. Row number breaks ties deterministically.
        for relative_index, value in enumerate(cosine):
            row = begin + relative_index
            item = (-float(value), -row, row)
            if len(retained) < count:
                heapq.heappush(retained, item)
            elif item > retained[0]:
                heapq.heapreplace(retained, item)
    result = []
    for _, _, row in sorted(retained, key=lambda item: (-item[0], item[2])):
        x, y = a[row].astype(np.float64), b[row].astype(np.float64)
        nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
        delta = x - y
        cosine = float(np.dot(x, y) / (nx * ny)) if nx and ny else float(nx == ny)
        data = {**row_coordinates(name, ne, row), "cosine": cosine,
                "actual_norm": nx, "reference_norm": ny,
                "reference_energy_fraction": ny * ny / reference_energy if reference_energy else 0.0,
                "norm_ratio": nx / ny if ny else (1.0 if nx == 0 else None),
                "relative_l2": float(np.linalg.norm(delta)) / ny if ny else (0.0 if nx == 0 else None),
                "rms_error": float(np.sqrt(np.mean(delta * delta))),
                "max_abs_error": float(np.max(np.abs(delta))), "row_width": width,
                "below_f16_row_cosine": cosine <= validation.THRESHOLDS["f16"]}
        if learned_stage(name) and data["below_f16_row_cosine"] and max_values:
            if width <= max_values:
                positions = np.arange(width)
            else:
                positions = np.sort(np.argsort(np.abs(delta), kind="stable")[-max_values:])
            data["internal_values"] = {"selection": "complete row" if width <= max_values else "largest absolute errors",
                                       "positions": positions.tolist(), "actual": x[positions].tolist(),
                                       "reference": y[positions].tolist(), "truncated": width > max_values}
        result.append(data)
    return {"rows": len(a), "row_width": width, "rows_below_f16_cosine": failed,
            "zero_row_pairs": zero_pairs, "zero_row_mismatches": zero_mismatches,
            "worst_rows": result}


def compare_pair(actual, expected, count, max_values, *, original_reference=False):
    if set(actual.tensors) != set(validation.REQUIRED_STAGES) or set(expected.tensors) != set(validation.REQUIRED_STAGES):
        raise ValueError("diagnostic captures must contain exactly the 35 actual required stages")
    reports = {}
    for name in validation.REQUIRED_STAGES:
        shape = expected.tensors[name]["shape_ggml"]
        if validation.shape4(actual.tensors[name]["shape_ggml"]) != validation.shape4(shape):
            raise ValueError(f"capture shape mismatch: {name}")
        a, b = actual.floats(name), expected.floats(name)
        stats = validation.tensor_metrics(a, b, "f16")
        stats["within_f16_global_limits"] = stats.pop("acceptance")
        if original_reference:
            stats.update(validation.row_metrics(a, b, validation.shape4(shape)[0]))
        stats.update(worst_rows(name, a, b, shape, count, max_values))
        if original_reference:
            for row in stats["worst_rows"]:
                within_floor = (row["rms_error"] <= stats["row_rms_error_limit"] and
                                row["max_abs_error"] <= stats["row_max_abs_error_limit"])
                row["within_reference_absolute_floor"] = within_floor
                row["passes_declared_f16_row_policy"] = not row["below_f16_row_cosine"] or within_floor
        stats["row_policy_evaluated_against_original"] = original_reference
        stats["shape_ggml"] = shape
        stats["f16_row_gate_applies"] = name in validation.F16_ROW_STAGES
        reports[name] = stats
    return reports


def round_matrix_parameters(model):
    """Round learned matrices only; keep all original computation in F32."""
    import torch

    reports = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.dtype != torch.float32 or parameter.device.type != "cpu":
                raise ValueError(f"original parameter is not F32 CPU: {name}")
            before = parameter.detach().cpu().numpy().copy()
            if not np.isfinite(before).all():
                raise ValueError(f"nonfinite original parameter: {name}")
            round_this = parameter.ndim >= 2
            if round_this:
                # Match convert-scnet-to-gguf.py's NumPy storage conversion.
                # The original model still executes with F32 parameters.
                with np.errstate(over="ignore", invalid="ignore"):
                    temporary = before.astype(np.float16)
                if not np.isfinite(temporary).all():
                    raise ValueError(f"F16 parameter-rounding overflow: {name}")
                parameter.copy_(torch.from_numpy(temporary.astype(np.float32)))
            after = parameter.detach().cpu().numpy()
            metrics = validation.tensor_metrics(after, before, "f16")
            metrics.pop("acceptance")
            reports[name] = {"shape": list(parameter.shape), "elements": parameter.numel(),
                             "rank_ge_two_rounded": round_this, "original_dtype": "float32",
                             "storage_after_roundtrip": "float32", "intermediate_dtype": "float16" if round_this else None,
                             "original_sha256": validation.pcm_sha256(before),
                             "rounded_sha256": validation.pcm_sha256(after),
                             "changed_elements": int(np.count_nonzero(before != after)), "difference": metrics}
            if not round_this and not np.array_equal(before, after):
                raise ValueError(f"rounding unexpectedly changed a norm/bias parameter: {name}")
    if not reports or not any(item["rank_ge_two_rounded"] for item in reports.values()):
        raise ValueError("original model has no matrix parameters to round")
    for name, buffer in model.named_buffers():
        if buffer.is_floating_point() and buffer.dtype != torch.float32:
            raise ValueError(f"original floating buffer is not F32: {name}")
    return reports


def verify_capture_input(capture, reference):
    if capture.metadata.get("scnet.reference.source_revision") != SOURCE_REVISION:
        raise ValueError("native capture has an incompatible source revision")
    for name in ("input_audio", "raw_input_audio"):
        if (name not in capture.tensors or
                validation.shape4(capture.tensors[name]["shape_ggml"]) !=
                validation.shape4(reference.tensors[name]["shape_ggml"])):
            raise ValueError(f"native capture has a different input/profile shape at {name}")
        # Whole-input bytes must be identical. In the full pipeline, native
        # waveform reduction can differ slightly in F32 before the first raw
        # forward; that difference is itself reported by C_vs_A/D_vs_A.
        if name == "input_audio" and not np.array_equal(capture.floats(name), reference.floats(name)):
            raise ValueError("native capture has different complete input PCM bytes")


def artifact_record(capture):
    return {"file": capture.path.name, "sha256": sha256(capture.path),
            "metadata": capture.metadata,
            "tensor_dtypes": {name: tensor["type"] for name, tensor in capture.tensors.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("source-dir", "checkpoint", "reference-a", "native-capture", "output-dir"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--variant", choices=("scnet", "scnet-large"), default="scnet")
    parser.add_argument("--native-f32-capture", type=Path)
    parser.add_argument("--native-f16-model", type=Path)
    parser.add_argument("--native-f32-model", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--worst-rows", type=int, default=5)
    parser.add_argument("--max-row-values", type=int, default=32)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 256 or not 1 <= args.worst_rows <= 32 or not 0 <= args.max_row_values <= 32:
        parser.error("threads must be 1..256, worst-rows 1..32, max-row-values 0..32")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "precision-diagnostics.json"
    report = {
        "schema": 1, "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "complete": False, "acceptance": False, "qualification": False,
        "scope": "Diagnostic only: A=original F32; B=original F32 arithmetic with rank>=2 weights rounded through F16; C=native F16; optional D=native F32.",
        "qualification_oracle": "A only; B/C/D cannot replace the original qualification reference",
        "capture_policy": "B archive is worker-local and deliberately lacks original-reference metadata; JSON stores no waveform or STFT samples.",
        "parameter_rounding_implementation": "NumPy float32 -> float16 -> float32, matching the GGUF converter's storage cast; original PyTorch forward stays F32",
        "fixed_f16_limits": {"cosine_strict_gt": validation.THRESHOLDS["f16"],
                             "norm_ratio": validation.NORM_RATIO_BOUNDS["f16"],
                             "relative_l2_max": validation.RELATIVE_L2_MAX["f16"]},
        "f16_row_policy": validation.F16_ROW_POLICY,
        "f16_row_floor_epsilon": validation.F16_ROW_FLOOR_EPSILON,
        "row_policy_scope": "Reported only for comparisons against original A; C/B remains a nonqualifying raw-error diagnostic.",
        "errors": [], "pairs": {},
    }
    save = lambda: validation.atomic_json_write(output, report)
    save()
    try:
        for variable in THREAD_VARIABLES:
            os.environ[variable] = str(args.threads)
        os.environ["ACCELERATE_USE_CPU"] = "true"
        import torch

        if torch.__version__.split("+")[0] != "2.8.0":
            raise ValueError("diagnostics require the upstream torch==2.8.0 CPU profile")
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        verify_source_files(args.source_dir)
        reference = validation.GGUFFile(args.reference_a)
        frames = reference.tensors["input_audio"]["shape_ggml"][0]
        audio = reference.floats("input_audio").reshape(2, frames).copy()
        profile = validation.validate_reference(reference, np.ascontiguousarray(audio.T),
                                                args.checkpoint_sha256, True, SOURCE_HASHES, args.variant, args.threads)
        captures = {"A": reference, "C": validation.GGUFFile(args.native_capture)}
        if args.native_f32_capture:
            captures["D"] = validation.GGUFFile(args.native_f32_capture)
        for key in ("C", "D"):
            if key in captures:
                verify_capture_input(captures[key], reference)
        report.update(source_revision=SOURCE_REVISION, source_file_hashes=SOURCE_HASHES,
                      checkpoint_sha256=args.checkpoint_sha256, variant=args.variant, profile=profile,
                      input_planar_f32_sha256=validation.pcm_sha256(audio), torch_version=torch.__version__,
                      torch_intraop_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                      thread_environment={key: os.environ.get(key) for key in THREAD_VARIABLES},
                      artifacts={key: artifact_record(value) for key, value in captures.items()})
        for label, path in (("native_f16_model", args.native_f16_model), ("native_f32_model", args.native_f32_model)):
            if path:
                model_file = validation.GGUFFile(path)
                if (model_file.metadata.get("general.source.revision") != SOURCE_REVISION or
                        model_file.metadata.get("scnet.checkpoint.sha256") != args.checkpoint_sha256):
                    raise ValueError(f"{label} does not identify the original checkpoint/source")
                if model_file.metadata.get("general.file_type") != (1 if label == "native_f16_model" else 0):
                    raise ValueError(f"{label} has the wrong declared storage precision")
                report["artifacts"][label] = artifact_record(model_file)
        save()
        model = load_original(args.source_dir, args.checkpoint, args.variant, "cpu",
                              expected_sha256=args.checkpoint_sha256)
        original_forward = type(model).forward.__code__
        report["parameter_rounding"] = round_matrix_parameters(model)
        save()
        b_path = args.output_dir / "rounded-original-B.nonqualifying.gguf"
        metadata = {
            "scnet.diagnostic.kind": "original-F32-arithmetic-with-F16-rounded-rank>=2-parameters",
            "scnet.diagnostic.qualification": False,
            "scnet.diagnostic.source_revision": SOURCE_REVISION,
            "scnet.diagnostic.checkpoint_sha256": args.checkpoint_sha256,
            "scnet.diagnostic.original_reference_a_sha256": report["artifacts"]["A"]["sha256"],
            "scnet.diagnostic.profile_json": json.dumps(profile, sort_keys=True),
        }
        with (args.output_dir / "original-rounded-B.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), ReferenceArchive(b_path, metadata) as archive:
                capture_original(model, audio, segment_seconds=profile["segment_seconds"], shifts=profile["shifts"],
                                 shift_seed=profile["shift_seed"], no_segment=profile["mode"] == "raw",
                                 capture=archive.capture)
        if type(model).forward.__code__ is not original_forward:
            raise ValueError("original forward method was modified during diagnostics")
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ValueError("rounded original model did not retain F32 arithmetic parameters")
        del model
        captures["B"] = validation.GGUFFile(b_path)
        if any(key.startswith("scnet.reference.") for key in captures["B"].metadata):
            raise ValueError("diagnostic B archive must not advertise original-reference provenance")
        for name in ("input_audio", "raw_input_audio"):
            if (validation.shape4(captures["B"].tensors[name]["shape_ggml"]) !=
                    validation.shape4(reference.tensors[name]["shape_ggml"]) or
                    not np.array_equal(captures["B"].floats(name), reference.floats(name))):
                raise ValueError(f"parameter rounding unexpectedly changed original input/profile: {name}")
        report["artifacts"]["B"] = artifact_record(captures["B"])
        save()
        pairs = (("B", "A"), ("C", "A"), ("C", "B")) + ((("D", "A"),) if "D" in captures else ())
        for actual, expected in pairs:
            name = f"{actual}_vs_{expected}"
            report["pairs"][name] = {"actual": actual, "reference": expected, "qualification": False,
                                      "interpretation": "diagnostic comparison only; no original gate is changed",
                                      "stages": compare_pair(captures[actual], captures[expected], args.worst_rows,
                                                             args.max_row_values, original_reference=expected == "A")}
            save()
        for key, capture in captures.items():
            if sha256(capture.path) != report["artifacts"][key]["sha256"]:
                raise ValueError(f"capture {key} changed during diagnostics")
        verify_source_files(args.source_dir)
        report["complete"] = True
    except BaseException as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        save()
    print(json.dumps({"report": str(output), "complete": report["complete"], "acceptance": False,
                      "qualification": False}))
    # Zero means this diagnostic executed successfully, never model acceptance.
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
