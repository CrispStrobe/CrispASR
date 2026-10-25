#!/usr/bin/env python3
"""Hosted C ABI capture of one fixed CSM backbone prefill, without sampling."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

import csm_original_prefill as original
import csm_original_rope_reference as rope_reference
import nightly_error_ab as ab


N_LAYERS = 16
D_MODEL = 2048
VOCAB = 2051
EXPECTED = {"backbone_layer0_normed": [14, D_MODEL],
            **{f"backbone_layer{i}_output": [14, D_MODEL] for i in range(N_LAYERS)},
            "backbone_prefill_last_h": [D_MODEL],
            "backbone_prefill_c0_logits": [VOCAB]}


class ContextParams(ctypes.Structure):
    # src/csm_tts.h: csm_tts_context_params (native x86_64 layout).
    _fields_ = [("n_threads", ctypes.c_int), ("verbosity", ctypes.c_int),
                ("use_gpu", ctypes.c_bool), ("temperature", ctypes.c_float),
                ("topk", ctypes.c_int), ("seed", ctypes.c_uint64),
                ("max_audio_tokens", ctypes.c_int)]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def hosted_paths(args):
    if os.environ.get("GITHUB_ACTIONS") != "true" or not os.environ.get("RUNNER_TEMP"):
        raise RuntimeError("native prefill capture requires a hosted GitHub Actions runner")
    scratch_value, out_value = os.environ.get("HEAVY_SCRATCH"), os.environ.get("HEAVY_OUT")
    if not scratch_value or not out_value:
        raise RuntimeError("HEAVY_SCRATCH and HEAVY_OUT are required")
    scratch, hosted_out = Path(scratch_value).resolve(strict=True), Path(out_value).resolve(strict=True)
    library, model = Path(args.library).resolve(strict=True), Path(args.model).resolve(strict=True)
    reference, out = Path(args.reference).resolve(strict=True), Path(args.out).resolve()
    if not (scratch.is_dir() and hosted_out.is_dir() and scratch != hosted_out and
            library.is_file() and model.is_file() and reference.is_dir() and
            library.is_relative_to(scratch) and model.is_relative_to(scratch) and
            (reference.is_relative_to(scratch) or reference.is_relative_to(hosted_out)) and
            out.is_relative_to(hosted_out) and out != hosted_out and
            not out.is_relative_to(reference) and not reference.is_relative_to(out)):
        raise RuntimeError("library/model must be in scratch, reference in scratch/output, capture in output")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise RuntimeError("capture output directory must be empty")
    return library, model, reference, out


def verify_reference(reference):
    receipt = json.loads((reference / "receipt.json").read_text())
    if receipt.get("complete") is not True or receipt.get("source_revision") != rope_reference.REVISION or \
            receipt.get("config_revision") != rope_reference.CONFIG_REVISION or \
            receipt.get("config_sha256") != rope_reference.CONFIG_SHA256 or \
            receipt.get("prompt_tokens") != original.TOKENS or receipt.get("text") != original.TEXT:
        raise RuntimeError("original F32 prefill receipt has wrong pins, prompt or completion state")
    if sha256(reference / "original-config.json") != rope_reference.CONFIG_SHA256:
        raise RuntimeError("original F32 prefill configuration differs from immutable pin")
    files = receipt.get("files", {})
    if {key: value.get("shape") for key, value in files.items()} != EXPECTED:
        raise RuntimeError("original F32 prefill stages or shapes differ from fixed C ABI contract")
    for name, metadata in files.items():
        if metadata.get("dtype") != "float32":
            raise RuntimeError(f"{name}: original dtype is not float32")
        for suffix, field in (("bin", "raw_sha256"), ("npy", "npy_sha256")):
            if sha256(reference / f"{name}.{suffix}") != metadata.get(field):
                raise RuntimeError(f"{name}: original {suffix} file hash differs from receipt")
    return receipt


def linked_libraries(library, out):
    proc = subprocess.run(["ldd", str(library)], capture_output=True, text=True, check=True, timeout=30)
    (out / "ldd.log").write_text(proc.stdout + proc.stderr)
    paths = {library}
    if "not found" in proc.stdout:
        raise RuntimeError("native library has an unresolved dependency")
    for line in proc.stdout.splitlines():
        match = re.search(r"(?:=>\s*|^\s*)(/[^\s(]+)", line)
        if match:
            paths.add(Path(match.group(1)).resolve(strict=True))
    return [file_record(path) for path in sorted(paths)]


def mapped_libraries(library):
    mapped = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = Path(fields[5].removesuffix(" (deleted)"))
        if path.exists() and (path.resolve() == library or
                              any(name in path.name for name in ("libggml", "libcrispasr", "libopenblas"))):
            mapped.add(path.resolve())
    if library not in mapped:
        raise RuntimeError("requested C ABI library was not mapped into this process")
    return [file_record(path) for path in sorted(mapped)]


def run(args):
    library, model, reference, out = hosted_paths(args)
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"complete": False, "pass": False, "scope": __doc__, "argv": sys.argv,
               "rope": args.rope, "reference": str(reference), "library": str(library),
               "model": str(model), "files": {}, "stages": {}}

    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    def headroom(stage):
        state = ab.snapshot()
        receipt["stages"][stage] = state
        save()
        ab.require_headroom(state)
        if int(state["memory"]["MemAvailable"].split()[0]) < 8 * 1024**2:
            raise RuntimeError(f"native F16 prefill requires 8 GiB available before {stage}")

    save()
    try:
        if sys.byteorder != "little" or os.uname().machine != "x86_64":
            raise RuntimeError("native CSM dump requires hosted little-endian x86_64")
        if ctypes.sizeof(ContextParams) != 40:
            raise RuntimeError("C ABI context parameter layout differs from expected 40 bytes")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", args.model_sha256):
            raise RuntimeError("--model-sha256 must be an exact 64-digit hex digest")
        original_receipt = verify_reference(reference)
        receipt["reference_receipt_sha256"] = sha256(reference / "receipt.json")
        receipt["reference_prompt_tokens"] = original_receipt["prompt_tokens"]
        receipt["library"] = file_record(library)
        receipt["model"] = file_record(model)
        if receipt["model"]["sha256"] != args.model_sha256.lower():
            raise RuntimeError("native model bytes differ from the caller's physical F16 pin")
        receipt["model_pin_sha256"] = args.model_sha256.lower()
        # Set the two diagnostic switches explicitly before loading the C ABI.
        os.environ["CRISPASR_CSM_ORIGINAL_ROPE"] = "1" if args.rope == "original" else "0"
        os.environ["CRISPASR_CORE_ATTN_EAGER_F32"] = "1" if args.attention == "eager-f32" else "0"
        os.environ.pop("CRISPASR_CSM_TRACE_PREFIX", None)
        receipt["environment"] = {
            "CRISPASR_CSM_ORIGINAL_ROPE": os.environ["CRISPASR_CSM_ORIGINAL_ROPE"],
            "CRISPASR_CORE_ATTN_EAGER_F32": os.environ["CRISPASR_CORE_ATTN_EAGER_F32"],
            "CRISPASR_CSM_TRACE_PREFIX": None}
        headroom("before_load")
        receipt["ldd_libraries"] = linked_libraries(library, out)
        receipt["ldd_log"] = file_record(out / "ldd.log")
        save()
        lib = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
        lib.csm_tts_context_default_params.argtypes = []
        lib.csm_tts_context_default_params.restype = ContextParams
        lib.csm_tts_init_from_file.argtypes = [ctypes.c_char_p, ContextParams]
        lib.csm_tts_init_from_file.restype = ctypes.c_void_p
        lib.csm_tts_free.argtypes = [ctypes.c_void_p]
        lib.csm_tts_free.restype = None
        float_ptr = ctypes.POINTER(ctypes.c_float)
        layer_ptrs = ctypes.POINTER(float_ptr)
        lib.csm_tts_run_backbone_dump.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                                                  float_ptr, layer_ptrs, ctypes.c_int, float_ptr, float_ptr]
        lib.csm_tts_run_backbone_dump.restype = ctypes.c_int
        params = lib.csm_tts_context_default_params()
        params.n_threads, params.verbosity, params.use_gpu = 4, 0, False
        params.temperature, params.topk, params.seed, params.max_audio_tokens = 0.9, 50, 1234, 0
        receipt["params"] = {name: getattr(params, name) for name, _ in ContextParams._fields_}
        save()
        prompt_t = len(original.TOKENS)
        layer0 = np.empty((prompt_t, D_MODEL), dtype="<f4")
        layers = [np.empty((prompt_t, D_MODEL), dtype="<f4") for _ in range(N_LAYERS)]
        last_h = np.empty(D_MODEL, dtype="<f4")
        c0 = np.empty(VOCAB, dtype="<f4")
        ptrs = (float_ptr * N_LAYERS)(*(array.ctypes.data_as(float_ptr) for array in layers))
        ctx = lib.csm_tts_init_from_file(os.fsencode(model), params)
        if not ctx:
            raise RuntimeError("csm_tts_init_from_file failed")
        try:
            receipt["mapped_runtime_libraries"] = mapped_libraries(library)
            headroom("before_inference")
            returned = lib.csm_tts_run_backbone_dump(ctx, original.TEXT.encode(), prompt_t,
                                                     layer0.ctypes.data_as(float_ptr), ptrs,
                                                     N_LAYERS, last_h.ctypes.data_as(float_ptr),
                                                     c0.ctypes.data_as(float_ptr))
        finally:
            lib.csm_tts_free(ctx)
        receipt["returned_tokens"] = returned
        if returned != prompt_t:
            raise RuntimeError("native C ABI dump returned a different prompt token count")
        arrays = {"backbone_layer0_normed": layer0,
                  **{f"backbone_layer{i}_output": value for i, value in enumerate(layers)},
                  "backbone_prefill_last_h": last_h, "backbone_prefill_c0_logits": c0}
        if {key: list(value.shape) for key, value in arrays.items()} != EXPECTED:
            raise RuntimeError("native C ABI dump produced unexpected stage shapes")
        for name, value in arrays.items():
            if not np.isfinite(value).all():
                raise RuntimeError(f"native C ABI dump produced nonfinite {name}")
            raw, npy = out / f"{name}.bin", out / f"{name}.npy"
            value.tofile(raw)
            np.save(npy, value, allow_pickle=False)
            receipt["files"][name] = {"shape": list(value.shape), "dtype": "float32",
                                      "raw_sha256": sha256(raw), "npy_sha256": sha256(npy)}
            save()
        receipt["complete"] = True
        receipt["pass"] = True
        save()
        print(json.dumps({"complete": True, "out": str(out), "receipt": str(out / "receipt.json")},
                         sort_keys=True))
        return receipt
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rope", choices=("legacy", "original"), default="original")
    parser.add_argument("--attention", choices=("default", "eager-f32"), default="default")
    args = parser.parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"native CSM prefill capture failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
