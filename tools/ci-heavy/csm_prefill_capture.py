#!/usr/bin/env python3
"""Hosted-only, read-only CSM prefill capture for one CPU profile per process.

This calls the existing C ABI dump. It never generates audio or samples tokens.
Use a fresh process for each library so the dynamic loader cannot alias ggml
symbols from a different CPU build. The sampled-path trace is the input oracle.
"""

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

import csm_cpu_profile_ab as profile
import nightly_error_ab as ab


MODEL_SHA256 = "1123f9e8ff2d9212700d1fface749b8ec4f9765569bb996667a15a9a9832ee13"
TEXT = "The quick brown fox jumps over the lazy dog"
N_LAYERS = 16
D_MODEL = 2048
VOCAB = 2051
MAX_TOKENS = 256


class ContextParams(ctypes.Structure):
    # src/csm_tts.h: csm_tts_context_params; native C alignment is intentional.
    _fields_ = [("n_threads", ctypes.c_int), ("verbosity", ctypes.c_int),
                ("use_gpu", ctypes.c_bool), ("temperature", ctypes.c_float),
                ("topk", ctypes.c_int), ("seed", ctypes.c_uint64),
                ("max_audio_tokens", ctypes.c_int)]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def hosted_paths(args):
    if os.environ.get("GITHUB_ACTIONS") != "true" or not os.environ.get("RUNNER_TEMP"):
        raise RuntimeError("this helper requires a GitHub Actions hosted heavy-CPU runner")
    scratch_value, out_value = os.environ.get("HEAVY_SCRATCH"), os.environ.get("HEAVY_OUT")
    if not scratch_value or not out_value:
        raise RuntimeError("HEAVY_SCRATCH and HEAVY_OUT are required")
    scratch, hosted_out = Path(scratch_value).resolve(), Path(out_value).resolve()
    if not scratch.is_dir() or not hosted_out.is_dir() or scratch == hosted_out:
        raise RuntimeError("hosted scratch/output directories must exist and differ")
    library, model = Path(args.library).resolve(strict=True), Path(args.model).resolve(strict=True)
    trace_prefix, out = Path(args.trace_prefix).resolve(), Path(args.out).resolve()
    if not (library.is_file() and model.is_file() and library.is_relative_to(scratch)
            and model.is_relative_to(scratch) and trace_prefix.is_relative_to(hosted_out)
            and out.is_relative_to(hosted_out) and out != hosted_out):
        raise RuntimeError("library/model must be in HEAVY_SCRATCH; trace/output in HEAVY_OUT")
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("output directory must be empty to preserve one-profile provenance")
    return library, model, trace_prefix, out


def loaded_libraries(library, out):
    proc = subprocess.run(["ldd", str(library)], capture_output=True, text=True, check=True)
    (out / "ldd.log").write_text(proc.stdout + proc.stderr)
    paths = {library}
    for line in proc.stdout.splitlines():
        match = re.search(r"(?:=>\s*|^\s*)(/[^\s(]+)", line)
        if match:
            paths.add(Path(match.group(1)).resolve(strict=True))
    return [file_record(path) for path in sorted(paths)]


def mapped_runtime_libraries(library):
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
        raise RuntimeError("requested shared library is absent from actual process maps")
    return [file_record(path) for path in sorted(mapped)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, help="one profile's shared library exporting the CSM C ABI")
    parser.add_argument("--model", required=True, help="pinned CSM Q4_K GGUF in HEAVY_SCRATCH")
    parser.add_argument("--trace-prefix", required=True, help="sampled CLI trace prefix in HEAVY_OUT")
    parser.add_argument("--out", required=True, help="new, empty evidence directory in HEAVY_OUT")
    parser.add_argument("--text", default=TEXT)
    args = parser.parse_args()
    library, model, trace_prefix, out = hosted_paths(args)
    if sys.byteorder != "little" or os.uname().machine != "x86_64":
        raise RuntimeError("trace ABI is supported only on hosted little-endian x86_64")
    if args.text != TEXT:
        raise RuntimeError("only the pinned regression phrase is supported")

    # Complete all validation and the resource guard before CDLL/model loading.
    trace = profile.read_profile_trace(trace_prefix)
    tokens, samples = trace["tokens"], trace["samples"]
    prompt_t = len(tokens)
    if not (1 <= prompt_t <= MAX_TOKENS and samples[0]["vocab"] == VOCAB
            and samples[0]["topk"] == 50 and abs(samples[0]["temperature"] - 0.9) < 1e-6
            and samples[0]["rng_before"] == 1234):
        raise RuntimeError("trace does not match pinned CSM dimensions or sampler parameters")
    resources = ab.snapshot()
    ab.require_headroom(resources)
    if int(resources["memory"]["MemAvailable"].split()[0]) < 8 * 1024**2:
        raise RuntimeError("CSM prefill capture requires 8 GiB available RAM")
    model_record = file_record(model)
    if model_record["sha256"] != MODEL_SHA256:
        raise RuntimeError("CSM GGUF bytes differ from the accepted physical Q4_K profile")
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"complete": False, "scope": "Existing CSM prefill dump only; no synthesis/sampling",
               "argv": sys.argv, "library": file_record(library), "model": model_record,
               "trace": {"prefix": str(trace_prefix), "files": trace["files"],
                         "prompt_tokens": prompt_t, "vocab": VOCAB,
                         "first_sample_rng_before": samples[0]["rng_before"]},
               "resources_before_load": resources, "params": {}, "files": {}}
    receipt["attention_environment"] = {
        "CRISPASR_CORE_ATTN_EAGER_F32": os.environ.get("CRISPASR_CORE_ATTN_EAGER_F32")}
    receipt["rope_environment"] = {
        "CRISPASR_CSM_ORIGINAL_ROPE": os.environ.get("CRISPASR_CSM_ORIGINAL_ROPE")}
    receipt_path = out / "receipt.json"

    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    save()
    receipt["ldd_libraries"] = loaded_libraries(library, out)
    receipt["files"]["ldd.log"] = file_record(out / "ldd.log")
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

    layer0 = np.empty((prompt_t, D_MODEL), dtype="<f4")
    layers = [np.empty((prompt_t, D_MODEL), dtype="<f4") for _ in range(N_LAYERS)]
    last_h = np.empty(D_MODEL, dtype="<f4")
    c0 = np.empty(VOCAB, dtype="<f4")
    ptrs = (float_ptr * N_LAYERS)(*(a.ctypes.data_as(float_ptr) for a in layers))
    ctx = lib.csm_tts_init_from_file(os.fsencode(model), params)
    if not ctx:
        raise RuntimeError("csm_tts_init_from_file failed")
    receipt["mapped_runtime_libraries"] = mapped_runtime_libraries(library)
    save()
    try:
        returned_t = lib.csm_tts_run_backbone_dump(ctx, args.text.encode(), prompt_t,
                                                    layer0.ctypes.data_as(float_ptr), ptrs,
                                                    N_LAYERS, last_h.ctypes.data_as(float_ptr),
                                                    c0.ctypes.data_as(float_ptr))
    finally:
        lib.csm_tts_free(ctx)
    receipt["returned_tokens"] = returned_t
    if returned_t != prompt_t:
        save()
        raise RuntimeError("prefill token count differs from retained sampled-path trace")
    arrays = {"backbone_layer0_normed": layer0,
              **{f"backbone_layer{i}_output": a for i, a in enumerate(layers)},
              "backbone_prefill_last_h": last_h, "backbone_prefill_c0_logits": c0}
    for name, array in arrays.items():
        path = out / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        receipt["files"][path.name] = file_record(path)
    observed = samples[0]["logits"].astype("<f4", copy=False).tobytes()
    actual = c0.tobytes()
    receipt["first_sample_c0_logits"] = {
        "dump_sha256": hashlib.sha256(actual).hexdigest(),
        "sampled_trace_sha256": hashlib.sha256(observed).hexdigest(),
        "byte_exact": actual == observed,
    }
    receipt["complete"] = actual == observed
    save()
    if not receipt["complete"]:
        raise RuntimeError("prefill dump c0 logits differ from actual first sampled logits; arrays retained")
    print(json.dumps({"complete": True, "out": str(out), "receipt": str(receipt_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
