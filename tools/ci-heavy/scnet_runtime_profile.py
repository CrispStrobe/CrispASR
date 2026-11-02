#!/usr/bin/env python3
"""Collect native SCNet stage timers; diagnostic only, never a speed qualification."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("library", "model", "audio", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--calls", type=int, default=3)
    args = parser.parse_args()
    if args.threads < 1 or args.calls < 2:
        parser.error("positive threads and at least two recorded calls are required")
    if args.output.exists():
        parser.error("refusing to overwrite an existing diagnostic receipt")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS", "CRISPASR_SCNET_THREADS"):
        os.environ[key] = str(args.threads)
    for key in ("CRISPASR_SCNET_CAPTURE_GGUF", "CRISPASR_SCNET_SCALAR_LSTM", "CRISPASR_SCNET_SCALAR_GATES", "CRISPASR_SCNET_BLAS", "CRISPASR_CPU_GEMM"):
        os.environ.pop(key, None)
    os.environ.update(CRISPASR_SCNET_GPU="0", CRISPASR_SCNET_BENCH="1")
    import numpy as np
    import soundfile as sf
    from validate_scnet import NativeModel, atomic_json_write, pcm_sha256, sha256
    from scnet_runtime_identity import capture_loaded_dsos, compare_snapshots

    audio, rate = sf.read(args.audio, dtype="float32", always_2d=True)
    audio = np.ascontiguousarray(audio)
    if rate != 44100 or audio.shape[1] != 2 or len(audio) < 2 or not np.isfinite(audio).all():
        parser.error("finite 44100-Hz stereo input with at least two frames is required")
    profile = {"mode": "full", "segment_seconds": 20, "shifts": 1, "shift_seed": 0, "overlap": .5}
    receipt = {"scope": "Native stage-timer diagnostic only; no original-model speed comparison or qualification",
        "acceptance": False, "complete": False, "profile": profile, "threads": args.threads,
        "runtime_platform": sys.platform,
        "model_sha256": sha256(args.model), "library_sha256": sha256(args.library),
        "audio_sha256": sha256(args.audio), "input_pcm_sha256": pcm_sha256(audio),
        "frames": len(audio), "calls": []}
    save = lambda: atomic_json_write(args.output, receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save()
    native = NativeModel(args.library.resolve(), args.model.resolve(), args.threads, profile)
    receipt.update(native_backend=native.backend, native_compute_precision=native.compute_precision,
                   native_blas_thread_queries=native.blas_thread_queries)
    save()
    try:
        observation = receipt["loaded_dso_observation"] = {"complete": False,
            "scope": "Instrumented native calls; first-use additions/removals are recorded, not a timed-pair proof",
            "before": capture_loaded_dsos(args.library, require_torch=False)}
        save()
        for index in range(args.calls):
            print(f"SCNet runtime diagnostic call {index}: begin", file=sys.stderr, flush=True)
            start = time.perf_counter()
            output = native.run(audio)
            seconds = time.perf_counter() - start
            maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            row = {"index": index, "seconds": seconds, "geometry_and_finite": False,
                   "process_max_rss_kib": maximum_rss / 1024 if sys.platform == "darwin" else maximum_rss}
            receipt["calls"].append(row)
            save()
            row.update(shape=list(output.shape), pcm_sha256=pcm_sha256(output),
                       geometry_and_finite=bool(output.shape == (4, 2, len(audio)) and np.isfinite(output).all()))
            save()
            print(f"SCNet runtime diagnostic call {index}: {seconds:.6f}s", file=sys.stderr, flush=True)
            if not row["geometry_and_finite"]:
                raise ValueError("native diagnostic output failed geometry/finite checks")
        observation["after"] = capture_loaded_dsos(args.library, require_torch=False)
        observation["transition"] = compare_snapshots(observation["before"], observation["after"])
        if not observation["transition"]["common_file_bytes_unchanged"] or not observation["transition"]["configuration_unchanged"]:
            raise ValueError("mapped runtime bytes/configuration changed during instrumented calls")
        observation["complete"] = True
        receipt["repeated_output_hashes_identical"] = len({row["pcm_sha256"] for row in receipt["calls"]}) == 1
        if pcm_sha256(audio) != receipt["input_pcm_sha256"]:
            raise ValueError("native inference modified the input")
        receipt["complete"] = True
    finally:
        native.close()
        save()
    print(json.dumps({"complete": receipt["complete"], "acceptance": False, "output": str(args.output)}))


if __name__ == "__main__":
    main()
