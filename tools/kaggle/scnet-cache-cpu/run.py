#!/usr/bin/env python3
"""Offline CPU graph-cache observation with actual pretrained SCNet weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time

RATE = 44100
FRAMES = 60 * RATE
MODEL_SHA = "8441b23c2b6ba47e8357fd863d717af88a5568f1937a3ff328c921c7143bd059"


def parse_allocator(log):
    records = []
    for line in log.splitlines():
        if line.startswith("scnet_bench_allocator:"):
            records.append(dict(re.findall(r"(\w+)=(\S+)", line)))
    return records


def fixture(path):
    import numpy as np
    import soundfile as sf

    # A nonconstant asymmetric stereo signal exercises real inference without
    # downloading or redistributing music. Geometry alone determines cache keys.
    t = np.arange(FRAMES, dtype=np.float64) / RATE
    left = .12 * np.sin(2 * np.pi * (170 * t + .8 * t * t)) + .05 * np.sin(2 * np.pi * 659 * t)
    right = .09 * np.sin(2 * np.pi * (233 * t + .3 * t * t)) + .04 * np.cos(2 * np.pi * 997 * t)
    sf.write(path, np.column_stack((left, right)), RATE, subtype="PCM_16")


def prepare_runtime(inputs, runtime):
    runtime.mkdir(parents=True, exist_ok=True)
    archive_path = inputs / "crispasr-linux-x86_64.tar.gz"
    if archive_path.is_file():
        with tarfile.open(archive_path) as archive:
            archive.extractall(runtime, filter="data")
    else:
        # Kaggle expands supported dataset archives during ingestion.
        mounted = [p for p in inputs.rglob("crispasr") if p.is_file()]
        if len(mounted) != 1:
            raise FileNotFoundError(f"Expected one mounted runtime; found {mounted}")
        extracted = mounted[0].parent
        shutil.copytree(extracted, runtime / extracted.name, dirs_exist_ok=True)
    candidates = [p for p in runtime.rglob("crispasr") if p.is_file()]
    assert len(candidates) == 1, f"CLI candidates: {candidates}"
    cli = candidates[0]
    cli.chmod(cli.stat().st_mode | 0o111)
    return cli


def run(inputs, output):
    import numpy as np
    import soundfile as sf

    output.mkdir(parents=True, exist_ok=True)
    runtime = output / "runtime"
    model = inputs / "scnet-f16.gguf"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == MODEL_SHA
    cli = prepare_runtime(inputs, runtime)
    audio = output / "cache-60s.wav"
    fixture(audio)
    stems = output / "stems"
    stems.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CRISPASR_SCNET_BENCH="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    libraries = sorted({str(p.parent) for p in runtime.rglob("*.so*")})
    env["LD_LIBRARY_PATH"] = ":".join(libraries + [env.get("LD_LIBRARY_PATH", "")])
    command = [str(cli), "--separate", "--backend", "scnet", "--no-gpu", "-t", "2",
               "-m", str(model), "-f", str(audio), "--seed", "0", "--sep-output-dir", str(stems)]
    print("Starting one real-model CPU separation: 60s, segment=20s, shifts=1, seed=0.", flush=True)
    start = time.monotonic()
    with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
        result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
    records = parse_allocator((output / "stderr.log").read_text())
    checks = []
    for path in sorted(stems.glob("*.wav")):
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        checks.append({"file": path.name, "shape": list(samples.shape), "sample_rate": rate,
                       "finite": bool(np.isfinite(samples).all()), "peak": float(np.max(np.abs(samples)))})
    measured_frames = [int(r["frames"]) for r in records]
    hits = sum(int(r["metadata_hit"]) for r in records)
    passed = (result.returncode == 0 and bool(records)
              and len(checks) == 4 and all(c["finite"] and c["shape"] == [FRAMES, 2]
                                         and c["sample_rate"] == RATE for c in checks))
    summary = {"passed": passed, "returncode": result.returncode, "elapsed_seconds": time.monotonic() - start,
               "scope": "CPU graph metadata hit observation; profiling enabled; not a throughput or separation-quality benchmark",
               "release": "v0.8.53", "model_sha256": MODEL_SHA,
               "input_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(), "seed": 0,
               "seconds": 60, "shifts": 1, "segment_seconds": 20,
               "allocator_records": records, "lookups": len(records), "hits": hits,
               "observed_frames": measured_frames,
               "source_predicted_frames": [872, 872, 884, 876, 868, 438],
               "frames_match_source_prediction": measured_frames == [872, 872, 884, 876, 868, 438],
               "hit_rate": hits / len(records) if records else None, "stems": checks}
    (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Measurement did not complete expected work; inspect full stderr.log and stdout.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    if args.inputs is None:
        matches = list(Path("/kaggle/input").rglob("scnet-f16.gguf"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one attached real model, found {len(matches)}")
        args.inputs = matches[0].parent
    run(args.inputs.resolve(), args.output.resolve())
