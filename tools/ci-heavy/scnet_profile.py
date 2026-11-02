#!/usr/bin/env python3
"""Profile one real SCNet F16 CPU call; never a fidelity or speed qualification.

Only the small author checkpoint is converted. The original-model benchmark,
quantized models and qualification matrix are intentionally not executed here.
The single cold call uses BENCH=1, including synchronizing graph-op callbacks;
its instrumented timing must not be substituted for accepted throughput data.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time
import urllib.request

from scnet_qualify import (
    MUSIC_URL, RECORDED_CHECKPOINT_SHA256, RECORDED_MUSIC_SHA256, ROOT,
    prepare_fixture, resolve_library, utc_now,
)
from reference_backends.scnet import CHECKPOINT_URLS, SOURCE_HASHES, SOURCE_REVISION, sha256, verify_source_files
from validate_scnet import NativeModel, atomic_json_write, pcm_sha256, runtime_provenance

PROFILE = {"mode": "pipeline", "segment_seconds": 20, "shifts": 1, "shift_seed": 0, "overlap": .5}
SOURCES = (
    "src/scnet.cpp", "src/scnet.h", "src/scnet_fft.h", "src/scnet_lstm.h", "src/scnet_shift.h",
    "src/CMakeLists.txt", "CMakeLists.txt", "tools/scnet_exports.cpp", "tools/validate_scnet.py",
    "models/convert-scnet-to-gguf.py", "tools/reference_backends/scnet.py", "tools/requirements-scnet.txt",
    "tools/ci-heavy/scnet_qualify.py", "tools/ci-heavy/scnet_profile.py", ".github/workflows/scnet-profile.yml",
)


def portable(value, args):
    if isinstance(value, dict):
        return {key: portable(item, args) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item, args) for item in value]
    if isinstance(value, str):
        for path, label in ((args.output_dir, "{output}"), (args.assets_dir, "{assets}"),
                            (args.build_dir, "{build}"), (ROOT, "{repo}")):
            value = value.replace(str(path), label)
    return value


def parse_timers(text):
    rows = {"lstm": [], "graph_ops": [], "graph_summaries": [], "coarse": []}
    specifications = {
        "scnet_bench_lstm: ": ("lstm", ("name", "mode", "scope"),
            ("hidden", "batch", "steps", "gemm_calls", "gate_steps"), ("gemm_ms", "gates_ms", "reset_ms")),
        "scnet_bench_op: ": ("graph_ops", ("stage", "op", "backend"), ("frames", "calls"), ("ms",)),
        "scnet_bench_ops: ": ("graph_summaries", ("scope",), ("frames", "nodes", "groups", "unmapped_nodes"),
                              ("measured_ms", "unmapped_ms")),
    }
    for line in text.splitlines():
        for prefix, (group, strings, integers, numbers) in specifications.items():
            if not line.startswith(prefix):
                continue
            pairs = re.findall(r"([a-z_]+)=([^\s]+)", line[len(prefix):])
            fields = dict(pairs)
            if len(pairs) != len(fields) or set(fields) != set(strings + integers + numbers):
                raise ValueError(f"incomplete/unexpected native timer fields: {prefix.strip()}")
            for key in integers:
                fields[key] = int(fields[key])
                if fields[key] < 0:
                    raise ValueError(f"negative timer counter: {key}")
            for key in numbers:
                fields[key] = float(fields[key])
                if not math.isfinite(fields[key]) or fields[key] < 0:
                    raise ValueError(f"nonfinite/negative timer duration: {key}")
            rows[group].append(fields)
        match = re.fullmatch(r"scnet_bench: (.+) ([0-9.eE+-]+) ms", line)
        if match:
            duration = float(match[2])
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("invalid coarse timer duration")
            rows["coarse"].append({"stage": match[1], "ms": duration})
    if not all(rows.values()):
        raise ValueError("missing LSTM, graph-op, graph-summary or coarse BENCH telemetry")
    if any(row["mode"] not in ("CBLAS", "portable") or row["scope"] not in ("whole_node", "worker_zero")
           for row in rows["lstm"]):
        raise ValueError("unrecognized LSTM timer scope")
    if any(row["scope"] != "per_node_synchronized" for row in rows["graph_summaries"]):
        raise ValueError("unrecognized graph callback timer scope")
    if any(row["hidden"] <= 0 or row["batch"] <= 0 or row["steps"] <= 0 or row["gate_steps"] <= 0
           for row in rows["lstm"]) or any(row["frames"] <= 0 or row["calls"] <= 0 for row in rows["graph_ops"]):
        raise ValueError("empty recurrent/graph-op telemetry")
    rows["all_graph_nodes_mapped"] = all(row["unmapped_nodes"] == 0 for row in rows["graph_summaries"])
    rows["interpretation"] = (
        "One instrumented cold full-pipeline call; no warm throughput or original comparison. "
        "Graph callbacks synchronize individual nodes and change scheduling granularity. "
        "LSTM parts and graph nodes are nested in coarse network/separate timers: do not add them together. "
        "Portable recurrence and FFT worker-zero spans are not whole-node CPU-work totals."
    )
    return rows


def worker(args):
    import numpy as np
    import soundfile as sf

    model = args.assets_dir / "scnet-profile-f16.gguf"
    audio_path = args.assets_dir / "music-profile-24s.wav"
    library = resolve_library(args.build_dir, "scnet-validation")
    audio, rate = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = np.ascontiguousarray(audio)
    if rate != 44100 or audio.shape != (24 * 44100, 2) or not np.isfinite(audio).all():
        raise ValueError("diagnostic fixture is not finite 24-second 44100-Hz stereo")
    receipt = {"schema": 1, "acceptance": False, "qualification": False, "complete": False,
        "scope": "One instrumented cold native call; no model-quality or comparative-performance qualification",
        "calls": 1, "warmups": 0, "threads": 2, "profile": PROFILE, "frames": len(audio),
        "model_sha256": sha256(model), "audio_sha256": sha256(audio_path), "library_sha256": sha256(library),
        "input_pcm_sha256": pcm_sha256(audio), "runtime": runtime_provenance(library)}
    save = lambda: atomic_json_write(args.output_dir / "native-profile.json", portable(receipt, args))
    save()
    native = NativeModel(library, model, 2, PROFILE)
    receipt.update(native_backend=native.backend, native_compute_precision=native.compute_precision,
                   native_blas_thread_queries=native.blas_thread_queries)
    save()
    try:
        start = time.perf_counter()
        output = native.run(audio)
        receipt["instrumented_seconds"] = time.perf_counter() - start
        save()
        receipt.update(output_shape=list(output.shape), output_pcm_sha256=pcm_sha256(output),
            geometry_and_finite=bool(output.shape == (4, 2, len(audio)) and np.isfinite(output).all()),
            input_unchanged=pcm_sha256(audio) == receipt["input_pcm_sha256"],
            process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if not receipt["geometry_and_finite"] or not receipt["input_unchanged"]:
            raise ValueError("native diagnostic output geometry/finiteness or input identity failed")
        receipt["complete"] = True
    finally:
        native.close()
        save()


def profile(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": 1, "acceptance": False, "qualification": False, "complete": False,
        "variant": "scnet", "precision": "f16", "source_commit": args.source_commit,
        "upstream_source_revision": SOURCE_REVISION, "upstream_source_hashes": SOURCE_HASHES,
        "profile": PROFILE, "calls": 1, "warmups": 0, "threads": 2, "commands": [], "errors": []}
    save = lambda: atomic_json_write(args.output_dir / "diagnostic.json", portable(receipt, args))
    save()

    def run(label, command, timeout=600):
        event = {"name": label, "argv": [str(part) for part in command], "complete": False}
        receipt["commands"].append(event)
        save()
        try:
            with (args.output_dir / (label + ".stdout.log")).open("w") as stdout, \
                    (args.output_dir / (label + ".stderr.log")).open("w") as stderr:
                result = subprocess.run(event["argv"], cwd=ROOT, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
            event.update(complete=True, returncode=result.returncode)
            if result.returncode:
                raise RuntimeError(f"{label} failed with exit {result.returncode}; retained logs identify the failure")
        finally:
            save()

    def asset(path, url, digest, google=False):
        if not path.exists():
            partial = path.with_name(path.name + ".partial")
            if partial.exists():
                raise FileExistsError(f"previous partial download retained: {partial.name}")
            if google:
                run("download-checkpoint", [sys.executable, "-m", "gdown", "--no-cookies", "--fuzzy", url, "-O", partial])
            else:
                with urllib.request.urlopen(url, timeout=120) as response, partial.open("xb") as output:
                    shutil.copyfileobj(response, output)
            partial.rename(path)
        if sha256(path) != digest:
            raise ValueError(f"original asset differs from recorded public bytes: {path.name}")
        provenance_path = path.with_name(path.name + ".provenance.json")
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text())
            if provenance.get("source_url") != url or provenance.get("sha256") != digest:
                raise ValueError(f"cached provenance disagrees with original asset: {path.name}")
        else:
            provenance = {"source_url": url, "bytes": path.stat().st_size, "sha256": digest,
                "discovered_at": utc_now(), "digest_status": "Verified against recorded first public download; not author-attested"}
            atomic_json_write(provenance_path, provenance)
        return dict(provenance)

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if head != args.source_commit:
            raise ValueError("requested source commit differs from the checkout")
        source = verify_source_files(args.assets_dir / "SCNet")
        manifest = {}
        for prefix, base, paths in (("CrispASR", ROOT, SOURCES), ("SCNet", source, SOURCE_HASHES)):
            for relative in paths:
                path = base / relative
                target = args.output_dir / "source-snapshot" / prefix / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                manifest[prefix + "/" + relative] = sha256(target)
        atomic_json_write(args.output_dir / "source-manifest.json", manifest)
        commands = json.loads((args.build_dir / "compile_commands.json").read_text())
        selected = [row for row in commands if row["file"].endswith("/src/scnet.cpp") or "/ggml/src/" in row["file"]]
        if not any(row["file"].endswith("/src/scnet.cpp") for row in selected):
            raise ValueError("compile database does not identify the SCNet translation unit")
        atomic_json_write(args.output_dir / "compile-commands.json", portable(selected, args))
        receipt["compile_database_sha256"] = sha256(args.build_dir / "compile_commands.json")
        receipt["cmake_cache_sha256"] = sha256(args.build_dir / "CMakeCache.txt")
        library = resolve_library(args.build_dir, "scnet-validation")
        receipt["library_sha256"] = sha256(library)
        receipt["ggml_library_sha256"] = {str(p.relative_to(args.build_dir)): sha256(p)
            for p in sorted(args.build_dir.rglob("libggml*.so*")) if p.is_file() and not p.is_symlink()}
        run("native-linkage", ["ldd", library])
        checkpoint = args.assets_dir / "scnet-author.th"
        receipt["checkpoint"] = asset(checkpoint, CHECKPOINT_URLS["scnet"], RECORDED_CHECKPOINT_SHA256["scnet"], True)
        receipt["checkpoint"]["digest_status"] = "First author-linked download identity; not an author-published checksum"
        music = args.assets_dir / "hdemucs_mix.wav"
        asset(music, MUSIC_URL, RECORDED_MUSIC_SHA256)
        receipt["fixture"] = prepare_fixture(music, args.assets_dir / "music-profile-24s.wav",
                                             args.assets_dir / "music-profile-unused-smoke.wav")
        model = args.assets_dir / "scnet-profile-f16.gguf"
        if model.exists():
            raise FileExistsError("derived diagnostic model already exists; preserve prior evidence")
        run("convert-f16", [sys.executable, ROOT / "models/convert-scnet-to-gguf.py", "--source-dir", source,
            "--checkpoint", checkpoint, "--checkpoint-sha256", RECORDED_CHECKPOINT_SHA256["scnet"],
            "--config", source / "conf/config.yaml", "--variant", "scnet", "--dtype", "f16", "--output", model])
        receipt["model"] = {"file": model.name, "sha256": sha256(model), "bytes": model.stat().st_size}
        for suffix in (".provenance.json", ".LICENSE.txt"):
            shutil.copyfile(model.with_name(model.name + suffix), args.output_dir / ("converted-model" + suffix))
        run("native-profile", [sys.executable, Path(__file__), "--worker", "--source-commit", args.source_commit,
            "--build-dir", args.build_dir, "--assets-dir", args.assets_dir, "--output-dir", args.output_dir])
        native = json.loads((args.output_dir / "native-profile.json").read_text())
        if native.get("complete") is not True or native.get("acceptance") is not False or \
                native.get("model_sha256") != receipt["model"]["sha256"]:
            raise ValueError("native-only diagnostic receipt is incomplete or inconsistent")
        timers = parse_timers((args.output_dir / "native-profile.stderr.log").read_text())
        atomic_json_write(args.output_dir / "timers.json", timers)
        if sha256(model) != receipt["model"]["sha256"] or sha256(library) != receipt["library_sha256"]:
            raise ValueError("model or native binary changed during profiling")
        receipt["native_profile"] = native
        receipt["timer_counts"] = {key: len(value) for key, value in timers.items() if isinstance(value, list)}
        receipt["complete"] = True
    except BaseException as error:
        receipt["errors"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    for name in ("build-dir", "assets-dir", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not re.fullmatch("[0-9a-f]{40}", args.source_commit):
        parser.error("--source-commit must be the exact checkout revision")
    for name in ("build_dir", "assets_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS", "CRISPASR_SCNET_THREADS"):
        os.environ[key] = "2"
    for key in ("CRISPASR_SCNET_CAPTURE_GGUF", "CRISPASR_SCNET_SCALAR_LSTM", "CRISPASR_SCNET_BLAS", "CRISPASR_CPU_GEMM"):
        os.environ.pop(key, None)
    os.environ.update(CRISPASR_SCNET_GPU="0", CRISPASR_SCNET_BENCH="1", ACCELERATE_USE_CPU="true")
    if args.worker:
        worker(args)
    else:
        profile(args)
    print(json.dumps({"complete": True, "acceptance": False, "qualification": False, "calls": 1}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
