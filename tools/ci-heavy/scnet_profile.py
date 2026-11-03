#!/usr/bin/env python3
"""Profile SCNet F16 and run a bounded same-host waveform/performance smoke.

Only the small author checkpoint is converted. One cold BENCH=1 call includes
synchronizing graph-op callbacks. A separate BENCH=0 process warms both engines
and runs three alternating original/native pairs with the validator's unchanged
F16 waveform gates. Optional --compare-wait-policy repeats those paired calls in
a fresh process with passive OpenMP waiting for both engines. The default pairs
remain the primary result. It does not capture the 35 intermediate stages or test
the quantized models, and every receipt keeps overall acceptance/qualification false.
"""
from __future__ import annotations

import argparse
import contextlib
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
from validate_scnet import (
    GGUFFile, NORM_RATIO_BOUNDS, RELATIVE_L2_MAX, STEMS, THREAD_VARIABLES, THRESHOLDS, NativeModel, all_pass,
    atomic_json_write, check_inventory, pcm_sha256, runtime_provenance, summarize_pairs, waveform_metrics,
)
from scnet_runtime_identity import LoadedDSOTracker, capture_loaded_dsos, compare_snapshots, verify_timed_identity

PROFILE = {"mode": "pipeline", "segment_seconds": 20, "shifts": 1, "shift_seed": 0, "overlap": .5}
SOURCES = (
    "src/scnet.cpp", "src/scnet.h", "src/scnet_fft.h", "src/scnet_lstm.h", "src/scnet_shift.h",
    "src/scnet_activation.h", "src/scnet_tensor.h", "src/scnet_im2col.h", "src/scnet_real_fft.h",
    "src/scnet_recurrent_gemm.h",
    "tests/test-scnet-math.py", "tests/test-scnet-tensor.cpp", "tests/test-scnet-im2col.py",
    "tests/test-scnet-recurrent-gemm.py",
    "tests/test-scnet-fft.py", "tests/test-scnet-real-fft.py", "tests/test_scnet_profile.py",
    "tests/test_scnet_runtime_identity.py", "tools/scnet_runtime_identity.py",
    "tests/test-scnet-blas-threads.cpp", "tests/test-scnet-allocator.cpp",
    "ggml/src/ggml-blas/ggml-blas.cpp", "ggml/src/ggml-alloc.c",
    "src/CMakeLists.txt", "CMakeLists.txt", "tools/scnet_exports.cpp", "tools/validate_scnet.py",
    "models/convert-scnet-to-gguf.py", "tools/reference_backends/scnet.py", "tools/requirements-scnet.txt",
    "tools/ci-heavy/scnet_qualify.py", "tools/ci-heavy/scnet_profile.py", "tools/ci-heavy/scnet_build_openblas.py",
    ".github/workflows/scnet-profile.yml",
)
CONTROLS = ("CRISPASR_SCNET_BENCH", "CRISPASR_SCNET_CAPTURE_GGUF", "CRISPASR_SCNET_SCALAR_LSTM",
            "CRISPASR_SCNET_SCALAR_GATES", "CRISPASR_SCNET_BLAS", "CRISPASR_CPU_GEMM", "CRISPASR_SCNET_GPU")
WAIT_POLICY_VARIABLES = ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "KMP_BLOCKTIME", "OPENBLAS_THREAD_TIMEOUT")
PASSIVE_WAIT_POLICY = {"OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": "0"}


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
    if any(row["mode"] not in ("CBLAS", "portable", "packed_avx512_fma", "packed_avx2_fma") or
           row["scope"] not in ("whole_node", "worker_zero") or
           (row["mode"].startswith("packed_") and row["scope"] != "worker_zero")
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
        "Portable/packed recurrence and FFT worker-zero spans are not whole-node CPU-work totals."
    )
    return rows


def load_fixture(args):
    import numpy as np
    import soundfile as sf

    audio_path = args.assets_dir / "music-profile-24s.wav"
    audio, rate = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = np.ascontiguousarray(audio)
    if rate != 44100 or audio.shape != (24 * 44100, 2) or not np.isfinite(audio).all():
        raise ValueError("diagnostic fixture is not finite 24-second 44100-Hz stereo")
    return audio, audio_path


def worker(args):
    import numpy as np

    model = args.assets_dir / "scnet-profile-f16.gguf"
    library = resolve_library(args.build_dir, "scnet-validation")
    audio, audio_path = load_fixture(args)
    receipt = {"schema": 1, "acceptance": False, "qualification": False, "complete": False,
        "scope": "One instrumented cold native call; no model-quality or comparative-performance qualification",
        "calls": 1, "warmups": 0, "threads": 2, "profile": PROFILE, "frames": len(audio),
        "model_sha256": sha256(model), "audio_sha256": sha256(audio_path), "library_sha256": sha256(library),
        "input_pcm_sha256": pcm_sha256(audio), "runtime": runtime_provenance(library),
        "execution_controls": {key: os.environ.get(key) for key in CONTROLS}}
    save = lambda: atomic_json_write(args.output_dir / "native-profile.json", portable(receipt, args))
    save()
    native = NativeModel(library, model, 2, PROFILE)
    receipt.update(native_backend=native.backend, native_compute_precision=native.compute_precision,
                   native_blas_thread_queries=native.blas_thread_queries)
    save()
    try:
        observation = receipt["loaded_dso_observation"] = {"complete": False,
            "scope": "One cold instrumented call; first-use additions/removals are recorded, not a stable timed-pair proof",
            "before": capture_loaded_dsos(library, require_torch=False)}
        receipt["runtime"]["loaded_dso_initialization"] = observation["before"]
        save()
        start = time.perf_counter()
        output = native.run(audio)
        receipt["instrumented_seconds"] = time.perf_counter() - start
        save()
        observation["after"] = capture_loaded_dsos(library, require_torch=False)
        observation["transition"] = compare_snapshots(observation["before"], observation["after"])
        if not observation["transition"]["common_file_bytes_unchanged"] or not observation["transition"]["configuration_unchanged"]:
            raise ValueError("mapped runtime bytes/configuration changed during the instrumented call")
        observation["complete"] = True
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


def paired_calls(original, native, receipt, save, input_unchanged, identity):
    """Schedule this smoke; all numeric gates and pair summaries come from the validator."""
    import numpy as np

    expected = None

    def call(name, function, collection, iteration):
        nonlocal expected
        measurement = {"iteration": iteration, "implementation": name, "complete": False,
                       "quality_acceptance": False}
        receipt[collection].append(measurement)
        save()
        started = time.perf_counter()
        try:
            output = function()
        except BaseException as error:
            measurement.update(elapsed_seconds_on_error=time.perf_counter() - started,
                               error=f"{type(error).__name__}: {error}")
            save()
            raise
        measurement.update(seconds=time.perf_counter() - started, complete=True)
        # Persist completed durations before hashing or evaluating any outputs.
        save()
        if expected is None:
            if collection != "warmups" or name != "original":
                raise ValueError("the waveform oracle must come from the genuine original warmup")
            expected = np.ascontiguousarray(output).copy()
            receipt["reference_pcm_sha256"] = pcm_sha256(expected)
        scores = waveform_metrics(output, expected, "f16")
        measurement.update(stems=scores, quality_acceptance=all_pass(scores), pcm_sha256=pcm_sha256(output),
                           input_unchanged=bool(input_unchanged()))
        save()
        if not measurement["quality_acceptance"] or not measurement["input_unchanged"]:
            raise ValueError(f"{name} {collection} output failed waveform/input-identity checks")
        print(f"paired smoke {collection} {iteration + 1}: {name} {measurement['seconds']:.6f}s quality=True",
              file=sys.stderr, flush=True)

    for name, function in (("original", original), ("native", native)):
        call(name, function, "warmups", 0)
    identity.before_timing()
    save()
    for iteration in range(3):
        order = ("original", "native") if iteration % 2 == 0 else ("native", "original")
        for name in order:
            call(name, original if name == "original" else native, "calls", iteration)
    identity.after_timing()
    save()
    receipt.update(summarize_pairs(receipt["calls"], 3), waveform_acceptance=True, complete=True)
    save()


def paired_worker(args):
    import numpy as np
    import torch
    from reference_backends.scnet import load_original, separate_original

    model = args.assets_dir / "scnet-profile-f16.gguf"
    checkpoint = args.assets_dir / "scnet-author.th"
    library = resolve_library(args.build_dir, "scnet-validation")
    audio, audio_path = load_fixture(args)
    planar = np.ascontiguousarray(audio.T)
    receipt = {"schema": 1, "acceptance": False, "qualification": False, "complete": False,
        "scope": "F16 final-waveform and same-host timing smoke; no intermediate-stage or quantized qualification",
        "variant": "scnet", "precision": "f16", "required_stages_checked": 0,
        "worker_process_id": os.getpid(),
        "source_commit": args.source_commit, "upstream_source_revision": SOURCE_REVISION,
        "upstream_source_hashes": SOURCE_HASHES, "profile": PROFILE, "threads": 2,
        "repetitions": 3, "warmups_per_implementation": 1, "warmups": [], "calls": [], "errors": [],
        "model_sha256": sha256(model), "checkpoint_sha256": sha256(checkpoint),
        "audio_sha256": sha256(audio_path), "library_sha256": sha256(library),
        "input_pcm_interleaved_sha256": pcm_sha256(audio), "input_pcm_planar_sha256": pcm_sha256(planar),
        "runtime": runtime_provenance(library), "execution_controls": {key: os.environ.get(key) for key in CONTROLS},
        "waveform_policy": {"strict_cosine_gt": THRESHOLDS["f16"], "norm_ratio_bounds": NORM_RATIO_BOUNDS["f16"],
                            "relative_l2_max": RELATIVE_L2_MAX["f16"]},
        "reference_origin": "Genuine unmodified original F32 CPU warmup output, retained in memory; no stage hooks",
        "timing_boundary": "public separation call, normalization/chunking and owned CPU result conversion; includes original helper printing to a local log; excludes model/audio load and quality metrics"}
    save = lambda: atomic_json_write(args.output_dir / "paired-smoke.json", portable(receipt, args))
    save()
    native = None
    try:
        if receipt["execution_controls"]["CRISPASR_SCNET_BENCH"] != "0" or any(
                receipt["execution_controls"][key] is not None for key in CONTROLS
                if key not in ("CRISPASR_SCNET_BENCH", "CRISPASR_SCNET_GPU")):
            raise ValueError("paired smoke requires BENCH=0 and no capture or forced execution overrides")
        if torch.__version__.split("+")[0] != "2.8.0":
            raise ValueError("original smoke must use upstream-required torch==2.8.0")
        if receipt["checkpoint_sha256"] != RECORDED_CHECKPOINT_SHA256["scnet"]:
            raise ValueError("paired smoke checkpoint differs from the recorded author release")
        source = verify_source_files(args.assets_dir / "SCNet")
        receipt["model_inventory"] = check_inventory(GGUFFile(model), "f16", None,
                                                     receipt["checkpoint_sha256"], SOURCE_HASHES)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        blueprint = load_original(source, checkpoint, "scnet", "cpu", expected_sha256=receipt["checkpoint_sha256"])
        if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32
               for parameter in blueprint.parameters()):
            raise ValueError("original smoke is not the published F32 CPU model")
        receipt.update(torch_version=torch.__version__, torch_intraop_threads=torch.get_num_threads(),
                       torch_interop_threads=torch.get_num_interop_threads(), original_dtype="float32",
                       original_provenance=blueprint._scnet_provenance)
        native = NativeModel(library, model, 2, PROFILE)
        identity = LoadedDSOTracker(library)
        receipt["loaded_dso_identity"] = identity.evidence
        receipt.update(native_backend=native.backend, native_compute_precision=native.compute_precision,
                       native_blas_thread_queries=native.blas_thread_queries,
                       runtime=runtime_provenance(library, identity.evidence["initialization"]))
        save()

        def original():
            return separate_original(blueprint, planar, segment_seconds=PROFILE["segment_seconds"],
                                     shifts=PROFILE["shifts"], shift_seed=PROFILE["shift_seed"], no_segment=False)

        def unchanged():
            return (pcm_sha256(audio) == receipt["input_pcm_interleaved_sha256"] and
                    pcm_sha256(planar) == receipt["input_pcm_planar_sha256"])

        with (args.output_dir / "paired-original-inference.log").open("w", encoding="utf-8") as original_log:
            with contextlib.redirect_stdout(original_log):
                paired_calls(original, lambda: native.run(audio), receipt, save, unchanged, identity)
        for key, path in (("model", model), ("checkpoint", checkpoint), ("audio", audio_path), ("library", library)):
            if sha256(path) != receipt[key + "_sha256"]:
                raise ValueError(f"{key} changed during the paired smoke")
        verify_source_files(source)
        receipt["process_max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except BaseException as error:
        receipt["complete"] = False
        receipt["errors"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        if native is not None:
            native.close()
        save()


def summarize_wait_policy_comparison(primary, passive):
    """Check two complete diagnostic receipts without changing qualification gates."""
    identities = ("source_commit", "upstream_source_revision", "upstream_source_hashes", "model_sha256",
                  "checkpoint_sha256", "audio_sha256", "library_sha256", "input_pcm_interleaved_sha256",
                  "input_pcm_planar_sha256", "original_provenance", "native_backend", "native_compute_precision",
                  "torch_version", "original_dtype", "profile", "waveform_policy")
    if any(key not in primary or primary[key] != passive.get(key) for key in identities):
        raise ValueError("wait-policy comparison changed model, source, fixture, runtime or numeric policy")
    process_ids = [row.get("worker_process_id") for row in (primary, passive)]
    if any(type(pid) is not int or pid <= 0 for pid in process_ids) or process_ids[0] == process_ids[1]:
        raise ValueError("wait-policy comparison requires separate fresh worker processes")
    summaries, hashes = {}, {}
    for label, row in (("default", primary), ("passive", passive)):
        loaded = verify_timed_identity(row.get("loaded_dso_identity"), native_sha256=row.get("library_sha256"))
        if loaded["process_id"] != row.get("worker_process_id"):
            raise ValueError(f"{label} loaded-runtime proof belongs to another worker")
        if row.get("complete") is not True or row.get("waveform_acceptance") is not True or \
                row.get("acceptance") is not False or row.get("qualification") is not False or \
                row.get("required_stages_checked") != 0 or row.get("profile") != PROFILE or \
                row.get("threads") != 2 or row.get("torch_intraop_threads") != 2 or \
                row.get("torch_interop_threads") != 1 or row.get("repetitions") != 3 or \
                row.get("warmups_per_implementation") != 1 or row.get("errors"):
            raise ValueError(f"{label} wait-policy receipt is incomplete or violates diagnostic scope")
        runtime = row.get("runtime", {})
        queries = row.get("native_blas_thread_queries", {})
        if runtime.get("thread_environment") != dict.fromkeys(THREAD_VARIABLES, "2") or \
                not queries or any(value != 2 for value in queries.values()):
            raise ValueError(f"{label} wait-policy receipt does not verify both two-thread budgets")
        controls = row.get("execution_controls", {})
        if set(controls) != set(CONTROLS) or controls.get("CRISPASR_SCNET_BENCH") != "0" or \
                controls.get("CRISPASR_SCNET_GPU") != "0" or \
                any(controls.get(key) is not None for key in CONTROLS
                    if key not in ("CRISPASR_SCNET_BENCH", "CRISPASR_SCNET_GPU")):
            raise ValueError(f"{label} wait-policy smoke enabled instrumentation or an execution override")
        warmups = row.get("warmups", [])
        if [(call.get("iteration"), call.get("implementation")) for call in warmups] != \
                [(0, "original"), (0, "native")]:
            raise ValueError(f"{label} wait-policy smoke did not warm both engines")
        summaries[label] = summarize_pairs(row.get("calls", []), 3)
        if any(row.get(key) != value for key, value in summaries[label].items()):
            raise ValueError(f"{label} wait-policy receipt disagrees with all retained timings")
        measurements = warmups + row["calls"]
        for call in measurements:
            scores = call.get("stems", {})
            if call.get("complete") is not True or call.get("quality_acceptance") is not True or \
                    call.get("input_unchanged") is not True or set(scores) != set(STEMS) or not all_pass(scores) or \
                    not isinstance(call.get("seconds"), (int, float)) or not math.isfinite(call["seconds"]) or \
                    call["seconds"] <= 0:
                raise ValueError(f"{label} wait-policy smoke has a failed or missing waveform/input check")
        hashes[label] = {}
        for implementation in ("original", "native"):
            digests = {call.get("pcm_sha256") for call in measurements
                       if call["implementation"] == implementation}
            if len(digests) != 1 or not all(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value)
                                          for value in digests):
                raise ValueError(f"{label} wait-policy smoke has unstable or missing PCM hashes")
            hashes[label][implementation] = next(iter(digests))
        if hashes[label]["original"] != row.get("reference_pcm_sha256"):
            raise ValueError(f"{label} wait-policy reference hash differs from its original warmup")
    if hashes["default"] != hashes["passive"]:
        raise ValueError("wait-policy comparison changed an engine's output PCM")
    first, second = (row["loaded_dso_identity"]["timed_before"] for row in (primary, passive))
    if any(first[key] != second[key] for key in ("files", "roles", "cmake_cache_sha256")):
        raise ValueError("wait-policy comparison changed its actual mapped runtime files")
    default_environment = primary["runtime"].get("wait_policy_environment", {})
    passive_environment = passive["runtime"].get("wait_policy_environment", {})
    if set(default_environment) != set(WAIT_POLICY_VARIABLES) or \
            passive_environment != {**default_environment, **PASSIVE_WAIT_POLICY}:
        raise ValueError("passive waiting was not applied equally while preserving other wait controls")
    return {"schema": 1, "complete": True, "acceptance": False, "qualification": False,
        "scope": "Optional same-host waiting-policy diagnostic; the default paired smoke remains primary. "
                 "Each policy has a fresh process, one warmup per engine and three alternating pairs. "
                 "Default runs before passive; cross-policy timing differences are descriptive, not randomized.",
        "default_receipt": "paired-smoke.json", "passive_receipt": "wait-policy-passive/paired-smoke.json",
        "default_wait_policy_environment": default_environment, "passive_wait_policy_environment": passive_environment,
        "default_results": summaries["default"], "passive_results": summaries["passive"],
        "output_pcm_sha256": hashes["default"], "output_hashes_identical_across_policies": True,
        "passive_minus_default_seconds": {name: summaries["passive"][name + "_median_seconds"] -
            summaries["default"][name + "_median_seconds"] for name in ("original", "native")},
        "passive_paired_smoke": passive}


def profile(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": 1, "acceptance": False, "qualification": False, "complete": False,
        "variant": "scnet", "precision": "f16", "source_commit": args.source_commit,
        "upstream_source_revision": SOURCE_REVISION, "upstream_source_hashes": SOURCE_HASHES,
        "profile": PROFILE, "calls": 1, "warmups": 0, "threads": 2, "commands": [], "errors": [],
        "scope": "One instrumented cold call plus separate uninstrumented F16 waveform/timing smoke; not a qualification",
        "paired_smoke_repetitions": 3, "paired_smoke_warmups_per_implementation": 1,
        "wait_policy_comparison_requested": args.compare_wait_policy}
    save = lambda: atomic_json_write(args.output_dir / "diagnostic.json", portable(receipt, args))
    save()

    def run(label, command, timeout=600, environment=None, log_dir=None):
        event = {"name": label, "argv": [str(part) for part in command], "complete": False}
        child_environment = None
        if environment is not None:
            child_environment = {**os.environ, **environment}
            event["environment_overrides"] = dict(environment)
        directory = log_dir or args.output_dir
        if log_dir is not None:
            event["log_directory"] = str(directory)
        receipt["commands"].append(event)
        save()
        try:
            with (directory / (label + ".stdout.log")).open("w") as stdout, \
                    (directory / (label + ".stderr.log")).open("w") as stderr:
                result = subprocess.run(event["argv"], cwd=ROOT, stdout=stdout, stderr=stderr, timeout=timeout,
                                        check=False, env=child_environment)
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
        run("paired-smoke", [sys.executable, Path(__file__), "--paired-worker", "--source-commit", args.source_commit,
            "--build-dir", args.build_dir, "--assets-dir", args.assets_dir, "--output-dir", args.output_dir], timeout=1200)
        paired = json.loads((args.output_dir / "paired-smoke.json").read_text())
        receipt["paired_smoke"] = paired
        if re.search(r"^scnet_bench(?:_|:)", (args.output_dir / "paired-smoke.stderr.log").read_text(), re.MULTILINE):
            raise ValueError("paired smoke emitted instrumentation despite requiring BENCH=0")
        if paired.get("complete") is not True or paired.get("acceptance") is not False or \
                paired.get("qualification") is not False or paired.get("waveform_acceptance") is not True or \
                paired.get("model_sha256") != receipt["model"]["sha256"] or \
                paired.get("library_sha256") != receipt["library_sha256"] or \
                paired.get("audio_sha256") != native.get("audio_sha256") or \
                paired.get("execution_controls", {}).get("CRISPASR_SCNET_BENCH") != "0":
            raise ValueError("paired waveform/timing smoke is incomplete or inconsistent")
        summarize_pairs(paired["calls"], 3)
        if args.compare_wait_policy:
            passive_dir = args.output_dir / "wait-policy-passive"
            passive_dir.mkdir(exist_ok=False)
            receipt["wait_policy_comparison"] = {"complete": False, "acceptance": False, "qualification": False,
                "default_receipt": "paired-smoke.json", "passive_receipt": "wait-policy-passive/paired-smoke.json",
                "environment_overrides": dict(PASSIVE_WAIT_POLICY)}
            save()
            run("paired-smoke-passive", [sys.executable, Path(__file__), "--paired-worker",
                "--source-commit", args.source_commit, "--build-dir", args.build_dir,
                "--assets-dir", args.assets_dir, "--output-dir", passive_dir], timeout=1200,
                environment=PASSIVE_WAIT_POLICY, log_dir=passive_dir)
            passive = json.loads((passive_dir / "paired-smoke.json").read_text())
            receipt["wait_policy_comparison"]["passive_paired_smoke"] = passive
            save()
            if re.search(r"^scnet_bench(?:_|:)", (passive_dir / "paired-smoke-passive.stderr.log").read_text(), re.MULTILINE):
                raise ValueError("passive-wait smoke emitted instrumentation despite requiring BENCH=0")
            receipt["wait_policy_comparison"] = summarize_wait_policy_comparison(paired, passive)
            atomic_json_write(args.output_dir / "wait-policy-comparison.json", portable(receipt["wait_policy_comparison"], args))
        receipt["complete"] = True
    except BaseException as error:
        receipt["errors"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--compare-wait-policy", action="store_true",
                        help="Also run a fresh same-environment passive-wait diagnostic for both engines")
    for name in ("build-dir", "assets-dir", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    workers = parser.add_mutually_exclusive_group()
    workers.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    workers.add_argument("--paired-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.compare_wait_policy and (args.worker or args.paired_worker):
        parser.error("--compare-wait-policy belongs to the supervising diagnostic, not a worker")
    if not re.fullmatch("[0-9a-f]{40}", args.source_commit):
        parser.error("--source-commit must be the exact checkout revision")
    for name in ("build_dir", "assets_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS", "CRISPASR_SCNET_THREADS"):
        os.environ[key] = "2"
    for key in ("CRISPASR_SCNET_CAPTURE_GGUF", "CRISPASR_SCNET_SCALAR_LSTM", "CRISPASR_SCNET_SCALAR_GATES",
                "CRISPASR_SCNET_BLAS", "CRISPASR_CPU_GEMM"):
        os.environ.pop(key, None)
    os.environ.update(CRISPASR_SCNET_GPU="0", CRISPASR_SCNET_BENCH="0" if args.paired_worker else "1",
                      ACCELERATE_USE_CPU="true")
    if args.paired_worker:
        paired_worker(args)
    elif args.worker:
        worker(args)
    else:
        profile(args)
    print(json.dumps({"complete": True, "acceptance": False, "qualification": False,
                      "mode": "paired-smoke" if args.paired_worker else "native-profile" if args.worker else "diagnostic"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
