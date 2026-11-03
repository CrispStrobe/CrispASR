#!/usr/bin/env python3
"""Run one fail-closed SCNet CPU qualification on a prepared hosted worker.

The workflow builds production binaries and installs the pinned reference
dependencies before this script runs. Assets come only from the author-linked
SCNet downloads and the attributed torchaudio tutorial fixture. Every command
has separate retained stdout/stderr, an exit status, and an incremental JSON
receipt. An existing output directory is never overwritten or resumed in a
way that could replace failed experiments or completed timing measurements.

  python tools/ci-heavy/scnet_qualify.py --variant scnet \
      --build-dir build-scnet --assets-dir scnet-assets --output-dir scnet-out

The downloaded checkpoints do not have an upstream published SHA-256. Their
first-fetch digests are explicitly recorded as discovered, not independently
author-attested, and are verified by every subsequent conversion/reference
load. These qualification assets are retained on the worker; the default
artifact upload should include JSON, logs, source/config snapshots and the
patch, but omit music, model weights and raw activation archives.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from reference_backends.scnet import (  # noqa: E402
    CHECKPOINT_URLS, SOURCE_HASHES, SOURCE_REVISION, sha256, verify_source_files,
)
from scnet_runtime_identity import verify_timed_identity  # noqa: E402

MUSIC_URL = "https://download.pytorch.org/torchaudio/tutorial-assets/hdemucs_mix.wav"
# First author-linked downloads, retained in run 41397985492. These identify
# the tested release bytes; they are discovered hashes, not author attestations.
RECORDED_CHECKPOINT_SHA256 = {
    "scnet": "f8aff3bc8dd816aad392800d9d69519add719a2fbd22550af6f8cb2138d25ff5",
    "scnet-large": "719e5abb8ed920305dad546ac3cd6fb0b1e9c3092d14ce21827bfc0423af3070",
}
RECORDED_LARGE_CONFIG_SHA256 = "629a4901184bf1d3a75b0b13904f35974785aa042cad3c010fd576248cdce3f0"
RECORDED_MUSIC_SHA256 = "f2420feedcdee9591dd649442c81149e31a0c06ee8c6e655d93d8f8e742cf706"
MUSIC_ATTRIBUTION = {
    "title": "A Classic Education - NightOwl", "license": "CC BY-NC-SA 4.0",
    "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "attribution_url": "https://docs.pytorch.org/audio/main/tutorials/hybrid_demucs_tutorial.html",
    "source_url": MUSIC_URL,
    "use": "Implementation-parity test only; not redistributed and not a held-out separation-quality evaluation.",
}
LARGE_CONFIG_URL = "https://drive.google.com/file/d/1qxK7SZx6-Gsp1s3wCrj98X7--UcI4O3K/view"
SOURCE_FILES = (
    "src/scnet.cpp", "src/scnet.h", "src/scnet_fft.h", "src/scnet_lstm.h", "src/scnet_shift.h",
    "src/scnet_activation.h", "src/scnet_tensor.h", "src/scnet_im2col.h", "src/scnet_real_fft.h",
    "src/scnet_recurrent_gemm.h",
    "tools/scnet_probe.cpp", "tools/scnet_exports.cpp", "tools/validate_scnet.py",
    "tools/scnet_runtime_identity.py", "tools/ci-heavy/scnet_build_openblas.py",
    "tools/ci-heavy/scnet_qualify.py", "tools/ci-heavy/scnet_precision_diagnostics.py",
    "tools/ci-heavy/scnet_runtime_profile.py", "tools/ci-heavy/scnet_package.py", "tools/dump_scnet_reference.py",
    "tools/reference_backends/scnet.py", "tools/requirements-scnet.txt",
    "models/convert-scnet-to-gguf.py", "models/licenses/SCNet-MIT.txt",
    "tests/test-scnet-converter.py", "tests/test-scnet-reference.py", "tests/test-scnet-native-diff.py",
    "tests/test-scnet-math.py", "tests/test-scnet-session-routing.py", "tests/test-scnet-quant-policy.py",
    "tests/test-scnet-recurrent-gemm.py",
    "tests/test-scnet-im2col.py", "tests/test-scnet-tensor.cpp", "tests/test-scnet-fft.py", "tests/test-scnet-real-fft.py",
    "tests/test-scnet-blas-threads.cpp", "tests/test-scnet-allocator.cpp",
    "ggml/src/ggml-blas/ggml-blas.cpp", "ggml/src/ggml-alloc.c",
    "tests/test_scnet_validation.py", "tests/test-scnet-surface.py", "examples/cli/crispasr_backend_scnet.cpp",
    "tests/test_scnet_linkage.py", "tests/test_scnet_runtime_identity.py",
    "examples/crispasr-quantize/main.cpp",
    "src/CMakeLists.txt", "CMakeLists.txt", ".github/workflows/scnet-qualification.yml",
)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class Qualification:
    def __init__(self, args):
        self.args = args
        self.output = args.output_dir
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "logs").mkdir()
        self.receipt_path = self.output / "qualification.json"
        self.receipt = {
            "schema": 1, "variant": args.variant, "started_at": utc_now(),
            "complete": False, "acceptance": False, "status": "initializing",
            "source_revision": SOURCE_REVISION, "commands": [], "steps": [], "errors": [],
            "threads": args.threads, "host": {"os": platform.system(), "machine": platform.machine()},
            "qualification_scope": "One 24-second contiguous music fixture and recorded CPU; GPU/held-out quality unqualified.",
            "artifact_policy": "Evidence uploads contain JSON/logs/source/config/patch only. A separate success-only bundle retains exactly the three qualified model GGUFs, MIT notice and provenance. Music, author checkpoints, raw captures and diagnostic models remain worker-local.",
        }
        self.environment = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "CRISPASR_SCNET_THREADS"):
            self.environment[name] = str(args.threads)
        self.environment.update(ACCELERATE_USE_CPU="true", CRISPASR_SCNET_GPU="0",
                                SCNET_SOURCE_DIR=str(args.assets_dir / "SCNet"))
        for name in ("CRISPASR_SCNET_CAPTURE_GGUF", "CRISPASR_SCNET_BENCH", "CRISPASR_SCNET_SCALAR_LSTM", "CRISPASR_SCNET_SCALAR_GATES",
                     "CRISPASR_SCNET_BLAS", "CRISPASR_CPU_GEMM"):
            self.environment.pop(name, None)
        self.save()

    def portable(self, value):
        text = str(value)
        for prefix, replacement in ((self.output, "{output}"), (self.args.assets_dir, "{assets}"),
                                    (self.args.build_dir, "{build}"), (ROOT, "{repo}")):
            text = text.replace(str(prefix), replacement)
        return text

    def save(self):
        write_json(self.receipt_path, self.receipt)

    def run(self, name, command, *, required=True, timeout=None, extra_env=None):
        if any(event["name"] == name for event in self.receipt["commands"]):
            raise ValueError(f"command name already used; refusing to replace evidence: {name}")
        command = [str(part) for part in command]
        event = {"name": name, "argv": [self.portable(part) for part in command], "started_at": utc_now(),
                 "stdout": f"logs/{name}.stdout.log", "stderr": f"logs/{name}.stderr.log", "complete": False}
        self.receipt["status"] = name
        self.receipt["commands"].append(event)
        self.save()
        print(f"SCNet qualification: {name}", flush=True)
        start = time.monotonic()
        code = None
        try:
            environment = dict(self.environment)
            environment.update(extra_env or {})
            with (self.output / event["stdout"]).open("w", encoding="utf-8") as stdout, \
                    (self.output / event["stderr"]).open("w", encoding="utf-8") as stderr:
                completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
                                           timeout=timeout or self.args.command_timeout, check=False)
            code = completed.returncode
            event.update(complete=True, returncode=code)
        except Exception as error:
            event.update(error=f"{type(error).__name__}: {error}", returncode=None)
        finally:
            event.update(seconds=time.monotonic() - start, finished_at=utc_now())
            if code != 0:
                self.receipt["errors"].append({"command": name, "returncode": code,
                                               "error": event.get("error", "nonzero process exit")})
            self.save()
        if required and code != 0:
            raise RuntimeError(f"{name} failed with exit {code}; see retained stdout/stderr and qualification.json")
        return code

    def step(self, name, callback):
        event = {"name": name, "started_at": utc_now(), "complete": False,
                 "stdout": f"logs/{name}.stdout.log", "stderr": f"logs/{name}.stderr.log"}
        self.receipt["steps"].append(event)
        self.receipt["status"] = name
        self.save()
        start = time.monotonic()
        try:
            with (self.output / event["stdout"]).open("w", encoding="utf-8") as stdout, \
                    (self.output / event["stderr"]).open("w", encoding="utf-8") as stderr, \
                    contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = callback()
            event.update(complete=True, returncode=0)
            return result
        except BaseException as error:
            event.update(returncode=1, error=f"{type(error).__name__}: {error}")
            self.receipt["errors"].append({"step": name, "error": event["error"]})
            raise
        finally:
            event.update(seconds=time.monotonic() - start, finished_at=utc_now())
            self.save()

    def asset(self, name, path, url, google=False, expected_sha256=None):
        provenance_path = path.with_name(path.name + ".provenance.json")
        if path.exists():
            if not provenance_path.is_file():
                raise ValueError(f"existing asset lacks discovery provenance: {self.portable(path)}")
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance.get("source_url") != url or provenance.get("sha256") != sha256(path):
                raise ValueError(f"cached asset changed or has a different origin: {self.portable(path)}")
        else:
            partial = path.with_name(path.name + ".download.partial")
            if partial.exists():
                raise ValueError(f"retained incomplete download exists; use a new assets directory: {self.portable(partial)}")
            if google:
                self.run("download-" + name, [sys.executable, "-m", "gdown", "--no-cookies", "--fuzzy",
                                              url, "-O", partial])
            else:
                def download():
                    request = urllib.request.Request(url, headers={"User-Agent": "CrispASR-SCNet-qualification"})
                    with urllib.request.urlopen(request, timeout=120) as response, partial.open("xb") as output:
                        shutil.copyfileobj(response, output, length=1024 * 1024)
                self.step("download-" + name, download)
            if not partial.is_file() or partial.stat().st_size == 0:
                raise ValueError(f"empty download: {name}")
            os.replace(partial, path)
            provenance = {"source_url": url, "sha256": sha256(path), "bytes": path.stat().st_size,
                          "discovered_at": utc_now(), "digest_status": "discovered on first public download; not author-attested"}
            write_json(provenance_path, provenance)
        if expected_sha256 is not None:
            provenance["recorded_release_sha256"] = expected_sha256
            provenance["matches_recorded_release"] = provenance["sha256"] == expected_sha256
        write_json(self.output / f"{name}-provenance.json", provenance)
        if expected_sha256 is not None and not provenance["matches_recorded_release"]:
            raise ValueError(f"download differs from the recorded release bytes: {name}")
        return provenance


def resolve_binary(build, name):
    for path in (build / "bin" / name, build / "src" / name, build / name):
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    raise FileNotFoundError(f"required built executable not found in bounded bin/src/root paths: {name}")


def resolve_library(build, name):
    for directory in (build / "src", build / "lib", build):
        for extension in (".so", ".dylib"):
            path = directory / ("lib" + name + extension)
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError(f"required shared library not found in bounded src/lib/root paths: {name}")


LINKAGE_ENTRYPOINTS = frozenset(("crispasr", "crispasr-quantize", "scnet-probe",
                               "production_library", "validation_library"))
LINKAGE_GGML = frozenset(("libggml.so.0", "libggml-base.so.0", "libggml-cpu.so.0", "libggml-blas.so.0"))
LINKAGE_ENVIRONMENT = ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT")
LINKAGE_METHOD = "linux-ldd-direct-backends-v1"


def resolve_qualification_binaries(build):
    binaries = {name: resolve_binary(build, name) for name in ("crispasr", "crispasr-quantize", "scnet-probe")}
    binaries.update(production_library=resolve_library(build, "crispasr"),
                    validation_library=resolve_library(build, "scnet-validation"))
    return binaries


def linkage_path(path, build):
    """Readable portable name; the full resolved path is separately fingerprinted."""
    path = Path(path)
    for root, alias in ((Path(build).resolve(), "{build}"), (ROOT, "{repo}")):
        if path.is_relative_to(root):
            return alias + "/" + path.relative_to(root).as_posix()
    return "{runtime}/" + path.name


def linkage_file(path, build):
    original = Path(path)
    try:
        path = original.resolve(strict=True)
        if not path.is_file():
            raise ValueError("not a regular file")
        return {"path": linkage_path(path, build), "sha256": sha256(path), "bytes": path.stat().st_size,
                "resolved_path_sha256": hashlib.sha256(os.fsencode(path)).hexdigest()}
    except (OSError, ValueError) as error:
        raise ValueError(f"runtime linkage file is missing/unreadable: {linkage_path(original, build)}") from error


def parse_ldd(text):
    """Parse file-backed dependencies, rejecting unresolved or ambiguous output."""
    result = {}
    for line in text.splitlines():
        if "not found" in line:
            raise ValueError("runtime linkage contains an unresolved dependency")
        match = re.fullmatch(r"\s*(\S+)\s+=>\s+(/\S+)\s+\(0x[0-9a-fA-F]+\)\s*", line)
        if match:
            soname, filename = match.groups()
        else:
            match = re.fullmatch(r"\s*(/\S+)\s+\(0x[0-9a-fA-F]+\)\s*", line)
            if not match:
                if re.fullmatch(r"\s*linux-(?:vdso|gate)\.so\.\d+\s+\(0x[0-9a-fA-F]+\)\s*", line):
                    continue  # Kernel-provided virtual DSO, no on-disk bytes.
                raise ValueError("runtime linkage resolver returned an unrecognized dependency line")
            filename = match.group(1)
            soname = Path(filename).name
        if soname in result:
            raise ValueError("runtime linkage resolver duplicated a dependency name")
        result[soname] = Path(filename)
    if not result:
        raise ValueError("runtime linkage resolver returned no file-backed dependencies")
    return result


def validate_linkage_snapshot(snapshot):
    """Validate receipt structure without trusting its success flags."""
    def require(condition, message):
        if not condition:
            raise ValueError("runtime linkage " + message)

    def valid_file(record):
        return (isinstance(record, dict) and isinstance(record.get("path"), str) and
                record["path"].startswith(("{build}/", "{repo}/", "{runtime}/")) and
                type(record.get("bytes")) is int and record["bytes"] > 0 and
                all(isinstance(record.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", record[key])
                    for key in ("sha256", "resolved_path_sha256")))

    require(isinstance(snapshot, dict) and snapshot.get("schema") == 1 and
            snapshot.get("method") == LINKAGE_METHOD and snapshot.get("backend_dl") is False,
            "snapshot is missing or incompatible")
    entrypoints, dependencies, resolutions = (snapshot.get(key, {}) for key in
                                              ("entrypoints", "dependencies", "resolutions"))
    require(isinstance(entrypoints, dict) and set(entrypoints) == LINKAGE_ENTRYPOINTS and
            all(valid_file(record) for record in entrypoints.values()), "entrypoint inventory is incomplete")
    require(isinstance(dependencies, dict) and bool(dependencies) and
            all(valid_file(record) and key == record["resolved_path_sha256"]
                for key, record in dependencies.items()), "dependency inventory is incomplete")
    require(isinstance(resolutions, dict) and set(resolutions) == LINKAGE_ENTRYPOINTS,
            "dependency resolution is incomplete")
    names, owners = {}, {}
    for entrypoint, records in resolutions.items():
        require(isinstance(records, dict) and bool(records) and all(isinstance(name, str) and name and
                isinstance(identity, str) and identity in dependencies for name, identity in records.items()),
                "dependency resolution is malformed")
        for name, identity in records.items():
            names.setdefault(identity, set()).add(name)
            owners.setdefault(identity, set()).add(entrypoint)
    require(set(names) == set(dependencies), "contains an unreferenced dependency")
    for identity, record in dependencies.items():
        require(record.get("sonames") == sorted(names[identity]) and
                record.get("required_by") == sorted(owners[identity]), "dependency ownership is inconsistent")
    all_names = {name for record in names.values() for name in record}
    require(LINKAGE_GGML <= all_names and any("openblas" in name.lower() for name in all_names),
            "must bind all four ggml libraries and the OpenBLAS provider")
    environment = snapshot.get("loader_environment_sha256", {})
    require(isinstance(environment, dict) and set(environment) == set(LINKAGE_ENVIRONMENT) and
            all(value is None or isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                for value in environment.values()), "loader environment is incomplete")
    logs = snapshot.get("resolver_logs", {})
    require(isinstance(logs, dict) and set(logs) == LINKAGE_ENTRYPOINTS,
            "resolver logs are incomplete")
    for name, record in logs.items():
        require(isinstance(record, dict) and record.get("returncode") == 0,
                "resolver did not complete successfully")
        for stream in ("stdout", "stderr"):
            output = record.get(stream, {})
            require(isinstance(output, dict) and isinstance(output.get("file"), str) and
                    re.fullmatch(r"linkage-[a-z0-9-]+-" + re.escape(name) + r"\." + stream + r"\.log", output["file"]) and
                    isinstance(output.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", output["sha256"]),
                    "resolver log identity is missing or malformed")
    return snapshot


def snapshot_runtime_linkage(binaries, build, environment, log_dir, epoch):
    """Bind standalone ldd resolution under the run environment, not a worker's link map.

    Dynamic backend loading is excluded explicitly. Hashing resolved paths as
    well as file bytes detects same-name/same-content resolution changes while
    keeping private machine paths out of public JSON and logs.
    """
    if platform.system() != "Linux":
        raise ValueError("runtime linkage qualification currently requires Linux")
    cache = (Path(build) / "CMakeCache.txt").read_text(encoding="utf-8")
    if not re.search(r"^GGML_BACKEND_DL:BOOL=OFF$", cache, re.M):
        raise ValueError("runtime linkage qualification requires GGML_BACKEND_DL=OFF")
    if set(binaries) != LINKAGE_ENTRYPOINTS or not re.fullmatch(r"[a-z0-9-]+", epoch):
        raise ValueError("runtime linkage needs all five entrypoints and a bounded log label")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {"schema": 1, "method": LINKAGE_METHOD, "backend_dl": False,
                "entrypoints": {}, "dependencies": {}, "resolutions": {}, "resolver_logs": {},
                "loader_environment_sha256": {name: hashlib.sha256(os.fsencode(environment[name])).hexdigest()
                    if name in environment else None for name in LINKAGE_ENVIRONMENT}}
    dependencies = snapshot["dependencies"]

    def portable_log(text):
        for prefix, alias in ((str(Path(build).resolve()), "{build}"), (str(ROOT), "{repo}")):
            text = text.replace(prefix, alias)
        return re.sub(r"(?<![\w}])(/[^\s()]+)", lambda match: "{runtime}/" + Path(match.group(1)).name, text)

    for name, binary in sorted(binaries.items()):
        snapshot["entrypoints"][name] = linkage_file(binary, build)
        stdout_path = log_dir / f"linkage-{epoch}-{name}.stdout.log"
        stderr_path = log_dir / f"linkage-{epoch}-{name}.stderr.log"
        # Reserve both logs before running, so retries cannot replace evidence.
        with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
            try:
                process = subprocess.run(["ldd", str(binary)], env={**environment, "LC_ALL": "C"},
                                         capture_output=True, text=True, timeout=30, check=False)
            except subprocess.TimeoutExpired as error:
                for handle, partial in ((stdout, error.stdout), (stderr, error.stderr)):
                    if isinstance(partial, bytes):
                        partial = partial.decode("utf-8", errors="replace")
                    handle.write(portable_log(partial or ""))
                raise ValueError(f"runtime linkage resolution timed out for {name}; see retained resolver logs") from error
            stdout.write(portable_log(process.stdout))
            stderr.write(portable_log(process.stderr))
        snapshot["resolver_logs"][name] = {
            "stdout": {"file": stdout_path.name, "sha256": sha256(stdout_path)},
            "stderr": {"file": stderr_path.name, "sha256": sha256(stderr_path)}, "returncode": process.returncode}
        if process.returncode != 0:
            raise ValueError(f"runtime linkage resolution failed for {name}; see retained resolver logs")
        resolution = snapshot["resolutions"][name] = {}
        for soname, path in sorted(parse_ldd(process.stdout).items()):
            record = linkage_file(path, build)
            identity = record["resolved_path_sha256"]
            previous = dependencies.setdefault(identity, {**record, "sonames": [], "required_by": []})
            if any(previous[key] != value for key, value in record.items()):
                raise ValueError("runtime dependency changed while resolving the entrypoints")
            previous["sonames"] = sorted(set(previous["sonames"]) | {soname})
            previous["required_by"] = sorted(set(previous["required_by"]) | {name})
            resolution[soname] = identity
    return validate_linkage_snapshot(snapshot)


def compare_runtime_linkage(before, after):
    before, after = validate_linkage_snapshot(before), validate_linkage_snapshot(after)
    return {"schema": 1, "method": LINKAGE_METHOD, "complete": True, "before": before, "after": after,
            "entrypoints_unchanged": before["entrypoints"] == after["entrypoints"],
            "dependencies_unchanged": before["dependencies"] == after["dependencies"],
            "resolution_unchanged": before["resolutions"] == after["resolutions"] and
                before["loader_environment_sha256"] == after["loader_environment_sha256"]}


def verify_runtime_linkage_receipt(linkage, binaries):
    if not isinstance(linkage, dict) or linkage.get("schema") != 1 or linkage.get("method") != LINKAGE_METHOD or \
            any(linkage.get(key) is not True for key in ("complete", "entrypoints_unchanged",
                                                       "dependencies_unchanged", "resolution_unchanged")):
        raise ValueError("runtime linkage before/after verification is missing, incomplete, or failed")
    recomputed = compare_runtime_linkage(linkage.get("before"), linkage.get("after"))
    if any(recomputed[key] is not True for key in ("entrypoints_unchanged", "dependencies_unchanged", "resolution_unchanged")):
        raise ValueError("runtime linkage bytes or resolution changed despite recorded success flags")
    if not isinstance(binaries, dict) or set(binaries) != LINKAGE_ENTRYPOINTS or any(
            not isinstance(binaries[name], dict) or any(
                binaries[name].get(key) != recomputed["before"]["entrypoints"][name][key] for key in ("path", "sha256"))
            for name in LINKAGE_ENTRYPOINTS):
        raise ValueError("runtime linkage entrypoints differ from the qualification binaries")
    return recomputed["after"]


def verify_validation_loaded_identities(validation, linkage_snapshot):
    """Cross-bind actual timing-worker mappings to native artifacts; ldd's GOMP may differ."""
    snapshot = validate_linkage_snapshot(linkage_snapshot)
    expected_library = snapshot["entrypoints"]["validation_library"]["sha256"]
    if validation.get("library_sha256") != expected_library:
        raise ValueError("loaded-runtime validation library differs from the qualified entrypoint")
    native_files = dict(snapshot["dependencies"])
    native_files.update({record["resolved_path_sha256"]: record for record in snapshot["entrypoints"].values()})
    benchmark = validation.get("benchmark", {})
    models = benchmark.get("models", {})
    precisions = ("f16", "q8_0", "q4_k")
    if set(models) != set(precisions):
        raise ValueError("loaded-runtime evidence must include all three precision timing phases")
    checked, processes = {}, set()
    for precision in precisions:
        evidence = models[precision].get("loaded_dso_identity")
        last = verify_timed_identity(evidence, native_sha256=expected_library, native_files=native_files)
        if last["cmake_cache_sha256"] != benchmark.get("runtime", {}).get("cmake_cache_sha256"):
            raise ValueError("loaded-runtime build configuration differs from benchmark provenance")
        processes.add(last["process_id"])
        checked[precision] = {"complete": True, "process_id": last["process_id"],
            "receipt_field": "validation/validation.json:benchmark.models." + precision + ".loaded_dso_identity"}
    if len(processes) != 1 or benchmark.get("runtime", {}).get("loaded_dso_initialization") != \
            models["f16"]["loaded_dso_identity"]["initialization"]:
        raise ValueError("loaded-runtime provenance does not identify the single actual benchmark worker")
    return {"complete": True,
        "scope": "Actual timing-worker mapped file identities after warmups and after all pairs; "
                 "native export/ggml/configured OpenBLAS cross-bound to standalone artifacts. "
                 "Torch/OpenMP mappings are recorded separately; this is not relocation-binding proof.",
        "models": checked}


def snapshot_sources(run, source):
    destination = run.output / "source-snapshot"
    manifest = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required qualification source is missing: {relative}")
        target = destination / "CrispASR" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest["CrispASR/" + relative] = sha256(target)
    for relative in SOURCE_HASHES:
        path = source / relative
        target = destination / "SCNet" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest["SCNet/" + relative] = sha256(target)
    write_json(run.output / "source-manifest.json", manifest)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run.receipt["crispasr_revision"] = head
    patch = subprocess.check_output(["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--", *SOURCE_FILES], cwd=ROOT)
    (destination / "uncommitted.patch").write_bytes(patch)
    (run.output / "REPRODUCE.txt").write_text(
        "SCNet qualification replication\n\n"
        f"CrispASR source: {head}\n"
        f"Original SCNet source: {SOURCE_REVISION}\n"
        "Use the retained scnet-qualification.yml for the exact build and dependency commands.\n"
        "The source-snapshot directory contains the exact converter, oracle, validator, native backend,\n"
        "runtime helper headers, quantizer source, original Python files and upstream MIT notice.\n"
        "Source hashes are in source-manifest.json; local generated-source changes are in uncommitted.patch.\n\n"
        "Clone starrytong/SCNet into a fresh assets-dir/SCNet and check out the revision above.\n"
        "After building the CLI, full Session library, scnet-probe, scnet-validation and quantizer, run:\n"
        f"  python tools/ci-heavy/scnet_qualify.py --variant {run.args.variant} "
        "--build-dir build-scnet --assets-dir fresh-scnet-assets --output-dir fresh-scnet-out --threads 2\n\n"
        "The script downloads only the author-linked checkpoint/config and the torchaudio music fixture.\n"
        "Checkpoint SHA-256 is discovered on that fetch; it is not independently author-attested.\n"
        "Every subsequent converter/reference/validator load binds to those exact bytes.\n"
        "Music attribution and exact contiguous interval are in fixture-provenance.json.\n"
        "No looping, resampling, gain fitting, output realignment or adaptive threshold changes are allowed.\n\n"
        "Default evidence artifacts omit the music/checkpoint/model files and raw reference/native GGUF\n"
        "captures. They are retained in the worker assets directory and validation directory during\n"
        "the run. Recreate them from the recorded source URLs, hashes, profile and commands above.\n"
        "After every gate passes, the separate qualified-model artifact retains the three converted\n"
        "GGUFs with their MIT notice, exact conversion/quantizer provenance and accepted receipts.\n"
        "Each command's stdout/stderr and failed exit code remain under logs/. Validation stores every\n"
        "completed timing call incrementally. Reuse neither an old output directory nor derived assets.\n",
        encoding="utf-8")
    cache = run.args.build_dir / "CMakeCache.txt"
    if cache.is_file():
        selected = {}
        for line in cache.read_text(encoding="utf-8").splitlines():
            if not line.startswith(("#", "//")) and ":" in line and "=" in line:
                key, rest = line.split(":", 1)
                if key in {"CMAKE_BUILD_TYPE", "BUILD_SHARED_LIBS", "GGML_BLAS", "GGML_BLAS_VENDOR",
                           "GGML_CUDA", "GGML_METAL", "GGML_OPENMP", "CRISPASR_BUILD_SCNET_PROBE"}:
                    selected[key] = rest.split("=", 1)[1]
        run.receipt["build_options"] = selected
    run.save()


def prepare_fixture(source, full, smoke):
    import numpy as np
    import soundfile as sf

    with sf.SoundFile(source) as handle:
        if handle.samplerate != 44100 or handle.channels != 2 or len(handle) < 24 * 44100:
            raise ValueError("authoritative fixture must contain >=24 contiguous seconds of native 44100-Hz stereo")
        original = handle.read(24 * 44100, dtype="float32", always_2d=True)
    if not np.isfinite(original).all():
        raise ValueError("source fixture contains nonfinite samples")
    for path, count in ((full, 24 * 44100), (smoke, 11025)):
        if path.exists():
            raise FileExistsError(f"refusing to replace existing derived fixture: {path.name}")
        sf.write(path, original[:count], 44100, format="WAV", subtype="FLOAT")
        decoded, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate != 44100 or not np.array_equal(decoded, original[:count]):
            raise ValueError("derived fixture did not preserve the exact contiguous decoded source samples")
    return dict(MUSIC_ATTRIBUTION, source_sha256=sha256(source), full_sha256=sha256(full),
                smoke_sha256=sha256(smoke), offset_samples=0, frames=24 * 44100, sample_rate=44100,
                audio_channels=2, resampled=False, looped=False, normalized=False, contiguous_exact_pcm=True)


def qualify(run):
    args = run.args
    source = args.assets_dir / "SCNet"
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    run.step("verify-original-source", lambda: verify_source_files(source))
    run.step("snapshot-sources", lambda: snapshot_sources(run, source))
    binaries = resolve_qualification_binaries(args.build_dir)
    production_library, validation_library = binaries["production_library"], binaries["validation_library"]
    linkage = run.receipt["runtime_linkage"] = {"schema": 1, "method": LINKAGE_METHOD, "complete": False,
        "entrypoints_unchanged": False, "dependencies_unchanged": False, "resolution_unchanged": False}
    linkage["before"] = run.step("snapshot-runtime-linkage-before", lambda: snapshot_runtime_linkage(
        binaries, args.build_dir, run.environment, run.output / "logs", "before"))
    run.receipt["binaries"] = {name: {key: record[key] for key in ("path", "sha256")}
                               for name, record in linkage["before"]["entrypoints"].items()}
    run.save()
    run.run("converter-controls", [sys.executable, ROOT / "tests/test-scnet-converter.py", "-v"])
    run.run("original-observer-controls", [sys.executable, ROOT / "tests/test-scnet-reference.py", "-v"])
    checkpoint = args.assets_dir / f"{args.variant}-author.th"
    checkpoint_provenance = run.asset("checkpoint", checkpoint, CHECKPOINT_URLS[args.variant], google=True,
                                     expected_sha256=RECORDED_CHECKPOINT_SHA256[args.variant])
    checkpoint_digest = checkpoint_provenance["sha256"]
    run.receipt["checkpoint"] = checkpoint_provenance
    config = source / "conf/config.yaml"
    if args.variant == "scnet-large":
        config = args.assets_dir / "scnet-large-author-config.yaml"
        run.receipt["config"] = run.asset("large-config", config, LARGE_CONFIG_URL, google=True,
                                         expected_sha256=RECORDED_LARGE_CONFIG_SHA256)
    else:
        run.receipt["config"] = {"sha256": sha256(config), "source_revision": SOURCE_REVISION,
                                   "source_path": "conf/config.yaml"}
    shutil.copyfile(config, run.output / "source-snapshot/SCNet/author-config.yaml")
    run.save()
    music = args.assets_dir / "hdemucs_mix.wav"
    run.asset("music", music, MUSIC_URL, expected_sha256=RECORDED_MUSIC_SHA256)
    full_audio, smoke_audio = args.assets_dir / "music-first-24s.wav", args.assets_dir / "music-first-0.25s.wav"
    run.receipt["fixture"] = run.step("prepare-fixture", lambda: prepare_fixture(music, full_audio, smoke_audio))
    write_json(run.output / "fixture-provenance.json", run.receipt["fixture"])
    models = {precision: args.assets_dir / f"{args.variant}-{precision}.gguf" for precision in ("f16", "q8_0", "q4_k")}
    if any(path.exists() for path in models.values()):
        raise FileExistsError("derived models already exist; use a fresh assets directory to retain previous experiments")
    common = ["--source-dir", source, "--checkpoint", checkpoint, "--checkpoint-sha256", checkpoint_digest,
              "--variant", args.variant]
    run.run("convert-f16", [sys.executable, ROOT / "models/convert-scnet-to-gguf.py", *common,
                            "--config", config, "--output", models["f16"], "--dtype", "f16"])
    for precision in ("q8_0", "q4_k"):
        run.run("quantize-" + precision, [binaries["crispasr-quantize"], models["f16"], models[precision], precision])
    run.receipt["models"] = {precision: {"path": run.portable(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                             for precision, path in models.items()}
    for suffix in (".provenance.json", ".LICENSE.txt"):
        shutil.copyfile(models["f16"].with_name(models["f16"].name + suffix), run.output / ("converted-model" + suffix))
    run.save()
    smoke_reference, full_reference = args.assets_dir / "reference-raw-smoke.gguf", args.assets_dir / "reference-full.gguf"
    reference_command = [sys.executable, ROOT / "tools/dump_scnet_reference.py", *common, "--device", "cpu", "--threads", args.threads]
    run.run("reference-raw-smoke", [*reference_command, "--audio", smoke_audio, "--raw", "--output", smoke_reference])
    from validate_scnet import GGUFFile, all_pass, compare_captures, parse_diff
    raw_smokes = run.receipt["raw_smoke"] = {}
    for precision, path in models.items():
        name = "native-raw-smoke-" + precision
        capture = args.assets_dir / (name + ".gguf")
        code = run.run(name, [binaries["scnet-probe"], "scnet", path, smoke_reference, smoke_audio],
            required=False, extra_env={"CRISPASR_SCNET_CAPTURE_GGUF": str(capture)})
        row = raw_smokes[precision] = {"returncode": code, "acceptance": False}
        try:
            row["capture_sha256"] = sha256(capture)
            row["stages"] = compare_captures(GGUFFile(capture), GGUFFile(smoke_reference), precision)
            if code == 0:
                row["native_log_metrics"] = parse_diff(
                    (run.output / "logs" / (name + ".stdout.log")).read_text(encoding="utf-8"), precision)
                if not all_pass(row["stages"]):
                    raise ValueError("independent raw capture metrics disagree with native acceptance")
                row["acceptance"] = True
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            run.receipt["errors"].append({"raw_smoke": precision, "error": row["error"]})
        run.save()
    smoke_capture = args.assets_dir / "native-raw-smoke-f16.gguf"
    smoke_code = raw_smokes["f16"]["returncode"]
    shutil.copyfile(smoke_reference.with_name(smoke_reference.name + ".json"), run.output / "reference-raw-smoke.json")
    if smoke_code != 0:
        # Keep the failed gate in errors/acceptance, but gather independent
        # storage/arithmetic evidence and the complete default inference path.
        # These diagnostic models/captures never replace the original oracle.
        diagnostic_model = args.assets_dir / f"{args.variant}-diagnostic-f32.gguf"
        diagnostic_capture = args.assets_dir / "native-raw-f32-diagnostic.gguf"
        converted = run.run("diagnostic-convert-f32", [sys.executable, ROOT / "models/convert-scnet-to-gguf.py", *common,
            "--config", config, "--output", diagnostic_model, "--dtype", "f32"], required=False)
        if converted == 0:
            run.run("diagnostic-native-f32", [binaries["scnet-probe"], "scnet", diagnostic_model, smoke_reference, smoke_audio],
                required=False, extra_env={"CRISPASR_SCNET_CAPTURE_GGUF": str(diagnostic_capture)})
        if smoke_capture.is_file():
            diagnostic_command = [sys.executable, ROOT / "tools/ci-heavy/scnet_precision_diagnostics.py", *common,
                "--reference-a", smoke_reference, "--native-capture", smoke_capture,
                "--native-f16-model", models["f16"], "--threads", args.threads,
                "--output-dir", run.output / "precision-diagnostics"]
            if diagnostic_capture.is_file():
                diagnostic_command += ["--native-f32-capture", diagnostic_capture, "--native-f32-model", diagnostic_model]
            run.run("diagnostic-precision-isolation", diagnostic_command, required=False)
        run.run("diagnostic-native-runtime", [sys.executable, ROOT / "tools/ci-heavy/scnet_runtime_profile.py",
            "--library", validation_library, "--model", models["f16"], "--audio", full_audio,
            "--threads", args.threads, "--calls", 3, "--output", run.output / "native-runtime-diagnostic.json"],
            required=False)
    run.run("reference-full", [*reference_command, "--audio", full_audio, "--output", full_reference])
    shutil.copyfile(full_reference.with_name(full_reference.name + ".json"), run.output / "reference-full.json")
    validation_dir = run.output / "validation"
    validation_code = run.run("full-qualification", [
        sys.executable, ROOT / "tools/validate_scnet.py", "--source-dir", source, "--checkpoint", checkpoint,
        "--variant", args.variant, "--f16", models["f16"], "--q8-0", models["q8_0"], "--q4-k", models["q4_k"],
        "--reference", full_reference, "--audio", full_audio, "--diff", binaries["scnet-probe"],
        "--library", validation_library, "--fixture-source", music, "--fixture-source-url", MUSIC_URL,
        "--threads", args.threads, "--warmups", 1, "--repetitions", 3, "--timeout", args.command_timeout,
        "--output-dir", validation_dir], required=False)
    validation_path = validation_dir / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else {}
    run.receipt["validation"] = {"receipt": "validation/validation.json", "returncode": validation_code,
                                  "complete": validation.get("complete", False),
                                  "acceptance": validation.get("acceptance", False)}
    if smoke_code == 0 and not validation.get("acceptance", False):
        run.run("diagnostic-native-runtime", [sys.executable, ROOT / "tools/ci-heavy/scnet_runtime_profile.py",
            "--library", validation_library, "--model", models["f16"], "--audio", full_audio,
            "--threads", args.threads, "--calls", 3, "--output", run.output / "native-runtime-diagnostic.json"],
            required=False)
    surfaces = run.receipt["surfaces"] = {}
    for precision, path in models.items():
        code = run.run("production-surface-" + precision, [sys.executable, ROOT / "tests/test-scnet-surface.py",
            "--crispasr", binaries["crispasr"], "--library", production_library, "--model", path], required=False)
        surfaces[precision] = {"returncode": code, "acceptance": code == 0,
                               "scope": "Actual production CLI and Python Session/C ABI; synthetic asymmetric stereo fixture."}
        run.save()
    verify_source_files(source)
    if sha256(checkpoint) != checkpoint_digest:
        raise ValueError("original checkpoint changed during qualification")
    for precision, path in models.items():
        if sha256(path) != run.receipt["models"][precision]["sha256"]:
            raise ValueError(f"converted model changed during qualification: {precision}")
    linkage["after"] = run.step("snapshot-runtime-linkage-after", lambda: snapshot_runtime_linkage(
        binaries, args.build_dir, run.environment, run.output / "logs", "after"))
    linkage.update(compare_runtime_linkage(linkage["before"], linkage["after"]))
    run.save()  # Preserve mismatched identities before rejecting a changed runtime.
    verify_runtime_linkage_receipt(linkage, run.receipt["binaries"])
    try:
        run.receipt["loaded_runtime_identities"] = verify_validation_loaded_identities(validation, linkage["after"])
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        run.receipt["loaded_runtime_identities"] = {"complete": False, "error": message}
        run.receipt["errors"].append(message)
    run.save()
    run.receipt["complete"] = bool(all(row["returncode"] is not None for row in raw_smokes.values())
                                   and validation.get("complete", False)
                                   and all(row["returncode"] is not None for row in surfaces.values()))
    run.receipt["acceptance"] = bool(all(row["acceptance"] for row in raw_smokes.values())
                                      and validation_code == 0 and validation.get("acceptance", False)
                                      and all(row["acceptance"] for row in surfaces.values()) and not run.receipt["errors"])
    run.receipt["status"] = "passed" if run.receipt["acceptance"] else "not_qualified"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=tuple(CHECKPOINT_URLS), required=True)
    for name in ("build-dir", "assets-dir", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--command-timeout", type=int, default=7200)
    args = parser.parse_args(argv)
    if args.threads != 2 or args.command_timeout <= 0:
        parser.error("this recorded CPU profile requires --threads 2 and a positive command timeout")
    for name in ("build_dir", "assets_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        run = Qualification(args)
    except FileExistsError:
        parser.exit(2, "output directory already exists; choose a new directory to preserve previous evidence\n")
    try:
        qualify(run)
    except BaseException as error:
        run.receipt["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "error"
        run.receipt["errors"].append({"exception": f"{type(error).__name__}: {error}"})
        run.receipt["acceptance"] = False
        print(f"SCNet qualification stopped: {error}", file=sys.stderr)
    finally:
        run.receipt["finished_at"] = utc_now()
        run.save()
    print(json.dumps({"receipt": str(run.receipt_path), "status": run.receipt["status"],
                      "complete": run.receipt["complete"], "acceptance": run.receipt["acceptance"]}))
    return 0 if run.receipt["acceptance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
