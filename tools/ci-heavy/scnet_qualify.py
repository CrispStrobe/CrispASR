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
import json
import os
from pathlib import Path
import platform
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

MUSIC_URL = "https://download.pytorch.org/torchaudio/tutorial-assets/hdemucs_mix.wav"
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
    "tools/scnet_probe.cpp", "tools/scnet_exports.cpp", "tools/validate_scnet.py",
    "tools/ci-heavy/scnet_qualify.py", "tools/dump_scnet_reference.py",
    "tools/reference_backends/scnet.py", "tools/requirements-scnet.txt",
    "models/convert-scnet-to-gguf.py", "models/licenses/SCNet-MIT.txt",
    "tests/test-scnet-converter.py", "tests/test-scnet-reference.py", "tests/test-scnet-native-diff.py",
    "tests/test-scnet-surface.py", "examples/cli/crispasr_backend_scnet.cpp",
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
            "artifact_policy": "Upload JSON/logs/source/config/patch only; music, checkpoints, GGUFs and raw captures remain worker-local.",
        }
        self.environment = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "CRISPASR_SCNET_THREADS"):
            self.environment[name] = str(args.threads)
        self.environment.update(ACCELERATE_USE_CPU="true", CRISPASR_SCNET_GPU="0",
                                SCNET_SOURCE_DIR=str(args.assets_dir / "SCNet"))
        for name in ("CRISPASR_SCNET_CAPTURE_GGUF", "CRISPASR_SCNET_BENCH", "CRISPASR_SCNET_SCALAR_LSTM",
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

    def asset(self, name, path, url, google=False):
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
        write_json(self.output / f"{name}-provenance.json", provenance)
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
    binaries = {name: resolve_binary(args.build_dir, name) for name in ("crispasr", "crispasr-quantize", "scnet-probe")}
    production_library = resolve_library(args.build_dir, "crispasr")
    validation_library = resolve_library(args.build_dir, "scnet-validation")
    run.receipt["binaries"] = {name: {"path": run.portable(path), "sha256": sha256(path)}
                               for name, path in dict(binaries, production_library=production_library,
                                                     validation_library=validation_library).items()}
    run.save()
    run.run("converter-controls", [sys.executable, ROOT / "tests/test-scnet-converter.py", "-v"])
    run.run("original-observer-controls", [sys.executable, ROOT / "tests/test-scnet-reference.py", "-v"])
    checkpoint = args.assets_dir / f"{args.variant}-author.th"
    checkpoint_provenance = run.asset("checkpoint", checkpoint, CHECKPOINT_URLS[args.variant], google=True)
    checkpoint_digest = checkpoint_provenance["sha256"]
    run.receipt["checkpoint"] = checkpoint_provenance
    config = source / "conf/config.yaml"
    if args.variant == "scnet-large":
        config = args.assets_dir / "scnet-large-author-config.yaml"
        run.receipt["config"] = run.asset("large-config", config, LARGE_CONFIG_URL, google=True)
    else:
        run.receipt["config"] = {"sha256": sha256(config), "source_revision": SOURCE_REVISION,
                                   "source_path": "conf/config.yaml"}
    shutil.copyfile(config, run.output / "source-snapshot/SCNet/author-config.yaml")
    run.save()
    music = args.assets_dir / "hdemucs_mix.wav"
    run.asset("music", music, MUSIC_URL)
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
    run.run("native-raw-smoke", [binaries["scnet-probe"], "scnet", models["f16"], smoke_reference, smoke_audio],
            extra_env={"CRISPASR_SCNET_CAPTURE_GGUF": str(args.assets_dir / "native-raw-smoke.gguf")})
    shutil.copyfile(smoke_reference.with_name(smoke_reference.name + ".json"), run.output / "reference-raw-smoke.json")
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
    run.receipt["complete"] = bool(validation.get("complete", False) and all(row["returncode"] is not None for row in surfaces.values()))
    run.receipt["acceptance"] = bool(validation_code == 0 and validation.get("acceptance", False)
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
