#!/usr/bin/env python3
"""Retain exact, fully qualified SCNet GGUFs as a GitHub Actions artifact.

This is an offline packaging step, not a model publisher or a new qualification.
It verifies the completed run and all input identities before creating output.
No audio, author checkpoint, activation capture, or diagnostic model is copied.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid

from scnet_qualify import (
    MUSIC_URL, RECORDED_CHECKPOINT_SHA256, RECORDED_LARGE_CONFIG_SHA256,
    RECORDED_MUSIC_SHA256, ROOT, SOURCE_FILES, compare_runtime_linkage, resolve_qualification_binaries,
    snapshot_runtime_linkage, verify_runtime_linkage_receipt, verify_validation_loaded_identities,
)
from reference_backends.scnet import (
    CHECKPOINT_URLS, LICENSE_URL, SOURCE_HASHES, SOURCE_REVISION, SOURCE_URL,
    SOURCES, sha256,
)
from validate_scnet import GGUFFile, PRECISIONS, REQUIRED_STAGES, check_inventory, summarize_pairs

RECONSTRUCTION_F16 = (
    "decoder.2.0.conv.weight", "decoder.2.1.convtrs.0.weight",
    "decoder.2.1.convtrs.1.weight", "decoder.2.1.convtrs.2.weight",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON value in {path.name}: {value}")
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)
    require(isinstance(value, dict), f"expected a JSON object: {path.name}")
    return value


def accepted(receipt, label):
    require(receipt.get("schema") == 1 and receipt.get("complete") is True and
            receipt.get("acceptance") is True and receipt.get("status") == "passed" and
            receipt.get("errors") == [], f"{label} is not a complete, accepted qualification")


def stages_pass(row, label):
    require(set(row.get("stages", {})) == set(REQUIRED_STAGES) and
            all(stage.get("acceptance") is True for stage in row["stages"].values()),
            f"{label} lacks all 35 accepted stages")


def verify_receipts(qualification, validation, variant, source_commit):
    accepted(qualification, "qualification receipt")
    accepted(validation, "validation receipt")
    linkage = verify_runtime_linkage_receipt(qualification.get("runtime_linkage"), qualification.get("binaries"))
    loaded = verify_validation_loaded_identities(validation, linkage)
    require(qualification.get("loaded_runtime_identities") == loaded,
            "qualification lacks the verified actual timing-worker runtime identities")
    for label, receipt in (("qualification", qualification), ("validation", validation)):
        require(receipt.get("variant") == variant and receipt.get("source_revision") == SOURCE_REVISION,
                f"{label} source/variant differs from the requested release")
        require(set(receipt.get("models", {})) == set(PRECISIONS), f"{label} must contain exactly three precisions")
        require(bool(receipt.get("commands")) and all(event.get("complete") is True and
                event.get("returncode") == 0 for event in receipt["commands"]), f"{label} has incomplete/failed commands")
    require(qualification.get("crispasr_revision") == source_commit, "qualification source commit differs")
    require(qualification.get("threads") == 2, "qualification does not use the recorded two-thread profile")
    require(qualification.get("validation") == {"receipt": "validation/validation.json", "returncode": 0,
            "complete": True, "acceptance": True}, "qualification does not identify the accepted validator")
    require(validation.get("diagnostic") is False and validation.get("stages_only") is False and
            validation.get("numerical_acceptance") is True, "diagnostic/stages-only result cannot be packaged")
    require(validation.get("source_file_hashes") == SOURCE_HASHES, "validator original-source hashes differ")
    checkpoint = qualification.get("checkpoint", {})
    require(checkpoint.get("sha256") == RECORDED_CHECKPOINT_SHA256[variant] == validation.get("checkpoint_sha256")
            and checkpoint.get("source_url") == CHECKPOINT_URLS[variant], "original checkpoint identity differs")
    fixture, observed = qualification.get("fixture", {}), validation.get("fixture", {})
    require(fixture.get("source_url") == observed.get("source_url") == MUSIC_URL and
            fixture.get("source_sha256") == observed.get("source_sha256") == RECORDED_MUSIC_SHA256 and
            fixture.get("full_sha256") == validation.get("audio_sha256") and
            fixture.get("frames") == observed.get("samples") == 24 * 44100 and
            fixture.get("offset_samples") == observed.get("offset_samples") == 0 and
            fixture.get("sample_rate") == 44100 and fixture.get("audio_channels") == 2 and
            fixture.get("contiguous_exact_pcm") is True and observed.get("contiguous_exact_pcm") is True and
            all(fixture.get(key) is False for key in ("resampled", "looped", "normalized")),
            "fixture does not match the exact contiguous 24-second CPU qualification")
    profile = {"mode": "pipeline", "segment_seconds": 20, "shifts": 1, "shift_seed": 0,
               "overlap": 0.5, "samples": 24 * 44100, "sample_rate": 44100, "full_profile": True}
    benchmark = validation.get("benchmark", {})
    require(validation.get("profile") == profile == benchmark.get("profile") and
            benchmark.get("acceptance") is True and benchmark.get("device") == "cpu" and
            benchmark.get("threads") == 2 and benchmark.get("repetitions") == 3 and
            benchmark.get("warmups") == 1 and benchmark.get("original_dtype") == "float32",
            "full original CPU inference/benchmark profile is missing")
    for key in ("raw_smoke", "surfaces"):
        require(set(qualification.get(key, {})) == set(PRECISIONS), f"missing {key} precision")
    require(set(benchmark.get("models", {})) == set(PRECISIONS), "missing benchmark precision")
    for precision in PRECISIONS:
        raw, surface = qualification["raw_smoke"][precision], qualification["surfaces"][precision]
        require(raw.get("returncode") == surface.get("returncode") == 0 and
                raw.get("acceptance") is True and surface.get("acceptance") is True,
                f"{precision} raw/surface qualification did not pass")
        stages_pass(raw, precision + " raw")
        row, timed = validation["models"][precision], benchmark["models"][precision]
        require(row.get("numerical_acceptance") is True, f"{precision} numerical gate failed")
        stages_pass(row, precision + " pipeline")
        require(timed.get("complete") is True and timed.get("acceptance") is True and
                timed.get("native_backend", "").upper().startswith("CPU"), f"{precision} CPU benchmark did not pass")
        summary = summarize_pairs(timed.get("calls", []), 3)
        require(summary["performance_acceptance"] and all(timed.get(key) == value for key, value in summary.items()),
                f"{precision} timing summary differs from retained paired calls")
        warmups = timed.get("warmups", [])
        require([(row.get("iteration"), row.get("implementation")) for row in warmups] ==
                [(0, "original"), (0, "native")], f"{precision} warmup sequence is incomplete")
        for call in warmups + timed["calls"]:
            require(call.get("quality_acceptance") is True and set(call.get("stems", {})) == set(SOURCES) and
                    all(stem.get("acceptance") is True for stem in call["stems"].values()),
                    f"{precision} measured waveform quality is incomplete")


def verify_packaging_linkage(qualification, binaries, build_dir, log_dir):
    previous = verify_runtime_linkage_receipt(qualification.get("runtime_linkage"), qualification.get("binaries"))
    for epoch in ("before", "after"):
        for record in qualification["runtime_linkage"][epoch]["resolver_logs"].values():
            for stream in ("stdout", "stderr"):
                output = record[stream]
                require(sha256(Path(log_dir) / output["file"]) == output["sha256"],
                        "qualified runtime linkage resolver log changed")
    current = snapshot_runtime_linkage(binaries, build_dir, dict(os.environ), log_dir,
                                       "package-" + uuid.uuid4().hex)
    comparison = compare_runtime_linkage(previous, current)
    verify_runtime_linkage_receipt(comparison, qualification["binaries"])
    return current


def prepare(args):
    directory = args.qualification_dir
    qualification = read_json(directory / "qualification.json")
    # Reject historical failures before looking for any absent model/validator.
    accepted(qualification, "qualification receipt")
    validation = read_json(directory / "validation/validation.json")
    verify_receipts(qualification, validation, args.variant, args.source_commit)
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == args.source_commit,
            "checkout differs from the requested source commit")
    sources = read_json(directory / "source-manifest.json")
    expected_sources = {"CrispASR/" + name for name in SOURCE_FILES} | {"SCNet/" + name for name in SOURCE_HASHES}
    require(set(sources) == expected_sources and "CrispASR/tools/ci-heavy/scnet_package.py" in sources,
            "source manifest is incomplete or does not bind this packaging helper")
    for relative, digest in sources.items():
        require(sha256(directory / "source-snapshot" / relative) == digest, f"source snapshot changed: {relative}")
        if relative.startswith("CrispASR/"):
            require(sha256(ROOT / relative.removeprefix("CrispASR/")) == digest, f"qualification source changed: {relative}")
        else:
            require(digest == SOURCE_HASHES[relative.removeprefix("SCNet/")], f"original source changed: {relative}")
    binaries = resolve_qualification_binaries(args.build_dir)
    require(set(qualification.get("binaries", {})) == set(binaries), "qualified binary inventory differs")
    for name, path in binaries.items():
        require(sha256(path) == qualification["binaries"][name].get("sha256"), f"qualified binary changed: {name}")
    require(validation.get("diff_sha256") == qualification["binaries"]["scnet-probe"]["sha256"] and
            validation.get("library_sha256") == qualification["binaries"]["validation_library"]["sha256"],
            "validator binary identity differs from the qualification")
    packaging_linkage = verify_packaging_linkage(qualification, binaries, args.build_dir, directory / "logs")
    commands = {event["name"]: event for event in qualification["commands"]}
    require(len(commands) == len(qualification["commands"]), "duplicate qualification command names")
    for precision in ("q8_0", "q4_k"):
        expected = [qualification["binaries"]["crispasr-quantize"]["path"],
                    "{assets}/" + args.variant + "-f16.gguf", "{assets}/" + args.variant + "-" + precision + ".gguf",
                    precision]
        require(commands.get("quantize-" + precision, {}).get("argv") == expected,
                f"{precision} was not produced by the default quantizer recipe")

    license_path = directory / "converted-model.LICENSE.txt"
    require(sha256(license_path) == SOURCE_HASHES["LICENSE"], "upstream MIT notice differs")
    license_text = license_path.read_text(encoding="utf-8")
    conversion_path = directory / "converted-model.provenance.json"
    conversion = read_json(conversion_path)
    checkpoint_hash = RECORDED_CHECKPOINT_SHA256[args.variant]
    require(conversion.get("architecture") == "scnet" and conversion.get("variant") == args.variant and
            conversion.get("dtype") == "f16" and conversion.get("source_revision") == SOURCE_REVISION and
            conversion.get("source_hashes") == SOURCE_HASHES and conversion.get("checkpoint_sha256") == checkpoint_hash
            and conversion.get("checkpoint_url") == CHECKPOINT_URLS[args.variant] and conversion.get("license") == "MIT",
            "F16 conversion provenance differs")
    config_hash = RECORDED_LARGE_CONFIG_SHA256 if args.variant == "scnet-large" else SOURCE_HASHES["conf/config.yaml"]
    require(qualification.get("config", {}).get("sha256") == config_hash, "author configuration identity differs")
    payloads, model_records = {}, {}
    base = GGUFFile(args.assets_dir / f"{args.variant}-f16.gguf")
    for precision in PRECISIONS:
        name = f"{args.variant}-{precision}.gguf"
        path = args.assets_dir / name
        model = base if precision == "f16" else GGUFFile(path)
        recorded, checked = qualification["models"][precision], validation["models"][precision]
        digest, size = sha256(path), path.stat().st_size
        require(recorded.get("path") == "{assets}/" + name and
                recorded.get("sha256") == checked.get("sha256") == digest and
                recorded.get("bytes") == checked.get("file_bytes") == size and
                validation["benchmark"]["models"][precision].get("model_sha256") == digest,
                f"{precision} model bytes differ from the accepted receipts")
        require(model.metadata.get("scnet.variant") == args.variant and
                model.metadata.get("scnet.config.sha256") == config_hash and
                model.metadata.get("general.license") == "mit" and
                model.metadata.get("general.license.text") == license_text,
                f"{precision} variant/configuration/license metadata differs")
        coverage = check_inventory(model, precision, None if precision == "f16" else base, checkpoint_hash, SOURCE_HASHES)
        require(coverage == checked.get("coverage"), f"{precision} storage coverage changed")
        for tensor in RECONSTRUCTION_F16:
            require(model.tensors.get(tensor, {}).get("type") == "f16" and
                    model.tensor_hash(tensor) == base.tensor_hash(tensor), f"{precision} reconstruction policy differs")
        payloads[name] = path
        model_records[precision] = {"file": name, "bytes": size, "sha256": digest, "coverage": coverage}
    require(conversion.get("gguf_sha256") == model_records["f16"]["sha256"], "F16 conversion payload hash differs")
    payloads.update({"LICENSE": license_path, "F16-CONVERSION.json": conversion_path,
                     "qualification.json": directory / "qualification.json",
                     "validation.json": directory / "validation/validation.json",
                     "source-manifest.json": directory / "source-manifest.json"})
    # Bind the interpreted receipt objects as well as the payload bytes before
    # copying, so a concurrently replaced receipt cannot describe other inputs.
    for name, expected in (("qualification.json", qualification), ("validation.json", validation),
                           ("source-manifest.json", sources), ("F16-CONVERSION.json", conversion)):
        require(read_json(payloads[name]) == expected, f"verified receipt changed: {name}")
    payloads = {name: {"path": path, "sha256": sha256(path), "bytes": path.stat().st_size}
                for name, path in payloads.items()}
    manifest = {
        "schema": 1, "variant": args.variant, "artifact_kind": "qualified_cpu_gguf_bundle",
        "complete": True, "acceptance": True, "publication": False, "models": model_records,
        "crispasr_source_commit": args.source_commit, "upstream_source_url": SOURCE_URL,
        "upstream_source_revision": SOURCE_REVISION, "upstream_source_hashes": SOURCE_HASHES,
        "checkpoint": qualification["checkpoint"], "config_sha256": config_hash,
        "license": "MIT", "checkpoint_license_evidence": LICENSE_URL,
        "quantization": {"source_file": "examples/crispasr-quantize/main.cpp",
            "source_sha256": sources["CrispASR/examples/crispasr-quantize/main.cpp"],
            "binary_sha256": qualification["binaries"]["crispasr-quantize"]["sha256"],
            "policy": "SCNet Q8_0/Q4_K preserve exactly the four final reconstruction matrices at F16; "
                      "norms/biases stay F32; Q4_K non-256-wide aligned rows fall back to Q8_0; "
                      "unsupported row widths retain source precision; no tensor overrides.",
            "reconstruction_f16_tensors": list(RECONSTRUCTION_F16)},
        "qualification": {"scope": qualification["qualification_scope"], "device": "cpu", "threads": 2,
            "profile": validation["profile"], "runtime": validation["benchmark"]["runtime"],
            "runtime_linkage": {"qualified_before_after_verified": True, "packaging_recheck_verified": True,
                "scope": "Standalone ldd resolution; actual timing-worker mappings are separately recorded",
                "qualified_receipt_field": "qualification.json:runtime_linkage", "packaging_snapshot": packaging_linkage},
            "loaded_runtime_identities": qualification["loaded_runtime_identities"],
            "fixture": qualification["fixture"], "gpu_qualified": False, "held_out_quality_qualified": False,
            "other_variant_qualified_by_this_bundle": False},
        "excluded": ["music/audio", "author checkpoint", "reference/native activation captures",
                     "diagnostic F32 models", "runtime binaries"],
    }
    return manifest, payloads


def write_bundle(args, manifest, payloads):
    require(not args.output_dir.exists(), "output already exists; preserve it and choose a new directory")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=args.output_dir.name + ".partial-", dir=args.output_dir.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        files = {}
        for name, record in payloads.items():
            source, digest, size = record["path"], record["sha256"], record["bytes"]
            target = staging / name
            shutil.copyfile(source, target)
            require(target.stat().st_size == size and sha256(target) == digest, f"copied artifact changed: {name}")
            if name.endswith(".gguf"):
                precision = next(p for p in PRECISIONS if manifest["models"][p]["file"] == name)
                require(digest == manifest["models"][precision]["sha256"], f"qualified model changed before copy: {name}")
            files[name] = {"bytes": size, "sha256": digest}
        readme = staging / "README.md"
        readme.write_text(f"""# {args.variant} — qualified CPU GGUFs

These three converted files passed the retained SCNet qualification for the
recorded CPU, two threads, and one contiguous 24-second implementation-parity
music fixture. This artifact covers **{args.variant} only**. It does not establish
GPU performance/fidelity, other-host performance, or held-out separation quality.

Use a CrispASR build from source commit `{args.source_commit}`:

```sh
mkdir -p stems
crispasr --separate -m {args.variant}-q8_0.gguf -f song.wav -t 2 --no-gpu --sep-output-dir stems
```

Replace the model filename with the included F16 or Q4_K file as desired.
The outputs are stereo drums, bass, other, and vocals at 44,100 Hz. The default
original inference profile uses 20-second chunks, 50% overlap, one random shift
per chunk, and seed zero. Use concrete paths; this bundle adds no auto-download
registry entry and contains no runtime binary.

F16/Q8_0/Q4_K name mixed storage profiles. Norms and biases remain F32; all
profiles preserve the final reconstruction matrices at F16. `manifest.json`
records the exact quantization policy, file sizes/hashes, and source identities.
`F16-CONVERSION.json` describes the F16 conversion; the manifest additionally
binds the two quantized derivatives. `qualification.json` and `validation.json`
retain the stage, waveform, production-surface, and paired timing evidence.
The qualification also binds all five native entrypoints and their resolved
ggml, OpenBLAS, and transitive runtime libraries before and after execution.
The manifest records a fresh identity check before packaging. Runtime binaries
are not included; resolver logs belong to the accompanying evidence artifact.

Weights and original source are MIT licensed by the SCNet authors; preserve
the included original `LICENSE`. Permission for converted checkpoints is linked
in the manifest. Checkpoint digests identify the first author-linked downloads;
they are not independently author-published checksums. No fixture audio, author
checkpoint, raw activation capture, or diagnostic F32 model is included.

This bundle retains the exact qualification provenance recorded in its receipts.
""", encoding="utf-8")
        files["README.md"] = {"bytes": readme.stat().st_size, "sha256": sha256(readme)}
        manifest["files"] = files
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                              encoding="utf-8")
        staging.rename(args.output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(CHECKPOINT_URLS), required=True)
    parser.add_argument("--source-commit", required=True)
    for name in ("build-dir", "assets-dir", "qualification-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="verify accepted inputs without creating a bundle")
    args = parser.parse_args(argv)
    if not re.fullmatch("[0-9a-f]{40}", args.source_commit):
        parser.error("--source-commit must be the exact 40-hex checkout revision")
    for name in ("build_dir", "assets_dir", "qualification_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        manifest, payloads = prepare(args)
        if not args.check_only:
            write_bundle(args, manifest, payloads)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"SCNet packaging rejected: {error}\n")
    print(json.dumps({"variant": args.variant, "source_commit": args.source_commit,
                      "verified_models": list(manifest["models"]), "bundle_written": not args.check_only}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
