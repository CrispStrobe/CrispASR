#!/usr/bin/env python3
"""Hosted, weight-free runner for the existing CSM RoPE probe and comparator."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import csm_rope_component_check as check
import nightly_error_ab as ab


ROOT = Path(__file__).resolve().parents[2]
MODES = ("original", "generated", "unscaled")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_file(path):
    if not path.is_file():
        raise RuntimeError(f"required file missing: {path}")
    return path


def _library(build, stem):
    matches = sorted(path for path in build.rglob(f"{stem}.so*") if path.is_file())
    if not matches:
        raise RuntimeError(f"built shared library missing: {stem}")
    real = {path.resolve() for path in matches}
    if len(real) != 1:
        raise RuntimeError(f"ambiguous built shared library: {stem}: {matches}")
    return real.pop()


def _command(argv, cwd, out, label, *, timeout, env=None):
    argv = list(map(str, argv))
    (out / f"{label}.argv.json").write_text(json.dumps(argv, indent=2) + "\n")
    with (out / f"{label}.stdout.log").open("wb") as stdout, \
            (out / f"{label}.stderr.log").open("wb") as stderr:
        try:
            result = subprocess.run(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                    timeout=timeout, check=False)
            return result.returncode
        except subprocess.TimeoutExpired:
            return 124


def _reference(reference):
    manifest = json.loads(_required_file(reference / "receipt.json").read_text())
    if manifest.get("complete") is not True or \
            manifest.get("source_revision") != check.SOURCE_REVISION or \
            manifest.get("config_sha256") != check.CONFIG_SHA256:
        raise RuntimeError("original operator receipt is incomplete or has wrong pins")
    if sha256(_required_file(reference / "original-config.json")) != check.CONFIG_SHA256:
        raise RuntimeError("original configuration bytes differ from immutable pin")
    config = json.loads((reference / "original-config.json").read_text())
    if set(manifest.get("cases", {})) != set(check.CASES):
        raise RuntimeError("original receipt must contain backbone and depth")
    for case in check.CASES:
        attributes = config if case == "backbone" else config["depth_decoder_config"]
        saved = manifest["cases"][case]["configuration"]
        if saved != {key: attributes[key] for key in saved}:
            raise RuntimeError(f"{case}: reference case config differs from pinned configuration")
        files = manifest["cases"][case]["files"]
        arrays = {key: check._reference_array(reference, files, case, key)
                  for key in check.REFERENCE_KEYS}
        positions = arrays["positions"]
        dim, qheads, kheads = (int(attributes[key]) for key in
                              ("head_dim", "num_attention_heads", "num_key_value_heads"))
        if arrays["q"].shape != (len(positions), qheads, dim) or \
                arrays["k"].shape != (len(positions), kheads, dim) or \
                arrays["freq_factors"].shape != (dim // 2,):
            raise RuntimeError(f"{case}: reference array shapes differ from pinned configuration")
        scaling = attributes["rope_scaling"]
        if scaling.get("rope_type", scaling.get("type")) != "llama3":
            raise RuntimeError(f"{case}: pinned scaling is not llama3")
    if not check.compare(reference, reference)["pass"]:
        raise RuntimeError("original fixture fails its own locked component gates")
    return manifest


def _unscaled_check(reference, candidate, manifest):
    result = check.compare(reference, candidate)
    if result["pass"]:
        raise RuntimeError("unscaled probe unexpectedly passes scaled original reference")
    controls = {}
    for case in check.CASES:
        files = manifest["cases"][case]["files"]
        controls[case] = {}
        for prefix in ("q", "k"):
            expected = check._reference_array(reference, files, case, f"{prefix}_unscaled")
            actual = check._read_array(candidate / case / f"{prefix}_rotated.bin", expected.shape, "float32")
            metrics = check._metrics(expected, actual)
            if not check._rotation_pass(metrics):
                raise RuntimeError(f"{case}/{prefix}: unscaled probe misses original unscaled operator")
            if result["cases"][case]["candidate"][f"{prefix}_rotated"]["pass"]:
                raise RuntimeError(f"{case}/{prefix}: unscaled rotation passes scaled original gate")
            controls[case][prefix] = metrics
        factors = check._read_array(candidate / case / "freq_factors.bin", (int(files["freq_factors"]["shape"][0]),),
                                    "float32")
        if not check.np.array_equal(factors, check.np.ones_like(factors)):
            raise RuntimeError(f"{case}: unscaled probe did not use unit frequency factors")
    return {"scaled_reference": result, "original_unscaled_match": controls, "pass": True}


def run(source, build, profile, reference, out):
    source, build, reference, out = map(Path, (source, build, reference, out))
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"complete": False, "pass": False, "scope": __doc__, "profile": profile,
               "source": str(source), "build": str(build), "reference": str(reference),
               "stages": {}, "cases": {}}

    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    try:
        if profile not in ("native", "avx2"):
            raise RuntimeError("profile must be native or avx2")
        manifest = _reference(reference)
        receipt["reference_receipt_sha256"] = sha256(reference / "receipt.json")
        receipt["reference_config_sha256"] = check.CONFIG_SHA256
        receipt["stages"]["reference_verified"] = True
        save()
        cache = _required_file(build / "CMakeCache.txt").read_text()
        expected = "GGML_NATIVE:BOOL=ON" if profile == "native" else "GGML_NATIVE:BOOL=OFF"
        if expected not in cache or "GGML_BACKEND_DL:BOOL=ON" in cache:
            raise RuntimeError("profile flags or direct shared GGML linkage differ from expected build")
        if profile == "avx2" and ("GGML_AVX2:BOOL=ON" not in cache or "GGML_AVX512:BOOL=OFF" not in cache):
            raise RuntimeError("AVX2 profile build flags are absent")
        libs = out / "libs"
        libs.mkdir(exist_ok=True)
        receipt["ggml_libraries"] = {}
        for stem in ("libggml", "libggml-base", "libggml-cpu"):
            built = _library(build, stem)
            copied = libs / f"{stem}.so.0"
            shutil.copy2(built, copied)
            receipt["ggml_libraries"][stem] = {"build_path": str(built), "sha256": sha256(copied)}
        for source_file, dest in (("LICENSE", "LICENSE"),
                                  ("THIRD_PARTY_NOTICES.txt", "THIRD_PARTY_NOTICES.txt"),
                                  ("ggml/LICENSE", "GGML_LICENSE")):
            shutil.copy2(_required_file(source / source_file), out / dest)
        probe_source = _required_file(source / "tools/ci-heavy/csm_rope_probe.cpp")
        probe = out / "csm-rope-probe"
        argv = ["c++", "-std=c++17", "-O2", "-DCSM_GENERATED_ROPE", "-DGGML_MAX_NAME=128",
                "-I", source / "src", "-I", source / "ggml/include", probe_source,
                libs / "libggml.so.0", libs / "libggml-base.so.0", libs / "libggml-cpu.so.0",
                "-Wl,-rpath,$ORIGIN/libs", "-ldl", "-pthread", "-o", probe]
        receipt["compiler_argv"] = list(map(str, argv))
        receipt["stages"]["before_compile"] = ab.snapshot()
        ab.require_headroom(receipt["stages"]["before_compile"])
        save()
        rc = _command(argv, source, out, "compile", timeout=120)
        receipt["stages"]["compile_rc"] = rc
        save()
        if rc != 0:
            raise RuntimeError(f"probe compilation failed, rc={rc}")
        receipt["probe_sha256"] = sha256(probe)
        receipt["probe_source_sha256"] = sha256(probe_source)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("LD_", "GGML_", "CRISPASR_"))}
        env["CRISPASR_CSM_ROPE_DEVICE"] = "cpu"
        rc = _command(["ldd", probe], source, out, "ldd", timeout=30, env=env)
        receipt["stages"]["ldd_rc"] = rc
        save()
        if rc != 0 or "not found" in (out / "ldd.stdout.log").read_text():
            raise RuntimeError("probe shared library resolution failed")
        linked = set()
        for line in (out / "ldd.stdout.log").read_text().splitlines():
            if "libggml" in line and "=>" in line:
                loaded = Path(line.split("=>", 1)[1].strip().split()[0]).resolve()
                if loaded.parent != libs.resolve():
                    raise RuntimeError(f"probe loaded GGML from outside bundle: {loaded}")
                linked.add(loaded.name.split(".so", 1)[0])
        if linked != {"libggml", "libggml-base", "libggml-cpu"}:
            raise RuntimeError(f"probe GGML linkage incomplete: {sorted(linked)}")
        for mode in MODES:
            for case in check.CASES:
                cfg = manifest["cases"][case]["configuration"]
                scaling = cfg["rope_scaling"]
                shape = manifest["cases"][case]["files"]["positions"]["shape"]
                dest = out / mode / case
                dest.mkdir(parents=True, exist_ok=True)
                argv = [probe, reference / case, dest, cfg["head_dim"], cfg["num_attention_heads"],
                        cfg["num_key_value_heads"], shape[0], scaling["original_max_position_embeddings"],
                        cfg["rope_theta"], mode]
                if mode == "generated":
                    argv.extend((scaling["factor"], scaling["low_freq_factor"], scaling["high_freq_factor"]))
                label = f"{mode}-{case}"
                receipt["stages"][f"before_{label}"] = ab.snapshot()
                ab.require_headroom(receipt["stages"][f"before_{label}"])
                save()
                rc = _command(argv, source, out, label, timeout=90, env=env)
                reported = None
                if rc == 0:
                    reported = json.loads((out / f"{label}.stdout.log").read_text())
                    if reported.get("mode") != mode or reported.get("fallback") is not False or \
                            "CPU" not in reported.get("backend", "").upper():
                        raise RuntimeError(f"{label}: probe did not report its requested CPU backend")
                outputs = {}
                for key in ("q_rotated", "k_rotated", "freq_factors"):
                    path = dest / f"{key}.bin"
                    if path.is_file():
                        outputs[key] = sha256(path)
                receipt["cases"][label] = {"rc": rc, "reported": reported, "outputs_sha256": outputs}
                save()
                if rc != 0 or len(outputs) != 3:
                    raise RuntimeError(f"{label}: probe failed or omitted outputs")
            if mode in ("original", "generated"):
                result = check.compare(reference, out / mode)
                receipt["stages"][f"{mode}_comparison"] = result
                save()
                if not result["pass"]:
                    raise RuntimeError(f"{mode}: original RoPE operator gates failed")
            else:
                receipt["stages"]["unscaled_comparison"] = _unscaled_check(reference, out / mode, manifest)
                save()
        receipt["complete"] = True
        receipt["pass"] = True
        save()
        return receipt
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=("native", "avx2"))
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        run(args.source, args.build, args.profile, args.reference, args.out)
    except Exception as exc:
        print(f"CSM RoPE component check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
