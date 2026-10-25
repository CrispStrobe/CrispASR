#!/usr/bin/env python3
"""One same-host CSM/ASR CPU-profile factorial; unchanged nightly speech gates.

Heavy CPU workflow dependencies: numpy huggingface_hub.
Native and explicit AVX2 builds use the same immutable source and models.
Two seeded syntheses per build are cross-scored by BOTH ASR executables.
This separates synthesizer variation from judge variation; it is not a speed
benchmark or proof that a passing host reproduces a previously failing host.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import wave

import nightly_error_ab as ab

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    "native": ["-DGGML_NATIVE=ON"],
    "avx2": ["-DGGML_NATIVE=OFF", "-DGGML_AVX=ON", "-DGGML_AVX2=ON",
             "-DGGML_FMA=ON", "-DGGML_F16C=ON", "-DGGML_AVX_VNNI=OFF",
             "-DGGML_AVX512=OFF", "-DGGML_AVX512_VBMI=OFF",
             "-DGGML_AVX512_VNNI=OFF", "-DGGML_AVX512_BF16=OFF",
             "-DGGML_AMX_TILE=OFF", "-DGGML_AMX_INT8=OFF", "-DGGML_AMX_BF16=OFF"],
}
SAMPLE_MAGIC = b"CSMSMP01"
CODE_MAGIC = b"CSMCOD01"
TOKEN_MAGIC = b"CSMTXT01"
MAX_SAMPLES = 2048
AVX2_BASELINE_PCM = "fff79f9c32a62304fb2348866f4a497436c67b359ea482909eaf56b58b6ded96"
NATIVE_BASELINE_PCM = "1ac415193d618b9ae6f8eb01b67efd95e02ac44430fbbee67414d6c584f289d1"
ROPE_REFERENCE_RUN = "40213576399"
ROPE_REFERENCE_RECEIPT = "ea74ac15c0a1ffa9f7e90e5bd2290ef0cc7f32ed85102d20af174dee5da80273"


def original_rope_speech(source, out, executables, paths, csm, asr, receipt, save, default_control=False):
    """Separate opt-in candidate cells; never relabel the legacy eight cells."""
    control = {"environment_delta": {"CRISPASR_CSM_ORIGINAL_ROPE": "1"},
               "scope": "experimental original-frequency correction, not default promotion",
               "syntheses": {}, "scores": {}, "prefill": {}}
    receipt["original_rope_control"] = control
    control["default_vs_explicit_on"] = default_control
    save()
    for profile, cli in executables.items():
        for repeat in range(2):
            label = f"rope-{profile}-{repeat}"
            output = out / f"{label}.wav"
            prefix = out / f"{label}-trace"
            argv = [cli, "-m", paths["csm"], "--tts", csm["tts_phrase"], "--tts-output", output,
                    "--seed", str(csm.get("tts_seed", 1234)), "--no-prints"]
            if csm.get("voice_preset"):
                argv += ["--voice", csm["voice_preset"]]
            argv += csm.get("tts_extra_args", [])
            if default_control and repeat == 0:
                os.environ.pop("CRISPASR_CSM_ORIGINAL_ROPE", None)
            else:
                os.environ["CRISPASR_CSM_ORIGINAL_ROPE"] = "1"
            if repeat == 1:
                os.environ["CRISPASR_CSM_TRACE_PREFIX"] = str(prefix)
            try:
                rc = ab.command(argv, source, out, f"{label}-synth", 300)
            finally:
                os.environ.pop("CRISPASR_CSM_ORIGINAL_ROPE", None)
                os.environ.pop("CRISPASR_CSM_TRACE_PREFIX", None)
            info = pcm_info(output)
            control["syntheses"][label] = {"profile": profile, "repeat": repeat, "rc": rc, "wav": info}
            save()
            for judge, asr_cli in executables.items():
                score_label = f"{label}-judge-{judge}"
                result = {"tts_profile": profile, "asr_profile": judge, "repeat": repeat, "gate": False}
                if (rc == 0 and info.get("bytes", 0) >= 1000 and "pcm_sha256" in info
                        and info.get("channels") == 1 and info.get("rate") == 24000
                        and info.get("sample_width") == 2):
                    score_rc = ab.command([asr_cli, "--backend", asr["backend_id"], "-m", paths["asr"],
                                           "-f", output, "--no-prints"], source, out, score_label, 180)
                    stdout = (out / f"{score_label}.stdout.log").read_text(errors="replace")
                    text = ab.regression.strip_ai_disclaimer(next(
                        (x.strip() for x in reversed(stdout.splitlines()) if x.strip()), ""))
                    wer = ab.regression.compute_transcript_metrics(csm["tts_phrase"], text)[1]
                    result.update(rc=score_rc, transcript=text, wer=wer, wer_max=csm["wer_max"],
                                  gate=bool(score_rc == 0 and wer <= csm["wer_max"]))
                control["scores"][score_label] = result
                save()
        first = control["syntheses"][f"rope-{profile}-0"]["wav"].get("pcm_sha256")
        second = control["syntheses"][f"rope-{profile}-1"]["wav"].get("pcm_sha256")
        if not first or first != second:
            raise RuntimeError(f"original RoPE {profile}: traced/untraced PCM differs or missing")
        trace = read_profile_trace(out / f"rope-{profile}-1-trace")
        control["syntheses"][f"rope-{profile}-1"]["trace"] = trace["files"]
        stage = out / f"prefill-rope-{profile}"
        os.environ["CRISPASR_CSM_ORIGINAL_ROPE"] = "1"
        try:
            rc = ab.command([sys.executable, ROOT / "tools/ci-heavy/csm_prefill_capture.py",
                             "--library", cli.parent.parent / "src/libcrispasr.so", "--model", paths["csm"],
                             "--trace-prefix", out / f"rope-{profile}-1-trace", "--out", stage,
                             "--text", csm["tts_phrase"]], source, out, f"prefill-rope-{profile}", 300)
        finally:
            os.environ.pop("CRISPASR_CSM_ORIGINAL_ROPE", None)
        child_path = stage / "receipt.json"
        child = json.loads(child_path.read_text()) if child_path.exists() else {}
        control["prefill"][profile] = {"rc": rc, "complete": bool(rc == 0 and child.get("complete")),
                                          "receipt_sha256": ab.sha256(child_path) if child_path.exists() else None}
        save()
        if not control["prefill"][profile]["complete"]:
            raise RuntimeError(f"original RoPE {profile}: prefill differs from sampled path")
    control["speech_acceptance"] = len(control["scores"]) == 8 and all(
        cell["gate"] for cell in control["scores"].values())
    control["traced_untraced_pcm_identical"] = True
    save()


def read_trace_bytes(path, magic, limit):
    size = path.stat().st_size
    if size < 8 or size > limit:
        raise ValueError(f"{path.name}: invalid trace size {size} (limit {limit})")
    data = path.read_bytes()
    if data[:8] != magic:
        raise ValueError(f"{path.name}: incorrect trace magic")
    return data


def read_sampling(path):
    import numpy as np

    data = read_trace_bytes(path, SAMPLE_MAGIC, 256 * 1024**2)
    records = []
    offset = 8
    while offset < len(data):
        if len(records) >= MAX_SAMPLES or len(data) - offset < 32:
            raise ValueError(f"{path.name}: too many or truncated sampling records")
        vocab, topk, temp, token, before, after = struct.unpack_from("<IIfiQQ", data, offset)
        offset += 32
        if not (1 <= vocab <= 65536 and 0 <= topk <= vocab and
                math.isfinite(temp) and temp >= 0 and 0 <= token < vocab):
            raise ValueError(f"{path.name}: invalid sampling record {len(records)}")
        width = vocab * 4
        if len(data) - offset < width:
            raise ValueError(f"{path.name}: truncated logits at record {len(records)}")
        logits = np.frombuffer(data, dtype="<f4", count=vocab, offset=offset).copy()
        if np.isnan(logits).any():
            raise ValueError(f"{path.name}: NaN logits at record {len(records)}")
        records.append({"vocab": vocab, "topk": topk, "temperature": temp,
                        "token": token, "rng_before": before, "rng_after": after,
                        "logits": logits})
        offset += width
    if not records:
        raise ValueError(f"{path.name}: no sampling records")
    return records


def read_codes(path):
    import numpy as np

    data = read_trace_bytes(path, CODE_MAGIC, 32 * 1024**2)
    if len(data) < 16:
        raise ValueError(f"{path.name}: missing code dimensions")
    frames, codebooks = struct.unpack_from("<II", data, 8)
    if not (1 <= frames <= 100000 and codebooks == 32 and len(data) == 16 + frames * codebooks * 4):
        raise ValueError(f"{path.name}: invalid or truncated code grid")
    return np.frombuffer(data, dtype="<i4", count=frames * codebooks, offset=16).reshape(frames, codebooks).copy()


def read_tokens(path):
    import numpy as np

    data = read_trace_bytes(path, TOKEN_MAGIC, 4 * 1024**2)
    if len(data) < 12:
        raise ValueError(f"{path.name}: missing token count")
    count = struct.unpack_from("<I", data, 8)[0]
    if not (1 <= count <= 100000 and len(data) == 12 + count * 4):
        raise ValueError(f"{path.name}: invalid or truncated prompt tokens")
    return np.frombuffer(data, dtype="<i4", count=count, offset=12).copy()


def read_profile_trace(prefix):
    import numpy as np

    samples_path = Path(str(prefix) + ".sampling.bin")
    codes_path = Path(str(prefix) + ".codes.bin")
    tokens_path = Path(str(prefix) + ".tokens.bin")
    samples = read_sampling(samples_path)
    codes = read_codes(codes_path)
    tokens = read_tokens(tokens_path)
    vocab = samples[0]["vocab"]
    if any(record["vocab"] != vocab for record in samples):
        raise ValueError(f"{prefix}: sampling vocabulary width changed")
    if np.any(codes < 0) or np.any(codes >= vocab) or np.any(tokens < 0):
        raise ValueError(f"{prefix}: code or prompt token IDs are out of range")
    shared = min(len(samples), codes.size)
    required = 32 * min(len(codes), 64)
    if shared < required or not np.array_equal(
            np.asarray([record["token"] for record in samples[:shared]], dtype=np.int32),
            codes.reshape(-1)[:shared]):
        raise ValueError(f"{prefix}: sampled tokens disagree with retained code prefix")
    return {"samples": samples, "codes": codes, "tokens": tokens,
            "files": {kind: {"sha256": ab.sha256(path), "bytes": path.stat().st_size}
                      for kind, path in (("sampling", samples_path), ("codes", codes_path),
                                         ("tokens", tokens_path))}}


def compare_traces(native, avx2, left_label="native", right_label="avx2"):
    import numpy as np

    if not np.array_equal(native["tokens"], avx2["tokens"]):
        raise ValueError(f"{left_label} and {right_label} prompt token IDs differ")
    a, b = native["samples"], avx2["samples"]
    shared = min(len(a), len(b))
    chosen = next((i for i in range(shared) if a[i]["token"] != b[i]["token"]), None)
    if chosen is None and len(a) != len(b):
        chosen = shared
    # At the first differing choice, both logits were computed from the same
    # preceding chosen-token prefix. Later records have different inputs.
    equal_input_records = shared if chosen is None else min(shared, chosen + 1)
    first_raw = None
    first_rng = None
    max_abs = norm_native_sq = norm_avx2_sq = norm_diff_sq = 0.0
    finite_count = nonfinite_mismatch = 0
    for i in range(equal_input_records):
        left, right = a[i], b[i]
        if left["vocab"] != right["vocab"] or left["topk"] != right["topk"] or left["temperature"] != right["temperature"]:
            raise ValueError(f"sampling configuration differs at record {i}")
        if first_raw is None and left["logits"].tobytes() != right["logits"].tobytes():
            first_raw = i
        if first_rng is None and left["rng_before"] != right["rng_before"]:
            first_rng = i
        lvals = left["logits"].astype(np.float64)
        rvals = right["logits"].astype(np.float64)
        finite = np.isfinite(lvals) & np.isfinite(rvals)
        nonfinite_mismatch += int(np.count_nonzero(~finite & (lvals != rvals)))
        lv, rv = lvals[finite], rvals[finite]
        delta = lv - rv
        finite_count += int(delta.size)
        if delta.size:
            max_abs = max(max_abs, float(np.max(np.abs(delta))))
            norm_native_sq += float(np.dot(lv, lv))
            norm_avx2_sq += float(np.dot(rv, rv))
            norm_diff_sq += float(np.dot(delta, delta))
    result = {f"{left_label}_records": len(a), f"{right_label}_records": len(b),
              "equal_input_records": equal_input_records,
              "first_chosen_token_divergence": chosen,
              "first_raw_logit_divergence": first_raw,
              "first_raw_logit_divergence_before_choice":
                  first_raw if first_raw is not None and (chosen is None or first_raw < chosen) else None,
              "first_rng_before_divergence": first_rng,
              "finite_values_compared": finite_count,
              "nonfinite_mismatch": nonfinite_mismatch,
              "max_abs": max_abs, f"{left_label}_norm": math.sqrt(norm_native_sq),
              f"{right_label}_norm": math.sqrt(norm_avx2_sq),
              f"relative_error_vs_{right_label}": math.sqrt(norm_diff_sq) / max(math.sqrt(norm_avx2_sq), 1e-30),
              "prompt_token_count": len(native["tokens"]),
              f"{left_label}_code_frames": len(native["codes"]),
              f"{right_label}_code_frames": len(avx2["codes"])}
    if chosen is not None and chosen < shared:
        result["chosen_tokens"] = {left_label: a[chosen]["token"], right_label: b[chosen]["token"],
                                   "frame": chosen // 32, "codebook": chosen % 32}
    return result


def pcm_info(path):
    info = ab.wav_info(path)
    if info.get("exists") and "wav_error" not in info:
        with wave.open(str(path), "rb") as stream:
            pcm = stream.readframes(stream.getnframes())
            info["sample_width"] = stream.getsampwidth()
        info["pcm_sha256"] = hashlib.sha256(pcm).hexdigest()
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="f469d6dd86e42d42d0358fbef6fa75cd0c2aba29")
    parser.add_argument("--trace", action="store_true", help="capture and compare the sampled CSM path")
    parser.add_argument("--allow-avx2-host", action="store_true",
                        help="with --trace, collect a portable baseline if AVX512 is unavailable; not a native fix")
    parser.add_argument("--attention-control", action="store_true",
                        help="with --trace, repeat native synthesis using the existing eager-F32 attention switch")
    parser.add_argument("--prefill-control", action="store_true",
                        help="with --trace, require existing-API prefill dumps to match actual sampled logits")
    parser.add_argument("--original-rope-control", action="store_true",
                        help="with --trace, first gate original RoPE components, then test opt-in speech cells")
    parser.add_argument("--default-rope-control", action="store_true",
                        help="with original control, compare bare corrected defaults to explicit ON; legacy cells force OFF")
    args = parser.parse_args()
    if args.allow_avx2_host and not args.trace:
        parser.error("--allow-avx2-host requires --trace")
    if args.prefill_control and not args.trace:
        parser.error("--prefill-control requires --trace")
    if args.attention_control and not args.trace:
        parser.error("--attention-control requires --trace")
    if args.original_rope_control and not args.trace:
        parser.error("--original-rope-control requires --trace")
    if args.default_rope_control and not args.original_rope_control:
        parser.error("--default-rope-control requires --original-rope-control")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source):
        parser.error("--source must be an immutable full Git commit")
    if args.trace and not any(arg == "--source" or arg.startswith("--source=") for arg in sys.argv[1:]):
        parser.error("--trace requires an explicit --source commit containing the trace writer")
    out = Path(os.environ["HEAVY_OUT"]).resolve()
    scratch = Path(os.environ["HEAVY_SCRATCH"]).resolve() / "csm-cpu-profile-ab"
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    receipt = {"complete": False, "acceptance": False, "source": args.source,
               "worker_source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "scope": __doc__, "profiles": {}, "syntheses": {}, "scores": {},
               "initial_resources": ab.snapshot(), "image": os.environ.get("ImageOS"),
               "image_version": os.environ.get("ImageVersion")}
    if args.trace:
        receipt["trace"] = {"requested": True, "source": args.source,
                            "format": "CSMSMP01/CSMCOD01/CSMTXT01 little-endian"}

    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    try:
        if args.trace:
            cpuinfo = Path("/proc/cpuinfo").read_bytes()
            (out / "cpuinfo.txt").write_bytes(cpuinfo)
            has_avx512f = bool(re.search(rb"\bavx512f\b", cpuinfo))
            features = (b"avx2", b"fma", b"f16c", b"bmi2", b"sse4_2") if args.allow_avx2_host else (b"avx512f",)
            eligible = sys.byteorder == "little" and all(
                re.search(rb"\b" + feature + rb"\b", cpuinfo) for feature in features)
            receipt["trace"]["host_avx512f"] = has_avx512f
            receipt["trace"]["purpose"] = (
                "native-profile comparison; known failure remains open until separately qualified"
                if has_avx512f else "portable trace baseline only; cannot clear the known native-profile failure")
            receipt["trace"]["cpuinfo_sha256"] = hashlib.sha256(cpuinfo).hexdigest()
            save()
            if not eligible:
                receipt["error"] = "trace prerequisite failed: little-endian host must expose " + ", ".join(
                    feature.decode() for feature in features)
                save()
                return 2
        ab.require_headroom(receipt["initial_resources"])
        shutil.copy2("/proc/cpuinfo", out / "cpuinfo.txt")
        ab.required(["lscpu", "--json"], ROOT, out, "cpu", 60)
        ab.required(["bash", ROOT / "tools/ci-apt.sh", "update"], ROOT, out, "apt-update", 600)
        ab.required(["bash", ROOT / "tools/ci-apt.sh", "install", "-y", "libopenblas-dev",
                     "libespeak-ng-dev", "espeak-ng", "pkg-config", "ccache"], ROOT, out, "apt-install", 600)
        source = scratch / "source"
        ab.required(["git", "fetch", "--depth=1", "origin", args.source], ROOT, out, "fetch", 300)
        ab.required(["git", "worktree", "add", "--detach", source, args.source], ROOT, out, "worktree", 300)
        ab.required(["git", "submodule", "update", "--init", "--recursive"], source, out, "submodules", 600)
        if args.trace:
            trace_source = (source / "src/csm_tts.cpp").read_text()
            markers = ("CRISPASR_CSM_TRACE_PREFIX", "CSMSMP01", "CSMCOD01", "CSMTXT01")
            if not all(marker in trace_source for marker in markers):
                raise RuntimeError("--source does not contain the required CSM trace writer")
        if args.original_rope_control:
            if "CRISPASR_CSM_ORIGINAL_ROPE" not in trace_source:
                raise RuntimeError("source does not contain the experimental original RoPE control")
            if args.default_rope_control and "c->original_rope = !rope_mode" not in trace_source:
                raise RuntimeError("source does not enable original RoPE by default")
            reference = out / "original-rope-reference"
            ab.required(["gh", "run", "download", ROPE_REFERENCE_RUN, "--repo", "CrispStrobe/CrispASR",
                         "--name", f"heavy-{ROPE_REFERENCE_RUN}", "--dir", reference], ROOT, out,
                        "original-rope-reference-download", 300)
            if ab.sha256(reference / "receipt.json") != ROPE_REFERENCE_RECEIPT:
                raise RuntimeError("original reference receipt hash differs from qualified producer")
            receipt["original_rope_reference"] = {"run": ROPE_REFERENCE_RUN,
                                                   "receipt_sha256": ROPE_REFERENCE_RECEIPT}
            save()
        manifest_path = source / "tests/regression/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        _, csm, asr = ab.selected(manifest)
        receipt["pins"] = {"csm": csm, "asr": asr}
        receipt["manifest_sha256"] = ab.sha256(manifest_path)
        assert csm["gguf"]["revision"] == "5abf936b9d17a839fca69e09baf31e0ea5068f97"
        assert csm["tts_phrase"] == "The quick brown fox jumps over the lazy dog"
        assert csm["wer_max"] == 0.15 and csm.get("tts_seed", 1234) == 1234
        models = scratch / "models"
        models.mkdir(exist_ok=True)
        paths = {name: ab.regression.hf_download(spec["repo"], spec["file"], spec["revision"], models)
                 for name, spec in (("csm", csm["gguf"]), ("asr", asr["gguf"]))}
        receipt["model_sha256"] = {name: ab.sha256(path) for name, path in paths.items()}
        save()
        executables = {}
        for profile, flags in PROFILES.items():
            case = out / profile
            case.mkdir(exist_ok=True)
            state = ab.snapshot()
            ab.require_headroom(state)
            if int(state["memory"]["MemAvailable"].split()[0]) < 8 * 1024**2:
                raise RuntimeError("Hosted build requires 8 GiB available RAM")
            build = scratch / f"build-{profile}"
            configuration = ["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                             "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                             "-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                             "-DCRISPASR_BUILD_TESTS=OFF", "-DCRISPASR_BUILD_EXAMPLES=ON",
                             "-DCRISPASR_BUILD_SERVER=OFF", *flags]
            receipt["profiles"][profile] = {"flags": configuration, "resources_before_build": state}
            save()
            ab.required(["cmake", "-S", source, "-B", build, *configuration], source, case, "configure", 600)
            ab.required(["cmake", "--build", build, "--target", "crispasr-cli", "-j4"], source, case, "build", 1800)
            for filename in ("CMakeCache.txt", "compile_commands.json"):
                shutil.copy2(build / filename, case / filename)
            commands = json.loads((build / "compile_commands.json").read_text())
            cpu_commands = [item["command"] for item in commands if "/ggml-cpu/" in item["file"]]
            assert cpu_commands, "No actual ggml CPU compiler commands"
            if profile == "native":
                assert any("-march=native" in command for command in cpu_commands)
            else:
                assert all("-march=native" not in command and "-mavx512" not in command for command in cpu_commands)
                assert any("-mavx2" in command and "-mfma" in command and "-mf16c" in command for command in cpu_commands)
            cli = build / "bin/crispasr"
            assert cli.is_file()
            executables[profile] = cli
            receipt["profiles"][profile]["cli_sha256"] = ab.sha256(cli)
            receipt["profiles"][profile]["cpu_compiler_commands_verified"] = len(cpu_commands)
            ab.required(["ldd", cli], source, case, "libraries", 60)
            if args.original_rope_control:
                import csm_rope_component_run
                component = csm_rope_component_run.run(source, build, profile, reference, case / "rope-component")
                receipt["profiles"][profile]["rope_component"] = component
            save()
        # Preserve the actual nightly synth argv, including default marking.
        # Whole WAV hashes can change with provenance metadata; compare PCM too.
        os.environ.pop("CRISPASR_CSM_TRACE_PREFIX", None)
        os.environ.pop("CRISPASR_CSM_ORIGINAL_ROPE", None)
        if args.default_rope_control:
            os.environ["CRISPASR_CSM_ORIGINAL_ROPE"] = "0"
            receipt["legacy_rope_environment"] = "CRISPASR_CSM_ORIGINAL_ROPE=0"
            save()
        if args.attention_control:
            os.environ.pop("CRISPASR_CORE_ATTN_EAGER_F32", None)
        for profile, cli in executables.items():
            for repeat in range(2):
                label = f"{profile}-{repeat}"
                output = out / f"{label}.wav"
                argv = [cli, "-m", paths["csm"], "--tts", csm["tts_phrase"], "--tts-output", output,
                        "--seed", str(csm.get("tts_seed", 1234)), "--no-prints"]
                if csm.get("voice_preset"):
                    argv += ["--voice", csm["voice_preset"]]
                argv += csm.get("tts_extra_args", [])
                trace_prefix = out / f"{label}-trace" if args.trace and repeat == 1 else None
                if trace_prefix is not None:
                    os.environ["CRISPASR_CSM_TRACE_PREFIX"] = str(trace_prefix)
                try:
                    rc = ab.command(argv, source, out, f"{label}-synth", 300)
                finally:
                    os.environ.pop("CRISPASR_CSM_TRACE_PREFIX", None)
                info = pcm_info(output)
                receipt["syntheses"][label] = {"profile": profile, "repeat": repeat, "rc": rc, "wav": info}
                save()
                for judge, asr_cli in executables.items():
                    score_label = f"{label}-judge-{judge}"
                    result = {"tts_profile": profile, "asr_profile": judge, "repeat": repeat, "gate": False}
                    if (rc == 0 and info.get("bytes", 0) >= 1000 and "pcm_sha256" in info
                            and info.get("channels") == 1 and info.get("rate") == 24000
                            and info.get("sample_width") == 2):
                        score_rc = ab.command([asr_cli, "--backend", asr["backend_id"], "-m", paths["asr"],
                                               "-f", output, "--no-prints"], source, out, score_label, 180)
                        stdout = (out / f"{score_label}.stdout.log").read_text(errors="replace")
                        text = ab.regression.strip_ai_disclaimer(next((x.strip() for x in reversed(stdout.splitlines()) if x.strip()), ""))
                        wer = ab.regression.compute_transcript_metrics(csm["tts_phrase"], text)[1]
                        result.update(rc=score_rc, transcript=text, wer=wer, wer_max=csm["wer_max"],
                                      gate=bool(score_rc == 0 and wer <= csm["wer_max"]))
                    receipt["scores"][score_label] = result
                    save()
        speech_acceptance = len(receipt["scores"]) == 8 and all(x["gate"] for x in receipt["scores"].values())
        receipt["speech_acceptance"] = speech_acceptance
        receipt["identical_repeat_pcm"] = {
            p: bool(receipt["syntheses"][f"{p}-0"]["wav"].get("pcm_sha256")) and
            receipt["syntheses"][f"{p}-0"]["wav"].get("pcm_sha256") ==
            receipt["syntheses"][f"{p}-1"]["wav"].get("pcm_sha256") for p in PROFILES}
        if args.trace:
            avx2_pcm = receipt["syntheses"]["avx2-1"]["wav"].get("pcm_sha256")
            native_pcm = receipt["syntheses"]["native-1"]["wav"].get("pcm_sha256")
            receipt["trace"]["pcm_baselines"] = {
                "avx2_expected": AVX2_BASELINE_PCM, "avx2_actual": avx2_pcm,
                "avx2_match": avx2_pcm == AVX2_BASELINE_PCM,
                "native_prior": NATIVE_BASELINE_PCM, "native_actual": native_pcm,
                "native_match_prior": native_pcm == NATIVE_BASELINE_PCM,
            }
            save()
            for profile in PROFILES:
                if not receipt["identical_repeat_pcm"][profile]:
                    raise RuntimeError(f"{profile}: traced repeat changed synthesized PCM")
            if avx2_pcm != AVX2_BASELINE_PCM:
                raise RuntimeError("AVX2 traced PCM differs from accepted same-host baseline")
            parsed = {profile: read_profile_trace(out / f"{profile}-1-trace") for profile in PROFILES}
            receipt["trace"]["files"] = {profile: data["files"] for profile, data in parsed.items()}
            receipt["trace"]["comparison"] = compare_traces(parsed["native"], parsed["avx2"])
            save()
            if args.attention_control:
                control = {"environment_delta": {"CRISPASR_CORE_ATTN_EAGER_F32":
                                                  {"baseline": None, "control": "1"}},
                           "source_profile": "native", "gating": "diagnostic only; original eight cells decide job result",
                           "syntheses": {}, "scores": {}}
                receipt["attention_control"] = control
                save()
                native_cli = executables["native"]
                for repeat in range(2):
                    label = f"attention-native-{repeat}"
                    output = out / f"{label}.wav"
                    argv = [native_cli, "-m", paths["csm"], "--tts", csm["tts_phrase"], "--tts-output", output,
                            "--seed", str(csm.get("tts_seed", 1234)), "--no-prints"]
                    if csm.get("voice_preset"):
                        argv += ["--voice", csm["voice_preset"]]
                    argv += csm.get("tts_extra_args", [])
                    os.environ["CRISPASR_CORE_ATTN_EAGER_F32"] = "1"
                    if repeat == 1:
                        os.environ["CRISPASR_CSM_TRACE_PREFIX"] = str(out / f"{label}-trace")
                    try:
                        rc = ab.command(argv, source, out, f"{label}-synth", 300)
                    finally:
                        os.environ.pop("CRISPASR_CSM_TRACE_PREFIX", None)
                        os.environ.pop("CRISPASR_CORE_ATTN_EAGER_F32", None)
                    info = pcm_info(output)
                    control["syntheses"][label] = {"repeat": repeat, "rc": rc, "wav": info}
                    save()
                    for judge, asr_cli in executables.items():
                        score_label = f"{label}-judge-{judge}"
                        result = {"tts_profile": "native-eager-f32", "asr_profile": judge,
                                  "repeat": repeat, "gate": False}
                        if (rc == 0 and info.get("bytes", 0) >= 1000 and "pcm_sha256" in info
                                and info.get("channels") == 1 and info.get("rate") == 24000
                                and info.get("sample_width") == 2):
                            score_rc = ab.command([asr_cli, "--backend", asr["backend_id"], "-m", paths["asr"],
                                                   "-f", output, "--no-prints"], source, out, score_label, 180)
                            stdout = (out / f"{score_label}.stdout.log").read_text(errors="replace")
                            text = ab.regression.strip_ai_disclaimer(
                                next((x.strip() for x in reversed(stdout.splitlines()) if x.strip()), ""))
                            wer = ab.regression.compute_transcript_metrics(csm["tts_phrase"], text)[1]
                            result.update(rc=score_rc, transcript=text, wer=wer, wer_max=csm["wer_max"],
                                          gate=bool(score_rc == 0 and wer <= csm["wer_max"]))
                        control["scores"][score_label] = result
                        save()
                first_pcm = control["syntheses"]["attention-native-0"]["wav"].get("pcm_sha256")
                second_pcm = control["syntheses"]["attention-native-1"]["wav"].get("pcm_sha256")
                control["traced_untraced_pcm_identical"] = bool(first_pcm) and first_pcm == second_pcm
                control["speech_acceptance"] = len(control["scores"]) == 4 and all(
                    cell["gate"] for cell in control["scores"].values())
                save()
                if not control["traced_untraced_pcm_identical"]:
                    raise RuntimeError("eager-F32 attention control: tracing changed synthesized PCM")
                eager = read_profile_trace(out / "attention-native-1-trace")
                control["trace"] = {"files": eager["files"],
                                    "native_vs_eager": compare_traces(
                                        parsed["native"], eager, "native", "eager_f32"),
                                    "eager_vs_avx2": compare_traces(
                                        eager, parsed["avx2"], "eager_f32", "avx2")}
                save()
            if args.prefill_control:
                cases = [(profile, cli, out / f"{profile}-1-trace", None)
                         for profile, cli in executables.items()]
                if args.attention_control:
                    cases.append(("eager-f32", executables["native"],
                                  out / "attention-native-1-trace", "1"))
                controls = {}
                receipt["prefill_controls"] = controls
                for profile, cli, prefix, attention in cases:
                    library = cli.parent.parent / "src/libcrispasr.so"
                    stage_out = out / f"prefill-{profile}"
                    if attention is not None:
                        os.environ["CRISPASR_CORE_ATTN_EAGER_F32"] = attention
                    try:
                        rc = ab.command([sys.executable, ROOT / "tools/ci-heavy/csm_prefill_capture.py",
                                         "--library", library, "--model", paths["csm"],
                                         "--trace-prefix", prefix, "--out", stage_out,
                                         "--text", csm["tts_phrase"]], source, out,
                                        f"prefill-{profile}-control", 300)
                    finally:
                        os.environ.pop("CRISPASR_CORE_ATTN_EAGER_F32", None)
                    stage_receipt = stage_out / "receipt.json"
                    child = json.loads(stage_receipt.read_text()) if stage_receipt.is_file() else {}
                    controls[profile] = {"rc": rc, "complete": bool(rc == 0 and child.get("complete")),
                                         "receipt_sha256": ab.sha256(stage_receipt) if stage_receipt.is_file() else None}
                    save()
                if not all(control["complete"] for control in controls.values()):
                    raise RuntimeError("prefill dump did not match actual sampled logits; raw evidence retained")
        if args.original_rope_control:
            original_rope_speech(source, out, executables, paths, csm, asr, receipt, save, args.default_rope_control)
        receipt["complete"] = True
        receipt["acceptance"] = speech_acceptance
        save()
        summary = (
            "# CSM CPU profile factorial\n\n" +
            ((receipt["trace"]["purpose"] + ".\n\n") if args.trace else "") +
            "| Synth/repeat | ASR | WER | Gate |\n|---|---|---|---|\n" +
            "".join(f"| {x['tts_profile']}/{x['repeat']} | {x['asr_profile']} | {x.get('wer', 'missing')} | {x['gate']} |\n"
                    for x in receipt["scores"].values()) +
            "\nExisting seed, model pins and 0.15 WER gate are unchanged. Full logs, audio and compiler/CPU evidence retained.\n")
        if args.attention_control:
            summary += ("\n## Eager-F32 attention control (diagnostic only)\n\n" +
                        "| Synth/repeat | ASR | WER | Gate |\n|---|---|---|---|\n" +
                        "".join(f"| {x['tts_profile']}/{x['repeat']} | {x['asr_profile']} | "
                                f"{x.get('wer', 'missing')} | {x['gate']} |\n"
                                for x in receipt["attention_control"]["scores"].values()))
        if args.original_rope_control:
            summary += ("\n## Original RoPE candidate (opt-in, default unchanged)\n\n" +
                        "| Synth/repeat | ASR | WER | Gate |\n|---|---|---|---|\n" +
                        "".join(f"| {x['tts_profile']}/{x['repeat']} | {x['asr_profile']} | "
                                f"{x.get('wer', 'missing')} | {x['gate']} |\n"
                                for x in receipt["original_rope_control"]["scores"].values()))
        (out / "summary.md").write_text(summary)
        return 0 if receipt["acceptance"] else 1
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    sys.exit(main())
