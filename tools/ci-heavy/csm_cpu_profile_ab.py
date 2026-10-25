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
import os
from pathlib import Path
import re
import shutil
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
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source):
        parser.error("--source must be an immutable full Git commit")
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

    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    try:
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
            save()
        # Preserve the actual nightly synth argv, including default marking.
        # Whole WAV hashes can change with provenance metadata; compare PCM too.
        for profile, cli in executables.items():
            for repeat in range(2):
                label = f"{profile}-{repeat}"
                output = out / f"{label}.wav"
                argv = [cli, "-m", paths["csm"], "--tts", csm["tts_phrase"], "--tts-output", output,
                        "--seed", str(csm.get("tts_seed", 1234)), "--no-prints"]
                if csm.get("voice_preset"):
                    argv += ["--voice", csm["voice_preset"]]
                argv += csm.get("tts_extra_args", [])
                rc = ab.command(argv, source, out, f"{label}-synth", 300)
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
        receipt["complete"] = True
        receipt["acceptance"] = len(receipt["scores"]) == 8 and all(x["gate"] for x in receipt["scores"].values())
        receipt["identical_repeat_pcm"] = {
            p: bool(receipt["syntheses"][f"{p}-0"]["wav"].get("pcm_sha256")) and
            receipt["syntheses"][f"{p}-0"]["wav"].get("pcm_sha256") ==
            receipt["syntheses"][f"{p}-1"]["wav"].get("pcm_sha256") for p in PROFILES}
        save()
        (out / "summary.md").write_text(
            "# CSM CPU profile factorial\n\n" +
            "| Synth/repeat | ASR | WER | Gate |\n|---|---|---|---|\n" +
            "".join(f"| {x['tts_profile']}/{x['repeat']} | {x['asr_profile']} | {x.get('wer', 'missing')} | {x['gate']} |\n"
                    for x in receipt["scores"].values()) +
            "\nExisting seed, model pins and 0.15 WER gate are unchanged. Full logs, audio and compiler/CPU evidence retained.\n")
        return 0 if receipt["acceptance"] else 1
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    sys.exit(main())
