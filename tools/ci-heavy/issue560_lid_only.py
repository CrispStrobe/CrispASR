#!/usr/bin/env python3
"""Hosted current CLI: standalone LID must bypass ASR model resolution."""

import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import tarfile
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[2]
MODELS = [
    dict(
        repo="ggerganov/whisper.cpp",
        revision="5359861c739e955e79d9a303bcbc70fb988958b1",
        file="ggml-tiny.bin",
        sha256="be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
    ),
    dict(
        repo="cstr/silero-lid-lang95-GGUF",
        revision="78fb93154156272b2448183dfbf62b648ece4e52",
        file="silero-lid-lang95-f32.gguf",
        sha256="ed161165bb6fabae895f8837761b57bde50ede273ec83f6ad37ba982ce3e54da",
    ),
]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("Hosted-only build and inference")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    out = Path(os.environ["HEAVY_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "issue560"
    scratch.mkdir(parents=True, exist_ok=True)
    build = scratch / "build"
    cache = scratch / "cache"
    cache.mkdir(exist_ok=True)
    receipt = dict(
        schema=1,
        complete=False,
        acceptance=False,
        source=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        models=MODELS,
        resource_checks=[],
        phases={},
    )

    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def guard(label):
        memory = dict(
            line.split(":", 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
        )
        row = dict(
            phase=label,
            available_bytes=int(memory["MemAvailable"].split()[0]) * 1024,
            load=list(os.getloadavg()),
            cpus=os.cpu_count() or 1,
            scratch_free_bytes=shutil.disk_usage(scratch).free,
        )
        receipt["resource_checks"].append(row)
        save()
        if (
            row["available_bytes"] < 3 * 1024**3
            or row["load"][0] > 2 * row["cpus"]
            or row["scratch_free_bytes"] < 2 * 1024**3
        ):
            raise RuntimeError("Insufficient hosted resources before " + label)

    def run(command, label, success=True, timeout=1800):
        guard(label)
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(
                list(map(str, command)),
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1"),
            )
        receipt["phases"][label] = dict(returncode=result.returncode)
        save()
        if (result.returncode == 0) != success:
            raise RuntimeError("Unexpected status for " + label)
        return (out / (label + ".log")).read_text()

    try:
        run(
            [
                "cmake",
                "-S",
                ROOT,
                "-B",
                build,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DGGML_NATIVE=OFF",
                "-DGGML_CUDA=OFF",
                "-DGGML_METAL=OFF",
                "-DGGML_VULKAN=OFF",
                "-DGGML_BLAS=OFF",
                "-DCRISPASR_BUILD_SERVER=OFF",
                "-DCRISPASR_BUILD_TESTS=OFF",
            ],
            "configure",
        )
        run(
            ["cmake", "--build", build, "--target", "crispasr-cli", "-j2"],
            "build-current-cli",
            timeout=7200,
        )
        cli = build / "bin/crispasr"
        receipt["cli_sha256"] = sha(cli)
        receipt["sample_sha256"] = sha(ROOT / "samples/jfk.wav")
        receipt["ko_sample_sha256"] = sha(ROOT / "samples/ko-369.wav")
        guard("checkpoint")
        with tarfile.open(
            out / "cli-validation-checkpoint.tar.gz", "w:gz", compresslevel=1
        ) as archive:
            archive.add(cli, arcname="build/bin/crispasr")
            for file in build.rglob("*.so*"):
                if file.is_file() or file.is_symlink():
                    archive.add(
                        file,
                        arcname="build/" + str(file.relative_to(build)),
                        recursive=False,
                    )
            for name in ["LICENSE", "THIRD_PARTY_NOTICES.txt"]:
                archive.add(ROOT / name, arcname=name)
        receipt["checkpoint_sha256"] = sha(out / "cli-validation-checkpoint.tar.gz")
        save()
        for model in MODELS:
            guard("download-" + model["file"])
            url = (
                "https://huggingface.co/"
                + model["repo"]
                + "/resolve/"
                + model["revision"]
                + "/"
                + model["file"]
            )
            path = cache / model["file"]
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                path.open("wb") as dest,
            ):
                shutil.copyfileobj(response, dest)
            if sha(path) != model["sha256"]:
                raise ValueError("LID model checksum mismatch")
        common = [cli, "--no-gpu", "-np", "-t", "2", "--cache-dir", cache]
        audio = ROOT / "samples/jfk.wav"

        def detect(flags, label, files=None, expected_language="en"):
            log = run(
                common
                + ["--detect-language"]
                + flags
                + sum([["-f", str(p)] for p in (files or [audio])], []),
                label,
                timeout=300,
            )
            rows = re.findall(
                r"^lang=(\S+)\tconfidence=([0-9.]+)\tsource=([^\t\n]+)\tfile=",
                log,
                re.M,
            )
            if len(rows) != len(files or [audio]) or any(
                lang != expected_language for lang, _, _ in rows
            ):
                raise RuntimeError("Standalone LID result missing or unexpected")
            receipt.setdefault("results", {})[label] = [
                dict(language=lang, confidence=float(conf), source=source)
                for lang, conf, source in rows
            ]
            save()
            return log, rows

        # Deliberately invalid ASR/HF choices must remain unused in this action.
        forbidden = scratch / "ASR-MUST-NOT-LOAD.gguf"
        bypass = [
            "-m",
            forbidden,
            "--hf-repo",
            "invalid-issue560/unused-asr",
            "--hf-file",
            "never.gguf",
        ]
        detect(bypass, "default-cached-bypasses-asr")
        detect(bypass + ["--lid-model", "auto"], "auto-model-bypasses-asr")
        detect(
            bypass
            + ["--lid-backend", "whisper", "--lid-model", cache / "ggml-tiny.bin"],
            "explicit-whisper-bypasses-asr",
        )
        _, rows = detect(
            bypass
            + [
                "--backend",
                "nemotron",
                "--lid-backend",
                "silero",
                "--lid-model",
                cache / "silero-lid-lang95-f32.gguf",
            ],
            "explicit-silero-bypasses-asr",
            files=[audio, audio],
        )
        if any(source != "silero" for _, _, source in rows):
            raise RuntimeError("Dedicated provider silently fell back")

        # Real long-clip controls for the recording-wide opt-in. All fixtures
        # use the two public repository clips; no model or ASR weights are
        # embedded. A Korean prefix followed by more English speech forces the
        # first-window and full-recording decisions to differ, proving that
        # the flag reaches late windows rather than merely changing output.
        def read_pcm(path):
            with wave.open(str(path), "rb") as wav:
                if (
                    wav.getnchannels() != 1
                    or wav.getsampwidth() != 2
                    or wav.getframerate() != 16000
                ):
                    raise RuntimeError("Unexpected public clip format")
                return wav.readframes(wav.getnframes())

        def write_pcm(name, data):
            path = scratch / name
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(data)
            receipt.setdefault("long_clip_fixtures", {})[name] = dict(
                sha256=sha(path), duration_samples=len(data) // 2
            )
            save()
            return path

        english = read_pcm(audio)
        korean = read_pcm(ROOT / "samples/ko-369.wav")
        silence_16s = bytes(16 * 16000 * 2)
        late = write_pcm("late-after-silence.wav", silence_16s + english)
        mixed = write_pcm("korean-then-english.wav", korean * 5 + english * 2)
        # 30 s of silence plus 4 s of real speech reaches the short-tail
        # context path; the classifier sees context but the vote weighs 4 s.
        tail = write_pcm("short-final-speech.wav", bytes(30 * 16000 * 2) + english[:4 * 16000 * 2])
        silent = write_pcm("silence.wav", bytes(18 * 16000 * 2))
        empty = write_pcm("empty.wav", b"")
        for provider, filename in [
            ("whisper", "ggml-tiny.bin"),
            ("silero", "silero-lid-lang95-f32.gguf"),
        ]:
            selected = bypass + ["--lid-backend", provider, "--lid-model", cache / filename]
            full = selected + ["--lid-full-recording"]
            def check_route(label, flags, *, files, expected_language):
                log, rows = detect(flags, label, files=files, expected_language=expected_language)
                # The winning language can collect windows from both Silero
                # and its Whisper fallback, in which case the caller reports
                # the combined source as "mixed".
                allowed = {provider} if provider == "whisper" else {"silero", "whisper", "mixed"}
                if any(source not in allowed for _, _, source in rows):
                    raise RuntimeError("Unexpected full-recording LID source")
                if "invalid model data (bad magic)" in log:
                    raise RuntimeError("Fallback used an incompatible provider model")
                fallback = "falling back to whisper-tiny LID" in log
                if provider == "silero" and fallback:
                    loaded = re.findall(
                        r"whisper_init_from_file_with_params_no_state: loading model from '([^']+)'",
                        log,
                    )
                    if not loaded or any(Path(path).name != "ggml-tiny.bin" for path in loaded):
                        raise RuntimeError("Silero fallback did not load Whisper's own checkpoint")
                receipt.setdefault("long_clip_routes", {})[label] = dict(
                    requested=provider, result_source=rows[0][2], fallback_observed=fallback
                )
                save()
                return rows[0][0]

            check_route(provider + "-late-after-silence", full, files=[late], expected_language="en")
            first = check_route(provider + "-first-window-control", selected,
                                files=[mixed], expected_language="ko")
            whole = check_route(provider + "-late-majority", full,
                                files=[mixed], expected_language="en")
            if first == whole:
                raise RuntimeError("Full-recording option did not change language coverage")
            check_route(provider + "-short-final-speech", full,
                        files=[tail], expected_language="en")
            for name, path in [("silence", silent), ("empty", empty)]:
                log = run(
                    common + ["--detect-language"] + full + ["-f", path],
                    provider + "-" + name,
                    success=False,
                    timeout=60,
                )
                if re.search(r"^lang=", log, re.M):
                    raise RuntimeError("No-speech input emitted a LID result")
        for backend in ["self", "probe", "off", "none"]:
            log = run(
                common + ["--detect-language", "--lid-backend", backend, "-f", audio],
                "unsupported-" + backend,
                success=False,
                timeout=60,
            )
            if re.search(r"^lang=", log, re.M):
                raise RuntimeError("Unsupported action emitted a result")
        run(common + ["--detect-language"], "missing-audio", success=False, timeout=60)
        run(
            common + ["--detect-language", "-f", scratch / "missing.wav"],
            "unreadable-audio",
            success=False,
            timeout=60,
        )
        # The normal transcription caller must still decode instead of taking
        # the new action route when only automatic language selection is set.
        log = run(
            common + ["-m", cache / "ggml-tiny.bin", "-l", "auto", "-f", audio],
            "ordinary-auto-transcription",
            timeout=300,
        )
        if "ask not" not in log.lower() or re.search(r"^lang=", log, re.M):
            raise RuntimeError("Normal automatic-language transcription regressed")
        receipt.update(complete=True, acceptance=True)
        save()
    except Exception:
        receipt["error"] = "Current CLI acceptance failed; inspect retained phase logs"
        save()
        raise


if __name__ == "__main__":
    main()
