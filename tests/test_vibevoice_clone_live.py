#!/usr/bin/env python3
"""Opt-in base VibeVoice cloning regression for configured local 1.5B/7B packs.

See tests/regression/README.md. No downloads, builds or dependency installation.
Exit 77 means configuration is unset; invalid configured inputs fail.
"""
from __future__ import annotations

import argparse
import array
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent / "regression"))
from run_one import compute_transcript_metrics, speaker_conditioning_gate, strip_ai_disclaimer

PREFIX = "CRISPASR_VIBEVOICE_CLONE_"
TEXT = "The library opens early tomorrow so please bring the blue notebook."


def configuration(size, environ, cli):
    names = {"model": PREFIX + ("15B_MODEL" if size == "1.5b" else "7B_MODEL"),
             "reference": PREFIX + "REFERENCE", "recognizer": PREFIX + "ASR_MODEL"}
    paths = {}
    missing = []
    # A typo in a configured path must fail even if another input is unset.
    for key, name in names.items():
        value = environ.get(name, "")
        if not value:
            missing.append(name)
        elif not Path(value).is_file():
            raise ValueError(f"{name} is not a file: {value}")
        else:
            paths[key] = Path(value).resolve()
    thresholds = {}
    for key, default in (("WER_MAX", "0.25"), ("MIN_SPEAKER_DELTA", "0.02")):
        thresholds[key] = float(environ.get(PREFIX + key, default))
        if not math.isfinite(thresholds[key]) or thresholds[key] < 0:
            raise ValueError(f"{PREFIX + key} must be finite and nonnegative")
    text = environ.get(PREFIX + "TEXT", TEXT)
    if not text.strip():
        raise ValueError(f"{PREFIX}TEXT must not be empty")
    explicit_cli = environ.get("CRISPASR_BIN")
    if explicit_cli and (not Path(explicit_cli).is_file() or not os.access(explicit_cli, os.X_OK)):
        raise ValueError(f"CRISPASR_BIN is not executable: {explicit_cli}")
    if missing:
        return None, missing
    binary = Path(explicit_cli or cli).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"crispasr is not executable: {binary}")
    return dict(paths, cli=binary, wer_max=thresholds["WER_MAX"],
                speaker_delta=thresholds["MIN_SPEAKER_DELTA"],
                recognizer_backend=environ.get(PREFIX + "ASR_BACKEND", "parakeet"),
                text=text), []


def usable_wav(path):
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"expected PCM16 WAV: {path}")
        seconds = wav.getnframes() / wav.getframerate()
        samples = array.array("h", wav.readframes(wav.getnframes()))
        if sys.byteorder != "little":
            samples.byteswap()
    rms = math.sqrt(sum(x * x for x in samples) / len(samples)) / 32768 if samples else 0
    if seconds < 0.4 or rms < 1e-4:
        raise ValueError(f"short or silent WAV: {path} ({seconds:.3f}s, RMS {rms:.6f})")
    return {"seconds": seconds, "rms": rms}


def run(argv, output, name, timeout):
    env = {key: value for key, value in os.environ.items()
           if key not in ("VIBEVOICE_VOICE_AUDIO", "CRISPASR_VIBEVOICE_VOICE_AUDIO")}
    result = subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=timeout, env=env)
    (output / f"{name}.stdout.txt").write_text(result.stdout)
    (output / f"{name}.stderr.txt").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit {result.returncode}; see {output}")


def evaluate(config, output, timeout=1800):
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").unlink(missing_ok=True)
    usable_wav(config["reference"])
    results = {}
    for arm in ("baseline", "clone"):
        wav = output / f"{arm}.wav"
        wav.unlink(missing_ok=True)
        argv = [config["cli"], "--backend", "vibevoice-1.5b", "-m", config["model"],
                "--tts", config["text"], "--tts-output", wav, "--seed", "42"]
        if arm == "clone":
            argv += ["--voice", config["reference"], "--i-have-rights"]
        run(argv, output, arm, timeout)
        results[arm] = usable_wav(wav)
        stem = output / f"{arm}-asr"
        transcript_file = stem.with_suffix(".txt")
        transcript_file.unlink(missing_ok=True)
        run([config["cli"], "--backend", config["recognizer_backend"],
             "-m", config["recognizer"], "-f", wav, "-l", "en", "-nt",
             "-of", stem, "-otxt"], output, f"{arm}-asr", timeout)
        transcript = strip_ai_disclaimer(transcript_file.read_text())
        _, wer = compute_transcript_metrics(config["text"], transcript)
        results[arm].update(transcript=transcript, wer=wer,
                            content_pass=bool(transcript.strip()) and wer <= config["wer_max"])
    speaker_pass, metrics = speaker_conditioning_gate(
        config["reference"], output / "clone.wav", output / "baseline.wav", config["speaker_delta"])
    results.update(speaker=metrics, speaker_pass=speaker_pass,
                   passed=speaker_pass and all(results[a]["content_pass"] for a in ("baseline", "clone")))
    (output / "result.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main(argv=None, environ=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="1.5b", choices=("1.5b", "7b"))
    parser.add_argument("--crispasr", default="build/bin/crispasr")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--keep-files", action="store_true",
                        help="retain generated files in a temporary directory when --output-dir is unset")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    try:
        config, missing = configuration(args.size, os.environ if environ is None else environ, args.crispasr)
        if missing:
            print("SKIP: unset " + ", ".join(missing))
            return 77
        if args.output_dir:
            result = evaluate(config, args.output_dir.resolve(), args.timeout)
        elif args.keep_files:
            output = Path(tempfile.mkdtemp(prefix="vibevoice-clone-"))
            print(f"Keeping files in {output}")
            result = evaluate(config, output, args.timeout)
        else:
            with tempfile.TemporaryDirectory(prefix="vibevoice-clone-") as temporary:
                result = evaluate(config, Path(temporary), args.timeout)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, wave.Error) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
