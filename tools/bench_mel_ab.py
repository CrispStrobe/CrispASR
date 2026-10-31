#!/usr/bin/env python3
"""Paired process A/B benchmark with required transcript gates on every run.

Each invocation starts a fresh process and includes startup/model loading.
Discarded runs prime filesystem caches; they are not in-process warm inference.
Supply an explicit JSON case list; missing models or empty suites are errors.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_transcript(text):
    """Ignore line wrapping/whitespace only; retain every word and punctuation."""
    return " ".join(text.split())


def validate_cases(cases, base_dir):
    if not isinstance(cases, list) or not cases:
        raise ValueError("The case manifest must contain a nonempty JSON list")
    validated = []
    seen = set()
    for index, source in enumerate(cases):
        if not isinstance(source, dict):
            raise ValueError(f"Case {index} must be an object")
        case = dict(source)
        name = case.get("name", f"case-{index}")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in seen:
            raise ValueError("Case names must be unique, nonempty filename-safe identifiers")
        seen.add(name)
        case["name"] = name
        for key in ("model", "audio"):
            value = case.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name}: {key} is required")
            path = Path(value)
            if not path.is_absolute():
                path = Path(base_dir) / path
            if not path.is_file():
                raise FileNotFoundError(f"{name}: required {key} not found: {path}")
            case[key] = str(path.resolve())
            case[f"{key}_sha256"] = sha256(path)
        expected = case.get("expected_transcript")
        if not isinstance(expected, str) or not normalize_transcript(expected):
            raise ValueError(f"{name}: nonempty expected_transcript is required")
        case["expected_transcript"] = normalize_transcript(expected)
        args = case.get("extra_args", [])
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ValueError(f"{name}: extra_args must be a list of strings")
        reserved = {"-m", "--model", "-f", "--file", "-t", "--threads", "-of", "--output-file", "-otxt", "--output-txt",
                    "--hf-repo", "-hfr", "--hf-file", "-hff"}
        if any(arg.split("=", 1)[0] in reserved for arg in args):
            raise ValueError(f"{name}: extra_args cannot override benchmark inputs, threads, or output")
        case["extra_args"] = args
        validated.append(case)
    return validated


def run_decode(binary, case, env, output_prefix, threads, timeout, runner=subprocess.run):
    command = [str(binary), "-m", case["model"], "-f", case["audio"],
               "-l", case.get("language", "auto"), "-t", str(threads),
               "-otxt", "-of", str(output_prefix)] + case["extra_args"]
    # A real transcript file, never the last logging line on stdout, is proof
    # of work. Each process receives a fresh path so stale output cannot pass.
    transcript_path = Path(str(output_prefix) + ".txt")
    transcript_path.unlink(missing_ok=True)
    started = time.perf_counter()
    result = runner(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, env=env, timeout=timeout, check=False)
    elapsed_ms = (time.perf_counter() - started) * 1000
    Path(str(output_prefix) + ".stdout.log").write_text(result.stdout, encoding="utf-8")
    Path(str(output_prefix) + ".stderr.log").write_text(result.stderr, encoding="utf-8")
    transcript = normalize_transcript(transcript_path.read_text(encoding="utf-8")) if transcript_path.is_file() else ""
    match = re.search(r"transcribed ([\d.]+)s audio in ([\d.]+)s \(([\d.]+)x realtime\)", result.stdout + "\n" + result.stderr)
    return {
        "process_ms": elapsed_ms,
        "returncode": result.returncode,
        "transcript": transcript,
        "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        "quality_pass": result.returncode == 0 and bool(transcript) and transcript == case["expected_transcript"],
        "reported_audio_s": float(match.group(1)) if match else None,
        "reported_inference_s": float(match.group(2)) if match else None,
        "reported_speed_x": float(match.group(3)) if match else None,
    }


def collect(binary, cases, output, *, base_dir=".", threads=4, warmups=1,
            runs=3, timeout=600, serial_env="CRISPASR_MEL_SERIAL", runner=subprocess.run):
    output = Path(output)
    # Separate runs must not silently replace a previous accepted receipt.
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 1, "complete": False, "quality_pass": False,
               "timing_scope": "fresh-process wall time including startup and model load",
               "warmup_scope": "discarded fresh processes; filesystem-cache priming only",
               "transcript_comparison": "exact after whitespace normalization",
               "platform": platform.platform(), "cpu_count": os.cpu_count(),
               "threads": threads, "serial_env": serial_env, "cases": []}

    def save():
        (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        if not isinstance(threads, int) or threads < 1 or not isinstance(runs, int) or runs < 1 or not isinstance(warmups, int) or warmups < 0 or timeout <= 0:
            raise ValueError("Require positive threads/runs/timeout and nonnegative warmups")
        if not re.fullmatch(r"CRISPASR_[A-Z0-9_]+_SERIAL", serial_env):
            raise ValueError("serial_env must name a CRISPASR_*_SERIAL control")
        binary = Path(binary).resolve()
        if not binary.is_file():
            raise FileNotFoundError(f"Required binary not found: {binary}")
        receipt["binary_sha256"] = sha256(binary)
        cases = validate_cases(cases, base_dir)
        for case in cases:
            record = {key: value for key, value in case.items() if key not in ("model", "audio")}
            record["model_file"] = Path(case["model"]).name
            record["audio_file"] = Path(case["audio"]).name
            record["events"] = []
            receipt["cases"].append(record)
            for phase, count in (("discarded", warmups), ("measured", runs)):
                for index in range(count):
                    order = ("A", "B") if index % 2 == 0 else ("B", "A")
                    for arm in order:
                        env = os.environ.copy()
                        # Some production controls (including MEL_SERIAL) test
                        # presence, not the value. "0" would still select the
                        # serial path. Explicitly clear an inherited control in
                        # B so both presence-only and value-based flags work.
                        if arm == "A":
                            env[serial_env] = "1"
                        else:
                            env.pop(serial_env, None)
                        label = f"{case['name']}-{phase}-{index}-{arm}"
                        event = {"phase": phase, "round": index, "arm": arm,
                                 "control": "1" if arm == "A" else "unset", "log_prefix": label}
                        record["events"].append(event)
                        save()
                        event.update(run_decode(binary, case, env, output / label, threads, timeout, runner))
                        save()
                        if not event["quality_pass"]:
                            raise RuntimeError(f"{label}: command/transcript quality gate failed; see retained logs and receipt")
            by_arm = {arm: [e["process_ms"] for e in record["events"] if e["phase"] == "measured" and e["arm"] == arm] for arm in ("A", "B")}
            record["median_process_ms"] = {arm: statistics.median(values) for arm, values in by_arm.items()}
            record["process_speedup_A_over_B"] = record["median_process_ms"]["A"] / record["median_process_ms"]["B"]
            # Report CLI instrumentation separately. Its boundary is defined by
            # the backend and is not labelled as warm/steady-state inference.
            record["median_reported_inference_s"] = {}
            for arm in ("A", "B"):
                events = [e for e in record["events"] if e["phase"] == "measured" and e["arm"] == arm]
                values = [e["reported_inference_s"] for e in events]
                record["median_reported_inference_s"][arm] = statistics.median(values) if all(v is not None for v in values) else None
            record["quality_pass"] = True
        receipt["complete"] = True
        receipt["quality_pass"] = True
        save()
        return receipt
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/bin/crispasr"))
    parser.add_argument("--cases", required=True, type=Path, help="JSON list; model/audio paths relative to this manifest")
    parser.add_argument("--output", required=True, type=Path, help="New directory for every run and its receipt")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--serial-env", default="CRISPASR_MEL_SERIAL")
    args = parser.parse_args()
    try:
        cases = json.loads(args.cases.read_text(encoding="utf-8"))
        receipt = collect(args.binary, cases, args.output, base_dir=args.cases.parent,
                          threads=args.threads, warmups=args.warmups, runs=args.runs,
                          timeout=args.timeout, serial_env=args.serial_env)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Benchmark failed: {exc}\n")
    print(f"All {len(receipt['cases'])} cases passed every transcript gate; receipt: {args.output / 'receipt.json'}")


if __name__ == "__main__":
    main()
