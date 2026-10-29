#!/usr/bin/env python3
"""Private, hosted CSM graph-step profile with off/on output controls."""

import atexit
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import statistics
import subprocess
import time
import wave

SOURCE = "3997582388c61febe5cfd4b0d741414c0da5233f"
OUT = Path("/kaggle/working")
WORK = Path("/kaggle/temp/csm-native-steps")
NATIVE = (
    "cstr/csm-1b-GGUF",
    "5abf936b9d17a839fca69e09baf31e0ea5068f97",
    "csm-1b-q4_k.gguf",
    "1123f9e8ff2d9212700d1fface749b8ec4f9765569bb996667a15a9a9832ee13",
)
ASR = (
    "cstr/parakeet-tdt-0.6b-v3-GGUF",
    "815cc1bcb5cfb92f9cd0bbd22c7bc2f6e2d4ec13",
    "parakeet-tdt-0.6b-v3-q4_k.gguf",
    "1a60f6e53e5781240dde6e69a47a47a8a71995a3a106517b009225afcc514457",
)
PHRASE = "The quick brown fox jumps over the lazy dog"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def guard():
    mem = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    row = {
        "available_bytes": int(mem["MemAvailable"].split()[0]) * 1024,
        "scratch_free_bytes": shutil.disk_usage(WORK).free,
        "gpu_free_mib": int(gpu),
        "load": list(os.getloadavg()),
        "cpus": os.cpu_count() or 1,
    }
    if (
        row["available_bytes"] < 8 * 1024**3
        or row["scratch_free_bytes"] < 10 * 1024**3
        or row["gpu_free_mib"] < 10 * 1024
        or row["load"][0] > 2 * row["cpus"]
    ):
        raise RuntimeError("Insufficient worker resources")
    return row


def wer(text):
    left, right = [re.findall(r"[a-z0-9]+", s.lower()) for s in (PHRASE, text)]
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        nxt = [i]
        for j, b in enumerate(right, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = nxt
    return row[-1] / len(left)


def parse_profile(text):
    head = re.findall(
        r"csm_step_profile: complete=(\d+) attempts=(\d+) accepted=(\d+) eos=(\d+) codebooks=(\d+)",
        text,
    )
    rows = re.findall(
        r"csm_step_profile: (backbone_prefill|backbone_step|depth_prefill|depth_step) calls=(\d+) total_us=(\d+)",
        text,
    )
    if len(head) != 1 or len(rows) != 4 or len({r[0] for r in rows}) != 4:
        raise ValueError("Missing or duplicate native profile")
    complete, attempts, accepted, eos, codebooks = map(int, head[0])
    stages = {
        name: {"calls": int(count), "total_us": int(us)} for name, count, us in rows
    }
    counts = {k: v["calls"] for k, v in stages.items()}
    expected = {
        "backbone_prefill": 1,
        "backbone_step": attempts - 1,
        "depth_prefill": attempts,
        "depth_step": (codebooks - 2) * attempts,
    }
    if (
        complete != 1
        or attempts < 1
        or accepted < 1
        or eos > 1
        or codebooks != 32
        or accepted + eos != attempts
        or counts != expected
        or any(v["total_us"] <= 0 for v in stages.values())
    ):
        raise ValueError("Incomplete or inconsistent native profile")
    return {
        "attempts": attempts,
        "accepted": accepted,
        "eos": eos,
        "codebooks": codebooks,
        "stages": stages,
    }


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    OUT.mkdir(exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    def preserve_private_evidence():
        destination = OUT / "private-evidence"
        destination.mkdir(exist_ok=True)
        for pattern in ("*.wav", "*.log"):
            for path in WORK.glob(pattern):
                if path.is_file():
                    shutil.copy2(path, destination / path.name)

    atexit.register(preserve_private_evidence)
    receipt = {
        "complete": False,
        "acceptance": False,
        "source": SOURCE,
        "runs": [],
        "scope": "Native Q4_K only; synchronized graph compute times omit frame setup and sampling. Profiling on/off alternates independent processes; output identity and GPU execution required.",
    }

    def save():
        (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    guard()
    hardware = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=name,compute_cap",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if "T4" not in hardware or "7.5" not in hardware:
        raise RuntimeError("Expected a T4")
    receipt["hardware"] = hardware
    save()
    repo = WORK / "source"
    subprocess.run(["git", "init", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "remote",
            "add",
            "origin",
            "https://github.com/CrispStrobe/CrispASR.git",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "fetch", "--depth=1", "origin", SOURCE],
        check=True,
        timeout=300,
    )
    subprocess.run(["git", "-C", str(repo), "checkout", "FETCH_HEAD"], check=True)
    if (
        subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        != SOURCE
    ):
        raise RuntimeError("Wrong source")
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=repo,
        check=True,
        timeout=300,
    )
    build = WORK / "build"
    guard()
    subprocess.run(
        [
            "cmake",
            "-S",
            str(repo),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA=ON",
            "-DGGML_CUDA_NO_VMM=ON",
            "-DGGML_NATIVE=OFF",
            "-DCMAKE_CUDA_ARCHITECTURES=75",
            "-DCRISPASR_BUILD_TESTS=OFF",
            "-DCRISPASR_BUILD_SERVER=OFF",
        ],
        check=True,
        timeout=300,
    )
    guard()
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "crispasr-cli", "-j2"],
        check=True,
        timeout=5400,
    )
    binary = build / "bin/crispasr"
    if not binary.is_file():
        raise RuntimeError("Compiled CLI missing")
    receipt["cli_sha256"] = sha(binary)
    save()
    from huggingface_hub import hf_hub_download

    models = []
    for spec in (NATIVE, ASR):
        guard()
        path = Path(
            hf_hub_download(
                spec[0],
                spec[2],
                revision=spec[1],
                local_dir=WORK / "models" / spec[0].split("/")[-1],
            )
        )
        if sha(path) != spec[3]:
            raise RuntimeError("Model checksum mismatch")
        models.append(path)
    env = dict(
        os.environ,
        GGML_BACKEND_PATH=str(binary.parent),
        LD_LIBRARY_PATH=str(binary.parent)
        + os.pathsep
        + os.environ.get("LD_LIBRARY_PATH", ""),
    )
    env.pop("CRISPASR_CSM_STEP_PROFILE", None)
    for phase, count in (("warmup", 1), ("measured", 2)):
        for pair in range(count):
            for enabled in (False, True) if pair % 2 == 0 else (True, False):
                guard()
                label = f"{phase}-{pair}-" + ("on" if enabled else "off")
                wav, log = WORK / (label + ".wav"), WORK / (label + ".log")
                cmd = [
                    str(binary),
                    "--backend",
                    "csm",
                    "-m",
                    str(models[0]),
                    "--gpu-backend",
                    "cuda",
                    "--tts",
                    PHRASE,
                    "--voice",
                    "default",
                    "--seed",
                    "1234",
                    "--temperature",
                    "0.9",
                    "--tts-output",
                    str(wav),
                    "-t",
                    "4",
                ]
                run_env = dict(env)
                if enabled:
                    run_env["CRISPASR_CSM_STEP_PROFILE"] = "1"
                start = time.monotonic()
                with log.open("wb") as stream:
                    proc = subprocess.Popen(
                        cmd, stdout=stream, stderr=subprocess.STDOUT, env=run_env
                    )
                    peak = 0
                    peak_sm = 0
                    while proc.poll() is None:
                        apps = subprocess.run(
                            [
                                "nvidia-smi",
                                "--id=0",
                                "--query-compute-apps=pid,used_gpu_memory",
                                "--format=csv,noheader,nounits",
                            ],
                            capture_output=True,
                            text=True,
                        )
                        for line in apps.stdout.splitlines():
                            pid, _, used = line.partition(",")
                            if pid.strip() == str(proc.pid) and used.strip().isdigit():
                                peak = max(peak, int(used.strip()))
                        activity = subprocess.run(
                            ["nvidia-smi", "pmon", "-c", "1", "-s", "u"],
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        for line in activity.stdout.splitlines():
                            fields = line.split()
                            if (
                                len(fields) >= 4
                                and fields[1] == str(proc.pid)
                                and fields[3].isdigit()
                            ):
                                peak_sm = max(peak_sm, int(fields[3]))
                        if time.monotonic() - start > 900:
                            proc.kill()
                            proc.wait()
                            raise TimeoutError(label)
                        time.sleep(0.2)
                text = log.read_text(errors="replace")
                if proc.returncode or peak < 512 or peak_sm <= 0 or not wav.is_file():
                    raise RuntimeError(
                        "Native run failed or GPU residency missing: " + label
                    )
                if not enabled and "csm_step_profile:" in text:
                    raise RuntimeError("Profile emitted while disabled")
                with wave.open(str(wav), "rb") as audio:
                    if (
                        audio.getnchannels() != 1
                        or audio.getsampwidth() != 2
                        or audio.getframerate() != 24000
                        or not 0.5 < audio.getnframes() / audio.getframerate() < 10.0
                    ):
                        raise RuntimeError("Invalid synthesized audio")
                row = {
                    "label": label,
                    "profile_enabled": enabled,
                    "process_seconds": time.monotonic() - start,
                    "peak_gpu_mib": peak,
                    "peak_gpu_sm_percent": peak_sm,
                    "wav_sha256": sha(wav),
                    "log_sha256": sha(log),
                }
                if enabled:
                    row["profile"] = parse_profile(text)
                guard()
                judge = subprocess.run(
                    [
                        str(binary),
                        "--backend",
                        "parakeet",
                        "-m",
                        str(models[1]),
                        "-f",
                        str(wav),
                        "--gpu-backend",
                        "cuda",
                        "--no-prints",
                        "-t",
                        "4",
                    ],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=300,
                )
                (WORK / (label + "-asr.log")).write_text(
                    judge.stdout + "\n" + judge.stderr
                )
                lines = [
                    line.strip()
                    for line in (judge.stdout + "\n" + judge.stderr).splitlines()
                    if line.strip()
                ]
                if judge.returncode or not lines:
                    raise RuntimeError("ASR judge failed")
                # CLI transcript is on stdout; logs on stderr must not contaminate it.
                out_lines = [
                    line.strip() for line in judge.stdout.splitlines() if line.strip()
                ]
                if not out_lines:
                    raise RuntimeError("ASR transcript missing")
                transcript = re.sub(
                    r"(?i)^this audio was generated by artificial intelligence[. ]*",
                    "",
                    out_lines[-1],
                )
                row["asr_wer"] = wer(transcript)
                if row["asr_wer"] > 0.15:
                    raise RuntimeError("ASR quality gate failed")
                receipt["runs"].append(row)
                save()
    if len({r["wav_sha256"] for r in receipt["runs"]}) != 1:
        raise RuntimeError("Profile on/off output mismatch")
    receipt["median_process_seconds"] = {
        str(enabled).lower(): statistics.median(
            r["process_seconds"]
            for r in receipt["runs"]
            if r["label"].startswith("measured") and r["profile_enabled"] == enabled
        )
        for enabled in (False, True)
    }
    receipt["complete"] = receipt["acceptance"] = True
    save()


if __name__ == "__main__":
    main()
