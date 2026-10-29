#!/usr/bin/env python3
"""Native Nemotron CUDA timing/count diagnostic with instrumentation off/on.

Runs only the accepted native F16 GGUF. Process timings include loading;
decoder instrumentation reports native frame and emission counts without
claiming equivalence to steps in the original implementation.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import statistics
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request

VERSION = "nemotron-step-diagnostic-v1"
ROOT = Path("/kaggle/working")
WORK = Path("/kaggle/temp/nemotron-step-diagnostic")
INPUT = ("https://raw.githubusercontent.com/CrispStrobe/CrispASR/"
         "cc114d2b4a5a9413738f390093e9131576743cb6/samples/jfk.wav",
         "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e")
RUNTIME = ("https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.47/"
           "crispasr-linux-x86_64-cuda.tar.gz",
           "90e23cffb44f4a80d929bf88cf3abb2d526ec4c0b95d88a82d1a8bf33135722d")
NATIVE = ("cstr/nemotron-3.5-asr-streaming-GGUF",
          "bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2",
          "nemotron-3.5-asr-streaming-0.6b-f16.gguf",
          "14cbc98df495d8ab452c6a4ac367136d94ab5da55e73afaa96ba4f749dd95d09")
ORIGINAL = ("nvidia/nemotron-3.5-asr-streaming-0.6b",
            "ea30d66debe3740a08b573244286791d423d6b3e",
            "9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174")
REFERENCE = "and so my fellow americans ask not what your country can do for you ask what you can do for your country"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def words(value):
    return re.findall(r"[a-z0-9]+", value.lower())


def wer(reference, hypothesis):
    left, right = words(reference), words(hypothesis)
    row = list(range(len(right) + 1))
    for i, token in enumerate(left, 1):
        nxt = [i]
        for j, other in enumerate(right, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (token != other)))
        row = nxt
    return row[-1] / len(left)


def headroom():
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    state = {"available_bytes": int(memory["MemAvailable"].split()[0]) * 1024,
             "load": list(os.getloadavg()), "cpus": os.cpu_count() or 1,
             "scratch_free_bytes": shutil.disk_usage(WORK).free}
    if (state["available_bytes"] < 3 * 1024**3 or state["scratch_free_bytes"] < 6 * 1024**3
            or state["load"][0] > 2 * state["cpus"]):
        raise RuntimeError("Insufficient worker resources")
    return state


def download(url, expected, target):
    with urllib.request.urlopen(url, timeout=180) as source, target.open("wb") as dest:
        shutil.copyfileobj(source, dest, 1024 * 1024)
    if sha(target) != expected:
        raise ValueError("Pinned input hash mismatch: " + target.name)
    return target


def run_call(command, env, label, receipt):
    state = headroom()
    log = ROOT / (label + ".log")
    started = time.monotonic()
    with log.open("wb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        observed = [False]
        stop = threading.Event()

        def watch():
            while not stop.is_set():
                probe = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
                observed[0] |= str(process.pid) in probe.stdout.split()
                stop.wait(0.1)

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise TimeoutError(label)
        finally:
            stop.set()
            watcher.join()
    row = {"label": label, "seconds": time.monotonic() - started, "returncode": process.returncode,
           "gpu_pid_observed": observed[0], "resources": state, "log_sha256": sha(log)}
    receipt["runs"].append(row)
    if process.returncode or not observed[0]:
        raise RuntimeError("CUDA process or exit gate failed: " + label)
    return row, log.read_text(errors="replace")



def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    ROOT.mkdir(exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HOME=str(WORK / "hf"), HF_XET_CACHE=str(WORK / "xet"),
                      TMPDIR=str(WORK), CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    receipt = {"version": VERSION, "complete": False, "acceptance": False, "runs": [],
               "scope": "Native F16 only: same binary/model/input, timed decode switch off/on; reported encoder frames and emitted tokens are not directly comparable to original model steps.",
               "pins": {"input": INPUT[1], "runtime": RUNTIME[1], "native": NATIVE}}
    def save():
        (ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    save()
    receipt["hardware"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap",
        "--format=csv,noheader"], text=True).strip()
    save()
    if float(receipt["hardware"].splitlines()[0].split(",")[1].strip()) < 7.5:
        receipt["inconclusive_reason"] = "release CUDA binary requires SM75 or newer"
        save()
        return
    headroom()
    download(*INPUT, WORK / "input.wav")
    archive = download(*RUNTIME, WORK / "runtime.tar.gz")
    runtime = WORK / "runtime"
    runtime.mkdir()
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError("Unsafe runtime archive")
        bundle.extractall(runtime, filter="data")
    binary = next(runtime.rglob("crispasr"))
    if not binary.is_file() or sha(binary) != "4c07dbc07453e80684786203990ab6e89c1266bc13e6c61ca0e7ea5045444819":
        raise ValueError("Unexpected native executable")
    from huggingface_hub import hf_hub_download
    native = Path(hf_hub_download(NATIVE[0], NATIVE[2], revision=NATIVE[1], local_dir=WORK / "native"))
    if sha(native) != NATIVE[3]:
        raise ValueError("Native model changed")
    base_env = dict(os.environ, CRISPASR_NEMOTRON_BENCH="1")
    # Compare the shipped default greedy decoder with only timing toggled.
    for name in ("CRISPASR_RNNT_BATCH", "CRISPASR_NEMOTRON_DEBUG",
                 "CRISPASR_NEMOTRON_DECODE_TIMING"):
        base_env.pop(name, None)
    base_env["LD_LIBRARY_PATH"] = str(binary.parent) + os.pathsep + base_env.get("LD_LIBRARY_PATH", "")
    for phase, count in (("warmup", 1), ("measured", 3)):
        for index in range(count):
            order = ("off", "on") if index % 2 == 0 else ("on", "off")
            for mode in order:
                label = f"{phase}-{index}-{mode}"
                output = WORK / label
                env = dict(base_env)
                if mode == "on":
                    env["CRISPASR_NEMOTRON_DECODE_TIMING"] = "1"
                else:
                    env.pop("CRISPASR_NEMOTRON_DECODE_TIMING", None)
                command = [str(binary), "--backend", "nemotron", "-m", str(native), "-f",
                           str(WORK / "input.wav"), "-l", "en-US", "-t", "4", "-otxt", "-of", str(output)]
                row, log = run_call(command, env, label, receipt)
                transcript = output.with_suffix(".txt").read_text()
                row["wer"] = wer(REFERENCE, transcript)
                row["transcript_sha256"] = hashlib.sha256(transcript.encode()).hexdigest()
                row["stages_ms"] = {key: float(value) for key, value in re.findall(
                    r"nemotron_bench:\s+([a-z_]+)\s+([0-9.]+) ms", log)}
                count_rows = re.findall(r"nemotron: rnnt_decode ([0-9.]+) ms \(([^,]+), T_enc=([0-9]+), ([0-9]+) tokens\)", log)
                if mode == "on":
                    if len(count_rows) != 1:
                        raise RuntimeError("Expected one actual timed decoder count: " + label)
                    ms, decoder, frames, tokens = count_rows[0]
                    row["decoder"] = {"milliseconds": float(ms), "implementation": decoder,
                                      "encoder_frames": int(frames), "emitted_tokens": int(tokens),
                                      "milliseconds_per_encoder_frame": float(ms) / int(frames)}
                save()
                if row["wer"] > 0.2 or "nemotron: backend = CUDA0" not in log:
                    raise RuntimeError("Quality or GPU gate failed: " + label)
    if len({r["transcript_sha256"] for r in receipt["runs"]}) != 1:
        raise RuntimeError("Output changed when timing was enabled")
    receipt["median_process_seconds"] = {mode: statistics.median(r["seconds"] for r in receipt["runs"]
        if r["label"].startswith("measured") and r["label"].endswith(mode)) for mode in ("off", "on")}
    receipt["complete"] = receipt["acceptance"] = True
    save()


if __name__ == "__main__":
    main()
