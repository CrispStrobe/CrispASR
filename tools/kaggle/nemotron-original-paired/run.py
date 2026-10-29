#!/usr/bin/env python3
"""Paired Nemotron CUDA timing with pinned outputs and per-stage native counters.

The original is F32 PyTorch; CrispASR is the accepted F16 GGUF. Process times
include model loading. Native stage counters exclude loading. These are distinct
precision profiles and no numerical speedup is claimed without that disclosure.
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

VERSION = "nemotron-original-paired-v1"
ROOT = Path("/kaggle/working")
WORK = Path("/kaggle/temp/nemotron-original-paired")
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
    if state["available_bytes"] < 3 * 1024**3 or state["scratch_free_bytes"] < 6 * 1024**3:
        raise RuntimeError("Insufficient worker resources")
    return state


def download(url, expected, target):
    with urllib.request.urlopen(url, timeout=180) as source, target.open("wb") as dest:
        shutil.copyfileobj(source, dest, 1024 * 1024)
    if sha(target) != expected:
        raise ValueError("Pinned input hash mismatch: " + target.name)
    return target


def original_child():
    # This process is independent for every call, like the native CLI process.
    import numpy as np
    import torch
    from transformers import AutoModelForRNNT, AutoProcessor
    from transformers.audio_utils import load_audio
    torch.set_num_threads(4)
    assert torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (7, 0)
    model_dir = os.environ["NEMOTRON_ORIGINAL_DIR"]
    model = AutoModelForRNNT.from_pretrained(model_dir, local_files_only=True,
                                              dtype=torch.float32).to("cuda:0").eval()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    processor.set_num_lookahead_tokens(3)
    assert model.device.type == "cuda" and model.dtype == torch.float32
    audio = load_audio(os.environ["NEMOTRON_AUDIO"], sampling_rate=16000)
    assert np.isfinite(audio).all() and audio.ndim == 1
    inputs = processor(audio, sampling_rate=16000, language="en-US", return_tensors="pt")
    inputs = inputs.to(model.device, dtype=model.dtype)
    torch.cuda.synchronize()
    start = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=512)
    torch.cuda.synchronize()
    transcript = processor.batch_decode(output.sequences, skip_special_tokens=True)[0]
    payload = {"transcript": transcript, "generate_seconds": time.monotonic() - start,
               "sequences_shape": list(output.sequences.shape), "device": str(model.device),
               "dtype": str(model.dtype), "torch": torch.__version__}
    Path(os.environ["NEMOTRON_CHILD_OUTPUT"]).write_text(json.dumps(payload, indent=2) + "\n")


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
                      TMPDIR=str(WORK), CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4",
                      OPENBLAS_NUM_THREADS="1", USE_TF="0")
    receipt = {"version": VERSION, "complete": False, "acceptance": False,
               "scope": "alternating process time including load; original F32 PyTorch versus native F16 GGUF; native per-stage timing only",
               "pins": {"input": INPUT[1], "runtime": RUNTIME[1], "native": NATIVE,
                        "original": ORIGINAL, "transformers": "5.19.0"}, "runs": []}

    def save():
        (ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    receipt["hardware"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
        "--format=csv,noheader"], text=True).strip()
    save()
    cap = float(receipt["hardware"].splitlines()[0].split(",")[1].strip())
    if cap < 7.0:
        receipt["inconclusive_reason"] = "Original PyTorch CUDA wheel requires compute capability >=7.0"
        save()
        print("P100_UNSUPPORTED_ORIGINAL", flush=True)
        return
    receipt["resources_before"] = headroom()
    save()
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
    from huggingface_hub import hf_hub_download, snapshot_download
    native = Path(hf_hub_download(NATIVE[0], NATIVE[2], revision=NATIVE[1], local_dir=WORK / "native"))
    if sha(native) != NATIVE[3]:
        raise ValueError("Native model changed")
    original = Path(snapshot_download(ORIGINAL[0], revision=ORIGINAL[1],
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"], local_dir=WORK / "original"))
    if sha(original / "model.safetensors") != ORIGINAL[2]:
        raise ValueError("Original model changed")
    receipt["original_files"] = {str(p.relative_to(original)): sha(p) for p in original.rglob("*") if p.is_file()}
    save()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.19.0",
                    "huggingface_hub>=0.36,<2", "soundfile", "accelerate"], check=True)
    import transformers
    if transformers.__version__ != "5.19.0":
        raise RuntimeError("Unexpected original software version")
    base_env = dict(os.environ)
    base_env["LD_LIBRARY_PATH"] = str(binary.parent) + os.pathsep + base_env.get("LD_LIBRARY_PATH", "")
    base_env.update(NEMOTRON_ORIGINAL_DIR=str(original), NEMOTRON_AUDIO=str(WORK / "input.wav"))
    for phase, count in (("warmup", 1), ("measured", 3)):
        for index in range(count):
            order = ("native", "original") if index % 2 == 0 else ("original", "native")
            for engine in order:
                label = f"{phase}-{index}-{engine}"
                if engine == "native":
                    output = WORK / label
                    env = dict(base_env, CRISPASR_NEMOTRON_BENCH="1")
                    command = [str(binary), "--backend", "nemotron", "-m", str(native), "-f",
                               str(WORK / "input.wav"), "-l", "en-US", "-t", "4", "-otxt", "-of", str(output)]
                else:
                    output = WORK / (label + ".json")
                    env = dict(base_env, NEMOTRON_CHILD_OUTPUT=str(output))
                    command = [sys.executable, "-B", __file__, "--original-child"]
                row, log = run_call(command, env, label, receipt)
                if engine == "native":
                    transcript_path = output.with_suffix(".txt")
                    if not transcript_path.is_file() or "nemotron: backend = CUDA0" not in log:
                        raise RuntimeError("Native output or CUDA backend missing: " + label)
                    transcript = transcript_path.read_text()
                    row["stages_ms"] = {key: float(value) for key, value in re.findall(
                        r"nemotron_bench:\s+([a-z_]+)\s+([0-9.]+) ms", log)}
                    row["decode"] = re.findall(r"nemotron: rnnt_decode ([0-9.]+) ms \(([^,]+), T_enc=([0-9]+), ([0-9]+) tokens\)", log)
                    if not {"mel", "encoder", "prompt", "rnnt_decode"} <= row["stages_ms"].keys():
                        raise RuntimeError("Missing native stage counters: " + label)
                else:
                    detail = json.loads(output.read_text())
                    if detail["device"] != "cuda:0" or detail["dtype"] != "torch.float32":
                        raise RuntimeError("Original did not use F32 CUDA")
                    transcript = detail.pop("transcript")
                    row["original"] = detail
                row["transcript"] = transcript
                row["wer"] = wer(REFERENCE, transcript)
                save()
                if row["wer"] > 0.2:
                    raise RuntimeError("Transcript quality gate failed: " + label)
    receipt["median_process_seconds"] = {engine: statistics.median(r["seconds"] for r in receipt["runs"]
        if r["label"].startswith("measured") and r["label"].endswith(engine)) for engine in ("native", "original")}
    receipt["complete"] = receipt["acceptance"] = True
    save()
    print("NEMOTRON_ORIGINAL_PAIRED_ACCEPTED", receipt["median_process_seconds"], flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--original-child"]:
        original_child()
    else:
        main()
