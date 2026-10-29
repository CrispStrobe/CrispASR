#!/usr/bin/env python3
"""Paired Granite Speech5 CUDA timing with pinned outputs and decode counters.

The original is F16 PyTorch on T4; CrispASR is the accepted mixed Q4_K GGUF.
Process times include model loading; native decode counters exclude it. These
are distinct precision profiles and both output quality and settings are gated.
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

VERSION = "granite5-original-paired-v1"
ROOT = Path("/kaggle/working")
WORK = Path("/kaggle/temp/granite5-original-paired")
INPUT = ("https://raw.githubusercontent.com/CrispStrobe/CrispASR/"
         "cc114d2b4a5a9413738f390093e9131576743cb6/samples/jfk.wav",
         "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e")
RUNTIME = ("https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.47/"
           "crispasr-linux-x86_64-cuda.tar.gz",
           "90e23cffb44f4a80d929bf88cf3abb2d526ec4c0b95d88a82d1a8bf33135722d")
NATIVE = ("cstr/crispasr-granite5-validation-public",
          "d0e6349c7d0601fdd8ea38d1aad90e6b92ac47e2",
          "q4-recipes/8fbb1794da4eea3851d8384934695d39b3518b72/granite-speech-5.0-2b-q4_k-value-down-f16.gguf",
          "63324167e3a85a67539a44712aef1a31e864674d")
ORIGINAL = ("ibm-granite/granite-speech-5.0-2b",
            "c075cf19e041a2e4d16c5ee24763e5b037d8f479")
REFERENCE = "and so my fellow americans ask not what your country can do for you ask what you can do for your country"
INSTRUCTION = "transcribe the speech with proper punctuation and capitalization."


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
    if (state["available_bytes"] < 3 * 1024**3 or state["scratch_free_bytes"] < 9 * 1024**3
            or state["load"][0] > 2 * state["cpus"]):
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
    import wave
    from transformers import GraniteSpeechForConditionalGeneration, AutoProcessor
    torch.set_num_threads(4)
    assert torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (7, 5)
    model_dir = os.environ["GRANITE_ORIGINAL_DIR"]
    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.float16, device_map="cuda:0").eval()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    assert model.device.type == "cuda" and model.dtype == torch.float16
    if any(p.device.type != "cuda" or (p.is_floating_point() and p.dtype != torch.float16)
           for p in model.parameters()):
        raise RuntimeError("Original model has CPU or non-F16 floating parameters")
    with wave.open(os.environ["GRANITE_AUDIO"]) as source:
        assert source.getframerate() == 16000 and source.getnchannels() == 1 and source.getsampwidth() == 2
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768
    assert np.isfinite(audio).all() and audio.ndim == 1
    prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": "<|audio|>" + INSTRUCTION}], tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, torch.from_numpy(audio).unsqueeze(0), device="cpu", return_tensors="pt")
    inputs = {key: value.to(model.device, dtype=model.dtype if value.is_floating_point() else value.dtype)
              for key, value in inputs.items()}
    calls = []
    for index in range(2):
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=96, do_sample=False, num_beams=1)
        torch.cuda.synchronize()
        generate_seconds = time.monotonic() - start
        new_tokens = output[0, inputs["input_ids"].shape[-1]:].unsqueeze(0)
        transcript = processor.tokenizer.batch_decode(new_tokens, add_special_tokens=False,
                                                      skip_special_tokens=True)[0]
        calls.append({"index": index, "transcript": transcript, "generate_seconds": generate_seconds,
                      "generated_token_count": int(new_tokens.shape[-1])})
    payload = {"calls": calls, "device": str(model.device), "dtype": str(model.dtype),
               "torch": torch.__version__}
    Path(os.environ["GRANITE_CHILD_OUTPUT"]).write_text(json.dumps(payload, indent=2) + "\n")


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
               "scope": "alternating two-inference process time including load; original F16 PyTorch versus accepted mixed Q4_K GGUF; within-process second-call generate/decode timing excludes load",
               "pins": {"input": INPUT[1], "runtime": RUNTIME[1], "native": NATIVE,
                        "original": ORIGINAL, "transformers": "4.57.3"},
               "instruction": INSTRUCTION, "max_new_tokens": 96, "runs": []}

    def save():
        (ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    receipt["hardware"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
        "--format=csv,noheader"], text=True).strip()
    save()
    cap = float(receipt["hardware"].splitlines()[0].split(",")[1].strip())
    if cap < 7.5:
        receipt["inconclusive_reason"] = "Original PyTorch CUDA wheel and F16 comparison require T4 or newer"
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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.3",
                    "huggingface_hub>=0.36,<2", "soundfile", "accelerate"], check=True)
    from importlib.metadata import version
    if version("transformers") != "4.57.3":
        raise RuntimeError("Unexpected original software version")
    from huggingface_hub import hf_hub_download, snapshot_download
    native = Path(hf_hub_download(NATIVE[0], NATIVE[2], revision=NATIVE[1], local_dir=WORK / "native"))
    if sha(native) != NATIVE[3]:
        raise ValueError("Native model changed")
    original = Path(snapshot_download(ORIGINAL[0], revision=ORIGINAL[1],
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
        local_dir=WORK / "original"))
    weights = list(original.glob("*.safetensors"))
    if not weights:
        raise ValueError("Pinned original checkpoint missing")
    receipt["original_files"] = {str(p.relative_to(original)): sha(p) for p in original.rglob("*")
                                 if p.is_file() and ".cache" not in p.parts}
    save()
    base_env = dict(os.environ)
    base_env["LD_LIBRARY_PATH"] = str(binary.parent) + os.pathsep + base_env.get("LD_LIBRARY_PATH", "")
    base_env.update(GRANITE_ORIGINAL_DIR=str(original), GRANITE_AUDIO=str(WORK / "input.wav"))
    for phase, count in (("warmup", 1), ("measured", 3)):
        for index in range(count):
            order = ("native", "original") if index % 2 == 0 else ("original", "native")
            for engine in order:
                label = f"{phase}-{index}-{engine}"
                if engine == "native":
                    output = [WORK / (label + f"-{j}") for j in range(2)]
                    env = dict(base_env, CRISPASR_GRANITE_DEC_PROFILE="1")
                    if phase == "warmup":
                        env["GGML_SCHED_DEBUG"] = "2"
                    command = [str(binary), "--backend", "granite", "-m", str(native), "-f",
                               str(WORK / "input.wav"), "-f", str(WORK / "input.wav"), "-l", "en",
                               "--ask", INSTRUCTION, "--max-new-tokens", "96", "-t", "4", "-p", "1",
                               "-otxt", "-of", str(output[0]), "-of", str(output[1])]
                else:
                    output = WORK / (label + ".json")
                    env = dict(base_env, GRANITE_CHILD_OUTPUT=str(output))
                    command = [sys.executable, "-B", __file__, "--original-child"]
                row, log = run_call(command, env, label, receipt)
                if engine == "native":
                    transcript_paths = [p.with_suffix(".txt") for p in output]
                    if any(not p.is_file() for p in transcript_paths) or "[granite dec-profile] backend=CUDA0" not in log:
                        raise RuntimeError("Native output or CUDA decode backend missing: " + label)
                    transcripts = [p.read_text() for p in transcript_paths]
                    profile = re.findall(r"\[granite dec-profile\] (\d+) steps, ([0-9.]+) ms total \| alloc ([0-9.]+) us/step, compute ([0-9.]+) us/step", log)
                    if len(profile) != 2 or any(int(p[0]) < 10 for p in profile):
                        raise RuntimeError("Missing native decode-step counters: " + label)
                    row["decode_profile"] = [{"steps": int(p[0]), "total_ms": float(p[1]),
                                              "alloc_us_per_step": float(p[2]),
                                              "compute_us_per_step": float(p[3])} for p in profile]
                    if phase == "warmup":
                        row["cuda_matmul_nodes"] = len(re.findall(
                            r"node\s+#\s*\d+\s+\(\s*MUL_MAT(?:_ID)?\s*\):[^\n]*\[\s*CUDA0\b[^\]]*\]\s+use=", log))
                        if row["cuda_matmul_nodes"] == 0:
                            raise RuntimeError("Native warmup lacks CUDA-assigned matmul nodes")
                else:
                    detail = json.loads(output.read_text())
                    if detail["device"] != "cuda:0" or detail["dtype"] != "torch.float16":
                        raise RuntimeError("Original did not use F16 CUDA")
                    calls = detail.pop("calls")
                    if len(calls) != 2 or any(c["generated_token_count"] < 10 for c in calls):
                        raise RuntimeError("Original generation lacked token work")
                    transcripts = [c["transcript"] for c in calls]
                    row["original"] = detail
                    row["original"]["calls"] = [{k: v for k, v in c.items() if k != "transcript"} for c in calls]
                row["transcripts"] = transcripts
                row["wer"] = [wer(REFERENCE, text) for text in transcripts]
                save()
                if any(value > 0.2 for value in row["wer"]):
                    raise RuntimeError("Transcript quality gate failed: " + label)
    receipt["median_process_seconds"] = {engine: statistics.median(r["seconds"] for r in receipt["runs"]
        if r["label"].startswith("measured") and r["label"].endswith(engine)) for engine in ("native", "original")}
    receipt["complete"] = receipt["acceptance"] = True
    save()
    print("GRANITE5_ORIGINAL_PAIRED_ACCEPTED", receipt["median_process_seconds"], flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--original-child"]:
        original_child()
    else:
        main()
