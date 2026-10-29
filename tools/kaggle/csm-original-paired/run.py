#!/usr/bin/env python3
"""Private paired CUDA pilot: Sesame CSM without watermark versus native Q4_K.

Aggregate JSON is separate from private raw evidence. The Kaggle kernel must
remain private; generated WAVs and ASR logs are never published as receipts.
"""

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
import sys
import tarfile
import time
import urllib.request
import wave

VERSION = "csm-original-paired-v1"
OUT = Path("/kaggle/working")
WORK = Path("/kaggle/temp/csm-original-paired")
HARNESS_SOURCE = "cc114d2b4a5a9413738f390093e9131576743cb6"
SESAME_SOURCE = "836f886515f0dec02c22ed2316cc78904bdc0f36"
SILENTCIPHER_SOURCE = "d46d7d0893a583d8968ab3a6626e2289faec9152"
CSM_REV = "c92a71e1c419772e25be7dc14d952c2521a740ab"
LLAMA_REV = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
MIMI_REV = "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289"
RUNTIME = ("https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.47/"
           "crispasr-linux-x86_64-cuda.tar.gz",
           "90e23cffb44f4a80d929bf88cf3abb2d526ec4c0b95d88a82d1a8bf33135722d")
NATIVE = ("cstr/csm-1b-GGUF", "5abf936b9d17a839fca69e09baf31e0ea5068f97",
          "csm-1b-q4_k.gguf", "1123f9e8ff2d9212700d1fface749b8ec4f9765569bb996667a15a9a9832ee13")
ASR = ("cstr/parakeet-tdt-0.6b-v3-GGUF", "815cc1bcb5cfb92f9cd0bbd22c7bc2f6e2d4ec13",
       "parakeet-tdt-0.6b-v3-q4_k.gguf", "1a60f6e53e5781240dde6e69a47a47a8a71995a3a106517b009225afcc514457")
PHRASE = "The quick brown fox jumps over the lazy dog"
SEED = 1234
TEMPERATURE = 0.9
TOPK = 50
MAX_AUDIO_MS = 10_000


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def headroom():
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    gpu = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.free,memory.total",
                                   "--format=csv,noheader,nounits"], text=True).strip()
    row = {"available_bytes": int(mem["MemAvailable"].split()[0]) * 1024,
           "scratch_free_bytes": shutil.disk_usage(WORK).free,
           "gpu_free_mib": int(gpu.split(",")[0]), "load": list(os.getloadavg()),
           "cpus": os.cpu_count() or 1}
    if (row["available_bytes"] < 8 * 1024**3 or row["scratch_free_bytes"] < 10 * 1024**3
            or row["gpu_free_mib"] < 10 * 1024 or row["load"][0] > 2 * row["cpus"]):
        raise RuntimeError("Insufficient hosted resources for paired CSM run")
    return row


def download(url, expected, destination):
    with urllib.request.urlopen(url, timeout=180) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, 1024 * 1024)
    if sha(destination) != expected:
        raise ValueError("Pinned download mismatch: " + destination.name)
    return destination


def clone_pin(url, revision, destination):
    subprocess.run(["git", "init", str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "remote", "add", "origin", url], check=True)
    subprocess.run(["git", "-C", str(destination), "fetch", "--depth=1", "origin", revision], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise RuntimeError("Source checkout differs from pin")


def wav_info(path):
    with wave.open(str(path), "rb") as audio:
        row = {"channels": audio.getnchannels(), "sample_width": audio.getsampwidth(),
               "rate": audio.getframerate(), "frames": audio.getnframes()}
    row["duration_seconds"] = row["frames"] / row["rate"]
    row["sha256"] = sha(path)
    if row["channels"] != 1 or row["sample_width"] != 2 or row["rate"] != 24000:
        raise RuntimeError("Invalid generated CSM WAV format")
    if not (0.5 < row["duration_seconds"] < 10.0):
        raise RuntimeError("CSM output did not finish below the original 10 s cap")
    return row


def wer(text):
    ref = re.findall(r"[a-z0-9]+", PHRASE.lower())
    hyp = re.findall(r"[a-z0-9]+", text.lower())
    row = list(range(len(hyp) + 1))
    for word in ref:
        nxt = [row[0] + 1]
        for j, other in enumerate(hyp, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (word != other)))
        row = nxt
    return row[-1] / len(ref)


def original_child():
    # Pinned official Model and Generator; only paths, dtype and the
    # watermark call are selected here. No SilentCipher weights are loaded.
    sys.path.insert(0, os.environ["CSM_SESAME_DIR"])
    import torch
    import torchaudio
    from transformers import AutoTokenizer
    import generator as sesame
    from models import Model

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("Original CSM requires CUDA")
    original_tokenizer = AutoTokenizer.from_pretrained

    class PinnedTokenizer:
        @staticmethod
        def from_pretrained(name, *args, **kwargs):
            if name != "meta-llama/Llama-3.2-1B":
                raise RuntimeError("Unexpected tokenizer source")
            return original_tokenizer(os.environ["CSM_TOKENIZER_DIR"], *args, local_files_only=True, **kwargs)

    sesame.AutoTokenizer = PinnedTokenizer
    expected_mimi = os.environ["CSM_MIMI_PATH"]

    def pinned_mimi(repo, filename):
        if repo != "kyutai/moshiko-pytorch-bf16" or filename != "tokenizer-e351c8d8-checkpoint125.safetensors":
            raise RuntimeError("Unexpected Mimi model source")
        return expected_mimi

    sesame.hf_hub_download = pinned_mimi
    sesame.load_watermarker = lambda device: None
    sesame.watermark = lambda _watermarker, audio, sample_rate, _key: (audio, sample_rate)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    t0 = time.monotonic()
    model = Model.from_pretrained(os.environ["CSM_ORIGINAL_DIR"])
    model = model.to(device="cuda:0", dtype=torch.float32).eval()
    generator = sesame.Generator(model)
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - t0
    if torch.cuda.memory_allocated(0) < 1024 * 1024**2:
        raise RuntimeError("Original model did not allocate on GPU")
    frame_calls = [0]
    generate_frame = model.generate_frame

    def counted_frame(*args, **kwargs):
        frame_calls[0] += 1
        return generate_frame(*args, **kwargs)

    model.generate_frame = counted_frame
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.synchronize()
    t0 = time.monotonic()
    audio = generator.generate(text=PHRASE, speaker=0, context=[],
                               max_audio_length_ms=MAX_AUDIO_MS,
                               temperature=TEMPERATURE, topk=TOPK)
    torch.cuda.synchronize()
    generate_seconds = time.monotonic() - t0
    torchaudio.save(os.environ["CSM_ORIGINAL_WAV"], audio.float().cpu().unsqueeze(0), generator.sample_rate)
    Path(os.environ["CSM_ORIGINAL_RESULT"]).write_text(json.dumps({
        "load_seconds": load_seconds, "generate_seconds": generate_seconds,
        "frame_calls": frame_calls[0], "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype), "torch": torch.__version__,
        "torchaudio": torchaudio.__version__, "gpu_allocated_bytes": torch.cuda.memory_allocated(0),
    }, indent=2) + "\n")


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    def preserve_private_evidence():
        dest = OUT / "private-evidence"
        dest.mkdir(exist_ok=True)
        for pattern in ("*.wav", "*.log", "warmup-*.json", "measured-*.json"):
            for path in WORK.glob(pattern):
                if path.is_file():
                    shutil.copy2(path, dest / path.name)

    atexit.register(preserve_private_evidence)
    os.environ.update(TMPDIR=str(WORK), HF_HOME=str(WORK / "hf"), HF_XET_CACHE=str(WORK / "xet"),
                      CUDA_VISIBLE_DEVICES="0", NO_TORCH_COMPILE="1", OMP_NUM_THREADS="4",
                      OPENBLAS_NUM_THREADS="1", USE_TF="0")
    receipt = {"version": VERSION, "complete": False, "acceptance": False,
               "scope": "private paired process timing; Sesame Model/Generator F32 with watermark disabled versus native Q4_K",
               "pins": {"runtime": RUNTIME, "native": NATIVE, "asr": ASR,
                        "sesame_source": SESAME_SOURCE, "silentcipher_source": SILENTCIPHER_SOURCE,
                        "csm_original": CSM_REV, "llama_tokenizer": LLAMA_REV,
                        "mimi": MIMI_REV, "silentcipher_weights_used": False},
               "settings": {"text": PHRASE, "speaker": 0, "context": [], "seed": SEED,
                            "temperature": TEMPERATURE, "topk": TOPK, "original_max_audio_ms": MAX_AUDIO_MS},
               "runs": [], "resource_checks": []}

    def save():
        (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def check(label):
        state = headroom()
        receipt["resource_checks"].append({"label": label, **state})
        save()

    save()
    check("admission")
    hardware = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=name,compute_cap,driver_version",
                                        "--format=csv,noheader"], text=True).strip()
    receipt["hardware"] = hardware
    clone_pin("https://github.com/CrispStrobe/CrispASR.git", HARNESS_SOURCE, WORK / "harness")
    sys.path.insert(0, str(WORK / "harness/tools/kaggle"))
    import kaggle_harness as kh
    kh.init_progress(hf_progress_repo="")
    os.environ["HF_TOKEN"] = kh.kaggle_token_from_dataset("hf_token.txt") or kh.resolve_hf_token(require=True)
    clone_pin("https://github.com/SesameAILabs/csm.git", SESAME_SOURCE, WORK / "sesame")
    check("before-dependencies")
    # Keep Kaggle's CUDA Torch/Torchaudio. All other original requirements are
    # pinned; --no-deps prevents a hidden multi-GB Torch replacement.
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-q",
                    "tokenizers==0.21.0", "transformers==4.49.0", "huggingface_hub==0.28.1",
                    "moshi==0.2.2", "torchtune==0.4.0", "torchao==0.9.0",
                    "git+https://github.com/SesameAILabs/silentcipher@" + SILENTCIPHER_SOURCE], check=True)
    import torch
    import torchaudio
    if not torch.cuda.is_available() or not torch.version.cuda or torch.__version__.split("+")[0] != torchaudio.__version__.split("+")[0]:
        raise RuntimeError("Kaggle Torch/Torchaudio CUDA pair is unavailable")
    receipt["software"] = {"torch": torch.__version__, "torchaudio": torchaudio.__version__}
    from huggingface_hub import hf_hub_download, snapshot_download
    check("before-model-downloads")
    original = Path(snapshot_download("sesame/csm-1b", revision=CSM_REV,
                                      allow_patterns=["*.json", "*.safetensors", "README.md"],
                                      local_dir=WORK / "original"))
    tokenizer = Path(snapshot_download("meta-llama/Llama-3.2-1B", revision=LLAMA_REV,
                                       allow_patterns=["tokenizer*", "*.json", "*.model"],
                                       local_dir=WORK / "tokenizer"))
    mimi = Path(hf_hub_download("kyutai/moshiko-pytorch-bf16",
                                "tokenizer-e351c8d8-checkpoint125.safetensors", revision=MIMI_REV,
                                local_dir=WORK / "mimi"))
    receipt["original_files"] = {str(p.relative_to(original)): sha(p) for p in original.rglob("*") if p.is_file()}
    receipt["tokenizer_files"] = {str(p.relative_to(tokenizer)): sha(p) for p in tokenizer.rglob("*") if p.is_file()}
    receipt["mimi_sha256"] = sha(mimi)
    archive = download(*RUNTIME, WORK / "runtime.tar.gz")
    runtime = WORK / "runtime"
    runtime.mkdir()
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if any(Path(m.name).is_absolute() or ".." in Path(m.name).parts or
               not (m.isfile() or m.isdir() or m.issym()) or
               (m.issym() and (Path(m.linkname).is_absolute() or ".." in Path(m.linkname).parts))
               for m in members):
            raise ValueError("Unsafe native archive")
        bundle.extractall(runtime, filter="data")
    binary = next(runtime.rglob("crispasr"))
    if not binary.is_file() or sha(binary) != "4c07dbc07453e80684786203990ab6e89c1266bc13e6c61ca0e7ea5045444819":
        raise RuntimeError("Native binary differs from pinned release")
    native = Path(hf_hub_download(NATIVE[0], NATIVE[2], revision=NATIVE[1], local_dir=WORK / "native"))
    asr = Path(hf_hub_download(ASR[0], ASR[2], revision=ASR[1], local_dir=WORK / "asr"))
    if sha(native) != NATIVE[3] or sha(asr) != ASR[3]:
        raise RuntimeError("Native model checksum differs from qualification")
    env = dict(os.environ, CSM_SESAME_DIR=str(WORK / "sesame"),
               CSM_ORIGINAL_DIR=str(original), CSM_TOKENIZER_DIR=str(tokenizer),
               CSM_MIMI_PATH=str(mimi), GGML_BACKEND_PATH=str(binary.parent),
               LD_LIBRARY_PATH=str(binary.parent) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))
    sys.path.insert(0, str(WORK / "sesame"))
    receipt["entry_sha256"] = sha(__file__)
    save()

    def call(engine, label):
        check("before-" + label)
        wav = WORK / (label + ".wav")
        log = WORK / (label + ".log")
        command = ([binary, "--backend", "csm", "-m", native, "--gpu-backend", "cuda",
                    "--tts", PHRASE, "--voice", "default", "--seed", str(SEED),
                    "--temperature", str(TEMPERATURE), "--tts-output", wav, "-t", "4"]
                   if engine == "native" else
                   [sys.executable, "-B", __file__, "--original-child"])
        child_env = dict(env, CRISPASR_CSM_BENCH="1")
        if engine == "original":
            child_env.update(CSM_ORIGINAL_WAV=str(wav), CSM_ORIGINAL_RESULT=str(WORK / (label + ".json")))
        started = time.monotonic()
        with log.open("wb") as output:
            process = subprocess.Popen(list(map(str, command)), stdout=output, stderr=subprocess.STDOUT,
                                       cwd=WORK / "sesame" if engine == "original" else binary.parent,
                                       env=child_env)
            peak = 0
            deadline = time.monotonic() + 900
            while process.poll() is None:
                apps = subprocess.run(["nvidia-smi", "--id=0", "--query-compute-apps=pid,used_gpu_memory",
                                       "--format=csv,noheader,nounits"], capture_output=True, text=True)
                for line in apps.stdout.splitlines():
                    pid, _, used = line.partition(",")
                    if pid.strip() == str(process.pid) and used.strip().isdigit():
                        peak = max(peak, int(used.strip()))
                if time.monotonic() > deadline:
                    process.kill()
                    process.wait()
                    raise TimeoutError(label)
                time.sleep(0.2)
        row = {"label": label, "engine": engine, "process_seconds": time.monotonic() - started,
               "returncode": process.returncode, "peak_gpu_mib": peak, "log_sha256": sha(log)}
        receipt["runs"].append(row)
        save()
        if process.returncode or peak < 512:
            raise RuntimeError("Native/original call failed or lacked GPU model residency: " + label)
        row["wav"] = wav_info(wav)
        text = log.read_text(errors="replace")
        if engine == "native":
            match = re.search(r"csm_tts: generated ([0-9]+) audio frames", text)
            if not match:
                raise RuntimeError("Native generation frame count missing")
            row["generated_frames"] = int(match.group(1))
            row["stages_ms"] = {name: float(ms) for name, ms in re.findall(
                r"csm_bench:\s+([a-z_]+)\s+([0-9.]+) ms", text)}
            if not {"tokenize", "mimi_decode"} <= row["stages_ms"].keys():
                raise RuntimeError("Native stage counters missing")
        else:
            details = json.loads((WORK / (label + ".json")).read_text())
            if details["device"] != "cuda:0" or details["dtype"] != "torch.float32" or details["frame_calls"] < 1:
                raise RuntimeError("Original device/precision/frame count gate failed")
            row["original"] = details
        judge_log = WORK / (label + "-asr.log")
        with judge_log.open("wb") as output:
            judge = subprocess.run([binary, "--backend", "parakeet", "-m", asr, "-f", wav,
                                    "--gpu-backend", "cuda", "--no-prints", "-t", "4"],
                                   stdout=output, stderr=subprocess.STDOUT, cwd=binary.parent, env=env,
                                   timeout=300)
        if judge.returncode:
            raise RuntimeError("ASR quality judge failed: " + label)
        lines = [line.strip() for line in judge_log.read_text(errors="replace").splitlines() if line.strip()]
        transcript = re.sub(r"(?i)^this audio was generated by artificial intelligence[. ]*", "", lines[-1])
        row["asr_wer"] = wer(transcript)
        row["asr_log_sha256"] = sha(judge_log)
        save()
        if row["asr_wer"] > 0.15:
            raise RuntimeError("CSM speech quality gate failed: " + label)

    for phase, count in (("warmup", 1), ("measured", 3)):
        for index in range(count):
            order = ("native", "original") if index % 2 == 0 else ("original", "native")
            for engine in order:
                call(engine, f"{phase}-{index}-{engine}")
    receipt["median_process_seconds"] = {
        engine: statistics.median(r["process_seconds"] for r in receipt["runs"]
                                  if r["label"].startswith("measured") and r["engine"] == engine)
        for engine in ("native", "original")}
    receipt["complete"] = receipt["acceptance"] = True
    save()
    print("CSM_ORIGINAL_PAIRED_ACCEPTED", receipt["median_process_seconds"], flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--original-child"]:
        original_child()
    else:
        main()
