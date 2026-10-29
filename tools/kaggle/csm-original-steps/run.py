#!/usr/bin/env python3
"""Private, GPU-only CSM original F32 step timing after the accepted v2 pilot.

This isolates the pinned Sesame Model/Generator and disables SilentCipher
watermarking exactly as the paired v2 run did. It does not compare native steps.
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
import threading
import time
import urllib.request
import wave


VERSION = "csm-original-steps-v1"
OUT = Path("/kaggle/working")
WORK = Path("/kaggle/temp/csm-original-steps")
HARNESS_SOURCE = "cc114d2b4a5a9413738f390093e9131576743cb6"
SESAME_SOURCE = "836f886515f0dec02c22ed2316cc78904bdc0f36"
SILENTCIPHER_SOURCE = "d46d7d0893a583d8968ab3a6626e2289faec9152"
CSM_REV = "c92a71e1c419772e25be7dc14d952c2521a740ab"
LLAMA_REV = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
MIMI_REV = "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289"
RUNTIME = (
    "https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.47/"
    "crispasr-linux-x86_64-cuda.tar.gz",
    "90e23cffb44f4a80d929bf88cf3abb2d526ec4c0b95d88a82d1a8bf33135722d",
)
ASR = (
    "cstr/parakeet-tdt-0.6b-v3-GGUF", "815cc1bcb5cfb92f9cd0bbd22c7bc2f6e2d4ec13",
    "parakeet-tdt-0.6b-v3-q4_k.gguf", "1a60f6e53e5781240dde6e69a47a47a8a71995a3a106517b009225afcc514457",
)
PHRASE = "The quick brown fox jumps over the lazy dog"
SEED = 1234
TEMPERATURE = 0.9
TOPK = 50
MAX_AUDIO_MS = 10_000
EXPECTED_CODEBOOKS = 32
MEASURED_PAIRS = 3


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def headroom():
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    gpu = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    row = {
        "available_bytes": int(memory["MemAvailable"].split()[0]) * 1024,
        "scratch_free_bytes": shutil.disk_usage(WORK).free,
        "gpu_free_mib": int(gpu.split(",")[0]),
        "load": list(os.getloadavg()), "cpus": os.cpu_count() or 1,
    }
    if (row["available_bytes"] < 8 * 1024**3 or row["scratch_free_bytes"] < 10 * 1024**3
            or row["gpu_free_mib"] < 10 * 1024 or row["load"][0] > 2 * row["cpus"]):
        raise RuntimeError("Insufficient hosted resources for CSM original step diagnostic")
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
    if (row["channels"], row["sample_width"], row["rate"]) != (1, 2, 24000):
        raise RuntimeError("Invalid original CSM WAV format")
    if not (0.5 < row["duration_seconds"] < 10.0):
        raise RuntimeError("CSM output did not stop below the original 10 s cap")
    return row


def wer(text):
    ref = re.findall(r"[a-z0-9]+", PHRASE.lower())
    hyp = re.findall(r"[a-z0-9]+", text.lower())
    row = list(range(len(hyp) + 1))
    for word in ref:
        next_row = [row[0] + 1]
        for j, other in enumerate(hyp, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (word != other)))
        row = next_row
    return row[-1] / len(ref)


class GpuPidWatch:
    def __init__(self):
        self.pid = os.getpid()
        self.stop = threading.Event()
        self.samples = []
        self.sm_samples = []
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self):
        while not self.stop.is_set():
            probe = subprocess.run(
                ["nvidia-smi", "--id=0", "--query-compute-apps=pid,used_gpu_memory",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
            )
            if probe.returncode == 0:
                for line in probe.stdout.splitlines():
                    pid, _, used = line.partition(",")
                    if pid.strip() == str(self.pid) and used.strip().isdigit():
                        self.samples.append(int(used.strip()))
            activity = subprocess.run(
                ["nvidia-smi", "pmon", "-c", "1", "-s", "u"],
                capture_output=True, text=True, timeout=10,
            )
            for line in activity.stdout.splitlines():
                fields = line.split()
                if len(fields) >= 4 and fields[1] == str(self.pid) and fields[3].isdigit():
                    self.sm_samples.append(int(fields[3]))
            self.stop.wait(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


class FrameProfile:
    """Observe the official frame method and its real backbone/depth forwards.

    `generate_frame` is a plain method, so hooks on Model.forward cannot see it.
    Sampling and projection are included in the full-frame timer, while the
    backbone/depth module hooks measure their own nested forward boundaries.
    """

    def __init__(self, model, torch, timed):
        self.model, self.torch, self.timed = model, torch, timed
        self.original_frame = model.generate_frame
        self.handles = []
        self.rows = []
        self.samples = []
        self.active = None
        self.child_start = {}

    def _now(self):
        self.torch.cuda.synchronize()
        return time.perf_counter_ns()

    def _pre(self, name):
        def hook(module, args, kwargs):
            if self.active is None or name in self.child_start:
                raise RuntimeError("Unexpected CSM " + name + " forward nesting")
            if not args or len(args[0].shape) < 2:
                raise RuntimeError("Missing CSM " + name + " forward sequence")
            self.child_start[name] = (self._now(), int(args[0].shape[1]))
        return hook

    def _post(self, name):
        def hook(module, args, kwargs, output):
            if self.active is None or name not in self.child_start:
                raise RuntimeError("Unexpected CSM " + name + " forward completion")
            end = self._now()
            begin, sequence_length = self.child_start.pop(name)
            self.active[name + "_calls"].append({
                "input_sequence_length": sequence_length,
                "forward_ms": (end - begin) / 1e6,
            })
        return hook

    def _frame(self, *args, **kwargs):
        if self.active is not None or not args or len(args[0].shape) < 2:
            raise RuntimeError("Unexpected CSM frame invocation")
        self.active = {"index": len(self.rows), "input_sequence_length": int(args[0].shape[1]),
                       "backbone_calls": [], "decoder_calls": []}
        begin = self._now() if self.timed else None
        result = self.original_frame(*args, **kwargs)
        end = self._now() if self.timed else None
        if not hasattr(result, "shape") or len(result.shape) != 2:
            raise RuntimeError("CSM frame result shape changed")
        self.samples.append(result.detach())  # hash after timed generation, never copy in a hook
        self.active["sample_shape"] = list(result.shape)
        if self.timed:
            self.active["full_frame_ms"] = (end - begin) / 1e6
        self.rows.append(self.active)
        self.active = None
        return result

    def __enter__(self):
        self.model.generate_frame = self._frame
        if self.timed:
            for name, module in (("backbone", self.model.backbone), ("decoder", self.model.decoder)):
                self.handles.append(module.register_forward_pre_hook(self._pre(name), with_kwargs=True))
                self.handles.append(module.register_forward_hook(self._post(name), with_kwargs=True))
        return self

    def __exit__(self, *_):
        self.model.generate_frame = self.original_frame
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        if self.active is not None or self.child_start:
            raise RuntimeError("Incomplete CSM forward trace")


def validate_frames(rows, accepted, eos_count, codebooks, timed):
    if not rows or codebooks != EXPECTED_CODEBOOKS:
        raise RuntimeError("Missing frames or unexpected codebook count")
    if eos_count != 1 or accepted != len(rows) - 1:
        raise RuntimeError("Original generation did not stop after one EOS attempt")
    if rows[0]["input_sequence_length"] <= 1 or any(r["input_sequence_length"] != 1 for r in rows[1:]):
        raise RuntimeError("CSM backbone prefill/single-frame sequence shapes changed")
    for row in rows:
        if row["sample_shape"] != [1, codebooks]:
            raise RuntimeError("CSM frame sample shape changed")
        if timed:
            if len(row["backbone_calls"]) != 1 or len(row["decoder_calls"]) != codebooks - 1:
                raise RuntimeError("CSM frame backbone/depth call count changed")
            if row["backbone_calls"][0]["input_sequence_length"] != row["input_sequence_length"]:
                raise RuntimeError("Backbone sequence shape differs from frame input")
            depth_lengths = [call["input_sequence_length"] for call in row["decoder_calls"]]
            if depth_lengths != [2] + [1] * (codebooks - 2):
                raise RuntimeError("Depth prefill or single-codebook call shape changed")
            durations = [row["full_frame_ms"]] + [c["forward_ms"] for c in row["backbone_calls"]]
            durations += [c["forward_ms"] for c in row["decoder_calls"]]
            if any(ms <= 0 for ms in durations):
                raise RuntimeError("Invalid synchronized forward duration")


def summarize_frames(rows):
    frame_times = [row["full_frame_ms"] for row in rows]
    backbone_times = [row["backbone_calls"][0]["forward_ms"] for row in rows]
    depth_times = [[call["forward_ms"] for call in row["decoder_calls"]] for row in rows]
    return {
        "prefill_frame_full_ms": frame_times[0],
        "prefill_backbone_forward_ms": backbone_times[0],
        "prefill_depth_31_forward_total_ms": sum(depth_times[0]),
        "later_frame_full_median_ms": statistics.median(frame_times[1:]) if len(rows) > 1 else None,
        "later_backbone_forward_median_ms": statistics.median(backbone_times[1:]) if len(rows) > 1 else None,
        "later_depth_31_forward_total_median_ms": (
            statistics.median(sum(frame) for frame in depth_times[1:]) if len(rows) > 1 else None
        ),
        "depth_forward_median_ms": statistics.median(ms for frame in depth_times for ms in frame),
    }


def validate_call_identity(calls):
    if len(calls) != 2 * (1 + MEASURED_PAIRS):
        raise RuntimeError("Missing CSM timing-off/on calls")
    for phase, pairs in (("warmup", 1), ("measured", MEASURED_PAIRS)):
        for pair in range(pairs):
            arms = [row["arm"] for row in calls if row["phase"] == phase and row["pair"] == pair]
            if sorted(arms) != ["timing_off", "timing_on"]:
                raise RuntimeError("Incomplete CSM timing-off/on pair")
    identities = {(row["attempted_frame_calls"], row["accepted_audio_frames"],
                   row["all_attempted_codes_sha256"], row["wav"]["sha256"]) for row in calls}
    if len(identities) != 1:
        raise RuntimeError("Timing-off/on original code or audio identity changed")


def original_child():
    sys.path.insert(0, os.environ["CSM_SESAME_DIR"])
    import torch
    import torchaudio
    from transformers import AutoTokenizer
    import generator as sesame
    from models import Model

    torch.set_num_threads(4)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (7, 5):
        raise RuntimeError("Original CSM diagnostic requires T4-or-newer CUDA")
    original_tokenizer = AutoTokenizer.from_pretrained

    class PinnedTokenizer:
        @staticmethod
        def from_pretrained(name, *args, **kwargs):
            if name != "meta-llama/Llama-3.2-1B":
                raise RuntimeError("Unexpected tokenizer source")
            return original_tokenizer(os.environ["CSM_TOKENIZER_DIR"], *args,
                                      local_files_only=True, **kwargs)

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
    load_begin = time.monotonic()
    model = Model.from_pretrained(os.environ["CSM_ORIGINAL_DIR"])
    model = model.to(device="cuda:0", dtype=torch.float32).eval()
    generator = sesame.Generator(model)
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - load_begin
    if model.config.audio_num_codebooks != EXPECTED_CODEBOOKS or torch.cuda.memory_allocated(0) < 1024**3:
        raise RuntimeError("Original GPU model or codebook count differs from paired v2")
    output_dir = Path(os.environ["CSM_OUTPUT_DIR"])
    calls = []
    with GpuPidWatch() as watch:
        for phase, pairs in (("warmup", 1), ("measured", MEASURED_PAIRS)):
            for pair in range(pairs):
                for timed in ((False, True) if pair % 2 == 0 else (True, False)):
                    label = f"{phase}-{pair}-{'on' if timed else 'off'}"
                    torch.manual_seed(SEED)
                    torch.cuda.manual_seed_all(SEED)
                    mimi_original = generator._audio_tokenizer.decode
                    mimi_ms = None

                    def timed_mimi(*args, **kwargs):
                        nonlocal mimi_ms
                        torch.cuda.synchronize()
                        start = time.perf_counter_ns()
                        result = mimi_original(*args, **kwargs)
                        torch.cuda.synchronize()
                        mimi_ms = (time.perf_counter_ns() - start) / 1e6
                        return result

                    if timed:
                        generator._audio_tokenizer.decode = timed_mimi
                    before_samples = len(watch.samples)
                    before_sm = len(watch.sm_samples)
                    try:
                        with FrameProfile(model, torch, timed) as profile:
                            torch.cuda.synchronize()
                            start = time.perf_counter_ns()
                            audio = generator.generate(text=PHRASE, speaker=0, context=[],
                                                       max_audio_length_ms=MAX_AUDIO_MS,
                                                       temperature=TEMPERATURE, topk=TOPK)
                            torch.cuda.synchronize()
                            generate_ms = (time.perf_counter_ns() - start) / 1e6
                    finally:
                        generator._audio_tokenizer.decode = mimi_original
                    samples = torch.stack(profile.samples).detach().cpu().contiguous()
                    eos = (samples == 0).all(dim=-1).reshape(-1)
                    if not bool(eos[-1].item()) or bool(eos[:-1].any().item()):
                        raise RuntimeError("EOS must be the last attempted frame")
                    eos_count = int(eos.sum().item())
                    accepted = int((~eos).sum().item())
                    validate_frames(profile.rows, accepted, eos_count, model.config.audio_num_codebooks, timed)
                    if timed and (mimi_ms is None or mimi_ms <= 0):
                        raise RuntimeError("Original Mimi decode timing missing")
                    wav = output_dir / (label + ".wav")
                    torchaudio.save(str(wav), audio.float().cpu().unsqueeze(0), generator.sample_rate)
                    row = {
                        "label": label, "phase": phase, "pair": pair,
                        "arm": "timing_on" if timed else "timing_off",
                        "generate_ms": generate_ms,
                        "attempted_frame_calls": len(profile.rows),
                        "accepted_audio_frames": accepted,
                        "eos_frame_attempts": eos_count,
                        "codebook_count": model.config.audio_num_codebooks,
                        "all_attempted_codes_sha256": hashlib.sha256(samples.numpy().tobytes()).hexdigest(),
                        "gpu_pid_samples_during_generate": len(watch.samples) - before_samples,
                        "gpu_sm_samples_during_generate": watch.sm_samples[before_sm:],
                        "wav": wav_info(wav),
                    }
                    if timed:
                        row["mimi_decode_ms"] = mimi_ms
                        row["frame_calls"] = profile.rows
                        row["frame_summary"] = summarize_frames(profile.rows)
                    calls.append(row)
                    (output_dir / "original-progress.json").write_text(json.dumps({
                        "calls": calls, "load_seconds": load_seconds,
                        "device": str(next(model.parameters()).device),
                        "dtype": str(next(model.parameters()).dtype),
                        "torch": torch.__version__, "torchaudio": torchaudio.__version__,
                    }, indent=2) + "\n")
    if not watch.samples or any(row["gpu_pid_samples_during_generate"] < 1 or
                                not any(row["gpu_sm_samples_during_generate"]) for row in calls):
        raise RuntimeError("Original process GPU activity not observed during every generation")
    validate_call_identity(calls)
    (output_dir / "original-result.json").write_text(json.dumps({
        "calls": calls, "load_seconds": load_seconds,
        "device": str(next(model.parameters()).device), "dtype": str(next(model.parameters()).dtype),
        "torch": torch.__version__, "torchaudio": torchaudio.__version__,
        "gpu_pid": os.getpid(), "gpu_pid_samples": len(watch.samples),
        "peak_gpu_mib": max(watch.samples), "gpu_allocated_bytes": torch.cuda.memory_allocated(0),
        "output_identity": True,
    }, indent=2) + "\n")


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    def preserve_private_evidence():
        destination = OUT / "private-evidence"
        destination.mkdir(exist_ok=True)
        for pattern in ("*.wav", "*.log", "original-progress.json", "original-result.json"):
            for path in WORK.glob(pattern):
                if path.is_file():
                    shutil.copy2(path, destination / path.name)

    atexit.register(preserve_private_evidence)
    os.environ.update(TMPDIR=str(WORK), HF_HOME=str(WORK / "hf"), HF_XET_CACHE=str(WORK / "xet"),
                      CUDA_VISIBLE_DEVICES="0", NO_TORCH_COMPILE="1", OMP_NUM_THREADS="4",
                      OPENBLAS_NUM_THREADS="1", USE_TF="0")
    receipt = {
        "version": VERSION, "complete": False, "acceptance": False,
        "scope": "private original-only F32 CSM forward-step timing, SilentCipher watermark disabled",
        "pins": {"runtime": RUNTIME, "asr": ASR, "sesame_source": SESAME_SOURCE,
                 "silentcipher_source": SILENTCIPHER_SOURCE, "csm_original": CSM_REV,
                 "llama_tokenizer": LLAMA_REV, "mimi": MIMI_REV,
                 "silentcipher_weights_used": False},
        "settings": {"text": PHRASE, "speaker": 0, "context": [], "seed": SEED,
                     "temperature": TEMPERATURE, "topk": TOPK, "original_max_audio_ms": MAX_AUDIO_MS},
        "timing_boundary": "generate_ms excludes model load and WAV save; frame and module timers synchronize CUDA at boundaries; nested timers include synchronization overhead and are not additive; Mimi decode is separate from frame generation",
        "runs": [], "resource_checks": [],
    }

    def save():
        (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def check(label):
        receipt["resource_checks"].append({"label": label, **headroom()})
        save()

    save()
    receipt["hardware"] = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    if float(receipt["hardware"].split(",")[1]) < 7.5:
        receipt["inconclusive_reason"] = "Pinned Torch CUDA environment requires T4 or newer"
        save()
        print("P100_UNSUPPORTED_ORIGINAL", flush=True)
        return
    check("admission")
    clone_pin("https://github.com/CrispStrobe/CrispASR.git", HARNESS_SOURCE, WORK / "harness")
    sys.path.insert(0, str(WORK / "harness/tools/kaggle"))
    import kaggle_harness as kh
    kh.init_progress(hf_progress_repo="")
    os.environ["HF_TOKEN"] = kh.kaggle_token_from_dataset("hf_token.txt") or kh.resolve_hf_token(require=True)
    clone_pin("https://github.com/SesameAILabs/csm.git", SESAME_SOURCE, WORK / "sesame")
    check("before-dependencies")
    # Same pinned original environment as paired v2; do not replace Kaggle Torch.
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--only-binary=:all:",
                    "-q", "bitsandbytes==0.48.2"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-q",
                    "tokenizers==0.21.0", "transformers==4.49.0", "huggingface_hub==0.28.1",
                    "moshi==0.2.2", "torchtune==0.4.0", "torchao==0.9.0",
                    "git+https://github.com/SesameAILabs/silentcipher@" + SILENTCIPHER_SOURCE], check=True)
    import torch
    import torchaudio
    if (not torch.cuda.is_available() or not torch.version.cuda
            or torch.__version__.split("+")[0] != torchaudio.__version__.split("+")[0]):
        raise RuntimeError("Kaggle Torch/Torchaudio CUDA pair unavailable")
    receipt["software"] = {"torch": torch.__version__, "torchaudio": torchaudio.__version__}
    from huggingface_hub import hf_hub_download, snapshot_download
    check("before-model-downloads")
    original = Path(snapshot_download("sesame/csm-1b", revision=CSM_REV,
                                      allow_patterns=["*.json", "*.safetensors", "README.md"],
                                      local_dir=WORK / "original"))
    tokenizer = Path(snapshot_download("meta-llama/Llama-3.2-1B", revision=LLAMA_REV,
                                       allow_patterns=["tokenizer*", "*.json", "*.model", "README.md"],
                                       local_dir=WORK / "tokenizer"))
    mimi = Path(hf_hub_download("kyutai/moshiko-pytorch-bf16",
                                "tokenizer-e351c8d8-checkpoint125.safetensors", revision=MIMI_REV,
                                local_dir=WORK / "mimi"))
    mimi_card = Path(hf_hub_download("kyutai/moshiko-pytorch-bf16", "README.md", revision=MIMI_REV,
                                     local_dir=WORK / "mimi"))
    cards = ((original / "README.md", r"(?m)^license: apache-2\.0\s*$"),
             (tokenizer / "README.md", r"(?m)^license: llama3\.2\s*$"),
             (mimi_card, r"(?m)^license: cc-by-4\.0\s*$"))
    for card, pattern in cards:
        if not card.is_file() or not re.search(pattern, card.read_text()):
            raise RuntimeError("Pinned model card license missing or changed: " + card.name)
    receipt["model_card_sha256"] = {"csm": sha(original / "README.md"),
                                     "llama_tokenizer": sha(tokenizer / "README.md"),
                                     "mimi": sha(mimi_card)}
    receipt["original_files"] = {str(p.relative_to(original)): sha(p) for p in original.rglob("*") if p.is_file()}
    receipt["tokenizer_files"] = {str(p.relative_to(tokenizer)): sha(p) for p in tokenizer.rglob("*") if p.is_file()}
    receipt["mimi_sha256"] = sha(mimi)
    archive = download(*RUNTIME, WORK / "runtime.tar.gz")
    runtime = WORK / "runtime"
    runtime.mkdir()
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if any(Path(member.name).is_absolute() or ".." in Path(member.name).parts
               or not (member.isfile() or member.isdir() or member.issym())
               or (member.issym() and (Path(member.linkname).is_absolute()
                                       or ".." in Path(member.linkname).parts)) for member in members):
            raise ValueError("Unsafe native archive")
        bundle.extractall(runtime, filter="data")
    binary = next(runtime.rglob("crispasr"))
    if not binary.is_file() or sha(binary) != "4c07dbc07453e80684786203990ab6e89c1266bc13e6c61ca0e7ea5045444819":
        raise RuntimeError("ASR judge binary differs from accepted v2")
    asr = Path(hf_hub_download(ASR[0], ASR[2], revision=ASR[1], local_dir=WORK / "asr"))
    if sha(asr) != ASR[3]:
        raise RuntimeError("ASR judge model differs from accepted v2")
    env = dict(os.environ, CSM_SESAME_DIR=str(WORK / "sesame"), CSM_ORIGINAL_DIR=str(original),
               CSM_TOKENIZER_DIR=str(tokenizer), CSM_MIMI_PATH=str(mimi), CSM_OUTPUT_DIR=str(WORK),
               GGML_BACKEND_PATH=str(binary.parent),
               LD_LIBRARY_PATH=str(binary.parent) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))
    receipt["entry_sha256"] = sha(__file__)
    save()
    check("before-original-run")
    log = WORK / "original.log"
    with log.open("wb") as output:
        result = subprocess.run([sys.executable, "-B", __file__, "--original-child"],
                                stdout=output, stderr=subprocess.STDOUT, cwd=WORK / "sesame",
                                env=env, timeout=900)
    receipt["original_log_sha256"] = sha(log)
    if result.returncode:
        save()
        raise RuntimeError("Original CSM step child failed; inspect private log")
    details = json.loads((WORK / "original-result.json").read_text())
    if (details["device"] != "cuda:0" or details["dtype"] != "torch.float32"
            or details["peak_gpu_mib"] < 1024 or not details["output_identity"]):
        raise RuntimeError("Original GPU, precision or output-identity gate failed")
    if len(details["calls"]) != 2 * (1 + MEASURED_PAIRS):
        raise RuntimeError("Unexpected original call count")
    receipt["original"] = {key: value for key, value in details.items() if key != "calls"}
    for row in details["calls"]:
        label = row["label"]
        wav = WORK / (label + ".wav")
        if sha(wav) != row["wav"]["sha256"]:
            raise RuntimeError("Original WAV changed before ASR judging")
        judge_log = WORK / (label + "-asr.log")
        with judge_log.open("wb") as output:
            judge = subprocess.run([binary, "--backend", "parakeet", "-m", asr, "-f", wav,
                                    "--gpu-backend", "cuda", "--no-prints", "-t", "4"],
                                   stdout=output, stderr=subprocess.STDOUT, cwd=binary.parent,
                                   env=env, timeout=300)
        if judge.returncode:
            raise RuntimeError("Pinned ASR judge failed: " + label)
        lines = [line.strip() for line in judge_log.read_text(errors="replace").splitlines() if line.strip()]
        transcript = re.sub(r"(?i)^this audio was generated by artificial intelligence[. ]*", "", lines[-1])
        row["asr_wer"] = wer(transcript)
        row["asr_log_sha256"] = sha(judge_log)
        receipt["runs"].append(row)
        save()
        if row["asr_wer"] > 0.15:
            raise RuntimeError("CSM speech quality gate failed: " + label)
    measured = [row for row in receipt["runs"] if row["phase"] == "measured"]
    receipt["measured_medians"] = {
        "timing_off_generate_ms": statistics.median(row["generate_ms"] for row in measured
                                                     if row["arm"] == "timing_off"),
        "timing_on_generate_ms": statistics.median(row["generate_ms"] for row in measured
                                                    if row["arm"] == "timing_on"),
        "prefill_frame_full_ms": statistics.median(row["frame_summary"]["prefill_frame_full_ms"]
                                                    for row in measured if row["arm"] == "timing_on"),
        "prefill_backbone_forward_ms": statistics.median(row["frame_summary"]["prefill_backbone_forward_ms"]
                                                           for row in measured if row["arm"] == "timing_on"),
        "later_frame_full_ms": statistics.median(row["frame_summary"]["later_frame_full_median_ms"]
                                                 for row in measured if row["arm"] == "timing_on"),
        "later_backbone_forward_ms": statistics.median(row["frame_summary"]["later_backbone_forward_median_ms"]
                                                        for row in measured if row["arm"] == "timing_on"),
        "later_depth_31_forward_total_ms": statistics.median(
            row["frame_summary"]["later_depth_31_forward_total_median_ms"] for row in measured
            if row["arm"] == "timing_on"),
        "mimi_decode_ms": statistics.median(row["mimi_decode_ms"] for row in measured
                                            if row["arm"] == "timing_on"),
    }
    receipt["complete"] = receipt["acceptance"] = True
    save()
    print("CSM_ORIGINAL_STEPS_ACCEPTED", json.dumps(receipt["measured_medians"], sort_keys=True), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--original-child"]:
        original_child()
    else:
        main()
