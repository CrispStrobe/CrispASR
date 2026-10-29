#!/usr/bin/env python3
"""GPU-only Granite Speech5 original F16 forward-step timing diagnostic.

This is a follow-up to granite5-original-paired-v3, not a native-vs-original
speed comparison. Hooking and synchronizing forwards perturbs the timed run;
the no-hook arm measures that perturbation on the same loaded model and input.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import wave


VERSION = "granite5-original-steps-v1"
ROOT = Path("/kaggle/working")
WORK = Path("/kaggle/temp/granite5-original-steps")
INPUT = (
    "https://raw.githubusercontent.com/CrispStrobe/CrispASR/"
    "cc114d2b4a5a9413738f390093e9131576743cb6/samples/jfk.wav",
    "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e",
)
ORIGINAL = ("ibm-granite/granite-speech-5.0-2b", "c075cf19e041a2e4d16c5ee24763e5b037d8f479")
TRANSFORMERS_VERSION = "4.57.3"
INSTRUCTION = "transcribe the speech with proper punctuation and capitalization."
REFERENCE = "and so my fellow americans ask not what your country can do for you ask what you can do for your country"
MAX_NEW_TOKENS = 96
THREADS = 4


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
    for token in left:
        next_row = [row[0] + 1]
        for j, other in enumerate(right, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (token != other)))
        row = next_row
    return row[-1] / len(left)


def worker_resources():
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    state = {
        "available_bytes": int(memory["MemAvailable"].split()[0]) * 1024,
        "scratch_free_bytes": shutil.disk_usage(WORK).free,
        "load": list(os.getloadavg()),
        "cpus": os.cpu_count() or 1,
    }
    if (state["available_bytes"] < 3 * 1024**3 or state["scratch_free_bytes"] < 9 * 1024**3
            or state["load"][0] > 2 * state["cpus"]):
        raise RuntimeError("Insufficient GPU worker resources")
    return state


def download(url, expected, target):
    with urllib.request.urlopen(url, timeout=180) as source, target.open("wb") as dest:
        shutil.copyfileobj(source, dest, 1024 * 1024)
    if sha(target) != expected:
        raise ValueError("Pinned input hash mismatch")
    return target


class GpuPidWatch:
    """Observe this process in the CUDA compute-app table during model inference."""

    def __init__(self):
        self.pid = os.getpid()
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self):
        while not self.stop.is_set():
            probe = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            if probe.returncode == 0:
                for line in probe.stdout.splitlines():
                    cells = [cell.strip() for cell in line.split(",")]
                    if len(cells) == 2 and cells[0] == str(self.pid) and cells[1].isdigit():
                        self.samples.append(int(cells[1]))
            self.stop.wait(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


class ForwardProfile:
    """Synchronize at real module forward boundaries; record actual call shapes."""

    def __init__(self, model, torch):
        self.model = model
        self.torch = torch
        self.handles = []
        self.rows = []
        self.active = None
        self.child_start = {}

    @staticmethod
    def _sequence_length(kwargs):
        for name in ("input_ids", "inputs_embeds"):
            value = kwargs.get(name)
            if value is not None:
                return int(value.shape[-2 if name == "inputs_embeds" else -1])
        return None

    def _sync_time(self):
        self.torch.cuda.synchronize()
        return time.perf_counter_ns()

    def _root_pre(self, module, args, kwargs):
        if self.active is not None:
            raise RuntimeError("Nested Granite root forward")
        feature = kwargs.get("input_features")
        length = self._sequence_length(kwargs)
        kind = "prefill" if feature is not None else "decode"
        if kind == "decode" and length != 1:
            raise RuntimeError(f"Expected one-token cached decode, observed length {length}")
        if kind == "prefill" and (length is None or length < 2):
            raise RuntimeError(f"Unexpected prefill length {length}")
        self.active = {
            "kind": kind,
            "input_sequence_length": length,
            "audio_feature_shape": list(feature.shape) if feature is not None else None,
            "encoder_calls": 0,
            "projector_calls": 0,
            "language_model_calls": 0,
        }
        self.active["_start_ns"] = self._sync_time()

    def _root_post(self, module, args, kwargs, output):
        if self.active is None:
            raise RuntimeError("Root post-hook without pre-hook")
        end = self._sync_time()
        row = self.active
        row["root_forward_ms"] = (end - row.pop("_start_ns")) / 1e6
        self.rows.append(row)
        self.active = None

    def _child_pre(self, name):
        def hook(module, args, kwargs):
            if self.active is None or name in self.child_start:
                raise RuntimeError(f"Unexpected {name} forward nesting")
            self.child_start[name] = self._sync_time()
        return hook

    def _child_post(self, name):
        def hook(module, args, kwargs, output):
            if self.active is None or name not in self.child_start:
                raise RuntimeError(f"Unexpected {name} forward completion")
            end = self._sync_time()
            duration = (end - self.child_start.pop(name)) / 1e6
            self.active[name + "_calls"] += 1
            self.active[name + "_ms"] = duration
        return hook

    def __enter__(self):
        self.handles.append(self.model.register_forward_pre_hook(self._root_pre, with_kwargs=True))
        self.handles.append(self.model.register_forward_hook(self._root_post, with_kwargs=True))
        for name, module in (("encoder", self.model.encoder), ("projector", self.model.projector),
                             ("language_model", self.model.language_model)):
            self.handles.append(module.register_forward_pre_hook(self._child_pre(name), with_kwargs=True))
            self.handles.append(module.register_forward_hook(self._child_post(name), with_kwargs=True))
        return self

    def __exit__(self, *_):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        if self.active is not None or self.child_start:
            raise RuntimeError("Incomplete forward hook trace")


def validate_profile(rows):
    if len(rows) < 2 or rows[0]["kind"] != "prefill" or any(r["kind"] != "decode" for r in rows[1:]):
        raise RuntimeError("Expected one prefill followed by observed cached-decode forwards")
    for index, row in enumerate(rows):
        prefill = index == 0
        if (row["encoder_calls"] != int(prefill) or row["projector_calls"] != int(prefill)
                or row["language_model_calls"] != 1):
            raise RuntimeError("Granite forward structure differs from pinned Transformers 4.57.3 source")
        for key in ("root_forward_ms", "language_model_ms"):
            if row[key] <= 0:
                raise RuntimeError("Invalid synchronized forward duration")
        if prefill and (row.get("encoder_ms", 0) <= 0 or row.get("projector_ms", 0) <= 0):
            raise RuntimeError("Missing audio prefill stage timing")


def summarize_profile(rows):
    validate_profile(rows)
    prefill = rows[0]
    decode = rows[1:]
    return {
        "actual_root_forward_calls": len(rows),
        "actual_prefill_calls": 1,
        "actual_single_token_decode_calls": len(decode),
        "prefill_root_ms": prefill["root_forward_ms"],
        "prefill_encoder_ms": prefill["encoder_ms"],
        "prefill_projector_ms": prefill["projector_ms"],
        "prefill_language_model_ms": prefill["language_model_ms"],
        "decode_root_total_ms": sum(r["root_forward_ms"] for r in decode),
        "decode_root_median_ms_per_call": statistics.median(r["root_forward_ms"] for r in decode),
        "decode_language_model_median_ms_per_call": statistics.median(r["language_model_ms"] for r in decode),
    }


def run_generation(model, processor, inputs, torch, profile):
    profiler = ForwardProfile(model, torch) if profile else None
    torch.cuda.synchronize()
    begin = time.perf_counter_ns()
    with torch.inference_mode():
        if profiler is None:
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
        else:
            with profiler:
                output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
    torch.cuda.synchronize()
    generate_ms = (time.perf_counter_ns() - begin) / 1e6
    new_tokens = output[0, inputs["input_ids"].shape[-1]:].detach().to("cpu")
    transcript = processor.tokenizer.batch_decode(
        new_tokens.unsqueeze(0), add_special_tokens=False, skip_special_tokens=True,
    )[0]
    row = {
        "arm": "hooks_on" if profile else "hooks_off",
        "generate_ms": generate_ms,
        "generated_token_count": int(new_tokens.numel()),
        "generated_token_ids_sha256": hashlib.sha256(new_tokens.numpy().tobytes()).hexdigest(),
        "transcript": transcript,
        "wer": wer(REFERENCE, transcript),
    }
    if profiler is not None:
        row["forward_calls"] = profiler.rows
        row["forward_summary"] = summarize_profile(profiler.rows)
    return row


def main():
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in (INPUT[1],)) or not re.fullmatch(
            r"[0-9a-f]{40}", ORIGINAL[1]):
        raise ValueError("Invalid immutable pin")
    ROOT.mkdir(exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HOME=str(WORK / "hf"), HF_XET_CACHE=str(WORK / "xet"),
                      TMPDIR=str(WORK), CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS=str(THREADS),
                      OPENBLAS_NUM_THREADS="1", USE_TF="0")
    receipt = {
        "version": VERSION, "complete": False, "acceptance": False,
        "scope": "Original F16 CUDA only; synchronized real forward hooks and alternating hooks-off/on calls on one loaded model",
        "pins": {"input_sha256": INPUT[1], "original_repo": ORIGINAL[0],
                 "original_revision": ORIGINAL[1], "transformers": TRANSFORMERS_VERSION},
        "settings": {"instruction": INSTRUCTION, "max_new_tokens": MAX_NEW_TOKENS,
                     "do_sample": False, "num_beams": 1, "threads": THREADS},
        "timing_boundary": "generate_ms excludes model loading and processor work; hooked root/encoder/projector/language-model forwards synchronize CUDA before and after each call; nested root duration includes hook synchronization overhead",
        "runs": [],
    }

    def save():
        (ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    receipt["hardware"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"], text=True,
    ).strip()
    cap = float(receipt["hardware"].splitlines()[0].split(",")[1].strip())
    if cap < 7.5:
        receipt["inconclusive_reason"] = "Pinned PyTorch CUDA wheel requires T4 or newer"
        save()
        print("P100_UNSUPPORTED_ORIGINAL", flush=True)
        return
    receipt["resources_before"] = worker_resources()
    save()
    audio_path = download(*INPUT, WORK / "input.wav")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.3",
                    "huggingface_hub>=0.36,<2", "soundfile", "accelerate"], check=True)
    from importlib.metadata import version
    if version("transformers") != TRANSFORMERS_VERSION:
        raise RuntimeError("Unexpected Transformers version")
    from huggingface_hub import snapshot_download
    original = Path(snapshot_download(
        ORIGINAL[0], revision=ORIGINAL[1],
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja", "README.md"],
        local_dir=WORK / "original",
    ))
    if not list(original.glob("*.safetensors")) or not (original / "chat_template.jinja").is_file():
        raise RuntimeError("Pinned original checkpoint or chat template missing")
    card = original / "README.md"
    if not card.is_file() or not re.search(r"(?m)^license: apache-2\.0\s*$", card.read_text()):
        raise RuntimeError("Pinned original model license card missing or changed")
    receipt["original_files"] = {str(path.relative_to(original)): sha(path) for path in original.rglob("*")
                                  if path.is_file() and ".cache" not in path.parts}
    receipt["license"] = {"model_card": "Apache-2.0", "model_card_sha256": sha(card),
                          "source": ORIGINAL[0] + "/blob/" + ORIGINAL[1] + "/README.md",
                          "weights_or_outputs_redistributed": False}
    save()

    import numpy as np
    import torch
    from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration
    torch.set_num_threads(THREADS)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (7, 5):
        raise RuntimeError("Expected T4-or-newer CUDA device")
    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        original, local_files_only=True, torch_dtype=torch.float16, device_map="cuda:0",
    ).eval()
    processor = AutoProcessor.from_pretrained(original, local_files_only=True)
    if model.device.type != "cuda" or model.dtype != torch.float16 or any(
            param.device.type != "cuda" or (param.is_floating_point() and param.dtype != torch.float16)
            for param in model.parameters()):
        raise RuntimeError("Original model has CPU or non-F16 parameters")
    with wave.open(str(audio_path)) as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
            raise RuntimeError("Pinned audio format changed")
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise RuntimeError("Invalid input audio")
    prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": "<|audio|>" + INSTRUCTION}], tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(prompt, torch.from_numpy(audio).unsqueeze(0), device="cpu", return_tensors="pt")
    inputs = {key: value.to(model.device, dtype=model.dtype if value.is_floating_point() else value.dtype)
              for key, value in inputs.items()}
    receipt["input_shapes"] = {key: list(value.shape) for key, value in inputs.items()}
    receipt["device"] = str(model.device)
    receipt["dtype"] = str(model.dtype)
    receipt["torch"] = torch.__version__
    save()

    with GpuPidWatch() as watch:
        # Warm each arm once. Alternate order for three measured pairs.
        for phase, pairs in (("warmup", 1), ("measured", 3)):
            for pair in range(pairs):
                order = (False, True) if pair % 2 == 0 else (True, False)
                for profile in order:
                    row = run_generation(model, processor, inputs, torch, profile)
                    row.update(phase=phase, pair=pair)
                    receipt["runs"].append(row)
                    save()
                    if row["wer"] != 0.0 or row["generated_token_count"] < 10:
                        raise RuntimeError("Original output quality or token-work gate failed")
    receipt["gpu_pid"] = os.getpid()
    receipt["gpu_pid_sample_count"] = len(watch.samples)
    receipt["gpu_pid_peak_memory_mib"] = max(watch.samples, default=0)
    if not watch.samples or receipt["gpu_pid_peak_memory_mib"] < 1024:
        raise RuntimeError("CUDA PID and model-residency gate failed")
    identities = {(r["generated_token_count"], r["generated_token_ids_sha256"], r["transcript"])
                  for r in receipt["runs"]}
    if len(identities) != 1:
        raise RuntimeError("Timing-off/on output identity failed")
    measured = [r for r in receipt["runs"] if r["phase"] == "measured"]
    receipt["measured_medians"] = {
        "hooks_off_generate_ms": statistics.median(r["generate_ms"] for r in measured if r["arm"] == "hooks_off"),
        "hooks_on_generate_ms": statistics.median(r["generate_ms"] for r in measured if r["arm"] == "hooks_on"),
        "prefill_root_ms": statistics.median(r["forward_summary"]["prefill_root_ms"] for r in measured
                                                if r["arm"] == "hooks_on"),
        "encoder_ms": statistics.median(r["forward_summary"]["prefill_encoder_ms"] for r in measured
                                        if r["arm"] == "hooks_on"),
        "projector_ms": statistics.median(r["forward_summary"]["prefill_projector_ms"] for r in measured
                                          if r["arm"] == "hooks_on"),
        "single_token_decode_root_ms_per_call": statistics.median(
            r["forward_summary"]["decode_root_median_ms_per_call"] for r in measured if r["arm"] == "hooks_on"),
        "single_token_decode_language_model_ms_per_call": statistics.median(
            r["forward_summary"]["decode_language_model_median_ms_per_call"] for r in measured
            if r["arm"] == "hooks_on"),
    }
    receipt["measured_pairs"] = [
        {"pair": pair, "hooks_off_generate_ms": next(r["generate_ms"] for r in measured
                                                     if r["pair"] == pair and r["arm"] == "hooks_off"),
         "hooks_on_generate_ms": next(r["generate_ms"] for r in measured
                                      if r["pair"] == pair and r["arm"] == "hooks_on")}
        for pair in range(3)
    ]
    receipt["complete"] = receipt["acceptance"] = True
    save()
    print("GRANITE5_ORIGINAL_STEPS_ACCEPTED", json.dumps(receipt["measured_medians"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
