#!/usr/bin/env python3
"""Haas-ja-v1.0 through its own upstream inference code (issue #605).

Runs the original NeMo model with the flash-linear-attention KDA kernels
(CUDA only; macOS cannot run them) via haas_asr.transcribe_longform, exactly
as the model card shows, on the #605 reporter's recording. Also:

  * the same chunks in float32, to compare numerics with the CrispASR port;
  * whisper-large-v3 (not turbo) as a second reference transcript, so the
    agreement between two whisper references bounds what any score against
    whisper can reach.

Everything lands in /kaggle/working/haas_blueprint.json. A failed step is
recorded there with its traceback; later steps still run.
"""
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
# transformers would import TensorFlow, whose protobuf clashes with NeMo's.
os.environ["USE_TF"] = "0"
WORK = Path("/kaggle/working")
OUT = WORK / "haas_blueprint.json"
HAAS_ASR_REF = "acc614f"  # upstream commit read for this run
AUDIO_URL = "https://github.com/user-attachments/files/35205014/test-00.00.00.000-00.03.00.723.zip"
result = {"steps": {}}
t0 = time.time()


def save():
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1))


def step(name, fn):
    print(f"[step] {name} t={time.time() - t0:.0f}s", flush=True)
    try:
        result["steps"][name] = {"ok": True, "value": fn()}
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        result["steps"][name] = {"ok": False, "error": tb[-4000:]}
    save()
    return result["steps"][name]


def gpu_info():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True)
    return out.stdout.strip() + " | python " + sys.version.split()[0]


def install():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           f"haas-asr @ git+https://github.com/third-intelligence/Haas-ASR.git@{HAAS_ASR_REF}"])
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    keep = ("nemo-toolkit", "flash-linear-attention", "torch==", "triton", "haas-asr", "transformers")
    return [line for line in out.splitlines() if line.lower().startswith(keep)]


def fetch_audio():
    import urllib.request
    z = WORK / "audio.zip"
    urllib.request.urlretrieve(AUDIO_URL, z)
    with zipfile.ZipFile(z) as f:
        name = f.namelist()[0]
        f.extract(name, WORK)
    return str(WORK / name)


def run_blueprint(wav, dtype_name):
    import torch
    import haas_asr

    dtype = {"bf16": "auto", "fp32": torch.float32}[dtype_name]
    # The first restore in a fresh process has failed with NeMo's "Can't
    # instantiate abstract class ASRModel" (run 1); the same call then works,
    # so retry once.
    try:
        model = haas_asr.load_model("third-intelligence/Haas-ja-v1.0", dtype=dtype)
    except TypeError:
        model = haas_asr.load_model("third-intelligence/Haas-ja-v1.0", dtype=dtype)
    audio = haas_asr.load_audio(wav)
    bounds = haas_asr.split_at_quiet_points(audio)
    chunks = [audio[s:e] for s, e in bounds]
    t = time.time()
    texts = haas_asr.transcribe(model, chunks, batch_size=64)
    joined = haas_asr.transcribe_longform(model, audio, batch_size=64)
    secs = time.time() - t
    del model
    torch.cuda.empty_cache()
    return {
        "dtype": dtype_name,
        "audio_seconds": len(audio) / 16000,
        "chunks": [[s / 16000, e / 16000, txt] for (s, e), txt in zip(bounds, texts)],
        "longform_text": joined,
        "decode_seconds": round(secs, 2),
    }


def run_whisper_large_v3(wav):
    import torch
    from transformers import pipeline
    import haas_asr

    audio = haas_asr.load_audio(wav)
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                    torch_dtype=torch.float16, device="cuda:0")
    out = pipe({"raw": audio, "sampling_rate": 16000}, chunk_length_s=30, batch_size=8,
               return_timestamps=True, generate_kwargs={"language": "ja", "task": "transcribe"})
    return {"text": out["text"], "chunks": [[c["timestamp"][0], c["timestamp"][1], c["text"]]
                                            for c in out.get("chunks", [])]}


def main():
    step("gpu", gpu_info)
    if not step("install", install)["ok"]:
        return
    wav_step = step("audio", fetch_audio)
    if not wav_step["ok"]:
        return
    wav = wav_step["value"]
    step("blueprint_bf16", lambda: run_blueprint(wav, "bf16"))
    step("blueprint_fp32", lambda: run_blueprint(wav, "fp32"))
    step("whisper_large_v3", lambda: run_whisper_large_v3(wav))
    result["total_seconds"] = round(time.time() - t0, 1)
    save()
    print("[done]", json.dumps({k: v["ok"] for k, v in result["steps"].items()}), flush=True)


if __name__ == "__main__":
    main()
