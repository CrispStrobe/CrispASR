#!/usr/bin/env python3
"""CrispASR #445 — Confucius4-R2T2 offline reference.

R2T2 (netease-youdao/Confucius4-R2T2) is a Qwen3-ASR-1.7B fine-tune. Its
offline path is Qwen3ASRModel.transcribe unchanged (R2T2ASRModel.transcribe
just calls super()), so this kernel produces, on CPU in float32:

  * transcripts (raw decode + parsed) for jfk.wav, paraformer_zh.wav and
    de/fr gTTS clips, through qwen_asr's own transformers backend
  * crispasr-diff reference dumps (tools/dump_reference.py --backend qwen3)
    for jfk.wav and paraformer_zh.wav

Outputs land in /kaggle/working/out/ with the exact WAVs used. Streaming is a
separate GPU/vLLM kernel; nothing is validated here.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
WORK = Path("/kaggle/working")
OUT = WORK / "out"
OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"
REF = os.environ.get("CRISPASR_REF", "main")
MODEL = "netease-youdao/Confucius4-R2T2"
MODEL_REV = os.environ.get("R2T2_REV", "185ce639118ad1362d049ca0d8ed04b6ec5cd6c9")

CLIPS = {
    "de": "Guten Morgen. Die Sitzung beginnt heute um neun Uhr im großen Saal.",
    "fr": "Bonjour à tous. La réunion commence demain matin à huit heures.",
}
results = {"clips": {}, "errors": []}


def save():
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))


def main():
    t0 = time.time()

    def step(name):
        print(f"[step] {time.time() - t0:7.1f}s {name}", flush=True)

    step("clone")
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", REF,
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    results["crispasr_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token()
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    step("pip")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "qwen-asr", "gtts", "gguf"])

    step("audio")
    import soundfile as sf
    from gtts import gTTS
    wavs = {"en_jfk": str(REPO / "samples" / "jfk.wav"), "zh": str(REPO / "samples" / "paraformer_zh.wav")}
    for lang, text in CLIPS.items():
        mp3 = WORK / f"{lang}.mp3"
        gTTS(text, lang=lang).save(str(mp3))
        wav = OUT / f"{lang}.wav"
        subprocess.check_call(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3),
                               "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)])
        wavs[lang] = str(wav)
        results["clips"][lang] = {"text_prompt": text}
    for k, p in wavs.items():
        a, sr = sf.read(p)
        results["clips"].setdefault(k, {}).update(
            {"seconds": round(len(a) / sr, 3), "sr": sr,
             "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()})
    save()

    step("download")
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(MODEL, revision=MODEL_REV, local_dir="/tmp/r2t2")
    results["model_rev"] = MODEL_REV
    save()

    step("transcribe")
    import torch
    from qwen_asr import Qwen3ASRModel
    asr = Qwen3ASRModel.from_pretrained(model_dir, dtype=torch.float32, device_map="cpu", max_new_tokens=256)
    for k, p in wavs.items():
        r = asr.transcribe(audio=p, language=None)[0]
        results["clips"][k]["ref_language"] = getattr(r, "language", "")
        results["clips"][k]["ref_text"] = getattr(r, "text", str(r))
        print(f"  {k}: [{results['clips'][k]['ref_language']}] {results['clips'][k]['ref_text']}", flush=True)
        save()
    del asr
    import gc
    gc.collect()

    step("ref_dump")
    env = dict(os.environ, QWEN3_REF_DTYPE="float32")
    for k in ("en_jfk", "zh"):
        out = OUT / f"r2t2-{k}-ref.gguf"
        r = subprocess.run([sys.executable, str(REPO / "tools" / "dump_reference.py"),
                            "--backend", "qwen3", "--model-dir", model_dir,
                            "--audio", wavs[k], "--output", str(out)],
                           capture_output=True, text=True, env=env)
        (OUT / f"refdump-{k}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
        results["clips"][k]["ref_gguf"] = out.name if r.returncode == 0 and out.exists() else None
        if r.returncode != 0:
            results["errors"].append(f"ref_dump {k}: rc={r.returncode}: {r.stderr[-800:]}")
        save()
    step("done")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        results["errors"].append(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        save()
        raise
    finally:
        save()
        # Keep /kaggle/working to the results: a repo clone in the output makes
        # `kaggle kernels output` page through thousands of files and hit 429s.
        import shutil
        shutil.rmtree(REPO, ignore_errors=True)
