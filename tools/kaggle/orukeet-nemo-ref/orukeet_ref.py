#!/usr/bin/env python3
"""CrispASR #445 — Orukeet NeMo reference.

Orukeet (oruk/orukeet, r3) is a parakeet-tdt-0.6b-v3 fine-tune with the
architecture unchanged. Its .nemo is 2.5 GB F32, too big for the dev VPS, so
this CPU kernel produces the reference the local diff runs against:

  * NeMo greedy TDT transcripts for jfk.wav plus de/fr/es gTTS clips
  * a crispasr-diff -ref.gguf (mel, pre-encode, every encoder layer, encoder
    output) for jfk.wav and the German clip

Everything lands in /kaggle/working/out/, including the exact WAVs used, so
the local run feeds the identical samples. Nothing is validated here.
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
for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(line_buffering=True)
    except Exception:
        pass

WORK = Path("/kaggle/working")
OUT = WORK / "out"
OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"
REF = os.environ.get("CRISPASR_REF", "main")
NEMO_REPO = "oruk/orukeet"
NEMO_REV = "555136b50265a132d4cea0d35560c26fc4f657ab"  # r3, as pinned in the report
NEMO_FILE = "orukeet-v0.1.0.nemo"
NEMO_SHA256 = "031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56"

CLIPS = {
    "de": "Guten Morgen. Die Sitzung beginnt heute um neun Uhr im großen Saal.",
    "fr": "Bonjour à tous. La réunion commence demain matin à huit heures.",
    "es": "Buenas tardes. El tren para Madrid sale a las cinco y media.",
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
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "nemo_toolkit[asr]", "gtts", "gguf"])

    step("audio")
    import soundfile as sf
    from gtts import gTTS
    wavs = {"en_jfk": str(REPO / "samples" / "jfk.wav")}
    for lang, text in CLIPS.items():
        mp3 = WORK / f"{lang}.mp3"
        gTTS(text, lang=lang).save(str(mp3))
        wav = OUT / f"{lang}.wav"
        subprocess.check_call(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3),
                               "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)])
        wavs[lang] = str(wav)
        results["clips"][lang] = {"text_prompt": text}
    results["clips"].setdefault("en_jfk", {})
    for k, p in wavs.items():
        a, sr = sf.read(p)
        results["clips"][k].update({"seconds": round(len(a) / sr, 3), "sr": sr,
                                    "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()})
    save()

    step("download")
    from huggingface_hub import hf_hub_download
    nemo_path = hf_hub_download(NEMO_REPO, NEMO_FILE, revision=NEMO_REV, local_dir="/tmp/orukeet")
    h = hashlib.sha256()
    with open(nemo_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    results["nemo_sha256"] = h.hexdigest()
    results["nemo_sha256_ok"] = h.hexdigest() == NEMO_SHA256
    save()
    if not results["nemo_sha256_ok"]:
        raise SystemExit("checkpoint hash mismatch")

    step("nemo_transcribe")
    import torch
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location="cpu")
    model.eval()
    # Greedy TDT, exactly what the report's evaluation used (greedy-batch).
    try:
        from omegaconf import open_dict
        dc = model.cfg.decoding
        with open_dict(dc):
            dc.strategy = "greedy_batch"
        model.change_decoding_strategy(dc)
    except Exception as e:
        results["errors"].append(f"decoding strategy: {e}")
    results["decoding"] = str(model.cfg.decoding.get("strategy"))
    with torch.no_grad():
        for k, p in wavs.items():
            hyp = model.transcribe([p], batch_size=1)[0]
            text = hyp.text if hasattr(hyp, "text") else str(hyp)
            results["clips"][k]["nemo_text"] = text
            print(f"  {k}: {text}", flush=True)
    save()
    del model
    import gc
    gc.collect()

    step("ref_dump")
    for k in ("en_jfk", "de"):
        out = OUT / f"orukeet-{k}-ref.gguf"
        r = subprocess.run([sys.executable, str(REPO / "tools" / "dump_reference.py"),
                            "--backend", "parakeet", "--model-dir", nemo_path,
                            "--audio", wavs[k], "--output", str(out)],
                           capture_output=True, text=True)
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
