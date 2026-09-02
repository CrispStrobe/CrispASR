#!/usr/bin/env python3
"""Raon-OpenTTS → GGUF conversion + upload (#387-adj, CC-BY-NC-4.0).

Download-and-convert only — no torch reference synthesis, no C++ build. The
16.7 GB 1B checkpoint cannot be loaded on the 8 GB VPS, so conversion runs
here; our converter mmaps the .pt and only materializes the ema DiT tensors,
so a CPU box (30 GB) is plenty and no GPU slot is burned (the roundtrip kernel
needs those). The reference-fixture dump lives in raon-ref-dump; end-to-end
validation is raon-roundtrip. Emits cstr/raon-opentts-<size>-GGUF.

Set RAON_SIZE=1B (default) or 0.3B. CPU kernel; datasets: crispasr-hf-token.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TMP = Path("/kaggle/temp")
TMP.mkdir(parents=True, exist_ok=True)
WORK = Path("/kaggle/working")

SIZE = os.environ.get("RAON_SIZE", "1B")
REPO_MODEL = f"KRAFTON/Raon-OpenTTS-{SIZE}"
CKPT_FILE = {"0.3B": "model_225000.pt", "1B": "model_520000.pt"}[SIZE]
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_REF = os.environ.get("CRISPASR_REF", "feat/raon-opentts-1b")
CLONE = TMP / "CrispASR"


def step(name, **kv):
    print(f"[{time.strftime('%H:%M:%S')}] {name} " + json.dumps(kv), flush=True)


# clone CrispASR (retry: Kaggle GitHub access is flaky, gotcha #18) — only the
# converter script is needed, so no submodules.
if not CLONE.exists():
    for _ in range(4):
        r = subprocess.run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF,
                            CRISPASR_URL, str(CLONE)], timeout=1800)
        if r.returncode == 0:
            break
        time.sleep(15)
    else:
        print("clone failed after retries", flush=True); sys.exit(1)
sys.path.insert(0, str(CLONE / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
HF_TOKEN = kh.resolve_hf_token()
# Small deps only; do NOT reinstall torch/torchaudio (Kaggle's match its arch).
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "huggingface_hub"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "gguf"], check=False)
from huggingface_hub import hf_hub_download, HfApi  # noqa: E402

MODELS = TMP / "models"
MODELS.mkdir(exist_ok=True)


def dl(repo, fname, sub=""):
    d = MODELS / sub if sub else MODELS
    d.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(repo, fname, local_dir=str(d), token=HF_TOKEN or None)


with kh.build_heartbeat("download", interval_s=30):
    ckpt = dl(REPO_MODEL, CKPT_FILE, SIZE)
    cfg = dl(REPO_MODEL, "config.yaml", SIZE)
    vocab = dl(REPO_MODEL, "vocab.txt", SIZE)
    gen_ckpt = dl("speechbrain/tts-hifigan-libritts-16kHz", "generator.ckpt", "sbhifigan")
step("downloaded", size=SIZE, ckpt_gb=round(os.path.getsize(ckpt) / 1e9, 1))

out_gguf = WORK / f"raon-opentts-{SIZE.lower()}-f16.gguf"
with kh.build_heartbeat("convert", interval_s=30):
    r = subprocess.run(f"{sys.executable} {CLONE}/models/convert-raon-opentts-to-gguf.py "
                       f"--checkpoint {ckpt} --config {cfg} --vocab {vocab} --hifigan {gen_ckpt} "
                       f"--output {out_gguf} --quant f16",
                       shell=True, capture_output=True, text=True, timeout=3600)
print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
step("converted", rc=r.returncode, exists=out_gguf.exists(),
     gguf_gb=round(os.path.getsize(out_gguf) / 1e9, 2) if out_gguf.exists() else 0)
if r.returncode != 0 or not out_gguf.exists():
    step("CONVERT_FAIL"); sys.exit(1)

api = HfApi(token=HF_TOKEN)
repo_id = f"cstr/raon-opentts-{SIZE.lower()}-GGUF"
api.create_repo(repo_id, exist_ok=True)
with kh.build_heartbeat("upload", interval_s=30):
    api.upload_file(path_or_fileobj=str(out_gguf), repo_id=repo_id, path_in_repo=out_gguf.name)
step("uploaded", repo=repo_id, file=out_gguf.name)
(WORK / "raon_convert.json").write_text(json.dumps(
    {"size": SIZE, "gguf": out_gguf.name, "repo": repo_id,
     "gguf_gb": round(os.path.getsize(out_gguf) / 1e9, 2)}, indent=2))
step("DONE")
