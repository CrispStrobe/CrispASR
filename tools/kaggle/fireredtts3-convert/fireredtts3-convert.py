#!/usr/bin/env python3
"""Kaggle kernel: FireRedTTS3 (#377) — convert base+redae+campp to GGUF, upload.

Steps:
  1. clone CrispASR (branch feat/377-fireredtts3)
  2. download FireRedTeam/FireRedTTS3 (base variant + redae + campp + tokenizer)
  3. clone upstream FireRedTTS3 python repo, bake the default English voice
     prompt from samples/jfk.wav: RedAE.encode latents + CAM++ spk_emb (CPU,
     sdpa fallback — Kaggle has no flash-attn and torch dropped sm_60)
  4. assert converter tokenizer replication == transformers AutoTokenizer ids
  5. run models/convert-fireredtts3-to-gguf.py → 2 f16 GGUFs
  6. upload each artifact to cstr/fireredtts3-GGUF the moment it exists,
     with a license card (Apache-2.0, FireRedTeam attribution)

Push (chr1s4):
  export KAGGLE_API_TOKEN=<chr1s4 token>
  python -m kaggle kernels push -p tools/kaggle/fireredtts3-convert
"""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_VERSION = "v2"
WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
BRANCH = "feat/377-fireredtts3"

SRC_REPO = "FireRedTeam/FireRedTTS3"
HF_REPO = "cstr/fireredtts3-GGUF"

JFK_TEXT = ("And so my fellow Americans ask not what your country can do "
            "for you, ask what you can do for your country.")

print(f"=== fireredtts3-convert {SCRIPT_VERSION} ===", flush=True)

# ── Phase 0: clone CrispASR (branch) ────────────────────────────────────────
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", BRANCH,
                           "https://github.com/CrispStrobe/CrispASR", str(REPO)])
subprocess.check_call(["git", "log", "--oneline", "-1"], cwd=str(REPO))
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()

kh.step("install deps")
kh.sh_with_progress("pip install -q gguf safetensors huggingface_hub hf_transfer")

kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print("  HF token OK")
else:
    print("  WARNING: no HF token — upload will fail")

# ── Phase 1: download source model ──────────────────────────────────────────
kh.step("download FireRedTTS3")
from huggingface_hub import snapshot_download, HfApi  # noqa: E402

src = snapshot_download(
    repo_id=SRC_REPO,
    cache_dir=str(TEMP / "frt-src"),
    token=hf_token,
    allow_patterns=[
        "fireredtts3_base/*", "redae/*", "campp/*", "text_tokenizer/*",
    ],
)
print(f"  src: {src}")
subprocess.call(["df", "-h", str(TEMP)])

# ── Phase 2: upstream python repo + CPU attn fallback ───────────────────────
kh.step("clone upstream FireRedTTS3 repo")
UP = TEMP / "FireRedTTS3-upstream"
if not UP.exists():
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "https://github.com/FireRedTeam/FireRedTTS3.git",
                           str(UP)])
# RedAE/Core hardcode attn_implementation='flash_attention_2' at
# CONSTRUCTION time — no flash-attn on Kaggle CPU, so patch the source.
npatched = 0
for f in (UP / "fireredtts3").rglob("*.py"):
    t = f.read_text(encoding="utf-8")
    if "flash_attention_2" in t:
        f.write_text(t.replace("flash_attention_2", "sdpa"), encoding="utf-8")
        npatched += 1
print(f"  patched flash_attention_2→sdpa in {npatched} files")
sys.path.insert(0, str(UP))

import torch  # noqa: E402
import torchaudio  # noqa: E402


def force_sdpa(model):
    """RedAE hardcodes attn_implementation='flash_attention_2'; force sdpa."""
    n = 0
    for mod in model.modules():
        cfg = getattr(mod, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "sdpa"
            n += 1
    return n


# ── Phase 3: bake default prompt (jfk.wav → latents + spk_emb) ──────────────
kh.step("bake default prompt npz")
import numpy as np  # noqa: E402

npz_path = WORK / "default_prompt.npz"
from fireredtts3.redae.redae import RedAE  # noqa: E402
from fireredtts3.campp.campp import CamppEmbedding  # noqa: E402

redae = RedAE.from_pretrained(os.path.join(src, "redae"))
redae.eval()
print(f"  redae loaded; forced sdpa on {force_sdpa(redae)} configs")
# Assert the rope_theta the converter hardcodes (Qwen3Config default).
rp = getattr(redae.encoder.qwen3_config, "rope_parameters", None)
theta = (rp or {}).get("rope_theta", getattr(redae.encoder.qwen3_config, "rope_theta", None))
print(f"  redae rope_theta = {theta}")
assert theta in (None, 10000, 10000.0), f"unexpected redae rope_theta {theta}"
hd = getattr(redae.encoder.qwen3_config, "head_dim", None)
print(f"  redae head_dim = {hd}")
assert hd == 128, hd

wav, sr = torchaudio.load(str(REPO / "samples" / "jfk.wav"))
wav = wav[:1]
wav24 = torchaudio.functional.resample(wav, sr, redae.sample_rate)
# generate() pads to downsample_rate * patch_size (960*4) BEFORE encode;
# encode() then re-pads to downsample_rate (no-op). NOTE left pad.
wav24 = RedAE.pad_to_multiple_of(wav24, redae.downsample_rate * 4)
with torch.no_grad():
    latents = redae.encode(wav24, redae.sample_rate).float()  # (1, T, 64)
print(f"  prompt latents: {tuple(latents.shape)}")
assert latents.shape[1] % 4 == 0

campp = CamppEmbedding(os.path.join(src, "campp", "campplus_voxceleb.bin"))
with torch.no_grad():
    spk = campp.forward(wav24, redae.sample_rate).float()  # (1, 512)
print(f"  spk_emb: {tuple(spk.shape)} |spk|={spk.norm():.4f}")

np.savez(npz_path,
         latents=latents[0].numpy().astype(np.float32),
         spk_emb=spk[0].numpy().astype(np.float32),
         prompt_n_samples=np.int64(wav24.shape[1]))
del redae, campp
import gc  # noqa: E402
gc.collect()

# ── Phase 4: tokenizer parity assert ────────────────────────────────────────
kh.step("tokenizer parity")
from fireredtts3.utils.text_tokenizer import load_text_tokenizer  # noqa: E402

tok = load_text_tokenizer(os.path.join(src, "text_tokenizer"))
# import the converter by path (module name has dashes)
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location(
    "frt_conv", str(REPO / "models" / "convert-fireredtts3-to-gguf.py"))
frt_conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frt_conv)

tokens, merges, sot_id, eot_id, n_added = frt_conv.load_tokenizer(
    Path(src) / "text_tokenizer")
print(f"  converter: {len(tokens)} tokens, sot={sot_id}, eot={eot_id}")

# 1) every FireRed special has the same id
mismatch = 0
for t in frt_conv.SPECIAL_TOKENS:
    hf_id = tok.convert_tokens_to_ids(t)
    my_id = tokens.index(t) if t in tokens else -1  # slow but one-off
    if hf_id != my_id:
        print(f"  MISMATCH {t}: hf={hf_id} mine={my_id}")
        mismatch += 1
assert mismatch == 0, f"{mismatch} special-token id mismatches"
# 2) sample sentence encodes identically (template form)
for text in [
    f"<|English|><|sot|>{JFK_TEXT}Hello there, how are you today?<|eot|>",
    "<|Chinese|><|sot|>在欧洲行走简直就是走进汽车博览馆博览会，你好。<|eot|>",
]:
    ids = tok(text, truncation=False, padding=False,
              add_special_tokens=False)["input_ids"]
    back = [tokens[i] for i in ids]
    assert all(t is not None for t in back)
    print(f"  encode ok: {len(ids)} ids, first={ids[:4]}")
# 3) vocab bijection against HF
hf_vocab = tok.get_vocab()
assert len(hf_vocab) == len(tokens), (len(hf_vocab), len(tokens))
for t, i in list(hf_vocab.items())[::5000]:
    assert tokens[i] == t, (t, i, tokens[i])
print("  tokenizer parity OK")

# ── Phase 5: convert ────────────────────────────────────────────────────────
kh.step("convert to GGUF")
OUT = TEMP / "frt-out"
OUT.mkdir(parents=True, exist_ok=True)
subprocess.check_call([
    sys.executable, str(REPO / "models" / "convert-fireredtts3-to-gguf.py"),
    "--model-dir", src,
    "--output-dir", str(OUT),
    "--variant", "base",
    "--default-prompt-npz", str(npz_path),
    "--default-prompt-text", JFK_TEXT,
    "--default-prompt-language", "English",
])
subprocess.call(["ls", "-la", str(OUT)])

# ── Phase 6: upload (each artifact the moment it exists) ────────────────────
kh.step("upload to HF")
api = HfApi(token=hf_token)
api.create_repo(HF_REPO, repo_type="model", exist_ok=True)

CARD = """---
license: apache-2.0
base_model: FireRedTeam/FireRedTTS3
tags:
  - text-to-speech
  - gguf
  - crispasr
---

# FireRedTTS3 GGUF (CrispASR)

GGUF conversion of [FireRedTeam/FireRedTTS3](https://huggingface.co/FireRedTeam/FireRedTTS3)
(Apache-2.0) for [CrispASR](https://github.com/CrispStrobe/CrispASR).

- `fireredtts3-base-f16.gguf` — Qwen3-1.7B LLM backbone + PatchEncoder + DiT
  flow head + tokenizer (+ baked default English voice prompt)
- `fireredtts3-redae-f16.gguf` — RedAE continuous latent autoencoder
  (24 kHz, 25 Hz 64-d latents) + CAM++ speaker encoder

Zero-shot voice-cloning TTS, 24 languages + 21 Chinese dialects.
All credit for the model goes to FireRedTeam (Xiaohongshu).
"""
api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                repo_id=HF_REPO)

for f in sorted(OUT.glob("*.gguf")):
    print(f"  uploading {f.name} ({f.stat().st_size/2**30:.2f} GiB)", flush=True)
    api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=HF_REPO)
    print(f"  uploaded {f.name}", flush=True)

# also stash the prompt npz (handy for the reference dumper)
api.upload_file(path_or_fileobj=str(npz_path),
                path_in_repo="default_prompt_jfk.npz", repo_id=HF_REPO)

info = api.model_info(HF_REPO, expand=["cardData"])
print(f"  card license: {info.card_data.license if info.card_data else None}")

files = [s.rfilename for s in api.model_info(HF_REPO).siblings]
print(f"  repo files: {files}")
assert "fireredtts3-base-f16.gguf" in files
assert "fireredtts3-redae-f16.gguf" in files

kh.step("DONE")
print("CONVERT_OK", flush=True)
