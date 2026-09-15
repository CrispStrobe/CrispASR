#!/usr/bin/env python3
"""Breeze TTS 2 (#412): HF safetensors -> f16 GGUF -> q8_0/q4_k, uploaded to HF.

WHY KAGGLE: the checkpoint is 6.97 GB bf16 and the f16 GGUF is ~5.7 GB. The
8 GB VPS can neither hold it nor build the quantizer beside it. No GPU is
needed anywhere in this pipeline, so the kernel runs with enable_gpu=false —
that keeps the GPU quota for work that actually needs a GPU, and sidesteps the
P100 lottery entirely (gotchas #21-#23).

DISK DISCIPLINE (the ~20 GB scratch will not hold everything at once):
    download src            7.0 GB
    convert -> f16         12.7 GB
    rm src                  5.7 GB   <- before any quantization
    upload f16                       <- the upload IS the checkpoint
    quantize q4_k           7.5 GB
    upload q4_k, rm q4_k    5.7 GB
    quantize q8_0           8.7 GB
    upload q8_0
Every artifact is uploaded the MOMENT it exists, so a later step that OOMs or
times out never loses work already paid for.

LICENSE — this is not boilerplate, it is a condition of redistribution.
Breeze TTS 2 is under the BreezeBlue Research and Non-Commercial License
Agreement, and §1.3 names quantization as producing a Derivative Model, so
these GGUFs inherit the NC terms. §4 requires, on the redistributing repo:
  (a) a complete copy of the Agreement          -> LICENSE, copied from source
  (b) a NOTICE file with the verbatim notice    -> NOTICE
  (c) the "Derived from Breeze TTS 2 ..." line prominently in the model card
  (d) the repo licensed under that Agreement, NOT Apache-2.0
§4 also bars "Breeze"/"BreezeBlue" as the PRIMARY name of the derivative, so
the repo name is descriptive attribution and the CrispASR backend key does not
lead with it. All four are asserted at the end of this run — an obligation
that is only intended is not discharged.

Push (chr1s4):
  export KAGGLE_API_TOKEN=<chr1s4 token>
  python -m kaggle kernels push -p tools/kaggle/breeze-convert
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_VERSION = "2026-09-15.2"
WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else WORK
REPO = WORK / "CrispASR"
BRANCH = os.environ.get("CRISPASR_REF", "feat/412-breeze-tts-2")

SRC_REPO = "BreezeBlue/Breeze-TTS-2"
HF_REPO = "cstr/breeze-tts-2-GGUF"

# §4(b) — must appear verbatim, byte for byte.
LICENSE_NOTICE = (
    "Breeze TTS 2 is licensed under the BreezeBlue Research and Non-Commercial "
    "License Agreement. Copyright (c) 2026 RESONIA, INC. All Rights Reserved."
)
# §4(c) — must appear prominently in the model card.
LICENSE_DERIVED_FROM = (
    "Derived from Breeze TTS 2 by BreezeBlue and licensed for research and "
    "non-commercial use only."
)

print(f"=== breeze-convert {SCRIPT_VERSION} (branch {BRANCH}) ===", flush=True)

# ── clone CrispASR ──────────────────────────────────────────────────────────
# Retried: Kaggle workers have flaky GitHub access (gotcha #18), and this runs
# before kaggle_harness exists, so the retry cannot live in the harness.
if not REPO.exists():
    for attempt in range(4):
        if REPO.exists():
            shutil.rmtree(REPO)
        # --recursive is not optional: ggml and third_party/c2pa-audio are
        # submodules, and cmake fails at configure time without them ("re-clone
        # with git clone --recursive"), which is how run 1 died after a clean
        # 7 GB download and a successful conversion.
        rc = subprocess.run(["git", "clone", "--depth", "1", "--recursive",
                             "--shallow-submodules", "-b", BRANCH,
                             "https://github.com/CrispStrobe/CrispASR",
                             str(REPO)]).returncode
        if rc == 0:
            break
        print(f"clone attempt {attempt + 1} failed rc={rc}; retrying", flush=True)
    else:
        raise SystemExit("clone failed after 4 attempts")
subprocess.check_call(["git", "log", "--oneline", "-1"], cwd=str(REPO))
# Belt and braces: a --recursive clone that partially failed still
# leaves an empty submodule dir, and cmake's error for that is the same.
subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"],
               cwd=str(REPO), check=False)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
if hasattr(kh, "provenance"):
    kh.provenance(SCRIPT_VERSION, clone_dir=REPO)

kh.step("install deps")
kh.sh_with_progress("pip install -q gguf safetensors huggingface_hub hf_transfer")

hf_token = kh.resolve_hf_token()
if not hf_token:
    raise SystemExit("no HF token — nothing could be uploaded, so do not spend "
                     "an hour converting first")
os.environ["HF_TOKEN"] = hf_token
os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402

api = HfApi(token=hf_token)


def df():
    subprocess.call(["df", "-h", str(TEMP), str(WORK)])


def upload(local: Path, name: str):
    print(f"  uploading {name} ({local.stat().st_size / 2**30:.2f} GiB)", flush=True)
    api.upload_file(path_or_fileobj=str(local), path_in_repo=name, repo_id=HF_REPO)
    print(f"  uploaded {name}", flush=True)


# ── download ────────────────────────────────────────────────────────────────
kh.step("download checkpoint")
SRC = TEMP / "breeze-src"
snapshot_download(
    repo_id=SRC_REPO, local_dir=str(SRC), token=hf_token,
    # audio_tokenizer/ is deliberately NOT fetched: its weights are bit-identical
    # to Qwen/Qwen3-TTS-Tokenizer-12Hz, which CrispASR already ships, and the
    # converter skips them anyway. Not downloading 682 MB we would only discard.
    allow_patterns=["*.json", "*.safetensors", "LICENSE", "tokenizer.model"],
    ignore_patterns=["audio_tokenizer/*"],
)
print(f"  src: {sum(p.stat().st_size for p in SRC.rglob('*') if p.is_file()) / 1e9:.2f} GB")
df()

LICENSE_SRC = SRC / "LICENSE"
if not LICENSE_SRC.is_file():
    raise SystemExit("LICENSE not found in the checkpoint — §4(a) requires shipping "
                     "a complete copy of the Agreement, so refuse to publish without it")

# ── repo + license obligations FIRST ────────────────────────────────────────
# Before a single weight goes up, so the terms are never briefly absent from a
# repo that already holds the weights.
kh.step("create repo and discharge LICENSE §4")
api.create_repo(HF_REPO, repo_type="model", exist_ok=True)

CARD = f"""---
license: other
license_name: breezeblue-research-and-non-commercial-license-1.1
license_link: https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE
base_model: BreezeBlue/Breeze-TTS-2
language:
  - en
  - zh
tags:
  - text-to-speech
  - gguf
  - crispasr
  - non-commercial
extra_gated_prompt: >-
  These weights are for research and non-commercial use only.
---

# Breeze TTS 2 — GGUF (CrispASR)

> **{LICENSE_DERIVED_FROM}**

> ⚠️ **NON-COMMERCIAL.** These files are Derivative Models under §1.3 of the
> BreezeBlue Research and Non-Commercial License Agreement, which names
> quantization explicitly. They inherit the licence of the original weights.
> Hosting them as a service is a commercial use under §1.7(b).
> The full Agreement is in [`LICENSE`](./LICENSE); the required notice is in
> [`NOTICE`](./NOTICE).

GGUF conversion of [BreezeBlue/Breeze-TTS-2]({{src}}) for
[CrispASR](https://github.com/CrispStrobe/CrispASR).

| File | Size | Notes |
|---|---|---|
| `breeze-tts-2-f16.gguf` | ~5.7 GB | reference precision |
| `breeze-tts-2-q8_0.gguf` | ~3.0 GB | |
| `breeze-tts-2-q4_k.gguf` | ~2.2 GB | default for the CrispASR registry |

Architecture: a CSM fork — T5Gemma2 text encoder (26L, bidirectional, symmetric
sliding window) → Qwen3 backbone (28L) → depth decoder (12L) over 16 codebooks
at 12.5 Hz. Languages: English and Chinese.

**The codec is not in these files.** Breeze's bundled audio tokenizer is
bit-identical to `Qwen/Qwen3-TTS-Tokenizer-12Hz`, which CrispASR already ships
as [`cstr/qwen3-tts-tokenizer-12hz-GGUF`](https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF).
It is wired in as a registry companion and downloaded alongside the model.
Roughly 1.16 GB of the original checkpoint (`embed_text_tokens` and a leftover
Mimi `codec_model.*`) is unreachable at inference and is dropped by the
converter.

Inference code in the upstream repo is Apache-2.0 and is not covered by the
Agreement (§1.2); the restriction here is on the **weights**.

All credit for the model goes to BreezeBlue / RESONIA, INC.
""".replace("{src}", f"https://huggingface.co/{SRC_REPO}")

api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=HF_REPO)
api.upload_file(path_or_fileobj=str(LICENSE_SRC), path_in_repo="LICENSE", repo_id=HF_REPO)
api.upload_file(path_or_fileobj=(LICENSE_NOTICE + "\n").encode(),
                path_in_repo="NOTICE", repo_id=HF_REPO)
print("  LICENSE + NOTICE + card uploaded", flush=True)

# ── convert ─────────────────────────────────────────────────────────────────
kh.step("convert to f16 GGUF")
OUT = TEMP / "breeze-out"
OUT.mkdir(parents=True, exist_ok=True)
F16 = OUT / "breeze-tts-2-f16.gguf"
subprocess.check_call([
    sys.executable, str(REPO / "models" / "convert-breeze-tts-2-to-gguf.py"),
    "--input", str(SRC),
    "--output", str(F16),
    "--outtype", "f16",
    "--tmpdir", str(OUT),
    "--speaker-identity", "synthetic",
])
print(f"  f16: {F16.stat().st_size / 2**30:.2f} GiB")

# The source is dead weight from here on and the quants need the room.
shutil.rmtree(SRC)
df()
upload(F16, F16.name)

# ── build the quantizer ─────────────────────────────────────────────────────
# crispasr-quantize links only common+ggml+ggml-base (examples/crispasr-quantize/
# CMakeLists.txt), so this is a small CPU build, not the whole 61-backend
# library. No CUDA flags: there is no GPU in this kernel by design.
kh.step("build crispasr-quantize")
kh.install_build_toolchain()
BUILD = REPO / "build"
subprocess.check_call(
    ["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO),
     "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF"]
    + kh.cache_and_link_flags())
# safe_build_jobs() returns a SHELL SNIPPET ("$(nproc)"), so this has to go
# through a shell — passing it to subprocess as an argv element hands the
# literal string "$(nproc)" to -j.
with kh.build_heartbeat("quantize-build"):
    kh.sh_with_progress(
        f"cmake --build {BUILD} --target crispasr-quantize "
        f"-j {kh.safe_build_jobs(gpu=False)}")
QUANT = BUILD / "bin" / "crispasr-quantize"
if not QUANT.is_file():
    QUANT = next(BUILD.rglob("crispasr-quantize"))
print(f"  quantizer: {QUANT}", flush=True)

# ── quantize, upload, delete — one at a time ────────────────────────────────
for qtype in ("q4_k", "q8_0"):
    kh.step(f"quantize {qtype}")
    dst = OUT / f"breeze-tts-2-{qtype}.gguf"
    subprocess.check_call([str(QUANT), str(F16), str(dst), qtype])
    print(f"  {qtype}: {dst.stat().st_size / 2**30:.2f} GiB", flush=True)
    upload(dst, dst.name)
    dst.unlink()
    df()

# ── verify the obligations actually landed ──────────────────────────────────
# An obligation that is only intended is not discharged. Read the repo back.
kh.step("verify")
files = [s.rfilename for s in api.model_info(HF_REPO).siblings]
print(f"  repo files: {files}", flush=True)
missing = [f for f in ("LICENSE", "NOTICE", "README.md",
                       "breeze-tts-2-f16.gguf", "breeze-tts-2-q4_k.gguf",
                       "breeze-tts-2-q8_0.gguf") if f not in files]
assert not missing, f"missing from {HF_REPO}: {missing}"

notice_back = Path(hf_hub_download(HF_REPO, "NOTICE", token=hf_token)).read_text()
assert LICENSE_NOTICE in notice_back, "§4(b): NOTICE does not contain the verbatim notice"
card_back = Path(hf_hub_download(HF_REPO, "README.md", token=hf_token)).read_text()
assert LICENSE_DERIVED_FROM in card_back, "§4(c): card lacks the 'Derived from' line"
info = api.model_info(HF_REPO, expand=["cardData"])
lic = info.card_data.license if info.card_data else None
assert lic == "other", f"§4(d): repo licensed as {lic!r}, must not be apache-2.0"
lic_back = Path(hf_hub_download(HF_REPO, "LICENSE", token=hf_token)).read_text()
assert "Non-Commercial" in lic_back and len(lic_back) > 10000, \
    "§4(a): LICENSE copy looks truncated"
print("  LICENSE §4(a)-(d) all verified against the live repo", flush=True)

kh.step("DONE")
print("CONVERT_OK", flush=True)
