#!/usr/bin/env python3
"""Kaggle kernel: convert efwkjn/whisper-ja-anime-v0.3 to CrispASR GGML.

Pipeline:
  1. clone CrispASR feature branch with custom Whisper vocab support
  2. download the HF model + OpenAI Whisper mel assets
  3. convert to F16 legacy GGML .bin using models/convert-h5-to-ggml.py
  4. build crispasr-quantize
  5. produce q8_0 and q4_k quantizations
  6. upload all artifacts to cstr/whisper-ja-anime-v0.3-GGML
"""

import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
REPO = Path("/kaggle/temp/CrispASR")

CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_BRANCH = "feat/issue-258-whisper-custom-vocab"
SRC_REPO = "efwkjn/whisper-ja-anime-v0.3"
HF_REPO = "cstr/whisper-ja-anime-v0.3-GGML"
NAME = "whisper-ja-anime-v0.3"

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["USE_TF"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TMPDIR"] = str(TEMP)

print("=== Phase 0: clone CrispASR feature branch ===", flush=True)
if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--recursive", "--depth", "1", "-b", CRISPASR_BRANCH,
        CRISPASR_URL, str(REPO),
    ])

if str(REPO / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))

import kaggle_harness as kh  # noqa: E402

kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
kh._HF_PUSH_INTERVAL_S = 0.0
kh.step("script.start", src=SRC_REPO, out=HF_REPO, branch=CRISPASR_BRANCH)

kh.step("install.deps")
kh.sh_with_progress(
    "pip install -q transformers safetensors huggingface_hub hf_transfer tiktoken"
)

kh.step("resolve.hf-token")
hf_token = kh.resolve_hf_token()
if not hf_token:
    raise RuntimeError("HF token is required for upload")
os.environ["HF_TOKEN"] = hf_token
os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

api = HfApi(token=hf_token)
try:
    api.create_repo(repo_id=HF_REPO, repo_type="model", private=True, exist_ok=True)
except Exception as exc:
    print(f"repo create warning: {exc}", flush=True)

kh.step("download.model.begin", free_gb=kh.free_gb(str(TEMP)))
model_dir = TEMP / "whisper-ja-anime-src"
model_dir.mkdir(parents=True, exist_ok=True)
model_files = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "normalizer.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "model.safetensors",
]
for filename in model_files:
    kh.step("download.model.file.begin", file=filename)
    path = hf_hub_download(
        repo_id=SRC_REPO,
        filename=filename,
        local_dir=str(model_dir),
        token=hf_token,
    )
    p = Path(path)
    kh.step("download.model.file.done", file=filename, mb=round(p.stat().st_size / (1024 ** 2), 1))
kh.step("download.model.done", model_dir=model_dir)

kh.step("download.openai-whisper-assets")
whisper_repo = TEMP / "openai-whisper"
if not whisper_repo.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/openai/whisper.git", str(whisper_repo),
    ])

out_dir = TEMP / "issue258-out"
out_dir.mkdir(parents=True, exist_ok=True)
f16_path = out_dir / "ggml-model.bin"
if f16_path.exists():
    f16_path.unlink()

kh.step("convert.f16.begin", free_gb=kh.free_gb(str(TEMP)))
with kh.build_heartbeat("convert.f16", interval_s=60):
    kh.sh_with_progress(
        f"python models/convert-h5-to-ggml.py {model_dir} {whisper_repo} {out_dir}",
        cwd=str(REPO),
    )
f16_named = out_dir / f"{NAME}-f16.bin"
f16_path.rename(f16_named)
kh.step("convert.f16.done", gb=round(f16_named.stat().st_size / (1024 ** 3), 3))

readme = out_dir / "README.md"
readme.write_text(
    f"""---
base_model: {SRC_REPO}
tags:
- automatic-speech-recognition
- whisper
- japanese
- crispasr
- ggml
---

# {NAME} GGML for CrispASR

Converted from `{SRC_REPO}` for CrispASR issue #258.

The source model currently does not publish license metadata on Hugging Face.
This repository is created private by the conversion kernel unless a maintainer
explicitly changes visibility after verifying redistribution terms.

This conversion preserves the model's custom 20,480-token Whisper vocabulary by
serializing `added_tokens.json` into the GGML tokenizer table. It requires a
CrispASR build with custom Whisper special-token loading support.

Files:

- `{NAME}-f16.bin`
- `{NAME}-q8_0.bin`
- `{NAME}-q4_k.bin`
""",
    encoding="utf-8",
)

kh.step("build.quantizer")
kh.install_build_toolchain()
build = TEMP / "build-issue258"
flags = kh.cache_and_link_flags()
kh.sh_with_progress(
    f"cmake -G Ninja -B {build} -S {REPO} "
    f"-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF " + " ".join(flags)
)
with kh.build_heartbeat("build.quantizer"):
    kh.sh_with_progress(
        f"cmake --build {build} -j{kh.safe_build_jobs(gpu=False)} --target crispasr-quantize"
    )
quantize_bin = build / "bin" / "crispasr-quantize"
if not quantize_bin.exists():
    raise RuntimeError(f"missing quantizer: {quantize_bin}")

produced = [f16_named, readme]
for quant in ("q8_0", "q4_k"):
    out = out_dir / f"{NAME}-{quant}.bin"
    kh.step(f"quantize.{quant}.begin", free_gb=kh.free_gb(str(TEMP)))
    with kh.build_heartbeat(f"quantize.{quant}", interval_s=60):
        kh.sh_with_progress(f"{quantize_bin} {f16_named} {out} {quant}")
    kh.step(f"quantize.{quant}.done", gb=round(out.stat().st_size / (1024 ** 3), 3))
    produced.append(out)

kh.step("upload.begin", files=[p.name for p in produced])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
api.upload_file(
    path_or_fileobj=str(readme),
    path_in_repo="README.md",
    repo_id=HF_REPO,
    repo_type="model",
    commit_message="Add model card for CrispASR GGML conversion",
)
for path in produced:
    if path.name == "README.md":
        continue
    kh.step("upload.file.begin", file=path.name, gb=round(path.stat().st_size / (1024 ** 3), 3))
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=HF_REPO,
        repo_type="model",
        commit_message=f"Add {path.name}",
    )
    kh.step("upload.file.done", file=path.name)

kh.step("script.done", repo=HF_REPO)
print(f"=== Done: uploaded to https://huggingface.co/{HF_REPO} ===", flush=True)
