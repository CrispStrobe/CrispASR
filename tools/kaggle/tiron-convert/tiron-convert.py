# CrispASR — Tiron (#295) convert → quantize → reference-dump, checkpointing to HF.
#
# The port pipeline steps 1–8 (see crispasr-crispembed-dev.md "THE PORT
# PIPELINE"). Two iron rules drive the sequencing:
#   * never let disk fill: produce → upload → delete, so at most ~1 big artifact
#     is on the ~70 GB /tmp layer at a time;
#   * never crash before a produced artifact is checkpointed to HF: upload each
#     the moment it exists, before the next (crash-prone) step.
#
#   1. convert Trelis/tiron -> f16 legacy whisper ggml bin
#   2. upload f16 -> cstr/tiron-GGML                                  (checkpoint)
#   3. quantize q4_k with crispasr-LEGACY-quantize (the whisper-bin quantizer;
#      crispasr-quantize is GGUF-only and rc=1s on a legacy bin)
#   4. upload q4_k -> cstr/tiron-GGML                                 (checkpoint)
#   5/6. rm f16 locally (keep q4_k)
#   7. dump tiron-ref.gguf via tools/dump_reference.py --backend tiron
#      (constrained-grammar transcript + mel + encoder_output)
#   8. upload tiron-ref.gguf -> cstr/crispasr-regression-fixtures     (checkpoint)
#
# Validation (crispasr-diff q4_k vs the ref) is LOCAL, not here.
# Datasets (chr1str): chr1str/crispasr-hf-token, chr1str/crispasr-ccache.

import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRATCH = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
WORK = Path("/kaggle/working")
REPO = SCRATCH / "CrispASR"
BUILD = SCRATCH / "build"
MODELS = SCRATCH / "models"
OUT = SCRATCH / "out"
for d in (MODELS, OUT):
    d.mkdir(parents=True, exist_ok=True)

CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
CRISPASR_REF = os.environ.get("CRISPASR_REF", "feat/tiron-asr")
TIRON_SRC = "Trelis/tiron"
TIRON_HF_OUT = "cstr/tiron-GGML"
FIXTURES_REPO = "cstr/crispasr-regression-fixtures"
NAME = "tiron"
TIRON_FILES = [
    "config.json", "generation_config.json", "preprocessor_config.json",
    "tokenizer_config.json", "special_tokens_map.json", "normalizer.json",
    "tokenizer.json", "vocab.json", "merges.txt", "added_tokens.json",
    "model.safetensors",
]

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["USE_TF"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"


def run(cmd, cwd=None, timeout=None, check=True, env=None):
    e = {**os.environ, **(env or {})}
    r = subprocess.run(cmd, cwd=cwd, timeout=timeout, env=e,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.stdout:
        print(r.stdout, flush=True)
    if check and r.returncode != 0:
        # RuntimeError (not SystemExit) so callers can catch + still checkpoint.
        raise RuntimeError(f"rc={r.returncode}: {' '.join(map(str, cmd))}")
    return r


# ---- clone + harness + auth (early, so the mirror is live from the build) ----
if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive", CRISPASR_REPO, str(REPO)])
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
kh._HF_PUSH_INTERVAL_S = 20.0
kh.step("start", ref=CRISPASR_REF, scratch=str(SCRATCH),
        sha=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())

token = kh.resolve_hf_token()          # exports HF_TOKEN
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"   # resolve flips it back to 1
if not token:
    raise SystemExit("no HF token — cannot upload checkpoints")
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
api = HfApi(token=token)
api.create_repo(repo_id=TIRON_HF_OUT, repo_type="model", private=True, exist_ok=True)


def upload(local: Path, path_in_repo: str, repo: str, repo_type: str = "model"):
    kh.step("upload.begin", file=path_in_repo, gb=round(local.stat().st_size / 1e9, 3))
    api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo,
                    repo_id=repo, repo_type=repo_type, commit_message=f"add {path_in_repo}")
    kh.step("upload.done", file=path_in_repo)


# ---- build ONLY the legacy quantizer (convert + refdump are pure Python) ----
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
run(["cmake", "-S", str(REPO), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
     "-DBUILD_SHARED_LIBS=ON"] + kh.cuda_build_flags(arch) + kh.cache_and_link_flags())
with kh.build_heartbeat("build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-legacy-quantize "
                        f"-j{kh.safe_build_jobs(gpu=True)}")
QUANT = next((c for c in BUILD.rglob("crispasr-legacy-quantize") if c.is_file() and os.access(c, os.X_OK)), None)
if QUANT is None:
    raise SystemExit("crispasr-legacy-quantize not built")
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
kh.step("build.done", quant=str(QUANT))

# ---- deps for convert + refdump ----
kh.sh_with_progress("pip install -q transformers safetensors 'git+https://github.com/TrelisResearch/tiron'")

# ---- 1. download source + convert -> f16 ----
kh.step("download.begin", free_gb=kh.free_gb(str(MODELS)))
src_dir = MODELS / "tiron-src"
src_dir.mkdir(parents=True, exist_ok=True)
for fn in TIRON_FILES:
    p = hf_hub_download(repo_id=TIRON_SRC, filename=fn, local_dir=str(src_dir), token=token)
    kh.step("download.file", file=fn, mb=round(os.path.getsize(p) / 1e6, 1))

whisper_repo = MODELS / "openai-whisper"
if not whisper_repo.exists():
    run(["git", "clone", "--depth", "1", "https://github.com/openai/whisper.git", str(whisper_repo)])

f16 = OUT / "ggml-model.bin"
if f16.exists():
    f16.unlink()
kh.step("convert.begin", free_gb=kh.free_gb(str(OUT)))
with kh.build_heartbeat("convert", interval_s=60):
    run(["python", "models/convert-h5-to-ggml.py", str(src_dir), str(whisper_repo), str(OUT)],
        cwd=str(REPO), timeout=3600)
f16_named = OUT / f"{NAME}-f16.bin"
f16.rename(f16_named)
kh.step("convert.done", gb=round(f16_named.stat().st_size / 1e9, 3))

# ---- 2. checkpoint f16 to HF (before the crash-prone quantize) ----
readme = OUT / "README.md"
readme.write_text(
    f"""---
license: apache-2.0
base_model: {TIRON_SRC}
tags: [automatic-speech-recognition, whisper, speaker-diarization, crispasr, ggml]
---
# {NAME} — GGML for CrispASR (#295)
Converted from [`{TIRON_SRC}`](https://huggingface.co/{TIRON_SRC}) (Apache-2.0).
Whisper large-v3 + inline `<|speakerN|>` markers; needs a CrispASR build with the
tiron decode mode. Files: `{NAME}-f16.bin`, `{NAME}-q4_k.bin`.
""", encoding="utf-8")
upload(readme, "README.md", TIRON_HF_OUT)
upload(f16_named, f"{NAME}-f16.bin", TIRON_HF_OUT)

# ---- 3. quantize q4_k with the LEGACY (whisper-bin) quantizer ----
q4k = OUT / f"{NAME}-q4_k.bin"
kh.step("quantize.begin", free_gb=kh.free_gb(str(OUT)))
try:
    with kh.build_heartbeat("quantize", interval_s=60):
        run([str(QUANT), str(f16_named), str(q4k), "q4_k"], timeout=1800)
    kh.step("quantize.done", gb=round(q4k.stat().st_size / 1e9, 3))
    # 4. checkpoint q4_k
    upload(q4k, f"{NAME}-q4_k.bin", TIRON_HF_OUT)
except Exception as ex:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    kh.step("quantize.FAILED", error=f"{type(ex).__name__}: {ex}")
    q4k = None  # continue: f16 is safe on HF; the ref dump still matters

# ---- 5/6. free the f16 locally (it lives on HF now) ----
if f16_named.exists():
    f16_named.unlink()
kh.step("f16.removed", free_gb=kh.free_gb(str(OUT)))

# ---- 7. dump the constrained-grammar reference from the Python blueprint ----
sample = REPO / "samples" / "multispeaker.wav"
ref = OUT / f"{NAME}-multispeaker-ref.gguf"
kh.step("refdump.begin", audio=str(sample), exists=sample.exists())
try:
    with kh.build_heartbeat("refdump", interval_s=60):
        run(["python", "tools/dump_reference.py", "--backend", "tiron",
             "--model-dir", str(src_dir), "--audio", str(sample),
             "--output", str(ref), "--max-new-tokens", "444"],
            cwd=str(REPO), timeout=1800)
    kh.step("refdump.done", mb=round(ref.stat().st_size / 1e6, 2))
    # 8. checkpoint the reference (fixtures repo, tiron/multispeaker/ref.gguf)
    upload(ref, "tiron/multispeaker/ref.gguf", FIXTURES_REPO, repo_type="dataset")
except Exception as ex:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    kh.step("refdump.FAILED", error=f"{type(ex).__name__}: {ex}")

# refresh ccache seed (single tar, page-1 retrievable)
try:
    kh.export_ccache_tar()
except Exception:
    pass

kh.step("DONE", tiron_ggml=TIRON_HF_OUT, q4k=bool(q4k), ref_repo=FIXTURES_REPO)
print("=== done ===", flush=True)
