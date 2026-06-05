# %% [markdown]
# # CrispASR — one-off VibeVoice-ASR F16 GGUF conversion
#
# Convert `microsoft/VibeVoice-ASR` (8-shard safetensors, ~28 GB)
# to F16 GGUF on Kaggle's 30 GB-RAM CPU notebook and upload to
# `cstr/vibevoice-asr-GGUF` (where the legacy file was just Q4_K).
#
# Local M1 can't fit it (16 GB unified, mostly used), VPS OOM'd
# (7.6 GB RAM consumed by transkript_app + crisp-lens). Kaggle's
# CPU notebook has plenty of headroom and a gigabit egress to HF.
#
# Triggered from the Kaggle UI ("Save Version → Run All") so that
# the UserSecretsClient call in cell 1 lands the HF_TOKEN attached
# to your chr1str account — that's the path we know reads cleanly
# (batch / CLI-pushed runs get a different JWT that 400s on
# GetUserSecretByLabel; see the rebake-refs kernel history).
#
# If HF_TOKEN reads OK → auto-uploads at the end.
# If not                → stages the F16 to `/kaggle/working/`
#                         where a local `kaggle kernels output`
#                         can fetch it, then `hf upload` from
#                         local finishes the publish.

# %% [code]
# ── Cell 1: read HF_TOKEN from Kaggle Secrets at the very top,
#    in its own cell. Matches Kaggle's documented pattern.
from kaggle_secrets import UserSecretsClient

try:
    hf_token_secret = UserSecretsClient().get_secret("HF_TOKEN")
    print("[cell 1] HF_TOKEN read OK from Kaggle Secrets")
except Exception as exc:
    print(f"[cell 1] HF_TOKEN unreadable ({type(exc).__name__}: {exc}). "
          "Will stage to /kaggle/working/ for local pickup.")
    hf_token_secret = None

# %% [code]
# ── Cell 2: set up env, clone CrispASR, install ML deps.
import os
import subprocess
import sys
from pathlib import Path

if hf_token_secret:
    os.environ["HF_TOKEN"] = hf_token_secret
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token_secret

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
OUT = WORK / "vibevoice-asr-7b-f16.gguf"

# clone (sparse — only need models/) to keep repo small
if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
        "https://github.com/CrispStrobe/CrispASR.git", str(REPO),
    ])
subprocess.check_call(["git", "-C", str(REPO), "sparse-checkout", "set",
                       "models", "tools"])

# CPU-only torch keeps install small + fast on Kaggle's image
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "torch", "transformers", "safetensors", "gguf", "huggingface_hub",
    "hf_transfer",
])
print("[cell 2] deps installed")

# %% [code]
# ── Cell 3: download source weights (anonymous works — public repo).
from huggingface_hub import snapshot_download

# Cache the 28 GB source somewhere NOT on /kaggle/working/ — that
# directory is preserved as the kernel's output (saved to Kaggle's
# storage) and is capped at ~20 GB. /kaggle/temp/ is ephemeral
# scratch with much more room; falls back to /tmp if not present.
import shutil
for candidate in ("/kaggle/temp", "/tmp"):
    if os.path.isdir(candidate):
        scratch = Path(candidate) / "vibevoice-asr-src"
        break
scratch.mkdir(parents=True, exist_ok=True)
print(f"[cell 3] source cache: {scratch}  "
      f"(free: {shutil.disk_usage(scratch).free / (1024**3):.1f} GiB)")

src = snapshot_download(
    repo_id="microsoft/VibeVoice-ASR",
    cache_dir=str(scratch),
)
print(f"[cell 3] source dir: {src}")

# %% [code]
# ── Cell 4: run the converter. F16 by default; no --include-decoder
#    flag because this is the ASR variant.
subprocess.check_call([
    sys.executable, "models/convert-vibevoice-to-gguf.py",
    "--input", src,
    "--output", str(OUT),
], cwd=str(REPO))

print(f"[cell 4] wrote {OUT} ({OUT.stat().st_size / (1024**3):.1f} GiB)")

# %% [code]
# ── Cell 5: upload to HF if HF_TOKEN is in env (UI-triggered case);
#    otherwise stage for local pickup via `kaggle kernels output`.
if hf_token_secret:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token_secret)
    print("[cell 5] uploading to cstr/vibevoice-asr-GGUF (hf_transfer on) …")
    commit = api.upload_file(
        path_or_fileobj=str(OUT),
        path_in_repo="vibevoice-asr-f16.gguf",
        repo_id="cstr/vibevoice-asr-GGUF",
        repo_type="model",
        commit_message="Add F16 GGUF (fixed: tokenizer, lm_head, "
                       "assistant header, v_proj.bias)",
    )
    print(f"[cell 5] uploaded; commit: {commit.oid if hasattr(commit,'oid') else commit}")
else:
    print(f"[cell 5] staged at {OUT}; fetch locally with:")
    print(f"  kaggle kernels output chr1str/crispasr-vibevoice-asr-convert -p .")
    print(f"  hf upload cstr/vibevoice-asr-GGUF vibevoice-asr-7b-f16.gguf "
          f"vibevoice-asr-f16.gguf")
