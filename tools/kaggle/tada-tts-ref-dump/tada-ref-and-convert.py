# %% [markdown]
# # CrispASR — TADA-3B-ML reference dump + GGUF conversion
#
# Downloads HumeAI/tada-3b-ml + HumeAI/tada-codec on Kaggle's
# 30 GB-RAM CPU notebook, runs the reference backend to dump
# intermediate tensors for the diff harness, converts both models
# to GGUF F16, and uploads everything to HuggingFace.
#
# Outputs:
#   - tada-ref.gguf: reference activation dump for crispasr-diff
#   - tada-tts-3b-ml-f16.gguf: main model GGUF
#   - tada-codec-f16.gguf: codec decoder GGUF
#
# Triggered from Kaggle UI ("Save Version → Run All").

# %% [code]
# ── Cell 1: read HF_TOKEN from Kaggle Secrets ──
import os
import sys

try:
    from kaggle_secrets import UserSecretsClient
    hf_token_secret = UserSecretsClient().get_secret("HF_TOKEN")
    print("[cell 1] HF_TOKEN read OK from Kaggle Secrets")
except Exception as exc:
    print(f"[cell 1] HF_TOKEN unreadable ({type(exc).__name__}: {exc})")
    hf_token_secret = os.environ.get("HF_TOKEN")

if hf_token_secret:
    os.environ["HF_TOKEN"] = hf_token_secret
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token_secret

# %% [code]
# ── Cell 2: clone CrispASR + install deps ──
import subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"

if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1", "--branch", "feature/tada-tts",
        "https://github.com/CrispStrobe/CrispASR.git", str(REPO),
    ])

# Import harness
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh
kh.init_progress()

# Install deps
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "torch", "torchaudio", "transformers", "safetensors", "gguf",
    "huggingface_hub", "hf_transfer",
    "hume-tada",  # TADA Python package (pulls dac etc.)
])
print("[cell 2] deps installed")

# Create a minimal 16 kHz WAV for the dump harness (audio is unused for TTS
# but dump_reference.py requires it). 1 second of silence.
import wave, struct
dummy_wav = REPO / "samples" / "jfk.wav"
if not dummy_wav.exists():
    dummy_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dummy_wav), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))
    print("[cell 2] created dummy jfk.wav")

# %% [code]
# ── Cell 3: download source models ──
import shutil
from huggingface_hub import snapshot_download

token = kh.resolve_hf_token()

for candidate in ("/kaggle/temp", "/tmp"):
    if os.path.isdir(candidate):
        scratch = Path(candidate) / "tada-models"
        break
scratch.mkdir(parents=True, exist_ok=True)
print(f"[cell 3] scratch: {scratch}  "
      f"(free: {shutil.disk_usage(scratch).free / (1024**3):.1f} GiB)")

kh.step("downloading tada-3b-ml")
model_dir = Path(snapshot_download(
    repo_id="HumeAI/tada-3b-ml",
    cache_dir=str(scratch),
    token=token,
))
print(f"[cell 3] model_dir: {model_dir}")

kh.step("downloading tada-codec")
codec_dir = Path(snapshot_download(
    repo_id="HumeAI/tada-codec",
    cache_dir=str(scratch),
    token=token,
    allow_patterns=["decoder/*"],
))
print(f"[cell 3] codec_dir: {codec_dir}")

# %% [code]
# ── Cell 4: run reference dump ──
kh.step("running reference dump")
os.environ["TADA_SYN_TEXT"] = "Hello world."
os.environ["TADA_NUM_FM_STEPS"] = "10"
os.environ["TADA_CFG_SCALE"] = "1.0"
os.environ["TADA_NOISE_TEMP"] = "0.0"
os.environ["TADA_SEED"] = "42"
os.environ["TADA_DEVICE"] = "cpu"

ref_output = WORK / "tada-ref.gguf"
subprocess.check_call([
    sys.executable, "tools/dump_reference.py",
    "--backend", "tada-tts",
    "--model-dir", str(model_dir),
    "--audio", "samples/jfk.wav",
    "--output", str(ref_output),
], cwd=str(REPO))
print(f"[cell 4] ref dump: {ref_output} ({ref_output.stat().st_size / 1e6:.1f} MB)")

# %% [code]
# ── Cell 5: convert main model to GGUF ──
kh.step("converting tada-3b-ml to GGUF")
model_gguf = WORK / "tada-tts-3b-ml-f16.gguf"
subprocess.check_call([
    sys.executable, "models/convert-tada-to-gguf.py",
    "--input", str(model_dir),
    "--output", str(model_gguf),
], cwd=str(REPO))
print(f"[cell 5] model GGUF: {model_gguf} ({model_gguf.stat().st_size / 1e9:.2f} GB)")

# %% [code]
# ── Cell 6: convert codec to GGUF ──
kh.step("converting tada-codec to GGUF")
codec_gguf = WORK / "tada-codec-f16.gguf"
subprocess.check_call([
    sys.executable, "models/convert-tada-codec-to-gguf.py",
    "--input", str(codec_dir),
    "--output", str(codec_gguf),
], cwd=str(REPO))
print(f"[cell 6] codec GGUF: {codec_gguf} ({codec_gguf.stat().st_size / 1e9:.2f} GB)")

# %% [code]
# ── Cell 7: upload to HuggingFace ──
kh.step("uploading to HuggingFace")
try:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    repo_id = "cstr/tada-tts-3b-ml-GGUF"

    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

    for fpath in [ref_output, model_gguf, codec_gguf]:
        if fpath.exists():
            print(f"  uploading {fpath.name}...")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fpath.name,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  ✓ {fpath.name}")

    print(f"[cell 7] uploaded to {repo_id}")
except Exception as exc:
    print(f"[cell 7] upload failed: {exc}")
    print("  Files staged at /kaggle/working/ for manual pickup")

kh.step("done")
print("\n=== ALL DONE ===")
for fpath in [ref_output, model_gguf, codec_gguf]:
    if fpath.exists():
        print(f"  {fpath.name}: {fpath.stat().st_size / 1e9:.2f} GB")
