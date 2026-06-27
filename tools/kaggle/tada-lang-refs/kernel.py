# %% [markdown]
# # TADA language voice reference GGUFs  (v10 — full traceback + remove token= from Encoder API)
#
# Generates tada-ref-{ar,ch,de,es,fr,it,ja,pl,pt}.gguf from FLEURS CC-BY-4.0
# clips using the HumeAI/tada-codec language-specific aligners.
#
# NO HF upload from here — fetch locally + upload:
#
#   kaggle kernels output chr1str/tada-language-voice-reference-ggufs \
#       -p /Volumes/backups/code/tada-lang-refs-stash/kaggle-out/
#   python tools/upload_tada_lang_refs.py \
#       --dir /Volumes/backups/code/tada-lang-refs-stash/kaggle-out/lang-refs/

# %% [code]
import os, sys, subprocess
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
OUT  = WORK / "lang-refs"
OUT.mkdir(exist_ok=True)

# -- Phase 0: Clone CrispASR + import harness (bundled copy as fallback) --
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
if not REPO.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1",
                               CRISPASR_URL, str(REPO)])
        sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    except Exception:
        pass  # fall through to bundled copy
else:
    subprocess.check_call(["git", "-C", str(REPO), "pull", "--ff-only"])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))

if str(REPO / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import kaggle_harness as kh
kh.init_progress()

# -- Phase 1: Install deps --
# hf_transfer MUST be installed before resolve_hf_token() sets
# HF_HUB_ENABLE_HF_TRANSFER=1, or every HF download in the gen script fails.
# Do NOT reinstall torch/torchaudio — Kaggle pre-installs them (gotcha #11).
kh.step("install deps")
kh.sh("pip uninstall -y tensorflow tf-keras", check=False)   # avoid protobuf clash
kh.sh_with_progress(
    "pip install -q "
    "gguf datasets soundfile scipy hf_transfer huggingface_hub hume-tada"
)

# -- Phase 2: Resolve HF token --
kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()   # env → Secret(retry) → dataset file
if hf_token:
    print("  HF_TOKEN resolved OK", flush=True)
else:
    print("  WARNING: no HF_TOKEN — HumeAI/tada-codec downloads may fail", flush=True)

# -- Phase 3: Generate lang refs --
# subprocess inherits HF_TOKEN + HF_HUB_ENABLE_HF_TRANSFER from env
kh.step("gen.begin")
rc = kh.sh(
    f"{sys.executable} {REPO}/tools/gen_tada_lang_refs.py "
    f"--output-dir {OUT} --skip-existing",
    check=False,
)
kh.step("gen.done", returncode=rc)

# -- Report --
files = sorted(OUT.glob("*.gguf"))
print(f"\n=== generated {len(files)} lang ref(s) ===")
for p in files:
    print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")

print("\n=== fetch + upload locally (NOT on Kaggle) ===")
print("  kaggle kernels output chr1str/tada-language-voice-reference-ggufs \\")
print("      -p /Volumes/backups/code/tada-lang-refs-stash/kaggle-out/")
print("  python tools/upload_tada_lang_refs.py \\")
print("      --dir /Volumes/backups/code/tada-lang-refs-stash/kaggle-out/lang-refs/")
