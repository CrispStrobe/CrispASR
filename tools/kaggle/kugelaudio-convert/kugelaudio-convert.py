#!/usr/bin/env python3
"""Kaggle kernel: convert KugelAudio-0-Open to GGUF + capture reference intermediates.

Two phases:
  1. GGUF conversion (runs on GPU kernel for the 18.7 GB model)
  2. Reference dump with forward hooks for C++ validation

Push: python -m kaggle kernels push -p tools/kaggle/kugelaudio-convert
"""

import os, sys, time
from pathlib import Path

REPO = Path("/kaggle/working/CrispASR")
BUILD = Path("/kaggle/working/build")
OUTPUT = Path("/kaggle/working/output")

# ── Kaggle harness ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh

kh.init_progress()

# ── Phase 0: Clone repo + install deps ──────────────────────────────────────
kh.report_status("cloning repo")
if not REPO.exists():
    kh.sh_with_progress("git clone --depth 1 https://github.com/CrispStrobe/CrispASR /kaggle/working/CrispASR")

kh.report_status("installing deps")
kh.sh_with_progress("pip install -q safetensors gguf transformers huggingface_hub torch")

# Try installing kugelaudio-open (may not be on PyPI)
try:
    kh.sh_with_progress("pip install -q kugelaudio-open 2>/dev/null || true")
except Exception:
    pass

# ── Phase 1: Download model ────────────────────────────────────────────────
kh.report_status("downloading model")
MODEL_ID = "kugelaudio/kugelaudio-0-open"
token = kh.resolve_hf_token()

from huggingface_hub import snapshot_download
model_dir = snapshot_download(
    MODEL_ID,
    cache_dir="/kaggle/working/hf-cache",
    token=token,
)
print(f"model downloaded to: {model_dir}")

# ── Phase 2: GGUF conversion ───────────────────────────────────────────────
kh.report_status("converting to GGUF")
OUTPUT.mkdir(parents=True, exist_ok=True)
gguf_path = OUTPUT / "kugelaudio-0-open-f16.gguf"

sys.path.insert(0, str(REPO / "models"))
# Run converter directly
converter = str(REPO / "models" / "convert-kugelaudio-to-gguf.py")
kh.sh_with_progress(
    f"python {converter} "
    f"--input {model_dir} "
    f"--output {gguf_path} "
    f"--no-encoders "
    f"--type f16"
)

print(f"GGUF written: {gguf_path} ({gguf_path.stat().st_size / 1e9:.2f} GB)")

# ── Phase 3: Reference dump ────────────────────────────────────────────────
kh.report_status("running reference dump")
ref_dir = OUTPUT / "reference"
ref_script = str(REPO / "tools" / "reference_backends" / "kugelaudio.py")

# This requires kugelaudio_open package or manual model loading
try:
    kh.sh_with_progress(
        f"python {ref_script} "
        f"--model {model_dir} "
        f"--text 'Hello, this is a test of the speech synthesis system.' "
        f"--output-dir {ref_dir} "
        f"--num-steps 20 --seed 42 --cfg-scale 3.0"
    )
    print(f"reference dump completed: {ref_dir}")
except Exception as e:
    print(f"reference dump failed (may need kugelaudio-open package): {e}")
    # Fallback: just dump tensor map
    kh.sh_with_progress(
        f"python {ref_script} "
        f"--model {MODEL_ID} "
        f"--output-dir {ref_dir} "
        f"--dump-tensor-map"
    )

# ── Phase 4: Upload artifacts ───────────────────────────────────────────────
kh.report_status("listing outputs")
for f in sorted(OUTPUT.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(OUTPUT)}: {f.stat().st_size / 1e6:.1f} MB")

kh.report_status("done")
print("All done. Download GGUF from Kaggle output tab.")
