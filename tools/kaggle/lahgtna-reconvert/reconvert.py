"""Re-convert lahgtna-chatterbox-v1 with the multilingual tokenizer (#170).

The original GGUF used tokenizer.json (704 tokens, English-only).
This kernel re-converts with mtl_tokenizer.json (2351 tokens, Arabic +
23 language tags) and uploads the fixed GGUF to HF.
"""
import os
import subprocess
import sys
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
WORK = Path("/kaggle/working")

# 1. Clone CrispASR for the converter
print("=== Cloning CrispASR ===", flush=True)
CRISPASR_DIR = WORK / "CrispASR"
if not CRISPASR_DIR.exists():
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "https://github.com/CrispStrobe/CrispASR.git",
                           str(CRISPASR_DIR)])

# 2. Install deps
print("=== Installing deps ===", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "gguf", "safetensors", "huggingface_hub"])

# 3. Download the source model
print("=== Downloading oddadmix/lahgtna-chatterbox-v1 ===", flush=True)
from huggingface_hub import snapshot_download
model_dir = snapshot_download("oddadmix/lahgtna-chatterbox-v1",
                              local_dir=str(WORK / "lahgtna-src"),
                              allow_patterns=["*.json", "*.safetensors", "conds.pt"])
print(f"Model at: {model_dir}", flush=True)
for f in sorted(os.listdir(model_dir)):
    sz = os.path.getsize(os.path.join(model_dir, f))
    print(f"  {f}  ({sz/1e6:.1f} MB)")

# Verify mtl_tokenizer.json exists
mtl_tok = Path(model_dir) / "mtl_tokenizer.json"
assert mtl_tok.exists(), f"mtl_tokenizer.json not found in {model_dir}"
import json
with open(mtl_tok) as f:
    tok = json.load(f)
vocab_size = len(tok.get("model", {}).get("vocab", {}))
print(f"mtl_tokenizer.json vocab size: {vocab_size}")

# 4. Run the converter
print("=== Converting to GGUF ===", flush=True)
out_dir = WORK / "gguf-out"
out_dir.mkdir(exist_ok=True)
subprocess.check_call([
    sys.executable, str(CRISPASR_DIR / "models" / "convert-chatterbox-to-gguf.py"),
    "--input", str(model_dir),
    "--output-dir", str(out_dir),
    "--t3-only",  # S3Gen is shared with base chatterbox, no need to re-convert
])

# 5. Verify the output
t3_gguf = out_dir / "chatterbox-t3-f16.gguf"
assert t3_gguf.exists(), f"Output GGUF not found: {t3_gguf}"
print(f"Output: {t3_gguf} ({t3_gguf.stat().st_size/1e6:.1f} MB)")

# Read and verify vocab
import gguf
r = gguf.GGUFReader(str(t3_gguf))
field = r.fields.get("chatterbox.t3.text_tokens")
if field:
    # Count string tokens
    n_tokens = 0
    has_ar = False
    for p in field.parts:
        if hasattr(p, 'tobytes'):
            try:
                s = p.tobytes().decode('utf-8')
                if s == '[ar]':
                    has_ar = True
                if len(s) > 0 and len(s) < 50:
                    n_tokens += 1
            except:
                pass
    # Subtract metadata parts
    n_tokens = n_tokens // 2  # each token has a length prefix
    print(f"GGUF vocab tokens: ~{n_tokens}")
    print(f"[ar] token present: {has_ar}")
else:
    print("WARNING: text_tokens field not found in GGUF!")

# 6. Upload to HF
print("=== Uploading to HF ===", flush=True)
# Get HF token from dataset
hf_token = None
for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
          "/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt"]:
    if os.path.exists(p):
        with open(p) as f:
            hf_token = f.read().strip()
        break

if hf_token:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(t3_gguf),
        path_in_repo="chatterbox-t3-f16.gguf",
        repo_id="cstr/lahgtna-chatterbox-v1-GGUF",
    )
    print("Uploaded to cstr/lahgtna-chatterbox-v1-GGUF")
else:
    print("WARNING: No HF token found — GGUF saved locally but not uploaded")
    print(f"Manual upload: huggingface-cli upload cstr/lahgtna-chatterbox-v1-GGUF {t3_gguf}")

print("\n=== Done ===")
