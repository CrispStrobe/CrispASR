#!/bin/bash
# CrispASR RunPod — KugelAudio full diff + ASR roundtrip
#
# Spins up an RTX 3090, builds CUDA, runs:
#   1. Python reference dump (kugelaudio-open → ref.gguf)
#   2. crispasr-diff kugelaudio (C++ vs Python cosine parity)
#   3. TTS synthesis → whisper ASR roundtrip
#
# Prerequisites: pip install runpod, RPOD_API in ~/.env, SSH key
#
# Usage:
#   ./tools/runpod-kugelaudio-diff.sh                         # full pipeline
#   ./tools/runpod-kugelaudio-diff.sh feature/kugelaudio-tts  # specific branch
#   ./tools/runpod-kugelaudio-diff.sh teardown                # terminate pod

set -euo pipefail

RPOD_API="${RUNPOD_API_KEY:-$(grep RPOD_API ~/.env 2>/dev/null | cut -d= -f2)}"
GPU_TYPE="NVIDIA GeForce RTX 3090"
GPU_ARCH=86
IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
PUBKEY=$(cat ~/.ssh/id_ed25519.pub 2>/dev/null || echo "")
POD_FILE="/tmp/crispasr-kugelaudio-diff-pod.txt"
BRANCH="${1:-feature/kugelaudio-tts}"

if [ "$BRANCH" = "teardown" ]; then
    if [ -f "$POD_FILE" ]; then
        POD_ID=$(cat "$POD_FILE")
        echo "Terminating pod $POD_ID..."
        python3 -c "
import runpod
runpod.api_key = '$RPOD_API'
runpod.terminate_pod('$POD_ID')
print('Terminated')
"
        rm -f "$POD_FILE"
    else
        echo "No pod ID file found at $POD_FILE"
    fi
    exit 0
fi

[ -z "$RPOD_API" ] && { echo "ERROR: RUNPOD_API_KEY not set"; exit 1; }
[ -z "$PUBKEY" ] && { echo "ERROR: no SSH key"; exit 1; }

echo "Creating RunPod $GPU_TYPE pod..."
POD_INFO=$(python3 << PYEOF
import runpod, json
runpod.api_key = '$RPOD_API'
pod = runpod.create_pod(
    name="crispasr-kugelaudio-diff",
    image_name="$IMAGE",
    gpu_type_id="$GPU_TYPE",
    gpu_count=1,
    volume_in_gb=50,
    container_disk_in_gb=30,
    min_vcpu_count=4,
    min_memory_in_gb=24,
    ports="22/tcp",
    env={"PUBLIC_KEY": "$PUBKEY"},
)
print(json.dumps({"id": pod["id"]}))
PYEOF
)
POD_ID=$(echo "$POD_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "$POD_ID" > "$POD_FILE"
echo "Pod ID: $POD_ID"

echo "Waiting for SSH..."
for i in $(seq 1 60); do
    SSH_INFO=$(python3 << PYEOF
import runpod, json
runpod.api_key = '$RPOD_API'
pod = runpod.get_pod('$POD_ID')
runtime = pod.get('runtime', {}) or {}
for p in (runtime.get('ports', []) or []):
    if p.get('privatePort') == 22:
        print(json.dumps({"ip": p["ip"], "port": p["publicPort"]}))
        break
PYEOF
    )
    if [ -n "$SSH_INFO" ]; then
        SSH_IP=$(echo "$SSH_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['ip'])")
        SSH_PORT=$(echo "$SSH_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['port'])")
        echo "SSH ready: root@$SSH_IP -p $SSH_PORT"
        break
    fi
    echo "  [$i] waiting..."
    sleep 10
done
[ -z "${SSH_IP:-}" ] && { echo "ERROR: SSH never came up"; exit 1; }
SSH="ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 root@$SSH_IP -p $SSH_PORT"

# ── Phase 1: Clone + Build ──────────────────────────────────────
echo "=== Phase 1: Clone + CUDA Build ==="
$SSH << REMOTE
set -e
export PATH=/usr/local/cuda/bin:\$PATH
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq ninja-build >/dev/null 2>&1
# Pin transformers<=4.51 to avoid float8_e8m0fnu on older PyTorch
pip3 install cmake huggingface_hub hf_transfer safetensors gguf 'transformers<=4.51' -q 2>/dev/null

export CMAKE=\$(find /usr -path '*/cmake/data/bin/cmake' 2>/dev/null | head -1)
[ -z "\$CMAKE" ] && CMAKE=cmake

cd /runpod-volume
rm -rf CrispASR build
git clone --depth 1 --branch $BRANCH https://github.com/CrispStrobe/CrispASR.git
cd CrispASR && git submodule update --init --depth 1 ggml
echo "SHA: \$(git rev-parse --short HEAD)"

mkdir -p /runpod-volume/build
\$CMAKE -G Ninja -S /runpod-volume/CrispASR -B /runpod-volume/build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DCRISPASR_BUILD_TESTS=OFF \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH
ninja -C /runpod-volume/build -j\$(nproc) crispasr-cli crispasr-diff
echo "=== Build OK ==="
ls -la /runpod-volume/build/bin/crispasr /runpod-volume/build/bin/crispasr-diff
REMOTE

# ── Phase 2: Download models ────────────────────────────────────
echo "=== Phase 2: Download models ==="
$SSH << 'REMOTE'
set -e
python3 -c "
from huggingface_hub import hf_hub_download, snapshot_download
import shutil, os

# KugelAudio Q4_K GGUF for C++ inference
path = hf_hub_download('cstr/kugelaudio-0-open-GGUF', 'kugelaudio-0-open-q4_k.gguf',
                       cache_dir='/runpod-volume/cache')
if not os.path.exists('/runpod-volume/kugelaudio-q4k.gguf'):
    os.symlink(path, '/runpod-volume/kugelaudio-q4k.gguf')
print(f'C++ GGUF: {path}')

# KugelAudio source model for Python reference dump
print('Downloading source model for reference dump...')
src = snapshot_download('kugelaudio/kugelaudio-0-open',
                       cache_dir='/runpod-volume/cache/hf-src')
print(f'Python model: {src}')

# Whisper for ASR roundtrip
path = hf_hub_download('ggerganov/whisper.cpp', 'ggml-base.en.bin',
                       cache_dir='/runpod-volume/cache')
shutil.copy2(path, '/runpod-volume/cache/ggml-base.en.bin')
print(f'Whisper: /runpod-volume/cache/ggml-base.en.bin')
"
REMOTE

# ── Phase 3: Install kugelaudio-open + run reference dump ───────
echo "=== Phase 3: Python reference dump ==="
$SSH << 'REMOTE'
set -e
# Install kugelaudio-open from GitHub
pip3 install diffusers -q 2>/dev/null
pip3 install git+https://github.com/Kugelaudio/kugelaudio-open.git -q 2>/dev/null || {
    echo "WARNING: kugelaudio-open install failed, trying pip"
    pip3 install kugelaudio-open -q 2>/dev/null || echo "kugelaudio-open not available"
}

export KUGELAUDIO_TEXT="Hello, this is a test of the speech synthesis system."
export KUGELAUDIO_SEED=42
export KUGELAUDIO_STEPS=20
# CFG=1.0 for diff (no batch doubling in hooks, matches C++ which has no CFG yet)
export KUGELAUDIO_CFG=1.0

# Find the source model path
SRC_MODEL=$(python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('kugelaudio/kugelaudio-0-open', cache_dir='/runpod-volume/cache/hf-src'))
")

cd /runpod-volume/CrispASR
python3 tools/dump_reference.py \
    --backend kugelaudio \
    --model-dir "$SRC_MODEL" \
    --audio samples/jfk.wav \
    --output /runpod-volume/kugelaudio-ref.gguf 2>&1 | tail -30

ls -lh /runpod-volume/kugelaudio-ref.gguf 2>/dev/null || echo "FAILED: no reference GGUF"
REMOTE

# ── Phase 4: crispasr-diff ──────────────────────────────────────
echo "=== Phase 4: crispasr-diff ==="
$SSH << 'REMOTE'
set -e
export LD_LIBRARY_PATH=/runpod-volume/build/src:/runpod-volume/build/ggml/src:/runpod-volume/build/ggml/src/ggml-cuda

if [ -f /runpod-volume/kugelaudio-ref.gguf ]; then
    /runpod-volume/build/bin/crispasr-diff kugelaudio \
        /runpod-volume/kugelaudio-q4k.gguf \
        /runpod-volume/kugelaudio-ref.gguf \
        /runpod-volume/CrispASR/samples/jfk.wav 2>&1
else
    echo "SKIPPED: no reference GGUF (Python dump failed)"
fi
REMOTE

# ── Phase 5: TTS + ASR roundtrip ────────────────────────────────
echo "=== Phase 5: TTS + ASR roundtrip ==="
$SSH << 'REMOTE'
set -e
export LD_LIBRARY_PATH=/runpod-volume/build/src:/runpod-volume/build/ggml/src:/runpod-volume/build/ggml/src/ggml-cuda
CLI=/runpod-volume/build/bin/crispasr
WHISPER=/runpod-volume/cache/ggml-base.en.bin

echo "--- TTS: English ---"
$CLI --backend kugelaudio -m /runpod-volume/kugelaudio-q4k.gguf \
  --tts "Hello, this is a test of the speech synthesis system." \
  --tts-output /tmp/kugelaudio-en.wav --seed 42 2>&1 | tail -5
ls -lh /tmp/kugelaudio-en.wav 2>/dev/null

echo ""
echo "--- ASR roundtrip ---"
if [ -f /tmp/kugelaudio-en.wav ]; then
    echo "Input:  Hello, this is a test of the speech synthesis system."
    echo -n "Output: "
    $CLI -m $WHISPER -f /tmp/kugelaudio-en.wav --no-timestamps 2>/dev/null | grep -v "^$" | head -3
fi
REMOTE

echo ""
echo "=== Done. Run './tools/runpod-kugelaudio-diff.sh teardown' to terminate ==="
echo "Pod ID: $POD_ID  Cost: ~\$0.22/hr"
