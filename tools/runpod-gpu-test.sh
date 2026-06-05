#!/bin/bash
# CrispASR RunPod GPU test — spins up an RTX 3090, builds CUDA, runs mimo-asr
#
# Prerequisites:
#   pip install runpod
#   RUNPOD_API_KEY set (from ~/.env RPOD_API)
#   SSH key at ~/.ssh/id_ed25519
#
# Usage:
#   ./tools/runpod-gpu-test.sh           # create pod, build, test
#   ./tools/runpod-gpu-test.sh teardown  # terminate pod
#
# Cost: ~$0.22/hr (RTX 3090 community). Typical run: 15-20 min = ~$0.07.
# Always terminate when done!

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────
RPOD_API="${RUNPOD_API_KEY:-$(grep RPOD_API ~/.env 2>/dev/null | cut -d= -f2)}"
GPU_TYPE="NVIDIA GeForce RTX 3090"
GPU_ARCH=86  # sm_86 for RTX 3090
IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
PUBKEY=$(cat ~/.ssh/id_ed25519.pub 2>/dev/null || echo "")
POD_FILE="/tmp/crispasr-runpod-pod-id.txt"
BRANCH="${1:-main}"

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
[ -z "$PUBKEY" ] && { echo "ERROR: no SSH key at ~/.ssh/id_ed25519.pub"; exit 1; }

# ── Create pod ──────────────────────────────────────────────────────
echo "Creating RunPod $GPU_TYPE pod..."
POD_INFO=$(python3 << PYEOF
import runpod, json
runpod.api_key = '$RPOD_API'
pod = runpod.create_pod(
    name="crispasr-gpu-test",
    image_name="$IMAGE",
    gpu_type_id="$GPU_TYPE",
    gpu_count=1,
    volume_in_gb=30,
    container_disk_in_gb=20,
    min_vcpu_count=4,
    min_memory_in_gb=16,
    ports="22/tcp",
    env={"PUBLIC_KEY": "$PUBKEY"},
)
print(json.dumps({"id": pod["id"]}))
PYEOF
)
POD_ID=$(echo "$POD_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "$POD_ID" > "$POD_FILE"
echo "Pod ID: $POD_ID (saved to $POD_FILE)"

# ── Wait for SSH ────────────────────────────────────────────────────
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

# ── Clone + Build ──────────────────────────────────────────────────
echo "=== Clone + CUDA Build ==="
$SSH << REMOTE
set -e
export PATH=/usr/local/cuda/bin:\$PATH
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq ninja-build >/dev/null 2>&1
pip3 install cmake -q 2>/dev/null
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
ninja -C /runpod-volume/build -j\$(nproc) crispasr-cli
echo "=== Build OK ==="
ls -la /runpod-volume/build/bin/crispasr
REMOTE

# ── Run tests ──────────────────────────────────────────────────────
echo "=== Running mimo-asr CPU test ==="
$SSH << 'REMOTE'
export LD_LIBRARY_PATH=/runpod-volume/build/src:/runpod-volume/build/ggml/src:/runpod-volume/build/ggml/src/ggml-cuda
CLI=/runpod-volume/build/bin/crispasr
$CLI --backend mimo-asr -m auto --auto-download --cache-dir /runpod-volume/cache \
  -f /runpod-volume/CrispASR/samples/jfk.wav -otxt -of /tmp/mimo-cpu 2>&1 | tail -15
echo "=== CPU transcript ==="
cat /tmp/mimo-cpu.txt 2>/dev/null || echo "(empty)"
REMOTE

echo "=== Running mimo-asr GPU test ==="
$SSH << 'REMOTE'
export LD_LIBRARY_PATH=/runpod-volume/build/src:/runpod-volume/build/ggml/src:/runpod-volume/build/ggml/src/ggml-cuda
CLI=/runpod-volume/build/bin/crispasr
MIMO_ASR_BENCH=1 \
  $CLI --backend mimo-asr -m auto --auto-download --cache-dir /runpod-volume/cache \
  -f /runpod-volume/CrispASR/samples/jfk.wav -otxt -of /tmp/mimo-gpu 2>&1 | tail -15
echo "=== GPU transcript ==="
cat /tmp/mimo-gpu.txt 2>/dev/null || echo "(empty)"
REMOTE

echo ""
echo "=== Done. Run './tools/runpod-gpu-test.sh teardown' to terminate the pod ==="
echo "Pod ID: $POD_ID  Cost: ~\$0.22/hr"
