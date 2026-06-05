#!/usr/bin/env bash
# Push + trigger the funasr CUDA debug kernel on Kaggle.
# Usage: bash tools/kaggle/funasr-cuda-debug/push.sh
set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$dir"

if ! command -v kaggle >/dev/null 2>&1; then
    echo "kaggle CLI not found. Install with: pip install kaggle" >&2
    exit 1
fi

echo "Pushing $(jq -r .id kernel-metadata.json) ..."
kaggle kernels push -p "$dir"
echo
echo "Triggered. Monitor with:"
echo "  kaggle kernels status chr1str/crispasr-funasr-cuda-debug"
echo "  kaggle kernels output chr1str/crispasr-funasr-cuda-debug -p ./out"
