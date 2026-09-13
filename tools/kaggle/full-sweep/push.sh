#!/bin/bash
# Stage the canonical sweep script + harness alongside the kernel metadata and
# push to Kaggle (chr1str -- NOT chr1s4; kernel-metadata.json's id and its
# dataset_sources are both chr1str, and private datasets are per-account, so a
# chr1s4 push fails on the dataset references). Avoids committing a duplicate of
# the 800-line script.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
STAGE="$(mktemp -d)"
cp "$ROOT/tools/kaggle-benchmark-all-backends.py" "$STAGE/"
cp "$ROOT/tools/kaggle/kaggle_harness.py" "$STAGE/"
cp "$HERE/kernel-metadata.json" "$STAGE/"
echo "Staged in $STAGE; pushing to Kaggle..."
kaggle kernels push -p "$STAGE"
rm -rf "$STAGE"
