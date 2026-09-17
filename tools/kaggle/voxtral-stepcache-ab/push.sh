#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kaggle kernels push -p "$DIR"
echo "Watch: https://www.kaggle.com/code/chr1s4/crispasr-voxtral-stepcache-ab"
