#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "kaggle kernels push -p $DIR"
kaggle kernels push -p "$DIR"

echo
echo "Push triggered. Watch live at:"
ID="$(python -c "import json; print(json.load(open('$DIR/kernel-metadata.json'))['id'])")"
echo "  https://www.kaggle.com/code/${ID}"
echo
echo "Poll status via CLI:"
echo "  kaggle kernels status $ID"
