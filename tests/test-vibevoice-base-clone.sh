#!/bin/bash
# Configured local VibeVoice 1.5B/7B clone quality regression; no auto-downloads.
# See tests/regression/README.md. --size defaults to 1.5b; use --size 7b for 7B.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python3}" tests/test_vibevoice_clone_live.py "$@"
