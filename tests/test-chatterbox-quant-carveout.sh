#!/usr/bin/env bash
# Hermetic regression guard for the Chatterbox Multilingual V3 Q4 policy.
set -euo pipefail

QUANT="${1:-}"
REPO="${2:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -x "$QUANT" ] || { echo "SKIP: crispasr-quantize binary not found"; exit 0; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PYTHONPATH="$REPO/gguf-py${PYTHONPATH:+:$PYTHONPATH}" python3 - "$TMP/in.gguf" <<'PY'
import sys
import numpy as np
from gguf import GGUFWriter

w = GGUFWriter(sys.argv[1], "chatterbox", use_temp_file=False)
matrix = np.linspace(-1.0, 1.0, 32 * 256, dtype=np.float32).reshape(32, 256)
for name in (
    "s3.tok.encoder.weight",
    "t3.speech_head.weight",
    "t3.tfmr.layers.0.attn.q_proj.weight",
    "s3.v.conv_pre.weight",
):
    w.add_tensor(name, matrix)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
PY

LOG="$TMP/quant.log"
"$QUANT" "$TMP/in.gguf" "$TMP/out.gguf" q4_k >"$LOG" 2>&1

line_for() { grep -F "$1" "$LOG" | head -1 || true; }
require() {
    local name="$1" decision="$2" line
    line="$(line_for "$name")"
    [ -n "$line" ] || { echo "FAIL: no quantizer decision for $name"; exit 1; }
    echo "$line" | grep -qi "$decision" || {
        echo "FAIL: $name should be $decision, got: $line"; exit 1;
    }
}

require "s3.tok.encoder.weight" "q8_0"
require "t3.speech_head.weight" "q8_0"
require "t3.tfmr.layers.0.attn.q_proj.weight" "q4_k"
require "s3.v.conv_pre.weight" "copying"
echo "PASS: Chatterbox Q4 keeps tokenizer/sampling head at Q8 and vocoder at source precision"
