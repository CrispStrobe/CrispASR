#!/usr/bin/env bash
# Hermetic regression guard for the Chatterbox Multilingual V3 Q4 policy.
set -euo pipefail

QUANT="${1:-}"
REPO="${2:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -x "$QUANT" ] || { echo "SKIP: crispasr-quantize binary not found"; exit 0; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

python3 - "$TMP/in.gguf" <<'PY'
import struct
import sys

# Write the deliberately tiny GGUF directly.  The unit-test image does not
# install NumPy (nor should this hermetic quantizer policy test require it).
names = (
    "s3.tok.encoder.weight",
    "t3.speech_head.weight",
    "t3.tfmr.layers.0.attn.q_proj.weight",
    "s3.v.conv_pre.weight",
)

def string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data

with open(sys.argv[1], "wb") as f:
    f.write(b"GGUF")
    f.write(struct.pack("<IQQ", 3, len(names), 1))
    f.write(string("general.architecture"))
    f.write(struct.pack("<I", 8))  # GGUF_TYPE_STRING
    f.write(string("chatterbox"))
    tensor_bytes = 32 * 256 * 4
    for index, name in enumerate(names):
        f.write(string(name))
        f.write(struct.pack("<IQQIQ", 2, 256, 32, 0, index * tensor_bytes))
    padding = (-f.tell()) % 32
    f.write(b"\0" * padding)
    row = b"".join(struct.pack("<f", -1.0 + 2.0 * i / 255.0) for i in range(256))
    for _ in names:
        f.write(row * 32)
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
