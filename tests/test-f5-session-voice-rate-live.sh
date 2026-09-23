#!/usr/bin/env bash
# test-f5-session-voice-rate-live.sh — the session ABI must hand the f5-tts
# runtime its reference voice at the MODEL's rate, like the CLI does.
#
# f5-tts is the one runtime whose rate depends on the checkpoint: 24 kHz for
# F5-TTS (Vocos), 16 kHz for Raon-OpenTTS (#387, sbhifigan16k). The CLI adapter
# resamples the reference to f5_tts_sample_rate(); crispasr_session_set_voice
# loaded it at a hard-coded 24 kHz, so every binding (Python, Dart, Go, ...)
# fed Raon a reference its mel front-end read at the wrong rate.
#
# What is compared is the reference mel itself, via the runtime's
# CRISPASR_F5_DUMP_REFMEL probe, which fires inside f5_tts_set_reference —
# before any synthesis. So the session side never synthesises, and the CLI is
# stopped as soon as it has written its dump. (A first version compared
# output lengths: it took 1 h 50 min on the 4-core VPS, and the CLI's spoken
# AI disclosure made the lengths incomparable anyway.)
#
# The two surfaces resample with different filters, so the mels are not
# bit-identical: allow a frame of length difference and require cos > 0.99.
# A wrong reference rate is far outside both.
#
# The session is opened as backend "raon", the name the CLI accepts; that
# also guards the session alias.
#
# SKIPs cleanly (exit 0) when the model, the binary or the library is missing.
set -u

BIN="${CRISPASR_BIN:-./build/bin/crispasr}"
LIB="${CRISPASR_LIB_PATH:-./build/src/libcrispasr.so}"
REF="samples/jfk.wav"
REF_TEXT="And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."

MODEL="${CRISPASR_RAON_MODEL:-}"
if [ -z "$MODEL" ]; then
    for d in "${CRISPASR_MODELS:-}" "${CRISPASR_MODELS_DIR:-}" "$HOME/.cache/crispasr"; do
        [ -n "$d" ] || continue
        [ -f "$d/raon-opentts-0.3b-f16.gguf" ] && MODEL="$d/raon-opentts-0.3b-f16.gguf" && break
    done
fi

[ -x "$BIN" ] || { echo "SKIP: crispasr binary not found at $BIN"; exit 0; }
[ -f "$LIB" ] || { echo "SKIP: libcrispasr not found at $LIB"; exit 0; }
[ -n "$MODEL" ] && [ -f "$MODEL" ] || {
    echo "SKIP: no Raon model (set CRISPASR_RAON_MODEL or CRISPASR_MODELS_DIR)"
    exit 0
}
python -c "import numpy" 2>/dev/null || { echo "SKIP: python numpy missing"; exit 0; }

OUT_DIR="$(mktemp -d "${TMPDIR:-/mnt/volume1/tmp-overflow}/f5-voice-rate.XXXXXX")"
CLI_PID=""
cleanup() {
    [ -n "$CLI_PID" ] && kill "$CLI_PID" 2>/dev/null
    rm -rf "$OUT_DIR"
}
trap cleanup EXIT

# CLI: run until the reference mel is on disk, then stop it.
CRISPASR_F5_DUMP_REFMEL="$OUT_DIR/cli.f32" "$BIN" --backend raon -m "$MODEL" --voice "$REF" \
    --ref-text "$REF_TEXT" --i-have-rights --tts "Hello there." --tts-output "$OUT_DIR/cli.wav" \
    -t 4 >"$OUT_DIR/cli.log" 2>&1 &
CLI_PID=$!
for _ in $(seq 1 600); do
    grep -q "dumped ref_mel" "$OUT_DIR/cli.log" 2>/dev/null && break
    kill -0 "$CLI_PID" 2>/dev/null || break
    sleep 1
done
kill "$CLI_PID" 2>/dev/null
wait "$CLI_PID" 2>/dev/null
CLI_PID=""
if ! grep -q "dumped ref_mel" "$OUT_DIR/cli.log"; then
    echo "FAIL: CLI never dumped its reference mel"
    tail -20 "$OUT_DIR/cli.log"
    exit 1
fi

CRISPASR_F5_DUMP_REFMEL="$OUT_DIR/ses.f32" CRISPASR_LIB_PATH="$LIB" PYTHONNOUSERSITE=1 \
    PYTHONPATH="python${PYTHONPATH:+:$PYTHONPATH}" \
    python - "$MODEL" "$REF" "$REF_TEXT" "$OUT_DIR" <<'EOF'
import re
import sys

import numpy as np
import crispasr

model, ref, ref_text, out_dir = sys.argv[1:5]

cli_log = open(f"{out_dir}/cli.log", errors="replace").read()
m = re.search(r"dumped ref_mel \(T=(\d+), mel=(\d+)\)", cli_log)
mel_dim = int(m.group(2))

s = crispasr.Session(model, backend="raon")
try:
    s.set_voice(ref, ref_text)
finally:
    s.close()

cli = np.fromfile(f"{out_dir}/cli.f32", dtype=np.float32).reshape(-1, mel_dim)
ses = np.fromfile(f"{out_dir}/ses.f32", dtype=np.float32).reshape(-1, mel_dim)
n = min(len(cli), len(ses))
a, b = cli[:n].ravel(), ses[:n].ravel()
cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
print(f"reference mel frames: cli {len(cli)}, session {len(ses)}; cos {cos:.5f}")
if abs(len(cli) - len(ses)) > 1 or cos < 0.99:
    print("FAIL: the session reference mel differs from the CLI's — the "
          "session did not load the reference at the model's rate")
    sys.exit(1)
print("PASS: session and CLI build the same reference mel")
EOF
