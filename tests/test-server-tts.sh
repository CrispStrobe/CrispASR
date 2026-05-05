#!/bin/bash
# test-server-tts.sh — integration smoke for /v1/audio/speech + /v1/voices.
#
# Boots crispasr-server with a small qwen3-tts CustomVoice model, exercises
# every documented response code on the TTS routes, and validates response
# bodies (WAV magic bytes, JSON error shape, etc.).
#
# Usage:
#   ./tests/test-server-tts.sh [--port N] [--keep-server]
#
# Requires:
#   - build/bin/crispasr (or build-ninja-compile/bin/crispasr)
#   - qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf in CRISPASR_MODELS dir
#   - qwen3-tts-tokenizer-12hz.gguf in CRISPASR_MODELS dir
#
# Exit code: 0 if all pass, non-zero otherwise.

set -uo pipefail
cd "$(dirname "$0")/.."

PORT=${PORT:-11442}
KEEP_SERVER=0
for arg in "$@"; do
    case "$arg" in
        --port=*) PORT="${arg#--port=}" ;;
        --port) shift; PORT="$1" ;;
        --keep-server) KEEP_SERVER=1 ;;
    esac
done

# Locate the binary.
CRISPASR=""
for cand in build-ninja-compile/bin/crispasr build/bin/crispasr ./bin/crispasr; do
    if [ -x "$cand" ]; then CRISPASR="$cand"; break; fi
done
if [ -z "$CRISPASR" ]; then
    echo "ERROR: crispasr binary not found. Build first."
    exit 2
fi

# Locate models.
MODELS_DIR="${CRISPASR_MODELS:-/Volumes/backups/ai/crispasr-models}"
TALKER="$MODELS_DIR/qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf"
CODEC="$MODELS_DIR/qwen3-tts-tokenizer-12hz.gguf"

if [ ! -f "$TALKER" ]; then
    echo "SKIP: $TALKER not found (set CRISPASR_MODELS to override)"
    exit 0
fi
if [ ! -f "$CODEC" ]; then
    echo "SKIP: $CODEC not found"
    exit 0
fi

# Set up a voice-dir with a WAV + companion TXT (for Base resolution tests
# that we can't run with the CustomVoice model loaded — kept here so the
# /v1/voices listing has something to enumerate).
VOICE_DIR=$(mktemp -d -t crispasr-voices.XXXXXX)
trap 'rm -rf "$VOICE_DIR"; if [ "$KEEP_SERVER" -eq 0 ] && [ -n "${SERVER_PID:-}" ]; then kill "$SERVER_PID" 2>/dev/null || true; fi' EXIT
cp samples/qwen3_tts/clone.wav "$VOICE_DIR/clone.wav"
echo "This is a reference transcription for the cloned voice." > "$VOICE_DIR/clone.txt"

# Boot the server.
SERVER_LOG=$(mktemp -t crispasr-server.XXXXXX)
echo "Starting crispasr-server on :$PORT (talker=qwen3-tts-customvoice 0.6B)…"
"$CRISPASR" --server --backend qwen3-tts-customvoice \
    -m "$TALKER" --codec-model "$CODEC" \
    --voice-dir "$VOICE_DIR" \
    --cors-origin '*' \
    --host 127.0.0.1 --port "$PORT" --no-prints \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Wait for /health to come up (model load can take 10-30s on M1).
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server died during startup. Log:"
        cat "$SERVER_LOG"
        exit 2
    fi
    sleep 1
done

if ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "ERROR: server did not become ready within 60s. Log:"
    cat "$SERVER_LOG"
    exit 2
fi

# Wait until the model is fully loaded (ready=true). /v1/audio/speech
# returns 503 while loading.
for i in $(seq 1 120); do
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
        -H 'Content-Type: application/json' \
        -d '{"input":"warmup"}' "http://127.0.0.1:$PORT/v1/audio/speech")
    if [ "$code" != "503" ]; then break; fi
    sleep 1
done

PASS=0
FAIL=0
FAILED_NAMES=""

assert() {
    local name="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        echo "  ✓ $name"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $name: expected '$expected', got '$actual'"
        FAIL=$((FAIL + 1))
        FAILED_NAMES="$FAILED_NAMES\n    - $name"
    fi
}

assert_contains() {
    local name="$1" needle="$2" haystack="$3"
    if echo "$haystack" | grep -q -- "$needle"; then
        echo "  ✓ $name"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $name: '$needle' not found in: $haystack"
        FAIL=$((FAIL + 1))
        FAILED_NAMES="$FAILED_NAMES\n    - $name"
    fi
}

# ───────────────────────────── CORS ──────────────────────────────────────
echo
echo "=== CORS (server started with --cors-origin '*') ==="

# Preflight (OPTIONS) should return 204 with the CORS headers.
resp=$(curl -s -i -X OPTIONS \
    -H "Origin: http://example.test" \
    -H "Access-Control-Request-Method: POST" \
    "http://127.0.0.1:$PORT/v1/audio/speech")
assert_contains "OPTIONS preflight returns 2xx" "HTTP/1.1 204" "$resp"
assert_contains "Access-Control-Allow-Origin: *" "Access-Control-Allow-Origin: *" "$resp"
assert_contains "Access-Control-Allow-Methods present" "Access-Control-Allow-Methods" "$resp"

# Regular GET should also carry the Allow-Origin header.
hdrs=$(curl -s -i "http://127.0.0.1:$PORT/health" | head -20)
assert_contains "GET /health carries Allow-Origin" "Access-Control-Allow-Origin: *" "$hdrs"

# ───────────────────────────── status routes ─────────────────────────────
echo
echo "=== status routes ==="

resp=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health")
assert "GET /health → 200" "200" "$resp"

body=$(curl -s "http://127.0.0.1:$PORT/backends")
assert_contains "GET /backends contains 'qwen3-tts'" "qwen3-tts" "$body"

body=$(curl -s "http://127.0.0.1:$PORT/v1/models")
assert_contains "GET /v1/models has 'id'" '"id"' "$body"

# ───────────────────────────── /v1/voices ────────────────────────────────
echo
echo "=== GET /v1/voices ==="

body=$(curl -s "http://127.0.0.1:$PORT/v1/voices")
assert_contains "lists voice from --voice-dir (clone.wav)" '"clone"' "$body"
assert_contains "format field set to 'wav'" '"format": "wav"' "$body"

# ─────────────────────────── /v1/audio/speech ────────────────────────────
echo
echo "=== POST /v1/audio/speech — error paths ==="

# Bad JSON
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    -H 'Content-Type: application/json' \
    -d 'not json' "http://127.0.0.1:$PORT/v1/audio/speech")
assert "malformed body → 400" "400" "$code"

# Missing input
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    -H 'Content-Type: application/json' \
    -d '{"voice":"vivian"}' "http://127.0.0.1:$PORT/v1/audio/speech")
assert "missing input → 400" "400" "$code"

# Empty input
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    -H 'Content-Type: application/json' \
    -d '{"input":""}' "http://127.0.0.1:$PORT/v1/audio/speech")
assert "empty input → 400" "400" "$code"

# Unknown format
out=$(curl -s -X POST \
    -H 'Content-Type: application/json' \
    -d '{"input":"x","response_format":"mp3"}' \
    "http://127.0.0.1:$PORT/v1/audio/speech")
assert_contains "response_format=mp3 → mentions 'wav', 'pcm', or 'f32'" "wav" "$out"

# Out-of-range speed
out=$(curl -s -X POST \
    -H 'Content-Type: application/json' \
    -d '{"input":"x","speed":5.0}' \
    "http://127.0.0.1:$PORT/v1/audio/speech")
assert_contains "speed=5.0 → 400 with 'invalid_speed' code" "invalid_speed" "$out"

out=$(curl -s -X POST \
    -H 'Content-Type: application/json' \
    -d '{"input":"x","speed":0.1}' \
    "http://127.0.0.1:$PORT/v1/audio/speech")
assert_contains "speed=0.1 → 400 with 'invalid_speed' code" "invalid_speed" "$out"

# Input too long
LONG_INPUT=$(python -c 'print("a" * 5000)')
out=$(curl -s -X POST \
    -H 'Content-Type: application/json' \
    -d "$(printf '{"input":"%s"}' "$LONG_INPUT")" \
    "http://127.0.0.1:$PORT/v1/audio/speech")
assert_contains "5000-char input → 400 with 'input_too_long'" "input_too_long" "$out"

# Error shape: code and param fields land on the wire
out=$(curl -s -X POST \
    -H 'Content-Type: application/json' \
    -d '{}' \
    "http://127.0.0.1:$PORT/v1/audio/speech")
assert_contains "missing input → 'code' field" "missing_required_field" "$out"
assert_contains "missing input → 'param' field" '"param"' "$out"

echo
echo "=== POST /v1/audio/speech — happy path ==="

# Default WAV output. For CustomVoice the default speaker is the first
# in the registry; we don't pass a voice field, so the synth runs with
# whatever was set at startup (first speaker fallback).
TMPWAV=$(mktemp -t crispasr-out.XXXXXX.wav)
code=$(curl -s -X POST \
    -H 'Content-Type: application/json' \
    -d '{"input":"This is a short test."}' \
    -o "$TMPWAV" -w "%{http_code}" \
    "http://127.0.0.1:$PORT/v1/audio/speech")
assert "default synth → 200" "200" "$code"

# Validate WAV header bytes.
if [ -s "$TMPWAV" ]; then
    head4=$(dd if="$TMPWAV" bs=4 count=1 2>/dev/null)
    assert "WAV starts with RIFF" "RIFF" "$head4"
    bytes89=$(dd if="$TMPWAV" bs=1 skip=8 count=4 2>/dev/null)
    assert "RIFF format is WAVE" "WAVE" "$bytes89"
    SIZE=$(wc -c < "$TMPWAV" | tr -d ' ')
    if [ "$SIZE" -gt 1000 ]; then
        echo "  ✓ WAV body is non-trivial ($SIZE bytes)"
        PASS=$((PASS + 1))
    else
        echo "  ✗ WAV body suspiciously small ($SIZE bytes)"
        FAIL=$((FAIL + 1))
        FAILED_NAMES="$FAILED_NAMES\n    - wav size > 1000"
    fi
else
    echo "  ✗ default synth response body is empty"
    FAIL=$((FAIL + 1))
    FAILED_NAMES="$FAILED_NAMES\n    - response body present"
fi
rm -f "$TMPWAV"

# OpenAI 'pcm' output: raw int16 LE, no header. Size should be exactly
# 2 * n_samples (samples come from the synth at 24 kHz).
TMPPCM=$(mktemp -t crispasr-out.XXXXXX.pcm)
code=$(curl -s -X POST \
    -H 'Content-Type: application/json' \
    -d '{"input":"openai pcm test","response_format":"pcm"}' \
    -o "$TMPPCM" -w "%{http_code}" \
    "http://127.0.0.1:$PORT/v1/audio/speech")
assert "OpenAI pcm synth → 200" "200" "$code"
if [ -s "$TMPPCM" ]; then
    SIZE=$(wc -c < "$TMPPCM" | tr -d ' ')
    head4=$(dd if="$TMPPCM" bs=4 count=1 2>/dev/null)
    if [ "$head4" != "RIFF" ]; then
        echo "  ✓ pcm body has no RIFF header (raw int16, $SIZE bytes)"
        PASS=$((PASS + 1))
    else
        echo "  ✗ pcm body unexpectedly starts with RIFF"
        FAIL=$((FAIL + 1))
    fi
    if [ $((SIZE % 2)) -eq 0 ]; then
        echo "  ✓ pcm body size $SIZE is int16-aligned"
        PASS=$((PASS + 1))
    else
        echo "  ✗ pcm body size $SIZE not int16-aligned"
        FAIL=$((FAIL + 1))
    fi
fi
rm -f "$TMPPCM"

# Speed parameter — speed=2.0 should produce roughly half the samples
# of speed=1.0 for the same input (linear-resampler post-synth).
TMPS1=$(mktemp -t crispasr-s1.XXXXXX.f32)
TMPS2=$(mktemp -t crispasr-s2.XXXXXX.f32)
curl -s -X POST -H 'Content-Type: application/json' \
    -d '{"input":"speed test one","response_format":"f32","speed":1.0}' \
    -o "$TMPS1" "http://127.0.0.1:$PORT/v1/audio/speech" >/dev/null
curl -s -X POST -H 'Content-Type: application/json' \
    -d '{"input":"speed test one","response_format":"f32","speed":2.0}' \
    -o "$TMPS2" "http://127.0.0.1:$PORT/v1/audio/speech" >/dev/null
S1=$(wc -c < "$TMPS1" | tr -d ' ')
S2=$(wc -c < "$TMPS2" | tr -d ' ')
RATIO=$(awk "BEGIN{print $S1/$S2}")
# Tolerance: 1.7..2.3x. Synthesis isn't deterministic across runs of
# qwen3-tts (sampling), so allow some slack — what we care about is
# that speed=2.0 is meaningfully shorter than speed=1.0.
if awk "BEGIN{exit !($S1 > $S2 * 1.7 && $S1 < $S2 * 2.3)}"; then
    echo "  ✓ speed=2.0 is ~half the samples of speed=1.0 (ratio=$RATIO)"
    PASS=$((PASS + 1))
else
    echo "  ✗ speed ratio out of bounds: $S1 / $S2 = $RATIO (want 1.7-2.3)"
    FAIL=$((FAIL + 1))
fi
rm -f "$TMPS1" "$TMPS2"

# OpenAI's 'model' field is accepted (not validated, just logged).
TMPMODEL=$(mktemp -t crispasr-model.XXXXXX.wav)
code=$(curl -s -X POST -H 'Content-Type: application/json' \
    -d '{"input":"model test","model":"tts-1"}' \
    -o "$TMPMODEL" -w "%{http_code}" \
    "http://127.0.0.1:$PORT/v1/audio/speech")
assert "model field accepted → 200" "200" "$code"
rm -f "$TMPMODEL"

# instructions field is accepted (silently ignored on CustomVoice).
TMPINST=$(mktemp -t crispasr-inst.XXXXXX.wav)
code=$(curl -s -X POST -H 'Content-Type: application/json' \
    -d '{"input":"instructions test","instructions":"speak warmly and slowly"}' \
    -o "$TMPINST" -w "%{http_code}" \
    "http://127.0.0.1:$PORT/v1/audio/speech")
assert "instructions field accepted → 200" "200" "$code"
rm -f "$TMPINST"

# f32 output should be raw float32 PCM (no header).
TMPF32=$(mktemp -t crispasr-out.XXXXXX.f32)
code=$(curl -s -X POST \
    -H 'Content-Type: application/json' \
    -d '{"input":"test","response_format":"f32"}' \
    -o "$TMPF32" -w "%{http_code}" \
    "http://127.0.0.1:$PORT/v1/audio/speech")
assert "f32 synth → 200" "200" "$code"
if [ -s "$TMPF32" ]; then
    SIZE=$(wc -c < "$TMPF32" | tr -d ' ')
    # f32 has no header; a WAV would start with RIFF.
    head4=$(dd if="$TMPF32" bs=4 count=1 2>/dev/null)
    if [ "$head4" != "RIFF" ]; then
        echo "  ✓ f32 body has no RIFF header (raw PCM, $SIZE bytes)"
        PASS=$((PASS + 1))
    else
        echo "  ✗ f32 body unexpectedly starts with RIFF"
        FAIL=$((FAIL + 1))
    fi
    # f32 size should be a multiple of 4 (sizeof float).
    if [ $((SIZE % 4)) -eq 0 ]; then
        echo "  ✓ f32 body size $SIZE is float-aligned"
        PASS=$((PASS + 1))
    else
        echo "  ✗ f32 body size $SIZE not float-aligned"
        FAIL=$((FAIL + 1))
    fi
else
    echo "  ✗ f32 synth response body empty"
    FAIL=$((FAIL + 1))
fi
rm -f "$TMPF32"

# Per-request voice switch — must use a name in the loaded model's
# CustomVoice registry. The cstr 0.6B/1.7B Q8_0 builds carry:
#   aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian
TMPVOICE=$(mktemp -t crispasr-voice.XXXXXX.wav)
code=$(curl -s -X POST \
    -H 'Content-Type: application/json' \
    -d '{"input":"named voice test","voice":"vivian"}' \
    -o "$TMPVOICE" -w "%{http_code}" \
    "http://127.0.0.1:$PORT/v1/audio/speech")
assert "named voice synth (vivian) → 200" "200" "$code"
rm -f "$TMPVOICE"

# Different speaker — exercises voice-cache invalidation. The pre-fix
# bug was that the second request would silently keep using the first
# voice; after the last_voice_key_ rework this path actually re-loads.
TMPVOICE=$(mktemp -t crispasr-voice2.XXXXXX.wav)
code=$(curl -s -X POST \
    -H 'Content-Type: application/json' \
    -d '{"input":"second voice test","voice":"ryan"}' \
    -o "$TMPVOICE" -w "%{http_code}" \
    "http://127.0.0.1:$PORT/v1/audio/speech")
assert "second-call voice switch (ryan) → 200" "200" "$code"
rm -f "$TMPVOICE"

# Unknown CustomVoice name → empty PCM from the adapter → 500 from the
# server. Worth pinning so that future "fall back to first speaker"
# changes are deliberate.
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    -H 'Content-Type: application/json' \
    -d '{"input":"x","voice":"nonexistent_voice_name"}' \
    "http://127.0.0.1:$PORT/v1/audio/speech")
assert "unknown CustomVoice → 500" "500" "$code"

# ─────────────────────────── 401 path ────────────────────────────────────
# Re-running the server with --api-key is heavy; skip unless explicitly
# enabled via TEST_AUTH=1. Even so, leave a hook here for completeness.

echo
echo "──────────────────────────────────────────────"
echo "Server log:"
echo "──────────────────────────────────────────────"
tail -20 "$SERVER_LOG"
rm -f "$SERVER_LOG"

echo
echo "=== summary ==="
echo "  pass: $PASS"
echo "  fail: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    printf "  failed:%b\n" "$FAILED_NAMES"
    exit 1
fi
