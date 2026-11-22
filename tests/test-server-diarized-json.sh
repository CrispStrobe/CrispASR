#!/bin/bash
# test-server-diarized-json.sh — live integration test for diarized_json
# response_format on /v1/audio/transcriptions (issue #206).
#
# Boots the server with a whisper model, sends a transcription request with
# response_format=diarized_json + diarize=true, and validates the response
# structure contains speaker labels, start/end, type fields.
#
# SKIPs (exit 0) when no model or test audio is found.
#
# Usage: ./tests/test-server-diarized-json.sh [--port N]

set -uo pipefail
cd "$(dirname "$0")/.."

PORT=${PORT:-11459}
for arg in "$@"; do
    case "$arg" in
        --port=*) PORT="${arg#--port=}" ;;
    esac
done

CRISPASR=""
for cand in build/bin/crispasr build-ninja-compile/bin/crispasr ./bin/crispasr; do
    if [ -x "$cand" ]; then CRISPASR="$cand"; break; fi
done
[ -n "$CRISPASR" ] || { echo "SKIP: crispasr binary not found"; exit 0; }

# Locate a whisper GGUF.
MODEL="${CRISPASR_MODEL_WHISPER:-}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    MODEL=""
    for d in "${CRISPASR_MODELS_DIR:-}" /mnt/storage/gguf-models "$HOME/.cache/crispasr"; do
        [ -n "$d" ] && [ -d "$d" ] || continue
        cand=$(ls "$d"/ggml-base*.bin "$d"/whisper-base*.gguf 2>/dev/null | head -1)
        if [ -n "$cand" ] && [ -f "$cand" ]; then MODEL="$cand"; break; fi
    done
fi
[ -n "$MODEL" ] && [ -f "$MODEL" ] || { echo "SKIP: no whisper GGUF found"; exit 0; }

# Locate test audio.
AUDIO="${CRISPASR_TEST_AUDIO:-}"
if [ -z "$AUDIO" ] || [ ! -f "$AUDIO" ]; then
    AUDIO=""
    for cand in samples/jfk.wav /mnt/akademie_storage/whisper.cpp/samples/jfk.wav; do
        if [ -f "$cand" ]; then AUDIO="$cand"; break; fi
    done
fi
[ -n "$AUDIO" ] && [ -f "$AUDIO" ] || { echo "SKIP: no test audio found"; exit 0; }

echo "Model: $MODEL"
echo "Audio: $AUDIO"
LOG=$(mktemp "${TMPDIR:-/tmp}/crispasr-diarized.XXXXXX")
trap 'kill "$SV" 2>/dev/null; rm -f "$LOG"' EXIT
"$CRISPASR" --server -m "$MODEL" --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
SV=$!
ready=0
for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
    kill -0 "$SV" 2>/dev/null || break
    sleep 1
done
[ "$ready" = 1 ] || { echo "ERROR: server not ready. Log:"; tail -5 "$LOG"; exit 2; }

PASS=0; FAIL=0

# ── Test 1: diarized_json accepted and returns valid structure ──
resp=$(curl -s "http://127.0.0.1:$PORT/v1/audio/transcriptions" \
    -F "file=@$AUDIO" \
    -F "response_format=diarized_json" \
    -F "diarize=true" \
    -F "diarize_method=energy")

if echo "$resp" | grep -q '"segments"'; then
    echo "  ✓ diarized_json returned segments"; PASS=$((PASS+1))
else
    echo "  ✗ no segments in response: $resp"; FAIL=$((FAIL+1))
fi

if echo "$resp" | grep -q '"speaker"'; then
    echo "  ✓ speaker field present"; PASS=$((PASS+1))
else
    echo "  ✗ no speaker field: $resp"; FAIL=$((FAIL+1))
fi

if echo "$resp" | grep -q '"type": "transcript.text.segment"'; then
    echo "  ✓ type field present"; PASS=$((PASS+1))
else
    echo "  ✗ no type field: $resp"; FAIL=$((FAIL+1))
fi

if echo "$resp" | grep -q '"start"'; then
    echo "  ✓ start field present"; PASS=$((PASS+1))
else
    echo "  ✗ no start field: $resp"; FAIL=$((FAIL+1))
fi

if echo "$resp" | grep -q '"text"'; then
    echo "  ✓ text field present"; PASS=$((PASS+1))
else
    echo "  ✗ no text field: $resp"; FAIL=$((FAIL+1))
fi

if echo "$resp" | grep -q '"task"'; then
    echo "  ✓ task field present"; PASS=$((PASS+1))
else
    echo "  ✗ no task field: $resp"; FAIL=$((FAIL+1))
fi

# ── Test 2: invalid response_format rejected ──
code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/audio/transcriptions" \
    -F "file=@$AUDIO" \
    -F "response_format=bogus_format")
if [ "$code" = "400" ]; then
    echo "  ✓ invalid format → 400"; PASS=$((PASS+1))
else
    echo "  ✗ invalid format → $code"; FAIL=$((FAIL+1))
fi

# ── Test 3: diarized_json without diarize still works (speaker defaults to A) ──
resp2=$(curl -s "http://127.0.0.1:$PORT/v1/audio/transcriptions" \
    -F "file=@$AUDIO" \
    -F "response_format=diarized_json")

if echo "$resp2" | grep -q '"speaker": "A"'; then
    echo "  ✓ no-diarize defaults speaker to A"; PASS=$((PASS+1))
else
    echo "  ✗ expected default speaker A: $resp2"; FAIL=$((FAIL+1))
fi

echo
echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
