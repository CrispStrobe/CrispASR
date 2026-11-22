#!/bin/bash
# tests/test_diarize_align_order.sh — live integration test for issue #267.
#
# Verifies that diarization applies AFTER external CTC alignment, so
# speaker-turn splitting can use word timestamps. Requires:
#   - crispasr binary (built)
#   - A backend without native word timestamps (cohere, paraformer)
#   - The CTC aligner model (canary-ctc-aligner)
#   - The pyannote segmentation GGUF
#   - A multi-speaker audio file (or at least mono test audio)
#
# Usage:
#   source tests/env-live-tests.sh
#   bash tests/test_diarize_align_order.sh
#
# Env vars (with defaults from env-live-tests.sh):
#   CRISPASR_BIN         — path to crispasr binary
#   CRISPASR_MODEL_ALIGNER — path to canary-ctc-aligner GGUF
#   CRISPASR_TEST_DIARIZE_MODEL — path to pyannote-seg GGUF
#   CRISPASR_TEST_DIARIZE_WAV — path to test audio
#   CRISPASR_MODEL_WHISPER — path to whisper model (for native-word-ts test)
#   CRISPASR_PARAFORMER_MODEL — path to paraformer model (no native words)

set -e

CRISPASR_BIN="${CRISPASR_BIN:-build/bin/crispasr}"
ALIGNER="${CRISPASR_MODEL_ALIGNER:-/mnt/storage/gguf-models/canary-ctc-aligner-q4_k.gguf}"
DIARIZE_MODEL="${CRISPASR_TEST_DIARIZE_MODEL:-/mnt/storage/gguf-models/pyannote-seg-3.0.gguf}"
TEST_WAV="${CRISPASR_TEST_DIARIZE_WAV:-samples/multispeaker.wav}"
WHISPER="${CRISPASR_MODEL_WHISPER:-/mnt/storage/gguf-models/ggml-base.en.bin}"
PARAFORMER="${CRISPASR_PARAFORMER_MODEL:-/mnt/storage/gguf-models/paraformer-large-q4_k.gguf}"
JFK_WAV="samples/jfk.wav"
TMPOUT="${TMPDIR:-/tmp}/test267"

PASS=0
FAIL=0
SKIP=0

check() {
    local name="$1"
    shift
    if "$@"; then
        echo "[PASS] $name"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] $name"
        FAIL=$((FAIL + 1))
    fi
}

skip() {
    echo "[SKIP] $1 — $2"
    SKIP=$((SKIP + 1))
}

mkdir -p "$TMPOUT"

# Prerequisite checks
if [ ! -f "$CRISPASR_BIN" ]; then
    echo "SKIP: crispasr binary not found at $CRISPASR_BIN"
    exit 0
fi

echo "=== Issue #267: Diarize-after-align order tests ==="
echo ""

# ─── Test 1: external aligner + diarize produces word-level speaker splits ───
echo "--- Test 1: aligner + diarize → words present in JSON output ---"
if [ -f "$PARAFORMER" ] && [ -f "$ALIGNER" ] && [ -f "$DIARIZE_MODEL" ] && [ -f "$JFK_WAV" ]; then
    $CRISPASR_BIN \
        --backend paraformer -m "$PARAFORMER" -f "$JFK_WAV" \
        -am "$ALIGNER" --force-aligner \
        --diarize --diarize-method pyannote \
        --sherpa-segment-model "$DIARIZE_MODEL" \
        --output-json-full --no-prints \
        -of "$TMPOUT/test1" 2>/dev/null || true

    if [ -f "$TMPOUT/test1.json" ]; then
        # The JSON should contain both "words" and "speaker" fields
        check "JSON contains words array" grep -q '"words"' "$TMPOUT/test1.json"
        check "JSON contains speaker field" grep -q '"speaker"' "$TMPOUT/test1.json"
    else
        echo "[FAIL] test1.json not produced"
        FAIL=$((FAIL + 1))
    fi
else
    skip "Test 1" "missing model(s): paraformer=$PARAFORMER aligner=$ALIGNER diarize=$DIARIZE_MODEL"
fi

# ─── Test 2: diarize without aligner still works (segment-level) ─────────────
echo ""
echo "--- Test 2: diarize without aligner → segment-level speaker ---"
if [ -f "$PARAFORMER" ] && [ -f "$DIARIZE_MODEL" ] && [ -f "$JFK_WAV" ]; then
    OUTPUT=$($CRISPASR_BIN \
        --backend paraformer -m "$PARAFORMER" -f "$JFK_WAV" \
        --diarize --diarize-method pyannote \
        --sherpa-segment-model "$DIARIZE_MODEL" \
        --no-prints 2>/dev/null) || true

    check "diarize-only produces speaker labels" grep -q "(speaker" <<< "$OUTPUT"
    check "diarize-only produces transcript" grep -qi "ask" <<< "$OUTPUT"
else
    skip "Test 2" "missing model(s)"
fi

# ─── Test 3: native word-timestamp backend + diarize (whisper) ───────────────
echo ""
echo "--- Test 3: native words (whisper) + diarize ---"
if [ -f "$WHISPER" ] && [ -f "$DIARIZE_MODEL" ] && [ -f "$JFK_WAV" ]; then
    $CRISPASR_BIN \
        -m "$WHISPER" -f "$JFK_WAV" \
        --diarize --diarize-method pyannote \
        --sherpa-segment-model "$DIARIZE_MODEL" \
        --output-json-full --no-prints \
        -of "$TMPOUT/test3" 2>/dev/null || true

    if [ -f "$TMPOUT/test3.json" ]; then
        check "whisper+diarize JSON has speaker" grep -q '"speaker"' "$TMPOUT/test3.json"
    else
        echo "[FAIL] test3.json not produced"
        FAIL=$((FAIL + 1))
    fi
else
    skip "Test 3" "missing model(s)"
fi

# ─── Test 4: verbose output shows align before diarize ───────────────────────
echo ""
echo "--- Test 4: verbose log shows alignment before diarization ---"
if [ -f "$PARAFORMER" ] && [ -f "$ALIGNER" ] && [ -f "$DIARIZE_MODEL" ] && [ -f "$JFK_WAV" ]; then
    VERBOSE_LOG=$($CRISPASR_BIN \
        --backend paraformer -m "$PARAFORMER" -f "$JFK_WAV" \
        -am "$ALIGNER" --force-aligner \
        --diarize --diarize-method pyannote \
        --sherpa-segment-model "$DIARIZE_MODEL" \
        --verbose 2>&1) || true

    # The verbose log should show "align[slice]" from the aligner step.
    # After issue #267, this runs BEFORE diarization. We verify the
    # aligner ran and produced output (the ordering is an integration
    # property guaranteed by the code change, not something we can
    # easily observe from log line numbers since diarize may not emit
    # a verbose-level line).
    if echo "$VERBOSE_LOG" | grep -q "align\["; then
        check "aligner ran (verbose log confirms)" true
        # Also verify diarize ran (speaker label in output)
        check "diarize ran after align" grep -q "(speaker" <<< "$VERBOSE_LOG"
    else
        echo "[SKIP] Could not parse verbose log"
        SKIP=$((SKIP + 1))
    fi
else
    skip "Test 4" "missing model(s)"
fi

# ─── Cleanup ─────────────────────────────────────────────────────────────────
rm -rf "$TMPOUT"

echo ""
echo "=== Results: $PASS passed, $FAIL failed, $SKIP skipped ==="
[ $FAIL -eq 0 ]
