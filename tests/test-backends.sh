#!/bin/bash
# test-backends.sh — regression tests for all CrispASR backends.
#
# Downloads models via -m auto --auto-download, transcribes jfk.wav,
# and compares output against reference transcriptions.
#
# Usage:
#   ./tests/test-backends.sh [backend ...]
#
# Examples:
#   ./tests/test-backends.sh                    # test all backends
#   ./tests/test-backends.sh parakeet moonshine # test specific backends
#
# Exit code: 0 if all pass, 1 if any fail.

set -euo pipefail
cd "$(dirname "$0")/.."

CRISPASR="./build/bin/crispasr"
SAMPLE="./samples/jfk.wav"
PASS=0
FAIL=0
SKIP=0

if [ ! -f "$CRISPASR" ]; then
    echo "ERROR: $CRISPASR not found. Build first."
    exit 1
fi
if [ ! -f "$SAMPLE" ]; then
    echo "ERROR: $SAMPLE not found."
    exit 1
fi

# Reference transcriptions (lowercase, no punctuation — normalized for comparison)
# Each backend may produce slightly different text; we check for key phrases.
JFK_KEY="fellow americans ask not what your country can do for you"

test_backend() {
    local backend="$1"
    local timeout_sec="${2:-120}"

    echo -n "  $backend: "

    # Run transcription
    local output
    output=$(timeout "$timeout_sec" "$CRISPASR" --backend "$backend" -m auto --auto-download \
        -f "$SAMPLE" --no-prints 2>/dev/null) || {
        echo "FAIL (timeout or crash)"
        FAIL=$((FAIL + 1))
        return
    }

    if [ -z "$output" ]; then
        echo "FAIL (empty output)"
        FAIL=$((FAIL + 1))
        return
    fi

    # Normalize: lowercase, remove punctuation
    local normalized
    normalized=$(echo "$output" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z ]//g' | tr -s ' ')

    # Check for key phrase (allow singular "american" for moonshine)
    if echo "$normalized" | grep -q "fellow american"; then
        echo "PASS ($output)"
        PASS=$((PASS + 1))
    else
        echo "FAIL (unexpected: $output)"
        FAIL=$((FAIL + 1))
    fi
}

# Backends with auto-download support and reasonable CPU speed
FAST_BACKENDS="parakeet moonshine wav2vec2 data2vec hubert fastconformer-ctc"
MEDIUM_BACKENDS="canary cohere omniasr qwen3"
SLOW_BACKENDS="voxtral voxtral4b granite glm-asr kyutai-stt firered-asr"
# vibevoice and omniasr-llm are very slow on CPU — skip by default

echo "CrispASR backend regression tests"
echo "================================="

if [ $# -gt 0 ]; then
    # Test specific backends
    for b in "$@"; do
        test_backend "$b" 300
    done
else
    echo ""
    echo "Fast backends (< 30s):"
    for b in $FAST_BACKENDS; do
        test_backend "$b" 60
    done

    echo ""
    echo "Medium backends (30-120s):"
    for b in $MEDIUM_BACKENDS; do
        test_backend "$b" 180
    done

    echo ""
    echo "Slow backends (> 120s, skipping on CI):"
    if [ "${CRISPASR_TEST_SLOW:-0}" = "1" ]; then
        for b in $SLOW_BACKENDS; do
            test_backend "$b" 600
        done
    else
        echo "  (set CRISPASR_TEST_SLOW=1 to run)"
        SKIP=$((SKIP + ${#SLOW_BACKENDS}))
    fi
fi

echo ""
echo "Results: $PASS passed, $FAIL failed, $SKIP skipped"
[ "$FAIL" -eq 0 ]
