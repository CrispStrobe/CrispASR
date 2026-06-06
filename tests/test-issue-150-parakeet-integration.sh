#!/usr/bin/env bash
# test-issue-150-parakeet-integration.sh — comprehensive integration test
# for issue #150 (CJK 42-char fallback splitting English words) and the
# related parakeet dispatcher paths: chunking, VAD, LCS dedup, split-on-punct.
#
# Generates test audio of various lengths (5s–90s) from samples/jfk.wav,
# then runs 74 combinations of:
#   - --split-on-punct on 9 audio lengths (Bug 1: no mid-word splits)
#   - default settings on 9 audio lengths (Bug 2: no audio discarded)
#   - --chunk-seconds 0/15/30/60 x 4 audio lengths
#   - --chunk-overlap 0/1.5/3.0/5.0 x 2 audio lengths
#   - --vad (silero default) x 4 audio lengths + 3 thresholds
#   - --vad-model firered x 3 audio lengths
#   - --lcs-dedup auto/on/off x 3 audio lengths
#   - cross-combinations of all the above
#
# Requires: ffmpeg (for test audio generation), parakeet-tdt-0.6b-v3 model
#           (auto-downloaded via -m auto).
#
# Usage: ./tests/test-issue-150-parakeet-integration.sh
# Exit:  0 = pass, 1 = fail, 2 = skipped (ffmpeg or model missing).

set -euo pipefail
cd "$(dirname "$0")/.."

CRISPASR="${CRISPASR_BIN:-./build/bin/crispasr}"
if [ ! -x "$CRISPASR" ]; then
    CRISPASR="./build-ninja-compile/bin/crispasr"
fi
if [ ! -x "$CRISPASR" ]; then
    echo "SKIP: crispasr binary not found" >&2
    exit 2
fi

if ! command -v ffmpeg &>/dev/null; then
    echo "SKIP: ffmpeg not found (needed to generate test audio)" >&2
    exit 2
fi

# Quick probe: does parakeet work at all?
if ! "$CRISPASR" --backend parakeet -m auto -f samples/jfk.wav \
    --no-prints 2>&1 | grep -qi "fellow"; then
    echo "SKIP: parakeet backend not functional or model unavailable" >&2
    exit 2
fi

# Generate test audio from jfk.wav (11 s)
AUDIO_DIR="$(mktemp -d)"
OUT_DIR="$(mktemp -d)"
JFK=samples/jfk.wav
trap 'rm -rf "$AUDIO_DIR" "$OUT_DIR"' EXIT

echo "Generating test audio in $AUDIO_DIR ..."
ffmpeg -y -i "$JFK" -t 5 -ar 16000 -ac 1 "$AUDIO_DIR/test_5s.wav" 2>/dev/null
ffmpeg -y -i "$JFK" -af "apad=whole_dur=15" -ar 16000 -ac 1 "$AUDIO_DIR/test_15s.wav" 2>/dev/null
ffmpeg -y -i "$JFK" -i "$JFK" -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1[a]" \
    -map "[a]" -ar 16000 -ac 1 "$AUDIO_DIR/test_22s.wav" 2>/dev/null
ffmpeg -y -i "$JFK" -i "$JFK" -i "$JFK" \
    -filter_complex "[0:a][1:a][2:a]concat=n=3:v=0:a=1,apad=whole_dur=30[a]" \
    -map "[a]" -ar 16000 -ac 1 "$AUDIO_DIR/test_30s.wav" 2>/dev/null
ffmpeg -y -i "$JFK" -i "$JFK" -i "$JFK" \
    -filter_complex "[0:a][1:a][2:a]concat=n=3:v=0:a=1,apad=whole_dur=35[a]" \
    -map "[a]" -ar 16000 -ac 1 "$AUDIO_DIR/test_35s.wav" 2>/dev/null
ffmpeg -y -i "$JFK" -i "$JFK" -i "$JFK" -i "$JFK" \
    -filter_complex "[0:a][1:a][2:a][3:a]concat=n=4:v=0:a=1,apad=whole_dur=45[a]" \
    -map "[a]" -ar 16000 -ac 1 "$AUDIO_DIR/test_45s.wav" 2>/dev/null
ffmpeg -y -i "$JFK" -i "$JFK" -i "$JFK" -i "$JFK" -i "$JFK" -i "$JFK" \
    -filter_complex "[0:a][1:a][2:a][3:a][4:a][5:a]concat=n=6:v=0:a=1,atrim=0:60[a]" \
    -map "[a]" -ar 16000 -ac 1 "$AUDIO_DIR/test_60s.wav" 2>/dev/null
ffmpeg -y -i "$JFK" -i "$JFK" -i "$JFK" -i "$JFK" -i "$JFK" -i "$JFK" -i "$JFK" -i "$JFK" \
    -filter_complex "[0:a][1:a][2:a][3:a][4:a][5:a][6:a][7:a]concat=n=8:v=0:a=1,apad=whole_dur=90[a]" \
    -map "[a]" -ar 16000 -ac 1 "$AUDIO_DIR/test_90s.wav" 2>/dev/null

PASS=0
FAIL=0
ERRORS=""

run_test() {
    local test_name="$1"
    local expected_check="$2"  # "no_mid_word_split" or "coverage_check" or "no_crash"
    shift 2
    local audio_file="$1"
    shift

    local out_base="$OUT_DIR/$(echo "$test_name" | tr ' /' '_')"
    local out_file="${out_base}.txt"
    local srt_file="${out_base}.srt"

    echo -n "  [$test_name] ... "

    if ! "$CRISPASR" --backend parakeet -m auto -f "$audio_file" \
        --output-srt -of "$out_base" \
        "$@" > "$out_file" 2>&1; then
        echo "FAIL (crashed)"
        FAIL=$((FAIL + 1))
        ERRORS="${ERRORS}\n  FAIL: $test_name (crash): $(tail -3 "$out_file")"
        return
    fi

    local result="PASS"

    case "$expected_check" in
        no_mid_word_split)
            if [ -f "$srt_file" ]; then
                if grep -q "abo$" "$srt_file" 2>/dev/null; then
                    result="FAIL (mid-word split: 'abo' found)"
                fi
            fi
            ;;
        coverage_check)
            if [ -f "$srt_file" ]; then
                local audio_dur
                audio_dur=$(ffprobe -v quiet -show_entries format=duration \
                    -of csv=p=0 "$audio_file" 2>/dev/null | cut -d. -f1)
                local last_ts
                last_ts=$(grep -oP '\d{2}:\d{2}:\d{2},\d{3}' "$srt_file" \
                    | tail -1 2>/dev/null || echo "00:00:00,000")
                local last_sec
                last_sec=$(echo "$last_ts" | awk -F'[:,]' '{print $1*3600 + $2*60 + $3}')
                local min_coverage=$((audio_dur * 60 / 100))
                if [ "$last_sec" -lt "$min_coverage" ]; then
                    result="FAIL (coverage: last_ts=${last_sec}s, audio=${audio_dur}s, need ${min_coverage}s)"
                fi
            else
                result="FAIL (no SRT output)"
            fi
            ;;
        no_crash)
            ;;
    esac

    if [[ "$result" == "PASS" ]]; then
        PASS=$((PASS + 1))
        echo "PASS"
    else
        FAIL=$((FAIL + 1))
        echo "$result"
        ERRORS="${ERRORS}\n  $result: $test_name"
    fi
}

echo "=========================================="
echo "CrispASR Issue #150 Integration Test Suite"
echo "=========================================="
echo ""

# ──────────────────────────────────────────────
echo "=== Bug 1: --split-on-punct word splitting ==="
echo ""

for audio in test_5s test_15s test_22s test_30s test_35s test_45s test_60s test_90s; do
    run_test "split-on-punct ${audio}" no_mid_word_split \
        "$AUDIO_DIR/${audio}.wav" --split-on-punct
done

run_test "split-on-punct jfk_11s" no_mid_word_split \
    "$JFK" --split-on-punct

echo ""

# ──────────────────────────────────────────────
echo "=== Bug 2: Audio coverage (no audio discarded) ==="
echo ""

for audio in test_5s test_15s test_22s test_30s test_35s test_45s test_60s test_90s; do
    run_test "coverage default ${audio}" coverage_check \
        "$AUDIO_DIR/${audio}.wav"
done

run_test "coverage jfk_11s" coverage_check "$JFK"

echo ""

# ──────────────────────────────────────────────
echo "=== Chunking: --chunk-seconds variations ==="
echo ""

for ck in 0 15 30 60; do
    for audio in test_22s test_35s test_60s test_90s; do
        run_test "chunk-${ck}s ${audio}" coverage_check \
            "$AUDIO_DIR/${audio}.wav" --chunk-seconds "$ck"
    done
done

echo ""

# ──────────────────────────────────────────────
echo "=== Chunking: --chunk-overlap variations ==="
echo ""

for co in 0 1.5 3.0 5.0; do
    for audio in test_35s test_60s; do
        run_test "overlap-${co}s ${audio}" coverage_check \
            "$AUDIO_DIR/${audio}.wav" --chunk-seconds 30 --chunk-overlap "$co"
    done
done

echo ""

# ──────────────────────────────────────────────
echo "=== VAD: different providers ==="
echo ""

for audio in test_5s test_22s test_35s test_60s; do
    run_test "vad-default ${audio}" coverage_check \
        "$AUDIO_DIR/${audio}.wav" --vad
done

for vt in 0.3 0.5 0.7; do
    run_test "vad-thresh-${vt} test_35s" coverage_check \
        "$AUDIO_DIR/test_35s.wav" --vad --vad-threshold "$vt"
done

for audio in test_22s test_35s test_60s; do
    run_test "vad-firered ${audio}" no_crash \
        "$AUDIO_DIR/${audio}.wav" --vad-model firered
done

echo ""

# ──────────────────────────────────────────────
echo "=== LCS dedup: auto/on/off ==="
echo ""

for lcs in auto on off; do
    for audio in test_35s test_60s test_90s; do
        run_test "lcs-${lcs} ${audio}" coverage_check \
            "$AUDIO_DIR/${audio}.wav" --lcs-dedup "$lcs"
    done
done

echo ""

# ──────────────────────────────────────────────
echo "=== Cross-combinations ==="
echo ""

for ck in 0 15 30; do
    run_test "split-on-punct+chunk-${ck}s test_60s" no_mid_word_split \
        "$AUDIO_DIR/test_60s.wav" --split-on-punct --chunk-seconds "$ck"
done

run_test "split-on-punct+vad test_35s" no_mid_word_split \
    "$AUDIO_DIR/test_35s.wav" --split-on-punct --vad

run_test "split-on-punct+vad test_60s" no_mid_word_split \
    "$AUDIO_DIR/test_60s.wav" --split-on-punct --vad

run_test "vad+chunk-30s test_60s" coverage_check \
    "$AUDIO_DIR/test_60s.wav" --vad --chunk-seconds 30

run_test "lcs-on+chunk-30+split test_60s" no_mid_word_split \
    "$AUDIO_DIR/test_60s.wav" --lcs-dedup on --chunk-seconds 30 --split-on-punct

run_test "lcs-off+chunk-30+split test_60s" no_mid_word_split \
    "$AUDIO_DIR/test_60s.wav" --lcs-dedup off --chunk-seconds 30 --split-on-punct

run_test "vad+lcs-on test_60s" coverage_check \
    "$AUDIO_DIR/test_60s.wav" --vad --lcs-dedup on

run_test "vad+lcs-off test_60s" coverage_check \
    "$AUDIO_DIR/test_60s.wav" --vad --lcs-dedup off

run_test "all-options test_60s" no_mid_word_split \
    "$AUDIO_DIR/test_60s.wav" --split-on-punct --vad --lcs-dedup on

run_test "all-options test_90s" no_mid_word_split \
    "$AUDIO_DIR/test_90s.wav" --split-on-punct --vad --lcs-dedup on

run_test "no-vad+no-chunk+split test_60s" no_mid_word_split \
    "$AUDIO_DIR/test_60s.wav" --split-on-punct --chunk-seconds 0

echo ""

# ──────────────────────────────────────────────
echo "=========================================="
echo "Results: $PASS passed, $FAIL failed"
echo "=========================================="

if [ "$FAIL" -gt 0 ]; then
    echo -e "\nFailures:$ERRORS"
    exit 1
else
    echo "All tests passed!"
    exit 0
fi
