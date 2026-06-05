#!/bin/bash
# tests/test_diarize_live.sh — live integration test for global sherpa diarization (#110).
#
# Requires:
#   - crispasr binary (built with sherpa support)
#   - sherpa-onnx-offline-speaker-diarization binary
#   - pyannote segmentation ONNX model
#   - speaker embedding ONNX model
#   - test audio files
#
# Usage:
#   SHERPA_BIN=/path/to/sherpa-onnx-offline-speaker-diarization \
#   SEG_MODEL=/path/to/model.onnx \
#   EMB_MODEL=/path/to/embedding.onnx \
#   CRISPASR=/path/to/crispasr \
#   bash tests/test_diarize_live.sh
#
# Defaults (this VPS):
SHERPA_BIN="${SHERPA_BIN:-/mnt/storage/sherpa-onnx/sherpa-onnx-v1.13.2-linux-x64-static/bin/sherpa-onnx-offline-speaker-diarization}"
SEG_MODEL="${SEG_MODEL:-/mnt/akademie_storage/test_cohere/sherpa-onnx-pyannote-segmentation-3-0/model.onnx}"
EMB_MODEL="${EMB_MODEL:-/mnt/storage/sherpa-onnx/models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx}"
CRISPASR="${CRISPASR:-/tmp/build-diarize/bin/crispasr}"
PARAFORMER_MODEL="${PARAFORMER_MODEL:-/mnt/storage/paraformer-zh/paraformer-zh-q4_k.gguf}"
JFK_WAV="${JFK_WAV:-/mnt/akademie_storage/whisper.cpp/samples/jfk.wav}"
ZH_WAV="${ZH_WAV:-/mnt/storage/paraformer-zh-upstream/example/asr_example.wav}"

set -e
PASS=0
FAIL=0

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

# Prerequisite checks
for f in "$SHERPA_BIN" "$SEG_MODEL" "$EMB_MODEL" "$CRISPASR"; do
    if [ ! -f "$f" ]; then
        echo "SKIP: required file not found: $f"
        exit 0
    fi
done

echo "=== Global Diarization Live Tests (#110) ==="
echo ""

# Test 1: sherpa binary works standalone
echo "--- Test 1: sherpa binary smoke test ---"
SHERPA_OUT=$($SHERPA_BIN \
    --clustering.num-clusters=1 \
    --segmentation.pyannote-model="$SEG_MODEL" \
    --embedding.model="$EMB_MODEL" \
    "$JFK_WAV" 2>/dev/null | grep "speaker_")
check "sherpa binary produces speaker regions" test -n "$SHERPA_OUT"

# Test 2: global diarization path is used (not per-slice)
echo ""
echo "--- Test 2: global sherpa path ---"
DIARIZE_LOG=$($CRISPASR \
    --backend paraformer -m "$PARAFORMER_MODEL" -f "$ZH_WAV" \
    --diarize --diarize-method sherpa \
    --sherpa-bin "$SHERPA_BIN" --sherpa-segment-model "$SEG_MODEL" \
    --sherpa-embedding-model "$EMB_MODEL" --sherpa-num-clusters 1 \
    2>&1)
echo "$DIARIZE_LOG" | grep -q "computing global sherpa timeline"
check "global pre-compute message present" test $? -eq 0
echo "$DIARIZE_LOG" | grep -q "sherpa global"
check "sherpa global regions parsed" test $? -eq 0
echo "$DIARIZE_LOG" | grep -q "(speaker 0)"
check "speaker label assigned" test $? -eq 0

# Test 3: single-speaker produces consistent label
echo ""
echo "--- Test 3: single speaker consistency ---"
OUTPUT=$(echo "$DIARIZE_LOG" | grep "^\[" | head -1)
echo "$OUTPUT" | grep -q "(speaker 0)"
check "single speaker file gets speaker 0" test $? -eq 0

# Test 4: transcript is still correct with diarization enabled
echo ""
echo "--- Test 4: transcript correctness with diarize ---"
TRANSCRIPT=$(echo "$DIARIZE_LOG" | grep "^\[" | sed 's/.*) //')
EXPECTED="正是因为存在绝对正义所以我们接受现实的相对正义但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义"
check "Chinese transcript matches with diarize on" test "$TRANSCRIPT" = "$EXPECTED"

# Test 5: JFK with diarization
echo ""
echo "--- Test 5: JFK English with global diarize ---"
JFK_OUT=$($CRISPASR \
    --backend paraformer -m "$PARAFORMER_MODEL" -f "$JFK_WAV" \
    --diarize --diarize-method sherpa \
    --sherpa-bin "$SHERPA_BIN" --sherpa-segment-model "$SEG_MODEL" \
    --sherpa-embedding-model "$EMB_MODEL" --sherpa-num-clusters 1 \
    --no-prints 2>/dev/null | head -1)
echo "$JFK_OUT" | grep -q "(speaker"
check "JFK has speaker label" test $? -eq 0
echo "$JFK_OUT" | grep -q "americans"
check "JFK transcript present" test $? -eq 0

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ $FAIL -eq 0 ]
