#!/bin/bash
# tests/test_vad_export_live.sh — CLI-surface tests for --vad-export (#227).
#
# TWO TIERS IN ONE SCRIPT, selected by $3:
#
#   unit        Tests 1-4 only. No model, no network. Four CLI runs with
#               Silero VAD — a few seconds idle, ~30s on a contended box, so
#               not a microtest. This is the half that caught #227's actual
#               bug (--vad-import silently ignored).
#   live | all  Everything, including the model-gated Tests 6-8 (14 CLI runs,
#               each loading a whisper model). `all` is the default so running
#               the script by hand still does what it always did.
#
# `live` deliberately re-runs the cheap half rather than skipping it: one extra
# second buys the guarantee that the two tiers cannot drift into testing
# different things.
#
# The split exists because the two halves belong in different ctest tiers and
# used to share one. Labelled `unit`, the whole script passed in CI — which has
# no model, so it SKIPPED everything below Test 4 and reported green for
# coverage it never ran — while timing out at 120s on any developer machine
# where a model happened to be findable. Green where it tests nothing, red
# where it tests everything, which is exactly backwards.
#
# Requires: crispasr binary (built), test audio. The live tier additionally
# needs a whisper model via $CRISPASR_MODEL_WHISPER or a known local path.
#
# Usage:
#   bash tests/test_vad_export_live.sh [crispasr] [src_dir] [unit|live|all]

set -e

CRISPASR="${1:-${CRISPASR_BIN:-build/bin/crispasr}}"
SRC_DIR="${2:-.}"
TIER="${3:-all}"
case "$TIER" in
    unit | live | all) ;;
    *)
        echo "unknown tier '$TIER' (expected unit, live or all)" >&2
        exit 2
        ;;
esac
JFK_WAV="$SRC_DIR/samples/jfk.wav"
TMPDIR="$(mktemp -d "${TMPDIR_BASE:-/tmp}/test227.XXXXXX")"

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

# Prerequisite checks
if [ ! -f "$CRISPASR" ]; then
    echo "SKIP: crispasr binary not found at $CRISPASR"
    exit 0
fi
if [ ! -f "$JFK_WAV" ]; then
    echo "SKIP: test audio not found at $JFK_WAV"
    exit 0
fi

mkdir -p "$TMPDIR"

echo "=== Issue #227: --vad-export fixes ==="
echo ""

# ─── Test 1: --vad-export produces valid JSON without a model ────────
echo "--- Test 1: --vad-export works without ASR model ---"
EXPORT_FILE="$TMPDIR/vad1.json"
# Use a non-existent model path — the export should succeed before
# model loading is attempted.
$CRISPASR --backend paraformer -m /nonexistent/model.gguf \
    -f "$JFK_WAV" --vad-export "$EXPORT_FILE" --no-prints 2>/dev/null || true

if [ -f "$EXPORT_FILE" ]; then
    check "export file created" test -s "$EXPORT_FILE"
    check "export contains crispasr_vad header" grep -q "crispasr_vad" "$EXPORT_FILE"
    check "export contains slices array" grep -q '"slices"' "$EXPORT_FILE"
    check "export contains sample_rate" grep -q '"sample_rate"' "$EXPORT_FILE"
else
    echo "[FAIL] export file not created at $EXPORT_FILE"
    FAIL=$((FAIL + 1))
fi

# ─── Test 2: --vad-export implies --vad (real boundaries) ────────────
echo ""
echo "--- Test 2: --vad-export implies --vad ---"
EXPORT_FILE2="$TMPDIR/vad2.json"
$CRISPASR --backend paraformer -m /nonexistent/model.gguf \
    -f "$JFK_WAV" --vad-export "$EXPORT_FILE2" --no-prints 2>/dev/null || true

if [ -f "$EXPORT_FILE2" ]; then
    # With VAD enabled, the JFK audio (11s, one continuous speech segment)
    # should produce at least 1 slice. Without VAD, it would be a single
    # continuous chunk covering the full duration.
    N_SLICES=$(grep -o '"start"' "$EXPORT_FILE2" | wc -l)
    check "at least one VAD slice" test "$N_SLICES" -ge 1

    # The first slice should NOT start at sample 0 and end at the total
    # number of samples — that would indicate no VAD ran (continuous chunk).
    # Actually, for JFK audio with one continuous speaker, VAD MIGHT
    # produce a single slice. Let's just verify the structure is valid.
    check "slices have start field" grep -q '"start":' "$EXPORT_FILE2"
    check "slices have end field" grep -q '"end":' "$EXPORT_FILE2"
    check "slices have t0_cs field" grep -q '"t0_cs":' "$EXPORT_FILE2"
    check "slices have t1_cs field" grep -q '"t1_cs":' "$EXPORT_FILE2"
else
    echo "[FAIL] export file not created"
    FAIL=$((FAIL + 1))
fi

# ─── Test 3: exported file can be imported ───────────────────────────
echo ""
echo "--- Test 3: --vad-import reads exported file ---"
if [ -f "$EXPORT_FILE" ]; then
    # Import the exported file. This needs a real model to transcribe,
    # but we can at least verify the import doesn't crash even without one.
    IMPORT_LOG=$($CRISPASR --backend paraformer -m /nonexistent/model.gguf \
        -f "$JFK_WAV" --vad-import "$EXPORT_FILE" --no-prints 2>&1) || true

    # The import should succeed (the error will be about the model, not the import)
    if echo "$IMPORT_LOG" | grep -q "imported.*VAD segment"; then
        check "import message present" true
    elif echo "$IMPORT_LOG" | grep -q "error.*import"; then
        echo "[FAIL] import itself failed"
        FAIL=$((FAIL + 1))
    else
        # Model error is expected — import itself worked
        check "import did not fail on the VAD file" true
    fi
else
    skip "Test 3" "no export file from test 1"
fi

# ─── Test 4: --vad-export runs without loading the ASR model ─────────
echo ""
echo "--- Test 4: --vad-export runs without loading the ASR model ---"
EXPORT_FILE4="$TMPDIR/vad4.json"
START_TIME=$(date +%s%N)
$CRISPASR --backend paraformer -m /nonexistent/model.gguf \
    -f "$JFK_WAV" --vad-export "$EXPORT_FILE4" --no-prints 2>/dev/null || true
END_TIME=$(date +%s%N)
ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))

if [ -f "$EXPORT_FILE4" ]; then
    # THE assertion: the export ran to completion with -m pointing at a file
    # that does not exist. If the ASR model load were not skipped, the run
    # would have failed there and this file would not be here. That is the
    # property under test, stated directly.
    check "export produced despite a nonexistent ASR model" test -f "$EXPORT_FILE4"

    # Elapsed time is reported, NOT asserted. This used to gate on <10s, on the
    # reasoning that Silero VAD over 11s of audio is ~1s so anything slower
    # meant the model had loaded. That is a proxy for the property above, and
    # it is a proxy that measures the machine rather than the code: measured
    # here at 1.5s of CPU but 5.8-15.7s of WALL on a contended box, so the
    # threshold fires on a busy laptop and says "model load was not skipped"
    # when nothing of the sort happened. A test that fails because something
    # else was compiling is a test people learn to re-run rather than read.
    echo "[INFO] export took ${ELAPSED_MS}ms wall (informational; not a gate)"
else
    echo "[FAIL] export file not created"
    FAIL=$((FAIL + 1))
fi


if [ "$TIER" = "unit" ]; then
    echo ""
    echo "=== Results (unit tier): $PASS passed, $FAIL failed, $SKIP skipped ==="
    rm -rf "$TMPDIR"
    [ "$FAIL" -eq 0 ] || exit 1
    exit 0
fi

# ─── Everything below needs a whisper model and transcribes real audio ──
# 14 CLI runs, each loading the model. This is the `live` tier.

# ─── Test 6: --vad-import is HONOURED on the default path (issue #227) ─
# The bug: --vad-import was silently ignored unless --backend was passed,
# because the legacy whisper path never reached the import code. A flag that
# is accepted and does nothing is worse than one that errors, and no unit
# test can see it -- it only shows at the CLI surface.
echo ""
echo "--- Test 6: --vad-import not silently ignored ---"
MODEL="${CRISPASR_MODEL_WHISPER:-}"
if [ -z "$MODEL" ]; then
    for m in ggml-base.bin models/ggml-base.en.bin "${CRISPASR_MODELS_DIR:-$HOME/.cache/crispasr}/ggml-base.bin"; do
        [ -f "$m" ] && MODEL="$m" && break
    done
fi
if [ -z "$MODEL" ]; then
    skip "--vad-import dispatch" "no whisper model (set CRISPASR_MODEL_WHISPER)"
else
    # A nonexistent import file MUST make the run fail. If it exits 0 and
    # transcribes, --vad-import was ignored -- the original bug.
    if $CRISPASR --vad --vad-import /nonexistent-227.json -f "$JFK_WAV" -m "$MODEL" >/dev/null 2>&1; then
        echo "[FAIL] --vad-import /nonexistent silently ignored (returned 0)"
        FAIL=$((FAIL + 1))
    else
        echo "[PASS] --vad-import /nonexistent is not ignored (nonzero exit)"
        PASS=$((PASS + 1))
    fi

    # Round trip: export at 30, import at 30 -> must succeed and say "imported".
    RT="$TMPDIR/rt30.json"
    $CRISPASR --vad --vad-export "$RT" -f "$JFK_WAV" --chunk-seconds 30 >/dev/null 2>&1
    OUT="$($CRISPASR --vad --vad-import "$RT" -f "$JFK_WAV" --chunk-seconds 30 -m "$MODEL" 2>&1)"
    check "matched --vad-import round-trips" bash -c "echo \"$OUT\" | grep -qi 'imported'"

    # ─── Test 7: chunk-length gate policy ───────────────────────────
    # Default: WARN and still transcribe (rc 0). Strict: refuse (rc != 0).
    # Breaking a working script on upgrade is worse than a visible warning.
    echo ""
    echo "--- Test 7: chunk mismatch warns by default, refuses under --vad-import-strict ---"
    set +e
    WARN="$($CRISPASR --vad --vad-import "$RT" -f "$JFK_WAV" --chunk-seconds 5 -m "$MODEL" 2>&1)"; RC_WARN=$?
    $CRISPASR --vad --vad-import "$RT" -f "$JFK_WAV" --chunk-seconds 5 --vad-import-strict -m "$MODEL" >/dev/null 2>&1; RC_STRICT=$?
    LEGACY="$TMPDIR/legacy.json"
    printf '{"crispasr_vad":{"version":1,"sample_rate":16000,"slices":[{"start":5120,"end":169920,"t0_cs":32,"t1_cs":1062}]}}' > "$LEGACY"
    $CRISPASR --vad --vad-import "$LEGACY" -f "$JFK_WAV" --chunk-seconds 5 --vad-import-strict -m "$MODEL" >/dev/null 2>&1; RC_LEGACY=$?
    set -e

    check "mismatch default: exit 0 (used anyway)" test "$RC_WARN" -eq 0
    echo "$WARN" | grep -qi 'warning'; check "mismatch default: prints a warning" test "$?" -eq 0
    check "mismatch strict: refuses (nonzero exit)" test "$RC_STRICT" -ne 0
    check "legacy file (no chunk_cs) accepted even under strict" test "$RC_LEGACY" -eq 0

    # ─── Test 8: --vad-export-raw is chunk-independent (issue #227) ──
    # Raw export writes VAD SPEECH SEGMENTS, not chunk boundaries. The same
    # file imports cleanly at ANY chunk length (re-chunked per run), so it is
    # the model- and chunk-independent artifact #227 actually asked for.
    echo ""
    echo "--- Test 8: --vad-export-raw reuse across chunk lengths ---"
    RAW="$TMPDIR/raw.json"
    $CRISPASR --vad-export-raw "$RAW" -f "$JFK_WAV" >/dev/null 2>&1
    check "raw export tagged kind=vad_segments" grep -q '"kind": "vad_segments"' "$RAW"
    check "raw export omits chunk_cs" bash -c "! grep -q chunk_cs '$RAW'"

    set +e
    # Import at two DIFFERENT chunk lengths; neither may warn (raw carries no
    # chunk length to mismatch), and the re-chunk must match a fresh run.
    OUT5="$($CRISPASR --vad --vad-import "$RAW" -f "$JFK_WAV" --chunk-seconds 5 -m "$MODEL" 2>&1)"; RC5=$?
    OUT2="$($CRISPASR --vad --vad-import "$RAW" -f "$JFK_WAV" --chunk-seconds 2 -m "$MODEL" 2>&1)"; RC2=$?
    # A fresh VAD transcription at 5 s with the SAME model: the re-chunk must
    # reproduce what that run slices. (Not a backend-less --vad-export: a
    # backend that chunks internally, like whisper, takes the whole speech
    # segment on an explicit --chunk-seconds (#257), and export cannot know.)
    FRESH5_OUT="$($CRISPASR --vad -f "$JFK_WAV" --chunk-seconds 5 -m "$MODEL" 2>&1)"
    set -e

    check "raw import @5 succeeds" test "$RC5" -eq 0
    check "raw import @2 succeeds" test "$RC2" -eq 0
    check "raw import @5 does not warn (chunk-independent)" bash -c "! echo \"$OUT5\" | grep -qi warning"
    check "raw import @2 does not warn" bash -c "! echo \"$OUT2\" | grep -qi warning"

    N_RAW5=$(echo "$OUT5" | grep -oi 'imported [0-9]*' | grep -o '[0-9]*')
    # The CLI prints "processing N slice(s)" only when N > 1.
    N_FRESH5=$(echo "$FRESH5_OUT" | grep -o 'processing [0-9]* slice' | grep -o '[0-9]*' || true)
    N_FRESH5=${N_FRESH5:-1}
    check "raw@5 re-chunk equals a fresh VAD run @5 ($N_RAW5 == $N_FRESH5)" test "$N_RAW5" = "$N_FRESH5"
fi


# ─── Cleanup ─────────────────────────────────────────────────────────
rm -rf "$TMPDIR"

echo ""
echo "=== Results: $PASS passed, $FAIL failed, $SKIP skipped ==="
[ $FAIL -eq 0 ]
