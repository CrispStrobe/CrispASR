#!/bin/bash
# test-dry-run-resolve.sh — regression for --dry-run-resolve model-key matching.
#
# build_preview() must mirror crispasr_resolve_model's match priority, or the
# preview lies about which model a `-m <alias>` selects. Specifically, a
# sub-variant registry key (e.g. parakeet-tdt_ctc-110m) must resolve to its own
# entry rather than being shadowed by the filename-inferred backend default.
#
# Offline: --dry-run-resolve only previews registry resolution; it never
# downloads. No models required. SKIPs only if the binary is missing.

set -uo pipefail
cd "$(dirname "$0")/.."

CRISPASR="${1:-}"
if [ -z "$CRISPASR" ]; then
    for cand in build/bin/crispasr build-ninja-compile/bin/crispasr ./bin/crispasr; do
        if [ -x "$cand" ]; then CRISPASR="$cand"; break; fi
    done
fi
if [ -z "$CRISPASR" ]; then
    echo "SKIP: crispasr binary not found (build first)"; exit 0
fi

PASS=0; FAIL=0

# #397: a downloaded community Piper voice is not necessarily a registry row.
# A bare explicit filename must resolve from CRISPASR_MODELS_DIR as that exact
# voice, before the backend fallback offers the unrelated US Lessac default.
PIPER_CACHE=$(mktemp -d "${CRISPASR_TEST_TMPDIR:-$PWD}/.piper-resolve.XXXXXX")
trap 'rm -rf "$PIPER_CACHE"' EXIT
PIPER_NAME=piper-en_GB-cori-medium-f16.gguf
printf 'test fixture' > "$PIPER_CACHE/$PIPER_NAME"
got=$(CRISPASR_MODELS_DIR="$PIPER_CACHE" "$CRISPASR" -m "$PIPER_NAME" \
    --dry-run-resolve 2>&1 | sed -n 's/^  path:[[:space:]]*//p' | head -1)
if [ "$got" = "$PIPER_CACHE/$PIPER_NAME" ]; then
    echo "  ✓ bare Piper community voice → exact CRISPASR_MODELS_DIR file"
    PASS=$((PASS+1))
else
    echo "  ✗ bare Piper community voice → '$got' (expected '$PIPER_CACHE/$PIPER_NAME')"
    FAIL=$((FAIL+1))
fi

# arg|expected-registry-filename[|backend (default parakeet)]
CASES="
parakeet-tdt_ctc-110m|parakeet-tdt_ctc-110m-q4_k.gguf
parakeet|parakeet-tdt-0.6b-v3-q4_k.gguf
parakeet-v2|parakeet-tdt-0.6b-v2-q4_k.gguf
parakeet-tdt-1.1b|parakeet-tdt-1.1b-q4_k.gguf
orukeet|orukeet-q4_k.gguf
confucius4-r2t2|confucius4-r2t2-q4_k.gguf
dolphin|dolphin-cn-dialect-small-streaming-q4_k.gguf|dolphin
xasr|x-asr-zh-en-q8_0.gguf|xasr
"
while IFS='|' read -r arg expected backend; do
    [ -z "$arg" ] && continue
    got=$("$CRISPASR" -m "$arg" --backend "${backend:-parakeet}" --auto-download --dry-run-resolve 2>&1 \
        | sed -n 's/^  registry:[[:space:]]*//p' | head -1)
    if [ "$got" = "$expected" ]; then
        echo "  ✓ -m $arg → $got"; PASS=$((PASS+1))
    else
        echo "  ✗ -m $arg → '$got' (expected '$expected')"; FAIL=$((FAIL+1))
    fi
done <<< "$CASES"

echo
echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
