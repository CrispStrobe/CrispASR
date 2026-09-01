#!/usr/bin/env bash
# test-htdemucs-separate-live.sh — htdemucs source-separation live test
# (#413/#414).
#
# On a CPU-only host the #414 AUTO gates must keep the legacy BLAS path
# (no behavior change), and the opt-in graph path must produce stems that
# match it. Three arms on a short clip:
#   A  defaults            — BLAS path (AUTO off-GPU), 4 stems written
#   B  GGML=1 FUSED=1      — CPU fused graph, 4 stems written
#   parity: per-stem max|diff| A-vs-B within tolerance (the same
#   equality the #398 Kaggle kernel proved for GPU-vs-BLAS)
#
# SKIPs cleanly when the model or binary is missing. The htdemucs GGUF is
# ~38 MB; a 5 s clip keeps the BLAS arm under a minute on 4 cores.
set -u

BIN="${CRISPASR_BIN:-./build/bin/crispasr}"
CLIP="${CRISPASR_SEP_CLIP:-samples/jfk.wav}"

MODEL="${CRISPASR_HTDEMUCS_MODEL:-}"
if [ -z "$MODEL" ] && [ -n "${CRISPASR_MODELS_DIR:-}" ]; then
    for c in "$CRISPASR_MODELS_DIR"/htdemucs-q4_k.gguf "$CRISPASR_MODELS_DIR"/htdemucs-*.gguf; do
        [ -f "$c" ] && MODEL="$c" && break
    done
fi

if [ ! -x "$BIN" ]; then
    echo "SKIP: crispasr binary not found at $BIN"
    exit 0
fi
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "SKIP: no htdemucs model (set CRISPASR_HTDEMUCS_MODEL or CRISPASR_MODELS_DIR)"
    exit 0
fi
if [ ! -f "$CLIP" ]; then
    echo "SKIP: no test clip at $CLIP"
    exit 0
fi

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
mkdir -p "$OUT/a" "$OUT/b"

run_arm() { # dir extra-env...
    local dir=$1
    shift
    env "$@" \
        "$BIN" --separate -m "$MODEL" -f "$CLIP" -d 5000 --sep-output-dir "$dir" \
        > "$dir/log.txt" 2>&1 || {
        echo "FAIL: --separate returned nonzero ($dir)"
        tail -5 "$dir/log.txt"
        exit 1
    }
    local n
    n=$(ls "$dir"/*_*.wav 2> /dev/null | wc -l)
    if [ "$n" -lt 4 ]; then
        echo "FAIL: expected 4 stems in $dir, got $n"
        ls -la "$dir"
        exit 1
    fi
}

echo "arm A (defaults / BLAS): $MODEL"
run_arm "$OUT/a"
grep -q "gates graph=0" "$OUT/a/log.txt" || {
    echo "FAIL: AUTO gates did not stay on BLAS on a CPU-only host"
    grep "gates" "$OUT/a/log.txt"
    exit 1
}

echo "arm B (GGML=1 FUSED=1 / CPU fused graph)"
run_arm "$OUT/b" CRISPASR_HTDEMUCS_GGML=1 CRISPASR_HTDEMUCS_FUSED=1
grep -q "gates graph=1 fused=1" "$OUT/b/log.txt" || {
    echo "FAIL: forced gates did not engage the fused graph"
    grep "gates" "$OUT/b/log.txt"
    exit 1
}

python3 - "$OUT/a" "$OUT/b" <<'EOF'
import sys, wave, numpy as np, glob, os
a_dir, b_dir = sys.argv[1], sys.argv[2]
worst = 0.0
for a_path in sorted(glob.glob(os.path.join(a_dir, "*_*.wav"))):
    b_path = os.path.join(b_dir, os.path.basename(a_path))
    if not os.path.exists(b_path):
        print(f"FAIL: stem missing in arm B: {os.path.basename(a_path)}")
        sys.exit(1)
    def load(p):
        with wave.open(p) as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    a, b = load(a_path), load(b_path)
    n = min(len(a), len(b))
    if n == 0:
        print(f"FAIL: empty stem {os.path.basename(a_path)}")
        sys.exit(1)
    d = float(np.abs(a[:n] - b[:n]).max())
    worst = max(worst, d)
    print(f"  {os.path.basename(a_path)}: max|diff| {d:.6f}")
# Same bar as the #398 GPU parity: numerically close, not bit-identical
# (different reduction trees). 1e-2 on int16-normalized audio is far above
# graph-order noise and far below any audible restructuring.
if worst > 1e-2:
    print(f"FAIL: BLAS-vs-graph stem divergence {worst:.6f} > 1e-2")
    sys.exit(1)
print(f"PASS: 4 stems, BLAS vs fused-graph parity (worst max|diff| {worst:.6f})")
EOF
exit $?
