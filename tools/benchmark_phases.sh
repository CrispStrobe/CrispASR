#!/bin/bash
# Systematic per-phase benchmark of all ASR backends
# Tests both short (11s) and long (55s) audio
set -euo pipefail

THREADS="${1:-4}"
BIN="./build/bin/crispasr"
DIR="/mnt/akademie_storage/test_cohere"
SHORT="samples/jfk.wav"
LONG="/tmp/jfk_55s.wav"

# Create long audio if missing
if [ ! -f "$LONG" ]; then
    python3 -c "
import wave
with wave.open('$SHORT','rb') as w:
    p=w.getparams(); f=w.readframes(w.getnframes())
with wave.open('$LONG','wb') as o:
    o.setparams(p); o.writeframes(f*5)
"
fi

export WAV2VEC2_BENCH=1
export OMNIASR_BENCH=1

echo "================================================================"
echo "CrispASR Phase Benchmark — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Threads: $THREADS, Short: 11s, Long: 55s"
echo "================================================================"

declare -A MODELS=(
    ["wav2vec2"]="$DIR/wav2vec2-xlsr-en-q4_k.gguf"
    ["data2vec"]="$DIR/data2vec-audio-base-960h-q4_k.gguf"
    ["omniasr-300m"]="$DIR/omniasr-ctc-300m-q4_k.gguf"
    ["omniasr-1b"]="$DIR/omniasr-ctc-1b-q4_k.gguf"
    ["parakeet"]="$DIR/parakeet-tdt-0.6b-v3-q4_k.gguf"
    ["canary"]="$DIR/canary-1b-v2-q4_k.gguf"
    ["cohere"]="$DIR/cohere-transcribe-q4_k.gguf"
    ["qwen3"]="$DIR/qwen3-asr-0.6b-q4_k.gguf"
    ["firered"]="$DIR/firered-asr2-aed-q4_k.gguf"
    ["moonshine"]="$DIR/moonshine-tiny-q4_k.gguf"
    ["fc-ctc"]="$DIR/stt-en-fastconformer-ctc-large-q4_k.gguf"
)

ORDER="wav2vec2 data2vec omniasr-300m omniasr-1b parakeet canary cohere qwen3 firered moonshine fc-ctc"

for name in $ORDER; do
    model="${MODELS[$name]:-}"
    [ -z "$model" ] && continue
    [ ! -f "$model" ] && { echo "SKIP $name"; continue; }

    backend="$name"
    case "$name" in
        data2vec) backend="wav2vec2" ;;
        omniasr-*) backend="omniasr" ;;
        firered) backend="firered-asr" ;;
        fc-ctc) backend="fastconformer-ctc" ;;
    esac

    for audio in "$SHORT" "$LONG"; do
        dur=$(python3 -c "import wave; w=wave.open('$audio','rb'); print(f'{w.getnframes()/16000:.0f}s')")
        echo ""
        echo "--- $name [$dur] ---"
        timeout 180 "$BIN" --backend "$backend" -m "$model" -f "$audio" -l en -t "$THREADS" -v 2>&1 | \
            grep -E "transcribed|performance|CNN|pos conv|encoder|decode|prefill|total|normalize|alloc|compute|read|mem.*RSS|verbose.*audio" || \
            echo "  TIMEOUT/ERROR"
    done
done

echo ""
echo "================================================================"
echo "Done"
echo "================================================================"
