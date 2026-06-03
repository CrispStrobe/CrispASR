#!/bin/bash
# Systematic benchmark of all ASR backends on jfk.wav (11s)
# Usage: bash tools/benchmark_all.sh [threads]
set -euo pipefail

THREADS="${1:-4}"
AUDIO="samples/jfk.wav"
BIN="./build/bin/crispasr"
DIR="/mnt/akademie_storage/test_cohere"

echo "================================================================"
echo "CrispASR Systematic Benchmark"
echo "Audio: $AUDIO (11.0s), Threads: $THREADS"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

# Enable per-backend timing env vars
export WAV2VEC2_BENCH=1
export OMNIASR_BENCH=1

declare -A MODELS=(
    ["whisper"]="models/ggml-base.en.bin"
    ["wav2vec2"]="$DIR/wav2vec2-xlsr-en-q4_k.gguf"
    ["wav2vec2-base"]="$DIR/wav2vec2-base-voxpopuli-it-q4_k.gguf"
    ["data2vec"]="$DIR/data2vec-audio-base-960h-q4_k.gguf"
    ["hubert"]="$DIR/hubert-large-ls960-ft-q4_k.gguf"
    ["omniasr-ctc-300m"]="$DIR/omniasr-ctc-300m-q4_k.gguf"
    ["omniasr-ctc-1b"]="$DIR/omniasr-ctc-1b-q4_k.gguf"
    ["omniasr-llm"]="$DIR/omniasr-llm-300m-v2-q4_k.gguf"
    ["parakeet"]="$DIR/parakeet-tdt-0.6b-v3-q4_k.gguf"
    ["canary"]="$DIR/canary-1b-v2-q4_k.gguf"
    ["cohere"]="$DIR/cohere-transcribe-q4_k.gguf"
    ["qwen3"]="$DIR/qwen3-asr-0.6b-q4_k.gguf"
    ["glm-asr"]="$DIR/glm-asr-nano-q4_k.gguf"
    ["kyutai-stt"]="$DIR/kyutai-stt-1b-q4_k.gguf"
    ["firered-asr"]="$DIR/firered-asr2-aed-q4_k.gguf"
    ["moonshine-tiny"]="$DIR/moonshine-tiny-q4_k.gguf"
    ["moonshine-base"]="$DIR/moonshine-base-q4_k.gguf"
    ["moonshine-streaming"]="$DIR/moonshine-streaming-tiny-q4_k.gguf"
    ["fc-ctc-large"]="$DIR/stt-en-fastconformer-ctc-large-q4_k.gguf"
    ["fc-ctc-xlarge"]="$DIR/stt-en-fastconformer-ctc-xlarge-q4_k.gguf"
    ["granite"]="$DIR/granite-speech-1b-q4_k.gguf"
    ["granite-4.1"]="$DIR/granite-speech-4.1-2b-q4_k.gguf"
    ["granite-4.1-plus"]="$DIR/granite-speech-4.1-2b-plus-q4_k.gguf"
    ["voxtral"]="$DIR/voxtral-mini-3b-2507-q4_k.gguf"
    ["voxtral4b"]="$DIR/voxtral-mini-4b-realtime-q4_k.gguf"
    ["gemma4-e2b"]="$DIR/gemma4-e2b-it-q4_k.gguf"
    ["mimo-asr"]="$DIR/mimo-asr-q4_k.gguf"
    ["funasr"]="$DIR/funasr-nano-2512-q8_0.gguf"
    ["sensevoice"]="$DIR/sensevoice-small-q4_k.gguf"
    ["paraformer"]="$DIR/paraformer-zh-q4_k.gguf"
)

for name in whisper wav2vec2 wav2vec2-base data2vec hubert \
            omniasr-ctc-300m omniasr-ctc-1b omniasr-llm \
            parakeet canary cohere qwen3 glm-asr kyutai-stt \
            firered-asr moonshine-tiny moonshine-base moonshine-streaming \
            fc-ctc-large fc-ctc-xlarge granite granite-4.1 granite-4.1-plus \
            voxtral voxtral4b gemma4-e2b mimo-asr \
            funasr sensevoice paraformer; do
    model="${MODELS[$name]:-}"
    [ -z "$model" ] && continue
    [ ! -f "$model" ] && { echo "  SKIP $name (model not found)"; continue; }

    # Determine backend name
    backend="$name"
    case "$name" in
        wav2vec2-base|data2vec|hubert) backend="wav2vec2" ;;
        omniasr-ctc-*|omniasr-llm) backend="omniasr" ;;
        moonshine-tiny|moonshine-base) backend="moonshine" ;;
        moonshine-streaming) backend="moonshine-streaming" ;;
        fc-ctc-*) backend="fastconformer-ctc" ;;
        granite-4.1|granite-4.1-plus) backend="granite" ;;
    esac

    echo ""
    echo "--- $name ($backend) ---"
    timeout 120 "$BIN" --backend "$backend" -m "$model" -f "$AUDIO" -l en -t "$THREADS" 2>&1 | \
        grep -E "transcribed|performance|audio|CNN|pos conv|encoder|decode|prefill|total|normalize|alloc|compute|read|decoded" || \
        echo "  TIMEOUT or ERROR"
done

echo ""
echo "================================================================"
echo "Benchmark complete"
echo "================================================================"
