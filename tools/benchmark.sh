#!/bin/bash
# tools/benchmark.sh — comprehensive CrispASR benchmark
#
# Runs each backend on standard audio samples at multiple lengths,
# measures wall time + real-time factor, and prints a markdown table.
#
# Usage:
#   bash tools/benchmark.sh [--backends "parakeet canary"] [--audio-dir /path]
#
# Requirements:
#   - build/bin/crispasr must exist
#   - Model GGUFs at the paths listed in MODELS below

set -euo pipefail

CRISPASR="${CRISPASR:-./build/bin/crispasr}"
THREADS="${THREADS:-4}"

# ---- Models (backend → GGUF path) ----
declare -A MODELS=(
  [parakeet]="/mnt/akademie_storage/test_cohere/parakeet-tdt-0.6b-v3.gguf"
  [canary]="/mnt/akademie_storage/test_cohere/canary-1b-v2.gguf"
  [cohere]="/mnt/akademie_storage/test_cohere/cohere-transcribe-q5_0.gguf"
  [granite]="/mnt/akademie_storage/granite_gguf/granite-speech-4.0-1b-q5_0.gguf"
  [voxtral]="/mnt/akademie_storage/test_cohere/voxtral-mini-3b-2507-q4_k.gguf"
  [qwen3]="/mnt/akademie_storage/test_cohere/qwen3-asr-0.6b-q4_k.gguf"
  [fastconformer-ctc]="/mnt/akademie_storage/test_cohere/stt-en-fastconformer-ctc-large-q4_k.gguf"
)

# ---- Audio samples ----
declare -A AUDIOS=(
  [jfk_11s]="samples/jfk.wav"
  [obama_89s]="/mnt/akademie_storage/test_cohere/tests/obama_speech_16k.wav"
  [german_79s]="/mnt/akademie_storage/german-samples/De-Abwasch-article.wav"
)

# ---- Check prerequisites ----
if [ ! -x "$CRISPASR" ]; then
  echo "ERROR: $CRISPASR not found. Build first: cmake --build build --target crispasr-cli" >&2
  exit 1
fi

echo "# CrispASR Benchmark"
echo ""
echo "Date: $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "CPU: $(grep 'model name' /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || echo 'unknown')"
echo "RAM: $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo 'unknown')"
echo "Threads: $THREADS"
echo ""

# ---- Header ----
printf "| %-20s | %-15s | %8s | %8s | %6s | %-60s |\n" \
       "Backend" "Audio" "Duration" "Time(s)" "RTF" "Transcript (first 60 chars)"
printf "|%-22s|%-17s|%10s|%10s|%8s|%-62s|\n" \
       "$(printf -- '-%.0s' {1..22})" "$(printf -- '-%.0s' {1..17})" \
       "$(printf -- '-%.0s' {1..10})" "$(printf -- '-%.0s' {1..10})" \
       "$(printf -- '-%.0s' {1..8})" "$(printf -- '-%.0s' {1..62})"

# ---- Run each combo ----
for audio_name in "${!AUDIOS[@]}"; do
  audio_path="${AUDIOS[$audio_name]}"
  if [ ! -f "$audio_path" ]; then
    echo "# SKIP: $audio_path not found" >&2
    continue
  fi

  # Get duration
  audio_dur=$(/opt/miniconda/bin/python3 -c "
import wave
try:
    w = wave.open('$audio_path', 'rb')
    print(f'{w.getnframes() / w.getframerate():.1f}')
except: print('0')
" 2>/dev/null)

  for backend in "${!MODELS[@]}"; do
    model="${MODELS[$backend]}"
    if [ ! -f "$model" ]; then
      printf "| %-20s | %-15s | %8ss | %8s | %6s | %-60s |\n" \
             "$backend" "$audio_name" "$audio_dur" "SKIP" "-" "model not found"
      continue
    fi

    # Run with timing
    start_ns=$(date +%s%N)
    transcript=$("$CRISPASR" --backend "$backend" -m "$model" -f "$audio_path" \
                 -np -t "$THREADS" 2>/dev/null || echo "ERROR")
    end_ns=$(date +%s%N)

    elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
    elapsed_s=$(echo "scale=2; $elapsed_ms / 1000" | bc)

    if [ "$audio_dur" != "0" ] && [ "$elapsed_ms" -gt 0 ]; then
      rtf=$(echo "scale=3; $elapsed_s / $audio_dur" | bc)
    else
      rtf="?"
    fi

    # Truncate transcript for display
    short=$(echo "$transcript" | head -1 | cut -c1-60)

    printf "| %-20s | %-15s | %7ss | %7ss | %5sx | %-60s |\n" \
           "$backend" "$audio_name" "$audio_dur" "$elapsed_s" "$rtf" "$short"
  done
done

echo ""
echo "RTF = Real-Time Factor (lower is faster; <1.0 = faster than real-time)"
