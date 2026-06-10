#!/bin/bash
# env-live-tests.sh — set env vars for running integration / live tests.
#
# Usage: source tests/env-live-tests.sh && ctest --test-dir build --rerun-failed
#
# Override CRISPASR_MODELS_DIR to point at your local model cache:
#   CRISPASR_MODELS_DIR=/my/models source tests/env-live-tests.sh
#
# Models are looked up via CRISPASR_MODELS_DIR (defaults to ~/.cache/crispasr).
# The auto-download cache and well-known search dirs also probe this path.

CRISPASR_MODELS_DIR="${CRISPASR_MODELS_DIR:-$HOME/.cache/crispasr}"
export CRISPASR_MODELS_DIR

# ── Whisper (beam search, VAD tests) ──
# Whisper models use the ggml-*.bin naming convention and are typically in
# the auto-download cache (~/.cache/crispasr), not the GGUF model dir.
_whisper_cache="${HOME}/.cache/crispasr"
if [ -f "$CRISPASR_MODELS_DIR/ggml-tiny.bin" ]; then
    _whisper_default="$CRISPASR_MODELS_DIR/ggml-tiny.bin"
elif [ -f "$_whisper_cache/ggml-tiny.bin" ]; then
    _whisper_default="$_whisper_cache/ggml-tiny.bin"
else
    _whisper_default="$CRISPASR_MODELS_DIR/ggml-tiny.bin"
fi
export CRISPASR_MODEL_WHISPER="${CRISPASR_MODEL_WHISPER:-$_whisper_default}"
unset _whisper_cache _whisper_default

# ── Beam search backends ──
export CRISPASR_MODEL_GLM_ASR="${CRISPASR_MODEL_GLM_ASR:-$CRISPASR_MODELS_DIR/glm-asr-nano.gguf}"
export CRISPASR_MODEL_QWEN3_ASR="${CRISPASR_MODEL_QWEN3_ASR:-$CRISPASR_MODELS_DIR/qwen3-asr-0.6b.gguf}"
export CRISPASR_MODEL_CANARY="${CRISPASR_MODEL_CANARY:-$CRISPASR_MODELS_DIR/canary-1b-v2.gguf}"
export CRISPASR_MODEL_COHERE="${CRISPASR_MODEL_COHERE:-$CRISPASR_MODELS_DIR/cohere-transcribe.gguf}"

# ── Paraformer ──
export PARAFORMER_MODEL="${PARAFORMER_MODEL:-$CRISPASR_MODELS_DIR/paraformer-zh-f16.gguf}"
export PARAFORMER_MODEL_Q4K="${PARAFORMER_MODEL_Q4K:-$CRISPASR_MODELS_DIR/paraformer-zh-q4_k.gguf}"

# ── Diarization ──
export CRISPASR_TEST_DIARIZE_MODEL="${CRISPASR_TEST_DIARIZE_MODEL:-$CRISPASR_MODELS_DIR/pyannote-seg-3.0.gguf}"
export CRISPASR_TEST_TITANET_MODEL="${CRISPASR_TEST_TITANET_MODEL:-$CRISPASR_MODELS_DIR/titanet-large.gguf}"
export CRISPASR_TEST_DIARIZE_WAV="${CRISPASR_TEST_DIARIZE_WAV:-samples/multispeaker.wav}"

# ── Chat (LLM) — requires a llama.cpp-compatible chat model with a chat
# template (e.g. qwen2.5-0.5b-instruct, smollm2-360m). Harrier is an
# embedding model and won't work. Leave unset to skip the test.
if [ -n "${CRISPASR_CHAT_TEST_MODEL:-}" ]; then
    export CRISPASR_CHAT_TEST_MODEL
fi

echo "Live test env configured (CRISPASR_MODELS_DIR=$CRISPASR_MODELS_DIR)"
