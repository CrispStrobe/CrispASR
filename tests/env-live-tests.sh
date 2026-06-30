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
export CRISPASR_MODEL_HIGGS_STT="${CRISPASR_MODEL_HIGGS_STT:-$CRISPASR_MODELS_DIR/higgs-stt-q8_0.gguf}"
export CRISPASR_MODEL_CANARY="${CRISPASR_MODEL_CANARY:-$CRISPASR_MODELS_DIR/canary-1b-v2.gguf}"
export CRISPASR_MODEL_LFM2_EN="${CRISPASR_MODEL_LFM2_EN:-$CRISPASR_MODELS_DIR/lfm2-audio-1.5b-q5_k.gguf}"
export CRISPASR_MODEL_LFM2_JP="${CRISPASR_MODEL_LFM2_JP:-$CRISPASR_MODELS_DIR/lfm2-audio-1.5b-jp-q5_k.gguf}"
# dots.tts: F16 core (the CFG flow-match derails on full-q8) + vocoder companion.
export CRISPASR_MODEL_DOTS_TTS="${CRISPASR_MODEL_DOTS_TTS:-$CRISPASR_MODELS_DIR/dots-tts-soar-f16.gguf}"
export CRISPASR_MODEL_DOTS_TTS_VOCODER="${CRISPASR_MODEL_DOTS_TTS_VOCODER:-$CRISPASR_MODELS_DIR/dots-tts-soar-vocoder-f16.gguf}"
export CRISPASR_MODEL_COHERE="${CRISPASR_MODEL_COHERE:-$CRISPASR_MODELS_DIR/cohere-transcribe.gguf}"

# ── Paraformer ──
export PARAFORMER_MODEL="${PARAFORMER_MODEL:-$CRISPASR_MODELS_DIR/paraformer-zh-f16.gguf}"
export PARAFORMER_MODEL_Q4K="${PARAFORMER_MODEL_Q4K:-$CRISPASR_MODELS_DIR/paraformer-zh-q4_k.gguf}"
export PARAFORMER_AUDIO_ZH="${PARAFORMER_AUDIO_ZH:-samples/paraformer_zh.wav}"

# ── Diarization ──
export CRISPASR_TEST_DIARIZE_MODEL="${CRISPASR_TEST_DIARIZE_MODEL:-$CRISPASR_MODELS_DIR/pyannote-seg-3.0.gguf}"
export CRISPASR_TEST_TITANET_MODEL="${CRISPASR_TEST_TITANET_MODEL:-$CRISPASR_MODELS_DIR/titanet-large.gguf}"
export CRISPASR_TEST_DIARIZE_WAV="${CRISPASR_TEST_DIARIZE_WAV:-samples/multispeaker.wav}"

# ── Chat (LLM) — requires a llama.cpp-compatible chat model with a chat
# template (e.g. smollm2-360m-instruct, qwen2.5-0.5b-instruct). Harrier
# is an embedding model and won't work.
_chat_default="$CRISPASR_MODELS_DIR/smollm2-360m-instruct-q4_k.gguf"
if [ -n "${CRISPASR_CHAT_TEST_MODEL:-}" ]; then
    export CRISPASR_CHAT_TEST_MODEL
elif [ -f "$_chat_default" ]; then
    export CRISPASR_CHAT_TEST_MODEL="$_chat_default"
fi
unset _chat_default

# MOSS-Audio (OpenMOSS-Team/MOSS-Audio-4B-Instruct): audio understanding + ASR
export CRISPASR_MODEL_MOSS_AUDIO="${CRISPASR_MODEL_MOSS_AUDIO:-$CRISPASR_MODELS_DIR/moss-audio-4b-instruct-q4_k.gguf}"

# MOSS-Transcribe (OpenMOSS-Team/MOSS-Transcribe-preview-2B): ASR
export CRISPASR_MODEL_MOSS_TRANSCRIBE="${CRISPASR_MODEL_MOSS_TRANSCRIBE:-$CRISPASR_MODELS_DIR/moss-transcribe-preview-2b-q4_k.gguf}"

# ARK-ASR-3B (AutoArk-AI/ARK-ASR-3B): Whisper-large-v3 enc (partial RoPE) + Qwen2.5-3B LM.
# ⚠️ experimental/WIP — CPU only. See PLAN.md §ARK.
export CRISPASR_MODEL_ARK_ASR="${CRISPASR_MODEL_ARK_ASR:-$CRISPASR_MODELS_DIR/ark-asr-3b-q8_0.gguf}"

# Mini-Omni2 (gpt-omni/mini-omni2): Whisper-small + Qwen2-0.5B
export CRISPASR_MODEL_MINI_OMNI2="${CRISPASR_MODEL_MINI_OMNI2:-$CRISPASR_MODELS_DIR/mini-omni2-q4_k.gguf}"
export CRISPASR_MODEL_SNAC="${CRISPASR_MODEL_SNAC:-$CRISPASR_MODELS_DIR/snac-24khz.gguf}"

# ── Nemotron (streaming ASR) ──
export CRISPASR_MODEL_NEMOTRON="${CRISPASR_MODEL_NEMOTRON:-$CRISPASR_MODELS_DIR/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf}"
export CRISPASR_MODEL_NEMOTRON_F16="${CRISPASR_MODEL_NEMOTRON_F16:-$CRISPASR_MODELS_DIR/nemotron-3.5-asr-streaming-0.6b-f16.gguf}"

# ── LFM2-Audio ──
export CRISPASR_MODEL_LFM2="${CRISPASR_MODEL_LFM2:-$CRISPASR_MODELS_DIR/lfm2-audio-1.5b-q5_k.gguf}"

# ── Dia TTS ──
export CRISPASR_MODEL_DIA="${CRISPASR_MODEL_DIA:-$CRISPASR_MODELS_DIR/dia-1.6b-q4_k.gguf}"

# ── OuteTTS + WavTokenizer ──
export CRISPASR_MODEL_OUTETTS="${CRISPASR_MODEL_OUTETTS:-$CRISPASR_MODELS_DIR/outetts-0.3-1b-q4k-final.gguf}"
export CRISPASR_MODEL_WAVTOK="${CRISPASR_MODEL_WAVTOK:-$CRISPASR_MODELS_DIR/wavtokenizer-decoder-f16.gguf}"

echo "Live test env configured (CRISPASR_MODELS_DIR=$CRISPASR_MODELS_DIR)"
