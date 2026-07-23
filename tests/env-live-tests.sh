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

# Tiron (#295): Whisper large-v3 + inline <|speakerN|> markers (legacy ggml bin).
export CRISPASR_MODEL_TIRON="${CRISPASR_MODEL_TIRON:-$CRISPASR_MODELS_DIR/tiron-q4_k.bin}"

# ── Beam search backends ──
export CRISPASR_MODEL_GLM_ASR="${CRISPASR_MODEL_GLM_ASR:-$CRISPASR_MODELS_DIR/glm-asr-nano.gguf}"
export CRISPASR_MODEL_QWEN3_ASR="${CRISPASR_MODEL_QWEN3_ASR:-$CRISPASR_MODELS_DIR/qwen3-asr-0.6b.gguf}"
export CRISPASR_MODEL_HIGGS_STT="${CRISPASR_MODEL_HIGGS_STT:-$CRISPASR_MODELS_DIR/higgs-stt-q8_0.gguf}"
export CRISPASR_MODEL_VOXTRAL_TTS="${CRISPASR_MODEL_VOXTRAL_TTS:-$CRISPASR_MODELS_DIR/voxtral-4b-tts-q4_k.gguf}"
export CRISPASR_MODEL_CANARY="${CRISPASR_MODEL_CANARY:-$CRISPASR_MODELS_DIR/canary-1b-v2.gguf}"
# canary-qwen SALM (nvidia/canary-qwen-2.5b). #247 short-window echo regression.
export CRISPASR_MODEL_CANARY_QWEN="${CRISPASR_MODEL_CANARY_QWEN:-$CRISPASR_MODELS_DIR/canary-qwen-2.5b-q8_0.gguf}"
export CRISPASR_MODEL_LFM2_EN="${CRISPASR_MODEL_LFM2_EN:-$CRISPASR_MODELS_DIR/lfm2-audio-1.5b-q5_k.gguf}"
export CRISPASR_MODEL_LFM2_JP="${CRISPASR_MODEL_LFM2_JP:-$CRISPASR_MODELS_DIR/lfm2-audio-1.5b-jp-q5_k.gguf}"
# dots.tts: F16 core (the CFG flow-match derails on full-q8) + vocoder companion.
export CRISPASR_MODEL_DOTS_TTS="${CRISPASR_MODEL_DOTS_TTS:-$CRISPASR_MODELS_DIR/dots-tts-soar-f16.gguf}"
export CRISPASR_MODEL_DOTS_TTS_VOCODER="${CRISPASR_MODEL_DOTS_TTS_VOCODER:-$CRISPASR_MODELS_DIR/dots-tts-soar-vocoder-f16.gguf}"
export CRISPASR_MODEL_COHERE="${CRISPASR_MODEL_COHERE:-$CRISPASR_MODELS_DIR/cohere-transcribe.gguf}"

# ── Parakeet JA long-form regression guard (issue #89) ──
# Fixture: hf download cstr/crispasr-regression-fixtures \
#     parakeet-tdt-0.6b-ja/reazon_baseball_14s/audio.wav --local-dir <dir>
export CRISPASR_MODEL_PARAKEET_JA="${CRISPASR_MODEL_PARAKEET_JA:-$CRISPASR_MODELS_DIR/parakeet-tdt-0.6b-ja.gguf}"
export CRISPASR_FIXTURE_PARAKEET_JA="${CRISPASR_FIXTURE_PARAKEET_JA:-$CRISPASR_MODELS_DIR/fixtures/reazon_baseball_14s.wav}"

# ── Paraformer ── (canonical CRISPASR_ names; old bare names honored if pre-set)
export CRISPASR_PARAFORMER_MODEL="${CRISPASR_PARAFORMER_MODEL:-${PARAFORMER_MODEL:-$CRISPASR_MODELS_DIR/paraformer-zh-f16.gguf}}"
export CRISPASR_PARAFORMER_MODEL_Q4K="${CRISPASR_PARAFORMER_MODEL_Q4K:-${PARAFORMER_MODEL_Q4K:-$CRISPASR_MODELS_DIR/paraformer-zh-q4_k.gguf}}"
export CRISPASR_PARAFORMER_AUDIO_ZH="${CRISPASR_PARAFORMER_AUDIO_ZH:-${PARAFORMER_AUDIO_ZH:-samples/paraformer_zh.wav}}"

# ── Aligner (issue #217) ──
export CRISPASR_MODEL_ALIGNER="${CRISPASR_MODEL_ALIGNER:-$CRISPASR_MODELS_DIR/canary-ctc-aligner-q4_k.gguf}"

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

# MOSS-Transcribe-Diarize (OpenMOSS-Team/MOSS-Transcribe-Diarize-0.9B): ASR + diarization + timestamps
export CRISPASR_MODEL_MOSS_DIARIZE="${CRISPASR_MODEL_MOSS_DIARIZE:-$CRISPASR_MODELS_DIR/moss-transcribe-diarize-0.9b-q4_k.gguf}"

# MOSS-TTS-v1.5 (OpenMOSS-Team/MOSS-TTS-v1.5): TTS — Qwen3-8B backbone + 32 RVQ
# codebooks + transformer codec companion (validated by ASR round-trip, #249).
# MioTTS-0.6B (Qwen3 + MioCodec, Apache-2.0)
export CRISPASR_MODEL_MIOTTS="${CRISPASR_MODEL_MIOTTS:-$CRISPASR_MODELS_DIR/miotts-0.6b-q8_0.gguf}"
export CRISPASR_MODEL_PIANO_TRANSCRIPTION="${CRISPASR_MODEL_PIANO_TRANSCRIPTION:-$CRISPASR_MODELS_DIR/piano-transcription-f16.gguf}"
export CRISPASR_MODEL_MOSS_TTS="${CRISPASR_MODEL_MOSS_TTS:-$CRISPASR_MODELS_DIR/moss-tts-v1.5-q4_k.gguf}"
export CRISPASR_MODEL_MOSS_TTS_CODEC="${CRISPASR_MODEL_MOSS_TTS_CODEC:-$CRISPASR_MODELS_DIR/moss-tts-v1.5-codec.gguf}"
export CRISPASR_MODEL_MOSS_TTS_LOCAL="${CRISPASR_MODEL_MOSS_TTS_LOCAL:-$CRISPASR_MODELS_DIR/moss-tts-local-v1.5-q4_k.gguf}"
export CRISPASR_MODEL_MOSS_TTS_LOCAL_CODEC="${CRISPASR_MODEL_MOSS_TTS_LOCAL_CODEC:-$CRISPASR_MODELS_DIR/moss-tts-local-v1.5-codec.gguf}"

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

# ── TADA TTS (talker + TADA codec companion) ──
export CRISPASR_MODEL_TADA="${CRISPASR_MODEL_TADA:-$CRISPASR_MODELS_DIR/tada-tts-1b-q4_k.gguf}"
export CRISPASR_MODEL_TADA_CODEC="${CRISPASR_MODEL_TADA_CODEC:-$CRISPASR_MODELS_DIR/tada-codec-f16.gguf}"

# ── KugelAudio (7B audio understanding) ──
export CRISPASR_MODEL_KUGELAUDIO="${CRISPASR_MODEL_KUGELAUDIO:-$CRISPASR_MODELS_DIR/kugelaudio-0-open-f16.gguf}"

# ── MeloTTS (VITS2) ──
export CRISPASR_MODEL_MELOTTS="${CRISPASR_MODEL_MELOTTS:-$CRISPASR_MODELS_DIR/melotts-en-v2-f16.gguf}"

# ── Dia TTS ──
export CRISPASR_MODEL_DIA="${CRISPASR_MODEL_DIA:-$CRISPASR_MODELS_DIR/dia-1.6b-q4_k.gguf}"

# ── OuteTTS + WavTokenizer ──
export CRISPASR_MODEL_OUTETTS="${CRISPASR_MODEL_OUTETTS:-$CRISPASR_MODELS_DIR/outetts-0.3-1b-q4k-final.gguf}"
export CRISPASR_MODEL_WAVTOK="${CRISPASR_MODEL_WAVTOK:-$CRISPASR_MODELS_DIR/wavtokenizer-decoder-f16.gguf}"

# ── Sidon speech restoration ──
export CRISPASR_MODEL_SIDON="${CRISPASR_MODEL_SIDON:-$CRISPASR_MODELS_DIR/sidon-v0.1-f16.gguf}"

# ── VoxCPM2 AudioVAE speech upscaler ──
export CRISPASR_MODEL_VOXCPM2_VAE="${CRISPASR_MODEL_VOXCPM2_VAE:-$CRISPASR_MODELS_DIR/voxcpm2-vae-f32.gguf}"
# Optional full-model path for the simultaneous TTS + upscaler lifecycle test.
export CRISPASR_MODEL_VOXCPM2_FULL="${CRISPASR_MODEL_VOXCPM2_FULL:-}"

# ── CREPE monophonic F0 / pitch (cstr/crepe-GGUF) ──
# `tiny` is the shipping default: `full` is 38x the compute for the same
# geometry (see docs/music-transcription/PLAN.md). CRISPASR_MODEL_CREPE_FULL is
# only read by manual runs of test-crepe-parity, not by ctest.
export CRISPASR_MODEL_CREPE="${CRISPASR_MODEL_CREPE:-$CRISPASR_MODELS_DIR/crepe-tiny-f16.gguf}"
export CRISPASR_MODEL_CREPE_FULL="${CRISPASR_MODEL_CREPE_FULL:-$CRISPASR_MODELS_DIR/crepe-full-f16.gguf}"

# ── BTC chord recognition (cstr/btc-chords-GGUF) ──
# NON-COMMERCIAL WEIGHTS. The BTC checkpoints are CC-BY-NC-SA (trained on
# Isophonics / Robbie Williams / UsPop2002 annotations) even though the
# upstream code and this library are MIT. Downloading them requires
# --accept-license cc-by-nc-sa-4.0 (or CRISPASR_ACCEPT_LICENSE).
# The 170-class model is the default: it collapses to maj/min with
# CRISPASR_BTC_MAJ_MIN=1, whereas the 25-class one can never be expanded.
# ── RVC voice conversion (§CB1) ──
# LICENCE VARIES PER CHECKPOINT. RVC's code is MIT but circulating voice models
# do not share one licence; the GGUF carries its own tag and the registry gate
# matches on it. The pretrained base (lj1995/VoiceConversionWebUI
# pretrained_v2/f0G40k.pth) is what the parity work used.
# No CLI verb: the input is ContentVec features, so the session C ABI is the
# only surface. See docs/music-transcription/SVC_RECORD_SHAPES.md.
export CRISPASR_MODEL_RVC="${CRISPASR_MODEL_RVC:-$CRISPASR_MODELS_DIR/rvc-40k-f32.gguf}"

export CRISPASR_MODEL_BTC_CHORDS="${CRISPASR_MODEL_BTC_CHORDS:-$CRISPASR_MODELS_DIR/btc-chords-large-f32.gguf}"
# TabCNN guitar tablature (--tab). CC BY 4.0 weights, cstr/tabcnn-GGUF.
export CRISPASR_MODEL_TABCNN="${CRISPASR_MODEL_TABCNN:-$CRISPASR_MODELS_DIR/tabcnn-f16.gguf}"

echo "Live test env configured (CRISPASR_MODELS_DIR=$CRISPASR_MODELS_DIR)"

# qwen3-tts live tests
export CRISPASR_MODEL_QWEN3_TTS="${CRISPASR_MODEL_QWEN3_TTS:-$CRISPASR_MODELS_DIR/qwen3-tts-0.6b-q4_k.gguf}"
