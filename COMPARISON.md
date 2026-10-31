# ASR Ecosystem Comparison

Comparison of CrispASR with other ggml-based ASR implementations.
Last reviewed: 2026-07-11.

## Audio Model Support

| Model | CrispASR | llama.cpp (MIT) | koboldcpp (AGPL3) |
|---|---|---|---|
| OpenAI Whisper (all sizes) | ✔ native | via mtmd | via crispasr |
| Parakeet TDT v3 | ✔ native | — | — |
| Canary (NeMo) | ✔ native | — | — |
| Cohere Transcribe | ✔ native | — | — |
| Granite Speech | ✔ native | — | — |
| Voxtral Mini 3B | ✔ native | ✔ via mtmd | ✔ |
| Voxtral 4B Realtime | ✔ native | — | — |
| Qwen3-ASR | ✔ native | ✔ via mtmd | ✔ |
| FastConformer-CTC | ✔ native | — | — |
| Wav2Vec2 | ✔ native | — | — |
| GLM-ASR-Nano | ✔ native | — | — |
| Kyutai STT (Mimi codec) | ✔ native | — | — |
| FireRedASR2-AED | ✔ native | — | — |
| Moonshine (tiny/base) | ✔ native | — | — |
| Ultravox | — | ✔ via mtmd | ✔ |
| Gemma 4 Audio Conformer | ✔ native (gemma4-e2b/e4b) | ✔ via mtmd | — |
| Qwen2.5/3 Omni | — | ✔ via mtmd | ✔ |
| LFM2-Audio | ✔ native (ASR + TTS) | ✔ via mtmd | — |

**CrispASR: 60 ASR backend families** (all native, single GGUF per model;
plus 69 TTS backends, translation, punctuation, LID — see the README
model table for the full list incl. canary-qwen, MOSS/Qwen3-Omni-based,
SenseVoice, Paraformer, OmniASR, mega-asr, higgs-stt, mimo-asr)
**llama.cpp: 7 audio models** (via libmtmd, mmproj-style split GGUF)
**koboldcpp: 4 audio models** (fork of llama.cpp, AGPL-3.0)

## Auxiliary Models

| Model | CrispASR | llama.cpp | koboldcpp |
|---|---|---|---|
| Silero VAD | ✔ | — | — |
| FireRedVAD (DFSMN) | ✔ | — | — |
| Pyannote Segmentation | ✔ | — | — |
| Silero LID | ✔ | — | — |
| CTC Forced Aligner | ✔ (canary-ctc, wav2vec2, firered, omniasr) | — | — |

## Architecture Approaches

### CrispASR
- **Monolithic GGUF**: one file per model, includes all weights + vocab + CMVN
- **Backend pattern**: each model has its own `src/<model>.{h,cpp}` + backend adapter
- **Shared core**: `core/gguf_loader.h`, `core/attention.h`, `core/ffn.h`, `core/mel.h`
- **License**: MIT

### llama.cpp (libmtmd)
- **Split GGUF**: separate encoder (mmproj) + LLM GGUF files
- **Unified preprocessor**: `mtmd_audio_preprocessor` with per-model params
- **Model-specific graph builders**: Whisper-style, Conformer (Gemma4A), Conv2d (Qwen3)
- **Chunked window attention**: for Gemma4A Conformer (streaming-capable)
- **License**: MIT — safe to reference/adopt patterns

### koboldcpp
- **Fork of llama.cpp**: 3961 commits ahead, 173 behind
- **Same audio models as llama.cpp** minus Gemma4A/LFM2
- **Browser-integrated**: VAD + Push-to-Talk in web UI
- **License**: AGPL-3.0 — **cannot copy code into MIT project**

## Optimizations to Adopt from llama.cpp

1. **`ggml_soft_max_ext` with baked scale** — saves one `ggml_scale` op per attention layer
2. ~~**Chunked window attention**~~ — DONE for qwen3-asr (`CRISP_AUDIO_WINDOWED_ATTN=1`, opt-in; default-flip evaluated and rejected, see PERFORMANCE.md)
3. ~~**Conv2d subsampling** via ggml ops~~ — DONE (crisp_audio qwen3 tower)
4. **Frame stacking** in projector (Voxtral does 4x stacking before LLM)

## Models We Could Add (from llama.cpp reference)

| Model | Effort | Value |
|---|---|---|
| Ultravox | Medium — Whisper encoder + Llama decoder | Speech understanding, not just ASR |
| Qwen2.5 Omni | Medium — Whisper encoder + Qwen decoder | Multimodal (audio+vision+text) |
