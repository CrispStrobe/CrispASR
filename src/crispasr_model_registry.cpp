// crispasr_model_registry.cpp — implementation. See the header.

#include "crispasr_model_registry.h"
#include "crispasr_cache.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <vector>

namespace {

struct Entry {
    const char* backend;
    const char* filename;
    const char* url;
    const char* approx_size;
    const char* companion_file; // optional extra file (e.g. tokenizer.bin, primary voice). NULL if none.
    const char* companion_url;
    const char* companion_size; // size of the companion file; NULL = falls back to approx_size
    const char* license;        // NULL = permissive (MIT/Apache/etc.), non-NULL = printed to stderr on download
};

// Extra companion files beyond the single inline `companion_file/url` slot.
// Used by TTS backends that need more than one auxiliary file — e.g. kokoro
// auto-download bundles the English-default voice (slot 0, inline) plus
// the German backbone + German-default voice (here). The resolver pulls
// these in addition to the inline companion. Adding extras for a backend:
// one row in `k_extras`, terminate the list with {nullptr, nullptr}.
struct ExtraCompanion {
    const char* file;
    const char* url;
};
struct ExtraList {
    const char* backend;
    const ExtraCompanion* items; // NULL-terminated
};

// Keep entries aligned with what the CLI-only registry used to ship.
// Adding a new backend: one row here + a PUBLIC-link in src/CMakeLists.txt.
// clang-format off
constexpr Entry k_registry[] = {
    {"whisper", "ggml-base.bin",
     "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin", "~147 MB", nullptr, nullptr},
    {"nemotron", "nemotron-3.5-asr-streaming-0.6b-q4_k.gguf",
     "https://huggingface.co/cstr/nemotron-3.5-asr-streaming-0.6b-GGUF/resolve/main/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf",
     "~458 MB", nullptr, nullptr},
    {"parakeet", "parakeet-tdt-0.6b-v3-q4_k.gguf",
     "https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF/resolve/main/parakeet-tdt-0.6b-v3-q4_k.gguf", "~467 MB", nullptr, nullptr},
    {"canary", "canary-1b-v2-q4_k.gguf",
     "https://huggingface.co/cstr/canary-1b-v2-GGUF/resolve/main/canary-1b-v2-q4_k.gguf", "~600 MB", nullptr, nullptr},
    // LiquidAI LFM2.5-Audio-1.5B: FastConformer + LFM2 hybrid
    // conv+attention backbone. ASR (+ TTS/speech-to-speech planned).
    // English base model — Q5_K recommended (Q4_K too aggressive for EN).
    {"lfm2-audio", "lfm2-audio-1.5b-q5_k.gguf",
     "https://huggingface.co/cstr/lfm2-audio-1.5b-GGUF/resolve/main/lfm2-audio-1.5b-q5_k.gguf",
     "~1.6 GB", nullptr, nullptr, nullptr,
     "LFM Open License v1.0 (commercial use OK under $10M revenue; see "
     "https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B)"},
    // Japanese variant — Q5_K minimum (Q4_K produces 0 tokens on hybrid backbone).
    {"lfm2-audio", "lfm2-audio-1.5b-jp-q5_k.gguf",
     "https://huggingface.co/cstr/lfm2-audio-1.5b-jp-GGUF/resolve/main/lfm2-audio-1.5b-jp-q5_k.gguf",
     "~1.5 GB", nullptr, nullptr, nullptr,
     "LFM Open License v1.0 (commercial use OK under $10M revenue; see "
     "https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP)"},
    // gpt-omni/mini-omni2: Whisper-small + Qwen2-0.5B multimodal (ASR+TTS+S2S).
    // Q4_K is safe — identical ASR transcript to F16 on JFK 11s.
    // SNAC 24kHz codec companion for TTS/S2S output.
    {"mini-omni2", "mini-omni2-q4_k.gguf",
     "https://huggingface.co/cstr/mini-omni2-GGUF/resolve/main/mini-omni2-q4_k.gguf",
     "~1.0 GB",
     "snac-24khz.gguf",
     "https://huggingface.co/cstr/snac-24khz-GGUF/resolve/main/snac-24khz.gguf",
     "~80 MB"},
    {"voxtral", "voxtral-mini-3b-2507-q4_k.gguf",
     "https://huggingface.co/cstr/voxtral-mini-3b-2507-GGUF/resolve/main/voxtral-mini-3b-2507-q4_k.gguf", "~2.5 GB", nullptr, nullptr},
    {"voxtral4b", "voxtral-mini-4b-realtime-q4_k.gguf",
     "https://huggingface.co/cstr/voxtral-mini-4b-realtime-GGUF/resolve/main/voxtral-mini-4b-realtime-q4_k.gguf",
     "~3.3 GB", nullptr, nullptr},
    {"granite", "granite-speech-4.0-1b-q4_k.gguf",
     "https://huggingface.co/cstr/granite-speech-4.0-1b-GGUF/resolve/main/granite-speech-4.0-1b-q4_k.gguf", "~2.94 GB", nullptr, nullptr},
    {"granite-4.1", "granite-speech-4.1-2b-q4_k.gguf",
     "https://huggingface.co/cstr/granite-speech-4.1-2b-GGUF/resolve/main/granite-speech-4.1-2b-q4_k.gguf", "~2.94 GB", nullptr, nullptr},
    {"granite-4.1-plus", "granite-speech-4.1-2b-plus-q4_k.gguf",
     "https://huggingface.co/cstr/granite-speech-4.1-2b-plus-GGUF/resolve/main/granite-speech-4.1-2b-plus-q4_k.gguf",
     "~2.96 GB", nullptr, nullptr},
    {"granite-4.1-nar", "granite-speech-4.1-2b-nar-q4_k.gguf",
     "https://huggingface.co/cstr/granite-speech-4.1-2b-nar-GGUF/resolve/main/granite-speech-4.1-2b-nar-q4_k.gguf",
     "~3.2 GB", nullptr, nullptr},
    {"qwen3", "qwen3-asr-0.6b-q4_k.gguf",
     "https://huggingface.co/cstr/qwen3-asr-0.6b-GGUF/resolve/main/qwen3-asr-0.6b-q4_k.gguf", "~500 MB", nullptr, nullptr},
    {"qwen3-1.7b", "qwen3-asr-1.7b-q4_k.gguf",
     "https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF/resolve/main/qwen3-asr-1.7b-q4_k.gguf",
     "~1.3 GB", nullptr, nullptr},
    // Mega-ASR: Qwen3-ASR-1.7B with the upstream robustness LoRA merged
    // offline. It uses the standard qwen3 backend at runtime; the upstream
    // router is not required for this always-on robust path.
    {"mega-asr", "mega-asr-1.7b-q4_k.gguf",
     "https://huggingface.co/cstr/mega-asr-GGUF/resolve/main/mega-asr-1.7b-q4_k.gguf",
     "~1.3 GB", nullptr, nullptr},
    // FunAudioLLM/Fun-ASR-Nano-2512: 70 SANM encoder blocks + 2-block
    // Transformer adaptor + Qwen3-0.6B LLM decoder. zh/yue/en/ja/ko.
    // Only F16 ships today (Q4_K + Q8_0 pending). Upstream code is
    // Apache-2.0; weights are FunASR Model License v1.1 (Alibaba) —
    // commercial use OK with attribution, so the license field below
    // is set so the cache layer prints it to stderr on first download.
    {"funasr", "funasr-nano-2512-q8_0.gguf",
     "https://huggingface.co/cstr/funasr-nano-GGUF/resolve/main/funasr-nano-2512-q8_0.gguf",
     "~1.06 GB", nullptr, nullptr, nullptr,
     // Default is Q8_0, not F16: the F16 weights hit the CUDA F16×F32 matmul
     // saturation bug (issue #38 CUDA counterpart) and degenerate into a
     // single-token "!-loop" on GPU. Q8_0 takes the MMQ/MMVQ path (per-block
     // scales, F32-range activations) so it's correct on CUDA too, and it's
     // ~half the download. Q8_0 is byte-identical to F16 on CPU/Metal.
     "FunASR Model License v1.1 (commercial use OK with attribution; see "
     "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/blob/main/LICENSE)"},
    // Multilingual sibling — same architecture, 31 languages including
    // Korean, Vietnamese, Indonesian, Thai, Malay, Filipino, Arabic,
    // Hindi, Bulgarian, German, French, Spanish, Italian, Portuguese,
    // Dutch, Polish, Czech, Romanian, Greek, Finnish, Swedish, Turkish,
    // Persian, Danish, Hungarian, Macedonian, Russian.
    {"fun-asr-mlt-nano", "funasr-mlt-nano-2512-f16.gguf",
     "https://huggingface.co/cstr/funasr-mlt-nano-GGUF/resolve/main/funasr-mlt-nano-2512-f16.gguf",
     "~1.98 GB", nullptr, nullptr, nullptr,
     "FunASR Model License v1.1 (commercial use OK with attribution; see "
     "https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512/blob/main/LICENSE)"},
    // FunAudioLLM/SenseVoiceSmall: encoder-only multi-task ASR — same
    // SANM encoder body as Fun-ASR-Nano but with a CTC head emitting
    // language + emotion + audio-event tags alongside the transcript.
    // 15× faster than Whisper-Large per upstream. Q4_K default: 70
    // attn.fsmn + 2 attn.qkv tensors stay F16 (non-256-aligned ne[0]
    // — kernel=11 and 560), all other weight matrices quantize cleanly;
    // ASR-validated byte-identical to F16 on English (JFK) + Japanese
    // (JSUT) clips on M1 Metal.
    {"sensevoice", "sensevoice-small-q4_k.gguf",
     "https://huggingface.co/cstr/sensevoice-small-GGUF/resolve/main/sensevoice-small-q4_k.gguf",
     "~129 MB", nullptr, nullptr, nullptr,
     "FunASR Model License v1.1 (commercial use OK with attribution; see "
     "https://huggingface.co/FunAudioLLM/SenseVoiceSmall/blob/main/LICENSE)"},
    // F16 + Q8_0 lookups by canonical filename (auto-download resolver
    // still routes here when the user passes the explicit filename or
    // -m auto:variant).
    {"sensevoice", "sensevoice-small-f16.gguf",
     "https://huggingface.co/cstr/sensevoice-small-GGUF/resolve/main/sensevoice-small-f16.gguf",
     "~448 MB", nullptr, nullptr, nullptr,
     "FunASR Model License v1.1 (commercial use OK with attribution; see "
     "https://huggingface.co/FunAudioLLM/SenseVoiceSmall/blob/main/LICENSE)"},
    {"sensevoice", "sensevoice-small-q8_0.gguf",
     "https://huggingface.co/cstr/sensevoice-small-GGUF/resolve/main/sensevoice-small-q8_0.gguf",
     "~240 MB", nullptr, nullptr, nullptr,
     "FunASR Model License v1.1 (commercial use OK with attribution; see "
     "https://huggingface.co/FunAudioLLM/SenseVoiceSmall/blob/main/LICENSE)"},
    // Paraformer-zh: NAR-ASR, 220M params, zh+en character-level.
    // Q4_K default — byte-identical transcript to F16, 3.4× smaller.
    {"paraformer", "paraformer-zh-q4_k.gguf",
     "https://huggingface.co/cstr/paraformer-zh-GGUF/resolve/main/paraformer-zh-q4_k.gguf",
     "~123 MB", nullptr, nullptr, nullptr,
     "FunASR Model License (commercial use OK with attribution; see "
     "https://huggingface.co/funasr/paraformer-zh)"},
    {"paraformer", "paraformer-zh-f16.gguf",
     "https://huggingface.co/cstr/paraformer-zh-GGUF/resolve/main/paraformer-zh-f16.gguf",
     "~421 MB", nullptr, nullptr, nullptr,
     "FunASR Model License (commercial use OK with attribution; see "
     "https://huggingface.co/funasr/paraformer-zh)"},
    {"paraformer", "paraformer-zh-q8_0.gguf",
     "https://huggingface.co/cstr/paraformer-zh-GGUF/resolve/main/paraformer-zh-q8_0.gguf",
     "~227 MB", nullptr, nullptr, nullptr,
     "FunASR Model License (commercial use OK with attribution; see "
     "https://huggingface.co/funasr/paraformer-zh)"},
    {"cohere", "cohere-transcribe-q4_k.gguf",
     "https://huggingface.co/cstr/cohere-transcribe-03-2026-GGUF/resolve/main/cohere-transcribe-q4_k.gguf", "~550 MB", nullptr, nullptr},
    // cohere-asr-ja — Japanese fine-tune of CohereLabs/cohere-transcribe-03-2026
    // (efwkjn/cohere-asr-ja → CKHO GGUF conversion). Apache-2.0.
    // Replaces v0.1 as the recommended JA variant per issue #127.
    {"cohere", "cohere-asr-ja-q4_k.gguf",
     "https://huggingface.co/CKHO/cohere-asr-ja-GGUF/resolve/main/cohere-asr-ja-q4_k.gguf",
     "~1.5 GB", nullptr, nullptr},
    {"cohere", "cohere-asr-ja-q8_0.gguf",
     "https://huggingface.co/CKHO/cohere-asr-ja-GGUF/resolve/main/cohere-asr-ja-q8_0.gguf",
     "~2.4 GB", nullptr, nullptr},
    // cohere-asr-ja-v0.1 — older JA fine-tune (TransWithAI, issue #123).
    // Kept for backwards compat; users with cached v0.1 GGUFs still work.
    {"cohere", "cohere-asr-ja-v0.1-q4_k.gguf",
     "https://huggingface.co/TransWithAI/cohere-transcribe-ja-v0.1-GGUF/resolve/main/cohere-asr-ja-v0.1-q4_k.gguf",
     "~1.5 GB", nullptr, nullptr},
    {"cohere", "cohere-asr-ja-v0.1-q8_0.gguf",
     "https://huggingface.co/TransWithAI/cohere-transcribe-ja-v0.1-GGUF/resolve/main/cohere-asr-ja-v0.1-q8_0.gguf",
     "~2.4 GB", nullptr, nullptr},
    {"wav2vec2", "wav2vec2-xlsr-en-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-large-xlsr-53-english-GGUF/resolve/main/wav2vec2-xlsr-en-q4_k.gguf",
     "~212 MB", nullptr, nullptr},
    // Generic wav2vec2 CTC forced-aligner aliases. These use the same
    // GGUFs as the wav2vec2 ASR backend, but resolve cleanly from
    // `-am wav2vec2-aligner[-en|-de]`.
    {"wav2vec2-aligner", "wav2vec2-xlsr-en-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-large-xlsr-53-english-GGUF/resolve/main/wav2vec2-xlsr-en-q4_k.gguf",
     "~212 MB", nullptr, nullptr},
    {"wav2vec2-aligner-en", "wav2vec2-xlsr-en-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-large-xlsr-53-english-GGUF/resolve/main/wav2vec2-xlsr-en-q4_k.gguf",
     "~212 MB", nullptr, nullptr},
    {"wav2vec2-aligner-fr", "wav2vec2-large-xlsr-53-french-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-large-xlsr-53-french-GGUF/resolve/main/wav2vec2-large-xlsr-53-french-q4_k.gguf",
     "~300 MB", nullptr, nullptr},
    {"wav2vec2-aligner-es", "wav2vec2-large-xlsr-53-spanish-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-large-xlsr-53-spanish-GGUF/resolve/main/wav2vec2-large-xlsr-53-spanish-q4_k.gguf",
     "~300 MB", nullptr, nullptr},
    {"wav2vec2-aligner-it", "wav2vec2-large-xlsr-53-italian-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-large-xlsr-53-italian-GGUF/resolve/main/wav2vec2-large-xlsr-53-italian-q4_k.gguf",
     "~300 MB", nullptr, nullptr},
    {"wav2vec2-aligner-ja", "wav2vec2-large-xlsr-53-japanese-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-large-xlsr-53-japanese-GGUF/resolve/main/wav2vec2-large-xlsr-53-japanese-q4_k.gguf",
     "~300 MB", nullptr, nullptr},
    {"wav2vec2-aligner-zh", "wav2vec2-large-xlsr-53-chinese-zh-cn-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-large-xlsr-53-chinese-zh-cn-GGUF/resolve/main/wav2vec2-large-xlsr-53-chinese-zh-cn-q4_k.gguf",
     "~300 MB", nullptr, nullptr},
    {"wav2vec2-aligner-nl", "wav2vec2-large-xlsr-53-dutch-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-large-xlsr-53-dutch-GGUF/resolve/main/wav2vec2-large-xlsr-53-dutch-q4_k.gguf",
     "~300 MB", nullptr, nullptr},
    {"wav2vec2-aligner-uk", "wav2vec2-xls-r-300m-uk-with-small-lm-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-xls-r-300m-uk-with-small-lm-GGUF/resolve/main/wav2vec2-xls-r-300m-uk-with-small-lm-q4_k.gguf",
     "~300 MB", nullptr, nullptr},
    {"wav2vec2-aligner-pt", "wav2vec2-large-xlsr-53-portuguese-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-large-xlsr-53-portuguese-GGUF/resolve/main/wav2vec2-large-xlsr-53-portuguese-q4_k.gguf",
     "~300 MB", nullptr, nullptr},
    {"wav2vec2-aligner-ar", "wav2vec2-large-xlsr-53-arabic-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-large-xlsr-53-arabic-GGUF/resolve/main/wav2vec2-large-xlsr-53-arabic-q4_k.gguf",
     "~300 MB", nullptr, nullptr},
    {"wav2vec2-aligner-cs", "wav2vec2-xls-r-300m-cs-250-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-xls-r-300m-cs-250-GGUF/resolve/main/wav2vec2-xls-r-300m-cs-250-q4_k.gguf",
     "~300 MB", nullptr, nullptr},
    {"mimo-asr", "mimo-asr-q4_k.gguf",
     "https://huggingface.co/cstr/mimo-asr-GGUF/resolve/main/mimo-asr-q4_k.gguf", "~4.2 GB",
     "mimo-tokenizer-q4_k.gguf",
     "https://huggingface.co/cstr/mimo-tokenizer-GGUF/resolve/main/mimo-tokenizer-q4_k.gguf",
     "~395 MB"},
    {"moss-audio", "moss-audio-4b-instruct-q4_k.gguf",
     "https://huggingface.co/cstr/MOSS-Audio-4B-Instruct-GGUF/resolve/main/moss-audio-4b-instruct-q4_k.gguf", "~3.8 GB",
     nullptr, nullptr},
    {"omniasr", "omniasr-ctc-1b-v2-q4_k.gguf",
     "https://huggingface.co/cstr/omniASR-CTC-1B-v2-GGUF/resolve/main/omniasr-ctc-1b-v2-q4_k.gguf", "~658 MB", nullptr, nullptr},
    {"omniasr-300m", "omniasr-ctc-300m-v2-q4_k.gguf",
     "https://huggingface.co/cstr/omniASR-CTC-300M-v2-GGUF/resolve/main/omniasr-ctc-300m-v2-q4_k.gguf", "~194 MB", nullptr, nullptr},
    {"omniasr-llm", "omniasr-llm-300m-v2-q4_k.gguf",
     "https://huggingface.co/cstr/omniasr-llm-300m-v2-GGUF/resolve/main/omniasr-llm-300m-v2-q4_k.gguf", "~1019 MB", nullptr, nullptr},
    {"omniasr-llm-1b", "omniasr-llm-1b-q4_k.gguf",
     "https://huggingface.co/cstr/omniasr-llm-1b-GGUF/resolve/main/omniasr-llm-1b-q4_k.gguf", "~1300 MB", nullptr, nullptr},
    {"hubert", "hubert-large-ls960-ft-q4_k.gguf",
     "https://huggingface.co/cstr/hubert-large-ls960-ft-GGUF/resolve/main/hubert-large-ls960-ft-q4_k.gguf", "~200 MB", nullptr, nullptr},
    {"data2vec", "data2vec-audio-base-960h-q4_k.gguf",
     "https://huggingface.co/cstr/data2vec-audio-960h-GGUF/resolve/main/data2vec-audio-base-960h-q4_k.gguf", "~60 MB", nullptr, nullptr},
    {"vibevoice", "vibevoice-asr-q4_k.gguf",
     "https://huggingface.co/cstr/vibevoice-asr-GGUF/resolve/main/vibevoice-asr-q4_k.gguf", "~4.5 GB", nullptr, nullptr},
    {"vibevoice-1.5b", "vibevoice-1.5b-tts-q4_k.gguf",
     "https://huggingface.co/cstr/vibevoice-1.5b-GGUF/resolve/main/vibevoice-1.5b-tts-q4_k.gguf", "~1.6 GB", nullptr,
     nullptr},
    {"vibevoice-tts", "vibevoice-realtime-0.5b-q4_k.gguf",
     "https://huggingface.co/cstr/vibevoice-realtime-0.5b-GGUF/resolve/main/vibevoice-realtime-0.5b-q4_k.gguf",
     "~636 MB",
     "vibevoice-voice-emma.gguf",
     "https://huggingface.co/cstr/vibevoice-realtime-0.5b-GGUF/resolve/main/vibevoice-voice-emma.gguf",
     "~3 MB"},
    {"kugelaudio", "kugelaudio-0-open-f16.gguf",
     "https://huggingface.co/cstr/kugelaudio-0-open-GGUF/resolve/main/kugelaudio-0-open-f16.gguf", "~14 GB", nullptr,
     nullptr},
    {"firered-asr", "firered-asr2-aed-q4_k.gguf",
     "https://huggingface.co/cstr/firered-asr2-aed-GGUF/resolve/main/firered-asr2-aed-q4_k.gguf", "~918 MB", nullptr, nullptr},
    {"kyutai-stt", "kyutai-stt-1b-q4_k.gguf",
     "https://huggingface.co/cstr/kyutai-stt-1b-GGUF/resolve/main/kyutai-stt-1b-q4_k.gguf", "~636 MB", nullptr, nullptr},
    {"glm-asr", "glm-asr-nano-q4_k.gguf",
     "https://huggingface.co/cstr/glm-asr-nano-GGUF/resolve/main/glm-asr-nano-q4_k.gguf", "~1.2 GB", nullptr, nullptr},
    {"moonshine", "moonshine-tiny-q4_k.gguf",
     "https://huggingface.co/cstr/moonshine-tiny-GGUF/resolve/main/moonshine-tiny-q4_k.gguf", "~20 MB",
     "tokenizer.bin", "https://huggingface.co/cstr/moonshine-tiny-GGUF/resolve/main/tokenizer.bin",
     "~2 MB"},
    // moonshine-de: fidoriel/moonshine-base-de fine-tune (61.5M, 6.9% WER
    // on CV22-de). Best quality German moonshine. CC-BY-NC-SA-4.0.
    {"moonshine-de", "moonshine-base-de-fidoriel-q4_k.gguf",
     "https://huggingface.co/cstr/moonshine-base-de-fidoriel-GGUF/resolve/main/moonshine-base-de-fidoriel-q4_k.gguf", "~39 MB",
     "tokenizer.bin", "https://huggingface.co/cstr/moonshine-base-de-fidoriel-GGUF/resolve/main/tokenizer.bin",
     "~2 MB", "CC-BY-NC-SA-4.0"},
    // moonshine-tiny-de: fidoriel/moonshine-tiny-de fine-tune (27M, 11.4%
    // WER on CV22-de). Smaller/faster alternative. CC-BY-NC-SA-4.0.
    {"moonshine-tiny-de", "moonshine-tiny-de-fidoriel-q4_k.gguf",
     "https://huggingface.co/cstr/moonshine-tiny-de-fidoriel-GGUF/resolve/main/moonshine-tiny-de-fidoriel-q4_k.gguf", "~17 MB",
     "tokenizer.bin", "https://huggingface.co/cstr/moonshine-tiny-de-fidoriel-GGUF/resolve/main/tokenizer.bin",
     "~2 MB", "CC-BY-NC-SA-4.0"},
    {"wav2vec2-de", "wav2vec2-large-xlsr-53-german-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-large-xlsr-53-german-GGUF/resolve/main/wav2vec2-large-xlsr-53-german-q4_k.gguf",
     "~222 MB", nullptr, nullptr},
    {"wav2vec2-aligner-de", "wav2vec2-large-xlsr-53-german-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-large-xlsr-53-german-GGUF/resolve/main/wav2vec2-large-xlsr-53-german-q4_k.gguf",
     "~222 MB", nullptr, nullptr},
    {"moonshine-streaming", "moonshine-streaming-tiny-q4_k.gguf",
     "https://huggingface.co/cstr/moonshine-streaming-tiny-GGUF/resolve/main/moonshine-streaming-tiny-q4_k.gguf", "~31 MB",
     "tokenizer.bin", "https://huggingface.co/cstr/moonshine-streaming-tiny-GGUF/resolve/main/tokenizer.bin",
     "~2 MB"},
    {"fastconformer-ctc", "stt-en-fastconformer-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-en-fastconformer-ctc-large-GGUF/resolve/main/stt-en-fastconformer-ctc-large-q4_k.gguf",
     "~83 MB", nullptr, nullptr},
    // nvidia/parakeet-ctc-{0.6b,1.1b} — same FastConformer-CTC architecture
    // as the stt_en_fastconformer_ctc_* family (24 / 42 layers respectively),
    // English-only, lowercase + light-punct output. Filename heuristic
    // routes parakeet-ctc-* GGUFs to this backend.
    {"parakeet-ctc-0.6b", "parakeet-ctc-0.6b-q4_k.gguf",
     "https://huggingface.co/cstr/parakeet-ctc-0.6b-GGUF/resolve/main/parakeet-ctc-0.6b-q4_k.gguf",
     "~455 MB", nullptr, nullptr},
    {"parakeet-ctc-1.1b", "parakeet-ctc-1.1b-q4_k.gguf",
     "https://huggingface.co/cstr/parakeet-ctc-1.1b-GGUF/resolve/main/parakeet-ctc-1.1b-q4_k.gguf",
     "~795 MB", nullptr, nullptr},
    {"gemma4-e2b", "gemma4-e2b-it-q4_k.gguf",
     "https://huggingface.co/cstr/gemma4-e2b-it-GGUF/resolve/main/gemma4-e2b-it-q4_k.gguf",
     "~2.5 GB", nullptr, nullptr},
    // gemma4-e4b: same `gemma4` architecture as E2B (byte-identical 1024d USM
    // audio tower) with a larger decoder (42L×2560) — runs on the same backend.
    {"gemma4-e4b", "gemma4-e4b-it-q4_k.gguf",
     "https://huggingface.co/cstr/gemma4-e4b-it-GGUF/resolve/main/gemma4-e4b-it-q4_k.gguf",
     "~4.1 GB", nullptr, nullptr},
    {"titanet", "titanet-large.gguf",
     "https://huggingface.co/cstr/titanet-large-GGUF/resolve/main/titanet-large.gguf",
     "~45 MB", nullptr, nullptr},
    // parakeet-ja: F16 is the auto-download default — Q4_K of this
    // model is quantisation-sensitive (joint.pred / decoder.embed
    // dimensions fall back to q4_0 inside q4_k mode) and the talker
    // enters a fixed-point loop after ~8 tokens. The Q4_K file is
    // available at the same repo for users who pin disk space, but
    // we'd rather have correct output by default.
    {"parakeet-ja", "parakeet-tdt-0.6b-ja.gguf",
     "https://huggingface.co/cstr/parakeet-tdt-0.6b-ja-GGUF/resolve/main/parakeet-tdt-0.6b-ja.gguf",
     "~1.24 GB", nullptr, nullptr},
    // parakeet-v2 — English-only TDT (1024-vocab BPE, pred_layers=2).
    // The original Open ASR Leaderboard topper before v3 spread capacity
    // to 25 languages; often stronger on plain English. Same FastConformer
    // encoder + TDT decoder as v3, just trained on a different corpus.
    {"parakeet-v2", "parakeet-tdt-0.6b-v2-q4_k.gguf",
     "https://huggingface.co/cstr/parakeet-tdt-0.6b-v2-GGUF/resolve/main/parakeet-tdt-0.6b-v2-q4_k.gguf",
     "~468 MB", nullptr, nullptr},
    // parakeet-tdt-1.1b — larger TDT, English-only, 42-layer encoder
    // (vs 24 for 0.6b). Lowercase + no punctuation output. Slower but
    // wins on very long-tail vocabulary.
    {"parakeet-tdt-1.1b", "parakeet-tdt-1.1b-q4_k.gguf",
     "https://huggingface.co/cstr/parakeet-tdt-1.1b-GGUF/resolve/main/parakeet-tdt-1.1b-q4_k.gguf",
     "~808 MB", nullptr, nullptr},
    // parakeet-tdt_ctc-110m — smallest hybrid TDT+CTC. pred_layers=1
    // (single LSTM) means the TDT path can't be used; parakeet.cpp
    // auto-flips to CTC decode on load (see parakeet_init_from_file).
    // English-only, 17-layer encoder, d_model=512.
    {"parakeet-tdt_ctc-110m", "parakeet-tdt_ctc-110m-q4_k.gguf",
     "https://huggingface.co/cstr/parakeet-tdt_ctc-110m-GGUF/resolve/main/parakeet-tdt_ctc-110m-q4_k.gguf",
     "~91 MB", nullptr, nullptr},
    // parakeet-tdt_ctc-1.1b — larger hybrid TDT+CTC, 42-layer encoder,
    // multilingual vocab (proper casing + punctuation). Default decode
    // is TDT; pass `--parakeet-decoder ctc` for the CTC head.
    {"parakeet-tdt_ctc-1.1b", "parakeet-tdt_ctc-1.1b-q4_k.gguf",
     "https://huggingface.co/cstr/parakeet-tdt_ctc-1.1b-GGUF/resolve/main/parakeet-tdt_ctc-1.1b-q4_k.gguf",
     "~810 MB", nullptr, nullptr},
    // parakeet-rnnt-0.6b — standard RNN-Transducer (no TDT duration head).
    // Same 24-layer FastConformer encoder as TDT variants; 80-mel input;
    // 1024-token vocab (BPE, lowercase). Runtime auto-detects RNNT via
    // n_tdt_durations==0 and uses the RNNT greedy decoder.
    {"parakeet-rnnt-0.6b", "parakeet-rnnt-0.6b-q4_k.gguf",
     "https://huggingface.co/cstr/parakeet-rnnt-0.6b-GGUF/resolve/main/parakeet-rnnt-0.6b-q4_k.gguf",
     "~447 MB", nullptr, nullptr},
    // parakeet-rnnt-1.1b — larger standard RNN-Transducer, 42-layer encoder.
    {"parakeet-rnnt-1.1b", "parakeet-rnnt-1.1b-q4_k.gguf",
     "https://huggingface.co/cstr/parakeet-rnnt-1.1b-GGUF/resolve/main/parakeet-rnnt-1.1b-q4_k.gguf",
     "~770 MB", nullptr, nullptr},
    // parakeet-ctc-1.1b-ja — Japanese FastConformer-CTC (42L, 1.1B params).
    // Fine-tuned from nvidia/parakeet-ctc-1.1b on Japanese data. Uses the
    // GAL checkpoint (parakeet-ja-gal.nemo) — the non-GAL checkpoint has
    // corrupt F32 weights in layers 26-28 (NaN + values >1e38).
    {"parakeet-ctc-1.1b-ja", "parakeet-ctc-1.1b-ja-q8_0.gguf",
     "https://huggingface.co/cstr/parakeet-ctc-1.1b-ja-GGUF/resolve/main/parakeet-ctc-1.1b-ja-q8_0.gguf",
     "~1.2 GB", nullptr, nullptr},
    // reazonspeech-nemo-v2 — Japanese FastConformer-RNNT (619M params).
    // 80-mel input, rel_pos_local_attn (window=128+128, 1 global token),
    // 3000-token SentencePiece vocab. Pure RNNT (n_tdt_durations=0).
    // Trained on the ReazonSpeech v2.0 corpus (multi-hour capability).
    {"reazonspeech", "reazonspeech-nemo-v2-q8_0.gguf",
     "https://huggingface.co/cstr/reazonspeech-nemo-v2-GGUF/resolve/main/reazonspeech-nemo-v2-q8_0.gguf",
     "~704 MB", nullptr, nullptr},
    // Qwen3-TTS: the talker LM and the codec live in two separate HF
    // repos. Default download is Q8_0 talker (the LEARNINGS-recommended
    // deployment quant — Q4_K drifts noticeably in strict diffs) paired
    // with F16 codec (quantising the codec hurts earlier than the talker
    // — the runtime_ref_codes path is sensitive). Q4_K talker is on the
    // same repo for users who pin disk space; pass `-m <path>` to use it.
    {"qwen3-tts", "qwen3-tts-12hz-0.6b-base-q8_0.gguf",
     "https://huggingface.co/cstr/qwen3-tts-0.6b-base-GGUF/resolve/main/qwen3-tts-12hz-0.6b-base-q8_0.gguf",
     "~986 MB",
     "qwen3-tts-tokenizer-12hz.gguf",
     "https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF/resolve/main/qwen3-tts-tokenizer-12hz.gguf",
     "~60 MB"},
    // gwen-tts-0.6B: Vietnamese-optimized Qwen3-TTS-0.6B-Base finetune
    // (MIT, g-group-ai-lab). Same architecture as qwen3-tts-0.6B-Base,
    // trained on ~1000h Vietnamese TikTok audio. Supports all 10 Qwen3-TTS
    // languages but excels at Vietnamese. Uses the same 12 Hz tokenizer.
    {"gwen-tts", "gwen-tts-0.6b-q8_0.gguf",
     "https://huggingface.co/cstr/gwen-tts-0.6b-GGUF/resolve/main/gwen-tts-0.6b-q8_0.gguf",
     "~968 MB",
     "qwen3-tts-tokenizer-12hz.gguf",
     "https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF/resolve/main/qwen3-tts-tokenizer-12hz.gguf",
     "~60 MB"},
    // Qwen3-TTS-CustomVoice: fixed-speaker fine-tune of qwen3-tts-Base
    // with 9 baked speakers (aiden, dylan, eric, ono_anna, ryan, serena,
    // sohee, uncle_fu, vivian). Runtime path: pick a speaker via
    // `--voice <name>`; the speaker_embed is lifted from
    // talker.token_embd[spk_id] (no ECAPA forward, no reference WAV).
    // Two speakers carry Chinese-dialect overrides (dylan→Beijing,
    // eric→Sichuan) that re-route language_id when synthesising
    // Chinese-or-auto. Reuses the same 12 Hz tokenizer as Base.
    {"qwen3-tts-customvoice", "qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf",
     "https://huggingface.co/cstr/qwen3-tts-0.6b-customvoice-GGUF/resolve/main/qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf",
     "~968 MB",
     "qwen3-tts-tokenizer-12hz.gguf",
     "https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF/resolve/main/qwen3-tts-tokenizer-12hz.gguf",
     "~60 MB"},
    // Qwen3-TTS-Base 1.7B: same ICL voice-clone path as 0.6B-Base
    // (`--voice <wav> --ref-text "..."`), with talker hidden=2048,
    // ECAPA enc_dim=2048, and a small_to_mtp_projection bridge to the
    // 1024-d code predictor. ~1.9 GB Q8_0 talker; reuses the 12 Hz
    // tokenizer.
    {"qwen3-tts-1.7b-base", "qwen3-tts-12hz-1.7b-base-q8_0.gguf",
     "https://huggingface.co/cstr/qwen3-tts-1.7b-base-GGUF/resolve/main/qwen3-tts-12hz-1.7b-base-q8_0.gguf",
     "~1.9 GB",
     "qwen3-tts-tokenizer-12hz.gguf",
     "https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF/resolve/main/qwen3-tts-tokenizer-12hz.gguf",
     "~60 MB"},
    // Qwen3-TTS-CustomVoice 1.7B: same fixed-speaker pattern as 0.6B-CV
    // (9 baked speakers, `--voice <name>`, no ECAPA / no reference WAV)
    // but on the 1.7B talker. Runtime applies small_to_mtp_projection
    // to per-step code_pred embeddings (steps 1..14, fix in commit
    // `2cc7aeb`). Reuses the 12 Hz tokenizer. URL points at planned
    // `cstr/qwen3-tts-1.7b-customvoice-GGUF`; flagged "(publish pending)"
    // until the upload lands.
    {"qwen3-tts-1.7b-customvoice", "qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf",
     "https://huggingface.co/cstr/qwen3-tts-1.7b-customvoice-GGUF/resolve/main/qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf",
     "~2.0 GB",
     "qwen3-tts-tokenizer-12hz.gguf",
     "https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF/resolve/main/qwen3-tts-tokenizer-12hz.gguf",
     "~60 MB"},
    // Qwen3-TTS-VoiceDesign 1.7B: instruct-tuned variant that picks a
    // voice from a natural-language description ("--instruct \"young
    // female with British accent, energetic\"") — no reference WAV,
    // no preset speaker. The instruct text is prepended to the prefill
    // and the codec bridge omits the speaker frame entirely. 1.7B-only
    // (no 0.6B-VoiceDesign upstream). Reuses the 12 Hz tokenizer.
    {"qwen3-tts-1.7b-voicedesign", "qwen3-tts-12hz-1.7b-voicedesign-q8_0.gguf",
     "https://huggingface.co/cstr/qwen3-tts-1.7b-voicedesign-GGUF/resolve/main/qwen3-tts-12hz-1.7b-voicedesign-q8_0.gguf",
     "~1.9 GB",
     "qwen3-tts-tokenizer-12hz.gguf",
     "https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF/resolve/main/qwen3-tts-tokenizer-12hz.gguf",
     "~60 MB"},
    // Orpheus-3B (canopylabs/orpheus-3b-0.1-ft is gated; we convert
    // from the non-gated mirror unsloth/orpheus-3b-0.1-ft, llama3.2 —
    // "Built with Llama"). Talker = Llama-3.2-3B-Instruct + 7×4096
    // custom audio tokens. Codec = hubertsiuzdak/snac_24khz
    // (3 codebooks × 4096, MIT). PLAN #57 Phase 2 slices (a)/(b)/(c)
    // all DONE (commit a0982d3): registry foundation + talker AR
    // forward + SNAC C++ decode end-to-end; `--temperature 0.6` is the
    // ship-default (greedy loops in a 7-slot pattern). The companion
    // URL points at the cstr/snac-24khz-GGUF mirror that gets published
    // alongside the talker GGUF.
    {"orpheus", "orpheus-3b-0.1-ft-q8_0.gguf",
     "https://huggingface.co/cstr/orpheus-3b-0.1-ft-GGUF/resolve/main/orpheus-3b-0.1-ft-q8_0.gguf",
     "~3.7 GB",
     "snac-24khz.gguf",
     "https://huggingface.co/cstr/snac-24khz-GGUF/resolve/main/snac-24khz.gguf",
     "~80 MB"},
    // TADA TTS 1B: Llama-3.2-1B + flow matching + TADA codec.
    // Q4_K is the canonical CLI auto-download target; F16 is also listed so
    // explicit filename resolution can find the matching HF URL.
    {"tada-1b", "tada-tts-1b-q4_k.gguf",
     "https://huggingface.co/cstr/tada-tts-1b-GGUF/resolve/main/tada-tts-1b-q4_k.gguf",
     "~1.7 GB",
     "tada-codec-f16.gguf",
     "https://huggingface.co/cstr/tada-tts-1b-GGUF/resolve/main/tada-codec-f16.gguf",
     "~1 GB"},
    {"tada-tts-1b", "tada-tts-1b-q4_k.gguf",
     "https://huggingface.co/cstr/tada-tts-1b-GGUF/resolve/main/tada-tts-1b-q4_k.gguf",
     "~1.7 GB",
     "tada-codec-f16.gguf",
     "https://huggingface.co/cstr/tada-tts-1b-GGUF/resolve/main/tada-codec-f16.gguf",
     "~1 GB"},
    {"tada-1b", "tada-tts-1b-f16.gguf",
     "https://huggingface.co/cstr/tada-tts-1b-GGUF/resolve/main/tada-tts-1b-f16.gguf",
     "~3.1 GB",
     "tada-codec-f16.gguf",
     "https://huggingface.co/cstr/tada-tts-1b-GGUF/resolve/main/tada-codec-f16.gguf",
     "~1 GB"},
    // TADA-3B-ML (HumeAI/tada-3b-ml): Llama-3.2-3B + flow matching + TADA codec.
    {"tada", "tada-tts-3b-ml-f16.gguf",
     "https://huggingface.co/cstr/tada-tts-3b-ml-GGUF/resolve/main/tada-tts-3b-ml-f16.gguf",
     "~6.6 GB",
     "tada-codec-f16.gguf",
     "https://huggingface.co/cstr/tada-tts-3b-ml-GGUF/resolve/main/tada-codec-f16.gguf",
     "~1 GB"},
    // lex-au's German Orpheus-3B fine-tune. Already published as a Q8_0
    // GGUF on HF (`lex-au/Orpheus-3b-German-FT-Q8_0.gguf`, 3.52 GB) — the
    // repo name itself ends in `.gguf`, lex-au's convention. License
    // tagged Apache-2.0 on HF; underlying weights are llama3.2 community
    // (Llama-3.2-3B fine-tune), so attribution still applies in practice.
    // Same SNAC codec as the base orpheus row.
    {"lex-au-orpheus-de", "Orpheus-3b-German-FT-Q8_0.gguf",
     "https://huggingface.co/lex-au/Orpheus-3b-German-FT-Q8_0.gguf/resolve/main/Orpheus-3b-German-FT-Q8_0.gguf",
     "~3.5 GB",
     "snac-24khz.gguf",
     "https://huggingface.co/cstr/snac-24khz-GGUF/resolve/main/snac-24khz.gguf",
     "~80 MB"},
    // Kartoffel-Orpheus 3B German variants — drop-in checkpoint swaps on the
    // Orpheus runtime. The natural variant is fine-tuned on natural German
    // speech (~19 speakers); the synthetic variant adds emotion + outburst
    // control on 4 speakers (Martin/Luca/Anne/Emma). Both are gated on the
    // upstream HF repo (click-through accept). The cstr/ mirrors are
    // converted via models/convert-orpheus-to-gguf.py with --variant
    // fixed_speaker. Same SNAC codec as the base orpheus row.
    {"kartoffel-orpheus-de-natural", "kartoffel-orpheus-de-natural-q8_0.gguf",
     "https://huggingface.co/cstr/kartoffel-orpheus-3b-german-natural-GGUF/resolve/main/kartoffel-orpheus-de-natural-q8_0.gguf",
     "~3.5 GB",
     "snac-24khz.gguf",
     "https://huggingface.co/cstr/snac-24khz-GGUF/resolve/main/snac-24khz.gguf",
     "~80 MB"},
    {"kartoffel-orpheus-de-synthetic", "kartoffel-orpheus-de-synthetic-q8_0.gguf",
     "https://huggingface.co/cstr/kartoffel-orpheus-3b-german-synthetic-GGUF/resolve/main/kartoffel-orpheus-de-synthetic-q8_0.gguf",
     "~3.5 GB",
     "snac-24khz.gguf",
     "https://huggingface.co/cstr/snac-24khz-GGUF/resolve/main/snac-24khz.gguf",
     "~80 MB"},
    // Chatterbox family — ResembleAI MIT TTS. Base model is the 23-language
    // multilingual Chatterbox; language-specific fine-tunes live in
    // `kartoffelbox-turbo` (German turbo) and `lahgtna-chatterbox` (Arabic).
    // Two-GGUF runtime:
    //   primary  = T3 (text → speech tokens) — also carries baked conds
    //   companion = S3Gen (tokens → 24 kHz waveform via CFM + HiFTGenerator)
    // CLI adapter (`crispasr_backend_chatterbox.cpp`) is shipped — surfaces
    // as `--backend chatterbox|chatterbox-turbo|kartoffelbox-turbo|
    // lahgtna-chatterbox` with CAP_TTS + AUTO_DOWNLOAD + TEMPERATURE +
    // FLASH_ATTN + VOICE_CLONING. `--model-quant Q` substitutes the T3
    // filename + URL (Q4_K / Q5_K / Q8_0 / F16); `--tts-codec-quant Q`
    // substitutes the S3Gen companion; both repos publish F16 / Q8_0 /
    // Q4_K for both halves.
    {"chatterbox", "chatterbox-t3-q8_0.gguf",
     "https://huggingface.co/cstr/chatterbox-GGUF/resolve/main/chatterbox-t3-q8_0.gguf",
     "~880 MB",
     "chatterbox-s3gen-q8_0.gguf",
     "https://huggingface.co/cstr/chatterbox-GGUF/resolve/main/chatterbox-s3gen-q8_0.gguf",
     "~627 MB"},
    // Chatterbox-Turbo: distilled GPT-2 T3 (24L) + 2-step meanflow S3Gen.
    // Different architecture from base — runtime keys off
    // `chatterbox.t3.arch` ("kartoffelbox" branch handles GPT-2 T3 for
    // both Turbo and the German Kartoffelbox fine-tune). Default download
    // is Q8_0 for both T3 and S3Gen; F16/Q4_K share the same filename stem
    // for explicit quant selection.
    {"chatterbox-turbo", "chatterbox-turbo-t3-q8_0.gguf",
     "https://huggingface.co/cstr/chatterbox-turbo-GGUF/resolve/main/chatterbox-turbo-t3-q8_0.gguf",
     "~980 MB",
     "chatterbox-turbo-s3gen-q8_0.gguf",
     "https://huggingface.co/cstr/chatterbox-turbo-GGUF/resolve/main/chatterbox-turbo-s3gen-q8_0.gguf",
     "~627 MB"},
    // Kartoffelbox-Turbo: SebastianBodza's German fine-tune of
    // chatterbox-turbo. Same GPT-2 T3 arch as Turbo; reuses the
    // chatterbox-turbo S3Gen verbatim (companion points at the Turbo
    // repo on purpose — saves a redundant 627 MB upload). T3 shipped at
    // F16/Q8_0/Q4_K; Q8_0 is the recommended deployment quant.
    {"kartoffelbox-turbo", "kartoffelbox-turbo-t3-q8_0.gguf",
     "https://huggingface.co/cstr/kartoffelbox-turbo-GGUF/resolve/main/kartoffelbox-turbo-t3-q8_0.gguf",
     "~1.25 GB",
     "chatterbox-turbo-s3gen-f16.gguf",
     "https://huggingface.co/cstr/chatterbox-turbo-GGUF/resolve/main/chatterbox-turbo-s3gen-f16.gguf",
     "~627 MB"},
    // Lahgtna-chatterbox-v1: oddadmix's Arabic T3 fine-tune of base
    // ResembleAI/chatterbox. Shares the base Llama T3 architecture
    // (default converter path, no `--variant`); reuses the base S3Gen
    // verbatim (companion points at cstr/chatterbox-GGUF). T3 shipped
    // at F16 only.
    {"lahgtna-chatterbox", "chatterbox-t3-f16.gguf",
     "https://huggingface.co/cstr/lahgtna-chatterbox-v1-GGUF/resolve/main/chatterbox-t3-f16.gguf",
     "~1.4 GB",
     "chatterbox-s3gen-q8_0.gguf",
     "https://huggingface.co/cstr/chatterbox-GGUF/resolve/main/chatterbox-s3gen-q8_0.gguf",
     "~627 MB"},
    // Nari Labs Dia-1.6B (nari-labs/Dia-1.6B): byte-level text encoder (12L) +
    // AR audio decoder (18L, GQA 16q/4kv, classifier-free guidance) emitting 9
    // interleaved DAC codebooks under a delay pattern, decoded by a 44.1 kHz DAC
    // codec companion. Surfaces as `--backend dia` with CAP_TTS + AUTO_DOWNLOAD +
    // TEMPERATURE. Dialogue style with [S1]/[S2] speaker tags; use >100-char
    // prompts — Dia is inconsistent on very short inputs. Ships F16 only for now
    // (scale=1.0 attention is precision-sensitive; lower-bit quants need an
    // ASR-roundtrip check before release). The DAC companion is quant-agnostic.
    {"dia", "dia-1.6b-f16.gguf",
     "https://huggingface.co/cstr/dia-1.6b-GGUF/resolve/main/dia-1.6b-f16.gguf",
     "~3.0 GB",
     "dac-44khz.gguf",
     "https://huggingface.co/cstr/dia-1.6b-GGUF/resolve/main/dac-44khz.gguf",
     "~80 MB"},
    // dots.tts: rednote-hilab's 2B continuous AR TTS (48 kHz, Apache-2.0).
    // Qwen2.5-1.5B LLM + 18L DiT flow-matching + BigVGAN vocoder.
    // No discrete codec tokens — generates continuous latents patch-by-patch.
    // BPE text input (no phonemes). Vocoder is a separate GGUF companion.
    {"dots-tts", "dots-tts-soar-f16.gguf",
     "https://huggingface.co/cstr/dots-tts-soar-GGUF/resolve/main/dots-tts-soar-f16.gguf",
     "~4.4 GB",
     "dots-tts-soar-vocoder-f16.gguf",
     "https://huggingface.co/cstr/dots-tts-soar-GGUF/resolve/main/dots-tts-soar-vocoder-f16.gguf",
     "~345 MB"},
    // Pocket TTS: Kyutai's 100M continuous-latent AR TTS (24 kHz, MIT/CC-BY-4.0).
    // Generates continuous 32-dim float vectors at 12.5 Hz via one-step LSD,
    // decoded by Mimi VAE to 24 kHz PCM. Single GGUF, no codec companion.
    // Voice cloning via --voice ref.wav (requires the -f16 variant with Mimi encoder).
    // The -novc variant lacks the encoder and produces near-silence without conditioning.
    {"pocket-tts", "pocket-tts-english-f16.gguf",
     "https://huggingface.co/cstr/pocket-tts-GGUF/resolve/main/pocket-tts-english-f16.gguf",
     "~220 MB"},
    // IndexTTS-1.5: GPT-2 AR mel-code generator + BigVGAN vocoder.
    // Voice cloning via Conformer+Perceiver conditioning on reference audio.
    // Two-file setup: GPT (mel codes) + BigVGAN (vocoder). Q8_0 recommended
    // (~870 MB total); F16 available for max quality (~2.4 GB).
    {"indextts", "indextts-gpt-q8_0.gguf",
     "https://huggingface.co/cstr/indextts-1.5-GGUF/resolve/main/indextts-gpt-q8_0.gguf",
     "~870 MB",
     "indextts-bigvgan.gguf",
     "https://huggingface.co/cstr/indextts-1.5-GGUF/resolve/main/indextts-bigvgan.gguf",
     "~112 MB"},
    // OuteTTS 0.3 1B: OLMo-1B LLM + WavTokenizer single-codebook VQ-GAN.
    // 24 kHz output, CC BY 4.0 license. Voice cloning via --voice <speaker.json>.
    // Two-file setup: OLMo talker (Q8_0, ~1.3 GB) + WavTokenizer decoder (~130 MB).
    // Also available: F16 (~2.4 GB), Q5_K (~820 MB). Q8_0 recommended.
    {"outetts", "outetts-0.3-1b-q8_0.gguf",
     "https://huggingface.co/cstr/outetts-0.3-1b-GGUF/resolve/main/outetts-0.3-1b-q8_0.gguf",
     "~1270 MB",
     "wavtokenizer-decoder-f16.gguf",
     "https://huggingface.co/cstr/outetts-0.3-1b-GGUF/resolve/main/wavtokenizer-decoder-f16.gguf",
     "~130 MB"},
    // F5-TTS v1 Base: DiT-based flow-matching TTS with zero-shot voice
    // cloning. Single GGUF containing DiT (330M) + Vocos vocoder (13M).
    // Character-level tokenization (2545 vocab), 24 kHz output.
    // Voice cloning via --voice <ref.wav> --ref-text "transcript".
    {"f5-tts", "f5-tts-v1-base-f16.gguf",
     "https://huggingface.co/cstr/f5-tts-GGUF/resolve/main/f5-tts-v1-base-f16.gguf",
     "~953 MB", nullptr, nullptr},
    // CTC forced aligner — used by `-am auto` to attach word-level
    // timestamps (LLM-decode backends, or any backend when paired
    // with `--force-aligner` / `-fa`). Q4_K is the recommended quant
    // (~442 MB; quality is indistinguishable from Q8_0 on word-onset
    // accuracy). Q5_0 / Q8_0 / F16 sit on the same repo.
    {"canary-ctc-aligner", "canary-ctc-aligner-q4_k.gguf",
     "https://huggingface.co/cstr/canary-ctc-aligner-GGUF/resolve/main/canary-ctc-aligner-q4_k.gguf",
     "~442 MB", nullptr, nullptr},
    // Qwen3-ForcedAligner-0.6B — same architecture as qwen3-asr-0.6B but
    // with a 5000-class timestamp head instead of a vocabulary head.
    // Identified at load time by the lm_head output dimension; crispasr_aligner
    // dispatches to the qwen3 path when the filename contains
    // "forced-aligner", "qwen3-fa", or "qwen3-forced". Q4_K (~500 MB)
    // is the recommended quant; Q5_0 / Q8_0 / F16 also available on the repo.
    {"qwen3-forced-aligner", "qwen3-forced-aligner-0.6b-q4_k.gguf",
     "https://huggingface.co/cstr/qwen3-forced-aligner-0.6b-GGUF/resolve/main/qwen3-forced-aligner-0.6b-q4_k.gguf",
     "~500 MB", nullptr, nullptr},
    {"qwen3-fa", "qwen3-forced-aligner-0.6b-q4_k.gguf",
     "https://huggingface.co/cstr/qwen3-forced-aligner-0.6b-GGUF/resolve/main/qwen3-forced-aligner-0.6b-q4_k.gguf",
     "~500 MB", nullptr, nullptr},
    // M2M-100 (facebook/m2m100_418M, MIT) — multilingual text-to-text
    // translation. 100 source/target languages via SentencePiece + lang
    // codes prefix. Encoder-decoder transformer with cross-attention
    // KV cache; en→de exact match to Python reference. Q8_0 (~502 MB)
    // is the recommended deployment quant; Q4_K (~272 MB) and F16 are
    // available on the same repo. The runtime is exposed today through
    // crispasr_session_translate_text in the C ABI; the standalone CLI
    // text-translate path is still pending — see PLAN.
    {"m2m100", "m2m100-418m-q8_0.gguf",
     "https://huggingface.co/cstr/m2m100-418m-GGUF/resolve/main/m2m100-418m-q8_0.gguf",
     "~502 MB", nullptr, nullptr},
    // WMT21 dense-24-wide-en-x (facebook, MIT) — same m2m100
    // architecture as the 418M base, scaled up to 4.7B params and
    // narrower in coverage (English → 7 target languages, won the
    // WMT21 News competition). Runs through the m2m100 backend
    // (same runtime, just a different scale + slightly different
    // vocab — vocab fix landed in commit 7f48bad). Pass
    // `--backend m2m100-wmt21` (alias for m2m100) to pick this row.
    {"m2m100-wmt21", "wmt21-dense-24-wide-en-x-q4_k.gguf",
     "https://huggingface.co/cstr/wmt21-dense-24-wide-en-x-GGUF/resolve/main/wmt21-dense-24-wide-en-x-q4_k.gguf",
     "~2.5 GB", nullptr, nullptr},
    // MADLAD-400 3B (google, Apache-2.0) — multilingual T5
    // translation with 419 languages. T5 encoder-decoder (12L+12L,
    // d=2048, gated-GELU FFN, RMSNorm, bucketed relative-position
    // bias, SentencePiece 256K). Pass `--backend madlad` or
    // `--backend t5` to dispatch to the t5_translate runtime.
    // Output is bit-token-identical to Python SP after the rel-pos
    // FUTURE/PAST + Viterbi tokenizer + special-token-id fixes
    // (commits 8c5818d / 04bd6ec / faf9f0a).
    {"madlad", "madlad400-3b-mt-q4_k.gguf",
     "https://huggingface.co/cstr/madlad400-3b-mt-GGUF/resolve/main/madlad400-3b-mt-q4_k.gguf",
     "~1.9 GB", nullptr, nullptr},
    // Kokoro-82M: official baseline + English default voice. The German
    // backbone + German default voice ride along via k_extras (see below)
    // so users running `-m auto --backend kokoro` get a working multilingual
    // setup without separate `--companion` flags. Q8_0 is the recommended
    // quant (Q4_K is below quality bar — see `cstr/kokoro-82m-GGUF` README).
    {"kokoro", "kokoro-82m-q8_0.gguf",
     "https://huggingface.co/cstr/kokoro-82m-GGUF/resolve/main/kokoro-82m-q8_0.gguf",
     "~135 MB",
     "kokoro-voice-af_heart.gguf",
     "https://huggingface.co/cstr/kokoro-voices-GGUF/resolve/main/kokoro-voice-af_heart.gguf",
     "~4 MB"},

    // Piper — rhasspy/piper VITS TTS. 250+ community voices, 30+ languages.
    // Default voice: en_US-lessac-medium (~16 MB F16).
    {"piper", "piper-en_US-lessac-medium-f16.gguf",
     "https://huggingface.co/cstr/piper-en_US-lessac-medium-GGUF/resolve/main/piper-en_US-lessac-medium-f16.gguf",
     "~16 MB", nullptr, nullptr},
    // German voices from the consolidated piper-voices-GGUF repo
    {"piper", "piper-de_DE-thorsten-medium-f16.gguf",
     "https://huggingface.co/cstr/piper-voices-GGUF/resolve/main/piper-de_DE-thorsten-medium-f16.gguf",
     "~30 MB", nullptr, nullptr},
    {"piper", "piper-de_DE-thorsten-high-f16.gguf",
     "https://huggingface.co/cstr/piper-voices-GGUF/resolve/main/piper-de_DE-thorsten-high-f16.gguf",
     "~60 MB", nullptr, nullptr},
    {"piper", "piper-de_DE-kerstin-low-f16.gguf",
     "https://huggingface.co/cstr/piper-voices-GGUF/resolve/main/piper-de_DE-kerstin-low-f16.gguf",
     "~12 MB", nullptr, nullptr},

    // Bark — suno/bark 3-stage hierarchical TTS (MIT). bark-small ~300M params,
    // 24 kHz, 10 German speakers (v2/de_speaker_0..9). Single GGUF packs all
    // 3 sub-models (semantic + coarse + fine GPT-2) + EnCodec decoder.
    {"bark", "bark-small-q8_0.gguf",
     "https://huggingface.co/cstr/bark-small-GGUF/resolve/main/bark-small-q8_0.gguf",
     "~500 MB", nullptr, nullptr},

    // CSM-1B — sesame/csm-1b conversational TTS (Apache 2.0). Llama-3.2 1B
    // backbone + 100M depth decoder + Kyutai Mimi codec, all in one GGUF.
    // 24 kHz. Default Q4_K (1.4 GB); F16 + Q8_0 on the same repo.
    {"csm", "csm-1b-q4_k.gguf",
     "https://huggingface.co/cstr/csm-1b-GGUF/resolve/main/csm-1b-q4_k.gguf",
     "~1.4 GB", nullptr, nullptr},

    // MeloTTS: VITS2 52M param TTS (myshell-ai/MeloTTS). 44.1 kHz mono.
    // MIT license. Single GGUF with embedded CMU dictionary + neural G2P.
    // V2: 4 English speakers (US/BR/India/AU), 112 symbols, 11 tones.
    // V3: 1 speaker (EN-Newest), 219 symbols, 16 tones (newest checkpoint).
    // Companion: bert-base-uncased Q4_K (52 MB) for BERT conditioning.
    // Also available: F16 (227 MB), Q8_0 (97 MB) on the same HF repo.
    {"melotts", "melotts-en-v2-f16.gguf",
     "https://huggingface.co/cstr/melotts-en-v2-GGUF/resolve/main/melotts-en-v2-f16.gguf",
     "~102+52 MB",
     "bert-base-uncased-q4k.gguf",
     "https://huggingface.co/cstr/melotts-en-v2-GGUF/resolve/main/bert-base-uncased-q4k.gguf",
     "~52 MB"},
    {"melotts-v3", "melotts-en-v3-f16.gguf",
     "https://huggingface.co/cstr/melotts-en-v3-GGUF/resolve/main/melotts-en-v3-f16.gguf",
     "~93+52 MB",
     "bert-base-uncased.gguf",
     "https://huggingface.co/cstr/melotts-en-v3-GGUF/resolve/main/bert-base-uncased.gguf",
     "~420 MB"},

    // SpeechT5 TTS: 80M param AR mel decoder + HiFi-GAN vocoder.
    // MIT license (microsoft/speecht5_tts). Needs a 512-d x-vector for
    // speaker conditioning; pass via --voice <xvector.bin> or set_voice().
    {"speecht5", "speecht5-tts-f16.gguf",
     "https://huggingface.co/cstr/speecht5-tts-GGUF/resolve/main/speecht5-tts-f16.gguf",
     "~300 MB", nullptr, nullptr},

    // Parler TTS Mini v1.1: prompt-conditioned TTS (~900M). T5 encoder +
    // MusicGen decoder + DAC 44.1 kHz codec. Describe the voice in text
    // via --instruct. Apache-2.0.
    {"parler-tts", "parler-mini-v1.1-q8_0.gguf",
     "https://huggingface.co/cstr/parler-tts-mini-v1.1-GGUF/resolve/main/parler-mini-v1.1-q8_0.gguf",
     "~900 MB", nullptr, nullptr},

    // Text-LID — three families, one auto-routing dispatcher
    // (`src/text_lid_dispatch.cpp`). `lid-cld3` is the default for
    // `crispasr-lid -m auto` because it's the smallest (440 KB F16),
    // Apache-2.0, and matches CLD3's full 109-language ISO 639-1 contract.
    {"lid-cld3", "cld3-f16.gguf",
     "https://huggingface.co/cstr/cld3-GGUF/resolve/main/cld3-f16.gguf", "~440 KB", nullptr, nullptr},
    // GlotLID-V3 — fastText supervised, 2102 ISO 639-3 + script labels.
    // Apache-2.0, ~250 MB F16. Best coverage for low-resource languages.
    {"lid-glotlid", "glotlid-f16.gguf",
     "https://huggingface.co/cstr/glotlid-GGUF/resolve/main/glotlid-f16.gguf", "~250 MB", nullptr, nullptr},
    // Facebook LID-176 — fastText hierarchical-softmax, 176 ISO 639-1.
    // CC-BY-SA-3.0 (viral). ~63 MB F16. Pick only if you need its
    // specific 176-label space and accept the SA obligation.
    {"lid-fasttext176", "fasttext-lid176-f16.gguf",
     "https://huggingface.co/cstr/fasttext-lid176-GGUF/resolve/main/fasttext-lid176-f16.gguf",
     "~63 MB", nullptr, nullptr},

    // Audio-LID family — speech-signal language identification.
    // Silero + Ecapa run through the module-level `detect_language_pcm`;
    // FireRed requires the session-level `Session::detect_language` (Phase 6).
    // `lid-silero` is the recommended default: 95 languages, ~16 MB, Apache-2.0.
    // Converted from deepghs/silero-lang95-onnx via models/convert-silero-lid-to-gguf.py.
    {"lid-silero", "silero-lid-95-f16.gguf",
     "https://huggingface.co/cstr/silero-lid-95-GGUF/resolve/main/silero-lid-95-f16.gguf",
     "~16 MB", nullptr, nullptr},
    // ECAPA-TDNN LID: speechbrain/lang-id-voxlingua107-ecapa (Apache-2.0),
    // 107 languages, attentive statistical pooling. ~42 MB F16.
    // Converted via models/convert-ecapa-tdnn-lid-to-gguf.py.
    {"lid-ecapa", "ecapa-lid-107-f16.gguf",
     "https://huggingface.co/cstr/ecapa-lid-107-GGUF/resolve/main/ecapa-lid-107-f16.gguf",
     "~42 MB", nullptr, nullptr},
    // FireRed LID: FireRedTeam/FireRedLID encoder + 6-layer LID Transformer,
    // 120 languages. Converted via models/convert-firered-lid-to-gguf.py.
    {"lid-firered", "firered-lid-f16.gguf",
     "https://huggingface.co/cstr/firered-lid-GGUF/resolve/main/firered-lid-f16.gguf",
     "~300 MB", nullptr, nullptr},
    // VoxCPM2: openbmb/VoxCPM2 diffusion AR TTS, 30 languages, 48kHz,
    // Apache 2.0. Tokenizer-free, voice cloning via reference audio.
    // Default: Q4_K (1.6 GB, practical for CPU). F16 at same repo.
    {"voxcpm2-tts", "voxcpm2-q4_k.gguf",
     "https://huggingface.co/cstr/voxcpm2-GGUF/resolve/main/voxcpm2-q4_k.gguf",
     "~1.6 GB", nullptr, nullptr},
    // CosyVoice3 0.5B-2512: FunAudioLLM streaming multilingual TTS,
    // Apache 2.0, 9 languages + 18 Chinese dialects, 24 kHz, zero-shot
    // voice cloning via baked voices.gguf. Three-stage pipeline (LLM
    // AR → flow Euler → HiFT vocoder). Default companion bundle:
    // Q4_K LLM (384 MB) + Q8_0 flow (361 MB) + F16 HiFT (42 MB) +
    // voices (57 KB) = ~745 MB. F16 reference also on the same repo.
    {"cosyvoice3-tts", "cosyvoice3-llm-q4_k.gguf",
     "https://huggingface.co/cstr/cosyvoice3-0.5b-2512-GGUF/resolve/main/cosyvoice3-llm-q4_k.gguf",
     "~384 MB",
     "cosyvoice3-flow-q8_0.gguf",
     "https://huggingface.co/cstr/cosyvoice3-0.5b-2512-GGUF/resolve/main/cosyvoice3-flow-q8_0.gguf",
     "~361 MB"},
    // FastPitch: NVIDIA non-autoregressive parallel TTS (single speaker,
    // English, 22 kHz). ~60M params (FastPitch + HiFi-GAN in one GGUF).
    // Deterministic — no sampling, same input always produces same output.
    {"fastpitch", "fastpitch-en-q8_0.gguf",
     "https://huggingface.co/cstr/fastpitch-en-GGUF/resolve/main/fastpitch-en-q8_0.gguf",
     "~120 MB", nullptr, nullptr},

    // Truecaser — mayhewsw/pytorch-truecaser ports. BiLSTM is the
    // recommended variant (97.9% F1 on German). Standalone .bin format
    // (not GGUF). The `crispasr_truecase_*` C-ABI loads these directly.
    // Published at cstr/truecaser-de on HuggingFace.
    {"truecaser-lstm-de", "truecaser-lstm-de.bin",
     "https://huggingface.co/cstr/truecaser-de/resolve/main/truecaser-lstm-de.bin",
     "~3 MB", nullptr, nullptr},
    {"truecaser-lstm-en", "truecaser-lstm-en.bin",
     "https://huggingface.co/cstr/truecaser-de/resolve/main/truecaser-lstm-en.bin",
     "~3 MB", nullptr, nullptr},
    {"truecaser-lstm-es", "truecaser-lstm-es.bin",
     "https://huggingface.co/cstr/truecaser-de/resolve/main/truecaser-lstm-es.bin",
     "~3 MB", nullptr, nullptr},
    {"truecaser-lstm-ru", "truecaser-lstm-ru.bin",
     "https://huggingface.co/cstr/truecaser-de/resolve/main/truecaser-lstm-ru.bin",
     "~4 MB", nullptr, nullptr},
    {"truecaser-crf-de", "truecaser-crf-de.bin",
     "https://huggingface.co/cstr/truecaser-de/resolve/main/truecaser-crf-de.bin",
     "~8 MB", nullptr, nullptr},
    {"truecaser-de", "truecaser-de.bin",
     "https://huggingface.co/cstr/truecaser-de/resolve/main/truecaser-de.bin",
     "~2 MB", nullptr, nullptr},
    // Zonos v0.1 — Zyphra 500M-param transformer TTS with emotion/pitch/rate
    // control, speaker cloning, 44.1 kHz via DAC codec. Apache 2.0.
    // Selective Q4_K: heads.*/embeddings.*/prefix_conditioner.* kept at F16
    // to prevent EOS-logit inflation; backbone projections quantized.
    // Runtime includes a 3-retry guard for residual step-0 EOS failures.
    // Companion: DAC 44.1 kHz F16 decoder from its own GGUF repo (MIT).
    {"zonos", "zonos-v0.1-transformer-q8_0.gguf",
     "https://huggingface.co/cstr/zonos-v0.1-transformer-GGUF/resolve/main/zonos-v0.1-transformer-q8_0.gguf",
     "~1.6 GB",
     "dac-44khz-f16.gguf",
     "https://huggingface.co/cstr/dac-44khz-GGUF/resolve/main/dac-44khz-f16.gguf",
     "~104 MB"},
};

// Multi-companion extras. When a backend needs >1 auxiliary file the
// extras here ride along with the inline `companion_file`.
constexpr ExtraCompanion k_kokoro_extras[] = {
    // German backbone — auto-routing kicks in when this sits next to
    // kokoro-82m-*.gguf and the user passes `-l de`. See PLAN #56 opt 2b.
    {"kokoro-de-hui-base-q8_0.gguf",
     "https://huggingface.co/cstr/kokoro-de-hui-base-GGUF/resolve/main/kokoro-de-hui-base-q8_0.gguf"},
    // German default voice (in-distribution to the dida-80b backbone).
    {"kokoro-voice-df_victoria.gguf",
     "https://huggingface.co/cstr/kokoro-voices-GGUF/resolve/main/kokoro-voice-df_victoria.gguf"},
    {nullptr, nullptr},
};

// VibeVoice-Realtime ships 26 voicepacks (~3 MB each) on
// cstr/vibevoice-realtime-0.5b-GGUF. The inline companion is `emma`
// (English default, ~3 MB). Extras here pull a representative German
// voice + a French voice so `-m auto --backend vibevoice-tts -l de`
// (or `-l fr`) produces native-language output without an explicit
// `--voice`. Other languages (ja/it/kr/...) ship as separate downloads
// — keeping the auto-download set lean.
constexpr ExtraCompanion k_vibevoice_tts_extras[] = {
    {"vibevoice-voice-de-Spk1_woman.gguf",
     "https://huggingface.co/cstr/vibevoice-realtime-0.5b-GGUF/resolve/main/vibevoice-voice-de-Spk1_woman.gguf"},
    {"vibevoice-voice-fr-Spk1_woman.gguf",
     "https://huggingface.co/cstr/vibevoice-realtime-0.5b-GGUF/resolve/main/vibevoice-voice-fr-Spk1_woman.gguf"},
    {nullptr, nullptr},
};

// CosyVoice3 needs HiFT + voices in addition to flow (which rides as the
// inline companion). The CLI auto-discovers them as siblings of the LLM
// at load time; staging both here makes `-m auto --backend cosyvoice3-tts`
// pull everything in one go.
constexpr ExtraCompanion k_cosyvoice3_tts_extras[] = {
    {"cosyvoice3-campplus-f16.gguf",
     "https://huggingface.co/cstr/cosyvoice3-0.5b-2512-GGUF/resolve/main/cosyvoice3-campplus-f16.gguf"},
    {"cosyvoice3-s3tok-f16.gguf",
     "https://huggingface.co/cstr/cosyvoice3-0.5b-2512-GGUF/resolve/main/cosyvoice3-s3tok-f16.gguf"},
    {"cosyvoice3-hift-f16.gguf",
     "https://huggingface.co/cstr/cosyvoice3-0.5b-2512-GGUF/resolve/main/cosyvoice3-hift-f16.gguf"},
    {"cosyvoice3-voices.gguf",
     "https://huggingface.co/cstr/cosyvoice3-0.5b-2512-GGUF/resolve/main/cosyvoice3-voices.gguf"},
    {nullptr, nullptr},
};

// qwen3-tts Base variants need a default voice pack so synthesis works
// without --voice. One pack covers both 0.6B and 1.7B Base since the
// voice embedding format is model-size-agnostic (spk_embedding + ref_code).
constexpr ExtraCompanion k_qwen3_tts_base_extras[] = {
    {"qwen3-tts-voice-default.gguf",
     "https://huggingface.co/cstr/qwen3-tts-voices-GGUF/resolve/main/qwen3-tts-voice-default.gguf"},
    {nullptr, nullptr},
};

// TADA generation expects an aligned acoustic prompt, matching the official
// model.generate(prompt=...) path. Ship the JFK prompt with auto-download so
// `-m auto --backend tada[-1b] --tts ...` does not fall back to the unprompted
// path with unstable timing.
constexpr ExtraCompanion k_tada_1b_extras[] = {
    {"tada-ref.gguf", "https://huggingface.co/cstr/tada-tts-1b-GGUF/resolve/main/tada-ref.gguf"},
    {nullptr, nullptr},
};

constexpr ExtraCompanion k_tada_3b_extras[] = {
    {"tada-ref.gguf", "https://huggingface.co/cstr/tada-tts-3b-ml-GGUF/resolve/main/tada-ref.gguf"},
    {nullptr, nullptr},
};

constexpr ExtraList k_extras[] = {
    {"kokoro", k_kokoro_extras},
    {"vibevoice-tts", k_vibevoice_tts_extras},
    {"cosyvoice3-tts", k_cosyvoice3_tts_extras},
    {"qwen3-tts", k_qwen3_tts_base_extras},
    {"qwen3-tts-1.7b-base", k_qwen3_tts_base_extras},
    {"tada", k_tada_3b_extras},
    {"tada-1b", k_tada_1b_extras},
    {"tada-tts-1b", k_tada_1b_extras},
    {nullptr, nullptr},
};
// clang-format on

const Entry* find_by_backend(const std::string& backend) {
    for (const auto& e : k_registry)
        if (backend == e.backend)
            return &e;
    // Fallback: the CLI passes the raw `--backend` alias (e.g. the short
    // `cosyvoice3` / `voxcpm2`) while the registry keys the canonical
    // `cosyvoice3-tts` / `voxcpm2-tts`. When the exact alias has no entry,
    // retry with a `-tts` suffix so `-m auto --backend cosyvoice3` resolves
    // instead of failing with "no default model registered". Exact match is
    // tried first, so this can never shadow a real non-`-tts` entry.
    if (backend.size() < 4 || backend.compare(backend.size() - 4, 4, "-tts") != 0) {
        const std::string with_tts = backend + "-tts";
        for (const auto& e : k_registry)
            if (with_tts == e.backend)
                return &e;
    }
    return nullptr;
}

std::string basename_of(const std::string& p) {
    std::string base = p;
    auto slash = base.rfind('/');
    if (slash != std::string::npos)
        base = base.substr(slash + 1);
    auto bslash = base.rfind('\\');
    if (bslash != std::string::npos)
        base = base.substr(bslash + 1);
    return base;
}

const Entry* find_by_filename(const std::string& filename) {
    const std::string base = basename_of(filename);
    for (const auto& e : k_registry)
        if (base == e.filename)
            return &e;
    for (const auto& e : k_registry)
        if (base.find(e.filename) != std::string::npos)
            return &e;
    return nullptr;
}

bool has_suffix_ci(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size())
        return false;
    for (size_t i = 0; i < suffix.size(); ++i) {
        char a = (char)std::tolower((unsigned char)s[s.size() - suffix.size() + i]);
        char b = (char)std::tolower((unsigned char)suffix[i]);
        if (a != b)
            return false;
    }
    return true;
}

std::string strip_known_quant_suffix(const std::string& filename) {
    const std::string dot = ".gguf";
    if (!has_suffix_ci(filename, dot))
        return filename;

    const size_t stem_end = filename.size() - dot.size();
    const std::string stem = filename.substr(0, stem_end);
    static const char* k_known_quants[] = {
        "-q2_k", "-q3_k", "-q4_0", "-q4_1", "-q4_k", "-q4_k_m", "-q5_0",
        "-q5_1", "-q5_k", "-q6_k", "-q8_0", "-f16",  "-bf16",
    };
    for (const char* q : k_known_quants) {
        const std::string suffix = q;
        if (has_suffix_ci(stem, suffix)) {
            return stem.substr(0, stem.size() - suffix.size()) + dot;
        }
    }
    return filename;
}

std::string apply_quant_to_filename(const std::string& filename, const std::string& preferred_quant) {
    if (preferred_quant.empty())
        return filename;

    const std::string lower_quant = preferred_quant;
    const std::string dot = ".gguf";
    if (!has_suffix_ci(filename, dot))
        return filename;

    const size_t stem_end = filename.size() - dot.size();
    const std::string stem = filename.substr(0, stem_end);
    static const char* k_known_quants[] = {
        "-q2_k", "-q3_k", "-q4_0", "-q4_1", "-q4_k", "-q4_k_m", "-q5_0",
        "-q5_1", "-q5_k", "-q6_k", "-q8_0", "-f16",  "-bf16",
    };
    for (const char* q : k_known_quants) {
        const std::string suffix = q;
        if (has_suffix_ci(stem, suffix)) {
            return stem.substr(0, stem.size() - suffix.size()) + "-" + lower_quant + dot;
        }
    }
    return filename;
}

std::string replace_tail_filename(const std::string& url, const std::string& old_name, const std::string& new_name) {
    const size_t pos = url.rfind(old_name);
    if (pos == std::string::npos)
        return url;
    return url.substr(0, pos) + new_name + url.substr(pos + old_name.size());
}

void fill(CrispasrRegistryEntry& out, const Entry& e, const std::string& preferred_quant) {
    const std::string filename = apply_quant_to_filename(e.filename, preferred_quant);
    out.backend = e.backend;
    out.filename = filename;
    out.url = replace_tail_filename(e.url, e.filename, filename);
    out.approx_size = e.approx_size;
    if (e.companion_file && e.companion_url) {
        out.companion_filename = apply_quant_to_filename(e.companion_file, preferred_quant);
        out.companion_url = replace_tail_filename(e.companion_url, e.companion_file, out.companion_filename);
        out.companion_approx_size = e.companion_size ? e.companion_size : e.approx_size;
    } else {
        out.companion_filename.clear();
        out.companion_url.clear();
        out.companion_approx_size.clear();
    }
    out.license = e.license ? e.license : "";
}

const ExtraCompanion* find_extras(const char* backend) {
    if (!backend)
        return nullptr;
    for (const auto& x : k_extras) {
        if (!x.backend)
            break;
        if (std::string(backend) == x.backend)
            return x.items;
    }
    return nullptr;
}

void download_extras(const Entry& e, bool quiet, const std::string& cache_dir_override) {
    const ExtraCompanion* extras = find_extras(e.backend);
    if (!extras)
        return;
    for (const ExtraCompanion* it = extras; it->file && it->url; ++it) {
        crispasr_cache::ensure_cached_file(it->file, it->url, quiet, "crispasr", cache_dir_override);
    }
}

static bool license_is_nc(const std::string& lic) {
    return lic.find("NC") != std::string::npos || lic.find("NonCommercial") != std::string::npos;
}

void print_license_note(const CrispasrRegistryEntry& e, bool quiet) {
    if (!quiet && !e.license.empty()) {
        if (license_is_nc(e.license)) {
            fprintf(stderr,
                    "crispasr: WARNING: %s is licensed %s — NON-COMMERCIAL USE ONLY.\n"
                    "  By loading this model you confirm you will not use it for commercial purposes.\n",
                    e.filename.c_str(), e.license.c_str());
        } else {
            fprintf(stderr, "crispasr: note: %s is licensed %s\n", e.filename.c_str(), e.license.c_str());
        }
    }
}

} // namespace

bool crispasr_registry_lookup(const std::string& backend, CrispasrRegistryEntry& out,
                              const std::string& preferred_quant) {
    const Entry* e = find_by_backend(backend);
    if (!e)
        return false;
    fill(out, *e, preferred_quant);
    return true;
}

bool crispasr_registry_lookup_by_filename(const std::string& filename, CrispasrRegistryEntry& out,
                                          const std::string& preferred_quant) {
    const Entry* e = find_by_filename(filename);
    if (e) {
        fill(out, *e, preferred_quant);
        return true;
    }

    const std::string base = basename_of(filename);
    for (const auto& row : k_registry) {
        if (!row.companion_file || !row.companion_url)
            continue;
        const std::string companion = apply_quant_to_filename(row.companion_file, preferred_quant);
        if (strip_known_quant_suffix(base) == strip_known_quant_suffix(companion)) {
            out.backend = row.backend;
            out.filename = base;
            out.url = replace_tail_filename(row.companion_url, row.companion_file, base);
            out.approx_size = row.companion_size ? row.companion_size : row.approx_size;
            out.companion_filename.clear();
            out.companion_url.clear();
            out.companion_approx_size.clear();
            return true;
        }
    }

    return false;
}

int crispasr_registry_count() {
    return (int)(sizeof(k_registry) / sizeof(k_registry[0]));
}

bool crispasr_registry_get_at(int i, CrispasrRegistryEntry& out, const std::string& preferred_quant) {
    if (i < 0 || i >= crispasr_registry_count())
        return false;
    fill(out, k_registry[i], preferred_quant);
    return true;
}

bool crispasr_find_cached_model(CrispasrRegistryEntry& out, const std::string& cache_dir_override,
                                const std::string& preferred_quant) {
    // k_registry is already ordered whisper > parakeet > canary > ... —
    // first entry wins, which matches the documented preference.
    const std::string dir = crispasr_cache::dir(cache_dir_override);
    for (const auto& e : k_registry) {
        const std::string path = dir + "/" + apply_quant_to_filename(e.filename, preferred_quant);
        if (crispasr_cache::file_present(path)) {
            fill(out, e, preferred_quant);
            return true;
        }
    }
    return false;
}

std::string crispasr_resolve_model(const std::string& model_arg, const std::string& backend_name, bool quiet,
                                   const std::string& cache_dir_override, bool allow_download,
                                   const std::string& preferred_quant) {
    // Concrete path that exists on disk — pass through.
    if (model_arg != "auto" && model_arg != "default") {
        FILE* f = fopen(model_arg.c_str(), "rb");
        if (f) {
            fclose(f);
            return model_arg;
        }

        // File not found — try registry-based download when permitted.
        // Match priority:
        //   1. exact filename / known-companion match (e.g. -m parakeet-tdt-0.6b-v2-q4_k.gguf)
        //   2. backend-key match on the literal -m arg (e.g. -m parakeet-v2 → the parakeet-v2 entry)
        //   3. fallback: backend name passed via --backend (e.g. -m foo.gguf --backend parakeet)
        // Step 2 must precede step 3, otherwise the CLI's filename-inferred
        // backend (always "parakeet" for any "parakeet*" arg) would shadow
        // sub-variant keys like "parakeet-v2" / "parakeet-tdt-1.1b" / etc.
        CrispasrRegistryEntry match;
        bool have_match = crispasr_registry_lookup_by_filename(model_arg, match, preferred_quant);
        if (!have_match)
            have_match = crispasr_registry_lookup(model_arg, match, preferred_quant);
        if (!have_match && !backend_name.empty())
            have_match = crispasr_registry_lookup(backend_name, match, preferred_quant);

        if (have_match) {
            const std::string cached = crispasr_cache::dir(cache_dir_override) + "/" + match.filename;
            if (crispasr_cache::file_present(cached))
                return cached;
        }

        if (have_match && allow_download) {
            if (!quiet) {
                fprintf(stderr, "crispasr: model '%s' not found locally — downloading %s (%s)\n", model_arg.c_str(),
                        match.filename.c_str(), match.approx_size.c_str());
            }
            std::string dl =
                crispasr_cache::ensure_cached_file(match.filename, match.url, quiet, "crispasr", cache_dir_override);
            if (!dl.empty() && !match.companion_filename.empty() && !match.companion_url.empty())
                crispasr_cache::ensure_cached_file(match.companion_filename, match.companion_url, quiet, "crispasr",
                                                   cache_dir_override);
            if (!dl.empty()) {
                if (const Entry* match_entry =
                        !backend_name.empty() ? find_by_backend(backend_name) : find_by_filename(model_arg))
                    download_extras(*match_entry, quiet, cache_dir_override);
                print_license_note(match, quiet);
            }
            return dl;
        }
        // Either no registry match or caller didn't authorise download —
        // return the arg untouched so the caller can decide (prompt,
        // error, etc.).
        return model_arg;
    }

    CrispasrRegistryEntry e;
    if (!crispasr_registry_lookup(backend_name, e, preferred_quant)) {
        fprintf(stderr, "crispasr: -m auto not supported for backend '%s' (no default model registered)\n",
                backend_name.c_str());
        return "";
    }

    if (!quiet)
        fprintf(stderr, "crispasr: resolving %s (%s) via -m auto\n", e.filename.c_str(), e.approx_size.c_str());
    std::string result = crispasr_cache::ensure_cached_file(e.filename, e.url, quiet, "crispasr", cache_dir_override);

    // Download companion file (e.g. tokenizer.bin for moonshine) if needed
    if (!result.empty() && !e.companion_filename.empty() && !e.companion_url.empty()) {
        crispasr_cache::ensure_cached_file(e.companion_filename, e.companion_url, quiet, "crispasr",
                                           cache_dir_override);
    }
    // Backend-specific extras (e.g. kokoro German backbone + voice) — opt-in
    // per backend via k_extras.
    if (!result.empty()) {
        if (const Entry* entry = find_by_backend(backend_name))
            download_extras(*entry, quiet, cache_dir_override);
        print_license_note(e, quiet);
    }

    return result;
}
