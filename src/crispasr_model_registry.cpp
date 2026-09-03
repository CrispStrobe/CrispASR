// crispasr_model_registry.cpp — implementation. See the header.

#include "crispasr_model_registry.h"
#include "crispasr_cache.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#ifdef _WIN32
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif
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
    const char* license;        // NULL = permissive; otherwise SPDX-ish tag + source/license URL
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
    // Tiron (#295, EXPERIMENTAL): Whisper large-v3 + inline <|speakerN|> markers.
    // `--backend tiron -m auto` (alias → whisper backend in crispasr_backend.cpp);
    // the whisper loader auto-detects the speaker vocab and enables the tiron
    // constrained-decoding grammar. Apache-2.0 (base Trelis/tiron).
    {"tiron", "tiron-q4_k.bin",
     "https://huggingface.co/cstr/tiron-GGML/resolve/main/tiron-q4_k.bin", "~889 MB", nullptr, nullptr},
    // AudioSeal (Meta FAIR) — optional NEURAL watermark model (MIT code + weights).
    // Not a transcription/TTS backend: resolved by the name "audioseal" when the
    // user passes `--watermark-model auto`. CrispASR's built-in zero-dependency
    // spread-spectrum watermark stays the always-on default; this is the opt-in
    // SOTA upgrade (imperceptible + robust neural watermark). #260.
    {"audioseal", "audioseal.gguf",
     "https://huggingface.co/cstr/audioseal-GGUF/resolve/main/audioseal.gguf", "~89 MB", nullptr, nullptr},
    // §248 source separation (--separate). Mel-Band RoFormer vocal/instrumental
    // split; MIT weights (KimberleyJSN) + MIT code (lucidrains). Resolved when
    // `--separate` runs with the default separation backend key.
    {"mel-band-roformer", "mel-band-roformer-vocals-f16.gguf",
     "https://huggingface.co/cstr/mel-band-roformer-vocals-GGUF/resolve/main/mel-band-roformer-vocals-f16.gguf",
     "~436 MB", nullptr, nullptr},
    // Sidon v0.1 (SaruLab) speech restoration (--s2s). w2v-BERT predictor +
    // continuous DAC decoder; MIT. Q8_0 is the best-fidelity practical quant
    // (predictor cos 0.998 vs the upstream F32; ASR round-trip identical). #283.
    {"sidon", "sidon-v0.1-q8_0.gguf",
     "https://huggingface.co/cstr/Sidon-GGUF/resolve/main/sidon-v0.1-q8_0.gguf", "~335 MB", nullptr, nullptr},
    // GigaAM-v3 (ai-sage/GigaAM-v3) — Russian ASR. Default to the e2e_rnnt
    // revision at Q8_0: lowest WER, emits punctuation + casing + ITN, and its
    // transcript is byte-identical to the PyTorch reference. Q4_K is published
    // too but drops a few capital letters on the SentencePiece variants, so it
    // is not the auto-download default.
    {"gigaam", "gigaam-v3-e2e-rnnt-q8_0.gguf",
     "https://huggingface.co/cstr/gigaam-v3-GGUF/resolve/main/gigaam-v3-e2e-rnnt-q8_0.gguf", "~249 MB", nullptr,
     nullptr},
    // #324: speaker embedder for --diarize-method foxnose. NOT an ASR backend —
    // registered so --diarize-embedder auto can fetch it. ⚠ CC-BY-4.0 weights:
    // redistribution requires attribution (see THIRD_PARTY_NOTICES.txt).
    {"wespeaker", "wespeaker-resnet34-lm.gguf",
     "https://huggingface.co/cstr/wespeaker-resnet34-lm-GGUF/resolve/main/wespeaker-resnet34-lm.gguf", "~24 MB",
     nullptr, nullptr, nullptr, "CC-BY-4.0 (weights; attribution required; see https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM)"},
    {"nemotron", "nemotron-3.5-asr-streaming-0.6b-q4_k.gguf",
     "https://huggingface.co/cstr/nemotron-3.5-asr-streaming-0.6b-GGUF/resolve/main/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf",
     "~458 MB", nullptr, nullptr},
    {"parakeet", "parakeet-tdt-0.6b-v3-q4_k.gguf",
     "https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF/resolve/main/parakeet-tdt-0.6b-v3-q4_k.gguf", "~467 MB", nullptr, nullptr, nullptr, "CC-BY-4.0 (see https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)"},
    // Quds v4 (#387): Persian ASR, FastConformer-RNNT fine-tune of the
    // CC-BY-4.0 NVIDIA stt_fa base, by hojreh (converted from the author's
    // ONNX export — models/convert-nemo-rnnt-onnx-to-gguf.py). Rides the
    // parakeet runtime (pred_layers=1, no CTC head in the export). Q8_0 is
    // transcript-identical to F16 on the CV-fa validation clips.
    {"quds", "quds-v4-fa-q8_0.gguf",
     "https://huggingface.co/cstr/quds-v4-fa-GGUF/resolve/main/quds-v4-fa-q8_0.gguf", "~122 MB", nullptr, nullptr,
     nullptr,
     "CC-BY-NC-4.0 — NON-COMMERCIAL use only (Quds v4 by hojreh, "
     "https://huggingface.co/hojreh/Quds-v4-onnx; base model nvidia/stt_fa_fastconformer_hybrid_large, CC-BY-4.0)"},
    {"quds-fa", "quds-v4-fa-q8_0.gguf",
     "https://huggingface.co/cstr/quds-v4-fa-GGUF/resolve/main/quds-v4-fa-q8_0.gguf", "~122 MB", nullptr, nullptr,
     nullptr,
     "CC-BY-NC-4.0 — NON-COMMERCIAL use only (Quds v4 by hojreh, "
     "https://huggingface.co/hojreh/Quds-v4-onnx; base model nvidia/stt_fa_fastconformer_hybrid_large, CC-BY-4.0)"},
    {"canary", "canary-1b-v2-q4_k.gguf",
     "https://huggingface.co/cstr/canary-1b-v2-GGUF/resolve/main/canary-1b-v2-q4_k.gguf", "~600 MB", nullptr, nullptr, nullptr, "CC-BY-4.0 (see https://huggingface.co/nvidia/canary-1b-v2)"},
    {"canary-qwen", "canary-qwen-2.5b-q8_0.gguf",
     "https://huggingface.co/cstr/canary-qwen-2.5b-GGUF/resolve/main/canary-qwen-2.5b-q8_0.gguf", "~4.1 GB", nullptr, nullptr, nullptr, "CC-BY-4.0 (see https://huggingface.co/nvidia/canary-qwen-2.5b)"},
    // AutoArk-AI/ARK-ASR-3B: Whisper-RoPE encoder + Qwen2.5-3B decoder (19-lang).
    // NOTE: GGUF repo to be published (PLAN §ARK) — placeholder URL.
    {"ark-asr", "ark-asr-3b-q4_k.gguf",
     "https://huggingface.co/cstr/ark-asr-3b-GGUF/resolve/main/ark-asr-3b-q4_k.gguf", "~2.2 GB", nullptr, nullptr},
    // LiquidAI LFM2.5-Audio-1.5B: FastConformer + LFM2 hybrid
    // conv+attention backbone. ASR (+ TTS/speech-to-speech planned).
    // English base model — Q5_K recommended (Q4_K too aggressive for EN).
    {"lfm2-audio", "lfm2-audio-1.5b-q5_k.gguf",
     "https://huggingface.co/cstr/lfm2-audio-1.5b-GGUF/resolve/main/lfm2-audio-1.5b-q5_k.gguf",
     "~1.6 GB", nullptr, nullptr, nullptr,
     "lfm1.0 (commercial use threshold and attribution; see "
     "https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B)"},
    // Japanese variant — Q5_K minimum (Q4_K produces 0 tokens on hybrid backbone).
    {"lfm2-audio", "lfm2-audio-1.5b-jp-q5_k.gguf",
     "https://huggingface.co/cstr/lfm2-audio-1.5b-jp-GGUF/resolve/main/lfm2-audio-1.5b-jp-q5_k.gguf",
     "~1.5 GB", nullptr, nullptr, nullptr,
     "lfm1.0 (commercial use threshold and attribution; see "
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
    // Qwen3-ASR-1.7B fine-tuned for Japanese anime/galgame speech (Apache-2.0).
    // Same architecture as qwen3-1.7b; uses the standard qwen3 backend.
    {"qwen3-ja-anime", "qwen3-asr-1.7b-ja-anime-q4_k.gguf",
     "https://huggingface.co/cstr/qwen3-asr-1.7b-ja-anime-GGUF/resolve/main/qwen3-asr-1.7b-ja-anime-q4_k.gguf",
     "~1.3 GB", nullptr, nullptr},
    // higgs-audio-v3-stt: Whisper-large-v3 encoder + Qwen3-1.7B decoder (Apache-2.0).
    {"higgs-stt", "higgs-stt-q4_k.gguf",
     "https://huggingface.co/cstr/higgs-audio-v3-stt-GGUF/resolve/main/higgs-stt-q4_k.gguf",
     "~2.3 GB", nullptr, nullptr},
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
     "funasr-v1.1 (commercial use OK with attribution; see "
     "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/blob/main/LICENSE)"},
    // Multilingual sibling — same architecture, 31 languages including
    // Korean, Vietnamese, Indonesian, Thai, Malay, Filipino, Arabic,
    // Hindi, Bulgarian, German, French, Spanish, Italian, Portuguese,
    // Dutch, Polish, Czech, Romanian, Greek, Finnish, Swedish, Turkish,
    // Persian, Danish, Hungarian, Macedonian, Russian.
    {"fun-asr-mlt-nano", "funasr-mlt-nano-2512-f16.gguf",
     "https://huggingface.co/cstr/funasr-mlt-nano-GGUF/resolve/main/funasr-mlt-nano-2512-f16.gguf",
     "~1.98 GB", nullptr, nullptr, nullptr,
     "funasr-v1.1 (commercial use OK with attribution; see "
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
     "funasr-v1.1 (commercial use OK with attribution; see "
     "https://huggingface.co/FunAudioLLM/SenseVoiceSmall/blob/main/LICENSE)"},
    // F16 + Q8_0 lookups by canonical filename (auto-download resolver
    // still routes here when the user passes the explicit filename or
    // -m auto:variant).
    {"sensevoice", "sensevoice-small-f16.gguf",
     "https://huggingface.co/cstr/sensevoice-small-GGUF/resolve/main/sensevoice-small-f16.gguf",
     "~448 MB", nullptr, nullptr, nullptr,
     "funasr-v1.1 (commercial use OK with attribution; see "
     "https://huggingface.co/FunAudioLLM/SenseVoiceSmall/blob/main/LICENSE)"},
    {"sensevoice", "sensevoice-small-q8_0.gguf",
     "https://huggingface.co/cstr/sensevoice-small-GGUF/resolve/main/sensevoice-small-q8_0.gguf",
     "~240 MB", nullptr, nullptr, nullptr,
     "funasr-v1.1 (commercial use OK with attribution; see "
     "https://huggingface.co/FunAudioLLM/SenseVoiceSmall/blob/main/LICENSE)"},
    // Paraformer-zh: NAR-ASR, 220M params, zh+en character-level.
    // Q4_K default — byte-identical transcript to F16, 3.4× smaller.
    {"paraformer", "paraformer-zh-q4_k.gguf",
     "https://huggingface.co/cstr/paraformer-zh-GGUF/resolve/main/paraformer-zh-q4_k.gguf",
     "~123 MB", nullptr, nullptr, nullptr,
     "funasr-v1.1 (commercial use OK with attribution; see "
     "https://huggingface.co/funasr/paraformer-zh)"},
    {"paraformer", "paraformer-zh-f16.gguf",
     "https://huggingface.co/cstr/paraformer-zh-GGUF/resolve/main/paraformer-zh-f16.gguf",
     "~421 MB", nullptr, nullptr, nullptr,
     "funasr-v1.1 (commercial use OK with attribution; see "
     "https://huggingface.co/funasr/paraformer-zh)"},
    {"paraformer", "paraformer-zh-q8_0.gguf",
     "https://huggingface.co/cstr/paraformer-zh-GGUF/resolve/main/paraformer-zh-q8_0.gguf",
     "~227 MB", nullptr, nullptr, nullptr,
     "funasr-v1.1 (commercial use OK with attribution; see "
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
    // cohere-transcribe-arabic — Arabic model (CohereLabs/cohere-transcribe-arabic-07-2026,
    // Apache-2.0; #231). q4_k-imatrix is the recommended Arabic variant (Arabic
    // CC0 Common Voice calibration; matches F16 transcript where plain q4_k drifts).
    {"cohere", "cohere-transcribe-arabic-q4_k-imatrix.gguf",
     "https://huggingface.co/cstr/cohere-transcribe-arabic-07-2026-GGUF/resolve/main/cohere-transcribe-arabic-q4_k-imatrix.gguf",
     "~1.5 GB", nullptr, nullptr},
    // `--backend cohere-ar` (or `-m auto --backend cohere-ar`) short alias →
    // recommended Arabic imatrix GGUF. Routes to the cohere runtime via the
    // factory alias in crispasr_backend.cpp. Still pass `-l ar`.
    {"cohere-ar", "cohere-transcribe-arabic-q4_k-imatrix.gguf",
     "https://huggingface.co/cstr/cohere-transcribe-arabic-07-2026-GGUF/resolve/main/cohere-transcribe-arabic-q4_k-imatrix.gguf",
     "~1.5 GB", nullptr, nullptr},
    {"cohere", "cohere-transcribe-arabic-q4_k.gguf",
     "https://huggingface.co/cstr/cohere-transcribe-arabic-07-2026-GGUF/resolve/main/cohere-transcribe-arabic-q4_k.gguf",
     "~1.5 GB", nullptr, nullptr},
    {"cohere", "cohere-transcribe-arabic-q8_0.gguf",
     "https://huggingface.co/cstr/cohere-transcribe-arabic-07-2026-GGUF/resolve/main/cohere-transcribe-arabic-q8_0.gguf",
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
    {"moss-transcribe", "moss-transcribe-preview-2b-q4_k.gguf",
     "https://huggingface.co/cstr/MOSS-Transcribe-preview-2B-GGUF/resolve/main/moss-transcribe-preview-2b-q4_k.gguf",
     "~1.6 GB", nullptr, nullptr},
    {"moss-diarize", "moss-transcribe-diarize-0.9b-q4_k.gguf",
     "https://huggingface.co/cstr/MOSS-Transcribe-Diarize-GGUF/resolve/main/moss-transcribe-diarize-0.9b-q4_k.gguf",
     "~1.2 GB", nullptr, nullptr},
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
    {"vibevoice-bitnet", "vibevoice-asr-bitnet-tq2.gguf",
     "https://huggingface.co/cstr/vibevoice-asr-bitnet-GGUF/resolve/main/vibevoice-asr-bitnet-tq2.gguf", "~1.6 GB",
     nullptr, nullptr},
    {"vibevoice-1.5b", "vibevoice-1.5b-tts-q4_k.gguf",
     "https://huggingface.co/cstr/vibevoice-1.5b-GGUF/resolve/main/vibevoice-1.5b-tts-q4_k.gguf", "~1.6 GB", nullptr,
     nullptr},
    {"vibevoice-tts", "vibevoice-realtime-0.5b-q4_k.gguf",
     "https://huggingface.co/cstr/vibevoice-realtime-0.5b-GGUF/resolve/main/vibevoice-realtime-0.5b-q4_k.gguf",
     "~636 MB",
     "vibevoice-voice-emma.gguf",
     "https://huggingface.co/cstr/vibevoice-realtime-0.5b-GGUF/resolve/main/vibevoice-voice-emma.gguf",
     "~3 MB"},
    // F16 (17.3 GB, not the "~14 GB" this once claimed), and it stays F16 because
    // the published Q4_K is BROKEN, not merely lossy: measured on Kaggle it
    // stutters and loops — "The quick brown fast The quick brown the quick brown
    // fox jobs over the lazy job ..." (WER 0.72, 13.7 s of audio for a 7.5 s
    // sentence) where F16 gives WER 0.056. Defaulting to it would trade a loud
    // OOM for silently broken speech.
    // The real cost: 17.3 GB does not fit a 16 GB card, so `-m auto` downloads
    // 17 GB and then fails to allocate. The fix is a quant that fits AND holds up
    // (Q6_K/Q8_0, likely keeping the DiT head + VAE decoder in F16 — the usual
    // shape for this family), not the Q4_K we have. Until then the loader at
    // least says what happened and that --no-gpu exists.
    {"kugelaudio", "kugelaudio-0-open-f16.gguf",
     "https://huggingface.co/cstr/kugelaudio-0-open-GGUF/resolve/main/kugelaudio-0-open-f16.gguf", "~17.3 GB", nullptr,
     nullptr},
    {"firered-asr", "firered-asr2-aed-q4_k.gguf",
     "https://huggingface.co/cstr/firered-asr2-aed-GGUF/resolve/main/firered-asr2-aed-q4_k.gguf", "~918 MB", nullptr, nullptr},
    {"kyutai-stt", "kyutai-stt-1b-q4_k.gguf",
     "https://huggingface.co/cstr/kyutai-stt-1b-GGUF/resolve/main/kyutai-stt-1b-q4_k.gguf", "~636 MB", nullptr, nullptr},
    {"kyutai-stt-2.6b", "kyutai-stt-2.6b-q4_k.gguf",
     "https://huggingface.co/cstr/kyutai-stt-2.6b-en-GGUF/resolve/main/kyutai-stt-2.6b-q4_k.gguf", "~1.5 GB",
     nullptr, nullptr},
    {"voxtral-tts", "voxtral-4b-tts-q4_k.gguf",
     "https://huggingface.co/cstr/voxtral-4b-tts-GGUF/resolve/main/voxtral-4b-tts-q4_k.gguf", "~2.5 GB", nullptr,
     nullptr, nullptr,
     "CC-BY-NC-4.0 — NON-COMMERCIAL use only (base model mistralai/Voxtral-4B-TTS-2603; see "
     "https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)"},
    {"htdemucs", "htdemucs-q4_k.gguf",
     "https://huggingface.co/cstr/htdemucs-GGUF/resolve/main/htdemucs-q4_k.gguf", "~38 MB", nullptr, nullptr},
    // BTC chord recognition (--chords). Upstream CODE is MIT, but the SHIPPED
    // WEIGHTS are CC-BY-NC-SA: they were trained on Isophonics / Robbie
    // Williams / UsPop2002 chord annotations, which are non-commercial. The
    // licence field below is what arms the acceptance gate — without it these
    // would download silently to commercial users.
    //
    // DEFAULT IS THE 170-CLASS MODEL: it collapses to the 25-class maj/min
    // vocabulary on demand (CRISPASR_BTC_MAJ_MIN=1), whereas a 25-class model
    // can never be expanded. One model, two output modes.
    {"btc-chords", "btc-chords-large-f16.gguf",
     "https://huggingface.co/cstr/btc-chords-GGUF/resolve/main/btc-chords-large-f16.gguf", "~6 MB", nullptr, nullptr,
     nullptr, "cc-by-nc-sa-4.0"},
    {"btc-chords-large", "btc-chords-large-f16.gguf",
     "https://huggingface.co/cstr/btc-chords-GGUF/resolve/main/btc-chords-large-f16.gguf", "~6 MB", nullptr, nullptr,
     nullptr, "cc-by-nc-sa-4.0"},
    {"btc-chords-majmin", "btc-chords-f16.gguf",
     "https://huggingface.co/cstr/btc-chords-GGUF/resolve/main/btc-chords-f16.gguf", "~6 MB", nullptr, nullptr, nullptr,
     "cc-by-nc-sa-4.0"},
    // q8_0 — 4.5 MB, and indistinguishable from f16: 13/13 diff stages, and on
    // 257 s of real music vs the torch reference root 99.17 % / tetrads
    // 98.52 % (f16 is 99.17 / 98.56). NO q4_k is published: it costs 3.1 points
    // of tetrad accuracy (95.46 %) to save 0.6 MB. Quantize from the f16 — only
    // 73/213 tensors are quantizable, so a q8_0 built from f32 lands at 7.5 MB,
    // LARGER than the f16 it was meant to shrink.
    {"btc-chords-q8_0", "btc-chords-large-q8_0.gguf",
     "https://huggingface.co/cstr/btc-chords-GGUF/resolve/main/btc-chords-large-q8_0.gguf", "~4.5 MB", nullptr, nullptr,
     nullptr, "cc-by-nc-sa-4.0"},
    {"btc-chords-majmin-q8_0", "btc-chords-q8_0.gguf",
     "https://huggingface.co/cstr/btc-chords-GGUF/resolve/main/btc-chords-q8_0.gguf", "~4.4 MB", nullptr, nullptr,
     nullptr, "cc-by-nc-sa-4.0"},
    // Beat This! beat/downbeat tracking (--beats). MIT for code AND weights,
    // so no licence gate — deliberately unlike btc-chords above. The reason it
    // can be MIT at all is that it needs no DBN: madmom's is Boeck-patented and
    // non-commercial, and this model reaches SOTA without one.
    //
    // DEFAULT IS F16: cos >= 0.9999997 per stage against the torch reference,
    // with the residual flat across all 12 sub-blocks (weight quantisation, not
    // drift). f32 is exact (cos = 1.00000000, rel err ~1e-6) and is kept for
    // parity debugging, not for production — it is 2x the size for no
    // measurable difference in detected beats.
    // TabCNN guitar tablature (--tab). Weights are CC BY 4.0 from the EGSet12
    // record (https://zenodo.org/records/11406378) — attribution required, commercial
    // use permitted. This is the GuitarProFX-augmented variant: the vanilla
    // GuitarSet-trained model collapses from tablature F1 0.748 to 0.447 on
    // real electric guitar, the augmented one recovers to 0.585 (DAFx-24).
    {"tabcnn", "tabcnn-f16.gguf", "https://huggingface.co/cstr/tabcnn-GGUF/resolve/main/tabcnn-f16.gguf", "~1.8 MB",
     nullptr, nullptr, nullptr, "cc-by-4.0"},
    // Quants get their OWN entries or `-m auto` can never reach them — the exact
    // gap that left crepe's q8_0/q4_k unreachable on HF for weeks. Both preserve
    // head.weight at f32; measured on EGSet12 track 01 they are indistinguishable
    // from f16 (F1 0.7749 vs 0.7732, within noise), so q4_k is the size pick.
    // NOTE the "q4_k" file is really Q4_0 — no tensor is 256-aligned, so k-quants
    // fall back. The name follows the request, not the content.
    {"tabcnn-q8_0", "tabcnn-q8_0.gguf", "https://huggingface.co/cstr/tabcnn-GGUF/resolve/main/tabcnn-q8_0.gguf",
     "~1.1 MB", nullptr, nullptr, nullptr, "cc-by-4.0"},
    {"tabcnn-q4_k", "tabcnn-q4_k.gguf", "https://huggingface.co/cstr/tabcnn-GGUF/resolve/main/tabcnn-q4_k.gguf",
     "~0.7 MB", nullptr, nullptr, nullptr, "cc-by-4.0"},
    {"beat-this", "beat-this-f16.gguf",
     "https://huggingface.co/cstr/beat-this-GGUF/resolve/main/beat-this-f16.gguf", "~41 MB", nullptr, nullptr, nullptr,
     "MIT"},
    {"beat-this-f32", "beat-this-f32.gguf",
     "https://huggingface.co/cstr/beat-this-GGUF/resolve/main/beat-this-f32.gguf", "~81 MB", nullptr, nullptr, nullptr,
     "MIT"},
    // CREPE monophonic F0 (--pitch). MIT weights + MIT code (torchcrepe).
    // DEFAULT IS TINY: measured RTF 0.28 on Metal vs 2.0 for full, and full is
    // 38x the MACs per frame. `full` stays available for offline accuracy.
    {"crepe", "crepe-tiny-f16.gguf",
     "https://huggingface.co/cstr/crepe-GGUF/resolve/main/crepe-tiny-f16.gguf", "~1.0 MB", nullptr, nullptr, nullptr,
     "MIT"},
    {"crepe-tiny", "crepe-tiny-f16.gguf",
     "https://huggingface.co/cstr/crepe-GGUF/resolve/main/crepe-tiny-f16.gguf", "~1.0 MB", nullptr, nullptr, nullptr,
     "MIT"},
    {"crepe-full", "crepe-full-f16.gguf",
     "https://huggingface.co/cstr/crepe-GGUF/resolve/main/crepe-full-f16.gguf", "~44.5 MB", nullptr, nullptr, nullptr,
     "MIT"},
    // Quantized CREPE. These were uploaded to cstr/crepe-GGUF but had NO
    // registry entry, so `-m auto` could never select them. Measured against
    // the f16 of the same capacity on a 3 s tone (test-crepe-parity):
    //   tiny q8_0  cos 0.999993   f0 872.54 vs 872.44 Hz
    //   tiny q4_k  cos 0.998757   f0 871.84 vs 872.44 Hz
    //   full q8_0  cos 0.999999   f0 880.69 vs 880.69 Hz (identical)
    //   full q4_k  cos 0.999933   f0 880.87 vs 880.69 Hz
    // All four are usable; q4_k tiny at 0.26 MB is the mobile pick.
    {"crepe-tiny-q8_0", "crepe-tiny-q8_0.gguf",
     "https://huggingface.co/cstr/crepe-GGUF/resolve/main/crepe-tiny-q8_0.gguf", "~0.5 MB", nullptr, nullptr, nullptr,
     "MIT"},
    {"crepe-tiny-q4_k", "crepe-tiny-q4_k.gguf",
     "https://huggingface.co/cstr/crepe-GGUF/resolve/main/crepe-tiny-q4_k.gguf", "~0.3 MB", nullptr, nullptr, nullptr,
     "MIT"},
    {"crepe-full-q8_0", "crepe-full-q8_0.gguf",
     "https://huggingface.co/cstr/crepe-GGUF/resolve/main/crepe-full-q8_0.gguf", "~22.6 MB", nullptr, nullptr, nullptr,
     "MIT"},
    {"crepe-full-q4_k", "crepe-full-q4_k.gguf",
     "https://huggingface.co/cstr/crepe-GGUF/resolve/main/crepe-full-q4_k.gguf", "~12.0 MB", nullptr, nullptr, nullptr,
     "MIT"},
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
     "~83 MB", nullptr, nullptr, nullptr, "CC-BY-4.0 (see https://huggingface.co/nvidia/stt_en_fastconformer_ctc_large)"},
    // FastConformer-CTC models double as compact CTC forced aligners
    // (-am <name>): the GGUF arch tag is canary-ctc, so crispasr_aligner's
    // default dispatch loads them directly. The *-aligner-* aliases below
    // mirror the wav2vec2-aligner-<lang> naming for discoverability.
    {"fastconformer-aligner", "stt-en-fastconformer-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-en-fastconformer-ctc-large-GGUF/resolve/main/stt-en-fastconformer-ctc-large-q4_k.gguf",
     "~83 MB", nullptr, nullptr},
    {"fastconformer-aligner-en", "stt-en-fastconformer-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-en-fastconformer-ctc-large-GGUF/resolve/main/stt-en-fastconformer-ctc-large-q4_k.gguf",
     "~83 MB", nullptr, nullptr},
    // CTC branches of the nvidia/stt_*_fastconformer_hybrid_large[_pc] fleet
    // (all CC-BY-4.0; pt is excluded upstream-NC): per-language ASR + forced
    // alignment, ~82 MB q4_k each. _pc variants add punctuation/capitalisation
    // (fa and kk-ru are non-pc). en-pc complements the uncased en standalone.
    {"fastconformer-ctc-en-pc", "stt-en-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-en-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-en-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-aligner-en-pc", "stt-en-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-en-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-en-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-ctc-es", "stt-es-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-es-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-es-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-aligner-es", "stt-es-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-es-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-es-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-ctc-fr", "stt-fr-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-fr-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-fr-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-aligner-fr", "stt-fr-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-fr-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-fr-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-ctc-it", "stt-it-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-it-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-it-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-aligner-it", "stt-it-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-it-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-it-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-ctc-nl", "stt-nl-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-nl-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-nl-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-aligner-nl", "stt-nl-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-nl-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-nl-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-ctc-pl", "stt-pl-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-pl-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-pl-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-aligner-pl", "stt-pl-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-pl-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-pl-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-ctc-ru", "stt-ru-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-ru-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-ru-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-aligner-ru", "stt-ru-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-ru-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-ru-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-ctc-ua", "stt-ua-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-ua-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-ua-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-aligner-ua", "stt-ua-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-ua-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-ua-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-ctc-hr", "stt-hr-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-hr-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-hr-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-aligner-hr", "stt-hr-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-hr-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-hr-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-ctc-be", "stt-be-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-be-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-be-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-aligner-be", "stt-be-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-be-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-be-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-ctc-ar", "stt-ar-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-ar-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-ar-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-aligner-ar", "stt-ar-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-ar-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-ar-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-ctc-fa", "stt-fa-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-fa-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-fa-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-aligner-fa", "stt-fa-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-fa-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-fa-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-ctc-ka", "stt-ka-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-ka-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-ka-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-aligner-ka", "stt-ka-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-ka-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-ka-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-ctc-hy", "stt-hy-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-hy-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-hy-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-aligner-hy", "stt-hy-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-hy-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-hy-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-ctc-uz", "stt-uz-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-uz-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-uz-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-aligner-uz", "stt-uz-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-uz-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-uz-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-ctc-kk-ru", "stt-kk-ru-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-kk-ru-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-kk-ru-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    {"fastconformer-aligner-kk-ru", "stt-kk-ru-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-kk-ru-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-kk-ru-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~82 MB", nullptr, nullptr},
    // CTC branch of nvidia/stt_de_fastconformer_hybrid_large_pc (CC-BY-4.0):
    // German ASR + forced alignment with punctuation/capitalisation.
    {"fastconformer-ctc-de", "stt-de-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-de-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-de-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~78 MB", nullptr, nullptr},
    {"fastconformer-aligner-de", "stt-de-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-de-fastconformer-hybrid-ctc-large-GGUF/resolve/main/stt-de-fastconformer-hybrid-ctc-large-q4_k.gguf",
     "~78 MB", nullptr, nullptr},
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
    // parakeet-ja: Q8_0 is the auto-download default — its TDT output is
    // byte-identical to F16 in our tests at half the size. Q4_K of this
    // model is quantisation-sensitive (joint.pred / decoder.embed
    // dimensions fall back to q4_0 inside q4_k mode) and the TDT decode
    // enters a fixed-point repetition loop; CTC decode over the same q4_k
    // file is clean (--parakeet-decoder ctc). All files at the repo carry
    // the hybrid model's CTC head since 2026-07 (#89 / #221d).
    {"parakeet-ja", "parakeet-tdt-0.6b-ja-q8_0.gguf",
     "https://huggingface.co/cstr/parakeet-tdt-0.6b-ja-GGUF/resolve/main/parakeet-tdt-0.6b-ja-q8_0.gguf",
     "~710 MB", nullptr, nullptr},
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
    // MOSS-TTS-v1.5 (MossTTSDelay): Qwen3-8B backbone + 32 RVQ audio codebooks
    // + a 1.6B transformer codec companion (kept F16 — quantising the codec
    // degrades audio earlier than the backbone). Default download is the Q4_K
    // backbone (~5 GB) paired with the F16 codec. (URLs populated at ship after
    // the GGUF upload — see Phase 6.)
    // MioCodec v2 — standalone audio codec (44.1kHz, 25Hz tokens, MIT).
    {"miocodec", "miocodec-v2-44k-q8_0.gguf",
     "https://huggingface.co/cstr/miocodec-v2-44k-GGUF/resolve/main/miocodec-v2-44k-q8_0.gguf",
     "~155 MB"},
    // MioTTS-0.6B (Qwen3 LLM + MioCodec-25Hz-24kHz, Apache-2.0).
    // Single GGUF, tokenizer.json loaded at runtime.
    {"miotts", "miotts-0.6b-q8_0.gguf",
     "https://huggingface.co/cstr/miotts-0.6b-GGUF/resolve/main/miotts-0.6b-q8_0.gguf",
     "~723 MB"},
    {"piano-transcription", "piano-transcription-f16.gguf",
     "https://huggingface.co/cstr/piano-transcription-GGUF/resolve/main/piano-transcription-f16.gguf",
     "~77 MB"},
    // Basic Pitch (Spotify, ICASSP 2022, Apache-2.0): polyphonic note events.
    // Tiny — the CQT kernels are most of the file. NOTE: the HF repo is not
    // uploaded yet, so auto-download will 404; build the GGUF locally with
    // models/convert-basic-pitch-to-gguf.py against the nmp.onnx that ships
    // inside the basic-pitch package.
    {"basic-pitch", "basic-pitch-f16.gguf",
     "https://huggingface.co/cstr/basic-pitch-GGUF/resolve/main/basic-pitch-f16.gguf",
     "~110 KB"},
    // Magenta MT3 (ISMIR 2021 / ICLR 2022, Apache-2.0): multi-instrument
    // transcription — every note carries a General-MIDI program. NOTE: the HF
    // repo is not uploaded yet, so auto-download will 404; build the GGUF
    // locally with models/convert-mt3-to-gguf.py against the T5X checkpoint at
    // gs://mt3/checkpoints/mt3.
    {"mt3", "mt3-f16.gguf",
     "https://huggingface.co/cstr/mt3-GGUF/resolve/main/mt3-f16.gguf",
     "~96 MB"},
    {"moss-tts", "moss-tts-v1.5-q4_k.gguf",
     "https://huggingface.co/cstr/moss-tts-v1.5-GGUF/resolve/main/moss-tts-v1.5-q4_k.gguf",
     "~5 GB",
     "moss-tts-v1.5-codec.gguf",
     "https://huggingface.co/cstr/moss-tts-v1.5-GGUF/resolve/main/moss-tts-v1.5-codec.gguf",
     "~3.4 GB"},
    // MOSS-TTS-Local-Transformer-v1.5 (4B, arch "moss-tts-local"): Qwen3-4B
    // backbone + MOSS-Audio-Tokenizer-v2 codec (48 kHz stereo, decode-only
    // companion). Apache-2.0. Default = F16 backbone: the acceptance target
    // (P5 round-trip overlap 0.969). Q4_K exists but its long trajectory runs away
    // (intrinsic quantized-AR drift) — F16 is the reliable config, fits a 16 GB GPU.
    {"moss-tts-local", "moss-tts-local-v1.5-f16.gguf",
     "https://huggingface.co/cstr/moss-tts-local-v1.5-GGUF/resolve/main/moss-tts-local-v1.5-f16.gguf",
     "~9.1 GB",
     "moss-tts-local-v1.5-codec.gguf",
     "https://huggingface.co/cstr/moss-tts-local-v1.5-GGUF/resolve/main/moss-tts-local-v1.5-codec.gguf",
     "~2.1 GB"},
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
    // OmniVoice: k2-fsa/OmniVoice — Qwen3-0.6B backbone with masked
    // iterative multi-codebook TTS (8 codebooks × 1025 vocab, 600+
    // languages). Uses HiggsAudioV2 audio tokenizer for encode/decode.
    // cstr/omnivoice-GGUF also hosts base quants omnivoice-{q8_0,q4_k}.gguf and a
    // smaller TTS-ONLY omnivoice-tokenizer-q8_0.gguf (320 MB) — the Q8 tokenizer
    // decodes fine but voice-clone ENCODE needs F16 (RVQ nearest-neighbor is
    // quant-sensitive), so the default tokenizer stays f16.
    {"omnivoice", "omnivoice-f16.gguf",
     "https://huggingface.co/cstr/omnivoice-GGUF/resolve/main/omnivoice-f16.gguf",
     "~1.2 GB",
     "omnivoice-tokenizer-f16.gguf",
     "https://huggingface.co/cstr/omnivoice-GGUF/resolve/main/omnivoice-tokenizer-f16.gguf",
     "~400 MB"},
    //
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
     "~250 MB"},
    {"tada-tts-1b", "tada-tts-1b-q4_k.gguf",
     "https://huggingface.co/cstr/tada-tts-1b-GGUF/resolve/main/tada-tts-1b-q4_k.gguf",
     "~1.7 GB",
     "tada-codec-f16.gguf",
     "https://huggingface.co/cstr/tada-tts-1b-GGUF/resolve/main/tada-codec-f16.gguf",
     "~250 MB"},
    {"tada-1b", "tada-tts-1b-f16.gguf",
     "https://huggingface.co/cstr/tada-tts-1b-GGUF/resolve/main/tada-tts-1b-f16.gguf",
     "~3.1 GB",
     "tada-codec-f16.gguf",
     "https://huggingface.co/cstr/tada-tts-1b-GGUF/resolve/main/tada-codec-f16.gguf",
     "~250 MB"},
    // TADA-3B-ML (HumeAI/tada-3b-ml): Llama-3.2-3B + flow matching + TADA codec.
    {"tada", "tada-tts-3b-ml-f16.gguf",
     "https://huggingface.co/cstr/tada-tts-3b-ml-GGUF/resolve/main/tada-tts-3b-ml-f16.gguf",
     "~6.6 GB",
     "tada-codec-f16.gguf",
     "https://huggingface.co/cstr/tada-tts-3b-ml-GGUF/resolve/main/tada-codec-f16.gguf",
     "~250 MB"},
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
    {"chatterbox", "chatterbox-v3-t3-q8_0.gguf",
     "https://huggingface.co/cstr/chatterbox-GGUF/resolve/main/chatterbox-v3-t3-q8_0.gguf",
     "~610 MB",
     "chatterbox-v3-s3gen-q8_0.gguf",
     "https://huggingface.co/cstr/chatterbox-GGUF/resolve/main/chatterbox-v3-s3gen-q8_0.gguf",
     "~349 MB"},
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
    // Chatterbox-Nano (#382): upstream's GPT2_small T3 (12L/768) on the
    // SAME meanflow S3Gen as Turbo — s3gen_meanflow/ve/conds in
    // ResembleAI/chatterbox-nano are byte-identical (same LFS oids) to
    // chatterbox-turbo, so the companion points at the Turbo repo.
    // Runtime path is the existing `chatterbox_turbo` GPT-2 branch; the
    // GGUF carries n_kv_heads=12 explicitly (C++ default is 16).
    {"chatterbox-nano", "chatterbox-nano-t3-q8_0.gguf",
     "https://huggingface.co/cstr/chatterbox-nano-GGUF/resolve/main/chatterbox-nano-t3-q8_0.gguf",
     "~345 MB",
     "chatterbox-turbo-s3gen-q8_0.gguf",
     "https://huggingface.co/cstr/chatterbox-turbo-GGUF/resolve/main/chatterbox-turbo-s3gen-q8_0.gguf",
     "~627 MB"},
    // Finnish-only Nano fine-tune, published and live-tested by the model
    // author. It uses the unchanged Turbo MeanFlow S3Gen companion.
    {"chatterbox-finnish-nano", "chatterbox-finnish-nano-v0.1.3-t3-q8_0.gguf",
     "https://huggingface.co/JJarvinen/chatterbox-finnish-nano-GGUF/resolve/main/chatterbox-finnish-nano-v0.1.3-t3-q8_0.gguf",
     "~329 MB",
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
    // Confucius4-TTS (§377): NetEase Youdao zero-shot voice-cloning TTS.
    // Three-stage: GPT-2 T2S (beam-sample, baked LlamaTokenizer vocab) →
    // flow-matching DiT+WaveNet S2A (CAMPPlus style encoder baked under
    // campplus.*) → BigVGAN 22.05 kHz (extras). Voice cloning via
    // --voice ref.wav (+ CRISPASR_CONFUCIUS4_COND_DIR w2v-BERT features
    // until that encoder is native). 14 languages, Chinese LANGUAGE_TOKEN_MAP
    // prompts baked into the runtime.
    {"confucius4-tts", "confucius4-tts-t2s-q4_k.gguf",
     "https://huggingface.co/cstr/confucius4-tts-GGUF/resolve/main/confucius4-tts-t2s-q4_k.gguf",
     "~376 MB",
     "confucius4-tts-s2a-q4_k.gguf",
     "https://huggingface.co/cstr/confucius4-tts-GGUF/resolve/main/confucius4-tts-s2a-q4_k.gguf",
     "~135 MB", "Apache-2.0 (base netease-youdao/Confucius4-TTS)"},
    // Pocket TTS: Kyutai's 100M continuous-latent AR TTS (CC-BY-4.0 plus gated-use conditions).
    // Generates continuous 32-dim float vectors at 12.5 Hz via one-step LSD,
    // decoded by Mimi VAE to 24 kHz PCM. Single GGUF, no codec companion.
    // Voice cloning via --voice ref.wav (requires the -f16 variant with Mimi encoder).
    // The -novc variant lacks the encoder and produces near-silence without conditioning.
    {"pocket-tts", "pocket-tts-english-f16.gguf",
     "https://huggingface.co/cstr/pocket-tts-GGUF/resolve/main/pocket-tts-english-f16.gguf",
     "~220 MB"},
    {"pocket-tts-de", "pocket-tts-german-q8_0.gguf",
     "https://huggingface.co/cstr/pocket-tts-GGUF/resolve/main/pocket-tts-german-q8_0.gguf",
     "~124 MB"},
    {"pocket-tts-es", "pocket-tts-spanish-q8_0.gguf",
     "https://huggingface.co/cstr/pocket-tts-GGUF/resolve/main/pocket-tts-spanish-q8_0.gguf",
     "~124 MB"},
    {"pocket-tts-it", "pocket-tts-italian-q8_0.gguf",
     "https://huggingface.co/cstr/pocket-tts-GGUF/resolve/main/pocket-tts-italian-q8_0.gguf",
     "~124 MB"},
    {"pocket-tts-pt", "pocket-tts-portuguese-q8_0.gguf",
     "https://huggingface.co/cstr/pocket-tts-GGUF/resolve/main/pocket-tts-portuguese-q8_0.gguf",
     "~124 MB"},
    // French is upstream's 24-layer preview checkpoint; the other languages
    // use the distilled 6-layer releases.
    {"pocket-tts-fr", "pocket-tts-french_24l-q8_0.gguf",
     "https://huggingface.co/cstr/pocket-tts-GGUF/resolve/main/pocket-tts-french_24l-q8_0.gguf",
     "~365 MB"},
    // IndexTTS-1.5: GPT-2 AR mel-code generator + BigVGAN vocoder.
    // Voice cloning via Conformer+Perceiver conditioning on reference audio.
    // Two-file setup: GPT (mel codes) + BigVGAN (vocoder). Q8_0 recommended
    // (~870 MB total); F16 available for max quality (~2.4 GB).
    {"indextts", "indextts-gpt-q8_0.gguf",
     "https://huggingface.co/cstr/indextts-1.5-GGUF/resolve/main/indextts-gpt-q8_0.gguf",
     "~870 MB",
     "indextts-bigvgan.gguf",
     "https://huggingface.co/cstr/indextts-1.5-GGUF/resolve/main/indextts-bigvgan.gguf",
     "~112 MB", "Apache-2.0 (see https://huggingface.co/IndexTeam/IndexTTS-1.5)"},
    // OuteTTS 0.3 1B: OLMo-1B LLM + WavTokenizer single-codebook VQ-GAN.
    // 24 kHz output, CC-BY-NC-SA-4.0 model license. Voice cloning via --voice <speaker.json>.
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
    // Raon-OpenTTS 0.3B (#387-adj): KRAFTON's F5-TTS DiT + sbhifigan16k
    // HiFi-GAN, English zero-shot voice cloning. Rides the f5-tts runtime
    // (same DiT; norm_type=rmsnorm is dead metadata with post_norm=False).
    // Single GGUF carries DiT + HiFi-GAN + the shipped slaney mel fb/window.
    // TTS→ASR roundtrip validated on Kaggle (word overlap 0.90). The HiFi-GAN
    // vocoder runs through the shared GPU-capable core_hifigan graph (#387);
    // the DiT ODE loop is the remaining cost and runs on-device in a GPU build.
    {"raon", "raon-opentts-0.3b-f16.gguf",
     "https://huggingface.co/cstr/raon-opentts-0.3b-GGUF/resolve/main/raon-opentts-0.3b-f16.gguf",
     "~959 MB", nullptr, nullptr, nullptr,
     "CC-BY-NC-4.0 — NON-COMMERCIAL use only (KRAFTON/Raon-OpenTTS-0.3B, "
     "https://huggingface.co/KRAFTON/Raon-OpenTTS-0.3B)"},
    // Raon-OpenTTS 1B: larger DiT (dim=1408 depth=28 heads=24x64 ff_mult=4);
    // same sbhifigan16k mel + HiFi-GAN vocoder as the 0.3B. Roundtrip validated
    // on Kaggle. Use --backend raon-1b (or -m auto with this key).
    {"raon-1b", "raon-opentts-1b-f16.gguf",
     "https://huggingface.co/cstr/raon-opentts-1b-GGUF/resolve/main/raon-opentts-1b-f16.gguf",
     "~2.8 GB", nullptr, nullptr, nullptr,
     "CC-BY-NC-4.0 — NON-COMMERCIAL use only (KRAFTON/Raon-OpenTTS-1B, "
     "https://huggingface.co/KRAFTON/Raon-OpenTTS-1B)"},
    // Irodori-TTS v3 500M: RF-DiT flow-matching TTS with zero-shot voice
    // cloning via DAC-VAE latents. 48 kHz output, Japanese-focused.
    {"irodori-tts", "irodori-tts-500m-v3-q4_k.gguf",
     "https://huggingface.co/cstr/irodori-tts-GGUF/resolve/main/irodori-tts-500m-v3-q4_k.gguf",
     "~852 MB",
     "dacvae-ja-32dim-f16.gguf",
     "https://huggingface.co/cstr/irodori-tts-GGUF/resolve/main/dacvae-ja-32dim-f16.gguf",
     "~135 MB"},
    // Irodori-TTS v3 600M VoiceDesign: caption encoder for style/emotion
    // conditioning via text descriptions. Same codec companion.
    {"irodori-tts-voicedesign", "irodori-tts-600m-v3-voicedesign-q4_k.gguf",
     "https://huggingface.co/cstr/irodori-tts-voicedesign-GGUF/resolve/main/irodori-tts-600m-v3-voicedesign-q4_k.gguf",
     "~526 MB",
     "dacvae-ja-32dim-f16.gguf",
     "https://huggingface.co/cstr/irodori-tts-voicedesign-GGUF/resolve/main/dacvae-ja-32dim-f16.gguf",
     "~135 MB"},
    // BananaMind-TTS: Tacotron-lite + HiFi-GAN vocoder. Lightweight
    // (~50 MB) single-speaker TTS, EN and DE variants.
    {"bananamind-tts", "bananamind-tts-en-q8_0.gguf",
     "https://huggingface.co/cstr/bananamind-tts-GGUF/resolve/main/bananamind-tts-en-q8_0.gguf",
     "~38 MB", nullptr, nullptr},
    {"bananamind-tts-de", "bananamind-tts-de-q8_0.gguf",
     "https://huggingface.co/cstr/bananamind-tts-GGUF/resolve/main/bananamind-tts-de-q8_0.gguf",
     "~38 MB", nullptr, nullptr},
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
    // f16 build — exact HF translation parity (q8_0 has rare quant-floor
    // decode flips on borderline words). Same faithful SP-BPE tokenizer.
    {"m2m100-f16", "m2m100-418m-f16.gguf",
     "https://huggingface.co/cstr/m2m100-418m-GGUF/resolve/main/m2m100-418m-f16.gguf",
     "~980 MB", nullptr, nullptr},
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
    // Default voice: en_US-lessac-medium (~30 MB F16).
    {"piper", "piper-en_US-lessac-medium-f16.gguf",
     "https://huggingface.co/cstr/piper-en_US-lessac-medium-GGUF/resolve/main/piper-en_US-lessac-medium-f16.gguf",
     "~30 MB", nullptr, nullptr},
    // German voices from the consolidated piper-voices-GGUF repo
    {"piper", "piper-de_DE-thorsten-medium-f16.gguf",
     "https://huggingface.co/cstr/piper-voices-GGUF/resolve/main/piper-de_DE-thorsten-medium-f16.gguf",
     "~30 MB", nullptr, nullptr},
    {"piper", "piper-de_DE-thorsten-high-f16.gguf",
     "https://huggingface.co/cstr/piper-voices-GGUF/resolve/main/piper-de_DE-thorsten-high-f16.gguf",
     "~54 MB", nullptr, nullptr},
    {"piper", "piper-de_DE-kerstin-low-f16.gguf",
     "https://huggingface.co/cstr/piper-voices-GGUF/resolve/main/piper-de_DE-kerstin-low-f16.gguf",
     "~30 MB", nullptr, nullptr},

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
    // NOTE the `parler-tts-` prefix. The repo also held a `parler-mini-` set
    // with byte-identical tensors that was MISSING parler.tokenizer.is_bpe —
    // and parler_tts.cpp defaults that to false, i.e. Viterbi unigram instead
    // of BPE, so `-m auto` silently tokenized prompts with the wrong algorithm.
    // The stale set has been removed upstream; keep this name.
    {"parler-tts", "parler-tts-mini-v1.1-q8_0.gguf",
     "https://huggingface.co/cstr/parler-tts-mini-v1.1-GGUF/resolve/main/parler-tts-mini-v1.1-q8_0.gguf",
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
    // CC-BY-NC-4.0 (non-commercial). ~63 MB F16. Pick only if you need its
    // specific 176-label space and accept the non-commercial restriction.
    {"lid-fasttext176", "fasttext-lid176-f16.gguf",
     "https://huggingface.co/cstr/fasttext-lid176-GGUF/resolve/main/fasttext-lid176-f16.gguf",
     "~63 MB", nullptr, nullptr},

    // Audio-LID family — speech-signal language identification.
    // Silero + Ecapa run through the module-level `detect_language_pcm`;
    // FireRed requires the session-level `Session::detect_language` (Phase 6).
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
    // Same engine, upstream's OTHER talker: llm.rl.pt, reinforcement-learning
    // tuned for speech quality, pronunciation accuracy and generation
    // stability. Only the LLM differs — flow / HiFT / CAMPPlus / s3tok /
    // voices are the shared companions below, so `--backend cosyvoice3-tts-rl
    // -m auto` swaps one 384 MB file (#334).
    {"cosyvoice3-tts-rl", "cosyvoice3-llm-rl-q4_k.gguf",
     "https://huggingface.co/cstr/cosyvoice3-0.5b-2512-GGUF/resolve/main/cosyvoice3-llm-rl-q4_k.gguf",
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

// dots.tts: the CAM++ speaker encoder (15 MB) rides along so that `--voice`
// voice cloning works after `-m auto --auto-download` — discover_speaker()
// finds it as a sibling of the core model. Negligible vs the 4.6 GB core.
constexpr ExtraCompanion k_dots_tts_extras[] = {
    {"dots-tts-soar-spk-f16.gguf",
     "https://huggingface.co/cstr/dots-tts-soar-GGUF/resolve/main/dots-tts-soar-spk-f16.gguf"},
    {nullptr, nullptr},
};

// Confucius4-TTS: the BigVGAN vocoder rides along; the CLI adapter discovers
// it as a sibling (confucius4-tts-bigvgan-22k-f16.gguf) next to the T2S.
constexpr ExtraCompanion k_confucius4_tts_extras[] = {
    {"confucius4-tts-bigvgan-22k-f16.gguf",
     "https://huggingface.co/cstr/confucius4-tts-GGUF/resolve/main/confucius4-tts-bigvgan-22k-f16.gguf"},
    // encoder-only w2v-BERT 2.0 (17L): fully native --voice conditioning
    {"confucius4-tts-w2v-f16.gguf",
     "https://huggingface.co/cstr/confucius4-tts-GGUF/resolve/main/confucius4-tts-w2v-f16.gguf"},
    {nullptr, nullptr},
};

constexpr ExtraList k_extras[] = {
    {"kokoro", k_kokoro_extras},
    {"dots-tts", k_dots_tts_extras},
    {"confucius4-tts", k_confucius4_tts_extras},
    {"vibevoice-tts", k_vibevoice_tts_extras},
    {"cosyvoice3-tts", k_cosyvoice3_tts_extras},
    {"cosyvoice3-tts-rl", k_cosyvoice3_tts_extras},
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

static std::string effective_license(const Entry& e);

void fill(CrispasrRegistryEntry& out, const Entry& e, const std::string& preferred_quant) {
    const std::string filename = apply_quant_to_filename(e.filename, preferred_quant);
    out.backend = e.backend;
    out.filename = filename;
    out.url = replace_tail_filename(e.url, e.filename, filename);
    // `approx_size` describes the artifact the registry row NAMES. Once
    // --model-quant has rewritten the filename to a different quant, that
    // number no longer describes what would be downloaded — reporting the
    // q4_k size for an f16 fetch understated a 4.7 GB download as "~1.3 GB"
    // (#393). The per-quant sizes are not in the registry, and a ratio-scaled
    // guess would be an invention (this quantizer keeps norms and embeddings
    // at F16, so the ratio is model-dependent), so say nothing rather than
    // something false; every consumer treats an empty size as "unknown".
    out.approx_size = (filename == e.filename && e.approx_size) ? e.approx_size : "";
    if (e.companion_file && e.companion_url) {
        out.companion_filename = apply_quant_to_filename(e.companion_file, preferred_quant);
        out.companion_url = replace_tail_filename(e.companion_url, e.companion_file, out.companion_filename);
        out.companion_approx_size = (out.companion_filename == e.companion_file)
                                        ? (e.companion_size ? e.companion_size : (e.approx_size ? e.approx_size : ""))
                                        : ""; // same reasoning as approx_size above
    } else {
        out.companion_filename.clear();
        out.companion_url.clear();
        out.companion_approx_size.clear();
    }
    out.license = effective_license(e);
}

// A few families have many quant/alias rows. Keep their license metadata
// attached to the family as a defensive backstop, so a newly added quant row
// cannot silently become an unlicensed download merely because its author
// forgot the final Entry field. Explicit Entry metadata always wins.
static std::string effective_license(const Entry& e) {
    if (e.license)
        return e.license;
    const std::string b = e.backend ? e.backend : "";
    if (b == "parakeet" || b == "parakeet-ja" || b.rfind("parakeet-", 0) == 0 || b == "canary" || b == "canary-qwen" ||
        b.rfind("fastconformer-", 0) == 0)
        return "CC-BY-4.0 (see the original NVIDIA model card)";
    if (b == "fastpitch")
        return "CC-BY-4.0 (see the original model card)";
    if (b == "outetts")
        return "CC-BY-NC-SA-4.0 (see https://huggingface.co/OuteAI/OuteTTS-0.3-1B)";
    if (b == "pocket-tts" || b.rfind("pocket-tts-", 0) == 0)
        return "pocket-tts-terms (CC-BY-4.0 plus gated-use conditions; see https://huggingface.co/kyutai/pocket-tts)";
    if (b == "lid-fasttext176")
        return "CC-BY-NC-4.0 (see https://huggingface.co/facebook/fasttext-language-identification)";
    if (b == "gemma4-e2b" || b == "gemma4-e4b")
        return "gemma-terms (see https://ai.google.dev/gemma/terms)";
    if (b == "tada" || b == "tada-1b" || b == "tada-tts-1b")
        return "llama3.2 (see https://huggingface.co/HumeAI/tada-3b-ml/blob/main/LICENSE)";
    if (b.rfind("orpheus", 0) == 0)
        return "llama3.2 (see https://huggingface.co/unsloth/orpheus-3b-0.1-ft)";
    if (b.rfind("lex-au-orpheus", 0) == 0 || b.rfind("kartoffel-orpheus", 0) == 0 || b.rfind("kartoffelbox", 0) == 0)
        return "llama3.2 (see the upstream Orpheus model card)";
    if (b == "lfm2-audio")
        return "lfm1.0 (commercial use threshold and attribution; see "
               "https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B)";
    if (b == "funasr" || b == "fun-asr-mlt-nano" || b == "sensevoice" || b == "paraformer")
        return "funasr-v1.1 (attribution required; see the original FunAudioLLM model card)";
    return "";
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

// ---------------------------------------------------------------------------
// Restricted-licence acceptance gate.
//
// Mirrors CrispEmbed's examples/cli/model_mgr.{h,cpp} so the two repos behave
// identically: acceptance is per-licence (an exact SPDX tag, or "all"), keyed
// off a tag rather than a substring, and `allow_download` alone is NEVER
// sufficient for a restricted model.
//
// Registry entries carry human prose today (e.g. "CC-BY-NC-4.0 — NON-COMMERCIAL
// use only (base model ...)"), so normalise to the leading SPDX-ish token first
// rather than rewriting every entry. New entries should use a bare tag.
// ---------------------------------------------------------------------------

static std::string license_tag(const std::string& lic) {
    std::string t;
    for (char c : lic) {
        if (c == ' ' || c == '\t' || c == '(' || c == ',')
            break;
        if ((unsigned char)c >= 0x80) // em-dash and friends
            break;
        t += (char)tolower((unsigned char)c);
    }
    while (!t.empty() && (t.back() == '-' || t.back() == '.'))
        t.pop_back();
    return t;
}

// True when the tag designates a licence the user must explicitly accept.
// Same list as CrispEmbed, so a future gemma/llama model is gated on arrival
// instead of shipping ungated.
static bool license_requires_acceptance_tag(const std::string& tag) {
    if (tag.rfind("cc-by-nc", 0) == 0)
        return true;
    if (tag.rfind("cc-by-sa", 0) == 0)
        return true;
    if (tag.rfind("llama", 0) == 0)
        return true;
    static const char* restricted[] = {"gemma",  "gemma-terms",  "qwen-research", "mistral-ai-research",
                                       "lfm1.0", "lfm-open-1.0", "funasr-v1.1",   "pocket-tts-terms",
                                       "other",  nullptr};
    for (const char** p = restricted; *p; ++p)
        if (tag == *p)
            return true;
    return false;
}

static bool license_is_nc(const std::string& lic) {
    return license_requires_acceptance_tag(license_tag(lic));
}

// Has the user accepted this specific licence? Exact tag, or "all" / "*".
// Checked against the caller-supplied string, then the process-level setting,
// then CRISPASR_ACCEPT_LICENSE.
static bool license_accepted(const std::string& lic, const std::string& accepted_arg) {
    const std::string tag = license_tag(lic);
    auto matches = [&](const std::string& acc) {
        if (acc.empty())
            return false;
        if (acc == "all" || acc == "*")
            return true;
        return license_tag(acc) == tag;
    };
    if (matches(accepted_arg))
        return true;
    if (const char* env = std::getenv("CRISPASR_ACCEPT_LICENSE"))
        if (matches(env))
            return true;
    return false;
}

// Gate a restricted model BEFORE any bytes are fetched. Returns true to
// proceed. On a TTY the user is shown the licence and prompted; otherwise the
// download is refused with instructions. `allow_download` alone is NOT enough.
static bool license_gate_allows_download(const CrispasrRegistryEntry& e, const std::string& accepted_license) {
    if (e.license.empty() || !license_is_nc(e.license))
        return true;
    if (license_accepted(e.license, accepted_license))
        return true;

    const std::string tag = license_tag(e.license);
    fprintf(stderr, "crispasr: model '%s' is released under a restricted licence:\n", e.filename.c_str());
    fprintf(stderr, "  Licence: %s\n", e.license.c_str());
    if (tag.rfind("cc-by-nc", 0) == 0)
        fprintf(stderr, "  Notice:  NON-COMMERCIAL USE ONLY — see the upstream model card for terms.\n");
    else
        fprintf(stderr, "  Notice:  review the upstream model card for the full licence terms.\n");

    fprintf(stderr,
            "crispasr: refusing to download without explicit licence acceptance.\n"
            "  Pass --accept-license %s (or set CRISPASR_ACCEPT_LICENSE=%s).\n"
            "  --auto-download / -m auto alone is NOT sufficient for a restricted licence.\n",
            tag.c_str(), tag.c_str());
    return false;
}

void print_license_note(const CrispasrRegistryEntry& e, bool quiet) {
    if (!quiet && !e.license.empty()) {
        if (license_tag(e.license).rfind("cc-by-nc", 0) == 0) {
            fprintf(stderr,
                    "crispasr: WARNING: %s is licensed %s — NON-COMMERCIAL USE ONLY.\n"
                    "  By loading this model you confirm you will not use it for commercial purposes.\n",
                    e.filename.c_str(), e.license.c_str());
        } else if (license_is_nc(e.license)) {
            fprintf(stderr, "crispasr: WARNING: %s is licensed %s — review and comply with the upstream terms.\n",
                    e.filename.c_str(), e.license.c_str());
        } else {
            fprintf(stderr, "crispasr: note: %s is licensed %s\n", e.filename.c_str(), e.license.c_str());
        }
    }
}

} // namespace

std::string crispasr_license_tag(const std::string& license) {
    return license_tag(license);
}

bool crispasr_license_requires_acceptance(const std::string& license) {
    return license_is_nc(license);
}

bool crispasr_license_accepted(const std::string& license, const std::string& accepted) {
    return license_accepted(license, accepted);
}

std::string crispasr_managed_download(const std::string& filename, const std::string& url, const std::string& license,
                                      bool quiet, const char* label, const std::string& cache_dir_override,
                                      const std::string& accepted_license) {
    CrispasrRegistryEntry artifact;
    artifact.filename = filename;
    artifact.url = url;
    artifact.license = license;
    if (!license_gate_allows_download(artifact, accepted_license))
        return {};
    print_license_note(artifact, quiet);
    return crispasr_cache::ensure_cached_file(filename, url, quiet, label, cache_dir_override);
}

bool crispasr_registry_lookup(const std::string& backend, CrispasrRegistryEntry& out,
                              const std::string& preferred_quant) {
    const Entry* e = find_by_backend(backend);
    if (!e)
        return false;
    fill(out, *e, preferred_quant);
    return true;
}

bool crispasr_registry_default_bundle(const std::string& backend, CrispasrRegistryBundle& out) {
    const Entry* e = find_by_backend(backend);
    if (!e)
        return false;

    out = {};
    out.backend = e->backend;
    out.license = effective_license(*e);
    out.requires_license_acceptance = crispasr_license_requires_acceptance(out.license);
    out.artifacts.push_back(
        {CrispasrRegistryArtifactKind::Primary, e->filename, e->url, e->approx_size ? e->approx_size : ""});

    if (e->companion_file && e->companion_url) {
        out.artifacts.push_back({CrispasrRegistryArtifactKind::Companion, e->companion_file, e->companion_url,
                                 e->companion_size ? e->companion_size : (e->approx_size ? e->approx_size : "")});
    }

    if (const ExtraCompanion* extras = find_extras(e->backend)) {
        for (const ExtraCompanion* it = extras; it->file && it->url; ++it)
            out.artifacts.push_back({CrispasrRegistryArtifactKind::Extra, it->file, it->url, ""});
    }
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
            out.license = effective_license(row);
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
                                   const std::string& preferred_quant, const std::string& accepted_license) {
    // Concrete path that exists on disk — pass through.
    if (model_arg != "auto" && model_arg != "default") {
        FILE* f = fopen(model_arg.c_str(), "rb");
        if (f) {
            fclose(f);
            return model_arg;
        }
        // A bare explicit filename is also allowed to live in the configured
        // model roots.  Preserve its exact identity: do this before registry
        // matching, whose backend fallback may name a different voice (#397).
        if (model_arg.find_first_of("/\\") == std::string::npos) {
            const std::string cached_explicit = crispasr_cache::probe_cached_file(model_arg, cache_dir_override);
            if (!cached_explicit.empty())
                return cached_explicit;
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
            if (crispasr_cache::file_present(cached)) {
                // A previously-downloaded restricted model used to load in
                // total silence — state the licence on every load, not just
                // the one where it happened to be fetched.
                print_license_note(match, quiet);
                return cached;
            }
        }

        if (have_match && allow_download) {
            // Restricted-licence gate BEFORE any bytes are fetched.
            if (!license_gate_allows_download(match, accepted_license))
                return "";
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

    // Restricted-licence gate BEFORE any bytes are fetched. Skipped when the
    // file is already cached (acceptance happened at download time).
    if (!crispasr_cache::file_present(crispasr_cache::dir(cache_dir_override) + "/" + e.filename) &&
        !license_gate_allows_download(e, accepted_license))
        return "";

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
