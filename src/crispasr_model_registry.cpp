// crispasr_model_registry.cpp — implementation. See the header.

#include "crispasr_model_registry.h"
#include "crispasr_cache.h"

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
    {"parakeet", "parakeet-tdt-0.6b-v3-q4_k.gguf",
     "https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF/resolve/main/parakeet-tdt-0.6b-v3-q4_k.gguf", "~467 MB", nullptr, nullptr},
    {"canary", "canary-1b-v2-q4_k.gguf",
     "https://huggingface.co/cstr/canary-1b-v2-GGUF/resolve/main/canary-1b-v2-q4_k.gguf", "~600 MB", nullptr, nullptr},
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
    {"cohere", "cohere-transcribe-q4_k.gguf",
     "https://huggingface.co/cstr/cohere-transcribe-03-2026-GGUF/resolve/main/cohere-transcribe-q4_k.gguf", "~550 MB", nullptr, nullptr},
    {"wav2vec2", "wav2vec2-xlsr-en-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-large-xlsr-53-english-GGUF/resolve/main/wav2vec2-xlsr-en-q4_k.gguf",
     "~212 MB", nullptr, nullptr},
    {"mimo-asr", "mimo-asr-q4_k.gguf",
     "https://huggingface.co/cstr/mimo-asr-GGUF/resolve/main/mimo-asr-q4_k.gguf", "~4.2 GB", nullptr, nullptr},
    {"omniasr", "omniasr-ctc-1b-v2-q4_k.gguf",
     "https://huggingface.co/cstr/omniASR-CTC-1B-v2-GGUF/resolve/main/omniasr-ctc-1b-v2-q4_k.gguf", "~580 MB", nullptr, nullptr},
    {"omniasr-llm", "omniasr-llm-300m-v2-q4_k.gguf",
     "https://huggingface.co/cstr/omniasr-llm-300m-v2-GGUF/resolve/main/omniasr-llm-300m-v2-q4_k.gguf", "~580 MB", nullptr, nullptr},
    {"hubert", "hubert-large-ls960-ft-q4_k.gguf",
     "https://huggingface.co/cstr/hubert-large-ls960-ft-GGUF/resolve/main/hubert-large-ls960-ft-q4_k.gguf", "~200 MB", nullptr, nullptr},
    {"data2vec", "data2vec-audio-base-960h-q4_k.gguf",
     "https://huggingface.co/cstr/data2vec-audio-960h-GGUF/resolve/main/data2vec-audio-base-960h-q4_k.gguf", "~60 MB", nullptr, nullptr},
    {"vibevoice", "vibevoice-asr-q4_k.gguf",
     "https://huggingface.co/cstr/vibevoice-asr-GGUF/resolve/main/vibevoice-asr-q4_k.gguf", "~4.5 GB", nullptr, nullptr},
    {"vibevoice-tts", "vibevoice-realtime-0.5b-q4_k.gguf",
     "https://huggingface.co/cstr/vibevoice-realtime-0.5b-GGUF/resolve/main/vibevoice-realtime-0.5b-q4_k.gguf",
     "~636 MB",
     "vibevoice-voice-emma.gguf",
     "https://huggingface.co/cstr/vibevoice-realtime-0.5b-GGUF/resolve/main/vibevoice-voice-emma.gguf"},
    {"firered-asr", "firered-asr2-aed-q4_k.gguf",
     "https://huggingface.co/cstr/firered-asr2-aed-GGUF/resolve/main/firered-asr2-aed-q4_k.gguf", "~918 MB", nullptr, nullptr},
    {"kyutai-stt", "kyutai-stt-1b-q4_k.gguf",
     "https://huggingface.co/cstr/kyutai-stt-1b-GGUF/resolve/main/kyutai-stt-1b-q4_k.gguf", "~636 MB", nullptr, nullptr},
    {"glm-asr", "glm-asr-nano-q4_k.gguf",
     "https://huggingface.co/cstr/glm-asr-nano-GGUF/resolve/main/glm-asr-nano-q4_k.gguf", "~1.2 GB", nullptr, nullptr},
    {"moonshine", "moonshine-tiny-q4_k.gguf",
     "https://huggingface.co/cstr/moonshine-tiny-GGUF/resolve/main/moonshine-tiny-q4_k.gguf", "~20 MB",
     "tokenizer.bin", "https://huggingface.co/cstr/moonshine-tiny-GGUF/resolve/main/tokenizer.bin"},
    {"wav2vec2-de", "wav2vec2-large-xlsr-53-german-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-large-xlsr-53-german-GGUF/resolve/main/wav2vec2-large-xlsr-53-german-q4_k.gguf",
     "~222 MB", nullptr, nullptr},
    {"moonshine-streaming", "moonshine-streaming-tiny-q4_k.gguf",
     "https://huggingface.co/cstr/moonshine-streaming-tiny-GGUF/resolve/main/moonshine-streaming-tiny-q4_k.gguf", "~31 MB",
     "tokenizer.bin", "https://huggingface.co/cstr/moonshine-streaming-tiny-GGUF/resolve/main/tokenizer.bin"},
    {"fastconformer-ctc", "stt-en-fastconformer-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-en-fastconformer-ctc-large-GGUF/resolve/main/stt-en-fastconformer-ctc-large-q4_k.gguf",
     "~83 MB", nullptr, nullptr},
    {"gemma4-e2b", "gemma4-e2b-it-q4_k.gguf",
     "https://huggingface.co/cstr/gemma4-e2b-it-GGUF/resolve/main/gemma4-e2b-it-q4_k.gguf",
     "~2.5 GB", nullptr, nullptr},
    // parakeet-ja: F16 is the auto-download default — Q4_K of this
    // model is quantisation-sensitive (joint.pred / decoder.embed
    // dimensions fall back to q4_0 inside q4_k mode) and the talker
    // enters a fixed-point loop after ~8 tokens. The Q4_K file is
    // available at the same repo for users who pin disk space, but
    // we'd rather have correct output by default.
    {"parakeet-ja", "parakeet-tdt-0.6b-ja.gguf",
     "https://huggingface.co/cstr/parakeet-tdt-0.6b-ja-GGUF/resolve/main/parakeet-tdt-0.6b-ja.gguf",
     "~1.24 GB", nullptr, nullptr},
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
     "https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF/resolve/main/qwen3-tts-tokenizer-12hz.gguf"},
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
     "https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF/resolve/main/qwen3-tts-tokenizer-12hz.gguf"},
    // Qwen3-TTS-Base 1.7B: same ICL voice-clone path as 0.6B-Base
    // (`--voice <wav> --ref-text "..."`), with talker hidden=2048,
    // ECAPA enc_dim=2048, and a small_to_mtp_projection bridge to the
    // 1024-d code predictor. ~1.9 GB Q8_0 talker; reuses the 12 Hz
    // tokenizer.
    {"qwen3-tts-1.7b-base", "qwen3-tts-12hz-1.7b-base-q8_0.gguf",
     "https://huggingface.co/cstr/qwen3-tts-1.7b-base-GGUF/resolve/main/qwen3-tts-12hz-1.7b-base-q8_0.gguf",
     "~1.9 GB",
     "qwen3-tts-tokenizer-12hz.gguf",
     "https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF/resolve/main/qwen3-tts-tokenizer-12hz.gguf"},
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
     "https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF/resolve/main/qwen3-tts-tokenizer-12hz.gguf"},
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
     "https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF/resolve/main/qwen3-tts-tokenizer-12hz.gguf"},
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
    {"orpheus", "orpheus-3b-base-q8_0.gguf",
     "https://huggingface.co/cstr/orpheus-3b-base-GGUF/resolve/main/orpheus-3b-base-q8_0.gguf",
     "~3.5 GB",
     "snac-24khz.gguf",
     "https://huggingface.co/cstr/snac-24khz-GGUF/resolve/main/snac-24khz.gguf"},
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
     "https://huggingface.co/cstr/snac-24khz-GGUF/resolve/main/snac-24khz.gguf"},
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
     "https://huggingface.co/cstr/snac-24khz-GGUF/resolve/main/snac-24khz.gguf"},
    {"kartoffel-orpheus-de-synthetic", "kartoffel-orpheus-de-synthetic-q8_0.gguf",
     "https://huggingface.co/cstr/kartoffel-orpheus-3b-german-synthetic-GGUF/resolve/main/kartoffel-orpheus-de-synthetic-q8_0.gguf",
     "~3.5 GB",
     "snac-24khz.gguf",
     "https://huggingface.co/cstr/snac-24khz-GGUF/resolve/main/snac-24khz.gguf"},
    // Chatterbox family — ResembleAI MIT TTS. Two-GGUF runtime:
    //   primary  = T3 (text → speech tokens) — also carries baked conds
    //   companion = S3Gen (tokens → 24 kHz waveform via CFM + HiFTGenerator)
    // The runtime is exposed only through the C ABI today
    // (`chatterbox_init_from_file` + `chatterbox_load_s3gen`); the
    // crispasr CLI adapter is still pending — see PLAN. Auto-download
    // via `-m auto --backend chatterbox` populates both files into the
    // cache so the C-API caller can resolve them.
    {"chatterbox", "chatterbox-t3-q8_0.gguf",
     "https://huggingface.co/cstr/chatterbox-GGUF/resolve/main/chatterbox-t3-q8_0.gguf",
     "~880 MB",
     "chatterbox-s3gen-q8_0.gguf",
     "https://huggingface.co/cstr/chatterbox-GGUF/resolve/main/chatterbox-s3gen-q8_0.gguf"},
    // Chatterbox-Turbo: distilled GPT-2 T3 (24L) + 2-step meanflow S3Gen.
    // Different architecture from base — runtime keys off
    // `chatterbox.t3.arch` ("kartoffelbox" branch handles GPT-2 T3 for
    // both Turbo and the German Kartoffelbox fine-tune). Only F16
    // published upstream.
    {"chatterbox-turbo", "chatterbox-turbo-t3-f16.gguf",
     "https://huggingface.co/cstr/chatterbox-turbo-GGUF/resolve/main/chatterbox-turbo-t3-f16.gguf",
     "~1.6 GB",
     "chatterbox-turbo-s3gen-f16.gguf",
     "https://huggingface.co/cstr/chatterbox-turbo-GGUF/resolve/main/chatterbox-turbo-s3gen-f16.gguf"},
    // Kartoffelbox-Turbo: SebastianBodza's German fine-tune of
    // chatterbox-turbo. Same GPT-2 T3 arch as Turbo; reuses the
    // chatterbox-turbo S3Gen verbatim (companion points at the Turbo
    // repo on purpose — saves a redundant 627 MB upload). T3 shipped at
    // F16/Q8_0/Q4_K; Q8_0 is the recommended deployment quant.
    {"kartoffelbox-turbo", "kartoffelbox-turbo-t3-q8_0.gguf",
     "https://huggingface.co/cstr/kartoffelbox-turbo-GGUF/resolve/main/kartoffelbox-turbo-t3-q8_0.gguf",
     "~1.25 GB",
     "chatterbox-turbo-s3gen-f16.gguf",
     "https://huggingface.co/cstr/chatterbox-turbo-GGUF/resolve/main/chatterbox-turbo-s3gen-f16.gguf"},
    // Lahgtna-chatterbox-v1: oddadmix's Arabic T3 fine-tune of base
    // ResembleAI/chatterbox. Shares the base Llama T3 architecture
    // (default converter path, no `--variant`); reuses the base S3Gen
    // verbatim (companion points at cstr/chatterbox-GGUF). T3 shipped
    // at F16 only.
    {"lahgtna-chatterbox", "chatterbox-t3-f16.gguf",
     "https://huggingface.co/cstr/lahgtna-chatterbox-v1-GGUF/resolve/main/chatterbox-t3-f16.gguf",
     "~1.4 GB",
     "chatterbox-s3gen-q8_0.gguf",
     "https://huggingface.co/cstr/chatterbox-GGUF/resolve/main/chatterbox-s3gen-q8_0.gguf"},
    // CTC forced aligner — used by `-am auto` to attach word-level
    // timestamps (LLM-decode backends, or any backend when paired
    // with `--force-aligner` / `-fa`). Q4_K is the recommended quant
    // (~442 MB; quality is indistinguishable from Q8_0 on word-onset
    // accuracy). Q5_0 / Q8_0 / F16 sit on the same repo.
    {"canary-ctc-aligner", "canary-ctc-aligner-q4_k.gguf",
     "https://huggingface.co/cstr/canary-ctc-aligner-GGUF/resolve/main/canary-ctc-aligner-q4_k.gguf",
     "~442 MB", nullptr, nullptr},
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
    // Status (2026-05-04): the t5_translate runtime is WIP — the
    // rel-pos bias path produces a repeating-token loop in decode
    // (see commit 1d9026c). The registry entry + adapter ship the
    // wiring; output quality is not yet correct. Track the upstream
    // T5 rel-pos debugging in PLAN.
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
     "https://huggingface.co/cstr/kokoro-voices-GGUF/resolve/main/kokoro-voice-af_heart.gguf"},
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

constexpr ExtraList k_extras[] = {
    {"kokoro", k_kokoro_extras},
    {"vibevoice-tts", k_vibevoice_tts_extras},
    {nullptr, nullptr},
};
// clang-format on

const Entry* find_by_backend(const std::string& backend) {
    for (const auto& e : k_registry)
        if (backend == e.backend)
            return &e;
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

void fill(CrispasrRegistryEntry& out, const Entry& e) {
    out.backend = e.backend;
    out.filename = e.filename;
    out.url = e.url;
    out.approx_size = e.approx_size;
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

} // namespace

bool crispasr_registry_lookup(const std::string& backend, CrispasrRegistryEntry& out) {
    const Entry* e = find_by_backend(backend);
    if (!e)
        return false;
    fill(out, *e);
    return true;
}

bool crispasr_registry_lookup_by_filename(const std::string& filename, CrispasrRegistryEntry& out) {
    const Entry* e = find_by_filename(filename);
    if (!e)
        return false;
    fill(out, *e);
    return true;
}

int crispasr_registry_count() {
    return (int)(sizeof(k_registry) / sizeof(k_registry[0]));
}

bool crispasr_registry_get_at(int i, CrispasrRegistryEntry& out) {
    if (i < 0 || i >= crispasr_registry_count())
        return false;
    fill(out, k_registry[i]);
    return true;
}

bool crispasr_find_cached_model(CrispasrRegistryEntry& out, const std::string& cache_dir_override) {
    // k_registry is already ordered whisper > parakeet > canary > ... —
    // first entry wins, which matches the documented preference.
    const std::string dir = crispasr_cache::dir(cache_dir_override);
    for (const auto& e : k_registry) {
        const std::string path = dir + "/" + e.filename;
        if (crispasr_cache::file_present(path)) {
            fill(out, e);
            return true;
        }
    }
    return false;
}

std::string crispasr_resolve_model(const std::string& model_arg, const std::string& backend_name, bool quiet,
                                   const std::string& cache_dir_override, bool allow_download) {
    // Concrete path that exists on disk — pass through.
    if (model_arg != "auto" && model_arg != "default") {
        FILE* f = fopen(model_arg.c_str(), "rb");
        if (f) {
            fclose(f);
            return model_arg;
        }

        // File not found — try registry-based download when permitted.
        const Entry* match = find_by_filename(model_arg);
        if (!match && !backend_name.empty())
            match = find_by_backend(backend_name);

        if (match && allow_download) {
            if (!quiet) {
                fprintf(stderr, "crispasr: model '%s' not found locally — downloading %s (%s)\n", model_arg.c_str(),
                        match->filename, match->approx_size);
            }
            std::string dl =
                crispasr_cache::ensure_cached_file(match->filename, match->url, quiet, "crispasr", cache_dir_override);
            if (!dl.empty() && match->companion_file && match->companion_url)
                crispasr_cache::ensure_cached_file(match->companion_file, match->companion_url, quiet, "crispasr",
                                                   cache_dir_override);
            if (!dl.empty())
                download_extras(*match, quiet, cache_dir_override);
            return dl;
        }
        // Either no registry match or caller didn't authorise download —
        // return the arg untouched so the caller can decide (prompt,
        // error, etc.).
        return model_arg;
    }

    const Entry* e = find_by_backend(backend_name);
    if (!e) {
        fprintf(stderr, "crispasr: -m auto not supported for backend '%s' (no default model registered)\n",
                backend_name.c_str());
        return "";
    }

    if (!quiet)
        fprintf(stderr, "crispasr: resolving %s (%s) via -m auto\n", e->filename, e->approx_size);
    std::string result = crispasr_cache::ensure_cached_file(e->filename, e->url, quiet, "crispasr", cache_dir_override);

    // Download companion file (e.g. tokenizer.bin for moonshine) if needed
    if (!result.empty() && e->companion_file && e->companion_url) {
        crispasr_cache::ensure_cached_file(e->companion_file, e->companion_url, quiet, "crispasr", cache_dir_override);
    }
    // Backend-specific extras (e.g. kokoro German backbone + voice) — opt-in
    // per backend via k_extras.
    if (!result.empty())
        download_extras(*e, quiet, cache_dir_override);

    return result;
}
