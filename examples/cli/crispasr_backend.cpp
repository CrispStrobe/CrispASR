// crispasr_backend.cpp — backend factory, auto-detection, and helpers.

#include "crispasr_backend.h"
#include "whisper_params.h"
#include "core/arch_backend_map.h" // #335: general.architecture → backend table, SHARED with the C ABI

// Forward declarations of per-backend constructors. Each is implemented in
// its own crispasr_backend_X.cpp file and compiled only if the backend's
// library is linked in.
std::unique_ptr<CrispasrBackend> crispasr_make_whisper_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_nemotron_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_gigaam_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_parakeet_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_canary_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_canary_qwen_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_lfm2_audio_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_mini_omni2_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_cohere_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_granite_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_granite_nle_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_voxtral_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_voxtral4b_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_voxtral_tts_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_higgs_stt_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_qwen3_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_fastconformer_ctc_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_wav2vec2_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_vibevoice_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_vibevoice_tts_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_vibevoice_1p5b_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_kugelaudio_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_qwen3_tts_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_qwen3_tts_base_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_orpheus_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_chatterbox_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_tada_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_indextts_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_m2m100_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_t5_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_kokoro_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_glm_asr_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_kyutai_stt_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_firered_asr_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_moonshine_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_moonshine_backend_lang(const char* sole_lang);
std::unique_ptr<CrispasrBackend> crispasr_make_moonshine_streaming_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_gemma4_e2b_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_omniasr_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_mimo_asr_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_htdemucs_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_mel_band_roformer_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_crepe_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_btc_chords_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_tabcnn_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_rvc_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_beat_this_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_ark_asr_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_moss_audio_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_moss_tts_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_moss_tts_local_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_moss_transcribe_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_moss_transcribe_diarize_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_funasr_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_paraformer_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_sensevoice_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_sidon_backend();
std::unique_ptr<CrispasrBackend> crispasr_create_miotts_backend();
std::unique_ptr<CrispasrBackend> crispasr_create_piano_transcription_backend();
std::unique_ptr<CrispasrBackend> crispasr_create_basic_pitch_backend();
std::unique_ptr<CrispasrBackend> crispasr_create_mt3_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_voxcpm2_tts_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_voxcpm2_vae_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_cosyvoice3_tts_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_piper_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_melotts_backend();
#ifdef CRISPASR_HAVE_OUTETTS
std::unique_ptr<CrispasrBackend> crispasr_make_outetts_backend();
#endif
std::unique_ptr<CrispasrBackend> crispasr_make_zonos_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_f5_tts_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_irodori_tts_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_bark_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_pocket_tts_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_speecht5_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_dia_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_dots_tts_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_confucius4_tts_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_parler_tts_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_fastpitch_backend();
// csm-tts (§135): sesame/csm-1b — Llama backbone + depth decoder + Mimi codec.
std::unique_ptr<CrispasrBackend> crispasr_make_csm_tts_backend();
// bananamind-tts: BananaMind-TTS-V2.1 Tacotron-lite + HiFi-GAN (en-us/de-de).
std::unique_ptr<CrispasrBackend> crispasr_make_bananamind_tts_backend();
// omnivoice: k2-fsa/OmniVoice — Qwen3-based masked iterative TTS (600+ languages).
std::unique_ptr<CrispasrBackend> crispasr_make_omnivoice_backend();

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<CrispasrBackend> crispasr_create_backend(const std::string& name) {
    // "tiron" (#295) is a Whisper large-v3 checkpoint with inline <|speakerN|>
    // markers — it runs on the whisper backend, which auto-detects the speaker
    // vocab and enables the tiron constrained-decoding grammar. The alias only
    // affects the model-registry lookup + CLI dispatch.
    if (name == "whisper" || name == "tiron")
        return crispasr_make_whisper_backend();
    if (name == "gigaam" || name == "gigaam-v3" || name == "gigaam3")
        return crispasr_make_gigaam_backend();
    if (name == "nemotron" || name == "nemotron-streaming" || name == "nemotron-3.5" || name == "nemotron-asr" ||
        name == "nemotron-speech-streaming")
        return crispasr_make_nemotron_backend();
    if (name == "parakeet" || name == "reazonspeech" || name == "quds" || name == "quds-fa")
        return crispasr_make_parakeet_backend();
    if (name == "canary")
        return crispasr_make_canary_backend();
    if (name == "canary-qwen" || name == "canary_qwen" || name == "canary-qwen-2.5b")
        return crispasr_make_canary_qwen_backend();
    if (name == "lfm2-audio")
        return crispasr_make_lfm2_audio_backend();
    if (name == "mini-omni2" || name == "mini_omni2" || name == "miniomni2")
        return crispasr_make_mini_omni2_backend();
    if (name == "cohere" || name == "cohere-ar")
        return crispasr_make_cohere_backend();
    if (name == "granite" || name == "granite-4.1" || name == "granite-4.1-plus")
        return crispasr_make_granite_backend();
    if (name == "granite-4.1-nar" || name == "granite-nar")
        return crispasr_make_granite_nle_backend();
    if (name == "voxtral")
        return crispasr_make_voxtral_backend();
    if (name == "voxtral4b")
        return crispasr_make_voxtral4b_backend();
    if (name == "voxtral-tts")
        return crispasr_make_voxtral_tts_backend();
    if (name == "higgs-stt" || name == "higgs_stt" || name == "higgs-audio-v3-stt" || name == "higgs-audio-stt" ||
        name == "higgsaudiostt")
        return crispasr_make_higgs_stt_backend();
    if (name == "qwen3" || name == "qwen3-1.7b" || name == "qwen3_1.7b" || name == "qwen3_17b" || name == "mega-asr" ||
        name == "mega_asr" || name == "megaasr")
        return crispasr_make_qwen3_backend();
    if (name == "fastconformer-ctc" || name == "fastconformer_ctc" || name == "canary-ctc" || name == "canary_ctc")
        return crispasr_make_fastconformer_ctc_backend();
    if (name == "wav2vec2" || name == "hubert" || name == "data2vec")
        return crispasr_make_wav2vec2_backend();
    if (name == "vibevoice" || name == "vibevoice-bitnet" || name == "vibevoice-asr-bitnet")
        return crispasr_make_vibevoice_backend();
    if (name == "vibevoice-tts")
        return crispasr_make_vibevoice_tts_backend();
    if (name == "kugelaudio" || name == "kugelaudio-tts" || name == "kugelaudio-0-open")
        return crispasr_make_kugelaudio_backend();
    if (name == "vibevoice-1.5b" || name == "vibevoice-tts-1.5b" || name == "vibevoice-tts-base")
        return crispasr_make_vibevoice_1p5b_backend();
    if (name == "qwen3-tts" || name == "qwen3_tts" || name == "qwen3tts" || name == "qwen3-tts-1.7b-base" ||
        name == "qwen3-tts-1.7b")
        return crispasr_make_qwen3_tts_base_backend();
    if (name == "qwen3-tts-customvoice" || name == "qwen3tts-customvoice" || name == "qwen3-tts-cv" ||
        name == "qwen3-tts-1.7b-customvoice" || name == "qwen3-tts-1.7b-cv" || name == "qwen3-tts-1.7b-voicedesign" ||
        name == "qwen3-tts-voicedesign" || name == "qwen3-tts-vd")
        return crispasr_make_qwen3_tts_backend();
    if (name == "miotts" || name == "mio-tts" || name == "mio_tts" || name == "miotts-0.6b")
        return crispasr_create_miotts_backend();
    if (name == "piano-transcription" || name == "piano_transcription" || name == "piano-trans")
        return crispasr_create_piano_transcription_backend();
    if (name == "basic-pitch" || name == "basic_pitch" || name == "basicpitch")
        return crispasr_create_basic_pitch_backend();
    if (name == "mt3" || name == "music-transcription" || name == "music_transcription")
        return crispasr_create_mt3_backend();
    if (name == "moss-tts-local" || name == "moss_tts_local" || name == "moss-tts-local-v1.5" ||
        name == "mosstts-local" || name == "moss-tts-local-transformer")
        return crispasr_make_moss_tts_local_backend();
    if (name == "moss-tts" || name == "moss_tts" || name == "mosstts" || name == "moss-tts-v1.5" ||
        name == "moss-tts-delay")
        return crispasr_make_moss_tts_backend();
    if (name == "orpheus" || name == "orpheus-tts" || name == "orpheus3b" || name == "kartoffel-orpheus" ||
        name == "kartoffel_orpheus" || name == "kartoffel-orpheus-de-natural" ||
        name == "kartoffel-orpheus-de-synthetic" || name == "kartoffel-orpheus-natural" ||
        name == "kartoffel-orpheus-synthetic" || name == "lex-au-orpheus-de" || name == "lex-au-orpheus")
        return crispasr_make_orpheus_backend();
    if (name == "chatterbox" || name == "chatterbox-tts" || name == "chatterbox-base" || name == "chatterbox-turbo" ||
        name == "chatterbox_turbo" || name == "chatterbox-nano" || name == "chatterbox_nano" ||
        name == "chatterbox-finnish-nano" || name == "chatterbox_finnish_nano" || name == "kartoffelbox" ||
        name == "kartoffelbox-turbo" || name == "kartoffelbox_turbo" || name == "lahgtna" ||
        name == "lahgtna-chatterbox" || name == "lahgtna-chatterbox-v1")
        return crispasr_make_chatterbox_backend();
    if (name == "tada" || name == "tada-tts" || name == "tada-1b" || name == "tada-tts-1b" || name == "tada-3b" ||
        name == "tada-3b-ml")
        return crispasr_make_tada_backend();
    if (name == "indextts" || name == "indextts-1.5" || name == "indextts1.5" || name == "index-tts")
        return crispasr_make_indextts_backend();
    if (name == "kokoro" || name == "styletts2" || name == "styletts2-ljspeech" || name == "kokoro-tts")
        return crispasr_make_kokoro_backend();
    if (name == "piper" || name == "piper-tts" || name == "piper-vits")
        return crispasr_make_piper_backend();
    if (name == "melotts" || name == "melo-tts" || name == "melo")
        return crispasr_make_melotts_backend();
#ifdef CRISPASR_HAVE_OUTETTS
    if (name == "outetts" || name == "outetts-tts" || name == "oute-tts" || name == "outetts-0.3-1b")
        return crispasr_make_outetts_backend();
#endif
    if (name == "f5-tts" || name == "f5_tts" || name == "f5tts" || name == "f5" || name == "raon" || name == "raon-1b")
        return crispasr_make_f5_tts_backend();
    if (name == "irodori-tts" || name == "irodori_tts" || name == "irodori")
        return crispasr_make_irodori_tts_backend();
    if (name == "pocket-tts" || name == "pocket_tts" || name == "pockettts" || name == "pocket" ||
        name == "pocket-tts-de" || name == "pocket-tts-german" || name == "pocket-tts-es" ||
        name == "pocket-tts-spanish" || name == "pocket-tts-it" || name == "pocket-tts-italian" ||
        name == "pocket-tts-pt" || name == "pocket-tts-portuguese" || name == "pocket-tts-fr" ||
        name == "pocket-tts-french")
        return crispasr_make_pocket_tts_backend();
    if (name == "fastpitch" || name == "fastpitch-tts" || name == "fastpitch_tts")
        return crispasr_make_fastpitch_backend();
    if (name == "voxcpm2-tts" || name == "voxcpm2" || name == "voxcpm" || name == "voxcpm2_tts")
        return crispasr_make_voxcpm2_tts_backend();
    if (name == "voxcpm2-vae" || name == "voxcpm2_vae" || name == "voxcpm2-upscaler")
        return crispasr_make_voxcpm2_vae_backend();
    // `cosyvoice3-tts-rl` is the same engine pointed at upstream's OTHER talker
    // checkpoint (llm.rl.pt — reinforcement-learning tuned for stability and
    // pronunciation accuracy). Flow, HiFT, CAMPPlus, the speech tokenizer and
    // the voice bank are shared, so the alias exists only so `-m auto` fetches
    // the RL LLM GGUF instead of the base one (#334).
    if (name == "cosyvoice3" || name == "cosyvoice3-tts" || name == "cosyvoice3_tts" || name == "cv3" ||
        name == "cv3-tts" || name == "cosyvoice3-rl" || name == "cosyvoice3-tts-rl" || name == "cv3-rl")
        return crispasr_make_cosyvoice3_tts_backend();
    if (name == "m2m100" || name == "m2m-100" || name == "translate" || name == "m2m100-wmt21" || name == "wmt21" ||
        name == "m2m100-1.2b")
        return crispasr_make_m2m100_backend();
    if (name == "madlad" || name == "madlad400" || name == "madlad-400" || name == "t5" || name == "t5-translate")
        return crispasr_make_t5_backend();
    if (name == "glm-asr" || name == "glmasr" || name == "glm" || name == "glm_asr")
        return crispasr_make_glm_asr_backend();
    if (name == "htdemucs" || name == "demucs" || name == "htdemucs-ft")
        return crispasr_make_htdemucs_backend();
    if (name == "mel-band-roformer" || name == "mel_band_roformer" || name == "melbandroformer" || name == "mbr")
        return crispasr_make_mel_band_roformer_backend();
    if (name == "crepe" || name == "crepe-tiny" || name == "crepe-full" || name == "pitch")
        return crispasr_make_crepe_backend();
    if (name == "btc-chords" || name == "btc" || name == "chords" || name == "btc-chords-large" ||
        name == "btc-chords-majmin")
        return crispasr_make_btc_chords_backend();
    if (name == "tabcnn" || name == "tab" || name == "tablature")
        return crispasr_make_tabcnn_backend();
    if (name == "rvc-svc" || name == "rvc" || name == "svc")
        return crispasr_make_rvc_backend();
    if (name == "beat-this" || name == "beatthis" || name == "beat_this" || name == "beats")
        return crispasr_make_beat_this_backend();
    if (name == "kyutai-stt" || name == "kyutai" || name == "moshi-stt" || name == "kyutai-stt-2.6b")
        return crispasr_make_kyutai_stt_backend();
    if (name == "firered-asr" || name == "firered")
        return crispasr_make_firered_asr_backend();
    if (name == "moonshine-streaming")
        return crispasr_make_moonshine_streaming_backend();
    if (name == "gemma4-e2b" || name == "gemma4e2b" || name == "gemma4")
        return crispasr_make_gemma4_e2b_backend();
    // The de fine-tunes share the runtime but are NOT en-only — the variant's
    // language must ride along or the sole-language guard rejects `-l de` and
    // the #227 auto shortcut mislabels output (found 2026-09-02).
    if (name == "moonshine-de" || name == "moonshine-tiny-de")
        return crispasr_make_moonshine_backend_lang("de");
    if (name == "moonshine")
        return crispasr_make_moonshine_backend();
    if (name.rfind("omniasr", 0) == 0)
        return crispasr_make_omniasr_backend();
    if (name == "mimo-asr" || name == "mimo_asr" || name == "mimoasr")
        return crispasr_make_mimo_asr_backend();
    if (name == "ark-asr" || name == "ark_asr" || name == "arkasr" || name == "ark")
        return crispasr_make_ark_asr_backend();
    if (name == "moss-audio" || name == "moss_audio" || name == "mossaudio")
        return crispasr_make_moss_audio_backend();
    if (name == "moss-transcribe" || name == "moss_transcribe" || name == "mosstranscribe")
        return crispasr_make_moss_transcribe_backend();
    if (name == "moss-diarize" || name == "moss_diarize" || name == "moss-transcribe-diarize" ||
        name == "moss_transcribe_diarize")
        return crispasr_make_moss_transcribe_diarize_backend();
    if (name == "funasr" || name == "fun-asr" || name == "fun-asr-nano" || name == "fun-asr-mlt-nano")
        return crispasr_make_funasr_backend();
    if (name == "paraformer" || name == "paraformer-zh" || name == "paraformer-en")
        return crispasr_make_paraformer_backend();
    if (name == "sensevoice" || name == "sensevoice-small" || name == "sense-voice")
        return crispasr_make_sensevoice_backend();
    if (name == "sidon" || name == "sidon-v0.1" || name == "speech-restoration")
        return crispasr_make_sidon_backend();
    if (name == "bark" || name == "bark-tts" || name == "bark-small" || name == "bark-large")
        return crispasr_make_bark_backend();
    if (name == "speecht5" || name == "speecht5-tts" || name == "speecht5_tts")
        return crispasr_make_speecht5_backend();
    if (name == "dia" || name == "dia-tts" || name == "dia-1.6b" || name == "dia_tts")
        return crispasr_make_dia_backend();
    if (name == "dots-tts" || name == "dots_tts" || name == "dots" || name == "dots.tts")
        return crispasr_make_dots_tts_backend();
    if (name == "confucius4-tts" || name == "confucius4_tts" || name == "confucius4")
        return crispasr_make_confucius4_tts_backend();
    if (name == "parler-tts" || name == "parler_tts" || name == "parler" || name == "parlertts")
        return crispasr_make_parler_tts_backend();
    if (name == "zonos" || name == "zonos-tts" || name == "zonos_tts")
        return crispasr_make_zonos_backend();
    if (name == "csm" || name == "csm-tts" || name == "csm_tts" || name == "sesame" || name == "sesame-csm")
        return crispasr_make_csm_tts_backend();
    if (name == "bananamind" || name == "bananamind-tts" || name == "bananamind_tts" || name == "banana-tts")
        return crispasr_make_bananamind_tts_backend();
    if (name == "omnivoice" || name == "omnivoice-tts" || name == "omnivoice_tts" || name == "omnivoice-singing")
        return crispasr_make_omnivoice_backend();

    fprintf(stderr, "crispasr: error: unknown backend '%s'\n", name.c_str());
    return nullptr;
}

std::vector<std::string> crispasr_list_backends() {
    return {
        "whisper",
        "nemotron",
        "gigaam",
        "parakeet",
        "reazonspeech",
        "quds-fa",
        "canary",
        "canary-qwen",
        "lfm2-audio",
        "mini-omni2",
        "cohere",
        "granite",
        "granite-4.1",
        "granite-4.1-plus",
        "granite-4.1-nar",
        "voxtral",
        "voxtral4b",
        "voxtral-tts",
        "qwen3",
        "qwen3-1.7b",
        "mega-asr",
        "higgs-stt",
        "fastconformer-ctc",
        "wav2vec2",
        "hubert",
        "data2vec",
        "vibevoice",
        "vibevoice-bitnet",
        "kugelaudio",
        "qwen3-tts",
        "miotts",
        "piano-transcription",
        "basic-pitch",
        "mt3",
        "moss-tts",
        "moss-tts-local",
        "vibevoice-1.5b",
        "qwen3-tts-customvoice",
        "qwen3-tts-1.7b-base",
        "qwen3-tts-1.7b-customvoice",
        "qwen3-tts-1.7b-voicedesign",
        "orpheus",
        "lex-au-orpheus-de",
        "kartoffel-orpheus-de-natural",
        "kartoffel-orpheus-de-synthetic",
        "chatterbox",
        "chatterbox-turbo",
        "chatterbox-nano",
        "chatterbox-finnish-nano",
        "kartoffelbox-turbo",
        "lahgtna-chatterbox",
        "tada",
        "tada-1b",
        "tada-tts-1b",
        "tada-3b-ml",
        "indextts",
        "f5-tts",
        "raon",
        "pocket-tts",
        "pocket-tts-de",
        "pocket-tts-es",
        "pocket-tts-it",
        "pocket-tts-pt",
        "pocket-tts-fr",
        "fastpitch",
        "kokoro",
        "melotts",
        "piper",
        "outetts",
        "voxcpm2-tts",
        "voxcpm2-vae",
        "cosyvoice3-tts",
        "cosyvoice3-tts-rl",
        "m2m100",
        "m2m100-wmt21",
        "madlad",
        "glm-asr",
        "kyutai-stt",
        "firered-asr",
        "moonshine",
        "moonshine-streaming",
        "gemma4-e2b",
        "omniasr",
        "omniasr-300m",
        "omniasr-llm",
        "omniasr-llm-1b",
        "mimo-asr",
        "ark-asr",
        "moss-audio",
        "moss-transcribe",
        "moss-diarize",
        "funasr",
        "fun-asr-mlt-nano",
        "paraformer",
        "sensevoice",
        "sidon",
        "bark",
        "bark-tts",
        "speecht5",
        "dia",
        "dia-tts",
        "dots-tts",
        "confucius4-tts",
        "parler-tts",
        "zonos",
        "zonos-tts",
        "csm",
        "csm-tts",
        "sesame",
        "bananamind-tts",
        "omnivoice",
        "omnivoice-singing",
        "htdemucs",
        "mel-band-roformer",
        "crepe",
        "btc-chords",
        "tabcnn",
        "rvc-svc",
        "beat-this",
    };
}

// ---------------------------------------------------------------------------
// Capability matrix for --list-backends
// ---------------------------------------------------------------------------

struct feature_col {
    const char* label;
    uint32_t flag;
};

static constexpr feature_col kFeatures[] = {
    {"ts-native", CAP_TIMESTAMPS_NATIVE},
    {"ts-ctc", CAP_TIMESTAMPS_CTC},
    {"word-ts", CAP_WORD_TIMESTAMPS},
    {"tok-conf", CAP_TOKEN_CONFIDENCE},
    {"lang-detect", CAP_LANGUAGE_DETECT},
    {"translate", CAP_TRANSLATE},
    {"diarize", CAP_DIARIZE},
    {"grammar", CAP_GRAMMAR},
    {"temperature", CAP_TEMPERATURE},
    {"beam", CAP_BEAM_SEARCH},
    {"flash", CAP_FLASH_ATTN},
    {"punctuation", CAP_PUNCTUATION_TOGGLE},
    {"punctuation-native", CAP_PUNCTUATION_NATIVE},
    {"src/tgt lang", CAP_SRC_TGT_LANGUAGE},
    {"auto-dl", CAP_AUTO_DOWNLOAD},
    {"tts", CAP_TTS},
    {"s2s", CAP_S2S},
    {"voice-clone", CAP_VOICE_CLONING},
    {"streaming", CAP_STREAMING},
    {"separate", CAP_SEPARATE},
    {"pitch", CAP_PITCH},
    {"chords", CAP_CHORDS},
    {"beats", CAP_BEATS},
    {"tab", CAP_TAB},
    {"piano", CAP_PIANO},
    {"tts-speed", CAP_TTS_SPEED},
};

void crispasr_print_backend_matrix() {
    const auto backends = crispasr_list_backends();

    // Column widths
    size_t name_w = 8;
    for (const auto& b : backends)
        if (b.size() > name_w)
            name_w = b.size();

    // Header
    printf("crispasr backends (%zu):\n\n", backends.size());
    printf("  %-*s", (int)name_w, "backend");
    for (const auto& f : kFeatures)
        printf(" %-12s", f.label);
    printf("\n  ");
    for (size_t i = 0; i < name_w; i++)
        printf("-");
    for (size_t i = 0; i < sizeof(kFeatures) / sizeof(kFeatures[0]); i++) {
        printf(" ------------");
    }
    printf("\n");

    // Each row: instantiate the backend just to read its capability bitmask.
    for (const auto& name : backends) {
        uint32_t caps = 0;
        auto be = crispasr_create_backend(name);
        if (be)
            caps = be->capabilities();
        // backend destroyed when unique_ptr goes out of scope
        printf("  %-*s", (int)name_w, name.c_str());
        for (const auto& f : kFeatures) {
            printf(" %-12s", (caps & f.flag) ? "   Y" : "    -");
        }
        printf("\n");
    }
    printf("\nUse --backend NAME to force a specific backend. When omitted, the\n");
    printf("backend is auto-detected from GGUF metadata or the filename.\n");
    printf("\n");
    printf("Language detection: backends that don't advertise lang-detect\n");
    printf("natively (cohere, canary, granite, voxtral, voxtral4b) can still\n");
    printf("accept `-l auto` via the LID pre-step. Pick the provider with\n");
    printf("--lid-backend whisper|silero (whisper-tiny is the default).\n");
}

// Stable cap-name slugs for JSON output. These map to the kFeatures
// table above but use long, kebab-cased names that are nicer for
// machine consumers (test-all-backends.py, etc.) than the table
// labels which are tuned for terminal-column-width.
struct cap_slug {
    const char* slug;
    uint32_t flag;
};
static constexpr cap_slug kCapSlugs[] = {
    {"timestamps-native", CAP_TIMESTAMPS_NATIVE},
    {"timestamps-ctc", CAP_TIMESTAMPS_CTC},
    {"word-timestamps", CAP_WORD_TIMESTAMPS},
    {"token-confidence", CAP_TOKEN_CONFIDENCE},
    {"language-detect", CAP_LANGUAGE_DETECT},
    {"translate", CAP_TRANSLATE},
    {"diarize", CAP_DIARIZE},
    {"grammar", CAP_GRAMMAR},
    {"temperature", CAP_TEMPERATURE},
    {"beam-search", CAP_BEAM_SEARCH},
    {"flash-attn", CAP_FLASH_ATTN},
    {"punctuation-toggle", CAP_PUNCTUATION_TOGGLE},
    {"punctuation-native", CAP_PUNCTUATION_NATIVE},
    {"src-tgt-language", CAP_SRC_TGT_LANGUAGE},
    {"auto-download", CAP_AUTO_DOWNLOAD},
    {"parallel-processors", CAP_PARALLEL_PROCESSORS},
    {"vad-internal", CAP_VAD_INTERNAL},
    {"tts", CAP_TTS},
    {"s2s", CAP_S2S},
    {"voice-cloning", CAP_VOICE_CLONING},
    {"streaming", CAP_STREAMING},
    {"separate", CAP_SEPARATE},
    {"pitch", CAP_PITCH},
    {"chords", CAP_CHORDS},
    {"beats", CAP_BEATS},
    {"tab", CAP_TAB},
    {"piano", CAP_PIANO},
    {"tts-speed", CAP_TTS_SPEED},
};

void crispasr_print_backend_matrix_json() {
    const auto backends = crispasr_list_backends();

    printf("{\n  \"backends\": [\n");
    bool first_backend = true;
    for (const auto& name : backends) {
        uint32_t caps = 0;
        auto be = crispasr_create_backend(name);
        if (be)
            caps = be->capabilities();
        // backend destroyed when unique_ptr goes out of scope

        if (!first_backend)
            printf(",\n");
        first_backend = false;
        printf("    {\n");
        printf("      \"name\": \"%s\",\n", name.c_str());
        printf("      \"caps_bitmask\": %u,\n", caps);
        printf("      \"caps\": [");
        bool first_cap = true;
        for (const auto& c : kCapSlugs) {
            if (caps & c.flag) {
                if (!first_cap)
                    printf(", ");
                first_cap = false;
                printf("\"%s\"", c.slug);
            }
        }
        printf("]\n");
        printf("    }");
    }
    printf("\n  ]\n}\n");
}

// ---------------------------------------------------------------------------
// GGUF auto-detection
// ---------------------------------------------------------------------------

// Read the "general.architecture" key from a GGUF file and map it to a
// backend name. Uses gguf_init_from_file() — which lives in ggml — so this
// is cheap: only the metadata is parsed, not the weight tensors.
//
// Mappings are based on the value that each model's converter writes into
// the GGUF file. When a converter doesn't write this key we fall back to
// filename heuristics.
std::string crispasr_detect_backend_from_gguf(const std::string& model_path) {
    if (model_path.empty())
        return "";

    // ---- Pass 1: filename heuristics ----
    //
    // Try filename matching first. This avoids two problems:
    //   1. Whisper's legacy ggml-*.bin files are not GGUF; calling
    //      gguf_init_from_file() on them prints a confusing stderr warning.
    //   2. It's a fast path that covers nearly every real-world case
    //      (users consistently name their models after the architecture).
    // Extract filename (after last / or \) for matching — avoid false
    // positives from directory names (e.g. "test_cohere/" matching "cohere").
    std::string fname = model_path;
    auto sep = fname.find_last_of("/\\");
    if (sep != std::string::npos)
        fname = fname.substr(sep + 1);
    auto contains_ci = [&](const char* needle) {
        std::string lo;
        lo.reserve(fname.size());
        for (char c : fname)
            lo += (char)std::tolower((unsigned char)c);
        return lo.find(needle) != std::string::npos;
    };

    if (contains_ci("htdemucs") || contains_ci("demucs"))
        return "htdemucs";
    if (contains_ci("mel-band-roformer") || contains_ci("mel_band_roformer") || contains_ci("roformer"))
        return "mel-band-roformer";
    if (contains_ci("crepe"))
        return "crepe";
    if (contains_ci("btc"))
        return "btc-chords";
    if (contains_ci("tabcnn"))
        return "tabcnn";
    if (contains_ci("beat-this") || contains_ci("beat_this") || contains_ci("beatthis"))
        return "beat-this";
    if (contains_ci("voxtral") && contains_ci("tts"))
        return "voxtral-tts";
    if (contains_ci("voxtral") && contains_ci("4b"))
        return "voxtral4b";
    if (contains_ci("voxtral"))
        return "voxtral";
    if (contains_ci("higgs") && (contains_ci("stt") || contains_ci("audio")))
        return "higgs-stt";
    // Distinguish parakeet-CTC standalones (parakeet-ctc-0.6b /
    // parakeet-ctc-1.1b — same FastConformer encoder + CTC head as the
    // stt_en_fastconformer_ctc family) from parakeet-TDT (transducer)
    // and the parakeet-tdt_ctc-*-ja hybrid, which are routed via the
    // `parakeet` backend's TDT path. The "tdt" guard keeps the JA
    // hybrid (parakeet-tdt_ctc-0.6b-ja) on the parakeet route even
    // though its filename also contains "ctc".
    if (contains_ci("parakeet") && contains_ci("ctc") && !contains_ci("tdt"))
        return "fastconformer-ctc";
    if (contains_ci("parakeet"))
        return "parakeet";
    if (contains_ci("reazonspeech"))
        return "parakeet";
    if (contains_ci("quds"))
        return "parakeet"; // #387: Persian FastConformer-RNNT rides the parakeet runtime

    // Check "fastconformer-ctc" / "stt_en_fc_ctc" style filenames before
    // the broader "canary" match so users who drop a NeMo standalone
    // model next to a canary aligner pick the right backend.
    if (contains_ci("fastconformer") && contains_ci("ctc"))
        return "fastconformer-ctc";
    if (contains_ci("omniasr"))
        return "omniasr";
    if (contains_ci("wav2vec2"))
        return "wav2vec2";
    if (contains_ci("hubert"))
        return "wav2vec2";
    if (contains_ci("data2vec"))
        return "wav2vec2";
    if (contains_ci("vibevoice"))
        return "vibevoice";
    if (contains_ci("ark-asr") || contains_ci("arkasr") || contains_ci("ark_asr"))
        return "ark-asr";
    if (contains_ci("kugelaudio"))
        return "kugelaudio";
    if (contains_ci("fireredpunc"))
        return "fireredpunc";
    // A canary-*-ctc filename is a FastConformer-CTC model (canary_ctc
    // runtime), not the AED encoder-decoder "canary" backend — match the
    // "ctc" qualifier before the broad canary catch-all below.
    if (contains_ci("canary") && contains_ci("ctc"))
        return "fastconformer-ctc";
    if (contains_ci("canary") && contains_ci("qwen"))
        return "canary-qwen";
    if (contains_ci("canary"))
        return "canary";
    if (contains_ci("lfm2-audio") || contains_ci("lfm2_audio"))
        return "lfm2-audio";
    if (contains_ci("mini-omni2") || contains_ci("mini_omni2") || contains_ci("miniomni2"))
        return "mini-omni2";
    if (contains_ci("cohere"))
        return "cohere";
    if (contains_ci("mega-asr") || contains_ci("mega_asr") || contains_ci("megaasr"))
        return "mega-asr";
    if (contains_ci("qwen3") && contains_ci("asr"))
        return "qwen3";
    if (contains_ci("qwen3") && contains_ci("tts"))
        return "qwen3-tts";
    if (contains_ci("orpheus") || contains_ci("kartoffel-orpheus") || contains_ci("kartoffel_orpheus"))
        return "orpheus";
    if (contains_ci("outetts") || contains_ci("oute-tts") || contains_ci("oute_tts"))
        return "outetts";
    if (contains_ci("indextts"))
        return "indextts";
    if (contains_ci("f5-tts") || contains_ci("f5tts") || contains_ci("F5TTS") || contains_ci("raon"))
        return "f5-tts"; // #387 Raon-OpenTTS rides the f5-tts runtime
    if (contains_ci("pocket-tts") || contains_ci("pocket_tts") || contains_ci("pockettts"))
        return "pocket-tts";
    if (contains_ci("fastpitch"))
        return "fastpitch";
    if (contains_ci("bananamind"))
        return "bananamind-tts";
    if (contains_ci("melotts") || contains_ci("melo-tts") || contains_ci("melo_tts"))
        return "melotts";
    if (contains_ci("piper") && !contains_ci("piper-phonemize"))
        return "piper";
    if (contains_ci("chatterbox") || contains_ci("kartoffelbox") || contains_ci("lahgtna"))
        return "chatterbox";
    if (contains_ci("tada"))
        return "tada";
    if (contains_ci("m2m100") || (contains_ci("m2m") && contains_ci("100")) || contains_ci("wmt21"))
        return "m2m100";
    if (contains_ci("madlad"))
        return "madlad";
    if (contains_ci("kokoro"))
        return "kokoro";
    if (contains_ci("styletts") && contains_ci("ljspeech"))
        return "kokoro";
    if (contains_ci("voxcpm2-vae") || contains_ci("voxcpm2_vae"))
        return "voxcpm2-vae";
    if (contains_ci("voxcpm2") || contains_ci("voxcpm"))
        return "voxcpm2-tts";
    if (contains_ci("cosyvoice3") || contains_ci("cosyvoice-3") || contains_ci("cv3"))
        return "cosyvoice3-tts";
    if (contains_ci("granite") && contains_ci("nar"))
        return "granite-4.1-nar";
    if (contains_ci("granite") && contains_ci("speech"))
        return "granite";
    if (contains_ci("glm") && contains_ci("asr"))
        return "glm-asr";
    if (contains_ci("kyutai") && contains_ci("stt"))
        return "kyutai-stt";
    if (contains_ci("moshi") && contains_ci("stt"))
        return "kyutai-stt";
    if (contains_ci("firered") && (contains_ci("asr") || contains_ci("lid")))
        return "firered-asr";
    if (contains_ci("gemma") && (contains_ci("e2b") || contains_ci("4-e2b")))
        return "gemma4-e2b";
    if (contains_ci("moonshine") && contains_ci("streaming"))
        return "moonshine-streaming";
    // The de fine-tune must resolve to its variant name so the factory hands
    // it the right sole language (a plain "moonshine" would be treated as
    // en-only by the pre-dispatch guard).
    if (contains_ci("moonshine") && contains_ci("-de"))
        return "moonshine-de";
    if (contains_ci("moonshine"))
        return "moonshine";
    if (contains_ci("fun-asr") || contains_ci("funasr") || contains_ci("fun_asr"))
        return "funasr";
    if (contains_ci("sensevoice") || contains_ci("sense-voice") || contains_ci("sense_voice"))
        return "sensevoice";
    if (contains_ci("sidon"))
        return "sidon";
    if (contains_ci("speecht5"))
        return "speecht5";
    if (contains_ci("bark"))
        return "bark";
    if (contains_ci("dia") && contains_ci("1.6b"))
        return "dia";
    if (contains_ci("dia-tts") || contains_ci("dia_tts"))
        return "dia";
    if (contains_ci("confucius4") || contains_ci("confucius4-tts") || contains_ci("confucius4_tts"))
        return "confucius4-tts";
    if (contains_ci("dots-tts") || contains_ci("dots_tts") || contains_ci("dots.tts"))
        return "dots-tts";
    if (contains_ci("csm") || contains_ci("sesame"))
        return "csm";
    if (contains_ci("parler") && contains_ci("tts"))
        return "parler-tts";
    if (contains_ci("zonos"))
        return "zonos";
    if (contains_ci("moss") && contains_ci("diarize"))
        return "moss-diarize";
    if (contains_ci("moss") && contains_ci("transcribe"))
        return "moss-transcribe";
    if (contains_ci("moss") && contains_ci("tts") && contains_ci("local"))
        return "moss-tts-local";
    if (contains_ci("moss") && contains_ci("tts"))
        return "moss-tts";
    if (contains_ci("moss") && contains_ci("audio"))
        return "moss-audio";
    if (contains_ci("miotts"))
        return "miotts";
    if (contains_ci("piano") && contains_ci("transcription"))
        return "piano-transcription";
    if (contains_ci("basic") && contains_ci("pitch"))
        return "basic-pitch";
    if (contains_ci("mt3"))
        return "mt3";
    if (contains_ci("gigaam"))
        return "gigaam";
    if (contains_ci("ggml-") && contains_ci(".bin"))
        return "whisper";

    // ---- Pass 2: GGUF metadata ----
    //
    // Only reached when the filename didn't clearly identify a backend.
    // Reads just the "general.architecture" key; no weight tensors.
    //
    // The architecture -> backend table is SHARED with the C ABI's
    // `crispasr_detect_backend_from_gguf` (src/core/arch_backend_map.h). Issue
    // #335: the two used to be independent copies and had drifted by 113
    // architecture strings. The filename pass above hid that from CLI users
    // while every binding got a NULL session for the very same GGUF.
    struct gguf_init_params gip = {/*.no_alloc=*/true, /*.ctx=*/nullptr};
    gguf_context* gctx = gguf_init_from_file(model_path.c_str(), gip);
    if (!gctx)
        return "";

    std::string result;
    const int key = gguf_find_key(gctx, "general.architecture");
    if (key >= 0) {
        const char* arch = gguf_get_val_str(gctx, key);
        if (arch)
            result = core_arch::backend_for_arch(arch);
    }
    gguf_free(gctx);
    return result;
}

bool crispasr_gguf_is_pure_ctc(const std::string& model_path) {
    // Mirror the parakeet backend's pure-CTC guard (src/parakeet.cpp): a NeMo
    // EncDecCTCModelBPE (parakeet-ctc-*, stt_*_fastconformer_ctc) has an encoder +
    // CTC head but NO RNN-T prediction network / joint — so it lacks both the
    // single-embedding predictor (decoder.embed.weight) and the LSTM predictor
    // (decoder.lstm.0.w_ih). Reads tensor infos only; no weights loaded.
    struct gguf_init_params gip = {/*.no_alloc=*/true, /*.ctx=*/nullptr};
    gguf_context* gctx = gguf_init_from_file(model_path.c_str(), gip);
    if (!gctx)
        return false;
    bool has_embed = false, has_lstm = false;
    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gctx, i);
        if (!name)
            continue;
        if (std::strcmp(name, "decoder.embed.weight") == 0)
            has_embed = true;
        else if (std::strcmp(name, "decoder.lstm.0.w_ih") == 0)
            has_lstm = true;
    }
    gguf_free(gctx);
    return !has_embed && !has_lstm;
}
