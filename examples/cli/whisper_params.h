// whisper_params.h — command-line parameter struct shared between the
// historical whisper CLI surface and the CrispASR backend dispatch layer.
//
// Keep the `whisper_params` name for CLI/source compatibility. This is a
// frontend params struct, not a signal that the whole project is still named
// whisper.

#pragma once

#include "crispasr.h"
#include "grammar-parser.h"

#include <algorithm>
#include <cfloat>
#include <string>
#include <thread>
#include <vector>

struct whisper_params {
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t n_processors = 1;
    int32_t offset_t_ms = 0;
    int32_t offset_n = 0;
    int32_t duration_ms = 0;
    int32_t progress_step = 5;
    int32_t max_context = -1;
    int32_t max_len = 0;
    bool split_on_punct = false;
    int32_t best_of = whisper_full_default_params(CRISPASR_SAMPLING_GREEDY).greedy.best_of;
    int32_t beam_size = whisper_full_default_params(CRISPASR_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
    int32_t audio_ctx = 0;

    float word_thold = 0.01f;
    float entropy_thold = 2.40f;
    float logprob_thold = -1.00f;
    float no_speech_thold = 0.6f;
    float grammar_penalty = 100.0f;
    float temperature = 0.0f;
    float temperature_inc = 0.2f;

    bool debug_mode = false;
    bool translate = false;
    bool detect_language = false;
    bool diarize = false;
    bool tinydiarize = false;
    bool split_on_word = false;
    bool no_fallback = false;
    bool output_txt = false;
    bool output_vtt = false;
    bool output_srt = false;
    bool output_wts = false;
    bool output_csv = false;
    bool output_jsn = false;
    bool output_jsn_full = false;
    bool output_lrc = false;
    bool no_prints = false;
    bool verbose = false;
    bool print_special = false;
    bool print_colors = false;
    bool print_confidence = false;
    bool print_progress = false;
    bool no_timestamps = false;
    bool log_score = false;
    bool use_gpu = true;
    bool flash_attn = true;
    int32_t gpu_device = 0;
    std::string gpu_backend;
    bool suppress_nst = false;
    bool carry_initial_prompt = false;

    std::string language = "auto";
    std::string prompt;
    std::string ask;
    std::string font_path = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf";
    std::string model = "auto";
    std::string model_quant;
    std::string grammar;
    std::string grammar_rule;

    std::string tdrz_speaker_turn = " [SPEAKER_TURN]";
    std::string suppress_regex;
    std::string openvino_encode_device = "CPU";
    std::string dtw = "";

    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};

    grammar_parser::parse_state grammar_parsed;

    bool vad = false;
    std::string vad_model = "";
    float vad_threshold = 0.5f;
    int vad_min_speech_duration_ms = 250;
    int vad_min_silence_duration_ms = 100;
    float vad_max_speech_duration_s = FLT_MAX;
    int vad_speech_pad_ms = 30;
    float vad_samples_overlap = 0.1f;
    bool vad_stitch = false;

    std::string backend;
    std::string source_lang;
    std::string target_lang;
    bool punctuation = true;
    std::string punc_model;
    int flush_after = 0;
    bool show_alternatives = false;
    int32_t n_alternatives = 3;
    std::string aligner_model;
    // PLAN issue #62: when true, the CTC forced aligner runs even on
    // backends that already produce native timestamps — replacing
    // their words rather than skipping. Default false keeps the existing
    // semantics ("aligner only if the native path didn't produce
    // words"). Useful when the user trusts the aligner's accuracy more
    // than the backend's native timing (whisper, parakeet, canary,
    // cohere, kyutai-stt).
    bool force_aligner = false;
    // SubtitleEdit #10775: opt out of the canary auto-aligner default.
    // For --backend canary, when the user requests word-level output
    // (srt/vtt/json-full/wts/max-len/split-on-punct/print-colors) and
    // hasn't passed --aligner-model explicitly, crispasr now defaults
    // to `-am auto --force-aligner` because canary's native cross-attn
    // DTW timing is ~5× looser than canary-ctc-aligner (~414 ms vs
    // ~78 ms MAE on word boundaries — see canary-ctc-aligner-GGUF
    // README). `--no-auto-aligner` reverts to the pre-default native
    // DTW path (no second forward pass, no ~442 MB download).
    bool no_auto_aligner = false;
    int32_t max_new_tokens = 512;
    int32_t chunk_seconds = 30;
    std::string lid_backend;
    std::string lid_model;
    std::string diarize_method;
    std::string sherpa_bin;
    std::string sherpa_segment_model;
    std::string sherpa_embedding_model;
    int sherpa_num_clusters = 0;
    bool stream = false;
    bool mic = false;
    bool stream_continuous = false;
    bool stream_monitor = false;
    bool server = false;
    std::string server_host = "127.0.0.1";
    int32_t server_port = 8080;
    std::string server_api_keys;
    int32_t stream_step_ms = 3000;
    int32_t stream_length_ms = 10000;
    int32_t stream_keep_ms = 200;
    bool auto_download = false;
    bool dry_run_resolve = false;
    bool dry_run_ignore_cache = false;
    std::string cache_dir;
    std::string tts_text;
    std::string tts_output;
    std::string tts_voice;
    int tts_steps = 20;
    std::string tts_codec_model;
    std::string tts_codec_quant;
    std::string tts_ref_text;
    std::string tts_instruct; // VoiceDesign: natural-language voice description
    bool tts_trim_silence = false;

    // Server mode: directory containing voice profiles for /v1/audio/speech.
    // Each profile is a sibling pair: <name>.wav + <name>.txt (the WAV is
    // the reference audio, the TXT is its transcription used by Qwen3-TTS
    // ICL prefill), or a baked voice pack <name>.gguf. The server's
    // POST /v1/audio/speech `voice` field maps to a stem in this directory.
    std::string tts_voice_dir;

    // Server mode: hard ceiling on /v1/audio/speech `input` length. Default
    // matches OpenAI's documented limit (4096 chars). Set to 0 to disable
    // the cap entirely (useful when long-form chunking is enabled and the
    // caller explicitly wants the chunker to handle whatever they send).
    // Above the cap, the route returns 400 with code="input_too_long".
    int tts_max_input_chars = 4096;

    // Server mode: speed multiplier (0.25 .. 4.0, OpenAI range). Applied
    // by the route handler as a post-synth linear resampler — backends
    // produce audio at their native rate, the server resamples before the
    // WAV/PCM/f32 dispatch. Default 1.0 = no resample. The CLI ignores
    // this; only the server route reads it.
    float tts_speed = 1.0f;

    // Server mode: when non-empty, every server response gets the
    // Access-Control-Allow-* headers set so browser clients can call us
    // cross-origin. Default empty = no CORS headers (server stays
    // default-locked; a deployed instance opts in explicitly with
    // --cors-origin '*' or a specific origin).
    std::string server_cors_origin;

    // Text-to-text translation input (m2m100 + future translate-only
    // backends). When `--text` is set on a backend that declares
    // CAP_TRANSLATE and has no input audio, crispasr_run dispatches to
    // backend->translate_text() instead of transcribe.
    //
    // Language handling has TWO independent pairs to support 2-stage
    // pipelines (e.g., ASR that only does EN→EN-text, then m2m100
    // translates the EN-text to Tamil — those two stages have different
    // source/target conventions):
    //   - source_lang / target_lang : primary backend (canary AST etc.)
    //   - translate_source_lang / translate_target_lang : second-stage
    //     translator (m2m100). Empty falls back to source_lang /
    //     target_lang. So standalone `--backend m2m100 -sl en -tl de`
    //     just works without learning new flags; the dedicated `-trsl`
    //     / `-trtl` flags only matter when the primary backend's
    //     `-sl`/`-tl` mean something else (e.g., 2-stage piping).
    std::string text_input;
    int translate_max_tokens = 256;
    std::string translate_source_lang; // overrides source_lang for the translator stage
    std::string translate_target_lang; // overrides target_lang for the translator stage
};
