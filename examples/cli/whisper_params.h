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
    uint64_t seed = 0; // RNG seed for sampling (0 = non-deterministic)

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
    // Issue #83 follow-up: tracks whether the user passed `-vt` /
    // `--vad-threshold` explicitly. Backends with non-Silero probability
    // calibration (whisper-vad-encdec) auto-lower the threshold when this
    // is false so the out-of-the-box default doesn't drop most speech.
    bool vad_threshold_explicit = false;
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
    std::string truecase_model;
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
    float frequency_penalty = 0.0f;
    int32_t chunk_seconds = 30;
    bool chunk_seconds_explicit = false; // true when user passed --chunk-seconds
    float chunk_overlap_seconds = 3.0f;  // overlap context on each side of chunk boundary
    // Issue #89 / #114 follow-up: NeMo-style LCS hypothesis stitching at
    // chunk boundaries. "auto" fires when chunking with overlap is active,
    // matching upstream BatchedFrameASRTDT. "on" forces it (e.g. for
    // bindings testing); "off" disables for A/B comparison or debugging.
    std::string lcs_dedup = "auto"; // {auto|on|off}
    // Lower bound on LCS length to act on. NeMo's default is 1; raise to
    // 3-4 if your audio has long-silence regions where blank tokens
    // dominate the boundary token run (avoids over-slicing).
    int lcs_min_length = 1;
    bool warmup = false;          // run a short dummy transcribe after init to amortize first-call overhead (PLAN #80e)
    std::string parakeet_decoder; // "tdt" (default), "ctc" — selects parakeet decode head
    std::string hotwords;         // comma-separated hotword list (PLAN #98)
    float hotwords_boost = 2.0f;  // per-token log-prob boost for hotword prefix matches
    std::string lid_backend;
    std::string lid_model;
    // Post-ASR text LID: when set, after transcription completes, run
    // a fastText-supervised classifier (GlotLID / LID-176) on the
    // assembled transcript and emit `lang=<code>\tconf=<score>` to
    // stderr. Path points at a lid_fasttext.* GGUF.
    std::string lid_on_transcript;
    std::string diarize_method;
    std::string sherpa_bin;
    std::string sherpa_segment_model;
    std::string sherpa_embedding_model;
    int sherpa_num_clusters = 0;

    // Speaker identification via TitaNet embeddings + profile DB.
    std::string speaker_db;         // path to speaker profile directory
    std::string enroll_speaker;     // enrollment mode: save embedding as this name
    std::string titanet_model;      // TitaNet GGUF path or "auto"
    float speaker_threshold = 0.7f; // cosine similarity threshold for matching

    // Embedding-based diarization clustering (issue #107 P3). When set,
    // after --diarize-method pyannote runs, each speech segment is
    // embedded by this model and clustered on cosine similarity to
    // produce globally stable speaker IDs (independent of pyannote's
    // local track indices). Path, "auto", or empty for no clustering.
    // Currently dispatches to the TitaNet adapter; pluggable.
    std::string diarize_embedder;           // model path or "auto"
    float diarize_cluster_threshold = 0.5f; // cosine merge threshold
    int diarize_max_speakers = 8;           // upper bound for cluster count
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
    // Issue #84: opt-in JSON-Lines streaming output for wrappers
    // (browser bridges, translators) that need machine-readable
    // partial/final/silence events instead of plain stdout text.
    // When true, the streaming loop emits one JSON object per event
    // on stdout: {"type":"partial"|"final"|"silence", "utterance_id":N,
    // "text":"…", "t0":sec, "t1":sec}. Plain-text stdout is suppressed.
    bool stream_json = false;
    // Trailing-silence threshold (ms) for promoting the open partial
    // to final and incrementing utterance_id. 0 = never auto-finalize
    // (only an open partial is emitted). Default 800 ms is a sensible
    // pause length for live captions / translation handoff and matches
    // the example in the issue. Only relevant with --stream-json.
    int32_t stream_final_silence_ms = 800;
    // JSON streaming + VAD only: merge adjacent VAD slices across tiny
    // detector jitter gaps, but never across --stream-final-on-silence-ms.
    // 0 disables streaming-specific post-merge.
    int32_t stream_vad_merge_gap_ms = 250;
    // JSON streaming + VAD only: minimum interval between live partial ASR
    // decodes. 0 = decode partials every --stream-step, preserving the
    // previous behavior. VAD timing/finalization still runs every step.
    int32_t stream_partial_decode_ms = 0;
    // JSON streaming + VAD only: control FireRedPunc placement when
    // --punc-model is loaded. "final" avoids the high-frequency partial
    // punc path; "partial" preserves the older partial+final behavior.
    std::string stream_punc = "final"; // off|final|partial
    // Issue #84 round 2 (CKwasd retest): how to compute `final.text`
    // when an utterance closes. The round-1 design just echoed the
    // last rolling-window partial — wrong because the rolling window
    // evicts old audio, so for utterances longer than `--stream-length`
    // the tail of the latest partial does not cover the full
    // utterance. Two modes:
    //   "redecode" (default) — buffer the utterance's PCM (capped at
    //     stream_utterance_max_sec) and run one extra `transcribe()`
    //     on the whole buffer at finalize time. Best quality, costs
    //     one encoder pass per utterance + ~4 MB / utterance memory.
    //   "prefix" — keep the round-1 cost (no extra transcribe) but
    //     accumulate stable text via longest-common-prefix matching
    //     across consecutive partials. Lower quality for utterances
    //     longer than the rolling window, useful when re-decoding
    //     would blow latency or compute budget. The latest partial
    //     beyond the committed prefix is concatenated at finalize.
    std::string stream_final_mode = "redecode";
    // Hard cap on how much PCM we buffer per utterance in redecode
    // mode. Force-finalize when exceeded so memory stays bounded on
    // monologues. 60 s × 16 kHz × 4 bytes ≈ 3.84 MB.
    int32_t stream_utterance_max_sec = 60;
    // Issue #84: gate the noisy `firered_vad: N frames, max_prob=…`
    // and `fbank[0,:3]=…` stderr dumps in src/firered_vad.cpp behind
    // an explicit opt-in. Set by --firered-vad-debug (or the
    // CRISPASR_FIRERED_VAD_DEBUG env var directly).
    bool firered_vad_debug = false;
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

    // Text-LLM chat (server-mode /v1/chat/completions). Independent from
    // the audio backend's `model` path so a server can serve ASR + chat
    // off two different GGUFs. When empty, /v1/chat/completions returns
    // 503 with `chat_disabled` rather than 404. See docs/prompts/chat-abi.md
    // for the underlying ABI.
    std::string chat_model; // path to a GGUF chat model
    int32_t chat_n_ctx = 4096;
    int32_t chat_n_gpu_layers = -1;
};
