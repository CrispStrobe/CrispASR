// whisper_params.h — command-line parameter struct shared between the
// original whisper CLI code in cli.cpp and the new crispasr backend
// dispatch layer (crispasr_backend.*, crispasr_run.cpp).
//
// This struct was historically defined inline in cli.cpp. Extracting it to a
// header lets the new backend modules read parameters directly without
// duplicating fields or introducing a parallel params type. The unified CLI
// continues to feel like whisper-cli because it literally is — the same
// struct, the same arg parser, just with a few extra fields at the bottom.

#pragma once

#include "whisper.h" // for whisper_full_default_params used in defaults

#include <cfloat>
#include <string>
#include <thread>
#include <vector>

#include "grammar-parser.h"

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
    int32_t best_of = whisper_full_default_params(WHISPER_SAMPLING_GREEDY).greedy.best_of;
    int32_t beam_size = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
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
    bool verbose = false; // -v / --verbose: enable per-backend debug output
    bool print_special = false;
    bool print_colors = false;
    bool print_confidence = false;
    bool print_progress = false;
    bool no_timestamps = false;
    bool log_score = false;
    bool use_gpu = true;
    bool flash_attn = true;
    int32_t gpu_device = 0;
    // Override the automatic GPU backend selection. When empty (default),
    // ggml_backend_init_best() picks the highest-priority compiled backend
    // (CUDA > Metal > Vulkan > CPU). Set to "vulkan", "cuda", "metal", or
    // "cpu" to force a specific backend.
    std::string gpu_backend;
    bool suppress_nst = false;
    bool carry_initial_prompt = false;

    // Default "auto" for both model and language: running the CLI on any
    // audio file with no further args auto-downloads a reasonable default
    // whisper model and auto-detects the spoken language. User can still
    // pin either with -l <lang> or -m <path>.
    std::string language = "auto";
    std::string prompt;
    std::string ask; // Q&A prompt for audio understanding (voxtral --ask)
    std::string font_path = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf";
    std::string model = "auto";
    std::string grammar;
    std::string grammar_rule;

    // [TDRZ] speaker turn string
    std::string tdrz_speaker_turn = " [SPEAKER_TURN]";

    // A regular expression that matches tokens to suppress
    std::string suppress_regex;

    std::string openvino_encode_device = "CPU";

    std::string dtw = "";

    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};

    grammar_parser::parse_state grammar_parsed;

    // Voice Activity Detection (VAD) parameters
    bool vad = false;
    std::string vad_model = "";
    float vad_threshold = 0.5f;
    int vad_min_speech_duration_ms = 250;
    int vad_min_silence_duration_ms = 100;
    float vad_max_speech_duration_s = FLT_MAX;
    int vad_speech_pad_ms = 30;
    float vad_samples_overlap = 0.1f;
    bool vad_stitch = false; // stitch VAD segments into one buffer (legacy, breaks SRT)

    // -----------------------------------------------------------------
    // crispasr extensions: fields used only by non-whisper backends.
    // The whisper code path ignores these; the backend dispatch layer
    // in crispasr_run.cpp reads them when params.backend != "whisper".
    // -----------------------------------------------------------------

    // Backend selector: "", "whisper", "parakeet", "canary", "cohere",
    // "qwen3", "voxtral", "voxtral4b", "granite". Empty or "whisper" =>
    // use the historical whisper code path unchanged.
    std::string backend;

    // Canary needs separate source and target languages (ASR vs AST).
    // When empty, they fall back to `language`.
    std::string source_lang;
    std::string target_lang;

    // Punctuation toggle (canary, cohere). Default on.
    bool punctuation = true;

    // Path to FireRedPunc GGUF model for punctuation restoration post-step.
    std::string punc_model;

    // Progressive SRT: flush stdout after every N segments (0 = all at end).
    // When set to 1, each VAD slice's SRT entries are printed immediately.
    int flush_after = 0;

    // Show alternative token candidates alongside the primary text.
    // Inspired by antirez/voxtral.c --alt mode.
    bool show_alternatives = false;
    int32_t n_alternatives = 3; // top-N alternatives per token

    // Path to a CTC aligner model (canary_ctc.gguf) used by LLM-based
    // backends to produce word-level timestamps via a second pass.
    std::string aligner_model;

    // Maximum new tokens to generate for LLM-based backends.
    int32_t max_new_tokens = 512;

    // Fallback chunk size for long audio when VAD is not enabled.
    int32_t chunk_seconds = 30;

    // Optional language-detection pre-step for backends that don't
    // support auto-language natively (cohere, canary, granite, voxtral,
    // voxtral4b). Runs before transcribe() and fills in `language` +
    // `source_lang` from the detected code. `lid_backend` picks the
    // implementation: "whisper" (default, uses a small ggml-*.bin LID
    // model auto-downloaded on first use), "silero" (reserved for a
    // future native GGUF port of Silero's language detector), or empty
    // to disable. `lid_model` optionally overrides the default model
    // path for the chosen backend.
    std::string lid_backend;
    std::string lid_model;

    // Generic stereo diarize post-step. Empty = "energy" (default
    // method, same as historical whisper-cli). Other choices:
    // "xcorr" (cross-correlation TDOA), "sherpa" / "sherpa-onnx"
    // (external subprocess), "pyannote" / "ecapa" (TODO, pending
    // native GGUF ports). Only fires when `diarize` is true.
    std::string diarize_method;

    // sherpa-onnx configuration (only used when --diarize-method=sherpa).
    // crispasr shells out to an externally-installed
    // `sherpa-onnx-offline-speaker-diarization` binary; these paths are
    // passed through to it. The binary discovers models at the paths
    // given here, runs diarization on the input WAV, and returns a
    // segment list that we merge with the ASR output by time overlap.
    std::string sherpa_bin;             // default: "sherpa-onnx-offline-speaker-diarization"
    std::string sherpa_segment_model;   // pyannote-style segmentation ONNX
    std::string sherpa_embedding_model; // speaker embedding ONNX (titanet / 3dspeaker)
    int sherpa_num_clusters = 0;        // 0 = auto-estimate (sherpa default)

    // Streaming mode: read raw 16 kHz s16le PCM from stdin in chunks,
    // transcribe each chunk, print output immediately. Useful for piping
    // from ffmpeg, sox, or a microphone capture utility:
    //   ffmpeg -i mic.wav -f s16le -ar 16000 -ac 1 - | crispasr --stream -m model.gguf
    bool stream = false;
    bool mic = false;               // capture from default microphone
    bool stream_continuous = false; // don't stop on EOS, keep transcribing
    bool stream_monitor = false;    // show unicode progress symbols during processing

    // Server mode: keep model loaded, accept HTTP requests
    bool server = false;
    std::string server_host = "127.0.0.1";
    int32_t server_port = 8080;
    std::string server_api_keys;
    int32_t stream_step_ms = 3000;    // chunk size in ms (default 3s)
    int32_t stream_length_ms = 10000; // context window in ms (default 10s)
    int32_t stream_keep_ms = 200;     // overlap to keep between chunks

    // Auto-download missing models without prompting (like pip install)
    bool auto_download = false;

    // Override the directory used to cache auto-downloaded model files.
    // Default (empty): platform default (~/.cache/crispasr on POSIX,
    // %USERPROFILE%/.cache/crispasr on Windows). Set via --cache-dir.
    std::string cache_dir;
};
