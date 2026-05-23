#include "common.h"
#include "common-crispasr.h"

#include "crispasr.h"
#include "grammar-parser.h"
#include "whisper_params.h"            // struct whisper_params (shared with crispasr_*)
#include "crispasr_backend.h"          // crispasr_run_backend() dispatch entry point
#include "crispasr_diagnostics.h"      // --version / --diagnostics + verbose banner (#31)
#include "crispasr_diarize_cli.h"      // crispasr_apply_diarize / pyannote cache (#107)
#include "crispasr_speaker_embedder.h" // pluggable speaker embedder (#107 P3)
#include "crispasr_stream_punc.h"      // streaming punctuation mode helpers (#112)
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "crispasr_output.h"   // crispasr_make_disp_segments — split-on-punct (#29)
#include "crispasr_server.h"   // crispasr_run_server()
#include "crispasr_vad_cli.h"  // crispasr_resolve_vad_model — auto-DL silero (#33)
#include "text_lid_dispatch.h" // text LID dispatcher for --lid-on-transcript (fastText or CLD3)

#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include <cstring>
#include <cfloat>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// helper function to replace substrings
static void replace_all(std::string& s, const std::string& search, const std::string& replace) {
    for (size_t pos = 0;; pos += replace.length()) {
        pos = s.find(search, pos);
        if (pos == std::string::npos)
            break;
        s.erase(pos, search.length());
        s.insert(pos, replace);
    }
}

#if 0  // Moved to whisper_params.h for sharing with crispasr backend dispatch.
// command-line parameters
struct whisper_params {
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors  = 1;
    int32_t offset_t_ms   = 0;
    int32_t offset_n      = 0;
    int32_t duration_ms   = 0;
    int32_t progress_step = 5;
    int32_t max_context   = -1;
    int32_t max_len       = 0;
    bool    split_on_punct = false;
    int32_t best_of       = whisper_full_default_params(CRISPASR_SAMPLING_GREEDY).greedy.best_of;
    int32_t beam_size     = whisper_full_default_params(CRISPASR_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
    int32_t audio_ctx     = 0;

    float word_thold      =  0.01f;
    float entropy_thold   =  2.40f;
    float logprob_thold   = -1.00f;
    float no_speech_thold =  0.6f;
    float grammar_penalty = 100.0f;
    float temperature     = 0.0f;
    float temperature_inc = 0.2f;

    bool debug_mode      = false;
    bool translate       = false;
    bool detect_language = false;
    bool diarize         = false;
    bool tinydiarize     = false;
    bool split_on_word   = false;
    bool no_fallback     = false;
    bool output_txt      = false;
    bool output_vtt      = false;
    bool output_srt      = false;
    bool output_wts      = false;
    bool output_csv      = false;
    bool output_jsn      = false;
    bool output_jsn_full = false;
    bool output_lrc      = false;
    bool no_prints       = false;
    bool print_special   = false;
    bool print_colors    = false;
    bool print_confidence= false;
    bool print_progress  = false;
    bool no_timestamps   = false;
    bool log_score       = false;
    bool use_gpu         = true;
    bool flash_attn      = true;
    int32_t gpu_device   = 0;
    bool suppress_nst    = false;
    bool carry_initial_prompt = false;

    std::string language  = "en";
    std::string prompt;
    std::string font_path = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf";
    std::string model     = "models/ggml-base.en.bin";
    std::string grammar;
    std::string grammar_rule;

    // [TDRZ] speaker turn string
    std::string tdrz_speaker_turn = " [SPEAKER_TURN]"; // TODO: set from command line

    // A regular expression that matches tokens to suppress
    std::string suppress_regex;

    std::string openvino_encode_device = "CPU";

    std::string dtw = "";

    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};

    grammar_parser::parse_state grammar_parsed;

    // Voice Activity Detection (VAD) parameters
    bool        vad           = false;
    std::string vad_model     = "";
    float       vad_threshold = 0.5f;
    int         vad_min_speech_duration_ms = 250;
    int         vad_min_silence_duration_ms = 100;
    float       vad_max_speech_duration_s = FLT_MAX;
    int         vad_speech_pad_ms = 30;
    float       vad_samples_overlap = 0.1f;
};
#endif // moved to whisper_params.h

static void whisper_print_usage(int argc, char** argv, const whisper_params& params);

static char* whisper_param_turn_lowercase(char* in) {
    int string_len = strlen(in);
    for (int i = 0; i < string_len; i++) {
        *(in + i) = tolower((unsigned char)*(in + i));
    }
    return in;
}

static char* requires_value_error(const std::string& arg) {
    fprintf(stderr, "error: argument %s requires value\n", arg.c_str());
    exit(0);
}

static bool parse_auto_quant_spec(const std::string& spec, std::string& base, std::string& quant) {
    const size_t pos = spec.find(':');
    if (pos == std::string::npos)
        return false;
    const std::string prefix = spec.substr(0, pos);
    if (prefix != "auto" && prefix != "default")
        return false;
    const std::string suffix = spec.substr(pos + 1);
    if (suffix.empty())
        return false;
    base = prefix;
    quant = suffix;
    return true;
}

static bool is_auto_model_arg(const std::string& model) {
    return model == "auto" || model == "default";
}

static void print_resolve_preview(const char* label, const CrispasrResolvePreview& preview) {
    fprintf(stderr, "%s:\n", label);
    fprintf(stderr, "  requested: %s\n", preview.requested.c_str());
    if (!preview.backend.empty())
        fprintf(stderr, "  backend:   %s\n", preview.backend.c_str());
    if (preview.unresolved) {
        fprintf(stderr, "  status:    unresolved\n");
        return;
    }
    if (preview.matched_registry) {
        fprintf(stderr, "  registry:  %s\n", preview.filename.c_str());
        fprintf(stderr, "  url:       %s\n", preview.url.c_str());
        if (!preview.approx_size.empty())
            fprintf(stderr, "  size:      %s\n", preview.approx_size.c_str());
    }
    if (preview.exists_locally) {
        fprintf(stderr, "  status:    cached/local\n");
    } else if (preview.would_download) {
        fprintf(stderr, "  status:    would download\n");
    } else {
        fprintf(stderr, "  status:    resolved\n");
    }
    fprintf(stderr, "  path:      %s\n", preview.resolved_path.c_str());
}

static bool whisper_params_parse_arg_general(int argc, char** argv, int& i, whisper_params& params) {
    std::string arg = argv[i];
#define ARGV_NEXT (((i + 1) < argc) ? argv[++i] : requires_value_error(arg))

    if (arg == "-t" || arg == "--threads") {
        params.n_threads = std::stoi(ARGV_NEXT);
    } else if (arg == "-p" || arg == "--processors") {
        params.n_processors = std::stoi(ARGV_NEXT);
    } else if (arg == "-ot" || arg == "--offset-t") {
        params.offset_t_ms = std::stoi(ARGV_NEXT);
    } else if (arg == "-on" || arg == "--offset-n") {
        params.offset_n = std::stoi(ARGV_NEXT);
    } else if (arg == "-d" || arg == "--duration") {
        params.duration_ms = std::stoi(ARGV_NEXT);
    } else if (arg == "-mc" || arg == "--max-context") {
        params.max_context = std::stoi(ARGV_NEXT);
    } else if (arg == "-ml" || arg == "--max-len") {
        params.max_len = std::stoi(ARGV_NEXT);
    } else if (arg == "-sp" || arg == "--split-on-punct") {
        params.split_on_punct = true;
    } else if (arg == "-bo" || arg == "--best-of") {
        params.best_of = std::stoi(ARGV_NEXT);
    } else if (arg == "-bs" || arg == "--beam-size") {
        params.beam_size = std::stoi(ARGV_NEXT);
    } else if (arg == "-ac" || arg == "--audio-ctx") {
        params.audio_ctx = std::stoi(ARGV_NEXT);
    } else if (arg == "-wt" || arg == "--word-thold") {
        params.word_thold = std::stof(ARGV_NEXT);
    } else if (arg == "-et" || arg == "--entropy-thold") {
        params.entropy_thold = std::stof(ARGV_NEXT);
    } else if (arg == "-lpt" || arg == "--logprob-thold") {
        params.logprob_thold = std::stof(ARGV_NEXT);
    } else if (arg == "-nth" || arg == "--no-speech-thold") {
        params.no_speech_thold = std::stof(ARGV_NEXT);
    } else if (arg == "-tp" || arg == "--temperature") {
        params.temperature = std::stof(ARGV_NEXT);
    } else if (arg == "--seed") {
        params.seed = (uint64_t)std::stoull(ARGV_NEXT);
    } else if (arg == "-tpi" || arg == "--temperature-inc") {
        params.temperature_inc = std::stof(ARGV_NEXT);
    } else if (arg == "-debug" || arg == "--debug-mode") {
        params.debug_mode = true;
    } else if (arg == "-tr" || arg == "--translate") {
        params.translate = true;
    } else if (arg == "-di" || arg == "--diarize") {
        params.diarize = true;
    } else if (arg == "-tdrz" || arg == "--tinydiarize") {
        params.tinydiarize = true;
    } else if (arg == "-sow" || arg == "--split-on-word") {
        params.split_on_word = true;
    } else if (arg == "-nf" || arg == "--no-fallback") {
        params.no_fallback = true;
    } else if (arg == "-f" || arg == "--file") {
        params.fname_inp.emplace_back(ARGV_NEXT);
    } else if (arg == "-ng" || arg == "--no-gpu") {
        params.use_gpu = false;
    } else if (arg == "-fa" || arg == "--flash-attn") {
        params.flash_attn = true;
    } else if (arg == "-nfa" || arg == "--no-flash-attn") {
        params.flash_attn = false;
    } else if (arg == "-sns" || arg == "--suppress-nst") {
        params.suppress_nst = true;
    } else if (arg == "--suppress-regex") {
        params.suppress_regex = ARGV_NEXT;
    } else if (arg == "--grammar") {
        params.grammar = ARGV_NEXT;
    } else if (arg == "--grammar-rule") {
        params.grammar_rule = ARGV_NEXT;
    } else if (arg == "--grammar-penalty") {
        params.grammar_penalty = std::stof(ARGV_NEXT);
    } else {
        return false;
    }
    return true;
#undef ARGV_NEXT
}

static bool whisper_params_parse_arg_output(int argc, char** argv, int& i, whisper_params& params) {
    std::string arg = argv[i];
#define ARGV_NEXT (((i + 1) < argc) ? argv[++i] : requires_value_error(arg))

    if (arg == "-otxt" || arg == "--output-txt") {
        params.output_txt = true;
    } else if (arg == "-ovtt" || arg == "--output-vtt") {
        params.output_vtt = true;
    } else if (arg == "-osrt" || arg == "--output-srt") {
        params.output_srt = true;
    } else if (arg == "-owts" || arg == "--output-words") {
        params.output_wts = true;
    } else if (arg == "-olrc" || arg == "--output-lrc") {
        params.output_lrc = true;
    } else if (arg == "-fp" || arg == "--font-path") {
        params.font_path = ARGV_NEXT;
    } else if (arg == "-ocsv" || arg == "--output-csv") {
        params.output_csv = true;
    } else if (arg == "-oj" || arg == "--output-json") {
        params.output_jsn = true;
    } else if (arg == "-ojf" || arg == "--output-json-full") {
        params.output_jsn_full = params.output_jsn = true;
    } else if (arg == "-of" || arg == "--output-file") {
        params.fname_out.emplace_back(ARGV_NEXT);
    } else if (arg == "-np" || arg == "--no-prints") {
        params.no_prints = true;
    } else if (arg == "-v" || arg == "--verbose") {
        params.verbose = true;
#if defined(_WIN32)
        _putenv_s("GGML_VK_PIPELINE_CACHE_DEBUG", "1");
        _putenv_s("GGML_CUDA_DEBUG", "1");
#else
        setenv("GGML_VK_PIPELINE_CACHE_DEBUG", "1", /*overwrite=*/0);
        setenv("GGML_CUDA_DEBUG", "1", /*overwrite=*/0);
#endif
    } else if (arg == "-ps" || arg == "--print-special") {
        params.print_special = true;
    } else if (arg == "-pc" || arg == "--print-colors") {
        params.print_colors = true;
    } else if (arg == "--print-confidence") {
        params.print_confidence = true;
    } else if (arg == "-pp" || arg == "--print-progress") {
        params.print_progress = true;
    } else if (arg == "-nt" || arg == "--no-timestamps") {
        params.no_timestamps = true;
    } else {
        return false;
    }
    return true;
#undef ARGV_NEXT
}

static bool whisper_params_parse_arg_backend_vad(int argc, char** argv, int& i, whisper_params& params) {
    std::string arg = argv[i];
#define ARGV_NEXT (((i + 1) < argc) ? argv[++i] : requires_value_error(arg))

    if (arg == "-l" || arg == "--language") {
        params.language = whisper_param_turn_lowercase(ARGV_NEXT);
    } else if (arg == "-dl" || arg == "--detect-language") {
        params.detect_language = true;
    } else if (arg == "--prompt") {
        params.prompt = ARGV_NEXT;
    } else if (arg == "--ask") {
        params.ask = ARGV_NEXT;
    } else if (arg == "--carry-initial-prompt") {
        params.carry_initial_prompt = true;
    } else if (arg == "-m" || arg == "--model") {
        params.model = ARGV_NEXT;
        std::string auto_base;
        std::string auto_quant;
        if (params.model_quant.empty() && parse_auto_quant_spec(params.model, auto_base, auto_quant)) {
            params.model = auto_base;
            params.model_quant = auto_quant;
        }
    } else if (arg == "--model-quant") {
        params.model_quant = ARGV_NEXT;
    } else if (arg == "-oved" || arg == "--ov-e-device") {
        params.openvino_encode_device = ARGV_NEXT;
    } else if (arg == "-dtw" || arg == "--dtw") {
        params.dtw = ARGV_NEXT;
    } else if (arg == "-ls" || arg == "--log-score") {
        params.log_score = true;
    } else if (arg == "-dev" || arg == "--device") {
        params.gpu_device = std::stoi(ARGV_NEXT);
        {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%d", params.gpu_device);
#if defined(_WIN32)
            if (!std::getenv("GGML_VK_VISIBLE_DEVICES"))
                _putenv_s("GGML_VK_VISIBLE_DEVICES", buf);
            if (!std::getenv("CUDA_VISIBLE_DEVICES"))
                _putenv_s("CUDA_VISIBLE_DEVICES", buf);
#else
            setenv("GGML_VK_VISIBLE_DEVICES", buf, 0);
            setenv("CUDA_VISIBLE_DEVICES", buf, 0);
#endif
        }
    } else if (arg == "--gpu-backend") {
        params.gpu_backend = ARGV_NEXT;
    } else if (arg == "--backend") {
        params.backend = ARGV_NEXT;
    } else if (arg == "-sl" || arg == "--source-lang") {
        params.source_lang = whisper_param_turn_lowercase(ARGV_NEXT);
    } else if (arg == "-tl" || arg == "--target-lang") {
        params.target_lang = whisper_param_turn_lowercase(ARGV_NEXT);
    } else if (arg == "--no-punctuation") {
        params.punctuation = false;
    } else if (arg == "--punc-model") {
        params.punc_model = ARGV_NEXT;
    } else if (arg == "--truecase-model") {
        params.truecase_model = ARGV_NEXT;
    } else if (arg == "--flush-after") {
        params.flush_after = std::stoi(ARGV_NEXT);
    } else if (arg == "-am" || arg == "--aligner-model") {
        params.aligner_model = ARGV_NEXT;
    } else if (arg == "--force-aligner" || arg == "-falign") {
        params.force_aligner = true;
    } else if (arg == "--no-auto-aligner") {
        params.no_auto_aligner = true;
    } else if (arg == "-n" || arg == "--max-new-tokens") {
        params.max_new_tokens = std::stoi(ARGV_NEXT);
    } else if (arg == "--frequency-penalty") {
        params.frequency_penalty = std::stof(ARGV_NEXT);
    } else if (arg == "-ck" || arg == "--chunk-seconds") {
        params.chunk_seconds = std::stoi(ARGV_NEXT);
        params.chunk_seconds_explicit = true;
    } else if (arg == "--chunk-overlap") {
        params.chunk_overlap_seconds = std::stof(ARGV_NEXT);
    } else if (arg == "--lcs-dedup") {
        std::string v = ARGV_NEXT;
        if (v != "auto" && v != "on" && v != "off") {
            fprintf(stderr, "crispasr: --lcs-dedup must be one of {auto|on|off} (got '%s')\n", v.c_str());
            return false;
        }
        params.lcs_dedup = std::move(v);
    } else if (arg == "--lcs-min-length") {
        const int v = std::stoi(ARGV_NEXT);
        if (v < 1) {
            fprintf(stderr, "crispasr: --lcs-min-length must be >= 1 (got %d)\n", v);
            return false;
        }
        params.lcs_min_length = v;
    } else if (arg == "--hotwords") {
        params.hotwords = ARGV_NEXT;
    } else if (arg == "--hotwords-file") {
        // Read one hotword per line from a file
        std::string path = ARGV_NEXT;
        FILE* f = fopen(path.c_str(), "r");
        if (f) {
            char line[1024];
            std::string acc;
            while (fgets(line, sizeof(line), f)) {
                std::string s(line);
                while (!s.empty() && (s.back() == '\n' || s.back() == '\r'))
                    s.pop_back();
                if (!s.empty()) {
                    if (!acc.empty())
                        acc += ",";
                    acc += s;
                }
            }
            fclose(f);
            params.hotwords = acc;
        } else {
            fprintf(stderr, "warning: cannot open hotwords file '%s'\n", path.c_str());
        }
    } else if (arg == "--hotwords-boost") {
        params.hotwords_boost = std::stof(ARGV_NEXT);
    } else if (arg == "--warmup") {
        params.warmup = true;
    } else if (arg == "--parakeet-decoder") {
        params.parakeet_decoder = ARGV_NEXT;
    } else if (arg == "--lid-backend") {
        params.lid_backend = ARGV_NEXT;
    } else if (arg == "--lid-model") {
        params.lid_model = ARGV_NEXT;
    } else if (arg == "--lid-on-transcript") {
        params.lid_on_transcript = ARGV_NEXT;
    } else if (arg == "--diarize-method") {
        params.diarize_method = ARGV_NEXT;
    } else if (arg == "--sherpa-bin") {
        params.sherpa_bin = ARGV_NEXT;
    } else if (arg == "--sherpa-segment-model") {
        params.sherpa_segment_model = ARGV_NEXT;
    } else if (arg == "--sherpa-embedding-model") {
        params.sherpa_embedding_model = ARGV_NEXT;
    } else if (arg == "--sherpa-num-clusters") {
        params.sherpa_num_clusters = std::stoi(ARGV_NEXT);
    } else if (arg == "--speaker-db") {
        params.speaker_db = ARGV_NEXT;
    } else if (arg == "--enroll-speaker") {
        params.enroll_speaker = ARGV_NEXT;
    } else if (arg == "--titanet-model" || arg == "--spk-model") {
        params.titanet_model = ARGV_NEXT;
    } else if (arg == "--speaker-threshold" || arg == "-st") {
        params.speaker_threshold = std::stof(ARGV_NEXT);
    } else if (arg == "--diarize-embedder") {
        params.diarize_embedder = ARGV_NEXT;
    } else if (arg == "--diarize-cluster-threshold") {
        params.diarize_cluster_threshold = std::stof(ARGV_NEXT);
    } else if (arg == "--diarize-max-speakers") {
        params.diarize_max_speakers = std::stoi(ARGV_NEXT);
    } else if (arg == "--cache-dir") {
        params.cache_dir = ARGV_NEXT;
    } else if (arg == "--alt") {
        params.show_alternatives = true;
    } else if (arg == "--alt-n") {
        params.n_alternatives = std::stoi(ARGV_NEXT);
    } else if (arg == "--stream") {
        params.stream = true;
    } else if (arg == "--mic") {
        params.mic = true;
        params.stream = true;
    } else if (arg == "--live") {
        params.mic = true;
        params.stream = true;
        params.stream_continuous = true;
    } else if (arg == "--monitor") {
        params.stream_monitor = true;
    } else {
        return false;
    }
    return true;
#undef ARGV_NEXT
}

// Continuation of whisper_params_parse_arg_backend_vad — split off so the
// `else if` chain doesn't trip MSVC's C1061 "blocks nested too deeply"
// limit. MSVC parses `else if` as `else { if ... }`, so each branch adds
// +2 to the cumulative nesting depth tracked across the function; with
// 70+ branches the depth exceeds the 128-level limit. Two ~35-branch
// functions stay comfortably under it. The dispatch loop in
// `whisper_params_parse` calls both in sequence.
static bool whisper_params_parse_arg_streaming_tts(int argc, char** argv, int& i, whisper_params& params) {
    std::string arg = argv[i];
#define ARGV_NEXT (((i + 1) < argc) ? argv[++i] : requires_value_error(arg))

    if (arg == "--tts") {
        params.tts_text = ARGV_NEXT;
    } else if (arg == "--tts-output") {
        params.tts_output = ARGV_NEXT;
    } else if (arg == "--voice") {
        params.tts_voice = ARGV_NEXT;
    } else if (arg == "--tts-steps") {
        params.tts_steps = std::stoi(ARGV_NEXT);
        if (params.tts_steps < 1)
            params.tts_steps = 1;
        if (params.tts_steps > 100)
            params.tts_steps = 100;
    } else if (arg == "--codec-model") {
        params.tts_codec_model = ARGV_NEXT;
        std::string auto_base;
        std::string auto_quant;
        if (params.tts_codec_quant.empty() && parse_auto_quant_spec(params.tts_codec_model, auto_base, auto_quant)) {
            params.tts_codec_model = auto_base;
            params.tts_codec_quant = auto_quant;
        }
    } else if (arg == "--codec-quant") {
        params.tts_codec_quant = ARGV_NEXT;
    } else if (arg == "--ref-text") {
        params.tts_ref_text = ARGV_NEXT;
    } else if (arg == "--instruct") {
        params.tts_instruct = ARGV_NEXT;
    } else if (arg == "--voice-dir") {
        params.tts_voice_dir = ARGV_NEXT;
    } else if (arg == "--tts-max-input-chars") {
        params.tts_max_input_chars = std::stoi(ARGV_NEXT);
    } else if (arg == "--cors-origin") {
        params.server_cors_origin = ARGV_NEXT;
    } else if (arg == "--chat-model") {
        // Optional GGUF chat model — enables POST /v1/chat/completions
        // on the server. Independent of the primary --model so a
        // server can host ASR + chat off two different GGUFs.
        params.chat_model = ARGV_NEXT;
    } else if (arg == "--chat-ctx") {
        params.chat_n_ctx = std::stoi(ARGV_NEXT);
    } else if (arg == "--chat-gpu-layers") {
        params.chat_n_gpu_layers = std::stoi(ARGV_NEXT);
    } else if (arg == "--tts-trim-silence") {
        params.tts_trim_silence = true;
    } else if (arg == "--text") {
        params.text_input = ARGV_NEXT;
    } else if (arg == "--translate-max-tokens") {
        params.translate_max_tokens = std::stoi(ARGV_NEXT);
    } else if (arg == "-trsl" || arg == "--tr-sl" || arg == "--translate-source-lang") {
        params.translate_source_lang = whisper_param_turn_lowercase(ARGV_NEXT);
    } else if (arg == "-trtl" || arg == "--tr-tl" || arg == "--translate-target-lang") {
        params.translate_target_lang = whisper_param_turn_lowercase(ARGV_NEXT);
    } else if (arg == "--auto-download") {
        params.auto_download = true;
    } else if (arg == "--dry-run-resolve") {
        params.dry_run_resolve = true;
    } else if (arg == "--dry-run-ignore-cache") {
        params.dry_run_ignore_cache = true;
    } else if (arg == "--server") {
        params.server = true;
    } else if (arg == "--host") {
        params.server_host = ARGV_NEXT;
    } else if (arg == "--port") {
        params.server_port = std::stoi(ARGV_NEXT);
    } else if (arg == "--api-keys") {
        params.server_api_keys = ARGV_NEXT;
    } else if (arg == "--stream-step") {
        params.stream_step_ms = std::stoi(ARGV_NEXT);
    } else if (arg == "--stream-length") {
        params.stream_length_ms = std::stoi(ARGV_NEXT);
    } else if (arg == "--stream-keep") {
        params.stream_keep_ms = std::stoi(ARGV_NEXT);
    } else if (arg == "--stream-json") {
        // Issue #84: machine-readable JSON-Lines streaming events.
        params.stream_json = true;
    } else if (arg == "--stream-final-on-silence-ms") {
        params.stream_final_silence_ms = std::stoi(ARGV_NEXT);
    } else if (arg == "--stream-vad-merge-gap-ms") {
        params.stream_vad_merge_gap_ms = std::stoi(ARGV_NEXT);
        if (params.stream_vad_merge_gap_ms < 0) {
            fprintf(stderr, "crispasr: --stream-vad-merge-gap-ms must be >= 0\n");
            exit(2);
        }
    } else if (arg == "--stream-partial-decode-ms") {
        params.stream_partial_decode_ms = std::stoi(ARGV_NEXT);
        if (params.stream_partial_decode_ms < 0) {
            fprintf(stderr, "crispasr: --stream-partial-decode-ms must be >= 0\n");
            exit(2);
        }
    } else if (arg == "--stream-punc") {
        std::string mode = ARGV_NEXT;
        if (!crispasr_stream_punc_mode_valid(mode)) {
            fprintf(stderr, "crispasr: --stream-punc must be 'off', 'final', or 'partial' (got '%s')\n", mode.c_str());
            exit(2);
        }
        params.stream_punc = mode;
    } else if (arg == "--stream-final-mode") {
        std::string mode = ARGV_NEXT;
        if (mode != "redecode" && mode != "prefix") {
            fprintf(stderr, "crispasr: --stream-final-mode must be 'redecode' or 'prefix' (got '%s')\n", mode.c_str());
            exit(2);
        }
        params.stream_final_mode = mode;
    } else if (arg == "--stream-utterance-max-sec") {
        params.stream_utterance_max_sec = std::stoi(ARGV_NEXT);
    } else if (arg == "--firered-vad-debug") {
        // Issue #84: opt-in debug dump from src/firered_vad.cpp.
        // Plumbed via env var so src/ doesn't have to learn about
        // whisper_params; firered_vad reads the env on each call.
        params.firered_vad_debug = true;
#ifdef _WIN32
        _putenv_s("CRISPASR_FIRERED_VAD_DEBUG", "1");
#else
        setenv("CRISPASR_FIRERED_VAD_DEBUG", "1", 1);
#endif
    } else if (arg == "--list-backends") {
        crispasr_print_backend_matrix();
        exit(0);
    } else if (arg == "--list-backends-json") {
        crispasr_print_backend_matrix_json();
        exit(0);
    } else if (arg == "--vad") {
        params.vad = true;
    } else if (arg == "-vm" || arg == "--vad-model") {
        params.vad_model = ARGV_NEXT;
    } else if (arg == "-vt" || arg == "--vad-threshold") {
        params.vad_threshold = std::stof(ARGV_NEXT);
        // Issue #83 follow-up: mark explicit so per-model auto-tuning
        // (whisper-vad-encdec lowers default to 0.30) doesn't override.
        params.vad_threshold_explicit = true;
    } else if (arg == "-vspd" || arg == "--vad-min-speech-duration-ms") {
        params.vad_min_speech_duration_ms = std::stoi(ARGV_NEXT);
    } else if (arg == "-vsd" || arg == "--vad-min-silence-duration-ms") {
        params.vad_min_silence_duration_ms = std::stoi(ARGV_NEXT);
    } else if (arg == "-vmsd" || arg == "--vad-max-speech-duration-s") {
        params.vad_max_speech_duration_s = std::stof(ARGV_NEXT);
    } else if (arg == "-vp" || arg == "--vad-speech-pad-ms") {
        params.vad_speech_pad_ms = std::stoi(ARGV_NEXT);
    } else if (arg == "-vo" || arg == "--vad-samples-overlap") {
        params.vad_samples_overlap = std::stof(ARGV_NEXT);
    } else if (arg == "--vad-stitch") {
        params.vad_stitch = true;
    } else {
        return false;
    }
    return true;
#undef ARGV_NEXT
}

static bool whisper_params_parse(int argc, char** argv, whisper_params& params) {
    if (const char* env_device = std::getenv("CRISPASR_ARG_DEVICE")) {
        params.gpu_device = std::stoi(env_device);
    }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-") {
            params.fname_inp.push_back(arg);
            continue;
        }

        if (arg[0] != '-') {
            params.fname_inp.push_back(arg);
            continue;
        }

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }

        if (arg == "--version") {
            crispasr_print_build_info(stdout);
            exit(0);
        }

        if (arg == "--diagnostics" || arg == "--diag") {
            crispasr_print_full_diagnostics(stderr);
            exit(0);
        }

        if (whisper_params_parse_arg_general(argc, argv, i, params)) {
            continue;
        }
        if (whisper_params_parse_arg_output(argc, argv, i, params)) {
            continue;
        }
        if (whisper_params_parse_arg_backend_vad(argc, argv, i, params)) {
            continue;
        }
        if (whisper_params_parse_arg_streaming_tts(argc, argv, i, params)) {
            continue;
        }

        fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    return true;
}

static void whisper_print_usage(int /*argc*/, char** argv, const whisper_params& params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options] file0 file1 ...\n", argv[0]);
    fprintf(stderr, "supported audio formats: flac, mp3, ogg, wav\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,        --help                 [default] show this help message and exit\n");
    fprintf(stderr, "             --version              print build info (version, git SHA, backends) and exit\n");
    fprintf(stderr, "             --diagnostics          full diagnostics (build + env + GPU enumeration) and exit\n");
    fprintf(stderr, "  -t N,      --threads N            [%-7d] number of threads to use during computation\n",
            params.n_threads);
    fprintf(stderr, "  -p N,      --processors N         [%-7d] number of processors to use during computation\n",
            params.n_processors);
    fprintf(stderr, "  -ot N,     --offset-t N           [%-7d] time offset in milliseconds\n", params.offset_t_ms);
    fprintf(stderr, "  -on N,     --offset-n N           [%-7d] segment index offset\n", params.offset_n);
    fprintf(stderr, "  -d  N,     --duration N           [%-7d] duration of audio to process in milliseconds\n",
            params.duration_ms);
    fprintf(stderr, "  -mc N,     --max-context N        [%-7d] maximum number of text context tokens to store\n",
            params.max_context);
    fprintf(stderr, "  -sp,       --split-on-punct       [%-7s] split subtitle lines at sentence-ending punctuation\n",
            params.split_on_punct ? "true" : "false");
    fprintf(stderr, "  -ml N,     --max-len N            [%-7d] maximum segment length in characters\n",
            params.max_len);
    fprintf(stderr, "  -sow,      --split-on-word        [%-7s] split on word rather than on token\n",
            params.split_on_word ? "true" : "false");
    fprintf(stderr, "  -bo N,     --best-of N            [%-7d] number of best candidates to keep\n", params.best_of);
    fprintf(stderr, "  -bs N,     --beam-size N          [%-7d] beam size for beam search\n", params.beam_size);
    fprintf(stderr, "  -ac N,     --audio-ctx N          [%-7d] audio context size (0 - all)\n", params.audio_ctx);
    fprintf(stderr, "  -wt N,     --word-thold N         [%-7.2f] word timestamp probability threshold\n",
            params.word_thold);
    fprintf(stderr, "  -et N,     --entropy-thold N      [%-7.2f] entropy threshold for decoder fail\n",
            params.entropy_thold);
    fprintf(stderr, "  -lpt N,    --logprob-thold N      [%-7.2f] log probability threshold for decoder fail\n",
            params.logprob_thold);
    fprintf(stderr, "  -nth N,    --no-speech-thold N    [%-7.2f] no speech threshold\n", params.no_speech_thold);
    fprintf(stderr, "  -tp,       --temperature N        [%-7.2f] The sampling temperature, between 0 and 1\n",
            params.temperature);
    fprintf(stderr, "             --seed N               [%-7d] RNG seed for sampling (0 = non-deterministic)\n",
            (int)params.seed);
    fprintf(stderr, "  -tpi,      --temperature-inc N    [%-7.2f] The increment of temperature, between 0 and 1\n",
            params.temperature_inc);
    fprintf(stderr, "  -debug,    --debug-mode           [%-7s] enable debug mode (eg. dump log_mel)\n",
            params.debug_mode ? "true" : "false");
    fprintf(stderr, "  -tr,       --translate            [%-7s] translate from source language to english\n",
            params.translate ? "true" : "false");
    fprintf(stderr, "  -di,       --diarize              [%-7s] stereo audio diarization\n",
            params.diarize ? "true" : "false");
    fprintf(stderr, "  -tdrz,     --tinydiarize          [%-7s] enable tinydiarize (requires a tdrz model)\n",
            params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -nf,       --no-fallback          [%-7s] do not use temperature fallback while decoding\n",
            params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -otxt,     --output-txt           [%-7s] output result in a text file\n",
            params.output_txt ? "true" : "false");
    fprintf(stderr, "  -ovtt,     --output-vtt           [%-7s] output result in a vtt file\n",
            params.output_vtt ? "true" : "false");
    fprintf(stderr, "  -osrt,     --output-srt           [%-7s] output result in a srt file\n",
            params.output_srt ? "true" : "false");
    fprintf(stderr, "  -olrc,     --output-lrc           [%-7s] output result in a lrc file\n",
            params.output_lrc ? "true" : "false");
    fprintf(stderr, "  -owts,     --output-words         [%-7s] output script for generating karaoke video\n",
            params.output_wts ? "true" : "false");
    fprintf(stderr, "  -fp,       --font-path            [%-7s] path to a monospace font for karaoke video\n",
            params.font_path.c_str());
    fprintf(stderr, "  -ocsv,     --output-csv           [%-7s] output result in a CSV file\n",
            params.output_csv ? "true" : "false");
    fprintf(stderr, "  -oj,       --output-json          [%-7s] output result in a JSON file\n",
            params.output_jsn ? "true" : "false");
    fprintf(stderr, "  -ojf,      --output-json-full     [%-7s] include more information in the JSON file\n",
            params.output_jsn_full ? "true" : "false");
    fprintf(stderr, "  -of FNAME, --output-file FNAME    [%-7s] output file path (without file extension)\n", "");
    fprintf(stderr, "  -np,       --no-prints            [%-7s] do not print anything other than the results\n",
            params.no_prints ? "true" : "false");
    fprintf(
        stderr,
        "  -v,        --verbose              [%-7s] verbose debug: model/cache paths, device pick, per-stage timings\n",
        params.verbose ? "true" : "false");
    fprintf(stderr, "  -ps,       --print-special        [%-7s] print special tokens\n",
            params.print_special ? "true" : "false");
    fprintf(stderr, "  -pc,       --print-colors         [%-7s] print colors\n",
            params.print_colors ? "true" : "false");
    fprintf(stderr, "             --print-confidence     [%-7s] print confidence\n",
            params.print_confidence ? "true" : "false");
    fprintf(stderr, "  -pp,       --print-progress       [%-7s] print progress\n",
            params.print_progress ? "true" : "false");
    fprintf(stderr, "  -nt,       --no-timestamps        [%-7s] do not print timestamps\n",
            params.no_timestamps ? "true" : "false");
    fprintf(stderr, "  -l LANG,   --language LANG        [%-7s] spoken language ('auto' for auto-detect)\n",
            params.language.c_str());
    fprintf(stderr, "  -dl,       --detect-language      [%-7s] exit after automatically detecting language\n",
            params.detect_language ? "true" : "false");
    fprintf(stderr, "             --prompt PROMPT        [%-7s] initial prompt (max n_text_ctx/2 tokens)\n",
            params.prompt.c_str());
    fprintf(stderr, "             --carry-initial-prompt [%-7s] always prepend initial prompt\n",
            params.carry_initial_prompt ? "true" : "false");
    fprintf(stderr, "  -m FNAME,  --model FNAME          [%-7s] model path\n", params.model.c_str());
    fprintf(stderr, "             --model-quant Q        [%-7s] preferred quant for registry / auto model resolution\n",
            params.model_quant.c_str());
    fprintf(stderr, "  -f FNAME,  --file FNAME           [%-7s] input audio file path\n", "");
    fprintf(stderr, "  -oved D,   --ov-e-device DNAME    [%-7s] the OpenVINO device used for encode inference\n",
            params.openvino_encode_device.c_str());
    fprintf(stderr, "  -dtw MODEL --dtw MODEL            [%-7s] compute token-level timestamps\n", params.dtw.c_str());
    fprintf(stderr, "  -ls,       --log-score            [%-7s] log best decoder scores of tokens\n",
            params.log_score ? "true" : "false");
    fprintf(stderr, "  -ng,       --no-gpu               [%-7s] disable GPU\n", params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -dev N,    --device N             [%-7d] GPU device ID (default: 0)\n", params.gpu_device);
    fprintf(stderr,
            "  --gpu-backend NAME                [%-7s] force GPU backend: cuda|vulkan|metal|cpu (default: auto)\n",
            params.gpu_backend.empty() ? "auto" : params.gpu_backend.c_str());
    fprintf(stderr, "  -fa,       --flash-attn           [%-7s] enable flash attention\n",
            params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -nfa,      --no-flash-attn        [%-7s] disable flash attention\n",
            params.flash_attn ? "false" : "true");
    fprintf(stderr, "  -sns,      --suppress-nst         [%-7s] suppress non-speech tokens\n",
            params.suppress_nst ? "true" : "false");
    fprintf(stderr, "  --suppress-regex REGEX            [%-7s] regular expression matching tokens to suppress\n",
            params.suppress_regex.c_str());
    fprintf(stderr, "  --grammar GRAMMAR                 [%-7s] GBNF grammar to guide decoding\n",
            params.grammar.c_str());
    fprintf(stderr, "  --grammar-rule RULE               [%-7s] top-level GBNF grammar rule name\n",
            params.grammar_rule.c_str());
    fprintf(stderr, "  --grammar-penalty N               [%-7.1f] scales down logits of nongrammar tokens\n",
            params.grammar_penalty);
    // crispasr backend dispatch
    fprintf(stderr, "\ncrispasr backend options (select a non-whisper model):\n");
    fprintf(stderr,
            "  --backend NAME                    [%-7s] backend: "
            "whisper|parakeet|canary|cohere|qwen3|qwen3-1.7b|mega-asr|voxtral|voxtral4b|granite\n",
            params.backend.c_str());
    fprintf(stderr, "  --list-backends                   list backends compiled into this binary and exit\n");
    fprintf(stderr, "  --list-backends-json              same as --list-backends but JSON-formatted, for tooling\n");
    fprintf(stderr, "  -sl LANG,  --source-lang LANG     [%-7s] source language (canary AST)\n",
            params.source_lang.c_str());
    fprintf(stderr, "  -tl LANG,  --target-lang LANG     [%-7s] target language (canary AST)\n",
            params.target_lang.c_str());
    fprintf(stderr, "             --no-punctuation       [%-7s] disable punctuation (canary, cohere)\n",
            params.punctuation ? "false" : "true");
    fprintf(stderr,
            "             --punc-model FNAME     [%-7s] punctuation GGUF: auto|firered|fullstop|punctuate-all\n",
            params.punc_model.c_str());
    fprintf(stderr, "             --truecase-model FNAME [%-7s] truecaser: auto (de) or path to .bin\n",
            params.truecase_model.c_str());
    fprintf(stderr, "             --flush-after N        [%-7d] flush SRT to stdout every N segments (0=all at end)\n",
            params.flush_after);
    fprintf(stderr, "  -am FNAME, --aligner-model FNAME  [%-7s] CTC aligner GGUF (LLM backends word timestamps)\n",
            params.aligner_model.c_str());
    fprintf(stderr,
            "  -falign,   --force-aligner        [%-7s] use the CTC aligner's word "
            "timestamps even when the backend produces native ones (issue #62)\n",
            params.force_aligner ? "true" : "false");
    fprintf(stderr,
            "             --no-auto-aligner      [%-7s] for --backend canary, skip the implicit "
            "`-am auto --force-aligner` default (SubtitleEdit #10775)\n",
            params.no_auto_aligner ? "true" : "false");
    fprintf(
        stderr,
        "  --lid-backend NAME                [%-7s] language-detect backend: whisper|silero|firered (for non-native "
        "backends)\n",
        params.lid_backend.c_str());
    fprintf(stderr, "  --lid-model FNAME                 [%-7s] optional LID model path (default ggml-tiny.bin)\n",
            params.lid_model.c_str());
    fprintf(stderr,
            "  --lid-on-transcript FNAME         [%-7s] post-ASR text LID. Path or "
            "'auto[:cld3|glotlid|lid-fasttext176]' (default cld3, auto-downloaded). "
            "Emits lang=<code>\\tconf=<x>\\tbackend=<n> to stderr.\n",
            params.lid_on_transcript.c_str());
    fprintf(stderr,
            "  --diarize-method NAME             [%-7s] diarize method: energy|xcorr|vad-turns|sherpa|pyannote|ecapa\n",
            params.diarize_method.c_str());
    fprintf(stderr,
            "                                             energy/xcorr: stereo channel split; vad-turns: gap-based "
            "turn proxy (mono)\n");
    fprintf(stderr,
            "                                             pyannote: native GGUF segmentation only (experimental — "
            "no speaker embeddings or clustering, IDs are not stable across long files; for reliable diarization "
            "use sherpa/ecapa)\n");
    fprintf(stderr, "                                             sherpa/ecapa: external sherpa-onnx subprocess with "
                    "segmentation + speaker embedding + clustering\n");
    fprintf(stderr,
            "  --diarize-embedder MODEL          [%-7s] speaker-embedding model used to cluster pyannote local "
            "tracks into globally stable speaker IDs. Pluggable; known aliases: 'auto' / 'titanet' (192-d "
            "TitaNet-Large) and 'indextts' / 'indextts-bigvgan' / 'ecapa' (512-d IndexTTS-BigVGAN ECAPA-TDNN). "
            "Pass a .gguf path to load directly. When unset, --diarize-method pyannote labels are local to each "
            "forward pass (#107).\n",
            params.diarize_embedder.empty() ? "off" : params.diarize_embedder.c_str());
    fprintf(stderr,
            "  --diarize-cluster-threshold X     [%-7.2f] cosine merge threshold for --diarize-embedder clustering "
            "(higher = more distinct clusters, lower = more merged)\n",
            params.diarize_cluster_threshold);
    fprintf(stderr, "  --diarize-max-speakers N          [%-7d] hard cap on cluster count for --diarize-embedder\n",
            params.diarize_max_speakers);
    fprintf(stderr,
            "  --sherpa-bin PATH                 [%-7s] sherpa-onnx-offline-speaker-diarization binary (default: in "
            "PATH)\n",
            params.sherpa_bin.c_str());
    fprintf(stderr, "  --sherpa-segment-model PATH       [%-7s] sherpa pyannote segmentation ONNX\n",
            params.sherpa_segment_model.c_str());
    fprintf(stderr, "  --sherpa-embedding-model PATH     [%-7s] sherpa speaker embedding ONNX\n",
            params.sherpa_embedding_model.c_str());
    fprintf(stderr, "  --sherpa-num-clusters N           [%-7d] sherpa cluster count (0 = auto)\n",
            params.sherpa_num_clusters);
    fprintf(stderr, "  --cache-dir DIR                   [%-7s] override auto-download cache directory\n",
            params.cache_dir.empty() ? "default" : params.cache_dir.c_str());
    fprintf(stderr, "  --auto-download                   [%-7s] auto-download missing models without prompting\n",
            params.auto_download ? "true" : "false");
    fprintf(stderr, "  --dry-run-resolve                 [%-7s] print resolved model / companion paths and exit\n",
            params.dry_run_resolve ? "true" : "false");
    fprintf(stderr, "  --dry-run-ignore-cache            [%-7s] dry-run as if cache were empty\n",
            params.dry_run_ignore_cache ? "true" : "false");
    fprintf(stderr, "  --alt                             [%-7s] show alternative token candidates with probabilities\n",
            params.show_alternatives ? "true" : "false");
    fprintf(stderr, "  --alt-n N                         [%-7d] number of alternatives per token\n",
            params.n_alternatives);
    fprintf(stderr, "  --stream                          [%-7s] streaming mode: read raw s16le PCM from stdin\n",
            params.stream ? "true" : "false");
    fprintf(stderr, "  --mic                             [%-7s] capture from default microphone (implies --stream)\n",
            params.mic ? "true" : "false");
    fprintf(stderr,
            "  --live                            [%-7s] continuous live transcription (implies --mic --stream)\n",
            params.stream_continuous ? "true" : "false");
    fprintf(stderr, "  --monitor                         [%-7s] show unicode progress symbols during streaming\n",
            params.stream_monitor ? "true" : "false");
    fprintf(stderr,
            "  --server                          [%-7s] run as HTTP server (persistent model, POST /inference)\n",
            params.server ? "true" : "false");
    fprintf(stderr, "  --host HOST                       [%-7s] server bind address\n", params.server_host.c_str());
    fprintf(stderr, "  --port PORT                       [%-7d] server port\n", params.server_port);
    fprintf(stderr, "  --api-keys K1,K2                  [%-7s] comma-separated server API keys\n",
            params.server_api_keys.empty() ? "" : "(set)");
    fprintf(stderr, "  --stream-step N                   [%-7d] chunk size in ms for streaming\n",
            params.stream_step_ms);
    fprintf(stderr,
            "  --stream-length N                 [%-7d] rolling context window cap in ms (true rolling buffer)\n",
            params.stream_length_ms);
    fprintf(stderr,
            "  --stream-keep N                   [%-7d] (legacy, no-op since #84) overlap to keep between chunks\n",
            params.stream_keep_ms);
    fprintf(stderr,
            "  --stream-json                     [%-7s] emit JSON-Lines partial/final/silence events on stdout\n",
            params.stream_json ? "true" : "false");
    fprintf(stderr,
            "  --stream-final-on-silence-ms N    [%-7d] trailing silence (ms) that promotes a partial to final\n",
            params.stream_final_silence_ms);
    fprintf(stderr,
            "  --stream-vad-merge-gap-ms N       [%-7d] JSON+VAD close-gap merge in ms; clamped below final silence\n",
            params.stream_vad_merge_gap_ms);
    fprintf(stderr,
            "  --stream-partial-decode-ms N      [%-7d] JSON+VAD minimum interval between partial ASR decodes; 0 = "
            "every step\n",
            params.stream_partial_decode_ms);
    fprintf(stderr, "  --stream-punc MODE                [%-7s] JSON+VAD FireRedPunc mode: off, final, or partial\n",
            params.stream_punc.c_str());
    fprintf(stderr,
            "  --stream-final-mode MODE          [%-7s] final.text source: 'redecode' (re-runs on the utterance "
            "PCM, best quality) or 'prefix' (LCP-accumulated, no extra encoder pass)\n",
            params.stream_final_mode.c_str());
    fprintf(stderr, "  --stream-utterance-max-sec N      [%-7d] cap on per-utterance PCM buffer in redecode mode\n",
            params.stream_utterance_max_sec);
    fprintf(stderr, "  --firered-vad-debug               [%-7s] enable FireRed VAD probability/fbank stderr dumps\n",
            params.firered_vad_debug ? "true" : "false");
    fprintf(stderr, "  -n N,      --max-new-tokens N     [%-7d] max new tokens for LLM backends\n",
            params.max_new_tokens);
    fprintf(stderr, "             --frequency-penalty F  [%-7.2f] penalize repeated generated tokens on AR backends\n",
            params.frequency_penalty);
    fprintf(stderr, "  -ck N,     --chunk-seconds N      [%-7d] fallback chunk size when VAD is disabled\n",
            params.chunk_seconds);
    fprintf(stderr, "             --chunk-overlap F      [%-7.1f] overlap context (sec) at chunk boundaries\n",
            params.chunk_overlap_seconds);
    fprintf(stderr,
            "             --lcs-dedup VAL        [%-7s] sub-word LCS dedup across chunk boundaries: auto|on|off\n",
            params.lcs_dedup.c_str());
    fprintf(stderr,
            "             --lcs-min-length N     [%-7d] minimum LCS length to act on (raise on long-silence audio)\n",
            params.lcs_min_length);
    fprintf(stderr, "             -m auto                        download a default model for the chosen backend\n");
    // Text-To-Speech (TTS) parameters — vibevoice and qwen3-tts backends
    fprintf(stderr, "\nText-to-speech (TTS) options:\n");
    fprintf(stderr,
            "             --tts \"TEXT\"            synthesise TEXT and write WAV to --tts-output (24 kHz mono)\n");
    fprintf(stderr, "             --tts-output FNAME      [%-7s] output WAV path (default: tts_output.wav)\n",
            params.tts_output.c_str());
    fprintf(stderr,
            "             --voice PATH            [%-7s] voice prompt: GGUF voice pack or reference WAV\n"
            "                                                 (.wav → 1.5B WAV cloning; .gguf → voice pack)\n",
            params.tts_voice.c_str());
    fprintf(stderr, "             --ref-text \"TEXT\"        reference transcription (qwen3-tts; required when --voice "
                    "is a WAV)\n");
    fprintf(stderr, "             --instruct \"TEXT\"        natural-language voice/style description "
                    "(qwen3-tts: VoiceDesign = voice description; CustomVoice = style control)\n");
    fprintf(stderr,
            "             --codec-model FNAME      codec / companion GGUF (defaults to sibling/cache/registry)\n");
    fprintf(stderr, "             --codec-quant Q          [%-7s] preferred quant for registry companion resolution\n",
            params.tts_codec_quant.c_str());
    fprintf(stderr, "             --voice-dir PATH         server: dir of <name>.wav (+ <name>.txt) or "
                    "<name>.gguf voice profiles for POST /v1/audio/speech\n");
    fprintf(stderr,
            "             --tts-max-input-chars N  [%-7d] server: cap on /v1/audio/speech `input` "
            "length (0 = no cap)\n",
            params.tts_max_input_chars);
    fprintf(stderr, "             --cors-origin ORIGIN     server: opt-in CORS for browser clients "
                    "('*' for any, or scheme://host[:port])\n");
    fprintf(stderr, "             --chat-model PATH        server: enable POST /v1/chat/completions backed by "
                    "this GGUF chat model (independent of --model)\n");
    fprintf(stderr,
            "             --chat-ctx N             [%-7d] server: context window in tokens for the chat model\n",
            params.chat_n_ctx);
    fprintf(stderr,
            "             --chat-gpu-layers N      [%-7d] server: GPU layers for the chat model "
            "(-1 = all, 0 = CPU only)\n",
            params.chat_n_gpu_layers);
    fprintf(stderr, "             --tts-steps N            [%-7d] DPM-Solver++ steps (10-20, vibevoice only)\n",
            params.tts_steps);
    fprintf(stderr, "             --tts-trim-silence       [%-7s] trim leading silence from TTS output\n",
            params.tts_trim_silence ? "true" : "false");
    // Text-to-text translation (m2m100)
    fprintf(stderr, "\nText-to-text translation (m2m100) options:\n");
    fprintf(stderr, "             --text \"TEXT\"           translate TEXT and write result to stdout "
                    "(use with --backend m2m100; pair with -sl / -tl)\n");
    fprintf(stderr, "             --tr-sl LANG / --tr-tl LANG\n"
                    "                                        translator-stage source/target language "
                    "(falls back to -sl / -tl); only needed for 2-stage pipelines where the primary "
                    "backend's -sl/-tl mean something else\n");
    fprintf(stderr, "             --translate-max-tokens N [%-7d] max output tokens for the translator stage\n",
            params.translate_max_tokens);
    // Voice Activity Detection (VAD) parameters
    fprintf(stderr, "\nVoice Activity Detection (VAD) options:\n");
    fprintf(stderr, "             --vad                           [%-7s] enable Voice Activity Detection (VAD)\n",
            params.vad ? "true" : "false");
    fprintf(stderr, "  -vm FNAME, --vad-model FNAME               [%-7s] VAD model (path, 'firered', or 'silero')\n",
            params.vad_model.c_str());
    fprintf(stderr, "  -vt N,     --vad-threshold N               [%-7.2f] VAD threshold for speech recognition\n",
            params.vad_threshold);
    fprintf(stderr, "  -vspd N,   --vad-min-speech-duration-ms  N [%-7d] VAD min speech duration (ms)\n",
            params.vad_min_speech_duration_ms);
    fprintf(stderr,
            "  -vsd N,    --vad-min-silence-duration-ms N [%-7d] VAD min silence duration (to split segments)\n",
            params.vad_min_silence_duration_ms);
    fprintf(stderr, "  -vmsd N,   --vad-max-speech-duration-s   N [%-7s] VAD max speech duration (auto-split longer)\n",
            params.vad_max_speech_duration_s == FLT_MAX ? std::string("FLT_MAX").c_str()
                                                        : std::to_string(params.vad_max_speech_duration_s).c_str());
    fprintf(stderr, "  -vp N,     --vad-speech-pad-ms           N [%-7d] VAD speech padding (extend segments)\n",
            params.vad_speech_pad_ms);
    fprintf(stderr,
            "  -vo N,     --vad-samples-overlap         N [%-7.2f] VAD samples overlap (seconds between segments)\n",
            params.vad_samples_overlap);
    fprintf(stderr, "\n");
}

struct whisper_print_user_data {
    const whisper_params* params;

    const std::vector<std::vector<float>>* pcmf32s;
    int progress_prev;
};

static std::string estimate_diarization_speaker(const std::vector<std::vector<float>>& pcmf32s, int64_t t0, int64_t t1,
                                                bool id_only = false) {
    std::string speaker = "";
    const int64_t n_samples = pcmf32s[0].size();

    const int64_t is0 = timestamp_to_sample(t0, n_samples, CRISPASR_SAMPLE_RATE);
    const int64_t is1 = timestamp_to_sample(t1, n_samples, CRISPASR_SAMPLE_RATE);

    double energy0 = 0.0f;
    double energy1 = 0.0f;

    for (int64_t j = is0; j < is1; j++) {
        energy0 += fabs(pcmf32s[0][j]);
        energy1 += fabs(pcmf32s[1][j]);
    }

    if (energy0 > 1.1 * energy1) {
        speaker = "0";
    } else if (energy1 > 1.1 * energy0) {
        speaker = "1";
    } else {
        speaker = "?";
    }

    //printf("is0 = %lld, is1 = %lld, energy0 = %f, energy1 = %f, speaker = %s\n", is0, is1, energy0, energy1, speaker.c_str());

    if (!id_only) {
        speaker.insert(0, "(speaker ");
        speaker.append(")");
    }

    return speaker;
}

static void whisper_print_progress_callback(struct whisper_context* /*ctx*/, struct whisper_state* /*state*/,
                                            int progress, void* user_data) {
    int progress_step = ((whisper_print_user_data*)user_data)->params->progress_step;
    int* progress_prev = &(((whisper_print_user_data*)user_data)->progress_prev);
    if (progress >= *progress_prev + progress_step) {
        *progress_prev += progress_step;
        fprintf(stderr, "%s: progress = %3d%%\n", __func__, progress);
    }
}

static void whisper_print_segment_callback(struct whisper_context* ctx, struct whisper_state* /*state*/, int n_new,
                                           void* user_data) {
    const auto& params = *((whisper_print_user_data*)user_data)->params;
    const auto& pcmf32s = *((whisper_print_user_data*)user_data)->pcmf32s;

    const int n_segments = whisper_full_n_segments(ctx);

    std::string speaker = "";

    int64_t t0 = 0;
    int64_t t1 = 0;

    // print the last n_new segments
    const int s0 = n_segments - n_new;

    if (s0 == 0) {
        printf("\n");
    }

    for (int i = s0; i < n_segments; i++) {
        if (!params.no_timestamps || params.diarize) {
            t0 = whisper_full_get_segment_t0(ctx, i);
            t1 = whisper_full_get_segment_t1(ctx, i);
        }

        if (!params.no_timestamps) {
            printf("[%s --> %s]  ", to_timestamp(t0).c_str(), to_timestamp(t1).c_str());
        }

        if (params.diarize && pcmf32s.size() == 2) {
            speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
        }

        if (params.print_colors) {
            for (int j = 0; j < whisper_full_n_tokens(ctx, i); ++j) {
                if (params.print_special == false) {
                    const whisper_token id = whisper_full_get_token_id(ctx, i, j);
                    if (id >= whisper_token_eot(ctx)) {
                        continue;
                    }
                }

                const char* text = whisper_full_get_token_text(ctx, i, j);
                const float p = whisper_full_get_token_p(ctx, i, j);

                const int n_colors = (int)k_colors.size();
                int raw_col = (int)(std::pow(p, 3) * float(n_colors));
                if (raw_col < 0)
                    raw_col = 0;
                if (raw_col > n_colors - 1)
                    raw_col = n_colors - 1;
                const int col = raw_col;

                printf("%s%s%s%s", speaker.c_str(), k_colors[col].c_str(), text, "\033[0m");
            }
        } else if (params.print_confidence) {
            for (int j = 0; j < whisper_full_n_tokens(ctx, i); ++j) {
                if (params.print_special == false) {
                    const whisper_token id = whisper_full_get_token_id(ctx, i, j);
                    if (id >= whisper_token_eot(ctx)) {
                        continue;
                    }
                }

                const char* text = whisper_full_get_token_text(ctx, i, j);
                const float p = whisper_full_get_token_p(ctx, i, j);

                int style_idx = 2; // High confidence - dim
                if (p < 0.33) {
                    style_idx = 0; // Low confidence - inverse (highlighted)
                } else if (p < 0.66) {
                    style_idx = 1; // Medium confidence - underlined
                }
                printf("%s%s%s%s", speaker.c_str(), k_styles[style_idx].c_str(), text, "\033[0m");
            }
        } else {
            const char* text = whisper_full_get_segment_text(ctx, i);

            printf("%s%s", speaker.c_str(), text);
        }

        if (params.tinydiarize) {
            if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                printf("%s", params.tdrz_speaker_turn.c_str());
            }
        }

        // with timestamps or speakers: each segment on new line
        if (!params.no_timestamps || params.diarize) {
            printf("\n");
        }

        fflush(stdout);
    }
}

// Collect whisper segments + tokens into the unified crispasr_segment
// vector. Called once per transcription, before any writer runs.
// Everything the output functions need lives on the vector afterwards so
// the writers are whisper_context-free (except output_json which still
// needs ctx for systeminfo/model metadata).
static std::vector<crispasr_segment> cli_whisper_collect_segments(struct whisper_context* ctx) {
    std::vector<crispasr_segment> out;
    const int n = whisper_full_n_segments(ctx);
    out.reserve(n);
    const whisper_token eot = whisper_token_eot(ctx);
    for (int i = 0; i < n; ++i) {
        crispasr_segment s;
        s.text = whisper_full_get_segment_text(ctx, i);
        s.t0 = whisper_full_get_segment_t0(ctx, i);
        s.t1 = whisper_full_get_segment_t1(ctx, i);
        s.speaker_turn_next = whisper_full_get_segment_speaker_turn_next(ctx, i);

        const int nt = whisper_full_n_tokens(ctx, i);
        s.tokens.reserve(nt);
        for (int j = 0; j < nt; ++j) {
            const auto d = whisper_full_get_token_data(ctx, i, j);
            crispasr_token t;
            t.id = d.id;
            t.text = whisper_token_to_str(ctx, d.id);
            t.confidence = d.p;
            t.t0 = d.t0;
            t.t1 = d.t1;
            t.t_dtw = d.t_dtw;
            t.is_special = (d.id >= eot);
            s.tokens.push_back(std::move(t));
        }
        out.push_back(std::move(s));
    }
    return out;
}

static void output_txt(const std::vector<crispasr_segment>& segs, std::ofstream& fout, const whisper_params& params,
                       const std::vector<std::vector<float>>& pcmf32s) {
    const int n_segments = (int)segs.size();
    for (int i = 0; i < n_segments; ++i) {
        const char* text = segs[i].text.c_str();
        // Prefer the unified segs[i].speaker string when set (e.g. by
        // the pyannote diarize post-step from #107). Fall back to the
        // legacy whisper.cpp stereo-energy estimator only when no
        // upstream speaker label is available and we have stereo input.
        std::string speaker = segs[i].speaker;
        if (speaker.empty() && params.diarize && pcmf32s.size() == 2) {
            const int64_t t0 = segs[i].t0;
            const int64_t t1 = segs[i].t1;
            speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
        }

        fout << speaker << text << "\n";
    }
}

static void output_vtt(const std::vector<crispasr_segment>& segs, std::ofstream& fout, const whisper_params& params,
                       const std::vector<std::vector<float>>& pcmf32s) {
    fout << "WEBVTT\n\n";

    const int n_segments = (int)segs.size();
    for (int i = 0; i < n_segments; ++i) {
        const char* text = segs[i].text.c_str();
        const int64_t t0 = segs[i].t0;
        const int64_t t1 = segs[i].t1;
        // Prefer the unified segs[i].speaker label (e.g. from pyannote
        // post-step #107). Strip the "(speaker N) " wrapper and use a
        // VTT-style <v Speaker N> instead.
        std::string speaker = "";
        if (!segs[i].speaker.empty()) {
            const std::string& s = segs[i].speaker;
            // Convert "(speaker 1) " -> "<v Speaker 1>".
            size_t open = s.find('(');
            size_t close = s.find(')');
            if (open != std::string::npos && close != std::string::npos && close > open) {
                std::string inner = s.substr(open + 1, close - open - 1); // "speaker 1"
                if (!inner.empty()) {
                    inner[0] = (char)std::toupper((unsigned char)inner[0]);
                    speaker = "<v " + inner + ">";
                }
            }
        }
        if (speaker.empty() && params.diarize && pcmf32s.size() == 2) {
            speaker = estimate_diarization_speaker(pcmf32s, t0, t1, true);
            speaker.insert(0, "<v Speaker");
            speaker.append(">");
        }

        fout << to_timestamp(t0) << " --> " << to_timestamp(t1) << "\n";
        fout << speaker << text << "\n\n";
    }
}

static void output_srt(const std::vector<crispasr_segment>& segs, std::ofstream& fout, const whisper_params& params,
                       const std::vector<std::vector<float>>& pcmf32s) {
    const int n_segments = (int)segs.size();
    for (int i = 0; i < n_segments; ++i) {
        const char* text = segs[i].text.c_str();
        const int64_t t0 = segs[i].t0;
        const int64_t t1 = segs[i].t1;
        std::string speaker = segs[i].speaker;
        if (speaker.empty() && params.diarize && pcmf32s.size() == 2) {
            speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
        }

        fout << i + 1 + params.offset_n << "\n";
        fout << to_timestamp(t0, true) << " --> " << to_timestamp(t1, true) << "\n";
        fout << speaker << text << "\n\n";
    }
}

static char* escape_double_quotes_and_backslashes(const char* str) {
    if (str == NULL) {
        return NULL;
    }

    size_t escaped_length = strlen(str) + 1;

    for (size_t i = 0; str[i] != '\0'; i++) {
        if (str[i] == '"' || str[i] == '\\') {
            escaped_length++;
        }
    }

    char* escaped = (char*)calloc(escaped_length, 1); // pre-zeroed
    if (escaped == NULL) {
        return NULL;
    }

    size_t pos = 0;
    for (size_t i = 0; str[i] != '\0'; i++) {
        if (str[i] == '"' || str[i] == '\\') {
            escaped[pos++] = '\\';
        }
        escaped[pos++] = str[i];
    }

    // no need to set zero due to calloc() being used prior

    return escaped;
}

// double quote should be escaped by another double quote. (rfc4180)
static char* escape_double_quotes_in_csv(const char* str) {
    if (str == NULL) {
        return NULL;
    }

    size_t escaped_length = strlen(str) + 1;

    for (size_t i = 0; str[i] != '\0'; i++) {
        if (str[i] == '"') {
            escaped_length++;
        }
    }

    char* escaped = (char*)calloc(escaped_length, 1); // pre-zeroed
    if (escaped == NULL) {
        return NULL;
    }

    size_t pos = 0;
    for (size_t i = 0; str[i] != '\0'; i++) {
        if (str[i] == '"') {
            escaped[pos++] = '"';
        }
        escaped[pos++] = str[i];
    }

    // no need to set zero due to calloc() being used prior

    return escaped;
}

static void output_csv(const std::vector<crispasr_segment>& segs, std::ofstream& fout, const whisper_params& params,
                       const std::vector<std::vector<float>>& pcmf32s) {
    const int n_segments = (int)segs.size();
    fout << "start,end,";
    if (params.diarize && pcmf32s.size() == 2) {
        fout << "speaker,";
    }
    fout << "text\n";

    for (int i = 0; i < n_segments; ++i) {
        const char* text = segs[i].text.c_str();
        const int64_t t0 = segs[i].t0;
        const int64_t t1 = segs[i].t1;
        char* text_escaped = escape_double_quotes_in_csv(text);

        //need to multiply times returned from whisper_full_get_segment_t{0,1}() by 10 to get milliseconds.
        fout << 10 * t0 << "," << 10 * t1 << ",";
        if (params.diarize && pcmf32s.size() == 2) {
            fout << estimate_diarization_speaker(pcmf32s, t0, t1, true) << ",";
        }
        fout << "\"" << text_escaped << "\"\n";
    }
}

static void output_score(const std::vector<crispasr_segment>& segs, std::ofstream& fout,
                         const whisper_params& /*params*/, std::vector<std::vector<float>> /*pcmf32s*/) {
    const int n_segments = (int)segs.size();
    // fprintf(stderr,"segments: %d\n",n_segments);
    for (int i = 0; i < n_segments; ++i) {
        const int n_tokens = (int)segs[i].tokens.size();
        // fprintf(stderr,"tokens: %d\n",n_tokens);
        for (int j = 0; j < n_tokens; j++) {
            const char* token = segs[i].tokens[j].text.c_str();
            const float probability = segs[i].tokens[j].confidence;
            fout << token << '\t' << probability << std::endl;
            // fprintf(stderr,"token: %s %f\n",token,probability);
        }
    }
}

static void output_json(const std::vector<crispasr_segment>& segs, std::ofstream& fout, const whisper_params& params,
                        const std::vector<std::vector<float>>& pcmf32s, struct whisper_context* ctx) {
    const bool full = params.output_jsn_full;
    int indent = 0;

    auto doindent = [&]() {
        for (int i = 0; i < indent; i++)
            fout << "\t";
    };

    auto start_arr = [&](const char* name) {
        doindent();
        fout << "\"" << name << "\": [\n";
        indent++;
    };

    auto end_arr = [&](bool end) {
        indent--;
        doindent();
        fout << (end ? "]\n" : "],\n");
    };

    auto start_obj = [&](const char* name) {
        doindent();
        if (name) {
            fout << "\"" << name << "\": {\n";
        } else {
            fout << "{\n";
        }
        indent++;
    };

    auto end_obj = [&](bool end) {
        indent--;
        doindent();
        fout << (end ? "}\n" : "},\n");
    };

    auto start_value = [&](const char* name) {
        doindent();
        fout << "\"" << name << "\": ";
    };

    auto value_s = [&](const char* name, const char* val, bool end) {
        start_value(name);
        char* val_escaped = escape_double_quotes_and_backslashes(val);
        fout << "\"" << val_escaped << (end ? "\"\n" : "\",\n");
        free(val_escaped);
    };

    auto end_value = [&](bool end) { fout << (end ? "\n" : ",\n"); };

    auto value_i = [&](const char* name, const int64_t val, bool end) {
        start_value(name);
        fout << val;
        end_value(end);
    };

    auto value_f = [&](const char* name, const float val, bool end) {
        start_value(name);
        fout << val;
        end_value(end);
    };

    auto value_b = [&](const char* name, const bool val, bool end) {
        start_value(name);
        fout << (val ? "true" : "false");
        end_value(end);
    };

    auto times_o = [&](int64_t t0, int64_t t1, bool end) {
        start_obj("timestamps");
        value_s("from", to_timestamp(t0, true).c_str(), false);
        value_s("to", to_timestamp(t1, true).c_str(), true);
        end_obj(false);
        start_obj("offsets");
        value_i("from", t0 * 10, false);
        value_i("to", t1 * 10, true);
        end_obj(end);
    };

    start_obj(nullptr);
    value_s("systeminfo", whisper_print_system_info(), false);
    start_obj("model");
    value_s("type", whisper_model_type_readable(ctx), false);
    value_b("multilingual", whisper_is_multilingual(ctx), false);
    value_i("vocab", whisper_model_n_vocab(ctx), false);
    start_obj("audio");
    value_i("ctx", whisper_model_n_audio_ctx(ctx), false);
    value_i("state", whisper_model_n_audio_state(ctx), false);
    value_i("head", whisper_model_n_audio_head(ctx), false);
    value_i("layer", whisper_model_n_audio_layer(ctx), true);
    end_obj(false);
    start_obj("text");
    value_i("ctx", whisper_model_n_text_ctx(ctx), false);
    value_i("state", whisper_model_n_text_state(ctx), false);
    value_i("head", whisper_model_n_text_head(ctx), false);
    value_i("layer", whisper_model_n_text_layer(ctx), true);
    end_obj(false);
    value_i("mels", whisper_model_n_mels(ctx), false);
    value_i("ftype", whisper_model_ftype(ctx), true);
    end_obj(false);
    start_obj("params");
    value_s("model", params.model.c_str(), false);
    value_s("language", params.language.c_str(), false);
    value_b("translate", params.translate, true);
    end_obj(false);
    start_obj("result");
    value_s("language", whisper_lang_str(whisper_full_lang_id(ctx)), true);
    end_obj(false);
    start_arr("transcription");

    const int n_segments = (int)segs.size();
    for (int i = 0; i < n_segments; ++i) {
        const char* text = segs[i].text.c_str();

        const int64_t t0 = segs[i].t0;
        const int64_t t1 = segs[i].t1;

        // Multi-task ASR metadata (SenseVoice and similar). Emit any
        // non-empty fields right after `text`. Each one is a flat string
        // sibling: language / emotion / audio_event / itn_flag.
        const bool has_lang = !segs[i].lang_id.empty();
        const bool has_emo = !segs[i].emotion.empty();
        const bool has_evt = !segs[i].audio_event.empty();
        const bool has_itn = !segs[i].itn_flag.empty();
        const bool has_meta = has_lang || has_emo || has_evt || has_itn;

        start_obj(nullptr);
        times_o(t0, t1, false);
        value_s("text", text, !has_meta && !params.diarize && !params.tinydiarize && !full);

        if (has_meta) {
            const bool meta_is_last = !full && !params.diarize && !params.tinydiarize;
            int remaining = (int)has_lang + (int)has_emo + (int)has_evt + (int)has_itn;
            auto emit = [&](const char* name, const std::string& v, bool present) {
                if (!present)
                    return;
                --remaining;
                value_s(name, v.c_str(), remaining == 0 && meta_is_last);
            };
            emit("language", segs[i].lang_id, has_lang);
            emit("audio_event", segs[i].audio_event, has_evt);
            emit("emotion", segs[i].emotion, has_emo);
            emit("itn_flag", segs[i].itn_flag, has_itn);
        }

        if (full) {
            start_arr("tokens");
            const int n = (int)segs[i].tokens.size();
            for (int j = 0; j < n; ++j) {
                const auto& token = segs[i].tokens[j];
                start_obj(nullptr);
                value_s("text", token.text.c_str(), false);
                if (token.t0 > -1 && token.t1 > -1) {
                    // If we have per-token timestamps, write them out
                    times_o(token.t0, token.t1, false);
                }
                value_i("id", token.id, false);
                value_f("p", token.confidence, false);
                value_f("t_dtw", token.t_dtw, true);
                end_obj(j == (n - 1));
            }
            end_arr(!params.diarize && !params.tinydiarize);
        }

        if (params.diarize) {
            // Prefer the unified segs[i].speaker (pyannote / energy / etc.
            // post-step #107). Strip the historical "(speaker N) "
            // wrapper down to just "N" for clean JSON; if the wrapper
            // isn't present, pass the string through. Fall back to the
            // legacy stereo-energy estimator only when no upstream
            // label is available AND we have stereo input.
            std::string label;
            const std::string& s = segs[i].speaker;
            if (!s.empty()) {
                size_t open = s.find('(');
                size_t close = s.find(')');
                if (open != std::string::npos && close != std::string::npos && close > open) {
                    // "(speaker 1) " -> "1"
                    std::string inner = s.substr(open + 1, close - open - 1);
                    const std::string prefix = "speaker ";
                    if (inner.size() > prefix.size() && inner.compare(0, prefix.size(), prefix) == 0) {
                        label = inner.substr(prefix.size());
                    } else {
                        label = inner;
                    }
                } else {
                    label = s;
                }
            } else if (pcmf32s.size() == 2) {
                label = estimate_diarization_speaker(pcmf32s, t0, t1, true);
            }
            if (!label.empty())
                value_s("speaker", label.c_str(), true);
        }

        if (params.tinydiarize) {
            value_b("speaker_turn_next", segs[i].speaker_turn_next, true);
        }
        end_obj(i == (n_segments - 1));
    }

    end_arr(true);
    end_obj(true);
}

// karaoke video generation
// outputs a bash script that uses ffmpeg to generate a video with the subtitles
// TODO: font parameter adjustments
static bool output_wts(const std::vector<crispasr_segment>& segs, std::ofstream& fout, const whisper_params& params,
                       const std::vector<std::vector<float>>& pcmf32s, const char* fname_inp, float t_sec,
                       const char* fname_out) {
    static const char* font = params.font_path.c_str();

    std::ifstream fin(font);
    if (!fin.is_open()) {
        fprintf(stderr, "%s: font not found at '%s', please specify a monospace font with -fp\n", __func__, font);
        return false;
    }

    fout << "#!/bin/bash" << "\n";
    fout << "\n";

    fout << "ffmpeg -i " << fname_inp << " -f lavfi -i color=size=1200x120:duration=" << t_sec
         << ":rate=25:color=black -vf \"";

    for (int i = 0; i < (int)segs.size(); i++) {
        const int64_t t0 = segs[i].t0;
        const int64_t t1 = segs[i].t1;

        const int n = (int)segs[i].tokens.size();

        const std::vector<crispasr_token>& tokens = segs[i].tokens;

        if (i > 0) {
            fout << ",";
        }

        // background text
        fout << "drawtext=fontfile='" << font
             << "':fontsize=24:fontcolor=gray:x=(w-text_w)/2:y=h/2:text='':enable='between(t," << t0 / 100.0 << ","
             << t0 / 100.0 << ")'";

        bool is_first = true;
        std::string speaker = "";

        if (params.diarize && pcmf32s.size() == 2) {
            speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
        }

        for (int j = 0; j < n; ++j) {
            const auto& token = tokens[j];

            if (tokens[j].is_special) {
                continue;
            }

            std::string txt_bg = "";
            std::string txt_fg = ""; // highlight token
            std::string txt_ul = ""; // underline

            if (params.diarize && pcmf32s.size() == 2) {
                txt_bg = speaker;
                txt_fg = speaker;
                txt_ul = "\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ ";
            }

            txt_bg.append("> ");
            txt_fg.append("> ");
            txt_ul.append("\\ \\ ");

            {
                for (int k = 0; k < n; ++k) {
                    const auto& token2 = tokens[k];

                    if (tokens[k].is_special) {
                        continue;
                    }

                    const std::string& txt = token2.text;

                    txt_bg += txt;

                    if (k == j) {
                        for (int l = 0; l < (int)txt.size(); ++l) {
                            txt_fg += txt[l];
                            txt_ul += "_";
                        }
                        txt_fg += "|";
                    } else {
                        for (int l = 0; l < (int)txt.size(); ++l) {
                            txt_fg += "\\ ";
                            txt_ul += "\\ ";
                        }
                    }
                }

                ::replace_all(txt_bg, "'", "\u2019");
                ::replace_all(txt_bg, "\"", "\\\"");
                ::replace_all(txt_fg, "'", "\u2019");
                ::replace_all(txt_fg, "\"", "\\\"");
            }

            if (is_first) {
                // background text
                fout << ",drawtext=fontfile='" << font << "':fontsize=24:fontcolor=gray:x=(w-text_w)/2:y=h/2:text='"
                     << txt_bg << "':enable='between(t," << t0 / 100.0 << "," << t1 / 100.0 << ")'";
                is_first = false;
            }

            // foreground text
            fout << ",drawtext=fontfile='" << font << "':fontsize=24:fontcolor=lightgreen:x=(w-text_w)/2+8:y=h/2:text='"
                 << txt_fg << "':enable='between(t," << token.t0 / 100.0 << "," << token.t1 / 100.0 << ")'";

            // underline
            fout << ",drawtext=fontfile='" << font
                 << "':fontsize=24:fontcolor=lightgreen:x=(w-text_w)/2+8:y=h/2+16:text='" << txt_ul
                 << "':enable='between(t," << token.t0 / 100.0 << "," << token.t1 / 100.0 << ")'";
        }
    }

    fout << "\" -c:v libx264 -pix_fmt yuv420p -y " << fname_inp << ".mp4" << "\n";

    fout << "\n\n";
    fout << "echo \"Your video has been saved to " << fname_inp << ".mp4\"" << "\n";
    fout << "\n";
    fout << "echo \"  ffplay " << fname_inp << ".mp4\"\n";
    fout << "\n";

    fout.close();

    fprintf(stderr, "# %s: run 'source %s' to generate karaoke video\n", __func__, fname_out);

    return true;
}

static void output_lrc(const std::vector<crispasr_segment>& segs, std::ofstream& fout, const whisper_params& params,
                       const std::vector<std::vector<float>>& pcmf32s) {
    fout << "[by:crispasr]\n";

    const int n_segments = (int)segs.size();
    for (int i = 0; i < n_segments; ++i) {
        const char* text = segs[i].text.c_str();
        const int64_t t = segs[i].t0;

        int64_t msec = t * 10;
        int64_t min = msec / (1000 * 60);
        msec = msec - min * (1000 * 60);
        int64_t sec = msec / 1000;
        msec = msec - sec * 1000;

        char buf[16];
        snprintf(buf, sizeof(buf), "%02d:%02d.%02d", (int)min, (int)sec, (int)(msec / 10));
        std::string timestamp_lrc = std::string(buf);
        std::string speaker = "";

        if (params.diarize && pcmf32s.size() == 2) {
            const int64_t t0 = segs[i].t0;
            const int64_t t1 = segs[i].t1;
            speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
        }

        fout << '[' << timestamp_lrc << ']' << speaker << text << "\n";
    }
}


static void cb_log_disable(enum ggml_log_level, const char*, void*) {}

int main(int argc, char** argv) {
#if defined(_WIN32)
    // Set the console output code page to UTF-8, while command line arguments
    // are still encoded in the system's code page. In this way, we can print
    // non-ASCII characters to the console, and access files with non-ASCII paths.
    SetConsoleOutputCP(CP_UTF8);
#endif

    whisper_params params;

    // If the only argument starts with "@", read arguments line-by-line
    // from the given file.
    std::vector<std::string> vec_args;
    std::vector<char*> vec_argv; // hoisted: must outlive argv usage below
    if (argc == 2 && argv != nullptr && argv[1] != nullptr && argv[1][0] == '@') {
        // Save the name of the executable.
        vec_args.push_back(argv[0]);

        // Open the response file.
        char const* rspfile = argv[1] + sizeof(char);
        std::ifstream fin(rspfile);
        if (fin.is_open() == false) {
            fprintf(stderr, "error: response file '%s' not found\n", rspfile);
            return 1;
        }

        // Read the entire response file.
        std::string line;
        while (std::getline(fin, line)) {
            vec_args.push_back(line);
        }

        // Use the contents of the response file as the command-line arguments.
        argc = static_cast<int>(vec_args.size());
        vec_argv.resize(argc);
        for (int i = 0; i < argc; ++i) {
            vec_argv[i] = const_cast<char*>(vec_args[i].c_str());
        }
        argv = vec_argv.data();
    }

    if (whisper_params_parse(argc, argv, params) == false) {
        whisper_print_usage(argc, argv, params);
        return 1;
    }

    // Always emit a one-line banner unless --no-prints. Cheap insurance:
    // any user log starts with the exact build identifier, so triage
    // doesn't have to ask "which tag did you pull?" (#31).
    if (!params.no_prints) {
        crispasr_print_short_banner(stderr);
    }

    // --verbose: dump the full build info + env + device list before we
    // touch the model. The CUDA enumeration in crispasr_print_devices()
    // logs through GGML_LOG_ERROR on driver/runtime mismatch, so users
    // hitting #31 get a complete capture in the same log block.
    if (params.verbose) {
        crispasr_print_full_diagnostics(stderr);
    }

    if (params.use_gpu && params.gpu_backend != "cpu") {
        ggml_backend_load_all();
    }

    // remove non-existent files
    for (auto it = params.fname_inp.begin(); it != params.fname_inp.end();) {
        const auto fname_inp = it->c_str();

        if (*it != "-" && !is_file_exist(fname_inp)) {
            fprintf(stderr, "error: input file not found '%s'\n", fname_inp);
            it = params.fname_inp.erase(it);
            continue;
        }

        it++;
    }

    // Server mode: keep model loaded, accept HTTP requests
    if (params.server) {
        return crispasr_run_server(params, params.server_host, params.server_port);
    }

    if (params.dry_run_resolve) {
        std::string backend_name = params.backend == "auto" ? "" : params.backend;
        if (backend_name.empty() && is_auto_model_arg(params.model)) {
            backend_name = "whisper";
        } else if (backend_name.empty() && !is_auto_model_arg(params.model)) {
            backend_name = crispasr_detect_backend_from_gguf(params.model);
        }

        const CrispasrResolvePreview model_preview = crispasr_preview_model_cli(
            params.model, backend_name, params.cache_dir, params.model_quant, params.dry_run_ignore_cache);
        print_resolve_preview("model", model_preview);

        bool ok = !model_preview.unresolved;
        if (!backend_name.empty()) {
            CrispasrRegistryEntry entry;
            if (crispasr_registry_lookup(backend_name, entry, params.tts_codec_quant) &&
                !entry.companion_filename.empty()) {
                const std::string codec_arg =
                    params.tts_codec_model.empty() ? entry.companion_filename : params.tts_codec_model;
                const CrispasrResolvePreview companion_preview = crispasr_preview_model_cli(
                    codec_arg, backend_name, params.cache_dir, params.tts_codec_quant, params.dry_run_ignore_cache);
                print_resolve_preview("companion", companion_preview);
                ok = ok && !companion_preview.unresolved;
            }
        }
        return ok ? 0 : 1;
    }

    if (params.fname_inp.empty() && !params.stream && params.tts_text.empty() && params.text_input.empty()) {
        fprintf(stderr, "error: no input files specified\n");
        whisper_print_usage(argc, argv, params);
        return 2;
    }

    // Speaker enrollment: extract TitaNet embedding from audio, save to DB, exit.
    // This is a standalone operation that doesn't need any ASR backend.
    if (!params.enroll_speaker.empty() && !params.fname_inp.empty()) {
        // Enrollment needs: audio → TitaNet → speaker_db_enroll → exit
        // Route through the unified dispatch which has the enrollment code.
        if (params.backend.empty())
            params.backend = "whisper"; // any backend, enrollment exits before init
        const int rc = crispasr_run_backend(params);
        return rc;
    }

    // crispasr backend dispatch ---------------------------------------------
    // Route through the unified dispatch layer (crispasr_run_backend) when:
    //   1. --backend is set explicitly (including --backend whisper, which
    //      opts into the reduced-feature unified whisper wrapper);
    //   2. -m auto / -m default requests an auto-download;
    //   3. GGUF metadata auto-identifies a non-whisper architecture.
    // Otherwise (empty --backend, non-auto model, .bin file, or GGUF that
    // says "whisper"), fall through to the historical whisper code path
    // below so the byte-identical default stays intact — it still has
    // access to whisper-only features like -owts, full-JSON DTW tokens,
    // grammar, n_processors, whisper-internal VAD and stereo diarize.
    {
        const bool explicit_backend = !params.backend.empty();
        const bool model_is_auto = is_auto_model_arg(params.model);

        bool auto_detected_non_whisper = false;
        if (!explicit_backend && !model_is_auto) {
            const std::string detected = crispasr_detect_backend_from_gguf(params.model);
            if (!detected.empty() && detected != "whisper") {
                if (!params.no_prints) {
                    fprintf(stderr, "crispasr: auto-detected backend '%s' from '%s'\n", detected.c_str(),
                            params.model.c_str());
                }
                params.backend = detected;
                auto_detected_non_whisper = true;
            }
        }

        if (explicit_backend || model_is_auto || auto_detected_non_whisper || params.stream ||
            !params.tts_text.empty()) {
            const int rc = crispasr_run_backend(params);
#if defined(_WIN32)
            // Bypass global C++ destructors (ggml Vulkan device teardown can
            // stall indefinitely on Windows when the GPU is idle post-inference).
            std::fflush(stdout);
            std::fflush(stderr);
            _Exit(rc);
#else
            return rc;
#endif
        }
    }
    // -----------------------------------------------------------------------

    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    if (params.diarize && params.tinydiarize) {
        fprintf(stderr, "error: cannot use both --diarize and --tinydiarize\n");
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    if (params.no_prints) {
        whisper_log_set(cb_log_disable, NULL);
    }

    // whisper init
    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu = params.use_gpu;
    cparams.gpu_device = params.gpu_device;
    cparams.flash_attn = params.flash_attn;

    if (!params.dtw.empty()) {
        cparams.dtw_token_timestamps = true;
        cparams.dtw_aheads_preset = CRISPASR_AHEADS_NONE;

        if (params.dtw == "tiny")
            cparams.dtw_aheads_preset = CRISPASR_AHEADS_TINY;
        if (params.dtw == "tiny.en")
            cparams.dtw_aheads_preset = CRISPASR_AHEADS_TINY_EN;
        if (params.dtw == "base")
            cparams.dtw_aheads_preset = CRISPASR_AHEADS_BASE;
        if (params.dtw == "base.en")
            cparams.dtw_aheads_preset = CRISPASR_AHEADS_BASE_EN;
        if (params.dtw == "small")
            cparams.dtw_aheads_preset = CRISPASR_AHEADS_SMALL;
        if (params.dtw == "small.en")
            cparams.dtw_aheads_preset = CRISPASR_AHEADS_SMALL_EN;
        if (params.dtw == "medium")
            cparams.dtw_aheads_preset = CRISPASR_AHEADS_MEDIUM;
        if (params.dtw == "medium.en")
            cparams.dtw_aheads_preset = CRISPASR_AHEADS_MEDIUM_EN;
        if (params.dtw == "large.v1")
            cparams.dtw_aheads_preset = CRISPASR_AHEADS_LARGE_V1;
        if (params.dtw == "large.v2")
            cparams.dtw_aheads_preset = CRISPASR_AHEADS_LARGE_V2;
        if (params.dtw == "large.v3")
            cparams.dtw_aheads_preset = CRISPASR_AHEADS_LARGE_V3;
        if (params.dtw == "large.v3.turbo")
            cparams.dtw_aheads_preset = CRISPASR_AHEADS_LARGE_V3_TURBO;

        if (cparams.dtw_aheads_preset == CRISPASR_AHEADS_NONE) {
            fprintf(stderr, "error: unknown DTW preset '%s'\n", params.dtw.c_str());
            return 3;
        }
    }

    struct whisper_context* ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 3;
    }

    // initialize openvino encoder. this has no effect on crispasr builds that don't have OpenVINO configured
    whisper_ctx_init_openvino_encoder(ctx, nullptr, params.openvino_encode_device.c_str(), nullptr);

    if (!params.grammar.empty()) {
        auto& grammar = params.grammar_parsed;
        if (is_file_exist(params.grammar.c_str())) {
            // read grammar from file
            std::ifstream ifs(params.grammar.c_str());
            const std::string txt =
                std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            grammar = grammar_parser::parse(txt.c_str());
        } else {
            // read grammar from string
            grammar = grammar_parser::parse(params.grammar.c_str());
        }

        // will be empty (default) if there are parse errors
        if (grammar.rules.empty()) {
            fprintf(stderr, "error: failed to parse grammar \"%s\"\n", params.grammar.c_str());
            return 4;
        } else {
            fprintf(stderr, "%s: grammar:\n", __func__);
            grammar_parser::print_grammar(stderr, grammar);
            fprintf(stderr, "\n");
        }
    }

    for (int f = 0; f < (int)params.fname_inp.size(); ++f) {
        const auto& fname_inp = params.fname_inp[f];
        struct fout_factory {
            std::string fname_out;
            const size_t basename_length;
            const bool is_stdout;
            bool used_stdout;
            decltype(whisper_print_segment_callback)* const print_segment_callback;
            std::ofstream fout;

            fout_factory(const std::string& fname_out_, const std::string& fname_inp, whisper_params& params)
                : fname_out{!fname_out_.empty() ? fname_out_ : fname_inp}, basename_length{fname_out.size()},
                  is_stdout{fname_out == "-"}, used_stdout{},
                  print_segment_callback{is_stdout ? nullptr : whisper_print_segment_callback} {
                if (!print_segment_callback) {
                    params.print_progress = false;
                }
            }

            bool open(const char* ext, const char* function) {
                if (is_stdout) {
                    if (used_stdout) {
                        fprintf(stderr, "warning: Not appending multiple file formats to stdout\n");
                        return false;
                    }

                    used_stdout = true;
#ifdef _WIN32
                    fout = std::ofstream{"CON"};
#else
                    fout = std::ofstream{"/dev/stdout"};
#endif
                    // Not using fprintf stderr here because it might equal stdout
                    // Also assuming /dev is mounted
                    return true;
                }

                fname_out.resize(basename_length);
                fname_out += ext;
                fout = std::ofstream{fname_out};
                if (!fout.is_open()) {
                    fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname_out.c_str());
                    return false;
                }
                fprintf(stderr, "%s: saving output to '%s'\n", function, fname_out.c_str());
                return true;
            }
        } fout_factory{f < (int)params.fname_out.size() ? params.fname_out[f] : "", fname_inp, params};

        std::vector<float> pcmf32;               // mono-channel F32 PCM
        std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

        if (!::read_audio_data(fname_inp, pcmf32, pcmf32s, params.diarize)) {
            fprintf(stderr, "error: failed to read audio file '%s'\n", fname_inp.c_str());
            continue;
        }

        if (!whisper_is_multilingual(ctx)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n",
                        __func__);
            }
        }
        if (params.detect_language) {
            params.language = "auto";
        }

        if (!params.no_prints) {
            // print system information
            fprintf(stderr, "\n");
            fprintf(stderr, "system_info: n_threads = %d / %d | %s\n", params.n_threads * params.n_processors,
                    std::thread::hardware_concurrency(), whisper_print_system_info());

            // print some info about the processing
            fprintf(stderr, "\n");
            fprintf(stderr,
                    "%s: processing '%s' (%d samples, %.1f sec), %d threads, %d processors, %d beams + best of %d, "
                    "lang = %s, task = %s, %stimestamps = %d ...\n",
                    __func__, fname_inp.c_str(), int(pcmf32.size()), float(pcmf32.size()) / CRISPASR_SAMPLE_RATE,
                    params.n_threads, params.n_processors, params.beam_size, params.best_of, params.language.c_str(),
                    params.translate ? "translate" : "transcribe", params.tinydiarize ? "tdrz = 1, " : "",
                    params.no_timestamps ? 0 : 1);

            if (params.print_colors) {
                fprintf(stderr, "%s: color scheme: red (low confidence), yellow (medium), green (high confidence)\n",
                        __func__);
            } else if (params.print_confidence) {
                fprintf(stderr,
                        "%s: confidence: highlighted (low confidence), underlined (medium), dim (high confidence)\n",
                        __func__);
            }
            fprintf(stderr, "\n");
        }

        // run the inference
        {
            whisper_full_params wparams = whisper_full_default_params(CRISPASR_SAMPLING_GREEDY);

            const bool use_grammar = (!params.grammar_parsed.rules.empty() && !params.grammar_rule.empty());
            wparams.strategy =
                (params.beam_size > 1 || use_grammar) ? CRISPASR_SAMPLING_BEAM_SEARCH : CRISPASR_SAMPLING_GREEDY;

            wparams.print_realtime = false;
            wparams.print_progress = params.print_progress;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.print_special = params.print_special;
            wparams.translate = params.translate;
            wparams.language = params.language.c_str();
            wparams.detect_language = params.detect_language;
            wparams.n_threads = params.n_threads;
            wparams.n_max_text_ctx = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
            wparams.offset_ms = params.offset_t_ms;
            wparams.duration_ms = params.duration_ms;

            wparams.token_timestamps = params.output_wts || params.output_jsn_full || params.max_len > 0;
            wparams.thold_pt = params.word_thold;
            wparams.max_len = params.output_wts && params.max_len == 0 ? 60 : params.max_len;
            wparams.split_on_word = params.split_on_word;
            wparams.audio_ctx = params.audio_ctx;

            wparams.debug_mode = params.debug_mode;

            wparams.tdrz_enable = params.tinydiarize; // [TDRZ]

            wparams.suppress_regex = params.suppress_regex.empty() ? nullptr : params.suppress_regex.c_str();

            wparams.initial_prompt = params.prompt.c_str();
            wparams.carry_initial_prompt = params.carry_initial_prompt;

            wparams.greedy.best_of = params.best_of;
            wparams.beam_search.beam_size = params.beam_size;

            wparams.temperature_inc = params.no_fallback ? 0.0f : params.temperature_inc;
            wparams.temperature = params.temperature;

            wparams.entropy_thold = params.entropy_thold;
            wparams.logprob_thold = params.logprob_thold;
            wparams.no_speech_thold = params.no_speech_thold;

            wparams.no_timestamps = params.no_timestamps;

            wparams.suppress_nst = params.suppress_nst;

            // Resolve `--vad` without `--vad-model` to the canonical
            // ggml-silero-v5.1.2.bin in the cache, downloading on first
            // use — matches the auto-cache UX of every non-whisper
            // backend (#33). Path must outlive whisper_full(); kept on
            // this stack frame.
            // FireRedVAD (GGUF) is not compatible with whisper's internal
            // Silero-only VAD loader (#34). Detect and warn.
            const bool firered_vad = crispasr_vad_is_firered(params);
            const std::string resolved_vad_path = firered_vad ? "" : crispasr_resolve_vad_model(params);
            wparams.vad = firered_vad ? false : params.vad;
            wparams.vad_model_path = resolved_vad_path.c_str();
            if (firered_vad) {
                fprintf(stderr, "crispasr: warning: FireRedVAD is not supported in the legacy whisper path.\n"
                                "  Use --backend whisper (unified dispatch) or a non-whisper backend:\n"
                                "  crispasr --backend whisper --vad --vad-model firered-vad.gguf ...\n"
                                "  crispasr -m parakeet.gguf --vad --vad-model firered-vad.gguf ...\n");
            }

            wparams.vad_params.threshold = params.vad_threshold;
            wparams.vad_params.min_speech_duration_ms = params.vad_min_speech_duration_ms;
            wparams.vad_params.min_silence_duration_ms = params.vad_min_silence_duration_ms;
            wparams.vad_params.max_speech_duration_s = params.vad_max_speech_duration_s;
            wparams.vad_params.speech_pad_ms = params.vad_speech_pad_ms;
            wparams.vad_params.samples_overlap = params.vad_samples_overlap;

            whisper_print_user_data user_data = {&params, &pcmf32s, 0};

            const auto& grammar_parsed = params.grammar_parsed;
            auto grammar_rules = grammar_parsed.c_rules();

            if (use_grammar) {
                if (grammar_parsed.symbol_ids.find(params.grammar_rule) == grammar_parsed.symbol_ids.end()) {
                    fprintf(stderr, "%s: warning: grammar rule '%s' not found - skipping grammar sampling\n", __func__,
                            params.grammar_rule.c_str());
                } else {
                    wparams.grammar_rules = grammar_rules.data();
                    wparams.n_grammar_rules = grammar_rules.size();
                    wparams.i_start_rule = grammar_parsed.symbol_ids.at(params.grammar_rule);
                    wparams.grammar_penalty = params.grammar_penalty;
                }
            }

            // this callback is called on each new segment
            if (!wparams.print_realtime) {
                wparams.new_segment_callback = fout_factory.print_segment_callback;
                wparams.new_segment_callback_user_data = &user_data;
            }

            if (wparams.print_progress) {
                wparams.progress_callback = whisper_print_progress_callback;
                wparams.progress_callback_user_data = &user_data;
            }

            // examples for abort mechanism
            // in examples below, we do not abort the processing, but we could if the flag is set to true

            // the callback is called before every encoder run - if it returns false, the processing is aborted
            {
                static bool is_aborted = false; // NOTE: this should be atomic to avoid data race

                wparams.encoder_begin_callback = [](struct whisper_context* /*ctx*/, struct whisper_state* /*state*/,
                                                    void* user_data) {
                    bool is_aborted = *(bool*)user_data;
                    return !is_aborted;
                };
                wparams.encoder_begin_callback_user_data = &is_aborted;
            }

            // the callback is called before every computation - if it returns true, the computation is aborted
            {
                static bool is_aborted = false; // NOTE: this should be atomic to avoid data race

                wparams.abort_callback = [](void* user_data) {
                    bool is_aborted = *(bool*)user_data;
                    return is_aborted;
                };
                wparams.abort_callback_user_data = &is_aborted;
            }

            if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0) {
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                return 10;
            }
        }

        // output stuff
        {
            // Collect whisper segments + tokens into the unified vector once,
            // so every writer below is whisper_context-free (except JSON,
            // which still needs ctx for systeminfo/model metadata).
            std::vector<crispasr_segment> segs = cli_whisper_collect_segments(ctx);

            // Honor --split-on-punct in the legacy whisper output path (#29).
            // Without this the writers below emit whisper's raw segments,
            // which can run 30+ seconds in CJK and ignore --split-on-punct
            // entirely. Re-segment via the unified display-segment splitter,
            // then convert back to crispasr_segment for the writers (which
            // only read t0/t1/text/speaker for non-token formats).
            if (params.split_on_punct || params.max_len > 0) {
                auto disp = crispasr_make_disp_segments(segs, params.max_len, params.split_on_punct);
                std::vector<crispasr_segment> resplit;
                resplit.reserve(disp.size());
                for (auto& d : disp) {
                    crispasr_segment s;
                    s.t0 = d.t0;
                    s.t1 = d.t1;
                    s.text = std::move(d.text);
                    s.speaker = std::move(d.speaker);
                    resplit.push_back(std::move(s));
                }
                segs = std::move(resplit);
            }

            // Diarization post-step (issue #107). The whisper backend
            // came through cli.cpp's legacy main path rather than the
            // backend dispatcher in crispasr_run.cpp, so until now
            // --diarize-method only delegated to whisper.cpp's built-in
            // stereo-only energy diarize. Wire it up to the shared
            // crispasr_apply_diarize() shim so pyannote (and the other
            // methods) work here too. Only fires when the user passed
            // an explicit method — otherwise the existing whisper.cpp
            // (speaker N) string is left in place unchanged.
            if (params.diarize && !params.diarize_method.empty() && !segs.empty()) {
                CrispasrPyannoteCache pyannote_cache;
                if (params.diarize_method == "pyannote" && !pcmf32.empty()) {
                    if (!crispasr_compute_pyannote_cache(pcmf32.data(), (int)pcmf32.size(), params, pyannote_cache)) {
                        pyannote_cache = {};
                    }
                }
                const bool is_stereo =
                    pcmf32s.size() == 2 && !pcmf32s[0].empty() && pcmf32s[0].size() == pcmf32s[1].size();
                const std::vector<float>& left = is_stereo ? pcmf32s[0] : pcmf32;
                const std::vector<float>& right = is_stereo ? pcmf32s[1] : pcmf32;
                crispasr_apply_diarize(left, right, is_stereo, /*slice_t0_cs=*/0, segs, params,
                                       pyannote_cache.valid() ? &pyannote_cache : nullptr);

                // Optional embedding-based clustering (#107 P3). When
                // --diarize-embedder is set, build the pluggable
                // embedder (TitaNet today, others later) and rewrite
                // each segment's speaker label with its global cluster
                // ID. Failure to build the embedder is a warning, not
                // an error — the pyannote-local labels above survive.
                if (!params.diarize_embedder.empty() && !pcmf32.empty()) {
                    auto embedder =
                        crispasr_make_speaker_embedder(params.diarize_embedder, params.n_threads, params.cache_dir);
                    if (embedder) {
                        crispasr_remap_speakers_via_embeddings(segs, pcmf32.data(), (int)pcmf32.size(), embedder.get(),
                                                               params);
                    }
                }
            }

            // macros to stringify function name
#define output_func(func, ext, param, ...)                                                                             \
    if (param && fout_factory.open(ext, #func)) {                                                                      \
        func(segs, fout_factory.fout, params, __VA_ARGS__);                                                            \
    }
#define output_ext(ext, ...) output_func(output_##ext, "." #ext, params.output_##ext, __VA_ARGS__)

            output_ext(txt, pcmf32s);
            output_ext(vtt, pcmf32s);
            output_ext(srt, pcmf32s);
            output_ext(wts, pcmf32s, fname_inp.c_str(), float(pcmf32.size() + 1000) / CRISPASR_SAMPLE_RATE,
                       fout_factory.fname_out.c_str());
            output_ext(csv, pcmf32s);
            output_func(output_json, ".json", params.output_jsn, pcmf32s, ctx);
            output_ext(lrc, pcmf32s);
            output_func(output_score, ".score.txt", params.log_score, pcmf32s);

#undef output_ext
#undef output_func

            if (fout_factory.is_stdout && !fout_factory.used_stdout) {
                fprintf(stderr, "warning: '--output-file -' used without any other '--output-*'");
            }
        }
    }

    // Post-ASR text LID: if --lid-on-transcript was set, run a text
    // classifier on the assembled transcript and emit
    // `lang=<code>\tconf=<score>` to stderr. The dispatcher peeks the
    // GGUF's general.architecture and picks fastText (GlotLID/LID-176)
    // or CLD3 automatically. Errors are logged but never fail the
    // run — the transcript output stays the source of truth.
    if (!params.lid_on_transcript.empty()) {
        std::vector<crispasr_segment> lid_segs = cli_whisper_collect_segments(ctx);
        std::string transcript;
        for (const auto& s : lid_segs) {
            if (!transcript.empty())
                transcript.push_back(' ');
            transcript += s.text;
        }
        if (transcript.empty()) {
            if (!params.no_prints)
                fprintf(stderr, "crispasr[lid-on-transcript]: empty transcript, skipping\n");
        } else {
            // Resolve `auto[:variant]` + bare-filename inputs against the
            // registry, downloading into ~/.cache/crispasr/ on first use.
            const std::string resolved =
                text_lid_resolve_path(params.lid_on_transcript, params.cache_dir, params.no_prints);
            text_lid_context* lid = resolved.empty() ? nullptr : text_lid_init_from_file(resolved.c_str(), 1);
            if (!lid) {
                fprintf(stderr, "crispasr[lid-on-transcript]: failed to load '%s'\n", params.lid_on_transcript.c_str());
            } else {
                float conf = 0.f;
                const char* lab = text_lid_predict(lid, transcript.c_str(), &conf);
                if (lab)
                    fprintf(stderr, "lang=%s\tconf=%.6f\tbackend=%s\n", lab, conf, text_lid_backend(lid));
                else
                    fprintf(stderr, "crispasr[lid-on-transcript]: prediction failed\n");
                text_lid_free(lid);
            }
        }
    }

    if (!params.no_prints) {
        whisper_print_timings(ctx);
    }
    whisper_free(ctx);

#if defined(_WIN32)
    // Bypass global C++ destructors (ggml Vulkan device teardown can
    // stall indefinitely on Windows when the GPU is idle post-inference).
    std::fflush(stdout);
    std::fflush(stderr);
    _Exit(0);
#else
    return 0;
#endif
}
