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
    std::string cache_dir;
    std::string tts_text;
    std::string tts_output;
    std::string tts_voice;
    int tts_steps = 20;
    std::string tts_codec_model;
    std::string tts_ref_text;
    bool tts_trim_silence = false;
};
