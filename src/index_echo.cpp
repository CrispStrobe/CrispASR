#include "index_echo.h"
#include "crisp_audio.h"
#include "core/gguf_loader.h"
#include "core/ggml_cpu_backend.h"
#include "core/index_echo_windows.h"
#include "core/silero_context.h"
#include "core/index_echo_batch.h"
#include "core/index_echo_connector.h"
#include "crispasr.h"
#include "ggml-alloc.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct index_echo_context {
    crisp_audio_context* audio = nullptr;
    llama_model* model = nullptr;
    llama_context* decoder = nullptr;
    const llama_vocab* vocab = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    whisper_vad_context* vad = nullptr;
    core_gguf::WeightLoad weights;
    index_echo_context_params params{};
    std::string lang = "en", glossary, context, ask;
    float temperature = 0;
    uint32_t seed = 0;
    int max_tokens = 2000;
    int position = 0;
    llama_token audio_pad = LLAMA_TOKEN_NULL, im_end = LLAMA_TOKEN_NULL;
    bool capture = false;
    bool projection = false;
    std::map<std::string, std::vector<float>> stages;
    std::vector<int32_t> prompt_ids;
};

namespace {
constexpr int kBatch = 256;

struct Bench {
    const char* name;
    bool enabled;
    std::chrono::steady_clock::time_point start;
    explicit Bench(const char* stage) : name(stage) {
        static const bool active = [] {
            const char* value = getenv("INDEX_ECHO_BENCH");
            return value && *value && *value != '0';
        }();
        enabled = active;
        if (enabled)
            start = std::chrono::steady_clock::now();
    }
    ~Bench() {
        if (enabled)
            fprintf(stderr, "index_echo_bench: %-20s %.2f ms\n", name,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
};

bool capture_decoder(ggml_tensor* t, bool ask, void* user);
llama_context* create_decoder(index_echo_context* ctx, uint32_t capacity) {
    auto cp = llama_context_default_params();
    cp.n_ctx = capacity;
    cp.n_batch = kBatch;
    cp.n_ubatch = kBatch;
    cp.n_threads = ctx->params.n_threads;
    cp.n_threads_batch = ctx->params.n_threads;
    cp.flash_attn_type = ctx->params.flash_attn ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = capture_decoder;
    cp.cb_eval_user_data = ctx;
    return llama_init_from_model(ctx->model, cp);
}

std::vector<llama_token> tokenize(index_echo_context* ctx, const std::string& text) {
    int n = -llama_tokenize(ctx->vocab, text.data(), (int)text.size(), nullptr, 0, false, true);
    if (n <= 0)
        return {};
    std::vector<llama_token> ids(n);
    n = llama_tokenize(ctx->vocab, text.data(), (int)text.size(), ids.data(), n, false, true);
    if (n < 0)
        return {};
    ids.resize(n);
    return ids;
}

char* copy_string(const std::string& s) {
    char* p = (char*)malloc(s.size() + 1);
    if (p)
        memcpy(p, s.c_str(), s.size() + 1);
    return p;
}

float* copy_floats(const float* x, size_t count) {
    float* p = (float*)malloc(count * sizeof(float));
    if (p)
        memcpy(p, x, count * sizeof(float));
    return p;
}

// Only the last prompt token is retained at every decoder block. The scheduler
// callback reads before scratch reuse; no production synchronization overhead.
bool capture_decoder(ggml_tensor* t, bool ask, void* user) {
    auto* ctx = (index_echo_context*)user;
    if (!ctx->capture || strncmp(t->name, "post_ffn-", 9) != 0)
        return false;
    if (ask)
        return true;
    if (t->type != GGML_TYPE_F32 || t->ne[2] != 1 || t->ne[3] != 1)
        return false;
    const int layer = atoi(t->name + 9);
    auto& dst = ctx->stages["llm_block_" + std::to_string(layer)];
    dst.resize(t->ne[0]);
    ggml_backend_tensor_get(t, dst.data(), (t->ne[1] - 1) * t->nb[1], dst.size() * sizeof(float));
    return true;
}

bool decode(index_echo_context* ctx, const llama_token* ids, const float* embd, int count, int dim) {
    for (int start = 0; start < count; start += kBatch) {
        int n = std::min(kBatch, count - start);
        llama_batch batch = llama_batch_init(n, embd ? dim : 0, 1);
        batch.n_tokens = n;
        const auto rope = llama_model_rope_type(ctx->model);
        int position_axes = embd && (rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;
        if (!core_index_echo::positions(batch, ctx->position, position_axes)) {
            llama_batch_free(batch);
            return false;
        }
        for (int i = 0; i < n; ++i) {
            if (embd)
                memcpy(batch.embd + (size_t)i * dim, embd + (size_t)(start + i) * dim, dim * sizeof(float));
            else
                batch.token[i] = ids[start + i];
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i == n - 1;
        }
        int rc = llama_decode(ctx->decoder, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "index-echo: decoder failed at position %d (%d)\n", ctx->position, rc);
            return false;
        }
        ctx->position += n;
    }
    return true;
}

std::string prompt(index_echo_context* ctx, int rows) {
    std::string text = "<|im_start|>user\n<|audio_start|>";
    for (int i = 0; i < rows; ++i)
        text += "<|audio_pad|>";
    text += "<|audio_end|>\n";
    if (!ctx->context.empty())
        text += "[Context]\n" + ctx->context + "\n\n";
    if (!ctx->glossary.empty())
        text += "[Glossary]\n" + ctx->glossary + "\n\n";
    std::string language = ctx->lang == "ja" ? "Japanese" : ctx->lang == "es" ? "Spanish" : "English";
    text +=
        ctx->ask.empty()
            ? "For each sentence, output three lines: the [MM:SS.CC-MM:SS.CC] timestamp, the transcript, then the " +
                  language + " translation."
            : ctx->ask;
    text += "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return text;
}

bool connector_tensor(const char* name, void*) {
    return strncmp(name, "connector.", 10) == 0;
}

std::string generate(index_echo_context* ctx) {
    Bench bench("decoder_generate");
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (ctx->temperature > 0) {
        // HF GenerationConfig defaults: top_k=50, top_p=1.
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(50));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(ctx->temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(ctx->seed));
    } else
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    std::vector<llama_token> ids;
    bool ok = true;
    for (int i = 0; i < ctx->max_tokens; ++i) {
        llama_token token = llama_sampler_sample(sampler, ctx->decoder, -1);
        if (token == ctx->im_end || token == llama_vocab_eos(ctx->vocab))
            break;
        ids.push_back(token);
        if (i + 1 < ctx->max_tokens && !decode(ctx, &token, nullptr, 1, 0)) {
            ok = false;
            break;
        }
    }
    llama_sampler_free(sampler);
    if (!ok)
        throw std::runtime_error("generation decode failed");
    int n = -llama_detokenize(ctx->vocab, ids.data(), (int)ids.size(), nullptr, 0, true, false);
    if (n <= 0)
        return {};
    std::string text(n, '\0');
    n = llama_detokenize(ctx->vocab, ids.data(), (int)ids.size(), text.data(), n, true, false);
    if (n < 0)
        throw std::runtime_error("detokenization failed");
    text.resize(n);
    return text;
}

struct Cue {
    double start = 0, end = 0;
    std::string transcript, translation;
};

std::string trim(std::string text) {
    size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

std::vector<Cue> parse(const std::string& text, int& warnings) {
    static const std::regex timestamp(R"(^\s*\[(\d+):(\d+(?:\.\d+)?)-(\d+):(\d+(?:\.\d+)?)\]\s*(.*)$)");
    std::istringstream stream(text);
    std::string line;
    std::vector<Cue> cues;
    Cue current;
    bool active = false;
    auto finish = [&]() {
        if (!active)
            return;
        if (current.translation.empty())
            ++warnings;
        if (!current.transcript.empty())
            cues.push_back(current);
    };
    while (std::getline(stream, line)) {
        line = trim(line);
        if (line.empty())
            continue;
        std::smatch match;
        if (std::regex_match(line, match, timestamp)) {
            finish();
            current = {60 * std::stod(match[1]) + std::stod(match[2]), 60 * std::stod(match[3]) + std::stod(match[4]),
                       match[5], ""};
            active = true;
        } else if (!active)
            ++warnings;
        else if (current.transcript.empty())
            current.transcript = line;
        else if (current.translation.empty())
            current.translation = line;
        else
            ++warnings;
    }
    finish();
    return cues;
}
} // namespace

index_echo_context_params index_echo_context_default_params(void) {
    return {4, 1, true, true};
}

index_echo_context* index_echo_init_from_file(const char* path, index_echo_context_params params) {
    Bench bench("model_load");
    if (!path || params.n_threads <= 0)
        return nullptr;
    std::unique_ptr<index_echo_context, decltype(&index_echo_free)> ctx(new index_echo_context(), index_echo_free);
    ctx->params = params;
    static std::once_flag init;
    std::call_once(init, [] { llama_backend_init(); });
    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta)
        return nullptr;
    std::string arch = core_gguf::kv_str(meta, "general.architecture", "");
    std::string companion = core_gguf::kv_str(meta, "index_echo.decoder_file", "");
    std::string connector = core_gguf::kv_str(meta, "index_echo.connector_type", "residual");
    std::string vad_file = core_gguf::kv_str(meta, "index_echo.vad_file", "ggml-silero-v6.2.0.bin");
    core_gguf::free_metadata(meta);
    if (arch != "index_echo" || (connector != "residual" && connector != "projection") || companion.empty() ||
        std::filesystem::path(companion).is_absolute() ||
        companion != std::filesystem::path(companion).filename().string()) {
        fprintf(stderr, "index-echo: invalid architecture or decoder companion metadata\n");
        return nullptr;
    }
    ctx->projection = connector == "projection";
    auto decoder_path = std::filesystem::path(path).parent_path() / companion;
    auto ap = crisp_audio_params_default();
    ap.n_threads = params.n_threads;
    ap.verbosity = params.verbosity;
    ap.use_gpu = params.use_gpu;
    ctx->audio = crisp_audio_init_from_file(path, &ap);
    if (!ctx->audio)
        return nullptr;
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = params.use_gpu ? 999 : 0;
    ctx->model = llama_model_load_from_file(decoder_path.string().c_str(), mp);
    if (!ctx->model)
        return nullptr;
    ctx->vocab = llama_model_get_vocab(ctx->model);
    ctx->decoder = create_decoder(ctx.get(), 8192);
    if (!ctx->decoder)
        return nullptr;
    auto pad = tokenize(ctx.get(), "<|audio_pad|>");
    auto end = tokenize(ctx.get(), "<|im_end|>");
    if (pad.size() != 1 || end.size() != 1)
        return nullptr;
    ctx->audio_pad = pad[0];
    ctx->im_end = end[0];
    if (params.use_gpu) {
        auto device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (device)
            ctx->backend = ggml_backend_dev_init(device, nullptr);
    }
    if (!ctx->backend)
        ctx->backend = core_cpu_backend::init();
    if (!ctx->backend)
        return nullptr;
    if (!core_cpu_backend::is_cpu(ctx->backend))
        ctx->cpu = core_cpu_backend::init();
    core_cpu_backend::set_n_threads(ctx->cpu ? ctx->cpu : ctx->backend, params.n_threads);
    if (!core_gguf::load_weights_filtered(path, ctx->backend, connector_tensor, nullptr, "index-echo", ctx->weights))
        return nullptr;
    if (!core_index_echo::connector_matches(ctx->weights.tensors, ctx->projection, crisp_audio_output_dim(ctx->audio),
                                            llama_model_n_embd_inp(ctx->model))) {
        fprintf(stderr, "index-echo: invalid %s connector weights/dimensions\n", connector.c_str());
        return nullptr;
    }
    ggml_backend_t backends[] = {ctx->backend, ctx->cpu};
    ctx->sched = ggml_backend_sched_new(backends, nullptr, ctx->cpu ? 2 : 1, 64, false, false);
    if (!ctx->sched)
        return nullptr;
    if (!vad_file.empty() && std::filesystem::path(vad_file).filename().string() == vad_file) {
        auto vad_path = std::filesystem::path(path).parent_path() / vad_file;
        if (std::filesystem::exists(vad_path) && !index_echo_set_vad_model(ctx.get(), vad_path.string().c_str()))
            return nullptr;
    }
    return ctx.release();
}

void index_echo_free(index_echo_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    core_gguf::free_weights(ctx->weights);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    if (ctx->cpu)
        ggml_backend_free(ctx->cpu);
    if (ctx->decoder) {
        const char* bench = std::getenv("INDEX_ECHO_BENCH");
        if (bench && *bench && *bench != '0')
            llama_perf_context_print(ctx->decoder);
        llama_free(ctx->decoder);
    }
    if (ctx->model)
        llama_model_free(ctx->model);
    if (ctx->audio)
        crisp_audio_free(ctx->audio);
    if (ctx->vad)
        whisper_vad_free(ctx->vad);
    delete ctx;
}

bool index_echo_set_target_lang(index_echo_context* ctx, const char* lang) {
    if (!ctx || !lang || (strcmp(lang, "en") && strcmp(lang, "ja") && strcmp(lang, "es")))
        return false;
    ctx->lang = lang;
    return true;
}
void index_echo_set_temperature(index_echo_context* ctx, float t, uint32_t seed) {
    if (ctx) {
        ctx->temperature = std::max(0.0f, t);
        ctx->seed = seed;
    }
}
void index_echo_set_max_new_tokens(index_echo_context* ctx, int n) {
    if (ctx)
        ctx->max_tokens = n > 0 ? n : 2000;
}
void index_echo_set_ask(index_echo_context* ctx, const char* instruction) {
    if (ctx)
        ctx->ask = instruction ? instruction : "";
}
bool index_echo_set_vad_model(index_echo_context* ctx, const char* path) {
    if (!ctx)
        return false;
    if (!path || !*path) {
        if (ctx->vad)
            whisper_vad_free(ctx->vad);
        ctx->vad = nullptr;
        return true;
    }
    auto params = whisper_vad_default_context_params();
    params.n_threads = ctx->params.n_threads;
    params.use_gpu = ctx->params.use_gpu;
    auto* vad = whisper_vad_init_from_file_with_params(path, params);
    if (!vad)
        return false;
    if (!crispasr_silero_enable_context(vad)) {
        whisper_vad_free(vad);
        return false;
    }
    if (ctx->vad)
        whisper_vad_free(ctx->vad);
    ctx->vad = vad;
    return true;
}
void index_echo_set_glossary(index_echo_context* ctx, const char* raw) {
    if (!ctx)
        return;
    ctx->glossary.clear();
    std::string input = raw ? raw : "";
    for (const char* separator : {"，", "、", "；"}) {
        size_t pos;
        while ((pos = input.find(separator)) != std::string::npos)
            input.replace(pos, strlen(separator), "\n");
    }
    for (char& c : input)
        if (c == ',' || c == ';')
            c = '\n';
    std::istringstream lines(input);
    std::string line;
    const std::regex mapping(R"(^(.*?)\s*(?:→|->|=>|:|：)\s*(.*)$)");
    while (std::getline(lines, line)) {
        line = trim(line);
        if (line.empty())
            continue;
        std::smatch m;
        if (std::regex_match(line, m, mapping) && !trim(m[2]).empty())
            line = trim(m[1]) + " → " + trim(m[2]);
        if (!ctx->glossary.empty())
            ctx->glossary += '\n';
        ctx->glossary += line;
    }
}

float* index_echo_compute_mel(index_echo_context* ctx, const float* samples, int n, int* mels, int* frames) {
    Bench bench("mel");
    if (!ctx || !samples || n <= 0 || !mels || !frames)
        return nullptr;
    if (n > 300 * 16000)
        return nullptr;
    // The released frontend pads to 300 seconds BEFORE centered STFT, then
    // slices to ceil(n/160) attention-mask frames. Two extra zero frames cover
    // every remaining nonzero FFT window and preserve its global log clamp;
    // evaluating the rest of the 300 seconds would only repeat zero frames.
    int valid_frames = (n + 159) / 160;
    std::vector<float> padded((valid_frames + 2) * 160, 0);
    std::copy(samples, samples + n, padded.begin());
    int computed = 0;
    std::unique_ptr<float, decltype(&free)> full(
        crisp_audio_compute_mel(ctx->audio, padded.data(), (int)padded.size(), mels, &computed), free);
    if (!full || computed < valid_frames)
        return nullptr;
    float* result = (float*)malloc((size_t)*mels * valid_frames * sizeof(float));
    if (!result)
        return nullptr;
    for (int i = 0; i < *mels; ++i)
        memcpy(result + (size_t)i * valid_frames, full.get() + (size_t)i * computed, valid_frames * sizeof(float));
    *frames = valid_frames;
    return result;
}

float* index_echo_run_encoder(index_echo_context* ctx, const float* mel, int mels, int frames, int* rows, int* dim) {
    if (!ctx || !mel || !rows || !dim)
        return nullptr;
    std::unique_ptr<float, decltype(&free)> encoded(nullptr, free);
    {
        Bench bench("audio_encode");
        encoded.reset(crisp_audio_encode(ctx->audio, mel, mels, frames, rows, dim));
    }
    if (!encoded)
        return nullptr;
    Bench bench("connector");
    std::vector<uint8_t> metadata(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
    ggml_init_params gp = {metadata.size(), metadata.data(), true};
    ggml_context* graph_ctx = ggml_init(gp);
    if (!graph_ctx)
        return nullptr;
    auto* input = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, *dim, *rows);
    ggml_set_name(input, "connector_input");
    ggml_set_input(input);
    auto* output = core_index_echo::connector_graph(graph_ctx, input, ctx->weights.tensors, ctx->projection);
    ggml_set_output(output);
    auto* graph = ggml_new_graph_custom(graph_ctx, 64, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_sched_reset(ctx->sched);
    float* result = nullptr;
    if (ggml_backend_sched_alloc_graph(ctx->sched, graph)) {
        ggml_backend_tensor_set(input, encoded.get(), 0, ggml_nbytes(input));
        if (ggml_backend_sched_graph_compute(ctx->sched, graph) == GGML_STATUS_SUCCESS) {
            result = (float*)malloc(ggml_nbytes(output));
            if (result) {
                ggml_backend_tensor_get(output, result, 0, ggml_nbytes(output));
                *dim = (int)output->ne[0];
            }
        }
    }
    ggml_backend_sched_reset(ctx->sched);
    ggml_free(graph_ctx);
    return result;
}

int index_echo_decoder_layers(index_echo_context* ctx) {
    return ctx && ctx->model ? llama_model_n_layer(ctx->model) : 0;
}

float* index_echo_prefill(index_echo_context* ctx, const float* audio, int rows, int dim, int* n_vocab) {
    Bench bench("decoder_prefill");
    if (!ctx || !audio || rows <= 0 || dim != llama_model_n_embd_inp(ctx->model) || !n_vocab)
        return nullptr;
    ctx->stages.clear();
    ctx->prompt_ids = tokenize(ctx, prompt(ctx, rows));
    auto first = std::find(ctx->prompt_ids.begin(), ctx->prompt_ids.end(), ctx->audio_pad);
    size_t begin = first - ctx->prompt_ids.begin();
    if (first == ctx->prompt_ids.end() || begin + rows > ctx->prompt_ids.size() ||
        std::count(ctx->prompt_ids.begin(), ctx->prompt_ids.end(), ctx->audio_pad) != rows ||
        !std::all_of(first, first + rows, [&](int t) { return t == ctx->audio_pad; }))
        return nullptr;
    size_t needed = ctx->prompt_ids.size() + (size_t)ctx->max_tokens;
    if (needed > (size_t)llama_model_n_ctx_train(ctx->model)) {
        fprintf(stderr, "index-echo: prompt and generation exceed the decoder's trained context\n");
        return nullptr;
    }
    if (!ctx->decoder || needed > llama_n_ctx(ctx->decoder)) {
        if (ctx->decoder)
            llama_free(ctx->decoder);
        ctx->decoder = create_decoder(ctx, (uint32_t)needed);
        if (!ctx->decoder)
            return nullptr;
    }
    llama_memory_clear(llama_get_memory(ctx->decoder), true);
    ctx->position = 0;
    ctx->capture = getenv("CRISPASR_INDEX_ECHO_DUMP_STAGES") != nullptr;
    bool ok =
        decode(ctx, ctx->prompt_ids.data(), nullptr, (int)begin, 0) && decode(ctx, nullptr, audio, rows, dim) &&
        decode(ctx, ctx->prompt_ids.data() + begin + rows, nullptr, (int)(ctx->prompt_ids.size() - begin - rows), 0);
    ctx->capture = false;
    if (!ok)
        return nullptr;
    *n_vocab = llama_vocab_n_tokens(ctx->vocab);
    return copy_floats(llama_get_logits_ith(ctx->decoder, -1), *n_vocab);
}

const float* index_echo_stage(index_echo_context* ctx, const char* name, int* count) {
    if (!ctx || !name || !count)
        return nullptr;
    auto it = ctx->stages.find(name);
    if (it == ctx->stages.end())
        return nullptr;
    *count = (int)it->second.size();
    return it->second.data();
}
float* index_echo_decode_token(index_echo_context* ctx, int32_t token, int* n_vocab) {
    if (!ctx || !n_vocab || token < 0 || token >= llama_vocab_n_tokens(ctx->vocab) ||
        ctx->position >= (int)llama_n_ctx(ctx->decoder))
        return nullptr;
    if (!decode(ctx, &token, nullptr, 1, 0))
        return nullptr;
    *n_vocab = llama_vocab_n_tokens(ctx->vocab);
    return copy_floats(llama_get_logits_ith(ctx->decoder, -1), *n_vocab);
}
const int32_t* index_echo_prompt_ids(index_echo_context* ctx, int* count) {
    if (!ctx || !count)
        return nullptr;
    *count = (int)ctx->prompt_ids.size();
    return ctx->prompt_ids.data();
}

index_echo_result* index_echo_transcribe(index_echo_context* ctx, const float* samples, int n) {
    if (!ctx || !samples || n <= 0)
        return nullptr;
    try {
        std::vector<Cue> all;
        std::vector<std::string> history;
        std::string raw;
        int warnings = 0;
        ctx->context.clear();
        std::vector<core_index_echo::Window> windows;
        if (ctx->vad) {
            if (!whisper_vad_detect_speech(ctx->vad, samples, n))
                return nullptr;
            auto speech = core_index_echo::speech_spans(whisper_vad_probs(ctx->vad), whisper_vad_n_probs(ctx->vad), n);
            windows = core_index_echo::windows(speech, n / 16000.0);
        } else {
            if (ctx->params.verbosity)
                fprintf(stderr, "index-echo: no Silero companion; using bounded 60s windows\n");
            for (int start = 0; start < n;) {
                const int end = start + std::min(n - start, 60 * 16000);
                windows.push_back({start / 16000.0, end / 16000.0, true});
                start = end;
            }
        }
        for (const auto& window : windows) {
            // Upstream formats ffmpeg seek/duration to milliseconds.
            int begin = (int)std::clamp<int64_t>(core_index_echo::window_samples(window.start), 0, n);
            int length =
                (int)std::clamp<int64_t>(core_index_echo::window_samples(window.end - window.start), 0, n - begin);
            if (length <= 0)
                continue;
            int mels = 0, frames = 0, rows = 0, dim = 0, vocab = 0;
            std::unique_ptr<float, decltype(&free)> mel(
                index_echo_compute_mel(ctx, samples + begin, length, &mels, &frames), free);
            if (!mel)
                return nullptr;
            std::unique_ptr<float, decltype(&free)> emb(
                index_echo_run_encoder(ctx, mel.get(), mels, frames, &rows, &dim), free);
            if (!emb)
                return nullptr;
            std::unique_ptr<float, decltype(&free)> logits(index_echo_prefill(ctx, emb.get(), rows, dim, &vocab), free);
            if (!logits)
                return nullptr;
            std::string text = trim(generate(ctx));
            if (!raw.empty())
                raw += '\n';
            raw += text;
            auto cues = parse(text, warnings);
            if (cues.empty() && !ctx->ask.empty() && !text.empty())
                cues.push_back({0, length / 16000.0, text, ""});
            std::string context;
            for (auto& cue : cues) {
                if (!cue.translation.empty()) {
                    if (!context.empty())
                        context += '\n';
                    context += cue.transcript + '\n' + cue.translation;
                }
                cue.start += window.start;
                cue.end += window.start;
                all.push_back(std::move(cue));
            }
            history.push_back(context);
            ctx->context.clear();
            for (size_t i = history.size() > 5 ? history.size() - 5 : 0; i < history.size(); ++i) {
                if (history[i].empty())
                    continue;
                if (!ctx->context.empty())
                    ctx->context += '\n';
                ctx->context += history[i];
            }
        }
        auto* result = (index_echo_result*)calloc(1, sizeof(index_echo_result));
        if (!result)
            return nullptr;
        result->raw_text = copy_string(raw);
        result->parse_warnings = warnings;
        result->n_cues = (int)all.size();
        result->cues = (index_echo_cue*)calloc(all.size(), sizeof(index_echo_cue));
        if (!result->raw_text || (!all.empty() && !result->cues)) {
            index_echo_result_free(result);
            return nullptr;
        }
        for (size_t i = 0; i < all.size(); ++i) {
            result->cues[i] = {all[i].start, all[i].end, copy_string(all[i].transcript),
                               copy_string(all[i].translation)};
            if (!result->cues[i].transcript || !result->cues[i].translation) {
                index_echo_result_free(result);
                return nullptr;
            }
        }
        ctx->context.clear();
        return result;
    } catch (const std::exception& e) {
        fprintf(stderr, "index-echo: %s\n", e.what());
        ctx->context.clear();
        return nullptr;
    }
}

void index_echo_result_free(index_echo_result* result) {
    if (!result)
        return;
    if (result->cues)
        for (int i = 0; i < result->n_cues; ++i) {
            free(result->cues[i].transcript);
            free(result->cues[i].translation);
        }
    free(result->cues);
    free(result->raw_text);
    free(result);
}
