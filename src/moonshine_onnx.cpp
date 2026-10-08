#include "moonshine_onnx.h"
#include "moonshine-tokenizer.h"
#include "../examples/json.hpp"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
Ort::Env& env() {
    static Ort::Env e(ORT_LOGGING_LEVEL_WARNING, "crispasr-moonshine");
    return e;
}
Ort::MemoryInfo& cpu() {
    static auto m = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return m;
}
Ort::AllocatorWithDefaultOptions& allocator() {
    static Ort::AllocatorWithDefaultOptions a;
    return a;
}

template <class T> Ort::Value tensor(std::vector<T>& data, std::vector<int64_t> shape) {
    return Ort::Value::CreateTensor<T>(cpu(), data.data(), data.size(), shape.data(), shape.size());
}
Ort::Value zeros(std::vector<int64_t> shape, ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    auto v = Ort::Value::CreateTensor(allocator(), shape.data(), shape.size(), type);
    size_t size = v.GetTensorTypeAndShapeInfo().GetElementCount();
    if (size)
        memset(v.GetTensorMutableRawData(), 0, size * (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ? 8 : 4));
    return v;
}
struct Graph {
    Ort::Session session{nullptr};
    std::vector<std::string> inputs, outputs;
    Graph(const std::filesystem::path& path, int threads) {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(std::max(1, threads));
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = Ort::Session(env(), path.c_str(), options);
        for (size_t i = 0; i < session.GetInputCount(); ++i)
            inputs.emplace_back(session.GetInputNameAllocated(i, allocator()).get());
        for (size_t i = 0; i < session.GetOutputCount(); ++i)
            outputs.emplace_back(session.GetOutputNameAllocated(i, allocator()).get());
    }
    std::vector<Ort::Value> run(std::vector<Ort::Value>& values) {
        std::vector<const char*> in, out;
        for (const auto& s : inputs)
            in.push_back(s.c_str());
        for (const auto& s : outputs)
            out.push_back(s.c_str());
        return session.Run(Ort::RunOptions{nullptr}, in.data(), values.data(), values.size(), out.data(), out.size());
    }
};
std::vector<float> floats(const Ort::Value& v) {
    auto n = v.GetTensorTypeAndShapeInfo().GetElementCount();
    const float* p = v.GetTensorData<float>();
    return {p, p + n};
}
void error(const char* where, const std::exception& e) {
    fprintf(stderr, "moonshine-onnx %s: %s\n", where, e.what());
}
} // namespace

struct moonshine_onnx_context {
    nlohmann::json config;
    bool incremental = false;
    int depth = 0, heads = 0, head_dim = 0, enc_dim = 0, dec_dim = 0, vocab = 0, bos = 1, eos = 2, lookahead = 0,
        left = 0, max_positions = 4096;
    std::unique_ptr<Graph> frontend, encoder, adapter, cross, decoder;
    moonshine_tokenizer tokenizer;
};

static std::filesystem::path config_file(const char* path) {
    std::filesystem::path p(path);
    if (p.extension() != ".onnx")
        return p;
    auto dir = p.parent_path();
    if (dir.filename() == "onnx")
        dir = dir.parent_path();
    if (std::filesystem::exists(dir / "streaming_config.json"))
        return dir / "streaming_config.json";
    return dir / "config.json";
}
bool moonshine_onnx_is_model(const char* path) {
    try {
        if (!path || std::filesystem::path(path).extension() != ".onnx")
            return false;
        std::ifstream f(config_file(path));
        auto j = nlohmann::json::parse(f);
        return j.contains("frontend_state_shapes") || j.value("model_type", "") == "moonshine";
    } catch (...) {
        return false;
    }
}

moonshine_onnx_context* moonshine_onnx_open(const char* config_path, int threads) {
    try {
        if (!config_path || !*config_path)
            return nullptr;
        auto ctx = std::make_unique<moonshine_onnx_context>();
        std::filesystem::path p = config_file(config_path), dir = p.parent_path();
        std::ifstream f(p);
        ctx->config = nlohmann::json::parse(f);
        auto& c = ctx->config;
        ctx->incremental = c.contains("frontend_state_shapes");
        ctx->vocab = c.at("vocab_size");
        if (ctx->incremental) {
            ctx->depth = c.at("depth");
            ctx->heads = c.at("nheads");
            ctx->head_dim = c.at("head_dim");
            ctx->enc_dim = c.at("encoder_dim");
            ctx->dec_dim = c.at("decoder_dim");
            ctx->bos = c.at("bos_id");
            ctx->eos = c.at("eos_id");
            ctx->lookahead = c.at("total_lookahead");
            ctx->left = c.at("total_left_context");
            ctx->max_positions = c.value("max_position_embeddings", 4096);
            // Prefer the complete int8 bundle; never mix missing graphs silently.
            bool quant = std::filesystem::path(config_path).filename() != "encoder.onnx" &&
                         std::filesystem::exists(dir / "encoder_int8.onnx");
            auto graph = [&](const char* name) {
                return std::make_unique<Graph>(dir / (std::string(name) + (quant ? "_int8.onnx" : ".onnx")), threads);
            };
            ctx->frontend = std::make_unique<Graph>(dir / "frontend.onnx", threads);
            ctx->encoder = graph("encoder");
            ctx->adapter = graph("adapter");
            ctx->cross = graph("cross_kv");
            ctx->decoder = graph("decoder_kv");
        } else {
            ctx->depth = c.at("decoder_num_hidden_layers");
            ctx->heads = c.at("decoder_num_attention_heads");
            ctx->enc_dim = c.at("hidden_size");
            ctx->dec_dim = ctx->enc_dim;
            ctx->head_dim = ctx->dec_dim / ctx->heads;
            ctx->bos = c.value("decoder_start_token_id", 1);
            ctx->eos = c.value("eos_token_id", 2);
            ctx->encoder = std::make_unique<Graph>(dir / "onnx/encoder_model.onnx", threads);
            ctx->decoder = std::make_unique<Graph>(dir / "onnx/decoder_model_merged.onnx", threads);
        }
        if (!ctx->tokenizer.load_json((dir / "tokenizer.json").string().c_str()))
            return nullptr;
        if (ctx->vocab < 1 || ctx->vocab > 1000000 || ctx->tokenizer.vocab_size() > static_cast<size_t>(ctx->vocab))
            throw std::runtime_error("invalid tokenizer/model vocabulary");
        return ctx.release();
    } catch (const std::exception& e) {
        error("open", e);
        return nullptr;
    }
}
void moonshine_onnx_close(moonshine_onnx_context* ctx) {
    delete ctx;
}
bool moonshine_onnx_incremental(const moonshine_onnx_context* ctx) {
    return ctx && ctx->incremental;
}

struct moonshine_onnx_stream {
    moonshine_onnx_context* ctx;
    std::vector<Ort::Value> frontend_state;
    std::vector<float> pending, features, encoded;
    int stable = 0, next_decode = 0, step = 5120;
    int64_t samples = 0, counter = 0;
    bool flushed = false, failed = false;
    std::string text;
    std::vector<int32_t> draft_tokens;
    explicit moonshine_onnx_stream(moonshine_onnx_context* c, int ms) : ctx(c) {
        step = std::max(640, (ms > 0 ? ms : 320) * 16);
        next_decode = step;
        for (const char* name : {"sample_buffer", "sample_len", "conv1_buffer", "conv2_buffer", "frame_count"}) {
            auto shape = c->config.at("frontend_state_shapes").at(name).get<std::vector<int64_t>>();
            bool integer = std::string(name) == "sample_len" || std::string(name) == "frame_count";
            frontend_state.push_back(
                zeros(shape, integer ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        }
    }
};

static std::string decode_memory(moonshine_onnx_context* c, std::vector<float>& memory, int frames, double seconds,
                                 std::vector<int32_t>* draft = nullptr) {
    if (frames <= 0)
        return {};
    std::vector<Ort::Value> cross_in;
    cross_in.push_back(tensor(memory, {1, frames, c->dec_dim}));
    auto cross = c->cross->run(cross_in);
    auto k = zeros({c->depth, 1, c->heads, 0, c->head_dim});
    auto v = zeros({c->depth, 1, c->heads, 0, c->head_dim});
    std::vector<int32_t> result;
    std::vector<int64_t> token{c->bos};
    // Prefill older draft tokens in one call and allow the trailing eight to
    // revise. Final flush decodes from BOS, avoiding draft error lock-in.
    if (draft && draft->size() > 8) {
        result.assign(draft->begin(), draft->end() - 8);
        token.insert(token.end(), result.begin(), result.end());
    }
    const int budget = std::min(1024, std::max(4, static_cast<int>(seconds * 6.5) + 2));
    for (int i = static_cast<int>(result.size()); i < budget; ++i) {
        std::vector<Ort::Value> input;
        input.push_back(tensor(token, {1, static_cast<int64_t>(token.size())}));
        input.push_back(std::move(k));
        input.push_back(std::move(v));
        input.push_back(std::move(cross[0]));
        input.push_back(std::move(cross[1]));
        auto out = c->decoder->run(input);
        auto info = out[0].GetTensorTypeAndShapeInfo();
        const float* logits = out[0].GetTensorData<float>() + info.GetElementCount() - c->vocab;
        int best = static_cast<int>(std::max_element(logits, logits + c->vocab) - logits);
        k = std::move(out[1]);
        v = std::move(out[2]);
        cross[0] = std::move(out[3]);
        cross[1] = std::move(out[4]);
        if (best == c->eos)
            break;
        result.push_back(best);
        token.assign(1, best);
        // A repeated 8-token suffix is a decoder loop, not more speech.
        if (result.size() >= 16 && std::equal(result.end() - 8, result.end(), result.end() - 16))
            break;
    }
    if (draft)
        *draft = result;
    return c->tokenizer.tokens_to_text(result);
}
static void frontend_chunk(moonshine_onnx_stream* s, std::vector<float>& audio) {
    std::vector<Ort::Value> inputs;
    inputs.push_back(tensor(audio, {1, static_cast<int64_t>(audio.size())}));
    for (auto& v : s->frontend_state)
        inputs.push_back(std::move(v));
    auto out = s->ctx->frontend->run(inputs);
    auto feat = floats(out[0]);
    s->features.insert(s->features.end(), feat.begin(), feat.end());
    for (size_t i = 0; i < s->frontend_state.size(); ++i)
        s->frontend_state[i] = std::move(out[i + 1]);
}
static void update(moonshine_onnx_stream* s, bool final, int64_t processed = -1) {
    auto* c = s->ctx;
    int frames = static_cast<int>(s->features.size() / c->enc_dim);
    if (frames == 0)
        return;
    if (frames > c->max_positions)
        throw std::runtime_error("Moonshine utterance exceeds positional capacity; split at a pause");
    // Encoder work is bounded to the dependency cone of the newly stable frames.
    // Previously stable output is immutable; unstable right-context is replaced.
    int start = std::max(0, s->stable - c->left);
    std::vector<float> window(s->features.begin() + static_cast<size_t>(start) * c->enc_dim, s->features.end());
    std::vector<Ort::Value> input;
    input.push_back(tensor(window, {1, frames - start, c->enc_dim}));
    auto out = c->encoder->run(input);
    auto fresh = floats(out[0]);
    s->encoded.resize(static_cast<size_t>(s->stable) * c->enc_dim);
    s->encoded.insert(s->encoded.end(), fresh.begin() + static_cast<size_t>(s->stable - start) * c->enc_dim,
                      fresh.end());
    s->stable = final ? frames : std::max(s->stable, frames - c->lookahead);
    std::vector<int64_t> offset{0};
    std::vector<Ort::Value> adapt;
    adapt.push_back(tensor(s->encoded, {1, frames, c->enc_dim}));
    adapt.push_back(tensor(offset, {1}));
    auto adapted = c->adapter->run(adapt);
    auto memory = floats(adapted[0]);
    s->text = decode_memory(c, memory, frames, (processed < 0 ? s->samples : processed) / 16000.0,
                            final ? nullptr : &s->draft_tokens);
    ++s->counter;
}
moonshine_onnx_stream* moonshine_onnx_stream_open(moonshine_onnx_context* c, int ms) {
    if (!c || !c->incremental)
        return nullptr;
    try {
        return new moonshine_onnx_stream(c, ms);
    } catch (const std::exception& e) {
        error("stream open", e);
        return nullptr;
    }
}
int moonshine_onnx_stream_feed(moonshine_onnx_stream* s, const float* pcm, int n) {
    if (!s || s->flushed || s->failed || n < 0 || (n && !pcm))
        return -1;
    try {
        if (n)
            s->pending.insert(s->pending.end(), pcm, pcm + n);
        s->samples += n;
        // Canonical 40 ms frontend packets: results do not depend on callers'
        // microphone packet sizes or the order of Dart message delivery.
        size_t consumed = 0;
        while (s->pending.size() - consumed >= 640) {
            std::vector<float> chunk(s->pending.begin() + consumed, s->pending.begin() + consumed + 640);
            frontend_chunk(s, chunk);
            consumed += 640;
            int64_t processed = s->samples - static_cast<int64_t>(s->pending.size() - consumed);
            if (processed >= s->next_decode) {
                update(s, false, processed);
                s->next_decode += s->step;
            }
        }
        s->pending.erase(s->pending.begin(), s->pending.begin() + consumed);
        return n;
    } catch (const std::exception& e) {
        error("feed", e);
        s->failed = true;
        return -1;
    }
}
int moonshine_onnx_stream_flush(moonshine_onnx_stream* s) {
    if (!s || s->failed)
        return -1;
    if (s->flushed)
        return 0;
    try {
        if (!s->pending.empty()) {
            s->pending.resize(640, 0);
            frontend_chunk(s, s->pending);
            s->pending.clear();
        }
        update(s, true);
        s->flushed = true;
        return 0;
    } catch (const std::exception& e) {
        error("flush", e);
        s->failed = true;
        return -1;
    }
}
int moonshine_onnx_stream_get_text(moonshine_onnx_stream* s, char* out, int cap, double* t0, double* t1,
                                   int64_t* counter) {
    if (!s || s->failed)
        return -1;
    if (t0)
        *t0 = 0;
    if (t1)
        *t1 = s->samples / 16000.0;
    if (counter)
        *counter = s->counter;
    int n = static_cast<int>(s->text.size());
    if (out && cap > 0) {
        int written = std::min(n, cap - 1);
        memcpy(out, s->text.data(), written);
        out[written] = 0;
    }
    return n;
}
void moonshine_onnx_stream_close(moonshine_onnx_stream* s) {
    delete s;
}

static std::string legacy_transcribe(moonshine_onnx_context* c, const float* pcm, int n) {
    if (n < 1024)
        return {};
    std::vector<float> audio(pcm, pcm + n);
    std::vector<Ort::Value> in;
    in.push_back(tensor(audio, {1, n}));
    auto encoded = c->encoder->run(in);
    auto hidden = floats(encoded[0]);
    auto shape = encoded[0].GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> ids{c->bos};
    std::vector<int32_t> result;
    int budget = std::min(512, std::max(4, static_cast<int>(n / 16000.0 * 10) + 2));
    // The merged V1 export's cached branch is defective for some checkpoints.
    // Use the uncached branch with the complete prefix; never feed zero cross-KV
    // into that defective branch and silently return plausible wrong text.
    for (int i = 0; i < budget; ++i) {
        std::vector<Ort::Value> values;
        for (size_t j = 0; j < c->decoder->inputs.size(); ++j) {
            const auto& name = c->decoder->inputs[j];
            if (name == "input_ids")
                values.push_back(tensor(ids, {1, static_cast<int64_t>(ids.size())}));
            else if (name == "encoder_hidden_states")
                values.push_back(tensor(hidden, shape));
            else if (name == "use_cache_branch") {
                auto value = Ort::Value::CreateTensor<bool>(allocator(), std::vector<int64_t>{1}.data(), 1);
                *value.GetTensorMutableData<bool>() = false;
                values.push_back(std::move(value));
            } else if (name.find("past_key_values") != std::string::npos)
                values.push_back(zeros({1, c->heads, 0, c->head_dim}));
            else
                throw std::runtime_error("unsupported decoder input: " + name);
        }
        auto out = c->decoder->run(values);
        auto size = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
        const float* p = out[0].GetTensorData<float>() + size - c->vocab;
        int best = static_cast<int>(std::max_element(p, p + c->vocab) - p);
        if (best == c->eos)
            break;
        ids.push_back(best);
        result.push_back(best);
        if (result.size() >= 16 && std::equal(result.end() - 8, result.end(), result.end() - 16))
            break;
    }
    return c->tokenizer.tokens_to_text(result);
}
std::string moonshine_onnx_transcribe(moonshine_onnx_context* c, const float* pcm, int n) {
    if (!c || n < 0 || (n && !pcm))
        throw std::runtime_error("invalid Moonshine audio");
    if (!c->incremental)
        return legacy_transcribe(c, pcm, n);
    // Batch reference: run the frontend once, then the whole masked encoder.
    auto s = std::unique_ptr<moonshine_onnx_stream>(moonshine_onnx_stream_open(c, 320));
    if (!s)
        throw std::runtime_error("failed to initialize streaming frontend");
    s->samples = n;
    if (n) {
        std::vector<float> audio(pcm, pcm + n);
        audio.resize((audio.size() + 639) / 640 * 640, 0);
        frontend_chunk(s.get(), audio);
    }
    update(s.get(), true);
    return s->text;
}
