// Native graph implementation of HiDolen Mini-BS-RoFormer V2.5 (MIT).
// All learned operations stay in ggml: batched axial GQA/NeoX RoPE,
// SwiGLU temporal grouping, probe pooling and the shared complex-mask decoder.
// Cached radix-2 transforms and overlap-add require no Python runtime.
#include "mini_bs_roformer.h"
#include "mini_bs_roformer_bf16.h"
#include "mini_bs_roformer_fft.h"
#include "core/crispasr_env.h"
#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/parallel_for.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// Named predicate rather than a lambda: cppcheck 2.7 misattributes a lambda's
// bool return to the enclosing pointer-returning function.
static bool is_finite_sample(float x) {
    return std::isfinite(x);
}

namespace {
constexpr size_t GRAPH_SIZE = 8192;
const std::string MODEL = "freq_domain_model.mask_model.";
const char* STEMS[] = {"bass", "drums", "other", "vocals"};

bool env_flag(const char* key, bool fallback = false) {
    const char* v = std::getenv(key);
    return v && *v ? std::strcmp(v, "0") != 0 : fallback;
}

struct StageTimer {
    const char* name;
    bool enabled = env_flag("CRISPASR_MINI_BS_BENCH");
    std::chrono::steady_clock::time_point start;
    explicit StageTimer(const char* n) : name(n) {
        if (enabled)
            start = std::chrono::steady_clock::now();
    }
    ~StageTimer() {
        if (enabled)
            std::fprintf(stderr, "mini_bs_roformer_bench: %s %.3f ms\n", name,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
};

struct HParams {
    int hidden = 0, band_dim = 0, intermediate = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0;
    int channels = 0, stems = 0, registers = 0, group = 0, time_depth = 0, freq_depth = 0;
    int fft = 0, fft_out = 0, hop = 0, chunk = 0, rate = 0;
    float eps = 0, rope = 0;
    std::vector<int> bands, bands_out;
};

std::vector<int> int_array(gguf_context* m, const char* key) {
    const int64_t k = gguf_find_key(m, key);
    if (k < 0 || gguf_get_kv_type(m, k) != GGUF_TYPE_ARRAY)
        return {};
    const auto type = gguf_get_arr_type(m, k);
    if (type != GGUF_TYPE_INT32 && type != GGUF_TYPE_UINT32)
        return {};
    const size_t n = gguf_get_arr_n(m, k);
    if (n > 1024)
        return {};
    const int32_t* values = static_cast<const int32_t*>(gguf_get_arr_data(m, k));
    return std::vector<int>(values, values + n);
}

bool read_hparams(gguf_context* meta, HParams& h) {
    if (core_gguf::kv_str(meta, "general.architecture", "") != "mini-bs-roformer")
        return false;
    const std::string p = "mini_bs_roformer.";
    auto integer = [&](const char* key) { return (int)core_gguf::kv_u32(meta, (p + key).c_str(), 0); };
    h.hidden = integer("hidden_size");
    h.band_dim = integer("band_proj_size");
    h.intermediate = integer("intermediate_size");
    h.layers = integer("num_hidden_layers");
    h.heads = integer("num_attention_heads");
    h.kv_heads = integer("num_key_value_heads");
    h.head_dim = integer("head_dim");
    h.channels = integer("num_input_channels");
    h.stems = integer("num_stems");
    h.registers = integer("register_token_num");
    h.group = integer("time_conv_length");
    h.time_depth = integer("time_transformer_depth");
    h.freq_depth = integer("freq_transformer_depth");
    h.fft = integer("stft_n_fft");
    h.fft_out = integer("stft_n_fft_out");
    h.hop = integer("stft_hop_length");
    h.chunk = integer("wave_chunk_size");
    h.rate = integer("wave_sample_rate");
    h.eps = core_gguf::kv_f32(meta, (p + "rms_norm_eps").c_str(), 0);
    h.rope = core_gguf::kv_f32(meta, (p + "rope_base").c_str(), 0);
    h.bands = int_array(meta, (p + "freqs_per_bands").c_str());
    h.bands_out = int_array(meta, (p + "freqs_per_bands_out").c_str());
    // This is an explicitly versioned architecture, not a speculative loader
    // for arbitrary future checkpoints. Reject incompatible metadata before a
    // ggml shape assertion can terminate the calling application.
    return h.hidden == 512 && h.band_dim == 256 && h.intermediate == 1024 && h.layers == 6 && h.heads == 8 &&
           h.kv_heads == 4 && h.head_dim == 64 && h.channels == 2 && h.stems == 4 && h.registers == 4 && h.group == 4 &&
           h.time_depth == 1 && h.freq_depth == 1 && h.fft == 4096 && h.fft_out == 2048 && h.hop == 512 &&
           h.chunk == 352800 && h.rate == 44100 && std::isfinite(h.eps) && h.eps == 1e-6f && h.rope == 10000.0f &&
           h.bands.size() == 80 && h.bands_out.size() == 80 &&
           std::all_of(h.bands.begin(), h.bands.end(), [](int n) { return n > 0 && n < 2049; }) &&
           std::all_of(h.bands_out.begin(), h.bands_out.end(), [](int n) { return n > 0 && n < 1025; }) &&
           std::accumulate(h.bands.begin(), h.bands.end(), 0) == h.fft / 2 + 1 &&
           std::accumulate(h.bands_out.begin(), h.bands_out.end(), 0) == h.fft_out / 2 + 1;
}

struct Graph {
    struct WeightOperand {
        ggml_tensor* values = nullptr;
        ggml_tensor* packed = nullptr;
    };
    ggml_context* ctx = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor *input = nullptr, *positions_t = nullptr, *positions_f = nullptr;
    ggml_tensor *hidden = nullptr, *embeddings = nullptr, *stem = nullptr, *mask = nullptr;
    std::array<ggml_tensor*, 4> stem_views{};
    std::array<ggml_tensor*, 4> stem_masks{};
    std::map<std::string, ggml_tensor*> captures;
    std::map<ggml_tensor*, ggml_tensor*> bf16_inputs;
    std::map<ggml_tensor*, WeightOperand> bf16_weights;
    int frames = 0;
    bool capture = false;
    ~Graph() { clear(); }
    void clear() {
        if (sched)
            ggml_backend_sched_free(sched);
        if (ctx)
            ggml_free(ctx);
        sched = nullptr;
        ctx = nullptr;
        graph = nullptr;
        input = positions_t = positions_f = nullptr;
        hidden = embeddings = stem = mask = nullptr;
        stem_views.fill(nullptr);
        stem_masks.fill(nullptr);
        captures.clear();
        bf16_inputs.clear();
        bf16_weights.clear();
        frames = 0;
    }
};

using Captures = std::map<std::string, std::vector<float>>;

std::vector<float> tensor_f32(ggml_tensor* t) {
    if (!t || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t))
        throw std::runtime_error("expected contiguous F32 capture tensor");
    std::vector<float> out((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    return out;
}

bool matmul_weight(const char* name, void*) {
    const std::string n(name);
    return n.size() >= 7 && n.compare(n.size() - 7, 7, ".weight") == 0;
}
} // namespace

struct mini_bs_roformer_context {
    HParams hp;
    mini_bs_roformer_params params{};
    core_gguf::WeightLoad weights;
    ggml_backend_t backend = nullptr, cpu = nullptr, blas = nullptr;
    Graph encoder, decoder;
    std::vector<float> window_in, window_out;
    std::unique_ptr<mini_bs_fft::Plan> fft_in, fft_out;
    bool quant_q4 = false, quant_q8 = false;
    bool flash = true;
    bool dense_frequency_attention = false;
    bool cpu_bf16 = false;
    bool cpu_packed_bf16 = false;
    bool shared_decoder = false;
};

namespace {
ggml_tensor* weight(mini_bs_roformer_context& m, const std::string& n) {
    const auto it = m.weights.tensors.find(n);
    if (it == m.weights.tensors.end())
        throw std::runtime_error("missing tensor: " + n);
    return it->second;
}

void expect(mini_bs_roformer_context& m, const std::string& n, int64_t d0, int64_t d1 = 1, int64_t d2 = 1) {
    const auto* w = weight(m, n);
    if (w->ne[0] != d0 || w->ne[1] != d1 || w->ne[2] != d2 || w->ne[3] != 1)
        throw std::runtime_error("incompatible tensor shape: " + n);
    const bool matrix = matmul_weight(n.c_str(), nullptr);
    if (!matrix && w->type != GGML_TYPE_F32)
        throw std::runtime_error("sensitive tensor must be F32: " + n);
    if (matrix && w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16 && w->type != GGML_TYPE_Q8_0 &&
        w->type != GGML_TYPE_Q4_K)
        throw std::runtime_error("unsupported matrix precision: " + n);
}

void validate_weights(mini_bs_roformer_context& m) {
    const auto& h = m.hp;
    auto mlp = [&](const std::string& p, int in, int mid, int out) {
        expect(m, p + ".gate_proj.weight", in, mid);
        expect(m, p + ".up_proj.weight", in, mid);
        expect(m, p + ".down_proj.weight", mid, out);
    };
    auto attention = [&](const std::string& p) {
        expect(m, p + ".q_proj.weight", h.hidden, h.heads * h.head_dim);
        expect(m, p + ".k_proj.weight", h.hidden, h.kv_heads * h.head_dim);
        expect(m, p + ".v_proj.weight", h.hidden, h.kv_heads * h.head_dim);
        expect(m, p + ".o_proj.weight", h.heads * h.head_dim, h.hidden);
    };
    for (size_t b = 0; b < h.bands.size(); ++b) {
        const std::string s = std::to_string(b);
        expect(m, MODEL + "band_split.to_features." + s + ".0.scale", h.bands[b] * 4);
        expect(m, MODEL + "band_split.to_features." + s + ".1.weight", h.bands[b] * 4, h.band_dim);
        expect(m, MODEL + "mask_estimator.to_freq_mlps." + s + ".weight", h.band_dim, h.bands_out[b] * 4);
        expect(m, MODEL + "mask_estimator.to_gate_mlps." + s + ".weight", h.band_dim, h.bands_out[b] * 2);
    }
    expect(m, MODEL + "time_conv_norm.scale", h.band_dim);
    expect(m, MODEL + "time_deconv_norm.scale", h.band_dim);
    mlp(MODEL + "time_conv", h.band_dim * h.group, h.hidden * h.group, h.hidden);
    mlp(MODEL + "time_deconv", h.hidden, h.hidden * h.group, h.band_dim * h.group);
    expect(m, MODEL + "encoder.register_tokens", h.hidden, h.registers, h.registers);
    expect(m, MODEL + "encoder.final_norm.scale", h.hidden);
    for (int i = 0; i < h.layers; ++i)
        for (int axis = 0; axis < 2; ++axis) {
            const std::string p =
                MODEL + "encoder.layers." + std::to_string(i) + "." + std::to_string(axis) + ".layers.0";
            expect(m, p + ".input_layernorm.scale", h.hidden);
            expect(m, p + ".post_attention_layernorm.scale", h.hidden);
            attention(p + ".self_attn");
            mlp(p + ".mlp", h.hidden, h.intermediate, h.hidden);
        }
    for (const char* pool : {"freq_pooling", "time_pooling"}) {
        const std::string p = std::string("freq_domain_model.") + pool;
        expect(m, p + ".probe", h.hidden, std::strcmp(pool, "freq_pooling") == 0 ? 1 : h.stems);
        expect(m, p + ".mlp_layernorm.scale", h.hidden);
        attention(p + ".attn");
        mlp(p + ".mlp", h.hidden, h.intermediate, h.hidden);
    }
    expect(m, "aux.stft_window", h.fft);
    expect(m, "aux.stft_out_window", h.fft_out);
    m.window_in = tensor_f32(weight(m, "aux.stft_window"));
    m.window_out = tensor_f32(weight(m, "aux.stft_out_window"));
    m.fft_in = std::make_unique<mini_bs_fft::Plan>(h.fft);
    m.fft_out = std::make_unique<mini_bs_fft::Plan>(h.fft_out);
    for (const auto& entry : m.weights.tensors) {
        m.quant_q4 |= entry.second->type == GGML_TYPE_Q4_K;
        m.quant_q8 |= entry.second->type == GGML_TYPE_Q8_0;
    }
}

void init_graph(mini_bs_roformer_context& m, Graph& g, bool capture, size_t capacity = GRAPH_SIZE) {
    g.clear();
    ggml_init_params ip{ggml_tensor_overhead() * capacity + ggml_graph_overhead_custom(capacity, false), nullptr, true};
    g.ctx = ggml_init(ip);
    if (!g.ctx)
        throw std::runtime_error("cannot allocate graph metadata");
    ggml_backend_t backends[] = {m.blas ? m.blas : m.backend, m.cpu};
    g.sched = ggml_backend_sched_new(backends, nullptr, m.backend == m.cpu && !m.blas ? 1 : 2, capacity, false, true);
    if (!g.sched)
        throw std::runtime_error("cannot create graph scheduler");
    g.graph = ggml_new_graph_custom(g.ctx, capacity, false);
    // Publish the cache key only after allocation and input setup succeed.
    // A failed build must be retryable, never mistaken for a complete graph.
    g.frames = 0;
    g.capture = capture;
}

ggml_tensor* linear(mini_bs_roformer_context& m, ggml_context* c, ggml_tensor* x, const std::string& n) {
    // A single 2-D GEMM over all tokens enables efficient CPU/GPU kernels
    // instead of one matmul per band/head/frame.
    ggml_tensor* w = weight(m, n + ".weight");
    ggml_tensor* original_x = x;
    const int64_t d1 = x->ne[1], d2 = x->ne[2], d3 = x->ne[3];
    if (!ggml_is_contiguous(x))
        x = ggml_cont(c, x);
    x = ggml_reshape_2d(c, x, x->ne[0], d1 * d2 * d3);
    ggml_tensor* y = nullptr;
    if (m.cpu_bf16) {
        // BF16 operands reach ggml's native CPU GEMM (including AVX512BF16)
        // while its output and accumulator remain F32. Casting both operands
        // also prevents BLAS from widening weights into an F32 GEMM instead.
        // Norms, residuals, nonlinearities and attention remain F32.
        auto& g = c == m.encoder.ctx ? m.encoder : m.decoder;
        // Q/K/V projections and each SwiGLU gate/up pair share an activation.
        // Reuse its conversion node so a forward rounds and reads it only once.
        auto input = g.bf16_inputs.find(original_x);
        ggml_tensor* bx = input == g.bf16_inputs.end() ? ggml_cast(c, x, GGML_TYPE_BF16) : input->second;
        g.bf16_inputs.emplace(original_x, bx);
        auto operand = g.bf16_weights.find(w);
        if (operand == g.bf16_weights.end()) {
            // CPU quantized DUP targets F32. Convert through that supported
            // representation, then share the BF16/packed nodes among uses of
            // the same immutable weight within this graph. The scheduler keeps
            // their scratch alive through the last stem that reads it.
            ggml_tensor* unpacked = ggml_is_quantized(w->type) ? ggml_cast(c, w, GGML_TYPE_F32) : w;
            Graph::WeightOperand value;
            value.values = ggml_cast(c, unpacked, GGML_TYPE_BF16);
            if (m.cpu_packed_bf16)
                value.packed = mini_bs_bf16::pack(c, value.values);
            operand = g.bf16_weights.emplace(w, value).first;
        }
        if (operand->second.packed) {
            // The private CPU kernel is a scheduled ggml node. Its packed
            // weight layout reuses each pair across multiple audio positions;
            // unsupported shapes retain ggml's portable BF16 matrix multiply.
            y = mini_bs_bf16::graph(c, operand->second.packed, bx, w->ne[1]);
        }
        if (!y)
            y = ggml_mul_mat(c, operand->second.values, bx);
    } else {
        y = ggml_mul_mat(c, w, x);
    }
    if (!m.cpu_bf16 && m.backend == m.cpu && w->type != GGML_TYPE_F32 &&
        (!m.blas || !ggml_backend_supports_op(m.blas, y))) {
        // Preserve F32 activation precision even for small pooling projections
        // below BLAS's dispatch threshold and for builds without BLAS. The
        // conversion is a native graph op with reusable scratch, not a full
        // resident F32 copy of the model. F32 GEMM also reaches llamafile on
        // x86, where this fork's mixed F16/F32 dot path cannot use that kernel.
        y = ggml_mul_mat(c, ggml_cast(c, w, GGML_TYPE_F32), x);
    }
    ggml_prec_set_acc(y, GGML_PREC_F32);
    return ggml_reshape_4d(c, y, w->ne[1], d1, d2, d3);
}

ggml_tensor* norm(mini_bs_roformer_context& m, ggml_context* c, ggml_tensor* x, const std::string& n) {
    // Original DeltaRMSNorm stores gamma - 1, not gamma.
    ggml_tensor* gamma = ggml_scale_bias(c, weight(m, n + ".scale"), 1.0f, 1.0f);
    return ggml_mul(c, ggml_rms_norm(c, x, m.hp.eps), gamma);
}

ggml_tensor* mlp(mini_bs_roformer_context& m, ggml_context* c, ggml_tensor* x, const std::string& p) {
    ggml_tensor* gate = linear(m, c, x, p + ".gate_proj");
    ggml_tensor* up = linear(m, c, x, p + ".up_proj");
    return linear(m, c, ggml_swiglu_split(c, gate, up), p + ".down_proj");
}

ggml_tensor* attention(mini_bs_roformer_context& m, ggml_context* c, ggml_tensor* q_in, ggml_tensor* kv_in,
                       const std::string& p, ggml_tensor* positions) {
    const auto& h = m.hp;
    const int64_t nq = q_in->ne[1], nk = kv_in->ne[1], batch = kv_in->ne[2];
    ggml_tensor* q = linear(m, c, q_in, p + ".q_proj");
    ggml_tensor* k = linear(m, c, kv_in, p + ".k_proj");
    ggml_tensor* v = linear(m, c, kv_in, p + ".v_proj");
    q = ggml_reshape_4d(c, q, h.head_dim, h.heads, nq, q_in->ne[2]);
    k = ggml_reshape_4d(c, k, h.head_dim, h.kv_heads, nk, batch);
    v = ggml_reshape_4d(c, v, h.head_dim, h.kv_heads, nk, batch);
    if (positions) {
        q = ggml_rope_ext(c, q, positions, nullptr, h.head_dim, GGML_ROPE_TYPE_NEOX, 4096, h.rope, 1.0f, 0.0f, 1.0f,
                          32.0f, 1.0f);
        k = ggml_rope_ext(c, k, positions, nullptr, h.head_dim, GGML_ROPE_TYPE_NEOX, 4096, h.rope, 1.0f, 0.0f, 1.0f,
                          32.0f, 1.0f);
    }
    q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));
    k = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
    v = ggml_cont(c, ggml_permute(c, v, 0, 2, 1, 3));
    if (q_in->ne[2] != batch)
        q = ggml_repeat_4d(c, q, h.head_dim, nq, h.heads, batch);
    ggml_tensor* a = nullptr;
    const float scale = 1.0f / std::sqrt((float)h.head_dim);
    // The short 84-position frequency grid is faster as exact-size BLAS
    // GEMMs plus ggml softmax on the packed CPU path. Keep temporal and
    // pooling attention fused, and honor an explicit FLASH override.
    const bool dense_frequency =
        m.dense_frequency_attention && positions && nq == (int64_t)h.bands.size() + h.registers && nk == nq;
    const bool allow_flash = positions != nullptr || core_cpu_backend::is_cpu(m.backend);
    if (m.flash && !dense_frequency && allow_flash) {
        // Native GQA broadcasting repeats each KV head contiguously. Explicit
        // head expansion would double KV storage and defeat fused kernels.
        ggml_tensor* fk = core_cpu_backend::is_cpu(m.backend) ? k : ggml_cast(c, k, GGML_TYPE_F16);
        ggml_tensor* fv = core_cpu_backend::is_cpu(m.backend) ? v : ggml_cast(c, v, GGML_TYPE_F16);
        a = ggml_flash_attn_ext(c, q, fk, fv, nullptr, scale, 0.0f, 0.0f);
        if (core_cpu_backend::is_cpu(m.backend)) {
            ggml_prec_set_acc(a, GGML_PREC_F32);
        }
        // An unsupported fused kernel must not quietly send every attention
        // block and its activations across the GPU/CPU boundary.
        if (!ggml_backend_supports_op(m.backend, a))
            a = nullptr;
    }
    if (!a) {
        // Portable fallback stays in the graph, useful for GPU backends without
        // fused attention and for independently checking the fused path.
        ggml_tensor* scores = ggml_mul_mat(c, k, q);
        ggml_prec_set_acc(scores, GGML_PREC_F32);
        scores = ggml_soft_max_ext(c, scores, nullptr, scale, 0.0f);
        a = ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, v)), scores);
        ggml_prec_set_acc(a, GGML_PREC_F32);
        a = ggml_cont(c, ggml_permute(c, a, 0, 2, 1, 3));
    }
    a = ggml_reshape_3d(c, a, h.hidden, nq, batch);
    return linear(m, c, a, p + ".o_proj");
}

ggml_tensor* concat_tree(ggml_context* c, std::vector<ggml_tensor*> tensors, int axis) {
    while (tensors.size() > 1) {
        std::vector<ggml_tensor*> next;
        for (size_t i = 0; i < tensors.size(); i += 2)
            next.push_back(i + 1 == tensors.size() ? tensors[i] : ggml_concat(c, tensors[i], tensors[i + 1], axis));
        tensors = std::move(next);
    }
    return tensors.at(0);
}

void capture(Graph& g, const std::string& n, ggml_tensor* x) {
    if (!g.capture)
        return;
    x = ggml_cont(g.ctx, x);
    ggml_set_name(x, n.c_str());
    ggml_set_output(x);
    ggml_build_forward_expand(g.graph, x);
    g.captures[n] = x;
}

ggml_tensor* pooling(mini_bs_roformer_context& m, ggml_context* c, ggml_tensor* x, const std::string& p) {
    ggml_tensor* probe = weight(m, p + ".probe");
    ggml_tensor* out = attention(m, c, probe, x, p + ".attn", nullptr);
    return ggml_add(c, out, mlp(m, c, norm(m, c, out, p + ".mlp_layernorm"), p + ".mlp"));
}

void build_encoder(mini_bs_roformer_context& m, int frames, bool want_capture) {
    auto& g = m.encoder;
    if (g.frames == frames && g.capture == want_capture)
        return;
    init_graph(m, g, want_capture);
    auto* c = g.ctx;
    const auto& h = m.hp;
    const int bands = (int)h.bands.size(), tc = (frames + h.group - 1) / h.group;
    g.input = ggml_new_tensor_2d(c, GGML_TYPE_F32, (h.fft / 2 + 1) * 4, frames);
    ggml_set_input(g.input);
    g.positions_t = ggml_new_tensor_1d(c, GGML_TYPE_I32, tc + h.registers);
    g.positions_f = ggml_new_tensor_1d(c, GGML_TYPE_I32, bands + h.registers);
    ggml_set_input(g.positions_t);
    ggml_set_input(g.positions_f);
    // These constant inputs are initialized once per cached graph. INPUT
    // permits storage reuse after the last RoPE; pin them across executions.
    ggml_set_output(g.positions_t);
    ggml_set_output(g.positions_f);
    std::vector<ggml_tensor*> band_features;
    int offset = 0;
    for (int b = 0; b < bands; ++b) {
        const int width = h.bands[b] * 4;
        const std::string p = MODEL + "band_split.to_features." + std::to_string(b);
        ggml_tensor* x = ggml_view_2d(c, g.input, width, frames, g.input->nb[1], offset * sizeof(float));
        x = norm(m, c, ggml_cont(c, x), p + ".0");
        x = linear(m, c, x, p + ".1");
        band_features.push_back(ggml_reshape_3d(c, x, h.band_dim, 1, frames));
        offset += width;
    }
    ggml_tensor* x = concat_tree(c, std::move(band_features), 1); // [D,F,T]
    capture(g, "band_split", x);
    x = ggml_pad(c, x, 0, 0, tc * h.group - frames, 0);
    x = norm(m, c, x, MODEL + "time_conv_norm");
    x = ggml_reshape_4d(c, x, h.band_dim, bands, h.group, tc);
    // einops: b (t tc) n d -> b t n (d tc), NOT (tc d).
    x = ggml_cont(c, ggml_permute(c, x, 1, 2, 0, 3));
    x = ggml_reshape_3d(c, x, h.band_dim * h.group, bands, tc);
    x = mlp(m, c, x, MODEL + "time_conv");
    capture(g, "time_compressed", x);
    x = ggml_pad(c, x, 0, h.registers, h.registers, 0);
    x = ggml_set(c, x, weight(m, MODEL + "encoder.register_tokens"), x->nb[1], x->nb[2], x->nb[3],
                 (size_t)bands * x->nb[1] + (size_t)tc * x->nb[2]);
    for (int i = 0; i < h.layers; ++i)
        for (int axis = 0; axis < 2; ++axis) {
            if (axis == 0)
                x = ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3)); // time is the sequence
            const std::string p =
                MODEL + "encoder.layers." + std::to_string(i) + "." + std::to_string(axis) + ".layers.0";
            ggml_tensor* n = norm(m, c, x, p + ".input_layernorm");
            x = ggml_add(c, x, attention(m, c, n, n, p + ".self_attn", axis == 0 ? g.positions_t : g.positions_f));
            x = ggml_add(c, x, mlp(m, c, norm(m, c, x, p + ".post_attention_layernorm"), p + ".mlp"));
            if (axis == 0)
                x = ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3));
            capture(g, "layer" + std::to_string(i) + (axis == 0 ? "_time" : "_freq"), x);
        }
    x = ggml_cont(c, ggml_view_3d(c, x, h.hidden, bands, tc, x->nb[1], x->nb[2], 0));
    g.hidden = norm(m, c, x, MODEL + "encoder.final_norm");
    ggml_set_output(g.hidden);
    capture(g, "encoder_output", g.hidden);
    x = pooling(m, c, g.hidden, "freq_domain_model.freq_pooling"); // [H,1,Tc]
    x = ggml_reshape_2d(c, x, h.hidden, tc);
    capture(g, "freq_pooled", x);
    g.embeddings = pooling(m, c, x, "freq_domain_model.time_pooling");
    ggml_set_output(g.embeddings);
    capture(g, "stem_embeddings", g.embeddings);
    ggml_build_forward_expand(g.graph, g.hidden);
    ggml_build_forward_expand(g.graph, g.embeddings);
    for (int s = 0; s < h.stems; ++s) {
        g.stem_views[s] = ggml_view_1d(c, g.embeddings, h.hidden, (size_t)s * h.hidden * sizeof(float));
        ggml_build_forward_expand(g.graph, g.stem_views[s]);
    }
    if (!ggml_backend_sched_alloc_graph(g.sched, g.graph))
        throw std::runtime_error("encoder graph allocation failed");
    std::vector<int32_t> pt(tc + h.registers, 0), pf(bands + h.registers, 0);
    std::iota(pt.begin(), pt.begin() + tc, 0);
    std::iota(pf.begin(), pf.begin() + bands, 0);
    // Register positions have cos=1/sin=0 in the blueprint, i.e. RoPE pos0.
    ggml_backend_tensor_set(g.positions_t, pt.data(), 0, pt.size() * sizeof(int32_t));
    ggml_backend_tensor_set(g.positions_f, pf.data(), 0, pf.size() * sizeof(int32_t));
    g.frames = frames;
}

ggml_tensor* decoder_mask(mini_bs_roformer_context& m, ggml_context* c, ggml_tensor* x, int frames) {
    const auto& h = m.hp;
    const int bands = (int)h.bands.size(), tc = (frames + h.group - 1) / h.group;
    x = ggml_reshape_4d(c, x, h.group, h.band_dim, bands, tc);
    x = ggml_cont(c, ggml_permute(c, x, 2, 0, 1, 3)); // [D,F,tc,Tc]
    x = ggml_reshape_3d(c, x, h.band_dim, bands, tc * h.group);
    x = norm(m, c, x, MODEL + "time_deconv_norm");
    std::vector<ggml_tensor*> masks;
    for (int b = 0; b < bands; ++b) {
        const std::string s = std::to_string(b);
        ggml_tensor* feature = ggml_cont(c, ggml_view_2d(c, x, h.band_dim, frames, x->nb[2], b * x->nb[1]));
        ggml_tensor* mask = linear(m, c, feature, MODEL + "mask_estimator.to_freq_mlps." + s);
        ggml_tensor* gate = linear(m, c, feature, MODEL + "mask_estimator.to_gate_mlps." + s);
        gate = ggml_sigmoid(c, gate);
        gate = ggml_reshape_3d(c, gate, 1, h.bands_out[b] * 2, frames);
        gate = ggml_repeat_4d(c, gate, 2, h.bands_out[b] * 2, frames, 1);
        gate = ggml_reshape_2d(c, gate, h.bands_out[b] * 4, frames);
        masks.push_back(ggml_mul(c, mask, gate));
    }
    return concat_tree(c, std::move(masks), 0);
}

void build_decoder(mini_bs_roformer_context& m, int frames) {
    auto& g = m.decoder;
    if (g.frames == frames)
        return;
    init_graph(m, g, false, m.shared_decoder ? GRAPH_SIZE * 2 : GRAPH_SIZE);
    auto* c = g.ctx;
    const auto& h = m.hp;
    const int bands = (int)h.bands.size(), tc = (frames + h.group - 1) / h.group;
    g.input = ggml_new_tensor_3d(c, GGML_TYPE_F32, h.hidden, bands, tc);
    ggml_set_input(g.input);
    // The same hidden state is consumed by all four decoder invocations in
    // the original path. OUTPUT protects it from scratch reuse between stems.
    // Keep this input protection on both paths and refresh it for each chunk.
    ggml_set_output(g.input);
    const std::string p = MODEL + "time_deconv";
    if (m.shared_decoder) {
        g.stem = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.hidden, h.stems);
        ggml_set_input(g.stem);
        // There is no normalization between H + stem and the first two
        // affine projections. Share W*H across the four stems, and project
        // their embeddings as a tiny four-column matrix. BF16 rounds the two
        // operands separately here, so this path must pass the same
        // numerical gates as the original sum-before-projection path.
        auto* gate_base = linear(m, c, g.input, p + ".gate_proj");
        auto* up_base = linear(m, c, g.input, p + ".up_proj");
        auto* stem_gate = linear(m, c, g.stem, p + ".gate_proj");
        auto* stem_up = linear(m, c, g.stem, p + ".up_proj");
        const int mid = h.hidden * h.group;
        for (int s = 0; s < h.stems; ++s) {
            auto* gate_bias = ggml_view_1d(c, stem_gate, mid, (size_t)s * stem_gate->nb[1]);
            auto* up_bias = ggml_view_1d(c, stem_up, mid, (size_t)s * stem_up->nb[1]);
            auto* gate = ggml_add(c, gate_base, gate_bias);
            auto* up = ggml_add(c, up_base, up_bias);
            auto* x = linear(m, c, ggml_swiglu_split(c, gate, up), p + ".down_proj");
            g.stem_masks[s] = decoder_mask(m, c, x, frames);
            ggml_set_output(g.stem_masks[s]);
            // Expand each complete branch in stem order. The scheduler can
            // reuse its temporary activations after the mask is produced;
            // only shared inputs/projections and the four masks stay live.
            ggml_build_forward_expand(g.graph, g.stem_masks[s]);
        }
    } else {
        g.stem = ggml_new_tensor_1d(c, GGML_TYPE_F32, h.hidden);
        ggml_set_input(g.stem);
        auto* x = mlp(m, c, ggml_add(c, g.input, g.stem), p);
        g.mask = decoder_mask(m, c, x, frames);
        ggml_set_output(g.mask);
        ggml_build_forward_expand(g.graph, g.mask);
    }
    if (!ggml_backend_sched_alloc_graph(g.sched, g.graph))
        throw std::runtime_error("decoder graph allocation failed");
    g.frames = frames;
}

int reflected(int i, int n) {
    // The production chunk is always longer than both windows. This repeated
    // reflection also gives defined behavior to explicit short diagnostic runs.
    if (n <= 1)
        return 0;
    const int period = 2 * (n - 1);
    i %= period;
    if (i < 0)
        i += period;
    return i < n ? i : period - i;
}

// F32 center=True, reflect padding, periodic Hann, normalized=False. Packing
// is exactly [time, frequency, channel, real/imag], the trained band order.
std::vector<float> stft(mini_bs_roformer_context& m, const float* pcm, int n, int fft,
                        const std::vector<float>& window) {
    const int frames = n / m.hp.hop + 1, bins = fft / 2 + 1, channels = m.hp.channels;
    const auto& plan = fft == m.hp.fft ? *m.fft_in : *m.fft_out;
    std::vector<float> out((size_t)frames * bins * channels * 2);
    core_parallel::for_each_chunk_work(
        frames * channels, m.params.n_threads, (uint64_t)fft * 12, 32768, [&](int begin, int end) {
            std::vector<float> re(fft), im(fft);
            for (int index = begin; index < end; ++index) {
                const int t = index / channels, ch = index % channels;
                for (int k = 0; k < fft; ++k) {
                    re[k] = pcm[(size_t)reflected(t * m.hp.hop + k - fft / 2, n) * channels + ch] * window[k];
                    im[k] = 0;
                }
                plan.transform(re.data(), im.data());
                for (int f = 0; f < bins; ++f) {
                    const size_t ix = ((size_t)t * bins * channels + f * channels + ch) * 2;
                    out[ix] = re[f];
                    out[ix + 1] = im[f];
                }
            }
        });
    return out;
}

std::vector<float> reconstruct(mini_bs_roformer_context& m, const std::vector<float>& spec,
                               const std::vector<float>& mask, int n) {
    const int fft = m.hp.fft_out, hop = m.hp.hop, frames = n / hop + 1, bins = fft / 2 + 1;
    const int channels = m.hp.channels, raw_n = (frames - 1) * hop + fft;
    std::vector<float> out((size_t)n * channels);
    // Independent channel overlap-add preserves a fixed reduction order while
    // keeping FFT scratch O(n_fft), not O(frames*n_fft).
    core_parallel::for_each_chunk(channels, m.params.n_threads, [&](int a, int b) {
        std::vector<float> re(fft), im(fft), wave(raw_n), envelope(raw_n);
        for (int ch = a; ch < b; ++ch) {
            std::fill(wave.begin(), wave.end(), 0);
            std::fill(envelope.begin(), envelope.end(), 0);
            for (int t = 0; t < frames; ++t) {
                for (int f = 0; f < bins; ++f) {
                    const size_t ix = ((size_t)t * bins * channels + f * channels + ch) * 2;
                    const float sr = spec[ix], si = spec[ix + 1], mr = mask[ix], mi = mask[ix + 1];
                    re[f] = sr * mr - si * mi;
                    im[f] = -(sr * mi + si * mr); // conjugate for inverse FFT
                }
                im[0] = im[fft / 2] = 0;
                for (int f = 1; f < fft / 2; ++f) {
                    re[fft - f] = re[f];
                    im[fft - f] = -im[f];
                }
                m.fft_out->transform(re.data(), im.data());
                for (int k = 0; k < fft; ++k) {
                    const float w = m.window_out[k];
                    wave[t * hop + k] += (re[k] / fft) * w;
                    envelope[t * hop + k] += w * w;
                }
            }
            for (int i = 0; i < n; ++i) {
                const int ix = i + fft / 2;
                out[(size_t)i * channels + ch] = envelope[ix] > 1e-11f ? wave[ix] / envelope[ix] : 0.0f;
            }
        }
    });
    return out;
}

std::vector<float> planar(const std::vector<float>& interleaved, int channels) {
    const size_t n = interleaved.size() / channels;
    std::vector<float> out(interleaved.size());
    for (int c = 0; c < channels; ++c)
        for (size_t t = 0; t < n; ++t)
            out[(size_t)c * n + t] = interleaved[t * channels + c];
    return out;
}

std::array<std::vector<float>, 4> forward(mini_bs_roformer_context& m, const float* input, int n,
                                          Captures* captures = nullptr) {
    StageTimer total("forward");
    const int frames = n / m.hp.hop + 1;
    std::vector<float> spectrum, output_spectrum;
    {
        StageTimer timer("stft");
        spectrum = stft(m, input, n, m.hp.fft, m.window_in);
        output_spectrum = stft(m, input, n, m.hp.fft_out, m.window_out);
    }
    if (captures) {
        (*captures)["stft_input"] = spectrum;
        const int bins = m.hp.fft_out / 2 + 1, channels = m.hp.channels;
        auto& a = (*captures)["stft_output"];
        a.resize(output_spectrum.size());
        for (int ch = 0; ch < channels; ++ch)
            for (int f = 0; f < bins; ++f)
                for (int t = 0; t < frames; ++t)
                    for (int ri = 0; ri < 2; ++ri)
                        a[(((size_t)ch * bins + f) * frames + t) * 2 + ri] =
                            output_spectrum[(((size_t)t * bins + f) * channels + ch) * 2 + ri];
    }
    build_encoder(m, frames, captures != nullptr);
    build_decoder(m, frames);
    {
        StageTimer timer("encoder_pooling");
        ggml_backend_tensor_set(m.encoder.input, spectrum.data(), 0, spectrum.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(m.encoder.sched, m.encoder.graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("encoder execution failed");
        if (m.backend)
            ggml_backend_synchronize(m.backend);
    }
    if (captures)
        for (const auto& item : m.encoder.captures)
            (*captures)[item.first] = tensor_f32(item.second);
    // Keep hidden states on the selected backend. The accelerated CPU graph
    // computes all four masks in one call; the original graph is reused once
    // per stem. Both preserve the same mask order and reconstruction path.
    if (m.shared_decoder) {
        StageTimer timer("shared_stem_decoder");
        ggml_backend_tensor_copy(m.encoder.hidden, m.decoder.input);
        ggml_backend_tensor_copy(m.encoder.embeddings, m.decoder.stem);
        if (ggml_backend_sched_graph_compute(m.decoder.sched, m.decoder.graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("shared decoder execution failed");
        if (m.backend)
            ggml_backend_synchronize(m.backend);
    }
    std::array<std::vector<float>, 4> result;
    for (int s = 0; s < m.hp.stems; ++s) {
        std::vector<float> mask;
        if (m.shared_decoder) {
            mask = tensor_f32(m.decoder.stem_masks[s]);
        } else {
            StageTimer timer("stem_decoder");
            ggml_backend_tensor_copy(m.encoder.hidden, m.decoder.input);
            ggml_backend_tensor_copy(m.encoder.stem_views[s], m.decoder.stem);
            if (ggml_backend_sched_graph_compute(m.decoder.sched, m.decoder.graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("decoder execution failed");
            if (m.backend)
                ggml_backend_synchronize(m.backend);
            mask = tensor_f32(m.decoder.mask);
        }
        if (captures) {
            auto& a = (*captures)["mask_raw"];
            a.insert(a.end(), mask.begin(), mask.end());
        }
        {
            StageTimer timer("istft");
            result[s] = reconstruct(m, output_spectrum, mask, n);
        }
        if (captures)
            (*captures)[std::string("output_") + STEMS[s]] = planar(result[s], m.hp.channels);
    }
    return result;
}
} // namespace

mini_bs_roformer_params mini_bs_roformer_default_params(void) {
    return {0, true, 0, 0, false};
}

mini_bs_roformer_context* mini_bs_roformer_init_from_file(const char* path, mini_bs_roformer_params params) {
    if (!path || !*path)
        return nullptr;
    std::unique_ptr<mini_bs_roformer_context, decltype(&mini_bs_roformer_free)> m(new mini_bs_roformer_context,
                                                                                  mini_bs_roformer_free);
    try {
        StageTimer timer("load");
        gguf_context* meta = core_gguf::open_metadata(path);
        if (!meta)
            return nullptr;
        const bool valid = read_hparams(meta, m->hp);
        core_gguf::free_metadata(meta);
        if (!valid)
            throw std::runtime_error("unsupported architecture or incomplete Mini-BS-RoFormer V2.5 metadata");
        if (params.gpu_device < 0 || params.segment_seconds > 60)
            throw std::runtime_error("invalid runtime parameters");
        params.n_threads = params.n_threads > 0 ? params.n_threads : std::min(8, core_parallel::resolve_threads(8, 0));
        m->params = params;
        ggml_backend_load_all();
        m->cpu = core_cpu_backend::init();
        if (!m->cpu)
            throw std::runtime_error("CPU backend unavailable");
        core_cpu_backend::set_n_threads(m->cpu, params.n_threads);
        if (params.use_gpu) {
            // Disable Metal graph optimize reordering for Mini-BS-RoFormer:
            // the model's multi-dimensional view/permute graphs violate
            // Metal's speculative concurrency reorder assumptions.
#ifdef __APPLE__ // Metal only; MSVC has no setenv
            setenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE", "1", 0);
#endif
            crispasr_set_gpu_device_pref(params.gpu_device);
            m->backend = crispasr_init_gpu_backend();
            if (m->backend && core_cpu_backend::is_cpu(m->backend)) {
                ggml_backend_free(m->backend);
                m->backend = nullptr;
            }
            // #568: on Metal the separated stems lose most of the signal
            // (vocals RMS 0 vs 0.154 on CPU). Until the Metal path is fixed and
            // qualified, run on CPU; CRISPASR_MINI_BS_ROFORMER_METAL=1 opts back
            // in for debugging.
            const char* bn = m->backend ? ggml_backend_name(m->backend) : nullptr;
            const bool metal = bn && (std::strncmp(bn, "MTL", 3) == 0 || std::strncmp(bn, "Metal", 5) == 0);
            if (metal && !crispasr_env::get("CRISPASR_MINI_BS_ROFORMER_METAL")) {
                fprintf(stderr, "mini_bs_roformer: Metal output is not yet correct (#568); using CPU\n");
                ggml_backend_free(m->backend);
                m->backend = nullptr;
            }
        }
        if (!m->backend)
            m->backend = m->cpu;
        // Select the native packed path only where its compiler and runtime
        // ISA guards succeed. Explicit controls retain the F32 and portable
        // BF16 paths for independent comparisons on the same model artifact.
        const bool packed_cpu = m->backend == m->cpu && mini_bs_bf16::supported(m->cpu);
        m->cpu_bf16 = m->backend == m->cpu && env_flag("CRISPASR_MINI_BS_CPU_BF16", packed_cpu);
        m->cpu_packed_bf16 = m->cpu_bf16 && packed_cpu && env_flag("CRISPASR_MINI_BS_PACKED_BF16", true);
        // Compute shared W*H projection across all 4 stems in a single pass
        // on both GPU and CPU backends, saving 75% of gate/up GEMMs in time_deconv.
        m->shared_decoder = env_flag("CRISPASR_MINI_BS_SHARED_DECODER", true);
        // Use the native ggml BLAS accelerator for large CPU GEMMs when it is
        // registered. Unlike the quantized dot kernels it preserves F32
        // activations. Registry lookup also works with GGML_BACKEND_DL builds.
        if (m->backend == m->cpu && env_flag("CRISPASR_MINI_BS_BLAS", true)) {
            m->blas = ggml_backend_init_by_name("BLAS", nullptr);
            if (m->blas) {
                auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(m->blas));
                auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
                if (set_threads)
                    set_threads(m->blas, params.n_threads);
            }
        }
        bool loaded;
        if (m->backend == m->cpu)
            // Repacked buffers have a private byte layout. Both BLAS and the
            // explicit graph casts require standard, readable weight storage.
            loaded = core_gguf::load_weights(path, m->cpu, "mini_bs_roformer", m->weights);
        else
            loaded = core_gguf::load_weights_fit(path, m->backend, m->cpu, "mini_bs_roformer", m->weights);
        if (!loaded)
            throw std::runtime_error("weight load failed");
        validate_weights(*m);
        m->flash = env_flag("CRISPASR_MINI_BS_FLASH", true);
        m->dense_frequency_attention = m->cpu_packed_bf16 && m->blas && !std::getenv("CRISPASR_MINI_BS_FLASH");
        std::fprintf(stderr,
                     "mini_bs_roformer: backend=%s%s threads=%d precision=%s compute=%s attention=%s decoder=%s\n",
                     ggml_backend_name(m->backend), m->blas ? "+BLAS" : "", params.n_threads,
                     m->quant_q4   ? "Q4_K mixed"
                     : m->quant_q8 ? "Q8_0 mixed"
                                   : "F16/F32",
                     mini_bs_roformer_compute_precision(m.get()),
                     !m->flash                      ? "graph-softmax-GQA"
                     : m->dense_frequency_attention ? "flash-time/dense-frequency-GQA"
                                                    : "flash-GQA",
                     m->shared_decoder ? "shared-affine" : "per-stem");
        return m.release();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mini_bs_roformer: %s\n", e.what());
        return nullptr;
    }
}

void mini_bs_roformer_free(mini_bs_roformer_context* ctx) {
    if (!ctx)
        return;
    ctx->encoder.clear();
    ctx->decoder.clear();
    core_gguf::free_weights(ctx->weights);
    if (ctx->blas)
        ggml_backend_free(ctx->blas);
    if (ctx->backend && ctx->backend != ctx->cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->cpu)
        ggml_backend_free(ctx->cpu);
    delete ctx;
}

mini_bs_roformer_result* mini_bs_roformer_separate(mini_bs_roformer_context* ctx, const float* pcm, int n_samples,
                                                   int in_channels) {
    if (!ctx || !pcm || n_samples <= 0 || (in_channels != 1 && in_channels != 2) ||
        n_samples > std::numeric_limits<int>::max() / 4)
        return nullptr;
    if (ctx->params.no_segment && n_samples <= ctx->hp.fft / 2)
        return nullptr; // match the original raw-forward reflection constraint
    for (size_t i = 0; i < (size_t)n_samples * in_channels; ++i)
        if (!std::isfinite(pcm[i]))
            return nullptr;
    try {
        StageTimer timer("separate");
        const auto& h = ctx->hp;
        std::vector<float> input((size_t)n_samples * h.channels);
        for (int i = 0; i < n_samples; ++i)
            for (int c = 0; c < h.channels; ++c)
                input[(size_t)i * h.channels + c] = pcm[(size_t)i * in_channels + (in_channels == 1 ? 0 : c)];
        std::array<std::vector<float>, 4> sources;
        if (ctx->params.no_segment) {
            sources = forward(*ctx, input.data(), n_samples);
        } else {
            const int chunk = ctx->params.segment_seconds > 0 ? ctx->params.segment_seconds * h.rate : h.chunk;
            const int hop = chunk / 2, fade = chunk / 10;
            if (chunk <= h.fft / 2 || hop <= 0 || fade < 2)
                return nullptr;
            const int chunks = (int)(((int64_t)std::max(n_samples - chunk, 0) + hop - 1) / hop + 1);
            std::vector<float> window(chunk, 1), counts(n_samples, 0), block((size_t)chunk * h.channels);
            for (int t = 0; t < fade; ++t) {
                window[t] = (float)t / (fade - 1);
                window[chunk - fade + t] = 1.0f - (float)t / (fade - 1);
            }
            for (auto& s : sources)
                s.assign((size_t)n_samples * h.channels, 0);
            for (int i = 0; i < chunks; ++i) {
                const int start = i * hop, count = std::min(chunk, n_samples - start);
                std::fill(block.begin(), block.end(), 0);
                std::copy_n(input.data() + (size_t)start * h.channels, (size_t)count * h.channels, block.data());
                const auto result = forward(*ctx, block.data(), chunk);
                for (int t = 0; t < count; ++t) {
                    const float w = window[t];
                    counts[start + t] += w;
                    for (int s = 0; s < h.stems; ++s)
                        for (int c = 0; c < h.channels; ++c)
                            sources[s][(size_t)(start + t) * h.channels + c] +=
                                result[s][(size_t)t * h.channels + c] * w;
                }
            }
            for (int t = 0; t < n_samples; ++t)
                for (int s = 0; s < h.stems; ++s)
                    for (int c = 0; c < h.channels; ++c)
                        sources[s][(size_t)t * h.channels + c] /= std::max(counts[t], 1e-8f);
        }
        std::unique_ptr<mini_bs_roformer_result, decltype(&mini_bs_roformer_result_free)> r(
            new mini_bs_roformer_result{}, mini_bs_roformer_result_free);
        r->n_sources = h.stems;
        r->n_channels = h.channels;
        r->n_samples = n_samples;
        r->sample_rate = h.rate;
        r->sources = new float* [h.stems] {};
        r->source_names = new const char* [h.stems] {};
        for (int s = 0; s < h.stems; ++s) {
            if (!std::all_of(sources[s].begin(), sources[s].end(), is_finite_sample))
                throw std::runtime_error("nonfinite separated waveform");
            r->sources[s] = new float[sources[s].size()];
            std::copy(sources[s].begin(), sources[s].end(), r->sources[s]);
            r->source_names[s] = STEMS[s];
        }
        return r.release();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mini_bs_roformer: separation failed: %s\n", e.what());
        return nullptr;
    }
}

void mini_bs_roformer_result_free(mini_bs_roformer_result* r) {
    if (!r)
        return;
    if (r->sources)
        for (int s = 0; s < r->n_sources; ++s)
            delete[] r->sources[s];
    delete[] r->sources;
    delete[] r->source_names;
    delete r;
}

int mini_bs_roformer_sample_rate(const mini_bs_roformer_context* ctx) {
    return ctx ? ctx->hp.rate : 0;
}
int mini_bs_roformer_n_sources(const mini_bs_roformer_context* ctx) {
    return ctx ? ctx->hp.stems : 0;
}
const char* mini_bs_roformer_source_name(const mini_bs_roformer_context* ctx, int idx) {
    return ctx && idx >= 0 && idx < ctx->hp.stems ? STEMS[idx] : nullptr;
}
const char* mini_bs_roformer_backend_name(const mini_bs_roformer_context* ctx) {
    return ctx && ctx->backend ? (ctx->blas ? "CPU+BLAS" : ggml_backend_name(ctx->backend)) : nullptr;
}
const char* mini_bs_roformer_compute_precision(const mini_bs_roformer_context* ctx) {
    if (!ctx)
        return nullptr;
    if (ctx->backend != ctx->cpu)
        return "GPU-ggml/F32-accum-ops";
    return ctx->cpu_packed_bf16 ? "packed-BF16-GEMM/F32-accum-ops" : ctx->cpu_bf16 ? "BF16-GEMM/F32-accum-ops" : "F32";
}

int mini_bs_roformer_diff(const char* model_gguf, const char* ref_gguf, const char* audio_wav, int verbosity) {
    (void)audio_wav; // The reference carries the exact, independently hashed waveform.
    auto params = mini_bs_roformer_default_params();
    params.use_gpu = env_flag("CRISPASR_MINI_BS_GPU", false);
    if (const char* threads = std::getenv("CRISPASR_MINI_BS_THREADS"))
        params.n_threads = std::max(1, std::atoi(threads));
    std::unique_ptr<mini_bs_roformer_context, decltype(&mini_bs_roformer_free)> m(
        mini_bs_roformer_init_from_file(model_gguf, params), mini_bs_roformer_free);
    if (!m)
        return 2;
    core_gguf::WeightLoad ref;
    if (!core_gguf::load_weights(ref_gguf, m->cpu, "mini_bs_reference", ref))
        return 2;
    int failures = 0;
    try {
        const auto it = ref.tensors.find("input_audio");
        if (it == ref.tensors.end() || it->second->ne[1] != m->hp.channels || it->second->ne[2] != 1 ||
            it->second->ne[3] != 1 || it->second->ne[0] <= m->hp.fft / 2 || it->second->ne[0] > m->hp.rate * 60)
            throw std::runtime_error("reference needs valid planar stereo input_audio, longer than half the input FFT");
        auto audio = tensor_f32(it->second);
        if (!std::all_of(audio.begin(), audio.end(), is_finite_sample))
            throw std::runtime_error("nonfinite reference waveform");
        const int n = (int)it->second->ne[0];
        std::vector<float> interleaved(audio.size());
        for (int t = 0; t < n; ++t)
            for (int ch = 0; ch < m->hp.channels; ++ch)
                interleaved[(size_t)t * m->hp.channels + ch] = audio[(size_t)ch * n + t];
        Captures actual;
        actual["input_audio"] = audio;
        const bool outputs_only = env_flag("CRISPASR_MINI_BS_OUTPUTS_ONLY");
        if (outputs_only) {
            auto result = forward(*m, interleaved.data(), n);
            for (int s = 0; s < m->hp.stems; ++s)
                actual[std::string("output_") + STEMS[s]] = planar(result[s], m->hp.channels);
        } else {
            forward(*m, interleaved.data(), n, &actual);
        }
        const int frames = n / m->hp.hop + 1, tc = (frames + 3) / 4;
        using Shape = std::array<int64_t, 4>;
        std::map<std::string, Shape> expected = {
            {"input_audio", {n, 2, 1, 1}},         {"stft_input", {8196, frames, 1, 1}},
            {"stft_output", {2, frames, 1025, 2}}, {"band_split", {256, 80, frames, 1}},
            {"time_compressed", {512, 80, tc, 1}}, {"encoder_output", {512, 80, tc, 1}},
            {"freq_pooled", {512, tc, 1, 1}},      {"stem_embeddings", {512, 4, 1, 1}},
            {"mask_raw", {4100, frames, 4, 1}},
        };
        for (int i = 0; i < 6; ++i)
            for (const char* axis : {"time", "freq"})
                expected["layer" + std::to_string(i) + "_" + axis] = {512, 84, tc + 4, 1};
        for (const char* stem : STEMS)
            expected[std::string("output_") + stem] = {n, 2, 1, 1};
        if (outputs_only) {
            for (auto stage = expected.begin(); stage != expected.end();) {
                if (stage->first != "input_audio" && stage->first.find("output_") != 0)
                    stage = expected.erase(stage);
                else
                    ++stage;
            }
        }
        const double threshold = m->quant_q4 ? 0.94 : m->quant_q8 ? 0.98 : 0.999;
        for (const auto& stage : expected) {
            const auto r = ref.tensors.find(stage.first);
            const auto a = actual.find(stage.first);
            bool shape_ok = r != ref.tensors.end() && a != actual.end();
            if (shape_ok)
                for (int d = 0; d < 4; ++d)
                    shape_ok &= r->second->ne[d] == stage.second[d];
            if (!shape_ok || (int64_t)a->second.size() != ggml_nelements(r->second)) {
                std::printf("[FAIL] %s missing stage or shape mismatch\n", stage.first.c_str());
                ++failures;
                continue;
            }
            const auto truth = tensor_f32(r->second);
            const auto& data = a->second;
            double dot = 0, aa = 0, bb = 0, max_abs = 0, cos_min = 1;
            size_t nonfinite = 0;
            const size_t width = (size_t)stage.second[0];
            for (size_t row = 0; row < data.size(); row += width) {
                double rd = 0, ra = 0, rb = 0;
                for (size_t j = row; j < row + width; ++j) {
                    const double x = data[j], y = truth[j];
                    if (!std::isfinite(x) || !std::isfinite(y)) {
                        ++nonfinite;
                        continue;
                    }
                    rd += x * y;
                    ra += x * x;
                    rb += y * y;
                    max_abs = std::max(max_abs, std::abs(x - y));
                }
                dot += rd;
                aa += ra;
                bb += rb;
                const double rc = ra > 0 && rb > 0 ? rd / std::sqrt(ra * rb) : ra == rb ? 1 : 0;
                cos_min = std::min(cos_min, rc);
            }
            const double cosine = aa > 0 && bb > 0 ? dot / std::sqrt(aa * bb) : aa == bb ? 1 : 0;
            const double ratio = bb > 0 ? std::sqrt(aa / bb) : aa == 0 ? 1 : std::numeric_limits<double>::infinity();
            // Cosine alone cannot detect collapsed or globally amplified stems.
            const bool pass = nonfinite == 0 && cosine > threshold && ratio >= 0.8 && ratio <= 1.2;
            if (!pass)
                ++failures;
            if (verbosity > 0 || !pass)
                std::printf("[%s] %s cos_global=%.9f cos_min=%.9f norm_ratio=%.9f max_abs=%.6g nonfinite=%zu n=%zu\n",
                            pass ? "PASS" : "FAIL", stage.first.c_str(), cosine, cos_min, ratio, max_abs, nonfinite,
                            data.size());
        }
        if (const char* output = std::getenv("CRISPASR_MINI_BS_CAPTURE_GGUF")) {
            if (*output) {
                gguf_context* file = gguf_init_empty();
                ggml_init_params ip{ggml_tensor_overhead() * (expected.size() + 1), nullptr, true};
                ggml_context* tensors = ggml_init(ip);
                if (!file || !tensors)
                    throw std::runtime_error("capture metadata allocation failed");
                gguf_set_val_str(file, "general.architecture", "mini-bs-roformer-capture");
                for (const auto& stage : expected) {
                    ggml_tensor* t = ggml_new_tensor(tensors, GGML_TYPE_F32, 4, stage.second.data());
                    ggml_set_name(t, stage.first.c_str());
                    t->data = actual.at(stage.first).data();
                    gguf_add_tensor(file, t);
                }
                const bool written = gguf_write_to_file(file, output, false);
                gguf_free(file);
                ggml_free(tensors);
                if (!written)
                    throw std::runtime_error("cannot write native capture GGUF");
            }
        }
        std::printf("mini_bs_roformer diff: %zu stages, %d failed, strict cosine > %.3f\n", expected.size(), failures,
                    threshold);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mini_bs_roformer diff: %s\n", e.what());
        failures = 1;
    }
    core_gguf::free_weights(ref);
    return failures ? 1 : 0;
}
