// csm_tts.cpp -- Sesame CSM-1B TTS runtime.
//
// Architecture:
//   Backbone:      Llama-3.2 1B (16L, 32H/8KVH, 2048d, SwiGLU, RMSNorm)
//                  Generates first-codebook Mimi tokens autoregressively.
//   Depth decoder: Llama-3.2 100M (4L, 8H/2KVH, 1024d, SwiGLU, RMSNorm)
//                  Fills remaining 31 codebooks per frame from backbone state.
//   Mimi decoder:  Kyutai Mimi codec (RVQ dequant + upsample + 8L transformer
//                  + SEANet decoder) -- 32 codebooks -> 24 kHz PCM.
//
// Inference flow:
//   1. Tokenize text with Llama-3.2 BPE.
//   2. Build token frames: each frame = (32 audio codebook slots + 1 text slot).
//   3. Backbone AR loop: for each step, embed all codebooks + text, sum,
//      run backbone, sample codebook-0 from backbone output.
//   4. Depth decoder loop: given backbone hidden state + codebook-0 embedding,
//      iteratively predict codebooks 1..31.
//   5. Collect all 32 codebook tokens per frame.
//   6. Mimi decode: RVQ dequant -> upsample conv -> transformer -> SEANet -> PCM.
//
// Reference: github.com/SesameAILabs/csm (Apache 2.0)

#include "csm_tts.h"

#include "core/attention.h"
#include "core/conv.h"
#include "core/bpe.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// ===================================================================
// Hyperparameters
// ===================================================================

namespace {

struct csm_hparams {
    // Backbone (Llama-3.2 1B)
    uint32_t bb_n_layers = 16;
    uint32_t bb_d_model = 2048;
    uint32_t bb_n_heads = 32;
    uint32_t bb_n_kv_heads = 8;
    uint32_t bb_head_dim = 64;
    uint32_t bb_ff_dim = 8192;
    uint32_t bb_max_pos = 2048;
    float bb_rope_theta = 500000.0f;
    float bb_rms_norm_eps = 1e-5f;

    // Depth decoder (Llama-3.2 100M)
    uint32_t dd_n_layers = 4;
    uint32_t dd_d_model = 1024;
    uint32_t dd_n_heads = 8;
    uint32_t dd_n_kv_heads = 2;
    uint32_t dd_head_dim = 128;
    uint32_t dd_ff_dim = 8192;
    uint32_t dd_backbone_hidden = 2048;

    // Audio / text
    uint32_t text_vocab_size = 128256;
    uint32_t audio_vocab_size = 2051;
    uint32_t audio_num_codebooks = 32;

    // Special tokens
    uint32_t audio_eos_token_id = 128003;
    uint32_t audio_token_id = 128002;
    uint32_t bos_token_id = 128000;
    uint32_t codebook_eos_token_id = 0;
    uint32_t codebook_pad_token_id = 2050;

    // Mimi codec
    uint32_t mimi_dim = 512;
    uint32_t mimi_n_heads = 8;
    uint32_t mimi_n_layers = 8;
    uint32_t mimi_codebook_dim = 256;
    uint32_t mimi_codebook_size = 2048;
    uint32_t mimi_n_quantizers = 32;
    uint32_t mimi_n_semantic = 1;
    uint32_t mimi_sample_rate = 24000;
    float mimi_frame_rate = 12.5f;
};

// ===================================================================
// Model structures
// ===================================================================

// Llama-style transformer layer (shared by backbone and depth decoder)
struct llama_layer {
    ggml_tensor* attn_norm_w = nullptr;   // RMSNorm
    ggml_tensor* attn_q_w = nullptr;      // (d, n_heads*head_dim)
    ggml_tensor* attn_k_w = nullptr;      // (d, n_kv_heads*head_dim)
    ggml_tensor* attn_v_w = nullptr;      // (d, n_kv_heads*head_dim)
    ggml_tensor* attn_output_w = nullptr; // (n_heads*head_dim, d)
    ggml_tensor* ffn_norm_w = nullptr;    // RMSNorm
    ggml_tensor* ffn_gate_w = nullptr;    // SwiGLU gate
    ggml_tensor* ffn_up_w = nullptr;      // SwiGLU up
    ggml_tensor* ffn_down_w = nullptr;    // SwiGLU down
};

// Mimi transformer layer (encoder or decoder)
struct mimi_tfm_layer {
    ggml_tensor* norm1_w = nullptr;
    ggml_tensor* norm1_b = nullptr;
    ggml_tensor* attn_qkv_w = nullptr; // fused QKV [3*dim, dim]
    ggml_tensor* attn_out_w = nullptr;
    ggml_tensor* ls1 = nullptr; // layer scale
    ggml_tensor* norm2_w = nullptr;
    ggml_tensor* norm2_b = nullptr;
    ggml_tensor* ffn_up_w = nullptr;   // dim -> 4*dim
    ggml_tensor* ffn_down_w = nullptr; // 4*dim -> dim
    ggml_tensor* ls2 = nullptr;        // layer scale
};

// SEANet conv layer
struct seanet_conv {
    ggml_tensor* w = nullptr;
    ggml_tensor* b = nullptr;
    ggml_tensor* w_perm = nullptr;
};

struct seanet_resblock {
    seanet_conv shortcut; // block.1
    seanet_conv expand;   // block.3
};

// SEANet decoder (symmetric to encoder, with transposed convs)
struct seanet_decoder {
    seanet_conv conv_init;       // model.0: Conv1d(512->1024, k=3, s=1)
    seanet_conv conv_stride[4];  // model.2, 5, 8, 11: ConvTranspose1d
    seanet_resblock resblock[4]; // model.3, 6, 9, 12
    seanet_conv conv_final;      // model.14: Conv1d(64->1, k=7, s=1)
};

// RVQ codebook
struct rvq_codebook {
    ggml_tensor* embedding = nullptr; // [codebook_dim, num_codes]
};

struct rvq_group {
    ggml_tensor* input_proj_w = nullptr;  // [mimi_dim, codebook_dim] (1x1 conv)
    ggml_tensor* output_proj_w = nullptr; // [codebook_dim, mimi_dim]
    std::vector<rvq_codebook> codebooks;
};

struct csm_model {
    csm_hparams hp;

    // Backbone
    ggml_tensor* bb_audio_embd_w = nullptr;     // (bb_d_model, audio_vocab_size * num_codebooks)
    ggml_tensor* bb_text_embd_w = nullptr;      // (bb_d_model, text_vocab_size)
    ggml_tensor* bb_output_norm_w = nullptr;    // RMSNorm
    ggml_tensor* bb_codebook0_head_w = nullptr; // (bb_d_model, audio_vocab_size)
    std::vector<llama_layer> bb_layers;

    // Depth decoder
    ggml_tensor* dd_token_embd_w = nullptr;     // (dd_d_model, audio_vocab_size)
    ggml_tensor* dd_projection_w = nullptr;     // (dd_backbone_hidden, dd_d_model)
    ggml_tensor* dd_output_norm_w = nullptr;    // RMSNorm
    ggml_tensor* dd_codebooks_head_w = nullptr; // ((num_codebooks-1) * dd_d_model, audio_vocab_size)
    std::vector<llama_layer> dd_layers;

    // Mimi codec decoder
    seanet_decoder seanet_dec;
    std::vector<mimi_tfm_layer> mimi_dec_layers;
    seanet_conv mimi_upsample; // stride-2 upsample (transposed conv)

    // Mimi RVQ (for dequantization in decode path)
    rvq_group rvq_first; // 1 semantic codebook
    rvq_group rvq_rest;  // 31 acoustic codebooks

    // Tokenizer
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;

    // Weight memory
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_context* ctx_perm = nullptr;
    ggml_backend_buffer_t buf_perm = nullptr;
};

} // namespace

struct csm_tts_context {
    csm_tts_context_params params{};
    int n_threads = 4;

    csm_model model;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Compute scheduler
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // Backbone KV cache
    ggml_context* bb_kv_ctx = nullptr;
    ggml_backend_buffer_t bb_kv_buf = nullptr;
    ggml_tensor* bb_kv_k = nullptr;
    ggml_tensor* bb_kv_v = nullptr;
    int bb_kv_max_ctx = 0;

    // Depth decoder KV cache (reset per frame)
    ggml_context* dd_kv_ctx = nullptr;
    ggml_backend_buffer_t dd_kv_buf = nullptr;
    ggml_tensor* dd_kv_k = nullptr;
    ggml_tensor* dd_kv_v = nullptr;

    // Sampler RNG state (xorshift64*)
    uint64_t rng_state = 0xdeadbeefcafebabeULL;

    // Diagnostic: the codes from the most recent synthesize() call
    // ([n_frames][audio_num_codebooks]). Populated before Mimi decode so the
    // diff harness can read them even if Mimi decode fails.
    std::vector<std::vector<int32_t>> last_codes;
    // Diagnostic: cap the greedy AR loop to this many frames (0 = unbounded).
    // Bounds the diff harness without shrinking the KV cache.
    int gen_frame_cap = 0;

    ~csm_tts_context() {
        if (sched)
            ggml_backend_sched_free(sched);
        if (bb_kv_buf)
            ggml_backend_buffer_free(bb_kv_buf);
        if (bb_kv_ctx)
            ggml_free(bb_kv_ctx);
        if (dd_kv_buf)
            ggml_backend_buffer_free(dd_kv_buf);
        if (dd_kv_ctx)
            ggml_free(dd_kv_ctx);
        if (model.buf_perm)
            ggml_backend_buffer_free(model.buf_perm);
        if (model.ctx_perm)
            ggml_free(model.ctx_perm);
        if (model.buf_w)
            ggml_backend_buffer_free(model.buf_w);
        if (model.ctx_w)
            ggml_free(model.ctx_w);
        if (backend && backend != backend_cpu)
            ggml_backend_free(backend);
        if (backend_cpu)
            ggml_backend_free(backend_cpu);
    }
};

namespace {

// ===================================================================
// Helper: read tensor data as F32 (handles F16/quantized tensors)
// ===================================================================

static void tensor_get_f32(ggml_tensor* t, float* out, size_t offset_bytes, size_t n_elem) {
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out, offset_bytes, n_elem * sizeof(float));
    } else {
        size_t raw_bytes = ggml_nbytes(t);
        if (offset_bytes > 0) {
            // Partial read: dequantize entire tensor then copy the slice
            std::vector<char> raw(raw_bytes);
            ggml_backend_tensor_get(t, raw.data(), 0, raw_bytes);
            size_t total_elem = ggml_nelements(t);
            std::vector<float> all(total_elem);
            const auto* tt = ggml_get_type_traits(t->type);
            if (tt && tt->to_float) {
                tt->to_float(raw.data(), all.data(), (int64_t)total_elem);
            } else {
                std::memset(all.data(), 0, total_elem * sizeof(float));
            }
            size_t start_elem = offset_bytes / sizeof(float);
            std::memcpy(out, all.data() + start_elem, n_elem * sizeof(float));
        } else {
            std::vector<char> raw(raw_bytes);
            ggml_backend_tensor_get(t, raw.data(), 0, raw_bytes);
            const auto* tt = ggml_get_type_traits(t->type);
            if (tt && tt->to_float) {
                tt->to_float(raw.data(), out, (int64_t)n_elem);
            } else {
                std::memset(out, 0, n_elem * sizeof(float));
            }
        }
    }
}

// ===================================================================
// xorshift64* sampler
// ===================================================================

static uint64_t xorshift64star(uint64_t& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 0x2545F4914F6CDD1DULL;
}

static float rand_uniform(uint64_t& state) {
    return (float)(xorshift64star(state) >> 11) / (float)(1ULL << 53);
}

// Top-k sampling with temperature
static int sample_topk(const float* logits, int vocab, int topk, float temperature, uint64_t& rng) {
    if (temperature <= 0.0f || topk <= 1) {
        // Greedy
        int best = 0;
        for (int i = 1; i < vocab; i++) {
            if (logits[i] > logits[best])
                best = i;
        }
        return best;
    }

    // Find top-k values
    std::vector<std::pair<float, int>> scored(vocab);
    for (int i = 0; i < vocab; i++) {
        scored[i] = {logits[i] / temperature, i};
    }
    std::partial_sort(scored.begin(), scored.begin() + topk, scored.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    // Softmax over top-k
    float maxv = scored[0].first;
    float sum = 0.0f;
    std::vector<float> probs(topk);
    for (int i = 0; i < topk; i++) {
        probs[i] = expf(scored[i].first - maxv);
        sum += probs[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < topk; i++) {
        probs[i] *= inv_sum;
    }

    // Sample
    float r = rand_uniform(rng);
    float acc = 0.0f;
    for (int i = 0; i < topk; i++) {
        acc += probs[i];
        if (acc >= r) {
            return scored[i].second;
        }
    }
    return scored[topk - 1].second;
}

// ===================================================================
// Conv1d helpers (mirrored from kyutai_stt.cpp)
// ===================================================================

static ggml_tensor* conv1d_causal(ggml_context* ctx, const seanet_conv& conv, ggml_tensor* x, int stride) {
    int kernel_size = (int)conv.w->ne[0];
    int pad_left = kernel_size - stride;
    if (pad_left > 0) {
        x = ggml_pad_ext(ctx, x, pad_left, 0, 0, 0, 0, 0, 0, 0);
    }
    ggml_tensor* out = ggml_conv_1d(ctx, conv.w, x, stride, 0, 1);
    out = ggml_reshape_2d(ctx, out, out->ne[0], out->ne[1]);
    if (conv.b) {
        ggml_tensor* b = ggml_reshape_2d(ctx, conv.b, 1, ggml_nelements(conv.b));
        out = ggml_add(ctx, out, b);
    }
    return out;
}

static ggml_tensor* elu(ggml_context* ctx, ggml_tensor* x) {
    return ggml_elu(ctx, x);
}

static ggml_tensor* resblock_fwd(ggml_context* ctx, const seanet_resblock& rb, ggml_tensor* x) {
    ggml_tensor* h = elu(ctx, x);
    h = conv1d_causal(ctx, rb.shortcut, h, 1);
    h = elu(ctx, h);
    h = conv1d_causal(ctx, rb.expand, h, 1);
    return ggml_add(ctx, x, h);
}

// Transposed conv1d for SEANet decoder (right-padded to match encoder's left-padded causal)
static ggml_tensor* conv1d_transpose(ggml_context* ctx, const seanet_conv& conv, ggml_tensor* x, int stride) {
    if (conv.w_perm) {
        const int K = (int)conv.w->ne[0];
        return core_convt::convt1d_decomp_tf(ctx, x, conv.w_perm, conv.b, stride, K, 0, K - stride);
    }
    // ggml_conv_transpose_1d: kernel a, data b
    ggml_tensor* out = ggml_conv_transpose_1d(ctx, conv.w, x, stride, 0, 1);
    // Trim the right padding: output length should be stride * input_len
    int in_len = (int)x->ne[0];
    int out_channels = (int)conv.w->ne[1];
    int expected_len = in_len * stride;
    int actual_len = (int)out->ne[0];
    if (actual_len > expected_len) {
        out = ggml_view_2d(ctx, out, expected_len, out_channels, out->nb[1], 0);
        out = ggml_cont(ctx, out);
    }
    out = ggml_reshape_2d(ctx, out, out->ne[0], out->ne[1]);
    if (conv.b) {
        ggml_tensor* b = ggml_reshape_2d(ctx, conv.b, 1, ggml_nelements(conv.b));
        out = ggml_add(ctx, out, b);
    }
    return out;
}

// ===================================================================
// Metadata / weight loading
// ===================================================================

static void load_metadata(csm_tts_context* c, gguf_context* g) {
    auto& hp = c->model.hp;

    // Backbone
    hp.bb_n_layers = core_gguf::kv_u32(g, "csm.backbone.n_layers", hp.bb_n_layers);
    hp.bb_d_model = core_gguf::kv_u32(g, "csm.backbone.d_model", hp.bb_d_model);
    hp.bb_n_heads = core_gguf::kv_u32(g, "csm.backbone.n_heads", hp.bb_n_heads);
    hp.bb_n_kv_heads = core_gguf::kv_u32(g, "csm.backbone.n_kv_heads", hp.bb_n_kv_heads);
    hp.bb_head_dim = core_gguf::kv_u32(g, "csm.backbone.head_dim", hp.bb_head_dim);
    hp.bb_ff_dim = core_gguf::kv_u32(g, "csm.backbone.ff_dim", hp.bb_ff_dim);
    hp.bb_max_pos = core_gguf::kv_u32(g, "csm.backbone.max_pos", hp.bb_max_pos);
    hp.bb_rope_theta = core_gguf::kv_f32(g, "csm.backbone.rope_theta", hp.bb_rope_theta);
    hp.bb_rms_norm_eps = core_gguf::kv_f32(g, "csm.backbone.rms_norm_eps", hp.bb_rms_norm_eps);

    // Depth decoder
    hp.dd_n_layers = core_gguf::kv_u32(g, "csm.depth.n_layers", hp.dd_n_layers);
    hp.dd_d_model = core_gguf::kv_u32(g, "csm.depth.d_model", hp.dd_d_model);
    hp.dd_n_heads = core_gguf::kv_u32(g, "csm.depth.n_heads", hp.dd_n_heads);
    hp.dd_n_kv_heads = core_gguf::kv_u32(g, "csm.depth.n_kv_heads", hp.dd_n_kv_heads);
    hp.dd_head_dim = core_gguf::kv_u32(g, "csm.depth.head_dim", hp.dd_head_dim);
    hp.dd_ff_dim = core_gguf::kv_u32(g, "csm.depth.ff_dim", hp.dd_ff_dim);
    hp.dd_backbone_hidden = core_gguf::kv_u32(g, "csm.depth.backbone_hidden", hp.dd_backbone_hidden);

    // Audio / text
    hp.text_vocab_size = core_gguf::kv_u32(g, "csm.text_vocab_size", hp.text_vocab_size);
    hp.audio_vocab_size = core_gguf::kv_u32(g, "csm.audio_vocab_size", hp.audio_vocab_size);
    hp.audio_num_codebooks = core_gguf::kv_u32(g, "csm.audio_num_codebooks", hp.audio_num_codebooks);

    // Special tokens
    hp.audio_eos_token_id = core_gguf::kv_u32(g, "csm.audio_eos_token_id", hp.audio_eos_token_id);
    hp.audio_token_id = core_gguf::kv_u32(g, "csm.audio_token_id", hp.audio_token_id);
    hp.bos_token_id = core_gguf::kv_u32(g, "csm.bos_token_id", hp.bos_token_id);
    hp.codebook_eos_token_id = core_gguf::kv_u32(g, "csm.codebook_eos_token_id", hp.codebook_eos_token_id);
    hp.codebook_pad_token_id = core_gguf::kv_u32(g, "csm.codebook_pad_token_id", hp.codebook_pad_token_id);

    // Mimi
    hp.mimi_dim = core_gguf::kv_u32(g, "csm.mimi.dim", hp.mimi_dim);
    hp.mimi_n_heads = core_gguf::kv_u32(g, "csm.mimi.n_heads", hp.mimi_n_heads);
    hp.mimi_n_layers = core_gguf::kv_u32(g, "csm.mimi.n_layers", hp.mimi_n_layers);
    hp.mimi_codebook_dim = core_gguf::kv_u32(g, "csm.mimi.codebook_dim", hp.mimi_codebook_dim);
    hp.mimi_codebook_size = core_gguf::kv_u32(g, "csm.mimi.codebook_size", hp.mimi_codebook_size);
    hp.mimi_n_quantizers = core_gguf::kv_u32(g, "csm.mimi.n_quantizers", hp.mimi_n_quantizers);
    hp.mimi_n_semantic = core_gguf::kv_u32(g, "csm.mimi.n_semantic", hp.mimi_n_semantic);
    hp.mimi_sample_rate = core_gguf::kv_u32(g, "csm.mimi.sample_rate", hp.mimi_sample_rate);
    hp.mimi_frame_rate = core_gguf::kv_f32(g, "csm.mimi.frame_rate", hp.mimi_frame_rate);

    // Tokenizer
    auto tok = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
    if (!tok.empty()) {
        c->model.id_to_token = std::move(tok);
        c->model.token_to_id.reserve(c->model.id_to_token.size());
        for (int i = 0; i < (int)c->model.id_to_token.size(); i++) {
            c->model.token_to_id[c->model.id_to_token[i]] = i;
        }
    }
    auto merges = core_gguf::kv_str_array(g, "tokenizer.ggml.merges");
    for (size_t i = 0; i < merges.size(); i++) {
        c->model.merge_rank[merges[i]] = (int32_t)i;
    }
}

static void load_seanet_conv(const std::map<std::string, ggml_tensor*>& ts, const char* prefix, seanet_conv& conv) {
    std::string w_name = std::string(prefix) + ".weight";
    std::string b_name = std::string(prefix) + ".bias";
    auto it_w = ts.find(w_name);
    auto it_b = ts.find(b_name);
    conv.w = (it_w != ts.end()) ? it_w->second : nullptr;
    conv.b = (it_b != ts.end()) ? it_b->second : nullptr;
}

static bool bind_weights(csm_tts_context* c) {
    auto& m = c->model;
    auto& hp = m.hp;
    auto& ts = c->tensors;

    // --- Backbone ---
    m.bb_audio_embd_w = core_gguf::require(ts, "backbone.audio_embd.weight", "csm");
    m.bb_text_embd_w = core_gguf::require(ts, "backbone.text_embd.weight", "csm");
    m.bb_output_norm_w = core_gguf::require(ts, "backbone.output_norm.weight", "csm");
    m.bb_codebook0_head_w = core_gguf::require(ts, "backbone.codebook0_head.weight", "csm");

    m.bb_layers.resize(hp.bb_n_layers);
    for (uint32_t i = 0; i < hp.bb_n_layers; i++) {
        auto& b = m.bb_layers[i];
        char key[80];
#define BB_FMT(fld, sub)                                                                                               \
    do {                                                                                                               \
        std::snprintf(key, sizeof(key), "backbone.blk.%u." sub ".weight", i);                                          \
        b.fld = core_gguf::require(ts, key, "csm");                                                                    \
    } while (0)
        BB_FMT(attn_norm_w, "attn_norm");
        BB_FMT(attn_q_w, "attn_q");
        BB_FMT(attn_k_w, "attn_k");
        BB_FMT(attn_v_w, "attn_v");
        BB_FMT(attn_output_w, "attn_output");
        BB_FMT(ffn_norm_w, "ffn_norm");
        BB_FMT(ffn_gate_w, "ffn_gate");
        BB_FMT(ffn_up_w, "ffn_up");
        BB_FMT(ffn_down_w, "ffn_down");
#undef BB_FMT
        if (!b.attn_norm_w || !b.attn_q_w)
            return false;
    }

    // --- Depth decoder ---
    m.dd_token_embd_w = core_gguf::require(ts, "depth.token_embd.weight", "csm");
    m.dd_projection_w = core_gguf::require(ts, "depth.projection.weight", "csm");
    m.dd_output_norm_w = core_gguf::require(ts, "depth.output_norm.weight", "csm");
    m.dd_codebooks_head_w = core_gguf::require(ts, "depth.codebooks_head.weight", "csm");

    m.dd_layers.resize(hp.dd_n_layers);
    for (uint32_t i = 0; i < hp.dd_n_layers; i++) {
        auto& b = m.dd_layers[i];
        char key[80];
#define DD_FMT(fld, sub)                                                                                               \
    do {                                                                                                               \
        std::snprintf(key, sizeof(key), "depth.blk.%u." sub ".weight", i);                                             \
        b.fld = core_gguf::require(ts, key, "csm");                                                                    \
    } while (0)
        DD_FMT(attn_norm_w, "attn_norm");
        DD_FMT(attn_q_w, "attn_q");
        DD_FMT(attn_k_w, "attn_k");
        DD_FMT(attn_v_w, "attn_v");
        DD_FMT(attn_output_w, "attn_output");
        DD_FMT(ffn_norm_w, "ffn_norm");
        DD_FMT(ffn_gate_w, "ffn_gate");
        DD_FMT(ffn_up_w, "ffn_up");
        DD_FMT(ffn_down_w, "ffn_down");
#undef DD_FMT
        if (!b.attn_norm_w || !b.attn_q_w)
            return false;
    }

    // --- Mimi decoder transformer ---
    m.mimi_dec_layers.resize(hp.mimi_n_layers);
    for (uint32_t i = 0; i < hp.mimi_n_layers; i++) {
        auto& L = m.mimi_dec_layers[i];
        char buf[128];
        std::snprintf(buf, sizeof(buf), "mimi.dec_tfm.layers.%u.norm1.weight", i);
        L.norm1_w = core_gguf::require(ts, buf, "csm");
        std::snprintf(buf, sizeof(buf), "mimi.dec_tfm.layers.%u.norm1.bias", i);
        L.norm1_b = core_gguf::try_get(ts, buf);
        std::snprintf(buf, sizeof(buf), "mimi.dec_tfm.layers.%u.attn.qkv_w", i);
        L.attn_qkv_w = core_gguf::require(ts, buf, "csm");
        std::snprintf(buf, sizeof(buf), "mimi.dec_tfm.layers.%u.attn.out_w", i);
        L.attn_out_w = core_gguf::require(ts, buf, "csm");
        std::snprintf(buf, sizeof(buf), "mimi.dec_tfm.layers.%u.ls1", i);
        L.ls1 = core_gguf::require(ts, buf, "csm");
        std::snprintf(buf, sizeof(buf), "mimi.dec_tfm.layers.%u.norm2.weight", i);
        L.norm2_w = core_gguf::require(ts, buf, "csm");
        std::snprintf(buf, sizeof(buf), "mimi.dec_tfm.layers.%u.norm2.bias", i);
        L.norm2_b = core_gguf::try_get(ts, buf);
        std::snprintf(buf, sizeof(buf), "mimi.dec_tfm.layers.%u.ffn_up_w", i);
        L.ffn_up_w = core_gguf::require(ts, buf, "csm");
        std::snprintf(buf, sizeof(buf), "mimi.dec_tfm.layers.%u.ffn_down_w", i);
        L.ffn_down_w = core_gguf::require(ts, buf, "csm");
        std::snprintf(buf, sizeof(buf), "mimi.dec_tfm.layers.%u.ls2", i);
        L.ls2 = core_gguf::require(ts, buf, "csm");
    }

    // --- Mimi SEANet decoder ---
    // Decoder is symmetric to encoder with reversed layer order:
    //   model.0:  Conv1d(512->1024, k=3, s=1) -- init conv
    //   model.2:  ConvTranspose1d(1024->512, k=16, s=8)
    //   model.3:  ResnetBlock(512)
    //   model.5:  ConvTranspose1d(512->256, k=12, s=6)
    //   model.6:  ResnetBlock(256)
    //   model.8:  ConvTranspose1d(256->128, k=10, s=5)
    //   model.9:  ResnetBlock(128)
    //   model.11: ConvTranspose1d(128->64, k=8, s=4)
    //   model.12: ResnetBlock(64)
    //   model.14: Conv1d(64->1, k=7, s=1) -- final conv
    load_seanet_conv(ts, "mimi.decoder.model.0.conv2", m.seanet_dec.conv_init);
    load_seanet_conv(ts, "mimi.decoder.model.2.conv2", m.seanet_dec.conv_stride[0]);
    load_seanet_conv(ts, "mimi.decoder.model.3.blk1", m.seanet_dec.resblock[0].shortcut);
    load_seanet_conv(ts, "mimi.decoder.model.3.blk3", m.seanet_dec.resblock[0].expand);
    load_seanet_conv(ts, "mimi.decoder.model.5.conv2", m.seanet_dec.conv_stride[1]);
    load_seanet_conv(ts, "mimi.decoder.model.6.blk1", m.seanet_dec.resblock[1].shortcut);
    load_seanet_conv(ts, "mimi.decoder.model.6.blk3", m.seanet_dec.resblock[1].expand);
    load_seanet_conv(ts, "mimi.decoder.model.8.conv2", m.seanet_dec.conv_stride[2]);
    load_seanet_conv(ts, "mimi.decoder.model.9.blk1", m.seanet_dec.resblock[2].shortcut);
    load_seanet_conv(ts, "mimi.decoder.model.9.blk3", m.seanet_dec.resblock[2].expand);
    load_seanet_conv(ts, "mimi.decoder.model.11.conv2", m.seanet_dec.conv_stride[3]);
    load_seanet_conv(ts, "mimi.decoder.model.12.blk1", m.seanet_dec.resblock[3].shortcut);
    load_seanet_conv(ts, "mimi.decoder.model.12.blk3", m.seanet_dec.resblock[3].expand);
    load_seanet_conv(ts, "mimi.decoder.model.14.conv2", m.seanet_dec.conv_final);

    // Mimi upsample (stride-2 transposed conv)
    load_seanet_conv(ts, "mimi.upsample.conv3", m.mimi_upsample);

    // --- RVQ codebooks ---
    uint32_t n_acoustic = hp.mimi_n_quantizers - hp.mimi_n_semantic;
    m.rvq_first.input_proj_w = core_gguf::try_get(ts, "mimi.quantizer.rvq_first.input_proj.weight");
    m.rvq_first.output_proj_w = core_gguf::try_get(ts, "mimi.quantizer.rvq_first.output_proj.weight");
    m.rvq_first.codebooks.resize(hp.mimi_n_semantic);
    for (uint32_t i = 0; i < hp.mimi_n_semantic; i++) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "mimi.quantizer.rvq_first.vq.layers.%u._codebook.embedding", i);
        m.rvq_first.codebooks[i].embedding = core_gguf::try_get(ts, buf);
    }

    m.rvq_rest.input_proj_w = core_gguf::try_get(ts, "mimi.quantizer.rvq_rest.input_proj.weight");
    m.rvq_rest.output_proj_w = core_gguf::try_get(ts, "mimi.quantizer.rvq_rest.output_proj.weight");
    m.rvq_rest.codebooks.resize(n_acoustic);
    for (uint32_t i = 0; i < n_acoustic; i++) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "mimi.quantizer.rvq_rest.vq.layers.%u._codebook.embedding", i);
        m.rvq_rest.codebooks[i].embedding = core_gguf::try_get(ts, buf);
    }

    return true;
}

// ===================================================================
// KV cache initialization
// ===================================================================

static bool init_bb_kv_cache(csm_tts_context* ctx, int max_ctx) {
    auto& hp = ctx->model.hp;

    struct ggml_init_params gp = {
        /*.mem_size   =*/ggml_tensor_overhead() * 2,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ctx->bb_kv_ctx = ggml_init(gp);
    if (!ctx->bb_kv_ctx)
        return false;

    const auto kv_pair = core_attn::kv_dtype_pair_from_env("csm_tts");
    ctx->bb_kv_k =
        ggml_new_tensor_4d(ctx->bb_kv_ctx, kv_pair.k, hp.bb_head_dim, max_ctx, hp.bb_n_kv_heads, hp.bb_n_layers);
    ctx->bb_kv_v =
        ggml_new_tensor_4d(ctx->bb_kv_ctx, kv_pair.v, hp.bb_head_dim, max_ctx, hp.bb_n_kv_heads, hp.bb_n_layers);

    ggml_backend_t kv_backend = core_attn::kv_backend_from_env(ctx->backend, ctx->backend_cpu, "csm_tts");
    ctx->bb_kv_buf = ggml_backend_alloc_ctx_tensors(ctx->bb_kv_ctx, kv_backend);
    if (!ctx->bb_kv_buf) {
        ggml_free(ctx->bb_kv_ctx);
        ctx->bb_kv_ctx = nullptr;
        return false;
    }
    ggml_backend_buffer_clear(ctx->bb_kv_buf, 0);
    ctx->bb_kv_max_ctx = max_ctx;
    return true;
}

static bool init_dd_kv_cache(csm_tts_context* ctx) {
    auto& hp = ctx->model.hp;
    // Depth decoder max sequence = audio_num_codebooks (33 positions max)
    int max_ctx = (int)hp.audio_num_codebooks + 1;

    struct ggml_init_params gp = {
        /*.mem_size   =*/ggml_tensor_overhead() * 2,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ctx->dd_kv_ctx = ggml_init(gp);
    if (!ctx->dd_kv_ctx)
        return false;

    ctx->dd_kv_k =
        ggml_new_tensor_4d(ctx->dd_kv_ctx, GGML_TYPE_F16, hp.dd_head_dim, max_ctx, hp.dd_n_kv_heads, hp.dd_n_layers);
    ctx->dd_kv_v =
        ggml_new_tensor_4d(ctx->dd_kv_ctx, GGML_TYPE_F16, hp.dd_head_dim, max_ctx, hp.dd_n_kv_heads, hp.dd_n_layers);

    ggml_backend_t kv_backend = core_attn::kv_backend_from_env(ctx->backend, ctx->backend_cpu, "csm_tts_depth");
    ctx->dd_kv_buf = ggml_backend_alloc_ctx_tensors(ctx->dd_kv_ctx, kv_backend);
    if (!ctx->dd_kv_buf) {
        ggml_free(ctx->dd_kv_ctx);
        ctx->dd_kv_ctx = nullptr;
        return false;
    }
    ggml_backend_buffer_clear(ctx->dd_kv_buf, 0);
    return true;
}

// ===================================================================
// BPE tokenization
// ===================================================================

static std::vector<int32_t> tokenize_text(const csm_model& m, const std::string& text, int speaker = 0) {
    // CSM format: "[{speaker}]{text}" wrapped with BOS + EOS.
    // The original repo: tokenizer.encode(f"[{speaker}]{text}")
    // With Llama-3.2 BPE template: [BOS] tokens [EOS]
    std::string formatted = "[" + std::to_string(speaker) + "]" + text;
    std::vector<int32_t> tokens = core_bpe::tokenize_simple(m.token_to_id, m.merge_rank, formatted);

    // Add BOS (128000) at start and EOS (128001) at end
    const int32_t bos_id = 128000;
    const int32_t eos_id = 128001;
    tokens.insert(tokens.begin(), bos_id);
    tokens.push_back(eos_id);
    return tokens;
}

// ===================================================================
// Mimi RVQ dequantize (codes -> continuous representation)
// ===================================================================

// Dequantize RVQ codes back to continuous representation.
// codes: [n_codebooks, T_frames] array of codebook indices.
// Returns: [mimi_dim, T_frames] continuous representation.
static bool rvq_dequantize(csm_tts_context* ctx, const int32_t* codes, int n_codebooks, int T_frames,
                           std::vector<float>& out) {
    auto& m = ctx->model;
    auto& hp = m.hp;
    int cdim = (int)hp.mimi_codebook_dim;
    int mdim = (int)hp.mimi_dim;

    // Step 1: For each RVQ group, look up codebook embeddings and sum.
    // semantic (first codebook):
    //   projected = output_proj(sum of codebook lookups)
    // acoustic (remaining codebooks):
    //   projected = output_proj(sum of codebook lookups)
    // final = semantic_projected + acoustic_projected

    // Read codebook embeddings to CPU
    auto read_codebook = [](const rvq_codebook& cb, int cb_dim, int num_codes) -> std::vector<float> {
        std::vector<float> data(num_codes * cb_dim);
        if (cb.embedding) {
            tensor_get_f32(cb.embedding, data.data(), 0, data.size());
        }
        return data;
    };

    auto read_proj = [](ggml_tensor* w) -> std::vector<float> {
        if (!w)
            return {};
        size_t n = ggml_nelements(w);
        std::vector<float> data(n);
        tensor_get_f32(w, data.data(), 0, n);
        return data;
    };

    // Semantic group (1 codebook)
    int n_semantic = (int)hp.mimi_n_semantic;
    int n_acoustic = n_codebooks - n_semantic;

    // Accumulate quantized vectors per group
    auto accumulate_group = [&](const rvq_group& grp, int cb_start, int n_cb) -> std::vector<float> {
        std::vector<float> acc(cdim * T_frames, 0.0f);
        for (int q = 0; q < n_cb; q++) {
            int cb_idx = q;
            if (cb_idx >= (int)grp.codebooks.size() || !grp.codebooks[cb_idx].embedding)
                continue;
            auto cb_data = read_codebook(grp.codebooks[cb_idx], cdim, (int)grp.codebooks[cb_idx].embedding->ne[1]);
            for (int t = 0; t < T_frames; t++) {
                int code = codes[(cb_start + q) * T_frames + t];
                for (int d = 0; d < cdim; d++) {
                    acc[t * cdim + d] += cb_data[code * cdim + d];
                }
            }
        }
        return acc;
    };

    // Project from codebook_dim to mimi_dim using output_proj.
    // The output_proj weight is a Conv1d(codebook_dim, mimi_dim, kernel_size=1),
    // stored in GGUF as 3D: ne[0]=kernel_size(1), ne[1]=in_ch(cdim=256), ne[2]=out_ch(mdim=512).
    // We treat it as a 2D matrix [in_ch, out_ch] = [256, 512].
    auto project = [&](const std::vector<float>& acc, ggml_tensor* proj_w) -> std::vector<float> {
        std::vector<float> result(mdim * T_frames, 0.0f);
        if (!proj_w) {
            // No projection, pad or truncate
            int copy_dim = std::min(cdim, mdim);
            for (int t = 0; t < T_frames; t++) {
                for (int d = 0; d < copy_dim; d++) {
                    result[t * mdim + d] = acc[t * cdim + d];
                }
            }
            return result;
        }
        auto proj_data = read_proj(proj_w);
        // For Conv1d(in=cdim, out=mdim, k=1): weight shape is [out, in, k] = [mdim, cdim, 1]
        // GGUF stores as ne=[1, cdim, mdim]. Data layout: proj_data[out * cdim * 1 + in * 1 + 0]
        // Effective 2D: proj_data[out * cdim + in]
        // result[t, out] = sum_in(proj_data[out * cdim + in] * acc[t * cdim + in])
        int in_ch = cdim;  // 256
        int out_ch = mdim; // 512
        // Handle both 2D [cdim, mdim] and 3D [1, cdim, mdim] GGUF layouts
        if (ggml_n_dims(proj_w) == 3) {
            in_ch = (int)proj_w->ne[1];
            out_ch = (int)proj_w->ne[2];
        } else {
            in_ch = (int)proj_w->ne[0];
            out_ch = (int)proj_w->ne[1];
        }
        for (int t = 0; t < T_frames; t++) {
            for (int d = 0; d < out_ch && d < mdim; d++) {
                float sum = 0.0f;
                for (int k = 0; k < in_ch && k < cdim; k++) {
                    sum += proj_data[(size_t)d * in_ch + k] * acc[t * cdim + k];
                }
                result[t * mdim + d] = sum;
            }
        }
        return result;
    };

    auto sem_acc = accumulate_group(m.rvq_first, 0, n_semantic);
    auto sem_proj = project(sem_acc, m.rvq_first.output_proj_w);

    auto aco_acc = accumulate_group(m.rvq_rest, n_semantic, n_acoustic);
    auto aco_proj = project(aco_acc, m.rvq_rest.output_proj_w);

    // Sum semantic + acoustic
    out.resize(mdim * T_frames);
    for (int i = 0; i < mdim * T_frames; i++) {
        out[i] = sem_proj[i] + aco_proj[i];
    }

    return true;
}

// ===================================================================
// Mimi decoder: continuous repr -> PCM
// ===================================================================

// Build the Mimi decoder graph:
// [mimi_dim, T_frames] -> upsample -> transformer -> SEANet decoder -> [n_samples]
//
// This is the symmetric counterpart of the Mimi encoder in kyutai_stt.cpp.
// The Mimi decoder architecture:
//   1. Upsample conv (stride-2 transposed conv): T_frames -> 2*T_frames
//   2. Transformer (8 layers): process upsampled features
//   3. SEANet decoder:
//      - init conv (512->1024, k=3)
//      - 4x [transpose conv (stride 8,6,5,4) + resblock]
//      - final conv (64->1, k=7)
//
// Note: The decoder reverses the encoder's striding schedule.

// Mimi decoder transformer (mirrors encoder transformer from kyutai_stt.cpp)
static ggml_tensor* build_mimi_dec_transformer(ggml_context* ctx, const std::vector<mimi_tfm_layer>& layers,
                                               ggml_tensor* x, int n_heads, int head_dim) {
    int T = (int)x->ne[1];
    int dim = (int)x->ne[0];

    ggml_tensor* positions = ggml_arange(ctx, 0.0f, (float)T, 1.0f);
    positions = ggml_cast(ctx, positions, GGML_TYPE_I32);

    for (size_t li = 0; li < layers.size(); li++) {
        const auto& L = layers[li];
        ggml_tensor* residual = x;

        // Pre-norm (LayerNorm)
        ggml_tensor* h = ggml_norm(ctx, x, 1e-5f);
        h = ggml_mul(ctx, h, L.norm1_w);
        if (L.norm1_b)
            h = ggml_add(ctx, h, L.norm1_b);

        // Self-attention with fused QKV
        ggml_tensor* qkv = ggml_mul_mat(ctx, L.attn_qkv_w, h);
        ggml_tensor* Q = ggml_view_2d(ctx, qkv, dim, T, qkv->nb[1], 0);
        ggml_tensor* K = ggml_view_2d(ctx, qkv, dim, T, qkv->nb[1], dim * ggml_type_size(qkv->type));
        ggml_tensor* V = ggml_view_2d(ctx, qkv, dim, T, qkv->nb[1], 2 * dim * ggml_type_size(qkv->type));

        Q = ggml_reshape_3d(ctx, ggml_cont(ctx, Q), head_dim, n_heads, T);
        K = ggml_reshape_3d(ctx, ggml_cont(ctx, K), head_dim, n_heads, T);
        V = ggml_reshape_3d(ctx, ggml_cont(ctx, V), head_dim, n_heads, T);

        // RoPE (decoder uses rope_theta=10000)
        Q = ggml_rope_ext(ctx, Q, positions, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f,
                          0.0f, 0.0f);
        K = ggml_rope_ext(ctx, K, positions, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f,
                          0.0f, 0.0f);

        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

        // Non-causal full attention for decoder transformer
        ggml_tensor* attn = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, 1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);
        attn = ggml_reshape_2d(ctx, attn, dim, T);

        attn = ggml_mul_mat(ctx, L.attn_out_w, attn);

        if (L.ls1) {
            ggml_tensor* ls = ggml_reshape_2d(ctx, L.ls1, dim, 1);
            attn = ggml_mul(ctx, attn, ls);
        }

        x = ggml_add(ctx, residual, attn);

        // FFN
        residual = x;
        h = ggml_norm(ctx, x, 1e-5f);
        h = ggml_mul(ctx, h, L.norm2_w);
        if (L.norm2_b)
            h = ggml_add(ctx, h, L.norm2_b);

        h = ggml_mul_mat(ctx, L.ffn_up_w, h);
        h = ggml_gelu(ctx, h);
        h = ggml_mul_mat(ctx, L.ffn_down_w, h);

        if (L.ls2) {
            ggml_tensor* ls = ggml_reshape_2d(ctx, L.ls2, dim, 1);
            h = ggml_mul(ctx, h, ls);
        }

        x = ggml_add(ctx, residual, h);
    }

    return x;
}

// Build SEANet decoder: [mimi_dim, T] -> [n_samples]
// Reverses the encoder: init conv -> 4x (transpose conv + resblock) -> final conv
static ggml_tensor* build_seanet_decoder(ggml_context* ctx, const seanet_decoder& sd, ggml_tensor* x) {
    // x is [mimi_dim=512, T] in column-major
    // Transpose to [T, 512] for conv1d (ggml conv1d: data b = [T, IC])
    x = ggml_cont(ctx, ggml_transpose(ctx, x));

    // Decoder stride schedule (reverse of encoder): [8, 6, 5, 4]
    static const int strides[] = {8, 6, 5, 4};

    // model.0: Conv1d(512->1024, k=3, s=1)
    x = conv1d_causal(ctx, sd.conv_init, x, 1);

    // 4 stages: transpose conv -> resblock
    for (int i = 0; i < 4; i++) {
        x = elu(ctx, x);
        x = conv1d_transpose(ctx, sd.conv_stride[i], x, strides[i]);
        x = resblock_fwd(ctx, sd.resblock[i], x);
    }

    // model.14: Conv1d(64->1, k=7, s=1)
    x = elu(ctx, x);
    x = conv1d_causal(ctx, sd.conv_final, x, 1);

    // Output: [T_pcm, 1] -> flatten to [T_pcm]
    x = ggml_reshape_1d(ctx, x, ggml_nelements(x));

    return x;
}

} // namespace

// ===================================================================
// Public C ABI
// ===================================================================

extern "C" struct csm_tts_context_params csm_tts_context_default_params(void) {
    csm_tts_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.temperature = 0.9f;
    p.topk = 50;
    p.seed = 0;
    p.max_audio_tokens = 0;
    return p;
}

extern "C" struct csm_tts_context* csm_tts_init_from_file(const char* path_model,
                                                          struct csm_tts_context_params params) {
    auto* c = new csm_tts_context();
    c->params = params;
    c->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    if (params.seed != 0) {
        c->rng_state = params.seed;
    }

    // Backend
    c->backend = params.use_gpu ? ggml_backend_init_best() : ggml_backend_cpu_init();
    if (!c->backend)
        c->backend = ggml_backend_cpu_init();
    c->backend_cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(c->backend_cpu, c->n_threads);
    if (ggml_backend_is_cpu(c->backend))
        ggml_backend_cpu_set_n_threads(c->backend, c->n_threads);

    // Pass 1: metadata
    {
        gguf_context* gctx = core_gguf::open_metadata(path_model);
        if (!gctx) {
            fprintf(stderr, "csm_tts: failed to open '%s'\n", path_model);
            delete c;
            return nullptr;
        }
        load_metadata(c, gctx);
        core_gguf::free_metadata(gctx);
    }

    // Pass 2: weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, c->backend, "csm_tts", wl)) {
        fprintf(stderr, "csm_tts: failed to load weights from '%s'\n", path_model);
        delete c;
        return nullptr;
    }
    c->model.ctx_w = wl.ctx;
    c->model.buf_w = wl.buf;
    c->tensors = std::move(wl.tensors);

    if (!bind_weights(c)) {
        fprintf(stderr, "csm_tts: failed to bind weights\n");
        delete c;
        return nullptr;
    }

    // Permute ConvTranspose1d weights
    {
        ggml_tensor *srcs[4], **dsts_arr[4];
        for (int i = 0; i < 4; i++) {
            srcs[i] = c->model.seanet_dec.conv_stride[i].w;
            dsts_arr[i] = &c->model.seanet_dec.conv_stride[i].w_perm;
        }
        core_convt::permute_convt1d_weights_batch(srcs, dsts_arr, 4, c->backend, &c->model.ctx_perm,
                                                  &c->model.buf_perm);
    }

    // Scheduler
    int n_be = 1;
    ggml_backend_t backends[2] = {c->backend, nullptr};
    if (c->backend_cpu && c->backend_cpu != c->backend) {
        backends[n_be++] = c->backend_cpu;
    }
    c->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    c->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    // KV caches
    int max_ctx = (int)c->model.hp.bb_max_pos;
    if (params.max_audio_tokens > 0 && params.max_audio_tokens < max_ctx) {
        max_ctx = params.max_audio_tokens;
    }
    if (!init_bb_kv_cache(c, max_ctx)) {
        fprintf(stderr, "csm_tts: failed to allocate backbone KV cache\n");
        delete c;
        return nullptr;
    }
    if (!init_dd_kv_cache(c)) {
        fprintf(stderr, "csm_tts: failed to allocate depth decoder KV cache\n");
        delete c;
        return nullptr;
    }

    auto& hp = c->model.hp;
    if (params.verbosity >= 1) {
        fprintf(stderr, "csm_tts: loaded backbone %uL/%ud/%uH + depth %uL/%ud/%uH\n", hp.bb_n_layers, hp.bb_d_model,
                hp.bb_n_heads, hp.dd_n_layers, hp.dd_d_model, hp.dd_n_heads);
        fprintf(stderr, "csm_tts: mimi %uL/%ud, codebooks=%u, audio_vocab=%u, text_vocab=%u\n", hp.mimi_n_layers,
                hp.mimi_dim, hp.audio_num_codebooks, hp.audio_vocab_size, hp.text_vocab_size);
    }

    return c;
}

extern "C" void csm_tts_free(struct csm_tts_context* ctx) {
    delete ctx;
}

extern "C" void csm_tts_pcm_free(float* pcm) {
    free(pcm);
}

extern "C" void csm_tts_set_temperature(struct csm_tts_context* ctx, float temperature) {
    if (ctx)
        ctx->params.temperature = temperature;
}

extern "C" void csm_tts_set_topk(struct csm_tts_context* ctx, int topk) {
    if (ctx)
        ctx->params.topk = topk;
}

extern "C" void csm_tts_set_seed(struct csm_tts_context* ctx, uint64_t seed) {
    if (ctx) {
        ctx->params.seed = seed;
        if (seed != 0) {
            ctx->rng_state = seed;
        }
    }
}

extern "C" void csm_tts_set_n_threads(struct csm_tts_context* ctx, int n_threads) {
    if (ctx) {
        ctx->n_threads = n_threads > 0 ? n_threads : 4;
        if (ggml_backend_is_cpu(ctx->backend))
            ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    }
}

// ===================================================================
// Synthesis (main entry point)
// ===================================================================

extern "C" float* csm_tts_synthesize(struct csm_tts_context* ctx, const char* text, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples) {
        return nullptr;
    }
    return csm_tts_synthesize_with_reference(ctx, text, nullptr, 0, nullptr, out_n_samples);
}

extern "C" float* csm_tts_synthesize_with_reference(struct csm_tts_context* ctx, const char* text,
                                                    const float* /*ref_pcm*/, int /*ref_n_samples*/,
                                                    const char* /*ref_text*/, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples) {
        return nullptr;
    }

    auto& m = ctx->model;
    auto& hp = m.hp;
    float temperature = ctx->params.temperature;
    int topk = ctx->params.topk;

    // 1. Tokenize text
    std::vector<int32_t> text_tokens = tokenize_text(m, std::string(text));
    if (text_tokens.empty()) {
        fprintf(stderr, "csm_tts: empty text after tokenization\n");
        return nullptr;
    }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "csm_tts: %zu text tokens:", text_tokens.size());
        for (size_t i = 0; i < text_tokens.size() && i < 20; i++) {
            fprintf(stderr, " %d", text_tokens[i]);
        }
        fprintf(stderr, "\n");
    }

    // 2. Build prompt frames
    // Each frame = (audio_num_codebooks audio tokens + 1 text token)
    // For the text prompt: audio tokens are padding, text tokens are real.
    // For the audio generation: text token is padding, audio tokens are predicted.

    int n_text_frames = (int)text_tokens.size();
    int max_audio_frames = hp.bb_max_pos - n_text_frames;
    if (ctx->params.max_audio_tokens > 0 && ctx->params.max_audio_tokens < max_audio_frames) {
        max_audio_frames = ctx->params.max_audio_tokens;
    }
    if (max_audio_frames <= 0) {
        fprintf(stderr, "csm_tts: text too long for context window\n");
        return nullptr;
    }

    // Collected audio codes: [n_generated_frames][audio_num_codebooks]
    std::vector<std::vector<int32_t>> all_codes;

    // -------------------------------------------------------------------
    // Backbone AR loop: generates codebook-0 tokens autoregressively.
    // -------------------------------------------------------------------
    // The backbone input at each frame is:
    //   sum of audio embeddings for all 32 codebooks (padded for text frames)
    //   + text embedding (padded for audio frames)
    //
    // During text-prompt frames: audio slots are padded, text is real.
    // During generation: text is the special audio_token_id, audio comes
    // from previously generated codebook tokens.
    // -------------------------------------------------------------------

    // Clear backbone KV cache
    ggml_backend_buffer_clear(ctx->bb_kv_buf, 0);

    int bb_d = (int)hp.bb_d_model;
    int avocab = (int)hp.audio_vocab_size;
    int n_cb = (int)hp.audio_num_codebooks;

    // Read backbone embedding tables to CPU for embedding lookups
    std::vector<float> audio_embd_data((size_t)bb_d * avocab * n_cb);
    tensor_get_f32(m.bb_audio_embd_w, audio_embd_data.data(), 0, audio_embd_data.size());
    std::vector<float> text_embd_data((size_t)bb_d * hp.text_vocab_size);
    tensor_get_f32(m.bb_text_embd_w, text_embd_data.data(), 0, text_embd_data.size());

    // Helper: embed a single token from the text vocab
    auto embed_text = [&](int32_t tok_id, float* out) {
        if (tok_id < 0 || tok_id >= (int)hp.text_vocab_size) {
            std::memset(out, 0, (size_t)bb_d * sizeof(float));
            return;
        }
        std::memcpy(out, &text_embd_data[(size_t)tok_id * bb_d], (size_t)bb_d * sizeof(float));
    };

    // Helper: embed a single token from audio codebook cb_idx
    auto embed_audio = [&](int cb_idx, int32_t tok_id, float* out) {
        if (tok_id < 0 || tok_id >= avocab) {
            std::memset(out, 0, (size_t)bb_d * sizeof(float));
            return;
        }
        size_t offset = ((size_t)cb_idx * avocab + tok_id) * bb_d;
        std::memcpy(out, &audio_embd_data[offset], (size_t)bb_d * sizeof(float));
    };

    // Build a text-prompt frame embedding: only text token is active (audio masked).
    // Original Python: text_frame[:, -1] = text_tokens; mask only text column = True.
    // embed = text_embedding only (audio slots are masked to zero).
    auto build_text_frame_embedding = [&](int32_t text_tok, float* out) { embed_text(text_tok, out); };

    // Build an audio-generation frame embedding: only audio tokens are active (text masked).
    // Original Python: audio slots = previous codebook tokens (masked True),
    //                   text slot = 0 (masked False → zeroed out).
    // embed = sum of audio embeddings for all 32 codebooks.
    auto build_audio_frame_embedding = [&](const int32_t* audio_tokens, float* out) {
        std::memset(out, 0, (size_t)bb_d * sizeof(float));
        std::vector<float> tmp(bb_d);
        for (int cb = 0; cb < n_cb; cb++) {
            embed_audio(cb, audio_tokens[cb], tmp.data());
            for (int d = 0; d < bb_d; d++) {
                out[d] += tmp[d];
            }
        }
    };

    // --- Build backbone transformer graph ---
    auto build_backbone_graph = [&](int n_past, int T) -> ggml_cgraph* {
        ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
        ggml_context* ctx0 = ggml_init(ip);
        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

        int Lk = n_past + T;
        ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, bb_d, T);
        ggml_set_name(embeds, "bb_embeds");
        ggml_set_input(embeds);
        ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
        ggml_set_name(positions, "bb_positions");
        ggml_set_input(positions);
        ggml_tensor* causal_mask = nullptr;
        if (T > 1) {
            causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
            ggml_set_name(causal_mask, "bb_mask");
            ggml_set_input(causal_mask);
        }

        const core_attn::KvSelfAttnParams kvp = {
            /*n_heads*/ (int)hp.bb_n_heads,
            /*n_kv_heads*/ (int)hp.bb_n_kv_heads,
            /*head_dim*/ (int)hp.bb_head_dim,
            /*n_kv_grp*/ (int)(hp.bb_n_heads / hp.bb_n_kv_heads),
            /*n_ctx_orig*/ (int)hp.bb_max_pos,
            /*rope_theta*/ hp.bb_rope_theta,
            /*rope_beta_fast*/ 0.0f,
            /*rope_beta_slow*/ 0.0f,
            /*attn_scale*/ 1.0f / std::sqrt((float)hp.bb_head_dim),
            /*qk_norm_eps*/ 0.0f,
            /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
        };

        ggml_tensor* cur = embeds;
        for (uint32_t il = 0; il < hp.bb_n_layers; il++) {
            const auto& blk = m.bb_layers[il];
            ggml_tensor* residual = cur;

            // Pre-attn RMSNorm
            ggml_tensor* x = ggml_rms_norm(ctx0, cur, hp.bb_rms_norm_eps);
            x = ggml_mul(ctx0, x, blk.attn_norm_w);

            // Self-attention with KV cache
            ggml_tensor* attn = core_attn::kv_self_attn(
                ctx0, gf, x, blk.attn_q_w, blk.attn_k_w, blk.attn_v_w, blk.attn_output_w, nullptr, nullptr, positions,
                (T == 1) ? nullptr : causal_mask, ctx->bb_kv_k, ctx->bb_kv_v, (int)il, n_past, kvp);
            cur = ggml_add(ctx0, residual, attn);

            // FFN
            residual = cur;
            x = ggml_rms_norm(ctx0, cur, hp.bb_rms_norm_eps);
            x = ggml_mul(ctx0, x, blk.ffn_norm_w);
            ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, blk.ffn_gate_w, blk.ffn_up_w, blk.ffn_down_w);
            cur = ggml_add(ctx0, residual, mlp);
        }

        // Final norm + codebook-0 head
        cur = ggml_rms_norm(ctx0, cur, hp.bb_rms_norm_eps);
        cur = ggml_mul(ctx0, cur, m.bb_output_norm_w);
        // Take only last position for logits
        if (T > 1) {
            cur = ggml_view_2d(ctx0, cur, bb_d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
        }
        // Also output the hidden state for depth decoder
        ggml_set_name(cur, "bb_hidden");
        ggml_build_forward_expand(gf, cur);

        ggml_tensor* logits = ggml_mul_mat(ctx0, m.bb_codebook0_head_w, cur);
        ggml_set_name(logits, "bb_logits");
        ggml_build_forward_expand(gf, logits);

        ggml_free(ctx0);
        return gf;
    };

    // --- Build depth decoder graph ---
    // codebook_idx: which codebook we're predicting (1..31). Selects the
    // correct slice of the 3D codebooks_head weight. Python's CsmCodebooksHead
    // uses weight[cache_position - 1] per position — each position has its OWN
    // linear head. We replicate that by extracting weight[codebook_idx - 1].
    // dd_n_past_for_graph: how many positions already in the depth KV cache.
    auto build_depth_graph = [&](int T, int codebook_idx = -1, int dd_n_past_for_graph = 0) -> ggml_cgraph* {
        ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
        ggml_context* ctx0 = ggml_init(ip);
        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

        int dd_d = (int)hp.dd_d_model;
        ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dd_d, T);
        ggml_set_name(embeds, "dd_embeds");
        ggml_set_input(embeds);
        ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
        ggml_set_name(positions, "dd_positions");
        ggml_set_input(positions);
        const int dd_Lk = dd_n_past_for_graph + T;
        ggml_tensor* causal_mask = nullptr;
        if (T > 1) {
            causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, dd_Lk, T);
            ggml_set_name(causal_mask, "dd_mask");
            ggml_set_input(causal_mask);
        }

        const core_attn::KvSelfAttnParams kvp = {
            /*n_heads*/ (int)hp.dd_n_heads,
            /*n_kv_heads*/ (int)hp.dd_n_kv_heads,
            /*head_dim*/ (int)hp.dd_head_dim,
            /*n_kv_grp*/ (int)(hp.dd_n_heads / hp.dd_n_kv_heads),
            /*n_ctx_orig*/ (int)(hp.audio_num_codebooks + 1),
            /*rope_theta*/ hp.bb_rope_theta,
            /*rope_beta_fast*/ 0.0f,
            /*rope_beta_slow*/ 0.0f,
            /*attn_scale*/ 1.0f / std::sqrt((float)hp.dd_head_dim),
            /*qk_norm_eps*/ 0.0f,
            /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
        };

        ggml_tensor* cur = embeds;
        for (uint32_t il = 0; il < hp.dd_n_layers; il++) {
            const auto& blk = m.dd_layers[il];
            ggml_tensor* residual = cur;

            // Pre-attn RMSNorm
            ggml_tensor* x = ggml_rms_norm(ctx0, cur, hp.bb_rms_norm_eps);
            x = ggml_mul(ctx0, x, blk.attn_norm_w);

            ggml_tensor* attn = core_attn::kv_self_attn(
                ctx0, gf, x, blk.attn_q_w, blk.attn_k_w, blk.attn_v_w, blk.attn_output_w, nullptr, nullptr, positions,
                (T == 1) ? nullptr : causal_mask, ctx->dd_kv_k, ctx->dd_kv_v, (int)il, dd_n_past_for_graph, kvp);
            cur = ggml_add(ctx0, residual, attn);

            // FFN
            residual = cur;
            x = ggml_rms_norm(ctx0, cur, hp.bb_rms_norm_eps);
            x = ggml_mul(ctx0, x, blk.ffn_norm_w);
            ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, blk.ffn_gate_w, blk.ffn_up_w, blk.ffn_down_w);
            cur = ggml_add(ctx0, residual, mlp);
        }

        // Final norm
        cur = ggml_rms_norm(ctx0, cur, hp.bb_rms_norm_eps);
        cur = ggml_mul(ctx0, cur, m.dd_output_norm_w);
        // Take last position
        if (T > 1) {
            cur = ggml_view_2d(ctx0, cur, dd_d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
        }
        // Project through the POSITION-SPECIFIC codebooks head.
        // Python CsmCodebooksHead: weight is [n_cb-1, hidden_size, vocab_size].
        // At each decode position k, it uses weight[k-1]:
        //   F.linear(hidden_states, weight[k].T) = hidden @ weight[k]
        //   where weight[k] is [hidden_size, vocab_size]
        //
        // In GGUF, dd_codebooks_head_w stored as 3D: ne[0]=vocab, ne[1]=hidden, ne[2]=n_cb-1
        // (Python shape [n_cb-1, hidden, vocab] maps to gguf ne=[vocab, hidden, n_cb-1])
        // A 2D slice at index k: ne[0]=vocab, ne[1]=hidden → that's already the
        // correct shape for ggml_mul_mat(slice, cur) since mul_mat needs
        // a->ne[0] == b->ne[0]: slice.ne[0]=vocab... NO wait.
        //
        // Actually: GGUF stores Python [31, 1024, 2051] as ne[0]=2051, ne[1]=1024, ne[2]=31.
        // A 2D slice: ne[0]=2051, ne[1]=1024.
        // For ggml_mul_mat(a, b): requires a->ne[0] == b->ne[0].
        // cur has ne[0]=dd_d=1024. Slice has ne[0]=2051. They don't match.
        // Need to transpose the slice: ne[0]=1024, ne[1]=2051.
        ggml_tensor* head_w = m.dd_codebooks_head_w;
        ggml_tensor* logits;
        if (codebook_idx > 0 && ggml_n_dims(head_w) == 3) {
            // Extract 2D slice for codebook (codebook_idx-1): shape [ne[0]=2051, ne[1]=1024]
            int64_t slice_offset = (int64_t)(codebook_idx - 1) * head_w->nb[2];
            ggml_tensor* head_slice =
                ggml_view_2d(ctx0, head_w, head_w->ne[0], head_w->ne[1], head_w->nb[1], slice_offset);
            // Transpose to [ne[0]=1024, ne[1]=2051] so mul_mat matches cur's ne[0]=1024
            // ggml_mul_mat requires non-transposed a, so cont after transpose
            head_slice = ggml_cont(ctx0, ggml_transpose(ctx0, head_slice));
            // Now: mul_mat([1024, 2051], [1024, 1]) -> [2051, 1] = logits
            logits = ggml_mul_mat(ctx0, head_slice, cur);
        } else {
            // Fallback for old behavior
            logits = ggml_mul_mat(ctx0, head_w, cur);
        }
        ggml_set_name(logits, "dd_logits");
        ggml_build_forward_expand(gf, logits);

        ggml_free(ctx0);
        return gf;
    };

    // Read projection weights to CPU for depth decoder input projection.
    // In CSM, the depth decoder has NO separate token embedding.
    // It reuses the backbone's audio_embeddings (dim=2048) and projects
    // ALL inputs (backbone hidden state + audio token embeddings) through
    // the same projection layer (2048 -> 1024).
    // Python: self.projection(curr_h) where curr_h is always backbone-dim.
    int dd_d = (int)hp.dd_d_model;
    std::vector<float> dd_proj_data((size_t)hp.dd_backbone_hidden * dd_d);
    tensor_get_f32(m.dd_projection_w, dd_proj_data.data(), 0, dd_proj_data.size());

    // Helper: project any backbone-dim vector (bb_d=2048) -> depth (dd_d=1024)
    // projection_w is [bb_d, dd_d] in GGUF (ne[0]=bb_d=2048, ne[1]=dd_d=1024)
    auto project_to_dd = [&](const float* input_bb_dim, float* dd_out) {
        int proj_ne0 = (int)m.dd_projection_w->ne[0]; // 2048
        int proj_ne1 = (int)m.dd_projection_w->ne[1]; // 1024
        for (int d = 0; d < proj_ne1; d++) {
            float sum = 0.0f;
            for (int k = 0; k < proj_ne0; k++) {
                sum += dd_proj_data[(size_t)d * proj_ne0 + k] * input_bb_dim[k];
            }
            dd_out[d] = sum;
        }
    };

    // Helper: embed audio token in backbone space (2048-dim) for depth decoder input.
    // Same as embed_audio — uses the shared backbone audio_embeddings.
    // Then caller must project_to_dd to get 1024-dim decoder input.
    // (This is what Python does: self.projection(self._embed_audio(cb, token)))
    auto embed_audio_for_depth = [&](int cb_idx, int32_t tok_id, float* out_bb_dim) {
        embed_audio(cb_idx, tok_id, out_bb_dim);
    };

    // Previous frame's codes (for embedding the next backbone step)
    std::vector<int32_t> prev_audio_codes(n_cb, 0);

    // --- Prefill: process text frames through backbone ---
    {
        // Build embeddings for all text frames (text-only: audio is masked out)
        std::vector<float> prefill_embeds((size_t)bb_d * n_text_frames);
        for (int i = 0; i < n_text_frames; i++) {
            build_text_frame_embedding(text_tokens[i], &prefill_embeds[(size_t)i * bb_d]);
        }

        // Build positions
        std::vector<int32_t> positions(n_text_frames);
        for (int i = 0; i < n_text_frames; i++) {
            positions[i] = i;
        }

        // Build causal mask
        int Lk = n_text_frames;
        std::vector<ggml_fp16_t> mask_data;
        if (n_text_frames > 1) {
            mask_data.assign((size_t)Lk * n_text_frames, ggml_fp32_to_fp16(0.0f));
            ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            for (int q = 0; q < n_text_frames; q++) {
                for (int k = q + 1; k < Lk; k++) {
                    mask_data[(size_t)q * Lk + k] = neg_inf;
                }
            }
        }

        ggml_cgraph* gf = build_backbone_graph(0, n_text_frames);
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            fprintf(stderr, "csm_tts: failed to alloc backbone prefill graph\n");
            return nullptr;
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "bb_embeds"), prefill_embeds.data(), 0,
                                prefill_embeds.size() * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "bb_positions"), positions.data(), 0,
                                positions.size() * sizeof(int32_t));
        if (n_text_frames > 1) {
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "bb_mask"), mask_data.data(), 0,
                                    mask_data.size() * sizeof(ggml_fp16_t));
        }
        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "csm_tts: backbone prefill compute failed\n");
            return nullptr;
        }

        // Read the prefill output: hidden state + c0 logits from last position.
        // In Python, the first frame's c0 comes from the prefill's last position.
        ggml_tensor* bb_hidden_t = ggml_graph_get_tensor(gf, "bb_hidden");
        ggml_tensor* bb_logits_t = ggml_graph_get_tensor(gf, "bb_logits");
        std::vector<float> bb_hidden(bb_d);
        std::vector<float> bb_logits(avocab);
        ggml_backend_tensor_get(bb_hidden_t, bb_hidden.data(), 0, (size_t)bb_d * sizeof(float));
        ggml_backend_tensor_get(bb_logits_t, bb_logits.data(), 0, (size_t)avocab * sizeof(float));

        // Debug: print prefill logits
        {
            float max_logit = bb_logits[0];
            int max_idx = 0;
            for (int i = 1; i < avocab; i++) {
                if (bb_logits[i] > max_logit) {
                    max_logit = bb_logits[i];
                    max_idx = i;
                }
            }
            if (ctx->params.verbosity >= 2) {
                fprintf(stderr, "csm_tts: prefill c0 logits: max=%.4f@%d\n", max_logit, max_idx);
            }
        }

        // Sample first frame's codebook-0 from the prefill output
        int32_t first_cb0 = sample_topk(bb_logits.data(), avocab, topk, temperature, ctx->rng_state);

        // Depth decode the first frame
        std::vector<int32_t> first_frame_codes(n_cb);
        first_frame_codes[0] = first_cb0;

        ggml_backend_buffer_clear(ctx->dd_kv_buf, 0);
        {
            std::vector<float> dd_init_embeds((size_t)dd_d * 2);
            project_to_dd(bb_hidden.data(), &dd_init_embeds[0]);
            std::vector<float> cb0_bb_embed(bb_d);
            embed_audio_for_depth(0, first_cb0, cb0_bb_embed.data());
            project_to_dd(cb0_bb_embed.data(), &dd_init_embeds[dd_d]);

            std::vector<int32_t> dd_positions = {0, 1};
            std::vector<ggml_fp16_t> dd_mask(2 * 2, ggml_fp32_to_fp16(0.0f));
            dd_mask[0 * 2 + 1] = ggml_fp32_to_fp16(-INFINITY);

            ggml_cgraph* dd_gf = build_depth_graph(2, 1, 0);
            ggml_backend_sched_reset(ctx->sched);
            if (!ggml_backend_sched_alloc_graph(ctx->sched, dd_gf)) {
                fprintf(stderr, "csm_tts: failed to alloc first depth prefill\n");
                return nullptr;
            }
            ggml_backend_tensor_set(ggml_graph_get_tensor(dd_gf, "dd_embeds"), dd_init_embeds.data(), 0,
                                    dd_init_embeds.size() * sizeof(float));
            ggml_backend_tensor_set(ggml_graph_get_tensor(dd_gf, "dd_positions"), dd_positions.data(), 0,
                                    dd_positions.size() * sizeof(int32_t));
            ggml_backend_tensor_set(ggml_graph_get_tensor(dd_gf, "dd_mask"), dd_mask.data(), 0,
                                    dd_mask.size() * sizeof(ggml_fp16_t));
            if (ggml_backend_sched_graph_compute(ctx->sched, dd_gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "csm_tts: first depth prefill compute failed\n");
                return nullptr;
            }
            ggml_tensor* dd_logits_t = ggml_graph_get_tensor(dd_gf, "dd_logits");
            std::vector<float> dd_logits(avocab);
            ggml_backend_tensor_get(dd_logits_t, dd_logits.data(), 0, dd_logits.size() * sizeof(float));
            first_frame_codes[1] = sample_topk(dd_logits.data(), avocab, topk, temperature, ctx->rng_state);
        }
        // Decode codebooks 2..31 for first frame
        {
            int dd_n_past = 2;
            for (int cb_idx = 2; cb_idx < n_cb; cb_idx++) {
                std::vector<float> prev_bb_embed(bb_d);
                embed_audio_for_depth(cb_idx - 1, first_frame_codes[cb_idx - 1], prev_bb_embed.data());
                std::vector<float> dd_step_embed(dd_d);
                project_to_dd(prev_bb_embed.data(), dd_step_embed.data());
                int32_t dd_pos = (int32_t)dd_n_past;

                ggml_cgraph* dd_gf = build_depth_graph(1, cb_idx, dd_n_past);
                ggml_backend_sched_reset(ctx->sched);
                if (!ggml_backend_sched_alloc_graph(ctx->sched, dd_gf))
                    break;
                ggml_backend_tensor_set(ggml_graph_get_tensor(dd_gf, "dd_embeds"), dd_step_embed.data(), 0,
                                        (size_t)dd_d * sizeof(float));
                ggml_backend_tensor_set(ggml_graph_get_tensor(dd_gf, "dd_positions"), &dd_pos, 0, sizeof(int32_t));
                if (ggml_backend_sched_graph_compute(ctx->sched, dd_gf) != GGML_STATUS_SUCCESS)
                    break;

                ggml_tensor* dd_logits_t = ggml_graph_get_tensor(dd_gf, "dd_logits");
                std::vector<float> dd_logits(avocab);
                ggml_backend_tensor_get(dd_logits_t, dd_logits.data(), 0, dd_logits.size() * sizeof(float));
                first_frame_codes[cb_idx] = sample_topk(dd_logits.data(), avocab, topk, temperature, ctx->rng_state);
                dd_n_past++;
            }
        }

        // Check if first frame is EOS
        bool first_eos = true;
        for (int q = 0; q < n_cb; q++) {
            if (first_frame_codes[q] != 0) {
                first_eos = false;
                break;
            }
        }
        if (!first_eos) {
            all_codes.push_back(first_frame_codes);
        }

        if (ctx->params.verbosity >= 2) {
            fprintf(stderr, "csm_tts: first frame cb0=%d\n", first_cb0);
        }

        prev_audio_codes = first_frame_codes;
    }

    if (all_codes.empty()) {
        // First frame was EOS — skip to Mimi decode (will fail gracefully)
        goto mimi_decode;
    }

    {
        int n_past_bb = n_text_frames;

        // --- AR decode loop: generate frames 1..N ---
        // Each iteration feeds the PREVIOUS frame's codes as input embedding.
        for (int frame_idx = 1; frame_idx < max_audio_frames; frame_idx++) {
            if (ctx->gen_frame_cap > 0 && (int)all_codes.size() >= ctx->gen_frame_cap) {
                break; // diagnostic frame cap (KV cache untouched)
            }
            // Build audio-frame embedding: sum of audio embeddings, text is masked out
            std::vector<float> step_embed(bb_d);
            build_audio_frame_embedding(prev_audio_codes.data(), step_embed.data());

            // Position
            int32_t pos = (int32_t)n_past_bb;

            // Run backbone single-step
            ggml_cgraph* gf = build_backbone_graph(n_past_bb, 1);
            ggml_backend_sched_reset(ctx->sched);
            if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
                fprintf(stderr, "csm_tts: failed to alloc backbone step graph\n");
                break;
            }
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "bb_embeds"), step_embed.data(), 0,
                                    (size_t)bb_d * sizeof(float));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "bb_positions"), &pos, 0, sizeof(int32_t));
            if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "csm_tts: backbone step %d compute failed\n", frame_idx);
                break;
            }

            // Read backbone hidden state and logits
            ggml_tensor* bb_hidden_t = ggml_graph_get_tensor(gf, "bb_hidden");
            ggml_tensor* bb_logits_t = ggml_graph_get_tensor(gf, "bb_logits");
            std::vector<float> bb_hidden(bb_d);
            std::vector<float> bb_logits(avocab);
            ggml_backend_tensor_get(bb_hidden_t, bb_hidden.data(), 0, (size_t)bb_d * sizeof(float));
            ggml_backend_tensor_get(bb_logits_t, bb_logits.data(), 0, (size_t)avocab * sizeof(float));

            n_past_bb++;

            // Debug: print early frame logits
            if (ctx->params.verbosity >= 2 && frame_idx < 3) {
                float max_logit = bb_logits[0];
                int max_idx = 0;
                for (int i = 1; i < avocab; i++) {
                    if (bb_logits[i] > max_logit) {
                        max_logit = bb_logits[i];
                        max_idx = i;
                    }
                }
                fprintf(stderr, "csm_tts: frame %d c0 logits: max=%.4f@%d\n", frame_idx, max_logit, max_idx);
            }

            // Sample codebook-0 token
            int32_t cb0_token = sample_topk(bb_logits.data(), avocab, topk, temperature, ctx->rng_state);

            // NOTE: EOS is checked AFTER generating all codebooks for the frame.
            // Python CSM stops when ALL codebooks are codebook_eos_token_id, not just cb0.
            // We check after the depth decoder fills all 32 codebooks.

            // --- Depth decoder: fill codebooks 1..31 ---
            std::vector<int32_t> frame_codes(n_cb);
            frame_codes[0] = cb0_token;

            // Clear depth KV cache for this frame
            ggml_backend_buffer_clear(ctx->dd_kv_buf, 0);

            // Depth decoder input sequence:
            //   Position 0: projected backbone hidden state
            //   Position 1: embedding of codebook-0 token
            //   Position 2..31: embeddings of codebook 1..30 as they're generated
            //
            // We process it incrementally: first prefill positions 0-1,
            // then decode positions 2..31 one at a time.

            // Build initial depth input: [projected_hidden, projected_cb0_embed]
            // Python: curr_h = [last_h, c0_embed] (both 2048-dim backbone space)
            //         self.projection(curr_h) -> both projected to 1024-dim decoder space
            std::vector<float> dd_init_embeds((size_t)dd_d * 2);
            project_to_dd(bb_hidden.data(), &dd_init_embeds[0]);
            // cb0 embedding in backbone space, then project to decoder space
            std::vector<float> cb0_bb_embed(bb_d);
            embed_audio_for_depth(0, cb0_token, cb0_bb_embed.data());
            project_to_dd(cb0_bb_embed.data(), &dd_init_embeds[dd_d]);

            // Prefill depth with 2 positions
            {
                std::vector<int32_t> dd_positions = {0, 1};
                std::vector<ggml_fp16_t> dd_mask(2 * 2, ggml_fp32_to_fp16(0.0f));
                ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
                // Position 0 can't see position 1
                dd_mask[0 * 2 + 1] = neg_inf;

                // Prefill with codebook_idx=1 (predicting codebook 1 from position 1), n_past=0
                ggml_cgraph* dd_gf = build_depth_graph(2, 1, 0);
                ggml_backend_sched_reset(ctx->sched);
                if (!ggml_backend_sched_alloc_graph(ctx->sched, dd_gf)) {
                    fprintf(stderr, "csm_tts: failed to alloc depth prefill graph\n");
                    break;
                }
                ggml_backend_tensor_set(ggml_graph_get_tensor(dd_gf, "dd_embeds"), dd_init_embeds.data(), 0,
                                        dd_init_embeds.size() * sizeof(float));
                ggml_backend_tensor_set(ggml_graph_get_tensor(dd_gf, "dd_positions"), dd_positions.data(), 0,
                                        dd_positions.size() * sizeof(int32_t));
                ggml_backend_tensor_set(ggml_graph_get_tensor(dd_gf, "dd_mask"), dd_mask.data(), 0,
                                        dd_mask.size() * sizeof(ggml_fp16_t));
                if (ggml_backend_sched_graph_compute(ctx->sched, dd_gf) != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "csm_tts: depth prefill compute failed\n");
                    break;
                }

                // Read logits: [avocab] — just codebook 1's logits from head[0]
                ggml_tensor* dd_logits_t = ggml_graph_get_tensor(dd_gf, "dd_logits");
                std::vector<float> dd_logits(avocab);
                ggml_backend_tensor_get(dd_logits_t, dd_logits.data(), 0, dd_logits.size() * sizeof(float));

                // Sample codebook-1 token
                int32_t cb1_token = sample_topk(dd_logits.data(), avocab, topk, temperature, ctx->rng_state);
                frame_codes[1] = cb1_token;
            }

            // Decode codebooks 2..31 one at a time
            int dd_n_past = 2;
            for (int cb_idx = 2; cb_idx < n_cb; cb_idx++) {
                // Input: embedding of previous codebook token in backbone space (2048),
                // then projected to decoder space (1024).
                // Python: prev_embed = self._embed_audio(cb_idx-1, token) -> 2048
                //         self.projection(prev_embed) -> 1024
                std::vector<float> prev_bb_embed(bb_d);
                embed_audio_for_depth(cb_idx - 1, frame_codes[cb_idx - 1], prev_bb_embed.data());
                std::vector<float> dd_step_embed(dd_d);
                project_to_dd(prev_bb_embed.data(), dd_step_embed.data());
                int32_t dd_pos = (int32_t)dd_n_past;

                // Pass codebook_idx so the graph uses the correct per-codebook head
                ggml_cgraph* dd_gf = build_depth_graph(1, cb_idx, dd_n_past);
                ggml_backend_sched_reset(ctx->sched);
                if (!ggml_backend_sched_alloc_graph(ctx->sched, dd_gf)) {
                    fprintf(stderr, "csm_tts: failed to alloc depth step graph (cb=%d)\n", cb_idx);
                    break;
                }
                ggml_backend_tensor_set(ggml_graph_get_tensor(dd_gf, "dd_embeds"), dd_step_embed.data(), 0,
                                        (size_t)dd_d * sizeof(float));
                ggml_backend_tensor_set(ggml_graph_get_tensor(dd_gf, "dd_positions"), &dd_pos, 0, sizeof(int32_t));
                if (ggml_backend_sched_graph_compute(ctx->sched, dd_gf) != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "csm_tts: depth step compute failed (cb=%d)\n", cb_idx);
                    break;
                }

                // Read logits — just [avocab] for this specific codebook
                ggml_tensor* dd_logits_t = ggml_graph_get_tensor(dd_gf, "dd_logits");
                std::vector<float> dd_logits(avocab);
                ggml_backend_tensor_get(dd_logits_t, dd_logits.data(), 0, dd_logits.size() * sizeof(float));

                int32_t cb_token = sample_topk(dd_logits.data(), avocab, topk, temperature, ctx->rng_state);
                frame_codes[cb_idx] = cb_token;
                dd_n_past++;
            }

            // Check EOS: Python uses torch.all(sample == 0) — ALL 32 codebooks are 0.
            bool all_eos = true;
            for (int q = 0; q < n_cb; q++) {
                if (frame_codes[q] != 0) {
                    all_eos = false;
                    break;
                }
            }
            if (all_eos) {
                if (ctx->params.verbosity >= 1) {
                    fprintf(stderr, "csm_tts: all-codebook EOS at frame %d\n", frame_idx);
                }
                break;
            }

            all_codes.push_back(frame_codes);
            prev_audio_codes = frame_codes;

            if (ctx->params.verbosity >= 2 && (frame_idx % 10 == 0)) {
                fprintf(stderr, "csm_tts: frame %d, cb0=%d\n", frame_idx, cb0_token);
            }
        }
    } // end AR decode block

mimi_decode:
    ctx->last_codes = all_codes; // capture before Mimi decode (which may fail)
    if (all_codes.empty()) {
        fprintf(stderr, "csm_tts: no audio frames generated\n");
        return nullptr;
    }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "csm_tts: generated %zu audio frames\n", all_codes.size());
    }

    // 3. Mimi decode: codes -> PCM
    int T_frames = (int)all_codes.size();

    // Flatten codes to [n_codebooks, T_frames] layout
    std::vector<int32_t> flat_codes(n_cb * T_frames);
    for (int t = 0; t < T_frames; t++) {
        for (int q = 0; q < n_cb; q++) {
            flat_codes[q * T_frames + t] = all_codes[t][q];
        }
    }

    // RVQ dequantize
    std::vector<float> continuous;
    if (!rvq_dequantize(ctx, flat_codes.data(), n_cb, T_frames, continuous)) {
        fprintf(stderr, "csm_tts: RVQ dequantize failed\n");
        return nullptr;
    }

    // Build Mimi decoder graph and compute
    // [mimi_dim, T_frames] -> upsample -> transformer -> SEANet -> PCM
    {
        int mdim = (int)hp.mimi_dim;

        struct ggml_init_params gp = {
            /*.mem_size   =*/ctx->compute_meta.size(),
            /*.mem_buffer =*/ctx->compute_meta.data(),
            /*.no_alloc   =*/true,
        };
        ggml_context* gctx = ggml_init(gp);
        if (!gctx) {
            fprintf(stderr, "csm_tts: failed to init compute context\n");
            return nullptr;
        }

        // Upsample: depthwise ConvTranspose1d(dim, dim, kernel=4, stride=2, groups=dim).
        // GGUF weight shape: ne[0]=kernel(4), ne[1]=out_per_group(1), ne[2]=groups(512).
        // This is a per-channel learned upsample (each of 512 channels has its own 1x4 kernel).
        // We implement it as a CPU operation since ggml doesn't have grouped transposed conv.
        ggml_tensor* x;
        {
            // Read upsample weight to CPU: (groups=512, 1, kernel=4) flat
            int kernel_size = m.mimi_upsample.w ? (int)m.mimi_upsample.w->ne[0] : 4;
            int groups = mdim; // depthwise: groups = channels = 512
            int stride = 2;    // compress = 2

            // inp is [mdim, T_frames] in column-major = T_frames columns of mdim elements
            // We need to upsample the time axis: T_frames -> stride * T_frames
            int T_up = stride * T_frames;

            // Read continuous data (already in memory as [T_frames, mdim] row-major from RVQ)
            // Convert to [mdim, T_frames] for processing
            std::vector<float> up_data(mdim * T_up, 0.0f);

            if (m.mimi_upsample.w) {
                // Read kernel weights: [kernel, 1, groups] in GGUF
                std::vector<float> kw(kernel_size * groups);
                tensor_get_f32(m.mimi_upsample.w, kw.data(), 0, kw.size());

                // Depthwise transposed conv1d: for each channel independently
                // output[ch, t_out] = sum_k(input[ch, t_in] * kernel[ch, k])
                //   where t_out = t_in * stride + k, for valid output range
                for (int ch = 0; ch < groups; ch++) {
                    for (int t_in = 0; t_in < T_frames; t_in++) {
                        float val = continuous[t_in * mdim + ch]; // input value
                        for (int k = 0; k < kernel_size; k++) {
                            int t_out = t_in * stride + k;
                            if (t_out < T_up) {
                                // kernel stored as [kernel, 1, groups], so element (k, 0, ch) = ch * kernel + k
                                up_data[t_out * mdim + ch] += val * kw[ch * kernel_size + k];
                            }
                        }
                    }
                }

                // Add bias if present
                if (m.mimi_upsample.b) {
                    std::vector<float> bias(groups);
                    tensor_get_f32(m.mimi_upsample.b, bias.data(), 0, groups);
                    for (int t = 0; t < T_up; t++) {
                        for (int ch = 0; ch < groups; ch++) {
                            up_data[t * mdim + ch] += bias[ch];
                        }
                    }
                }
            } else {
                // No upsample weight — fallback to repeat
                for (int t = 0; t < T_frames; t++) {
                    for (int s = 0; s < stride; s++) {
                        for (int d = 0; d < mdim; d++) {
                            up_data[(t * stride + s) * mdim + d] = continuous[t * mdim + d];
                        }
                    }
                }
            }

            // Create the upsampled tensor: [mdim, T_up]
            x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, mdim, T_up);
            ggml_set_name(x, "mimi_upsampled");
            ggml_set_input(x);

            // We'll set the data after graph allocation
            // For now, store T_up for later use
            T_frames = T_up; // update T_frames for downstream

            // Store up_data for later tensor_set (need to do after sched alloc)
            // Actually, we need to pre-compute this and set it as input data.
            // Let me restructure: compute upsample on CPU, then feed to ggml graph
            // for the transformer + SEANet parts.
            continuous = std::move(up_data);
        }

        // Transformer
        int mimi_head_dim = mdim / (int)hp.mimi_n_heads;
        x = build_mimi_dec_transformer(gctx, m.mimi_dec_layers, x, (int)hp.mimi_n_heads, mimi_head_dim);

        // SEANet decoder
        x = build_seanet_decoder(gctx, m.seanet_dec, x);

        ggml_set_output(x);
        ggml_set_name(x, "pcm_output");

        ggml_cgraph* graph = ggml_new_graph(gctx);
        ggml_build_forward_expand(graph, x);

        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, graph)) {
            fprintf(stderr, "csm_tts: failed to alloc Mimi decoder graph\n");
            ggml_free(gctx);
            return nullptr;
        }

        // Set upsampled input data
        ggml_tensor* inp = ggml_graph_get_tensor(graph, "mimi_upsampled");
        ggml_backend_tensor_set(inp, continuous.data(), 0, continuous.size() * sizeof(float));

        // Compute
        ggml_backend_sched_graph_compute(ctx->sched, graph);

        // Read output
        int n_pcm = (int)ggml_nelements(x);
        float* pcm = (float*)malloc(n_pcm * sizeof(float));
        ggml_backend_tensor_get(x, pcm, 0, n_pcm * sizeof(float));

        ggml_free(gctx);

        *out_n_samples = n_pcm;
        return pcm;
    }
}

// ===================================================================
// Diagnostic: backbone prefill per-layer dump (for crispasr-diff csm)
// ===================================================================

extern "C" int csm_tts_run_backbone_dump(struct csm_tts_context* ctx, const char* text, int max_T, float* layer0_normed,
                                         float** layer_outputs, int n_layers, float* last_h, float* c0_logits) {
    if (!ctx || !text) {
        return -1;
    }
    auto& m = ctx->model;
    auto& hp = m.hp;
    int bb_d = (int)hp.bb_d_model;
    int avocab = (int)hp.audio_vocab_size;

    std::vector<int32_t> tokens = tokenize_text(m, std::string(text));
    int T = (int)tokens.size();
    if (T <= 0) {
        return -2;
    }
    int Tw = (max_T > 0 && T > max_T) ? max_T : T; // rows actually written

    // Build text-only prefill embeddings (audio slots masked → text embed only).
    std::vector<float> text_embd_data((size_t)bb_d * hp.text_vocab_size);
    tensor_get_f32(m.bb_text_embd_w, text_embd_data.data(), 0, text_embd_data.size());
    std::vector<float> embeds((size_t)bb_d * T, 0.0f);
    for (int i = 0; i < T; i++) {
        int32_t tok = tokens[i];
        if (tok >= 0 && tok < (int)hp.text_vocab_size) {
            std::memcpy(&embeds[(size_t)i * bb_d], &text_embd_data[(size_t)tok * bb_d], (size_t)bb_d * sizeof(float));
        }
    }

    // Positions + causal mask (prefill, n_past=0).
    std::vector<int32_t> positions(T);
    for (int i = 0; i < T; i++) {
        positions[i] = i;
    }
    std::vector<ggml_fp16_t> mask_data;
    if (T > 1) {
        mask_data.assign((size_t)T * T, ggml_fp32_to_fp16(0.0f));
        ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < T; q++) {
            for (int k = q + 1; k < T; k++) {
                mask_data[(size_t)q * T + k] = neg_inf;
            }
        }
    }

    // --- Build graph mirroring build_backbone_graph, but tag every layer ---
    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* in_embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, bb_d, T);
    ggml_set_name(in_embeds, "bb_embeds");
    ggml_set_input(in_embeds);
    ggml_tensor* in_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(in_pos, "bb_positions");
    ggml_set_input(in_pos);
    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T, T);
        ggml_set_name(causal_mask, "bb_mask");
        ggml_set_input(causal_mask);
    }

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ (int)hp.bb_n_heads,
        /*n_kv_heads*/ (int)hp.bb_n_kv_heads,
        /*head_dim*/ (int)hp.bb_head_dim,
        /*n_kv_grp*/ (int)(hp.bb_n_heads / hp.bb_n_kv_heads),
        /*n_ctx_orig*/ (int)hp.bb_max_pos,
        /*rope_theta*/ hp.bb_rope_theta,
        /*rope_beta_fast*/ 0.0f,
        /*rope_beta_slow*/ 0.0f,
        /*attn_scale*/ 1.0f / std::sqrt((float)hp.bb_head_dim),
        /*qk_norm_eps*/ 0.0f,
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
    };

    ggml_backend_buffer_clear(ctx->bb_kv_buf, 0);

    ggml_tensor* cur = in_embeds;
    for (uint32_t il = 0; il < hp.bb_n_layers; il++) {
        const auto& blk = m.bb_layers[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, hp.bb_rms_norm_eps);
        x = ggml_mul(ctx0, x, blk.attn_norm_w);
        if (il == 0) {
            ggml_set_name(x, "dump_l0_normed");
            ggml_set_output(x);
            ggml_build_forward_expand(gf, x);
        }

        ggml_tensor* attn = core_attn::kv_self_attn(
            ctx0, gf, x, blk.attn_q_w, blk.attn_k_w, blk.attn_v_w, blk.attn_output_w, nullptr, nullptr, in_pos,
            (T == 1) ? nullptr : causal_mask, ctx->bb_kv_k, ctx->bb_kv_v, (int)il, 0, kvp);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        x = ggml_rms_norm(ctx0, cur, hp.bb_rms_norm_eps);
        x = ggml_mul(ctx0, x, blk.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, blk.ffn_gate_w, blk.ffn_up_w, blk.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);

        char nm[32];
        std::snprintf(nm, sizeof(nm), "dump_layer_%u", il);
        ggml_set_name(cur, nm);
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
    }

    // Final norm + last-position hidden + c0 logits.
    ggml_tensor* normed = ggml_rms_norm(ctx0, cur, hp.bb_rms_norm_eps);
    normed = ggml_mul(ctx0, normed, m.bb_output_norm_w);
    ggml_tensor* lh = normed;
    if (T > 1) {
        lh = ggml_view_2d(ctx0, normed, bb_d, 1, normed->nb[1], (size_t)(T - 1) * normed->nb[1]);
        lh = ggml_cont(ctx0, lh);
    }
    ggml_set_name(lh, "dump_last_h");
    ggml_set_output(lh);
    ggml_build_forward_expand(gf, lh);

    ggml_tensor* logits = ggml_mul_mat(ctx0, m.bb_codebook0_head_w, lh);
    ggml_set_name(logits, "dump_c0_logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "csm_tts: backbone dump alloc failed\n");
        ggml_free(ctx0);
        return -3;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "bb_embeds"), embeds.data(), 0, embeds.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "bb_positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));
    if (T > 1) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "bb_mask"), mask_data.data(), 0,
                                mask_data.size() * sizeof(ggml_fp16_t));
    }
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "csm_tts: backbone dump compute failed\n");
        ggml_free(ctx0);
        return -4;
    }

    // Read back tagged tensors. Per-layer tensors are [bb_d, T] → row-major
    // gives T rows of bb_d, matching the reference (T, d_model) layout.
    if (layer0_normed) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, "dump_l0_normed");
        ggml_backend_tensor_get(t, layer0_normed, 0, (size_t)Tw * bb_d * sizeof(float));
    }
    if (layer_outputs) {
        for (int il = 0; il < (int)hp.bb_n_layers && il < n_layers; il++) {
            if (!layer_outputs[il]) {
                continue;
            }
            char nm[32];
            std::snprintf(nm, sizeof(nm), "dump_layer_%d", il);
            ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
            ggml_backend_tensor_get(t, layer_outputs[il], 0, (size_t)Tw * bb_d * sizeof(float));
        }
    }
    if (last_h) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, "dump_last_h");
        ggml_backend_tensor_get(t, last_h, 0, (size_t)bb_d * sizeof(float));
    }
    if (c0_logits) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, "dump_c0_logits");
        ggml_backend_tensor_get(t, c0_logits, 0, (size_t)avocab * sizeof(float));
    }

    ggml_free(ctx0);
    return T;
}

extern "C" int csm_tts_run_depth_dump(struct csm_tts_context* ctx, const float* last_h, int c0_token,
                                      float* initial_proj, float* c1_logits) {
    if (!ctx || !last_h) {
        return -1;
    }
    auto& m = ctx->model;
    auto& hp = m.hp;
    int bb_d = (int)hp.bb_d_model;
    int dd_d = (int)hp.dd_d_model;
    int avocab = (int)hp.audio_vocab_size;

    // projection (bb_d -> dd_d) and audio-embedding row for codebook 0, token c0.
    std::vector<float> dd_proj_data((size_t)m.dd_projection_w->ne[0] * m.dd_projection_w->ne[1]);
    tensor_get_f32(m.dd_projection_w, dd_proj_data.data(), 0, dd_proj_data.size());
    int proj_ne0 = (int)m.dd_projection_w->ne[0]; // bb_d
    int proj_ne1 = (int)m.dd_projection_w->ne[1]; // dd_d
    auto project_to_dd = [&](const float* in_bb, float* out_dd) {
        for (int d = 0; d < proj_ne1; d++) {
            float s = 0.0f;
            for (int k = 0; k < proj_ne0; k++) {
                s += dd_proj_data[(size_t)d * proj_ne0 + k] * in_bb[k];
            }
            out_dd[d] = s;
        }
    };

    // c0 embedding in backbone space: audio_embeddings[0*avocab + c0_token].
    std::vector<float> c0_bb_embed(bb_d, 0.0f);
    if (c0_token >= 0 && c0_token < avocab) {
        size_t off_bytes = ((size_t)0 * avocab + c0_token) * bb_d * sizeof(float);
        tensor_get_f32(m.bb_audio_embd_w, c0_bb_embed.data(), off_bytes, bb_d);
    }

    // depth input = projection([last_h, c0_embed]) -> (2, dd_d), row-major.
    std::vector<float> dd_init((size_t)dd_d * 2);
    project_to_dd(last_h, &dd_init[0]);
    project_to_dd(c0_bb_embed.data(), &dd_init[dd_d]);
    if (initial_proj) {
        std::memcpy(initial_proj, dd_init.data(), (size_t)dd_d * 2 * sizeof(float));
    }

    // Build the depth graph (T=2, predicting codebook 1, n_past=0), mirroring
    // build_depth_graph in the synth path.
    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    const int T = 2;
    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dd_d, T);
    ggml_set_name(embeds, "dd_embeds");
    ggml_set_input(embeds);
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "dd_positions");
    ggml_set_input(positions);
    ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T, T);
    ggml_set_name(causal_mask, "dd_mask");
    ggml_set_input(causal_mask);

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ (int)hp.dd_n_heads,
        /*n_kv_heads*/ (int)hp.dd_n_kv_heads,
        /*head_dim*/ (int)hp.dd_head_dim,
        /*n_kv_grp*/ (int)(hp.dd_n_heads / hp.dd_n_kv_heads),
        /*n_ctx_orig*/ (int)(hp.audio_num_codebooks + 1),
        /*rope_theta*/ hp.bb_rope_theta,
        /*rope_beta_fast*/ 0.0f,
        /*rope_beta_slow*/ 0.0f,
        /*attn_scale*/ 1.0f / std::sqrt((float)hp.dd_head_dim),
        /*qk_norm_eps*/ 0.0f,
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
    };

    ggml_backend_buffer_clear(ctx->dd_kv_buf, 0);

    ggml_tensor* cur = embeds;
    for (uint32_t il = 0; il < hp.dd_n_layers; il++) {
        const auto& blk = m.dd_layers[il];
        ggml_tensor* residual = cur;
        ggml_tensor* x = ggml_rms_norm(ctx0, cur, hp.bb_rms_norm_eps);
        x = ggml_mul(ctx0, x, blk.attn_norm_w);
        ggml_tensor* attn =
            core_attn::kv_self_attn(ctx0, gf, x, blk.attn_q_w, blk.attn_k_w, blk.attn_v_w, blk.attn_output_w, nullptr,
                                    nullptr, positions, causal_mask, ctx->dd_kv_k, ctx->dd_kv_v, (int)il, 0, kvp);
        cur = ggml_add(ctx0, residual, attn);
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, hp.bb_rms_norm_eps);
        x = ggml_mul(ctx0, x, blk.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, blk.ffn_gate_w, blk.ffn_up_w, blk.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }
    cur = ggml_rms_norm(ctx0, cur, hp.bb_rms_norm_eps);
    cur = ggml_mul(ctx0, cur, m.dd_output_norm_w);
    cur = ggml_view_2d(ctx0, cur, dd_d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]); // last position

    // codebooks_head[0] for codebook 1: 3D ne=[vocab, dd_d, n_cb-1], slice 0.
    ggml_tensor* head_w = m.dd_codebooks_head_w;
    ggml_tensor* logits;
    if (ggml_n_dims(head_w) == 3) {
        ggml_tensor* slice = ggml_view_2d(ctx0, head_w, head_w->ne[0], head_w->ne[1], head_w->nb[1], 0);
        slice = ggml_cont(ctx0, ggml_transpose(ctx0, slice));
        logits = ggml_mul_mat(ctx0, slice, cur);
    } else {
        logits = ggml_mul_mat(ctx0, head_w, cur);
    }
    ggml_set_name(logits, "dd_logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    std::vector<int32_t> dd_pos = {0, 1};
    std::vector<ggml_fp16_t> dd_mask(4, ggml_fp32_to_fp16(0.0f));
    dd_mask[0 * 2 + 1] = ggml_fp32_to_fp16(-INFINITY); // position 0 cannot see position 1

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        ggml_free(ctx0);
        return -3;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dd_embeds"), dd_init.data(), 0, dd_init.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dd_positions"), dd_pos.data(), 0,
                            dd_pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dd_mask"), dd_mask.data(), 0,
                            dd_mask.size() * sizeof(ggml_fp16_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx0);
        return -4;
    }
    if (c1_logits) {
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "dd_logits"), c1_logits, 0, (size_t)avocab * sizeof(float));
    }
    ggml_free(ctx0);
    return 0;
}

extern "C" int csm_tts_run_generate_codes(struct csm_tts_context* ctx, const char* text, int32_t* out_codes,
                                          int max_frames_cap) {
    if (!ctx || !text) {
        return -1;
    }
    ctx->last_codes.clear();
    float saved_t = ctx->params.temperature;
    int saved_cap = ctx->gen_frame_cap;
    ctx->params.temperature = 0.0f; // greedy
    ctx->gen_frame_cap = (max_frames_cap > 0) ? max_frames_cap : 0;
    int n = 0;
    float* pcm = csm_tts_synthesize_with_reference(ctx, text, nullptr, 0, nullptr, &n);
    if (pcm) {
        free(pcm);
    }
    ctx->params.temperature = saved_t;
    ctx->gen_frame_cap = saved_cap;

    int nf = (int)ctx->last_codes.size();
    int n_cb = (int)ctx->model.hp.audio_num_codebooks;
    int w = (max_frames_cap > 0 && nf > max_frames_cap) ? max_frames_cap : nf;
    if (out_codes) {
        for (int t = 0; t < w; t++) {
            for (int q = 0; q < n_cb; q++) {
                out_codes[(size_t)t * n_cb + q] =
                    (t < nf && q < (int)ctx->last_codes[t].size()) ? ctx->last_codes[t][q] : 0;
            }
        }
    }
    return nf;
}

extern "C" int csm_tts_run_mimi_dump(struct csm_tts_context* ctx, const int32_t* codes_T_cb, int T, int n_cb,
                                     float* rvq_out) {
    if (!ctx || !codes_T_cb || T <= 0 || n_cb <= 0) {
        return -1;
    }
    int mdim = (int)ctx->model.hp.mimi_dim;
    // ref codes are [T, n_cb] row-major; rvq_dequantize wants [n_cb, T].
    std::vector<int32_t> flat((size_t)n_cb * T);
    for (int t = 0; t < T; t++) {
        for (int q = 0; q < n_cb; q++) {
            flat[(size_t)q * T + t] = codes_T_cb[(size_t)t * n_cb + q];
        }
    }
    std::vector<float> cont;
    if (!rvq_dequantize(ctx, flat.data(), n_cb, T, cont)) {
        return -2;
    }
    if (rvq_out) {
        std::memcpy(rvq_out, cont.data(), (size_t)T * mdim * sizeof(float));
    }
    return 0;
}

extern "C" int csm_tts_diag_synth_wav(struct csm_tts_context* ctx, const char* text, const char* wav_path,
                                      float temperature, int frame_cap) {
    if (!ctx || !text || !wav_path) {
        return -1;
    }
    float saved_t = ctx->params.temperature;
    int saved_cap = ctx->gen_frame_cap;
    ctx->params.temperature = temperature;
    ctx->gen_frame_cap = frame_cap;
    int n = 0;
    float* pcm = csm_tts_synthesize_with_reference(ctx, text, nullptr, 0, nullptr, &n);
    ctx->params.temperature = saved_t;
    ctx->gen_frame_cap = saved_cap;
    if (!pcm || n <= 0) {
        return -2;
    }

    // Peak-normalise to 0.95 to avoid clipping on write.
    float peak = 1e-6f;
    for (int i = 0; i < n; i++) {
        float a = std::fabs(pcm[i]);
        if (a > peak) {
            peak = a;
        }
    }
    float scale = 0.95f / peak;

    FILE* f = std::fopen(wav_path, "wb");
    if (!f) {
        free(pcm);
        return -3;
    }
    const uint32_t sr = 24000, nch = 1, bps = 16;
    const uint32_t byte_rate = sr * nch * bps / 8;
    const uint32_t data_bytes = (uint32_t)n * nch * bps / 8;
    const uint32_t riff = 36 + data_bytes;
    const uint16_t block_align = nch * bps / 8, fmt_pcm = 1, audio_fmt = 1;
    const uint32_t fmt_size = 16;
    std::fwrite("RIFF", 1, 4, f);
    std::fwrite(&riff, 4, 1, f);
    std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f);
    std::fwrite(&fmt_size, 4, 1, f);
    std::fwrite(&audio_fmt, 2, 1, f);
    std::fwrite(&nch, 2, 1, f); // nch is u32; write low 2 bytes
    std::fwrite(&sr, 4, 1, f);
    std::fwrite(&byte_rate, 4, 1, f);
    std::fwrite(&block_align, 2, 1, f);
    std::fwrite(&bps, 2, 1, f); // bps is u32; write low 2 bytes
    std::fwrite("data", 1, 4, f);
    std::fwrite(&data_bytes, 4, 1, f);
    (void)fmt_pcm;
    for (int i = 0; i < n; i++) {
        float v = pcm[i] * scale;
        if (v > 1.0f)
            v = 1.0f;
        if (v < -1.0f)
            v = -1.0f;
        int16_t s = (int16_t)lrintf(v * 32767.0f);
        std::fwrite(&s, 2, 1, f);
    }
    std::fclose(f);
    free(pcm);
    return n;
}
