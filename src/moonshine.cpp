#include "moonshine.h"
#include "moonshine-impl.h"
#include "moonshine-tokenizer.h"

#include "core/beam_decode.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct moonshine_context {
    moonshine_model model;
    moonshine_tokenizer tokenizer;
    ggml_backend_t backend = nullptr;
    std::string result_text;
    std::string scratch_token_text; // backs `moonshine_token_text` return value

    // Decoder state
    moonshine_kv_cache kv_self;
    moonshine_kv_cache kv_cross;
    std::vector<float> encoder_out;
    int enc_len = 0;

    int n_threads = 4;
    float temperature = 0.0f; // 0 = greedy argmax; > 0 = multinomial sampling
    int beam_size = 1;        // 1 = greedy/sampled; >1 = beam search (deterministic)
    // Sticky per-call seed override for best-of-N. 0 = derive deterministically
    // from the input audio buffer (the historical default — repeated calls
    // with the same input give identical samples). Non-zero values let the
    // caller inject run-index salt to draw N independent samples from the
    // same audio.
    uint64_t seed_override = 0;
    moonshine_timing timing = {};
};

// helper: get tensor by name, track failures
static struct ggml_tensor* checked_get_tensor(struct ggml_context* ctx, const char* name, bool& ok) {
    struct ggml_tensor* t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "%s: tensor '%s' not found\n", __func__, name);
        ok = false;
    }
    return t;
}

// helper: read uint32 from GGUF KV
static uint32_t gguf_get_u32(struct gguf_context* ctx, const char* key) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        fprintf(stderr, "warning: GGUF key '%s' not found, using 0\n", key);
        return 0;
    }
    return gguf_get_val_u32(ctx, id);
}

// helper: read float32 from GGUF KV
static float gguf_get_f32(struct gguf_context* ctx, const char* key) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        fprintf(stderr, "warning: GGUF key '%s' not found, using 0\n", key);
        return 0.0f;
    }
    return gguf_get_val_f32(ctx, id);
}

// helper: extract directory from file path
static std::string dir_of(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return ".";
    }
    return path.substr(0, pos);
}

struct moonshine_context* moonshine_init(const char* model_path) {
    struct moonshine_init_params params = {};
    params.model_path = model_path;
    params.tokenizer_path = nullptr;
    params.n_threads = 0;
    return moonshine_init_with_params(params);
}

struct moonshine_context* moonshine_init_with_params(struct moonshine_init_params params) {
    const char* model_path = params.model_path;
    auto* ctx = new moonshine_context();
    auto& model = ctx->model;

    ctx->n_threads = (params.n_threads > 0) ? params.n_threads : 4;

    // 1. Open GGUF file (no_alloc: create tensor metadata only)
    struct gguf_init_params gguf_params = {
        /*.no_alloc =*/true,
        /*.ctx      =*/&model.ctx_w,
    };

    struct gguf_context* ctx_gguf = gguf_init_from_file(model_path, gguf_params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s: failed to open GGUF file '%s'\n", __func__, model_path);
        delete ctx;
        return nullptr;
    }

    // 2. Read hyperparameters from GGUF KV pairs
    auto& hp = model.hparams;
    hp.enc_hidden_size = gguf_get_u32(ctx_gguf, "moonshine.encoder.embedding_length");
    hp.enc_n_layers = gguf_get_u32(ctx_gguf, "moonshine.encoder.block_count");
    hp.n_heads = gguf_get_u32(ctx_gguf, "moonshine.encoder.attention.head_count");
    hp.n_kv_heads = gguf_get_u32(ctx_gguf, "moonshine.encoder.attention.head_count_kv");
    hp.enc_intermediate = gguf_get_u32(ctx_gguf, "moonshine.encoder.feed_forward_length");
    hp.dec_n_layers = gguf_get_u32(ctx_gguf, "moonshine.decoder.block_count");
    hp.dec_intermediate = gguf_get_u32(ctx_gguf, "moonshine.decoder.feed_forward_length");
    hp.vocab_size = gguf_get_u32(ctx_gguf, "moonshine.vocab_size");
    hp.bos_token_id = gguf_get_u32(ctx_gguf, "moonshine.bos_token_id");
    hp.eos_token_id = gguf_get_u32(ctx_gguf, "moonshine.eos_token_id");
    hp.layer_norm_eps = gguf_get_f32(ctx_gguf, "moonshine.attention.layer_norm_epsilon");
    hp.rope_theta = gguf_get_f32(ctx_gguf, "moonshine.rope.freq_base");
    hp.partial_rotary_factor = gguf_get_f32(ctx_gguf, "moonshine.encoder.partial_rotary_factor");
    hp.conv1_kernel_size = gguf_get_u32(ctx_gguf, "moonshine.encoder.conv1.kernel_size");
    hp.conv1_stride = gguf_get_u32(ctx_gguf, "moonshine.encoder.conv1.stride");
    hp.conv2_kernel_size = gguf_get_u32(ctx_gguf, "moonshine.encoder.conv2.kernel_size");
    hp.conv2_stride = gguf_get_u32(ctx_gguf, "moonshine.encoder.conv2.stride");
    hp.conv3_kernel_size = gguf_get_u32(ctx_gguf, "moonshine.encoder.conv3.kernel_size");
    hp.conv3_stride = gguf_get_u32(ctx_gguf, "moonshine.encoder.conv3.stride");

    // validate critical hparams
    if (hp.enc_hidden_size == 0 || hp.n_heads == 0 || hp.enc_n_layers == 0 || hp.dec_n_layers == 0) {
        fprintf(stderr, "%s: invalid model hparams (hidden=%u heads=%u enc_layers=%u dec_layers=%u)\n", __func__,
                hp.enc_hidden_size, hp.n_heads, hp.enc_n_layers, hp.dec_n_layers);
        gguf_free(ctx_gguf);
        delete ctx;
        return nullptr;
    }

    // derived
    hp.head_dim = hp.enc_hidden_size / hp.n_heads;
    hp.rotary_dim = (uint32_t)(hp.head_dim * hp.partial_rotary_factor);

    // 3. Allocate a single CPU buffer for all tensor data
    model.buf_w = ggml_backend_alloc_ctx_tensors_from_buft(model.ctx_w, ggml_backend_cpu_buffer_type());
    if (!model.buf_w) {
        fprintf(stderr, "%s: failed to allocate tensor buffer\n", __func__);
        gguf_free(ctx_gguf);
        delete ctx;
        return nullptr;
    }

    // 4. Load tensor data from GGUF file
    FILE* f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__, model_path);
        gguf_free(ctx_gguf);
        delete ctx;
        return nullptr;
    }

    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    const size_t data_offset = gguf_get_data_offset(ctx_gguf);
    std::vector<uint8_t> read_buf;

    for (int64_t i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor* tensor = ggml_get_tensor(model.ctx_w, name);
        if (!tensor) {
            fprintf(stderr, "%s: tensor '%s' not found in context\n", __func__, name);
            fclose(f);
            gguf_free(ctx_gguf);
            delete ctx;
            return nullptr;
        }

        const size_t nbytes = ggml_nbytes(tensor);
        const size_t offs = data_offset + gguf_get_tensor_offset(ctx_gguf, i);
        read_buf.resize(nbytes);

        if (fseek(f, offs, SEEK_SET) != 0) {
            fprintf(stderr, "%s: fseek failed for tensor '%s'\n", __func__, name);
            fclose(f);
            gguf_free(ctx_gguf);
            delete ctx;
            return nullptr;
        }

        if (fread(read_buf.data(), 1, nbytes, f) != nbytes) {
            fprintf(stderr, "%s: fread failed for tensor '%s'\n", __func__, name);
            fclose(f);
            gguf_free(ctx_gguf);
            delete ctx;
            return nullptr;
        }

        ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);
    }

    fclose(f);
    gguf_free(ctx_gguf);

    // 5. Map tensors to model struct fields
    bool ok = true;

    // Encoder conv stem
    model.enc_conv1_w = checked_get_tensor(model.ctx_w, "encoder.conv1.weight", ok);
    model.enc_groupnorm_w = checked_get_tensor(model.ctx_w, "encoder.groupnorm.weight", ok);
    model.enc_groupnorm_b = checked_get_tensor(model.ctx_w, "encoder.groupnorm.bias", ok);
    model.enc_conv2_w = checked_get_tensor(model.ctx_w, "encoder.conv2.weight", ok);
    model.enc_conv2_b = checked_get_tensor(model.ctx_w, "encoder.conv2.bias", ok);
    model.enc_conv3_w = checked_get_tensor(model.ctx_w, "encoder.conv3.weight", ok);
    model.enc_conv3_b = checked_get_tensor(model.ctx_w, "encoder.conv3.bias", ok);

    // Encoder output norm
    model.enc_output_norm = checked_get_tensor(model.ctx_w, "encoder.output_norm.weight", ok);

    // Encoder layers
    model.enc_layers.resize(hp.enc_n_layers);
    for (int i = 0; (uint32_t)i < hp.enc_n_layers; i++) {
        auto& layer = model.enc_layers[i];
        char name[128];

        snprintf(name, sizeof(name), "encoder.layers.%d.attn_norm.weight", i);
        layer.attn_norm = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "encoder.layers.%d.attn.q.weight", i);
        layer.attn_q = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "encoder.layers.%d.attn.k.weight", i);
        layer.attn_k = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "encoder.layers.%d.attn.v.weight", i);
        layer.attn_v = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "encoder.layers.%d.attn.o.weight", i);
        layer.attn_o = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "encoder.layers.%d.ffn_norm.weight", i);
        layer.ffn_norm = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "encoder.layers.%d.ffn.fc1.weight", i);
        layer.ffn_fc1_w = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "encoder.layers.%d.ffn.fc1.bias", i);
        layer.ffn_fc1_b = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "encoder.layers.%d.ffn.fc2.weight", i);
        layer.ffn_fc2_w = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "encoder.layers.%d.ffn.fc2.bias", i);
        layer.ffn_fc2_b = checked_get_tensor(model.ctx_w, name, ok);
    }

    // Decoder
    model.dec_embed = checked_get_tensor(model.ctx_w, "decoder.embed_tokens.weight", ok);
    model.dec_output_norm = checked_get_tensor(model.ctx_w, "decoder.output_norm.weight", ok);
    // Weight tying: use decoder.output.weight if present, else share with embed
    model.dec_output = ggml_get_tensor(model.ctx_w, "decoder.output.weight");
    if (!model.dec_output) {
        model.dec_output = model.dec_embed;
    }

    // Decoder layers
    model.dec_layers.resize(hp.dec_n_layers);
    for (int i = 0; (uint32_t)i < hp.dec_n_layers; i++) {
        auto& layer = model.dec_layers[i];
        char name[128];

        snprintf(name, sizeof(name), "decoder.layers.%d.attn_norm.weight", i);
        layer.attn_norm = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "decoder.layers.%d.attn.q.weight", i);
        layer.attn_q = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "decoder.layers.%d.attn.k.weight", i);
        layer.attn_k = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "decoder.layers.%d.attn.v.weight", i);
        layer.attn_v = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "decoder.layers.%d.attn.o.weight", i);
        layer.attn_o = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "decoder.layers.%d.cross_attn_norm.weight", i);
        layer.cross_attn_norm = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "decoder.layers.%d.cross_attn.q.weight", i);
        layer.cross_attn_q = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "decoder.layers.%d.cross_attn.k.weight", i);
        layer.cross_attn_k = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "decoder.layers.%d.cross_attn.v.weight", i);
        layer.cross_attn_v = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "decoder.layers.%d.cross_attn.o.weight", i);
        layer.cross_attn_o = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "decoder.layers.%d.ffn_norm.weight", i);
        layer.ffn_norm = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "decoder.layers.%d.ffn.fc1.weight", i);
        layer.ffn_fc1_w = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "decoder.layers.%d.ffn.fc1.bias", i);
        layer.ffn_fc1_b = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "decoder.layers.%d.ffn.fc2.weight", i);
        layer.ffn_fc2_w = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "decoder.layers.%d.ffn.fc2.bias", i);
        layer.ffn_fc2_b = checked_get_tensor(model.ctx_w, name, ok);
    }

    if (!ok) {
        fprintf(stderr, "%s: one or more tensors missing from model\n", __func__);
        delete ctx;
        return nullptr;
    }

    // 6. Load tokenizer
    std::string tokenizer_path;
    if (params.tokenizer_path) {
        tokenizer_path = params.tokenizer_path;
    } else {
        tokenizer_path = dir_of(model_path) + "/tokenizer.bin";
    }
    if (!ctx->tokenizer.load(tokenizer_path.c_str())) {
        fprintf(stderr, "%s: failed to load tokenizer from '%s'\n", __func__, tokenizer_path.c_str());
        delete ctx;
        return nullptr;
    }

    // 7. Initialize backend.
    // Force CPU: weights are allocated on CPU buffer (line 138) and the graph
    // uses ggml_backend_graph_compute (no scheduler). GPU would crash because
    // it can't access CPU-resident weight tensors without a scheduler.
    // The model is tiny (27M params) so GPU overhead exceeds compute benefit.
    ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) {
        fprintf(stderr, "%s: failed to init CPU backend\n", __func__);
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    return ctx;
}

// conv_1d using F32 im2col. Kernel can be any quantized type (F16, Q4_K, etc.)
// since it goes into src0 of ggml_mul_mat; im2col (F32) goes into src1.
static struct ggml_tensor* conv_1d_f32(struct ggml_context* ctx0,
                                       struct ggml_tensor* kernel, // [K, IC, OC]
                                       struct ggml_tensor* input,  // [IL, IC, N]
                                       int stride, int pad, int dil) {
    struct ggml_tensor* im2col = ggml_im2col(ctx0, kernel, input, stride, 0, pad, 0, dil, 0, false, GGML_TYPE_F32);

    // ggml_mul_mat(A, B) = A^T @ B. Kernel as src0 (can be quantized), im2col as src1 (F32).
    // kernel_2d: [IC*K, OC], im2col_2d: [IC*K, N*OL]
    // result: kernel^T[OC, IC*K] @ im2col[IC*K, N*OL] = [OC, N*OL]
    struct ggml_tensor* result =
        ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]),
                     ggml_reshape_2d(ctx0, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]));

    // result is [OC, N*OL] → need [OL, OC, N]
    // Transpose to [N*OL, OC] then reshape
    result = ggml_cont(ctx0, ggml_transpose(ctx0, result));                              // [N*OL, OC]
    result = ggml_reshape_3d(ctx0, result, im2col->ne[1], kernel->ne[2], im2col->ne[2]); // [OL, OC, N]

    return result;
}

// Build the conv stem subgraph: raw audio -> feature vectors
static struct ggml_tensor* build_conv_stem(struct ggml_context* ctx0, const moonshine_model& model,
                                           struct ggml_tensor* audio) {
    const auto& hp = model.hparams;
    const int hidden = hp.enc_hidden_size;

    // Conv1 (no bias) + tanh
    struct ggml_tensor* cur = conv_1d_f32(ctx0, model.enc_conv1_w, audio, hp.conv1_stride, 0, 1);
    cur = ggml_tanh(ctx0, cur);

    // GroupNorm(1) + affine
    cur = ggml_group_norm(ctx0, cur, 1, hp.layer_norm_eps);
    cur = ggml_mul(ctx0, cur, ggml_reshape_3d(ctx0, model.enc_groupnorm_w, 1, hidden, 1));
    cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model.enc_groupnorm_b, 1, hidden, 1));

    // Conv2 + bias + GELU
    cur = conv_1d_f32(ctx0, model.enc_conv2_w, cur, hp.conv2_stride, 0, 1);
    cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model.enc_conv2_b, 1, model.enc_conv2_b->ne[0], 1));
    cur = ggml_gelu_erf(ctx0, cur);

    // Conv3 + bias + GELU
    cur = conv_1d_f32(ctx0, model.enc_conv3_w, cur, hp.conv3_stride, 0, 1);
    cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model.enc_conv3_b, 1, hidden, 1));
    cur = ggml_gelu_erf(ctx0, cur);

    // Reshape to [seq_len, hidden] and transpose to [hidden, seq_len]
    const int64_t seq_len = cur->ne[0];
    cur = ggml_reshape_2d(ctx0, cur, seq_len, hidden);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    return cur;
}

// Build the encoder transformer layers
static struct ggml_tensor* moonshine_build_encoder(struct ggml_context* ctx0, const moonshine_model& model,
                                                   struct ggml_tensor* conv_output, // [hidden, seq_len]
                                                   struct ggml_tensor* pos,         // [seq_len] I32 position IDs
                                                   int seq_len) {
    const auto& hp = model.hparams;
    const int n_heads = hp.n_heads;
    const int n_kv_heads = hp.n_kv_heads;
    const int head_dim = hp.head_dim;
    const int rotary_dim = hp.rotary_dim;
    const float eps = hp.layer_norm_eps;
    const float rope_theta = hp.rope_theta;

    struct ggml_tensor* cur = conv_output;

    for (uint32_t il = 0; il < hp.enc_n_layers; il++) {
        const auto& layer = model.enc_layers[il];

        struct ggml_tensor* residual = cur;

        // Pre-norm for attention
        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);

        // QKV projections: [hidden, seq_len] -> [head_dim*n_heads, seq_len]
        struct ggml_tensor* Q = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor* K = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor* V = ggml_mul_mat(ctx0, layer.attn_v, cur);

        // Reshape to multi-head: [head_dim, n_heads, seq_len]
        Q = ggml_reshape_3d(ctx0, Q, head_dim, n_heads, seq_len);
        K = ggml_reshape_3d(ctx0, K, head_dim, n_kv_heads, seq_len);
        V = ggml_reshape_3d(ctx0, V, head_dim, n_kv_heads, seq_len);

        // Partial RoPE (only first rotary_dim dimensions, consecutive pairs mode=0)
        Q = ggml_rope_ext(ctx0, Q, pos, nullptr, rotary_dim, 0, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx0, K, pos, nullptr, rotary_dim, 0, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Permute for flash attention:
        //   [head_dim, n_heads, seq_len] -> [head_dim, seq_len, n_heads]
        Q = ggml_permute(ctx0, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);

        // Flash attention (bidirectional — no causal mask)
        float scale = 1.0f / sqrtf((float)head_dim);
        struct ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, scale, 0.0f, 0.0f);

        // Result is [head_dim, n_heads, seq_len] — reshape to [hidden, seq_len]
        attn = ggml_reshape_2d(ctx0, attn, (int64_t)head_dim * n_heads, seq_len);

        // Output projection
        cur = ggml_mul_mat(ctx0, layer.attn_o, attn);

        // Residual connection (attention)
        cur = ggml_add(ctx0, cur, residual);

        residual = cur;

        // Pre-norm for FFN
        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);

        // FFN: fc1 + bias + GELU + fc2 + bias
        cur = ggml_mul_mat(ctx0, layer.ffn_fc1_w, cur);
        cur = ggml_add(ctx0, cur, layer.ffn_fc1_b);
        cur = ggml_gelu_erf(ctx0, cur);
        cur = ggml_mul_mat(ctx0, layer.ffn_fc2_w, cur);
        cur = ggml_add(ctx0, cur, layer.ffn_fc2_b);

        // Residual connection
        cur = ggml_add(ctx0, cur, residual);
    }

    // Final encoder output norm
    cur = ggml_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model.enc_output_norm);

    return cur;
}

// ---------- KV cache management ----------

static bool moonshine_kv_cache_init(moonshine_kv_cache& cache, int n_layers, int max_len, int n_kv_heads,
                                    int head_dim) {
    cache.max_len = max_len;
    cache.n = 0;
    cache.k.resize(n_layers);
    cache.v.resize(n_layers);

    const size_t n_tensors = 2 * (size_t)n_layers;
    const size_t mem_size = ggml_tensor_overhead() * n_tensors + 256;
    struct ggml_init_params params = {
        /*.mem_size   =*/mem_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    cache.ctx = ggml_init(params);
    if (!cache.ctx) {
        return false;
    }

    for (int i = 0; i < n_layers; i++) {
        // Moonshine decoder writes F32 directly to cache via ggml_set_*d;
        // F16 would need explicit casts in the decoder graph. Keep F32.
        cache.k[i] = ggml_new_tensor_3d(cache.ctx, GGML_TYPE_F32, head_dim, max_len, n_kv_heads);
        cache.v[i] = ggml_new_tensor_3d(cache.ctx, GGML_TYPE_F32, head_dim, max_len, n_kv_heads);
    }

    cache.buf = ggml_backend_alloc_ctx_tensors_from_buft(cache.ctx, ggml_backend_cpu_buffer_type());
    if (!cache.buf) {
        return false;
    }

    ggml_backend_buffer_clear(cache.buf, 0);
    return true;
}


// ---------- Encoder ----------

// Internal: run encoder, store output in ctx->encoder_out / ctx->enc_len
static int moonshine_run_encoder(struct moonshine_context* ctx, const float* audio, int n_samples) {
    const auto& hp = ctx->model.hparams;

    const size_t n_tensors = hp.enc_n_layers * 25 + 100;
    const size_t mem_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/mem_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context* ctx0 = ggml_init(params);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init ggml context\n", __func__);
        return -1;
    }

    struct ggml_tensor* input = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_samples, 1, 1);
    ggml_set_name(input, "audio_input");
    ggml_set_input(input);

    struct ggml_tensor* conv_out = build_conv_stem(ctx0, ctx->model, input);

    const int seq_len = (int)conv_out->ne[1];

    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, seq_len);
    ggml_set_name(pos, "enc_pos");
    ggml_set_input(pos);

    struct ggml_tensor* output = moonshine_build_encoder(ctx0, ctx->model, conv_out, pos, seq_len);
    ggml_set_name(output, "encoder_output");
    ggml_set_output(output);

    struct ggml_cgraph* graph = ggml_new_graph(ctx0);
    ggml_build_forward_expand(graph, output);

    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(gallocr, graph)) {
        fprintf(stderr, "%s: failed to alloc graph\n", __func__);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx0);
        return -1;
    }

    ggml_backend_tensor_set(input, audio, 0, n_samples * sizeof(float));

    std::vector<int32_t> pos_data(seq_len);
    for (int i = 0; i < seq_len; i++) {
        pos_data[i] = i;
    }
    ggml_backend_tensor_set(pos, pos_data.data(), 0, seq_len * sizeof(int32_t));

    if (ggml_backend_graph_compute(ctx->backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph compute failed\n", __func__);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx0);
        return -1;
    }

    const int hidden_dim = (int)output->ne[0];
    const int out_seq = (int)output->ne[1];
    const size_t out_bytes = hidden_dim * out_seq * sizeof(float);

    ctx->encoder_out.resize(hidden_dim * out_seq);
    ctx->enc_len = out_seq;
    ggml_backend_tensor_get(output, ctx->encoder_out.data(), 0, out_bytes);

    ggml_gallocr_free(gallocr);
    ggml_free(ctx0);
    return 0;
}

// Public API: run encoder and return results to caller
int moonshine_encode(struct moonshine_context* ctx, const float* audio, int n_samples, float** out_features,
                     int* out_seq_len, int* out_hidden_dim) {
    if (!ctx || !audio || n_samples <= 0) {
        return -1;
    }

    int ret = moonshine_run_encoder(ctx, audio, n_samples);
    if (ret != 0) {
        return ret;
    }

    const int hidden_dim = (int)ctx->model.hparams.enc_hidden_size;
    const int seq_len = ctx->enc_len;
    const size_t out_bytes = hidden_dim * seq_len * sizeof(float);

    float* features = (float*)malloc(out_bytes);
    if (!features) {
        fprintf(stderr, "%s: malloc failed\n", __func__);
        return -1;
    }

    memcpy(features, ctx->encoder_out.data(), out_bytes);

    *out_features = features;
    *out_seq_len = seq_len;
    *out_hidden_dim = hidden_dim;
    return 0;
}

// ---------- Cross-attention KV precomputation ----------

static int moonshine_precompute_cross_kv(struct moonshine_context* ctx) {
    const auto& model = ctx->model;
    const auto& hp = model.hparams;
    const int n_layers = hp.dec_n_layers;
    const int hidden = hp.enc_hidden_size;
    const int n_kv_heads = hp.n_kv_heads;
    const int head_dim = hp.head_dim;
    const int enc_len = ctx->enc_len;

    const size_t n_tensors = n_layers * 10 + 10;
    const size_t mem_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/mem_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context* ctx0 = ggml_init(params);
    if (!ctx0) {
        return -1;
    }

    // Input: encoder output [hidden, enc_len]
    struct ggml_tensor* enc_out = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden, enc_len);
    ggml_set_name(enc_out, "cross_enc_out");
    ggml_set_input(enc_out);

    struct ggml_cgraph* graph = ggml_new_graph(ctx0);

    std::vector<struct ggml_tensor*> k_outputs(n_layers);
    std::vector<struct ggml_tensor*> v_outputs(n_layers);

    for (int i = 0; i < n_layers; i++) {
        const auto& layer = model.dec_layers[i];

        // K = cross_attn_k * enc_out -> [n_kv_heads*head_dim, enc_len]
        struct ggml_tensor* K = ggml_mul_mat(ctx0, layer.cross_attn_k, enc_out);
        // Reshape to [head_dim, n_kv_heads, enc_len]
        K = ggml_reshape_3d(ctx0, K, head_dim, n_kv_heads, enc_len);
        // Permute to [head_dim, enc_len, n_kv_heads] for flash_attn layout
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        char name_k[64];
        snprintf(name_k, sizeof(name_k), "cross_k_%d", i);
        ggml_set_name(K, name_k);
        ggml_set_output(K);
        k_outputs[i] = K;
        ggml_build_forward_expand(graph, K);

        // V = cross_attn_v * enc_out -> [n_kv_heads*head_dim, enc_len]
        struct ggml_tensor* V = ggml_mul_mat(ctx0, layer.cross_attn_v, enc_out);
        V = ggml_reshape_3d(ctx0, V, head_dim, n_kv_heads, enc_len);
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));
        char name_v[64];
        snprintf(name_v, sizeof(name_v), "cross_v_%d", i);
        ggml_set_name(V, name_v);
        ggml_set_output(V);
        v_outputs[i] = V;
        ggml_build_forward_expand(graph, V);
    }

    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(gallocr, graph)) {
        fprintf(stderr, "%s: failed to alloc graph\n", __func__);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx0);
        return -1;
    }

    ggml_backend_tensor_set(enc_out, ctx->encoder_out.data(), 0, (size_t)hidden * enc_len * sizeof(float));

    if (ggml_backend_graph_compute(ctx->backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph compute failed\n", __func__);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx0);
        return -1;
    }

    // Copy results into cross KV cache
    const size_t kv_bytes = (size_t)head_dim * enc_len * n_kv_heads * sizeof(float);
    for (int i = 0; i < n_layers; i++) {
        ggml_backend_tensor_get(k_outputs[i], ctx->kv_cross.k[i]->data, 0, kv_bytes);
        ggml_backend_tensor_get(v_outputs[i], ctx->kv_cross.v[i]->data, 0, kv_bytes);
    }
    ctx->kv_cross.n = enc_len;

    ggml_gallocr_free(gallocr);
    ggml_free(ctx0);
    return 0;
}

// ---------- Decoder ----------

// Build a single decoder step graph
static struct ggml_tensor* moonshine_build_decoder_step(struct ggml_context* ctx0, const moonshine_model& model,
                                                        moonshine_kv_cache& kv_self, moonshine_kv_cache& kv_cross,
                                                        struct ggml_tensor* token_id, // [1] I32
                                                        struct ggml_tensor* dec_pos,  // [1] I32
                                                        int enc_len, int cur_pos, struct ggml_cgraph* graph) {
    const auto& hp = model.hparams;
    const int n_heads = hp.n_heads;
    const int n_kv_heads = hp.n_kv_heads;
    const int head_dim = hp.head_dim;
    const int rotary_dim = hp.rotary_dim;
    const int intermediate = hp.dec_intermediate;
    const float eps = hp.layer_norm_eps;
    const float rope_theta = hp.rope_theta;
    const float scale = 1.0f / sqrtf((float)head_dim);

    // Token embedding: [hidden, 1]
    // ggml_get_rows inherits the embed type; cast to F32 so mul_mat src1 is always F32
    // (multilingual GGUFs may store dec_embed as F16)
    struct ggml_tensor* cur = ggml_get_rows(ctx0, model.dec_embed, token_id);
    if (cur->type != GGML_TYPE_F32) {
        cur = ggml_cast(ctx0, cur, GGML_TYPE_F32);
    }

    for (uint32_t il = 0; il < hp.dec_n_layers; il++) {
        const auto& layer = model.dec_layers[il];

        // === Self-attention ===
        struct ggml_tensor* residual = cur;

        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);

        struct ggml_tensor* Q = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor* K_new = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor* V_new = ggml_mul_mat(ctx0, layer.attn_v, cur);

        Q = ggml_reshape_3d(ctx0, Q, head_dim, n_heads, 1);
        K_new = ggml_reshape_3d(ctx0, K_new, head_dim, n_kv_heads, 1);
        V_new = ggml_reshape_3d(ctx0, V_new, head_dim, n_kv_heads, 1);

        // Partial RoPE on Q and K (not V)
        Q = ggml_rope_ext(ctx0, Q, dec_pos, nullptr, rotary_dim, 0, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K_new =
            ggml_rope_ext(ctx0, K_new, dec_pos, nullptr, rotary_dim, 0, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Permute to [head_dim, 1, n_heads/n_kv_heads] for cache layout
        K_new = ggml_permute(ctx0, K_new, 0, 2, 1, 3);
        V_new = ggml_permute(ctx0, V_new, 0, 2, 1, 3);

        // Write K_new, V_new into self-attn cache at cur_pos
        struct ggml_tensor* k_cache_slice =
            ggml_view_3d(ctx0, kv_self.k[il], head_dim, 1, n_kv_heads, kv_self.k[il]->nb[1], kv_self.k[il]->nb[2],
                         cur_pos * kv_self.k[il]->nb[1]);
        struct ggml_tensor* v_cache_slice =
            ggml_view_3d(ctx0, kv_self.v[il], head_dim, 1, n_kv_heads, kv_self.v[il]->nb[1], kv_self.v[il]->nb[2],
                         cur_pos * kv_self.v[il]->nb[1]);

        ggml_build_forward_expand(graph, ggml_cpy(ctx0, K_new, k_cache_slice));
        ggml_build_forward_expand(graph, ggml_cpy(ctx0, V_new, v_cache_slice));

        // Read filled portion of cache [0..cur_pos+1]
        int kv_len = cur_pos + 1;
        struct ggml_tensor* K_cached = ggml_view_3d(ctx0, kv_self.k[il], head_dim, kv_len, n_kv_heads,
                                                    kv_self.k[il]->nb[1], kv_self.k[il]->nb[2], 0);
        struct ggml_tensor* V_cached = ggml_view_3d(ctx0, kv_self.v[il], head_dim, kv_len, n_kv_heads,
                                                    kv_self.v[il]->nb[1], kv_self.v[il]->nb[2], 0);

        // Permute Q for flash_attn: [head_dim, n_heads, 1] -> [head_dim, 1, n_heads]
        Q = ggml_permute(ctx0, Q, 0, 2, 1, 3);
        // K_cached and V_cached already in [head_dim, kv_len, n_kv_heads]

        struct ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K_cached, V_cached, nullptr, scale, 0.0f, 0.0f);
        // Output: [head_dim, n_heads, 1] -> [hidden, 1]
        attn = ggml_reshape_2d(ctx0, attn, (int64_t)n_heads * head_dim, 1);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_o, attn), residual);

        // === Cross-attention ===
        residual = cur;

        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.cross_attn_norm);

        struct ggml_tensor* Q_cross = ggml_mul_mat(ctx0, layer.cross_attn_q, cur);
        Q_cross = ggml_reshape_3d(ctx0, Q_cross, head_dim, n_heads, 1);
        // No RoPE for cross-attention
        Q_cross = ggml_permute(ctx0, Q_cross, 0, 2, 1, 3);

        // Read cross KV from precomputed cache: [head_dim, enc_len, n_kv_heads]
        struct ggml_tensor* K_cross = ggml_view_3d(ctx0, kv_cross.k[il], head_dim, enc_len, n_kv_heads,
                                                   kv_cross.k[il]->nb[1], kv_cross.k[il]->nb[2], 0);
        struct ggml_tensor* V_cross = ggml_view_3d(ctx0, kv_cross.v[il], head_dim, enc_len, n_kv_heads,
                                                   kv_cross.v[il]->nb[1], kv_cross.v[il]->nb[2], 0);

        struct ggml_tensor* cross_attn =
            ggml_flash_attn_ext(ctx0, Q_cross, K_cross, V_cross, nullptr, scale, 0.0f, 0.0f);
        cross_attn = ggml_reshape_2d(ctx0, cross_attn, (int64_t)n_heads * head_dim, 1);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.cross_attn_o, cross_attn), residual);

        // === Gated SiLU FFN ===
        residual = cur;

        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);

        struct ggml_tensor* fc1_out = ggml_mul_mat(ctx0, layer.ffn_fc1_w, cur);
        fc1_out = ggml_add(ctx0, fc1_out, layer.ffn_fc1_b);

        // Split: first half = value, second half = gate
        struct ggml_tensor* value_half = ggml_view_2d(ctx0, fc1_out, intermediate, 1, fc1_out->nb[1], 0);
        struct ggml_tensor* gate_half =
            ggml_view_2d(ctx0, fc1_out, intermediate, 1, fc1_out->nb[1], intermediate * sizeof(float));

        gate_half = ggml_silu(ctx0, gate_half);
        cur = ggml_mul(ctx0, gate_half, value_half);

        cur = ggml_mul_mat(ctx0, layer.ffn_fc2_w, cur);
        cur = ggml_add(ctx0, cur, layer.ffn_fc2_b);
        cur = ggml_add(ctx0, cur, residual);
    }

    // Final output norm
    cur = ggml_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model.dec_output_norm);

    // Project to vocab: [vocab_size, 1]
    struct ggml_tensor* logits = ggml_mul_mat(ctx0, model.dec_output, cur);

    return logits;
}

// Build a decoder step graph and return it via out params for the caller to manage
static int moonshine_decode_step(struct moonshine_context* ctx, int32_t token_id, std::vector<float>& logits_out,
                                 ggml_gallocr_t gallocr) {
    const auto& hp = ctx->model.hparams;
    const int cur_pos = ctx->kv_self.n;

    const size_t n_tensors = hp.dec_n_layers * 60 + 50;
    const size_t mem_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/mem_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context* ctx0 = ggml_init(params);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init ggml context\n", __func__);
        return -1;
    }

    struct ggml_tensor* inp_token = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_token, "token_id");
    ggml_set_input(inp_token);

    struct ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_pos, "dec_pos");
    ggml_set_input(inp_pos);

    struct ggml_cgraph* graph = ggml_new_graph(ctx0);

    struct ggml_tensor* logits = moonshine_build_decoder_step(ctx0, ctx->model, ctx->kv_self, ctx->kv_cross, inp_token,
                                                              inp_pos, ctx->enc_len, cur_pos, graph);

    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(graph, logits);

    if (!ggml_gallocr_alloc_graph(gallocr, graph)) {
        fprintf(stderr, "%s: failed to alloc graph\n", __func__);
        ggml_free(ctx0);
        return -1;
    }

    ggml_backend_tensor_set(inp_token, &token_id, 0, sizeof(int32_t));
    int32_t pos_val = cur_pos;
    ggml_backend_tensor_set(inp_pos, &pos_val, 0, sizeof(int32_t));

    if (ggml_backend_graph_compute(ctx->backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph compute failed\n", __func__);
        ggml_free(ctx0);
        return -1;
    }

    logits_out.resize(hp.vocab_size);
    ggml_backend_tensor_get(logits, logits_out.data(), 0, hp.vocab_size * sizeof(float));

    ctx->kv_self.n++;

    ggml_free(ctx0);
    return 0;
}

// ---------- Transcribe ----------

// Internal: run encoder + greedy/sampled decode loop. Always populates
// `out_tokens`. When `out_token_probs` is non-null, also fills it with the
// softmax probability of the picked token at each step (parallel to
// `out_tokens`). Returns 0 on success.
static int moonshine_transcribe_impl(struct moonshine_context* ctx, const float* audio, int n_samples,
                                     std::vector<int32_t>& out_tokens, std::vector<float>* out_token_probs) {
    out_tokens.clear();
    if (out_token_probs)
        out_token_probs->clear();

    if (!ctx || !audio || n_samples <= 0)
        return -1;

    const auto& hp = ctx->model.hparams;

    ctx->timing = {};
    ctx->timing.n_samples = n_samples;

    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. Run encoder
    int ret = moonshine_run_encoder(ctx, audio, n_samples);
    if (ret != 0) {
        return ret;
    }

    // 2. Init KV caches
    int max_gen = (int)(ceil((double)n_samples / 16000.0 * 6.5));
    if (max_gen > 194) {
        max_gen = 194;
    }
    int max_len = max_gen + 1; // +1 for BOS

    if (!moonshine_kv_cache_init(ctx->kv_self, hp.dec_n_layers, max_len, hp.n_kv_heads, hp.head_dim)) {
        return -2;
    }

    if (!moonshine_kv_cache_init(ctx->kv_cross, hp.dec_n_layers, ctx->enc_len, hp.n_kv_heads, hp.head_dim)) {
        ctx->kv_self.reset();
        return -2;
    }

    // 3. Precompute cross-attention KV
    ret = moonshine_precompute_cross_kv(ctx);
    if (ret != 0) {
        ctx->kv_self.reset();
        ctx->kv_cross.reset();
        return ret;
    }

    auto t_encode_done = std::chrono::high_resolution_clock::now();
    ctx->timing.encode_ms = std::chrono::duration<double, std::milli>(t_encode_done - t_start).count();

    // Encoder output no longer needed after cross-KV precompute
    { std::vector<float>().swap(ctx->encoder_out); }

    // 4. Pre-allocate decoder compute buffer with max-size graph
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    {
        const size_t n_tensors = hp.dec_n_layers * 60 + 50;
        const size_t mem_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead();
        struct ggml_init_params gp = {mem_size, nullptr, true};
        struct ggml_context* plan_ctx = ggml_init(gp);
        struct ggml_tensor* t = ggml_new_tensor_1d(plan_ctx, GGML_TYPE_I32, 1);
        ggml_set_name(t, "token_id");
        ggml_set_input(t);
        struct ggml_tensor* p = ggml_new_tensor_1d(plan_ctx, GGML_TYPE_I32, 1);
        ggml_set_name(p, "dec_pos");
        ggml_set_input(p);
        struct ggml_cgraph* plan_graph = ggml_new_graph(plan_ctx);
        struct ggml_tensor* plan_logits = moonshine_build_decoder_step(
            plan_ctx, ctx->model, ctx->kv_self, ctx->kv_cross, t, p, ctx->enc_len, max_len - 1, plan_graph);
        ggml_set_output(plan_logits);
        ggml_build_forward_expand(plan_graph, plan_logits);
        ggml_gallocr_reserve(gallocr, plan_graph);
        ggml_free(plan_ctx);
    }

    // 5. Decode loop. Picks via argmax when temperature == 0, otherwise
    // softmax(logits/T) + multinomial sample. Beam search (beam_size > 1)
    // takes a separate path below — it ignores temperature and always
    // returns the highest cumulative-log-prob hypothesis.
    if (ctx->beam_size > 1) {
        // 5a. Prefill BOS to populate slot 0 + capture initial logits.
        std::vector<float> bos_logits;
        ret = moonshine_decode_step(ctx, (int32_t)hp.bos_token_id, bos_logits, gallocr);
        if (ret != 0) {
            ggml_gallocr_free(gallocr);
            ctx->kv_self.reset();
            ctx->kv_cross.reset();
            return ret;
        }

        // 5b. KV snapshot helpers — branched beam needs each beam to fork
        // its own KV state. moonshine's kv_self.k[il] / v[il] are per-layer
        // tensors of shape [head_dim, max_len, n_kv_heads]; we snapshot the
        // full tensor (small — single-MB total for the 8L Tiny model). The
        // self-attention KV is the only state that diverges between beams;
        // kv_cross is precomputed once and shared across beams via ctx.
        struct moonshine_kv_snap {
            int n;
            std::vector<std::vector<uint8_t>> k_data;
            std::vector<std::vector<uint8_t>> v_data;
        };
        auto save = [](moonshine_context* c) -> moonshine_kv_snap* {
            auto* s = new moonshine_kv_snap();
            s->n = c->kv_self.n;
            const int L = (int)c->kv_self.k.size();
            s->k_data.resize((size_t)L);
            s->v_data.resize((size_t)L);
            for (int il = 0; il < L; il++) {
                size_t kb = ggml_nbytes(c->kv_self.k[il]);
                size_t vb = ggml_nbytes(c->kv_self.v[il]);
                s->k_data[(size_t)il].resize(kb);
                s->v_data[(size_t)il].resize(vb);
                ggml_backend_tensor_get(c->kv_self.k[il], s->k_data[(size_t)il].data(), 0, kb);
                ggml_backend_tensor_get(c->kv_self.v[il], s->v_data[(size_t)il].data(), 0, vb);
            }
            return s;
        };
        auto restore = [](moonshine_context* c, moonshine_kv_snap* s) {
            c->kv_self.n = s->n;
            const int L = (int)c->kv_self.k.size();
            for (int il = 0; il < L; il++) {
                ggml_backend_tensor_set(c->kv_self.k[il], s->k_data[(size_t)il].data(), 0,
                                        s->k_data[(size_t)il].size());
                ggml_backend_tensor_set(c->kv_self.v[il], s->v_data[(size_t)il].data(), 0,
                                        s->v_data[(size_t)il].size());
            }
        };
        auto snap_free = [](moonshine_kv_snap* s) { delete s; };
        std::vector<float> step_logits_buf;
        auto step = [&step_logits_buf, gallocr](moonshine_context* c, int32_t tok, int /*n_past*/) -> float* {
            // n_past is implicit in c->kv_self.n (set by restore_fn just
            // before this call). decode_step uses it directly.
            if (moonshine_decode_step(c, tok, step_logits_buf, gallocr) != 0)
                return nullptr;
            const int V = (int)c->model.hparams.vocab_size;
            float* out = (float*)std::malloc((size_t)V * sizeof(float));
            std::memcpy(out, step_logits_buf.data(), (size_t)V * sizeof(float));
            return out;
        };

        core_beam_decode::Config cfg;
        cfg.max_new_tokens = max_len - 1;
        cfg.eos_id = (int)hp.eos_token_id;
        cfg.vocab_size = (int)hp.vocab_size;
        cfg.beam_size = ctx->beam_size;
        cfg.prompt_len = 1; // BOS occupies slot 0

        auto r = core_beam_decode::run_with_probs_branched(ctx, bos_logits.data(), save, restore, snap_free, step, cfg);

        for (size_t i = 0; i < r.tokens.size(); i++) {
            if (r.tokens[i] == (int32_t)hp.eos_token_id)
                break;
            out_tokens.push_back(r.tokens[i]);
            if (out_token_probs)
                out_token_probs->push_back(r.probs[i]);
        }

        auto t_decode_done = std::chrono::high_resolution_clock::now();
        ctx->timing.decode_ms = std::chrono::duration<double, std::milli>(t_decode_done - t_encode_done).count();
        ctx->timing.n_tokens = (int)out_tokens.size();

        ggml_gallocr_free(gallocr);
        ctx->kv_self.reset();
        ctx->kv_cross.reset();
        return 0;
    }

    int32_t token = (int32_t)hp.bos_token_id;
    std::vector<float> logits((size_t)hp.vocab_size);
    const float T = ctx->temperature;
    const bool sample = (T > 0.0f);

    // Per-call seed: derive from samples + audio length so repeated runs are
    // deterministic but different from each other. When the caller has
    // injected a sticky `seed_override` (best-of-N), mix it in too so each
    // run draws an independent sample from the same audio.
    uint64_t rng_state = 0x9E3779B97F4A7C15ull ^ (uint64_t)(uintptr_t)audio ^ (uint64_t)n_samples ^
                         (ctx->seed_override * 0xBF58476D1CE4E5B9ull);
    if (rng_state == 0)
        rng_state = 0x9E3779B97F4A7C15ull;
    auto rand_uniform = [&]() -> float {
        // xorshift64
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        return (float)((rng_state >> 11) & 0x1FFFFF) / (float)(1 << 21);
    };

    for (int step = 0; step < max_len; step++) {
        ret = moonshine_decode_step(ctx, token, logits, gallocr);
        if (ret != 0) {
            break;
        }

        const int V = (int)hp.vocab_size;
        int32_t picked = 0;
        float picked_prob = 0.0f;

        if (!sample) {
            // Greedy argmax
            int32_t best = 0;
            float best_val = logits[0];
            for (int i = 1; i < V; i++) {
                if (logits[i] > best_val) {
                    best_val = logits[i];
                    best = i;
                }
            }
            picked = best;
            if (out_token_probs) {
                // Softmax of the picked logit (numerically stable).
                float mx = best_val;
                float s = 0.f;
                for (int i = 0; i < V; i++)
                    s += expf(logits[i] - mx);
                picked_prob = 1.0f / s;
            }
        } else {
            // Multinomial sample from softmax(logits / T).
            float mx = logits[0];
            for (int i = 1; i < V; i++)
                if (logits[i] > mx)
                    mx = logits[i];
            std::vector<float> probs((size_t)V);
            float s = 0.f;
            const float inv_T = 1.0f / T;
            for (int i = 0; i < V; i++) {
                probs[i] = expf((logits[i] - mx) * inv_T);
                s += probs[i];
            }
            const float inv_s = 1.0f / s;
            for (int i = 0; i < V; i++)
                probs[i] *= inv_s;
            float u = rand_uniform();
            float c = 0.f;
            picked = V - 1;
            for (int i = 0; i < V; i++) {
                c += probs[i];
                if (u <= c) {
                    picked = i;
                    break;
                }
            }
            if (out_token_probs)
                picked_prob = probs[picked];
        }

        if (picked == (int32_t)hp.eos_token_id) {
            break;
        }

        out_tokens.push_back(picked);
        if (out_token_probs)
            out_token_probs->push_back(picked_prob);
        token = picked;
    }

    auto t_decode_done = std::chrono::high_resolution_clock::now();
    ctx->timing.decode_ms = std::chrono::duration<double, std::milli>(t_decode_done - t_encode_done).count();
    ctx->timing.n_tokens = (int)out_tokens.size();

    // Cleanup
    ggml_gallocr_free(gallocr);
    ctx->kv_self.reset();
    ctx->kv_cross.reset();

    return 0;
}

const char* moonshine_transcribe(struct moonshine_context* ctx, const float* audio, int n_samples) {
    if (!ctx)
        return "";

    std::vector<int32_t> tokens;
    if (moonshine_transcribe_impl(ctx, audio, n_samples, tokens, nullptr) != 0)
        return "";

    ctx->result_text = ctx->tokenizer.tokens_to_text(tokens);
    return ctx->result_text.c_str();
}

extern "C" struct moonshine_result* moonshine_transcribe_with_probs(struct moonshine_context* ctx, const float* audio,
                                                                    int n_samples) {
    if (!ctx)
        return nullptr;

    std::vector<int32_t> tokens;
    std::vector<float> probs;
    if (moonshine_transcribe_impl(ctx, audio, n_samples, tokens, &probs) != 0)
        return nullptr;

    ctx->result_text = ctx->tokenizer.tokens_to_text(tokens);

    auto* r = (moonshine_result*)calloc(1, sizeof(moonshine_result));
    r->n_tokens = (int)tokens.size();
    r->text = strdup(ctx->result_text.c_str());
    if (r->n_tokens > 0) {
        r->token_ids = (int*)malloc(sizeof(int) * (size_t)r->n_tokens);
        r->token_probs = (float*)malloc(sizeof(float) * (size_t)r->n_tokens);
        for (int i = 0; i < r->n_tokens; i++) {
            r->token_ids[i] = tokens[i];
            r->token_probs[i] = (i < (int)probs.size()) ? probs[i] : 0.0f;
        }
    }
    return r;
}

extern "C" void moonshine_result_free(struct moonshine_result* r) {
    if (!r)
        return;
    free(r->text);
    free(r->token_ids);
    free(r->token_probs);
    free(r);
}

extern "C" void moonshine_set_temperature(struct moonshine_context* ctx, float temperature) {
    if (ctx)
        ctx->temperature = (temperature > 0.0f) ? temperature : 0.0f;
}

extern "C" void moonshine_set_seed(struct moonshine_context* ctx, uint64_t seed) {
    if (ctx)
        ctx->seed_override = seed;
}

extern "C" void moonshine_set_beam_size(struct moonshine_context* ctx, int beam_size) {
    if (ctx)
        ctx->beam_size = (beam_size > 0) ? beam_size : 1;
}

extern "C" const char* moonshine_token_text(struct moonshine_context* ctx, int token_id) {
    if (!ctx)
        return "";
    ctx->scratch_token_text = ctx->tokenizer.token_to_piece(token_id);
    return ctx->scratch_token_text.c_str();
}

void moonshine_free(struct moonshine_context* ctx) {
    if (!ctx) {
        return;
    }
    // moonshine_model's destructor frees buf_w + ctx_w. It must run BEFORE
    // we free the backend, otherwise ggml_metal sees a buffer outliving its
    // device and ggml_metal_rsets_free's assert fires at process exit.
    auto* backend = ctx->backend;
    delete ctx;
    if (backend) {
        ggml_backend_free(backend);
    }
}

void moonshine_print_model_info(struct moonshine_context* ctx) {
    if (!ctx) {
        fprintf(stderr, "%s: null context\n", __func__);
        return;
    }

    const auto& hp = ctx->model.hparams;

    printf("=== Moonshine Model Info ===\n");
    printf("Hyperparameters:\n");
    printf("  enc_hidden_size:      %u\n", hp.enc_hidden_size);
    printf("  enc_n_layers:         %u\n", hp.enc_n_layers);
    printf("  dec_n_layers:         %u\n", hp.dec_n_layers);
    printf("  n_heads:              %u\n", hp.n_heads);
    printf("  n_kv_heads:           %u\n", hp.n_kv_heads);
    printf("  head_dim:             %u\n", hp.head_dim);
    printf("  rotary_dim:           %u\n", hp.rotary_dim);
    printf("  enc_intermediate:     %u\n", hp.enc_intermediate);
    printf("  dec_intermediate:     %u\n", hp.dec_intermediate);
    printf("  vocab_size:           %u\n", hp.vocab_size);
    printf("  bos_token_id:         %u\n", hp.bos_token_id);
    printf("  eos_token_id:         %u\n", hp.eos_token_id);
    printf("  layer_norm_eps:       %g\n", hp.layer_norm_eps);
    printf("  rope_theta:           %g\n", hp.rope_theta);
    printf("  partial_rotary_factor: %g\n", hp.partial_rotary_factor);
    printf("  conv1: kernel=%u stride=%u\n", hp.conv1_kernel_size, hp.conv1_stride);
    printf("  conv2: kernel=%u stride=%u\n", hp.conv2_kernel_size, hp.conv2_stride);
    printf("  conv3: kernel=%u stride=%u\n", hp.conv3_kernel_size, hp.conv3_stride);

    // Tensor table
    printf("\nTensors:\n");
    printf("  %-45s %6s %20s %10s\n", "Name", "Type", "Shape", "Bytes");
    printf("  %-45s %6s %20s %10s\n", "----", "----", "-----", "-----");

    int n_tensors = 0;
    size_t total_bytes = 0;
    for (struct ggml_tensor* t = ggml_get_first_tensor(ctx->model.ctx_w); t != nullptr;
         t = ggml_get_next_tensor(ctx->model.ctx_w, t)) {
        char shape[64];
        if (t->ne[3] > 1) {
            snprintf(shape, sizeof(shape), "[%lld,%lld,%lld,%lld]", (long long)t->ne[0], (long long)t->ne[1],
                     (long long)t->ne[2], (long long)t->ne[3]);
        } else if (t->ne[2] > 1) {
            snprintf(shape, sizeof(shape), "[%lld,%lld,%lld]", (long long)t->ne[0], (long long)t->ne[1],
                     (long long)t->ne[2]);
        } else if (t->ne[1] > 1) {
            snprintf(shape, sizeof(shape), "[%lld,%lld]", (long long)t->ne[0], (long long)t->ne[1]);
        } else {
            snprintf(shape, sizeof(shape), "[%lld]", (long long)t->ne[0]);
        }

        size_t nbytes = ggml_nbytes(t);
        printf("  %-45s %6s %20s %10zu\n", ggml_get_name(t), ggml_type_name(t->type), shape, nbytes);
        n_tensors++;
        total_bytes += nbytes;
    }
    printf("  Total: %d tensors, %zu bytes (%.1f MB)\n", n_tensors, total_bytes,
           (double)total_bytes / (1024.0 * 1024.0));

    // Tokenizer info
    printf("\nTokenizer:\n");
    printf("  vocab_size: %zu\n", ctx->tokenizer.vocab_size());

    // Sample decodes
    const int32_t sample_ids[] = {1, 2, 100, 1000};
    for (int32_t id : sample_ids) {
        std::vector<int32_t> tokens = {id};
        std::string text = ctx->tokenizer.tokens_to_text(tokens);
        if (text.empty()) {
            printf("  token %5d -> (empty/special)\n", id);
        } else {
            printf("  token %5d -> \"%s\"\n", id, text.c_str());
        }
    }
}

void moonshine_set_n_threads(struct moonshine_context* ctx, int n_threads) {
    if (!ctx || n_threads < 1) {
        return;
    }
    ctx->n_threads = n_threads;
    if (ctx->backend) {
        ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);
    }
}

int moonshine_get_n_threads(struct moonshine_context* ctx) {
    if (!ctx) {
        return 0;
    }
    return ctx->n_threads;
}

int moonshine_get_timing(struct moonshine_context* ctx, struct moonshine_timing* timing) {
    if (!ctx || !timing) {
        return -1;
    }
    *timing = ctx->timing;
    return 0;
}
