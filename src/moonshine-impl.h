#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

struct moonshine_kv_cache {
    int n = 0;
    int max_len = 0;
    std::vector<struct ggml_tensor*> k;
    std::vector<struct ggml_tensor*> v;
    struct ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    moonshine_kv_cache() = default;
    ~moonshine_kv_cache() { reset(); }

    void reset() {
        if (buf) {
            ggml_backend_buffer_free(buf);
            buf = nullptr;
        }
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        k.clear();
        v.clear();
        n = 0;
        max_len = 0;
    }

    moonshine_kv_cache(const moonshine_kv_cache&) = delete;
    moonshine_kv_cache& operator=(const moonshine_kv_cache&) = delete;
};

struct moonshine_hparams {
    uint32_t enc_hidden_size;
    uint32_t enc_n_layers;
    uint32_t dec_n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t enc_intermediate;
    uint32_t dec_intermediate;
    uint32_t vocab_size;
    uint32_t bos_token_id;
    uint32_t eos_token_id;

    float layer_norm_eps;
    float rope_theta;
    float partial_rotary_factor;

    uint32_t conv1_kernel_size;
    uint32_t conv1_stride;
    uint32_t conv2_kernel_size;
    uint32_t conv2_stride;
    uint32_t conv3_kernel_size;
    uint32_t conv3_stride;

    // derived
    uint32_t head_dim;
    uint32_t rotary_dim;
};

struct moonshine_encoder_layer {
    struct ggml_tensor* attn_norm = nullptr;
    struct ggml_tensor* attn_q = nullptr;
    struct ggml_tensor* attn_k = nullptr;
    struct ggml_tensor* attn_v = nullptr;
    struct ggml_tensor* attn_o = nullptr;
    struct ggml_tensor* ffn_norm = nullptr;
    struct ggml_tensor* ffn_fc1_w = nullptr;
    struct ggml_tensor* ffn_fc1_b = nullptr;
    struct ggml_tensor* ffn_fc2_w = nullptr;
    struct ggml_tensor* ffn_fc2_b = nullptr;
};

struct moonshine_decoder_layer {
    struct ggml_tensor* attn_norm = nullptr;
    struct ggml_tensor* attn_q = nullptr;
    struct ggml_tensor* attn_k = nullptr;
    struct ggml_tensor* attn_v = nullptr;
    struct ggml_tensor* attn_o = nullptr;
    struct ggml_tensor* cross_attn_norm = nullptr;
    struct ggml_tensor* cross_attn_q = nullptr;
    struct ggml_tensor* cross_attn_k = nullptr;
    struct ggml_tensor* cross_attn_v = nullptr;
    struct ggml_tensor* cross_attn_o = nullptr;
    struct ggml_tensor* ffn_norm = nullptr;
    struct ggml_tensor* ffn_fc1_w = nullptr;
    struct ggml_tensor* ffn_fc1_b = nullptr;
    struct ggml_tensor* ffn_fc2_w = nullptr;
    struct ggml_tensor* ffn_fc2_b = nullptr;
};

struct moonshine_model {
    moonshine_hparams hparams = {};

    // encoder conv stem
    struct ggml_tensor* enc_conv1_w = nullptr;
    struct ggml_tensor* enc_groupnorm_w = nullptr;
    struct ggml_tensor* enc_groupnorm_b = nullptr;
    struct ggml_tensor* enc_conv2_w = nullptr;
    struct ggml_tensor* enc_conv2_b = nullptr;
    struct ggml_tensor* enc_conv3_w = nullptr;
    struct ggml_tensor* enc_conv3_b = nullptr;

    // encoder layers
    std::vector<moonshine_encoder_layer> enc_layers;

    // encoder output norm
    struct ggml_tensor* enc_output_norm = nullptr;

    // decoder
    struct ggml_tensor* dec_embed = nullptr;
    struct ggml_tensor* dec_output_norm = nullptr;
    struct ggml_tensor* dec_output = nullptr;

    // decoder layers
    std::vector<moonshine_decoder_layer> dec_layers;

    // ggml state
    ggml_backend_buffer_t buf_w = nullptr;
    struct ggml_context* ctx_w = nullptr;

    moonshine_model() = default;
    ~moonshine_model() {
        ggml_backend_buffer_free(buf_w);
        ggml_free(ctx_w);
    }

    moonshine_model(const moonshine_model&) = delete;
    moonshine_model& operator=(const moonshine_model&) = delete;
};
