/**
 * wav2vec2-ggml.h  —  Internal C++ header for the wav2vec2 CTC encoder.
 *
 * Loads any Wav2Vec2ForCTC GGUF model (produced by convert-wav2vec2-to-gguf.py)
 * and exposes two primitives used by the CTC forced-alignment pipeline:
 *
 *   wav2vec2_load()           — load model from disk
 *   wav2vec2_compute_logits() — run encoder + LM head, return [T × V] logits
 *   wav2vec2_greedy_decode()  — greedy CTC decode (for sanity checks)
 *
 * Not a public API; included only by wav2vec2-ggml.cpp and align.cpp.
 *
 * Adapted from nabil6391/wav2vec2.cpp (MIT licence).
 */

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Hyper-parameters (loaded from GGUF metadata)
// ---------------------------------------------------------------------------

struct wav2vec2_hparams {
    uint32_t vocab_size = 39;
    uint32_t hidden_size = 384;
    uint32_t num_hidden_layers = 10;
    uint32_t num_attention_heads = 6;
    uint32_t intermediate_size = 1536;
    uint32_t num_feat_extract_layers = 7;
    uint32_t num_conv_pos_embeddings = 128;
    uint32_t num_conv_pos_embedding_groups = 16;
    float layer_norm_eps = 1e-5f;
    uint32_t pad_token_id = 38; // CTC blank token
    // 0 = group-norm variant (InstanceNorm on layer 0), e.g. wav2vec2-base
    // 1 = layer-norm variant (LayerNorm on all CNN layers), e.g. wav2vec2-large
    uint32_t feat_extract_norm_type = 0;
    // 0 = post-norm (standard, base models): attn → add → LN → FFN → add → LN
    // 1 = pre-norm (stable, large models): LN → attn → add → LN → FFN → add
    uint32_t do_stable_layer_norm = 1; // default 1 for backward compat (existing GGUFs are large models)
    // Data2Vec: applies global encoder LN BEFORE transformer layers (unique behavior)
    uint32_t global_ln_before_encoder = 0;

    uint32_t conv_dim[7] = {256, 256, 512, 512, 512, 512, 512};
    uint32_t conv_kernel[7] = {10, 3, 3, 3, 3, 2, 2};
    uint32_t conv_stride[7] = {5, 2, 2, 2, 2, 2, 2};
};

// ---------------------------------------------------------------------------
// Per-layer tensor containers
// ---------------------------------------------------------------------------

struct w2v_cnn_layer {
    ggml_tensor* conv_w = nullptr; // [Cout, Cin, K] F32
    ggml_tensor* conv_b = nullptr; // [Cout]
    ggml_tensor* norm_w = nullptr; // [Cout] GroupNorm / LayerNorm weight (may be null)
    ggml_tensor* norm_b = nullptr; // [Cout]
    bool has_norm = false;
};

struct w2v_enc_layer {
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr; // pre-attention LN
    ggml_tensor *q_w = nullptr, *q_b = nullptr;
    ggml_tensor *k_w = nullptr, *k_b = nullptr;
    ggml_tensor *v_w = nullptr, *v_b = nullptr;
    ggml_tensor *o_w = nullptr, *o_b = nullptr;
    ggml_tensor *ln2_w = nullptr, *ln2_b = nullptr; // pre-FFN LN
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
    ggml_tensor *fc2_w = nullptr, *fc2_b = nullptr;
};

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

struct wav2vec2_model {
    wav2vec2_hparams hparams;

    w2v_cnn_layer cnn[7];
    ggml_tensor *fp_ln_w = nullptr, *fp_ln_b = nullptr;       // feature-proj LN
    ggml_tensor *fp_w = nullptr, *fp_b = nullptr;             // feature-proj linear
    ggml_tensor *pos_conv_w = nullptr, *pos_conv_b = nullptr; // single-layer (wav2vec2)
    struct {
        ggml_tensor *w = nullptr, *b = nullptr;
    } pos_conv_layers[8];      // multi-layer (data2vec)
    int n_pos_conv_layers = 1; // 1 for wav2vec2, 5 for data2vec
    ggml_tensor *enc_ln_w = nullptr, *enc_ln_b = nullptr;
    std::vector<w2v_enc_layer> enc;
    ggml_tensor *lm_w = nullptr, *lm_b = nullptr; // CTC head

    std::vector<std::string> vocab; // id → token string (e.g. "a", "|", "<pad>")

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr; // backend buffer (core_gguf)
    ggml_backend_t backend = nullptr;    // backend that owns buf
    std::map<std::string, ggml_tensor*> tensors;

    wav2vec2_model() = default;
    wav2vec2_model(const wav2vec2_model&) = delete;
    wav2vec2_model& operator=(const wav2vec2_model&) = delete;
    // Free in dependency order: buf (depends on backend) → ctx → backend.
    // Without this, on Metal the residency set survives past main() and
    // ggml_metal's static teardown trips ggml_metal_rsets_free's assert.
    ~wav2vec2_model() noexcept {
        if (buf)
            ggml_backend_buffer_free(buf);
        if (ctx)
            ggml_free(ctx);
        if (backend)
            ggml_backend_free(backend);
    }
};

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

/**
 * Load model from a GGUF file produced by convert-wav2vec2-to-gguf.py.
 * Returns true on success; prints errors to stderr.
 */
bool wav2vec2_load(const char* fname, wav2vec2_model& model);

/**
 * Run the full encoder + LM head on n_samples of 16 kHz mono PCM.
 * Returns a flat vector of size T * V (row-major: frame 0 first).
 * Returns an empty vector on error.
 */
std::vector<float> wav2vec2_compute_logits(const wav2vec2_model& m, const float* raw_audio, int n_samples,
                                           int n_threads = 1);

/**
 * Greedy CTC decode from pre-computed logits[T * V].
 * Collapses repeated tokens and strips blanks.
 */
std::string wav2vec2_greedy_decode(const wav2vec2_model& m, const float* logits, int T);

/**
 * Compute the encoder frame duration in seconds.
 * For standard wav2vec2 at 16 kHz with default strides this is 0.02 s (50 fps).
 */
inline float wav2vec2_frame_dur(const wav2vec2_model& m, int sample_rate = 16000) {
    float stride = 1.f;
    for (uint32_t i = 0; i < m.hparams.num_feat_extract_layers; i++)
        stride *= (float)m.hparams.conv_stride[i];
    return stride / (float)sample_rate;
}
