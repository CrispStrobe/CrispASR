// arktts.cpp — ArkTTS backend (Audio8-TTS-Realistic-0.6b).
//
// Dual-stage transformer (slow AR causal transformer + fast RAS layers)
// with 44.1 kHz Snake-activated vocoder decoder.

#include "arktts.h"

#include "core/activation.h"
#include "core/attention.h"
#include "core/bpe.h"
#include "core/ffn.h"
#include "core/gpu_backend_pref.h"
#include "gguf.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/ggml_cpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

struct arktts_hparams {
    uint32_t dim = 896;
    uint32_t intermediate_size = 4864;
    uint32_t n_layer = 24;
    uint32_t n_head = 14;
    uint32_t n_local_heads = 2;
    uint32_t head_dim = 64;
    float norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    uint32_t vocab_size = 155776;
    uint32_t num_codebooks = 10;
    uint32_t codebook_size = 4096;
    uint32_t semantic_begin_id = 151678;
    uint32_t semantic_end_id = 155773;
    uint32_t eos_token_id = 151645;

    uint32_t fast_dim = 896;
    uint32_t fast_intermediate_size = 4864;
    uint32_t n_fast_layer = 4;
    uint32_t fast_n_head = 14;
    uint32_t fast_n_local_heads = 2;
    uint32_t fast_head_dim = 64;

    uint32_t sample_rate = 44100;
    uint32_t frame_size = 2048;
    uint32_t hop_length = 512;
};

struct arktts_context {
    arktts_hparams hp;
    arktts_context_params params;
    ggml_context* ctx_weights = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf_weights = nullptr;
    std::mt19937_64 rng;
    std::vector<float> ref_audio;
    std::string ref_text;
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
};

struct arktts_context_params arktts_context_default_params(void) {
    struct arktts_context_params p;
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.temperature = 0.7f;
    p.top_p = 0.9f;
    p.top_k = 50;
    p.seed = 42;
    p.max_tokens = 512;
    p.flash_attn = false;
    return p;
}

int arktts_get_sample_rate(const struct arktts_context* ctx) {
    return ctx ? ctx->hp.sample_rate : 44100;
}

void arktts_set_temperature(struct arktts_context* ctx, float temperature) {
    if (ctx)
        ctx->params.temperature = temperature;
}

void arktts_set_seed(struct arktts_context* ctx, uint64_t seed) {
    if (ctx) {
        ctx->params.seed = seed;
        ctx->rng.seed(seed ? seed : 42);
    }
}

int arktts_set_reference(struct arktts_context* ctx, const float* audio_44k, int n_samples,
                         const char* reference_text) {
    if (!ctx)
        return -1;
    if (audio_44k && n_samples > 0) {
        ctx->ref_audio.assign(audio_44k, audio_44k + n_samples);
        ctx->ref_text = reference_text ? reference_text : "";
    } else {
        ctx->ref_audio.clear();
        ctx->ref_text.clear();
    }
    return 0;
}

int arktts_load_tokenizer(struct arktts_context* /*ctx*/, const char* /*tokenizer_json_path*/) {
    return 0;
}

struct arktts_context* arktts_init_from_file(const char* path_model, struct arktts_context_params params) {
    auto c = std::make_unique<arktts_context>();
    c->params = params;
    c->rng.seed(params.seed ? params.seed : 42);

    c->backend = crispasr_init_gpu_backend();
    if (!c->backend) {
        c->backend = ggml_backend_cpu_init();
    }

    gguf_init_params gip = {/*.no_alloc=*/true, /*.ctx=*/&c->ctx_weights};
    gguf_context* meta = gguf_init_from_file(path_model, gip);
    if (!meta) {
        return nullptr;
    }

    auto get_u32 = [&](const char* key, uint32_t def) -> uint32_t {
        int idx = gguf_find_key(meta, key);
        return idx >= 0 ? (uint32_t)gguf_get_val_u32(meta, idx) : def;
    };

    c->hp.dim = get_u32("arktts.dim", c->hp.dim);
    c->hp.intermediate_size = get_u32("arktts.intermediate_size", c->hp.intermediate_size);
    c->hp.n_layer = get_u32("arktts.n_layer", c->hp.n_layer);
    c->hp.n_head = get_u32("arktts.n_head", c->hp.n_head);
    c->hp.n_local_heads = get_u32("arktts.n_local_heads", c->hp.n_local_heads);
    c->hp.head_dim = get_u32("arktts.head_dim", c->hp.head_dim);
    c->hp.sample_rate = get_u32("arktts.codec.sample_rate", c->hp.sample_rate);
    c->hp.num_codebooks = get_u32("arktts.num_codebooks", c->hp.num_codebooks);

    c->buf_weights = ggml_backend_alloc_ctx_tensors(c->ctx_weights, c->backend);
    gguf_free(meta);

    return c.release();
}

float* arktts_synthesize(struct arktts_context* ctx, const char* text, int* out_n) {
    if (!ctx || !text || !out_n)
        return nullptr;
    // Initial synthesizer stub producing model-rate silence buffer
    int n_samples = ctx->hp.sample_rate * 2;
    float* out = (float*)malloc(n_samples * sizeof(float));
    if (out) {
        memset(out, 0, n_samples * sizeof(float));
        *out_n = n_samples;
    }
    return out;
}

void arktts_free_audio(float* pcm) {
    if (pcm)
        free(pcm);
}

void arktts_free(struct arktts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->buf_weights) {
        ggml_backend_buffer_free(ctx->buf_weights);
    }
    if (ctx->ctx_weights) {
        ggml_free(ctx->ctx_weights);
    }
    if (ctx->backend) {
        ggml_backend_free(ctx->backend);
    }
    delete ctx;
}
