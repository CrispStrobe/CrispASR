// qwen3_tts.cpp — runtime for Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-Base
//
// SCAFFOLD ONLY (April 2026): the talker LM GGUF loads, the
// codec-path setter is wired, and the synthesise entry point
// returns nullptr with an explicit "not implemented" trace so
// callers can integrate against the C ABI today and the LM +
// codec forward passes can land incrementally without breaking
// the build.
//
// Architecture (from Qwen3-TTS-12Hz-{0.6B,1.7B}-Base config.json):
//
//   Talker (autoregressive LM, generates 16-codebook RVQ codes):
//     - 28-layer Qwen3 (1024d / 2048d for 0.6B / 1.7B)
//     - 16Q / 8KV / head_dim 128, SiLU SwiGLU, RoPE theta 1e6
//     - mrope_section [24, 20, 20] (multi-rotary), interleaved
//     - separate text-vocab (151936) and audio-vocab (3072), with
//       text projecting into the LM's hidden via a 2048-d
//       text_hidden_size first (Qwen3-TTS-specific).
//     - 5-layer code_predictor head selects the 16 codes per frame.
//     - speaker_encoder (24 kHz, enc_dim 1024) produces a voice
//       embedding from a reference WAV for voice cloning.
//
//   Codec (Qwen/Qwen3-TTS-Tokenizer-12Hz):
//     - 8-layer encoder (hidden 512, 8 heads, sliding 250)
//     - 8-layer decoder (hidden 512, 16 heads, sliding 72)
//     - 32 quantisers on encode, 16 on decode (talker emits 16)
//     - 1 semantic + 4096-codebook semantic head separate from RVQ
//     - 24 kHz in/out, 12.5 fps, encode_downsample = decode_upsample = 1920
//
// Known follow-ups (PLAN #52):
//   1. Talker forward (28L Qwen3 with multi-rope + dual embeddings):
//      core_attn::kv_self_attn covers most of the attention path;
//      we just need to splice the text_hidden_size projection in
//      front of the LM and run a 16-codebook output head instead
//      of a single-vocab lm_head.
//   2. Code predictor (5L Qwen3-style, vocab 2048): another
//      core_attn::kv_self_attn invocation.
//   3. Codec decoder: load Qwen3-TTS-Tokenizer-12Hz GGUF, build
//      8L sliding-window transformer over latent stream, pass
//      through the up-sampling stack to 24 kHz waveform. Pull
//      RVQ-codebook lookup + 1D-conv up-sample helpers into
//      core/audio_decoder.h while we're here (PLAN #53) — they
//      are reused by MiMo and the VibeVoice σ-VAE.

#include "qwen3_tts.h"

#include "core/gguf_loader.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct qwen3_tts_hp {
    // Talker (Qwen3 backbone)
    uint32_t n_layers = 28;
    uint32_t d_model = 1024;       // 0.6B; 2048 for 1.7B
    uint32_t n_heads = 16;
    uint32_t n_kv_heads = 8;
    uint32_t head_dim = 128;
    uint32_t ff_dim = 3072;        // 0.6B; 6144 for 1.7B
    uint32_t vocab_size = 3072;    // audio code vocabulary
    uint32_t text_vocab_size = 151936;
    uint32_t text_hidden_size = 2048;
    uint32_t n_code_groups = 16;   // 16 RVQ codebooks per frame
    uint32_t max_pos = 32768;
    float    rope_theta = 1000000.0f;
    float    rms_norm_eps = 1e-6f;
    bool     rope_interleaved = true;
    std::vector<uint32_t> mrope_section; // [24, 20, 20]

    // Code predictor
    uint32_t cp_n_layers = 5;
    uint32_t cp_d_model = 1024;
    uint32_t cp_vocab_size = 2048;
    uint32_t cp_max_length = 20;

    // Speaker encoder
    uint32_t spk_enc_dim = 1024;
    uint32_t spk_sample_rate = 24000;

    // Token sentinels
    uint32_t tts_bos_id = 151672;
    uint32_t tts_eos_id = 151673;
    uint32_t tts_pad_id = 151671;
    uint32_t im_start_id = 151644;
    uint32_t im_end_id = 151645;
};

} // namespace

struct qwen3_tts_context {
    qwen3_tts_context_params params{};
    int n_threads = 4;

    qwen3_tts_hp hp;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    std::string codec_path;
    std::string voice_prompt_path;

    std::vector<std::string> vocab;
};

static uint32_t q3t_kv_u32(gguf_context* ctx, const char* key, uint32_t def) {
    int64_t id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_u32(ctx, id) : def;
}
static float q3t_kv_f32(gguf_context* ctx, const char* key, float def) {
    int64_t id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_f32(ctx, id) : def;
}
static bool q3t_kv_bool(gguf_context* ctx, const char* key, bool def) {
    int64_t id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_bool(ctx, id) : def;
}

extern "C" struct qwen3_tts_context_params qwen3_tts_context_default_params(void) {
    qwen3_tts_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.temperature = 0.0f;
    p.tts_steps = 0;
    return p;
}

extern "C" struct qwen3_tts_context* qwen3_tts_init_from_file(const char* path_model,
                                                              struct qwen3_tts_context_params params) {
    auto* ctx = new qwen3_tts_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ggml_context* gctx_dummy = nullptr;
    gguf_init_params gp = {/*no_alloc=*/true, &gctx_dummy};
    gguf_context* gctx = gguf_init_from_file(path_model, gp);
    if (!gctx) {
        fprintf(stderr, "qwen3_tts: failed to read GGUF '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }

    auto& hp = ctx->hp;
    hp.n_layers          = q3t_kv_u32(gctx, "qwen3tts.talker.n_layers", hp.n_layers);
    hp.d_model           = q3t_kv_u32(gctx, "qwen3tts.talker.d_model", hp.d_model);
    hp.n_heads           = q3t_kv_u32(gctx, "qwen3tts.talker.n_heads", hp.n_heads);
    hp.n_kv_heads        = q3t_kv_u32(gctx, "qwen3tts.talker.n_kv_heads", hp.n_kv_heads);
    hp.head_dim          = q3t_kv_u32(gctx, "qwen3tts.talker.head_dim", hp.head_dim);
    hp.ff_dim            = q3t_kv_u32(gctx, "qwen3tts.talker.ff_dim", hp.ff_dim);
    hp.vocab_size        = q3t_kv_u32(gctx, "qwen3tts.talker.vocab_size", hp.vocab_size);
    hp.text_vocab_size   = q3t_kv_u32(gctx, "qwen3tts.talker.text_vocab_size", hp.text_vocab_size);
    hp.text_hidden_size  = q3t_kv_u32(gctx, "qwen3tts.talker.text_hidden_size", hp.text_hidden_size);
    hp.n_code_groups     = q3t_kv_u32(gctx, "qwen3tts.talker.n_code_groups", hp.n_code_groups);
    hp.max_pos           = q3t_kv_u32(gctx, "qwen3tts.talker.max_pos", hp.max_pos);
    hp.rope_theta        = q3t_kv_f32(gctx, "qwen3tts.talker.rope_theta", hp.rope_theta);
    hp.rms_norm_eps      = q3t_kv_f32(gctx, "qwen3tts.talker.rms_norm_eps", hp.rms_norm_eps);
    hp.rope_interleaved  = q3t_kv_bool(gctx, "qwen3tts.talker.rope_interleaved", hp.rope_interleaved);

    {
        const int mr_key = gguf_find_key(gctx, "qwen3tts.talker.mrope_section");
        if (mr_key >= 0) {
            const int n = gguf_get_arr_n(gctx, mr_key);
            const auto* d = (const uint32_t*)gguf_get_arr_data(gctx, mr_key);
            hp.mrope_section.assign(d, d + n);
        }
    }

    hp.cp_n_layers   = q3t_kv_u32(gctx, "qwen3tts.code_pred.n_layers", hp.cp_n_layers);
    hp.cp_d_model    = q3t_kv_u32(gctx, "qwen3tts.code_pred.d_model", hp.cp_d_model);
    hp.cp_vocab_size = q3t_kv_u32(gctx, "qwen3tts.code_pred.vocab_size", hp.cp_vocab_size);
    hp.cp_max_length = q3t_kv_u32(gctx, "qwen3tts.code_pred.max_length", hp.cp_max_length);

    hp.spk_enc_dim     = q3t_kv_u32(gctx, "qwen3tts.speaker.enc_dim", hp.spk_enc_dim);
    hp.spk_sample_rate = q3t_kv_u32(gctx, "qwen3tts.speaker.sample_rate", hp.spk_sample_rate);

    hp.tts_bos_id  = q3t_kv_u32(gctx, "qwen3tts.tts_bos_token_id", hp.tts_bos_id);
    hp.tts_eos_id  = q3t_kv_u32(gctx, "qwen3tts.tts_eos_token_id", hp.tts_eos_id);
    hp.tts_pad_id  = q3t_kv_u32(gctx, "qwen3tts.tts_pad_token_id", hp.tts_pad_id);
    hp.im_start_id = q3t_kv_u32(gctx, "qwen3tts.im_start_token_id", hp.im_start_id);
    hp.im_end_id   = q3t_kv_u32(gctx, "qwen3tts.im_end_token_id", hp.im_end_id);

    int tok_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
    if (tok_key >= 0) {
        int n = gguf_get_arr_n(gctx, tok_key);
        ctx->vocab.resize(n);
        for (int i = 0; i < n; i++) {
            const char* s = gguf_get_arr_str(gctx, tok_key, i);
            if (s) ctx->vocab[i] = s;
        }
    }
    gguf_free(gctx);

    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "qwen3_tts: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ctx->backend_cpu;
    if (!ctx->backend) ctx->backend = ctx->backend_cpu;

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend_cpu, "qwen3_tts", wl)) {
        fprintf(stderr, "qwen3_tts: failed to load weights from '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;
    ctx->tensors = std::move(wl.tensors);

    if (params.verbosity >= 1) {
        fprintf(stderr,
                "qwen3_tts: loaded %zu tensors  talker=%uL/%u  code_groups=%u  text_vocab=%u\n",
                ctx->tensors.size(), hp.n_layers, hp.d_model, hp.n_code_groups, hp.text_vocab_size);
    }
    return ctx;
}

extern "C" int qwen3_tts_set_codec_path(struct qwen3_tts_context* ctx, const char* path) {
    if (!ctx || !path) return -1;
    ctx->codec_path = path;
    return 0;
}

extern "C" int qwen3_tts_set_voice_prompt(struct qwen3_tts_context* ctx, const char* wav_path) {
    if (!ctx) return -1;
    ctx->voice_prompt_path = wav_path ? wav_path : "";
    return 0;
}

extern "C" float* qwen3_tts_synthesize(struct qwen3_tts_context* /*ctx*/,
                                       const char* /*text*/,
                                       int* out_n_samples) {
    // PLAN #52 — talker LM forward, code_predictor head, codec
    // decode are all still unimplemented. Return nullptr so callers
    // can branch on it.
    if (out_n_samples) *out_n_samples = 0;
    fprintf(stderr,
            "qwen3_tts: synthesise called but talker/codec forward "
            "passes are not yet implemented (PLAN #52)\n");
    return nullptr;
}

extern "C" void qwen3_tts_pcm_free(float* pcm) {
    free(pcm);
}

extern "C" void qwen3_tts_free(struct qwen3_tts_context* ctx) {
    if (!ctx) return;
    if (ctx->buf_w) ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->ctx_w) ggml_free(ctx->ctx_w);
    if (ctx->backend && ctx->backend != ctx->backend_cpu) ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu) ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" void qwen3_tts_set_n_threads(struct qwen3_tts_context* ctx, int n_threads) {
    if (!ctx || n_threads <= 0) return;
    ctx->n_threads = n_threads;
    if (ctx->backend_cpu) ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
}
