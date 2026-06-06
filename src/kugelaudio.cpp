// kugelaudio.cpp — KugelAudio-0-Open TTS runtime.
//
// Architecture overview (see kugelaudio.h for full description):
//   Qwen2.5-7B LM → DiT diffusion head → acoustic VAE decoder → 24 kHz PCM.
//
// This implementation reuses core/ primitives:
//   core_attn  — Qwen2.5 GQA attention with KV cache
//   core_ffn   — SwiGLU FFN (LM + diffusion head)
//
// DPM-Solver++ SDE with v-prediction is implemented inline following the
// vibevoice.cpp pattern, extended for the SDE noise injection variant.

#include "kugelaudio.h"

#include "core/attention.h"
#include "core/bpe.h"
#include "core/conv.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/torch_rng.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

// ── Debug ───────────────────────────────────────────────────────────────────
// Trace-level debug: set KUGELAUDIO_DEBUG=1 to see per-layer/per-step shapes.
// Normal verbosity (params.verbosity >= 1) still prints load + synthesis summary.
static bool kugelaudio_debug_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("KUGELAUDIO_DEBUG");
        cached = (e && (e[0] == '1' || e[0] == 't' || e[0] == 'T')) ? 1 : 0;
    }
    return cached != 0;
}
#define KUGELAUDIO_TRACE(...) do { if (kugelaudio_debug_enabled()) fprintf(stderr, __VA_ARGS__); } while(0)

// ── Constants ───────────────────────────────────────────────────────────────

static constexpr int KUGELAUDIO_SPEECH_START_ID    = 151652;
static constexpr int KUGELAUDIO_SPEECH_END_ID      = 151653;
static constexpr int KUGELAUDIO_SPEECH_DIFFUSION_ID = 151654;
static constexpr int KUGELAUDIO_EOS_TOKEN_ID       = 151643;

// ── DPM-Solver++ SDE schedule ──────────────────────────────────────────────

struct dpm_sde_schedule {
    int num_train_steps;
    int num_inference_steps;
    std::vector<float> alphas_cumprod;
    std::vector<float> sigmas;    // sqrt((1-ac)/ac) — diffusers sigma convention
    std::vector<int>   timesteps;
};

static dpm_sde_schedule make_dpm_sde_schedule(int num_inference_steps, int num_train_steps = 1000) {
    dpm_sde_schedule s;
    s.num_train_steps = num_train_steps;
    s.num_inference_steps = num_inference_steps;

    // Cosine beta schedule
    float offset = 0.008f;
    auto alpha_bar_fn = [&](float t) -> float {
        float frac = (t + offset) / (1.0f + offset);
        float val = cosf(frac * (float)M_PI * 0.5f);
        return val * val;
    };

    s.alphas_cumprod.resize(num_train_steps);
    s.sigmas.resize(num_train_steps);
    float a_prod = 1.0f;
    for (int i = 0; i < num_train_steps; i++) {
        float t1 = (float)i / (float)num_train_steps;
        float t2 = (float)(i + 1) / (float)num_train_steps;
        float beta = std::min(1.0f - alpha_bar_fn(t2) / alpha_bar_fn(t1), 0.999f);
        a_prod *= (1.0f - beta);
        s.alphas_cumprod[i] = a_prod;
        s.sigmas[i] = sqrtf((1.0f - a_prod) / a_prod);
    }

    // Linspace timesteps
    s.timesteps.resize(num_inference_steps);
    for (int i = 0; i < num_inference_steps; i++) {
        float val = (float)(num_train_steps - 1) * (float)(num_inference_steps - i)
                    / (float)num_inference_steps;
        s.timesteps[i] = (int)roundf(val);
    }

    return s;
}

// sigma → (alpha_t, sigma_t) conversion
static void sigma_to_alpha_sigma(float sigma, float& out_alpha, float& out_sigma) {
    out_alpha = 1.0f / sqrtf(sigma * sigma + 1.0f);
    out_sigma = sigma * out_alpha;
}

// Convert v-prediction to x0
static void v_to_x0(const float* x_t, const float* v, float* x0, int n,
                    float alpha_t, float sigma_t) {
    for (int i = 0; i < n; i++)
        x0[i] = alpha_t * x_t[i] - sigma_t * v[i];
}

// DPM-Solver++ SDE 1st order
static void dpm_sde_first_order(const dpm_sde_schedule& sched, int step_idx,
                                float* x, const float* x0_pred, const float* noise, int n) {
    float sigma_cur = sched.sigmas[sched.timesteps[step_idx]];
    int t_next = (step_idx + 1 < sched.num_inference_steps)
                     ? sched.timesteps[step_idx + 1] : -1;
    float sigma_next = (t_next >= 0) ? sched.sigmas[t_next] : 0.0f;

    float alpha_cur, sig_cur, alpha_next, sig_next;
    sigma_to_alpha_sigma(sigma_cur, alpha_cur, sig_cur);
    if (t_next >= 0) sigma_to_alpha_sigma(sigma_next, alpha_next, sig_next);
    else { alpha_next = 1.0f; sig_next = 0.0f; }

    float lambda_cur = logf(alpha_cur / sig_cur);
    float lambda_next = (t_next >= 0) ? logf(alpha_next / sig_next) : 20.0f;
    float h = lambda_next - lambda_cur;

    float exp_neg_h = expf(-h);
    float coeff_x = (sig_next / sig_cur) * exp_neg_h;
    float coeff_x0 = alpha_next * (1.0f - expf(-2.0f * h));
    float coeff_noise = sig_next * sqrtf(std::max(0.0f, 1.0f - expf(-2.0f * h)));

    for (int i = 0; i < n; i++)
        x[i] = coeff_x * x[i] + coeff_x0 * x0_pred[i] + coeff_noise * noise[i];
}

// DPM-Solver++ SDE 2nd order midpoint
static void dpm_sde_second_order(const dpm_sde_schedule& sched, int step_idx,
                                 float* x, const float* x0_cur, const float* x0_prev,
                                 int prev_step_idx, const float* noise, int n) {
    float sigma_cur = sched.sigmas[sched.timesteps[step_idx]];
    int t_next = (step_idx + 1 < sched.num_inference_steps)
                     ? sched.timesteps[step_idx + 1] : -1;
    float sigma_next = (t_next >= 0) ? sched.sigmas[t_next] : 0.0f;
    float sigma_prev = sched.sigmas[sched.timesteps[prev_step_idx]];

    float alpha_cur, sig_cur, alpha_next, sig_next, alpha_prev, sig_prev;
    sigma_to_alpha_sigma(sigma_cur, alpha_cur, sig_cur);
    if (t_next >= 0) sigma_to_alpha_sigma(sigma_next, alpha_next, sig_next);
    else { alpha_next = 1.0f; sig_next = 0.0f; }
    sigma_to_alpha_sigma(sigma_prev, alpha_prev, sig_prev);

    float lambda_cur = logf(alpha_cur / sig_cur);
    float lambda_next = (t_next >= 0) ? logf(alpha_next / sig_next) : 20.0f;
    float lambda_prev = logf(alpha_prev / sig_prev);

    float h = lambda_next - lambda_cur;
    float h_0 = lambda_cur - lambda_prev;
    float r = h_0 / h;

    float exp_neg_h = expf(-h);
    float coeff_x = (sig_next / sig_cur) * exp_neg_h;
    float eh_term = 1.0f - expf(-2.0f * h);
    float coeff_noise = sig_next * sqrtf(std::max(0.0f, eh_term));

    for (int i = 0; i < n; i++) {
        float D0 = x0_cur[i];
        float D1 = (1.0f / r) * (x0_cur[i] - x0_prev[i]);
        x[i] = coeff_x * x[i]
             + alpha_next * eh_term * D0
             + 0.5f * alpha_next * eh_term * D1
             + coeff_noise * noise[i];
    }
}

// ── Sinusoidal timestep embedding ──────────────────────────────────────────

static void compute_sinusoidal_embed(float t, float* out, int dim = 256) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = expf(-logf(10000.0f) * (float)i / (float)half);
        float arg = t * freq;
        out[i] = cosf(arg);
        out[half + i] = sinf(arg);
    }
}

// ── Model / hparams ────────────────────────────────────────────────────────

struct kugelaudio_hparams {
    int d_lm = 3584;
    int n_lm_layers = 28;
    int n_heads = 28;
    int n_kv_heads = 4;
    int d_ffn = 18944;
    int vocab_size = 152064;
    int head_dim = 128;
    float rope_theta = 1000000.0f;
    float rms_norm_eps = 1e-6f;
    int vae_dim_acoustic = 64;
    int vae_dim_semantic = 128;

    // Diffusion head
    int head_layers = 4;
    float head_ffn_ratio = 3.0f;
    int latent_size = 64;
    int ddpm_num_steps = 1000;
    int ddpm_inference_steps = 20;
    float diff_norm_eps = 1e-5f;

    // Decoder
    int n_decoder_stages = 7;
    int dec_n_filters = 32;
    int total_upsample = 3200;
    float at_norm_eps = 1e-5f;
    std::vector<int> decoder_ratios;
    std::vector<int> decoder_depths;
    std::vector<int> encoder_ratios;
    std::vector<int> encoder_depths;
    int has_encoders = 0;

    // Special token IDs
    int speech_start_id = KUGELAUDIO_SPEECH_START_ID;
    int speech_end_id = KUGELAUDIO_SPEECH_END_ID;
    int speech_diffusion_id = KUGELAUDIO_SPEECH_DIFFUSION_ID;
    int eos_token_id = KUGELAUDIO_EOS_TOKEN_ID;

    // Scaling
    float speech_scaling_factor = 1.0f;
    float speech_bias_factor = 0.0f;
};

struct kugelaudio_model {
    kugelaudio_hparams hp;
    std::map<std::string, ggml_tensor*> tensors;
    std::vector<std::string> vocab;
};

// ── Context ────────────────────────────────────────────────────────────────

struct kugelaudio_context {
    kugelaudio_model model;
    kugelaudio_context_params params;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_backend_buffer_t buf_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context* weight_ctx = nullptr;

    // KV cache for positive (main) LM path
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;
    int kv_n_used = 0;

    // KV cache for negative (CFG) path
    ggml_context* kv_neg_ctx = nullptr;
    ggml_backend_buffer_t kv_neg_buf = nullptr;
    ggml_tensor* kv_neg_k = nullptr;
    ggml_tensor* kv_neg_v = nullptr;

    std::vector<uint8_t> compute_meta;

    // Voice cache
    std::vector<float> voice_acoustic_mean; // [n_voice_frames * vae_dim]
    int n_voice_frames = 0;

    // RNG for diffusion noise
    crispasr::core::mt19937_state rng;
};

// ── Tensor lookup helpers ──────────────────────────────────────────────────

static ggml_tensor* T(kugelaudio_context* ctx, const char* name) {
    return core_gguf::require(ctx->model.tensors, name, "kugelaudio");
}

static ggml_tensor* Topt(kugelaudio_context* ctx, const char* name) {
    return core_gguf::try_get(ctx->model.tensors, name);
}

// ── Public API ─────────────────────────────────────────────────────────────

extern "C" struct kugelaudio_context_params kugelaudio_context_default_params(void) {
    kugelaudio_context_params p;
    p.n_threads = 4;
    p.max_new_tokens = 2048;
    p.verbosity = 1;
    p.use_gpu = true;
    p.tts_steps = 20;
    p.cfg_scale = 3.0f;
    p.seed = 0;
    p.flash_attn = true;
    return p;
}

extern "C" void kugelaudio_set_tts_steps(struct kugelaudio_context* ctx, int steps) {
    if (!ctx) return;
    ctx->params.tts_steps = std::clamp(steps, 4, 100);
}

extern "C" void kugelaudio_set_cfg_scale(struct kugelaudio_context* ctx, float scale) {
    if (ctx) ctx->params.cfg_scale = scale;
}

extern "C" void kugelaudio_set_seed(struct kugelaudio_context* ctx, uint32_t seed) {
    if (!ctx) return;
    ctx->params.seed = seed;
    if (seed != 0)
        crispasr::core::mt19937_seed(ctx->rng, seed);
}

// ── Init ───────────────────────────────────────────────────────────────────

extern "C" struct kugelaudio_context* kugelaudio_init_from_file(
    const char* path_model, struct kugelaudio_context_params params) {

    auto* ctx = new kugelaudio_context();
    ctx->params = params;
    auto& m = ctx->model;
    auto& hp = m.hp;

    // ── Pass 1: metadata ────────────────────────────────────────────────
    gguf_context* gctx = core_gguf::open_metadata(path_model);
    if (!gctx) {
        delete ctx;
        return nullptr;
    }

    hp.d_lm = core_gguf::kv_u32(gctx, "kugelaudio.d_lm", 3584);
    hp.n_lm_layers = core_gguf::kv_u32(gctx, "kugelaudio.n_lm_layers", 28);
    hp.n_heads = core_gguf::kv_u32(gctx, "kugelaudio.n_heads", 28);
    hp.n_kv_heads = core_gguf::kv_u32(gctx, "kugelaudio.n_kv_heads", 4);
    hp.d_ffn = core_gguf::kv_u32(gctx, "kugelaudio.d_ffn", 18944);
    hp.vocab_size = core_gguf::kv_u32(gctx, "kugelaudio.vocab_size", 152064);
    hp.head_dim = core_gguf::kv_u32(gctx, "kugelaudio.head_dim", 128);
    hp.rope_theta = core_gguf::kv_f32(gctx, "kugelaudio.rope_theta", 1000000.0f);
    hp.rms_norm_eps = core_gguf::kv_f32(gctx, "kugelaudio.rms_norm_eps", 1e-6f);
    hp.vae_dim_acoustic = core_gguf::kv_u32(gctx, "kugelaudio.vae_dim_acoustic", 64);
    hp.vae_dim_semantic = core_gguf::kv_u32(gctx, "kugelaudio.vae_dim_semantic", 128);
    hp.head_layers = core_gguf::kv_u32(gctx, "kugelaudio.head_layers", 4);
    hp.head_ffn_ratio = core_gguf::kv_f32(gctx, "kugelaudio.head_ffn_ratio", 3.0f);
    hp.latent_size = core_gguf::kv_u32(gctx, "kugelaudio.latent_size", 64);
    hp.ddpm_num_steps = core_gguf::kv_u32(gctx, "kugelaudio.ddpm_num_steps", 1000);
    hp.ddpm_inference_steps = core_gguf::kv_u32(gctx, "kugelaudio.ddpm_inference_steps", 20);
    hp.diff_norm_eps = core_gguf::kv_f32(gctx, "kugelaudio.diff_norm_eps", 1e-5f);
    hp.n_decoder_stages = core_gguf::kv_u32(gctx, "kugelaudio.n_decoder_stages", 7);
    hp.dec_n_filters = core_gguf::kv_u32(gctx, "kugelaudio.dec_n_filters", 32);
    hp.total_upsample = core_gguf::kv_u32(gctx, "kugelaudio.total_upsample", 3200);
    hp.at_norm_eps = core_gguf::kv_f32(gctx, "kugelaudio.at_norm_eps", 1e-5f);
    hp.has_encoders = core_gguf::kv_u32(gctx, "kugelaudio.has_encoders", 0);
    hp.speech_start_id = core_gguf::kv_u32(gctx, "kugelaudio.speech_start_id", KUGELAUDIO_SPEECH_START_ID);
    hp.speech_end_id = core_gguf::kv_u32(gctx, "kugelaudio.speech_end_id", KUGELAUDIO_SPEECH_END_ID);
    hp.speech_diffusion_id = core_gguf::kv_u32(gctx, "kugelaudio.speech_diffusion_id", KUGELAUDIO_SPEECH_DIFFUSION_ID);
    hp.eos_token_id = core_gguf::kv_u32(gctx, "kugelaudio.eos_token_id", KUGELAUDIO_EOS_TOKEN_ID);

    // Read arrays
    auto read_int_array = [&](const char* key, std::vector<int>& out, const std::vector<int>& defaults) {
        int k = gguf_find_key(gctx, key);
        if (k >= 0) {
            int n = gguf_get_arr_n(gctx, k);
            out.resize(n);
            for (int i = 0; i < n; i++)
                out[i] = ((const int32_t*)gguf_get_arr_data(gctx, k))[i];
        } else {
            out = defaults;
        }
    };

    read_int_array("kugelaudio.decoder_ratios", hp.decoder_ratios, {8, 5, 5, 4, 2, 2});
    read_int_array("kugelaudio.decoder_depths", hp.decoder_depths, {8, 3, 3, 3, 3, 3, 3});
    read_int_array("kugelaudio.encoder_ratios", hp.encoder_ratios, {8, 5, 5, 4, 2, 2});
    read_int_array("kugelaudio.encoder_depths", hp.encoder_depths, {3, 3, 3, 3, 3, 3, 8});

    // Load vocabulary
    const int tok_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
    if (tok_key >= 0) {
        int n = gguf_get_arr_n(gctx, tok_key);
        m.vocab.resize(n);
        for (int i = 0; i < n; i++) {
            const char* s = gguf_get_arr_str(gctx, tok_key, i);
            if (s) m.vocab[i] = s;
        }
    }

    gguf_free(gctx);

    if (params.verbosity >= 1) {
        fprintf(stderr, "kugelaudio: d_lm=%d, layers=%d, heads=%d/%d, ffn=%d, vocab=%d\n",
                hp.d_lm, hp.n_lm_layers, hp.n_heads, hp.n_kv_heads, hp.d_ffn, hp.vocab_size);
        fprintf(stderr, "kugelaudio: vae_dim=%d, latent=%d, diff_steps=%d, head_layers=%d\n",
                hp.vae_dim_acoustic, hp.latent_size, hp.ddpm_inference_steps, hp.head_layers);
        fprintf(stderr, "kugelaudio: decoder stages=%d, upsample=%dx\n",
                hp.n_decoder_stages, hp.total_upsample);
    }

    // ── Backend init ────────────────────────────────────────────────────
    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ggml_backend_cpu_init();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, std::max(1, params.n_threads));
    if (ctx->backend && ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, std::max(1, params.n_threads));
    if (!ctx->backend) {
        delete ctx;
        return nullptr;
    }

    // ── Pass 2: weight loading ──────────────────────────────────────────
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "kugelaudio", wl)) {
        ggml_backend_free(ctx->backend);
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            ggml_backend_free(ctx->backend_cpu);
        delete ctx;
        return nullptr;
    }
    ctx->weight_ctx = wl.ctx;
    ctx->buf = wl.buf;
    ctx->buf_cpu = wl.buf_cpu;
    m.tensors = std::move(wl.tensors);

    // Create scheduler
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 65536, false, false);
    }
    ctx->compute_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(65536, false));

    // Read scaling factors from tensor data
    auto* sf = Topt(ctx, "model.speech_scaling_factor");
    auto* bf = Topt(ctx, "model.speech_bias_factor");
    if (sf && bf) {
        // These are scalar tensors; read their values
        ggml_backend_tensor_get(sf, &hp.speech_scaling_factor, 0, sizeof(float));
        ggml_backend_tensor_get(bf, &hp.speech_bias_factor, 0, sizeof(float));
    }

    // Init RNG
    crispasr::core::mt19937_seed(ctx->rng, params.seed != 0 ? params.seed : (uint32_t)time(nullptr));

    if (params.verbosity >= 1) {
        fprintf(stderr, "kugelaudio: loaded %zu tensors (backend: %s)\n",
                m.tensors.size(), ggml_backend_name(ctx->backend));
        fprintf(stderr, "kugelaudio: scaling=%.6f bias=%.6f\n",
                hp.speech_scaling_factor, hp.speech_bias_factor);
    }

    return ctx;
}

// ── Free ───────────────────────────────────────────────────────────────────

extern "C" void kugelaudio_free(struct kugelaudio_context* ctx) {
    if (!ctx) return;
    if (ctx->kv_neg_buf) ggml_backend_buffer_free(ctx->kv_neg_buf);
    if (ctx->kv_neg_ctx) ggml_free(ctx->kv_neg_ctx);
    if (ctx->kv_buf) ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx) ggml_free(ctx->kv_ctx);
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    if (ctx->buf) ggml_backend_buffer_free(ctx->buf);
    if (ctx->buf_cpu) ggml_backend_buffer_free(ctx->buf_cpu);
    if (ctx->weight_ctx) ggml_free(ctx->weight_ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

// ── Qwen2.5 byte encoder (GPT-2 style) ────────────────────────────────────

static const std::vector<int>& qwen_byte_encoder() {
    static std::vector<int> enc(256, -1);
    static bool initialized = false;
    if (initialized) return enc;

    std::vector<int> bs, cs;
    for (int b = 0x21; b <= 0x7e; ++b) { bs.push_back(b); cs.push_back(b); }
    for (int b = 0xa1; b <= 0xac; ++b) { bs.push_back(b); cs.push_back(b); }
    for (int b = 0xae; b <= 0xff; ++b) { bs.push_back(b); cs.push_back(b); }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool found = false;
        for (int x : bs) if (x == b) { found = true; break; }
        if (!found) { bs.push_back(b); cs.push_back(256 + n); ++n; }
    }
    for (size_t i = 0; i < bs.size(); ++i) enc[bs[i]] = cs[i];
    initialized = true;
    return enc;
}

// Greedy longest-match tokenizer for Qwen2.5 BPE vocabulary
static std::vector<int32_t> tokenize_text_greedy(const kugelaudio_model& m, const char* text) {
    static std::map<std::string, int> vocab_map;
    static int max_token_len = 0;
    static bool built = false;
    if (!built && !m.vocab.empty()) {
        for (int i = 0; i < (int)m.vocab.size(); i++) {
            if (!m.vocab[i].empty()) vocab_map[m.vocab[i]] = i;
            if ((int)m.vocab[i].size() > max_token_len)
                max_token_len = (int)m.vocab[i].size();
        }
        built = true;
    }

    // Convert text to GPT-2 byte-encoded string
    const auto& enc = qwen_byte_encoder();
    std::string encoded;
    for (const uint8_t* p = (const uint8_t*)text; *p; p++) {
        int cp = enc[*p];
        if (cp < 0) continue;
        if (cp < 0x80) {
            encoded += (char)cp;
        } else if (cp < 0x800) {
            encoded += (char)(0xC0 | (cp >> 6));
            encoded += (char)(0x80 | (cp & 0x3F));
        } else {
            encoded += (char)(0xE0 | (cp >> 12));
            encoded += (char)(0x80 | ((cp >> 6) & 0x3F));
            encoded += (char)(0x80 | (cp & 0x3F));
        }
    }

    // Greedy longest match
    std::vector<int32_t> ids;
    size_t pos = 0;
    while (pos < encoded.size()) {
        int best_len = 0, best_id = -1;
        int try_len = std::min(max_token_len, (int)(encoded.size() - pos));
        for (int len = try_len; len >= 1; len--) {
            auto it = vocab_map.find(encoded.substr(pos, len));
            if (it != vocab_map.end()) { best_len = len; best_id = it->second; break; }
        }
        if (best_id >= 0) { ids.push_back(best_id); pos += best_len; }
        else pos++;
    }
    return ids;
}

// ── ConvRMSNorm: transpose → rms_norm → mul(weight) → transpose ───────────

static ggml_tensor* build_conv_rms_norm(ggml_context* ctx0, ggml_tensor* x, ggml_tensor* w, float eps = 1e-5f) {
    // x: [C, T], w: [C]. ggml_rms_norm normalizes over ne[0]=C. OK.
    // ggml_mul(x=[C,T], w=[C]): w broadcasts over T. OK.
    // No transpose needed — ggml_rms_norm over ne[0] is what ConvRMSNorm does.
    x = ggml_rms_norm(ctx0, x, eps);
    if (w) x = ggml_mul(ctx0, x, w);
    return x;
}

// ── Causal Conv1d ──────────────────────────────────────────────────────────

static ggml_tensor* build_causal_conv1d(ggml_context* ctx0, ggml_tensor* x,
                                        ggml_tensor* w, ggml_tensor* b, int stride) {
    int K = (int)w->ne[0];
    int pad_left = (K - 1) - (stride - 1);
    if (pad_left < 0) pad_left = 0;

    x = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // [C, T] → [T, C]
    int T_in = (int)x->ne[0];

    int pad_right = 0;
    if (stride > 1) {
        double n_frames = (double)(T_in - K + pad_left) / stride + 1.0;
        int ideal_length = ((int)ceil(n_frames) - 1) * stride + (K - pad_left);
        pad_right = ideal_length - T_in;
        if (pad_right < 0) pad_right = 0;
    }
    if (pad_left > 0 || pad_right > 0)
        x = ggml_pad_ext(ctx0, x, pad_left, pad_right, 0, 0, 0, 0, 0, 0);

    x = ggml_conv_1d(ctx0, w, x, stride, 0, 1);

    if (b) {
        ggml_tensor* xt = ggml_cont(ctx0, ggml_transpose(ctx0, x));
        xt = ggml_add(ctx0, xt, b);
        return xt;
    }
    return ggml_cont(ctx0, ggml_transpose(ctx0, x));
}

// ── Causal Depthwise Conv1d ────────────────────────────────────────────────

static ggml_tensor* build_causal_dw_conv1d(ggml_context* ctx0, ggml_tensor* x,
                                           ggml_tensor* w, ggml_tensor* b) {
    int K = (int)w->ne[0];
    int pad_left = K - 1;

    x = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // [C, T] → [T, C]
    if (pad_left > 0)
        x = ggml_pad_ext(ctx0, x, pad_left, 0, 0, 0, 0, 0, 0, 0);
    x = ggml_conv_1d_dw(ctx0, w, x, 1, 0, 1);
    if (ggml_n_dims(x) > 2)
        x = ggml_reshape_2d(ctx0, x, x->ne[0], x->ne[1] * x->ne[2]);
    x = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // [T, C] → [C, T]
    if (b) x = ggml_add(ctx0, x, b);
    return x;
}

// ── Transposed Conv1d (upsample) ───────────────────────────────────────────

static ggml_tensor* build_transposed_conv1d(ggml_context* ctx0, ggml_tensor* x,
                                            ggml_tensor* w, ggml_tensor* b, int stride) {
    int T_in = (int)x->ne[1];
    x = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // [C, T] → [T, C]
    x = ggml_conv_transpose_1d(ctx0, w, x, stride, 0, 1);
    x = ggml_reshape_2d(ctx0, x, x->ne[0], x->ne[1]);

    // Trim: raw = (T_in-1)*stride + K, target = T_in*stride
    int K = (int)w->ne[0];
    int T_out_raw = (T_in - 1) * stride + K;
    int T_target = T_in * stride;
    if (T_out_raw > T_target) {
        int C_out = (int)x->ne[1];
        x = ggml_view_2d(ctx0, x, T_target, C_out, x->nb[1], 0);
        x = ggml_cont(ctx0, x);
    }
    x = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // [T, C] → [C, T]
    if (b) x = ggml_add(ctx0, x, b);
    return x;
}

// ── Block1D: depthwise conv mixer + GELU FFN + layer scale ─────────────────

static ggml_tensor* build_block1d(ggml_context* ctx0, ggml_tensor* x,
                                  ggml_tensor* norm_w, ggml_tensor* dw_conv_w,
                                  ggml_tensor* dw_conv_b, ggml_tensor* gamma,
                                  ggml_tensor* ffn_norm_w, ggml_tensor* ffn_up_w,
                                  ggml_tensor* ffn_up_b, ggml_tensor* ffn_down_w,
                                  ggml_tensor* ffn_down_b, ggml_tensor* ffn_gamma,
                                  float norm_eps = 1e-5f) {
    // Mixer path
    ggml_tensor* residual = x;
    ggml_tensor* h = build_conv_rms_norm(ctx0, x, norm_w, norm_eps);
    h = build_causal_dw_conv1d(ctx0, h, dw_conv_w, dw_conv_b);
    if (gamma) h = ggml_mul(ctx0, h, gamma);
    x = ggml_add(ctx0, residual, h);

    // FFN path: GELU activation (matches KugelAudio's FFN class)
    residual = x;
    h = build_conv_rms_norm(ctx0, x, ffn_norm_w, norm_eps);
    h = ggml_mul_mat(ctx0, ffn_up_w, h);
    if (ffn_up_b) h = ggml_add(ctx0, h, ffn_up_b);
    h = ggml_gelu(ctx0, h);
    h = ggml_mul_mat(ctx0, ffn_down_w, h);
    if (ffn_down_b) h = ggml_add(ctx0, h, ffn_down_b);
    if (ffn_gamma) h = ggml_mul(ctx0, h, ffn_gamma);
    x = ggml_add(ctx0, residual, h);

    return x;
}

// ── Build diffusion head graph ─────────────────────────────────────────────

static ggml_cgraph* build_pred_head_graph(kugelaudio_context* ctx, int n_frames) {
    auto& hp = ctx->model.hp;
    auto& ts = ctx->model.tensors;
    auto G = [&](const std::string& name) -> ggml_tensor* {
        auto it = ts.find(name);
        return it != ts.end() ? it->second : nullptr;
    };

    int vae_dim = hp.vae_dim_acoustic;
    int d_lm = hp.d_lm;

    size_t mem = ctx->compute_meta.size();
    ggml_init_params ip = {mem, ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    ggml_tensor* noisy = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, vae_dim, n_frames);
    ggml_set_name(noisy, "pred_noisy");
    ggml_set_input(noisy);

    ggml_tensor* t_sin = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 256);
    ggml_set_name(t_sin, "pred_t_sin");
    ggml_set_input(t_sin);

    ggml_tensor* condition = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_lm, n_frames);
    ggml_set_name(condition, "pred_condition");
    ggml_set_input(condition);

    // Time embedding MLP: sinusoidal[256] → Linear → SiLU → Linear → [d_lm]
    ggml_tensor* t_emb = ggml_mul_mat(ctx0, G("model.prediction_head.t_embedder.mlp.0.weight"), t_sin);
    t_emb = ggml_silu(ctx0, t_emb);
    t_emb = ggml_mul_mat(ctx0, G("model.prediction_head.t_embedder.mlp.2.weight"), t_emb);

    // Project noisy latent: [vae_dim, n_frames] → [d_lm, n_frames]
    ggml_tensor* x = ggml_mul_mat(ctx0, G("model.prediction_head.noisy_images_proj.weight"), noisy);

    // Project condition: [d_lm, n_frames] → [d_lm, n_frames]
    ggml_tensor* cond = ggml_mul_mat(ctx0, G("model.prediction_head.cond_proj.weight"), condition);

    // Combined c = cond + t_emb (t_emb broadcasts over n_frames)
    KUGELAUDIO_TRACE("pred: t_emb=[%lld,%lld] x=[%lld,%lld] cond=[%lld,%lld]\n",
            (long long)t_emb->ne[0],(long long)t_emb->ne[1],
            (long long)x->ne[0],(long long)x->ne[1],
            (long long)cond->ne[0],(long long)cond->ne[1]);
    ggml_tensor* c = ggml_add(ctx0, cond, t_emb);
    KUGELAUDIO_TRACE("pred: c=[%lld,%lld] OK\n", (long long)c->ne[0],(long long)c->ne[1]);

    // 4 HeadLayers: AdaLN(3-way) + SwiGLU FFN
    for (int i = 0; i < hp.head_layers; i++) {
        char base[128];
        snprintf(base, sizeof(base), "model.prediction_head.layers.%d", i);
        KUGELAUDIO_TRACE("pred: layer %d x=[%lld,%lld]\n", i, (long long)x->ne[0],(long long)x->ne[1]);

        // AdaLN: SiLU(c) → Linear → chunk(3) → (shift, scale, gate)
        ggml_tensor* c_silu = ggml_silu(ctx0, c);
        ggml_tensor* adaln_w = G(std::string(base) + ".adaLN_modulation.1.weight");
        KUGELAUDIO_TRACE("pred: L%d adaln_w=[%lld,%lld] c_silu=[%lld,%lld]\n", i,
                (long long)adaln_w->ne[0],(long long)adaln_w->ne[1],
                (long long)c_silu->ne[0],(long long)c_silu->ne[1]);
        ggml_tensor* adaln_out = ggml_mul_mat(ctx0, adaln_w, c_silu);
        KUGELAUDIO_TRACE("pred: L%d adaln_out=[%lld,%lld]\n", i,
                (long long)adaln_out->ne[0],(long long)adaln_out->ne[1]);
        size_t nb1 = adaln_out->nb[1];
        ggml_tensor* shift = ggml_view_2d(ctx0, adaln_out, d_lm, n_frames, nb1, 0);
        ggml_tensor* scale = ggml_view_2d(ctx0, adaln_out, d_lm, n_frames, nb1, (size_t)d_lm * sizeof(float));
        ggml_tensor* gate  = ggml_view_2d(ctx0, adaln_out, d_lm, n_frames, nb1, (size_t)2 * d_lm * sizeof(float));
        KUGELAUDIO_TRACE("pred: L%d shift=[%lld,%lld] scale=[%lld,%lld] gate=[%lld,%lld]\n", i,
                (long long)shift->ne[0],(long long)shift->ne[1],
                (long long)scale->ne[0],(long long)scale->ne[1],
                (long long)gate->ne[0],(long long)gate->ne[1]);

        // RMSNorm(x) * (1 + scale) + shift
        ggml_tensor* h = ggml_rms_norm(ctx0, x, hp.diff_norm_eps);
        ggml_tensor* norm_w = G(std::string(base) + ".norm.weight");
        KUGELAUDIO_TRACE("pred: L%d h=[%lld,%lld] norm_w=[%lld,%lld]\n", i,
                (long long)h->ne[0],(long long)h->ne[1],
                (long long)norm_w->ne[0],(long long)norm_w->ne[1]);
        h = ggml_mul(ctx0, h, norm_w);
        KUGELAUDIO_TRACE("pred: L%d after norm_mul OK\n", i);
        ggml_tensor* h_scaled = ggml_mul(ctx0, h, scale);
        KUGELAUDIO_TRACE("pred: L%d after scale_mul OK\n", i);
        h = ggml_add(ctx0, h, h_scaled);
        KUGELAUDIO_TRACE("pred: L%d after add(h, h_scaled) OK\n", i);
        h = ggml_add(ctx0, h, shift);
        KUGELAUDIO_TRACE("pred: L%d after add(h, shift) OK\n", i);

        // SwiGLU FFN
        h = core_ffn::swiglu(ctx0, h,
                             G(std::string(base) + ".ffn.gate_proj.weight"),
                             G(std::string(base) + ".ffn.up_proj.weight"),
                             G(std::string(base) + ".ffn.down_proj.weight"));
        KUGELAUDIO_TRACE("pred: L%d after swiglu OK\n", i);

        // gate * ffn + residual
        KUGELAUDIO_TRACE("pred: L%d gate_mul h=[%lld,%lld,%lld,%lld] gate=[%lld,%lld,%lld,%lld]\n", i,
                (long long)h->ne[0],(long long)h->ne[1],(long long)h->ne[2],(long long)h->ne[3],
                (long long)gate->ne[0],(long long)gate->ne[1],(long long)gate->ne[2],(long long)gate->ne[3]);
        h = ggml_mul(ctx0, h, gate);
        KUGELAUDIO_TRACE("pred: L%d gate_mul OK\n", i);
        x = ggml_add(ctx0, x, h);
    }

    // Final layer: AdaLN(2-way) + Linear → [vae_dim]
    {
        KUGELAUDIO_TRACE("pred: final layer x=[%lld,%lld]\n", (long long)x->ne[0],(long long)x->ne[1]);
        ggml_tensor* c_silu_f = ggml_silu(ctx0, c);
        ggml_tensor* final_adaln_w = G("model.prediction_head.final_layer.adaLN_modulation.1.weight");
        KUGELAUDIO_TRACE("pred: final adaln_w=[%lld,%lld] c_silu=[%lld,%lld]\n",
                (long long)final_adaln_w->ne[0],(long long)final_adaln_w->ne[1],
                (long long)c_silu_f->ne[0],(long long)c_silu_f->ne[1]);
        ggml_tensor* adaln_out = ggml_mul_mat(ctx0, final_adaln_w, c_silu_f);
        KUGELAUDIO_TRACE("pred: final adaln_out=[%lld,%lld]\n", (long long)adaln_out->ne[0],(long long)adaln_out->ne[1]);
        size_t nb1_f = adaln_out->nb[1];
        ggml_tensor* shift = ggml_view_2d(ctx0, adaln_out, d_lm, n_frames, nb1_f, 0);
        ggml_tensor* scale = ggml_view_2d(ctx0, adaln_out, d_lm, n_frames, nb1_f,
                                           (size_t)d_lm * sizeof(float));
        KUGELAUDIO_TRACE("pred: final shift=[%lld,%lld] scale=[%lld,%lld]\n",
                (long long)shift->ne[0],(long long)shift->ne[1],
                (long long)scale->ne[0],(long long)scale->ne[1]);

        // RMSNorm without affine (elementwise_affine=False in Python)
        ggml_tensor* h = ggml_rms_norm(ctx0, x, hp.diff_norm_eps);
        KUGELAUDIO_TRACE("pred: final rms h=[%lld,%lld]\n", (long long)h->ne[0],(long long)h->ne[1]);
        ggml_tensor* h_scaled = ggml_mul(ctx0, h, scale);
        KUGELAUDIO_TRACE("pred: final mul(h,scale) OK\n");
        h = ggml_add(ctx0, h, h_scaled);
        KUGELAUDIO_TRACE("pred: final add(h,h_scaled) OK\n");
        h = ggml_add(ctx0, h, shift);
        KUGELAUDIO_TRACE("pred: final add(h,shift) OK\n");

        ggml_tensor* final_lin_w = G("model.prediction_head.final_layer.linear.weight");
        KUGELAUDIO_TRACE("pred: final linear_w=[%lld,%lld] h=[%lld,%lld]\n",
                (long long)final_lin_w->ne[0],(long long)final_lin_w->ne[1],
                (long long)h->ne[0],(long long)h->ne[1]);
        ggml_tensor* output = ggml_mul_mat(ctx0, final_lin_w, h);
        KUGELAUDIO_TRACE("pred: final output=[%lld,%lld] OK\n", (long long)output->ne[0],(long long)output->ne[1]);
        ggml_set_name(output, "pred_output");
        ggml_set_output(output);
        ggml_build_forward_expand(gf, output);
    }

    return gf;
}

// ── Build VAE decoder graph ────────────────────────────────────────────────

static ggml_cgraph* build_vae_decoder_graph(kugelaudio_context* ctx, int n_frames) {
    auto& hp = ctx->model.hp;
    auto& ts = ctx->model.tensors;
    auto G = [&](const std::string& name) -> ggml_tensor* {
        auto it = ts.find(name);
        return it != ts.end() ? it->second : nullptr;
    };

    size_t mem = ctx->compute_meta.size();
    ggml_init_params ip = {mem, ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);

    ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.vae_dim_acoustic, n_frames);
    ggml_set_name(inp, "dec_latent");
    ggml_set_input(inp);

    ggml_tensor* h = inp;
    float eps = hp.at_norm_eps;

    // Decoder depths = reversed encoder depths (already stored in hp.decoder_depths)
    const auto& depths = hp.decoder_depths;
    const auto& ratios = hp.decoder_ratios;

    // Stage 0: stem conv (SConv1d, stride=1)
    KUGELAUDIO_TRACE("dec: stem inp=[%lld,%lld]\n", (long long)h->ne[0],(long long)h->ne[1]);
    h = build_causal_conv1d(ctx0, h,
        G("model.acoustic_tokenizer.decoder.upsample_layers.0.0.conv.conv.weight"),
        G("model.acoustic_tokenizer.decoder.upsample_layers.0.0.conv.conv.bias"), 1);
    KUGELAUDIO_TRACE("dec: stem out=[%lld,%lld]\n", (long long)h->ne[0],(long long)h->ne[1]);

    // Stage 0 blocks
    for (int bi = 0; bi < depths[0]; bi++) {
        char base[256];
        snprintf(base, sizeof(base), "model.acoustic_tokenizer.decoder.stages.0.%d", bi);
        KUGELAUDIO_TRACE("dec: s0.b%d h=[%lld,%lld]\n", bi, (long long)h->ne[0],(long long)h->ne[1]);
        h = build_block1d(ctx0, h,
            G(std::string(base) + ".norm.weight"),
            G(std::string(base) + ".mixer.conv.conv.conv.weight"),
            G(std::string(base) + ".mixer.conv.conv.conv.bias"),
            G(std::string(base) + ".gamma"),
            G(std::string(base) + ".ffn_norm.weight"),
            G(std::string(base) + ".ffn.linear1.weight"),
            G(std::string(base) + ".ffn.linear1.bias"),
            G(std::string(base) + ".ffn.linear2.weight"),
            G(std::string(base) + ".ffn.linear2.bias"),
            G(std::string(base) + ".ffn_gamma"),
            eps);
    }
    KUGELAUDIO_TRACE("dec: stage 0 done h=[%lld,%lld]\n", (long long)h->ne[0],(long long)h->ne[1]);

    // Stages 1-6: ConvTranspose1d upsample + ConvNeXt blocks
    int n_upsample = (int)ratios.size();
    for (int si = 1; si <= n_upsample; si++) {
        char wn[256], bn[256];
        snprintf(wn, sizeof(wn), "model.acoustic_tokenizer.decoder.upsample_layers.%d.0.convtr.convtr.weight", si);
        snprintf(bn, sizeof(bn), "model.acoustic_tokenizer.decoder.upsample_layers.%d.0.convtr.convtr.bias", si);
        int stride = ratios[si - 1];
        KUGELAUDIO_TRACE("dec: stage %d upsample stride=%d h=[%lld,%lld]\n", si, stride,
                (long long)h->ne[0],(long long)h->ne[1]);
        h = build_transposed_conv1d(ctx0, h, G(wn), G(bn), stride);
        KUGELAUDIO_TRACE("dec: stage %d upsample out=[%lld,%lld]\n", si, (long long)h->ne[0],(long long)h->ne[1]);

        int n_blocks = (si < (int)depths.size()) ? depths[si] : 3;
        for (int bi = 0; bi < n_blocks; bi++) {
            char base[256];
            snprintf(base, sizeof(base), "model.acoustic_tokenizer.decoder.stages.%d.%d", si, bi);
            h = build_block1d(ctx0, h,
                G(std::string(base) + ".norm.weight"),
                G(std::string(base) + ".mixer.conv.conv.conv.weight"),
                G(std::string(base) + ".mixer.conv.conv.conv.bias"),
                G(std::string(base) + ".gamma"),
                G(std::string(base) + ".ffn_norm.weight"),
                G(std::string(base) + ".ffn.linear1.weight"),
                G(std::string(base) + ".ffn.linear1.bias"),
                G(std::string(base) + ".ffn.linear2.weight"),
                G(std::string(base) + ".ffn.linear2.bias"),
                G(std::string(base) + ".ffn_gamma"),
                eps);
        }
    }

    // Head conv: Conv1d → 1 channel (mono audio)
    h = build_causal_conv1d(ctx0, h,
        G("model.acoustic_tokenizer.decoder.head.conv.conv.weight"),
        G("model.acoustic_tokenizer.decoder.head.conv.conv.bias"), 1);

    ggml_set_name(h, "dec_audio");
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);

    return gf;
}

// ── Build LM decoder graph (with KV cache) ─────────────────────────────────

static ggml_cgraph* build_lm_graph(kugelaudio_context* ctx, int n_tokens, int n_past,
                                   bool output_hidden = false) {
    auto& hp = ctx->model.hp;
    auto& ts = ctx->model.tensors;
    auto G = [&](const std::string& name) -> ggml_tensor* {
        auto it = ts.find(name);
        return it != ts.end() ? it->second : nullptr;
    };

    size_t mem = ctx->compute_meta.size();
    ggml_init_params ip = {mem, ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.d_lm, n_tokens);
    ggml_set_name(embeds, "lm_input");
    ggml_set_input(embeds);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor* causal_mask = nullptr;
    if (n_tokens > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_past + n_tokens, n_tokens);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    ggml_tensor* cur = embeds;
    for (int il = 0; il < hp.n_lm_layers; il++) {
        char p[128];
        snprintf(p, sizeof(p), "model.language_model.layers.%d", il);
        ggml_tensor* residual = cur;

        // Pre-RMSNorm
        cur = ggml_rms_norm(ctx0, cur, hp.rms_norm_eps);
        ggml_tensor* ln_w = G(std::string(p) + ".input_layernorm.weight");
        if (!ln_w) {
            KUGELAUDIO_TRACE("kugelaudio: MISSING %s.input_layernorm.weight\n", p);
            return nullptr;
        }
        if (il <= 1) {
            KUGELAUDIO_TRACE( "kugelaudio: L%d cur=[%lld,%lld] ln_w=[%lld,%lld]\n", il,
                    (long long)cur->ne[0], (long long)cur->ne[1],
                    (long long)ln_w->ne[0], (long long)ln_w->ne[1]);
        }
        cur = ggml_mul(ctx0, cur, ln_w);

        // Q/K/V with bias
        int T_cur = (int)cur->ne[1];
        int Lk = n_past + T_cur;

        ggml_tensor* q_w = G(std::string(p) + ".self_attn.q_proj.weight");
        ggml_tensor* k_w = G(std::string(p) + ".self_attn.k_proj.weight");
        ggml_tensor* v_w = G(std::string(p) + ".self_attn.v_proj.weight");
        ggml_tensor* q_b = G(std::string(p) + ".self_attn.q_proj.bias");
        ggml_tensor* k_b = G(std::string(p) + ".self_attn.k_proj.bias");
        ggml_tensor* v_b = G(std::string(p) + ".self_attn.v_proj.bias");
        if (il == 0) {
            KUGELAUDIO_TRACE("kugelaudio: L q_w=[%lld,%lld] k_w=[%lld,%lld] v_w=[%lld,%lld]\n",
                    (long long)q_w->ne[0], (long long)q_w->ne[1],
                    (long long)k_w->ne[0], (long long)k_w->ne[1],
                    (long long)v_w->ne[0], (long long)v_w->ne[1]);
            if (q_b) KUGELAUDIO_TRACE("kugelaudio: L q_b=[%lld,%lld] k_b=[%lld,%lld] v_b=[%lld,%lld]\n",
                    (long long)q_b->ne[0], (long long)q_b->ne[1],
                    k_b ? (long long)k_b->ne[0] : -1, k_b ? (long long)k_b->ne[1] : -1,
                    v_b ? (long long)v_b->ne[0] : -1, v_b ? (long long)v_b->ne[1] : -1);
        }
        ggml_tensor* Q = ggml_mul_mat(ctx0, q_w, cur);
        if (il <= 1) KUGELAUDIO_TRACE("kugelaudio: L Q=[%lld,%lld]\n", (long long)Q->ne[0], (long long)Q->ne[1]);
        if (q_b) { if (il<=1) KUGELAUDIO_TRACE("kugelaudio: L add Q+qb\n"); Q = ggml_add(ctx0, Q, q_b); }
        ggml_tensor* K = ggml_mul_mat(ctx0, k_w, cur);
        if (il <= 1) KUGELAUDIO_TRACE("kugelaudio: L K=[%lld,%lld]\n", (long long)K->ne[0], (long long)K->ne[1]);
        if (k_b) { if (il<=1) KUGELAUDIO_TRACE("kugelaudio: L add K+kb\n"); K = ggml_add(ctx0, K, k_b); }
        if (il <= 1) KUGELAUDIO_TRACE("kugelaudio: L K after bias OK\n");
        ggml_tensor* V = ggml_mul_mat(ctx0, v_w, cur);
        if (il <= 1) KUGELAUDIO_TRACE("kugelaudio: L V=[%lld,%lld]\n", (long long)V->ne[0], (long long)V->ne[1]);
        if (v_b) { if (il<=1) KUGELAUDIO_TRACE("kugelaudio: L add V+vb\n"); V = ggml_add(ctx0, V, v_b); }
        if (il <= 1) KUGELAUDIO_TRACE("kugelaudio: L QKV done\n");

        if (il <= 1) KUGELAUDIO_TRACE("kugelaudio: L reshape\n");
        // Reshape for GQA
        Q = ggml_reshape_3d(ctx0, Q, hp.head_dim, hp.n_heads, T_cur);
        K = ggml_reshape_3d(ctx0, K, hp.head_dim, hp.n_kv_heads, T_cur);
        V = ggml_reshape_3d(ctx0, V, hp.head_dim, hp.n_kv_heads, T_cur);

        if (il <= 1) KUGELAUDIO_TRACE("kugelaudio: L rope\n");
        // RoPE (Qwen2.5 uses NEOX layout)
        Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hp.head_dim, GGML_ROPE_TYPE_NEOX,
                          0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx0, K, positions, nullptr, hp.head_dim, GGML_ROPE_TYPE_NEOX,
                          0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        if (il <= 1) KUGELAUDIO_TRACE("kugelaudio: L kv write\n");
        // Write K, V to KV cache
        ggml_tensor* K_perm = ggml_permute(ctx0, K, 0, 2, 1, 3);
        ggml_tensor* V_perm = ggml_permute(ctx0, V, 0, 2, 1, 3);
        ggml_tensor* k_view = ggml_view_4d(ctx0, ctx->kv_k,
            hp.head_dim, T_cur, hp.n_kv_heads, 1,
            ctx->kv_k->nb[1], ctx->kv_k->nb[2], ctx->kv_k->nb[3],
            (size_t)il * ctx->kv_k->nb[3] + (size_t)n_past * ctx->kv_k->nb[1]);
        ggml_tensor* v_view = ggml_view_4d(ctx0, ctx->kv_v,
            hp.head_dim, T_cur, hp.n_kv_heads, 1,
            ctx->kv_v->nb[1], ctx->kv_v->nb[2], ctx->kv_v->nb[3],
            (size_t)il * ctx->kv_v->nb[3] + (size_t)n_past * ctx->kv_v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_perm, k_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_perm, v_view));

        if (il <= 1) KUGELAUDIO_TRACE("kugelaudio: L kv read\n");
        // Read full K, V from cache
        ggml_tensor* Kfull = ggml_cont(ctx0, ggml_view_3d(ctx0, ctx->kv_k,
            hp.head_dim, Lk, hp.n_kv_heads,
            ctx->kv_k->nb[1], ctx->kv_k->nb[2],
            (size_t)il * ctx->kv_k->nb[3]));
        ggml_tensor* Vfull = ggml_cont(ctx0, ggml_view_3d(ctx0, ctx->kv_v,
            hp.head_dim, Lk, hp.n_kv_heads,
            ctx->kv_v->nb[1], ctx->kv_v->nb[2],
            (size_t)il * ctx->kv_v->nb[3]));

        if (il <= 1) KUGELAUDIO_TRACE("kugelaudio: L flash_attn Q=[%lld,%lld,%lld] K=[%lld,%lld,%lld]\n",
                (long long)Q->ne[0],(long long)Q->ne[1],(long long)Q->ne[2],
                (long long)Kfull->ne[0],(long long)Kfull->ne[1],(long long)Kfull->ne[2]);
        // Flash attention (native GQA)
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        float scale = 1.0f / sqrtf((float)hp.head_dim);
        ggml_tensor* attn_out = ggml_flash_attn_ext(ctx0, Q, Kfull, Vfull, causal_mask, scale, 0.0f, 0.0f);

        if (il <= 1) KUGELAUDIO_TRACE("kugelaudio: L attn_out=[%lld,%lld,%lld]\n",
                (long long)attn_out->ne[0],(long long)attn_out->ne[1],(long long)attn_out->ne[2]);
        attn_out = ggml_reshape_2d(ctx0, attn_out, hp.d_lm, T_cur);
        attn_out = ggml_mul_mat(ctx0, G(std::string(p) + ".self_attn.o_proj.weight"), attn_out);
        if (il <= 1) KUGELAUDIO_TRACE("kugelaudio: L o_proj done, adding residual\n");
        cur = ggml_add(ctx0, residual, attn_out);
        if (il <= 1) KUGELAUDIO_TRACE("kugelaudio: L attn residual done\n");

        // FFN: RMSNorm + SwiGLU
        if (il <= 1) KUGELAUDIO_TRACE("kugelaudio: L FFN start\n");
        residual = cur;
        cur = ggml_rms_norm(ctx0, cur, hp.rms_norm_eps);
        cur = ggml_mul(ctx0, cur, G(std::string(p) + ".post_attention_layernorm.weight"));
        ggml_tensor* gate_w = G(std::string(p) + ".mlp.gate_proj.weight");
        ggml_tensor* up_w = G(std::string(p) + ".mlp.up_proj.weight");
        ggml_tensor* down_w = G(std::string(p) + ".mlp.down_proj.weight");
        if (il <= 1) KUGELAUDIO_TRACE("kugelaudio: L FFN gate=[%lld,%lld] up=[%lld,%lld] down=[%lld,%lld] cur=[%lld,%lld]\n",
                (long long)gate_w->ne[0],(long long)gate_w->ne[1],
                (long long)up_w->ne[0],(long long)up_w->ne[1],
                (long long)down_w->ne[0],(long long)down_w->ne[1],
                (long long)cur->ne[0],(long long)cur->ne[1]);
        ggml_tensor* ffn = core_ffn::swiglu(ctx0, cur, gate_w, up_w, down_w);
        if (il <= 1) KUGELAUDIO_TRACE("kugelaudio: L FFN done, ffn=[%lld,%lld]\n",
                (long long)ffn->ne[0],(long long)ffn->ne[1]);
        cur = ggml_add(ctx0, residual, ffn);
        if (il <= 1 || il == hp.n_lm_layers - 1)
            KUGELAUDIO_TRACE( "kugelaudio: L%d done\n", il);
    }

    // Final RMSNorm
    KUGELAUDIO_TRACE("kugelaudio: final_norm cur=[%lld,%lld]\n", (long long)cur->ne[0], (long long)cur->ne[1]);
    cur = ggml_rms_norm(ctx0, cur, hp.rms_norm_eps);
    ggml_tensor* fnw = G("model.language_model.norm.weight");
    KUGELAUDIO_TRACE("kugelaudio: final_norm w=%p [%lld,%lld]\n", (void*)fnw,
            fnw?(long long)fnw->ne[0]:-1, fnw?(long long)fnw->ne[1]:-1);
    cur = ggml_mul(ctx0, cur, fnw);
    KUGELAUDIO_TRACE("kugelaudio: final_norm OK\n");

    // Output both hidden states and logits
    if (output_hidden) {
        // Output the last hidden state for diffusion conditioning
        ggml_tensor* hidden_last;
        if (n_tokens > 1) {
            hidden_last = ggml_view_1d(ctx0, cur, hp.d_lm,
                (size_t)(n_tokens - 1) * hp.d_lm * sizeof(float));
        } else {
            hidden_last = ggml_view_1d(ctx0, cur, hp.d_lm, 0);
        }
        ggml_set_name(hidden_last, "hidden_last");
        ggml_set_output(hidden_last);
        ggml_build_forward_expand(gf, hidden_last);
    }

    // LM head → logits (last token only)
    ggml_tensor* last_tok;
    if (n_tokens > 1) {
        last_tok = ggml_view_1d(ctx0, cur, hp.d_lm,
            (size_t)(n_tokens - 1) * hp.d_lm * sizeof(float));
        last_tok = ggml_reshape_2d(ctx0, last_tok, hp.d_lm, 1);
    } else {
        last_tok = cur;
    }
    ggml_tensor* lm_head_w = G("lm_head.weight");
    KUGELAUDIO_TRACE("kugelaudio: lm_head w=[%lld,%lld] last_tok=[%lld,%lld]\n",
            lm_head_w?(long long)lm_head_w->ne[0]:-1, lm_head_w?(long long)lm_head_w->ne[1]:-1,
            (long long)last_tok->ne[0], (long long)last_tok->ne[1]);
    ggml_tensor* logits = ggml_mul_mat(ctx0, lm_head_w, last_tok);
    KUGELAUDIO_TRACE("kugelaudio: logits=[%lld,%lld] OK\n", (long long)logits->ne[0], (long long)logits->ne[1]);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    KUGELAUDIO_TRACE("kugelaudio: build_lm_graph done\n");

    return gf;
}

// ── KV cache allocation ────────────────────────────────────────────────────

static bool alloc_kv_cache(kugelaudio_context* ctx, int max_ctx,
                           ggml_context*& kv_ctx_out, ggml_backend_buffer_t& kv_buf_out,
                           ggml_tensor*& kv_k_out, ggml_tensor*& kv_v_out) {
    auto& hp = ctx->model.hp;
    enum ggml_type kv_type = core_attn::kv_dtype_from_env("kugelaudio");

    size_t kv_mem = 2 * ggml_tensor_overhead() + 256;
    ggml_init_params kv_ip = {kv_mem, nullptr, true};  // no_alloc=true: data goes on backend buffer
    kv_ctx_out = ggml_init(kv_ip);

    kv_k_out = ggml_new_tensor_4d(kv_ctx_out, kv_type,
        hp.head_dim, max_ctx, hp.n_kv_heads, hp.n_lm_layers);
    kv_v_out = ggml_new_tensor_4d(kv_ctx_out, kv_type,
        hp.head_dim, max_ctx, hp.n_kv_heads, hp.n_lm_layers);

    kv_buf_out = ggml_backend_alloc_ctx_tensors(kv_ctx_out, ctx->backend);
    if (!kv_buf_out) {
        fprintf(stderr, "kugelaudio: failed to allocate KV cache (%d tokens)\n", max_ctx);
        ggml_free(kv_ctx_out);
        kv_ctx_out = nullptr;
        return false;
    }

    // Zero-fill
    ggml_backend_buffer_clear(kv_buf_out, 0);
    return true;
}

// ── Synthesize ─────────────────────────────────────────────────────────────

extern "C" float* kugelaudio_synthesize(struct kugelaudio_context* ctx,
                                        const char* text,
                                        int* out_n_samples) {
    if (!ctx || !text || !out_n_samples) return nullptr;
    *out_n_samples = 0;

    auto& hp = ctx->model.hp;
    auto& ts = ctx->model.tensors;
    auto G = [&](const std::string& name) -> ggml_tensor* {
        auto it = ts.find(name);
        return it != ts.end() ? it->second : nullptr;
    };

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "kugelaudio: synthesizing: \"%s\"\n", text);
    }

    // ── 1. Tokenize ────────────────────────────────────────────────────
    // Build prompt template matching KugelAudio training format:
    //   system_prompt + [voice_input section if voice loaded] + text_input + speech_output + speech_start
    std::string system_prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n";

    bool has_voice = (ctx->n_voice_frames > 0 && !ctx->voice_acoustic_mean.empty());

    // Voice input section: " Voice input:\n Speaker 0:" + [VAE placeholders] + "\n"
    std::string voice_section;
    if (has_voice) {
        voice_section = " Voice input:\n Speaker 0:";
        // The VAE placeholder tokens will be replaced with voice embeddings after tokenization
    }

    std::string text_section = " Text input:\n Speaker 0: " + std::string(text) + "\n Speech output:\n";
    std::string full_prompt = system_prompt + voice_section;

    // Tokenize the text parts
    std::vector<int32_t> token_ids;
    if (!ctx->model.vocab.empty()) {
        token_ids = tokenize_text_greedy(ctx->model, full_prompt.c_str());
    }

    // Insert voice placeholder tokens (speech_diffusion_id) for voice frames
    int voice_token_start = -1;
    if (has_voice) {
        voice_token_start = (int)token_ids.size();
        for (int i = 0; i < ctx->n_voice_frames; i++)
            token_ids.push_back(hp.speech_diffusion_id); // placeholder
        // Tokenize newline + text section
        auto rest_ids = tokenize_text_greedy(ctx->model, ("\n" + text_section).c_str());
        token_ids.insert(token_ids.end(), rest_ids.begin(), rest_ids.end());
    } else {
        auto rest_ids = tokenize_text_greedy(ctx->model, text_section.c_str());
        token_ids.insert(token_ids.end(), rest_ids.begin(), rest_ids.end());
    }

    if (token_ids.empty()) {
        fprintf(stderr, "kugelaudio: tokenization failed (vocab has %zu entries)\n",
                ctx->model.vocab.size());
        return nullptr;
    }

    // Append speech_start token
    token_ids.push_back(hp.speech_start_id);

    int n_prompt = (int)token_ids.size();
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "kugelaudio: prompt has %d tokens\n", n_prompt);
    }

    // ── 2. Allocate KV cache ───────────────────────────────────────────
    int max_ctx = n_prompt + ctx->params.max_new_tokens + 64;

    if (!ctx->kv_ctx) {
        if (!alloc_kv_cache(ctx, max_ctx, ctx->kv_ctx, ctx->kv_buf, ctx->kv_k, ctx->kv_v)) {
            return nullptr;
        }
        ctx->kv_max_ctx = max_ctx;
    }
    ctx->kv_n_used = 0;

    // ── 3. Embed prompt tokens ─────────────────────────────────────────
    ggml_tensor* tok_emb_w = G("model.language_model.embed_tokens.weight");
    if (!tok_emb_w) {
        fprintf(stderr, "kugelaudio: missing embedding weights\n");
        return nullptr;
    }

    // Read embedding matrix to CPU
    // In GGUF, embeddings are [vocab, d_lm] stored as [d_lm, vocab] in ggml
    // tok_emb_w->ne[0] = d_lm, tok_emb_w->ne[1] = vocab
    (void)tok_emb_w; // used via closure below

    // Build embedding lookup via ggml graph
    auto run_embed_lookup = [&](const int32_t* ids, int n_ids, std::vector<float>& embeds) -> bool {
        size_t mem = ctx->compute_meta.size();
        ggml_init_params ip = {mem, ctx->compute_meta.data(), true};
        ggml_context* ctx0 = ggml_init(ip);
        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

        ggml_tensor* inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ids);
        ggml_set_name(inp, "emb_ids");
        ggml_set_input(inp);

        ggml_tensor* out = ggml_get_rows(ctx0, tok_emb_w, inp);
        // out is [d_lm, n_ids]
        ggml_set_name(out, "emb_out");
        ggml_set_output(out);
        ggml_build_forward_expand(gf, out);

        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) return false;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "emb_ids"), ids, 0,
                                (size_t)n_ids * sizeof(int32_t));
        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) return false;

        embeds.resize((size_t)hp.d_lm * n_ids);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "emb_out"), embeds.data(), 0,
                                embeds.size() * sizeof(float));
        return true;
    };

    std::vector<float> prompt_embeds;
    if (!run_embed_lookup(token_ids.data(), n_prompt, prompt_embeds)) {
        fprintf(stderr, "kugelaudio: embedding lookup failed\n");
        return nullptr;
    }

    // ── 3b. Replace voice placeholder embeddings with connector output ─
    if (has_voice && voice_token_start >= 0) {
        // Scale voice latents: (acoustic_mean + bias) * scaling
        int vae_dim = hp.vae_dim_acoustic;
        std::vector<float> scaled_voice(ctx->n_voice_frames * vae_dim);
        for (int i = 0; i < ctx->n_voice_frames * vae_dim; i++) {
            scaled_voice[i] = (ctx->voice_acoustic_mean[i] + hp.speech_bias_factor) * hp.speech_scaling_factor;
        }

        // Run acoustic connector on each voice frame
        for (int fi = 0; fi < ctx->n_voice_frames; fi++) {
            size_t mem = ctx->compute_meta.size();
            ggml_init_params ip = {mem, ctx->compute_meta.data(), true};
            ggml_context* ctx0 = ggml_init(ip);
            ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 256, false);

            ggml_tensor* lat_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, vae_dim);
            ggml_set_name(lat_in, "conn_in");
            ggml_set_input(lat_in);

            ggml_tensor* h = ggml_mul_mat(ctx0, G("model.acoustic_connector.fc1.weight"), lat_in);
            if (auto* b = G("model.acoustic_connector.fc1.bias")) h = ggml_add(ctx0, h, b);
            h = ggml_rms_norm(ctx0, h, 1e-6f);
            h = ggml_mul(ctx0, h, G("model.acoustic_connector.norm.weight"));
            h = ggml_mul_mat(ctx0, G("model.acoustic_connector.fc2.weight"), h);
            if (auto* b = G("model.acoustic_connector.fc2.bias")) h = ggml_add(ctx0, h, b);

            ggml_set_name(h, "conn_out");
            ggml_set_output(h);
            ggml_build_forward_expand(gf, h);

            ggml_backend_sched_reset(ctx->sched);
            if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) break;
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "conn_in"),
                                    &scaled_voice[fi * vae_dim], 0, vae_dim * sizeof(float));
            if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) break;

            // Replace embedding at voice_token_start + fi
            int emb_offset = (voice_token_start + fi) * hp.d_lm;
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "conn_out"),
                                    &prompt_embeds[emb_offset], 0, hp.d_lm * sizeof(float));
        }
        if (ctx->params.verbosity >= 1) {
            KUGELAUDIO_TRACE("kugelaudio: injected %d voice frames into prompt\n", ctx->n_voice_frames);
        }
    }

    // ── 4. AR decode loop ──────────────────────────────────────────────
    auto run_lm_step = [&](const float* embeds_data, int n_toks, int n_past,
                           std::vector<float>& logits, std::vector<float>* hidden_out) -> bool {
        bool need_hidden = (hidden_out != nullptr);
        ggml_cgraph* gf = build_lm_graph(ctx, n_toks, n_past, need_hidden);

        std::vector<int32_t> pos(n_toks);
        for (int i = 0; i < n_toks; i++) pos[i] = n_past + i;

        std::vector<ggml_fp16_t> mask;
        if (n_toks > 1) {
            int Lk = n_past + n_toks;
            mask.resize((size_t)n_toks * Lk, ggml_fp32_to_fp16(0.0f));
            ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            for (int q = 0; q < n_toks; q++)
                for (int k = 0; k < Lk; k++)
                    if (k > n_past + q) mask[(size_t)q * Lk + k] = neg_inf;
        }

        KUGELAUDIO_TRACE("kugelaudio: sched_reset...\n");
        ggml_backend_sched_reset(ctx->sched);
        KUGELAUDIO_TRACE("kugelaudio: alloc_graph n_nodes=%d...\n", ggml_graph_n_nodes(gf));
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) return false;
        KUGELAUDIO_TRACE("kugelaudio: alloc_graph OK, setting inputs...\n");

        ggml_tensor* t_inp = ggml_graph_get_tensor(gf, "lm_input");
        KUGELAUDIO_TRACE("kugelaudio: lm_input tensor=%p\n", (void*)t_inp);
        ggml_backend_tensor_set(t_inp, embeds_data, 0, (size_t)hp.d_lm * n_toks * sizeof(float));
        KUGELAUDIO_TRACE("kugelaudio: lm_input set OK\n");

        ggml_tensor* t_pos = ggml_graph_get_tensor(gf, "positions");
        KUGELAUDIO_TRACE("kugelaudio: positions tensor=%p\n", (void*)t_pos);
        ggml_backend_tensor_set(t_pos, pos.data(), 0, pos.size() * sizeof(int32_t));
        KUGELAUDIO_TRACE("kugelaudio: positions set OK\n");

        if (n_toks > 1) {
            ggml_tensor* t_mask = ggml_graph_get_tensor(gf, "causal_mask");
            KUGELAUDIO_TRACE("kugelaudio: causal_mask tensor=%p shape=[%lld,%lld]\n",
                    (void*)t_mask, t_mask?(long long)t_mask->ne[0]:-1, t_mask?(long long)t_mask->ne[1]:-1);
            ggml_backend_tensor_set(t_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
            KUGELAUDIO_TRACE("kugelaudio: causal_mask set OK\n");
        }

        KUGELAUDIO_TRACE("kugelaudio: graph_compute...\n");
        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) return false;
        KUGELAUDIO_TRACE("kugelaudio: graph_compute OK\n");

        logits.resize(hp.vocab_size);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "logits"), logits.data(), 0,
                                logits.size() * sizeof(float));

        if (need_hidden) {
            hidden_out->resize(hp.d_lm);
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "hidden_last"),
                                    hidden_out->data(), 0, hp.d_lm * sizeof(float));
        }
        return true;
    };

    // Constrained token selection
    auto constrained_argmax = [&](const std::vector<float>& logits) -> int {
        int valid_ids[] = {hp.speech_start_id, hp.speech_end_id,
                           hp.speech_diffusion_id, hp.eos_token_id};
        int best_id = hp.eos_token_id;
        float best_score = -1e30f;
        for (int id : valid_ids) {
            if (id < (int)logits.size() && logits[id] > best_score) {
                best_score = logits[id];
                best_id = id;
            }
        }
        return best_id;
    };

    // ── Prefill ────────────────────────────────────────────────────────
    std::vector<float> logits;
    std::vector<float> hidden_state;
    KUGELAUDIO_TRACE("kugelaudio: prefill %d tokens, d_lm=%d, embeds size=%zu\n",
            n_prompt, hp.d_lm, prompt_embeds.size());
    KUGELAUDIO_TRACE("kugelaudio: kv_k shape=[%lld,%lld,%lld,%lld] kv_v shape=[%lld,%lld,%lld,%lld]\n",
            (long long)ctx->kv_k->ne[0], (long long)ctx->kv_k->ne[1],
            (long long)ctx->kv_k->ne[2], (long long)ctx->kv_k->ne[3],
            (long long)ctx->kv_v->ne[0], (long long)ctx->kv_v->ne[1],
            (long long)ctx->kv_v->ne[2], (long long)ctx->kv_v->ne[3]);
    if (!run_lm_step(prompt_embeds.data(), n_prompt, 0, logits, &hidden_state)) {
        fprintf(stderr, "kugelaudio: LM prefill failed\n");
        return nullptr;
    }
    ctx->kv_n_used = n_prompt;

    // ── Generate tokens ────────────────────────────────────────────────
    std::vector<std::vector<float>> audio_chunks;
    int max_tokens = ctx->params.max_new_tokens;
    int tts_steps = ctx->params.tts_steps;
    float cfg_scale = ctx->params.cfg_scale;
    (void)cfg_scale; // TODO: implement CFG with negative path
    bool finished = false;

    for (int step = 0; step < max_tokens && !finished; step++) {
        int next_token = constrained_argmax(logits);

        KUGELAUDIO_TRACE("  step %d: token %d (start=%d diff=%d end=%d eos=%d)\n",
                step, next_token, hp.speech_start_id, hp.speech_diffusion_id,
                hp.speech_end_id, hp.eos_token_id);

        if (next_token == hp.eos_token_id || next_token == hp.speech_end_id) {
            finished = true;
            break;
        }

        if (next_token == hp.speech_diffusion_id) {
            KUGELAUDIO_TRACE("kugelaudio: diffusion path\n");
            // ── Run diffusion → decode → audio chunk ───────────────
            // Condition = last hidden state from LM

            // Generate diffusion noise
            dpm_sde_schedule sched = make_dpm_sde_schedule(tts_steps, hp.ddpm_num_steps);
            int vae_dim = hp.vae_dim_acoustic;

            std::vector<float> speech(vae_dim);
            crispasr::core::fill_gaussian_noise(speech.data(), vae_dim, ctx->rng);

            // DPM-Solver++ SDE loop
            std::vector<float> x0_prev(vae_dim, 0.0f);
            std::vector<float> x0_cur(vae_dim, 0.0f);

            KUGELAUDIO_TRACE("kugelaudio: starting %d diffusion steps\n", tts_steps);
            for (int si = 0; si < tts_steps; si++) {
                KUGELAUDIO_TRACE("kugelaudio: diff step %d/%d t=%d\n", si, tts_steps, sched.timesteps[si]);
                // Compute sinusoidal timestep embedding
                float t_val = (float)sched.timesteps[si];
                std::vector<float> t_sin(256);
                compute_sinusoidal_embed(t_val, t_sin.data(), 256);

                // Run prediction head
                ggml_cgraph* pred_gf = build_pred_head_graph(ctx, 1);
                ggml_backend_sched_reset(ctx->sched);
                if (!ggml_backend_sched_alloc_graph(ctx->sched, pred_gf)) {
                    fprintf(stderr, "kugelaudio: pred head alloc failed at step %d\n", si);
                    break;
                }

                ggml_backend_tensor_set(ggml_graph_get_tensor(pred_gf, "pred_noisy"),
                                        speech.data(), 0, vae_dim * sizeof(float));
                ggml_backend_tensor_set(ggml_graph_get_tensor(pred_gf, "pred_t_sin"),
                                        t_sin.data(), 0, 256 * sizeof(float));
                ggml_backend_tensor_set(ggml_graph_get_tensor(pred_gf, "pred_condition"),
                                        hidden_state.data(), 0, hp.d_lm * sizeof(float));

                KUGELAUDIO_TRACE("kugelaudio: diff step %d pred compute...\n", si);
                if (ggml_backend_sched_graph_compute(ctx->sched, pred_gf) != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "kugelaudio: pred head compute failed at step %d\n", si);
                    break;
                }
                KUGELAUDIO_TRACE("kugelaudio: diff step %d pred OK\n", si);

                std::vector<float> v_pred(vae_dim);
                ggml_backend_tensor_get(ggml_graph_get_tensor(pred_gf, "pred_output"),
                                        v_pred.data(), 0, vae_dim * sizeof(float));

                // v-prediction → x0
                int t_idx = sched.timesteps[si];
                float alpha_t = sqrtf(sched.alphas_cumprod[t_idx]);
                float sigma_t = sqrtf(1.0f - sched.alphas_cumprod[t_idx]);
                v_to_x0(speech.data(), v_pred.data(), x0_cur.data(), vae_dim, alpha_t, sigma_t);

                // Generate noise for SDE
                std::vector<float> noise(vae_dim);
                crispasr::core::fill_gaussian_noise(noise.data(), vae_dim, ctx->rng);

                // DPM-Solver step
                bool is_last = (si == tts_steps - 1);
                bool lower_order = (si < 1) || is_last;
                if (lower_order) {
                    dpm_sde_first_order(sched, si, speech.data(), x0_cur.data(),
                                        noise.data(), vae_dim);
                } else {
                    dpm_sde_second_order(sched, si, speech.data(), x0_cur.data(),
                                         x0_prev.data(), si - 1, noise.data(), vae_dim);
                }

                x0_prev = x0_cur;
            }

            // Unscale latent: z_raw = z / scaling - bias
            for (int i = 0; i < vae_dim; i++) {
                speech[i] = speech[i] / hp.speech_scaling_factor - hp.speech_bias_factor;
            }

            // Run acoustic decoder
            KUGELAUDIO_TRACE("kugelaudio: building VAE decoder graph...\n");
            ggml_cgraph* dec_gf = build_vae_decoder_graph(ctx, 1);
            KUGELAUDIO_TRACE("kugelaudio: VAE decoder graph built OK\n");
            ggml_backend_sched_reset(ctx->sched);
            if (!ggml_backend_sched_alloc_graph(ctx->sched, dec_gf)) {
                fprintf(stderr, "kugelaudio: decoder alloc failed\n");
                break;
            }
            ggml_backend_tensor_set(ggml_graph_get_tensor(dec_gf, "dec_latent"),
                                    speech.data(), 0, vae_dim * sizeof(float));
            if (ggml_backend_sched_graph_compute(ctx->sched, dec_gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "kugelaudio: decoder compute failed\n");
                break;
            }

            ggml_tensor* audio_out = ggml_graph_get_tensor(dec_gf, "dec_audio");
            int n_audio = (int)ggml_nelements(audio_out);
            std::vector<float> chunk(n_audio);
            ggml_backend_tensor_get(audio_out, chunk.data(), 0, n_audio * sizeof(float));
            audio_chunks.push_back(std::move(chunk));

            // Feed back: acoustic_connector(latent) → embedding for next token
            // Scale latent back for connector
            std::vector<float> scaled_latent(vae_dim);
            for (int i = 0; i < vae_dim; i++) {
                scaled_latent[i] = (speech[i] + hp.speech_bias_factor) * hp.speech_scaling_factor;
            }

            // Run acoustic connector: FC1 → RMSNorm → FC2
            // Build a small graph for the connector
            {
                size_t mem = ctx->compute_meta.size();
                ggml_init_params ip = {mem, ctx->compute_meta.data(), true};
                ggml_context* ctx0 = ggml_init(ip);
                ggml_cgraph* gf_conn = ggml_new_graph_custom(ctx0, 256, false);

                ggml_tensor* lat_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, vae_dim);
                ggml_set_name(lat_in, "conn_in");
                ggml_set_input(lat_in);

                ggml_tensor* h = ggml_mul_mat(ctx0, G("model.acoustic_connector.fc1.weight"), lat_in);
                if (auto* b = G("model.acoustic_connector.fc1.bias"))
                    h = ggml_add(ctx0, h, b);
                h = ggml_rms_norm(ctx0, h, 1e-6f);
                h = ggml_mul(ctx0, h, G("model.acoustic_connector.norm.weight"));
                h = ggml_mul_mat(ctx0, G("model.acoustic_connector.fc2.weight"), h);
                if (auto* b = G("model.acoustic_connector.fc2.bias"))
                    h = ggml_add(ctx0, h, b);

                ggml_set_name(h, "conn_out");
                ggml_set_output(h);
                ggml_build_forward_expand(gf_conn, h);

                ggml_backend_sched_reset(ctx->sched);
                if (!ggml_backend_sched_alloc_graph(ctx->sched, gf_conn)) break;
                ggml_backend_tensor_set(ggml_graph_get_tensor(gf_conn, "conn_in"),
                                        scaled_latent.data(), 0, vae_dim * sizeof(float));
                if (ggml_backend_sched_graph_compute(ctx->sched, gf_conn) != GGML_STATUS_SUCCESS)
                    break;

                std::vector<float> next_embed(hp.d_lm);
                ggml_backend_tensor_get(ggml_graph_get_tensor(gf_conn, "conn_out"),
                                        next_embed.data(), 0, hp.d_lm * sizeof(float));

                // Run LM with the connector output as next token embedding
                if (!run_lm_step(next_embed.data(), 1, ctx->kv_n_used, logits, &hidden_state)) {
                    fprintf(stderr, "kugelaudio: LM step failed after diffusion\n");
                    break;
                }
                ctx->kv_n_used++;
            }
        } else {
            KUGELAUDIO_TRACE("kugelaudio: embed token %d path\n", next_token);
            // speech_start or other token — embed and continue
            std::vector<float> tok_embed;
            if (!run_embed_lookup(&next_token, 1, tok_embed)) break;
            KUGELAUDIO_TRACE("kugelaudio: embed OK, running LM step n_past=%d\n", ctx->kv_n_used);

            if (!run_lm_step(tok_embed.data(), 1, ctx->kv_n_used, logits, &hidden_state)) {
                fprintf(stderr, "kugelaudio: LM step failed\n");
                break;
            }
            ctx->kv_n_used++;
        }
    }

    // ── 5. Concatenate audio chunks ────────────────────────────────────
    if (audio_chunks.empty()) {
        fprintf(stderr, "kugelaudio: no audio generated\n");
        return nullptr;
    }

    int total_samples = 0;
    for (const auto& c : audio_chunks) total_samples += (int)c.size();

    float* output = (float*)malloc(total_samples * sizeof(float));
    if (!output) return nullptr;

    int offset = 0;
    for (const auto& c : audio_chunks) {
        memcpy(output + offset, c.data(), c.size() * sizeof(float));
        offset += (int)c.size();
    }

    // Normalize to prevent clipping
    float max_val = 0.0f;
    for (int i = 0; i < total_samples; i++)
        max_val = std::max(max_val, fabsf(output[i]));
    if (max_val > 1.0f) {
        float scale = 0.95f / max_val;
        for (int i = 0; i < total_samples; i++)
            output[i] *= scale;
    }

    *out_n_samples = total_samples;
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "kugelaudio: generated %d samples (%.2f s @ 24 kHz)\n",
                total_samples, (float)total_samples / 24000.0f);
    }
    return output;
}

// ── Stage-level APIs ───────────────────────────────────────────────────────

extern "C" float* kugelaudio_run_diffusion_step(struct kugelaudio_context* ctx,
                                                 const float* noisy_latent, int vae_dim,
                                                 int timestep,
                                                 const float* condition, int d_lm,
                                                 int* out_dim) {
    if (!ctx || !noisy_latent || !condition || !out_dim) return nullptr;

    std::vector<float> t_sin(256);
    compute_sinusoidal_embed((float)timestep, t_sin.data(), 256);

    ggml_cgraph* gf = build_pred_head_graph(ctx, 1);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) return nullptr;

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pred_noisy"),
                            noisy_latent, 0, vae_dim * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pred_t_sin"),
                            t_sin.data(), 0, 256 * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pred_condition"),
                            condition, 0, d_lm * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;

    *out_dim = vae_dim;
    float* result = (float*)malloc(vae_dim * sizeof(float));
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "pred_output"),
                            result, 0, vae_dim * sizeof(float));
    return result;
}

extern "C" float* kugelaudio_run_acoustic_decoder(struct kugelaudio_context* ctx,
                                                   const float* latent, int vae_dim,
                                                   int* out_n_samples) {
    if (!ctx || !latent || !out_n_samples) return nullptr;

    ggml_cgraph* gf = build_vae_decoder_graph(ctx, 1);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) return nullptr;

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dec_latent"),
                            latent, 0, vae_dim * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;

    ggml_tensor* audio = ggml_graph_get_tensor(gf, "dec_audio");
    int n = (int)ggml_nelements(audio);
    float* result = (float*)malloc(n * sizeof(float));
    ggml_backend_tensor_get(audio, result, 0, n * sizeof(float));
    *out_n_samples = n;
    return result;
}

extern "C" int kugelaudio_load_voice(struct kugelaudio_context* ctx, const char* voice_path) {
    if (!ctx || !voice_path) return -1;

    // Load voice GGUF (small file: just acoustic_mean tensor + metadata)
    gguf_context* gctx = core_gguf::open_metadata(voice_path);
    if (!gctx) {
        fprintf(stderr, "kugelaudio: failed to open voice GGUF: %s\n", voice_path);
        return -1;
    }

    int n_frames = core_gguf::kv_u32(gctx, "kugelaudio.voice.n_frames", 0);
    int vae_dim = core_gguf::kv_u32(gctx, "kugelaudio.voice.vae_dim", ctx->model.hp.vae_dim_acoustic);
    gguf_free(gctx);

    if (n_frames == 0) {
        fprintf(stderr, "kugelaudio: voice GGUF has 0 frames\n");
        return -1;
    }

    // Load tensor data
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(voice_path, ctx->backend, "kugelaudio-voice", wl)) {
        fprintf(stderr, "kugelaudio: failed to load voice weights\n");
        return -1;
    }

    ggml_tensor* am = core_gguf::try_get(wl.tensors, "voice.acoustic_mean");
    if (!am) {
        fprintf(stderr, "kugelaudio: voice GGUF missing voice.acoustic_mean tensor\n");
        core_gguf::free_weights(wl);
        return -1;
    }

    // Copy to host
    int total = n_frames * vae_dim;
    ctx->voice_acoustic_mean.resize(total);
    ggml_backend_tensor_get(am, ctx->voice_acoustic_mean.data(), 0, total * sizeof(float));
    ctx->n_voice_frames = n_frames;

    core_gguf::free_weights(wl);

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "kugelaudio: loaded voice from %s (%d frames, vae_dim=%d)\n",
                voice_path, n_frames, vae_dim);
    }
    return 0;
}
