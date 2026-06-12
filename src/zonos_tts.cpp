// zonos_tts.cpp -- Zonos TTS backend (Zyphra/Zonos-v0.1-transformer).
//
// PLAN #130: Apache 2.0 licensed ~500M-param TTS with rich conditioning.
//
// Pipeline:
//   1. Text -> eSpeak phoneme IDs -> phoneme embedding lookup
//   2. Build conditioning prefix: concat(phoneme_embs, speaker_emb,
//      emotion_fourier, fmax_fourier, pitch_std_fourier, rate_fourier,
//      language_int_emb) -> LayerNorm -> linear projection
//   3. CFG: stack [cond_prefix; uncond_prefix] for classifier-free guidance
//   4. Prefill: prefix -> backbone -> first 9 codebook logits
//   5. AR decode loop: each step generates 9 codebook tokens via delay
//      pattern, sampling with min_p=0.1 from upstream
//   6. Revert delay pattern -> 9 codebook streams
//   7. DAC decoder: codes -> 44.1 kHz PCM (separate GGUF)
//
// Weight layout (from convert-zonos-to-gguf.py):
//   backbone.blk.N.attn_qkv.weight    [3072, 2048]  fused Q/K/V
//   backbone.blk.N.attn_output.weight  [2048, 2048]
//   backbone.blk.N.ffn_gate_up.weight  [16384, 2048] fused SwiGLU gate+up
//   backbone.blk.N.ffn_down.weight     [2048, 8192]
//   backbone.blk.N.attn_norm.{weight,bias}  [2048]   LayerNorm
//   backbone.blk.N.ffn_norm.{weight,bias}   [2048]   LayerNorm
//   backbone.output_norm.{weight,bias}      [2048]   final LayerNorm
//   embeddings.K.weight                [1026, 2048]   per-codebook
//   heads.K.weight                     [1025, 2048]   per-codebook
//   prefix_conditioner.*               conditioning weights

#include "zonos_tts.h"
#include "core/attention.h"
#include "core/dac_decoder.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// -----------------------------------------------------------------------
// Internal structures
// -----------------------------------------------------------------------

namespace {

struct zonos_hp {
    uint32_t d_model = 2048;
    uint32_t n_layer = 26;
    uint32_t n_heads = 16;
    uint32_t n_kv_heads = 4;
    uint32_t head_dim = 128;
    uint32_t ff_dim = 8192;
    float norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    uint32_t rope_dim = 128;

    uint32_t eos_token_id = 1024;
    uint32_t masked_token_id = 1025;
    uint32_t n_codebooks = 9;
    uint32_t codebook_size = 1024;
    uint32_t sample_rate = 44100;

    // Conditioner config
    uint32_t n_conditioners = 7;
    uint32_t phoneme_vocab_size = 189; // 4 special + 185 symbols
    uint32_t n_languages = 128;        // IntegerConditioner: max_val-min_val+1 = 126-(-1)+1 = 128

    // Derived
    uint32_t emb_vocab_size = 1026;  // codebook_size + 2 (eos + mask)
    uint32_t head_vocab_size = 1025; // codebook_size + 1 (eos)
};

struct zonos_layer {
    // Pre-attention LayerNorm
    ggml_tensor* attn_norm_w = nullptr; // (d_model,)
    ggml_tensor* attn_norm_b = nullptr; // (d_model,)
    // Fused Q/K/V projection
    ggml_tensor* attn_qkv_w = nullptr;    // (d_model, (n_heads+2*n_kv_heads)*head_dim)
    ggml_tensor* attn_output_w = nullptr; // (n_heads*head_dim, d_model)
    // Pre-FFN LayerNorm
    ggml_tensor* ffn_norm_w = nullptr; // (d_model,)
    ggml_tensor* ffn_norm_b = nullptr; // (d_model,)
    // Fused gate+up (SwiGLU)
    ggml_tensor* ffn_gate_up_w = nullptr; // (d_model, 2*ff_dim)
    ggml_tensor* ffn_down_w = nullptr;    // (ff_dim, d_model)
};

struct zonos_backbone {
    std::vector<zonos_layer> layers;
    ggml_tensor* output_norm_w = nullptr; // (d_model,)
    ggml_tensor* output_norm_b = nullptr; // (d_model,)
};

struct zonos_conditioner_weights {
    // Phoneme embedder (conditioner 0 - EspeakPhonemeConditioner)
    ggml_tensor* phoneme_emb_w = nullptr; // (phoneme_vocab_size, d_model)

    // Speaker projection (conditioner 1 - PassthroughConditioner + linear)
    ggml_tensor* speaker_proj_w = nullptr; // (cond_dim=128, d_model)
    ggml_tensor* speaker_proj_b = nullptr; // (d_model,)
    ggml_tensor* speaker_uncond = nullptr; // (d_model,)

    // Fourier conditioners (2=emotion, 3=fmax, 4=pitch_std, 5=speaking_rate)
    // Each has a random Fourier feature weight and an uncond vector
    struct fourier_cond {
        ggml_tensor* weight = nullptr;     // (d_model/2, input_dim)
        ggml_tensor* uncond_vec = nullptr; // (d_model,)
    };
    fourier_cond emotion;       // input_dim=8
    fourier_cond fmax;          // input_dim=1
    fourier_cond pitch_std;     // input_dim=1
    fourier_cond speaking_rate; // input_dim=1

    // Integer conditioner (6 - language_id)
    ggml_tensor* lang_emb_w = nullptr;  // (n_languages, d_model)
    ggml_tensor* lang_uncond = nullptr; // (d_model,)

    // Prefix conditioner top-level
    ggml_tensor* norm_w = nullptr; // (d_model,)
    ggml_tensor* norm_b = nullptr; // (d_model,)
    ggml_tensor* proj_w = nullptr; // (d_model, d_model)
    ggml_tensor* proj_b = nullptr; // (d_model,)
};

// Conditioning state (set by the user before synthesis)
struct zonos_cond_state {
    // Speaker embedding (128-d LDA projected)
    std::vector<float> speaker_emb; // 128 floats

    // Emotion vector (8 floats, normalized to sum=1)
    float emotion[8] = {0.3077f, 0.0256f, 0.0256f, 0.0256f, 0.0256f, 0.0256f, 0.2564f, 0.3077f};

    float fmax = 22050.0f;
    float pitch_std = 20.0f;
    float speaking_rate = 15.0f;
    int language_id = 24; // default: en-us (index 24 in the language list)

    // Language code strings for lookup
    std::vector<std::string> language_codes;
};

} // namespace

// -----------------------------------------------------------------------
// Context
// -----------------------------------------------------------------------

struct zonos_tts_context {
    zonos_tts_params params{};
    int n_threads = 4;

    zonos_hp hp;
    zonos_backbone backbone;
    zonos_conditioner_weights cond_w;
    zonos_cond_state cond_state;

    // Per-codebook embeddings and heads
    std::vector<ggml_tensor*> emb_w;  // [n_codebooks] each (emb_vocab_size, d_model)
    std::vector<ggml_tensor*> head_w; // [n_codebooks] each (head_vocab_size, d_model)

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Compute scheduler
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // KV caches — conditioned (primary) and unconditioned (for CFG)
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr; // (head_dim, max_ctx, n_kv_heads, n_layer)
    ggml_tensor* kv_v = nullptr;
    ggml_tensor* kv_k_uncond = nullptr; // CFG unconditioned path
    ggml_tensor* kv_v_uncond = nullptr;
    int kv_max_ctx = 0;

    // DAC codec (loaded lazily on first synthesize call)
    std::string dac_codec_path;
    bool dac_loaded = false;
    core_dac::DacWeights dac_w;
    ggml_context* dac_ctx_w = nullptr;
    ggml_backend_buffer_t dac_buf_w = nullptr;
    ggml_context* dac_ctx_perm = nullptr;
    ggml_backend_buffer_t dac_buf_perm = nullptr;

    // Sampler RNG
    uint64_t rng_state = 0xdeadbeefcafebabeULL;
};

// -----------------------------------------------------------------------
// Defaults
// -----------------------------------------------------------------------

struct zonos_tts_params zonos_tts_default_params(void) {
    struct zonos_tts_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.temperature = 1.0f; // upstream uses raw logits + min_p=0.1 (equivalent to temp=1.0)
    p.seed = 0;
    p.max_audio_tokens = 0; // 0 = default (86*30=2580)
    p.flash_attn = false;
    p.cfg_scale = 2.0f; // Zonos requires CFG for meaningful output
    return p;
}

// -----------------------------------------------------------------------
// GGUF loading
// -----------------------------------------------------------------------

static bool load_hparams(struct gguf_context* gguf_ctx, zonos_hp& hp) {
    auto get_u32 = [&](const char* key, uint32_t def) -> uint32_t {
        int idx = gguf_find_key(gguf_ctx, key);
        return idx >= 0 ? (uint32_t)gguf_get_val_u32(gguf_ctx, idx) : def;
    };
    auto get_f32 = [&](const char* key, float def) -> float {
        int idx = gguf_find_key(gguf_ctx, key);
        return idx >= 0 ? gguf_get_val_f32(gguf_ctx, idx) : def;
    };

    hp.d_model = get_u32("zonos.d_model", 2048);
    hp.n_layer = get_u32("zonos.n_layer", 26);
    hp.n_heads = get_u32("zonos.n_heads", 16);
    hp.n_kv_heads = get_u32("zonos.n_kv_heads", 4);
    hp.head_dim = get_u32("zonos.head_dim", 128);
    hp.ff_dim = get_u32("zonos.ff_dim", 8192);
    hp.norm_eps = get_f32("zonos.norm_eps", 1e-5f);
    hp.rope_theta = get_f32("zonos.rope_theta", 10000.0f);
    hp.rope_dim = get_u32("zonos.rope_dim", 128);
    hp.eos_token_id = get_u32("zonos.eos_token_id", 1024);
    hp.masked_token_id = get_u32("zonos.masked_token_id", 1025);
    hp.n_codebooks = get_u32("zonos.n_codebooks", 9);
    hp.codebook_size = get_u32("zonos.codebook_size", 1024);
    hp.sample_rate = get_u32("zonos.sample_rate", 44100);
    hp.n_conditioners = get_u32("zonos.n_conditioners", 7);
    hp.phoneme_vocab_size = get_u32("zonos.phoneme_vocab_size", 189);
    hp.n_languages = get_u32("zonos.n_languages", 128);

    hp.emb_vocab_size = hp.codebook_size + 2;  // eos + mask
    hp.head_vocab_size = hp.codebook_size + 1; // eos

    return true;
}

static bool load_language_codes(struct gguf_context* gguf_ctx, zonos_cond_state& state) {
    int idx = gguf_find_key(gguf_ctx, "zonos.language_codes");
    if (idx < 0)
        return false;
    const char* str = gguf_get_val_str(gguf_ctx, idx);
    if (!str)
        return false;

    // Parse newline-separated language codes
    std::string s(str);
    size_t pos = 0;
    while (pos < s.size()) {
        size_t next = s.find('\n', pos);
        if (next == std::string::npos)
            next = s.size();
        if (next > pos) {
            state.language_codes.push_back(s.substr(pos, next - pos));
        }
        pos = next + 1;
    }
    return true;
}

// -----------------------------------------------------------------------
// Init
// -----------------------------------------------------------------------

struct zonos_tts_context* zonos_tts_init_from_file(const char* path_model, struct zonos_tts_params params) {
    if (!path_model)
        return nullptr;

    auto* ctx = new zonos_tts_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // Pass 1: metadata (hyperparameters + language codes)
    {
        struct gguf_context* gguf_ctx = core_gguf::open_metadata(path_model);
        if (!gguf_ctx) {
            fprintf(stderr, "zonos_tts: failed to load %s\n", path_model);
            delete ctx;
            return nullptr;
        }
        if (!load_hparams(gguf_ctx, ctx->hp)) {
            fprintf(stderr, "zonos_tts: failed to load hyperparameters\n");
            core_gguf::free_metadata(gguf_ctx);
            delete ctx;
            return nullptr;
        }
        load_language_codes(gguf_ctx, ctx->cond_state);
        // Fix default language_id by name lookup (hardcoded index is fragile)
        for (int i = 0; i < (int)ctx->cond_state.language_codes.size(); i++) {
            if (ctx->cond_state.language_codes[i] == "en-us") {
                ctx->cond_state.language_id = i;
                break;
            }
        }
        core_gguf::free_metadata(gguf_ctx);
    }

    const auto& hp = ctx->hp;
    if (params.verbosity >= 1) {
        fprintf(stderr, "zonos_tts: d_model=%u n_layer=%u n_heads=%u n_kv=%u\n", hp.d_model, hp.n_layer, hp.n_heads,
                hp.n_kv_heads);
        fprintf(stderr, "zonos_tts: ff_dim=%u head_dim=%u n_codebooks=%u\n", hp.ff_dim, hp.head_dim, hp.n_codebooks);
    }

    // Backend init
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "zonos_tts: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ctx->backend_cpu;
    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    }
    if (!ctx->backend) {
        ctx->backend = ctx->backend_cpu;
    }

    // Pass 2: load weights via core_gguf helper
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "zonos_tts", wl)) {
        fprintf(stderr, "zonos_tts: failed to load weights from '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;
    ctx->tensors = std::move(wl.tensors);

    // Wire up weight pointers
    auto find_t = [&](const char* name) -> ggml_tensor* {
        auto it = ctx->tensors.find(name);
        return it != ctx->tensors.end() ? it->second : nullptr;
    };

    // Backbone layers
    ctx->backbone.layers.resize(hp.n_layer);
    for (uint32_t i = 0; i < hp.n_layer; i++) {
        auto& layer = ctx->backbone.layers[i];
        char buf[128];

        snprintf(buf, sizeof(buf), "backbone.blk.%u.attn_norm.weight", i);
        layer.attn_norm_w = find_t(buf);
        snprintf(buf, sizeof(buf), "backbone.blk.%u.attn_norm.bias", i);
        layer.attn_norm_b = find_t(buf);
        snprintf(buf, sizeof(buf), "backbone.blk.%u.attn_qkv.weight", i);
        layer.attn_qkv_w = find_t(buf);
        snprintf(buf, sizeof(buf), "backbone.blk.%u.attn_output.weight", i);
        layer.attn_output_w = find_t(buf);
        snprintf(buf, sizeof(buf), "backbone.blk.%u.ffn_norm.weight", i);
        layer.ffn_norm_w = find_t(buf);
        snprintf(buf, sizeof(buf), "backbone.blk.%u.ffn_norm.bias", i);
        layer.ffn_norm_b = find_t(buf);
        snprintf(buf, sizeof(buf), "backbone.blk.%u.ffn_gate_up.weight", i);
        layer.ffn_gate_up_w = find_t(buf);
        snprintf(buf, sizeof(buf), "backbone.blk.%u.ffn_down.weight", i);
        layer.ffn_down_w = find_t(buf);
    }
    ctx->backbone.output_norm_w = find_t("backbone.output_norm.weight");
    ctx->backbone.output_norm_b = find_t("backbone.output_norm.bias");

    // Codebook embeddings and heads
    ctx->emb_w.resize(hp.n_codebooks);
    ctx->head_w.resize(hp.n_codebooks);
    for (uint32_t k = 0; k < hp.n_codebooks; k++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "embeddings.%u.weight", k);
        ctx->emb_w[k] = find_t(buf);
        snprintf(buf, sizeof(buf), "heads.%u.weight", k);
        ctx->head_w[k] = find_t(buf);
    }

    // Conditioner weights
    auto& cw = ctx->cond_w;
    cw.phoneme_emb_w = find_t("prefix_conditioner.conditioners.0.phoneme_embedder.weight");
    cw.speaker_proj_w = find_t("prefix_conditioner.conditioners.1.project.weight");
    cw.speaker_proj_b = find_t("prefix_conditioner.conditioners.1.project.bias");
    cw.speaker_uncond = find_t("prefix_conditioner.conditioners.1.uncond_vector");

    cw.emotion.weight = find_t("prefix_conditioner.conditioners.2.weight");
    cw.emotion.uncond_vec = find_t("prefix_conditioner.conditioners.2.uncond_vector");
    cw.fmax.weight = find_t("prefix_conditioner.conditioners.3.weight");
    cw.fmax.uncond_vec = find_t("prefix_conditioner.conditioners.3.uncond_vector");
    cw.pitch_std.weight = find_t("prefix_conditioner.conditioners.4.weight");
    cw.pitch_std.uncond_vec = find_t("prefix_conditioner.conditioners.4.uncond_vector");
    cw.speaking_rate.weight = find_t("prefix_conditioner.conditioners.5.weight");
    cw.speaking_rate.uncond_vec = find_t("prefix_conditioner.conditioners.5.uncond_vector");

    cw.lang_emb_w = find_t("prefix_conditioner.conditioners.6.int_embedder.weight");
    cw.lang_uncond = find_t("prefix_conditioner.conditioners.6.uncond_vector");

    cw.norm_w = find_t("prefix_conditioner.norm.weight");
    cw.norm_b = find_t("prefix_conditioner.norm.bias");
    cw.proj_w = find_t("prefix_conditioner.project.weight");
    cw.proj_b = find_t("prefix_conditioner.project.bias");

    // Initialize RNG first (needed for speaker embedding)
    if (params.seed != 0) {
        ctx->rng_state = params.seed;
    }

    // Try to load speaker embedding from file, else use random Gaussian.
    ctx->cond_state.speaker_emb.resize(128);
    {
        const char* spk_path = getenv("ZONOS_SPEAKER_EMB_PATH");
        if (!spk_path)
            spk_path = "/mnt/storage/zonos-tts/jfk_speaker_emb.bin";
        FILE* sf = fopen(spk_path, "rb");
        if (sf) {
            int32_t dim = 0;
            if (fread(&dim, sizeof(int32_t), 1, sf) == 1 && dim == 128) {
                fread(ctx->cond_state.speaker_emb.data(), sizeof(float), 128, sf);
                if (params.verbosity >= 1)
                    fprintf(stderr, "zonos_tts: loaded speaker embedding from %s\n", spk_path);
            }
            fclose(sf);
        } else {
            // Random Gaussian fallback
            uint64_t rng = ctx->rng_state;
            for (int i = 0; i < 128; i++) {
                rng ^= rng >> 12;
                rng ^= rng << 25;
                rng ^= rng >> 27;
                float u1 = (float)((rng * 0x2545F4914F6CDD1DULL) >> 11) / (float)(1ULL << 53);
                rng ^= rng >> 12;
                rng ^= rng << 25;
                rng ^= rng >> 27;
                float u2 = (float)((rng * 0x2545F4914F6CDD1DULL) >> 11) / (float)(1ULL << 53);
                u1 = std::max(u1, 1e-7f);
                ctx->cond_state.speaker_emb[i] = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265f * u2);
            }
        }
    }

    if (params.verbosity >= 1) {
        fprintf(stderr, "zonos_tts: loaded %zu tensors from %s\n", ctx->tensors.size(), path_model);
    }

    // Create compute scheduler
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend) {
            backends[n_be++] = ctx->backend_cpu;
        }
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 4096, false, false);
    }
    ctx->compute_meta.resize(ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false));

    return ctx;
}

// -----------------------------------------------------------------------
// Conditioning setters
// -----------------------------------------------------------------------

void zonos_tts_set_pitch_std(struct zonos_tts_context* ctx, float pitch_std) {
    if (ctx)
        ctx->cond_state.pitch_std = pitch_std;
}

void zonos_tts_set_speaking_rate(struct zonos_tts_context* ctx, float rate) {
    if (ctx)
        ctx->cond_state.speaking_rate = rate;
}

void zonos_tts_set_emotion(struct zonos_tts_context* ctx, const float* emotion, int len) {
    if (!ctx || !emotion || len < 1)
        return;
    float sum = 0.0f;
    for (int i = 0; i < 8 && i < len; i++) {
        ctx->cond_state.emotion[i] = emotion[i];
        sum += emotion[i];
    }
    // Normalize
    if (sum > 0.0f) {
        for (int i = 0; i < 8; i++) {
            ctx->cond_state.emotion[i] /= sum;
        }
    }
}

void zonos_tts_set_fmax(struct zonos_tts_context* ctx, float fmax) {
    if (ctx)
        ctx->cond_state.fmax = fmax;
}

int zonos_tts_set_language(struct zonos_tts_context* ctx, const char* lang_code) {
    if (!ctx || !lang_code)
        return -1;
    // "auto" = keep the default (en-us); not a real language code
    if (strcmp(lang_code, "auto") == 0 || strcmp(lang_code, "") == 0)
        return 0;
    for (size_t i = 0; i < ctx->cond_state.language_codes.size(); i++) {
        if (ctx->cond_state.language_codes[i] == lang_code) {
            ctx->cond_state.language_id = (int)i;
            return 0;
        }
    }
    fprintf(stderr, "zonos_tts: unknown language code '%s'\n", lang_code);
    return -1;
}

void zonos_tts_set_speaker_embedding(struct zonos_tts_context* ctx, const float* emb, int dim) {
    if (!ctx || !emb || dim < 1)
        return;
    ctx->cond_state.speaker_emb.assign(emb, emb + dim);
}

int zonos_tts_set_codec_path(struct zonos_tts_context* ctx, const char* path) {
    if (!ctx || !path)
        return -1;
    ctx->dac_codec_path = path;
    return 0;
}

int zonos_tts_set_voice(struct zonos_tts_context* ctx, const char* wav_path) {
    if (!ctx || !wav_path)
        return -1;
    // TODO: implement ResNet293 speaker encoder in C++ or load pre-computed
    // embeddings. For now, this is a stub that requires pre-computed embeddings
    // via zonos_tts_set_speaker_embedding().
    fprintf(stderr, "zonos_tts: set_voice not yet implemented; use set_speaker_embedding\n");
    return -1;
}

// -----------------------------------------------------------------------
// Phoneme tokenization (simplified -- maps IPA characters to embedding IDs)
// -----------------------------------------------------------------------

namespace {

// The phoneme symbol table from conditioning.py, indexed starting at 4
// (after PAD=0, UNK=1, BOS=2, EOS=3).
static const char* const PHONEME_SPECIAL_NAMES[] = {"<pad>", "<unk>", "<bos>", "<eos>"};

// Build a simple char -> ID map. The full eSpeak phonemizer runs
// externally; this is a fallback for when phonemes are pre-computed
// or for simple Latin-script inputs.
static std::unordered_map<uint32_t, int> build_phoneme_map() {
    // These must match the symbols list in conditioning.py exactly
    static const char punctuation[] = ";:,.!?\xc2\xa1\xc2\xbf\xe2\x80\x94\xe2\x80\xa6\""
                                      "\xc2\xab\xc2\xbb\xe2\x80\x9c\xe2\x80\x9d() *~-/\\&";
    static const char letters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    std::unordered_map<uint32_t, int> map;
    int id = 4; // start after special tokens

    // We just map ASCII printable + common IPA to sequential IDs.
    // The converter script embeds the full symbol table as GGUF metadata.
    // For now, map basic ASCII characters.
    for (const char* p = punctuation; *p;) {
        // UTF-8 decode
        uint32_t cp = 0;
        uint8_t c = (uint8_t)*p;
        if (c < 0x80) {
            cp = c;
            p++;
        } else if ((c & 0xE0) == 0xC0) {
            cp = (c & 0x1F) << 6 | ((uint8_t)p[1] & 0x3F);
            p += 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = (c & 0x0F) << 12 | ((uint8_t)p[1] & 0x3F) << 6 | ((uint8_t)p[2] & 0x3F);
            p += 3;
        } else {
            cp = (c & 0x07) << 18 | ((uint8_t)p[1] & 0x3F) << 12 | ((uint8_t)p[2] & 0x3F) << 6 | ((uint8_t)p[3] & 0x3F);
            p += 4;
        }
        map[cp] = id++;
    }
    for (const char* p = letters; *p; p++) {
        map[(uint32_t)*p] = id++;
    }
    // IPA symbols from conditioning.py: _letters_ipa
    static const char ipa_symbols[] =
        "\xc9\x91\xc9\x90\xc9\x92\xc3\xa6\xc9\x93\xca\x99\xce\xb2\xc9\x94\xc9\x95\xc3\xa7\xc9\x97\xc9\x96\xc3\xb0"
        "\xca\xa4\xc9\x99\xc9\x98\xc9\x9a\xc9\x9b\xc9\x9c\xc9\x9d\xc9\x9e\xc9\x9f\xca\x84\xc9\xa1\xc9\xa0\xc9\xa2"
        "\xca\x9b\xc9\xa6\xc9\xa7\xc4\xa7\xc9\xa5\xca\x9c\xc9\xa8\xc9\xaa\xca\x9d\xc9\xad\xc9\xac\xc9\xab\xc9\xae"
        "\xca\x9f\xc9\xb1\xc9\xaf\xc9\xb0\xc5\x8b\xc9\xb3\xc9\xb2\xc9\xb4\xc3\xb8\xc9\xb5\xc9\xb8\xce\xb8\xc5\x93"
        "\xc9\xb6\xca\x98\xc9\xb9\xc9\xba\xc9\xbe\xc9\xbb\xca\x80\xca\x81\xc9\xbd\xca\x82\xca\x83\xca\x88\xca\xa7"
        "\xca\x89\xca\x8a\xca\x8b\xe2\xb1\xb1\xca\x8c\xc9\xa3\xc9\xa4\xca\x8d\xcf\x87\xca\x8e\xca\x8f\xca\x91\xca\x90"
        "\xca\x92\xca\x94\xca\xa1\xca\x95\xca\xa2\xc7\x80\xc7\x81\xc7\x82\xc7\x83\xcb\x88\xcb\x8c\xcb\x90\xcb\x91"
        "\xca\xbc\xca\xb4\xca\xb0\xca\xb1\xca\xb2\xca\xb7\xcb\xa0\xcb\xa4\xcb\x9e\xe2\x86\x93\xe2\x86\x91\xe2\x86\x92"
        "\xe2\x86\x97\xe2\x86\x98\xe2\x80\x99\xcc\xa9\xe2\x80\x99\xe1\xb5\xbb";
    for (const char* p = ipa_symbols; *p;) {
        uint32_t cp = 0;
        uint8_t c = (uint8_t)*p;
        if (c < 0x80) {
            cp = c;
            p++;
        } else if ((c & 0xE0) == 0xC0) {
            cp = (c & 0x1F) << 6 | ((uint8_t)p[1] & 0x3F);
            p += 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = (c & 0x0F) << 12 | ((uint8_t)p[1] & 0x3F) << 6 | ((uint8_t)p[2] & 0x3F);
            p += 3;
        } else {
            cp = (c & 0x07) << 18 | ((uint8_t)p[1] & 0x3F) << 12 | ((uint8_t)p[2] & 0x3F) << 6 | ((uint8_t)p[3] & 0x3F);
            p += 4;
        }
        map[cp] = id++;
    }
    return map;
}

// UTF-8 decode helper for phoneme tokenization
static uint32_t utf8_decode(const char** pp) {
    const uint8_t* p = (const uint8_t*)*pp;
    uint32_t cp;
    if (*p < 0x80) {
        cp = *p;
        *pp += 1;
    } else if ((*p & 0xE0) == 0xC0) {
        cp = (*p & 0x1F) << 6 | (p[1] & 0x3F);
        *pp += 2;
    } else if ((*p & 0xF0) == 0xE0) {
        cp = (*p & 0x0F) << 12 | (p[1] & 0x3F) << 6 | (p[2] & 0x3F);
        *pp += 3;
    } else {
        cp = (*p & 0x07) << 18 | (p[1] & 0x3F) << 12 | (p[2] & 0x3F) << 6 | (p[3] & 0x3F);
        *pp += 4;
    }
    return cp;
}

// Convert a text string (already IPA if from espeak-ng, or raw text) to phoneme IDs.
// Skips zero-width joiners (U+200D) and other Unicode control chars that
// espeak-ng may insert but Python's phonemizer strips.
static std::vector<int32_t> text_to_phoneme_ids(const char* text) {
    static auto map = build_phoneme_map();
    std::vector<int32_t> ids;
    ids.push_back(2); // BOS
    for (const char* p = text; *p;) {
        uint32_t cp = utf8_decode(&p);
        // Skip Unicode control/format characters (Cf category): ZWJ, ZWNJ, etc.
        if (cp == 0x200D || cp == 0x200C || cp == 0x200B || cp == 0xFEFF)
            continue;
        auto it = map.find(cp);
        if (it != map.end()) {
            ids.push_back(it->second);
        }
        // Skip unmapped characters silently (matching Python's behavior of
        // only tokenizing known symbols and ignoring others)
    }
    ids.push_back(3); // EOS
    return ids;
}

// MSVC uses _popen/_pclose instead of popen/pclose.
#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

// Run espeak-ng via popen to get IPA phonemes for the given text.
// Returns the IPA string, or empty string on failure.
static std::string phonemize_espeak(const std::string& lang, const std::string& text) {
    // Build command: espeak-ng -q --ipa=3 -v <lang> "<text>"
    // Escape text for shell
    std::string escaped;
    for (char c : text) {
        if (c == '\'')
            escaped += "'\\''";
        else
            escaped += c;
    }
#ifdef _WIN32
    std::string cmd = "espeak-ng -q --ipa=3 -v " + lang + " \"" + escaped + "\" 2>NUL";
#else
    std::string cmd = "espeak-ng -q --ipa=3 -v " + lang + " '" + escaped + "' 2>/dev/null";
#endif
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp)
        return "";
    std::string out;
    char buf[256];
    while (fgets(buf, sizeof(buf), fp)) {
        out += buf;
    }
    pclose(fp);
    // Trim trailing whitespace
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' '))
        out.pop_back();

    // Python phonemizer uses preserve_punctuation=True with punctuation_marks from
    // conditioning.py: ';:,.!?¡¿—…"«»""() *~-/\\&'. Non-space punctuation characters
    // that appear at the TAIL of the original text are appended to the IPA so the
    // token count matches. espeak-ng drops them; the phonemizer library keeps them.
    static const uint8_t PUNCT_BYTES[] = {';',  ':',  ',',  '.', '!', '?', 0xc2, 0xa1, // ¡ (U+00A1)
                                          0xc2, 0xbf,                                  // ¿ (U+00BF)
                                          0xe2, 0x80, 0x94,                            // — (U+2014)
                                          0xe2, 0x80, 0xa6,                            // … (U+2026)
                                          '"',  ')',  '(',  '*', '~', '-', '/',  '\\', '&', 0};
    // Build a set of punctuation bytes for fast lookup
    static const std::string PUNCT_ASCII = ";:,.!?\"()*~-/\\&)";
    // Scan the original text from the end, collecting non-space punctuation
    std::string tail_punct;
    size_t tpos = text.size();
    while (tpos > 0) {
        const uint8_t c = (uint8_t)text[tpos - 1];
        // Quick ASCII check
        bool is_punct = false;
        if (c < 0x80) {
            if (PUNCT_ASCII.find((char)c) != std::string::npos)
                is_punct = true;
        } else {
            // Check multi-byte sequences against PUNCT_BYTES
            for (const uint8_t* p = PUNCT_BYTES; *p;) {
                if (*p >= 0x80) {
                    // How many bytes does this sequence take?
                    int seq = (*p & 0xE0) == 0xC0 ? 2 : (*p & 0xF0) == 0xE0 ? 3 : 4;
                    if ((int)(tpos) >= seq) {
                        bool match = true;
                        for (int k = 0; k < seq && match; k++)
                            match = (uint8_t)text[tpos - seq + k] == p[k];
                        if (match) {
                            is_punct = true;
                            break;
                        }
                    }
                    p += (*p & 0xE0) == 0xC0 ? 2 : (*p & 0xF0) == 0xE0 ? 3 : 4;
                } else {
                    p++;
                }
            }
        }
        if (!is_punct)
            break;
        // Collect this punctuation char/sequence
        if (c < 0x80) {
            tail_punct = std::string(1, (char)c) + tail_punct;
            tpos--;
        } else {
            int seq = (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
            if ((int)tpos >= seq) {
                tail_punct = text.substr(tpos - seq, seq) + tail_punct;
                tpos -= seq;
            } else {
                break;
            }
        }
    }
    if (!tail_punct.empty())
        out += tail_punct;

    return out;
}

// Full tokenization: text -> espeak-ng IPA -> phoneme IDs.
// Falls back to raw character tokenization if espeak-ng is unavailable.
static std::vector<int32_t> tokenize_text_full(const char* text, const char* lang = "en-us") {
    std::string ipa = phonemize_espeak(lang ? lang : "en-us", text);
    if (!ipa.empty()) {
        return text_to_phoneme_ids(ipa.c_str());
    }
    // Fallback: tokenize raw text characters
    fprintf(stderr, "zonos_tts: WARN: espeak-ng not available, using raw text tokenization\n");
    return text_to_phoneme_ids(text);
}

} // namespace

// -----------------------------------------------------------------------
// Delay pattern (from codebook_pattern.py)
// -----------------------------------------------------------------------

namespace {

// Apply delay pattern: shift codebook k by (k+1) positions, fill with mask_token.
// Input codes: (n_codebooks, seq_len), output: (n_codebooks, seq_len + n_codebooks)
static void apply_delay_pattern(const std::vector<std::vector<int32_t>>& codes, int32_t mask_token,
                                std::vector<std::vector<int32_t>>& out) {
    int n_cb = (int)codes.size();
    int seq_len = codes.empty() ? 0 : (int)codes[0].size();
    int out_len = seq_len + n_cb;
    out.resize(n_cb);
    for (int k = 0; k < n_cb; k++) {
        out[k].resize(out_len, mask_token);
        // Roll by (k+1): codes[k][i] goes to position (i + k + 1) % out_len
        for (int i = 0; i < seq_len; i++) {
            int dest = (i + k + 1) % out_len;
            out[k][dest] = codes[k][i];
        }
    }
}

// Revert delay pattern: undo the shift.
// Input: (n_codebooks, delayed_len), output: (n_codebooks, delayed_len - n_codebooks)
static void revert_delay_pattern(const std::vector<std::vector<int32_t>>& delayed,
                                 std::vector<std::vector<int32_t>>& out) {
    int n_cb = (int)delayed.size();
    if (n_cb == 0)
        return;
    int delayed_len = (int)delayed[0].size();
    int out_len = delayed_len - n_cb;
    if (out_len <= 0) {
        out.clear();
        return;
    }
    out.resize(n_cb);
    for (int k = 0; k < n_cb; k++) {
        out[k].resize(out_len);
        for (int i = 0; i < out_len; i++) {
            out[k][i] = delayed[k][k + 1 + i];
        }
    }
}

} // namespace

// -----------------------------------------------------------------------
// KV cache allocation
// -----------------------------------------------------------------------

static bool kv_alloc(zonos_tts_context* ctx, int max_ctx) {
    if (ctx->kv_k) {
        return true; // already allocated
    }
    const auto& hp = ctx->hp;
    const int hd = (int)hp.head_dim;
    const int n_kv = (int)hp.n_kv_heads;
    const int nl = (int)hp.n_layer;

    // Allocate 2 KV caches: conditioned (primary) + unconditioned (for CFG)
    ggml_init_params kp = {ggml_tensor_overhead() * 8 + 1024, nullptr, true};
    ctx->kv_ctx = ggml_init(kp);
    const auto kv_pair = core_attn::kv_dtype_pair_from_env("zonos_tts");
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.k, hd, max_ctx, n_kv, nl);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.v, hd, max_ctx, n_kv, nl);
    ctx->kv_k_uncond = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.k, hd, max_ctx, n_kv, nl);
    ctx->kv_v_uncond = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.v, hd, max_ctx, n_kv, nl);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");
    ggml_set_name(ctx->kv_k_uncond, "kv_k_u");
    ggml_set_name(ctx->kv_v_uncond, "kv_v_u");

    const size_t kb = ggml_nbytes(ctx->kv_k), vb = ggml_nbytes(ctx->kv_v);
    const size_t total_kv = (kb + vb) * 2; // 2 caches
    ggml_backend_t kv_backend = core_attn::kv_backend_from_env(ctx->backend, ctx->backend_cpu, "zonos_tts");
    ctx->kv_buf = ggml_backend_alloc_buffer(kv_backend, total_kv);
    if (!ctx->kv_buf) {
        fprintf(stderr, "zonos_tts: kv alloc failed\n");
        return false;
    }
    char* base = (char*)ggml_backend_buffer_get_base(ctx->kv_buf);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + kb);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k_uncond, base + kb + vb);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v_uncond, base + 2 * kb + vb);
    ctx->kv_max_ctx = max_ctx;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "zonos_tts: kv cache %d MiB x2 (hd=%d max=%d n_kv=%d nl=%d)\n", (int)((kb + vb) / 1048576), hd,
                max_ctx, n_kv, nl);
    }
    return true;
}

// -----------------------------------------------------------------------
// Conditioning prefix (CPU compute, builds float buffer)
// -----------------------------------------------------------------------

namespace {

// Read a tensor into a float vector, dequantizing if needed (F16/Q8_0/etc → F32).
static std::vector<float> tensor_to_float(ggml_tensor* t) {
    const int64_t n_el = ggml_nelements(t);
    const size_t nbytes = ggml_nbytes(t);
    std::vector<float> out(n_el);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, nbytes);
    } else {
        // Read raw bytes, then dequantize
        std::vector<uint8_t> raw(nbytes);
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
        const struct ggml_type_traits* traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            traits->to_float(raw.data(), out.data(), (int)n_el);
        } else if (t->type == GGML_TYPE_F16) {
            const ggml_fp16_t* src = (const ggml_fp16_t*)raw.data();
            for (int64_t i = 0; i < n_el; i++) {
                out[i] = ggml_fp16_to_fp32(src[i]);
            }
        }
    }
    return out;
}

// Read a single row from a 2D tensor as float, dequantizing if needed.
// Tensor ne = [row_size, n_rows]. row_idx selects which row.
static void tensor_get_row_f32(ggml_tensor* t, int row_idx, float* out, int row_size) {
    if (t->type == GGML_TYPE_F32) {
        size_t offset = (size_t)row_idx * row_size * sizeof(float);
        ggml_backend_tensor_get(t, out, offset, (size_t)row_size * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        size_t offset = (size_t)row_idx * row_size * sizeof(ggml_fp16_t);
        std::vector<ggml_fp16_t> raw(row_size);
        ggml_backend_tensor_get(t, raw.data(), offset, (size_t)row_size * sizeof(ggml_fp16_t));
        for (int i = 0; i < row_size; i++) {
            out[i] = ggml_fp16_to_fp32(raw[i]);
        }
    } else {
        // For quantized types, read the full tensor and extract the row
        // (quantized row sizes are block-aligned, so we dequant the whole thing)
        auto all = tensor_to_float(t);
        std::memcpy(out, &all[(size_t)row_idx * row_size], (size_t)row_size * sizeof(float));
    }
}

// Random Fourier features:
//   Python: x_norm = (x - min_val) / (max_val - min_val)
//           f = 2*pi * x_norm @ weight.T
//           return cat([cos(f), sin(f)])
// weight shape: (d_model/2, input_dim), GGUF ne=[input_dim, half_d].
static void compute_fourier_features(const float* input, int input_dim, ggml_tensor* weight_tensor, float* out,
                                     int d_model, float min_val = 0.0f, float max_val = 1.0f) {
    const int half_d = d_model / 2;
    auto weight_data = tensor_to_float(weight_tensor);

    // Normalize input to [0,1] range (critical — raw values like fmax=22050 would explode)
    std::vector<float> normed(input_dim);
    float range = max_val - min_val;
    if (range <= 0.0f)
        range = 1.0f;
    for (int j = 0; j < input_dim; j++) {
        normed[j] = (input[j] - min_val) / range;
    }

    for (int i = 0; i < half_d; i++) {
        float dot = 0.0f;
        for (int j = 0; j < input_dim; j++) {
            dot += weight_data[(size_t)i * input_dim + j] * normed[j];
        }
        dot *= 2.0f * 3.14159265358979323846f;
        out[i] = std::cos(dot);
        out[half_d + i] = std::sin(dot);
    }
}

// Build conditioned or unconditioned prefix on CPU.
// Returns float buffer of shape (prefix_len, d_model). Caller frees.
static float* build_prefix_cpu(zonos_tts_context* ctx, const std::vector<int32_t>& phoneme_ids, bool unconditioned,
                               int* out_prefix_len) {
    const auto& hp = ctx->hp;
    const auto& cw = ctx->cond_w;
    const auto& cs = ctx->cond_state;
    const int d = (int)hp.d_model;

    // 1. Phoneme embeddings: (n_phonemes, d_model)
    int n_ph = (int)phoneme_ids.size();
    std::vector<float> phoneme_embs((size_t)n_ph * d);
    {
        // phoneme_emb_w: ne = [d_model, phoneme_vocab_size] in ggml
        for (int i = 0; i < n_ph; i++) {
            int pid = phoneme_ids[i];
            if (pid < 0 || pid >= (int)hp.phoneme_vocab_size)
                pid = 1; // UNK
            tensor_get_row_f32(cw.phoneme_emb_w, pid, &phoneme_embs[(size_t)i * d], d);
        }
    }

    // 2. Speaker embedding: (1, d_model)
    std::vector<float> speaker_out(d, 0.0f);
    if (unconditioned) {
        auto tmp = tensor_to_float(cw.speaker_uncond);
        std::memcpy(speaker_out.data(), tmp.data(), (size_t)d * sizeof(float));
    } else {
        // Linear(128, 2048): out = input @ W^T + bias
        auto proj_w = tensor_to_float(cw.speaker_proj_w);
        auto proj_b = tensor_to_float(cw.speaker_proj_b);
        for (int i = 0; i < d; i++) {
            float sum = proj_b[i];
            for (int j = 0; j < 128 && j < (int)cs.speaker_emb.size(); j++) {
                sum += proj_w[(size_t)i * 128 + j] * cs.speaker_emb[j];
            }
            speaker_out[i] = sum;
        }
    }

    // 3. Emotion Fourier features: (1, d_model)
    std::vector<float> emotion_out(d, 0.0f);
    if (unconditioned) {
        auto tmp = tensor_to_float(cw.emotion.uncond_vec);
        std::memcpy(emotion_out.data(), tmp.data(), (size_t)d * sizeof(float));
    } else {
        // emotion: no explicit min/max in config → default [0, 1]
        compute_fourier_features(cs.emotion, 8, cw.emotion.weight, emotion_out.data(), d, 0.0f, 1.0f);
    }

    // 4. Fmax Fourier features: (1, d_model)
    std::vector<float> fmax_out(d, 0.0f);
    if (unconditioned) {
        auto tmp = tensor_to_float(cw.fmax.uncond_vec);
        std::memcpy(fmax_out.data(), tmp.data(), (size_t)d * sizeof(float));
    } else {
        float fmax_val = cs.fmax;
        // fmax: min=0, max=24000 (from config)
        compute_fourier_features(&fmax_val, 1, cw.fmax.weight, fmax_out.data(), d, 0.0f, 24000.0f);
    }

    // 5. Pitch std Fourier features: (1, d_model)
    std::vector<float> pitch_out(d, 0.0f);
    if (unconditioned) {
        auto tmp = tensor_to_float(cw.pitch_std.uncond_vec);
        std::memcpy(pitch_out.data(), tmp.data(), (size_t)d * sizeof(float));
    } else {
        float pitch_val = cs.pitch_std;
        // pitch_std: min=0, max=400 (from config)
        compute_fourier_features(&pitch_val, 1, cw.pitch_std.weight, pitch_out.data(), d, 0.0f, 400.0f);
    }

    // 6. Speaking rate Fourier features: (1, d_model)
    std::vector<float> rate_out(d, 0.0f);
    if (unconditioned) {
        auto tmp = tensor_to_float(cw.speaking_rate.uncond_vec);
        std::memcpy(rate_out.data(), tmp.data(), (size_t)d * sizeof(float));
    } else {
        float rate_val = cs.speaking_rate;
        // speaking_rate: min=0, max=40 (from config)
        compute_fourier_features(&rate_val, 1, cw.speaking_rate.weight, rate_out.data(), d, 0.0f, 40.0f);
    }

    // 7. Language integer embedding: (1, d_model)
    std::vector<float> lang_out(d, 0.0f);
    if (unconditioned) {
        auto tmp = tensor_to_float(cw.lang_uncond);
        std::memcpy(lang_out.data(), tmp.data(), (size_t)d * sizeof(float));
    } else {
        // IntegerConditioner has min_val=-1, max_val=126 → embedding index = language_id - min_val = language_id + 1
        // Embedding table has 128 entries (max_val - min_val + 1 = 126 - (-1) + 1 = 128)
        int lid = cs.language_id + 1;
        if (lid < 0 || lid > 127)
            lid = 0;
        tensor_get_row_f32(cw.lang_emb_w, lid, lang_out.data(), d);
    }

    // Concatenate: (n_ph + 6, d_model)
    int prefix_len = n_ph + 6; // phonemes + speaker + emotion + fmax + pitch + rate + lang
    std::vector<float> concat((size_t)prefix_len * d);
    // Copy phoneme embeddings
    std::memcpy(concat.data(), phoneme_embs.data(), (size_t)n_ph * d * sizeof(float));
    int offset = n_ph;
    std::memcpy(&concat[(size_t)offset * d], speaker_out.data(), (size_t)d * sizeof(float));
    offset++;
    std::memcpy(&concat[(size_t)offset * d], emotion_out.data(), (size_t)d * sizeof(float));
    offset++;
    std::memcpy(&concat[(size_t)offset * d], fmax_out.data(), (size_t)d * sizeof(float));
    offset++;
    std::memcpy(&concat[(size_t)offset * d], pitch_out.data(), (size_t)d * sizeof(float));
    offset++;
    std::memcpy(&concat[(size_t)offset * d], rate_out.data(), (size_t)d * sizeof(float));
    offset++;
    std::memcpy(&concat[(size_t)offset * d], lang_out.data(), (size_t)d * sizeof(float));

    // LayerNorm + Linear projection (CPU)
    auto norm_w_data = tensor_to_float(cw.norm_w);
    auto norm_b_data = tensor_to_float(cw.norm_b);
    auto proj_w_data = tensor_to_float(cw.proj_w);
    auto proj_b_data = tensor_to_float(cw.proj_b);

    float* result = (float*)malloc((size_t)prefix_len * d * sizeof(float));
    if (!result)
        return nullptr;

    // Python: self.norm(self.project(concat)) — project FIRST, then LayerNorm
    const float eps = hp.norm_eps;
    for (int t = 0; t < prefix_len; t++) {
        float* row = &concat[(size_t)t * d];
        // Step 1: Linear projection: projected = row @ proj_w^T + proj_b
        float projected[2048]; // d_model = 2048
        for (int i = 0; i < d; i++) {
            float sum = proj_b_data[i];
            for (int j = 0; j < d; j++) {
                sum += proj_w_data[(size_t)i * d + j] * row[j];
            }
            projected[i] = sum;
        }
        // Step 2: LayerNorm on the projected output
        float mean = 0.0f;
        for (int i = 0; i < d; i++)
            mean += projected[i];
        mean /= (float)d;
        float var = 0.0f;
        for (int i = 0; i < d; i++) {
            float diff = projected[i] - mean;
            var += diff * diff;
        }
        var /= (float)d;
        float inv_std = 1.0f / std::sqrt(var + eps);
        float* out_row = &result[(size_t)t * d];
        for (int i = 0; i < d; i++) {
            out_row[i] = (projected[i] - mean) * inv_std * norm_w_data[i] + norm_b_data[i];
        }
    }

    *out_prefix_len = prefix_len;
    return result;
}

} // namespace

// -----------------------------------------------------------------------
// Graph builders
// -----------------------------------------------------------------------

namespace {

// Build a transformer decode graph for T tokens at position n_past.
// Input: embeddings (d_model, T). Output: logits for all 9 codebooks
// concatenated: (head_vocab_size * n_codebooks, 1) for the last token.
static ggml_cgraph* build_graph_backbone(zonos_tts_context* ctx, int n_past, int T, ggml_tensor* use_kv_k = nullptr,
                                         ggml_tensor* use_kv_v = nullptr) {
    const auto& hp = ctx->hp;
    const int d = (int)hp.d_model;
    const int n_q = (int)hp.n_heads;
    const int n_kv = (int)hp.n_kv_heads;
    const int hd = (int)hp.head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.norm_eps;
    const float theta = hp.rope_theta;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int Lk = n_past + T;

    // Use specified KV cache or default to ctx->kv_k/kv_v
    ggml_tensor* kv_k_ptr = use_kv_k ? use_kv_k : ctx->kv_k;
    ggml_tensor* kv_v_ptr = use_kv_v ? use_kv_v : ctx->kv_v;
    GGML_ASSERT(kv_k_ptr && kv_v_ptr && Lk <= ctx->kv_max_ctx);

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_kv_grp,
        /*n_ctx_orig*/ 4096,
        /*rope_theta*/ theta,
        /*rope_beta_fast*/ 32.0f,
        /*rope_beta_slow*/ 1.0f,
        /*attn_scale*/ attn_scale,
        /*qk_norm_eps*/ 0.0f,
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
    };
    // Zonos uses consecutive-pair RoPE (x[2i],x[2i+1]) not half-split (x[i],x[i+d/2]).
    kvp.rope_type = GGML_ROPE_TYPE_NORMAL;

    // Helper to upcast F16 tensors to F32 for elementwise ops
    auto cast_f32 = [&](ggml_tensor* t) -> ggml_tensor* {
        if (!t)
            return t;
        if (t->type != GGML_TYPE_F32) {
            return ggml_cast(ctx0, t, GGML_TYPE_F32);
        }
        return t;
    };

    ggml_tensor* cur = embeds;
    for (uint32_t il = 0; il < hp.n_layer; il++) {
        const auto& layer = ctx->backbone.layers[il];
        ggml_tensor* residual = cur;

        // Pre-attention LayerNorm (with bias)
        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, cast_f32(layer.attn_norm_w));
        cur = ggml_add(ctx0, cur, cast_f32(layer.attn_norm_b));

        // Self-attention with fused QKV and KV cache
        ggml_tensor* attn = core_attn::kv_self_attn(ctx0, gf, cur, nullptr, nullptr, nullptr, layer.attn_output_w,
                                                    nullptr, nullptr, positions, (T == 1) ? nullptr : causal_mask,
                                                    kv_k_ptr, kv_v_ptr, (int)il, n_past, kvp, layer.attn_qkv_w);
        cur = ggml_add(ctx0, residual, attn);

        // Pre-FFN LayerNorm (with bias)
        residual = cur;
        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, cast_f32(layer.ffn_norm_w));
        cur = ggml_add(ctx0, cur, cast_f32(layer.ffn_norm_b));

        // GatedMLP (Zonos): y, gate = fc1(x).chunk(2, dim=-1); fc2(y * silu(gate))
        // The first chunk is y (no activation), the second is gate (silu).
        // This is the OPPOSITE of the standard swiglu_fused_gate_up convention
        // which applies silu to the first chunk.
        {
            ggml_tensor* gate_up = ggml_mul_mat(ctx0, layer.ffn_gate_up_w, cur);
            const int ff = (int)hp.ff_dim;
            const size_t ts = ggml_type_size(gate_up->type);
            ggml_tensor* y = ggml_view_2d(ctx0, gate_up, ff, (int)gate_up->ne[1], gate_up->nb[1], 0);
            ggml_tensor* gate = ggml_view_2d(ctx0, gate_up, ff, (int)gate_up->ne[1], gate_up->nb[1], (size_t)ff * ts);
            cur = ggml_add(ctx0, residual,
                           ggml_mul_mat(ctx0, layer.ffn_down_w, ggml_mul(ctx0, y, ggml_silu(ctx0, gate))));
        }
    }

    // Final LayerNorm
    cur = ggml_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, cast_f32(ctx->backbone.output_norm_w));
    cur = ggml_add(ctx0, cur, cast_f32(ctx->backbone.output_norm_b));

    // Take last token only
    if (T > 1) {
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }
    ggml_set_name(cur, "last_hidden");
    ggml_build_forward_expand(gf, cur);

    // Project to all 9 codebook heads, concatenate logits
    // Each head: (head_vocab_size, d_model) @ (d_model, 1) -> (head_vocab_size, 1)
    // Concatenate: (head_vocab_size * n_codebooks, 1)
    ggml_tensor* all_logits = nullptr;
    for (uint32_t k = 0; k < hp.n_codebooks; k++) {
        ggml_tensor* head_logits = ggml_mul_mat(ctx0, ctx->head_w[k], cur);
        if (k == 0) {
            all_logits = head_logits;
        } else {
            all_logits = ggml_concat(ctx0, all_logits, head_logits, 0);
        }
    }
    ggml_set_name(all_logits, "logits");
    ggml_build_forward_expand(gf, all_logits);
    ggml_free(ctx0);
    return gf;
}

// Run the backbone graph, return logits for all 9 codebooks at the last position.
// Returns allocated float array of size (head_vocab_size * n_codebooks). Caller frees.
// If out_hidden is non-null and points to a buffer of size d_model floats, the
// last-token hidden state (after final LayerNorm, before head projection) is written there.
static float* run_backbone(zonos_tts_context* ctx, const float* embeds, int T, int n_past,
                           ggml_tensor* use_kv_k = nullptr, ggml_tensor* use_kv_v = nullptr,
                           float* out_hidden = nullptr) {
    if (n_past + T > ctx->kv_max_ctx) {
        fprintf(stderr, "zonos_tts: kv overflow (%d+%d > %d)\n", n_past, T, ctx->kv_max_ctx);
        return nullptr;
    }
    const auto& hp = ctx->hp;
    const int d = (int)hp.d_model;
    const int total_logits = (int)hp.head_vocab_size * (int)hp.n_codebooks;
    const int Lk = n_past + T;

    std::vector<int32_t> positions(T);
    for (int i = 0; i < T; i++) {
        positions[i] = n_past + i;
    }

    std::vector<ggml_fp16_t> mask;
    if (T > 1) {
        mask.assign((size_t)Lk * T, ggml_fp32_to_fp16(0.0f));
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < T; q++) {
            for (int k = n_past + q + 1; k < Lk; k++) {
                mask[(size_t)q * Lk + k] = neg_inf;
            }
        }
    }

    ggml_cgraph* gf = build_graph_backbone(ctx, n_past, T, use_kv_k, use_kv_v);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "zonos_tts: failed to alloc backbone graph\n");
        return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), embeds, 0, (size_t)d * T * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));
    if (T > 1) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));
    }
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "zonos_tts: backbone compute failed\n");
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    float* r = (float*)malloc((size_t)total_logits * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)total_logits * sizeof(float));

    // Read hidden state when requested (programmatic API or dump dir)
    ggml_tensor* hs = ggml_graph_get_tensor(gf, "last_hidden");
    if (hs) {
        if (out_hidden) {
            ggml_backend_tensor_get(hs, out_hidden, 0, (size_t)d * sizeof(float));
        } else if (n_past == 0) {
            const char* ddir = getenv("ZONOS_CPP_DUMP_DIR");
            if (ddir) {
                std::vector<float> hs_buf(d);
                ggml_backend_tensor_get(hs, hs_buf.data(), 0, (size_t)d * sizeof(float));
                char dp[512];
                snprintf(dp, sizeof(dp), "%s/cpp_last_hidden_%s.npy", ddir,
                         (use_kv_k == ctx->kv_k || !use_kv_k) ? "cond" : "uncond");
                FILE* df = fopen(dp, "wb");
                if (df) {
                    const char magic[] = "\x93NUMPY\x01\x00";
                    fwrite(magic, 1, 8, df);
                    char hdr[128];
                    int hlen =
                        snprintf(hdr, sizeof(hdr), "{'descr': '<f4', 'fortran_order': False, 'shape': (%d,), }", d);
                    int padded = ((hlen + 10 + 63) / 64) * 64 - 10;
                    while (hlen < padded)
                        hdr[hlen++] = ' ';
                    hdr[hlen - 1] = '\n';
                    uint16_t hlen16 = (uint16_t)hlen;
                    fwrite(&hlen16, 2, 1, df);
                    fwrite(hdr, 1, hlen, df);
                    fwrite(hs_buf.data(), sizeof(float), d, df);
                    fclose(df);
                    fprintf(stderr, "zonos_tts: dumped last_hidden (%s) to %s\n",
                            (use_kv_k == ctx->kv_k || !use_kv_k) ? "cond" : "uncond", dp);
                }
            }
        }
    }

    return r;
}

// Build codebook input embeddings for a single AR step.
// tokens: [n_codebooks] token IDs for the current delayed frame.
// Returns a (d_model,) vector = sum of all 9 codebook embeddings at those tokens.
static void embed_codebook_tokens(zonos_tts_context* ctx, const int32_t* tokens, float* out_embed) {
    const auto& hp = ctx->hp;
    const int d = (int)hp.d_model;
    std::memset(out_embed, 0, (size_t)d * sizeof(float));

    std::vector<float> row(d);
    for (uint32_t k = 0; k < hp.n_codebooks; k++) {
        int tid = tokens[k];
        if (tid < 0 || tid >= (int)hp.emb_vocab_size)
            tid = (int)hp.masked_token_id;
        // emb_w[k]: ne = [d_model, emb_vocab_size]
        tensor_get_row_f32(ctx->emb_w[k], tid, row.data(), d);
        for (int i = 0; i < d; i++) {
            out_embed[i] += row[i];
        }
    }
}

// Sampling with min_p (Zonos upstream uses min_p=0.1 by default)
static int sample_with_min_p(const float* logits, int n, float temperature, float min_p, uint64_t* rng_state) {
    if (temperature <= 0.0f) {
        // Greedy
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < n; i++) {
            if (logits[i] > bv) {
                bv = logits[i];
                best = i;
            }
        }
        return best;
    }

    // Apply temperature
    std::vector<float> scaled(n);
    float max_l = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_l)
            max_l = logits[i];
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double p = std::exp((double)(logits[i] - max_l) / (double)temperature);
        scaled[i] = (float)p;
        sum += p;
    }
    if (sum <= 0.0) {
        return 0;
    }
    for (int i = 0; i < n; i++) {
        scaled[i] = (float)(scaled[i] / sum);
    }

    // Find max probability for min_p threshold
    float max_prob = 0.0f;
    for (int i = 0; i < n; i++) {
        if (scaled[i] > max_prob)
            max_prob = scaled[i];
    }
    float threshold = min_p * max_prob;

    // Filter and renormalize
    double filtered_sum = 0.0;
    for (int i = 0; i < n; i++) {
        if (scaled[i] < threshold) {
            scaled[i] = 0.0f;
        } else {
            filtered_sum += scaled[i];
        }
    }
    if (filtered_sum <= 0.0) {
        // Fallback to greedy
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < n; i++) {
            if (logits[i] > bv) {
                bv = logits[i];
                best = i;
            }
        }
        return best;
    }
    for (int i = 0; i < n; i++) {
        scaled[i] = (float)(scaled[i] / filtered_sum);
    }

    // Sample
    uint64_t x = *rng_state ? *rng_state : 0xdeadbeefcafebabeULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *rng_state = x;
    double r = (double)((x * 0x2545f4914f6cdd1dULL) >> 11) / (double)(1ULL << 53);
    double cum = 0.0;
    for (int i = 0; i < n; i++) {
        cum += scaled[i];
        if (r < cum) {
            return i;
        }
    }
    return n - 1;
}

} // namespace

// -----------------------------------------------------------------------
// Diff-harness stage API
// -----------------------------------------------------------------------

float* zonos_tts_build_conditioning_prefix(struct zonos_tts_context* ctx, const char* text, int* out_prefix_len,
                                           int* out_d_model) {
    if (!ctx || !text || !out_prefix_len || !out_d_model)
        return nullptr;

    const auto& hp = ctx->hp;
    const int d = (int)hp.d_model;

    const char* lang = "en-us";
    if (ctx->cond_state.language_id >= 0 && ctx->cond_state.language_id < (int)ctx->cond_state.language_codes.size())
        lang = ctx->cond_state.language_codes[ctx->cond_state.language_id].c_str();

    auto phoneme_ids = tokenize_text_full(text, lang);
    int cond_len = 0, uncond_len = 0;
    float* cond = build_prefix_cpu(ctx, phoneme_ids, false, &cond_len);
    float* uncond = build_prefix_cpu(ctx, phoneme_ids, true, &uncond_len);
    if (!cond || !uncond || cond_len != uncond_len) {
        free(cond);
        free(uncond);
        return nullptr;
    }

    // Return layout: (2 * prefix_len, d_model) — cond rows first, uncond rows second.
    // Matches Python prepare_conditioning() which does torch.cat([cond, uncond]).
    float* out = (float*)malloc((size_t)2 * cond_len * d * sizeof(float));
    if (!out) {
        free(cond);
        free(uncond);
        return nullptr;
    }
    std::memcpy(out, cond, (size_t)cond_len * d * sizeof(float));
    std::memcpy(out + (size_t)cond_len * d, uncond, (size_t)uncond_len * d * sizeof(float));
    free(cond);
    free(uncond);

    *out_prefix_len = cond_len;
    *out_d_model = d;
    return out;
}

float* zonos_tts_get_prefill_hidden(struct zonos_tts_context* ctx, const char* text, int* out_d_model) {
    if (!ctx || !text || !out_d_model)
        return nullptr;

    const auto& hp = ctx->hp;
    const int d = (int)hp.d_model;

    const char* lang = "en-us";
    if (ctx->cond_state.language_id >= 0 && ctx->cond_state.language_id < (int)ctx->cond_state.language_codes.size())
        lang = ctx->cond_state.language_codes[ctx->cond_state.language_id].c_str();

    auto phoneme_ids = tokenize_text_full(text, lang);
    int cond_len = 0, uncond_len = 0;
    float* cond_prefix = build_prefix_cpu(ctx, phoneme_ids, false, &cond_len);
    float* uncond_prefix = build_prefix_cpu(ctx, phoneme_ids, true, &uncond_len);
    if (!cond_prefix || !uncond_prefix) {
        free(cond_prefix);
        free(uncond_prefix);
        return nullptr;
    }

    // Append mask frame
    {
        std::vector<float> mask_emb(d, 0.0f);
        std::vector<float> row(d);
        for (uint32_t k = 0; k < hp.n_codebooks; k++) {
            tensor_get_row_f32(ctx->emb_w[k], (int)hp.masked_token_id, row.data(), d);
            for (int i = 0; i < d; i++)
                mask_emb[i] += row[i];
        }
        auto extend = [&](float*& p, int& len) {
            float* e = (float*)malloc((size_t)(len + 1) * d * sizeof(float));
            std::memcpy(e, p, (size_t)len * d * sizeof(float));
            std::memcpy(e + (size_t)len * d, mask_emb.data(), (size_t)d * sizeof(float));
            free(p);
            p = e;
            len++;
        };
        extend(cond_prefix, cond_len);
        extend(uncond_prefix, uncond_len);
    }

    int kv_need = std::max(cond_len, uncond_len) + 2;
    if (!kv_alloc(ctx, kv_need)) {
        free(cond_prefix);
        free(uncond_prefix);
        return nullptr;
    }
    ggml_backend_buffer_clear(ctx->kv_buf, 0);

    // Return layout: (2, d_model) — row 0 = cond, row 1 = uncond
    float* out = (float*)malloc((size_t)2 * d * sizeof(float));
    if (!out) {
        free(cond_prefix);
        free(uncond_prefix);
        return nullptr;
    }

    float* logits_u =
        run_backbone(ctx, uncond_prefix, uncond_len, 0, ctx->kv_k_uncond, ctx->kv_v_uncond, out + (size_t)d);
    free(logits_u);
    float* logits_c = run_backbone(ctx, cond_prefix, cond_len, 0, ctx->kv_k, ctx->kv_v, out);
    free(logits_c);
    free(cond_prefix);
    free(uncond_prefix);

    *out_d_model = d;
    return out;
}

float* zonos_tts_run_ar_steps_dump(struct zonos_tts_context* ctx, const char* text, int n_steps_req, int* out_n_steps,
                                   int* out_n_cb, int* out_vocab) {
    if (!ctx || !text || n_steps_req <= 0 || !out_n_steps || !out_n_cb || !out_vocab)
        return nullptr;

    const auto& hp = ctx->hp;
    const int d = (int)hp.d_model;
    const int n_cb = (int)hp.n_codebooks;
    const int vocab = (int)hp.head_vocab_size;
    const int eos_id = (int)hp.eos_token_id;
    const int mask_id = (int)hp.masked_token_id;
    const float cfg_scale = ctx->params.cfg_scale;
    const bool use_cfg = (cfg_scale != 1.0f);

    // Slot 0 = prefill logits, slots 1..n = AR step 0..n-1 logits
    const int total_slots = 1 + n_steps_req;
    float* result = (float*)malloc((size_t)total_slots * n_cb * vocab * sizeof(float));
    if (!result)
        return nullptr;

    // ── Phonemize and build conditioning prefix ──
    const char* lang = "en-us";
    if (ctx->cond_state.language_id >= 0 && ctx->cond_state.language_id < (int)ctx->cond_state.language_codes.size())
        lang = ctx->cond_state.language_codes[ctx->cond_state.language_id].c_str();
    auto phoneme_ids = tokenize_text_full(text, lang);

    int cond_len = 0, uncond_len = 0;
    float* cond_prefix = build_prefix_cpu(ctx, phoneme_ids, false, &cond_len);
    float* uncond_prefix = build_prefix_cpu(ctx, phoneme_ids, true, &uncond_len);
    if (!cond_prefix || !uncond_prefix) {
        free(cond_prefix);
        free(uncond_prefix);
        free(result);
        return nullptr;
    }

    // Append mask frame to each prefix
    {
        std::vector<float> mask_emb(d, 0.0f);
        std::vector<float> row(d);
        for (uint32_t k = 0; k < hp.n_codebooks; k++) {
            tensor_get_row_f32(ctx->emb_w[k], mask_id, row.data(), d);
            for (int i = 0; i < d; i++)
                mask_emb[i] += row[i];
        }
        auto extend = [&](float*& p, int& len) {
            float* e = (float*)malloc((size_t)(len + 1) * d * sizeof(float));
            std::memcpy(e, p, (size_t)len * d * sizeof(float));
            std::memcpy(e + (size_t)len * d, mask_emb.data(), (size_t)d * sizeof(float));
            free(p);
            p = e;
            len++;
        };
        extend(cond_prefix, cond_len);
        if (uncond_prefix)
            extend(uncond_prefix, uncond_len);
    }

    // Allocate KV cache
    int kv_need = cond_len + n_steps_req + 16;
    if (!kv_alloc(ctx, kv_need)) {
        free(cond_prefix);
        free(uncond_prefix);
        free(result);
        return nullptr;
    }

    ggml_backend_buffer_clear(ctx->kv_buf, 0);

    float* logits_cond = nullptr;
    float* logits_uncond = nullptr;
    int n_past = cond_len, n_past_u = uncond_len;

    if (use_cfg && uncond_prefix) {
        logits_uncond = run_backbone(ctx, uncond_prefix, uncond_len, 0, ctx->kv_k_uncond, ctx->kv_v_uncond);
    }
    logits_cond = run_backbone(ctx, cond_prefix, cond_len, 0, ctx->kv_k, ctx->kv_v);
    free(cond_prefix);
    free(uncond_prefix);

    if (!logits_cond) {
        free(logits_uncond);
        free(result);
        return nullptr;
    }

    // Slot 0: prefill CFG logits (no EOS masking — matches Python call 0 before logit_bias)
    {
        float* slot = &result[0];
        if (use_cfg && logits_uncond) {
            for (int i = 0; i < n_cb * vocab; i++)
                slot[i] = logits_uncond[i] + cfg_scale * (logits_cond[i] - logits_uncond[i]);
        } else {
            std::memcpy(slot, logits_cond, (size_t)n_cb * vocab * sizeof(float));
        }
    }

    // AR steps
    std::vector<float> cfg_buf((size_t)n_cb * vocab);
    int steps_done = 0;

    for (int step = 0; step < n_steps_req; step++) {
        // CFG blend + EOS masking (matches Python logits + logit_bias)
        if (use_cfg && logits_uncond) {
            for (int i = 0; i < n_cb * vocab; i++)
                cfg_buf[i] = logits_uncond[i] + cfg_scale * (logits_cond[i] - logits_uncond[i]);
            for (int k = 1; k < n_cb; k++)
                cfg_buf[(size_t)k * vocab + eos_id] = -INFINITY;
        } else {
            std::memcpy(cfg_buf.data(), logits_cond, (size_t)n_cb * vocab * sizeof(float));
        }

        // Store in slot step+1
        float* slot = &result[(size_t)(step + 1) * n_cb * vocab];
        std::memcpy(slot, cfg_buf.data(), (size_t)n_cb * vocab * sizeof(float));

        // Greedy sample (temperature=0) — deterministic, matches Python with same logits
        std::vector<int32_t> new_tokens(n_cb);
        for (int k = 0; k < n_cb; k++) {
            const float* cb_logits = &cfg_buf[(size_t)k * vocab];
            int best = 0;
            for (int i = 1; i < vocab; i++)
                if (cb_logits[i] > cb_logits[best])
                    best = i;
            new_tokens[k] = best;
        }

        free(logits_cond);
        logits_cond = nullptr;
        steps_done++;

        if (new_tokens[0] == eos_id)
            break;

        // Build delayed embedding.
        // apply_delay_pattern shifts codebook k by k+1 positions, so codebook k
        // is first non-masked at delayed position k+1. In the AR loop this means
        // codebook k is visible starting at AR iteration t >= k (0-indexed), which
        // maps directly to C++ step >= k.
        std::vector<int32_t> delayed(n_cb, mask_id);
        for (int k = 0; k < n_cb; k++)
            if (step >= k)
                delayed[k] = new_tokens[k];
        std::vector<float> embed(d);
        embed_codebook_tokens(ctx, delayed.data(), embed.data());

        logits_cond = run_backbone(ctx, embed.data(), 1, n_past, ctx->kv_k, ctx->kv_v);
        n_past++;
        if (use_cfg) {
            free(logits_uncond);
            logits_uncond = run_backbone(ctx, embed.data(), 1, n_past_u, ctx->kv_k_uncond, ctx->kv_v_uncond);
            n_past_u++;
        }
        if (!logits_cond)
            break;
    }

    free(logits_cond);
    free(logits_uncond);

    *out_n_steps = 1 + steps_done; // slot 0 = prefill + AR steps
    *out_n_cb = n_cb;
    *out_vocab = vocab;
    return result;
}

// -----------------------------------------------------------------------
// Synthesis: AR decode with CFG + multi-codebook delay pattern
// -----------------------------------------------------------------------

int32_t* zonos_tts_synthesize_codes(struct zonos_tts_context* ctx, const char* text, int* out_n_codes,
                                    int* out_n_codebooks) {
    if (!ctx || !text)
        return nullptr;
    if (out_n_codes)
        *out_n_codes = 0;
    if (out_n_codebooks)
        *out_n_codebooks = 0;

    const auto& hp = ctx->hp;
    const int d = (int)hp.d_model;
    const int n_cb = (int)hp.n_codebooks;
    const int vocab = (int)hp.head_vocab_size;
    const int eos_id = (int)hp.eos_token_id;
    const int mask_id = (int)hp.masked_token_id;
    const float cfg_scale = ctx->params.cfg_scale;
    const float temperature = ctx->params.temperature;
    const float min_p = 0.1f; // upstream default
    const int max_steps = ctx->params.max_audio_tokens > 0 ? ctx->params.max_audio_tokens : (86 * 10);

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "zonos_tts: synthesize_codes '%s'\n", text);
    }

    // Step 1: Tokenize text -> IPA phonemes -> phoneme IDs
    // Determine language from cond_state (default en-us = index 25)
    const char* lang = "en-us";
    if (ctx->cond_state.language_id >= 0 && ctx->cond_state.language_id < (int)ctx->cond_state.language_codes.size()) {
        lang = ctx->cond_state.language_codes[ctx->cond_state.language_id].c_str();
    }
    auto phoneme_ids = tokenize_text_full(text, lang);
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "zonos_tts: %zu phoneme tokens (lang=%s)\n", phoneme_ids.size(), lang);
    }

    // Step 2: Build conditioning prefixes (conditioned + unconditioned)
    int cond_len = 0, uncond_len = 0;
    float* cond_prefix = build_prefix_cpu(ctx, phoneme_ids, false, &cond_len);
    float* uncond_prefix = build_prefix_cpu(ctx, phoneme_ids, true, &uncond_len);
    if (!cond_prefix || !uncond_prefix) {
        free(cond_prefix);
        free(uncond_prefix);
        fprintf(stderr, "zonos_tts: failed to build conditioning prefix\n");
        return nullptr;
    }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "zonos_tts: prefix_len=%d (cond=%d uncond=%d) cfg=%.1f temp=%.2f max_steps=%d\n",
                cond_len + uncond_len, cond_len, uncond_len, cfg_scale, temperature, max_steps);
    }

    // Dump conditioning prefix for diff testing (ZONOS_CPP_DUMP_DIR controls path).
    // Python prepare_conditioning() returns cat([cond, uncond]) → (2, T, d_model).
    // We write the same layout: cond rows first, then uncond rows.
    {
        const char* dump_dir = getenv("ZONOS_CPP_DUMP_DIR");
        if (dump_dir) {
            // Write phoneme IDs
            {
                char path[512];
                snprintf(path, sizeof(path), "%s/cpp_phoneme_ids.txt", dump_dir);
                FILE* f = fopen(path, "w");
                if (f) {
                    for (int id : phoneme_ids)
                        fprintf(f, "%d\n", id);
                    fclose(f);
                }
            }
            // Write conditioning prefix as NumPy F32 array shape (2*T, d_model).
            // Each row is a token vector. First cond_len rows = cond, next uncond_len = uncond.
            {
                char path[512];
                snprintf(path, sizeof(path), "%s/cpp_conditioning_prefix.npy", dump_dir);
                FILE* f = fopen(path, "wb");
                if (f) {
                    // NumPy .npy v1.0 header for shape (2*cond_len, d_model) float32 C-order
                    const char magic[] = "\x93NUMPY\x01\x00";
                    fwrite(magic, 1, 8, f);
                    char hdr[128];
                    int hdr_len = snprintf(hdr, sizeof(hdr),
                                           "{'descr': '<f4', 'fortran_order': False, "
                                           "'shape': (%d, %d), }",
                                           cond_len + uncond_len, d);
                    // Pad header to multiple of 64 with spaces, terminated by '\n'
                    int padded = ((hdr_len + 10 + 63) / 64) * 64 - 10;
                    while (hdr_len < padded)
                        hdr[hdr_len++] = ' ';
                    hdr[hdr_len - 1] = '\n';
                    uint16_t hdr_len16 = (uint16_t)(hdr_len);
                    fwrite(&hdr_len16, 2, 1, f);
                    fwrite(hdr, 1, hdr_len, f);
                    fwrite(cond_prefix, sizeof(float), (size_t)cond_len * d, f);
                    if (uncond_prefix)
                        fwrite(uncond_prefix, sizeof(float), (size_t)uncond_len * d, f);
                    fclose(f);
                    fprintf(stderr, "zonos_tts: dumped conditioning prefix (%d+%d x %d) to %s\n", cond_len, uncond_len,
                            d, path);
                }
            }
        }
    }

    // Allocate KV cache: prefix + max decode steps + safety margin
    int kv_need = cond_len + max_steps + n_cb + 16;
    if (!kv_alloc(ctx, kv_need)) {
        free(cond_prefix);
        free(uncond_prefix);
        return nullptr;
    }

    // Step 3: Append initial mask frame to each prefix.
    // Python: _prefill(conditioning, delayed_prefix_audio_codes) where
    // delayed_prefix_audio_codes[..., 0] = [mask_id]*9 for all codebooks.
    // embed_codes sums all 9 codebook embeddings at mask_id → 1 frame of (d_model,).
    // The backbone sees [conditioning (14 tokens), mask_frame (1 token)] = 15 tokens.
    // Without this, the model predicts EOS immediately.
    {
        std::vector<float> mask_frame_embed(d, 0.0f);
        std::vector<float> row(d);
        for (uint32_t k = 0; k < hp.n_codebooks; k++) {
            tensor_get_row_f32(ctx->emb_w[k], (int)hp.masked_token_id, row.data(), d);
            for (int i = 0; i < d; i++)
                mask_frame_embed[i] += row[i];
        }

        // Extend cond_prefix: append mask frame
        float* cond_ext = (float*)malloc((size_t)(cond_len + 1) * d * sizeof(float));
        std::memcpy(cond_ext, cond_prefix, (size_t)cond_len * d * sizeof(float));
        std::memcpy(&cond_ext[(size_t)cond_len * d], mask_frame_embed.data(), (size_t)d * sizeof(float));
        free(cond_prefix);
        cond_prefix = cond_ext;
        cond_len++;

        // Extend uncond_prefix: append same mask frame
        if (uncond_prefix) {
            float* uncond_ext = (float*)malloc((size_t)(uncond_len + 1) * d * sizeof(float));
            std::memcpy(uncond_ext, uncond_prefix, (size_t)uncond_len * d * sizeof(float));
            std::memcpy(&uncond_ext[(size_t)uncond_len * d], mask_frame_embed.data(), (size_t)d * sizeof(float));
            free(uncond_prefix);
            uncond_prefix = uncond_ext;
            uncond_len++;
        }
    }

    // Step 4: Full dual-KV CFG prefill.
    float* logits_cond = nullptr;
    float* logits_uncond = nullptr;
    int n_past = cond_len;
    int n_past_uncond = uncond_len;
    bool use_cfg = (cfg_scale != 1.0f);

    ggml_backend_buffer_clear(ctx->kv_buf, 0);

    if (use_cfg && uncond_prefix) {
        logits_uncond = run_backbone(ctx, uncond_prefix, uncond_len, 0, ctx->kv_k_uncond, ctx->kv_v_uncond);
        if (!logits_uncond) {
            fprintf(stderr, "zonos_tts: prefill (uncond) failed, disabling CFG\n");
            use_cfg = false;
        }
    }

    logits_cond = run_backbone(ctx, cond_prefix, cond_len, 0, ctx->kv_k, ctx->kv_v);
    free(cond_prefix);
    free(uncond_prefix);
    cond_prefix = nullptr;
    uncond_prefix = nullptr;
    if (!logits_cond) {
        free(logits_uncond);
        fprintf(stderr, "zonos_tts: prefill (cond) failed\n");
        return nullptr;
    }

    // Dump prefill logits for diff testing
    {
        int best = 0;
        for (int i = 1; i < vocab; i++)
            if (logits_cond[i] > logits_cond[best])
                best = i;
        fprintf(stderr, "zonos_tts: DIFF cond prefill cb0 argmax=%d (%.2f) first5=[%.2f,%.2f,%.2f,%.2f,%.2f]\n", best,
                logits_cond[best], logits_cond[0], logits_cond[1], logits_cond[2], logits_cond[3], logits_cond[4]);
        if (logits_uncond) {
            int best_u = 0;
            for (int i = 1; i < vocab; i++)
                if (logits_uncond[i] > logits_uncond[best_u])
                    best_u = i;
            fprintf(stderr, "zonos_tts: DIFF uncond prefill cb0 argmax=%d (%.2f)\n", best_u, logits_uncond[best_u]);
        }
        const char* dump_dir = getenv("ZONOS_CPP_DUMP_DIR");
        if (!dump_dir)
            dump_dir = "/mnt/storage/zonos-tts";
        char df_path[512];
        snprintf(df_path, sizeof(df_path), "%s/cpp_prefill_logits.npy", dump_dir);
        FILE* df = fopen(df_path, "wb");
        if (df) {
            // NumPy .npy v1.0: shape (n_codebooks, head_vocab_size) float32
            const char magic[] = "\x93NUMPY\x01\x00";
            fwrite(magic, 1, 8, df);
            char hdr[128];
            int hlen = snprintf(hdr, sizeof(hdr),
                                "{'descr': '<f4', 'fortran_order': False, "
                                "'shape': (%d, %d), }",
                                (int)hp.n_codebooks, (int)hp.head_vocab_size);
            int padded = ((hlen + 10 + 63) / 64) * 64 - 10;
            while (hlen < padded)
                hdr[hlen++] = ' ';
            hdr[hlen - 1] = '\n';
            uint16_t hlen16 = (uint16_t)hlen;
            fwrite(&hlen16, 2, 1, df);
            fwrite(hdr, 1, hlen, df);
            fwrite(logits_cond, sizeof(float), (size_t)hp.n_codebooks * hp.head_vocab_size, df);
            fclose(df);
        }
    }

    // Step 4: AR decode loop with delay pattern
    // The delay pattern shifts codebook k by (k+1) positions.
    // At each step t, we predict tokens for all 9 codebooks.
    // The "valid" codebook k token at step t corresponds to the actual
    // output sequence position (t - k - 1). Masked positions use mask_id.

    std::vector<std::vector<int32_t>> delayed_codes(n_cb); // generated delayed sequences

    // Initialize: first frame is all mask tokens
    std::vector<int32_t> frame_tokens(n_cb, mask_id);

    bool eos_reached = false;
    int n_decode_steps = 0;

    // Temporary buffer for CFG-blended logits
    std::vector<float> cfg_logits_buf((size_t)n_cb * vocab);

    for (int step = 0; step < max_steps && !eos_reached; step++) {
        // Build sampling buffer: CFG blend + EOS masking + repetition penalty.
        // Always write through cfg_logits_buf so the rep-penalty pass has a
        // single non-aliased target regardless of whether CFG is active.
        if (use_cfg && logits_uncond) {
            for (int i = 0; i < n_cb * vocab; i++)
                cfg_logits_buf[i] = logits_uncond[i] + cfg_scale * (logits_cond[i] - logits_uncond[i]);
        } else {
            std::memcpy(cfg_logits_buf.data(), logits_cond, (size_t)n_cb * vocab * sizeof(float));
        }
        // Mask padding tokens (index >= head_vocab_size) to -inf
        for (int k = 0; k < n_cb; k++) {
            for (int i = (int)hp.head_vocab_size; i < vocab; i++)
                cfg_logits_buf[(size_t)k * vocab + i] = -INFINITY;
        }
        // Only codebook 0 can predict EOS (upstream: logit_bias[:, 1:, eos] = -inf)
        for (int k = 1; k < n_cb; k++)
            cfg_logits_buf[(size_t)k * vocab + eos_id] = -INFINITY;

        // Dump CFG-blended logits at step 0 for comparison with Python
        if (step == 0) {
            const char* ddir = getenv("ZONOS_CPP_DUMP_DIR");
            if (ddir) {
                char dp[512];
                snprintf(dp, sizeof(dp), "%s/cpp_cfg_step0_logits.npy", ddir);
                FILE* df = fopen(dp, "wb");
                if (df) {
                    const char magic[] = "\x93NUMPY\x01\x00";
                    fwrite(magic, 1, 8, df);
                    char hdr[128];
                    int hlen =
                        snprintf(hdr, sizeof(hdr), "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }",
                                 n_cb, (int)hp.head_vocab_size);
                    int padded = ((hlen + 10 + 63) / 64) * 64 - 10;
                    while (hlen < padded)
                        hdr[hlen++] = ' ';
                    hdr[hlen - 1] = '\n';
                    uint16_t hlen16 = (uint16_t)hlen;
                    fwrite(&hlen16, 2, 1, df);
                    fwrite(hdr, 1, hlen, df);
                    fwrite(cfg_logits_buf.data(), sizeof(float), (size_t)n_cb * (int)hp.head_vocab_size, df);
                    fclose(df);
                    fprintf(stderr, "zonos_tts: dumped CFG step0 logits to %s\n", dp);
                }
            }
        }

        // Repetition penalty (upstream default: factor=3.0, window=2).
        // Without this the model loops indefinitely and never emits EOS.
        {
            const float rep_factor = 3.0f;
            const int rep_window = 2;
            for (int k = 0; k < n_cb; k++) {
                const auto& hist = delayed_codes[k];
                const int start = std::max(0, (int)hist.size() - rep_window);
                float* cb = &cfg_logits_buf[(size_t)k * vocab];
                for (int i = start; i < (int)hist.size(); ++i) {
                    const int tok = hist[i];
                    if (tok < 0 || tok >= vocab)
                        continue;
                    if (cb[tok] <= 0.0f)
                        cb[tok] *= rep_factor;
                    else
                        cb[tok] /= rep_factor;
                }
            }
        }

        float* sampling_logits = cfg_logits_buf.data();

        // Sample from logits for each codebook
        std::vector<int32_t> new_tokens(n_cb);
        for (int k = 0; k < n_cb; k++) {
            const float* cb_logits = &sampling_logits[k * vocab];
            int tok = sample_with_min_p(cb_logits, vocab, temperature, min_p, &ctx->rng_state);
            new_tokens[k] = tok;
        }
        free(logits_cond);
        logits_cond = nullptr;

        // Per-step debug: print cb0 argmax + EOS logit + all 9 sampled tokens
        if (ctx->params.verbosity >= 1) {
            const float* cb0 = &cfg_logits_buf[0];
            int argmax = 0;
            for (int i = 1; i < vocab; i++)
                if (cb0[i] > cb0[argmax])
                    argmax = i;
            fprintf(stderr, "zonos_tts: step=%d cb0 argmax=%d (%.2f) eos=%.2f tok=[", step, argmax, cb0[argmax],
                    cb0[eos_id]);
            for (int k = 0; k < n_cb; k++)
                fprintf(stderr, k ? ",%d" : "%d", new_tokens[k]);
            fprintf(stderr, "]\n");
        }

        // Check EOS on codebook 0 (after delay offset is accounted for)
        if (new_tokens[0] == eos_id) {
            eos_reached = true;
            // Still record tokens for draining other codebooks
        }

        // Record into delayed codes
        for (int k = 0; k < n_cb; k++) {
            delayed_codes[k].push_back(new_tokens[k]);
        }
        n_decode_steps++;

        if (eos_reached) {
            // Drain: continue generating for remaining codebooks that haven't
            // gotten their delayed EOS yet. We need up to n_cb more steps.
            for (int drain = 0; drain < n_cb - 1; drain++) {
                // Build embed from current tokens
                std::vector<int32_t> drain_tokens(n_cb, mask_id);
                // Use last generated as input
                for (int k = 0; k < n_cb; k++) {
                    drain_tokens[k] = delayed_codes[k].back();
                }
                std::vector<float> embed(d);
                embed_codebook_tokens(ctx, drain_tokens.data(), embed.data());
                float* drain_logits = run_backbone(ctx, embed.data(), 1, n_past);
                n_past++;
                if (!drain_logits)
                    break;
                // Sample only needed codebooks (those that haven't seen EOS yet)
                std::vector<int32_t> drain_new(n_cb, eos_id);
                for (int k = drain + 1; k < n_cb; k++) {
                    const float* cb_logits = &drain_logits[k * vocab];
                    drain_new[k] = sample_with_min_p(cb_logits, vocab, temperature, min_p, &ctx->rng_state);
                }
                free(drain_logits);
                for (int k = 0; k < n_cb; k++) {
                    delayed_codes[k].push_back(drain_new[k]);
                }
            }
            break;
        }

        // Build embedding for next step with the delay pattern applied.
        // apply_delay_pattern shifts codebook k by k+1 positions: codebook k
        // is non-masked starting at AR iteration t >= k (0-indexed) = C++ step >= k.
        std::vector<int32_t> delayed_embed(n_cb, mask_id);
        for (int k = 0; k < n_cb; k++) {
            if (step >= k)
                delayed_embed[k] = new_tokens[k];
        }
        std::vector<float> embed(d);
        embed_codebook_tokens(ctx, delayed_embed.data(), embed.data());

        // Run backbone for single token — conditioned path
        logits_cond = run_backbone(ctx, embed.data(), 1, n_past, ctx->kv_k, ctx->kv_v);
        n_past++;

        // Run backbone for single token — unconditioned path (same input, different KV)
        if (use_cfg) {
            free(logits_uncond);
            logits_uncond = run_backbone(ctx, embed.data(), 1, n_past_uncond, ctx->kv_k_uncond, ctx->kv_v_uncond);
            n_past_uncond++;
        }

        if (!logits_cond) {
            fprintf(stderr, "zonos_tts: AR decode failed at step %d\n", step);
            break;
        }
    }
    if (logits_cond) {
        free(logits_cond);
    }
    free(logits_uncond);

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "zonos_tts: AR generated %d delayed steps (eos=%d)\n", n_decode_steps, eos_reached ? 1 : 0);
    }

    // Step 5: Revert delay pattern to get actual codebook sequences
    std::vector<std::vector<int32_t>> codes;
    revert_delay_pattern(delayed_codes, codes);

    if (codes.empty() || codes[0].empty()) {
        fprintf(stderr, "zonos_tts: no codes generated after delay revert\n");
        return nullptr;
    }

    // Trim EOS tokens from the end of each codebook
    int seq_len = (int)codes[0].size();
    for (int k = 0; k < n_cb; k++) {
        while (seq_len > 0 && codes[k][seq_len - 1] == eos_id) {
            seq_len--;
        }
    }
    if (seq_len <= 0) {
        fprintf(stderr, "zonos_tts: all codes are EOS\n");
        return nullptr;
    }

    // Clamp any remaining EOS/mask tokens to valid codebook range
    for (int k = 0; k < n_cb; k++) {
        codes[k].resize(seq_len);
        for (int i = 0; i < seq_len; i++) {
            if (codes[k][i] < 0 || codes[k][i] >= (int)hp.codebook_size) {
                codes[k][i] = 0;
            }
        }
    }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "zonos_tts: output codes: %d codebooks x %d steps (~%.1f s at 86 tok/s)\n", n_cb, seq_len,
                (float)seq_len / 86.0f);
    }

    // Return interleaved: n_codebooks * seq_len
    int32_t* result = (int32_t*)malloc((size_t)n_cb * seq_len * sizeof(int32_t));
    if (!result)
        return nullptr;
    for (int k = 0; k < n_cb; k++) {
        std::memcpy(&result[(size_t)k * seq_len], codes[k].data(), (size_t)seq_len * sizeof(int32_t));
    }
    if (out_n_codes)
        *out_n_codes = seq_len;
    if (out_n_codebooks)
        *out_n_codebooks = n_cb;
    return result;
}

// -----------------------------------------------------------------------
// DAC codec loading + decode
// -----------------------------------------------------------------------

static bool load_dac_codec(zonos_tts_context* ctx) {
    if (ctx->dac_loaded)
        return true;
    if (ctx->dac_codec_path.empty()) {
        fprintf(stderr, "zonos_tts: no DAC codec path set\n");
        return false;
    }
    const char* path = ctx->dac_codec_path.c_str();

    // Load weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "zonos_dac", wl)) {
        fprintf(stderr, "zonos_tts: failed to load DAC codec from '%s'\n", path);
        return false;
    }
    ctx->dac_ctx_w = wl.ctx;
    ctx->dac_buf_w = wl.buf;

    auto get = [&](const char* name) -> ggml_tensor* {
        auto it = wl.tensors.find(name);
        return (it != wl.tensors.end()) ? it->second : nullptr;
    };

    auto& dw = ctx->dac_w;
    const int n_cb = dw.config.n_codebooks; // 9

    // Quantizer codebooks + out_proj
    // GGUF names: quantizer.quantizers.K.codebook.weight, .out_proj.weight/bias
    dw.quantizers.resize(n_cb);
    for (int k = 0; k < n_cb; k++) {
        char name[128];
        snprintf(name, sizeof(name), "quantizer.quantizers.%d.codebook.weight", k);
        dw.quantizers[k].codebook = get(name);
        snprintf(name, sizeof(name), "quantizer.quantizers.%d.out_proj.weight", k);
        dw.quantizers[k].out_proj_w = get(name);
        snprintf(name, sizeof(name), "quantizer.quantizers.%d.out_proj.bias", k);
        dw.quantizers[k].out_proj_b = get(name);
    }

    // Decoder input conv: decoder.conv1.weight/bias
    dw.in_conv_w = get("decoder.conv1.weight");
    dw.in_conv_b = get("decoder.conv1.bias");

    // Decoder blocks: decoder.block.B.*
    for (int b = 0; b < 4; b++) {
        auto& blk = dw.blocks[b];
        char name[128];

        // Block-level Snake: decoder.block.B.snake1.alpha
        snprintf(name, sizeof(name), "decoder.block.%d.snake1.alpha", b);
        blk.snake_alpha = get(name);

        // ConvTranspose1d: decoder.block.B.conv_t1.weight/bias
        snprintf(name, sizeof(name), "decoder.block.%d.conv_t1.weight", b);
        blk.up_w = get(name);
        snprintf(name, sizeof(name), "decoder.block.%d.conv_t1.bias", b);
        blk.up_b = get(name);

        // 3 ResidualUnits: decoder.block.B.res_unitR.*
        for (int r = 0; r < 3; r++) {
            auto& ru = blk.res[r];
            int ri = r + 1; // res_unit1, res_unit2, res_unit3

            snprintf(name, sizeof(name), "decoder.block.%d.res_unit%d.snake1.alpha", b, ri);
            ru.alpha0 = get(name);
            snprintf(name, sizeof(name), "decoder.block.%d.res_unit%d.conv1.weight", b, ri);
            ru.conv0_w = get(name);
            snprintf(name, sizeof(name), "decoder.block.%d.res_unit%d.conv1.bias", b, ri);
            ru.conv0_b = get(name);
            snprintf(name, sizeof(name), "decoder.block.%d.res_unit%d.snake2.alpha", b, ri);
            ru.alpha1 = get(name);
            snprintf(name, sizeof(name), "decoder.block.%d.res_unit%d.conv2.weight", b, ri);
            ru.conv1_w = get(name);
            snprintf(name, sizeof(name), "decoder.block.%d.res_unit%d.conv2.bias", b, ri);
            ru.conv1_b = get(name);
        }
    }

    // Output Snake + Conv: decoder.snake1.alpha, decoder.conv2.weight/bias
    dw.out_snake_alpha = get("decoder.snake1.alpha");
    dw.out_conv_w = get("decoder.conv2.weight");
    dw.out_conv_b = get("decoder.conv2.bias");

    // Validate critical tensors
    if (!dw.quantizers[0].codebook || !dw.in_conv_w || !dw.out_conv_w) {
        fprintf(stderr, "zonos_tts: DAC codec missing critical tensors\n");
        return false;
    }

    // Permute DAC ConvTranspose1d weights for decomposed col2im path
    {
        const int n = 4;
        ggml_tensor* srcs[4];
        ggml_tensor** dsts_arr[4];
        for (int i = 0; i < n; i++) {
            srcs[i] = dw.blocks[i].up_w;
            dsts_arr[i] = &dw.blocks[i].up_w_perm;
        }
        core_convt::permute_convt1d_weights_batch(srcs, dsts_arr, n, ctx->backend, &ctx->dac_ctx_perm,
                                                  &ctx->dac_buf_perm);
    }

    ctx->dac_loaded = true;
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "zonos_tts: DAC codec loaded (%zu tensors)\n", wl.tensors.size());
    }
    return true;
}

// Run DAC decoder: codes (n_codebooks x T_codes) -> PCM (T_codes * 512)
static float* dac_decode(zonos_tts_context* ctx, const int32_t* codes, int n_codes, int n_codebooks,
                         int* out_n_samples) {
    const auto& dw = ctx->dac_w;
    const int n_cb = n_codebooks;
    const int T = n_codes;

    // Build graph
    const size_t n_tensors = 2000;
    const size_t mem_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead();
    struct ggml_init_params gp = {mem_size, nullptr, true};
    ggml_context* ctx0 = ggml_init(gp);
    if (!ctx0)
        return nullptr;

    // Create code input tensors
    ggml_tensor* codes_in[9] = {};
    for (int k = 0; k < n_cb && k < 9; k++) {
        char name[32];
        snprintf(name, sizeof(name), "dac_codes_%d", k);
        codes_in[k] = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
        ggml_set_name(codes_in[k], name);
        ggml_set_input(codes_in[k]);
    }

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);
    ggml_tensor* pcm_out = core_dac::build_decode_graph(ctx0, dw, codes_in, T, gf);

    // Allocate + set inputs
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "zonos_tts: DAC decode alloc failed\n");
        ggml_free(ctx0);
        return nullptr;
    }

    // Set code inputs (codes layout: codes[k * n_codes + t])
    for (int k = 0; k < n_cb && k < 9; k++) {
        ggml_backend_tensor_set(codes_in[k], &codes[k * n_codes], 0, T * sizeof(int32_t));
    }

    // Compute
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "zonos_tts: DAC decode compute failed\n");
        ggml_free(ctx0);
        return nullptr;
    }

    // Read PCM output
    int n_pcm = (int)pcm_out->ne[0];
    float* pcm = (float*)malloc((size_t)n_pcm * sizeof(float));
    if (!pcm) {
        ggml_free(ctx0);
        return nullptr;
    }
    ggml_backend_tensor_get(pcm_out, pcm, 0, (size_t)n_pcm * sizeof(float));

    ggml_free(ctx0);
    *out_n_samples = n_pcm;

    if (ctx->params.verbosity >= 1) {
        float duration = (float)n_pcm / 44100.0f;
        fprintf(stderr, "zonos_tts: DAC decoded %d codes -> %d samples (%.2f s)\n", n_codes, n_pcm, duration);
    }
    return pcm;
}

float* zonos_tts_synthesize(struct zonos_tts_context* ctx, const char* text, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;

    // Generate codes. Retry up to 3× if EOS is sampled at step 0 — this
    // manifests as zero output frames and is the known failure mode for
    // Q4_K quantization (EOS logit inflated by ~0.9 units at the first
    // AR step). Each retry bumps rng_state with a prime offset so it
    // draws a different token from the same distribution. The original
    // RNG state is restored after the loop.
    int n_codes = 0, n_codebooks = 0;
    const uint64_t orig_rng = ctx->rng_state;
    int32_t* codes = zonos_tts_synthesize_codes(ctx, text, &n_codes, &n_codebooks);
    for (int retry = 1; retry <= 3 && (!codes || n_codes <= 0); retry++) {
        if (codes) {
            free(codes);
            codes = nullptr;
        }
        n_codes = 0;
        n_codebooks = 0;
        ctx->rng_state = orig_rng + (uint64_t)retry * 7919ULL;
        if (ctx->params.verbosity >= 1)
            fprintf(stderr, "zonos_tts: step-0 EOS retry %d (rng bump)\n", retry);
        codes = zonos_tts_synthesize_codes(ctx, text, &n_codes, &n_codebooks);
    }
    ctx->rng_state = orig_rng;
    if (!codes || n_codes <= 0) {
        return nullptr;
    }

    // Dump codes for external verification
    {
        const char* dump_path = "/mnt/storage/zonos-tts/cpp_codes.txt";
        FILE* f = fopen(dump_path, "w");
        if (f) {
            fprintf(f, "%d %d\n", n_codes, n_codebooks);
            for (int t = 0; t < n_codes; t++) {
                for (int k = 0; k < n_codebooks; k++) {
                    fprintf(f, "%d%c", codes[k * n_codes + t], k < n_codebooks - 1 ? ' ' : '\n');
                }
            }
            fclose(f);
            if (ctx->params.verbosity >= 1)
                fprintf(stderr, "zonos_tts: dumped codes to %s\n", dump_path);
        }
    }

    // Load DAC codec lazily on first call
    if (!ctx->dac_codec_path.empty() && !ctx->dac_loaded) {
        if (!load_dac_codec(ctx)) {
            fprintf(stderr, "zonos_tts: DAC codec load failed; codes dumped but no audio output\n");
            free(codes);
            return nullptr;
        }
    }
    if (!ctx->dac_loaded) {
        fprintf(stderr, "zonos_tts: no DAC codec path set -- codes dumped, use external DAC decoder.\n");
        free(codes);
        return nullptr;
    }

    // DAC decode: codes -> 44.1 kHz PCM
    int n_samples = 0;
    float* pcm = dac_decode(ctx, codes, n_codes, n_codebooks, &n_samples);
    free(codes);
    if (!pcm) {
        return nullptr;
    }

    *out_n_samples = n_samples;
    return pcm;
}

// -----------------------------------------------------------------------
// Cleanup
// -----------------------------------------------------------------------

void zonos_tts_codes_free(int32_t* codes) {
    free(codes);
}
void zonos_tts_pcm_free(float* pcm) {
    free(pcm);
}

void zonos_tts_free(struct zonos_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->dac_buf_perm)
        ggml_backend_buffer_free(ctx->dac_buf_perm);
    if (ctx->dac_ctx_perm)
        ggml_free(ctx->dac_ctx_perm);
    if (ctx->dac_buf_w)
        ggml_backend_buffer_free(ctx->dac_buf_w);
    if (ctx->dac_ctx_w)
        ggml_free(ctx->dac_ctx_w);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->buf_w)
        ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

void zonos_tts_set_n_threads(struct zonos_tts_context* ctx, int n_threads) {
    if (ctx)
        ctx->n_threads = n_threads > 0 ? n_threads : 1;
}

void zonos_tts_set_temperature(struct zonos_tts_context* ctx, float temperature) {
    if (ctx)
        ctx->params.temperature = temperature;
}

void zonos_tts_set_seed(struct zonos_tts_context* ctx, uint64_t seed) {
    if (ctx) {
        ctx->params.seed = seed;
        ctx->rng_state = seed ? seed : 0xdeadbeefcafebabeULL;
    }
}

void zonos_tts_set_cfg_scale(struct zonos_tts_context* ctx, float cfg_scale) {
    if (ctx)
        ctx->params.cfg_scale = cfg_scale;
}
