// omnivoice.cpp — runtime for k2-fsa/OmniVoice TTS.
//
// Architecture: Qwen3-0.6B backbone with audio_embeddings + audio_heads
// for masked iterative multi-codebook TTS (SoundStorm-style).
//
// Status (July 2026):
//   ✓ LLM forward (28L Qwen3 with Q/K-norm, standard RoPE)
//   ✓ Masked iterative generation loop
//   ✓ Audio tokenizer decode (HiggsAudioV2 DAC: codes → waveform)
//   ✗ Audio tokenizer encode (reference audio → codes for voice cloning)
//
// Env knobs:
//   OMNIVOICE_DEBUG=1      — verbose per-step trace
//   OMNIVOICE_BENCH=1      — per-stage wall-clock timings
//   OMNIVOICE_DUMP_DIR=/d  — dump intermediate tensors

#include "omnivoice.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <cctype>
#include "core/activation.h"
#include "core/attention.h"
#include "core/bpe.h"
#include "core/conv.h"
#include "core/audio_resample.h"
#include "core/dac_decoder.h"
#include "core/ffn.h"
#include "core/rvq.h"
#include "core/wav_reader.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/tts_ref_cache.h" // shared content-addressed reference-voice cache (issue #265)
#include "core/crispasr_env.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Env helpers
// ---------------------------------------------------------------------------

bool env_bool(const char* k) {
    const char* v = crispasr_env::get(k);
    return v && *v && std::strcmp(v, "0") != 0;
}
const char* env_str(const char* k) {
    const char* v = crispasr_env::get(k);
    return (v && *v) ? v : nullptr;
}

// ---------------------------------------------------------------------------
// Bench instrumentation
// ---------------------------------------------------------------------------

static bool ov_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_OMNIVOICE_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct ov_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit ov_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~ov_bench_stage() {
        if (!ov_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  omnivoice_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ---------------------------------------------------------------------------
// Parallel-for over [0, n) in contiguous chunks (scoring hot loop)
// ---------------------------------------------------------------------------

static void ov_parallel_for(int n_threads, int n, const std::function<void(int, int)>& fn) {
    n_threads = std::max(1, std::min(n_threads, n));
    if (n_threads == 1) {
        fn(0, n);
        return;
    }
    const int chunk = (n + n_threads - 1) / n_threads;
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (int i = 0; i < n_threads; i++) {
        const int a = i * chunk;
        const int b = std::min(n, a + chunk);
        if (a >= b)
            break;
        workers.emplace_back(fn, a, b);
    }
    for (auto& w : workers)
        w.join();
}

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

struct ov_hp {
    // LLM backbone (Qwen3)
    uint32_t n_layers = 28;
    uint32_t d_model = 1024;
    uint32_t n_heads = 16;
    uint32_t n_kv_heads = 8;
    uint32_t head_dim = 128;
    uint32_t ff_dim = 3072;
    uint32_t vocab_size = 151676;
    uint32_t max_pos = 40960;
    float rope_theta = 1000000.0f;
    float rms_norm_eps = 1e-6f;
    bool tie_word_embeddings = true;

    // Audio
    uint32_t audio_vocab_size = 1025;
    uint32_t audio_mask_id = 1024;
    uint32_t n_codebooks = 8;
    std::vector<float> codebook_weights; // [8, 8, 6, 6, 4, 4, 2, 2]

    // Token sentinels
    uint32_t eos_token_id = 151645;
    uint32_t pad_token_id = 151643;
};

// ---------------------------------------------------------------------------
// Model weights
// ---------------------------------------------------------------------------

struct ov_layer {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_output_w = nullptr;
    ggml_tensor* attn_q_norm_w = nullptr;
    ggml_tensor* attn_k_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct ov_model {
    // LLM
    ggml_tensor* token_embd_w = nullptr; // (vocab_size, d_model)
    std::vector<ov_layer> blocks;
    ggml_tensor* output_norm_w = nullptr;

    // Audio
    ggml_tensor* audio_embd_w = nullptr;   // (n_codebooks * audio_vocab_size, d_model)
    ggml_tensor* audio_output_w = nullptr; // (n_codebooks * audio_vocab_size, d_model)
};

struct ov_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
    // Special tokens: <|text_start|>, <|lang_start|>, etc.
    std::unordered_map<std::string, int32_t> special_tokens;
};

// ---------------------------------------------------------------------------
// HiggsAudioV2 tokenizer (decode path only)
// ---------------------------------------------------------------------------

struct ov_higgs_vq {
    ggml_tensor* codebook = nullptr; // (codebook_dim=64, codebook_size=1024)
    ggml_tensor* proj_out_w = nullptr;
    ggml_tensor* proj_out_b = nullptr;
    // Encode side (voice cloning): project_in 1024->64 before the Euclidean NN.
    ggml_tensor* proj_in_w = nullptr; // (hidden=1024, codebook_dim=64)
    ggml_tensor* proj_in_b = nullptr; // (codebook_dim=64)
};

struct ov_higgs_res_unit {
    ggml_tensor* snake1_alpha = nullptr;
    ggml_tensor* conv1_w = nullptr;
    ggml_tensor* conv1_b = nullptr;
    ggml_tensor* snake2_alpha = nullptr;
    ggml_tensor* conv2_w = nullptr;
    ggml_tensor* conv2_b = nullptr;
};

struct ov_higgs_dec_block {
    ggml_tensor* snake_alpha = nullptr;
    ggml_tensor* conv_t1_w = nullptr;
    ggml_tensor* conv_t1_b = nullptr;
    ov_higgs_res_unit res[3]; // dilation 1, 3, 9
};

// --- Encode-side (voice cloning) ------------------------------------------
// DAC acoustic-encoder block: 3 ResidualUnits (d=1,3,9) then Snake + strided
// downsampling conv1 (k = 2*stride). Mirror of ov_higgs_dec_block.
struct ov_higgs_enc_block {
    ov_higgs_res_unit res[3]; // dilation 1, 3, 9
    ggml_tensor* snake1_alpha = nullptr;
    ggml_tensor* conv1_w = nullptr; // strided downsample conv (k=2*stride)
    ggml_tensor* conv1_b = nullptr;
};

// HuBERT feature-extractor conv layer. Layer 0 has a GroupNorm (ln.*); layers
// 1..6 have none. conv is bias-free.
struct ov_sem_conv_layer {
    ggml_tensor* conv_w = nullptr;
    ggml_tensor* ln_w = nullptr; // GroupNorm weight (layer 0 only)
    ggml_tensor* ln_b = nullptr;
};

// HuBERT transformer block (post-norm variant).
struct ov_sem_block {
    ggml_tensor *attn_q_w = nullptr, *attn_q_b = nullptr;
    ggml_tensor *attn_k_w = nullptr, *attn_k_b = nullptr;
    ggml_tensor *attn_v_w = nullptr, *attn_v_b = nullptr;
    ggml_tensor *attn_out_w = nullptr, *attn_out_b = nullptr;
    ggml_tensor *ln_w = nullptr, *ln_b = nullptr;         // post-attention LN
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;       // 768->3072
    ggml_tensor *fc2_w = nullptr, *fc2_b = nullptr;       // 3072->768
    ggml_tensor *ffn_ln_w = nullptr, *ffn_ln_b = nullptr; // final LN
};

// encoder_semantic bridge block: 2 ResidualUnits (ELU-based, dil=1) + conv.
struct ov_encsem_block {
    ov_higgs_res_unit res[2]; // snake fields reused as ELU markers (no alpha)
    ggml_tensor* conv_w = nullptr;
    ggml_tensor* conv_b = nullptr;
};

struct ov_higgs_tokenizer {
    // Quantizer: 8 VQ layers (for OmniVoice's 8 codebooks)
    std::vector<ov_higgs_vq> quantizers;

    // FASTCONV (#254): shared codec-conv fast path (baked F32 kernels + k=1→matmul),
    // so the fork's per-graph F16→F32 conv-kernel cast becomes a no-op. See
    // core_dac::fastconv_cache / docs/perf-sweep/PLAN.md item 1.
    core_dac::fastconv_cache fc;

    // fc2: project from quantizer hidden_size (1024) → DAC hidden (256)
    ggml_tensor* fc2_w = nullptr;
    ggml_tensor* fc2_b = nullptr;

    // DAC decoder
    ggml_tensor* dec_conv1_w = nullptr; // Conv1d(256, 1024, k=7)
    ggml_tensor* dec_conv1_b = nullptr;
    std::vector<ov_higgs_dec_block> dec_blocks; // 5 blocks
    ggml_tensor* dec_snake_alpha = nullptr;     // final Snake1d
    ggml_tensor* dec_conv2_w = nullptr;         // Conv1d(32, 1, k=7)
    ggml_tensor* dec_conv2_b = nullptr;

    // --- Encode side (voice cloning) --------------------------------------
    // fc: project concat([e_acoustic(256), e_semantic(768)]) 1024 -> 1024
    ggml_tensor* fc_w = nullptr;
    ggml_tensor* fc_b = nullptr;
    // DAC acoustic encoder
    ggml_tensor* enc_conv1_w = nullptr; // Conv1d(1, 64, k=7)
    ggml_tensor* enc_conv1_b = nullptr;
    std::vector<ov_higgs_enc_block> enc_blocks; // 5 blocks, strides [8,5,4,2,3]
    ggml_tensor* enc_snake1_alpha = nullptr;    // final Snake1d (2048 ch)
    ggml_tensor* enc_conv2_w = nullptr;         // Conv1d(2048, 256, k=3)
    ggml_tensor* enc_conv2_b = nullptr;
    // HuBERT semantic encoder
    std::vector<ov_sem_conv_layer> sem_conv;                                // 7 feat-extract conv layers
    ggml_tensor *sem_featproj_ln_w = nullptr, *sem_featproj_ln_b = nullptr; // LN(512)
    ggml_tensor *sem_featproj_w = nullptr, *sem_featproj_b = nullptr;       // 512->768
    ggml_tensor *sem_posconv_g = nullptr, *sem_posconv_v = nullptr, *sem_posconv_b = nullptr; // weight-norm pos conv
    ggml_tensor *sem_enc_ln_w = nullptr, *sem_enc_ln_b = nullptr;                             // pre-stack LN
    std::vector<ov_sem_block> sem_blocks;                                                     // 12 transformer layers
    // encoder_semantic bridge
    ggml_tensor* encsem_conv_w = nullptr;       // Conv1d(768,768,k=3,bias=F)
    std::vector<ov_encsem_block> encsem_blocks; // 2 blocks

    // Config
    int n_quantizers = 8;
    int codebook_size = 1024;
    int codebook_dim = 64;
    int hidden_size = 1024; // quantizer output dim
    int dac_hidden = 256;   // DAC input (after fc2)
    int dec_hidden = 1024;  // DAC decoder first conv output
    int n_dec_blocks = 5;
    int upsampling_ratios[5] = {8, 5, 4, 2, 3};
    int dec_channels[6] = {1024, 512, 256, 128, 64, 32};
    int hop_length = 960;    // total upsample = 8*5*4*2*3 = 960
    int sample_rate = 24000; // output sample rate

    // Encode-side config
    int downsampling_ratios[5] = {8, 5, 4, 2, 3};          // acoustic encoder strides
    int enc_channels[6] = {64, 128, 256, 512, 1024, 2048}; // conv1 out .. conv2 in
    int n_sem_layers = 12;
    int sem_d = 768;
    int sem_heads = 12;
    int sem_ff = 3072;
    int n_sem_conv = 7;
    int sem_conv_k[7] = {10, 3, 3, 3, 3, 2, 2};
    int sem_conv_s[7] = {5, 2, 2, 2, 2, 2, 2};
    int semantic_sample_rate = 16000;
    int semantic_downsample_factor = 2;
    float sem_ln_eps = 1e-5f;
    int n_encsem_blocks = 2;

    // Backend
    ggml_context* ctx_w = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    bool loaded = false;
};

// ---------------------------------------------------------------------------
// Generation config
// ---------------------------------------------------------------------------

// Defaults MUST match the blueprint's OmniVoiceGenerationConfig
// (k2-fsa/OmniVoice omnivoice/models/omnivoice.py). The previous values
// (guidance 1.0, class_temp 0.7, layer_penalty 0.5, t_shift 1.0) ran a
// completely different decode policy and degenerated into near-constant
// "silence" codes on every platform.
struct ov_gen_config {
    int num_steps = 32;
    float guidance_scale = 2.0f;
    float class_temperature = 0.0f;
    float position_temperature = 5.0f;
    float layer_penalty_factor = 5.0f;
    float t_shift = 0.1f;
    uint64_t seed = 42;
};

} // namespace

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct omnivoice_context {
    ov_hp hp;
    ov_model model;
    ov_vocab vocab;
    ov_gen_config gen;
    ov_higgs_tokenizer tokenizer;

    int n_threads = 4;
    int verbosity = 1;
    bool use_gpu = false;
    bool flash_attn = false;

    // ggml
    ggml_context* ctx_w = nullptr; // weight context
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;

    // Voice cloning state
    std::vector<int32_t> ref_audio_codes; // (n_codebooks, T_ref) row-major
    int ref_T = 0;
    float ref_rms = 0.0f;
    std::string ref_text;

    // Language / instruct
    std::string language;
    std::string instruct;

    // Speaking-rate multiplier for the target-length estimate (>1 faster/shorter).
    float speed = 1.0f;

    // Audio tokenizer path (separate GGUF)
    std::string tokenizer_path;
};

// ---------------------------------------------------------------------------
// Default params
// ---------------------------------------------------------------------------

struct omnivoice_context_params omnivoice_context_default_params(void) {
    struct omnivoice_context_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.num_steps = 0;
    p.guidance_scale = 0.0f;
    p.class_temperature = 0.0f;
    p.position_temperature = 0.0f;
    p.layer_penalty_factor = 0.0f;
    p.t_shift = 0.0f;
    p.seed = 0;
    p.flash_attn = false;
    return p;
}

namespace {

// ---------------------------------------------------------------------------
// GGUF loading
// ---------------------------------------------------------------------------

static bool load_model(omnivoice_context* ctx, const char* path) {
    auto& hp = ctx->hp;
    auto& m = ctx->model;

    struct gguf_init_params gp = {/*.no_alloc=*/true, /*.ctx=*/&ctx->ctx_w};
    struct gguf_context* gf = gguf_init_from_file(path, gp);
    if (!gf) {
        fprintf(stderr, "omnivoice: failed to open %s\n", path);
        return false;
    }

    auto read_u32 = [&](const char* k, uint32_t dflt) -> uint32_t {
        int idx = gguf_find_key(gf, k);
        return idx >= 0 ? (uint32_t)gguf_get_val_u32(gf, idx) : dflt;
    };
    auto read_f32 = [&](const char* k, float dflt) -> float {
        int idx = gguf_find_key(gf, k);
        return idx >= 0 ? gguf_get_val_f32(gf, idx) : dflt;
    };
    auto read_bool = [&](const char* k, bool dflt) -> bool {
        int idx = gguf_find_key(gf, k);
        return idx >= 0 ? gguf_get_val_bool(gf, idx) : dflt;
    };

    // LLM
    hp.n_layers = read_u32("omnivoice.llm.n_layers", 28);
    hp.d_model = read_u32("omnivoice.llm.d_model", 1024);
    hp.n_heads = read_u32("omnivoice.llm.n_heads", 16);
    hp.n_kv_heads = read_u32("omnivoice.llm.n_kv_heads", 8);
    hp.head_dim = read_u32("omnivoice.llm.head_dim", 128);
    hp.ff_dim = read_u32("omnivoice.llm.ff_dim", 3072);
    hp.vocab_size = read_u32("omnivoice.llm.vocab_size", 151676);
    hp.max_pos = read_u32("omnivoice.llm.max_pos", 40960);
    hp.rope_theta = read_f32("omnivoice.llm.rope_theta", 1000000.0f);
    hp.rms_norm_eps = read_f32("omnivoice.llm.rms_norm_eps", 1e-6f);
    hp.tie_word_embeddings = read_bool("omnivoice.llm.tie_word_embeddings", true);

    // Audio
    hp.audio_vocab_size = read_u32("omnivoice.audio.vocab_size", 1025);
    hp.audio_mask_id = read_u32("omnivoice.audio.mask_id", 1024);
    hp.n_codebooks = read_u32("omnivoice.audio.n_codebooks", 8);

    // Codebook weights
    {
        int idx = gguf_find_key(gf, "omnivoice.audio.codebook_weights");
        if (idx >= 0) {
            int n = (int)gguf_get_arr_n(gf, idx);
            hp.codebook_weights.resize(n);
            const int32_t* arr = (const int32_t*)gguf_get_arr_data(gf, idx);
            for (int i = 0; i < n; i++) {
                hp.codebook_weights[i] = (float)arr[i];
            }
        } else {
            hp.codebook_weights = {8, 8, 6, 6, 4, 4, 2, 2};
        }
    }

    // Tokens
    hp.eos_token_id = read_u32("omnivoice.eos_token_id", 151645);
    hp.pad_token_id = read_u32("omnivoice.pad_token_id", 151643);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "omnivoice: LLM %uL d=%u heads=%u/%u hd=%u ff=%u vocab=%u\n", hp.n_layers, hp.d_model,
                hp.n_heads, hp.n_kv_heads, hp.head_dim, hp.ff_dim, hp.vocab_size);
        fprintf(stderr, "omnivoice: Audio %u codebooks vocab=%u mask=%u\n", hp.n_codebooks, hp.audio_vocab_size,
                hp.audio_mask_id);
    }

    // Allocate tensors
    auto find = [&](const char* name) -> ggml_tensor* { return ggml_get_tensor(ctx->ctx_w, name); };

    m.token_embd_w = find("llm.token_embd.weight");
    m.output_norm_w = find("llm.output_norm.weight");
    m.audio_embd_w = find("audio_embd.weight");
    m.audio_output_w = find("audio_output.weight");

    m.blocks.resize(hp.n_layers);
    for (uint32_t i = 0; i < hp.n_layers; i++) {
        char buf[128];
        auto L = [&](const char* sfx) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "llm.blk.%u.%s", i, sfx);
            return find(buf);
        };
        auto& b = m.blocks[i];
        b.attn_norm_w = L("attn_norm.weight");
        b.attn_q_w = L("attn_q.weight");
        b.attn_k_w = L("attn_k.weight");
        b.attn_v_w = L("attn_v.weight");
        b.attn_output_w = L("attn_output.weight");
        b.attn_q_norm_w = L("attn_q_norm.weight");
        b.attn_k_norm_w = L("attn_k_norm.weight");
        b.ffn_norm_w = L("ffn_norm.weight");
        b.ffn_gate_w = L("ffn_gate.weight");
        b.ffn_up_w = L("ffn_up.weight");
        b.ffn_down_w = L("ffn_down.weight");
    }

    // Verify critical weights
    const char* required[] = {
        "llm.token_embd.weight",
        "llm.output_norm.weight",
        "audio_embd.weight",
        "audio_output.weight",
    };
    bool ok = true;
    for (const char* name : required) {
        if (!find(name)) {
            fprintf(stderr, "omnivoice: required tensor missing: %s\n", name);
            ok = false;
        }
    }
    for (uint32_t i = 0; i < hp.n_layers && ok; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "llm.blk.%u.attn_q.weight", i);
        if (!find(buf)) {
            fprintf(stderr, "omnivoice: required tensor missing: %s\n", buf);
            ok = false;
        }
    }
    if (!ok) {
        gguf_free(gf);
        return false;
    }

    // Create backend + buffer. use_gpu picks the best GPU (CUDA/Metal/Vulkan);
    // CRISPASR_OMNIVOICE_CPU=1 forces CPU. Only the LLM iterative loop runs here
    // — the DAC codec stays on CPU (see load_tokenizer). Falls back to CPU if
    // GPU init fails.
    {
        const char* e = std::getenv("CRISPASR_OMNIVOICE_CPU");
        const bool force_cpu = e && *e && *e != '0';
        ctx->backend = (ctx->use_gpu && !force_cpu) ? crispasr_init_gpu_backend() : nullptr;
        if (!ctx->backend)
            ctx->backend = ggml_backend_cpu_init();
    }
    if (!ctx->backend) {
        fprintf(stderr, "omnivoice: failed to init backend\n");
        gguf_free(gf);
        return false;
    }
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    if (ctx->verbosity >= 1)
        fprintf(stderr, "omnivoice: compute backend = %s\n", ggml_backend_name(ctx->backend));

    ctx->buf_w = ggml_backend_alloc_ctx_tensors(ctx->ctx_w, ctx->backend);
    if (!ctx->buf_w) {
        fprintf(stderr, "omnivoice: failed to allocate weight buffer\n");
        gguf_free(gf);
        return false;
    }

    // Load tensor data from file
    {
        FILE* fp = fopen(path, "rb");
        if (!fp) {
            fprintf(stderr, "omnivoice: cannot reopen %s\n", path);
            gguf_free(gf);
            return false;
        }
        int n_tensors = gguf_get_n_tensors(gf);
        for (int i = 0; i < n_tensors; i++) {
            const char* name = gguf_get_tensor_name(gf, i);
            ggml_tensor* t = ggml_get_tensor(ctx->ctx_w, name);
            if (!t)
                continue;
            size_t offset = gguf_get_data_offset(gf) + gguf_get_tensor_offset(gf, i);
            fseek(fp, (long)offset, SEEK_SET);
            size_t nbytes = ggml_nbytes(t);
            std::vector<uint8_t> tmp(nbytes);
            if (fread(tmp.data(), 1, nbytes, fp) != nbytes) {
                fprintf(stderr, "omnivoice: short read for %s\n", name);
                fclose(fp);
                gguf_free(gf);
                return false;
            }
            ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
        }
        fclose(fp);
    }

    // Load vocab
    {
        int idx = gguf_find_key(gf, "tokenizer.ggml.tokens");
        if (idx >= 0) {
            int n = gguf_get_arr_n(gf, idx);
            ctx->vocab.id_to_token.resize(n);
            for (int i = 0; i < n; i++) {
                ctx->vocab.id_to_token[i] = gguf_get_arr_str(gf, idx, i);
                ctx->vocab.token_to_id[ctx->vocab.id_to_token[i]] = i;
            }
            if (ctx->verbosity >= 1) {
                fprintf(stderr, "omnivoice: loaded %d tokens\n", n);
            }
        }

        int midx = gguf_find_key(gf, "tokenizer.ggml.merges");
        if (midx >= 0) {
            int n = gguf_get_arr_n(gf, midx);
            for (int i = 0; i < n; i++) {
                ctx->vocab.merge_rank[gguf_get_arr_str(gf, midx, i)] = i;
            }
            if (ctx->verbosity >= 1) {
                fprintf(stderr, "omnivoice: loaded %d merges\n", n);
            }
        }
    }

    // Load special tokens (omnivoice.special_token_names + _ids)
    {
        int nidx = gguf_find_key(gf, "omnivoice.special_token_names");
        int iidx = gguf_find_key(gf, "omnivoice.special_token_ids");
        if (nidx >= 0 && iidx >= 0) {
            int n = (int)gguf_get_arr_n(gf, nidx);
            const int32_t* ids = (const int32_t*)gguf_get_arr_data(gf, iidx);
            for (int i = 0; i < n; i++) {
                std::string name = gguf_get_arr_str(gf, nidx, i);
                ctx->vocab.special_tokens[name] = ids[i];
            }
            if (ctx->verbosity >= 1) {
                fprintf(stderr, "omnivoice: loaded %d special tokens\n", n);
            }
        }
    }

    gguf_free(gf);

    if (ctx->verbosity >= 1) {
        size_t total = ggml_backend_buffer_get_size(ctx->buf_w);
        fprintf(stderr, "omnivoice: loaded %s (%.2f GB)\n", path, total / 1e9);
    }

    return true;
}

// ---------------------------------------------------------------------------
// HiggsAudioV2 tokenizer loading
// ---------------------------------------------------------------------------

static bool codec_fastconv_enabled();
static bool codec_gpu_enabled();

static bool load_tokenizer(omnivoice_context* ctx, const char* path) {
    auto& tok = ctx->tokenizer;

    struct gguf_init_params gp = {/*.no_alloc=*/true, /*.ctx=*/&tok.ctx_w};
    struct gguf_context* gf = gguf_init_from_file(path, gp);
    if (!gf) {
        fprintf(stderr, "omnivoice: failed to open tokenizer %s\n", path);
        return false;
    }

    auto find = [&](const char* name) -> ggml_tensor* { return ggml_get_tensor(tok.ctx_w, name); };

    // Quantizer: 8 VQ layers
    tok.n_quantizers = 8; // OmniVoice uses 8 of HiggsAudioV2's quantizers
    tok.quantizers.resize(tok.n_quantizers);
    for (int i = 0; i < tok.n_quantizers; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "quantizer.quantizers.%d.codebook.embed", i);
        tok.quantizers[i].codebook = find(buf);
        snprintf(buf, sizeof(buf), "quantizer.quantizers.%d.project_out.weight", i);
        tok.quantizers[i].proj_out_w = find(buf);
        snprintf(buf, sizeof(buf), "quantizer.quantizers.%d.project_out.bias", i);
        tok.quantizers[i].proj_out_b = find(buf);
    }

    // fc2: project quantizer output → DAC decoder input
    tok.fc2_w = find("fc2.weight");
    tok.fc2_b = find("fc2.bias");

    // DAC decoder
    tok.dec_conv1_w = find("acoustic_decoder.conv1.weight");
    tok.dec_conv1_b = find("acoustic_decoder.conv1.bias");
    tok.dec_snake_alpha = find("acoustic_decoder.snake1.alpha");
    tok.dec_conv2_w = find("acoustic_decoder.conv2.weight");
    tok.dec_conv2_b = find("acoustic_decoder.conv2.bias");

    tok.n_dec_blocks = 5;
    tok.dec_blocks.resize(tok.n_dec_blocks);
    for (int b = 0; b < tok.n_dec_blocks; b++) {
        char buf[128];
        auto& blk = tok.dec_blocks[b];
        snprintf(buf, sizeof(buf), "acoustic_decoder.block.%d.snake1.alpha", b);
        blk.snake_alpha = find(buf);
        snprintf(buf, sizeof(buf), "acoustic_decoder.block.%d.conv_t1.weight", b);
        blk.conv_t1_w = find(buf);
        snprintf(buf, sizeof(buf), "acoustic_decoder.block.%d.conv_t1.bias", b);
        blk.conv_t1_b = find(buf);

        static const char* ru_names[] = {"res_unit1", "res_unit2", "res_unit3"};
        for (int r = 0; r < 3; r++) {
            snprintf(buf, sizeof(buf), "acoustic_decoder.block.%d.%s.snake1.alpha", b, ru_names[r]);
            blk.res[r].snake1_alpha = find(buf);
            snprintf(buf, sizeof(buf), "acoustic_decoder.block.%d.%s.conv1.weight", b, ru_names[r]);
            blk.res[r].conv1_w = find(buf);
            snprintf(buf, sizeof(buf), "acoustic_decoder.block.%d.%s.conv1.bias", b, ru_names[r]);
            blk.res[r].conv1_b = find(buf);
            snprintf(buf, sizeof(buf), "acoustic_decoder.block.%d.%s.snake2.alpha", b, ru_names[r]);
            blk.res[r].snake2_alpha = find(buf);
            snprintf(buf, sizeof(buf), "acoustic_decoder.block.%d.%s.conv2.weight", b, ru_names[r]);
            blk.res[r].conv2_w = find(buf);
            snprintf(buf, sizeof(buf), "acoustic_decoder.block.%d.%s.conv2.bias", b, ru_names[r]);
            blk.res[r].conv2_b = find(buf);
        }
    }

    // --- Encode-side weights (voice cloning). Absent-tolerant: a decode-only
    //     tokenizer still loads; encode paths check for null at use. ---
    for (int i = 0; i < tok.n_quantizers; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "quantizer.quantizers.%d.project_in.weight", i);
        tok.quantizers[i].proj_in_w = find(buf);
        snprintf(buf, sizeof(buf), "quantizer.quantizers.%d.project_in.bias", i);
        tok.quantizers[i].proj_in_b = find(buf);
    }
    tok.fc_w = find("fc.weight");
    tok.fc_b = find("fc.bias");
    // DAC acoustic encoder
    tok.enc_conv1_w = find("acoustic_encoder.conv1.weight");
    tok.enc_conv1_b = find("acoustic_encoder.conv1.bias");
    tok.enc_snake1_alpha = find("acoustic_encoder.snake1.alpha");
    tok.enc_conv2_w = find("acoustic_encoder.conv2.weight");
    tok.enc_conv2_b = find("acoustic_encoder.conv2.bias");
    tok.enc_blocks.resize(5);
    for (int b = 0; b < 5; b++) {
        char buf[160];
        auto& blk = tok.enc_blocks[b];
        static const char* ru_names[] = {"res_unit1", "res_unit2", "res_unit3"};
        for (int r = 0; r < 3; r++) {
            snprintf(buf, sizeof(buf), "acoustic_encoder.block.%d.%s.snake1.alpha", b, ru_names[r]);
            blk.res[r].snake1_alpha = find(buf);
            snprintf(buf, sizeof(buf), "acoustic_encoder.block.%d.%s.conv1.weight", b, ru_names[r]);
            blk.res[r].conv1_w = find(buf);
            snprintf(buf, sizeof(buf), "acoustic_encoder.block.%d.%s.conv1.bias", b, ru_names[r]);
            blk.res[r].conv1_b = find(buf);
            snprintf(buf, sizeof(buf), "acoustic_encoder.block.%d.%s.snake2.alpha", b, ru_names[r]);
            blk.res[r].snake2_alpha = find(buf);
            snprintf(buf, sizeof(buf), "acoustic_encoder.block.%d.%s.conv2.weight", b, ru_names[r]);
            blk.res[r].conv2_w = find(buf);
            snprintf(buf, sizeof(buf), "acoustic_encoder.block.%d.%s.conv2.bias", b, ru_names[r]);
            blk.res[r].conv2_b = find(buf);
        }
        snprintf(buf, sizeof(buf), "acoustic_encoder.block.%d.snake1.alpha", b);
        blk.snake1_alpha = find(buf);
        snprintf(buf, sizeof(buf), "acoustic_encoder.block.%d.conv1.weight", b);
        blk.conv1_w = find(buf);
        snprintf(buf, sizeof(buf), "acoustic_encoder.block.%d.conv1.bias", b);
        blk.conv1_b = find(buf);
    }
    // HuBERT semantic encoder
    tok.sem_conv.resize(tok.n_sem_conv);
    for (int i = 0; i < tok.n_sem_conv; i++) {
        char buf[96];
        snprintf(buf, sizeof(buf), "sem.conv.%d.conv.weight", i);
        tok.sem_conv[i].conv_w = find(buf);
        snprintf(buf, sizeof(buf), "sem.conv.%d.ln.weight", i);
        tok.sem_conv[i].ln_w = find(buf);
        snprintf(buf, sizeof(buf), "sem.conv.%d.ln.bias", i);
        tok.sem_conv[i].ln_b = find(buf);
    }
    tok.sem_featproj_ln_w = find("sem.feat_proj_ln.weight");
    tok.sem_featproj_ln_b = find("sem.feat_proj_ln.bias");
    tok.sem_featproj_w = find("sem.feat_proj.weight");
    tok.sem_featproj_b = find("sem.feat_proj.bias");
    tok.sem_posconv_g = find("sem.pos_conv.parametrizations.weight.original0");
    tok.sem_posconv_v = find("sem.pos_conv.parametrizations.weight.original1");
    tok.sem_posconv_b = find("sem.pos_conv.bias");
    tok.sem_enc_ln_w = find("sem.enc_ln.weight");
    tok.sem_enc_ln_b = find("sem.enc_ln.bias");
    tok.sem_blocks.resize(tok.n_sem_layers);
    for (int i = 0; i < tok.n_sem_layers; i++) {
        char buf[96];
        auto& b = tok.sem_blocks[i];
        auto g = [&](const char* suffix) {
            snprintf(buf, sizeof(buf), "sem.blk.%d.%s", i, suffix);
            return find(buf);
        };
        b.attn_q_w = g("attn_q.weight");
        b.attn_q_b = g("attn_q.bias");
        b.attn_k_w = g("attn_k.weight");
        b.attn_k_b = g("attn_k.bias");
        b.attn_v_w = g("attn_v.weight");
        b.attn_v_b = g("attn_v.bias");
        b.attn_out_w = g("attn_output.weight");
        b.attn_out_b = g("attn_output.bias");
        b.ln_w = g("ln.weight");
        b.ln_b = g("ln.bias");
        b.fc1_w = g("fc1.weight");
        b.fc1_b = g("fc1.bias");
        b.fc2_w = g("fc2.weight");
        b.fc2_b = g("fc2.bias");
        b.ffn_ln_w = g("ffn_ln.weight");
        b.ffn_ln_b = g("ffn_ln.bias");
    }
    // encoder_semantic bridge
    tok.encsem_conv_w = find("encoder_semantic.conv.weight");
    tok.encsem_blocks.resize(tok.n_encsem_blocks);
    for (int b = 0; b < tok.n_encsem_blocks; b++) {
        char buf[128];
        auto& blk = tok.encsem_blocks[b];
        for (int r = 0; r < 2; r++) {
            snprintf(buf, sizeof(buf), "encoder_semantic.conv_blocks.%d.res_units.%d.conv1.weight", b, r);
            blk.res[r].conv1_w = find(buf);
            snprintf(buf, sizeof(buf), "encoder_semantic.conv_blocks.%d.res_units.%d.conv2.weight", b, r);
            blk.res[r].conv2_w = find(buf);
        }
        snprintf(buf, sizeof(buf), "encoder_semantic.conv_blocks.%d.conv.weight", b);
        blk.conv_w = find(buf);
        snprintf(buf, sizeof(buf), "encoder_semantic.conv_blocks.%d.conv.bias", b);
        blk.conv_b = find(buf);
    }

    // Verify critical weights
    bool ok = true;
    if (!tok.fc2_w) {
        fprintf(stderr, "omnivoice: tokenizer missing fc2.weight\n");
        ok = false;
    }
    if (!tok.dec_conv1_w) {
        fprintf(stderr, "omnivoice: tokenizer missing acoustic_decoder.conv1.weight\n");
        ok = false;
    }
    for (int i = 0; i < tok.n_quantizers && ok; i++) {
        if (!tok.quantizers[i].codebook) {
            fprintf(stderr, "omnivoice: tokenizer missing quantizer.%d.codebook.embed\n", i);
            ok = false;
        }
    }
    if (!ok) {
        gguf_free(gf);
        return false;
    }

    // Create backend + buffer. Codec decode is normally CPU; OMNIVOICE_CODEC_GPU=1
    // (with a GPU main backend) puts the conv-heavy decode/encode graphs on the GPU
    // — experimental, default OFF (see codec_gpu_enabled). A fresh GPU backend keeps
    // the tokenizer independent of ctx->backend for clean teardown.
    const bool codec_gpu = ctx->use_gpu && codec_gpu_enabled();
    tok.backend = codec_gpu ? crispasr_init_gpu_backend() : nullptr;
    if (!tok.backend)
        tok.backend = ggml_backend_cpu_init();
    if (ggml_backend_is_cpu(tok.backend))
        ggml_backend_cpu_set_n_threads(tok.backend, ctx->n_threads);
    if (ctx->verbosity >= 1)
        fprintf(stderr, "omnivoice: codec backend = %s\n", ggml_backend_name(tok.backend));
    tok.buf_w = ggml_backend_alloc_ctx_tensors(tok.ctx_w, tok.backend);

    // Load tensor data
    {
        FILE* fp = fopen(path, "rb");
        if (!fp) {
            gguf_free(gf);
            return false;
        }
        int n_tensors = gguf_get_n_tensors(gf);
        for (int i = 0; i < n_tensors; i++) {
            const char* name = gguf_get_tensor_name(gf, i);
            ggml_tensor* t = ggml_get_tensor(tok.ctx_w, name);
            if (!t)
                continue;
            size_t offset = gguf_get_data_offset(gf) + gguf_get_tensor_offset(gf, i);
            fseek(fp, (long)offset, SEEK_SET);
            size_t nbytes = ggml_nbytes(t);
            std::vector<uint8_t> tmp(nbytes);
            if (fread(tmp.data(), 1, nbytes, fp) != nbytes) {
                fclose(fp);
                gguf_free(gf);
                return false;
            }
            ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
        }
        fclose(fp);
    }

    gguf_free(gf);
    tok.loaded = true;

    // FASTCONV (#254): pre-bake F32 decode conv kernels (shared core_dac helper) so
    // the per-graph F16→F32 cast disappears. env OMNIVOICE_CODEC_FASTCONV=0 skips it.
    {
        std::vector<ggml_tensor*> convs = {tok.dec_conv1_w, tok.dec_conv2_w};
        for (auto& blk : tok.dec_blocks) {
            convs.push_back(blk.conv_t1_w);
            for (int r = 0; r < 3; r++) {
                convs.push_back(blk.res[r].conv1_w);
                convs.push_back(blk.res[r].conv2_w);
            }
        }
        tok.fc.bake(tok.backend, convs, codec_fastconv_enabled());
    }

    if (ctx->verbosity >= 1) {
        size_t total = ggml_backend_buffer_get_size(tok.buf_w);
        fprintf(stderr, "omnivoice: tokenizer loaded %s (%.2f MB, %d quantizers)\n", path, total / 1e6,
                tok.n_quantizers);
    }
    return true;
}

// ---------------------------------------------------------------------------
// FASTCONV (#254): codec-decode conv hygiene.
// ---------------------------------------------------------------------------
//
// The DAC decoder runs entirely on CPU and dominates RTF on a fast box (the
// reporter's observation): 11.4 s to decode 11.7 s of audio, 6.8 s for a 2.5 s
// tail chunk. Per the dev-guide QWEN3_TTS_CODEC_FASTCONV learning, three wastes
// in the fork's conv path dwarf the actual conv FLOPs:
//   1. F16 kernel cast to F32 inside EVERY ggml_conv_1d/conv_transpose_1d graph
//      (activations are F32) — bake F32 kernels ONCE at load → cast becomes a no-op.
//   2. A k=1 conv is a channel matmul — routing it through im2col materializes a
//      pure copy at audio-rate T. Emit ggml_mul_mat on [Cin,T] directly.
//   3. (kept) the transpose/cont wrap around each conv is unavoidable with the
//      GGUF (K,Cin,Cout) layout; the two above are the real cost.
// Output is numerically equivalent to the F16 path (A/B'd: max |Δ| ≈ 20/32768,
// rmse ≈ 1.75 int16 ≈ −85 dB — inaudible reduction-order drift from the k=1
// matmul + the conv_transpose _f32 vs _f16_f32 path, the same F16-codec-level
// drift the parity methodology treats as normal; ASR roundtrip identical).
// ~2.9× faster decode on M1 CPU (10.6 s → 3.6 s for the reporter's paragraph).
// Gated for regression bisection, default ON. OMNIVOICE_CODEC_FASTCONV=0 = legacy.

static bool codec_fastconv_enabled() {
    const char* e = crispasr_env::get("CRISPASR_OMNIVOICE_CODEC_FASTCONV");
    return !(e && e[0] == '0');
}

// GPU codec decode (#254, experimental). The DAC decode is one big conv-heavy
// graph (not a per-step small-matmul loop), so unlike an AR decode it is a good
// GPU candidate — the dev-guide "codec 3× Metal" case. Every op in the decode
// graph is Metal-supported (CONV_TRANSPOSE_1D, IM2COL, MUL_MAT, SIN/SQR/DIV,
// CAST, GET_ROWS). Default OFF: a GPU default must be Kaggle-CUDA-verified per
// perf-discipline before flipping. Enable with OMNIVOICE_CODEC_GPU=1 (requires
// --use-gpu / GPU main backend). Pairs with FASTCONV (baked F32 kernels).
static bool codec_gpu_enabled() {
    const char* e = crispasr_env::get("CRISPASR_OMNIVOICE_CODEC_GPU");
    return e && e[0] && e[0] != '0';
}

// Bake F32 copies of every decode-path conv kernel into a dedicated buffer on
// the tokenizer backend. Only F16 kernels are baked (k-quant would need a
// different dequant; the shipped tokenizer is F16). Idempotent.
// ---------------------------------------------------------------------------
// HiggsAudioV2 decode: codes → PCM
// ---------------------------------------------------------------------------

static std::vector<float> higgs_decode(omnivoice_context* ctx, const int32_t* codes, int n_codebooks, int T_frames) {
    auto& tok = ctx->tokenizer;
    if (!tok.loaded)
        return {};

    // Graph context
    // Generous allocation — conv1d creates ~7 intermediate tensors each (transpose,
    // im2col, cast, reshape×3, mul_mat). 5 blocks × (snake+convt+3×resunit×2conv) =
    // ~40 convolutions × 7 = 280 tensor ops, plus RVQ/fc2/reshapes. Add 50% margin.
    size_t n_tensors = 700;
    // ggml_tensor_overhead() doesn't include GGML_MEM_ALIGN padding per object.
    // Use 2× overhead + generous flat padding to avoid off-by-alignment failures.
    size_t mem_size = n_tensors * 2 * ggml_tensor_overhead() + ggml_graph_overhead_custom(4096, false);
    std::vector<uint8_t> mem_buf(mem_size);
    ggml_init_params ip = {mem_size, mem_buf.data(), true};
    ggml_context* ctx0 = ggml_init(ip);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    // Create code input tensors
    std::vector<ggml_tensor*> code_inputs(n_codebooks);
    for (int k = 0; k < n_codebooks; k++) {
        code_inputs[k] = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_frames);
        char name[32];
        snprintf(name, sizeof(name), "codes_%d", k);
        ggml_set_name(code_inputs[k], name);
        ggml_set_input(code_inputs[k]);
    }

    // RVQ decode: for each quantizer, lookup + project_out + sum
    ggml_tensor* z_q = nullptr;
    for (int k = 0; k < n_codebooks; k++) {
        auto& q = tok.quantizers[k];
        // Codebook lookup: get_rows on (codebook_dim=64, codebook_size=1024)
        // returns (codebook_dim=64, T_frames) — ne[0]=64
        ggml_tensor* z = ggml_get_rows(ctx0, q.codebook, code_inputs[k]);
        z = ggml_cont(ctx0, ggml_cast(ctx0, z, GGML_TYPE_F32));
        // project_out: Linear(64 → 1024). proj_out_w ne[0]=64, z ne[0]=64 → match
        z = ggml_mul_mat(ctx0, tok.quantizers[k].proj_out_w, z);
        if (q.proj_out_b)
            z = ggml_add(ctx0, z, q.proj_out_b);
        // z: (hidden_size=1024, T_frames)
        z_q = k == 0 ? z : ggml_add(ctx0, z_q, z);
    }

    // FASTCONV: route decode convs through the shared core_dac fast path. The
    // cache (&tok.fc) selects baked-F32 kernels + k=1→matmul when enabled, and is
    // a no-op (identical to legacy conv1d) when OMNIVOICE_CODEC_FASTCONV=0.
    const core_dac::fastconv_cache* fc = &tok.fc;

    // fc2: Linear(hidden_size=1024 → dac_hidden=256)
    // fc2_w is (hidden_size, dac_hidden) in ggml convention
    ggml_tensor* h = ggml_mul_mat(ctx0, tok.fc2_w, z_q);
    if (tok.fc2_b)
        h = ggml_add(ctx0, h, tok.fc2_b);
    // h is now (dac_hidden=256, T_frames)

    // DAC decoder: conv1 → 5 blocks → snake → conv2
    // Input conv: Conv1d(256, 1024, k=7, p=3)
    if (env_bool("CRISPASR_OMNIVOICE_DEBUG")) {
        fprintf(stderr, "  decode: pre-fc2 z_q ne=[%ld,%ld]\n", z_q->ne[0], z_q->ne[1]);
        fprintf(stderr, "  decode: fc2_w ne=[%ld,%ld] fc2_b ne=[%ld]\n", tok.fc2_w->ne[0], tok.fc2_w->ne[1],
                tok.fc2_b ? tok.fc2_b->ne[0] : -1);
        fprintf(stderr, "  decode: h ne=[%ld,%ld] conv1_w ne=[%ld,%ld,%ld]\n", h->ne[0], h->ne[1],
                tok.dec_conv1_w->ne[0], tok.dec_conv1_w->ne[1], tok.dec_conv1_w->ne[2]);
    }
    h = core_dac::conv1d(ctx0, h, tok.dec_conv1_w, tok.dec_conv1_b, 7, 1, fc);

    // 5 decoder blocks with strides [8, 5, 4, 2, 3]
    static const int dilations[3] = {1, 3, 9};
    for (int b = 0; b < tok.n_dec_blocks; b++) {
        auto& blk = tok.dec_blocks[b];
        int stride = tok.upsampling_ratios[b];

        // Snake → ConvTranspose1d (baked-F32 kernel selects the direct _f32 CPU path)
        h = core_dac::snake(ctx0, h, blk.snake_alpha);
        int pad = (int)std::ceil((double)stride / 2.0);
        h = core_convt::convt1d_crop(ctx0, h, fc->get(blk.conv_t1_w), blk.conv_t1_b, stride, pad, pad);

        // 3 ResidualUnits (d=1, 3, 9)
        for (int r = 0; r < 3; r++) {
            auto& ru = blk.res[r];
            ggml_tensor* y = core_dac::snake(ctx0, h, ru.snake1_alpha);
            y = core_dac::conv1d(ctx0, y, ru.conv1_w, ru.conv1_b, 7, dilations[r], fc);
            y = core_dac::snake(ctx0, y, ru.snake2_alpha);
            y = core_dac::conv1d(ctx0, y, ru.conv2_w, ru.conv2_b, 1, 1, fc); // k=1 → matmul
            h = ggml_add(ctx0, h, y);
        }
        h = ggml_cont(ctx0, h);
    }

    // Final: Snake → Conv1d(32, 1, k=7) — no Tanh (HiggsAudio removes it)
    h = core_dac::snake(ctx0, h, tok.dec_snake_alpha);
    h = core_dac::conv1d(ctx0, h, tok.dec_conv2_w, tok.dec_conv2_b, 7, 1, fc);

    // Output: (1, T_pcm) → flatten
    int T_pcm = (int)h->ne[1];
    h = ggml_reshape_1d(ctx0, h, T_pcm);
    ggml_set_name(h, "pcm");
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);

    // Allocate and compute
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(tok.backend));
    ggml_gallocr_alloc_graph(ga, gf);

    // Set code inputs
    for (int k = 0; k < n_codebooks; k++) {
        ggml_backend_tensor_set(code_inputs[k], codes + (size_t)k * T_frames, 0, T_frames * sizeof(int32_t));
    }

    ggml_backend_graph_compute(tok.backend, gf);

    // Read PCM output
    ggml_tensor* pcm_out = ggml_graph_get_tensor(gf, "pcm");
    std::vector<float> pcm(T_pcm);
    ggml_backend_tensor_get(pcm_out, pcm.data(), 0, T_pcm * sizeof(float));

    ggml_gallocr_free(ga);
    ggml_free(ctx0);

    return pcm;
}

// ---------------------------------------------------------------------------
// BPE tokenizer
// ---------------------------------------------------------------------------

// Tokenize with special token handling. Scans for <|...|> patterns first,
// replaces with their IDs, then BPE-tokenizes the remaining text segments.
static std::vector<int32_t> tokenize(const ov_vocab& vocab, const std::string& text) {
    if (vocab.special_tokens.empty()) {
        return core_bpe::tokenize_simple(vocab.token_to_id, vocab.merge_rank, text);
    }
    std::vector<int32_t> result;
    size_t pos = 0;
    while (pos < text.size()) {
        // Look for <| at current position
        if (text[pos] == '<' && pos + 1 < text.size() && text[pos + 1] == '|') {
            // Find closing |>
            size_t end = text.find("|>", pos + 2);
            if (end != std::string::npos) {
                std::string token = text.substr(pos, end + 2 - pos);
                auto it = vocab.special_tokens.find(token);
                if (it != vocab.special_tokens.end()) {
                    result.push_back(it->second);
                    pos = end + 2;
                    continue;
                }
            }
        }
        // Find next special token or end
        size_t next = text.find("<|", pos + 1);
        if (next == std::string::npos)
            next = text.size();
        std::string segment = text.substr(pos, next - pos);
        if (!segment.empty()) {
            auto seg_tokens = core_bpe::tokenize_simple(vocab.token_to_id, vocab.merge_rank, segment);
            result.insert(result.end(), seg_tokens.begin(), seg_tokens.end());
        }
        pos = next;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Estimate target audio length (frames) from text length
// ---------------------------------------------------------------------------

// Per-codepoint "phonetic weight" (relative speech duration). Adapted from
// OmniVoice's rule-based estimator (omnivoice/utils/duration.py — Apache-2.0,
// © Xiaomi/k2-fsa; C++ mirror in ServeurpersoCom/omnivoice.cpp — MIT). Weights:
// mark 0, separator 0.2, punct/symbol 0.5, digit 3.5 (spoken as words),
// latin/cyrillic/greek 1.0, arabic/hebrew/thai 1.5, indic 1.8, kana 2.2,
// hangul 2.5, cjk 3.0.
static double duration_cp_weight(uint32_t cp) {
    if ((cp >= 0x30 && cp <= 0x39) || (cp >= 0xFF10 && cp <= 0xFF19))
        return 3.5; // digits — spoken as words
    if ((cp >= 0x41 && cp <= 0x5A) || (cp >= 0x61 && cp <= 0x7A))
        return 1.0; // ASCII letters
    if (cp < 0x80)
        return (cp == 0x20 || cp == 0x09 || cp == 0x0A || cp == 0x0D) ? 0.2 : 0.5;
    if ((cp >= 0x0300 && cp <= 0x036F) || (cp >= 0x1DC0 && cp <= 0x1DFF) || (cp >= 0x20D0 && cp <= 0x20FF))
        return 0.0; // combining marks — silent
    if (cp == 0x3000)
        return 0.2; // ideographic space
    if ((cp >= 0x3001 && cp <= 0x303F) || (cp >= 0xFF00 && cp <= 0xFF0F) || (cp >= 0xFF1A && cp <= 0xFF20) ||
        (cp >= 0xFF3B && cp <= 0xFF40) || (cp >= 0xFF5B && cp <= 0xFF65))
        return 0.5; // CJK punctuation
    if (cp >= 0x00C0 && cp <= 0x024F)
        return 1.0; // Latin extended
    if (cp >= 0x0370 && cp <= 0x03FF)
        return 1.0; // Greek
    if (cp >= 0x0400 && cp <= 0x04FF)
        return 1.0; // Cyrillic
    if (cp >= 0x0590 && cp <= 0x06FF)
        return 1.5; // Hebrew + Arabic
    if (cp >= 0x0E00 && cp <= 0x0EFF)
        return 1.5; // Thai + Lao
    if (cp >= 0x0900 && cp <= 0x0DFF)
        return 1.8; // Indic (Devanagari..Sinhala)
    if (cp >= 0x3040 && cp <= 0x30FF)
        return 2.2; // Hiragana + Katakana
    if ((cp >= 0xAC00 && cp <= 0xD7AF) || (cp >= 0x1100 && cp <= 0x11FF))
        return 2.5; // Hangul
    if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2FA1F))
        return 3.0; // CJK ideographs
    return 1.0;     // default letter-ish
}

static double duration_text_weight(const std::string& text) {
    double w = 0.0;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = text[i];
        uint32_t cp;
        int adv;
        if (c < 0x80) {
            cp = c;
            adv = 1;
        } else if (c < 0xE0) {
            cp = c & 0x1F;
            adv = 2;
        } else if (c < 0xF0) {
            cp = c & 0x0F;
            adv = 3;
        } else {
            cp = c & 0x07;
            adv = 4;
        }
        for (int k = 1; k < adv && i + (size_t)k < text.size(); k++)
            cp = (cp << 6) | ((unsigned char)text[i + k] & 0x3F);
        w += duration_cp_weight(cp);
        i += adv;
    }
    return w;
}

// Estimate target length (in ref_dur's unit) from a reference (text, dur) pair,
// mirroring RuleDurationEstimator.estimate_duration:
//   speed = weight(ref_text) / ref_dur;  est = weight(target) / speed
// with a power-curve BOOST when the linear estimate is short (< low_threshold):
//   est = low_threshold * (est/low_threshold)^(1/boost_strength)
// The boost lengthens short clips so they don't render too fast / skip
// characters (#254). Defaults (low_threshold=50, boost_strength=3) match Python.
static double duration_estimate(const std::string& target_text, const std::string& ref_text, double ref_dur,
                                double low_threshold = 50.0, double boost_strength = 3.0) {
    if (ref_dur <= 0.0 || ref_text.empty())
        return 0.0;
    double ref_weight = duration_text_weight(ref_text);
    if (ref_weight == 0.0)
        return 0.0;
    double speed = ref_weight / ref_dur;
    double est = duration_text_weight(target_text) / speed;
    if (est < low_threshold)
        return low_threshold * std::pow(est / low_threshold, 1.0 / boost_strength);
    return est;
}

// Target audio length (frames). With a voice prompt the anchor IS the reference
// (ref_text → ref_T frames) so the length tracks the reference speaker's actual
// rate (fixes "duration doesn't change with a reference voice", #254). Without
// one, fall back to the canonical "Nice to meet you." ≈ 25-frame anchor.
// OMNIVOICE_FRAMES_PER_CHAR overrides the no-ref frames/weighted-char rate.
static int estimate_target_tokens(const std::string& text, const std::string& ref_text = std::string(), int ref_T = 0,
                                  float speed = 1.0f) {
    std::string rt;
    double rd;
    if (ref_T > 0 && !ref_text.empty()) {
        rt = ref_text;
        rd = (double)ref_T;
    } else {
        rt = "Nice to meet you.";
        rd = 25.0;
        if (const char* e = crispasr_env::get("CRISPASR_OMNIVOICE_FRAMES_PER_CHAR")) {
            float v = (float)atof(e);
            if (v > 0.0f)
                rd = v * duration_text_weight(rt); // env = frames per weighted char
        }
    }
    double est = duration_estimate(text, rt, rd) / speed;
    return std::max((int)est, 10);
}

// ---------------------------------------------------------------------------
// Time steps for masked iterative schedule
// ---------------------------------------------------------------------------

static std::vector<float> get_time_steps(float t_start, float t_end, int num_step, float t_shift) {
    std::vector<float> steps(num_step + 1);
    for (int i = 0; i <= num_step; i++) {
        float t = t_start + (t_end - t_start) * i / num_step;
        // Apply shift: t' = t * shift / (1 + (shift - 1) * t)
        if (t_shift != 1.0f) {
            t = t * t_shift / (1.0f + (t_shift - 1.0f) * t);
        }
        steps[i] = t;
    }
    return steps;
}

// ---------------------------------------------------------------------------
// Build the Qwen3 LLM graph for a forward pass
// ---------------------------------------------------------------------------

static ggml_cgraph* build_llm_graph(omnivoice_context* ctx, ggml_context* ctx0, ggml_tensor* input_embeds, int T,
                                    int block_split = 0) {
    auto& hp = ctx->hp;
    auto& m = ctx->model;

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    ggml_tensor* cur = input_embeds; // (d_model, T)

    // Position IDs: [0, 1, 2, ..., T-1] (per-block for the unified CFG graph).
    ggml_tensor* pos_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);

    // Unified CFG (block_split>0): the sequence is [cond(0:split) | uncond(split:T)].
    // All matmuls/FFN/norms run over the full T (one dispatch), but ATTENTION is
    // split per block (independent flash-attns) — no wasted cross-block compute,
    // no padding. Beats both the 2-forward path (2× dispatch) and a block-diagonal
    // mask (which still computes T² attention).
    const bool split_attn = block_split > 0 && block_split < T;

    const float attn_scale = 1.0f / sqrtf((float)hp.head_dim);

    for (uint32_t il = 0; il < hp.n_layers; il++) {
        auto& b = m.blocks[il];

        // Pre-attention RMSNorm
        ggml_tensor* attn_in = ggml_rms_norm(ctx0, cur, hp.rms_norm_eps);
        attn_in = ggml_mul(ctx0, attn_in, b.attn_norm_w);

        // Q, K, V projections
        ggml_tensor* Q = ggml_mul_mat(ctx0, b.attn_q_w, attn_in);
        ggml_tensor* K = ggml_mul_mat(ctx0, b.attn_k_w, attn_in);
        ggml_tensor* V = ggml_mul_mat(ctx0, b.attn_v_w, attn_in);

        // Reshape to (head_dim, n_heads, T) / (head_dim, n_kv_heads, T)
        Q = ggml_reshape_3d(ctx0, Q, hp.head_dim, hp.n_heads, T);
        K = ggml_reshape_3d(ctx0, K, hp.head_dim, hp.n_kv_heads, T);
        V = ggml_reshape_3d(ctx0, V, hp.head_dim, hp.n_kv_heads, T);

        // Q/K norms (Qwen3 style)
        if (b.attn_q_norm_w) {
            Q = ggml_rms_norm(ctx0, Q, hp.rms_norm_eps);
            Q = ggml_mul(ctx0, Q, b.attn_q_norm_w);
        }
        if (b.attn_k_norm_w) {
            K = ggml_rms_norm(ctx0, K, hp.rms_norm_eps);
            K = ggml_mul(ctx0, K, b.attn_k_norm_w);
        }

        // RoPE (standard, not mRoPE — OmniVoice uses default rope_type)
        Q = ggml_rope_ext(ctx0, Q, pos_ids, nullptr, hp.head_dim, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f,
                          1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx0, K, pos_ids, nullptr, hp.head_dim, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f,
                          1.0f, 0.0f, 0.0f);

        // ggml_flash_attn_ext requires Q/K/V in (head_dim, T, n_heads) layout.
        // Permute from (head_dim, n_heads, T) → (head_dim, T, n_heads).
        // ggml handles GQA natively (Q.ne[2]=n_heads, K.ne[2]=n_kv_heads).
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        // flash_attn_ext: Q/K/V in (hd, T, n_heads/n_kv) → output (hd, n_heads, T).
        ggml_tensor* attn_out;
        if (split_attn) {
            // Independent flash-attn per block (cond [0:split], uncond [split:T]).
            const int Ta = block_split, Tb = T - block_split;
            auto blk = [&](ggml_tensor* X, int nh, int t0, int tn) {
                return ggml_cont(ctx0,
                                 ggml_view_3d(ctx0, X, hp.head_dim, tn, nh, X->nb[1], X->nb[2], (size_t)t0 * X->nb[1]));
            };
            ggml_tensor* Ac = ggml_flash_attn_ext(ctx0, blk(Q, hp.n_heads, 0, Ta), blk(K, hp.n_kv_heads, 0, Ta),
                                                  blk(V, hp.n_kv_heads, 0, Ta), nullptr, attn_scale, 0.0f, 0.0f);
            ggml_tensor* Au = ggml_flash_attn_ext(ctx0, blk(Q, hp.n_heads, Ta, Tb), blk(K, hp.n_kv_heads, Ta, Tb),
                                                  blk(V, hp.n_kv_heads, Ta, Tb), nullptr, attn_scale, 0.0f, 0.0f);
            attn_out = ggml_concat(ctx0, Ac, Au, 2); // concat along T (ne[2])
        } else {
            attn_out = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, attn_scale, 0.0f, 0.0f);
        }
        // Output shape: (head_dim, n_heads, T) → reshape to (n_heads*head_dim, T)
        attn_out = ggml_reshape_2d(ctx0, attn_out, hp.n_heads * hp.head_dim, T);

        // Output projection
        attn_out = ggml_mul_mat(ctx0, b.attn_output_w, attn_out);

        // Residual
        cur = ggml_add(ctx0, cur, attn_out);

        // FFN: pre-norm + SwiGLU
        ggml_tensor* ffn_in = ggml_rms_norm(ctx0, cur, hp.rms_norm_eps);
        ffn_in = ggml_mul(ctx0, ffn_in, b.ffn_norm_w);

        ggml_tensor* gate = ggml_mul_mat(ctx0, b.ffn_gate_w, ffn_in);
        ggml_tensor* up = ggml_mul_mat(ctx0, b.ffn_up_w, ffn_in);
        gate = ggml_silu(ctx0, gate);
        ggml_tensor* ffn_out = ggml_mul(ctx0, gate, up);
        ffn_out = ggml_mul_mat(ctx0, b.ffn_down_w, ffn_out);

        cur = ggml_add(ctx0, cur, ffn_out);
    }

    // Final RMSNorm
    cur = ggml_rms_norm(ctx0, cur, hp.rms_norm_eps);
    cur = ggml_mul(ctx0, cur, m.output_norm_w);

    // Audio heads projection: (d_model, T) → (n_codebooks * audio_vocab_size, T)
    ggml_tensor* logits = ggml_mul_mat(ctx0, m.audio_output_w, cur);
    ggml_set_name(logits, "audio_logits");
    ggml_set_output(logits);

    ggml_build_forward_expand(gf, logits);

    return gf;
}

// ---------------------------------------------------------------------------
// Prepare embeddings: text + audio tokens → mixed embedding
// ---------------------------------------------------------------------------
//
// Python: _prepare_embed_inputs()
//   text_embeds = llm.embed_tokens(input_ids[:, 0, :])
//   shifted_ids = (input_ids * audio_mask) + codebook_layer_offsets
//   audio_embeds = audio_embeddings(shifted_ids).sum(dim=1)
//   return where(audio_mask, audio_embeds, text_embeds)
//
// For the ggml graph we do this entirely in CPU since it's a one-time
// embedding lookup before the main transformer forward.

// Helper: read an embedding table (potentially F16) to CPU float32.
// Uses ggml_get_rows via a one-shot graph to dequantize properly.
static std::vector<float> read_embedding_rows(ggml_backend_t backend, ggml_tensor* embd_w, const int32_t* row_ids,
                                              int n_rows, int d) {
    size_t mem_size = (size_t)(n_rows + 4) * ggml_tensor_overhead() + ggml_graph_overhead();
    std::vector<uint8_t> mem_buf(mem_size);
    ggml_init_params ip = {mem_size, mem_buf.data(), true};
    ggml_context* ctx0 = ggml_init(ip);

    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rows);
    ggml_set_name(ids, "row_ids");
    ggml_set_input(ids);
    ggml_tensor* out = ggml_get_rows(ctx0, embd_w, ids);
    ggml_set_name(out, "embd_out");
    ggml_set_output(out);

    ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(ids, row_ids, 0, n_rows * sizeof(int32_t));
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> result(n_rows * d);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_gallocr_free(ga);
    ggml_free(ctx0);
    return result;
}

// Persistent embedding-lookup graph: avoids per-call ggml_context + gallocr
// creation in the hot MaskGIT loop. Shape is fixed (n_rows constant across
// steps), so the graph and buffer allocation are reused — only the row-ID
// input data changes per step.
struct ov_persist_embed {
    std::vector<uint8_t> mem_buf;
    ggml_context* ctx0 = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_tensor* ids = nullptr;
    ggml_tensor* out = nullptr;
    ggml_gallocr_t ga = nullptr;
    int n_rows = 0;
    int dim = 0;

    void init(ggml_backend_t backend, ggml_tensor* embd_w, int nr, int d) {
        n_rows = nr;
        dim = d;
        size_t mem = (size_t)(n_rows + 6) * ggml_tensor_overhead() + ggml_graph_overhead();
        mem_buf.resize(mem);
        ggml_init_params ip = {mem, mem_buf.data(), true};
        ctx0 = ggml_init(ip);
        ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rows);
        ggml_set_name(ids, "pe_row_ids");
        ggml_set_input(ids);
        out = ggml_get_rows(ctx0, embd_w, ids);
        ggml_set_name(out, "pe_embd_out");
        ggml_set_output(out);
        gf = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf, out);
        ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(ga, gf);
    }

    void compute(ggml_backend_t backend, const int32_t* row_data, float* result) {
        ggml_backend_tensor_set(ids, row_data, 0, (size_t)n_rows * sizeof(int32_t));
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_tensor_get(out, result, 0, (size_t)n_rows * dim * sizeof(float));
    }

    void release() {
        if (ga) {
            ggml_gallocr_free(ga);
            ga = nullptr;
        }
        if (ctx0) {
            ggml_free(ctx0);
            ctx0 = nullptr;
        }
    }
};

static std::vector<float> prepare_embeddings(
    omnivoice_context* ctx, const std::vector<int32_t>& text_ids,
    const std::vector<int32_t>& audio_tokens, // (n_codebooks * T_audio) row-major
    const std::vector<bool>& audio_mask,      // length = total_seq_len
    int total_T) {
    auto& hp = ctx->hp;
    auto& m = ctx->model;
    const int d = (int)hp.d_model;

    // Collect unique text token IDs for batch lookup
    std::vector<int32_t> text_row_ids;
    std::vector<int> text_positions; // which positions in embeds are text
    for (int t = 0; t < total_T; t++) {
        if (!audio_mask[t]) {
            text_row_ids.push_back(text_ids[t]);
            text_positions.push_back(t);
        }
    }

    // Collect audio embedding IDs (shifted by codebook offset)
    int n_audio_pos = 0;
    for (int t = 0; t < total_T; t++) {
        if (audio_mask[t])
            n_audio_pos++;
    }

    std::vector<int32_t> audio_row_ids;
    std::vector<int> audio_pos_map; // which output position
    {
        int apos = 0;
        for (int t = 0; t < total_T; t++) {
            if (!audio_mask[t])
                continue;
            for (uint32_t cb = 0; cb < hp.n_codebooks; cb++) {
                int code = audio_tokens[cb * n_audio_pos + apos];
                int shifted = code + (int)(cb * hp.audio_vocab_size);
                audio_row_ids.push_back(shifted);
                audio_pos_map.push_back(t);
            }
            apos++;
        }
    }

    std::vector<float> embeds(total_T * d, 0.0f);

    // Look up text embeddings
    if (!text_row_ids.empty()) {
        auto text_emb =
            read_embedding_rows(ctx->backend, m.token_embd_w, text_row_ids.data(), (int)text_row_ids.size(), d);
        for (size_t i = 0; i < text_positions.size(); i++) {
            std::memcpy(embeds.data() + (size_t)text_positions[i] * d, text_emb.data() + i * d, d * sizeof(float));
        }
    }

    // Look up audio embeddings and sum across codebooks
    if (!audio_row_ids.empty()) {
        auto audio_emb =
            read_embedding_rows(ctx->backend, m.audio_embd_w, audio_row_ids.data(), (int)audio_row_ids.size(), d);
        for (size_t i = 0; i < audio_row_ids.size(); i++) {
            int t = audio_pos_map[i];
            float* dst = embeds.data() + (size_t)t * d;
            const float* src = audio_emb.data() + i * d;
            for (int j = 0; j < d; j++) {
                dst[j] += src[j];
            }
        }
    }

    return embeds;
}

// ---------------------------------------------------------------------------
// Masked iterative generation loop
// ---------------------------------------------------------------------------

struct ov_gen_result {
    std::vector<int32_t> codes; // (n_codebooks * T) row-major
    int T = 0;
    int n_codebooks = 0;
};

static ov_gen_result generate_iterative(omnivoice_context* ctx, const std::string& text) {
    auto& hp = ctx->hp;
    auto& gen = ctx->gen;
    bool debug = env_bool("CRISPASR_OMNIVOICE_DEBUG");

    ov_gen_result result;
    result.n_codebooks = (int)hp.n_codebooks;

    // 1. Tokenize text. Voice cloning prepends the reference transcript into ONE
    //    combined text stream (Python _combine_text: ref_text + " " + target).
    auto trim = [](const std::string& s) {
        size_t a = s.find_first_not_of(" \t\r\n");
        if (a == std::string::npos)
            return std::string();
        size_t b = s.find_last_not_of(" \t\r\n");
        return s.substr(a, b - a + 1);
    };
    std::string combined_text = text;
    if (ctx->ref_T > 0 && !ctx->ref_text.empty())
        combined_text = trim(ctx->ref_text) + " " + trim(text);
    std::string wrapped = "<|text_start|>" + combined_text + "<|text_end|>";
    std::vector<int32_t> text_token_ids = tokenize(ctx->vocab, wrapped);
    int T_text = (int)text_token_ids.size();

    // 2. Estimate target audio length. When cloning, anchor the length to the
    //    reference (ref_text → ref_T frames) so it tracks the reference speaker's
    //    actual rate; otherwise use the built-in speaking-rate anchor.
    int T_target = estimate_target_tokens(text, ctx->ref_text, ctx->ref_T, ctx->speed);
    result.T = T_target;

    if (debug) {
        fprintf(stderr, "omnivoice: text tokens=%d, target audio frames=%d\n", T_text, T_target);
        fprintf(stderr, "omnivoice: token ids:");
        for (int i = 0; i < std::min(T_text, 20); i++)
            fprintf(stderr, " %d", text_token_ids[i]);
        if (T_text > 20)
            fprintf(stderr, " ...");
        fprintf(stderr, "\n");
    }

    // 3. Build style prefix. When cloning, a <|denoise|> token leads the style
    //    (Python create_voice_clone_prompt), then language + instruct.
    std::string style_text;
    if (ctx->ref_T > 0)
        style_text += "<|denoise|>";
    std::string lang_str = ctx->language.empty() ? "None" : ctx->language;
    std::string instruct_str = ctx->instruct.empty() ? "None" : ctx->instruct;
    style_text += "<|lang_start|>" + lang_str + "<|lang_end|>";
    style_text += "<|instruct_start|>" + instruct_str + "<|instruct_end|>";
    std::vector<int32_t> style_ids = tokenize(ctx->vocab, style_text);
    int T_style = (int)style_ids.size();

    // 4. Build the full input sequence
    // Layout: [style_tokens | text_tokens | ref_audio_tokens? | target_mask_tokens]
    int T_ref = ctx->ref_T;
    int T_total = T_style + T_text + T_ref + T_target;

    // Build text_ids (the first codebook layer for text positions)
    std::vector<int32_t> full_text_ids(T_total, 0);
    for (int i = 0; i < T_style; i++)
        full_text_ids[i] = style_ids[i];
    for (int i = 0; i < T_text; i++)
        full_text_ids[T_style + i] = text_token_ids[i];
    // ref audio and target positions get pad/mask (handled by audio path)

    // Audio mask: true for audio positions (ref + target)
    std::vector<bool> audio_mask(T_total, false);
    int audio_start = T_style + T_text;
    for (int i = audio_start; i < T_total; i++) {
        audio_mask[i] = true;
    }

    // Audio tokens: (n_codebooks, T_audio) where T_audio = T_ref + T_target
    int T_audio = T_ref + T_target;
    std::vector<int32_t> audio_tokens(hp.n_codebooks * T_audio, (int)hp.audio_mask_id);

    // Fill in reference codes if available
    if (T_ref > 0 && !ctx->ref_audio_codes.empty()) {
        for (uint32_t cb = 0; cb < hp.n_codebooks; cb++) {
            for (int t = 0; t < T_ref; t++) {
                audio_tokens[cb * T_audio + t] = ctx->ref_audio_codes[cb * T_ref + t];
            }
        }
    }

    // 5. Initialize result codes (all mask)
    std::vector<int32_t> tokens(hp.n_codebooks * T_target, (int)hp.audio_mask_id);

    // 6. Compute time steps and unmask schedule
    auto timesteps = get_time_steps(0.0f, 1.0f, gen.num_steps, gen.t_shift);
    int total_mask = T_target * (int)hp.n_codebooks;

    std::vector<int> schedule(gen.num_steps);
    int rem = total_mask;
    for (int step = 0; step < gen.num_steps; step++) {
        if (step == gen.num_steps - 1) {
            schedule[step] = rem;
        } else {
            int num = (int)std::ceil(total_mask * (timesteps[step + 1] - timesteps[step]));
            num = std::min(num, rem);
            schedule[step] = num;
            rem -= num;
        }
    }

    // 7. RNG for Gumbel sampling
    std::mt19937 rng(gen.seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    auto gumbel_noise = [&]() -> float {
        float u = uniform(rng);
        u = std::max(u, 1e-10f);
        return -std::log(-std::log(u));
    };

    // 8. Persistent forward graphs (#245 follow-up). The graph topology and
    // every tensor shape are identical across all masked-iteration steps —
    // T_total / T_target are fixed for the whole synthesis and there is no
    // KV cache (bidirectional attention, full recompute per step). Build and
    // gallocr-allocate each arm once, refresh only the input embeddings per
    // step; positions never change so they are set once at init.
    int out_dim = (int)(hp.n_codebooks * hp.audio_vocab_size);
    struct ov_step_graph {
        std::vector<uint8_t> mem_buf;
        ggml_context* ctx0 = nullptr;
        ggml_cgraph* gf = nullptr;
        ggml_tensor* inp = nullptr;
        ggml_tensor* logits = nullptr;
        ggml_gallocr_t ga = nullptr;
    };
    ov_step_graph fwd_c, fwd_u, fwd_uni;
    auto fwd_free = [](ov_step_graph& g) {
        if (g.ga)
            ggml_gallocr_free(g.ga);
        if (g.ctx0)
            ggml_free(g.ctx0);
        g.ga = nullptr;
        g.ctx0 = nullptr;
    };
    const bool persistent = [] {
        const char* e = crispasr_env::get("CRISPASR_OMNIVOICE_PERSISTENT_GRAPH");
        return !(e && e[0] == '0');
    }();
    // Unified CFG: fuse cond + uncond into ONE graph (seq-concat + per-block
    // attention split, per-block RoPE positions). Kaggle CUDA A/B verdict: on CUDA
    // it's ~13% faster (67 vs 77 ms/step, codes byte-identical) so it's the default
    // there; on Metal/CPU the forward is compute-bound and fusion is ~3% slower, so
    // 2-forward stays the default. Override with OMNIVOICE_UNIFIED_CFG=0/1.
    const bool unified_cfg = [&] {
        const char* e = crispasr_env::get("CRISPASR_OMNIVOICE_UNIFIED_CFG");
        if (e && e[0])
            return e[0] != '0'; // explicit override
        return ctx->backend && std::strstr(ggml_backend_name(ctx->backend), "CUDA") != nullptr;
    }();
    // Interval-CFG (#254, opt-in, APPROXIMATE): the uncond forward is ~44% of
    // stage0 but its output is smooth across masked-iterative steps. Recompute it
    // only every K steps and reuse the cached u_logits in between; the cond
    // forward stays fresh every step. K=1 = exact (default). K=2 ≈ −22% stage0.
    // This CHANGES output (guided uses a slightly stale uncond), so it's gated OFF
    // and validated by ASR + listening, never a silent default. Only applies to
    // the 2-forward path (unified fuses cond+uncond, so K>1 forces 2-forward).
    const int cfg_interval = [&] {
        const char* e = crispasr_env::get("CRISPASR_OMNIVOICE_CFG_INTERVAL");
        int k = e ? atoi(e) : 1;
        return k >= 1 ? k : 1;
    }();
    if (cfg_interval > 1 && debug)
        fprintf(stderr, "omnivoice: interval-CFG K=%d (uncond recomputed every %d steps)\n", cfg_interval,
                cfg_interval);
    // Fused step graph (#254): move the audio-embedding lookup + codebook sum +
    // text-embed concat INTO the forward graph (token ids in, target-slice
    // logits out). Kills the per-step ~18 MB embed readback, the CPU codebook
    // sum, the ~5 MB embed re-upload, and two extra graph dispatches. The
    // embedding math is the same get_rows + same-order F32 adds the legacy
    // path does on the host, so codes are byte-identical (A/B-gated).
    // OMNIVOICE_FUSED_STEP=0 restores the legacy split path. Default ON
    // everywhere: byte-identical codes proven on M1 Metal (5 config classes)
    // AND on CUDA by the #254 reporter (RTX 5070 Ti, cmp identical, gen
    // 3.55 s → 1.53 s, RTF 0.17 → 0.07, single CUDA-graph warmup).
    const bool fused_step = [&] {
        const char* e = crispasr_env::get("CRISPASR_OMNIVOICE_FUSED_STEP");
        if (e && e[0])
            return e[0] != '0';
        return persistent; // fused graphs are persistent-only
    }();
    auto run_llm_forward = [&](ov_step_graph& g, const std::vector<float>& emb, int T_in,
                               int target_offset) -> std::vector<float> {
        if (!g.ctx0) {
            size_t n_tensors = hp.n_layers * 40 + 100;
            size_t mem_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(8192, false);
            g.mem_buf.resize(mem_size);
            ggml_init_params ip2 = {mem_size, g.mem_buf.data(), true};
            g.ctx0 = ggml_init(ip2);
            g.inp = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, hp.d_model, T_in);
            ggml_set_name(g.inp, "input_embeds");
            ggml_set_input(g.inp);
            g.gf = build_llm_graph(ctx, g.ctx0, g.inp, T_in);
            g.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
            ggml_gallocr_alloc_graph(g.ga, g.gf);
            g.logits = ggml_graph_get_tensor(g.gf, "audio_logits");
        }
        // Refresh BOTH inputs every compute: gallocr may alias an
        // input-flagged tensor's slot with intermediates of the previous
        // compute, so a set-once position tensor reads back clobbered data
        // from the second compute on (bisected: step 0 bitwise-equal to the
        // per-call path, step 1+ diverged until pos_ids was re-set).
        std::vector<int32_t> pos_data(T_in);
        for (int i = 0; i < T_in; i++)
            pos_data[i] = i;
        ggml_tensor* pos_t = ggml_graph_get_tensor(g.gf, "pos_ids");
        if (pos_t)
            ggml_backend_tensor_set(pos_t, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));
        ggml_backend_tensor_set(g.inp, emb.data(), 0, emb.size() * sizeof(float));
        ggml_backend_graph_compute(ctx->backend, g.gf);
        std::vector<float> all_logits(out_dim * T_in);
        ggml_backend_tensor_get(g.logits, all_logits.data(), 0, all_logits.size() * sizeof(float));
        // Extract target portion
        std::vector<float> tgt(out_dim * T_target);
        for (int t = 0; t < T_target; t++) {
            std::memcpy(tgt.data() + (size_t)t * out_dim, all_logits.data() + (size_t)(target_offset + t) * out_dim,
                        out_dim * sizeof(float));
        }
        if (!persistent)
            fwd_free(g);
        if (crispasr_env::get("CRISPASR_OMNIVOICE_DEBUG_SUM")) {
            double s = 0;
            for (float v : tgt)
                s += v;
            fprintf(stderr, "omnivoice-dbg: T_in=%d off=%d sum=%.9e\n", T_in, target_offset, s);
        }
        return tgt;
    };

    // Unified cond+uncond forward: concat [cond(T_total) | uncond(T_target)] into
    // one sequence, per-block RoPE positions, block-diagonal mask, single forward;
    // splits out cond-target + uncond-target logits. Output-equivalent to two
    // separate full-attention forwards.
    auto run_unified = [&](const std::vector<float>& cond_emb, int T_tot, const std::vector<float>& uncond_emb,
                           int T_tgt, int cond_target_off, std::vector<float>& c_out, std::vector<float>& u_out) {
        int T_c = T_tot + T_tgt;
        ov_step_graph& g = fwd_uni;
        if (!g.ctx0) {
            // Attention-split adds ~15 tensors/layer (per-block views/conts/2 flash-
            // attns/concat), so budget more than the plain forward's 40/layer.
            size_t n_tensors = hp.n_layers * 64 + 100;
            size_t mem_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(8192, false);
            g.mem_buf.resize(mem_size);
            ggml_init_params ip2 = {mem_size, g.mem_buf.data(), true};
            g.ctx0 = ggml_init(ip2);
            g.inp = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, hp.d_model, T_c);
            ggml_set_name(g.inp, "input_embeds");
            ggml_set_input(g.inp);
            g.gf = build_llm_graph(ctx, g.ctx0, g.inp, T_c, /*block_split=*/T_tot);
            g.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
            ggml_gallocr_alloc_graph(g.ga, g.gf);
            g.logits = ggml_graph_get_tensor(g.gf, "audio_logits");
        }
        // Concat embeds (d_model, T_c).
        std::vector<float> cat((size_t)hp.d_model * T_c);
        std::memcpy(cat.data(), cond_emb.data(), (size_t)hp.d_model * T_tot * sizeof(float));
        std::memcpy(cat.data() + (size_t)hp.d_model * T_tot, uncond_emb.data(),
                    (size_t)hp.d_model * T_tgt * sizeof(float));
        // Per-block positions.
        std::vector<int32_t> pos(T_c);
        for (int i = 0; i < T_tot; i++)
            pos[i] = i;
        for (int i = 0; i < T_tgt; i++)
            pos[T_tot + i] = i;
        if (ggml_tensor* pt = ggml_graph_get_tensor(g.gf, "pos_ids"))
            ggml_backend_tensor_set(pt, pos.data(), 0, pos.size() * sizeof(int32_t));
        // Attention is split per block inside the graph (block_split=T_tot) — no
        // mask tensor needed.
        ggml_backend_tensor_set(g.inp, cat.data(), 0, cat.size() * sizeof(float));
        ggml_backend_graph_compute(ctx->backend, g.gf);
        std::vector<float> all_logits((size_t)out_dim * T_c);
        ggml_backend_tensor_get(g.logits, all_logits.data(), 0, all_logits.size() * sizeof(float));
        c_out.resize((size_t)out_dim * T_tgt);
        u_out.resize((size_t)out_dim * T_tgt);
        for (int t = 0; t < T_tgt; t++) {
            std::memcpy(c_out.data() + (size_t)t * out_dim, all_logits.data() + (size_t)(cond_target_off + t) * out_dim,
                        out_dim * sizeof(float));
            std::memcpy(u_out.data() + (size_t)t * out_dim, all_logits.data() + (size_t)(T_tot + t) * out_dim,
                        out_dim * sizeof(float));
        }
        if (!persistent)
            fwd_free(g);
    };

    // 8. Pre-computed/persistent embeddings for the hot loop.
    //
    // Text embeddings are constant across MaskGIT steps (only audio tokens
    // change as masks lift). Compute once and reuse. Audio embedding graphs
    // are shape-constant (n_codebooks × T_audio / T_target rows), so we
    // build one persistent gallocr per arm and reuse the buffer across steps
    // — eliminating the per-step ggml_context + gallocr alloc/free overhead
    // and the repeated text-embedding GPU→CPU round-trip. (#254)
    const int d = (int)hp.d_model;
    const int target_start = T_total - T_target;

    // 8a. Text embedding cache (positions 0..audio_start-1) — computed once.
    std::vector<float> text_emb_cache;
    if (audio_start > 0) {
        std::vector<int32_t> text_only_ids(full_text_ids.begin(), full_text_ids.begin() + audio_start);
        text_emb_cache =
            read_embedding_rows(ctx->backend, ctx->model.token_embd_w, text_only_ids.data(), audio_start, d);
    }

    // 8b. Persistent audio embedding graphs (reused across all 32 steps).
    const int n_cond_audio_rows = (int)hp.n_codebooks * T_audio;
    const int n_uncond_audio_rows = (int)hp.n_codebooks * T_target;
    ov_persist_embed audio_cond_pe, audio_uncond_pe;
    if (!fused_step) {
        audio_cond_pe.init(ctx->backend, ctx->model.audio_embd_w, n_cond_audio_rows, d);
        if (gen.guidance_scale != 0.0f)
            audio_uncond_pe.init(ctx->backend, ctx->model.audio_embd_w, n_uncond_audio_rows, d);
    }

    // 8b'. Fused step graphs (#254). One persistent graph per arm that takes
    // shifted audio-token IDs as its only per-step input and emits ONLY the
    // target-slice logits. The text embeddings live in a dedicated backend
    // buffer (outside any gallocr, so they can't be aliased) and are uploaded
    // once; the audio embedding lookup + per-position codebook sum run
    // in-graph (get_rows + chained adds in the same cb-ascending order as the
    // host sum — bitwise-identical F32).
    struct ov_fused_graph {
        std::vector<uint8_t> mem_buf;
        ggml_context* ctx0 = nullptr;
        ggml_cgraph* gf = nullptr;
        ggml_tensor* ids_c = nullptr; // cond audio ids (n_cb * T_audio) or null
        ggml_tensor* ids_u = nullptr; // uncond audio ids (n_cb * T_target) or null
        ggml_tensor* pos = nullptr;
        ggml_tensor* c_out = nullptr; // (out_dim, T_target) cond-target logits
        ggml_tensor* u_out = nullptr; // (out_dim, T_target) uncond logits
        void release() {
            if (ga)
                ggml_gallocr_free(ga);
            if (ctx0)
                ggml_free(ctx0);
            ga = nullptr;
            ctx0 = nullptr;
        }
        ggml_gallocr_t ga = nullptr;
    };
    ov_fused_graph fu_c, fu_u, fu_uni;
    ggml_context* ctx_ext = nullptr;
    ggml_backend_buffer_t buf_ext = nullptr;
    ggml_tensor* text_ext = nullptr;
    if (fused_step && audio_start > 0) {
        ggml_init_params ipe = {2 * ggml_tensor_overhead(), nullptr, true};
        ctx_ext = ggml_init(ipe);
        text_ext = ggml_new_tensor_2d(ctx_ext, GGML_TYPE_F32, d, audio_start);
        ggml_set_name(text_ext, "text_emb_ext");
        buf_ext = ggml_backend_alloc_ctx_tensors(ctx_ext, ctx->backend);
        ggml_backend_tensor_set(text_ext, text_emb_cache.data(), 0, (size_t)audio_start * d * sizeof(float));
    }

    // In-graph audio embedding: ids ordered [cb * T_a + t] (the audio_tokens
    // layout) → get_rows (d, n_cb*T_a) → reshape (d, T_a, n_cb) → sum the
    // n_cb slices with chained adds (cb ascending, matching the host loop).
    auto build_audio_sum = [&](ggml_context* c, ggml_tensor* ids, int T_a) -> ggml_tensor* {
        ggml_tensor* rows = ggml_get_rows(c, ctx->model.audio_embd_w, ids);
        ggml_tensor* r3 = ggml_reshape_3d(c, rows, d, T_a, hp.n_codebooks);
        ggml_tensor* acc = ggml_view_2d(c, r3, d, T_a, r3->nb[1], 0);
        for (uint32_t cb = 1; cb < hp.n_codebooks; cb++)
            acc = ggml_add(c, acc, ggml_view_2d(c, r3, d, T_a, r3->nb[1], (size_t)cb * r3->nb[2]));
        return acc;
    };

    // mode: 0 = cond-only (T_total), 1 = uncond-only (T_target),
    //       2 = unified cond+uncond (T_total + T_target, split attention)
    auto build_fused = [&](ov_fused_graph& g, int mode) {
        const size_t n_tensors = (size_t)hp.n_layers * 64 + 256;
        const size_t mem_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(8192, false);
        g.mem_buf.resize(mem_size);
        ggml_init_params ip2 = {mem_size, g.mem_buf.data(), true};
        g.ctx0 = ggml_init(ip2);

        ggml_tensor* emb_in = nullptr;
        int T_in = 0;
        int split = 0;
        if (mode == 0 || mode == 2) {
            g.ids_c = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, n_cond_audio_rows);
            ggml_set_name(g.ids_c, "fused_ids_c");
            ggml_set_input(g.ids_c);
            ggml_tensor* audio_c = build_audio_sum(g.ctx0, g.ids_c, T_audio);
            emb_in = (audio_start > 0) ? ggml_concat(g.ctx0, text_ext, audio_c, 1) : audio_c;
            T_in = T_total;
        }
        if (mode == 1 || mode == 2) {
            g.ids_u = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, n_uncond_audio_rows);
            ggml_set_name(g.ids_u, "fused_ids_u");
            ggml_set_input(g.ids_u);
            ggml_tensor* audio_u = build_audio_sum(g.ctx0, g.ids_u, T_target);
            if (mode == 2) {
                emb_in = ggml_concat(g.ctx0, emb_in, audio_u, 1);
                T_in = T_total + T_target;
                split = T_total;
            } else {
                emb_in = audio_u;
                T_in = T_target;
            }
        }

        g.gf = build_llm_graph(ctx, g.ctx0, emb_in, T_in, split);
        g.pos = ggml_graph_get_tensor(g.gf, "pos_ids");
        ggml_tensor* logits = ggml_graph_get_tensor(g.gf, "audio_logits");

        if (mode == 0 || mode == 2) {
            g.c_out = ggml_cont(g.ctx0, ggml_view_2d(g.ctx0, logits, out_dim, T_target, logits->nb[1],
                                                     (size_t)target_start * logits->nb[1]));
            ggml_set_name(g.c_out, "fused_c_tgt");
            ggml_set_output(g.c_out);
            ggml_build_forward_expand(g.gf, g.c_out);
        }
        if (mode == 1) {
            g.u_out = logits; // uncond-only graph IS the target slice already
        } else if (mode == 2) {
            g.u_out = ggml_cont(g.ctx0, ggml_view_2d(g.ctx0, logits, out_dim, T_target, logits->nb[1],
                                                     (size_t)T_total * logits->nb[1]));
            ggml_set_name(g.u_out, "fused_u_tgt");
            ggml_set_output(g.u_out);
            ggml_build_forward_expand(g.gf, g.u_out);
        }

        g.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        ggml_gallocr_alloc_graph(g.ga, g.gf);
    };

    // Per-step fused inputs (rebuilt on the host each step — a few hundred KB
    // of int32, trivially cheap) and persistent logit read-back buffers.
    std::vector<int32_t> fused_cond_ids(n_cond_audio_rows);
    std::vector<int32_t> fused_uncond_ids(n_uncond_audio_rows);
    std::vector<int32_t> fused_pos_buf;
    std::vector<float> c_buf, u_buf;
    if (fused_step) {
        c_buf.resize((size_t)out_dim * T_target);
        if (gen.guidance_scale != 0.0f)
            u_buf.resize((size_t)out_dim * T_target);
    }

    // Refresh EVERY gallocr-managed input each compute (aliasing gotcha —
    // see run_llm_forward). text_ext lives in its own buffer and is exempt.
    auto run_fused = [&](ov_fused_graph& g, int mode) {
        {
            ov_bench_stage b("  ids_upload");
            if (g.ids_c) {
                for (int t = 0; t < T_audio; t++)
                    for (uint32_t cb = 0; cb < hp.n_codebooks; cb++)
                        fused_cond_ids[(size_t)cb * T_audio + t] =
                            audio_tokens[cb * T_audio + t] + (int)(cb * hp.audio_vocab_size);
                ggml_backend_tensor_set(g.ids_c, fused_cond_ids.data(), 0, (size_t)n_cond_audio_rows * sizeof(int32_t));
            }
            if (g.ids_u) {
                for (int t = 0; t < T_target; t++)
                    for (uint32_t cb = 0; cb < hp.n_codebooks; cb++)
                        fused_uncond_ids[(size_t)cb * T_target + t] =
                            audio_tokens[cb * T_audio + T_ref + t] + (int)(cb * hp.audio_vocab_size);
                ggml_backend_tensor_set(g.ids_u, fused_uncond_ids.data(), 0,
                                        (size_t)n_uncond_audio_rows * sizeof(int32_t));
            }
            if (g.pos) {
                const int T_in = (mode == 2) ? T_total + T_target : (mode == 0 ? T_total : T_target);
                fused_pos_buf.resize(T_in);
                if (mode == 2) {
                    for (int i = 0; i < T_total; i++)
                        fused_pos_buf[i] = i;
                    for (int i = 0; i < T_target; i++)
                        fused_pos_buf[T_total + i] = i;
                } else {
                    for (int i = 0; i < T_in; i++)
                        fused_pos_buf[i] = i;
                }
                ggml_backend_tensor_set(g.pos, fused_pos_buf.data(), 0, (size_t)T_in * sizeof(int32_t));
            }
        }
        {
            ov_bench_stage b("  fwd_fused");
            ggml_backend_graph_compute(ctx->backend, g.gf);
        }
        {
            ov_bench_stage b("  read_logits");
            if (g.c_out)
                ggml_backend_tensor_get(g.c_out, c_buf.data(), 0, c_buf.size() * sizeof(float));
            if (g.u_out)
                ggml_backend_tensor_get(g.u_out, u_buf.data(), 0, u_buf.size() * sizeof(float));
        }
    };

    // 8c. Pre-allocated per-step buffers (avoid heap churn in the hot loop).
    std::vector<int32_t> cond_audio_ids(n_cond_audio_rows);
    std::vector<int32_t> uncond_audio_ids(n_uncond_audio_rows);
    std::vector<float> cond_audio_emb((size_t)n_cond_audio_rows * d);
    std::vector<float> uncond_audio_emb((size_t)n_uncond_audio_rows * d);
    std::vector<float> cond_embeds_buf((size_t)T_total * d);
    std::vector<float> uncond_embeds_buf((size_t)T_target * d);

    // 8d. Fast embedding helpers: text from cache, audio via persistent graph.
    auto prepare_cond_fast = [&]() {
        // Text portion: copy from cache (constant across steps)
        if (audio_start > 0)
            std::memcpy(cond_embeds_buf.data(), text_emb_cache.data(), (size_t)audio_start * d * sizeof(float));
        // Audio portion: zero then accumulate codebook embeddings
        std::memset(cond_embeds_buf.data() + (size_t)audio_start * d, 0, (size_t)T_audio * d * sizeof(float));
        // Build shifted row IDs for all codebooks × all audio positions
        for (int t = 0; t < T_audio; t++)
            for (uint32_t cb = 0; cb < hp.n_codebooks; cb++)
                cond_audio_ids[(size_t)t * hp.n_codebooks + cb] =
                    audio_tokens[cb * T_audio + t] + (int)(cb * hp.audio_vocab_size);
        // Single persistent GPU compute (no alloc/free)
        audio_cond_pe.compute(ctx->backend, cond_audio_ids.data(), cond_audio_emb.data());
        // Sum across codebooks per position
        for (int t = 0; t < T_audio; t++) {
            float* dst = cond_embeds_buf.data() + (size_t)(audio_start + t) * d;
            for (uint32_t cb = 0; cb < hp.n_codebooks; cb++) {
                const float* src = cond_audio_emb.data() + (size_t)((size_t)t * hp.n_codebooks + cb) * d;
                for (int j = 0; j < d; j++)
                    dst[j] += src[j];
            }
        }
    };

    auto prepare_uncond_fast = [&]() {
        std::memset(uncond_embeds_buf.data(), 0, (size_t)T_target * d * sizeof(float));
        for (int t = 0; t < T_target; t++)
            for (uint32_t cb = 0; cb < hp.n_codebooks; cb++)
                uncond_audio_ids[(size_t)t * hp.n_codebooks + cb] =
                    audio_tokens[cb * T_audio + T_ref + t] + (int)(cb * hp.audio_vocab_size);
        audio_uncond_pe.compute(ctx->backend, uncond_audio_ids.data(), uncond_audio_emb.data());
        for (int t = 0; t < T_target; t++) {
            float* dst = uncond_embeds_buf.data() + (size_t)t * d;
            for (uint32_t cb = 0; cb < hp.n_codebooks; cb++) {
                const float* src = uncond_audio_emb.data() + (size_t)((size_t)t * hp.n_codebooks + cb) * d;
                for (int j = 0; j < d; j++)
                    dst[j] += src[j];
            }
        }
    };

    // 9. Iterative generation loop
    std::vector<float> u_logits_cache; // interval-CFG (legacy path): last computed uncond logits
    bool fused_u_valid = false;        // interval-CFG (fused path): u_buf holds a computed uncond
    // Scoring buffers hoisted out of the hot loop (multi-MB, reused each step).
    std::vector<float> log_probs((size_t)out_dim * T_target);
    std::vector<int32_t> pred_tokens(hp.n_codebooks * T_target);
    std::vector<float> confidence(hp.n_codebooks * T_target);
    std::vector<int> indices(hp.n_codebooks * T_target);
    std::vector<float> pos_noise; // precomputed Gumbel noise (rng-order preserving)
    for (int step = 0; step < gen.num_steps; step++) {
        int k = schedule[step];
        if (k <= 0)
            continue;

        if (debug) {
            fprintf(stderr, "omnivoice: step %d/%d, unmask %d tokens\n", step + 1, gen.num_steps, k);
        }

        ov_bench_stage bench_step("gen_step");

        // Update audio tokens with current state
        for (uint32_t cb = 0; cb < hp.n_codebooks; cb++) {
            for (int t = 0; t < T_target; t++) {
                audio_tokens[cb * T_audio + T_ref + t] = tokens[cb * T_target + t];
            }
        }

        std::vector<float> c_logits, u_logits; // legacy-path storage
        const float* c_lg = nullptr;
        const float* u_lg = nullptr;

        if (fused_step) {
            if (unified_cfg && cfg_interval <= 1 && gen.guidance_scale != 0.0f) {
                if (!fu_uni.ctx0)
                    build_fused(fu_uni, 2);
                run_fused(fu_uni, 2);
                c_lg = c_buf.data();
                u_lg = u_buf.data();
            } else {
                if (!fu_c.ctx0)
                    build_fused(fu_c, 0);
                run_fused(fu_c, 0);
                c_lg = c_buf.data();
                if (gen.guidance_scale != 0.0f) {
                    const bool recompute = (step % cfg_interval == 0) || !fused_u_valid || (step == gen.num_steps - 1);
                    if (recompute) {
                        if (!fu_u.ctx0)
                            build_fused(fu_u, 1);
                        run_fused(fu_u, 1);
                        fused_u_valid = true;
                    }
                    u_lg = u_buf.data();
                }
            }
        } else if (unified_cfg && cfg_interval <= 1 && gen.guidance_scale != 0.0f) {
            // Fused single-graph CFG (exact only; interval-CFG needs the 2-forward path).
            {
                ov_bench_stage b("  embeds");
                prepare_cond_fast();
                prepare_uncond_fast();
            }
            ov_bench_stage b("  fwd_unified");
            run_unified(cond_embeds_buf, T_total, uncond_embeds_buf, T_target, target_start, c_logits, u_logits);
            c_lg = c_logits.data();
            u_lg = u_logits.data();
        } else {
            // Prepare conditional embeddings (full context: style + text + ref + target)
            {
                ov_bench_stage b("  embeds_cond");
                prepare_cond_fast();
            }
            {
                ov_bench_stage b("  fwd_cond");
                c_logits = run_llm_forward(fwd_c, cond_embeds_buf, T_total, target_start);
            }
            // Unconditional forward (target tokens only) for classifier-free guidance.
            // Interval-CFG: recompute only every cfg_interval steps (plus the first
            // and last for accuracy); reuse the cached u_logits otherwise. K=1 =
            // every step = exact.
            if (gen.guidance_scale != 0.0f) {
                const bool recompute =
                    (step % cfg_interval == 0) || u_logits_cache.empty() || (step == gen.num_steps - 1);
                if (recompute) {
                    {
                        ov_bench_stage b("  embeds_uncond");
                        prepare_uncond_fast();
                    }
                    ov_bench_stage b("  fwd_uncond");
                    u_logits_cache = run_llm_forward(fwd_u, uncond_embeds_buf, T_target, 0);
                }
                u_logits = u_logits_cache;
            }
            c_lg = c_logits.data();
            u_lg = u_logits.empty() ? nullptr : u_logits.data();
        }

        // _predict_tokens_with_scoring — mirrors Python exactly:
        //
        //   c_log = log_softmax(c_logits)        — mask_id IN denominator
        //   u_log = log_softmax(u_logits)        — mask_id IN denominator
        //   guided = c_log + scale*(c_log-u_log)
        //   log_probs = log_softmax(guided)      — mask_id IN denominator
        //   log_probs[mask_id] = -inf            — set AFTER all three softmaxes
        //   filtered = top_k(log_probs, ceil(0.1*V))
        //   pred_token = argmax(filtered/T + Gumbel)
        //   confidence = max(log_probs)          — argmax log-prob, NOT sampled token's

        // log_softmax over ALL V tokens (mask_id included in denominator, matching
        // Python's F.log_softmax before the post-softmax mask zeroing).
        auto log_softmax_all = [](const float* raw, float* out, int V) {
            float mx = raw[0];
            for (int v = 1; v < V; v++)
                mx = std::max(mx, raw[v]);
            float se = 0.0f;
            for (int v = 0; v < V; v++)
                se += std::exp(raw[v] - mx);
            float ls = mx + std::log(se);
            for (int v = 0; v < V; v++)
                out[v] = raw[v] - ls;
        };

        int V = (int)hp.audio_vocab_size;
        int mask_id = (int)hp.audio_mask_id;

        // Pass 1: compute per-codebook log-probs with CFG, store in log_probs.
        // Parallel over target positions — each (t, cb) cell is independent and
        // rng-free, so the arithmetic (and output) is identical to the serial
        // loop. This pass is ~13M exp() calls per step at T_target≈545.
        {
            ov_bench_stage b("  score_cfg");
            ov_parallel_for(ctx->n_threads, T_target, [&](int t0, int t1) {
                std::vector<float> c_lp(V), u_lp(V), guided(V), final_lp(V);
                for (int t = t0; t < t1; t++) {
                    for (uint32_t cb = 0; cb < hp.n_codebooks; cb++) {
                        size_t base = (size_t)t * out_dim + cb * V;
                        const float* c_raw = c_lg + base;

                        if (gen.guidance_scale != 0.0f && u_lg) {
                            const float* u_raw = u_lg + base;
                            log_softmax_all(c_raw, c_lp.data(), V);
                            log_softmax_all(u_raw, u_lp.data(), V);
                            for (int v = 0; v < V; v++)
                                guided[v] = c_lp[v] + gen.guidance_scale * (c_lp[v] - u_lp[v]);
                            log_softmax_all(guided.data(), final_lp.data(), V);
                        } else {
                            log_softmax_all(c_raw, final_lp.data(), V);
                        }
                        // Set mask_id to -inf AFTER all log_softmax passes
                        // (Python: log_probs[..., audio_mask_id] = -float("inf"))
                        final_lp[mask_id] = -1e30f;
                        std::memcpy(log_probs.data() + base, final_lp.data(), V * sizeof(float));
                    }
                }
            });
        }

        // Pass 2: top-k filter → Gumbel sample pred_tokens; compute confidence.
        ov_bench_stage bench_sample("  sample_select");
        if (gen.class_temperature <= 0.0f) {
            // Greedy path (default). Draw the position-Gumbel noise serially
            // FIRST — same rng stream, same (t-outer, cb-inner) order as the
            // serial loop — then consume it from the threaded loop, so the
            // output is bitwise-identical to the single-threaded original.
            if (gen.position_temperature > 0.0f) {
                pos_noise.resize((size_t)T_target * hp.n_codebooks);
                for (size_t i = 0; i < pos_noise.size(); i++)
                    pos_noise[i] = gumbel_noise();
            }
            ov_parallel_for(ctx->n_threads, T_target, [&](int t0, int t1) {
                for (int t = t0; t < t1; t++) {
                    for (uint32_t cb = 0; cb < hp.n_codebooks; cb++) {
                        const float* lp = log_probs.data() + (size_t)t * out_dim + cb * V;
                        // Confidence = max log-prob; greedy pick = argmax (first
                        // maximum, matching std::max_element tie-breaking).
                        int best_tok = 0;
                        float max_lp = -1e30f;
                        for (int v = 0; v < V; v++) {
                            if (lp[v] > max_lp) {
                                max_lp = lp[v];
                                best_tok = v;
                            }
                        }
                        int idx = (int)cb * T_target + t;
                        pred_tokens[idx] = best_tok;
                        // Confidence = max log-prob - layer penalty
                        confidence[idx] = max_lp - cb * gen.layer_penalty_factor;
                        // Position Gumbel noise
                        if (gen.position_temperature > 0.0f)
                            confidence[idx] =
                                confidence[idx] / gen.position_temperature + pos_noise[(size_t)t * hp.n_codebooks + cb];
                    }
                }
            });
        } else {
            // Sampled path (class_temperature > 0): draws a data-dependent
            // number of gumbels per cell, so the rng stream can't be
            // precomputed — stays serial.
            for (int t = 0; t < T_target; t++) {
                for (uint32_t cb = 0; cb < hp.n_codebooks; cb++) {
                    float* lp = log_probs.data() + (size_t)t * out_dim + cb * V;
                    // mask_id already -inf in lp

                    // Top-k filter: keep top ceil(0.1*V) tokens
                    // (Python: _filter_top_k(log_probs, ratio=0.1), k=ceil(0.1*1025)=103)
                    int top_k = std::max(1, (int)std::ceil(0.1f * V));
                    std::vector<float> vals(lp, lp + V);
                    std::nth_element(vals.begin(), vals.begin() + (V - top_k), vals.end());
                    float threshold = vals[V - top_k];
                    for (int v = 0; v < V; v++)
                        if (lp[v] < threshold)
                            lp[v] = -1e30f;

                    // Confidence = max log-prob after filter (Python: log_probs.max(dim=-1)[0]).
                    // The top-k always preserves the maximum, so max-after-filter = max-before-filter.
                    float max_lp = *std::max_element(lp, lp + V);

                    // Sample token: Gumbel.
                    int best_tok = 0;
                    float best_score = -1e30f;
                    for (int v = 0; v < V; v++) {
                        if (lp[v] < -1e20f)
                            continue;
                        float g = lp[v] / gen.class_temperature + gumbel_noise();
                        if (g > best_score) {
                            best_score = g;
                            best_tok = v;
                        }
                    }

                    int idx = (int)cb * T_target + t;
                    pred_tokens[idx] = best_tok;

                    // Confidence = max log-prob - layer penalty
                    // (Python: scores = confidence_scores - layer_ids * layer_penalty_factor)
                    confidence[idx] = max_lp - cb * gen.layer_penalty_factor;

                    // Position Gumbel noise
                    // (Python: scores = _gumbel_sample(scores, position_temperature))
                    if (gen.position_temperature > 0.0f) {
                        confidence[idx] = confidence[idx] / gen.position_temperature + gumbel_noise();
                    }
                }
            }
        }

        // Mask out already-unmasked positions (set confidence to -inf)
        for (int i = 0; i < (int)(hp.n_codebooks * T_target); i++) {
            if (tokens[i] != (int)hp.audio_mask_id) {
                confidence[i] = -1e30f;
            }
        }

        // Select top-k positions to unmask
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                          [&](int a, int b) { return confidence[a] > confidence[b]; });

        for (int i = 0; i < k; i++) {
            tokens[indices[i]] = pred_tokens[indices[i]];
        }
    }

    fwd_free(fwd_c);
    fwd_free(fwd_u);
    fwd_free(fwd_uni);
    fu_c.release();
    fu_u.release();
    fu_uni.release();
    if (buf_ext)
        ggml_backend_buffer_free(buf_ext);
    if (ctx_ext)
        ggml_free(ctx_ext);
    audio_cond_pe.release();
    audio_uncond_pe.release();

    // Byte-exact A/B hook: dump the raw generated codes to a file so two runs
    // (e.g. OMNIVOICE_FUSED_STEP=0 vs 1) can be diffed with cmp.
    if (const char* dump_path = env_str("CRISPASR_OMNIVOICE_DUMP_CODES")) {
        if (FILE* f = fopen(dump_path, "wb")) {
            fwrite(tokens.data(), sizeof(int32_t), tokens.size(), f);
            fclose(f);
            fprintf(stderr, "omnivoice: dumped %zu codes to %s\n", tokens.size(), dump_path);
        }
    }

    if (crispasr_env::get("CRISPASR_OMNIVOICE_DEBUG_CODES")) {
        int n_mask = 0;
        std::map<int32_t, int> hist;
        for (int32_t t : tokens) {
            if (t == (int32_t)hp.audio_mask_id)
                n_mask++;
            hist[t]++;
        }
        int top = 0;
        int32_t top_tok = -1;
        for (auto& kv : hist)
            if (kv.second > top) {
                top = kv.second;
                top_tok = kv.first;
            }
        fprintf(stderr, "omnivoice-codes: total=%zu mask=%d uniq=%zu top_tok=%d(x%d) cb0[0:24]=", tokens.size(), n_mask,
                hist.size(), top_tok, top);
        for (int t = 0; t < std::min(24, T_target); t++)
            fprintf(stderr, "%d ", tokens[t]);
        fprintf(stderr, "\n");
    }

    result.codes = std::move(tokens);
    return result;
}

// ===========================================================================
// Voice-cloning ENCODE path (WAV → codes). Built stage-by-stage and validated
// against the Python reference (omnivoice-encode-ref.gguf) via the diff harness.
// ===========================================================================

// Read a (possibly F16) weight tensor to a flat F32 row-major vector.
static std::vector<float> read_tensor_f32(ggml_tensor* t) {
    size_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), out.data(), (int64_t)n);
    } else {
        fprintf(stderr, "omnivoice: read_tensor_f32: unsupported type %d for %s\n", (int)t->type, t->name);
        out.assign(n, 0.0f);
    }
    return out;
}

// Factorized residual VQ encode: emb (T, hidden=1024 row-major) → codes (n_q, T).
// Per stage k: project_in (1024→64), Euclidean NN over codebook_k (1024×64),
// subtract project_out (64→1024) of the matched centroid from the residual.
// Mirrors HiggsAudioV2 quantizer.encode (project_in ≠ project_out, both learned).
// Dense y = W x + b for T frames. W row-major ne=[in,out] (flat [o*in + i]);
// x (T, in) row-major → y (T, out). b may be empty.
static void higgs_linear(const std::vector<float>& W, const std::vector<float>& b, const float* x, int T, int in,
                         int out, std::vector<float>& y) {
    y.assign((size_t)T * out, 0.0f);
    for (int t = 0; t < T; t++) {
        const float* xt = x + (size_t)t * in;
        float* yt = y.data() + (size_t)t * out;
        for (int o = 0; o < out; o++) {
            float acc = b.empty() ? 0.0f : b[o];
            const float* w = W.data() + (size_t)o * in;
            for (int i = 0; i < in; i++)
                acc += w[i] * xt[i];
            yt[o] = acc;
        }
    }
}

// Per-row cosine (min/mean) + max_abs between two (T, C) row-major buffers.
static void higgs_cos(const float* a, const float* b, int T, int C, float& cos_min, float& cos_mean, float& max_abs) {
    cos_min = 1.0f;
    double sum_cos = 0.0;
    max_abs = 0.0f;
    for (int t = 0; t < T; t++) {
        const float* x = a + (size_t)t * C;
        const float* y = b + (size_t)t * C;
        double dot = 0, nx = 0, ny = 0;
        for (int c = 0; c < C; c++) {
            dot += (double)x[c] * y[c];
            nx += (double)x[c] * x[c];
            ny += (double)y[c] * y[c];
            float d = std::fabs(x[c] - y[c]);
            if (d > max_abs)
                max_abs = d;
        }
        float cs = (nx > 0 && ny > 0) ? (float)(dot / (std::sqrt(nx) * std::sqrt(ny))) : 1.0f;
        cos_min = std::min(cos_min, cs);
        sum_cos += cs;
    }
    cos_mean = (float)(sum_cos / std::max(1, T));
}

static bool higgs_rvq_encode(ov_higgs_tokenizer& tok, const std::vector<float>& emb, int T,
                             std::vector<int32_t>& out_codes) {
    const int H = tok.hidden_size;   // 1024
    const int D = tok.codebook_dim;  // 64
    const int S = tok.codebook_size; // 1024
    const int NQ = tok.n_quantizers;
    for (int k = 0; k < NQ; k++) {
        auto& q = tok.quantizers[k];
        if (!q.proj_in_w || !q.proj_in_b || !q.codebook || !q.proj_out_w || !q.proj_out_b) {
            fprintf(stderr, "omnivoice: RVQ stage %d missing encode weights\n", k);
            return false;
        }
    }
    out_codes.assign((size_t)NQ * T, 0);
    std::vector<float> residual = emb; // (T, H)

    for (int k = 0; k < NQ; k++) {
        auto& q = tok.quantizers[k];
        std::vector<float> pin_w = read_tensor_f32(q.proj_in_w);   // (D, H) row-major: [o*H + i]
        std::vector<float> pin_b = read_tensor_f32(q.proj_in_b);   // (D,)
        std::vector<float> cb = read_tensor_f32(q.codebook);       // (S, D) row-major: [s*D + d]
        std::vector<float> pout_w = read_tensor_f32(q.proj_out_w); // (H, D) row-major: [o*D + d]
        std::vector<float> pout_b = read_tensor_f32(q.proj_out_b); // (H,)

        // Project all frames' residual → P (T, D).
        std::vector<float> P((size_t)T * D);
        for (int t = 0; t < T; t++) {
            const float* r = residual.data() + (size_t)t * H;
            float* p = P.data() + (size_t)t * D;
            for (int o = 0; o < D; o++) {
                float acc = pin_b[o];
                const float* w = pin_w.data() + (size_t)o * H;
                for (int i = 0; i < H; i++)
                    acc += w[i] * r[i];
                p[o] = acc;
            }
        }

        // Euclidean nearest-neighbor over the codebook (reuse core_rvq).
        std::vector<float> norm_sq(S);
        for (int s = 0; s < S; s++) {
            float ss = 0.0f;
            const float* e = cb.data() + (size_t)s * D;
            for (int d = 0; d < D; d++)
                ss += e[d] * e[d];
            norm_sq[s] = ss;
        }
        core_rvq::Codebook cbk{cb.data(), norm_sq.data(), S, D};
        std::vector<int32_t> idx(T);
        if (!core_rvq::encode_euclidean(P.data(), T, D, &cbk, 1, idx.data()))
            return false;

        // Store codes + subtract project_out(centroid) from the residual.
        for (int t = 0; t < T; t++) {
            int s = idx[t];
            out_codes[(size_t)k * T + t] = s;
            const float* e = cb.data() + (size_t)s * D;
            float* r = residual.data() + (size_t)t * H;
            for (int o = 0; o < H; o++) {
                float acc = pout_b[o];
                const float* w = pout_w.data() + (size_t)o * D;
                for (int d = 0; d < D; d++)
                    acc += w[d] * e[d];
                r[o] -= acc;
            }
        }
    }
    return true;
}

// Minimal reader for a reference GGUF: pull a named F32/I32 tensor to a flat
// float vector (int codes are returned as float for uniform compare/print).
struct ov_ref_gguf {
    gguf_context* gf = nullptr;
    ggml_context* ctx = nullptr;
    bool load(const char* path) {
        gguf_init_params gp = {/*.no_alloc=*/false, /*.ctx=*/&ctx};
        gf = gguf_init_from_file(path, gp);
        return gf != nullptr;
    }
    ~ov_ref_gguf() {
        if (gf)
            gguf_free(gf);
        if (ctx)
            ggml_free(ctx);
    }
    ggml_tensor* get(const char* name) const { return ctx ? ggml_get_tensor(ctx, name) : nullptr; }
};

// Strided Conv1d for the DAC encoder downsampling convs (core_dac::conv1d is
// stride-1 only). x (Cin, T), w (K, Cin, Cout) → (Cout, T_out); explicit pad.
static ggml_tensor* enc_conv1d_strided(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride,
                                       int pad) {
    const int Cout = (int)w->ne[2];
    ggml_tensor* y = ggml_cont(ctx, ggml_transpose(ctx, x)); // (T, Cin)
    y = ggml_conv_1d(ctx, w, y, stride, pad, 1);             // (T_out, Cout, 1)
    const int T_out = (int)y->ne[0];
    y = ggml_reshape_2d(ctx, y, T_out, Cout);
    y = ggml_cont(ctx, ggml_transpose(ctx, y)); // (Cout, T_out)
    if (b)
        y = ggml_add(ctx, y, b);
    return y;
}

// DAC acoustic encoder: wav (1, T_samp) @ 24 kHz → e_acoustic (256, T25).
// conv1(1,64,k7) → 5 blocks [3 ResidualUnit(d=1,3,9) then Snake + strided conv
// (k=2·s)] → Snake → conv2(2048,256,k3). Mirror of higgs_decode's DAC decoder.
struct ov_dbg_tensor {
    std::string name;
    int C = 0, T = 0;
    std::vector<float> data; // (T, C) row-major
};

static std::vector<float> higgs_acoustic_encode(omnivoice_context* ctx, const float* wav, int T_samp, int* out_T,
                                                std::vector<ov_dbg_tensor>* dbg = nullptr) {
    auto& tok = ctx->tokenizer;
    if (!tok.enc_conv1_w || tok.enc_blocks.size() != 5 || !tok.enc_conv2_w)
        return {};

    // The full DAC encoder graph is ~2–3k tensors (each conv1d expands to ~10);
    // an undersized context silently returns NULL for the LAST block's tensors →
    // partially-garbage output (bisected: blocks 0–3 exact, block4 corrupt).
    size_t mem_size = 8192 * 2 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16384, false);
    std::vector<uint8_t> mem_buf(mem_size);
    ggml_init_params ip = {mem_size, mem_buf.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);
    std::vector<std::pair<std::string, ggml_tensor*>> tagged;
    auto tag = [&](ggml_tensor* t, const char* name) {
        if (dbg) {
            ggml_set_name(t, name);
            ggml_set_output(t);
            tagged.emplace_back(name, t);
        }
    };

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_samp, 1); // (T, 1ch)
    ggml_set_name(x, "wav");
    ggml_set_input(x);
    // conv1: (1, T) → (64, T). core_dac::conv1d wants (Cin, T).
    ggml_tensor* h = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // (1, T)
    h = core_dac::conv1d(ctx0, h, tok.enc_conv1_w, tok.enc_conv1_b, 7);
    tag(h, "ac_conv1");

    static const int dil[3] = {1, 3, 9};
    for (int bl = 0; bl < 5; bl++) {
        auto& blk = tok.enc_blocks[bl];
        for (int r = 0; r < 3; r++) {
            auto& ru = blk.res[r];
            ggml_tensor* y = core_dac::snake(ctx0, h, ru.snake1_alpha);
            y = core_dac::conv1d(ctx0, y, ru.conv1_w, ru.conv1_b, 7, dil[r]);
            y = core_dac::snake(ctx0, y, ru.snake2_alpha);
            y = core_dac::conv1d(ctx0, y, ru.conv2_w, ru.conv2_b, 1);
            h = ggml_add(ctx0, h, y);
        }
        h = core_dac::snake(ctx0, h, blk.snake1_alpha);
        char snm[20];
        snprintf(snm, sizeof(snm), "ac_b%d_snake", bl);
        tag(h, snm);
        int stride = tok.downsampling_ratios[bl];
        int pad = (stride + 1) / 2; // ceil(stride/2); k = 2*stride
        h = enc_conv1d_strided(ctx0, h, blk.conv1_w, blk.conv1_b, stride, pad);
        h = ggml_cont(ctx0, h);
        char nm[16];
        snprintf(nm, sizeof(nm), "ac_block%d", bl);
        tag(h, nm);
    }
    h = core_dac::snake(ctx0, h, tok.enc_snake1_alpha);
    tag(h, "ac_snake1");
    h = core_dac::conv1d(ctx0, h, tok.enc_conv2_w, tok.enc_conv2_b, 3); // (256, T25)
    ggml_set_name(h, "e_acoustic");
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(tok.backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(x, wav, 0, (size_t)T_samp * sizeof(float));
    ggml_backend_graph_compute(tok.backend, gf);

    // Snapshot debug intermediates into caller-owned buffers before ctx0 is freed.
    // ggml (C,T) column-major == (T,C) row-major, so a straight copy suffices.
    if (dbg) {
        for (auto& pr : tagged) {
            ggml_tensor* t = pr.second;
            ov_dbg_tensor d;
            d.name = pr.first;
            d.C = (int)t->ne[0];
            d.T = (int)t->ne[1];
            d.data.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, d.data.data(), 0, d.data.size() * sizeof(float));
            dbg->push_back(std::move(d));
        }
    }

    int C = (int)h->ne[0], T = (int)h->ne[1];
    std::vector<float> raw((size_t)C * T);
    ggml_backend_tensor_get(h, raw.data(), 0, raw.size() * sizeof(float));
    // (C, T) column-major → (T, C) row-major for the diff.
    std::vector<float> out((size_t)T * C);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            out[(size_t)t * C + c] = raw[(size_t)t * C + c];
    *out_T = T;
    ggml_gallocr_free(ga);
    ggml_free(ctx0);
    return out;
}

// encoder_semantic bridge: sem (768, T) → e_semantic (768, T). conv(k3) then
// 2 blocks of {2 ELU ResidualUnits [ELU→conv k3→ELU→conv k1] + block conv k3}.
// All res-unit convs are bias-free; block conv has bias. No temporal change.
static std::vector<float> higgs_encoder_semantic(omnivoice_context* ctx, const float* sem, int T) {
    auto& tok = ctx->tokenizer;
    if (!tok.encsem_conv_w || tok.encsem_blocks.size() != 2)
        return {};
    const int D = tok.sem_d; // 768
    size_t mem_size = 2048 * 2 * ggml_tensor_overhead() + ggml_graph_overhead_custom(4096, false);
    std::vector<uint8_t> mem_buf(mem_size);
    ggml_init_params ip = {mem_size, mem_buf.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T); // (768, T)
    ggml_set_name(x, "sem");
    ggml_set_input(x);
    ggml_tensor* h = core_dac::conv1d(ctx0, x, tok.encsem_conv_w, nullptr, 3);
    for (int b = 0; b < 2; b++) {
        auto& blk = tok.encsem_blocks[b];
        for (int r = 0; r < 2; r++) {
            ggml_tensor* y = ggml_elu(ctx0, h);
            y = core_dac::conv1d(ctx0, y, blk.res[r].conv1_w, nullptr, 3);
            y = ggml_elu(ctx0, y);
            y = core_dac::conv1d(ctx0, y, blk.res[r].conv2_w, nullptr, 1);
            h = ggml_add(ctx0, h, y);
        }
        h = core_dac::conv1d(ctx0, h, blk.conv_w, blk.conv_b, 3);
    }
    ggml_set_name(h, "e_semantic");
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(tok.backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(x, sem, 0, (size_t)D * T * sizeof(float));
    ggml_backend_graph_compute(tok.backend, gf);

    int C = (int)h->ne[0], To = (int)h->ne[1];
    std::vector<float> out((size_t)To * C);
    ggml_backend_tensor_get(h, out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(ctx0);
    return out;
}

// Per-channel GroupNorm(num_groups=C) = InstanceNorm over time. h (C,T) → (C,T).
static ggml_tensor* hubert_groupnorm(ggml_context* ctx, ggml_tensor* h, ggml_tensor* w, ggml_tensor* b, float eps) {
    int C = (int)h->ne[0];
    ggml_tensor* hT = ggml_cont(ctx, ggml_transpose(ctx, h)); // (T, C)
    hT = ggml_norm(ctx, hT, eps);                             // normalize over ne0=T per channel
    hT = ggml_mul(ctx, hT, ggml_reshape_2d(ctx, w, 1, C));
    hT = ggml_add(ctx, hT, ggml_reshape_2d(ctx, b, 1, C));
    return ggml_cont(ctx, ggml_transpose(ctx, hT)); // (C, T)
}

// HuBERT frontend: wav16k (1, T_samp) → feat_proj (768, T_feat). Feature
// extractor = 7 VALID (no-pad) strided convs, GroupNorm on layer 0, GELU(erf);
// feat_projection = LayerNorm(512) + Linear(512→768). Captures hb_featextract.
static std::vector<float> higgs_hubert_frontend(omnivoice_context* ctx, const float* wav, int T_samp, int* out_T,
                                                std::vector<ov_dbg_tensor>* dbg = nullptr) {
    auto& tok = ctx->tokenizer;
    if (tok.sem_conv.size() != 7 || !tok.sem_featproj_w)
        return {};
    size_t mem_size = 4096 * 2 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8192, false);
    std::vector<uint8_t> mem_buf(mem_size);
    ggml_init_params ip = {mem_size, mem_buf.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_samp, 1);
    ggml_set_name(x, "wav16k");
    ggml_set_input(x);
    ggml_tensor* h = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // (1, T)
    for (int i = 0; i < 7; i++) {
        // VALID conv (pad=0), stride sem_conv_s[i].
        h = enc_conv1d_strided(ctx0, h, tok.sem_conv[i].conv_w, nullptr, tok.sem_conv_s[i], 0);
        if (i == 0 && tok.sem_conv[0].ln_w)
            h = hubert_groupnorm(ctx0, h, tok.sem_conv[0].ln_w, tok.sem_conv[0].ln_b, 1e-5f);
        h = ggml_gelu_erf(ctx0, h);
        h = ggml_cont(ctx0, h);
    }
    // h: (512, T_feat).
    ggml_tensor* feat = h;
    ggml_set_name(feat, "hb_featextract");
    ggml_set_output(feat);
    // feat_projection: LayerNorm(512) over channels per frame, then Linear(512→768).
    int Cf = (int)h->ne[0];
    ggml_tensor* p = ggml_norm(ctx0, h, tok.sem_ln_eps); // over ne0=512 per frame
    p = ggml_mul(ctx0, p, ggml_reshape_2d(ctx0, tok.sem_featproj_ln_w, Cf, 1));
    p = ggml_add(ctx0, p, ggml_reshape_2d(ctx0, tok.sem_featproj_ln_b, Cf, 1));
    p = ggml_mul_mat(ctx0, tok.sem_featproj_w, p); // (768, T)
    if (tok.sem_featproj_b)
        p = ggml_add(ctx0, p, tok.sem_featproj_b);
    ggml_set_name(p, "hb_featproj");
    ggml_set_output(p);
    ggml_build_forward_expand(gf, p);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(tok.backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(x, wav, 0, (size_t)T_samp * sizeof(float));
    ggml_backend_graph_compute(tok.backend, gf);

    if (dbg) {
        for (ggml_tensor* t : {feat}) {
            ov_dbg_tensor d;
            d.name = ggml_get_name(t);
            d.C = (int)t->ne[0];
            d.T = (int)t->ne[1];
            d.data.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, d.data.data(), 0, d.data.size() * sizeof(float));
            dbg->push_back(std::move(d));
        }
    }

    int C = (int)p->ne[0], T = (int)p->ne[1];
    std::vector<float> out((size_t)T * C);
    ggml_backend_tensor_get(p, out.data(), 0, out.size() * sizeof(float));
    *out_T = T;
    ggml_gallocr_free(ga);
    ggml_free(ctx0);
    return out;
}

// Fold pos_conv weight-norm (dim=2): w[o,i,k] = g[k]·v[o,i,k]/‖v[:,:,k]‖.
// GGUF layout: v ne=[K=128, in_g=48, out=768], g ne=[K,1,1]. Returns flat
// [o*in_g*K + i*K + k] = ggml conv-weight layout ne=[K, in_g, out].
static std::vector<float> fold_posconv_weight(ov_higgs_tokenizer& tok) {
    std::vector<float> v = read_tensor_f32(tok.sem_posconv_v); // [out*in_g*K]
    std::vector<float> g = read_tensor_f32(tok.sem_posconv_g); // [K]
    int K = (int)tok.sem_posconv_v->ne[0];
    int in_g = (int)tok.sem_posconv_v->ne[1];
    int out = (int)tok.sem_posconv_v->ne[2];
    std::vector<double> nrm(K, 0.0);
    for (int o = 0; o < out; o++)
        for (int i = 0; i < in_g; i++)
            for (int k = 0; k < K; k++) {
                float x = v[(size_t)o * in_g * K + (size_t)i * K + k];
                nrm[k] += (double)x * x;
            }
    for (int k = 0; k < K; k++)
        nrm[k] = std::sqrt(nrm[k]);
    std::vector<float> w(v.size());
    for (int o = 0; o < out; o++)
        for (int i = 0; i < in_g; i++)
            for (int k = 0; k < K; k++) {
                size_t idx = (size_t)o * in_g * K + (size_t)i * K + k;
                w[idx] = (float)(g[k] * v[idx] / (nrm[k] > 0 ? nrm[k] : 1.0));
            }
    return w;
}

// LayerNorm over ne0 (channels), affine (w,b length = ne0). x (C, T) → (C, T).
static ggml_tensor* hubert_layernorm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, float eps) {
    int C = (int)x->ne[0];
    ggml_tensor* y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, ggml_reshape_2d(ctx, w, C, 1));
    y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, C, 1));
    return y;
}

// HuBERT transformer encoder: feat_proj (768, T) → mean of 13 hidden states
// (768, T). pos_conv (weight-normed grouped conv, SamePad, GELU) + residual;
// encoder LayerNorm; 12 post-norm layers (12-head MHA + FFN); mean of the
// encoder-input + 12 layer outputs. Captures hb_layer0/hb_layer11.
static std::vector<float> higgs_hubert_encoder(omnivoice_context* ctx, const float* featproj, int T,
                                               std::vector<ov_dbg_tensor>* dbg = nullptr) {
    auto& tok = ctx->tokenizer;
    const int D = tok.sem_d;     // 768
    const int H = tok.sem_heads; // 12
    const int hd = D / H;        // 64
    const int groups = 16;
    const int in_g = D / groups; // 48
    const int K = 128;
    std::vector<float> pcw = fold_posconv_weight(tok);

    size_t mem_size = 8192 * 2 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16384, false);
    std::vector<uint8_t> mem_buf(mem_size);
    ggml_init_params ip = {mem_size, mem_buf.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T); // (768, T)
    ggml_set_name(x, "featproj");
    ggml_set_input(x);
    ggml_tensor* pc_w = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, K, in_g, D); // grouped conv weight
    ggml_set_name(pc_w, "pc_w");
    ggml_set_input(pc_w);

    // pos_conv: grouped conv (16 groups), pad=64, stride=1, then SamePad (drop
    // last col, even kernel) + GELU; residual add.
    std::vector<ggml_tensor*> gouts(groups);
    for (int grp = 0; grp < groups; grp++) {
        ggml_tensor* xin = ggml_view_2d(ctx0, x, in_g, T, x->nb[1], (size_t)grp * in_g * x->nb[0]);
        xin = ggml_cont(ctx0, xin); // (48, T)
        ggml_tensor* wg =
            ggml_view_3d(ctx0, pc_w, K, in_g, in_g, pc_w->nb[1], pc_w->nb[2], (size_t)grp * in_g * pc_w->nb[2]);
        wg = ggml_cont(ctx0, wg);                                       // (K, 48, 48)
        gouts[grp] = enc_conv1d_strided(ctx0, xin, wg, nullptr, 1, 64); // (48, T+1)
    }
    ggml_tensor* pc = gouts[0];
    for (int grp = 1; grp < groups; grp++)
        pc = ggml_concat(ctx0, pc, gouts[grp], 0); // (768, T+1)
    // + bias, SamePad (drop last column), GELU.
    pc = ggml_add(ctx0, pc, ggml_reshape_2d(ctx0, tok.sem_posconv_b, D, 1));
    pc = ggml_cont(ctx0, ggml_view_2d(ctx0, pc, D, T, pc->nb[1], 0)); // (768, T) drop last
    pc = ggml_gelu_erf(ctx0, pc);
    ggml_tensor* h = ggml_add(ctx0, x, pc);
    // encoder LayerNorm.
    h = hubert_layernorm(ctx0, h, tok.sem_enc_ln_w, tok.sem_enc_ln_b, tok.sem_ln_eps);

    const float scale = 1.0f / sqrtf((float)hd);
    std::vector<ggml_tensor*> hidden; // 13 states: input + 12 layer outputs
    hidden.push_back(h);
    for (int l = 0; l < tok.n_sem_layers; l++) {
        auto& b = tok.sem_blocks[l];
        // Self-attention (bidirectional, no mask).
        ggml_tensor* q = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_q_w, h), b.attn_q_b);
        ggml_tensor* k = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_k_w, h), b.attn_k_b);
        ggml_tensor* v = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_v_w, h), b.attn_v_b);
        q = ggml_scale(ctx0, q, scale);
        q = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, q, hd, H, T), 0, 2, 1, 3)); // (hd,T,H)
        k = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, k, hd, H, T), 0, 2, 1, 3));
        v = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, v, hd, H, T), 0, 2, 1, 3));
        ggml_tensor* scores = ggml_mul_mat(ctx0, k, q); // (T, T, H)
        scores = ggml_soft_max(ctx0, scores);
        ggml_tensor* vt = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 0, 2, 3)); // (T, hd, H)
        ggml_tensor* attn = ggml_mul_mat(ctx0, vt, scores);                   // (hd, T, H)
        attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));         // (hd, H, T)
        attn = ggml_reshape_2d(ctx0, attn, D, T);
        attn = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_out_w, attn), b.attn_out_b);
        h = ggml_add(ctx0, h, attn);
        h = hubert_layernorm(ctx0, h, b.ln_w, b.ln_b, tok.sem_ln_eps);
        // FFN.
        ggml_tensor* f = ggml_add(ctx0, ggml_mul_mat(ctx0, b.fc1_w, h), b.fc1_b);
        f = ggml_gelu_erf(ctx0, f);
        f = ggml_add(ctx0, ggml_mul_mat(ctx0, b.fc2_w, f), b.fc2_b);
        h = ggml_add(ctx0, h, f);
        h = hubert_layernorm(ctx0, h, b.ffn_ln_w, b.ffn_ln_b, tok.sem_ln_eps);
        hidden.push_back(h);
        if (dbg && (l == 0 || l == 11)) {
            char nm[16];
            snprintf(nm, sizeof(nm), "hb_layer%d", l);
            ggml_set_name(h, nm);
            ggml_set_output(h);
        }
    }
    // Mean of the 13 hidden states.
    ggml_tensor* mean = hidden[0];
    for (size_t i = 1; i < hidden.size(); i++)
        mean = ggml_add(ctx0, mean, hidden[i]);
    mean = ggml_scale(ctx0, mean, 1.0f / (float)hidden.size());
    ggml_set_name(mean, "hb_mean13");
    ggml_set_output(mean);
    ggml_build_forward_expand(gf, mean);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(tok.backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(x, featproj, 0, (size_t)D * T * sizeof(float));
    ggml_backend_tensor_set(pc_w, pcw.data(), 0, pcw.size() * sizeof(float));
    ggml_backend_graph_compute(tok.backend, gf);

    if (dbg) {
        for (const char* nm : {"hb_layer0", "hb_layer11"}) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
            if (!t)
                continue;
            ov_dbg_tensor d;
            d.name = nm;
            d.C = (int)t->ne[0];
            d.T = (int)t->ne[1];
            d.data.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, d.data.data(), 0, d.data.size() * sizeof(float));
            dbg->push_back(std::move(d));
        }
    }
    int C = (int)mean->ne[0], To = (int)mean->ne[1];
    std::vector<float> out((size_t)To * C);
    ggml_backend_tensor_get(mean, out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(ctx0);
    return out;
}

// Full voice-clone encode: wav @ 24 kHz mono → codes (n_q, T25). Semantic branch
// resamples to 16 kHz, F.pad(160,160), HuBERT, mean-13, [::2]; acoustic branch runs
// the DAC encoder on 24 kHz; concat[acoustic,semantic] → fc → RVQ.
static std::vector<int32_t> higgs_encode(omnivoice_context* ctx, const float* wav24k, int n24k, int* out_T) {
    auto& tok = ctx->tokenizer;
    // Acoustic branch (24 kHz).
    int T_ac = 0;
    auto e_acoustic = higgs_acoustic_encode(ctx, wav24k, n24k, &T_ac); // (T_ac, 256)
    if (e_acoustic.empty())
        return {};
    // Semantic branch: resample 24→16k, pad 160 both sides.
    auto wav16 = core_audio::resample_polyphase(wav24k, n24k, 24000, 16000);
    std::vector<float> wav16p(wav16.size() + 320, 0.0f);
    std::memcpy(wav16p.data() + 160, wav16.data(), wav16.size() * sizeof(float));
    int T_fe = 0;
    auto featproj = higgs_hubert_frontend(ctx, wav16p.data(), (int)wav16p.size(), &T_fe, nullptr);
    if (featproj.empty())
        return {};
    auto mean13 = higgs_hubert_encoder(ctx, featproj.data(), T_fe, nullptr); // (T_fe, 768)
    // [::2] downsample.
    int T_se = (T_fe + 1) / 2;
    std::vector<float> sem_ds((size_t)T_se * tok.sem_d);
    for (int t = 0; t < T_se; t++)
        std::memcpy(sem_ds.data() + (size_t)t * tok.sem_d, mean13.data() + (size_t)(2 * t) * tok.sem_d,
                    tok.sem_d * sizeof(float));
    auto e_semantic = higgs_encoder_semantic(ctx, sem_ds.data(), T_se); // (T_se, 768)
    // Align lengths (branches can differ by ±1).
    int T = std::min(T_ac, T_se);
    const int Ca = tok.dac_hidden, Cs = tok.sem_d, Hh = tok.hidden_size; // 256, 768, 1024
    std::vector<float> cat((size_t)T * Hh);
    for (int t = 0; t < T; t++) {
        std::memcpy(cat.data() + (size_t)t * Hh, e_acoustic.data() + (size_t)t * Ca, Ca * sizeof(float));
        std::memcpy(cat.data() + (size_t)t * Hh + Ca, e_semantic.data() + (size_t)t * Cs, Cs * sizeof(float));
    }
    std::vector<float> fc_w = read_tensor_f32(tok.fc_w);
    std::vector<float> fc_b = tok.fc_b ? read_tensor_f32(tok.fc_b) : std::vector<float>();
    std::vector<float> emb;
    higgs_linear(fc_w, fc_b, cat.data(), T, Hh, Hh, emb);
    std::vector<int32_t> codes;
    if (!higgs_rvq_encode(tok, emb, T, codes))
        return {};
    *out_T = T;
    return codes; // (n_q, T)
}

// Run the encode diff against a Python reference archive. Stage-by-stage: each
// ported stage is fed the REFERENCE input for the previous stage, so the first
// mismatch localizes the bug. Returns 0 on success.
static int run_encode_diff(omnivoice_context* ctx, const char* ref_path) {
    auto& tok = ctx->tokenizer;
    if (!tok.loaded) {
        fprintf(stderr, "omnivoice: encode-diff needs the tokenizer loaded\n");
        return -1;
    }
    ov_ref_gguf ref;
    if (!ref.load(ref_path)) {
        fprintf(stderr, "omnivoice: encode-diff cannot open ref %s\n", ref_path);
        return -1;
    }
    fprintf(stderr, "omnivoice encode-diff vs %s\n", ref_path);

    // --- Stage HuBERT frontend: feed ref wav16k_pad (isolates resampling) →
    //     hb_featextract + hb_featproj. Uses a separate archive via env. ---
    if (const char* hp = crispasr_env::get("CRISPASR_OMNIVOICE_HUBERT_REF")) {
        ov_ref_gguf href;
        if (href.load(hp)) {
            ggml_tensor* r_wav = href.get("wav16k_pad");
            ggml_tensor* r_fe = href.get("hb_featextract");
            ggml_tensor* r_fp = href.get("hb_featproj");
            if (r_wav && r_fp) {
                int T_samp = (int)r_wav->ne[0];
                int T_out = 0;
                std::vector<ov_dbg_tensor> dbg;
                auto mine = higgs_hubert_frontend(ctx, (const float*)r_wav->data, T_samp, &T_out, &dbg);
                for (auto& d : dbg) {
                    if (d.name == "hb_featextract" && r_fe && (int)r_fe->ne[0] == d.C && (int)r_fe->ne[1] == d.T) {
                        float cmin, cmean, mabs;
                        higgs_cos(d.data.data(), (const float*)r_fe->data, d.T, d.C, cmin, cmean, mabs);
                        fprintf(stderr, "  hb_featextract: cos_min=%.6f cos_mean=%.6f max_abs=%.3e  %s\n", cmin, cmean,
                                mabs, cmin >= 0.999f ? "PASS" : "FAIL");
                    }
                }
                int C = (int)r_fp->ne[0], T = (int)r_fp->ne[1];
                if ((int)mine.size() == T * C && T_out == T) {
                    float cmin, cmean, mabs;
                    higgs_cos(mine.data(), (const float*)r_fp->data, T, C, cmin, cmean, mabs);
                    fprintf(stderr, "  hb_featproj: cos_min=%.6f cos_mean=%.6f max_abs=%.3e  %s\n", cmin, cmean, mabs,
                            cmin >= 0.999f ? "PASS" : "FAIL");
                    // Encoder: feed ref featproj → hb_layer0/11 + hb_mean13.
                    std::vector<ov_dbg_tensor> edbg;
                    auto emean = higgs_hubert_encoder(ctx, (const float*)r_fp->data, T, &edbg);
                    auto cmp = [&](const char* nm, const float* md, int mc, int mt) {
                        ggml_tensor* rt = href.get(nm);
                        if (!rt || (int)rt->ne[0] != mc || (int)rt->ne[1] != mt) {
                            fprintf(stderr, "    %-11s ref missing/shape\n", nm);
                            return;
                        }
                        float a, b2, c2;
                        higgs_cos(md, (const float*)rt->data, mt, mc, a, b2, c2);
                        fprintf(stderr, "    %-11s cos_min=%.6f cos_mean=%.6f max_abs=%.3e %s\n", nm, a, b2, c2,
                                a >= 0.999f ? "PASS" : "<< FAIL");
                    };
                    for (auto& d : edbg)
                        cmp(d.name.c_str(), d.data.data(), d.C, d.T);
                    if ((int)emean.size() == T * C)
                        cmp("hb_mean13", emean.data(), C, T);
                }
            }
        }
    }

    // --- Stage acoustic encoder: ref input_wav24k → e_acoustic (T,256). ---
    {
        ggml_tensor* r_wav = ref.get("input_wav24k");
        ggml_tensor* r_ac = ref.get("e_acoustic");
        if (r_wav && r_ac) {
            int T_samp = (int)r_wav->ne[0];
            int T_out = 0;
            // Optional per-block bisect vs a separate intermediates archive.
            std::vector<ov_dbg_tensor> dbg;
            const char* bisect = crispasr_env::get("CRISPASR_OMNIVOICE_ACENC_BISECT");
            auto mine = higgs_acoustic_encode(ctx, (const float*)r_wav->data, T_samp, &T_out, bisect ? &dbg : nullptr);
            if (bisect) {
                ov_ref_gguf bref;
                if (bref.load(bisect)) {
                    fprintf(stderr, "  acoustic_enc bisect vs %s:\n", bisect);
                    for (auto& d : dbg) {
                        ggml_tensor* rt = bref.get(d.name.c_str());
                        if (!rt) {
                            fprintf(stderr, "    %-10s ref missing\n", d.name.c_str());
                            continue;
                        }
                        int rc = (int)rt->ne[0], rT = (int)rt->ne[1];
                        if (rc != d.C || rT != d.T) {
                            fprintf(stderr, "    %-10s SHAPE mine=(%d,%d) ref=(%d,%d)\n", d.name.c_str(), d.T, d.C, rT,
                                    rc);
                            continue;
                        }
                        float cmin, cmean, mabs;
                        higgs_cos(d.data.data(), (const float*)rt->data, d.T, d.C, cmin, cmean, mabs);
                        fprintf(stderr, "    %-10s (%d,%d) cos_min=%.6f cos_mean=%.6f max_abs=%.3e %s\n",
                                d.name.c_str(), d.T, d.C, cmin, cmean, mabs, cmin >= 0.999f ? "ok" : "<< DIVERGES");
                        if (cmin < 0.999f) {
                            const float* rf = (const float*)rt->data;
                            // Find the worst row and probe channel ranges of frame 0.
                            int worst = 0;
                            float worst_cos = 2.0f;
                            for (int t = 0; t < d.T; t++) {
                                const float* x = d.data.data() + (size_t)t * d.C;
                                const float* y = rf + (size_t)t * d.C;
                                double dot = 0, nx = 0, ny = 0;
                                for (int c = 0; c < d.C; c++) {
                                    dot += (double)x[c] * y[c];
                                    nx += (double)x[c] * x[c];
                                    ny += (double)y[c] * y[c];
                                }
                                float cs = (nx > 0 && ny > 0) ? (float)(dot / (sqrt(nx) * sqrt(ny))) : 1.0f;
                                if (cs < worst_cos) {
                                    worst_cos = cs;
                                    worst = t;
                                }
                            }
                            fprintf(stderr, "        worst row t=%d cos=%.4f\n", worst, worst_cos);
                            // Locate the single max-abs-error element.
                            int mt = 0, mc = 0;
                            float me = 0;
                            for (int t = 0; t < d.T; t++)
                                for (int c = 0; c < d.C; c++) {
                                    float e = std::fabs(d.data[(size_t)t * d.C + c] - rf[(size_t)t * d.C + c]);
                                    if (e > me) {
                                        me = e;
                                        mt = t;
                                        mc = c;
                                    }
                                }
                            fprintf(stderr, "        max-err at t=%d c=%d: mine=%+.4f ref=%+.4f (|d|=%.3f)\n", mt, mc,
                                    d.data[(size_t)mt * d.C + mc], rf[(size_t)mt * d.C + mc], me);
                            auto probe = [&](int t, int c0) {
                                fprintf(stderr, "        t%d c%d: mine", t, c0);
                                for (int c = c0; c < c0 + 4; c++)
                                    fprintf(stderr, " %+.3f", d.data[(size_t)t * d.C + c]);
                                fprintf(stderr, " | ref");
                                for (int c = c0; c < c0 + 4; c++)
                                    fprintf(stderr, " %+.3f", rf[(size_t)t * d.C + c]);
                                fprintf(stderr, "\n");
                            };
                            probe(worst, 0);
                            probe(worst, d.C / 2);
                            probe(worst, d.C - 4);
                        }
                    }
                }
            }
            int C = (int)r_ac->ne[0], T = (int)r_ac->ne[1];
            if ((int)mine.size() == T * C && T_out == T) {
                float cmin, cmean, mabs;
                higgs_cos(mine.data(), (const float*)r_ac->data, T, C, cmin, cmean, mabs);
                fprintf(stderr, "  acoustic_enc (→e_acoustic): cos_min=%.6f cos_mean=%.6f max_abs=%.3e  %s\n", cmin,
                        cmean, mabs, cmin >= 0.999f ? "PASS" : "FAIL");
            } else {
                fprintf(stderr, "  acoustic_enc: shape mismatch mine=%zu (T_out=%d) ref=(%d,%d)\n", mine.size(), T_out,
                        T, C);
            }
        } else {
            fprintf(stderr, "  acoustic_enc: ref missing input_wav24k/e_acoustic\n");
        }
    }

    // --- Stage encoder_semantic: ref sem_ds (T,768) → e_semantic (T,768). ---
    {
        ggml_tensor* r_sem = ref.get("sem_ds");
        ggml_tensor* r_es = ref.get("e_semantic");
        if (r_sem && r_es) {
            int T = (int)r_sem->ne[1], C = (int)r_sem->ne[0];
            auto mine = higgs_encoder_semantic(ctx, (const float*)r_sem->data, T);
            if ((int)mine.size() == T * C) {
                float cmin, cmean, mabs;
                higgs_cos(mine.data(), (const float*)r_es->data, T, C, cmin, cmean, mabs);
                fprintf(stderr, "  encoder_semantic (→e_semantic): cos_min=%.6f cos_mean=%.6f max_abs=%.3e  %s\n", cmin,
                        cmean, mabs, cmin >= 0.999f ? "PASS" : "FAIL");
            } else {
                fprintf(stderr, "  encoder_semantic: size mismatch mine=%zu ref=%d\n", mine.size(), T * C);
            }
        } else {
            fprintf(stderr, "  encoder_semantic: ref missing sem_ds/e_semantic\n");
        }
    }

    // --- Stage concat+fc: ref e_acoustic(T,256) + e_semantic(T,768) → emb_fc. ---
    {
        ggml_tensor* r_ac = ref.get("e_acoustic");
        ggml_tensor* r_se = ref.get("e_semantic");
        ggml_tensor* r_emb = ref.get("emb_fc");
        if (r_ac && r_se && r_emb && tok.fc_w) {
            int T = (int)r_ac->ne[1];
            int Ca = (int)r_ac->ne[0]; // 256
            int Cs = (int)r_se->ne[0]; // 768
            int H = Ca + Cs;           // 1024
            const float* ac = (const float*)r_ac->data;
            const float* se = (const float*)r_se->data;
            std::vector<float> cat((size_t)T * H);
            for (int t = 0; t < T; t++) {
                std::memcpy(cat.data() + (size_t)t * H, ac + (size_t)t * Ca, Ca * sizeof(float));
                std::memcpy(cat.data() + (size_t)t * H + Ca, se + (size_t)t * Cs, Cs * sizeof(float));
            }
            std::vector<float> fc_w = read_tensor_f32(tok.fc_w);
            std::vector<float> fc_b = tok.fc_b ? read_tensor_f32(tok.fc_b) : std::vector<float>();
            std::vector<float> mine;
            higgs_linear(fc_w, fc_b, cat.data(), T, H, H, mine);
            float cmin, cmean, mabs;
            higgs_cos(mine.data(), (const float*)r_emb->data, T, H, cmin, cmean, mabs);
            fprintf(stderr, "  concat+fc (→emb_fc): cos_min=%.6f cos_mean=%.6f max_abs=%.3e  %s\n", cmin, cmean, mabs,
                    cmin >= 0.999f ? "PASS" : "FAIL");
        } else {
            fprintf(stderr, "  concat+fc: ref missing e_acoustic/e_semantic/emb_fc or fc weights\n");
        }
    }

    // --- Stage RVQ: ref emb_fc (T,1024) → codes; compare to ref codes (NQ,T). ---
    ggml_tensor* r_emb = ref.get("emb_fc");
    ggml_tensor* r_codes = ref.get("codes");
    if (r_emb && r_codes) {
        int T = (int)r_emb->ne[1]; // ne=[1024, T]
        int H = (int)r_emb->ne[0];
        // Ref tensors live in the gguf ctx's own memory (no backend buffer) —
        // read tensor->data directly.
        std::vector<float> emb((size_t)T * H);
        std::memcpy(emb.data(), r_emb->data, emb.size() * sizeof(float));
        std::vector<int32_t> mine;
        if (higgs_rvq_encode(tok, emb, T, mine)) {
            // ref codes stored (NQ, T) row-major as float/int32.
            int NQ = tok.n_quantizers;
            std::vector<int32_t> refc((size_t)NQ * T);
            if (r_codes->type == GGML_TYPE_I32) {
                std::memcpy(refc.data(), r_codes->data, refc.size() * sizeof(int32_t));
            } else {
                const float* tmp = (const float*)r_codes->data;
                for (size_t i = 0; i < refc.size(); i++)
                    refc[i] = (int32_t)llroundf(tmp[i]);
            }
            long match = 0, tot = (long)NQ * T;
            std::vector<int> per_cb(NQ, 0);
            for (int k = 0; k < NQ; k++)
                for (int t = 0; t < T; t++)
                    if (mine[(size_t)k * T + t] == refc[(size_t)k * T + t]) {
                        match++;
                        per_cb[k]++;
                    }
            fprintf(stderr, "  RVQ (emb_fc→codes): %ld/%ld exact (%.1f%%)  per-cb:", match, tot, 100.0 * match / tot);
            for (int k = 0; k < NQ; k++)
                fprintf(stderr, " %d/%d", per_cb[k], T);
            fprintf(stderr, "\n    mine cb0[:12]:");
            for (int t = 0; t < std::min(12, T); t++)
                fprintf(stderr, " %d", mine[t]);
            fprintf(stderr, "\n    ref  cb0[:12]:");
            for (int t = 0; t < std::min(12, T); t++)
                fprintf(stderr, " %d", refc[t]);
            fprintf(stderr, "\n");
        }
    } else {
        fprintf(stderr, "  RVQ: ref missing emb_fc/codes\n");
    }

    // --- Full chain: ref input_wav24k → higgs_encode → codes (measures the
    //     24→16k resample impact; acceptance is the roundtrip, not exact codes). ---
    {
        ggml_tensor* r_wav = ref.get("input_wav24k");
        ggml_tensor* r_codes = ref.get("codes");
        if (r_wav && r_codes) {
            int T_out = 0;
            auto mine = higgs_encode(ctx, (const float*)r_wav->data, (int)r_wav->ne[0], &T_out);
            int NQ = tok.n_quantizers, T = (int)r_codes->ne[0];
            if (!mine.empty() && T_out == T) {
                std::vector<int32_t> refc((size_t)NQ * T);
                if (r_codes->type == GGML_TYPE_I32)
                    std::memcpy(refc.data(), r_codes->data, refc.size() * sizeof(int32_t));
                else
                    for (size_t i = 0; i < refc.size(); i++)
                        refc[i] = (int32_t)llroundf(((const float*)r_codes->data)[i]);
                long match = 0;
                for (size_t i = 0; i < refc.size(); i++)
                    if (mine[i] == refc[i])
                        match++;
                fprintf(stderr, "  FULL wav→codes: %ld/%zu exact (%.1f%%), T=%d\n", match, refc.size(),
                        100.0 * match / refc.size(), T_out);
            } else {
                fprintf(stderr, "  FULL wav→codes: T_out=%d ref T=%d (align)\n", T_out, T);
            }
        }
    }
    return 0;
}

} // namespace

// ===========================================================================
// Public API
// ===========================================================================

struct omnivoice_context* omnivoice_init_from_file(const char* path_model, struct omnivoice_context_params params) {
    auto* ctx = new omnivoice_context();
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx->verbosity = params.verbosity;
    ctx->use_gpu = params.use_gpu;
    ctx->flash_attn = params.flash_attn;

    // Generation config
    ctx->gen.num_steps = params.num_steps > 0 ? params.num_steps : 32;
    ctx->gen.guidance_scale = params.guidance_scale > 0.0f ? params.guidance_scale : 2.0f;
    ctx->gen.class_temperature = params.class_temperature > 0.0f ? params.class_temperature : 0.0f;
    ctx->gen.position_temperature = params.position_temperature > 0.0f ? params.position_temperature : 5.0f;
    ctx->gen.layer_penalty_factor = params.layer_penalty_factor > 0.0f ? params.layer_penalty_factor : 5.0f;
    ctx->gen.t_shift = params.t_shift > 0.0f ? params.t_shift : 0.1f;
    ctx->gen.seed = params.seed > 0 ? params.seed : 42;

    // Diagnostic env overrides (can set exact 0, unlike the CLI which treats 0 as
    // "use default"). Used to bisect the #254 word-dropping (CFG vs forward).
    if (const char* e = crispasr_env::get("CRISPASR_OMNIVOICE_GUIDANCE"))
        ctx->gen.guidance_scale = (float)atof(e);
    if (const char* e = crispasr_env::get("CRISPASR_OMNIVOICE_POS_TEMP"))
        ctx->gen.position_temperature = (float)atof(e);
    if (const char* e = crispasr_env::get("CRISPASR_OMNIVOICE_CLASS_TEMP"))
        ctx->gen.class_temperature = (float)atof(e);
    // num_steps is the dominant speed lever (stage0 = num_steps × 2 forwards).
    // Env override for quick A/B; the CLI --tts-steps / session setter also drive it.
    if (const char* e = crispasr_env::get("CRISPASR_OMNIVOICE_NUM_STEPS")) {
        int n = atoi(e);
        if (n > 0)
            ctx->gen.num_steps = n;
    }

    if (!load_model(ctx, path_model)) {
        delete ctx;
        return nullptr;
    }

    return ctx;
}

int omnivoice_set_tokenizer_path(struct omnivoice_context* ctx, const char* path) {
    if (!ctx || !path)
        return -1;
    ctx->tokenizer_path = path;
    if (!load_tokenizer(ctx, path)) {
        fprintf(stderr, "omnivoice: failed to load tokenizer from %s\n", path);
        return -1;
    }
    return 0;
}

int omnivoice_set_voice_prompt(struct omnivoice_context* ctx, const char* wav_path, const char* ref_text) {
    if (!ctx)
        return -1;
    if (!wav_path || !*wav_path) {
        ctx->ref_audio_codes.clear();
        ctx->ref_T = 0;
        ctx->ref_rms = 0.0f;
        ctx->ref_text.clear();
        return 0;
    }
    ctx->ref_text = ref_text ? ref_text : "";
    if (!ctx->tokenizer.loaded) {
        fprintf(stderr, "omnivoice: voice cloning requires the audio tokenizer — call "
                        "omnivoice_set_tokenizer_path first\n");
        return -1;
    }
    // Load WAV (mono), resample → 24 kHz, RMS-normalize, clip to a multiple of
    // hop_length(960), then encode through the tokenizer (HuBERT + DAC + RVQ).
    std::vector<float> pcm;
    int sr = 0;
    if (!crispasr::core::read_wav_mono_pcm16(wav_path, pcm, sr) || pcm.empty()) {
        fprintf(stderr, "omnivoice: failed to read voice WAV '%s'\n", wav_path);
        return -1;
    }
    if (sr != 24000)
        pcm = core_audio::resample_polyphase(pcm.data(), (int)pcm.size(), sr, 24000);
    double ss = 0;
    for (float v : pcm)
        ss += (double)v * v;
    float rms = (float)std::sqrt(ss / std::max<size_t>(1, pcm.size()));
    if (rms > 0.0f && rms < 0.1f) {
        float g = 0.1f / rms;
        for (float& v : pcm)
            v *= g;
    }
    size_t clip = pcm.size() % 960;
    if (clip)
        pcm.resize(pcm.size() - clip);
    if (pcm.empty()) {
        fprintf(stderr, "omnivoice: voice WAV too short after clipping\n");
        return -1;
    }

    // Reference-voice disk cache (#254 reporter request; omnivoice.cpp
    // --ref-rvq parity, but automatic). The RVQ encode (HuBERT + DAC + 8-stage
    // RVQ over the whole reference) is the expensive part of voice cloning and
    // its result is deterministic, so cache the codes content-addressed.
    // Content-addressed key = FNV-1a over the PREPROCESSED pcm (post
    // resample/RMS/clip — robust to container differences) + an encoder-weight
    // fingerprint (re-encode after a tokenizer re-conversion). Issue #265:
    // stored via the shared crispasr_ref_cache so OmniVoice now uses the SAME
    // <temp>/crispasr-tts-refcache dir (or CRISPASR_TTS_REF_CACHE_DIR) and the
    // SAME CRISPASR_TTS_REF_CACHE=0 disable switch as every other TTS backend.
    // CRISPASR_OMNIVOICE_VOICE_CACHE=0 is kept as a backward-compat alias.
    uint64_t cache_key = 0;
    bool cache_on = false;
    {
        const char* legacy = std::getenv("CRISPASR_OMNIVOICE_VOICE_CACHE");
        const bool legacy_off = legacy && *legacy && *legacy == '0';
        cache_on = !legacy_off && !crispasr_ref_cache::disabled() && ctx->tokenizer.enc_conv1_w != nullptr;
        if (cache_on) {
            uint64_t h = 1469598103934665603ull; // FNV-1a 64
            auto mix = [&h](const void* p, size_t n) {
                const uint8_t* b = (const uint8_t*)p;
                for (size_t i = 0; i < n; i++) {
                    h ^= b[i];
                    h *= 1099511628211ull;
                }
            };
            mix(pcm.data(), pcm.size() * sizeof(float));
            uint64_t n_samp = (uint64_t)pcm.size();
            mix(&n_samp, sizeof(n_samp));
            ggml_tensor* fp = ctx->tokenizer.enc_conv1_w;
            size_t fp_n = std::min<size_t>(4096, ggml_nbytes(fp));
            std::vector<uint8_t> fp_buf(fp_n);
            ggml_backend_tensor_get(fp, fp_buf.data(), 0, fp_n);
            mix(fp_buf.data(), fp_n);
            uint64_t total = (uint64_t)ggml_nbytes(fp);
            mix(&total, sizeof(total));
            cache_key = h;
        }
    }

    int T_ref = 0;
    std::vector<int32_t> codes;
    if (cache_on) {
        std::vector<uint32_t> shape;
        std::vector<uint8_t> payload;
        if (crispasr_ref_cache::get_bytes("omnivoice-voice", &cache_key, sizeof(cache_key), shape, payload) &&
            shape.size() == 2 && shape[0] == (uint32_t)ctx->hp.n_codebooks && shape[1] > 0 &&
            payload.size() == (size_t)shape[0] * shape[1] * sizeof(int32_t)) {
            T_ref = (int)shape[1];
            codes.resize((size_t)shape[0] * shape[1]);
            std::memcpy(codes.data(), payload.data(), payload.size());
            if (ctx->verbosity >= 1)
                fprintf(stderr, "omnivoice: voice codes loaded from cache (%d ref frames)\n", T_ref);
        }
    }
    if (codes.empty()) {
        codes = higgs_encode(ctx, pcm.data(), (int)pcm.size(), &T_ref);
        if (codes.empty() || T_ref <= 0) {
            fprintf(stderr, "omnivoice: voice-prompt encode failed\n");
            return -1;
        }
        if (ctx->verbosity >= 1)
            fprintf(stderr, "omnivoice: voice prompt encoded — %d ref frames (%d codebooks)\n", T_ref,
                    (int)ctx->hp.n_codebooks);
        if (cache_on)
            crispasr_ref_cache::put_bytes("omnivoice-voice", &cache_key, sizeof(cache_key),
                                          {(uint32_t)ctx->hp.n_codebooks, (uint32_t)T_ref}, codes.data(),
                                          codes.size() * sizeof(int32_t));
    }
    ctx->ref_audio_codes = std::move(codes);
    ctx->ref_T = T_ref;
    ctx->ref_rms = rms;
    return 0;
}

int omnivoice_set_language(struct omnivoice_context* ctx, const char* lang) {
    if (!ctx)
        return -1;
    ctx->language = lang ? lang : "";
    return 0;
}

int omnivoice_set_instruct(struct omnivoice_context* ctx, const char* instruct) {
    if (!ctx)
        return -1;
    ctx->instruct = instruct ? instruct : "";
    return 0;
}

int omnivoice_set_speed(struct omnivoice_context* ctx, float speed) {
    if (!ctx)
        return -1;
    ctx->speed = (speed > 0.0f) ? speed : 1.0f;
    return 0;
}

int omnivoice_set_num_steps(struct omnivoice_context* ctx, int num_steps) {
    if (!ctx)
        return -1;
    if (num_steps >= 1)
        ctx->gen.num_steps = num_steps; // read live by generate_iterative
    return 0;
}

int32_t* omnivoice_synthesize_codes(struct omnivoice_context* ctx, const char* text, int* out_n_codes) {
    if (!ctx || !text || !out_n_codes)
        return nullptr;

    ov_gen_result result = generate_iterative(ctx, text);

    if (result.codes.empty()) {
        *out_n_codes = 0;
        return nullptr;
    }

    int n = (int)result.codes.size();
    int32_t* out = (int32_t*)malloc(n * sizeof(int32_t));
    std::memcpy(out, result.codes.data(), n * sizeof(int32_t));
    *out_n_codes = n;
    return out;
}

void omnivoice_codes_free(int32_t* codes) {
    free(codes);
}

float* omnivoice_decode_codes(struct omnivoice_context* ctx, const int32_t* codes, int n_codes, int* out_n_samples) {
    if (!ctx || !codes || n_codes <= 0 || !out_n_samples)
        return nullptr;
    if (!ctx->tokenizer.loaded) {
        fprintf(stderr, "omnivoice: decode requires audio tokenizer — call omnivoice_set_tokenizer_path first\n");
        *out_n_samples = 0;
        return nullptr;
    }

    int n_cb = (int)ctx->hp.n_codebooks;
    int T_frames = n_codes / n_cb;
    if (T_frames <= 0 || n_codes != n_cb * T_frames) {
        fprintf(stderr, "omnivoice: decode_codes: n_codes=%d not divisible by n_codebooks=%d\n", n_codes, n_cb);
        *out_n_samples = 0;
        return nullptr;
    }

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "omnivoice: decoding %d codes (%d codebooks × %d frames)\n", n_codes, n_cb, T_frames);
    }

    ov_bench_stage bench("decode");
    auto pcm = higgs_decode(ctx, codes, n_cb, T_frames);
    if (pcm.empty()) {
        *out_n_samples = 0;
        return nullptr;
    }

    // Match the reference implementation's post-processing: quiet prompts are
    // normalized to 0.1 RMS for encoding, then decoded audio is restored to the
    // prompt's original loudness.
    if (ctx->ref_rms > 0.0f && ctx->ref_rms < 0.1f) {
        const float gain = ctx->ref_rms / 0.1f;
        for (float& sample : pcm)
            sample *= gain;
    }

    int n = (int)pcm.size();
    float* out = (float*)malloc(n * sizeof(float));
    std::memcpy(out, pcm.data(), n * sizeof(float));
    *out_n_samples = n;

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "omnivoice: decoded %d samples (%.2f s at %d Hz)\n", n, (float)n / 24000.0f, 24000);
    }
    return out;
}

float* omnivoice_synthesize(struct omnivoice_context* ctx, const char* text, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;

    // Per-stage timing + RTF summary (reporter ask, #254): surfaced at normal
    // verbosity so callers don't have to wrap the process in `time`. The
    // OMNIVOICE_BENCH env still emits the fine-grained per-step breakdown.
    using clk = std::chrono::steady_clock;
    auto t_gen0 = clk::now();
    int n_codes = 0;
    int32_t* codes = omnivoice_synthesize_codes(ctx, text, &n_codes);
    auto t_gen1 = clk::now();
    if (!codes) {
        *out_n_samples = 0;
        return nullptr;
    }

    float* pcm = omnivoice_decode_codes(ctx, codes, n_codes, out_n_samples);
    auto t_dec1 = clk::now();
    omnivoice_codes_free(codes);

    if (ctx->verbosity >= 1 && pcm && *out_n_samples > 0) {
        double gen_s = std::chrono::duration<double>(t_gen1 - t_gen0).count();
        double dec_s = std::chrono::duration<double>(t_dec1 - t_gen1).count();
        double total_s = gen_s + dec_s;
        double audio_s = (double)*out_n_samples / 24000.0;
        fprintf(stderr,
                "omnivoice: timing — gen %.2fs + decode %.2fs = %.2fs for %.2fs audio "
                "(RTF %.2f; decode RTF %.2f)\n",
                gen_s, dec_s, total_s, audio_s, total_s / (audio_s > 0 ? audio_s : 1.0),
                dec_s / (audio_s > 0 ? audio_s : 1.0));
    }
    return pcm;
}

void omnivoice_pcm_free(float* pcm) {
    free(pcm);
}

void omnivoice_free(struct omnivoice_context* ctx) {
    if (!ctx)
        return;
    // Main model
    if (ctx->buf_w)
        ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    // Tokenizer
    ctx->tokenizer.fc.free();
    if (ctx->tokenizer.buf_w)
        ggml_backend_buffer_free(ctx->tokenizer.buf_w);
    if (ctx->tokenizer.backend)
        ggml_backend_free(ctx->tokenizer.backend);
    if (ctx->tokenizer.ctx_w)
        ggml_free(ctx->tokenizer.ctx_w);
    delete ctx;
}

void omnivoice_sync(struct omnivoice_context* ctx) {
    if (ctx && ctx->backend)
        ggml_backend_synchronize(ctx->backend);
}

int omnivoice_encode_diff(struct omnivoice_context* ctx, const char* ref_gguf_path) {
    if (!ctx || !ref_gguf_path)
        return -1;
    return run_encode_diff(ctx, ref_gguf_path);
}

void omnivoice_set_n_threads(struct omnivoice_context* ctx, int n_threads) {
    if (ctx && n_threads > 0) {
        ctx->n_threads = n_threads;
        if (ctx->backend && ggml_backend_is_cpu(ctx->backend))
            ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);
    }
}
