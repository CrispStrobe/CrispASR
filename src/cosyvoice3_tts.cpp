// cosyvoice3_tts.cpp — runtime for FunAudioLLM/Fun-CosyVoice3-0.5B-2512
//
// Phase 2: the CosyVoice3LM forward — Qwen2-0.5B body with two extra
// tensors over a vanilla Qwen2-0.5B-Instruct:
//
//   speech_embd     (6761, 896) — speech-token input embedding
//   speech_lm_head  (6761, 896) — speech-token AR head
//
// The body itself is identical to mimo_asr's Qwen2 step graph minus
// the fused-QKV path (the cosyvoice3 converter emits separate
// q/k/v tensors), so this file mostly leans on `core_attn::kv_self_attn`
// + `core_ffn::swiglu` and only adds the speech-side I/O.
//
// Architecture refs (lifted from PLAN.md §51):
//   d_model       = 896    (hidden_size)
//   n_layers      = 24
//   n_heads       = 14, n_kv_heads = 2  (GQA group = 7)
//   head_dim      = 64     (= d_model / n_heads)
//   ff_dim        = 4864   (intermediate_size)
//   rope_theta    = 1e6, rms_norm_eps = 1e-6
//   text_vocab    = 151936 (Qwen2 tokenizer, gpt2-BPE encoded into GGUF)
//   speech_vocab  = 6761   (head dim; codebook is [0, 6561), the upper
//                            ~200 entries are special / EOS markers)

// MSVC's <cmath> strips POSIX-only math constants like M_PI unless
// _USE_MATH_DEFINES is set BEFORE the include. Defining it here keeps
// the Windows whisper.dll build working — Unix toolchains expose M_PI
// unconditionally so the define is a no-op there.
#define _USE_MATH_DEFINES

#include "cosyvoice3_tts.h"

#include "core/attention.h"
#include "core/bpe.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/audio_resample.h"
#include "core/fft.h"
#include "core/mel.h"
#include "core/wav_reader.h"
#include "chatterbox_campplus.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#ifndef _WIN32
#include <unistd.h>
#endif

namespace {

// ===========================================================================
// Bench instrumentation — `COSYVOICE3_BENCH=1` for per-stage timings.
// ===========================================================================

static bool cosyvoice3_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("COSYVOICE3_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct cosyvoice3_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit cosyvoice3_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~cosyvoice3_bench_stage() {
        if (!cosyvoice3_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  cosyvoice3_bench: %-22s %.2f ms\n", name, ms);
    }
};

struct cv3_hp {
    uint32_t n_layers = 24;
    uint32_t d_model = 896;
    uint32_t n_heads = 14;
    uint32_t n_kv_heads = 2;
    uint32_t head_dim = 64;
    uint32_t ff_dim = 4864;
    uint32_t text_vocab = 151936;
    uint32_t max_pos = 32768;
    uint32_t speech_vocab = 6761;
    uint32_t speech_codebook = 6561;
    float rope_theta = 1e6f;
    float rms_norm_eps = 1e-6f;
};

struct cv3_qwen2_block {
    ggml_tensor* attn_norm_w = nullptr; // RMSNorm γ
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_k_b = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_v_b = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct cv3_lm {
    ggml_tensor* token_embd_w = nullptr;     // [d_model, text_vocab]
    ggml_tensor* output_norm_w = nullptr;    // [d_model]
    ggml_tensor* text_output_w = nullptr;    // [d_model, text_vocab] (unused in speech AR)
    ggml_tensor* speech_embd_w = nullptr;    // [d_model, speech_vocab]
    ggml_tensor* speech_lm_head_w = nullptr; // [d_model, speech_vocab]
    std::vector<cv3_qwen2_block> blocks;
};

// ---------------------------------------------------------------------------
// Phase 3 — Flow (DiT-CFM) hparams + tensor binding
// ---------------------------------------------------------------------------

struct cv3_flow_hp {
    uint32_t n_dit_layers = 22;
    uint32_t dit_dim = 1024;
    uint32_t dit_heads = 16;
    uint32_t dit_head_dim = 64;
    uint32_t dit_ff_dim = 2048;
    uint32_t dit_input_dim = 320;
    uint32_t mel_dim = 80;
    uint32_t spk_dim_in = 192;
    uint32_t spk_dim_out = 80;
    uint32_t speech_codebook = 6561;
    uint32_t pre_lookahead_len = 3;
    uint32_t token_mel_ratio = 2;
    uint32_t input_frame_rate = 25;
    uint32_t cfm_n_steps = 10;
    float cfm_inference_cfg_rate = 0.7f;
    float cfm_sigma_min = 1e-6f;
    float rope_theta = 10000.0f;
};

// One DiT block — AdaLN-Zero modulation (6×dim split for γ/β/gate × 2)
// projected from time-embed, followed by MHA (with RoPE inside) and an
// FFN (l1 → SiLU → l2).
struct cv3_dit_block {
    ggml_tensor* adaln_w = nullptr; // [dit_dim, 6*dit_dim]
    ggml_tensor* adaln_b = nullptr; // [6*dit_dim] F32
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_k_b = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_v_b = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* attn_o_b = nullptr;
    ggml_tensor* ffn_l1_w = nullptr; // [dit_dim, ff_dim]
    ggml_tensor* ffn_l1_b = nullptr;
    ggml_tensor* ffn_l2_w = nullptr; // [ff_dim, dit_dim]
    ggml_tensor* ffn_l2_b = nullptr;
};

struct cv3_flow {
    bool loaded = false;
    cv3_flow_hp hp;

    // Top-level
    ggml_tensor* input_embd_w = nullptr; // (mel_dim=80, speech_codebook=6561)
    ggml_tensor* pre_la_c1_w = nullptr;  // (K=4, 80, 1024) ggml ne
    ggml_tensor* pre_la_c1_b = nullptr;
    ggml_tensor* pre_la_c2_w = nullptr; // (K=3, 1024, 80)
    ggml_tensor* pre_la_c2_b = nullptr;
    ggml_tensor* spk_affine_w = nullptr; // (spk_dim_in=192, spk_dim_out=80)
    ggml_tensor* spk_affine_b = nullptr;

    // DiT input / time / position / output
    ggml_tensor* dit_in_proj_w = nullptr; // (320, 1024)
    ggml_tensor* dit_in_proj_b = nullptr;
    ggml_tensor* dit_conv_pos_c1_w = nullptr; // grouped conv1d-31 (K, in_per_grp, out)
    ggml_tensor* dit_conv_pos_c1_b = nullptr;
    ggml_tensor* dit_conv_pos_c2_w = nullptr;
    ggml_tensor* dit_conv_pos_c2_b = nullptr;
    ggml_tensor* dit_time_mlp_0_w = nullptr; // (256, 1024)
    ggml_tensor* dit_time_mlp_0_b = nullptr;
    ggml_tensor* dit_time_mlp_2_w = nullptr; // (1024, 1024)
    ggml_tensor* dit_time_mlp_2_b = nullptr;
    ggml_tensor* dit_rope_inv_freq = nullptr; // (head_dim/2,)
    ggml_tensor* dit_norm_out_w = nullptr;
    ggml_tensor* dit_norm_out_b = nullptr;
    ggml_tensor* dit_proj_out_w = nullptr;
    ggml_tensor* dit_proj_out_b = nullptr;

    std::vector<cv3_dit_block> blocks;

    // Flow-side ggml context + buffer (separate from the LM's).
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

struct cv3_s3tok_hp {
    uint32_t n_mels = 128;
    uint32_t n_audio_state = 1280;
    uint32_t n_audio_head = 20;
    uint32_t n_audio_layer = 12;
    uint32_t n_codebook_size = 6561;
    uint32_t n_fft = 400;
    uint32_t hop_length = 160;
    uint32_t win_length = 400;
    uint32_t fsmn_kernel = 31;
    float rope_theta = 10000.0f;
    float attn_ln_eps = 1e-6f;
    float mlp_ln_eps = 1e-5f;
};

struct cv3_s3tok_block {
    ggml_tensor* attn_ln_w = nullptr;
    ggml_tensor* attn_ln_b = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_b = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_o_b = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* mlp_ln_w = nullptr;
    ggml_tensor* mlp_ln_b = nullptr;
    ggml_tensor* mlp_up_b = nullptr;
    ggml_tensor* mlp_up_w = nullptr;
    ggml_tensor* mlp_dn_b = nullptr;
    ggml_tensor* mlp_dn_w = nullptr;
    ggml_tensor* fsmn_w = nullptr;
};

struct cv3_s3tok {
    bool loaded = false;
    cv3_s3tok_hp hp;
    ggml_tensor* conv0_w = nullptr;
    ggml_tensor* conv0_b = nullptr;
    ggml_tensor* conv1_w = nullptr;
    ggml_tensor* conv1_b = nullptr;
    std::vector<cv3_s3tok_block> blocks;
    ggml_tensor* fsq_proj_w = nullptr;
    ggml_tensor* fsq_proj_b = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

struct cv3_campplus {
    bool loaded = false;
    cb_campplus_model model;
    chatterbox_campplus::cb_campplus_runtime cache{};
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

// ---------------------------------------------------------------------------
// Phase 4 — HiFT (CausalHiFTGenerator) hparams + tensor binding
// ---------------------------------------------------------------------------

struct cv3_hift_hp {
    uint32_t sample_rate = 24000;
    uint32_t mel_dim = 80;
    uint32_t base_channels = 512;
    uint32_t nb_harmonics = 8;
    uint32_t istft_n_fft = 16;
    uint32_t istft_hop = 4;
    uint32_t n_upsample_stages = 3;
    // Per-stage upsample rate / kernel (read from upsample_rate.N keys).
    uint32_t upsample_rates[3] = {8, 5, 3};
    uint32_t upsample_kernels[3] = {16, 11, 7};
    float lrelu_slope = 0.1f;
    float audio_limit = 0.99f;
    uint32_t conv_pre_look_right = 4;
};

// One HiFT ResBlock: 3× (Conv1d, Conv1d, Snake-Beta alpha) for c1 + 3× for c2.
// `c1` is the first conv in each sub-block; `c2` is the second. `a1`/`a2` are
// the Snake activation alpha parameters (channels-wide).
struct cv3_hift_resblock {
    ggml_tensor* c1_w[3] = {nullptr, nullptr, nullptr};
    ggml_tensor* c1_b[3] = {nullptr, nullptr, nullptr};
    ggml_tensor* c2_w[3] = {nullptr, nullptr, nullptr};
    ggml_tensor* c2_b[3] = {nullptr, nullptr, nullptr};
    ggml_tensor* a1_alpha[3] = {nullptr, nullptr, nullptr};
    ggml_tensor* a2_alpha[3] = {nullptr, nullptr, nullptr};
};

// Qwen2 BPE vocab loaded from the LLM GGUF's
// `tokenizer.ggml.tokens` + `tokenizer.ggml.merges` (the converter
// writes the gpt2-byte-encoded vocab.json verbatim). Plus the small
// set of CV3 special tokens (`<|endoftext|>`, `<|im_start|>`,
// `<|im_end|>`, `<|endofprompt|>`) that the BlankEN vocab.json does
// not include but the model was trained against.
struct cv3_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
};

// One baked voice for zero-shot cloning. The runtime keeps these
// resident in CPU memory — they're at most a few hundred KB each.
struct cv3_voice {
    std::string name;
    std::vector<int32_t> prompt_speech_tokens;
    std::string prompt_text;
    std::vector<float> spk_emb; // size 192
    std::vector<float> ref_mel; // (T_ref_mel, 80) row-major
    int t_ref_mel = 0;
};

struct cv3_voices {
    bool loaded = false;
    std::vector<cv3_voice> voices;
    std::unordered_map<std::string, int> by_name;
};

struct cv3_hift {
    bool loaded = false;
    cv3_hift_hp hp;

    // Top-level convs.
    ggml_tensor* conv_pre_w = nullptr;
    ggml_tensor* conv_pre_b = nullptr;
    ggml_tensor* conv_post_w = nullptr;
    ggml_tensor* conv_post_b = nullptr;

    // Upsample (ConvTranspose1d) — one per upsample stage.
    ggml_tensor* ups_w[3] = {nullptr, nullptr, nullptr};
    ggml_tensor* ups_b[3] = {nullptr, nullptr, nullptr};

    // 9 main ResBlocks (3 per upsample stage × 3 stages).
    std::vector<cv3_hift_resblock> resblocks; // size = 9

    // 3 source_downs + 3 source ResBlocks.
    ggml_tensor* src_down_w[3] = {nullptr, nullptr, nullptr};
    ggml_tensor* src_down_b[3] = {nullptr, nullptr, nullptr};
    std::vector<cv3_hift_resblock> src_resblocks; // size = 3

    // SineGen / source projection.
    ggml_tensor* m_source_l_linear_w = nullptr;
    ggml_tensor* m_source_l_linear_b = nullptr;

    // F0 predictor (CausalConvRNNF0Predictor): 5 condnet convs + classifier.
    ggml_tensor* f0_condnet_w[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    ggml_tensor* f0_condnet_b[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    ggml_tensor* f0_classifier_w = nullptr;
    ggml_tensor* f0_classifier_b = nullptr;

    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

} // namespace

struct cosyvoice3_tts_context {
    cosyvoice3_tts_context_params params{};
    int n_threads = 4;
    uint64_t seed = 42;

    cv3_hp hp;
    cv3_lm lm;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_backend_buffer_t buf_w_cpu = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    std::vector<uint8_t> compute_meta;

    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;

    // Cached T=1 step graph (speech-token in). Built lazily on first
    // step call with fixed_kv_len = kv_max_ctx; subsequent steps reuse
    // the same plan via skip_plan.
    ggml_cgraph* step_t1_gf = nullptr;
    int step_t1_fixed_kv_len = 0;

    // §192 CPU embed cache: all speech_embd rows dequantized to F32 at
    // init. Avoids a GPU round-trip + graph-cache invalidation per AR step.
    std::vector<float> speech_embd_cache; // [speech_vocab * d_model]

    // RAS sampler RNG. Seeded once at init from params.seed (or 42);
    // re-seedable via cosyvoice3_tts_set_seed. Advances through every
    // RAS sample so repeated generate() calls don't replay.
    std::mt19937_64 rng{42};

    // Phase 3 — Flow sub-model (DiT-CFM). Populated by
    // cosyvoice3_tts_init_flow_from_file(). Stays empty if only the
    // LM was loaded (`flow.loaded == false`).
    cv3_flow flow{};

    // Phase 4 — HiFT (CausalHiFTGenerator) vocoder. Populated by
    // cosyvoice3_tts_init_hift_from_file(). Stays empty if HiFT was
    // not loaded (`hift.loaded == false`).
    cv3_hift hift{};

    // Phase 6 — speech_tokenizer_v3 encoder for arbitrary-WAV cloning.
    // Loaded from the dedicated s3tok GGUF when available.
    cv3_s3tok s3tok{};

    // Phase 6 — CAMPPlus speaker encoder for arbitrary-WAV cloning.
    // Loaded from the dedicated CAMPPlus GGUF when available.
    cv3_campplus campplus{};

    // Phase 5 — Qwen2 BPE vocab + voice-clone bundle. The vocab is
    // populated alongside the LLM init (the LLM GGUF already carries
    // `tokenizer.ggml.tokens` + `tokenizer.ggml.merges`); the voices
    // table is populated separately via init_voices_from_file().
    cv3_vocab vocab{};
    cv3_voices voices{};
};

namespace {

uint32_t cv3_kv_u32(gguf_context* ctx, const char* key, uint32_t def) {
    int64_t id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_u32(ctx, id) : def;
}
float cv3_kv_f32(gguf_context* ctx, const char* key, float def) {
    int64_t id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_f32(ctx, id) : def;
}

bool cv3_kv_init(cosyvoice3_tts_context* ctx, int max_ctx) {
    if (ctx->kv_k && ctx->kv_max_ctx >= max_ctx)
        return true;
    if (ctx->kv_buf) {
        ggml_backend_buffer_free(ctx->kv_buf);
        ctx->kv_buf = nullptr;
    }
    if (ctx->kv_ctx) {
        ggml_free(ctx->kv_ctx);
        ctx->kv_ctx = nullptr;
    }
    const auto& hp = ctx->hp;
    const int hd = (int)hp.head_dim;
    const int n_kv = (int)hp.n_kv_heads;
    const int n_lay = (int)hp.n_layers;
    const auto kv_pair = core_attn::kv_dtype_pair_from_env("cosyvoice3_tts");
    ggml_init_params kp = {ggml_tensor_overhead() * 4 + 1024, nullptr, true};
    ctx->kv_ctx = ggml_init(kp);
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.k, hd, max_ctx, n_kv, n_lay);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.v, hd, max_ctx, n_kv, n_lay);
    ggml_set_name(ctx->kv_k, "cv3_kv_k");
    ggml_set_name(ctx->kv_v, "cv3_kv_v");
    const size_t kbytes = ggml_nbytes(ctx->kv_k);
    const size_t vbytes = ggml_nbytes(ctx->kv_v);
    ggml_backend_t kv_backend = core_attn::kv_backend_from_env(ctx->backend, ctx->backend_cpu, "cosyvoice3_tts");
    ctx->kv_buf = ggml_backend_alloc_buffer(kv_backend, kbytes + vbytes);
    if (!ctx->kv_buf) {
        fprintf(stderr, "cosyvoice3_tts: failed to alloc KV buffer (%zu bytes)\n", kbytes + vbytes);
        return false;
    }
    char* base = (char*)ggml_backend_buffer_get_base(ctx->kv_buf);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + kbytes);
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
    ctx->kv_max_ctx = max_ctx;
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "cosyvoice3_tts: kv cache %d MiB k=%s v=%s (head_dim=%d max_ctx=%d n_kv=%d n_layers=%d)\n",
                (int)((kbytes + vbytes) / 1048576), ggml_type_name(kv_pair.k), ggml_type_name(kv_pair.v), hd, max_ctx,
                n_kv, n_lay);
    }
    return true;
}

// Build the per-step graph (T=1, embeds in, speech-logits out).
//
// Inputs (set externally before compute):
//   "inputs_embeds"   [d_model, T] F32
//   "lm_positions"    [T] I32
//   "lm_causal_mask"  [Lk, T] F16
// Output: "step_logits" [speech_vocab, T] F32 (typically T=1 → 6761)
//
// When fixed_kv_len > 0, positions doubles as kv_indices so the same
// graph topology is reused across steps with different n_past.
ggml_cgraph* cv3_build_lm_graph(cosyvoice3_tts_context* ctx, int n_tokens, int n_past, int fixed_kv_len) {
    const auto& hp = ctx->hp;
    const auto& m = ctx->lm;
    const int n_q = (int)hp.n_heads;
    const int n_kv = (int)hp.n_kv_heads;
    const int hd = (int)hp.head_dim;
    const int d = (int)hp.d_model;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.rms_norm_eps;
    const float theta = hp.rope_theta;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = n_tokens;
    const int Lk = fixed_kv_len > 0 ? fixed_kv_len : (n_past + T);

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_input(embeds);
    ggml_set_name(embeds, "inputs_embeds");

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_input(positions);
    ggml_set_name(positions, "lm_positions");

    // Always declare the causal mask so the cached-graph reuse path
    // (T=1 + fixed_kv_len > 0) keeps the topology invariant across
    // steps. At T=1 with fixed_kv_len=0 we'd normally skip the mask,
    // but matching mimo_asr_build_step_graph here keeps the patterns
    // consistent.
    ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
    ggml_set_input(causal_mask);
    ggml_set_name(causal_mask, "lm_causal_mask");

    GGML_ASSERT(ctx->kv_k && ctx->kv_v);
    GGML_ASSERT(n_past + T <= ctx->kv_max_ctx);
    GGML_ASSERT(Lk <= ctx->kv_max_ctx);

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_kv_grp,
        /*n_ctx_orig*/ (int)hp.max_pos,
        /*rope_theta*/ theta,
        /*rope_beta_fast*/ 0.0f,
        /*rope_beta_slow*/ 0.0f,
        /*attn_scale*/ attn_scale,
        /*qk_norm_eps*/ eps,
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
    };

    ggml_tensor* eff_kv_indices = (fixed_kv_len > 0) ? positions : nullptr;

    ggml_tensor* cur = embeds;
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        const auto& b = m.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* h = ggml_rms_norm(ctx0, cur, eps);
        h = ggml_mul(ctx0, h, b.attn_norm_w);

        ggml_tensor* attn = core_attn::kv_self_attn(
            ctx0, gf, h, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_o_w,
            /*q_norm_w*/ nullptr, /*k_norm_w*/ nullptr, positions, causal_mask, ctx->kv_k, ctx->kv_v, (int)il,
            /*n_past*/ n_past, kvp,
            /*qkv_w*/ nullptr, /*fixed_kv_len*/ fixed_kv_len, /*kv_indices*/ eff_kv_indices, b.attn_q_b, b.attn_k_b,
            b.attn_v_b, /*o_b*/ nullptr, /*qkv_b*/ nullptr);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        h = ggml_rms_norm(ctx0, cur, eps);
        h = ggml_mul(ctx0, h, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, h, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.output_norm_w);

    // For T>1 the AR head only needs the last position. The convention
    // matches qwen3_tts.cpp build_graph_talker_kv: slice at T-1 to keep
    // the head matmul small.
    if (T > 1) {
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }
    ggml_tensor* logits = ggml_mul_mat(ctx0, m.speech_lm_head_w, cur);
    ggml_set_name(logits, "step_logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx0);
    return gf;
}

// Build a one-shot "embedding lookup" graph: ids[N] -> get_rows(table) ->
// [d_model, N] F32. Useful to keep the table on whatever backend the
// weights live on without bouncing them through the CPU.
ggml_cgraph* cv3_build_embed_graph(cosyvoice3_tts_context* ctx, ggml_tensor* table, int n_tokens) {
    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 64, false);

    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(ids);
    ggml_set_name(ids, "embed_ids");

    ggml_tensor* out = ggml_get_rows(ctx0, table, ids);
    // get_rows on F16 weights produces F32 output, which is what we want
    // to hand back to the caller. ggml_cont to materialise a contiguous
    // buffer the backend can dump straight.
    out = ggml_cont(ctx0, out);
    ggml_set_name(out, "embed_out");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_free(ctx0);
    return gf;
}

float* cv3_run_embed(cosyvoice3_tts_context* ctx, ggml_tensor* table, const int32_t* ids, int n_tokens) {
    if (!table || !ids || n_tokens <= 0)
        return nullptr;
    // Building any other graph in `ctx->compute_meta` overwrites the
    // cached step graph's tensor metadata in place, so the next
    // `step_t1_gf` re-use would read garbage. Invalidate the cache.
    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;
    ggml_cgraph* gf = cv3_build_embed_graph(ctx, table, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "cosyvoice3_tts: embed alloc_graph failed\n");
        return nullptr;
    }
    ggml_tensor* ids_t = ggml_graph_get_tensor(gf, "embed_ids");
    ggml_backend_tensor_set(ids_t, ids, 0, (size_t)n_tokens * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cosyvoice3_tts: embed compute failed\n");
        return nullptr;
    }
    ggml_tensor* out_t = ggml_graph_get_tensor(gf, "embed_out");
    if (!out_t)
        return nullptr;
    const size_t n = (size_t)ggml_nelements(out_t);
    float* out = (float*)malloc(n * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(out_t, out, 0, n * sizeof(float));
    return out;
}

std::vector<float> cv3_compute_s3tok_log_mel(const float* pcm_16k, int n_samples, int& T_out) {
    T_out = 0;
    if (!pcm_16k || n_samples <= 0)
        return {};

    constexpr int kSampleRate = 16000;
    constexpr int kNFft = 400;
    constexpr int kHop = 160;
    constexpr int kNMels = 128;

    static thread_local std::vector<float> hann;
    if ((int)hann.size() != kNFft) {
        hann.resize(kNFft);
        for (int i = 0; i < kNFft; i++) {
            hann[(size_t)i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i / (float)kNFft));
        }
    }

    static thread_local std::vector<float> mel_fb;
    if (mel_fb.empty()) {
        mel_fb = core_mel::build_slaney_fb(kSampleRate, kNFft, kNMels, /*fmin*/ 0.0f, /*fmax*/ 8000.0f,
                                           core_mel::FbLayout::MelsFreqs);
    }

    core_mel::Params p;
    p.n_fft = kNFft;
    p.hop_length = kHop;
    p.win_length = kNFft;
    p.n_mels = kNMels;
    p.log_base = core_mel::LogBase::Log10;
    p.log_guard = core_mel::LogGuard::MaxClip;
    p.spec_kind = core_mel::SpecKind::Power;
    p.norm = core_mel::Normalization::GlobalClipMax;
    p.layout = core_mel::Layout::MelsTime;
    p.fb_layout = core_mel::FbLayout::MelsFreqs;
    p.matmul = core_mel::MatmulPrecision::Double;
    p.log_eps = 1e-10f;
    p.center_pad = true;
    p.preemph = 0.0f;
    p.drop_last_frame = true;

    int T = 0;
    auto mel = core_mel::compute(pcm_16k, n_samples, hann.data(), kNFft, mel_fb.data(), kNFft / 2 + 1,
                                 &core_fft::fft_radix2_wrapper, p, T);
    T_out = T;
    return mel;
}

ggml_cgraph* cv3_build_s3tok_graph(cosyvoice3_tts_context* ctx, int T_use) {
    const auto& m = ctx->s3tok;
    const auto& hp = m.hp;
    const int d = (int)hp.n_audio_state;
    const int n_h = (int)hp.n_audio_head;
    const int hd = d / n_h;
    const float attn_scale = 1.0f / std::sqrt((float)hd);

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_use, (int)hp.n_mels);
    ggml_set_input(mel);
    ggml_set_name(mel, "s3tok_mel_in");

    auto bias_1d = [&](ggml_tensor* b) { return ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1); };

    ggml_tensor* x = ggml_conv_1d(ctx0, m.conv0_w, mel, /*s*/ 2, /*p*/ 1, /*d*/ 1);
    x = ggml_add(ctx0, x, bias_1d(m.conv0_b));
    x = ggml_gelu_erf(ctx0, x);
    x = ggml_conv_1d(ctx0, m.conv1_w, x, /*s*/ 2, /*p*/ 1, /*d*/ 1);
    x = ggml_add(ctx0, x, bias_1d(m.conv1_b));
    x = ggml_gelu_erf(ctx0, x);

    const int T_tok = (int)x->ne[0];
    x = ggml_reshape_2d(ctx0, x, T_tok, d);
    x = ggml_cont(ctx0, ggml_transpose(ctx0, x));
    // Diff-harness stage tag: post-subsampler, ne=(d, T_tok) == ONNX (T,1280).
    ggml_set_name(x, "s3tok_subsample");
    ggml_set_output(x);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_tok);
    ggml_set_input(positions);
    ggml_set_name(positions, "positions");

    for (uint32_t il = 0; il < hp.n_audio_layer; il++) {
        const auto& b = m.blocks[il];
        ggml_tensor* residual = x;
        ggml_tensor* h = ggml_norm(ctx0, x, hp.attn_ln_eps);
        h = ggml_mul(ctx0, h, b.attn_ln_w);
        h = ggml_add(ctx0, h, b.attn_ln_b);

        ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_q_w, h), b.attn_q_b);
        ggml_tensor* K = ggml_mul_mat(ctx0, b.attn_k_w, h);
        ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_v_w, h), b.attn_v_b);

        ggml_tensor* v_t = ggml_cont(ctx0, ggml_transpose(ctx0, V));
        ggml_tensor* fsmn = ggml_conv_1d_dw(ctx0, b.fsmn_w, v_t, /*s*/ 1, /*p*/ (int)hp.fsmn_kernel / 2, /*d*/ 1);
        if (ggml_n_dims(fsmn) > 2) {
            fsmn = ggml_reshape_2d(ctx0, fsmn, fsmn->ne[0], fsmn->ne[1]);
        }
        fsmn = ggml_add(ctx0, fsmn, v_t);
        fsmn = ggml_cont(ctx0, ggml_transpose(ctx0, fsmn));

        Q = ggml_reshape_3d(ctx0, Q, hd, n_h, T_tok);
        K = ggml_reshape_3d(ctx0, K, hd, n_h, T_tok);
        V = ggml_reshape_3d(ctx0, V, hd, n_h, T_tok);
        Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f,
                          0.0f, 0.0f);
        K = ggml_rope_ext(ctx0, K, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f,
                          0.0f, 0.0f);
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));
        ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, attn_scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(ctx0, attn, d, T_tok);
        attn = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_o_w, attn), b.attn_o_b);
        attn = ggml_add(ctx0, attn, fsmn);
        x = ggml_add(ctx0, residual, attn);

        residual = x;
        h = ggml_norm(ctx0, x, hp.mlp_ln_eps);
        h = ggml_mul(ctx0, h, b.mlp_ln_w);
        h = ggml_add(ctx0, h, b.mlp_ln_b);
        h = ggml_add(ctx0, ggml_mul_mat(ctx0, b.mlp_up_w, h), b.mlp_up_b);
        h = ggml_gelu_erf(ctx0, h);
        h = ggml_add(ctx0, ggml_mul_mat(ctx0, b.mlp_dn_w, h), b.mlp_dn_b);
        x = ggml_add(ctx0, residual, h);
        // Diff-harness stage tag: per-block output, ne=(d, T_tok) == ONNX (T,1280).
        char blk_name[32];
        std::snprintf(blk_name, sizeof(blk_name), "s3tok_blk_%u", il);
        ggml_set_name(x, blk_name);
        ggml_set_output(x);
    }

    ggml_tensor* proj = ggml_add(ctx0, ggml_mul_mat(ctx0, m.fsq_proj_w, x), m.fsq_proj_b);
    ggml_set_name(proj, "s3tok_proj_down");
    ggml_set_output(proj);
    ggml_build_forward_expand(gf, proj);
    ggml_free(ctx0);
    return gf;
}

std::vector<int32_t> cv3_tokenize_s3tok(cosyvoice3_tts_context* ctx, const float* pcm_16k, int n_samples,
                                        int max_tokens) {
    std::vector<int32_t> out;
    if (!ctx || !ctx->s3tok.loaded || !pcm_16k || n_samples <= 0)
        return out;
    int T = 0;
    auto mel = cv3_compute_s3tok_log_mel(pcm_16k, n_samples, T);
    if (mel.empty() || T <= 0)
        return out;
    int T_use = T;
    if (max_tokens > 0 && max_tokens * 4 < T_use)
        T_use = max_tokens * 4;

    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;
    ggml_cgraph* gf = cv3_build_s3tok_graph(ctx, T_use);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "cosyvoice3_tts: s3tok alloc_graph failed\n");
        return out;
    }
    ggml_tensor* mel_t = ggml_graph_get_tensor(gf, "s3tok_mel_in");
    ggml_backend_tensor_set(mel_t, mel.data(), 0, (size_t)T_use * 128 * sizeof(float));
    if (ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "positions")) {
        std::vector<int32_t> pos((size_t)pos_t->ne[0]);
        for (int i = 0; i < (int)pos.size(); i++)
            pos[(size_t)i] = i;
        ggml_backend_tensor_set(pos_t, pos.data(), 0, pos.size() * sizeof(int32_t));
    }
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cosyvoice3_tts: s3tok graph compute failed\n");
        return out;
    }
    ggml_tensor* proj_t = ggml_graph_get_tensor(gf, "s3tok_proj_down");
    if (!proj_t)
        return out;
    std::vector<float> proj((size_t)ggml_nelements(proj_t));
    ggml_backend_tensor_get(proj_t, proj.data(), 0, proj.size() * sizeof(float));
    const int T_tok = (int)proj_t->ne[1];
    out.resize((size_t)T_tok);
    constexpr float kFsqGain = 0.9990000128746033f;
    constexpr int kPowers[8] = {1, 3, 9, 27, 81, 243, 729, 2187};
    for (int t = 0; t < T_tok; t++) {
        const float* row = proj.data() + (size_t)t * 8;
        int code = 0;
        for (int i = 0; i < 8; i++) {
            float h = std::tanh(row[i]) * kFsqGain;
            int v = (int)std::nearbyint(h) + 1;
            if (v < 0)
                v = 0;
            if (v > 2)
                v = 2;
            code += v * kPowers[i];
        }
        out[(size_t)t] = code;
    }
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

extern "C" struct cosyvoice3_tts_context_params cosyvoice3_tts_context_default_params(void) {
    cosyvoice3_tts_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.flash_attn = false;
    p.temperature = 0.0f;
    p.seed = 0; // 0 -> use default 42
    p.max_tokens = 0;
    p.ras_top_k = 25;
    p.ras_top_p = 0.8f;
    p.ras_win_size = 10;
    p.ras_tau_r = 0.1f;
    return p;
}

extern "C" struct cosyvoice3_tts_context* cosyvoice3_tts_init_from_file(const char* path_model,
                                                                        struct cosyvoice3_tts_context_params params) {
    auto* ctx = new cosyvoice3_tts_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx->seed = params.seed ? params.seed : 42;
    ctx->rng.seed(ctx->seed);

    // ---- Metadata pass ----
    ggml_context* gctx_dummy = nullptr;
    gguf_init_params gp = {/*no_alloc=*/true, &gctx_dummy};
    gguf_context* gctx = gguf_init_from_file(path_model, gp);
    if (!gctx) {
        fprintf(stderr, "cosyvoice3_tts: failed to read GGUF '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }

    auto& hp = ctx->hp;
    hp.n_layers = cv3_kv_u32(gctx, "cosyvoice3.llm.n_layers", hp.n_layers);
    hp.d_model = cv3_kv_u32(gctx, "cosyvoice3.llm.d_model", hp.d_model);
    hp.n_heads = cv3_kv_u32(gctx, "cosyvoice3.llm.n_heads", hp.n_heads);
    hp.n_kv_heads = cv3_kv_u32(gctx, "cosyvoice3.llm.n_kv_heads", hp.n_kv_heads);
    hp.head_dim = cv3_kv_u32(gctx, "cosyvoice3.llm.head_dim", hp.head_dim);
    hp.ff_dim = cv3_kv_u32(gctx, "cosyvoice3.llm.ff_dim", hp.ff_dim);
    hp.rope_theta = cv3_kv_f32(gctx, "cosyvoice3.llm.rope_theta", hp.rope_theta);
    hp.rms_norm_eps = cv3_kv_f32(gctx, "cosyvoice3.llm.rms_norm_eps", hp.rms_norm_eps);
    hp.text_vocab = cv3_kv_u32(gctx, "cosyvoice3.llm.vocab_size", hp.text_vocab);
    hp.max_pos = cv3_kv_u32(gctx, "cosyvoice3.llm.max_pos", hp.max_pos);
    hp.speech_vocab = cv3_kv_u32(gctx, "cosyvoice3.llm.speech_vocab_size", hp.speech_vocab);
    hp.speech_codebook = cv3_kv_u32(gctx, "cosyvoice3.llm.speech_token_codebook", hp.speech_codebook);

    // BPE vocab. The CV3 LLM converter writes the Qwen2 vocab.json
    // verbatim (gpt2-byte-encoded), but vocab.json caps out at id
    // 151642. The CV3 chat-style prompt depends on a handful of
    // special tokens added on top of that — register them here so
    // `<|endofprompt|>` etc. tokenise to the correct ids.
    auto tok = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
    if (!tok.empty()) {
        ctx->vocab.id_to_token = std::move(tok);
        ctx->vocab.token_to_id.reserve(ctx->vocab.id_to_token.size() + 8);
        for (int i = 0; i < (int)ctx->vocab.id_to_token.size(); i++) {
            ctx->vocab.token_to_id[ctx->vocab.id_to_token[i]] = i;
        }
    }
    {
        // CV3 added_special_tokens (from upstream tokenizer.py
        // CosyVoice3Tokenizer): ids 151643..151645 are the universal
        // chat markers, 151646 is `<|endofprompt|>` (asserted on by
        // upstream `Qwen2LM.inference`). We only need to register the
        // ones referenced by our prompt_text — keeping the table
        // small avoids accidentally shadowing trained vocab entries.
        const struct {
            int id;
            const char* text;
        } specials[] = {
            {151643, "<|endoftext|>"},
            {151644, "<|im_start|>"},
            {151645, "<|im_end|>"},
            {151646, "<|endofprompt|>"},
        };
        int max_id = 151646;
        if ((int)ctx->vocab.id_to_token.size() <= max_id) {
            ctx->vocab.id_to_token.resize((size_t)max_id + 1);
        }
        for (const auto& sp : specials) {
            ctx->vocab.id_to_token[sp.id] = sp.text;
            ctx->vocab.token_to_id[sp.text] = sp.id;
        }
    }
    auto merges = core_gguf::kv_str_array(gctx, "tokenizer.ggml.merges");
    for (size_t i = 0; i < merges.size(); i++) {
        ctx->vocab.merge_rank[merges[i]] = (int32_t)i;
    }

    gguf_free(gctx);

    // ---- Backend init ----
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "cosyvoice3_tts: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    // The original diff-validation implementation pinned every CosyVoice3
    // stage to CPU even when the caller requested GPU execution.  Besides the
    // AR LLM, the flow estimator runs 22 transformer blocks twice for every
    // Euler step, so that stale pin makes normal synthesis appear hung on
    // CUDA machines.  All stages use the shared backend scheduler below and
    // can therefore follow their GPU-resident weights, with CPU retained as
    // the fallback for unsupported operations.
    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ctx->backend_cpu;
    if (!ctx->backend) {
        if (params.use_gpu && params.verbosity >= 1) {
            fprintf(stderr, "cosyvoice3_tts: GPU backend unavailable, falling back to CPU\n");
        }
        ctx->backend = ctx->backend_cpu;
    }
    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    }
    if (params.verbosity >= 1) {
        fprintf(stderr, "cosyvoice3_tts: using %s backend: %s\n", ggml_backend_is_cpu(ctx->backend) ? "CPU" : "GPU",
                ggml_backend_name(ctx->backend));
    }

    // ---- Weight pass ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "cosyvoice3_tts", wl)) {
        fprintf(stderr, "cosyvoice3_tts: failed to load weights from '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;
    ctx->buf_w_cpu = wl.buf_cpu;
    ctx->tensors = std::move(wl.tensors);

    // ---- Tensor binding ----
    auto require_t = [&](const std::string& name) -> ggml_tensor* {
        return core_gguf::require(ctx->tensors, name.c_str(), "cosyvoice3_tts");
    };
    auto try_t = [&](const std::string& name) -> ggml_tensor* {
        return core_gguf::try_get(ctx->tensors, name.c_str());
    };

    auto& m = ctx->lm;
    m.token_embd_w = require_t("token_embd.weight");
    m.output_norm_w = require_t("output_norm.weight");
    m.text_output_w = try_t("output.weight"); // optional; unused in speech AR
    m.speech_embd_w = require_t("cosyvoice3.speech_embd.weight");
    m.speech_lm_head_w = require_t("cosyvoice3.speech_lm_head.weight");
    if (!m.token_embd_w || !m.output_norm_w || !m.speech_embd_w || !m.speech_lm_head_w) {
        fprintf(stderr, "cosyvoice3_tts: missing core LLM tensors\n");
        delete ctx;
        return nullptr;
    }

    m.blocks.resize(hp.n_layers);
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        char prefix[32];
        snprintf(prefix, sizeof(prefix), "blk.%u", il);
        auto& b = m.blocks[il];
        std::string p = prefix;
        b.attn_norm_w = require_t(p + ".attn_norm.weight");
        b.attn_q_w = require_t(p + ".attn_q.weight");
        b.attn_q_b = require_t(p + ".attn_q.bias");
        b.attn_k_w = require_t(p + ".attn_k.weight");
        b.attn_k_b = require_t(p + ".attn_k.bias");
        b.attn_v_w = require_t(p + ".attn_v.weight");
        b.attn_v_b = require_t(p + ".attn_v.bias");
        b.attn_o_w = require_t(p + ".attn_output.weight");
        b.ffn_norm_w = require_t(p + ".ffn_norm.weight");
        b.ffn_gate_w = require_t(p + ".ffn_gate.weight");
        b.ffn_up_w = require_t(p + ".ffn_up.weight");
        b.ffn_down_w = require_t(p + ".ffn_down.weight");
        if (!b.attn_norm_w || !b.attn_q_w || !b.attn_q_b || !b.attn_k_w || !b.attn_k_b || !b.attn_v_w || !b.attn_v_b ||
            !b.attn_o_w || !b.ffn_norm_w || !b.ffn_gate_w || !b.ffn_up_w || !b.ffn_down_w) {
            fprintf(stderr, "cosyvoice3_tts: missing tensors in %s.*\n", prefix);
            delete ctx;
            return nullptr;
        }
    }

    // ---- Scheduler + compute_meta ----
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    if (params.verbosity >= 1) {
        fprintf(stderr,
                "cosyvoice3_tts: loaded %zu tensors  llm=%uL d=%u h=%u/kv=%u hd=%u ff=%u "
                "text_vocab=%u speech_vocab=%u (codebook=%u)\n",
                ctx->tensors.size(), hp.n_layers, hp.d_model, hp.n_heads, hp.n_kv_heads, hp.head_dim, hp.ff_dim,
                hp.text_vocab, hp.speech_vocab, hp.speech_codebook);
    }

    // §192 Pre-dequantize speech embedding table to CPU F32. The table is
    // [d_model, speech_vocab] = 896×6761 ≈ 24 MB in F32; negligible RAM
    // cost but saves a GPU round-trip + graph-cache invalidation per AR step.
    {
        ggml_tensor* w = ctx->lm.speech_embd_w;
        const int d = (int)hp.d_model;
        const int v = (int)hp.speech_vocab;
        ctx->speech_embd_cache.resize((size_t)v * d);
        if (w->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(w, ctx->speech_embd_cache.data(), 0, (size_t)v * d * sizeof(float));
        } else {
            const size_t nbytes = ggml_nbytes(w);
            std::vector<uint8_t> raw(nbytes);
            ggml_backend_tensor_get(w, raw.data(), 0, nbytes);
            ggml_get_type_traits(w->type)->to_float(raw.data(), ctx->speech_embd_cache.data(), (size_t)v * d);
        }
    }

    return ctx;
}

extern "C" void cosyvoice3_tts_free(struct cosyvoice3_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->buf_w)
        ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->buf_w_cpu)
        ggml_backend_buffer_free(ctx->buf_w_cpu);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->flow.buf_w)
        ggml_backend_buffer_free(ctx->flow.buf_w);
    if (ctx->flow.ctx_w)
        ggml_free(ctx->flow.ctx_w);
    if (ctx->hift.buf_w)
        ggml_backend_buffer_free(ctx->hift.buf_w);
    if (ctx->hift.ctx_w)
        ggml_free(ctx->hift.ctx_w);
    if (ctx->s3tok.buf_w)
        ggml_backend_buffer_free(ctx->s3tok.buf_w);
    if (ctx->s3tok.ctx_w)
        ggml_free(ctx->s3tok.ctx_w);
    if (ctx->campplus.buf_w)
        ggml_backend_buffer_free(ctx->campplus.buf_w);
    if (ctx->campplus.ctx_w)
        ggml_free(ctx->campplus.ctx_w);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" void cosyvoice3_tts_set_n_threads(struct cosyvoice3_tts_context* ctx, int n_threads) {
    if (!ctx)
        return;
    ctx->n_threads = n_threads > 0 ? n_threads : 4;
    if (ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
}

extern "C" void cosyvoice3_tts_set_seed(struct cosyvoice3_tts_context* ctx, uint64_t seed) {
    if (!ctx)
        return;
    ctx->seed = seed ? seed : 42;
    ctx->rng.seed(ctx->seed);
}

extern "C" void cosyvoice3_tts_set_temperature(struct cosyvoice3_tts_context* ctx, float temperature) {
    if (!ctx)
        return;
    ctx->params.temperature = temperature;
}

extern "C" int cosyvoice3_tts_get_hparams(struct cosyvoice3_tts_context* ctx, uint32_t* d_model, uint32_t* n_layers,
                                          uint32_t* n_heads, uint32_t* n_kv_heads, uint32_t* head_dim,
                                          uint32_t* text_vocab, uint32_t* speech_vocab, uint32_t* speech_codebook) {
    if (!ctx)
        return -1;
    const auto& hp = ctx->hp;
    if (d_model)
        *d_model = hp.d_model;
    if (n_layers)
        *n_layers = hp.n_layers;
    if (n_heads)
        *n_heads = hp.n_heads;
    if (n_kv_heads)
        *n_kv_heads = hp.n_kv_heads;
    if (head_dim)
        *head_dim = hp.head_dim;
    if (text_vocab)
        *text_vocab = hp.text_vocab;
    if (speech_vocab)
        *speech_vocab = hp.speech_vocab;
    if (speech_codebook)
        *speech_codebook = hp.speech_codebook;
    return 0;
}

extern "C" void cosyvoice3_tts_reset_kv(struct cosyvoice3_tts_context* ctx) {
    if (!ctx || !ctx->kv_buf)
        return;
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;
}

extern "C" float* cosyvoice3_tts_embed_text(struct cosyvoice3_tts_context* ctx, const int32_t* ids, int n_tokens) {
    if (!ctx || !ids || n_tokens <= 0)
        return nullptr;
    return cv3_run_embed(ctx, ctx->lm.token_embd_w, ids, n_tokens);
}

extern "C" float* cosyvoice3_tts_embed_speech(struct cosyvoice3_tts_context* ctx, const int32_t* ids, int n_tokens) {
    if (!ctx || !ids || n_tokens <= 0)
        return nullptr;
    return cv3_run_embed(ctx, ctx->lm.speech_embd_w, ids, n_tokens);
}

extern "C" float* cosyvoice3_tts_prefill_with_embeds(struct cosyvoice3_tts_context* ctx, const float* embeds,
                                                     int n_tokens, int n_past) {
    if (!ctx || !embeds || n_tokens <= 0 || n_past < 0)
        return nullptr;
    const auto& hp = ctx->hp;
    const int d = (int)hp.d_model;
    // Standalone prefill callers need some decode headroom, but allocating a
    // fixed 4096-token cache makes the cached T=1 Metal graph attend over all
    // 4096 slots on every generated token.  The generation entry points size
    // the cache to their known token budget before reaching this fallback.
    const int prefill_capacity = ctx->kv_max_ctx > 0 ? ctx->kv_max_ctx : std::max(n_past + n_tokens + 256, 512);
    if (!cv3_kv_init(ctx, std::max(n_past + n_tokens, prefill_capacity)))
        return nullptr;

    // Invalidate any cached step graph — prefill builds in the same
    // shared compute_meta and clobbers the step graph's metadata.
    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;

    ggml_cgraph* gf = cv3_build_lm_graph(ctx, n_tokens, n_past, /*fixed_kv_len*/ 0);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "cosyvoice3_tts: prefill alloc_graph failed\n");
        return nullptr;
    }

    auto set_t = [&](const char* nm, const void* data, size_t bytes) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
        if (!t)
            return false;
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };

    if (!set_t("inputs_embeds", embeds, (size_t)d * n_tokens * sizeof(float)))
        return nullptr;

    std::vector<int32_t> pos((size_t)n_tokens);
    for (int i = 0; i < n_tokens; i++)
        pos[i] = n_past + i;
    if (!set_t("lm_positions", pos.data(), pos.size() * sizeof(int32_t)))
        return nullptr;

    const int Lk = n_past + n_tokens;
    std::vector<ggml_fp16_t> mask((size_t)n_tokens * Lk);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < n_tokens; q++)
        for (int k = 0; k < Lk; k++)
            mask[(size_t)q * Lk + k] = (k <= n_past + q) ? z : ninf;
    if (!set_t("lm_causal_mask", mask.data(), mask.size() * sizeof(ggml_fp16_t)))
        return nullptr;

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cosyvoice3_tts: prefill compute failed\n");
        return nullptr;
    }

    ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "step_logits");
    if (!logits_t)
        return nullptr;
    const size_t n_log = (size_t)ggml_nelements(logits_t);
    float* out = (float*)malloc(n_log * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(logits_t, out, 0, n_log * sizeof(float));
    return out;
}

extern "C" float* cosyvoice3_tts_step_speech(struct cosyvoice3_tts_context* ctx, int32_t speech_id, int n_past) {
    if (!ctx || speech_id < 0 || n_past < 0)
        return nullptr;
    const auto& hp = ctx->hp;
    const int d = (int)hp.d_model;
    if ((uint32_t)speech_id >= hp.speech_vocab) {
        fprintf(stderr, "cosyvoice3_tts: speech_id %d out of range [0, %u)\n", speech_id, hp.speech_vocab);
        return nullptr;
    }
    // Generation pre-sizes the cache to its full token budget.  Keep a
    // modest fallback for direct step callers without inflating an existing
    // request-sized cache to 4096 slots.
    const int step_capacity = ctx->kv_max_ctx > 0 ? ctx->kv_max_ctx : 512;
    if (!cv3_kv_init(ctx, std::max(n_past + 1, step_capacity)))
        return nullptr;
    if (n_past + 1 > ctx->kv_max_ctx) {
        fprintf(stderr, "cosyvoice3_tts: kv overflow (%d+1 > %d)\n", n_past, ctx->kv_max_ctx);
        return nullptr;
    }

    // Step 1: look up speech_embd[speech_id]. §192: use the CPU F32 cache
    // when available — avoids a GPU round-trip and preserves step_t1_gf.
    float* embed_alloc = nullptr;
    const float* embed;
    if (!ctx->speech_embd_cache.empty()) {
        embed = ctx->speech_embd_cache.data() + (size_t)speech_id * d;
    } else {
        embed_alloc = cv3_run_embed(ctx, ctx->lm.speech_embd_w, &speech_id, 1);
        if (!embed_alloc)
            return nullptr;
        embed = embed_alloc;
    }

    // Step 2: run the LM forward with that embedding. Cache the T=1 graph
    // topology, but always reset + allocate the scheduler before execution.
    // Skipping scheduler allocation for a cached graph leaves stale compute
    // buffer bindings on GPU backends; the second AR step then crashes while
    // encoding the first normalization op (issue #194). This mirrors the
    // VibeVoice bucket-cache fix: graph construction stays amortized while
    // the inexpensive scheduler split/allocation pass is refreshed.
    const int fixed_kv = ctx->kv_max_ctx;
    const bool can_skip = (ctx->step_t1_gf != nullptr && ctx->step_t1_fixed_kv_len == fixed_kv);

    ggml_cgraph* gf;
    if (can_skip) {
        gf = ctx->step_t1_gf;
    } else {
        gf = cv3_build_lm_graph(ctx, /*n_tokens*/ 1, /*n_past*/ 0, /*fixed_kv_len*/ fixed_kv);
        if (!gf) {
            free(embed_alloc);
            return nullptr;
        }
        ctx->step_t1_gf = gf;
        ctx->step_t1_fixed_kv_len = fixed_kv;
    }

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "cosyvoice3_tts: step alloc_graph failed\n");
        free(embed_alloc);
        return nullptr;
    }

    auto set_t = [&](const char* nm, const void* data, size_t bytes) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
        if (!t)
            return false;
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };

    if (!set_t("inputs_embeds", embed, (size_t)d * sizeof(float))) {
        free(embed_alloc);
        return nullptr;
    }
    free(embed_alloc);

    int32_t pos = n_past;
    if (!set_t("lm_positions", &pos, sizeof(pos)))
        return nullptr;

    std::vector<ggml_fp16_t> mask((size_t)fixed_kv);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int k = 0; k < fixed_kv; k++)
        mask[k] = (k <= n_past) ? z : ninf;
    if (!set_t("lm_causal_mask", mask.data(), mask.size() * sizeof(ggml_fp16_t)))
        return nullptr;

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cosyvoice3_tts: step compute failed\n");
        return nullptr;
    }

    ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "step_logits");
    if (!logits_t)
        return nullptr;
    const size_t n_log = (size_t)ggml_nelements(logits_t);
    float* out = (float*)malloc(n_log * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(logits_t, out, 0, n_log * sizeof(float));
    return out;
}

// ---------------------------------------------------------------------------
// Sampling — Repetition-Aware Sampling (RAS), ported from upstream
// CosyVoice/cosyvoice/utils/common.py::ras_sampling.
//
// nucleus_sampling: softmax(logits) → stable-sort descending → take
//   while cum_prob < top_p AND count < top_k → multinomial-sample over
//   the kept (non-renormalised) probabilities.
//
// ras_sampling: nucleus sample → if the picked token appears in
//   decoded_history[-win_size:] ≥ win_size·tau_r times, suppress it
//   (logits[id] = -INF) and re-sample via plain softmax-multinomial
//   over the modified logits.
// ---------------------------------------------------------------------------

namespace {

// Stable-softmax: subtract max to avoid overflow.
void softmax_inplace(std::vector<float>& v) {
    float vmax = v.empty() ? 0.0f : v[0];
    for (float x : v)
        if (x > vmax)
            vmax = x;
    double s = 0.0;
    for (auto& x : v) {
        x = std::exp(x - vmax);
        s += x;
    }
    if (s > 0.0) {
        const float inv = (float)(1.0 / s);
        for (auto& x : v)
            x *= inv;
    }
}

// Multinomial sample given an unnormalised positive weight vector. Mirrors
// torch.multinomial(weights, 1, replacement=True): treat weights / sum as
// the categorical distribution, draw one via inverse-CDF. Returns -1 if
// weights are all zero / NaN / negative.
int32_t multinomial_pick(const std::vector<float>& weights, std::mt19937_64& rng) {
    double sum = 0.0;
    for (float w : weights) {
        if (!(w > 0.0f) || std::isnan(w))
            continue;
        sum += w;
    }
    if (!(sum > 0.0))
        return -1;
    std::uniform_real_distribution<double> U(0.0, sum);
    double r = U(rng);
    double acc = 0.0;
    for (size_t i = 0; i < weights.size(); i++) {
        if (!(weights[i] > 0.0f) || std::isnan(weights[i]))
            continue;
        acc += weights[i];
        if (r <= acc)
            return (int32_t)i;
    }
    return (int32_t)weights.size() - 1; // floating-point tail
}

// Upstream `nucleus_sampling`: select top tokens until cum_prob >= top_p
// or count >= top_k (whichever comes first), then multinomial-sample
// over the kept (unrenormalised) probabilities.
int32_t nucleus_sample(const float* logits, int n_vocab, float top_p, int top_k, std::mt19937_64& rng) {
    std::vector<float> probs((size_t)n_vocab);
    std::memcpy(probs.data(), logits, (size_t)n_vocab * sizeof(float));
    softmax_inplace(probs);
    // Stable-sort indices by descending prob. std::stable_sort matches
    // PyTorch's `sort(stable=True)` ordering for ties.
    std::vector<int32_t> idx((size_t)n_vocab);
    for (int i = 0; i < n_vocab; i++)
        idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) { return probs[a] > probs[b]; });
    std::vector<float> kept;
    std::vector<int32_t> kept_ids;
    double cum = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        // Upstream guard: stop when cum_prob >= top_p OR count >= top_k.
        if (cum >= (double)top_p || (int)kept.size() >= top_k)
            break;
        const float p = probs[idx[i]];
        cum += (double)p;
        kept.push_back(p);
        kept_ids.push_back(idx[i]);
    }
    if (kept.empty())
        return -1;
    int32_t pick = multinomial_pick(kept, rng);
    if (pick < 0)
        return -1;
    return kept_ids[(size_t)pick];
}

} // namespace

extern "C" int32_t cosyvoice3_tts_sample_ras(struct cosyvoice3_tts_context* ctx, const float* logits,
                                             const int32_t* decoded_history, int n_history) {
    if (!ctx || !logits)
        return -1;
    const auto& hp = ctx->hp;
    const int n_vocab = (int)hp.speech_vocab;
    const auto& sp = ctx->params;
    const float top_p = sp.ras_top_p > 0.0f ? sp.ras_top_p : 0.8f;
    const int top_k = sp.ras_top_k > 0 ? sp.ras_top_k : 25;
    const int win_size = sp.ras_win_size > 0 ? sp.ras_win_size : 10;
    const float tau_r = sp.ras_tau_r > 0.0f ? sp.ras_tau_r : 0.1f;

    int32_t pick = nucleus_sample(logits, n_vocab, top_p, top_k, ctx->rng);
    if (pick < 0)
        return -1;

    if (!decoded_history || n_history <= 0)
        return pick;

    // Repetition check over the trailing `win_size` of decoded_history.
    // `pick` is counted in the suffix; if it appears ≥ win_size·tau_r
    // times, suppress and re-sample via plain softmax-multinomial over
    // the FULL distribution (matches upstream `random_sampling`).
    int rep = 0;
    const int start = std::max(0, n_history - win_size);
    for (int i = start; i < n_history; i++) {
        if (decoded_history[i] == pick)
            rep++;
    }
    const float thresh = (float)win_size * tau_r;
    if ((float)rep >= thresh) {
        std::vector<float> mod((size_t)n_vocab);
        std::memcpy(mod.data(), logits, (size_t)n_vocab * sizeof(float));
        mod[(size_t)pick] = -INFINITY;
        softmax_inplace(mod);
        pick = multinomial_pick(mod, ctx->rng);
    }
    return pick;
}

extern "C" int32_t* cosyvoice3_tts_generate_tokens_from_embeds(struct cosyvoice3_tts_context* ctx, const float* embeds,
                                                               int n_tokens, int max_tokens, int stop_token_id,
                                                               int* out_n) {
    if (!ctx || !embeds || n_tokens <= 0 || !out_n)
        return nullptr;
    *out_n = 0;
    const auto& hp = ctx->hp;
    const int speech_vocab = (int)hp.speech_vocab;
    const int max_steps = max_tokens > 0 ? max_tokens : (ctx->params.max_tokens > 0 ? ctx->params.max_tokens : 1500);

    if (!cv3_kv_init(ctx, n_tokens + max_steps))
        return nullptr;
    cosyvoice3_tts_reset_kv(ctx);
    float* logits = cosyvoice3_tts_prefill_with_embeds(ctx, embeds, n_tokens, /*n_past*/ 0);
    if (!logits)
        return nullptr;

    std::vector<int32_t> out;
    out.reserve((size_t)max_steps);
    const bool greedy = !(ctx->params.temperature > 0.0f);

    int n_past = n_tokens;
    for (int step = 0; step < max_steps; step++) {
        int32_t pick;
        if (greedy) {
            // Greedy argmax. Restrict to the codebook range so we never
            // emit special-token rows that lie past index speech_codebook.
            int n_pick_range = (int)hp.speech_codebook > 0 ? (int)hp.speech_codebook : speech_vocab;
            float bv = logits[0];
            pick = 0;
            for (int i = 1; i < n_pick_range; i++)
                if (logits[i] > bv) {
                    bv = logits[i];
                    pick = i;
                }
        } else {
            pick = cosyvoice3_tts_sample_ras(ctx, logits, out.empty() ? nullptr : out.data(), (int)out.size());
            if (pick < 0) {
                free(logits);
                *out_n = 0;
                return nullptr;
            }
        }
        free(logits);
        if (stop_token_id >= 0 && pick == stop_token_id)
            break;
        out.push_back(pick);
        if (n_past + 1 > ctx->kv_max_ctx) {
            fprintf(stderr, "cosyvoice3_tts: generate_tokens: kv overflow at step %d\n", step);
            break;
        }
        logits = cosyvoice3_tts_step_speech(ctx, pick, n_past);
        if (!logits) {
            fprintf(stderr, "cosyvoice3_tts: generate_tokens: step %d failed\n", step);
            *out_n = (int)out.size();
            int32_t* dup = (int32_t*)malloc(out.size() * sizeof(int32_t));
            if (!dup)
                return nullptr;
            std::memcpy(dup, out.data(), out.size() * sizeof(int32_t));
            return dup;
        }
        n_past++;
    }
    // The trailing logits buffer is freed inside the loop on success;
    // we only get here after the break, where logits is already freed.

    *out_n = (int)out.size();
    if (out.empty())
        return (int32_t*)malloc(0); // benign 0-length pointer
    int32_t* arr = (int32_t*)malloc(out.size() * sizeof(int32_t));
    if (!arr)
        return nullptr;
    std::memcpy(arr, out.data(), out.size() * sizeof(int32_t));
    return arr;
}

namespace {
// Forward declarations; definitions live further down in this file
// (alongside the per-graph builders they depend on).
float* cv3_extract_flow_dit_stage(cosyvoice3_tts_context* ctx, int block_idx, const float* x, int T, const float* t_emb,
                                  const char* tensor_name);
float* cv3_extract_pre_la_stage(cosyvoice3_tts_context* ctx, const int32_t* ids, int T_tok, const char* tensor_name);
float* cv3_extract_in_pipe_stage(cosyvoice3_tts_context* ctx, const float* pre_la_out, int T_mel, const float* spk_emb,
                                 const float* x_noisy, const float* cond, const char* tensor_name);
float* cv3_extract_dit_full_stage(cosyvoice3_tts_context* ctx, const float* x, int T_mel, const float* t_emb,
                                  const char* tensor_name);
float* cv3_extract_euler_stage(cosyvoice3_tts_context* ctx, const float* mu, int T_mel, const float* spks_proj,
                               const float* cond, const float* x_init, int n_steps, float cfg_rate,
                               const char* tensor_name);
float* cv3_run_solve_euler(cosyvoice3_tts_context* ctx, const float* mu, int T_mel, const float* spks_proj,
                           const float* cond, const float* x_init, int n_steps, float cfg_rate, float* dphi_step0_out);
float* cv3_extract_hift_f0_stage(cosyvoice3_tts_context* ctx, const float* mel, int T_mel);
float* cv3_extract_hift_decode_stage(cosyvoice3_tts_context* ctx, const float* mel, int T_mel, const float* s_stft_in,
                                     const char* stage_name, int* out_n);
float* cv3_extract_hift_source_stage(cosyvoice3_tts_context* ctx, const float* f0_mel, int T_mel,
                                     const float* noise_buf, const char* stage_name, int* out_n);
float* cv3_extract_hift_inference(cosyvoice3_tts_context* ctx, const float* mel, int T_mel, const float* noise_buf,
                                  int* out_n);
} // namespace

extern "C" float* cosyvoice3_tts_extract_stage(struct cosyvoice3_tts_context* ctx, const char* stage_name,
                                               const int32_t* ids, int n_ids, const float* embeds_in,
                                               int n_embed_tokens, int* out_n) {
    if (!ctx || !stage_name || !out_n)
        return nullptr;
    *out_n = 0;
    const auto& hp = ctx->hp;
    // ---- Phase 6 s3tokenizer_v3 per-stage diff ----
    // Input mel rides in `embeds_in` as T_use*128 floats in the same
    // channel-major layout cv3_compute_s3tok_log_mel produces (ggml
    // ne=(T_use,128)); n_embed_tokens = T_use (mel frames). Stages:
    //   s3tok_subsample / s3tok_blk_<K> / s3tok_proj  -> named tensor
    //   s3tok_tokens                                   -> FSQ-quantised ids
    if (strncmp(stage_name, "s3tok_", 6) == 0) {
        if (!ctx->s3tok.loaded) {
            fprintf(stderr, "cosyvoice3_tts: %s requested but s3tok not loaded\n", stage_name);
            return nullptr;
        }
        if (!embeds_in || n_embed_tokens <= 0)
            return nullptr;
        const int T_use = n_embed_tokens;
        ggml_cgraph* gf = cv3_build_s3tok_graph(ctx, T_use);
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            fprintf(stderr, "cosyvoice3_tts: %s alloc_graph failed\n", stage_name);
            return nullptr;
        }
        ggml_tensor* mel_t = ggml_graph_get_tensor(gf, "s3tok_mel_in");
        if (!mel_t)
            return nullptr;
        ggml_backend_tensor_set(mel_t, embeds_in, 0, (size_t)T_use * 128 * sizeof(float));
        if (ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "positions")) {
            std::vector<int32_t> pos((size_t)pos_t->ne[0]);
            for (int i = 0; i < (int)pos.size(); i++)
                pos[(size_t)i] = i;
            ggml_backend_tensor_set(pos_t, pos.data(), 0, pos.size() * sizeof(int32_t));
        }
        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "cosyvoice3_tts: %s compute failed\n", stage_name);
            return nullptr;
        }
        if (strcmp(stage_name, "s3tok_tokens") == 0) {
            ggml_tensor* proj_t = ggml_graph_get_tensor(gf, "s3tok_proj_down");
            if (!proj_t)
                return nullptr;
            const int T_tok = (int)proj_t->ne[1];
            std::vector<float> proj((size_t)ggml_nelements(proj_t));
            ggml_backend_tensor_get(proj_t, proj.data(), 0, proj.size() * sizeof(float));
            float* out = (float*)malloc((size_t)T_tok * sizeof(float));
            if (!out)
                return nullptr;
            constexpr float kFsqGain = 0.9990000128746033f;
            constexpr int kPowers[8] = {1, 3, 9, 27, 81, 243, 729, 2187};
            for (int t = 0; t < T_tok; t++) {
                const float* row = proj.data() + (size_t)t * 8;
                int code = 0;
                for (int i = 0; i < 8; i++) {
                    int v = (int)std::nearbyint(std::tanh(row[i]) * kFsqGain) + 1;
                    v = v < 0 ? 0 : (v > 2 ? 2 : v);
                    code += v * kPowers[i];
                }
                out[(size_t)t] = (float)code;
            }
            *out_n = T_tok;
            return out;
        }
        const char* tname = (strcmp(stage_name, "s3tok_proj") == 0) ? "s3tok_proj_down" : stage_name;
        ggml_tensor* t = ggml_graph_get_tensor(gf, tname);
        if (!t) {
            fprintf(stderr, "cosyvoice3_tts: %s: no such stage tensor\n", stage_name);
            return nullptr;
        }
        const size_t n = (size_t)ggml_nelements(t);
        float* out = (float*)malloc(n * sizeof(float));
        if (!out)
            return nullptr;
        ggml_backend_tensor_get(t, out, 0, n * sizeof(float));
        *out_n = (int)n;
        return out;
    }
    if (strcmp(stage_name, "lm_token_embd") == 0) {
        if (!ids || n_ids <= 0)
            return nullptr;
        float* out = cosyvoice3_tts_embed_text(ctx, ids, n_ids);
        if (!out)
            return nullptr;
        *out_n = n_ids * (int)hp.d_model;
        return out;
    }
    if (strcmp(stage_name, "lm_speech_embd") == 0) {
        if (!ids || n_ids <= 0)
            return nullptr;
        float* out = cosyvoice3_tts_embed_speech(ctx, ids, n_ids);
        if (!out)
            return nullptr;
        *out_n = n_ids * (int)hp.d_model;
        return out;
    }
    if (strcmp(stage_name, "lm_step0_logits") == 0) {
        if (!embeds_in || n_embed_tokens <= 0)
            return nullptr;
        cosyvoice3_tts_reset_kv(ctx);
        float* out = cosyvoice3_tts_prefill_with_embeds(ctx, embeds_in, n_embed_tokens, /*n_past*/ 0);
        if (!out)
            return nullptr;
        *out_n = (int)hp.speech_vocab;
        return out;
    }
    // Flow Phase 3b single-block diff stages:
    //   "flow_dit_blk_<N>_out"     — final block output  [T, dit_dim] F32
    //   "flow_dit_blk_<N>_lnx_a"   — LN(x) before modulation
    //   "flow_dit_blk_<N>_h_a"     — post-modulate, pre-attn
    //   "flow_dit_blk_<N>_attn"    — attention out (pre-residual)
    //   "flow_dit_blk_<N>_xattn"   — x + gate_msa * attn_out
    //   "flow_dit_blk_<N>_ff"      — FFN out (pre-residual)
    //
    // `embeds_in` carries the packed [x | t_emb] block: first
    // n_embed_tokens*dit_dim floats are x [T, dit_dim], remaining dit_dim
    // floats are t_emb (post time_mlp). The caller computes T from the
    // ref archive's tensor shape.
    if (strncmp(stage_name, "flow_dit_blk_", 13) == 0) {
        if (!ctx->flow.loaded || !embeds_in || n_embed_tokens <= 0)
            return nullptr;
        const auto& fh = ctx->flow.hp;
        const int d = (int)fh.dit_dim;
        // Parse block index: "flow_dit_blk_<N>_<sfx>". Find the underscore
        // after the digits.
        const char* p = stage_name + 13;
        int block_idx = 0;
        const char* sfx = p;
        while (*sfx >= '0' && *sfx <= '9') {
            block_idx = block_idx * 10 + (*sfx - '0');
            sfx++;
        }
        if (*sfx != '_')
            return nullptr;
        sfx++; // past the underscore separating idx from suffix
        const char* tensor_name = nullptr;
        if (strcmp(sfx, "out") == 0)
            tensor_name = "dit_block_out";
        else if (strcmp(sfx, "lnx_a") == 0)
            tensor_name = "dbg_lnx_a";
        else if (strcmp(sfx, "h_a") == 0)
            tensor_name = "dbg_h_a";
        else if (strcmp(sfx, "attn") == 0)
            tensor_name = "dbg_attn_raw";
        else if (strcmp(sfx, "xattn") == 0)
            tensor_name = "dbg_x_after_attn";
        else if (strcmp(sfx, "ff") == 0)
            tensor_name = "dbg_ff_raw";
        else {
            fprintf(stderr, "cosyvoice3_tts: unknown flow_dit_blk stage suffix '%s'\n", sfx);
            return nullptr;
        }
        const int T = n_embed_tokens;
        // embeds_in layout: T*d floats of x, then d floats of t_emb.
        const float* x = embeds_in;
        const float* t_emb = embeds_in + (size_t)T * d;
        float* out = cv3_extract_flow_dit_stage(ctx, block_idx, x, T, t_emb, tensor_name);
        if (!out)
            return nullptr;
        *out_n = T * d;
        return out;
    }
    // Flow Phase 3c pre-lookahead conv stack:
    //   "flow_pre_la_tok_emb"  — input_embedding(speech_ids)  [T_tok, mel_dim]
    //   "flow_pre_la_c1"       — leaky_relu(conv1(...))       [T_tok, 1024]
    //   "flow_pre_la_c2"       — conv2(post-causal-pad)       [T_tok, mel_dim]
    //   "flow_pre_la"          — final pre_la output (with residual)
    //                                                          [T_tok, mel_dim]
    //
    // `ids` carries speech-token IDs (length n_ids = T_tok). The graph
    // does the input_embedding lookup inside, so the diff harness
    // exercises the embedding + conv stack as a single unit.
    if (strncmp(stage_name, "flow_pre_la", 11) == 0) {
        if (!ctx->flow.loaded || !ids || n_ids <= 0)
            return nullptr;
        const auto& fh = ctx->flow.hp;
        const char* sfx = stage_name + 11;
        const char* tensor_name = nullptr;
        if (*sfx == 0)
            tensor_name = "pre_la_out";
        else if (strcmp(sfx, "_tok_emb") == 0)
            tensor_name = "pre_la_tok_emb";
        else if (strcmp(sfx, "_c1") == 0)
            tensor_name = "pre_la_c1";
        else if (strcmp(sfx, "_c2") == 0)
            tensor_name = "pre_la_c2";
        else {
            fprintf(stderr, "cosyvoice3_tts: unknown flow_pre_la stage suffix '%s'\n", sfx);
            return nullptr;
        }
        float* out = cv3_extract_pre_la_stage(ctx, ids, n_ids, tensor_name);
        if (!out)
            return nullptr;
        // pre_la_tok_emb and pre_la / pre_la_c2 are (T_tok, mel_dim);
        // pre_la_c1 is (T_tok, 1024).
        const int out_dim = (strcmp(tensor_name, "pre_la_c1") == 0) ? 1024 : (int)fh.mel_dim;
        *out_n = n_ids * out_dim;
        return out;
    }
    // Flow Phase 3c InputEmbedding (input pipeline) stages:
    //   "flow_in_pipe_spk"   — F.normalize(spk) -> spk_affine    [spk_dim_out]
    //   "flow_in_pipe_cat"   — cat[x, cond, mu, spks]            [T_mel, 320]
    //   "flow_in_pipe_proj"  — in_proj(cat)                      [T_mel, 1024]
    //   "flow_in_pipe_pos"   — conv_pos_embed(proj)              [T_mel, 1024]
    //   "flow_in_pipe"       — proj + conv_pos_embed(proj)       [T_mel, 1024]
    //
    // `embeds_in` packs (in this order):
    //   pre_la_out          [T_mel, mel_dim]  F32 — already repeat-interleaved
    //                                              upstream by token_mel_ratio,
    //                                              so length is T_mel (= 2*T_tok)
    //   spk_emb_raw         [spk_dim_in]      F32 — pre-normalize, pre-projection
    //   x_noisy             [T_mel, mel_dim]  F32 — CFM solver iterate
    //   cond                [T_mel, mel_dim]  F32 — prompt-prefix conditioning
    //
    // n_embed_tokens carries T_mel. The graph builder normalises spk and
    // broadcasts it over T_mel internally.
    if (strncmp(stage_name, "flow_in_pipe", 12) == 0) {
        if (!ctx->flow.loaded || !embeds_in || n_embed_tokens <= 0)
            return nullptr;
        const auto& fh = ctx->flow.hp;
        const int mel = (int)fh.mel_dim;
        const int spk_in = (int)fh.spk_dim_in;
        const int dit_dim = (int)fh.dit_dim;
        const int dit_in_dim = (int)fh.dit_input_dim;
        const int T_mel = n_embed_tokens;
        const char* sfx = stage_name + 12;
        const char* tensor_name = nullptr;
        if (*sfx == 0)
            tensor_name = "in_pipe_out";
        else if (strcmp(sfx, "_spk") == 0)
            tensor_name = "in_pipe_spk";
        else if (strcmp(sfx, "_cat") == 0)
            tensor_name = "in_pipe_cat";
        else if (strcmp(sfx, "_proj") == 0)
            tensor_name = "in_pipe_proj";
        else if (strcmp(sfx, "_pos") == 0)
            tensor_name = "in_pipe_pos";
        else {
            fprintf(stderr, "cosyvoice3_tts: unknown flow_in_pipe stage suffix '%s'\n", sfx);
            return nullptr;
        }
        // embeds_in layout: pre_la (T_mel*mel) | spk_raw (spk_in) | x (T_mel*mel) | cond (T_mel*mel)
        const float* pre_la = embeds_in;
        const float* spk_raw = pre_la + (size_t)T_mel * mel;
        const float* x_noisy = spk_raw + (size_t)spk_in;
        const float* cond = x_noisy + (size_t)T_mel * mel;
        float* out = cv3_extract_in_pipe_stage(ctx, pre_la, T_mel, spk_raw, x_noisy, cond, tensor_name);
        if (!out)
            return nullptr;
        int out_n_local;
        if (strcmp(tensor_name, "in_pipe_spk") == 0)
            out_n_local = (int)fh.spk_dim_out;
        else if (strcmp(tensor_name, "in_pipe_cat") == 0)
            out_n_local = T_mel * dit_in_dim;
        else
            out_n_local = T_mel * dit_dim;
        *out_n = out_n_local;
        return out;
    }
    // Full DiT estimator forward — the entire 22-block stack +
    // AdaLayerNormZero_Final + proj_out. Inputs are post-input-pipeline
    // (validated by the `flow_in_pipe*` stages):
    //
    //   "flow_dit_full_norm"  — output of AdaLN-Final, pre-proj_out
    //                            [T_mel, dit_dim]
    //   "flow_dit_full"       — final mel output (post proj_out)
    //                            [T_mel, mel_dim]
    //
    // `embeds_in` packs [x | t_emb] = T_mel*dit_dim + dit_dim floats.
    // n_embed_tokens = T_mel.
    if (strncmp(stage_name, "flow_dit_full", 13) == 0) {
        if (!ctx->flow.loaded || !embeds_in || n_embed_tokens <= 0)
            return nullptr;
        const auto& fh = ctx->flow.hp;
        const int d = (int)fh.dit_dim;
        const int mel = (int)fh.mel_dim;
        const int T_mel = n_embed_tokens;
        const char* sfx = stage_name + 13;
        const char* tensor_name = nullptr;
        if (*sfx == 0)
            tensor_name = "dit_full_out";
        else if (strcmp(sfx, "_norm") == 0)
            tensor_name = "dit_full_norm";
        else {
            fprintf(stderr, "cosyvoice3_tts: unknown flow_dit_full stage suffix '%s'\n", sfx);
            return nullptr;
        }
        const float* x = embeds_in;
        const float* t_emb = embeds_in + (size_t)T_mel * d;
        float* out = cv3_extract_dit_full_stage(ctx, x, T_mel, t_emb, tensor_name);
        if (!out)
            return nullptr;
        const int out_dim = (strcmp(tensor_name, "dit_full_out") == 0) ? mel : d;
        *out_n = T_mel * out_dim;
        return out;
    }
    // Phase 3d-B — CFM Euler ODE driver. Inputs (packed in `embeds_in` in
    // this order):
    //   mu          [T_mel, mel_dim]   F32 — pre_la + repeat_interleave output
    //   spks_proj   [spk_dim_out]      F32 — already through normalize +
    //                                          spk_affine (caller-supplied)
    //   cond        [T_mel, mel_dim]   F32 — prompt-prefix conditioning (zeros
    //                                          for zero-shot)
    //   x_init      [T_mel, mel_dim]   F32 — initial Euler noise (caller-supplied
    //                                          so the diff harness can match
    //                                          upstream's seeded `rand_noise`)
    //
    // n_embed_tokens = T_mel. Stage suffixes:
    //   "flow_euler"            — final mel [T_mel, mel_dim] after 10 Euler steps
    //   "flow_euler_dphi_step0" — dphi_dt at step 0 (after CFG combine), debug
    //
    // `n_steps` and `cfg_rate` are pinned to the upstream defaults (10, 0.7);
    // exposing them on extract_stage would inflate the API for marginal gain.
    if (strncmp(stage_name, "flow_euler", 10) == 0) {
        if (!ctx->flow.loaded || !embeds_in || n_embed_tokens <= 0)
            return nullptr;
        const auto& fh = ctx->flow.hp;
        const int mel = (int)fh.mel_dim;
        const int spk_out = (int)fh.spk_dim_out;
        const int T_mel = n_embed_tokens;
        const char* sfx = stage_name + 10;
        const char* tensor_name = nullptr;
        if (*sfx == 0)
            tensor_name = "euler_out";
        else if (strcmp(sfx, "_dphi_step0") == 0)
            tensor_name = "euler_dphi_step0";
        else {
            fprintf(stderr, "cosyvoice3_tts: unknown flow_euler stage suffix '%s'\n", sfx);
            return nullptr;
        }
        // Unpack: mu | spks_proj | cond | x_init.
        const float* mu = embeds_in;
        const float* spks_proj = mu + (size_t)T_mel * mel;
        const float* cond = spks_proj + (size_t)spk_out;
        const float* x_init = cond + (size_t)T_mel * mel;
        const int n_steps = (int)fh.cfm_n_steps;          // 10
        const float cfg_rate = fh.cfm_inference_cfg_rate; // 0.7
        float* out = cv3_extract_euler_stage(ctx, mu, T_mel, spks_proj, cond, x_init, n_steps, cfg_rate, tensor_name);
        if (!out)
            return nullptr;
        *out_n = T_mel * mel;
        return out;
    }
    // Phase 4-A — HiFT F0 predictor (CausalConvRNNF0Predictor):
    //   "hift_f0"    — abs(classifier(condnet(mel))).squeeze(-1)  [T_mel]
    //
    // `embeds_in` carries the mel input packed as (T_mel * mel_dim) F32.
    // n_embed_tokens = T_mel.
    if (strcmp(stage_name, "hift_f0") == 0) {
        if (!ctx->hift.loaded || !embeds_in || n_embed_tokens <= 0)
            return nullptr;
        float* out = cv3_extract_hift_f0_stage(ctx, embeds_in, n_embed_tokens);
        if (!out)
            return nullptr;
        *out_n = n_embed_tokens;
        return out;
    }
    // Phase 4-B — HiFT decode forward (Option B: caller supplies s_stft).
    //   "hift_decode"                  — final 24 kHz audio (T_mel * 480 samples)
    //   "hift_decode_s_stft"           — input round-trip (18, T_stft)
    //   "hift_decode_conv_pre_out"     — (512, T_mel)
    //   "hift_decode_post_stage_{0,1,2}_x"
    //   "hift_decode_conv_post_out"    — (18,  T_stft)
    //   "hift_decode_mag" / "_phase"   — (9,   T_stft)
    //
    // `embeds_in` packs [mel | s_stft] where:
    //   mel:    n_embed_tokens * mel_dim   floats   ( T_mel = n_embed_tokens )
    //   s_stft: T_stft * 18                floats   ( T_stft = T_mel * 120 + 1 )
    if (strncmp(stage_name, "hift_decode", 11) == 0) {
        if (!ctx->hift.loaded || !embeds_in || n_embed_tokens <= 0)
            return nullptr;
        const int T_mel = n_embed_tokens;
        const int mel_dim = (int)ctx->hift.hp.mel_dim;
        const float* mel = embeds_in;
        const float* s_stft = embeds_in + (size_t)T_mel * mel_dim;
        return cv3_extract_hift_decode_stage(ctx, mel, T_mel, s_stft, stage_name, out_n);
    }
    // Phase 4-B-1 — HiFT source path (SineGen + m_source + STFT).
    //   embeds_in = [f0_mel | noise_buf] — f0_mel: T_mel floats; noise_buf:
    //                                       T_audio*9 floats (T_audio=T_mel*480).
    //   n_embed_tokens = T_mel.
    // Stages: hift_source_f0_up, hift_source_sine_waves, hift_source_sine_merge,
    //         hift_source (= s_stft).
    if (strncmp(stage_name, "hift_source", 11) == 0) {
        if (!ctx->hift.loaded || !embeds_in || n_embed_tokens <= 0)
            return nullptr;
        const int T_mel = n_embed_tokens;
        const float* f0_mel = embeds_in;
        const float* noise_buf = embeds_in + (size_t)T_mel;
        return cv3_extract_hift_source_stage(ctx, f0_mel, T_mel, noise_buf, stage_name, out_n);
    }
    // Phase 4-C — end-to-end mel → 24 kHz audio inference.
    //   embeds_in = [mel | noise_buf] — mel: T_mel*mel_dim; noise: T_audio*9.
    //   n_embed_tokens = T_mel.
    if (strcmp(stage_name, "hift_inference") == 0) {
        if (!ctx->hift.loaded || !embeds_in || n_embed_tokens <= 0)
            return nullptr;
        const int T_mel = n_embed_tokens;
        const int mel_dim = (int)ctx->hift.hp.mel_dim;
        const float* mel = embeds_in;
        const float* noise_buf = embeds_in + (size_t)T_mel * mel_dim;
        return cv3_extract_hift_inference(ctx, mel, T_mel, noise_buf, out_n);
    }
    if (strcmp(stage_name, "flow_inventory") == 0) {
        if (!ctx->flow.loaded) {
            fprintf(stderr, "cosyvoice3_tts: flow_inventory requested but flow not loaded\n");
            return nullptr;
        }
        // Return a small sentinel buffer encoding (n_dit_layers, dit_dim,
        // dit_heads, head_dim, ff_dim, input_dim, mel_dim, spk_dim_in,
        // spk_dim_out, n_steps) — useful for the diff harness to verify
        // it sees the flow GGUF.
        float* out = (float*)malloc(10 * sizeof(float));
        if (!out)
            return nullptr;
        const auto& fh = ctx->flow.hp;
        out[0] = (float)fh.n_dit_layers;
        out[1] = (float)fh.dit_dim;
        out[2] = (float)fh.dit_heads;
        out[3] = (float)fh.dit_head_dim;
        out[4] = (float)fh.dit_ff_dim;
        out[5] = (float)fh.dit_input_dim;
        out[6] = (float)fh.mel_dim;
        out[7] = (float)fh.spk_dim_in;
        out[8] = (float)fh.spk_dim_out;
        out[9] = (float)fh.cfm_n_steps;
        *out_n = 10;
        return out;
    }
    fprintf(stderr, "cosyvoice3_tts: unknown stage '%s'\n", stage_name);
    return nullptr;
}

// ---------------------------------------------------------------------------
// Phase 3 — Flow loader + hparam reader
// ---------------------------------------------------------------------------

extern "C" int cosyvoice3_tts_init_flow_from_file(struct cosyvoice3_tts_context* ctx, const char* path) {
    if (!ctx || !path) {
        fprintf(stderr, "cosyvoice3_tts: init_flow: bad args\n");
        return -1;
    }
    if (ctx->flow.loaded) {
        fprintf(stderr, "cosyvoice3_tts: flow already loaded\n");
        return 0;
    }

    // ---- Metadata pass ----
    ggml_context* gctx_dummy = nullptr;
    gguf_init_params gp = {/*no_alloc=*/true, &gctx_dummy};
    gguf_context* gctx = gguf_init_from_file(path, gp);
    if (!gctx) {
        fprintf(stderr, "cosyvoice3_tts: init_flow: failed to read GGUF '%s'\n", path);
        return -1;
    }

    auto& fh = ctx->flow.hp;
    fh.n_dit_layers = cv3_kv_u32(gctx, "cosyvoice3.flow.n_dit_layers", fh.n_dit_layers);
    fh.dit_dim = cv3_kv_u32(gctx, "cosyvoice3.flow.dit_dim", fh.dit_dim);
    fh.dit_heads = cv3_kv_u32(gctx, "cosyvoice3.flow.dit_heads", fh.dit_heads);
    fh.dit_head_dim = cv3_kv_u32(gctx, "cosyvoice3.flow.dit_head_dim", fh.dit_head_dim);
    fh.dit_ff_dim = cv3_kv_u32(gctx, "cosyvoice3.flow.dit_ff_dim", fh.dit_ff_dim);
    fh.dit_input_dim = cv3_kv_u32(gctx, "cosyvoice3.flow.dit_input_dim", fh.dit_input_dim);
    fh.mel_dim = cv3_kv_u32(gctx, "cosyvoice3.flow.mel_dim", fh.mel_dim);
    fh.spk_dim_in = cv3_kv_u32(gctx, "cosyvoice3.flow.spk_dim_in", fh.spk_dim_in);
    fh.spk_dim_out = cv3_kv_u32(gctx, "cosyvoice3.flow.spk_dim_out", fh.spk_dim_out);
    fh.speech_codebook = cv3_kv_u32(gctx, "cosyvoice3.flow.speech_codebook", fh.speech_codebook);
    fh.pre_lookahead_len = cv3_kv_u32(gctx, "cosyvoice3.flow.pre_lookahead_len", fh.pre_lookahead_len);
    fh.token_mel_ratio = cv3_kv_u32(gctx, "cosyvoice3.flow.token_mel_ratio", fh.token_mel_ratio);
    fh.input_frame_rate = cv3_kv_u32(gctx, "cosyvoice3.flow.input_frame_rate", fh.input_frame_rate);
    fh.cfm_n_steps = cv3_kv_u32(gctx, "cosyvoice3.flow.cfm_n_steps", fh.cfm_n_steps);
    fh.cfm_inference_cfg_rate = cv3_kv_f32(gctx, "cosyvoice3.flow.cfm_inference_cfg_rate", fh.cfm_inference_cfg_rate);
    fh.cfm_sigma_min = cv3_kv_f32(gctx, "cosyvoice3.flow.cfm_sigma_min", fh.cfm_sigma_min);
    fh.rope_theta = cv3_kv_f32(gctx, "cosyvoice3.flow.rope_theta", fh.rope_theta);
    gguf_free(gctx);

    // ---- Weight pass ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "cosyvoice3_tts:flow", wl)) {
        fprintf(stderr, "cosyvoice3_tts: init_flow: load_weights failed for '%s'\n", path);
        return -1;
    }
    ctx->flow.ctx_w = wl.ctx;
    ctx->flow.buf_w = wl.buf;
    ctx->flow.tensors = std::move(wl.tensors);

    auto require_t = [&](const std::string& name) -> ggml_tensor* {
        return core_gguf::require(ctx->flow.tensors, name.c_str(), "cosyvoice3_tts:flow");
    };

    auto& f = ctx->flow;
    f.input_embd_w = require_t("cosyvoice3.flow.input_embd.w");
    f.pre_la_c1_w = require_t("cosyvoice3.flow.pre_la.conv1.w");
    f.pre_la_c1_b = require_t("cosyvoice3.flow.pre_la.conv1.b");
    f.pre_la_c2_w = require_t("cosyvoice3.flow.pre_la.conv2.w");
    f.pre_la_c2_b = require_t("cosyvoice3.flow.pre_la.conv2.b");
    f.spk_affine_w = require_t("cosyvoice3.flow.spk_affine.w");
    f.spk_affine_b = require_t("cosyvoice3.flow.spk_affine.b");
    f.dit_in_proj_w = require_t("cosyvoice3.flow.dit.in_proj.w");
    f.dit_in_proj_b = require_t("cosyvoice3.flow.dit.in_proj.b");
    f.dit_conv_pos_c1_w = require_t("cosyvoice3.flow.dit.conv_pos.c1.w");
    f.dit_conv_pos_c1_b = require_t("cosyvoice3.flow.dit.conv_pos.c1.b");
    f.dit_conv_pos_c2_w = require_t("cosyvoice3.flow.dit.conv_pos.c2.w");
    f.dit_conv_pos_c2_b = require_t("cosyvoice3.flow.dit.conv_pos.c2.b");
    f.dit_time_mlp_0_w = require_t("cosyvoice3.flow.dit.time_mlp.0.w");
    f.dit_time_mlp_0_b = require_t("cosyvoice3.flow.dit.time_mlp.0.b");
    f.dit_time_mlp_2_w = require_t("cosyvoice3.flow.dit.time_mlp.2.w");
    f.dit_time_mlp_2_b = require_t("cosyvoice3.flow.dit.time_mlp.2.b");
    f.dit_rope_inv_freq = require_t("cosyvoice3.flow.dit.rope_inv_freq");
    f.dit_norm_out_w = require_t("cosyvoice3.flow.dit.norm_out.w");
    f.dit_norm_out_b = require_t("cosyvoice3.flow.dit.norm_out.b");
    f.dit_proj_out_w = require_t("cosyvoice3.flow.dit.proj_out.w");
    f.dit_proj_out_b = require_t("cosyvoice3.flow.dit.proj_out.b");

    f.blocks.resize(fh.n_dit_layers);
    for (uint32_t L = 0; L < fh.n_dit_layers; L++) {
        char prefix[48];
        snprintf(prefix, sizeof(prefix), "cosyvoice3.flow.dit.blk.%u", L);
        auto& b = f.blocks[L];
        std::string p = prefix;
        b.adaln_w = require_t(p + ".adaln.w");
        b.adaln_b = require_t(p + ".adaln.b");
        b.attn_q_w = require_t(p + ".attn.q.w");
        b.attn_q_b = require_t(p + ".attn.q.b");
        b.attn_k_w = require_t(p + ".attn.k.w");
        b.attn_k_b = require_t(p + ".attn.k.b");
        b.attn_v_w = require_t(p + ".attn.v.w");
        b.attn_v_b = require_t(p + ".attn.v.b");
        b.attn_o_w = require_t(p + ".attn.o.w");
        b.attn_o_b = require_t(p + ".attn.o.b");
        b.ffn_l1_w = require_t(p + ".ffn.l1.w");
        b.ffn_l1_b = require_t(p + ".ffn.l1.b");
        b.ffn_l2_w = require_t(p + ".ffn.l2.w");
        b.ffn_l2_b = require_t(p + ".ffn.l2.b");
        if (!b.adaln_w || !b.attn_q_w || !b.ffn_l1_w) {
            fprintf(stderr, "cosyvoice3_tts: init_flow: missing tensors in %s.*\n", prefix);
            return -1;
        }
    }

    if (!f.input_embd_w || !f.dit_in_proj_w || !f.dit_proj_out_w) {
        fprintf(stderr, "cosyvoice3_tts: init_flow: missing top-level flow tensors\n");
        return -1;
    }

    f.loaded = true;
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr,
                "cosyvoice3_tts:flow loaded %zu tensors  dit=%uL d=%u h=%u/hd=%u ff=%u "
                "in_dim=%u mel=%u spk=%u/%u codebook=%u cfm_steps=%u cfg=%.2f\n",
                f.tensors.size(), fh.n_dit_layers, fh.dit_dim, fh.dit_heads, fh.dit_head_dim, fh.dit_ff_dim,
                fh.dit_input_dim, fh.mel_dim, fh.spk_dim_in, fh.spk_dim_out, fh.speech_codebook, fh.cfm_n_steps,
                (double)fh.cfm_inference_cfg_rate);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Phase 3b — single DiT block forward (AdaLN-Zero + bidirectional MHA + FFN)
// ---------------------------------------------------------------------------
//
// Upstream ref (cosyvoice/flow/DiT/modules.py, lucidrains-style block):
//
//   AdaLayerNormZero.forward(x, emb):
//     emb = self.linear(self.silu(emb))                              # (B, 6d)
//     shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
//         = torch.chunk(emb, 6, dim=1)
//     x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
//     return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
//
//   DiTBlock.forward(x, t, mask=None, rope=...):
//     norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, t)
//     x = x + gate_msa[:, None] * self.attn(norm, rope=rope)
//     ff = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
//     x = x + gate_mlp[:, None] * self.ff(ff)
//
// Notes verified against upstream PyTorch source (modules.py l. 230–245
// for AdaLN, l. 500–531 for DiTBlock):
//   - SiLU is applied to t_emb BEFORE the AdaLN linear (not after).
//   - chunk order is (shift, scale, gate) × {msa, mlp}, in that order.
//   - both LayerNorms (AdaLN's `norm` and DiTBlock's `ff_norm`) are
//     `nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)` — affine-free
//     (no γ/β weights), NOT RMSNorm. Use `ggml_norm`.
//   - FFN is plain Linear(d→ff)→GELU(approximate="tanh")→Linear(ff→d). No
//     SiLU, no GLU gating. ggml_gelu maps to the tanh approximation.
//   - RoPE comes from `x_transformers.RotaryEmbedding`. Its `freqs =
//     stack((freqs, freqs), -1)` + adjacent-pair `rotate_half` is the
//     interleaved (GPT-J/Llama-classic) RoPE, which is ggml's
//     GGML_ROPE_TYPE_NORMAL (=0), NOT NEOX. theta=10000 from
//     `cosyvoice3.flow.rope_theta`.

namespace {

ggml_cgraph* cv3_build_flow_dit_block_graph(cosyvoice3_tts_context* ctx, int block_idx, int T) {
    const auto& fh = ctx->flow.hp;
    GGML_ASSERT(block_idx >= 0 && (uint32_t)block_idx < fh.n_dit_layers);
    GGML_ASSERT(T > 0);
    const auto& b = ctx->flow.blocks[block_idx];
    const int d = (int)fh.dit_dim;       // 1024
    const int n_h = (int)fh.dit_heads;   // 16
    const int hd = (int)fh.dit_head_dim; // 64
    const float ln_eps = 1e-6f;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const float rope_theta = fh.rope_theta;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 1024, false);

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_input(x);
    ggml_set_name(x, "dit_x_in");

    ggml_tensor* t_emb = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d);
    ggml_set_input(t_emb);
    ggml_set_name(t_emb, "dit_t_emb_in");

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_input(positions);
    ggml_set_name(positions, "dit_positions");

    // AdaLN-Zero modulation: linear(silu(t_emb)) → chunk(6) along last dim.
    // adaln.w has ne=(d, 6d), adaln.b has ne=(6d,) F32.
    ggml_tensor* t_silu = ggml_silu(ctx0, t_emb);
    ggml_tensor* mod = ggml_mul_mat(ctx0, b.adaln_w, t_silu); // (6d,)
    mod = ggml_add(ctx0, mod, b.adaln_b);                     // (6d,)
    const size_t fs = sizeof(float);
    auto chunk = [&](int idx) { return ggml_view_1d(ctx0, mod, d, (size_t)(idx * d) * fs); };
    ggml_tensor* shift_msa = chunk(0);
    ggml_tensor* scale_msa = chunk(1);
    ggml_tensor* gate_msa = chunk(2);
    ggml_tensor* shift_mlp = chunk(3);
    ggml_tensor* scale_mlp = chunk(4);
    ggml_tensor* gate_mlp = chunk(5);

    // Pre-attention LayerNorm (affine-free) + modulation.
    //   h = LN(x) * (1 + scale_msa) + shift_msa
    //     = LN(x) + LN(x) * scale_msa + shift_msa
    ggml_tensor* lnx_a = ggml_norm(ctx0, x, ln_eps);
    ggml_set_name(lnx_a, "dbg_lnx_a");
    ggml_set_output(lnx_a);
    ggml_tensor* h_a = ggml_add(ctx0, lnx_a, ggml_mul(ctx0, lnx_a, scale_msa));
    h_a = ggml_add(ctx0, h_a, shift_msa);
    ggml_set_name(h_a, "dbg_h_a");
    ggml_set_output(h_a);

    // Bidirectional MHA with partial RoPE. Upstream `AttnProcessor` calls
    // `apply_rotary_pos_emb` on the pre-reshape Q/K (shape (B, T, n_h*hd))
    // with `rot_dim = head_dim = 64`. The x_transformers helper splits as
    // `t[..., :rot_dim]` + `t[..., rot_dim:]` and only rotates the first
    // 64 channels — which corresponds to *only head 0*. Heads 1..15 carry
    // no positional info. Match by ropeing the full Q/K tensor with
    // `n_dims=hd` so only the first `hd` channels rotate (NORMAL = adjacent
    // pairs, matches x_transformers' interleaved freqs+rotate_half).
    ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_q_w, h_a), b.attn_q_b); // (d, T)
    ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_k_w, h_a), b.attn_k_b);
    ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_v_w, h_a), b.attn_v_b);
    // ggml_rope_ext requires ne[2] == positions length. Treat the full
    // d-dim as a "single big head" via reshape (d, 1, T), then apply
    // partial RoPE over `hd` elements.
    Q = ggml_reshape_3d(ctx0, Q, d, 1, T);
    K = ggml_reshape_3d(ctx0, K, d, 1, T);
    Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hd, GGML_ROPE_TYPE_NORMAL, /*n_ctx_orig*/ 0, rope_theta, 1.0f, 0.0f,
                      1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(ctx0, K, positions, nullptr, hd, GGML_ROPE_TYPE_NORMAL, /*n_ctx_orig*/ 0, rope_theta, 1.0f, 0.0f,
                      1.0f, 0.0f, 0.0f);
    // Now reshape into per-head layout for flash-attn.
    Q = ggml_reshape_3d(ctx0, Q, hd, n_h, T);
    K = ggml_reshape_3d(ctx0, K, hd, n_h, T);
    V = ggml_reshape_3d(ctx0, V, hd, n_h, T);
    Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
    V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));
    ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K, V, /*mask*/ nullptr, attn_scale, 0.0f, 0.0f);
    attn = ggml_reshape_2d(ctx0, attn, d, T);
    attn = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_o_w, attn), b.attn_o_b);
    ggml_set_name(attn, "dbg_attn_raw");
    ggml_set_output(attn);

    // Gated residual: x = x + gate_msa * attn_out.
    ggml_tensor* x_after_attn = ggml_add(ctx0, x, ggml_mul(ctx0, attn, gate_msa));
    ggml_set_name(x_after_attn, "dbg_x_after_attn");
    ggml_set_output(x_after_attn);

    // Pre-FFN LayerNorm + modulation.
    ggml_tensor* lnx_f = ggml_norm(ctx0, x_after_attn, ln_eps);
    ggml_tensor* h_f = ggml_add(ctx0, lnx_f, ggml_mul(ctx0, lnx_f, scale_mlp));
    h_f = ggml_add(ctx0, h_f, shift_mlp);

    // FFN: Linear(d→ff) → GELU(tanh) → Linear(ff→d).
    ggml_tensor* ff = ggml_add(ctx0, ggml_mul_mat(ctx0, b.ffn_l1_w, h_f), b.ffn_l1_b);
    ff = ggml_gelu(ctx0, ff);
    ff = ggml_add(ctx0, ggml_mul_mat(ctx0, b.ffn_l2_w, ff), b.ffn_l2_b);
    ggml_set_name(ff, "dbg_ff_raw");
    ggml_set_output(ff);

    // Gated residual: y = x' + gate_mlp * ff_out.
    ggml_tensor* y = ggml_add(ctx0, x_after_attn, ggml_mul(ctx0, ff, gate_mlp));
    ggml_set_name(y, "dit_block_out");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_free(ctx0);
    return gf;
}

} // namespace

extern "C" float* cosyvoice3_tts_run_flow_dit_block(struct cosyvoice3_tts_context* ctx, int block_idx, const float* x,
                                                    int T, const float* t_emb) {
    if (!ctx || !ctx->flow.loaded || !x || !t_emb || T <= 0 || block_idx < 0)
        return nullptr;
    const auto& fh = ctx->flow.hp;
    if ((uint32_t)block_idx >= fh.n_dit_layers) {
        fprintf(stderr, "cosyvoice3_tts: block_idx %d out of range [0, %u)\n", block_idx, fh.n_dit_layers);
        return nullptr;
    }
    const int d = (int)fh.dit_dim;

    // Building any graph in ctx->compute_meta clobbers the cached LM
    // step graph's metadata. Invalidate so the next AR step rebuilds.
    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;

    ggml_cgraph* gf = cv3_build_flow_dit_block_graph(ctx, block_idx, T);
    if (!gf)
        return nullptr;
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "cosyvoice3_tts: dit_block alloc_graph failed\n");
        return nullptr;
    }

    auto set_t = [&](const char* nm, const void* data, size_t bytes) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
        if (!t)
            return false;
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };

    if (!set_t("dit_x_in", x, (size_t)d * T * sizeof(float)))
        return nullptr;
    if (!set_t("dit_t_emb_in", t_emb, (size_t)d * sizeof(float)))
        return nullptr;
    std::vector<int32_t> pos((size_t)T);
    for (int i = 0; i < T; i++)
        pos[i] = i;
    if (!set_t("dit_positions", pos.data(), pos.size() * sizeof(int32_t)))
        return nullptr;

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cosyvoice3_tts: dit_block compute failed\n");
        return nullptr;
    }

    ggml_tensor* out_t = ggml_graph_get_tensor(gf, "dit_block_out");
    if (!out_t)
        return nullptr;
    const size_t n = (size_t)ggml_nelements(out_t);
    float* out = (float*)malloc(n * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(out_t, out, 0, n * sizeof(float));
    return out;
}

namespace {

// Build + run the per-block graph and extract a specific named tensor.
// Used by `cosyvoice3_tts_extract_stage` for the `flow_dit_blk_<N>_*`
// stages so the diff harness can hit any intermediate (post-LN,
// post-modulate, post-attn, post-residual, post-FFN, block-out) on the
// same graph build.
float* cv3_extract_flow_dit_stage(cosyvoice3_tts_context* ctx, int block_idx, const float* x, int T, const float* t_emb,
                                  const char* tensor_name) {
    if (!ctx || !ctx->flow.loaded || !x || !t_emb || T <= 0 || block_idx < 0 || !tensor_name)
        return nullptr;
    const auto& fh = ctx->flow.hp;
    if ((uint32_t)block_idx >= fh.n_dit_layers)
        return nullptr;
    const int d = (int)fh.dit_dim;

    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;

    ggml_cgraph* gf = cv3_build_flow_dit_block_graph(ctx, block_idx, T);
    if (!gf)
        return nullptr;
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;

    auto set_t = [&](const char* nm, const void* data, size_t bytes) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
        if (!t)
            return false;
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };
    if (!set_t("dit_x_in", x, (size_t)d * T * sizeof(float)))
        return nullptr;
    if (!set_t("dit_t_emb_in", t_emb, (size_t)d * sizeof(float)))
        return nullptr;
    std::vector<int32_t> pos((size_t)T);
    for (int i = 0; i < T; i++)
        pos[i] = i;
    if (!set_t("dit_positions", pos.data(), pos.size() * sizeof(int32_t)))
        return nullptr;
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;

    ggml_tensor* out_t = ggml_graph_get_tensor(gf, tensor_name);
    if (!out_t) {
        fprintf(stderr, "cosyvoice3_tts: tensor '%s' not in flow_dit_block graph\n", tensor_name);
        return nullptr;
    }
    const size_t n = (size_t)ggml_nelements(out_t);
    float* out = (float*)malloc(n * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(out_t, out, 0, n * sizeof(float));
    return out;
}

} // namespace

extern "C" float* cosyvoice3_tts_solve_flow_euler(struct cosyvoice3_tts_context* ctx, const float* mu, int T_mel,
                                                  const float* spks_proj, const float* cond, const float* x_init,
                                                  int n_steps, float cfg_rate) {
    return cv3_run_solve_euler(ctx, mu, T_mel, spks_proj, cond, x_init, n_steps, cfg_rate, /*dphi_step0_out*/ nullptr);
}

extern "C" int cosyvoice3_tts_get_flow_hparams(struct cosyvoice3_tts_context* ctx, uint32_t* n_dit_layers,
                                               uint32_t* dit_dim, uint32_t* dit_heads, uint32_t* dit_head_dim,
                                               uint32_t* dit_ff_dim, uint32_t* dit_input_dim, uint32_t* mel_dim,
                                               uint32_t* spk_dim_in, uint32_t* spk_dim_out, uint32_t* cfm_n_steps,
                                               float* cfm_cfg_rate) {
    if (!ctx || !ctx->flow.loaded)
        return -1;
    const auto& fh = ctx->flow.hp;
    if (n_dit_layers)
        *n_dit_layers = fh.n_dit_layers;
    if (dit_dim)
        *dit_dim = fh.dit_dim;
    if (dit_heads)
        *dit_heads = fh.dit_heads;
    if (dit_head_dim)
        *dit_head_dim = fh.dit_head_dim;
    if (dit_ff_dim)
        *dit_ff_dim = fh.dit_ff_dim;
    if (dit_input_dim)
        *dit_input_dim = fh.dit_input_dim;
    if (mel_dim)
        *mel_dim = fh.mel_dim;
    if (spk_dim_in)
        *spk_dim_in = fh.spk_dim_in;
    if (spk_dim_out)
        *spk_dim_out = fh.spk_dim_out;
    if (cfm_n_steps)
        *cfm_n_steps = fh.cfm_n_steps;
    if (cfm_cfg_rate)
        *cfm_cfg_rate = fh.cfm_inference_cfg_rate;
    return 0;
}

// ---------------------------------------------------------------------------
// Phase 3c — pre-lookahead conv stack + InputEmbedding (input pipeline)
// ---------------------------------------------------------------------------
//
// Upstream refs:
//   - PreLookaheadLayer (cosyvoice/transformer/upsample_encoder.py l. 66-103,
//     instantiated for cv3 with in=80, channels=1024, pre_lookahead_len=3):
//
//       outputs = inputs.transpose(1, 2)                   # (B, C, T)
//       outputs = F.pad(outputs, (0, pre_lookahead_len))   # right-pad 3 (lookahead)
//       outputs = F.leaky_relu(conv1(outputs))             # Conv1d(80, 1024, k=4)
//       outputs = F.pad(outputs, (K2 - 1, 0))              # left-pad 2 (causal)
//       outputs = conv2(outputs)                            # Conv1d(1024, 80, k=3)
//       outputs = outputs.transpose(1, 2)                  # (B, T, C)
//       return outputs + inputs                            # residual
//
//   - InputEmbedding (cosyvoice/flow/DiT/dit.py l. 76-98):
//
//       to_cat = [x, cond, text_embed, spks_broadcast]    # 4 × mel_dim = 320
//       x = self.proj(torch.cat(to_cat, dim=-1))          # Linear(320, 1024)
//       x = self.conv_pos_embed(x) + x                    # grouped causal conv
//
//   - CausalConvPositionEmbedding (cosyvoice/flow/DiT/modules.py l. 115-144):
//     2 × `nn.Conv1d(dim, dim, k=31, groups=16, padding=0)` + nn.Mish() with
//     each conv preceded by `F.pad(x, (K-1, 0))` (causal). Note: the helper
//     processes (B, C, T) — channel-first — and returns (B, T, C) after the
//     final transpose.

namespace {

// Mish = x * tanh(softplus(x)). ggml has ggml_softplus + ggml_tanh; chatterbox
// also exposes its own Mish helper but it's static to that .cpp. Local copy
// is cheap and keeps phase 3c self-contained.
ggml_tensor* cv3_mish(ggml_context* ctx, ggml_tensor* x) {
    ggml_tensor* sp = ggml_softplus(ctx, x);
    ggml_tensor* th = ggml_tanh(ctx, sp);
    return ggml_mul(ctx, x, th);
}

// Causal grouped conv1d for conv_pos_embed.
// h: (C, T) F32  — channel-first ggml layout, C inner / T outer.
// w: (K, C_per_group, C) F16  — sub-group weights at offset c0 = g*C_per_group
//                                 along the out dim.
// b: (C,) F32 — per-channel bias.
// Output: (C, T) with the same length T (left-pad K-1 on T, conv pad=0).
ggml_tensor* cv3_causal_grouped_conv1d(ggml_context* ctx, ggml_tensor* h, ggml_tensor* w, ggml_tensor* b) {
    const int K = (int)w->ne[0];
    const int C_per_g = (int)w->ne[1];
    const int C = (int)w->ne[2];
    const int T = (int)h->ne[1];
    const int G = C / C_per_g;
    GGML_ASSERT(h->ne[0] == C);
    GGML_ASSERT(G * C_per_g == C);

    ggml_tensor* out = nullptr;
    for (int g = 0; g < G; g++) {
        const size_t c0 = (size_t)g * C_per_g;
        // Per-group input slice: (C_per_g, T) — then transpose to (T, C_per_g)
        // matching ggml_conv_1d's expected (T, C_in) data layout (see
        // chatterbox_s3gen::causal_conv1d for the established convention).
        ggml_tensor* h_g = ggml_view_2d(ctx, h, C_per_g, T, h->nb[1], c0 * h->nb[0]);
        h_g = ggml_cont(ctx, ggml_transpose(ctx, h_g)); // (T, C_per_g)
        // Per-group weight slice: (K, C_per_g, C_per_g).
        ggml_tensor* w_g = ggml_view_3d(ctx, w, K, C_per_g, C_per_g, w->nb[1], w->nb[2], c0 * w->nb[2]);
        w_g = ggml_cont(ctx, w_g);
        // Left-pad K-1 (causal): ggml_conv_1d with symmetric pad=K-1 produces
        // (T + K - 1, C_per_g); take the first T entries (drops K-1 from
        // RIGHT) — which equals left-pad-only-by-K-1 conv. Same trick as
        // chatterbox_s3gen::causal_conv1d.
        ggml_tensor* y = ggml_conv_1d(ctx, w_g, h_g, /*stride*/ 1, /*pad*/ K - 1, /*dil*/ 1);
        y = ggml_view_2d(ctx, y, T, C_per_g, y->nb[1], 0);
        y = ggml_cont(ctx, y);
        // Transpose back to (C_per_g, T) and add per-group bias slice.
        y = ggml_cont(ctx, ggml_transpose(ctx, y));
        ggml_tensor* b_g = ggml_view_1d(ctx, b, C_per_g, c0 * b->nb[0]);
        y = ggml_add(ctx, y, b_g);
        // Concatenate per-group outputs along channel dim (dim 0).
        out = out ? ggml_concat(ctx, out, y, 0) : y;
    }
    return out;
}

// "Right-padded" dense conv1d — pads right by K-1 (lookahead), then conv
// with stride 1 and zero padding. Output length matches input length.
// Layout: x ne=(T, C_in); w ne=(K, C_in, C_out); b ne=(C_out).
ggml_tensor* cv3_lookahead_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    const int K = (int)w->ne[0];
    const int T = (int)x->ne[0];
    // Symmetric pad K-1 produces output of length T + K - 1. Take the
    // RIGHT T entries (drops K-1 from LEFT) — equivalent to right-pad
    // only by K-1.
    ggml_tensor* y = ggml_conv_1d(ctx, w, x, /*stride*/ 1, /*pad*/ K - 1, /*dil*/ 1);
    y = ggml_view_2d(ctx, y, T, (int)y->ne[1], y->nb[1], (size_t)(K - 1) * y->nb[0]);
    y = ggml_cont(ctx, y);
    if (b) {
        ggml_tensor* b2d = ggml_reshape_2d(ctx, b, 1, (int)b->ne[0]);
        y = ggml_add(ctx, y, b2d);
    }
    return y;
}

// Mirror of chatterbox_s3gen::causal_conv1d (left-pad K-1) — local copy
// to keep phase 3c self-contained.
ggml_tensor* cv3_causal_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    const int K = (int)w->ne[0];
    const int T = (int)x->ne[0];
    ggml_tensor* y = ggml_conv_1d(ctx, w, x, /*stride*/ 1, /*pad*/ K - 1, /*dil*/ 1);
    // Take the LEFT T entries (drops K-1 from RIGHT) → left-pad-only conv.
    y = ggml_view_2d(ctx, y, T, (int)y->ne[1], y->nb[1], 0);
    y = ggml_cont(ctx, y);
    if (b) {
        ggml_tensor* b2d = ggml_reshape_2d(ctx, b, 1, (int)b->ne[0]);
        y = ggml_add(ctx, y, b2d);
    }
    return y;
}

// Build the pre-lookahead conv stack graph:
//   ids (T_tok) -> input_embedding -> (T_tok, mel_dim)
//                       └> right-pad 3, conv1 (k=4, 80→1024), leaky_relu
//                                                 └> left-pad 2, conv2 (k=3, 1024→80)
//                                                          └> + residual -> (T_tok, mel_dim)
// Named graph outputs (settable via cv3_extract_pre_la_stage):
//   pre_la_tok_emb, pre_la_c1, pre_la_c2, pre_la_out
ggml_cgraph* cv3_build_pre_la_graph(cosyvoice3_tts_context* ctx, int T_tok) {
    const auto& f = ctx->flow;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 1024, false);

    ggml_tensor* ids_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_tok);
    ggml_set_input(ids_t);
    ggml_set_name(ids_t, "pre_la_ids_in");

    // Embedding lookup. input_embd.w ne=(mel, vocab); get_rows result ne=(mel, T_tok) F32.
    ggml_tensor* tok_emb = ggml_get_rows(ctx0, f.input_embd_w, ids_t);
    tok_emb = ggml_cont(ctx0, tok_emb);
    ggml_set_name(tok_emb, "pre_la_tok_emb");
    ggml_set_output(tok_emb);

    // Upstream wants (T, C) layout for the conv chain (transpose of get_rows).
    ggml_tensor* x_tc = ggml_cont(ctx0, ggml_transpose(ctx0, tok_emb)); // (T_tok, mel)

    // conv1: lookahead (right-pad K-1=3), kernel 4, in=80 out=1024, then LeakyReLU(0.01).
    ggml_tensor* c1 = cv3_lookahead_conv1d(ctx0, x_tc, f.pre_la_c1_w, f.pre_la_c1_b); // (T_tok, 1024)
    c1 = ggml_leaky_relu(ctx0, c1, 0.01f, /*inplace*/ false);
    // Transpose back to (C, T) for the named-output dump (matches upstream's
    // post-conv1 channel-first layout).
    ggml_tensor* c1_out = ggml_cont(ctx0, ggml_transpose(ctx0, c1)); // (1024, T_tok)
    ggml_set_name(c1_out, "pre_la_c1");
    ggml_set_output(c1_out);

    // conv2: causal (left-pad K-1=2), kernel 3, in=1024 out=80.
    ggml_tensor* c2 = cv3_causal_conv1d(ctx0, c1, f.pre_la_c2_w, f.pre_la_c2_b); // (T_tok, mel)
    ggml_tensor* c2_out = ggml_cont(ctx0, ggml_transpose(ctx0, c2));             // (mel, T_tok)
    ggml_set_name(c2_out, "pre_la_c2");
    ggml_set_output(c2_out);

    // Residual: + tok_emb (channel-first).
    ggml_tensor* y = ggml_add(ctx0, c2_out, tok_emb);
    ggml_set_name(y, "pre_la_out");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);
    // c1_out and tok_emb are SIDE branches (not on the path to y after the
    // transpose ops materialise them as cont-tensors). Expand them
    // explicitly so they survive into the executed graph.
    ggml_build_forward_expand(gf, tok_emb);
    ggml_build_forward_expand(gf, c1_out);
    ggml_build_forward_expand(gf, c2_out);

    ggml_free(ctx0);
    return gf;
}

float* cv3_extract_pre_la_stage(cosyvoice3_tts_context* ctx, const int32_t* ids, int T_tok, const char* tensor_name) {
    if (!ctx || !ctx->flow.loaded || !ids || T_tok <= 0 || !tensor_name)
        return nullptr;
    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;

    ggml_cgraph* gf = cv3_build_pre_la_graph(ctx, T_tok);
    if (!gf)
        return nullptr;
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;

    ggml_tensor* ids_t = ggml_graph_get_tensor(gf, "pre_la_ids_in");
    if (!ids_t)
        return nullptr;
    ggml_backend_tensor_set(ids_t, ids, 0, (size_t)T_tok * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cosyvoice3_tts: pre_la compute failed\n");
        return nullptr;
    }

    ggml_tensor* out_t = ggml_graph_get_tensor(gf, tensor_name);
    if (!out_t) {
        fprintf(stderr, "cosyvoice3_tts: tensor '%s' not in pre_la graph\n", tensor_name);
        return nullptr;
    }
    const size_t n = (size_t)ggml_nelements(out_t);
    float* out = (float*)malloc(n * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(out_t, out, 0, n * sizeof(float));
    return out;
}

// Build the InputEmbedding (input pipeline) graph:
//   spks_raw (spk_in)   ──F.normalize──> ─spk_affine─> spks_proj (spk_out=80)
//                                                    └ broadcast to (T_mel, 80)
//   pre_la (T_mel, 80) ┐
//   x_noisy (T_mel, 80)├──cat dim=-1──> (T_mel, 320) ─in_proj(320→1024)─> proj
//   cond    (T_mel, 80)│                                                    │
//   spks_bc (T_mel, 80)┘                                                    │
//                                                                          ├──+──> in_pipe_out
//                                                            conv_pos_embed(proj)
//                                                          (2× grouped causal
//                                                           conv1d-31 + Mish)
//
// Cat order per upstream InputEmbedding.forward:
//   to_cat = [x, cond, text_embed, spks] — x first.
//
// Named outputs (settable via cv3_extract_in_pipe_stage):
//   in_pipe_spk    (spk_dim_out,)
//   in_pipe_cat    (dit_input_dim, T_mel)  — channel-first
//   in_pipe_proj   (dit_dim, T_mel)
//   in_pipe_pos    (dit_dim, T_mel)  — conv_pos_embed(proj) without residual
//   in_pipe_out    (dit_dim, T_mel)  — proj + conv_pos_embed(proj)
ggml_cgraph* cv3_build_in_pipe_graph(cosyvoice3_tts_context* ctx, int T_mel) {
    const auto& fh = ctx->flow.hp;
    const auto& f = ctx->flow;
    const int mel = (int)fh.mel_dim;
    const int spk_in = (int)fh.spk_dim_in;
    const int spk_out = (int)fh.spk_dim_out;
    const int dit_in_dim = (int)fh.dit_input_dim;
    GGML_ASSERT(spk_out == mel);
    GGML_ASSERT(dit_in_dim == 4 * mel); // x + cond + mu + spks = 4 × mel_dim

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 2048, false);

    // ---- Inputs ----
    ggml_tensor* pre_la = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mel, T_mel);
    ggml_set_input(pre_la);
    ggml_set_name(pre_la, "in_pipe_pre_la_in"); // (mel, T_mel)
    ggml_tensor* spk = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, spk_in);
    ggml_set_input(spk);
    ggml_set_name(spk, "in_pipe_spk_in"); // (spk_in,)
    ggml_tensor* x_noisy = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mel, T_mel);
    ggml_set_input(x_noisy);
    ggml_set_name(x_noisy, "in_pipe_x_in"); // (mel, T_mel)
    ggml_tensor* cond = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mel, T_mel);
    ggml_set_input(cond);
    ggml_set_name(cond, "in_pipe_cond_in");

    // ---- spk projection: F.normalize(spk, dim=1) -> spk_affine ----
    // F.normalize: x / max(||x||_2, eps), default eps=1e-12.
    ggml_tensor* spk_2d = ggml_reshape_2d(ctx0, spk, spk_in, 1); // (spk_in, 1)
    ggml_tensor* spk_norm = ggml_rms_norm(ctx0, spk_2d, 0.0f);
    // ggml_rms_norm divides by sqrt(mean(x^2) + eps) = sqrt(sum(x^2)/N + eps).
    // F.normalize divides by sqrt(sum(x^2) + eps_l2). Compensate by
    // multiplying with sqrt(N) to convert "RMS" → "L2 norm" denominator.
    spk_norm = ggml_scale(ctx0, spk_norm, 1.0f / std::sqrt((float)spk_in));
    ggml_tensor* spk_proj = ggml_mul_mat(ctx0, f.spk_affine_w, spk_norm); // (spk_out, 1)
    spk_proj = ggml_add(ctx0, spk_proj, f.spk_affine_b);
    // ggml_cont so the named output owns its buffer — set_output on a
    // reshape view of an add result is fragile under sched allocation.
    spk_proj = ggml_cont(ctx0, ggml_reshape_1d(ctx0, spk_proj, spk_out));
    ggml_set_name(spk_proj, "in_pipe_spk");
    ggml_set_output(spk_proj);

    // ---- Broadcast spk over T_mel: (spk_out,) → (spk_out, T_mel) ----
    // Use ggml_repeat with a same-shape target.
    ggml_tensor* spk_template = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, spk_out, T_mel);
    ggml_tensor* spk_bc = ggml_repeat(ctx0, spk_proj, spk_template);

    // ---- Concat [x, cond, mu(=pre_la), spks] along channel dim ----
    // Each piece is (mel, T_mel) col-major; concat along dim 0 stacks
    // channels → (4*mel = dit_input_dim, T_mel).
    ggml_tensor* cat01 = ggml_concat(ctx0, x_noisy, cond, 0);   // (2*mel, T)
    ggml_tensor* cat012 = ggml_concat(ctx0, cat01, pre_la, 0);  // (3*mel, T)
    ggml_tensor* catted = ggml_concat(ctx0, cat012, spk_bc, 0); // (4*mel = dit_in_dim, T)
    ggml_set_name(catted, "in_pipe_cat");
    ggml_set_output(catted);

    // ---- in_proj: Linear(320, 1024) ----
    ggml_tensor* proj = ggml_mul_mat(ctx0, f.dit_in_proj_w, catted); // (1024, T)
    proj = ggml_add(ctx0, proj, f.dit_in_proj_b);
    ggml_set_name(proj, "in_pipe_proj");
    ggml_set_output(proj);

    // ---- conv_pos_embed: 2× grouped causal conv1d (k=31, groups=16) + Mish ----
    ggml_tensor* pos = cv3_causal_grouped_conv1d(ctx0, proj, f.dit_conv_pos_c1_w, f.dit_conv_pos_c1_b);
    pos = cv3_mish(ctx0, pos);
    pos = cv3_causal_grouped_conv1d(ctx0, pos, f.dit_conv_pos_c2_w, f.dit_conv_pos_c2_b);
    pos = cv3_mish(ctx0, pos);
    ggml_set_name(pos, "in_pipe_pos");
    ggml_set_output(pos);

    // ---- Residual: in_pipe_out = pos + proj ----
    ggml_tensor* y = ggml_add(ctx0, pos, proj);
    ggml_set_name(y, "in_pipe_out");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);
    // Side branches not directly on the path to y.
    ggml_build_forward_expand(gf, spk_proj);
    ggml_build_forward_expand(gf, catted);
    ggml_build_forward_expand(gf, proj);
    ggml_build_forward_expand(gf, pos);

    ggml_free(ctx0);
    return gf;
}

float* cv3_extract_in_pipe_stage(cosyvoice3_tts_context* ctx, const float* pre_la_out, int T_mel, const float* spk_emb,
                                 const float* x_noisy, const float* cond, const char* tensor_name) {
    if (!ctx || !ctx->flow.loaded || !pre_la_out || !spk_emb || !x_noisy || !cond || T_mel <= 0 || !tensor_name)
        return nullptr;
    const auto& fh = ctx->flow.hp;
    const int mel = (int)fh.mel_dim;
    const int spk_in = (int)fh.spk_dim_in;
    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;

    ggml_cgraph* gf = cv3_build_in_pipe_graph(ctx, T_mel);
    if (!gf)
        return nullptr;
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "cosyvoice3_tts: in_pipe alloc_graph failed\n");
        return nullptr;
    }

    auto set_t = [&](const char* nm, const void* data, size_t bytes) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
        if (!t)
            return false;
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };
    if (!set_t("in_pipe_pre_la_in", pre_la_out, (size_t)mel * T_mel * sizeof(float)))
        return nullptr;
    if (!set_t("in_pipe_spk_in", spk_emb, (size_t)spk_in * sizeof(float)))
        return nullptr;
    if (!set_t("in_pipe_x_in", x_noisy, (size_t)mel * T_mel * sizeof(float)))
        return nullptr;
    if (!set_t("in_pipe_cond_in", cond, (size_t)mel * T_mel * sizeof(float)))
        return nullptr;

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cosyvoice3_tts: in_pipe compute failed\n");
        return nullptr;
    }

    ggml_tensor* out_t = ggml_graph_get_tensor(gf, tensor_name);
    if (!out_t) {
        fprintf(stderr, "cosyvoice3_tts: tensor '%s' not in in_pipe graph\n", tensor_name);
        return nullptr;
    }
    const size_t n = (size_t)ggml_nelements(out_t);
    float* out = (float*)malloc(n * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(out_t, out, 0, n * sizeof(float));
    return out;
}

// ---------------------------------------------------------------------------
// Phase 3d-A — full 22-block DiT estimator forward + norm_out + proj_out
// ---------------------------------------------------------------------------
//
// Composes the per-block forward (phase 3b) into the full 22-layer stack,
// then applies `AdaLayerNormZero_Final` (norm_out: 2-chunk scale/shift) and
// `Linear(1024, 80)` (proj_out). This is exactly the function the CFM Euler
// solver calls inside its 10-step loop.
//
// Upstream ref (`cosyvoice/flow/DiT/dit.py::DiT.forward`, post-input-embed):
//
//     for block in self.transformer_blocks:
//         x = block(x, t, mask=attn_mask.bool(), rope=rope)
//     x = self.norm_out(x, t)                # AdaLayerNormZero_Final
//     output = self.proj_out(x).transpose(1, 2)
//
// AdaLayerNormZero_Final (`modules.py::AdaLayerNormZero_Final`):
//     emb = self.linear(self.silu(emb))      # (B, 2*dim)
//     scale, shift = torch.chunk(emb, 2, dim=1)  # scale FIRST
//     x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
//
// Note: AdaLN-Final's chunk order is (scale, shift) — opposite of the
// per-block AdaLN which is (shift, scale, gate) × {msa, mlp}.

// Inline per-block forward (matches cv3_build_flow_dit_block_graph but
// without the debug `dbg_*` named outputs). Takes pre-computed
// `silu(t_emb)` so we don't recompute the silu 22 times.
ggml_tensor* cv3_dit_block_apply(ggml_context* ctx0, ggml_tensor* x, ggml_tensor* t_silu, ggml_tensor* positions,
                                 const cv3_dit_block& b, int d, int n_h, int hd, float rope_theta) {
    const float ln_eps = 1e-6f;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = (int)x->ne[1];

    ggml_tensor* mod = ggml_mul_mat(ctx0, b.adaln_w, t_silu); // (6d,)
    mod = ggml_add(ctx0, mod, b.adaln_b);
    const size_t fs = sizeof(float);
    auto chunk = [&](int idx) { return ggml_view_1d(ctx0, mod, d, (size_t)(idx * d) * fs); };
    ggml_tensor* shift_msa = chunk(0);
    ggml_tensor* scale_msa = chunk(1);
    ggml_tensor* gate_msa = chunk(2);
    ggml_tensor* shift_mlp = chunk(3);
    ggml_tensor* scale_mlp = chunk(4);
    ggml_tensor* gate_mlp = chunk(5);

    ggml_tensor* lnx_a = ggml_norm(ctx0, x, ln_eps);
    ggml_tensor* h_a = ggml_add(ctx0, lnx_a, ggml_mul(ctx0, lnx_a, scale_msa));
    h_a = ggml_add(ctx0, h_a, shift_msa);

    ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_q_w, h_a), b.attn_q_b);
    ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_k_w, h_a), b.attn_k_b);
    ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_v_w, h_a), b.attn_v_b);
    Q = ggml_reshape_3d(ctx0, Q, d, 1, T);
    K = ggml_reshape_3d(ctx0, K, d, 1, T);
    Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hd, GGML_ROPE_TYPE_NORMAL, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    K = ggml_rope_ext(ctx0, K, positions, nullptr, hd, GGML_ROPE_TYPE_NORMAL, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    Q = ggml_reshape_3d(ctx0, Q, hd, n_h, T);
    K = ggml_reshape_3d(ctx0, K, hd, n_h, T);
    V = ggml_reshape_3d(ctx0, V, hd, n_h, T);
    Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
    V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));
    ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, attn_scale, 0.0f, 0.0f);
    attn = ggml_reshape_2d(ctx0, attn, d, T);
    attn = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_o_w, attn), b.attn_o_b);
    ggml_tensor* x_after_attn = ggml_add(ctx0, x, ggml_mul(ctx0, attn, gate_msa));

    ggml_tensor* lnx_f = ggml_norm(ctx0, x_after_attn, ln_eps);
    ggml_tensor* h_f = ggml_add(ctx0, lnx_f, ggml_mul(ctx0, lnx_f, scale_mlp));
    h_f = ggml_add(ctx0, h_f, shift_mlp);
    ggml_tensor* ff = ggml_add(ctx0, ggml_mul_mat(ctx0, b.ffn_l1_w, h_f), b.ffn_l1_b);
    ff = ggml_gelu(ctx0, ff);
    ff = ggml_add(ctx0, ggml_mul_mat(ctx0, b.ffn_l2_w, ff), b.ffn_l2_b);
    return ggml_add(ctx0, x_after_attn, ggml_mul(ctx0, ff, gate_mlp));
}

ggml_cgraph* cv3_build_dit_full_graph(cosyvoice3_tts_context* ctx, int T_mel) {
    const auto& fh = ctx->flow.hp;
    const auto& f = ctx->flow;
    const int d = (int)fh.dit_dim;
    const int n_h = (int)fh.dit_heads;
    const int hd = (int)fh.dit_head_dim;
    const int L = (int)fh.n_dit_layers;
    const float rope_theta = fh.rope_theta;
    const float ln_eps = 1e-6f;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    // 22 blocks * ~40 ops + norm_out + proj_out + intermediates ⇒ a few
    // thousand nodes. 8192 leaves ample headroom.
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T_mel);
    ggml_set_input(x);
    ggml_set_name(x, "dit_full_x_in");
    ggml_tensor* t_emb = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d);
    ggml_set_input(t_emb);
    ggml_set_name(t_emb, "dit_full_t_emb_in");
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_mel);
    ggml_set_input(positions);
    ggml_set_name(positions, "dit_full_positions");

    // Shared silu(t_emb) — used by every block AND by norm_out.
    ggml_tensor* t_silu = ggml_silu(ctx0, t_emb);

    for (int il = 0; il < L; il++) {
        x = cv3_dit_block_apply(ctx0, x, t_silu, positions, f.blocks[il], d, n_h, hd, rope_theta);
    }

    // norm_out: AdaLayerNormZero_Final — chunk(emb, 2) yields (scale, shift)
    // in THIS order (per upstream `AdaLayerNormZero_Final.forward`).
    ggml_tensor* nmod = ggml_mul_mat(ctx0, f.dit_norm_out_w, t_silu); // (2d,)
    nmod = ggml_add(ctx0, nmod, f.dit_norm_out_b);
    const size_t fs = sizeof(float);
    ggml_tensor* nscale = ggml_view_1d(ctx0, nmod, d, 0);
    ggml_tensor* nshift = ggml_view_1d(ctx0, nmod, d, (size_t)d * fs);
    ggml_tensor* lnx = ggml_norm(ctx0, x, ln_eps);
    ggml_tensor* normed = ggml_add(ctx0, lnx, ggml_mul(ctx0, lnx, nscale));
    normed = ggml_add(ctx0, normed, nshift);
    ggml_set_name(normed, "dit_full_norm");
    ggml_set_output(normed);

    // proj_out: Linear(dit_dim, mel_dim).
    ggml_tensor* mel_out = ggml_mul_mat(ctx0, f.dit_proj_out_w, normed); // (mel_dim, T_mel)
    mel_out = ggml_add(ctx0, mel_out, f.dit_proj_out_b);
    ggml_set_name(mel_out, "dit_full_out");
    ggml_set_output(mel_out);
    ggml_build_forward_expand(gf, mel_out);
    ggml_build_forward_expand(gf, normed); // side branch — keep it alive in the executed graph

    ggml_free(ctx0);
    return gf;
}

float* cv3_extract_dit_full_stage(cosyvoice3_tts_context* ctx, const float* x, int T_mel, const float* t_emb,
                                  const char* tensor_name) {
    if (!ctx || !ctx->flow.loaded || !x || !t_emb || T_mel <= 0 || !tensor_name)
        return nullptr;
    const auto& fh = ctx->flow.hp;
    const int d = (int)fh.dit_dim;

    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;

    ggml_cgraph* gf = cv3_build_dit_full_graph(ctx, T_mel);
    if (!gf)
        return nullptr;
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "cosyvoice3_tts: dit_full alloc_graph failed\n");
        return nullptr;
    }

    auto set_t = [&](const char* nm, const void* data, size_t bytes) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
        if (!t)
            return false;
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };
    if (!set_t("dit_full_x_in", x, (size_t)d * T_mel * sizeof(float)))
        return nullptr;
    if (!set_t("dit_full_t_emb_in", t_emb, (size_t)d * sizeof(float)))
        return nullptr;
    std::vector<int32_t> pos((size_t)T_mel);
    for (int i = 0; i < T_mel; i++)
        pos[i] = i;
    if (!set_t("dit_full_positions", pos.data(), pos.size() * sizeof(int32_t)))
        return nullptr;

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cosyvoice3_tts: dit_full compute failed\n");
        return nullptr;
    }

    ggml_tensor* out_t = ggml_graph_get_tensor(gf, tensor_name);
    if (!out_t) {
        fprintf(stderr, "cosyvoice3_tts: tensor '%s' not in dit_full graph\n", tensor_name);
        return nullptr;
    }
    const size_t n = (size_t)ggml_nelements(out_t);
    float* out = (float*)malloc(n * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(out_t, out, 0, n * sizeof(float));
    return out;
}

// ---------------------------------------------------------------------------
// Phase 3d-B — CFM Euler ODE driver
// ---------------------------------------------------------------------------
//
// Upstream refs (cosyvoice/flow/flow_matching.py):
//
//   CausalConditionalCFM.forward(mu, mask, n_timesteps, spks, cond):
//     z = self.rand_noise[:, :, :mu.size(2)] * temperature
//     t_span = linspace(0, 1, n_timesteps + 1)
//     if t_scheduler == 'cosine':
//         t_span = 1 - cos(t_span * pi/2)
//     return solve_euler(z, t_span, mu, mask, spks, cond)
//
//   ConditionalCFM.solve_euler(x, t_span, mu, mask, spks, cond):
//     t, dt = t_span[0], t_span[1] - t_span[0]
//     for step in range(1, len(t_span)):
//         # Classifier-Free Guidance with a batched (2, ...) call:
//         x_in    = stack([x,    x])     # both identical
//         mu_in   = stack([mu,   0])     # uncond branch zeroed
//         spks_in = stack([spks, 0])
//         cond_in = stack([cond, 0])
//         t_in    = [t, t]
//         dphi_dt = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
//         dphi_dt, cfg_dphi_dt = split(dphi_dt, [batch, batch])
//         dphi_dt = (1 + cfg_rate) * dphi_dt - cfg_rate * cfg_dphi_dt
//         x = x + dt * dphi_dt
//         t = t + dt
//         dt = t_span[step + 1] - t
//     return x
//
// Our port runs the estimator TWICE per step (cond + uncond, both B=1)
// rather than once with B=2; the numerical result is identical (the
// estimator is deterministic and batch-wise independent) and the
// per-call overhead is small compared to the 22-block forward itself.
//
// For diff parity, we accept `x_init` from the caller — generating
// `torch.manual_seed(0); randn([1,80,15000])[:, :, :T_mel]` bit-exactly
// in C++ would require porting torch's Mersenne-Twister + Box-Muller,
// which is out of scope for the ODE driver. The Python ref harness
// dumps the seeded noise into the GGUF archive and the diff harness
// hands it in via extract_stage.

// Sinusoidal positional embedding for the Euler time step. Mirrors
// `SinusPositionEmbedding.forward(x, scale=1000)` in
// cosyvoice/flow/DiT/modules.py l. 76-83. Returns a freshly allocated
// vector of size `dim` (= time_mlp.0 input = 256).
std::vector<float> cv3_compute_sin_emb(float t, int dim, float scale = 1000.0f) {
    const int half = dim / 2;
    std::vector<float> out((size_t)dim);
    const double decay = std::log(10000.0) / (half - 1);
    for (int i = 0; i < half; i++) {
        const double freq = std::exp(-(double)i * decay);
        const double pos = (double)scale * (double)t * freq;
        out[(size_t)i] = (float)std::sin(pos);          // first half: sin
        out[(size_t)(i + half)] = (float)std::cos(pos); // second half: cos
    }
    return out;
}

// Build the FULL estimator forward graph: time_mlp + InputEmbedding +
// 22 DiT blocks + AdaLN-Final norm_out + proj_out.
//
// Inputs (set externally):
//   "est_x_in"      [mel_dim, T_mel]    F32  current Euler iterate
//   "est_mu_in"     [mel_dim, T_mel]    F32  pre_la output (post repeat-interleave)
//   "est_spks_in"   [spk_dim_out]       F32  ALREADY projected (caller-supplied)
//   "est_cond_in"   [mel_dim, T_mel]    F32  prompt-prefix conditioning
//   "est_sin_emb"   [freq_embed=256]    F32  sinusoidal(t, 256)
//
// Output: "est_dphi" [mel_dim, T_mel]  F32  estimator velocity prediction.
ggml_cgraph* cv3_build_estimator_full_graph(cosyvoice3_tts_context* ctx, int T_mel) {
    const auto& fh = ctx->flow.hp;
    const auto& f = ctx->flow;
    const int mel = (int)fh.mel_dim;
    const int spk_out = (int)fh.spk_dim_out;
    const int dit_in_dim = (int)fh.dit_input_dim; // 320
    const int dit_dim = (int)fh.dit_dim;          // 1024
    const int n_h = (int)fh.dit_heads;
    const int hd = (int)fh.dit_head_dim;
    const int L = (int)fh.n_dit_layers;
    const int freq_embed_dim = 256;
    const float rope_theta = fh.rope_theta;
    const float ln_eps = 1e-6f;
    GGML_ASSERT(spk_out == mel);
    GGML_ASSERT(dit_in_dim == 4 * mel);

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // ---- Inputs ----
    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mel, T_mel);
    ggml_set_input(x);
    ggml_set_name(x, "est_x_in");
    ggml_tensor* mu = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mel, T_mel);
    ggml_set_input(mu);
    ggml_set_name(mu, "est_mu_in");
    ggml_tensor* spks_proj_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, spk_out);
    ggml_set_input(spks_proj_in);
    ggml_set_name(spks_proj_in, "est_spks_in");
    ggml_tensor* cond = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mel, T_mel);
    ggml_set_input(cond);
    ggml_set_name(cond, "est_cond_in");
    ggml_tensor* sin_emb = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, freq_embed_dim);
    ggml_set_input(sin_emb);
    ggml_set_name(sin_emb, "est_sin_emb");
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_mel);
    ggml_set_input(positions);
    ggml_set_name(positions, "est_positions");

    // ---- TimestepEmbedding's time_mlp: Linear -> SiLU -> Linear ----
    ggml_tensor* t_emb = ggml_mul_mat(ctx0, f.dit_time_mlp_0_w, sin_emb); // (dit_dim,)
    t_emb = ggml_add(ctx0, t_emb, f.dit_time_mlp_0_b);
    t_emb = ggml_silu(ctx0, t_emb);
    t_emb = ggml_mul_mat(ctx0, f.dit_time_mlp_2_w, t_emb);
    t_emb = ggml_add(ctx0, t_emb, f.dit_time_mlp_2_b);

    // ---- Broadcast spks_proj over T_mel (shape (spk_out, T_mel)) ----
    ggml_tensor* spk_template = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, spk_out, T_mel);
    ggml_tensor* spk_bc = ggml_repeat(ctx0, spks_proj_in, spk_template);

    // ---- InputEmbedding concat (cat dim = channels): [x, cond, mu, spk_bc] ----
    ggml_tensor* cat01 = ggml_concat(ctx0, x, cond, 0);         // (2*mel, T)
    ggml_tensor* cat012 = ggml_concat(ctx0, cat01, mu, 0);      // (3*mel, T)
    ggml_tensor* catted = ggml_concat(ctx0, cat012, spk_bc, 0); // (4*mel = dit_input_dim, T)

    // ---- in_proj: Linear(320, 1024) ----
    ggml_tensor* proj = ggml_mul_mat(ctx0, f.dit_in_proj_w, catted); // (dit_dim, T)
    proj = ggml_add(ctx0, proj, f.dit_in_proj_b);

    // ---- conv_pos_embed: 2× grouped causal conv1d-31 + Mish, + residual ----
    ggml_tensor* pos_h = cv3_causal_grouped_conv1d(ctx0, proj, f.dit_conv_pos_c1_w, f.dit_conv_pos_c1_b);
    pos_h = cv3_mish(ctx0, pos_h);
    pos_h = cv3_causal_grouped_conv1d(ctx0, pos_h, f.dit_conv_pos_c2_w, f.dit_conv_pos_c2_b);
    pos_h = cv3_mish(ctx0, pos_h);
    ggml_tensor* h = ggml_add(ctx0, pos_h, proj); // (dit_dim, T_mel) — input to first DiT block.

    // ---- Shared silu(t_emb) for all 22 blocks + norm_out ----
    ggml_tensor* t_silu = ggml_silu(ctx0, t_emb);

    // ---- 22-block DiT stack ----
    for (int il = 0; il < L; il++) {
        h = cv3_dit_block_apply(ctx0, h, t_silu, positions, f.blocks[il], dit_dim, n_h, hd, rope_theta);
    }

    // ---- norm_out (AdaLN-Final, chunk order (scale, shift)) ----
    ggml_tensor* nmod = ggml_mul_mat(ctx0, f.dit_norm_out_w, t_silu); // (2*dit_dim,)
    nmod = ggml_add(ctx0, nmod, f.dit_norm_out_b);
    const size_t fs = sizeof(float);
    ggml_tensor* nscale = ggml_view_1d(ctx0, nmod, dit_dim, 0);
    ggml_tensor* nshift = ggml_view_1d(ctx0, nmod, dit_dim, (size_t)dit_dim * fs);
    ggml_tensor* lnx = ggml_norm(ctx0, h, ln_eps);
    ggml_tensor* normed = ggml_add(ctx0, lnx, ggml_mul(ctx0, lnx, nscale));
    normed = ggml_add(ctx0, normed, nshift);

    // ---- proj_out: Linear(dit_dim, mel_dim) ----
    ggml_tensor* dphi = ggml_mul_mat(ctx0, f.dit_proj_out_w, normed); // (mel, T_mel)
    dphi = ggml_add(ctx0, dphi, f.dit_proj_out_b);
    ggml_set_name(dphi, "est_dphi");
    ggml_set_output(dphi);
    ggml_build_forward_expand(gf, dphi);

    ggml_free(ctx0);
    return gf;
}

// Run the full-estimator graph once. Returns a malloc'd float[T_mel*mel_dim]
// buffer (caller frees).
float* cv3_run_estimator_full(cosyvoice3_tts_context* ctx, const float* x, int T_mel, const float* mu,
                              const float* spks_proj, const float* cond, const float* sin_emb_t) {
    const auto& fh = ctx->flow.hp;
    const int mel = (int)fh.mel_dim;
    const int spk_out = (int)fh.spk_dim_out;
    const int freq_embed_dim = 256;

    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;

    ggml_cgraph* gf = cv3_build_estimator_full_graph(ctx, T_mel);
    if (!gf)
        return nullptr;
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;

    auto set_t = [&](const char* nm, const void* data, size_t bytes) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
        if (!t)
            return false;
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };
    const size_t mel_bytes = (size_t)mel * T_mel * sizeof(float);
    if (!set_t("est_x_in", x, mel_bytes))
        return nullptr;
    if (!set_t("est_mu_in", mu, mel_bytes))
        return nullptr;
    if (!set_t("est_spks_in", spks_proj, (size_t)spk_out * sizeof(float)))
        return nullptr;
    if (!set_t("est_cond_in", cond, mel_bytes))
        return nullptr;
    if (!set_t("est_sin_emb", sin_emb_t, (size_t)freq_embed_dim * sizeof(float)))
        return nullptr;
    std::vector<int32_t> pos((size_t)T_mel);
    for (int i = 0; i < T_mel; i++)
        pos[i] = i;
    if (!set_t("est_positions", pos.data(), pos.size() * sizeof(int32_t)))
        return nullptr;

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cosyvoice3_tts: estimator_full compute failed\n");
        return nullptr;
    }
    ggml_tensor* out_t = ggml_graph_get_tensor(gf, "est_dphi");
    if (!out_t)
        return nullptr;
    const size_t n = (size_t)ggml_nelements(out_t);
    float* out = (float*)malloc(n * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(out_t, out, 0, n * sizeof(float));
    return out;
}

// 10-step (configurable) cosine-schedule Euler ODE with classifier-free
// guidance. Inputs are caller-provided post-projection — `mu` is post
// `pre_la + repeat_interleave`, `spks_proj` is post `F.normalize +
// spk_affine`. `x_init` is the initial noise (caller matches upstream's
// seeded `rand_noise[:, :, :T_mel]`).
//
// Returns malloc'd float[T_mel*mel_dim] with the final mel iterate.
// If `dphi_step0_out` is non-null, also writes the post-CFG dphi_dt at
// step 1 (the FIRST loop iteration — upstream's `step in range(1, ...)`)
// there; the caller must pre-allocate T_mel*mel_dim floats.
float* cv3_run_solve_euler(cosyvoice3_tts_context* ctx, const float* mu, int T_mel, const float* spks_proj,
                           const float* cond, const float* x_init, int n_steps, float cfg_rate, float* dphi_step0_out) {
    if (!ctx || !ctx->flow.loaded || !mu || !spks_proj || !cond || !x_init || T_mel <= 0 || n_steps <= 0)
        return nullptr;
    const auto& fh = ctx->flow.hp;
    const int mel = (int)fh.mel_dim;
    const int spk_out = (int)fh.spk_dim_out;
    const size_t mel_n = (size_t)mel * T_mel;

    // Cosine t-schedule: t_span[k] = 1 - cos(k/N * pi/2), k in [0, N+1].
    std::vector<double> t_span((size_t)(n_steps + 1));
    for (int k = 0; k <= n_steps; k++) {
        const double u = (double)k / (double)n_steps; // linspace(0, 1, N+1)
        t_span[(size_t)k] = 1.0 - std::cos(u * 0.5 * M_PI);
    }

    // x starts as the caller-supplied init noise.
    std::vector<float> x(mel_n);
    std::memcpy(x.data(), x_init, mel_n * sizeof(float));

    // Pre-allocate the zero buffers used for the CFG uncond pass.
    std::vector<float> mu_zero(mel_n, 0.0f);
    std::vector<float> cond_zero(mel_n, 0.0f);
    std::vector<float> spks_zero((size_t)spk_out, 0.0f);

    double t = t_span[0];
    double dt = t_span[1] - t_span[0];
    for (int step = 1; step <= n_steps; step++) {
        std::vector<float> sin_emb = cv3_compute_sin_emb((float)t, 256);

        float* dphi_cond = cv3_run_estimator_full(ctx, x.data(), T_mel, mu, spks_proj, cond, sin_emb.data());
        if (!dphi_cond)
            return nullptr;
        float* dphi_unc = cv3_run_estimator_full(ctx, x.data(), T_mel, mu_zero.data(), spks_zero.data(),
                                                 cond_zero.data(), sin_emb.data());
        if (!dphi_unc) {
            free(dphi_cond);
            return nullptr;
        }

        // CFG combine: dphi = (1 + cfg) * dphi_cond - cfg * dphi_unc.
        const float w_cond = 1.0f + cfg_rate;
        const float w_unc = cfg_rate;
        for (size_t i = 0; i < mel_n; i++) {
            dphi_cond[i] = w_cond * dphi_cond[i] - w_unc * dphi_unc[i];
        }
        free(dphi_unc);

        if (step == 1 && dphi_step0_out) {
            std::memcpy(dphi_step0_out, dphi_cond, mel_n * sizeof(float));
        }

        // Euler step: x = x + dt * dphi.
        const float dtf = (float)dt;
        for (size_t i = 0; i < mel_n; i++) {
            x[i] += dtf * dphi_cond[i];
        }
        free(dphi_cond);

        t = t + dt;
        if (step < n_steps) {
            dt = t_span[(size_t)(step + 1)] - t;
        }
    }

    float* out = (float*)malloc(mel_n * sizeof(float));
    if (!out)
        return nullptr;
    std::memcpy(out, x.data(), mel_n * sizeof(float));
    return out;
}

float* cv3_extract_euler_stage(cosyvoice3_tts_context* ctx, const float* mu, int T_mel, const float* spks_proj,
                               const float* cond, const float* x_init, int n_steps, float cfg_rate,
                               const char* tensor_name) {
    const auto& fh = ctx->flow.hp;
    const int mel = (int)fh.mel_dim;
    const size_t mel_n = (size_t)mel * T_mel;

    if (strcmp(tensor_name, "euler_out") == 0) {
        return cv3_run_solve_euler(ctx, mu, T_mel, spks_proj, cond, x_init, n_steps, cfg_rate, nullptr);
    }
    if (strcmp(tensor_name, "euler_dphi_step0") == 0) {
        std::vector<float> dphi_buf(mel_n);
        float* mel_out =
            cv3_run_solve_euler(ctx, mu, T_mel, spks_proj, cond, x_init, n_steps, cfg_rate, dphi_buf.data());
        if (!mel_out)
            return nullptr;
        free(mel_out);
        float* out = (float*)malloc(mel_n * sizeof(float));
        if (!out)
            return nullptr;
        std::memcpy(out, dphi_buf.data(), mel_n * sizeof(float));
        return out;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Phase 4-A — HiFT loader + F0 predictor (CausalConvRNNF0Predictor)
// ---------------------------------------------------------------------------
//
// Upstream ref (cosyvoice/hifigan/f0_predictor.py l. 62-103):
//
//   condnet = nn.Sequential(
//     weight_norm(CausalConv1d(80, 512, kernel_size=4, causal_type='right')),
//     nn.ELU(),
//     weight_norm(CausalConv1d(512, 512, kernel_size=3, causal_type='left')),
//     nn.ELU(),
//     weight_norm(CausalConv1d(512, 512, kernel_size=3, causal_type='left')),
//     nn.ELU(),
//     weight_norm(CausalConv1d(512, 512, kernel_size=3, causal_type='left')),
//     nn.ELU(),
//     weight_norm(CausalConv1d(512, 512, kernel_size=3, causal_type='left')),
//     nn.ELU(),
//   )
//   classifier = nn.Linear(in_features=512, out_features=1)
//
//   def forward(x):                          # x: (B, 80, T_mel)
//     x = self.condnet[0](x)                 # first conv has right-pad (lookahead)
//     for i in range(1, len(self.condnet)):
//       x = self.condnet[i](x)
//     x = x.transpose(1, 2)                  # (B, T_mel, 512)
//     return torch.abs(self.classifier(x).squeeze(-1))   # (B, T_mel)
//
// The GGUF converter already materialises weight_norm into plain weights
// via `wn_resolve(g, v) = g * v / ||v||`, so the bindings are bare Conv1d
// weights/biases — no parametrisation handling needed at runtime.

ggml_cgraph* cv3_build_hift_f0_graph(cosyvoice3_tts_context* ctx, int T_mel) {
    const auto& h = ctx->hift;
    const int mel = (int)h.hp.mel_dim;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 1024, false);

    ggml_tensor* mel_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mel, T_mel);
    ggml_set_input(mel_in);
    ggml_set_name(mel_in, "hift_f0_mel_in");

    // Transpose to (T, C) for the chatterbox-pattern conv1d helpers.
    ggml_tensor* x = ggml_cont(ctx0, ggml_transpose(ctx0, mel_in)); // (T, mel)

    // Layer 0: lookahead conv (k=4, right-pad 3) — in=80, out=512.
    x = cv3_lookahead_conv1d(ctx0, x, h.f0_condnet_w[0], h.f0_condnet_b[0]); // (T, 512)
    x = ggml_elu(ctx0, x);

    // Layers 1-4: causal conv (k=3, left-pad 2) — in=512, out=512.
    for (int i = 1; i < 5; i++) {
        x = cv3_causal_conv1d(ctx0, x, h.f0_condnet_w[i], h.f0_condnet_b[i]);
        x = ggml_elu(ctx0, x);
    }

    // Linear classifier(512, 1). x is (T, 512) post-conv-helpers;
    // ggml_mul_mat(w, x) needs w->ne[0] == x->ne[0]. Transpose x to
    // (512, T) so the matmul produces (1, T), then squeeze + abs.
    x = ggml_cont(ctx0, ggml_transpose(ctx0, x));              // (cond_ch, T_mel)
    ggml_tensor* y = ggml_mul_mat(ctx0, h.f0_classifier_w, x); // (1, T_mel)
    y = ggml_add(ctx0, y, h.f0_classifier_b);
    y = ggml_abs(ctx0, y);
    // Reshape (1, T) → (T,) for the output buffer.
    y = ggml_reshape_1d(ctx0, y, T_mel);
    y = ggml_cont(ctx0, y);
    ggml_set_name(y, "hift_f0_out");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_free(ctx0);
    return gf;
}

float* cv3_extract_hift_f0_stage(cosyvoice3_tts_context* ctx, const float* mel, int T_mel) {
    if (!ctx || !ctx->hift.loaded || !mel || T_mel <= 0)
        return nullptr;
    const int mel_dim = (int)ctx->hift.hp.mel_dim;

    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;

    ggml_cgraph* gf = cv3_build_hift_f0_graph(ctx, T_mel);
    if (!gf)
        return nullptr;
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "cosyvoice3_tts: hift_f0 alloc_graph failed\n");
        return nullptr;
    }
    ggml_tensor* in_t = ggml_graph_get_tensor(gf, "hift_f0_mel_in");
    if (!in_t)
        return nullptr;
    ggml_backend_tensor_set(in_t, mel, 0, (size_t)mel_dim * T_mel * sizeof(float));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cosyvoice3_tts: hift_f0 compute failed\n");
        return nullptr;
    }
    ggml_tensor* out_t = ggml_graph_get_tensor(gf, "hift_f0_out");
    if (!out_t)
        return nullptr;
    const size_t n = (size_t)ggml_nelements(out_t);
    float* out = (float*)malloc(n * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(out_t, out, 0, n * sizeof(float));
    return out;
}

// ---------------------------------------------------------------------------
// Phase 4-B — HiFT decode forward (mel + s_stft → 24 kHz audio)
// ---------------------------------------------------------------------------
//
// Mirrors upstream `cosyvoice/hifigan/generator.py::CausalHiFTGenerator.decode`.
// The caller supplies the pre-computed source STFT (real + imag concatenated
// into (18, T_stft) at audio_rate / hop_len + 1 frames). This matches the
// "Option B" handover plan — isolating the big main-path implementation from
// the SineGen reproducibility question.
//
// Tensor layout: ggml (T, C) with T innermost matches PyTorch (B, C, T)
// row-major memory exactly when B=1. So we can dump ggml outputs straight
// to PyTorch (C, T) tensors for the diff.
//
// Stages exposed via cv3_extract_hift_decode_stage:
//   hift_decode_s_stft         caller input → (18, T_stft)
//   hift_decode_conv_pre_out   (512, T_mel)
//   hift_decode_post_stage_0_x (256, T_mel*8)
//   hift_decode_post_stage_1_x (128, T_mel*40)
//   hift_decode_post_stage_2_x (64,  T_mel*120 + 1)
//   hift_decode_conv_post_out  (18,  T_mel*120 + 1)
//   hift_decode_mag            (9,   T_mel*120 + 1)
//   hift_decode_phase          (9,   T_mel*120 + 1)
//   hift_decode                (T_mel * 480,) — final 24 kHz audio

// ResBlock causal conv with arbitrary dilation. Layout: x (T, C_in);
// w (K, C_in, C_out); b (C_out,). Output (T, C_out). Left-pads by (K-1)*dil.
ggml_tensor* cv3_causal_conv1d_dil(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int dilation) {
    const int K = (int)w->ne[0];
    const int T = (int)x->ne[0];
    const int pad = (K - 1) * dilation;
    ggml_tensor* y = ggml_conv_1d(ctx, w, x, /*stride*/ 1, /*pad*/ pad, /*dil*/ dilation);
    // ggml_conv_1d with symmetric pad produces length T + (K-1)*dil. Take
    // the LEFT T entries (drops causal pad from RIGHT) → causal conv.
    y = ggml_view_2d(ctx, y, T, (int)y->ne[1], y->nb[1], 0);
    y = ggml_cont(ctx, y);
    if (b) {
        ggml_tensor* b2d = ggml_reshape_2d(ctx, b, 1, (int)b->ne[0]);
        y = ggml_add(ctx, y, b2d);
    }
    return y;
}

// CausalConv1dDownSample: stride > 1, kernel = K, causal_padding = stride - 1.
// Pads LEFT by stride-1 with zeros, then strided unpadded conv. Layout
// x (T_in, C_in) → output (T_out, C_out) with T_out = (T_in + stride - 1 - K) / stride + 1.
ggml_tensor* cv3_causal_conv1d_downsample(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b,
                                          int stride) {
    const int lpad = stride - 1;
    ggml_tensor* xp = lpad > 0 ? ggml_pad_ext(ctx, x, lpad, 0, 0, 0, 0, 0, 0, 0) : x;
    ggml_tensor* y = ggml_conv_1d(ctx, w, xp, /*stride*/ stride, /*pad*/ 0, /*dil*/ 1);
    if (b) {
        ggml_tensor* b2d = ggml_reshape_2d(ctx, b, 1, (int)b->ne[0]);
        y = ggml_add(ctx, y, b2d);
    }
    return y;
}

// Snake-Beta activation (alpha_logscale=False): y = x + sin²(α·x) / (α + ε).
// Per-channel α. x (T, C); alpha (C,). Matches Python `Snake.no_div_by_zero=1e-9`.
ggml_tensor* cv3_snake(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha) {
    ggml_tensor* a = ggml_reshape_2d(ctx, alpha, 1, (int)alpha->ne[0]); // (1, C) broadcast
    ggml_tensor* ax = ggml_mul(ctx, x, a);
    ggml_tensor* s = ggml_sin(ctx, ax);
    ggml_tensor* s2 = ggml_mul(ctx, s, s);
    ggml_tensor* a_safe = ggml_scale_bias(ctx, a, 1.0f, 1e-9f);
    return ggml_add(ctx, x, ggml_div(ctx, s2, a_safe));
}

// Nearest-neighbor upsample along the T dimension by `scale`. x (T, C) → (T*scale, C).
// Implementation: reshape (1, T, C), repeat to (scale, T, C), flatten dim 0+1.
// Memory layout: ne=(scale, T, C) stores elements at offset = s + t*scale + c*T*scale,
// which equals the desired (T*scale, C) layout where each input row gets replicated
// `scale` times in sequence. Match torch.nn.Upsample(mode='nearest').
ggml_tensor* cv3_nearest_upsample_t(ggml_context* ctx, ggml_tensor* x, int scale) {
    if (scale <= 1)
        return x;
    const int T = (int)x->ne[0];
    const int C = (int)x->ne[1];
    ggml_tensor* r = ggml_reshape_3d(ctx, x, 1, T, C);
    ggml_tensor* y = ggml_repeat_4d(ctx, r, scale, T, C, 1);
    y = ggml_cont(ctx, y);
    return ggml_reshape_2d(ctx, y, T * scale, C);
}

// HiFT main ResBlock forward. 3 sub-blocks (snake1 → c1@dil → snake2 → c2 → residual).
// x (T, C). Returns (T, C). c1 uses per-sub-block dilation; c2 always dilation=1.
ggml_tensor* cv3_hift_resblock_fwd(ggml_context* ctx, ggml_tensor* x, const cv3_hift_resblock& rb,
                                   const int dilations[3]) {
    for (int j = 0; j < 3; j++) {
        ggml_tensor* xt = cv3_snake(ctx, x, rb.a1_alpha[j]);
        xt = cv3_causal_conv1d_dil(ctx, xt, rb.c1_w[j], rb.c1_b[j], dilations[j]);
        xt = cv3_snake(ctx, xt, rb.a2_alpha[j]);
        xt = cv3_causal_conv1d(ctx, xt, rb.c2_w[j], rb.c2_b[j]);
        x = ggml_add(ctx, x, xt);
    }
    return x;
}

// Overlap-add iSTFT (n_fft=16, hop_len=4, periodic Hann window). Matches
// torch.istft(complex(mag·cos(phase), mag·sin(phase)), n_fft=16, hop=4,
// window=hann(16, periodic=True), center=True). Output length = T_mel * 480
// (with the standard COLA reconstruction trimmed by n_fft/2 from the head).
//
// Input layout: conv_post_out is (T_stft, 18) in ggml memory order, which
// equals PyTorch (1, 18, T_stft) row-major. Bins 0..8 = log-magnitude raw;
// bins 9..17 = phase argument. We clip exp(mag) to 1e2 (matching upstream
// `_istft`).
std::vector<float> cv3_hift_istft(const float* conv_post_out, int T_stft, int T_mel, float audio_limit) {
    const int n_fft = 16;
    const int hop = 4;
    const int n_freq = n_fft / 2 + 1; // 9

    const int n_samples = (T_stft - 1) * hop + n_fft;
    std::vector<float> wav((size_t)n_samples, 0.0f);
    std::vector<float> win_sum((size_t)n_samples, 0.0f);

    std::vector<float> win(n_fft);
    for (int i = 0; i < n_fft; i++) {
        win[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i / (float)n_fft));
    }

    for (int frame = 0; frame < T_stft; frame++) {
        float re[9], im[9];
        for (int f = 0; f < n_freq; f++) {
            const float raw_mag = conv_post_out[(size_t)f * T_stft + (size_t)frame];
            const float raw_ph = conv_post_out[(size_t)(n_freq + f) * T_stft + (size_t)frame];
            const float mag = std::min(100.0f, std::exp(raw_mag));
            const float ph = std::sin(raw_ph);
            re[f] = mag * std::cos(ph);
            im[f] = mag * std::sin(ph);
        }
        const int start = frame * hop;
        // Half-spectrum IDFT with 2× factor on the interior bins. Exact for
        // real-valued time-domain signal (the conjugate symmetry is implicit).
        for (int n = 0; n < n_fft && (start + n) < n_samples; n++) {
            float sample = re[0];
            for (int f = 1; f < n_freq - 1; f++) {
                const float angle = 2.0f * (float)M_PI * (float)f * (float)n / (float)n_fft;
                sample += 2.0f * (re[f] * std::cos(angle) - im[f] * std::sin(angle));
            }
            const float angle_ny = 2.0f * (float)M_PI * (float)(n_freq - 1) * (float)n / (float)n_fft;
            sample += re[n_freq - 1] * std::cos(angle_ny) - im[n_freq - 1] * std::sin(angle_ny);
            sample /= (float)n_fft;
            wav[start + n] += sample * win[n];
            win_sum[start + n] += win[n] * win[n];
        }
    }
    for (int i = 0; i < n_samples; i++) {
        if (win_sum[i] > 1e-8f) {
            wav[i] /= win_sum[i];
        }
    }
    // center=True: trim n_fft/2 samples from the front.
    const int center_pad = n_fft / 2;
    const int final_len = T_mel * 480;
    std::vector<float> out;
    out.reserve((size_t)final_len);
    for (int i = 0; i < final_len; i++) {
        const int src = center_pad + i;
        const float v = (src >= 0 && src < (int)wav.size()) ? wav[src] : 0.0f;
        out.push_back(std::max(-audio_limit, std::min(audio_limit, v)));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Phase 4-B-1 — HiFT source path (CPU): f0 → upsample → SineGen2 → m_source
//                                        → STFT → s_stft (18, T_stft)
// ---------------------------------------------------------------------------
//
// Mirrors upstream `CausalHiFTGenerator.inference`'s pre-decode chain:
//
//   f0 [T_mel] → f0_upsamp(scale=480, nearest)  → f0_up [T_audio]
//             → SineGen2(f0_up):
//                 fn[t,h]        = f0_up[t] * (h+1)
//                 rad[t,h]       = (fn[t,h] / sr) % 1
//                 rad[0,:]      += rand_ini[:] (causal=True, seeded)
//                 rad_down       = interpolate(rad.T, scale=1/upsample_scale, mode=linear)
//                 phase_down     = cumsum(rad_down, dim=T) * 2π
//                 phase_up       = interpolate(phase_down*upsample_scale, scale=upsample_scale,
//                                              mode=nearest if causal else linear)
//                 sines          = sin(phase_up)
//                 uv             = (f0_up > 10) ? 1 : 0
//                 noise_amp      = uv*0.003 + (1-uv)*0.1/3
//                 noise          = noise_amp * noise_buf[:T_audio]   ← seeded uniform[0,1)
//                 sine_waves     = sines*0.1*uv + noise
//             → l_linear(sine_waves) + tanh                            → sine_merge [T_audio]
//             → STFT(sine_merge, n_fft=16, hop=4, win=hann_periodic,
//                    center=True)                                       → (9, T_stft) complex
//             → cat(real, imag, dim=0)                                  → s_stft (18, T_stft)
//
// All steps run CPU-side — the per-element work is small and threading is
// trivial. Bit-exact diff to PyTorch requires the caller to supply
// `rand_ini[9]` and `noise_buf[T_audio*9]` (deterministic uniform[0,1)
// buffers seeded by torch's `set_all_random_seed(0)`).
//
// Note: PyTorch's `F.interpolate(mode='linear', scale=1/480)` samples
// output index i at input position `(i+0.5)*480 - 0.5` (align_corners=False).
// For our nearest-replicated f0_up, neighboring input samples within a
// 480-block are identical, so the linear interpolation collapses to the
// f0_mel value. `rad_ini[h]` is added at f0_up index 0 only — it gets
// washed out by the centered linear interpolation (sampled at 239.5, never
// touches t=0), so we skip applying it here. Phase 4-B's diff confirms
// the residual error is below the cos ≥ 0.99 gate.

constexpr int CV3_HIFT_HARMONICS = 9;
constexpr int CV3_HIFT_UPSAMPLE_SCALE = 480;
constexpr int CV3_HIFT_SR = 24000;
constexpr float CV3_HIFT_SINE_AMP = 0.1f;
constexpr float CV3_HIFT_NOISE_STD = 0.003f;
constexpr float CV3_HIFT_VOICED_THR = 10.0f;
constexpr int CV3_HIFT_NFFT = 16;
constexpr int CV3_HIFT_HOP = 4;

// SineGen2 + m_source.l_linear + tanh, all CPU. Outputs row-major:
//   sine_waves_out  [T_audio, 9]  — post sin + noise mix (caller frees)
//   sine_merge_out  [T_audio]     — tanh(linear)         (caller frees)
//
// `l_linear_w` is the m_source.l_linear weight read straight from the GGUF
// (9 floats, row-major). `l_linear_b` is the scalar bias.
void cv3_hift_sinegen_msource_cpu(const float* f0_mel, int T_mel,
                                  const float* noise_buf, // [T_audio, 9]
                                  const float* l_linear_w, float l_linear_b,
                                  float* sine_waves_out, // [T_audio, 9]
                                  float* sine_merge_out, // [T_audio]
                                  float* f0_up_out)      // [T_audio] (optional, may be nullptr)
{
    const int T_audio = T_mel * CV3_HIFT_UPSAMPLE_SCALE;

    // f0_upsamp (nearest replicate by 480).
    std::vector<float> f0_up((size_t)T_audio);
    for (int t = 0; t < T_audio; t++) {
        f0_up[t] = f0_mel[t / CV3_HIFT_UPSAMPLE_SCALE];
    }
    if (f0_up_out) {
        std::memcpy(f0_up_out, f0_up.data(), (size_t)T_audio * sizeof(float));
    }

    // Compute downsampled rad values directly at T_mel rate. With the
    // nearest-replicated f0_up, linear-interpolate-downsample by 1/480
    // collapses to f0_mel[t] for t>=1; for t=0 the linear interp centers
    // at 239.5 — also f0_mel[0] (rand_ini contribution at f0_up[0] is
    // washed out, see header comment).
    std::vector<float> rad_down((size_t)T_mel * CV3_HIFT_HARMONICS);
    for (int t = 0; t < T_mel; t++) {
        const float f = f0_mel[t];
        for (int h = 0; h < CV3_HIFT_HARMONICS; h++) {
            float r = f * (float)(h + 1) / (float)CV3_HIFT_SR;
            r -= std::floor(r);
            rad_down[(size_t)t * CV3_HIFT_HARMONICS + h] = r;
        }
    }

    // cumsum along T per harmonic. Upstream then multiplies by upsample_scale
    // BEFORE the nearest upsample (`phase = phase * upsample_scale`), which
    // converts "per-T_mel-frame integrated phase" into "per-audio-sample
    // integrated phase up to that frame's start". We fold the * 480 in here.
    std::vector<float> phase_down((size_t)T_mel * CV3_HIFT_HARMONICS);
    const float two_pi_us = 2.0f * (float)M_PI * (float)CV3_HIFT_UPSAMPLE_SCALE;
    for (int h = 0; h < CV3_HIFT_HARMONICS; h++) {
        float acc = 0.0f;
        for (int t = 0; t < T_mel; t++) {
            acc += rad_down[(size_t)t * CV3_HIFT_HARMONICS + h];
            phase_down[(size_t)t * CV3_HIFT_HARMONICS + h] = acc * two_pi_us;
        }
    }

    // Upsample by 480 (nearest, causal=True) then sin. Compose with the
    // sine_amp scaling, uv mask, and noise injection in a single pass.
    for (int t = 0; t < T_audio; t++) {
        const int t_mel = t / CV3_HIFT_UPSAMPLE_SCALE;
        const float uv = (f0_up[t] > CV3_HIFT_VOICED_THR) ? 1.0f : 0.0f;
        const float noise_amp = uv * CV3_HIFT_NOISE_STD + (1.0f - uv) * CV3_HIFT_SINE_AMP / 3.0f;
        for (int h = 0; h < CV3_HIFT_HARMONICS; h++) {
            const float sine = std::sin(phase_down[(size_t)t_mel * CV3_HIFT_HARMONICS + h]) * CV3_HIFT_SINE_AMP * uv;
            const float noise = noise_amp * noise_buf[(size_t)t * CV3_HIFT_HARMONICS + h];
            sine_waves_out[(size_t)t * CV3_HIFT_HARMONICS + h] = sine + noise;
        }
    }

    // m_source.l_linear(sine_waves) + tanh.
    for (int t = 0; t < T_audio; t++) {
        float sum = l_linear_b;
        for (int h = 0; h < CV3_HIFT_HARMONICS; h++) {
            sum += sine_waves_out[(size_t)t * CV3_HIFT_HARMONICS + h] * l_linear_w[h];
        }
        sine_merge_out[t] = std::tanh(sum);
    }
}

// Source-side STFT: real signal (T_audio,) → (18, T_stft) row-major,
// matching the byte layout the decode forward expects (ggml ne=(T_stft, 18)).
// Parameters: n_fft=16, hop=4, periodic Hann window, center=True with
// reflect-pad (matches torch.stft default).
void cv3_hift_source_stft_cpu(const float* sine_merge, int T_audio, float* s_stft_out) {
    const int n_fft = CV3_HIFT_NFFT;
    const int hop = CV3_HIFT_HOP;
    const int n_freq = n_fft / 2 + 1;
    const int n_pad = n_fft / 2;
    const int T_stft = T_audio / hop + 1;

    // Reflect-pad: padded[i] = sine_merge[reflect(i - n_pad)] for i in [0, T_audio + 2*n_pad).
    // PyTorch's reflect pad mode mirrors around the boundary (so pos -1 = sine_merge[1]).
    const int T_padded = T_audio + 2 * n_pad;
    std::vector<float> padded((size_t)T_padded);
    auto reflect_at = [&](int idx) -> float {
        // idx in [0, T_padded); map to source signal index via reflect padding.
        int src = idx - n_pad;
        if (T_audio <= 1) {
            return T_audio == 1 ? sine_merge[0] : 0.0f;
        }
        const int period = 2 * (T_audio - 1);
        // Reduce src into [-(T_audio-1), T_audio-1] using full period 2*(T_audio-1).
        int r = src % period;
        if (r < 0)
            r += period;
        if (r >= T_audio)
            r = period - r;
        return sine_merge[r];
    };
    for (int i = 0; i < T_padded; i++) {
        padded[i] = reflect_at(i);
    }

    // Periodic Hann window.
    std::vector<float> win((size_t)n_fft);
    for (int i = 0; i < n_fft; i++) {
        win[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i / (float)n_fft));
    }

    // For each frame, naive DFT (n_fft=16 — O(N²) is trivially fine here).
    // s_stft_out layout: (T_stft, 18) row-major = bytes f-outer, t-inner →
    // index [f * T_stft + t]. First 9 bins = real, next 9 = imag.
    for (int frame = 0; frame < T_stft; frame++) {
        const int start = frame * hop;
        for (int f = 0; f < n_freq; f++) {
            float re = 0.0f;
            float im = 0.0f;
            for (int n = 0; n < n_fft; n++) {
                const float x = padded[start + n] * win[n];
                const float angle = -2.0f * (float)M_PI * (float)f * (float)n / (float)n_fft;
                re += x * std::cos(angle);
                im += x * std::sin(angle);
            }
            s_stft_out[(size_t)f * T_stft + (size_t)frame] = re;
            s_stft_out[(size_t)(n_freq + f) * T_stft + (size_t)frame] = im;
        }
    }
}

// Composes f0 → upsample → SineGen → l_linear → STFT into s_stft (18, T_stft).
// Caller supplies `noise_buf` (seeded; size T_audio * 9). All output buffers
// are caller-allocated; pass nullptr for any intermediate not needed.
void cv3_hift_source_path_cpu(const cv3_hift& h, const float* f0_mel, int T_mel, const float* noise_buf,
                              float* f0_up_out,      // [T_audio]      or nullptr
                              float* sine_waves_out, // [T_audio, 9]   or nullptr
                              float* sine_merge_out, // [T_audio]      or nullptr
                              float* s_stft_out)     // [T_stft, 18]   or nullptr
{
    const int T_audio = T_mel * CV3_HIFT_UPSAMPLE_SCALE;
    const int T_stft = T_audio / CV3_HIFT_HOP + 1;

    // Read l_linear weights off the loaded GGUF tensors.
    // m_source.l_linear.w shape is (9, 1) — Linear(in=9, out=1), stored as
    // ne=(9, 1) F32 (9 inner). Bytes: 9 weights then 1 bias.
    GGML_ASSERT(h.m_source_l_linear_w && h.m_source_l_linear_b);
    GGML_ASSERT(ggml_nelements(h.m_source_l_linear_w) == CV3_HIFT_HARMONICS);
    GGML_ASSERT(ggml_nelements(h.m_source_l_linear_b) == 1);
    float l_w[CV3_HIFT_HARMONICS];
    float l_b = 0.0f;
    ggml_backend_tensor_get(h.m_source_l_linear_w, l_w, 0, (size_t)CV3_HIFT_HARMONICS * sizeof(float));
    ggml_backend_tensor_get(h.m_source_l_linear_b, &l_b, 0, sizeof(float));

    std::vector<float> sine_waves_buf;
    std::vector<float> sine_merge_buf;
    float* sw_ptr = sine_waves_out;
    if (!sw_ptr) {
        sine_waves_buf.resize((size_t)T_audio * CV3_HIFT_HARMONICS);
        sw_ptr = sine_waves_buf.data();
    }
    float* sm_ptr = sine_merge_out;
    if (!sm_ptr) {
        sine_merge_buf.resize((size_t)T_audio);
        sm_ptr = sine_merge_buf.data();
    }

    cv3_hift_sinegen_msource_cpu(f0_mel, T_mel, noise_buf, l_w, l_b, sw_ptr, sm_ptr, f0_up_out);

    if (s_stft_out) {
        cv3_hift_source_stft_cpu(sm_ptr, T_audio, s_stft_out);
    }
    (void)T_stft;
}

// Build the HiFT decode graph. Inputs (set externally):
//   hift_decode_mel_in    [mel_dim, T_mel]              F32
//   hift_decode_s_stft_in [T_stft, 18]                  F32 — concat of real+imag
// T_stft must equal T_mel * 120 + 1 (= T_audio/hop + 1 with center=True).
//
// The graph stops at conv_post (pre-iSTFT). iSTFT runs CPU-side because it's
// a custom op not in ggml (overlap-add with window^2 normalization).
ggml_cgraph* cv3_build_hift_decode_graph(cosyvoice3_tts_context* ctx, int T_mel) {
    const auto& h = ctx->hift;
    const int mel = (int)h.hp.mel_dim;
    const int base = (int)h.hp.base_channels;
    const float slope = h.hp.lrelu_slope;
    const int n_fft = (int)h.hp.istft_n_fft;
    const int n_freq = n_fft / 2 + 1; // 9
    const int s_stft_ch = n_fft + 2;  // 18

    // Derived shapes (verified against upstream + GGUF tensors):
    //   T_audio   = T_mel * 480
    //   T_stft    = T_audio / hop + 1 = T_mel * 120 + 1   (torch.stft center=True)
    //   per-stage main T: T_mel*8, T_mel*40, T_mel*120 + 1 (last + reflection_pad)
    const int T_audio = T_mel * 480;
    const int T_stft = T_audio / (int)h.hp.istft_hop + 1; // = T_mel*120 + 1
    const int up_rates[3] = {(int)h.hp.upsample_rates[0], (int)h.hp.upsample_rates[1], (int)h.hp.upsample_rates[2]};
    const int down_strides[3] = {15, 3, 1}; // cumprod([3,5][::-1]) = [3,15] reversed → [15, 3], plus 1
    // (downsample_rates = [1]+[3,5], cum=[1,3,15], reversed → [15,3,1])

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    ggml_tensor* mel_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mel, T_mel);
    ggml_set_input(mel_in);
    ggml_set_name(mel_in, "hift_decode_mel_in");

    ggml_tensor* s_stft = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_stft, s_stft_ch);
    ggml_set_input(s_stft);
    ggml_set_name(s_stft, "hift_decode_s_stft_in");
    // Expose s_stft as a stage output for diff visibility.
    ggml_tensor* s_stft_out = ggml_cont(ctx0, s_stft);
    ggml_set_name(s_stft_out, "hift_decode_s_stft");
    ggml_set_output(s_stft_out);
    ggml_build_forward_expand(gf, s_stft_out);

    // ---- conv_pre — CausalConv1d k=5 right-pad 4 (causal_type='right') ----
    // Input (mel, T_mel) → transpose to (T_mel, mel) → lookahead conv (k=5,
    // 80→512) → (T_mel, 512).
    ggml_tensor* x = ggml_cont(ctx0, ggml_transpose(ctx0, mel_in)); // (T_mel, mel)
    x = cv3_lookahead_conv1d(ctx0, x, h.conv_pre_w, h.conv_pre_b);  // (T_mel, 512)
    {
        // Dump in PyTorch (C, T) layout. ggml (T, C) memory order already
        // matches PyTorch (C, T) byte-for-byte (T contiguous within each C).
        ggml_tensor* dump = ggml_cont(ctx0, x);
        ggml_set_name(dump, "hift_decode_conv_pre_out");
        ggml_set_output(dump);
        ggml_build_forward_expand(gf, dump);
    }

    // ---- 3-stage upsample tower with source fusion + main resblock fusion ----
    const int rb_kernels[3] = {3, 7, 11};
    const int dilations[3] = {1, 3, 5};
    const int src_rb_kernels[3] = {7, 7, 11};

    for (int i = 0; i < 3; i++) {
        const int u = up_rates[i];
        // ch_out = base >> (i + 1) = 256, 128, 64

        // LeakyReLU(slope=0.1) → CausalConv1dUpsample = Upsample(nearest, u) + CausalConv1d(K, left-pad K-1).
        x = ggml_leaky_relu(ctx0, x, slope, /*inplace*/ false);
        x = cv3_nearest_upsample_t(ctx0, x, u);
        x = cv3_causal_conv1d(ctx0, x, h.ups_w[i], h.ups_b[i]); // (T*u, ch_out)

        // ReflectionPad1d((1, 0)) only at the LAST stage. PyTorch reflects
        // ACROSS the boundary: pad position -1 holds x[1], not x[0]. So new
        // tensor T is T+1 with new[0]=x[1], new[1..T]=x[0..T-1].
        if (i == 2) {
            const int T_x = (int)x->ne[0];
            const int C_x = (int)x->ne[1];
            // Slice x[1:2, :] then prepend via concat along dim 0.
            ggml_tensor* head = ggml_view_2d(ctx0, x, 1, C_x, x->nb[1], 1 * x->nb[0]);
            head = ggml_cont(ctx0, head);
            x = ggml_concat(ctx0, head, x, 0); // (T_x + 1, C_x)
            (void)T_x;
        }

        // Source fusion: source_downs[i] → source_resblocks[i] → add.
        ggml_tensor* si;
        if (down_strides[i] > 1) {
            si = cv3_causal_conv1d_downsample(ctx0, s_stft, h.src_down_w[i], h.src_down_b[i], down_strides[i]);
        } else {
            // CausalConv1d k=1 left-pad 0 — degenerate, no padding needed.
            si = cv3_causal_conv1d(ctx0, s_stft, h.src_down_w[i], h.src_down_b[i]);
        }
        // Source resblock (single ResBlock per stage).
        si = cv3_hift_resblock_fwd(ctx0, si, h.src_resblocks[i], dilations);
        (void)src_rb_kernels; // kernels are baked into the loaded weights
        // Align T (defensive — the dim math should already match).
        {
            const int T_x = (int)x->ne[0];
            const int T_si = (int)si->ne[0];
            const int T_min = T_x < T_si ? T_x : T_si;
            if (T_si > T_min) {
                si = ggml_cont(ctx0, ggml_view_2d(ctx0, si, T_min, (int)si->ne[1], si->nb[1], 0));
            }
            if (T_x > T_min) {
                x = ggml_cont(ctx0, ggml_view_2d(ctx0, x, T_min, (int)x->ne[1], x->nb[1], 0));
            }
        }
        x = ggml_add(ctx0, x, si);

        // Main resblock fusion: 3 ResBlocks (kernel ∈ {3,7,11}), each applied
        // INDEPENDENTLY to the same x, outputs averaged: x = mean_j rb_j(x).
        ggml_tensor* xs = nullptr;
        ggml_tensor* rb_in = x;
        for (int j = 0; j < 3; j++) {
            const int K = rb_kernels[j];
            (void)K; // kernel baked into weights via cv3_causal_conv1d_dil
            ggml_tensor* rj = cv3_hift_resblock_fwd(ctx0, rb_in, h.resblocks[i * 3 + j], dilations);
            xs = xs ? ggml_add(ctx0, xs, rj) : rj;
        }
        x = ggml_scale(ctx0, xs, 1.0f / 3.0f);

        {
            ggml_tensor* dump = ggml_cont(ctx0, x);
            char name[64];
            std::snprintf(name, sizeof(name), "hift_decode_post_stage_%d_x", i);
            ggml_set_name(dump, name);
            ggml_set_output(dump);
            ggml_build_forward_expand(gf, dump);
        }
    }

    // ---- conv_post — CausalConv1d k=7 left-pad 6 ----
    // Upstream calls plain F.leaky_relu (no explicit slope → default 0.01)
    // BEFORE conv_post. The HiFT YAML lrelu_slope=0.1 is only used inside
    // the upsample loop.
    x = ggml_leaky_relu(ctx0, x, 0.01f, /*inplace*/ false);
    x = cv3_causal_conv1d(ctx0, x, h.conv_post_w, h.conv_post_b); // (T_stft, 18)
    ggml_tensor* conv_post_out = ggml_cont(ctx0, x);
    ggml_set_name(conv_post_out, "hift_decode_conv_post_out");
    ggml_set_output(conv_post_out);
    ggml_build_forward_expand(gf, conv_post_out);

    // Split out magnitude (first 9 channels) and phase (last 9). Dumped only
    // for diff localisation — they're not used further inside the graph (the
    // iSTFT runs CPU-side after compute).
    {
        const int T_out = (int)x->ne[0];
        // Channels are dim 1. Slice [:, 0:9] and [:, 9:18].
        ggml_tensor* mag_raw = ggml_view_2d(ctx0, x, T_out, n_freq, x->nb[1], 0);
        mag_raw = ggml_cont(ctx0, mag_raw);
        ggml_tensor* mag = ggml_exp(ctx0, mag_raw);
        // Clip mag to 1e2 (match upstream `_istft`).
        mag = ggml_clamp(ctx0, mag, 0.0f, 100.0f);
        ggml_set_name(mag, "hift_decode_mag");
        ggml_set_output(mag);
        ggml_build_forward_expand(gf, mag);

        ggml_tensor* phase_raw = ggml_view_2d(ctx0, x, T_out, n_freq, x->nb[1], (size_t)n_freq * x->nb[1]);
        phase_raw = ggml_cont(ctx0, phase_raw);
        ggml_tensor* phase = ggml_sin(ctx0, phase_raw);
        ggml_set_name(phase, "hift_decode_phase");
        ggml_set_output(phase);
        ggml_build_forward_expand(gf, phase);
    }

    ggml_free(ctx0);
    return gf;
}

// Run the HiFT decode graph + CPU iSTFT, returning the requested stage. For
// "hift_decode" returns the final 24 kHz audio (T_mel * 480 samples). For any
// of the named intermediates, returns the per-stage buffer.
//
// `s_stft_in` packs the 18-channel concatenation of STFT(real) + STFT(imag)
// in (T_stft, 18) row-major (same byte layout as ggml (T_stft, 18)).
float* cv3_extract_hift_decode_stage(cosyvoice3_tts_context* ctx, const float* mel, int T_mel, const float* s_stft_in,
                                     const char* stage_name, int* out_n) {
    if (!ctx || !ctx->hift.loaded || !mel || T_mel <= 0 || !s_stft_in || !stage_name || !out_n)
        return nullptr;
    *out_n = 0;
    const auto& h = ctx->hift;
    const int mel_dim = (int)h.hp.mel_dim;
    const int s_stft_ch = (int)h.hp.istft_n_fft + 2;
    const int T_stft = T_mel * 120 + 1;
    const int T_audio = T_mel * 480;
    const float audio_limit = h.hp.audio_limit;

    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;

    ggml_cgraph* gf = cv3_build_hift_decode_graph(ctx, T_mel);
    if (!gf)
        return nullptr;
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "cosyvoice3_tts: hift_decode alloc_graph failed\n");
        return nullptr;
    }
    ggml_tensor* mel_t = ggml_graph_get_tensor(gf, "hift_decode_mel_in");
    ggml_tensor* s_t = ggml_graph_get_tensor(gf, "hift_decode_s_stft_in");
    if (!mel_t || !s_t)
        return nullptr;
    ggml_backend_tensor_set(mel_t, mel, 0, (size_t)mel_dim * T_mel * sizeof(float));
    ggml_backend_tensor_set(s_t, s_stft_in, 0, (size_t)T_stft * s_stft_ch * sizeof(float));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cosyvoice3_tts: hift_decode compute failed\n");
        return nullptr;
    }

    // Final audio: run CPU iSTFT on conv_post output.
    if (strcmp(stage_name, "hift_decode") == 0) {
        ggml_tensor* cp = ggml_graph_get_tensor(gf, "hift_decode_conv_post_out");
        if (!cp)
            return nullptr;
        std::vector<float> tmp((size_t)ggml_nelements(cp));
        ggml_backend_tensor_get(cp, tmp.data(), 0, tmp.size() * sizeof(float));
        std::vector<float> wav = cv3_hift_istft(tmp.data(), T_stft, T_mel, audio_limit);
        float* out = (float*)malloc((size_t)T_audio * sizeof(float));
        if (!out)
            return nullptr;
        std::memcpy(out, wav.data(), (size_t)T_audio * sizeof(float));
        *out_n = T_audio;
        return out;
    }

    // Otherwise: just return the named graph output.
    ggml_tensor* out_t = ggml_graph_get_tensor(gf, stage_name);
    if (!out_t) {
        fprintf(stderr, "cosyvoice3_tts: hift_decode stage '%s' not found in graph\n", stage_name);
        return nullptr;
    }
    const size_t n = (size_t)ggml_nelements(out_t);
    float* out = (float*)malloc(n * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(out_t, out, 0, n * sizeof(float));
    *out_n = (int)n;
    return out;
}

// ---------------------------------------------------------------------------
// Phase 4-B-1 — source path stage extractor + end-to-end inference (4-C)
// ---------------------------------------------------------------------------
//
// Stages:
//   hift_source_f0_up        [T_audio]      nearest-replicate f0_mel by 480
//   hift_source_sine_waves   [T_audio, 9]   post sin + noise mix
//   hift_source_sine_merge   [T_audio]      tanh(l_linear)
//   hift_source              [T_stft, 18]   final STFT (same byte layout as
//                                            hift_decode_s_stft_in)
//
// `embeds_in` packs [f0_mel | noise_buf] with sizes T_mel + T_audio*9. The
// rand_ini buffer is NOT needed (its contribution is lost in the centered
// linear interpolation — see cv3_hift_sinegen_msource_cpu header).
float* cv3_extract_hift_source_stage(cosyvoice3_tts_context* ctx, const float* f0_mel, int T_mel,
                                     const float* noise_buf, const char* stage_name, int* out_n) {
    if (!ctx || !ctx->hift.loaded || !f0_mel || T_mel <= 0 || !noise_buf || !stage_name || !out_n)
        return nullptr;
    *out_n = 0;
    const auto& h = ctx->hift;
    const int T_audio = T_mel * 480;
    const int T_stft = T_audio / 4 + 1;

    if (strcmp(stage_name, "hift_source_f0_up") == 0) {
        float* out = (float*)malloc((size_t)T_audio * sizeof(float));
        if (!out)
            return nullptr;
        cv3_hift_source_path_cpu(h, f0_mel, T_mel, noise_buf, /*f0_up*/ out,
                                 /*sine_waves*/ nullptr, /*sine_merge*/ nullptr, /*s_stft*/ nullptr);
        *out_n = T_audio;
        return out;
    }
    if (strcmp(stage_name, "hift_source_sine_waves") == 0) {
        float* out = (float*)malloc((size_t)T_audio * 9 * sizeof(float));
        if (!out)
            return nullptr;
        cv3_hift_source_path_cpu(h, f0_mel, T_mel, noise_buf, nullptr, /*sine_waves*/ out, nullptr, nullptr);
        *out_n = T_audio * 9;
        return out;
    }
    if (strcmp(stage_name, "hift_source_sine_merge") == 0) {
        float* out = (float*)malloc((size_t)T_audio * sizeof(float));
        if (!out)
            return nullptr;
        cv3_hift_source_path_cpu(h, f0_mel, T_mel, noise_buf, nullptr, nullptr, /*sine_merge*/ out, nullptr);
        *out_n = T_audio;
        return out;
    }
    if (strcmp(stage_name, "hift_source") == 0) {
        const size_t n = (size_t)T_stft * 18;
        float* out = (float*)malloc(n * sizeof(float));
        if (!out)
            return nullptr;
        cv3_hift_source_path_cpu(h, f0_mel, T_mel, noise_buf, nullptr, nullptr, nullptr, /*s_stft*/ out);
        *out_n = (int)n;
        return out;
    }
    fprintf(stderr, "cosyvoice3_tts: unknown hift_source stage '%s'\n", stage_name);
    return nullptr;
}

// End-to-end Phase 4-C: mel → f0_predictor → source path → hift_decode →
// 24 kHz audio. Caller supplies seeded `noise_buf` (T_audio * 9 floats) for
// bit-equivalent diff to PyTorch's `set_all_random_seed(0)`. The F0 predictor
// is run via the graph builder from phase 4-A (`cv3_build_hift_f0_graph`),
// matching the ggml-resident path; the source path + decode run via the
// helpers above + `cv3_extract_hift_decode_stage`.
//
// Returns malloc'd float[T_mel * 480]. *out_n is set to T_audio on success.
float* cv3_extract_hift_inference(cosyvoice3_tts_context* ctx, const float* mel, int T_mel, const float* noise_buf,
                                  int* out_n) {
    if (!ctx || !ctx->hift.loaded || !mel || T_mel <= 0 || !noise_buf || !out_n)
        return nullptr;
    *out_n = 0;
    const int T_audio = T_mel * 480;
    const int T_stft = T_audio / 4 + 1;
    const auto& h = ctx->hift;

    // 1) F0 predictor (mel → f0_mel).
    float* f0_mel = cv3_extract_hift_f0_stage(ctx, mel, T_mel);
    if (!f0_mel) {
        fprintf(stderr, "cosyvoice3_tts: hift_inference: F0 predictor failed\n");
        return nullptr;
    }

    // 2) Source path: f0 → upsample → SineGen → l_linear → STFT.
    std::vector<float> s_stft((size_t)T_stft * 18);
    cv3_hift_source_path_cpu(h, f0_mel, T_mel, noise_buf, nullptr, nullptr, nullptr, s_stft.data());
    free(f0_mel);

    // 3) HiFT decode (mel + s_stft → audio).
    int dec_n = 0;
    float* audio = cv3_extract_hift_decode_stage(ctx, mel, T_mel, s_stft.data(), "hift_decode", &dec_n);
    if (!audio || dec_n != T_audio) {
        if (audio)
            free(audio);
        fprintf(stderr, "cosyvoice3_tts: hift_inference: decode forward failed (n=%d)\n", dec_n);
        return nullptr;
    }
    *out_n = T_audio;
    return audio;
}

} // namespace

extern "C" float* cosyvoice3_tts_run_hift_decode(struct cosyvoice3_tts_context* ctx, const float* mel, int T_mel,
                                                 const float* s_stft) {
    if (!ctx || !ctx->hift.loaded || !mel || T_mel <= 0 || !s_stft)
        return nullptr;
    int out_n = 0;
    return cv3_extract_hift_decode_stage(ctx, mel, T_mel, s_stft, "hift_decode", &out_n);
}

extern "C" float* cosyvoice3_tts_run_hift_source(struct cosyvoice3_tts_context* ctx, const float* f0_mel, int T_mel,
                                                 const float* noise_buf) {
    if (!ctx || !ctx->hift.loaded || !f0_mel || T_mel <= 0 || !noise_buf)
        return nullptr;
    int out_n = 0;
    return cv3_extract_hift_source_stage(ctx, f0_mel, T_mel, noise_buf, "hift_source", &out_n);
}

extern "C" float* cosyvoice3_tts_run_hift_inference(struct cosyvoice3_tts_context* ctx, const float* mel, int T_mel,
                                                    const float* noise_buf) {
    if (!ctx || !ctx->hift.loaded || !mel || T_mel <= 0 || !noise_buf)
        return nullptr;
    int out_n = 0;
    return cv3_extract_hift_inference(ctx, mel, T_mel, noise_buf, &out_n);
}

// Phase 4-A — HiFT loader. Binds the 246 hift GGUF tensors into ctx->hift
// (or specifically: 2 conv_pre, 2 conv_post, 6 ups, 162 main resblocks,
// 6 source_downs, 54 source resblocks, 2 m_source, 12 f0). The converter
// already resolved weight_norm so we read plain conv1d weights here.
extern "C" int cosyvoice3_tts_init_hift_from_file(struct cosyvoice3_tts_context* ctx, const char* path) {
    if (!ctx || !path) {
        fprintf(stderr, "cosyvoice3_tts: init_hift: bad args\n");
        return -1;
    }
    if (ctx->hift.loaded) {
        fprintf(stderr, "cosyvoice3_tts: hift already loaded\n");
        return 0;
    }

    // ---- Metadata pass ----
    ggml_context* gctx_dummy = nullptr;
    gguf_init_params gp = {/*no_alloc=*/true, &gctx_dummy};
    gguf_context* gctx = gguf_init_from_file(path, gp);
    if (!gctx) {
        fprintf(stderr, "cosyvoice3_tts: init_hift: failed to read GGUF '%s'\n", path);
        return -1;
    }
    auto& hp = ctx->hift.hp;
    hp.sample_rate = cv3_kv_u32(gctx, "cosyvoice3.hift.sample_rate", hp.sample_rate);
    hp.mel_dim = cv3_kv_u32(gctx, "cosyvoice3.hift.mel_dim", hp.mel_dim);
    hp.base_channels = cv3_kv_u32(gctx, "cosyvoice3.hift.base_channels", hp.base_channels);
    hp.nb_harmonics = cv3_kv_u32(gctx, "cosyvoice3.hift.nb_harmonics", hp.nb_harmonics);
    hp.istft_n_fft = cv3_kv_u32(gctx, "cosyvoice3.hift.istft_n_fft", hp.istft_n_fft);
    hp.istft_hop = cv3_kv_u32(gctx, "cosyvoice3.hift.istft_hop", hp.istft_hop);
    hp.n_upsample_stages = cv3_kv_u32(gctx, "cosyvoice3.hift.n_upsample_stages", hp.n_upsample_stages);
    for (uint32_t i = 0; i < hp.n_upsample_stages && i < 3; i++) {
        char key[64];
        snprintf(key, sizeof(key), "cosyvoice3.hift.upsample_rate.%u", i);
        hp.upsample_rates[i] = cv3_kv_u32(gctx, key, hp.upsample_rates[i]);
        snprintf(key, sizeof(key), "cosyvoice3.hift.upsample_kernel.%u", i);
        hp.upsample_kernels[i] = cv3_kv_u32(gctx, key, hp.upsample_kernels[i]);
    }
    gguf_free(gctx);

    // ---- Weight pass ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "cosyvoice3_tts:hift", wl)) {
        fprintf(stderr, "cosyvoice3_tts: init_hift: load_weights failed for '%s'\n", path);
        return -1;
    }
    ctx->hift.ctx_w = wl.ctx;
    ctx->hift.buf_w = wl.buf;
    ctx->hift.tensors = std::move(wl.tensors);

    auto require_t = [&](const std::string& name) -> ggml_tensor* {
        return core_gguf::require(ctx->hift.tensors, name.c_str(), "cosyvoice3_tts:hift");
    };

    auto& hf = ctx->hift;
    // Top-level convs.
    hf.conv_pre_w = require_t("cosyvoice3.hift.conv_pre.w");
    hf.conv_pre_b = require_t("cosyvoice3.hift.conv_pre.b");
    hf.conv_post_w = require_t("cosyvoice3.hift.conv_post.w");
    hf.conv_post_b = require_t("cosyvoice3.hift.conv_post.b");

    // Upsamples (3 stages for cv3).
    for (uint32_t i = 0; i < 3; i++) {
        char prefix[48];
        snprintf(prefix, sizeof(prefix), "cosyvoice3.hift.ups.%u", i);
        std::string p = prefix;
        hf.ups_w[i] = require_t(p + ".w");
        hf.ups_b[i] = require_t(p + ".b");
    }

    // 9 main ResBlocks.
    const int n_resblocks = 9;
    hf.resblocks.resize(n_resblocks);
    for (int i = 0; i < n_resblocks; i++) {
        for (int j = 0; j < 3; j++) {
            char prefix[64];
            snprintf(prefix, sizeof(prefix), "cosyvoice3.hift.resblocks.%d", i);
            std::string p = prefix;
            char idx[8];
            snprintf(idx, sizeof(idx), ".%d", j);
            std::string sj = idx;
            hf.resblocks[i].c1_w[j] = require_t(p + ".c1" + sj + ".w");
            hf.resblocks[i].c1_b[j] = require_t(p + ".c1" + sj + ".b");
            hf.resblocks[i].c2_w[j] = require_t(p + ".c2" + sj + ".w");
            hf.resblocks[i].c2_b[j] = require_t(p + ".c2" + sj + ".b");
            hf.resblocks[i].a1_alpha[j] = require_t(p + ".a1" + sj + ".alpha");
            hf.resblocks[i].a2_alpha[j] = require_t(p + ".a2" + sj + ".alpha");
        }
    }

    // 3 source_downs + 3 source ResBlocks.
    hf.src_resblocks.resize(3);
    for (int i = 0; i < 3; i++) {
        char prefix[48];
        snprintf(prefix, sizeof(prefix), "cosyvoice3.hift.source_downs.%d", i);
        std::string p = prefix;
        hf.src_down_w[i] = require_t(p + ".w");
        hf.src_down_b[i] = require_t(p + ".b");
        for (int j = 0; j < 3; j++) {
            char prefix2[64];
            snprintf(prefix2, sizeof(prefix2), "cosyvoice3.hift.src_resblk.%d", i);
            std::string p2 = prefix2;
            char idx[8];
            snprintf(idx, sizeof(idx), ".%d", j);
            std::string sj = idx;
            hf.src_resblocks[i].c1_w[j] = require_t(p2 + ".c1" + sj + ".w");
            hf.src_resblocks[i].c1_b[j] = require_t(p2 + ".c1" + sj + ".b");
            hf.src_resblocks[i].c2_w[j] = require_t(p2 + ".c2" + sj + ".w");
            hf.src_resblocks[i].c2_b[j] = require_t(p2 + ".c2" + sj + ".b");
            hf.src_resblocks[i].a1_alpha[j] = require_t(p2 + ".a1" + sj + ".alpha");
            hf.src_resblocks[i].a2_alpha[j] = require_t(p2 + ".a2" + sj + ".alpha");
        }
    }

    // SineGen source linear projection.
    hf.m_source_l_linear_w = require_t("cosyvoice3.hift.m_source.l_linear.w");
    hf.m_source_l_linear_b = require_t("cosyvoice3.hift.m_source.l_linear.b");

    // F0 predictor (CausalConvRNNF0Predictor): 5 condnet convs + classifier.
    for (int i = 0; i < 5; i++) {
        char prefix[48];
        snprintf(prefix, sizeof(prefix), "cosyvoice3.hift.f0.condnet.%d", i);
        std::string p = prefix;
        hf.f0_condnet_w[i] = require_t(p + ".w");
        hf.f0_condnet_b[i] = require_t(p + ".b");
    }
    hf.f0_classifier_w = require_t("cosyvoice3.hift.f0.classifier.w");
    hf.f0_classifier_b = require_t("cosyvoice3.hift.f0.classifier.b");

    hf.loaded = true;
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr,
                "cosyvoice3_tts:hift loaded %zu tensors  sr=%u mel=%u base_ch=%u "
                "n_fft=%u hop=%u up_stages=%u rates=[%u,%u,%u] kernels=[%u,%u,%u]\n",
                hf.tensors.size(), hp.sample_rate, hp.mel_dim, hp.base_channels, hp.istft_n_fft, hp.istft_hop,
                hp.n_upsample_stages, hp.upsample_rates[0], hp.upsample_rates[1], hp.upsample_rates[2],
                hp.upsample_kernels[0], hp.upsample_kernels[1], hp.upsample_kernels[2]);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Phase 6 — speech_tokenizer_v3 loader + native WAV token extraction
// ---------------------------------------------------------------------------

extern "C" int cosyvoice3_tts_init_s3tok_from_file(struct cosyvoice3_tts_context* ctx, const char* path) {
    if (!ctx || !path) {
        fprintf(stderr, "cosyvoice3_tts: init_s3tok: bad args\n");
        return -1;
    }
    if (ctx->s3tok.loaded) {
        fprintf(stderr, "cosyvoice3_tts: s3tok already loaded\n");
        return 0;
    }

    ggml_context* gctx_dummy = nullptr;
    gguf_init_params gp = {/*no_alloc=*/true, &gctx_dummy};
    gguf_context* gctx = gguf_init_from_file(path, gp);
    if (!gctx) {
        fprintf(stderr, "cosyvoice3_tts: init_s3tok: failed to read GGUF '%s'\n", path);
        return -1;
    }

    auto& hp = ctx->s3tok.hp;
    hp.n_audio_layer = cv3_kv_u32(gctx, "cosyvoice3.s3tok.n_blocks", hp.n_audio_layer);
    hp.n_audio_state = cv3_kv_u32(gctx, "cosyvoice3.s3tok.d_model", hp.n_audio_state);
    hp.n_audio_head = cv3_kv_u32(gctx, "cosyvoice3.s3tok.n_heads", hp.n_audio_head);
    hp.fsmn_kernel = cv3_kv_u32(gctx, "cosyvoice3.s3tok.fsmn_kernel", hp.fsmn_kernel);
    hp.n_codebook_size = cv3_kv_u32(gctx, "cosyvoice3.s3tok.codebook_size", hp.n_codebook_size);
    gguf_free(gctx);

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "cosyvoice3_tts:s3tok", wl)) {
        fprintf(stderr, "cosyvoice3_tts: init_s3tok: load_weights failed for '%s'\n", path);
        return -1;
    }
    ctx->s3tok.ctx_w = wl.ctx;
    ctx->s3tok.buf_w = wl.buf;
    ctx->s3tok.tensors = std::move(wl.tensors);

    auto require_t = [&](const std::string& name) -> ggml_tensor* {
        return core_gguf::require(ctx->s3tok.tensors, name.c_str(), "cosyvoice3_tts:s3tok");
    };

    auto& m = ctx->s3tok;
    m.conv0_w = require_t("cosyvoice3.s3tok.subsample.conv0.w");
    m.conv0_b = require_t("cosyvoice3.s3tok.subsample.conv0.b");
    m.conv1_w = require_t("cosyvoice3.s3tok.subsample.conv1.w");
    m.conv1_b = require_t("cosyvoice3.s3tok.subsample.conv1.b");
    m.blocks.resize(hp.n_audio_layer);
    for (uint32_t il = 0; il < hp.n_audio_layer; il++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "cosyvoice3.s3tok.blk.%u", il);
        std::string p = prefix;
        auto& b = m.blocks[il];
        b.attn_ln_w = require_t(p + ".attn_ln.w");
        b.attn_ln_b = require_t(p + ".attn_ln.b");
        b.attn_q_b = require_t(p + ".attn_q.b");
        b.attn_q_w = require_t(p + ".attn_q.w");
        b.attn_k_w = require_t(p + ".attn_k.w");
        b.attn_v_b = require_t(p + ".attn_v.b");
        b.attn_v_w = require_t(p + ".attn_v.w");
        b.attn_o_b = require_t(p + ".attn_o.b");
        b.attn_o_w = require_t(p + ".attn_o.w");
        b.mlp_ln_w = require_t(p + ".mlp_ln.w");
        b.mlp_ln_b = require_t(p + ".mlp_ln.b");
        b.mlp_up_b = require_t(p + ".mlp_up.b");
        b.mlp_up_w = require_t(p + ".mlp_up.w");
        b.mlp_dn_b = require_t(p + ".mlp_dn.b");
        b.mlp_dn_w = require_t(p + ".mlp_dn.w");
        b.fsmn_w = require_t(p + ".attn.fsmn_block.w");
        if (!b.attn_ln_w || !b.attn_q_w || !b.attn_k_w || !b.mlp_up_w || !b.fsmn_w) {
            fprintf(stderr, "cosyvoice3_tts: init_s3tok: missing tensors in %s.*\n", prefix);
            return -1;
        }
    }
    m.fsq_proj_w = require_t("cosyvoice3.s3tok.fsq.proj.w");
    m.fsq_proj_b = require_t("cosyvoice3.s3tok.fsq.proj.b");
    if (!m.conv0_w || !m.conv1_w || !m.fsq_proj_w) {
        fprintf(stderr, "cosyvoice3_tts: init_s3tok: missing core tensors\n");
        return -1;
    }

    m.loaded = true;
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "cosyvoice3_tts:s3tok loaded %zu tensors  blocks=%u d=%u h=%u kernel=%u codebook=%u\n",
                m.tensors.size(), hp.n_audio_layer, hp.n_audio_state, hp.n_audio_head, hp.fsmn_kernel,
                hp.n_codebook_size);
    }
    return 0;
}

extern "C" int cosyvoice3_tts_init_campplus_from_file(struct cosyvoice3_tts_context* ctx, const char* path) {
    if (!ctx || !path) {
        fprintf(stderr, "cosyvoice3_tts: init_campplus: bad args\n");
        return -1;
    }
    if (ctx->campplus.ctx_w) {
        fprintf(stderr, "cosyvoice3_tts: campplus already loaded\n");
        return 0;
    }

    ggml_context* gctx_dummy = nullptr;
    gguf_init_params gp = {/*no_alloc=*/true, &gctx_dummy};
    gguf_context* gctx = gguf_init_from_file(path, gp);
    if (!gctx) {
        fprintf(stderr, "cosyvoice3_tts: init_campplus: failed to read GGUF '%s'\n", path);
        return -1;
    }
    gguf_free(gctx);

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "cosyvoice3_tts:campplus", wl)) {
        fprintf(stderr, "cosyvoice3_tts: init_campplus: load_weights failed for '%s'\n", path);
        return -1;
    }
    ctx->campplus.ctx_w = wl.ctx;
    ctx->campplus.buf_w = wl.buf;

    auto T = [&](const char* name) -> ggml_tensor* { return core_gguf::try_get(wl.tensors, name); };
    auto& m = ctx->campplus.model;
    auto& head = m.head;
    head.conv1_w = T("s3.se.head.conv1.weight");
    head.conv1_b = T("s3.se.head.conv1.bias");
    head.bn1_w = T("s3.se.head.bn1.weight");
    head.bn1_b = T("s3.se.head.bn1.bias");
    head.bn1_m = T("s3.se.head.bn1.running_mean");
    head.bn1_v = T("s3.se.head.bn1.running_var");
    head.conv2_w = T("s3.se.head.conv2.weight");
    head.conv2_b = T("s3.se.head.conv2.bias");
    head.bn2_w = T("s3.se.head.bn2.weight");
    head.bn2_b = T("s3.se.head.bn2.bias");
    head.bn2_m = T("s3.se.head.bn2.running_mean");
    head.bn2_v = T("s3.se.head.bn2.running_var");
    head.layer1.assign(2, cb_campplus_resblock{});
    head.layer2.assign(2, cb_campplus_resblock{});
    for (int i = 0; i < 2; i++) {
        char base[64];
        std::snprintf(base, sizeof(base), "s3.se.head.layer1.%d", i);
        auto bind_resblock = [&](cb_campplus_resblock& b, const char* pfx) {
            char k[128];
            std::snprintf(k, sizeof(k), "%s.conv1.weight", pfx);
            b.conv1_w = T(k);
            std::snprintf(k, sizeof(k), "%s.conv1.bias", pfx);
            b.conv1_b = T(k);
            std::snprintf(k, sizeof(k), "%s.bn1.weight", pfx);
            b.bn1_w = T(k);
            std::snprintf(k, sizeof(k), "%s.bn1.bias", pfx);
            b.bn1_b = T(k);
            std::snprintf(k, sizeof(k), "%s.bn1.running_mean", pfx);
            b.bn1_m = T(k);
            std::snprintf(k, sizeof(k), "%s.bn1.running_var", pfx);
            b.bn1_v = T(k);
            std::snprintf(k, sizeof(k), "%s.conv2.weight", pfx);
            b.conv2_w = T(k);
            std::snprintf(k, sizeof(k), "%s.conv2.bias", pfx);
            b.conv2_b = T(k);
            std::snprintf(k, sizeof(k), "%s.bn2.weight", pfx);
            b.bn2_w = T(k);
            std::snprintf(k, sizeof(k), "%s.bn2.bias", pfx);
            b.bn2_b = T(k);
            std::snprintf(k, sizeof(k), "%s.bn2.running_mean", pfx);
            b.bn2_m = T(k);
            std::snprintf(k, sizeof(k), "%s.bn2.running_var", pfx);
            b.bn2_v = T(k);
            std::snprintf(k, sizeof(k), "%s.shortcut.0.weight", pfx);
            b.sc_w = T(k);
            std::snprintf(k, sizeof(k), "%s.shortcut.0.bias", pfx);
            b.sc_b = T(k);
            std::snprintf(k, sizeof(k), "%s.shortcut.1.weight", pfx);
            b.sc_bn_w = T(k);
            std::snprintf(k, sizeof(k), "%s.shortcut.1.bias", pfx);
            b.sc_bn_b = T(k);
            std::snprintf(k, sizeof(k), "%s.shortcut.1.running_mean", pfx);
            b.sc_bn_m = T(k);
            std::snprintf(k, sizeof(k), "%s.shortcut.1.running_var", pfx);
            b.sc_bn_v = T(k);
        };
        bind_resblock(head.layer1[i], base);
        std::snprintf(base, sizeof(base), "s3.se.head.layer2.%d", i);
        bind_resblock(head.layer2[i], base);
    }
    head.layer1[0].stride = 2;
    head.layer2[0].stride = 2;

    auto bind_unit = [&](cb_campplus_unit& u, const char* base) {
        char k[128];
        std::snprintf(k, sizeof(k), "%s.linear.weight", base);
        u.lin_w = T(k);
        std::snprintf(k, sizeof(k), "%s.linear.bias", base);
        u.lin_b = core_gguf::try_get(wl.tensors, k);
        std::snprintf(k, sizeof(k), "%s.nl.bn.weight", base);
        u.bn_w = T(k);
        std::snprintf(k, sizeof(k), "%s.nl.bn.bias", base);
        u.bn_b = T(k);
        std::snprintf(k, sizeof(k), "%s.nl.bn.running_mean", base);
        u.bn_m = T(k);
        std::snprintf(k, sizeof(k), "%s.nl.bn.running_var", base);
        u.bn_v = T(k);
    };
    bind_unit(m.tdnn, "s3.se.xv.tdnn");
    bind_unit(m.transit1, "s3.se.xv.transit1");
    bind_unit(m.transit2, "s3.se.xv.transit2");
    bind_unit(m.transit3, "s3.se.xv.transit3");
    m.out_nl.lin_w = nullptr;
    m.out_nl.lin_b = nullptr;
    m.out_nl.bn_w = T("s3.se.xv.out_nl.bn.weight");
    m.out_nl.bn_b = T("s3.se.xv.out_nl.bn.bias");
    m.out_nl.bn_m = T("s3.se.xv.out_nl.bn.running_mean");
    m.out_nl.bn_v = T("s3.se.xv.out_nl.bn.running_var");
    bind_unit(m.dense, "s3.se.xv.dense");

    const int block_layer_counts[3] = {12, 24, 16};
    const int block_dilations[3] = {1, 2, 2};
    cb_campplus_dense_block* blocks_ptr[3] = {&m.block1, &m.block2, &m.block3};
    for (int bi = 0; bi < 3; bi++) {
        auto& blk = *blocks_ptr[bi];
        blk.num_layers = block_layer_counts[bi];
        blk.dilation = block_dilations[bi];
        blk.layers.assign((size_t)blk.num_layers, cb_campplus_dense_layer{});
        for (int li = 0; li < blk.num_layers; li++) {
            char base[80];
            std::snprintf(base, sizeof(base), "s3.se.xv.block%d.tdnnd%d", bi + 1, li + 1);
            auto& l = blk.layers[(size_t)li];
            std::string pfx = base;
            l.nonl1_bn_w = T((pfx + ".nonl1.bn.weight").c_str());
            l.nonl1_bn_b = T((pfx + ".nonl1.bn.bias").c_str());
            l.nonl1_bn_m = T((pfx + ".nonl1.bn.running_mean").c_str());
            l.nonl1_bn_v = T((pfx + ".nonl1.bn.running_var").c_str());
            l.l1_w = T((pfx + ".l1.weight").c_str());
            l.l1_b = T((pfx + ".l1.bias").c_str());
            l.nonl2_bn_w = T((pfx + ".nonl2.bn.weight").c_str());
            l.nonl2_bn_b = T((pfx + ".nonl2.bn.bias").c_str());
            l.nonl2_bn_m = T((pfx + ".nonl2.bn.running_mean").c_str());
            l.nonl2_bn_v = T((pfx + ".nonl2.bn.running_var").c_str());
            l.cam_ll_w = T((pfx + ".cam.ll.weight").c_str());
            l.cam_l1_w = T((pfx + ".cam.l1.weight").c_str());
            l.cam_l1_b = T((pfx + ".cam.l1.bias").c_str());
            l.cam_l2_w = T((pfx + ".cam.l2.weight").c_str());
            l.cam_l2_b = T((pfx + ".cam.l2.bias").c_str());
        }
    }

    if (!m.head.conv1_w || !m.tdnn.lin_w || !m.dense.lin_w || m.block1.layers.empty()) {
        fprintf(stderr, "cosyvoice3_tts: init_campplus: missing core tensors\n");
        return -1;
    }

    ctx->campplus.loaded = true;
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "cosyvoice3_tts:campplus loaded %zu tensors\n", wl.tensors.size());
    }
    return 0;
}

// ===========================================================================
// Phase 5 — Voice cloning + end-to-end synth
// ===========================================================================

namespace {

// Tokenise a CV3 prompt fragment. The only special marker we expect in
// user-supplied prompt_text is `<|endofprompt|>`; everything around it
// is regular Qwen2 BPE. Splits on the literal substring and emits the
// special id (151646) between chunks.
std::vector<int32_t> cv3_tokenise_prompt(const cv3_vocab& v, const std::string& text) {
    std::vector<int32_t> ids;
    const std::string delim = "<|endofprompt|>";
    auto it_eop = v.token_to_id.find(delim);
    const int32_t eop_id = (it_eop != v.token_to_id.end()) ? it_eop->second : -1;
    size_t p = 0;
    while (p < text.size()) {
        size_t q = text.find(delim, p);
        if (q == std::string::npos) {
            auto chunk = core_bpe::tokenize_simple(v.token_to_id, v.merge_rank, text.substr(p));
            ids.insert(ids.end(), chunk.begin(), chunk.end());
            break;
        }
        if (q > p) {
            auto chunk = core_bpe::tokenize_simple(v.token_to_id, v.merge_rank, text.substr(p, q - p));
            ids.insert(ids.end(), chunk.begin(), chunk.end());
        }
        if (eop_id >= 0)
            ids.push_back(eop_id);
        p = q + delim.size();
    }
    return ids;
}

// Find a voice by name (case-sensitive). Returns nullptr if absent.
const cv3_voice* cv3_find_voice(const cv3_voices& vs, const std::string& name) {
    auto it = vs.by_name.find(name);
    if (it == vs.by_name.end())
        return nullptr;
    return &vs.voices[(size_t)it->second];
}

// Look up a tensor in `vs_tensors` and read its raw F32 / I32 data into
// the provided buffer. The voices.gguf converter writes everything as
// either F32 or I32 (no quantisation) so we can copy straight out of
// the backend buffer.
template <typename T>
bool cv3_voice_read_tensor(const std::map<std::string, ggml_tensor*>& tens, const std::string& name,
                           std::vector<T>& out) {
    auto it = tens.find(name);
    if (it == tens.end())
        return false;
    ggml_tensor* t = it->second;
    const size_t n = (size_t)ggml_nelements(t);
    if (ggml_nbytes(t) != n * sizeof(T))
        return false;
    out.resize(n);
    ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(T));
    return true;
}

std::string cv3_shell_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'')
            out += "'\"'\"'";
        else
            out.push_back(c);
    }
    out.push_back('\'');
    return out;
}

bool cv3_write_file(const std::string& path, const std::string& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open())
        return false;
    f.write(data.data(), (std::streamsize)data.size());
    return f.good();
}

std::string cv3_json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
                out += buf;
            } else {
                out.push_back((char)c);
            }
            break;
        }
    }
    return out;
}

std::string cv3_temp_path(const char* suffix) {
#ifndef _WIN32
    char tmpl[] = "/tmp/cv3_phase6_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0)
        return {};
    close(fd);
    std::remove(tmpl);
    std::string path = tmpl;
#else
    char buf[L_tmpnam];
    if (!std::tmpnam(buf))
        return {};
    std::string path = buf;
#endif
    if (suffix && *suffix)
        path += suffix;
    return path;
}

bool cv3_bake_runtime_voice_bundle(const char* wav_path, const char* ref_text, std::string& out_gguf_path) {
    const std::string manifest_path = cv3_temp_path(".json");
    const std::string gguf_path = cv3_temp_path(".gguf");
    if (manifest_path.empty() || gguf_path.empty())
        return false;

    const std::string upstream_base = "/Volumes/backups/code/cosyvoice3-stash/CosyVoice-upstream";
    std::ostringstream manifest;
    manifest << "[{\"name\":\"runtime\",\"wav\":\"" << cv3_json_escape(wav_path ? wav_path : "")
             << "\",\"prompt_text\":\"" << cv3_json_escape(ref_text ? ref_text : "") << "\"}]";
    if (!cv3_write_file(manifest_path, manifest.str())) {
        std::remove(manifest_path.c_str());
        std::remove(gguf_path.c_str());
        return false;
    }

#if defined(__APPLE__) && defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
    // system() is unavailable on iOS.
    (void)upstream_base;
    std::remove(manifest_path.c_str());
    std::remove(gguf_path.c_str());
    return false;
#else
    const std::string cmd = "python models/convert-cosyvoice3-voices-to-gguf.py --manifest " +
                            cv3_shell_quote(manifest_path) + " --upstream-base " + cv3_shell_quote(upstream_base) +
                            " --output " + cv3_shell_quote(gguf_path) +
#ifdef _WIN32
                            " >NUL 2>&1";
#else
                            " >/dev/null 2>&1";
#endif
    const int rc = std::system(cmd.c_str());
    std::remove(manifest_path.c_str());
    if (rc != 0) {
        std::remove(gguf_path.c_str());
        return false;
    }
#endif
    out_gguf_path = gguf_path;
    return true;
}

bool cv3_load_voice_bundle_from_file(const char* path, std::vector<cv3_voice>& voices,
                                     std::unordered_map<std::string, int>& by_name) {
    voices.clear();
    by_name.clear();

    ggml_context* gctx_dummy = nullptr;
    gguf_init_params gp = {/*no_alloc=*/true, &gctx_dummy};
    gguf_context* gctx = gguf_init_from_file(path, gp);
    if (!gctx)
        return false;

    auto names = core_gguf::kv_str_array(gctx, "voice.names");
    if (names.empty()) {
        gguf_free(gctx);
        return false;
    }
    std::vector<std::string> prompt_texts(names.size());
    for (size_t i = 0; i < names.size(); i++) {
        const std::string key = std::string("voice.") + names[i] + ".prompt_text";
        prompt_texts[i] = core_gguf::kv_str(gctx, key.c_str(), "");
    }
    gguf_free(gctx);

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend)
        return false;
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "cosyvoice3_tts:voices", wl)) {
        ggml_backend_free(backend);
        return false;
    }

    voices.reserve(names.size());
    for (size_t i = 0; i < names.size(); i++) {
        cv3_voice v;
        v.name = names[i];
        v.prompt_text = prompt_texts[i];
        const std::string prefix = "voice." + v.name;
        std::vector<int32_t> tokens;
        std::vector<float> spk_emb;
        std::vector<float> ref_mel;
        if (!cv3_voice_read_tensor(wl.tensors, prefix + ".prompt_speech_tokens", tokens) ||
            !cv3_voice_read_tensor(wl.tensors, prefix + ".spk_emb", spk_emb) ||
            !cv3_voice_read_tensor(wl.tensors, prefix + ".ref_mel", ref_mel)) {
            core_gguf::free_weights(wl);
            ggml_backend_free(backend);
            return false;
        }
        if (spk_emb.size() != 192 || ref_mel.size() % 80 != 0) {
            core_gguf::free_weights(wl);
            ggml_backend_free(backend);
            return false;
        }
        v.prompt_speech_tokens = std::move(tokens);
        v.spk_emb = std::move(spk_emb);
        v.t_ref_mel = (int)(ref_mel.size() / 80);
        v.ref_mel = std::move(ref_mel);
        by_name[v.name] = (int)voices.size();
        voices.push_back(std::move(v));
    }
    core_gguf::free_weights(wl);
    ggml_backend_free(backend);
    return !voices.empty();
}

bool cv3_load_runtime_voice(const char* wav_path, const char* ref_text, cv3_voice& out_voice) {
    std::string baked_path;
    if (!cv3_bake_runtime_voice_bundle(wav_path, ref_text, baked_path))
        return false;
    std::vector<cv3_voice> voices;
    std::unordered_map<std::string, int> by_name;
    const bool ok = cv3_load_voice_bundle_from_file(baked_path.c_str(), voices, by_name) && !voices.empty();
    std::remove(baked_path.c_str());
    if (!ok)
        return false;
    out_voice = std::move(voices.front());
    return true;
}

bool cv3_extract_native_runtime_voice(cosyvoice3_tts_context* ctx, const char* wav_path, const char* ref_text,
                                      cv3_voice& out_voice) {
    if (!ctx || !wav_path || !ref_text || !*ref_text)
        return false;
    if (!ctx->s3tok.loaded || !ctx->campplus.loaded)
        return false;

    std::vector<float> pcm;
    int sr = 0;
    if (!crispasr::core::read_wav_mono_pcm16(wav_path, pcm, sr))
        return false;

    std::vector<float> pcm16 = pcm;
    if (sr != 16000)
        pcm16 = core_audio::resample_polyphase(pcm.data(), (int)pcm.size(), sr, 16000);
    std::vector<float> pcm24 = pcm;
    if (sr != 24000)
        pcm24 = core_audio::resample_polyphase(pcm.data(), (int)pcm.size(), sr, 24000);

    std::vector<int32_t> native_tokens = cv3_tokenize_s3tok(ctx, pcm16.data(), (int)pcm16.size(), /*max_tokens*/ 0);
    if (native_tokens.empty())
        return false;

    std::vector<float> native_spk =
        chatterbox_campplus::embed_speaker(ctx->campplus.model, ctx->campplus.cache, pcm16.data(), (int)pcm16.size());
    if (native_spk.size() != 192)
        return false;

    int native_t_ref_mel = 0;
    std::vector<float> native_ref_mel = chatterbox_campplus::compute_prompt_feat_24k(
        pcm24.data(), (int)pcm24.size(), /*max_samples*/ 10 * 24000, native_t_ref_mel);
    if (native_ref_mel.empty() || native_t_ref_mel <= 0)
        return false;

    out_voice.name = "runtime";
    out_voice.prompt_text = ref_text;
    out_voice.prompt_speech_tokens = std::move(native_tokens);
    out_voice.spk_emb = std::move(native_spk);
    out_voice.ref_mel = std::move(native_ref_mel);
    out_voice.t_ref_mel = native_t_ref_mel;
    return true;
}

} // namespace

extern "C" int cosyvoice3_tts_init_voices_from_file(struct cosyvoice3_tts_context* ctx, const char* path) {
    if (!ctx || !path) {
        fprintf(stderr, "cosyvoice3_tts: init_voices: bad args\n");
        return -1;
    }
    std::vector<cv3_voice> voices;
    std::unordered_map<std::string, int> by_name;
    if (!cv3_load_voice_bundle_from_file(path, voices, by_name)) {
        fprintf(stderr, "cosyvoice3_tts: init_voices: failed to load GGUF '%s'\n", path);
        return -1;
    }
    ctx->voices.voices.clear();
    ctx->voices.by_name.clear();
    ctx->voices.voices = std::move(voices);
    ctx->voices.by_name = std::move(by_name);
    ctx->voices.loaded = true;
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "cosyvoice3_tts:voices loaded %zu voice(s):", ctx->voices.voices.size());
        for (const auto& v : ctx->voices.voices) {
            fprintf(stderr, " %s(T_tok=%zu,T_mel=%d)", v.name.c_str(), v.prompt_speech_tokens.size(), v.t_ref_mel);
        }
        fprintf(stderr, "\n");
    }
    return 0;
}

extern "C" int cosyvoice3_tts_n_voices(struct cosyvoice3_tts_context* ctx) {
    if (!ctx)
        return 0;
    return (int)ctx->voices.voices.size();
}

extern "C" const char* cosyvoice3_tts_voice_name(struct cosyvoice3_tts_context* ctx, int idx) {
    if (!ctx || idx < 0 || (size_t)idx >= ctx->voices.voices.size())
        return nullptr;
    return ctx->voices.voices[(size_t)idx].name.c_str();
}

// ---------------------------------------------------------------------------
// Synth pipeline glue
// ---------------------------------------------------------------------------

namespace {

// Concatenate token-embed lookups [sos_emb | text_embeds | task_id_emb |
// prompt_speech_embeds] into a single row-major (T_total, d_model) F32
// buffer that `cosyvoice3_tts_generate_tokens_from_embeds` can consume.
//
// Upstream CosyVoice3LM uses:
//   sos     = speech_embedding[speech_token_size + 0]  (id 6561)
//   task_id = speech_embedding[speech_token_size + 2]  (id 6563)
// — i.e. the SOS/task markers live in the SPEECH embedding table, not
// the text embedding table. (Vanilla Qwen2LM uses `llm_embedding`,
// which is a separate 2-row embedding that CosyVoice3LM drops.)
bool cv3_build_lm_input_embeds(cosyvoice3_tts_context* ctx, const std::vector<int32_t>& text_ids,
                               const std::vector<int32_t>& prompt_speech_ids, std::vector<float>& out_embeds,
                               int& out_n_tokens) {
    const int d = (int)ctx->hp.d_model;
    const int speech_codebook = (int)ctx->hp.speech_codebook;
    const int sos_id = speech_codebook + 0;  // 6561
    const int task_id = speech_codebook + 2; // 6563

    auto lookup_text = [&](const std::vector<int32_t>& ids, std::vector<float>& dst) -> bool {
        if (ids.empty()) {
            dst.clear();
            return true;
        }
        float* e = cosyvoice3_tts_embed_text(ctx, ids.data(), (int)ids.size());
        if (!e)
            return false;
        dst.assign(e, e + (size_t)ids.size() * (size_t)d);
        free(e);
        return true;
    };
    auto lookup_speech = [&](const std::vector<int32_t>& ids, std::vector<float>& dst) -> bool {
        if (ids.empty()) {
            dst.clear();
            return true;
        }
        float* e = cosyvoice3_tts_embed_speech(ctx, ids.data(), (int)ids.size());
        if (!e)
            return false;
        dst.assign(e, e + (size_t)ids.size() * (size_t)d);
        free(e);
        return true;
    };

    std::vector<int32_t> sos = {(int32_t)sos_id};
    std::vector<int32_t> taskv = {(int32_t)task_id};
    std::vector<float> sos_emb, text_emb, task_emb, prompt_emb;
    if (!lookup_speech(sos, sos_emb))
        return false;
    if (!lookup_text(text_ids, text_emb))
        return false;
    if (!lookup_speech(taskv, task_emb))
        return false;
    if (!lookup_speech(prompt_speech_ids, prompt_emb))
        return false;

    const int n_total = (int)(1 + text_ids.size() + 1 + prompt_speech_ids.size());
    out_embeds.resize((size_t)n_total * (size_t)d);
    size_t off = 0;
    auto append = [&](const std::vector<float>& src) {
        std::memcpy(out_embeds.data() + off, src.data(), src.size() * sizeof(float));
        off += src.size();
    };
    append(sos_emb);
    append(text_emb);
    append(task_emb);
    append(prompt_emb);
    out_n_tokens = n_total;
    return true;
}

// Stop-floor variant of cosyvoice3_tts_generate_tokens_from_embeds: AR
// decodes speech tokens until the sampled id is >= `stop_floor`. CV3's
// upstream considers every id in [speech_codebook, speech_vocab) a
// stop marker (the canonical set is [6561, 6760]). The greedy path is
// already restricted to the codebook range inside
// `cosyvoice3_tts_generate_tokens_from_embeds`, but the RAS sampler
// can land on a stop id — and we want to break on that.
std::vector<int32_t> cv3_generate_tokens_with_stop_floor(cosyvoice3_tts_context* ctx, const float* embeds, int n_tokens,
                                                         int max_tokens, int stop_floor) {
    std::vector<int32_t> out;
    const int speech_codebook = (int)ctx->hp.speech_codebook;
    const int speech_vocab = (int)ctx->hp.speech_vocab;
    const int max_steps = max_tokens > 0 ? max_tokens : (ctx->params.max_tokens > 0 ? ctx->params.max_tokens : 1500);

    // Right-size the fixed-shape T=1 graph to this request.  This preserves
    // graph reuse while avoiding attention over thousands of unused KV slots.
    if (!cv3_kv_init(ctx, n_tokens + max_steps))
        return out;
    cosyvoice3_tts_reset_kv(ctx);
    float* logits = cosyvoice3_tts_prefill_with_embeds(ctx, embeds, n_tokens, /*n_past*/ 0);
    if (!logits)
        return out;

    const bool greedy = !(ctx->params.temperature > 0.0f);
    int n_past = n_tokens;
    for (int step = 0; step < max_steps; step++) {
        int32_t pick = -1;
        if (greedy) {
            // Greedy in full vocab — let the model end naturally.
            int n_pick = speech_vocab;
            float bv = logits[0];
            pick = 0;
            for (int i = 1; i < n_pick; i++)
                if (logits[i] > bv) {
                    bv = logits[i];
                    pick = i;
                }
        } else {
            pick = cosyvoice3_tts_sample_ras(ctx, logits, out.empty() ? nullptr : out.data(), (int)out.size());
        }
        free(logits);
        if (pick < 0)
            break;
        if (pick >= stop_floor)
            break;
        if ((int32_t)pick >= speech_codebook) {
            // Defensive: pick is past codebook but below stop_floor —
            // shouldn't happen but skip rather than emit garbage.
            break;
        }
        out.push_back(pick);
        if (n_past + 1 > ctx->kv_max_ctx)
            break;
        logits = cosyvoice3_tts_step_speech(ctx, pick, n_past);
        if (!logits)
            break;
        n_past++;
    }
    return out;
}

// pre_la + repeat_interleave(token_mel_ratio=2) for a full token
// sequence. Returns mu = (T_mel = 2 * T_tok, mel_dim) row-major F32.
std::vector<float> cv3_run_pre_la_and_interleave(cosyvoice3_tts_context* ctx, const std::vector<int32_t>& tokens) {
    std::vector<float> mu;
    if (tokens.empty())
        return mu;
    const int mel = (int)ctx->flow.hp.mel_dim;
    const int T_tok = (int)tokens.size();
    const int ratio = (int)ctx->flow.hp.token_mel_ratio;
    float* pre_out = cv3_extract_pre_la_stage(ctx, tokens.data(), T_tok, "pre_la_out");
    if (!pre_out)
        return mu;
    // pre_out layout is (mel, T_tok) ggml ne == (T_tok, mel) row-major
    // flat buffer (per-time-step strides of `mel` floats).
    mu.resize((size_t)T_tok * (size_t)ratio * (size_t)mel);
    for (int t = 0; t < T_tok; t++) {
        for (int r = 0; r < ratio; r++) {
            int dst_t = t * ratio + r;
            std::memcpy(mu.data() + (size_t)dst_t * (size_t)mel, pre_out + (size_t)t * (size_t)mel,
                        (size_t)mel * sizeof(float));
        }
    }
    free(pre_out);
    return mu;
}

// Run F.normalize(spk_emb) → spk_affine via the existing in_pipe graph
// builder. We pass T_mel=1 + dummy zero buffers for the other inputs
// because the named output `in_pipe_spk` only depends on the spk input.
std::vector<float> cv3_run_spk_proj(cosyvoice3_tts_context* ctx, const std::vector<float>& spk_emb) {
    std::vector<float> out;
    const int mel = (int)ctx->flow.hp.mel_dim;
    std::vector<float> dummy_mel((size_t)mel, 0.0f);
    float* spk = cv3_extract_in_pipe_stage(ctx, dummy_mel.data(), /*T_mel*/ 1, spk_emb.data(), dummy_mel.data(),
                                           dummy_mel.data(), "in_pipe_spk");
    if (!spk)
        return out;
    out.assign(spk, spk + ctx->flow.hp.spk_dim_out);
    free(spk);
    return out;
}

// Seeded uniform[0, 1) noise buffer for the HiFT SineGen source path.
// Upstream calls `set_all_random_seed(0)` once per inference call, so
// we use seed=0 here for determinism.
std::vector<float> cv3_seeded_uniform_noise(size_t n_samples, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n_samples);
    for (size_t i = 0; i < n_samples; i++)
        v[i] = dist(rng);
    return v;
}

// Seeded standard-normal noise buffer for the CFM Euler ODE init.
// Upstream uses `torch.manual_seed(0); randn(1, 80, 50*300)[..., :T_mel]`.
// We do NOT match PyTorch's RNG bit-for-bit (different Box-Muller stream),
// but for inference quality this is irrelevant — the 10-step ODE
// converges to the same audio modulo a tiny perceptual jitter.
std::vector<float> cv3_seeded_gaussian(size_t n_samples, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n_samples);
    for (size_t i = 0; i < n_samples; i++)
        v[i] = dist(rng);
    return v;
}

float* cv3_synth_with_voice(cosyvoice3_tts_context* ctx, const char* text, const cv3_voice* voice, int* out_n_samples) {
    if (!ctx || !text || !voice || !out_n_samples)
        return nullptr;
    if (!ctx->flow.loaded || !ctx->hift.loaded) {
        fprintf(stderr, "cosyvoice3_tts: synth requires LLM + flow + hift to be loaded\n");
        return nullptr;
    }
    if (ctx->vocab.id_to_token.empty()) {
        fprintf(stderr, "cosyvoice3_tts: synth: LLM vocab not loaded (was the GGUF written without "
                        "tokenizer.ggml.tokens?)\n");
        return nullptr;
    }

    // ---- 0. Align prompt tokens + ref_mel to token_mel_ratio ----
    // Upstream frontend trims BOTH the prompt speech tokens and the prompt
    // mel so that len(ref_mel) == ratio * len(prompt_tokens):
    //   token_len = min(ref_mel_frames / ratio, n_prompt_tokens)
    // Skipping this leaves the flow's prompt region (ratio*n_tokens frames)
    // misaligned with the cond prefix (ref_mel frames), which de-energises
    // the generated mel — WAV clones from clips where matcha-mel frames !=
    // ratio*s3tok-tokens (i.e. anything much longer than the ~3 s baked
    // zero_shot prompt) come out ~14 dB quiet. The baked zero_shot voice is
    // already aligned (174 == 2*87) so this is a no-op there.
    const int mel_ratio = (int)ctx->flow.hp.token_mel_ratio;
    int prompt_token_len = (int)voice->prompt_speech_tokens.size();
    if (voice->t_ref_mel > 0 && mel_ratio > 0) {
        const int by_mel = voice->t_ref_mel / mel_ratio;
        if (by_mel < prompt_token_len)
            prompt_token_len = by_mel;
    }
    if (prompt_token_len < 0)
        prompt_token_len = 0;
    const std::vector<int32_t> prompt_tokens(voice->prompt_speech_tokens.begin(),
                                             voice->prompt_speech_tokens.begin() + prompt_token_len);
    const int aligned_t_ref_mel = prompt_token_len * mel_ratio;

    // ---- 1. Tokenise prompt_text + user_text ----
    std::vector<int32_t> text_ids;
    {
        cosyvoice3_bench_stage _b("tokenize");
        std::vector<int32_t> prompt_ids = cv3_tokenise_prompt(ctx->vocab, voice->prompt_text);
        std::vector<int32_t> user_ids = cv3_tokenise_prompt(ctx->vocab, std::string(text));
        text_ids.reserve(prompt_ids.size() + user_ids.size());
        text_ids.insert(text_ids.end(), prompt_ids.begin(), prompt_ids.end());
        text_ids.insert(text_ids.end(), user_ids.begin(), user_ids.end());
    }
    if (text_ids.empty()) {
        fprintf(stderr, "cosyvoice3_tts: synth: empty text after tokenisation\n");
        return nullptr;
    }
    if (ctx->params.verbosity >= 2) {
        fprintf(stderr, "cosyvoice3_tts: synth: voice='%s' text_ids=%zu prompt_speech_tokens=%zu\n",
                voice->name.c_str(), text_ids.size(), voice->prompt_speech_tokens.size());
    }

    // ---- 2. Build LM input embeddings + AR-decode speech tokens ----
    std::vector<float> lm_embeds;
    int n_lm = 0;
    if (!cv3_build_lm_input_embeds(ctx, text_ids, prompt_tokens, lm_embeds, n_lm))
        return nullptr;

    const int stop_floor = (int)ctx->hp.speech_codebook;
    int max_steps = ctx->params.max_tokens > 0 ? ctx->params.max_tokens : (int)text_ids.size() * 20;
    if (max_steps < 16)
        max_steps = 16;
    std::vector<int32_t> gen_tokens;
    {
        cosyvoice3_bench_stage _b("lm_ar_decode");
        gen_tokens = cv3_generate_tokens_with_stop_floor(ctx, lm_embeds.data(), n_lm, max_steps, stop_floor);
    }
    if (gen_tokens.empty()) {
        fprintf(stderr, "cosyvoice3_tts: synth: AR decode produced 0 tokens\n");
        return nullptr;
    }
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "cosyvoice3_tts: synth: generated %zu speech tokens\n", gen_tokens.size());
    }

    if (const char* dump = std::getenv("COSYVOICE3_DUMP_TOKENS")) {
        FILE* f = std::fopen(dump, "w");
        if (f) {
            std::fprintf(f, "text_ids(%zu):", text_ids.size());
            for (int32_t id : text_ids)
                std::fprintf(f, " %d", id);
            std::fprintf(f, "\nprompt_speech_tokens(%zu):", voice->prompt_speech_tokens.size());
            for (int32_t id : voice->prompt_speech_tokens)
                std::fprintf(f, " %d", id);
            std::fprintf(f, "\ngenerated_tokens(%zu):", gen_tokens.size());
            for (int32_t id : gen_tokens)
                std::fprintf(f, " %d", id);
            std::fprintf(f, "\n");
            std::fclose(f);
            fprintf(stderr, "cosyvoice3_tts: synth: wrote token dump to %s\n", dump);
        }
    }

    // ---- 3. Compose full speech-token sequence + run pre_la + repeat_interleave ----
    const size_t n_full_tokens = prompt_tokens.size() + gen_tokens.size();
    std::vector<float> mu;
    {
        cosyvoice3_bench_stage _b("pre_la+interleave");
        std::vector<int32_t> full_tokens;
        full_tokens.reserve(n_full_tokens);
        full_tokens.insert(full_tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
        full_tokens.insert(full_tokens.end(), gen_tokens.begin(), gen_tokens.end());
        mu = cv3_run_pre_la_and_interleave(ctx, full_tokens);
    }
    if (mu.empty())
        return nullptr;
    const int mel = (int)ctx->flow.hp.mel_dim;
    const int T_mel_total = (int)(n_full_tokens * (size_t)ctx->flow.hp.token_mel_ratio);
    const int T_ref_mel = aligned_t_ref_mel;
    if (T_ref_mel > T_mel_total) {
        fprintf(stderr, "cosyvoice3_tts: synth: voice ref_mel (%d) longer than full T_mel (%d) — corrupt voice\n",
                T_ref_mel, T_mel_total);
        return nullptr;
    }

    // ---- 4. Project speaker embedding ----
    std::vector<float> spks_proj = cv3_run_spk_proj(ctx, voice->spk_emb);
    if (spks_proj.empty())
        return nullptr;

    // ---- 5. Build cond: leading T_ref_mel slots = ref_mel, rest = 0 ----
    std::vector<float> cond((size_t)T_mel_total * (size_t)mel, 0.0f);
    if (T_ref_mel > 0) {
        std::memcpy(cond.data(), voice->ref_mel.data(), (size_t)T_ref_mel * (size_t)mel * sizeof(float));
    }

    // ---- 6. Seed x_init for the CFM Euler ODE ----
    std::vector<float> x_init = cv3_seeded_gaussian((size_t)T_mel_total * (size_t)mel, /*seed*/ 0);

    // ---- 7. Run flow Euler → mel ----
    int flow_steps = (int)ctx->flow.hp.cfm_n_steps;
    if (const char* env_steps = std::getenv("COSYVOICE3_FLOW_STEPS")) {
        char* end = nullptr;
        const long parsed = std::strtol(env_steps, &end, 10);
        if (end != env_steps && *end == '\0' && parsed >= 1 && parsed <= 100) {
            flow_steps = (int)parsed;
        } else if (ctx->params.verbosity >= 1) {
            fprintf(stderr, "cosyvoice3_tts: ignoring invalid COSYVOICE3_FLOW_STEPS='%s' (expected 1..100)\n",
                    env_steps);
        }
    }
    float* mel_full;
    {
        cosyvoice3_bench_stage _b("flow_euler");
        mel_full = cosyvoice3_tts_solve_flow_euler(ctx, mu.data(), T_mel_total, spks_proj.data(), cond.data(),
                                                   x_init.data(), flow_steps, ctx->flow.hp.cfm_inference_cfg_rate);
    }
    if (!mel_full)
        return nullptr;

    // ---- 8. Slice off the prompt-mel prefix ----
    const int T_mel_out = T_mel_total - T_ref_mel;
    std::vector<float> mel_out((size_t)T_mel_out * (size_t)mel);
    std::memcpy(mel_out.data(), mel_full + (size_t)T_ref_mel * (size_t)mel,
                (size_t)T_mel_out * (size_t)mel * sizeof(float));
    free(mel_full);

    // ---- 9. HiFT inference → 24 kHz audio ----
    float* audio;
    {
        cosyvoice3_bench_stage _b("hift_vocoder");
        std::vector<float> noise_buf = cv3_seeded_uniform_noise((size_t)T_mel_out * 480 * 9, /*seed*/ 0);
        audio = cosyvoice3_tts_run_hift_inference(ctx, mel_out.data(), T_mel_out, noise_buf.data());
    }
    if (!audio)
        return nullptr;

    *out_n_samples = T_mel_out * 480;
    return audio;
}

} // namespace

extern "C" float* cosyvoice3_tts_synth(struct cosyvoice3_tts_context* ctx, const char* text, const char* voice_name,
                                       int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;
    if (!ctx->flow.loaded || !ctx->hift.loaded || !ctx->voices.loaded) {
        fprintf(stderr, "cosyvoice3_tts: synth requires LLM + flow + hift + voices to be loaded\n");
        return nullptr;
    }
    if (ctx->vocab.id_to_token.empty()) {
        fprintf(stderr, "cosyvoice3_tts: synth: LLM vocab not loaded (was the GGUF written without "
                        "tokenizer.ggml.tokens?)\n");
        return nullptr;
    }
    std::string vname = voice_name ? voice_name : ctx->voices.voices.front().name;
    const cv3_voice* voice = cv3_find_voice(ctx->voices, vname);
    if (!voice) {
        fprintf(stderr, "cosyvoice3_tts: synth: voice '%s' not found (have %zu)\n", vname.c_str(),
                ctx->voices.voices.size());
        return nullptr;
    }
    return cv3_synth_with_voice(ctx, text, voice, out_n_samples);
}

extern "C" float* cosyvoice3_tts_synth_from_wav(struct cosyvoice3_tts_context* ctx, const char* text,
                                                const char* wav_path, const char* ref_text, int* out_n_samples) {
    if (!ctx || !text || !wav_path || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;
    if (!ref_text || !*ref_text) {
        fprintf(stderr, "cosyvoice3_tts: synth_from_wav requires --ref-text\n");
        return nullptr;
    }
    if (!ctx->flow.loaded || !ctx->hift.loaded) {
        fprintf(stderr, "cosyvoice3_tts: synth_from_wav requires LLM + flow + hift to be loaded\n");
        return nullptr;
    }
    cv3_voice voice;
    if (!cv3_extract_native_runtime_voice(ctx, wav_path, ref_text ? ref_text : "", voice) &&
        !cv3_load_runtime_voice(wav_path, ref_text ? ref_text : "", voice)) {
        fprintf(stderr, "cosyvoice3_tts: synth_from_wav: failed to bake runtime voice from '%s'\n", wav_path);
        return nullptr;
    }
    return cv3_synth_with_voice(ctx, text, &voice, out_n_samples);
}

extern "C" int32_t* cosyvoice3_tts_extract_speech_tokens(struct cosyvoice3_tts_context* ctx, const char* wav_path,
                                                         const char* ref_text, int* out_n_tokens) {
    if (out_n_tokens)
        *out_n_tokens = 0;
    if (!ctx || !wav_path || !out_n_tokens)
        return nullptr;
    if (ctx->s3tok.loaded) {
        std::vector<float> pcm;
        int sr = 0;
        if (!crispasr::core::read_wav_mono_pcm16(wav_path, pcm, sr))
            return nullptr;
        if (sr != 16000) {
            pcm = core_audio::resample_polyphase(pcm.data(), (int)pcm.size(), sr, 16000);
        }
        std::vector<int32_t> toks = cv3_tokenize_s3tok(ctx, pcm.data(), (int)pcm.size(), /*max_tokens*/ 0);
        if (toks.empty())
            return nullptr;
        int32_t* out = (int32_t*)malloc(toks.size() * sizeof(int32_t));
        if (!out)
            return nullptr;
        std::memcpy(out, toks.data(), toks.size() * sizeof(int32_t));
        *out_n_tokens = (int)toks.size();
        return out;
    }
    cv3_voice voice;
    if (!cv3_load_runtime_voice(wav_path, ref_text ? ref_text : "", voice))
        return nullptr;
    const size_t n = voice.prompt_speech_tokens.size();
    int32_t* out = (int32_t*)malloc(n * sizeof(int32_t));
    if (!out)
        return nullptr;
    std::memcpy(out, voice.prompt_speech_tokens.data(), n * sizeof(int32_t));
    *out_n_tokens = (int)n;
    return out;
}

extern "C" int cosyvoice3_tts_extract_spk_emb(struct cosyvoice3_tts_context* ctx, const char* wav_path,
                                              float out_spk_emb[192]) {
    if (!ctx || !wav_path || !out_spk_emb)
        return -1;
    if (ctx->campplus.loaded) {
        std::vector<float> pcm;
        int sr = 0;
        if (!crispasr::core::read_wav_mono_pcm16(wav_path, pcm, sr))
            return -1;
        if (sr != 16000)
            pcm = core_audio::resample_polyphase(pcm.data(), (int)pcm.size(), sr, 16000);
        auto emb =
            chatterbox_campplus::embed_speaker(ctx->campplus.model, ctx->campplus.cache, pcm.data(), (int)pcm.size());
        if (emb.size() != 192)
            return -1;
        std::memcpy(out_spk_emb, emb.data(), 192 * sizeof(float));
        return 0;
    }
    cv3_voice voice;
    if (!cv3_load_runtime_voice(wav_path, "", voice) || voice.spk_emb.size() != 192)
        return -1;
    std::memcpy(out_spk_emb, voice.spk_emb.data(), 192 * sizeof(float));
    return 0;
}

extern "C" float* cosyvoice3_tts_extract_ref_mel(struct cosyvoice3_tts_context* ctx, const char* wav_path,
                                                 const char* ref_text, int* out_T_mel) {
    if (out_T_mel)
        *out_T_mel = 0;
    if (!ctx || !wav_path || !out_T_mel)
        return nullptr;
    (void)ref_text;
    std::vector<float> pcm;
    int sr = 0;
    if (!crispasr::core::read_wav_mono_pcm16(wav_path, pcm, sr))
        return nullptr;
    if (sr != 24000) {
        pcm = core_audio::resample_polyphase(pcm.data(), (int)pcm.size(), sr, 24000);
    }
    int T_mel = 0;
    auto mel =
        chatterbox_campplus::compute_prompt_feat_24k(pcm.data(), (int)pcm.size(), /*max_samples*/ 10 * 24000, T_mel);
    if (mel.empty() || T_mel <= 0)
        return nullptr;
    float* out = (float*)malloc(mel.size() * sizeof(float));
    if (!out)
        return nullptr;
    std::memcpy(out, mel.data(), mel.size() * sizeof(float));
    *out_T_mel = T_mel;
    return out;
}
