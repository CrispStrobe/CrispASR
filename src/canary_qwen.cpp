// canary_qwen.cpp — nvidia/canary-qwen-2.5b ggml runtime
//
// SALM: FastConformer encoder (32L, d=1024) → linear projection (1024→2048)
// → Qwen3-1.7B LLM decoder with merged LoRA. English ASR only.
//
// Architecture:
//   Mel:           128 mels @ 16 kHz, n_fft=512, win=400, hop=160 (Hann)
//   Encoder:       32× FastConformer block (use_bias=True), d_model=1024,
//                  8 heads, head_dim=128, ff_dim=4096, conv kernel=9,
//                  8× temporal subsampling via dw_striding
//   Projection:    linear (1024 → 2048)
//   LLM:           Qwen3-1.7B — 28L, d=2048, GQA 16/8, head_dim=128,
//                  ff=6144 (SwiGLU), RoPE θ=1e6, RMSNorm ε=1e-6
//   Vocab:         151936 (Qwen3 BPE, tied embed/lm_head)
//
// Prompt format (verified byte-identical to NeMo's QwenPromptFormatter.
// encode_dialog: ids [151644,872,198,3167,3114,279,2701,25,220,<audio>,
// 151645,198,151644,77091,198]):
//   <|im_start|>user\nTranscribe the following: <|audio_pad|>×N<|im_end|>\n
//   <|im_start|>assistant\n
//
// NeMo blueprint: QwenPromptFormatter (NOT Qwen3). No system role, no BOS.
// No <|audio_start|>/<|audio_end|> framing — the model was never trained with
// those tokens.
//
// #247 ("Transcript"/"PASS" leak): NOT the framing tokens (removing them did
// not fix it) and NOT the prompt (it matches NeMo exactly). Root cause is a
// too-short audio window: the FastConformer subsamples 8x, so a window with
// only a few encoder frames (T_enc<=5, ~<=0.4 s) gives the Qwen3 LLM decoder
// no acoustic content to ground on, and it falls back to its language prior,
// echoing the task framing as meta words. NeMo's SALM.generate() is plain
// greedy with no suppress/min-length, so it echoes too — this is inherent to
// the model on degenerate input, and must be handled in the pipeline. See the
// degenerate-window gate + instruction-echo safety net in the transcribe path.

#include "canary_qwen.h"
#include "core/gpu_layers.h"
#include "core/crispasr_env.h"
#include "canary_qwen_echo.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "crispasr_imatrix.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include "core/attention.h"
#include "core/bpe.h"
#include "core/cpu_ops.h"
#include "core/fastconformer.h"
#include "core/ffn.h"
#include "core/gpu_backend_pref.h"
#include "core/mel.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// ===========================================================================
// Bench instrumentation — CANARY_QWEN_BENCH=1 for per-stage timings.
// ===========================================================================

static bool cq_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_CANARY_QWEN_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct cq_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit cq_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~cq_bench_stage() {
        if (!cq_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  canary_qwen_bench: %-22s %.2f ms\n", name, ms);
    }
};

static bool cq_debug_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("CRISPASR_CANARY_QWEN_DEBUG");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

// ===========================================================================
// Hyper-parameters
// ===========================================================================

struct canary_qwen_hparams {
    // Audio / mel
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 128;
    uint32_t n_fft = 512;
    uint32_t win_length = 400;
    uint32_t hop_length = 160;
    // Encoder (FastConformer)
    uint32_t enc_d_model = 1024;
    uint32_t enc_n_layers = 32;
    uint32_t enc_n_heads = 8;
    uint32_t enc_head_dim = 128;
    uint32_t enc_ff_dim = 4096;
    uint32_t subsampling_factor = 8;
    uint32_t subsampling_channels = 256;
    uint32_t conv_kernel = 9;
    uint32_t frame_dur_cs = 8;
    // LLM (Qwen3-1.7B)
    uint32_t llm_d_model = 2048;
    uint32_t llm_n_layers = 28;
    uint32_t llm_n_heads = 16;
    uint32_t llm_n_kv_heads = 8;
    uint32_t llm_head_dim = 128;
    uint32_t llm_ff_dim = 6144;
    float llm_rope_theta = 1e6f;
    float llm_rms_eps = 1e-6f;
    uint32_t llm_vocab_size = 151936;
};

// ===========================================================================
// Per-layer tensor containers
// ===========================================================================

// Encoder: reuse core_conformer shared layout (all biases populated).
using cq_pre_encode = core_conformer::PreEncodeWeights;

struct cq_enc_layer : core_conformer::BlockWeights {
    ggml_tensor *conv_bn_w = nullptr, *conv_bn_b = nullptr;
    ggml_tensor *conv_bn_rm = nullptr, *conv_bn_rv = nullptr;
};

// Projection
struct cq_proj {
    ggml_tensor* w = nullptr; // (2048, 1024)
    ggml_tensor* b = nullptr; // (2048,)
};

// LLM layer (Qwen3 — identical struct to qwen3_asr)
struct cq_llm_block {
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

struct cq_llm {
    ggml_tensor* token_embd_w = nullptr; // (151936, 2048)
    std::vector<cq_llm_block> blocks;
    ggml_tensor* output_norm_w = nullptr;
    ggml_tensor* output_w = nullptr; // (151936, 2048) — tied to token_embd
};

struct canary_qwen_model {
    canary_qwen_hparams hparams;

    ggml_tensor* mel_fb = nullptr;
    ggml_tensor* mel_window = nullptr;

    cq_pre_encode pre_encode;
    std::vector<cq_enc_layer> enc;
    cq_proj proj;
    cq_llm llm;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_backend_buffer_t buf_cpu = nullptr;

    // Q8_0 repack of the F16 conv pw1/pw2 weights (issue #81, CRISPASR_FC_PW_Q8)
    core_conformer::PwRepackBuf pw_q8;
    // Fused Q/K/V weight concat for the encoder blocks (CRISPASR_FC_FUSED_QKV)
    core_conformer::PwRepackBuf qkv_fused;

    std::map<std::string, ggml_tensor*> tensors;
};

struct canary_qwen_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
};

struct canary_qwen_context {
    canary_qwen_context_params params;

    canary_qwen_model model;
    canary_qwen_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // KV cache for LLM decoder
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;
    int kv_n_used = 0;

    int n_threads = 4;

    // Cached encoder graph — reused when T_mel matches.
    ggml_cgraph* cached_enc_gf = nullptr;
    ggml_context* cached_enc_ctx = nullptr;
    std::vector<uint8_t> cached_enc_meta;
    int cached_enc_T_mel = 0;

    // Sampling controls
    float decode_temperature = 0.0f;
    uint64_t decode_seed = 0;
    int beam_size = 1;
    // #292: decode cap; forwarded from --max-new-tokens when set, else this default.
    int max_new_tokens = 256;
};

// ===========================================================================
// Loader helpers
// ===========================================================================

#include "core/gguf_loader.h"
#include "core/ggml_cpu_backend.h"

static ggml_tensor* try_get(canary_qwen_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}

static ggml_tensor* require(canary_qwen_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "canary_qwen");
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool canary_qwen_load_model(canary_qwen_model& model, canary_qwen_vocab& vocab, const char* path,
                                   ggml_backend_t backend, ggml_backend_t backend_cpu) {
    // ---- pass 1: hparams + vocab ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;

        auto& hp = model.hparams;
        hp.sample_rate = core_gguf::kv_u32(gctx, "canary_qwen.sample_rate", hp.sample_rate);
        hp.n_mels = core_gguf::kv_u32(gctx, "canary_qwen.n_mels", hp.n_mels);
        hp.n_fft = core_gguf::kv_u32(gctx, "canary_qwen.n_fft", hp.n_fft);
        hp.win_length = core_gguf::kv_u32(gctx, "canary_qwen.win_length", hp.win_length);
        hp.hop_length = core_gguf::kv_u32(gctx, "canary_qwen.hop_length", hp.hop_length);
        hp.enc_d_model = core_gguf::kv_u32(gctx, "canary_qwen.enc_d_model", hp.enc_d_model);
        hp.enc_n_layers = core_gguf::kv_u32(gctx, "canary_qwen.enc_n_layers", hp.enc_n_layers);
        hp.enc_n_heads = core_gguf::kv_u32(gctx, "canary_qwen.enc_n_heads", hp.enc_n_heads);
        hp.enc_head_dim = core_gguf::kv_u32(gctx, "canary_qwen.enc_head_dim", hp.enc_head_dim);
        hp.enc_ff_dim = core_gguf::kv_u32(gctx, "canary_qwen.enc_ff_dim", hp.enc_ff_dim);
        hp.subsampling_factor = core_gguf::kv_u32(gctx, "canary_qwen.subsampling_factor", hp.subsampling_factor);
        hp.subsampling_channels = core_gguf::kv_u32(gctx, "canary_qwen.subsampling_channels", hp.subsampling_channels);
        hp.conv_kernel = core_gguf::kv_u32(gctx, "canary_qwen.conv_kernel", hp.conv_kernel);
        hp.frame_dur_cs = core_gguf::kv_u32(gctx, "canary_qwen.frame_dur_cs", hp.frame_dur_cs);

        hp.llm_d_model = core_gguf::kv_u32(gctx, "canary_qwen.llm_d_model", hp.llm_d_model);
        hp.llm_n_layers = core_gguf::kv_u32(gctx, "canary_qwen.llm_n_layers", hp.llm_n_layers);
        hp.llm_n_heads = core_gguf::kv_u32(gctx, "canary_qwen.llm_n_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = core_gguf::kv_u32(gctx, "canary_qwen.llm_n_kv_heads", hp.llm_n_kv_heads);
        hp.llm_head_dim = core_gguf::kv_u32(gctx, "canary_qwen.llm_head_dim", hp.llm_head_dim);
        hp.llm_ff_dim = core_gguf::kv_u32(gctx, "canary_qwen.llm_ff_dim", hp.llm_ff_dim);
        hp.llm_rope_theta = core_gguf::kv_f32(gctx, "canary_qwen.llm_rope_theta", hp.llm_rope_theta);
        hp.llm_rms_eps = core_gguf::kv_f32(gctx, "canary_qwen.llm_rms_norm_eps", hp.llm_rms_eps);
        hp.llm_vocab_size = core_gguf::kv_u32(gctx, "canary_qwen.llm_vocab_size", hp.llm_vocab_size);

        // Vocab
        auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            vocab.id_to_token = std::move(tokens);
            vocab.token_to_id.reserve(vocab.id_to_token.size());
            for (int i = 0; i < (int)vocab.id_to_token.size(); i++) {
                vocab.token_to_id[vocab.id_to_token[i]] = i;
            }
        }

        // Patch Qwen3 special tokens
        struct SpecialTok {
            int id;
            const char* text;
        };
        static const SpecialTok specials[] = {
            {151643, "<|endoftext|>"},   {151644, "<|im_start|>"},  {151645, "<|im_end|>"},
            {151669, "<|audio_start|>"}, {151670, "<|audio_end|>"}, {151676, "<|audio_pad|>"},
        };
        for (const auto& sp : specials) {
            if (sp.id < (int)vocab.id_to_token.size()) {
                auto old_it = vocab.token_to_id.find(vocab.id_to_token[sp.id]);
                if (old_it != vocab.token_to_id.end() && old_it->second == sp.id)
                    vocab.token_to_id.erase(old_it);
                vocab.id_to_token[sp.id] = sp.text;
                vocab.token_to_id[sp.text] = sp.id;
            }
        }

        // BPE merges
        auto merges = core_gguf::kv_str_array(gctx, "tokenizer.ggml.merges");
        for (int i = 0; i < (int)merges.size(); i++) {
            vocab.merge_rank[merges[i]] = i;
        }

        core_gguf::free_metadata(gctx);
    }

    // ---- pass 2: tensor data ----
    core_gguf::WeightLoad wl;
    // Session n_gpu_layers, else CRISPASR_N_GPU_LAYERS (#610).
    const int n_gpu_layers_env = core_gpu_layers::resolve();
    const int total_layers = (int)model.hparams.llm_n_layers;
    const bool do_split =
        backend_cpu && backend_cpu != backend && n_gpu_layers_env >= 0 && n_gpu_layers_env < total_layers;
    if (do_split) {
        int threshold = n_gpu_layers_env;
        if (!core_gguf::load_weights_split(path, backend, backend_cpu, core_gguf::is_gpu_tensor_blk, &threshold,
                                           "canary_qwen", wl))
            return false;
        fprintf(stderr, "canary_qwen: layer offload: gpu=[0,%d), cpu=[%d,%d)\n", n_gpu_layers_env, n_gpu_layers_env,
                total_layers);
    } else {
        if (!core_gguf::load_weights(path, backend, "canary_qwen", wl))
            return false;
    }
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.buf_cpu = wl.buf_cpu;
    model.tensors = std::move(wl.tensors);

    // ---- bind named tensors ----
    // Preprocessor
    model.mel_fb = try_get(model, "preprocessor.fb");
    model.mel_window = try_get(model, "preprocessor.window");

    // Pre-encode (dw_striding — identical naming to canary)
    model.pre_encode.conv0_w = require(model, "encoder.pre.conv.0.weight");
    model.pre_encode.conv0_b = require(model, "encoder.pre.conv.0.bias");
    model.pre_encode.conv2_w = require(model, "encoder.pre.conv.2.weight");
    model.pre_encode.conv2_b = require(model, "encoder.pre.conv.2.bias");
    model.pre_encode.conv3_w = try_get(model, "encoder.pre.conv.3.weight");
    model.pre_encode.conv3_b = try_get(model, "encoder.pre.conv.3.bias");
    model.pre_encode.conv5_w = require(model, "encoder.pre.conv.5.weight");
    model.pre_encode.conv5_b = require(model, "encoder.pre.conv.5.bias");
    model.pre_encode.conv6_w = try_get(model, "encoder.pre.conv.6.weight");
    model.pre_encode.conv6_b = try_get(model, "encoder.pre.conv.6.bias");
    model.pre_encode.out_w = require(model, "encoder.pre.out.weight");
    model.pre_encode.out_b = require(model, "encoder.pre.out.bias");

    // Encoder layers
    model.enc.resize(model.hparams.enc_n_layers);
    for (uint32_t i = 0; i < model.hparams.enc_n_layers; i++) {
        char buf[128];
        auto& e = model.enc[i];
        auto get = [&](const char* suf) {
            snprintf(buf, sizeof(buf), "encoder.layers.%u.%s", i, suf);
            return require(model, buf);
        };
        auto get_opt = [&](const char* suf) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "encoder.layers.%u.%s", i, suf);
            return try_get(model, buf);
        };

        e.norm_ff1_w = get("norm_ff1.weight");
        e.norm_ff1_b = get("norm_ff1.bias");
        e.ff1_l1_w = get("ff1.linear1.weight");
        e.ff1_l1_b = get("ff1.linear1.bias");
        e.ff1_l2_w = get("ff1.linear2.weight");
        e.ff1_l2_b = get("ff1.linear2.bias");

        e.norm_attn_w = get("norm_attn.weight");
        e.norm_attn_b = get("norm_attn.bias");
        e.attn_q_w = get("attn.q.weight");
        e.attn_q_b = get("attn.q.bias");
        e.attn_k_w = get("attn.k.weight");
        e.attn_k_b = get("attn.k.bias");
        e.attn_v_w = get("attn.v.weight");
        e.attn_v_b = get("attn.v.bias");
        e.attn_out_w = get("attn.out.weight");
        e.attn_out_b = get("attn.out.bias");
        e.attn_pos_w = get("attn.pos.weight");
        e.pos_bias_u = get("attn.pos_bias_u");
        e.pos_bias_v = get("attn.pos_bias_v");

        e.norm_conv_w = get("norm_conv.weight");
        e.norm_conv_b = get("norm_conv.bias");
        e.conv_pw1_w = get("conv.pw1.weight");
        e.conv_pw1_b = get("conv.pw1.bias");
        e.conv_dw_w = get("conv.dw.weight");
        e.conv_dw_b = get("conv.dw.bias");
        e.conv_pw2_w = get("conv.pw2.weight");
        e.conv_pw2_b = get("conv.pw2.bias");
        e.conv_bn_w = get_opt("conv.bn.weight");
        e.conv_bn_b = get_opt("conv.bn.bias");
        e.conv_bn_rm = get_opt("conv.bn.running_mean");
        e.conv_bn_rv = get_opt("conv.bn.running_var");

        e.norm_ff2_w = get("norm_ff2.weight");
        e.norm_ff2_b = get("norm_ff2.bias");
        e.ff2_l1_w = get("ff2.linear1.weight");
        e.ff2_l1_b = get("ff2.linear1.bias");
        e.ff2_l2_w = get("ff2.linear2.weight");
        e.ff2_l2_b = get("ff2.linear2.bias");

        e.norm_out_w = get("norm_out.weight");
        e.norm_out_b = get("norm_out.bias");
    }

    // Projection (encoder 1024 → LLM 2048)
    model.proj.w = require(model, "proj.weight");
    model.proj.b = require(model, "proj.bias");

    // LLM
    model.llm.token_embd_w = require(model, "token_embd.weight");
    model.llm.output_norm_w = require(model, "output_norm.weight");
    model.llm.output_w = require(model, "output.weight");
    model.llm.blocks.resize(model.hparams.llm_n_layers);
    for (uint32_t i = 0; i < model.hparams.llm_n_layers; i++) {
        char buf[128];
        auto& b = model.llm.blocks[i];
        auto get = [&](const char* suf) {
            snprintf(buf, sizeof(buf), "blk.%u.%s", i, suf);
            return require(model, buf);
        };
        b.attn_norm_w = get("attn_norm.weight");
        b.attn_q_w = get("attn_q.weight");
        b.attn_k_w = get("attn_k.weight");
        b.attn_v_w = get("attn_v.weight");
        b.attn_output_w = get("attn_output.weight");
        b.attn_q_norm_w = get("attn_q_norm.weight");
        b.attn_k_norm_w = get("attn_k_norm.weight");
        b.ffn_norm_w = get("ffn_norm.weight");
        b.ffn_gate_w = get("ffn_gate.weight");
        b.ffn_up_w = get("ffn_up.weight");
        b.ffn_down_w = get("ffn_down.weight");
    }

    // BN folding for encoder conv layers (same as canary.cpp)
    {
        const int d = (int)model.hparams.enc_d_model;
        const int K = (int)model.hparams.conv_kernel;
        const float eps = 1e-5f;
        for (uint32_t il = 0; il < model.hparams.enc_n_layers; il++) {
            auto& e = model.enc[il];
            if (!e.conv_dw_w || !e.conv_dw_b || !e.conv_bn_w || !e.conv_bn_b || !e.conv_bn_rm || !e.conv_bn_rv)
                continue;
            std::vector<float> bn_mean(d), bn_var(d), bn_w(d), bn_b(d), dw_b(d);
            ggml_backend_tensor_get(e.conv_bn_rm, bn_mean.data(), 0, d * sizeof(float));
            ggml_backend_tensor_get(e.conv_bn_rv, bn_var.data(), 0, d * sizeof(float));
            ggml_backend_tensor_get(e.conv_bn_w, bn_w.data(), 0, d * sizeof(float));
            ggml_backend_tensor_get(e.conv_bn_b, bn_b.data(), 0, d * sizeof(float));
            ggml_backend_tensor_get(e.conv_dw_b, dw_b.data(), 0, d * sizeof(float));
            std::vector<float> s(d);
            for (int c = 0; c < d; c++)
                s[c] = bn_w[c] / sqrtf(bn_var[c] + eps);
            std::vector<float> w_f32 = core_cpu::to_f32(e.conv_dw_w);
            for (int c = 0; c < d; c++)
                for (int ki = 0; ki < K; ki++)
                    w_f32[ki + c * K] *= s[c];
            // Write back in native dtype — don't clobber an F32 conv_dw_w with F16 (c9a4de65).
            if (e.conv_dw_w->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(e.conv_dw_w, w_f32.data(), 0, w_f32.size() * sizeof(float));
            } else {
                std::vector<ggml_fp16_t> w_f16(w_f32.size());
                for (size_t j = 0; j < w_f16.size(); j++)
                    w_f16[j] = ggml_fp32_to_fp16(w_f32[j]);
                ggml_backend_tensor_set(e.conv_dw_w, w_f16.data(), 0, w_f16.size() * sizeof(ggml_fp16_t));
            }
            for (int c = 0; c < d; c++)
                dw_b[c] = (dw_b[c] - bn_mean[c]) * s[c] + bn_b[c];
            ggml_backend_tensor_set(e.conv_dw_b, dw_b.data(), 0, d * sizeof(float));
        }
        fprintf(stderr, "canary_qwen: BN folded into conv_dw for %u layers\n", model.hparams.enc_n_layers);
    }

    fprintf(stderr, "canary_qwen: enc=%uL d=%u | proj %u→%u | llm=%uL d=%u vocab=%u\n", model.hparams.enc_n_layers,
            model.hparams.enc_d_model, model.hparams.enc_d_model, model.hparams.llm_d_model, model.hparams.llm_n_layers,
            model.hparams.llm_d_model, model.hparams.llm_vocab_size);
    return true;
}

// ===========================================================================
// FFT (same as canary — iterative Cooley-Tukey)
// ===========================================================================

static void canary_qwen_fft_r2c(const float* in, int N, float* out) {
    int bits = 0;
    for (int n = N; n > 1; n >>= 1)
        bits++;
    for (int i = 0; i < N; i++) {
        int rev = 0;
        for (int b = 0; b < bits; b++)
            rev = (rev << 1) | ((i >> b) & 1);
        out[2 * rev] = in[i];
        out[2 * rev + 1] = 0.0f;
    }
    for (int len = 2; len <= N; len <<= 1) {
        float ang = -2.0f * (float)M_PI / (float)len;
        float wre = cosf(ang), wim = sinf(ang);
        for (int i = 0; i < N; i += len) {
            float ure = 1.0f, uim = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                int a = i + j, b2 = i + j + len / 2;
                float are = out[2 * a], aim = out[2 * a + 1];
                float bre = out[2 * b2], bim = out[2 * b2 + 1];
                float tre = ure * bre - uim * bim, tim = ure * bim + uim * bre;
                out[2 * a] = are + tre;
                out[2 * a + 1] = aim + tim;
                out[2 * b2] = are - tre;
                out[2 * b2 + 1] = aim - tim;
                float new_ure = ure * wre - uim * wim;
                uim = ure * wim + uim * wre;
                ure = new_ure;
            }
        }
    }
}

// NeMo mel: 128 mel, 16 kHz, n_fft=512, win=400, hop=160, per_feature Z-norm.
static std::vector<float> canary_qwen_compute_mel_impl(canary_qwen_context* ctx, const float* samples, int n_samples,
                                                       int& T_out) {
    const auto& hp = ctx->model.hparams;
    const int n_fft = (int)hp.n_fft;
    const int hop = (int)hp.hop_length;
    const int win = (int)hp.win_length;
    const int n_freqs = n_fft / 2 + 1;
    const int n_mels = (int)hp.n_mels;

    if (!ctx->model.mel_fb || !ctx->model.mel_window) {
        fprintf(stderr, "canary_qwen: missing preprocessor.fb / window\n");
        return {};
    }

    std::vector<float> window_raw((size_t)win);
    ggml_backend_tensor_get(ctx->model.mel_window, window_raw.data(), 0, win * sizeof(float));

    std::vector<float> mel_fb((size_t)n_mels * n_freqs);
    ggml_backend_tensor_get(ctx->model.mel_fb, mel_fb.data(), 0, mel_fb.size() * sizeof(float));

    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = win;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Ln;
    p.norm = core_mel::Normalization::PerFeatureZ;
    p.layout = core_mel::Layout::TimeMels;
    p.log_eps = (float)(1.0 / (1 << 24));
    p.center_pad = true;
    p.drop_last_frame = true;
    p.preemph = 0.97f;

    return core_mel::compute(samples, n_samples, window_raw.data(), win, mel_fb.data(), n_freqs, canary_qwen_fft_r2c, p,
                             T_out);
}

// ===========================================================================
// Encoder graph (FastConformer — same structure as canary)
// ===========================================================================

static const float kLayerNormEps = 1e-5f;

static ggml_cgraph* canary_qwen_build_graph_encoder(canary_qwen_context* ctx, int T_mel,
                                                    ggml_context* arena_ctx = nullptr) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int n_mels = (int)hp.n_mels;

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = arena_ctx ? arena_ctx : ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, T_mel);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    int T = 0;
    ggml_tensor* cur = core_conformer::build_pre_encode(ctx0, mel, m.pre_encode, (int)hp.subsampling_channels, &T);

    ggml_tensor* pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int)hp.enc_d_model, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc");
    ggml_set_input(pos_enc);

    core_conformer::BlockParams bp = {
        (int)hp.enc_d_model, (int)hp.enc_n_heads, (int)hp.enc_head_dim, (int)hp.conv_kernel, kLayerNormEps,
    };
    bp.manual_attn = core_conformer::fc_gpu_manual_attn(ctx->backend);
    for (uint32_t il = 0; il < hp.enc_n_layers; il++) {
        cur = core_conformer::build_block(ctx0, cur, pos_enc, T, m.enc[il], bp);
    }

    // Projection: enc_out (d_enc, T) → (d_llm, T)
    cur = ggml_mul_mat(ctx0, m.proj.w, cur);
    if (m.proj.b)
        cur = ggml_add(ctx0, cur, m.proj.b);

    ggml_set_name(cur, "projected");
    ggml_build_forward_expand(gf, cur);
    if (!arena_ctx)
        ggml_free(ctx0);
    return gf;
}

// Run encoder + projection. Returns flat row-major [T_enc * d_llm].
static std::vector<float> canary_qwen_encode_mel(canary_qwen_context* ctx, const float* mel, int n_mels, int T_mel,
                                                 int* out_T_enc) {
    if (n_mels != (int)ctx->model.hparams.n_mels) {
        fprintf(stderr, "canary_qwen: mel feature mismatch (%d vs %d)\n", n_mels, (int)ctx->model.hparams.n_mels);
        return {};
    }

    if (!ctx->sched) {
        ggml_backend_t backends[2] = {ctx->backend, ctx->backend_cpu};
        int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
        crispasr_imatrix_install(ctx->sched);
    }
    if (ctx->compute_meta.empty())
        ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    // #235: always rebuild — cached graph has stale GPU buffer handles after sched regrow
    ggml_cgraph* gf;
    {
        if (ctx->cached_enc_ctx) {
            ggml_free(ctx->cached_enc_ctx);
            ctx->cached_enc_ctx = nullptr;
            ctx->cached_enc_gf = nullptr;
        }
        ctx->cached_enc_meta.assign(ctx->compute_meta.size(), 0);
        ggml_init_params aip = {ctx->cached_enc_meta.size(), ctx->cached_enc_meta.data(), true};
        ctx->cached_enc_ctx = ggml_init(aip);
        gf = canary_qwen_build_graph_encoder(ctx, T_mel, ctx->cached_enc_ctx);
        ctx->cached_enc_gf = gf;
        ctx->cached_enc_T_mel = T_mel;
    }

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "canary_qwen: encoder alloc failed\n");
        return {};
    }

    // Set mel input
    ggml_tensor* mel_t = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_t, mel, 0, n_mels * T_mel * sizeof(float));

    // Compute pos_enc (rel-pos sinusoidals) — read the actual tensor shape
    // from the graph rather than estimating T, since build_pre_encode's exact
    // T depends on conv padding which ceil(T_mel/8) doesn't capture.
    // pos_enc tensor has shape (d, 2*T-1). Recover T from the tensor shape,
    // then make_pos_enc(d, T) returns d*(2*T-1) floats.
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "pos_enc");
    if (pos_t) {
        const int pos_d = (int)pos_t->ne[0];
        int T_enc_actual = ((int)pos_t->ne[1] + 1) / 2;
        auto pos_data = core_conformer::make_pos_enc(pos_d, T_enc_actual);
        ggml_backend_tensor_set(pos_t, pos_data.data(), 0, pos_data.size() * sizeof(float));
    }

    ggml_backend_sched_graph_compute(ctx->sched, gf);

    ggml_tensor* out = ggml_graph_get_tensor(gf, "projected");
    if (!out) {
        fprintf(stderr, "canary_qwen: projected tensor not found\n");
        return {};
    }
    const int d_out = (int)out->ne[0]; // d_llm
    const int T_out = (int)out->ne[1]; // T_enc
    *out_T_enc = T_out;
    std::vector<float> result((size_t)T_out * d_out);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    return result;
}

// ===========================================================================
// LLM KV-cache graph (Qwen3 — same structure as qwen3_asr)
// ===========================================================================

static bool canary_qwen_kv_init(canary_qwen_context* ctx, int max_ctx) {
    const auto& hp = ctx->model.hparams;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int n_layers = (int)hp.llm_n_layers;

    size_t overhead = 2 * ggml_tensor_overhead();
    ggml_init_params ip = {overhead, nullptr, true};
    ctx->kv_ctx = ggml_init(ip);
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, n_layers);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, n_layers);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");

    ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, ctx->backend);
    if (!ctx->kv_buf) {
        fprintf(stderr, "canary_qwen: KV alloc failed (max_ctx=%d)\n", max_ctx);
        ggml_free(ctx->kv_ctx);
        ctx->kv_ctx = nullptr;
        return false;
    }
    ctx->kv_max_ctx = max_ctx;
    ctx->kv_n_used = 0;
    return true;
}

static void canary_qwen_kv_reset(canary_qwen_context* ctx) {
    ctx->kv_n_used = 0;
}

static ggml_cgraph* canary_qwen_build_graph_llm_kv(canary_qwen_context* ctx, int n_past, int n_tokens,
                                                   bool last_token_only = true) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.llm_d_model;
    const int n_q = (int)hp.llm_n_heads;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.llm_rms_eps;
    const float theta = hp.llm_rope_theta;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = n_tokens;
    const int Lk = n_past + T;

    GGML_ASSERT(ctx->kv_k && ctx->kv_v);
    GGML_ASSERT(Lk <= ctx->kv_max_ctx);

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

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

    ggml_tensor* cur = embeds;

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_kv_grp,
        /*n_ctx_orig*/ 40960,
        /*rope_theta*/ theta,
        /*rope_beta_fast*/ 32.0f,
        /*rope_beta_slow*/ 1.0f,
        /*attn_scale*/ attn_scale,
        /*qk_norm_eps*/ eps,
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
    };

    for (uint32_t il = 0; il < hp.llm_n_layers; il++) {
        const auto& b = m.llm.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        ggml_tensor* attn = core_attn::kv_self_attn(
            ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w, b.attn_q_norm_w, b.attn_k_norm_w,
            positions, (T == 1) ? nullptr : causal_mask, ctx->kv_k, ctx->kv_v, (int)il, n_past, kvp, nullptr);
        cur = ggml_add(ctx0, residual, attn);

        // FFN (SwiGLU)
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.llm.output_norm_w);

    if (last_token_only && T > 1) {
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }
    cur = ggml_mul_mat(ctx0, m.llm.output_w, cur);

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Tiny graph for token embedding lookup.
static ggml_cgraph* canary_qwen_build_graph_embed(canary_qwen_context* ctx, int n_tokens) {
    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 64, false);
    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "input_ids");
    ggml_set_input(ids);
    ggml_tensor* out = ggml_get_rows(ctx0, ctx->model.llm.token_embd_w, ids);
    ggml_set_name(out, "embeds");
    ggml_build_forward_expand(gf, out);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Token embedding helper
// ===========================================================================

static std::vector<float> canary_qwen_embed_tokens(canary_qwen_context* ctx, const int32_t* ids, int n_tokens) {
    const int d = (int)ctx->model.hparams.llm_d_model;
    ggml_cgraph* gf = canary_qwen_build_graph_embed(ctx, n_tokens);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return {};

    ggml_tensor* ids_t = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(ids_t, ids, 0, n_tokens * sizeof(int32_t));

    ggml_backend_sched_graph_compute(ctx->sched, gf);

    ggml_tensor* out = ggml_graph_get_tensor(gf, "embeds");
    std::vector<float> result((size_t)n_tokens * d);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    return result;
}

// ===========================================================================
// LLM forward with KV cache
// ===========================================================================

static std::vector<float> canary_qwen_run_llm_kv_impl(canary_qwen_context* ctx, const float* embeds, int n_tokens,
                                                      int n_past) {
    const int d = (int)ctx->model.hparams.llm_d_model;

    ggml_cgraph* gf = canary_qwen_build_graph_llm_kv(ctx, n_past, n_tokens);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "canary_qwen: LLM alloc failed\n");
        return {};
    }

    ggml_tensor* emb_t = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(emb_t, embeds, 0, (size_t)n_tokens * d * sizeof(float));

    // Positions
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "positions");
    std::vector<int32_t> pos(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        pos[i] = n_past + i;
    ggml_backend_tensor_set(pos_t, pos.data(), 0, n_tokens * sizeof(int32_t));

    // Causal mask (prefill only)
    ggml_tensor* mask_t = ggml_graph_get_tensor(gf, "causal_mask");
    if (mask_t && n_tokens > 1) {
        const int Lk = n_past + n_tokens;
        std::vector<ggml_fp16_t> mask((size_t)Lk * n_tokens);
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = 0; k < Lk; k++) {
                mask[(size_t)q * Lk + k] = (k <= n_past + q) ? zero : neg_inf;
            }
        }
        ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    ggml_backend_sched_graph_compute(ctx->sched, gf);

    ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "logits");
    const int out_n = (int)logits_t->ne[0]; // vocab
    std::vector<float> logits(out_n);
    ggml_backend_tensor_get(logits_t, logits.data(), 0, out_n * sizeof(float));
    return logits;
}

// ===========================================================================
// Tokenizer (GPT-2 byte-level BPE — same as qwen3_asr)
// ===========================================================================

static std::vector<int32_t> canary_qwen_tokenize(canary_qwen_context* ctx, const char* text) {
    const auto& v = ctx->vocab;
    std::vector<int32_t> result;
    const std::string s = text;
    size_t i = 0;
    while (i < s.size()) {
        // Special token check
        if (s[i] == '<' && i + 1 < s.size() && s[i + 1] == '|') {
            size_t end = s.find("|>", i + 2);
            if (end != std::string::npos) {
                std::string special = s.substr(i, end + 2 - i);
                auto it = v.token_to_id.find(special);
                if (it != v.token_to_id.end()) {
                    result.push_back(it->second);
                    i = end + 2;
                    continue;
                }
            }
        }
        // Plain text segment
        size_t j = i;
        if (s[j] == '<' && j + 1 < s.size() && s[j + 1] == '|')
            j++;
        while (j < s.size()) {
            if (s[j] == '<' && j + 1 < s.size() && s[j + 1] == '|') {
                size_t end = s.find("|>", j + 2);
                if (end != std::string::npos) {
                    std::string special = s.substr(j, end + 2 - j);
                    if (v.token_to_id.find(special) != v.token_to_id.end())
                        break;
                }
            }
            j++;
        }
        std::string chunk = s.substr(i, j - i);
        i = j;
        if (chunk.empty())
            continue;

        // Pre-split on whitespace + BPE merge
        size_t k = 0;
        while (k < chunk.size()) {
            size_t start = k;
            if (chunk[k] == ' ' || chunk[k] == '\t' || chunk[k] == '\n')
                k++;
            while (k < chunk.size() && chunk[k] != ' ' && chunk[k] != '\t' && chunk[k] != '\n')
                k++;
            if (k == start)
                k++;
            std::string pre(chunk, start, k - start);
            std::string encoded = core_bpe::bytes_to_unicode(pre.data(), pre.size());
            core_bpe::bpe_one(v.token_to_id, v.merge_rank, encoded, result);
        }
    }
    return result;
}

// Decode token IDs back to UTF-8.
static std::string canary_qwen_decode_tokens(canary_qwen_context* ctx, const int32_t* ids, int n_ids) {
    return core_bpe::detokenize(ctx->vocab.id_to_token, ids, (size_t)n_ids);
}

// #247: Count how many leading generated tokens form an instruction-echo
// artifact. When the audio window is too short/low-content to ground a
// transcription, the Qwen3 LLM decoder falls back to its language prior and
// echoes the task framing — emitting meta words like "Transcript",
// "Transcription" or "PASS" instead of a transcript (NeMo's SALM.generate()
// runs plain greedy with no suppress/min-length guard, so this is inherent to
// the model on degenerate input). Those exact leading words are never a real
// ASR transcript under the "Transcribe the following:" prompt, so we strip them
// from BOTH the token array and the text. A multi-token word ("Trans"+"cript")
// is handled by accumulating up to `max_probe` leading tokens and matching the
// decoded, normalised string. Returns the number of leading tokens to drop.
static int canary_qwen_echo_prefix_tokens(canary_qwen_context* ctx, const std::vector<int32_t>& ids) {
    const int max_probe = std::min((int)ids.size(), 4);
    for (int k = max_probe; k >= 1; --k) {
        std::string s = canary_qwen_decode_tokens(ctx, ids.data(), k);
        if (canary_qwen_echo::is_meta_echo(s))
            return k;
    }
    return 0;
}

// ===========================================================================
// Full transcription pipeline
// ===========================================================================

// Build the Qwen-style prompt with audio_start/pad/end framing:
//   <|im_start|>user\nTranscribe the following: <|audio_pad|>×N<|im_end|>\n
//   <|im_start|>assistant\n
//
// The <|audio_pad|> tokens (id 151676) are placeholders — their text
// embeddings are replaced with the projected encoder output frames.

static std::vector<float> canary_qwen_build_prompt_embeds(canary_qwen_context* ctx, const float* audio_embeds,
                                                          int T_enc, int* out_total_tokens) {
    const int d = (int)ctx->model.hparams.llm_d_model;
    const int32_t audio_pad_id = 151676; // <|audio_pad|>

    // Build the prompt with N audio_pad tokens that we'll replace.
    // NeMo blueprint (QwenPromptFormatter + SALM) uses a single
    // <|audioplaceholder|> whose embedding is replaced with the projected
    // encoder frames. NO <|audio_start|> / <|audio_end|> framing tokens —
    // the model was never trained with them, and including them causes
    // instruction-echo artifacts ("Transcript:", "PASS") (#247).
    std::string prompt_text = "<|im_start|>user\nTranscribe the following: ";
    for (int i = 0; i < T_enc; i++)
        prompt_text += "<|audio_pad|>";
    prompt_text += "<|im_end|>\n<|im_start|>assistant\n";

    // Tokenize
    auto prompt_ids = canary_qwen_tokenize(ctx, prompt_text.c_str());

    if (cq_debug_enabled()) {
        fprintf(stderr, "canary_qwen: prompt tokens (%d), T_enc=%d\n", (int)prompt_ids.size(), T_enc);
        // Print first and last few tokens
        int show = std::min(10, (int)prompt_ids.size());
        fprintf(stderr, "  first %d:", show);
        for (int i = 0; i < show; i++)
            fprintf(stderr, " %d", prompt_ids[i]);
        fprintf(stderr, " ...\n");
    }

    // Embed all text tokens
    auto text_embeds = canary_qwen_embed_tokens(ctx, prompt_ids.data(), (int)prompt_ids.size());
    if (text_embeds.empty())
        return {};

    // Replace audio_pad positions with audio encoder embeddings
    int audio_idx = 0;
    for (int i = 0; i < (int)prompt_ids.size() && audio_idx < T_enc; i++) {
        if (prompt_ids[i] == audio_pad_id) {
            memcpy(text_embeds.data() + (size_t)i * d, audio_embeds + (size_t)audio_idx * d, (size_t)d * sizeof(float));
            audio_idx++;
        }
    }

    if (cq_debug_enabled() && audio_idx != T_enc)
        fprintf(stderr, "canary_qwen: WARNING: only %d/%d audio_pad positions found\n", audio_idx, T_enc);

    *out_total_tokens = (int)prompt_ids.size();
    return text_embeds;
}

static canary_qwen_result* canary_qwen_transcribe_impl(canary_qwen_context* ctx, const float* samples, int n_samples) {
    const auto& hp = ctx->model.hparams;
    const int vocab = (int)hp.llm_vocab_size;

    // 1. Compute mel
    int T_mel = 0;
    std::vector<float> mel;
    {
        cq_bench_stage bs("mel");
        mel = canary_qwen_compute_mel_impl(ctx, samples, n_samples, T_mel);
        if (mel.empty())
            return nullptr;
    }

    // 2. Encode + project
    int T_enc = 0;
    std::vector<float> audio_embeds;
    {
        cq_bench_stage bs("encoder+proj");
        audio_embeds = canary_qwen_encode_mel(ctx, mel.data(), (int)hp.n_mels, T_mel, &T_enc);
        if (audio_embeds.empty())
            return nullptr;
    }

    if (cq_debug_enabled())
        fprintf(stderr, "canary_qwen: T_mel=%d T_enc=%d\n", T_mel, T_enc);

    // #247: Degenerate-window guard (root cause of the "Transcript"/"PASS"
    // leak). The FastConformer applies 8x subsampling, so T_enc encoder frames
    // correspond to ~T_enc*80 ms of audio. With only a handful of frames there
    // is not enough acoustic content to ground a transcription, and the Qwen3
    // LLM decoder falls back to its language prior — emitting instruction-echo
    // meta tokens ("Transcript", "Transcription", "PASS", "Okay") instead of
    // words. Measured on canary-qwen-2.5b-q8_0: T_enc<=5 (~<=0.4 s) reliably
    // echoes; T_enc>=6 transcribes. NeMo's SALM.generate() has no guard against
    // this (plain greedy, no suppress/min-length), so the pipeline must not
    // feed the model such windows. Return an empty transcript; in a chunking
    // pipeline the overlapping neighbour already covers any real fragment.
    // Tunable/disable via env (0 disables the gate → pre-fix behaviour).
    int min_enc_frames = 6;
    if (const char* e = std::getenv("CRISPASR_CANARY_QWEN_MIN_ENC_FRAMES"))
        min_enc_frames = std::atoi(e);
    if (min_enc_frames > 0 && T_enc < min_enc_frames) {
        if (cq_debug_enabled())
            fprintf(stderr, "canary_qwen: window too short (T_enc=%d < %d) — empty result to avoid instruction echo\n",
                    T_enc, min_enc_frames);
        auto* r = (canary_qwen_result*)calloc(1, sizeof(canary_qwen_result));
        r->text = strdup("");
        r->n_tokens = 0;
        return r;
    }

    // 3. Build prompt embeddings (text + audio splice)
    int total_prompt = 0;
    std::vector<float> prompt_embeds;
    {
        cq_bench_stage bs("prompt_build");
        prompt_embeds = canary_qwen_build_prompt_embeds(ctx, audio_embeds.data(), T_enc, &total_prompt);
        if (prompt_embeds.empty())
            return nullptr;
    }

    // 4. Allocate KV cache
    const int max_new_tokens = ctx->max_new_tokens > 0 ? ctx->max_new_tokens : 256; // #292
    const int max_ctx = total_prompt + max_new_tokens;
    if (!ctx->kv_k || ctx->kv_max_ctx < max_ctx) {
        if (ctx->kv_buf) {
            ggml_backend_buffer_free(ctx->kv_buf);
            ctx->kv_buf = nullptr;
        }
        if (ctx->kv_ctx) {
            ggml_free(ctx->kv_ctx);
            ctx->kv_ctx = nullptr;
        }
        ctx->kv_k = nullptr;
        ctx->kv_v = nullptr;
        if (!canary_qwen_kv_init(ctx, max_ctx))
            return nullptr;
    }
    canary_qwen_kv_reset(ctx);

    // 5. Prefill — process entire prompt
    std::vector<float> logits;
    {
        cq_bench_stage bs("prefill");
        logits = canary_qwen_run_llm_kv_impl(ctx, prompt_embeds.data(), total_prompt, 0);
        if (logits.empty())
            return nullptr;
    }
    ctx->kv_n_used = total_prompt;

    // 6. Greedy decode loop
    const int32_t eos_id = 151645; // <|im_end|>
    const int32_t endoftext_id = 151643;
    std::vector<int32_t> generated_ids;
    std::vector<float> generated_probs;

    {
        cq_bench_stage bs("decode");
        for (int step = 0; step < max_new_tokens; step++) {
            // Argmax — NaN-robust. Seeding best_val from logits[0] pins best_id at
            // 0 forever when logits[0] is non-finite: `x > NaN` is false for every
            // x, so the scan never moves off 0 and the decode silently spews token
            // 0 (== "!") for the whole budget. That is exactly what a corrupt q4_k
            // LLM projection produced (q8_0 fine, q4_k → all "!"). Seed from -inf,
            // skip non-finite logits, and if the entire row is non-finite treat it
            // as a numerics failure and stop rather than emit garbage.
            int32_t best_id = -1;
            float best_val = -std::numeric_limits<float>::infinity();
            for (int v2 = 0; v2 < vocab; v2++) {
                if (std::isfinite(logits[v2]) && logits[v2] > best_val) {
                    best_val = logits[v2];
                    best_id = v2;
                }
            }
            if (best_id < 0) {
                fprintf(stderr,
                        "canary_qwen: non-finite logits at step %d — aborting decode "
                        "(corrupt/over-quantized weights? this quant is unusable)\n",
                        step);
                break;
            }

            // Compute softmax probability for the chosen token. Skip non-finite
            // logits so a stray Inf/NaN elsewhere in the row can't poison the
            // reported confidence of an otherwise-valid argmax.
            float max_logit = best_val;
            float sum_exp = 0.0f;
            for (int v2 = 0; v2 < vocab; v2++)
                if (std::isfinite(logits[v2]))
                    sum_exp += expf(logits[v2] - max_logit);
            float prob = 1.0f / sum_exp;

            if (best_id == eos_id || best_id == endoftext_id)
                break;

            generated_ids.push_back(best_id);
            generated_probs.push_back(prob);

            // Embed the generated token
            auto tok_embed = canary_qwen_embed_tokens(ctx, &best_id, 1);
            if (tok_embed.empty())
                break;

            // Single-token decode step
            logits = canary_qwen_run_llm_kv_impl(ctx, tok_embed.data(), 1, ctx->kv_n_used);
            ctx->kv_n_used++;
            if (logits.empty())
                break;
        }
    }

    // #247: Instruction-echo safety net. The degenerate-window gate above stops
    // the common case (too-short audio), but a borderline window — or a GPU
    // miscompute on another backend — can still start the output with an
    // instruction-echo meta word ("Transcript"/"Transcription"/"PASS"). Those
    // are never a real transcript here, so drop the leading echo tokens from
    // BOTH the token array and the text (the previous #247 workaround stripped
    // text only, leaving the tokens array inconsistent — see #218). Disable via
    // env for A/B.
    if (!generated_ids.empty() && !std::getenv("CRISPASR_CANARY_QWEN_NO_ECHO_STRIP")) {
        int drop = canary_qwen_echo_prefix_tokens(ctx, generated_ids);
        if (drop > 0) {
            if (cq_debug_enabled())
                fprintf(stderr, "canary_qwen: stripped %d leading instruction-echo token(s)\n", drop);
            generated_ids.erase(generated_ids.begin(), generated_ids.begin() + drop);
            generated_probs.erase(generated_probs.begin(), generated_probs.begin() + drop);
        }
    }

    // 7. Decode tokens to text
    std::string text = canary_qwen_decode_tokens(ctx, generated_ids.data(), (int)generated_ids.size());

    // Strip leading whitespace
    size_t start = 0;
    while (start < text.size() && (text[start] == ' ' || text[start] == '\n'))
        start++;
    if (start > 0)
        text = text.substr(start);

    // Build result
    auto* r = (canary_qwen_result*)calloc(1, sizeof(canary_qwen_result));
    r->text = strdup(text.c_str());
    r->n_tokens = (int)generated_ids.size();
    if (r->n_tokens > 0) {
        r->tokens = (canary_qwen_token_data*)calloc(r->n_tokens, sizeof(canary_qwen_token_data));
        for (int i = 0; i < r->n_tokens; i++) {
            r->tokens[i].id = generated_ids[i];
            r->tokens[i].p = generated_probs[i];
            std::string tok_text = canary_qwen_decode_tokens(ctx, &generated_ids[i], 1);
            snprintf(r->tokens[i].text, sizeof(r->tokens[i].text), "%s", tok_text.c_str());
        }
    }

    if (cq_debug_enabled() || cq_bench_enabled())
        fprintf(stderr, "canary_qwen: generated %d tokens: %s\n", r->n_tokens, r->text);

    return r;
}

// ===========================================================================
// Public API
// ===========================================================================

extern "C" struct canary_qwen_context_params canary_qwen_context_default_params(void) {
    canary_qwen_context_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.flash_attn = false;
    return p;
}

extern "C" struct canary_qwen_context* canary_qwen_init_from_file(const char* path_model,
                                                                  struct canary_qwen_context_params params) {
    auto* ctx = new canary_qwen_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads;

    ctx->backend_cpu = ggml_backend_init_by_name("cpu", nullptr);
    if (!ctx->backend_cpu) {
        delete ctx;
        return nullptr;
    }
    core_cpu_backend::set_n_threads(ctx->backend_cpu, params.n_threads);

    ctx->backend = params.use_gpu ? crispasr_init_gpu_backend() : nullptr;
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;

    if (!canary_qwen_load_model(ctx->model, ctx->vocab, path_model, ctx->backend, ctx->backend_cpu)) {
        canary_qwen_free(ctx);
        return nullptr;
    }

    // Repack F16 conv pw1/pw2 to Q8_0 + fuse encoder Q/K/V (issue #81 — the 3D
    // conv layout dodges crispasr-quantize; fusion is bit-identical).
    {
        auto& m = ctx->model;
        std::vector<core_conformer::BlockWeights*> layers;
        for (auto& e : m.enc)
            layers.push_back(&e);
        const bool quantized = !m.enc.empty() && m.enc[0].attn_q_w && ggml_is_quantized(m.enc[0].attn_q_w->type);
        core_conformer::repack_conv_pw_q8(layers, ctx->backend, quantized, m.pw_q8, "canary_qwen");
        core_conformer::fuse_qkv(layers, ctx->backend, m.qkv_fused, "canary_qwen");
    }

    return ctx;
}

extern "C" void canary_qwen_free(struct canary_qwen_context* ctx) {
    if (!ctx)
        return;
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->cached_enc_ctx)
        ggml_free(ctx->cached_enc_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    ctx->model.pw_q8.free();
    ctx->model.qkv_fused.free();
    if (ctx->model.buf)
        core_gguf::release_weight_buffer(ctx->model.buf);
    if (ctx->model.buf_cpu)
        core_gguf::release_weight_buffer(ctx->model.buf_cpu);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" void canary_qwen_result_free(struct canary_qwen_result* r) {
    if (!r)
        return;
    free(r->text);
    free(r->tokens);
    free(r);
}

extern "C" char* canary_qwen_transcribe(struct canary_qwen_context* ctx, const float* samples, int n_samples) {
    auto* r = canary_qwen_transcribe_impl(ctx, samples, n_samples);
    if (!r)
        return nullptr;
    char* text = r->text;
    r->text = nullptr;
    canary_qwen_result_free(r);
    return text;
}

extern "C" struct canary_qwen_result* canary_qwen_transcribe_ex(struct canary_qwen_context* ctx, const float* samples,
                                                                int n_samples) {
    return canary_qwen_transcribe_impl(ctx, samples, n_samples);
}

extern "C" int canary_qwen_n_vocab(struct canary_qwen_context* ctx) {
    return ctx ? (int)ctx->model.hparams.llm_vocab_size : 0;
}

extern "C" const char* canary_qwen_token_to_str(struct canary_qwen_context* ctx, int token_id) {
    if (!ctx || token_id < 0 || token_id >= (int)ctx->vocab.id_to_token.size())
        return "";
    return ctx->vocab.id_to_token[token_id].c_str();
}

extern "C" void canary_qwen_set_temperature(struct canary_qwen_context* ctx, float temperature, uint64_t seed) {
    if (ctx) {
        ctx->decode_temperature = temperature;
        ctx->decode_seed = seed;
    }
}

extern "C" void canary_qwen_set_max_new_tokens(struct canary_qwen_context* ctx, int n) {
    if (ctx && n > 0) // #292: <= 0 keeps the backend's 256 default
        ctx->max_new_tokens = n;
}

extern "C" void canary_qwen_set_beam_size(struct canary_qwen_context* ctx, int n) {
    if (ctx)
        ctx->beam_size = (n > 0) ? n : 1;
}

extern "C" int canary_qwen_frame_dur_cs(struct canary_qwen_context* ctx) {
    return ctx ? (int)ctx->model.hparams.frame_dur_cs : 8;
}

extern "C" int canary_qwen_n_mels(struct canary_qwen_context* ctx) {
    return ctx ? (int)ctx->model.hparams.n_mels : 128;
}

extern "C" int canary_qwen_sample_rate(struct canary_qwen_context* ctx) {
    return ctx ? (int)ctx->model.hparams.sample_rate : 16000;
}

// ---- Stage-level entry points (for diff testing) ----

extern "C" float* canary_qwen_compute_mel(struct canary_qwen_context* ctx, const float* samples, int n_samples,
                                          int* out_n_mels, int* out_T_mel) {
    if (!ctx)
        return nullptr;
    int T_mel = 0;
    auto mel = canary_qwen_compute_mel_impl(ctx, samples, n_samples, T_mel);
    if (mel.empty())
        return nullptr;
    *out_n_mels = (int)ctx->model.hparams.n_mels;
    *out_T_mel = T_mel;
    float* out = (float*)malloc(mel.size() * sizeof(float));
    memcpy(out, mel.data(), mel.size() * sizeof(float));
    return out;
}

extern "C" float* canary_qwen_run_encoder(struct canary_qwen_context* ctx, const float* mel, int n_mels, int T_mel,
                                          int* out_T_enc, int* out_d_enc) {
    if (!ctx)
        return nullptr;

    // Initialize sched if needed
    if (!ctx->sched) {
        ggml_backend_t backends[2] = {ctx->backend, ctx->backend_cpu};
        int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
        crispasr_imatrix_install(ctx->sched);
    }
    if (ctx->compute_meta.empty())
        ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    int T_enc = 0;
    auto result = canary_qwen_encode_mel(ctx, mel, n_mels, T_mel, &T_enc);
    if (result.empty())
        return nullptr;

    *out_T_enc = T_enc;
    *out_d_enc = (int)ctx->model.hparams.llm_d_model;
    float* out = (float*)malloc(result.size() * sizeof(float));
    memcpy(out, result.data(), result.size() * sizeof(float));
    return out;
}

extern "C" float* canary_qwen_run_projection(struct canary_qwen_context* ctx, const float* enc_out, int T_enc,
                                             int d_enc, int* out_T, int* out_d_llm) {
    // The encoder graph already includes the projection, so this is a no-op
    // passthrough — the caller should use canary_qwen_run_encoder which
    // returns projected output directly.
    (void)ctx;
    (void)enc_out;
    (void)T_enc;
    (void)d_enc;
    (void)out_T;
    (void)out_d_llm;
    fprintf(stderr, "canary_qwen: run_projection is a no-op; use run_encoder\n");
    return nullptr;
}
