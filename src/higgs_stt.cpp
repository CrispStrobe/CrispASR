// higgs_stt.cpp — Qwen/Qwen3-ASR-0.6B ggml runtime
//
// STAGE 1 (current commit): loader + audio encoder conv front-end only.
//   - Loads the GGUF produced by models/convert-qwen3-asr-to-gguf.py
//   - Computes the per-chunk Conv2D subsampler (conv2d1/2/3 + GELU) and the
//     conv_out linear projection. Output shape (num_chunks, T_chunk_out, 896).
//   - Exposed via higgs_stt_run_conv() for differential testing against
//     /tmp/qwen3-asr-ref/jfk/conv_out.npy
//
// Subsequent stages will add the chunked self-attention encoder body, the
// projector head, the Qwen3 0.6B LLM forward, and the audio-injection glue.
//
// See qwen3-asr-todo.md for the full plan.

#include "higgs_stt.h"
#include "core/gpu_layers.h"
#include "core/crispasr_env.h"
#include "../crisp_audio/include/crisp_audio.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "crispasr_imatrix.h"
#include "ggml-cpu.h"
#include "gguf.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Bench instrumentation — `HIGGS_STT_BENCH=1` for per-stage timings.
// ===========================================================================

static bool higgs_stt_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_HIGGS_STT_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct higgs_stt_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit higgs_stt_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~higgs_stt_bench_stage() {
        if (!higgs_stt_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  higgs_stt_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ===========================================================================
// Hyper-parameters
// ===========================================================================

struct higgs_stt_hparams {
    // Audio
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 128;
    uint32_t n_fft = 400;
    uint32_t win_length = 400;
    uint32_t hop_length = 160;
    // Audio encoder — Whisper-large-v3 (audio_tower.*)
    uint32_t audio_n_layers = 32;
    uint32_t audio_d_model = 1280;
    uint32_t audio_n_heads = 20;
    uint32_t audio_head_dim = 64;
    uint32_t audio_ff_dim = 5120;
    uint32_t audio_max_pos = 1500;
    uint32_t audio_avgpool = 2; // AvgPool1d(2,2) after the encoder layers (-> 25 fps)

    // Projector (audio_encoder_proj.*): depthwise temporal Conv1d(k3,s2,p1) +
    // Linear(in->hidden) + ReLU + Linear(hidden->out). -> 12.5 tps, dim = llm hidden.
    uint32_t proj_in_dim = 1280;
    uint32_t proj_hidden_dim = 2048;
    uint32_t proj_out_dim = 2048;
    uint32_t proj_temporal_downsample = 2;

    // LLM (Qwen3-1.7B-Base)
    uint32_t llm_n_layers = 28;
    uint32_t llm_d_model = 2048;
    uint32_t llm_n_heads = 16;
    uint32_t llm_n_kv_heads = 8;
    uint32_t llm_head_dim = 128;
    uint32_t llm_ff_dim = 6144;
    float llm_rope_theta = 1e6f;
    float llm_rms_eps = 1e-6f;
    uint32_t llm_vocab_size = 151936;
    uint32_t llm_max_pos = 32768;

    // The LM head's actual output dimension. For the standard ASR
    // models this equals llm_vocab_size (152064 for 1.7B, 151936 for
    // 0.6B). For the Qwen3-ForcedAligner-0.6B variant the head
    // outputs 5000 timestamp classes instead — we read the real
    // value from the loaded `output.weight` tensor's shape rather
    // than asserting equality with vocab_size.
    uint32_t llm_lm_head_dim = 0; // 0 = "use vocab_size" sentinel

    // Special tokens (higgs ChatML). audio_pad_token_id is the <|AUDIO|>
    // placeholder whose embeddings get replaced by projected audio frames.
    uint32_t audio_start_token_id = 151669; // <|audio_bos|>
    uint32_t audio_end_token_id = 151670;   // <|audio_eos|>
    uint32_t audio_pad_token_id = 151672;   // <|AUDIO|>
    uint32_t im_start_token_id = 151644;    // <|im_start|>
    uint32_t im_end_token_id = 151645;      // <|im_end|>
    uint32_t eos_token_id = 151645;         // stop on <|im_end|>
    uint32_t pad_token_id = 151643;         // <|endoftext|>
};

// ===========================================================================
// Per-layer tensor containers
// ===========================================================================

struct higgs_stt_audio_block {
    // Pre-LN self-attention
    ggml_tensor *attn_norm_w = nullptr, *attn_norm_b = nullptr;
    ggml_tensor *attn_q_w = nullptr, *attn_q_b = nullptr;
    ggml_tensor *attn_k_w = nullptr, *attn_k_b = nullptr;
    ggml_tensor *attn_v_w = nullptr, *attn_v_b = nullptr;
    ggml_tensor *attn_out_w = nullptr, *attn_out_b = nullptr;
    // Pre-LN FFN (GELU)
    ggml_tensor *ffn_norm_w = nullptr, *ffn_norm_b = nullptr;
    ggml_tensor *ffn_up_w = nullptr, *ffn_up_b = nullptr;
    ggml_tensor *ffn_down_w = nullptr, *ffn_down_b = nullptr;
};

struct higgs_stt_audio_tower {
    // Whisper-large-v3 conv front-end
    ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr; // (3, 128, 1280) stride 1
    ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr; // (3, 1280, 1280) stride 2
    ggml_tensor* embed_positions = nullptr;             // (1280, 1500) learned pos

    // Encoder body (32 Whisper-style pre-LN blocks)
    std::vector<higgs_stt_audio_block> blocks;

    // Post-AvgPool final norm
    ggml_tensor *ln_post_w = nullptr, *ln_post_b = nullptr;

    // Projector (audio_encoder_proj): depthwise temporal conv + Linear/ReLU/Linear
    ggml_tensor *proj_temporal_w = nullptr, *proj_temporal_b = nullptr; // (3,1,1280) depthwise
    ggml_tensor *proj_linear1_w = nullptr, *proj_linear1_b = nullptr;   // (1280, 2048)
    ggml_tensor *proj_linear2_w = nullptr, *proj_linear2_b = nullptr;   // (2048, 2048)

    // Mel preprocessor (baked from WhisperFeatureExtractor by the converter)
    ggml_tensor* mel_filters = nullptr; // (n_freqs=201, n_mels=128) F32
    ggml_tensor* mel_window = nullptr;  // (400,) F32 hann window
};

struct higgs_stt_llm_block {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_qkv_w = nullptr; // fused Q+K+V for single matmul
    ggml_tensor* attn_output_w = nullptr;
    ggml_tensor* attn_q_norm_w = nullptr; // Qwen3 per-head Q RMSNorm
    ggml_tensor* attn_k_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct higgs_stt_llm {
    ggml_tensor* token_embd_w = nullptr; // (151936, 1024)
    std::vector<higgs_stt_llm_block> blocks;
    ggml_tensor* output_norm_w = nullptr;
    ggml_tensor* output_w = nullptr;
};

struct higgs_stt_model {
    higgs_stt_hparams hparams;
    higgs_stt_audio_tower audio;
    higgs_stt_llm llm;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    // PLAN #69a: optional second buffer for layers spilled to CPU.
    // Non-null only when CRISPASR_N_GPU_LAYERS triggered a split load.
    ggml_backend_buffer_t buf_cpu = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Sinusoidal positional embedding for the audio encoder, computed once
    // at load time. Layout: row-major (max_pos, d_model) where row p is the
    // pos embed for position p.
    std::vector<float> audio_pe; // size = audio_max_pos * audio_d_model
};

struct higgs_stt_vocab {
    std::vector<std::string> id_to_token;

    // Reverse lookup: byte-encoded vocab string → token id (for BPE encode).
    std::unordered_map<std::string, int32_t> token_to_id;

    // BPE merges loaded from the GGUF. Each merge is a (left, right) pair
    // of byte-encoded strings; the rank is the index in the list (lower = earlier
    // merge = higher priority, matching GPT-2 / Qwen2 BPE convention).
    std::unordered_map<std::string, int32_t> merge_rank; // "left right" → rank
};

// Forward decl from crisp_audio — full type lives in crisp_audio/include/crisp_audio.h
struct crisp_audio_context;

struct higgs_stt_context {
    higgs_stt_context_params params;

    higgs_stt_model model;
    higgs_stt_vocab vocab;

    // Optional custom instruction (set via higgs_stt_set_ask). Empty = the
    // default "Transcribe the speech. …" prompt. Used to override the task
    // (e.g. a language hint or a Q&A prompt) without recompiling.
    std::string ask;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    // Fused QKV weights (optional optimization)
    ggml_context* fused_ctx = nullptr;
    ggml_backend_buffer_t fused_buf = nullptr;

    std::vector<uint8_t> compute_meta;

    // KV cache (Stage 5). Single tensor for K, single for V, both shape
    // (head_dim, max_ctx, n_kv_heads, n_layers). Allocated to backend.
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;
    int kv_n_used = 0;

    int n_threads = 4;
    int beam_size = 1;         // 1 = greedy (default); >1 = beam search (core_beam_decode)
    int max_new_tokens = 1024; // #292: decode cap; forwarded from --max-new-tokens when set, else this default.

    // Shared audio-tower runtime (loaded lazily on first audio call). The
    // higgs_stt_audio_tower struct above is kept around so existing in-tree
    // tests / fallbacks compile; it is no longer the path used by
    // higgs_stt_compute_mel / higgs_stt_run_encoder once `audio_ca` is open.
    crisp_audio_context* audio_ca = nullptr;
    std::string model_path; // remembered for lazy crisp_audio init

    // §176s: cached encoder graph — reused when (T_chunk, num_chunks, T_chunk_out) match.
    ggml_cgraph* cached_enc_gf = nullptr;
    std::vector<uint8_t> cached_enc_meta;
    int cached_enc_T_chunk = 0;
    int cached_enc_num_chunks = 0;
    int cached_enc_T_chunk_out = 0;
};

// ===========================================================================
// Loader helpers
// ===========================================================================

#include "core/gguf_loader.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static ggml_tensor* try_get(higgs_stt_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}

static ggml_tensor* require(higgs_stt_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "higgs_stt");
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool higgs_stt_load_model(higgs_stt_model& model, higgs_stt_vocab& vocab, const char* path,
                                 ggml_backend_t backend, ggml_backend_t backend_cpu) {
    // ---- pass 1: read hparams + vocab via metadata-only context ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;

        auto& hp = model.hparams;
        hp.sample_rate = core_gguf::kv_u32(gctx, "higgs.sample_rate", hp.sample_rate);
        hp.n_mels = core_gguf::kv_u32(gctx, "higgs.n_mels", hp.n_mels);
        hp.n_fft = core_gguf::kv_u32(gctx, "higgs.n_fft", hp.n_fft);
        hp.win_length = core_gguf::kv_u32(gctx, "higgs.win_length", hp.win_length);
        hp.hop_length = core_gguf::kv_u32(gctx, "higgs.hop_length", hp.hop_length);
        hp.audio_n_layers = core_gguf::kv_u32(gctx, "higgs.audio.n_layers", hp.audio_n_layers);
        hp.audio_d_model = core_gguf::kv_u32(gctx, "higgs.audio.d_model", hp.audio_d_model);
        hp.audio_n_heads = core_gguf::kv_u32(gctx, "higgs.audio.n_heads", hp.audio_n_heads);
        hp.audio_head_dim = core_gguf::kv_u32(gctx, "higgs.audio.head_dim", hp.audio_head_dim);
        hp.audio_ff_dim = core_gguf::kv_u32(gctx, "higgs.audio.ff_dim", hp.audio_ff_dim);
        hp.audio_max_pos = core_gguf::kv_u32(gctx, "higgs.audio.max_pos", hp.audio_max_pos);
        hp.audio_avgpool = core_gguf::kv_u32(gctx, "higgs.audio.avgpool", hp.audio_avgpool);

        hp.proj_in_dim = core_gguf::kv_u32(gctx, "higgs.proj.in_dim", hp.proj_in_dim);
        hp.proj_hidden_dim = core_gguf::kv_u32(gctx, "higgs.proj.hidden_dim", hp.proj_hidden_dim);
        hp.proj_out_dim = core_gguf::kv_u32(gctx, "higgs.proj.out_dim", hp.proj_out_dim);
        hp.proj_temporal_downsample =
            core_gguf::kv_u32(gctx, "higgs.proj.temporal_downsample", hp.proj_temporal_downsample);

        hp.llm_n_layers = core_gguf::kv_u32(gctx, "higgs.llm.n_layers", hp.llm_n_layers);
        hp.llm_d_model = core_gguf::kv_u32(gctx, "higgs.llm.d_model", hp.llm_d_model);
        hp.llm_n_heads = core_gguf::kv_u32(gctx, "higgs.llm.n_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = core_gguf::kv_u32(gctx, "higgs.llm.n_kv_heads", hp.llm_n_kv_heads);
        hp.llm_head_dim = core_gguf::kv_u32(gctx, "higgs.llm.head_dim", hp.llm_head_dim);
        hp.llm_ff_dim = core_gguf::kv_u32(gctx, "higgs.llm.ff_dim", hp.llm_ff_dim);
        hp.llm_rope_theta = core_gguf::kv_f32(gctx, "higgs.llm.rope_theta", hp.llm_rope_theta);
        hp.llm_rms_eps = core_gguf::kv_f32(gctx, "higgs.llm.rms_norm_eps", hp.llm_rms_eps);
        hp.llm_vocab_size = core_gguf::kv_u32(gctx, "higgs.llm.vocab_size", hp.llm_vocab_size);
        hp.llm_max_pos = core_gguf::kv_u32(gctx, "higgs.llm.max_pos", hp.llm_max_pos);

        hp.audio_start_token_id = core_gguf::kv_u32(gctx, "higgs.audio_bos_token_id", hp.audio_start_token_id);
        hp.audio_end_token_id = core_gguf::kv_u32(gctx, "higgs.audio_eos_token_id", hp.audio_end_token_id);
        hp.audio_pad_token_id = core_gguf::kv_u32(gctx, "higgs.audio_in_token_id", hp.audio_pad_token_id);
        hp.im_start_token_id = core_gguf::kv_u32(gctx, "higgs.im_start_token_id", hp.im_start_token_id);
        hp.im_end_token_id = core_gguf::kv_u32(gctx, "higgs.im_end_token_id", hp.im_end_token_id);
        hp.eos_token_id = core_gguf::kv_u32(gctx, "higgs.eos_token_id", hp.eos_token_id);
        hp.pad_token_id = core_gguf::kv_u32(gctx, "higgs.pad_token_id", hp.pad_token_id);

        auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            vocab.id_to_token = std::move(tokens);
            vocab.token_to_id.reserve(vocab.id_to_token.size());
            for (int i = 0; i < (int)vocab.id_to_token.size(); i++) {
                vocab.token_to_id[vocab.id_to_token[i]] = i;
            }
        }

        // Register Qwen2/Qwen3 special tokens explicitly. The original
        // vocab.json that the converter pulls strings from only has 151 643
        // regular tokens; the special tokens (<|im_start|>, <|audio_pad|>,
        // ...) live in tokenizer_config.json's added_tokens list, which the
        // converter currently doesn't propagate, so they end up as
        // "[PAD151644]" etc. in vocab.id_to_token. Patch them in here so
        // higgs_stt_tokenize and higgs_stt_token_text can find them.
        struct SpecialTok {
            int id;
            const char* text;
        };
        static const SpecialTok specials[] = {
            {151643, "<|endoftext|>"},        {151644, "<|im_start|>"},       {151645, "<|im_end|>"},
            {151646, "<|object_ref_start|>"}, {151647, "<|object_ref_end|>"}, {151648, "<|box_start|>"},
            {151649, "<|box_end|>"},          {151650, "<|quad_start|>"},     {151651, "<|quad_end|>"},
            {151652, "<|vision_start|>"},     {151653, "<|vision_end|>"},     {151654, "<|vision_pad|>"},
            {151655, "<|image_pad|>"},        {151656, "<|video_pad|>"},      {151669, "<|audio_start|>"},
            {151670, "<|audio_end|>"},        {151676, "<|audio_pad|>"},
        };
        for (const auto& sp : specials) {
            if (sp.id < (int)vocab.id_to_token.size()) {
                // Drop the old [PAD<id>] reverse-map entry if present
                auto old_it = vocab.token_to_id.find(vocab.id_to_token[sp.id]);
                if (old_it != vocab.token_to_id.end() && old_it->second == sp.id) {
                    vocab.token_to_id.erase(old_it);
                }
                vocab.id_to_token[sp.id] = sp.text;
                vocab.token_to_id[sp.text] = sp.id;
            }
        }
        // Merges (BPE encode side). Each entry is a "left right" pair string;
        // the ARRAY index is the merge's rank (lowest rank = highest priority).
        auto merges = core_gguf::kv_str_array(gctx, "tokenizer.ggml.merges");
        for (int i = 0; i < (int)merges.size(); i++) {
            vocab.merge_rank[merges[i]] = i;
        }

        core_gguf::free_metadata(gctx);
    }

    // ---- pass 2: tensor data via shared helper ----
    // PLAN #69a: when CRISPASR_N_GPU_LAYERS is set and < total layers,
    // route layers [N..total) onto the CPU backend.
    core_gguf::WeightLoad wl;
    // Session n_gpu_layers, else CRISPASR_N_GPU_LAYERS (#610).
    const int n_gpu_layers_env = core_gpu_layers::resolve();
    const int total_layers = (int)model.hparams.llm_n_layers;
    const bool do_split =
        backend_cpu && backend_cpu != backend && n_gpu_layers_env >= 0 && n_gpu_layers_env < total_layers;
    if (do_split) {
        int threshold = n_gpu_layers_env;
        if (!core_gguf::load_weights_split(path, backend, backend_cpu, core_gguf::is_gpu_tensor_blk, &threshold,
                                           "higgs_stt", wl)) {
            return false;
        }
        fprintf(stderr, "higgs_stt: layer offload: gpu=[0,%d), cpu=[%d,%d) (CRISPASR_N_GPU_LAYERS=%d)\n",
                n_gpu_layers_env, n_gpu_layers_env, total_layers, n_gpu_layers_env);
    } else {
        if (!core_gguf::load_weights(path, backend, "higgs_stt", wl)) {
            return false;
        }
    }
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.buf_cpu = wl.buf_cpu;
    model.tensors = std::move(wl.tensors);

    // ---- bind named tensors into the per-layer structs ----
    auto& a = model.audio;
    a.conv1_w = require(model, "audio.conv.1.weight");
    a.conv1_b = require(model, "audio.conv.1.bias");
    a.conv2_w = require(model, "audio.conv.2.weight");
    a.conv2_b = require(model, "audio.conv.2.bias");
    a.embed_positions = require(model, "audio.embed_positions");
    a.ln_post_w = require(model, "audio.ln_post.weight");
    a.ln_post_b = require(model, "audio.ln_post.bias");
    a.proj_temporal_w = require(model, "proj.temporal.weight");
    a.proj_temporal_b = require(model, "proj.temporal.bias");
    a.proj_linear1_w = require(model, "proj.linear1.weight");
    a.proj_linear1_b = require(model, "proj.linear1.bias");
    a.proj_linear2_w = require(model, "proj.linear2.weight");
    a.proj_linear2_b = require(model, "proj.linear2.bias");
    a.mel_filters = try_get(model, "audio.mel_filters"); // optional (may be missing in older GGUFs)
    a.mel_window = try_get(model, "audio.mel_window");

    a.blocks.resize(model.hparams.audio_n_layers);
    for (uint32_t i = 0; i < model.hparams.audio_n_layers; i++) {
        char buf[128];
        auto& b = a.blocks[i];
        auto get = [&](const char* suf) {
            snprintf(buf, sizeof(buf), "audio.blk.%u.%s", i, suf);
            return require(model, buf);
        };
        b.attn_norm_w = get("attn_norm.weight");
        b.attn_norm_b = get("attn_norm.bias");
        b.attn_q_w = get("attn_q.weight");
        b.attn_q_b = get("attn_q.bias");
        b.attn_k_w = get("attn_k.weight");
        b.attn_k_b = nullptr; // Whisper quirk: K projection has no bias
        b.attn_v_w = get("attn_v.weight");
        b.attn_v_b = get("attn_v.bias");
        b.attn_out_w = get("attn_out.weight");
        b.attn_out_b = get("attn_out.bias");
        b.ffn_norm_w = get("ffn_norm.weight");
        b.ffn_norm_b = get("ffn_norm.bias");
        b.ffn_up_w = get("ffn_up.weight");
        b.ffn_up_b = get("ffn_up.bias");
        b.ffn_down_w = get("ffn_down.weight");
        b.ffn_down_b = get("ffn_down.bias");
    }

    auto& l = model.llm;
    l.token_embd_w = require(model, "token_embd.weight");
    l.output_norm_w = require(model, "output_norm.weight");
    l.output_w = require(model, "output.weight");
    // Read the actual lm_head output dimension from the loaded tensor
    // shape rather than asserting it equals vocab_size. The standard
    // Qwen3-ASR-{0.6B,1.7B} models have lm_head = (vocab, d), but the
    // Qwen3-ForcedAligner variant has lm_head = (5000, d) — same body,
    // different head. ne[1] is the row count after ggml's [in, out]
    // storage convention.
    model.hparams.llm_lm_head_dim = (uint32_t)l.output_w->ne[1];
    if (model.hparams.llm_lm_head_dim == 0) {
        model.hparams.llm_lm_head_dim = model.hparams.llm_vocab_size;
    }
    l.blocks.resize(model.hparams.llm_n_layers);
    for (uint32_t i = 0; i < model.hparams.llm_n_layers; i++) {
        char buf[128];
        auto& b = l.blocks[i];
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

    // higgs uses a LEARNED positional embedding (audio.embed_positions), loaded
    // above as a tensor — no sinusoidal precompute needed (unlike Qwen3-ASR).

    return true;
}

// ===========================================================================
// FFT (Cooley-Tukey for even sizes, falls back to DFT for odd leaves).
// Handles n_fft=400 (= 2^4 * 25) by recursing down to a 25-point DFT.
// ===========================================================================

static void higgs_stt_dft(const float* in, int N, float* out) {
    for (int k = 0; k < N; k++) {
        float re = 0.0f, im = 0.0f;
        for (int n = 0; n < N; n++) {
            float ang = -2.0f * (float)M_PI * (float)k * (float)n / (float)N;
            re += in[n] * std::cos(ang);
            im += in[n] * std::sin(ang);
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
}

// Real-input FFT, output complex (out has 2*N floats interleaved real/imag).
// in/out are scratch buffers; in must have at least 2*N floats of writable space.
static void higgs_stt_fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0.0f;
        return;
    }
    int half_N = N / 2;
    if (N - half_N * 2 == 1) {
        higgs_stt_dft(in, N, out);
        return;
    }

    float* even = in + N;
    for (int i = 0; i < half_N; i++)
        even[i] = in[2 * i];
    float* even_fft = out + 2 * N;
    higgs_stt_fft(even, half_N, even_fft);

    float* odd = even;
    for (int i = 0; i < half_N; i++)
        odd[i] = in[2 * i + 1];
    float* odd_fft = even_fft + N;
    higgs_stt_fft(odd, half_N, odd_fft);

    for (int k = 0; k < half_N; k++) {
        float ang = -2.0f * (float)M_PI * (float)k / (float)N;
        float re = std::cos(ang);
        float im = std::sin(ang);
        float re_odd = odd_fft[2 * k];
        float im_odd = odd_fft[2 * k + 1];
        out[2 * k] = even_fft[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;
        out[2 * (k + half_N)] = even_fft[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + half_N) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

// ===========================================================================
// Whisper-style log-mel spectrogram
//
// Pipeline (matches WhisperFeatureExtractor._np_extract_fbank_features):
//   1. center-pad audio with n_fft/2 zeros on each side
//   2. STFT: hann window length 400, hop 160, n_fft 400 → (n_freqs=201, T)
//   3. power = |STFT|^2
//   4. mel = power @ filters^T → (n_mels=128, T)
//   5. log10(max(mel, 1e-10))
//   6. drop the last frame
//   7. clip: log_spec = max(log_spec, log_spec.max() - 8.0)
//   8. normalize: log_spec = (log_spec + 4) / 4
//
// Returns a flat (n_mels, T) row-major buffer.
// ===========================================================================

#include "core/mel.h"
#include "core/ffn.h"
#include "core/attention.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// higgs_stt_fft uses its input buffer as scratch during recursion (needs
// ~4N extra floats past the input pointer). Wrap it to match core_mel's
// const-input FftR2C signature, same trick as voxtral / voxtral4b.
static void higgs_stt_fft_wrapper(const float* in, int N, float* out) {
    static thread_local std::vector<float> scratch_in;
    static thread_local std::vector<float> scratch_out;
    if ((int)scratch_in.size() < 4 * N)
        scratch_in.assign((size_t)4 * N, 0.0f);
    if ((int)scratch_out.size() < 8 * N)
        scratch_out.assign((size_t)8 * N, 0.0f);
    std::memcpy(scratch_in.data(), in, (size_t)N * sizeof(float));
    higgs_stt_fft(scratch_in.data(), N, scratch_out.data());
    std::memcpy(out, scratch_out.data(), (size_t)(2 * N) * sizeof(float));
}

extern "C" float* higgs_stt_compute_mel(higgs_stt_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                                        int* out_T_mel) {
    // higgs uses the in-tree Whisper mel directly (no crisp_audio shared tower).
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    if (!ctx->model.audio.mel_filters || !ctx->model.audio.mel_window) {
        fprintf(stderr, "higgs_stt: model GGUF missing audio.mel_filters / audio.mel_window\n");
        return nullptr;
    }

    const int n_fft = (int)hp.n_fft;    // 400
    const int hop = (int)hp.hop_length; // 160
    const int n_mels = (int)hp.n_mels;  // 128
    const int n_freqs = n_fft / 2 + 1;  // 201

    std::vector<float> hann(n_fft);
    ggml_backend_tensor_get(ctx->model.audio.mel_window, hann.data(), 0, n_fft * sizeof(float));
    std::vector<float> filt((size_t)n_freqs * n_mels);
    ggml_backend_tensor_get(ctx->model.audio.mel_filters, filt.data(), 0, filt.size() * sizeof(float));

    // higgs-audio chunks the waveform into chunk_size_seconds (=4 s) pieces and
    // runs the WhisperFeatureExtractor PER CHUNK, so the mel length and the
    // GlobalClipMax normalization max are per-chunk (NOT a single padded 30 s
    // window). Compute the mel on exactly the samples handed in — the caller
    // (higgs_stt_transcribe) splits the audio into chunks and calls this once
    // per chunk. The per-chunk max is dominated by the speech frames, so a
    // chunk computed at its true length matches the reference's zero-padded
    // chunk for the valid frames.

    // Qwen/Whisper HF feature extractor: log10 + max-clip guard, double-accum
    // matmul, drop last STFT frame, fb in (n_freqs, n_mels) layout.
    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = n_fft;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Log10;
    p.log_guard = core_mel::LogGuard::MaxClip;
    p.norm = core_mel::Normalization::GlobalClipMax;
    p.layout = core_mel::Layout::MelsTime;
    p.fb_layout = core_mel::FbLayout::FreqsMels;
    p.matmul = core_mel::MatmulPrecision::Double;
    p.log_eps = 1e-10f;
    p.center_pad = true;
    p.drop_last_frame = true;

    int T_ret = 0;
    auto mel = core_mel::compute(samples, n_samples, hann.data(), n_fft, filt.data(), n_freqs, higgs_stt_fft_wrapper, p,
                                 T_ret);

    if (mel.empty())
        return nullptr;

    if (out_n_mels)
        *out_n_mels = n_mels;
    if (out_T_mel)
        *out_T_mel = T_ret;

    float* result = (float*)malloc(mel.size() * sizeof(float));
    std::memcpy(result, mel.data(), mel.size() * sizeof(float));
    return result;
}

// ===========================================================================
// Conv front-end graph (Stage 1)
//
// Input  (set on the CPU side as a contiguous F32 buffer):
//   mel_batched: shape (T_chunk, n_mels, 1, num_chunks)  in ggml ne order
//                = num_chunks chunks of (1, n_mels, T_chunk) per the
//                  reference impl's per-chunk processing
//
// Output:
//   conv_out: shape (audio_d_model, T_chunk_out, num_chunks)
//             = num_chunks chunks of (T_chunk_out, audio_d_model) frames
//
// Each chunk is processed independently through:
//   conv2d1 + bias + GELU      (in_ch=1,   out_ch=480, k=3, stride=2, pad=1)
//   conv2d2 + bias + GELU      (in_ch=480, out_ch=480, k=3, stride=2, pad=1)
//   conv2d3 + bias + GELU      (in_ch=480, out_ch=480, k=3, stride=2, pad=1)
//   permute + flatten freq → (num_chunks * T_chunk_out, 480 * F_out)
//   conv_out linear (480*16=7680 → 896) + optional bias
//
// For our reference test on jfk.wav:
//   T_chunk=100, n_mels=128, num_chunks=11 → conv1: (50,64,480), conv2:
//   (25,32,480), conv3: (13,16,480) → flatten: (13, 7680) → linear: (13, 896)
// ===========================================================================

static const float kLayerNormEps = 1e-5f;

static ggml_cgraph* higgs_stt_build_graph_conv(higgs_stt_context* ctx) {
    // Whisper conv front-end only (diff stage: post-conv2 hidden = (d, T_enc)).
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.audio_d_model;
    const int n_mels = (int)hp.n_mels;
    const int T_mel = 3000;
    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_mel, n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    auto bias_1d = [&](ggml_tensor* b) { return ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1); };
    ggml_tensor* cur = ggml_conv_1d(ctx0, m.audio.conv1_w, mel, /*s*/ 1, /*p*/ 1, /*d*/ 1);
    cur = ggml_add(ctx0, cur, bias_1d(m.audio.conv1_b));
    cur = ggml_gelu_erf(ctx0, cur);
    cur = ggml_conv_1d(ctx0, m.audio.conv2_w, cur, /*s*/ 2, /*p*/ 1, /*d*/ 1);
    cur = ggml_add(ctx0, cur, bias_1d(m.audio.conv2_b));
    cur = ggml_gelu_erf(ctx0, cur);
    const int T_enc = T_mel / 2;
    cur = ggml_reshape_2d(ctx0, cur, T_enc, d);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // (d, T_enc)
    ggml_set_name(cur, "conv_front_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Full audio encoder graph (Whisper-large-v3 + AvgPool + projector)
//
// Pipeline (matches HiggsAudioEncoder.forward + HiggsAudioFeatureProjector):
//   mel (128, 3000)
//   -> conv1(k3,s1,p1)+GELU -> conv2(k3,s2,p1)+GELU          (d, 1500)
//   -> + learned embed_positions[:1500]
//   -> 32 x Whisper pre-LN block (full bidir attn, biased Q/V/out, no K bias)
//   -> AvgPool1d(2,2) over time                              (d, 750) = 25 fps
//   -> final layer_norm                          ["encoder_hidden"]
//   -> projector: depthwise temporal Conv1d(k3,s2,p1) -> Linear(1280->2048)
//      -> ReLU -> Linear(2048->2048)             ["encoder_out"] (2048, 375)
// ===========================================================================

static ggml_cgraph* higgs_stt_build_graph_encoder(higgs_stt_context* ctx, int T_mel,
                                                  ggml_context* arena_ctx = nullptr) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.audio_d_model;         // 1280
    const int n_heads = (int)hp.audio_n_heads;   // 20
    const int head_dim = (int)hp.audio_head_dim; // 64
    const int n_mels = (int)hp.n_mels;           // 128
    // conv2 stride 2 (k3,p1): out length = (T_mel - 1) / 2 + 1. The chunk mel
    // length is variable (4 s chunks = 400 frames, last chunk shorter).
    const int T_enc = (T_mel - 1) / 2 + 1;
    const float attn_scale = 1.0f / std::sqrt((float)head_dim);

    ggml_context* ctx0;
    if (arena_ctx) {
        ctx0 = arena_ctx;
    } else {
        ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
        ctx0 = ggml_init(ip);
    }
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_mel, n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    auto bias_1d = [&](ggml_tensor* b) { return ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1); };
    ggml_tensor* cur = ggml_conv_1d(ctx0, m.audio.conv1_w, mel, /*s*/ 1, /*p*/ 1, /*d*/ 1);
    cur = ggml_add(ctx0, cur, bias_1d(m.audio.conv1_b));
    cur = ggml_gelu_erf(ctx0, cur);
    cur = ggml_conv_1d(ctx0, m.audio.conv2_w, cur, /*s*/ 2, /*p*/ 1, /*d*/ 1);
    cur = ggml_add(ctx0, cur, bias_1d(m.audio.conv2_b));
    cur = ggml_gelu_erf(ctx0, cur);
    cur = ggml_reshape_2d(ctx0, cur, T_enc, d);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // (d, T_enc)

    // learned absolute positional embedding, sliced to this chunk's length:
    // embed_positions.weight[:T_enc] (the encoder uses chunk-local positions).
    ggml_tensor* embp = ggml_view_2d(ctx0, m.audio.embed_positions, d, T_enc, m.audio.embed_positions->nb[1], 0);
    cur = ggml_add(ctx0, cur, embp);

    for (uint32_t il = 0; il < hp.audio_n_layers; il++) {
        const auto& b = m.audio.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);
        x = ggml_add(ctx0, x, b.attn_norm_b);

        // biased Q/V/out, NO K bias (Whisper quirk); no RoPE, no mask (bidir).
        ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_q_w, x), b.attn_q_b);
        ggml_tensor* K = ggml_mul_mat(ctx0, b.attn_k_w, x);
        ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_v_w, x), b.attn_v_b);
        Q = ggml_reshape_3d(ctx0, Q, head_dim, n_heads, T_enc);
        K = ggml_reshape_3d(ctx0, K, head_dim, n_heads, T_enc);
        V = ggml_reshape_3d(ctx0, V, head_dim, n_heads, T_enc);
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));
        ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q);
        scores = ggml_soft_max_ext(ctx0, scores, nullptr, attn_scale, 0.0f);
        ggml_tensor* V2 = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3));
        ggml_tensor* attn = ggml_mul_mat(ctx0, V2, scores);
        attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(ctx0, attn, d, T_enc);
        attn = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_out_w, attn), b.attn_out_b);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        x = ggml_add(ctx0, x, b.ffn_norm_b);
        x = ggml_add(ctx0, ggml_mul_mat(ctx0, b.ffn_up_w, x), b.ffn_up_b);
        x = ggml_gelu_erf(ctx0, x);
        x = ggml_add(ctx0, ggml_mul_mat(ctx0, b.ffn_down_w, x), b.ffn_down_b);
        cur = ggml_add(ctx0, residual, x);
    }

    // AvgPool1d(2, stride 2) over time, applied BEFORE the final layer_norm.
    // cur ne = (d, T_enc); average adjacent frame pairs via even/odd views.
    const int T_pool = T_enc / 2; // 750
    ggml_tensor* even = ggml_view_2d(ctx0, cur, d, T_pool, cur->nb[1] * 2, 0);
    ggml_tensor* odd = ggml_view_2d(ctx0, cur, d, T_pool, cur->nb[1] * 2, cur->nb[1]);
    cur = ggml_scale(ctx0, ggml_add(ctx0, ggml_cont(ctx0, even), ggml_cont(ctx0, odd)), 0.5f);

    cur = ggml_norm(ctx0, cur, kLayerNormEps);
    cur = ggml_mul(ctx0, cur, m.audio.ln_post_w);
    cur = ggml_add(ctx0, cur, m.audio.ln_post_b);
    ggml_set_name(cur, "encoder_hidden"); // (d=1280, 750) — 25 fps, pre-projector
    ggml_set_output(cur);

    // ---- Projector: depthwise temporal conv (s2) + Linear/ReLU/Linear ----
    // conv_1d_dw expects input (T, C) + weight (K, 1, C) -> output (T_out, C, 1).
    ggml_tensor* xt = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // (T_pool, d)
    ggml_tensor* tconv = ggml_conv_1d_dw(ctx0, m.audio.proj_temporal_w, xt, /*s*/ 2, /*p*/ 1, /*d*/ 1);
    if (ggml_n_dims(tconv) > 2)
        tconv = ggml_reshape_2d(ctx0, tconv, tconv->ne[0], tconv->ne[1]); // (T_out, d)
    ggml_tensor* p = ggml_cont(ctx0, ggml_transpose(ctx0, tconv));        // (d, T_out)
    p = ggml_add(ctx0, p, m.audio.proj_temporal_b);
    p = ggml_add(ctx0, ggml_mul_mat(ctx0, m.audio.proj_linear1_w, p), m.audio.proj_linear1_b);
    p = ggml_relu(ctx0, p);
    p = ggml_add(ctx0, ggml_mul_mat(ctx0, m.audio.proj_linear2_w, p), m.audio.proj_linear2_b);
    // p ne = (proj_out=2048, T_out=375) — 12.5 tps audio embeddings for the LLM.
    ggml_set_name(p, "encoder_out");
    ggml_set_output(p);
    ggml_build_forward_expand(gf, p);
    if (!arena_ctx)
        ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Qwen3 LLM forward graph (Stage 3)
//
// Architecture: 28 layers, hidden=1024, GQA(16/8), head_dim=128, RMSNorm,
//   SwiGLU FFN, RoPE θ=1e6 NEOX-style, Q-norm/K-norm per-head along head_dim.
//
// Pipeline:
//   x = embed(input_ids)              # (1024, T)
//   for layer in 28 layers:
//     residual = x
//     x = RMSNorm(x) * attn_norm_w
//     Q = q_proj(x).view(head_dim, n_q,  T)
//     K = k_proj(x).view(head_dim, n_kv, T)
//     V = v_proj(x).view(head_dim, n_kv, T)
//     Q = q_norm(Q) along head_dim
//     K = k_norm(K) along head_dim
//     Q = rope_neox(Q, positions)
//     K = rope_neox(K, positions)
//     # GQA: repeat K, V from n_kv to n_q heads
//     K_rep = repeat_each(K, n_q / n_kv)   # (head_dim, n_q, T)
//     V_rep = repeat_each(V, n_q / n_kv)
//     # Standard attention
//     scores = (Q @ K_rep^T) * (1/sqrt(head_dim)) + causal_mask
//     attn   = softmax(scores) @ V_rep
//     attn   = o_proj(attn.reshape(d, T))
//     x = residual + attn
//     residual = x
//     x = RMSNorm(x) * ffn_norm_w
//     x = down_proj(silu(gate_proj(x)) * up_proj(x))
//     x = residual + x
//   x = RMSNorm(x) * output_norm_w
//   logits = lm_head(x)               # (vocab, T)
//
// First iteration: no KV cache, full forward each call. Used for diff testing.
// ===========================================================================

// Internal: builds the 28-layer transformer + lm_head graph starting from
// a (d, T) hidden state. Used by both build_graph_llm (which prepends a
// get_rows token-embed lookup) and build_graph_llm_from_embeds (which takes
// pre-computed embeddings as input).
static void higgs_stt_build_llm_body(higgs_stt_context* ctx, ggml_context* ctx0, ggml_cgraph* gf,
                                     ggml_tensor* cur, // (d, T) input hidden state
                                     ggml_tensor* positions, ggml_tensor* causal_mask, int T) {
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

    for (uint32_t il = 0; il < hp.llm_n_layers; il++) {
        const auto& b = m.llm.blocks[il];
        ggml_tensor* residual = cur;

        // ---- LN1 (RMSNorm + multiplicative weight, no bias) ----
        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        // ---- Q, K, V projections ----
        ggml_tensor* Q = ggml_mul_mat(ctx0, b.attn_q_w, x);
        ggml_tensor* K = ggml_mul_mat(ctx0, b.attn_k_w, x);
        ggml_tensor* V = ggml_mul_mat(ctx0, b.attn_v_w, x);
        Q = ggml_reshape_3d(ctx0, Q, hd, n_q, T);
        K = ggml_reshape_3d(ctx0, K, hd, n_kv, T);
        V = ggml_reshape_3d(ctx0, V, hd, n_kv, T);

        // ---- Q-norm / K-norm ----
        Q = ggml_rms_norm(ctx0, Q, eps);
        Q = ggml_mul(ctx0, Q, b.attn_q_norm_w);
        K = ggml_rms_norm(ctx0, K, eps);
        K = ggml_mul(ctx0, K, b.attn_k_norm_w);

        // ---- RoPE NEOX ----
        Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, (int)hp.llm_max_pos, theta, 1.0f, 0.0f,
                          1.0f, 32.0f, 1.0f);
        K = ggml_rope_ext(ctx0, K, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, (int)hp.llm_max_pos, theta, 1.0f, 0.0f,
                          1.0f, 32.0f, 1.0f);

        // ---- GQA expand ----
        if (n_kv_grp > 1) {
            ggml_tensor* K4 = ggml_reshape_4d(ctx0, K, hd, 1, n_kv, T);
            ggml_tensor* V4 = ggml_reshape_4d(ctx0, V, hd, 1, n_kv, T);
            K4 = ggml_repeat_4d(ctx0, K4, hd, n_kv_grp, n_kv, T);
            V4 = ggml_repeat_4d(ctx0, V4, hd, n_kv_grp, n_kv, T);
            K = ggml_cont(ctx0, ggml_reshape_3d(ctx0, K4, hd, n_q, T));
            V = ggml_cont(ctx0, ggml_reshape_3d(ctx0, V4, hd, n_q, T));
        }

        // ---- Permute for attention ----
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q);
        scores = ggml_add(ctx0, scores, causal_mask);
        scores = ggml_soft_max_ext(ctx0, scores, nullptr, attn_scale, 0.0f);

        ggml_tensor* V2 = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3));
        ggml_tensor* attn = ggml_mul_mat(ctx0, V2, scores);
        attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(ctx0, attn, hd * n_q, T);

        attn = ggml_mul_mat(ctx0, b.attn_output_w, attn);
        cur = ggml_add(ctx0, residual, attn);

        // ---- FFN ----
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.llm.output_norm_w);
    cur = ggml_mul_mat(ctx0, m.llm.output_w, cur);

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    (void)d;
}

static ggml_cgraph* higgs_stt_build_graph_llm(higgs_stt_context* ctx, int n_tokens) {
    const auto& m = ctx->model;
    const int T = n_tokens;

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // ------- Inputs -------
    ggml_tensor* input_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(input_ids, "input_ids");
    ggml_set_input(input_ids);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, T);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);

    // Token embedding lookup → (d, T)
    ggml_tensor* cur = ggml_get_rows(ctx0, m.llm.token_embd_w, input_ids);
    higgs_stt_build_llm_body(ctx, ctx0, gf, cur, positions, causal_mask, T);
    ggml_free(ctx0);
    return gf;
}

// Variant: takes pre-computed inputs_embeds (d, T) F32 instead of input_ids.
// Used by the audio-injection path after splicing audio frames into the
// text-token embedding sequence.
static ggml_cgraph* higgs_stt_build_graph_llm_from_embeds(higgs_stt_context* ctx, int n_tokens) {
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    const int T = n_tokens;

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

    ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, T);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);

    higgs_stt_build_llm_body(ctx, ctx0, gf, embeds, positions, causal_mask, T);
    ggml_free(ctx0);
    return gf;
}

// Graph builder for the KV-cached LLM forward. Used by both prefill
// (n_past=0, n_tokens=T_prompt) and incremental decode (n_past>0, n_tokens=1).
//
// Inputs:
//   inputs_embeds: F32 (d, n_tokens)
//   positions:     I32 (n_tokens,) — absolute positions n_past, n_past+1, ...
//   causal_mask:   F32 (n_kv_total, n_tokens) where n_kv_total = n_past+n_tokens
//                  mask[k, q] = 0 if k <= n_past+q else -inf
//
// Per layer, the new K/V are written into the persistent cache at positions
// [n_past, n_past+n_tokens) and attention reads from [0, n_past+n_tokens).
static ggml_cgraph* higgs_stt_build_graph_llm_kv(higgs_stt_context* ctx, int n_past, int n_tokens,
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
    const int Lk = n_past + T; // total cache length after this call

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

    // Only the prefill path (T > 1) needs the causal mask. Decode (T = 1)
    // uses ggml_flash_attn_ext with no mask — the single new query attends
    // to all cached keys including itself. If we always declared the mask
    // input, the scheduler would optimize it away on the decode path and
    // ggml_graph_get_tensor("causal_mask") would return null.
    //
    // For prefill we use flash_attn_ext too, which requires the mask to
    // be F16 (and contiguous, broadcast-compatible with Q's trailing dims).
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
        /*n_ctx_orig*/ (int)hp.llm_max_pos,
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

        // KV-cached GQA self-attention with Q/K norm — qwen3 is the only
        // backend currently passing non-null q_norm_w/k_norm_w to the
        // shared helper.
        ggml_tensor* attn = core_attn::kv_self_attn(
            ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w, b.attn_q_norm_w, b.attn_k_norm_w,
            positions, (T == 1) ? nullptr : causal_mask, ctx->kv_k, ctx->kv_v, (int)il, n_past, kvp, b.attn_qkv_w);
        cur = ggml_add(ctx0, residual, attn);

        // ---- FFN ----
        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.llm.output_norm_w);

    // Last-token-only lm_head (default): slice (d, T) → (d, 1) before
    // the big matmul. The autoregressive decode loop only ever needs the
    // next-token logits, so we save the (152064, 2048) matmul on T-1
    // columns.
    //
    // Forced-alignment mode (last_token_only=false) keeps all T columns
    // because the FA model needs the lm_head output at every position
    // where input == <|timestamp|>. The lm_head shape itself is read
    // from output_w (5000 for FA, vocab_size for ASR).
    if (last_token_only && T > 1) {
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }
    cur = ggml_mul_mat(ctx0, m.llm.output_w, cur);
    // logits ne = (lm_head_dim, 1)        (decode / last-token mode)
    //         or (lm_head_dim, T)         (FA / full-T mode)

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Tiny standalone graph for token embedding lookup (used by run_embed_tokens).
static ggml_cgraph* higgs_stt_build_graph_embed(higgs_stt_context* ctx, int n_tokens) {
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
// Public API
// ===========================================================================

extern "C" const char* higgs_stt_token_text(higgs_stt_context* ctx, int id) {
    if (!ctx || id < 0 || id >= (int)ctx->vocab.id_to_token.size())
        return "";
    return ctx->vocab.id_to_token[id].c_str();
}

// ===========================================================================
// BPE tokenizer (GPT-2 byte-level, Qwen2/Qwen3 compatible)
//
// Two-stage encode:
//   1. Pre-tokenize: split the input on `<|special|>` markers and on
//      whitespace boundaries. Special tokens are looked up directly in the
//      vocab as full strings; the remaining text segments are byte-encoded
//      via the GPT-2 byte→unicode mapping and then BPE-merged.
//   2. BPE merge loop: for each pre-token, find the lowest-rank merge in
//      the merge table and apply it, repeating until no merges apply.
//      The final symbols are then looked up in the vocab.
// ===========================================================================

// All four building blocks (byte_encoder, utf8_encode, bytes_to_unicode,
// bpe_one) used to live inline here as `qwen3_*` statics. They now live
// once in src/core/bpe.h — qwen3 calls into them via the namespace alias
// below, granite uses the same primitives, and any future GPT-2-family
// model gets them for free.
#include "core/bpe.h"
#include "core/beam_decode.h"
#include "core/gpu_backend_pref.h" // crispasr_init_gpu_backend (#214)
#include "core/ngram_loop_fix.h"   // core_ngram::fix_loops (shared with moss-transcribe)
#include "core/ggml_cpu_backend.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
extern "C" int32_t* higgs_stt_tokenize(higgs_stt_context* ctx, const char* text, int* out_n_tokens) {
    if (!ctx || !text) {
        if (out_n_tokens)
            *out_n_tokens = 0;
        return nullptr;
    }
    const auto& v = ctx->vocab;
    std::vector<int32_t> result;

    const std::string s = text;
    size_t i = 0;
    while (i < s.size()) {
        // 1. Special-token check: if the next chars are "<|...|>" and the
        //    full token exists in the vocab, emit it directly.
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

        // 2. Plain text segment: collect chars up to the next "<|...|>" we
        //    can recognize. We treat a "<|" as a candidate boundary only if
        //    it's an actual special token we know — otherwise we keep
        //    extending the plain-text segment past it (a literal "<|" in
        //    user text isn't a special token).
        size_t j = i;
        // Ensure we always advance by at least one char on every outer
        // iteration, even if step 1 just failed on a "<|...|>" lookalike
        // that isn't actually a special token.
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

        // 3. Pre-split the chunk on whitespace boundaries the way GPT-2
        //    does: each pre-token starts at a non-space char and includes
        //    the leading space if present (Ġ marker). We approximate this
        //    by splitting on transitions between "leading space + word",
        //    "word", "punctuation", etc.
        size_t k = 0;
        while (k < chunk.size()) {
            size_t start = k;
            // Optional leading whitespace (single char)
            if (chunk[k] == ' ' || chunk[k] == '\t' || chunk[k] == '\n')
                k++;
            // The "word body" — everything until the next whitespace OR a
            // standalone punctuation transition.
            while (k < chunk.size() && chunk[k] != ' ' && chunk[k] != '\t' && chunk[k] != '\n') {
                k++;
            }
            if (k == start)
                k++; // pure whitespace boundary, advance one
            std::string pre(chunk, start, k - start);
            // 4. Byte-encode and BPE-merge via the shared helpers.
            std::string encoded = core_bpe::bytes_to_unicode(pre.data(), pre.size());
            core_bpe::bpe_one(v.token_to_id, v.merge_rank, encoded, result);
        }
    }

    if (out_n_tokens)
        *out_n_tokens = (int)result.size();
    int32_t* out = (int32_t*)malloc(result.size() * sizeof(int32_t));
    if (!out) {
        if (out_n_tokens)
            *out_n_tokens = 0;
        return nullptr;
    }
    std::memcpy(out, result.data(), result.size() * sizeof(int32_t));
    return out;
}

extern "C" higgs_stt_context_params higgs_stt_context_default_params(void) {
    higgs_stt_context_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.flash_attn = true;
    return p;
}

extern "C" higgs_stt_context* higgs_stt_init_from_file(const char* path, higgs_stt_context_params params) {
    higgs_stt_context* ctx = new higgs_stt_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    if (path)
        ctx->model_path = path;

    // Try GPU backend first (Metal, CUDA, Vulkan...), fall back to CPU.
    // crispasr_init_gpu_backend() picks the highest-priority available backend.
    ctx->backend = params.use_gpu ? crispasr_init_gpu_backend() : core_cpu_backend::init();
    if (!ctx->backend)
        ctx->backend = core_cpu_backend::init();
    ctx->backend_cpu = core_cpu_backend::init();
    if (ctx->backend_cpu) {
        core_cpu_backend::set_n_threads(ctx->backend_cpu, ctx->n_threads);
    }
    if (core_cpu_backend::is_cpu(ctx->backend)) {
        core_cpu_backend::set_n_threads(ctx->backend, ctx->n_threads);
    }

    if (!higgs_stt_load_model(ctx->model, ctx->vocab, path, ctx->backend, ctx->backend_cpu)) {
        delete ctx;
        return nullptr;
    }

    // ---- Fuse Q+K+V weights for single-matmul LLM attention ----
    // PLAN #60d: type gate dropped May 2026 — concatenating along the
    // output axis is byte-concat for any row-wise quantized format
    // (Q4_K, Q4_0, Q5_K, Q8_0, ...) just as it is for F16/F32.
    // Each output row is a self-contained block group; no requantization.
    // Buffer allocation switched from CPU to default-backend buffer:
    // for Q-format weights on Metal the CPU-buffer path would pay a
    // backend-transfer cost per matmul. See LEARNINGS § "runtime
    // QKV/MLP fusion on row-wise quantized weights is just byte-concat".
    // Opt-out: CRISPASR_HIGGS_STT_FUSED_QKV=0.
    {
        const char* fuse_env = getenv("CRISPASR_HIGGS_STT_FUSED_QKV");
        const bool fuse_enabled = (fuse_env == nullptr) || (atoi(fuse_env) != 0);
        auto& blocks = ctx->model.llm.blocks;
        bool can_fuse =
            fuse_enabled && !blocks.empty() && blocks[0].attn_q_w && blocks[0].attn_k_w && blocks[0].attn_v_w;
        if (can_fuse) {
            const ggml_type t0 = blocks[0].attn_q_w->type;
            for (auto& b : blocks) {
                if (!b.attn_q_w || !b.attn_k_w || !b.attn_v_w || b.attn_q_w->type != t0 || b.attn_k_w->type != t0 ||
                    b.attn_v_w->type != t0 || b.attn_q_w->ne[0] != b.attn_k_w->ne[0] ||
                    b.attn_q_w->ne[0] != b.attn_v_w->ne[0]) {
                    can_fuse = false;
                    break;
                }
            }
        }
        if (can_fuse) {
            int q_out = (int)blocks[0].attn_q_w->ne[1];
            int k_out = (int)blocks[0].attn_k_w->ne[1];
            int hidden = (int)blocks[0].attn_q_w->ne[0];
            int qkv_out = q_out + 2 * k_out;
            size_t fused_mem = ggml_tensor_overhead() * blocks.size() + 256;
            ggml_init_params fgp = {fused_mem, nullptr, true};
            ctx->fused_ctx = ggml_init(fgp);
            if (ctx->fused_ctx) {
                for (auto& b : blocks) {
                    b.attn_qkv_w = ggml_new_tensor_2d(ctx->fused_ctx, b.attn_q_w->type, hidden, qkv_out);
                }
                ctx->fused_buf = ggml_backend_alloc_ctx_tensors_from_buft(
                    ctx->fused_ctx, ggml_backend_get_default_buffer_type(ctx->backend));
                if (ctx->fused_buf) {
                    for (auto& b : blocks) {
                        size_t qb = ggml_nbytes(b.attn_q_w), kb = ggml_nbytes(b.attn_k_w);
                        std::vector<uint8_t> tmp(qb + 2 * kb);
                        ggml_backend_tensor_get(b.attn_q_w, tmp.data(), 0, qb);
                        ggml_backend_tensor_get(b.attn_k_w, tmp.data() + qb, 0, kb);
                        ggml_backend_tensor_get(b.attn_v_w, tmp.data() + qb + kb, 0, kb);
                        ggml_backend_tensor_set(b.attn_qkv_w, tmp.data(), 0, tmp.size());
                    }
                    if (params.verbosity >= 1)
                        fprintf(stderr, "higgs_stt: fused QKV for %zu LLM layers (%d+%d+%d→%d, type=%s)\n",
                                blocks.size(), q_out, k_out, k_out, qkv_out, ggml_type_name(blocks[0].attn_q_w->type));
                } else {
                    ggml_free(ctx->fused_ctx);
                    ctx->fused_ctx = nullptr;
                    for (auto& b : blocks)
                        b.attn_qkv_w = nullptr;
                }
            }
        }
    }

    // Create the backend scheduler once with the worst-case node budget.
    // All compute functions reuse this scheduler via ggml_backend_sched_reset().
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
        crispasr_imatrix_install(ctx->sched); // no-op unless CRISPASR_IMATRIX_OUT is set
    }
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    if (params.verbosity >= 1) {
        fprintf(stderr, "higgs_stt: loaded %s  (audio %u layers, llm %u layers, vocab %u)\n", path,
                ctx->model.hparams.audio_n_layers, ctx->model.hparams.llm_n_layers,
                (uint32_t)ctx->vocab.id_to_token.size());
    }
    return ctx;
}

extern "C" void higgs_stt_free(higgs_stt_context* ctx) {
    if (!ctx)
        return;
    if (ctx->audio_ca) {
        crisp_audio_free(ctx->audio_ca);
        ctx->audio_ca = nullptr;
    }
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->fused_buf)
        ggml_backend_buffer_free(ctx->fused_buf);
    if (ctx->fused_ctx)
        ggml_free(ctx->fused_ctx);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->model.buf)
        core_gguf::release_weight_buffer(ctx->model.buf);
    if (ctx->model.buf_cpu)
        core_gguf::release_weight_buffer(ctx->model.buf_cpu);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    // Free the primary backend last — buffers above were allocated against it,
    // and on Metal an unreleased backend leaves the residency set live and
    // trips ggml_metal_rsets_free's assert at process exit.
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

// Greedy decode on this model degenerates into repeated n-gram loops; the
// upstream pipeline collapses them in `_post_process` (ngram_loop_fix.py) rather
// than constraining generation, so we reproduce that here via the shared
// core_ngram::fix_loops helper (also used by moss-transcribe, issue #218).

// Encode audio the way higgs-audio does: split into chunk_size_seconds (4 s)
// chunks, run the Whisper tower + projector on EACH chunk independently (chunk-
// local positions + within-chunk attention), then concatenate the per-chunk
// audio embeddings (frame-major: [frame*pdim + d]). Encoding the whole clip as
// one padded 30 s window corrupts the result — every valid frame attends
// globally to the ~1900 silence-pad frames — and derails the decoder. Each
// chunk is encoded at its true length (no padding/mask): the per-chunk
// GlobalClipMax norm is dominated by speech frames, so this matches the
// reference's zero-padded chunk for the valid frames, and the valid token count
// falls out exactly. Returns a malloc'd buffer the caller frees.
extern "C" float* higgs_stt_encode_audio(higgs_stt_context* ctx, const float* samples, int n_samples, int* out_N,
                                         int* out_pdim) {
    const auto& hp = ctx->model.hparams;
    const int chunk_samples = 4 * (int)hp.sample_rate; // chunk_size_seconds=4.0 -> 64000
    const int hop = (int)hp.hop_length;                // 160
    int pdim = 0;
    std::vector<float> audio_embeds; // frame-major: [frame*pdim + d]
    for (int off = 0; off < n_samples; off += chunk_samples) {
        const int clen = std::min(chunk_samples, n_samples - off);
        if (clen < 16 * hop) // skip a negligible (<160 ms) trailing remainder
            continue;
        int cm = 0, ct = 0;
        float* mel = higgs_stt_compute_mel(ctx, samples + off, clen, &cm, &ct);
        if (!mel)
            continue;
        int N_c = 0, pd = 0;
        float* emb = higgs_stt_run_encoder(ctx, mel, cm, ct, &N_c, &pd);
        free(mel);
        if (!emb || N_c <= 0) {
            free(emb);
            continue;
        }
        pdim = pd;
        audio_embeds.insert(audio_embeds.end(), emb, emb + (size_t)N_c * pd);
        free(emb);
    }
    const int N = pdim > 0 ? (int)(audio_embeds.size() / pdim) : 0;
    if (out_N)
        *out_N = N;
    if (out_pdim)
        *out_pdim = pdim;
    if (N <= 0)
        return nullptr;
    float* audio = (float*)malloc(audio_embeds.size() * sizeof(float));
    std::memcpy(audio, audio_embeds.data(), audio_embeds.size() * sizeof(float));
    return audio;
}

// Override the transcription instruction (the ChatML user-turn text). Pass
// nullptr or "" to restore the default. Used to inject a language hint
// ("Transcribe the speech in German. …") or a custom task prompt.
extern "C" void higgs_stt_set_ask(higgs_stt_context* ctx, const char* instruction) {
    if (!ctx)
        return;
    ctx->ask = (instruction && *instruction) ? instruction : "";
}

// Beam search width. 1 = greedy (default); >1 = beam search (core_beam_decode).
extern "C" void higgs_stt_set_beam_size(higgs_stt_context* ctx, int beam_size) {
    if (ctx)
        ctx->beam_size = beam_size > 0 ? beam_size : 1;
}

// #292: forward --max-new-tokens. <= 0 keeps the backend's 1024 default (the KV
// cache is sized from this, so raising it grows the cache accordingly).
extern "C" void higgs_stt_set_max_new_tokens(higgs_stt_context* ctx, int n) {
    if (ctx && n > 0)
        ctx->max_new_tokens = n;
}

extern "C" char* higgs_stt_transcribe(higgs_stt_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0)
        return strdup("");
    const auto& hp = ctx->model.hparams;

    // ---- chunked mel + audio encoder/projector ----
    int N_enc = 0, pdim = 0;
    float* audio = higgs_stt_encode_audio(ctx, samples, n_samples, &N_enc, &pdim);
    if (!audio || N_enc <= 0)
        return strdup("");

    // ---- build the ChatML prompt token ids ----
    //   <|im_start|>user\n{instruction}<|audio_bos|>{N x <|AUDIO|>}<|audio_eos|>
    //   <|im_end|>\n<|im_start|>assistant\n
    auto tok = [&](const char* s) {
        int n = 0;
        int32_t* a = higgs_stt_tokenize(ctx, s, &n);
        std::vector<int32_t> v;
        if (a && n > 0)
            v.assign(a, a + n);
        free(a);
        return v;
    };
    std::vector<int32_t> ids;
    auto append = [&](const std::vector<int32_t>& v) { ids.insert(ids.end(), v.begin(), v.end()); };
    ids.push_back((int32_t)hp.im_start_token_id);
    append(tok("user\n"));
    append(tok(ctx->ask.empty()
                   ? "Transcribe the speech. Output only the spoken words in lowercase with no punctuation."
                   : ctx->ask.c_str()));
    ids.push_back((int32_t)hp.audio_start_token_id); // <|audio_bos|>
    for (int i = 0; i < N_enc; i++)
        ids.push_back((int32_t)hp.audio_pad_token_id); // <|AUDIO|>
    ids.push_back((int32_t)hp.audio_end_token_id);     // <|audio_eos|>
    ids.push_back((int32_t)hp.im_end_token_id);
    append(tok("\n"));
    ids.push_back((int32_t)hp.im_start_token_id);
    append(tok("assistant\n"));
    const int T_prompt = (int)ids.size();

    // ---- embed prompt, splice projected audio at the <|AUDIO|> positions ----
    float* te = higgs_stt_embed_tokens(ctx, ids.data(), T_prompt);
    if (!te) {
        free(audio);
        return strdup("");
    }
    int spliced = 0;
    for (int i = 0; i < T_prompt && spliced < N_enc; i++) {
        if (ids[i] == (int32_t)hp.audio_pad_token_id) {
            std::memcpy(te + (size_t)i * pdim, audio + (size_t)spliced * pdim, (size_t)pdim * sizeof(float));
            spliced++;
        }
    }
    free(audio);

    // ---- KV-cached greedy decode ----
    const int higgs_max_new = ctx->max_new_tokens > 0 ? ctx->max_new_tokens : 1024;
    if (!higgs_stt_kv_init(ctx, std::max(4096, T_prompt + higgs_max_new))) {
        free(te);
        return strdup("");
    }
    higgs_stt_kv_reset(ctx);
    int nt = 0, vs = 0;
    float* logits = higgs_stt_run_llm_kv(ctx, te, T_prompt, 0, &nt, &vs);
    free(te);
    int n_past = T_prompt;
    std::vector<int32_t> out_ids;
    const int max_new = higgs_max_new; // #292: honors --max-new-tokens (see kv_init above)
    const int eos1 = (int)hp.im_end_token_id;
    const int eos2 = (int)hp.pad_token_id; // <|endoftext|>

    if (ctx->beam_size > 1 && logits) {
        // Beam search via core_beam_decode replay-from-prefix (§167g). The
        // prompt's K/V occupy slots [0, T_prompt); each beam-step rebuilds its
        // suffix by replaying from that anchor.
        auto replay = [](higgs_stt_context* c, const int32_t* toks, int n, int prompt_len) -> float* {
            float* emb = higgs_stt_embed_tokens(c, toks, n);
            if (!emb)
                return nullptr;
            float* lg = higgs_stt_run_llm_kv(c, emb, n, prompt_len, nullptr, nullptr);
            free(emb);
            return lg;
        };
        core_beam_decode::Config cfg;
        cfg.max_new_tokens = max_new;
        cfg.eos_ids = {eos1, eos2};
        cfg.vocab_size = vs;
        cfg.beam_size = ctx->beam_size;
        cfg.prompt_len = T_prompt;
        auto br = core_beam_decode::run_with_probs(ctx, logits, replay, cfg);
        logits = nullptr; // ownership transferred to core_beam_decode
        out_ids = std::move(br.tokens);
        if (!out_ids.empty() && (out_ids.back() == eos1 || out_ids.back() == eos2))
            out_ids.pop_back();
    } else {
        for (int step = 0; step < max_new && logits; step++) {
            int best = 0;
            float bv = logits[0];
            for (int i = 1; i < vs; i++)
                if (logits[i] > bv) {
                    bv = logits[i];
                    best = i;
                }
            free(logits);
            logits = nullptr;
            if (best == eos1 || best == eos2)
                break;
            out_ids.push_back(best);
            int32_t t = best;
            float* oe = higgs_stt_embed_tokens(ctx, &t, 1);
            if (!oe)
                break;
            logits = higgs_stt_run_llm_kv(ctx, oe, 1, n_past, &nt, &vs);
            free(oe);
            n_past++;
        }
        if (logits)
            free(logits);
    }

    // ---- detokenize + post-process ----
    std::string text = core_bpe::detokenize(ctx->vocab.id_to_token, out_ids.data(), out_ids.size());
    if (crispasr_env::get("CRISPASR_HIGGS_DEBUG")) {
        fprintf(stderr, "[higgs] raw %zu tokens, raw text:\n>>>%s<<<\n", out_ids.size(), text.c_str());
    }
    // Strip an optional <think>...</think> reasoning block.
    size_t ts = text.find("<think>");
    if (ts != std::string::npos) {
        size_t te2 = text.find("</think>", ts);
        if (te2 != std::string::npos)
            text.erase(ts, te2 + 8 - ts);
    }
    // Trim surrounding whitespace.
    size_t b = text.find_first_not_of(" \t\r\n");
    size_t e = text.find_last_not_of(" \t\r\n");
    text = (b == std::string::npos) ? "" : text.substr(b, e - b + 1);
    // Collapse degenerate greedy n-gram loops (model's _post_process step).
    text = core_ngram::fix_loops(text);
    return strdup(text.c_str());
}

// Pad/truncate a (n_mels, T_mel) mel to the fixed (n_mels, 3000) Whisper window.
// Source layout is row-major (n_mels outer, time inner): mel[f*T_mel + t].
// Target ggml "mel" tensor is ne=(3000, n_mels): buf[t + 3000*f].
static std::vector<float> higgs_stt_pad_mel(const float* mel, int n_mels, int T_mel) {
    const int T = 3000;
    std::vector<float> out((size_t)T * n_mels, 0.0f);
    const int tcopy = std::min(T_mel, T);
    for (int f = 0; f < n_mels; f++)
        for (int t = 0; t < tcopy; t++)
            out[(size_t)t + (size_t)T * f] = mel[(size_t)f * T_mel + t];
    return out;
}

extern "C" float* higgs_stt_run_conv(higgs_stt_context* ctx, const float* mel_features, int n_mels, int T_mel,
                                     int* out_n_chunks, int* out_T_chunk_out, int* out_d) {
    if (!ctx || !mel_features)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    if (n_mels != (int)hp.n_mels) {
        fprintf(stderr, "higgs_stt: mel feature mismatch (%d vs %d)\n", n_mels, (int)hp.n_mels);
        return nullptr;
    }
    std::vector<float> mel = higgs_stt_pad_mel(mel_features, n_mels, T_mel);

    ggml_cgraph* gf = higgs_stt_build_graph_conv(ctx);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "higgs_stt: failed to alloc conv graph\n");
        return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mel"), mel.data(), 0, mel.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "higgs_stt: conv graph compute failed\n");
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "conv_front_out");
    if (!out)
        return nullptr;
    const int d = (int)out->ne[0]; // 1280
    const int T = (int)out->ne[1]; // 1500
    if (out_d)
        *out_d = d;
    if (out_T_chunk_out)
        *out_T_chunk_out = T;
    if (out_n_chunks)
        *out_n_chunks = 1;
    const size_t total = (size_t)d * T;
    float* result = (float*)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

// Run the full audio encoder + projector. Returns the projected audio
// embeddings (proj_out=2048, N=375 for a 30 s window) — the frames spliced
// into the LLM at the <|AUDIO|> positions. Caller frees with free().
extern "C" float* higgs_stt_run_encoder(higgs_stt_context* ctx, const float* mel_features, int n_mels, int T_mel,
                                        int* out_N_total, int* out_proj_dim) {
    if (!ctx || !mel_features)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    if (n_mels != (int)hp.n_mels) {
        fprintf(stderr, "higgs_stt: mel feature mismatch (%d vs %d)\n", n_mels, (int)hp.n_mels);
        return nullptr;
    }
    // Lay the row-major (n_mels, T_mel) mel into the graph's (T_mel, n_mels)
    // input tensor: buf[t + T_mel*f] = mel[f*T_mel + t]. No padding — the chunk
    // is encoded at its true length.
    std::vector<float> mel((size_t)T_mel * n_mels);
    for (int f = 0; f < n_mels; f++)
        for (int t = 0; t < T_mel; t++)
            mel[(size_t)t + (size_t)T_mel * f] = mel_features[(size_t)f * T_mel + t];

    ggml_cgraph* gf = higgs_stt_build_graph_encoder(ctx, T_mel);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "higgs_stt: failed to alloc encoder graph\n");
        return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mel"), mel.data(), 0, mel.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "higgs_stt: encoder graph compute failed\n");
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "encoder_out");
    if (!out) {
        fprintf(stderr, "higgs_stt: missing encoder_out tensor\n");
        return nullptr;
    }
    const int pdim = (int)out->ne[0]; // 2048
    const int N = (int)out->ne[1];    // 375
    if (out_proj_dim)
        *out_proj_dim = pdim;
    if (out_N_total)
        *out_N_total = N;
    const size_t total = (size_t)pdim * N;
    float* result = (float*)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

extern "C" bool higgs_stt_kv_init(higgs_stt_context* ctx, int max_ctx) {
    if (!ctx || max_ctx <= 0)
        return false;
    if (ctx->kv_k)
        return true; // already initialized

    const auto& hp = ctx->model.hparams;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int n_lay = (int)hp.llm_n_layers;

    ggml_init_params kp = {
        /*mem_size=*/ggml_tensor_overhead() * 4 + 1024,
        /*mem_buffer=*/nullptr,
        /*no_alloc=*/true,
    };
    ctx->kv_ctx = ggml_init(kp);
    // F16 KV cache: halves memory + ~2× cache read bandwidth on decode.
    // Conversion happens at the ggml_cpy() write into the cache view, and
    // ggml_mul_mat handles F16-on-F32 dot products natively for the read path.
    // PLAN #60e + #69e: per-half KV dtype. CRISPASR_KV_QUANT sets both,
    // CRISPASR_KV_QUANT_{K,V} override per half (default f16/f16).
    const auto kv_pair = core_attn::kv_dtype_pair_from_env("higgs_stt");
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.k, hd, max_ctx, n_kv, n_lay);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.v, hd, max_ctx, n_kv, n_lay);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");

    const size_t kbytes = ggml_nbytes(ctx->kv_k);
    const size_t vbytes = ggml_nbytes(ctx->kv_v);
    // PLAN #69b: optional KV-on-CPU spill for long-context / tight-VRAM users.
    ggml_backend_t kv_backend = core_attn::kv_backend_from_env(ctx->backend, ctx->backend_cpu, "higgs_stt");
    ctx->kv_buf = ggml_backend_alloc_buffer(kv_backend, kbytes + vbytes);
    if (!ctx->kv_buf) {
        fprintf(stderr, "higgs_stt: failed to allocate kv buffer\n");
        return false;
    }
    char* base = (char*)ggml_backend_buffer_get_base(ctx->kv_buf);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + kbytes);
    ctx->kv_max_ctx = max_ctx;
    ctx->kv_n_used = 0;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "higgs_stt: kv cache %d MiB (head_dim=%d max_ctx=%d n_kv=%d n_layers=%d)\n",
                (int)((kbytes + vbytes) / 1048576), hd, max_ctx, n_kv, n_lay);
    }
    return true;
}

extern "C" void higgs_stt_kv_reset(higgs_stt_context* ctx) {
    if (ctx)
        ctx->kv_n_used = 0;
}

extern "C" float* higgs_stt_run_llm_kv(higgs_stt_context* ctx, const float* inputs_embeds, int n_tokens, int n_past,
                                       int* out_n_tokens, int* out_vocab_size) {
    if (!ctx || !inputs_embeds || n_tokens <= 0)
        return nullptr;
    if (!ctx->kv_k) {
        fprintf(stderr, "higgs_stt: kv cache not initialized — call higgs_stt_kv_init first\n");
        return nullptr;
    }
    if (n_past + n_tokens > ctx->kv_max_ctx) {
        fprintf(stderr, "higgs_stt: kv overflow (n_past=%d + n_tokens=%d > max_ctx=%d)\n", n_past, n_tokens,
                ctx->kv_max_ctx);
        return nullptr;
    }
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    // Use the LM head's actual output dim, not the token vocab size.
    // For ASR models the two are equal; for the ForcedAligner variant
    // the head is (5000, d) while vocab_size stays 152064.
    const int vocab = (int)(hp.llm_lm_head_dim ? hp.llm_lm_head_dim : hp.llm_vocab_size);
    const int Lk = n_past + n_tokens;

    // Positions [n_past, n_past+T)
    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = n_past + i;

    // Causal mask: only needed for prefill (T > 1). Decode (T = 1) uses
    // ggml_flash_attn_ext with no mask — the single new query attends to
    // all cached keys including itself, so no masking is needed.
    // ggml_flash_attn_ext requires the mask to be F16.
    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
        mask.assign((size_t)Lk * n_tokens, zero_h);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = n_past + q + 1; k < Lk; k++) {
                mask[(size_t)q * Lk + k] = neginf_h;
            }
        }
    }

    ggml_cgraph* gf = higgs_stt_build_graph_llm_kv(ctx, n_past, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "higgs_stt: failed to alloc llm_kv graph\n");
        return nullptr;
    }

    ggml_tensor* embeds_in = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(embeds_in, inputs_embeds, 0, (size_t)d * n_tokens * sizeof(float));
    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(pos_in, positions.data(), 0, positions.size() * sizeof(int32_t));
    if (n_tokens > 1) {
        ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "higgs_stt: llm_kv graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    if (!out)
        return nullptr;
    ctx->kv_n_used = n_past + n_tokens;
    // Output is only the last token's logits — see lm_head slice in build_graph_llm_kv.
    if (out_n_tokens)
        *out_n_tokens = 1;
    if (out_vocab_size)
        *out_vocab_size = vocab;
    float* result = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)vocab * sizeof(float));
    return result;
}

extern "C" float* higgs_stt_embed_tokens(higgs_stt_context* ctx, const int32_t* input_ids, int n_tokens) {
    if (!ctx || !input_ids || n_tokens <= 0)
        return nullptr;
    const int d = (int)ctx->model.hparams.llm_d_model;

    // Fast path: single-token lookup avoids graph build + sched overhead.
    // Gated by CRISPASR_HIGGS_STT_EMBED_FAST (default ON).
    static int use_fast = -1;
    if (use_fast < 0) {
        const char* e = std::getenv("CRISPASR_HIGGS_STT_EMBED_FAST");
        use_fast = (!e || *e != '0') ? 1 : 0;
    }
    if (n_tokens == 1 && use_fast && ctx->model.llm.token_embd_w) {
        const ggml_tensor* w = ctx->model.llm.token_embd_w;
        const size_t row_bytes = ggml_row_size(w->type, d);
        float* result = (float*)malloc((size_t)d * sizeof(float));
        if (!result)
            return nullptr;
        std::vector<uint8_t> raw(row_bytes);
        ggml_backend_tensor_get(w, raw.data(), (size_t)input_ids[0] * row_bytes, row_bytes);
        if (w->type == GGML_TYPE_F32) {
            std::memcpy(result, raw.data(), (size_t)d * sizeof(float));
        } else {
            ggml_get_type_traits(w->type)->to_float(raw.data(), result, d);
        }
        return result;
    }

    ggml_cgraph* gf = higgs_stt_build_graph_embed(ctx, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "higgs_stt: failed to alloc embed graph\n");
        return nullptr;
    }
    ggml_tensor* ids_in = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(ids_in, input_ids, 0, (size_t)n_tokens * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "higgs_stt: embed graph compute failed\n");
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "embeds");
    const size_t total = (size_t)d * n_tokens;
    float* result = (float*)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

extern "C" float* higgs_stt_run_llm_from_embeds(higgs_stt_context* ctx, const float* inputs_embeds, int n_tokens,
                                                int* out_n_tokens, int* out_vocab_size) {
    if (!ctx || !inputs_embeds || n_tokens <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    const int vocab = (int)(hp.llm_lm_head_dim ? hp.llm_lm_head_dim : hp.llm_vocab_size);

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = i;
    std::vector<float> mask((size_t)n_tokens * n_tokens, 0.0f);
    for (int i = 0; i < n_tokens; i++)
        for (int j = i + 1; j < n_tokens; j++)
            mask[(size_t)i * n_tokens + j] = -INFINITY;

    ggml_cgraph* gf = higgs_stt_build_graph_llm_from_embeds(ctx, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "higgs_stt: failed to alloc llm-from-embeds graph\n");
        return nullptr;
    }

    ggml_tensor* embeds_in = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(embeds_in, inputs_embeds, 0, (size_t)d * n_tokens * sizeof(float));
    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(pos_in, positions.data(), 0, positions.size() * sizeof(int32_t));
    ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
    ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "higgs_stt: llm-from-embeds graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    if (!out) {
        fprintf(stderr, "missing logits\n");
        return nullptr;
    }
    if (out_n_tokens)
        *out_n_tokens = n_tokens;
    if (out_vocab_size)
        *out_vocab_size = vocab;
    const size_t total = (size_t)vocab * n_tokens;
    float* result = (float*)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

extern "C" float* higgs_stt_run_llm(higgs_stt_context* ctx, const int32_t* input_ids, int n_tokens, int* out_n_tokens,
                                    int* out_vocab_size) {
    if (!ctx || !input_ids || n_tokens <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int vocab = (int)(hp.llm_lm_head_dim ? hp.llm_lm_head_dim : hp.llm_vocab_size);

    // Build positions = [0, 1, ..., T-1]
    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = i;

    // Build causal mask: (T, T) F32. mask[i, j] = 0 if j <= i else -inf.
    // ggml ne[0]=j (key, fast), ne[1]=i (query). Disallowed → -inf.
    std::vector<float> mask((size_t)n_tokens * n_tokens, 0.0f);
    for (int i = 0; i < n_tokens; i++) {
        for (int j = 0; j < n_tokens; j++) {
            if (j > i)
                mask[(size_t)i * n_tokens + j] = -INFINITY;
        }
    }

    ggml_cgraph* gf = higgs_stt_build_graph_llm(ctx, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "higgs_stt: failed to alloc llm graph\n");
        return nullptr;
    }

    ggml_tensor* ids_in = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(ids_in, input_ids, 0, (size_t)n_tokens * sizeof(int32_t));

    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(pos_in, positions.data(), 0, positions.size() * sizeof(int32_t));

    ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
    ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "higgs_stt: llm graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    if (!out) {
        fprintf(stderr, "higgs_stt: missing logits tensor\n");
        return nullptr;
    }
    if (out_n_tokens)
        *out_n_tokens = n_tokens;
    if (out_vocab_size)
        *out_vocab_size = vocab;

    const size_t total = (size_t)vocab * n_tokens;
    float* result = (float*)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

extern "C" int higgs_stt_lm_head_dim(struct higgs_stt_context* ctx) {
    if (!ctx)
        return 0;
    const auto& hp = ctx->model.hparams;
    return (int)(hp.llm_lm_head_dim ? hp.llm_lm_head_dim : hp.llm_vocab_size);
}

// High-level forced-alignment dispatch.
//
// Mirrors qwen_asr/inference/qwen3_forced_aligner.py::Qwen3ForcedAligner.align
// minus the language-specific tokenizers (we whitespace-split words and let
// each word go through the standard byte-level BPE encoder; CJK char-level
// tokenization is a follow-up).
extern "C" int higgs_stt_align_words(struct higgs_stt_context* ctx, const float* samples, int n_samples,
                                     const char** words, int n_words, int64_t* out_start_ms, int64_t* out_end_ms) {
    if (!ctx || !samples || n_samples <= 0 || !words || n_words <= 0 || !out_start_ms || !out_end_ms)
        return -1;

    constexpr int TIMESTAMP_TOKEN_ID = 151705;
    // Default ms-per-class. Forced-aligner config sets this to 80; the
    // value isn't carried in the GGUF metadata yet, so we hardcode it
    // for now and add a kv field on the next converter bump.
    constexpr float TIMESTAMP_SEGMENT_TIME_MS = 80.0f;

    // 1+2. Chunked mel + audio encoder/projector (see higgs_stt_encode_audio).
    int N_enc = 0, pdim = 0;
    float* audio_embeds;
    {
        higgs_stt_bench_stage _b("encoder");
        audio_embeds = higgs_stt_encode_audio(ctx, samples, n_samples, &N_enc, &pdim);
    }
    if (!audio_embeds) {
        fprintf(stderr, "higgs_stt[align]: encoder failed\n");
        return -3;
    }

    // 3. Build the prompt token-id sequence:
    //      <|audio_start|>  <|audio_pad|>×N_enc  <|audio_end|>
    //      word_1 <timestamp> <timestamp>
    //      word_2 <timestamp> <timestamp>
    //      ...
    //      word_M <timestamp> <timestamp>
    const auto& hp = ctx->model.hparams;
    const int audio_start_id = (int)hp.audio_start_token_id;
    const int audio_end_id = (int)hp.audio_end_token_id;
    const int audio_pad_id = (int)hp.audio_pad_token_id;

    std::vector<int32_t> ids;
    ids.reserve((size_t)(N_enc + n_words * 6 + 4));
    ids.push_back(audio_start_id);
    for (int i = 0; i < N_enc; i++)
        ids.push_back(audio_pad_id);
    ids.push_back(audio_end_id);

    // Tokenize each word separately and append two timestamp markers
    // after it. Whitespace-split English / Latin scripts work fine
    // through the standard BPE encoder; CJK languages need char-level
    // pre-tokenization which is tracked as a follow-up. The leading
    // space convention matches GPT-2 BPE: each non-first word starts
    // with a space so the tokenizer recognises it as a word boundary.
    for (int w = 0; w < n_words; w++) {
        const std::string word = (w == 0) ? std::string(words[w]) : std::string(" ") + words[w];
        int n = 0;
        int32_t* arr = higgs_stt_tokenize(ctx, word.c_str(), &n);
        if (arr && n > 0) {
            for (int i = 0; i < n; i++)
                ids.push_back(arr[i]);
        }
        free(arr);
        ids.push_back(TIMESTAMP_TOKEN_ID);
        ids.push_back(TIMESTAMP_TOKEN_ID);
    }

    const int T_prompt = (int)ids.size();

    // 4. Embed and splice audio
    float* text_embeds = higgs_stt_embed_tokens(ctx, ids.data(), T_prompt);
    if (!text_embeds) {
        free(audio_embeds);
        fprintf(stderr, "higgs_stt[align]: embed failed\n");
        return -4;
    }
    int spliced = 0;
    for (int i = 0; i < T_prompt && spliced < N_enc; i++) {
        if (ids[i] == audio_pad_id) {
            std::memcpy(text_embeds + (size_t)i * pdim, audio_embeds + (size_t)spliced * pdim, pdim * sizeof(float));
            spliced++;
        }
    }
    free(audio_embeds);

    // 5. KV cache + aligner forward
    {
        higgs_stt_bench_stage _b("kv_init");
        if (!higgs_stt_kv_init(ctx, /*max_ctx*/ std::max(4096, T_prompt + 16))) {
            free(text_embeds);
            fprintf(stderr, "higgs_stt[align]: kv_init failed\n");
            return -5;
        }
    }
    int n_t_out = 0, H = 0;
    float* logits;
    {
        higgs_stt_bench_stage _b("aligner_forward");
        logits = higgs_stt_run_aligner(ctx, text_embeds, T_prompt, &n_t_out, &H);
    }
    free(text_embeds);
    if (!logits) {
        fprintf(stderr, "higgs_stt[align]: aligner forward failed\n");
        return -6;
    }

    // 6. argmax over the lm_head_dim classes at each <timestamp>
    // position. The graph stores logits as ne[0]=H, ne[1]=T_prompt
    // row-major, i.e. logits[t * H + k] = score(k, t).
    std::vector<int> ts_classes;
    ts_classes.reserve((size_t)(n_words * 2));
    for (int t = 0; t < T_prompt; t++) {
        if (ids[t] != TIMESTAMP_TOKEN_ID)
            continue;
        const float* row = logits + (size_t)t * H;
        int best = 0;
        float mx = row[0];
        for (int k = 1; k < H; k++) {
            if (row[k] > mx) {
                mx = row[k];
                best = k;
            }
        }
        ts_classes.push_back(best);
    }
    free(logits);

    if ((int)ts_classes.size() != 2 * n_words) {
        fprintf(stderr,
                "higgs_stt[align]: timestamp marker count mismatch — "
                "got %zu placeholders, expected %d\n",
                ts_classes.size(), 2 * n_words);
        return -7;
    }

    // 6b. Fix timestamp monotonicity via LIS (longest increasing subsequence).
    // The reference Qwen3-ForcedAligner uses LIS to find the longest monotone
    // chain, then interpolates outliers. This is more robust than a simple
    // forward clamp for cases where large inversions occur mid-sequence.
    {
        const int M = (int)ts_classes.size();
        // O(n log n) LIS — find indices of the longest non-decreasing subsequence
        std::vector<int> dp; // dp[i] = smallest tail value for IS of length i+1
        std::vector<int> parent(M, -1);
        std::vector<int> idx_map; // which index produced each dp entry

        for (int i = 0; i < M; i++) {
            int val = ts_classes[i];
            // Binary search for first dp entry > val (upper_bound for non-decreasing)
            auto it = std::upper_bound(dp.begin(), dp.end(), val);
            int pos = (int)(it - dp.begin());
            if (pos == (int)dp.size()) {
                dp.push_back(val);
                idx_map.push_back(i);
            } else {
                dp[pos] = val;
                idx_map[pos] = i;
            }
            parent[i] = (pos > 0) ? idx_map[pos - 1] : -1;
        }

        // Traceback: find which indices are in the LIS
        std::vector<bool> in_lis(M, false);
        int k = idx_map.back();
        while (k >= 0) {
            in_lis[k] = true;
            k = parent[k];
        }

        // Interpolate outliers: for each non-LIS element, set it to the
        // value of the nearest LIS neighbor (linear interpolation between
        // the previous and next LIS values).
        int prev_lis = -1;
        int prev_val = 0;
        for (int i = 0; i < M; i++) {
            if (in_lis[i]) {
                // Fill any gap between prev_lis and i
                if (prev_lis >= 0) {
                    for (int j = prev_lis + 1; j < i; j++) {
                        // Linear interpolation
                        float frac = (float)(j - prev_lis) / (float)(i - prev_lis);
                        ts_classes[j] = prev_val + (int)(frac * (float)(ts_classes[i] - prev_val));
                    }
                }
                prev_lis = i;
                prev_val = ts_classes[i];
            }
        }
        // Fill trailing non-LIS elements
        for (int j = prev_lis + 1; j < M; j++)
            ts_classes[j] = prev_val;
    }

    // Ensure each word's end >= start.
    for (int w = 0; w < n_words; w++) {
        if (ts_classes[2 * w + 1] < ts_classes[2 * w])
            ts_classes[2 * w + 1] = ts_classes[2 * w];
    }

    // 7. Convert classes → ms and write into the caller's parallel arrays.
    for (int w = 0; w < n_words; w++) {
        out_start_ms[w] = (int64_t)((float)ts_classes[2 * w + 0] * TIMESTAMP_SEGMENT_TIME_MS);
        out_end_ms[w] = (int64_t)((float)ts_classes[2 * w + 1] * TIMESTAMP_SEGMENT_TIME_MS);
    }
    return 0;
}

// One full-T forward pass for the Qwen3-ForcedAligner. Same KV-cached
// graph the ASR backend uses for prefill, but with last_token_only=false
// so the lm_head sees every position. The KV cache is reset at the start
// because forced alignment is one-shot — no autoregressive decode loop.
extern "C" float* higgs_stt_run_aligner(struct higgs_stt_context* ctx, const float* inputs_embeds, int n_tokens,
                                        int* out_n_tokens, int* out_lm_head_dim) {
    if (!ctx || !inputs_embeds || n_tokens <= 0)
        return nullptr;
    if (!ctx->kv_k) {
        fprintf(stderr, "higgs_stt: kv cache not initialized — call higgs_stt_kv_init first\n");
        return nullptr;
    }
    if (n_tokens > ctx->kv_max_ctx) {
        fprintf(stderr, "higgs_stt: aligner needs %d tokens but kv max_ctx is %d\n", n_tokens, ctx->kv_max_ctx);
        return nullptr;
    }
    higgs_stt_kv_reset(ctx);

    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    const int H = (int)(hp.llm_lm_head_dim ? hp.llm_lm_head_dim : hp.llm_vocab_size);
    const int Lk = n_tokens;

    // Positions 0..n_tokens-1.
    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = i;

    // Causal mask for the prefill path. Same encoding the run_llm_kv
    // path uses (F16, ne[0]=Lk for fast key axis, ne[1]=n_tokens for
    // queries; -inf above the diagonal).
    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
        mask.assign((size_t)Lk * n_tokens, zero_h);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = q + 1; k < Lk; k++) {
                mask[(size_t)q * Lk + k] = neginf_h;
            }
        }
    }

    ggml_cgraph* gf = higgs_stt_build_graph_llm_kv(ctx, /*n_past=*/0, n_tokens, /*last_token_only=*/false);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "higgs_stt: failed to alloc aligner graph\n");
        return nullptr;
    }

    ggml_tensor* embeds_in = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(embeds_in, inputs_embeds, 0, (size_t)d * n_tokens * sizeof(float));
    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(pos_in, positions.data(), 0, positions.size() * sizeof(int32_t));
    if (n_tokens > 1) {
        ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "higgs_stt: aligner graph compute failed\n");
        return nullptr;
    }

    ctx->kv_n_used = n_tokens;

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    if (!out)
        return nullptr;
    if (out_n_tokens)
        *out_n_tokens = n_tokens;
    if (out_lm_head_dim)
        *out_lm_head_dim = H;

    const size_t total = (size_t)H * n_tokens;
    float* result = (float*)malloc(total * sizeof(float));
    if (!result)
        return nullptr;
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}
