// moss_transcribe_diarize.cpp — MOSS-Transcribe-Diarize-0.9B ggml runtime
//
// Stock Whisper encoder (Conv1d stem + 24L global attention + ln_post) →
// 4x temporal merge → VQAdaptor (Linear+SiLU+Linear+LayerNorm) →
// time-marker injection into audio_pad sequence → Qwen3-0.6B LM decode →
// output parser for [timestamp][Sxx]text[timestamp] segments.
//
// See moss_transcribe_diarize.h for the architecture summary.

#include "moss_transcribe_diarize.h"
#include "core/moss_diarize_segments.h"
#include "core/win_compat.h"

#include "core/beam_decode.h"
#include "core/logit_sampler.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "crispasr_imatrix.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/gguf_loader.h"
#include "core/ffn.h"
#include "core/attention.h"
#include "core/mel.h"
#include "core/bpe.h"
#include "core/gpu_backend_pref.h"
#include "core/ngram_loop_fix.h"
#include "core/crispasr_env.h"
#include "core/ggml_cpu_backend.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Bench instrumentation — `MOSS_DIARIZE_BENCH=1`
// ===========================================================================

static bool moss_diarize_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_MOSS_DIARIZE_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct moss_diarize_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit moss_diarize_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~moss_diarize_bench_stage() {
        if (!moss_diarize_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  moss_diarize_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ===========================================================================
// Hyperparameters
// ===========================================================================

struct moss_diarize_hparams {
    // Mel front-end (Whisper-style: 80 bins, n_fft 400, hop 160)
    uint32_t n_mels = 80;
    uint32_t n_fft = 400;
    uint32_t hop_length = 160;
    uint32_t sample_rate = 16000;

    // Whisper encoder (Conv1d stem + 24L transformer)
    uint32_t enc_layers = 24;
    uint32_t enc_d_model = 1024;
    uint32_t enc_n_heads = 16;
    uint32_t enc_head_dim = 64; // 1024 / 16
    uint32_t enc_ffn_dim = 4096;
    uint32_t enc_max_pos = 1500;
    float enc_ln_eps = 1e-5f;

    // VQAdaptor
    uint32_t audio_merge_size = 4;
    uint32_t adaptor_in_dim = 4096;  // enc_d_model * merge_size
    uint32_t adaptor_out_dim = 1024; // = llm_dim

    // LLM (Qwen3-0.6B)
    uint32_t llm_dim = 1024;
    uint32_t llm_layers = 28;
    uint32_t llm_n_heads = 16;
    uint32_t llm_n_kv_heads = 8;
    uint32_t llm_head_dim = 128;
    uint32_t llm_ff_dim = 3072;
    uint32_t llm_vocab_size = 151936;
    float llm_rope_theta = 1000000.0f;
    float llm_rms_eps = 1e-6f;

    // Special tokens
    uint32_t im_start_id = 151644;    // <|im_start|>
    uint32_t im_end_id = 151645;      // <|im_end|>
    uint32_t audio_start_id = 151669; // <|audio_start|>
    uint32_t audio_end_id = 151670;   // <|audio_end|>
    uint32_t audio_pad_id = 151671;   // <|audio_pad|>

    // Diarize/timestamp params
    uint32_t time_marker_every_seconds = 5; // inject digit markers every N seconds (processor_config.json)
    float audio_tokens_per_second = 12.5f;  // after 4x merge
};

// ===========================================================================
// Tensor containers
// ===========================================================================

struct moss_diarize_enc_block {
    ggml_tensor *attn_norm_w = nullptr, *attn_norm_b = nullptr;
    ggml_tensor *attn_q_w = nullptr, *attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr; // k has NO bias in this Whisper variant
    ggml_tensor *attn_v_w = nullptr, *attn_v_b = nullptr;
    ggml_tensor *attn_out_w = nullptr, *attn_out_b = nullptr;
    ggml_tensor *ffn_norm_w = nullptr, *ffn_norm_b = nullptr;
    ggml_tensor *ffn_up_w = nullptr, *ffn_up_b = nullptr;
    ggml_tensor *ffn_down_w = nullptr, *ffn_down_b = nullptr;
};

struct moss_diarize_encoder {
    // Conv1d stem: conv1(80→1024,k=3,s=1,p=1) + conv2(1024→1024,k=3,s=2,p=1)
    ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr;
    ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr;
    // Learned position embeddings (1500, 1024)
    ggml_tensor* pos_embed_w = nullptr;
    // Post-layer LayerNorm
    ggml_tensor *ln_post_w = nullptr, *ln_post_b = nullptr;
    // Transformer layers
    std::vector<moss_diarize_enc_block> blocks;
};

struct moss_diarize_adaptor {
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;   // Linear(4096, 1024)
    ggml_tensor *fc2_w = nullptr, *fc2_b = nullptr;   // Linear(1024, 1024)
    ggml_tensor *norm_w = nullptr, *norm_b = nullptr; // LayerNorm(1024)
};

struct moss_diarize_llm_block {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* q_norm_w = nullptr;
    ggml_tensor* k_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct moss_diarize_llm {
    ggml_tensor* embed_w = nullptr;
    std::vector<moss_diarize_llm_block> blocks;
    ggml_tensor* final_norm_w = nullptr;
    ggml_tensor* lm_head_w = nullptr; // tied → == embed_w
};

struct moss_diarize_model {
    moss_diarize_hparams hparams;
    moss_diarize_encoder enc;
    moss_diarize_adaptor adaptor;
    moss_diarize_llm llm;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

struct moss_diarize_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
};

struct moss_diarize_context {
    moss_diarize_params params;
    moss_diarize_model model;
    moss_diarize_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> compute_meta;

    // KV cache for LLM
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;
    int kv_n_used = 0;

    int n_threads = 4;
    std::string model_path;
    int beam_size = 1;
    // #564 sampling for the beam_size == 1 path; temperature <= 0 = greedy.
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 1.0f;
    uint64_t seed = 0;
    // Decode cap. Diarize output is long (speaker tags + timestamps + text), so
    // the default is 1024 rather than the global 512 — a single-pass 300 s file
    // otherwise truncated (#292). The CLI/session forwards --max-new-tokens here
    // when the user set it explicitly; unset keeps this 1024 default.
    int max_new_tokens = 1024;
    std::string hotwords;
    std::string ask_override; // custom system instruction (set_ask)
    std::string language;     // language hint
    bool enc_use_flash = true;
};

// ===========================================================================
// Helpers
// ===========================================================================

static ggml_tensor* try_get(moss_diarize_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}
static ggml_tensor* require(moss_diarize_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "moss_diarize");
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool moss_diarize_load_model(moss_diarize_model& model, moss_diarize_vocab& vocab, const char* path,
                                    ggml_backend_t backend) {
    // ---- pass 1: metadata + vocab ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;

        auto& hp = model.hparams;
        hp.enc_layers = core_gguf::kv_u32(gctx, "moss_diarize.enc.n_layers", hp.enc_layers);
        hp.enc_d_model = core_gguf::kv_u32(gctx, "moss_diarize.enc.d_model", hp.enc_d_model);
        hp.enc_n_heads = core_gguf::kv_u32(gctx, "moss_diarize.enc.n_heads", hp.enc_n_heads);
        hp.n_mels = core_gguf::kv_u32(gctx, "moss_diarize.enc.n_mels", hp.n_mels);
        hp.enc_ffn_dim = core_gguf::kv_u32(gctx, "moss_diarize.enc.ffn_dim", hp.enc_ffn_dim);
        hp.enc_max_pos = core_gguf::kv_u32(gctx, "moss_diarize.enc.max_pos", hp.enc_max_pos);
        hp.enc_head_dim = hp.enc_d_model / hp.enc_n_heads;

        hp.adaptor_in_dim = core_gguf::kv_u32(gctx, "moss_diarize.adaptor.in_dim", hp.adaptor_in_dim);
        hp.adaptor_out_dim = core_gguf::kv_u32(gctx, "moss_diarize.adaptor.out_dim", hp.adaptor_out_dim);

        hp.llm_layers = core_gguf::kv_u32(gctx, "moss_diarize.llm.n_layers", hp.llm_layers);
        hp.llm_dim = core_gguf::kv_u32(gctx, "moss_diarize.llm.dim", hp.llm_dim);
        hp.llm_n_heads = core_gguf::kv_u32(gctx, "moss_diarize.llm.n_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = core_gguf::kv_u32(gctx, "moss_diarize.llm.n_kv_heads", hp.llm_n_kv_heads);
        hp.llm_head_dim = core_gguf::kv_u32(gctx, "moss_diarize.llm.head_dim", hp.llm_head_dim);
        hp.llm_ff_dim = core_gguf::kv_u32(gctx, "moss_diarize.llm.ff_dim", hp.llm_ff_dim);
        hp.llm_vocab_size = core_gguf::kv_u32(gctx, "moss_diarize.llm.vocab_size", hp.llm_vocab_size);
        hp.llm_rope_theta = core_gguf::kv_f32(gctx, "moss_diarize.llm.rope_theta", hp.llm_rope_theta);
        hp.llm_rms_eps = core_gguf::kv_f32(gctx, "moss_diarize.llm.rms_eps", hp.llm_rms_eps);

        hp.audio_pad_id = core_gguf::kv_u32(gctx, "moss_diarize.audio_token_id", hp.audio_pad_id);
        hp.time_marker_every_seconds =
            core_gguf::kv_u32(gctx, "moss_diarize.time_marker_every_seconds", hp.time_marker_every_seconds);
        hp.audio_tokens_per_second =
            core_gguf::kv_f32(gctx, "moss_diarize.audio_tokens_per_second", hp.audio_tokens_per_second);
        hp.audio_merge_size = core_gguf::kv_u32(gctx, "moss_diarize.audio_merge_size", hp.audio_merge_size);

        auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            vocab.id_to_token = std::move(tokens);
            vocab.token_to_id.reserve(vocab.id_to_token.size());
            for (int i = 0; i < (int)vocab.id_to_token.size(); i++)
                vocab.token_to_id[vocab.id_to_token[i]] = i;
        }

        struct SpecialTok {
            int id;
            const char* text;
        };
        static const SpecialTok specials[] = {
            {151643, "<|endoftext|>"},   {151644, "<|im_start|>"},  {151645, "<|im_end|>"},
            {151669, "<|audio_start|>"}, {151670, "<|audio_end|>"}, {151671, "<|audio_pad|>"},
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

        auto merges = core_gguf::kv_str_array(gctx, "tokenizer.ggml.merges");
        for (int i = 0; i < (int)merges.size(); i++)
            vocab.merge_rank[merges[i]] = i;

        core_gguf::free_metadata(gctx);
    }

    // ---- pass 2: tensor data ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "moss_diarize", wl))
        return false;
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.tensors = std::move(wl.tensors);

    // ---- bind encoder tensors ----
    auto& enc = model.enc;
    enc.conv1_w = require(model, "enc.conv1.weight");
    enc.conv1_b = require(model, "enc.conv1.bias");
    enc.conv2_w = require(model, "enc.conv2.weight");
    enc.conv2_b = require(model, "enc.conv2.bias");
    enc.pos_embed_w = require(model, "enc.pos_embed.weight");
    enc.ln_post_w = require(model, "enc.ln_post.weight");
    enc.ln_post_b = require(model, "enc.ln_post.bias");

    enc.blocks.resize(model.hparams.enc_layers);
    for (uint32_t i = 0; i < model.hparams.enc_layers; i++) {
        char buf[128];
        auto& b = enc.blocks[i];
        auto get = [&](const char* suf) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "enc.blk.%u.%s", i, suf);
            return require(model, buf);
        };
        b.attn_norm_w = get("attn_norm.weight");
        b.attn_norm_b = get("attn_norm.bias");
        b.attn_q_w = get("attn_q.weight");
        b.attn_q_b = get("attn_q.bias");
        b.attn_k_w = get("attn_k.weight");
        // K has no bias in this Whisper variant
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

    // ---- bind adaptor tensors ----
    model.adaptor.fc1_w = require(model, "adaptor.fc1.weight");
    model.adaptor.fc1_b = require(model, "adaptor.fc1.bias");
    model.adaptor.fc2_w = require(model, "adaptor.fc2.weight");
    model.adaptor.fc2_b = require(model, "adaptor.fc2.bias");
    model.adaptor.norm_w = require(model, "adaptor.norm.weight");
    model.adaptor.norm_b = require(model, "adaptor.norm.bias");

    // ---- bind LLM tensors ----
    auto& llm = model.llm;
    llm.embed_w = require(model, "token_embd.weight");
    llm.final_norm_w = require(model, "output_norm.weight");
    // Tied lm_head
    llm.lm_head_w = try_get(model, "output.weight");
    if (!llm.lm_head_w)
        llm.lm_head_w = llm.embed_w;

    llm.blocks.resize(model.hparams.llm_layers);
    for (uint32_t i = 0; i < model.hparams.llm_layers; i++) {
        char buf[128];
        auto& b = llm.blocks[i];
        auto get = [&](const char* suf) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "blk.%u.%s", i, suf);
            return require(model, buf);
        };
        b.attn_norm_w = get("attn_norm.weight");
        b.attn_q_w = get("attn_q.weight");
        b.attn_k_w = get("attn_k.weight");
        b.attn_v_w = get("attn_v.weight");
        b.attn_o_w = get("attn_o.weight");
        b.q_norm_w = get("q_norm.weight");
        b.k_norm_w = get("k_norm.weight");
        b.ffn_norm_w = get("ffn_norm.weight");
        b.ffn_gate_w = get("ffn_gate.weight");
        b.ffn_up_w = get("ffn_up.weight");
        b.ffn_down_w = get("ffn_down.weight");
    }

    const auto& hp = model.hparams;
    fprintf(stderr,
            "moss_diarize: loaded %u enc layers (d=%u, mels=%u), adaptor (%u→%u, merge=%u), "
            "%u LLM layers (d=%u), vocab=%u\n",
            hp.enc_layers, hp.enc_d_model, hp.n_mels, hp.adaptor_in_dim, hp.adaptor_out_dim, hp.audio_merge_size,
            hp.llm_layers, hp.llm_dim, hp.llm_vocab_size);
    return true;
}

// ===========================================================================
// Mel spectrogram (Whisper-style, 80-bin)
// ===========================================================================

static void moss_diarize_dft(const float* in, int N, float* out) {
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

static void moss_diarize_fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0.0f;
        return;
    }
    int half_N = N / 2;
    if (N - half_N * 2 == 1) {
        moss_diarize_dft(in, N, out);
        return;
    }
    std::vector<float> even_in(half_N), odd_in(half_N);
    for (int i = 0; i < half_N; i++) {
        even_in[i] = in[2 * i];
        odd_in[i] = in[2 * i + 1];
    }
    std::vector<float> even_out(2 * half_N), odd_out(2 * half_N);
    moss_diarize_fft(even_in.data(), half_N, even_out.data());
    moss_diarize_fft(odd_in.data(), half_N, odd_out.data());
    for (int k = 0; k < half_N; k++) {
        float ang = -2.0f * (float)M_PI * (float)k / (float)N;
        float cos_a = std::cos(ang), sin_a = std::sin(ang);
        float tre = odd_out[2 * k] * cos_a - odd_out[2 * k + 1] * sin_a;
        float tim = odd_out[2 * k] * sin_a + odd_out[2 * k + 1] * cos_a;
        out[2 * k] = even_out[2 * k] + tre;
        out[2 * k + 1] = even_out[2 * k + 1] + tim;
        out[2 * (k + half_N)] = even_out[2 * k] - tre;
        out[2 * (k + half_N) + 1] = even_out[2 * k + 1] - tim;
    }
}

extern "C" float* moss_diarize_compute_mel(struct moss_diarize_context* ctx, const float* samples, int n_samples,
                                           int* out_n_mels, int* out_T_mel) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int n_fft = (int)hp.n_fft;
    const int hop = (int)hp.hop_length;
    const int n_mels_val = (int)hp.n_mels;
    const int n_freqs = n_fft / 2 + 1;

    std::vector<float> hann(n_fft);
    for (int i = 0; i < n_fft; i++)
        hann[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i / (float)n_fft));

    std::vector<float> mel_filters =
        core_mel::build_slaney_fb((int)hp.sample_rate, n_fft, n_mels_val, 0.0f, -1.0f, core_mel::FbLayout::MelsFreqs);

    core_mel::FftR2C fft_fn = [](const float* in, int N, float* out) {
        moss_diarize_fft(const_cast<float*>(in), N, out);
    };
    core_mel::Params mel_params;
    mel_params.n_fft = n_fft;
    mel_params.hop_length = hop;
    mel_params.win_length = n_fft;
    mel_params.n_mels = n_mels_val;
    mel_params.log_base = core_mel::LogBase::Log10;
    mel_params.spec_kind = core_mel::SpecKind::Power;
    mel_params.norm = core_mel::Normalization::GlobalClipMax;
    mel_params.layout = core_mel::Layout::MelsTime;
    mel_params.log_guard = core_mel::LogGuard::MaxClip;
    mel_params.log_eps = 1e-10f;
    mel_params.center_pad = true;
    mel_params.center_pad_reflect = true;

    int T_mel_actual = 0;
    std::vector<float> mel_out = core_mel::compute(samples, n_samples, hann.data(), n_fft, mel_filters.data(), n_freqs,
                                                   fft_fn, mel_params, T_mel_actual);

    // Whisper truncation: exactly n_samples/hop frames.
    const int T_target = n_samples / hop;
    if (T_mel_actual > T_target && T_target > 0) {
        std::vector<float> trunc((size_t)n_mels_val * T_target);
        for (int f = 0; f < n_mels_val; f++)
            memcpy(trunc.data() + (size_t)f * T_target, mel_out.data() + (size_t)f * T_mel_actual,
                   (size_t)T_target * sizeof(float));
        mel_out = std::move(trunc);
        T_mel_actual = T_target;
    }

    float* result = (float*)malloc(mel_out.size() * sizeof(float));
    memcpy(result, mel_out.data(), mel_out.size() * sizeof(float));
    if (out_n_mels)
        *out_n_mels = n_mels_val;
    if (out_T_mel)
        *out_T_mel = T_mel_actual;
    return result;
}

// ===========================================================================
// Whisper encoder — Conv1d stem + 24L transformer + ln_post
// ===========================================================================

// Conv1d output length: L_out = floor((L_in + 2*pad - kernel) / stride) + 1
// conv1: k=3, s=1, p=1 → same length
// conv2: k=3, s=2, p=1 → floor((L-1)/2) + 1

static ggml_cgraph* moss_diarize_build_encoder_graph(moss_diarize_context* ctx, int T_mel) {
    const auto& hp = ctx->model.hparams;
    const auto& enc = ctx->model.enc;
    const int d = (int)hp.enc_d_model;
    const int n_heads = (int)hp.enc_n_heads;
    const int head_dim = (int)hp.enc_head_dim;
    const int n_mels = (int)hp.n_mels;

    // T after conv2 stride-2: floor((T_mel - 1)/2) + 1
    const int T_enc = (T_mel - 1) / 2 + 1;

    struct ggml_init_params gparams = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(gparams);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // mel input: (T_mel, n_mels) — ggml ne[0]=T fastest, ne[1]=n_mels
    // Matches higgs_stt / original Whisper convention for ggml_conv_1d.
    ggml_tensor* mel_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_mel, n_mels);
    ggml_set_name(mel_in, "mel_input");
    ggml_set_input(mel_in);

    // Conv1d bias broadcast helper: reshape (OC,) → (1, OC, 1) for 3D conv output
    auto bias_1d = [&](ggml_tensor* b) { return ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1); };

    // CRISPASR_MOSS_DIARIZE_DUMP_CONV=1 enables per-stage conv snapshots
    const bool dump_conv = [] {
        const char* e = std::getenv("CRISPASR_MOSS_DIARIZE_DUMP_CONV");
        return e && e[0] == '1';
    }();

    if (dump_conv) {
        ggml_tensor* mel_snap = ggml_cont(ctx0, mel_in);
        ggml_set_name(mel_snap, "mel_in_graph");
        ggml_set_output(mel_snap);
        ggml_build_forward_expand(gf, mel_snap);
    }

    // conv1: k=3, s=1, p=1 → same length
    ggml_tensor* x = ggml_conv_1d(ctx0, enc.conv1_w, mel_in, 1, 1, 1);

    if (dump_conv) {
        ggml_tensor* snap = ggml_cont(ctx0, x);
        ggml_set_name(snap, "conv1_raw");
        ggml_set_output(snap);
        ggml_build_forward_expand(gf, snap);
    }

    x = ggml_add(ctx0, x, bias_1d(enc.conv1_b));
    x = ggml_gelu_erf(ctx0, x);

    if (dump_conv) {
        ggml_tensor* snap = ggml_cont(ctx0, x);
        ggml_set_name(snap, "conv1_out");
        ggml_set_output(snap);
        ggml_build_forward_expand(gf, snap);
    }

    // conv2: k=3, s=2, p=1 → T_enc = T_mel/2
    x = ggml_conv_1d(ctx0, enc.conv2_w, x, 2, 1, 1);
    x = ggml_add(ctx0, x, bias_1d(enc.conv2_b));
    x = ggml_gelu_erf(ctx0, x);

    // Reshape conv output from (OL, OC, 1) to (T_enc, d), then transpose to (d, T_enc)
    x = ggml_reshape_2d(ctx0, x, T_enc, d);
    x = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // (d, T_enc)

    // Mark conv stem output for diff harness extraction
    ggml_tensor* conv_out_snap = ggml_cont(ctx0, x);
    ggml_set_name(conv_out_snap, "conv_stem_out");
    ggml_set_output(conv_out_snap);
    ggml_build_forward_expand(gf, conv_out_snap);

    // Add learned position embeddings. pos_embed is ggml ne (d, 1500), may be F16.
    ggml_tensor* pos = ggml_view_2d(ctx0, enc.pos_embed_w, d, T_enc, enc.pos_embed_w->nb[1], 0);
    if (pos->type != GGML_TYPE_F32)
        pos = ggml_cast(ctx0, pos, GGML_TYPE_F32);
    x = ggml_add(ctx0, x, pos);

    const float attn_scale = 1.0f / std::sqrt((float)head_dim);

    // 24 transformer layers with global attention (no windowing)
    for (uint32_t il = 0; il < hp.enc_layers; il++) {
        const auto& blk = enc.blocks[il];
        ggml_tensor* residual = x;

        // Pre-LN self-attention
        ggml_tensor* h = ggml_norm(ctx0, x, hp.enc_ln_eps);
        h = ggml_add(ctx0, ggml_mul(ctx0, h, blk.attn_norm_w), blk.attn_norm_b);

        ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.attn_q_w, h), blk.attn_q_b);
        ggml_tensor* K = ggml_mul_mat(ctx0, blk.attn_k_w, h); // no bias
        ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.attn_v_w, h), blk.attn_v_b);

        Q = ggml_reshape_3d(ctx0, Q, head_dim, n_heads, T_enc);
        K = ggml_reshape_3d(ctx0, K, head_dim, n_heads, T_enc);
        V = ggml_reshape_3d(ctx0, V, head_dim, n_heads, T_enc);
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3)); // (hd, T, n_h)
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        ggml_tensor* attn;
        if (ctx->enc_use_flash) {
            // Global bidirectional attention — no mask needed (nullptr = full attention)
            attn = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, attn_scale, 0.0f, 0.0f);
            attn = ggml_reshape_2d(ctx0, attn, d, T_enc);
        } else {
            ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q);
            scores = ggml_soft_max_ext(ctx0, scores, nullptr, attn_scale, 0.0f);
            ggml_tensor* V_perm = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3));
            attn = ggml_mul_mat(ctx0, V_perm, scores);
            attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
            attn = ggml_reshape_2d(ctx0, attn, d, T_enc);
        }

        ggml_tensor* attn_out = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.attn_out_w, attn), blk.attn_out_b);
        x = ggml_add(ctx0, residual, attn_out);

        // Pre-LN FFN (GELU)
        residual = x;
        h = ggml_norm(ctx0, x, hp.enc_ln_eps);
        h = ggml_add(ctx0, ggml_mul(ctx0, h, blk.ffn_norm_w), blk.ffn_norm_b);
        h = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.ffn_up_w, h), blk.ffn_up_b);
        h = ggml_gelu_erf(ctx0, h);
        h = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.ffn_down_w, h), blk.ffn_down_b);
        x = ggml_add(ctx0, residual, h);
    }

    // ln_post
    x = ggml_norm(ctx0, x, hp.enc_ln_eps);
    x = ggml_add(ctx0, ggml_mul(ctx0, x, enc.ln_post_w), enc.ln_post_b);

    ggml_set_name(x, "encoder_output");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

// Run encoder on a single 30s-padded mel chunk. Internal helper.
static float* moss_diarize_run_encoder_chunk(moss_diarize_context* ctx, const float* mel_melstime, int n_mels,
                                             int T_mel, int* out_T_enc) {
    const int d = (int)ctx->model.hparams.enc_d_model;
    const int T_enc = (T_mel - 1) / 2 + 1;

    ggml_cgraph* gf = moss_diarize_build_encoder_graph(ctx, T_mel);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "moss_diarize: encoder alloc failed\n");
        return nullptr;
    }

    // MelsTime layout mel[f*T+t] IS ggml ne=(T, IC) order: IC groups at stride T,
    // T values contiguous within each IC — the correct Conv1d input layout.
    // No transposition needed.
    ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel_input");
    ggml_backend_tensor_set(mel_in, mel_melstime, 0, (size_t)n_mels * T_mel * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "moss_diarize: encoder compute failed\n");
        return nullptr;
    }
    ggml_tensor* eo = ggml_graph_get_tensor(gf, "encoder_output");
    float* result = (float*)malloc((size_t)d * T_enc * sizeof(float));
    ggml_backend_tensor_get(eo, result, 0, (size_t)d * T_enc * sizeof(float));

    // CRISPASR_MOSS_DIARIZE_DUMP_CONV=1: dump per-stage conv intermediates to stderr
    {
        static const char* _dc = std::getenv("CRISPASR_MOSS_DIARIZE_DUMP_CONV");
        if (_dc && _dc[0] == '1') {
            auto dump_tensor = [](const char* name, ggml_cgraph* g) {
                ggml_tensor* t = ggml_graph_get_tensor(g, name);
                if (!t)
                    return;
                size_t n = ggml_nelements(t);
                std::vector<float> buf(n);
                ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
                fprintf(stderr, "moss_diarize: %s ne=[%lld,%lld,%lld] first5=[%.4f,%.4f,%.4f,%.4f,%.4f]\n", name,
                        (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], buf[0], n > 1 ? buf[1] : 0.f,
                        n > 2 ? buf[2] : 0.f, n > 3 ? buf[3] : 0.f, n > 4 ? buf[4] : 0.f);
            };
            dump_tensor("mel_in_graph", gf);
            dump_tensor("conv1_raw", gf);
            dump_tensor("conv1_out", gf);
            dump_tensor("conv_stem_out", gf);
        }
    }

    if (out_T_enc)
        *out_T_enc = T_enc;
    return result;
}

// Compute audio token count for a raw-sample chunk (matches Python _compute_audio_token_length).
static int moss_diarize_audio_token_len(int n_samples, int hop, int merge_size) {
    int stride = hop * 2 * merge_size; // 160 * 2 * 4 = 1280
    return (n_samples - 1) / stride + 1;
}

// Encode raw audio with 30s chunking (matches the Python processor exactly).
// Returns concatenated valid encoder frames from all chunks.
extern "C" float* moss_diarize_run_encoder(struct moss_diarize_context* ctx, const float* samples, int n_samples,
                                           int* out_T_total, int* out_d) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.enc_d_model;
    const int n_mels = (int)hp.n_mels;
    const int hop = (int)hp.hop_length;
    const int merge = (int)hp.audio_merge_size;
    const int sample_rate = (int)hp.sample_rate;
    const int chunk_samples = sample_rate * 30; // 480000 samples = 30s
    const int chunk_mel_frames = 3000;          // Whisper 30s = 3000 mel frames

    // Chunk the audio
    std::vector<int> chunk_starts;
    for (int s = 0; s < n_samples; s += chunk_samples)
        chunk_starts.push_back(s);
    if (chunk_starts.empty())
        chunk_starts.push_back(0);

    // Compute valid token lengths per chunk
    std::vector<int> token_lengths(chunk_starts.size());
    for (size_t c = 0; c < chunk_starts.size(); c++) {
        int chunk_len = std::min(chunk_samples, n_samples - chunk_starts[c]);
        token_lengths[c] = moss_diarize_audio_token_len(chunk_len, hop, merge);
    }

    // Process each chunk: compute mel (padded to 30s), encode, keep valid frames
    int T_total = 0;
    for (size_t c = 0; c < chunk_starts.size(); c++)
        T_total += token_lengths[c] * 4; // encoder frames = token_len * merge_size

    std::vector<float> all_features((size_t)d * T_total);
    int out_offset = 0;

    for (size_t c = 0; c < chunk_starts.size(); c++) {
        int start = chunk_starts[c];
        int chunk_len = std::min(chunk_samples, n_samples - start);

        // Pad or trim to exactly 30s for mel computation
        std::vector<float> chunk_pcm(chunk_samples, 0.0f);
        memcpy(chunk_pcm.data(), samples + start, (size_t)chunk_len * sizeof(float));

        // Compute mel for this 30s chunk
        int mel_n = 0, mel_T = 0;
        float* mel = moss_diarize_compute_mel(ctx, chunk_pcm.data(), chunk_samples, &mel_n, &mel_T);
        if (!mel)
            return nullptr;

        // Pad mel to 3000 frames if needed
        if (mel_T < chunk_mel_frames) {
            std::vector<float> padded((size_t)mel_n * chunk_mel_frames, 0.0f);
            for (int f = 0; f < mel_n; f++)
                memcpy(padded.data() + (size_t)f * chunk_mel_frames, mel + (size_t)f * mel_T,
                       (size_t)mel_T * sizeof(float));
            free(mel);
            mel = (float*)malloc(padded.size() * sizeof(float));
            memcpy(mel, padded.data(), padded.size() * sizeof(float));
            mel_T = chunk_mel_frames;
        }

        // Run encoder on this chunk
        int enc_T = 0;
        float* enc_out = moss_diarize_run_encoder_chunk(ctx, mel, mel_n, mel_T, &enc_T);
        free(mel);
        if (!enc_out)
            return nullptr;

        // Keep only valid frames (token_len * 4)
        int valid_frames = token_lengths[c] * 4;
        if (valid_frames > enc_T)
            valid_frames = enc_T;

        memcpy(all_features.data() + (size_t)out_offset * d, enc_out, (size_t)valid_frames * d * sizeof(float));
        out_offset += valid_frames;
        free(enc_out);
    }

    float* result = (float*)malloc((size_t)d * out_offset * sizeof(float));
    memcpy(result, all_features.data(), (size_t)d * out_offset * sizeof(float));

    if (out_T_total)
        *out_T_total = out_offset;
    if (out_d)
        *out_d = d;
    return result;
}

// ===========================================================================
// VQAdaptor: 4x temporal merge → Linear+SiLU+Linear+LayerNorm
// ===========================================================================

extern "C" float* moss_diarize_run_adaptor(struct moss_diarize_context* ctx, const float* encoder_out, int T_enc,
                                           int d_enc, int* out_T, int* out_d) {
    if (!ctx || !encoder_out)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int merge = (int)hp.audio_merge_size;
    const int d_out = (int)hp.adaptor_out_dim;

    // 1. 4x temporal merge: (d_enc, T_enc) → (d_enc*4, T_enc/4)
    int T_merged = T_enc / merge;
    if (T_merged < 1)
        T_merged = 1;

    // encoder_out is in ggml ne-order (d_enc, T_enc): d_enc fastest.
    // Merge: concatenate 4 consecutive d_enc-dim frames → (d_enc*4, T_merged)
    int in_dim = d_enc * merge; // 4096
    std::vector<float> merged((size_t)in_dim * T_merged, 0.0f);
    for (int t = 0; t < T_merged; t++) {
        for (int m = 0; m < merge; m++) {
            int src_t = t * merge + m;
            if (src_t >= T_enc)
                break;
            memcpy(merged.data() + (size_t)t * in_dim + (size_t)m * d_enc, encoder_out + (size_t)src_t * d_enc,
                   (size_t)d_enc * sizeof(float));
        }
    }

    // 2. Build adaptor graph: Linear(4096→1024) + SiLU + Linear(1024→1024) + LayerNorm
    struct ggml_init_params gparams = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(gparams);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, in_dim, T_merged);
    ggml_set_name(x, "adaptor_input");
    ggml_set_input(x);

    // fc1: Linear(4096, 1024)
    x = ggml_add(ctx0, ggml_mul_mat(ctx0, ctx->model.adaptor.fc1_w, x), ctx->model.adaptor.fc1_b);
    // SiLU
    x = ggml_silu(ctx0, x);
    // fc2: Linear(1024, 1024)
    x = ggml_add(ctx0, ggml_mul_mat(ctx0, ctx->model.adaptor.fc2_w, x), ctx->model.adaptor.fc2_b);
    // LayerNorm
    x = ggml_norm(ctx0, x, 1e-5f);
    x = ggml_add(ctx0, ggml_mul(ctx0, x, ctx->model.adaptor.norm_w), ctx->model.adaptor.norm_b);

    ggml_set_name(x, "adaptor_output");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "moss_diarize: adaptor alloc failed\n");
        return nullptr;
    }
    ggml_tensor* in_t = ggml_graph_get_tensor(gf, "adaptor_input");
    ggml_backend_tensor_set(in_t, merged.data(), 0, (size_t)in_dim * T_merged * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "moss_diarize: adaptor compute failed\n");
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "adaptor_output");
    float* result = (float*)malloc((size_t)d_out * T_merged * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)d_out * T_merged * sizeof(float));

    if (out_T)
        *out_T = T_merged;
    if (out_d)
        *out_d = d_out;
    return result;
}

// ===========================================================================
// LLM graph with KV cache (Qwen3-0.6B)
// ===========================================================================

static ggml_cgraph* moss_diarize_build_llm_kv_graph(moss_diarize_context* ctx, int n_tokens, int n_past,
                                                    bool last_token_only) {
    const auto& hp = ctx->model.hparams;
    const auto& llm = ctx->model.llm;
    const int d = (int)hp.llm_dim;
    const int n_heads = (int)hp.llm_n_heads;
    const int n_kv_heads = (int)hp.llm_n_kv_heads;
    const int head_dim = (int)hp.llm_head_dim;
    const int Lk = n_past + n_tokens;

    struct ggml_init_params gparams = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(gparams);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, n_tokens);
    ggml_set_name(embeds_in, "inputs_embeds");
    ggml_set_input(embeds_in);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor* causal_mask = nullptr;
    if (n_tokens > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, n_tokens);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    core_attn::KvSelfAttnParams attn_p = {};
    attn_p.n_heads = n_heads;
    attn_p.n_kv_heads = n_kv_heads;
    attn_p.head_dim = head_dim;
    attn_p.n_kv_grp = n_heads / n_kv_heads;
    attn_p.n_ctx_orig = 0;
    attn_p.rope_theta = hp.llm_rope_theta;
    attn_p.rope_beta_fast = 32.0f;
    attn_p.rope_beta_slow = 1.0f;
    attn_p.attn_scale = 1.0f / std::sqrt((float)head_dim);
    attn_p.qk_norm_eps = hp.llm_rms_eps;
    attn_p.gqa_mode = core_attn::GQA_MANUAL_CONT;
    attn_p.rope_type = GGML_ROPE_TYPE_NEOX;

    ggml_tensor* cur = embeds_in;
    for (uint32_t il = 0; il < hp.llm_layers; il++) {
        const auto& blk = llm.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* h = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
        h = ggml_mul(ctx0, h, blk.attn_norm_w);

        ggml_tensor* attn_out = core_attn::kv_self_attn(ctx0, gf, h, blk.attn_q_w, blk.attn_k_w, blk.attn_v_w,
                                                        blk.attn_o_w, blk.q_norm_w, blk.k_norm_w, positions,
                                                        causal_mask, ctx->kv_k, ctx->kv_v, (int)il, n_past, attn_p);
        cur = ggml_add(ctx0, residual, attn_out);

        residual = cur;
        h = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
        h = ggml_mul(ctx0, h, blk.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, h, blk.ffn_gate_w, blk.ffn_up_w, blk.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
    cur = ggml_mul(ctx0, cur, llm.final_norm_w);

    if (last_token_only && n_tokens > 1)
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(n_tokens - 1) * cur->nb[1]);
    cur = ggml_mul_mat(ctx0, llm.lm_head_w, cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

// ===========================================================================
// KV cache management
// ===========================================================================

extern "C" bool moss_diarize_kv_init(struct moss_diarize_context* ctx, int max_ctx) {
    if (!ctx)
        return false;
    const auto& hp = ctx->model.hparams;
    const int n_layers = (int)hp.llm_layers;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int hd = (int)hp.llm_head_dim;

    ggml_type kv_type = core_attn::kv_dtype_from_env("moss_diarize");

    struct ggml_init_params kv_params = {2 * ggml_tensor_overhead(), nullptr, true};
    ctx->kv_ctx = ggml_init(kv_params);
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_type, hd, max_ctx, n_kv, n_layers);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_type, hd, max_ctx, n_kv, n_layers);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");

    ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, ctx->backend);
    if (!ctx->kv_buf) {
        fprintf(stderr, "moss_diarize: kv alloc failed for max_ctx=%d\n", max_ctx);
        ggml_free(ctx->kv_ctx);
        ctx->kv_ctx = nullptr;
        return false;
    }
    ctx->kv_max_ctx = max_ctx;
    ctx->kv_n_used = 0;
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
    return true;
}

extern "C" void moss_diarize_kv_reset(struct moss_diarize_context* ctx) {
    if (ctx && ctx->kv_buf) {
        ggml_backend_buffer_clear(ctx->kv_buf, 0);
        ctx->kv_n_used = 0;
    }
}

// ===========================================================================
// Tokenizer (GPT-2 byte-level BPE via core_bpe)
// ===========================================================================

extern "C" int moss_diarize_tokenize(struct moss_diarize_context* ctx, const char* text, int32_t* out_tokens,
                                     int max_tokens) {
    if (!ctx || !text || !out_tokens || max_tokens <= 0)
        return 0;
    const auto& v = ctx->vocab;
    std::vector<int32_t> result;
    const std::string s = text;
    size_t i = 0;
    while (i < s.size()) {
        // Check for special tokens <|...|>
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
    int n = std::min((int)result.size(), max_tokens);
    std::memcpy(out_tokens, result.data(), (size_t)n * sizeof(int32_t));
    return n;
}

extern "C" const char* moss_diarize_token_text(struct moss_diarize_context* ctx, int token_id) {
    if (!ctx || token_id < 0 || token_id >= (int)ctx->vocab.id_to_token.size())
        return nullptr;
    return ctx->vocab.id_to_token[token_id].c_str();
}

// ===========================================================================
// Embed tokens
// ===========================================================================

extern "C" float* moss_diarize_embed_tokens(struct moss_diarize_context* ctx, const int32_t* token_ids, int n_tokens) {
    if (!ctx || !token_ids || n_tokens <= 0)
        return nullptr;
    const int d = (int)ctx->model.hparams.llm_dim;

    if (n_tokens == 1 && ctx->model.llm.embed_w) {
        const ggml_tensor* w = ctx->model.llm.embed_w;
        const size_t row_bytes = ggml_row_size(w->type, d);
        float* result = (float*)malloc((size_t)d * sizeof(float));
        if (!result)
            return nullptr;
        std::vector<uint8_t> raw(row_bytes);
        ggml_backend_tensor_get(w, raw.data(), (size_t)token_ids[0] * row_bytes, row_bytes);
        if (w->type == GGML_TYPE_F32)
            std::memcpy(result, raw.data(), (size_t)d * sizeof(float));
        else
            ggml_get_type_traits(w->type)->to_float(raw.data(), result, d);
        return result;
    }

    struct ggml_init_params gp = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(gp);
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "token_ids");
    ggml_set_input(ids);
    ggml_tensor* emb = ggml_get_rows(ctx0, ctx->model.llm.embed_w, ids);
    ggml_set_name(emb, "embeds");
    ggml_set_output(emb);
    ggml_build_forward_expand(gf, emb);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        ggml_free(ctx0);
        return nullptr;
    }
    ggml_backend_tensor_set(ids, token_ids, 0, (size_t)n_tokens * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx0);
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "embeds");
    float* result = (float*)malloc((size_t)d * n_tokens * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)d * n_tokens * sizeof(float));
    ggml_free(ctx0);
    return result;
}

// ===========================================================================
// Run LLM with KV cache
// ===========================================================================

extern "C" float* moss_diarize_run_llm_kv(struct moss_diarize_context* ctx, const float* inputs_embeds, int n_tokens,
                                          int n_past, int* out_n_tokens, int* out_vocab_size) {
    if (!ctx || !inputs_embeds || n_tokens <= 0 || !ctx->kv_k)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_dim;
    const int vocab = (int)hp.llm_vocab_size;
    const int Lk = n_past + n_tokens;

    ggml_cgraph* gf = moss_diarize_build_llm_kv_graph(ctx, n_tokens, n_past, /*last_token_only=*/true);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "moss_diarize: llm alloc failed\n");
        return nullptr;
    }
    ggml_tensor* emb_in = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(emb_in, inputs_embeds, 0, (size_t)d * n_tokens * sizeof(float));

    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = n_past + i;
    ggml_backend_tensor_set(pos_in, positions.data(), 0, (size_t)n_tokens * sizeof(int32_t));

    if (n_tokens > 1) {
        ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
        std::vector<ggml_fp16_t> mask((size_t)Lk * n_tokens);
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; q++) {
            int abs_q = n_past + q;
            for (int k = 0; k < Lk; k++)
                mask[(size_t)q * Lk + k] = (k <= abs_q) ? zero_h : neginf_h;
        }
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "moss_diarize: llm compute failed\n");
        return nullptr;
    }
    ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "logits");
    if (!logits_t)
        return nullptr;
    if (out_n_tokens)
        *out_n_tokens = 1;
    if (out_vocab_size)
        *out_vocab_size = vocab;
    float* result = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(logits_t, result, 0, (size_t)vocab * sizeof(float));
    ctx->kv_n_used = Lk;
    return result;
}

// ===========================================================================
// Prompt builder — ChatML with system instruction + time markers
// ===========================================================================

// Build prompt token IDs matching the Python processor exactly.
//
// The Python prompt (apply_chat_template + _expand_audio_token) is a SINGLE
// user turn with audio first, then text:
//
//   <|im_start|>user\n
//   <|audio_start|>[audio_pad × N with time markers]<|audio_end|>\n
//   PROMPT_TEXT<|im_end|>\n
//   <|im_start|>assistant\n
//
// NO system turn. The instruction lives inside the user message.
// Time markers: every `time_marker_every_seconds` seconds, inject bare digit
// tokens for the integer second count into the audio_pad sequence.
static std::vector<int32_t> moss_diarize_build_prompt(moss_diarize_context* ctx, int n_audio_tokens,
                                                      float audio_duration_s) {
    (void)audio_duration_s;
    const auto& hp = ctx->model.hparams;
    std::vector<int32_t> ids;

    // Instruction text — use ask_override if set, else default diarize instruction
    std::string prompt_text;
    if (!ctx->ask_override.empty()) {
        prompt_text = ctx->ask_override;
    } else {
        prompt_text = "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，"
                      "正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。";
    }
    if (!ctx->language.empty())
        prompt_text += "\nLanguage: " + ctx->language;
    if (!ctx->hotwords.empty())
        prompt_text += "热词提示：" + ctx->hotwords;

    // Tokenize instruction text
    std::vector<int32_t> text_toks(prompt_text.size() * 4 + 64);
    int text_n = moss_diarize_tokenize(ctx, prompt_text.c_str(), text_toks.data(), (int)text_toks.size());
    text_toks.resize(text_n);

    const int32_t TOK_USER = 872;        // "user"
    const int32_t TOK_ASSISTANT = 77091; // "assistant"
    const int32_t TOK_NL = 198;          // "\n"

    // <|im_start|>user\n
    ids.push_back((int32_t)hp.im_start_id);
    ids.push_back(TOK_USER);
    ids.push_back(TOK_NL);

    // <|audio_start|>
    ids.push_back((int32_t)hp.audio_start_id);

    // Audio pad tokens with time markers (bare digit tokens every N seconds).
    int32_t digit_ids[10];
    for (int d = 0; d < 10; d++) {
        char ds[2] = {(char)('0' + d), 0};
        std::vector<int32_t> dtoks(4);
        int dn = moss_diarize_tokenize(ctx, ds, dtoks.data(), (int)dtoks.size());
        digit_ids[d] = (dn > 0) ? dtoks[0] : (int32_t)('0' + d);
    }

    const int marker_every = (int)hp.time_marker_every_seconds;
    const int tokens_per_marker = (int)(hp.audio_tokens_per_second * (float)marker_every);
    const float duration = (float)n_audio_tokens / hp.audio_tokens_per_second;
    int consumed = 0;

    if (marker_every > 0 && tokens_per_marker > 0) {
        for (int sec = marker_every; sec <= (int)duration; sec += marker_every) {
            int pos = (sec / marker_every) * tokens_per_marker;
            int segment_len = pos - consumed;
            for (int j = 0; j < segment_len && consumed < n_audio_tokens; j++) {
                ids.push_back((int32_t)hp.audio_pad_id);
                consumed++;
            }
            char sec_str[16];
            snprintf(sec_str, sizeof(sec_str), "%d", sec);
            for (int k = 0; sec_str[k]; k++)
                ids.push_back(digit_ids[sec_str[k] - '0']);
        }
    }
    for (int j = consumed; j < n_audio_tokens; j++)
        ids.push_back((int32_t)hp.audio_pad_id);

    // <|audio_end|>\n
    ids.push_back((int32_t)hp.audio_end_id);
    ids.push_back(TOK_NL);

    // PROMPT_TEXT<|im_end|>\n
    ids.insert(ids.end(), text_toks.begin(), text_toks.end());
    ids.push_back((int32_t)hp.im_end_id);
    ids.push_back(TOK_NL);

    // <|im_start|>assistant\n
    ids.push_back((int32_t)hp.im_start_id);
    ids.push_back(TOK_ASSISTANT);
    ids.push_back(TOK_NL);

    return ids;
}

// ===========================================================================
// Output parser: [timestamp][Sxx]text[timestamp]
// ===========================================================================


// ===========================================================================
// High-level transcribe
// ===========================================================================

static char* moss_diarize_impl(struct moss_diarize_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d_llm = (int)hp.llm_dim;
    moss_diarize_bench_stage _b_total("total");

    float audio_duration_s = (float)n_samples / (float)hp.sample_rate;
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "moss_diarize: %d samples (%.1f s)\n", n_samples, audio_duration_s);

    // 1+2. Encoder (includes mel computation + 30s chunking internally)
    int T_enc = 0, enc_d = 0;
    float* encoder_out = nullptr;
    {
        moss_diarize_bench_stage _b("encoder");
        encoder_out = moss_diarize_run_encoder(ctx, samples, n_samples, &T_enc, &enc_d);
    }
    if (!encoder_out) {
        fprintf(stderr, "moss_diarize: encoder failed\n");
        return nullptr;
    }
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "moss_diarize: encoder %d frames × %d dims\n", T_enc, enc_d);

    // 3. Adaptor (4x merge + VQAdaptor)
    int adapt_T = 0, adapt_d = 0;
    float* audio_embeds = nullptr;
    {
        moss_diarize_bench_stage _b("adaptor");
        audio_embeds = moss_diarize_run_adaptor(ctx, encoder_out, T_enc, enc_d, &adapt_T, &adapt_d);
    }
    free(encoder_out);
    if (!audio_embeds)
        return nullptr;
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "moss_diarize: adaptor %d frames × %d dims\n", adapt_T, adapt_d);

    // 4. Build prompt + embed text tokens
    std::vector<int32_t> prompt_ids = moss_diarize_build_prompt(ctx, adapt_T, audio_duration_s);
    int n_prompt = (int)prompt_ids.size();

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "moss_diarize: %d enc, %d merged, %d prompt tokens\n", T_enc, adapt_T, n_prompt);
    const char* _dbg = std::getenv("CRISPASR_MOSS_DIARIZE_DEBUG");
    if (ctx->params.verbosity >= 2 || (_dbg && _dbg[0] == '1')) {
        fprintf(stderr, "moss_diarize: prompt first20:");
        for (int i = 0; i < std::min(20, n_prompt); i++)
            fprintf(stderr, " %d", prompt_ids[i]);
        fprintf(stderr, " ... last10:");
        for (int i = std::max(0, n_prompt - 10); i < n_prompt; i++)
            fprintf(stderr, " %d", prompt_ids[i]);
        fprintf(stderr, "\n");
        // Count audio_pad tokens
        int n_pads = 0;
        for (int i = 0; i < n_prompt; i++)
            if (prompt_ids[i] == (int32_t)hp.audio_pad_id)
                n_pads++;
        fprintf(stderr, "moss_diarize: %d audio_pad tokens (expected %d), audio_pad_id=%d\n", n_pads, adapt_T,
                (int)hp.audio_pad_id);
    }

    float* text_embeds = moss_diarize_embed_tokens(ctx, prompt_ids.data(), n_prompt);
    if (!text_embeds) {
        free(audio_embeds);
        return nullptr;
    }

    // 5. Scatter audio embeddings at audio_pad positions
    {
        int audio_idx = 0;
        for (int pos = 0; pos < n_prompt; pos++) {
            if (prompt_ids[pos] == (int32_t)hp.audio_pad_id && audio_idx < adapt_T) {
                memcpy(text_embeds + (size_t)pos * d_llm, audio_embeds + (size_t)audio_idx * d_llm,
                       (size_t)d_llm * sizeof(float));
                audio_idx++;
            }
        }
    }
    free(audio_embeds);

    // 6. KV cache + prefill
    const int max_new = ctx->max_new_tokens > 0 ? ctx->max_new_tokens : 1024;
    int max_ctx = n_prompt + max_new;
    if (ctx->kv_k) {
        if (ctx->kv_max_ctx < max_ctx) {
            if (ctx->kv_buf)
                ggml_backend_buffer_free(ctx->kv_buf);
            if (ctx->kv_ctx)
                ggml_free(ctx->kv_ctx);
            ctx->kv_buf = nullptr;
            ctx->kv_ctx = nullptr;
            ctx->kv_k = nullptr;
            ctx->kv_v = nullptr;
            moss_diarize_kv_init(ctx, max_ctx);
        } else {
            moss_diarize_kv_reset(ctx);
        }
    } else {
        moss_diarize_kv_init(ctx, max_ctx);
    }

    int vocab = 0;
    float* logits = moss_diarize_run_llm_kv(ctx, text_embeds, n_prompt, 0, nullptr, &vocab);
    free(text_embeds);
    if (!logits)
        return nullptr;

    // 7. Decode  (max_new computed above, honors --max-new-tokens, #292)
    std::vector<int32_t> generated;
    if (ctx->beam_size > 1 && ctx->temperature > 0.0f && ctx->params.verbosity >= 1)
        fprintf(stderr, "moss_diarize: beam search active; sampling settings ignored\n");
    if (ctx->beam_size > 1) {
        auto replay = [&vocab](moss_diarize_context* c, const int32_t* toks, int n, int prompt_len) -> float* {
            float* emb = moss_diarize_embed_tokens(c, toks, n);
            if (!emb)
                return nullptr;
            int dummy = 0;
            float* lg = moss_diarize_run_llm_kv(c, emb, n, prompt_len, &dummy, &vocab);
            std::free(emb);
            return lg;
        };
        core_beam_decode::Config cfg;
        cfg.max_new_tokens = max_new;
        cfg.eos_id = (int)hp.im_end_id;
        cfg.vocab_size = vocab;
        cfg.beam_size = ctx->beam_size;
        cfg.prompt_len = n_prompt;
        auto br = core_beam_decode::run_with_probs(ctx, logits, replay, cfg);
        generated = std::move(br.tokens);
        logits = nullptr;
        if (!generated.empty() && generated.back() == (int)hp.im_end_id)
            generated.pop_back();
    } else {
        const bool sampling = ctx->temperature > 0.0f;
        core_logit_sampler::Params sp;
        sp.temperature = ctx->temperature;
        sp.top_k = ctx->top_k;
        sp.top_p = ctx->top_p;
        std::mt19937_64 rng(ctx->seed != 0 ? ctx->seed : (uint64_t)std::random_device{}());
        for (int step = 0; step < max_new; step++) {
            // NaN-robust argmax (see canary_qwen note): seed -inf, skip non-finite,
            // abort if the whole row is non-finite. Sampling (#564) also skips
            // non-finite logits and returns -1 for an all-non-finite row.
            int best_id = -1;
            float best_val = -std::numeric_limits<float>::infinity();
            if (sampling) {
                best_id = core_logit_sampler::sample(logits, vocab, sp, rng);
                if (best_id >= 0)
                    best_val = logits[best_id];
            } else {
                for (int i = 0; i < vocab; i++)
                    if (std::isfinite(logits[i]) && logits[i] > best_val) {
                        best_val = logits[i];
                        best_id = i;
                    }
            }
            if (best_id < 0) {
                free(logits);
                logits = nullptr;
                fprintf(stderr, "moss_diarize: non-finite logits at step %d — aborting decode\n", step);
                break;
            }
            free(logits);
            logits = nullptr;
            if (ctx->params.verbosity >= 2 && step < 10)
                fprintf(stderr, "moss_diarize: step %d argmax=%d (%.4f)\n", step, best_id, best_val);
            if (best_id == (int)hp.im_end_id)
                break;
            generated.push_back(best_id);
            float* next_emb = moss_diarize_embed_tokens(ctx, &best_id, 1);
            if (!next_emb)
                break;
            int dummy = 0;
            logits = moss_diarize_run_llm_kv(ctx, next_emb, 1, n_prompt + (int)generated.size() - 1, &dummy, &vocab);
            free(next_emb);
            if (!logits)
                break;
        }
        if (logits)
            free(logits);
    }

    // 8. Detokenize
    std::string result;
    for (int id : generated) {
        const char* t = moss_diarize_token_text(ctx, id);
        if (t)
            result += core_bpe::token_bytes_to_utf8(t);
    }
    if (ctx->params.verbosity >= 1)
        // #318: print the FULL tagged transcript, not the first 200 chars — this
        // `[t][Sxx] …[t]` line (speaker tags + per-phrase timings) is the only
        // place the rich structure surfaces during streaming (the partial JSONs
        // carry text only), and users rely on it. One line, printed once per call.
        fprintf(stderr, "moss_diarize: %zu tokens: \"%s\"\n", generated.size(), result.c_str());

    // n-gram loop fix
    {
        const char* no_fix = std::getenv("CRISPASR_MOSS_DIARIZE_NO_LOOPFIX");
        if (!(no_fix && no_fix[0] == '1')) {
            std::string fixed = core_ngram::fix_loops(result);
            if (fixed != result && ctx->params.verbosity >= 1)
                fprintf(stderr, "moss_diarize: collapsed n-gram loop(s) (%zu → %zu chars)\n", result.size(),
                        fixed.size());
            result = std::move(fixed);
        }
    }

    char* out = (char*)malloc(result.size() + 1);
    memcpy(out, result.c_str(), result.size() + 1);
    return out;
}

extern "C" char* moss_diarize_transcribe(struct moss_diarize_context* ctx, const float* samples, int n_samples) {
    return moss_diarize_impl(ctx, samples, n_samples);
}

extern "C" int moss_diarize_transcribe_segments(struct moss_diarize_context* ctx, const float* samples, int n_samples,
                                                struct moss_diarize_segment* out_segments, int max_segments) {
    char* raw = moss_diarize_impl(ctx, samples, n_samples);
    if (!raw)
        return 0;
    std::string text(raw);
    free(raw);

    auto segs = moss_diarize_parse_output(text);
    int n = std::min((int)segs.size(), max_segments);
    for (int i = 0; i < n; i++)
        out_segments[i] = segs[i];
    return n;
}

// ===========================================================================
// Hotwords / beam size setters
// ===========================================================================

extern "C" void moss_diarize_set_hotwords(struct moss_diarize_context* ctx, const char* hotwords) {
    if (!ctx)
        return;
    ctx->hotwords = hotwords ? hotwords : "";
}

extern "C" void moss_diarize_set_beam_size(struct moss_diarize_context* ctx, int beam_size) {
    if (ctx)
        ctx->beam_size = beam_size > 0 ? beam_size : 1;
}

extern "C" void moss_diarize_set_sampling(struct moss_diarize_context* ctx, float temperature, int top_k, float top_p,
                                          uint64_t seed) {
    if (!ctx)
        return;
    ctx->temperature = std::isfinite(temperature) && temperature > 0.0f ? temperature : 0.0f;
    ctx->top_k = top_k > 0 ? top_k : 0;
    ctx->top_p = std::isfinite(top_p) && top_p > 0.0f && top_p < 1.0f ? top_p : 1.0f;
    ctx->seed = seed;
}

// #292: forward the user's --max-new-tokens. Pass <= 0 to keep the 1024 default
// (the caller decides "explicit" — see the CLI adapter's max_new_tokens_explicit
// gate, so an unset CLI default of 512 does not silently shrink this backend).
extern "C" void moss_diarize_set_max_new_tokens(struct moss_diarize_context* ctx, int max_new_tokens) {
    if (ctx && max_new_tokens > 0)
        ctx->max_new_tokens = max_new_tokens;
}

extern "C" void moss_diarize_set_ask(struct moss_diarize_context* ctx, const char* instruction) {
    if (!ctx)
        return;
    ctx->ask_override = (instruction && instruction[0]) ? instruction : "";
}

extern "C" void moss_diarize_set_language(struct moss_diarize_context* ctx, const char* lang) {
    if (!ctx)
        return;
    ctx->language = (lang && lang[0]) ? lang : "";
}

// ===========================================================================
// Init / Free
// ===========================================================================

extern "C" struct moss_diarize_params moss_diarize_default_params(void) {
    moss_diarize_params p;
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.flash_attn = false;
    return p;
}

extern "C" struct moss_diarize_context* moss_diarize_init_from_file(const char* path_model,
                                                                    struct moss_diarize_params params) {
    auto* ctx = new moss_diarize_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads;
    ctx->model_path = path_model;

    ctx->backend = params.use_gpu ? crispasr_init_gpu_backend() : core_cpu_backend::init();
    if (!ctx->backend)
        ctx->backend = core_cpu_backend::init();
    ctx->backend_cpu = core_cpu_backend::init();
    if (ctx->backend_cpu)
        core_cpu_backend::set_n_threads(ctx->backend_cpu, ctx->n_threads);

    {
        const char* force_cpu = std::getenv("CRISPASR_MOSS_DIARIZE_FORCE_CPU");
        if (force_cpu && force_cpu[0] == '1')
            ctx->backend = ctx->backend_cpu;
    }
    if (core_cpu_backend::is_cpu(ctx->backend))
        core_cpu_backend::set_n_threads(ctx->backend, ctx->n_threads);

    // Encoder attention path — flash by default, manual on Vulkan
    {
        const char* force_flash = std::getenv("CRISPASR_MOSS_DIARIZE_ENC_FLASH");
        const char* force_manual = std::getenv("CRISPASR_MOSS_DIARIZE_ENC_MANUAL");
        if (force_flash && force_flash[0] == '1') {
            ctx->enc_use_flash = true;
        } else if (force_manual && force_manual[0] == '1') {
            ctx->enc_use_flash = false;
        } else {
            const char* bname = ggml_backend_name(ctx->backend);
            ctx->enc_use_flash = !(bname && std::strncmp(bname, "Vulkan", 6) == 0);
        }
    }

    // Scheduler
    ggml_backend_t backends[] = {ctx->backend, ctx->backend_cpu};
    int n_backends = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_backends, 16384, false, false);
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    if (!moss_diarize_load_model(ctx->model, ctx->vocab, path_model, ctx->backend)) {
        fprintf(stderr, "moss_diarize: failed to load model from '%s'\n", path_model);
        moss_diarize_free(ctx);
        return nullptr;
    }

    return ctx;
}

extern "C" void moss_diarize_free(struct moss_diarize_context* ctx) {
    if (!ctx)
        return;
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf)
        core_gguf::release_weight_buffer(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}
