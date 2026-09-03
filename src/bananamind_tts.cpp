// bananamind_tts.cpp -- BananaMind-TTS-V2.1 (Tacotron-lite + HiFi-GAN) native ggml runtime.
//
// Architecture overview:
//   Encoder:  Embedding -> 3x (Conv1d(256,256,k=5) + BN + ReLU) -> BiLSTM(256)
//   Decoder:  Autoregressive GRU loop with location-sensitive attention
//     Per step:
//       1. Prenet(last_mel) -> (128,)
//       2. Attention GRU: GRUCell([prenet;context], attn_hidden) -> (512,)
//       3. Location-sensitive attention(attn_hidden, encoder_out) -> context(256,), weights
//       4. Decoder GRU: GRUCell([attn_hidden;context], dec_hidden) -> (512,)
//       5. mel_proj([dec_hidden;context]) -> (80*4,)  [reduction_factor=4]
//       6. stop_proj([dec_hidden;context]) -> (4,)
//   Postnet:  5x (Conv1d + BN + Tanh) residual refinement
//   Vocoder:  HiFi-GAN (reuses core_hifigan::forward)

#include "bananamind_tts.h"

#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h" // crispasr_init_gpu_backend (#214)
#include "core/hifigan.h"
#include "core/lstm.h"
#include "core/crispasr_env.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "core/ggml_cpu_backend.h"

// ---------------------------------------------------------------------------
// Debug gate
// ---------------------------------------------------------------------------

static bool debug_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("CRISPASR_BANANAMIND_DEBUG");
        v = (e && e[0] == '1') ? 1 : 0;
    }
    return v == 1;
}

// ---------------------------------------------------------------------------
// Bench instrumentation (BANANAMIND_TTS_BENCH=1)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Vocoder path gate (CRISPASR_BANANAMIND_VOC_LEGACY=1)
// ---------------------------------------------------------------------------
//
// The HiFi-GAN vocoder is ~50-78% of this backend's synthesis wall. It has
// always run through core_hifigan::forward(), but was missed by the two
// rollouts that sped that primitive up for its siblings:
//
//   * a862f2de wired col2im_1d into every ConvTranspose1d runtime — speecht5,
//     fastpitch and zonos got pre-permuted upsample kernels; bananamind did
//     not, so its four `voc.ups.*` stages still took the legacy
//     ggml_conv_transpose_1d + view + cont branch in core/hifigan.h.
//   * the FASTCONV cast-kill (cf. 323e96f2) never reached it either.
//
// The new path is the default. Setting CRISPASR_BANANAMIND_VOC_LEGACY=1
// restores the old one exactly (no ups_w_perm, no FASTCONV) as the A/B arm.
static bool bananamind_voc_legacy() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_BANANAMIND_VOC_LEGACY");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v == 1;
}

static bool bananamind_tts_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_BANANAMIND_TTS_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct bananamind_tts_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit bananamind_tts_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~bananamind_tts_bench_stage() {
        if (!bananamind_tts_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  bananamind_tts_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

struct bananamind_hparams {
    int hidden_size = 256;
    int decoder_dim = 512;
    int attention_dim = 128;
    int n_mels = 80;
    int reduction_factor = 4;
    int encoder_conv_layers = 3;
    int location_channels = 32;
    int location_kernel_size = 31;
    int postnet_channels = 512;
    int postnet_layers = 5;
    int sample_rate = 22050;
    int max_decoder_steps = 1200;
    float stop_threshold = 0.55f;
    int attention_window = 12;
    int vocab_size = 39;

    // Prenet sizes
    int prenet_size_0 = 256;
    int prenet_size_1 = 128;

    // Mel normalization
    bool mel_normalized = false;
    float mel_mean = 0.0f;
    float mel_std = 1.0f;
    float mel_min = -12.0f;
    float mel_max = 8.0f;

    // Vocoder
    core_hifigan::hparams voc_hp;

    // Ampersand replacement
    std::string ampersand_replacement = " and ";
};

// ---------------------------------------------------------------------------
// Character tokenizer
// ---------------------------------------------------------------------------

struct bananamind_tokenizer {
    std::vector<std::string> symbols;
    std::map<std::string, int> sym_to_id;
    int pad_id = 0;
    int unk_id = 1;
    int bos_id = 2;
    int eos_id = 3;
    std::string ampersand_replacement = " and ";

    void build(const std::vector<std::string>& syms, const std::string& amp_repl) {
        symbols = syms;
        ampersand_replacement = amp_repl;
        sym_to_id.clear();
        for (int i = 0; i < (int)symbols.size(); i++) {
            sym_to_id[symbols[i]] = i;
            if (symbols[i] == "<pad>")
                pad_id = i;
            if (symbols[i] == "<unk>")
                unk_id = i;
            if (symbols[i] == "<bos>")
                bos_id = i;
            if (symbols[i] == "<eos>")
                eos_id = i;
        }
    }

    std::vector<int> encode(const std::string& text) const {
        // Normalize: lowercase, replace unicode quotes, replace &, filter
        std::string norm;
        norm.reserve(text.size());

        // First pass: lowercase + basic replacements
        std::string lower;
        lower.reserve(text.size());
        for (size_t i = 0; i < text.size();) {
            unsigned char c = (unsigned char)text[i];
            if (c < 0x80) {
                if (c == '&') {
                    lower += ampersand_replacement;
                    i++;
                } else {
                    lower += (char)tolower(c);
                    i++;
                }
            } else if (c == 0xE2 && i + 2 < text.size()) {
                // Handle UTF-8 3-byte sequences for smart quotes, dashes
                unsigned char c2 = (unsigned char)text[i + 1];
                unsigned char c3 = (unsigned char)text[i + 2];
                if (c2 == 0x80) {
                    if (c3 == 0x99 || c3 == 0x98) {
                        lower += '\'';
                        i += 3;
                        continue;
                    } // curly quotes
                    if (c3 == 0x9C || c3 == 0x9D || c3 == 0x9E || c3 == 0x9F) {
                        lower += '"';
                        i += 3;
                        continue;
                    }
                    if (c3 == 0x93 || c3 == 0x94) {
                        lower += '-';
                        i += 3;
                        continue;
                    } // em/en dash
                }
                // Skip other multi-byte
                i += 3;
            } else if (c == 0xC2 && i + 1 < text.size()) {
                unsigned char c2 = (unsigned char)text[i + 1];
                if (c2 == 0xAB || c2 == 0xBB) {
                    lower += '"';
                    i += 2;
                    continue;
                } // guillemets
                i += 2;
            } else if (c == 0xC3 && i + 1 < text.size()) {
                // German umlauts: ä=C3A4, ö=C3B6, ü=C3BC, ß=C39F
                // Uppercase: Ä=C384, Ö=C396, Ü=C39C
                unsigned char c2 = (unsigned char)text[i + 1];
                if (c2 == 0xA4 || c2 == 0x84) {
                    lower += "\xC3\xA4";
                    i += 2;
                    continue;
                } // ä/Ä
                else if (c2 == 0xB6 || c2 == 0x96) {
                    lower += "\xC3\xB6";
                    i += 2;
                    continue;
                } // ö/Ö
                else if (c2 == 0xBC || c2 == 0x9C) {
                    lower += "\xC3\xBC";
                    i += 2;
                    continue;
                } // ü/Ü
                else if (c2 == 0x9F) {
                    lower += "\xC3\x9F";
                    i += 2;
                    continue;
                } // ß
                i += 2;
            } else {
                // Skip other multi-byte
                int skip = 1;
                if (c >= 0xC0)
                    skip = 2;
                if (c >= 0xE0)
                    skip = 3;
                if (c >= 0xF0)
                    skip = 4;
                i += skip;
            }
        }

        // Second pass: filter to allowed characters only
        for (size_t i = 0; i < lower.size();) {
            unsigned char c = (unsigned char)lower[i];
            if (c < 0x80) {
                // ASCII: check if in vocab
                std::string s(1, (char)c);
                if (sym_to_id.count(s)) {
                    norm += s;
                } else {
                    norm += ' ';
                }
                i++;
            } else if (c == 0xC3 && i + 1 < lower.size()) {
                // 2-byte UTF-8 (German umlauts)
                std::string s = lower.substr(i, 2);
                if (sym_to_id.count(s)) {
                    norm += s;
                } else {
                    norm += ' ';
                }
                i += 2;
            } else {
                norm += ' ';
                int skip = 1;
                if (c >= 0xC0)
                    skip = 2;
                if (c >= 0xE0)
                    skip = 3;
                if (c >= 0xF0)
                    skip = 4;
                i += skip;
            }
        }

        // Collapse whitespace
        std::string clean;
        clean.reserve(norm.size());
        bool last_space = true;
        for (char c : norm) {
            if (c == ' ') {
                if (!last_space)
                    clean += c;
                last_space = true;
            } else {
                clean += c;
                last_space = false;
            }
        }
        // Trim trailing space
        if (!clean.empty() && clean.back() == ' ')
            clean.pop_back();
        // Trim leading space
        if (!clean.empty() && clean.front() == ' ')
            clean = clean.substr(1);

        // Encode to IDs
        std::vector<int> ids;
        ids.push_back(bos_id);
        for (size_t i = 0; i < clean.size();) {
            unsigned char c = (unsigned char)clean[i];
            std::string s;
            if (c < 0x80) {
                s = std::string(1, (char)c);
                i++;
            } else if (c == 0xC3 && i + 1 < clean.size()) {
                s = clean.substr(i, 2);
                i += 2;
            } else {
                i++;
                continue;
            }
            auto it = sym_to_id.find(s);
            ids.push_back(it != sym_to_id.end() ? it->second : unk_id);
        }
        ids.push_back(eos_id);
        return ids;
    }
};

// ---------------------------------------------------------------------------
// Model context
// ---------------------------------------------------------------------------

struct bananamind_tts_context {
    bananamind_hparams hp;
    bananamind_tokenizer tokenizer;

    // Backend
    ggml_backend_t backend = nullptr;     // primary (GPU when available, else CPU)
    ggml_backend_t backend_cpu = nullptr; // always CPU (n_threads + per-op fallback)
    ggml_backend_sched_t sched = nullptr;

    // Weight storage
    ggml_context* w_ctx = nullptr;
    ggml_backend_buffer_t w_buf = nullptr;
    core_gguf::tensor_map tensors;

    // Pre-permuted HiFi-GAN upsample weights for the decomposed col2im path,
    // and the FASTCONV F32 kernel cache. Both empty under
    // CRISPASR_BANANAMIND_VOC_LEGACY=1, which makes core_hifigan::forward()
    // take exactly the branches it took before.
    std::vector<ggml_tensor*> ups_w_perm;
    ggml_context* ctx_perm = nullptr;
    ggml_backend_buffer_t buf_perm = nullptr;
    core_dac::fastconv_cache voc_fc;

    // Helper
    ggml_tensor* W(const std::string& name) const {
        auto it = tensors.find(name);
        return (it != tensors.end()) ? it->second : nullptr;
    }

    int n_threads = 4;
    int verbosity = 1;
};

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

static bool load_model(bananamind_tts_context* ctx, const char* path) {
    auto& hp = ctx->hp;

    // Pass 1: metadata
    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta)
        return false;

    hp.hidden_size = (int)core_gguf::kv_u32(meta, "bananamind_tts.hidden_size", hp.hidden_size);
    hp.decoder_dim = (int)core_gguf::kv_u32(meta, "bananamind_tts.decoder_dim", hp.decoder_dim);
    hp.attention_dim = (int)core_gguf::kv_u32(meta, "bananamind_tts.attention_dim", hp.attention_dim);
    hp.n_mels = (int)core_gguf::kv_u32(meta, "bananamind_tts.n_mels", hp.n_mels);
    hp.reduction_factor = (int)core_gguf::kv_u32(meta, "bananamind_tts.reduction_factor", hp.reduction_factor);
    hp.encoder_conv_layers = (int)core_gguf::kv_u32(meta, "bananamind_tts.encoder_conv_layers", hp.encoder_conv_layers);
    hp.location_channels = (int)core_gguf::kv_u32(meta, "bananamind_tts.location_channels", hp.location_channels);
    hp.location_kernel_size =
        (int)core_gguf::kv_u32(meta, "bananamind_tts.location_kernel_size", hp.location_kernel_size);
    hp.postnet_channels = (int)core_gguf::kv_u32(meta, "bananamind_tts.postnet_channels", hp.postnet_channels);
    hp.postnet_layers = (int)core_gguf::kv_u32(meta, "bananamind_tts.postnet_layers", hp.postnet_layers);
    hp.sample_rate = (int)core_gguf::kv_u32(meta, "bananamind_tts.sample_rate", hp.sample_rate);
    hp.max_decoder_steps = (int)core_gguf::kv_u32(meta, "bananamind_tts.max_decoder_steps", hp.max_decoder_steps);
    hp.stop_threshold = core_gguf::kv_f32(meta, "bananamind_tts.stop_threshold", hp.stop_threshold);
    hp.attention_window = (int)core_gguf::kv_u32(meta, "bananamind_tts.attention_window", hp.attention_window);
    hp.vocab_size = (int)core_gguf::kv_u32(meta, "bananamind_tts.vocab_size", hp.vocab_size);

    // Mel normalization
    hp.mel_mean = core_gguf::kv_f32(meta, "bananamind_tts.mel_mean", 0.0f);
    hp.mel_std = core_gguf::kv_f32(meta, "bananamind_tts.mel_std", 1.0f);
    hp.mel_min = core_gguf::kv_f32(meta, "bananamind_tts.mel_min", -12.0f);
    hp.mel_max = core_gguf::kv_f32(meta, "bananamind_tts.mel_max", 8.0f);
    hp.mel_normalized = (hp.mel_std > 0.01f && hp.mel_std != 1.0f);

    // Ampersand replacement
    hp.ampersand_replacement = core_gguf::kv_str(meta, "bananamind_tts.ampersand_replacement", " and ");

    // Vocoder hparams
    hp.voc_hp.model_in_dim = hp.n_mels;
    hp.voc_hp.upsample_initial_ch = (int)core_gguf::kv_u32(meta, "bananamind_tts.voc_initial_channels", 256);
    hp.voc_hp.leaky_relu_slope = core_gguf::kv_f32(meta, "bananamind_tts.voc_leaky_relu_slope", 0.1f);
    hp.voc_hp.normalize_before = false; // BananaMind vocoder has no normalization

    // Read vocoder array params from GGUF KV
    auto read_u32_array = [&](const char* key, std::vector<int>& out, std::vector<int> fallback) {
        int kid = gguf_find_key(meta, key);
        if (kid >= 0) {
            int n = gguf_get_arr_n(meta, kid);
            const auto* d = (const uint32_t*)gguf_get_arr_data(meta, kid);
            out.resize(n);
            for (int i = 0; i < n; i++)
                out[i] = (int)d[i];
        } else {
            out = std::move(fallback);
        }
    };
    read_u32_array("bananamind_tts.voc_upsample_rates", hp.voc_hp.upsample_rates, {8, 8, 2, 2});
    read_u32_array("bananamind_tts.voc_upsample_kernel_sizes", hp.voc_hp.upsample_kernel_sizes, {16, 16, 4, 4});
    read_u32_array("bananamind_tts.voc_resblock_kernel_sizes", hp.voc_hp.resblock_kernel_sizes, {3, 7, 11});
    {
        int key_id = gguf_find_key(meta, "bananamind_tts.voc_resblock_dilations");
        int n_dilations = (int)core_gguf::kv_u32(meta, "bananamind_tts.voc_n_dilations", 3);
        if (key_id >= 0) {
            int n = gguf_get_arr_n(meta, key_id);
            const auto* d = (const uint32_t*)gguf_get_arr_data(meta, key_id);
            int n_kernels = (int)hp.voc_hp.resblock_kernel_sizes.size();
            hp.voc_hp.resblock_dilation_sizes.resize(n_kernels);
            for (int k = 0; k < n_kernels; k++) {
                hp.voc_hp.resblock_dilation_sizes[k].resize(n_dilations);
                for (int dd = 0; dd < n_dilations && (k * n_dilations + dd) < n; dd++) {
                    hp.voc_hp.resblock_dilation_sizes[k][dd] = (int)d[k * n_dilations + dd];
                }
            }
        } else {
            hp.voc_hp.resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
        }
    }

    // Tokenizer
    std::vector<std::string> vocab = core_gguf::kv_str_array(meta, "tokenizer.ggml.tokens");
    if (vocab.empty()) {
        fprintf(stderr, "bananamind_tts: no tokenizer vocabulary found in GGUF\n");
        core_gguf::free_metadata(meta);
        return false;
    }
    ctx->tokenizer.build(vocab, hp.ampersand_replacement);
    hp.vocab_size = (int)vocab.size();

    core_gguf::free_metadata(meta);

    // Pass 2: weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "bananamind_tts", wl)) {
        return false;
    }
    ctx->w_ctx = wl.ctx;
    ctx->w_buf = wl.buf;
    ctx->tensors = std::move(wl.tensors);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "bananamind_tts: loaded %zu tensors\n", ctx->tensors.size());
        fprintf(stderr, "  hidden=%d decoder=%d attn=%d mels=%d rf=%d vocab=%d sr=%d\n", hp.hidden_size, hp.decoder_dim,
                hp.attention_dim, hp.n_mels, hp.reduction_factor, hp.vocab_size, hp.sample_rate);
        if (hp.mel_normalized) {
            fprintf(stderr, "  mel_norm: mean=%.4f std=%.4f min=%.1f max=%.1f\n", hp.mel_mean, hp.mel_std, hp.mel_min,
                    hp.mel_max);
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Encoder: Embedding -> Conv+BN+ReLU -> BiLSTM
// ---------------------------------------------------------------------------

static std::vector<float> run_encoder(bananamind_tts_context* ctx, const std::vector<int>& token_ids) {
    const auto& hp = ctx->hp;
    const int T = (int)token_ids.size();
    const int H = hp.hidden_size;

    // Estimate tensor count for graph
    // BiLSTM creates ~20 tensors per timestep (gates, states, cpy ops)
    // Plus conv layers, embedding, etc.
    const int max_tensors = 512 + T * 40;
    const size_t mem_size = (size_t)max_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(32768, false);

    ggml_init_params ip = {mem_size, nullptr, true};
    ggml_context* gc = ggml_init(ip);
    if (!gc) {
        fprintf(stderr, "bananamind_tts: encoder graph context alloc failed\n");
        return {};
    }

    // Input token tensor
    ggml_tensor* tokens_t = ggml_new_tensor_1d(gc, GGML_TYPE_I32, T);
    ggml_set_name(tokens_t, "tokens");
    ggml_set_input(tokens_t);

    // Embedding lookup
    ggml_tensor* emb_w = ctx->W("enc.emb.weight");
    ggml_tensor* x = ggml_get_rows(gc, emb_w, tokens_t); // (H, T) in ggml

    // Conv blocks: conv1d + batch_norm + relu
    // Input x is (H, T) from embedding. ggml conv1d expects (T, C_in).
    // We need to transpose: (H, T) -> (T, H)
    x = ggml_cont(gc, ggml_transpose(gc, x)); // (T, H)

    for (int i = 0; i < hp.encoder_conv_layers; i++) {
        std::string pfx = "enc.conv." + std::to_string(i);
        ggml_tensor* conv_w = ctx->W(pfx + ".weight");

        if (!conv_w) {
            fprintf(stderr, "bananamind_tts: missing tensor %s.weight\n", pfx.c_str());
            ggml_free(gc);
            return {};
        }

        // Conv1d(H, H, k=5, p=2)
        x = ggml_conv_1d(gc, conv_w, x, 1, 2, 1);
        ggml_tensor* conv_b = ctx->W(pfx + ".bias");
        if (conv_b) {
            ggml_tensor* b = ggml_reshape_2d(gc, conv_b, 1, (int)conv_b->ne[0]);
            x = ggml_add(gc, x, b);
        }

        // BatchNorm: we need to compute BN at runtime since the params are
        // stored as separate running_mean/running_var/weight/bias tensors.
        // We'll use the fused BN approach from SpeechT5: pre-compute scale/shift
        // on the CPU side and inject as input tensors.
        // But first we need to build the graph, then set values after alloc.
        // Use the deferred pattern: mark scale/shift as inputs, fill after alloc.
        ggml_tensor* bn_scale = ggml_new_tensor_2d(gc, GGML_TYPE_F32, 1, H);
        ggml_set_name(bn_scale, (std::string("bn_scale_") + std::to_string(i)).c_str());
        ggml_set_input(bn_scale);

        ggml_tensor* bn_shift = ggml_new_tensor_2d(gc, GGML_TYPE_F32, 1, H);
        ggml_set_name(bn_shift, (std::string("bn_shift_") + std::to_string(i)).c_str());
        ggml_set_input(bn_shift);

        x = ggml_mul(gc, x, bn_scale);
        x = ggml_add(gc, x, bn_shift);

        // ReLU
        x = ggml_relu(gc, x);
    }

    // Transpose back for LSTM: (T, H) -> (H, T)
    x = ggml_cont(gc, ggml_transpose(gc, x)); // (H, T)

    // BiLSTM
    ggml_cgraph* gf = ggml_new_graph_custom(gc, 32768, false);

    ggml_tensor* lstm_out = core_lstm::lstm_bidir(
        gc, gf, x, ctx->W("enc.lstm.weight_ih_l0"), ctx->W("enc.lstm.weight_hh_l0"), ctx->W("enc.lstm.bias_ih_l0"),
        ctx->W("enc.lstm.bias_hh_l0"), ctx->W("enc.lstm.weight_ih_l0_reverse"), ctx->W("enc.lstm.weight_hh_l0_reverse"),
        ctx->W("enc.lstm.bias_ih_l0_reverse"), ctx->W("enc.lstm.bias_hh_l0_reverse"),
        H / 2); // hidden_size/2 per direction, concat -> H

    ggml_set_name(lstm_out, "encoder_out");
    ggml_set_output(lstm_out);
    ggml_build_forward_expand(gf, lstm_out);

    // Allocate through the scheduler so unsupported ops fall back to CPU
    // per-op when `backend` is a GPU.
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "bananamind_tts: encoder graph alloc failed\n");
        ggml_free(gc);
        return {};
    }

    // Set input: tokens
    ggml_backend_tensor_set(tokens_t, token_ids.data(), 0, T * sizeof(int));

    // Set BN scale/shift (pre-compute fused params from weight tensors)
    for (int i = 0; i < hp.encoder_conv_layers; i++) {
        std::string pfx = "enc.conv." + std::to_string(i);
        ggml_tensor* bn_w = ctx->W(pfx + ".bn_w");
        ggml_tensor* bn_b = ctx->W(pfx + ".bn_b");
        ggml_tensor* bn_mean = ctx->W(pfx + ".bn_mean");
        ggml_tensor* bn_var = ctx->W(pfx + ".bn_var");

        std::vector<float> scale(H), shift(H);
        if (bn_mean && bn_var && bn_w && bn_b) {
            std::vector<float> mean_v(H), var_v(H), w_v(H), b_v(H);
            ggml_backend_tensor_get(bn_mean, mean_v.data(), 0, H * sizeof(float));
            ggml_backend_tensor_get(bn_var, var_v.data(), 0, H * sizeof(float));
            ggml_backend_tensor_get(bn_w, w_v.data(), 0, H * sizeof(float));
            ggml_backend_tensor_get(bn_b, b_v.data(), 0, H * sizeof(float));

            for (int c = 0; c < H; c++) {
                float inv_std = 1.0f / sqrtf(var_v[c] + 1e-5f);
                scale[c] = w_v[c] * inv_std;
                shift[c] = b_v[c] - mean_v[c] * scale[c];
            }
        } else {
            for (int c = 0; c < H; c++) {
                scale[c] = 1.0f;
                shift[c] = 0.0f;
            }
        }

        ggml_tensor* bn_scale_t = ggml_graph_get_tensor(gf, (std::string("bn_scale_") + std::to_string(i)).c_str());
        ggml_tensor* bn_shift_t = ggml_graph_get_tensor(gf, (std::string("bn_shift_") + std::to_string(i)).c_str());
        if (bn_scale_t)
            ggml_backend_tensor_set(bn_scale_t, scale.data(), 0, H * sizeof(float));
        if (bn_shift_t)
            ggml_backend_tensor_set(bn_shift_t, shift.data(), 0, H * sizeof(float));
    }

    // Compute
    ggml_backend_sched_graph_compute(ctx->sched, gf);

    // Read output: (H, T) -> flat (ne[0]-contiguous = T_H interleaved)
    ggml_tensor* out = ggml_graph_get_tensor(gf, "encoder_out");
    std::vector<float> result(H * T);
    ggml_backend_tensor_get(out, result.data(), 0, H * T * sizeof(float));

    if (debug_enabled()) {
        // Data is ne[0]-contiguous: data[t * H + h]. Print first token's features.
        fprintf(stderr, "bananamind_tts: enc[0,:8] =");
        for (int h = 0; h < 8 && h < H; h++)
            fprintf(stderr, " %.7f", result[0 * H + h]);
        fprintf(stderr, "\n");
        fprintf(stderr, "bananamind_tts: enc[1,:8] =");
        for (int h = 0; h < 8 && h < H; h++)
            fprintf(stderr, " %.7f", result[1 * H + h]);
        fprintf(stderr, "\n");
    }

    ggml_free(gc);
    return result;
}

// ---------------------------------------------------------------------------
// GRU cell: manual gate computation (CPU-side for AR loop efficiency)
//
// PyTorch GRUCell stores weights as fused [3*H, input_dim] and [3*H, hidden_dim]
// Gate order: r (reset), z (update), n (new)
//   r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
//   z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
//   n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
//   h' = (1 - z) * n + z * h
// ---------------------------------------------------------------------------

static void gru_cell_cpu(const float* W_ih, const float* b_ih, const float* W_hh, const float* b_hh, const float* x,
                         int input_dim, float* h, int hidden_dim) {
    // Compute W_ih @ x + b_ih -> (3*H,)
    std::vector<float> ih(3 * hidden_dim, 0.0f);
    for (int i = 0; i < 3 * hidden_dim; i++) {
        float sum = b_ih[i];
        for (int j = 0; j < input_dim; j++) {
            sum += W_ih[i * input_dim + j] * x[j];
        }
        ih[i] = sum;
    }

    // Compute W_hh @ h + b_hh -> (3*H,)
    std::vector<float> hh(3 * hidden_dim, 0.0f);
    for (int i = 0; i < 3 * hidden_dim; i++) {
        float sum = b_hh[i];
        for (int j = 0; j < hidden_dim; j++) {
            sum += W_hh[i * hidden_dim + j] * h[j];
        }
        hh[i] = sum;
    }

    // Gate computation
    int H = hidden_dim;
    for (int i = 0; i < H; i++) {
        float r = 1.0f / (1.0f + expf(-(ih[i] + hh[i])));         // reset gate
        float z = 1.0f / (1.0f + expf(-(ih[H + i] + hh[H + i]))); // update gate
        float n = tanhf(ih[2 * H + i] + r * hh[2 * H + i]);       // new gate
        h[i] = (1.0f - z) * n + z * h[i];
    }
}

// ---------------------------------------------------------------------------
// Location-sensitive attention (CPU-side)
//
// energies = V * tanh(W_query @ query + W_memory @ memory + W_loc @ loc_features)
// weights = softmax(energies, masked)
// context = weights @ memory
// ---------------------------------------------------------------------------

static void location_attention_cpu(
    const float* query,             // (decoder_dim,)
    const float* memory,            // (hidden, T_enc) row-major
    const float* proc_memory,       // (attn_dim, T_enc) row-major — pre-computed W_memory @ memory
    const float* attn_weights_prev, // (T_enc,)
    const float* attn_cum,          // (T_enc,)
    int T_enc, int hidden, int decoder_dim, int attn_dim, int loc_channels, int loc_kernel,
    // Weight tensors (CPU-side float data)
    const float* W_query,    // (attn_dim, decoder_dim)
    const float* W_loc_conv, // (loc_channels, 2, loc_kernel)
    const float* W_loc_fc,   // (attn_dim, loc_channels)
    const float* W_v,        // (1, attn_dim)
    const float* b_v,        // (1,)
    // Attention window
    int prev_attn_idx, int window_size,
    // Outputs
    float* context_out, // (hidden,)
    float* weights_out, // (T_enc,)
    int* new_attn_idx_out) {
    int pad = loc_kernel / 2;

    // 1. Location features: Conv1d(2, loc_channels, loc_kernel) on stacked [weights, cum]
    std::vector<float> loc_feat(loc_channels * T_enc, 0.0f);
    for (int c_out = 0; c_out < loc_channels; c_out++) {
        for (int t = 0; t < T_enc; t++) {
            float sum = 0.0f;
            for (int k = 0; k < loc_kernel; k++) {
                int t_in = t + k - pad;
                if (t_in >= 0 && t_in < T_enc) {
                    // channel 0 = attn_weights_prev, channel 1 = attn_cum
                    sum += W_loc_conv[(c_out * 2 + 0) * loc_kernel + k] * attn_weights_prev[t_in];
                    sum += W_loc_conv[(c_out * 2 + 1) * loc_kernel + k] * attn_cum[t_in];
                }
            }
            loc_feat[c_out * T_enc + t] = sum;
        }
    }

    // 2. W_loc_fc @ loc_feat: (attn_dim, loc_channels) @ (loc_channels, T_enc) -> (attn_dim, T_enc)
    std::vector<float> loc_proj(attn_dim * T_enc, 0.0f);
    for (int a = 0; a < attn_dim; a++) {
        for (int t = 0; t < T_enc; t++) {
            float sum = 0.0f;
            for (int c = 0; c < loc_channels; c++) {
                sum += W_loc_fc[a * loc_channels + c] * loc_feat[c * T_enc + t];
            }
            loc_proj[a * T_enc + t] = sum;
        }
    }

    // 3. W_query @ query: (attn_dim, decoder_dim) @ (decoder_dim,) -> (attn_dim,)
    std::vector<float> q_proj(attn_dim, 0.0f);
    for (int a = 0; a < attn_dim; a++) {
        float sum = 0.0f;
        for (int d = 0; d < decoder_dim; d++) {
            sum += W_query[a * decoder_dim + d] * query[d];
        }
        q_proj[a] = sum;
    }

    // 4. energies = V * tanh(q_proj + proc_memory + loc_proj)
    std::vector<float> energies(T_enc);
    float bias_v = b_v ? b_v[0] : 0.0f;
    for (int t = 0; t < T_enc; t++) {
        float sum = bias_v;
        for (int a = 0; a < attn_dim; a++) {
            float act = tanhf(q_proj[a] + proc_memory[a * T_enc + t] + loc_proj[a * T_enc + t]);
            sum += W_v[a] * act;
        }
        energies[t] = sum;
    }

    // 5. Apply attention window mask
    if (window_size > 0) {
        int left = std::max(0, prev_attn_idx - 1);
        int right = std::min(T_enc - 1, prev_attn_idx + window_size);
        for (int t = 0; t < T_enc; t++) {
            if (t < left || t > right) {
                energies[t] = -1e4f;
            }
        }
    }

    // 6. Softmax
    float max_e = *std::max_element(energies.begin(), energies.end());
    float sum_exp = 0.0f;
    for (int t = 0; t < T_enc; t++) {
        energies[t] = expf(energies[t] - max_e);
        sum_exp += energies[t];
    }
    for (int t = 0; t < T_enc; t++) {
        weights_out[t] = energies[t] / sum_exp;
    }

    // 7. Context = weights @ memory^T: (1, T_enc) @ (T_enc, hidden) -> (hidden,)
    for (int h = 0; h < hidden; h++) {
        float sum = 0.0f;
        for (int t = 0; t < T_enc; t++) {
            sum += weights_out[t] * memory[h * T_enc + t]; // memory is (hidden, T_enc) col-major as read from ggml
        }
        context_out[h] = sum;
    }

    // 8. Update attention index (argmax, clamped to be monotonic)
    int max_idx = prev_attn_idx;
    float max_w = weights_out[prev_attn_idx < T_enc ? prev_attn_idx : 0];
    for (int t = prev_attn_idx; t < T_enc; t++) {
        if (weights_out[t] > max_w) {
            max_w = weights_out[t];
            max_idx = t;
        }
    }
    *new_attn_idx_out = max_idx;
}

// ---------------------------------------------------------------------------
// Prenet (CPU-side for AR loop)
// 2x (Linear + ReLU), dropout=0 at inference
// ---------------------------------------------------------------------------

static void prenet_cpu(const float* input, int in_dim, const float* w0, const float* b0, int out0, const float* w1,
                       const float* b1, int out1, float* output) {
    // Layer 0: Linear(in_dim -> out0) + ReLU
    std::vector<float> h(out0);
    for (int i = 0; i < out0; i++) {
        float sum = b0[i];
        for (int j = 0; j < in_dim; j++) {
            sum += w0[i * in_dim + j] * input[j];
        }
        h[i] = sum > 0.0f ? sum : 0.0f; // ReLU
    }

    // Layer 1: Linear(out0 -> out1) + ReLU
    for (int i = 0; i < out1; i++) {
        float sum = b1[i];
        for (int j = 0; j < out0; j++) {
            sum += w1[i * out0 + j] * h[j];
        }
        output[i] = sum > 0.0f ? sum : 0.0f; // ReLU
    }
}

// ---------------------------------------------------------------------------
// Read weight tensor data to CPU float vector
// ---------------------------------------------------------------------------

static std::vector<float> read_tensor_f32_raw(ggml_tensor* t) {
    if (!t)
        return {};
    int n = (int)ggml_nelements(t);
    std::vector<float> data(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (int i = 0; i < n; i++)
            data[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        // Fallback: dequantize
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, ggml_nbytes(t));
        const auto* traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            traits->to_float(raw.data(), data.data(), n);
        }
    }
    return data;
}

// Read a 2D weight tensor and return in PyTorch row-major (out, in) layout.
// ggml stores 2D tensors as ne[0]-contiguous (column-major from PyTorch's
// perspective): a PyTorch (out, in) weight becomes ggml ne=(out, in) with
// ne[0]=out as the fast axis. But PyTorch row-major stores data as
// data[row * cols + col] = data[out_idx * in_dim + in_idx], while ggml
// stores data[col * rows + row] = data[in_idx * out_dim + out_idx].
// So we need to transpose.
//
// NOTE: The converter writes numpy (out, in) row-major arrays. The gguf
// library stores them as-is in the data section. The gguf reader then
// sets ne[0]=out (last numpy axis), ne[1]=in. The memory layout is
// actually *already* row-major from numpy's perspective — ne[0] is the
// fast axis which corresponds to numpy's last axis. So for a Linear weight
// that numpy stored as (out, in), the flat data is ALREADY in
// data[out_idx * in_dim + in_idx] order.
//
// TL;DR: ggml raw data for 2D tensors from numpy/GGUF is already row-major.
// No transpose needed — just read flat.
static std::vector<float> read_tensor_f32(ggml_tensor* t) {
    return read_tensor_f32_raw(t);
}

// ---------------------------------------------------------------------------
// Autoregressive decoder
// ---------------------------------------------------------------------------

struct decoder_result {
    std::vector<float> mel; // (T_mel * n_mels), row-major
    int T_mel;
};

static decoder_result run_decoder(bananamind_tts_context* ctx, const std::vector<float>& encoder_out, int T_enc) {
    const auto& hp = ctx->hp;
    const int H = hp.hidden_size;      // 256
    const int D = hp.decoder_dim;      // 512
    const int A = hp.attention_dim;    // 128
    const int M = hp.n_mels;           // 80
    const int R = hp.reduction_factor; // 4

    // Pre-read all weight tensors to CPU for the AR loop
    auto prenet_w0 = read_tensor_f32(ctx->W("prenet.0.weight"));
    auto prenet_b0 = read_tensor_f32(ctx->W("prenet.0.bias"));
    auto prenet_w1 = read_tensor_f32(ctx->W("prenet.1.weight"));
    auto prenet_b1 = read_tensor_f32(ctx->W("prenet.1.bias"));

    auto attn_rnn_wih = read_tensor_f32(ctx->W("attn_rnn.weight_ih"));
    auto attn_rnn_bih = read_tensor_f32(ctx->W("attn_rnn.bias_ih"));
    auto attn_rnn_whh = read_tensor_f32(ctx->W("attn_rnn.weight_hh"));
    auto attn_rnn_bhh = read_tensor_f32(ctx->W("attn_rnn.bias_hh"));

    auto dec_rnn_wih = read_tensor_f32(ctx->W("dec_rnn.weight_ih"));
    auto dec_rnn_bih = read_tensor_f32(ctx->W("dec_rnn.bias_ih"));
    auto dec_rnn_whh = read_tensor_f32(ctx->W("dec_rnn.weight_hh"));
    auto dec_rnn_bhh = read_tensor_f32(ctx->W("dec_rnn.bias_hh"));

    auto mel_proj_w = read_tensor_f32(ctx->W("mel_proj.weight"));
    auto mel_proj_b = read_tensor_f32(ctx->W("mel_proj.bias"));
    auto stop_proj_w = read_tensor_f32(ctx->W("stop_proj.weight"));
    auto stop_proj_b = read_tensor_f32(ctx->W("stop_proj.bias"));

    auto attn_query_w = read_tensor_f32(ctx->W("attn.query.weight"));
    auto attn_memory_w = read_tensor_f32(ctx->W("attn.memory.weight"));
    auto attn_loc_conv_w = read_tensor_f32(ctx->W("attn.loc_conv.weight"));
    auto attn_loc_fc_w = read_tensor_f32(ctx->W("attn.loc_fc.weight"));
    auto attn_v_w = read_tensor_f32(ctx->W("attn.v.weight"));
    auto attn_v_b = read_tensor_f32(ctx->W("attn.v.bias"));

    // Encoder output is (H, T) in ggml, read as ne[0]-contiguous flat data.
    // Flat layout: data[t * H + h] — i.e. (T, H) row-major.
    // We need it as (H, T) for the matmuls. Transpose to (H, T) row-major.
    std::vector<float> enc_ht(H * T_enc);
    for (int t = 0; t < T_enc; t++) {
        for (int h = 0; h < H; h++) {
            enc_ht[h * T_enc + t] = encoder_out[t * H + h];
        }
    }

    // Pre-compute processed_memory = W_memory @ memory: (A, H) @ (H, T_enc) -> (A, T_enc)
    std::vector<float> proc_memory(A * T_enc, 0.0f);
    for (int a = 0; a < A; a++) {
        for (int t = 0; t < T_enc; t++) {
            float sum = 0.0f;
            for (int h = 0; h < H; h++) {
                sum += attn_memory_w[a * H + h] * enc_ht[h * T_enc + t];
            }
            proc_memory[a * T_enc + t] = sum;
        }
    }

    // Initialize decoder states
    std::vector<float> attn_hidden(D, 0.0f);
    std::vector<float> dec_hidden(D, 0.0f);
    std::vector<float> context(H, 0.0f);
    std::vector<float> attn_weights(T_enc, 0.0f);
    attn_weights[0] = 1.0f;
    std::vector<float> attn_cum(T_enc, 0.0f);
    attn_cum[0] = 1.0f;
    int prev_attn_idx = 0;

    std::vector<float> decoder_input(M, 0.0f); // last mel frame (zeros initially)

    int max_steps = std::max(1, hp.max_decoder_steps / R);
    int min_steps = std::max(1, (int)(T_enc * 3) / R);

    std::vector<float> all_mel;
    all_mel.reserve(max_steps * R * M);

    int prenet_out_dim = hp.prenet_size_1; // 128
    int proj_dim = D + H;                  // 768

    if (debug_enabled()) {
        fprintf(stderr, "bananamind_tts: decoder: T_enc=%d max_steps=%d min_steps=%d\n", T_enc, max_steps, min_steps);
    }

    for (int step = 0; step < max_steps; step++) {
        // 1. Prenet
        std::vector<float> prenet_out(prenet_out_dim);
        prenet_cpu(decoder_input.data(), M, prenet_w0.data(), prenet_b0.data(), hp.prenet_size_0, prenet_w1.data(),
                   prenet_b1.data(), prenet_out_dim, prenet_out.data());

        // 2. Attention GRU: input = cat(prenet_out, context) = (128+256=384)
        int attn_rnn_input_dim = prenet_out_dim + H;
        std::vector<float> attn_rnn_input(attn_rnn_input_dim);
        std::copy(prenet_out.begin(), prenet_out.end(), attn_rnn_input.begin());
        std::copy(context.begin(), context.end(), attn_rnn_input.begin() + prenet_out_dim);

        gru_cell_cpu(attn_rnn_wih.data(), attn_rnn_bih.data(), attn_rnn_whh.data(), attn_rnn_bhh.data(),
                     attn_rnn_input.data(), attn_rnn_input_dim, attn_hidden.data(), D);

        // 3. Location-sensitive attention
        // enc_ht is (H, T_enc) row-major — matches location_attention_cpu's expectation
        location_attention_cpu(attn_hidden.data(), enc_ht.data(), proc_memory.data(), attn_weights.data(),
                               attn_cum.data(), T_enc, H, D, A, hp.location_channels, hp.location_kernel_size,
                               attn_query_w.data(), attn_loc_conv_w.data(), attn_loc_fc_w.data(), attn_v_w.data(),
                               attn_v_b.data(), prev_attn_idx, hp.attention_window, context.data(), attn_weights.data(),
                               &prev_attn_idx);

        // Update cumulative attention
        for (int t = 0; t < T_enc; t++) {
            attn_cum[t] += attn_weights[t];
        }

        // 4. Decoder GRU: input = cat(attn_hidden, context) = (512+256=768)
        int dec_rnn_input_dim = D + H;
        std::vector<float> dec_rnn_input(dec_rnn_input_dim);
        std::copy(attn_hidden.begin(), attn_hidden.end(), dec_rnn_input.begin());
        std::copy(context.begin(), context.end(), dec_rnn_input.begin() + D);

        gru_cell_cpu(dec_rnn_wih.data(), dec_rnn_bih.data(), dec_rnn_whh.data(), dec_rnn_bhh.data(),
                     dec_rnn_input.data(), dec_rnn_input_dim, dec_hidden.data(), D);

        // 5. Projection input = cat(dec_hidden, context) = (768,)
        std::vector<float> proj_input(proj_dim);
        std::copy(dec_hidden.begin(), dec_hidden.end(), proj_input.begin());
        std::copy(context.begin(), context.end(), proj_input.begin() + D);

        // Mel projection: (M*R, proj_dim) @ (proj_dim,) -> (M*R,)
        std::vector<float> mel_frame(M * R);
        for (int i = 0; i < M * R; i++) {
            float sum = mel_proj_b[i];
            for (int j = 0; j < proj_dim; j++) {
                sum += mel_proj_w[i * proj_dim + j] * proj_input[j];
            }
            mel_frame[i] = sum;
        }

        // Stop projection: (R, proj_dim) @ (proj_dim,) -> (R,)
        std::vector<float> stop_logits(R);
        for (int i = 0; i < R; i++) {
            float sum = stop_proj_b[i];
            for (int j = 0; j < proj_dim; j++) {
                sum += stop_proj_w[i * proj_dim + j] * proj_input[j];
            }
            stop_logits[i] = sum;
        }

        // Store mel frames (R frames of M mels each)
        // mel_frame is (M*R,) = R frames of M mels, reshape to (R, M) row-major
        all_mel.insert(all_mel.end(), mel_frame.begin(), mel_frame.end());

        // Update decoder input with last mel frame
        // Last frame = mel_frame[(R-1)*M .. R*M-1]
        std::copy(mel_frame.begin() + (R - 1) * M, mel_frame.begin() + R * M, decoder_input.begin());

        // Debug: print diagnostics for first few steps
        if (debug_enabled() && (step < 3 || step % 50 == 0)) {
            float stop_prob_dbg = 1.0f / (1.0f + expf(-stop_logits[R - 1]));
            float ctx_norm = 0.0f;
            for (int h = 0; h < H; h++)
                ctx_norm += context[h] * context[h];
            ctx_norm = sqrtf(ctx_norm);
            float mel_min_v = mel_frame[0], mel_max_v = mel_frame[0];
            for (int i = 1; i < M * R; i++) {
                mel_min_v = std::min(mel_min_v, mel_frame[i]);
                mel_max_v = std::max(mel_max_v, mel_frame[i]);
            }
            fprintf(stderr,
                    "  step %3d: attn_idx=%d stop=%.3f ctx_norm=%.3f mel=[%.3f,%.3f] "
                    "attn_h[0]=%.4f dec_h[0]=%.4f\n",
                    step, prev_attn_idx, stop_prob_dbg, ctx_norm, mel_min_v, mel_max_v, attn_hidden[0], dec_hidden[0]);
        }

        // Check stop condition (last frame's stop logit)
        if (step >= min_steps) {
            float stop_prob = 1.0f / (1.0f + expf(-stop_logits[R - 1]));
            if (stop_prob > hp.stop_threshold) {
                if (debug_enabled()) {
                    fprintf(stderr, "bananamind_tts: stop at step %d (prob=%.3f)\n", step, stop_prob);
                }
                break;
            }
        }
    }

    int T_mel = (int)(all_mel.size() / M);

    decoder_result res;
    res.mel = std::move(all_mel);
    res.T_mel = T_mel;
    return res;
}

// ---------------------------------------------------------------------------
// Postnet: 5-layer Conv1d + BatchNorm + Tanh (residual refinement)
// ---------------------------------------------------------------------------

static std::vector<float> run_postnet(bananamind_tts_context* ctx, const std::vector<float>& mel, int T_mel) {
    const auto& hp = ctx->hp;
    const int M = hp.n_mels;

    const int max_tensors = 256;
    const size_t mem_size = (size_t)max_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(4096, false);

    ggml_init_params ip = {mem_size, nullptr, true};
    ggml_context* gc = ggml_init(ip);
    if (!gc)
        return mel;

    // Input: (T_mel, M) — ggml conv expects (T, C_in)
    ggml_tensor* x = ggml_new_tensor_2d(gc, GGML_TYPE_F32, T_mel, M);
    ggml_set_name(x, "postnet_in");
    ggml_set_input(x);

    ggml_tensor* h = x;

    struct bn_deferred {
        ggml_tensor* scale_t;
        ggml_tensor* shift_t;
        std::vector<float> scale_data;
        std::vector<float> shift_data;
    };
    std::vector<bn_deferred> bn_inputs;

    for (int i = 0; i < hp.postnet_layers; i++) {
        ggml_tensor* conv_w = ctx->W("postnet.conv." + std::to_string(i) + ".weight");
        if (!conv_w)
            continue;

        // Conv1d with padding = (5-1)/2 = 2
        h = ggml_conv_1d(gc, conv_w, h, 1, 2, 1);
        ggml_tensor* conv_b = ctx->W("postnet.conv." + std::to_string(i) + ".bias");
        if (conv_b) {
            ggml_tensor* b = ggml_reshape_2d(gc, conv_b, 1, (int)conv_b->ne[0]);
            h = ggml_add(gc, h, b);
        }

        // BatchNorm (deferred)
        ggml_tensor* bn_mean = ctx->W("postnet.bn." + std::to_string(i) + ".mean");
        ggml_tensor* bn_var = ctx->W("postnet.bn." + std::to_string(i) + ".var");
        ggml_tensor* bn_w = ctx->W("postnet.bn." + std::to_string(i) + ".weight");
        ggml_tensor* bn_b = ctx->W("postnet.bn." + std::to_string(i) + ".bias");

        if (bn_mean && bn_var && bn_w && bn_b) {
            int C = (int)bn_mean->ne[0];
            std::vector<float> mean_v(C), var_v(C), w_v(C), b_v(C);
            ggml_backend_tensor_get(bn_mean, mean_v.data(), 0, C * sizeof(float));
            ggml_backend_tensor_get(bn_var, var_v.data(), 0, C * sizeof(float));
            ggml_backend_tensor_get(bn_w, w_v.data(), 0, C * sizeof(float));
            ggml_backend_tensor_get(bn_b, b_v.data(), 0, C * sizeof(float));

            std::vector<float> scale(C), shift(C);
            for (int c = 0; c < C; c++) {
                float inv_std = 1.0f / sqrtf(var_v[c] + 1e-5f);
                scale[c] = w_v[c] * inv_std;
                shift[c] = b_v[c] - mean_v[c] * scale[c];
            }

            ggml_tensor* s_t = ggml_new_tensor_2d(gc, GGML_TYPE_F32, 1, C);
            ggml_set_name(s_t, (std::string("pn_scale_") + std::to_string(i)).c_str());
            ggml_set_input(s_t);
            ggml_tensor* sh_t = ggml_new_tensor_2d(gc, GGML_TYPE_F32, 1, C);
            ggml_set_name(sh_t, (std::string("pn_shift_") + std::to_string(i)).c_str());
            ggml_set_input(sh_t);

            h = ggml_mul(gc, h, s_t);
            h = ggml_add(gc, h, sh_t);

            bn_inputs.push_back({s_t, sh_t, std::move(scale), std::move(shift)});
        }

        // Tanh on all except last layer
        if (i < hp.postnet_layers - 1) {
            h = ggml_tanh(gc, h);
        }
    }

    // Residual
    h = ggml_add(gc, h, x);
    ggml_set_name(h, "postnet_out");
    ggml_set_output(h);

    ggml_cgraph* gf = ggml_new_graph_custom(gc, 4096, false);
    ggml_build_forward_expand(gf, h);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "bananamind_tts: postnet graph alloc failed\n");
        ggml_free(gc);
        return mel;
    }

    // Set inputs — transpose from row-major (T, M) to ggml ne[0]-contiguous
    ggml_tensor* in_t = ggml_graph_get_tensor(gf, "postnet_in");
    {
        std::vector<float> mel_tr(T_mel * M);
        for (int t = 0; t < T_mel; t++)
            for (int m = 0; m < M; m++)
                mel_tr[m * T_mel + t] = mel[t * M + m];
        ggml_backend_tensor_set(in_t, mel_tr.data(), 0, T_mel * M * sizeof(float));
    }

    for (auto& bn : bn_inputs) {
        ggml_backend_tensor_set(bn.scale_t, bn.scale_data.data(), 0, bn.scale_data.size() * sizeof(float));
        ggml_backend_tensor_set(bn.shift_t, bn.shift_data.data(), 0, bn.shift_data.size() * sizeof(float));
    }

    ggml_backend_sched_graph_compute(ctx->sched, gf);

    ggml_tensor* out = ggml_graph_get_tensor(gf, "postnet_out");
    // Read ggml ne[0]-contiguous data and transpose back to row-major (T, M)
    std::vector<float> raw(T_mel * M);
    ggml_backend_tensor_get(out, raw.data(), 0, T_mel * M * sizeof(float));
    std::vector<float> result(T_mel * M);
    for (int t = 0; t < T_mel; t++)
        for (int m = 0; m < M; m++)
            result[t * M + m] = raw[m * T_mel + t];

    ggml_free(gc);
    return result;
}

// ---------------------------------------------------------------------------
// Mel denormalization
// ---------------------------------------------------------------------------

static void denormalize_mel(std::vector<float>& mel, const bananamind_hparams& hp) {
    if (!hp.mel_normalized)
        return;
    float s = std::max(hp.mel_std, 1e-5f);
    for (float& v : mel) {
        v = v * s + hp.mel_mean;
        v = std::max(hp.mel_min, std::min(hp.mel_max, v));
    }
}

// ---------------------------------------------------------------------------
// HiFi-GAN vocoder
// ---------------------------------------------------------------------------

static std::vector<float> run_vocoder(bananamind_tts_context* ctx, const std::vector<float>& mel, int T_mel) {
    const auto& hp = ctx->hp;
    const int M = hp.n_mels;

    // Estimate vocoder graph size
    const int max_tensors = 1024;
    const size_t mem_size = (size_t)max_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(16384, false);

    ggml_init_params ip = {mem_size, nullptr, true};
    ggml_context* gc = ggml_init(ip);
    if (!gc)
        return {};

    // Input mel: ggml (T_mel, M) means ne[0]=T_mel, ne[1]=M.
    // Data stored ne[0]-contiguous: element at (t, m) = data[m * T_mel + t].
    // Our mel array is row-major (T, M): data[t * M + m].
    // We need to transpose the data to match ggml's layout.
    ggml_tensor* mel_t = ggml_new_tensor_2d(gc, GGML_TYPE_F32, T_mel, M);
    ggml_set_name(mel_t, "voc_mel_in");
    ggml_set_input(mel_t);

    // core_hifigan::forward expects (T, C) time-first which is what we have.
    // ups_w_perm selects the decomposed mul_mat + col2im_1d transpose-conv
    // (empty under CRISPASR_BANANAMIND_VOC_LEGACY=1 → old ggml_conv_transpose_1d
    // branch); voc_fc is the FASTCONV F32 kernel cache (disabled likewise).
    ggml_tensor* wav = core_hifigan::forward(gc, mel_t, ctx->tensors, "voc", hp.voc_hp, ctx->ups_w_perm, &ctx->voc_fc);
    ggml_set_name(wav, "voc_wav_out");
    ggml_set_output(wav);

    ggml_cgraph* gf = ggml_new_graph_custom(gc, 16384, false);
    ggml_build_forward_expand(gf, wav);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "bananamind_tts: vocoder graph alloc failed\n");
        ggml_free(gc);
        return {};
    }

    // Transpose mel from row-major (T, M) -> ggml ne[0]-contiguous (T, M)
    // ggml layout: data[m * T_mel + t], our layout: data[t * M + m]
    std::vector<float> mel_transposed(T_mel * M);
    for (int t = 0; t < T_mel; t++) {
        for (int m = 0; m < M; m++) {
            mel_transposed[m * T_mel + t] = mel[t * M + m];
        }
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "voc_mel_in"), mel_transposed.data(), 0,
                            T_mel * M * sizeof(float));

    ggml_backend_sched_graph_compute(ctx->sched, gf);

    ggml_tensor* wav_out = ggml_graph_get_tensor(gf, "voc_wav_out");
    int n_samples = (int)ggml_nelements(wav_out);
    std::vector<float> pcm(n_samples);
    ggml_backend_tensor_get(wav_out, pcm.data(), 0, n_samples * sizeof(float));

    ggml_free(gc);

    // Trim to expected length
    int expected = std::max(1, (T_mel - 1) * (int)hp.voc_hp.upsample_rates[0]);
    // Compute total upsample rate
    int total_rate = 1;
    for (int r : hp.voc_hp.upsample_rates)
        total_rate *= r;
    expected = T_mel * total_rate;
    if ((int)pcm.size() > expected) {
        pcm.resize(expected);
    }

    return pcm;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

struct bananamind_tts_params bananamind_tts_default_params(void) {
    bananamind_tts_params p;
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    return p;
}

struct bananamind_tts_context* bananamind_tts_init_from_file(const char* path_model,
                                                             struct bananamind_tts_params params) {
    auto* ctx = new bananamind_tts_context();
    ctx->n_threads = params.n_threads;
    ctx->verbosity = params.verbosity;

    ctx->backend_cpu = core_cpu_backend::init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "bananamind_tts: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    core_cpu_backend::set_n_threads(ctx->backend_cpu, params.n_threads);

    // GPU when one is available and asked for; CPU otherwise. The scheduler
    // below keeps the CPU backend in the list so any op without a device
    // kernel still runs.
    //
    // On a CPU-only build crispasr_init_gpu_backend() falls through to
    // ggml_backend_init_best(), which hands back a SECOND, DISTINCT CPU
    // backend. Keeping it would put two CPU backends in the scheduler and
    // silently drop params.n_threads (set_n_threads was applied to
    // backend_cpu, not to that one), so detect it and collapse back.
    ctx->backend = params.use_gpu ? crispasr_init_gpu_backend() : ctx->backend_cpu;
    if (ctx->backend && ctx->backend != ctx->backend_cpu && core_cpu_backend::is_cpu(ctx->backend)) {
        ggml_backend_free(ctx->backend);
        ctx->backend = ctx->backend_cpu;
    }
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;
    if (params.verbosity >= 1 && ctx->backend != ctx->backend_cpu)
        fprintf(stderr, "bananamind_tts: using GPU backend: %s\n", ggml_backend_name(ctx->backend));

    if (!load_model(ctx, path_model)) {
        if (ctx->backend != ctx->backend_cpu)
            ggml_backend_free(ctx->backend);
        ggml_backend_free(ctx->backend_cpu);
        delete ctx;
        return nullptr;
    }

    // ── Backend scheduler (all three graphs run through it) ──
    {
        ggml_backend_t backends[2];
        int n_be = 0;
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;

        // 32768: headroom over the largest of the three graphs (the vocoder,
        // whose own cgraph is sized 16384).
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be,
                                            /*graph_size=*/32768,
                                            /*parallel=*/false, /*op_offload=*/false);
        if (!ctx->sched) {
            fprintf(stderr, "bananamind_tts: failed to create backend scheduler\n");
            bananamind_tts_free(ctx);
            return nullptr;
        }
    }

    // ── HiFi-GAN vocoder priming (skipped entirely on the legacy arm) ──
    if (!bananamind_voc_legacy()) {
        // Pre-permute the four ConvTranspose1d upsample kernels for the
        // decomposed col2im_1d path. Mirrors fastpitch_tts.cpp / speecht5_tts.cpp.
        const int n = ctx->hp.voc_hp.num_upsamples();
        std::vector<ggml_tensor*> srcs(n);
        std::vector<ggml_tensor**> dsts(n);
        ctx->ups_w_perm.resize(n, nullptr);
        for (int i = 0; i < n; i++) {
            srcs[i] = ctx->W("voc.ups." + std::to_string(i) + ".weight");
            dsts[i] = &ctx->ups_w_perm[i];
        }
        core_convt::permute_convt1d_weights_batch(srcs.data(), dsts.data(), n, ctx->backend, &ctx->ctx_perm,
                                                  &ctx->buf_perm);

        // FASTCONV: bake one F32 copy of each F16 vocoder conv kernel so the
        // fork's per-graph F16→F32 cast becomes a no-op. Every published
        // bananamind build keeps voc.* in F32 (even the q8_0 one), so this
        // bakes 0 kernels there and only pays off for a future F16 build.
        const char* e = crispasr_env::get("CRISPASR_BANANAMIND_FASTCONV");
        const bool on = !e || (e[0] != '0');
        auto convs = core_hifigan::collect_fastconv_kernels(ctx->tensors, "voc", ctx->hp.voc_hp);
        ctx->voc_fc.bake(ctx->backend, convs, on);
        if (crispasr_env::get("CRISPASR_BANANAMIND_FASTCONV_DEBUG")) {
            fprintf(stderr, "bananamind_tts: FASTCONV %s — baked %zu F32 kernels from %zu voc convs\n",
                    ctx->voc_fc.enabled ? "ON" : "OFF", ctx->voc_fc.f32.size(), convs.size());
        }
    } else if (params.verbosity >= 1) {
        fprintf(stderr, "bananamind_tts: CRISPASR_BANANAMIND_VOC_LEGACY=1 — legacy vocoder path\n");
    }

    return ctx;
}

void bananamind_tts_free(struct bananamind_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    ctx->voc_fc.free();
    if (ctx->buf_perm)
        ggml_backend_buffer_free(ctx->buf_perm); // permuted ups kernels
    if (ctx->ctx_perm)
        ggml_free(ctx->ctx_perm);
    if (ctx->w_buf)
        core_gguf::release_weight_buffer(ctx->w_buf);
    if (ctx->w_ctx)
        ggml_free(ctx->w_ctx);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

int bananamind_tts_synthesize(struct bananamind_tts_context* ctx, const char* text, float** pcm_out,
                              int* sample_rate_out) {
    if (!ctx || !text || !pcm_out || !sample_rate_out)
        return 0;

    *pcm_out = nullptr;
    *sample_rate_out = ctx->hp.sample_rate;

    // Tokenize
    std::vector<int> token_ids = ctx->tokenizer.encode(std::string(text));
    if (token_ids.size() <= 2) { // only BOS+EOS
        fprintf(stderr, "bananamind_tts: empty text after tokenization\n");
        return 0;
    }

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "bananamind_tts: %zu tokens, text: \"%s\"\n", token_ids.size(), text);
    }

    // Encoder
    std::vector<float> enc_out;
    {
        bananamind_tts_bench_stage _b("encoder");
        enc_out = run_encoder(ctx, token_ids);
    }
    if (enc_out.empty()) {
        fprintf(stderr, "bananamind_tts: encoder failed\n");
        return 0;
    }
    int T_enc = (int)token_ids.size();

    if (debug_enabled()) {
        fprintf(stderr, "bananamind_tts: encoder output: %d x %d\n", ctx->hp.hidden_size, T_enc);
    }

    // Decoder (autoregressive)
    decoder_result dec;
    {
        bananamind_tts_bench_stage _b("ar_decoder");
        dec = run_decoder(ctx, enc_out, T_enc);
    }
    if (dec.mel.empty()) {
        fprintf(stderr, "bananamind_tts: decoder failed\n");
        return 0;
    }

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "bananamind_tts: decoder produced %d mel frames\n", dec.T_mel);
    }

    // Postnet refinement
    std::vector<float> mel_refined;
    {
        bananamind_tts_bench_stage _b("postnet");
        mel_refined = run_postnet(ctx, dec.mel, dec.T_mel);
    }

    // Denormalize mel
    denormalize_mel(mel_refined, ctx->hp);

    if (debug_enabled()) {
        int M = ctx->hp.n_mels;
        float mel_min_v = mel_refined[0], mel_max_v = mel_refined[0], mel_sum = 0;
        for (float v : mel_refined) {
            mel_min_v = std::min(mel_min_v, v);
            mel_max_v = std::max(mel_max_v, v);
            mel_sum += v;
        }
        fprintf(stderr, "bananamind_tts: mel denorm: min=%.4f max=%.4f mean=%.4f\n", mel_min_v, mel_max_v,
                mel_sum / (float)mel_refined.size());
        // mel is (T_mel * M) flat, row-major (T_mel, M). Print mel[0,:8]
        fprintf(stderr, "bananamind_tts: mel[0,:8] =");
        for (int h = 0; h < 8 && h < M; h++)
            fprintf(stderr, " %.4f", mel_refined[0 * M + h]);
        fprintf(stderr, "\n");
    }

    // Vocoder
    std::vector<float> pcm;
    {
        bananamind_tts_bench_stage _b("vocoder");
        pcm = run_vocoder(ctx, mel_refined, dec.T_mel);
    }
    if (pcm.empty()) {
        fprintf(stderr, "bananamind_tts: vocoder failed\n");
        return 0;
    }

    // Normalize waveform to [-0.95, 0.95] (matches Python tts() normalize_wav=True)
    {
        float peak = 0.0f;
        for (float v : pcm)
            peak = std::max(peak, std::abs(v));
        if (peak > 0.0f) {
            float scale = 0.95f / peak;
            for (float& v : pcm)
                v *= scale;
        }
    }

    if (ctx->verbosity >= 1) {
        float dur = (float)pcm.size() / ctx->hp.sample_rate;
        fprintf(stderr, "bananamind_tts: synthesized %.2fs (%d samples @ %d Hz)\n", dur, (int)pcm.size(),
                ctx->hp.sample_rate);
    }

    // Output
    int n = (int)pcm.size();
    float* out = (float*)malloc(n * sizeof(float));
    if (!out)
        return 0;
    memcpy(out, pcm.data(), n * sizeof(float));
    *pcm_out = out;
    return n;
}

int bananamind_tts_sample_rate(const struct bananamind_tts_context* ctx) {
    return ctx ? ctx->hp.sample_rate : 22050;
}
