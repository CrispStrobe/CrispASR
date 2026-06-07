// piper_tts.cpp — native ggml runtime for rhasspy/piper VITS models.
//
// Architecture (all F32 on CPU, F16/Q4_K for storage):
//   1. TextEncoder: embedding + 6-layer relative-position transformer
//      → mean + log_var (192-d each, split from 384-d proj output)
//   2. StochasticDurationPredictor: DDSConv conditioning + 3 ConvFlow
//      layers with rational quadratic spline transforms → log-durations
//   3. ResidualCouplingFlow (inverse): 4 affine coupling blocks with
//      WaveNet conditioning (4 dilated layers per block) → latent z
//   4. HiFi-GAN decoder: conv_pre + 3 upsample stages + 9 resblocks
//      + conv_post + tanh → 22.05 kHz mono PCM
//
// The implementation uses per-module ggml sub-graphs. Each module builds
// its own graph, computes, and returns CPU-side results. This keeps the
// graph size small and avoids the complexity of stitching data-dependent
// control flow (duration predictor splines) into one monolithic graph.

#include "piper_tts.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "core/conv.h"
#include "core/gguf_loader.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <vector>

// espeak-ng phonemizer (shared with kokoro.cpp)
#ifdef CRISPASR_HAVE_ESPEAK_NG
// Match kokoro.cpp: the CMake find_path locates `espeak-ng/speak_lib.h`,
// so the resolved include dir is the PARENT — the header must be included
// with its `espeak-ng/` prefix. A bare <speak_lib.h> fails on Homebrew
// macOS (header under .../include/espeak-ng/) even though it happens to
// resolve on some Linux layouts.
#include <espeak-ng/speak_lib.h>
#endif

// ── JSON-lite parser for phoneme_id_map ────────────────────────────
// The phoneme_id_map is stored as a JSON string in GGUF metadata.
// Format: {"_": [0], "^": [1], "$": [2], " ": [3], ...}
// We parse it into a map from UTF-8 char → vector of phoneme IDs.

struct phoneme_map_entry {
    std::string phoneme;  // UTF-8 phoneme character(s)
    std::vector<int> ids; // mapped IDs (usually 1 element)
};

static bool parse_phoneme_id_map(const std::string& json, std::vector<phoneme_map_entry>& out) {
    out.clear();
    // Very simple JSON object parser for {"key": [int, ...], ...}
    size_t pos = json.find('{');
    if (pos == std::string::npos)
        return false;
    pos++;

    while (pos < json.size()) {
        // Skip whitespace
        while (pos < json.size() &&
               (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t' || json[pos] == ','))
            pos++;
        if (pos >= json.size() || json[pos] == '}')
            break;

        // Parse key string
        if (json[pos] != '"')
            return false;
        pos++;
        std::string key;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                pos++;
                if (json[pos] == '"')
                    key += '"';
                else if (json[pos] == '\\')
                    key += '\\';
                else if (json[pos] == 'n')
                    key += '\n';
                else
                    key += json[pos];
            } else {
                key += json[pos];
            }
            pos++;
        }
        if (pos >= json.size())
            return false;
        pos++; // skip closing "

        // Skip : and whitespace
        while (pos < json.size() && (json[pos] == ':' || json[pos] == ' '))
            pos++;

        // Parse array of ints
        if (pos >= json.size() || json[pos] != '[')
            return false;
        pos++;
        std::vector<int> ids;
        while (pos < json.size() && json[pos] != ']') {
            while (pos < json.size() && (json[pos] == ' ' || json[pos] == ','))
                pos++;
            if (pos >= json.size() || json[pos] == ']')
                break;
            int val = 0;
            bool neg = false;
            if (json[pos] == '-') {
                neg = true;
                pos++;
            }
            while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
                val = val * 10 + (json[pos] - '0');
                pos++;
            }
            ids.push_back(neg ? -val : val);
        }
        if (pos < json.size())
            pos++; // skip ]

        out.push_back({key, ids});
    }
    return true;
}

// ── Phoneme encoding ───────────────────────────────────────────────

static std::vector<int64_t> encode_phonemes(const std::string& ipa_text, const std::vector<phoneme_map_entry>& pmap) {
    // Build a lookup from phoneme string to id list
    // Try longest match first (some phonemes are multi-byte UTF-8)
    std::vector<int64_t> ids;
    ids.push_back(1); // BOS = ^

    size_t pos = 0;
    while (pos < ipa_text.size()) {
        // Try to find the longest matching phoneme
        bool found = false;
        for (int len = 4; len >= 1; len--) {
            if (pos + len > ipa_text.size())
                continue;
            std::string candidate = ipa_text.substr(pos, len);
            for (const auto& entry : pmap) {
                if (entry.phoneme == candidate) {
                    for (int id : entry.ids) {
                        ids.push_back(id);
                    }
                    ids.push_back(0); // PAD between phonemes (intersperse)
                    pos += len;
                    found = true;
                    break;
                }
            }
            if (found)
                break;
        }
        if (!found) {
            // Skip unknown character
            // Advance by one UTF-8 character
            unsigned char c = (unsigned char)ipa_text[pos];
            if (c < 0x80)
                pos += 1;
            else if (c < 0xE0)
                pos += 2;
            else if (c < 0xF0)
                pos += 3;
            else
                pos += 4;
            if (pos > ipa_text.size())
                pos = ipa_text.size();
        }
    }
    ids.push_back(2); // EOS = $
    return ids;
}

// ── espeak-ng phonemizer ───────────────────────────────────────────

static std::mutex g_piper_espeak_mu;
static bool g_piper_espeak_inited = false;
static bool g_piper_espeak_init_failed = false;

static bool phonemize_espeak(const std::string& voice, const std::string& text, std::string& out) {
#ifdef CRISPASR_HAVE_ESPEAK_NG
    std::lock_guard<std::mutex> g(g_piper_espeak_mu);
    if (g_piper_espeak_init_failed)
        return false;
    if (!g_piper_espeak_inited) {
        const char* data_path = getenv("CRISPASR_ESPEAK_DATA_PATH");
#ifdef CRISPASR_ESPEAK_DATA_PATH
        // Bundled build: use the compiled-in data path as fallback
        if (!data_path) data_path = CRISPASR_ESPEAK_DATA_PATH;
#endif
        int sr = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, data_path,
                                   espeakINITIALIZE_PHONEME_IPA | espeakINITIALIZE_DONT_EXIT);
        if (sr < 0) {
            fprintf(stderr, "piper_tts: espeak_Initialize failed (rc=%d, path=%s)\n",
                    sr, data_path ? data_path : "<default>");
            g_piper_espeak_init_failed = true;
            return false;
        }
        g_piper_espeak_inited = true;
    }
    espeak_SetVoiceByName(voice.c_str());

    out.clear();
    const char* tp = text.c_str();
    while (tp != nullptr && *tp != '\0') {
        const char* phon = espeak_TextToPhonemes((const void**)&tp, espeakCHARS_UTF8, 0x02 /* IPA */);
        if (phon && *phon) {
            if (!out.empty())
                out += ' ';
            out += phon;
        }
    }
    return !out.empty();
#else
    // Fallback: popen to espeak-ng
#ifdef _WIN32
#define piper_popen _popen
#define piper_pclose _pclose
    const char* redir = " 2>NUL";
#else
#define piper_popen popen
#define piper_pclose pclose
    const char* redir = " 2>/dev/null";
#endif
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "espeak-ng -q --ipa=3 -v %s \"%s\"%s", voice.c_str(), text.c_str(), redir);
    FILE* fp = piper_popen(cmd, "r");
    if (!fp)
        return false;
    out.clear();
    char buf[256];
    while (fgets(buf, sizeof(buf), fp)) {
        // Strip trailing newline
        size_t len = strlen(buf);
        while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r'))
            len--;
        if (!out.empty() && len > 0)
            out += ' ';
        out.append(buf, len);
    }
    piper_pclose(fp);
    return !out.empty();
#endif
}

// ── Hparams ────────────────────────────────────────────────────────

struct piper_hparams {
    uint32_t hidden_channels = 192;
    uint32_t inter_channels = 192;
    uint32_t filter_channels = 768;
    uint32_t n_heads = 2;
    uint32_t head_dim = 96;
    uint32_t n_layers_enc = 6;
    uint32_t n_flow_blocks = 4;
    uint32_t n_wn_layers = 4;
    uint32_t wn_kernel_size = 5;
    uint32_t n_upsample_stages = 3;
    uint32_t upsample_initial_channel = 256;
    uint32_t num_symbols = 256;
    uint32_t num_speakers = 1;
    uint32_t sample_rate = 22050;
    uint32_t n_sdp_flows = 3;
    uint32_t sdp_num_bins = 10;
    uint32_t n_sdp_dds_layers = 3;
    uint32_t n_sdp_main_dds_layers = 3;

    float noise_scale = 0.667f;
    float length_scale = 1.0f;
    float noise_w = 0.8f;

    std::string espeak_voice = "en-us";
    std::string phoneme_id_map_json;

    std::vector<uint32_t> upsample_rates;   // e.g. {8, 8, 4}
    std::vector<uint32_t> upsample_kernels; // e.g. {16, 16, 8}
    std::vector<uint32_t> resblock_kernels; // e.g. {3, 5, 7}
};

// ── Weight struct ──────────────────────────────────────────────────

struct piper_enc_layer {
    ggml_tensor* conv_q_w;
    ggml_tensor* conv_q_b;
    ggml_tensor* conv_k_w;
    ggml_tensor* conv_k_b;
    ggml_tensor* conv_v_w;
    ggml_tensor* conv_v_b;
    ggml_tensor* conv_o_w;
    ggml_tensor* conv_o_b;
    ggml_tensor* emb_rel_k; // (1, 2*window+1, head_dim)
    ggml_tensor* emb_rel_v;
    ggml_tensor* norm1_g;
    ggml_tensor* norm1_b;
    ggml_tensor* ffn_c1_w;
    ggml_tensor* ffn_c1_b;
    ggml_tensor* ffn_c2_w;
    ggml_tensor* ffn_c2_b;
    ggml_tensor* norm2_g;
    ggml_tensor* norm2_b;
};

struct piper_dds_conv {
    // DDSConv: n_layers of (depthwise sep conv k=3 + 1x1 conv + layer norms)
    struct layer {
        ggml_tensor* conv_sep_w; // depthwise (channels, 1, 3)
        ggml_tensor* conv_sep_b;
        ggml_tensor* conv_1x1_w; // (channels, channels, 1)
        ggml_tensor* conv_1x1_b;
        ggml_tensor* norm1_g;
        ggml_tensor* norm1_b;
        ggml_tensor* norm2_g;
        ggml_tensor* norm2_b;
    };
    std::vector<layer> layers;
};

struct piper_sdp_convflow {
    ggml_tensor* pre_w;
    ggml_tensor* pre_b; // (channels, 1, 1)
    piper_dds_conv dds;
    ggml_tensor* proj_w;
    ggml_tensor* proj_b; // (3*bins-1, channels, 1)
};

struct piper_flow_block {
    ggml_tensor* pre_w;
    ggml_tensor* pre_b; // (hidden, half, 1)
    ggml_tensor* post_w;
    ggml_tensor* post_b; // (half, hidden, 1)
    // WaveNet layers
    struct wn_layer {
        ggml_tensor* in_w;
        ggml_tensor* in_b; // dilated conv
        ggml_tensor* res_w;
        ggml_tensor* res_b; // 1x1 res/skip
    };
    std::vector<wn_layer> wn;
};

struct piper_resblock {
    // 2 conv layers per resblock
    ggml_tensor* conv0_w;
    ggml_tensor* conv0_b;
    ggml_tensor* conv1_w;
    ggml_tensor* conv1_b;
};

struct piper_weights {
    // Text encoder
    ggml_tensor* emb;    // (num_symbols, hidden)
    ggml_tensor* proj_w; // (2*inter, hidden, 1)
    ggml_tensor* proj_b;
    std::vector<piper_enc_layer> enc_layers;

    // Stochastic duration predictor
    ggml_tensor* dp_pre_w;
    ggml_tensor* dp_pre_b;
    ggml_tensor* dp_proj_w;
    ggml_tensor* dp_proj_b;
    piper_dds_conv dp_convs;
    ggml_tensor* dp_flow0_m;                  // ElementwiseAffine mean (2, 1)
    ggml_tensor* dp_flow0_logs;               // ElementwiseAffine log-scale (2, 1), may be null
    std::vector<piper_sdp_convflow> dp_flows; // ConvFlow layers

    // Residual coupling flow
    std::vector<piper_flow_block> flow_blocks;

    // HiFi-GAN decoder
    ggml_tensor* dec_conv_pre_w;
    ggml_tensor* dec_conv_pre_b;
    struct ups_stage {
        ggml_tensor* w;
        ggml_tensor* b;
    };
    std::vector<ups_stage> dec_ups;
    std::vector<piper_resblock> dec_resblocks;
    ggml_tensor* dec_conv_post_w;

    // Speaker embedding (multi-speaker models only)
    ggml_tensor* spk_emb; // nullptr for single-speaker
};

// ── Context ────────────────────────────────────────────────────────

struct piper_tts_context {
    piper_hparams hp;
    piper_weights w;
    std::vector<phoneme_map_entry> pmap;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    // Weight storage
    ggml_context* w_ctx = nullptr;
    ggml_backend_buffer_t w_buf = nullptr;

    // Runtime params (mutable)
    float noise_scale;
    float length_scale;
    float noise_w;
    int speaker_id;
    std::string espeak_voice;

    int verbosity;
    int n_threads;
    std::string dump_dir; // if non-empty, dump intermediates for diff harness
};

// ── Diff harness: dump intermediate tensors to binary files ────────
static void dump_stage(const piper_tts_context* ctx, const char* label, const float* data, size_t n) {
    if (ctx->dump_dir.empty())
        return;
    std::string path = ctx->dump_dir + "/" + label + ".bin";
    FILE* f = fopen(path.c_str(), "wb");
    if (f) {
        fwrite(data, sizeof(float), n, f);
        fclose(f);
    }
}

static void dump_stage_i64(const piper_tts_context* ctx, const char* label, const int64_t* data, size_t n) {
    if (ctx->dump_dir.empty())
        return;
    std::string path = ctx->dump_dir + "/" + label + ".bin";
    FILE* f = fopen(path.c_str(), "wb");
    if (f) {
        fwrite(data, sizeof(int64_t), n, f);
        fclose(f);
    }
}

// ── Helper: tiny ggml graph compute ────────────────────────────────
// Build a small graph, allocate ephemeral buffers, compute, return.
// Used for each sub-module.
namespace {
struct mini_graph {
    ggml_context* ctx = nullptr;
    ggml_backend_sched_t sched = nullptr;

    mini_graph(ggml_backend_sched_t sched_, size_t ctx_size = 16 * 1024 * 1024) : sched(sched_) {
        struct ggml_init_params params = {
            ctx_size, nullptr, true /* no_alloc */
        };
        ctx = ggml_init(params);
    }
    ~mini_graph() {
        if (ctx)
            ggml_free(ctx);
    }

    // Build, reserve, compute, read output into a float vector.
    std::vector<float> compute(ggml_tensor* output, int /*n_threads*/) {
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, 16384, false);
        ggml_build_forward_expand(gf, output);

        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            fprintf(stderr, "piper_tts: graph alloc failed\n");
            return {};
        }
        ggml_backend_sched_graph_compute(sched, gf);

        int n = (int)ggml_nelements(output);
        std::vector<float> result(n);
        ggml_backend_tensor_get(output, result.data(), 0, n * sizeof(float));
        return result;
    }

    // Compute and read into a pre-allocated buffer.
    bool compute_into(ggml_tensor* output, float* dst, size_t nbytes, int /*n_threads*/) {
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, 16384, false);
        ggml_build_forward_expand(gf, output);

        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            fprintf(stderr, "piper_tts: graph alloc failed\n");
            return false;
        }
        ggml_backend_sched_graph_compute(sched, gf);
        ggml_backend_tensor_get(output, dst, 0, nbytes);
        return true;
    }

    // Set input data on a tensor (after alloc).
    void set_input(ggml_tensor* t, const void* data, size_t nbytes) { ggml_backend_tensor_set(t, data, 0, nbytes); }
};
} // namespace

// ── Tensor read helper (handles F16→F32 conversion) ───────────────

static void read_tensor_f32(ggml_tensor* t, std::vector<float>& out) {
    const int64_t n = ggml_nelements(t);
    out.resize(n);
    const size_t nbytes = ggml_nbytes(t);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, nbytes);
    } else {
        // F16 or quantized (Q4_K, Q8_0, etc.) — dequantize via type traits.
        std::vector<uint8_t> raw(nbytes);
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
        const auto to_float = ggml_get_type_traits(t->type)->to_float;
        if (to_float) {
            to_float(raw.data(), out.data(), n);
        } else {
            fprintf(stderr, "piper_tts: unsupported tensor type %d for '%s'\n", (int)t->type, t->name);
            std::fill(out.begin(), out.end(), 0.0f);
        }
    }
}

// ── Layer norm ─────────────────────────────────────────────────────
// VITS uses channels-first layout (C, T). LayerNorm is over the C dim.
// ggml_norm works on dim 0, which IS the C dim in our (C, T) layout.

static ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* gamma, ggml_tensor* beta) {
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_mul(ctx, x, gamma);
    x = ggml_add(ctx, x, beta);
    return x;
}

// ── Conv1d helper (channels-first) ─────────────────────────────────
// Input:  x (C_in, T), w (K, C_in, C_out) [ggml layout], b (C_out,)
// Output: (C_out, T_out) where T_out depends on padding.

static ggml_tensor* conv1d_cf(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride = 1,
                              int pad = 0, int dilation = 1) {
    // ggml_conv_1d expects (T, C_in) input; our x is (C_in, T)
    ggml_tensor* xT = ggml_cont(ctx, ggml_transpose(ctx, x));         // (T, C_in)
    ggml_tensor* y = ggml_conv_1d(ctx, w, xT, stride, pad, dilation); // (T_out, C_out)
    y = ggml_cont(ctx, ggml_transpose(ctx, y));                       // (C_out, T_out)
    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

// ── CPU attention with windowed relative position bias ─────────────
// VITS uses relative position embeddings (window_size=4 → 9 positions).
// All data layout: [t * C + c] for (C, T) tensors (channel-major per timestep).

static void cpu_multihead_attention_relpos(
    const std::vector<float>& x,                                     // (C, T) input
    const piper_enc_layer& layer, int C, int T, int H, int D, int W, // W = window_size
    std::vector<float>& out) // (C, T) output (attention result before output proj)
{
    // 1. Linear projections via 1x1 conv weights
    std::vector<float> wq, bq, wk, bk, wv, bv;
    read_tensor_f32(layer.conv_q_w, wq);
    read_tensor_f32(layer.conv_q_b, bq);
    read_tensor_f32(layer.conv_k_w, wk);
    read_tensor_f32(layer.conv_k_b, bk);
    read_tensor_f32(layer.conv_v_w, wv);
    read_tensor_f32(layer.conv_v_b, bv);

    // Q, K, V: (C, T) via 1x1 conv. ne=[1,C,C] → flat[ci + co*C]
    std::vector<float> Q(C * T), K(C * T), V(C * T);
    for (int t = 0; t < T; t++) {
        for (int co = 0; co < C; co++) {
            float sq = bq[co], sk = bk[co], sv = bv[co];
            for (int ci = 0; ci < C; ci++) {
                float xv = x[t * C + ci];
                sq += xv * wq[ci + co * C];
                sk += xv * wk[ci + co * C];
                sv += xv * wv[ci + co * C];
            }
            Q[t * C + co] = sq;
            K[t * C + co] = sk;
            V[t * C + co] = sv;
        }
    }

    // 2. Load relative position embeddings
    // emb_rel_k: ggml ne=(D, 2W+1, 1) → flat[d + r*D], r ∈ [0, 2W]
    // emb_rel_v: same shape
    int n_rel = 2 * W + 1;
    std::vector<float> rel_k, rel_v;
    read_tensor_f32(layer.emb_rel_k, rel_k);
    read_tensor_f32(layer.emb_rel_v, rel_v);

    float inv_sqrt_d = 1.0f / sqrtf((float)D);

    // 3. Compute attention per head
    // Q,K,V are (C,T) = (H*D, T). Head h uses channels [h*D, (h+1)*D).
    out.resize(C * T, 0.0f);

    for (int h = 0; h < H; h++) {
        int ch_off = h * D;

        // 3a. Compute attention scores: scores[i][j] = Q[i] · K[j] / sqrt(D)
        //     + relative key bias: Q[i] · rel_k[clip(j-i+W, 0, 2W)] / sqrt(D)
        std::vector<float> scores(T * T);
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                float s = 0;
                for (int d = 0; d < D; d++) {
                    s += Q[i * C + ch_off + d] * K[j * C + ch_off + d];
                }
                s *= inv_sqrt_d;

                // Relative position key bias
                int r = j - i + W;
                if (r >= 0 && r < n_rel) {
                    float rel_s = 0;
                    for (int d = 0; d < D; d++) {
                        rel_s += Q[i * C + ch_off + d] * rel_k[d + r * D];
                    }
                    s += rel_s * inv_sqrt_d;
                }

                scores[i * T + j] = s;
            }
        }

        // 3b. Softmax over j dimension
        for (int i = 0; i < T; i++) {
            float max_s = scores[i * T];
            for (int j = 1; j < T; j++) {
                if (scores[i * T + j] > max_s)
                    max_s = scores[i * T + j];
            }
            float sum_e = 0;
            for (int j = 0; j < T; j++) {
                scores[i * T + j] = expf(scores[i * T + j] - max_s);
                sum_e += scores[i * T + j];
            }
            for (int j = 0; j < T; j++) {
                scores[i * T + j] /= sum_e;
            }
        }

        // 3c. Weighted sum of V + relative value bias
        for (int i = 0; i < T; i++) {
            for (int d = 0; d < D; d++) {
                float s = 0;
                for (int j = 0; j < T; j++) {
                    float w_ij = scores[i * T + j];
                    s += w_ij * V[j * C + ch_off + d];

                    // Relative position value bias
                    int r = j - i + W;
                    if (r >= 0 && r < n_rel) {
                        s += w_ij * rel_v[d + r * D];
                    }
                }
                out[i * C + ch_off + d] = s;
            }
        }
    }
}

// ── CPU 1x1 conv ──────────────────────────────────────────────────
static void cpu_conv1x1(const std::vector<float>& x, // (C_in, T)
                        ggml_tensor* w_t, ggml_tensor* b_t, int C_in, int C_out, int T,
                        std::vector<float>& out) // (C_out, T)
{
    std::vector<float> w, b;
    read_tensor_f32(w_t, w);
    read_tensor_f32(b_t, b);
    out.resize(C_out * T);
    for (int t = 0; t < T; t++) {
        for (int co = 0; co < C_out; co++) {
            float s = b[co];
            for (int ci = 0; ci < C_in; ci++) {
                s += x[t * C_in + ci] * w[ci + co * C_in];
            }
            out[t * C_out + co] = s;
        }
    }
}

// ── CPU layer norm (over channel dim) ─────────────────────────────
static void cpu_layer_norm(std::vector<float>& x, // (C, T) — modified in place
                           ggml_tensor* g_t, ggml_tensor* b_t, int C, int T) {
    std::vector<float> g, b;
    read_tensor_f32(g_t, g);
    read_tensor_f32(b_t, b);
    for (int t = 0; t < T; t++) {
        float mean = 0, var = 0;
        for (int c = 0; c < C; c++)
            mean += x[t * C + c];
        mean /= C;
        for (int c = 0; c < C; c++) {
            float d = x[t * C + c] - mean;
            var += d * d;
        }
        var /= C;
        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (int c = 0; c < C; c++) {
            x[t * C + c] = (x[t * C + c] - mean) * inv_std * g[c] + b[c];
        }
    }
}

// ── Text Encoder forward ───────────────────────────────────────────
// Implements the VITS TextEncoder with relative position attention.
// Attention is computed on CPU for the relative position bias.
// FFN and projection use ggml graphs.

static void text_encoder_forward(piper_tts_context* ctx, const std::vector<int64_t>& phoneme_ids,
                                 std::vector<float>& out_enc,    // (hidden_channels, T) raw encoder output for SDP
                                 std::vector<float>& out_mean,   // (inter_channels, T)
                                 std::vector<float>& out_logvar) // (inter_channels, T)
{
    const auto& hp = ctx->hp;
    const auto& w = ctx->w;
    const int T = (int)phoneme_ids.size();
    const int C = (int)hp.hidden_channels;
    const int half = (int)hp.inter_channels;
    const int H = (int)hp.n_heads;
    const int D = (int)hp.head_dim;
    const int W = 4; // VITS window_size for relative attention

    if (ctx->verbosity >= 2) {
        fprintf(stderr, "piper_tts: text_encoder_forward T=%d C=%d\n", T, C);
    }

    // ── Embedding lookup ──
    float sqrt_c = sqrtf((float)C);
    int emb_ne0 = (int)ctx->w.emb->ne[0]; // C
    int emb_ne1 = (int)ctx->w.emb->ne[1]; // n_vocab
    std::vector<float> emb_table(emb_ne0 * emb_ne1);
    {
        size_t emb_bytes = ggml_nbytes(ctx->w.emb);
        std::vector<uint8_t> emb_raw(emb_bytes);
        ggml_backend_tensor_get(ctx->w.emb, emb_raw.data(), 0, emb_bytes);
        if (ctx->w.emb->type == GGML_TYPE_F16) {
            const ggml_fp16_t* src = (const ggml_fp16_t*)emb_raw.data();
            for (int i = 0; i < emb_ne0 * emb_ne1; i++)
                emb_table[i] = ggml_fp16_to_fp32(src[i]);
        } else {
            memcpy(emb_table.data(), emb_raw.data(), emb_ne0 * emb_ne1 * sizeof(float));
        }
    }

    std::vector<float> x(C * T);
    for (int t = 0; t < T; t++) {
        int id = (int)phoneme_ids[t];
        if (id < 0 || id >= emb_ne1)
            id = 0;
        for (int c = 0; c < C; c++) {
            x[t * C + c] = emb_table[id * emb_ne0 + c] * sqrt_c;
        }
    }

    // ── Encoder layers (attention on CPU, FFN via ggml) ──
    for (uint32_t il = 0; il < hp.n_layers_enc; il++) {
        const auto& layer = w.enc_layers[il];

        // Attention with relative position bias (CPU)
        std::vector<float> attn_out;
        cpu_multihead_attention_relpos(x, layer, C, T, H, D, W, attn_out);

        // Output projection (1x1 conv)
        std::vector<float> o;
        cpu_conv1x1(attn_out, layer.conv_o_w, layer.conv_o_b, C, C, T, o);

        // Post-norm: x = norm1(x + o)
        for (int i = 0; i < C * T; i++)
            x[i] += o[i];
        cpu_layer_norm(x, layer.norm1_g, layer.norm1_b, C, T);

        {
            char label[64];
            snprintf(label, sizeof(label), "enc_layer%u_post_attn", il);
            dump_stage(ctx, label, x.data(), x.size());
        }

        // FFN via ggml graph (conv k=3 pad=1 → ReLU → conv k=3 pad=1)
        {
            mini_graph mg(ctx->sched, 4 * 1024 * 1024);
            auto* gc = mg.ctx;

            ggml_tensor* x_in = ggml_new_tensor_2d(gc, GGML_TYPE_F32, C, T);
            ggml_set_name(x_in, "ffn_in");
            ggml_set_input(x_in);

            ggml_tensor* ff = conv1d_cf(gc, x_in, layer.ffn_c1_w, layer.ffn_c1_b, 1, 1, 1);
            ff = ggml_relu(gc, ff);
            ff = conv1d_cf(gc, ff, layer.ffn_c2_w, layer.ffn_c2_b, 1, 1, 1);

            ggml_cgraph* gf = ggml_new_graph_custom(gc, 1024, false);
            ggml_build_forward_expand(gf, ff);
            ggml_backend_sched_reset(mg.sched);
            if (!ggml_backend_sched_alloc_graph(mg.sched, gf)) {
                fprintf(stderr, "piper_tts: FFN graph alloc failed\n");
                return;
            }
            ggml_backend_tensor_set(x_in, x.data(), 0, C * T * sizeof(float));
            ggml_backend_sched_graph_compute(mg.sched, gf);

            std::vector<float> ff_out(C * T);
            ggml_backend_tensor_get(ff, ff_out.data(), 0, C * T * sizeof(float));

            // Post-norm: x = norm2(x + ff_out)
            for (int i = 0; i < C * T; i++)
                x[i] += ff_out[i];
            cpu_layer_norm(x, layer.norm2_g, layer.norm2_b, C, T);

            {
                char label[64];
                snprintf(label, sizeof(label), "enc_layer%u_post_ffn", il);
                dump_stage(ctx, label, x.data(), x.size());
            }
        }
    }

    // Save raw encoder output for SDP
    out_enc = x;
    dump_stage(ctx, "enc_output", x.data(), x.size());

    // ── Final projection for mean/logvar via ggml ──
    {
        mini_graph mg(ctx->sched, 4 * 1024 * 1024);
        auto* gc = mg.ctx;

        ggml_tensor* x_in = ggml_new_tensor_2d(gc, GGML_TYPE_F32, C, T);
        ggml_set_name(x_in, "proj_in");
        ggml_set_input(x_in);

        ggml_tensor* proj = conv1d_cf(gc, x_in, w.proj_w, w.proj_b);

        ggml_cgraph* gf = ggml_new_graph_custom(gc, 256, false);
        ggml_build_forward_expand(gf, proj);
        ggml_backend_sched_reset(mg.sched);
        if (!ggml_backend_sched_alloc_graph(mg.sched, gf)) {
            fprintf(stderr, "piper_tts: proj graph alloc failed\n");
            return;
        }
        ggml_backend_tensor_set(x_in, x.data(), 0, C * T * sizeof(float));
        ggml_backend_sched_graph_compute(mg.sched, gf);

        int proj_size = 2 * half * T;
        std::vector<float> proj_data(proj_size);
        ggml_backend_tensor_get(proj, proj_data.data(), 0, proj_size * sizeof(float));
        dump_stage(ctx, "enc_proj", proj_data.data(), proj_size);

        out_mean.resize(half * T);
        out_logvar.resize(half * T);
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < half; c++) {
                out_mean[c + t * half] = proj_data[c + t * 2 * half];
                out_logvar[c + t * half] = proj_data[c + half + t * 2 * half];
            }
        }
    }
}

// ── Stochastic Duration Predictor ──────────────────────────────────
// Forward (reverse=true for inference): noise → SDP flows → log-durations
// This is the most complex module due to the rational quadratic spline
// transforms in the ConvFlow layers.

// Rational quadratic spline transform (forward or inverse).
// References: Neural Spline Flows (Durkan et al. 2019)
// Input/output: single scalar value, conditioned by bin params.
static float rqs_forward(float x_in, const float* w_bins, const float* h_bins, const float* d_knots, int num_bins,
                         float range_min, float range_max, float* log_det) {
    // Softmax over bin widths and heights
    std::vector<float> widths(num_bins), heights(num_bins), derivatives(num_bins + 1);

    // Softmax for widths
    float max_w = *std::max_element(w_bins, w_bins + num_bins);
    float sum_w = 0;
    for (int i = 0; i < num_bins; i++) {
        widths[i] = expf(w_bins[i] - max_w);
        sum_w += widths[i];
    }
    for (int i = 0; i < num_bins; i++) {
        widths[i] = widths[i] / sum_w * (range_max - range_min) + 1e-5f;
    }

    // Softmax for heights
    float max_h = *std::max_element(h_bins, h_bins + num_bins);
    float sum_h = 0;
    for (int i = 0; i < num_bins; i++) {
        heights[i] = expf(h_bins[i] - max_h);
        sum_h += heights[i];
    }
    for (int i = 0; i < num_bins; i++) {
        heights[i] = heights[i] / sum_h * (range_max - range_min) + 1e-5f;
    }

    // Softplus for derivatives (ensure > 0)
    derivatives[0] = 1.0f;
    derivatives[num_bins] = 1.0f;
    for (int i = 0; i < num_bins - 1; i++) {
        derivatives[i + 1] = logf(1.0f + expf(d_knots[i])) + 1e-5f;
    }

    // Cumulative sums for knot positions
    std::vector<float> cum_w(num_bins + 1), cum_h(num_bins + 1);
    cum_w[0] = range_min;
    cum_h[0] = range_min;
    for (int i = 0; i < num_bins; i++) {
        cum_w[i + 1] = cum_w[i] + widths[i];
        cum_h[i + 1] = cum_h[i] + heights[i];
    }

    // Find which bin x_in falls into
    int bin_idx = num_bins - 1;
    for (int i = 0; i < num_bins; i++) {
        if (x_in < cum_w[i + 1]) {
            bin_idx = i;
            break;
        }
    }

    // Compute the rational quadratic transform within the bin
    float w_k = widths[bin_idx];
    float h_k = heights[bin_idx];
    float d_k = derivatives[bin_idx];
    float d_k1 = derivatives[bin_idx + 1];
    float xi = (x_in - cum_w[bin_idx]) / w_k; // normalized position in bin

    float num = h_k * (d_k * xi * xi + 2.0f * xi * (1.0f - xi));
    float den = d_k + (d_k1 + d_k - 2.0f) * xi * (1.0f - xi);
    float y = cum_h[bin_idx] + num / den;

    if (log_det) {
        float dnum = 2.0f * d_k * xi + 2.0f * (1.0f - 2.0f * xi);
        float dden = (d_k1 + d_k - 2.0f) * (1.0f - 2.0f * xi);
        float dy_dxi = h_k * (dnum * den - num * dden) / (den * den);
        float dy_dx = dy_dxi / w_k;
        *log_det = logf(fabsf(dy_dx) + 1e-10f);
    }

    return y;
}

static float rqs_inverse(float y_in, const float* w_bins, const float* h_bins, const float* d_knots, int num_bins,
                         float range_min, float range_max) {
    // Same setup as forward, but solve for x given y
    std::vector<float> widths(num_bins), heights(num_bins), derivatives(num_bins + 1);

    float max_w = *std::max_element(w_bins, w_bins + num_bins);
    float sum_w = 0;
    for (int i = 0; i < num_bins; i++) {
        widths[i] = expf(w_bins[i] - max_w);
        sum_w += widths[i];
    }
    for (int i = 0; i < num_bins; i++) {
        widths[i] = widths[i] / sum_w * (range_max - range_min) + 1e-5f;
    }

    float max_h = *std::max_element(h_bins, h_bins + num_bins);
    float sum_h = 0;
    for (int i = 0; i < num_bins; i++) {
        heights[i] = expf(h_bins[i] - max_h);
        sum_h += heights[i];
    }
    for (int i = 0; i < num_bins; i++) {
        heights[i] = heights[i] / sum_h * (range_max - range_min) + 1e-5f;
    }

    derivatives[0] = 1.0f;
    derivatives[num_bins] = 1.0f;
    for (int i = 0; i < num_bins - 1; i++) {
        derivatives[i + 1] = logf(1.0f + expf(d_knots[i])) + 1e-5f;
    }

    std::vector<float> cum_w(num_bins + 1), cum_h(num_bins + 1);
    cum_w[0] = range_min;
    cum_h[0] = range_min;
    for (int i = 0; i < num_bins; i++) {
        cum_w[i + 1] = cum_w[i] + widths[i];
        cum_h[i + 1] = cum_h[i] + heights[i];
    }

    // Find which bin y_in falls into (using heights cumsum)
    int bin_idx = num_bins - 1;
    for (int i = 0; i < num_bins; i++) {
        if (y_in < cum_h[i + 1]) {
            bin_idx = i;
            break;
        }
    }

    float w_k = widths[bin_idx];
    float h_k = heights[bin_idx];
    float d_k = derivatives[bin_idx];
    float d_k1 = derivatives[bin_idx + 1];

    // Analytical quadratic inverse from Neural Spline Flows (Durkan 2019).
    // The VITS/piper code uses: a = h_k(s_k - d_k) + (y-y_k)(d_k1+d_k-2s_k)
    //                           b = h_k*d_k - (y-y_k)(d_k1+d_k-2s_k)
    //                           c = -s_k*(y-y_k)
    // with s_k = (d_k1+d_k-2) + 2  ... no, s_k is the slope of the
    // secant: s_k = h_k / w_k.
    float s_k = h_k / (w_k + 1e-10f);
    float y_yk = y_in - cum_h[bin_idx];
    float t2 = d_k1 + d_k - 2.0f * s_k;

    float a = h_k * (s_k - d_k) + y_yk * t2;
    float b = h_k * d_k - y_yk * t2;
    float c = -s_k * y_yk;

    float disc = b * b - 4.0f * a * c;
    if (disc < 0)
        disc = 0;

    float xi = (2.0f * c) / (-b - sqrtf(disc + 1e-10f));
    xi = std::clamp(xi, 0.0f, 1.0f);

    return xi * w_k + cum_w[bin_idx];
}

// DDSConv forward: n_layers of depthwise separable conv + norms
static void dds_conv_forward(piper_tts_context* /*pctx*/, const piper_dds_conv& dds,
                             const std::vector<float>& x_in, // (C, T)
                             int C, int T,
                             std::vector<float>& out) // (C, T)
{
    // Each DDSConv layer:
    //   x → depthwise conv (k=3, pad=dilation, dilation=2^layer) → norm1
    //   → 1x1 conv → norm2 → ReLU → residual add
    // We compute this on CPU without ggml for simplicity (small tensors).

    out = x_in; // start with residual

    for (size_t il = 0; il < dds.layers.size(); il++) {
        const auto& layer = dds.layers[il];
        int K = 3;
        // VITS DDSConv: dilation = kernel_size^i (k=3 → 1, 3, 9, 27, ...)
        int dilation = 1;
        for (size_t p = 0; p < il; p++)
            dilation *= K;
        int pad = dilation; // (K-1)*dilation/2 = dilation for K=3
        std::vector<float> w_sep, b_sep;
        read_tensor_f32(layer.conv_sep_w, w_sep);
        read_tensor_f32(layer.conv_sep_b, b_sep);

        // Depthwise conv: for each channel independently
        std::vector<float> padded(C * (T + 2 * pad), 0.0f);
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                padded[(t + pad) * C + c] = out[t * C + c];
            }
        }

        std::vector<float> dw_out(C * T, 0.0f);
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                float sum = b_sep[c];
                for (int k = 0; k < K; k++) {
                    int ti = t + k * dilation;
                    // ggml ne=[K,1,C]: flat[k + c*K]
                    sum += padded[ti * C + c] * w_sep[k + c * K];
                }
                dw_out[t * C + c] = sum;
            }
        }

        // Layer norm 1
        std::vector<float> n1g, n1b;
        read_tensor_f32(layer.norm1_g, n1g);
        read_tensor_f32(layer.norm1_b, n1b);
        for (int t = 0; t < T; t++) {
            float mean = 0, var = 0;
            for (int c = 0; c < C; c++)
                mean += dw_out[t * C + c];
            mean /= C;
            for (int c = 0; c < C; c++) {
                float d = dw_out[t * C + c] - mean;
                var += d * d;
            }
            var /= C;
            for (int c = 0; c < C; c++) {
                dw_out[t * C + c] = (dw_out[t * C + c] - mean) / sqrtf(var + 1e-5f) * n1g[c] + n1b[c];
            }
        }

        // GELU after norm1 (VITS DDSConv applies GELU after both norms)
        for (int i = 0; i < C * T; i++) {
            float v = dw_out[i];
            dw_out[i] = v * 0.5f * (1.0f + erff(v * 0.7071067811865476f));
        }

        // 1x1 conv (linear)
        std::vector<float> w_1x1, b_1x1;
        read_tensor_f32(layer.conv_1x1_w, w_1x1);
        read_tensor_f32(layer.conv_1x1_b, b_1x1);

        std::vector<float> lin_out(C * T, 0.0f);
        // w_1x1 is (1, C_in, C_out) in ggml → (C_out, C_in) matmul
        for (int t = 0; t < T; t++) {
            for (int co = 0; co < C; co++) {
                float sum = b_1x1[co];
                for (int ci = 0; ci < C; ci++) {
                    // ggml 1x1 conv weight ne=[1,Cin,Cout]: flat[ci + co*Cin]
                    sum += dw_out[t * C + ci] * w_1x1[ci + co * C];
                }
                lin_out[t * C + co] = sum;
            }
        }

        // Layer norm 2
        std::vector<float> n2g, n2b;
        read_tensor_f32(layer.norm2_g, n2g);
        read_tensor_f32(layer.norm2_b, n2b);
        for (int t = 0; t < T; t++) {
            float mean = 0, var = 0;
            for (int c = 0; c < C; c++)
                mean += lin_out[t * C + c];
            mean /= C;
            for (int c = 0; c < C; c++) {
                float d = lin_out[t * C + c] - mean;
                var += d * d;
            }
            var /= C;
            for (int c = 0; c < C; c++) {
                lin_out[t * C + c] = (lin_out[t * C + c] - mean) / sqrtf(var + 1e-5f) * n2g[c] + n2b[c];
            }
        }

        // GELU + residual (VITS DDSConv uses GELU activation)
        for (int i = 0; i < C * T; i++) {
            float v = lin_out[i];
            // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            float g = v * 0.5f * (1.0f + erff(v * 0.7071067811865476f));
            out[i] += g;
        }
    }
}

// SDP ConvFlow inverse: z → conditioning → spline inverse → z_updated
static void sdp_convflow_inverse(piper_tts_context* pctx, const piper_sdp_convflow& cf,
                                 std::vector<float>& z,       // (2, T) — modified in place
                                 const std::vector<float>& h, // (C, T) conditioning from encoder
                                 int C, int T, int num_bins) {
    // ConvFlow with coupling:
    // Split z into z0 (channel 0) and z1 (channel 1)
    // z0 is unchanged, z1 is transformed conditioned on z0
    // In reverse: z1 = spline_inverse(z1, params_from(z0, h))

    // 1. Project z0 to hidden dim
    std::vector<float> z0(T);
    for (int t = 0; t < T; t++)
        z0[t] = z[t * 2 + 0];

    // pre conv (1→C)
    std::vector<float> w_pre, b_pre;
    read_tensor_f32(cf.pre_w, w_pre);
    read_tensor_f32(cf.pre_b, b_pre);

    std::vector<float> h_cond(C * T);
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < C; c++) {
            h_cond[t * C + c] = z0[t] * w_pre[c] + b_pre[c] + h[t * C + c];
        }
    }

    // 2. DDSConv conditioning
    std::vector<float> h_out;
    dds_conv_forward(pctx, cf.dds, h_cond, C, T, h_out);

    // 3. Project to spline params: (C, T) → (3*bins-1, T)
    int n_params = 3 * num_bins - 1;
    std::vector<float> w_proj, b_proj;
    read_tensor_f32(cf.proj_w, w_proj);
    read_tensor_f32(cf.proj_b, b_proj);

    std::vector<float> params(n_params * T);
    for (int t = 0; t < T; t++) {
        for (int p = 0; p < n_params; p++) {
            float sum = b_proj[p];
            for (int c = 0; c < C; c++) {
                // ggml ne=[1,C,n_params]: flat[c + p*C]
                sum += h_out[t * C + c] * w_proj[c + p * C];
            }
            params[t * n_params + p] = sum;
        }
    }

    // 4. Scale widths and heights by 1/sqrt(filter_channels) (VITS ConvFlow)
    float inv_sqrt_fc = 1.0f / sqrtf((float)C);
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < 2 * num_bins; i++) { // first 2*num_bins = widths + heights
            params[t * n_params + i] *= inv_sqrt_fc;
        }
        // derivatives (last num_bins-1 values) are NOT scaled
    }

    // 5. Apply spline inverse to z1
    float range_min = -5.0f, range_max = 5.0f;
    for (int t = 0; t < T; t++) {
        float z1 = z[t * 2 + 1];

        // Identity pass-through for values outside spline range
        if (z1 <= range_min || z1 >= range_max) {
            continue; // identity tail
        }

        const float* p = &params[t * n_params];
        const float* w_bins = p;
        const float* h_bins = p + num_bins;
        const float* d_knots = p + 2 * num_bins;
        z[t * 2 + 1] = rqs_inverse(z1, w_bins, h_bins, d_knots, num_bins, range_min, range_max);
    }
}

static void sdp_forward(piper_tts_context* ctx,
                        const std::vector<float>& h_enc, // (hidden, T) encoder output
                        int T, float noise_w,
                        std::vector<float>& log_durations) // (T,) output
{
    const auto& hp = ctx->hp;
    const auto& w = ctx->w;
    int C = (int)hp.hidden_channels;
    int num_bins = (int)hp.sdp_num_bins;

    // VITS SDP: x = dp.proj(dp.convs(dp.pre(detach(h_enc))))
    // Read pre weights
    std::vector<float> w_pre2, b_pre2;
    read_tensor_f32(w.dp_pre_w, w_pre2);
    read_tensor_f32(w.dp_pre_b, b_pre2);

    // 1x1 conv: dp.pre on h_enc → h
    std::vector<float> h(C * T, 0.0f);
    for (int t = 0; t < T; t++) {
        for (int co = 0; co < C; co++) {
            float s = b_pre2[co];
            for (int ci = 0; ci < C; ci++) {
                s += h_enc[t * C + ci] * w_pre2[ci + co * C];
            }
            h[t * C + co] = s;
        }
    }
    dump_stage(ctx, "sdp_pre", h.data(), h.size());

    // 2. DDSConv conditioning
    std::vector<float> h_dds;
    dds_conv_forward(ctx, w.dp_convs, h, C, T, h_dds);
    dump_stage(ctx, "sdp_ddsconv", h_dds.data(), h_dds.size());

    // 3. dp.proj after DDSConv (VITS: x = self.proj(self.convs(self.pre(x))))
    std::vector<float> w_proj2, b_proj2;
    read_tensor_f32(w.dp_proj_w, w_proj2);
    read_tensor_f32(w.dp_proj_b, b_proj2);

    std::vector<float> h_cond(C * T, 0.0f);
    for (int t = 0; t < T; t++) {
        for (int co = 0; co < C; co++) {
            float s = b_proj2[co];
            for (int ci = 0; ci < C; ci++) {
                s += h_dds[t * C + ci] * w_proj2[ci + co * C];
            }
            h_cond[t * C + co] = s;
        }
    }
    dump_stage(ctx, "sdp_proj", h_cond.data(), h_cond.size());

    if (ctx->verbosity >= 2) {
        float h_sum = 0, h_max = 0;
        for (size_t i = 0; i < h_cond.size(); i++) {
            h_sum += h_cond[i];
            if (fabsf(h_cond[i]) > h_max)
                h_max = fabsf(h_cond[i]);
        }
        fprintf(stderr, "piper_tts: SDP conditioning mean=%.4f max_abs=%.4f\n", h_sum / (float)h_cond.size(), h_max);
    }

    // 3. Generate noise z ~ N(0, noise_w²)
    std::vector<float> z(2 * T, 0.0f);
    if (noise_w > 0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, noise_w);
        for (int i = 0; i < 2 * T; i++) {
            z[i] = dist(gen);
        }
    }

    // 4. Run SDP flows in reverse mode.
    // Original flow list: [ElemAffine(0), Flip(1), Flip(2), ConvFlow(3), Flip(4), ConvFlow(5), Flip(6), ConvFlow(7)]
    // VITS reverse: reversed(flows)[:-2] = [ConvFlow(7), Flip(6), ConvFlow(5), Flip(4), ConvFlow(3), Flip(2)]
    // So we skip ElemAffine(0) and Flip(1) entirely.
    // Sequence: ConvFlow_7 → Flip → ConvFlow_5 → Flip → ConvFlow_3 → Flip

    int sdp_flow_onnx_ids[] = {7, 5, 3}; // ONNX flow indices for labeling
    for (int fi = (int)w.dp_flows.size() - 1; fi >= 0; fi--) {
        sdp_convflow_inverse(ctx, w.dp_flows[fi], z, h_cond, C, T, num_bins);

        // Dump z after this ConvFlow (before flip)
        {
            char label[64];
            snprintf(label, sizeof(label), "sdp_flow%d_z", sdp_flow_onnx_ids[w.dp_flows.size() - 1 - fi]);
            dump_stage(ctx, label, z.data(), z.size());
        }

        // Flip after each ConvFlow
        for (int t = 0; t < T; t++) {
            std::swap(z[t * 2 + 0], z[t * 2 + 1]);
        }
    }
    // ElementwiseAffine (flows.0) inverse: z = (z - m) * exp(-logs)
    // Piper includes this in the ONNX model (unlike standard VITS which skips it).
    if (w.dp_flow0_m) {
        std::vector<float> ea_m, ea_logs;
        read_tensor_f32(w.dp_flow0_m, ea_m);

        if (w.dp_flow0_logs) {
            read_tensor_f32(w.dp_flow0_logs, ea_logs);
        }

        for (int t = 0; t < T; t++) {
            for (int c = 0; c < 2; c++) {
                float m_val = (c < (int)ea_m.size()) ? ea_m[c] : 0.0f;
                float neg_logs = (w.dp_flow0_logs && c < (int)ea_logs.size()) ? -ea_logs[c] : 0.0f;
                z[t * 2 + c] = (z[t * 2 + c] - m_val) * expf(neg_logs);
            }
        }
    }

    // Dump the final logw (both channels, for comparison with ONNX which has (2,T))
    dump_stage(ctx, "sdp_logw", z.data(), z.size());

    // 5. Extract log-durations from z (take first channel)
    log_durations.resize(T);
    for (int t = 0; t < T; t++) {
        log_durations[t] = z[t * 2 + 0];
    }

    // Debug: print first few log-durations
    if (ctx->verbosity >= 2) {
        fprintf(stderr, "piper_tts: SDP log_dur[0:5] =");
        for (int t = 0; t < std::min(T, 5); t++) {
            fprintf(stderr, " %.4f", log_durations[t]);
        }
        fprintf(stderr, "\n");
    }
}

// ── Duration alignment ─────────────────────────────────────────────
// Expand encoder outputs from (C, T_text) to (C, T_audio) using
// duration-based alignment. Each phoneme position t is repeated
// duration[t] times.

static void duration_align(const std::vector<float>& enc_out, // (C, T_text) mean or logvar
                           const std::vector<int>& durations, // (T_text,) integer durations
                           int C, int T_text,
                           std::vector<float>& out) // (C, T_audio)
{
    int T_audio = 0;
    for (int d : durations)
        T_audio += d;

    out.resize(C * T_audio);
    int pos = 0;
    for (int t = 0; t < T_text; t++) {
        for (int d = 0; d < durations[t]; d++) {
            for (int c = 0; c < C; c++) {
                out[pos * C + c] = enc_out[t * C + c];
            }
            pos++;
        }
    }
}

// ── WaveNet forward (in coupling flow) ─────────────────────────────
// WaveNet conditioning network used in ResidualCouplingBlock.
// Input: x (half, T), speaker_cond ignored for single-speaker.
// Output: (half, T) — mean and log_std of the affine transform.

static void wavenet_forward(piper_tts_context* pctx, const piper_flow_block& fb,
                            const std::vector<float>& x_in, // (hidden, T) after pre conv
                            int hidden, int T,
                            std::vector<float>& out) // (hidden, T)
{
    // WaveNet: n_layers of dilated gated conv + residual + skip connections
    // in_layer: dilated conv (hidden → 2*hidden, k=5, d=2^i)
    // split into two halves: tanh(a) * sigmoid(b)
    // res_skip_layer: 1x1 conv (hidden → 2*hidden or hidden)
    // split into residual and skip parts (or just skip for last layer)

    int n_layers = (int)fb.wn.size();
    std::vector<float> x = x_in;
    std::vector<float> skip_sum(hidden * T, 0.0f);

    for (int il = 0; il < n_layers; il++) {
        const auto& wn = fb.wn[il];
        int K = (int)pctx->hp.wn_kernel_size; // 5
        int dilation = 1;                     // VITS flow uses dilation_rate=1
        int pad = ((K - 1) * dilation) / 2;   // = 2 for K=5, d=1

        // Read in_layer weights
        int in_ch = hidden, out_ch = 2 * hidden;
        std::vector<float> w_in, b_in;
        read_tensor_f32(wn.in_w, w_in);
        read_tensor_f32(wn.in_b, b_in);

        // Dilated conv: pad input
        int T_padded = T + 2 * pad;
        std::vector<float> xp(in_ch * T_padded, 0.0f);
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < in_ch; c++) {
                xp[(t + pad) * in_ch + c] = x[t * in_ch + c];
            }
        }

        // Conv with dilation
        std::vector<float> conv_out(out_ch * T, 0.0f);
        for (int t = 0; t < T; t++) {
            for (int oc = 0; oc < out_ch; oc++) {
                float sum = b_in[oc];
                for (int k = 0; k < K; k++) {
                    int ti = t + k * dilation;
                    for (int ic = 0; ic < in_ch; ic++) {
                        // Weight layout in ggml: (K, C_in, C_out)
                        // ggml ne=[K,Cin,Cout]: flat[k + ci*K + co*K*Cin]
                        sum += xp[ti * in_ch + ic] * w_in[k + ic * K + oc * K * in_ch];
                    }
                }
                conv_out[t * out_ch + oc] = sum;
            }
        }

        // Gated activation: tanh(first half) * sigmoid(second half)
        std::vector<float> gated(hidden * T);
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < hidden; c++) {
                float a = tanhf(conv_out[t * out_ch + c]);
                float b = 1.0f / (1.0f + expf(-conv_out[t * out_ch + hidden + c]));
                gated[t * hidden + c] = a * b;
            }
        }

        // Res+skip layer: 1x1 conv
        int res_skip_out_ch = (int)wn.res_b->ne[0]; // bias = output channels

        std::vector<float> w_rs, b_rs;
        read_tensor_f32(wn.res_w, w_rs);
        read_tensor_f32(wn.res_b, b_rs);

        std::vector<float> rs_out_data(res_skip_out_ch * T, 0.0f);
        for (int t = 0; t < T; t++) {
            for (int oc = 0; oc < res_skip_out_ch; oc++) {
                float sum = b_rs[oc];
                for (int ic = 0; ic < hidden; ic++) {
                    // ggml 1x1 conv ne=[1,hidden,rs_out]: flat[ic + oc*hidden]
                    sum += gated[t * hidden + ic] * w_rs[ic + oc * hidden];
                }
                rs_out_data[t * res_skip_out_ch + oc] = sum;
            }
        }

        if (il < n_layers - 1) {
            // Non-last layer: split into residual (hidden) + skip (hidden)
            for (int t = 0; t < T; t++) {
                for (int c = 0; c < hidden; c++) {
                    x[t * hidden + c] = x[t * hidden + c] + rs_out_data[t * res_skip_out_ch + c];
                    skip_sum[t * hidden + c] += rs_out_data[t * res_skip_out_ch + hidden + c];
                }
            }
        } else {
            // Last layer: all goes to skip
            for (int t = 0; t < T; t++) {
                for (int c = 0; c < hidden; c++) {
                    skip_sum[t * hidden + c] += rs_out_data[t * res_skip_out_ch + c];
                }
            }
        }
    }

    out = skip_sum;
}

// ── Residual Coupling Flow (inverse) ───────────────────────────────
// z → split into z0, z1 → WaveNet(z0) → affine inverse on z1 → concat

static void flow_inverse(piper_tts_context* pctx,
                         std::vector<float>& z, // (inter_channels, T) — modified in place
                         int T) {
    const auto& hp = pctx->hp;
    int C = (int)hp.inter_channels;
    int half = C / 2;

    // Process flow blocks in reverse order (inverse)
    for (int fi = (int)pctx->w.flow_blocks.size() - 1; fi >= 0; fi--) {
        const auto& fb = pctx->w.flow_blocks[fi];

        // Flip channels: VITS Flip = torch.flip(x, [1]) = reverse ALL channels
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C / 2; c++) {
                std::swap(z[t * C + c], z[t * C + C - 1 - c]);
            }
        }

        // Split: z0 = z[:half], z1 = z[half:]
        std::vector<float> z0(half * T), z1(half * T);
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < half; c++) {
                z0[t * half + c] = z[t * C + c];
                z1[t * half + c] = z[t * C + half + c];
            }
        }

        // Pre conv: (half → hidden, 1x1)
        int hidden = C; // WaveNet hidden channels = inter_channels
        std::vector<float> w_pre, b_pre;
        read_tensor_f32(fb.pre_w, w_pre);
        read_tensor_f32(fb.pre_b, b_pre);

        std::vector<float> h(hidden * T, 0.0f);
        for (int t = 0; t < T; t++) {
            for (int oc = 0; oc < hidden; oc++) {
                float sum = b_pre[oc];
                for (int ic = 0; ic < half; ic++) {
                    // ggml 1x1 conv ne=[1,half,hidden]: flat[ic + oc*half]
                    sum += z0[t * half + ic] * w_pre[ic + oc * half];
                }
                h[t * hidden + oc] = sum;
            }
        }

        // Dump pre conv output
        {
            int flow_onnx_ids[] = {0, 2, 4, 6};
            char label[64];
            snprintf(label, sizeof(label), "flow%d_pre", flow_onnx_ids[fi]);
            dump_stage(pctx, label, h.data(), h.size());
        }

        // WaveNet
        std::vector<float> wn_out;
        wavenet_forward(pctx, fb, h, hidden, T, wn_out);

        // Dump WN output (skip_sum)
        {
            int flow_onnx_ids[] = {0, 2, 4, 6};
            char label[64];
            snprintf(label, sizeof(label), "flow%d_wn_out", flow_onnx_ids[fi]);
            dump_stage(pctx, label, wn_out.data(), wn_out.size());
        }

        // Post conv: (hidden → half, 1x1) → split into mean, log_std
        // Actually post projects hidden → half, and the output IS the mean.
        // The log_std is zero-initialized (VITS uses mean_only=True for coupling).
        // Wait, let me check: in VITS ResidualCouplingLayer, if mean_only:
        //   post = Conv1d(hidden, half, 1) → m = post(WN(z0))
        //   z1 = (z1 - m) in reverse mode
        // No log_std transform. Let me verify with the weight shapes.
        // post.weight is (half, hidden, 1) = 96 × 192 × 1 → checks out.
        // post.bias is (half,) = 96. Matches.

        std::vector<float> w_post, b_post;
        read_tensor_f32(fb.post_w, w_post);
        read_tensor_f32(fb.post_b, b_post);

        std::vector<float> m(half * T, 0.0f);
        for (int t = 0; t < T; t++) {
            for (int oc = 0; oc < half; oc++) {
                float sum = b_post[oc];
                for (int ic = 0; ic < hidden; ic++) {
                    // ggml 1x1 conv ne=[1,hidden,half]: flat[ic + oc*hidden]
                    sum += wn_out[t * hidden + ic] * w_post[ic + oc * hidden];
                }
                m[t * half + oc] = sum;
            }
        }

        // Dump flow block m
        {
            int flow_onnx_ids[] = {0, 2, 4, 6};
            char label[64];
            snprintf(label, sizeof(label), "flow%d_m", flow_onnx_ids[fi]);
            dump_stage(pctx, label, m.data(), m.size());
        }

        // Inverse affine (mean_only): z1 = z1 - m
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < half; c++) {
                z1[t * half + c] -= m[t * half + c];
            }
        }

        // Recombine
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < half; c++) {
                z[t * C + c] = z0[t * half + c];
                z[t * C + half + c] = z1[t * half + c];
            }
        }
    }
}

// ── HiFi-GAN Decoder ───────────────────────────────────────────────
// Uses ggml graph for the decoder since it's the largest compute.

static bool hifigan_decode(piper_tts_context* pctx,
                           const std::vector<float>& z, // (inter_channels, T_latent)
                           int T_latent, std::vector<float>& pcm_out) {
    const auto& hp = pctx->hp;
    const auto& w = pctx->w;
    int C_in = (int)hp.inter_channels;

    mini_graph mg(pctx->sched);
    auto* gc = mg.ctx;

    // Input tensor
    ggml_tensor* x_input = ggml_new_tensor_2d(gc, GGML_TYPE_F32, C_in, T_latent);
    ggml_set_name(x_input, "dec_input");
    ggml_set_input(x_input);

    // conv_pre: (inter → upsample_initial_channel, k=7, pad=3)
    ggml_tensor* x = conv1d_cf(gc, x_input, w.dec_conv_pre_w, w.dec_conv_pre_b, 1, 3, 1);

    // Probe: conv_pre output
    ggml_tensor* probe_conv_pre = ggml_cont(gc, x);
    ggml_set_name(probe_conv_pre, "probe_conv_pre");

    // Upsample stages
    int n_resblocks_per_stage = (int)hp.resblock_kernels.size(); // 3
    int rb_idx = 0;

    for (uint32_t us = 0; us < hp.n_upsample_stages; us++) {
        // LeakyReLU(0.1)
        x = ggml_leaky_relu(gc, x, 0.1f, false);

        // ConvTranspose1d upsample
        int stride = (int)hp.upsample_rates[us];
        int kernel = (int)hp.upsample_kernels[us];
        int crop_each = (kernel - stride) / 2;

        x = core_convt::convt1d_crop(gc, x, w.dec_ups[us].w, w.dec_ups[us].b, stride, crop_each, crop_each);

        // MRF (Multi-Receptive-Field Fusion): average of resblocks
        ggml_tensor* sum_rb = nullptr;

        for (int ri = 0; ri < n_resblocks_per_stage; ri++) {
            const auto& rb = w.dec_resblocks[rb_idx + ri];
            int rk = (int)hp.resblock_kernels[ri];

            // Piper resblock dilations: conv0 dilation = (k-1)/2, conv1 = d0*(d0+1)
            int d0 = (rk - 1) / 2;
            int d1 = d0 * (d0 + 1);
            int pad0 = (rk - 1) * d0 / 2;
            int pad1 = (rk - 1) * d1 / 2;

            // ResBlock: two sub-blocks, each with LeakyReLU → dilated conv → residual
            ggml_tensor* y1 = ggml_leaky_relu(gc, x, 0.1f, false);
            y1 = conv1d_cf(gc, y1, rb.conv0_w, rb.conv0_b, 1, pad0, d0);
            y1 = ggml_add(gc, y1, x); // residual after conv0

            ggml_tensor* xr = ggml_leaky_relu(gc, y1, 0.1f, false);
            xr = conv1d_cf(gc, xr, rb.conv1_w, rb.conv1_b, 1, pad1, d1);
            xr = ggml_add(gc, xr, y1); // residual after conv1

            if (sum_rb == nullptr) {
                sum_rb = xr;
            } else {
                sum_rb = ggml_add(gc, sum_rb, xr);
            }
        }

        // Average
        x = ggml_scale(gc, sum_rb, 1.0f / (float)n_resblocks_per_stage);
        rb_idx += n_resblocks_per_stage;
    }

    // Probe: stage MRF outputs (use the last x as dec_stage2_mrf proxy)
    ggml_tensor* probe_mrf_last = ggml_cont(gc, x);
    ggml_set_name(probe_mrf_last, "probe_mrf_last");

    // Final: LeakyReLU → conv_post → tanh
    x = ggml_leaky_relu(gc, x, 0.1f, false);
    x = conv1d_cf(gc, x, w.dec_conv_post_w, nullptr, 1, 3, 1);
    x = ggml_tanh(gc, x);

    // Build and compute
    ggml_cgraph* gf = ggml_new_graph_custom(gc, 16384, false);
    ggml_build_forward_expand(gf, x);
    ggml_build_forward_expand(gf, probe_conv_pre);
    ggml_build_forward_expand(gf, probe_mrf_last);

    ggml_backend_sched_reset(mg.sched);
    if (!ggml_backend_sched_alloc_graph(mg.sched, gf)) {
        fprintf(stderr, "piper_tts: HiFi-GAN graph alloc failed\n");
        return false;
    }

    // Set input
    ggml_backend_tensor_set(x_input, z.data(), 0, z.size() * sizeof(float));

    ggml_backend_sched_graph_compute(mg.sched, gf);

    // Dump decoder probes
    if (!pctx->dump_dir.empty()) {
        {
            int n = (int)ggml_nelements(probe_conv_pre);
            std::vector<float> tmp(n);
            ggml_backend_tensor_get(probe_conv_pre, tmp.data(), 0, n * sizeof(float));
            dump_stage(pctx, "dec_conv_pre", tmp.data(), n);
        }
        {
            int n = (int)ggml_nelements(probe_mrf_last);
            std::vector<float> tmp(n);
            ggml_backend_tensor_get(probe_mrf_last, tmp.data(), 0, n * sizeof(float));
            dump_stage(pctx, "dec_stage2_mrf", tmp.data(), n);
        }
    }

    // Read output
    int T_audio = (int)ggml_nelements(x);
    pcm_out.resize(T_audio);
    ggml_backend_tensor_get(x, pcm_out.data(), 0, T_audio * sizeof(float));
    return true;
}

// ── Weight loading ─────────────────────────────────────────────────

static ggml_tensor* require_tensor(const std::map<std::string, ggml_tensor*>& tensors, const std::string& name) {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        fprintf(stderr, "piper_tts: missing tensor '%s'\n", name.c_str());
        return nullptr;
    }
    return it->second;
}

static ggml_tensor* try_tensor(const std::map<std::string, ggml_tensor*>& tensors, const std::string& name) {
    auto it = tensors.find(name);
    return (it != tensors.end()) ? it->second : nullptr;
}

static bool load_dds_conv(const std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix, int n_layers,
                          piper_dds_conv& dds) {
    dds.layers.resize(n_layers);
    for (int i = 0; i < n_layers; i++) {
        auto& l = dds.layers[i];
        std::string p = prefix + "." + std::to_string(i);
        l.conv_sep_w = require_tensor(tensors, prefix.substr(0, prefix.rfind('.')) + ".convs_sep." + std::to_string(i) +
                                                   ".weight");
        l.conv_sep_b =
            require_tensor(tensors, prefix.substr(0, prefix.rfind('.')) + ".convs_sep." + std::to_string(i) + ".bias");
        l.conv_1x1_w = require_tensor(tensors, prefix.substr(0, prefix.rfind('.')) + ".convs_1x1." + std::to_string(i) +
                                                   ".weight");
        l.conv_1x1_b =
            require_tensor(tensors, prefix.substr(0, prefix.rfind('.')) + ".convs_1x1." + std::to_string(i) + ".bias");
        l.norm1_g =
            require_tensor(tensors, prefix.substr(0, prefix.rfind('.')) + ".norms_1." + std::to_string(i) + ".gamma");
        l.norm1_b =
            require_tensor(tensors, prefix.substr(0, prefix.rfind('.')) + ".norms_1." + std::to_string(i) + ".beta");
        l.norm2_g =
            require_tensor(tensors, prefix.substr(0, prefix.rfind('.')) + ".norms_2." + std::to_string(i) + ".gamma");
        l.norm2_b =
            require_tensor(tensors, prefix.substr(0, prefix.rfind('.')) + ".norms_2." + std::to_string(i) + ".beta");
    }
    return true;
}

static bool load_weights(piper_tts_context* ctx, const char* path) {
    auto& hp = ctx->hp;
    auto& w = ctx->w;

    // Pass 1: metadata
    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta)
        return false;

    hp.hidden_channels = core_gguf::kv_u32(meta, "piper.hidden_channels", hp.hidden_channels);
    hp.inter_channels = core_gguf::kv_u32(meta, "piper.inter_channels", hp.inter_channels);
    hp.filter_channels = core_gguf::kv_u32(meta, "piper.filter_channels", hp.filter_channels);
    hp.n_heads = core_gguf::kv_u32(meta, "piper.n_heads", hp.n_heads);
    hp.head_dim = core_gguf::kv_u32(meta, "piper.head_dim", hp.head_dim);
    hp.n_layers_enc = core_gguf::kv_u32(meta, "piper.n_layers_enc", hp.n_layers_enc);
    hp.n_flow_blocks = core_gguf::kv_u32(meta, "piper.n_flow_blocks", hp.n_flow_blocks);
    hp.n_wn_layers = core_gguf::kv_u32(meta, "piper.n_wn_layers", hp.n_wn_layers);
    hp.wn_kernel_size = core_gguf::kv_u32(meta, "piper.wn_kernel_size", hp.wn_kernel_size);
    hp.n_upsample_stages = core_gguf::kv_u32(meta, "piper.n_upsample_stages", hp.n_upsample_stages);
    hp.upsample_initial_channel =
        core_gguf::kv_u32(meta, "piper.upsample_initial_channel", hp.upsample_initial_channel);
    hp.num_symbols = core_gguf::kv_u32(meta, "piper.num_symbols", hp.num_symbols);
    hp.num_speakers = core_gguf::kv_u32(meta, "piper.num_speakers", hp.num_speakers);
    hp.sample_rate = core_gguf::kv_u32(meta, "piper.sample_rate", hp.sample_rate);
    hp.n_sdp_flows = core_gguf::kv_u32(meta, "piper.n_sdp_flows", hp.n_sdp_flows);
    hp.sdp_num_bins = core_gguf::kv_u32(meta, "piper.sdp_num_bins", hp.sdp_num_bins);
    hp.n_sdp_dds_layers = core_gguf::kv_u32(meta, "piper.n_sdp_dds_layers", hp.n_sdp_dds_layers);
    hp.n_sdp_main_dds_layers = core_gguf::kv_u32(meta, "piper.n_sdp_main_dds_layers", hp.n_sdp_main_dds_layers);
    hp.noise_scale = core_gguf::kv_f32(meta, "piper.noise_scale", hp.noise_scale);
    hp.length_scale = core_gguf::kv_f32(meta, "piper.length_scale", hp.length_scale);
    hp.noise_w = core_gguf::kv_f32(meta, "piper.noise_w", hp.noise_w);
    hp.espeak_voice = core_gguf::kv_str(meta, "piper.espeak_voice", "en-us");
    hp.phoneme_id_map_json = core_gguf::kv_str(meta, "piper.phoneme_id_map", "{}");

    // Read upsample rates/kernels
    hp.upsample_rates.resize(hp.n_upsample_stages);
    hp.upsample_kernels.resize(hp.n_upsample_stages);
    for (uint32_t i = 0; i < hp.n_upsample_stages; i++) {
        char key[64];
        snprintf(key, sizeof(key), "piper.upsample_rate.%u", i);
        hp.upsample_rates[i] = core_gguf::kv_u32(meta, key, 8);
        snprintf(key, sizeof(key), "piper.upsample_kernel.%u", i);
        hp.upsample_kernels[i] = core_gguf::kv_u32(meta, key, 16);
    }

    // Read resblock kernels
    hp.resblock_kernels.clear();
    for (int i = 0; i < 10; i++) {
        char key[64];
        snprintf(key, sizeof(key), "piper.resblock_kernel.%d", i);
        uint32_t k = core_gguf::kv_u32(meta, key, 0);
        if (k == 0)
            break;
        hp.resblock_kernels.push_back(k);
    }
    if (hp.resblock_kernels.empty()) {
        hp.resblock_kernels = {3, 5, 7}; // default
    }

    core_gguf::free_metadata(meta);

    // Parse phoneme ID map
    if (!parse_phoneme_id_map(hp.phoneme_id_map_json, ctx->pmap)) {
        fprintf(stderr, "piper_tts: failed to parse phoneme_id_map\n");
        return false;
    }

    // Pass 2: load weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "piper", wl)) {
        return false;
    }
    ctx->w_ctx = wl.ctx;
    ctx->w_buf = wl.buf;

    auto& tensors = wl.tensors;

    // ── Text encoder ──
    w.emb = require_tensor(tensors, "enc_p.emb.weight");
    w.proj_w = require_tensor(tensors, "enc_p.proj.weight");
    w.proj_b = require_tensor(tensors, "enc_p.proj.bias");
    if (!w.emb || !w.proj_w)
        return false;

    w.enc_layers.resize(hp.n_layers_enc);
    for (uint32_t i = 0; i < hp.n_layers_enc; i++) {
        auto& l = w.enc_layers[i];
        std::string p = "enc_p.encoder.attn_layers." + std::to_string(i);
        l.conv_q_w = require_tensor(tensors, p + ".conv_q.weight");
        l.conv_q_b = require_tensor(tensors, p + ".conv_q.bias");
        l.conv_k_w = require_tensor(tensors, p + ".conv_k.weight");
        l.conv_k_b = require_tensor(tensors, p + ".conv_k.bias");
        l.conv_v_w = require_tensor(tensors, p + ".conv_v.weight");
        l.conv_v_b = require_tensor(tensors, p + ".conv_v.bias");
        l.conv_o_w = require_tensor(tensors, p + ".conv_o.weight");
        l.conv_o_b = require_tensor(tensors, p + ".conv_o.bias");
        l.emb_rel_k = require_tensor(tensors, p + ".emb_rel_k");
        l.emb_rel_v = require_tensor(tensors, p + ".emb_rel_v");

        std::string np = "enc_p.encoder.norm_layers_1." + std::to_string(i);
        l.norm1_g = require_tensor(tensors, np + ".gamma");
        l.norm1_b = require_tensor(tensors, np + ".beta");

        std::string fp = "enc_p.encoder.ffn_layers." + std::to_string(i);
        l.ffn_c1_w = require_tensor(tensors, fp + ".conv_1.weight");
        l.ffn_c1_b = require_tensor(tensors, fp + ".conv_1.bias");
        l.ffn_c2_w = require_tensor(tensors, fp + ".conv_2.weight");
        l.ffn_c2_b = require_tensor(tensors, fp + ".conv_2.bias");

        std::string n2p = "enc_p.encoder.norm_layers_2." + std::to_string(i);
        l.norm2_g = require_tensor(tensors, n2p + ".gamma");
        l.norm2_b = require_tensor(tensors, n2p + ".beta");
    }

    // ── SDP ──
    w.dp_pre_w = require_tensor(tensors, "dp.pre.weight");
    w.dp_pre_b = require_tensor(tensors, "dp.pre.bias");
    w.dp_proj_w = require_tensor(tensors, "dp.proj.weight");
    w.dp_proj_b = require_tensor(tensors, "dp.proj.bias");
    w.dp_flow0_m = require_tensor(tensors, "dp.flows.0.m");
    w.dp_flow0_logs = try_tensor(tensors, "dp.flows.0.logs");

    // SDP main DDSConv
    load_dds_conv(tensors, "dp.convs.convs_sep", hp.n_sdp_main_dds_layers, w.dp_convs);

    // SDP ConvFlow layers (at indices 3, 5, 7, ...)
    w.dp_flows.resize(hp.n_sdp_flows);
    int sdp_flow_indices[] = {3, 5, 7, 9, 11}; // up to 5 flows
    for (uint32_t fi = 0; fi < hp.n_sdp_flows; fi++) {
        int idx = sdp_flow_indices[fi];
        auto& cf = w.dp_flows[fi];
        std::string p = "dp.flows." + std::to_string(idx);
        cf.pre_w = require_tensor(tensors, p + ".pre.weight");
        cf.pre_b = require_tensor(tensors, p + ".pre.bias");
        cf.proj_w = require_tensor(tensors, p + ".proj.weight");
        cf.proj_b = require_tensor(tensors, p + ".proj.bias");

        // DDSConv inside this ConvFlow
        cf.dds.layers.resize(hp.n_sdp_dds_layers);
        for (uint32_t di = 0; di < hp.n_sdp_dds_layers; di++) {
            auto& dl = cf.dds.layers[di];
            std::string dp = p + ".convs";
            dl.conv_sep_w = require_tensor(tensors, dp + ".convs_sep." + std::to_string(di) + ".weight");
            dl.conv_sep_b = require_tensor(tensors, dp + ".convs_sep." + std::to_string(di) + ".bias");
            dl.conv_1x1_w = require_tensor(tensors, dp + ".convs_1x1." + std::to_string(di) + ".weight");
            dl.conv_1x1_b = require_tensor(tensors, dp + ".convs_1x1." + std::to_string(di) + ".bias");
            dl.norm1_g = require_tensor(tensors, dp + ".norms_1." + std::to_string(di) + ".gamma");
            dl.norm1_b = require_tensor(tensors, dp + ".norms_1." + std::to_string(di) + ".beta");
            dl.norm2_g = require_tensor(tensors, dp + ".norms_2." + std::to_string(di) + ".gamma");
            dl.norm2_b = require_tensor(tensors, dp + ".norms_2." + std::to_string(di) + ".beta");
        }
    }

    // ── Residual Coupling Flow ──
    w.flow_blocks.resize(hp.n_flow_blocks);
    int flow_indices[] = {0, 2, 4, 6, 8, 10}; // even indices
    for (uint32_t fi = 0; fi < hp.n_flow_blocks; fi++) {
        int idx = flow_indices[fi];
        auto& fb = w.flow_blocks[fi];
        std::string p = "flow.flows." + std::to_string(idx);
        fb.pre_w = require_tensor(tensors, p + ".pre.weight");
        fb.pre_b = require_tensor(tensors, p + ".pre.bias");
        fb.post_w = require_tensor(tensors, p + ".post.weight");
        fb.post_b = require_tensor(tensors, p + ".post.bias");

        fb.wn.resize(hp.n_wn_layers);
        for (uint32_t wi = 0; wi < hp.n_wn_layers; wi++) {
            auto& wn = fb.wn[wi];
            std::string ep = p + ".enc";
            wn.in_w = require_tensor(tensors, ep + ".in_layers." + std::to_string(wi) + ".weight");
            wn.in_b = require_tensor(tensors, ep + ".in_layers." + std::to_string(wi) + ".bias");
            wn.res_w = require_tensor(tensors, ep + ".res_skip_layers." + std::to_string(wi) + ".weight");
            wn.res_b = require_tensor(tensors, ep + ".res_skip_layers." + std::to_string(wi) + ".bias");
        }
    }

    // ── HiFi-GAN Decoder ──
    w.dec_conv_pre_w = require_tensor(tensors, "dec.conv_pre.weight");
    w.dec_conv_pre_b = require_tensor(tensors, "dec.conv_pre.bias");
    w.dec_conv_post_w = require_tensor(tensors, "dec.conv_post.weight");

    w.dec_ups.resize(hp.n_upsample_stages);
    for (uint32_t i = 0; i < hp.n_upsample_stages; i++) {
        std::string p = "dec.ups." + std::to_string(i);
        w.dec_ups[i].w = require_tensor(tensors, p + ".weight");
        w.dec_ups[i].b = require_tensor(tensors, p + ".bias");
    }

    int n_resblocks = (int)hp.n_upsample_stages * (int)hp.resblock_kernels.size();
    w.dec_resblocks.resize(n_resblocks);
    for (int i = 0; i < n_resblocks; i++) {
        std::string p = "dec.resblocks." + std::to_string(i);
        w.dec_resblocks[i].conv0_w = require_tensor(tensors, p + ".convs.0.weight");
        w.dec_resblocks[i].conv0_b = require_tensor(tensors, p + ".convs.0.bias");
        w.dec_resblocks[i].conv1_w = require_tensor(tensors, p + ".convs.1.weight");
        w.dec_resblocks[i].conv1_b = require_tensor(tensors, p + ".convs.1.bias");
    }

    return true;
}

// ── Public API ─────────────────────────────────────────────────────

struct piper_tts_params piper_tts_default_params(void) {
    struct piper_tts_params p;
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.noise_scale = -1.0f; // -1 = use model default
    p.length_scale = -1.0f;
    p.noise_w = -1.0f;
    p.speaker_id = 0;
    return p;
}

struct piper_tts_context* piper_tts_init_from_file(const char* path_model, struct piper_tts_params params) {
    auto* ctx = new piper_tts_context();
    ctx->verbosity = params.verbosity;
    ctx->n_threads = params.n_threads;

    // Backend init
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "piper_tts: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, params.n_threads);

    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ctx->backend_cpu;
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;

    if (!load_weights(ctx, path_model)) {
        piper_tts_free(ctx);
        return nullptr;
    }

    // Create backend scheduler
    {
        ggml_backend_t backends[2];
        int n_be = 0;
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }

    // Apply params (use model defaults if -1)
    ctx->noise_scale = params.noise_scale >= 0 ? params.noise_scale : ctx->hp.noise_scale;
    ctx->length_scale = params.length_scale >= 0 ? params.length_scale : ctx->hp.length_scale;
    ctx->noise_w = params.noise_w >= 0 ? params.noise_w : ctx->hp.noise_w;
    ctx->speaker_id = params.speaker_id;
    ctx->espeak_voice = ctx->hp.espeak_voice;

    if (ctx->verbosity >= 1) {
        fprintf(stderr,
                "piper_tts: loaded %s (hidden=%u, enc=%u layers, "
                "flow=%u blocks, sr=%u, speakers=%u)\n",
                path_model, ctx->hp.hidden_channels, ctx->hp.n_layers_enc, ctx->hp.n_flow_blocks, ctx->hp.sample_rate,
                ctx->hp.num_speakers);
    }

    return ctx;
}

void piper_tts_free(struct piper_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->w_buf)
        ggml_backend_buffer_free(ctx->w_buf);
    if (ctx->w_ctx)
        ggml_free(ctx->w_ctx);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

int piper_tts_synthesize(struct piper_tts_context* ctx, const char* text, float** pcm_out, int* sample_rate_out) {
    if (!ctx || !text || !pcm_out)
        return 0;

    // 1. Phonemize
    std::string ipa;
    if (!phonemize_espeak(ctx->espeak_voice, text, ipa)) {
        fprintf(stderr, "piper_tts: phonemization failed for '%s'\n", text);
        return 0;
    }
    if (ctx->verbosity >= 2) {
        fprintf(stderr, "piper_tts: IPA: %s\n", ipa.c_str());
    }

    return piper_tts_synthesize_phonemes(ctx, ipa.c_str(), pcm_out, sample_rate_out);
}

int piper_tts_synthesize_phonemes(struct piper_tts_context* ctx, const char* ipa_phonemes, float** pcm_out,
                                  int* sample_rate_out) {
    if (!ctx || !ipa_phonemes || !pcm_out)
        return 0;

    // 1. Encode phonemes to IDs
    std::vector<int64_t> phoneme_ids = encode_phonemes(ipa_phonemes, ctx->pmap);
    int T = (int)phoneme_ids.size();
    dump_stage_i64(ctx, "phoneme_ids", phoneme_ids.data(), phoneme_ids.size());

    if (ctx->verbosity >= 2) {
        fprintf(stderr, "piper_tts: %d phoneme IDs\n", T);
    }

    // 2. Text encoder
    std::vector<float> enc_out, enc_mean, enc_logvar;
    text_encoder_forward(ctx, phoneme_ids, enc_out, enc_mean, enc_logvar);

    int C = (int)ctx->hp.inter_channels;

    if (enc_mean.empty()) {
        fprintf(stderr, "piper_tts: text encoder produced empty output\n");
        return 0;
    }

    if (ctx->verbosity >= 2) {
        fprintf(stderr, "piper_tts: enc_mean size=%zu, enc_logvar size=%zu\n", enc_mean.size(), enc_logvar.size());
        if (!enc_mean.empty()) {
            fprintf(stderr, "piper_tts: enc_mean[0:4] = %.4f %.4f %.4f %.4f\n", enc_mean[0], enc_mean[1], enc_mean[2],
                    enc_mean[3]);
        }
    }

    // 3. Duration prediction — SDP takes raw encoder output, not projected mean
    std::vector<float> log_dur;
    sdp_forward(ctx, enc_out, T, ctx->noise_w, log_dur);

    // Convert log-durations to integer durations
    std::vector<int> durations(T);
    int total_dur = 0;
    for (int t = 0; t < T; t++) {
        int d = (int)ceilf(expf(log_dur[t]) * ctx->length_scale);
        if (d < 1)
            d = 1;
        durations[t] = d;
        total_dur += d;
    }

    // Dump durations as float for comparison
    {
        std::vector<float> dur_f(T);
        for (int t = 0; t < T; t++)
            dur_f[t] = (float)durations[t];
        dump_stage(ctx, "durations", dur_f.data(), dur_f.size());
    }

    if (ctx->verbosity >= 2) {
        fprintf(stderr, "piper_tts: total duration = %d frames\n", total_dur);
    }

    // 4. Expand encoder outputs by duration alignment
    std::vector<float> z_mean, z_logvar;
    duration_align(enc_mean, durations, C, T, z_mean);
    duration_align(enc_logvar, durations, C, T, z_logvar);

    // 5. Sample latent z ~ N(mean, exp(logvar) * noise_scale)
    int T_latent = total_dur;
    std::vector<float> z(C * T_latent);
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < C * T_latent; i++) {
            float noise = dist(gen) * ctx->noise_scale;
            z[i] = z_mean[i] + noise * expf(z_logvar[i]);
        }
    }
    dump_stage(ctx, "z_p", z.data(), z.size());

    // 6. Residual coupling flow (inverse)
    flow_inverse(ctx, z, T_latent);
    dump_stage(ctx, "z_dec", z.data(), z.size());

    // 7. HiFi-GAN decode
    std::vector<float> pcm;
    if (!hifigan_decode(ctx, z, T_latent, pcm)) {
        return 0;
    }
    dump_stage(ctx, "audio", pcm.data(), pcm.size());

    // 8. Return PCM
    int n_samples = (int)pcm.size();
    *pcm_out = (float*)malloc(n_samples * sizeof(float));
    memcpy(*pcm_out, pcm.data(), n_samples * sizeof(float));
    if (sample_rate_out)
        *sample_rate_out = (int)ctx->hp.sample_rate;

    if (ctx->verbosity >= 1) {
        float dur_sec = (float)n_samples / (float)ctx->hp.sample_rate;
        fprintf(stderr, "piper_tts: synthesized %.2f s (%d samples @ %u Hz)\n", dur_sec, n_samples,
                ctx->hp.sample_rate);
    }

    return n_samples;
}

void piper_tts_set_language(struct piper_tts_context* ctx, const char* espeak_voice) {
    if (ctx && espeak_voice)
        ctx->espeak_voice = espeak_voice;
}

void piper_tts_set_noise_scale(struct piper_tts_context* ctx, float v) {
    if (ctx)
        ctx->noise_scale = v;
}

void piper_tts_set_length_scale(struct piper_tts_context* ctx, float v) {
    if (ctx)
        ctx->length_scale = v;
}

void piper_tts_set_noise_w(struct piper_tts_context* ctx, float v) {
    if (ctx)
        ctx->noise_w = v;
}

void piper_tts_set_speaker_id(struct piper_tts_context* ctx, int id) {
    if (ctx)
        ctx->speaker_id = id;
}

int piper_tts_sample_rate(const struct piper_tts_context* ctx) {
    return ctx ? (int)ctx->hp.sample_rate : 22050;
}

int piper_tts_num_speakers(const struct piper_tts_context* ctx) {
    return ctx ? (int)ctx->hp.num_speakers : 1;
}

const char* piper_tts_espeak_voice(const struct piper_tts_context* ctx) {
    return ctx ? ctx->espeak_voice.c_str() : "en-us";
}

bool piper_tts_has_espeak(void) {
#ifdef CRISPASR_HAVE_ESPEAK_NG
    return true;
#else
    // Check if espeak-ng binary is on $PATH
#ifdef _WIN32
    FILE* fp = _popen("espeak-ng --version 2>NUL", "r");
#else
    FILE* fp = popen("espeak-ng --version 2>/dev/null", "r");
#endif
    if (!fp)
        return false;
    char buf[128];
    bool ok = fgets(buf, sizeof(buf), fp) != nullptr;
#ifdef _WIN32
    _pclose(fp);
#else
    pclose(fp);
#endif
    return ok;
#endif
}

void piper_tts_set_dump_dir(struct piper_tts_context* ctx, const char* dir) {
    if (ctx && dir)
        ctx->dump_dir = dir;
}
