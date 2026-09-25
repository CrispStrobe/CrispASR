// nemotron3_diar.cpp — NVIDIA Nemotron-3-Diarization (streaming Sortformer v3), #466.
//
// Reference: transformers Nemotron3DiarizationForAudioFrameClassification
// (modeling_nemotron3_diarization.py) — graph, offline chunk loop and the
// Arrival-Order Speaker Cache are ported from it line by line. NVIDIA's
// Apache-2.0 NeMo-Speech.cpp implements the same model on a patched ggml; this
// file uses stock ggml ops only.
//
// Per chunk (graph):
//   x = LN_in(cache | fifo | chunk embeddings)          positions restart at 0
//   31 x pre-LN layer: x += o_proj(attn(RoPE(q), RoPE(k), v)); x += fc2(gelu(fc1(LN2 x)))
//   h = proj(LN_final(x))                                  512 -> 192
//   h = subpixel_conv1d(h)                                 192 -> 8 x 192 (k=3, pad 1)
//   logits = out(relu(dense(relu(h))))                     one row per 10 ms
// Host: mel, 8x feature stacking + projection, the chunk loop and the cache.

#include "nemotron3_diar.h"

#include "core/crispasr_env.h"
#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/mel.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool n3d_bench() {
    static const int v = [] {
        const char* e = crispasr_env::get("CRISPASR_NEMOTRON3_DIAR_BENCH");
        return (e && *e && *e != '0') ? 1 : 0;
    }();
    return v != 0;
}

// Attention: ggml_flash_attn_ext or the manual softmax(QK^T)V. Measured on the
// 60 s AMI clip (#466): on the CPU flash is exact in every mode and 11-13 %
// faster, so it is the CPU default; on a T4 it is exact offline (-42 % encoder
// time) but flips a few borderline frames in streaming (F16 accumulation), and
// GPUs differ in honouring GGML_PREC_F32, so GPU runs keep the manual path.
// CRISPASR_NEMOTRON3_DIAR_ATTN=flash|manual overrides. (Contiguous Q/K and a
// BLAS/ACCEL device measured no gain and were dropped.)
int n3d_attn_override() { // -1 unset, 0 manual, 1 flash
    static const int v = [] {
        const char* e = crispasr_env::get("CRISPASR_NEMOTRON3_DIAR_ATTN");
        if (e && !std::strcmp(e, "flash"))
            return 1;
        if (e && !std::strcmp(e, "manual"))
            return 0;
        return -1;
    }();
    return v;
}

// Bench accumulators: graph build + alloc vs compute, summed over chunks.
struct n3d_bench_acc {
    double build_ms = 0, compute_ms = 0;
    double mel_ms = 0, embed_ms = 0, update_ms = 0; // streaming, outside the encoder graph
    int n = 0;
};
n3d_bench_acc& n3d_acc() {
    static n3d_bench_acc a;
    return a;
}

struct n3d_layer {
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    ggml_tensor* qkv_w = nullptr;
    ggml_tensor *o_w = nullptr, *o_b = nullptr;
    ggml_tensor *norm2_w = nullptr, *norm2_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
    ggml_tensor *fc2_w = nullptr, *fc2_b = nullptr;
};

struct n3d_hparams {
    int d = 512, n_layers = 31, n_heads = 8, d_ff = 2048, subsample = 8, n_mels = 128;
    float rope_base = 10000.0f;
    int head_hidden = 192, n_spk = 8;
    int sr = 16000, n_fft = 512, win = 400, hop = 160;
    float preemph = 0.97f, log_guard = 5.960464477539063e-08f;
    // offline chunk schedule (transformers config.json)
    int chunk_len = 340, chunk_rc = 40, fifo_len = 40, update_period = 300;
    // streaming schedule: FIFO / update period (streaming_config) and the
    // processor's modes, (chunk_len, chunk_right_context) in encoder frames
    int s_fifo_len = 264, s_update_period = 222;
    int s_modes[3][2] = {{9, 4}, {6, 2}, {3, 1}};
    // speaker cache policy (streaming_config)
    int cache_len = 264, n_sil = 1;
    float thr = 0.25f, boost_latest = 0.05f, strong_rate = 0.75f, weak_rate = 1.5f, min_pos_rate = 0.5f;
};

} // namespace

struct nemotron3_diar_context {
    nemotron3_diar_params params{};
    n3d_hparams hp;
    int mode = -1; // -1 offline, else index into hp.s_modes
    ggml_backend_t backend = nullptr, backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    core_gguf::WeightLoad wl;
    std::vector<uint8_t> meta;
    // weights
    ggml_tensor* fb = nullptr;
    ggml_tensor* proj_in_w = nullptr;
    ggml_tensor *embed_norm_w = nullptr, *embed_norm_b = nullptr;
    ggml_tensor *final_norm_w = nullptr, *final_norm_b = nullptr;
    std::vector<n3d_layer> layers;
    ggml_tensor *enc_proj_w = nullptr, *enc_proj_b = nullptr;
    ggml_tensor *up_w = nullptr, *up_b = nullptr;
    ggml_tensor *dense_w = nullptr, *dense_b = nullptr;
    ggml_tensor *out_w = nullptr, *out_b = nullptr;
    ggml_tensor* sil = nullptr;
    // host copies
    std::vector<float> fb_host, window, sil_host;
};

namespace {

void fft_r2c(const float* in, int N, float* out) {
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
    // Twiddles per stage, cached: recomputing cos/sin in every butterfly cost
    // 14 ms per streaming chunk (#466, T4 low_latency: 1.2 s of 4.2 s). Same
    // expressions as before, so the values - and the spectrum - are bit-identical.
    thread_local int tw_n = 0;
    thread_local std::vector<double> tw; // stage len at offset len/2 - 1: (cos, sin) pairs
    if (tw_n != N) {
        tw.assign((size_t)2 * N, 0.0);
        for (int len = 2; len <= N; len <<= 1) {
            const double ang = -2.0 * M_PI / (double)len;
            for (int j = 0; j < len / 2; j++) {
                tw[(size_t)2 * (len / 2 - 1 + j)] = std::cos(ang * j);
                tw[(size_t)2 * (len / 2 - 1 + j) + 1] = std::sin(ang * j);
            }
        }
        tw_n = N;
    }
    for (int len = 2; len <= N; len <<= 1) {
        const double* w = tw.data() + (size_t)2 * (len / 2 - 1);
        for (int i = 0; i < N; i += len) {
            for (int j = 0; j < len / 2; j++) {
                const double wre = w[2 * j], wim = w[2 * j + 1];
                const int a = i + j, b = i + j + len / 2;
                const float bre = out[2 * b], bim = out[2 * b + 1];
                const float tre = (float)(wre * bre - wim * bim), tim = (float)(wre * bim + wim * bre);
                const float are = out[2 * a], aim = out[2 * a + 1];
                out[2 * a] = are + tre;
                out[2 * a + 1] = aim + tim;
                out[2 * b] = are - tre;
                out[2 * b + 1] = aim - tim;
            }
        }
    }
}

std::vector<float> tensor_to_f32(ggml_tensor* t) {
    std::vector<float> out((size_t)ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    } else {
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        ggml_get_type_traits(t->type)->to_float(raw.data(), out.data(), (int64_t)out.size());
    }
    return out;
}

// ---------------------------------------------------------------------------
// Arrival-Order Speaker Cache + FIFO (Nemotron3DiarizationSpeakerCache), B = 1.
// Rows are frames; embeds are d-wide, probs n_spk-wide.
// ---------------------------------------------------------------------------
struct SpeakerCache {
    int d = 0, S = 0, cache_len = 0, fifo_len = 0, update_period = 0, n_sil = 0, factor = 8;
    float thr = 0.25f, boost_latest = 0.05f;
    int min_pos = 0, n_strong = 0, n_weak = 0;
    std::vector<float> embeds, probs, fifo; // capacity cache_len / cache_len / fifo_len rows
    int n_cache = 0, n_fifo = 0;
    bool compressed = false;

    void init(const n3d_hparams& hp, int fifo_length, int update) {
        d = hp.d;
        S = hp.n_spk;
        cache_len = hp.cache_len;
        fifo_len = fifo_length;
        update_period = update;
        n_sil = hp.n_sil;
        factor = hp.subsample;
        thr = hp.thr;
        boost_latest = hp.boost_latest;
        const int budget = cache_len / S - n_sil;
        min_pos = (int)std::floor(budget * hp.min_pos_rate);
        n_strong = (int)std::floor(budget * hp.strong_rate);
        n_weak = (int)std::floor(budget * hp.weak_rate);
        embeds.assign((size_t)cache_len * d, 0.0f);
        probs.assign((size_t)cache_len * S, 0.0f);
        fifo.assign((size_t)std::max(fifo_len, 1) * d, 0.0f);
        n_cache = n_fifo = 0;
        compressed = false;
    }

    // cat(cache[:n_cache], fifo[:n_fifo]) -> rows
    void get_embeds(std::vector<float>& out) const {
        out.clear();
        out.insert(out.end(), embeds.begin(), embeds.begin() + (size_t)n_cache * d);
        out.insert(out.end(), fifo.begin(), fifo.begin() + (size_t)n_fifo * d);
    }

    // speaker probabilities at the encoder rate: avg-pool of sigmoid(logits) over `factor` rows, x mask
    std::vector<float> pool_probs(const std::vector<float>& logits, int n_rows,
                                  const std::vector<uint8_t>& mask) const {
        std::vector<float> p((size_t)n_rows * S, 0.0f);
        for (int r = 0; r < n_rows; r++) {
            for (int s = 0; s < S; s++) {
                double acc = 0.0;
                for (int k = 0; k < factor; k++) {
                    const float l = logits[((size_t)r * factor + k) * S + s];
                    acc += 1.0 / (1.0 + std::exp(-(double)l));
                }
                p[(size_t)r * S + s] = mask.empty() || mask[r] ? (float)(acc / factor) : 0.0f;
            }
        }
        return p;
    }

    int num_popped(int n) const {
        if (n <= fifo_len)
            return 0;
        return std::min(std::max(update_period, n - fifo_len), n);
    }

    // scores [F][S]
    std::vector<float> frame_scores(const std::vector<float>& pr, int F) const {
        const float NINF = -std::numeric_limits<float>::infinity();
        std::vector<float> sc((size_t)F * S);
        const double ln_half = std::log(0.5);
        for (int f = 0; f < F; f++) {
            double sum_lc = 0.0;
            for (int s = 0; s < S; s++)
                sum_lc += std::log(std::max(1.0f - pr[(size_t)f * S + s], thr));
            for (int s = 0; s < S; s++) {
                const float p = pr[(size_t)f * S + s];
                const double lp = std::log(std::max(p, thr)), lc = std::log(std::max(1.0f - p, thr));
                float v = (float)(lp - lc + sum_lc - ln_half);
                if (!(p > 0.5f))
                    v = NINF;
                sc[(size_t)f * S + s] = v;
            }
        }
        for (int s = 0; s < S; s++) {
            int n_pos = 0;
            for (int f = 0; f < F; f++)
                n_pos += sc[(size_t)f * S + s] > 0.0f;
            if (n_pos < min_pos)
                continue;
            for (int f = 0; f < F; f++) {
                float& v = sc[(size_t)f * S + s];
                const bool is_speech = pr[(size_t)f * S + s] > 0.5f;
                if (is_speech && !(v > 0.0f))
                    v = NINF;
            }
        }
        return sc;
    }

    // torch.topk(scores, k, dim=frames) per speaker, += boost
    void boost(std::vector<float>& sc, int F, int k, float b) const {
        if (k <= 0)
            return;
        std::vector<int> idx(F);
        for (int s = 0; s < S; s++) {
            std::iota(idx.begin(), idx.end(), 0);
            const int kk = std::min(k, F);
            std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int c) {
                const float va = sc[(size_t)a * S + s], vc = sc[(size_t)c * S + s];
                return va > vc || (va == vc && a < c);
            });
            for (int i = 0; i < kk; i++)
                sc[(size_t)idx[i] * S + s] += b;
        }
    }

    void compress(std::vector<float>& e, std::vector<float>& pr, int F, const std::vector<float>& sil) {
        const float NINF = -std::numeric_limits<float>::infinity(), PINF = std::numeric_limits<float>::infinity();
        std::vector<float> sc = frame_scores(pr, F);
        for (int f = cache_len; f < F; f++)
            for (int s = 0; s < S; s++)
                sc[(size_t)f * S + s] += boost_latest;
        boost(sc, F, n_strong, (float)(-2.0 * std::log(0.5)));
        boost(sc, F, n_weak, (float)(-std::log(0.5)));
        const int Fs = F + n_sil; // scored frames: + silence rows (+inf)
        // flat speaker-major scores
        std::vector<std::pair<float, int>> flat;
        flat.reserve((size_t)Fs * S);
        for (int s = 0; s < S; s++)
            for (int f = 0; f < Fs; f++)
                flat.push_back({f < F ? sc[(size_t)f * S + s] : PINF, s * Fs + f});
        const int K = cache_len;
        std::partial_sort(flat.begin(), flat.begin() + K, flat.end(), [](const auto& a, const auto& b) {
            return a.first > b.first || (a.first == b.first && a.second < b.second);
        });
        const int sentinel = Fs * S;
        std::vector<int> top(K);
        for (int i = 0; i < K; i++)
            top[i] = flat[i].first == NINF ? sentinel : flat[i].second;
        std::sort(top.begin(), top.end());
        std::vector<float> ne((size_t)K * d), np((size_t)K * S, 0.0f);
        for (int i = 0; i < K; i++) {
            const int fi = top[i] == sentinel ? F : std::min(top[i] % Fs, F);
            if (fi == F) { // silence slot: learned embedding, zero probs
                std::memcpy(&ne[(size_t)i * d], sil.data(), (size_t)d * sizeof(float));
            } else {
                std::memcpy(&ne[(size_t)i * d], &e[(size_t)fi * d], (size_t)d * sizeof(float));
                std::memcpy(&np[(size_t)i * S], &pr[(size_t)fi * S], (size_t)S * sizeof(float));
            }
        }
        e.swap(ne);
        pr.swap(np);
    }

    // chunk_in: [N][d] (cache | fifo | chunk | look-ahead), logits: [N*factor][S]
    void update(const std::vector<float>& chunk_in, int N, const std::vector<float>& logits, int n_chunk,
                const std::vector<uint8_t>& mask, const std::vector<float>& sil) {
        const std::vector<float> pr = pool_probs(logits, N, mask);
        const int start = n_cache + n_fifo;
        // fifo_embeds = cat(fifo[:n_fifo], chunk rows)
        std::vector<float> fe(fifo.begin(), fifo.begin() + (size_t)n_fifo * d);
        fe.insert(fe.end(), chunk_in.begin() + (size_t)start * d, chunk_in.begin() + (size_t)(start + n_chunk) * d);
        int n_fe = n_fifo + n_chunk;
        const int popped = num_popped(n_fe);
        if (popped) {
            // cache probs: stored ones for a compressed cache, this step's estimate otherwise
            std::vector<float> ce(embeds.begin(), embeds.begin() + (size_t)n_cache * d);
            std::vector<float> cp;
            if (compressed)
                cp.assign(probs.begin(), probs.begin() + (size_t)n_cache * S);
            else
                cp.assign(pr.begin(), pr.begin() + (size_t)n_cache * S);
            ce.insert(ce.end(), fe.begin(), fe.begin() + (size_t)popped * d);
            cp.insert(cp.end(), pr.begin() + (size_t)n_cache * S, pr.begin() + (size_t)(n_cache + popped) * S);
            fe.erase(fe.begin(), fe.begin() + (size_t)popped * d);
            n_fe -= popped;
            int n_ce = n_cache + popped;
            if (n_ce > cache_len) {
                compress(ce, cp, n_ce, sil);
                n_ce = cache_len;
                compressed = true;
            }
            n_cache = n_ce;
            std::memcpy(embeds.data(), ce.data(), (size_t)n_cache * d * sizeof(float));
            std::memcpy(probs.data(), cp.data(), (size_t)n_cache * S * sizeof(float));
        }
        n_fifo = n_fe;
        if ((size_t)n_fifo * d > fifo.size())
            fifo.resize((size_t)n_fifo * d);
        std::memcpy(fifo.data(), fe.data(), (size_t)n_fifo * d * sizeof(float));
    }
};

bool load(nemotron3_diar_context* c, const char* path) {
    auto& hp = c->hp;
    if (gguf_context* g = core_gguf::open_metadata(path)) {
        const std::string arch = core_gguf::kv_str(g, "general.architecture", "");
        if (arch != "sortformer") {
            fprintf(stderr, "nemotron3_diar: '%s' is '%s', expected a sortformer GGUF\n", path, arch.c_str());
            core_gguf::free_metadata(g);
            return false;
        }
        const std::string E = "sortformer.encoder.";
        hp.d = (int)core_gguf::kv_u32(g, (E + "d_model").c_str(), hp.d);
        hp.n_layers = (int)core_gguf::kv_u32(g, (E + "n_layers").c_str(), hp.n_layers);
        hp.n_heads = (int)core_gguf::kv_u32(g, (E + "n_heads").c_str(), hp.n_heads);
        hp.d_ff = (int)core_gguf::kv_u32(g, (E + "d_ff").c_str(), hp.d_ff);
        hp.subsample = (int)core_gguf::kv_u32(g, (E + "subsampling_factor").c_str(), hp.subsample);
        hp.n_mels = (int)core_gguf::kv_u32(g, (E + "feat_in").c_str(), hp.n_mels);
        hp.rope_base = core_gguf::kv_f32(g, (E + "rope_base").c_str(), hp.rope_base);
        hp.head_hidden = (int)core_gguf::kv_u32(g, "sortformer.transformer.hidden_size", hp.head_hidden);
        hp.n_spk = (int)core_gguf::kv_u32(g, "sortformer.num_speakers", hp.n_spk);
        hp.sr = (int)core_gguf::kv_u32(g, "sortformer.preprocessor.sample_rate", hp.sr);
        hp.n_fft = (int)core_gguf::kv_u32(g, "sortformer.preprocessor.n_fft", hp.n_fft);
        hp.win = (int)std::lround(core_gguf::kv_f32(g, "sortformer.preprocessor.window_size", 0.025f) * hp.sr);
        hp.hop = (int)std::lround(core_gguf::kv_f32(g, "sortformer.preprocessor.window_stride", 0.01f) * hp.sr);
        hp.preemph = core_gguf::kv_f32(g, "sortformer.preprocessor.preemph", hp.preemph);
        hp.log_guard = core_gguf::kv_f32(g, "sortformer.preprocessor.log_zero_guard", hp.log_guard);
        const std::string S = "sortformer.scoring.";
        hp.n_sil = (int)core_gguf::kv_u32(g, (S + "spkcache_sil_frames_per_spk").c_str(), hp.n_sil);
        hp.thr = core_gguf::kv_f32(g, (S + "pred_score_threshold").c_str(), hp.thr);
        hp.boost_latest = core_gguf::kv_f32(g, (S + "scores_boost_latest").c_str(), hp.boost_latest);
        hp.strong_rate = core_gguf::kv_f32(g, (S + "strong_boost_rate").c_str(), hp.strong_rate);
        hp.weak_rate = core_gguf::kv_f32(g, (S + "weak_boost_rate").c_str(), hp.weak_rate);
        hp.min_pos_rate = core_gguf::kv_f32(g, (S + "min_pos_scores_rate").c_str(), hp.min_pos_rate);
        hp.cache_len = (int)core_gguf::kv_u32(g, "sortformer.streaming.spkcache_len", hp.cache_len);
        // Offline schedule: transformers' (our converter writes it); NVIDIA's GGUF
        // lacks these keys, so the defaults (config.json values) apply.
        const std::string O = "nemotron3diar.offline.";
        hp.chunk_len = (int)core_gguf::kv_u32(g, (O + "chunk_len").c_str(), hp.chunk_len);
        hp.chunk_rc = (int)core_gguf::kv_u32(g, (O + "chunk_right_context").c_str(), hp.chunk_rc);
        hp.fifo_len = (int)core_gguf::kv_u32(g, (O + "fifo_len").c_str(), hp.fifo_len);
        hp.update_period = (int)core_gguf::kv_u32(g, (O + "spkcache_update_period").c_str(), hp.update_period);
        // Streaming schedule: only our converter's GGUF carries transformers' values.
        // NVIDIA's sortformer.streaming.fifo_len / update period (0 / 264) are a
        // different, NeMo-side schedule, so without our keys the defaults
        // (config.json streaming_config, processor_config.json modes) apply.
        static const char* k_modes[3] = {"low_latency", "very_low_latency", "ultra_low_latency"};
        if (core_gguf::kv_u32(g, "nemotron3diar.streaming.low_latency.chunk_len", 0) > 0) {
            hp.s_fifo_len = (int)core_gguf::kv_u32(g, "sortformer.streaming.fifo_len", hp.s_fifo_len);
            hp.s_update_period =
                (int)core_gguf::kv_u32(g, "sortformer.streaming.spkcache_update_period", hp.s_update_period);
            for (int m = 0; m < 3; m++) {
                const std::string P = std::string("nemotron3diar.streaming.") + k_modes[m] + ".";
                hp.s_modes[m][0] = (int)core_gguf::kv_u32(g, (P + "chunk_len").c_str(), hp.s_modes[m][0]);
                hp.s_modes[m][1] = (int)core_gguf::kv_u32(g, (P + "chunk_right_context").c_str(), hp.s_modes[m][1]);
            }
        }
        core_gguf::free_metadata(g);
    } else {
        return false;
    }
    if (!core_gguf::load_weights(path, c->backend, "nemotron3_diar", c->wl))
        return false;
    auto& T = c->wl.tensors;
    bool missing = false;
    auto req = [&](const std::string& n) {
        ggml_tensor* t = core_gguf::require(T, n.c_str(), "nemotron3_diar");
        missing |= t == nullptr;
        return t;
    };
    c->fb = req("preprocessor.fb");
    c->proj_in_w = req("encoder.pre_encode.proj.weight");
    c->embed_norm_w = req("encoder.embed_norm.weight");
    c->embed_norm_b = req("encoder.embed_norm.bias");
    c->final_norm_w = req("encoder.final_norm.weight");
    c->final_norm_b = req("encoder.final_norm.bias");
    c->layers.resize(hp.n_layers);
    for (int i = 0; i < hp.n_layers; i++) {
        const std::string p = "encoder.layers." + std::to_string(i) + ".";
        auto& L = c->layers[i];
        L.norm1_w = req(p + "norm1.weight");
        L.norm1_b = req(p + "norm1.bias");
        L.qkv_w = req(p + "attn.w_qkv.weight");
        L.o_w = req(p + "attn.out_proj.weight");
        L.o_b = req(p + "attn.out_proj.bias");
        L.norm2_w = req(p + "norm2.weight");
        L.norm2_b = req(p + "norm2.bias");
        L.fc1_w = req(p + "ffn.net.0.weight");
        L.fc1_b = req(p + "ffn.net.0.bias");
        L.fc2_w = req(p + "ffn.net.3.weight");
        L.fc2_b = req(p + "ffn.net.3.bias");
    }
    c->enc_proj_w = req("encoder_proj.weight");
    c->enc_proj_b = req("encoder_proj.bias");
    c->up_w = req("subpixel_upsample.weight");
    c->up_b = req("subpixel_upsample.bias");
    c->dense_w = req("head.first_hidden_to_hidden.weight");
    c->dense_b = req("head.first_hidden_to_hidden.bias");
    c->out_w = req("head.single_hidden_to_spks.weight");
    c->out_b = req("head.single_hidden_to_spks.bias");
    c->sil = req("learnable_sil_emb");
    if (missing)
        return false;
    c->fb_host = tensor_to_f32(c->fb);
    c->sil_host = tensor_to_f32(c->sil);
    // torch.hann_window(win, periodic=False): symmetric
    c->window.resize(hp.win);
    for (int i = 0; i < hp.win; i++)
        c->window[i] = (float)(0.5 - 0.5 * std::cos(2.0 * M_PI * i / (hp.win - 1)));
    return true;
}

ggml_tensor* layer_norm(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    return ggml_add(g, ggml_mul(g, ggml_norm(g, x, 1e-5f), w), b);
}

// One chunk: x_in [d][N] -> logits [S][N*factor]
bool run_chunk(nemotron3_diar_context* c, const std::vector<float>& x_rows, int N, const std::vector<uint8_t>& mask,
               std::vector<float>& logits) {
    const auto& hp = c->hp;
    const int d = hp.d, H = hp.n_heads, hd = d / H, S = hp.n_spk, F = hp.subsample, Hh = hp.head_hidden;
    const double t_start = n3d_bench() ? now_ms() : 0.0;
    ggml_init_params ip = {c->meta.size(), c->meta.data(), true};
    ggml_context* g = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(g, 8192, false);

    ggml_tensor* x = ggml_new_tensor_2d(g, GGML_TYPE_F32, d, N);
    ggml_set_name(x, "x_in");
    ggml_set_input(x);
    ggml_tensor* pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, N);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);
    const bool use_mask = std::find(mask.begin(), mask.end(), 0) != mask.end();
    const bool flash = n3d_attn_override() >= 0 ? n3d_attn_override() == 1 : c->backend == c->backend_cpu;
    ggml_tensor* kq_mask = nullptr;
    if (use_mask && !flash) {
        kq_mask = ggml_new_tensor_2d(g, GGML_TYPE_F32, N, N);
        ggml_set_name(kq_mask, "kq_mask");
        ggml_set_input(kq_mask);
    }

    // flash attention takes an F16 mask [n_kv, n_batch] instead
    ggml_tensor* kq_mask_fa = nullptr;
    if (flash && use_mask) {
        kq_mask_fa = ggml_new_tensor_2d(g, GGML_TYPE_F16, N, N);
        ggml_set_name(kq_mask_fa, "kq_mask_fa");
        ggml_set_input(kq_mask_fa);
    }

    ggml_tensor* cur = layer_norm(g, x, c->embed_norm_w, c->embed_norm_b);
    const float scale = 1.0f / std::sqrt((float)hd);
    for (int il = 0; il < hp.n_layers; il++) {
        const n3d_layer& L = c->layers[il];
        ggml_tensor* h = layer_norm(g, cur, L.norm1_w, L.norm1_b);
        ggml_tensor* qkv = ggml_mul_mat(g, L.qkv_w, h); // [3d, N]
        const size_t es = ggml_element_size(qkv);
        ggml_tensor* Q = ggml_view_3d(g, qkv, hd, H, N, hd * es, qkv->nb[1], 0);
        ggml_tensor* K = ggml_view_3d(g, qkv, hd, H, N, hd * es, qkv->nb[1], (size_t)d * es);
        ggml_tensor* V = ggml_view_3d(g, qkv, hd, H, N, hd * es, qkv->nb[1], (size_t)2 * d * es);
        Q = ggml_rope_ext(g, ggml_cont(g, Q), pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, hp.rope_base, 1.0f, 0.0f, 1.0f,
                          0.0f, 0.0f);
        K = ggml_rope_ext(g, ggml_cont(g, K), pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, hp.rope_base, 1.0f, 0.0f, 1.0f,
                          0.0f, 0.0f);
        Q = ggml_permute(g, Q, 0, 2, 1, 3); // [hd, N, H]
        K = ggml_permute(g, K, 0, 2, 1, 3); // [hd, N, H]
        ggml_tensor* o = nullptr;
        if (flash) {
            ggml_tensor* Vp = ggml_permute(g, V, 0, 2, 1, 3); // [hd, N, H]
            o = ggml_flash_attn_ext(g, Q, K, Vp, kq_mask_fa, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
            o = ggml_reshape_2d(g, o, d, N); // result [hd, H, N]
        } else {
            ggml_tensor* Vt = ggml_cont(g, ggml_permute(g, V, 1, 2, 0, 3)); // [N, hd, H]
            ggml_tensor* kq = ggml_mul_mat(g, K, Q);                        // [N_k, N_q, H]
            kq = ggml_soft_max_ext(g, kq, kq_mask, scale, 0.0f);
            o = ggml_mul_mat(g, Vt, kq);                      // [hd, N_q, H]
            o = ggml_cont(g, ggml_permute(g, o, 0, 2, 1, 3)); // [hd, H, N]
            o = ggml_reshape_2d(g, o, d, N);
        }
        o = ggml_add(g, ggml_mul_mat(g, L.o_w, o), L.o_b);
        cur = ggml_add(g, cur, o);
        h = layer_norm(g, cur, L.norm2_w, L.norm2_b);
        h = ggml_add(g, ggml_mul_mat(g, L.fc1_w, h), L.fc1_b);
        h = ggml_gelu_erf(g, h);
        h = ggml_add(g, ggml_mul_mat(g, L.fc2_w, h), L.fc2_b);
        cur = ggml_add(g, cur, h);
    }
    cur = layer_norm(g, cur, c->final_norm_w, c->final_norm_b);
    ggml_tensor* hh = ggml_add(g, ggml_mul_mat(g, c->enc_proj_w, cur), c->enc_proj_b); // [Hh, N]

    // Sub-pixel Conv1d(Hh -> Hh*F, k=3, pad=1) as im2col (F32) + matmul, i.e.
    // ggml_conv_1d's body without its F16 im2col.
    ggml_tensor* xt = ggml_cont(g, ggml_transpose(g, hh));                                  // [N, Hh]  (time fastest)
    ggml_tensor* col = ggml_im2col(g, c->up_w, xt, 1, 0, 1, 0, 1, 0, false, GGML_TYPE_F32); // [K*Hh, N]
    ggml_tensor* kw = c->up_w->type == GGML_TYPE_F32 ? c->up_w : ggml_cast(g, c->up_w, GGML_TYPE_F32);
    ggml_tensor* conv = ggml_mul_mat(g, ggml_reshape_2d(g, col, col->ne[0], col->ne[1] * col->ne[2]),
                                     ggml_reshape_2d(g, kw, kw->ne[0] * kw->ne[1], kw->ne[2])); // [N, Hh*F]
    conv = ggml_cont(g, ggml_transpose(g, conv));                                               // [Hh*F, N]
    conv = ggml_add(g, conv, c->up_b);
    ggml_tensor* up = ggml_reshape_2d(g, conv, Hh, (int64_t)N * F); // row (n*F+s) = channels [s*Hh, (s+1)*Hh)

    ggml_tensor* y = ggml_relu(g, up);
    y = ggml_add(g, ggml_mul_mat(g, c->dense_w, y), c->dense_b);
    y = ggml_relu(g, y);
    y = ggml_add(g, ggml_mul_mat(g, c->out_w, y), c->out_b); // [S, N*F]
    ggml_set_name(y, "logits");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);
    ggml_free(g);

    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "nemotron3_diar: graph alloc failed (N=%d)\n", N);
        return false;
    }
    const double t_built = n3d_bench() ? now_ms() : 0.0;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x_in"), x_rows.data(), 0, (size_t)d * N * sizeof(float));
    std::vector<int32_t> p(N);
    std::iota(p.begin(), p.end(), 0);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos"), p.data(), 0, p.size() * sizeof(int32_t));
    if (use_mask && kq_mask) {
        std::vector<float> m((size_t)N * N, 0.0f);
        for (int q = 0; q < N; q++)
            for (int k = 0; k < N; k++)
                if (!mask[k])
                    m[(size_t)q * N + k] = -INFINITY;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "kq_mask"), m.data(), 0, m.size() * sizeof(float));
    }
    if (kq_mask_fa) {
        std::vector<ggml_fp16_t> m((size_t)N * N, ggml_fp32_to_fp16(0.0f));
        const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < N; q++)
            for (int k = 0; k < N; k++)
                if (!mask[k])
                    m[(size_t)q * N + k] = ninf;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "kq_mask_fa"), m.data(), 0, m.size() * sizeof(ggml_fp16_t));
    }
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "nemotron3_diar: graph compute failed\n");
        return false;
    }
    if (n3d_bench()) {
        auto& a = n3d_acc();
        a.build_ms += t_built - t_start;
        a.compute_ms += now_ms() - t_built;
        a.n++;
    }
    ggml_tensor* lo = ggml_graph_get_tensor(gf, "logits");
    logits.resize((size_t)S * N * F);
    ggml_backend_tensor_get(lo, logits.data(), 0, logits.size() * sizeof(float));
    return true;
}

// stacked-projection embeddings [Ne][d] for mel [T][n_mels]
bool embed(nemotron3_diar_context* c, const std::vector<float>& mel, int T, std::vector<float>& out, int& Ne) {
    const auto& hp = c->hp;
    const int F = hp.subsample, M = hp.n_mels, d = hp.d;
    Ne = (T + F - 1) / F;
    std::vector<float> stacked((size_t)Ne * F * M, 0.0f); // zero-pads the last group
    std::memcpy(stacked.data(), mel.data(), (size_t)T * M * sizeof(float));
    ggml_init_params ip = {c->meta.size(), c->meta.data(), true};
    ggml_context* g = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(g, 64, false);
    ggml_tensor* s = ggml_new_tensor_2d(g, GGML_TYPE_F32, F * M, Ne);
    ggml_set_name(s, "stacked");
    ggml_set_input(s);
    ggml_tensor* e = ggml_mul_mat(g, c->proj_in_w, s); // [d, Ne]
    ggml_set_name(e, "embeds");
    ggml_set_output(e);
    ggml_build_forward_expand(gf, e);
    ggml_free(g);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf))
        return false;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "stacked"), stacked.data(), 0, stacked.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS)
        return false;
    out.resize((size_t)Ne * d);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "embeds"), out.data(), 0, out.size() * sizeof(float));
    return true;
}


} // namespace

extern "C" nemotron3_diar_params nemotron3_diar_default_params(void) {
    nemotron3_diar_params p;
    p.n_threads = 4;
    p.use_gpu = true;
    p.verbosity = 1;
    return p;
}

extern "C" nemotron3_diar_context* nemotron3_diar_init_from_file(const char* path, nemotron3_diar_params params) {
    auto* c = new nemotron3_diar_context();
    c->params = params;
    c->backend_cpu = core_cpu_backend::init();
    core_cpu_backend::set_n_threads(c->backend_cpu, params.n_threads);
    c->backend = params.use_gpu ? crispasr_init_gpu_backend() : nullptr;
    if (!c->backend)
        c->backend = c->backend_cpu;
    if (!load(c, path)) {
        nemotron3_diar_free(c);
        return nullptr;
    }
    ggml_backend_t be[2] = {c->backend, c->backend_cpu};
    const int n_be = c->backend == c->backend_cpu ? 1 : 2;
    c->sched = ggml_backend_sched_new(be + (n_be == 1 ? 1 : 0), nullptr, n_be, 8192, false, false);
    c->meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    if (params.verbosity >= 1)
        fprintf(stderr, "nemotron3_diar: %d layers d=%d, %d speakers, chunk %d+%d (fifo %d, cache %d), %s\n",
                c->hp.n_layers, c->hp.d, c->hp.n_spk, c->hp.chunk_len, c->hp.chunk_rc, c->hp.fifo_len, c->hp.cache_len,
                ggml_backend_name(c->backend));
    return c;
}

extern "C" void nemotron3_diar_free(nemotron3_diar_context* c) {
    if (!c)
        return;
    if (c->sched)
        ggml_backend_sched_free(c->sched);
    core_gguf::free_weights(c->wl);
    if (c->backend && c->backend != c->backend_cpu)
        ggml_backend_free(c->backend);
    if (c->backend_cpu)
        ggml_backend_free(c->backend_cpu);
    delete c;
}

extern "C" int nemotron3_diar_n_speakers(nemotron3_diar_context* c) {
    return c ? c->hp.n_spk : 0;
}

namespace {

struct n3d_chunk {
    int start, n, lookahead; // encoder frames: scored [start, start + n), attended up to + lookahead
};

// Streaming mode on a whole recording, the way the model card's driver feeds
// Nemotron3DiarizationProcessor: a first chunk of (chunk + look-ahead) * 8 mel
// frames (centred STFT), then uncentred chunks while one fits entirely in the
// audio, then a last chunk with every remaining frame, no look-ahead, scored
// whole (never re-split). The per-chunk features reproduce the full-utterance
// mel frame for frame, so only the frame count differs from offline mode:
// uncentred windows must lie inside the audio. Returns the mel frame count.
int n3d_streaming_plan(const n3d_hparams& hp, int mode, int n_samples, std::vector<n3d_chunk>* plan) {
    const int F = hp.subsample, cl = hp.s_modes[mode][0], rc = hp.s_modes[mode][1];
    const int n_chunk_mel = (cl + rc) * F;
    if (plan)
        plan->clear();
    const long first_samples = (long)(n_chunk_mel - 1) * hp.hop + hp.win / 2;
    if (n_samples < first_samples) { // the whole recording is one (last) chunk
        const int T = n_samples / hp.hop;
        if (plan && T > 0)
            plan->push_back({0, (T + F - 1) / F, 0});
        return T;
    }
    if (plan)
        plan->push_back({0, cl, rc});
    long mel_idx = (long)cl * F;
    const long per_chunk_samples = (long)n_chunk_mel * hp.hop + hp.win;
    long start = mel_idx * hp.hop - hp.n_fft / 2;
    while (start + per_chunk_samples <= n_samples) {
        if (plan)
            plan->push_back({(int)(mel_idx / F), cl, rc});
        mel_idx += (long)cl * F;
        start = mel_idx * hp.hop - hp.n_fft / 2;
    }
    const long rest = n_samples - start;
    const long last = rest >= hp.n_fft ? (rest - hp.n_fft) / hp.hop + 1 : 0;
    if (plan && last > 0)
        plan->push_back({(int)(mel_idx / F), (int)((last + F - 1) / F), 0});
    return (int)(mel_idx + last);
}

} // namespace

extern "C" int nemotron3_diar_set_mode(nemotron3_diar_context* c, const char* mode) {
    if (!c || !mode)
        return -1;
    static const char* k_modes[3] = {"low_latency", "very_low_latency", "ultra_low_latency"};
    if (!std::strcmp(mode, "offline")) {
        c->mode = -1;
        return 0;
    }
    for (int m = 0; m < 3; m++) {
        if (!std::strcmp(mode, k_modes[m])) {
            c->mode = m;
            return 0;
        }
    }
    return -1;
}

extern "C" int nemotron3_diar_n_valid_frames(nemotron3_diar_context* c, int n_samples) {
    if (!c || c->hp.hop <= 0 || n_samples <= 0)
        return 0;
    return c->mode < 0 ? n_samples / c->hp.hop : n3d_streaming_plan(c->hp, c->mode, n_samples, nullptr);
}

namespace {

// NemotronAsrStreamingFeatureExtractor on one span of audio: centred (first
// chunk / offline) or uncentred (later streaming chunks), keeping the first
// `keep` frames. Empty on failure.
std::vector<float> n3d_mel(nemotron3_diar_context* c, const float* pcm, int n, bool center, int keep) {
    const auto& hp = c->hp;
    const double t0 = n3d_bench() ? now_ms() : 0.0;
    core_mel::Params mp;
    mp.n_fft = hp.n_fft;
    mp.hop_length = hp.hop;
    mp.win_length = hp.win;
    mp.n_mels = hp.n_mels;
    mp.log_base = core_mel::LogBase::Ln;
    mp.log_guard = core_mel::LogGuard::AddEpsilon;
    mp.log_eps = hp.log_guard;
    mp.norm = core_mel::Normalization::None;
    mp.layout = core_mel::Layout::TimeMels;
    mp.fb_layout = core_mel::FbLayout::MelsFreqs;
    mp.center_pad = center;
    mp.drop_last_frame = false;
    mp.preemph = hp.preemph;
    int T = 0;
    std::vector<float> mel =
        core_mel::compute(pcm, n, c->window.data(), hp.win, c->fb_host.data(), hp.n_fft / 2 + 1, fft_r2c, mp, T);
    if (n3d_bench())
        n3d_acc().mel_ms += now_ms() - t0;
    if (T < keep)
        return {};
    mel.resize((size_t)keep * hp.n_mels);
    return mel;
}

} // namespace

// Streaming session: the chunk-by-chunk forward of transformers' streaming mode
// (Nemotron3DiarizationProcessor is_streaming=True + the model's speaker_cache).
struct nemotron3_diar_stream {
    nemotron3_diar_context* c = nullptr;
    int cl = 0, rc = 0;
    SpeakerCache cache;
    std::vector<float> audio; // samples from audio_base on
    long audio_base = 0, n_total = 0;
    long mel_idx = 0; // first mel frame of the next chunk
    bool started = false, ended = false;
    int max_catchup = 1; // chunks one forward may merge when audio backs up (1 = strict preset)
    // stage captures for the diff harness (scored rows only)
    bool capture = false;
    std::vector<float> cap_mel, cap_emb, cap_logits;
    int cap_T = 0, cap_Ne = 0;
};

namespace {

// One streaming forward: mel rows [T_chunk][M] (chunk + look-ahead), `scored`
// encoder frames of it scored; appends the scored logits (at most `keep_frames`
// 10 ms rows) to `out`.
bool n3d_stream_step(nemotron3_diar_stream* st, const std::vector<float>& mel, int T_chunk, int scored, int keep_frames,
                     std::vector<float>& out) {
    nemotron3_diar_context* c = st->c;
    const auto& hp = c->hp;
    const int F = hp.subsample, S = hp.n_spk, d = hp.d, M = hp.n_mels;
    std::vector<float> emb, x_rows, lg;
    int Ne = 0;
    const bool bench = n3d_bench();
    double t = bench ? now_ms() : 0.0;
    if (!embed(c, mel, T_chunk, emb, Ne))
        return false;
    if (bench) {
        n3d_acc().embed_ms += now_ms() - t;
    }
    scored = std::min(scored, Ne);
    st->cache.get_embeds(x_rows);
    const int cached = (int)(x_rows.size() / d);
    x_rows.insert(x_rows.end(), emb.begin(), emb.end());
    const int N = cached + Ne;
    const std::vector<uint8_t> mask(N, 1); // every streamed frame is audio
    if (!run_chunk(c, x_rows, N, mask, lg))
        return false;
    t = bench ? now_ms() : 0.0;
    st->cache.update(x_rows, N, lg, scored, mask, c->sil_host);
    if (bench) {
        n3d_acc().update_ms += now_ms() - t;
    }
    const int rows = std::min(scored * F, keep_frames);
    out.insert(out.end(), lg.begin() + (size_t)cached * F * S, lg.begin() + (size_t)(cached * F + rows) * S);
    if (st->capture) {
        st->cap_mel.insert(st->cap_mel.end(), mel.begin(), mel.begin() + (size_t)std::min(rows, T_chunk) * M);
        st->cap_emb.insert(st->cap_emb.end(), emb.begin(), emb.begin() + (size_t)scored * d);
        st->cap_logits.insert(st->cap_logits.end(), lg.begin() + (size_t)cached * F * S,
                              lg.begin() + (size_t)(cached * F + rows) * S);
        st->cap_T += std::min(rows, T_chunk);
        st->cap_Ne += scored;
    }
    return true;
}

// Runs every chunk the buffered audio completes; `last` also runs the final one.
// With max_catchup > 1, up to that many complete chunks that are already
// buffered run as ONE forward (a chunk of k * chunk_len frames, same look-ahead)
// - the compute of a single step, so a session that fell behind catches up
// instead of falling further behind. transformers' streaming forward takes a
// chunk of any length; the labels then differ slightly from the strict preset.
bool n3d_stream_drain(nemotron3_diar_stream* st, bool last, std::vector<float>& out) {
    const auto& hp = st->c->hp;
    const int F = hp.subsample;
    auto chunk_mel = [&](int k) { return (k * st->cl + st->rc) * F; };
    auto first_samples = [&](int k) { return (long)(chunk_mel(k) - 1) * hp.hop + hp.win / 2; };
    auto later_samples = [&](int k) { return (long)chunk_mel(k) * hp.hop + hp.win; };
    const int kmax = std::max(1, st->max_catchup);
    if (!st->started) {
        if (st->n_total >= first_samples(1)) {
            // first chunk(s): centred windows
            int k = 1;
            while (k < kmax && st->n_total >= first_samples(k + 1))
                k++;
            std::vector<float> mel = n3d_mel(st->c, st->audio.data(), (int)first_samples(k), true, chunk_mel(k));
            if (mel.empty() || !n3d_stream_step(st, mel, chunk_mel(k), k * st->cl, k * st->cl * F, out))
                return false;
            st->started = true;
            st->mel_idx = (long)k * st->cl * F;
        } else if (last) {
            // the whole session is one (last) chunk
            const int T = (int)(st->n_total / hp.hop);
            if (T > 0) {
                std::vector<float> mel = n3d_mel(st->c, st->audio.data(), (int)st->n_total, true, T);
                if (mel.empty() || !n3d_stream_step(st, mel, T, (T + F - 1) / F, T, out))
                    return false;
            }
            return true;
        } else {
            return true;
        }
    }
    for (;;) {
        const long start = st->mel_idx * hp.hop - hp.n_fft / 2; // absolute sample
        const long off = start - st->audio_base;
        if (start + later_samples(1) <= st->n_total) {
            int k = 1;
            while (k < kmax && start + later_samples(k + 1) <= st->n_total)
                k++;
            std::vector<float> mel = n3d_mel(st->c, st->audio.data() + off, (int)later_samples(k), false, chunk_mel(k));
            if (mel.empty() || !n3d_stream_step(st, mel, chunk_mel(k), k * st->cl, k * st->cl * F, out))
                return false;
            st->mel_idx += (long)k * st->cl * F;
            continue;
        }
        if (last) {
            const long rest = st->n_total - start;
            const int T = rest >= hp.n_fft ? (int)((rest - hp.n_fft) / hp.hop + 1) : 0;
            if (T > 0) {
                std::vector<float> mel = n3d_mel(st->c, st->audio.data() + off, (int)rest, false, T);
                if (mel.empty() || !n3d_stream_step(st, mel, T, (T + F - 1) / F, T, out))
                    return false;
                st->mel_idx += T;
            }
        }
        // drop audio no later chunk reads
        const long next = st->mel_idx * hp.hop - hp.n_fft / 2;
        if (next > st->audio_base) {
            st->audio.erase(st->audio.begin(), st->audio.begin() + (next - st->audio_base));
            st->audio_base = next;
        }
        return true;
    }
}

float* n3d_sigmoid_rows(const std::vector<float>& lg, int S, int* out_rows) {
    const int rows = (int)(lg.size() / S);
    if (out_rows)
        *out_rows = rows;
    if (rows == 0)
        return nullptr;
    float* p = (float*)malloc(lg.size() * sizeof(float));
    for (size_t i = 0; i < lg.size(); i++)
        p[i] = 1.0f / (1.0f + std::exp(-lg[i]));
    return p;
}

} // namespace

extern "C" nemotron3_diar_stream* nemotron3_diar_stream_begin(nemotron3_diar_context* c, const char* mode) {
    if (!c)
        return nullptr;
    static const char* k_modes[3] = {"low_latency", "very_low_latency", "ultra_low_latency"};
    int m = 0;
    if (mode && *mode) {
        for (m = 0; m < 3 && std::strcmp(mode, k_modes[m]); m++) {
        }
        if (m == 3)
            return nullptr;
    }
    auto* st = new nemotron3_diar_stream();
    st->c = c;
    st->cl = c->hp.s_modes[m][0];
    st->rc = c->hp.s_modes[m][1];
    st->cache.init(c->hp, c->hp.s_fifo_len, c->hp.s_update_period);
    if (const char* e = crispasr_env::get("CRISPASR_SORTFORMER_CATCHUP"))
        st->max_catchup = std::max(1, std::atoi(e));
    return st;
}

extern "C" void nemotron3_diar_stream_set_catchup(nemotron3_diar_stream* st, int max_chunks) {
    if (st)
        st->max_catchup = std::max(1, max_chunks);
}

extern "C" float* nemotron3_diar_stream_push(nemotron3_diar_stream* st, const float* pcm, int n_samples,
                                             int* out_rows) {
    if (out_rows)
        *out_rows = 0;
    if (!st || st->ended || n_samples < 0 || (n_samples > 0 && !pcm))
        return nullptr;
    st->audio.insert(st->audio.end(), pcm, pcm + n_samples);
    st->n_total += n_samples;
    std::vector<float> lg;
    if (!n3d_stream_drain(st, false, lg))
        return nullptr;
    return n3d_sigmoid_rows(lg, st->c->hp.n_spk, out_rows);
}

extern "C" float* nemotron3_diar_stream_end(nemotron3_diar_stream* st, int* out_rows) {
    if (out_rows)
        *out_rows = 0;
    if (!st || st->ended)
        return nullptr;
    std::vector<float> lg;
    const bool ok = n3d_stream_drain(st, true, lg);
    st->ended = true;
    return ok ? n3d_sigmoid_rows(lg, st->c->hp.n_spk, out_rows) : nullptr;
}

extern "C" void nemotron3_diar_stream_free(nemotron3_diar_stream* st) {
    delete st;
}

extern "C" float* nemotron3_diar_probs_stages(nemotron3_diar_context* c, const float* pcm, int n_samples, int* out_T,
                                              int* out_S, float** out_mel, int* out_n_mels, float** out_embeds,
                                              int* out_Ne, int* out_d, float** out_logits) {
    if (!c || !pcm || n_samples <= 0)
        return nullptr;
    const auto& hp = c->hp;
    const int M = hp.n_mels, F = hp.subsample, S = hp.n_spk, d = hp.d;
    const bool bench = n3d_bench();
    double t0 = bench ? now_ms() : 0.0;

    if (c->mode >= 0) {
        // Streaming preset: one session fed the whole recording, so this path
        // is the same code a live caller of nemotron3_diar_stream_* runs.
        static const char* k_modes[3] = {"low_latency", "very_low_latency", "ultra_low_latency"};
        nemotron3_diar_stream* st = nemotron3_diar_stream_begin(c, k_modes[c->mode]);
        if (!st)
            return nullptr;
        st->capture = out_mel || out_embeds || out_logits;
        st->max_catchup = 1; // a whole recording is not a backlog: keep the strict preset
        int r1 = 0, r2 = 0;
        float* p1 = nemotron3_diar_stream_push(st, pcm, n_samples, &r1);
        float* p2 = st->ended ? nullptr : nemotron3_diar_stream_end(st, &r2);
        const bool ok = st->ended && (r1 == 0 || p1) && (r2 == 0 || p2) && r1 + r2 > 0;
        float* probs = nullptr;
        if (ok) {
            probs = (float*)malloc((size_t)(r1 + r2) * S * sizeof(float));
            if (r1)
                std::memcpy(probs, p1, (size_t)r1 * S * sizeof(float));
            if (r2)
                std::memcpy(probs + (size_t)r1 * S, p2, (size_t)r2 * S * sizeof(float));
            auto dup = [](const std::vector<float>& v) {
                float* o = (float*)malloc(std::max<size_t>(v.size(), 1) * sizeof(float));
                if (!v.empty())
                    std::memcpy(o, v.data(), v.size() * sizeof(float));
                return o;
            };
            if (out_T)
                *out_T = r1 + r2;
            if (out_S)
                *out_S = S;
            if (out_mel)
                *out_mel = dup(st->cap_mel);
            if (out_n_mels)
                *out_n_mels = M;
            if (out_embeds)
                *out_embeds = dup(st->cap_emb);
            if (out_Ne)
                *out_Ne = st->cap_Ne;
            if (out_d)
                *out_d = d;
            if (out_logits)
                *out_logits = dup(st->cap_logits);
        }
        std::free(p1);
        std::free(p2);
        nemotron3_diar_stream_free(st);
        if (bench) {
            auto& a = n3d_acc();
            fprintf(stderr,
                    "  nemotron3_diar_bench: streaming %s, %.1f ms (%d chunks: build+alloc %.1f, compute %.1f, "
                    "mel %.1f, embed %.1f, cache update %.1f), %d frames\n",
                    k_modes[c->mode], now_ms() - t0, a.n, a.build_ms, a.compute_ms, a.mel_ms, a.embed_ms, a.update_ms,
                    r1 + r2);
            a = {};
        }
        return probs;
    }

    // ---- mel: NemotronAsrStreamingFeatureExtractor (preemph, zero-padded centred STFT,
    //      symmetric Hann(win) in n_fft, slaney mel, ln(x + 2^-24), no normalisation)
    core_mel::Params mp;
    mp.n_fft = hp.n_fft;
    mp.hop_length = hp.hop;
    mp.win_length = hp.win;
    mp.n_mels = M;
    mp.log_base = core_mel::LogBase::Ln;
    mp.log_guard = core_mel::LogGuard::AddEpsilon;
    mp.log_eps = hp.log_guard;
    mp.norm = core_mel::Normalization::None;
    mp.layout = core_mel::Layout::TimeMels;
    mp.fb_layout = core_mel::FbLayout::MelsFreqs;
    mp.center_pad = true;
    mp.drop_last_frame = false;
    mp.preemph = hp.preemph;
    int T = 0;
    std::vector<float> mel = core_mel::compute(pcm, n_samples, c->window.data(), hp.win, c->fb_host.data(),
                                               hp.n_fft / 2 + 1, fft_r2c, mp, T);
    if (mel.empty() || T <= 0)
        return nullptr;
    // valid frames: floor(L / hop); the processor zeroes the rest
    const int valid = n_samples / hp.hop;
    for (int t = valid; t < T; t++)
        std::fill(mel.begin() + (size_t)t * M, mel.begin() + (size_t)(t + 1) * M, 0.0f);
    const double t_mel = bench ? now_ms() : 0.0;

    // ---- 8x feature stacking + projection
    std::vector<float> emb;
    int Ne = 0;
    if (!embed(c, mel, T, emb, Ne))
        return nullptr;
    std::vector<uint8_t> emask(Ne);
    for (int j = 0; j < Ne; j++)
        emask[j] = (j * F) < valid;

    // ---- offline chunk loop with the speaker cache
    std::vector<n3d_chunk> plan;
    for (int start = 0; start < Ne; start += hp.chunk_len)
        plan.push_back({start, std::min(hp.chunk_len, Ne - start), hp.chunk_rc});
    SpeakerCache cache;
    cache.init(hp, hp.fifo_len, hp.update_period);
    std::vector<float> all_logits;
    all_logits.reserve((size_t)Ne * F * S);
    std::vector<float> x_rows, lg;
    std::vector<uint8_t> mask;
    int n_chunks = 0;
    for (const n3d_chunk& ch : plan) {
        const int start = ch.start;
        const int end = std::min(start + ch.n, Ne);
        const int n_chunk = end - start;
        const int take_end = std::min(end + ch.lookahead, Ne);
        cache.get_embeds(x_rows);
        const int cached = (int)(x_rows.size() / d);
        x_rows.insert(x_rows.end(), emb.begin() + (size_t)start * d, emb.begin() + (size_t)take_end * d);
        const int N = cached + (take_end - start);
        mask.assign(cached, 1);
        mask.insert(mask.end(), emask.begin() + start, emask.begin() + take_end);
        if (!run_chunk(c, x_rows, N, mask, lg))
            return nullptr;
        cache.update(x_rows, N, lg, n_chunk, mask, c->sil_host);
        all_logits.insert(all_logits.end(), lg.begin() + (size_t)cached * F * S,
                          lg.begin() + (size_t)(cached + n_chunk) * F * S);
        n_chunks++;
    }
    all_logits.resize((size_t)T * S); // the stacking pad frames past T are dropped
    if (bench) {
        auto& a = n3d_acc();
        fprintf(stderr,
                "  nemotron3_diar_bench: mel %.1f ms, encoder %d chunk(s) %.1f ms (build+alloc %.1f, compute %.1f), "
                "%d frames\n",
                t_mel - t0, n_chunks, now_ms() - t_mel, a.build_ms, a.compute_ms, T);
        a = {};
    }

    float* probs = (float*)malloc(all_logits.size() * sizeof(float));
    for (size_t i = 0; i < all_logits.size(); i++)
        probs[i] = 1.0f / (1.0f + std::exp(-all_logits[i]));
    if (out_T)
        *out_T = T;
    if (out_S)
        *out_S = S;
    if (out_mel) {
        *out_mel = (float*)malloc(mel.size() * sizeof(float));
        std::memcpy(*out_mel, mel.data(), mel.size() * sizeof(float));
    }
    if (out_n_mels)
        *out_n_mels = M;
    if (out_embeds) {
        *out_embeds = (float*)malloc(emb.size() * sizeof(float));
        std::memcpy(*out_embeds, emb.data(), emb.size() * sizeof(float));
    }
    if (out_Ne)
        *out_Ne = Ne;
    if (out_d)
        *out_d = d;
    if (out_logits) {
        *out_logits = (float*)malloc(all_logits.size() * sizeof(float));
        std::memcpy(*out_logits, all_logits.data(), all_logits.size() * sizeof(float));
    }
    return probs;
}

extern "C" float* nemotron3_diar_probs(nemotron3_diar_context* c, const float* pcm, int n_samples, int* out_T,
                                       int* out_S) {
    return nemotron3_diar_probs_stages(c, pcm, n_samples, out_T, out_S, nullptr, nullptr, nullptr, nullptr, nullptr,
                                       nullptr);
}
