// MossFormer2_SS_16K: model-free guards for the shared MossFormer2 blocks
// (depthwise-conv kernels on both execution paths against a naive oracle) and
// the runtime's failure handling, plus an env-gated live check:
//   CRISPASR_MOSSFORMER2_SS_MODEL=<converted gguf>      separate() contract
//   CRISPASR_MOSSFORMER2_SS_REF=<dump_mossformer2_ss_reference.py --gguf>
//                                                       full stage + e2e parity
#include <catch2/catch_test_macros.hpp>

#include "mossformer2_blocks.h"
#include "mossformer2_ss.h"

#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr double kPi = 3.14159265358979323846;

struct MetadataFile {
    std::string path;
    explicit MetadataFile(const char* arch) {
        path = (std::filesystem::temp_directory_path() /
                ("mossformer2-ss-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                 ".gguf"))
                   .string();
        gguf_context* meta = gguf_init_empty();
        gguf_set_val_str(meta, "general.architecture", arch);
        const bool ok = gguf_write_to_file(meta, path.c_str(), true);
        gguf_free(meta);
        REQUIRE(ok);
    }
    ~MetadataFile() { std::remove(path.c_str()); }
};

// PyTorch Conv1d(groups=C) semantics: y[c,t] = sum_j sum_k w[c,j,k] x[c*gin+j, t + k*dil - pad_l]
std::vector<float> naive_dwconv(const std::vector<float>& x, const std::vector<float>& w, int C, int gin, int T, int K,
                                int dil, int pad_l, int pad_r) {
    const int Tout = T + pad_l + pad_r - (K - 1) * dil;
    std::vector<float> y((size_t)C * Tout, 0.0f);
    for (int t = 0; t < Tout; ++t)
        for (int c = 0; c < C; ++c) {
            double acc = 0;
            for (int j = 0; j < gin; ++j)
                for (int k = 0; k < K; ++k) {
                    const int src = t + k * dil - pad_l;
                    if (src >= 0 && src < T)
                        acc += (double)w[((size_t)c * gin + j) * K + k] * x[(size_t)src * C * gin + c * gin + j];
                }
            y[(size_t)t * C + c] = (float)acc;
        }
    return y;
}

std::vector<float> run_dwconv(bool portable, const std::vector<float>& x, const std::vector<float>& w, int C, int gin,
                              int T, int K, int dil, int pad_l, int pad_r) {
    ggml_init_params ip{256 * 1024 * 1024, nullptr, false};
    ggml_context* g = ggml_init(ip);
    REQUIRE(g);
    ggml_tensor* xt = ggml_new_tensor_2d(g, GGML_TYPE_F32, (int64_t)C * gin, T);
    std::memcpy(xt->data, x.data(), x.size() * sizeof(float));
    // GGUF layout of the PyTorch [C, gin, K] kernel: ne = [K, gin, C] (gin == 1: [K, C]).
    ggml_tensor* wt =
        gin == 1 ? ggml_new_tensor_2d(g, GGML_TYPE_F32, K, C) : ggml_new_tensor_3d(g, GGML_TYPE_F32, K, gin, C);
    std::memcpy(wt->data, w.data(), w.size() * sizeof(float));
    std::map<std::string, ggml_tensor*> weights{{"k", wt}}, derived;
    if (!portable) {
        const auto tap = mf2::tap_kernel(w.data(), C, gin, K);
        ggml_tensor* t = ggml_new_tensor_2d(g, GGML_TYPE_F32, (int64_t)C * gin, K);
        std::memcpy(t->data, tap.data(), tap.size() * sizeof(float));
        derived[mf2::tap_name("k")] = t;
    } else if (gin > 1 || dil > 1) {
        const auto dk = mf2::dilate_kernel(w.data(), C, gin, K, dil);
        for (int j = 0; j < gin; ++j) {
            ggml_tensor* t = ggml_new_tensor_2d(g, GGML_TYPE_F32, (int64_t)(K - 1) * dil + 1, C);
            std::memcpy(t->data, dk[j].data(), dk[j].size() * sizeof(float));
            derived[mf2::dilated_name("k", j)] = t;
        }
    }
    mf2::Ctx c;
    c.g = g;
    c.weight = [&](const std::string& n) { return weights.at(n); };
    c.derived = &derived;
    c.portable_dwconv = portable;
    ggml_tensor* y = mf2::dwconv(c, xt, "k", gin, dil, pad_l, pad_r);
    ggml_cgraph* gf = ggml_new_graph(g);
    ggml_build_forward_expand(gf, y);
    REQUIRE(ggml_graph_compute_with_ctx(g, gf, 2) == GGML_STATUS_SUCCESS);
    REQUIRE(y->ne[0] == C);
    std::vector<float> out((size_t)ggml_nelements(y));
    std::memcpy(out.data(), y->data, out.size() * sizeof(float));
    ggml_free(g);
    return out;
}
} // namespace

TEST_CASE("MossFormer2 portable kernel split uses GPU-supported widths", "[unit][mossformer2-ss]") {
    for (int K : {3, 17, 39, 77}) {
        int sum = 0;
        for (int k : mf2::portable_kernel_split(K)) {
            CHECK((k == 3 || k == 4 || k == 5 || k == 9 || k == 15));
            sum += k;
        }
        CHECK(sum == K);
    }
}

TEST_CASE("MossFormer2 depthwise conv matches a naive grouped Conv1d on both paths", "[unit][mossformer2-ss]") {
    struct Case {
        int C, gin, T, K, dil, pad;
    };
    // ConvModule (k17, pad 8), DDN conv1 (k39, pad 19), DDN conv2 (k39, groups of 2, dil 2, pad 38),
    // and a sequence shorter than the receptive field.
    for (const Case cs : {Case{48, 1, 70, 17, 1, 8}, Case{16, 1, 57, 39, 1, 19}, Case{16, 2, 90, 39, 2, 38},
                          Case{8, 2, 20, 39, 2, 38}}) {
        INFO("C=" << cs.C << " gin=" << cs.gin << " T=" << cs.T << " K=" << cs.K << " dil=" << cs.dil);
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd;
        std::vector<float> x((size_t)cs.C * cs.gin * cs.T), w((size_t)cs.C * cs.gin * cs.K);
        for (auto& v : x)
            v = nd(rng);
        for (auto& v : w)
            v = nd(rng);
        const auto want = naive_dwconv(x, w, cs.C, cs.gin, cs.T, cs.K, cs.dil, cs.pad, cs.pad);
        for (bool portable : {false, true}) {
            INFO("portable=" << portable);
            const auto got = run_dwconv(portable, x, w, cs.C, cs.gin, cs.T, cs.K, cs.dil, cs.pad, cs.pad);
            REQUIRE(got.size() == want.size());
            double max_err = 0;
            for (size_t i = 0; i < got.size(); ++i)
                max_err = std::max(max_err, (double)std::abs(got[i] - want[i]));
            CHECK(max_err < 1e-4);
        }
    }
}

TEST_CASE("MossFormer2 scaled sinusoidal embedding layout", "[unit][mossformer2-ss]") {
    const std::vector<float> inv_freq{1.0f, 0.1f};
    const auto e = mf2::sinu_embedding(3, inv_freq, 2.0f);
    REQUIRE(e.size() == 12);
    // row t = [sin(t f0), sin(t f1), cos(t f0), cos(t f1)] * scale
    CHECK(std::abs(e[4 + 0] - 2 * std::sin(1.0f)) < 1e-6);
    CHECK(std::abs(e[4 + 1] - 2 * std::sin(0.1f)) < 1e-6);
    CHECK(std::abs(e[4 + 2] - 2 * std::cos(1.0f)) < 1e-6);
    CHECK(std::abs(e[8 + 3] - 2 * std::cos(0.2f)) < 1e-6);
}

TEST_CASE("MossFormer2 SS rejects unavailable and incomplete models", "[unit][mossformer2-ss]") {
    auto params = mossformer2_ss_default_params();
    params.use_gpu = false;
    params.n_threads = 1;
    CHECK(mossformer2_ss_init_from_file(nullptr, params) == nullptr);
    CHECK(mossformer2_ss_init_from_file("/no-such-mossformer2-ss.gguf", params) == nullptr);
    {
        MetadataFile file("mini-bs-roformer");
        CHECK(mossformer2_ss_init_from_file(file.path.c_str(), params) == nullptr);
    }
    {
        MetadataFile file("mossformer2-ss");
        CHECK(mossformer2_ss_init_from_file(file.path.c_str(), params) == nullptr);
    }
    float pcm[4] = {0, 0, 0, 0};
    CHECK(mossformer2_ss_separate(nullptr, pcm, 4, 1) == nullptr);
    CHECK(mossformer2_ss_sample_rate(nullptr) == 0);
    CHECK(mossformer2_ss_source_name(nullptr, 0) == nullptr);
    mossformer2_ss_result_free(nullptr);
    mossformer2_ss_free(nullptr);
}

TEST_CASE("MossFormer2 SS live separation contract", "[live][mossformer2-ss]") {
    const char* model = std::getenv("CRISPASR_MOSSFORMER2_SS_MODEL");
    if (!model || !*model)
        SKIP("set CRISPASR_MOSSFORMER2_SS_MODEL to a converted MossFormer2_SS_16K GGUF");
    auto params = mossformer2_ss_default_params();
    params.use_gpu = false;
    params.n_threads = 4;
    mossformer2_ss_context* ctx = mossformer2_ss_init_from_file(model, params);
    REQUIRE(ctx);
    CHECK(mossformer2_ss_sample_rate(ctx) == 16000);
    CHECK(mossformer2_ss_n_sources(ctx) == 2);
    CHECK(std::string(mossformer2_ss_source_name(ctx, 1)) == "spk2");
    // Two "speakers": a voiced 140-Hz harmonic tone and a gated 900-Hz chirp.
    // 40000 samples (2.5 s) takes upstream's segmented path (two 2-s windows).
    for (int n : {20000, 40000}) {
        INFO("n_samples=" << n);
        std::vector<float> pcm(n);
        for (int i = 0; i < n; ++i) {
            const double t = i / 16000.0;
            pcm[i] = (float)(0.2 * std::sin(2 * kPi * 140 * t) + 0.1 * std::sin(2 * kPi * 280 * t) +
                             (std::fmod(t, 0.5) < 0.25 ? 0.15 * std::sin(2 * kPi * (900 + 300 * t) * t) : 0.0));
        }
        mossformer2_ss_result* a = mossformer2_ss_separate(ctx, pcm.data(), n, 1);
        REQUIRE(a);
        CHECK(a->n_sources == 2);
        CHECK(a->n_channels == 1);
        CHECK(a->n_samples == n);
        CHECK(a->sample_rate == 16000);
        // Same input twice through the cached graph must be bit-identical.
        mossformer2_ss_result* b = mossformer2_ss_separate(ctx, pcm.data(), n, 1);
        REQUIRE(b);
        for (int s = 0; s < 2; ++s) {
            double energy = 0;
            for (int i = 0; i < n; ++i) {
                REQUIRE(std::isfinite(a->sources[s][i]));
                CHECK(a->sources[s][i] == b->sources[s][i]);
                energy += (double)a->sources[s][i] * a->sources[s][i];
            }
            CHECK(energy > 0);
        }
        mossformer2_ss_result_free(a);
        mossformer2_ss_result_free(b);
    }
    mossformer2_ss_free(ctx);
    if (const char* ref = std::getenv("CRISPASR_MOSSFORMER2_SS_REF"); ref && *ref)
        CHECK(mossformer2_ss_diff(model, ref, "", 1) == 0);
}
