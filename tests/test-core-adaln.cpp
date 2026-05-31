// test-core-adaln.cpp — unit test for core/adaln.h (AdaLN-Zero modulation).
//
// Builds the header's modulate6 + apply_norm_modulation ggml graph and
// compares to an independent plain-C++ reference (silu → projection → 6-way
// split; affine-free LayerNorm → norm*(1+scale)+shift). Also checks the
// apply_silu=false path (cosyvoice3 production: caller pre-applies silu).
//
// CPU backend, no models. cond_dim=5, dim=4, T_x=3.

#include <catch2/catch_test_macros.hpp>

#include "core/adaln.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <vector>

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint32_t nx() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 11); }
    float sym() { return (float)((nx() % 20001) / 10000.0 - 1.0); }
};

float silu(float v) { return v / (1.0f + std::exp(-v)); }

// Run the header graph for one config, return the modulated output (dim x T_x).
std::vector<float> run_header(const std::vector<float>& temb, const std::vector<float>& w,
                             const std::vector<float>& b, const std::vector<float>& x, int cond_dim, int dim,
                             int T_x, bool apply_silu) {
    std::vector<uint8_t> meta(4 * 1024 * 1024);
    ggml_init_params ip{meta.size(), meta.data(), true};
    ggml_context* ctx = ggml_init(ip);

    ggml_tensor* t_temb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cond_dim, 1);
    ggml_tensor* t_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cond_dim, 6 * dim);
    ggml_tensor* t_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6 * dim);
    ggml_tensor* t_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, T_x);
    for (auto* t : {t_temb, t_w, t_b, t_x})
        ggml_set_input(t);
    ggml_set_name(t_temb, "temb"); ggml_set_name(t_w, "w");
    ggml_set_name(t_b, "b"); ggml_set_name(t_x, "x");

    auto m = core_adaln::modulate6(ctx, t_temb, t_w, t_b, apply_silu);
    ggml_tensor* out = core_adaln::apply_norm_modulation(ctx, t_x, m.scale_msa, m.shift_msa, 1e-6f);
    ggml_set_output(out);
    ggml_set_name(out, "out");

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, nullptr, 1, 512, false, false);
    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "temb"), temb.data(), 0, temb.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "w"), w.data(), 0, w.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "b"), b.data(), 0, b.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"), x.data(), 0, x.size() * sizeof(float));
    ggml_backend_sched_graph_compute(sched, gf);

    std::vector<float> got((size_t)dim * T_x);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "out"), got.data(), 0, got.size() * sizeof(float));

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return got;
}

// Plain reference: emb = (silu?)t_emb; proj = w·emb + b; shift=proj[0..dim),
// scale=proj[dim..2dim); norm_x = affine-free LayerNorm(x) over dim;
// out = norm_x*(1+scale)+shift.
std::vector<float> reference(const std::vector<float>& temb, const std::vector<float>& w,
                            const std::vector<float>& b, const std::vector<float>& x, int cond_dim, int dim,
                            int T_x, bool apply_silu) {
    std::vector<float> emb(cond_dim);
    for (int i = 0; i < cond_dim; i++)
        emb[i] = apply_silu ? silu(temb[i]) : temb[i];
    // proj[o] = sum_i w[i,o]*emb[i] + b[o]   (w ne=[cond_dim,6dim], [i,o] at o*cond_dim+i)
    std::vector<float> proj(6 * dim);
    for (int o = 0; o < 6 * dim; o++) {
        float acc = 0;
        for (int i = 0; i < cond_dim; i++)
            acc += w[(size_t)o * cond_dim + i] * emb[i];
        proj[o] = acc + b[o];
    }
    const float* shift = proj.data() + 0 * dim; // shift_msa
    const float* scale = proj.data() + 1 * dim; // scale_msa

    std::vector<float> out((size_t)dim * T_x);
    for (int t = 0; t < T_x; t++) {
        // affine-free LayerNorm over the dim axis (ggml_norm: 1/N variance).
        float mean = 0;
        for (int d = 0; d < dim; d++)
            mean += x[(size_t)t * dim + d];
        mean /= dim;
        float var = 0;
        for (int d = 0; d < dim; d++) {
            float c = x[(size_t)t * dim + d] - mean;
            var += c * c;
        }
        var /= dim;
        float inv = 1.0f / std::sqrt(var + 1e-6f);
        for (int d = 0; d < dim; d++) {
            float nx = (x[(size_t)t * dim + d] - mean) * inv;
            out[(size_t)t * dim + d] = nx * (1.0f + scale[d]) + shift[d];
        }
    }
    return out;
}

} // namespace

TEST_CASE("core_adaln: modulate6 + apply_norm_modulation == plain reference", "[unit][core-adaln]") {
    // The header derives dim = t_emb->ne[0], i.e. it requires the conditioning
    // embedding to already be at the model dim (true in f5/cosyvoice3, where
    // the time-embedding is projected to dim before modulate6). So cond_dim == dim.
    const int dim = 6, cond_dim = dim, T_x = 3;
    Rng rng(0xADA1FULL);
    std::vector<float> temb(cond_dim), w((size_t)cond_dim * 6 * dim), b(6 * dim), x((size_t)dim * T_x);
    for (auto* v : {&temb, &w, &b, &x})
        for (auto& e : *v)
            e = rng.sym();

    SECTION("apply_silu=true (f5 / cosyvoice3 debug path)") {
        auto got = run_header(temb, w, b, x, cond_dim, dim, T_x, true);
        auto ref = reference(temb, w, b, x, cond_dim, dim, T_x, true);
        float max_abs = 0;
        for (size_t i = 0; i < got.size(); i++)
            max_abs = std::max(max_abs, std::fabs(got[i] - ref[i]));
        INFO("max|Δ| = " << max_abs);
        REQUIRE(max_abs < 1e-4f);
    }
    SECTION("apply_silu=false (cosyvoice3 production: caller pre-silus)") {
        auto got = run_header(temb, w, b, x, cond_dim, dim, T_x, false);
        auto ref = reference(temb, w, b, x, cond_dim, dim, T_x, false);
        float max_abs = 0;
        for (size_t i = 0; i < got.size(); i++)
            max_abs = std::max(max_abs, std::fabs(got[i] - ref[i]));
        INFO("max|Δ| = " << max_abs);
        REQUIRE(max_abs < 1e-4f);
    }
}
