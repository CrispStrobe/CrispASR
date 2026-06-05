// Minimal test: verify ggml snake activation matches Python.
// Input: ups_0(t=0, c=0..4) from ref, alpha from GGUF.
// Expected (Python): [-0.2993, -0.1853, -0.1020, 0.1934, -0.1158]

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    // Known values from the vocoder debug
    float x_vals[] = {-0.6878f, -0.2112f, -0.1095f, 0.1693f, -0.1307f}; // ups_0(t=0, c=0..4)
    float alpha_vals[] = {0.9485f, 0.5825f, 0.6274f, 0.8460f, 0.8754f}; // rb0.a1.0 first 5
    float expected[] = {-0.2993f, -0.1853f, -0.1020f, 0.1934f, -0.1158f}; // Python snake1_d0

    printf("=== Manual Snake computation ===\n");
    for (int c = 0; c < 5; c++) {
        float a = alpha_vals[c];
        float x = x_vals[c];
        float ax = a * x;
        float s = sinf(ax);
        float s2 = s * s;
        float s2_over_a = s2 / (a + 1e-9f);
        float result = x + s2_over_a;
        printf("  c=%d: x=%.4f alpha=%.4f ax=%.4f sin=%.4f sin2=%.6f /a=%.6f result=%.4f (expected=%.4f) %s\n",
               c, x, a, ax, s, s2, s2_over_a, result, expected[c],
               fabsf(result - expected[c]) < 0.001f ? "OK" : "MISMATCH");
    }

    // Now test with ggml graph (same as the vocoder does)
    printf("\n=== ggml Snake computation (C=5, T=1) ===\n");

    // Create a minimal graph
    size_t buf_size = 1024 * 1024;
    std::vector<uint8_t> meta(buf_size);
    ggml_init_params ip = {meta.size(), meta.data(), true};
    ggml_context* ctx = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph(ctx);

    // x: ne[0]=T=1, ne[1]=C=5  (like the vocoder)
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 5);
    ggml_set_name(x, "x");
    ggml_set_input(x);

    // alpha: ne[0]=5  → reshape to ne[0]=1, ne[1]=5
    ggml_tensor* alpha = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
    ggml_set_name(alpha, "alpha");
    ggml_set_input(alpha);

    ggml_tensor* a = ggml_reshape_2d(ctx, alpha, 1, 5);
    ggml_tensor* ax = ggml_mul(ctx, x, a);
    ggml_set_name(ax, "ax");
    ggml_set_output(ax);

    ggml_tensor* sin_ax = ggml_sin(ctx, ax);
    ggml_set_name(sin_ax, "sin_ax");
    ggml_set_output(sin_ax);

    ggml_tensor* sin2 = ggml_mul(ctx, sin_ax, sin_ax);
    ggml_set_name(sin2, "sin2");
    ggml_set_output(sin2);

    ggml_tensor* sin2_over_a = ggml_div(ctx, sin2, a);
    ggml_set_name(sin2_over_a, "sin2_over_a");
    ggml_set_output(sin2_over_a);

    ggml_tensor* result = ggml_add(ctx, x, sin2_over_a);
    ggml_set_name(result, "result");
    ggml_set_output(result);

    ggml_build_forward_expand(gf, result);

    // Create backend and run
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, nullptr, 1, 256, false, false);

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);

    // Set inputs: x is (T=1, C=5) → data[c*1+0] = data[c]
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"), x_vals, 0, 5 * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "alpha"), alpha_vals, 0, 5 * sizeof(float));

    ggml_backend_sched_graph_compute(sched, gf);

    // Read intermediates
    auto read = [&](const char* name) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, name);
        if (!t) { printf("  %s: NOT FOUND\n", name); return; }
        float buf[5];
        ggml_backend_tensor_get(t, buf, 0, 5 * sizeof(float));
        // ne[0]=1, ne[1]=5 → data[c*1+0] = data[c]
        printf("  %-16s ne=(%lld,%lld): ", name, (long long)t->ne[0], (long long)t->ne[1]);
        for (int c = 0; c < 5; c++)
            printf("%.4f ", buf[c]);
        printf("\n");
    };

    read("ax");
    read("sin_ax");
    read("sin2");
    read("sin2_over_a");
    read("result");

    printf("\n  Expected (Python): ");
    for (int c = 0; c < 5; c++)
        printf("%.4f ", expected[c]);
    printf("\n");

    // Now test with T=496 (actual vocoder size) to check if size matters
    printf("\n=== ggml Snake computation (C=256, T=496) ===\n");
    ggml_free(ctx);
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);

    std::vector<uint8_t> meta2(64 * 1024 * 1024);
    ggml_init_params ip2 = {meta2.size(), meta2.data(), true};
    ctx = ggml_init(ip2);
    gf = ggml_new_graph(ctx);

    x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 496, 256);
    ggml_set_name(x, "x");
    ggml_set_input(x);

    alpha = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    ggml_set_name(alpha, "alpha");
    ggml_set_input(alpha);

    a = ggml_reshape_2d(ctx, alpha, 1, 256);
    ax = ggml_mul(ctx, x, a);
    sin_ax = ggml_sin(ctx, ax);
    sin2 = ggml_mul(ctx, sin_ax, sin_ax);
    sin2_over_a = ggml_div(ctx, sin2, a);
    result = ggml_add(ctx, x, sin2_over_a);
    ggml_set_name(result, "result");
    ggml_set_output(result);
    ggml_build_forward_expand(gf, result);

    backend = ggml_backend_cpu_init();
    sched = ggml_backend_sched_new(&backend, nullptr, 1, 256, false, false);
    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);

    // Fill with zeros except (t=0, c=0..4) from test values
    std::vector<float> x_big(496 * 256, 0.0f);
    for (int c = 0; c < 5; c++)
        x_big[c * 496 + 0] = x_vals[c]; // data[c*T+t] layout
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"), x_big.data(), 0, x_big.size() * sizeof(float));

    std::vector<float> alpha_big(256, 1.0f);
    for (int c = 0; c < 5; c++)
        alpha_big[c] = alpha_vals[c];
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "alpha"), alpha_big.data(), 0, 256 * sizeof(float));

    ggml_backend_sched_graph_compute(sched, gf);

    float r_big[5];
    ggml_tensor* res = ggml_graph_get_tensor(gf, "result");
    // Read (t=0, c=0..4): data at c*496+0
    for (int c = 0; c < 5; c++) {
        ggml_backend_tensor_get(res, &r_big[c], c * 496 * sizeof(float), sizeof(float));
    }
    printf("  result (t=0,c=0..4): ");
    for (int c = 0; c < 5; c++)
        printf("%.4f ", r_big[c]);
    printf("\n  expected:            ");
    for (int c = 0; c < 5; c++)
        printf("%.4f ", expected[c]);
    printf("\n");

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
