// Does ggml_conv_1d produce correct results for batch N > 1?
// Compares against a hand-rolled CPU reference for N = 1, 2, 3.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    // Small but non-degenerate: OC != OL, both > 1, so a layout transpose shows.
    const int T = 12, IC = 3, OC = 5, K = 3, S = 1, P = 1;
    const int OL = (T + 2 * P - K) / S + 1; // 12

    for (int N = 1; N <= 3; N++) {
        std::vector<float> wdat((size_t)K * IC * OC), xdat((size_t)T * IC * N);
        for (size_t i = 0; i < wdat.size(); i++)
            wdat[i] = std::sin(0.7f * (float)i + 0.3f);
        for (size_t i = 0; i < xdat.size(); i++)
            xdat[i] = std::cos(0.4f * (float)i + 0.1f);

        ggml_backend_t be = ggml_backend_cpu_init();
        ggml_init_params ip = {(size_t)64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
        ggml_context* c = ggml_init(ip);
        ggml_cgraph* gf = ggml_new_graph(c);

        ggml_tensor* w = ggml_new_tensor_3d(c, GGML_TYPE_F32, K, IC, OC);
        ggml_tensor* x = ggml_new_tensor_3d(c, GGML_TYPE_F32, T, IC, N);
        ggml_set_input(w);
        ggml_set_input(x);
        ggml_tensor* y = ggml_conv_1d(c, w, x, S, P, 1);
        ggml_set_output(y);
        ggml_build_forward_expand(gf, y);

        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_tensor_set(w, wdat.data(), 0, wdat.size() * sizeof(float));
        ggml_backend_tensor_set(x, xdat.data(), 0, xdat.size() * sizeof(float));
        ggml_backend_graph_compute(be, gf);

        std::vector<float> got((size_t)ggml_nelements(y));
        ggml_backend_tensor_get(y, got.data(), 0, got.size() * sizeof(float));

        // Reference: out[n][oc][ol] = sum_ic sum_k x[n][ic][ol*S - P + k] * w[oc][ic][k]
        std::vector<float> ref((size_t)OL * OC * N, 0.0f);
        for (int n = 0; n < N; n++)
            for (int oc = 0; oc < OC; oc++)
                for (int ol = 0; ol < OL; ol++) {
                    float acc = 0.0f;
                    for (int ic = 0; ic < IC; ic++)
                        for (int k = 0; k < K; k++) {
                            const int t = ol * S - P + k;
                            if (t < 0 || t >= T)
                                continue;
                            acc += xdat[(size_t)n * T * IC + (size_t)ic * T + t] *
                                   wdat[(size_t)oc * IC * K + (size_t)ic * K + k];
                        }
                    // ggml claims y ne = (OL, OC, N)
                    ref[(size_t)n * OL * OC + (size_t)oc * OL + ol] = acc;
                }

        double num = 0, da = 0, db = 0, maxabs = 0;
        for (size_t i = 0; i < ref.size(); i++) {
            num += got[i] * ref[i];
            da += got[i] * got[i];
            db += ref[i] * ref[i];
            maxabs = std::max(maxabs, (double)std::fabs(got[i] - ref[i]));
        }
        const double cos = num / (std::sqrt(da) * std::sqrt(db));
        printf("N=%d  y.ne=(%lld,%lld,%lld)  cos=%.8f  max_abs=%.3e  %s\n", N, (long long)y->ne[0], (long long)y->ne[1],
               (long long)y->ne[2], cos, maxabs, (cos > 0.9999 && maxabs < 1e-4) ? "OK" : "MISMATCH");

        ggml_gallocr_free(ga);
        ggml_free(c);
        ggml_backend_free(be);
    }
    return 0;
}
