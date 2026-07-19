// What does ggml_conv_1d_dw actually DO for batch N > 1?
//   argv[1] = N. Run each N as a SEPARATE process: if it hits a GGML_ASSERT
//   the process aborts, and we want to distinguish "aborts loudly" (safe
//   failure) from "returns wrong data" (silent corruption).
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    const int N = (argc > 1) ? atoi(argv[1]) : 1;
    const int T = 16, C = 4, K = 3, S = 1, P = 1;
    const int OL = (T + 2 * P - K) / S + 1;

    // depthwise: kernel a = [K, 1, C], input b = [T, C, N]
    std::vector<float> wdat((size_t)K * 1 * C), xdat((size_t)T * C * N);
    for (size_t i = 0; i < wdat.size(); i++)
        wdat[i] = std::sin(0.7f * (float)i + 0.3f);
    for (size_t i = 0; i < xdat.size(); i++)
        xdat[i] = std::cos(0.4f * (float)i + 0.1f);

    fprintf(stderr, "[dw_probe] N=%d: building graph...\n", N);
    fflush(stderr);

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_init_params ip = {(size_t)64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
    ggml_context* c = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph(c);

    ggml_tensor* w = ggml_new_tensor_3d(c, GGML_TYPE_F32, K, 1, C);
    ggml_tensor* x = ggml_new_tensor_3d(c, GGML_TYPE_F32, T, C, N);
    ggml_set_input(w);
    ggml_set_input(x);
    ggml_tensor* y = ggml_conv_1d_dw(c, w, x, S, P, 1);
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    fprintf(stderr, "[dw_probe] N=%d: SURVIVED build, y.ne=(%lld,%lld,%lld,%lld)\n", N, (long long)y->ne[0],
            (long long)y->ne[1], (long long)y->ne[2], (long long)y->ne[3]);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(w, wdat.data(), 0, wdat.size() * sizeof(float));
    ggml_backend_tensor_set(x, xdat.data(), 0, xdat.size() * sizeof(float));
    ggml_backend_graph_compute(be, gf);

    std::vector<float> got((size_t)ggml_nelements(y));
    ggml_backend_tensor_get(y, got.data(), 0, got.size() * sizeof(float));

    // reference: out[n][ch][ol] = sum_k x[n][ch][ol - P + k] * w[ch][k]
    const size_t need = (size_t)OL * C * N;
    std::vector<float> ref(need, 0.0f);
    for (int n = 0; n < N; n++)
        for (int ch = 0; ch < C; ch++)
            for (int ol = 0; ol < OL; ol++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    const int t = ol * S - P + k;
                    if (t < 0 || t >= T)
                        continue;
                    acc += xdat[(size_t)n * T * C + (size_t)ch * T + t] * wdat[(size_t)ch * K + k];
                }
                ref[(size_t)n * OL * C + (size_t)ch * OL + ol] = acc;
            }

    printf("N=%d  y.ne=(%lld,%lld,%lld,%lld)  nelem=%lld  expected=%zu  ", N, (long long)y->ne[0], (long long)y->ne[1],
           (long long)y->ne[2], (long long)y->ne[3], (long long)ggml_nelements(y), need);
    if ((size_t)ggml_nelements(y) != need) {
        printf("SHAPE LOSS (batch dropped)\n");
        return 0;
    }
    double num = 0, da = 0, db = 0, mx = 0;
    for (size_t i = 0; i < need; i++) {
        num += got[i] * ref[i];
        da += got[i] * got[i];
        db += ref[i] * ref[i];
        mx = std::max(mx, (double)std::fabs(got[i] - ref[i]));
    }
    const double cos = num / (std::sqrt(da) * std::sqrt(db));
    printf("cos=%.8f max_abs=%.3e %s\n", cos, mx, (cos > 0.9999 && mx < 1e-4) ? "OK" : "MISMATCH");
    return 0;
}
