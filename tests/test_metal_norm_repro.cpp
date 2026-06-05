// Standalone ggml_norm Metal vs CPU repro for the kokoro short-input
// regression. Builds a single ggml_norm op over shape (T=65, C=512)
// and compares Metal output to CPU. Designed to confirm/refute that
// the bug is in ggml-metal's kernel_norm_fuse_impl independently of
// kokoro's AdainResBlk1d.
//
// Build: cmake --build build-ninja-compile --target test_metal_norm_repro
// Run:   ./build-ninja-compile/bin/test_metal_norm_repro [T] [C]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" ggml_backend_t ggml_backend_metal_init(void);

static void fill_random(float* data, int n, unsigned seed) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (int i = 0; i < n; i++)
        data[i] = u(g);
}

static void run_norm(ggml_backend_t backend, const char* tag, const float* src, int T, int C, float* dst) {
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context* ctx = ggml_init(ip);
    // Layout: ne=(T inner, C outer) so kernel_norm normalises along T per row of C.
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, C);
    ggml_set_input(x);
    ggml_tensor* y = ggml_norm(ctx, x, 1e-5f);
    ggml_set_output(y);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "%s: alloc_ctx_tensors failed\n", tag);
        return;
    }
    ggml_backend_tensor_set(x, src, 0, (size_t)T * C * sizeof(float));

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "%s: alloc_graph failed\n", tag);
        ggml_gallocr_free(galloc);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return;
    }
    enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: compute failed (%d)\n", tag, (int)st);
    }
    ggml_backend_tensor_get(y, dst, 0, (size_t)T * C * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char** argv) {
    int T = argc > 1 ? std::atoi(argv[1]) : 65;
    int C = argc > 2 ? std::atoi(argv[2]) : 512;
    int N = T * C;
    std::vector<float> src(N), dst_cpu(N), dst_mtl(N);
    fill_random(src.data(), N, 42);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    run_norm(cpu, "CPU", src.data(), T, C, dst_cpu.data());
    ggml_backend_free(cpu);

    ggml_backend_t mtl = ggml_backend_metal_init();
    if (!mtl) {
        fprintf(stderr, "metal_init failed\n");
        return 2;
    }
    run_norm(mtl, "MTL", src.data(), T, C, dst_mtl.data());
    ggml_backend_free(mtl);

    // Per-row stats (rows = C, each of length T)
    int n_rows_bad = 0;
    double max_abs = 0, sse = 0;
    for (int c = 0; c < C; c++) {
        double row_max = 0;
        for (int t = 0; t < T; t++) {
            double d = std::fabs((double)dst_cpu[c * T + t] - (double)dst_mtl[c * T + t]);
            if (d > row_max)
                row_max = d;
            if (d > max_abs)
                max_abs = d;
            sse += d * d;
        }
        if (row_max > 1e-3)
            n_rows_bad++;
    }
    double rmse = std::sqrt(sse / N);
    printf("shape T=%d C=%d  N=%d\n", T, C, N);
    // Check per-row mean/std to see whether InstanceNorm output is correctly normalized
    int bad_mean = 0, bad_std = 0;
    double worst_mean = 0, worst_std_dev = 0;
    int worst_mean_row = -1, worst_std_row = -1;
    for (int c = 0; c < C; c++) {
        double sum = 0, sumsq = 0;
        for (int t = 0; t < T; t++) {
            float v = dst_mtl[c * T + t];
            sum += v;
            sumsq += (double)v * v;
        }
        double mean = sum / T;
        double var = sumsq / T - mean * mean;
        double sd = std::sqrt(std::max(var, 0.0));
        if (std::fabs(mean) > 0.01) {
            bad_mean++;
            if (std::fabs(mean) > std::fabs(worst_mean)) {
                worst_mean = mean;
                worst_mean_row = c;
            }
        }
        if (std::fabs(sd - 1.0) > 0.05) {
            bad_std++;
            if (std::fabs(sd - 1.0) > worst_std_dev) {
                worst_std_dev = std::fabs(sd - 1.0);
                worst_std_row = c;
            }
        }
    }
    printf("MTL per-row stats: bad_mean (|mean|>0.01)=%d/%d  worst=%.4f@row%d\n", bad_mean, C, worst_mean,
           worst_mean_row);
    printf("                   bad_std  (|std-1|>0.05)=%d/%d  worst=%.4f@row%d\n", bad_std, C, worst_std_dev,
           worst_std_row);
    // Find row with WORST element-level diff
    int worst_diff_row = 0;
    double worst_row_max = 0;
    for (int c = 0; c < C; c++) {
        double row_max = 0;
        for (int t = 0; t < T; t++) {
            double d = std::fabs((double)dst_cpu[c * T + t] - (double)dst_mtl[c * T + t]);
            if (d > row_max)
                row_max = d;
        }
        if (row_max > worst_row_max) {
            worst_row_max = row_max;
            worst_diff_row = c;
        }
    }
    printf("Worst element diff: row %d max_diff=%.4f\n", worst_diff_row, worst_row_max);
    // Find the worst element WITHIN the row
    int worst_t = 0;
    double worst_t_diff = 0;
    for (int t = 0; t < T; t++) {
        double d = std::fabs((double)dst_cpu[worst_diff_row * T + t] - (double)dst_mtl[worst_diff_row * T + t]);
        if (d > worst_t_diff) {
            worst_t_diff = d;
            worst_t = t;
        }
    }
    printf("worst t=%d  CPU=%+.4f  MTL=%+.4f  diff=%.4f\n", worst_t, dst_cpu[worst_diff_row * T + worst_t],
           dst_mtl[worst_diff_row * T + worst_t], worst_t_diff);
    printf("CPU row[%d] full:", worst_diff_row);
    for (int t = 0; t < T; t++)
        printf(" %+.3f", dst_cpu[worst_diff_row * T + t]);
    printf("\nMTL row[%d] full:", worst_diff_row);
    for (int t = 0; t < T; t++)
        printf(" %+.3f", dst_mtl[worst_diff_row * T + t]);
    printf("\n");
    // Compute CPU vs MTL stats for that row
    double cpu_mean = 0, mtl_mean = 0, cpu_sumsq = 0, mtl_sumsq = 0;
    for (int t = 0; t < T; t++) {
        cpu_mean += dst_cpu[worst_diff_row * T + t];
        mtl_mean += dst_mtl[worst_diff_row * T + t];
        cpu_sumsq += (double)dst_cpu[worst_diff_row * T + t] * dst_cpu[worst_diff_row * T + t];
        mtl_sumsq += (double)dst_mtl[worst_diff_row * T + t] * dst_mtl[worst_diff_row * T + t];
    }
    cpu_mean /= T;
    mtl_mean /= T;
    double cpu_std = std::sqrt(cpu_sumsq / T - cpu_mean * cpu_mean);
    double mtl_std = std::sqrt(mtl_sumsq / T - mtl_mean * mtl_mean);
    printf("CPU stats: mean=%.4f  std=%.4f\n", cpu_mean, cpu_std);
    printf("MTL stats: mean=%.4f  std=%.4f\n", mtl_mean, mtl_std);
    printf("\nDIFF: max_abs=%.6f  rmse=%.6f  rows_with_max_diff>1e-3=%d/%d\n", max_abs, rmse, n_rows_bad, C);
    return n_rows_bad == 0 ? 0 : 1;
}
