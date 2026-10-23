// Actual shared Granite attention producer/consumer vs original-module Q/K.
// Input: two int32 dimensions, int32 positions, then F32 q/k/rotated-q/rotated-k.
#include "core/ggml_cpu_backend.h"
#include "core/granite_llm.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4)
        return 2;
    const bool cuda = argc == 4 && std::strcmp(argv[3], "cuda") == 0;
    if (argc == 4 && !cuda && std::strcmp(argv[3], "cpu") != 0)
        return 2;
    FILE* f = std::fopen(argv[1], "rb");
    if (!f)
        return 3;
    int hdr[2];
    if (std::fread(hdr, sizeof(int), 2, f) != 2 || hdr[0] != 128 || hdr[1] != 7)
        return 4;
    const int D = hdr[0], T = hdr[1], N = D * T;
    std::vector<int> positions(T);
    if (std::fread(positions.data(), sizeof(int), T, f) != size_t(T))
        return 5;
    std::vector<float> q(N), k(N), qr(N), kr(N);
    for (auto* v : {&q, &k, &qr, &kr})
        if (std::fread(v->data(), sizeof(float), N, f) != size_t(N))
            return 6;
    std::fclose(f);
    for (int row = 0; row < T; ++row)
        for (int col = 0; col < D; ++col)
            if (k[row * D + col] != q[row * D + D - 1 - col])
                return 13;
    const int mode = std::atoi(argv[2]);
    const bool yarn = mode != 0;
    ggml_backend_t be = nullptr;
    if (cuda) {
        ggml_backend_load_all();
        auto* device = ggml_backend_dev_by_name("CUDA0");
        if (device)
            be = ggml_backend_dev_init(device, nullptr);
        // Direct backend dispatch below has no scheduler CPU fallback.
        if (!be || std::strcmp(ggml_backend_name(be), "CUDA0") != 0)
            return 14;
    } else {
        be = core_cpu_backend::init();
    }
    auto* ctx = ggml_init({1024 * 1024, nullptr, true});
    if (!be || !ctx)
        return 7;
    auto* gf = ggml_new_graph(ctx);
    auto* p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    auto* Q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, T);
    core_granite_llm::Hparams hp = {};
    hp.n_heads = hp.n_kv_heads = 1;
    hp.head_dim = D;
    hp.rope_theta = 10000.0f;
    hp.attention_multiplier = 1.0f / D;
    if (yarn &&
        !core_granite_llm::configure_yarn(hp.rope, 4.0f, mode == 2 ? 16384 : 4096, 1.138629436111989f, 32.0f, 1.0f))
        return 11;
    auto params = core_granite_llm::attention_params(hp);
    if (mode == 3)
        params.rope_attn_factor = 1.138629436111989f;
    if (mode == 4)
        params.rope_type = GGML_ROPE_TYPE_NORMAL;
    params.eager_f32_attn = true;
    auto* Wq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, D);
    auto* Wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, D);
    auto* kvk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, T, 1, 1);
    auto* kvv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, T, 1, 1);
    auto* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, T, T);
#ifdef _WIN32
    _putenv_s("CRISPASR_CORE_ATTN_DUMP_FA_LAYER", "0");
#else
    setenv("CRISPASR_CORE_ATTN_DUMP_FA_LAYER", "0", 1);
#endif
    auto* result = core_attn::kv_self_attn(ctx, gf, ggml_reshape_2d(ctx, Q, D, T), Wq, Wk, Wq, Wq, nullptr, nullptr, p,
                                           mask, kvk, kvv, 0, 0, params);
    ggml_build_forward_expand(gf, result);
    auto* Qo = ggml_get_tensor(ctx, "DBG_Q_post_rope");
    auto* Ko = ggml_get_tensor(ctx, "DBG_Kfull");
    if (!Qo || !Ko)
        return 12;
    auto buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf)
        return 8;
    std::vector<float> eye(D * D, 0), flip(D * D, 0);
    for (int i = 0; i < D; ++i) {
        eye[i * D + i] = 1;
        flip[i * D + D - 1 - i] = 1;
    }
    ggml_backend_buffer_clear(buf, 0);
    // Clear before uploading inputs, since cache and graph share this buffer.
    ggml_backend_tensor_set(p, positions.data(), 0, T * sizeof(int));
    ggml_backend_tensor_set(Q, q.data(), 0, N * sizeof(float));
    ggml_backend_tensor_set(Wq, eye.data(), 0, D * D * sizeof(float));
    ggml_backend_tensor_set(Wk, flip.data(), 0, D * D * sizeof(float));
    std::vector<ggml_fp16_t> mask_values(T * T, 0);
    for (int row = 0; row < T; ++row)
        for (int col = row + 1; col < T; ++col)
            mask_values[row * T + col] = ggml_fp32_to_fp16(-INFINITY);
    ggml_backend_tensor_set(mask, mask_values.data(), 0, T * T * sizeof(ggml_fp16_t));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS)
        return 9;
    std::vector<float> got(N);
    double maxabs = 0, lowmaxabs = 0, mine2 = 0, ref2 = 0, err2 = 0;
    for (int j = 0; j < 2; ++j) {
        ggml_backend_tensor_get(j == 0 ? Qo : Ko, got.data(), 0, N * sizeof(float));
        auto& expected = j == 0 ? qr : kr;
        for (int i = 0; i < N; ++i) {
            if (!std::isfinite(got[i]))
                return 10;
            const double diff = got[i] - expected[i];
            maxabs = std::max(maxabs, std::fabs(diff));
            if (i < 3 * D)
                lowmaxabs = std::max(lowmaxabs, std::fabs(diff));
            mine2 += got[i] * got[i];
            ref2 += expected[i] * expected[i];
            err2 += diff * diff;
        }
    }
    std::printf("{\"backend\":\"%s\",\"mode\":%d,\"maxabs\":%.10g,\"low_position_maxabs\":%.10g,"
                "\"norm_ratio\":%.10g,\"relative_l2\":%.10g}\n",
                ggml_backend_name(be), mode, maxabs, lowmaxabs, std::sqrt(mine2 / ref2), std::sqrt(err2 / ref2));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    // CUDA builds use -use_fast_math: powf/sinf/cosf use lower-accuracy
    // intrinsics, and phase roundoff grows with position. This normalized
    // fixture permits 0.1% RMS / 0.5% peak error across CPU and CUDA while
    // retaining the independent 1 ppm magnitude invariant. All four wrong
    // architecture controls must fail this SAME budget. Full model gates
    // remain cosine >= .99, norm within 1%, and exact decoded output.
    return maxabs < 0.005 && lowmaxabs < 2e-5 && std::fabs(std::sqrt(mine2 / ref2) - 1) < 1e-6 &&
                   std::sqrt(err2 / ref2) < 0.001
               ? 0
               : 1;
}
