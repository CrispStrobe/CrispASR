// Model-free CPU attention invariant: identical V rows must stay identical.
#include "ggml-cpu.h"
#include "ggml.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

static void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static void save(const std::filesystem::path& path, const ggml_tensor* tensor) {
    FILE* file = std::fopen(path.c_str(), "wb");
    require(file != nullptr, "open fresh tensor output");
    require(std::fwrite(tensor->data, 1, ggml_nbytes(tensor), file) == ggml_nbytes(tensor), "write tensor");
    require(std::fclose(file) == 0, "close tensor");
}

int main(int argc, char** argv) {
    require(argc == 2, "fresh_output_directory");
    const std::filesystem::path output(argv[1]);
    require(!std::filesystem::exists(output), "preserve earlier evidence");
    std::filesystem::create_directories(output);
    ggml_cpu_init();
    require(ggml_get_type_traits_cpu(GGML_TYPE_F16)->vec_dot_type == GGML_TYPE_F32,
            "actual patched F16-weight/F32-operand CPU trait");
    constexpr int d = 32, keys = 71, heads = 2;
    for (int pattern = 0; pattern < 2; ++pattern) {
        for (int queries : {1, 63, 64, 71}) {
            for (int threads : {1, 4}) {
                for (int half : {0, 1}) {
                    for (int reference : {0, 1}) {
                        const auto folder = output / ("p" + std::to_string(pattern) + "-t" + std::to_string(queries) +
                                                      "-n" + std::to_string(threads) + "-f16" + std::to_string(half) +
                                                      "-ref" + std::to_string(reference));
                        std::filesystem::create_directories(folder);
                        ggml_init_params params{16 * 1024 * 1024, nullptr, false};
                        auto* ctx = ggml_init(params);
                        require(ctx != nullptr, "bounded context");
                        const auto type = half ? GGML_TYPE_F16 : GGML_TYPE_F32;
                        auto* q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, queries, heads);
                        auto* k = ggml_new_tensor_3d(ctx, type, d, keys, heads);
                        auto* v = ggml_new_tensor_3d(ctx, type, d, keys, heads);
                        auto* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, keys, queries);
                        // F32 and F16 arms own exactly the same representable K/V values.
                        for (int h = 0; h < heads; ++h) {
                            for (int t = 0; t < queries; ++t) {
                                for (int c = 0; c < d; ++c) {
                                    ((float*)q->data)[(h * queries + t) * d + c] =
                                        pattern ? std::sin(float(c + 3 * h + 1) * .17f) : 0.0f;
                                }
                            }
                            for (int t = 0; t < keys; ++t) {
                                for (int c = 0; c < d; ++c) {
                                    const int index = (h * keys + t) * d + c;
                                    const float kval = pattern ? std::cos(float(t * 3 + c + h) * .11f) : 0.0f;
                                    const float vval = pattern ? std::sin(float(t + c * 7 + h) * .13f)
                                                               : float(c + h + 1) * .1f / float(d);
                                    const auto kh = ggml_fp32_to_fp16(kval), vh = ggml_fp32_to_fp16(vval);
                                    if (half) {
                                        ((ggml_fp16_t*)k->data)[index] = kh;
                                        ((ggml_fp16_t*)v->data)[index] = vh;
                                    } else {
                                        ((float*)k->data)[index] = ggml_fp16_to_fp32(kh);
                                        ((float*)v->data)[index] = ggml_fp16_to_fp32(vh);
                                    }
                                }
                            }
                        }
                        for (int t = 0; t < queries; ++t) {
                            for (int i = 0; i < keys; ++i) {
                                // Identical mask/query across rows isolates the dispatch boundary.
                                ((ggml_fp16_t*)mask->data)[t * keys + i] =
                                    ggml_fp32_to_fp16(pattern && i % 7 == 0 ? -INFINITY : 0.0f);
                            }
                        }
                        const float scale = 1.0f / std::sqrt(float(d));
                        auto* scores = ggml_mul_mat(ctx, k, q);
                        ggml_prec_set_acc(scores, GGML_PREC_F32);
                        auto* probabilities = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
                        auto* vt = ggml_cont(ctx, ggml_transpose(ctx, v));
                        auto* eager = ggml_mul_mat(ctx, vt, probabilities);
                        eager = ggml_cont(ctx, ggml_permute(ctx, eager, 0, 2, 1, 3));
                        auto* flash = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
                        auto* graph = ggml_new_graph_custom(ctx, 64, false);
                        for (auto* result : {scores, probabilities, eager, flash}) {
                            ggml_set_output(result);
                            ggml_build_forward_expand(graph, result);
                        }
                        auto plan = ggml_graph_plan(graph, threads, nullptr);
                        plan.use_ref = reference != 0;
                        std::vector<unsigned char> work(plan.work_size);
                        plan.work_data = work.data();
                        require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS, "actual CPU graph compute");
                        save(folder / "q.f32", q);
                        save(folder / (half ? "k.f16" : "k.f32"), k);
                        save(folder / (half ? "v.f16" : "v.f32"), v);
                        save(folder / "mask.f16", mask);
                        save(folder / "scores.f32", scores);
                        save(folder / "probabilities.f32", probabilities);
                        save(folder / "eager.f32", eager);
                        save(folder / "flash.f32", flash);
                        ggml_free(ctx);
                        std::printf("ACTUAL_ATTENTION_CELL_PASS %s d=%d keys=%d heads=%d\n", folder.filename().c_str(),
                                    d, keys, heads);
                    }
                }
            }
        }
    }
    std::puts("MODEL_FREE_ATTENTION_CAPTURE_COMPLETE_NO_MODEL_ACCEPTANCE");
}
