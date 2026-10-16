#include "dia_serial_head.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}
static void read(const char* path, ggml_tensor* tensor) {
    FILE* file = std::fopen(path, "rb");
    require(file != nullptr, "input opens");
    require(std::fread(tensor->data, 1, ggml_nbytes(tensor), file) == ggml_nbytes(tensor), "input size");
    require(std::fgetc(file) == EOF, "input closure");
    require(std::fclose(file) == 0, "input closes");
}
int main(int argc, char** argv) {
    require(argc == 6, "native weights, hidden, mode, threads, output");
    const int selected = dia_serial_head::mode(argv[3]);
    const int threads = std::atoi(argv[4]);
    require(selected >= 0 && (threads == 1 || threads == 4), "mode and threads");
    auto* ctx = ggml_init({128 * 1024 * 1024, nullptr, false});
    require(ctx != nullptr, "context");
    auto* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 9 * 1028);
    auto* hidden = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 1, 2);
    read(argv[1], weight);
    read(argv[2], hidden);
    auto* original = ggml_mul_mat(ctx, weight, hidden);
    auto* result = dia_serial_head::apply(ctx, original, weight, hidden, selected);
    ggml_set_output(result);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    require(ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS, "graph compute");
    FILE* output = std::fopen(argv[5], "wbx");
    require(output != nullptr, "fresh output");
    require(std::fwrite(result->data, 1, ggml_nbytes(result), output) == ggml_nbytes(result), "output bytes");
    require(std::fclose(output) == 0, "output closes");
    ggml_free(ctx);
    std::puts("DIA_SERIAL_HEAD_GRAPH_PASS");
}
