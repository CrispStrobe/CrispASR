// Weight-free checks of the actual optional CPU graph node and admission policy.
#include "dia_serial_head.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int checks = 0;
static void require(bool condition, const char* message) {
    ++checks;
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

int main() {
    auto* ctx = ggml_init({2 * 1024 * 1024, nullptr, false});
    require(ctx != nullptr, "context allocation");
    auto* weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 3);
    auto* hidden = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 1, 2);
    auto* half = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 3);
    std::vector<ggml_tensor*> heads{weights};
    require(dia_serial_head::mode(nullptr) == 0 && dia_serial_head::mode("") == 0 && dia_serial_head::mode("off") == 0,
            "default and explicit off");
    require(dia_serial_head::mode("identity") == 1 && dia_serial_head::mode("serial") == 2, "explicit modes");
    for (const char* invalid : {"SERIAL", "1", "invalid"})
        require(dia_serial_head::mode(invalid) == -1, "invalid mode rejects");
    for (int selected : {1, 2}) {
        require(dia_serial_head::supported(selected, true, heads, 64, 3), "CPU F32 admitted");
        require(!dia_serial_head::supported(selected, false, heads, 64, 3), "GPU rejects");
        require(!dia_serial_head::supported(selected, true, {half}, 64, 3), "non-F32 rejects");
        require(!dia_serial_head::supported(selected, true, {nullptr}, 64, 3), "missing head rejects");
        require(!dia_serial_head::supported(selected, true, {}, 64, 3), "empty heads reject");
        require(!dia_serial_head::supported(selected, true, heads, 32, 3), "hidden size mismatch rejects");
        require(!dia_serial_head::supported(selected, true, heads, 64, 4), "vocabulary mismatch rejects");
    }
    require(dia_serial_head::supported(0, false, {half}, 64, 3), "off preserves GPU and other precisions");
    require(!dia_serial_head::supported(-1, true, heads, 64, 3), "invalid mode not admitted");
    require(!dia_serial_head::supported(3, true, heads, 64, 3), "unknown positive mode not admitted");
    auto* w = static_cast<float*>(weights->data);
    auto* x = static_cast<float*>(hidden->data);
    std::memset(w, 0, ggml_nbytes(weights));
    std::memset(x, 0, ggml_nbytes(hidden));
    // Exact analytic witness: sequential separate products give 0; fused
    // multiplication/addition may retain -2^-46. No tolerance hides that bit.
    w[0] = -1.0f;
    w[32] = 1.0f + 0x1p-23f;
    w[64 + 2] = 2.0f;
    w[64 + 3] = -3.0f;
    w[128 + 63] = 0.5f;
    x[0] = x[64] = 1.0f;
    x[32] = x[64 + 32] = 1.0f - 0x1p-23f;
    x[2] = 4.0f;
    x[3] = 5.0f;
    x[63] = 8.0f;
    x[64 + 2] = -1.0f;
    x[64 + 3] = 2.0f;
    x[64 + 63] = -4.0f;
    auto* native = ggml_mul_mat(ctx, weights, hidden);
    require(dia_serial_head::apply(ctx, native, weights, hidden, 0) == native, "off returns original tensor");
    auto* identity = dia_serial_head::apply(ctx, native, weights, hidden, 1);
    auto* serial = dia_serial_head::apply(ctx, native, weights, hidden, 2);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, identity);
    ggml_build_forward_expand(graph, serial);
    const float expected[] = {0.0f, -7.0f, 4.0f, 0.0f, -8.0f, -2.0f};
    for (int threads : {1, 4}) {
        require(ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS, "actual graph computes");
        require(std::memcmp(native->data, identity->data, sizeof(expected)) == 0, "identity byte-exact");
        require(std::memcmp(serial->data, expected, sizeof(expected)) == 0, "serial analytic outputs byte-exact");
    }
    ggml_free(ctx);
    std::printf("DIA_SERIAL_HEAD_UNIT_PASS %d checks\n", checks);
}
