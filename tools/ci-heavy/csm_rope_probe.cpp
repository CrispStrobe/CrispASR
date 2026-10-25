// Bounded, weight-free probe of the actual ggml F32 NEOX RoPE operator.
// Inputs are physically exported original-operator arrays, never regenerated.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "core/ggml_cpu_backend.h"

#ifdef CSM_GENERATED_ROPE
#include "csm_rope.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

template <typename T> static std::vector<T> read_array(const fs::path& path, size_t count) {
    if (fs::file_size(path) != count * sizeof(T))
        throw std::runtime_error("input array size mismatch: " + path.filename().string());
    std::vector<T> values(count);
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(values.data()), (std::streamsize)(count * sizeof(T)));
    if (!file)
        throw std::runtime_error("cannot read complete input array");
    return values;
}

static void write_array(const fs::path& path, const float* data, size_t count) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data), (std::streamsize)(count * sizeof(float)));
    if (!file)
        throw std::runtime_error("cannot write complete output array");
}

int main(int argc, char** argv) {
    try {
        if (argc != 10 && argc != 13)
            throw std::runtime_error(
                "input-dir output-dir dim q-heads k-heads positions original-context theta mode [factor low high]");
        const fs::path input = argv[1], output = argv[2];
        const int dim = std::stoi(argv[3]), qh = std::stoi(argv[4]), kh = std::stoi(argv[5]);
        const int positions = std::stoi(argv[6]), context = std::stoi(argv[7]);
        const float theta = std::stof(argv[8]);
        const std::string mode = argv[9];
        if (dim <= 0 || dim > 256 || dim % 2 || qh <= 0 || qh > 64 || kh <= 0 || kh > 64 || positions <= 0 ||
            positions > 64 || context <= 0 || !std::isfinite(theta) || theta <= 0 ||
            (size_t)dim * qh * positions > 65536 || (size_t)dim * kh * positions > 65536)
            throw std::runtime_error("invalid bounded RoPE shape or configuration");
        const auto q = read_array<float>(input / "q.bin", (size_t)dim * qh * positions);
        const auto k = read_array<float>(input / "k.bin", (size_t)dim * kh * positions);
        const auto pos = read_array<int32_t>(input / "positions.bin", (size_t)positions);
        std::vector<float> factors((size_t)dim / 2, 1.0f);
        if (mode == "original") {
            factors = read_array<float>(input / "freq_factors.bin", factors.size());
        } else if (mode == "generated") {
#ifdef CSM_GENERATED_ROPE
            if (argc != 13)
                throw std::runtime_error("generated mode requires factor, low and high");
            factors = csm_rope::frequency_factors(dim, theta, std::stof(argv[10]), std::stof(argv[11]),
                                                  std::stof(argv[12]), context);
#else
            throw std::runtime_error("generated frequency helper is not compiled in this probe");
#endif
        } else if (mode != "unscaled") {
            throw std::runtime_error("unknown probe mode");
        }
        if (factors.size() != (size_t)dim / 2)
            throw std::runtime_error("frequency helper returned an invalid shape");
        for (float factor : factors)
            if (!std::isfinite(factor) || factor <= 0)
                throw std::runtime_error("invalid frequency factor");

        const char* requested = std::getenv("CRISPASR_CSM_ROPE_DEVICE");
        const std::string device = requested ? requested : "cpu";
        if (device != "cpu" && device != "cuda")
            throw std::runtime_error("probe device must be cpu or cuda");
        ggml_backend_load_all();
        const auto wanted = device == "cuda" ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU;
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_init_by_type(wanted, nullptr),
                                                                            ggml_backend_free);
        if (!backend || ggml_backend_dev_type(ggml_backend_get_device(backend.get())) != wanted)
            throw std::runtime_error("requested backend unavailable; fallback is forbidden");
        if (device == "cuda" && std::strncmp(ggml_backend_name(backend.get()), "CUDA", 4) != 0)
            throw std::runtime_error("GPU backend is not CUDA");
        if (device == "cpu")
            core_cpu_backend::set_n_threads(backend.get(), 1);
        ggml_init_params params = {8 * 1024 * 1024, nullptr, true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
        if (!ctx)
            throw std::runtime_error("cannot allocate tiny CPU context");
        auto* qt = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, dim, qh, positions);
        auto* kt = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, dim, kh, positions);
        auto* pt = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, positions);
        auto* ft = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, dim / 2);
        auto* qr = ggml_rope_ext(ctx.get(), qt, pt, mode == "unscaled" ? nullptr : ft, dim, GGML_ROPE_TYPE_NEOX,
                                 context, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        auto* kr = ggml_rope_ext(ctx.get(), kt, pt, mode == "unscaled" ? nullptr : ft, dim, GGML_ROPE_TYPE_NEOX,
                                 context, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        auto* graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, qr);
        ggml_build_forward_expand(graph, kr);
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
        if (!buffer)
            throw std::runtime_error("requested backend tensor allocation failed");
        ggml_backend_tensor_set(qt, q.data(), 0, q.size() * sizeof(float));
        ggml_backend_tensor_set(kt, k.data(), 0, k.size() * sizeof(float));
        ggml_backend_tensor_set(pt, pos.data(), 0, pos.size() * sizeof(int32_t));
        ggml_backend_tensor_set(ft, factors.data(), 0, factors.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("requested backend RoPE graph failed");
        ggml_backend_synchronize(backend.get());
        std::vector<float> q_result(q.size()), k_result(k.size());
        ggml_backend_tensor_get(qr, q_result.data(), 0, q_result.size() * sizeof(float));
        ggml_backend_tensor_get(kr, k_result.data(), 0, k_result.size() * sizeof(float));
        fs::create_directories(output);
        write_array(output / "q_rotated.bin", q_result.data(), q_result.size());
        write_array(output / "k_rotated.bin", k_result.data(), k_result.size());
        write_array(output / "freq_factors.bin", factors.data(), factors.size());
        std::printf("{\"backend\":\"%s\",\"mode\":\"%s\",\"positions\":%d,\"fallback\":false}\n",
                    ggml_backend_name(backend.get()), mode.c_str(), positions);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "csm_rope_probe: %s\n", error.what());
        return 1;
    }
}
