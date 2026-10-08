// Diagnostic only: identical saved Q/K/V, direct CUDA allocation, no scheduler.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
static std::vector<float> read(const std::string& name, size_t n) {
    std::vector<float> x(n);
    std::ifstream f(name, std::ios::binary);
    f.read(reinterpret_cast<char*>(x.data()), n * sizeof(float));
    require(static_cast<size_t>(f.gcount()) == n * sizeof(float), "input size");
    require(f.peek() == EOF, "input trailing bytes");
    return x;
}
struct Arm {
    ggml_context* ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor* output = nullptr;
    std::string name;
    std::vector<double> timings;
    ggml_tensor* inputs[3] = {};
    std::vector<float> original_inputs[3];
    std::vector<float> first_output;
    int verified_repetitions = 0;
    Arm(ggml_backend_t backend, const std::string& dir, const std::string& mode, int t, int h, int d) : name(mode) {
        ggml_init_params ip = {ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false), nullptr, true};
        ctx = ggml_init(ip);
        require(ctx != nullptr, "context");
        auto q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, t, h);
        auto k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, t, h);
        auto v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, t, h);
        ggml_set_input(q);
        ggml_set_input(k);
        ggml_set_input(v);
        // Preserve constants across gallocr reuse and CUDA graph capture.
        // Marking an input alone does not promise it survives its last use.
        ggml_set_output(q);
        ggml_set_output(k);
        ggml_set_output(v);
        inputs[0] = q;
        inputs[1] = k;
        inputs[2] = v;
        const float scale = 1.0f / std::sqrt(static_cast<float>(d));
        if (mode == "eager" || mode == "half-eager") {
            auto scores = ggml_mul_mat(ctx, k, q);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            scores = ggml_soft_max_ext(ctx, scores, nullptr, scale, 0.0f);
            auto vt = ggml_cont(ctx, ggml_transpose(ctx, v));
            output = ggml_mul_mat(ctx, vt, scores);
            ggml_mul_mat_set_prec(output, GGML_PREC_F32);
            output = ggml_cont(ctx, ggml_permute(ctx, output, 0, 2, 1, 3));
        } else {
            output = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
            if (mode == "flash-prec")
                ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
        }
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
            require(ggml_backend_supports_op(backend, ggml_graph_node(graph, i)), "op unsupported by CUDA");
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(ggml_gallocr_alloc_graph(alloc, graph), "allocation");
        const std::string prefix = mode == "half-eager" ? "half-" : "";
        const char* names[] = {"q", "k", "v"};
        for (int i = 0; i < 3; ++i) {
            original_inputs[i] = read(dir + "/" + prefix + names[i] + ".bin", static_cast<size_t>(t) * h * d);
            const auto& x = original_inputs[i];
            ggml_backend_tensor_set(inputs[i], x.data(), 0, x.size() * sizeof(float));
        }
    }
    void compute(ggml_backend_t backend, bool record) {
        if (name == "tile-f32")
            setenv("CRISPASR_DIAG_MIMO_TILE_F32", "1", 1);
        else
            unsetenv("CRISPASR_DIAG_MIMO_TILE_F32");
        auto start = std::chrono::steady_clock::now();
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute");
        ggml_backend_synchronize(backend);
        if (record)
            timings.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        // Readbacks are outside timing. A fast corrupted/no-op replay cannot pass.
        for (int i = 0; i < 3; ++i) {
            std::vector<float> current(original_inputs[i].size());
            ggml_backend_tensor_get(inputs[i], current.data(), 0, current.size() * sizeof(float));
            require(std::memcmp(current.data(), original_inputs[i].data(), current.size() * sizeof(float)) == 0,
                    "resident input overwritten");
        }
        std::vector<float> current(ggml_nelements(output));
        ggml_backend_tensor_get(output, current.data(), 0, current.size() * sizeof(float));
        if (first_output.empty())
            first_output = current;
        require(std::memcmp(current.data(), first_output.data(), current.size() * sizeof(float)) == 0,
                "repeated output changed");
        ++verified_repetitions;
    }
    void save(const std::string& dir) {
        std::vector<float> x(ggml_nelements(output));
        ggml_backend_tensor_get(output, x.data(), 0, x.size() * sizeof(float));
        std::ofstream f(dir + "/" + name + ".bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(float));
        require(f.good(), "output write");
    }
    ~Arm() {
        if (alloc)
            ggml_gallocr_free(alloc);
        if (ctx)
            ggml_free(ctx);
    }
};
int main(int argc, char** argv) {
    try {
        require(argc == 5, "usage: probe DIR T H D");
        const std::string dir = argv[1];
        int t = std::stoi(argv[2]), h = std::stoi(argv[3]), d = std::stoi(argv[4]);
        require(t > 0 && h > 0 && d > 0, "dimensions");
        ggml_backend_load_all();
        auto dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        require(dev != nullptr, "no GPU");
        auto backend = ggml_backend_dev_init(dev, nullptr);
        require(backend != nullptr && std::strstr(ggml_backend_name(backend), "CUDA"), "not CUDA");
        size_t free_before, total, free_after;
        ggml_backend_dev_memory(dev, &free_before, &total);
        std::vector<std::unique_ptr<Arm>> arms;
        for (const char* mode : {"flash", "flash-prec", "eager", "half-eager", "tile-f32"})
            arms.emplace_back(new Arm(backend, dir, mode, t, h, d));
        for (int repeat = 0; repeat < 8; ++repeat) {
            // Alternating order on one backend; no graph rebuild, allocation or transfer in timed region.
            for (int j = 0; j < static_cast<int>(arms.size()); ++j)
                arms[repeat % 2 ? arms.size() - 1 - j : j]->compute(backend, repeat >= 2);
        }
        ggml_backend_dev_memory(dev, &free_after, &total);
        std::ofstream receipt(dir + "/native.json");
        receipt << "{\"backend\":\"" << ggml_backend_name(backend) << "\",\"free_before\":" << free_before
                << ",\"free_after\":" << free_after << ",\"total\":" << total << ",\"arms\":{";
        for (size_t i = 0; i < arms.size(); ++i) {
            auto& arm = *arms[i];
            arm.save(dir);
            if (i)
                receipt << ',';
            receipt << '"' << arm.name << "\":{\"allocation_bytes\":" << ggml_gallocr_get_buffer_size(arm.alloc, 0)
                    << ",\"verified_repetitions\":" << arm.verified_repetitions << ",\"seconds\":[";
            for (size_t k = 0; k < arm.timings.size(); ++k) {
                if (k)
                    receipt << ',';
                receipt << arm.timings[k];
            }
            receipt << "]}";
        }
        receipt << "}}\n";
        require(receipt.good(), "receipt write");
        arms.clear();
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
