#include "../src/core/reuse_model_loader.h"
#include "../src/core/ggml_cpu_backend.h"
#include <cstdio>
#include <fstream>

static bool same(std::ifstream& file, const std::vector<float>& actual) {
    std::vector<float> expected(actual.size());
    return bool(file.read(reinterpret_cast<char*>(expected.data()), expected.size() * sizeof(float))) &&
           expected == actual;
}
int main(int argc, char** argv) {
    auto* backend = core_cpu_backend::init();
    if (!backend)
        return 1;
    reuse_model_loader::Model model;
    model.encoder.emplace_back();
    if (reuse_model_loader::load(nullptr, backend, model) || reuse_model_loader::load("", backend, model) ||
        reuse_model_loader::load("/nonexistent/reuse.gguf", backend, model) ||
        reuse_model_loader::load("invalid", nullptr, model) || model.encoder.size() != 1) {
        ggml_backend_free(backend);
        return 2;
    }
    if (argc == 1) {
        ggml_backend_free(backend);
        std::puts("RE-USE loader null and missing-file controls passed");
        return 0;
    }
    if (argc != 7) {
        ggml_backend_free(backend);
        return 3;
    }
    for (int i = 1; i <= 4; ++i)
        if (reuse_model_loader::load(argv[i], backend, model) || model.encoder.size() != 1) {
            ggml_backend_free(backend);
            return 4;
        }
    if (!reuse_model_loader::load(argv[5], backend, model)) {
        ggml_backend_free(backend);
        return 5;
    }
    ggml_backend_free(backend);
    if (model.encoder.size() != 6 || model.blocks.size() != 30 || model.magnitude.dense.size() != 4 ||
        model.phase.dense.size() != 4)
        return 6;
    std::ifstream probe(argv[6], std::ios::binary);
    if (!same(probe, model.encoder[0].weights) || !same(probe, model.blocks[0].time.forward.in_proj) ||
        !same(probe, model.blocks[29].frequency.backward.a_log) || !same(probe, model.magnitude.time.weights) ||
        !same(probe, model.phase.second_head) || probe.peek() != std::char_traits<char>::eof())
        return 7;
    std::puts("RE-USE loader: strict rejection and original tensor probes passed");
    return 0;
}
