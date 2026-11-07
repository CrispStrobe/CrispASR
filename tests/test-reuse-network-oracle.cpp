#include "../src/core/reuse_reference_network.h"
#include "../src/core/ggml_cpu_backend.h"
#include <cstdio>
#include <fstream>

static bool write(const char* name, const std::vector<float>& values) {
    std::ofstream file(name, std::ios::binary);
    return bool(file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float)));
}
int main(int argc, char** argv) {
    if (argc != 3)
        return 1;
    std::ifstream input(argv[2], std::ios::binary);
    std::vector<float> audio(800);
    if (!input.read(reinterpret_cast<char*>(audio.data()), audio.size() * sizeof(float)) ||
        input.peek() != std::char_traits<char>::eof())
        return 2;
    reuse_frontend::Spectrum spectrum;
    if (!reuse_frontend::analyze(audio, 8000, spectrum) ||
        !write("reuse-trace-frontend_magnitude.f32le", spectrum.magnitude) ||
        !write("reuse-trace-frontend_phase.f32le", spectrum.phase))
        return 8;
    auto* backend = core_cpu_backend::init();
    if (!backend)
        return 3;
    reuse_model_loader::Model model;
    const bool loaded = reuse_model_loader::load(argv[1], backend, model);
    ggml_backend_free(backend);
    if (!loaded)
        return 4;
    std::vector<float> output;
    reuse_reference_network::Trace trace;
    if (!reuse_reference_network::run(model, audio, 8000, output, &trace) || output.size() != audio.size())
        return 5;
    const auto shape_ok = [](const reuse_dense_encoder::Tensor& tensor) {
        return tensor.channels == 64 && tensor.time == 6 && tensor.freq == 81;
    };
    if (!shape_ok(trace.encoder) || !shape_ok(trace.block0) || !shape_ok(trace.block14) || !shape_ok(trace.block29))
        return 7;
    if (!write("reuse-network-actual.f32le", output) || !write("reuse-trace-encoder.f32le", trace.encoder.values) ||
        !write("reuse-trace-block0.f32le", trace.block0.values) ||
        !write("reuse-trace-block14.f32le", trace.block14.values) ||
        !write("reuse-trace-block29.f32le", trace.block29.values))
        return 6;
    std::puts("REUSE_NETWORK_ORACLE_WRITTEN");
    return 0;
}
