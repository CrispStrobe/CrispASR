// Exercise the real GGUF comparison gate without a model or injected stages.
#include "../examples/cli/reuse_diff.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <limits>

int main(int argc, char** argv) {
    if (argc != 2)
        return 1;
    ggml_init_params params = {16 * ggml_tensor_overhead() + 4096, nullptr, false};
    ggml_context* tensors = ggml_init(params);
    gguf_context* archive = gguf_init_empty();
    if (!tensors || !archive)
        return 2;
    const auto add = [&](const char* name, const std::vector<float>& data, int64_t rows) {
        auto* tensor = ggml_new_tensor_2d(tensors, GGML_TYPE_F32, data.size() / rows, rows);
        ggml_set_name(tensor, name);
        std::memcpy(tensor->data, data.data(), data.size() * sizeof(float));
        gguf_add_tensor(archive, tensor);
    };
    const std::vector<float> rows = {1000, 1000, 0.001f, 0.001f};
    const std::vector<float> phase = {3.1415927f, 0.5f};
    add("rows", rows, 2);
    add("phase", phase, 1);
    const bool written = gguf_write_to_file(archive, argv[1], false);
    gguf_free(archive);
    ggml_free(tensors);
    if (!written)
        return 3;
    crispasr_diff::Ref ref;
    const bool loaded = ref.load(argv[1]);
    std::remove(argv[1]);
    if (!loaded)
        return 4;
    const auto check = [&](const char* name, const std::vector<float>& data, const std::vector<int64_t>& shape) {
        return reuse_diff::compare(ref, name, data, shape, "control/", 0.002);
    };
    if (!check("rows", rows, {2, 2}))
        return 5;
    // Aggregate RMS and cosine hide this tiny-energy row's wrong direction.
    if (check("rows", {1000, 1000, 0.001f, -0.001f}, {2, 2}))
        return 6;
    // A 1% scale defect is inside the amplitude-ratio band, but fails RMS.
    if (check("rows", {1010, 1010, 0.00101f, 0.00101f}, {2, 2}))
        return 7;
    if (!check("phase", phase, {2}) || check("phase", {-phase[0], phase[1]}, {2}))
        return 8;
    if (check("rows", rows, {4, 1}) || check("missing", rows, {2, 2}))
        return 9;
    auto nonfinite = rows;
    nonfinite[0] = std::numeric_limits<float>::quiet_NaN();
    if (check("rows", nonfinite, {2, 2}))
        return 10;
    const std::string wave_path = std::string(argv[1]) + ".wave";
    if (!reuse_diff::write_waveform(wave_path.c_str(), rows))
        return 11;
    FILE* wave = std::fopen(wave_path.c_str(), "rb");
    std::vector<float> readback(rows.size());
    if (!wave || std::fread(readback.data(), sizeof(float), readback.size(), wave) != rows.size())
        return 12;
    std::fclose(wave);
    std::remove(wave_path.c_str());
    if (readback != rows || reuse_diff::write_waveform((wave_path + "/invalid").c_str(), rows))
        return 13;
    std::puts("RE-USE actual diff gate: identity and six independent rejection controls passed");
    return 0;
}
