#pragma once

// Exact F32 tensor inventory for NVIDIA NemotronLabs RE-USE revision
// 31f82f2efad1095034492f236e8dac9239ba713b. Shapes here follow the original
// PyTorch order; a GGML loader must reverse them for its tensor dimensions and
// independently enforce F32 for every tensor.
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace reuse_model_inventory {
using Shape = std::vector<int64_t>;
using Inventory = std::map<std::string, Shape>;

inline Inventory expected() {
    Inventory result;
    const auto add = [&](const std::string& name, Shape shape) { result.emplace(name, std::move(shape)); };
    const auto norm = [&](const std::string& prefix) {
        add(prefix + ".1.weight", {64});
        add(prefix + ".1.bias", {64});
        add(prefix + ".2.weight", {64});
    };
    const auto dense = [&](const std::string& prefix) {
        for (int i = 0; i < 4; ++i) {
            const std::string path = prefix + ".dense_block." + std::to_string(i);
            add(path + ".0.weight", {64, 64 * (i + 1), 3, 3});
            add(path + ".0.bias", {64});
            norm(path);
        }
    };
    const auto mamba = [&](const std::string& prefix) {
        add(prefix + ".in_proj.weight", {512, 64});
        add(prefix + ".conv1d.weight", {256, 1, 4});
        add(prefix + ".conv1d.bias", {256});
        add(prefix + ".x_proj.weight", {36, 256});
        add(prefix + ".dt_proj.weight", {256, 4});
        add(prefix + ".dt_proj.bias", {256});
        add(prefix + ".A_log", {256, 16});
        add(prefix + ".D", {256});
        add(prefix + ".out_proj.weight", {64, 256});
    };
    const auto bidirectional = [&](const std::string& prefix) {
        mamba(prefix + ".forward_blocks");
        mamba(prefix + ".backward_blocks");
        add(prefix + ".output_proj.weight", {64, 128});
        add(prefix + ".output_proj.bias", {64});
        add(prefix + ".norm.weight", {64});
        add(prefix + ".norm.bias", {64});
    };
    const auto up = [&](const std::string& prefix, int scale) {
        add(prefix + ".0.conv.weight", {64 * scale, 64, 1, 3});
        add(prefix + ".0.conv.bias", {64 * scale});
        norm(prefix);
    };
    const auto head = [&](const std::string& prefix) {
        add(prefix + ".weight", {1, 64, 1, 1});
        add(prefix + ".bias", {1});
    };
    add("dense_encoder.dense_conv_1.0.weight", {64, 2, 1, 1});
    add("dense_encoder.dense_conv_1.0.bias", {64});
    norm("dense_encoder.dense_conv_1");
    dense("dense_encoder.dense_block");
    add("dense_encoder.dense_conv_2.0.weight", {64, 64, 1, 3});
    add("dense_encoder.dense_conv_2.0.bias", {64});
    norm("dense_encoder.dense_conv_2");
    for (int i = 0; i < 30; ++i) {
        const std::string prefix = "TSMamba." + std::to_string(i);
        bidirectional(prefix + ".time_mamba");
        bidirectional(prefix + ".freq_mamba");
    }
    for (const std::string prefix : {"mask_decoder", "phase_decoder"}) {
        dense(prefix + ".dense_block");
        up(prefix + ".up_conv1", 2);
        up(prefix + ".up_conv2", 4);
    }
    head("mask_decoder.final_conv");
    head("phase_decoder.phase_conv_r");
    head("phase_decoder.phase_conv_i");
    return result;
}

inline bool validate_shapes(const Inventory& actual) {
    return actual == expected();
}
} // namespace reuse_model_inventory
