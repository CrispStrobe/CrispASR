#include "reuse_model_loader.h"
#include "gguf_loader.h"
#include "reuse_model_inventory.h"
#include <ggml.h>
#include <gguf.h>
#include <algorithm>
#include <array>
#include <memory>
#include <utility>

namespace reuse_model_loader {
namespace {
using Values = std::map<std::string, std::vector<float>>;

std::vector<float> take(Values& values, const std::string& name) {
    auto& source = values.at(name);
    auto result = std::move(source);
    source.clear();
    return result;
}

reuse_dense_encoder::Layer dense_layer(Values& values, const std::string& prefix, size_t inputs, size_t kt, size_t kf) {
    reuse_dense_encoder::Layer layer;
    layer.in_channels = inputs;
    layer.out_channels = 64;
    layer.kernel_time = kt;
    layer.kernel_freq = kf;
    layer.weights = take(values, prefix + ".0.weight");
    layer.bias = take(values, prefix + ".0.bias");
    layer.norm_weight = take(values, prefix + ".1.weight");
    layer.norm_bias = take(values, prefix + ".1.bias");
    layer.slope = take(values, prefix + ".2.weight");
    return layer;
}

std::vector<reuse_dense_encoder::Layer> dense_stack(Values& values, const std::string& prefix) {
    std::vector<reuse_dense_encoder::Layer> result;
    for (size_t i = 0; i < 4; ++i)
        result.push_back(dense_layer(values, prefix + ".dense_block." + std::to_string(i), 64 * (i + 1), 3, 3));
    return result;
}

reuse_mamba::Weights mamba(Values& values, const std::string& prefix) {
    reuse_mamba::Weights w;
    w.model = 64;
    w.inner = 256;
    w.rank = 4;
    w.states = 16;
    w.kernel = 4;
    w.in_proj = take(values, prefix + ".in_proj.weight");
    w.conv = take(values, prefix + ".conv1d.weight");
    w.conv_bias = take(values, prefix + ".conv1d.bias");
    w.x_proj = take(values, prefix + ".x_proj.weight");
    w.dt_proj = take(values, prefix + ".dt_proj.weight");
    w.dt_bias = take(values, prefix + ".dt_proj.bias");
    w.a_log = take(values, prefix + ".A_log");
    w.d = take(values, prefix + ".D");
    w.out_proj = take(values, prefix + ".out_proj.weight");
    return w;
}

reuse_bimamba::Weights bidirectional(Values& values, const std::string& prefix) {
    reuse_bimamba::Weights w;
    w.forward = mamba(values, prefix + ".forward_blocks");
    w.backward = mamba(values, prefix + ".backward_blocks");
    w.projection = take(values, prefix + ".output_proj.weight");
    w.projection_bias = take(values, prefix + ".output_proj.bias");
    w.norm_weight = take(values, prefix + ".norm.weight");
    w.norm_bias = take(values, prefix + ".norm.bias");
    return w;
}

reuse_decoder::UpLayer up(Values& values, const std::string& prefix, size_t scale) {
    reuse_decoder::UpLayer layer;
    layer.channels = 64;
    layer.scale = scale;
    layer.weights = take(values, prefix + ".0.conv.weight");
    layer.bias = take(values, prefix + ".0.conv.bias");
    layer.norm_weight = take(values, prefix + ".1.weight");
    layer.norm_bias = take(values, prefix + ".1.bias");
    layer.slope = take(values, prefix + ".2.weight");
    return layer;
}

Decoder decoder(Values& values, const std::string& prefix, bool phase) {
    Decoder result;
    result.dense = dense_stack(values, prefix + ".dense_block");
    result.frequency = up(values, prefix + ".up_conv1", 2);
    result.time = up(values, prefix + ".up_conv2", 4);
    const std::string first = prefix + (phase ? ".phase_conv_r" : ".final_conv");
    result.first_head = take(values, first + ".weight");
    result.first_bias = take(values, first + ".bias").at(0);
    if (phase) {
        result.second_head = take(values, prefix + ".phase_conv_i.weight");
        result.second_bias = take(values, prefix + ".phase_conv_i.bias").at(0);
    }
    return result;
}
} // namespace

bool load(const char* path, ggml_backend_t backend, Model& output) {
    if (!path || !*path || !backend)
        return false;
    try {
        const auto inventory = reuse_model_inventory::expected();
        std::unique_ptr<gguf_context, decltype(&core_gguf::free_metadata)> metadata(core_gguf::open_metadata(path),
                                                                                    core_gguf::free_metadata);
        if (!metadata || core_gguf::kv_str(metadata.get(), "general.architecture", "") != "reuse" ||
            core_gguf::kv_str(metadata.get(), "reuse.source_revision", "") !=
                "31f82f2efad1095034492f236e8dac9239ba713b" ||
            gguf_get_n_tensors(metadata.get()) != static_cast<int64_t>(inventory.size()))
            return false;
        for (const auto& entry : inventory) {
            const int64_t id = gguf_find_tensor(metadata.get(), entry.first.c_str());
            if (id < 0 || gguf_get_tensor_type(metadata.get(), id) != GGML_TYPE_F32)
                return false;
            const int64_t* dimensions = gguf_get_tensor_ne(metadata.get(), id);
            std::array<int64_t, 4> expected{1, 1, 1, 1};
            std::reverse_copy(entry.second.begin(), entry.second.end(), expected.begin());
            if (!dimensions || !std::equal(expected.begin(), expected.end(), dimensions))
                return false;
        }
        metadata.reset();
        core_gguf::WeightLoad loaded;
        if (!core_gguf::load_weights(path, backend, "reuse", loaded))
            return false;
        struct Guard {
            core_gguf::WeightLoad& value;
            ~Guard() { core_gguf::free_weights(value); }
        } guard{loaded};
        Values values;
        for (const auto& entry : inventory) {
            const auto it = loaded.tensors.find(entry.first);
            if (it == loaded.tensors.end() || !it->second || it->second->type != GGML_TYPE_F32 ||
                !ggml_is_contiguous(it->second))
                return false;
            auto& data = values[entry.first];
            data.resize(static_cast<size_t>(ggml_nelements(it->second)));
            ggml_backend_tensor_get(it->second, data.data(), 0, data.size() * sizeof(float));
            if (!reuse_dense_encoder::finite(data))
                return false;
        }
        Model result;
        result.encoder.push_back(dense_layer(values, "dense_encoder.dense_conv_1", 2, 1, 1));
        auto encoder_dense = dense_stack(values, "dense_encoder.dense_block");
        for (auto& layer : encoder_dense)
            result.encoder.push_back(std::move(layer));
        result.encoder.push_back(dense_layer(values, "dense_encoder.dense_conv_2", 64, 1, 3));
        for (int i = 0; i < 30; ++i) {
            const std::string prefix = "TSMamba." + std::to_string(i);
            result.blocks.push_back(
                {bidirectional(values, prefix + ".time_mamba"), bidirectional(values, prefix + ".freq_mamba")});
        }
        result.magnitude = decoder(values, "mask_decoder", false);
        result.phase = decoder(values, "phase_decoder", true);
        for (const auto& entry : values)
            if (!entry.second.empty())
                return false; // every tensor must be consumed exactly once
        output = std::move(result);
        return true;
    } catch (...) {
        return false;
    }
}
} // namespace reuse_model_loader
