#include "../src/core/reuse_reference_network.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

static reuse_dense_encoder::Layer dense(size_t inputs, size_t kt, size_t kf) {
    reuse_dense_encoder::Layer layer;
    layer.in_channels = inputs;
    layer.out_channels = 1;
    layer.kernel_time = kt;
    layer.kernel_freq = kf;
    layer.weights.assign(inputs * kt * kf, 0);
    layer.bias = {0};
    layer.norm_weight = {1};
    layer.norm_bias = {0};
    layer.slope = {0.25f};
    return layer;
}
static reuse_decoder::UpLayer up(size_t scale) {
    reuse_decoder::UpLayer layer;
    layer.channels = 1;
    layer.scale = scale;
    layer.weights.assign(scale * 3, 0);
    layer.bias.assign(scale, 0);
    layer.norm_weight = {1};
    layer.norm_bias = {0};
    layer.slope = {0.25f};
    return layer;
}
static reuse_mamba::Weights mamba() {
    reuse_mamba::Weights w;
    w.model = w.inner = w.rank = w.states = w.kernel = 1;
    w.in_proj = {0, 0};
    w.conv = {0};
    w.conv_bias = {0};
    w.x_proj = {0, 0, 0};
    w.dt_proj = {0};
    w.dt_bias = {0};
    w.a_log = {0};
    w.d = {0};
    w.out_proj = {0};
    return w;
}
static reuse_bimamba::Weights bidirectional() {
    reuse_bimamba::Weights w;
    w.forward = mamba();
    w.backward = mamba();
    w.projection = {0, 0};
    w.projection_bias = {0};
    w.norm_weight = {1};
    w.norm_bias = {0};
    return w;
}
static reuse_model_loader::Decoder decoder(bool phase) {
    reuse_model_loader::Decoder result;
    for (size_t i = 1; i <= 4; ++i)
        result.dense.push_back(dense(i, 3, 3));
    result.frequency = up(2);
    result.time = up(4);
    result.first_head = {0};
    if (phase)
        result.second_head = {0};
    return result;
}
int main() {
    reuse_model_loader::Model model;
    model.encoder.push_back(dense(2, 1, 1));
    for (size_t i = 1; i <= 4; ++i)
        model.encoder.push_back(dense(i, 3, 3));
    model.encoder.push_back(dense(1, 1, 3));
    model.blocks.assign(30, {bidirectional(), bidirectional()});
    model.magnitude = decoder(false);
    model.phase = decoder(true);
    std::vector<float> input(200), output{42};
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = 0.1f * std::sin(static_cast<float>(i) * 0.1f);
    reuse_reference_network::Trace trace;
    if (!reuse_reference_network::run(model, input, 8000, output, &trace) || output.size() != input.size() ||
        trace.encoder.channels != 1 || trace.block0.values.empty() || trace.block14.values.empty() ||
        trace.block29.values.empty())
        return 1;
    for (float sample : output)
        if (sample != 0)
            return 2;
    model.magnitude.first_bias = 0.5f;
    model.phase.second_bias = 1.0f;
    if (!reuse_reference_network::run(model, input, 8000, output) ||
        std::none_of(output.begin(), output.end(), [](float sample) { return std::abs(sample) > 1e-6f; }))
        return 5;
    output = {42};
    model.blocks.pop_back();
    if (reuse_reference_network::run(model, input, 8000, output) || output != std::vector<float>{42})
        return 3;
    model.blocks.push_back({bidirectional(), bidirectional()});
    model.phase.second_head.clear();
    trace.encoder.values = {17};
    if (reuse_reference_network::run(model, input, 8000, output, &trace) || output != std::vector<float>{42} ||
        trace.encoder.values != std::vector<float>{17})
        return 4;
    std::puts("RE-USE reference network synthetic full-path controls passed");
    return 0;
}
