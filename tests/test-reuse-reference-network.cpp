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
    // Captured from torch.hann_window(320), including samples where changing
    // the order of the float32 multiply/divide moves the value by 2 ULPs.
    const auto window = reuse_frontend::periodic_hann(320);
    if (window.size() != 320 || std::abs(window[211] - 0.769569159f) > 1e-7f ||
        std::abs(window[218] - 0.709329724f) > 1e-7f)
        return 9;
    std::vector<float> negative_nyquist(200);
    for (size_t i = 0; i < negative_nyquist.size(); ++i)
        negative_nyquist[i] = i % 2 ? 1.0f : -1.0f;
    reuse_frontend::Spectrum endpoint;
    if (!reuse_frontend::analyze(negative_nyquist, 8000, endpoint))
        return 10;
    for (size_t t = 0; t < endpoint.frames; ++t)
        if (std::abs(endpoint.phase[t * endpoint.bins + endpoint.bins - 1] - std::acos(-1.0f)) > 1e-6f)
            return 11;
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
        trace.block29.values.empty() || trace.encoder_layers.size() != 6 || trace.block_time.size() != 30 ||
        trace.blocks.size() != 30 || trace.phase.size() != trace.magnitude.size() ||
        trace.amplitude.size() != trace.phase.size() || trace.output_phase.size() != trace.phase.size())
        return 1;
    for (float sample : output)
        if (sample != 0)
            return 2;
    const auto saved_phase = trace.phase;
    std::vector<float> conditioned;
    conditioned = {42};
    if (reuse_reference_network::run(model, input, 8000, conditioned, nullptr, nullptr, nullptr) ||
        conditioned != std::vector<float>{42})
        return 12;
    if (!reuse_reference_network::run(model, input, 8000, conditioned, nullptr, &saved_phase) || conditioned != output)
        return 6;
    auto invalid_phase = saved_phase;
    invalid_phase[0] = std::numeric_limits<float>::quiet_NaN();
    conditioned = {42};
    if (reuse_reference_network::run(model, input, 8000, conditioned, nullptr, &invalid_phase) ||
        conditioned != std::vector<float>{42})
        return 7;
    invalid_phase.pop_back();
    if (reuse_reference_network::run(model, input, 8000, conditioned, nullptr, &invalid_phase))
        return 8;
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
