// Exercise the production private codec builders with synthetic weights.
#include "../src/vibevoice.cpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>

static void check_stream(int kind, int kernel, int stride, int chunk) {
    auto* backend = core_cpu_backend::init();
    core_cpu_backend::set_n_threads(backend, 1);
    ggml_init_params wp = {1024 * 1024, nullptr, false};
    auto* weights = ggml_init(wp);
    auto* w = kind == 1 ? ggml_new_tensor_3d(weights, GGML_TYPE_F32, kernel, 1, 2)
                        : ggml_new_tensor_3d(weights, GGML_TYPE_F32, kernel, 2, 2);
    ggml_set_name(w, "test.conv.weight");
    for (int i = 0; i < ggml_nelements(w); ++i)
        ((float*)w->data)[i] = 0.03f * (i % 11 - 5);
    auto* bias = ggml_new_tensor_1d(weights, GGML_TYPE_F32, 2);
    ((float*)bias->data)[0] = 0.125f;
    ((float*)bias->data)[1] = -0.25f;
    vibevoice_codec_stream stream;
    std::vector<float> prefix, actual;
    size_t retained = 0;
    for (int step = 0; step < 20; ++step) {
        std::vector<float> input((size_t)chunk * 2);
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = sinf((float)(prefix.size() + i) * 0.7f);
        prefix.insert(prefix.end(), input.begin(), input.end());
        auto run = [&](const std::vector<float>& values, vibevoice_codec_stream* state) {
            ggml_init_params gp = {8 * 1024 * 1024, nullptr, true};
            auto* g = ggml_init(gp);
            auto* gf = ggml_new_graph_custom(g, 1024, false);
            auto* in = ggml_new_tensor_2d(g, GGML_TYPE_F32, 2, values.size() / 2);
            ggml_set_input(in);
            auto* out = kind == 0   ? build_causal_conv1d(g, in, w, bias, stride, state)
                        : kind == 1 ? build_causal_dw_conv1d(g, in, w, bias, state)
                                    : build_transposed_conv1d(g, in, w, bias, stride, nullptr, state);
            ggml_set_output(out);
            ggml_build_forward_expand(gf, out);
            if (state)
                state->expand(gf);
            auto* alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            assert(ggml_gallocr_alloc_graph(alloc, gf));
            ggml_backend_tensor_set(in, values.data(), 0, values.size() * sizeof(float));
            if (state)
                state->set_inputs();
            assert(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
            std::vector<float> result(ggml_nelements(out));
            ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
            if (state)
                state->capture();
            ggml_gallocr_free(alloc);
            ggml_free(g);
            return result;
        };
        auto current = run(input, &stream);
        actual.insert(actual.end(), current.begin(), current.end());
        auto expected = run(prefix, nullptr);
        assert(actual.size() == expected.size());
        for (size_t i = 0; i < actual.size(); ++i)
            assert(fabsf(actual[i] - expected[i]) < 2e-6f);
        size_t state_size = 0;
        for (auto& entry : stream.edges)
            state_size += entry.second.values.size();
        assert(state_size <= (size_t)2 * kernel);
        if (step >= kernel)
            assert(state_size == retained);
        retained = state_size;
    }
    ggml_free(weights);
    ggml_backend_free(backend);
}

int main() {
    check_stream(0, 7, 1, 1); // stem/head, history longer than first chunk
    check_stream(0, 4, 2, 4); // stride-aligned downsampling
    check_stream(1, 7, 1, 1); // depthwise ConvNeXt mixer
    check_stream(2, 8, 4, 1); // causal transposed upsampling, including bias
    check_stream(2, 7, 2, 1); // history spans multiple preceding input frames
    check_stream(0, 1, 1, 1); // no history
    std::cout << "VibeVoice codec stream: 120 prefix comparisons and bounded-state checks passed\n";
}
