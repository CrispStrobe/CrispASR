#include <catch2/catch_test_macros.hpp>
#include "core/index_echo_connector.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <memory>

TEST_CASE("Index-Echo 9B projection changes width without residual scaling", "[unit][index-echo]") {
    const ggml_init_params params = {1024 * 1024, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    REQUIRE(ctx);
    auto* input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 2, 2);
    auto* weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 2, 3);
    const float x[] = {4, -1, 2, 3}, w[] = {1, 2, -3, .5f, 0, -2};
    memcpy(input->data, x, sizeof(x));
    memcpy(weight->data, w, sizeof(w));
    core_index_echo::ConnectorWeights weights = {{"connector.proj.weight", weight}};
    REQUIRE(core_index_echo::connector_matches(weights, true, 2, 3));
    CHECK_FALSE(core_index_echo::connector_matches(weights, true, 3, 2));
    CHECK_FALSE(core_index_echo::connector_matches(weights, false, 2, 3));
    auto* output = core_index_echo::connector_graph(ctx.get(), input, weights, true);
    auto* graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    REQUIRE(ggml_graph_compute_with_ctx(ctx.get(), graph, 1) == GGML_STATUS_SUCCESS);
    REQUIRE(output->ne[0] == 3);
    REQUIRE(output->ne[1] == 2);
    const float expected[] = {2, -12.5f, 2, 8, -4.5f, -6};
    for (int i = 0; i < 6; ++i)
        CHECK(std::abs(((float*)output->data)[i] - expected[i]) < 1e-6f);
}

TEST_CASE("Index-Echo 2B retains residual GELU beta and exponential alpha", "[unit][index-echo]") {
    const ggml_init_params params = {1024 * 1024, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    REQUIRE(ctx);
    auto* input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 2, 2);
    const float x[] = {-1, 0, 1, 2};
    memcpy(input->data, x, sizeof(x));
    core_index_echo::ConnectorWeights weights;
    const float identity[] = {1, 0, 0, 1};
    for (const char* name : {"connector.w1.weight", "connector.w2.weight"}) {
        auto* tensor = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 2, 2);
        memcpy(tensor->data, identity, sizeof(identity));
        weights[name] = tensor;
    }
    weights["connector.beta"] = ggml_new_f32(ctx.get(), .5f);
    weights["connector.log_alpha"] = ggml_new_f32(ctx.get(), std::log(2.0f));
    REQUIRE(core_index_echo::connector_matches(weights, false, 2, 2));
    CHECK_FALSE(core_index_echo::connector_matches(weights, false, 2, 3));
    auto* output = core_index_echo::connector_graph(ctx.get(), input, weights, false);
    auto* graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    REQUIRE(ggml_graph_compute_with_ctx(ctx.get(), graph, 1) == GGML_STATUS_SUCCESS);
    // Independently tabulated exact GELU values, with alpha=2 and beta=1/2.
    const float expected[] = {-2.15865525f, 0, 2.84134475f, 5.95449974f};
    for (int i = 0; i < 4; ++i)
        CHECK(std::abs(((float*)output->data)[i] - expected[i]) < 1e-5f);
}
