#include "core/vibevoice_semantic_weights.h"
#include "ggml-cpu.h"
#include <catch2/catch_test_macros.hpp>
#include <vector>

TEST_CASE("VibeVoice semantic recovery preserves dequantized rows and unrelated weights", "[unit][vibevoice]") {
    auto* backend = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr);
    const ggml_init_params params = {8 * ggml_tensor_overhead(), nullptr, true};
    auto* ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    for (const auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0}) {
        // More than 1 MiB of decoded rows exercises the final partial chunk. Both K-quant
        // and legacy Q4/Q8 packs must work; small early blocks use Q4_0.
        auto* src = ggml_new_tensor_2d(ctx, type, 256, 1031);
        auto* original_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(original_buf != nullptr);
        std::vector<float> values(256 * 1031);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = (float)((int)(i % 31) - 15) * 0.03125f + (i >= 256 ? 0.7f : 0.0f);
        std::vector<uint8_t> raw(ggml_nbytes(src));
        REQUIRE(ggml_quantize_chunk(type, values.data(), raw.data(), 0, 1031, 256, nullptr) == raw.size());
        ggml_backend_tensor_set(src, raw.data(), 0, raw.size());
        std::vector<float> expected(values.size());
        ggml_get_type_traits(type)->to_float(raw.data(), expected.data(), expected.size());
        std::vector<ggml_fp16_t> expected_half(expected.size());
        ggml_fp32_to_fp16_row(expected.data(), expected_half.data(), expected.size());

        std::map<std::string, ggml_tensor*> tensors = {{"st_enc.s.0.0.ffn.up.weight", src},
                                                       {"st_enc.s.0.0.ffn.down.weight", src},
                                                       {"at_enc.s.0.0.ffn.up.weight", src},
                                                       {"st_enc.ds.0.0.conv.weight", src},
                                                       {"lm.layers.0.ffn.up.weight", src}};
        vibevoice_semantic_weights::Copies copies;
        REQUIRE(vibevoice_semantic_weights::recover(tensors, backend, copies));
        CHECK(tensors.at("at_enc.s.0.0.ffn.up.weight") == src);
        CHECK(tensors.at("st_enc.ds.0.0.conv.weight") == src);
        CHECK(tensors.at("lm.layers.0.ffn.up.weight") == src);
        // Copies have independent lifetime, including when a model tensor is
        // aliased by two names. Releasing source storage cannot invalidate them.
        ggml_backend_buffer_free(original_buf);
        for (const auto* name : {"st_enc.s.0.0.ffn.up.weight", "st_enc.s.0.0.ffn.down.weight"}) {
            auto* dst = tensors.at(name);
            REQUIRE(dst != src);
            REQUIRE(dst->type == GGML_TYPE_F16);
            std::vector<ggml_fp16_t> actual(expected.size());
            ggml_backend_tensor_get(dst, actual.data(), 0, actual.size() * sizeof(ggml_fp16_t));
            CHECK(actual == expected_half);
        }
        copies.clear();
        copies.clear(); // destruction path is safe after an explicit cleanup
        ggml_free(ctx);
        ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
    }
    ggml_free(ctx);
    ggml_backend_free(backend);
}

TEST_CASE("VibeVoice floating-point semantic weights need no recovery allocation", "[unit][vibevoice]") {
    const ggml_init_params params = {2 * ggml_tensor_overhead(), nullptr, true};
    auto* ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    auto* src = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 2);
    std::map<std::string, ggml_tensor*> tensors = {{"st_enc.s.0.0.ffn.up.weight", src}};
    vibevoice_semantic_weights::Copies copies;
    REQUIRE(vibevoice_semantic_weights::recover(tensors, nullptr, copies));
    CHECK(tensors.begin()->second == src);
    CHECK(copies.ctx == nullptr);
    CHECK(copies.buf == nullptr);
    ggml_free(ctx);
}
