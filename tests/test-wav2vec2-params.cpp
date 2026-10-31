#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "wav2vec2-ggml.h"
#include "core/cpu_attention.h"

#include <cmath>
#include <vector>

TEST_CASE("wav2vec2: default params and architecture", "[unit][wav2vec2]") {
    wav2vec2_hparams hp;

    REQUIRE(hp.vocab_size == 39);
    REQUIRE(hp.hidden_size == 384);
    REQUIRE(hp.num_hidden_layers == 10);
    REQUIRE(hp.num_attention_heads == 6);
    REQUIRE(hp.intermediate_size == 1536);
    REQUIRE(hp.num_feat_extract_layers == 7);
    REQUIRE(hp.num_conv_pos_embeddings == 128);
    REQUIRE(hp.num_conv_pos_embedding_groups == 16);
    REQUIRE(hp.pad_token_id == 38);
    REQUIRE(hp.do_stable_layer_norm == 1);

    // Compute total CNN downsampling stride: 5 * 2^6 = 320 samples (20ms at 16kHz)
    uint32_t total_stride = 1;
    for (uint32_t i = 0; i < hp.num_feat_extract_layers; ++i) {
        total_stride *= hp.conv_stride[i];
    }
    REQUIRE(total_stride == 320);

    // CNN channels: first 2 layers 256, remaining 5 layers 512
    REQUIRE(hp.conv_dim[0] == 256);
    REQUIRE(hp.conv_dim[1] == 256);
    for (int i = 2; i < 7; ++i) {
        REQUIRE(hp.conv_dim[i] == 512);
    }

    // Kernel sizes: 10, 3, 3, 3, 3, 2, 2
    REQUIRE(hp.conv_kernel[0] == 10);
    REQUIRE(hp.conv_kernel[1] == 3);
    REQUIRE(hp.conv_kernel[2] == 3);
    REQUIRE(hp.conv_kernel[3] == 3);
    REQUIRE(hp.conv_kernel[4] == 3);
    REQUIRE(hp.conv_kernel[5] == 2);
    REQUIRE(hp.conv_kernel[6] == 2);
}

TEST_CASE("wav2vec2: cpu_dot vectorization parity", "[unit][wav2vec2]") {
    const std::vector<int> lengths = {1, 3, 4, 7, 8, 15, 16, 32, 64, 128, 256};

    for (int len : lengths) {
        std::vector<float> a(len);
        std::vector<float> b(len);
        float expected = 0.0f;
        for (int i = 0; i < len; ++i) {
            a[i] = 0.25f * (float)(i % 7 - 3);
            b[i] = 0.5f * (float)(i % 5 - 2);
            expected += a[i] * b[i];
        }

        float dot = crispasr::cpu_attn::cpu_dot(a.data(), b.data(), len);
        REQUIRE(dot == Catch::Approx(expected).epsilon(1e-5f));
    }
}

TEST_CASE("wav2vec2: emission record structure", "[unit][wav2vec2]") {
    wav2vec2_token_prob em{};
    em.id = 12;
    em.text = "hello";
    em.frame_start = 5;
    em.frame_end = 9;
    em.prob = 0.985f;

    REQUIRE(em.id == 12);
    REQUIRE(em.text == "hello");
    REQUIRE(em.frame_start == 5);
    REQUIRE(em.frame_end == 9);
    REQUIRE(em.prob == Catch::Approx(0.985f));
}
