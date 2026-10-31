// Exercise the production prefill and incremental graph builders with tiny
// deterministic weights. Full-prefix recomputation is the independent oracle
// for the state carried between decoding calls; no model download is required.
#include "core/ggml_cpu_backend.h"
#include "../src/arktts.cpp"

#include <catch2/catch_test_macros.hpp>

namespace {

struct TinyArkTTS {
    arktts_context ctx;
    ggml_context* weights = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;

    explicit TinyArkTTS(int kv_heads) {
        ctx.params = arktts_context_default_params();
        ctx.params.n_threads = 1;
        ctx.params.use_gpu = false;
        ctx.backend = core_cpu_backend::init();
        REQUIRE(ctx.backend != nullptr);
        ctx.hp.dim = 16;
        ctx.hp.n_head = 2;
        ctx.hp.n_local_heads = kv_heads;
        ctx.hp.head_dim = 8;
        ctx.hp.n_layer = 3;
        ctx.hp.vocab_size = 32;
        ctx.max_seq_len = 16;
        weights = ggml_init({2 * 1024 * 1024, nullptr, true});
        REQUIRE(weights != nullptr);

        std::mt19937 rng(12345);
        std::vector<std::pair<ggml_tensor*, std::vector<float>>> values;
        auto weight = [&](const std::string& name, int a, int b, bool unit = false) {
            ggml_tensor* t = ggml_new_tensor_2d(weights, GGML_TYPE_F32, a, b);
            ggml_set_name(t, name.c_str());
            ctx.tensors[name] = t;
            std::vector<float> data((size_t)a * b);
            for (float& x : data)
                x = unit ? 1.0f : ((int)(rng() % 2001) - 1000) * 0.0002f;
            values.emplace_back(t, std::move(data));
        };
        const int kv_dim = kv_heads * 8;
        const int qkv_dim = 16 + 2 * kv_dim;
        for (int l = 0; l < 3; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            weight(p + "attention_norm.weight", 16, 1, true);
            weight(p + "attention.wqkv.weight", 16, qkv_dim);
            weight(p + "attention.wqkv.bias", qkv_dim, 1);
            weight(p + "attention.wo.weight", 16, 16);
            weight(p + "ffn_norm.weight", 16, 1, true);
            weight(p + "feed_forward.w1.weight", 16, 32);
            weight(p + "feed_forward.w2.weight", 32, 16);
            weight(p + "feed_forward.w3.weight", 16, 32);
        }
        weight("norm.weight", 16, 1, true);
        weight("embeddings.weight", 16, 32);
        weight_buffer = ggml_backend_alloc_ctx_tensors(weights, ctx.backend);
        REQUIRE(weight_buffer != nullptr);
        for (const auto& value : values)
            ggml_backend_tensor_set(value.first, value.second.data(), 0, value.second.size() * sizeof(float));
        ctx.slow_kv_k.assign(3, std::vector<float>((size_t)ctx.max_seq_len * kv_dim, 0.0f));
        ctx.slow_kv_v = ctx.slow_kv_k;
    }

    ~TinyArkTTS() {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(weights);
        ggml_backend_free(ctx.backend);
    }

    TinyArkTTS(const TinyArkTTS&) = delete;
    TinyArkTTS& operator=(const TinyArkTTS&) = delete;
};

void require_close(const float* actual, const float* expected, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        CAPTURE(i, actual[i], expected[i]);
        REQUIRE(std::isfinite(actual[i]));
        REQUIRE(std::isfinite(expected[i]));
        // Batched matmul and single-token matmul can use different reduction
        // kernels. This tolerance is far below the corrupted-cache discrepancy.
        REQUIRE(std::abs(actual[i] - expected[i]) <= 2e-5f + 2e-5f * std::abs(expected[i]));
    }
}

using OwnedFloats = std::unique_ptr<float, decltype(&std::free)>;

} // namespace

TEST_CASE("ArkTTS cached decoding matches full-prefix recomputation", "[unit][arktts][cache]") {
    for (const int kv_heads : {1, 2}) {
        for (const int prompt_tokens : {1, 3}) {
            DYNAMIC_SECTION("KV heads=" << kv_heads << ", prompt tokens=" << prompt_tokens) {
                TinyArkTTS cached(kv_heads);
                TinyArkTTS full(kv_heads);
                std::mt19937 rng(7123);
                std::vector<float> embeddings((size_t)(prompt_tokens + 4) * 16);
                for (float& x : embeddings)
                    x = ((int)(rng() % 2001) - 1000) * 0.0002f;
                int vocab = 0;
                OwnedFloats prefill(arktts_run_slow_transformer(&cached.ctx, embeddings.data(), prompt_tokens, &vocab),
                                    std::free);
                REQUIRE(prefill != nullptr);
                REQUIRE(vocab == 32);
                for (int step = 0; step < 4; ++step) {
                    const int pos = prompt_tokens + step;
                    CAPTURE(step, pos);
                    OwnedFloats next(
                        arktts_step_slow_transformer(&cached.ctx, embeddings.data() + (size_t)pos * 16, pos, &vocab),
                        std::free);
                    OwnedFloats reference(arktts_run_slow_transformer(&full.ctx, embeddings.data(), pos + 1, &vocab),
                                          std::free);
                    REQUIRE(next != nullptr);
                    REQUIRE(reference != nullptr);
                    REQUIRE(vocab == 32);
                    require_close(next.get(), reference.get(), 32);
                    require_close(cached.ctx.last_slow_normed.data(), full.ctx.last_slow_normed.data(), 16);
                    for (int l = 0; l < 3; ++l) {
                        CAPTURE(l);
                        const size_t count = (size_t)(pos + 1) * kv_heads * 8;
                        require_close(cached.ctx.slow_kv_k[l].data(), full.ctx.slow_kv_k[l].data(), count);
                        require_close(cached.ctx.slow_kv_v[l].data(), full.ctx.slow_kv_v[l].data(), count);
                    }
                }
            }
        }
    }
}
