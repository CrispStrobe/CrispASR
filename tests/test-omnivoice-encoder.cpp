#include "core/ggml_cpu_backend.h"
#include "../src/omnivoice.cpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "ggml-backend-impl.h"
#include <cassert>
#include <iostream>
static void gate(const char* suffix, bool on) {
    auto key = std::string("CRISPASR_OMNIVOICE_ENCODER_") + suffix;
#ifdef _WIN32
    _putenv_s(key.c_str(), on ? "1" : "0");
#else
    setenv(key.c_str(), on ? "1" : "0", 1);
#endif
}
static void fill(ggml_tensor* t, int seed) {
    std::vector<float> v(ggml_nelements(t));
    for (size_t i = 0; i < v.size(); i++)
        v[i] = .17f * std::sin((float)(i + seed) * .37f);
    if (t->type == GGML_TYPE_F32)
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
    else {
        std::vector<ggml_fp16_t> h(v.size());
        ggml_fp32_to_fp16_row(v.data(), h.data(), h.size());
        ggml_backend_tensor_set(t, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
    }
}
static void unset_gate(const char* suffix) {
    const auto key = std::string("CRISPASR_OMNIVOICE_ENCODER_") + suffix;
#ifdef _WIN32
    _putenv_s(key.c_str(), "");
#else
    unsetenv(key.c_str());
#endif
}
int main() {
    unset_gate("CACHE");
    assert(encoder_cache_enabled());
    gate("CACHE", false);
    assert(!encoder_cache_enabled());
    ov_higgs_tokenizer policy;
    policy.backend = core_cpu_backend::init();
    unset_gate("FASTCONV");
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    assert(encoder_fastconv_enabled(policy));
#else
    assert(!encoder_fastconv_enabled(policy));
#endif
    gate("FASTCONV", false);
    assert(!encoder_fastconv_enabled(policy, true));
    gate("FASTCONV", true);
    assert(encoder_fastconv_enabled(policy));
    // An all-F32/no-conversion model still activates the K=1 graph rewrite.
    assert(prepare_encoder_fastconv(policy));
    assert(policy.enc_fc.enabled && !policy.enc_fc.buf);
    tok_release_backend(policy);
    // Default policy must follow the actual backend/profile and explicit overrides.
    {
        auto* cpu = core_cpu_backend::init();
        ggml_backend mock = *cpu;
        mock.iface.get_name = [](ggml_backend_t) { return "MTL0"; };
        omnivoice_context c;
        c.use_gpu = true;
        c.backend = &mock;
        c.tokenizer.backend = &mock;
        c.tokenizer.loaded = true;
        ggml_init_params ip = {4 * ggml_tensor_overhead(), nullptr, true};
        c.tokenizer.ctx_w = ggml_init(ip);
        c.tokenizer.fc_w = c.tokenizer.sem_featproj_w = ggml_new_tensor_1d(c.tokenizer.ctx_w, GGML_TYPE_F16, 32);
        auto* extra = ggml_new_tensor_1d(c.tokenizer.ctx_w, GGML_TYPE_F16, 32);
        for (auto* name : {"HYBRID", "GRAPH", "FUSED", "SEMANTIC_F32"})
            unset_gate(name);
#if defined(__APPLE__) && TARGET_OS_OSX && (defined(__aarch64__) || defined(__arm64__))
        assert(encoder_hybrid_enabled(&c) && encoder_gpu_defaults(&c));
        assert(encoder_precision_enabled(c.tokenizer));
#else
        assert(!encoder_hybrid_enabled(&c) && !encoder_gpu_defaults(&c));
        assert(!encoder_precision_enabled(c.tokenizer));
#endif
        c.use_gpu = false;
        assert(!encoder_hybrid_enabled(&c));
        c.use_gpu = true;
        gate("HYBRID", false);
        assert(!encoder_hybrid_enabled(&c) && !encoder_gpu_defaults(&c));
        gate("SEMANTIC_F32", false);
        assert(!encoder_precision_enabled(c.tokenizer));
        gate("HYBRID", true);
        assert(encoder_hybrid_enabled(&c));
        c.encoder_gpu_failed = true;
        assert(!encoder_gpu_defaults(&c));
        c.encoder_gpu_failed = false;
        unset_gate("HYBRID");
        extra->type = GGML_TYPE_Q8_0;
        assert(!encoder_hybrid_enabled(&c));
        extra->type = GGML_TYPE_F16;
        c.tokenizer.hidden_size = 32;
        assert(!encoder_hybrid_enabled(&c));
        c.tokenizer.hidden_size = 1024;
        mock.iface.get_name = [](ggml_backend_t) { return "CUDA0"; };
        assert(!encoder_hybrid_enabled(&c) && !encoder_precision_enabled(c.tokenizer));
        for (auto* name : {"HYBRID", "GRAPH", "FUSED", "SEMANTIC_F32"})
            gate(name, false);
        ggml_free(c.tokenizer.ctx_w);
        c.tokenizer.ctx_w = nullptr;
        ggml_backend_free(cpu);
    }
    assert(tok_is_semantic_weight("sem.blk.0.attn_q.weight"));
    assert(tok_is_semantic_weight("quantizer.quantizers.0.project_in.weight"));
    assert(tok_is_semantic_weight("quantizer.quantizers.0.codebook.embed"));
    assert(!tok_is_semantic_weight("quantizer.quantizers.0.codebook.embed_avg"));
    assert(!tok_is_semantic_weight("acoustic_encoder.conv1.weight"));
    assert(!tok_is_semantic_weight("fc2.weight"));
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        ov_higgs_tokenizer t;
        t.backend = core_cpu_backend::init();
        core_cpu_backend::set_n_threads(t.backend, 1);
        t.hidden_size = 32;
        t.codebook_dim = 8;
        t.codebook_size = 16;
        t.n_quantizers = 3;
        t.quantizers.resize(3);
        ggml_init_params ip = {64 * ggml_tensor_overhead(), nullptr, true};
        t.ctx_w = ggml_init(ip);
        auto mat = [&](int a, int b) { return ggml_new_tensor_2d(t.ctx_w, type, a, b); };
        auto bias = [&](int n) { return ggml_new_tensor_1d(t.ctx_w, type, n); };
        t.fc_w = mat(32, 32);
        t.fc_b = bias(32);
        t.enc_conv1_w = ggml_new_tensor_3d(t.ctx_w, GGML_TYPE_F16, 3, 2, 2);
        for (auto& q : t.quantizers) {
            q.codebook = mat(8, 16);
            q.proj_in_w = mat(32, 8);
            q.proj_in_b = bias(8);
            q.proj_out_w = mat(8, 32);
            q.proj_out_b = bias(32);
        }
        t.enc_blocks.resize(1);
        auto* pointwise = t.enc_blocks[0].res[0].conv2_w = ggml_new_tensor_3d(t.ctx_w, GGML_TYPE_F16, 1, 2, 2);
        t.buf_w = ggml_backend_alloc_ctx_tensors(t.ctx_w, t.backend);
        int seed = 1;
        for (auto* w = ggml_get_first_tensor(t.ctx_w); w; w = ggml_get_next_tensor(t.ctx_w, w))
            fill(w, seed += 19);
        // Exercise the actual failed allocator before either cache publishes slots.
        auto* cpu = t.backend;
        ggml_backend_buffer_type failing = *core_cpu_backend::buffer_type();
        failing.iface.alloc_buffer = [](ggml_backend_buffer_type_t, size_t) -> ggml_backend_buffer_t {
            return nullptr;
        };
        ggml_backend_device dev = *ggml_backend_get_device(cpu);
        dev.context = &failing;
        dev.iface.get_buffer_type = [](ggml_backend_dev_t d) {
            return static_cast<ggml_backend_buffer_type_t>(d->context);
        };
        ggml_backend backend = *cpu;
        backend.device = &dev;
        t.backend = &backend;
        assert(!prepare_encoder_fastconv(t));
        assert(!t.enc_fc.buf && !t.enc_fc.ctx && t.enc_fc.f32.empty());
        assert(!prepare_encoder_projection(t));
        assert(!t.enc_proj.buf && !t.enc_proj.ctx && t.enc_graph_norms.empty());
        t.backend = cpu;
        gate("CACHE", false);
        std::vector<float> scratch;
        auto original = encoder_values(t, t.fc_w, scratch);
        gate("CACHE", true);
        auto cached = encoder_values(t, t.fc_w, scratch);
        assert(cached == original);
        assert(prepare_encoder_fastconv(t));
        auto* retained = t.enc_fc.buf;
        assert(prepare_encoder_fastconv(t) && t.enc_fc.buf == retained);
        std::vector<float> baked(ggml_nelements(pointwise));
        ggml_backend_tensor_get(t.enc_fc.get(pointwise), baked.data(), 0, baked.size() * sizeof(float));
        assert(baked == read_tensor_f32(pointwise));
        assert(t.enc_fc.get(t.enc_conv1_w) == t.enc_conv1_w);
        // Explicit full baking and return to pointwise must rebuild the cache.
#ifdef _WIN32
        _putenv_s("CRISPASR_OMNIVOICE_ENCODER_FASTCONV", "2");
#else
        setenv("CRISPASR_OMNIVOICE_ENCODER_FASTCONV", "2", 1);
#endif
        assert(prepare_encoder_fastconv(t) && t.enc_fc_all);
        assert(t.enc_fc.get(t.enc_conv1_w) != t.enc_conv1_w);
        gate("FASTCONV", true);
        assert(prepare_encoder_fastconv(t) && !t.enc_fc_all);
        assert(t.enc_fc.get(t.enc_conv1_w) == t.enc_conv1_w);
        for (int T : {1, 7, 11}) {
            std::vector<float> x((size_t)32 * T);
            for (size_t i = 0; i < x.size(); i++)
                x[i] = std::sin((float)i * .23f);
            std::vector<float> emb;
            higgs_linear(read_tensor_f32(t.fc_w), read_tensor_f32(t.fc_b), x.data(), T, 32, 32, emb);
            std::vector<int32_t> old;
            gate("CACHE", false);
            assert(higgs_rvq_encode(t, emb, T, old, 1));
            std::vector<int32_t> cache;
            gate("CACHE", true);
            assert(higgs_rvq_encode(t, emb, T, cache, 1));
            assert(old == cache);
            auto graph = higgs_projection_graph(t, x.data(), T);
            assert(graph == old);
            assert(graph.size() == (size_t)3 * T);
            for (auto c : graph)
                assert(c >= 0 && c < 16);
        }
        // A missing optional FC bias matches the original bias-free projection.
        auto* saved_bias = t.fc_b;
        t.fc_b = nullptr;
        std::vector<float> optional_scratch;
        assert(encoder_values(t, nullptr, optional_scratch).empty());
        std::vector<float> bias_free(32 * 3, .3f), embedding;
        higgs_linear(read_tensor_f32(t.fc_w), {}, bias_free.data(), 3, 32, 32, embedding);
        std::vector<int32_t> expected;
        assert(higgs_rvq_encode(t, embedding, 3, expected, 1));
        assert(higgs_projection_graph(t, bias_free.data(), 3) == expected);
        t.fc_b = saved_bias;
        // Equal centroids must retain the first index, including later residual stages.
        for (auto& q : t.quantizers) {
            std::vector<uint8_t> zero(ggml_nbytes(q.codebook), 0);
            ggml_backend_tensor_set(q.codebook, zero.data(), 0, zero.size());
        }
        t.enc_proj.free();
        t.enc_graph_norms.clear();
        t.enc_values.clear();
        t.enc_norms.clear();
        std::vector<float> x(32 * 3, .2f);
        auto ties = higgs_projection_graph(t, x.data(), 3);
        assert(ties.size() == 9);
        for (auto c : ties)
            assert(c == 0);
        tok_release_backend(t);
        assert(!t.enc_fc.buf && !t.enc_proj.buf && t.enc_values.empty() && t.enc_graph_norms.empty());
    }
    std::cout << "PASS encoder F32/F16 scalar/cache/graph parity, ties, transactional allocation, filtered placement "
                 "and cleanup\n";
}
