// Standalone parity probe: compile against ggml, without the main backend library.
// Includes the implementation so this exercises the exact base TTS cached graph.
#include "../src/vibevoice.cpp"
#include <fstream>

static std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        return {};
    std::vector<float> x((size_t)f.tellg() / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(x.data()), x.size() * sizeof(float));
    return x;
}

int main(int argc, char** argv) {
    if (argc != 4)
        return 2;
    auto params = vibevoice_context_default_params();
    params.use_gpu = false;
    params.n_threads = 1;
    auto* ctx = vibevoice_init_from_file(argv[1], params);
    if (!ctx)
        return 3;
    const int dim = ctx->model.hp.d_lm;
    auto inputs = read_f32(argv[2]);
    if (inputs.size() != (size_t)dim * 4)
        return 4;
    ggml_init_params kp = {4 * ggml_tensor_overhead(), nullptr, true};
    ctx->kv_ctx = ggml_init(kp);
    auto cache = [&]() {
        return ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, ctx->model.hp.head_dim, 8, ctx->model.hp.n_kv_heads,
                                  ctx->model.hp.n_lm_layers);
    };
    ctx->kv_k = cache();
    ctx->kv_v = cache();
    auto* neg_k = cache();
    auto* neg_v = cache();
    ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, ctx->backend);
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
    std::vector<float> logits, hidden, result;
    // Interleave a deliberately different positive history. Negative output must
    // retain its own speech-start and feedback, even after scheduler reuse.
    for (int i = 0; i < 4; ++i) {
        if (!run_base_tts_decoder(ctx, inputs.data() + (3 - i) * dim, 1, i, ctx->kv_k, ctx->kv_v, logits, hidden))
            return 5;
        if (!run_base_tts_decoder(ctx, inputs.data() + i * dim, 1, i, neg_k, neg_v, logits, hidden))
            return 6;
        result.insert(result.end(), hidden.begin(), hidden.end());
    }
    // Upstream refresh at a later speech-start retains only the initial start KV.
    // Its masked physical prefix is equivalent to this compact two-token history.
    if (!run_base_tts_decoder(ctx, inputs.data(), 1, 1, neg_k, neg_v, logits, hidden))
        return 7;
    result.insert(result.end(), hidden.begin(), hidden.end());
    if (!run_base_tts_decoder(ctx, inputs.data() + dim, 1, 2, neg_k, neg_v, logits, hidden))
        return 8;
    result.insert(result.end(), hidden.begin(), hidden.end());
    std::ofstream out(argv[3], std::ios::binary);
    out.write(reinterpret_cast<const char*>(result.data()), result.size() * sizeof(float));
    const bool ok = out.good();
    vibevoice_free(ctx);
    return ok ? 0 : 9;
}
