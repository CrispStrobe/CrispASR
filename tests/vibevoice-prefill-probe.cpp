// Tiny-only probe of genuine truncated production cached Qwen2 prefill graphs.
#include "../src/vibevoice.cpp"
#include <fstream>

int main(int argc, char** argv) {
    if (argc != 4)
        return 2;
    auto params = vibevoice_context_default_params();
    params.use_gpu = false;
    params.n_threads = 1;
    auto* ctx = vibevoice_init_from_file(argv[1], params);
    if (!ctx)
        return 3;
    auto& hp = ctx->model.hp;
    if (hp.d_lm != 32 || hp.n_lm_layers != 2)
        return 4; // Never load/execute a real model through this test helper.
    std::vector<float> input(39 * hp.d_lm);
    std::ifstream stream(argv[2], std::ios::binary);
    stream.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(float));
    if (!stream || stream.peek() != EOF)
        return 5;
    ggml_init_params kp = {2 * ggml_tensor_overhead(), nullptr, true};
    ctx->kv_ctx = ggml_init(kp);
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hp.head_dim, 39, hp.n_kv_heads, 2);
    ctx->kv_v = ggml_dup_tensor(ctx->kv_ctx, ctx->kv_k);
    ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, ctx->backend);
    if (!ctx->kv_buf)
        return 6;
    std::ofstream output(argv[3], std::ios::binary);
    for (int depth = 1; depth <= 2; ++depth) {
        hp.n_lm_layers = depth;
        ggml_backend_buffer_clear(ctx->kv_buf, 0);
        std::vector<float> logits, hidden;
        if (!run_base_tts_decoder(ctx, input.data(), 39, 0, ctx->kv_k, ctx->kv_v, logits, hidden))
            return 7;
        output.write(reinterpret_cast<const char*>(hidden.data()), hidden.size() * sizeof(float));
        output.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
    }
    const bool ok = output.good();
    vibevoice_free(ctx);
    return ok ? 0 : 8;
}
