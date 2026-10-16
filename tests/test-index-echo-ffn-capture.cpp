#include "core/index_echo_ffn_capture.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

#ifdef CRISPASR_INDEX_ECHO_LITERAL_FFN
#include CRISPASR_INDEX_ECHO_LITERAL_FFN
#endif

using Stages = std::map<std::string, std::vector<float>>;
static void check(bool ok, int line) {
    if (!ok)
        throw std::runtime_error("Echo FFN capture control failed at line " + std::to_string(line));
}
#define require(value) check((value), __LINE__)

struct Observer {
    bool enabled;
    Stages values;
};
static bool observe(ggml_tensor* tensor, bool ask, void* opaque) {
    auto& state = *static_cast<Observer*>(opaque);
    return state.enabled && core_index_echo::capture_ffn(tensor, ask, state.values);
}
int main() {
    try {
        const ggml_init_params params{1024 * 1024, nullptr, true};
        auto* weights = ggml_init(params);
        auto* graph_ctx = ggml_init(params);
        require(weights && graph_ctx);
        auto* up = ggml_new_tensor_2d(weights, GGML_TYPE_F32, 3, 4);
        auto* gate = ggml_new_tensor_2d(weights, GGML_TYPE_F32, 3, 4);
        auto* down = ggml_new_tensor_2d(weights, GGML_TYPE_F32, 4, 2);
        auto* cpu = ggml_backend_cpu_init();
        auto* buffer = ggml_backend_alloc_ctx_tensors(weights, cpu);
        require(cpu && buffer);
        const float matrix[] = {1, 0, -1, 2, .5f, 1, -1, 2, 1, .5f, -1, 2};
        ggml_backend_tensor_set(up, matrix, 0, sizeof(matrix));
        ggml_backend_tensor_set(gate, matrix, 0, sizeof(matrix));
        ggml_backend_tensor_set(down, matrix, 0, 8 * sizeof(float));
        auto* x = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, 3, 3);
        ggml_set_input(x);
#ifdef CRISPASR_INDEX_ECHO_LITERAL_FFN
        auto* out = crispasr_literal_ffn(graph_ctx, up, gate, down, x, 4);
        auto* u = ggml_get_tensor(graph_ctx, "ffn_up-4");
        require(u);
#else
        auto* u = ggml_mul_mat(graph_ctx, up, x);
        ggml_set_name(u, "ffn_up-4");
        auto* g = ggml_mul_mat(graph_ctx, gate, x);
        ggml_set_name(g, "ffn_gate-4");
        auto* product = ggml_swiglu_split(graph_ctx, g, u);
        ggml_set_name(product, "ffn_swiglu-4");
        auto* out = ggml_mul_mat(graph_ctx, down, product);
        ggml_set_name(out, "ffn_out-4");
#endif
        ggml_set_output(out);
        auto* graph = ggml_new_graph(graph_ctx);
        ggml_build_forward_expand(graph, out);
        auto* sched = ggml_backend_sched_new(&cpu, nullptr, 1, 128, false, true);
        require(sched && ggml_backend_sched_alloc_graph(sched, graph));
        const float data[] = {9, 8, 7, 4, 5, 6, 1, 2, 3};
        ggml_backend_tensor_set(x, data, 0, sizeof(data));
        Observer state{false, {}};
        ggml_backend_sched_set_eval_callback(sched, observe, &state);
        require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        std::vector<float> off(6), on(6), off2(6);
        ggml_backend_tensor_get(out, off.data(), 0, off.size() * sizeof(float));
        require(state.values.empty());
        state.enabled = true;
        ggml_backend_tensor_set(x, data, 0, sizeof(data));
        require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, on.data(), 0, on.size() * sizeof(float));
        if (off != on || state.values.size() != 6) {
            std::fprintf(stderr, "OFF/ON equality=%d captured=%zu\n", off == on, state.values.size());
            for (size_t i = 0; i < off.size(); ++i)
                std::fprintf(stderr, "output[%zu] off=%a on=%a\n", i, off[i], on[i]);
            for (const auto& row : state.values)
                std::fprintf(stderr, "%s: %zu values\n", row.first.c_str(), row.second.size());
        }
        require(off == on && state.values.size() == 6);
        require(state.values.at("llm_ffn_4_mlp_in") == std::vector<float>({1, 2, 3}));
        require(state.values.at("llm_ffn_4_up_out") == std::vector<float>({-2, 6, 6, 4.5f}));
        require(state.values.at("llm_ffn_4_gate_out") == state.values.at("llm_ffn_4_up_out"));
        require(state.values.at("llm_ffn_4_down_out") == std::vector<float>(on.end() - 2, on.end()));
        require(state.values.at("llm_ffn_4_mlp_out") == state.values.at("llm_ffn_4_down_out"));
        const auto saved = state.values;
        const auto& uv = state.values.at("llm_ffn_4_up_out");
        const auto& pv = state.values.at("llm_ffn_4_down_in");
        for (size_t i = 0; i < uv.size(); ++i)
            require(std::abs(pv[i] - uv[i] * uv[i] / (1 + std::exp(-uv[i]))) < 1e-5f);
        state.enabled = false;
        ggml_backend_tensor_set(x, data, 0, sizeof(data));
        require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, off2.data(), 0, off2.size() * sizeof(float));
        require(off == off2 && state.values == saved);
        const float zeros[9] = {};
        ggml_backend_tensor_set(x, zeros, 0, sizeof(zeros));
        require(state.values == saved); // Captures own their storage.
        // Read a padded row using a real backend tensor/view.
        auto* view = ggml_view_2d(weights, up, 2, 2, 3 * sizeof(float), sizeof(float));
        std::vector<float> padded;
        require(core_index_echo::ffn_last_row(view, padded));
        require(padded == std::vector<float>({.5f, 1}));
        auto wrong = *view;
        wrong.type = GGML_TYPE_F16;
        require(!core_index_echo::ffn_last_row(&wrong, padded));
        wrong = *view;
        wrong.ne[2] = 2;
        require(!core_index_echo::ffn_last_row(&wrong, padded));
        wrong = *view;
        wrong.nb[0] = 8;
        require(!core_index_echo::ffn_last_row(&wrong, padded));
        require(!core_index_echo::ffn_last_row(nullptr, padded));
        auto bad_up = *u;
        auto bad_input = *x;
        bad_input.type = GGML_TYPE_F16;
        bad_up.src[1] = &bad_input;
        require(!core_index_echo::capture_ffn(&bad_up, false, state.values));
        require(state.values == saved); // Failed observation publishes nothing.

        for (const char* name : {"ffn_up-", "ffn_up--1", "ffn_up-2147483648", "ffn_up-4junk", "post_ffn-4"}) {
            std::string key;
            bool input, output;
            require(!core_index_echo::ffn_capture_name(name, key, input, output));
        }
        require(!core_index_echo::capture_ffn(x, true, state.values));
        ggml_backend_sched_free(sched);
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(cpu);
        ggml_free(graph_ctx);
        ggml_free(weights);
        std::puts("ECHO_FFN_CAPTURE_PASS: actual scheduler OFF/ON/OFF, six boundaries, ownership, stride and rejection "
                  "controls");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
