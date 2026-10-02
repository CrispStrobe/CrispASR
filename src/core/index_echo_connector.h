#pragma once
#include "ggml.h"
#include "gguf_loader.h"
#include <string>

namespace core_index_echo {
using ConnectorWeights = core_gguf::tensor_map;

inline bool connector_matches(const ConnectorWeights& weights, bool projection, int audio_dim, int decoder_dim) {
    auto matrix = [&](const char* name, int in, int out) {
        auto found = weights.find(name);
        return found != weights.end() && found->second->ne[0] == in && found->second->ne[1] == out &&
               found->second->ne[2] == 1 && found->second->ne[3] == 1;
    };
    if (audio_dim <= 0 || decoder_dim <= 0)
        return false;
    if (projection)
        return weights.size() == 1 && matrix("connector.proj.weight", audio_dim, decoder_dim);
    if (weights.size() != 4 || audio_dim != decoder_dim || !matrix("connector.w1.weight", audio_dim, audio_dim) ||
        !matrix("connector.w2.weight", audio_dim, audio_dim))
        return false;
    for (const char* name : {"connector.log_alpha", "connector.beta"}) {
        auto found = weights.find(name);
        if (found == weights.end() || ggml_nelements(found->second) != 1)
            return false;
    }
    return true;
}

inline ggml_tensor* connector_graph(ggml_context* graph, ggml_tensor* input, const ConnectorWeights& weights,
                                    bool projection) {
    if (projection)
        return ggml_mul_mat(graph, weights.at("connector.proj.weight"), input);
    auto* hidden = ggml_mul_mat(graph, weights.at("connector.w1.weight"), input);
    hidden = ggml_gelu_erf(graph, hidden);
    hidden = ggml_mul_mat(graph, weights.at("connector.w2.weight"), hidden);
    hidden = ggml_mul(graph, hidden, weights.at("connector.beta"));
    hidden = ggml_add(graph, input, hidden);
    return ggml_mul(graph, hidden, ggml_exp(graph, weights.at("connector.log_alpha")));
}
} // namespace core_index_echo
