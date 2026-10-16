#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <climits>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>

namespace core_index_echo {

// These are existing Qwen3.5 graph names. No node, operation or weight changes.
inline bool ffn_capture_name(const char* name, std::string& key, bool& input, bool& output) {
    input = output = false;
    struct Stage {
        const char* prefix;
        const char* suffix;
    };
    for (const Stage stage :
         {Stage{"ffn_up-", "up_out"}, Stage{"ffn_gate-", "gate_out"}, Stage{"ffn_swiglu-", "down_in"},
          Stage{"ffn_gate_par-", "down_in"}, Stage{"ffn_out-", "down_out"}}) {
        const size_t n = std::strlen(stage.prefix);
        if (std::strncmp(name, stage.prefix, n) != 0)
            continue;
        const char* p = name + n;
        if (!*p)
            return false;
        unsigned layer = 0;
        for (; *p; ++p) {
            if (*p < '0' || *p > '9' || layer > (unsigned(INT_MAX) - unsigned(*p - '0')) / 10)
                return false;
            layer = layer * 10 + unsigned(*p - '0');
        }
        key = "llm_ffn_" + std::to_string(layer) + "_" + stage.suffix;
        input = std::strcmp(stage.suffix, "up_out") == 0;
        output = std::strcmp(stage.suffix, "down_out") == 0;
        return true;
    }
    return false;
}

// Read the final token before scheduler scratch reuse, honoring row padding.
// Admission rejects unsupported layouts instead of publishing partial arrays.
inline bool ffn_last_row(const ggml_tensor* tensor, std::vector<float>& values) {
    if (!tensor || tensor->type != GGML_TYPE_F32 || tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] != 1 ||
        tensor->ne[3] != 1 || tensor->nb[0] != sizeof(float) || tensor->nb[1] < size_t(tensor->ne[0]) * sizeof(float))
        return false;
    values.resize(tensor->ne[0]);
    ggml_backend_tensor_get(tensor, values.data(), size_t(tensor->ne[1] - 1) * tensor->nb[1],
                            values.size() * sizeof(float));
    return true;
}

inline bool capture_ffn(ggml_tensor* tensor, bool ask, std::map<std::string, std::vector<float>>& stages) {
    std::string key;
    bool input, output;
    if (!tensor || !ffn_capture_name(tensor->name, key, input, output))
        return false;
    if (ask)
        return true;
    std::vector<float> value, x;
    if (!ffn_last_row(tensor, value) || (input && !ffn_last_row(tensor->src[1], x)))
        return false;
    const std::string prefix = key.substr(0, key.find('_', 8) + 1);
    if (input)
        stages[prefix + "mlp_in"] = std::move(x);
    if (output)
        stages[prefix + "mlp_out"] = value;
    stages[key] = std::move(value);
    return true;
}

} // namespace core_index_echo
