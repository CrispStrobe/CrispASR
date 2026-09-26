#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>

// Opt-in wall-clock profiler for ggml scheduler graphs. The eval callback asks
// the scheduler to materialise every node, so timings include per-node dispatch
// overhead and are intended for finding relative hotspots, not benchmarking.
//
// CRISPASR_SCHED_PROFILE=1 enables every call site wired through compute(). A
// runtime may also pass its old environment variable as legacy_env.
namespace core_sched_prof {

inline bool env_enabled(const char* name) {
    if (!name || !*name)
        return false;
    const char* value = std::getenv(name);
    return value && *value && *value != '0';
}

inline bool enabled(const char* legacy_env = nullptr) {
    return env_enabled("CRISPASR_SCHED_PROFILE") || env_enabled(legacy_env);
}

struct State {
    std::map<std::string, std::pair<double, int64_t>> aggregate;
    std::chrono::steady_clock::time_point last;
    const char* label = "sched";

    void dump() const {
        std::vector<std::pair<std::string, std::pair<double, int64_t>>> rows(aggregate.begin(), aggregate.end());
        std::sort(rows.begin(), rows.end(),
                  [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
        double total = 0.0;
        for (const auto& row : rows)
            total += row.second.first;
        std::fprintf(stderr, "  sched_profile[%s]: %-58s %9s %6s %6s\n", label, "op", "ms", "%", "n");
        for (const auto& row : rows) {
            const double share = total > 0.0 ? 100.0 * row.second.first / total : 0.0;
            std::fprintf(stderr, "  sched_profile[%s]: %-58s %9.2f %5.1f%% %6lld\n", label, row.first.c_str(),
                         row.second.first, share, (long long)row.second.second);
        }
        std::fprintf(stderr, "  sched_profile[%s]: total %.2f ms\n", label, total);
    }
};

inline bool callback(ggml_tensor* tensor, bool ask, void* user_data) {
    auto* state = static_cast<State*>(user_data);
    if (ask)
        return true;

    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - state->last).count();
    state->last = now;

    char key[256];
    if (tensor->op == GGML_OP_MUL_MAT && tensor->src[0] && tensor->src[1]) {
        std::snprintf(
            key, sizeof(key), "MUL_MAT %s [%lldx%lldx%lld] @ [%lldx%lldx%lld]", ggml_type_name(tensor->src[0]->type),
            (long long)tensor->src[0]->ne[0], (long long)tensor->src[0]->ne[1], (long long)tensor->src[0]->ne[2],
            (long long)tensor->src[1]->ne[0], (long long)tensor->src[1]->ne[1], (long long)tensor->src[1]->ne[2]);
    } else if ((tensor->op == GGML_OP_CPY || tensor->op == GGML_OP_CONT || tensor->op == GGML_OP_DUP) &&
               tensor->src[0]) {
        std::snprintf(key, sizeof(key), "%s %s->%s [%lldx%lldx%lld]", ggml_op_name(tensor->op),
                      ggml_type_name(tensor->src[0]->type), ggml_type_name(tensor->type), (long long)tensor->ne[0],
                      (long long)tensor->ne[1], (long long)tensor->ne[2]);
    } else {
        std::snprintf(key, sizeof(key), "%s", ggml_op_name(tensor->op));
    }
    auto& row = state->aggregate[key];
    row.first += ms;
    row.second++;
    return true;
}

inline ggml_status compute(ggml_backend_sched_t sched, ggml_cgraph* graph, const char* label,
                           const char* legacy_env = nullptr) {
    if (!enabled(legacy_env))
        return ggml_backend_sched_graph_compute(sched, graph);

    State state;
    state.label = label && *label ? label : "sched";
    state.last = std::chrono::steady_clock::now();
    ggml_backend_sched_set_eval_callback(sched, callback, &state);
    const ggml_status status = ggml_backend_sched_graph_compute(sched, graph);
    ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
    state.dump();
    return status;
}

} // namespace core_sched_prof
