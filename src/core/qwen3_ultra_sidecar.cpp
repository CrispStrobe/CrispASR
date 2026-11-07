#include "qwen3_ultra_sidecar.h"
#include "gguf_loader.h"
#include <ggml.h>
#include <gguf.h>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <utility>

namespace qwen3_ultra_sidecar {
namespace {
struct TensorSpec {
    const char* name;
    std::array<int64_t, 4> ne; // GGML order, reversed from PyTorch
};
constexpr TensorSpec specs[] = {
    {"audio.0.weight", {5, 91, 128, 1}},
    {"audio.0.bias", {128, 1, 1, 1}},
    {"audio.2.weight", {5, 128, 128, 1}},
    {"audio.2.bias", {128, 1, 1, 1}},
    {"context_projection.weight", {1024, 128, 1, 1}},
    {"context_projection.bias", {128, 1, 1, 1}},
    {"role_embedding.weight", {128, 4, 1, 1}},
    {"endpoint_embedding.weight", {128, 2, 1, 1}},
    {"context_attention.in_proj_weight", {128, 384, 1, 1}},
    {"context_attention.in_proj_bias", {384, 1, 1, 1}},
    {"context_attention.out_proj.weight", {128, 128, 1, 1}},
    {"context_attention.out_proj.bias", {128, 1, 1, 1}},
    {"audio_attention.in_proj_weight", {128, 384, 1, 1}},
    {"audio_attention.in_proj_bias", {384, 1, 1, 1}},
    {"audio_attention.out_proj.weight", {128, 128, 1, 1}},
    {"audio_attention.out_proj.bias", {128, 1, 1, 1}},
    {"query_norm.weight", {128, 1, 1, 1}},
    {"query_norm.bias", {128, 1, 1, 1}},
    {"audio_norm.weight", {128, 1, 1, 1}},
    {"audio_norm.bias", {128, 1, 1, 1}},
    {"activity_bias.weight", {128, 1, 1, 1}},
    {"activity_bias.bias", {1, 1, 1, 1}},
    {"boundary_bias.weight", {128, 1, 1, 1}},
    {"boundary_bias.bias", {1, 1, 1, 1}},
    {"derivative_scale", {1, 1, 1, 1}},
    {"qwen_prior", {13, 1, 1, 1}},
    {"normalizer.mean", {91, 1, 1, 1}},
    {"normalizer.std", {91, 1, 1, 1}},
};
constexpr const char* revision = "765dc1d714e4c80196782a5df5bfbcb3e1416736";
} // namespace

bool load(const char* path, ggml_backend_t backend, Package& out) {
    if (!path || !*path || !backend)
        return false;
    auto* meta = core_gguf::open_metadata(path);
    if (!meta)
        return false;
    bool valid = core_gguf::kv_str(meta, "general.architecture", "") == "qwen3_ultra_head" &&
                 core_gguf::kv_str(meta, "qwen3_ultra_head.source_revision", "") == revision &&
                 gguf_get_n_tensors(meta) == static_cast<int64_t>(sizeof(specs) / sizeof(specs[0]));
    std::set<std::string> names;
    for (const auto& spec : specs) {
        const int64_t id = gguf_find_tensor(meta, spec.name);
        if (id < 0 || !names.insert(spec.name).second || gguf_get_tensor_type(meta, id) != GGML_TYPE_F32) {
            valid = false;
            break;
        }
        const int64_t* ne = gguf_get_tensor_ne(meta, id);
        if (!ne || !std::equal(spec.ne.begin(), spec.ne.end(), ne)) {
            valid = false;
            break;
        }
    }
    core_gguf::free_metadata(meta);
    if (!valid)
        return false;
    core_gguf::WeightLoad loaded;
    if (!core_gguf::load_weights(path, backend, "qwen3-ultra-head", loaded))
        return false;
    struct LoadedGuard {
        core_gguf::WeightLoad& value;
        ~LoadedGuard() { core_gguf::free_weights(value); }
    } guard{loaded};
    std::map<std::string, std::vector<float>> values;
    for (const auto& spec : specs) {
        const auto it = loaded.tensors.find(spec.name);
        if (it == loaded.tensors.end() || !it->second || it->second->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(it->second)) {
            valid = false;
            break;
        }
        const auto count = static_cast<size_t>(ggml_nelements(it->second));
        auto& data = values[spec.name];
        data.resize(count);
        ggml_backend_tensor_get(it->second, data.data(), 0, count * sizeof(float));
        if (!std::all_of(data.begin(), data.end(), [](float x) { return std::isfinite(x); })) {
            valid = false;
            break;
        }
    }
    if (!valid)
        return false;
    Package p;
    auto& w = p.weights;
    w.audio0_weight = std::move(values["audio.0.weight"]);
    w.audio0_bias = std::move(values["audio.0.bias"]);
    w.audio2_weight = std::move(values["audio.2.weight"]);
    w.audio2_bias = std::move(values["audio.2.bias"]);
    w.context_weight = std::move(values["context_projection.weight"]);
    w.context_bias = std::move(values["context_projection.bias"]);
    w.role_embedding = std::move(values["role_embedding.weight"]);
    w.endpoint_embedding = std::move(values["endpoint_embedding.weight"]);
    auto attention = [&](qwen3_ultra_head::AttentionWeights& a, const std::string& prefix) {
        a.in_weight = std::move(values[prefix + ".in_proj_weight"]);
        a.in_bias = std::move(values[prefix + ".in_proj_bias"]);
        a.out_weight = std::move(values[prefix + ".out_proj.weight"]);
        a.out_bias = std::move(values[prefix + ".out_proj.bias"]);
    };
    attention(w.context_attention, "context_attention");
    attention(w.audio_attention, "audio_attention");
    w.query_norm_weight = std::move(values["query_norm.weight"]);
    w.query_norm_bias = std::move(values["query_norm.bias"]);
    w.audio_norm_weight = std::move(values["audio_norm.weight"]);
    w.audio_norm_bias = std::move(values["audio_norm.bias"]);
    w.activity_weight = std::move(values["activity_bias.weight"]);
    w.activity_bias = std::move(values["activity_bias.bias"]);
    w.boundary_weight = std::move(values["boundary_bias.weight"]);
    w.boundary_bias = std::move(values["boundary_bias.bias"]);
    w.derivative_scale = std::move(values["derivative_scale"]);
    w.prior = std::move(values["qwen_prior"]);
    const auto& mean = values["normalizer.mean"];
    const auto& stddev = values["normalizer.std"];
    if (!w.valid() || mean.size() != p.mean.size() || stddev.size() != p.stddev.size() ||
        !std::all_of(stddev.begin(), stddev.end(), [](float x) { return x > 0.0f; }))
        return false;
    std::copy(mean.begin(), mean.end(), p.mean.begin());
    std::copy(stddev.begin(), stddev.end(), p.stddev.begin());
    out = std::move(p);
    return true;
}
} // namespace qwen3_ultra_sidecar
