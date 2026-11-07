// Native SCNet / SCNet-large inference. Original architecture and checkpoints:
// starrytong/SCNet (MIT), 5d95bf96b19c3eede63248d171efeca8e3abb948.
// Learned convolutions/projections and normalization use ggml. Exact-length
// Fourier transforms and the sequential LSTM are fused native CPU graph nodes.
#include "scnet.h"
#include "scnet_fft.h"
#include "scnet_real_fft.h"
#include "scnet_im2col.h"
#include "scnet_lstm.h"
#include "scnet_shift.h"
#include "scnet_tensor.h"
#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/parallel_for.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
constexpr int FFT = 4096, HOP = 1024, RATE = 44100, STEM_COUNT = 4;
constexpr size_t GRAPH_CAPACITY = 32768;
constexpr const char* REVISION = "5d95bf96b19c3eede63248d171efeca8e3abb948";
const char* STEMS[] = {"drums", "bass", "other", "vocals"};
using Shape = std::array<int64_t, 4>;
struct Capture {
    Shape shape{};
    std::vector<float> values;
};
using Captures = std::map<std::string, Capture>;
using Sources = std::array<std::vector<float>, STEM_COUNT>;

bool flag(const char* name, bool fallback = false) {
    const char* value = std::getenv(name);
    return value && *value ? std::strcmp(value, "0") != 0 : fallback;
}

struct Timer {
    const char* name;
    bool enabled;
    std::chrono::steady_clock::time_point start;
    explicit Timer(const char* stage, bool active = true)
        : name(stage), enabled(active && flag("CRISPASR_SCNET_BENCH")) {
        if (enabled)
            start = std::chrono::steady_clock::now();
    }
    ~Timer() {
        if (enabled)
            std::fprintf(stderr, "scnet_bench: %s %.3f ms\n", name,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
};

// Diagnostic-only attribution. Asking the scheduler to observe every node
// splits execution and synchronizes each observation. These measurements help
// locate costs; they are not a normal-path throughput measurement. The callback
// is never installed when CRISPASR_SCNET_BENCH is disabled.
struct GraphProfile {
    using Key = std::tuple<std::string, std::string, std::string>;
    struct Sample {
        double milliseconds = 0;
        uint64_t calls = 0;
    };
    bool enabled = false;
    std::map<const ggml_tensor*, std::string> labels;
    std::map<Key, Sample> samples;
    std::map<const ggml_tensor*, Sample*> nodes;
    Sample unmapped;
    Sample* active = nullptr;
    std::chrono::steady_clock::time_point started;

    void label(ggml_tensor* tensor, const std::string& stage) {
        if (!enabled)
            return;
        std::vector<ggml_tensor*> pending{tensor};
        while (!pending.empty()) {
            auto* current = pending.back();
            pending.pop_back();
            if (!labels.emplace(current, stage).second)
                continue;
            for (auto* source : current->src)
                if (source)
                    pending.push_back(source);
        }
    }

    void begin(ggml_cgraph* graph, ggml_backend_sched_t scheduler) {
        samples.clear();
        nodes.clear();
        unmapped = {};
        active = nullptr;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto* tensor = ggml_graph_node(graph, i);
            const auto stage = labels.find(tensor);
            const auto backend = ggml_backend_sched_get_tensor_backend(scheduler, tensor);
            const char* operation = ggml_op_desc(tensor);
            const char* name = ggml_get_name(tensor);
            if (tensor->op == GGML_OP_CUSTOM && std::strncmp(name, "scnet_cpu_", 10) == 0)
                operation = name;
            const Key key{stage == labels.end() ? "unattributed" : stage->second, operation,
                          backend ? ggml_backend_name(backend) : "unassigned"};
            nodes.emplace(tensor, &samples[key]);
        }
        ggml_backend_sched_set_eval_callback(scheduler, observe, this);
    }

    static bool observe(ggml_tensor* tensor, bool ask, void* opaque) {
        auto& profile = *static_cast<GraphProfile*>(opaque);
        if (ask) {
            const auto found = profile.nodes.find(tensor);
            profile.active = found == profile.nodes.end() ? &profile.unmapped : found->second;
            profile.started = std::chrono::steady_clock::now();
        } else if (profile.active) {
            const auto finished = std::chrono::steady_clock::now();
            profile.active->milliseconds +=
                std::chrono::duration<double, std::milli>(finished - profile.started).count();
            ++profile.active->calls;
            profile.active = nullptr;
        }
        return true;
    }

    void report(int frames) const {
        double milliseconds = unmapped.milliseconds;
        uint64_t count = unmapped.calls, groups = 0;
        for (const auto& entry : samples) {
            if (!entry.second.calls)
                continue;
            ++groups;
            count += entry.second.calls;
            milliseconds += entry.second.milliseconds;
            std::fprintf(stderr, "scnet_bench_op: frames=%d stage=%s op=%s backend=%s calls=%llu ms=%.3f\n", frames,
                         std::get<0>(entry.first).c_str(), std::get<1>(entry.first).c_str(),
                         std::get<2>(entry.first).c_str(), static_cast<unsigned long long>(entry.second.calls),
                         entry.second.milliseconds);
        }
        std::fprintf(stderr,
                     "scnet_bench_ops: frames=%d nodes=%llu groups=%llu measured_ms=%.3f unmapped_nodes=%llu "
                     "unmapped_ms=%.3f scope=per_node_synchronized\n",
                     frames, static_cast<unsigned long long>(count), static_cast<unsigned long long>(groups),
                     milliseconds, static_cast<unsigned long long>(unmapped.calls), unmapped.milliseconds);
    }
};

std::vector<int> integer_array(gguf_context* meta, const char* key) {
    const int64_t k = gguf_find_key(meta, key);
    if (k < 0 || gguf_get_kv_type(meta, k) != GGUF_TYPE_ARRAY || gguf_get_arr_n(meta, k) > 64)
        return {};
    const auto type = gguf_get_arr_type(meta, k);
    if (type != GGUF_TYPE_INT32 && type != GGUF_TYPE_UINT32)
        return {};
    const int32_t* data = static_cast<const int32_t*>(gguf_get_arr_data(meta, k));
    return std::vector<int>(data, data + gguf_get_arr_n(meta, k));
}

struct HParams {
    std::vector<int> dims;
    uint32_t file_type = std::numeric_limits<uint32_t>::max();
    std::array<double, 3> canonical_band_rates() const {
        // These are the author's Python/YAML double constants, not promoted
        // F32 GGUF values: rounding can move a ceil boundary by one bin.
        return dims == std::vector<int>{4, 64, 128, 256} ? std::array<double, 3>{0.225, 0.372, 0.403}
                                                         : std::array<double, 3>{0.175, 0.392, 0.433};
    }
    std::array<int, 4> band_boundaries(int frequencies) const {
        const auto rates = canonical_band_rates();
        return {0, static_cast<int>(std::ceil(frequencies * rates[0])),
                static_cast<int>(std::ceil(frequencies * (rates[0] + rates[1]))), frequencies};
    }
    bool read(gguf_context* meta) {
        dims = integer_array(meta, "scnet.dims");
        file_type = core_gguf::kv_u32(meta, "general.file_type", std::numeric_limits<uint32_t>::max());
        auto integer = [&](const char* key) { return core_gguf::kv_u32(meta, key, 0); };
        const auto bands = core_gguf::kv_f32_array(meta, "scnet.band_sr");
        const auto rates = canonical_band_rates();
        const std::vector<float> expected_bands(rates.begin(), rates.end());
        const auto sources = core_gguf::kv_str_array(meta, "scnet.source_names");
        const std::vector<std::string> expected_sources(STEMS, STEMS + STEM_COUNT);
        return core_gguf::kv_str(meta, "general.architecture", "") == "scnet" &&
               (file_type == GGML_FTYPE_ALL_F32 || file_type == GGML_FTYPE_MOSTLY_F16 ||
                file_type == GGML_FTYPE_MOSTLY_Q8_0 || file_type == GGML_FTYPE_MOSTLY_Q4_K) &&
               integer("scnet.tensor_layout_version") == 1 &&
               (dims == std::vector<int>{4, 32, 64, 128} || dims == std::vector<int>{4, 64, 128, 256}) &&
               integer("scnet.n_fft") == FFT && integer("scnet.hop_length") == HOP &&
               integer("scnet.win_length") == FFT && integer("scnet.sample_rate") == RATE &&
               integer("scnet.audio_channels") == 2 && integer("scnet.num_sources") == STEM_COUNT &&
               integer("scnet.num_dplayer") == 6 && integer("scnet.expand") == 1 && integer("scnet.compress") == 4 &&
               integer("scnet.conv_kernel") == 3 && core_gguf::kv_bool(meta, "scnet.normalized", false) &&
               integer_array(meta, "scnet.band_stride") == std::vector<int>{1, 4, 16} &&
               integer_array(meta, "scnet.band_kernel") == std::vector<int>{3, 4, 16} &&
               integer_array(meta, "scnet.conv_depths") == std::vector<int>{3, 2, 1} && bands == expected_bands &&
               sources == expected_sources;
    }
};

std::vector<float> read_tensor(const ggml_tensor* tensor) {
    if (!tensor || tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor))
        throw std::runtime_error("SCNet captures require contiguous F32 tensors");
    std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

void put(Captures* captures, const std::string& name, Shape shape, std::vector<float> values) {
    if (captures)
        (*captures)[name] = {shape, std::move(values)};
}

struct FourierNode {
    scnet_fft::Plan plan;
    bool inverse;
    struct Scratch {
        std::vector<float> real, imag;
        scnet_fft::Plan::Scratch transform;
        explicit Scratch(int n) : real(n), imag(n) {}
    };
    std::vector<Scratch> scratch;
    FourierNode(int n, bool inv, int threads) : plan(n), inverse(inv) {
        for (int i = 0; i < threads; ++i)
            scratch.emplace_back(n);
    }
};

void fourier_compute(ggml_tensor* output, int ith, int nth, void* opaque) {
    // Diagnostic worker-zero span, not the sum of parallel worker CPU time.
    Timer timer("fft", ith == 0);
    auto& node = *static_cast<FourierNode*>(opaque);
    auto& scratch = node.scratch[ith];
    const auto* input = output->src[0];
    const float* src = static_cast<const float*>(input->data);
    float* dst = static_cast<float*>(output->data);
    const int frequencies = static_cast<int>(input->ne[1]);
    const int channels = static_cast<int>(node.inverse ? input->ne[2] / 2 : input->ne[2]);
    scnet_fft::transform_real_rows(node.plan, node.inverse, src, dst, frequencies, channels, ith, nth,
                                   scratch.real.data(), scratch.imag.data(), scratch.transform);
}

enum class RecurrentMode { portable, cblas, packed };

struct RecurrentNode {
    RecurrentMode mode;
    scnet_recurrent::Kernel kernel;
    std::array<const scnet_recurrent::PackedWeight*, 2> packed{};
    std::string name;
    std::vector<scnet_detail::LstmScratch> scratch;
    RecurrentNode(int batch, int hidden, int threads, RecurrentMode execution, const std::string& label = "unnamed")
        : mode(execution), name(label) {
        const bool blas = mode == RecurrentMode::cblas;
        for (int i = 0; i < (blas ? 1 : threads); ++i)
            scratch.emplace_back(blas ? batch : (batch + threads - 1) / threads, hidden);
    }
    const char* mode_name() const {
        return mode == RecurrentMode::packed ? kernel.name : mode == RecurrentMode::cblas ? "CBLAS" : "portable";
    }
};

void im2col_compute(ggml_tensor* output, int ith, int nth, void* opaque) {
    const auto& shape = *static_cast<const scnet_detail::Im2ColShape*>(opaque);
    const auto* input = output->src[0];
    scnet_detail::im2col_rows(static_cast<const float*>(input->data), static_cast<float*>(output->data), shape,
                              input->nb[1] / sizeof(float), input->nb[2] / sizeof(float), ith, nth);
}

void im2col_columns_compute(ggml_tensor* output, int ith, int nth, void* opaque) {
    const auto& shape = *static_cast<const scnet_detail::Im2ColShape*>(opaque);
    const auto* input = output->src[0];
    scnet_detail::im2col_columns(static_cast<const float*>(input->data), static_cast<float*>(output->data), shape,
                                 input->nb[1] / sizeof(float), input->nb[2] / sizeof(float), ith, nth);
}

void recurrent_compute(ggml_tensor* output, int ith, int nth, void* opaque) {
    auto& node = *static_cast<RecurrentNode*>(opaque);
    const bool blas = node.mode == RecurrentMode::cblas;
    // This ggml revision can invoke CUSTOM for every worker in the surrounding
    // CPU subgraph even when the node requests one task. BLAS owns its own
    // threading; only worker zero may enter the fused recurrence.
    if (blas && ith != 0)
        return;
    // CBLAS is owned entirely by worker zero. Packed and portable paths own
    // disjoint complete batch recurrences and need no per-step synchronization.
    // Their timer reports worker zero only, not summed overlapping worker time;
    // the separate graph-node timer measures the complete parallel operation.
    Timer timer("lstm", ith == 0);
    const int hidden = static_cast<int>(output->ne[0] / 2), batch = static_cast<int>(output->ne[1]);
    const int steps = static_cast<int>(output->ne[2]);
    const int begin = blas ? 0 : static_cast<int>(int64_t(batch) * ith / nth);
    const int end = blas ? batch : static_cast<int>(int64_t(batch) * (ith + 1) / nth);
    if (begin == end)
        return;
    auto& scratch = node.scratch[ith];
    const int capacity = static_cast<int>(scratch.hidden.size() / hidden);
    const bool profiling = ith == 0 && flag("CRISPASR_SCNET_BENCH");
    scnet_detail::LstmTiming timings;
    for (int direction = 0; direction < 2; ++direction)
        for (int first = begin; first < end; first += capacity)
            scnet_detail::lstm_direction(
                static_cast<const float*>(output->src[direction]->data),
                static_cast<const float*>(output->src[2 + direction]->data),
                static_cast<const float*>(output->src[4 + direction]->data), static_cast<float*>(output->data), hidden,
                batch, steps, direction != 0, first, std::min(first + capacity, end), blas, scratch,
                profiling ? &timings : nullptr,
                output->src[6 + direction] ? static_cast<const float*>(output->src[6 + direction]->data) : nullptr,
                output->src[direction]->nb[1] / sizeof(float), node.packed[direction], node.kernel.product);
    if (profiling)
        std::fprintf(stderr,
                     "scnet_bench_lstm: name=%s mode=%s scope=%s hidden=%d batch=%d steps=%d gemm_calls=%llu "
                     "gate_steps=%llu gemm_ms=%.3f gates_ms=%.3f reset_ms=%.3f\n",
                     node.name.c_str(), node.mode_name(), blas ? "whole_node" : "worker_zero", hidden, batch, steps,
                     static_cast<unsigned long long>(timings.gemm_calls),
                     static_cast<unsigned long long>(timings.gate_steps), timings.gemm_ms, timings.gates_ms,
                     timings.reset_ms);
}

struct Graph {
    ggml_context* context = nullptr;
    ggml_backend_sched_t scheduler = nullptr;
    bool owns_scheduler = true;
    ggml_cgraph* graph = nullptr;
    ggml_tensor *input = nullptr, *output = nullptr;
    int frames = 0;
    bool capture = false;
    bool cpu_only = false;
    std::map<std::string, ggml_tensor*> captures;
    std::map<ggml_tensor*, ggml_tensor*> f32;
    std::vector<std::unique_ptr<FourierNode>> fourier_nodes;
    std::vector<std::unique_ptr<RecurrentNode>> recurrent_nodes;
    std::vector<std::unique_ptr<scnet_detail::Im2ColShape>> convolution_inputs;
    GraphProfile profile;
    ~Graph() {
        if (scheduler) {
            if (owns_scheduler) {
                ggml_backend_sched_free(scheduler);
            } else {
                // The model owns the CPU allocator. Retire its references to
                // this graph while the tensors and callback data are alive,
                // including when allocation failed before the first compute.
                ggml_backend_sched_synchronize(scheduler);
                ggml_backend_sched_set_eval_callback(scheduler, nullptr, nullptr);
                ggml_backend_sched_reset(scheduler);
            }
        }
        if (context)
            ggml_free(context);
    }
};

size_t graph_buffer_bytes(ggml_backend_sched_t scheduler) {
    size_t bytes = 0;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(scheduler); ++i)
        bytes += ggml_backend_sched_get_buffer_size(scheduler, ggml_backend_sched_get_backend(scheduler, i));
    return bytes;
}

void profile_graph_allocator(const Graph& graph, bool metadata_hit, bool scheduler_created, size_t before_bytes) {
    if (!graph.cpu_only || !graph.profile.enabled)
        return;
    const size_t after_bytes = graph_buffer_bytes(graph.scheduler);
    void* buffer = graph.input && graph.input->buffer ? ggml_backend_buffer_get_base(graph.input->buffer) : nullptr;
    // Capacity/base identity is reuse evidence, not a count of allocator calls.
    std::fprintf(stderr,
                 "scnet_bench_allocator: frames=%d capture=%d profile=%d shared_cpu=%d metadata_hit=%d "
                 "scheduler=%p buffer=%p before_bytes=%zu after_bytes=%zu scheduler_created=%d\n",
                 graph.frames, int(graph.capture), int(graph.profile.enabled), int(!graph.owns_scheduler),
                 int(metadata_hit), static_cast<void*>(graph.scheduler), buffer, before_bytes, after_bytes,
                 int(scheduler_created));
}
} // namespace

struct scnet_context {
    HParams hp;
    scnet_params params{};
    core_gguf::WeightLoad weights;
    // Immutable CPU+BLAS W_hh copies keep dequantization out of the LSTM
    // subgraph. Their F32 payload is 15 MiB for SCNet / 60 MiB for SCNet-large.
    // Original GGUF tensors and their quantization/provenance remain intact.
    core_gguf::WeightLoad recurrent_weights;
    // Pure-permutation SIMD packs add 15/60 MiB. The original GGUF and exact
    // F32 copies above remain available and retain their provenance/lifetime.
    std::map<std::string, scnet_recurrent::PackedWeight> recurrent_packed_weights;
    scnet_recurrent::Kernel recurrent_kernel;
    RecurrentMode recurrent_mode = RecurrentMode::portable;
    ggml_backend_t backend = nullptr, cpu = nullptr, blas = nullptr;
    // CPU graph metadata is replaceable; one allocator retains its largest
    // compute buffer instead of keeping separate buffers for each geometry.
    ggml_backend_sched_t cpu_scheduler = nullptr;
    std::vector<std::unique_ptr<Graph>> graphs;
    scnet_fft::Plan stft_plan{FFT};
    bool quant_q4 = false, quant_q8 = false;
    std::set<std::string> required_weights;
    bool recurrent_blas = false;
    std::string backend_description, compute_description;
};

namespace {
ggml_tensor* weight(scnet_context& model, const std::string& name) {
    const auto found = model.weights.tensors.find(name);
    if (found == model.weights.tensors.end())
        throw std::runtime_error("missing SCNet tensor: " + name);
    return found->second;
}

void expect(scnet_context& model, const std::string& name, int64_t d0, int64_t d1 = 1, bool matrix = false) {
    const auto* tensor = weight(model, name);
    if (tensor->ne[0] != d0 || tensor->ne[1] != d1 || tensor->ne[2] != 1 || tensor->ne[3] != 1)
        throw std::runtime_error("incompatible SCNet tensor shape: " + name);
    if ((!matrix && tensor->type != GGML_TYPE_F32) ||
        (matrix && tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_Q8_0 &&
         tensor->type != GGML_TYPE_Q4_K))
        throw std::runtime_error("unsupported SCNet tensor precision: " + name);
    model.required_weights.insert(name);
}

void validate_weights(scnet_context& model) {
    const auto& dims = model.hp.dims;
    const int kernels[] = {3, 4, 16}, depths[] = {3, 2, 1};
    auto norm = [&](const std::string& p, int channels) {
        expect(model, p + ".weight", channels);
        expect(model, p + ".bias", channels);
    };
    auto conv = [&](const std::string& p, int in, int out, int kernel) {
        expect(model, p + ".weight", int64_t(in) * kernel, out, true);
        expect(model, p + ".bias", out);
    };
    for (int level = 0; level < 3; ++level) {
        const int in = dims[level], out = dims[level + 1], hidden = out / 4;
        const std::string p = "encoder." + std::to_string(level);
        for (int band = 0; band < 3; ++band) {
            conv(p + ".SDlayer.convs." + std::to_string(band), in, out, kernels[band]);
            for (int depth = 0; depth < depths[band]; ++depth) {
                const std::string q = p + ".conv_modules." + std::to_string(band) + ".layers." + std::to_string(depth);
                norm(q + ".0", out);
                conv(q + ".1", out, 2 * hidden, 3);
                conv(q + ".3", 1, hidden, 3);
                norm(q + ".4", hidden);
                conv(q + ".6", hidden, out, 1);
            }
        }
        conv(p + ".globalconv", out, out, 9);
        const std::string q = "decoder." + std::to_string(2 - level);
        conv(q + ".0.conv", 2 * out, 2 * out, 9);
        for (int band = 0; band < 3; ++band) {
            const int output = level == 0 ? 16 : in;
            const std::string r = q + ".1.convtrs." + std::to_string(band);
            // Converter: original [in,out,k,1] -> [out*k,in].
            expect(model, r + ".weight", out, int64_t(output) * kernels[band], true);
            expect(model, r + ".bias", output);
        }
    }
    for (int layer = 0; layer < 6; ++layer) {
        const int channels = dims.back() * (layer % 2 ? 2 : 1);
        const std::string p = "separation_net.dp_modules." + std::to_string(layer);
        for (int axis = 0; axis < 2; ++axis) {
            norm(p + ".norm_layers." + std::to_string(axis), channels);
            conv(p + ".linear_layers." + std::to_string(axis), 2 * channels, channels, 1);
            const std::string q = p + ".lstm_layers." + std::to_string(axis);
            for (const char* suffix : {"", "_reverse"}) {
                expect(model, q + ".weight_ih_l0" + suffix, channels, 4 * channels, true);
                expect(model, q + ".weight_hh_l0" + suffix, channels, 4 * channels, true);
                expect(model, q + ".bias_ih_l0" + suffix, 4 * channels);
                expect(model, q + ".bias_hh_l0" + suffix, 4 * channels);
            }
        }
    }
    bool q4 = false, q8 = false, f16 = false;
    for (const auto& entry : model.weights.tensors) {
        if (!model.required_weights.count(entry.first))
            throw std::runtime_error("unexpected SCNet tensor: " + entry.first);
        q4 |= entry.second->type == GGML_TYPE_Q4_K;
        q8 |= entry.second->type == GGML_TYPE_Q8_0;
        f16 |= entry.second->type == GGML_TYPE_F16;
    }
    const auto type = model.hp.file_type;
    if ((type == GGML_FTYPE_ALL_F32 && (f16 || q4 || q8)) || (type == GGML_FTYPE_MOSTLY_F16 && (q4 || q8)) ||
        (type == GGML_FTYPE_MOSTLY_Q8_0 && (q4 || !q8)) || (type == GGML_FTYPE_MOSTLY_Q4_K && !q4))
        throw std::runtime_error("SCNet declared precision does not match the required weight tensors");
    model.quant_q4 = type == GGML_FTYPE_MOSTLY_Q4_K;
    model.quant_q8 = type == GGML_FTYPE_MOSTLY_Q8_0;
}

void materialize_recurrent_weights(scnet_context& model) {
    if (model.backend != model.cpu || !model.blas || !model.recurrent_blas)
        return;
    auto& cache = model.recurrent_weights;
    constexpr size_t count = 6 * 2 * 2;
    const ggml_init_params params{count * ggml_tensor_overhead(), nullptr, true};
    cache.ctx = ggml_init(params);
    if (!cache.ctx)
        throw std::runtime_error("SCNet recurrent weight metadata allocation failed");
    for (int layer = 0; layer < 6; ++layer)
        for (int axis = 0; axis < 2; ++axis)
            for (const char* suffix : {"", "_reverse"}) {
                const std::string name = "separation_net.dp_modules." + std::to_string(layer) + ".lstm_layers." +
                                         std::to_string(axis) + ".weight_hh_l0" + suffix;
                const auto* source = weight(model, name);
                if (!ggml_is_contiguous(source) || !source->data || !source->buffer ||
                    !ggml_backend_buffer_is_host(source->buffer))
                    throw std::runtime_error("SCNet recurrent weight is not contiguous host storage: " + name);
                auto* tensor = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, source->ne[0], source->ne[1]);
                ggml_set_name(tensor, name.c_str());
                // A new tensor is an op-NONE leaf, with no graph edge back to
                // its original stored representation and no conversion node.
                cache.tensors.emplace(name, tensor);
            }
    cache.buf = ggml_backend_alloc_ctx_tensors(cache.ctx, model.cpu);
    if (!cache.buf)
        throw std::runtime_error("SCNet recurrent F32 weight allocation failed");
    ggml_backend_buffer_set_usage(cache.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const auto& entry : cache.tensors) {
        const auto* source = weight(model, entry.first);
        auto* destination = entry.second;
        if (source->type == GGML_TYPE_F32) {
            std::memcpy(destination->data, source->data, ggml_nbytes(source));
            continue;
        }
        const auto* traits = ggml_get_type_traits(source->type);
        if (!traits->to_float)
            throw std::runtime_error("SCNet recurrent weight has no F32 converter: " + entry.first);
        // Quantized CPY uses this exact type-trait converter one block at a
        // time. F16's canonical row converter preserves every finite half
        // value exactly, like CPY's CPU half-to-float conversion.
        const int64_t block = ggml_is_quantized(source->type) ? traits->blck_size : source->ne[0];
        for (int64_t row = 0; row < source->ne[1]; ++row) {
            const auto* from = static_cast<const char*>(source->data) + row * source->nb[1];
            auto* to = reinterpret_cast<float*>(static_cast<char*>(destination->data) + row * destination->nb[1]);
            for (int64_t first = 0; first < source->ne[0]; first += block)
                traits->to_float(from + (first / traits->blck_size) * traits->type_size, to + first, block);
        }
    }
    model.recurrent_kernel = scnet_recurrent::selected_kernel();
    if (model.recurrent_kernel.product) {
        for (const auto& entry : cache.tensors)
            model.recurrent_packed_weights.try_emplace(entry.first, static_cast<const float*>(entry.second->data),
                                                       static_cast<int>(entry.second->ne[0]));
        model.recurrent_mode = RecurrentMode::packed;
    }
}

ggml_tensor* as_f32(Graph& graph, ggml_tensor* tensor) {
    if (tensor->type == GGML_TYPE_F32)
        return tensor;
    auto found = graph.f32.find(tensor);
    if (found != graph.f32.end())
        return found->second;
    auto* cast = ggml_cast(graph.context, tensor, GGML_TYPE_F32);
    graph.f32.emplace(tensor, cast);
    return cast;
}

// These graph edges only require a contiguous view; they do not require a
// second allocation. Captures still explicitly copy and retain their outputs.
ggml_tensor* contiguous(Graph& graph, ggml_tensor* tensor) {
    return ggml_is_contiguous(tensor) ? tensor : scnet_tensor::cont(graph.context, tensor, graph.cpu_only);
}

ggml_tensor* tensor_add(Graph& graph, ggml_tensor* a, ggml_tensor* b) {
    return scnet_tensor::add(graph.context, a, b, graph.cpu_only);
}

ggml_tensor* tensor_mul(Graph& graph, ggml_tensor* a, ggml_tensor* b) {
    return scnet_tensor::mul(graph.context, a, b, graph.cpu_only);
}

ggml_tensor* multiply(scnet_context& model, Graph& graph, ggml_tensor* weights, ggml_tensor* input) {
    auto* ctx = graph.context;
    input = contiguous(graph, input);
    const int64_t n1 = input->ne[1], n2 = input->ne[2], n3 = input->ne[3];
    input = ggml_reshape_2d(ctx, input, input->ne[0], n1 * n2 * n3);
    ggml_tensor* result = ggml_mul_mat(ctx, weights, input);
    if (model.backend == model.cpu && weights->type != GGML_TYPE_F32 &&
        (!model.blas || !ggml_backend_supports_op(model.blas, result)))
        result = ggml_mul_mat(ctx, as_f32(graph, weights), input);
    ggml_prec_set_acc(result, GGML_PREC_F32);
    return ggml_reshape_4d(ctx, result, weights->ne[1], n1, n2, n3);
}

ggml_tensor* affine(scnet_context& model, Graph& graph, ggml_tensor* input, const std::string& name) {
    auto* result = multiply(model, graph, weight(model, name + ".weight"), input);
    return tensor_add(graph, result, weight(model, name + ".bias"));
}

ggml_tensor* join(Graph& graph, std::vector<ggml_tensor*> tensors, int axis) {
    auto* ctx = graph.context;
    while (tensors.size() > 1) {
        std::vector<ggml_tensor*> next;
        for (size_t i = 0; i < tensors.size(); i += 2)
            next.push_back(i + 1 < tensors.size()
                               ? scnet_tensor::concat(ctx, tensors[i], tensors[i + 1], axis, graph.cpu_only)
                               : tensors[i]);
        tensors = std::move(next);
    }
    return tensors.at(0);
}

void capture(Graph& graph, const std::string& name, ggml_tensor* tensor) {
    if (!graph.capture) {
        graph.profile.label(tensor, name);
        return;
    }
    tensor = scnet_tensor::cont(graph.context, tensor, graph.cpu_only);
    graph.profile.label(tensor, name);
    ggml_set_name(tensor, name.c_str());
    ggml_set_output(tensor);
    ggml_build_forward_expand(graph.graph, tensor);
    graph.captures[name] = tensor;
}

// Canonical activation layout is [T,F,C] in ggml, corresponding to Python's
// contiguous [C,F,T]. Every band/time permutation is explicit.
ggml_tensor* slice_frequency(ggml_context* ctx, ggml_tensor* input, int begin, int count) {
    return ggml_view_3d(ctx, input, input->ne[0], count, input->ne[2], input->nb[1], input->nb[2],
                        static_cast<size_t>(begin) * input->nb[1]);
}

ggml_tensor* channel_bias(scnet_context& model, Graph& graph, ggml_tensor* input, const std::string& name) {
    auto* ctx = graph.context;
    auto* bias = weight(model, name + ".bias");
    return tensor_add(graph, input, ggml_reshape_3d(ctx, bias, 1, 1, bias->ne[0]));
}

ggml_tensor* convolution(scnet_context& model, Graph& graph, ggml_tensor* input, ggml_tensor* weights,
                         const std::string& name, int frequency_kernel, int time_kernel, int frequency_stride,
                         int left_frequency_pad, int right_frequency_pad, int time_pad) {
    auto* ctx = graph.context;
    const int64_t channels = input->ne[2], out_channels = weights->ne[1];
    if (left_frequency_pad || right_frequency_pad || time_pad)
        input = scnet_tensor::pad_ext(ctx, input, time_pad, time_pad, left_frequency_pad, right_frequency_pad, 0, 0, 0,
                                      0, graph.cpu_only);
    const int times = static_cast<int>(input->ne[0]) - time_kernel + 1;
    const int frequencies = (static_cast<int>(input->ne[1]) - frequency_kernel) / frequency_stride + 1;
    // A column-oriented BLAS product writes [T,F,Cout] directly. Its input
    // packing copies contiguous time rows instead of interleaving every kernel
    // element, and its output needs no channel/time transpose. Keep the
    // ordinary ggml path for GPU and CPU builds without the BLAS backend.
    const bool column_product = model.backend == model.cpu && model.blas;
    auto* column_weights = column_product ? ggml_transpose(ctx, as_f32(graph, weights)) : nullptr;
    if (column_product && frequency_kernel == 1 && time_kernel == 1 && frequency_stride == 1) {
        // A frequency tile retains the parent's channel stride. Use the whole
        // contiguous input so a pointwise convolution needs neither im2col nor
        // a chain of output-tile concatenations.
        auto* columns = ggml_reshape_2d(ctx, contiguous(graph, input), int64_t(times) * frequencies, channels);
        auto* output = ggml_out_prod(ctx, columns, column_weights);
        return channel_bias(model, graph, ggml_reshape_3d(ctx, output, times, frequencies, out_channels), name);
    }
    auto* shape = model.backend == model.cpu
                      ? nullptr
                      : ggml_new_tensor_4d(ctx, GGML_TYPE_F16, time_kernel, frequency_kernel, channels, out_channels);
    std::vector<ggml_tensor*> tiles;
    // Keep complete time rows, but grow CPU BLAS tiles to reduce output joins.
    // Bound each packing tensor to 128 MiB; other paths retain 32 frequencies.
    constexpr int TILE = 32;
    int tile = TILE;
    if (column_product)
        tile = scnet_detail::im2col_column_tile_size(times, frequencies, weights->ne[0]);
    for (int begin = 0; begin < frequencies; begin += tile) {
        const int count = std::min(tile, frequencies - begin);
        auto* band =
            slice_frequency(ctx, input, begin * frequency_stride, (count - 1) * frequency_stride + frequency_kernel);
        ggml_tensor* columns;
        if (model.backend == model.cpu) {
            auto layout = std::make_unique<scnet_detail::Im2ColShape>();
            *layout = {times, count, static_cast<int>(channels), time_kernel, frequency_kernel, frequency_stride};
            ggml_tensor* sources[] = {band};
            if (column_product) {
                columns = ggml_custom_4d(ctx, GGML_TYPE_F32, int64_t(times) * count, weights->ne[0], 1, 1, sources, 1,
                                         im2col_columns_compute, GGML_N_TASKS_MAX, layout.get());
                ggml_set_name(columns, "scnet_cpu_im2col_columns");
            } else {
                columns = ggml_custom_4d(ctx, GGML_TYPE_F32, weights->ne[0], times, count, 1, sources, 1,
                                         im2col_compute, GGML_N_TASKS_MAX, layout.get());
                ggml_set_name(columns, "scnet_cpu_im2col");
            }
            graph.convolution_inputs.push_back(std::move(layout));
        } else {
            columns = ggml_im2col(ctx, shape, band, 1, frequency_stride, 0, 0, 1, 1, true, GGML_TYPE_F32);
        }
        ggml_tensor* output;
        if (column_product) {
            output = ggml_out_prod(ctx, columns, column_weights);
            output = ggml_reshape_3d(ctx, output, times, count, out_channels);
        } else {
            output =
                multiply(model, graph, weights, ggml_reshape_2d(ctx, columns, weights->ne[0], int64_t(times) * count));
            output = ggml_reshape_3d(ctx, output, out_channels, times, count);
            output = contiguous(graph, ggml_permute(ctx, output, 2, 0, 1, 3));
        }
        tiles.push_back(output);
    }
    return channel_bias(model, graph, join(graph, std::move(tiles), 1), name);
}

ggml_tensor* convolution(scnet_context& model, Graph& graph, ggml_tensor* input, const std::string& name, int kf,
                         int kt, int sf = 1, int left = 0, int right = 0, int time_pad = 0) {
    return convolution(model, graph, input, weight(model, name + ".weight"), name, kf, kt, sf, left, right, time_pad);
}

ggml_tensor* group_norm(scnet_context& model, Graph& graph, ggml_tensor* input, const std::string& name,
                        bool per_frequency) {
    auto* ctx = graph.context;
    const int64_t times = input->ne[0], frequencies = input->ne[1], channels = input->ne[2];
    ggml_tensor* normalized;
    if (per_frequency) {
        auto* rows = contiguous(graph, ggml_permute(ctx, input, 0, 2, 1, 3)); // [T,C,F]
        rows = ggml_norm(ctx, ggml_reshape_2d(ctx, rows, times * channels, frequencies), 1e-5f);
        rows = ggml_reshape_3d(ctx, rows, times, channels, frequencies);
        normalized = contiguous(graph, ggml_permute(ctx, rows, 0, 2, 1, 3));
    } else {
        auto* values = ggml_reshape_1d(ctx, contiguous(graph, input), times * frequencies * channels);
        normalized = ggml_reshape_3d(ctx, ggml_norm(ctx, values, 1e-5f), times, frequencies, channels);
    }
    auto* scale = ggml_reshape_3d(ctx, weight(model, name + ".weight"), 1, 1, channels);
    return channel_bias(model, graph, tensor_mul(graph, normalized, scale), name);
}

ggml_tensor* glu(Graph& graph, ggml_tensor* input) {
    auto* ctx = graph.context;
    const int64_t channels = input->ne[2] / 2;
    auto* a = ggml_view_3d(ctx, input, input->ne[0], input->ne[1], channels, input->nb[1], input->nb[2], 0);
    auto* b = ggml_view_3d(ctx, input, input->ne[0], input->ne[1], channels, input->nb[1], input->nb[2],
                           static_cast<size_t>(channels) * input->nb[2]);
    return tensor_mul(graph, a, ggml_sigmoid(ctx, b));
}

ggml_tensor* depthwise(scnet_context& model, Graph& graph, ggml_tensor* input, const std::string& name) {
    auto* ctx = graph.context;
    const int64_t times = input->ne[0], frequencies = input->ne[1], channels = input->ne[2];
    auto* padded = scnet_tensor::pad_ext(ctx, input, 1, 1, 0, 0, 0, 0, 0, 0, graph.cpu_only);
    auto* weights = as_f32(graph, weight(model, name + ".weight"));
    ggml_tensor* result = nullptr;
    for (int k = 0; k < 3; ++k) {
        auto* x =
            ggml_view_3d(ctx, padded, times, frequencies, channels, padded->nb[1], padded->nb[2], k * sizeof(float));
        auto* w = contiguous(
            graph, ggml_view_3d(ctx, weights, 1, 1, channels, sizeof(float), 3 * sizeof(float), k * sizeof(float)));
        auto* term = tensor_mul(graph, x, w);
        result = result ? tensor_add(graph, result, term) : term;
    }
    return channel_bias(model, graph, result, name);
}

ggml_tensor* convolution_module(scnet_context& model, Graph& graph, ggml_tensor* input, const std::string& name,
                                int depth) {
    auto* ctx = graph.context;
    for (int layer = 0; layer < depth; ++layer) {
        const std::string p = name + ".layers." + std::to_string(layer);
        auto* x = group_norm(model, graph, input, p + ".0", true);
        x = glu(graph, convolution(model, graph, x, p + ".1", 1, 3, 1, 0, 0, 1));
        x = depthwise(model, graph, x, p + ".3");
        x = ggml_silu(ctx, group_norm(model, graph, x, p + ".4", true));
        x = convolution(model, graph, x, p + ".6", 1, 1);
        input = tensor_add(graph, input, x);
    }
    return ggml_gelu_erf(ctx, input);
}

struct BandLengths {
    std::array<int, 3> original{}, compressed{};
};

ggml_tensor* encoder(scnet_context& model, Graph& graph, ggml_tensor* input, int level, ggml_tensor*& skip,
                     BandLengths& lengths) {
    auto* ctx = graph.context;
    const std::string p = "encoder." + std::to_string(level);
    const int frequencies = static_cast<int>(input->ne[1]);
    const auto boundaries = model.hp.band_boundaries(frequencies);
    const int strides[] = {1, 4, 16}, kernels[] = {3, 4, 16}, depths[] = {3, 2, 1};
    std::vector<ggml_tensor*> bands;
    for (int band = 0; band < 3; ++band) {
        const int count = boundaries[band + 1] - boundaries[band];
        const int pad =
            strides[band] == 1 ? kernels[band] - 1 : (strides[band] - count % strides[band]) % strides[band];
        auto* x = slice_frequency(ctx, input, boundaries[band], count);
        x = convolution(model, graph, x, p + ".SDlayer.convs." + std::to_string(band), kernels[band], 1, strides[band],
                        pad / 2, pad - pad / 2);
        lengths.original[band] = count;
        lengths.compressed[band] = static_cast<int>(x->ne[1]);
        x = convolution_module(model, graph, x, p + ".conv_modules." + std::to_string(band), depths[band]);
        bands.push_back(x);
    }
    skip = join(graph, std::move(bands), 1);
    capture(graph, "skip" + std::to_string(level), skip);
    auto* result = convolution(model, graph, skip, p + ".globalconv", 3, 3, 1, 1, 1, 1);
    capture(graph, "encoder" + std::to_string(level), result);
    return result;
}

ggml_tensor* recurrent(scnet_context& model, Graph& graph, ggml_tensor* input, const std::string& name) {
    auto* ctx = graph.context;
    const int hidden = static_cast<int>(input->ne[0]);
    ggml_tensor* sources[8]{};
    if (graph.cpu_only) {
        // Both directions consume the same input. Join output rows of the F32
        // weights so one matrix operation projects them together. The views
        // retain 8H row strides; the recurrent kernel consumes them directly,
        // without copying either [4H,B,T] half into a separate buffer.
        auto* forward = as_f32(graph, weight(model, name + ".weight_ih_l0"));
        auto* reverse = as_f32(graph, weight(model, name + ".weight_ih_l0_reverse"));
        auto* weights = scnet_tensor::concat(ctx, forward, reverse, 1, true);
        auto* projected = multiply(model, graph, weights, input);
        for (int direction = 0; direction < 2; ++direction)
            sources[direction] =
                ggml_view_3d(ctx, projected, 4 * hidden, input->ne[1], input->ne[2], projected->nb[1], projected->nb[2],
                             static_cast<size_t>(direction) * 4 * hidden * sizeof(float));
    }
    for (int direction = 0; direction < 2; ++direction) {
        const std::string suffix = direction ? "_reverse" : "";
        if (!graph.cpu_only)
            sources[direction] = multiply(model, graph, weight(model, name + ".weight_ih_l0" + suffix), input);
        const std::string recurrent_name = name + ".weight_hh_l0" + suffix;
        sources[2 + direction] = model.recurrent_weights.ctx ? model.recurrent_weights.tensors.at(recurrent_name)
                                                             : as_f32(graph, weight(model, recurrent_name));
        sources[4 + direction] = weight(model, name + ".bias_hh_l0" + suffix);
        sources[6 + direction] = weight(model, name + ".bias_ih_l0" + suffix);
    }
    auto node = std::make_unique<RecurrentNode>(static_cast<int>(input->ne[1]), hidden, model.params.n_threads,
                                                model.recurrent_mode, name);
    if (model.recurrent_mode == RecurrentMode::packed) {
        node->kernel = model.recurrent_kernel;
        for (int direction = 0; direction < 2; ++direction) {
            const std::string suffix = direction ? "_reverse" : "";
            node->packed[direction] = &model.recurrent_packed_weights.at(name + ".weight_hh_l0" + suffix);
        }
    }
    auto* result =
        ggml_custom_4d(ctx, GGML_TYPE_F32, 2 * hidden, input->ne[1], input->ne[2], 1, sources, 8, recurrent_compute,
                       model.recurrent_mode == RecurrentMode::cblas ? 1 : GGML_N_TASKS_MAX, node.get());
    ggml_set_name(result, "scnet_cpu_lstm");
    graph.recurrent_nodes.push_back(std::move(node));
    return result;
}

ggml_tensor* dual_path(scnet_context& model, Graph& graph, ggml_tensor* input, int layer) {
    auto* ctx = graph.context;
    const std::string p = "separation_net.dp_modules." + std::to_string(layer);
    for (int axis = 0; axis < 2; ++axis) {
        const std::string q = std::to_string(axis);
        auto* x = group_norm(model, graph, input, p + ".norm_layers." + q, false);
        // Frequency path [C,T,F]; time path [C,F,T]. ne1 is the batch
        // dimension and ne2 the sequential dimension in the recurrent node.
        x = contiguous(graph, axis == 0 ? ggml_permute(ctx, x, 1, 2, 0, 3) : ggml_permute(ctx, x, 2, 1, 0, 3));
        x = recurrent(model, graph, x, p + ".lstm_layers." + q);
        x = affine(model, graph, x, p + ".linear_layers." + q);
        x = contiguous(graph, axis == 0 ? ggml_permute(ctx, x, 2, 0, 1, 3) : ggml_permute(ctx, x, 2, 1, 0, 3));
        input = tensor_add(graph, input, x);
    }
    capture(graph, "dp" + std::to_string(layer), input);
    const bool inverse = layer % 2 != 0;
    const int size = static_cast<int>(inverse ? 2 * (input->ne[0] - 1) : input->ne[0]);
    auto node = std::make_unique<FourierNode>(size, inverse, model.params.n_threads);
    ggml_tensor* sources[] = {contiguous(graph, input)};
    auto* result = ggml_custom_4d(ctx, GGML_TYPE_F32, inverse ? size : size / 2 + 1, input->ne[1],
                                  inverse ? input->ne[2] / 2 : input->ne[2] * 2, 1, sources, 1, fourier_compute,
                                  GGML_N_TASKS_MAX, node.get());
    ggml_set_name(result, "scnet_cpu_fft");
    graph.fourier_nodes.push_back(std::move(node));
    capture(graph, "feature" + std::to_string(layer), result);
    return result;
}

ggml_tensor* upsample(scnet_context& model, Graph& graph, ggml_tensor* input, const std::string& name, int kernel,
                      int stride, int output_frequencies) {
    auto* ctx = graph.context;
    auto* weights = weight(model, name + ".weight");
    const int64_t times = input->ne[0], frequencies = input->ne[1], channels = weights->ne[1] / kernel;
    auto* x = contiguous(graph, ggml_permute(ctx, input, 1, 2, 0, 3)); // [C,T,F]
    x = multiply(model, graph, weights, x);                            // [K*Cout,T,F]
    x = ggml_reshape_4d(ctx, x, kernel, channels, times, frequencies);
    x = contiguous(graph, ggml_permute(ctx, x, 0, 2, 1, 3)); // [K,T,Cout,F]
    x = ggml_reshape_2d(ctx, x, kernel * times * channels, frequencies);
    x = ggml_col2im_1d(ctx, x, stride, static_cast<int>(times * channels), 0);
    const int64_t produced = (frequencies - 1) * stride + kernel;
    x = ggml_reshape_3d(ctx, x, produced, times, channels);
    x = contiguous(graph, ggml_permute(ctx, x, 1, 0, 2, 3));
    const int crop = static_cast<int>(std::abs(produced - output_frequencies) / 2);
    if (produced < output_frequencies || crop + output_frequencies > produced)
        throw std::runtime_error("SCNet transposed convolution cannot recover the original band");
    x = slice_frequency(ctx, x, crop, output_frequencies);
    return channel_bias(model, graph, x, name);
}

ggml_tensor* decoder(scnet_context& model, Graph& graph, ggml_tensor* input, ggml_tensor* skip,
                     const BandLengths& lengths, int level) {
    auto* ctx = graph.context;
    const std::string p = "decoder." + std::to_string(level);
    input = tensor_add(graph, input, skip);
    // The original repeats its channels before a dense convolution. Adding
    // the corresponding kernel halves removes that redundant activation copy.
    auto* weights = as_f32(graph, weight(model, p + ".0.conv.weight"));
    const int64_t half = weights->ne[0] / 2;
    auto* a = ggml_view_2d(ctx, weights, half, weights->ne[1], weights->nb[1], 0);
    auto* b =
        ggml_view_2d(ctx, weights, half, weights->ne[1], weights->nb[1], static_cast<size_t>(half) * sizeof(float));
    weights = tensor_add(graph, a, b);
    input = glu(graph, convolution(model, graph, input, weights, p + ".0.conv", 3, 3, 1, 1, 1, 1));
    const int kernels[] = {3, 4, 16}, strides[] = {1, 4, 16};
    std::vector<ggml_tensor*> bands;
    int start = 0;
    for (int band = 0; band < 3; ++band) {
        auto* x = slice_frequency(ctx, input, start, lengths.compressed[band]);
        start += lengths.compressed[band];
        bands.push_back(upsample(model, graph, x, p + ".1.convtrs." + std::to_string(band), kernels[band],
                                 strides[band], lengths.original[band]));
    }
    auto* result = join(graph, std::move(bands), 1);
    capture(graph, "decoder" + std::to_string(level), result);
    return result;
}

Graph& network_graph(scnet_context& model, int frames, bool want_capture) {
    const bool want_profile = flag("CRISPASR_SCNET_BENCH");
    const bool cpu_only = model.backend == model.cpu;
    for (size_t i = 0; i < model.graphs.size(); ++i)
        if (model.graphs[i]->frames == frames && model.graphs[i]->capture == want_capture &&
            model.graphs[i]->profile.enabled == want_profile) {
            auto cached = std::move(model.graphs[i]);
            model.graphs.erase(model.graphs.begin() + static_cast<std::ptrdiff_t>(i));
            model.graphs.insert(model.graphs.begin(), std::move(cached));
            if (want_profile && cpu_only)
                profile_graph_allocator(*model.graphs.front(), true, false,
                                        graph_buffer_bytes(model.graphs.front()->scheduler));
            return *model.graphs.front();
        }
    Timer timer("graph_build");
    // Reset before discarding the old tensor metadata: scheduler reset makes
    // its tensor pointers invalid. Only this current CPU graph is retained.
    if (cpu_only)
        model.graphs.clear();
    auto graph = std::make_unique<Graph>();
    graph->capture = want_capture;
    graph->cpu_only = cpu_only;
    graph->profile.enabled = want_profile;
    ggml_init_params params{ggml_tensor_overhead() * GRAPH_CAPACITY + ggml_graph_overhead_custom(GRAPH_CAPACITY, false),
                            nullptr, true};
    graph->context = ggml_init(params);
    if (!graph->context)
        throw std::runtime_error("SCNet graph metadata allocation failed");
    ggml_backend_t backends[] = {model.blas ? model.blas : model.backend, model.cpu};
    bool scheduler_created = false;
    if (cpu_only) {
        if (!model.cpu_scheduler) {
            model.cpu_scheduler =
                ggml_backend_sched_new(backends, nullptr, model.blas ? 2 : 1, GRAPH_CAPACITY, false, true);
            scheduler_created = true;
        }
        graph->scheduler = model.cpu_scheduler;
        graph->owns_scheduler = false;
    } else {
        graph->scheduler = ggml_backend_sched_new(backends, nullptr, 2, GRAPH_CAPACITY, false, true);
    }
    if (!graph->scheduler)
        throw std::runtime_error("SCNet graph scheduler allocation failed");
    graph->graph = ggml_new_graph_custom(graph->context, GRAPH_CAPACITY, false);
    graph->input = ggml_new_tensor_3d(graph->context, GGML_TYPE_F32, frames, FFT / 2 + 1, 4);
    ggml_set_input(graph->input);
    auto* x = graph->input;
    std::array<ggml_tensor*, 3> skips{};
    std::array<BandLengths, 3> lengths{};
    for (int i = 0; i < 3; ++i)
        x = encoder(model, *graph, x, i, skips[i], lengths[i]);
    for (int i = 0; i < 6; ++i)
        x = dual_path(model, *graph, x, i);
    for (int i = 0; i < 3; ++i)
        x = decoder(model, *graph, x, skips[2 - i], lengths[2 - i], i);
    graph->output = contiguous(*graph, x);
    graph->profile.label(graph->output, "output");
    ggml_set_output(graph->output);
    ggml_build_forward_expand(graph->graph, graph->output);
    // Preserve the existing device cache policy. CPU shape changes reuse the
    // model-owned allocator above and never retain stale allocated tensors.
    if (!cpu_only && model.graphs.size() == 2)
        model.graphs.pop_back();
    const size_t before_bytes = want_profile && cpu_only ? graph_buffer_bytes(graph->scheduler) : 0;
    if (!ggml_backend_sched_alloc_graph(graph->scheduler, graph->graph))
        throw std::runtime_error("SCNet graph buffer allocation failed");
    graph->frames = frames;
    if (want_profile && cpu_only)
        profile_graph_allocator(*graph, false, scheduler_created, before_bytes);
    model.graphs.insert(model.graphs.begin(), std::move(graph));
    return *model.graphs.front();
}

std::vector<float> planar(const std::vector<float>& interleaved) {
    const size_t samples = interleaved.size() / 2;
    std::vector<float> result(interleaved.size());
    for (size_t i = 0; i < samples; ++i) {
        result[i] = interleaved[2 * i];
        result[samples + i] = interleaved[2 * i + 1];
    }
    return result;
}

Sources forward(scnet_context& model, const float* pcm, int samples, Captures* captures = nullptr) {
    if (samples <= 0 || samples > 120 * RATE)
        throw std::runtime_error("SCNet raw forward length is outside the supported 120-second bound");
    int padding = HOP - samples % HOP;
    if (((samples + padding) / HOP) % 2 == 0)
        padding += HOP;
    const int padded = samples + padding, frames = padded / HOP + 1, bins = FFT / 2 + 1;
    if (padded <= FFT / 2) {
        // Too short for the original reflective STFT padding (a very short
        // file or a short final segment). Zero-extend to the minimum length,
        // run, and trim back, instead of aborting the whole separation
        // (#566 review). Longer inputs never reach this branch, so qualified
        // results are unchanged.
        const int extended = FFT / 2 + 1;
        std::vector<float> zero_padded(static_cast<size_t>(extended) * 2, 0.0f);
        std::copy(pcm, pcm + static_cast<size_t>(samples) * 2, zero_padded.begin());
        Sources out = forward(model, zero_padded.data(), extended, captures);
        for (auto& source : out)
            source.resize(static_cast<size_t>(samples) * 2);
        return out;
    }
    if (captures)
        put(captures, "raw_input_audio", {samples, 2, 1, 1},
            planar(std::vector<float>(pcm, pcm + size_t(samples) * 2)));
    std::vector<float> spectrum(static_cast<size_t>(frames) * bins * 4);
    {
        Timer timer("stft");
        core_parallel::for_each_chunk(2 * frames, model.params.n_threads, [&](int begin, int end) {
            std::vector<float> real(FFT), imag(FFT);
            scnet_fft::Plan::Scratch scratch;
            const float scale = 1.0f / std::sqrt(static_cast<float>(FFT));
            for (int row = begin; row < end; ++row) {
                const int channel = row / frames, frame = row % frames;
                for (int k = 0; k < FFT; ++k) {
                    int index = frame * HOP + k - FFT / 2;
                    if (index < 0)
                        index = -index;
                    else if (index >= padded)
                        index = 2 * padded - 2 - index;
                    real[k] = index < samples ? pcm[static_cast<size_t>(index) * 2 + channel] : 0.0f;
                    imag[k] = 0;
                }
                model.stft_plan.transform(real.data(), imag.data(), scratch);
                for (int frequency = 0; frequency < bins; ++frequency) {
                    spectrum[(static_cast<size_t>(channel * 2) * bins + frequency) * frames + frame] =
                        real[frequency] * scale;
                    spectrum[(static_cast<size_t>(channel * 2 + 1) * bins + frequency) * frames + frame] =
                        imag[frequency] * scale;
                }
            }
        });
    }
    if (captures)
        put(captures, "stft", {frames, bins, 4, 1}, spectrum);
    double sum = std::accumulate(spectrum.begin(), spectrum.end(), 0.0);
    const float mean = static_cast<float>(sum / spectrum.size());
    double variance = 0;
    for (float value : spectrum)
        variance += (static_cast<double>(value) - mean) * (static_cast<double>(value) - mean);
    const float stddev = static_cast<float>(std::sqrt(variance / (spectrum.size() - 1)));
    for (float& value : spectrum)
        value = (value - mean) / (stddev + 1e-5f);
    if (captures) {
        put(captures, "stft_stats", {2, 1, 1, 1}, {mean, stddev});
        put(captures, "normalized", {frames, bins, 4, 1}, spectrum);
    }
    Graph& graph = network_graph(model, frames, captures != nullptr);
    if (graph.profile.enabled)
        graph.profile.begin(graph.graph, graph.scheduler);
    {
        Timer timer("network");
        ggml_backend_tensor_set(graph.input, spectrum.data(), 0, spectrum.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(graph.scheduler, graph.graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("SCNet graph computation failed");
    }
    if (graph.profile.enabled)
        graph.profile.report(frames);
    if (captures)
        for (const auto& item : graph.captures) {
            const auto* t = item.second;
            put(captures, item.first, {t->ne[0], t->ne[1], t->ne[2], t->ne[3]}, read_tensor(t));
        }
    auto output = read_tensor(graph.output);
    Sources sources;
    for (auto& source : sources)
        source.assign(static_cast<size_t>(samples) * 2, 0);
    {
        Timer timer("istft");
        std::vector<int> counts(samples, 0);
        for (int frame = 0; frame < frames; ++frame) {
            const int start = frame * HOP - FFT / 2;
            for (int index = std::max(0, start); index < std::min(samples, start + FFT); ++index)
                ++counts[index];
        }
        core_parallel::for_each_chunk(8, model.params.n_threads, [&](int begin, int end) {
            std::vector<float> real(FFT), imag(FFT);
            scnet_fft::Plan::Scratch scratch;
            const float scale = 1.0f / std::sqrt(static_cast<float>(FFT));
            for (int row = begin; row < end; ++row) {
                float* waveform = sources[row / 2].data();
                const int channel = row % 2;
                for (int frame = 0; frame < frames; ++frame) {
                    for (int k = 0; k < bins; ++k) {
                        real[k] = output[(static_cast<size_t>(row * 2) * bins + k) * frames + frame] * stddev + mean;
                        imag[k] =
                            k == 0 || k == FFT / 2
                                ? 0.0f
                                : -(output[(static_cast<size_t>(row * 2 + 1) * bins + k) * frames + frame] * stddev +
                                    mean);
                    }
                    for (int k = bins; k < FFT; ++k) {
                        real[k] = real[FFT - k];
                        imag[k] = -imag[FFT - k];
                    }
                    model.stft_plan.transform(real.data(), imag.data(), scratch);
                    const int start = frame * HOP - FFT / 2;
                    for (int index = std::max(0, start); index < std::min(samples, start + FFT); ++index)
                        waveform[static_cast<size_t>(index) * 2 + channel] += real[index - start] * scale;
                }
                for (int index = 0; index < samples; ++index)
                    waveform[static_cast<size_t>(index) * 2 + channel] /= counts[index];
            }
        });
    }
    if (captures)
        for (int source = 0; source < STEM_COUNT; ++source)
            put(captures, std::string("raw_output_") + STEMS[source], {samples, 2, 1, 1}, planar(sources[source]));
    return sources;
}

Sources separate(scnet_context& model, const std::vector<float>& input, Captures* captures = nullptr) {
    Timer timer("separate");
    const int samples = static_cast<int>(input.size() / 2);
    if (captures)
        put(captures, "input_audio", {samples, 2, 1, 1}, planar(input));
    Sources sources;
    if (model.params.no_segment) {
        put(captures, "waveform_stats", {2, 1, 1, 1}, {0.0f, 1.0f});
        sources = forward(model, input.data(), samples, captures);
    } else {
        std::vector<float> normalized(input.size());
        std::vector<float> mono(samples);
        for (int i = 0; i < samples; ++i)
            mono[i] = (input[static_cast<size_t>(i) * 2] + input[static_cast<size_t>(i) * 2 + 1]) * 0.5f;
        const float mean = static_cast<float>(std::accumulate(mono.begin(), mono.end(), 0.0) / samples);
        double variance = 0;
        for (float value : mono)
            variance += (static_cast<double>(value) - mean) * (static_cast<double>(value) - mean);
        const float stddev = samples > 1 ? static_cast<float>(std::sqrt(variance / (samples - 1))) : 0.0f;
        put(captures, "waveform_stats", {2, 1, 1, 1}, {mean, stddev});
        for (auto& source : sources)
            source.assign(input.size(), 0);
        if (!(stddev > 0) || !std::isfinite(stddev)) {
            // The original divides by zero for a zero-variance mono mixture.
            // Silence has an unambiguous zero result; other degenerate inputs
            // fail explicitly rather than manufacturing separated stems.
            if (!std::all_of(input.begin(), input.end(), [](float value) { return value == 0; }))
                throw std::runtime_error("SCNet original waveform normalization has zero mono standard deviation");
            return sources;
        }
        for (size_t i = 0; i < input.size(); ++i)
            normalized[i] = (input[i] - mean) / stddev;
        const int segment = (model.params.segment_seconds > 0 ? model.params.segment_seconds : 20) * RATE;
        const int stride = segment / 2, max_shift = RATE / 2;
        std::vector<float> counts(samples, 0);
        scnet_detail::PythonRandom random(model.params.shift_seed);
        bool captured = false;
        for (int64_t offset = 0; offset < samples; offset += stride) {
            const int count = std::min<int64_t>(segment, samples - offset);
            Sources chunk;
            for (auto& source : chunk)
                source.assign(static_cast<size_t>(count) * 2, 0);
            const int shifts = std::max(1, model.params.shifts);
            for (int shift = 0; shift < shifts; ++shift) {
                const int prefix = model.params.shifts > 0 ? max_shift - random.randint(max_shift) : 0;
                const int length = count + prefix;
                std::vector<float> block(static_cast<size_t>(length) * 2, 0);
                // TensorChunk.padded() uses the preceding mixture, including
                // context before an overlapping chunk. It does not zero-pad
                // every chunk independently or reuse one shift for all chunks.
                const int64_t begin = offset - prefix;
                for (int i = 0; i < length; ++i)
                    if (begin + i >= 0) {
                        block[static_cast<size_t>(i) * 2] = normalized[static_cast<size_t>(begin + i) * 2];
                        block[static_cast<size_t>(i) * 2 + 1] = normalized[static_cast<size_t>(begin + i) * 2 + 1];
                    }
                Sources output = forward(model, block.data(), length, captures && !captured ? captures : nullptr);
                captured = true;
                for (int source = 0; source < STEM_COUNT; ++source)
                    for (int i = 0; i < count; ++i)
                        for (int channel = 0; channel < 2; ++channel)
                            chunk[source][static_cast<size_t>(i) * 2 + channel] +=
                                output[source][static_cast<size_t>(prefix + i) * 2 + channel];
            }
            for (int i = 0; i < count; ++i) {
                const float taper = static_cast<float>(i < segment / 2 ? i + 1 : segment - i) / (segment / 2);
                counts[static_cast<size_t>(offset) + i] += taper;
                for (int source = 0; source < STEM_COUNT; ++source)
                    for (int channel = 0; channel < 2; ++channel)
                        sources[source][(static_cast<size_t>(offset) + i) * 2 + channel] +=
                            (chunk[source][static_cast<size_t>(i) * 2 + channel] / shifts) * taper;
            }
        }
        for (int i = 0; i < samples; ++i) {
            if (!(counts[i] > 0))
                throw std::runtime_error("SCNet overlap-add left an uncovered output frame");
            for (int source = 0; source < STEM_COUNT; ++source)
                for (int channel = 0; channel < 2; ++channel) {
                    float& value = sources[source][static_cast<size_t>(i) * 2 + channel];
                    value = (value / counts[i]) * stddev + mean;
                }
        }
    }
    for (int source = 0; source < STEM_COUNT; ++source) {
        if (!std::all_of(sources[source].begin(), sources[source].end(),
                         [](float value) { return std::isfinite(value); }))
            throw std::runtime_error("SCNet produced nonfinite separated audio");
        if (captures)
            put(captures, std::string("output_") + STEMS[source], {samples, 2, 1, 1}, planar(sources[source]));
    }
    return sources;
}

std::vector<std::string> required_stages() {
    std::vector<std::string> names = {"input_audio", "waveform_stats", "raw_input_audio",
                                      "stft",        "stft_stats",     "normalized"};
    for (int i = 0; i < 3; ++i) {
        names.push_back("skip" + std::to_string(i));
        names.push_back("encoder" + std::to_string(i));
        names.push_back("decoder" + std::to_string(i));
    }
    for (int i = 0; i < 6; ++i) {
        names.push_back("dp" + std::to_string(i));
        names.push_back("feature" + std::to_string(i));
    }
    for (const char* stem : STEMS) {
        names.push_back(std::string("raw_output_") + stem);
        names.push_back(std::string("output_") + stem);
    }
    return names;
}

void save_captures(const Captures& captures, const char* path) {
    const auto required = required_stages();
    gguf_context* file = gguf_init_empty();
    ggml_init_params params{ggml_tensor_overhead() * (required.size() + 1), nullptr, true};
    ggml_context* tensors = ggml_init(params);
    if (!file || !tensors) {
        if (file)
            gguf_free(file);
        if (tensors)
            ggml_free(tensors);
        throw std::runtime_error("SCNet capture metadata allocation failed");
    }
    bool complete = true;
    gguf_set_val_str(file, "general.architecture", "scnet-capture");
    gguf_set_val_str(file, "scnet.reference.source_revision", REVISION);
    for (const auto& name : required) {
        const auto found = captures.find(name);
        if (found == captures.end()) {
            complete = false;
            break;
        }
        const auto& capture = found->second;
        auto* tensor = ggml_new_tensor(tensors, GGML_TYPE_F32, 4, capture.shape.data());
        ggml_set_name(tensor, name.c_str());
        tensor->data = const_cast<float*>(capture.values.data());
        gguf_add_tensor(file, tensor);
    }
    const bool written = complete && gguf_write_to_file(file, path, false);
    gguf_free(file);
    ggml_free(tensors);
    if (!written)
        throw std::runtime_error("SCNet capture is incomplete or could not be written");
}
} // namespace

scnet_params scnet_default_params(void) {
    scnet_params params{};
    params.n_threads = 0;
    params.use_gpu = true;
    params.gpu_device = 0;
    params.segment_seconds = 20;
    params.no_segment = false;
    params.shifts = 1;
    params.shift_seed = 0;
    return params;
}

scnet_context* scnet_init_from_file(const char* path, scnet_params params) {
    if (!path || !*path)
        return nullptr;
    std::unique_ptr<scnet_context, decltype(&scnet_free)> model(nullptr, scnet_free);
    try {
        model.reset(new scnet_context);
        std::unique_ptr<gguf_context, decltype(&core_gguf::free_metadata)> meta(core_gguf::open_metadata(path),
                                                                                core_gguf::free_metadata);
        if (!meta)
            throw std::runtime_error("cannot read SCNet metadata");
        const bool valid = model->hp.read(meta.get());
        meta.reset();
        if (!valid)
            throw std::runtime_error("unsupported SCNet architecture, tensor layout, or missing checkpoint metadata");
        if (params.gpu_device < 0 || params.n_threads > 256 || params.segment_seconds > 119 || params.shifts < 0 ||
            params.shifts > 16)
            throw std::runtime_error("invalid SCNet runtime parameters");
        params.n_threads = params.n_threads > 0 ? params.n_threads : std::min(8, core_parallel::resolve_threads(8, 0));
        model->params = params;
        ggml_backend_load_all();
        model->cpu = core_cpu_backend::init();
        if (!model->cpu)
            throw std::runtime_error("SCNet CPU graph backend is unavailable");
        core_cpu_backend::set_n_threads(model->cpu, params.n_threads);
        if (params.use_gpu) {
            crispasr_set_gpu_device_pref(params.gpu_device);
            model->backend = crispasr_init_gpu_backend();
            if (model->backend && core_cpu_backend::is_cpu(model->backend)) {
                ggml_backend_free(model->backend);
                model->backend = nullptr;
            }
        }
        if (!model->backend)
            model->backend = model->cpu;
        if (model->backend == model->cpu && flag("CRISPASR_SCNET_BLAS", true)) {
            model->blas = ggml_backend_init_by_name("BLAS", nullptr);
            if (model->blas) {
                const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(model->blas));
                auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
                if (set_threads)
                    set_threads(model->blas, params.n_threads);
            }
        }
#if defined(HAVE_ACCELERATE) || defined(HAVE_BLAS)
        model->recurrent_blas = !flag("CRISPASR_SCNET_SCALAR_LSTM");
        model->recurrent_mode = model->recurrent_blas ? RecurrentMode::cblas : RecurrentMode::portable;
#endif
        const bool loaded =
            model->backend == model->cpu
                ? core_gguf::load_weights(path, model->cpu, "scnet", model->weights)
                : core_gguf::load_weights_fit(path, model->backend, model->cpu, "scnet", model->weights);
        if (!loaded)
            throw std::runtime_error("SCNet weight loading failed");
        validate_weights(*model);
        materialize_recurrent_weights(*model);
        model->backend_description =
            std::string(ggml_backend_name(model->backend)) + (model->blas ? "+BLAS" : "") + "/CPU-LSTM-FFT";
        model->compute_description = std::string("F32 activations/accumulation; CPU recurrent/FFT; gates=") +
                                     scnet_activation::selected_kernel().name +
                                     "; fft=" + model->stft_plan.kernel_name() + "; recurrent=" +
                                     (model->recurrent_mode == RecurrentMode::packed  ? model->recurrent_kernel.name
                                      : model->recurrent_mode == RecurrentMode::cblas ? "CBLAS"
                                                                                      : "portable");
        std::fprintf(stderr, "scnet: backend=%s threads=%d precision=%s recurrence=%s\n",
                     model->backend_description.c_str(), params.n_threads,
                     model->quant_q4   ? "Q4_K mixed"
                     : model->quant_q8 ? "Q8_0 mixed"
                                       : "F16/F32",
                     model->recurrent_mode == RecurrentMode::packed  ? model->recurrent_kernel.name
                     : model->recurrent_mode == RecurrentMode::cblas ? "native-CBLAS"
                                                                     : "native-portable");
        return model.release();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "scnet: %s\n", error.what());
        return nullptr;
    }
}

void scnet_free(scnet_context* model) {
    if (!model)
        return;
    model->graphs.clear();
    if (model->cpu_scheduler)
        ggml_backend_sched_free(model->cpu_scheduler);
    // Graphs and the shared scheduler must release their tensor references
    // before either the derived CPU leaves or original GGUF weights are freed.
    model->recurrent_packed_weights.clear();
    core_gguf::free_weights(model->recurrent_weights);
    core_gguf::free_weights(model->weights);
    if (model->blas)
        ggml_backend_free(model->blas);
    if (model->backend && model->backend != model->cpu)
        ggml_backend_free(model->backend);
    if (model->cpu)
        ggml_backend_free(model->cpu);
    delete model;
}

scnet_result* scnet_separate(scnet_context* model, const float* pcm, int n_samples, int in_channels) {
    if (!model || !pcm || n_samples <= 0 || n_samples > std::numeric_limits<int>::max() / 8 ||
        (in_channels != 1 && in_channels != 2))
        return nullptr;
    try {
        std::vector<float> input(static_cast<size_t>(n_samples) * 2);
        for (int i = 0; i < n_samples; ++i)
            for (int channel = 0; channel < 2; ++channel) {
                const float value = pcm[static_cast<size_t>(i) * in_channels + (in_channels == 1 ? 0 : channel)];
                if (!std::isfinite(value))
                    throw std::runtime_error("nonfinite SCNet input waveform");
                input[static_cast<size_t>(i) * 2 + channel] = value;
            }
        const auto sources = separate(*model, input);
        std::unique_ptr<scnet_result, decltype(&scnet_result_free)> result(new scnet_result{}, scnet_result_free);
        result->n_sources = STEM_COUNT;
        result->n_channels = 2;
        result->n_samples = n_samples;
        result->sample_rate = RATE;
        result->sources = new float* [STEM_COUNT] {};
        result->source_names = new const char* [STEM_COUNT] {};
        for (int source = 0; source < STEM_COUNT; ++source) {
            result->sources[source] = new float[sources[source].size()];
            std::copy(sources[source].begin(), sources[source].end(), result->sources[source]);
            result->source_names[source] = STEMS[source];
        }
        return result.release();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "scnet: separation failed: %s\n", error.what());
        return nullptr;
    }
}

void scnet_result_free(scnet_result* result) {
    if (!result)
        return;
    if (result->sources)
        for (int source = 0; source < result->n_sources; ++source)
            delete[] result->sources[source];
    delete[] result->sources;
    delete[] result->source_names;
    delete result;
}

int scnet_sample_rate(const scnet_context* model) {
    return model ? RATE : 0;
}
int scnet_n_sources(const scnet_context* model) {
    return model ? STEM_COUNT : 0;
}
const char* scnet_source_name(const scnet_context* model, int index) {
    return model && index >= 0 && index < STEM_COUNT ? STEMS[index] : nullptr;
}
const char* scnet_backend_name(const scnet_context* model) {
    return model ? model->backend_description.c_str() : nullptr;
}
const char* scnet_compute_precision(const scnet_context* model) {
    return model ? model->compute_description.c_str() : nullptr;
}

static int diff_impl(const char* model_gguf, const char* ref_gguf, const char* audio_wav, int verbosity) {
    (void)audio_wav; // Exact planar audio bytes are embedded in the independent reference.
    constexpr const char* row_policy = "cosine_or_binary16_reference_floor_v1";
    constexpr double row_cosine_threshold = 0.999;
    constexpr double row_floor_epsilon = 1.0 / 1024.0;
    const auto required = required_stages();
    const std::set<std::string> required_names(required.begin(), required.end());
    const auto require_key = [](gguf_context* meta, const char* name, gguf_type type) {
        const int64_t key = gguf_find_key(meta, name);
        if (key < 0 || gguf_get_kv_type(meta, key) != type)
            throw std::runtime_error(std::string("missing or invalid reference/model metadata: ") + name);
        return key;
    };
    const auto require_string = [&](gguf_context* meta, const char* name) {
        return std::string(gguf_get_val_str(meta, require_key(meta, name, GGUF_TYPE_STRING)));
    };
    const auto require_unsigned = [](gguf_context* meta, const char* name) -> uint64_t {
        const int64_t key = gguf_find_key(meta, name);
        if (key >= 0) {
            if (gguf_get_kv_type(meta, key) == GGUF_TYPE_UINT32)
                return gguf_get_val_u32(meta, key);
            if (gguf_get_kv_type(meta, key) == GGUF_TYPE_UINT64)
                return gguf_get_val_u64(meta, key);
        }
        throw std::runtime_error(std::string("missing or invalid reference metadata: ") + name);
    };
    const auto valid_sha256 = [](const std::string& value) {
        return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
                   return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
               });
    };
    auto params = scnet_default_params();
    params.use_gpu = flag("CRISPASR_SCNET_GPU");
    if (const char* threads = std::getenv("CRISPASR_SCNET_THREADS"))
        params.n_threads = std::atoi(threads);
    std::unique_ptr<gguf_context, decltype(&core_gguf::free_metadata)> metadata(core_gguf::open_metadata(ref_gguf),
                                                                                core_gguf::free_metadata);
    gguf_context* meta = metadata.get();
    if (!meta)
        return 2;
    const std::string revision = require_string(meta, "scnet.reference.source_revision");
    const std::string checkpoint = require_string(meta, "scnet.reference.checkpoint_sha256");
    const std::string mode = require_string(meta, "scnet.reference.mode");
    const uint64_t segment = require_unsigned(meta, "scnet.reference.segment_seconds");
    const uint64_t shifts = require_unsigned(meta, "scnet.reference.shifts");
    const uint64_t seed = require_unsigned(meta, "scnet.reference.shift_seed");
    const uint64_t sample_rate = require_unsigned(meta, "scnet.reference.sample_rate");
    const uint64_t stage_count = require_unsigned(meta, "scnet.reference.required_stage_count");
    const float overlap = gguf_get_val_f32(meta, require_key(meta, "scnet.reference.overlap", GGUF_TYPE_FLOAT32));
    if (revision != REVISION || !valid_sha256(checkpoint) || (mode != "pipeline" && mode != "raw") || segment < 1 ||
        segment > 119 || shifts > 16 || overlap != 0.5f || sample_rate != RATE || stage_count != 35 ||
        required.size() != 35)
        throw std::runtime_error("incompatible original-reference provenance/profile");
    if (gguf_get_n_tensors(meta) != static_cast<int64_t>(required.size()))
        throw std::runtime_error("reference must contain exactly all 35 required SCNet stages");
    std::set<std::string> reference_names;
    for (int64_t i = 0; i < gguf_get_n_tensors(meta); ++i)
        reference_names.insert(gguf_get_tensor_name(meta, i));
    if (reference_names != required_names)
        throw std::runtime_error("reference must contain exactly all 35 required SCNet stages");
    params.no_segment = mode == "raw";
    params.segment_seconds = static_cast<int>(segment);
    params.shifts = static_cast<int>(shifts);
    params.shift_seed = seed;
    metadata.reset();
    metadata.reset(core_gguf::open_metadata(model_gguf));
    if (!metadata)
        throw std::runtime_error("cannot read model provenance metadata");
    if (require_string(metadata.get(), "general.source.revision") != revision ||
        require_string(metadata.get(), "scnet.checkpoint.sha256") != checkpoint)
        throw std::runtime_error("reference source revision/checkpoint SHA256 does not match the model");
    metadata.reset();
    std::unique_ptr<scnet_context, decltype(&scnet_free)> model(scnet_init_from_file(model_gguf, params), scnet_free);
    if (!model)
        return 2;
    core_gguf::WeightLoad reference;
    if (!core_gguf::load_weights(ref_gguf, model->cpu, "scnet-reference", reference))
        return 2;
    int failures = 0;
    try {
        const auto input = reference.tensors.find("input_audio");
        if (input == reference.tensors.end() || input->second->ne[1] != 2 || input->second->ne[2] != 1 ||
            input->second->ne[3] != 1 || input->second->ne[0] <= 0 || input->second->ne[0] > 120 * RATE)
            throw std::runtime_error("reference needs bounded planar stereo input_audio");
        auto audio = read_tensor(input->second);
        if (!std::all_of(audio.begin(), audio.end(), [](float value) { return std::isfinite(value); }))
            throw std::runtime_error("nonfinite original reference input");
        const int samples = static_cast<int>(input->second->ne[0]);
        std::vector<float> interleaved(audio.size());
        for (int i = 0; i < samples; ++i) {
            interleaved[static_cast<size_t>(i) * 2] = audio[i];
            interleaved[static_cast<size_t>(i) * 2 + 1] = audio[static_cast<size_t>(samples) + i];
        }
        Captures actual;
        separate(*model, interleaved, &actual);
        if (actual.size() != required.size() || !std::all_of(actual.begin(), actual.end(), [&](const auto& stage) {
                return required_names.count(stage.first) == 1;
            }))
            throw std::runtime_error("native capture must contain exactly all 35 required SCNet stages");
        const double threshold = model->quant_q4 ? 0.94 : model->quant_q8 ? 0.98 : 0.999;
        const double norm_min = model->quant_q4 ? 0.8 : model->quant_q8 ? 0.95 : 0.99;
        const double norm_max = model->quant_q4 ? 1.2 : model->quant_q8 ? 1.05 : 1.01;
        const double relative_max = model->quant_q4 ? 0.40 : model->quant_q8 ? 0.20 : 0.05;
        const std::set<std::string> analytic_stages = {"input_audio", "raw_input_audio", "waveform_stats",
                                                       "stft",        "stft_stats",      "normalized"};
        std::printf("scnet row policy: %s epsilon=0.0009765625\n", row_policy);
        for (const auto& name : required) {
            const auto truth = reference.tensors.find(name);
            const auto observed = actual.find(name);
            bool valid_shape = truth != reference.tensors.end() && observed != actual.end();
            if (valid_shape)
                for (int d = 0; d < 4; ++d)
                    valid_shape &= truth->second->ne[d] == observed->second.shape[d];
            if (!valid_shape) {
                std::printf("[FAIL] %s missing required stage or shape mismatch\n", name.c_str());
                ++failures;
                continue;
            }
            const auto expected = read_tensor(truth->second);
            const auto& observed_values = observed->second.values;
            const size_t row_width = static_cast<size_t>(truth->second->ne[0]);
            if (expected.empty() || expected.size() != observed_values.size() || row_width == 0 ||
                expected.size() % row_width != 0) {
                std::printf("[FAIL] %s malformed capture size\n", name.c_str());
                ++failures;
                continue;
            }
            double dot = 0, aa = 0, bb = 0, difference = 0, maximum = 0;
            double minimum_cosine = 1;
            double reference_peak = 0;
            size_t nonfinite = 0, nonfinite_rows = 0, failing_rows = 0, worst_row = 0;
            double worst_aa = 0, worst_bb = 0, worst_difference = 0, worst_maximum = 0;
            // Retain errors only for raw cosine failures until the whole-stage
            // reference scale is known. The original global metrics and raw
            // row cosines remain unchanged and are always reported.
            struct RowError {
                double rms, peak;
                bool finite;
            };
            std::vector<RowError> cosine_failures;
            for (size_t row = 0; row < expected.size(); row += row_width) {
                double row_dot = 0, row_aa = 0, row_bb = 0;
                double row_difference = 0, row_maximum = 0;
                const size_t previous_nonfinite = nonfinite;
                for (size_t i = row; i < row + row_width; ++i) {
                    const double a = observed_values[i], b = expected[i];
                    if (!std::isfinite(a) || !std::isfinite(b)) {
                        ++nonfinite;
                        continue;
                    }
                    row_dot += a * b;
                    row_aa += a * a;
                    row_bb += b * b;
                    difference += (a - b) * (a - b);
                    maximum = std::max(maximum, std::abs(a - b));
                    reference_peak = std::max(reference_peak, std::abs(b));
                    row_difference += (a - b) * (a - b);
                    row_maximum = std::max(row_maximum, std::abs(a - b));
                }
                dot += row_dot;
                aa += row_aa;
                bb += row_bb;
                const double row_cosine = row_aa > 0 && row_bb > 0 ? row_dot / std::sqrt(row_aa * row_bb)
                                          : row_aa == row_bb       ? 1.0
                                                                   : 0.0;
                if (row == 0 || row_cosine < minimum_cosine) {
                    worst_row = row / row_width;
                    worst_aa = row_aa;
                    worst_bb = row_bb;
                    worst_difference = row_difference;
                    worst_maximum = row_maximum;
                }
                failing_rows += row_cosine <= row_cosine_threshold;
                if (row_cosine <= row_cosine_threshold)
                    cosine_failures.push_back(
                        {std::sqrt(row_difference / row_width), row_maximum, nonfinite == previous_nonfinite});
                nonfinite_rows += nonfinite != previous_nonfinite;
                minimum_cosine = std::min(minimum_cosine, row_cosine);
            }
            const double cosine = aa > 0 && bb > 0 ? dot / std::sqrt(aa * bb) : aa == bb ? 1.0 : 0.0;
            const double ratio = bb > 0 ? std::sqrt(aa / bb) : aa == 0 ? 1 : std::numeric_limits<double>::infinity();
            const double relative = bb > 0            ? std::sqrt(difference / bb)
                                    : difference == 0 ? 0
                                                      : std::numeric_limits<double>::infinity();
            const double reference_rms = std::sqrt(bb / expected.size());
            const double row_rms_error_limit = row_floor_epsilon * reference_rms;
            const double row_max_abs_error_limit = row_floor_epsilon * reference_peak;
            const size_t row_rescued_count =
                std::count_if(cosine_failures.begin(), cosine_failures.end(), [&](const RowError& row) {
                    return row.finite && row.rms <= row_rms_error_limit && row.peak <= row_max_abs_error_limit;
                });
            const size_t row_failed_count = failing_rows - row_rescued_count;
            const bool require_rows = !model->quant_q4 && !model->quant_q8 && analytic_stages.count(name) == 0;
            const bool passed = nonfinite == 0 && std::isfinite(cosine) && std::isfinite(ratio) &&
                                std::isfinite(relative) && cosine > threshold && ratio >= norm_min &&
                                ratio <= norm_max && relative <= relative_max &&
                                (!require_rows || row_failed_count == 0);
            failures += !passed;
            if (verbosity > 0 || !passed) {
                const size_t ne1 = static_cast<size_t>(truth->second->ne[1]);
                const size_t ne2 = static_cast<size_t>(truth->second->ne[2]);
                const size_t ne3 = static_cast<size_t>(truth->second->ne[3]);
                // Rows traverse ne0 contiguously. Learned stages are [T,F,C,1]
                // in ggml, so i1 is frequency and i2 is channel; audio [N,2,1,1]
                // uses i1 for channel. Raw cosine failures stay visible even
                // when both reference-scaled binary16 error bounds pass.
                std::printf("[%s] %s cos_global=%.9f norm_ratio=%.9f relative_l2=%.9g max_abs=%.9g nonfinite=%zu n=%zu "
                            "cos_min=%.9f rows=%zu row_width=%zu ne1=%zu ne2=%zu ne3=%zu row_cosine_gated=%d "
                            "rows_failing_cosine=%zu nonfinite_rows=%zu worst_row=%zu worst_i1=%zu worst_i2=%zu "
                            "worst_i3=%zu worst_actual_norm=%.9g worst_reference_norm=%.9g worst_rms_error=%.9g "
                            "worst_max_abs=%.9g row_policy=%s row_cosine_threshold=0.999 "
                            "row_floor_epsilon=0.0009765625 reference_rms=%.17g "
                            "reference_peak=%.17g row_rms_error_limit=%.17g row_max_abs_error_limit=%.17g "
                            "row_rescued_count=%zu row_failed_count=%zu\n",
                            passed ? "PASS" : "FAIL", name.c_str(), cosine, ratio, relative, maximum, nonfinite,
                            expected.size(), minimum_cosine, expected.size() / row_width, row_width, ne1, ne2, ne3,
                            int(require_rows), failing_rows, nonfinite_rows, worst_row, worst_row % ne1,
                            (worst_row / ne1) % ne2, worst_row / (ne1 * ne2), std::sqrt(worst_aa), std::sqrt(worst_bb),
                            std::sqrt(worst_difference / row_width), worst_maximum, row_policy, reference_rms,
                            reference_peak, row_rms_error_limit, row_max_abs_error_limit, row_rescued_count,
                            row_failed_count);
                if (verbosity > 1 && !passed && require_rows && minimum_cosine <= threshold && row_width <= 32) {
                    std::printf("[ROW] %s worst_row=%zu actual=[", name.c_str(), worst_row);
                    for (size_t i = 0; i < row_width; ++i)
                        std::printf("%s%.9g", i ? "," : "", observed_values[worst_row * row_width + i]);
                    std::printf("] reference=[");
                    for (size_t i = 0; i < row_width; ++i)
                        std::printf("%s%.9g", i ? "," : "", expected[worst_row * row_width + i]);
                    std::printf("]\n");
                }
            }
        }
        if (const char* path = std::getenv("CRISPASR_SCNET_CAPTURE_GGUF"))
            if (*path)
                save_captures(actual, path);
        std::printf("scnet diff: %zu required stages, %d failed, strict cosine > %.3f\n", required.size(), failures,
                    threshold);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "scnet diff: %s\n", error.what());
        ++failures;
    }
    core_gguf::free_weights(reference);
    return failures ? 1 : 0;
}

int scnet_diff(const char* model_gguf, const char* ref_gguf, const char* audio_wav, int verbosity) {
    try {
        return diff_impl(model_gguf, ref_gguf, audio_wav, verbosity);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "scnet diff: %s\n", error.what());
        return 2;
    }
}
