// Model-free regression for the persistent SCNet scheduler's CPU allocator.
// Two equal buffer types reproduce CPU+BLAS's shared allocation slots.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Allocate = ggml_backend_buffer_t (*)(ggml_backend_buffer_type_t, size_t);
using Free = void (*)(ggml_backend_buffer_t);
Allocate original_allocate = nullptr;
Free original_free = nullptr;
size_t allocation_calls = 0, allocation_failures = 0, frees = 0;
bool fail_next_allocation = false;
std::set<ggml_backend_buffer_t> live_buffers;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void tracked_free(ggml_backend_buffer_t buffer) {
    if (live_buffers.erase(buffer) != 1) {
        std::fprintf(stderr, "allocator control: duplicate or untracked buffer free\n");
        std::abort();
    }
    ++frees;
    original_free(buffer);
}

ggml_backend_buffer_t tracked_allocate(ggml_backend_buffer_type_t type, size_t bytes) {
    ++allocation_calls;
    if (fail_next_allocation) {
        fail_next_allocation = false;
        ++allocation_failures;
        return nullptr;
    }
    auto* buffer = original_allocate(type, bytes);
    if (buffer) {
        if (!original_free)
            original_free = buffer->iface.free_buffer;
        if (!original_free || buffer->iface.free_buffer != original_free || !live_buffers.insert(buffer).second) {
            std::fprintf(stderr, "allocator control: incompatible CPU buffer callback\n");
            std::abort();
        }
        buffer->iface.free_buffer = tracked_free;
    }
    return buffer;
}

struct AllocationHook {
    ggml_backend_buffer_type_t type = ggml_backend_cpu_buffer_type();
    AllocationHook() {
        original_allocate = type->iface.alloc_buffer;
        type->iface.alloc_buffer = tracked_allocate;
    }
    ~AllocationHook() { type->iface.alloc_buffer = original_allocate; }
};

struct Graph {
    ggml_context* context = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor *a = nullptr, *b = nullptr, *output = nullptr;
    int elements;
    explicit Graph(int count) : elements(count) {
        ggml_init_params params{16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true};
        context = ggml_init(params);
        require(context != nullptr, "graph metadata allocation failed");
        graph = ggml_new_graph_custom(context, 16, false);
        a = ggml_new_tensor_1d(context, GGML_TYPE_F32, count);
        b = ggml_new_tensor_1d(context, GGML_TYPE_F32, count);
        ggml_set_input(a);
        ggml_set_input(b);
        output = ggml_add(context, a, b);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
    }
    ~Graph() { ggml_free(context); }
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
};

struct Allocator {
    ggml_gallocr_t value = nullptr;
    int slots;
    explicit Allocator(int count) : slots(count) {
        ggml_backend_buffer_type_t types[] = {ggml_backend_cpu_buffer_type(), ggml_backend_cpu_buffer_type()};
        value = ggml_gallocr_new_n(types, count);
        require(value != nullptr, "graph allocator creation failed");
    }
    ~Allocator() { ggml_gallocr_free(value); }
    bool reserve(Graph& graph) { return ggml_gallocr_reserve_n(value, graph.graph, nullptr, nullptr); }
    void check_shared() const {
        require(ggml_gallocr_get_buffer_size(value, 0) > 0, "primary CPU buffer is missing");
        if (slots == 2)
            require(ggml_gallocr_get_buffer_size(value, 1) == 0, "shared CPU capacity was counted twice");
    }
    void check_empty() const {
        for (int i = 0; i < slots; ++i)
            require(ggml_gallocr_get_buffer_size(value, i) == 0, "failed allocation retained a stale buffer alias");
    }
    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;
};

void compute(Graph& graph, ggml_backend_t cpu) {
    std::vector<float> a(graph.elements), b(graph.elements), output(graph.elements);
    for (int i = 0; i < graph.elements; ++i) {
        a[i] = static_cast<float>(i) * 0.25f;
        b[i] = 1.0f;
    }
    ggml_backend_tensor_set(graph.a, a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_set(graph.b, b.data(), 0, b.size() * sizeof(float));
    require(ggml_backend_graph_compute(cpu, graph.graph) == GGML_STATUS_SUCCESS, "recovered graph compute failed");
    ggml_backend_tensor_get(graph.output, output.data(), 0, output.size() * sizeof(float));
    for (int i = 0; i < graph.elements; ++i)
        require(output[i] == a[i] + b[i], "recovered graph output is incorrect");
}

void fail_growth(Allocator& allocator, Graph& growth) {
    const size_t previous_failures = allocation_failures;
    fail_next_allocation = true;
    require(!allocator.reserve(growth), "forced allocation failure unexpectedly succeeded");
    require(!fail_next_allocation && allocation_failures == previous_failures + 1,
            "growth did not reach the forced allocation callback");
    allocator.check_empty();
}

void initial_failure(int slots) {
    Allocator allocator(slots);
    Graph graph(64);
    fail_growth(allocator, graph);
}

void growth_teardown(int slots, ggml_backend_t cpu) {
    Allocator allocator(slots);
    Graph initial(64);
    require(allocator.reserve(initial) && ggml_gallocr_alloc_graph(allocator.value, initial.graph),
            "initial graph allocation failed");
    allocator.check_shared();
    compute(initial, cpu);
    Graph growth(65536);
    fail_growth(allocator, growth);
    // Destruction after this return must free neither a stale alias nor a null
    // buffer through a previously committed allocation plan.
}

void growth_retry(int slots, int retry_elements, ggml_backend_t cpu) {
    Allocator allocator(slots);
    Graph initial(64);
    require(allocator.reserve(initial) && ggml_gallocr_alloc_graph(allocator.value, initial.graph),
            "initial graph allocation failed");
    allocator.check_shared();
    compute(initial, cpu);
    Graph growth(65536);
    fail_growth(allocator, growth);
    Graph retry(retry_elements);
    const size_t before_retry = allocation_calls;
    if (slots == 1) {
        // The single-buffer API reserves automatically when its plan is invalid.
        require(ggml_gallocr_alloc_graph(allocator.value, retry.graph), "single-buffer retry failed");
    } else {
        // A multi-buffer scheduler must be told to reserve again. A smaller
        // graph has the same topology and fits the failed plan's recorded sizes;
        // returning true here would use missing buffer storage.
        require(!ggml_gallocr_alloc_graph(allocator.value, retry.graph), "failed allocation plan was reused");
        require(allocation_calls == before_retry, "multi-buffer alloc_graph reserved unexpectedly");
        require(allocator.reserve(retry) && ggml_gallocr_alloc_graph(allocator.value, retry.graph),
                "multi-buffer retry failed after reserve");
    }
    require(allocation_calls == before_retry + 1, "retry did not allocate one deduplicated CPU buffer");
    allocator.check_shared();
    compute(retry, cpu);
}

void ordinary_reuse(int slots, ggml_backend_t cpu) {
    Allocator allocator(slots);
    size_t previous_capacity = 0;
    void* previous_base = nullptr;
    for (int elements : {64, 1024, 128, 768, 32, 2048, 256}) {
        Graph graph(elements);
        const size_t before = allocation_calls;
        require(allocator.reserve(graph) && ggml_gallocr_alloc_graph(allocator.value, graph.graph),
                "ordinary graph allocation failed");
        allocator.check_shared();
        const size_t capacity = ggml_gallocr_get_buffer_size(allocator.value, 0);
        void* base = ggml_backend_buffer_get_base(graph.a->buffer);
        require(capacity >= previous_capacity, "allocator capacity shrank unexpectedly");
        if (capacity == previous_capacity) {
            require(allocation_calls == before && base == previous_base,
                    "fitting graph unnecessarily replaced its buffer");
        } else {
            require(allocation_calls == before + 1, "growth did not allocate one shared buffer");
        }
        compute(graph, cpu);
        previous_capacity = capacity;
        previous_base = base;
    }
}
} // namespace

int main(int argc, char** argv) {
    const std::string selected = argc == 2 ? argv[1] : "all";
    try {
        require(argc <= 2, "usage: scnet-allocator-test [case-name]");
        AllocationHook hook;
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
        require(cpu != nullptr, "CPU backend initialization failed");
        ggml_backend_cpu_set_n_threads(cpu.get(), 1);
        int passed = 0;
        auto run = [&](const std::string& name, auto function) {
            if (selected != "all" && selected != name)
                return;
            const size_t allocations_before = allocation_calls, failures_before = allocation_failures;
            const size_t frees_before = frees;
            require(live_buffers.empty(), "previous control leaked a CPU buffer");
            function();
            require(live_buffers.empty(), "control leaked a CPU buffer");
            require(!fail_next_allocation, "control left failure injection armed");
            std::printf("PASS %s allocations=%zu injected_failures=%zu frees=%zu\n", name.c_str(),
                        allocation_calls - allocations_before, allocation_failures - failures_before,
                        frees - frees_before);
            std::fflush(stdout);
            ++passed;
        };
        for (int slots : {1, 2}) {
            const std::string prefix = slots == 2 ? "shared" : "single";
            run(prefix + "-initial-failure", [&] { initial_failure(slots); });
            run(prefix + "-growth-teardown", [&] { growth_teardown(slots, cpu.get()); });
            run(prefix + "-retry-smaller", [&] { growth_retry(slots, 16, cpu.get()); });
            run(prefix + "-retry-same", [&] { growth_retry(slots, 65536, cpu.get()); });
            run(prefix + "-retry-larger", [&] { growth_retry(slots, 131072, cpu.get()); });
            run(prefix + "-reuse-grow-shrink", [&] { ordinary_reuse(slots, cpu.get()); });
        }
        require(passed > 0, "unknown allocator control name");
        std::printf("SCNet allocator controls: %d groups passed; no model inference\n", passed);
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "SCNet allocator control failed: %s\n", error.what());
        return 1;
    }
}
