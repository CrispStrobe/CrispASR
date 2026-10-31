// core/parallel_for.h — one place for "run this loop across the cores".
//
// Five copies of this had accumulated, each written for its own file:
//   firered_vad.cpp        firered_parallel_for   (static template)
//   marblenet_vad.cpp      marblenet_parallel_for (static template, identical)
//   omnivoice.cpp          ov_parallel_for        (std::function variant)
//   core/mel.cpp           inlined, no helper
//   core/foxnose_pipeline.cpp  inlined, no helper
//
// They come in TWO genuinely different shapes, and both are kept here rather
// than forcing one onto the other:
//
//   for_each_chunk — splits [0, n) into one contiguous block per thread.
//     Right when every item costs about the same (audio frames, feature rows),
//     which is what four of the five above do. Cheapest possible: no atomics,
//     one call per thread.
//
//   for_each_task — hands out item indices from an atomic counter, so threads
//     take more work as they finish. Right when per-item cost varies a lot,
//     where a static split would leave a thread idle holding the cheap items.
//     Also carries a SLOT index for callers that own per-thread resources
//     (foxnose_pipeline gives each worker its own model context this way).
//
// Both reuse the CALLING thread as one of the workers, so `n_threads == 1`
// costs nothing beyond the loop itself.
//
// DETERMINISM CONTRACT (why every caller here is safe to parallelise): `fn`
// must write only to storage owned by its own index/range. Any order-sensitive
// reduction — argmin, argmax, accumulation — belongs in a SERIAL pass
// afterwards, so results do not depend on thread timing.
//
// Not std::execution::par: libc++ gates <execution> behind
// _LIBCPP_HAS_EXPERIMENTAL_PSTL and ships no backend to link against (verified
// on Apple clang 21: the parallel overloads fail to resolve, and force-enabling
// them fails to link __pstl::__libdispatch::__dispatch_apply), while libstdc++
// implements it over Intel TBB, which this tree does not depend on.
//
// Not OpenMP: the tree does use it (~28 pragma sites), but every one is
// #ifdef _OPENMP-guarded and OpenMP is absent from the stock macOS toolchain,
// so those loops run serial there. core/mel.cpp keeps its OpenMP path for the
// platforms that have it and uses this helper as the portable path; the rest
// had no OpenMP path to begin with.

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace core_parallel {

// Number of threads to use for `n_items`, given a caller budget.
// `n_threads <= 0` means "ask the machine".
inline int resolve_threads(int n_items, int n_threads) {
    if (n_items <= 1)
        return 1;
    if (n_threads <= 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        n_threads = (int)(hw == 0 ? 1u : hw);
    }
    return std::max(1, std::min(n_threads, n_items));
}

// A thread budget is a ceiling, not a reason to create a worker for a tiny
// loop. Callers estimate independent work in their own units (e.g. float
// elements or multiply-adds), and require this much work per worker. Keep
// the estimate separate from n_items: one FFT frame can cost much more than
// one output sample. This helper is also usable before allocating scratch.
inline int resolve_threads_for_work(int n_items, int n_threads, uint64_t work_per_item, uint64_t min_work_per_thread) {
    if (n_items <= 1 || work_per_item == 0 || n_threads == 1)
        return 1;
    const uint64_t count = (uint64_t)n_items;
    const uint64_t total = work_per_item > std::numeric_limits<uint64_t>::max() / count
                               ? std::numeric_limits<uint64_t>::max()
                               : count * work_per_item;
    const uint64_t useful =
        min_work_per_thread == 0 ? (uint64_t)n_items : std::max<uint64_t>(1, total / min_work_per_thread);
    // Avoid even querying the runtime/hardware for cheap loops.
    if (useful == 1)
        return 1;
#if defined(_OPENMP)
    // Do not create nested teams under an existing parallel operation.
    if (omp_in_parallel())
        return 1;
    if (n_threads <= 0)
        n_threads = omp_get_max_threads();
#endif
    const int budget = resolve_threads(n_items, n_threads);
    return (int)std::min<uint64_t>((uint64_t)budget, useful);
}

// fn(begin, end) over contiguous blocks partitioning [0, n_items).
template <typename F> void for_each_chunk(int n_items, int n_threads, F&& fn) {
    if (n_items <= 0)
        return;
    const int nt = resolve_threads(n_items, n_threads);
    if (nt == 1) {
        fn(0, n_items);
        return;
    }
    const int chunk = n_items / nt + (n_items % nt != 0);
    std::vector<std::thread> pool;
    pool.reserve((size_t)nt - 1);
    for (int t = 1; t < nt; t++) {
        const int64_t start = (int64_t)t * chunk;
        if (start >= n_items)
            break;
        const int a = (int)start;
        const int b = (int)std::min<int64_t>(n_items, start + chunk);
        pool.emplace_back([&fn, a, b]() { fn(a, b); });
    }
    fn(0, std::min(n_items, chunk)); // calling thread takes the first block
    for (auto& t : pool)
        t.join();
}

// Work-aware variant for CPU kernels. Reuse an OpenMP team when available;
// repeatedly constructing std::threads regressed kernels on OpenMP builds.
// With no OpenMP runtime, retain the portable contiguous-chunk implementation.
template <typename F>
void for_each_chunk_work(int n_items, int n_threads, uint64_t work_per_item, uint64_t min_work_per_thread, F&& fn) {
    if (n_items <= 0)
        return;
    const int nt = resolve_threads_for_work(n_items, n_threads, work_per_item, min_work_per_thread);
    if (nt == 1) {
        fn(0, n_items);
        return;
    }
#if defined(_OPENMP)
    // Divide with int64_t intermediates so large ranges cannot overflow.
#pragma omp parallel for schedule(static) num_threads(nt)
    for (int t = 0; t < nt; ++t) {
        const int begin = (int)((int64_t)n_items * t / nt);
        const int end = (int)((int64_t)n_items * (t + 1) / nt);
        fn(begin, end);
    }
#else
    for_each_chunk(n_items, nt, std::forward<F>(fn));
#endif
}

// fn(task_index, slot) for every task in [0, n_tasks), pulled from an atomic
// counter. `slot` is in [0, n_threads) and identifies WHICH per-thread
// resource to use; no two concurrent calls share a slot.
template <typename F> void for_each_task(int n_tasks, int n_threads, F&& fn) {
    if (n_tasks <= 0)
        return;
    const int nt = resolve_threads(n_tasks, n_threads);
    if (nt == 1) {
        for (int i = 0; i < n_tasks; i++)
            fn(i, 0);
        return;
    }
    std::atomic<int> next{0};
    auto run = [&](int slot) {
        for (;;) {
            const int i = next.fetch_add(1);
            if (i >= n_tasks)
                return;
            fn(i, slot);
        }
    };
    std::vector<std::thread> pool;
    pool.reserve((size_t)nt - 1);
    for (int t = 1; t < nt; t++)
        pool.emplace_back(run, t);
    run(0); // calling thread is slot 0
    for (auto& t : pool)
        t.join();
}

} // namespace core_parallel
