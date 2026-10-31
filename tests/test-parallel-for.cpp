// test-parallel-for — core/parallel_for.h, the shared fan-out helpers.
//
// These replaced five hand-rolled copies, so the contract they all relied on
// has to be pinned: every item is visited exactly once, ranges partition the
// input, slots are unique among CONCURRENT callers, and the result never
// depends on the thread count.

#include <catch2/catch_test_macros.hpp>

#include "core/parallel_for.h"

#include <atomic>
#include <climits>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>
#include <vector>

using namespace core_parallel;

TEST_CASE("for_each_chunk covers every item exactly once", "[parallel]") {
    for (int n : {1, 2, 3, 7, 8, 9, 64, 1000}) {
        for (int nt : {0, 1, 2, 3, 8, 64}) {
            std::vector<int> hits((size_t)n, 0);
            for_each_chunk(n, nt, [&](int a, int b) {
                for (int i = a; i < b; i++)
                    hits[(size_t)i]++;
            });
            for (int i = 0; i < n; i++) {
                INFO("n=" << n << " nt=" << nt << " i=" << i);
                REQUIRE(hits[(size_t)i] == 1);
            }
        }
    }
}

TEST_CASE("for_each_chunk hands out ordered, non-overlapping ranges", "[parallel]") {
    std::mutex m;
    std::vector<std::pair<int, int>> ranges;
    for_each_chunk(1000, 8, [&](int a, int b) {
        std::lock_guard<std::mutex> g(m);
        ranges.emplace_back(a, b);
    });
    std::sort(ranges.begin(), ranges.end());
    int expect = 0;
    for (auto& r : ranges) {
        CHECK(r.first == expect); // contiguous, no gaps or overlaps
        CHECK(r.second > r.first);
        expect = r.second;
    }
    CHECK(expect == 1000);
}

TEST_CASE("for_each_task covers every task exactly once", "[parallel]") {
    for (int n : {1, 5, 33, 500}) {
        for (int nt : {0, 1, 4, 16}) {
            std::vector<int> hits((size_t)n, 0);
            std::mutex m;
            for_each_task(n, nt, [&](int i, int) {
                std::lock_guard<std::mutex> g(m);
                hits[(size_t)i]++;
            });
            for (int i = 0; i < n; i++) {
                INFO("n=" << n << " nt=" << nt << " i=" << i);
                REQUIRE(hits[(size_t)i] == 1);
            }
        }
    }
}

// foxnose_pipeline hands each worker its OWN model context and relies on never
// getting two concurrent calls with the same slot; that is the invariant here.
TEST_CASE("for_each_task never reuses a slot concurrently", "[parallel]") {
    constexpr int kSlots = 4;
    std::vector<std::atomic<int>> live(kSlots);
    for (auto& v : live)
        v.store(0);
    std::atomic<bool> clash{false};
    std::atomic<int> max_slot{-1};

    for_each_task(200, kSlots, [&](int, int slot) {
        if (slot < 0 || slot >= kSlots) {
            clash.store(true);
            return;
        }
        int prev = max_slot.load();
        while (slot > prev && !max_slot.compare_exchange_weak(prev, slot)) {
        }
        if (live[(size_t)slot].fetch_add(1) != 0)
            clash.store(true); // someone else already inside this slot
        for (volatile int spin = 0; spin < 2000; spin++) {
        }
        live[(size_t)slot].fetch_sub(1);
    });

    CHECK_FALSE(clash.load());
    CHECK(max_slot.load() < kSlots);
}

// The reason for_each_task exists: with wildly uneven per-task cost a static
// split idles threads, so results must still be complete and independent of
// how the work happened to land.
TEST_CASE("for_each_task result is independent of thread count", "[parallel]") {
    auto run = [](int nt) {
        std::vector<long> out(16, 0);
        for_each_task(16, nt, [&](int i, int) {
            long acc = 0;
            const long work = (i % 4 == 3) ? 200000L : 1000L; // deliberately uneven
            for (long j = 0; j < work; j++)
                acc += j % 7;
            out[(size_t)i] = acc + i;
        });
        return out;
    };
    const auto serial = run(1);
    CHECK(run(2) == serial);
    CHECK(run(8) == serial);
    CHECK(run(0) == serial);
}

TEST_CASE("empty and degenerate inputs are no-ops", "[parallel]") {
    int calls = 0;
    for_each_chunk(0, 4, [&](int, int) { calls++; });
    for_each_task(0, 4, [&](int, int) { calls++; });
    for_each_chunk(-1, 4, [&](int, int) { calls++; });
    for_each_task(-5, 4, [&](int, int) { calls++; });
    CHECK(calls == 0);
    CHECK(resolve_threads(1, 8) == 1);   // one item never spawns
    CHECK(resolve_threads(4, 100) == 4); // never more threads than items
}

TEST_CASE("work-aware chunks honor work and caller budgets", "[parallel][unit]") {
    CHECK(resolve_threads_for_work(16, 8, 1024, 65536) == 1);
    CHECK(resolve_threads_for_work(512, 8, 1024, 65536) == 8);
    CHECK(resolve_threads_for_work(512, 2, 1024, 65536) == 2);
    CHECK(resolve_threads_for_work(512, 1, 1024, 65536) == 1);
    CHECK(resolve_threads_for_work(2, 8, 1000000, 1) == 2);
    CHECK(resolve_threads_for_work(512, 8, 0, 1) == 1);
    CHECK(resolve_threads_for_work(512, 8, UINT64_MAX, 1) == 8);
    CHECK(resolve_threads_for_work(0, 8, 1024, 1) == 1);

    const auto caller = std::this_thread::get_id();
    int calls = 0;
    for_each_chunk_work(16, 8, 1024, 65536, [&](int begin, int end) {
        CHECK(std::this_thread::get_id() == caller);
        CHECK(begin == 0);
        CHECK(end == 16);
        ++calls;
    });
    CHECK(calls == 1);

    std::vector<int> visits(1003, 0);
    std::mutex mutex;
    std::set<std::thread::id> workers;
    for_each_chunk_work(1003, 3, 10000, 1, [&](int begin, int end) {
        for (int i = begin; i < end; ++i)
            ++visits[(size_t)i];
        std::lock_guard<std::mutex> lock(mutex);
        workers.insert(std::this_thread::get_id());
    });
    CHECK(workers.size() <= 3);
    for (int visits_to_item : visits)
        CHECK(visits_to_item == 1);

    for_each_chunk_work(0, 8, 100000, 1, [&](int, int) { ++calls; });
    CHECK(calls == 1);
}

TEST_CASE("work-aware chunks cover large integer ranges without overflow", "[parallel][unit]") {
    std::mutex mutex;
    std::vector<std::pair<int, int>> ranges;
    for_each_chunk_work(INT_MAX, 3, 1, 1, [&](int begin, int end) {
        std::lock_guard<std::mutex> lock(mutex);
        ranges.emplace_back(begin, end);
    });
    std::sort(ranges.begin(), ranges.end());
    int next = 0;
    for (const auto& range : ranges) {
        CHECK(range.first == next);
        CHECK(range.second > range.first);
        next = range.second;
    }
    CHECK(next == INT_MAX);
}

#if defined(_OPENMP)
TEST_CASE("work-aware chunks do not create nested OpenMP teams", "[parallel][unit]") {
    std::atomic<int> unexpected{0};
#pragma omp parallel num_threads(2)
    {
        if (core_parallel::resolve_threads_for_work(100, 8, 100000, 1) != 1)
            ++unexpected;
        const auto caller = std::this_thread::get_id();
        core_parallel::for_each_chunk_work(100, 8, 100000, 1, [&](int begin, int end) {
            if (std::this_thread::get_id() != caller || begin != 0 || end != 100)
                ++unexpected;
        });
    }
    CHECK(unexpected == 0);
}
#endif
