// Model-free regression for inverse-STFT frame scratch. This counts live C++
// allocations during istft, not process RSS or a BLAS library's private mallocs.
// The required PCM/overlap buffers may grow with output length; inverse frames
// must not add an unbounded T_frames * n_fft allocation.
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <new>

#include "core/istft.h"
#include "scoped_test_env.h"

namespace {
std::atomic<bool> tracking{false};
std::atomic<size_t> live_bytes{0}, peak_bytes{0};
struct alignas(std::max_align_t) Allocation {
    size_t size;
    bool tracked;
};
} // namespace

void* operator new(size_t n) {
    auto* p = static_cast<Allocation*>(std::malloc(sizeof(Allocation) + n));
    if (!p)
        throw std::bad_alloc();
    p->size = n;
    p->tracked = tracking.load();
    if (p->tracked) {
        const size_t current = live_bytes.fetch_add(n) + n;
        size_t peak = peak_bytes.load();
        while (current > peak && !peak_bytes.compare_exchange_weak(peak, current)) {
        }
    }
    return p + 1;
}
void operator delete(void* p) noexcept {
    if (!p)
        return;
    auto* allocation = static_cast<Allocation*>(p) - 1;
    if (allocation->tracked)
        live_bytes.fetch_sub(allocation->size);
    std::free(allocation);
}
void operator delete(void* p, size_t) noexcept {
    ::operator delete(p);
}
void* operator new[](size_t n) {
    return ::operator new(n);
}
void operator delete[](void* p) noexcept {
    ::operator delete(p);
}
void operator delete[](void* p, size_t) noexcept {
    ::operator delete(p);
}

int main() {
    const int N = 64, T = 5000, hop = N / 4, bins = N / 2 + 1;
    std::vector<float> mag((size_t)T * bins, 0.3f), phase((size_t)T * bins, 0.6f);
    ScopedTestEnv serial("CRISPASR_ISTFT_SERIAL");
    // An immutable CBLAS Fourier basis is independent of T and persists across
    // calls. Warm it outside the measurement, like a model's constant weights.
    (void)core_istft::istft(mag.data(), phase.data(), N, hop, 2);
    const size_t output_bytes = ((size_t)(T - 1) * hop + N) * sizeof(float);
    for (const int workers : {1, 4, 0}) {
        serial.set(workers == 0 ? "1" : nullptr);
        live_bytes = 0;
        peak_bytes = 0;
        tracking = true;
        auto pcm = core_istft::istft(mag.data(), phase.data(), N, hop, T, nullptr, core_istft::TRIM_CENTER, 0.0f, 1e-8f,
                                     false, workers);
        tracking = false;
        // Three output-sized buffers plus fixed slack for windows, spectra,
        // worker bookkeeping and a bounded tile. The old all-frames path
        // exceeds this by >600 KiB for this shape.
        const size_t scratch_slack = (workers == 1 || workers == 0) ? 128 * 1024 : 384 * 1024;
        const size_t limit = 3 * output_bytes + scratch_slack;
        const size_t peak = peak_bytes.load();
        std::printf("workers=%d forced_serial=%d peak_cpp_bytes=%zu limit_bytes=%zu samples=%zu\n", workers,
                    workers == 0, peak, limit, pcm.size());
        if (pcm.size() != (size_t)(T - 1) * hop || peak > limit)
            return 1;
        for (const float sample : pcm)
            if (!std::isfinite(sample))
                return 2;
    }
    return 0;
}
