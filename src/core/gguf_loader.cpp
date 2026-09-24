// src/core/gguf_loader.cpp — implementation of core_gguf:: helpers.
// See gguf_loader.h for the interface contract.

#include "gguf_loader.h"

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>

// core_cpu_backend:: is used unconditionally below (the zero-copy CPU mmap
// path), so this include must NOT sit in the POSIX arm of the block that
// follows — it did, and Windows built without it. MSVC caught that;
// Clang and GCC did not, because they happened to reach the header
// transitively.
#include "core/ggml_cpu_backend.h"

#if defined(_WIN32)
#include <io.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace core_gguf {

// ---------------------------------------------------------------------------
// Pass 1: metadata
// ---------------------------------------------------------------------------

gguf_context* open_metadata(const char* path) {
    gguf_init_params gp = {/*.no_alloc=*/true, /*.ctx=*/nullptr};
    gguf_context* g = gguf_init_from_file(path, gp);
    if (!g) {
        fprintf(stderr, "core_gguf: failed to open '%s' for metadata read\n", path);
    }
    return g;
}

void free_metadata(gguf_context* gctx) {
    if (gctx)
        gguf_free(gctx);
}

// Type-checked scalar readers. The GGUF format stores types explicitly so
// we can validate; if the file has a mismatched type the reader silently
// returns the default rather than crashing, matching the existing inline
// helpers in each model.

uint32_t kv_u32(gguf_context* gctx, const char* key, uint32_t default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return default_val;
    const gguf_type t = gguf_get_kv_type(gctx, k);
    switch (t) {
    case GGUF_TYPE_UINT32:
        return gguf_get_val_u32(gctx, k);
    case GGUF_TYPE_INT32:
        return (uint32_t)gguf_get_val_i32(gctx, k);
    case GGUF_TYPE_UINT64:
        return (uint32_t)gguf_get_val_u64(gctx, k);
    case GGUF_TYPE_INT64:
        return (uint32_t)gguf_get_val_i64(gctx, k);
    case GGUF_TYPE_UINT16:
        return gguf_get_val_u16(gctx, k);
    case GGUF_TYPE_INT16:
        return (uint32_t)gguf_get_val_i16(gctx, k);
    case GGUF_TYPE_UINT8:
        return gguf_get_val_u8(gctx, k);
    case GGUF_TYPE_INT8:
        return (uint32_t)gguf_get_val_i8(gctx, k);
    default:
        return default_val;
    }
}

int32_t kv_i32(gguf_context* gctx, const char* key, int32_t default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return default_val;
    const gguf_type t = gguf_get_kv_type(gctx, k);
    switch (t) {
    case GGUF_TYPE_INT32:
        return gguf_get_val_i32(gctx, k);
    case GGUF_TYPE_UINT32:
        return (int32_t)gguf_get_val_u32(gctx, k);
    case GGUF_TYPE_INT64:
        return (int32_t)gguf_get_val_i64(gctx, k);
    case GGUF_TYPE_UINT64:
        return (int32_t)gguf_get_val_u64(gctx, k);
    default:
        return default_val;
    }
}

float kv_f32(gguf_context* gctx, const char* key, float default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return default_val;
    const gguf_type t = gguf_get_kv_type(gctx, k);
    if (t == GGUF_TYPE_FLOAT32)
        return gguf_get_val_f32(gctx, k);
    if (t == GGUF_TYPE_FLOAT64)
        return (float)gguf_get_val_f64(gctx, k);
    return default_val;
}

bool kv_bool(gguf_context* gctx, const char* key, bool default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return default_val;
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_BOOL)
        return default_val;
    return gguf_get_val_bool(gctx, k);
}

std::string kv_str(gguf_context* gctx, const char* key, const char* default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return default_val ? default_val : "";
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_STRING)
        return default_val ? default_val : "";
    const char* s = gguf_get_val_str(gctx, k);
    return s ? std::string(s) : std::string(default_val ? default_val : "");
}

std::vector<std::string> kv_str_array(gguf_context* gctx, const char* key) {
    std::vector<std::string> out;
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return out;
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_ARRAY)
        return out;
    if (gguf_get_arr_type(gctx, k) != GGUF_TYPE_STRING)
        return out;
    const int n = gguf_get_arr_n(gctx, k);
    out.reserve((size_t)n);
    for (int i = 0; i < n; i++) {
        out.emplace_back(gguf_get_arr_str(gctx, k, i));
    }
    return out;
}

std::vector<float> kv_f32_array(gguf_context* gctx, const char* key) {
    std::vector<float> out;
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return out;
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_ARRAY)
        return out;
    if (gguf_get_arr_type(gctx, k) != GGUF_TYPE_FLOAT32)
        return out;
    const size_t n = gguf_get_arr_n(gctx, k);
    const float* data = (const float*)gguf_get_arr_data(gctx, k);
    out.assign(data, data + n);
    return out;
}

// ---------------------------------------------------------------------------
// Pass 2: tensor allocation + weight data copy.
// ---------------------------------------------------------------------------

namespace {

// Read a file slice into a backend tensor. Uses mmap on POSIX; falls back
// to pread/lseek+read when mmap is unavailable (rare in practice).
//
// On POSIX the mmap lives for the duration of one load call — we copy via
// ggml_backend_tensor_set then unmap. No mmap persists past load_weights().
struct MappedFile {
    int fd = -1;
    void* base = nullptr;
    size_t size = 0;
    bool ok = false;

    // When `writable` is true, the mapping is created with copy-on-write
    // semantics (POSIX MAP_PRIVATE + PROT_READ|PROT_WRITE / Win32
    // FILE_MAP_COPY). Reads share the file's page cache; writes get a
    // private anonymous duplicate of the touched page. This lets backends
    // that mutate weights post-load (e.g. parakeet's BN-into-conv fold) run
    // unchanged on the zero-copy path without modifying the underlying file.
    explicit MappedFile(const char* path, bool writable = false) {
#if defined(_WIN32)
        const DWORD page_protect = writable ? PAGE_WRITECOPY : PAGE_READONLY;
        const DWORD view_access = writable ? FILE_MAP_COPY : FILE_MAP_READ;
        HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        if (hFile == INVALID_HANDLE_VALUE)
            return;
        LARGE_INTEGER fsize;
        if (!GetFileSizeEx(hFile, &fsize)) {
            CloseHandle(hFile);
            return;
        }
        size = (size_t)fsize.QuadPart;
        HANDLE hMap = CreateFileMappingA(hFile, nullptr, page_protect, 0, 0, nullptr);
        CloseHandle(hFile);
        if (!hMap)
            return;
        base = MapViewOfFile(hMap, view_access, 0, 0, 0);
        CloseHandle(hMap);
        if (!base)
            return;
        ok = true;
#else
        fd = ::open(path, O_RDONLY);
        if (fd < 0)
            return;
        struct stat st;
        if (fstat(fd, &st) != 0) {
            ::close(fd);
            fd = -1;
            return;
        }
        size = (size_t)st.st_size;
        const int prot = writable ? (PROT_READ | PROT_WRITE) : PROT_READ;
        const int flags = writable ? MAP_PRIVATE : MAP_SHARED;
        base = ::mmap(nullptr, size, prot, flags, fd, 0);
        ::close(fd);
        fd = -1;
        if (base == MAP_FAILED) {
            base = nullptr;
            return;
        }
        ok = true;
#endif
    }
    ~MappedFile() {
#if defined(_WIN32)
        if (base)
            UnmapViewOfFile(base);
#else
        if (base)
            ::munmap(base, size);
#endif
    }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    // Transfer ownership of the mmap region out of the RAII handle so it
    // outlives the destructor. Used by the CRISPASR_GGUF_MMAP=1 path to
    // hand the mapping to a backend buffer that owns it for the model's
    // lifetime.
    void release() {
        base = nullptr;
        size = 0;
    }
};

// PLAN #51a: a CPU backend buffer whose backing memory is an mmap'd file
// region. On free_buffer the mapping is unmapped — that's the entire
// reason this buffer type exists. Tensors must be bound with
// ggml_backend_tensor_alloc(); we do not provide an init_tensor path.
//
// We reuse core_cpu_backend::buffer_type() so ggml_backend_buffer_is_host()
// returns true on this buffer (some scheduler paths key off that).
struct mmap_buffer_ctx {
    void* mmap_base = nullptr;   // page-aligned start of the mmap
    size_t mmap_size = 0;        // length of the mmap
    void* tensor_base = nullptr; // mmap_base + data_off, 32-byte aligned
};

static void* mmap_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return ((mmap_buffer_ctx*)buffer->context)->tensor_base;
}

static void mmap_buffer_free(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    auto* mctx = (mmap_buffer_ctx*)buffer->context;
    if (mctx->mmap_base) {
#if defined(_WIN32)
        UnmapViewOfFile(mctx->mmap_base);
#else
        ::munmap(mctx->mmap_base, mctx->mmap_size);
#endif
    }
    delete mctx;
}

static void mmap_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor, uint8_t value, size_t offset,
                                      size_t size) {
    GGML_ASSERT(tensor);
    memset((char*)tensor->data + offset, value, size);
    GGML_UNUSED(buffer);
}

static void mmap_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor, const void* data, size_t offset,
                                   size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char*)tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

static void mmap_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor* tensor, void* data, size_t offset,
                                   size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char*)tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static bool mmap_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor* src, ggml_tensor* dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
    GGML_UNUSED(buffer);
}

static void mmap_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto* mctx = (mmap_buffer_ctx*)buffer->context;
    memset(mctx->tensor_base, value, buffer->size);
}

static const ggml_backend_buffer_i mmap_buffer_iface = {
    /* .free_buffer    = */ mmap_buffer_free,
    /* .get_base       = */ mmap_buffer_get_base,
    /* .init_tensor    = */ nullptr,
    /* .memset_tensor  = */ mmap_buffer_memset_tensor,
    /* .set_tensor     = */ mmap_buffer_set_tensor,
    /* .get_tensor     = */ mmap_buffer_get_tensor,
    /* .set_tensor_2d  = */ nullptr,
    /* .get_tensor_2d  = */ nullptr,
    /* .cpy_tensor     = */ mmap_buffer_cpy_tensor,
    /* .clear          = */ mmap_buffer_clear,
    /* .reset          = */ nullptr,
};

// PLAN #51a (Metal variant): non-CPU backends use `buffer_from_host_ptr`
// to wrap our mmap region directly into a backend buffer (e.g. an
// MTLResourceStorageModeShared MTLBuffer on Apple-Silicon Metal). We
// CANNOT wrap that inner buffer with our own iface to attach a munmap
// callback: ggml-metal pierces the iface abstraction and casts
// `buffer->context` straight to its private `ggml_metal_buffer_t` (see
// `ggml_metal_get_buffer_id` in ggml-metal-context.m), so any wrapper
// makes Metal read garbage and emit "tensor 'X' buffer is nil" for
// every weight — the kokoro Metal gibberish-audio regression.
//
// Instead we hand the inner buffer back as-is and track the mmap region
// in this static side-map. The inner backend's free callback releases its
// device-side reference but cannot touch the host mapping — Metal's
// `newBufferWithBytesNoCopy:options:deallocator:nil` doesn't own the host
// pages, and `buffer_from_host_ptr` has no deallocator parameter through
// which a backend could take ownership of them. Releasing the mapping is
// therefore the loader's job, and `release_weight_buffer()` below is where
// it happens: it frees the backend buffer and then unmaps the region the
// side-map recorded for it.
//
// The mapping is `MAP_PRIVATE | PROT_READ|PROT_WRITE` (see MappedFile's
// `writable` parameter, needed by backends that fold weights in place after
// load), so every page privatizes on first read and the resident pages are
// dirty and anonymous. They can be compressed or swapped, never dropped —
// which is why holding them costs real memory for the life of the process
// rather than page cache the kernel can reclaim.
struct gpu_mmap_handle {
    void* base = nullptr;
    size_t size = 0;
};
static std::mutex g_gpu_mmap_mu;
static std::map<ggml_backend_buffer_t, gpu_mmap_handle> g_gpu_mmap;

static void register_gpu_mmap(ggml_backend_buffer_t buf, void* base, size_t size) {
    std::lock_guard<std::mutex> lk(g_gpu_mmap_mu);
    g_gpu_mmap[buf] = {base, size};
}
static gpu_mmap_handle lookup_gpu_mmap(ggml_backend_buffer_t buf) {
    std::lock_guard<std::mutex> lk(g_gpu_mmap_mu);
    auto it = g_gpu_mmap.find(buf);
    return it != g_gpu_mmap.end() ? it->second : gpu_mmap_handle{};
}
// Look the region up and remove the entry in one critical section. Splitting
// this into a lookup followed by an erase would let a second release of the
// same buffer read the entry before the first erased it and unmap twice; it
// would also race a concurrent load whose fresh buffer landed on the same
// address after the free. A default-constructed handle means no entry, which
// is the ordinary case for the CPU mmap path and the legacy alloc+copy path.
static gpu_mmap_handle take_gpu_mmap(ggml_backend_buffer_t buf) {
    std::lock_guard<std::mutex> lk(g_gpu_mmap_mu);
    auto it = g_gpu_mmap.find(buf);
    if (it == g_gpu_mmap.end())
        return gpu_mmap_handle{};
    const gpu_mmap_handle h = it->second;
    g_gpu_mmap.erase(it);
    return h;
}

static void unmap_region(void* base, size_t size) {
    if (!base || size == 0)
        return;
#if defined(_WIN32)
    (void)size;
    UnmapViewOfFile(base);
#else
    ::munmap(base, size);
#endif
}

// Issue #276 overflow chunks. A partition larger than the 1.5 GiB chunk limit
// is allocated as several backend buffers; only the first goes into
// WeightLoad::buf / ::buf_cpu, and the rest were left for the caller to free
// out of WeightLoad::split_bufs. Every one of the eighteen backends that calls
// load_weights_split() moves buf and buf_cpu into its own model struct and
// drops the vector, so those chunks were freed by nobody. An obligation that
// no caller has ever honoured belongs somewhere else.
//
// The overflow chunks are therefore owned here, keyed to the first buffer of
// their own partition, and released with it. split_bufs still lists them so a
// caller can see how a load was partitioned, but it is no longer a set of
// handles the caller must free.
static std::mutex g_split_mu;
static std::map<ggml_backend_buffer_t, std::vector<ggml_backend_buffer_t>> g_split_extra;

static void register_split_extra(ggml_backend_buffer_t primary, const std::vector<ggml_backend_buffer_t>& extra) {
    if (!primary || extra.empty())
        return;
    std::lock_guard<std::mutex> lk(g_split_mu);
    auto& slot = g_split_extra[primary];
    slot.insert(slot.end(), extra.begin(), extra.end());
}
static std::vector<ggml_backend_buffer_t> take_split_extra(ggml_backend_buffer_t primary) {
    std::lock_guard<std::mutex> lk(g_split_mu);
    auto it = g_split_extra.find(primary);
    if (it == g_split_extra.end())
        return {};
    std::vector<ggml_backend_buffer_t> extra = std::move(it->second);
    g_split_extra.erase(it);
    return extra;
}

namespace {
struct FitSet {
    std::set<std::string> cpu;
};
bool fit_is_gpu(const char* name, void* user) {
    return static_cast<FitSet*>(user)->cpu.count(name) == 0;
}
// Names of the tensors in `path` that `backend` cannot bind: its own
// supports_op() on each weight leaf, the check ggml_backend_sched aborts on.
// Empty when the file cannot be read (the real load then reports the error).
FitSet scan_unbindable(const char* path, ggml_backend_t backend) {
    FitSet fit;
    ggml_context* mctx = nullptr;
    gguf_init_params gp = {/*.no_alloc=*/true, /*.ctx=*/&mctx};
    gguf_context* g = gguf_init_from_file(path, gp);
    if (g && mctx) {
        for (ggml_tensor* t = ggml_get_first_tensor(mctx); t; t = ggml_get_next_tensor(mctx, t)) {
            if (!ggml_backend_supports_op(backend, t))
                fit.cpu.insert(ggml_get_name(t));
        }
    }
    if (g)
        gguf_free(g);
    if (mctx)
        ggml_free(mctx);
    return fit;
}
void log_unbindable(const char* tag, const FitSet& fit, ggml_backend_t backend) {
    for (const auto& n : fit.cpu)
        fprintf(stderr, "%s: '%s' exceeds what %s can bind; keeping it on the CPU\n", tag, n.c_str(),
                ggml_backend_name(backend));
}
} // namespace

// Issue #94 (chatterbox-turbo segfault during init on macOS / Apple
// Silicon): the legacy alloc+copy load path takes 30-60 s for the
// chatterbox-turbo T3 (658 MB Q8_0) on slow disks and reproducibly
// fails for some users. The zero-copy mmap path completes the same
// load in ~5-10 s and uses half the peak RSS. The CPU mmap path has
// been validated for every backend that goes through this loader
// (mimo-asr, voxtral, voxtral4b, chatterbox base + turbo, kokoro,
// qwen3-tts, vibevoice, parakeet, granite, …) since the PLAN #51a
// rollout in late April; flipping the default after a month of opt-in
// testing matches llama.cpp's behaviour and resolves the slow-load
// reports.
//
// Opt out with `CRISPASR_GGUF_MMAP=0` for users whose model files live
// on volumes that may disappear mid-run (network mounts, removable
// disks); mmap-backed weights SIGBUS if the underlying file vanishes.
static bool mmap_loader_enabled() {
    const char* v = std::getenv("CRISPASR_GGUF_MMAP");
    if (!v || !*v)
        return true;
    return *v != '0';
}

// PLAN #60c: opt-in preload — page-walk the entire mmap region so every
// page is resident before we return. Trades cold-start *load* time for
// cold-start *prefill* time; useful for benchmarking and for users with
// enough RAM to keep the working set resident. POSIX path uses a 1-byte
// volatile read per page; Linux MADV_POPULATE_READ would be a cleaner
// single-syscall version when available.
static bool preload_enabled() {
    const char* v = std::getenv("CRISPASR_GGUF_PRELOAD");
    return v && *v && *v != '0';
}
static void preload_pages(void* base, size_t size) {
#if !defined(_WIN32)
    const long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0)
        return;
    volatile const unsigned char* p = (const unsigned char*)base;
    size_t touched = 0;
    for (size_t off = 0; off < size; off += (size_t)pg) {
        (void)p[off];
        touched++;
    }
    (void)touched;
#else
    (void)base;
    (void)size;
#endif
}

// PLAN #60f: opt-in mlock — pin the mmap region in physical RAM so the
// kernel can't evict pages under memory pressure. Risky on RAM-tight
// hosts (a 16 GB model on a 16 GB box would starve the rest of the
// system). Useful as opt-in for users with comfortable headroom (32+
// GB). Failure (typically RLIMIT_MEMLOCK exceeded) prints a warning
// and continues — mmap'd weights still work, just without the pin.
static bool mlock_enabled() {
    const char* v = std::getenv("CRISPASR_MLOCK");
    return v && *v && *v != '0';
}
static void try_mlock(const char* tag, void* base, size_t size) {
#if !defined(_WIN32)
    if (::mlock(base, size) != 0) {
        fprintf(stderr,
                "%s: mlock(%zu MiB) failed (errno=%d) — pages may still be evicted under "
                "memory pressure. Raise RLIMIT_MEMLOCK (`ulimit -l unlimited`) if you want "
                "the pin to take effect.\n",
                tag, size / (1024 * 1024), errno);
    }
#else
    (void)tag;
    (void)base;
    (void)size;
#endif
}

} // namespace

static bool load_weights_impl(const char* path, ggml_backend_t backend, IncludeTensor include_tensor, void* user,
                              const char* model_tag, WeightLoad& out) {
    const char* tag = model_tag ? model_tag : "core_gguf";

    // Issue #405: under GGML_BACKEND_DL every backend init can legitimately
    // return null — when the host CPU is below the ISA floor of every shipped
    // libggml-cpu variant, ggml_backend_score() refuses them all and the
    // registry has NO CPU device. A null backend used to sail into
    // ggml_backend_get_device() and abort the whole process
    // (GGML_ASSERT(backend), ggml-backend.cpp:471). Fail the load cleanly so
    // the caller's init path reports an error instead.
    if (!backend) {
        fprintf(stderr,
                "%s: no ggml backend available to load '%s' into.\n"
                "%s: if this is a packaged (dynamic-backend) build, the shipped libggml-cpu\n"
                "%s: modules may all require CPU features this host lacks — use the\n"
                "%s: crispasr-*-cpu-legacy artifact or build from source on this machine.\n",
                tag, path, tag, tag, tag);
        return false;
    }

    ggml_context* source_ctx = nullptr;
    gguf_init_params gp = {/*.no_alloc=*/true, /*.ctx=*/&source_ctx};
    gguf_context* gctx = gguf_init_from_file(path, gp);
    if (!gctx || !source_ctx) {
        fprintf(stderr, "%s: failed to load tensor metadata from '%s'\n", tag, path);
        if (gctx)
            gguf_free(gctx);
        if (source_ctx)
            ggml_free(source_ctx);
        return false;
    }

    if (include_tensor) {
        size_t n_selected = 0;
        for (ggml_tensor* t = ggml_get_first_tensor(source_ctx); t; t = ggml_get_next_tensor(source_ctx, t)) {
            if (include_tensor(ggml_get_name(t), user))
                n_selected++;
        }
        if (n_selected == 0) {
            fprintf(stderr, "%s: tensor filter selected no weights from '%s'\n", tag, path);
            gguf_free(gctx);
            ggml_free(source_ctx);
            return false;
        }

        const size_t tensor_overhead = ggml_tensor_overhead();
        if (tensor_overhead != 0 && n_selected > (std::numeric_limits<size_t>::max() - 1024) / tensor_overhead) {
            fprintf(stderr, "%s: filtered tensor metadata size overflow\n", tag);
            gguf_free(gctx);
            ggml_free(source_ctx);
            return false;
        }
        ggml_init_params fp = {
            /*.mem_size   =*/n_selected * tensor_overhead + 1024,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        out.ctx = ggml_init(fp);
        if (!out.ctx) {
            fprintf(stderr, "%s: failed to create filtered tensor context\n", tag);
            gguf_free(gctx);
            ggml_free(source_ctx);
            return false;
        }
        for (ggml_tensor* t = ggml_get_first_tensor(source_ctx); t; t = ggml_get_next_tensor(source_ctx, t)) {
            if (!include_tensor(ggml_get_name(t), user))
                continue;
            ggml_tensor* selected = ggml_dup_tensor(out.ctx, t);
            if (!selected) {
                fprintf(stderr, "%s: failed to duplicate filtered tensor metadata for '%s'\n", tag, ggml_get_name(t));
                gguf_free(gctx);
                ggml_free(source_ctx);
                ggml_free(out.ctx);
                out.ctx = nullptr;
                return false;
            }
            ggml_set_name(selected, ggml_get_name(t));
        }
        ggml_free(source_ctx);
    } else {
        out.ctx = source_ctx;
    }

    // PLAN #51a: zero-copy CPU path. Skip ggml_backend_alloc_ctx_tensors
    // (which would allocate a fresh backend-side buffer) and instead bind
    // each tensor directly into the mmap'd file. Saves one full copy of
    // the weights — the difference between a 14.9 GB F16 GGUF loading on
    // a 16 GB Mac and thrashing swap. Default-on as of issue #94 (slow /
    // failing chatterbox-turbo load on macOS); opt out with
    // `CRISPASR_GGUF_MMAP=0`.
    if (mmap_loader_enabled() && core_cpu_backend::is_cpu(backend)) {
        MappedFile mf(path, /*writable=*/true);
        if (mf.ok) {
            const size_t data_off = gguf_get_data_offset(gctx);
            const size_t mmap_size = mf.size;
            if (data_off > mmap_size) {
                fprintf(stderr, "%s: GGUF data offset %zu exceeds file size %zu - file truncated?\n", tag, data_off,
                        mmap_size);
                gguf_free(gctx);
                ggml_free(out.ctx);
                out.ctx = nullptr;
                return false;
            }
            char* tensor_base = (char*)mf.base + data_off;
            const size_t buf_size = mmap_size > data_off ? (mmap_size - data_off) : 0;

            // cppcheck-suppress uninitvar
            auto* mctx = new mmap_buffer_ctx{};
            mctx->mmap_base = mf.base;
            mctx->mmap_size = mmap_size;
            mctx->tensor_base = tensor_base;
            mf.release();
            // Hint kernel to start async readahead of the entire weight
            // region. Without this we hit a synchronous page fault on every
            // first access during prefill (~5-10 ms each on the 99%-full
            // external disk we hit during PLAN #51c F16 testing). Mirrors
            // llama.cpp's `llama_mmap` populate path.
#if !defined(_WIN32)
            if (!include_tensor)
                ::posix_madvise(mctx->mmap_base, mctx->mmap_size, POSIX_MADV_WILLNEED);
#endif
            // PLAN #60c / #60f: optional preload + mlock, opt-in via env.
            // A filtered component may live inside a much larger parent GGUF.
            // Whole-file readahead/preload/mlock would defeat filtered loading
            // by making every unrelated tensor resident (or pinned), so let
            // ordinary page faults bring in only the selected tensor ranges.
            if (!include_tensor && preload_enabled())
                preload_pages(mctx->mmap_base, mctx->mmap_size);
            if (!include_tensor && mlock_enabled())
                try_mlock(tag, mctx->mmap_base, mctx->mmap_size);

            out.buf = ggml_backend_buffer_init(core_cpu_backend::buffer_type(), mmap_buffer_iface, mctx, buf_size);
            if (!out.buf) {
                fprintf(stderr, "%s: failed to wrap mmap in backend buffer\n", tag);
#if defined(_WIN32)
                UnmapViewOfFile(mctx->mmap_base);
#else
                ::munmap(mctx->mmap_base, mctx->mmap_size);
#endif
                delete mctx;
                gguf_free(gctx);
                ggml_free(out.ctx);
                out.ctx = nullptr;
                return false;
            }

            // Pre-validate tensor bounds before calling ggml_backend_tensor_alloc,
            // which fires a hard GGML_ASSERT when any tensor's range exceeds
            // buf_size.  Seen on macOS 26.x Tahoe (issue #94) — likely a
            // mismatched or truncated GGUF file.  Detect it here and fall back
            // gracefully to the legacy alloc+copy path instead of crashing.
            bool bounds_ok = true;
            for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
                const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
                if (tid < 0)
                    continue;
                const size_t off = gguf_get_tensor_offset(gctx, tid);
                const size_t nbytes = ggml_nbytes(t);
                if (off > buf_size || nbytes > buf_size - off) {
                    fprintf(stderr,
                            "%s: mmap bounds check failed for tensor '%s' "
                            "(off=%zu + nbytes=%zu > buf_size=%zu) - "
                            "falling back to legacy loader\n",
                            tag, ggml_get_name(t), off, nbytes, buf_size);
                    bounds_ok = false;
                    break;
                }
            }
            if (bounds_ok) {
                for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
                    out.tensors[ggml_get_name(t)] = t;
                    const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
                    if (tid < 0)
                        continue;
                    const size_t off = gguf_get_tensor_offset(gctx, tid);
                    ggml_backend_tensor_alloc(out.buf, t, tensor_base + off);
                }
                gguf_free(gctx);
                return true;
            }
            // Bounds check failed — release the mmap buffer and fall through to
            // the legacy alloc+copy path below.
            release_weight_buffer(out.buf);
        }
        // mmap failed or bounds check failed — fall through to the legacy
        // alloc + copy path. Functionally equivalent, just with more RSS.
    }

    // PLAN #51a (Metal variant): zero-copy GPU path via the device's
    // `buffer_from_host_ptr` capability. On Apple-Silicon Metal this maps
    // to `[MTLDevice newBufferWithBytesNoCopy:length:options:deallocator:]`
    // wrapping our mmap region in an MTLResourceStorageModeShared buffer
    // — same physical pages the CPU sees thanks to unified memory, no
    // device-side allocation, no copy. On discrete-Metal hosts (Intel +
    // eGPU) this lets the GPU page-fault from the file directly. The
    // device-cap probe means we silently fall through on backends that
    // don't advertise host-pointer support (CUDA without managed memory,
    // Vulkan, etc.).
    //
    // The inner buffer is returned as `out.buf` UNWRAPPED. We used to
    // wrap it to attach a munmap callback to free_buffer, but ggml-metal
    // pierces the iface abstraction and casts `buffer->context` straight
    // to its `ggml_metal_buffer_t` — wrapping made Metal read garbage
    // and emit "tensor 'X' buffer is nil" for every weight (kokoro
    // gibberish-audio regression). The mmap region is instead recorded in
    // g_gpu_mmap against the buffer, and release_weight_buffer() unmaps it
    // after freeing that buffer. Metal does not own the pages
    // (deallocator=nil) and `buffer_from_host_ptr` offers no way to hand it
    // ownership, so the caller must release through that entry point rather
    // than through ggml_backend_buffer_free().
    if (mmap_loader_enabled() && !core_cpu_backend::is_cpu(backend)) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(dev, &props);
        if (props.caps.buffer_from_host_ptr) {
            MappedFile mf(path, /*writable=*/true);
            if (mf.ok) {
                const size_t data_off = gguf_get_data_offset(gctx);
                if (data_off > mf.size) {
                    fprintf(stderr, "%s: GGUF data offset %zu exceeds file size %zu - file truncated?\n", tag, data_off,
                            mf.size);
                    gguf_free(gctx);
                    ggml_free(out.ctx);
                    out.ctx = nullptr;
                    return false;
                }
                char* tensor_base = (char*)mf.base + data_off;

                // Hand the entire mmap region (including GGUF header) to
                // the device. The backend's `buffer_from_host_ptr` uses
                // the size for its internal MTLBuffer view; tensor binds
                // below offset into this base.
                ggml_backend_buffer_t inner = ggml_backend_dev_buffer_from_host_ptr(dev, mf.base, mf.size,
                                                                                    /*max_tensor_size=*/0);
                if (inner) {
                    void* leaked_base = mf.base;
                    size_t leaked_size = mf.size;
                    mf.release();
                    // Hint kernel async readahead — same rationale as the
                    // CPU branch above. On Apple Silicon the unified-memory
                    // shared-storage MTLBuffer reads the same physical
                    // pages, so this readahead benefits both CPU and GPU
                    // accesses with one call.
#if !defined(_WIN32)
                    if (!include_tensor)
                        ::posix_madvise(leaked_base, leaked_size, POSIX_MADV_WILLNEED);
#endif
                    // PLAN #60c / #60f: optional preload + mlock, opt-in
                    // via env. mlock is particularly meaningful here —
                    // pinning prevents Metal's shared-storage reads from
                    // racing CPU page faults under memory pressure.
                    if (!include_tensor && preload_enabled())
                        preload_pages(leaked_base, leaked_size);
                    if (!include_tensor && mlock_enabled())
                        try_mlock(tag, leaked_base, leaked_size);

                    register_gpu_mmap(inner, leaked_base, leaked_size);
                    out.buf = inner;

                    // Pre-validate bounds (same rationale as CPU mmap path).
                    bool bounds_ok = true;
                    for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
                        const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
                        if (tid < 0)
                            continue;
                        const size_t off = gguf_get_tensor_offset(gctx, tid);
                        // Overflow-safe: a crafted GGUF can set off/nbytes near SIZE_MAX
                        // so the additive data_off+off+nbytes wraps and passes. Compare
                        // subtractively.
                        const size_t nb = ggml_nbytes(t);
                        // A mmap'd buffer binds tensors at their PACKED GGUF file
                        // offsets, so a backend whose alloc size exceeds
                        // ggml_nbytes (CUDA pads quantized rows) cannot use this
                        // path at all — there is nowhere to put the padding, and
                        // ggml_backend_tensor_alloc would assert. No such backend
                        // advertises buffer_from_host_ptr today (CUDA reports
                        // false; this is Metal's path), so this is a guard, not a
                        // live fix: refuse into the legacy copy path instead of
                        // silently overrunning if that ever changes.
                        if (ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(out.buf), t) != nb) {
                            fprintf(stderr,
                                    "%s: backend pads '%s' beyond its packed size — "
                                    "mmap binding impossible, using legacy loader\n",
                                    tag, ggml_get_name(t));
                            bounds_ok = false;
                            break;
                        }
                        if (data_off > leaked_size || off > leaked_size - data_off ||
                            nb > leaked_size - data_off - off) {
                            fprintf(stderr,
                                    "%s: GPU mmap bounds check failed for tensor '%s' "
                                    "(data_off=%zu + off=%zu + nbytes=%zu > file_size=%zu) — "
                                    "falling back to legacy loader\n",
                                    tag, ggml_get_name(t), data_off, off, ggml_nbytes(t), leaked_size);
                            bounds_ok = false;
                            break;
                        }
                    }
                    if (!bounds_ok) {
                        // Release before falling through to the legacy path.
                        // Abandoning `inner` here would leak both the backend
                        // buffer and the whole-file mapping registered for it,
                        // and the failure that reaches this branch is a
                        // truncated or crafted GGUF — attacker-reachable.
                        release_weight_buffer(out.buf);
                    } else {
                        for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
                            out.tensors[ggml_get_name(t)] = t;
                            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
                            if (tid < 0)
                                continue;
                            const size_t off = gguf_get_tensor_offset(gctx, tid);
                            ggml_backend_tensor_alloc(out.buf, t, tensor_base + off);
                        }

                        gguf_free(gctx);
                        return true;
                    }
                }
                // buffer_from_host_ptr returned null — release mmap and fall
                // through to the legacy path. No warning; same rationale as
                // the CPU mmap branch.
            }
        }
    }

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        // Almost always the weights simply don't fit in VRAM. The bare message
        // sent people hunting for a ggml bug; say how much was wanted and what
        // to do about it. (KugelAudio's F16 asks 17.3 GB — a hard stop on any
        // 16 GB card, with nothing in the old message to suggest a smaller
        // quant would work.)
        size_t wanted = 0;
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t))
            wanted += ggml_backend_buft_get_alloc_size(buft, t);
        fprintf(stderr,
                "%s: failed to allocate %.2f GB on '%s' for the model weights.\n"
                "  If this is a GPU, the model does not fit in VRAM. Try a smaller quantization\n"
                "  (--model-quant q4_k), or run on CPU with --no-gpu.\n",
                tag, (double)wanted / 1e9, ggml_backend_name(backend));
        gguf_free(gctx);
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    MappedFile mf(path);
    if (mf.ok) {
        // Issue #94: the legacy alloc+copy path took 30-60 s for the
        // 658 MB chatterbox-turbo T3 GGUF on slow disks, because each
        // ggml_backend_tensor_set hit a synchronous page fault on
        // its first access to the mmap source region. The zero-copy
        // mmap path already hints WILLNEED (line 526) for the same
        // reason; doing it here brings the legacy path's load time
        // back in line for users who opt out of the zero-copy path
        // with CRISPASR_GGUF_MMAP=0 (e.g. model files on network
        // mounts where mmap would SIGBUS on disconnect).
#if !defined(_WIN32)
        if (!include_tensor)
            ::posix_madvise(mf.base, mf.size, POSIX_MADV_WILLNEED);
#endif
    }
    if (!mf.ok) {
        // Fallback: read via FILE* pread/fseek. This is the rare path —
        // most systems have working mmap. We implement it inline here so
        // models don't have to.
        FILE* fp = fopen(path, "rb");
        if (!fp) {
            fprintf(stderr, "%s: cannot open '%s' for fread fallback\n", tag, path);
            gguf_free(gctx);
            release_weight_buffer(out.buf);
            ggml_free(out.ctx);
            out.ctx = nullptr;
            return false;
        }
        const size_t data_off = gguf_get_data_offset(gctx);
        std::vector<uint8_t> tbuf;
        bool load_ok = true;
        for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
            out.tensors[ggml_get_name(t)] = t;
            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0)
                continue;
            const size_t off = gguf_get_tensor_offset(gctx, tid);
            const size_t nbytes = ggml_nbytes(t);
            if (tbuf.size() < nbytes)
                tbuf.resize(nbytes);
#if defined(_WIN32)
            if (_fseeki64(fp, (int64_t)(data_off + off), SEEK_SET) != 0) {
                fprintf(stderr, "%s: fseek failed for tensor '%s' — file truncated?\n", tag, ggml_get_name(t));
                load_ok = false;
                break;
            }
#else
            if (fseeko(fp, (off_t)(data_off + off), SEEK_SET) != 0) {
                fprintf(stderr, "%s: fseek failed for tensor '%s' — file truncated?\n", tag, ggml_get_name(t));
                load_ok = false;
                break;
            }
#endif
            if (fread(tbuf.data(), 1, nbytes, fp) != nbytes) {
                fprintf(stderr, "%s: short read for tensor '%s' (%zu bytes expected) — file truncated?\n", tag,
                        ggml_get_name(t), nbytes);
                load_ok = false;
                break;
            }
            ggml_backend_tensor_set(t, tbuf.data(), 0, nbytes);
        }
        fclose(fp);
        if (!load_ok) {
            fprintf(stderr, "%s: legacy loader failed — model file may be truncated or corrupt\n", tag);
            gguf_free(gctx);
            release_weight_buffer(out.buf);
            ggml_free(out.ctx);
            out.ctx = nullptr;
            return false;
        }
    } else {
        const size_t data_off = gguf_get_data_offset(gctx);
        for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
            out.tensors[ggml_get_name(t)] = t;
            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0)
                continue;
            const size_t off = gguf_get_tensor_offset(gctx, tid);
            const size_t nbytes = ggml_nbytes(t);
            // Bounds check: prevent segfault on truncated GGUF files
            // Overflow-safe: a crafted GGUF can set off/nbytes near SIZE_MAX so the
            // additive data_off+off+nbytes wraps and passes the check. Compare
            // subtractively (each subtraction is valid once the prior guard holds).
            if (data_off > mf.size || off > mf.size - data_off || nbytes > mf.size - data_off - off) {
                fprintf(stderr,
                        "%s: mmap legacy path: tensor '%s' exceeds file bounds "
                        "(off=%zu + nbytes=%zu > file_size=%zu) — file truncated?\n",
                        tag, ggml_get_name(t), data_off + off, nbytes, mf.size);
                gguf_free(gctx);
                release_weight_buffer(out.buf);
                ggml_free(out.ctx);
                out.ctx = nullptr;
                return false;
            }
            ggml_backend_tensor_set(t, (const char*)mf.base + data_off + off, 0, nbytes);
        }
    }

    gguf_free(gctx);
    return true;
}

bool load_weights(const char* path, ggml_backend_t backend, const char* model_tag, WeightLoad& out) {
    // A GPU backend that cannot bind some weights (Vulkan: a tensor past
    // maxStorageBufferRange - allocatable, not bindable) would otherwise load
    // "fine" and abort later in ggml_backend_sched. Those weights go to a CPU
    // buffer instead (see load_weights_fit). Callers of load_weights own only
    // out.buf, so the CPU partition is registered as an extra of it and is
    // released with it by release_weight_buffer(). CRISPASR_GGUF_NO_FIT=1 turns
    // this off. A no-op on CUDA / Metal, whose supports_op accepts any leaf.
    if (backend && !core_cpu_backend::is_cpu(backend) && !std::getenv("CRISPASR_GGUF_NO_FIT")) {
        FitSet fit = scan_unbindable(path, backend);
        if (!fit.cpu.empty()) {
            const char* tag = model_tag ? model_tag : "core_gguf";
            log_unbindable(tag, fit, backend);
            ggml_backend_t cpu = core_cpu_backend::init();
            const bool ok = cpu && load_weights_split(path, backend, cpu, fit_is_gpu, &fit, model_tag, out);
            if (cpu)
                ggml_backend_free(cpu); // the CPU buffers do not depend on the backend instance
            if (ok && out.buf_cpu) {
                if (out.buf) {
                    std::vector<ggml_backend_buffer_t> owned = take_split_extra(out.buf_cpu);
                    owned.insert(owned.begin(), out.buf_cpu);
                    register_split_extra(out.buf, owned);
                } else {
                    out.buf = out.buf_cpu; // every tensor was unbindable
                }
                out.buf_cpu = nullptr;
            }
            return ok;
        }
    }
    return load_weights_impl(path, backend, nullptr, nullptr, model_tag, out);
}

bool load_weights_filtered(const char* path, ggml_backend_t backend, IncludeTensor include_tensor, void* user,
                           const char* model_tag, WeightLoad& out) {
    if (!include_tensor) {
        fprintf(stderr, "%s: load_weights_filtered requires a non-null tensor predicate\n",
                model_tag ? model_tag : "core_gguf");
        return false;
    }
    return load_weights_impl(path, backend, include_tensor, user, model_tag, out);
}

void release_weight_buffer(ggml_backend_buffer_t& buf) {
    if (!buf)
        return;
    // Take the entry before the free, not after: between a free and a later
    // erase, a concurrent load could receive a new buffer at the same address
    // and register it, and the erase would then drop a live mapping's record.
    const gpu_mmap_handle h = take_gpu_mmap(buf);
    // Overflow chunks of the same partition (issue #276) are released with it.
    for (ggml_backend_buffer_t extra : take_split_extra(buf))
        ggml_backend_buffer_free(extra);
    // Free the backend buffer first. Metal's shared-storage MTLBuffer is a
    // view onto these pages, so unmapping them while the buffer is alive would
    // leave the GPU addressing unmapped memory. Freeing first inherits ggml's
    // existing caller contract — ggml_metal_buffer_free already vm_deallocates
    // the host pages of a buffer it owns, so "no work referencing this buffer
    // may still be in flight" is a precondition every caller already meets.
    ggml_backend_buffer_free(buf);
    buf = nullptr;
    unmap_region(h.base, h.size);
}

void free_weights(WeightLoad& wl) {
    // Issue #276: the overflow chunks in split_bufs are released with the
    // primary buffer of their partition, so freeing them again here would be
    // a double free. Clearing the vector drops the now-dangling handles.
    release_weight_buffer(wl.buf);
    release_weight_buffer(wl.buf_cpu);
    wl.split_bufs.clear();
    if (wl.ctx) {
        ggml_free(wl.ctx);
        wl.ctx = nullptr;
    }
    wl.tensors.clear();
}

int blk_layer_of_with_prefix(const char* tensor_name, const char* prefix) {
    if (!tensor_name || !prefix)
        return -1;
    const size_t plen = std::strlen(prefix);
    if (std::strncmp(tensor_name, prefix, plen) != 0)
        return -1;
    char* end = nullptr;
    long il = std::strtol(tensor_name + plen, &end, 10);
    if (!end || *end != '.' || il < 0)
        return -1;
    return (int)il;
}

int blk_layer_of(const char* tensor_name) {
    return blk_layer_of_with_prefix(tensor_name, "blk.");
}

bool is_gpu_tensor_with_prefix(const char* tensor_name, void* user) {
    const auto* cfg = static_cast<const LayerSplitConfig*>(user);
    const int il = blk_layer_of_with_prefix(tensor_name, cfg->prefix);
    if (il < 0)
        return true; // non-layered tensors stay on GPU
    return il < cfg->threshold;
}

bool is_gpu_tensor_blk(const char* tensor_name, void* user) {
    const int threshold = *static_cast<const int*>(user);
    const int il = blk_layer_of(tensor_name);
    if (il < 0)
        return true;
    return il < threshold;
}

bool load_weights_fit(const char* path, ggml_backend_t backend, ggml_backend_t cpu_backend, const char* model_tag,
                      WeightLoad& out, int* n_offloaded) {
    const char* tag = model_tag ? model_tag : "core_gguf";
    if (n_offloaded)
        *n_offloaded = 0;
    if (!backend || !cpu_backend || backend == cpu_backend)
        return load_weights_impl(path, backend ? backend : cpu_backend, nullptr, nullptr, model_tag, out);
    const FitSet fit = scan_unbindable(path, backend);
    if (fit.cpu.empty())
        return load_weights_impl(path, backend, nullptr, nullptr, model_tag, out);
    log_unbindable(tag, fit, backend);
    if (n_offloaded)
        *n_offloaded = (int)fit.cpu.size();
    FitSet f = fit;
    return load_weights_split(path, backend, cpu_backend, fit_is_gpu, &f, model_tag, out);
}

bool load_weights_split(const char* path, ggml_backend_t gpu_backend, ggml_backend_t cpu_backend, IsGpuTensor is_gpu,
                        void* user, const char* model_tag, WeightLoad& out) {
    const char* tag = model_tag ? model_tag : "core_gguf";

    if (!gpu_backend || !cpu_backend) {
        fprintf(stderr, "%s: load_weights_split requires both gpu and cpu backends\n", tag);
        return false;
    }
    if (!is_gpu) {
        fprintf(stderr, "%s: load_weights_split requires a non-null is_gpu predicate\n", tag);
        return false;
    }

    // Open metadata + create ggml_context with all tensor metadata (no_alloc).
    gguf_init_params gp = {/*.no_alloc=*/true, /*.ctx=*/&out.ctx};
    gguf_context* gctx = gguf_init_from_file(path, gp);
    if (!gctx || !out.ctx) {
        fprintf(stderr, "%s: failed to load tensor metadata from '%s'\n", tag, path);
        if (gctx)
            gguf_free(gctx);
        return false;
    }

    // Pass 1: partition tensors by predicate, sum sizes per partition.
    std::vector<ggml_tensor*> gpu_tensors, cpu_tensors;
    size_t gpu_size = 0, cpu_size = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
        const char* tname = ggml_get_name(t);
        const bool to_gpu = is_gpu(tname, user);
        if (to_gpu) {
            gpu_tensors.push_back(t);
            gpu_size += ggml_nbytes(t);
        } else {
            cpu_tensors.push_back(t);
            cpu_size += ggml_nbytes(t);
        }
        out.tensors[tname] = t;
    }

    // Allocate per-partition backend buffers. Tensor alignment within the
    // buffer follows the backend buffer-type's alignment requirement;
    // pad each per-tensor offset up to that alignment.
    //
    // Issue #276: AMD Vulkan (proprietary driver on Windows) caps a single
    // device allocation at 2 GiB (maxMemoryAllocationSize). Models larger
    // than that need to be split across multiple backend buffers. We chunk
    // tensors into groups of <= 1.5 GiB each and allocate one buffer per
    // chunk; the 1.5 GiB limit leaves headroom for alignment padding.
    //
    // CRISPASR_GGUF_MAX_ALLOC_CHUNK (bytes) lowers the limit. A driver with a
    // tighter cap than AMD's is the field use; the test use is that reaching
    // the chunked path otherwise costs a multi-gigabyte allocation, so without
    // this the branch that produces overflow buffers has no coverage at all.
    size_t max_alloc_chunk = (size_t)1536 * 1024 * 1024; // 1.5 GiB
    if (const char* v = std::getenv("CRISPASR_GGUF_MAX_ALLOC_CHUNK")) {
        const long long parsed = std::atoll(v);
        if (parsed > 0)
            max_alloc_chunk = (size_t)parsed;
    }

    auto round_up = [](size_t n, size_t a) { return (n + a - 1) & ~(a - 1); };
    auto bind_partition = [&](ggml_backend_t be, const std::vector<ggml_tensor*>& tensors,
                              std::vector<ggml_backend_buffer_t>& out_bufs) -> bool {
        if (tensors.empty())
            return true;
        const size_t align = ggml_backend_get_alignment(be);

        // Size tensors by the BUFFER TYPE's alloc size, not ggml_nbytes().
        //
        // ggml_backend_tensor_alloc() asserts that
        //   addr + ggml_backend_buffer_get_alloc_size(buf, t) <= base + buf_size
        // and CUDA's alloc size EXCEEDS ggml_nbytes for quantized tensors — it
        // pads each row up to MATRIX_ROW_PADDING so MMQ can over-read safely.
        // Sizing the buffer with ggml_nbytes therefore left the last tensor of
        // every chunk short by that padding, and the write went past the end.
        //
        // ggml only started asserting this in v0.17, so the overrun was silent
        // before: it surfaced as moonshine-tiny (q4_k) aborting at load on CUDA
        // while passing on Metal, whose alloc size equals ggml_nbytes.
        // Everything using load_weights_split with quantized weights on CUDA was
        // affected, not just moonshine.
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(be);
        auto tensor_size = [&](const ggml_tensor* t) { return ggml_backend_buft_get_alloc_size(buft, t); };

        // Partition tensors into chunks that each fit under max_alloc_chunk.
        struct Chunk {
            std::vector<ggml_tensor*> ts;
            size_t aligned_total = 0;
        };
        std::vector<Chunk> chunks(1);
        for (ggml_tensor* t : tensors) {
            const size_t nb = tensor_size(t);
            const size_t next = round_up(chunks.back().aligned_total, align) + nb;
            if (next > max_alloc_chunk && !chunks.back().ts.empty()) {
                // Start a new chunk.
                chunks.push_back({});
                chunks.back().ts.push_back(t);
                chunks.back().aligned_total = nb;
            } else {
                chunks.back().ts.push_back(t);
                chunks.back().aligned_total = next;
            }
        }

        for (auto& chunk : chunks) {
            ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(be, chunk.aligned_total);
            if (!buf) {
                fprintf(stderr, "%s: failed to allocate %zu MiB backend buffer\n", tag, chunk.aligned_total / 1048576);
                for (auto& b : out_bufs)
                    release_weight_buffer(b);
                out_bufs.clear();
                return false;
            }
            char* base = (char*)ggml_backend_buffer_get_base(buf);
            size_t cursor = 0;
            for (ggml_tensor* t : chunk.ts) {
                cursor = round_up(cursor, align);
                ggml_backend_tensor_alloc(buf, t, base + cursor);
                cursor += tensor_size(t);
            }
            out_bufs.push_back(buf);
        }
        return true;
    };

    std::vector<ggml_backend_buffer_t> gpu_bufs, cpu_bufs;
    if (!bind_partition(gpu_backend, gpu_tensors, gpu_bufs)) {
        gguf_free(gctx);
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }
    if (!bind_partition(cpu_backend, cpu_tensors, cpu_bufs)) {
        for (auto& b : gpu_bufs)
            release_weight_buffer(b);
        gguf_free(gctx);
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    // First buffer of each partition goes into the canonical fields; the
    // overflow chunks are owned by this loader and released with the primary
    // buffer of their own partition. split_bufs lists them for inspection.
    if (!gpu_bufs.empty()) {
        out.buf = gpu_bufs[0];
        const std::vector<ggml_backend_buffer_t> extra(gpu_bufs.begin() + 1, gpu_bufs.end());
        register_split_extra(out.buf, extra);
        out.split_bufs.insert(out.split_bufs.end(), extra.begin(), extra.end());
    }
    if (!cpu_bufs.empty()) {
        out.buf_cpu = cpu_bufs[0];
        const std::vector<ggml_backend_buffer_t> extra(cpu_bufs.begin() + 1, cpu_bufs.end());
        register_split_extra(out.buf_cpu, extra);
        out.split_bufs.insert(out.split_bufs.end(), extra.begin(), extra.end());
    }

    // Copy tensor data from the file. Use mmap when available for zero-
    // copy where the kernel will demand-page; fall back to pread.
    MappedFile mf(path);
    if (!mf.ok) {
        FILE* fp = fopen(path, "rb");
        if (!fp) {
            fprintf(stderr, "%s: cannot open '%s' for fread fallback\n", tag, path);
            free_weights(out);
            gguf_free(gctx);
            return false;
        }
        const size_t data_off = gguf_get_data_offset(gctx);
        std::vector<uint8_t> tbuf;
        for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0)
                continue;
            const size_t off = gguf_get_tensor_offset(gctx, tid);
            const size_t nbytes = ggml_nbytes(t);
            if (tbuf.size() < nbytes)
                tbuf.resize(nbytes);
#if defined(_WIN32)
            if (_fseeki64(fp, (int64_t)(data_off + off), SEEK_SET) != 0)
                break;
#else
            if (fseeko(fp, (off_t)(data_off + off), SEEK_SET) != 0)
                break;
#endif
            if (fread(tbuf.data(), 1, nbytes, fp) != nbytes)
                break;
            ggml_backend_tensor_set(t, tbuf.data(), 0, nbytes);
        }
        fclose(fp);
    } else {
        const size_t data_off = gguf_get_data_offset(gctx);
        for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0)
                continue;
            const size_t off = gguf_get_tensor_offset(gctx, tid);
            const size_t nbytes = ggml_nbytes(t);
            // Bounds check (mirrors load_weights' mmap paths): a crafted/truncated
            // GGUF whose tensor offset+size runs past the mapping would SIGBUS, or
            // copy adjacent memory into the tensor buffer (info leak). This split
            // mmap path was the only one of the four missing the guard.
            // Overflow-safe: a crafted GGUF can set off/nbytes near SIZE_MAX so the
            // additive data_off+off+nbytes wraps and passes the check. Compare
            // subtractively (each subtraction is valid once the prior guard holds).
            if (data_off > mf.size || off > mf.size - data_off || nbytes > mf.size - data_off - off) {
                fprintf(stderr,
                        "%s: split mmap path: tensor '%s' exceeds file bounds "
                        "(off=%zu + nbytes=%zu > file_size=%zu) — file truncated?\n",
                        tag, ggml_get_name(t), data_off + off, nbytes, mf.size);
                free_weights(out);
                gguf_free(gctx);
                return false;
            }
            ggml_backend_tensor_set(t, (const char*)mf.base + data_off + off, 0, nbytes);
        }
    }

    fprintf(stderr, "%s: weight residency: gpu=%zu MiB (%zu tensors), cpu=%zu MiB (%zu tensors)\n", tag,
            gpu_size / 1048576, gpu_tensors.size(), cpu_size / 1048576, cpu_tensors.size());

    gguf_free(gctx);
    return true;
}

// ---------------------------------------------------------------------------
// ggml CPU repack buffer type (docs/ggml-optimisation-playbook.md §4)
// ---------------------------------------------------------------------------

// The CPU device's extra buffer types, or an empty list. Cached: the ggml side
// builds them once in a function-local static anyway.
static const std::vector<ggml_backend_buffer_type_t>& cpu_extra_bufts() {
    static const std::vector<ggml_backend_buffer_type_t> bufts = [] {
        std::vector<ggml_backend_buffer_type_t> v;
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!dev)
            return v;
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (!reg)
            return v;
        auto fn = (ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_dev_get_extra_bufts");
        if (!fn)
            return v;
        for (ggml_backend_buffer_type_t* b = fn(dev); b && *b; ++b)
            v.push_back(*b);
        return v;
    }();
    return bufts;
}

// Pick the extra buffer type to use.
//
// ⚠ NOT v[0]. The CPU device can offer more than one extra buffer type, and
// the order is fixed by ggml_backend_cpu_get_extra_buffer_types()
// (ggml-cpu.cpp:42), which pushes AMX FIRST when the build has
// __AMX_INT8__ && __AVX512VNNI__ — i.e. on any GGML_NATIVE build on Sapphire
// Rapids or Emerald Rapids. Taking v[0] there silently selects AMX instead of
// repack. That is not a hypothetical: GitHub's ubuntu-24.04 pool is
// heterogeneous and hands out both AMD EPYC 7763 (AVX2, no VNNI) and Intel
// Xeon Platinum 8573C (AVX-512 VNNI + AMX-INT8), and on the latter the first
// version of this function selected AMX, repacked q8_0 on x86 — which has no
// q8_0 *repack* kernel at all — and measured hFT 1.4–1.5× SLOWER than f32.
//
// So select by name, and let the name be overridden for A/B work.
static ggml_backend_buffer_type_t repack_buft() {
    static ggml_backend_buffer_type_t chosen = []() -> ggml_backend_buffer_type_t {
        const char* want = std::getenv("CRISPASR_GGUF_EXTRA_BUFT");
        const char* name = (want && *want) ? want : "CPU_REPACK";
        for (auto* b : cpu_extra_bufts()) {
            const char* n = ggml_backend_buft_name(b);
            if (n && std::strcmp(n, name) == 0)
                return b;
        }
        return nullptr;
    }();
    return chosen;
}

// Opt-out, so a field failure can be bisected without a rebuild.
static bool repack_enabled() {
    const char* v = std::getenv("CRISPASR_GGUF_REPACK");
    return !(v && (v[0] == '0' || v[0] == 'n' || v[0] == 'N'));
}

bool repack_buft_accepts(ggml_type type, int64_t ne0, int64_t ne1) {
    ggml_backend_buffer_type_t buft = repack_buft();
    if (!buft || ne0 <= 0 || ne1 <= 0)
        return false;

    // Ask ggml rather than reimplementing ggml_repack_get_optimal_repack_type,
    // whose table is ISA- and shape-dependent and changes upstream. The buffer
    // type's init_tensor sets tensor->extra to the chosen traits, or leaves it
    // null when it has no kernel. Costs one tensor's worth of memory, briefly.
    ggml_init_params ip = {/*.mem_size=*/ggml_tensor_overhead() * 2, /*.mem_buffer=*/nullptr, /*.no_alloc=*/true};
    ggml_context* c = ggml_init(ip);
    if (!c)
        return false;
    ggml_tensor* t = ggml_new_tensor_2d(c, type, ne0, ne1);
    bool ok = false;
    if (t) {
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(c, buft);
        if (b) {
            ok = (t->extra != nullptr);
            ggml_backend_buffer_free(b);
        }
    }
    ggml_free(c);
    return ok;
}

bool load_weights_repack(const char* path, ggml_backend_t cpu_backend, IsMatmulWeight is_matmul_weight, void* user,
                         const char* model_tag, WeightLoad& out, int* n_repacked) {
    const char* tag = model_tag ? model_tag : "core_gguf";
    if (n_repacked)
        *n_repacked = 0;

    if (!cpu_backend) {
        fprintf(stderr, "%s: load_weights_repack requires a backend\n", tag);
        return false;
    }
    // Repack is a CPU-backend concept. On anything else, and when the host
    // offers no extra buffer type, fall straight back to the normal loader so
    // callers can adopt this unconditionally.
    if (!is_matmul_weight || !repack_enabled() || !repack_buft() || !core_cpu_backend::is_cpu(cpu_backend))
        return load_weights(path, cpu_backend, model_tag, out);

    gguf_init_params gp = {/*.no_alloc=*/true, /*.ctx=*/&out.ctx};
    gguf_context* gctx = gguf_init_from_file(path, gp);
    if (!gctx || !out.ctx) {
        fprintf(stderr, "%s: failed to load tensor metadata from '%s'\n", tag, path);
        if (gctx)
            gguf_free(gctx);
        return false;
    }

    ggml_backend_buffer_type_t rbuft = repack_buft();
    ggml_backend_buffer_type_t dbuft = ggml_backend_get_default_buffer_type(cpu_backend);

    // Pass 1: partition. A tensor is repacked only if the MODEL says it is a
    // matmul weight (the loader cannot know) AND ggml says it has a kernel for
    // that type/shape on this host (repack's set_tensor would otherwise null-
    // dereference). Cache the acceptance answer per (type, ne0, ne1) — a
    // transformer has many identically shaped weights and each probe costs an
    // allocation.
    std::vector<ggml_tensor*> rep_tensors, def_tensors;
    size_t rep_size = 0, def_size = 0;
    std::map<std::tuple<int, int64_t, int64_t>, bool> accept_cache;
    for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
        const char* tname = ggml_get_name(t);
        bool to_repack = false;
        if (ggml_n_dims(t) == 2 && is_matmul_weight(tname, user)) {
            const auto key = std::make_tuple((int)t->type, t->ne[0], t->ne[1]);
            auto it = accept_cache.find(key);
            if (it == accept_cache.end())
                it = accept_cache.emplace(key, repack_buft_accepts(t->type, t->ne[0], t->ne[1])).first;
            to_repack = it->second;
        }
        if (to_repack) {
            rep_tensors.push_back(t);
            rep_size += ggml_nbytes(t);
        } else {
            def_tensors.push_back(t);
            def_size += ggml_nbytes(t);
        }
        out.tensors[tname] = t;
    }

    if (rep_tensors.empty()) {
        // Nothing to gain. Throw the partition away and take the normal path,
        // which keeps the zero-copy mmap this one would have cost.
        gguf_free(gctx);
        ggml_free(out.ctx);
        out.ctx = nullptr;
        out.tensors.clear();
        return load_weights(path, cpu_backend, model_tag, out);
    }

    auto round_up = [](size_t n, size_t a) { return (n + a - 1) & ~(a - 1); };
    auto bind_partition = [&](ggml_backend_buffer_type_t buft, const std::vector<ggml_tensor*>& tensors,
                              ggml_backend_buffer_t& out_buf) -> bool {
        if (tensors.empty())
            return true;
        const size_t align = ggml_backend_buft_get_alignment(buft);
        size_t total = 0;
        for (ggml_tensor* t : tensors)
            total = round_up(total, align) + ggml_backend_buft_get_alloc_size(buft, t);
        out_buf = ggml_backend_buft_alloc_buffer(buft, total);
        if (!out_buf) {
            fprintf(stderr, "%s: failed to allocate %zu MiB %s buffer\n", tag, total / 1048576,
                    ggml_backend_buft_name(buft));
            return false;
        }
        char* base = (char*)ggml_backend_buffer_get_base(out_buf);
        size_t cursor = 0;
        for (ggml_tensor* t : tensors) {
            cursor = round_up(cursor, align);
            ggml_backend_tensor_alloc(out_buf, t, base + cursor);
            cursor += ggml_backend_buft_get_alloc_size(buft, t);
        }
        return true;
    };

    if (!bind_partition(rbuft, rep_tensors, out.buf) || !bind_partition(dbuft, def_tensors, out.buf_cpu)) {
        free_weights(out);
        gguf_free(gctx);
        return false;
    }

    // Copy. Every tensor goes through ggml_backend_tensor_set, which is a
    // memcpy for the default partition and the repack rewrite for the other.
    // There is no zero-copy variant of that second one — see the header.
    MappedFile mf(path);
    const size_t data_off = gguf_get_data_offset(gctx);
    if (!mf.ok) {
        FILE* fp = fopen(path, "rb");
        if (!fp) {
            fprintf(stderr, "%s: cannot open '%s' for fread fallback\n", tag, path);
            free_weights(out);
            gguf_free(gctx);
            return false;
        }
        std::vector<uint8_t> tbuf;
        for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0)
                continue;
            const size_t off = gguf_get_tensor_offset(gctx, tid);
            const size_t nbytes = ggml_nbytes(t);
            if (tbuf.size() < nbytes)
                tbuf.resize(nbytes);
#if defined(_WIN32)
            if (_fseeki64(fp, (int64_t)(data_off + off), SEEK_SET) != 0)
                break;
#else
            if (fseeko(fp, (off_t)(data_off + off), SEEK_SET) != 0)
                break;
#endif
            if (fread(tbuf.data(), 1, nbytes, fp) != nbytes)
                break;
            ggml_backend_tensor_set(t, tbuf.data(), 0, nbytes);
        }
        fclose(fp);
    } else {
        for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0)
                continue;
            const size_t off = gguf_get_tensor_offset(gctx, tid);
            const size_t nbytes = ggml_nbytes(t);
            // Same overflow-safe bounds check as the split mmap path.
            if (data_off > mf.size || off > mf.size - data_off || nbytes > mf.size - data_off - off) {
                fprintf(stderr,
                        "%s: repack path: tensor '%s' exceeds file bounds "
                        "(off=%zu + nbytes=%zu > file_size=%zu) — file truncated?\n",
                        tag, ggml_get_name(t), data_off + off, nbytes, mf.size);
                free_weights(out);
                gguf_free(gctx);
                return false;
            }
            ggml_backend_tensor_set(t, (const char*)mf.base + data_off + off, 0, nbytes);
        }
    }

    if (n_repacked)
        *n_repacked = (int)rep_tensors.size();
    // Name the buffer type. An A/B that does not say which extra buffer type
    // it selected is not reproducible — see the note on repack_buft().
    fprintf(stderr, "%s: extra buffer type '%s': %zu MiB (%zu tensors) repacked, %zu MiB (%zu tensors) default\n", tag,
            ggml_backend_buft_name(rbuft), rep_size / 1048576, rep_tensors.size(), def_size / 1048576,
            def_tensors.size());

    gguf_free(gctx);
    return true;
}

// PLAN #60g: switch a previously-WILLNEED-hinted region to MADV_RANDOM.
// Used by callers (e.g., mimo-asr's transcribe loop) to tell the kernel
// "I'm done with sequential prefill access; my next reads will be
// random-order layer revisits during decode — please stop wasting IO
// on speculative readahead."
//
// We dispatch on the buffer's iface fields to detect which of our two
// mmap paths the buffer came from, since the buffer types themselves
// aren't exposed publicly. No-op if the buffer wasn't allocated through
// either path (incl. the legacy alloc+copy fallback when MMAP=0 or
// mmap failed).
void mmap_advise_random(ggml_backend_buffer_t buf) {
#if !defined(_WIN32)
    if (!buf)
        return;
    void* base = nullptr;
    size_t size = 0;
    if (buf->iface.free_buffer == mmap_buffer_free) {
        // CPU mmap path — context is mmap_buffer_ctx (our own iface).
        auto* mctx = (mmap_buffer_ctx*)buf->context;
        base = mctx->mmap_base;
        size = mctx->mmap_size;
    } else {
        // Non-CPU (Metal) mmap path — buffer is the inner backend buffer,
        // its context belongs to that backend; look the mmap region up
        // in our side-map instead.
        gpu_mmap_handle h = lookup_gpu_mmap(buf);
        base = h.base;
        size = h.size;
    }
    if (base && size > 0)
        ::posix_madvise(base, size, POSIX_MADV_RANDOM);
#else
    (void)buf;
#endif
}

// ---------------------------------------------------------------------------
// Tensor lookup helpers
// ---------------------------------------------------------------------------

// Signatures use `core_gguf::tensor_map` (see gguf_loader.h cross-repo contract).
ggml_tensor* try_get(const tensor_map& tensors, const char* name) {
    auto it = tensors.find(name);
    return it != tensors.end() ? it->second : nullptr;
}

ggml_tensor* require(const tensor_map& tensors, const char* name, const char* model_tag) {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        fprintf(stderr, "%s: required tensor '%s' not found in GGUF\n", model_tag ? model_tag : "core_gguf", name);
        return nullptr;
    }
    return it->second;
}

std::string format_layer_name(const char* fmt, int i) {
    char buf[256];
    snprintf(buf, sizeof(buf), fmt, i);
    return std::string(buf);
}

std::string format_layer_name(const char* fmt, int i, int j) {
    char buf[256];
    snprintf(buf, sizeof(buf), fmt, i, j);
    return std::string(buf);
}

} // namespace core_gguf
