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

#if defined(_WIN32)
#include <io.h>
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
// We reuse ggml_backend_cpu_buffer_type() so ggml_backend_buffer_is_host()
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

// PLAN #51a (Metal variant): wrap a non-CPU backend buffer (built via the
// device's `buffer_from_host_ptr` capability) so its `free_buffer` callback
// also munmaps the host memory the inner buffer was constructed from.
//
// Why a wrapper instead of modifying the inner iface: `ggml_backend_buffer_i`
// is a `static const` table inside each backend (ggml-metal.cpp etc.); we
// can't override per-instance. And we can't rely on the inner buffer to
// munmap (Metal's `newBufferWithBytesNoCopy:length:options:deallocator:nil`
// explicitly takes no ownership — `ggml_metal_buffer_free` only frees the
// MTLBuffer ref, leaving the host pages mapped).
//
// All non-`free_buffer` ops delegate straight to the inner iface so Metal's
// shared-buffer semantics (storage-mode-shared, Apple-Silicon unified memory
// sharing) are preserved.
struct mmap_wrap_ctx {
    ggml_backend_buffer_t inner = nullptr;
    void* mmap_base = nullptr;
    size_t mmap_size = 0;
};

static void* mmap_wrap_get_base(ggml_backend_buffer_t buffer) {
    return ggml_backend_buffer_get_base(((mmap_wrap_ctx*)buffer->context)->inner);
}
static enum ggml_status mmap_wrap_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor) {
    auto* w = (mmap_wrap_ctx*)buffer->context;
    if (w->inner->iface.init_tensor)
        return w->inner->iface.init_tensor(w->inner, tensor);
    return GGML_STATUS_SUCCESS;
}
static void mmap_wrap_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor, uint8_t value, size_t offset,
                                    size_t size) {
    auto* w = (mmap_wrap_ctx*)buffer->context;
    w->inner->iface.memset_tensor(w->inner, tensor, value, offset, size);
}
static void mmap_wrap_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor, const void* data, size_t offset,
                                 size_t size) {
    auto* w = (mmap_wrap_ctx*)buffer->context;
    w->inner->iface.set_tensor(w->inner, tensor, data, offset, size);
}
static void mmap_wrap_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor* tensor, void* data, size_t offset,
                                 size_t size) {
    auto* w = (mmap_wrap_ctx*)buffer->context;
    w->inner->iface.get_tensor(w->inner, tensor, data, offset, size);
}
static bool mmap_wrap_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor* src, ggml_tensor* dst) {
    auto* w = (mmap_wrap_ctx*)buffer->context;
    if (w->inner->iface.cpy_tensor)
        return w->inner->iface.cpy_tensor(w->inner, src, dst);
    return false;
}
static void mmap_wrap_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto* w = (mmap_wrap_ctx*)buffer->context;
    w->inner->iface.clear(w->inner, value);
}
static void mmap_wrap_free(ggml_backend_buffer_t buffer) {
    auto* w = (mmap_wrap_ctx*)buffer->context;
    // Free the inner backend buffer first — for Metal this releases the
    // MTLBuffer reference that newBufferWithBytesNoCopy held over our
    // mmap region. Once that ref is gone, the OS is free to release any
    // page-table entries the GPU was holding.
    ggml_backend_buffer_free(w->inner);
    if (w->mmap_base) {
#if defined(_WIN32)
        UnmapViewOfFile(w->mmap_base);
#else
        ::munmap(w->mmap_base, w->mmap_size);
#endif
    }
    delete w;
}

// PLAN #60b: forward-compat delegations for set_tensor_2d / get_tensor_2d
// / reset. If a future ggml dispatch hits these on a wrapped buffer,
// route through to the inner Metal buffer's optimized path instead of
// silently falling back to N×set_tensor calls. Inner iface methods are
// optional — guard each delegation with a null check; on null we leave
// the wrapper's slot null too so the scheduler picks its own fallback
// (matches what would happen pre-wrap).
static void mmap_wrap_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor* tensor, const void* data, size_t offset,
                                    size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    auto* w = (mmap_wrap_ctx*)buffer->context;
    if (w->inner->iface.set_tensor_2d)
        w->inner->iface.set_tensor_2d(w->inner, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}
static void mmap_wrap_get_tensor_2d(ggml_backend_buffer_t buffer, const ggml_tensor* tensor, void* data, size_t offset,
                                    size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    auto* w = (mmap_wrap_ctx*)buffer->context;
    if (w->inner->iface.get_tensor_2d)
        w->inner->iface.get_tensor_2d(w->inner, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}
static void mmap_wrap_reset(ggml_backend_buffer_t buffer) {
    auto* w = (mmap_wrap_ctx*)buffer->context;
    if (w->inner->iface.reset)
        w->inner->iface.reset(w->inner);
}

static const ggml_backend_buffer_i mmap_wrap_iface = {
    /* .free_buffer    = */ mmap_wrap_free,
    /* .get_base       = */ mmap_wrap_get_base,
    /* .init_tensor    = */ mmap_wrap_init_tensor,
    /* .memset_tensor  = */ mmap_wrap_memset_tensor,
    /* .set_tensor     = */ mmap_wrap_set_tensor,
    /* .get_tensor     = */ mmap_wrap_get_tensor,
    /* .set_tensor_2d  = */ mmap_wrap_set_tensor_2d,
    /* .get_tensor_2d  = */ mmap_wrap_get_tensor_2d,
    /* .cpy_tensor     = */ mmap_wrap_cpy_tensor,
    /* .clear          = */ mmap_wrap_clear,
    /* .reset          = */ mmap_wrap_reset,
};

static bool mmap_loader_enabled() {
    const char* v = std::getenv("CRISPASR_GGUF_MMAP");
    return v && *v && *v != '0';
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

bool load_weights(const char* path, ggml_backend_t backend, const char* model_tag, WeightLoad& out) {
    const char* tag = model_tag ? model_tag : "core_gguf";

    gguf_init_params gp = {/*.no_alloc=*/true, /*.ctx=*/&out.ctx};
    gguf_context* gctx = gguf_init_from_file(path, gp);
    if (!gctx || !out.ctx) {
        fprintf(stderr, "%s: failed to load tensor metadata from '%s'\n", tag, path);
        if (gctx)
            gguf_free(gctx);
        return false;
    }

    // PLAN #51a: zero-copy CPU path. Skip ggml_backend_alloc_ctx_tensors
    // (which would allocate a fresh backend-side buffer) and instead bind
    // each tensor directly into the mmap'd file. Saves one full copy of
    // the weights — the difference between a 14.9 GB F16 GGUF loading on
    // a 16 GB Mac and thrashing swap. Gated on CRISPASR_GGUF_MMAP=1 +
    // CPU backend until we've validated all 24 backends.
    if (mmap_loader_enabled() && ggml_backend_is_cpu(backend)) {
        MappedFile mf(path, /*writable=*/true);
        if (mf.ok) {
            const size_t data_off = gguf_get_data_offset(gctx);
            const size_t mmap_size = mf.size;
            char* tensor_base = (char*)mf.base + data_off;
            const size_t buf_size = mmap_size > data_off ? (mmap_size - data_off) : 0;

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
            ::posix_madvise(mctx->mmap_base, mctx->mmap_size, POSIX_MADV_WILLNEED);
#endif
            // PLAN #60c / #60f: optional preload + mlock, opt-in via env.
            if (preload_enabled())
                preload_pages(mctx->mmap_base, mctx->mmap_size);
            if (mlock_enabled())
                try_mlock(tag, mctx->mmap_base, mctx->mmap_size);

            out.buf = ggml_backend_buffer_init(ggml_backend_cpu_buffer_type(), mmap_buffer_iface, mctx, buf_size);
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
        // mmap failed — fall through to the legacy alloc + copy path. We
        // do NOT print a warning here; mmap is best-effort and the legacy
        // path is functionally equivalent (just with more RSS).
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
    if (mmap_loader_enabled() && !ggml_backend_is_cpu(backend)) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(dev, &props);
        if (props.caps.buffer_from_host_ptr) {
            MappedFile mf(path, /*writable=*/true);
            if (mf.ok) {
                const size_t data_off = gguf_get_data_offset(gctx);
                char* tensor_base = (char*)mf.base + data_off;

                // Hand the entire mmap region (including GGUF header) to
                // the device. The backend's `buffer_from_host_ptr` uses
                // the size for its internal MTLBuffer view; tensor binds
                // below offset into this base.
                ggml_backend_buffer_t inner = ggml_backend_dev_buffer_from_host_ptr(dev, mf.base, mf.size,
                                                                                    /*max_tensor_size=*/0);
                if (inner) {
                    auto* w = new mmap_wrap_ctx{};
                    w->inner = inner;
                    w->mmap_base = mf.base;
                    w->mmap_size = mf.size;
                    mf.release();
                    // Hint kernel async readahead — same rationale as the
                    // CPU branch above. On Apple Silicon the unified-memory
                    // shared-storage MTLBuffer reads the same physical
                    // pages, so this readahead benefits both CPU and GPU
                    // accesses with one call.
#if !defined(_WIN32)
                    ::posix_madvise(w->mmap_base, w->mmap_size, POSIX_MADV_WILLNEED);
#endif
                    // PLAN #60c / #60f: optional preload + mlock, opt-in
                    // via env. mlock is particularly meaningful here —
                    // pinning prevents Metal's shared-storage reads from
                    // racing CPU page faults under memory pressure.
                    if (preload_enabled())
                        preload_pages(w->mmap_base, w->mmap_size);
                    if (mlock_enabled())
                        try_mlock(tag, w->mmap_base, w->mmap_size);

                    // Reuse the inner buffer's buft so usage/alignment/etc.
                    // stay consistent with the device's expectations. The
                    // wrapping iface intercepts `free_buffer` only.
                    out.buf =
                        ggml_backend_buffer_init(inner->buft, mmap_wrap_iface, w, ggml_backend_buffer_get_size(inner));
                    if (!out.buf) {
                        // Wrapper init failed: roll back the inner buffer
                        // and the mmap to keep the file descriptor + page
                        // tables clean.
                        fprintf(stderr, "%s: failed to wrap inner buffer (Metal mmap path)\n", tag);
                        ggml_backend_buffer_free(inner);
#if defined(_WIN32)
                        UnmapViewOfFile(w->mmap_base);
#else
                        ::munmap(w->mmap_base, w->mmap_size);
#endif
                        delete w;
                        gguf_free(gctx);
                        ggml_free(out.ctx);
                        out.ctx = nullptr;
                        return false;
                    }

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
                // buffer_from_host_ptr returned null — release mmap and fall
                // through to the legacy path. No warning; same rationale as
                // the CPU mmap branch.
            }
        }
    }

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        fprintf(stderr, "%s: failed to allocate backend buffer\n", tag);
        gguf_free(gctx);
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    MappedFile mf(path);
    if (!mf.ok) {
        // Fallback: read via FILE* pread/fseek. This is the rare path —
        // most systems have working mmap. We implement it inline here so
        // models don't have to.
        FILE* fp = fopen(path, "rb");
        if (!fp) {
            fprintf(stderr, "%s: cannot open '%s' for fread fallback\n", tag, path);
            gguf_free(gctx);
            return false;
        }
        const size_t data_off = gguf_get_data_offset(gctx);
        std::vector<uint8_t> tbuf;
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
            out.tensors[ggml_get_name(t)] = t;
            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0)
                continue;
            const size_t off = gguf_get_tensor_offset(gctx, tid);
            const size_t nbytes = ggml_nbytes(t);
            ggml_backend_tensor_set(t, (const char*)mf.base + data_off + off, 0, nbytes);
        }
    }

    gguf_free(gctx);
    return true;
}

void free_weights(WeightLoad& wl) {
    if (wl.buf) {
        ggml_backend_buffer_free(wl.buf);
        wl.buf = nullptr;
    }
    if (wl.buf_cpu) {
        ggml_backend_buffer_free(wl.buf_cpu);
        wl.buf_cpu = nullptr;
    }
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
    auto round_up = [](size_t n, size_t a) { return (n + a - 1) & ~(a - 1); };
    auto bind_partition = [&](ggml_backend_t be, const std::vector<ggml_tensor*>& tensors, size_t total,
                              ggml_backend_buffer_t* out_buf) -> bool {
        if (tensors.empty())
            return true;
        const size_t align = ggml_backend_get_alignment(be);
        // Compute final size with per-tensor alignment slack.
        size_t aligned_total = 0;
        for (ggml_tensor* t : tensors)
            aligned_total = round_up(aligned_total, align) + ggml_nbytes(t);
        (void)total;
        ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(be, aligned_total);
        if (!buf) {
            fprintf(stderr, "%s: failed to allocate %zu MiB backend buffer\n", tag, aligned_total / 1048576);
            return false;
        }
        char* base = (char*)ggml_backend_buffer_get_base(buf);
        size_t cursor = 0;
        for (ggml_tensor* t : tensors) {
            cursor = round_up(cursor, align);
            ggml_backend_tensor_alloc(buf, t, base + cursor);
            cursor += ggml_nbytes(t);
        }
        *out_buf = buf;
        return true;
    };

    if (!bind_partition(gpu_backend, gpu_tensors, gpu_size, &out.buf)) {
        gguf_free(gctx);
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }
    if (!bind_partition(cpu_backend, cpu_tensors, cpu_size, &out.buf_cpu)) {
        if (out.buf) {
            ggml_backend_buffer_free(out.buf);
            out.buf = nullptr;
        }
        gguf_free(gctx);
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
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
            ggml_backend_tensor_set(t, (const char*)mf.base + data_off + off, 0, nbytes);
        }
    }

    fprintf(stderr, "%s: weight residency: gpu=%zu MiB (%zu tensors), cpu=%zu MiB (%zu tensors)\n", tag,
            gpu_size / 1048576, gpu_tensors.size(), cpu_size / 1048576, cpu_tensors.size());

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
        // CPU mmap path — context is mmap_buffer_ctx.
        auto* mctx = (mmap_buffer_ctx*)buf->context;
        base = mctx->mmap_base;
        size = mctx->mmap_size;
    } else if (buf->iface.free_buffer == mmap_wrap_free) {
        // Metal mmap path — context is mmap_wrap_ctx.
        auto* w = (mmap_wrap_ctx*)buf->context;
        base = w->mmap_base;
        size = w->mmap_size;
    } else {
        return; // not one of our mmap buffers; nothing to advise
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

ggml_tensor* try_get(const std::map<std::string, ggml_tensor*>& tensors, const char* name) {
    auto it = tensors.find(name);
    return it != tensors.end() ? it->second : nullptr;
}

ggml_tensor* require(const std::map<std::string, ggml_tensor*>& tensors, const char* name, const char* model_tag) {
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
