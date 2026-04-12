// src/core/gguf_loader.cpp — implementation of core_gguf:: helpers.
// See gguf_loader.h for the interface contract.

#include "gguf_loader.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#  include <io.h>
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace core_gguf {

// ---------------------------------------------------------------------------
// Pass 1: metadata
// ---------------------------------------------------------------------------

gguf_context * open_metadata(const char * path) {
    gguf_init_params gp = { /*.no_alloc=*/true, /*.ctx=*/nullptr };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) {
        fprintf(stderr, "core_gguf: failed to open '%s' for metadata read\n", path);
    }
    return g;
}

void free_metadata(gguf_context * gctx) {
    if (gctx) gguf_free(gctx);
}

// Type-checked scalar readers. The GGUF format stores types explicitly so
// we can validate; if the file has a mismatched type the reader silently
// returns the default rather than crashing, matching the existing inline
// helpers in each model.

uint32_t kv_u32(gguf_context * gctx, const char * key, uint32_t default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0) return default_val;
    const gguf_type t = gguf_get_kv_type(gctx, k);
    switch (t) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(gctx, k);
        case GGUF_TYPE_INT32:  return (uint32_t)gguf_get_val_i32(gctx, k);
        case GGUF_TYPE_UINT64: return (uint32_t)gguf_get_val_u64(gctx, k);
        case GGUF_TYPE_INT64:  return (uint32_t)gguf_get_val_i64(gctx, k);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(gctx, k);
        case GGUF_TYPE_INT16:  return (uint32_t)gguf_get_val_i16(gctx, k);
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(gctx, k);
        case GGUF_TYPE_INT8:   return (uint32_t)gguf_get_val_i8(gctx, k);
        default:               return default_val;
    }
}

int32_t kv_i32(gguf_context * gctx, const char * key, int32_t default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0) return default_val;
    const gguf_type t = gguf_get_kv_type(gctx, k);
    switch (t) {
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(gctx, k);
        case GGUF_TYPE_UINT32: return (int32_t)gguf_get_val_u32(gctx, k);
        case GGUF_TYPE_INT64:  return (int32_t)gguf_get_val_i64(gctx, k);
        case GGUF_TYPE_UINT64: return (int32_t)gguf_get_val_u64(gctx, k);
        default:               return default_val;
    }
}

float kv_f32(gguf_context * gctx, const char * key, float default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0) return default_val;
    const gguf_type t = gguf_get_kv_type(gctx, k);
    if (t == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(gctx, k);
    if (t == GGUF_TYPE_FLOAT64) return (float)gguf_get_val_f64(gctx, k);
    return default_val;
}

bool kv_bool(gguf_context * gctx, const char * key, bool default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0) return default_val;
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_BOOL) return default_val;
    return gguf_get_val_bool(gctx, k);
}

std::string kv_str(gguf_context * gctx, const char * key, const char * default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0) return default_val ? default_val : "";
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_STRING) return default_val ? default_val : "";
    const char * s = gguf_get_val_str(gctx, k);
    return s ? std::string(s) : std::string(default_val ? default_val : "");
}

std::vector<std::string> kv_str_array(gguf_context * gctx, const char * key) {
    std::vector<std::string> out;
    const int k = gguf_find_key(gctx, key);
    if (k < 0) return out;
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_ARRAY) return out;
    if (gguf_get_arr_type(gctx, k) != GGUF_TYPE_STRING) return out;
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
    int    fd = -1;
    void * base = nullptr;
    size_t size = 0;
    bool   ok   = false;

    explicit MappedFile(const char * path) {
#if defined(_WIN32)
        HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                                   nullptr, OPEN_EXISTING, 0, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) return;
        LARGE_INTEGER fsize;
        if (!GetFileSizeEx(hFile, &fsize)) { CloseHandle(hFile); return; }
        size = (size_t)fsize.QuadPart;
        HANDLE hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        CloseHandle(hFile);
        if (!hMap) return;
        base = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(hMap);
        if (!base) return;
        ok = true;
#else
        fd = ::open(path, O_RDONLY);
        if (fd < 0) return;
        struct stat st;
        if (fstat(fd, &st) != 0) { ::close(fd); fd = -1; return; }
        size = (size_t)st.st_size;
        base = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        fd = -1;
        if (base == MAP_FAILED) { base = nullptr; return; }
        ok = true;
#endif
    }
    ~MappedFile() {
#if defined(_WIN32)
        if (base) UnmapViewOfFile(base);
#else
        if (base) ::munmap(base, size);
#endif
    }
    MappedFile(const MappedFile &) = delete;
    MappedFile & operator=(const MappedFile &) = delete;
};

} // namespace

bool load_weights(const char  * path,
                  ggml_backend_t backend,
                  const char  * model_tag,
                  WeightLoad  & out)
{
    const char * tag = model_tag ? model_tag : "core_gguf";

    gguf_init_params gp = { /*.no_alloc=*/true, /*.ctx=*/&out.ctx };
    gguf_context * gctx = gguf_init_from_file(path, gp);
    if (!gctx || !out.ctx) {
        fprintf(stderr, "%s: failed to load tensor metadata from '%s'\n", tag, path);
        if (gctx) gguf_free(gctx);
        return false;
    }

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        fprintf(stderr, "%s: failed to allocate backend buffer\n", tag);
        gguf_free(gctx);
        ggml_free(out.ctx); out.ctx = nullptr;
        return false;
    }

    MappedFile mf(path);
    if (!mf.ok) {
        // Fallback: read via FILE* pread/fseek. This is the rare path —
        // most systems have working mmap. We implement it inline here so
        // models don't have to.
        FILE * fp = fopen(path, "rb");
        if (!fp) {
            fprintf(stderr, "%s: cannot open '%s' for fread fallback\n", tag, path);
            gguf_free(gctx);
            return false;
        }
        const size_t data_off = gguf_get_data_offset(gctx);
        std::vector<uint8_t> tbuf;
        for (ggml_tensor * t = ggml_get_first_tensor(out.ctx); t;
             t = ggml_get_next_tensor(out.ctx, t)) {
            out.tensors[ggml_get_name(t)] = t;
            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0) continue;
            const size_t off    = gguf_get_tensor_offset(gctx, tid);
            const size_t nbytes = ggml_nbytes(t);
            if (tbuf.size() < nbytes) tbuf.resize(nbytes);
#if defined(_WIN32)
            if (_fseeki64(fp, (int64_t)(data_off + off), SEEK_SET) != 0) break;
#else
            if (fseeko(fp, (off_t)(data_off + off), SEEK_SET) != 0) break;
#endif
            if (fread(tbuf.data(), 1, nbytes, fp) != nbytes) break;
            ggml_backend_tensor_set(t, tbuf.data(), 0, nbytes);
        }
        fclose(fp);
    } else {
        const size_t data_off = gguf_get_data_offset(gctx);
        for (ggml_tensor * t = ggml_get_first_tensor(out.ctx); t;
             t = ggml_get_next_tensor(out.ctx, t)) {
            out.tensors[ggml_get_name(t)] = t;
            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0) continue;
            const size_t off    = gguf_get_tensor_offset(gctx, tid);
            const size_t nbytes = ggml_nbytes(t);
            ggml_backend_tensor_set(
                t,
                (const char *)mf.base + data_off + off,
                0, nbytes);
        }
    }

    gguf_free(gctx);
    return true;
}

void free_weights(WeightLoad & wl) {
    if (wl.buf) { ggml_backend_buffer_free(wl.buf); wl.buf = nullptr; }
    if (wl.ctx) { ggml_free(wl.ctx);                wl.ctx = nullptr; }
    wl.tensors.clear();
}

// ---------------------------------------------------------------------------
// Tensor lookup helpers
// ---------------------------------------------------------------------------

ggml_tensor * try_get(
    const std::map<std::string, ggml_tensor *> & tensors,
    const char * name)
{
    auto it = tensors.find(name);
    return it != tensors.end() ? it->second : nullptr;
}

ggml_tensor * require(
    const std::map<std::string, ggml_tensor *> & tensors,
    const char * name,
    const char * model_tag)
{
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        fprintf(stderr, "%s: required tensor '%s' not found in GGUF\n",
                model_tag ? model_tag : "core_gguf", name);
        return nullptr;
    }
    return it->second;
}

std::string format_layer_name(const char * fmt, int i) {
    char buf[256];
    snprintf(buf, sizeof(buf), fmt, i);
    return std::string(buf);
}

std::string format_layer_name(const char * fmt, int i, int j) {
    char buf[256];
    snprintf(buf, sizeof(buf), fmt, i, j);
    return std::string(buf);
}

} // namespace core_gguf
