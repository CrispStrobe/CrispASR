// crispasr_diff.cpp — implementation of the ground-truth diff harness.
// See crispasr_diff.h for the interface contract.

#include "crispasr_diff.h"

#include "core/gguf_loader.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace crispasr_diff {

// ---------------------------------------------------------------------------
// Implementation state
// ---------------------------------------------------------------------------

struct Ref::Impl {
    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr;

    // Cached float views of each tensor, keyed by name. Filled lazily.
    mutable std::map<std::string, std::vector<float>> f32_cache;

    // String metadata values keyed by GGUF key name (without the
    // "crispasr.ref." prefix).
    std::map<std::string, std::string> meta_strings;

    ~Impl() {
        core_gguf::free_weights(wl);
        if (backend)
            ggml_backend_free(backend);
    }
};

Ref::~Ref() {
    if (impl_) {
        delete impl_;
        impl_ = nullptr;
    }
}

Ref::Ref(Ref&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

Ref& Ref::operator=(Ref&& other) noexcept {
    if (this != &other) {
        if (impl_) {
            delete impl_;
        }
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

bool Ref::load(const std::string& path) {
    if (impl_) {
        delete impl_;
        impl_ = nullptr;
    }
    impl_ = new Impl();

    // The archive can be loaded onto the CPU backend — these tensors
    // never participate in compute, we just read them back to float.
    impl_->backend = ggml_backend_cpu_init();
    if (!impl_->backend) {
        fprintf(stderr, "crispasr_diff: failed to init CPU backend\n");
        delete impl_;
        impl_ = nullptr;
        return false;
    }

    if (!core_gguf::load_weights(path.c_str(), impl_->backend, "crispasr_diff", impl_->wl)) {
        fprintf(stderr, "crispasr_diff: failed to load '%s'\n", path.c_str());
        delete impl_;
        impl_ = nullptr;
        return false;
    }

    // Read the string metadata (crispasr.ref.*) directly from the GGUF
    // header so callers can query backend / model_dir / generated_text.
    // We re-open the GGUF in metadata-only mode because
    // core_gguf::load_weights closes its gguf_context after weight load.
    if (gguf_context* gctx = core_gguf::open_metadata(path.c_str())) {
        const int n = gguf_get_n_kv(gctx);
        for (int i = 0; i < n; i++) {
            const char* k = gguf_get_key(gctx, i);
            if (!k || strncmp(k, "crispasr.ref.", 13) != 0)
                continue;
            if (gguf_get_kv_type(gctx, i) != GGUF_TYPE_STRING)
                continue;
            const char* v = gguf_get_val_str(gctx, i);
            if (v)
                impl_->meta_strings[k + 13] = v;
        }
        core_gguf::free_metadata(gctx);
    }

    return true;
}

// ---------------------------------------------------------------------------
// Tensor lookup
// ---------------------------------------------------------------------------

bool Ref::has(const std::string& name) const {
    if (!impl_)
        return false;
    return impl_->wl.tensors.find(name) != impl_->wl.tensors.end();
}

static std::vector<int64_t> tensor_shape(const ggml_tensor* t) {
    std::vector<int64_t> out;
    for (int d = 0; d < GGML_MAX_DIMS; d++) {
        if (t->ne[d] <= 1 && d > 0)
            continue; // strip trailing 1s after dim 0
        out.push_back(t->ne[d]);
    }
    // But keep at least one dim
    if (out.empty())
        out.push_back(1);
    return out;
}

std::vector<int64_t> Ref::shape(const std::string& name) const {
    if (!impl_)
        return {};
    auto it = impl_->wl.tensors.find(name);
    if (it == impl_->wl.tensors.end())
        return {};
    return tensor_shape(it->second);
}

static const std::vector<float>& cache_f32(Ref::Impl* impl, const std::string& name) {
    auto it = impl->f32_cache.find(name);
    if (it != impl->f32_cache.end())
        return it->second;

    auto tt = impl->wl.tensors.find(name);
    if (tt == impl->wl.tensors.end()) {
        static const std::vector<float> empty;
        return empty;
    }
    ggml_tensor* t = tt->second;
    const size_t nb = ggml_nbytes(t);
    std::vector<float> buf;
    // Only handle F32 for now — dump_reference.py writes every activation
    // as F32 by default. Adding I32 handling when we need argmax-only refs.
    if (t->type == GGML_TYPE_F32) {
        buf.resize(nb / sizeof(float));
        ggml_backend_tensor_get(t, buf.data(), 0, nb);
    } else {
        fprintf(stderr,
                "crispasr_diff: tensor '%s' has type %s, only F32 is "
                "currently supported in the Ref loader\n",
                name.c_str(), ggml_type_name(t->type));
    }
    auto ins = impl->f32_cache.emplace(name, std::move(buf));
    return ins.first->second;
}

std::pair<const float*, size_t> Ref::get_f32(const std::string& name) const {
    if (!impl_)
        return {nullptr, 0};
    const auto& v = cache_f32(impl_, name);
    if (v.empty())
        return {nullptr, 0};
    return {v.data(), v.size()};
}

std::string Ref::meta(const std::string& key) const {
    if (!impl_)
        return "";
    auto it = impl_->meta_strings.find(key);
    return it != impl_->meta_strings.end() ? it->second : std::string();
}

std::vector<std::string> Ref::tensor_names() const {
    std::vector<std::string> out;
    if (!impl_)
        return out;
    out.reserve(impl_->wl.tensors.size());
    for (const auto& kv : impl_->wl.tensors)
        out.push_back(kv.first);
    return out;
}

// ---------------------------------------------------------------------------
// Compare
// ---------------------------------------------------------------------------

Report Ref::compare(const std::string& name, const float* data, size_t n_elem, CompareMode mode) const {
    Report r;
    if (!impl_)
        return r;
    auto tt = impl_->wl.tensors.find(name);
    if (tt == impl_->wl.tensors.end())
        return r; // r.found stays false
    r.found = true;
    r.shape = tensor_shape(tt->second);

    const auto& ref = cache_f32(impl_, name);
    if (ref.empty()) {
        fprintf(stderr, "crispasr_diff: reference '%s' is empty or non-F32\n", name.c_str());
        return r;
    }

    const size_t n = std::min(n_elem, ref.size());
    r.n_elem = n;
    if (n == 0)
        return r;

    // Element-wise diff
    double sum_abs = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        const float d = data[i] - ref[i];
        const float ad = d < 0 ? -d : d;
        if (ad > r.max_abs)
            r.max_abs = ad;
        sum_abs += ad;
        sum_sq += (double)d * (double)d;
    }
    r.mean_abs = (float)(sum_abs / n);
    r.rms = (float)std::sqrt(sum_sq / n);

    // Cosine similarity over the last dimension (rows)
    if (mode == COS_LAST_DIM && !r.shape.empty()) {
        const int row_w = (int)r.shape.back();
        if (row_w > 0) {
            const size_t n_rows = n / row_w;
            r.cos_min = 1.0f;
            double cos_sum = 0.0;
            size_t cos_rows = 0;
            for (size_t i = 0; i < n_rows; i++) {
                double dot = 0.0, na = 0.0, nb = 0.0;
                for (int k = 0; k < row_w; k++) {
                    const float a = data[i * row_w + k];
                    const float b = ref[i * row_w + k];
                    dot += (double)a * b;
                    na += (double)a * a;
                    nb += (double)b * b;
                }
                const double denom = std::sqrt(na) * std::sqrt(nb);
                if (denom > 1e-12) {
                    const float cs = (float)(dot / denom);
                    if (cs < r.cos_min)
                        r.cos_min = cs;
                    cos_sum += cs;
                    cos_rows++;
                }
            }
            if (cos_rows > 0)
                r.cos_mean = (float)(cos_sum / cos_rows);
        }
    }
    return r;
}

Report Ref::compare_argmax(const std::string& name, const float* data, size_t n_elem) const {
    Report r;
    if (!impl_)
        return r;
    auto tt = impl_->wl.tensors.find(name);
    if (tt == impl_->wl.tensors.end())
        return r;
    r.found = true;
    r.shape = tensor_shape(tt->second);
    if (r.shape.empty())
        return r;

    const auto& ref = cache_f32(impl_, name);
    if (ref.empty())
        return r;

    // ggml convention: ne[0] is the innermost (fastest-changing) dim. For
    // an (T, V) logits tensor that's V — first entry of `shape`, NOT the
    // numpy-style last entry. shape.back() would pick the outer dim
    // (T_text), giving meaningless argmax-over-T comparisons.
    const int vocab = (int)r.shape.front();
    if (vocab <= 0)
        return r;

    const size_t T_ref = ref.size() / vocab;
    const size_t T_cpp = std::min(n_elem, ref.size()) / vocab;
    const size_t T = std::min(T_ref, T_cpp);

    r.n_elem = T * vocab;
    r.top1_total = (int)T;
    r.top1_match = 0;
    for (size_t t = 0; t < T; t++) {
        int argmax_cpp = 0, argmax_ref = 0;
        float max_cpp = data[t * vocab];
        float max_ref = ref[t * vocab];
        for (int k = 1; k < vocab; k++) {
            const float a = data[t * vocab + k];
            const float b = ref[t * vocab + k];
            if (a > max_cpp) {
                max_cpp = a;
                argmax_cpp = k;
            }
            if (b > max_ref) {
                max_ref = b;
                argmax_ref = k;
            }
        }
        if (argmax_cpp == argmax_ref)
            r.top1_match++;
    }
    return r;
}

} // namespace crispasr_diff
