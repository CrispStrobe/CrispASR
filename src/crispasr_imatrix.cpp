// crispasr_imatrix.cpp — see crispasr_imatrix.h.
//
// File format (GGUF-based; consumed by crispasr-quantize --imatrix):
//   KV  general.architecture = "crispasr-imatrix"
//   KV  imatrix.version       = 1  (u32)
//   KV  count.<weight-name>   = u64   number of activation rows accumulated
//   tensor <weight-name>      = F32[n_per_row]  running sum of squares per column
// The quantizer reads importance[c] = sum_sq[c] / count.
//
// Ported from CrispEmbed src/imatrix.cpp.

#include "crispasr_imatrix.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {

std::mutex g_mu;
bool g_checked = false;
bool g_imatrix_active = false;
bool g_dump_active = false;
bool g_atexit = false;
bool g_flushed = false; // guard: write the imatrix at most once per process
std::string g_path;
// weight name -> running sum of squares (double for numerical stability)
std::map<std::string, std::vector<double>> g_sumsq;
// weight name -> running row count
std::map<std::string, uint64_t> g_count;

// Optional named-tensor dump (parity harness). When CRISPASR_ACTDUMP_OUT is
// set, the FIRST computation of the tensor named CRISPASR_ACTDUMP_TENSOR
// (default "logits" — the prefill logits, fixed-length and
// generation-independent) is written to that file as: int64 n_elem (LE) then
// n_elem × f32. Used by tools/imatrix_ab.py to cosine a quantized model's
// prefill logits against the f16 gold.
std::string g_dump_path;
std::string g_dump_tensor;
bool g_dump_done = false;

// Active if EITHER feature is requested; both share one eval callback.
bool is_active() {
    if (!g_checked) {
        const char* p = getenv("CRISPASR_IMATRIX_OUT");
        if (p && p[0]) {
            g_imatrix_active = true;
            g_path = p;
        }
        const char* d = getenv("CRISPASR_ACTDUMP_OUT");
        if (d && d[0]) {
            g_dump_active = true;
            g_dump_path = d;
            const char* tn = getenv("CRISPASR_ACTDUMP_TENSOR");
            g_dump_tensor = (tn && tn[0]) ? tn : "logits";
        }
        g_checked = true;
    }
    return g_imatrix_active || g_dump_active;
}

// Chase view chains to the underlying leaf tensor (the actual weight).
const ggml_tensor* underlying(const ggml_tensor* t) {
    while (t && t->view_src)
        t = t->view_src;
    return t;
}

// eval callback: ask==true  -> return whether we want a post-compute callback
//                ask==false -> the node's data is ready; collect it
// Write the first matching tensor to the actdump file (once per process).
void maybe_dump(struct ggml_tensor* t) {
    if (!g_dump_active || g_dump_done)
        return;
    if (t->name[0] == '\0' || g_dump_tensor != t->name)
        return;
    if (!ggml_is_contiguous(t) || t->type != GGML_TYPE_F32)
        return;
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_dump_done)
        return;
    const int64_t n = ggml_nelements(t);
    std::vector<float> buf((size_t)n);
    ggml_backend_tensor_get(t, buf.data(), 0, (size_t)n * sizeof(float));
    FILE* f = fopen(g_dump_path.c_str(), "wb");
    if (f) {
        fwrite(&n, sizeof(int64_t), 1, f);
        fwrite(buf.data(), sizeof(float), (size_t)n, f);
        fclose(f);
        fprintf(stderr, "actdump: wrote '%s' (%lld floats) to '%s'\n", t->name, (long long)n, g_dump_path.c_str());
    }
    g_dump_done = true;
}

bool eval_cb(struct ggml_tensor* t, bool ask, void* /*ud*/) {
    // Named-tensor dump path (independent of the mul_mat/imatrix logic): the
    // dump target may be any op, so request its post-compute callback here.
    const bool is_dump_target = g_dump_active && !g_dump_done && t->name[0] != '\0' && g_dump_tensor == t->name;

    if (t->op != GGML_OP_MUL_MAT)
        return is_dump_target ? (ask ? true : (maybe_dump(t), true)) : false;
    const ggml_tensor* w = underlying(t->src[0]);
    // src0 must be a named leaf weight (activations are computed nodes, op != NONE)
    if (!w || w->op != GGML_OP_NONE || w->name[0] == '\0')
        return is_dump_target ? (ask ? true : (maybe_dump(t), true)) : false;

    if (ask)
        return true; // request the post-compute callback

    maybe_dump(t); // logits is a mul_mat(output.weight, hidden) — may be the dump target

    if (!g_imatrix_active)
        return true;

    struct ggml_tensor* a = t->src[1]; // the activation input
    if (!a || !ggml_is_contiguous(a))
        return true;
    const int64_t ne0 = a->ne[0];
    if (ne0 <= 0)
        return true;
    const int64_t nrows = ggml_nelements(a) / ne0;
    if (nrows <= 0)
        return true;

    // Read the activation into a typed F32 buffer (dequantising F16 on the
    // way) — read directly into the correctly-typed pointer rather than
    // reinterpreting a byte buffer, so there is no cross-representation cast.
    const int64_t nelem = ggml_nelements(a);
    std::vector<float> fx((size_t)nelem);
    if (a->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(a, fx.data(), 0, ggml_nbytes(a));
    } else if (a->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h((size_t)nelem);
        ggml_backend_tensor_get(a, h.data(), 0, ggml_nbytes(a));
        for (int64_t i = 0; i < nelem; i++)
            fx[i] = ggml_fp16_to_fp32(h[i]);
    } else {
        return true; // unsupported activation type; skip
    }

    std::lock_guard<std::mutex> lk(g_mu);
    auto& acc = g_sumsq[w->name];
    if (acc.empty())
        acc.assign((size_t)ne0, 0.0);
    if ((int64_t)acc.size() != ne0)
        return true; // shape drift; skip defensively
    for (int64_t r = 0; r < nrows; r++) {
        const float* row = fx.data() + r * ne0;
        for (int64_t c = 0; c < ne0; c++)
            acc[c] += (double)row[c] * (double)row[c];
    }
    g_count[w->name] += (uint64_t)nrows;
    return true;
}

// Merge any pre-existing imatrix file into the in-memory accumulators.
void merge_existing() {
    struct ggml_context* ctx = nullptr;
    struct gguf_init_params p = {/*no_alloc*/ false, /*ctx*/ &ctx};
    struct gguf_context* g = gguf_init_from_file(g_path.c_str(), p);
    if (!g)
        return; // no prior file (first run)
    const int64_t nt = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < nt; i++) {
        const char* name = gguf_get_tensor_name(g, i);
        struct ggml_tensor* t = ggml_get_tensor(ctx, name);
        if (!t || t->type != GGML_TYPE_F32)
            continue;
        const int64_t ne0 = t->ne[0];
        const float* d = (const float*)t->data;
        auto& acc = g_sumsq[name];
        if (acc.empty())
            acc.assign((size_t)ne0, 0.0);
        if ((int64_t)acc.size() != ne0)
            continue;
        for (int64_t c = 0; c < ne0; c++)
            acc[c] += (double)d[c];
        std::string ck = std::string("count.") + name;
        int64_t kid = gguf_find_key(g, ck.c_str());
        if (kid >= 0)
            g_count[name] += gguf_get_val_u64(g, kid);
    }
    gguf_free(g);
    ggml_free(ctx);
}

} // namespace

ggml_backend_sched_eval_callback crispasr_imatrix_callback(void) {
    if (!is_active())
        return nullptr;
    if (!g_atexit) {
        atexit(crispasr_imatrix_flush);
        g_atexit = true;
    }
    return eval_cb;
}

void crispasr_imatrix_install(ggml_backend_sched_t sched) {
    if (!sched)
        return;
    if (auto callback = crispasr_imatrix_callback())
        ggml_backend_sched_set_eval_callback(sched, callback, nullptr);
}

void crispasr_imatrix_flush(void) {
    std::lock_guard<std::mutex> lk(g_mu);
    // one-shot binaries may bypass atexit; guard against writing twice.
    if (g_flushed || !g_imatrix_active || g_sumsq.empty())
        return;
    g_flushed = true;

    merge_existing();

    // Allocate a ggml context large enough to hold all sum-of-squares tensors.
    size_t total = 0;
    for (auto& kv : g_sumsq)
        total += kv.second.size();
    struct ggml_init_params ip = {
        /*mem_size*/ total * sizeof(float) + g_sumsq.size() * ggml_tensor_overhead() + (1u << 20),
        /*mem_buffer*/ nullptr,
        /*no_alloc*/ false,
    };
    struct ggml_context* ctx = ggml_init(ip);
    if (!ctx) {
        fprintf(stderr, "imatrix: ggml_init failed\n");
        return;
    }

    struct gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "crispasr-imatrix");
    gguf_set_val_u32(g, "imatrix.version", 1);

    for (auto& kv : g_sumsq) {
        const std::string& name = kv.first;
        const std::vector<double>& acc = kv.second;
        struct ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)acc.size());
        ggml_set_name(t, name.c_str());
        float* d = (float*)t->data;
        for (size_t c = 0; c < acc.size(); c++)
            d[c] = (float)acc[c];
        gguf_add_tensor(g, t);
        std::string ck = std::string("count.") + name;
        gguf_set_val_u64(g, ck.c_str(), g_count[name]);
    }

    if (!gguf_write_to_file(g, g_path.c_str(), /*only_meta*/ false)) {
        fprintf(stderr, "imatrix: failed to write '%s'\n", g_path.c_str());
    } else {
        fprintf(stderr, "imatrix: wrote %zu tensors to '%s'\n", g_sumsq.size(), g_path.c_str());
    }
    gguf_free(g);
    ggml_free(ctx);
}
