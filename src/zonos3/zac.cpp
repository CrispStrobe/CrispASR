// ZONOS3 Mini codec decoder (mossdual_istft; fixed600k hybrid dual Opus, 15 Hz): semantic RVQ +
// causal ConvNeXt head, residual RVQ, windowed-causal MOSS transformer stages with patch upsampling
// (240 Hz), the multi-resolution iSTFT head (ConvNeXt trunk on the backend; iSTFT synthesis and the
// FIR mixer on the host) at 44160 Hz, and the exact 44160->44100 sinc resampler (host side).
#include "zac.h"
#include "zac-encoder.h"
#include "zonos3-text.h"

#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.hpp"

#include "compat.h"
#include "backend-device.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__APPLE__)
// The CPU decoder convs call Accelerate's sgemm directly (see conv_gemm), so taps accumulate into
// one output buffer that already holds the bias (and residual) instead of ggml's separate
// matmul/add/concat passes.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#define ZAC_CPU_GEMM 1
#endif

#include <deque>

ggml_tensor* zac_model::get(const std::string& n) const {
    auto it = t.find(n);
    if (it == t.end()) {
        fprintf(stderr, "zac: missing tensor '%s'\n", n.c_str());
        return nullptr;
    }
    return it->second;
}

// Creates the backend for `dev`; on CPU also the ZAC_THREADS setting and the BLAS offload.
static bool zac_init_backend(zac_model& m, ggml_backend_dev_t dev) {
    // F16 weight storage does not authorize narrowing F32 decoder activations.
    // Match the codec's full-precision accumulation before CUDA creates handles.
#ifdef _WIN32
    _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
#else
    setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif
    m.backend = ggml_backend_dev_init(dev, nullptr);
    if (!m.backend)
        return false;
    m.buft = ggml_backend_dev_buffer_type(dev);
    m.is_gpu = zonos3_backend_is_gpu(dev);
    if (!m.is_gpu) {
        const char* th = getenv("ZAC_THREADS");
        if (th)
            ggml_backend_cpu_set_n_threads(m.backend, std::max(1, atoi(th)));
        // The decoder convs are large f32 GEMMs; ggml's CPU kernels run them ~4x slower than
        // a BLAS backend such as Accelerate. Offload them when one is built in.
        if (ggml_backend_dev_t accel = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL)) {
            m.accel = ggml_backend_dev_init(accel, nullptr);
            if (m.accel) {
                ggml_backend_t backends[2] = {m.accel, m.backend};
                m.sched = ggml_backend_sched_new(backends, nullptr, 2, 16384, false, false);
            }
        }
    }
    return true;
}

// i for the per-codebook residual bias "res.{i}.bias" (0 <= i < 9: the n_codebooks - 1 residual codebooks), else -1.
static int res_bias_index(const std::string& name) {
    if (name.size() != 10 || name.compare(0, 4, "res.") != 0 || name.compare(5, 5, ".bias") != 0)
        return -1;
    return name[4] >= '0' && name[4] <= '8' ? name[4] - '0' : -1;
}

bool zac_load(zac_model& m, const char* path, bool use_gpu) {
    ggml_context* ctx_meta = nullptr;
    gguf_init_params gp = {/*no_alloc=*/true, /*ctx=*/&ctx_meta};
    gguf_context* gguf = gguf_init_from_file(path, gp);
    if (!gguf) {
        fprintf(stderr, "zac: cannot open gguf '%s'\n", path);
        return false;
    }
    auto reject = [&](const char* why) {
        fprintf(stderr,
                "zac: '%s' is not a ZONOS3 Mini iSTFT decoder GGUF (reconvert with tools/convert_gguf.py "
                "codec-decoder): %s\n",
                path, why);
        gguf_free(gguf);
        ggml_free(ctx_meta);
        return false;
    };
    int64_t arch = gguf_find_key(gguf, "general.architecture");
    if (arch < 0 || gguf_get_kv_type(gguf, arch) != GGUF_TYPE_STRING ||
        strcmp(gguf_get_val_str(gguf, arch), "mossdual_istft"))
        return reject("architecture");
    auto u32 = [&](const char* key, int& val) {
        int64_t id = gguf_find_key(gguf, (std::string("mossdual_istft.") + key).c_str());
        if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_UINT32)
            return false;
        val = (int)gguf_get_val_u32(gguf, id);
        return val > 0;
    };
    auto f32 = [&](const char* key, float& val) {
        int64_t id = gguf_find_key(gguf, (std::string("mossdual_istft.") + key).c_str());
        if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_FLOAT32)
            return false;
        val = gguf_get_val_f32(gguf, id);
        return val > 0;
    };
    auto arr = [&](const char* key, std::vector<int>& dst, size_t n, int lo = 1) {
        int64_t id = gguf_find_key(gguf, (std::string("mossdual_istft.") + key).c_str());
        if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gguf, id) != GGUF_TYPE_INT32 ||
            gguf_get_arr_n(gguf, id) != n)
            return false;
        auto p = (const int32_t*)gguf_get_arr_data(gguf, id);
        dst.assign(p, p + n);
        return std::all_of(dst.begin(), dst.end(), [lo](int v) { return v >= lo; });
    };
    constexpr int NS = 5;
    int head_hidden = 0, head_step = 0;
    std::vector<int> n_fft_v;
    std::vector<int> in_dim, out_dim, d_model, n_head, n_layer, n_ff, context, patch;
    if (!u32("sample_rate", m.sample_rate) || !u32("internal_sample_rate", m.internal_rate) ||
        !u32("frame_rate", m.frame_rate) || !u32("hop", m.hop) || !u32("internal_hop", m.internal_hop) ||
        !u32("n_codebooks", m.n_codebooks) || !u32("codebook_size", m.codebook_size) ||
        !u32("latent_dim", m.latent_dim) || !u32("semantic_dim", m.semantic_dim) ||
        !u32("sem_head.depth", m.sem_depth) || !u32("resample.orig", m.rs_orig) || !u32("resample.new", m.rs_new) ||
        !u32("resample.width", m.rs_width) || !f32("ln_eps", m.ln_eps) || !f32("convnext_eps", m.convnext_eps) ||
        !arr("stage.in_dim", in_dim, NS) || !arr("stage.out_dim", out_dim, NS, 0) ||
        !arr("stage.d_model", d_model, NS) || !arr("stage.n_head", n_head, NS) || !arr("stage.n_layer", n_layer, NS) ||
        !arr("stage.n_ff", n_ff, NS) || !arr("stage.context", context, NS) || !arr("stage.patch", patch, NS) ||
        !u32("head.depth", m.head_depth) || !u32("head.hidden", head_hidden) || !u32("head.hop", m.head_hop) ||
        !u32("head.fir_kernel", m.fir_kernel) || !u32("head.step", head_step) ||
        !f32("head.max_log_magnitude", m.max_log_mag) || !f32("headroom_db", m.headroom_db) ||
        !arr("head.n_fft", n_fft_v, 2))
        return reject("missing metadata");
    m.stages.clear();
    for (int s = 0; s < NS; ++s)
        m.stages.push_back({in_dim[s], out_dim[s], d_model[s], n_head[s], n_layer[s], n_ff[s], context[s], patch[s]});
    {
        const int64_t id = gguf_find_key(gguf, "mossdual_istft.head.variant");
        if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING)
            return reject("missing metadata");
        m.head_variant = gguf_get_val_str(gguf, id);
    }
    m.n_fft = {n_fft_v[0], n_fft_v[1]};
    int up = m.head_hop;
    for (int s : patch)
        up *= s;
    bool consistent = m.sample_rate == 44100 && m.internal_rate == 44160 && m.frame_rate == 15 && m.hop == 2940 &&
                      m.internal_hop == 2944 && m.n_codebooks == 10 && m.codebook_size == 4096 &&
                      (int64_t)m.internal_hop * m.rs_new == (int64_t)m.hop * m.rs_orig &&
                      patch == std::vector<int>({2, 2, 2, 2, 1}) && out_dim[NS - 1] == 0 && up == m.internal_hop &&
                      16 * m.head_hop == m.internal_hop && in_dim[0] == m.latent_dim &&
                      m.n_fft == std::array<int, 2>({736, 2208}) && m.n_fft[0] % m.head_hop == 0 &&
                      m.n_fft[1] % m.head_hop == 0 && m.fir_kernel % 2 == 1 && head_hidden > 0 && head_step > 0;
    for (int s = 0; s < NS; ++s) {
        consistent = consistent && d_model[s] % n_head[s] == 0 && (d_model[s] / n_head[s]) % 2 == 0;
        if (s + 1 < NS)
            consistent = consistent && in_dim[s + 1] * patch[s] == out_dim[s];
    }
    if (!consistent)
        return reject("inconsistent dimensions");
    { // Head tensor shapes (ggml ne order: [in, out] for matrices, [taps, channels] for depthwise).
        const int d4 = d_model[NS - 1];
        std::string bad;
        auto shape = [&](const std::string& name, std::initializer_list<int64_t> ne) {
            const ggml_tensor* t = ggml_get_tensor(ctx_meta, name.c_str());
            int i = 0;
            bool ok = t != nullptr;
            for (int64_t v : ne) {
                ok = ok && t->ne[i] == v;
                ++i;
            }
            for (; ok && i < GGML_MAX_DIMS; ++i)
                ok = t->ne[i] == 1;
            if (!ok && bad.empty())
                bad = name;
        };
        for (int b = 0; b < m.head_depth; ++b) {
            const std::string p = "head." + std::to_string(b);
            const ggml_tensor* dw = ggml_get_tensor(ctx_meta, (p + ".dw.weight").c_str());
            if (!dw) {
                bad = p + ".dw.weight";
                break;
            }
            shape(p + ".dw.weight", {dw->ne[0], d4});
            shape(p + ".dw.bias", {d4});
            shape(p + ".norm.weight", {d4});
            shape(p + ".norm.bias", {d4});
            shape(p + ".pw1.weight", {d4, head_hidden});
            shape(p + ".pw1.bias", {head_hidden});
            shape(p + ".pw2.weight", {head_hidden, d4});
            shape(p + ".pw2.bias", {d4});
        }
        shape("head.norm.weight", {d4});
        shape("head.norm.bias", {d4});
        for (int i = 0; i < 2; ++i) {
            shape("head.out." + std::to_string(i) + ".weight", {d4, m.n_fft[i] + 2});
            shape("head.out." + std::to_string(i) + ".bias", {m.n_fft[i] + 2});
            shape("head.window." + std::to_string(i), {m.n_fft[i]});
        }
        shape("head.mixer.weight", {m.fir_kernel, 2});
        shape("res.bias", {m.latent_dim});
        if (const ggml_tensor* rb = ggml_get_tensor(ctx_meta, "res.bias");
            rb && rb->type != GGML_TYPE_F32 && bad.empty())
            bad = "res.bias";
        // Per-codebook residual biases (early partial-frame decode): all of res.0.bias .. res.{n-2}.bias or none.
        int n_res_bias = 0;
        for (int i = 0; i + 1 < m.n_codebooks; ++i) {
            const std::string name = "res." + std::to_string(i) + ".bias";
            if (!ggml_get_tensor(ctx_meta, name.c_str()))
                continue;
            shape(name, {m.latent_dim});
            ++n_res_bias;
        }
        if (n_res_bias != 0 && n_res_bias != m.n_codebooks - 1 && bad.empty())
            bad = "res.{i}.bias (some codebooks only)";
        if (!bad.empty())
            return reject(("tensor '" + bad + "' missing or wrongly shaped").c_str());
    }

    ggml_backend_dev_t dev = zonos3_backend_device(use_gpu);
    if (!zac_init_backend(m, dev))
        return reject("cannot init backend");
    // MAT = transformer / semantic-head / iSTFT-head matrices: F16 on GPU, F32 on CPU (Accelerate
    // sgemm). Q8_0 on GPU measured no faster on M3 Metal (buffered RTF 0.123 vs 0.122 F16). Quantized
    // MAT weights and RVQ tables (quantize-cli) load as stored. DW = depthwise taps, repacked [K,C] ->
    // [C,K]. HOST = resampler kernel, FIR mixer and iSTFT windows, kept on the host as F32.
    enum Kind { PLAIN, DW, MAT, HOST };
    const ggml_type wtype = m.is_gpu ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_context* ctx_w = ggml_init({(size_t)(gguf_get_n_tensors(gguf) + 1) * ggml_tensor_overhead(), nullptr, true});
    struct Load {
        ggml_tensor *src, *dst;
        Kind kind;
    };
    std::vector<Load> loads;
    for (ggml_tensor* t = ggml_get_first_tensor(ctx_meta); t; t = ggml_get_next_tensor(ctx_meta, t)) {
        const std::string name = ggml_get_name(t);
        Kind kind = PLAIN;
        if (name == "resample.kernel" || name == "head.mixer.weight" || name == "head.window.0" ||
            name == "head.window.1" || res_bias_index(name) >= 0)
            kind = HOST;
        else if (ends_with(name, ".dw.weight"))
            kind = DW;
        else if ((name.rfind("stage.", 0) == 0 || name.rfind("sem_head.", 0) == 0 || name.rfind("head.", 0) == 0) &&
                 ends_with(name, ".weight") && ggml_n_dims(t) == 2 && name.find(".dw.") == std::string::npos &&
                 name.find("norm") == std::string::npos)
            kind = MAT;
        if (kind == MAT ? (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && !ggml_is_quantized(t->type))
                        : (kind != PLAIN && t->type != GGML_TYPE_F32)) {
            ggml_free(ctx_w);
            zac_free(m);
            return reject("unexpected tensor type");
        }
        if (kind == HOST) {
            loads.push_back({t, nullptr, kind});
            continue;
        }
        ggml_tensor* d;
        switch (kind) {
        case DW:
            d = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, t->ne[1], t->ne[0]);
            break; // [K,C] -> [C,K]
        case MAT:
            d = ggml_new_tensor_2d(ctx_w, ggml_is_quantized(t->type) ? t->type : wtype, t->ne[0], t->ne[1]);
            break;
        default:
            d = ggml_dup_tensor(ctx_w, t);
            break;
        }
        ggml_set_name(d, name.c_str());
        loads.push_back({t, d, kind});
    }
    m.ctx_w = ctx_w;
    m.buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, m.backend);
    FILE* f = fopen(path, "rb");
    if (!m.buf_w || !f) {
        if (f)
            fclose(f);
        gguf_free(gguf);
        ggml_free(ctx_meta);
        zac_free(m);
        return false;
    }
    const size_t data_off = gguf_get_data_offset(gguf);
    std::vector<char> tmp;
    std::vector<float> rep;
    std::vector<ggml_fp16_t> half;
    for (const Load& l : loads) {
        const char* name = ggml_get_name(l.src);
        const int64_t tid = gguf_find_tensor(gguf, name);
        const size_t off = data_off + gguf_get_tensor_offset(gguf, tid);
        const size_t nb = ggml_nbytes(l.src);
        tmp.resize(nb);
        if (fseeko(f, off, SEEK_SET) || fread(tmp.data(), 1, nb, f) != nb) {
            fprintf(stderr, "zac: read fail %s\n", name);
            fclose(f);
            gguf_free(gguf);
            ggml_free(ctx_meta);
            zac_free(m);
            return false;
        }
        const int64_t n = ggml_nelements(l.src);
        if (l.kind == HOST) {
            const float* hp = (const float*)tmp.data();
            const std::string nm = name;
            if (nm == "resample.kernel")
                m.rs_kernel.assign(hp, hp + n);
            else if (nm == "head.mixer.weight")
                m.mixer.assign(hp, hp + n);
            else if (const int c = res_bias_index(nm); c >= 0) {
                m.res_cb_bias.resize((size_t)(m.n_codebooks - 1) * m.latent_dim);
                std::copy(hp, hp + n, m.res_cb_bias.begin() + (size_t)c * m.latent_dim);
            } else
                m.window[nm.back() - '0'].assign(hp, hp + n);
            continue;
        }
        if (l.kind == PLAIN || (l.kind == MAT && l.src->type == l.dst->type)) {
            ggml_backend_tensor_set(l.dst, tmp.data(), 0, nb);
        } else if (l.kind == MAT && l.src->type == GGML_TYPE_F16) {
            rep.resize((size_t)n);
            ggml_fp16_to_fp32_row((const ggml_fp16_t*)tmp.data(), rep.data(), n);
            ggml_backend_tensor_set(l.dst, rep.data(), 0, (size_t)n * sizeof(float));
        } else if (l.kind == MAT) {
            half.resize((size_t)n);
            ggml_fp32_to_fp16_row((const float*)tmp.data(), half.data(), n);
            ggml_backend_tensor_set(l.dst, half.data(), 0, (size_t)n * sizeof(ggml_fp16_t));
        } else {
            const float* src = (const float*)tmp.data();
            rep.resize((size_t)n);
            if (l.kind == DW) {
                const int64_t K = l.src->ne[0], C = l.src->ne[1];
                for (int64_t c = 0; c < C; ++c)
                    for (int64_t k = 0; k < K; ++k)
                        rep[c + C * k] = src[k + K * c];
            }
            if (l.dst->type == GGML_TYPE_F16) {
                half.resize((size_t)n);
                ggml_fp32_to_fp16_row(rep.data(), half.data(), n);
                ggml_backend_tensor_set(l.dst, half.data(), 0, (size_t)n * sizeof(ggml_fp16_t));
            } else {
                ggml_backend_tensor_set(l.dst, rep.data(), 0, (size_t)n * sizeof(float));
            }
        }
        if (std::string(name) == "res.bias")
            m.res_bias.assign((const float*)tmp.data(), (const float*)tmp.data() + n);
        m.t[name] = l.dst;
    }
    fclose(f);
    gguf_free(gguf);
    ggml_free(ctx_meta);
    if (m.rs_kernel.size() != (size_t)m.rs_new * (2 * m.rs_width + m.rs_orig)) {
        fprintf(stderr,
                "zac: '%s' is not a ZONOS3 Mini iSTFT decoder GGUF (reconvert with tools/convert_gguf.py "
                "codec-decoder): resample.kernel\n",
                path);
        zac_free(m);
        return false;
    }
    for (int i = 0; i < 2; ++i)
        m.irfft[i] = std::make_shared<const pocketfft::detail::pocketfft_r<float>>((size_t)m.n_fft[i]);
    int n_moss = 0;
    for (const auto& s : m.stages)
        n_moss += s.n_layer;
    { // Probe flash attention at the real MOSS shapes (head_dim 64, F16 cache, F16 mask).
        ggml_context* pc = ggml_init({8 * ggml_tensor_overhead(), nullptr, true});
        ggml_tensor* q = ggml_new_tensor_3d(pc, GGML_TYPE_F32, 64, 4, 12);
        ggml_tensor* k = ggml_new_tensor_3d(pc, GGML_TYPE_F16, 64, 32, 12);
        ggml_tensor* v = ggml_new_tensor_3d(pc, GGML_TYPE_F16, 64, 32, 12);
        ggml_tensor* mk = ggml_new_tensor_2d(pc, GGML_TYPE_F16, 32, 4);
        ggml_tensor* fa = ggml_flash_attn_ext(pc, q, k, v, mk, 0.125f, 0.0f, 0.0f);
        ggml_prec_set_acc(fa, GGML_PREC_F32);
        m.use_fattn = ggml_backend_supports_op(m.backend, fa);
        ggml_free(pc);
        if (!m.use_fattn)
            fprintf(stderr, "zac: flash attention unsupported on this backend, using matmul attention\n");
    }
    fprintf(stderr, "zac: loaded %zu mossdual_istft tensors on %s (%d MOSS layers, head %s)\n", m.t.size(),
            m.is_gpu ? "GPU" : "CPU", n_moss, m.head_variant.c_str());
    return true;
}

void zac_free(zac_model& m) {
    m.graphs.clear();
    ggml_gallocr_free(m.galloc);
    if (m.sched)
        ggml_backend_sched_free(m.sched);
    if (m.owns_weights) {
        if (m.buf_w)
            ggml_backend_buffer_free(m.buf_w);
        if (m.ctx_w)
            ggml_free(m.ctx_w);
    }
    if (m.accel)
        ggml_backend_free(m.accel);
    if (m.backend)
        ggml_backend_free(m.backend);
    m.galloc = nullptr;
    m.sched = nullptr;
    m.accel = nullptr;
    m.buf_w = nullptr;
    m.ctx_w = nullptr;
    m.backend = nullptr;
    m.t.clear();
    m.owns_weights = true;
}

bool zac_clone(zac_model& dst, const zac_model& src) {
    dst = src;
    dst.backend = nullptr;
    dst.accel = nullptr;
    dst.sched = nullptr;
    dst.galloc = nullptr;
    dst.graphs.clear(); // a clone has its own backend: never share graphs
    dst.owns_weights = false;
    if (zac_init_backend(dst, ggml_backend_get_device(src.backend)))
        return true;
    zac_free(dst);
    return false;
}

// ---- render graph; activations are CHANNEL-major [C, T] (ne0=channels) ----

namespace {
// Per-op constants of a fused CPU conv (conv_gemm_cpu). Lives in render_ctx until compute ends.
struct conv_gemm_params {
    int K = 0, d = 1;          // taps and dilation over the history-prefixed input
    std::vector<size_t> w_off; // per-tap offset into w, in floats
};

struct render_ctx {
    ggml_context* ctx;
    const zac_model& m;
    int n_valid; // frames of real input; the rest is right padding
    int cpf;     // columns per frame at the current resolution
    std::vector<ggml_tensor*> hist_in, hist_out;
    std::deque<conv_gemm_params> conv_params;
    ggml_cgraph* graph = nullptr; // render graph, for explicit node ordering
    zac_stream* st = nullptr;     // MOSS K/V caches (st->kc/vc), consumed in layer order
    int layer = 0;                // next MOSS layer index into st->kc/vc
    bool encoder = false;         // independent lane, no streaming history
    bool input_bf16 = false;
};
} // namespace

// Prepend the R input columns carried over from the previous render and export the last R
// columns of this render's valid input for the next one. Returns [C, R + T].
static ggml_tensor* with_history(render_ctx& rc, ggml_tensor* x, int R) {
    if (R == 0)
        return x;
    if (rc.encoder)
        return ggml_pad_ext(rc.ctx, x, 0, 0, R, 0, 0, 0, 0, 0);
    ggml_context* ctx = rc.ctx;
    const int64_t C = x->ne[0];
    ggml_tensor* hist = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, R);
    ggml_set_input(hist);
    ggml_tensor* xf = ggml_concat(ctx, hist, x, 1);
    ggml_tensor* out = ggml_cont(ctx, ggml_view_2d(ctx, xf, C, R, xf->nb[1], (size_t)rc.n_valid * rc.cpf * xf->nb[1]));
    ggml_set_output(out);
    rc.hist_in.push_back(hist);
    rc.hist_out.push_back(out);
    return xf;
}

static ggml_tensor* add_bias(ggml_context* ctx, ggml_tensor* y, ggml_tensor* bias) {
    return ggml_add(ctx, y, ggml_reshape_2d(ctx, bias, y->ne[0], 1));
}

// Production encoding uses autocast BF16, but residuals, Snake and RVQ remain F32.
// Round operation outputs explicitly so CPU/GPU encoders have the same dtype boundaries.
static ggml_tensor* encoder_round(render_ctx& rc, ggml_tensor* x) {
    return rc.encoder ? ggml_cast(rc.ctx, ggml_cast(rc.ctx, x, GGML_TYPE_BF16), GGML_TYPE_F32) : x;
}

#if ZAC_CPU_GEMM
// dst [OC, T] = bias (+ residual) + sum_k W_k^T xf[:, t + k*d], where xf = [hist | x] is never
// materialized: each tap's column range is split into the part reading `hist` and the part
// reading `x`. Threads take contiguous output column ranges. Sources: x, w, bias, hist?, res?.
static void conv_gemm_cpu(ggml_tensor* dst, int ith, int nth, void* ud) {
    const conv_gemm_params& p = *(const conv_gemm_params*)ud;
    const ggml_tensor *x = dst->src[0], *w = dst->src[1], *bias = dst->src[2];
    const ggml_tensor *hist = dst->src[3], *res = dst->src[4];
    const int IC = (int)x->ne[0], OC = (int)dst->ne[0], T = (int)dst->ne[1];
    const int R = hist ? (int)hist->ne[1] : 0;
    const int c0 = (int)((int64_t)T * ith / nth), c1 = (int)((int64_t)T * (ith + 1) / nth);
    if (c0 >= c1)
        return;
    float* y = (float*)dst->data;
    const float* b = (const float*)bias->data;
    const int nb = (int)ggml_nelements(bias);
    GGML_ASSERT(OC % nb == 0);
    for (int t = c0; t < c1; ++t) {
        float* yt = y + (size_t)t * OC;
        for (int o = 0; o < OC; o += nb)
            memcpy(yt + o, b, (size_t)nb * sizeof(float));
        if (res) {
            const float* rt = (const float*)res->data + (size_t)t * OC;
            for (int o = 0; o < OC; ++o)
                yt[o] += rt[o];
        }
    }
    const float* wd = (const float*)w->data;
    auto gemm = [&](const float* wk, const float* src, int t0, int t1) {
        if (t0 >= t1)
            return;
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, OC, t1 - t0, IC, 1.0f, wk, IC, src, IC, 1.0f,
                    y + (size_t)t0 * OC, OC);
    };
    // Tile output columns so each tile's input/output stays cache-resident across the K taps
    // (measured 1.2-2.1x on T >= 10k-column convs, neutral on small T).
    constexpr int TILE_ELEMS = 98304;
    int NT = std::max(256, TILE_ELEMS / OC);
    if (c1 - c0 < 4 * NT)
        NT = c1 - c0;
    for (int a0 = c0; a0 < c1; a0 += NT) {
        const int a1 = std::min(a0 + NT, c1);
        for (int k = 0; k < p.K; ++k) {
            const float* wk = wd + p.w_off[k];
            const int sh = k * p.d, split = std::clamp(R - sh, a0, a1); // columns < split read hist
            if (split > a0)
                gemm(wk, (const float*)hist->data + (size_t)(a0 + sh) * IC, a0, split);
            gemm(wk, (const float*)x->data + (size_t)(split + sh - R) * IC, split, a1);
        }
    }
}

// Fused CPU conv over the history-prefixed input. Taps index w [IC, OC, *] at w_off (floats);
// the output has dst_oc rows. Exports the history for the next render like with_history.
static ggml_tensor* conv_gemm(render_ctx& rc, ggml_tensor* x, ggml_tensor* w, ggml_tensor* bias, int R, int d,
                              std::vector<size_t> w_off, int64_t dst_oc, ggml_tensor* res = nullptr) {
    ggml_context* ctx = rc.ctx;
    const int64_t C = x->ne[0], T = x->ne[1];
    GGML_ASSERT(w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && ggml_is_contiguous(bias));
    GGML_ASSERT(!res || (ggml_is_contiguous(res) && res->ne[0] == dst_oc && res->ne[1] == T));
    if (!ggml_is_contiguous(x))
        x = ggml_cont(ctx, x);
    ggml_tensor* hist = nullptr;
    if (R > 0) {
        hist = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, R);
        ggml_set_input(hist);
        // Next render's history: xf columns [Tv, Tv + R) with Tv = valid input columns.
        const int64_t Tv = (int64_t)rc.n_valid * rc.cpf;
        ggml_tensor* out;
        if (Tv >= R) {
            out = ggml_cont(ctx, ggml_view_2d(ctx, x, C, R, x->nb[1], (size_t)(Tv - R) * x->nb[1]));
        } else {
            out = ggml_concat(ctx, ggml_view_2d(ctx, hist, C, R - Tv, hist->nb[1], (size_t)Tv * hist->nb[1]),
                              ggml_view_2d(ctx, x, C, Tv, x->nb[1], 0), 1);
        }
        ggml_set_output(out);
        rc.hist_in.push_back(hist);
        rc.hist_out.push_back(out);
    }
    conv_gemm_params& p = rc.conv_params.emplace_back();
    p.K = (int)w_off.size();
    p.d = d;
    p.w_off = std::move(w_off);
    ggml_tensor* args[5] = {x, w, bias, hist, res};
    // One ggml task: Accelerate threads the GEMM itself. Splitting output columns across ggml's
    // threads (all convs, or only IC <= 192) measured slower on M3.
    return ggml_custom_4d(ctx, GGML_TYPE_F32, dst_oc, T, 1, 1, args, 5, conv_gemm_cpu, 1, &p);
}
#endif
// Production encoder Conv1d reduces channels and taps in one BF16 tensor-core
// GEMM. Splitting taps into independent F32 GEMMs changes the accumulation before
// the BF16 output boundary. Decoder rendering retains its existing tap recipe.
static ggml_tensor* encoder_conv(render_ctx& rc, ggml_tensor* w, ggml_tensor* bias, ggml_tensor* x, int dilation,
                                 int stride) {
    ggml_context* ctx = rc.ctx;
    const int64_t IC = w->ne[0], OC = w->ne[1], K = w->ne[2];
    x = encoder_round(rc, x);
    ggml_tensor* weights;
    ggml_tensor* columns;
    if (K == 1 && stride == 1) {
        weights = ggml_cast(ctx, ggml_reshape_2d(ctx, w, IC, OC), GGML_TYPE_BF16);
        columns = x;
    } else {
        // PyTorch [OC,IC,K] order: tap is the fastest reduction coordinate.
        auto* kernel = ggml_cont(ctx, ggml_permute(ctx, w, 1, 2, 0, 3)); // [K,IC,OC]
        weights = ggml_cast(ctx, ggml_reshape_2d(ctx, kernel, K * IC, OC), GGML_TYPE_BF16);
        const int left = (int)(K - 1) * dilation + 1 - stride;
#ifdef ZONOS3_HAVE_CONV_1D_IMPLICIT
        // CrispASR: ggml_conv_1d_implicit comes from upstream's patches/metal-conv1d-implicit.patch, which our
        // vendored ggml does not carry; use_conv1d is only ever set by zac-encoder.cpp (not vendored).
        if (rc.m.use_conv1d) {
            // Implicit GEMM: gathers the im2col windows inside the BF16 matmul tile loader, with
            // the same accumulation order as the im2col path below.
            auto* y = ggml_conv_1d_implicit(ctx, weights, x, (int)K, stride, dilation, left);
            return encoder_round(rc, add_bias(ctx, encoder_round(rc, y), bias));
        }
#endif
        auto* padded = ggml_pad_ext(ctx, x, 0, 0, left, 0, 0, 0, 0, 0);
        auto* input = ggml_cont(ctx, ggml_transpose(ctx, padded)); // [T+left,IC]
        // F32 im2col preserves the already-rounded BF16 inputs exactly; F16
        // im2col would alter their exponent range. GEMM casts columns to BF16.
        auto* windows = ggml_im2col(ctx, kernel, input, stride, 0, 0, 0, dilation, 0, false, GGML_TYPE_F32);
        columns = ggml_reshape_2d(ctx, windows, K * IC, windows->ne[1]);
    }
    auto* y = ggml_mul_mat(ctx, weights, columns);
    ggml_prec_set_acc(y, GGML_PREC_F32);
    // ATen's cuDNN branch returns the BF16 convolution first, then adds bias
    // in-place in BF16. Conv1d does not share Linear's fused bias boundary.
    return encoder_round(rc, add_bias(ctx, encoder_round(rc, y), bias));
}

// Causal dilated conv as a sum of per-tap matmuls; w [IC, OC, K]. `res`, if given, is added.
static ggml_tensor* conv_causal(render_ctx& rc, ggml_tensor* w, ggml_tensor* bias, ggml_tensor* x, int d,
                                ggml_tensor* res = nullptr) {
    ggml_context* ctx = rc.ctx;
    const int64_t IC = w->ne[0], OC = w->ne[1], K = w->ne[2], T = x->ne[1];
    GGML_ASSERT(x->ne[0] == IC);
    if (rc.encoder) {
        auto* y = encoder_conv(rc, w, bias, x, d, 1);
        return res ? ggml_add(ctx, res, y) : y;
    }
#if ZAC_CPU_GEMM
    if (!rc.m.is_gpu && !rc.encoder) {
        std::vector<size_t> off(K);
        for (int64_t k = 0; k < K; ++k)
            off[k] = (size_t)(k * IC * OC);
        return conv_gemm(rc, x, w, bias, (int)(K - 1) * d, d, std::move(off), OC, res);
    }
#endif
    ggml_tensor* xf = with_history(rc, x, (int)(K - 1) * d);
    // Emit all tap matmuls before the adds: consecutive same-layout ADDs then fuse into one
    // Metal kernel (ggml_metal_op_bin fuses up to 8).
    std::vector<ggml_tensor*> taps(K);
    for (int64_t k = 0; k < K; ++k) {
        taps[k] = ggml_mul_mat(ctx, ggml_view_2d(ctx, w, IC, OC, w->nb[1], (size_t)k * w->nb[2]),
                               ggml_view_2d(ctx, xf, IC, T, xf->nb[1], (size_t)k * d * xf->nb[1]));
        ggml_build_forward_expand(rc.graph, taps[k]);
    }
    ggml_tensor* y = taps[0];
    for (int64_t k = 1; k < K; ++k)
        y = ggml_add(ctx, y, taps[k]);
    y = encoder_round(rc, add_bias(ctx, y, bias));
    return res ? ggml_add(ctx, res, y) : y;
}

// sin(v)^2 for one block of floats, branchless so the loop vectorizes (ggml's CPU sin is scalar
// sinf). Cody-Waite reduction v = q*pi/2 + r, |r| <= pi/4, then sin^2(v) is sin^2(r) for even q
// and cos^2(r) for odd q; minimax polynomials are the Cephes sinf/cosf ones (~1 ulp on |v| < 8192).
static inline void sin2_f32(int n, float* y, const float* v) {
    for (int i = 0; i < n; ++i) {
        const float q = rintf(v[i] * 0.63661977236758134f);
        float r = v[i] - q * 1.5703125f;
        r -= q * 4.837512969970703125e-4f;
        r -= q * 7.54978995489188216e-8f;
        const float r2 = r * r;
        const float s = r + r * r2 * ((-1.9515295891e-4f * r2 + 8.3321608736e-3f) * r2 - 1.6666654611e-1f);
        const float c = 1.0f - 0.5f * r2 +
                        r2 * r2 * ((2.443315711809948e-5f * r2 - 1.388731625493765e-3f) * r2 + 4.166664568298827e-2f);
        const float t = ((int)q & 1) ? c : s;
        y[i] = t * t;
    }
}

// Fused CPU snake over columns of a contiguous [C, T] tensor: one pass instead of five ops.
static void snake_cpu(ggml_tensor* dst, const ggml_tensor* x, const ggml_tensor* alpha, const ggml_tensor* inv_alpha,
                      int ith, int nth, void*) {
    const int64_t C = x->ne[0], T = ggml_nrows(x);
    const float *a = (const float*)alpha->data, *ia = (const float*)inv_alpha->data;
    float buf[256];
    for (int64_t t = T * ith / nth, t1 = T * (ith + 1) / nth; t < t1; ++t) {
        const float* xs = (const float*)x->data + t * C;
        float* ys = (float*)dst->data + t * C;
        for (int64_t c0 = 0; c0 < C; c0 += 256) {
            const int n = (int)std::min<int64_t>(256, C - c0);
            for (int i = 0; i < n; ++i)
                buf[i] = a[c0 + i] * xs[c0 + i];
            sin2_f32(n, buf, buf);
            for (int i = 0; i < n; ++i)
                ys[c0 + i] = xs[c0 + i] + buf[i] * ia[c0 + i];
        }
    }
}

// snake(x) = x + (1/(alpha+1e-9)) * sin(alpha*x)^2 ; alpha,inv_alpha stored ne=[1,C] (per-channel).
static ggml_tensor* snake(render_ctx& rc, ggml_tensor* x, const std::string& p) {
    ggml_context* ctx = rc.ctx;
    const int64_t C = x->ne[0];
    ggml_tensor *alpha = rc.m.get(p + ".alpha"), *inv_alpha = rc.m.get(p + ".inv_alpha");
    if (!rc.m.is_gpu) {
        GGML_ASSERT(ggml_is_contiguous(x) && x->type == GGML_TYPE_F32);
        return ggml_map_custom3(ctx, x, alpha, inv_alpha, snake_cpu, GGML_N_TASKS_MAX, nullptr);
    }
    ggml_tensor* ax = ggml_mul(ctx, x, ggml_reshape_2d(ctx, alpha, C, 1));
    ggml_tensor* s2 = ggml_sqr(ctx, ggml_sin(ctx, ax));
    return ggml_add(ctx, x, ggml_mul(ctx, s2, ggml_reshape_2d(ctx, inv_alpha, C, 1)));
}

#if ZAC_CPU_GEMM
// Exact GELU 0.5*x*(1 + erf(x/sqrt2)) over a contiguous F32 tensor (ggml's CPU gelu_erf calls
// scalar erff). erf by Abramowitz-Stegun 7.1.26 (|err| <= 1.5e-7), exp(-z^2) in blocks via vvexpf.
static void gelu_erf_cpu(ggml_tensor* dst, const ggml_tensor* x, int ith, int nth, void*) {
    const int64_t C = x->ne[0], T = ggml_nrows(x);
    float z[256], e[256];
    for (int64_t t = T * ith / nth, t1 = T * (ith + 1) / nth; t < t1; ++t) {
        const float* xs = (const float*)x->data + t * C;
        float* ys = (float*)dst->data + t * C;
        for (int64_t c0 = 0; c0 < C; c0 += 256) {
            const int n = (int)std::min<int64_t>(256, C - c0);
            for (int i = 0; i < n; ++i) {
                z[i] = xs[c0 + i] * 0.70710678118654752f;
                e[i] = -z[i] * z[i];
            }
            vvexpf(e, e, &n);
            for (int i = 0; i < n; ++i) {
                const float a = fabsf(z[i]), u = 1.0f / (1.0f + 0.3275911f * a);
                const float poly =
                    ((((1.061405429f * u - 1.453152027f) * u + 1.421413741f) * u - 0.284496736f) * u + 0.254829592f) *
                    u;
                const float erf_a = 1.0f - poly * e[i];
                ys[c0 + i] = 0.5f * xs[c0 + i] * (1.0f + copysignf(erf_a, z[i]));
            }
        }
    }
}
#endif

static ggml_tensor* gelu(render_ctx& rc, ggml_tensor* x) {
#if ZAC_CPU_GEMM
    if (!rc.m.is_gpu && !rc.encoder) {
        GGML_ASSERT(ggml_is_contiguous(x) && x->type == GGML_TYPE_F32);
        return ggml_map_custom1(rc.ctx, x, gelu_erf_cpu, GGML_N_TASKS_MAX, nullptr);
    }
#endif
    return encoder_round(rc, ggml_gelu_erf(rc.ctx, x));
}

// Causal residual unit.
static ggml_tensor* res_unit(render_ctx& rc, ggml_tensor* x, const std::string& p, int dil) {
    const zac_model& m = rc.m;
    ggml_tensor* y = snake(rc, x, p + ".snake1");
    y = conv_causal(rc, m.get(p + ".conv1.weight"), m.get(p + ".conv1.bias"), y, dil);
    y = snake(rc, y, p + ".snake2");
    return conv_causal(rc, m.get(p + ".conv2.weight"), m.get(p + ".conv2.bias"), y, 1, x);
}

int zac_samples_per_frame(const zac_model& m) {
    return m.hop;
}

bool zac_partial_frames(const zac_model& m) {
    return !m.res_cb_bias.empty();
}

static ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, float eps) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), w), b);
}

// Causal ConvNeXt block (DualCodec semantic head): depthwise k7 over the left-padded input,
// LayerNorm, pw1, exact GELU, pw2, residual. No layer scale.
static ggml_tensor* convnext_block(render_ctx& rc, ggml_tensor* x, const std::string& p) {
    ggml_context* ctx = rc.ctx;
    const zac_model& m = rc.m;
    const int64_t C = x->ne[0], T = x->ne[1];
    ggml_tensor* dw = m.get(p + ".dw.weight"); // [C, K]
    const int K = (int)dw->ne[1];
    ggml_tensor* xf = with_history(rc, rc.encoder ? encoder_round(rc, x) : x, K - 1);
    // h[:, t] = sum_k x[:, t+k-(K-1)] * w[:, k].
    ggml_tensor* h = nullptr;
    for (int k = 0; k < K; ++k) {
        ggml_tensor* term = ggml_mul(ctx, ggml_view_2d(ctx, xf, C, T, xf->nb[1], (size_t)k * xf->nb[1]),
                                     ggml_view_1d(ctx, dw, C, (size_t)k * C * sizeof(float)));
        h = h ? ggml_add(ctx, h, term) : term;
    }
    h = encoder_round(rc, ggml_add(ctx, encoder_round(rc, h), m.get(p + ".dw.bias")));
    h = layer_norm(ctx, h, m.get(p + ".norm.weight"), m.get(p + ".norm.bias"), m.convnext_eps);
    h = encoder_round(rc, h);
    h = gelu(rc,
             encoder_round(rc, ggml_add(ctx, ggml_mul_mat(ctx, m.get(p + ".pw1.weight"), h), m.get(p + ".pw1.bias"))));
    h = encoder_round(rc, ggml_add(ctx, ggml_mul_mat(ctx, m.get(p + ".pw2.weight"), h), m.get(p + ".pw2.bias")));
    ggml_tensor* y = ggml_add(ctx, x, h);
    return rc.input_bf16 ? encoder_round(rc, y) : y;
}

// Patch up h: out[ch, t*h + j] = in[ch*h + j, t].
static ggml_tensor* patch_up(render_ctx& rc, ggml_tensor* x, int h) {
    ggml_context* ctx = rc.ctx;
    const int64_t C = x->ne[0], T = x->ne[1];
    rc.cpf *= h;
    ggml_tensor* y = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, x, h, C / h, T), 1, 0, 2, 3));
    return ggml_reshape_2d(ctx, y, C / h, h * T);
}

// One MOSS layer: pre-LN windowed-causal attention (adjacent-pair RoPE on absolute positions)
// and a GELU FFN, each with a per-channel layer scale. This chunk's keys/values are written into
// the stream's ring cache at rows `slots`; attention reads the whole ring, and `mask` [cap, T]
// keeps only slots holding positions with 0 <= q_pos - k_pos < context and k_pos >= 0.
static ggml_tensor* moss_layer(render_ctx& rc, ggml_tensor* x, const std::string& p, const zac_stage_hp& hp,
                               ggml_tensor* pos, ggml_tensor* slots, ggml_tensor* mask) {
    ggml_context* ctx = rc.ctx;
    const zac_model& m = rc.m;
    const int d = hp.d_model, nh = hp.n_head, hd = d / nh;
    const int64_t T = x->ne[1];
    ggml_tensor* kc = rc.st ? rc.st->kc[rc.layer] : nullptr;
    ggml_tensor* vc = rc.st ? rc.st->vc[rc.layer] : nullptr;
    ++rc.layer;
    ggml_tensor* h = layer_norm(ctx, x, m.get(p + ".norm1.weight"), m.get(p + ".norm1.bias"), m.ln_eps);
    if (rc.encoder && p == "stage.0.l0")
        ggml_set_name(h, "stage.0.l0.norm1");
    if (rc.input_bf16)
        h = encoder_round(rc, h);
    ggml_tensor* qkv = encoder_round(rc, ggml_mul_mat(ctx, m.get(p + ".qkv.weight"), encoder_round(rc, h))); // [3d,T]
    if (rc.encoder && p == "stage.0.l0")
        ggml_set_name(qkv, "stage.0.l0.qkv");
    auto section = [&](int s) {
        return ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, nh, T, (size_t)hd * sizeof(float), qkv->nb[1],
                                           (size_t)s * d * sizeof(float)));
    };
    auto rope = [&](ggml_tensor* z) {
        return encoder_round(rc, ggml_rope_ext(ctx, z, pos, nullptr, hd, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f,
                                               1.0f, 0.0f, 0.0f));
    };
    ggml_tensor* q = rope(section(0));                                                 // [hd, nh, T]
    ggml_tensor* kb = ggml_cont(ctx, ggml_permute(ctx, rope(section(1)), 0, 2, 1, 3)); // [hd, T, nh]
    ggml_tensor* vb = ggml_cont(ctx, ggml_permute(ctx, section(2), 0, 2, 1, 3));
    // The returned views stand for the caches after the write, so attention depends on it.
    ggml_tensor* kw = rc.st ? ggml_set_rows(ctx, kc, kb, slots) : kb; // [hd,cap,nh]
    ggml_tensor* vw = rc.st ? ggml_set_rows(ctx, vc, vb, slots) : vb;

    ggml_tensor* qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [hd, T, nh]
    ggml_tensor* o;
    if (rc.encoder) {
        // The frozen reference uses 512-query blocks and only the intersecting left window.
        // Avoid a heads*T*T attention workspace for the 4,800-column first encoder stage.
        o = nullptr;
        for (int a = 0; a < T; a += 512) {
            const int end = std::min<int64_t>(a + 512, T);
            const int left = std::max(0, a - hp.context + 1), nk = end - left, nq = end - a;
            auto* qb = ggml_view_3d(ctx, qp, hd, nq, nh, qp->nb[1], qp->nb[2], a * qp->nb[1]);
            auto* kk = ggml_view_3d(ctx, kw, hd, nk, nh, kw->nb[1], kw->nb[2], left * kw->nb[1]);
            auto* vv = ggml_view_3d(ctx, vw, hd, nk, nh, vw->nb[1], vw->nb[2], left * vw->nb[1]);
            auto* mb =
                ggml_cont(ctx, ggml_view_2d(ctx, mask, nk, nq, mask->nb[1], left * mask->nb[0] + a * mask->nb[1]));
            auto* score = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, kk, qb), mb, 1.0f / sqrtf((float)hd), 0.0f);
            auto* vt = ggml_cont(ctx, ggml_permute(ctx, vv, 1, 0, 2, 3));
            auto* ob = ggml_mul_mat(ctx, vt, score);
            ob = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, ob, 0, 2, 1, 3)), d, nq);
            o = o ? ggml_concat(ctx, o, ob, 1) : ob;
        }
    } else if (m.use_fattn) {
        o = ggml_flash_attn_ext(ctx, qp, kw, vw, mask, 1.0f / sqrtf((float)hd), 0.0f, 0.0f);
        ggml_prec_set_acc(o, GGML_PREC_F32);
        o = ggml_reshape_2d(ctx, o, d, T); // [hd, nh, T]
    } else {
        ggml_tensor* kq = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, kw, qp), mask, 1.0f / sqrtf((float)hd), 0.0f);
        ggml_tensor* vp = ggml_cont(ctx, ggml_permute(ctx, vw, 1, 0, 2, 3)); // [cap, hd, nh]
        o = ggml_mul_mat(ctx, vp, kq);                                       // [hd, T, nh]
        o = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3)), d, T);
    }
    o = encoder_round(rc, o);
    if (rc.encoder && p == "stage.0.l0")
        ggml_set_name(o, "stage.0.l0.sdpa");
    o = ggml_mul(ctx, encoder_round(rc, ggml_mul_mat(ctx, m.get(p + ".wo.weight"), o)), m.get(p + ".attn_scale"));
    x = ggml_add(ctx, x, o);
    h = layer_norm(ctx, x, m.get(p + ".norm2.weight"), m.get(p + ".norm2.bias"), m.ln_eps);
    h = encoder_round(
        rc,
        ggml_mul_mat(ctx, m.get(p + ".ffn2.weight"),
                     gelu(rc, encoder_round(rc, ggml_mul_mat(ctx, m.get(p + ".ffn1.weight"), encoder_round(rc, h))))));
    return ggml_add(ctx, x, ggml_mul(ctx, h, m.get(p + ".ffn_scale")));
}

namespace zac_encoder_detail {
static render_ctx lane(ggml_context* ctx, ggml_cgraph* graph, const zac_model& m, ggml_tensor* x) {
    render_ctx rc{ctx, m, (int)x->ne[1], 1, {}, {}, {}, graph, nullptr, 0};
    rc.encoder = true;
    return rc;
}
ggml_tensor* conv(ggml_context* ctx, ggml_cgraph* graph, const zac_model& m, ggml_tensor* x, const std::string& p,
                  int dilation, int stride) {
    auto rc = lane(ctx, graph, m, x);
    auto* w = m.get(p + ".weight");
    auto* b = m.get(p + ".bias");
    if (stride == 1)
        return conv_causal(rc, w, b, x, dilation);
    return encoder_conv(rc, w, b, x, dilation, stride);
}
ggml_tensor* snake(ggml_context* ctx, ggml_cgraph* graph, const zac_model& m, ggml_tensor* x, const std::string& p) {
    auto rc = lane(ctx, graph, m, x);
    return ::snake(rc, x, p);
}
ggml_tensor* residual(ggml_context* ctx, ggml_cgraph* graph, const zac_model& m, ggml_tensor* x, const std::string& p,
                      int dilation) {
    auto rc = lane(ctx, graph, m, x);
    return res_unit(rc, x, p, dilation);
}
ggml_tensor* convnext(ggml_context* ctx, ggml_cgraph* graph, const zac_model& m, ggml_tensor* x, const std::string& p,
                      bool input_bf16) {
    auto rc = lane(ctx, graph, m, x);
    rc.input_bf16 = input_bf16;
    return convnext_block(rc, x, p);
}
ggml_tensor* moss(ggml_context* ctx, ggml_cgraph* graph, const zac_model& m, ggml_tensor* x, const std::string& p,
                  const zac_stage_hp& hp, ggml_tensor* pos, ggml_tensor* mask, bool first) {
    auto rc = lane(ctx, graph, m, x);
    rc.input_bf16 = first;
    return moss_layer(rc, x, p, hp, pos, nullptr, mask);
}
} // namespace zac_encoder_detail

// GPU graphs take their width from a small ladder (right padding cannot affect earlier causal
// samples) so the reused allocator sees only a few shapes. Streamed GPU output matches buffered
// output to sub-millisample level, not bit-for-bit (reductions depend on the graph width).
constexpr int GPU_MAX_CHUNK = 32;
constexpr int GPU_WIDTHS[] = {4, 8, 16, 32};
constexpr int CPU_CHUNK = 64; // measured best of {32, 64, 128} on M3 (buffered RTF 0.267/0.241/0.306)

static int gpu_width(int n) {
    for (int w : GPU_WIDTHS)
        if (w >= n)
            return w;
    return GPU_MAX_CHUNK;
}

zac_stream::~zac_stream() {
    if (buf_kv)
        ggml_backend_buffer_free(buf_kv);
    if (ctx_kv)
        ggml_free(ctx_kv);
    if (buf_hist)
        ggml_backend_buffer_free(buf_hist);
    if (ctx_hist)
        ggml_free(ctx_hist);
}

zac_stream::zac_stream(zac_stream&& o) noexcept {
    *this = std::move(o);
}

zac_stream& zac_stream::operator=(zac_stream&& o) noexcept {
    if (this == &o)
        return *this;
    if (buf_kv)
        ggml_backend_buffer_free(buf_kv);
    if (ctx_kv)
        ggml_free(ctx_kv);
    if (buf_hist)
        ggml_backend_buffer_free(buf_hist);
    if (ctx_hist)
        ggml_free(ctx_hist);
    n_done = o.n_done;
    conv = std::move(o.conv);
    ola = std::move(o.ola);
    env = std::move(o.env);
    ola_base = o.ola_base;
    head_frames = o.head_frames;
    head_done = o.head_done;
    mix_in = std::move(o.mix_in);
    mix_avail = o.mix_avail;
    mix_base = o.mix_base;
    mix_next = o.mix_next;
    rs_buf = std::move(o.rs_buf);
    rs_base = o.rs_base;
    rs_in = o.rs_in;
    rs_next_block = o.rs_next_block;
    owner = o.owner;
    ctx_kv = o.ctx_kv;
    buf_kv = o.buf_kv;
    kc = std::move(o.kc);
    vc = std::move(o.vc);
    cap = std::move(o.cap);
    kv_id = o.kv_id;
    ctx_hist = o.ctx_hist;
    buf_hist = o.buf_hist;
    hist = std::move(o.hist);
    o.owner = nullptr;
    o.ctx_kv = nullptr;
    o.buf_kv = nullptr;
    o.kc.clear();
    o.vc.clear();
    o.cap.clear();
    o.kv_id = 0;
    o.ctx_hist = nullptr;
    o.buf_hist = nullptr;
    o.hist.clear();
    return *this;
}

// Per-stage K/V ring caches sized so the window [p0 - context + 1, p0 + T - 1] of any chunk
// (T <= max chunk width in columns) maps to distinct rows.
static bool zac_stream_alloc(const zac_model& m, zac_stream& st) {
    const int max_chunk = m.is_gpu ? GPU_MAX_CHUNK : CPU_CHUNK;
    size_t n_tensors = 0;
    for (const zac_stage_hp& hp : m.stages)
        n_tensors += 2 * (size_t)hp.n_layer;
    st.ctx_kv = ggml_init({n_tensors * ggml_tensor_overhead(), nullptr, true});
    // F32 on CPU: ggml's CPU flash attention accumulates V*softmax in F16 for an F16 cache, which
    // made streamed (block 3) and buffered output differ by 2.4e-3. Metal accumulates in F32.
    const ggml_type kv_type = m.is_gpu ? GGML_TYPE_F16 : GGML_TYPE_F32;
    int cpf = 1;
    for (const zac_stage_hp& hp : m.stages) {
        const int cap = hp.context - 1 + max_chunk * cpf, hd = hp.d_model / hp.n_head;
        st.cap.push_back(cap);
        for (int l = 0; l < hp.n_layer; ++l) {
            st.kc.push_back(ggml_new_tensor_3d(st.ctx_kv, kv_type, hd, cap, hp.n_head));
            st.vc.push_back(ggml_new_tensor_3d(st.ctx_kv, kv_type, hd, cap, hp.n_head));
        }
        cpf *= hp.patch;
    }
    st.buf_kv = ggml_backend_alloc_ctx_tensors(st.ctx_kv, m.backend);
    if (!st.buf_kv) {
        fprintf(stderr, "zac: stream KV cache allocation failed\n");
        ggml_free(st.ctx_kv);
        st.ctx_kv = nullptr;
        st.kc.clear();
        st.vc.clear();
        st.cap.clear();
        return false;
    }
    ggml_backend_buffer_clear(st.buf_kv, 0); // masked slots are read too: no NaN garbage
    static std::atomic<uint64_t> next_kv_id{0};
    st.kv_id = ++next_kv_id;
    st.owner = &m;
    return true;
}

// ---- host iSTFT + FIR mixer ----
// Streaming equivalent of torch.istft(center=True, length=frames*hop) per branch followed by the
// non-causal mixer Conv1d (2 -> 1, taps K, zero padding (K-1)/2). Frame t of branch i overlaps
// padded positions [t*hop, t*hop + n); branch sample m is padded position m + n/2.

// Emit mixer outputs y[m] for every m whose taps are available. `flush`: all remaining m < L.
static void mix_emit(const zac_model& m, zac_stream& st, std::vector<float>& out) {
    const int K = m.fir_kernel;
    // mix_in index j holds s[j - h]; y[m] = sum_c sum_k w[c][k] * mix_in_c[m + k].
    const int64_t avail = std::min(st.mix_avail[0], st.mix_avail[1]);
    const int64_t m1 = avail - K + 1; // exclusive bound on m
    if (m1 <= st.mix_next)
        return;
    const size_t n = (size_t)(m1 - st.mix_next), o0 = out.size();
    out.resize(o0 + n, 0.0f);
    float* y = out.data() + o0;
    for (int c = 0; c < 2; ++c) {
        const float* w = m.mixer.data() + (size_t)c * K;
        const float* s = st.mix_in[c].data() + (st.mix_next - st.mix_base);
        for (int k = 0; k < K; ++k) {
            const float wk = w[k];
            for (size_t i = 0; i < n; ++i)
                y[i] += wk * s[i + k];
        }
    }
    st.mix_next = m1;
    for (int c = 0; c < 2; ++c) {
        // Later outputs read from index mix_next on.
        if (st.mix_next > st.mix_base) {
            st.mix_in[c].erase(st.mix_in[c].begin(), st.mix_in[c].begin() + (st.mix_next - st.mix_base));
        }
    }
    st.mix_base = st.mix_next;
}

// Finalize overlap-add positions [ola_base, pf) of branch i into the mixer input (dropping the
// centre padding m < 0).
static void ola_finalize(const zac_model& m, zac_stream& st, int i, int64_t pf) {
    const int64_t pad = m.n_fft[i] / 2;
    const int64_t n = pf - st.ola_base[i];
    if (n <= 0)
        return;
    for (int64_t j = 0; j < n; ++j) {
        if (st.ola_base[i] + j < pad)
            continue;
        st.mix_in[i].push_back(st.ola[i][j] / st.env[i][j]);
        ++st.mix_avail[i];
    }
    st.ola[i].erase(st.ola[i].begin(), st.ola[i].begin() + n);
    st.env[i].erase(st.env[i].begin(), st.env[i].begin() + n);
    st.ola_base[i] = pf;
}

static void head_init(const zac_model& m, zac_stream& st) {
    const int h = (m.fir_kernel - 1) / 2;
    for (int i = 0; i < 2; ++i) {
        st.mix_in[i].assign((size_t)h, 0.0f);
        st.mix_avail[i] = h;
    }
    st.mix_base = 0;
    st.mix_next = 0;
}

// Add `n_cols` head frames: y0 [n_fft0 + 2, n_cols], y1 [n_fft1 + 2, n_cols] (column-major as read
// from the backend: logmag[0:bins], phase[bins:2*bins]); append the finished 44160 Hz samples.
static void head_push(const zac_model& m, zac_stream& st, const float* y0, const float* y1, int64_t n_cols,
                      std::vector<float>& out) {
    if (st.head_frames == 0)
        head_init(m, st);
    const float* y[2] = {y0, y1};
    const int hop = m.head_hop;
    std::vector<float> buf;
    for (int i = 0; i < 2; ++i) {
        const int n = m.n_fft[i], bins = n / 2 + 1;
        const float* win = m.window[i].data();
        buf.resize((size_t)n);
        for (int64_t c = 0; c < n_cols; ++c) {
            const int64_t t = st.head_frames + c;
            const float* f = y[i] + (size_t)c * (n + 2);
            const float *logmag = f, *phase = f + bins;
            auto mag = [&](int k) { return expf(std::min(logmag[k], m.max_log_mag)); };
            buf[0] = mag(0) * cosf(phase[0]);
            for (int k = 1; k < n / 2; ++k) {
                const float a = mag(k);
                buf[2 * k - 1] = a * cosf(phase[k]);
                buf[2 * k] = a * sinf(phase[k]);
            }
            buf[n - 1] = mag(n / 2) * cosf(phase[n / 2]);
            m.irfft[i]->exec(buf.data(), 1.0f / (float)n, false);
            const size_t need = (size_t)(t * hop + n - st.ola_base[i]);
            if (st.ola[i].size() < need) {
                st.ola[i].resize(need, 0.0f);
                st.env[i].resize(need, 0.0f);
            }
            float* ola = st.ola[i].data() + (t * hop - st.ola_base[i]);
            float* env = st.env[i].data() + (t * hop - st.ola_base[i]);
            for (int j = 0; j < n; ++j) {
                ola[j] += win[j] * buf[j];
                env[j] += win[j] * win[j];
            }
        }
    }
    st.head_frames += n_cols;
    for (int i = 0; i < 2; ++i)
        ola_finalize(m, st, i, st.head_frames * hop);
    mix_emit(m, st, out);
}

// End of stream: every sample m < L = head_frames*hop is final (the envelope only covers existing
// frames, like torch); the mixer's right zero padding is h zeros after the last branch sample.
static void head_flush(const zac_model& m, zac_stream& st, std::vector<float>& out) {
    if (st.head_done || st.head_frames == 0)
        return;
    st.head_done = true;
    const int h = (m.fir_kernel - 1) / 2;
    for (int i = 0; i < 2; ++i) {
        ola_finalize(m, st, i, st.ola_base[i] + (int64_t)st.ola[i].size());
        // Samples m >= L belong to the discarded tail padding; ola_finalize above produced them too.
        const int64_t L = st.head_frames * m.head_hop;
        const int64_t keep = h + L - st.mix_base;
        st.mix_in[i].resize((size_t)keep, 0.0f);
        st.mix_in[i].resize((size_t)keep + h, 0.0f);
        st.mix_avail[i] = st.mix_base + keep + h;
    }
    mix_emit(m, st, out);
}

// One decode graph: `gf` columns of which `n_valid` frames are real (the history export reads at n_valid).
// GPU graphs of complete chunks are cached in zac_model::graphs and bind the K/V caches of the stream with id kv_id;
// a chunk with missing residual cells (early partial-frame decode) builds its own graph.
struct zac_model::zac_graph {
    ggml_context* ctx = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_gallocr_t alloc = nullptr;
    std::vector<ggml_tensor*> ids, hist_in, hist_out;
    std::vector<ggml_tensor*> keep;  // partial graphs: [1, gf] 0/1 mask per residual codebook with missing cells
    ggml_tensor* res_bias = nullptr; // partial graphs: [latent, gf] per-frame residual bias
    ggml_tensor* spec[2] = {nullptr, nullptr}; // [n_fft + 2, 16*gf]
    struct stage {
        ggml_tensor *pos, *slots, *mask;
        int cpf, T, context, cap;
    };
    std::vector<stage> stages;
    std::deque<conv_gemm_params> conv_params; // CPU fused convs: read during compute
    uint64_t kv_id = 0;
    // Host staging of the per-call inputs.
    std::vector<int32_t> col, pos, slots;
    std::vector<ggml_fp16_t> mask_host;
    zac_graph() = default;
    zac_graph(const zac_graph&) = delete;
    zac_graph& operator=(const zac_graph&) = delete;
    ~zac_graph() {
        if (alloc)
            ggml_gallocr_free(alloc);
        if (ctx)
            ggml_free(ctx);
    }
};

// Builds and allocates (GPU: own allocator; CPU: m.sched or a fresh allocator) the decode graph. `cb_missing` (empty
// for a complete chunk) marks the residual codebooks with a missing cell in the chunk.
static std::shared_ptr<zac_model::zac_graph> build_decode_graph(const zac_model& m, zac_stream& st, int gf, int n_valid,
                                                                const std::vector<bool>& cb_missing) {
    const int ncb = m.n_codebooks;
    auto g = std::make_shared<zac_model::zac_graph>();
    g->ctx = ggml_init({(size_t)64 * 1024 * 1024, nullptr, true});
    ggml_context* ctx = g->ctx;
    g->graph = ggml_new_graph_custom(ctx, 16384, false);
    ggml_cgraph* graph = g->graph;
    g->kv_id = st.kv_id;
    render_ctx rc{ctx, m, n_valid, 1, {}, {}, {}, graph, &st, 0};

    g->ids.resize(ncb);
    g->keep.assign(ncb, nullptr);
    for (int c = 0; c < ncb; ++c) {
        g->ids[c] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, gf);
        ggml_set_input(g->ids[c]);
    }
    ggml_tensor* x = ggml_add(ctx, ggml_get_rows(ctx, m.get("sem.table"), g->ids[0]), m.get("sem.bias"));
    for (int b = 0; b < m.sem_depth; ++b)
        x = convnext_block(rc, x, "sem_head." + std::to_string(b));
    x = ggml_add(ctx, ggml_mul_mat(ctx, m.get("sem_head.out.weight"), x), m.get("sem_head.out.bias"));
    for (int c = 1; c < ncb; ++c) {
        ggml_tensor* part = ggml_get_rows(ctx, m.get("res." + std::to_string(c - 1) + ".table"), g->ids[c]);
        if (!cb_missing.empty() && cb_missing[c]) {
            g->keep[c] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, gf);
            ggml_set_input(g->keep[c]);
            part = ggml_mul(ctx, part, g->keep[c]);
        }
        x = ggml_add(ctx, x, part);
    }
    ggml_tensor* res_bias = m.get("res.bias");
    if (!cb_missing.empty()) {
        res_bias = g->res_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m.latent_dim, gf);
        ggml_set_input(res_bias);
    }
    x = ggml_add(ctx, x, res_bias); // [latent, gf]

    for (size_t s = 0; s < m.stages.size(); ++s) {
        const zac_stage_hp& hp = m.stages[s];
        const std::string p = "stage." + std::to_string(s);
        x = ggml_mul_mat(ctx, m.get(p + ".in.weight"), x);
        const int T = (int)x->ne[1], cap = st.cap[s];
        zac_model::zac_graph::stage si{ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T),
                                       ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T),
                                       ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cap, T),
                                       rc.cpf,
                                       T,
                                       hp.context,
                                       cap};
        ggml_set_input(si.pos);
        ggml_set_input(si.slots);
        ggml_set_input(si.mask);
        g->stages.push_back(si);
        for (int l = 0; l < hp.n_layer; ++l)
            x = moss_layer(rc, x, p + ".l" + std::to_string(l), hp, si.pos, si.slots, si.mask);
        if (hp.out_dim > 0)
            x = ggml_mul_mat(ctx, m.get(p + ".out.weight"), x);
        if (hp.patch > 1)
            x = patch_up(rc, x, hp.patch);
    }
    for (int b = 0; b < m.head_depth; ++b)
        x = convnext_block(rc, x, "head." + std::to_string(b));
    x = layer_norm(ctx, x, m.get("head.norm.weight"), m.get("head.norm.bias"), m.convnext_eps);
    for (int i = 0; i < 2; ++i) {
        const std::string p = "head.out." + std::to_string(i);
        g->spec[i] = ggml_add(ctx, ggml_mul_mat(ctx, m.get(p + ".weight"), x), m.get(p + ".bias"));
        ggml_set_output(g->spec[i]);
        ggml_build_forward_expand(graph, g->spec[i]);
    }
    for (ggml_tensor* h : rc.hist_out)
        ggml_build_forward_expand(graph, h);
    // Chunk width may select different CUDA GEMM kernels. Keep their operand/
    // accumulation precision identical for buffered and streamed rendering.
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto* node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_MUL_MAT)
            ggml_prec_set_acc(node, GGML_PREC_F32);
    }
    g->hist_in = std::move(rc.hist_in);
    g->hist_out = std::move(rc.hist_out);
    g->conv_params = std::move(rc.conv_params); // deque move keeps the element addresses the custom ops hold

    bool ok;
    if (m.sched) {
        ggml_backend_sched_reset(m.sched);
        ok = ggml_backend_sched_alloc_graph(m.sched, graph);
    } else {
        g->alloc = ggml_gallocr_new(m.buft);
        ok = g->alloc && ggml_gallocr_alloc_graph(g->alloc, graph);
    }
    if (!ok) {
        fprintf(stderr, "zac: graph allocation failed\n");
        return nullptr;
    }
    return g;
}

// GPU: device-resident conv histories of `st`, zeroed (= the first chunk's history), shaped like g's inputs.
static bool alloc_stream_hist(const zac_model& m, zac_stream& st, const zac_model::zac_graph& g) {
    st.ctx_hist = ggml_init({(g.hist_in.size() + 1) * ggml_tensor_overhead(), nullptr, true});
    for (ggml_tensor* h : g.hist_in)
        st.hist.push_back(ggml_new_tensor(st.ctx_hist, h->type, GGML_MAX_DIMS, h->ne));
    st.buf_hist = ggml_backend_alloc_ctx_tensors(st.ctx_hist, m.backend);
    if (!st.buf_hist) {
        fprintf(stderr, "zac: stream history allocation failed\n");
        ggml_free(st.ctx_hist);
        st.ctx_hist = nullptr;
        st.hist.clear();
        return false;
    }
    ggml_backend_buffer_clear(st.buf_hist, 0);
    return true;
}

// Decode frames [f0, f0 + n_valid) through the trunk and head projections, advancing the conv/KV
// history in `st`, and append the internal-rate samples that are final so far to `internal`.
// A residual cell whose row is >= H or whose code is negative is missing (zac_stream_decode).
static bool decode_chunk(const zac_model& m, zac_stream& st, const int32_t* codes, int H, int f0, int n_valid,
                         std::vector<float>& internal) {
    const int ncb = m.n_codebooks, gf = m.is_gpu ? gpu_width(n_valid) : n_valid, L = m.latent_dim;
    // miss[c * n_valid + i]: residual codebook c of frame f0 + i is missing. A chunk without missing cells runs the
    // plain graph; one with them multiplies each affected codebook's rows by a 0/1 keep mask (1 is exact) and adds a
    // per-frame bias (res.bias itself for complete frames, minus the missing codebooks' res.{c-1}.bias otherwise).
    std::vector<uint8_t> miss((size_t)ncb * n_valid, 0);
    std::vector<bool> cb_missing(ncb, false);
    bool partial = false;
    for (int i = 0; i < n_valid; ++i) {
        if (codes[(size_t)(f0 + i) * ncb] < 0) {
            fprintf(stderr, "zac: frame %d has no semantic code\n", f0 + i);
            return false;
        }
        for (int c = 1; c < ncb; ++c) {
            const int t = f0 + i + c;
            if (t < H && codes[(size_t)t * ncb + c] >= 0)
                continue;
            miss[(size_t)c * n_valid + i] = 1;
            cb_missing[c] = partial = true;
        }
    }
    if (partial && !zac_partial_frames(m)) {
        fprintf(stderr, "zac: missing residual codes need a decoder GGUF with res.{i}.bias (reconvert with "
                        "tools/convert_gguf.py codec-decoder)\n");
        return false;
    }
    if (!partial)
        cb_missing.clear();
    std::shared_ptr<zac_model::zac_graph> g;
    if (m.is_gpu && !partial) {
        auto& slot = m.graphs[{gf, n_valid}];
        if (!slot || slot->kv_id != st.kv_id) {
            slot.reset(); // release the old graph's buffers before allocating the new one
            slot = build_decode_graph(m, st, gf, n_valid, cb_missing);
            if (!slot)
                return false;
        }
        g = slot;
    } else {
        g = build_decode_graph(m, st, gf, n_valid, cb_missing);
        if (!g)
            return false;
    }
    if (m.is_gpu) {
        if (st.hist.empty() && !alloc_stream_hist(m, st, *g))
            return false;
        if (st.hist.size() != g->hist_in.size()) {
            fprintf(stderr, "zac: stream state mismatch\n");
            return false;
        }
    } else if (!st.conv.empty() && st.conv.size() != g->hist_in.size()) {
        fprintf(stderr, "zac: stream state mismatch\n");
        return false;
    }

    // Un-shear: codebook c of frame t is row t + c; rows past H (right-padding columns, missing cells) are clamped
    // to a valid id, a missing cell's row then multiplied by 0.
    g->col.resize(gf);
    for (int c = 0; c < ncb; ++c) {
        for (int i = 0; i < gf; ++i) {
            const int t = f0 + i + c;
            g->col[i] = std::clamp(t < H ? codes[(size_t)t * ncb + c] : 4097, 0, m.codebook_size - 1);
        }
        ggml_backend_tensor_set(g->ids[c], g->col.data(), 0, (size_t)gf * sizeof(int32_t));
    }
    if (partial) {
        std::vector<float> kv(gf);
        for (int c = 1; c < ncb; ++c) {
            if (!g->keep[c])
                continue;
            for (int i = 0; i < gf; ++i)
                kv[i] = i < n_valid && miss[(size_t)c * n_valid + i] ? 0.0f : 1.0f;
            ggml_backend_tensor_set(g->keep[c], kv.data(), 0, (size_t)gf * sizeof(float));
        }
        std::vector<float> bias((size_t)L * gf);
        std::vector<double> v(L);
        for (int i = 0; i < gf; ++i) {
            float* b = bias.data() + (size_t)i * L;
            std::copy(m.res_bias.begin(), m.res_bias.end(), b);
            bool left_out = false;
            for (int c = 1; c < ncb && i < n_valid; ++c) {
                if (!miss[(size_t)c * n_valid + i])
                    continue;
                if (!left_out)
                    std::copy(m.res_bias.begin(), m.res_bias.end(), v.begin());
                left_out = true;
                const float* cb = m.res_cb_bias.data() + (size_t)(c - 1) * L;
                for (int j = 0; j < L; ++j)
                    v[j] -= cb[j];
            }
            if (left_out)
                std::transform(v.begin(), v.end(), b, [](double d) { return (float)d; });
        }
        ggml_backend_tensor_set(g->res_bias, bias.data(), 0, bias.size() * sizeof(float));
    }

    const ggml_fp16_t M0 = ggml_fp32_to_fp16(0.0f), MINF = ggml_fp32_to_fp16(-INFINITY);
    for (const auto& si : g->stages) {
        const int64_t p0 = (int64_t)f0 * si.cpf, p_last = p0 + si.T - 1, cap = si.cap;
        g->pos.resize(si.T);
        g->slots.resize(si.T);
        for (int i = 0; i < si.T; ++i) {
            g->pos[i] = (int32_t)(p0 + i);
            g->slots[i] = (int32_t)((p0 + i) % cap);
        }
        ggml_backend_tensor_set(si.pos, g->pos.data(), 0, g->pos.size() * sizeof(int32_t));
        ggml_backend_tensor_set(si.slots, g->slots.data(), 0, g->slots.size() * sizeof(int32_t));
        // Slot j holds the latest position written there: cap >= context - 1 + T, so this chunk's
        // window [p0 - context + 1, p_last] occupies distinct slots. Right-padding columns write
        // future positions that causality hides and the next chunk rewrites before use.
        std::vector<ggml_fp16_t>& mask = g->mask_host;
        mask.resize((size_t)cap * si.T);
        for (int64_t j = 0; j < cap; ++j) {
            const int64_t kp = p_last - (((p_last - j) % cap) + cap) % cap;
            for (int i = 0; i < si.T; ++i) {
                const int64_t qp = p0 + i;
                mask[(size_t)i * cap + j] = (kp >= 0 && kp <= qp && qp - kp < si.context) ? M0 : MINF;
            }
        }
        ggml_backend_tensor_set(si.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }
    if (m.is_gpu) {
        for (size_t i = 0; i < g->hist_in.size(); ++i)
            ggml_backend_tensor_copy(st.hist[i], g->hist_in[i]);
    } else {
        std::vector<float> zeros;
        for (size_t i = 0; i < g->hist_in.size(); ++i) {
            ggml_tensor* h = g->hist_in[i];
            if (!st.conv.empty()) {
                ggml_backend_tensor_set(h, st.conv[i].data(), 0, ggml_nbytes(h));
                continue;
            }
            zeros.assign((size_t)ggml_nelements(h), 0.0f);
            ggml_backend_tensor_set(h, zeros.data(), 0, ggml_nbytes(h));
        }
    }

    const ggml_status status =
        m.sched ? ggml_backend_sched_graph_compute(m.sched, g->graph) : ggml_backend_graph_compute(m.backend, g->graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "zac: compute failed\n");
        return false;
    }
    const int64_t n_cols = (int64_t)n_valid * (m.internal_hop / m.head_hop); // head frames in this chunk
    std::array<std::vector<float>, 2> spec_h;
    for (int i = 0; i < 2; ++i) {
        spec_h[i].resize((size_t)(m.n_fft[i] + 2) * n_cols);
        ggml_backend_tensor_get(g->spec[i], spec_h[i].data(), 0, spec_h[i].size() * sizeof(float));
    }
    if (m.is_gpu) {
        for (size_t i = 0; i < g->hist_out.size(); ++i)
            ggml_backend_tensor_copy(g->hist_out[i], st.hist[i]);
    } else {
        st.conv.resize(g->hist_out.size());
        for (size_t i = 0; i < g->hist_out.size(); ++i) {
            st.conv[i].resize((size_t)ggml_nelements(g->hist_out[i]));
            ggml_backend_tensor_get(g->hist_out[i], st.conv[i].data(), 0, ggml_nbytes(g->hist_out[i]));
        }
    }
    g.reset(); // CPU: frees the per-call graph before the host iSTFT
    head_push(m, st, spec_h[0].data(), spec_h[1].data(), n_cols, internal);
    return true;
}

// torchaudio sinc resampler, 44160 -> 44100 (orig 736 -> new 735 per block):
//   out[n*new + i] = sum_k K[i][k] * x[n*orig + k - width],  x = 0 outside [0, rs_in).
// Block n needs x up to n*orig + (2*width + orig - 1) - width; without `flush` it waits for those
// samples, with `flush` it zero-extends and truncates the total to ceil(new * rs_in / orig).
static void resample(const zac_model& m, zac_stream& st, bool flush, std::vector<float>& audio) {
    const int O = m.rs_orig, N = m.rs_new, W = m.rs_width, KL = 2 * W + O;
    const int64_t total = flush ? ((int64_t)N * st.rs_in + O - 1) / O : -1;
    std::vector<float> xw(KL), y(N);
    for (;; ++st.rs_next_block) {
        const int64_t n = st.rs_next_block, x0 = n * O - W;
        if (flush ? n * N >= total : x0 + KL > st.rs_in)
            break;
        for (int k = 0; k < KL; ++k) {
            const int64_t j = x0 + k;
            xw[k] = (j < 0 || j >= st.rs_in) ? 0.0f : st.rs_buf[(size_t)(j - st.rs_base)];
        }
#if ZAC_CPU_GEMM
        cblas_sgemv(CblasRowMajor, CblasNoTrans, N, KL, 1.0f, m.rs_kernel.data(), KL, xw.data(), 1, 0.0f, y.data(), 1);
#else
        for (int i = 0; i < N; ++i) {
            const float* ki = m.rs_kernel.data() + (size_t)i * KL;
            float acc = 0.0f;
            for (int k = 0; k < KL; ++k)
                acc += ki[k] * xw[k];
            y[i] = acc;
        }
#endif
        const int64_t emit = flush ? std::min<int64_t>(N, total - n * N) : N;
        audio.insert(audio.end(), y.begin(), y.begin() + emit);
    }
    // Drop samples no future block reads.
    const int64_t keep_from = std::min(st.rs_in, std::max<int64_t>(0, st.rs_next_block * O - W));
    if (keep_from > st.rs_base) {
        st.rs_buf.erase(st.rs_buf.begin(), st.rs_buf.begin() + (keep_from - st.rs_base));
        st.rs_base = keep_from;
    }
}

bool zac_stream_decode(const zac_model& m, zac_stream& st, const int32_t* codes, int H, int f_hi,
                       std::vector<float>& audio) {
    if (f_hi < 0 || H < 0 || (f_hi > st.n_done && (!codes || f_hi > H)))
        return false;
    if (st.kc.empty() && !zac_stream_alloc(m, st))
        return false;
    if (st.owner != &m) {
        fprintf(stderr, "zac: stream used with a different codec model\n");
        return false;
    }
    const int chunk = m.is_gpu ? GPU_MAX_CHUNK : CPU_CHUNK;
    std::vector<float> internal;
    while (st.n_done < f_hi) {
        const int n = std::min(chunk, f_hi - st.n_done);
        internal.clear();
        if (!decode_chunk(m, st, codes, H, st.n_done, n, internal))
            return false;
        st.n_done += n;
        st.rs_buf.insert(st.rs_buf.end(), internal.begin(), internal.end());
        st.rs_in += (int64_t)internal.size();
        resample(m, st, false, audio);
    }
    return true;
}

bool zac_stream_flush(const zac_model& m, zac_stream& st, std::vector<float>& audio) {
    std::vector<float> internal;
    head_flush(m, st, internal);
    st.rs_buf.insert(st.rs_buf.end(), internal.begin(), internal.end());
    st.rs_in += (int64_t)internal.size();
    resample(m, st, true, audio);
    return true;
}

float zac_output_gain(const zac_model& m, double headroom_db) {
    const double db = headroom_db < 0 ? (double)m.headroom_db : headroom_db;
    return powf(10.0f, (float)(-db / 20.0));
}

bool zac_decode(const zac_model& m, const int32_t* codes, int H, int W, int eos, std::vector<float>& audio) {
    if (W != m.n_codebooks) {
        fprintf(stderr, "zac: codes have %d codebooks, expected %d\n", W, m.n_codebooks);
        return false;
    }
    const int E = std::min(eos >= 0 ? eos : H, H - (m.n_codebooks - 1));
    if (E <= 0) {
        fprintf(stderr, "zac: no frames to decode (eos=%d, H=%d)\n", eos, H);
        return false;
    }
    zac_stream st;
    audio.clear();
    const auto t0 = std::chrono::steady_clock::now();
    if (!zac_stream_decode(m, st, codes, H, E, audio) || !zac_stream_flush(m, st, audio))
        return false;
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double audio_s = (double)audio.size() / m.sample_rate;
    double rms = 0, peak = 0;
    for (float v : audio) {
        rms += (double)v * v;
        peak = std::max(peak, (double)fabsf(v));
    }
    fprintf(stderr, "zac: H=%d eos=%d -> %d frames, %zu samples = %.2fs, rms=%.4f peak=%.4f, decode %.2fs (RTF %.3f)\n",
            H, eos, E, audio.size(), audio_s, sqrt(rms / audio.size()), peak, secs, secs / audio_s);
    return true;
}
