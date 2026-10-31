// tabcnn.cpp — TabCNN guitar tablature emission scorer. See tabcnn.h.
//
// LAYOUT NOTES, because this is where a CNN port silently goes wrong.
//
// torch is NCHW row-major; ggml is (ne0, ne1, ne2, ne3) with ne0 fastest. For
// this graph they line up without any transpose:
//
//   torch input   [N=T, C=1, H=192, W=9]  -> ggml [9, 192, 1, T]
//   torch kernel  [OC, IC, KH, KW]        -> ggml [KW, KH, IC, OC]
// Both have KW/W fastest in memory, so the converter's plain copy is correct
// and no permute is needed on load.
//
// THE FLATTEN IS THE TRAP, and it is benign here for one specific reason.
// torch does `pool.reshape(T, -1)` on [T, 64, 93, 1], i.e. C-order with W
// fastest, then H, then C. The ggml tensor at that point is [1, 93, 64, T] —
// ne0 = W = 1. Because W collapses to a single element, ggml's natural
// (w + h*1 + c*93) ordering is IDENTICAL to torch's (c*93 + h). So a bare
// reshape to [5952, T] is right. If MaxPool ever left W > 1 this would need an
// explicit permute, and the failure mode would be a silently permuted 5952-wide
// vector into dense0 — no error, just wrong tablature.
//
// The front end is two-pass by construction: amplitude_to_db uses the max of
// the WHOLE clip as its reference, so the CQT must be complete before any
// frame can be normalised. See tabcnn.h.

#include "tabcnn.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include "core/audio_resample.h"
#include "core/cqt.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "core/ggml_cpu_backend.h"

namespace {

constexpr int kConvKernel = 3;
constexpr int kPool = 2;

struct FrontEnd {
    int sample_rate = TABCNN_SAMPLE_RATE;
    int hop_length = 512;
    int n_bins = 192;
    int bins_per_octave = 24;
    int frame_width = 9;
    float fmin_hz = 32.703195662574829f; // C1
    float db_floor = 80.0f;
};

} // namespace

struct tabcnn_context {
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad wl;
    int n_threads = 4;

    FrontEnd fe;
    int num_strings = TABCNN_NUM_STRINGS;
    int num_classes = TABCNN_NUM_CLASSES;
    int silent_class = TABCNN_NUM_CLASSES - 1;
    std::vector<int> open_midi; // per string, -1 when unknown

    ggml_tensor* conv0_w = nullptr;
    ggml_tensor* conv0_b = nullptr;
    ggml_tensor* conv1_w = nullptr;
    ggml_tensor* conv1_b = nullptr;
    ggml_tensor* conv2_w = nullptr;
    ggml_tensor* conv2_b = nullptr;
    ggml_tensor* dense0_w = nullptr;
    ggml_tensor* dense0_b = nullptr;
    ggml_tensor* head_w = nullptr;
    ggml_tensor* head_b = nullptr;

    std::vector<core_cqt::Kernel> cqt_kernels;
    bool debug = false;
};

namespace {

// MIDI note number from a scientific-pitch name like "E2" / "A#3" / "Db4".
// Returns -1 if unparseable — the caller degrades to "tuning unknown" rather
// than inventing standard tuning, which would silently mis-transpose a decoder.
int midi_from_note(const std::string& s) {
    static const int kSemis[7] = {9, 11, 0, 2, 4, 5, 7}; // A B C D E F G
    if (s.empty())
        return -1;
    const char letter = (char)std::toupper((unsigned char)s[0]);
    if (letter < 'A' || letter > 'G')
        return -1;
    int semi = kSemis[letter - 'A'];
    size_t i = 1;
    while (i < s.size() && (s[i] == '#' || s[i] == 'b')) {
        semi += (s[i] == '#') ? 1 : -1;
        i++;
    }
    if (i >= s.size())
        return -1;
    const int octave = std::atoi(s.c_str() + i);
    return (octave + 1) * 12 + semi;
}

// CQT -> the model's input feature, in [0, 1].
//
// This is librosa's amplitude_to_db(S, ref=np.max) followed by amt_tools'
// post_proc rescale (/80 + 1), written out literally so it can be diffed
// line-by-line against tools/reference_backends/tabcnn.py. Getting only the dB
// half of this wrong is not hypothetical: omitting the rescale while building
// the reference dumper produced cos = -0.544 against the real module.
void to_feature(std::vector<float>& mag, float db_floor) {
    float ref = 0.0f;
    for (float v : mag)
        ref = std::max(ref, v);
    ref = std::max(ref, 1e-10f);
    const float inv = 1.0f / db_floor;
    for (float& v : mag) {
        const float db = 20.0f * std::log10(std::max(v, 1e-10f) / ref);
        v = std::max(db, -db_floor) * inv + 1.0f;
    }
}

// [T][n_bins] (bin fastest, as core_cqt emits) -> [T][n_bins][frame_width]
// with the context window fastest, which is ggml's [W, H, C, N]. Edges are
// zero-padded, matching amt_tools framify_activations(pad=True).
void build_windows(const std::vector<float>& feat, int T, int n_bins, int frame_width, std::vector<float>& out) {
    const int pad = frame_width / 2;
    out.assign((size_t)T * n_bins * frame_width, 0.0f);
    for (int t = 0; t < T; t++) {
        for (int b = 0; b < n_bins; b++) {
            float* dst = out.data() + ((size_t)t * n_bins + b) * frame_width;
            for (int w = 0; w < frame_width; w++) {
                const int src = t - pad + w;
                dst[w] = (src >= 0 && src < T) ? feat[(size_t)src * n_bins + b] : 0.0f;
            }
        }
    }
}

ggml_tensor* conv_relu(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    ggml_tensor* y = ggml_conv_2d(g, w, x, 1, 1, 0, 0, 1, 1);
    // bias is per output channel -> broadcast over (W, H, N)
    y = ggml_add(g, y, ggml_reshape_4d(g, b, 1, 1, b->ne[0], 1));
    return ggml_relu(g, y);
}

} // namespace

int tabcnn_n_frames(const struct tabcnn_context* ctx, int n_samples, int sample_rate) {
    if (!ctx || n_samples <= 0 || sample_rate <= 0)
        return 0;
    const long long resampled = (long long)n_samples * ctx->fe.sample_rate / sample_rate;
    core_cqt::Params p;
    p.sample_rate = ctx->fe.sample_rate;
    p.fmin = ctx->fe.fmin_hz;
    p.n_bins = ctx->fe.n_bins;
    p.bins_per_octave = ctx->fe.bins_per_octave;
    p.hop_length = ctx->fe.hop_length;
    return core_cqt::n_frames(p, (int)resampled);
}

float tabcnn_frame_period(const struct tabcnn_context* ctx) {
    if (!ctx || ctx->fe.sample_rate <= 0)
        return 0.0f;
    return (float)ctx->fe.hop_length / (float)ctx->fe.sample_rate;
}

int tabcnn_silent_class(const struct tabcnn_context* ctx) {
    return ctx ? ctx->silent_class : -1;
}

int tabcnn_string_open_midi(const struct tabcnn_context* ctx, int string) {
    if (!ctx || string < 0 || string >= (int)ctx->open_midi.size())
        return -1;
    return ctx->open_midi[(size_t)string];
}

struct tabcnn_context* tabcnn_init(const char* model_path, int n_threads) {
    if (!model_path)
        return nullptr;
    auto* ctx = new tabcnn_context();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;
    ctx->debug = getenv("CRISPASR_TABCNN_DEBUG") != nullptr;

    gguf_context* gctx = core_gguf::open_metadata(model_path);
    if (!gctx) {
        fprintf(stderr, "tabcnn: cannot open %s\n", model_path);
        delete ctx;
        return nullptr;
    }
    // Read the front end from the file. Hardcoding it here is exactly how the
    // runtime drifts from the reference dumper, and a wrong fmin still RUNS.
    ctx->fe.sample_rate = (int)core_gguf::kv_u32(gctx, "tabcnn.sample_rate", ctx->fe.sample_rate);
    ctx->fe.hop_length = (int)core_gguf::kv_u32(gctx, "tabcnn.hop_length", ctx->fe.hop_length);
    ctx->fe.n_bins = (int)core_gguf::kv_u32(gctx, "tabcnn.n_bins", ctx->fe.n_bins);
    ctx->fe.bins_per_octave = (int)core_gguf::kv_u32(gctx, "tabcnn.bins_per_octave", ctx->fe.bins_per_octave);
    ctx->fe.frame_width = (int)core_gguf::kv_u32(gctx, "tabcnn.frame_width", ctx->fe.frame_width);
    ctx->fe.fmin_hz = core_gguf::kv_f32(gctx, "tabcnn.fmin_hz", ctx->fe.fmin_hz);
    ctx->fe.db_floor = core_gguf::kv_f32(gctx, "tabcnn.db_floor", ctx->fe.db_floor);
    ctx->num_strings = (int)core_gguf::kv_u32(gctx, "tabcnn.num_strings", TABCNN_NUM_STRINGS);
    ctx->num_classes = (int)core_gguf::kv_u32(gctx, "tabcnn.num_classes", TABCNN_NUM_CLASSES);
    ctx->silent_class = (int)core_gguf::kv_u32(gctx, "tabcnn.silent_class", ctx->num_classes - 1);
    ctx->open_midi.assign((size_t)ctx->num_strings, -1);
    {
        const std::vector<std::string> tuning = core_gguf::kv_str_array(gctx, "tabcnn.tuning");
        for (int s = 0; s < ctx->num_strings && s < (int)tuning.size(); s++)
            ctx->open_midi[(size_t)s] = midi_from_note(tuning[(size_t)s]);
    }
    core_gguf::free_metadata(gctx);

    if (getenv("CRISPASR_TABCNN_NO_GPU") == nullptr)
        ctx->backend = crispasr_init_gpu_backend();
    if (!ctx->backend)
        ctx->backend = core_cpu_backend::init();
    if (!ctx->backend) {
        delete ctx;
        return nullptr;
    }
    if (core_cpu_backend::is_cpu(ctx->backend))
        core_cpu_backend::set_n_threads(ctx->backend, ctx->n_threads);

    if (!core_gguf::load_weights(model_path, ctx->backend, "tabcnn", ctx->wl)) {
        tabcnn_free(ctx);
        return nullptr;
    }
    auto req = [&](const char* n) { return core_gguf::require(ctx->wl.tensors, n, "tabcnn"); };
    ctx->conv0_w = req("conv0.weight");
    ctx->conv0_b = req("conv0.bias");
    ctx->conv1_w = req("conv1.weight");
    ctx->conv1_b = req("conv1.bias");
    ctx->conv2_w = req("conv2.weight");
    ctx->conv2_b = req("conv2.bias");
    ctx->dense0_w = req("dense0.weight");
    ctx->dense0_b = req("dense0.bias");
    ctx->head_w = req("head.weight");
    ctx->head_b = req("head.bias");
    if (!ctx->conv0_w || !ctx->conv1_w || !ctx->conv2_w || !ctx->dense0_w || !ctx->head_w) {
        fprintf(stderr, "tabcnn: model is missing required tensors\n");
        tabcnn_free(ctx);
        return nullptr;
    }
    // Cross-check the head against the declared string/class counts, so a
    // mismatched GGUF fails at load rather than producing mis-shaped tablature.
    const int64_t head_out = ctx->head_w->ne[1];
    if (head_out != (int64_t)ctx->num_strings * ctx->num_classes) {
        fprintf(stderr, "tabcnn: head emits %lld but metadata says %d strings x %d classes\n", (long long)head_out,
                ctx->num_strings, ctx->num_classes);
        tabcnn_free(ctx);
        return nullptr;
    }

    core_cqt::Params p;
    p.sample_rate = ctx->fe.sample_rate;
    p.fmin = ctx->fe.fmin_hz;
    p.n_bins = ctx->fe.n_bins;
    p.bins_per_octave = ctx->fe.bins_per_octave;
    p.hop_length = ctx->fe.hop_length;
    ctx->cqt_kernels = core_cqt::build_kernels(p);

    if (ctx->debug)
        fprintf(stderr, "tabcnn: sr=%d hop=%d bins=%d bpo=%d fmin=%.3f width=%d strings=%d classes=%d silent=%d\n",
                ctx->fe.sample_rate, ctx->fe.hop_length, ctx->fe.n_bins, ctx->fe.bins_per_octave, ctx->fe.fmin_hz,
                ctx->fe.frame_width, ctx->num_strings, ctx->num_classes, ctx->silent_class);
    return ctx;
}

void tabcnn_free(struct tabcnn_context* ctx) {
    if (!ctx)
        return;
    core_gguf::free_weights(ctx->wl);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

namespace {

// Whole pipeline up to the softmax. `probs` receives [T][strings][classes].
// Optionally captures one named intermediate for crispasr-diff.
int run_model(tabcnn_context* ctx, const float* pcm, int n_samples, int sample_rate, std::vector<float>& probs,
              const char* want_stage, std::vector<float>* stage_out) {
    if (!ctx || !pcm || n_samples <= 0 || sample_rate <= 0)
        return 0;

    std::vector<float> audio;
    if (sample_rate != ctx->fe.sample_rate)
        audio = core_audio::resample_polyphase(pcm, n_samples, sample_rate, ctx->fe.sample_rate);
    else
        audio.assign(pcm, pcm + n_samples);
    if (audio.empty())
        return 0;

    core_cqt::Params p;
    p.sample_rate = ctx->fe.sample_rate;
    p.fmin = ctx->fe.fmin_hz;
    p.n_bins = ctx->fe.n_bins;
    p.bins_per_octave = ctx->fe.bins_per_octave;
    p.hop_length = ctx->fe.hop_length;

    std::vector<float> mag;
    const int T = core_cqt::magnitude(p, ctx->cqt_kernels, audio.data(), (int)audio.size(), mag, ctx->n_threads);
    if (T <= 0)
        return 0;
    to_feature(mag, ctx->fe.db_floor);
    if (want_stage && std::strcmp(want_stage, "cqt_db") == 0 && stage_out) {
        *stage_out = mag;
        return T;
    }

    std::vector<float> win;
    build_windows(mag, T, ctx->fe.n_bins, ctx->fe.frame_width, win);

    const size_t n_nodes = 64;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * n_nodes + ggml_graph_overhead());
    ggml_init_params ip = {meta.size(), meta.data(), true};
    ggml_context* g = ggml_init(ip);
    if (!g)
        return 0;

    ggml_tensor* in = ggml_new_tensor_4d(g, GGML_TYPE_F32, ctx->fe.frame_width, ctx->fe.n_bins, 1, T);
    ggml_set_name(in, "input");
    ggml_set_input(in);

    ggml_tensor* x = conv_relu(g, in, ctx->conv0_w, ctx->conv0_b);
    ggml_tensor* c0 = x;
    x = conv_relu(g, x, ctx->conv1_w, ctx->conv1_b);
    ggml_tensor* c1 = x;
    x = conv_relu(g, x, ctx->conv2_w, ctx->conv2_b);
    ggml_tensor* c2 = x;
    x = ggml_pool_2d(g, x, GGML_OP_POOL_MAX, kPool, kPool, kPool, kPool, 0, 0);
    ggml_tensor* pooled = x;
    // See the flatten note at the top: valid only because ne0 (W) is 1 here.
    const int64_t flat = pooled->ne[0] * pooled->ne[1] * pooled->ne[2];
    x = ggml_reshape_2d(g, ggml_cont(g, x), flat, T);
    x = ggml_add(g, ggml_mul_mat(g, ctx->dense0_w, x), ctx->dense0_b);
    x = ggml_relu(g, x);
    ggml_tensor* d0 = x;
    x = ggml_add(g, ggml_mul_mat(g, ctx->head_w, x), ctx->head_b);
    ggml_tensor* logits = x;
    x = ggml_soft_max(g, ggml_reshape_3d(g, x, ctx->num_classes, ctx->num_strings, T));
    ggml_set_name(x, "probs");
    ggml_set_output(x);

    ggml_tensor* cap = nullptr;
    if (want_stage && stage_out) {
        if (!std::strcmp(want_stage, "conv0_relu"))
            cap = c0;
        else if (!std::strcmp(want_stage, "conv1_relu"))
            cap = c1;
        else if (!std::strcmp(want_stage, "conv2_relu"))
            cap = c2;
        else if (!std::strcmp(want_stage, "pool"))
            cap = pooled;
        else if (!std::strcmp(want_stage, "dense0_relu"))
            cap = d0;
        else if (!std::strcmp(want_stage, "logits"))
            cap = logits;
        if (cap)
            ggml_set_output(cap); // without this ggml reuses the buffer
    }

    ggml_cgraph* gf = ggml_new_graph(g);
    ggml_build_forward_expand(gf, x);
    if (cap)
        ggml_build_forward_expand(gf, cap);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "tabcnn: graph allocation failed\n");
        if (alloc)
            ggml_gallocr_free(alloc);
        ggml_free(g);
        return 0;
    }
    ggml_backend_tensor_set(in, win.data(), 0, win.size() * sizeof(float));
    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "tabcnn: graph compute failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(g);
        return 0;
    }

    if (cap && stage_out) {
        stage_out->resize((size_t)ggml_nelements(cap));
        ggml_backend_tensor_get(cap, stage_out->data(), 0, stage_out->size() * sizeof(float));
    }
    probs.resize((size_t)T * ctx->num_strings * ctx->num_classes);
    ggml_backend_tensor_get(x, probs.data(), 0, probs.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(g);
    return T;
}

} // namespace

int tabcnn_compute(struct tabcnn_context* ctx, const float* pcm, int n_samples, int sample_rate, float* out,
                   int max_frames) {
    if (!out || max_frames <= 0)
        return 0;
    std::vector<float> probs;
    const int T = run_model(ctx, pcm, n_samples, sample_rate, probs, nullptr, nullptr);
    if (T <= 0)
        return 0;
    const int n = std::min(T, max_frames);
    const size_t per = (size_t)ctx->num_strings * ctx->num_classes;
    // Log here rather than in the graph: the contract is log-probabilities (a
    // DP sums costs), and the floor keeps a zeroed class from producing -inf,
    // which would poison every path through it.
    for (size_t i = 0; i < (size_t)n * per; i++)
        out[i] = std::log(std::max(probs[i], 1e-20f));
    return n;
}

int tabcnn_compute_argmax(struct tabcnn_context* ctx, const float* pcm, int n_samples, int sample_rate,
                          int8_t* out_frets, int max_frames) {
    if (!out_frets || max_frames <= 0)
        return 0;
    std::vector<float> probs;
    const int T = run_model(ctx, pcm, n_samples, sample_rate, probs, nullptr, nullptr);
    if (T <= 0)
        return 0;
    const int n = std::min(T, max_frames);
    for (int t = 0; t < n; t++) {
        for (int s = 0; s < ctx->num_strings; s++) {
            const float* row = probs.data() + ((size_t)t * ctx->num_strings + s) * ctx->num_classes;
            int best = 0;
            for (int c = 1; c < ctx->num_classes; c++)
                if (row[c] > row[best])
                    best = c;
            out_frets[(size_t)t * ctx->num_strings + s] = (int8_t)best;
        }
    }
    return n;
}

int tabcnn_extract_stage(struct tabcnn_context* ctx, const float* pcm, int n_samples, int sample_rate,
                         const char* stage, float* out, int max_elems) {
    if (!stage || !out || max_elems <= 0)
        return 0;
    std::vector<float> probs, captured;
    const int T = run_model(ctx, pcm, n_samples, sample_rate, probs, stage, &captured);
    if (T <= 0)
        return 0;
    const std::vector<float>& src = captured.empty() ? probs : captured;
    const int n = (int)std::min((size_t)max_elems, src.size());
    std::memcpy(out, src.data(), (size_t)n * sizeof(float));
    return n;
}

// ---------------------------------------------------------------------------
// crispasr-diff entry point
// ---------------------------------------------------------------------------
//
// Runs from the WAVEFORM stored in the reference archive, not from replayed
// features. That is deliberate: model.frontend is empty, so the CQT lives
// outside the network, and a diff that starts at cqt_db would never test it.
// BTC learned this the expensive way -- its harness replayed input_feat by
// design and a front-end mismatch survived all the way to a real-music run
// (mir_eval 86.63% -> 98.56% once the CQT was fixed).
//
// Prints |mine| and |ref| alongside the cosine on every stage. Cosine is
// scale-blind, and this backend has already produced one scale bug that cosine
// alone would have passed (the missing /80 + 1 rescale, cos -0.544 with the
// magnitudes 15895 vs 88 giving it away).

namespace {

bool tabcnn_ref_get(core_gguf::WeightLoad& rw, const char* name, std::vector<float>& out) {
    auto it = rw.tensors.find(name);
    if (it == rw.tensors.end() || !it->second)
        return false;
    const int64_t n = ggml_nelements(it->second);
    out.resize((size_t)n);
    ggml_backend_tensor_get(it->second, out.data(), 0, (size_t)n * sizeof(float));
    return true;
}

// cos plus the two magnitudes. Returns cos.
double tabcnn_cmp(const char* stage, const std::vector<float>& mine, const std::vector<float>& ref, int verbosity,
                  double tol) {
    const size_t n = std::min(mine.size(), ref.size());
    if (n == 0) {
        if (verbosity)
            fprintf(stderr, "  %-14s SKIP (empty)\n", stage);
        return -1.0;
    }
    double dot = 0.0, na = 0.0, nb = 0.0, maxd = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double a = mine[i], b = ref[i];
        dot += a * b;
        na += a * a;
        nb += b * b;
        maxd = std::max(maxd, std::fabs(a - b));
    }
    const double cos = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0;
    if (verbosity)
        fprintf(stderr, "  %-14s n=%-9zu cos=%.9f  |mine|=%11.3f |ref|=%11.3f  max|d|=%.3e%s\n", stage, n, cos,
                std::sqrt(na), std::sqrt(nb), maxd, cos >= tol ? "  PASS" : "  FAIL");
    return cos;
}

} // namespace

int tabcnn_diff(const char* model_gguf, const char* ref_gguf, int verbosity) {
    tabcnn_context* ctx = tabcnn_init(model_gguf, 4);
    if (!ctx) {
        fprintf(stderr, "tabcnn_diff: failed to load %s\n", model_gguf);
        return 1;
    }
    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, ctx->backend, "tabcnn_ref", rw)) {
        fprintf(stderr, "tabcnn_diff: failed to load reference %s\n", ref_gguf);
        tabcnn_free(ctx);
        return 1;
    }

    std::vector<float> audio;
    if (!tabcnn_ref_get(rw, "audio", audio) || audio.empty()) {
        fprintf(stderr, "tabcnn_diff: reference has no `audio` stage — re-dump with\n"
                        "  tools/reference_backends/tabcnn.py, which emits it precisely so the\n"
                        "  front end is covered rather than replayed.\n");
        core_gguf::free_weights(rw);
        tabcnn_free(ctx);
        return 1;
    }

    // The dumper writes audio at the ORIGINAL rate it was handed; the runtime
    // resamples internally, so feed the model's own rate to keep one resampler
    // out of the comparison.
    const int sr = TABCNN_SAMPLE_RATE;
    if (verbosity)
        fprintf(stderr, "tabcnn_diff: %zu samples @ %d Hz\n", audio.size(), sr);

    int fails = 0;
    static const char* kStages[] = {"cqt_db", "conv0_relu",  "conv1_relu", "conv2_relu",
                                    "pool",   "dense0_relu", "logits"};
    for (const char* stage : kStages) {
        std::vector<float> ref;
        if (!tabcnn_ref_get(rw, stage, ref))
            continue; // stage not in this dump
        std::vector<float> mine(ref.size() + 16);
        const int got =
            tabcnn_extract_stage(ctx, audio.data(), (int)audio.size(), sr, stage, mine.data(), (int)mine.size());
        if (got <= 0) {
            fprintf(stderr, "  %-14s FAIL (runtime produced nothing)\n", stage);
            fails++;
            continue;
        }
        mine.resize((size_t)got);
        // Thresholds encode a KNOWN difference, not a guess. core/cqt.h is an
        // independent CQT (direct Brown kernels) versus librosa's recursive
        // downsampling, so the front end agrees to ~0.9988 and every graph
        // stage inherits that. Measured end to end this costs nothing that
        // matters: tablature F1 0.7732 vs the torch reference's 0.7708
        // (delta +0.0024) with 98.57% argmax agreement on EGSet12 track 01.
        // The acceptance gate for this backend is that task metric, NOT these
        // cosines -- they exist to localise a graph bug, and a graph bug shows
        // up as one stage collapsing, not as a uniform 0.98.
        const double tol = std::strcmp(stage, "cqt_db") == 0 ? 0.998 : 0.96;
        if (tabcnn_cmp(stage, mine, ref, verbosity, tol) < tol)
            fails++;
    }

    core_gguf::free_weights(rw);
    tabcnn_free(ctx);
    if (verbosity)
        fprintf(stderr, "tabcnn_diff: %s\n", fails == 0 ? "ALL STAGES PASS" : "FAILURES PRESENT");
    return fails == 0 ? 0 : 1;
}
