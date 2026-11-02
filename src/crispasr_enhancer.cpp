// crispasr_enhancer.cpp — see crispasr_enhancer.h.

#include "crispasr_enhancer.h"

#include "core/audio_resample.h"

#if defined(CRISPASR_RNNOISE)
#include "crispasr_enhance.h"
#endif

#include "gguf.h"
#include "mossformer2_se.h"
#include "mossformergan_se.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Model-port registry. Each port exposes `const CrispasrEnhanceModelOps*
// <name>_enhance_ops()` and is listed here.
// ---------------------------------------------------------------------------
const CrispasrEnhanceModelOps* frcrn_se_enhance_ops();

static const CrispasrEnhanceModelOps* const* enhance_model_table(int* n) {
    static const CrispasrEnhanceModelOps* const table[] = {
        mossformergan_se_enhance_ops(),
        mossformer2_se_enhance_ops(),
        frcrn_se_enhance_ops(),
    };
    *n = (int)(sizeof(table) / sizeof(table[0]));
    return table;
}

struct CrispasrEnhancer {
    std::string name;
    int native_rate = 16000;
    bool is_rnnoise = false;
    const CrispasrEnhanceModelOps* ops = nullptr;
    void* ctx = nullptr;
    bool verbose = false;
};

bool crispasr_enhance_spec_is_none(const char* spec) {
    if (!spec)
        return true;
    std::string s(spec);
    while (!s.empty() && std::isspace((unsigned char)s.back()))
        s.pop_back();
    size_t b = 0;
    while (b < s.size() && std::isspace((unsigned char)s[b]))
        b++;
    s = s.substr(b);
    for (auto& c : s)
        c = (char)std::tolower((unsigned char)c);
    return s.empty() || s == "none" || s == "off" || s == "0" || s == "false" || s == "no";
}

bool crispasr_enhancer_has_rnnoise() {
#if defined(CRISPASR_RNNOISE)
    return true;
#else
    return false;
#endif
}

const char* crispasr_enhancer_model_archs() {
    static std::string archs;
    if (archs.empty()) {
        int n = 0;
        auto t = enhance_model_table(&n);
        for (int i = 0; i < n; i++) {
            if (!archs.empty())
                archs += ",";
            archs += t[i]->arch;
        }
    }
    return archs.c_str();
}

static std::string read_gguf_arch(const char* path) {
    gguf_init_params p = {/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context* g = gguf_init_from_file(path, p);
    if (!g)
        return "";
    std::string arch;
    const int64_t k = gguf_find_key(g, "general.architecture");
    if (k >= 0 && gguf_get_kv_type(g, k) == GGUF_TYPE_STRING)
        arch = gguf_get_val_str(g, k);
    gguf_free(g);
    return arch;
}

CrispasrEnhancer* crispasr_enhancer_create(const char* spec, int n_threads, bool verbose, std::string* err) {
    if (err)
        err->clear();
    if (crispasr_enhance_spec_is_none(spec))
        return nullptr;
    std::string s(spec);
    std::string lower = s;
    for (auto& c : lower)
        c = (char)std::tolower((unsigned char)c);

    if (lower == "rnnoise") {
        if (!crispasr_enhancer_has_rnnoise()) {
            if (err)
                *err = "RNNoise is not compiled into this build (MSVC builds exclude it)";
            return nullptr;
        }
        auto* e = new CrispasrEnhancer();
        e->name = "rnnoise";
        e->native_rate = 48000; // RNNoise's own rate: inputs are resampled once, straight to 48 kHz
        e->is_rnnoise = true;
        e->verbose = verbose;
        return e;
    }

    FILE* f = std::fopen(s.c_str(), "rb");
    if (!f) {
        if (err)
            *err = "unknown enhance method '" + s + "' (expected none, rnnoise, or a path to an enhancement GGUF)";
        return nullptr;
    }
    std::fclose(f);
    const std::string arch = read_gguf_arch(s.c_str());
    if (arch.empty()) {
        if (err)
            *err = "'" + s + "' is not a readable GGUF file";
        return nullptr;
    }
    int n = 0;
    auto table = enhance_model_table(&n);
    for (int i = 0; i < n; i++) {
        const CrispasrEnhanceModelOps* ops = table[i];
        if (arch != ops->arch)
            continue;
        void* ctx = ops->init(s.c_str(), n_threads > 0 ? n_threads : 4, verbose);
        if (!ctx) {
            if (err)
                *err = std::string("failed to load ") + ops->name + " model '" + s + "'";
            return nullptr;
        }
        auto* e = new CrispasrEnhancer();
        e->name = ops->name;
        e->ops = ops;
        e->ctx = ctx;
        e->native_rate = ops->native_rate(ctx);
        e->verbose = verbose;
        return e;
    }
    if (err) {
        *err = "'" + s + "' has architecture '" + arch + "', which is not an enhancement model";
        const char* known = crispasr_enhancer_model_archs();
        if (known && *known)
            *err += std::string(" (accepted: ") + known + ")";
    }
    return nullptr;
}

bool crispasr_enhancer_process(CrispasrEnhancer* e, const float* in, int n_samples, int sample_rate,
                               std::vector<float>& out) {
    if (!e || !in || n_samples <= 0 || sample_rate <= 0)
        return false;
    const int nr = e->native_rate;
    std::vector<float> work;
    const float* src = in;
    int n_src = n_samples;
    if (sample_rate != nr) {
        work = core_audio::resample_polyphase(in, n_samples, sample_rate, nr);
        src = work.data();
        n_src = (int)work.size();
    }
    if (n_src <= 0)
        return false;

    std::vector<float> enhanced;
    if (e->is_rnnoise) {
#if defined(CRISPASR_RNNOISE)
        enhanced.resize(n_src);
        if (!crispasr_enhance_rnnoise_48k(src, n_src, enhanced.data(), e->verbose))
            return false;
#else
        return false;
#endif
    } else {
        if (!e->ops->process(e->ctx, src, n_src, enhanced))
            return false;
        if ((int)enhanced.size() != n_src)
            enhanced.resize(n_src, 0.0f);
    }

    if (sample_rate != nr) {
        out = core_audio::resample_polyphase(enhanced.data(), (int)enhanced.size(), nr, sample_rate);
    } else {
        out.swap(enhanced);
    }
    out.resize(n_samples, 0.0f);
    return true;
}

const char* crispasr_enhancer_name(const CrispasrEnhancer* e) {
    return e ? e->name.c_str() : "none";
}

int crispasr_enhancer_native_rate(const CrispasrEnhancer* e) {
    return e ? e->native_rate : 0;
}

void crispasr_enhancer_free(CrispasrEnhancer* e) {
    if (!e)
        return;
    if (e->ops && e->ctx)
        e->ops->free(e->ctx);
    delete e;
}
