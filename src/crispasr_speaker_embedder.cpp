// crispasr_speaker_embedder.cpp — concrete embedder adapters and the
// factory that picks one for a given CLI model spec.
//
// Two adapters live here today:
//   * TitaNet-Large       (192-d, NVIDIA/Apache-2.0, 16 kHz input)
//   * IndexTTS-BigVGAN    (512-d, ECAPA-TDNN inside the IndexTTS-1.5
//                          BigVGAN GGUF, 24 kHz input — resampled
//                          internally so callers stay 16 kHz only)
//
// Adding a third is intentionally cheap: subclass
// CrispasrSpeakerEmbedder, implement embed(), and add a dispatch
// branch in crispasr_make_speaker_embedder() that maps model_spec
// strings or resolved paths to it.

#include "crispasr_speaker_embedder.h"

#include "crispasr_cache.h"
#include "crispasr_model_registry.h"
#include "indextts_voc.h"
#include "titanet.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

bool contains_ci(const std::string& haystack, const char* needle) {
    if (!needle || !*needle)
        return false;
    std::string h, n;
    h.reserve(haystack.size());
    for (char c : haystack)
        h.push_back((char)std::tolower((unsigned char)c));
    for (const char* p = needle; *p; ++p)
        n.push_back((char)std::tolower((unsigned char)*p));
    return h.find(n) != std::string::npos;
}

class TitaNetEmbedder : public CrispasrSpeakerEmbedder {
public:
    explicit TitaNetEmbedder(titanet_context* ctx) : ctx_(ctx) {}
    ~TitaNetEmbedder() override {
        if (ctx_)
            titanet_free(ctx_);
    }

    int dim() const override { return 192; }
    const char* name() const override { return "titanet-large"; }

    bool embed(const float* pcm_16k, int n_samples, float* out) override {
        if (!ctx_ || !pcm_16k || n_samples <= 0 || !out)
            return false;
        const int n = titanet_embed(ctx_, pcm_16k, n_samples, out);
        return n == 192;
    }

private:
    titanet_context* ctx_;
};

// IndexTTS-BigVGAN ECAPA-TDNN adapter. The runtime expects 24 kHz mono
// PCM but the diarize pipeline feeds 16 kHz; resampling is done here
// with a simple linear interpolator so callers see the same 16 kHz API
// as TitaNet. Linear is fine for speaker-embedding purposes — the
// network operates on mel features and is insensitive to mild
// resampling artifacts at the highest frequencies.
class IndexTtsEcapaEmbedder : public CrispasrSpeakerEmbedder {
public:
    explicit IndexTtsEcapaEmbedder(indextts_voc_context* ctx) : ctx_(ctx) {}
    ~IndexTtsEcapaEmbedder() override {
        if (ctx_)
            indextts_voc_free(ctx_);
    }

    int dim() const override { return 512; }
    const char* name() const override { return "indextts-bigvgan-ecapa"; }

    bool embed(const float* pcm_16k, int n_samples, float* out) override {
        if (!ctx_ || !pcm_16k || n_samples <= 0 || !out)
            return false;

        // 16 kHz -> 24 kHz: each output sample at index i maps to input
        // index i * 16 / 24 = i * 2/3. Two-tap linear interp.
        const int n_out = (int)((int64_t)n_samples * 3 / 2);
        std::vector<float> pcm_24k((size_t)n_out);
        const double ratio = (double)n_samples / (double)n_out;
        for (int i = 0; i < n_out; i++) {
            const double x = i * ratio;
            int i0 = (int)x;
            int i1 = i0 + 1;
            if (i1 >= n_samples)
                i1 = n_samples - 1;
            if (i0 >= n_samples)
                i0 = n_samples - 1;
            const float a = (float)(x - i0);
            pcm_24k[(size_t)i] = pcm_16k[i0] * (1.0f - a) + pcm_16k[i1] * a;
        }

        float* emb = indextts_voc_speaker_embed(ctx_, pcm_24k.data(), n_out);
        if (!emb)
            return false;
        std::memcpy(out, emb, sizeof(float) * 512);
        std::free(emb);
        return true;
    }

private:
    indextts_voc_context* ctx_;
};

// Resolve the IndexTTS-BigVGAN companion GGUF, auto-downloading on
// first use. `model_spec` is one of "indextts", "indextts-bigvgan",
// "indextts-ecapa", or a path to a `.gguf` containing the BigVGAN
// branch. Returns the path or empty on failure.
std::string resolve_indextts_bigvgan(const std::string& model_spec, const std::string& cache_dir, bool quiet) {
    if (!model_spec.empty() && model_spec.find('/') != std::string::npos && model_spec.size() >= 5 &&
        model_spec.compare(model_spec.size() - 5, 5, ".gguf") == 0) {
        return model_spec;
    }
    return crispasr_cache::ensure_cached_file(
        "indextts-bigvgan.gguf", "https://huggingface.co/cstr/indextts-1.5-GGUF/resolve/main/indextts-bigvgan.gguf",
        quiet, "crispasr[diarize-embedder]", cache_dir);
}

} // namespace

std::unique_ptr<CrispasrSpeakerEmbedder> crispasr_make_speaker_embedder(const std::string& model_spec, int n_threads,
                                                                        const std::string& cache_dir) {
    if (model_spec.empty())
        return nullptr;

    // Dispatch order:
    //   1. Model spec or resolved path mentions "indextts" -> IndexTTS-
    //      BigVGAN ECAPA-TDNN adapter.
    //   2. Default: TitaNet-Large adapter ("auto", "titanet", or any
    //      other GGUF path) — keeps the existing behavior.
    //
    // We dispatch on the spec string first (cheap), then on the
    // resolved filename for "auto"-style aliases. Anything we can't
    // identify falls through to TitaNet, which matches users'
    // expectation when they just point at a random speaker GGUF.

    const bool want_indextts = contains_ci(model_spec, "indextts") || model_spec == "ecapa" ||
                               model_spec == "indextts-bigvgan" || model_spec == "indextts-ecapa";

    if (want_indextts) {
        const std::string resolved = resolve_indextts_bigvgan(model_spec, cache_dir, /*quiet=*/false);
        if (resolved.empty()) {
            fprintf(stderr, "crispasr[diarize]: failed to resolve IndexTTS-BigVGAN model for '%s'\n",
                    model_spec.c_str());
            return nullptr;
        }
        indextts_voc_context* ctx = indextts_voc_init(resolved.c_str(), n_threads, /*use_gpu=*/false);
        if (!ctx) {
            fprintf(stderr, "crispasr[diarize]: failed to load IndexTTS-BigVGAN from '%s'\n", resolved.c_str());
            return nullptr;
        }
        return std::make_unique<IndexTtsEcapaEmbedder>(ctx);
    }

    // Default: TitaNet. Resolve "auto" + bare filenames via the
    // registry, which handles HF auto-download into the cache.
    std::string resolved = model_spec;
    if (resolved == "auto" || resolved.find('/') == std::string::npos) {
        resolved = crispasr_resolve_model(resolved == "auto" ? std::string("auto") : resolved, "titanet",
                                          /*no_prints=*/false, cache_dir, /*auto_download=*/true, "");
    }
    if (resolved.empty()) {
        fprintf(stderr, "crispasr[diarize]: failed to resolve speaker embedder model '%s'\n", model_spec.c_str());
        return nullptr;
    }

    titanet_context* ctx = titanet_init(resolved.c_str(), n_threads);
    if (!ctx) {
        fprintf(stderr, "crispasr[diarize]: failed to load speaker embedder from '%s'\n", resolved.c_str());
        return nullptr;
    }
    return std::make_unique<TitaNetEmbedder>(ctx);
}
