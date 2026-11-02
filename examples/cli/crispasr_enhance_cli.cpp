// crispasr_enhance_cli.cpp — see crispasr_enhance_cli.h.

#include "crispasr_enhance_cli.h"

#include "common-crispasr.h" // read_audio_data
#include "crispasr_enhancer.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "core/separation_io.h" // crispasr_stem_to_wav

#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

bool file_exists(const std::string& p) {
    if (p.empty())
        return false;
    FILE* f = std::fopen(p.c_str(), "rb");
    if (f)
        std::fclose(f);
    return f != nullptr;
}

// rnnoise and existing paths pass through; a registry name (e.g. "frcrn-se")
// resolves to its cached / downloaded GGUF.
std::string resolve_spec(const whisper_params& params) {
    const std::string& spec = params.enhance;
    if (crispasr_enhance_spec_is_none(spec.c_str()))
        return "";
    if (spec == "rnnoise" || file_exists(spec))
        return spec;
    CrispasrRegistryEntry e;
    if (crispasr_registry_lookup(spec, e)) {
        const std::string path = crispasr_resolve_model_cli("auto", spec, params.no_prints, params.cache_dir,
                                                            params.auto_download, "", params.accept_license);
        if (!path.empty())
            return path;
    }
    return spec; // let the library report the precise error
}

struct EnhancerDeleter {
    void operator()(CrispasrEnhancer* e) const { crispasr_enhancer_free(e); }
};

// One enhancer per process (inputs are processed sequentially), keyed by spec.
CrispasrEnhancer* cached_enhancer(const whisper_params& params, bool& failed) {
    static std::unique_ptr<CrispasrEnhancer, EnhancerDeleter> enh;
    static std::string key;
    static bool load_failed = false;
    failed = false;
    if (enh && key == params.enhance)
        return enh.get();
    if (load_failed && key == params.enhance) {
        failed = true;
        return nullptr;
    }
    key = params.enhance;
    enh.reset();
    const std::string spec = resolve_spec(params);
    std::string err;
    enh.reset(crispasr_enhancer_create(spec.c_str(), params.n_threads, params.verbose, &err));
    if (!enh) {
        fprintf(stderr, "crispasr: error: --enhance %s: %s\n", params.enhance.c_str(), err.c_str());
        load_failed = true;
        failed = true;
        return nullptr;
    }
    load_failed = false;
    if (!params.no_prints)
        fprintf(stderr, "crispasr: enhancement: %s (native %d Hz)\n", crispasr_enhancer_name(enh.get()),
                crispasr_enhancer_native_rate(enh.get()));
    return enh.get();
}

bool enhance_one(CrispasrEnhancer* e, std::vector<float>& pcm, int sr) {
    if (pcm.empty())
        return true;
    std::vector<float> out;
    if (!crispasr_enhancer_process(e, pcm.data(), (int)pcm.size(), sr, out) || out.size() != pcm.size()) {
        fprintf(stderr, "crispasr: warning: %s enhancement failed; keeping the original audio\n",
                crispasr_enhancer_name(e));
        return false;
    }
    pcm.swap(out);
    return true;
}

} // namespace

bool crispasr_cli_enhance_requested(const whisper_params& params) {
    return !crispasr_enhance_spec_is_none(params.enhance.c_str());
}

bool crispasr_cli_apply_enhance(const whisper_params& params, std::vector<float>& mono,
                                std::vector<std::vector<float>>& stereo, int sample_rate) {
    if (!crispasr_cli_enhance_requested(params))
        return true;
    // The server calls this from request threads; one enhancer is shared.
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    bool failed = false;
    CrispasrEnhancer* e = cached_enhancer(params, failed);
    if (!e)
        return !failed;
    enhance_one(e, mono, sample_rate);
    for (auto& ch : stereo)
        enhance_one(e, ch, sample_rate);
    return true;
}

int crispasr_run_enhance(const whisper_params& params) {
    if (params.fname_inp.empty()) {
        fprintf(stderr, "crispasr: --enhance-output needs an input file (-f)\n");
        return 2;
    }
    if (!crispasr_cli_enhance_requested(params)) {
        fprintf(stderr, "crispasr: --enhance-output needs --enhance SPEC (or --denoise)\n");
        return 2;
    }
    bool failed = false;
    CrispasrEnhancer* e = cached_enhancer(params, failed);
    if (!e)
        return 2;
    // Decode at the enhancer's native rate so a 48 kHz model keeps full band.
    const int sr = crispasr_enhancer_native_rate(e);
    const bool multi = params.fname_inp.size() > 1;
    int rc = 0;
    for (size_t i = 0; i < params.fname_inp.size(); i++) {
        const std::string& fname = params.fname_inp[i];
        std::vector<float> mono;
        std::vector<std::vector<float>> stereo;
        if (!read_audio_data(fname, mono, stereo, /*stereo=*/false, sr)) {
            fprintf(stderr, "crispasr: error: cannot read '%s'\n", fname.c_str());
            rc = 20;
            continue;
        }
        std::vector<float> out;
        if (!crispasr_enhancer_process(e, mono.data(), (int)mono.size(), sr, out)) {
            fprintf(stderr, "crispasr: error: enhancement failed on '%s'\n", fname.c_str());
            rc = 1;
            continue;
        }
        std::string path = params.enhance_output;
        if (multi) {
            // One output per input: <output-stem>_<i>.wav
            const size_t dot = path.find_last_of('.');
            const std::string base = (dot == std::string::npos) ? path : path.substr(0, dot);
            path = base + "_" + std::to_string(i) + ".wav";
        }
        const float* src = out.data();
        const char* name = "enhanced";
        crispasr_separation_view v;
        v.n_sources = 1;
        v.n_channels = 1;
        v.n_frames = (int)out.size();
        v.sample_rate = sr;
        v.sources = &src;
        v.source_names = &name;
        const std::string blob = crispasr_stem_to_wav(v, 0);
        std::ofstream f(path, std::ios::binary);
        if (!f) {
            fprintf(stderr, "crispasr: error: cannot write '%s'\n", path.c_str());
            rc = 1;
            continue;
        }
        f.write(blob.data(), (std::streamsize)blob.size());
        if (!params.no_prints)
            fprintf(stderr, "crispasr: wrote %s (%s, %d samples @ %d Hz)\n", path.c_str(), crispasr_enhancer_name(e),
                    (int)out.size(), sr);
    }
    return rc;
}
