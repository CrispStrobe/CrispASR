// crispasr_separate_cli.cpp — see crispasr_separate_cli.h.

#include "crispasr_separate_cli.h"

#include "common-crispasr.h" // read_audio_data
#include "crispasr_model_mgr_cli.h"
#include "whisper_params.h"

#include "core/gguf_loader.h"   // core_gguf::open_metadata / kv_str
#include "core/separation_io.h" // crispasr_separation_view + stem output
#include "htdemucs.h"
#include "mel_band_roformer.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

// Interleave the model's channels from read_audio_data output. `stereo` holds
// per-channel buffers; when it has fewer channels than `channels`, the last is
// duplicated (mono -> stereo). Returns n_samples per channel.
int interleave(const std::vector<float>& mono, const std::vector<std::vector<float>>& stereo, int channels,
               std::vector<float>& out) {
    const bool have_stereo = stereo.size() >= 2 && !stereo[0].empty();
    const int n = have_stereo ? (int)stereo[0].size() : (int)mono.size();
    out.assign((size_t)n * channels, 0.0f);
    for (int i = 0; i < n; i++)
        for (int s = 0; s < channels; s++) {
            float v;
            if (have_stereo)
                v = stereo[s < (int)stereo.size() ? s : (int)stereo.size() - 1][i];
            else
                v = mono[i];
            out[(size_t)i * channels + s] = v;
        }
    return n;
}

// Write the selected stems of a result view to disk. Returns files written.
int write_stems(const crispasr_separation_view& v, const std::string& input, const whisper_params& params) {
    int written = 0;
    for (int s = 0; s < v.n_sources; s++) {
        const std::string name = v.source_names[s] ? v.source_names[s] : ("stem" + std::to_string(s));
        if (!crispasr_stem_selected(params.stems, name))
            continue;
        const std::string path = crispasr_stem_output_path(input, name, params.sep_output_dir);
        const std::string blob = crispasr_stem_to_wav(v, s);
        std::ofstream f(path, std::ios::binary);
        if (!f) {
            fprintf(stderr, "crispasr: error: cannot write '%s'\n", path.c_str());
            continue;
        }
        f.write(blob.data(), (std::streamsize)blob.size());
        if (!params.no_prints)
            fprintf(stderr, "crispasr: wrote %s (%d ch x %d @ %d Hz)\n", path.c_str(), v.n_channels, v.n_frames,
                    v.sample_rate);
        written++;
    }
    return written;
}

} // namespace

int crispasr_run_separate(const whisper_params& params) {
    if (params.fname_inp.empty()) {
        fprintf(stderr, "crispasr: --separate needs an input file (-f)\n");
        return 2;
    }
    // Resolve the model. A separation model must be named/pathed or auto-
    // resolved via a separation backend key (--backend htdemucs|mel-band-roformer).
    const std::string backend_key = params.backend.empty() ? "mel-band-roformer" : params.backend;
    const std::string model =
        crispasr_resolve_model_cli(params.model, backend_key, params.no_prints, params.cache_dir, params.auto_download);
    if (model.empty()) {
        fprintf(stderr, "crispasr: --separate: could not resolve a model (pass -m <gguf> or "
                        "--backend htdemucs|mel-band-roformer with --auto-download)\n");
        return 2;
    }

    // Detect architecture from the GGUF.
    gguf_context* meta = core_gguf::open_metadata(model.c_str());
    if (!meta) {
        fprintf(stderr, "crispasr: --separate: cannot open '%s'\n", model.c_str());
        return 2;
    }
    const std::string arch = core_gguf::kv_str(meta, "general.architecture", "");
    core_gguf::free_metadata(meta);

    int rc = 0;
    for (const auto& fname : params.fname_inp) {
        std::vector<float> mono;
        std::vector<std::vector<float>> stereo;

        if (arch == "mel-band-roformer") {
            // #414: forward the CLI's GPU intent — zero-initialized default
            // params carry use_gpu=false, which silently pinned separation to
            // CPU even on GPU builds (the moonshine "encoder gap" trap).
            auto mp = mel_band_roformer_default_params();
            mp.use_gpu = params.use_gpu;
            mp.n_threads = params.n_threads;
            mp.gpu_device = params.gpu_device;
            auto* ctx = mel_band_roformer_init_from_file(model.c_str(), mp);
            if (!ctx) {
                rc = 2;
                break;
            }
            const int sr = mel_band_roformer_sample_rate(ctx);
            if (!read_audio_data(fname, mono, stereo, /*stereo=*/true, /*target_rate=*/sr)) {
                fprintf(stderr, "crispasr: error: cannot read '%s'\n", fname.c_str());
                mel_band_roformer_free(ctx);
                rc = 20;
                continue;
            }
            std::vector<float> pcm;
            const int n = interleave(mono, stereo, 2, pcm);
            auto* r = mel_band_roformer_separate(ctx, pcm.data(), n, 2);
            if (!r) {
                fprintf(stderr, "crispasr: --separate failed on '%s'\n", fname.c_str());
                mel_band_roformer_free(ctx);
                rc = 1;
                continue;
            }
            crispasr_separation_view v;
            v.n_sources = r->n_sources;
            v.n_channels = r->n_channels;
            v.n_frames = r->n_samples;
            v.sample_rate = r->sample_rate;
            v.sources = r->sources;
            v.source_names = r->source_names;
            write_stems(v, fname, params);
            mel_band_roformer_result_free(r);
            mel_band_roformer_free(ctx);

        } else if (arch == "htdemucs") {
            // #414: same forwarding — without it the AUTO gates in
            // htdemucs_gates.h see use_gpu=false and never probe the GPU
            // (caught by the Kaggle proof kernel: all arms ran BLAS).
            auto hp = htdemucs_default_params();
            hp.use_gpu = params.use_gpu;
            hp.n_threads = params.n_threads;
            hp.gpu_device = params.gpu_device;
            auto* ctx = htdemucs_init_from_file(model.c_str(), hp);
            if (!ctx) {
                rc = 2;
                break;
            }
            const int sr = htdemucs_sample_rate(ctx);
            if (!read_audio_data(fname, mono, stereo, /*stereo=*/true, /*target_rate=*/sr)) {
                fprintf(stderr, "crispasr: error: cannot read '%s'\n", fname.c_str());
                htdemucs_free(ctx);
                rc = 20;
                continue;
            }
            std::vector<float> pcm;
            const int n = interleave(mono, stereo, 2, pcm);
            auto* r = htdemucs_separate(ctx, pcm.data(), n);
            if (!r) {
                fprintf(stderr, "crispasr: --separate failed on '%s'\n", fname.c_str());
                htdemucs_free(ctx);
                rc = 1;
                continue;
            }
            crispasr_separation_view v;
            v.n_sources = r->n_sources;
            v.n_channels = r->n_channels;
            v.n_frames = r->n_samples;
            v.sample_rate = r->sample_rate;
            v.sources = r->sources;
            v.source_names = r->source_names;
            write_stems(v, fname, params);
            htdemucs_result_free(r);
            htdemucs_free(ctx);

        } else {
            fprintf(stderr,
                    "crispasr: --separate: '%s' is not a separation model (arch='%s'). "
                    "Expected mel-band-roformer or htdemucs.\n",
                    model.c_str(), arch.c_str());
            return 2;
        }
    }
    return rc;
}
