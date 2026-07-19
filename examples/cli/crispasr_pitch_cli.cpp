// crispasr_pitch_cli.cpp — see crispasr_pitch_cli.h.

#include "crispasr_pitch_cli.h"

#include "common-crispasr.h" // read_audio_data
#include "crispasr_model_mgr_cli.h"
#include "whisper_params.h"

#include "core/gguf_loader.h" // core_gguf::open_metadata / kv_str
#include "crepe.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

// One line per frame, greppable: "time_ms f0_hz voiced_prob".
void print_frames_text(const std::vector<crepe_frame>& frames) {
    for (const auto& f : frames)
        printf("%.1f\t%.3f\t%.4f\n", f.time_ms, f.f0_hz, f.voiced_prob);
}

void print_frames_json(const std::vector<crepe_frame>& frames, const std::string& fname, const char* capacity) {
    printf("{\n");
    printf("  \"file\": \"%s\",\n", fname.c_str());
    printf("  \"model\": \"crepe-%s\",\n", capacity ? capacity : "");
    printf("  \"n_frames\": %d,\n", (int)frames.size());
    printf("  \"frames\": [\n");
    for (size_t i = 0; i < frames.size(); i++) {
        const auto& f = frames[i];
        printf("    {\"time_ms\": %.1f, \"f0_hz\": %.3f, \"voiced_prob\": %.4f}%s\n", f.time_ms, f.f0_hz, f.voiced_prob,
               i + 1 < frames.size() ? "," : "");
    }
    printf("  ]\n}\n");
}

} // namespace

int crispasr_run_pitch(const whisper_params& params) {
    if (params.fname_inp.empty()) {
        fprintf(stderr, "crispasr: --pitch needs an input file (-f)\n");
        return 2;
    }

    const bool json = params.pitch_format == "json";
    if (!json && !params.pitch_format.empty() && params.pitch_format != "text") {
        fprintf(stderr, "crispasr: --pitch-format: unknown format '%s' (expected text or json)\n",
                params.pitch_format.c_str());
        return 2;
    }

    // Resolve the model. Defaults to the `crepe` registry entry (tiny), which is
    // the shipping default: full is RTF 2.0 on Metal where tiny is RTF 0.28.
    const std::string backend_key = params.backend.empty() ? "crepe" : params.backend;
    const std::string model =
        crispasr_resolve_model_cli(params.model, backend_key, params.no_prints, params.cache_dir, params.auto_download);
    if (model.empty()) {
        fprintf(stderr, "crispasr: --pitch: could not resolve a model (pass -m <gguf> or "
                        "--backend crepe with --auto-download)\n");
        return 2;
    }

    // Detect architecture from the GGUF, exactly as --separate does.
    gguf_context* meta = core_gguf::open_metadata(model.c_str());
    if (!meta) {
        fprintf(stderr, "crispasr: --pitch: cannot open '%s'\n", model.c_str());
        return 2;
    }
    const std::string arch = core_gguf::kv_str(meta, "general.architecture", "");
    core_gguf::free_metadata(meta);

    if (arch != "crepe") {
        fprintf(stderr,
                "crispasr: --pitch: '%s' is not a pitch model (arch='%s'). "
                "Expected crepe.\n",
                model.c_str(), arch.c_str());
        return 2;
    }

    crepe_context* ctx = crepe_init(model.c_str(), params.n_threads);
    if (!ctx) {
        fprintf(stderr, "crispasr: --pitch: failed to load '%s'\n", model.c_str());
        return 2;
    }
    const float hop_ms = params.pitch_hop_ms > 0.0f ? params.pitch_hop_ms : 10.0f;

    int rc = 0;
    for (const auto& fname : params.fname_inp) {
        std::vector<float> mono;
        std::vector<std::vector<float>> stereo;
        if (!read_audio_data(fname, mono, stereo, /*stereo=*/false, /*target_rate=*/CREPE_SAMPLE_RATE)) {
            fprintf(stderr, "crispasr: error: cannot read '%s'\n", fname.c_str());
            rc = 20;
            continue;
        }

        const int n_max = crepe_n_frames(ctx, (int)mono.size(), hop_ms);
        if (n_max <= 0) {
            fprintf(stderr, "crispasr: --pitch: no frames for '%s'\n", fname.c_str());
            rc = 1;
            continue;
        }
        std::vector<crepe_frame> frames((size_t)n_max);
        const int n = crepe_compute_f0(ctx, mono.data(), (int)mono.size(), hop_ms, frames.data(), n_max);
        if (n <= 0) {
            fprintf(stderr, "crispasr: --pitch failed on '%s'\n", fname.c_str());
            rc = 1;
            continue;
        }
        frames.resize((size_t)n);

        if (json) {
            print_frames_json(frames, fname, crepe_capacity(ctx));
        } else {
            if (!params.no_prints && params.fname_inp.size() > 1)
                printf("# %s\n", fname.c_str());
            print_frames_text(frames);
        }
        if (!params.no_prints)
            fprintf(stderr, "crispasr: %s: %d pitch frames (crepe-%s, hop %.1f ms)\n", fname.c_str(), n,
                    crepe_capacity(ctx) ? crepe_capacity(ctx) : "?", hop_ms);
    }

    crepe_free(ctx);
    return rc;
}
