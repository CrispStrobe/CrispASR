// crispasr_diarize_cli.cpp — CLI-side diarization shim.
//
// Routes the four in-process diarization methods to the shared library
// (`src/crispasr_diarize.cpp`) and keeps the sherpa-ONNX subprocess
// method here, since it shells out to an externally installed binary
// and is CLI-shaped UX. Also handles auto-download of the pyannote
// GGUF via `crispasr_cache`.

#include "crispasr_diarize_cli.h"
#include "crispasr_cache.h"
#include "whisper_params.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#define close _close
#define popen _popen
#define pclose _pclose
static int mkstemps(char* t, int s) {
    (void)s;
    return _mktemp_s(t, strlen(t) + 1) == 0 ? _open(t, _O_CREAT | _O_WRONLY, 0600) : -1;
}
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

// Map the library's integer speaker index to the `"(speaker N) "` string
// shape CLI consumers have relied on since the original crispasr
// `--diarize` flag. -1 (method had no info) leaves the field empty.
void apply_int_speakers_to_crispasr_segments(const std::vector<CrispasrDiarizeSegment>& in,
                                             std::vector<crispasr_segment>& out) {
    const size_t n = std::min(in.size(), out.size());
    for (size_t i = 0; i < n; i++) {
        if (in[i].speaker >= 0)
            out[i].speaker = "(speaker " + std::to_string(in[i].speaker) + ") ";
    }
}

// Build a lib-style view over the CLI segments (just t0/t1 copied).
std::vector<CrispasrDiarizeSegment> lib_view(const std::vector<crispasr_segment>& cli) {
    std::vector<CrispasrDiarizeSegment> v;
    v.reserve(cli.size());
    for (const auto& s : cli)
        v.push_back({s.t0, s.t1, -1});
    return v;
}

// Helper: write a temporary 16 kHz mono f32→int16 WAV that sherpa can read.
std::string write_temp_mono_wav(const float* samples, int n_samples) {
    char buf[] = "/tmp/crispasr-sherpa-XXXXXX.wav";
    int fd = mkstemps(buf, 4);
    if (fd < 0)
        return {};
    close(fd);
    std::string path = buf;
    FILE* f = fopen(path.c_str(), "wb");
    if (!f)
        return {};

    const uint32_t sr = 16000;
    const uint16_t ch = 1;
    const uint16_t bps = 16;
    const uint32_t byte_rate = sr * ch * bps / 8;
    const uint16_t block_align = ch * bps / 8;
    const uint32_t data_bytes = (uint32_t)n_samples * block_align;
    const uint32_t riff_size = 36 + data_bytes;

    auto w32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto w16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
    fwrite("RIFF", 1, 4, f);
    w32(riff_size);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    w32(16);
    w16(1);
    w16(ch);
    w32(sr);
    w32(byte_rate);
    w16(block_align);
    w16(bps);
    fwrite("data", 1, 4, f);
    w32(data_bytes);
    std::vector<int16_t> pcm(n_samples);
    for (int i = 0; i < n_samples; i++) {
        float v = samples[i];
        if (v > 1.0f)
            v = 1.0f;
        if (v < -1.0f)
            v = -1.0f;
        pcm[i] = (int16_t)(v * 32767.0f);
    }
    fwrite(pcm.data(), sizeof(int16_t), pcm.size(), f);
    fclose(f);
    return path;
}

struct SherpaSegment {
    double t0_s;
    double t1_s;
    int speaker;
};

// Parse a line emitted by sherpa-onnx-offline-speaker-diarization.
//   "0.320 -- 3.680 speaker_00 duration=3.360"   — newer format
//   "0.320 3.680 0"                               — older format
bool parse_sherpa_line(const std::string& line, SherpaSegment& out) {
    double t0 = 0, t1 = 0;
    char rest[256] = {0};
    if (std::sscanf(line.c_str(), "%lf -- %lf %255s", &t0, &t1, rest) == 3) {
        out.t0_s = t0;
        out.t1_s = t1;
        const char* p = rest;
        while (*p && !isdigit((unsigned char)*p))
            p++;
        out.speaker = *p ? std::atoi(p) : 0;
        return true;
    }
    int spk = 0;
    if (std::sscanf(line.c_str(), "%lf %lf %d", &t0, &t1, &spk) == 3) {
        out.t0_s = t0;
        out.t1_s = t1;
        out.speaker = spk;
        return true;
    }
    return false;
}

// For each ASR segment, pick the sherpa speaker whose time interval
// overlaps the segment the most.
void assign_speakers_from_sherpa(std::vector<crispasr_segment>& segs, const std::vector<SherpaSegment>& sherpa) {
    if (sherpa.empty())
        return;
    for (auto& seg : segs) {
        const double a0 = (double)seg.t0 / 100.0;
        const double a1 = (double)seg.t1 / 100.0;
        std::vector<double> overlap_per_speaker(32, 0.0);
        int max_spk = 0;
        for (const auto& s : sherpa) {
            const double lo = std::max(a0, s.t0_s);
            const double hi = std::min(a1, s.t1_s);
            if (hi > lo) {
                if (s.speaker >= (int)overlap_per_speaker.size())
                    overlap_per_speaker.resize(s.speaker + 1, 0.0);
                overlap_per_speaker[s.speaker] += (hi - lo);
                if (s.speaker > max_spk)
                    max_spk = s.speaker;
            }
        }
        int best = -1;
        double best_overlap = 0.0;
        for (int i = 0; i <= max_spk; i++) {
            if (overlap_per_speaker[i] > best_overlap) {
                best_overlap = overlap_per_speaker[i];
                best = i;
            }
        }
        if (best >= 0) {
            seg.speaker = "(speaker " + std::to_string(best) + ") ";
        }
    }
}

bool apply_sherpa(const std::vector<float>& mono, int64_t slice_t0_cs, std::vector<crispasr_segment>& segs,
                  const whisper_params& params) {
    const std::string bin =
        params.sherpa_bin.empty() ? std::string("sherpa-onnx-offline-speaker-diarization") : params.sherpa_bin;
    if (params.sherpa_segment_model.empty() || params.sherpa_embedding_model.empty()) {
        fprintf(stderr, "crispasr[diarize]: sherpa needs --sherpa-segment-model and\n"
                        "                   --sherpa-embedding-model. Download them from\n"
                        "                   https://github.com/k2-fsa/sherpa-onnx — e.g.\n"
                        "                     sherpa-pyannote-segmentation-3.0.onnx\n"
                        "                     3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx\n");
        return false;
    }

    if (bin.find('/') != std::string::npos) {
        struct stat st;
        if (::stat(bin.c_str(), &st) != 0) {
            fprintf(stderr,
                    "crispasr[diarize]: sherpa binary '%s' not found — pass "
                    "--sherpa-bin or install k2-fsa/sherpa-onnx\n",
                    bin.c_str());
            return false;
        }
    }

    const std::string wav_path = write_temp_mono_wav(mono.data(), (int)mono.size());
    if (wav_path.empty()) {
        fprintf(stderr, "crispasr[diarize]: failed to write temp wav\n");
        return false;
    }

    std::ostringstream cmd;
    // clang-format off
    cmd << bin
        << " --clustering.num-clusters=" << params.sherpa_num_clusters
        << " --segmentation.pyannote.model='" << params.sherpa_segment_model << "'"
        << " --embedding.model='" << params.sherpa_embedding_model << "'"
        << " '" << wav_path << "'";
    // clang-format on
    if (!params.no_prints)
        fprintf(stderr, "crispasr[diarize]: %s\n", cmd.str().c_str());
    cmd << " 2>/dev/null";

    std::unique_ptr<FILE, int (*)(FILE*)> pipe(popen(cmd.str().c_str(), "r"), pclose);
    if (!pipe) {
        fprintf(stderr, "crispasr[diarize]: failed to spawn sherpa subprocess\n");
        std::remove(wav_path.c_str());
        return false;
    }

    std::vector<SherpaSegment> parsed;
    char linebuf[1024];
    while (fgets(linebuf, sizeof(linebuf), pipe.get())) {
        SherpaSegment s;
        if (parse_sherpa_line(linebuf, s))
            parsed.push_back(s);
    }
    std::remove(wav_path.c_str());

    if (parsed.empty()) {
        fprintf(stderr, "crispasr[diarize]: sherpa subprocess produced no parseable "
                        "segments — check that the two --sherpa-*-model paths are "
                        "correct and that the binary prints results on stdout.\n");
        return false;
    }

    // sherpa reports times relative to the audio it was handed (i.e. the
    // slice), so shift by slice_t0_cs before merging with our absolute-cs
    // segments.
    for (auto& s : parsed) {
        s.t0_s += (double)slice_t0_cs / 100.0;
        s.t1_s += (double)slice_t0_cs / 100.0;
    }
    assign_speakers_from_sherpa(segs, parsed);

    if (!params.no_prints) {
        fprintf(stderr, "crispasr[diarize]: sherpa → %zu speaker regions over %zu ASR segments\n", parsed.size(),
                segs.size());
    }
    return true;
}

// Resolve the pyannote GGUF path from the CLI flags, auto-downloading
// the canonical one from HF on first use if the user passed "auto".
std::string resolve_pyannote_model(const whisper_params& params) {
    std::string mp = params.sherpa_segment_model;
    if (mp.empty() || mp == "auto") {
        mp = crispasr_cache::ensure_cached_file(
            "pyannote-seg-3.0.gguf",
            "https://huggingface.co/cstr/pyannote-v3-segmentation-GGUF/resolve/main/pyannote-seg-3.0.gguf",
            params.no_prints, "crispasr[diarize]", params.cache_dir);
    }
    if (mp.size() < 5 || mp.compare(mp.size() - 5, 5, ".gguf") != 0)
        return {}; // not GGUF → caller can fall back to sherpa subprocess
    return mp;
}

} // namespace

bool crispasr_apply_diarize(const std::vector<float>& left, const std::vector<float>& right, bool is_stereo,
                            int64_t slice_t0_cs, std::vector<crispasr_segment>& segs, const whisper_params& params) {
    if (segs.empty())
        return true;

    std::string method = params.diarize_method;
    if (method.empty()) {
        // Historical defaults: stereo → "energy", mono → "vad-turns".
        method = is_stereo ? "energy" : "vad-turns";
    }

    // Shared in-process methods go through the library.
    CrispasrDiarizeMethod lib_method;
    bool use_lib = true;
    if (method == "energy") {
        lib_method = CrispasrDiarizeMethod::Energy;
    } else if (method == "xcorr" || method == "cross-correlation") {
        lib_method = CrispasrDiarizeMethod::Xcorr;
    } else if (method == "vad-turns" || method == "turns") {
        lib_method = CrispasrDiarizeMethod::VadTurns;
    } else if (method == "pyannote") {
        lib_method = CrispasrDiarizeMethod::Pyannote;
    } else {
        use_lib = false;
    }

    if (use_lib) {
        CrispasrDiarizeOptions opts;
        opts.method = lib_method;
        opts.n_threads = params.n_threads;
        opts.slice_t0_cs = slice_t0_cs;
        if (lib_method == CrispasrDiarizeMethod::Pyannote)
            opts.pyannote_model_path = resolve_pyannote_model(params);

        auto lib_segs = lib_view(segs);
        const int n = (int)left.size();
        const float* l = left.data();
        const float* r = (is_stereo && !right.empty()) ? right.data() : l;
        if (!crispasr_diarize_segments(l, r, n, is_stereo, lib_segs, opts)) {
            // pyannote model load failed — try sherpa subprocess fallback
            // when we can (mono input is what sherpa is best at).
            if (lib_method == CrispasrDiarizeMethod::Pyannote) {
                std::vector<float> mono = is_stereo ? std::vector<float>(left) : left;
                if (is_stereo) {
                    for (size_t j = 0; j < mono.size() && j < right.size(); j++)
                        mono[j] = 0.5f * (left[j] + right[j]);
                }
                return apply_sherpa(mono, slice_t0_cs, segs, params);
            }
            return false;
        }
        apply_int_speakers_to_crispasr_segments(lib_segs, segs);
        return true;
    }

    // CLI-only method: sherpa-onnx subprocess.
    if (method == "sherpa" || method == "sherpa-onnx" || method == "ecapa") {
        std::vector<float> mono = left;
        if (is_stereo && !right.empty()) {
            const size_t n = std::min(left.size(), right.size());
            mono.resize(n);
            for (size_t j = 0; j < n; j++)
                mono[j] = 0.5f * (left[j] + right[j]);
        }
        return apply_sherpa(mono, slice_t0_cs, segs, params);
    }

    fprintf(stderr,
            "crispasr[diarize]: unknown --diarize-method '%s'. Known: energy, xcorr, "
            "vad-turns, pyannote, sherpa. Defaulting to '%s'.\n",
            method.c_str(), is_stereo ? "energy" : "vad-turns");
    return false;
}
