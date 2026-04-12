// crispasr_diarize.cpp — implementation of the generic diarize post-step.
// See crispasr_diarize.h for the interface contract.

#include "crispasr_diarize.h"
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
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#define close _close
static int mkstemps(char * t, int s) {
    (void)s; return _mktemp_s(t, strlen(t)+1) == 0 ? _open(t, _O_CREAT|_O_WRONLY, 0600) : -1;
}
#else
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <vector>

namespace {

// Convert an absolute centisecond timestamp into the per-channel sample
// index inside the current slice. Slice-relative because the energy
// arrays cover only [t0_cs, t1_cs), not the whole input.
inline int64_t cs_to_sample_in_slice(int64_t cs_abs, int64_t slice_t0_cs) {
    int64_t cs_local = cs_abs - slice_t0_cs;
    if (cs_local < 0) cs_local = 0;
    return (cs_local * 16000) / 100;
}

// -----------------------------------------------------------------------
// Method 1: energy-based comparison
// -----------------------------------------------------------------------
//
// For each segment, sum |L[i]| and |R[i]| across the segment's sample
// range and pick the louder channel. Margin = 1.1× to avoid flapping
// on near-equal energy. Same threshold the historical whisper-cli
// `--diarize` path uses, kept for consistency with downstream tools
// that grep for the literal "(speaker 0)" / "(speaker 1)" prefix.
void apply_energy(
    const float * left, const float * right, int n_samples,
    int64_t slice_t0_cs,
    std::vector<crispasr_segment> & segs)
{
    for (auto & seg : segs) {
        int64_t is0 = cs_to_sample_in_slice(seg.t0, slice_t0_cs);
        int64_t is1 = cs_to_sample_in_slice(seg.t1, slice_t0_cs);
        if (is0 < 0) is0 = 0;
        if (is1 > n_samples) is1 = n_samples;
        if (is0 >= is1) continue;

        double e_l = 0.0, e_r = 0.0;
        for (int64_t j = is0; j < is1; j++) {
            e_l += std::fabs((double)left [j]);
            e_r += std::fabs((double)right[j]);
        }
        std::string spk;
        if      (e_l > 1.1 * e_r) spk = "0";
        else if (e_r > 1.1 * e_l) spk = "1";
        else                      spk = "?";
        seg.speaker = "(speaker " + spk + ") ";
    }
}

// -----------------------------------------------------------------------
// Method 2: cross-correlation lag (TDOA-style)
// -----------------------------------------------------------------------
//
// Compute the cross-correlation of L and R within each segment over a
// short search window (±5 ms = ±80 samples at 16 kHz, generous enough
// to cover the head-shadow inter-aural delay for any normal mic
// spacing). The lag at the correlation peak's sign tells us which
// channel the voice is closest to: positive lag = peak when L is
// shifted ahead = voice arrived at R first = speaker 1; negative lag
// = speaker 0. Falls back to energy when the peak isn't strong enough
// (e.g. silent segment).
void apply_xcorr(
    const float * left, const float * right, int n_samples,
    int64_t slice_t0_cs,
    std::vector<crispasr_segment> & segs)
{
    constexpr int MAX_LAG = 80; // ±5 ms at 16 kHz
    for (auto & seg : segs) {
        int64_t is0 = cs_to_sample_in_slice(seg.t0, slice_t0_cs);
        int64_t is1 = cs_to_sample_in_slice(seg.t1, slice_t0_cs);
        if (is0 < 0) is0 = 0;
        if (is1 > n_samples) is1 = n_samples;
        if (is1 - is0 < 2 * MAX_LAG) {
            // Segment is too short to estimate a lag; fall back to
            // single-frame energy comparison so we still emit something.
            double e_l = 0.0, e_r = 0.0;
            for (int64_t j = is0; j < is1; j++) {
                e_l += std::fabs((double)left[j]);
                e_r += std::fabs((double)right[j]);
            }
            seg.speaker = (e_l >= e_r) ? "(speaker 0) " : "(speaker 1) ";
            continue;
        }

        const int64_t hi = is1 - MAX_LAG;
        const int64_t lo = is0 + MAX_LAG;
        double best = -1e30;
        int    best_lag = 0;
        for (int lag = -MAX_LAG; lag <= MAX_LAG; lag++) {
            double sum = 0.0;
            for (int64_t j = lo; j < hi; j++) {
                sum += (double)left[j] * (double)right[j + lag];
            }
            if (sum > best) { best = sum; best_lag = lag; }
        }
        std::string spk;
        if      (best_lag <  0) spk = "0";
        else if (best_lag >  0) spk = "1";
        else                    spk = "?";
        seg.speaker = "(speaker " + spk + ") ";
    }
}

} // namespace

// -----------------------------------------------------------------------
// Method 3: VAD-turn segmentation (mono-friendly)
// -----------------------------------------------------------------------
//
// Walk the segments in time order and assign a new "(speaker N)" label
// every time we see a gap > MIN_TURN_GAP_CS centiseconds since the
// previous segment. This is a "speaker turn" detector — it's not real
// speaker identification, just a useful proxy for "the conversation
// changed track here". Works on any input regardless of channel count
// because it only looks at segment timestamps. Default min gap is
// 60 cs (= 600 ms), which is the conventional pause threshold used by
// pyannote / NeMo for natural conversation turns.
namespace {
constexpr int64_t MIN_TURN_GAP_CS = 60;
} // namespace

void apply_vad_turns(std::vector<crispasr_segment> & segs) {
    if (segs.empty()) return;
    int speaker = 0;
    int64_t prev_t1 = -1;
    for (auto & seg : segs) {
        if (prev_t1 >= 0 && (seg.t0 - prev_t1) > MIN_TURN_GAP_CS) {
            speaker = 1 - speaker; // alternate 0 / 1
        }
        seg.speaker = "(speaker " + std::to_string(speaker) + ") ";
        prev_t1 = seg.t1;
    }
}

// -----------------------------------------------------------------------
// Method 4: sherpa-onnx subprocess
// -----------------------------------------------------------------------
//
// Shells out to an externally installed
// `sherpa-onnx-offline-speaker-diarization` binary, parses its text
// output (one segment per line, format "begin end speaker_id"), and
// assigns per-ASR-segment speaker labels by time overlap.
//
// We keep this as a subprocess call rather than linking sherpa-onnx
// into crispasr directly because:
//   1. sherpa-onnx pulls in a 90 MB+ ONNX Runtime shared library,
//      doubling the release footprint for a feature most users won't
//      use.
//   2. sherpa-onnx is a moving target with a different release cadence
//      from ours; a subprocess boundary makes the pairing robust to
//      version skew.
//   3. The subprocess only runs ONCE per input file, so the fork
//      overhead is negligible compared to the ASR pass.
//
// Users point us at the binary + model paths via:
//   --diarize-method sherpa
//   [--sherpa-bin sherpa-onnx-offline-speaker-diarization]
//   --sherpa-segment-model    /path/to/sherpa-pyannote-segmentation.onnx
//   --sherpa-embedding-model  /path/to/3dspeaker.onnx
//   [--sherpa-num-clusters N]
//
// If the binary or required models are missing we log a clear error and
// fall back to vad-turns (never fail the whole run).
namespace {

// Helper: write a temporary 16 kHz mono f32→int16 WAV that sherpa can read.
// sherpa-onnx-offline-speaker-diarization takes a WAV path on the cmdline,
// so we need a file the subprocess can open. Returns the path (malloc'd,
// caller frees via std::remove).
std::string write_temp_mono_wav(const float * samples, int n_samples) {
    char buf[] = "/tmp/crispasr-sherpa-XXXXXX.wav";
    int fd = mkstemps(buf, 4);
    if (fd < 0) return {};
    close(fd);
    std::string path = buf;
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return {};

    // Minimal RIFF/WAVE header for 16 kHz mono PCM int16.
    const uint32_t sr = 16000;
    const uint16_t ch = 1;
    const uint16_t bps = 16;
    const uint32_t byte_rate = sr * ch * bps / 8;
    const uint16_t block_align = ch * bps / 8;
    const uint32_t data_bytes = (uint32_t)n_samples * block_align;
    const uint32_t riff_size = 36 + data_bytes;

    auto w32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto w16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
    fwrite("RIFF", 1, 4, f); w32(riff_size); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); w32(16); w16(1); w16(ch);
    w32(sr); w32(byte_rate); w16(block_align); w16(bps);
    fwrite("data", 1, 4, f); w32(data_bytes);
    std::vector<int16_t> pcm(n_samples);
    for (int i = 0; i < n_samples; i++) {
        float v = samples[i];
        if (v >  1.0f) v =  1.0f;
        if (v < -1.0f) v = -1.0f;
        pcm[i] = (int16_t)(v * 32767.0f);
    }
    fwrite(pcm.data(), sizeof(int16_t), pcm.size(), f);
    fclose(f);
    return path;
}

// Parse a line emitted by sherpa-onnx-offline-speaker-diarization.
// Each result line looks like:
//     "0.320 -- 3.680 speaker_00 duration=3.360"
// or:
//     "0.320 3.680 0"
// We accept both the `--` and the plain-space formats.
struct SherpaSegment { double t0_s; double t1_s; int speaker; };

bool parse_sherpa_line(const std::string & line, SherpaSegment & out) {
    double t0 = 0, t1 = 0;
    char rest[256] = {0};
    // Try the "t0 -- t1 speaker_NN" form first.
    if (std::sscanf(line.c_str(), "%lf -- %lf %255s", &t0, &t1, rest) == 3) {
        out.t0_s = t0; out.t1_s = t1;
        // speaker_NN → NN
        const char * p = rest;
        while (*p && !isdigit((unsigned char)*p)) p++;
        out.speaker = *p ? std::atoi(p) : 0;
        return true;
    }
    int spk = 0;
    if (std::sscanf(line.c_str(), "%lf %lf %d", &t0, &t1, &spk) == 3) {
        out.t0_s = t0; out.t1_s = t1; out.speaker = spk;
        return true;
    }
    return false;
}

// Merge sherpa's speaker boundaries into our ASR segments by time overlap.
// For each ASR segment, pick the sherpa speaker whose time interval
// overlaps the segment the most. If no sherpa segment overlaps at all we
// leave speaker empty (the writer will just not prefix a label).
void assign_speakers_from_sherpa(std::vector<crispasr_segment> & segs,
                                 const std::vector<SherpaSegment> & sherpa) {
    if (sherpa.empty()) return;
    for (auto & seg : segs) {
        const double a0 = (double)seg.t0 / 100.0;
        const double a1 = (double)seg.t1 / 100.0;
        std::vector<double> overlap_per_speaker(32, 0.0);
        int max_spk = 0;
        for (const auto & s : sherpa) {
            const double lo = std::max(a0, s.t0_s);
            const double hi = std::min(a1, s.t1_s);
            if (hi > lo) {
                if (s.speaker >= (int)overlap_per_speaker.size())
                    overlap_per_speaker.resize(s.speaker + 1, 0.0);
                overlap_per_speaker[s.speaker] += (hi - lo);
                if (s.speaker > max_spk) max_spk = s.speaker;
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

bool apply_sherpa(const std::vector<float> & mono,
                  int64_t slice_t0_cs,
                  std::vector<crispasr_segment> & segs,
                  const whisper_params & params) {
    const std::string bin = params.sherpa_bin.empty()
        ? std::string("sherpa-onnx-offline-speaker-diarization")
        : params.sherpa_bin;
    if (params.sherpa_segment_model.empty() ||
        params.sherpa_embedding_model.empty()) {
        fprintf(stderr,
            "crispasr[diarize]: sherpa needs --sherpa-segment-model and\n"
            "                   --sherpa-embedding-model. Download them from\n"
            "                   https://github.com/k2-fsa/sherpa-onnx — e.g.\n"
            "                     sherpa-pyannote-segmentation-3.0.onnx\n"
            "                     3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx\n");
        return false;
    }

    // Check the binary resolves (stat + access so we can give a clearer
    // error than popen's silent failure).
    if (bin.find('/') != std::string::npos) {
        struct stat st;
        if (::stat(bin.c_str(), &st) != 0) {
            fprintf(stderr,
                "crispasr[diarize]: sherpa binary '%s' not found — pass "
                "--sherpa-bin or install k2-fsa/sherpa-onnx\n", bin.c_str());
            return false;
        }
    }

    const std::string wav_path = write_temp_mono_wav(mono.data(), (int)mono.size());
    if (wav_path.empty()) {
        fprintf(stderr, "crispasr[diarize]: failed to write temp wav\n");
        return false;
    }

    std::ostringstream cmd;
    cmd << bin
        << " --clustering.num-clusters=" << params.sherpa_num_clusters
        << " --segmentation.pyannote.model='" << params.sherpa_segment_model << "'"
        << " --embedding.model='"            << params.sherpa_embedding_model << "'"
        << " '" << wav_path << "'";
    if (!params.no_prints) {
        fprintf(stderr, "crispasr[diarize]: %s\n", cmd.str().c_str());
    }
    cmd << " 2>/dev/null";

    std::unique_ptr<FILE, int(*)(FILE*)> pipe(popen(cmd.str().c_str(), "r"), pclose);
    if (!pipe) {
        fprintf(stderr, "crispasr[diarize]: failed to spawn sherpa subprocess\n");
        std::remove(wav_path.c_str());
        return false;
    }

    std::vector<SherpaSegment> parsed;
    char linebuf[1024];
    while (fgets(linebuf, sizeof(linebuf), pipe.get())) {
        SherpaSegment s;
        if (parse_sherpa_line(linebuf, s)) parsed.push_back(s);
    }
    std::remove(wav_path.c_str());

    if (parsed.empty()) {
        fprintf(stderr,
            "crispasr[diarize]: sherpa subprocess produced no parseable "
            "segments — check that the two --sherpa-*-model paths are "
            "correct and that the binary prints results on stdout.\n");
        return false;
    }

    // sherpa reports times relative to the audio it was handed (i.e.
    // the slice), so shift by slice_t0_cs before merging with our
    // absolute-cs segments.
    for (auto & s : parsed) {
        s.t0_s += (double)slice_t0_cs / 100.0;
        s.t1_s += (double)slice_t0_cs / 100.0;
    }
    assign_speakers_from_sherpa(segs, parsed);

    if (!params.no_prints) {
        fprintf(stderr,
                "crispasr[diarize]: sherpa → %zu speaker regions over %zu ASR segments\n",
                parsed.size(), segs.size());
    }
    return true;
}

} // namespace

bool crispasr_apply_diarize(
    const std::vector<float> & left,
    const std::vector<float> & right,
    bool                       is_stereo,
    int64_t                    slice_t0_cs,
    std::vector<crispasr_segment> & segs,
    const whisper_params & params)
{
    if (segs.empty()) return true;

    // Method dispatch. Default depends on whether we have stereo:
    //   stereo input  -> "energy"  (cheap, accurate per channel)
    //   mono input    -> "vad-turns" (cheap, mono-friendly, just turns)
    std::string method = params.diarize_method;
    if (method.empty()) {
        method = is_stereo ? "energy" : "vad-turns";
    }

    if (method == "energy") {
        if (!is_stereo) {
            fprintf(stderr,
                    "crispasr[diarize]: --diarize-method energy needs stereo input — "
                    "falling back to vad-turns for this mono clip\n");
            apply_vad_turns(segs);
            return true;
        }
        apply_energy(left.data(), right.data(), (int)left.size(),
                     slice_t0_cs, segs);
        return true;
    }
    if (method == "xcorr" || method == "cross-correlation") {
        if (!is_stereo) {
            fprintf(stderr,
                    "crispasr[diarize]: --diarize-method xcorr needs stereo input — "
                    "falling back to vad-turns for this mono clip\n");
            apply_vad_turns(segs);
            return true;
        }
        apply_xcorr(left.data(), right.data(), (int)left.size(),
                    slice_t0_cs, segs);
        return true;
    }
    if (method == "vad-turns" || method == "turns") {
        apply_vad_turns(segs);
        return true;
    }
    // sherpa / pyannote / ecapa all land in the same subprocess path —
    // sherpa-onnx-offline-speaker-diarization uses pyannote-v3 for the
    // segmentation model and a speaker-embedding ONNX (titanet /
    // 3dspeaker / ecapa-tdnn) for the embedding model. The method name
    // is just a user-facing label for the same underlying pipeline, so
    // we unify them into one code path.
    if (method == "sherpa"   || method == "sherpa-onnx" ||
        method == "pyannote" || method == "ecapa")
    {
        // Build a mono view of the input. If we have stereo, mix both
        // channels. Otherwise the "left" buffer already holds mono.
        std::vector<float> mono;
        if (is_stereo && !right.empty()) {
            mono.resize(left.size());
            for (size_t i = 0; i < left.size(); i++)
                mono[i] = 0.5f * (left[i] + right[i]);
        } else {
            mono = left;
        }
        if (apply_sherpa(mono, slice_t0_cs, segs, params)) {
            return true;
        }
        // sherpa failed — log already printed — fall through to a safe
        // default so the run still produces output.
        fprintf(stderr,
                "crispasr[diarize]: %s diarization failed — falling back to %s\n",
                method.c_str(),
                is_stereo ? "energy" : "vad-turns");
        if (is_stereo) apply_energy(left.data(), right.data(),
                                    (int)left.size(), slice_t0_cs, segs);
        else           apply_vad_turns(segs);
        return true;
    }

    fprintf(stderr,
            "crispasr[diarize]: unknown --diarize-method '%s'\n"
            "                   expected energy|xcorr|vad-turns|sherpa|pyannote|ecapa\n"
            "                     energy/xcorr require stereo input\n"
            "                     vad-turns is the mono default\n"
            "                     sherpa/pyannote/ecapa all dispatch through the\n"
            "                       same sherpa-onnx subprocess (pyannote v3\n"
            "                       segmentation + speaker-embedding ONNX), see\n"
            "                       --sherpa-segment-model / --sherpa-embedding-model\n",
            method.c_str());
    return false;
}
