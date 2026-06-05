// crispasr_diarize_cli.cpp — CLI-side diarization shim.
//
// Routes the four in-process diarization methods to the shared library
// (`src/crispasr_diarize.cpp`) and keeps the sherpa-ONNX subprocess
// method here, since it shells out to an externally installed binary
// and is CLI-shaped UX. Also handles auto-download of the pyannote
// GGUF via `crispasr_cache`.

#include "crispasr_diarize_cli.h"
#include "crispasr_cache.h"
#include "crispasr_diarize_internal.h"
#include "crispasr_speaker_cluster.h"
#include "crispasr_speaker_embedder.h"
#include "pyannote_seg.h"
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
#include <direct.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#define close _close
#define popen _popen
#define pclose _pclose
#define mkdir(d, m) _mkdir(d)
static int mkstemps(char* t, int s) {
    (void)s;
    return _mktemp_s(t, strlen(t) + 1) == 0 ? _open(t, _O_CREAT | _O_WRONLY, 0600) : -1;
}
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

std::string scratch_dir() {
    const char* env = std::getenv("CRISPASR_SCRATCH_DIR");
    std::string d = (env && *env) ? std::string(env) : crispasr_cache::dir() + "/scratch";
    mkdir(d.c_str(), 0755);
    return d;
}

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
    std::string tmpl_s = scratch_dir() + "/crispasr-sherpa-XXXXXX.wav";
    std::vector<char> buf(tmpl_s.begin(), tmpl_s.end());
    buf.push_back('\0');
    int fd = mkstemps(buf.data(), 4);
    if (fd < 0)
        return {};
    close(fd);
    std::string path = buf.data();
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
        << " --segmentation.pyannote-model='" << params.sherpa_segment_model << "'"
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

// Assign speakers from a pre-computed global sherpa timeline.
// Same logic as assign_speakers_from_sherpa but also splits segments
// at speaker-turn boundaries when word timestamps are available.
void assign_speakers_from_global_sherpa(std::vector<crispasr_segment>& segs, const CrispasrSherpaCache& cache) {
    if (!cache.valid() || segs.empty())
        return;

    // Convert cache segments to the local SherpaSegment type for reuse
    std::vector<SherpaSegment> sherpa_segs;
    sherpa_segs.reserve(cache.segments.size());
    for (const auto& cs : cache.segments)
        sherpa_segs.push_back({cs.t0_s, cs.t1_s, cs.speaker});

    // Phase 1: assign dominant speaker to each whole ASR segment
    assign_speakers_from_sherpa(segs, sherpa_segs);

    // Phase 2: split segments at speaker-turn boundaries using word timestamps
    std::vector<crispasr_segment> out;
    out.reserve(segs.size());

    for (auto& seg : segs) {
        if (seg.words.empty()) {
            out.push_back(std::move(seg));
            continue;
        }

        // Per-word speaker label from global timeline
        std::vector<int> word_spk(seg.words.size(), -1);
        for (size_t i = 0; i < seg.words.size(); i++) {
            const auto& w = seg.words[i];
            if (w.t1 <= w.t0)
                continue;
            const double w0 = (double)w.t0 / 100.0;
            const double w1 = (double)w.t1 / 100.0;
            // Find the sherpa segment with maximum overlap
            int best = -1;
            double best_ov = 0.0;
            for (const auto& s : sherpa_segs) {
                double lo = std::max(w0, s.t0_s);
                double hi = std::min(w1, s.t1_s);
                if (hi > lo && (hi - lo) > best_ov) {
                    best_ov = hi - lo;
                    best = s.speaker;
                }
            }
            word_spk[i] = best;
        }

        // Carry-forward for unaligned words
        int last_known = -1;
        for (size_t i = 0; i < word_spk.size(); i++) {
            if (word_spk[i] >= 0)
                last_known = word_spk[i];
            else if (last_known >= 0)
                word_spk[i] = last_known;
        }
        // Back-fill leading unknowns
        for (size_t i = 0; i < word_spk.size() && word_spk[i] < 0; i++) {
            for (size_t j = i + 1; j < word_spk.size(); j++) {
                if (word_spk[j] >= 0) {
                    word_spk[i] = word_spk[j];
                    break;
                }
            }
        }

        // Check if all words have the same speaker — skip splitting if so
        bool all_same = true;
        for (size_t i = 1; i < word_spk.size(); i++) {
            if (word_spk[i] != word_spk[0]) {
                all_same = false;
                break;
            }
        }
        if (all_same) {
            out.push_back(std::move(seg));
            continue;
        }

        // Split at speaker transitions
        size_t run_start = 0;
        for (size_t i = 1; i <= seg.words.size(); i++) {
            if (i < seg.words.size() && word_spk[i] == word_spk[run_start])
                continue;
            // Emit sub-segment [run_start, i)
            crispasr_segment sub;
            sub.t0 = seg.words[run_start].t0 > 0 ? seg.words[run_start].t0 : seg.t0;
            sub.t1 = seg.words[i - 1].t1 > 0 ? seg.words[i - 1].t1 : seg.t1;
            if (word_spk[run_start] >= 0)
                sub.speaker = "(speaker " + std::to_string(word_spk[run_start]) + ") ";
            else
                sub.speaker = seg.speaker;
            // Rebuild text from words
            std::string txt;
            for (size_t j = run_start; j < i; j++) {
                if (!txt.empty())
                    txt += ' ';
                txt += seg.words[j].text;
            }
            sub.text = txt;
            // Copy word data for the sub-segment
            sub.words.assign(seg.words.begin() + run_start, seg.words.begin() + i);
            out.push_back(std::move(sub));
            run_start = i;
        }
    }

    segs = std::move(out);
}

} // namespace

namespace {

// Split each multi-speaker ASR segment into runs of same-speaker words.
// Requires per-word timestamps (seg.words populated with non-zero t1
// for at least the boundary words). Segments without words are left
// untouched and labelled with the segment-level dominant speaker.
//
// This is what makes a 27-second cohere segment that actually contains
// two speakers come out as multiple sub-segments after diarize, instead
// of inheriting a single dominant label (#107 segment-splitting bit).
//
// Algorithm:
//   1. For each word, look up the dominant pyannote speaker over
//      [w.t0, w.t1] against the cached posteriors. Words with no
//      timestamp (t0==t1==0) inherit the previous word's speaker.
//   2. Walk the words. Whenever the per-word speaker label changes
//      (ignoring -1 silence runs, which are absorbed by their
//      neighbours), close the previous run as a sub-segment and start
//      a new one.
//   3. Each emitted sub-segment carries the contiguous word run, a
//      rebuilt `text` (joined on space), and inherits other fields
//      from the original.
void split_segments_on_pyannote_turns(std::vector<crispasr_segment>& segs, const CrispasrPyannoteCache& cache) {
    if (!cache.valid() || segs.empty())
        return;

    using crispasr_diarize_internal::score_speaker_for_range;

    std::vector<crispasr_segment> out;
    out.reserve(segs.size());

    for (auto& seg : segs) {
        if (seg.words.empty()) {
            // No word timestamps — can't split text. Leave segment-
            // level labelling (already applied) in place.
            out.push_back(std::move(seg));
            continue;
        }

        // Per-word speaker label. -1 for "no info" (zero-duration
        // word or silence-dominated frame range).
        std::vector<int> word_spk(seg.words.size(), -1);
        int last_known = -1;
        for (size_t i = 0; i < seg.words.size(); i++) {
            const auto& w = seg.words[i];
            int spk = -1;
            if (w.t1 > w.t0) {
                spk = score_speaker_for_range(cache.log_probs.data(), cache.T, cache.frame_dur_s, w.t0, w.t1);
            }
            // Carry forward the previous known speaker for unaligned
            // words so they stay attached to their neighbour's run
            // instead of becoming their own silence stub.
            if (spk < 0)
                spk = last_known;
            word_spk[i] = spk;
            if (spk >= 0)
                last_known = spk;
        }
        // Back-fill leading -1s with the first known speaker so a
        // run-start never sits at -1.
        for (size_t i = 0; i < word_spk.size() && word_spk[i] < 0; i++) {
            for (size_t j = i + 1; j < word_spk.size(); j++) {
                if (word_spk[j] >= 0) {
                    word_spk[i] = word_spk[j];
                    break;
                }
            }
        }

        // Stability filter: pyannote-seg without speaker embeddings
        // can flip its local track index mid-phrase, producing one-word
        // "(speaker 2)" stubs inside a contiguous "(speaker 1)" run.
        // We collapse any run shorter than MIN_RUN_CS into the longer
        // adjacent run before emitting sub-segments. Tuned at 50 cs
        // (0.5 s) — long enough to suppress per-word flips, short enough
        // to keep genuine turns in fast back-and-forth conversation.
        constexpr int64_t MIN_RUN_CS = 50;
        auto word_duration_cs = [&](size_t a, size_t b /*exclusive*/) -> int64_t {
            if (b <= a || b > word_spk.size())
                return 0;
            int64_t t0 = seg.words[a].t0;
            int64_t t1 = seg.words[b - 1].t1;
            if (t0 <= 0)
                t0 = seg.t0;
            if (t1 <= 0)
                t1 = seg.t1;
            return std::max<int64_t>(0, t1 - t0);
        };
        // Build initial runs as [start, end_exclusive, speaker] triples.
        struct Run {
            size_t s, e;
            int spk;
        };
        std::vector<Run> runs;
        {
            size_t rs = 0;
            while (rs < word_spk.size()) {
                const int rspk = word_spk[rs];
                size_t re = rs + 1;
                while (re < word_spk.size() && word_spk[re] == rspk)
                    re++;
                runs.push_back({rs, re, rspk});
                rs = re;
            }
        }
        // Iterate: fold every run shorter than MIN_RUN_CS into the
        // longer neighbour. Repeat until stable.
        bool changed = true;
        while (changed && runs.size() >= 2) {
            changed = false;
            for (size_t i = 0; i < runs.size(); i++) {
                if (word_duration_cs(runs[i].s, runs[i].e) >= MIN_RUN_CS)
                    continue;
                int merge_into = -1;
                if (i == 0)
                    merge_into = (int)i + 1;
                else if (i == runs.size() - 1)
                    merge_into = (int)i - 1;
                else {
                    int64_t prev_dur = word_duration_cs(runs[i - 1].s, runs[i - 1].e);
                    int64_t next_dur = word_duration_cs(runs[i + 1].s, runs[i + 1].e);
                    merge_into = (prev_dur >= next_dur) ? (int)i - 1 : (int)i + 1;
                }
                if (merge_into == (int)i + 1) {
                    runs[i + 1].s = runs[i].s;
                } else if (merge_into == (int)i - 1) {
                    runs[i - 1].e = runs[i].e;
                }
                runs.erase(runs.begin() + i);
                changed = true;
                break;
            }
        }

        // Collect distinct speakers across the (now-filtered) runs to
        // decide whether to actually split.
        int first_spk = -1;
        bool multi = false;
        for (const auto& r : runs) {
            if (r.spk < 0)
                continue;
            if (first_spk < 0) {
                first_spk = r.spk;
            } else if (r.spk != first_spk) {
                multi = true;
                break;
            }
        }
        if (!multi) {
            // Single speaker (or none) after filtering — keep segment.
            out.push_back(std::move(seg));
            continue;
        }

        // Emit one sub-segment per run.
        for (size_t ri = 0; ri < runs.size(); ri++) {
            const size_t run_start = runs[ri].s;
            const size_t run_end = runs[ri].e;
            const int run_spk = runs[ri].spk;

            crispasr_segment sub;
            sub.t0 = seg.words[run_start].t0 > 0 ? seg.words[run_start].t0 : seg.t0;
            sub.t1 = seg.words[run_end - 1].t1 > 0 ? seg.words[run_end - 1].t1 : seg.t1;
            sub.speaker_turn_next = (ri + 1 < runs.size()); // turn change at boundary
            sub.words.assign(seg.words.begin() + run_start, seg.words.begin() + run_end);
            // Rebuild text from the words in this run. Whitespace
            // handling matches the canonical CrispASR convention:
            // each word stands alone and we join on a single space.
            for (size_t j = run_start; j < run_end; j++) {
                if (!sub.text.empty() && !sub.words[j - run_start].text.empty())
                    sub.text += ' ';
                sub.text += seg.words[j].text;
            }
            // Tokens carry no per-word alignment, so we drop them on
            // a split rather than dividing them ambiguously.
            sub.tokens.clear();
            if (run_spk >= 0)
                sub.speaker = "(speaker " + std::to_string(run_spk) + ") ";
            out.push_back(std::move(sub));
        }
    }

    segs = std::move(out);
}

} // namespace

bool crispasr_compute_pyannote_cache(const float* full_audio, int n_samples, const whisper_params& params,
                                     CrispasrPyannoteCache& out) {
    out = {};
    if (!full_audio || n_samples <= 0)
        return false;
    std::string mp = resolve_pyannote_model(params);
    if (mp.empty())
        return false;

    pyannote_seg_context* pctx = pyannote_seg_init(mp.c_str(), params.n_threads);
    if (!pctx)
        return false;
    int T = 0;
    float* probs = pyannote_seg_run(pctx, full_audio, n_samples, &T);
    pyannote_seg_free(pctx);
    if (!probs || T <= 0) {
        if (probs)
            std::free(probs);
        return false;
    }

    out.log_probs.assign(probs, probs + (size_t)T * 7);
    out.T = T;
    // Frame duration: sinc(stride=10) × 3 maxpools(stride=3) = 270 samples = 16.875 ms.
    out.frame_dur_s = 270.0 / 16000.0;
    std::free(probs);
    return true;
}

bool crispasr_compute_sherpa_cache(const float* full_audio, int n_samples, const whisper_params& params,
                                   CrispasrSherpaCache& out) {
    out = {};
    if (!full_audio || n_samples <= 0)
        return false;

    const std::string bin =
        params.sherpa_bin.empty() ? std::string("sherpa-onnx-offline-speaker-diarization") : params.sherpa_bin;
    if (params.sherpa_segment_model.empty() || params.sherpa_embedding_model.empty()) {
        fprintf(stderr, "crispasr[diarize]: sherpa global cache needs --sherpa-segment-model and\n"
                        "                   --sherpa-embedding-model.\n");
        return false;
    }

    if (!params.no_prints)
        fprintf(stderr, "crispasr[diarize]: computing global sherpa timeline over %d samples (%.1f s)...\n", n_samples,
                (double)n_samples / 16000.0);

    const std::string wav_path = write_temp_mono_wav(full_audio, n_samples);
    if (wav_path.empty()) {
        fprintf(stderr, "crispasr[diarize]: failed to write temp wav for global sherpa\n");
        return false;
    }

    std::ostringstream cmd;
    cmd << bin << " --clustering.num-clusters=" << params.sherpa_num_clusters << " --segmentation.pyannote-model='"
        << params.sherpa_segment_model << "'" << " --embedding.model='" << params.sherpa_embedding_model << "'" << " '"
        << wav_path << "'";
    if (!params.no_prints)
        fprintf(stderr, "crispasr[diarize]: %s\n", cmd.str().c_str());
    cmd << " 2>/dev/null";

    std::unique_ptr<FILE, int (*)(FILE*)> pipe(popen(cmd.str().c_str(), "r"), pclose);
    if (!pipe) {
        fprintf(stderr, "crispasr[diarize]: failed to spawn sherpa subprocess for global cache\n");
        std::remove(wav_path.c_str());
        return false;
    }

    char linebuf[1024];
    while (fgets(linebuf, sizeof(linebuf), pipe.get())) {
        SherpaSegment s;
        if (parse_sherpa_line(linebuf, s))
            out.segments.push_back({s.t0_s, s.t1_s, s.speaker});
    }
    std::remove(wav_path.c_str());

    if (out.segments.empty()) {
        fprintf(stderr, "crispasr[diarize]: sherpa global run produced no segments\n");
        return false;
    }

    if (!params.no_prints)
        fprintf(stderr, "crispasr[diarize]: sherpa global → %zu speaker regions\n", out.segments.size());
    return true;
}

bool crispasr_apply_diarize(const std::vector<float>& left, const std::vector<float>& right, bool is_stereo,
                            int64_t slice_t0_cs, std::vector<crispasr_segment>& segs, const whisper_params& params,
                            const CrispasrPyannoteCache* pyannote_cache, const CrispasrSherpaCache* sherpa_cache) {
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

    // Pyannote cache short-circuit: when the runner pre-computed the
    // segmentation posteriors over the full audio (issue #107 cross-
    // slice fix), skip the per-slice pyannote_seg_run and score this
    // slice's segments directly against the cached posteriors.
    //
    // The cached posteriors cover absolute time [0, T*frame_dur_s), and
    // segs[i].t0/.t1 are already absolute, so we pass slice_t0_cs=0
    // into assign_speakers_from_log_posteriors (the cache buffer's
    // origin), independent of the per-slice slice_t0_cs.
    if (use_lib && lib_method == CrispasrDiarizeMethod::Pyannote && pyannote_cache && pyannote_cache->valid()) {
        // Phase 1 first: assign each ASR segment its dominant speaker
        // (and crucially DO leave seg.speaker populated for segments
        // that get further split by the next step, since the splitter
        // overrides per sub-segment via word-range scoring).
        {
            auto lib_segs = lib_view(segs);
            crispasr_diarize_internal::assign_speakers_from_log_posteriors(
                pyannote_cache->log_probs.data(), pyannote_cache->T, pyannote_cache->frame_dur_s,
                /*slice_t0_cs=*/0, lib_segs);
            apply_int_speakers_to_crispasr_segments(lib_segs, segs);
        }
        // Phase 2: split multi-speaker segments at word-aligned turn
        // boundaries (#107). Segments without word timestamps keep
        // their Phase 1 single-speaker label.
        split_segments_on_pyannote_turns(segs, *pyannote_cache);
        return true;
    }

    // Sherpa cache short-circuit (issue #110): when the runner pre-computed
    // the global sherpa speaker timeline over the full audio, use it
    // directly instead of re-invoking the subprocess per slice.
    if (sherpa_cache && sherpa_cache->valid() &&
        (method == "sherpa" || method == "sherpa-onnx" || method == "ecapa" || !use_lib)) {
        assign_speakers_from_global_sherpa(segs, *sherpa_cache);
        return true;
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

void crispasr_remap_speakers_via_embeddings(std::vector<crispasr_segment>& segs, const float* full_audio, int n_samples,
                                            CrispasrSpeakerEmbedder* embedder, const whisper_params& params) {
    if (!embedder || segs.empty() || !full_audio || n_samples <= 0)
        return;

    // Skip very short segments — TitaNet (and similar) need ~250 ms+
    // of speech to produce a reliable embedding. Shorter segments
    // keep their existing pyannote-local label; clustering then
    // ignores them.
    constexpr int64_t MIN_EMBED_CS = 25; // 0.25 s

    const int d = embedder->dim();
    if (d <= 0)
        return;

    std::vector<size_t> embed_idx;
    std::vector<float> embeddings;
    embed_idx.reserve(segs.size());
    embeddings.reserve((size_t)segs.size() * (size_t)d);

    std::vector<float> tmp(d);
    for (size_t i = 0; i < segs.size(); i++) {
        const auto& seg = segs[i];
        if ((seg.t1 - seg.t0) < MIN_EMBED_CS)
            continue;
        // Convert cs (1/100 s) → 16 kHz samples. full_audio is mono 16k.
        const int64_t s0 = std::max<int64_t>(0, seg.t0 * 160);
        const int64_t s1 = std::min<int64_t>(n_samples, seg.t1 * 160);
        if (s1 - s0 < 4000) // <250 ms after clamping
            continue;
        if (!embedder->embed(full_audio + s0, (int)(s1 - s0), tmp.data()))
            continue;
        embed_idx.push_back(i);
        embeddings.insert(embeddings.end(), tmp.begin(), tmp.end());
    }
    if (embed_idx.size() < 2) {
        // Nothing to cluster — either zero or one usable segment.
        if (embed_idx.size() == 1) {
            // Force the one embeddable segment to (speaker 0) so the
            // global label exists even on single-speaker inputs.
            segs[embed_idx[0]].speaker = "(speaker 0) ";
        }
        return;
    }

    const float thr = params.diarize_cluster_threshold;
    const int max_spk = params.diarize_max_speakers > 0 ? params.diarize_max_speakers : 8;
    std::vector<int> labels = crispasr_agglomerative_cluster(embeddings, (int)embed_idx.size(), d, thr, max_spk);

    // Rewrite segment speakers from clustering output. Segments that
    // couldn't be embedded (too short) keep their existing pyannote-
    // local label as a best-effort fallback.
    for (size_t k = 0; k < embed_idx.size(); k++) {
        const int spk = labels[k];
        if (spk < 0)
            continue;
        segs[embed_idx[k]].speaker = "(speaker " + std::to_string(spk) + ") ";
    }
}
