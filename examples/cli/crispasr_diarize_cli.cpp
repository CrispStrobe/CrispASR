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
#include "crispasr_model_registry.h"
#include "crispasr_speaker_cluster.h"
#include "crispasr_speaker_embedder.h"
#include "crispasr_subprocess.h"
#include "pyannote_seg.h"
#include "speaker_db.h"
#include "whisper_params.h"

#include <set>

#include "core/spectral_diarize.h"

#include <algorithm>
#include <chrono>
#include <atomic> // cross-segment embed workers (#326)
#include <thread>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
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
#define mkdir(d, m) _mkdir(d)
static int mkstemps(char* t, int s) {
    (void)s;
    return _mktemp_s(t, strlen(t) + 1) == 0 ? _open(t, _O_CREAT | _O_WRONLY, 0600) : -1;
}
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

// #324: conservative ceiling for foxnose speaker-count estimation. See the
// comment at its use site — this is an empirical value, not a guess.
// Was 4 while the BIC + silhouette estimator saturated at whatever ceiling it
// was given. The eigengap estimator (now the default) is robust to a loose
// bound — measured 2 speakers at max=8 on samples/multispeaker.wav where BIC
// reported 8 — so this no longer has to be defensively small, and a genuine
// 5-6 speaker meeting is reachable again.
static constexpr int kFoxnoseDefaultMaxSpeakers = 8;

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

std::vector<std::string> make_sherpa_args(const std::string& bin, const whisper_params& params,
                                          const std::string& wav_path) {
    return {
        bin,
        "--clustering.num-clusters=" + std::to_string(params.sherpa_num_clusters),
        "--segmentation.pyannote-model=" + params.sherpa_segment_model,
        "--embedding.model=" + params.sherpa_embedding_model,
        wav_path,
    };
}

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

    if (bin.find('/') != std::string::npos || bin.find('\\') != std::string::npos) {
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

    const auto args = make_sherpa_args(bin, params, wav_path);
    if (!params.no_prints)
        fprintf(stderr, "crispasr[diarize]: %s\n", crispasr_cli_process::join_cmdline(args).c_str());

    const int timeout_sec =
        crispasr_cli_process::timeout_from_audio_samples("CRISPASR_SHERPA_TIMEOUT_SEC", (int)mono.size());
    const auto run = crispasr_cli_process::run_capture_stdout(args, timeout_sec);
    if (run.timed_out) {
        fprintf(stderr, "crispasr[diarize]: sherpa subprocess timed out after %d s\n", timeout_sec);
        std::remove(wav_path.c_str());
        return false;
    }
    if (run.exit_code != 0) {
        fprintf(stderr, "crispasr[diarize]: sherpa subprocess failed with exit code %d\n", run.exit_code);
        std::remove(wav_path.c_str());
        return false;
    }
    std::vector<SherpaSegment> parsed;
    std::istringstream lines(run.output);
    std::string line;
    while (std::getline(lines, line)) {
        SherpaSegment s;
        if (parse_sherpa_line(line, s))
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
        // MIT, and the GGUF repo is ungated — same licence as the upstream
        // pyannote/segmentation-3.0 it was exported from.
        //
        // This was tagged "other", which the registry treats as restricted (a
        // correct default for an unknown licence — see the `restricted[]` list
        // in crispasr_model_registry.cpp). The effect was that `--diarize-method
        // pyannote` with the default "auto" refused to fetch its own
        // segmentation model and printed a licence warning, so pyannote
        // diarization did not work out of the box at all: it fell through to
        // "sherpa needs --sherpa-segment-model" and produced no speaker turns.
        mp = crispasr_managed_download(
            "pyannote-seg-3.0.gguf",
            "https://huggingface.co/cstr/pyannote-v3-segmentation-GGUF/resolve/main/pyannote-seg-3.0.gguf", "mit",
            params.no_prints, "crispasr[diarize]", params.cache_dir, params.accept_license);
    }
    if (mp.size() < 5 || mp.compare(mp.size() - 5, 5, ".gguf") != 0)
        return {}; // not GGUF → caller can fall back to sherpa subprocess
    return mp;
}

// #324: resolve the WeSpeaker embedder path, auto-downloading the canonical
// GGUF on first use when the user passed "auto" (or left it as the registry
// default). ⚠ CC-BY-4.0 weights — see THIRD_PARTY_NOTICES.txt.
std::string resolve_foxnose_embedder(const whisper_params& params) {
    std::string mp = params.diarize_embedder;
    if (mp.empty() || mp == "auto") {
        CrispasrRegistryEntry entry;
        if (crispasr_registry_lookup("wespeaker", entry)) {
            if (crispasr_license_requires_acceptance(entry.license) &&
                !crispasr_license_accepted(entry.license, params.accept_license)) {
                fprintf(stderr,
                        "crispasr[diarize]: refusing restricted WeSpeaker weights without --accept-license %s\n",
                        crispasr_license_tag(entry.license).c_str());
                return {};
            }
            if (!params.no_prints)
                fprintf(stderr, "crispasr[diarize]: weights license: %s\n", entry.license.c_str());
        }
        mp = crispasr_cache::ensure_cached_file(
            "wespeaker-resnet34-lm.gguf",
            "https://huggingface.co/cstr/wespeaker-resnet34-lm-GGUF/resolve/main/wespeaker-resnet34-lm.gguf",
            params.no_prints, "crispasr[diarize]", params.cache_dir);
    }
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

        // Stability filter (#107/#267): pyannote-seg without speaker
        // embeddings can flip its local track index mid-phrase, producing
        // one-word "(speaker 2)" stubs inside a contiguous "(speaker 1)"
        // run. group_words_into_speaker_runs folds any run shorter than
        // MIN_RUN_CS (0.5 s) into the longer adjacent run — long enough to
        // suppress per-word flips, short enough to keep genuine turns in
        // fast back-and-forth conversation.
        constexpr int64_t MIN_RUN_CS = 50;
        // Sanitize per-word bounds (fall back to segment bounds for missing
        // word timestamps) so each run's span is well-defined.
        std::vector<int64_t> word_t0(seg.words.size()), word_t1(seg.words.size());
        for (size_t i = 0; i < seg.words.size(); i++) {
            word_t0[i] = seg.words[i].t0 > 0 ? seg.words[i].t0 : seg.t0;
            word_t1[i] = seg.words[i].t1 > 0 ? seg.words[i].t1 : seg.t1;
        }
        const auto runs =
            crispasr_diarize_internal::group_words_into_speaker_runs(word_spk, word_t0, word_t1, MIN_RUN_CS);

        // Collect distinct speakers across the (now-filtered) runs to
        // decide whether to actually split.
        int first_spk = -1;
        bool multi = false;
        for (const auto& r : runs) {
            if (r.speaker < 0)
                continue;
            if (first_spk < 0) {
                first_spk = r.speaker;
            } else if (r.speaker != first_spk) {
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
            const size_t run_start = runs[ri].start;
            const size_t run_end = runs[ri].end;
            const int run_spk = runs[ri].speaker;

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

// #324: split segments at FoxNose turn boundaries.
//
// The pyannote splitter above scores each word against pyannote's frame
// posteriors, which FoxNose does not have — it produces explicit TURNS
// instead. Everything after per-word labelling is identical, so this reuses
// group_words_into_speaker_runs and the same sub-segment emission.
//
// Without this, labels attach at the caller's segment granularity: an ASR
// emitting one 26 s segment across several speakers gets ONE label, however
// good the turns are.
void split_segments_on_foxnose_turns(std::vector<crispasr_segment>& segs, const std::vector<CrispasrDiarizeTurn>& turns,
                                     int64_t slice_t0_cs) {
    if (turns.empty() || segs.empty())
        return;

    // Speaker covering the centre of [t0, t1] (centiseconds, absolute).
    auto speaker_at = [&](int64_t t0, int64_t t1) -> int {
        const double mid = ((double)(t0 + t1) * 0.5 - (double)slice_t0_cs) / 100.0;
        for (const auto& t : turns)
            if (mid >= t.start_s && mid < t.end_s)
                return t.speaker;
        return -1;
    };

    std::vector<crispasr_segment> out;
    out.reserve(segs.size());
    for (auto& seg : segs) {
        if (seg.words.empty()) {
            out.push_back(std::move(seg));
            continue;
        }
        std::vector<int> word_spk(seg.words.size(), -1);
        int last_known = -1;
        for (size_t i = 0; i < seg.words.size(); i++) {
            const auto& w = seg.words[i];
            int spk = (w.t1 > w.t0) ? speaker_at(w.t0, w.t1) : -1;
            if (spk < 0)
                spk = last_known; // keep unaligned words attached to their neighbour
            word_spk[i] = spk;
            if (spk >= 0)
                last_known = spk;
        }
        for (size_t i = 0; i < word_spk.size() && word_spk[i] < 0; i++)
            for (size_t j = i + 1; j < word_spk.size(); j++)
                if (word_spk[j] >= 0) {
                    word_spk[i] = word_spk[j];
                    break;
                }

        constexpr int64_t MIN_RUN_CS = 50;
        std::vector<int64_t> word_t0(seg.words.size()), word_t1(seg.words.size());
        for (size_t i = 0; i < seg.words.size(); i++) {
            word_t0[i] = seg.words[i].t0 > 0 ? seg.words[i].t0 : seg.t0;
            word_t1[i] = seg.words[i].t1 > 0 ? seg.words[i].t1 : seg.t1;
        }
        const auto runs =
            crispasr_diarize_internal::group_words_into_speaker_runs(word_spk, word_t0, word_t1, MIN_RUN_CS);

        int first_spk = -1;
        bool multi = false;
        for (const auto& r : runs) {
            if (r.speaker < 0)
                continue;
            if (first_spk < 0)
                first_spk = r.speaker;
            else if (r.speaker != first_spk) {
                multi = true;
                break;
            }
        }
        if (!multi) {
            out.push_back(std::move(seg));
            continue;
        }

        for (size_t ri = 0; ri < runs.size(); ri++) {
            const size_t rs = runs[ri].start, re = runs[ri].end;
            crispasr_segment sub;
            sub.t0 = seg.words[rs].t0 > 0 ? seg.words[rs].t0 : seg.t0;
            sub.t1 = seg.words[re - 1].t1 > 0 ? seg.words[re - 1].t1 : seg.t1;
            sub.speaker_turn_next = (ri + 1 < runs.size());
            sub.words.assign(seg.words.begin() + (long)rs, seg.words.begin() + (long)re);
            for (size_t j = rs; j < re; j++) {
                if (!sub.text.empty() && !sub.words[j - rs].text.empty())
                    sub.text += ' ';
                sub.text += seg.words[j].text;
            }
            sub.tokens.clear(); // no per-word alignment; dividing them would be arbitrary
            if (runs[ri].speaker >= 0)
                sub.speaker = "(speaker " + std::to_string(runs[ri].speaker) + ") ";
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

    const auto args = make_sherpa_args(bin, params, wav_path);
    if (!params.no_prints)
        fprintf(stderr, "crispasr[diarize]: %s\n", crispasr_cli_process::join_cmdline(args).c_str());

    const int timeout_sec = crispasr_cli_process::timeout_from_audio_samples("CRISPASR_SHERPA_TIMEOUT_SEC", n_samples);
    const auto run = crispasr_cli_process::run_capture_stdout(args, timeout_sec);
    if (run.timed_out) {
        fprintf(stderr, "crispasr[diarize]: sherpa global run timed out after %d s\n", timeout_sec);
        std::remove(wav_path.c_str());
        return false;
    }
    if (run.exit_code != 0) {
        fprintf(stderr, "crispasr[diarize]: sherpa global run failed with exit code %d\n", run.exit_code);
        std::remove(wav_path.c_str());
        return false;
    }

    std::istringstream lines(run.output);
    std::string line;
    while (std::getline(lines, line)) {
        SherpaSegment s;
        if (parse_sherpa_line(line, s))
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

void crispasr_diarize_merged_by_slice(
    std::vector<crispasr_segment>& segs, const std::vector<crispasr_audio_slice>& slices,
    const std::function<void(const crispasr_audio_slice&, std::vector<crispasr_segment>&)>& diarize_slice) {
    if (segs.empty() || slices.empty() || !diarize_slice)
        return;

    std::vector<crispasr_segment> out;
    out.reserve(segs.size());

    size_t seg_offset = 0;
    for (size_t i = 0; i < slices.size(); ++i) {
        // Segments belonging to this slice: everything up to the first one that
        // starts at or after the NEXT slice's t0. The last slice takes the rest.
        size_t seg_count = 0;
        for (size_t j = seg_offset; j < segs.size(); ++j) {
            if (i + 1 < slices.size() && segs[j].t0 >= slices[i + 1].t0_cs)
                break;
            seg_count++;
        }
        if (seg_count == 0)
            continue;

        std::vector<crispasr_segment> slice_segs(
            std::make_move_iterator(segs.begin() + (ptrdiff_t)seg_offset),
            std::make_move_iterator(segs.begin() + (ptrdiff_t)(seg_offset + seg_count)));
        seg_offset += seg_count;

        diarize_slice(slices[i], slice_segs);

        // Append whatever came back — MORE than went in when the diarizer split
        // a segment at a speaker turn, fewer if it dropped one. Neither case can
        // lose a segment or index past the end here, which the old in-place
        // copy-back of `seg_count` elements did both of (#324).
        for (auto& s : slice_segs)
            out.push_back(std::move(s));
    }

    // Anything the slice walk never claimed survives unlabelled rather than
    // being dropped. The last slice takes the remainder, so this is normally
    // empty — it keeps "never lose a segment" a property of this function
    // instead of a consequence of that special case.
    for (; seg_offset < segs.size(); ++seg_offset)
        out.push_back(std::move(segs[seg_offset]));

    segs = std::move(out);
}

bool crispasr_apply_foxnose_global(std::vector<crispasr_segment>& all_segs, const std::vector<float>& samples,
                                   const whisper_params& params) {
    if (!params.diarize || !params.diarize_embedder_is_foxnose() || all_segs.empty() || samples.empty())
        return false;
    if (params.diarize_embedder.empty()) {
        fprintf(stderr, "crispasr[diarize]: foxnose needs --diarize-embedder <wespeaker.gguf>\n");
        return false;
    }

    CrispasrDiarizeOptions opts;
    opts.method = CrispasrDiarizeMethod::FoxNose;
    opts.n_threads = params.n_threads;
    opts.slice_t0_cs = 0; // all_segs timestamps are absolute
    opts.foxnose_embedder_path = resolve_foxnose_embedder(params);
    opts.max_speakers = params.diarize_max_speakers_explicit ? params.diarize_max_speakers : kFoxnoseDefaultMaxSpeakers;
    opts.num_speakers = params.diarize_num_speakers;

    auto lib_segs = lib_view(all_segs);
    std::vector<CrispasrDiarizeTurn> turns;
    const float* pcm = samples.data();
    if (!crispasr_diarize_segments(pcm, pcm, (int)samples.size(), /*is_stereo=*/false, lib_segs, opts, &turns))
        return false;
    apply_int_speakers_to_crispasr_segments(lib_segs, all_segs);
    split_segments_on_foxnose_turns(all_segs, turns, /*slice_t0_cs=*/0);
    if (!params.no_prints)
        fprintf(stderr, "crispasr[diarize]: foxnose global pass — %zu turn(s) over %.1f s\n", turns.size(),
                samples.size() / 16000.0);
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
    } else if (method == "foxnose" || method == "foxnose-diarize") {
        // The unified runner diarizes foxnose GLOBALLY after transcription
        // (crispasr_apply_foxnose_global) so speaker identities are consistent
        // across slices. Doing it per slice as well would reload the embedder
        // for every slice and produce labels the global pass then overwrites.
        if (params.diarize_foxnose_global)
            return true;
        lib_method = CrispasrDiarizeMethod::FoxNose;
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
        if (lib_method == CrispasrDiarizeMethod::FoxNose) {
            // Reuses the existing --diarize-embedder / --diarize-max-speakers
            // knobs rather than inventing parallel ones.
            opts.foxnose_embedder_path = resolve_foxnose_embedder(params);
            // MEASURED (docs/foxnose-diarize/PLAN.md): silhouette saturates
            // and climbs monotonically to the ceiling on real speaker
            // embeddings, so a loose bound is actively harmful — on
            // samples/multispeaker.wav a bound of 8 yields 8 speakers with
            // heavy flicker while a bound of 4 yields the correct 2. Upstream
            // defaults to 20. Default conservatively; an explicit
            // --diarize-max-speakers always wins.
            opts.max_speakers =
                params.diarize_max_speakers_explicit ? params.diarize_max_speakers : kFoxnoseDefaultMaxSpeakers;
            opts.num_speakers = params.diarize_num_speakers;
        }

        auto lib_segs = lib_view(segs);
        const int n = (int)left.size();
        const float* l = left.data();
        const float* r = (is_stereo && !right.empty()) ? right.data() : l;
        std::vector<CrispasrDiarizeTurn> foxnose_turns;
        if (!crispasr_diarize_segments(l, r, n, is_stereo, lib_segs, opts, &foxnose_turns)) {
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
        // #324 phase 2: FoxNose derives real speaker TURNS from the audio, so
        // a caller segment spanning several speakers can be split at word-
        // aligned boundaries instead of collapsing to one label. Segments
        // without word timestamps keep their segment-level label.
        if (lib_method == CrispasrDiarizeMethod::FoxNose)
            split_segments_on_foxnose_turns(segs, foxnose_turns, slice_t0_cs);
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

// Embed every segment long enough for a reliable speaker embedding
// (~250 ms+; shorter clips give noisy vectors on TitaNet and similar).
// Fills `embed_idx` with the segs indices that produced an embedding
// and `embeddings` with embed_idx.size()*dim floats, row-major.
// Number of segments to embed concurrently (#326).
//
// Once pyannote segmentation was chunked, the embedder became the dominant cost
// of diarization — one forward per segment, strictly sequentially. It cannot be
// sped up with `-t`: on TitaNet those per-segment graphs do not engage ggml's
// threads at all (measured on an M1, user CPU within 3% of wall from -t 1 to
// -t 4, at both 2 s and 10 s segments), so running segments concurrently is the
// only parallelism this path has.
//
// Each worker needs its own model instance, because a ggml backend is not safe
// for concurrent use. That costs memory — TitaNet-Large is ~45 MB — which is
// why this is capped rather than opened up to every core on a 24-core box.
// Measured speedup on an M1 (8 cores, 4 of them performance): 1.40x at 2,
// 2.11x at 4, 2.34x at 6, 2.54x at 8 — it tracks the performance cores and
// then flattens, so a big cap would buy memory rather than throughput.
static int crispasr_embed_workers(size_t n_candidates) {
    if (const char* e = std::getenv("CRISPASR_SPEAKER_EMBED_WORKERS")) {
        const int v = std::atoi(e);
        if (v > 0)
            return v;
        if (v == 0)
            return 1; // explicit opt-out
    }
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0)
        hw = 4;
    int k = (int)std::min<unsigned>(hw, 4);
    // Never spin up more workers than there is work, and never pay the extra
    // model loads for a handful of segments.
    if (n_candidates < 8)
        return 1;
    return std::max(1, std::min<int>(k, (int)(n_candidates / 2)));
}

static void crispasr_embed_segments(const std::vector<crispasr_segment>& segs, const float* full_audio, int n_samples,
                                    CrispasrSpeakerEmbedder* embedder, int d, std::vector<size_t>& embed_idx,
                                    std::vector<float>& embeddings) {
    constexpr int64_t MIN_EMBED_CS = 25; // 0.25 s
    // This stage is the dominant diarization cost on files with many segments
    // (#326), and its share swings enormously with segment count — it is ~3% of
    // a 600 s single-speaker run and was ~60% of the 2888 s file in that report.
    // Without a per-stage number, "is the embedder slow here?" is unanswerable
    // from a total runtime, and any speedup claim is unfalsifiable.
    const auto embed_t0 = std::chrono::steady_clock::now();

    // Decide which segments are embeddable first, so the workers below split a
    // known list and the output order does not depend on thread scheduling.
    struct Job {
        size_t seg_i;
        int64_t s0, s1;
    };
    std::vector<Job> jobs;
    jobs.reserve(segs.size());
    for (size_t i = 0; i < segs.size(); i++) {
        const auto& seg = segs[i];
        if ((seg.t1 - seg.t0) < MIN_EMBED_CS)
            continue;
        // Convert cs (1/100 s) → 16 kHz samples. full_audio is mono 16k.
        const int64_t s0 = std::max<int64_t>(0, seg.t0 * 160);
        const int64_t s1 = std::min<int64_t>(n_samples, seg.t1 * 160);
        if (s1 - s0 < 4000) // <250 ms after clamping
            continue;
        jobs.push_back({i, s0, s1});
    }

    embed_idx.reserve(jobs.size());
    embeddings.reserve(jobs.size() * (size_t)d);

    // One embedder per worker; worker 0 reuses the caller's. If cloning is not
    // supported, or only one worker is wanted, this stays exactly the old loop.
    std::vector<std::unique_ptr<CrispasrSpeakerEmbedder>> owned;
    std::vector<CrispasrSpeakerEmbedder*> workers{embedder};
    const int want = crispasr_embed_workers(jobs.size());
    for (int k = 1; k < want; k++) {
        auto c = embedder->clone();
        if (!c)
            break; // adapter cannot clone — run with what we have
        workers.push_back(c.get());
        owned.push_back(std::move(c));
    }

    // Results are written into per-job slots, so the output is identical
    // whatever order the workers finish in.
    std::vector<std::vector<float>> out(jobs.size());
    std::atomic<size_t> next{0};
    auto run = [&](CrispasrSpeakerEmbedder* emb) {
        std::vector<float> tmp(d);
        for (;;) {
            const size_t j = next.fetch_add(1);
            if (j >= jobs.size())
                return;
            const Job& job = jobs[j];
            if (emb->embed(full_audio + job.s0, (int)(job.s1 - job.s0), tmp.data()))
                out[j].assign(tmp.begin(), tmp.end());
        }
    };

    if (workers.size() == 1) {
        run(workers[0]);
    } else {
        std::vector<std::thread> th;
        th.reserve(workers.size() - 1);
        for (size_t k = 1; k < workers.size(); k++)
            th.emplace_back(run, workers[k]);
        run(workers[0]);
        for (auto& t : th)
            t.join();
    }

    // Collect in job order. A segment whose embed() failed is skipped, exactly
    // as the sequential version skipped it.
    for (size_t j = 0; j < jobs.size(); j++) {
        if (out[j].empty())
            continue;
        embed_idx.push_back(jobs[j].seg_i);
        embeddings.insert(embeddings.end(), out[j].begin(), out[j].end());
    }

    if (std::getenv("CRISPASR_DIARIZE_DEBUG")) {
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - embed_t0).count();
        fprintf(stderr, "crispasr[diarize]: embed %zu segments in %.0f ms (%.1f ms/seg, %zu worker%s)\n", jobs.size(),
                ms, jobs.empty() ? 0.0 : ms / (double)jobs.size(), workers.size(), workers.size() == 1 ? "" : "s");
    }
}

void crispasr_remap_speakers_via_embeddings(std::vector<crispasr_segment>& segs, const float* full_audio, int n_samples,
                                            CrispasrSpeakerEmbedder* embedder, const whisper_params& params,
                                            CrispasrClusterEmbeddings* out_clusters) {
    if (!embedder || segs.empty() || !full_audio || n_samples <= 0)
        return;

    const int d = embedder->dim();
    if (d <= 0)
        return;

    std::vector<size_t> embed_idx;
    std::vector<float> embeddings;
    crispasr_embed_segments(segs, full_audio, n_samples, embedder, d, embed_idx, embeddings);

    if (embed_idx.size() < 2) {
        // Nothing to cluster — either zero or one usable segment.
        if (embed_idx.size() == 1) {
            // Force the one embeddable segment to (speaker 0) so the
            // global label exists even on single-speaker inputs.
            segs[embed_idx[0]].speaker = "(speaker 0) ";
            if (out_clusters) {
                out_clusters->seg_idx = std::move(embed_idx);
                out_clusters->embeddings = std::move(embeddings);
                out_clusters->labels = {0};
                out_clusters->dim = d;
                out_clusters->n_clusters = 1;
            }
        }
        return;
    }

    const float thr = params.diarize_cluster_threshold;
    const int max_spk = params.diarize_max_speakers > 0 ? params.diarize_max_speakers : 8;
    const int n_emb = (int)embed_idx.size();

    // #326: estimate the speaker COUNT instead of letting the cap decide it.
    //
    // This used to be single-linkage agglomerative with a fixed 0.5 cosine
    // merge threshold and a hard max_speakers cap. Single linkage chains, and
    // a fixed threshold does not adapt to the embedder's spread on a given
    // recording, so on real audio the merge loop never got below the cap and
    // the CAP became the answer. On the VoxConverse dev shard it pinned to
    // --diarize-max-speakers 8 on 4 of 8 files (esrit 8 hypothesised vs 5
    // real, mesob 8 vs 4, nnqfq 8 vs 5, fsaal 8 vs 7), and mesob alone scored
    // 33.03% DER.
    //
    // core_spectral::cluster_speakers (#324) estimates the count first — PCA
    // + full-covariance GMM/BIC, refined on silhouette — then runs
    // Ng-Jordan-Weiss spectral clustering and a spherical refinement. It is
    // the same clusterer that gets --diarize-method foxnose to 7.32% on these
    // files, so this is reuse of validated in-tree code, not a new heuristic.
    //
    // --diarize-cluster-threshold still works, but only when the caller
    // actually passes it: the threshold is meaningless to the spectral path,
    // so honouring its DEFAULT would just reinstate the bug.
    std::vector<int> labels;
    // The pyannote local tracks already on the segments are a lower
    // bound on the true speaker count. They come from a single forward pass
    // (the #107 full-audio cache), so within one pass the track indices are
    // globally consistent — distinct tracks are distinct speakers. The
    // GMM/BIC estimator can collapse to k=1 on short inputs (few segments,
    // near-duplicate embeddings), which would silently merge distinct
    // speakers into one label; clamp min_speakers to at least the number of
    // distinct local tracks seen on the embeddable segments.
    int min_spk = 1;
    std::set<int> local_tracks;
    for (size_t k = 0; k < embed_idx.size(); k++) {
        const std::string& sp = segs[embed_idx[k]].speaker;
        if (sp.rfind("(speaker ", 0) == 0) {
            const int n = std::atoi(sp.c_str() + 9);
            if (n >= 0)
                local_tracks.insert(n);
        }
    }
    // Distinct local tracks, not max+1: the track indices within one
    // forward pass are contiguous labels, but only the COUNT of distinct
    // tracks is a lower bound on the speaker count.
    min_spk = std::max(min_spk, (int)local_tracks.size());
    if (params.diarize_cluster_threshold_explicit) {
        labels = crispasr_agglomerative_cluster(embeddings, n_emb, d, thr, max_spk);
    } else {
        core_spectral::SpeakerEstimate est;
        labels = core_spectral::cluster_speakers(embeddings.data(), n_emb, d, /*min_speakers=*/min_spk, max_spk,
                                                 /*num_speakers=*/params.diarize_num_speakers, &est);
        if (std::getenv("CRISPASR_DIARIZE_DEBUG"))
            fprintf(stderr, "crispasr[diarize]: n_emb=%d dim=%d -> k=%d (min_spk=%d, %s, cos_p10=%.4f, pca=%d)\n",
                    n_emb, d, est.best_k, min_spk, est.reason, est.cosine_sim_p10, est.pca_dim);
    }

    // Rewrite segment speakers from clustering output. Segments that
    // couldn't be embedded (too short) keep their existing pyannote-
    // local label as a best-effort fallback.
    int n_clusters = 0;
    for (size_t k = 0; k < embed_idx.size(); k++) {
        const int spk = labels[k];
        if (spk < 0)
            continue;
        n_clusters = std::max(n_clusters, spk + 1);
        segs[embed_idx[k]].speaker = "(speaker " + std::to_string(spk) + ") ";
    }

    if (out_clusters) {
        out_clusters->seg_idx = std::move(embed_idx);
        out_clusters->embeddings = std::move(embeddings);
        out_clusters->labels = std::move(labels);
        out_clusters->dim = d;
        out_clusters->n_clusters = n_clusters;
    }
}

int crispasr_identify_speaker_clusters(std::vector<crispasr_segment>& segs, const CrispasrClusterEmbeddings& ce,
                                       const struct speaker_db* db, float threshold, bool no_prints) {
    if (!ce.valid() || !db || speaker_db_count(db) <= 0)
        return 0;

    const auto centroids = crispasr_cluster_centroids(ce.embeddings, ce.labels, (int)ce.seg_idx.size(), ce.dim);
    if ((int)centroids.size() < ce.n_clusters * ce.dim)
        return 0;

    int n_named = 0;
    for (int c = 0; c < ce.n_clusters; c++) {
        float score = 0.0f;
        const char* name = speaker_db_match(db, centroids.data() + (size_t)c * ce.dim, ce.dim, threshold, &score);
        if (!name) {
            if (!no_prints)
                fprintf(stderr, "crispasr: speaker-db: cluster %d -> unmatched, keeps (speaker %d) (best cos %.2f)\n",
                        c, c, score);
            continue;
        }
        const std::string label = std::string("(") + name + ") ";
        for (size_t k = 0; k < ce.seg_idx.size(); k++) {
            if (ce.labels[k] == c)
                segs[ce.seg_idx[k]].speaker = label;
        }
        if (!no_prints)
            fprintf(stderr, "crispasr: speaker-db: cluster %d -> '%s' (cos %.2f)\n", c, name, score);
        n_named++;
    }
    return n_named;
}

bool crispasr_identify_single_speaker(std::vector<crispasr_segment>& segs, const float* full_audio, int n_samples,
                                      CrispasrSpeakerEmbedder* embedder, const struct speaker_db* db, float threshold,
                                      bool no_prints) {
    if (!embedder || segs.empty() || !full_audio || n_samples <= 0 || !db || speaker_db_count(db) <= 0)
        return false;

    const int d = embedder->dim();
    if (d <= 0)
        return false;

    std::vector<size_t> embed_idx;
    std::vector<float> embeddings;
    crispasr_embed_segments(segs, full_audio, n_samples, embedder, d, embed_idx, embeddings);
    if (embed_idx.empty())
        return false;

    const std::vector<int> labels(embed_idx.size(), 0);
    const auto centroid = crispasr_cluster_centroids(embeddings, labels, (int)embed_idx.size(), d);
    if ((int)centroid.size() < d)
        return false;

    float score = 0.0f;
    const char* name = speaker_db_match(db, centroid.data(), d, threshold, &score);
    if (!name) {
        if (!no_prints)
            fprintf(stderr, "crispasr: speaker-db: recording unmatched, labels stay anonymous (best cos %.2f)\n",
                    score);
        return false;
    }

    // No diarization ran, so the recording is asserted single-speaker:
    // the one matched identity labels every segment, short ones included.
    const std::string label = std::string("(") + name + ") ";
    for (auto& seg : segs)
        seg.speaker = label;
    if (!no_prints)
        fprintf(stderr, "crispasr: speaker-db: recording -> '%s' (cos %.2f)\n", name, score);
    return true;
}
