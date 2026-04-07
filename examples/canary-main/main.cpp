// canary-main — CLI driver for nvidia/canary-1b-v2.
//
// Mirrors parakeet-main's flag set: VAD-based segmentation, long-audio
// chunking fallback, SRT/VTT/TXT output. Adds canary-specific flags
// for explicit source/target language (no auto-detect).
//
// ASR mode:        -sl en -tl en (or any same-language pair)
// Translation:     -sl de -tl en (German speech → English text), etc.

#include "canary.h"
#include "common-whisper.h"
#include "whisper.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

struct canary_params {
    std::string model;
    std::string fname_inp;
    std::string source_lang     = "en";
    std::string target_lang     = "en";
    int         n_threads       = std::min(4, (int)std::thread::hardware_concurrency());
    int         verbosity       = 1;
    bool        punctuation     = true;
    bool        output_txt      = false;
    bool        output_srt      = false;
    bool        output_vtt      = false;
    int         max_len         = 0;
    int         chunk_seconds   = 30;
    std::string vad_model;
    float       vad_thold       = 0.5f;
    int         vad_min_speech_ms  = 250;
    int         vad_min_silence_ms = 100;
    float       vad_speech_pad_ms  = 30.0f;
};

static void print_usage(const char * prog) {
    fprintf(stderr,
        "\nusage: %s [options] -m MODEL.gguf -f AUDIO.wav -sl LANG [-tl LANG]\n\n"
        "options:\n"
        "  -h,  --help              show this help\n"
        "  -m   FNAME               canary GGUF model (from convert-canary-to-gguf.py)\n"
        "  -f   FNAME               input audio (16 kHz mono WAV)\n"
        "  -sl  LANG                source language (REQUIRED, ISO-639-1: en, de, fr, ...)\n"
        "  -tl  LANG                target language (default: same as -sl → ASR; differs → translation)\n"
        "  -t   N                   threads (default: %d)\n"
        "  -ot, --output-txt        write transcript to <audio>.txt\n"
        "  -osrt, --output-srt      write .srt subtitle file\n"
        "  -ovtt, --output-vtt      write .vtt subtitle file\n"
        "  -ml  N                   max chars per output segment (0=unlimited, 1=per-token)\n"
        "  -ck  N                   long-audio chunk size in seconds when no VAD (default: 30)\n"
        "  -vad-model FNAME         Silero VAD model (recommended for long audio)\n"
        "  -vad-thold F             VAD threshold (default: 0.5)\n"
        "  -npnc, --no-punctuation  disable punctuation in the output\n"
        "  -v,  --verbose           dump per-token decoder steps for debugging\n"
        "  -np, --no-prints         suppress informational output\n\n"
        "supported languages (25):\n"
        "  bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk\n\n"
        "modes:\n"
        "  ASR:                  -sl LANG -tl LANG    (same language)\n"
        "  Speech translation:   -sl SRC -tl TGT      (different languages)\n\n",
        prog,
        std::min(4, (int)std::thread::hardware_concurrency()));
}

static bool parse_args(int argc, char ** argv, canary_params & p) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "-h" || a == "--help")          { print_usage(argv[0]); exit(0); }
        else if ((a == "-m")  && i+1 < argc)            p.model         = argv[++i];
        else if ((a == "-f")  && i+1 < argc)            p.fname_inp     = argv[++i];
        else if ((a == "-sl") && i+1 < argc)            p.source_lang   = argv[++i];
        else if ((a == "-tl") && i+1 < argc)            p.target_lang   = argv[++i];
        else if ((a == "-t")  && i+1 < argc)            p.n_threads     = std::atoi(argv[++i]);
        else if ((a == "-ml") && i+1 < argc)            p.max_len       = std::atoi(argv[++i]);
        else if ((a == "-ck") && i+1 < argc)            p.chunk_seconds = std::atoi(argv[++i]);
        else if (a == "-vad-model" && i+1 < argc)       p.vad_model     = argv[++i];
        else if (a == "-vad-thold" && i+1 < argc)       p.vad_thold     = std::atof(argv[++i]);
        else if (a == "-ot"   || a == "--output-txt")   p.output_txt    = true;
        else if (a == "-osrt" || a == "--output-srt")   p.output_srt    = true;
        else if (a == "-ovtt" || a == "--output-vtt")   p.output_vtt    = true;
        else if (a == "-npnc" || a == "--no-punctuation") p.punctuation = false;
        else if (a == "-v"    || a == "--verbose")      p.verbosity     = 2;
        else if (a == "-np"   || a == "--no-prints")    p.verbosity     = 0;
        else {
            fprintf(stderr, "error: unknown option '%s'\n\n", a.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    if (p.model.empty() || p.fname_inp.empty()) {
        fprintf(stderr, "error: -m MODEL and -f AUDIO are required\n\n");
        print_usage(argv[0]);
        return false;
    }
    if (p.target_lang == "en" && p.source_lang != "en") {
        // explicit translation mode is fine
    }
    return true;
}

// ---------------------------------------------------------------------------
// Display segments
// ---------------------------------------------------------------------------

struct disp_segment {
    int64_t     t0, t1;
    std::string text;
};

static std::vector<disp_segment> make_disp_segments(
        const canary_result * r,
        int64_t t0_seg, int64_t t1_seg,
        int max_len)
{
    std::vector<disp_segment> out;
    if (!r) return out;

    if (r->n_words == 0) {
        if (r->text && r->text[0]) {
            out.push_back({t0_seg, t1_seg, r->text});
        }
        return out;
    }

    if (max_len == 1) {
        for (int i = 0; i < r->n_words; i++) {
            const auto & w = r->words[i];
            out.push_back({w.t0, w.t1, w.text});
        }
        return out;
    }

    disp_segment cur;
    cur.t0 = -1;
    auto flush = [&]() {
        if (!cur.text.empty()) out.push_back(cur);
        cur = {};
        cur.t0 = -1;
    };

    for (int i = 0; i < r->n_words; i++) {
        const auto & w = r->words[i];
        if (cur.t0 < 0) cur.t0 = w.t0;
        cur.t1 = w.t1;

        const std::string sep = cur.text.empty() ? "" : " ";
        const bool would_overflow =
            max_len > 0 && !cur.text.empty() &&
            (int)(cur.text.size() + sep.size() + strlen(w.text)) > max_len;

        if (would_overflow) {
            flush();
            cur.t0 = w.t0;
            cur.t1 = w.t1;
        }
        cur.text += sep + w.text;
    }
    flush();
    return out;
}

// ---------------------------------------------------------------------------
// Output writers
// ---------------------------------------------------------------------------

static std::string make_out_path(const std::string & audio, const std::string & ext) {
    std::string base = audio;
    for (const char * e : {".wav", ".WAV", ".mp3", ".MP3", ".flac", ".FLAC",
                            ".ogg", ".OGG", ".m4a", ".M4A", ".opus", ".OPUS"}) {
        size_t el = strlen(e);
        if (base.size() > el && base.compare(base.size()-el, el, e) == 0) {
            base = base.substr(0, base.size()-el);
            break;
        }
    }
    return base + ext;
}

static void write_srt(const std::string & path,
                      const std::vector<disp_segment> & segs,
                      int verbosity, const char * prog)
{
    std::ofstream f(path);
    if (!f) { fprintf(stderr, "%s: warning: cannot write SRT '%s'\n", prog, path.c_str()); return; }
    for (int i = 0; i < (int)segs.size(); i++) {
        f << (i + 1) << "\n"
          << to_timestamp(segs[i].t0, /*comma=*/true)
          << " --> "
          << to_timestamp(segs[i].t1, /*comma=*/true) << "\n"
          << segs[i].text << "\n\n";
    }
    if (verbosity >= 1) fprintf(stderr, "%s: SRT written to '%s'\n", prog, path.c_str());
}

static void write_vtt(const std::string & path,
                      const std::vector<disp_segment> & segs,
                      int verbosity, const char * prog)
{
    std::ofstream f(path);
    if (!f) { fprintf(stderr, "%s: warning: cannot write VTT '%s'\n", prog, path.c_str()); return; }
    f << "WEBVTT\n\n";
    for (const auto & s : segs)
        f << to_timestamp(s.t0) << " --> " << to_timestamp(s.t1) << "\n" << s.text << "\n\n";
    if (verbosity >= 1) fprintf(stderr, "%s: VTT written to '%s'\n", prog, path.c_str());
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    canary_params p;
    if (!parse_args(argc, argv, p)) return 1;

    canary_context_params cp = canary_context_default_params();
    cp.n_threads = p.n_threads;
    cp.verbosity = p.verbosity;

    if (p.verbosity >= 1)
        fprintf(stderr, "%s: loading '%s'\n", argv[0], p.model.c_str());

    canary_context * ctx = canary_init_from_file(p.model.c_str(), cp);
    if (!ctx) {
        fprintf(stderr, "%s: failed to load model '%s'\n", argv[0], p.model.c_str());
        return 1;
    }

    // ---- load audio ----
    std::vector<float> samples;
    std::vector<std::vector<float>> stereo;
    if (!read_audio_data(p.fname_inp, samples, stereo, /*stereo=*/false)) {
        fprintf(stderr, "%s: failed to read audio '%s'\n", argv[0], p.fname_inp.c_str());
        canary_free(ctx);
        return 2;
    }

    const int SR = canary_sample_rate(ctx);
    if (p.verbosity >= 1) {
        fprintf(stderr, "%s: audio: %d samples (%.1f s) @ %d Hz, %d threads\n",
                argv[0], (int)samples.size(),
                (double)samples.size() / SR, SR, p.n_threads);
        fprintf(stderr, "%s: mode: %s  src=%s  tgt=%s\n",
                argv[0],
                p.source_lang == p.target_lang ? "ASR" : "speech translation",
                p.source_lang.c_str(), p.target_lang.c_str());
    }

    // ---- build audio slices ----
    struct audio_slice {
        int     start, end;
        int64_t t0_cs, t1_cs;
    };
    std::vector<audio_slice> slices;

    whisper_vad_context * vctx = nullptr;
    if (!p.vad_model.empty()) {
        whisper_vad_context_params vcp = whisper_vad_default_context_params();
        vcp.n_threads = p.n_threads;
        vctx = whisper_vad_init_from_file_with_params(p.vad_model.c_str(), vcp);
        if (!vctx)
            fprintf(stderr, "%s: warning: VAD load failed, using fixed chunking\n", argv[0]);
    }

    if (vctx) {
        whisper_vad_params vp = whisper_vad_default_params();
        vp.threshold               = p.vad_thold;
        vp.min_speech_duration_ms  = p.vad_min_speech_ms;
        vp.min_silence_duration_ms = p.vad_min_silence_ms;
        vp.speech_pad_ms           = p.vad_speech_pad_ms;

        whisper_vad_segments * vseg =
            whisper_vad_segments_from_samples(vctx, vp, samples.data(), (int)samples.size());

        int nv = vseg ? whisper_vad_segments_n_segments(vseg) : 0;
        if (nv == 0) {
            fprintf(stderr, "%s: VAD found no speech\n", argv[0]);
            if (vseg) whisper_vad_free_segments(vseg);
            whisper_vad_free(vctx);
            canary_free(ctx);
            return 3;
        }
        if (p.verbosity >= 1)
            fprintf(stderr, "%s: VAD detected %d speech segment(s)\n", argv[0], nv);

        for (int i = 0; i < nv; i++) {
            float t0s = whisper_vad_segments_get_segment_t0(vseg, i);
            float t1s = whisper_vad_segments_get_segment_t1(vseg, i);
            int s = std::max(0, (int)(t0s * SR));
            int e = std::min((int)samples.size(), (int)(t1s * SR));
            if (e > s)
                slices.push_back({s, e, (int64_t)(t0s*100.f), (int64_t)(t1s*100.f)});
        }
        whisper_vad_free_segments(vseg);
        whisper_vad_free(vctx);
    } else {
        const int chunk_samples = p.chunk_seconds * SR;
        if ((int)samples.size() <= chunk_samples) {
            int64_t dur = (int64_t)((double)samples.size() / SR * 100.0);
            slices.push_back({0, (int)samples.size(), 0, dur});
        } else {
            for (int s = 0; s < (int)samples.size(); s += chunk_samples) {
                int e = std::min((int)samples.size(), s + chunk_samples);
                slices.push_back({
                    s, e,
                    (int64_t)((double)s / SR * 100.0),
                    (int64_t)((double)e / SR * 100.0)});
            }
            if (p.verbosity >= 1)
                fprintf(stderr, "%s: long audio chunked into %d × %d s windows\n",
                        argv[0], (int)slices.size(), p.chunk_seconds);
        }
    }

    // ---- transcribe each slice ----
    std::vector<disp_segment> all_segs;
    std::string plain_text;

    for (const auto & sl : slices) {
        canary_result * r = canary_transcribe_ex(
            ctx,
            samples.data() + sl.start,
            sl.end - sl.start,
            p.source_lang.c_str(),
            p.target_lang.c_str(),
            p.punctuation,
            sl.t0_cs);

        if (!r) {
            fprintf(stderr, "%s: transcription failed for [%s --> %s]\n",
                    argv[0],
                    to_timestamp(sl.t0_cs).c_str(),
                    to_timestamp(sl.t1_cs).c_str());
            continue;
        }

        auto segs = make_disp_segments(r, sl.t0_cs, sl.t1_cs, p.max_len);
        all_segs.insert(all_segs.end(), segs.begin(), segs.end());

        if (r->text && r->text[0]) {
            if (!plain_text.empty()) plain_text += " ";
            plain_text += r->text;
        }

        canary_result_free(r);
    }

    // ---- output ----
    const bool show_timestamps = p.output_srt || p.output_vtt || p.max_len > 0 || p.verbosity >= 2;

    if (show_timestamps) {
        for (const auto & s : all_segs) {
            printf("[%s --> %s]  %s\n",
                to_timestamp(s.t0).c_str(),
                to_timestamp(s.t1).c_str(),
                s.text.c_str());
        }
    } else {
        printf("%s\n", plain_text.c_str());
    }

    if (p.output_srt)
        write_srt(make_out_path(p.fname_inp, ".srt"), all_segs, p.verbosity, argv[0]);
    if (p.output_vtt)
        write_vtt(make_out_path(p.fname_inp, ".vtt"), all_segs, p.verbosity, argv[0]);

    if (p.output_txt) {
        const std::string tp = make_out_path(p.fname_inp, ".txt");
        std::ofstream f(tp);
        if (f) {
            f << plain_text << "\n";
            if (p.verbosity >= 1) fprintf(stderr, "%s: text written to '%s'\n", argv[0], tp.c_str());
        }
    }

    if (p.verbosity >= 1) fprintf(stderr, "\n%s: done\n", argv[0]);

    canary_free(ctx);
    return 0;
}
