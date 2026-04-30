/**
 * cohere-align — Word-level timestamps via Cohere Transcribe + CTC forced alignment.
 *
 * Pipeline
 * --------
 *  1. Transcribe audio with the Cohere Transcribe model (cross-attention DTW
 *     timestamps used as initial fallback).
 *  2. Run the same audio through a wav2vec2 CTC model to obtain per-frame logits.
 *  3. CTC forced alignment (Viterbi) refines the timestamps to ~30-50 ms accuracy,
 *     compared to ~360 ms for cross-attention DTW.
 *
 * Usage
 * -----
 *  cohere-align -m cohere.gguf -cw wav2vec2-xlsr.gguf -f audio.wav [-l en]
 *               [-t 4] [-osrt] [-ovtt] [--no-ctc] [--verbose]
 *
 * Model download
 * --------------
 *  Cohere model:
 *    See models/download-ggml-model.sh
 *
 *  wav2vec2 CTC model (covers all 14 Cohere languages via xlsr-53):
 *    pip install gguf transformers torch
 *    python models/convert-wav2vec2-to-gguf.py \
 *        --model-dir jonatasgrosman/wav2vec2-large-xlsr-53-english \
 *        --output    models/wav2vec2-xlsr-en.gguf
 *
 *  Per-language models give better accuracy when available.
 */

#include "cohere.h"
#include "wav2vec2-ggml.h"
#include "align.h"
#include "common.h"
#include "common-crispasr.h"
#include "ggml.h"
#include "crispasr.h" // for Silero VAD API

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
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

struct align_params {
    std::string model_cohere;
    std::string model_ctc;
    std::string fname_inp;
    std::string language = "en";
    int n_threads = std::min(4, (int)std::thread::hardware_concurrency());
    int verbosity = 1;
    bool use_flash = false;
    bool no_punctuation = false;
    bool no_ctc = false; // skip CTC, output cross-attention stamps only
    bool output_txt = false;
    bool output_srt = false;
    bool output_vtt = false;
    std::string vad_model;
    float vad_thold = 0.5f;
    int vad_min_speech_ms = 250;
    int vad_min_silence_ms = 100;
    float vad_speech_pad_ms = 30.f;
};

static void print_usage(const char* prog) {
    fprintf(stderr,
            "\nusage: %s [options] -m COHERE.gguf -cw CTC.gguf -f AUDIO.wav\n\n"
            "options:\n"
            "  -h,  --help              show this help\n"
            "  -m   FNAME               cohere-transcribe GGUF model\n"
            "  -cw  FNAME               wav2vec2 CTC GGUF model (convert-wav2vec2-to-gguf.py)\n"
            "  -f   FNAME               input audio (16 kHz mono WAV)\n"
            "  -l   LANG                language code, default: en\n"
            "  -t   N                   threads (default: %d)\n"
            "  -ot, --output-txt        write transcript to <audio>.txt\n"
            "  -osrt, --output-srt      write .srt subtitle file\n"
            "  -ovtt, --output-vtt      write .vtt subtitle file\n"
            "  -vad-model FNAME         Silero VAD model for audio segmentation\n"
            "  -vad-thold F             VAD threshold (default: 0.5)\n"
            "  --no-ctc                 skip CTC, use cross-attention timestamps only\n"
            "  -v,  --verbose           extra timing/debug output\n"
            "  -np, --no-prints         suppress all informational output\n"
            "  --flash                  enable flash attention in Cohere decoder\n"
            "  -npnc, --no-punctuation  disable punctuation in Cohere output\n\n"
            "CTC model: use convert-wav2vec2-to-gguf.py to convert any Wav2Vec2ForCTC\n"
            "model from HuggingFace. For 14-language coverage, use:\n"
            "  jonatasgrosman/wav2vec2-large-xlsr-53-<lang>  (e.g. -english, -french, ...)\n\n",
            prog, std::min(4, (int)std::thread::hardware_concurrency()));
}

static bool parse_args(int argc, char** argv, align_params& p) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "-m" && i + 1 < argc)
            p.model_cohere = argv[++i];
        else if (arg == "-cw" && i + 1 < argc)
            p.model_ctc = argv[++i];
        else if (arg == "-f" && i + 1 < argc)
            p.fname_inp = argv[++i];
        else if (arg == "-l" && i + 1 < argc)
            p.language = argv[++i];
        else if (arg == "-t" && i + 1 < argc)
            p.n_threads = std::atoi(argv[++i]);
        else if (arg == "-vad-model" && i + 1 < argc)
            p.vad_model = argv[++i];
        else if (arg == "-vad-thold" && i + 1 < argc)
            p.vad_thold = std::atof(argv[++i]);
        else if (arg == "-ot" || arg == "--output-txt")
            p.output_txt = true;
        else if (arg == "-osrt" || arg == "--output-srt")
            p.output_srt = true;
        else if (arg == "-ovtt" || arg == "--output-vtt")
            p.output_vtt = true;
        else if (arg == "--no-ctc")
            p.no_ctc = true;
        else if (arg == "-v" || arg == "--verbose")
            p.verbosity = 2;
        else if (arg == "-np" || arg == "--no-prints")
            p.verbosity = 0;
        else if (arg == "--flash")
            p.use_flash = true;
        else if (arg == "-npnc" || arg == "--no-punctuation")
            p.no_punctuation = true;
        else {
            fprintf(stderr, "error: unknown option '%s'\n\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    if (p.model_cohere.empty() || p.fname_inp.empty()) {
        fprintf(stderr, "error: -m COHERE.gguf and -f AUDIO are required\n\n");
        print_usage(argv[0]);
        return false;
    }
    if (!p.no_ctc && p.model_ctc.empty()) {
        fprintf(stderr, "error: -cw CTC.gguf required (or pass --no-ctc)\n\n");
        print_usage(argv[0]);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

static std::string make_out_path(const std::string& audio, const std::string& ext) {
    std::string base = audio;
    for (const char* e :
         {".wav", ".WAV", ".mp3", ".MP3", ".flac", ".FLAC", ".ogg", ".OGG", ".m4a", ".M4A", ".opus", ".OPUS"}) {
        size_t el = strlen(e);
        if (base.size() > el && base.compare(base.size() - el, el, e) == 0) {
            base = base.substr(0, base.size() - el);
            break;
        }
    }
    return base + ext;
}

// One output line: [t0 --> t1]  word
struct stamp_line {
    float t0, t1;
    std::string text;
};

static std::string fmt_ts(float sec) {
    int cs = (int)(sec * 100.f + 0.5f);
    return to_timestamp((int64_t)cs);
}

static void write_srt(const std::string& path, const std::vector<stamp_line>& lines, int v, const char* prog) {
    std::ofstream f(path);
    if (!f) {
        fprintf(stderr, "%s: warning: cannot write SRT '%s'\n", prog, path.c_str());
        return;
    }
    for (int i = 0; i < (int)lines.size(); i++) {
        f << (i + 1) << "\n"
          << to_timestamp((int64_t)(lines[i].t0 * 100.f), /*comma=*/true) << " --> "
          << to_timestamp((int64_t)(lines[i].t1 * 100.f), /*comma=*/true) << "\n"
          << lines[i].text << "\n\n";
    }
    if (v >= 1)
        fprintf(stderr, "%s: SRT written to '%s'\n", prog, path.c_str());
}

static void write_vtt(const std::string& path, const std::vector<stamp_line>& lines, int v, const char* prog) {
    std::ofstream f(path);
    if (!f) {
        fprintf(stderr, "%s: warning: cannot write VTT '%s'\n", prog, path.c_str());
        return;
    }
    f << "WEBVTT\n\n";
    for (const auto& l : lines)
        f << fmt_ts(l.t0) << " --> " << fmt_ts(l.t1) << "\n" << l.text << "\n\n";
    if (v >= 1)
        fprintf(stderr, "%s: VTT written to '%s'\n", prog, path.c_str());
}

// ---------------------------------------------------------------------------
// Word extraction from cohere_result
// ---------------------------------------------------------------------------

// Split cohere_result tokens into words (merge sub-word tokens).
// A token starting with ' ' is a new word; otherwise it continues the previous.
// Returns pairs of (word_text, {t0_cs, t1_cs}).
struct word_stamp {
    std::string text;
    int64_t t0_cs, t1_cs; // centiseconds
};

static std::vector<word_stamp> result_to_words(const cohere_result* r, int64_t seg_t0_cs, int64_t seg_t1_cs) {
    std::vector<word_stamp> out;
    if (!r || r->n_tokens == 0) {
        // No per-token data — return the full text as one "word"
        if (r && r->text && r->text[0])
            out.push_back({r->text, seg_t0_cs, seg_t1_cs});
        return out;
    }

    word_stamp cur;
    bool first = true;

    for (int i = 0; i < r->n_tokens; i++) {
        const cohere_token_data& td = r->tokens[i];
        if (!td.text[0])
            continue;

        bool is_word_start = (td.text[0] == ' ') || first;

        if (is_word_start && !cur.text.empty()) {
            // Strip leading space that was used as word-boundary marker
            if (cur.text[0] == ' ')
                cur.text = cur.text.substr(1);
            out.push_back(cur);
            cur = word_stamp{};
        }

        if (cur.text.empty()) {
            cur.t0_cs = td.t0;
            first = false;
        }
        cur.t1_cs = td.t1;
        cur.text += td.text;
    }
    if (!cur.text.empty()) {
        if (cur.text[0] == ' ')
            cur.text = cur.text.substr(1);
        out.push_back(cur);
    }
    return out;
}

// Plain word text (no leading space, lowercase, letters only) suitable for CTC.
static std::string word_for_ctc(const std::string& w) {
    std::string out;
    for (unsigned char c : w) {
        if (std::isalpha(c))
            out += (char)std::tolower(c);
        // Keep raw non-ASCII bytes; CTC vocab may contain UTF-8 fragments.
        else if (c > 127)
            out += (char)c;
    }
    return out;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    align_params p;
    if (!parse_args(argc, argv, p))
        return 1;

    // -----------------------------------------------------------------------
    // Load Cohere model
    // -----------------------------------------------------------------------
    struct cohere_context_params cparams = cohere_context_default_params();
    cparams.n_threads = p.n_threads;
    cparams.use_flash = p.use_flash;
    cparams.no_punctuation = p.no_punctuation;
    cparams.verbosity = p.verbosity;

    if (p.verbosity >= 1)
        fprintf(stderr, "%s: loading Cohere model '%s'\n", argv[0], p.model_cohere.c_str());

    struct cohere_context* cctx = cohere_init_from_file(p.model_cohere.c_str(), cparams);
    if (!cctx) {
        fprintf(stderr, "%s: failed to load Cohere model\n", argv[0]);
        return 1;
    }

    // -----------------------------------------------------------------------
    // Load wav2vec2 CTC model (optional)
    // -----------------------------------------------------------------------
    wav2vec2_model w2v_model;
    bool have_ctc = false;

    if (!p.no_ctc) {
        if (p.verbosity >= 1)
            fprintf(stderr, "%s: loading CTC model '%s'\n", argv[0], p.model_ctc.c_str());
        have_ctc = wav2vec2_load(p.model_ctc.c_str(), w2v_model);
        if (!have_ctc)
            fprintf(stderr, "%s: warning: CTC model load failed, falling back to cross-attention timestamps\n",
                    argv[0]);
    }

    // -----------------------------------------------------------------------
    // Load audio
    // -----------------------------------------------------------------------
    std::vector<float> samples;
    std::vector<std::vector<float>> samples_stereo;
    if (!read_audio_data(p.fname_inp, samples, samples_stereo, /*stereo=*/false)) {
        fprintf(stderr, "%s: failed to read audio '%s'\n", argv[0], p.fname_inp.c_str());
        cohere_free(cctx);
        return 1;
    }

    if (p.verbosity >= 1)
        fprintf(stderr, "%s: audio: %d samples  (%.1f s)  threads: %d\n", argv[0], (int)samples.size(),
                (float)samples.size() / 16000.f, p.n_threads);

    const int SR = 16000;

    // -----------------------------------------------------------------------
    // Build audio slices (VAD optional)
    // -----------------------------------------------------------------------
    struct audio_slice {
        int start, end;
        int64_t t0_cs, t1_cs;
    };
    std::vector<audio_slice> slices;

    struct whisper_vad_context* vctx = nullptr;
    if (!p.vad_model.empty()) {
        struct whisper_vad_context_params vcp = whisper_vad_default_context_params();
        vcp.n_threads = p.n_threads;
        vctx = whisper_vad_init_from_file_with_params(p.vad_model.c_str(), vcp);
        if (!vctx)
            fprintf(stderr, "%s: warning: VAD load failed, using full audio\n", argv[0]);
    }

    if (vctx) {
        struct whisper_vad_params vp = whisper_vad_default_params();
        vp.threshold = p.vad_thold;
        vp.min_speech_duration_ms = p.vad_min_speech_ms;
        vp.min_silence_duration_ms = p.vad_min_silence_ms;
        vp.speech_pad_ms = p.vad_speech_pad_ms;

        struct whisper_vad_segments* vseg =
            whisper_vad_segments_from_samples(vctx, vp, samples.data(), (int)samples.size());

        int nv = vseg ? whisper_vad_segments_n_segments(vseg) : 0;
        if (nv == 0) {
            fprintf(stderr, "%s: VAD found no speech\n", argv[0]);
            if (vseg)
                whisper_vad_free_segments(vseg);
            whisper_vad_free(vctx);
            cohere_free(cctx);
            return 1;
        }
        if (p.verbosity >= 1)
            fprintf(stderr, "%s: VAD: %d segment(s)\n", argv[0], nv);

        for (int i = 0; i < nv; i++) {
            float t0s = whisper_vad_segments_get_segment_t0(vseg, i);
            float t1s = whisper_vad_segments_get_segment_t1(vseg, i);
            int s = std::max(0, (int)(t0s * SR));
            int e = std::min((int)samples.size(), (int)(t1s * SR));
            if (e > s)
                slices.push_back({s, e, (int64_t)(t0s * 100.f), (int64_t)(t1s * 100.f)});
        }
        whisper_vad_free_segments(vseg);
        whisper_vad_free(vctx);
    } else {
        int64_t dur = (int64_t)((double)samples.size() / SR * 100.0);
        slices.push_back({0, (int)samples.size(), 0LL, dur});
    }

    // -----------------------------------------------------------------------
    // Transcribe + align each slice
    // -----------------------------------------------------------------------
    std::vector<stamp_line> all_lines;

    float ctc_frame_dur = have_ctc ? wav2vec2_frame_dur(w2v_model, SR) : 0.f;

    for (const auto& sl : slices) {
        // --- Cohere transcription ---
        struct cohere_result* r =
            cohere_transcribe_ex(cctx, samples.data() + sl.start, sl.end - sl.start, p.language.c_str(), sl.t0_cs);

        if (!r) {
            fprintf(stderr, "%s: transcription failed for segment [%s --> %s]\n", argv[0],
                    to_timestamp(sl.t0_cs).c_str(), to_timestamp(sl.t1_cs).c_str());
            continue;
        }

        // Extract word-level stamps from Cohere
        auto cohere_words = result_to_words(r, sl.t0_cs, sl.t1_cs);

        // --- CTC forced alignment (optional) ---
        if (have_ctc && !cohere_words.empty()) {
            // Build word list for alignment (letters only, lowercase)
            std::vector<std::string> ctc_words;
            ctc_words.reserve(cohere_words.size());
            for (const auto& w : cohere_words)
                ctc_words.push_back(word_for_ctc(w.text));

            // Remove empty words (pure punctuation)
            // Keep a mapping from ctc_words index → cohere_words index
            std::vector<int> valid_idx;
            std::vector<std::string> valid_words;
            for (int i = 0; i < (int)ctc_words.size(); i++) {
                if (!ctc_words[i].empty()) {
                    valid_idx.push_back(i);
                    valid_words.push_back(ctc_words[i]);
                }
            }

            if (!valid_words.empty()) {
                const float* seg_audio = samples.data() + sl.start;
                int seg_n = sl.end - sl.start;
                float seg_t0_s = (float)sl.t0_cs * 0.01f;

                auto logits = wav2vec2_compute_logits(w2v_model, seg_audio, seg_n, p.n_threads);
                int T_ctc = (int)logits.size() / (int)w2v_model.hparams.vocab_size;

                auto stamps = ctc_forced_align(logits.data(), T_ctc, (int)w2v_model.hparams.vocab_size, valid_words,
                                               w2v_model.vocab, (int)w2v_model.hparams.pad_token_id, ctc_frame_dur);

                if (!stamps.empty()) {
                    // Patch cohere_words timestamps with CTC results
                    for (int k = 0; k < (int)valid_idx.size() && k < (int)stamps.size(); k++) {
                        int ci = valid_idx[k];
                        cohere_words[ci].t0_cs = (int64_t)((seg_t0_s + stamps[k].t0) * 100.f);
                        cohere_words[ci].t1_cs = (int64_t)((seg_t0_s + stamps[k].t1) * 100.f);
                    }
                    // Propagate CTC times to adjacent punctuation-only words
                    // by copying from the nearest valid neighbour.
                    for (int i = 0; i < (int)cohere_words.size(); i++) {
                        if (ctc_words[i].empty()) {
                            // find nearest valid neighbour
                            int64_t t0 = -1, t1 = -1;
                            for (int j = i - 1; j >= 0 && t0 < 0; j--)
                                if (!ctc_words[j].empty()) {
                                    t0 = cohere_words[j].t1_cs;
                                    t1 = t0;
                                }
                            for (int j = i + 1; j < (int)cohere_words.size() && t1 < 0; j++)
                                if (!ctc_words[j].empty()) {
                                    t1 = cohere_words[j].t0_cs;
                                }
                            if (t0 < 0)
                                t0 = sl.t0_cs;
                            if (t1 < 0)
                                t1 = sl.t1_cs;
                            cohere_words[i].t0_cs = t0;
                            cohere_words[i].t1_cs = t1;
                        }
                    }
                    if (p.verbosity >= 2)
                        fprintf(stderr, "  CTC aligned %d words in [%.2f s, %.2f s]\n", (int)stamps.size(),
                                (float)sl.t0_cs * 0.01f, (float)sl.t1_cs * 0.01f);
                }
            }
        }

        // --- Emit one output line per word ---
        for (const auto& w : cohere_words) {
            if (w.text.empty())
                continue;
            stamp_line sl_out;
            sl_out.t0 = (float)w.t0_cs * 0.01f;
            sl_out.t1 = (float)w.t1_cs * 0.01f;
            sl_out.text = w.text;
            all_lines.push_back(sl_out);
        }

        cohere_result_free(r);
    }

    // -----------------------------------------------------------------------
    // Output
    // -----------------------------------------------------------------------
    for (const auto& l : all_lines)
        printf("[%s --> %s]  %s\n", fmt_ts(l.t0).c_str(), fmt_ts(l.t1).c_str(), l.text.c_str());

    if (p.output_srt)
        write_srt(make_out_path(p.fname_inp, ".srt"), all_lines, p.verbosity, argv[0]);
    if (p.output_vtt)
        write_vtt(make_out_path(p.fname_inp, ".vtt"), all_lines, p.verbosity, argv[0]);

    if (p.output_txt) {
        const std::string tp = make_out_path(p.fname_inp, ".txt");
        std::ofstream ofs(tp);
        if (ofs) {
            for (const auto& l : all_lines)
                ofs << l.text << " ";
            ofs << "\n";
            if (p.verbosity >= 1)
                fprintf(stderr, "%s: text written to '%s'\n", argv[0], tp.c_str());
        }
    }

    if (p.verbosity >= 1)
        fprintf(stderr, "\n%s: done\n", argv[0]);

    cohere_free(cctx);
    return 0;
}
