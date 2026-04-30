// nfa-align — universal multilingual subword CTC forced alignment CLI.
//
// Named after NeMo Forced Aligner (NFA), the official NVIDIA tool that uses
// the same auxiliary CTC model the same way to recover word-level timestamps
// for canary's transcripts.
//
// Takes an audio clip + a transcript and produces per-word timestamps using
// the auxiliary CTC alignment model that ships inside nvidia/canary-1b-v2.
// Works with any transcript text (canary, parakeet, cohere, whisper, even
// human-typed) — the tool re-tokenises the words through the CTC model's
// SentencePiece vocabulary and runs Viterbi to find the optimal alignment.
//
// Usage:
//   nfa-align -m canary-ctc-aligner.gguf -f audio.wav -tt "transcript text"
//   nfa-align -m canary-ctc-aligner.gguf -f audio.wav -tf transcript.txt
//   nfa-align -m canary-ctc-aligner.gguf -f audio.wav -decode    # greedy CTC

#include "canary_ctc.h"
#include "common-crispasr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static void print_usage(const char* prog) {
    fprintf(stderr,
            "\nusage: %s -m MODEL.gguf -f AUDIO.wav (-tt TEXT | -tf FILE | -decode)\n\n"
            "options:\n"
            "  -h, --help        show this help\n"
            "  -m FNAME          canary CTC aligner GGUF (from convert-canary-ctc-to-gguf.py)\n"
            "  -f FNAME          input audio (16 kHz mono WAV)\n"
            "  -tt TEXT          transcript text to align (space-separated words)\n"
            "  -tf FNAME         transcript text file (whitespace-separated)\n"
            "  -decode           greedy CTC decode mode (no alignment, just transcribe)\n"
            "  -t  N             threads (default: %d)\n\n",
            prog, std::min(4, (int)std::thread::hardware_concurrency()));
}

static std::vector<std::string> tokenise_words(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : text) {
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        } else {
            cur += c;
        }
    }
    if (!cur.empty())
        out.push_back(cur);
    return out;
}

int main(int argc, char** argv) {
    std::string model_path, audio_path, transcript_text, transcript_file;
    int n_threads = 4;
    bool decode_mode = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (a == "-m" && i + 1 < argc)
            model_path = argv[++i];
        else if (a == "-f" && i + 1 < argc)
            audio_path = argv[++i];
        else if (a == "-tt" && i + 1 < argc)
            transcript_text = argv[++i];
        else if (a == "-tf" && i + 1 < argc)
            transcript_file = argv[++i];
        else if (a == "-t" && i + 1 < argc)
            n_threads = std::atoi(argv[++i]);
        else if (a == "-decode")
            decode_mode = true;
        else {
            fprintf(stderr, "unknown option '%s'\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty() || audio_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    if (!decode_mode && transcript_text.empty() && transcript_file.empty()) {
        fprintf(stderr, "error: must pass -tt TEXT or -tf FILE or -decode\n\n");
        print_usage(argv[0]);
        return 1;
    }

    canary_ctc_context_params cp = canary_ctc_context_default_params();
    cp.n_threads = n_threads;
    canary_ctc_context* ctx = canary_ctc_init_from_file(model_path.c_str(), cp);
    if (!ctx) {
        fprintf(stderr, "%s: failed to load model\n", argv[0]);
        return 1;
    }

    // Load audio
    std::vector<float> samples;
    std::vector<std::vector<float>> stereo;
    if (!read_audio_data(audio_path, samples, stereo, /*stereo=*/false)) {
        fprintf(stderr, "%s: failed to read audio\n", argv[0]);
        canary_ctc_free(ctx);
        return 2;
    }
    fprintf(stderr, "%s: audio: %d samples (%.2fs) @ 16 kHz\n", argv[0], (int)samples.size(),
            (double)samples.size() / 16000.0);

    // Compute CTC logits via the encoder + CTC head
    float* logits = nullptr;
    int T_enc = 0, V = 0;
    int rc = canary_ctc_compute_logits(ctx, samples.data(), (int)samples.size(), &logits, &T_enc, &V);
    if (rc != 0) {
        fprintf(stderr, "%s: compute_logits failed (rc=%d)\n", argv[0], rc);
        canary_ctc_free(ctx);
        return 3;
    }
    fprintf(stderr, "%s: T_enc=%d  vocab_total=%d\n", argv[0], T_enc, V);

    if (decode_mode) {
        char* txt = canary_ctc_greedy_decode(ctx, logits, T_enc, V);
        printf("%s\n", txt ? txt : "");
        free(txt);
    } else {
        std::string text = transcript_text;
        if (!transcript_file.empty()) {
            std::ifstream f(transcript_file);
            if (!f) {
                fprintf(stderr, "cannot open '%s'\n", transcript_file.c_str());
                free(logits);
                canary_ctc_free(ctx);
                return 4;
            }
            std::stringstream ss;
            ss << f.rdbuf();
            text = ss.str();
        }
        auto words = tokenise_words(text);
        if (words.empty()) {
            fprintf(stderr, "%s: no words to align\n", argv[0]);
            free(logits);
            canary_ctc_free(ctx);
            return 5;
        }

        std::vector<canary_ctc_word> out_words(words.size());
        std::vector<const char*> word_ptrs(words.size());
        for (size_t i = 0; i < words.size(); i++)
            word_ptrs[i] = words[i].c_str();

        rc = canary_ctc_align_words(ctx, logits, T_enc, V, word_ptrs.data(), (int)words.size(), out_words.data());
        if (rc != 0) {
            fprintf(stderr, "%s: align_words failed (rc=%d)\n", argv[0], rc);
        } else {
            for (const auto& w : out_words) {
                printf("[%5d.%02ds → %5d.%02ds]  %s\n", (int)(w.t0 / 100), (int)(w.t0 % 100), (int)(w.t1 / 100),
                       (int)(w.t1 % 100), w.text);
            }
        }
    }

    free(logits);
    canary_ctc_free(ctx);
    return 0;
}
