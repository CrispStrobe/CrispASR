#ifndef CRISPASR_TRUECASE_LOADER_H
#define CRISPASR_TRUECASE_LOADER_H

#include "crispasr_cache.h"
#include "truecaser.h"
#include "truecaser_crf.h"
#include "truecaser_lstm.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

// Shared `--truecase-model` resolution + apply for the CLI one-shot path
// (crispasr_run.cpp) and the HTTP server (crispasr_server.cpp). There are three
// truecaser flavours (statistical / CRF / BiLSTM) selected by alias or by magic
// bytes of a direct path; keeping that resolution in one place stops the two
// front-ends from drifting (the server originally applied no truecasing at all).

using crispasr_truecaser_ptr = std::unique_ptr<truecaser_context, decltype(&truecaser_free)>;
using crispasr_truecaser_crf_ptr = std::unique_ptr<truecaser_crf_context, decltype(&truecaser_crf_free)>;
using crispasr_truecaser_lstm_ptr = std::unique_ptr<truecaser_lstm_context, decltype(&truecaser_lstm_free)>;

// Resolve `truecase_model` and load it into exactly one of the three contexts.
// Mirrors the resolution in crispasr_run.cpp:
//   none|off            → nothing
//   lstm[-de|en|es|ru]  → BiLSTM truecaser (auto-download)
//   crf|crf-de          → CRF truecaser (auto-download)
//   auto|de             → statistical German truecaser (auto-download)
//   <path>              → format detected by 4-byte magic (LSTM / CRF1 / else statistical)
inline void crispasr_load_truecase(const std::string& truecase_model, bool no_prints, const std::string& cache_dir,
                                   crispasr_truecaser_ptr& tc, crispasr_truecaser_crf_ptr& tc_crf,
                                   crispasr_truecaser_lstm_ptr& tc_lstm, const char* label = "crispasr") {
    std::string tc_path = truecase_model;
    if (tc_path == "none" || tc_path == "off")
        return;

    if (tc_path == "lstm" || tc_path == "lstm-de" || tc_path == "lstm-en" || tc_path == "lstm-es" ||
        tc_path == "lstm-ru") {
        std::string lang = "de";
        if (tc_path == "lstm-en")
            lang = "en";
        else if (tc_path == "lstm-es")
            lang = "es";
        else if (tc_path == "lstm-ru")
            lang = "ru";
        std::string fname = "truecaser-lstm-" + lang + ".bin";
        std::string url = "https://huggingface.co/cstr/truecaser-de/resolve/main/" + fname;
        tc_path = crispasr_cache::ensure_cached_file(fname, url, no_prints, label, cache_dir);
        if (!tc_path.empty()) {
            tc_lstm.reset(truecaser_lstm_init(tc_path.c_str()));
            if (tc_lstm && !no_prints)
                fprintf(stderr, "%s: loaded BiLSTM truecaser '%s'\n", label, tc_path.c_str());
        }
        return;
    }
    if (tc_path == "crf" || tc_path == "crf-de") {
        tc_path = crispasr_cache::ensure_cached_file(
            "truecaser-crf-de.bin", "https://huggingface.co/cstr/truecaser-de/resolve/main/truecaser-crf-de.bin",
            no_prints, label, cache_dir);
        if (!tc_path.empty()) {
            tc_crf.reset(truecaser_crf_init(tc_path.c_str()));
            if (tc_crf && !no_prints)
                fprintf(stderr, "%s: loaded CRF truecaser '%s'\n", label, tc_path.c_str());
        }
        return;
    }
    if (!tc_path.empty() && tc_path != "auto" && tc_path != "de") {
        // Direct path — detect format by magic bytes.
        FILE* probe = fopen(tc_path.c_str(), "rb");
        if (probe) {
            char magic[4] = {};
            (void)!fread(magic, 1, 4, probe);
            fclose(probe);
            if (memcmp(magic, "LSTM", 4) == 0) {
                tc_lstm.reset(truecaser_lstm_init(tc_path.c_str()));
                if (tc_lstm && !no_prints)
                    fprintf(stderr, "%s: loaded BiLSTM truecaser '%s'\n", label, tc_path.c_str());
                return;
            }
            if (memcmp(magic, "CRF1", 4) == 0) {
                tc_crf.reset(truecaser_crf_init(tc_path.c_str()));
                if (tc_crf && !no_prints)
                    fprintf(stderr, "%s: loaded CRF truecaser '%s'\n", label, tc_path.c_str());
                return;
            }
        }
    }
    if (tc_path == "auto" || tc_path == "de") {
        tc_path = crispasr_cache::ensure_cached_file(
            "truecaser-de.bin", "https://huggingface.co/cstr/truecaser-de/resolve/main/truecaser-de.bin", no_prints,
            label, cache_dir);
    }
    if (!tc_path.empty()) {
        tc.reset(truecaser_init(tc_path.c_str()));
        if (!tc)
            fprintf(stderr, "%s: warning: failed to load truecaser '%s' — continuing without\n", label,
                    tc_path.c_str());
        else if (!no_prints)
            fprintf(stderr, "%s: loaded truecaser '%s'\n", label, tc_path.c_str());
    }
}

// Apply whichever truecaser is loaded to `text` in place. At most one of the
// three is ever populated.
inline void crispasr_apply_truecase(truecaser_context* tc, truecaser_crf_context* tc_crf,
                                    truecaser_lstm_context* tc_lstm, std::string& text) {
    char* out = nullptr;
    if (tc_lstm)
        out = truecaser_lstm_process(tc_lstm, text.c_str());
    else if (tc_crf)
        out = truecaser_crf_process(tc_crf, text.c_str());
    else if (tc)
        out = truecaser_process(tc, text.c_str());
    if (out) {
        text = out;
        free(out);
    }
}

#endif // CRISPASR_TRUECASE_LOADER_H
