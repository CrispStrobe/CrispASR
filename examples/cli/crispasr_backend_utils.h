#pragma once

#include "whisper_params.h"
#include "core/lang_names.h"
#include "core/ngram_loop_fix.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

inline bool crispasr_backend_should_use_gpu(const whisper_params& params) {
    return params.use_gpu && params.gpu_backend != "cpu";
}

// Strip ASCII punctuation characters from `s` in-place. Keeps ASCII letters,
// digits, whitespace, and any byte ≥ 0x80 (so multi-byte UTF-8 sequences pass
// through untouched). Used by adapters whose runtime produces punctuated
// output but the user passed --no-punctuation.
inline void crispasr_strip_ascii_punctuation(std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        unsigned char u = (unsigned char)c;
        if ((u >= 'A' && u <= 'Z') || (u >= 'a' && u <= 'z') || (u >= '0' && u <= '9') || u == ' ' || u == '\t' ||
            u == '\n' || u >= 0x80) {
            out += c;
        }
    }
    s = std::move(out);
    // Collapse runs of multiple spaces left over by stripped punctuation
    // (e.g. "Hi, there" → "Hi  there" → "Hi there").
    std::string collapsed;
    collapsed.reserve(s.size());
    bool prev_space = false;
    for (char c : s) {
        const bool is_space = (c == ' ' || c == '\t' || c == '\n');
        if (is_space) {
            if (!prev_space)
                collapsed += ' ';
            prev_space = true;
        } else {
            collapsed += c;
            prev_space = false;
        }
    }
    s = std::move(collapsed);
    // Trim leading / trailing whitespace.
    auto lo = s.find_first_not_of(' ');
    auto hi = s.find_last_not_of(' ');
    s = (lo == std::string::npos) ? "" : s.substr(lo, hi - lo + 1);
}

// ASCII-lowercase pass. Multi-byte UTF-8 bytes (≥ 0x80) pass through
// unchanged — matches the historical CTC convention of "lowercase Latin
// transcript", without breaking non-Latin scripts.
inline void crispasr_lowercase_ascii(std::string& s) {
    for (char& c : s) {
        unsigned char u = (unsigned char)c;
        if (u >= 'A' && u <= 'Z')
            c = (char)(u + 32);
    }
}

// Map an ISO-639-1 code to a plain English language name for prompt
// injection in instruct-tuned audio-LLM backends. Thin alias over the
// shared core_lang::iso_to_english() (src/core/lang_names.h); kept as a
// named CLI-side function so existing adapter call sites are unchanged.
inline std::string crispasr_iso_to_english_lang(const std::string& code) {
    return core_lang::iso_to_english(code);
}

// Collapse degenerate repeated phrases from autoregressive ASR decoders after
// detokenization. This preserves logits/token parity and can be disabled for
// raw-reference comparisons with CRISPASR_NO_NGRAM_LOOPFIX=1.
inline void crispasr_apply_ngram_loop_fix(std::string& text, const char* backend_name, bool quiet) {
    const char* off = std::getenv("CRISPASR_NO_NGRAM_LOOPFIX");
    if (off && off[0] == '1')
        return;
    std::string fixed = core_ngram::fix_loops(text);
    if (fixed != text) {
        if (!quiet) {
            std::fprintf(stderr, "crispasr[%s]: collapsed n-gram loop(s) (%zu -> %zu chars)\n", backend_name,
                         text.size(), fixed.size());
        }
        text = std::move(fixed);
    }
}
