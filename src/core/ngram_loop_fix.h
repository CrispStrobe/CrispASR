// core/ngram_loop_fix.h — collapse degenerate greedy n-gram loops in decoded
// text.
//
// Autoregressive ASR decoders (higgs-audio-v3-stt, MOSS-Transcribe, ... — all
// Qwen3-1.7B-class LMs decoded greedily) occasionally fall into a repeated
// n-gram attractor and emit the same phrase until the max-token cap
// ("Hey, hey, hey, ..." / "run hey hey hey run ..."). higgs-audio ships an
// `ngram_loop_fix.py` post-process for exactly this; the collapse below is that
// algorithm, extracted here so multiple backends share one implementation.
//
// The transform is a pure text post-process: it never touches the token
// stream, so per-token / logit parity against a Python reference is unchanged.
// It is a no-op on non-degenerate text — only *immediately* repeated n-grams
// beyond `max_rep` reps are trimmed — so a clean transcript passes through
// byte-for-byte.

#pragma once

#include <string>
#include <vector>

namespace core_ngram {

// Collapse immediately-repeated n-grams (window size `n`) in `w` to at most
// `max_rep` consecutive reps. Walks left-to-right building `out`; whenever the
// next n words equal the tail of `out` and that tail already repeats >= max_rep
// times, the duplicate n-gram is dropped.
inline std::vector<std::string> collapse(const std::vector<std::string>& w, int n, int max_rep) {
    std::vector<std::string> out;
    const int L = (int)w.size();
    int i = 0;
    auto tail_eq = [&]() {
        for (int k = 0; k < n; k++)
            if (w[i + k] != out[out.size() - n + k])
                return false;
        return true;
    };
    while (i < L) {
        bool matched = false;
        if ((int)out.size() >= n && i + n <= L && tail_eq()) {
            int reps = 1;
            while ((int)out.size() >= n * (reps + 1)) {
                bool eq = true;
                const size_t b = out.size() - (size_t)n * (reps + 1);
                for (int k = 0; k < n; k++)
                    if (out[b + k] != out[out.size() - n + k]) {
                        eq = false;
                        break;
                    }
                if (!eq)
                    break;
                reps++;
            }
            if (reps >= max_rep) {
                i += n;
                matched = true;
            }
        }
        if (!matched) {
            out.push_back(w[i]);
            i++;
        }
    }
    return out;
}

// Split `text` on whitespace, collapse repeated n-grams from `max_n` down to 1
// (unigrams kept up to 3 reps, longer n-grams up to 2), and re-join with single
// spaces. Returns cleaned text.
inline std::string fix_loops(const std::string& text, int max_n = 16) {
    std::vector<std::string> words;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && std::isspace((unsigned char)text[i]))
            i++;
        size_t j = i;
        while (j < text.size() && !std::isspace((unsigned char)text[j]))
            j++;
        if (j > i)
            words.push_back(text.substr(i, j - i));
        i = j;
    }
    for (int n = max_n; n >= 1; n--)
        words = collapse(words, n, n == 1 ? 3 : 2);
    std::string out;
    for (size_t k = 0; k < words.size(); k++) {
        if (k)
            out += ' ';
        out += words[k];
    }
    return out;
}

} // namespace core_ngram
