// src/core/ctc_punct_space.h — NeMo's "no space before punctuation" detokenizer step.
//
// SentencePiece vocabularies of NeMo's punctuated (*_pc) models carry marks as
// word-initial pieces ("▁,", "▁."), so a plain SentencePiece decode yields
// "сказал , что". NeMo does not stop there: AbstractCTCDecoding (and the RNNT
// twin) collects `supported_punctuation` with
// nemo/collections/asr/parts/utils/tokenizer_utils.py::extract_punctuation_from_vocab
// and, after decoding, applies `re.sub(r'(\s)(<marks>)', r'\2', text)`, which
// removes ONE whitespace directly before each supported mark. Without it every
// comma/period of fastconformer-ctc-ru came out space-separated, while the
// NeMo blueprint gives "сказал, что" (heavy-cpu run 44679321586).
//
// Header-only and weight-free so the rule is unit-tested on its own
// (tests/test-ctc-punct-space.cpp).

#pragma once

#include "core/unicode_categ.h"
#include "core/unicode_punct.h"

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace core_ctc_punct {

// extract_punctuation_from_vocab: every P* character of every non-special
// token. Special = ^\[.*\]$ | ^<.*>$ | ^## | ^▁ | ^\s*$.
inline std::set<uint32_t> supported_punctuation(const std::vector<std::string>& vocab) {
    std::set<uint32_t> out;
    for (const std::string& t : vocab) {
        if (t.size() >= 2 && ((t.front() == '[' && t.back() == ']') || (t.front() == '<' && t.back() == '>')))
            continue;
        if (t.rfind("##", 0) == 0 || t.rfind("\xE2\x96\x81", 0) == 0)
            continue;
        bool all_space = true;
        std::vector<uint32_t> cps;
        for (size_t i = 0; i < t.size();) {
            const uint32_t cp = core_unicode::utf8_next(t.data(), t.size(), i);
            cps.push_back(cp);
            if (!core_unicode::is_space(core_unicode::category(cp)))
                all_space = false;
        }
        if (all_space)
            continue;
        for (uint32_t cp : cps)
            if (core_unicode::is_punct_p(cp))
                out.insert(cp);
    }
    return out;
}

// Applies the substitution. When `kept_before` is non-null it receives
// text.size()+1 entries: kept_before[i] = number of input bytes in [0, i) that
// survive, i.e. the output offset of input offset i (for remapping per-token
// text offsets).
inline std::string strip_space_before(const std::string& text, const std::set<uint32_t>& marks,
                                      std::vector<int>* kept_before = nullptr) {
    std::string out;
    out.reserve(text.size());
    if (kept_before)
        kept_before->assign(text.size() + 1, 0);
    size_t i = 0;
    while (i < text.size()) {
        const size_t start = i;
        const uint32_t cp = core_unicode::utf8_next(text.data(), text.size(), i);
        bool drop = false;
        if (!marks.empty() && i < text.size() && core_unicode::is_space(core_unicode::category(cp))) {
            size_t j = i;
            drop = marks.count(core_unicode::utf8_next(text.data(), text.size(), j)) != 0;
        }
        if (kept_before)
            for (size_t b = start; b < i; b++)
                (*kept_before)[b + 1] = (*kept_before)[b] + (drop ? 0 : 1);
        if (!drop)
            out.append(text, start, i - start);
    }
    return out;
}

} // namespace core_ctc_punct
