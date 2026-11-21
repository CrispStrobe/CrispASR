// gap_fill_policy.h — the decisions the issue #89 gap-fill second pass makes,
// shared by its two implementations: the CLI/server pass
// (examples/cli/crispasr_gap_fill.h) and the session C ABI's
// transcribe_ja_sliced (src/crispasr_c_api.cpp). One copy, so the surfaces
// cannot drift apart (they did once already, #356).
//
// Gap-fill re-transcribes a span a slice left empty, in isolation. On the model
// it was built for (parakeet-tdt-0.6b-ja) that span is often real speech the
// encoder blanked, and it comes back verbatim. On noisy audio it is just as
// often a pause, music or a sound effect, and a short window of noise makes the
// TDT decoder invent words (#605). Three guards keep the first kind and reject
// the second:
//
//   enabled()           whether the pass runs at all. CRISPASR_GAP_FILL forces
//                       it on or off; unset, the backend's own default applies.
//                       Haas-ja (KDA predictor) shows no #89 drop: on the #89
//                       podcast it reaches 97 % recall without gap-fill and the
//                       pass adds nothing, while on the #605 game stream it
//                       only adds noise. So it defaults off there.
//   run_is_confident()  keep a gap's recovery only if the decoder was sure of
//                       it. Measured on the #605 clip: first-pass words have a
//                       median mean token confidence of 0.997, recovered junk
//                       0.1-0.6. On the #89 podcast every parakeet-ja recovery
//                       clears 0.85, so the default gate loses none of them.
//   is_seam_duplicate() a refill window re-hears the first-pass word at the
//                       hole's edge and times it slightly later, inside the
//                       hole ("はい" + "はい。"). Same 0.5 s rule as the
//                       library's non-JA repair (parakeet_orchestrate.cpp), but
//                       the comparison also ignores CJK punctuation.
//
// Hermetic: tests/test-core-gap-fill-policy.cpp.
#pragma once

#include "env_gate.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace core_gap_fill {

// Default minimum mean token confidence for a recovered run. The #605/#89
// measurements put the cut between 0.85 (no #89 recovery lost) and 0.9 (some
// lost), so the default sits at the safe end.
inline constexpr float kDefaultMinRunConfidence = 0.85f;

// A recovered word this close (centiseconds) to a kept word with the same text
// is the same word heard twice across the seam.
inline constexpr int64_t kSeamCs = 50;

// CRISPASR_GAP_FILL: an off spelling (0/false/off/no) disables the pass, any
// other non-empty value forces it on, unset or empty leaves `backend_default`.
inline bool enabled(bool backend_default) {
    const char* e = std::getenv("CRISPASR_GAP_FILL");
    if (!e || !*e)
        return backend_default;
    return !core_env::spells_off(e);
}

// CRISPASR_GAP_FILL_MIN_CONF overrides the gate (clamped to [0, 1]); 0 turns
// it off, restoring the pre-#605 behaviour.
inline float min_run_confidence() {
    float v = kDefaultMinRunConfidence;
    if (const char* e = std::getenv("CRISPASR_GAP_FILL_MIN_CONF")) {
        if (*e) {
            char* end = nullptr;
            const float f = std::strtof(e, &end);
            if (end != e)
                v = f;
        }
    }
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

// True when the mean of the available confidences reaches `min_conf`.
// Negative entries mean "unavailable" and are skipped; a run with no
// confidence at all cannot be judged and is kept, as before #605.
inline bool run_is_confident(const std::vector<float>& confidences, float min_conf) {
    double sum = 0.0;
    int n = 0;
    for (float c : confidences)
        if (c >= 0.0f) {
            sum += c;
            ++n;
        }
    if (n == 0)
        return true;
    return sum / n >= (double)min_conf;
}

// True for a token/word text that is only whitespace (a TDT word-boundary
// token carries no content and should not vote in the confidence gate).
inline bool is_blank(const std::string& s) {
    for (unsigned char c : s)
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r')
            return false;
    return true;
}

// Comparison form of a word: ASCII letters/digits lower-cased; ASCII
// punctuation, whitespace, CJK symbols and punctuation (U+3000-U+303F) and the
// full-width ASCII punctuation block dropped; everything else kept as is.
inline std::string norm_word(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        if (c < 0x80) {
            if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z'))
                o += (char)c;
            else if (c >= 'A' && c <= 'Z')
                o += (char)(c - 'A' + 'a');
            ++i;
            continue;
        }
        size_t len = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
        if (i + len > s.size())
            len = s.size() - i;
        uint32_t cp = 0;
        if (len == 3)
            cp = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)((unsigned char)s[i + 1] & 0x3F) << 6) |
                 (uint32_t)((unsigned char)s[i + 2] & 0x3F);
        const bool cjk_punct = cp >= 0x3000 && cp <= 0x303F;
        const bool fullwidth_punct = (cp >= 0xFF01 && cp <= 0xFF0F) || (cp >= 0xFF1A && cp <= 0xFF20) ||
                                     (cp >= 0xFF3B && cp <= 0xFF40) || (cp >= 0xFF5B && cp <= 0xFF65);
        if (!cjk_punct && !fullwidth_punct)
            o.append(s, i, len);
        i += len;
    }
    return o;
}

// True when a word starting at `t0` with comparison form `norm` repeats a kept
// word within kSeamCs. `kept` holds (t0, norm_word(text)) of every word kept so
// far. An empty comparison form (pure punctuation) never matches.
inline bool is_seam_duplicate(int64_t t0, const std::string& norm,
                              const std::vector<std::pair<int64_t, std::string>>& kept) {
    if (norm.empty())
        return false;
    for (const auto& k : kept) {
        const int64_t d = k.first > t0 ? k.first - t0 : t0 - k.first;
        if (d < kSeamCs && k.second == norm)
            return true;
    }
    return false;
}

} // namespace core_gap_fill
