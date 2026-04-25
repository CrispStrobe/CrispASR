// crispasr_aligner.cpp — shared CTC / forced-alignment implementation.
// See crispasr_aligner.h.
//
// Extracted from examples/cli/crispasr_aligner.cpp so every CrispASR
// consumer can reach both the canary-ctc and qwen3-forced-aligner paths
// through one function call.

#include "crispasr_aligner.h"
#include "canary_ctc.h"
#include "qwen3_asr.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// Check if a Unicode codepoint is CJK (Chinese/Japanese/Korean).
// CJK characters need per-character splitting since there are no spaces.
static bool is_cjk_codepoint(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF)     // CJK Unified Ideographs
           || (cp >= 0x3400 && cp <= 0x4DBF)  // CJK Extension A
           || (cp >= 0x3040 && cp <= 0x309F)  // Hiragana
           || (cp >= 0x30A0 && cp <= 0x30FF)  // Katakana
           || (cp >= 0xAC00 && cp <= 0xD7AF)  // Hangul Syllables
           || (cp >= 0x3000 && cp <= 0x303F)  // CJK Symbols and Punctuation
           || (cp >= 0xFF00 && cp <= 0xFFEF); // Fullwidth Forms
}

// Decode one UTF-8 codepoint from pos, return (codepoint, byte_length).
static std::pair<uint32_t, int> decode_utf8(const std::string& s, size_t pos) {
    unsigned char b = (unsigned char)s[pos];
    if (b < 0x80)
        return {b, 1};
    if ((b & 0xE0) == 0xC0 && pos + 1 < s.size())
        return {((b & 0x1F) << 6) | (s[pos + 1] & 0x3F), 2};
    if ((b & 0xF0) == 0xE0 && pos + 2 < s.size())
        return {((b & 0x0F) << 12) | ((s[pos + 1] & 0x3F) << 6) | (s[pos + 2] & 0x3F), 3};
    if ((b & 0xF8) == 0xF0 && pos + 3 < s.size())
        return {((b & 0x07) << 18) | ((s[pos + 1] & 0x3F) << 12) | ((s[pos + 2] & 0x3F) << 6) | (s[pos + 3] & 0x3F), 4};
    return {b, 1};
}

// Split text into "words" for CTC alignment.
// For space-delimited languages: split on whitespace.
// For CJK: split per character (no spaces between words).
std::vector<std::string> tokenise_words(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    size_t i = 0;
    while (i < text.size()) {
        auto [cp, len] = decode_utf8(text, i);
        if (cp == ' ' || cp == '\n' || cp == '\t' || cp == '\r') {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        } else if (is_cjk_codepoint(cp)) {
            // Flush any accumulated non-CJK text
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
            // Each CJK character is its own "word" for alignment
            out.push_back(text.substr(i, len));
        } else {
            cur += text.substr(i, len);
        }
        i += len;
    }
    if (!cur.empty())
        out.push_back(cur);
    return out;
}

bool path_contains_ci(const std::string& p, const char* needle) {
    std::string lo;
    lo.reserve(p.size());
    for (char c : p)
        lo += (char)std::tolower((unsigned char)c);
    return lo.find(needle) != std::string::npos;
}

std::vector<CrispasrAlignedWord> align_qwen3_fa(const std::string& model_path, const std::vector<std::string>& words,
                                                const float* samples, int n_samples, int64_t t_offset_cs,
                                                int n_threads) {
    std::vector<CrispasrAlignedWord> out;
    if (words.empty())
        return out;

    qwen3_asr_context_params cp = qwen3_asr_context_default_params();
    cp.n_threads = n_threads;
    cp.verbosity = 0;
    qwen3_asr_context* ctx = qwen3_asr_init_from_file(model_path.c_str(), cp);
    if (!ctx) {
        fprintf(stderr, "crispasr[aligner-qwen3]: failed to load '%s'\n", model_path.c_str());
        return out;
    }
    if (qwen3_asr_lm_head_dim(ctx) == 0 || qwen3_asr_lm_head_dim(ctx) > 10000) {
        fprintf(stderr,
                "crispasr[aligner-qwen3]: model '%s' lm_head dim is %d "
                "(expected ~5000 for forced-aligner)\n",
                model_path.c_str(), qwen3_asr_lm_head_dim(ctx));
        qwen3_asr_free(ctx);
        return out;
    }

    std::vector<const char*> word_ptrs(words.size());
    for (size_t i = 0; i < words.size(); i++)
        word_ptrs[i] = words[i].c_str();

    std::vector<int64_t> start_ms(words.size(), 0);
    std::vector<int64_t> end_ms(words.size(), 0);
    int rc = qwen3_asr_align_words(ctx, samples, n_samples, word_ptrs.data(), (int)words.size(), start_ms.data(),
                                   end_ms.data());
    qwen3_asr_free(ctx);
    if (rc != 0) {
        fprintf(stderr, "crispasr[aligner-qwen3]: align_words rc=%d\n", rc);
        return out;
    }

    out.reserve(words.size());
    for (size_t i = 0; i < words.size(); i++) {
        CrispasrAlignedWord cw;
        cw.text = words[i];
        // ms → centiseconds; add slice offset so words are absolute
        // against the original audio.
        cw.t0_cs = t_offset_cs + start_ms[i] / 10;
        cw.t1_cs = t_offset_cs + end_ms[i] / 10;
        out.push_back(std::move(cw));
    }
    return out;
}

} // namespace

std::vector<CrispasrAlignedWord> crispasr_align_words(const std::string& aligner_model, const std::string& transcript,
                                                      const float* samples, int n_samples, int64_t t_offset_cs,
                                                      int n_threads) {
    std::vector<CrispasrAlignedWord> out;
    if (aligner_model.empty() || transcript.empty() || !samples || n_samples <= 0)
        return out;

    const bool is_qwen3_fa = path_contains_ci(aligner_model, "forced-aligner") ||
                             path_contains_ci(aligner_model, "qwen3-fa") ||
                             path_contains_ci(aligner_model, "qwen3-forced");
    if (is_qwen3_fa) {
        const auto words = tokenise_words(transcript);
        return align_qwen3_fa(aligner_model, words, samples, n_samples, t_offset_cs, n_threads);
    }

    canary_ctc_context_params acp = canary_ctc_context_default_params();
    acp.n_threads = n_threads;
    canary_ctc_context* actx = canary_ctc_init_from_file(aligner_model.c_str(), acp);
    if (!actx) {
        fprintf(stderr, "crispasr[aligner]: failed to load '%s'\n", aligner_model.c_str());
        return out;
    }

    float* ctc_logits = nullptr;
    int T_ctc = 0, V_ctc = 0;
    int rc = canary_ctc_compute_logits(actx, samples, n_samples, &ctc_logits, &T_ctc, &V_ctc);
    if (rc != 0) {
        fprintf(stderr, "crispasr[aligner]: compute_logits failed (rc=%d)\n", rc);
        canary_ctc_free(actx);
        return out;
    }

    const auto words = tokenise_words(transcript);
    if (words.empty()) {
        free(ctc_logits);
        canary_ctc_free(actx);
        return out;
    }

    std::vector<canary_ctc_word> aligned(words.size());
    std::vector<const char*> word_ptrs(words.size());
    for (size_t i = 0; i < words.size(); i++)
        word_ptrs[i] = words[i].c_str();

    rc = canary_ctc_align_words(actx, ctc_logits, T_ctc, V_ctc, word_ptrs.data(), (int)words.size(), aligned.data());
    free(ctc_logits);
    canary_ctc_free(actx);

    if (rc != 0) {
        fprintf(stderr, "crispasr[aligner]: align_words failed (rc=%d)\n", rc);
        return out;
    }

    out.reserve(aligned.size());
    for (const auto& w : aligned) {
        CrispasrAlignedWord cw;
        cw.text = w.text;
        cw.t0_cs = t_offset_cs + w.t0;
        cw.t1_cs = t_offset_cs + w.t1;
        out.push_back(std::move(cw));
    }
    return out;
}
