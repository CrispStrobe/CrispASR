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

// Whitespace split — same semantics the CLI has used since the aligner
// landed. Punctuation stays attached to the word; the aligner vocab
// handles re-tokenisation.
std::vector<std::string> tokenise_words(const std::string& text) {
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
