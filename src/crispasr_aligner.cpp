// crispasr_aligner.cpp — shared CTC / forced-alignment implementation.
// See crispasr_aligner.h.
//
// Extracted from examples/cli/crispasr_aligner.cpp so every CrispASR
// consumer can reach both the canary-ctc and qwen3-forced-aligner paths
// through one function call.

#include "crispasr_aligner.h"
#include "align.h"
#include "canary_ctc.h"
#include "gguf.h"
#include "qwen3_asr.h"
#include "wav2vec2-ggml.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>

// §176e: static cache for aligner model contexts. Avoids loading/freeing
// the GGUF model on every crispasr_align_words call — alignment models
// are 300 MB–1 GB and the load/free dominates wall time for short segments.
namespace {

enum class AlignerType { None, Qwen3FA, Wav2Vec2, CanaryCTC };

static std::mutex g_aligner_cache_mtx;
static void* g_aligner_cache_ctx = nullptr;
static std::string g_aligner_cache_path;
static AlignerType g_aligner_cache_type = AlignerType::None;

static void aligner_cache_free_locked() {
    if (!g_aligner_cache_ctx)
        return;
    switch (g_aligner_cache_type) {
    case AlignerType::Qwen3FA:
        qwen3_asr_free((qwen3_asr_context*)g_aligner_cache_ctx);
        break;
    case AlignerType::Wav2Vec2:
        // wav2vec2 doesn't have a public C context — not cached.
        break;
    case AlignerType::CanaryCTC:
        canary_ctc_free((canary_ctc_context*)g_aligner_cache_ctx);
        break;
    default:
        break;
    }
    g_aligner_cache_ctx = nullptr;
    g_aligner_cache_path.clear();
    g_aligner_cache_type = AlignerType::None;
}

} // namespace

void crispasr_aligner_free_cache() {
    std::lock_guard<std::mutex> lock(g_aligner_cache_mtx);
    aligner_cache_free_locked();
}

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

std::string gguf_architecture(const std::string& path) {
    struct gguf_init_params gip = {/*.no_alloc=*/true, /*.ctx=*/nullptr};
    gguf_context* gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx)
        return {};

    std::string arch;
    const int key = gguf_find_key(gctx, "general.architecture");
    if (key >= 0) {
        if (const char* s = gguf_get_val_str(gctx, key))
            arch = s;
    }
    gguf_free(gctx);

    for (char& c : arch)
        c = (char)std::tolower((unsigned char)c);
    return arch;
}

bool is_wav2vec2_aligner_model(const std::string& model_path, const std::string& arch) {
    return path_contains_ci(model_path, "wav2vec2") || path_contains_ci(model_path, "wav2vec") ||
           path_contains_ci(model_path, "xlsr") || path_contains_ci(model_path, "xls-r") ||
           path_contains_ci(model_path, "hubert") || path_contains_ci(model_path, "data2vec") || arch == "wav2vec2" ||
           arch == "wav2vec2-ctc" || arch == "hubert" || arch == "hubert-ctc" || arch == "data2vec" ||
           arch == "data2vec-audio" || arch == "data2vec_audio";
}

std::vector<CrispasrAlignedWord> align_qwen3_fa(const std::string& model_path, const std::vector<std::string>& words,
                                                const float* samples, int n_samples, int64_t t_offset_cs,
                                                int n_threads) {
    std::vector<CrispasrAlignedWord> out;
    if (words.empty())
        return out;

    // §176e: reuse cached context if same model path.
    std::lock_guard<std::mutex> lock(g_aligner_cache_mtx);
    if (g_aligner_cache_type != AlignerType::Qwen3FA || g_aligner_cache_path != model_path)
        aligner_cache_free_locked();
    qwen3_asr_context* ctx = (qwen3_asr_context*)g_aligner_cache_ctx;
    if (!ctx) {
        qwen3_asr_context_params cp = qwen3_asr_context_default_params();
        cp.n_threads = n_threads;
        cp.verbosity = 0;
        ctx = qwen3_asr_init_from_file(model_path.c_str(), cp);
        if (!ctx) {
            fprintf(stderr, "crispasr[aligner-qwen3]: failed to load '%s'\n", model_path.c_str());
            return out;
        }
        g_aligner_cache_ctx = ctx;
        g_aligner_cache_path = model_path;
        g_aligner_cache_type = AlignerType::Qwen3FA;
    }
    if (qwen3_asr_lm_head_dim(ctx) == 0 || qwen3_asr_lm_head_dim(ctx) > 10000) {
        fprintf(stderr,
                "crispasr[aligner-qwen3]: model '%s' lm_head dim is %d "
                "(expected ~5000 for forced-aligner)\n",
                model_path.c_str(), qwen3_asr_lm_head_dim(ctx));
        aligner_cache_free_locked(); // wrong model — evict from cache
        return out;
    }

    std::vector<const char*> word_ptrs(words.size());
    for (size_t i = 0; i < words.size(); i++)
        word_ptrs[i] = words[i].c_str();

    std::vector<int64_t> start_ms(words.size(), 0);
    std::vector<int64_t> end_ms(words.size(), 0);
    int rc = qwen3_asr_align_words(ctx, samples, n_samples, word_ptrs.data(), (int)words.size(), start_ms.data(),
                                   end_ms.data());
    // Do NOT free ctx — it's cached (§176e).
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

std::vector<CrispasrAlignedWord> align_wav2vec2_ctc(const std::string& model_path,
                                                    const std::vector<std::string>& words, const float* samples,
                                                    int n_samples, int64_t t_offset_cs, int n_threads) {
    std::vector<CrispasrAlignedWord> out;
    if (words.empty())
        return out;

    wav2vec2_model model;
    if (!wav2vec2_load(model_path.c_str(), model)) {
        fprintf(stderr, "crispasr[aligner-wav2vec2]: failed to load '%s'\n", model_path.c_str());
        return out;
    }

    auto logits = wav2vec2_compute_logits(model, samples, n_samples, n_threads);
    if (logits.empty()) {
        fprintf(stderr, "crispasr[aligner-wav2vec2]: compute_logits failed\n");
        return out;
    }

    const int V = (int)model.hparams.vocab_size;
    const int T = (int)(logits.size() / V);
    auto aligned = ctc_forced_align(logits.data(), T, V, words, model.vocab, (int)model.hparams.pad_token_id,
                                    wav2vec2_frame_dur(model));
    if (aligned.empty()) {
        fprintf(stderr, "crispasr[aligner-wav2vec2]: align_words failed\n");
        return out;
    }

    out.reserve(aligned.size());
    for (const auto& w : aligned) {
        CrispasrAlignedWord cw;
        cw.text = w.word;
        cw.t0_cs = t_offset_cs + (int64_t)std::llround((double)w.t0 * 100.0);
        cw.t1_cs = t_offset_cs + (int64_t)std::llround((double)w.t1 * 100.0);
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

    const std::string arch = gguf_architecture(aligner_model);
    if (is_wav2vec2_aligner_model(aligner_model, arch)) {
        const auto words = tokenise_words(transcript);
        return align_wav2vec2_ctc(aligner_model, words, samples, n_samples, t_offset_cs, n_threads);
    }

    // §176e: reuse cached canary-ctc context if same model path.
    std::lock_guard<std::mutex> lock(g_aligner_cache_mtx);
    if (g_aligner_cache_type != AlignerType::CanaryCTC || g_aligner_cache_path != aligner_model)
        aligner_cache_free_locked();
    canary_ctc_context* actx = (canary_ctc_context*)g_aligner_cache_ctx;
    if (!actx) {
        canary_ctc_context_params acp = canary_ctc_context_default_params();
        acp.n_threads = n_threads;
        actx = canary_ctc_init_from_file(aligner_model.c_str(), acp);
        if (!actx) {
            fprintf(stderr, "crispasr[aligner]: failed to load '%s'\n", aligner_model.c_str());
            return out;
        }
        g_aligner_cache_ctx = actx;
        g_aligner_cache_path = aligner_model;
        g_aligner_cache_type = AlignerType::CanaryCTC;
    }

    float* ctc_logits = nullptr;
    int T_ctc = 0, V_ctc = 0;
    int rc = canary_ctc_compute_logits(actx, samples, n_samples, &ctc_logits, &T_ctc, &V_ctc);
    if (rc != 0) {
        fprintf(stderr, "crispasr[aligner]: compute_logits failed (rc=%d)\n", rc);
        return out;
    }

    const auto words = tokenise_words(transcript);
    if (words.empty()) {
        free(ctc_logits);
        return out;
    }

    std::vector<canary_ctc_word> aligned(words.size());
    std::vector<const char*> word_ptrs(words.size());
    for (size_t i = 0; i < words.size(); i++)
        word_ptrs[i] = words[i].c_str();

    rc = canary_ctc_align_words(actx, ctc_logits, T_ctc, V_ctc, word_ptrs.data(), (int)words.size(), aligned.data());
    free(ctc_logits);
    // Do NOT free actx — it's cached (§176e).

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
