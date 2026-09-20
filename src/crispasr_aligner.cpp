// crispasr_aligner.cpp — shared CTC / forced-alignment implementation.
// See crispasr_aligner.h.
//
// Extracted from examples/cli/crispasr_aligner.cpp so every CrispASR
// consumer can reach both the canary-ctc and qwen3-forced-aligner paths
// through one function call.

#include "crispasr_aligner.h"
#include "align.h"
#include "canary_ctc.h"
#include "core/align_sentinel.h" // §W3 collapse detection
#include "core/uroman.h"
#include "gguf.h"
#include "qwen3_asr.h"
#include "wav2vec2-ggml.h"

#include <algorithm>
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
        // Heap-allocated wav2vec2_model (value type, non-copyable). Its
        // destructor frees buf→ctx→backend in Metal-safe order, so delete
        // suffices — no separate free fn. This runs from
        // crispasr_aligner_free_cache() at CLI shutdown (crispasr_run.cpp),
        // i.e. before ggml_metal's static teardown, avoiding the residency-set
        // assert the ~wav2vec2_model() comment warns about.
        delete (wav2vec2_model*)g_aligner_cache_ctx;
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

// Qwen's Python aligner removes punctuation before it creates timestamp
// slots.  Keep apostrophes (the blueprint explicitly does), and treat the
// common Unicode punctuation blocks as separators rather than alignment
// units.  Non-ASCII letters remain intact for the CTC backends.
static bool is_alignment_punctuation(uint32_t cp) {
    if (cp == '\'')
        return false;
    if (cp < 0x80)
        return !std::isalnum((unsigned char)cp) && !std::isspace((unsigned char)cp);
    return (cp >= 0x2000 && cp <= 0x206F) || // General Punctuation
           (cp >= 0x3000 && cp <= 0x303F) || // CJK Symbols and Punctuation
           (cp >= 0xFE10 && cp <= 0xFE1F) || // Vertical Forms
           (cp >= 0xFE30 && cp <= 0xFE4F) || // CJK Compatibility Forms
           (cp >= 0xFF00 && cp <= 0xFF65);   // Fullwidth punctuation
}

static bool is_opening_punctuation(uint32_t cp) {
    return cp == '(' || cp == '[' || cp == '{' || cp == 0x2018 || cp == 0x201C || // ‘ “
           cp == 0x3008 || cp == 0x300A || cp == 0x300C || cp == 0x300E ||        // 〈《「『
           cp == 0x3010 || cp == 0x3014 || cp == 0x3016 || cp == 0x3018 ||        // 【〔〖〘
           cp == 0x301A || cp == 0xFF08 || cp == 0xFF3B || cp == 0xFF5B;          // 〚（［｛
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
        } else if (is_alignment_punctuation(cp)) {
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

// Build the same number of units as tokenise_words(), but preserve punctuation
// on the unit next to it. The aligner labels remain punctuation-free; this form
// is only restored after inference so --split-on-punct can use measured word
// timestamps. Without it, crispasr_make_disp_segments treats otherwise-useful
// CJK character timings as unusable and interpolates every sentence across the
// enclosing VAD segment (#444's v0.8.34 timing-accuracy regression).
std::vector<std::string> tokenise_display_words(const std::string& text) {
    std::vector<std::string> out;
    std::string current;
    std::string prefix;
    const auto flush = [&]() {
        if (current.empty())
            return;
        out.push_back(prefix + current);
        prefix.clear();
        current.clear();
    };

    for (size_t i = 0; i < text.size();) {
        const auto [cp, len] = decode_utf8(text, i);
        const std::string bytes = text.substr(i, len);
        if (cp == ' ' || cp == '\n' || cp == '\t' || cp == '\r') {
            flush();
        } else if (is_alignment_punctuation(cp)) {
            flush();
            if (is_opening_punctuation(cp))
                prefix += bytes;
            else if (!out.empty())
                out.back() += bytes;
            else
                prefix += bytes;
        } else if (is_cjk_codepoint(cp)) {
            flush();
            out.push_back(prefix + bytes);
            prefix.clear();
        } else {
            current += bytes;
        }
        i += len;
    }
    flush();
    if (!prefix.empty() && !out.empty())
        out.back() += prefix;
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

// Maximum audio chunk duration in samples for the Qwen3-FA encoder.
// The Whisper-style conv front-end produces tensors whose element count
// is proportional to (audio_seconds * conv_channels * mel_bins).  At
// ~1400 s the first conv output exceeds INT32_MAX elements, causing a
// CUDA kernel launch failure ("invalid configuration argument") because
// ggml's unary CUDA kernels pass the element count as `int`.  We cap at
// 120 s per chunk to stay well within limits and keep VRAM reasonable.
// Issue #217.
static constexpr int QWEN3_FA_MAX_CHUNK_SAMPLES = 120 * 16000; // 120 s @ 16 kHz

// Align a single chunk — factored out so the chunking loop and the
// short-audio fast path share the same code.
static std::vector<CrispasrAlignedWord> align_qwen3_fa_one_chunk(qwen3_asr_context* ctx,
                                                                 const std::vector<std::string>& words,
                                                                 const float* samples, int n_samples,
                                                                 int64_t t_offset_cs) {
    std::vector<CrispasrAlignedWord> out;
    if (words.empty())
        return out;

    std::vector<const char*> word_ptrs(words.size());
    for (size_t i = 0; i < words.size(); i++)
        word_ptrs[i] = words[i].c_str();

    std::vector<int64_t> start_ms(words.size(), 0);
    std::vector<int64_t> end_ms(words.size(), 0);
    int rc = qwen3_asr_align_words(ctx, samples, n_samples, word_ptrs.data(), (int)words.size(), start_ms.data(),
                                   end_ms.data());
    if (rc != 0) {
        fprintf(stderr, "crispasr[aligner-qwen3]: align_words rc=%d (chunk at %.1fs)\n", rc,
                (double)t_offset_cs / 100.0);
        return out;
    }

    out.reserve(words.size());
    for (size_t i = 0; i < words.size(); i++) {
        CrispasrAlignedWord cw;
        cw.text = words[i];
        cw.t0_cs = t_offset_cs + start_ms[i] / 10;
        cw.t1_cs = t_offset_cs + end_ms[i] / 10;
        out.push_back(std::move(cw));
    }
    return out;
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

    // ── Short audio fast path: fits in one chunk ────────────────────────
    if (n_samples <= QWEN3_FA_MAX_CHUNK_SAMPLES) {
        return align_qwen3_fa_one_chunk(ctx, words, samples, n_samples, t_offset_cs);
    }

    // ── Long audio: split into chunks of QWEN3_FA_MAX_CHUNK_SAMPLES ────
    // Words are assigned proportionally to audio chunks by their index
    // position. This assumes roughly uniform speech density, which is
    // imperfect but sufficient for forced alignment — the model corrects
    // any drift within each chunk.
    const int n_chunks = (n_samples + QWEN3_FA_MAX_CHUNK_SAMPLES - 1) / QWEN3_FA_MAX_CHUNK_SAMPLES;
    const int n_words_total = (int)words.size();

    fprintf(stderr, "crispasr[aligner-qwen3]: audio %.1fs exceeds %ds limit, splitting into %d chunks\n",
            (double)n_samples / 16000.0, QWEN3_FA_MAX_CHUNK_SAMPLES / 16000, n_chunks);

    out.reserve(words.size());
    int word_cursor = 0;

    for (int c = 0; c < n_chunks; c++) {
        const int chunk_start = c * QWEN3_FA_MAX_CHUNK_SAMPLES;
        const int chunk_samples = std::min(QWEN3_FA_MAX_CHUNK_SAMPLES, n_samples - chunk_start);
        const int64_t chunk_offset_cs = t_offset_cs + (int64_t)chunk_start * 100 / 16000;

        // Assign words: proportional split by audio position.
        // Word range for chunk c: [word_cursor, word_end)
        int word_end;
        if (c == n_chunks - 1) {
            word_end = n_words_total; // last chunk gets all remaining words
        } else {
            // Proportional: fraction of total audio covered after this chunk
            double frac = (double)(chunk_start + chunk_samples) / (double)n_samples;
            word_end = std::min(n_words_total, std::max(word_cursor + 1, (int)std::round(frac * n_words_total)));
        }
        if (word_cursor >= n_words_total)
            break; // no more words to align

        const int chunk_n_words = word_end - word_cursor;
        std::vector<std::string> chunk_words(words.begin() + word_cursor, words.begin() + word_end);

        auto chunk_aligned =
            align_qwen3_fa_one_chunk(ctx, chunk_words, samples + chunk_start, chunk_samples, chunk_offset_cs);

        if (chunk_aligned.empty()) {
            fprintf(stderr, "crispasr[aligner-qwen3]: chunk %d/%d failed (%d words, %.1fs audio)\n", c + 1, n_chunks,
                    chunk_n_words, (double)chunk_samples / 16000.0);
            // Continue with remaining chunks — partial alignment is better
            // than no alignment. The missing words will leave a gap.
        }

        for (auto& w : chunk_aligned)
            out.push_back(std::move(w));

        word_cursor = word_end;
    }

    // Ensure monotonicity across chunk boundaries — the last word of
    // chunk N may have a later timestamp than the first word of chunk
    // N+1 due to proportional word assignment imprecision.
    for (size_t i = 1; i < out.size(); i++) {
        if (out[i].t0_cs < out[i - 1].t1_cs)
            out[i].t0_cs = out[i - 1].t1_cs;
        if (out[i].t1_cs < out[i].t0_cs)
            out[i].t1_cs = out[i].t0_cs;
    }

    return out;
}

std::vector<CrispasrAlignedWord> align_wav2vec2_ctc(const std::string& model_path,
                                                    const std::vector<std::string>& words, const float* samples,
                                                    int n_samples, int64_t t_offset_cs, int n_threads) {
    std::vector<CrispasrAlignedWord> out;
    if (words.empty())
        return out;

    // §176e: reuse cached model context if same model path. wav2vec2 aligner
    // GGUFs are 300 MB–1 GB, so reloading per call dominated wall time for the
    // resident server / repeated bindings-API calls (crispasr_align_words is a
    // single pass over the whole audio, so the win is cross-call, not in-call).
    // Hold the lock across compute like the qwen3-FA / canary-ctc paths — the
    // model's mutable tensor_cache is populated during compute and is not
    // thread-safe; serialising also lets that dequant cache persist across calls.
    std::lock_guard<std::mutex> lock(g_aligner_cache_mtx);
    if (g_aligner_cache_type != AlignerType::Wav2Vec2 || g_aligner_cache_path != model_path)
        aligner_cache_free_locked();
    wav2vec2_model* model = (wav2vec2_model*)g_aligner_cache_ctx;
    if (!model) {
        model = new wav2vec2_model();
        if (!wav2vec2_load(model_path.c_str(), *model)) {
            fprintf(stderr, "crispasr[aligner-wav2vec2]: failed to load '%s'\n", model_path.c_str());
            delete model;
            return out;
        }
        g_aligner_cache_ctx = model;
        g_aligner_cache_path = model_path;
        g_aligner_cache_type = AlignerType::Wav2Vec2;
    }

    auto logits = wav2vec2_compute_logits(*model, samples, n_samples, n_threads);
    if (logits.empty()) {
        fprintf(stderr, "crispasr[aligner-wav2vec2]: compute_logits failed\n");
        return out;
    }

    const int V = (int)model->hparams.vocab_size;
    const int T = (int)(logits.size() / V);
    auto aligned = ctc_forced_align(logits.data(), T, V, words, model->vocab, (int)model->hparams.pad_token_id,
                                    wav2vec2_frame_dur(*model));
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

std::vector<std::string> crispasr_tokenise_align_words(const std::string& text) {
    return tokenise_words(text);
}

std::vector<std::string> crispasr_tokenise_align_display_words(const std::string& text) {
    return tokenise_display_words(text);
}

std::vector<std::string> crispasr_parse_srt_cues(const std::string& raw) {
    std::vector<std::string> cues;
    std::string cue;
    enum { S_INDEX, S_TIME, S_TEXT } state = S_INDEX;
    auto flush = [&]() {
        // Drop whitespace-only cues so every emitted cue tokenises to >= 1 word.
        if (cue.find_first_not_of(" \t") != std::string::npos)
            cues.push_back(cue);
        cue.clear();
    };
    size_t i = 0;
    while (i < raw.size()) {
        size_t nl = raw.find('\n', i);
        if (nl == std::string::npos)
            nl = raw.size();
        std::string line = raw.substr(i, nl - i);
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        i = nl + 1;

        if (state == S_INDEX) {
            if (line.empty())
                continue;
            state = S_TIME;
        } else if (state == S_TIME) {
            state = S_TEXT;
        } else { // S_TEXT
            if (line.empty()) {
                flush();
                state = S_INDEX;
            } else {
                if (!cue.empty())
                    cue += ' ';
                cue += line;
            }
        }
    }
    flush();
    return cues;
}

// #317: extract segment texts from CrispASR JSON output. Handles two shapes:
//   1. Full output:  {"transcription": [{"text": "...", ...}, ...]}
//   2. Align output: [{"text": "...", "start": N, "end": N}, ...]
// No nlohmann dependency — the structure is simple enough for string scanning.
// Extracts every JSON string value whose key is "text" at one level of nesting
// inside an array. Robust against whitespace/newline variations.
std::vector<std::string> crispasr_parse_json_segments(const std::string& raw) {
    std::vector<std::string> segs;

    // Find the array to scan: either "transcription": [...] or a top-level [...].
    size_t arr_start = std::string::npos;
    const size_t tkey = raw.find("\"transcription\"");
    if (tkey != std::string::npos) {
        arr_start = raw.find('[', tkey);
    } else {
        // Top-level array (align-only JSON output).
        for (size_t i = 0; i < raw.size(); i++) {
            if (raw[i] == ' ' || raw[i] == '\t' || raw[i] == '\n' || raw[i] == '\r')
                continue;
            if (raw[i] == '[') {
                arr_start = i;
            }
            break;
        }
    }
    if (arr_start == std::string::npos)
        return segs;

    // Scan for "text": "..." pairs inside the array.
    size_t pos = arr_start;
    while (pos < raw.size()) {
        size_t tk = raw.find("\"text\"", pos);
        if (tk == std::string::npos)
            break;
        // Skip past the colon.
        size_t colon = raw.find(':', tk + 6);
        if (colon == std::string::npos)
            break;
        // Find the opening quote of the value.
        size_t q1 = raw.find('"', colon + 1);
        if (q1 == std::string::npos)
            break;
        // Parse the JSON string value (handle escapes).
        std::string val;
        size_t i = q1 + 1;
        while (i < raw.size() && raw[i] != '"') {
            if (raw[i] == '\\' && i + 1 < raw.size()) {
                i++;
                switch (raw[i]) {
                case '"':
                    val += '"';
                    break;
                case '\\':
                    val += '\\';
                    break;
                case 'n':
                    val += '\n';
                    break;
                case 't':
                    val += '\t';
                    break;
                case 'r':
                    val += '\r';
                    break;
                default:
                    val += raw[i];
                    break;
                }
            } else {
                val += raw[i];
            }
            i++;
        }
        pos = (i < raw.size()) ? i + 1 : raw.size();

        // Trim and emit non-empty texts.
        while (!val.empty() && (val.front() == ' ' || val.front() == '\t'))
            val.erase(val.begin());
        while (!val.empty() && (val.back() == ' ' || val.back() == '\t'))
            val.pop_back();
        if (!val.empty())
            segs.push_back(std::move(val));
    }
    return segs;
}

std::vector<CrispasrAlignedSegment> crispasr_group_aligned_segments(const std::vector<std::string>& segment_texts,
                                                                    const std::vector<CrispasrAlignedWord>& words) {
    std::vector<CrispasrAlignedSegment> out;
    size_t w = 0;
    for (size_t s = 0; s < segment_texts.size(); s++) {
        const size_t n = tokenise_words(segment_texts[s]).size();
        if (n == 0)
            continue;
        if (w >= words.size())
            break; // alignment ended early — drop the uncovered tail
        CrispasrAlignedSegment seg;
        seg.text = segment_texts[s];
        seg.word_begin = w;
        seg.word_end = std::min(w + n, words.size());
        // Leftover words (count mismatch) extend the last segment.
        if (s + 1 == segment_texts.size())
            seg.word_end = words.size();
        seg.t0_cs = words[seg.word_begin].t0_cs;
        seg.t1_cs = words[seg.word_end - 1].t1_cs;
        if (seg.t1_cs < seg.t0_cs)
            seg.t1_cs = seg.t0_cs;
        out.push_back(std::move(seg));
        w = out.back().word_end;
    }
    return out;
}

// PLAN.md §W3: forced-alignment collapse check.
//
// Installed at the ONE join every aligner backend funnels through, rather than
// in each of the three branches below — the multi-surface trap is that a check
// added to one path silently misses the CLI, the session ABI, the bindings or
// the server. `crispasr_align_words` is what all of them call.
//
// Default is DETECT + WARN. `assess()` is a new heuristic against a failure we
// have never measured in the field, so it does not get to silently rewrite
// timestamps; a wrong auto-repair would be just as invisible as the collapse.
// Opt in to repair with CRISPASR_ALIGN_SENTINEL_REDISTRIBUTE=1, or turn the
// whole check off with CRISPASR_ALIGN_SENTINEL=0.
static void align_sentinel_check(std::vector<CrispasrAlignedWord>& words, int n_samples, int64_t t_offset_cs) {
    if (words.empty())
        return;
    {
        const char* off = std::getenv("CRISPASR_ALIGN_SENTINEL");
        if (off && off[0] == '0')
            return;
    }

    // Assess in CLIP-RELATIVE seconds. The offset must come back out first:
    // align.cpp's silent-zero words are (0,0) before `t_offset_cs` is added,
    // and a chunk at offset 30 s would otherwise present them as (30.0, 30.0)
    // — a plausible-looking position, and the signal would never fire.
    std::vector<core_align_sentinel::Word> probe;
    probe.reserve(words.size());
    for (const auto& w : words)
        probe.push_back({w.text, (float)((double)(w.t0_cs - t_offset_cs) / 100.0),
                         (float)((double)(w.t1_cs - t_offset_cs) / 100.0)});

    const float audio_sec = n_samples > 0 ? (float)n_samples / 16000.0f : -1.0f;
    const auto a = core_align_sentinel::assess(probe, audio_sec);
    if (!a.collapsed)
        return;

    fprintf(stderr, "crispasr[aligner]: WARNING: %s\n", core_align_sentinel::describe(a).c_str());

    const char* fix = std::getenv("CRISPASR_ALIGN_SENTINEL_REDISTRIBUTE");
    if (!(fix && fix[0] == '1'))
        return;
    if (audio_sec <= 0.0f)
        return;

    const auto spread = core_align_sentinel::redistribute(probe, 0.0f, audio_sec);
    for (size_t i = 0; i < words.size() && i < spread.size(); i++) {
        words[i].t0_cs = t_offset_cs + (int64_t)std::llround((double)spread[i].t0 * 100.0);
        words[i].t1_cs = t_offset_cs + (int64_t)std::llround((double)spread[i].t1 * 100.0);
    }
    fprintf(stderr, "crispasr[aligner]: redistributed %zu words across %.2fs\n", words.size(), (double)audio_sec);
}

static std::vector<CrispasrAlignedWord> align_words_impl(const std::string& aligner_model,
                                                         const std::string& transcript, const float* samples,
                                                         int n_samples, int64_t t_offset_cs, int n_threads,
                                                         bool* out_load_failed);

std::vector<CrispasrAlignedWord> crispasr_align_words(const std::string& aligner_model, const std::string& transcript,
                                                      const float* samples, int n_samples, int64_t t_offset_cs,
                                                      int n_threads, bool* out_load_failed) {
    std::vector<CrispasrAlignedWord> out =
        align_words_impl(aligner_model, transcript, samples, n_samples, t_offset_cs, n_threads, out_load_failed);
    align_sentinel_check(out, n_samples, t_offset_cs);
    return out;
}

static std::vector<CrispasrAlignedWord> align_words_impl(const std::string& aligner_model,
                                                         const std::string& transcript, const float* samples,
                                                         int n_samples, int64_t t_offset_cs, int n_threads,
                                                         bool* out_load_failed) {
    std::vector<CrispasrAlignedWord> out;
    if (out_load_failed)
        *out_load_failed = false;
    if (aligner_model.empty() || transcript.empty() || !samples || n_samples <= 0)
        return out;

    const bool is_qwen3_fa = path_contains_ci(aligner_model, "forced-aligner") ||
                             path_contains_ci(aligner_model, "qwen3-fa") ||
                             path_contains_ci(aligner_model, "qwen3-forced");

    // #252: auto-romanize non-Latin reference text for CTC aligners with Latin
    // vocab. #419: romanization is an ALIGNMENT KEY, never a display
    // transform. It used to romanize the whole transcript and let the
    // romanized strings flow back out as the aligned words' text — so every
    // output path that rebuilds display text from words (-sp/-sow/srt/vtt/
    // karaoke, i.e. exactly what subtitle tools request) silently replaced
    // Cyrillic/CJK transcripts with their transliteration ("викинги" came
    // back "vikingi"). Now: tokenise the ORIGINAL transcript first, romanize
    // per word (1:1 by construction) for the aligner's labels, and map the
    // original words back onto the aligned timings before returning.
    // Disable the romanized labels with CRISPASR_ALIGN_NO_ROMANIZE=1.
    const auto orig_words = tokenise_words(transcript);
    const auto display_words = tokenise_display_words(transcript);
    std::vector<std::string> label_words = orig_words;
    {
        const char* no_rom = std::getenv("CRISPASR_ALIGN_NO_ROMANIZE");
        // Qwen3-ForcedAligner is multilingual and its blueprint feeds the
        // original script. Romanizing Chinese here changed every prompt token
        // and was the primary #444 divergence.
        if (!is_qwen3_fa && !(no_rom && no_rom[0] == '1') && core_uroman::needs_romanization(transcript)) {
            for (auto& w : label_words) {
                if (core_uroman::needs_romanization(w))
                    w = core_uroman::romanize(w);
            }
            const char* dbg = std::getenv("CRISPASR_ALIGN_DEBUG");
            if (dbg && dbg[0] == '1') {
                std::string joined;
                for (const auto& w : label_words) {
                    if (!joined.empty())
                        joined += ' ';
                    joined += w;
                }
                fprintf(stderr, "crispasr[aligner]: romanized labels → \"%s\"\n", joined.c_str());
            }
        }
    }
    // Restore the original-script text onto the aligned words. Every arm
    // aligns label_words 1:1, so a size match is the expected case; on a
    // mismatch (an arm dropped words) keep the arm's text rather than guess.
    auto restore_text = [&](std::vector<CrispasrAlignedWord> v) {
        if (v.size() == orig_words.size() && display_words.size() == orig_words.size()) {
            for (size_t i = 0; i < v.size(); i++)
                v[i].text = display_words[i];
        } else if (!v.empty()) {
            fprintf(stderr,
                    "crispasr[aligner]: aligned %zu words for %zu inputs — keeping the aligner's "
                    "own labels for this segment\n",
                    v.size(), orig_words.size());
        }
        return v;
    };

    if (is_qwen3_fa) {
        return restore_text(align_qwen3_fa(aligner_model, label_words, samples, n_samples, t_offset_cs, n_threads));
    }

    const std::string arch = gguf_architecture(aligner_model);
    if (is_wav2vec2_aligner_model(aligner_model, arch)) {
        return restore_text(align_wav2vec2_ctc(aligner_model, label_words, samples, n_samples, t_offset_cs, n_threads));
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
            if (out_load_failed)
                *out_load_failed = true;
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

    if (label_words.empty()) {
        free(ctc_logits);
        return out;
    }

    std::vector<canary_ctc_word> aligned(label_words.size());
    std::vector<const char*> word_ptrs(label_words.size());
    for (size_t i = 0; i < label_words.size(); i++)
        word_ptrs[i] = label_words[i].c_str();

    rc = canary_ctc_align_words(actx, ctc_logits, T_ctc, V_ctc, word_ptrs.data(), (int)label_words.size(),
                                aligned.data());
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
    return restore_text(std::move(out));
}
// Convert a segment's absolute centisecond timestamps to a sample interval and
// clamp it to the audio slice that produced the segment.  Use integer math so
// adjacent segments get stable boundaries on every platform; round the end up
// to avoid dropping a partial centisecond of speech.
CrispasrAlignmentAudioRange crispasr_alignment_audio_range(int64_t segment_t0_cs, int64_t segment_t1_cs,
                                                           int slice_start, int slice_end, int sample_rate) {
    CrispasrAlignmentAudioRange out;
    if (sample_rate <= 0 || slice_end <= slice_start || segment_t1_cs <= segment_t0_cs)
        return out;

    const int64_t raw_start = segment_t0_cs * sample_rate / 100;
    const int64_t raw_end = (segment_t1_cs * sample_rate + 99) / 100;
    out.start = (int)std::clamp<int64_t>(raw_start, slice_start, slice_end);
    out.end = (int)std::clamp<int64_t>(raw_end, slice_start, slice_end);
    if (!out.valid())
        return {};
    out.offset_cs = (int64_t)out.start * 100 / sample_rate;
    return out;
}
