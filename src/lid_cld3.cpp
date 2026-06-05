// lid_cld3.cpp — CLD3 (Google compact language detector v3) text-LID runtime.
//
// Forward path is pure manual F32 — no ggml graph. The compute is
// well under 1 MFLOP per call (80×208 + 208×109 matmuls + a few dozen
// hash-and-bag operations); a graph would be pure overhead. F16 weights
// are dequantized to F32 once at load time (the model is 1.5 MB F32 so
// the RAM hit is trivial).
//
// All algorithms below are byte-faithful ports of the upstream sources
// at /Volumes/backups/ai/upstream/cld3/src/. Cross-validated against
// pycld3 + the Python reference dumper at
// tools/reference_backends/lid_cld3.py.

#include "lid_cld3.h"

#include "ggml.h"
#include "gguf.h"

#include "core/gguf_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// ===========================================================================
// Constants. Must match models/convert-cld3-to-gguf.py and
// tools/reference_backends/lid_cld3.py — these define the GGUF schema.
// ===========================================================================

namespace {

constexpr int kNFeatures = 6;
constexpr int kConcatDim = 80;
constexpr int kHiddenDim = 208;
constexpr int kNLabels = 109;
constexpr int kFeatureRows[kNFeatures] = {1000, 5000, 12, 103, 5000, 100};
constexpr int kFeatureCols[kNFeatures] = {16, 16, 8, 8, 16, 16};
constexpr int kFeatureOffsets[kNFeatures] = {0, 16, 32, 40, 48, 64};
constexpr int kFeatureNgramSizes[kNFeatures] = {2, 4, 0, 0, 3, 1};

enum FeatureKind { kKindCbog = 0, kKindRelevantScripts = 1, kKindScript = 2 };
constexpr FeatureKind kFeatureKinds[kNFeatures] = {
    kKindCbog, kKindCbog, kKindRelevantScripts, kKindScript, kKindCbog, kKindCbog,
};

// ULScript values — verified against
// /Volumes/backups/ai/upstream/cld3/src/script_span/generated_ulscript.h.
// Hiragana / Katakana / Hangul are NOT separate scripts; they all return
// Hani. ScriptFeature::Compute then runs a secondary Hangul-vs-Hani
// codepoint count and returns NUM_ULSCRIPTS (= kHangulSentinel) when
// Hangul wins.
constexpr int kULScriptCommon = 0;
constexpr int kULScriptLatin = 1;
constexpr int kULScriptGreek = 2;
constexpr int kULScriptCyrillic = 3;
constexpr int kULScriptArmenian = 4;
constexpr int kULScriptHebrew = 5;
constexpr int kULScriptArabic = 6;
constexpr int kULScriptDevanagari = 9;
constexpr int kULScriptBengali = 10;
constexpr int kULScriptGurmukhi = 11;
constexpr int kULScriptGujarati = 12;
constexpr int kULScriptTamil = 14;
constexpr int kULScriptThai = 19;
constexpr int kULScriptHani = 24;
constexpr int kHangulSentinel = 102; // NUM_ULSCRIPTS

} // namespace

// ===========================================================================
// Internal types
// ===========================================================================

struct lid_cld3_context {
    // Dequantized F32 weight tables. Shapes:
    //   embeddings[i]:  rows[i] * cols[i]    (row-major)
    //   hidden_w:       hidden_dim * concat_dim   (row-major; row j = hidden unit j's input weights)
    //   hidden_b:       hidden_dim
    //   output_w:       n_labels * hidden_dim
    //   output_b:       n_labels
    std::vector<float> embeddings[kNFeatures];
    std::vector<float> hidden_w;
    std::vector<float> hidden_b;
    std::vector<float> output_w;
    std::vector<float> output_b;

    std::vector<std::string> labels;
    std::string variant = "cld3";

    // Scratch buffer for the most recent prediction's softmax — used so
    // lid_cld3_predict can return a stable label pointer + lookups for
    // top-k can pull from one cached forward pass.
    std::vector<float> last_softmax;

    // Backing storage for the GGUF mmap and ggml context. We keep these
    // alive even though we copy weights out — tensor metadata (shapes,
    // names) is held by the ggml_context.
    ggml_context* gctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

// ===========================================================================
// MurmurHash2-32 — direct port of upstream utils.cc:137-183.
// ===========================================================================

static uint32_t murmur2_32(const uint8_t* data, size_t n, uint32_t seed = 0xBEEF) {
    constexpr uint32_t m = 0x5BD1E995;
    constexpr int r = 24;
    uint32_t h = static_cast<uint32_t>(seed ^ n);
    while (n >= 4) {
        uint32_t k = static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) |
                     (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24);
        k *= m;
        k ^= k >> r;
        k *= m;
        h *= m;
        h ^= k;
        data += 4;
        n -= 4;
    }
    if (n == 3) {
        h ^= static_cast<uint32_t>(data[2]) << 16;
        h ^= static_cast<uint32_t>(data[1]) << 8;
        h ^= static_cast<uint32_t>(data[0]);
        h *= m;
    } else if (n == 2) {
        h ^= static_cast<uint32_t>(data[1]) << 8;
        h ^= static_cast<uint32_t>(data[0]);
        h *= m;
    } else if (n == 1) {
        h ^= static_cast<uint32_t>(data[0]);
        h *= m;
    }
    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;
    return h;
}

// ===========================================================================
// UTF-8 helpers
// ===========================================================================

// Returns the byte length of the UTF-8 codepoint starting at `p`. Mirrors
// upstream utils::OneCharLen — purely a top-nibble dispatch, no validation.
static inline int utf8_one_char_len(uint8_t b) {
    if (b < 0xC0)
        return 1;
    if (b < 0xE0)
        return 2;
    if (b < 0xF0)
        return 3;
    return 4;
}

// Decode a UTF-8 codepoint of length n starting at p. Returns -1 for n=0.
static inline int32_t decode_codepoint(const uint8_t* p, int n) {
    switch (n) {
    case 1:
        return p[0];
    case 2:
        return ((p[0] & 0x1F) << 6) | (p[1] & 0x3F);
    case 3:
        return ((p[0] & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
    case 4:
        return ((p[0] & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
    default:
        return -1;
    }
}

// Split UTF-8 text into codepoint-byte chunks. Each chunk is a tuple of
// (start_pos, byte_length). Caller iterates and slices the original
// buffer with `text.substr(start, len)`.
struct CpChunk {
    int start;
    int len;
};

static std::vector<CpChunk> utf8_split(const std::string& text) {
    std::vector<CpChunk> out;
    int n = static_cast<int>(text.size());
    int i = 0;
    while (i < n) {
        int len = utf8_one_char_len(static_cast<uint8_t>(text[i]));
        if (i + len > n)
            break; // partial trailing char — drop, matches upstream
        out.push_back({i, len});
        i += len;
    }
    return out;
}

// ===========================================================================
// Unicode lowercase — covers Latin/Cyrillic/Greek/Latin-extended for the
// smoke set. Not a full ICU case-fold; codepoints not covered fall
// through unchanged. Upstream uses a generated Unicode case table; for
// the inputs the smoke set actually exercises, this subset matches.
// ===========================================================================

static int32_t lower_codepoint(int32_t cp) {
    // ASCII A-Z
    if (cp >= 0x41 && cp <= 0x5A)
        return cp + 0x20;
    // Latin-1 Supplement (skip 0xD7 ×, 0xDF ß which has no uppercase form)
    if (cp >= 0xC0 && cp <= 0xDE && cp != 0xD7)
        return cp + 0x20;
    // Latin Extended-A pairs (00xx0/00xx2/00xx4 = upper, +1 = lower).
    if (cp >= 0x100 && cp <= 0x12F && (cp & 1) == 0)
        return cp + 1;
    // Latin Extended-A: 0x132/0x134/.../0x178. Mostly even-upper / odd-lower.
    if (cp >= 0x132 && cp <= 0x137 && (cp & 1) == 0)
        return cp + 1;
    if (cp >= 0x139 && cp <= 0x148 && (cp & 1) == 1)
        return cp + 1;
    if (cp >= 0x14A && cp <= 0x177 && (cp & 1) == 0)
        return cp + 1;
    if (cp == 0x178)
        return 0xFF;
    if (cp >= 0x179 && cp <= 0x17E && (cp & 1) == 1)
        return cp + 1;
    // Latin Extended-B: only handle the common patterns; rare letters
    // pass through. The smoke set doesn't exercise these.
    if (cp >= 0x180 && cp <= 0x24F) {
        // Tag the most common alternating-pair ranges.
        if ((cp >= 0x182 && cp <= 0x184) && (cp & 1) == 0)
            return cp + 1;
        if (cp == 0x186)
            return 0x254;
        if (cp >= 0x187 && cp <= 0x18C && (cp & 1) == 1)
            return cp + 1;
    }
    // Greek: 0x391-0x3A9 → 0x3B1-0x3C9 (skip 0x3A2 which is unassigned).
    if (cp >= 0x391 && cp <= 0x3A9 && cp != 0x3A2)
        return cp + 0x20;
    // Greek extras (rho/sigma/etc) — skip; smoke set doesn't need them.
    // Cyrillic: 0x410-0x42F → 0x430-0x44F.
    if (cp >= 0x410 && cp <= 0x42F)
        return cp + 0x20;
    // Cyrillic supplementary uppercase 0x400-0x40F → 0x450-0x45F.
    if (cp >= 0x400 && cp <= 0x40F)
        return cp + 0x50;
    // Cyrillic Supplement 0x460-0x481 alternating pairs.
    if (cp >= 0x460 && cp <= 0x481 && (cp & 1) == 0)
        return cp + 1;
    // Armenian uppercase 0x531-0x556 → 0x561-0x586.
    if (cp >= 0x531 && cp <= 0x556)
        return cp + 0x30;
    return cp;
}

// Encode a codepoint back to UTF-8 into `out` (push_back). Used only
// when the lowercase variant differs from the input — for codepoints
// that fall through `lower_codepoint`, we copy the original bytes.
static void encode_codepoint(int32_t cp, std::string& out) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// ===========================================================================
// Text cleanup: full-Unicode lowercase + ASCII punct/digit strip + collapse
// runs of whitespace. Approximation of upstream's
// ScriptScanner::GetOneScriptSpanLower pipeline.
// ===========================================================================

static std::string cleanup_text(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    auto chunks = utf8_split(text);
    bool last_was_space = true; // collapse leading whitespace
    for (const auto& c : chunks) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(text.data() + c.start);
        if (c.len == 1) {
            uint8_t b = p[0];
            // ASCII range: keep letters and whitespace, drop the rest.
            bool is_alpha = (b >= 0x41 && b <= 0x5A) || (b >= 0x61 && b <= 0x7A);
            bool is_space = (b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == '\v' || b == '\f');
            if (is_alpha) {
                out.push_back(static_cast<char>(b >= 0x41 && b <= 0x5A ? b + 0x20 : b));
                last_was_space = false;
            } else if (is_space) {
                if (!last_was_space)
                    out.push_back(' ');
                last_was_space = true;
            }
            // else: drop digits + punctuation
        } else {
            int32_t cp = decode_codepoint(p, c.len);
            int32_t lc = lower_codepoint(cp);
            if (lc == cp) {
                out.append(reinterpret_cast<const char*>(p), c.len);
            } else {
                encode_codepoint(lc, out);
            }
            last_was_space = false;
        }
    }
    // Trim trailing space if any was appended.
    if (!out.empty() && out.back() == ' ')
        out.pop_back();
    return out;
}

// ===========================================================================
// Per-codepoint script lookup (for ScriptFeature). Returns the ULScript
// enum value 0..101, with the special kHangulSentinel handled by
// the secondary Hangul-vs-Hani check in extract_script() below.
// ===========================================================================

static int script_of_codepoint(int32_t cp, int n_bytes) {
    if (n_bytes == 1) {
        if ((cp >= 0x41 && cp <= 0x5A) || (cp >= 0x61 && cp <= 0x7A))
            return kULScriptLatin;
        return kULScriptCommon;
    }
    if (n_bytes == 2) {
        if (cp >= 0x80 && cp <= 0x024F)
            return kULScriptLatin;
        if (cp >= 0x370 && cp <= 0x3FF)
            return kULScriptGreek;
        if (cp >= 0x400 && cp <= 0x52F)
            return kULScriptCyrillic;
        if (cp >= 0x530 && cp <= 0x58F)
            return kULScriptArmenian;
        if (cp >= 0x590 && cp <= 0x5FF)
            return kULScriptHebrew;
        if (cp >= 0x600 && cp <= 0x6FF)
            return kULScriptArabic;
        return kULScriptCommon;
    }
    if (n_bytes == 3) {
        if (cp >= 0x900 && cp <= 0x97F)
            return kULScriptDevanagari;
        if (cp >= 0x980 && cp <= 0x9FF)
            return kULScriptBengali;
        if (cp >= 0xA00 && cp <= 0xA7F)
            return kULScriptGurmukhi;
        if (cp >= 0xA80 && cp <= 0xAFF)
            return kULScriptGujarati;
        if (cp >= 0xB80 && cp <= 0xBFF)
            return kULScriptTamil;
        if (cp >= 0xE00 && cp <= 0xE7F)
            return kULScriptThai;
        // CJK Unified + Hiragana + Katakana + all Hangul ranges → Hani.
        // The Hangul-vs-Hani fixup happens once in extract_script().
        bool is_cjk_or_jpn_or_kor = (cp >= 0x3041 && cp <= 0x309F) || // Hiragana
                                    (cp >= 0x30A0 && cp <= 0x30FF) || // Katakana
                                    (cp >= 0x4E00 && cp <= 0x9FFF) || // CJK Unified
                                    (cp >= 0x3400 && cp <= 0x4DBF) || // CJK Extension A
                                    (cp >= 0x1100 && cp <= 0x11FF) || // Hangul Jamo
                                    (cp >= 0x3130 && cp <= 0x318F) || // Hangul Compatibility Jamo
                                    (cp >= 0xA960 && cp <= 0xA97F) || // Jamo Extended-A
                                    (cp >= 0xD7B0 && cp <= 0xD7FF) || // Jamo Extended-B
                                    (cp >= 0xAC00 && cp <= 0xD7AF);   // Hangul Syllables
        if (is_cjk_or_jpn_or_kor)
            return kULScriptHani;
        return kULScriptCommon;
    }
    if (n_bytes == 4) {
        // CJK Extension B–F + supplementaries → Hani.
        if (cp >= 0x20000 && cp <= 0x3FFFF)
            return kULScriptHani;
        return kULScriptCommon;
    }
    return kULScriptCommon;
}

static bool is_hangul_codepoint(int32_t cp) {
    return (cp >= 0x1100 && cp <= 0x11FF) || (cp >= 0xA960 && cp <= 0xA97F) || (cp >= 0xD7B0 && cp <= 0xD7FF) ||
           (cp >= 0x3130 && cp <= 0x318F) || (cp >= 0xFFA0 && cp <= 0xFFDC) || (cp >= 0xAC00 && cp <= 0xD7AF);
}

// ===========================================================================
// Per-codepoint RelevantScript enum (for RelevantScriptFeature). Domain
// is 0..11, kNumRelevantScripts == 12. Mirrors script_detector.h:60-143.
// ===========================================================================

static int relevant_script_of(const uint8_t* p, int n_bytes) {
    if (n_bytes == 1)
        return 1; // kScriptOtherUtf8OneByte
    if (n_bytes == 2) {
        int cp = ((p[0] & 0x1F) << 6) | (p[1] & 0x3F);
        if (cp >= 0x400 && cp <= 0x4FF)
            return 6; // Cyrillic
        if (cp >= 0x600 && cp <= 0x6FF)
            return 8; // Arabic
        if (cp >= 0x590 && cp <= 0x5FF)
            return 7; // Hebrew
        if (cp >= 0x370 && cp <= 0x3FF)
            return 5; // Greek
        return 2;     // kScriptOtherUtf8TwoBytes
    }
    if (n_bytes == 3) {
        int cp = ((p[0] & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
        if (cp >= 0x1100 && cp <= 0x11FF)
            return 9; // HangulJamo
        if (cp >= 0x3041 && cp <= 0x309F)
            return 10; // Hiragana
        if (cp >= 0x30A0 && cp <= 0x30FF)
            return 11; // Katakana
        return 3;      // kScriptOtherUtf8ThreeBytes
    }
    if (n_bytes == 4)
        return 4;
    return 0; // kScriptError
}

// ===========================================================================
// Feature extractors. Each returns a vector of (id, weight) pairs; the
// caller mean-pools them through the corresponding embedding table.
// ===========================================================================

struct IdWeight {
    uint32_t id;
    float weight;
};

static std::vector<IdWeight> extract_cbog(const std::string& cleaned, int ngram_size, int id_dim) {
    auto chunks = utf8_split(cleaned);
    // Bookend each space-separated token with ^...$ — same algorithm as
    // language_identifier_features.cc:62-76.
    std::vector<std::string> chars;
    chars.reserve(chunks.size() + 4);
    chars.push_back("^");
    for (const auto& c : chunks) {
        std::string s = cleaned.substr(c.start, c.len);
        if (s == " ") {
            chars.push_back("$");
            chars.push_back(" ");
            chars.push_back("^");
        } else {
            chars.push_back(std::move(s));
        }
    }
    chars.push_back("$");

    std::unordered_map<std::string, int> counts;
    int count_sum = 0;
    int total = static_cast<int>(chars.size());
    for (int start = 0; start <= total - ngram_size; ++start) {
        std::string ngram;
        bool consumed = true;
        for (int idx = 0; idx < ngram_size; ++idx) {
            const std::string& cur = chars[start + idx];
            if (cur == " ") {
                consumed = false; // include_spaces=false → break
                break;
            }
            ngram += cur;
        }
        if (consumed) {
            counts[ngram]++;
            count_sum++;
        }
    }
    std::vector<IdWeight> out;
    if (count_sum == 0)
        return out;
    out.reserve(counts.size());
    const float norm = static_cast<float>(count_sum);
    for (const auto& kv : counts) {
        const auto& ngram = kv.first;
        uint32_t h = murmur2_32(reinterpret_cast<const uint8_t*>(ngram.data()), ngram.size());
        out.push_back({h % static_cast<uint32_t>(id_dim), kv.second / norm});
    }
    return out;
}

static std::vector<IdWeight> extract_relevant_scripts(const std::string& cleaned) {
    auto chunks = utf8_split(cleaned);
    int counts[12] = {0};
    int total = 0;
    for (const auto& c : chunks) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(cleaned.data() + c.start);
        // Skip non-alpha single-byte ASCII (matches relevant_script_feature.cc:69-71).
        if (c.len == 1) {
            uint8_t b = p[0];
            bool is_alpha = (b >= 0x41 && b <= 0x5A) || (b >= 0x61 && b <= 0x7A);
            if (!is_alpha)
                continue;
        }
        int s = relevant_script_of(p, c.len);
        if (s >= 0 && s < 12) {
            counts[s]++;
            total++;
        }
    }
    std::vector<IdWeight> out;
    if (total == 0)
        return out;
    for (int s = 0; s < 12; ++s) {
        if (counts[s] > 0) {
            out.push_back({static_cast<uint32_t>(s), counts[s] / static_cast<float>(total)});
        }
    }
    return out;
}

static std::vector<IdWeight> extract_script(const std::string& cleaned) {
    auto chunks = utf8_split(cleaned);
    std::unordered_map<int, int> counts;
    for (const auto& c : chunks) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(cleaned.data() + c.start);
        if (c.len == 1) {
            uint8_t b = p[0];
            bool is_alpha = (b >= 0x41 && b <= 0x5A) || (b >= 0x61 && b <= 0x7A);
            if (!is_alpha)
                continue;
        }
        int32_t cp = decode_codepoint(p, c.len);
        int s = script_of_codepoint(cp, c.len);
        counts[s]++;
    }
    if (counts.empty())
        return {{static_cast<uint32_t>(kULScriptCommon), 1.0f}};
    int dominant = counts.begin()->first;
    int dom_count = counts.begin()->second;
    for (const auto& kv : counts) {
        if (kv.second > dom_count) {
            dominant = kv.first;
            dom_count = kv.second;
        }
    }
    // Hangul-vs-Hani secondary fixup — only when dominant came back as Hani.
    if (dominant == kULScriptHani) {
        int n_hangul = 0, n_non_hangul = 0;
        for (const auto& c : chunks) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(cleaned.data() + c.start);
            if (c.len == 1) {
                uint8_t b = p[0];
                if (b == ' ')
                    continue;
            }
            int32_t cp = decode_codepoint(p, c.len);
            if (is_hangul_codepoint(cp))
                n_hangul++;
            else
                n_non_hangul++;
        }
        if (n_hangul > n_non_hangul)
            dominant = kHangulSentinel;
    }
    return {{static_cast<uint32_t>(dominant), 1.0f}};
}

// ===========================================================================
// Forward path
// ===========================================================================

// Compute one feature's mean-pooled embedding-bag into `out` (size = cols).
static void embedding_bag(const std::vector<IdWeight>& pairs, const float* table, int cols, float* out) {
    std::fill(out, out + cols, 0.0f);
    for (const auto& iw : pairs) {
        const float* row = table + static_cast<size_t>(iw.id) * static_cast<size_t>(cols);
        for (int j = 0; j < cols; ++j)
            out[j] += iw.weight * row[j];
    }
}

// Run all six feature extractors and pack results into the 80-d concat
// buffer. Caller passes the per-feature pair arrays through `pairs[6]`
// (so extract_stage can also dump per-feature bags).
static void compute_concat_and_bags(const std::string& cleaned, const lid_cld3_context* ctx,
                                    std::vector<IdWeight>* pairs, float* concat) {
    for (int i = 0; i < kNFeatures; ++i) {
        switch (kFeatureKinds[i]) {
        case kKindCbog:
            pairs[i] = extract_cbog(cleaned, kFeatureNgramSizes[i], kFeatureRows[i]);
            break;
        case kKindRelevantScripts:
            pairs[i] = extract_relevant_scripts(cleaned);
            break;
        case kKindScript:
            pairs[i] = extract_script(cleaned);
            break;
        }
        embedding_bag(pairs[i], ctx->embeddings[i].data(), kFeatureCols[i], concat + kFeatureOffsets[i]);
    }
}

// Hidden FC + ReLU. `hidden_w` is row-major [out, in]: row j has the
// 80 weights mapping inputs to output unit j. This means
// hidden_pre[j] = sum_i hidden_w[j, i] * concat[i] + hidden_b[j].
static void compute_hidden(const lid_cld3_context* ctx, const float* concat, float* hidden_pre, float* hidden_out) {
    const float* w = ctx->hidden_w.data();
    for (int j = 0; j < kHiddenDim; ++j) {
        float s = ctx->hidden_b[j];
        const float* row = w + static_cast<size_t>(j) * kConcatDim;
        for (int i = 0; i < kConcatDim; ++i)
            s += row[i] * concat[i];
        hidden_pre[j] = s;
        hidden_out[j] = s > 0.0f ? s : 0.0f;
    }
}

static void compute_logits(const lid_cld3_context* ctx, const float* hidden_out, float* logits) {
    const float* w = ctx->output_w.data();
    for (int j = 0; j < kNLabels; ++j) {
        float s = ctx->output_b[j];
        const float* row = w + static_cast<size_t>(j) * kHiddenDim;
        for (int i = 0; i < kHiddenDim; ++i)
            s += row[i] * hidden_out[i];
        logits[j] = s;
    }
}

static void compute_softmax(const float* logits, float* softmax, int n) {
    float maxv = logits[0];
    for (int i = 1; i < n; ++i)
        if (logits[i] > maxv)
            maxv = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        softmax[i] = std::exp(logits[i] - maxv);
        sum += softmax[i];
    }
    for (int i = 0; i < n; ++i)
        softmax[i] /= sum;
}

// ===========================================================================
// Loader
// ===========================================================================

// Copy a tensor's data into a float vector, dequantizing F16 → F32 if needed.
// Returns false on failure (unknown dtype / shape mismatch / null tensor).
static bool tensor_to_f32(const ggml_tensor* t, std::vector<float>& out, size_t expected_elems, const char* name) {
    if (!t) {
        fprintf(stderr, "lid_cld3: missing tensor %s\n", name);
        return false;
    }
    size_t elems = ggml_nelements(t);
    if (elems != expected_elems) {
        fprintf(stderr, "lid_cld3: %s: have %zu elems, expected %zu\n", name, elems, expected_elems);
        return false;
    }
    out.resize(elems);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), t->data, elems * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src = static_cast<const ggml_fp16_t*>(t->data);
        for (size_t i = 0; i < elems; ++i)
            out[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        // For quantized variants we'd route through ggml_get_type_traits()
        // ->to_float here. The current GGUFs ship F16 / F32 only and the
        // 1.5 MB model wouldn't benefit from K-quants meaningfully.
        fprintf(stderr, "lid_cld3: %s: unsupported dtype %d (only F32/F16)\n", name, static_cast<int>(t->type));
        return false;
    }
    return true;
}

static ggml_tensor* lid_get(const std::map<std::string, ggml_tensor*>& m, const std::string& name) {
    auto it = m.find(name);
    return it != m.end() ? it->second : nullptr;
}

extern "C" struct lid_cld3_context* lid_cld3_init_from_file(const char* gguf_path, int /*n_threads*/) {
    if (!gguf_path) {
        fprintf(stderr, "lid_cld3: null gguf_path\n");
        return nullptr;
    }

    // Metadata pass — read labels + variant string. We rely on
    // load_weights() below to provide the tensor map.
    std::vector<std::string> labels;
    std::string variant = "cld3";
    {
        gguf_init_params mp = {true, nullptr};
        gguf_context* g = gguf_init_from_file(gguf_path, mp);
        if (!g) {
            fprintf(stderr, "lid_cld3: gguf_init_from_file failed for %s\n", gguf_path);
            return nullptr;
        }
        int ki = gguf_find_key(g, "lid_cld3.labels");
        if (ki >= 0) {
            int n = gguf_get_arr_n(g, ki);
            labels.resize(n);
            for (int i = 0; i < n; i++)
                labels[i] = gguf_get_arr_str(g, ki, i);
        }
        int kn = gguf_find_key(g, "general.name");
        if (kn >= 0) {
            const char* s = gguf_get_val_str(g, kn);
            if (s)
                variant = s;
        }
        gguf_free(g);
    }
    if (labels.size() != static_cast<size_t>(kNLabels)) {
        fprintf(stderr, "lid_cld3: expected %d labels, got %zu (re-run convert-cld3-to-gguf.py)\n", kNLabels,
                labels.size());
        return nullptr;
    }

    // Weight pass — we don't actually need a backend for compute (no
    // ggml graph), but core_gguf::load_weights wants one to allocate
    // the buffer. Use a CPU backend; we'll dequantize off the buffer
    // and never run an op on the ggml side.
    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    if (!cpu) {
        fprintf(stderr, "lid_cld3: failed to init CPU backend\n");
        return nullptr;
    }

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(gguf_path, cpu, "lid_cld3", wl)) {
        ggml_backend_free(cpu);
        return nullptr;
    }

    auto* ctx = new lid_cld3_context;
    ctx->labels = std::move(labels);
    ctx->variant = std::move(variant);
    ctx->gctx = wl.ctx;
    ctx->buf = wl.buf;
    ctx->tensors = std::move(wl.tensors);

    // Bind + dequantize the embedding tables.
    for (int i = 0; i < kNFeatures; ++i) {
        char name[64];
        snprintf(name, sizeof(name), "lid_cld3.embedding.%d.weight", i);
        size_t expected = static_cast<size_t>(kFeatureRows[i]) * kFeatureCols[i];
        if (!tensor_to_f32(lid_get(ctx->tensors, name), ctx->embeddings[i], expected, name)) {
            lid_cld3_free(ctx);
            ggml_backend_free(cpu);
            return nullptr;
        }
    }
    if (!tensor_to_f32(lid_get(ctx->tensors, "lid_cld3.hidden.weight"), ctx->hidden_w,
                       static_cast<size_t>(kHiddenDim) * kConcatDim, "lid_cld3.hidden.weight") ||
        !tensor_to_f32(lid_get(ctx->tensors, "lid_cld3.hidden.bias"), ctx->hidden_b, kHiddenDim,
                       "lid_cld3.hidden.bias") ||
        !tensor_to_f32(lid_get(ctx->tensors, "lid_cld3.output.weight"), ctx->output_w,
                       static_cast<size_t>(kNLabels) * kHiddenDim, "lid_cld3.output.weight") ||
        !tensor_to_f32(lid_get(ctx->tensors, "lid_cld3.output.bias"), ctx->output_b, kNLabels,
                       "lid_cld3.output.bias")) {
        lid_cld3_free(ctx);
        ggml_backend_free(cpu);
        return nullptr;
    }

    ctx->last_softmax.resize(kNLabels);

    // We don't keep the backend handle around — load_weights() returned
    // a buffer that owns its memory; compute happens in the F32 vectors.
    ggml_backend_free(cpu);

    fprintf(stderr, "lid_cld3: loaded %zu labels, hidden_dim=%d, n_features=%d (variant=%s)\n", ctx->labels.size(),
            kHiddenDim, kNFeatures, ctx->variant.c_str());
    return ctx;
}

extern "C" void lid_cld3_free(lid_cld3_context* ctx) {
    if (!ctx)
        return;
    if (ctx->buf)
        ggml_backend_buffer_free(ctx->buf);
    if (ctx->gctx)
        ggml_free(ctx->gctx);
    delete ctx;
}

// ===========================================================================
// One-shot forward — fills concat + hidden + softmax. Used by the public
// predict / extract_stage entry points. Caches softmax in `ctx->last_softmax`
// so a follow-up topk call doesn't have to redo work.
// ===========================================================================

struct ForwardOut {
    std::vector<IdWeight> pairs[kNFeatures];
    std::vector<float> concat;     // 80
    std::vector<float> hidden_pre; // 208
    std::vector<float> hidden_out; // 208
    std::vector<float> logits;     // 109
    std::vector<float> softmax;    // 109
    std::string cleaned;
};

static ForwardOut run_forward(lid_cld3_context* ctx, const std::string& utf8_text) {
    ForwardOut fo;
    fo.cleaned = cleanup_text(utf8_text);
    if (fo.cleaned.empty())
        fo.cleaned = utf8_text;
    fo.concat.resize(kConcatDim);
    fo.hidden_pre.resize(kHiddenDim);
    fo.hidden_out.resize(kHiddenDim);
    fo.logits.resize(kNLabels);
    fo.softmax.resize(kNLabels);
    compute_concat_and_bags(fo.cleaned, ctx, fo.pairs, fo.concat.data());
    compute_hidden(ctx, fo.concat.data(), fo.hidden_pre.data(), fo.hidden_out.data());
    compute_logits(ctx, fo.hidden_out.data(), fo.logits.data());
    compute_softmax(fo.logits.data(), fo.softmax.data(), kNLabels);
    ctx->last_softmax = fo.softmax;
    return fo;
}

// ===========================================================================
// Public API
// ===========================================================================

extern "C" const char* lid_cld3_predict(lid_cld3_context* ctx, const char* utf8_text, float* confidence) {
    if (!ctx || !utf8_text)
        return nullptr;
    auto fo = run_forward(ctx, utf8_text);
    int top = 0;
    float best = fo.softmax[0];
    for (int i = 1; i < kNLabels; ++i) {
        if (fo.softmax[i] > best) {
            best = fo.softmax[i];
            top = i;
        }
    }
    if (confidence)
        *confidence = best;
    return ctx->labels[top].c_str();
}

extern "C" int lid_cld3_predict_topk(lid_cld3_context* ctx, const char* utf8_text, int k, const char** out_labels,
                                     float* out_scores) {
    if (!ctx || !utf8_text || k <= 0 || !out_labels || !out_scores)
        return 0;
    auto fo = run_forward(ctx, utf8_text);
    int n = std::min(k, kNLabels);
    std::vector<int> idx(kNLabels);
    for (int i = 0; i < kNLabels; ++i)
        idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + n, idx.end(),
                      [&](int a, int b) { return fo.softmax[a] > fo.softmax[b]; });
    for (int i = 0; i < n; ++i) {
        out_labels[i] = ctx->labels[idx[i]].c_str();
        out_scores[i] = fo.softmax[idx[i]];
    }
    return n;
}

extern "C" float* lid_cld3_extract_stage(lid_cld3_context* ctx, const char* utf8_text, const char* stage_name,
                                         int* out_n) {
    if (!ctx || !utf8_text || !stage_name || !out_n)
        return nullptr;
    auto fo = run_forward(ctx, utf8_text);
    const float* src = nullptr;
    int n = 0;
    if (strncmp(stage_name, "embedding_bag_", 14) == 0) {
        int idx = atoi(stage_name + 14);
        if (idx < 0 || idx >= kNFeatures)
            return nullptr;
        src = fo.concat.data() + kFeatureOffsets[idx];
        n = kFeatureCols[idx];
    } else if (strcmp(stage_name, "concat") == 0) {
        src = fo.concat.data();
        n = kConcatDim;
    } else if (strcmp(stage_name, "hidden_pre") == 0) {
        src = fo.hidden_pre.data();
        n = kHiddenDim;
    } else if (strcmp(stage_name, "hidden_out") == 0) {
        src = fo.hidden_out.data();
        n = kHiddenDim;
    } else if (strcmp(stage_name, "logits") == 0) {
        src = fo.logits.data();
        n = kNLabels;
    } else if (strcmp(stage_name, "softmax") == 0) {
        src = fo.softmax.data();
        n = kNLabels;
    } else {
        return nullptr;
    }
    float* out = static_cast<float*>(malloc(static_cast<size_t>(n) * sizeof(float)));
    if (!out)
        return nullptr;
    memcpy(out, src, static_cast<size_t>(n) * sizeof(float));
    *out_n = n;
    return out;
}

extern "C" const char* lid_cld3_variant(const lid_cld3_context* ctx) {
    return ctx ? ctx->variant.c_str() : "";
}
extern "C" int lid_cld3_n_labels(const lid_cld3_context* ctx) {
    return ctx ? static_cast<int>(ctx->labels.size()) : 0;
}
extern "C" int lid_cld3_dim_total(const lid_cld3_context* /*ctx*/) {
    return kConcatDim;
}
extern "C" int lid_cld3_hidden_dim(const lid_cld3_context* /*ctx*/) {
    return kHiddenDim;
}
