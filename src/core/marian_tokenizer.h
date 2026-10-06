// src/core/marian_tokenizer.h — MarianMT / Opus-MT tokenizer (header-only).
//
// What Hugging Face's MarianTokenizer does for the source side, reproduced
// stage by stage so the ids are the reference's ids and not "close":
//
//   1. an optional leading `>>xxx<<` target-language code is cut off and
//      looked up as ONE vocabulary token (multi-target checkpoints);
//   2. the rest goes through the SOURCE SentencePiece model:
//        a. the normalizer — the model's precompiled charsmap (nmt_nfkc: a
//           Darts double-array trie, longest match), then whitespace
//           handling (strip, collapse, dummy prefix, space → U+2581);
//        b. unigram Viterbi, as SentencePiece's EncodeOptimized does it:
//           scores summed in double, compared against the float already
//           stored, an <unk> edge offered only where no one-character piece
//           exists, at min_score - 10;
//        c. consecutive unknowns merged into one piece whose text is the
//           merged surface;
//   3. each PIECE STRING is looked up in vocab.json. Marian's vocabulary is not
//      the SentencePiece id space: a piece the SP model knows but vocab.json
//      does not (`▁peoples` in opus-mt-en-de) is <unk>, and an unknown surface
//      that happens to be in vocab.json is that token;
//   4. </s> is appended.
//
// The pieces of step 2 and the ids of step 3 are therefore two different
// tables, and a converter that stores only one of them cannot be faithful.
//
// Not reproduced: Hugging Face splits the literal strings "</s>", "<unk>" and
// "<pad>" out of the input before step 1 and maps them to their ids. Here they
// are ordinary text.
//
// Decoding (HF `decode(skip_special_tokens=True)`): drop </s> / <unk> / <pad>,
// concatenate the pieces, U+2581 → space, strip.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace core_marian_tok {

// ── SentencePiece normalizer ───────────────────────────────────────────────

struct Normalizer {
    // precompiled charsmap, split into its two parts
    std::vector<uint32_t> trie; // Darts-clone units
    std::string pool;           // '\0'-terminated replacement strings
    bool add_dummy_prefix = true;
    bool remove_extra_whitespaces = true;
    bool escape_whitespaces = true;

    // blob = uint32 trie byte size | trie units | replacement strings.
    // An empty blob is a valid model (normalization rule "identity").
    bool load_charsmap(const uint8_t* blob, size_t n) {
        trie.clear();
        pool.clear();
        if (n == 0)
            return true;
        if (!blob || n < 4)
            return false;
        uint32_t trie_bytes = 0;
        std::memcpy(&trie_bytes, blob, 4);
        if (trie_bytes % 4 != 0 || (size_t)trie_bytes > n - 4)
            return false;
        trie.resize(trie_bytes / 4);
        if (trie_bytes)
            std::memcpy(trie.data(), blob + 4, trie_bytes);
        pool.assign((const char*)blob + 4 + trie_bytes, n - 4 - trie_bytes);
        return true;
    }

    // Longest key of the charsmap that prefixes s[pos..]. Returns its byte
    // length (0 = none) and the offset of its replacement in `pool`.
    size_t longest_match(const std::string& s, size_t pos, uint32_t& value) const {
        if (trie.empty())
            return 0;
        auto offset_of = [](uint32_t u) -> uint32_t { return (u >> 10) << ((u & (1u << 9)) >> 6); };
        size_t best = 0;
        size_t node = 0;
        uint32_t unit = trie[0];
        node ^= offset_of(unit);
        for (size_t i = pos; i < s.size(); i++) {
            const uint32_t c = (unsigned char)s[i];
            node ^= c;
            if (node >= trie.size())
                break;
            unit = trie[node];
            if ((unit & ((1u << 31) | 0xFFu)) != c)
                break;
            node ^= offset_of(unit);
            if ((unit >> 8) & 1u) {
                if (node >= trie.size())
                    break;
                value = trie[node] & 0x7FFFFFFFu;
                best = i - pos + 1;
            }
        }
        return best;
    }

    static int one_char_len(unsigned char lead) {
        static const char k_len[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
        return k_len[lead >> 4];
    }

    // Length of the well-formed UTF-8 sequence at s[pos], or 0 when it is not
    // one (SentencePiece then emits U+FFFD and consumes a single byte).
    static int valid_char_len(const std::string& s, size_t pos) {
        const unsigned char c0 = (unsigned char)s[pos];
        if (c0 < 0x80)
            return 1;
        int len = 0;
        uint32_t cp = 0;
        if ((c0 & 0xE0) == 0xC0) {
            len = 2;
            cp = c0 & 0x1F;
        } else if ((c0 & 0xF0) == 0xE0) {
            len = 3;
            cp = c0 & 0x0F;
        } else if ((c0 & 0xF8) == 0xF0) {
            len = 4;
            cp = c0 & 0x07;
        } else {
            return 0;
        }
        if (pos + (size_t)len > s.size())
            return 0;
        for (int k = 1; k < len; k++) {
            const unsigned char ck = (unsigned char)s[pos + k];
            if ((ck & 0xC0) != 0x80)
                return 0;
            cp = (cp << 6) | (ck & 0x3F);
        }
        static const uint32_t k_min[5] = {0, 0, 0x80, 0x800, 0x10000};
        if (cp < k_min[len] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
            return 0;
        return len;
    }

    // One normalization step at s[pos]: appends nothing, returns the
    // replacement text and how many input bytes it stands for.
    size_t normalize_prefix(const std::string& s, size_t pos, std::string& out) const {
        out.clear();
        uint32_t value = 0;
        const size_t m = longest_match(s, pos, value);
        if (m > 0) {
            if (value < pool.size())
                out.assign(pool.c_str() + value);
            return m;
        }
        const int len = valid_char_len(s, pos);
        if (len == 0) {
            out.assign("\xEF\xBF\xBD");
            return 1;
        }
        out.assign(s, pos, (size_t)len);
        return (size_t)len;
    }

    std::string normalize(const std::string& in) const {
        static const char k_space[] = "\xE2\x96\x81"; // U+2581
        std::string out;
        std::string piece;
        size_t pos = 0;
        if (remove_extra_whitespaces) {
            while (pos < in.size()) {
                const size_t used = normalize_prefix(in, pos, piece);
                if (piece != " ")
                    break;
                pos += used;
            }
        }
        if (pos >= in.size())
            return out;
        out.reserve(in.size() * 2 + 4);
        if (add_dummy_prefix)
            out.append(escape_whitespaces ? k_space : " ");
        bool is_prev_space = remove_extra_whitespaces;
        while (pos < in.size()) {
            pos += normalize_prefix(in, pos, piece);
            size_t k = 0;
            while (is_prev_space && k < piece.size() && piece[k] == ' ')
                k++;
            if (k < piece.size()) {
                for (size_t i = k; i < piece.size(); i++) {
                    if (escape_whitespaces && piece[i] == ' ')
                        out.append(k_space);
                    else
                        out.push_back(piece[i]);
                }
                is_prev_space = piece.back() == ' ';
            }
            if (!remove_extra_whitespaces)
                is_prev_space = false;
        }
        if (remove_extra_whitespaces) {
            const std::string space = escape_whitespaces ? k_space : " ";
            while (out.size() >= space.size() && out.compare(out.size() - space.size(), space.size(), space) == 0)
                out.resize(out.size() - space.size());
        }
        return out;
    }
};

// ── SentencePiece unigram model ────────────────────────────────────────────

// SentencePiece piece types, as in sentencepiece_model.proto.
enum PieceType { PIECE_NORMAL = 1, PIECE_UNKNOWN = 2, PIECE_CONTROL = 3, PIECE_USER_DEFINED = 4, PIECE_UNUSED = 5 };

struct Piece {
    std::string text;
    bool is_unk = false;
};

struct Unigram {
    // Only NORMAL and USER_DEFINED pieces can be matched in text; control
    // pieces ("</s>") and <unk> are not in SentencePiece's trie either.
    std::unordered_map<std::string, int32_t> piece_to_id;
    std::vector<float> scores;
    std::vector<uint8_t> user_defined;
    float min_score = 0.0f;
    float max_score = 0.0f;
    int max_piece_len = 0;

    // types may be empty: every piece is then NORMAL.
    void load(const std::vector<std::string>& pieces, const std::vector<float>& piece_scores,
              const std::vector<int32_t>& types) {
        piece_to_id.clear();
        piece_to_id.reserve(pieces.size() * 2);
        scores = piece_scores;
        scores.resize(pieces.size(), 0.0f);
        user_defined.assign(pieces.size(), 0);
        max_piece_len = 0;
        bool have = false;
        for (size_t i = 0; i < pieces.size(); i++) {
            const int type = i < types.size() ? types[i] : (int)PIECE_NORMAL;
            if (type == PIECE_NORMAL) {
                if (!have || scores[i] < min_score)
                    min_score = scores[i];
                if (!have || scores[i] > max_score)
                    max_score = scores[i];
                have = true;
            }
            if (type != PIECE_NORMAL && type != PIECE_USER_DEFINED)
                continue;
            if (pieces[i].empty())
                continue;
            user_defined[i] = type == PIECE_USER_DEFINED;
            piece_to_id.emplace(pieces[i], (int32_t)i);
            if ((int)pieces[i].size() > max_piece_len)
                max_piece_len = (int)pieces[i].size();
        }
    }

    // `s` is normalized text. Mirrors unigram::Model::EncodeOptimized.
    std::vector<Piece> encode(const std::string& s) const {
        std::vector<Piece> out;
        // int positions below; no sentence is anywhere near this.
        if (s.empty() || s.size() >= 0x7fffffffu)
            return out;
        struct Node {
            int id = -1;
            float best = 0.0f;
            int starts_at = -1;
        };
        const int size = (int)s.size();
        const float unk_score = min_score - 10.0f; // kUnkPenalty
        std::vector<Node> ends((size_t)size + 1);
        int starts_at = 0;
        std::string key;
        while (starts_at < size) {
            const float till_here = ends[starts_at].best;
            bool has_single_node = false;
            const int mblen = std::min(Normalizer::one_char_len((unsigned char)s[starts_at]), size - starts_at);
            const int limit = std::min(size, starts_at + max_piece_len);
            for (int end = starts_at + 1; end <= limit; end++) {
                key.assign(s, (size_t)starts_at, (size_t)(end - starts_at));
                auto it = piece_to_id.find(key);
                if (it == piece_to_id.end())
                    continue;
                const int32_t id = it->second;
                const int length = end - starts_at;
                // A user-defined symbol gets a bonus so that it is always chosen.
                const double score = user_defined[id] ? ((double)length * max_score - 0.1) : (double)scores[id];
                const double cand = score + (double)till_here;
                Node& t = ends[end];
                if (t.starts_at == -1 || cand > (double)t.best) {
                    t.best = (float)cand;
                    t.starts_at = starts_at;
                    t.id = id;
                }
                if (!has_single_node && length == mblen)
                    has_single_node = true;
            }
            if (!has_single_node) {
                Node& t = ends[starts_at + mblen];
                const float cand = unk_score + till_here;
                if (t.starts_at == -1 || cand > t.best) {
                    t.best = cand;
                    t.starts_at = starts_at;
                    t.id = -1; // <unk>
                }
            }
            starts_at += mblen;
        }
        // Backtrack, then merge runs of unknowns (their text is the surface).
        std::vector<std::pair<int, int>> spans; // (begin, id)
        std::vector<int> span_end;
        int end = size;
        while (end > 0) {
            const Node& n = ends[end];
            if (n.starts_at < 0)
                return {}; // unreachable: every character has at least the <unk> edge
            spans.emplace_back(n.starts_at, n.id);
            span_end.push_back(end);
            end = n.starts_at;
        }
        for (size_t k = spans.size(); k-- > 0;) {
            const bool is_unk = spans[k].second < 0;
            const std::string text = s.substr((size_t)spans[k].first, (size_t)(span_end[k] - spans[k].first));
            if (is_unk && !out.empty() && out.back().is_unk) {
                out.back().text += text;
            } else {
                out.push_back({text, is_unk});
            }
        }
        return out;
    }
};

// ── The Marian tokenizer ───────────────────────────────────────────────────

struct Tokenizer {
    Normalizer norm;
    Unigram src;
    // Target side (target.spm): only for text that is fed to the DECODER, such
    // as a fixed start of the translation. Same vocabulary.
    Normalizer tgt_norm;
    Unigram tgt;
    bool has_tgt = false;
    std::unordered_map<std::string, int32_t> vocab; // vocab.json: token → id
    int32_t unk_id = 1;
    int32_t eos_id = 0;

    int32_t lookup(const std::string& token) const {
        auto it = vocab.find(token);
        return it == vocab.end() ? unk_id : it->second;
    }

    // HF MarianTokenizer.remove_language_code: a leading ">>xx<<".
    static size_t language_code_len(const std::string& text) {
        if (text.size() < 4 || text[0] != '>' || text[1] != '>')
            return 0;
        const size_t end = text.find("<<");
        return end == std::string::npos ? 0 : end + 2;
    }

    // Target-language text as decoder ids (no language code, no </s>), the way
    // the reference tokenizes text_target. Empty when the target model is absent.
    std::vector<int32_t> encode_target(const std::string& text) const {
        std::vector<int32_t> ids;
        if (!has_tgt)
            return ids;
        for (const Piece& p : tgt.encode(tgt_norm.normalize(text)))
            ids.push_back(lookup(p.text));
        return ids;
    }

    std::vector<int32_t> encode(const std::string& text, bool add_eos = true) const {
        std::vector<int32_t> ids;
        const size_t code = language_code_len(text);
        if (code > 0)
            ids.push_back(lookup(text.substr(0, code)));
        for (const Piece& p : src.encode(norm.normalize(text.substr(code))))
            ids.push_back(lookup(p.text));
        if (add_eos)
            ids.push_back(eos_id);
        return ids;
    }
};

// HF `decode(skip_special_tokens=True)`: the caller has already dropped the
// special ids; this joins the pieces.
static inline std::string join_pieces(const std::vector<std::string>& pieces) {
    std::string s;
    for (const std::string& p : pieces) {
        size_t from = 0;
        for (;;) {
            const size_t at = p.find("\xE2\x96\x81", from);
            if (at == std::string::npos) {
                s.append(p, from, std::string::npos);
                break;
            }
            s.append(p, from, at - from);
            s.push_back(' ');
            from = at + 3;
        }
    }
    auto is_space = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0b || c == 0x0c;
    };
    size_t b = 0;
    size_t e = s.size();
    while (b < e && is_space((unsigned char)s[b]))
        b++;
    while (e > b && is_space((unsigned char)s[e - 1]))
        e--;
    return s.substr(b, e - b);
}

} // namespace core_marian_tok
