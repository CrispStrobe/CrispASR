// src/voxtral_tekken_vocab.h — decoding the packed Tekken vocabulary blob,
// and the one bound that decode has to respect (#338).
//
// The blob is a flat `[u16 len][len bytes]` stream of BPE pieces. Token ids
// are assigned `n_specials + rank`, because ids `0 .. n_specials-1` belong to
// the special tokens. That much is mechanical.
//
// What is NOT mechanical: **the blob can be longer than the model's embedding
// table.** Mistral's Tekken vocabularies serialize more entries than a given
// checkpoint activates — Voxtral-4B-TTS-2603 has `llm_vocab_size = 131072`
// with 1000 specials, so only the first 130072 BPE pieces are live, and the
// tail is inert padding. Feeding a tail piece into the encoder yields a token
// id >= llm_vocab_size, which then reaches
// `ggml_get_rows(model.token_embd, …)` out of bounds: a row-index assertion on
// CPU, and on CUDA a non-finite first frame followed by runaway generation
// until the KV cache fills. It is input-dependent — it fires only for texts
// whose BPE merge path happens to land on a tail entry — which is why it
// survived every smoke test (reported in #338).
//
// The vendored reference tokenizer used by the diff harness
// (`tools/kaggle/voxtral-diff-harness/refsrc/voxtral_tts_tokenizer.c`) has
// always enforced this: `#define MAX_VOCAB 130072`, with "IDs 1000..131071"
// spelled out at the top of the file. The production runtime did not. This
// header exists so the two cannot drift again, and so the rule is reachable
// from a unit test without loading a 4 B model.

#pragma once

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace voxtral_tekken {

// Number of BPE pieces a checkpoint can actually address, i.e. the ids left
// over once the specials have taken `0 .. n_specials-1`. Returns 0 when the
// header values are nonsensical rather than a negative count.
inline int active_bpe_count(int llm_vocab_size, int n_specials) {
    if (llm_vocab_size <= 0 || n_specials < 0 || n_specials >= llm_vocab_size)
        return 0;
    return llm_vocab_size - n_specials;
}

// True when `token_id` can be used as a row index into an embedding table of
// `llm_vocab_size` rows. Callers must gate on this before `ggml_get_rows`.
inline bool token_id_in_range(int32_t token_id, int llm_vocab_size) {
    return token_id >= 0 && token_id < llm_vocab_size;
}

struct DecodeStats {
    int n_active = 0;   // pieces admitted to the encoder
    int n_inactive = 0; // pieces parsed but past the embedding table
};

// Decode the packed blob into `id_to_piece` (every parsed piece, so a debug
// dump of an inert tail is still possible) and `piece_to_id` (**only** the
// pieces the model can address — this map is what the BPE encoder searches,
// so an entry here is a token the runtime may emit).
//
// `active_limit` is `active_bpe_count(llm_vocab_size, n_specials)`; pass 0 to
// admit everything, which is only correct when the caller genuinely has no
// embedding table to overrun.
inline DecodeStats decode_blob(const std::vector<uint8_t>& blob, int n_specials,
                               const std::vector<std::string>& specials, int active_limit,
                               std::vector<std::string>& id_to_piece, std::map<std::string, int>& piece_to_id) {
    DecodeStats st;
    id_to_piece.clear();
    piece_to_id.clear();

    // Specials occupy the low ids. They are always addressable — they are
    // inside llm_vocab_size by construction.
    for (int i = 0; i < n_specials && i < (int)specials.size(); i++) {
        id_to_piece.push_back(specials[i]);
        piece_to_id[specials[i]] = i;
    }
    while ((int)id_to_piece.size() < n_specials)
        id_to_piece.push_back("");

    const uint8_t* p = blob.data();
    const uint8_t* end = p + blob.size();
    int bpe_id = n_specials;
    while (p + 2 <= end) {
        uint16_t len = 0;
        std::memcpy(&len, p, 2);
        p += 2;
        if (p + len > end)
            break;
        std::string piece((const char*)p, len);
        p += len;
        id_to_piece.push_back(piece);
        // Keep parsing past the limit so the blob is fully consumed and the
        // inactive count is honest, but do NOT let those pieces into the map
        // the encoder searches.
        if (active_limit <= 0 || st.n_active < active_limit) {
            piece_to_id[piece] = bpe_id;
            st.n_active++;
        } else {
            st.n_inactive++;
        }
        bpe_id++;
    }
    return st;
}

// ---------------------------------------------------------------------------
// Rank-indexed form (#472) — what the Voxtral Mini 3B runtime (src/voxtral.cpp)
// uses. It keeps per-rank offsets into the blob for id -> text and a
// bytes -> rank map for the tiktoken-style merge, with token id
// `rank + n_specials`. #338 bounded the other two runtimes; this one kept
// putting every serialized rank into the merge map, so `--hotwords epigastric`
// merged to rank 145371 (id 146371 against a 131072-row token_embd).
// ---------------------------------------------------------------------------

// Walk the length-prefixed blob into per-rank offset/length tables. Stops at
// `max_ranks` (the serialized `tokenizer.tekken.n_vocab`) or at the first
// truncated entry. The tables stay COMPLETE — the tail is inert, not invalid,
// and keeping it lets id -> text still render it in a debug dump.
inline void index_blob_ranks(const std::vector<uint8_t>& blob, int max_ranks, std::vector<uint32_t>& rank_offset,
                             std::vector<uint32_t>& rank_length) {
    rank_offset.clear();
    rank_length.clear();
    if (max_ranks > 0) {
        rank_offset.reserve(max_ranks);
        rank_length.reserve(max_ranks);
    }
    const size_t n = blob.size();
    size_t pos = 0;
    for (int r = 0; max_ranks <= 0 || r < max_ranks; r++) {
        if (pos + 2 > n)
            break;
        uint16_t len = 0;
        std::memcpy(&len, blob.data() + pos, 2);
        pos += 2;
        if (pos + len > n)
            break;
        rank_offset.push_back((uint32_t)pos);
        rank_length.push_back(len);
        pos += len;
    }
}

// Build the map the merge searches, admitting only the first `active_limit`
// ranks (`active_bpe_count(llm_vocab_size, n_specials)`; 0 admits everything,
// which is only correct with no embedding table to overrun). Returns the number
// of ranks left out.
inline int build_rank_map(const std::vector<uint8_t>& blob, const std::vector<uint32_t>& rank_offset,
                          const std::vector<uint32_t>& rank_length, int active_limit,
                          std::unordered_map<std::string, int32_t>& bytes_to_rank) {
    bytes_to_rank.clear();
    const size_t n_ranks = rank_offset.size();
    const size_t n_admit = (active_limit > 0 && (size_t)active_limit < n_ranks) ? (size_t)active_limit : n_ranks;
    bytes_to_rank.reserve(n_admit);
    for (size_t r = 0; r < n_admit; r++)
        bytes_to_rank[std::string((const char*)blob.data() + rank_offset[r], rank_length[r])] = (int32_t)r;
    return (int)(n_ranks - n_admit);
}

// tiktoken merge over one pre-token: start from single bytes and repeatedly
// merge the adjacent pair with the lowest rank. Appends `rank + n_specials`
// per final piece, or 0 (`<unk>`) for a piece with no rank.
inline void bpe_encode_ranked(const std::unordered_map<std::string, int32_t>& bytes_to_rank, int n_specials,
                              const uint8_t* data, size_t len, std::vector<int32_t>& out) {
    auto rank_of = [&](size_t start, size_t l) -> int32_t {
        auto it = bytes_to_rank.find(std::string((const char*)data + start, l));
        return it != bytes_to_rank.end() ? it->second : -1;
    };
    if (len == 0)
        return;
    struct piece {
        size_t start;
        size_t len;
    };
    std::vector<piece> pieces(len);
    for (size_t i = 0; i < len; i++)
        pieces[i] = {i, 1};
    while (pieces.size() > 1) {
        int32_t best_rank = INT32_MAX;
        size_t best_idx = SIZE_MAX;
        for (size_t i = 0; i + 1 < pieces.size(); i++) {
            const int32_t r = rank_of(pieces[i].start, pieces[i].len + pieces[i + 1].len);
            if (r >= 0 && r < best_rank) {
                best_rank = r;
                best_idx = i;
            }
        }
        if (best_idx == SIZE_MAX)
            break;
        pieces[best_idx].len += pieces[best_idx + 1].len;
        pieces.erase(pieces.begin() + best_idx + 1);
    }
    for (const auto& p : pieces) {
        const int32_t r = rank_of(p.start, p.len);
        out.push_back(r >= 0 ? r + n_specials : 0);
    }
}

// Tekken regex pre-tokenizer, hand-rolled. The tekken.json pattern
// (`[^\r\n\p{L}\p{N}]?[\p{Lu}...]*[\p{Ll}...]+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`)
// attaches an optional single leading non-alphanumeric byte (typically a space)
// to the following letter/number run — so "Hello world" → ["Hello", " world"] not
// ["Hello", " ", "world"]. Skipping this lets greedy BPE merge across word
// boundaries and mis-tokenise. UTF-8 continuation/lead bytes (≥0x80) are treated
// as letters (approximates \p{L} without a full Unicode table).
// Decode one UTF-8 scalar at `i`. Returns the codepoint and advances `len`.
// Malformed input degrades to a single byte, which keeps the tokenizer total.
inline uint32_t utf8_at(const std::string& s, size_t i, size_t& len) {
    const unsigned char c = (unsigned char)s[i];
    const size_t n = s.size();
    if (c < 0x80) {
        len = 1;
        return c;
    }
    auto cont = [&](size_t k) { return i + k < n && ((unsigned char)s[i + k] & 0xC0) == 0x80; };
    if ((c & 0xE0) == 0xC0 && cont(1)) {
        len = 2;
        return ((c & 0x1Fu) << 6) | ((unsigned char)s[i + 1] & 0x3Fu);
    }
    if ((c & 0xF0) == 0xE0 && cont(1) && cont(2)) {
        len = 3;
        return ((c & 0x0Fu) << 12) | (((unsigned char)s[i + 1] & 0x3Fu) << 6) | ((unsigned char)s[i + 2] & 0x3Fu);
    }
    if ((c & 0xF8) == 0xF0 && cont(1) && cont(2) && cont(3)) {
        len = 4;
        return ((c & 0x07u) << 18) | (((unsigned char)s[i + 1] & 0x3Fu) << 12) |
               (((unsigned char)s[i + 2] & 0x3Fu) << 6) | ((unsigned char)s[i + 3] & 0x3Fu);
    }
    len = 1;
    return c;
}

// `\p{L}`-ish, by exclusion. Treating every byte >= 0x80 as a letter — what
// this did before — is right for the scripts but wrong for the punctuation and
// symbols that live above ASCII, and those are what break a run: an en dash or
// a curly quote adjacent to other punctuation split the `[^\s\p{L}\p{N}]+`
// alternative in two. Measured against mistral-common, that was the entire
// residual mismatch class once whitespace was fixed (233/4000 on a punctuation
// fuzz, 232 of them containing multibyte punctuation).
//
// This is deliberately an exclusion list, not a Unicode table: the scripts we
// synthesize are letters by default, and only the well-known punctuation and
// symbol blocks are carved out. A full \p{L} would need real Unicode data.
inline bool cp_is_letter(uint32_t cp) {
    if (cp < 0x80)
        return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    // Latin-1 punctuation/symbols, minus the three letters living among them
    // (ª U+00AA, µ U+00B5, º U+00BA) and keeping ² ³ ¹ out as No.
    if (cp >= 0x00A0 && cp <= 0x00BF)
        return cp == 0x00AA || cp == 0x00B5 || cp == 0x00BA;
    if (cp == 0x00D7 || cp == 0x00F7) // multiplication / division sign
        return false;
    if (cp >= 0x2000 && cp <= 0x206F) // General Punctuation: – — ' ' " " …
        return false;
    if (cp >= 0x20A0 && cp <= 0x20CF) // Currency Symbols: €
        return false;
    if (cp >= 0x2100 && cp <= 0x214F) // Letterlike Symbols: ™ ® ℃
        return false;
    if (cp >= 0x2190 && cp <= 0x2BFF) // arrows, math, misc symbols, dingbats
        return false;
    if (cp >= 0x3000 && cp <= 0x303F) // CJK punctuation: 、。「」
        return false;
    if (cp >= 0xFE30 && cp <= 0xFE4F) // CJK compatibility forms
        return false;
    if (cp >= 0xFF01 && cp <= 0xFF20) // fullwidth punctuation and digits
        return false;
    if (cp >= 0xFF3B && cp <= 0xFF40)
        return false;
    if (cp >= 0xFF5B && cp <= 0xFF65)
        return false;
    if (cp >= 0x1F000 && cp <= 0x1FAFF) // emoji and pictographs
        return false;
    return true;
}

inline std::vector<std::string> pre_tokenize(const std::string& text) {
    std::vector<std::string> out;
    const size_t n = text.size();

    // Everything below steps by CODEPOINT, not by byte: classification is a
    // property of the scalar, and a byte-wise walk would stop a punctuation run
    // at the lead byte of an en dash and resume on its continuation bytes.
    auto clen = [&](size_t p) {
        size_t l = 0;
        utf8_at(text, p, l);
        return l;
    };
    auto is_letter = [&](size_t p) {
        size_t l = 0;
        return cp_is_letter(utf8_at(text, p, l));
    };
    auto is_digit = [&](size_t p) {
        size_t l = 0;
        const uint32_t cp = utf8_at(text, p, l);
        return cp >= '0' && cp <= '9';
    };
    auto is_ws = [&](size_t p) {
        size_t l = 0;
        const uint32_t cp = utf8_at(text, p, l);
        return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r';
    };
    auto is_nl = [&](size_t p) { return text[p] == '\n' || text[p] == '\r'; };
    auto is_punct = [&](size_t p) { return !is_letter(p) && !is_digit(p) && !is_ws(p); };

    size_t i = 0;
    while (i < n) {
        if (is_ws(i)) {
            // Three tekken alternatives compete for a whitespace run, and which
            // one wins decides who owns its LAST character:
            //     \s*[\r\n]+  — a run containing a newline ends at that newline
            //     \s+(?!\S)   — otherwise the run keeps all BUT its last
            //                    character (or all of it at end of input)
            //     \s+          — that leftover character, if nothing claims it
            // The leftover is then offered to the next token's optional leading
            // slot, and the slots differ: the letter alternative accepts any
            // non-alphanumeric (` b`, `\tb`), the punctuation alternative accepts
            // a literal space only (` [` but not `\t[`), and `\p{N}` accepts
            // nothing, so a digit leaves the space standing alone. Consuming the
            // whole run — what this did before #339-era testing — mistokenised
            // every multi-space gap and every " (" / " [" / " -" (#338).
            size_t j = i;
            while (j < n && is_ws(j))
                j += clen(j);

            size_t last_nl = std::string::npos;
            for (size_t t = i; t < j; t++)
                if (is_nl(t))
                    last_nl = t;
            if (last_nl != std::string::npos) {
                out.push_back(text.substr(i, last_nl - i + 1));
                i = last_nl + 1;
                continue;
            }
            if (j == n) {
                out.push_back(text.substr(i, j - i));
                i = j;
                continue;
            }
            // Whitespace here is always single-byte, so j-1 is its last char.
            if (j - i >= 2) {
                out.push_back(text.substr(i, j - i - 1));
                i = j - 1;
            }
            if (is_letter(i + 1)) {
                size_t e = i + 1;
                while (e < n && is_letter(e))
                    e += clen(e);
                out.push_back(text.substr(i, e - i));
                i = e;
            } else if (text[i] == ' ' && is_punct(i + 1)) {
                size_t e = i + 1;
                while (e < n && is_punct(e))
                    e += clen(e);
                out.push_back(text.substr(i, e - i));
                i = e;
            } else {
                out.push_back(text.substr(i, 1));
                i++;
            }
            continue;
        }

        // `\p{N}` is a SINGLE codepoint with no leading slot — digits neither
        // group into runs nor absorb a preceding space. With this checkpoint's
        // vocabulary the difference is invisible in token ids (no multi-digit
        // piece exists, so BPE re-splits a grouped run into the same ids), which
        // is why an id-level parity sweep passes either way. It is still the
        // wrong split, and it would surface the moment a vocabulary carried one.
        if (is_digit(i)) {
            out.push_back(text.substr(i, clen(i)));
            i += clen(i);
            continue;
        }

        // Optional single leading non-alphanumeric, non-newline scalar before a
        // word: `[^\r\n\p{L}\p{N}]?`.
        size_t k = i;
        if (!is_letter(i) && !is_nl(i))
            k = i + clen(i);

        if (k < n && is_letter(k)) {
            size_t e = k;
            while (e < n && is_letter(e))
                e += clen(e);
            out.push_back(text.substr(i, e - i));
            i = e;
        } else {
            size_t e = i;
            while (e < n && is_punct(e))
                e += clen(e);
            if (e == i) // never stall
                e += clen(i);
            out.push_back(text.substr(i, e - i));
            i = e;
        }
    }
    return out;
}

} // namespace voxtral_tekken
