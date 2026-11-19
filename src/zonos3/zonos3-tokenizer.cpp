// zonos3-tokenizer.cpp — SuperBPE: added-token split, the Split regex pre-tokenizer
//   \p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)   (behavior Isolated)
// as a hand-written matcher, GPT-2 byte-level mapping, then rank-ordered BPE merges.
#include "zonos3-tokenizer.h"
#include "zonos3-text.h"
#include "unicode-data.h"

#include "json.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace {

bool in_ranges(const uint32_t (*r)[2], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < r[mid][0])
            hi = mid;
        else if (cp > r[mid][1])
            lo = mid + 1;
        else
            return true;
    }
    return false;
}

bool is_L(uint32_t cp) {
    return in_ranges(ZONOS3_UNI_L, sizeof(ZONOS3_UNI_L) / sizeof(ZONOS3_UNI_L[0]), cp);
}
bool is_N(uint32_t cp) {
    return in_ranges(ZONOS3_UNI_N, sizeof(ZONOS3_UNI_N) / sizeof(ZONOS3_UNI_N[0]), cp);
}
// Unicode White_Space (Oniguruma \s): unlike Python str.isspace() (is_py_space) it excludes 0x1C-0x1F.
bool is_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}
bool is_P(uint32_t cp) {
    return !is_space(cp) && !is_L(cp) && !is_N(cp);
}

using cp_span = utf8_cp; // code point and its byte range in the segment

// Split [0, n) into pieces: each regex match is a piece, each maximal gap between matches is one.
std::vector<std::pair<size_t, size_t>> pretokenize(const std::vector<cp_span>& cps) {
    const size_t n = cps.size();
    std::vector<std::pair<size_t, size_t>> pieces;
    size_t gap = 0, i = 0;
    auto match_at = [&](size_t i) -> size_t { // end of the match starting at i, or i if none
        // A1: \p{N}{1,3}
        if (is_N(cps[i].cp)) {
            size_t j = i;
            while (j < n && j - i < 3 && is_N(cps[j].cp))
                ++j;
            return j;
        }
        // A2: ' '?[^\s\p{L}\p{N}]{2,}[\r\n/]*
        {
            const size_t j = i + (cps[i].cp == 0x20 ? 1 : 0);
            size_t k = j;
            while (k < n && is_P(cps[k].cp))
                ++k;
            if (k - j >= 2) {
                while (k < n && (cps[k].cp == '\r' || cps[k].cp == '\n' || cps[k].cp == '/'))
                    ++k;
                return k;
            }
        }
        // A3: ' +(?!\S)'
        if (cps[i].cp == 0x20) {
            size_t e = i;
            while (e < n && cps[e].cp == 0x20)
                ++e;
            if (e == n || is_space(cps[e].cp))
                return e;
            if (e - i >= 2)
                return e - 1;
        }
        return i;
    };
    while (i < n) {
        const size_t e = match_at(i);
        if (e == i) {
            ++i;
            continue;
        }
        if (gap < i)
            pieces.push_back({gap, i});
        pieces.push_back({i, e});
        i = gap = e;
    }
    if (gap < n)
        pieces.push_back({gap, n});
    return pieces;
}

// GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to U+0100 + k.
const std::vector<std::string>& byte_symbols() {
    static const std::vector<std::string> tbl = [] {
        std::vector<std::string> t(256);
        int extra = 0;
        for (int b = 0; b < 256; ++b) {
            const bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            const uint32_t cp = keep ? (uint32_t)b : 256u + (uint32_t)extra++;
            std::string s;
            if (cp < 0x80)
                s += (char)cp;
            else {
                s += (char)(0xC0 | (cp >> 6));
                s += (char)(0x80 | (cp & 0x3F));
            }
            t[b] = s;
        }
        return t;
    }();
    return tbl;
}

void bpe_piece(const zonos3_tokenizer& tok, const std::string& s, size_t b0, size_t b1, std::vector<int32_t>& out) {
    const auto& bs = byte_symbols();
    std::vector<std::string> sym;
    sym.reserve(b1 - b0);
    for (size_t i = b0; i < b1; ++i)
        sym.push_back(bs[(unsigned char)s[i]]);
    std::string key;
    while (sym.size() > 1) {
        int32_t best = INT32_MAX;
        size_t at = 0;
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            key.assign(sym[i]);
            key += '\x01';
            key += sym[i + 1];
            const auto it = tok.merge_rank.find(key);
            if (it != tok.merge_rank.end() && it->second < best) {
                best = it->second;
                at = i;
            }
        }
        if (best == INT32_MAX)
            break;
        sym[at] += sym[at + 1];
        sym.erase(sym.begin() + (ptrdiff_t)at + 1);
    }
    for (const auto& x : sym) {
        const auto it = tok.vocab.find(x);
        if (it == tok.vocab.end()) {
            fprintf(stderr, "zonos3: tokenizer: symbol not in vocab (skipped)\n");
            continue;
        }
        out.push_back(it->second + tok.offset);
    }
}

} // namespace

bool zonos3_tokenizer_load(zonos3_tokenizer& tok, const std::string& json, int32_t offset, int32_t bos, int32_t eos,
                           std::string& err) {
    tok = zonos3_tokenizer();
    tok.offset = offset;
    tok.bos = bos;
    tok.eos = eos;
    nlohmann::json j = nlohmann::json::parse(json, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) {
        err = "tokenizer: invalid JSON";
        return false;
    }
    const auto& model = j["model"];
    if (!model.is_object() || model.value("type", "") != "BPE") {
        err = "tokenizer: model.type is not BPE";
        return false;
    }
    if (!model["vocab"].is_object() || !model["merges"].is_array()) {
        err = "tokenizer: missing vocab/merges";
        return false;
    }
    for (auto it = model["vocab"].begin(); it != model["vocab"].end(); ++it) {
        if (!it.value().is_number_integer()) {
            err = "tokenizer: non-integer vocab id";
            return false;
        }
        tok.vocab[it.key()] = it.value().get<int32_t>();
    }
    int32_t rank = 0;
    for (const auto& mg : model["merges"]) {
        if (!mg.is_array() || mg.size() != 2 || !mg[0].is_string() || !mg[1].is_string()) {
            err = "tokenizer: merges must be [a,b] arrays";
            return false;
        }
        tok.merge_rank.emplace(mg[0].get<std::string>() + '\x01' + mg[1].get<std::string>(), rank++);
    }
    if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
        for (const auto& a : j["added_tokens"]) {
            if (!a.is_object() || !a["content"].is_string() || !a["id"].is_number_integer()) {
                err = "tokenizer: malformed added_tokens";
                return false;
            }
            const std::string c = a["content"].get<std::string>();
            if (!c.empty())
                tok.added.push_back({c, a["id"].get<int32_t>()});
        }
    }
    if (tok.vocab.empty()) {
        err = "tokenizer: empty vocab";
        return false;
    }
    return true;
}

namespace {

// SuperBPE of `utf8` (added tokens split first), no BOS/EOS.
void bpe_text(const zonos3_tokenizer& tok, const std::string& utf8, std::vector<int32_t>& out) {
    auto encode_segment = [&](size_t b, size_t e) {
        if (b >= e)
            return;
        const std::vector<cp_span> cps = utf8_spans(utf8, b, e);
        for (const auto& pc : pretokenize(cps))
            bpe_piece(tok, utf8, cps[pc.first].b0, cps[pc.second - 1].b1, out);
    };
    size_t pos = 0;
    while (pos < utf8.size()) {
        size_t best = std::string::npos, best_len = 0;
        int32_t best_id = 0;
        for (const auto& a : tok.added) {
            const size_t f = utf8.find(a.first, pos);
            if (f != std::string::npos && (f < best || (f == best && a.first.size() > best_len))) {
                best = f;
                best_len = a.first.size();
                best_id = a.second;
            }
        }
        if (best == std::string::npos) {
            encode_segment(pos, utf8.size());
            break;
        }
        encode_segment(pos, best);
        out.push_back(best_id + tok.offset);
        pos = best + best_len;
    }
}

// Python repr() of a one-code-point str (printable code points verbatim; quotes as Python picks them).
std::string py_char_repr(const std::string& ch) {
    constexpr uint32_t not_one_cp = 0xFFFFFFFFu; // ch is not exactly one code point
    const std::vector<cp_span> cps = utf8_spans(ch, 0, ch.size());
    const uint32_t cp = cps.size() == 1 ? cps[0].cp : not_one_cp;
    char buf[16];
    std::string body;
    if (cp == '\\')
        body = "\\\\";
    else if (cp == '\n')
        body = "\\n";
    else if (cp == '\r')
        body = "\\r";
    else if (cp == '\t')
        body = "\\t";
    else if (cp < 0x20 || (cp >= 0x7F && cp <= 0xA0) || cp == 0xAD) {
        snprintf(buf, sizeof buf, "\\x%02x", cp);
        body = buf;
    } else if (cp != not_one_cp &&
               ((cp >= 0xE000 && cp <= 0xF8FF) || (cp >= 0x2000 && cp <= 0x200F) || (cp >= 0x2028 && cp <= 0x202F) ||
                (cp >= 0x205F && cp <= 0x206F) || cp == 0x1680 || cp == 0x3000 || cp == 0xFEFF)) {
        snprintf(buf, sizeof buf, "\\u%04x", cp);
        body = buf;
    } else
        body = ch;
    if (cp == '\'')
        return "\"'\"";
    return "'" + body + "'";
}

} // namespace

std::vector<int32_t> zonos3_tokenize(const zonos3_tokenizer& tok, const std::string& utf8) {
    std::vector<int32_t> out{tok.bos};
    bpe_text(tok, utf8, out);
    out.push_back(tok.eos);
    return out;
}

bool zonos3_tokenizer_set_spec(zonos3_tokenizer& tok, const std::string& spec_json, uint32_t text_vocab,
                               uint32_t text_extra_tokens, std::string& err) {
    const nlohmann::json spec = nlohmann::json::parse(spec_json, nullptr, /*allow_exceptions=*/false);
    if (spec.is_discarded() || !spec.is_object()) {
        err = "text_tokenizer_spec: invalid JSON";
        return false;
    }
    try {
        if (spec.at("kind").get<std::string>() != "superbpe" || spec.at("offset").get<int64_t>() != 4 ||
            spec.at("bos").get<int64_t>() != 2 || spec.at("eos").get<int64_t>() != 3 ||
            (int64_t)text_vocab != spec.at("vocab_size").get<int64_t>() + 4 || tok.offset != 4 || tok.bos != 2 ||
            tok.eos != 3) {
            err = "SuperBPE ID convention mismatch";
            return false;
        }
        tok.has_phonemes = false;
        tok.phonemes = zonos3_phoneme_spec();
        tok.start_id = -1;
        tok.pause_ids.clear();
        tok.pause_seconds.clear();
        tok.block_first = -1;
        tok.block_rows = 0;
        zonos3_phoneme_spec ph;
        bool has_phonemes = false;
        if (spec.contains("phonemes") && !spec["phonemes"].is_null()) {
            const auto& p = spec["phonemes"];
            ph.open_marker = p.at("open_marker").get<std::string>();
            ph.close_marker = p.at("close_marker").get<std::string>();
            ph.open_id = p.at("open_id").get<int32_t>();
            ph.close_id = p.at("close_id").get<int32_t>();
            ph.symbols = p.at("symbols").get<std::vector<std::string>>();
            if (ph.open_id < (int64_t)text_vocab + 1 || ph.close_id != ph.open_id + 1) {
                err = "Phoneme token IDs must follow the text padding row (and any edit marker rows)";
                return false;
            }
            bool valid = utf8_spans(ph.open_marker, 0, ph.open_marker.size()).size() == 1 &&
                         utf8_spans(ph.close_marker, 0, ph.close_marker.size()).size() == 1;
            for (size_t i = 0; valid && i < ph.symbols.size(); ++i) {
                const auto& s = ph.symbols[i];
                valid = utf8_spans(s, 0, s.size()).size() == 1 && s != ph.open_marker && s != ph.close_marker &&
                        std::find(ph.symbols.begin(), ph.symbols.begin() + (ptrdiff_t)i, s) ==
                            ph.symbols.begin() + (ptrdiff_t)i;
            }
            if (!valid) {
                err = "Invalid phoneme symbols";
                return false;
            }
            has_phonemes = true;
        }
        const bool has_start = spec.contains("start") && !spec["start"].is_null();
        const bool has_pauses = spec.contains("pauses") && !spec["pauses"].is_null();
        // superbpe._block_first / phoneme_token_count
        const int64_t first = has_phonemes ? ph.open_id
                              : has_start  ? spec["start"].at("id").get<int64_t>()
                              : has_pauses ? spec["pauses"].at("ids").at(0).get<int64_t>()
                                           : -1;
        const int64_t n_ph = has_phonemes ? 2 + (int64_t)ph.symbols.size() : 0;
        int32_t start_id = -1;
        if (has_start) {
            const auto& s = spec["start"];
            if (s.at("marker").get<std::string>() != ZONOS3_START_MARKER) {
                err = "Invalid start marker";
                return false;
            }
            start_id = s.at("id").get<int32_t>();
            if (first < (int64_t)text_vocab + 1 || start_id != first + n_ph) {
                err = "Start token ID must follow the phoneme rows";
                return false;
            }
        }
        std::vector<int32_t> pause_ids;
        std::vector<double> pause_seconds;
        if (has_pauses) {
            const auto& p = spec["pauses"];
            pause_ids = p.at("ids").get<std::vector<int32_t>>();
            pause_seconds = p.at("seconds").get<std::vector<double>>();
            const auto markers = p.at("markers").get<std::vector<std::string>>();
            bool valid = !pause_ids.empty() && pause_ids.size() <= 8 && markers.size() == pause_ids.size() &&
                         pause_seconds.size() == pause_ids.size();
            for (size_t j = 0; valid && j < markers.size(); ++j)
                valid = markers[j] == std::string("\xEE\x80") + (char)(0x83 + j);
            if (!valid) {
                err = "Invalid pause markers";
                return false;
            }
            const int64_t lo = first + n_ph + (has_start ? 1 : 0);
            valid = first >= (int64_t)text_vocab + 1;
            for (size_t j = 0; valid && j < pause_ids.size(); ++j)
                valid = pause_ids[j] == lo + (int64_t)j;
            if (!valid) {
                err = "Pause token IDs must follow the start row";
                return false;
            }
        }
        const int64_t rows = n_ph + (has_start ? 1 : 0) + (int64_t)pause_ids.size(); // superbpe.extra_token_count
        if (rows > 0 && (first - text_vocab - 1) + rows != (int64_t)text_extra_tokens) {
            err = "Phoneme / start / pause tokens do not match text_extra_tokens (they must be the last extra rows)";
            return false;
        }
        if (has_phonemes)
            tok.phonemes = std::move(ph);
        tok.has_phonemes = has_phonemes;
        tok.start_id = start_id;
        tok.pause_ids = std::move(pause_ids);
        tok.pause_seconds = std::move(pause_seconds);
        tok.block_first = (int32_t)first;
        tok.block_rows = (int32_t)rows;
    } catch (const std::exception& e) {
        err = std::string("text_tokenizer_spec: ") + e.what();
        return false;
    }
    tok.spec_json = spec_json;
    return true;
}

namespace {

const char* const NO_START_TOKEN = "This checkpoint has no start token: remove the U+E002 start marker";
const char* const NO_PAUSE_TOKENS = "This checkpoint has no pause tokens: remove the U+E003-U+E00A pause markers";
const char* const START_NOT_FIRST = "The start marker (U+E002) must be the first character of the text, once";

// Byte offset of the first pause marker (U+E003..U+E00A = EE 80 83..8A) at or after `pos`, npos without one.
size_t find_pause(const std::string& text, size_t pos) {
    for (size_t i = text.find('\xEE', pos); i != std::string::npos; i = text.find('\xEE', i + 1))
        if (i + 2 < text.size() && (unsigned char)text[i + 1] == 0x80 && (unsigned char)text[i + 2] >= 0x83 &&
            (unsigned char)text[i + 2] <= 0x8A)
            return i;
    return std::string::npos;
}

// Bucket (0..7) of the pause marker at byte `i`.
int pause_bucket(const std::string& text, size_t i) {
    return (unsigned char)text[i + 2] - 0x83;
}

std::string pause_outside(int bucket, size_t n) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "Pause marker U+%04X outside the %zu pause buckets", 0xE003 + bucket, n);
    return buf;
}

// The start marker is the first character of `text`, once.
bool start_first_once(const std::string& text) {
    return text.rfind(ZONOS3_START_MARKER, 0) == 0 && text.find(ZONOS3_START_MARKER, 3) == std::string::npos;
}

// superbpe.payload_ids past the start and pause branches: SuperBPE with pronunciation groups.
bool phoneme_payload(const zonos3_tokenizer& tok, const std::string& text, std::vector<int32_t>& ids,
                     std::string& err) {
    if (!tok.has_phonemes) {
        bpe_text(tok, text, ids);
        return true;
    }
    const auto& ph = tok.phonemes;
    const std::string &open = ph.open_marker, &close = ph.close_marker;
    if (text.find(open) == std::string::npos && text.find(close) == std::string::npos) {
        bpe_text(tok, text, ids);
        return true;
    }
    size_t pos = 0;
    while (pos < text.size()) {
        const size_t start = text.find(open, pos);
        const std::string stretch = start == std::string::npos ? text.substr(pos) : text.substr(pos, start - pos);
        if (stretch.find(close) != std::string::npos) {
            err = "Unbalanced phoneme group markers";
            return false;
        }
        if (!stretch.empty())
            bpe_text(tok, stretch, ids);
        if (start == std::string::npos)
            break;
        const size_t gstart = start + open.size();
        const size_t end = text.find(close, gstart);
        const std::string group = end != std::string::npos ? text.substr(gstart, end - gstart) : std::string();
        if (end == std::string::npos || group.empty() || group.find(open) != std::string::npos) {
            err = "Unbalanced or empty phoneme group";
            return false;
        }
        std::vector<std::string> unknown;
        std::vector<int32_t> phones;
        for (const auto& c : utf8_spans(group, 0, group.size())) {
            const std::string s = group.substr(c.b0, c.b1 - c.b0);
            const auto it = std::find(ph.symbols.begin(), ph.symbols.end(), s);
            if (it == ph.symbols.end())
                unknown.push_back(s);
            else
                phones.push_back(ph.open_id + 2 + (int32_t)(it - ph.symbols.begin()));
        }
        if (!unknown.empty()) {
            std::sort(unknown.begin(), unknown.end());
            unknown.erase(std::unique(unknown.begin(), unknown.end()), unknown.end());
            err = "Phones outside the inventory: [";
            for (size_t i = 0; i < unknown.size(); ++i)
                err += (i ? ", " : "") + py_char_repr(unknown[i]);
            err += "]";
            return false;
        }
        ids.push_back(ph.open_id);
        ids.insert(ids.end(), phones.begin(), phones.end());
        ids.push_back(ph.close_id);
        pos = end + close.size();
    }
    return true;
}

} // namespace

bool zonos3_payload_ids(const zonos3_tokenizer& tok, const std::string& text, std::vector<int32_t>& ids,
                        std::string& err) {
    if (text.find(ZONOS3_START_MARKER) != std::string::npos) {
        if (tok.start_id < 0) {
            err = NO_START_TOKEN;
            return false;
        }
        if (!start_first_once(text)) {
            err = START_NOT_FIRST;
            return false;
        }
        ids.push_back(tok.start_id);
        return zonos3_payload_ids(tok, text.substr(3), ids, err);
    }
    size_t at = find_pause(text, 0);
    if (at != std::string::npos) {
        if (tok.pause_ids.empty()) {
            err = NO_PAUSE_TOKENS;
            return false;
        }
        size_t pos = 0;
        for (; at != std::string::npos; at = find_pause(text, pos)) {
            const int b = pause_bucket(text, at);
            if (b >= (int)tok.pause_ids.size()) {
                err = pause_outside(b, tok.pause_ids.size());
                return false;
            }
            if (at > pos && !zonos3_payload_ids(tok, text.substr(pos, at - pos), ids, err))
                return false;
            ids.push_back(tok.pause_ids[b]);
            pos = at + 3;
        }
        if (pos < text.size())
            return zonos3_payload_ids(tok, text.substr(pos), ids, err);
        return true;
    }
    return phoneme_payload(tok, text, ids, err);
}

bool zonos3_text_ids(const zonos3_tokenizer& tok, const std::string& utf8, std::vector<int32_t>& out,
                     std::string& err) {
    out.assign(1, tok.bos);
    if (!zonos3_payload_ids(tok, utf8, out, err))
        return false;
    out.push_back(tok.eos);
    return true;
}

std::string zonos3_check_text_markers(const zonos3_tokenizer& tok, const std::string& utf8, bool edit) {
    const bool has_start = utf8.find(ZONOS3_START_MARKER) != std::string::npos;
    const bool pause = find_pause(utf8, 0) != std::string::npos;
    if (edit && (has_start || pause))
        return "Speech editing takes no start or pause markers: remove U+E002 / U+E003-U+E00A from target_text";
    const zonos3_phoneme_spec defaults;
    const auto& ph = tok.has_phonemes ? tok.phonemes : defaults;
    const bool has_groups =
        utf8.find(ph.open_marker) != std::string::npos || utf8.find(ph.close_marker) != std::string::npos;
    if (has_groups && !tok.has_phonemes)
        return "This checkpoint has no phoneme tokens: remove the U+E000 / U+E001 pronunciation groups";
    if (has_start && tok.start_id < 0)
        return NO_START_TOKEN;
    if (pause && tok.pause_ids.empty())
        return NO_PAUSE_TOKENS;
    if (has_start && !start_first_once(utf8))
        return START_NOT_FIRST;
    for (size_t at = find_pause(utf8, 0); at != std::string::npos; at = find_pause(utf8, at + 3)) {
        const int b = pause_bucket(utf8, at);
        if (b >= (int)tok.pause_ids.size())
            return pause_outside(b, tok.pause_ids.size());
    }
    if (has_groups) {
        std::vector<int32_t> ids;
        std::string err;
        if (!zonos3_payload_ids(tok, utf8, ids, err))
            return "Pronunciation groups: " + err;
    }
    return "";
}
