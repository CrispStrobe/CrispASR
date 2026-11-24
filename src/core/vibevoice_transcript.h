// core/vibevoice_transcript.h — turn VibeVoice-ASR's answer into segments.
//
// VibeVoice-ASR is prompted with "Start time, End time, Speaker ID, Content"
// (see src/vibevoice.cpp) and answers with a JSON array, one object per
// utterance:
//
//   [{"Start":0.0,"End":11.0,"Speaker":0,"Content":"And so, my fellow …"}]
//
// So the speaker turns and their timings ARE there — they were just never read
// (#300): the adapter handed the whole blob back as one segment's text, which
// left `seg.speaker` empty, printed raw JSON into `--stream`, dropped the
// per-utterance timings, and made the `--stream-json` "speaker" field
// unreachable for this backend.
//
// Why a hand parser rather than json.hpp: this is LLM output. It is usually
// well-formed, but a decode that hits the token cap ends mid-array, and a
// strict parse of a truncated blob throws away every COMPLETE utterance before
// the cut. The scanner below takes objects one at a time, so a truncated tail
// costs only the unfinished object. It also accepts the long key spellings the
// prompt itself uses ("Start time" / "Speaker ID"), which the model does emit.
//
// Weight-free and self-contained on purpose — tests/test-vibevoice-transcript.cpp
// covers it without a model, which is the tier CI actually runs.
//
// It lives in core/ rather than examples/cli/ because BOTH surfaces need it: the
// CLI adapter and the session C-ABI, which reimplements each backend's
// transcribe inline (docs/contributing.md point 6). A parse that landed only in
// the adapter would leave every binding — Python, Go, Flutter — still handing
// its callers the raw JSON blob.

#pragma once

#include "ngram_loop_fix.h"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace core_vibevoice {

struct Utterance {
    double start_s = -1.0;  // <0 when the model omitted it
    double end_s = -1.0;    // <0 when the model omitted it
    int speaker = -1;       // <0 when the model omitted it
    std::string text;       // "Content", unescaped, repetition loops collapsed
    std::string raw_text;   // "Content" exactly as decoded (token mapping)
    bool truncated = false; // Content was cut off by the token limit
};

namespace detail {

inline bool iequals(const std::string& a, const char* b) {
    size_t i = 0;
    for (; i < a.size() && b[i]; i++)
        if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i]))
            return false;
    return i == a.size() && b[i] == '\0';
}

// Append `cp` as UTF-8.
inline void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += (char)cp;
    } else if (cp < 0x800) {
        out += (char)(0xC0 | (cp >> 6));
        out += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += (char)(0xE0 | (cp >> 12));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    } else {
        out += (char)(0xF0 | (cp >> 18));
        out += (char)(0x80 | ((cp >> 12) & 0x3F));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    }
}

inline bool hex4(const std::string& s, size_t p, uint32_t& out) {
    if (p + 4 > s.size())
        return false;
    uint32_t v = 0;
    for (size_t i = 0; i < 4; i++) {
        const char c = s[p + i];
        v <<= 4;
        if (c >= '0' && c <= '9')
            v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f')
            v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F')
            v |= (uint32_t)(c - 'A' + 10);
        else
            return false;
    }
    out = v;
    return true;
}

// Read a JSON string starting at s[i] == '"'. Leaves `i` past the closing
// quote. Returns false if the string never closes (a truncated decode).
inline bool read_string(const std::string& s, size_t& i, std::string& out) {
    if (i >= s.size() || s[i] != '"')
        return false;
    i++;
    out.clear();
    while (i < s.size()) {
        const char c = s[i];
        if (c == '"') {
            i++;
            return true;
        }
        if (c != '\\') {
            out += c;
            i++;
            continue;
        }
        if (++i >= s.size())
            return false;
        const char e = s[i++];
        switch (e) {
        case '"':
            out += '"';
            break;
        case '\\':
            out += '\\';
            break;
        case '/':
            out += '/';
            break;
        case 'b':
            out += '\b';
            break;
        case 'f':
            out += '\f';
            break;
        case 'n':
            out += '\n';
            break;
        case 'r':
            out += '\r';
            break;
        case 't':
            out += '\t';
            break;
        case 'u': {
            uint32_t cp = 0;
            if (!hex4(s, i, cp))
                return false;
            i += 4;
            // Surrogate pair.
            if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                uint32_t lo = 0;
                if (hex4(s, i + 2, lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    i += 6;
                }
            }
            append_utf8(out, cp);
            break;
        }
        default:
            // Unknown escape: keep the character, don't lose text.
            out += e;
            break;
        }
    }
    return false; // unterminated
}

// Read a bare (unquoted) scalar — number, true/false/null — up to the next
// ',' or '}'. Leaves `i` on that delimiter.
inline std::string read_bare(const std::string& s, size_t& i) {
    const size_t b = i;
    while (i < s.size() && s[i] != ',' && s[i] != '}')
        i++;
    std::string v = s.substr(b, i - b);
    while (!v.empty() && (unsigned char)v.back() <= ' ')
        v.pop_back();
    return v;
}

// Leading integer of a speaker value: 0, "0", "SPEAKER_02", "Speaker 1".
inline int parse_speaker(const std::string& v) {
    for (size_t i = 0; i < v.size(); i++) {
        if (std::isdigit((unsigned char)v[i])) {
            return (int)std::strtol(v.c_str() + i, nullptr, 10);
        }
    }
    return -1;
}

inline bool parse_seconds(const std::string& v, double& out) {
    if (v.empty())
        return false;
    char* end = nullptr;
    const double d = std::strtod(v.c_str(), &end);
    if (end == v.c_str())
        return false;
    out = d;
    return true;
}

// Collapse a repetition loop (core_ngram::fix_loops). fix_loops re-joins
// words with single spaces, so its output is used only when it actually
// removed something; clean Content passes through byte for byte.
inline std::string collapse_loops(const std::string& text) {
    std::string fixed = core_ngram::fix_loops(text);
    std::string normalized;
    for (const std::string& w : core_ngram::split_words(text)) {
        if (!normalized.empty())
            normalized += ' ';
        normalized += w;
    }
    return fixed == normalized ? text : fixed;
}

} // namespace detail

// True when an utterance's Content is one of the model's own NON-SPEECH
// markers rather than a transcript.
//
// VibeVoice-ASR is trained on JSON transcripts that label non-speech regions
// instead of transcribing them, so "[Silence]" is a Content value the MODEL
// emits — it appears nowhere in this codebase. Handing it through as segment
// text puts the literal string "[Silence]" into the user's SRT for audio that
// is plainly not silent, and, worse, makes it invisible: the CLI's "no text
// produced for N s of non-silent audio" warning cannot fire, because there IS
// text. Observed on the 7B checkpoint for heavily time-stretched speech
// (#369), where a passage inside a long recording would be dropped in silence.
//
// Deliberately narrow. The list is what has actually been observed, not what
// might exist: a marker is only recognised when the WHOLE content is a single
// bracketed token that matches. "[Music]" and "[Laughter]" are left alone —
// those are annotations a user may legitimately want in a transcript, and
// guessing at the model's full vocabulary here would silently delete content.
inline bool is_non_speech_marker(const std::string& text) {
    size_t b = 0, e = text.size();
    while (b < e && (unsigned char)text[b] <= ' ')
        b++;
    while (e > b && (unsigned char)text[e - 1] <= ' ')
        e--;
    if (e - b < 2 || text[b] != '[' || text[e - 1] != ']')
        return false;
    std::string inner = text.substr(b + 1, e - b - 2);
    // Trailing punctuation the model sometimes appends inside the object.
    while (!inner.empty() && (inner.back() == '.' || (unsigned char)inner.back() <= ' '))
        inner.pop_back();
    static const char* kMarkers[] = {"silence", "blank_audio", "blank audio", "no speech", "inaudible"};
    for (const char* m : kMarkers)
        if (detail::iequals(inner, m))
            return true;
    return false;
}

// Scan `raw` for utterance objects. Returns them in emission order; an empty
// result means "this is not a VibeVoice transcript blob" and the caller should
// fall back to treating `raw` as plain text.
inline std::vector<Utterance> parse(const std::string& raw) {
    std::vector<Utterance> out;
    size_t i = 0;
    while (i < raw.size()) {
        if (raw[i] != '{') {
            i++;
            continue;
        }
        i++; // past '{'
        Utterance u;
        bool has_content = false;
        bool truncated = false;
        while (i < raw.size()) {
            while (i < raw.size() && (unsigned char)raw[i] <= ' ')
                i++;
            if (i < raw.size() && (raw[i] == ',' || raw[i] == ':')) {
                i++;
                continue;
            }
            if (i >= raw.size() || raw[i] == '}') {
                i = (i < raw.size()) ? i + 1 : i;
                break;
            }
            std::string key;
            if (raw[i] == '"') {
                if (!detail::read_string(raw, i, key)) {
                    truncated = true;
                    break;
                }
            } else {
                // Not a quoted key — a nested structure or garbage. Skip the
                // object rather than guess.
                truncated = true;
                break;
            }
            while (i < raw.size() && ((unsigned char)raw[i] <= ' ' || raw[i] == ':'))
                i++;
            std::string val;
            const bool is_content = detail::iequals(key, "Content") || detail::iequals(key, "Text");
            if (i < raw.size() && raw[i] == '"') {
                if (!detail::read_string(raw, i, val)) {
                    // #610: a decode that ends inside a Content (usually a
                    // repetition loop running into the token limit) still
                    // carries everything said before the loop. Keep it.
                    if (is_content) {
                        u.raw_text = val;
                        u.truncated = true;
                        has_content = true;
                    }
                    truncated = true;
                    break;
                }
            } else {
                val = detail::read_bare(raw, i);
            }

            if (is_content) {
                u.raw_text = val;
                has_content = true;
            } else if (detail::iequals(key, "Speaker") || detail::iequals(key, "Speaker ID") ||
                       detail::iequals(key, "SpeakerID")) {
                u.speaker = detail::parse_speaker(val);
            } else if (detail::iequals(key, "Start") || detail::iequals(key, "Start time") ||
                       detail::iequals(key, "StartTime")) {
                detail::parse_seconds(val, u.start_s);
            } else if (detail::iequals(key, "End") || detail::iequals(key, "End time") ||
                       detail::iequals(key, "EndTime")) {
                detail::parse_seconds(val, u.end_s);
            }
        }
        // Everything decoded before a truncation is kept — that is the whole
        // reason this is not a strict JSON parse. A cut-off object is kept
        // only when the cut fell inside its Content (#610).
        if (has_content && (!truncated || u.truncated)) {
            u.text = detail::collapse_loops(u.raw_text);
            out.push_back(std::move(u));
        }
        if (truncated)
            break;
    }
    // "Content" is what makes this a transcript rather than some other JSON the
    // model happened to emit; with nothing carrying text, report no match so
    // the caller keeps the raw string.
    return out;
}

// True when `raw` is a VibeVoice transcript answer even if parse() found no
// usable utterance in it (cut off before any Content). Callers must then
// report no text rather than hand the raw JSON back as the transcript (#610).
inline bool looks_like_transcript(const std::string& raw) {
    size_t i = 0;
    while (i < raw.size() && (unsigned char)raw[i] <= ' ')
        i++;
    if (i < raw.size() && raw[i] == '[') {
        i++;
        while (i < raw.size() && (unsigned char)raw[i] <= ' ')
            i++;
        if (i >= raw.size() || raw[i] == '{')
            return true;
    }
    return raw.find("\"Content\"") != std::string::npos;
}

// Assign decode tokens to utterances.
//
// The session ABI exposes a per-token confidence list alongside the transcript.
// Once the answer is split into utterances, that list has to be split the same
// way — otherwise a caller reading segment 2's "words" gets tokens belonging to
// segment 0, plus every `[`, `{` and `"Speaker"` of the JSON scaffolding.
//
// The mapping is exact rather than heuristic: each Content is a verbatim
// substring of the concatenated token texts (the parser only unescapes, and the
// escapes below are handled by searching for the raw form first). So walk the
// tokens once, accumulating character offsets, and find each Content's span in
// that same string, resuming each search where the previous one ended. A token
// belongs to the utterance whose span it overlaps; scaffolding tokens overlap
// nothing and are dropped, which is the point.
//
// `token_texts` must be in decode order. Returns one index list per utterance
// (parallel to `utts`); an utterance whose Content could not be located — a
// heavily escaped Content, say — gets an empty list rather than a wrong one.
inline std::vector<std::vector<int>> assign_tokens(const std::vector<Utterance>& utts,
                                                   const std::vector<std::string>& token_texts) {
    std::vector<std::vector<int>> out(utts.size());
    std::string joined;
    std::vector<size_t> tok_start(token_texts.size(), 0);
    for (size_t i = 0; i < token_texts.size(); i++) {
        tok_start[i] = joined.size();
        joined += token_texts[i];
    }
    size_t search_from = 0;
    for (size_t u = 0; u < utts.size(); u++) {
        const std::string& needle = utts[u].raw_text.empty() ? utts[u].text : utts[u].raw_text;
        if (needle.empty())
            continue;
        size_t pos = joined.find(needle, search_from);
        if (pos == std::string::npos) {
            // Fall back to a search from the top — the model can repeat a line
            // (our own multispeaker fixture does), so a miss ahead of the
            // cursor is worth one retry before giving up on this utterance.
            pos = joined.find(needle);
            if (pos == std::string::npos)
                continue;
        }
        const size_t end = pos + needle.size();
        for (size_t i = 0; i < token_texts.size(); i++) {
            const size_t s = tok_start[i];
            const size_t e = s + token_texts[i].size();
            if (e > pos && s < end) // overlap, half-open
                out[u].push_back((int)i);
        }
        search_from = end;
    }
    return out;
}

} // namespace core_vibevoice
