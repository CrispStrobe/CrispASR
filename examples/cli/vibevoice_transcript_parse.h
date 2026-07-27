// vibevoice_transcript_parse.h — turn VibeVoice-ASR's answer into segments.
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

#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace vibevoice_transcript {

struct Utterance {
    double start_s = -1.0; // <0 when the model omitted it
    double end_s = -1.0;   // <0 when the model omitted it
    int speaker = -1;      // <0 when the model omitted it
    std::string text;      // "Content", unescaped
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

} // namespace detail

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
            if (i < raw.size() && raw[i] == '"') {
                if (!detail::read_string(raw, i, val)) {
                    truncated = true;
                    break;
                }
            } else {
                val = detail::read_bare(raw, i);
            }

            if (detail::iequals(key, "Content") || detail::iequals(key, "Text")) {
                u.text = val;
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
        // A truncated object is dropped, but everything decoded before it is
        // kept — that is the whole reason this is not a strict JSON parse.
        if (truncated)
            break;
        if (has_content)
            out.push_back(std::move(u));
    }
    // "Content" is what makes this a transcript rather than some other JSON the
    // model happened to emit; with nothing carrying text, report no match so
    // the caller keeps the raw string.
    return out;
}

} // namespace vibevoice_transcript
