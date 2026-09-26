// crispasr_output.cpp — output writers for non-whisper backends.
//
// Extracted / generalized from parakeet-main/main.cpp and cohere-main.cpp.

#include "crispasr_output.h"

#include "core/asr_time_order.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <cstdlib>

// ---------------------------------------------------------------------------
// Timestamp + path helpers
// ---------------------------------------------------------------------------

std::string crispasr_to_timestamp(int64_t cs, bool comma) {
    int64_t msec = cs * 10;
    const int64_t hr = msec / (1000 * 60 * 60);
    msec -= hr * (1000 * 60 * 60);
    const int64_t min = msec / (1000 * 60);
    msec -= min * (1000 * 60);
    const int64_t sec = msec / 1000;
    msec -= sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int)hr, (int)min, (int)sec, comma ? "," : ".", (int)msec);
    return buf;
}

std::string crispasr_make_out_path(const std::string& audio, const std::string& ext) {
    std::string base = audio;
    static const char* exts[] = {
        ".wav", ".WAV",  ".mp3",  ".MP3", ".flac", ".FLAC", ".ogg",  ".OGG", ".m4a",
        ".M4A", ".opus", ".OPUS", ".mp4", ".MP4",  ".webm", ".WEBM", ".aac", ".AAC",
    };
    for (const char* e : exts) {
        const size_t el = strlen(e);
        if (base.size() > el && base.compare(base.size() - el, el, e) == 0) {
            base.resize(base.size() - el);
            break;
        }
    }
    return base + ext;
}

// ---------------------------------------------------------------------------
// Display segment builder
// ---------------------------------------------------------------------------

// Check if position i in text is a sentence-ending punctuation.
// Handles ASCII (. ! ?) and full-width CJK punctuation (。？！).
// Returns the byte length of the punctuation mark (1 for ASCII, 3 for UTF-8 CJK).
static int is_sentence_end_at(const std::string& text, size_t i) {
    char c = text[i];
    // ASCII sentence enders
    if (c == '.' || c == '!' || c == '?')
        return 1;
    // Full-width CJK: 。(E3 80 82), ？(EF BC 9F), ！(EF BC 81), ．(EF BC 8E)
    if (i + 2 < text.size()) {
        unsigned char b0 = (unsigned char)text[i];
        unsigned char b1 = (unsigned char)text[i + 1];
        unsigned char b2 = (unsigned char)text[i + 2];
        if (b0 == 0xE3 && b1 == 0x80 && b2 == 0x82)
            return 3; // 。
        if (b0 == 0xEF && b1 == 0xBC && b2 == 0x9F)
            return 3; // ？
        if (b0 == 0xEF && b1 == 0xBC && b2 == 0x81)
            return 3; // ！
    }
    return 0;
}

// Check if position i is a CJK comma/clause marker (soft break point).
// Returns byte length of the marker (3 for CJK), 0 if not.
static int is_clause_break_at(const std::string& text, size_t i) {
    if (i + 2 < text.size()) {
        unsigned char b0 = (unsigned char)text[i];
        unsigned char b1 = (unsigned char)text[i + 1];
        unsigned char b2 = (unsigned char)text[i + 2];
        // 、(E3 80 81) — Japanese/Chinese comma (読点)
        if (b0 == 0xE3 && b1 == 0x80 && b2 == 0x81)
            return 3;
        // ，(EF BC 8C) — full-width comma
        if (b0 == 0xEF && b1 == 0xBC && b2 == 0x8C)
            return 3;
    }
    return 0;
}

// Returns true if the text contains at least one CJK/Hangul/kana character.
// Used to guard the 42-char split fallback in split_text_at_punct, which is
// designed for CJK text that lacks sentence-ending punctuation (#29).
// Without this guard the fallback fires on long English sentences and splits
// words at raw character boundaries (e.g. "about" → "abo" / "ut", #150).
static bool text_has_cjk(const std::string& text) {
    for (size_t i = 0; i < text.size();) {
        unsigned char b0 = (unsigned char)text[i];
        // 3-byte UTF-8 sequences with first byte 0xE3–0xEF cover U+3000–U+FFFF,
        // which includes all CJK Unified Ideographs, Hiragana, Katakana, and Hangul.
        // Latin text (including accented Latin) only uses 1- and 2-byte sequences
        // (first byte ≤ 0xDF, i.e. U+0000–U+07FF).
        if (b0 >= 0xE3 && b0 < 0xF0)
            return true;
        if (b0 < 0x80)
            i += 1;
        else if (b0 < 0xE0)
            i += 2;
        else if (b0 < 0xF0)
            i += 3; // 0xE0–0xE2: non-CJK 3-byte (Samaritan, arrows, math, …)
        else
            i += 4;
    }
    return false;
}

// Count UTF-8 codepoints in a string
static int utf8_len(const std::string& s) {
    int n = 0;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x80)
            i += 1;
        else if (c < 0xE0)
            i += 2;
        else if (c < 0xF0)
            i += 3;
        else
            i += 4;
        n++;
    }
    return n;
}

// Split a long text into sentences at punctuation boundaries.
// Returns pairs of (sentence_text, approximate_fraction_through_segment).
// For CJK text without sentence-ending punctuation (e.g. many ASR backends
// don't produce 。for Japanese), falls back to splitting at clause breaks
// (、，) or at ~42 CJK characters — a typical subtitle line length.
static std::vector<std::pair<std::string, float>> split_text_at_punct(const std::string& text) {
    std::vector<std::pair<std::string, float>> sentences;
    size_t start = 0;
    size_t len = text.size();

    for (size_t i = 0; i < len; i++) {
        int plen = is_sentence_end_at(text, i);
        // Sentence end: punctuation followed by space, next char, or end of text.
        // For CJK, no space is needed — the punctuation mark itself is the boundary.
        if (plen > 0 && (i + plen >= len || text[i + plen] == ' ' || plen == 3)) {
            i += plen - 1; // advance past the punctuation bytes
            size_t end = i + 1;
            std::string sentence = text.substr(start, end - start);
            // Trim leading whitespace
            size_t first = sentence.find_first_not_of(" \t");
            if (first != std::string::npos)
                sentence = sentence.substr(first);
            if (!sentence.empty()) {
                float frac = (float)(end) / (float)len;
                sentences.push_back({sentence, frac});
            }
            start = end;
        }
    }
    // Remainder (text after last punctuation)
    if (start < len) {
        std::string remainder = text.substr(start);
        size_t first = remainder.find_first_not_of(" \t");
        if (first != std::string::npos)
            remainder = remainder.substr(first);
        if (!remainder.empty())
            sentences.push_back({remainder, 1.0f});
    }

    // Fallback for CJK text without sentence-ending punctuation (#29):
    // If we got <=1 sentence and the text is long (>42 codepoints),
    // re-split at clause breaks (、，) or at ~42 character boundaries.
    // Guard with text_has_cjk: long English sentences with only a trailing
    // period must not be split at raw character boundaries (#150).
    if (sentences.size() <= 1 && utf8_len(text) > 42 && text_has_cjk(text)) {
        sentences.clear();
        start = 0;
        int chars_since_split = 0;
        size_t last_clause_break = std::string::npos;

        for (size_t i = 0; i < len;) {
            int clen = is_clause_break_at(text, i);
            if (clen > 0)
                last_clause_break = i + clen;

            unsigned char c = (unsigned char)text[i];
            int cplen = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
            i += cplen;
            chars_since_split++;

            bool do_split = false;
            size_t split_at = i;

            if (chars_since_split >= 42) {
                // Force split — prefer last clause break if available
                if (last_clause_break != std::string::npos && last_clause_break > start)
                    split_at = last_clause_break;
                do_split = true;
            } else if (clen > 0 && chars_since_split >= 20) {
                // Split at clause break after ≥20 chars
                split_at = last_clause_break;
                do_split = true;
            }

            if (do_split && split_at > start) {
                std::string chunk = text.substr(start, split_at - start);
                size_t first = chunk.find_first_not_of(" \t");
                if (first != std::string::npos)
                    chunk = chunk.substr(first);
                if (!chunk.empty())
                    sentences.push_back({chunk, (float)split_at / (float)len});
                start = split_at;
                chars_since_split = 0;
                last_clause_break = std::string::npos;
            }
        }
        if (start < len) {
            std::string remainder = text.substr(start);
            size_t first = remainder.find_first_not_of(" \t");
            if (first != std::string::npos)
                remainder = remainder.substr(first);
            if (!remainder.empty())
                sentences.push_back({remainder, 1.0f});
        }
    }

    return sentences;
}

// True if `text` contains at least two sentence-ending marks (so splitting
// would produce multiple sentences). Used to decide between text-level and
// word-level splitting.
static int count_sentence_ends(const std::string& text) {
    int n = 0;
    for (size_t i = 0; i < text.size();) {
        int plen = is_sentence_end_at(text, i);
        if (plen > 0) {
            n++;
            i += plen;
        } else {
            i++;
        }
    }
    return n;
}

// True if any word in `words` ends with a sentence-ending punctuation mark.
// Used to detect when word-level splitting would actually produce sentence
// boundaries — false for tokenizers that don't carry punctuation as a word
// suffix (e.g. cohere with CJK input groups everything into one pseudo-word).
static bool words_carry_sentence_ends(const std::vector<crispasr_word>& words) {
    for (const auto& w : words) {
        if (w.text.empty())
            continue;
        const size_t n = w.text.size();
        if (n >= 1 && is_sentence_end_at(w.text, n - 1) == 1)
            return true;
        if (n >= 3 && is_sentence_end_at(w.text, n - 3) == 3)
            return true;
    }
    return false;
}

// Split `text` on ASCII whitespace into tokens (no empty tokens).
static std::vector<std::string> split_ws_words(const std::string& s) {
    std::vector<std::string> w;
    std::string cur;
    for (char c : s) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!cur.empty()) {
                w.push_back(cur);
                cur.clear();
            }
        } else {
            cur += c;
        }
    }
    if (!cur.empty())
        w.push_back(cur);
    return w;
}

// Append `tok` to `out`, broken at UTF-8 codepoint boundaries into pieces of
// at most `max_len` bytes. Used for a single token longer than max_len and for
// space-less scripts (CJK), where word packing can't break the text.
static void split_long_token(const std::string& tok, int max_len, std::vector<std::string>& out) {
    std::string cur;
    for (size_t i = 0; i < tok.size();) {
        size_t cp = 1;
        unsigned char b = (unsigned char)tok[i];
        if (b >= 0xF0)
            cp = 4;
        else if (b >= 0xE0)
            cp = 3;
        else if (b >= 0xC0)
            cp = 2;
        if (i + cp > tok.size())
            cp = tok.size() - i;
        if (!cur.empty() && (int)(cur.size() + cp) > max_len) {
            out.push_back(cur);
            cur.clear();
        }
        cur.append(tok, i, cp);
        i += cp;
    }
    if (!cur.empty())
        out.push_back(cur);
}

// Greedily pack `text` into lines of at most `max_len` bytes on whitespace word
// boundaries (over-long tokens are split at codepoint boundaries). For text-only
// backends that carry no word timings this is how --max-len is honoured (#205).
static std::vector<std::string> pack_text_to_maxlen(const std::string& text, int max_len) {
    std::vector<std::string> out;
    auto words = split_ws_words(text);
    if (words.empty()) {
        if (!text.empty())
            split_long_token(text, max_len, out);
        return out;
    }
    std::string cur;
    for (const auto& w : words) {
        if ((int)w.size() > max_len) {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
            split_long_token(w, max_len, out);
            continue;
        }
        const size_t sep = cur.empty() ? 0 : 1;
        if (!cur.empty() && (int)(cur.size() + sep + w.size()) > max_len) {
            out.push_back(cur);
            cur.clear();
        }
        if (!cur.empty())
            cur += ' ';
        cur += w;
    }
    if (!cur.empty())
        out.push_back(cur);
    return out;
}

// ---------------------------------------------------------------------------
// Issue #356: time-order guard
//
// The predicate itself lives in src/core/asr_time_order.h because the session
// C ABI needs the same check on a different segment struct and cannot include
// anything from examples/cli. These are the CLI/server-side names for it.
// ---------------------------------------------------------------------------

int crispasr_first_backward_segment(const std::vector<crispasr_segment>& segments, int64_t* prev_cs, int64_t* cur_cs) {
    return core_time_order::first_backward(segments, prev_cs, cur_cs);
}

void crispasr_warn_if_segments_backward(const std::vector<crispasr_segment>& segments, const char* where) {
    core_time_order::warn_if_backward(segments, where);
}

std::vector<crispasr_disp_segment> crispasr_make_disp_segments(const std::vector<crispasr_segment>& segments,
                                                               int max_len, bool split_on_punct) {
    std::vector<crispasr_disp_segment> out;

    for (const auto& seg : segments) {
        // Easy path: no word data, or no splitting/max_len requested.
        // Also forced when split_on_punct is on but the words don't carry
        // sentence-ending punctuation — e.g. cohere on Japanese, where every
        // token is grouped into one giant pseudo-word so the word-packing
        // splitter never sees a 。/？/！ at a word boundary (#29).
        const bool punct_in_text = split_on_punct && count_sentence_ends(seg.text) >= 2;
        const bool words_unhelpful = punct_in_text && !words_carry_sentence_ends(seg.words);
        if (seg.words.empty() || words_unhelpful || (max_len == 0 && !split_on_punct)) {
            if (seg.text.empty())
                continue;

            // No usable word timings. When max_len <= 0, keep the historical
            // behaviour (split_on_punct splits sentences with frac-accurate
            // timing; otherwise emit the whole text as one segment).
            if (max_len <= 0) {
                if (split_on_punct) {
                    auto sentences = split_text_at_punct(seg.text);
                    if (sentences.size() <= 1) {
                        out.push_back({seg.t0, seg.t1, seg.text, seg.speaker});
                    } else {
                        float prev_frac = 0.0f;
                        int64_t duration = seg.t1 - seg.t0;
                        for (const auto& [sent, frac] : sentences) {
                            int64_t s_t0 = seg.t0 + (int64_t)(prev_frac * duration);
                            int64_t s_t1 = seg.t0 + (int64_t)(frac * duration);
                            out.push_back({s_t0, s_t1, sent, seg.speaker});
                            prev_frac = frac;
                        }
                    }
                } else {
                    out.push_back({seg.t0, seg.t1, seg.text, seg.speaker});
                }
                continue;
            }

            // max_len > 0 but no word timings (#205: granite/qwen3 non-plus
            // return text-only segments — --max-len used to be a no-op). Split
            // the TEXT instead: optionally at sentence boundaries first, then
            // pack to <= max_len bytes (max_len == 1 → one line per word), and
            // interpolate timestamps by cumulative character length.
            std::vector<std::string> sentences;
            if (split_on_punct) {
                for (const auto& sf : split_text_at_punct(seg.text))
                    sentences.push_back(sf.first);
            }
            if (sentences.empty())
                sentences.push_back(seg.text);

            std::vector<std::string> pieces;
            for (const auto& sent : sentences) {
                if (max_len == 1) {
                    for (const auto& wd : split_ws_words(sent))
                        pieces.push_back(wd);
                } else {
                    for (auto& chunk : pack_text_to_maxlen(sent, max_len))
                        pieces.push_back(std::move(chunk));
                }
            }
            if (pieces.empty()) {
                out.push_back({seg.t0, seg.t1, seg.text, seg.speaker});
                continue;
            }

            const int64_t duration = seg.t1 - seg.t0;
            size_t total = 0;
            for (const auto& p : pieces)
                total += p.size();
            if (total == 0)
                total = 1;
            size_t acc = 0;
            for (auto& p : pieces) {
                int64_t s_t0 = seg.t0 + (int64_t)((double)acc / (double)total * (double)duration);
                acc += p.size();
                int64_t s_t1 = seg.t0 + (int64_t)((double)acc / (double)total * (double)duration);
                out.push_back({s_t0, s_t1, std::move(p), seg.speaker});
            }
            continue;
        }

        // max_len == 1 means one display segment per word.
        if (max_len == 1) {
            for (const auto& w : seg.words) {
                out.push_back({w.t0, w.t1, w.text, seg.speaker});
            }
            continue;
        }

        // max_len > 1 or split_on_punct: pack words into segments.
        crispasr_disp_segment cur;
        cur.t0 = -1;
        cur.speaker = seg.speaker;

        auto flush = [&]() {
            // A line may open on a word that carries its leading space
            // (whisper's " word", the aligner's Hangul " 오", #465).
            if (!cur.text.empty() && cur.text[0] == ' ')
                cur.text.erase(0, 1);
            if (!cur.text.empty())
                out.push_back(cur);
            cur = {};
            cur.t0 = -1;
            cur.speaker = seg.speaker;
        };

        for (const auto& w : seg.words) {
            // No space separator for CJK characters (Japanese/Chinese/Korean have no word spaces)
            bool prev_is_cjk = false;
            bool cur_is_cjk = false;
            if (!cur.text.empty()) {
                // Check last codepoint of current text
                size_t last = cur.text.size() - 1;
                while (last > 0 && (cur.text[last] & 0xC0) == 0x80)
                    last--;
                unsigned char b = (unsigned char)cur.text[last];
                prev_is_cjk = (b >= 0xE0); // CJK chars are 3+ bytes in UTF-8
            }
            if (!w.text.empty()) {
                unsigned char b = (unsigned char)w.text[0];
                cur_is_cjk = (b >= 0xE0); // 3+ byte UTF-8 = likely CJK
            }
            // A word that carries its own leading space (whisper " word", the
            // aligner's Hangul " 오") needs no separator - like the other
            // rebuild sites' already_spaced (this one doubled it: "hello  world").
            const bool already_spaced = !w.text.empty() && w.text[0] == ' ';
            const std::string sep = (cur.text.empty() || already_spaced) ? "" : (prev_is_cjk || cur_is_cjk) ? "" : " ";
            // -ml counts CHARACTERS, not bytes (#465: -ml 10 held ~3 Hangul
            // syllables). And a Hangul syllable without a leading space
            // continues the previous word, so do not break there - unless the
            // line has already run to twice the limit (one enormous word).
            auto n_chars = [](const std::string& s) {
                int n = 0;
                for (unsigned char c : s)
                    n += (c & 0xC0) != 0x80;
                return n;
            };
            auto is_hangul_first = [](const std::string& s) {
                if (s.size() < 3)
                    return false;
                const unsigned char b0 = (unsigned char)s[0], b1 = (unsigned char)s[1], b2 = (unsigned char)s[2];
                if ((b0 & 0xF0) != 0xE0)
                    return false;
                const uint32_t cp = ((b0 & 0x0Fu) << 12) | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
                return (cp >= 0xAC00 && cp <= 0xD7AF) || (cp >= 0x1100 && cp <= 0x11FF) ||
                       (cp >= 0x3130 && cp <= 0x318F);
            };
            const int line_chars = n_chars(cur.text) + n_chars(sep) + n_chars(w.text);
            const bool mid_word = is_hangul_first(w.text) && prev_is_cjk;
            const bool would_overflow =
                max_len > 1 && !cur.text.empty() && line_chars > max_len && (!mid_word || line_chars > 2 * max_len);

            // Split at sentence-ending punctuation. Check BEFORE updating
            // cur.t1 so the flushed sentence keeps its last word's end time,
            // not the next word's end time.
            const bool at_sentence_end =
                split_on_punct && !cur.text.empty() && !w.text.empty() &&
                (cur.text.size() >= 1 && is_sentence_end_at(cur.text, cur.text.size() - 1) == 1 ||
                 cur.text.size() >= 3 && is_sentence_end_at(cur.text, cur.text.size() - 3) == 3);

            if (would_overflow || at_sentence_end) {
                flush();
            }

            if (cur.t0 < 0)
                cur.t0 = w.t0;
            cur.t1 = w.t1;
            cur.text += sep + w.text;
        }
        flush();
    }
    return out;
}

// ---------------------------------------------------------------------------
// Writers
// ---------------------------------------------------------------------------

static const char* prefix_speaker(const std::string& speaker) {
    return speaker.empty() ? "" : speaker.c_str();
}

bool crispasr_write_txt(const std::string& path, const std::vector<crispasr_disp_segment>& segs) {
    std::ofstream f(path);
    if (!f) {
        fprintf(stderr, "crispasr: warning: cannot write TXT '%s'\n", path.c_str());
        return false;
    }
    for (const auto& s : segs) {
        f << prefix_speaker(s.speaker) << s.text << "\n";
    }
    return true;
}

bool crispasr_write_srt(const std::string& path, const std::vector<crispasr_disp_segment>& segs) {
    std::ofstream f(path);
    if (!f) {
        fprintf(stderr, "crispasr: warning: cannot write SRT '%s'\n", path.c_str());
        return false;
    }
    for (size_t i = 0; i < segs.size(); i++) {
        f << (i + 1) << "\n"
          << crispasr_to_timestamp(segs[i].t0, /*comma=*/true) << " --> "
          << crispasr_to_timestamp(segs[i].t1, /*comma=*/true) << "\n"
          << prefix_speaker(segs[i].speaker) << segs[i].text << "\n\n";
    }
    return true;
}

bool crispasr_write_vtt(const std::string& path, const std::vector<crispasr_disp_segment>& segs) {
    std::ofstream f(path);
    if (!f) {
        fprintf(stderr, "crispasr: warning: cannot write VTT '%s'\n", path.c_str());
        return false;
    }
    f << "WEBVTT\n\n";
    for (const auto& s : segs) {
        f << crispasr_to_timestamp(s.t0) << " --> " << crispasr_to_timestamp(s.t1) << "\n"
          << prefix_speaker(s.speaker) << s.text << "\n\n";
    }
    return true;
}

// Escape a cell for CSV (RFC 4180: quote if it contains comma, quote, or
// newline; double-up quotes inside).
static std::string csv_escape(const std::string& s) {
    bool needs_quoting = false;
    for (char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            needs_quoting = true;
            break;
        }
    }
    if (!needs_quoting)
        return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"')
            out += "\"\"";
        else
            out += c;
    }
    out += "\"";
    return out;
}

bool crispasr_write_csv(const std::string& path, const std::vector<crispasr_disp_segment>& segs) {
    std::ofstream f(path);
    if (!f) {
        fprintf(stderr, "crispasr: warning: cannot write CSV '%s'\n", path.c_str());
        return false;
    }
    f << "start,end,text\n";
    for (const auto& s : segs) {
        f << (s.t0 * 10) << "," << (s.t1 * 10) << "," << csv_escape(s.text) << "\n";
    }
    return true;
}

// Minimal JSON escape (RFC 8259): backslash, quote, control chars.
// Exposed publicly as crispasr_json_escape(); the static alias keeps
// call sites in this file short.
std::string crispasr_json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (unsigned char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20) {
                char buf[8];
                snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += (char)c;
            }
        }
    }
    return out;
}

// Internal short alias.
static inline const std::string json_escape(const std::string& s) {
    return crispasr_json_escape(s);
}

bool crispasr_write_json(const std::string& path, const std::vector<crispasr_segment>& segs,
                         const std::string& backend_name, const std::string& model_path, const std::string& language,
                         bool full, const crispasr_lid_info* lid) {
    std::ofstream f(path);
    if (!f) {
        fprintf(stderr, "crispasr: warning: cannot write JSON '%s'\n", path.c_str());
        return false;
    }
    f << "{\n";
    f << "  \"crispasr\": {\n";
    f << "    \"backend\": \"" << json_escape(backend_name) << "\",\n";
    f << "    \"model\":   \"" << json_escape(model_path) << "\",\n";
    f << "    \"language\":\"" << json_escape(language) << "\"";
    if (lid && !lid->lang_code.empty()) {
        f << ",\n    \"language_detected\": \"" << json_escape(lid->lang_code) << "\",\n";
        f << "    \"language_confidence\": " << lid->confidence << ",\n";
        f << "    \"language_source\": \"" << json_escape(lid->source) << "\"";
    }
    f << "\n  },\n";
    f << "  \"transcription\": [\n";
    for (size_t i = 0; i < segs.size(); i++) {
        const auto& s = segs[i];
        f << "    {\n";
        f << "      \"timestamps\": { \"from\": \"" << crispasr_to_timestamp(s.t0, true) << "\", \"to\": \""
          << crispasr_to_timestamp(s.t1, true) << "\" },\n";
        f << "      \"offsets\":    { \"from\": " << (s.t0 * 10) << ", \"to\": " << (s.t1 * 10) << " },\n";
        if (!s.speaker.empty()) {
            f << "      \"speaker\": \"" << json_escape(s.speaker) << "\",\n";
        }
        f << "      \"text\":       \"" << json_escape(s.text) << "\"";
        // Multi-task ASR metadata (SenseVoice and similar). Emit any
        // non-empty fields after `text`.
        if (!s.lang_id.empty())
            f << ",\n      \"language\":   \"" << json_escape(s.lang_id) << "\"";
        if (!s.audio_event.empty())
            f << ",\n      \"audio_event\":\"" << json_escape(s.audio_event) << "\"";
        // No "emotion" key by design — see crispasr_backend.h / docs/eu-ai-act.md.
        if (!s.itn_flag.empty())
            f << ",\n      \"itn_flag\":   \"" << json_escape(s.itn_flag) << "\"";
        // #292: chunk scope. Only present on multi-chunk runs (single-pass leaves
        // chunk_id at -1). Diarize "(speaker N)" labels are chunk-local, so a
        // consumer must not assume "speaker 1" in different chunk_ids is the same
        // person — segments sharing a chunk_id share a speaker numbering.
        if (s.chunk_id >= 0)
            f << ",\n      \"chunk_id\":   " << s.chunk_id;
        // NOTE ON UNITS (issue #228): segment `offsets` above are in milliseconds
        // (t0/t1 are the internal centisecond timebase * 10). For back-compat the
        // per-word / per-token `t0`/`t1` fields are kept in their original
        // centisecond units, and each entry additionally carries an `offsets`
        // object in milliseconds so every timed object in this document exposes a
        // consistent ms field matching the segment shape.
        if (full && !s.words.empty()) {
            f << ",\n      \"words\": [\n";
            for (size_t j = 0; j < s.words.size(); j++) {
                const auto& w = s.words[j];
                f << "        { \"text\": \"" << json_escape(w.text) << "\", \"t0\": " << w.t0 << ", \"t1\": " << w.t1
                  << ", \"offsets\": { \"from\": " << (w.t0 * 10) << ", \"to\": " << (w.t1 * 10) << " } }"
                  << (j + 1 < s.words.size() ? "," : "") << "\n";
            }
            f << "      ]";
        }
        if (full && !s.tokens.empty()) {
            f << ",\n      \"tokens\": [\n";
            for (size_t j = 0; j < s.tokens.size(); j++) {
                const auto& t = s.tokens[j];
                f << "        { \"text\": \"" << json_escape(t.text) << "\", \"p\": " << t.confidence
                  << ", \"t0\": " << t.t0 << ", \"t1\": " << t.t1 << ", \"offsets\": { \"from\": " << (t.t0 * 10)
                  << ", \"to\": " << (t.t1 * 10) << " } }" << (j + 1 < s.tokens.size() ? "," : "") << "\n";
            }
            f << "      ]";
        }
        f << "\n    }" << (i + 1 < segs.size() ? "," : "") << "\n";
    }
    f << "  ]\n";
    f << "}\n";
    return true;
}

bool crispasr_write_lrc(const std::string& path, const std::vector<crispasr_disp_segment>& segs) {
    std::ofstream f(path);
    if (!f) {
        fprintf(stderr, "crispasr: warning: cannot write LRC '%s'\n", path.c_str());
        return false;
    }
    f << "[by:crispasr]\n";
    for (const auto& s : segs) {
        // LRC format: [mm:ss.xx]text
        const int64_t cs = s.t0;
        const int mm = (int)(cs / 6000);
        const int ss = (int)((cs % 6000) / 100);
        const int xx = (int)(cs % 100);
        char buf[16];
        snprintf(buf, sizeof(buf), "[%02d:%02d.%02d]", mm, ss, xx);
        f << buf << prefix_speaker(s.speaker) << s.text << "\n";
    }
    return true;
}

std::string crispasr_ctc_logits_to_json(const crispasr_ctc_logits& logits) {
    std::ostringstream js;
    js << "{";
    js << "\"n_frames\":" << logits.n_frames << ",";
    js << "\"n_vocab\":" << logits.n_vocab << ",";
    js << "\"layout\":\"frame_major\",";
    js << "\"normalization\":\"" << json_escape(logits.normalization) << "\",";
    if (!logits.vocab.empty()) {
        js << "\"vocab\":[";
        for (size_t i = 0; i < logits.vocab.size(); i++) {
            if (i)
                js << ",";
            js << "\"" << json_escape(logits.vocab[i]) << "\"";
        }
        js << "],";
    }
    js << "\"data\":[";
    for (size_t i = 0; i < logits.data.size(); i++) {
        if (i)
            js << ",";
        js << logits.data[i];
    }
    js << "]}";
    return js.str();
}

bool crispasr_write_ctc_logits_json(const std::string& path, const crispasr_ctc_logits& logits,
                                    const std::string& backend_name) {
    std::ofstream f(path);
    if (!f) {
        fprintf(stderr, "crispasr: warning: cannot write CTC logits JSON '%s'\n", path.c_str());
        return false;
    }
    f << "{\n";
    f << "  \"backend\": \"" << json_escape(backend_name) << "\",\n";
    f << "  \"ctc_logits\": " << crispasr_ctc_logits_to_json(logits) << "\n";
    f << "}\n";
    return true;
}

// ---------------------------------------------------------------------------
// Punctuation stripping
// ---------------------------------------------------------------------------

namespace {

// Strip ASCII punctuation + a handful of common Unicode marks from the
// input string. Collapses resulting double-spaces and trims the ends.
// Not trying to be clever: the point is to give users a "give me the
// words only" view of an LLM transcript, not a grammar-preserving edit.
std::string strip_punct_str(const std::string& in) {
    static const char* ASCII_DROP = ",.?!:;\"()[]{}<>/@#$%^&*=|\\~`";
    std::string out;
    out.reserve(in.size());
    size_t i = 0;
    while (i < in.size()) {
        const unsigned char c = (unsigned char)in[i];
        // Fast path: ASCII punctuation characters to drop.
        if (c < 0x80) {
            if (strchr(ASCII_DROP, (char)c)) {
                i++;
                continue;
            }
            // Keep apostrophe-in-word ("don't") but drop leading/trailing.
            if (c == '\'') {
                const bool prev_alpha = !out.empty() && ((out.back() >= 'a' && out.back() <= 'z') ||
                                                         (out.back() >= 'A' && out.back() <= 'Z'));
                const bool next_alpha = i + 1 < in.size() && ((in[i + 1] >= 'a' && in[i + 1] <= 'z') ||
                                                              (in[i + 1] >= 'A' && in[i + 1] <= 'Z'));
                if (!(prev_alpha && next_alpha)) {
                    i++;
                    continue;
                }
            }
            out += (char)c;
            i++;
            continue;
        }
        // Multi-byte UTF-8: decode just enough to recognise a few
        // Unicode punctuation marks the LLM backends commonly emit.
        //   U+2018 ' U+2019 ' (smart quotes)        -> drop
        //   U+201C " U+201D "                       -> drop
        //   U+2013 – U+2014 — (en/em dashes)         -> drop
        //   U+2026 … (ellipsis)                     -> drop
        //   U+00A0 nbsp                              -> space
        //   U+00BF ¿ U+00A1 ¡                        -> drop
        // Everything else is passed through untouched.
        int cp = 0, len = 1;
        if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F;
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F;
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07;
            len = 4;
        } else {
            out += (char)c;
            i++;
            continue;
        }
        if (i + len > in.size()) {
            out += (char)c;
            i++;
            continue;
        }
        for (int k = 1; k < len; k++)
            cp = (cp << 6) | ((unsigned char)in[i + k] & 0x3F);

        auto is_drop_cp = [](int p) {
            return p == 0x2018 || p == 0x2019 || p == 0x201C || p == 0x201D || p == 0x2013 || p == 0x2014 ||
                   p == 0x2026 || p == 0x00BF || p == 0x00A1;
        };
        if (is_drop_cp(cp)) {
            i += (size_t)len;
            continue;
        }
        if (cp == 0x00A0) {
            out += ' ';
            i += (size_t)len;
            continue;
        }
        for (int k = 0; k < len; k++)
            out += in[i + k];
        i += (size_t)len;
    }

    // Collapse runs of spaces and trim.
    std::string final_out;
    final_out.reserve(out.size());
    bool last_space = true;
    for (char c : out) {
        if (c == ' ' || c == '\t' || c == '\n') {
            if (!last_space) {
                final_out += ' ';
                last_space = true;
            }
        } else {
            final_out += c;
            last_space = false;
        }
    }
    while (!final_out.empty() && final_out.back() == ' ')
        final_out.pop_back();
    return final_out;
}

} // namespace

void crispasr_strip_punctuation(crispasr_segment& seg) {
    seg.text = strip_punct_str(seg.text);
    for (auto& w : seg.words)
        w.text = strip_punct_str(w.text);
    for (auto& t : seg.tokens)
        t.text = strip_punct_str(t.text);
}

// ---------------------------------------------------------------------------
// Stdout printing
// ---------------------------------------------------------------------------

void crispasr_print_alternatives(const std::vector<crispasr_segment>& segs, int n_alt) {
    for (const auto& seg : segs) {
        if (seg.tokens.empty()) {
            // No token-level info — show segment text with overall confidence
            printf("  \"%s\"", seg.text.c_str());
            printf("\n");
            continue;
        }
        for (const auto& tok : seg.tokens) {
            if (tok.is_special)
                continue;
            // Primary token with confidence
            printf("  %-12s", tok.text.c_str());
            if (tok.confidence >= 0) {
                printf(" [%.1f%%]", tok.confidence * 100.0f);
            }
            // Show alternatives if available
            if (!tok.alts.empty()) {
                printf("  (");
                int n = std::min(n_alt, (int)tok.alts.size());
                for (int i = 0; i < n; i++) {
                    if (i > 0)
                        printf(", ");
                    printf("%s %.1f%%", tok.alts[i].text.c_str(), tok.alts[i].prob * 100.0f);
                }
                printf(")");
            } else if (tok.confidence >= 0 && tok.confidence < 0.8f) {
                // No alternatives stored, but low confidence — flag it
                printf("  [uncertain]");
            }
            printf("\n");
        }
        printf("\n");
    }
    fflush(stdout);
}

void crispasr_print_confidence(const std::vector<crispasr_segment>& segs) {
    for (const auto& seg : segs) {
        // No token-level confidence (backend didn't emit tokens) — fall back to
        // the plain segment text so the flag still yields readable output.
        bool any_tok = false;
        for (const auto& tok : seg.tokens) {
            if (!tok.is_special) {
                any_tok = true;
                break;
            }
        }
        if (!any_tok) {
            if (!seg.text.empty())
                printf("%s\n", seg.text.c_str());
            continue;
        }
        bool first = true;
        for (const auto& tok : seg.tokens) {
            if (tok.is_special)
                continue;
            // Tokens carry the leading-space word-boundary convention; strip it
            // so the annotation attaches to the word, not a stray gap.
            const char* t = tok.text.c_str();
            while (*t == ' ')
                ++t;
            if (!*t)
                continue;
            printf("%s%s", first ? "" : " ", t);
            first = false;
            if (tok.confidence >= 0.0f)
                printf("[%.0f%%]", tok.confidence * 100.0f);
        }
        printf("\n");
    }
    fflush(stdout);
}

void crispasr_print_stdout(const std::vector<crispasr_disp_segment>& segs, bool show_timestamps) {
    if (show_timestamps) {
        for (const auto& s : segs) {
            printf("[%s --> %s]  %s%s\n", crispasr_to_timestamp(s.t0).c_str(), crispasr_to_timestamp(s.t1).c_str(),
                   prefix_speaker(s.speaker), s.text.c_str());
        }
    } else {
        std::string joined;
        for (const auto& s : segs) {
            if (!joined.empty())
                joined += " ";
            joined += prefix_speaker(s.speaker);
            joined += s.text;
        }
        printf("%s\n", joined.c_str());
    }
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// String-based formatters (for HTTP server responses)
// ---------------------------------------------------------------------------

std::string crispasr_segments_to_text(const std::vector<crispasr_segment>& segs) {
    std::string out;
    for (const auto& s : segs) {
        if (!out.empty())
            out += ' ';
        out += s.text;
    }
    return out;
}

std::string crispasr_segments_to_srt(const std::vector<crispasr_segment>& segs, int max_len) {
    auto disp = crispasr_make_disp_segments(segs, max_len);
    std::ostringstream out;
    for (size_t i = 0; i < disp.size(); i++) {
        out << (i + 1) << "\n"
            << crispasr_to_timestamp(disp[i].t0, /*comma=*/true) << " --> "
            << crispasr_to_timestamp(disp[i].t1, /*comma=*/true) << "\n"
            << prefix_speaker(disp[i].speaker) << disp[i].text << "\n\n";
    }
    return out.str();
}

std::string crispasr_segments_to_vtt(const std::vector<crispasr_segment>& segs, int max_len) {
    auto disp = crispasr_make_disp_segments(segs, max_len);
    std::ostringstream out;
    out << "WEBVTT\n\n";
    for (const auto& s : disp) {
        out << crispasr_to_timestamp(s.t0) << " --> " << crispasr_to_timestamp(s.t1) << "\n"
            << prefix_speaker(s.speaker) << s.text << "\n\n";
    }
    return out.str();
}

std::string crispasr_segments_to_openai_json(const std::vector<crispasr_segment>& segs) {
    std::string text = crispasr_segments_to_text(segs);
    return "{\"text\": \"" + json_escape(text) + "\"}";
}

// Convert centiseconds to seconds as a double for OpenAI JSON output.
static double cs_to_sec(int64_t cs) {
    return cs / 100.0;
}

// Defined below, next to diarized_json which has always used it. Declared here
// so verbose_json can emit the same "A"/"B" labels rather than the raw internal
// "(speaker 0) " — two formats of one API disagreeing about what a speaker is
// called would be worse than the omission this fixes (#326).
static std::string normalise_speaker(const std::string& raw);

std::string crispasr_segments_to_openai_verbose_json(const std::vector<crispasr_segment>& segs, double duration_s,
                                                     const std::string& language, const std::string& task,
                                                     float temperature) {
    std::string full_text = crispasr_segments_to_text(segs);

    std::ostringstream js;
    js << std::fixed;
    js << "{\n";
    js << "  \"task\": \"" << json_escape(task) << "\",\n";
    js << "  \"language\": \"" << json_escape(language) << "\",\n";
    js << std::setprecision(3);
    js << "  \"duration\": " << duration_s << ",\n";
    js << "  \"text\": \"" << json_escape(full_text) << "\",\n";
    js << "  \"segments\": [\n";
    for (size_t i = 0; i < segs.size(); i++) {
        const auto& s = segs[i];
        js << "    {\n";
        js << "      \"id\": " << i << ",\n";
        js << std::setprecision(2);
        js << "      \"start\": " << cs_to_sec(s.t0) << ",\n";
        js << "      \"end\": " << cs_to_sec(s.t1) << ",\n";
        js << "      \"text\": \"" << json_escape(s.text) << "\",\n";
        js << std::setprecision(6);
        js << "      \"temperature\": " << temperature << ",\n";

        // Compute avg_logprob from token confidences if available.
        double avg_logprob = 0.0;
        int n_scored = 0;
        for (const auto& t : s.tokens) {
            if (t.confidence > 0.0f) {
                avg_logprob += std::log(t.confidence);
                n_scored++;
            }
        }
        if (n_scored > 0)
            avg_logprob /= n_scored;
        js << "      \"avg_logprob\": " << avg_logprob << ",\n";

        // no_speech_prob — not available from most backends, emit 0.
        js << "      \"no_speech_prob\": 0.0";

        // Speaker label, when diarization produced one (#326). Not part of
        // OpenAI's verbose_json schema, which is why it is emitted only when
        // non-empty: a client that does not know the field never sees it, and
        // one that asked for --diarize is not silently charged for a stage
        // whose entire output this format used to discard. `diarized_json`
        // remains the richer format; this is so the standard one stops lying.
        if (!s.speaker.empty()) {
            js << ",\n      \"speaker\": \"" << json_escape(normalise_speaker(s.speaker)) << "\"";
        }

        // Word-level timestamps if available.
        if (!s.words.empty()) {
            js << ",\n      \"words\": [\n";
            for (size_t j = 0; j < s.words.size(); j++) {
                const auto& w = s.words[j];
                js << "        {\"word\": \"" << json_escape(w.text) << "\", ";
                js << std::setprecision(2);
                js << "\"start\": " << cs_to_sec(w.t0) << ", ";
                js << "\"end\": " << cs_to_sec(w.t1) << "}";
                if (j + 1 < s.words.size())
                    js << ",";
                js << "\n";
            }
            js << "      ]";
        }

        // Token IDs if available.
        if (!s.tokens.empty()) {
            js << ",\n      \"tokens\": [";
            bool first = true;
            for (const auto& t : s.tokens) {
                if (t.is_special)
                    continue;
                if (!first)
                    js << ", ";
                js << t.id;
                first = false;
            }
            js << "]";
        }

        js << "\n    }";
        if (i + 1 < segs.size())
            js << ",";
        js << "\n";
    }
    js << "  ]\n";
    js << "}\n";
    return js.str();
}

// ---------------------------------------------------------------------------
// Normalise speaker labels: "(speaker 0) " → "A", "(speaker 1) " → "B", etc.
// Falls through to the raw string (trimmed) for non-standard labels.
// ---------------------------------------------------------------------------
static std::string normalise_speaker(const std::string& raw) {
    // Trim trailing whitespace.
    std::string s = raw;
    while (!s.empty() && (s.back() == ' ' || s.back() == ')'))
        s.pop_back();
    // Extract number from "(speaker N" pattern.
    const char prefix[] = "(speaker ";
    if (s.size() > sizeof(prefix) - 1 && s.compare(0, sizeof(prefix) - 1, prefix) == 0) {
        int n = std::atoi(s.c_str() + sizeof(prefix) - 1);
        if (n >= 0 && n < 26)
            return std::string(1, (char)('A' + n));
        // More than 26 speakers: "AA", "AB", …
        return std::string(1, (char)('A' + n / 26)) + (char)('A' + n % 26);
    }
    // Already a clean label — return trimmed.
    if (!s.empty() && s.front() == '(')
        s.erase(s.begin());
    return s.empty() ? "A" : s;
}

std::string crispasr_segments_to_diarized_json(const std::vector<crispasr_segment>& segs, double duration_s,
                                               const std::string& language, const std::string& task,
                                               float temperature) {
    std::string full_text = crispasr_segments_to_text(segs);

    std::ostringstream js;
    js << std::fixed;
    js << "{\n";
    js << "  \"task\": \"" << json_escape(task) << "\",\n";
    js << "  \"language\": \"" << json_escape(language) << "\",\n";
    js << std::setprecision(3);
    js << "  \"duration\": " << duration_s << ",\n";
    js << "  \"text\": \"" << json_escape(full_text) << "\",\n";
    js << "  \"segments\": [\n";
    for (size_t i = 0; i < segs.size(); i++) {
        const auto& s = segs[i];
        js << "    {\n";
        js << "      \"id\": " << i << ",\n";
        js << std::setprecision(2);
        js << "      \"start\": " << cs_to_sec(s.t0) << ",\n";
        js << "      \"end\": " << cs_to_sec(s.t1) << ",\n";
        js << "      \"text\": \"" << json_escape(s.text) << "\",\n";

        // Speaker label — normalised to letter form.
        std::string spk = s.speaker.empty() ? "A" : normalise_speaker(s.speaker);
        js << "      \"speaker\": \"" << json_escape(spk) << "\",\n";

        js << "      \"type\": \"transcript.text.segment\"";
        // #292: chunk scope — diarize speaker labels are CHUNK-LOCAL, so a
        // consumer must not treat the same label across different chunk_ids as
        // the same person. Present only on multi-chunk runs (single-pass = -1).
        if (s.chunk_id >= 0)
            js << ",\n      \"chunk_id\": " << s.chunk_id;

        // Word-level timestamps if available.
        if (!s.words.empty()) {
            js << ",\n      \"words\": [\n";
            for (size_t j = 0; j < s.words.size(); j++) {
                const auto& w = s.words[j];
                js << "        {\"word\": \"" << json_escape(w.text) << "\", ";
                js << std::setprecision(2);
                js << "\"start\": " << cs_to_sec(w.t0) << ", ";
                js << "\"end\": " << cs_to_sec(w.t1) << "}";
                if (j + 1 < s.words.size())
                    js << ",";
                js << "\n";
            }
            js << "      ]";
        }

        js << "\n    }";
        if (i + 1 < segs.size())
            js << ",";
        js << "\n";
    }
    js << "  ]\n";
    js << "}\n";
    return js.str();
}

std::string crispasr_segments_to_native_json(const std::vector<crispasr_segment>& segs, const std::string& backend_name,
                                             double duration_s) {
    std::ostringstream js;
    js << "{\n";
    js << "  \"backend\": \"" << json_escape(backend_name) << "\",\n";
    js << "  \"duration\": " << duration_s << ",\n";
    js << "  \"segments\": [\n";
    for (size_t i = 0; i < segs.size(); i++) {
        const auto& s = segs[i];
        js << "    {\n";
        // t0/t1 are centiseconds; multiply by 10 to get milliseconds.
        js << "      \"t0\": " << (s.t0 * 10) << ",\n";
        js << "      \"t1\": " << (s.t1 * 10) << ",\n";
        js << "      \"text\": \"" << json_escape(s.text) << "\"";
        if (!s.speaker.empty()) {
            js << ",\n      \"speaker\": \"" << json_escape(s.speaker) << "\"";
        }
        if (s.chunk_id >= 0) // #292: chunk-local speaker scope (multi-chunk only)
            js << ",\n      \"chunk_id\": " << s.chunk_id;
        if (!s.tokens.empty()) {
            js << ",\n      \"tokens\": [\n";
            for (size_t j = 0; j < s.tokens.size(); j++) {
                const auto& t = s.tokens[j];
                js << "        {\"text\": \"" << json_escape(t.text) << "\"";
                if (t.confidence >= 0)
                    js << ", \"confidence\": " << t.confidence;
                if (t.t0 >= 0)
                    js << ", \"t0\": " << (t.t0 * 10);
                if (t.t1 >= 0)
                    js << ", \"t1\": " << (t.t1 * 10);
                js << "}";
                if (j + 1 < s.tokens.size())
                    js << ",";
                js << "\n";
            }
            js << "      ]";
        }
        js << "\n    }";
        if (i + 1 < segs.size())
            js << ",";
        js << "\n";
    }
    js << "  ],\n";
    std::string full_text = crispasr_segments_to_text(segs);
    js << "  \"text\": \"" << json_escape(full_text) << "\"\n";
    js << "}\n";
    return js.str();
}
