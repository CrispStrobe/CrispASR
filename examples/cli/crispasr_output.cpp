// crispasr_output.cpp — output writers for non-whisper backends.
//
// Extracted / generalized from parakeet-main/main.cpp and cohere-main.cpp.

#include "crispasr_output.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

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

std::vector<crispasr_disp_segment> crispasr_make_disp_segments(const std::vector<crispasr_segment>& segments,
                                                               int max_len) {
    std::vector<crispasr_disp_segment> out;

    for (const auto& seg : segments) {
        // Easy path: no word data or no splitting requested — one display
        // segment covers the whole backend segment.
        if (seg.words.empty() || max_len == 0) {
            if (!seg.text.empty()) {
                out.push_back({seg.t0, seg.t1, seg.text, seg.speaker});
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

        // max_len > 1: pack words into segments up to max_len characters.
        crispasr_disp_segment cur;
        cur.t0 = -1;
        cur.speaker = seg.speaker;

        auto flush = [&]() {
            if (!cur.text.empty())
                out.push_back(cur);
            cur = {};
            cur.t0 = -1;
            cur.speaker = seg.speaker;
        };

        for (const auto& w : seg.words) {
            if (cur.t0 < 0)
                cur.t0 = w.t0;
            cur.t1 = w.t1;

            const std::string sep = cur.text.empty() ? "" : " ";
            const bool would_overflow =
                !cur.text.empty() && (int)(cur.text.size() + sep.size() + w.text.size()) > max_len;

            if (would_overflow) {
                flush();
                cur.t0 = w.t0;
                cur.t1 = w.t1;
            }
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
static std::string json_escape(const std::string& s) {
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

bool crispasr_write_json(const std::string& path, const std::vector<crispasr_segment>& segs,
                         const std::string& backend_name, const std::string& model_path, const std::string& language,
                         bool full) {
    std::ofstream f(path);
    if (!f) {
        fprintf(stderr, "crispasr: warning: cannot write JSON '%s'\n", path.c_str());
        return false;
    }
    f << "{\n";
    f << "  \"crispasr\": {\n";
    f << "    \"backend\": \"" << json_escape(backend_name) << "\",\n";
    f << "    \"model\":   \"" << json_escape(model_path) << "\",\n";
    f << "    \"language\":\"" << json_escape(language) << "\"\n";
    f << "  },\n";
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
        if (full && !s.words.empty()) {
            f << ",\n      \"words\": [\n";
            for (size_t j = 0; j < s.words.size(); j++) {
                const auto& w = s.words[j];
                f << "        { \"text\": \"" << json_escape(w.text) << "\", \"t0\": " << w.t0 << ", \"t1\": " << w.t1
                  << " }" << (j + 1 < s.words.size() ? "," : "") << "\n";
            }
            f << "      ]";
        }
        if (full && !s.tokens.empty()) {
            f << ",\n      \"tokens\": [\n";
            for (size_t j = 0; j < s.tokens.size(); j++) {
                const auto& t = s.tokens[j];
                f << "        { \"text\": \"" << json_escape(t.text) << "\", \"p\": " << t.confidence
                  << ", \"t0\": " << t.t0 << ", \"t1\": " << t.t1 << " }" << (j + 1 < s.tokens.size() ? "," : "")
                  << "\n";
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
