// core/qwen3_stream.h — Qwen3-ASR / Confucius4-R2T2 prefix-rollback streaming.
//
// A line-by-line port of R2T2ASRModel.streaming_transcribe /
// finish_streaming_transcribe (netease-youdao/Confucius4-R2T2, r2t2/r2t2_asr.py)
// and the qwen_asr helpers they call (parse_asr_output,
// detect_and_fix_repetitions, normalize_language_name), plus the adaptive
// max_new_tokens schedule of the repo's example.py driver.
//
// The algorithm never keeps model state between chunks. Every chunk:
//   1. append the chunk to all audio seen so far;
//   2. prefix = previous raw decode with its last K tokens rolled back;
//   3. decode(prompt + prefix, all audio, max_new_tokens) -> gen_text;
//   4. raw = prefix + gen_text, re-parsed into (language, text).
//
// Everything here is string logic around ONE model call, which the caller
// supplies (Hooks). That keeps the port testable without a model: replaying
// the reference's recorded gen_text per chunk must reproduce the reference's
// prefixes and texts exactly (tests/test-qwen3-stream.cpp).
//
// Strings are processed as Unicode codepoints, the way Python str indexes,
// strips and compares them; tokens go through the caller's tokenizer.

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace core_qwen3_stream {

// ---------------------------------------------------------------- UTF-8 ----

inline bool utf8_decode(const std::string& s, std::u32string& out) {
    out.clear();
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char)s[i];
        char32_t cp;
        int n;
        if (c < 0x80) {
            cp = c;
            n = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F;
            n = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F;
            n = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07;
            n = 4;
        } else {
            return false;
        }
        if (i + n > s.size())
            return false;
        for (int k = 1; k < n; k++) {
            const unsigned char cc = (unsigned char)s[i + k];
            if ((cc & 0xC0) != 0x80)
                return false;
            cp = (cp << 6) | (cc & 0x3F);
        }
        if ((n == 2 && cp < 0x80) || (n == 3 && cp < 0x800) || (n == 4 && cp < 0x10000) || cp > 0x10FFFF ||
            (cp >= 0xD800 && cp <= 0xDFFF))
            return false;
        out.push_back(cp);
        i += n;
    }
    return true;
}

// Python's bytes.decode("utf-8", errors="replace"): invalid sequences become
// U+FFFD. Used where the reference decodes a token prefix that may end in the
// middle of a multi-byte character.
inline std::u32string utf8_decode_replace(const std::string& s) {
    std::u32string out;
    size_t i = 0;
    while (i < s.size()) {
        std::u32string one;
        bool ok = false;
        for (int n = 1; n <= 4 && i + n <= s.size(); n++) {
            if (utf8_decode(s.substr(i, n), one) && one.size() == 1) {
                out += one;
                i += n;
                ok = true;
                break;
            }
        }
        if (!ok) {
            out.push_back(U'\uFFFD');
            i += 1;
        }
    }
    return out;
}

inline std::string utf8_encode(const std::u32string& s) {
    std::string out;
    for (char32_t cp : s) {
        if (cp < 0x80) {
            out.push_back((char)cp);
        } else if (cp < 0x800) {
            out.push_back((char)(0xC0 | (cp >> 6)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back((char)(0xE0 | (cp >> 12)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char)(0xF0 | (cp >> 18)));
            out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        }
    }
    return out;
}

// ------------------------------------------------------ Python str helpers ----

// str.isspace() for the characters it is True on.
inline bool py_isspace(char32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

inline std::u32string py_strip(const std::u32string& s) {
    size_t a = 0, b = s.size();
    while (a < b && py_isspace(s[a]))
        a++;
    while (b > a && py_isspace(s[b - 1]))
        b--;
    return s.substr(a, b - a);
}

inline std::u32string py_rstrip(const std::u32string& s) {
    size_t b = s.size();
    while (b > 0 && py_isspace(s[b - 1]))
        b--;
    return s.substr(0, b);
}

// str.splitlines() boundaries.
inline std::vector<std::u32string> py_splitlines(const std::u32string& s) {
    std::vector<std::u32string> out;
    std::u32string cur;
    for (size_t i = 0; i < s.size(); i++) {
        const char32_t c = s[i];
        const bool brk = c == U'\n' || c == U'\r' || c == 0x0B || c == 0x0C || c == 0x1C || c == 0x1D || c == 0x1E ||
                         c == 0x85 || c == 0x2028 || c == 0x2029;
        if (brk) {
            out.push_back(cur);
            cur.clear();
            if (c == U'\r' && i + 1 < s.size() && s[i + 1] == U'\n')
                i++;
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty())
        out.push_back(cur);
    return out;
}

inline size_t py_find(const std::u32string& s, const std::u32string& sub) {
    return s.find(sub);
}

// s.split(sep)[0]
inline std::u32string py_split0(const std::u32string& s, char32_t sep) {
    const size_t p = s.find(sep);
    return p == std::u32string::npos ? s : s.substr(0, p);
}

inline char32_t ascii_lower(char32_t c) {
    return (c >= U'A' && c <= U'Z') ? c + 32 : c;
}

inline std::u32string ascii_lower(const std::u32string& s) {
    std::u32string o = s;
    for (auto& c : o)
        c = ascii_lower(c);
    return o;
}

inline bool is_cjk(char32_t c) {
    return c >= 0x4E00 && c <= 0x9FFF;
}

inline const std::u32string& asr_text_tag() {
    static const std::u32string t = U"<asr_text>";
    return t;
}

// -------------------------------------------------- qwen_asr / R2T2 helpers ----

// qwen_asr normalize_language_name: first letter upper, the rest lower. The
// Python version uses str.upper/lower; language names are ASCII.
inline std::u32string normalize_language_name(const std::u32string& language) {
    std::u32string s = py_strip(language);
    if (s.empty())
        return s;
    std::u32string o;
    o.push_back((s[0] >= U'a' && s[0] <= U'z') ? s[0] - 32 : s[0]);
    for (size_t i = 1; i < s.size(); i++)
        o.push_back(ascii_lower(s[i]));
    return o;
}

// qwen_asr detect_and_fix_repetitions (threshold 20, max_len 20).
inline std::u32string fix_char_repeats(const std::u32string& s, size_t thresh) {
    std::u32string res;
    size_t i = 0, n = s.size();
    while (i < n) {
        size_t count = 1;
        while (i + count < n && s[i + count] == s[i])
            count++;
        if (count > thresh)
            res.push_back(s[i]);
        else
            res += s.substr(i, count);
        i += count;
    }
    return res;
}

inline std::u32string fix_pattern_repeats(const std::u32string& s, size_t thresh, size_t max_len = 20) {
    const size_t n = s.size();
    const size_t min_repeat_chars = thresh * 2;
    if (n < min_repeat_chars)
        return s;
    size_t i = 0;
    std::u32string result;
    bool found = false;
    while (i + min_repeat_chars <= n) {
        found = false;
        for (size_t k = 1; k <= max_len; k++) {
            if (i + k * thresh > n)
                break;
            const std::u32string pattern = s.substr(i, k);
            bool valid = true;
            for (size_t rep = 1; rep < thresh; rep++) {
                const size_t start = i + rep * k;
                if (s.substr(start, k) != pattern) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                size_t end_index = i + thresh * k;
                while (end_index + k <= n && s.substr(end_index, k) == pattern)
                    end_index += k;
                result += pattern;
                result += fix_pattern_repeats(s.substr(end_index), thresh, max_len);
                i = n;
                found = true;
                break;
            }
        }
        if (found)
            break;
        result.push_back(s[i]);
        i++;
    }
    if (!found)
        result += s.substr(i);
    return result;
}

inline std::u32string detect_and_fix_repetitions(const std::u32string& text, size_t threshold = 20) {
    return fix_pattern_repeats(fix_char_repeats(text, threshold), threshold);
}

struct LangText {
    std::u32string language;
    std::u32string text;
};

// Shared tail of parse_asr_output / parse_language_output once s is ready.
inline LangText parse_tagged(const std::u32string& s) {
    const size_t p = s.find(asr_text_tag());
    if (p == std::u32string::npos)
        return {U"", py_strip(s)};
    const std::u32string meta = s.substr(0, p);
    const std::u32string text = s.substr(p + asr_text_tag().size());
    if (ascii_lower(meta).find(U"language none") != std::u32string::npos) {
        const std::u32string t = py_strip(text);
        return {U"", t};
    }
    std::u32string lang;
    for (const auto& raw_line : py_splitlines(meta)) {
        const std::u32string line = py_strip(raw_line);
        if (line.empty())
            continue;
        const std::u32string prefix = U"language ";
        if (ascii_lower(line).compare(0, prefix.size(), prefix) == 0) {
            const std::u32string val = py_strip(line.substr(prefix.size()));
            if (!val.empty())
                lang = normalize_language_name(val);
        }
        break;
    }
    return {lang, py_strip(text)};
}

// qwen_asr parse_asr_output.
inline LangText parse_asr_output(const std::u32string& raw, const std::u32string& user_language) {
    std::u32string s = py_strip(raw);
    if (s.empty())
        return {U"", U""};
    s = detect_and_fix_repetitions(s);
    if (!user_language.empty())
        return {user_language, s};
    return parse_tagged(s);
}

// r2t2_asr.parse_language_output: like parse_asr_output without the
// repetition fix, and English is only right-stripped.
inline LangText parse_language_output(const std::u32string& raw, const std::u32string& user_language) {
    const std::u32string s = (user_language == U"English") ? py_rstrip(raw) : py_strip(raw);
    if (s.empty())
        return {U"", U""};
    if (!user_language.empty())
        return {user_language, s};
    return parse_tagged(s);
}

// r2t2_asr._normalize_punct_by_context.
inline std::u32string normalize_punct_by_context(const std::u32string& text) {
    static const std::u32string en = U",.!?;:()";
    static const std::u32string zh = U"\uFF0C\u3002\uFF01\uFF1F\uFF1B\uFF1A\uFF08\uFF09";
    std::u32string out = text;
    for (size_t pos = 0; pos < text.size(); pos++) {
        const char32_t p = text[pos];
        const size_t ie = en.find(p), iz = zh.find(p);
        if (ie == std::u32string::npos && iz == std::u32string::npos)
            continue;
        char32_t prev = 0;
        for (size_t j = pos; j-- > 0;) {
            if (!py_isspace(text[j])) {
                prev = text[j];
                break;
            }
        }
        if (!prev)
            continue;
        if (is_cjk(prev)) {
            if (ie != std::u32string::npos)
                out[pos] = zh[ie];
        } else if (prev < 0x80 && ((prev >= U'0' && prev <= U'9') || (prev >= U'a' && prev <= U'z') ||
                                   (prev >= U'A' && prev <= U'Z') || prev == U'"' || prev == U'\'')) {
            if (iz != std::u32string::npos)
                out[pos] = en[iz];
        }
    }
    return out;
}

// re.sub(r'(?<=[一-鿿])\s+(?=[一-鿿])', '', s)
inline std::u32string strip_cjk_inner_spaces(const std::u32string& s) {
    std::u32string out;
    size_t i = 0;
    while (i < s.size()) {
        if (py_isspace(s[i]) && !out.empty() && is_cjk(out.back())) {
            size_t j = i;
            while (j < s.size() && py_isspace(s[j]))
                j++;
            if (j < s.size() && is_cjk(s[j])) {
                i = j;
                continue;
            }
        }
        out.push_back(s[i]);
        i++;
    }
    return out;
}

// ----------------------------------------------------------------- state ----

struct Hooks {
    // tokenizer.encode(text) — the model's tokenizer on plain text.
    std::function<std::vector<int32_t>(const std::string& utf8)> encode;
    // tokenizer.decode(ids) as raw bytes (may end mid-character).
    std::function<std::string(const std::vector<int32_t>& ids)> decode_bytes;
    // One model call: greedy decode of prompt_raw + prefix over all audio so
    // far, at most max_new_tokens, special tokens skipped. Returns UTF-8.
    std::function<std::string(const std::vector<float>& audio, const std::string& prefix, int max_new_tokens)> generate;
};

struct Config {
    int unfixed_chunk_num = 0;
    int unfixed_token_num = 1;
    bool rollback_punctuation = false;
    std::u32string force_language; // canonical name, empty = auto
};

struct State {
    Config cfg;
    int chunk_id = 0;
    std::vector<float> audio_accum;
    std::u32string raw_decoded;
    std::u32string language;
    std::u32string text;
};

// Decode ids[0 : len(ids) - k], growing k until the result has no U+FFFD
// (the reference's `while '�' in prefix: k += 1` loop).
inline std::u32string rollback_decode(const Hooks& h, const std::vector<int32_t>& ids, int k) {
    while (true) {
        const int end_idx = std::max(0, (int)ids.size() - k);
        if (end_idx == 0)
            return U"";
        const std::u32string d =
            utf8_decode_replace(h.decode_bytes(std::vector<int32_t>(ids.begin(), ids.begin() + end_idx)));
        if (d.find(U'\uFFFD') == std::u32string::npos)
            return d;
        k++;
    }
}

inline int rollback_k(const Config& cfg, const std::u32string& raw, const std::u32string& punct) {
    if (cfg.rollback_punctuation) {
        const std::u32string st = py_strip(raw);
        if (!st.empty() && punct.find(st.back()) != std::u32string::npos)
            return 0;
    }
    return cfg.unfixed_token_num;
}

struct StepResult {
    std::u32string prefix;   // what was appended to the prompt
    std::u32string gen_text; // model output after normalisation
    std::u32string fixed_text;
    bool counted = false; // chunk_id advanced (the reference `continue`s otherwise)
};

// One streaming_transcribe chunk. `chunk` has already been cut to size by the
// caller (the reference slices state.buffer by chunk_size_samples).
inline StepResult step(State& st, const Hooks& h, const float* chunk, size_t n, int max_new_tokens) {
    StepResult r;
    st.audio_accum.insert(st.audio_accum.end(), chunk, chunk + n);

    std::u32string prefix;
    if (st.chunk_id >= st.cfg.unfixed_chunk_num) {
        st.raw_decoded = py_split0(st.raw_decoded, U'|');
        const auto ids = h.encode(utf8_encode(st.raw_decoded));
        const int k = rollback_k(st.cfg, st.raw_decoded, U"\uFF0C\u3002\uFF01\uFF1F\u3001\uFF1B\uFF1A.!?;:");
        prefix = rollback_decode(h, ids, k);
    }
    prefix = py_split0(prefix, U'|');
    r.prefix = prefix;

    std::u32string gen = utf8_decode_replace(h.generate(st.audio_accum, utf8_encode(prefix), max_new_tokens));
    gen = normalize_punct_by_context(gen);
    gen.erase(std::remove(gen.begin(), gen.end(), U'\uFFFD'), gen.end());
    r.gen_text = gen;
    st.raw_decoded = prefix + gen;

    std::u32string lang;
    if (st.cfg.force_language.empty())
        lang = parse_language_output(st.raw_decoded, st.cfg.force_language).language;
    if (st.cfg.force_language == U"Chinese" || lang == U"Chinese")
        st.raw_decoded = strip_cjk_inner_spaces(st.raw_decoded);
    const LangText lt = parse_asr_output(st.raw_decoded, st.cfg.force_language);

    const size_t tag = st.raw_decoded.find(asr_text_tag());
    if (tag != std::u32string::npos)
        st.raw_decoded = st.raw_decoded.substr(0, tag) + asr_text_tag() + lt.text;
    else
        st.raw_decoded = lt.text;

    st.raw_decoded = py_split0(st.raw_decoded, U'|');
    const auto ids = h.encode(utf8_encode(st.raw_decoded));
    int k = rollback_k(st.cfg, st.raw_decoded, U"\uFF0C\u3002\uFF01\uFF1F\u3001\uFF1B\uFF1A,.!?;:");
    // `raw.split('<asr_text>')[1] == ""`: the piece between the first tag and
    // the next one (or the end) is empty.
    const size_t tag2 = st.raw_decoded.find(asr_text_tag());
    if (tag2 != std::u32string::npos) {
        const size_t after = tag2 + asr_text_tag().size();
        const size_t next = st.raw_decoded.find(asr_text_tag(), after);
        if ((next == std::u32string::npos ? st.raw_decoded.size() : next) == after)
            k = 0;
    }
    std::u32string fixed = rollback_decode(h, ids, k);
    const size_t tag3 = fixed.find(asr_text_tag());
    if (tag3 != std::u32string::npos)
        fixed = fixed.substr(tag3 + asr_text_tag().size());
    r.fixed_text = py_split0(fixed, U'|');

    if (tag2 == std::u32string::npos && st.cfg.force_language.empty()) {
        st.text.clear();
        r.fixed_text.clear();
        return r; // the reference `continue`s: chunk_id is NOT advanced
    }
    st.language = lt.language;
    st.text = py_split0(lt.text, U'|');
    st.chunk_id++;
    r.counted = true;
    return r;
}

// finish_streaming_transcribe: decode the tail once. Note its rollback keeps
// at least one token (max(1, ...)) and has no U+FFFD loop, unlike step().
inline StepResult finish(State& st, const Hooks& h, const float* tail, size_t n, int max_new_tokens) {
    StepResult r;
    if (n == 0)
        return r;
    st.audio_accum.insert(st.audio_accum.end(), tail, tail + n);
    std::u32string prefix;
    if (st.chunk_id >= st.cfg.unfixed_chunk_num) {
        const auto ids = h.encode(utf8_encode(st.raw_decoded));
        const int end_idx = std::max(1, (int)ids.size() - st.cfg.unfixed_token_num);
        prefix = utf8_decode_replace(
            h.decode_bytes(std::vector<int32_t>(ids.begin(), ids.begin() + std::min<size_t>(end_idx, ids.size()))));
    }
    prefix = py_split0(prefix, U'|');
    r.prefix = prefix;
    std::u32string gen = utf8_decode_replace(h.generate(st.audio_accum, utf8_encode(prefix), max_new_tokens));
    gen = normalize_punct_by_context(gen);
    gen.erase(std::remove(gen.begin(), gen.end(), U'\uFFFD'), gen.end());
    r.gen_text = gen;
    st.raw_decoded = py_split0(prefix + gen, U'|');
    const LangText lt = parse_asr_output(st.raw_decoded, st.cfg.force_language);
    st.language = lt.language;
    st.text = py_split0(lt.text, U'|');
    st.chunk_id++;
    r.counted = true;
    return r;
}

// example.py's driver schedule, as a small state machine so the session and
// the test share it. step/lookahead in samples at 16 kHz.
struct Schedule {
    int step = 2560;      // 160 ms
    int lookahead = 2560; // 160 ms
    bool first = true;
    int max_new = 0;
    int first_max_new = 0;
    int floor_cap = 0;
    std::u32string last_text;

    void init(int step_samples, int lookahead_samples) {
        step = step_samples;
        lookahead = lookahead_samples;
        first = true;
        max_new = std::max(1, (step + lookahead) / 1280);
        first_max_new = max_new;
        floor_cap = std::min(32, std::max(4, 2 * (step / 1280)));
        last_text.clear();
    }
    int next_chunk_samples() const { return first ? step + lookahead : step; }
    // After a chunk: text is the reference's `text.split("|")[0]` return value
    // (fixed_text). is_last_token_chinese() is always False in example.py
    // (its token list is never appended to), so that branch is not ported.
    void update(const std::u32string& text) {
        first = false;
        if (text.size() > last_text.size()) {
            last_text = text;
            max_new = std::max(1, step / 1280);
        } else {
            max_new = max_new + 1;
        }
        max_new = std::min(floor_cap, max_new);
    }
};

} // namespace core_qwen3_stream
