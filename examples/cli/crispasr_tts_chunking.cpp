// crispasr_tts_chunking.cpp — sentence splitter + silence-padded concat.
#include <cstdlib>
//
// See crispasr_tts_chunking.h for the design rationale (issue #66).

#include "crispasr_tts_chunking.h"
#include "crispasr_sentence_end.h"

#include <algorithm>
#include <cstdint>

namespace {

// True if `s[i ..]` begins with a sentence terminator. ASCII .!? are
// single-byte; the two Unicode terminators we recognise are 3-byte
// UTF-8 sequences. Caller guarantees i < s.size().
bool is_terminator_at(const std::string& s, std::size_t i, std::size_t& out_len) {
    char c = s[i];
    if (c == '.' || c == '!' || c == '?') {
        out_len = 1;
        return true;
    }
    if (i + 2 < s.size()) {
        const std::uint8_t b0 = (std::uint8_t)s[i];
        const std::uint8_t b1 = (std::uint8_t)s[i + 1];
        const std::uint8_t b2 = (std::uint8_t)s[i + 2];
        // U+3002 IDEOGRAPHIC FULL STOP: E3 80 82
        if (b0 == 0xE3 && b1 == 0x80 && b2 == 0x82) {
            out_len = 3;
            return true;
        }
        // U+0964 DEVANAGARI DANDA: E0 A5 A4
        if (b0 == 0xE0 && b1 == 0xA5 && b2 == 0xA4) {
            out_len = 3;
            return true;
        }
    }
    return false;
}

// True if position `i` is whitespace or end-of-input. Treats ASCII
// space, tab, newline, carriage return as whitespace; we don't expand
// to Unicode whitespace because the only tokens we care about are
// terminators followed by SOMETHING that ends the sentence visually.
bool is_space_or_end(const std::string& s, std::size_t i) {
    if (i >= s.size())
        return true;
    char c = s[i];
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

std::string trim(const std::string& s) {
    std::size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\n' || s[a] == '\r'))
        a++;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\n' || s[b - 1] == '\r'))
        b--;
    return s.substr(a, b - a);
}

// Break a chunk longer than max_chars into pieces along whitespace
// boundaries. If we can't find a whitespace boundary inside the
// window, split mid-word (last resort — protects against pathological
// no-whitespace input like a 10 KB hex dump in `input`).
void whitespace_split(const std::string& chunk, std::size_t max_chars, std::vector<std::string>& out) {
    if (chunk.size() <= max_chars) {
        if (!chunk.empty())
            out.push_back(chunk);
        return;
    }
    std::size_t start = 0;
    while (start < chunk.size()) {
        std::size_t end = std::min(start + max_chars, chunk.size());
        if (end < chunk.size()) {
            // Walk back to a whitespace boundary if one exists in the
            // [start+1, end) window.
            std::size_t back = end;
            while (back > start + 1 && !is_space_or_end(chunk, back))
                back--;
            if (back > start + 1)
                end = back;
        }
        std::string piece = trim(chunk.substr(start, end - start));
        if (!piece.empty())
            out.push_back(piece);
        start = end;
    }
}

} // namespace

std::vector<std::string> crispasr_tts_split_sentences(const std::string& text, std::size_t max_chars) {
    std::vector<std::string> result;
    if (text.empty())
        return result;

    std::size_t chunk_start = 0;
    std::size_t i = 0;
    while (i < text.size()) {
        std::size_t term_len = 0;
        if (is_terminator_at(text, i, term_len)) {
            // Terminator must be followed by whitespace or end-of-input
            // for it to count as a sentence break — otherwise it's a
            // decimal point ("1.5") or in-word punctuation.
            // A '.' after an ordinal, initial or abbreviation ("am 3. Oktober",
            // "Dr.", "z. B.") does not end the sentence; see crispasr_sentence_end.h.
            if (is_space_or_end(text, i + term_len) && (text[i] != '.' || crispasr_period_ends_sentence(text, i))) {
                std::string chunk = trim(text.substr(chunk_start, i + term_len - chunk_start));
                if (!chunk.empty())
                    whitespace_split(chunk, max_chars, result);
                chunk_start = i + term_len;
                i = chunk_start;
                continue;
            }
            i += term_len;
            continue;
        }
        i++;
    }
    // Tail (input that doesn't end with a terminator).
    if (chunk_start < text.size()) {
        std::string tail = trim(text.substr(chunk_start));
        if (!tail.empty())
            whitespace_split(tail, max_chars, result);
    }
    return result;
}

std::vector<std::string> crispasr_tts_plan_chunks_for_backend(const std::string& text, const std::string& backend_name,
                                                              std::size_t max_chars) {
    // VibeVoice, Qwen3-TTS and TADA rely on continuous generated-text context to
    // preserve speaker identity/prosody across sentences. Match all registered
    // variants by prefix; the concrete backend names include suffixes such as
    // "vibevoice-tts" and "qwen3-tts-1.7b-base".
    //
    // TADA (#197): the AR Llama-3.2 talker generates multi-sentence utterances in
    // a single pass, exactly like HumeAI's reference tada.py generate(). Splitting
    // at punctuation synthesizes each sentence in isolation, where the per-token
    // duration head over-predicts the trailing pause for a sentence-final period
    // (e.g. "Hi." alone expands to ~500 frames / ~9 s of silence+hum), and inserts
    // extra silence between chunks. Both diverge from the reference, so feed the
    // whole text as one chunk.
    //
    // dots.tts (#200): a continuous-latent AR model that generates multi-sentence
    // utterances in one pass with its own EOS, exactly like the reference
    // dots.tts generate(text). Splitting re-runs the per-sentence trailing-pause
    // over-prediction (as TADA) AND — when voice cloning — re-emits the spoken
    // AI-disclaimer that dots_tts_synthesize prepends on every call (one per
    // chunk). Feed the whole text as one chunk.
    //
    // omnivoice (#254): a masked-iterative (SoundStorm/MaskGIT-style) model that
    // synthesizes the WHOLE target span in one fixed-num_steps generation — the
    // reference omnivoice.cpp does the full paragraph as a single T=544 pass. No
    // per-token duration head (T_target is one length estimate, no MAX_FRAMES
    // truncation), so single-shot is safe. Sentence-splitting was pure overhead:
    // N chunks = N graph builds + N CUDA-graph warmups + N×num_steps iterations
    // instead of one, which is exactly the reporter's 15–20% gap vs omnivoice.cpp
    // (each chunk reshapes T so the CUDA graph can't be reused across chunks).
    if (backend_name.rfind("vibevoice", 0) == 0 || backend_name.rfind("qwen3-tts", 0) == 0 ||
        backend_name.rfind("tada", 0) == 0 || backend_name.rfind("dots-tts", 0) == 0)
        return {text};
    // Dia generates a dialogue as one delayed-code sequence. Splitting a
    // paragraph creates short prompts (often <100 bytes), loses speaker-turn
    // context, and applies an explicit generation limit afresh to each sentence.
    // Match the official generate(text) flow in both CLI and server requests.
    if (backend_name == "dia" || backend_name.rfind("dia-", 0) == 0 || backend_name.rfind("dia_", 0) == 0)
        return {text};
    // ZONOS3-Mini (#603): one delayed-code AR pass over the whole text with its own EOA, as upstream's
    // POST /v2/generate (registry max_frames 2000 = 133 s). Splitting re-plays the silence prefill and the
    // conditioning rows per sentence and diverges from the reference prompt.
    if (backend_name.rfind("zonos3", 0) == 0)
        return {text};

    // omnivoice defaults to single-shot too, but keep an escape hatch: on a
    // GPU without CUDA-graph reuse (Metal/CPU) single-shot's ~2.7× attention
    // (O(T²)) can cost more than the per-chunk warmup it saves. CRISPASR_OMNIVOICE_CHUNK=1
    // forces the legacy sentence-split path (also the A/B toggle for that tradeoff).
    if (backend_name.rfind("omnivoice", 0) == 0) {
        const char* e = std::getenv("CRISPASR_OMNIVOICE_CHUNK");
        if (!(e && e[0] && e[0] != '0'))
            return {text};
    }

    std::vector<std::string> result = crispasr_tts_split_sentences(text, max_chars);
    if (result.empty())
        result.push_back(text);
    // bt2-tts (#594): upstream breeze-tts generates the whole request in one
    // generate(text) with the instruction in front. One generation per
    // SENTENCE handed the model fragments like "Good God!" or "Do not laugh in
    // that manner." behind an instruction many times their length; the model
    // then kept reading the instruction aloud instead of the fragment. Pack
    // whole sentences back into paragraph-sized chunks: max_chars bytes stays
    // well inside the 750-frame (60 s) generation cap.
    if (backend_name.rfind("bt2", 0) == 0)
        return crispasr_tts_pack_sentences(result, max_chars);
    return result;
}

std::vector<std::string> crispasr_tts_pack_sentences(const std::vector<std::string>& sentences, std::size_t max_chars) {
    // Groups of sentence indices, packed greedily.
    std::vector<std::vector<std::size_t>> groups;
    std::vector<std::size_t> lens;
    for (std::size_t i = 0; i < sentences.size(); i++) {
        const std::size_t n = sentences[i].size();
        if (n == 0)
            continue;
        if (groups.empty() || lens.back() + 1 + n > max_chars) {
            groups.push_back({i});
            lens.push_back(n);
        } else {
            groups.back().push_back(i);
            lens.back() += 1 + n;
        }
    }
    // Rebalance a short tail: move trailing sentences of the previous group
    // into it while that makes the two sizes closer, so the request never
    // ends with a lone fragment ("Good God!") as its own generation.
    if (groups.size() >= 2) {
        auto& prev = groups[groups.size() - 2];
        auto& tail = groups.back();
        std::size_t& lp = lens[lens.size() - 2];
        std::size_t& lt = lens.back();
        while (prev.size() > 1) {
            const std::size_t n = sentences[prev.back()].size();
            if (lt + 1 + n > max_chars || lt + 1 + n >= lp - n - 1)
                break;
            tail.insert(tail.begin(), prev.back());
            prev.pop_back();
            lt += 1 + n;
            lp -= 1 + n;
        }
    }
    std::vector<std::string> out;
    for (const auto& g : groups) {
        std::string cur;
        for (std::size_t i : g) {
            if (!cur.empty())
                cur += ' ';
            cur += sentences[i];
        }
        out.push_back(std::move(cur));
    }
    return out;
}

std::vector<float> crispasr_tts_concat_with_silence(const std::vector<std::vector<float>>& chunks,
                                                    int silence_samples) {
    if (chunks.empty())
        return {};
    if (silence_samples < 0)
        silence_samples = 0;

    std::size_t total = 0;
    for (const auto& c : chunks)
        total += c.size();
    if (chunks.size() > 1)
        total += (chunks.size() - 1) * (std::size_t)silence_samples;

    std::vector<float> out;
    out.reserve(total);
    for (std::size_t i = 0; i < chunks.size(); i++) {
        if (i > 0 && silence_samples > 0)
            out.insert(out.end(), (std::size_t)silence_samples, 0.0f);
        out.insert(out.end(), chunks[i].begin(), chunks[i].end());
    }
    return out;
}
