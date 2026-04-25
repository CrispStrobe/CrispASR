/**
 * align.cpp  —  CTC forced alignment via Viterbi DP.
 *
 * Algorithm
 * ---------
 * 1. Normalise each word: strip punctuation, lowercase → character sequence.
 *    Word boundaries are represented by the "|" token (same convention as HF
 *    wav2vec2 CTC decoders).
 *
 * 2. Build the CTC expanded sequence:
 *    seq = [blank, c0, blank, c1, ..., blank, cN-1, blank]
 *    Length S = 2*N + 1  (N = total number of label tokens incl. "|")
 *
 * 3. Viterbi DP in log-probability space:
 *    alpha[j] = log-prob of the best path up to current time ending at label j.
 *    Transitions at each time step t:
 *      stay:        alpha[j] ← alpha_prev[j]
 *      advance:     alpha[j] ← alpha_prev[j-1]
 *      skip-blank:  alpha[j] ← alpha_prev[j-2]  (only when seq[j] ≠ blank AND
 *                                                  seq[j] ≠ seq[j-2])
 *    back-pointer back[t][j] records which of stay/advance/skip-blank was best.
 *
 * 4. Traceback from the last valid label position at t = T-1.
 *
 * 5. Map each word's character range in `seq` to frames → word t0/t1.
 *
 * Memory: O(T × S) for back-pointers — manageable for typical utterances
 * (T ≤ 5000 frames × S ≤ 2000 labels → ~10 MB at int8).
 */

#include "align.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Text normalisation helpers
// ---------------------------------------------------------------------------

// Return true if character c is in the wav2vec2 token vocabulary.
// We index the char as a single-char std::string (ascii or UTF-8 continuation).
static int lookup_char(const std::unordered_map<std::string, int>& v2id, unsigned char c) {
    char buf[2] = {(char)c, '\0'};
    auto it = v2id.find(buf);
    return it != v2id.end() ? it->second : -1;
}

// Decode the UTF-8 sequence starting at `word[i]` and return its byte length
// (1–4). Returns 1 for invalid leading bytes (graceful degrade — caller will
// look up that single byte and miss, which is no worse than before).
static int utf8_seq_len(const std::string& word, size_t i) {
    unsigned char b = (unsigned char)word[i];
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 1; // invalid leading byte — fall back to single-byte
}

// Append the CTC token IDs for a single word to `out_chars`.
// Returns false if no characters of the word appear in vocab.
//
// Iterates by UTF-8 codepoint, not byte. For multibyte chars (Japanese/Chinese/
// Korean/accented Latin etc.) the vocab usually carries the full UTF-8 string
// (e.g. "こ" as a 3-byte entry), not individual bytes. Per-byte iteration
// previously made every CJK alignment fail with zero overlap (#32).
static bool word_to_ids(const std::string& word, const std::unordered_map<std::string, int>& v2id, int blank_id,
                        std::vector<int>& out_chars) {
    bool any = false;
    for (size_t i = 0; i < word.size(); ) {
        const int n = utf8_seq_len(word, i);
        // Single ASCII byte — try lowercase first, then raw.
        if (n == 1) {
            const char lc = (char)std::tolower((unsigned char)word[i]);
            int id = lookup_char(v2id, (unsigned char)lc);
            if (id < 0)
                id = lookup_char(v2id, (unsigned char)word[i]);
            if (id >= 0 && id != blank_id) {
                out_chars.push_back(id);
                any = true;
            }
        } else if (i + (size_t)n <= word.size()) {
            // Multi-byte UTF-8 codepoint — look up the full sequence.
            std::string cp(word, i, (size_t)n);
            auto it = v2id.find(cp);
            if (it != v2id.end() && it->second != blank_id) {
                out_chars.push_back(it->second);
                any = true;
            }
        }
        i += (size_t)n;
    }
    return any;
}

// ---------------------------------------------------------------------------
// ctc_forced_align
// ---------------------------------------------------------------------------

std::vector<ctc_word_stamp> ctc_forced_align(const float* logits, int T, int V, const std::vector<std::string>& words,
                                             const std::vector<std::string>& vocab, int blank_id, float frame_dur) {
    if (T <= 0 || V <= 0 || words.empty())
        return {};

    // ------------------------------------------------------------------
    // 1. Build vocab reverse map  token → id
    // ------------------------------------------------------------------
    std::unordered_map<std::string, int> v2id;
    v2id.reserve(vocab.size());
    for (int i = 0; i < (int)vocab.size(); i++)
        v2id[vocab[i]] = i;

    int bar_id = -1;
    {
        auto it = v2id.find("|");
        if (it != v2id.end())
            bar_id = it->second;
    }

    // ------------------------------------------------------------------
    // 2. Build character (label) sequence & track per-word char ranges
    //    chars[cs .. ce] (inclusive) are the CTC labels for word wi.
    //    A "|" separator is inserted before each word (except the first)
    //    but is not included in the word's range.
    // ------------------------------------------------------------------
    struct word_range {
        int cs, ce;
    }; // inclusive, in chars[]

    std::vector<int> chars;
    std::vector<word_range> wranges;

    for (int wi = 0; wi < (int)words.size(); wi++) {
        if (wi > 0 && bar_id >= 0)
            chars.push_back(bar_id);

        int cs = (int)chars.size();
        bool ok = word_to_ids(words[wi], v2id, blank_id, chars);
        int ce = (int)chars.size() - 1;

        if (!ok || ce < cs) {
            // Word had no recognisable characters — insert a single blank
            // placeholder so the word gets a t0=t1=0 stamp later.
            wranges.push_back({-1, -1});
        } else {
            wranges.push_back({cs, ce});
        }
    }

    int N = (int)chars.size();
    if (N == 0) {
        // Nothing to align; return zeroed stamps
        std::vector<ctc_word_stamp> out;
        for (const auto& w : words)
            out.push_back({w, 0.f, 0.f});
        return out;
    }

    // ------------------------------------------------------------------
    // 3. Build expanded sequence  seq[0..S-1]
    //    Even positions: blank.  Odd positions: chars[j/2].
    // ------------------------------------------------------------------
    int S = 2 * N + 1;
    std::vector<int> seq(S);
    for (int j = 0; j < S; j++)
        seq[j] = (j % 2 == 0) ? blank_id : chars[j / 2];

    // ------------------------------------------------------------------
    // 4. Compute log-softmax for every frame  [T × V]
    // ------------------------------------------------------------------
    std::vector<float> lp(T * V);
    for (int t = 0; t < T; t++) {
        const float* src = logits + t * V;
        float* dst = lp.data() + t * V;
        float mx = *std::max_element(src, src + V);
        float s = 0.f;
        for (int i = 0; i < V; i++) {
            dst[i] = src[i] - mx;
            s += expf(dst[i]);
        }
        float ls = logf(std::max(s, 1e-30f));
        for (int i = 0; i < V; i++)
            dst[i] -= ls;
    }

    // ------------------------------------------------------------------
    // 5. Viterbi DP
    //    back[t][j]: 0 = stay, 1 = advance from j-1, 2 = skip-blank from j-2
    // ------------------------------------------------------------------
    const float NEG_INF = -1e30f;

    std::vector<float> alpha(S, NEG_INF), alpha_next(S);
    // back is T × S bytes — for large audio/long transcripts, capped to avoid OOM.
    // Max practical: 5000 frames × 2000 labels = 10 MB.
    std::vector<std::vector<int8_t>> back(T, std::vector<int8_t>(S, 0));

    // Initialise t = 0
    {
        const float* lp0 = lp.data();
        alpha[0] = lp0[seq[0]]; // blank
        if (S > 1)
            alpha[1] = lp0[seq[1]]; // first label
    }

    for (int t = 1; t < T; t++) {
        const float* lpt = lp.data() + t * V;
        std::fill(alpha_next.begin(), alpha_next.end(), NEG_INF);

        for (int j = 0; j < S; j++) {
            int tok = seq[j];

            // Stay
            float best = alpha[j];
            int8_t bsrc = 0;

            // Advance from j-1
            if (j >= 1 && alpha[j - 1] > best) {
                best = alpha[j - 1];
                bsrc = 1;
            }

            // Skip-blank: j-2 → j (only when current label is not blank
            // and differs from j-2, avoiding repeated same-char skips)
            if (j >= 2 && tok != blank_id && seq[j] != seq[j - 2] && alpha[j - 2] > best) {
                best = alpha[j - 2];
                bsrc = 2;
            }

            if (best > NEG_INF) {
                alpha_next[j] = best + lpt[tok];
                back[t][j] = bsrc;
            }
        }
        alpha.swap(alpha_next);
    }

    // ------------------------------------------------------------------
    // 6. Traceback
    // ------------------------------------------------------------------
    std::vector<int> path(T);

    // Start at the last label (S-1 or S-2, whichever is better)
    int j_cur = (alpha[S - 1] >= alpha[S - 2]) ? S - 1 : S - 2;

    for (int t = T - 1; t >= 0; t--) {
        path[t] = j_cur;
        if (t > 0) {
            switch (back[t][j_cur]) {
            case 0:
                break; // stay
            case 1:
                j_cur--;
                break;
            case 2:
                j_cur -= 2;
                break;
            }
        }
    }

    // ------------------------------------------------------------------
    // 7. Map path to per-word timestamps
    //    For word wi with char range [cs, ce] (inclusive, in chars[]):
    //      expanded positions: 2*cs+1 … 2*ce+1 (odd = label positions)
    //    t0 = first frame assigned to any label in that range
    //    t1 = (last such frame + 1)  → end-exclusive
    // ------------------------------------------------------------------
    std::vector<ctc_word_stamp> result;
    result.reserve(words.size());

    for (int wi = 0; wi < (int)words.size(); wi++) {
        ctc_word_stamp ws;
        ws.word = words[wi];
        ws.t0 = 0.f;
        ws.t1 = 0.f;

        if (wranges[wi].cs < 0) {
            result.push_back(ws);
            continue;
        }

        int es0 = 2 * wranges[wi].cs + 1; // first odd position in seq for this word
        int es1 = 2 * wranges[wi].ce + 1; // last  odd position

        int t0_frame = -1, t1_frame = -1;
        for (int t = 0; t < T; t++) {
            int j = path[t];
            if (j >= es0 && j <= es1) {
                if (t0_frame < 0)
                    t0_frame = t;
                t1_frame = t;
            }
        }

        if (t0_frame >= 0) {
            ws.t0 = (float)t0_frame * frame_dur;
            ws.t1 = (float)(t1_frame + 1) * frame_dur;
        }
        result.push_back(ws);
    }

    return result;
}
