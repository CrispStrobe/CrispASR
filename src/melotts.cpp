// melotts.cpp — native ggml runtime for myshell-ai/MeloTTS (VITS2).
//
// Architecture (inference only):
//   1. TextEncoder: phone/tone/lang embeddings + 6-layer relative-position
//      transformer (with speaker injection at layer 2)
//      → mean + log_var (192-d each)
//   2. StochasticDurationPredictor (SDP): DDSConv + ConvFlow splines
//      → log-durations (blended with DurationPredictor via sdp_ratio)
//   3. DurationPredictor (DP): Conv1d + LayerNorm + ReLU → log-durations
//   4. TransformerCouplingFlow (inverse): 4 transformer coupling blocks
//      → latent z
//   5. HiFi-GAN decoder: conv_pre + 5 upsample stages + 15 resblocks
//      + conv_post + tanh → 44.1 kHz mono PCM
//
// Per-module ggml sub-graphs keep graph size small and avoid stitching
// data-dependent control flow into one monolithic graph.

#include "melotts.h"
#include "bert_encoder.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "core/conv.h"
#include "core/gguf_loader.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

// ── JSON-lite helpers ─────────────────────────────────────────────
// Parse JSON arrays/objects from GGUF metadata strings.

static std::vector<std::string> json_parse_string_array(const std::string& json) {
    std::vector<std::string> out;
    size_t pos = json.find('[');
    if (pos == std::string::npos)
        return out;
    pos++;
    while (pos < json.size()) {
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == ',' || json[pos] == '\n'))
            pos++;
        if (pos >= json.size() || json[pos] == ']')
            break;
        if (json[pos] != '"')
            break;
        pos++;
        std::string s;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                pos++;
                if (json[pos] == '"')
                    s += '"';
                else if (json[pos] == '\\')
                    s += '\\';
                else if (json[pos] == 'n')
                    s += '\n';
                else
                    s += json[pos];
            } else {
                s += json[pos];
            }
            pos++;
        }
        if (pos < json.size())
            pos++; // skip closing "
        out.push_back(s);
    }
    return out;
}

// ── G2P ───────────────────────────────────────────────────────────
// Minimal English G2P using embedded CMU dictionary.
// Falls back to character-level mapping for unknown words.

// ── Neural G2P (g2p_en encoder-decoder) ───────────────────────────
// Tiny GRU seq2seq: 29 graphemes → 74 ARPAbet phonemes.
// Weights loaded from GGUF metadata (base64 JSON, ~4 KB).

struct neural_g2p_model {
    bool loaded = false;
    // Grapheme/phoneme vocabularies
    std::vector<std::string> graphemes; // 29: <pad> <unk> </s> a-z
    std::vector<std::string> phonemes;  // 74: <pad> <unk> <s> </s> AA0..ZH
    std::map<std::string, int> g2idx;
    // Weights (all F32)
    std::vector<float> enc_emb;                                // (29, 256)
    std::vector<float> dec_emb;                                // (74, 256)
    std::vector<float> enc_w_ih, enc_w_hh, enc_b_ih, enc_b_hh; // GRU (768, 256)
    std::vector<float> dec_w_ih, dec_w_hh, dec_b_ih, dec_b_hh;
    std::vector<float> fc_w, fc_b; // (74, 256), (74,)
    int hidden_dim = 256;
};

static float nn_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static void gru_cell(const float* x, const float* h_prev, int input_dim, int hidden_dim, const float* w_ih,
                     const float* w_hh, const float* b_ih, const float* b_hh, float* h_out) {
    // GRU: 3 gates (reset, update, new) packed as [r, z, n]
    std::vector<float> g_ih(3 * hidden_dim, 0.0f);
    std::vector<float> g_hh(3 * hidden_dim, 0.0f);
    for (int o = 0; o < 3 * hidden_dim; o++) {
        float s1 = b_ih[o], s2 = b_hh[o];
        for (int i = 0; i < input_dim; i++)
            s1 += x[i] * w_ih[o * input_dim + i];
        for (int i = 0; i < hidden_dim; i++)
            s2 += h_prev[i] * w_hh[o * hidden_dim + i];
        g_ih[o] = s1;
        g_hh[o] = s2;
    }
    for (int i = 0; i < hidden_dim; i++) {
        float r = nn_sigmoid(g_ih[i] + g_hh[i]);
        float z = nn_sigmoid(g_ih[hidden_dim + i] + g_hh[hidden_dim + i]);
        float n = tanhf(g_ih[2 * hidden_dim + i] + r * g_hh[2 * hidden_dim + i]);
        h_out[i] = (1.0f - z) * n + z * h_prev[i];
    }
}

static std::vector<std::string> neural_g2p_predict(const neural_g2p_model& m, const std::string& word) {
    if (!m.loaded)
        return {};
    int D = m.hidden_dim;

    // Encode: chars + </s>
    std::string lower;
    for (char c : word)
        lower += (char)tolower((unsigned char)c);

    std::vector<int> char_ids;
    for (char c : lower) {
        std::string cs(1, c);
        auto it = m.g2idx.find(cs);
        char_ids.push_back(it != m.g2idx.end() ? it->second : 1); // 1=<unk>
    }
    char_ids.push_back(2); // 2=</s>

    // Run encoder GRU
    std::vector<float> h(D, 0.0f);
    for (int cid : char_ids) {
        const float* emb = &m.enc_emb[cid * D];
        std::vector<float> h_new(D);
        gru_cell(emb, h.data(), D, D, m.enc_w_ih.data(), m.enc_w_hh.data(), m.enc_b_ih.data(), m.enc_b_hh.data(),
                 h_new.data());
        h = h_new;
    }

    // Decode: start with <s> (id=2), greedy until </s> (id=3) or 20 steps
    std::vector<std::string> preds;
    int dec_id = 2; // <s>
    for (int step = 0; step < 20; step++) {
        const float* dec_emb = &m.dec_emb[dec_id * D];
        std::vector<float> h_new(D);
        gru_cell(dec_emb, h.data(), D, D, m.dec_w_ih.data(), m.dec_w_hh.data(), m.dec_b_ih.data(), m.dec_b_hh.data(),
                 h_new.data());
        h = h_new;

        // FC: logits = h @ fc_w^T + fc_b
        int n_ph = (int)m.phonemes.size();
        float best_val = -1e30f;
        int best_id = 0;
        for (int p = 0; p < n_ph; p++) {
            float s = m.fc_b[p];
            for (int d = 0; d < D; d++)
                s += h[d] * m.fc_w[p * D + d];
            if (s > best_val) {
                best_val = s;
                best_id = p;
            }
        }

        if (best_id == 3)
            break; // </s>
        if (best_id >= 4 && best_id < n_ph)
            preds.push_back(m.phonemes[best_id]);
        dec_id = best_id;
    }
    return preds;
}

struct melotts_g2p {
    // CMU dict: word -> list of syllables, each syllable is a list of phonemes
    std::map<std::string, std::vector<std::vector<std::string>>> cmudict;
    std::map<std::string, int> symbol_to_id;
    std::vector<std::string> symbols;
    int tone_start_en;
    // Neural G2P fallback (g2p_en model)
    neural_g2p_model neural;
};

// ARPAbet stress marker -> tone: 0=no stress, 1=primary+1, 2=secondary+1, 3=tertiary+1
static int arpa_to_tone(const std::string& ph) {
    if (ph.empty())
        return 0;
    char last = ph.back();
    if (last >= '0' && last <= '3')
        return (int)(last - '0') + 1;
    return 0;
}

static std::string arpa_strip_stress(const std::string& ph) {
    if (ph.empty())
        return ph;
    char last = ph.back();
    if (last >= '0' && last <= '3')
        return ph.substr(0, ph.size() - 1);
    return ph;
}

static std::string to_lower(const std::string& s) {
    std::string out = s;
    for (auto& c : out)
        c = (char)tolower((unsigned char)c);
    return out;
}

static std::string to_upper(const std::string& s) {
    std::string out = s;
    for (auto& c : out)
        c = (char)toupper((unsigned char)c);
    return out;
}

static std::string post_replace_ph(const std::string& ph, const std::map<std::string, int>& sym2id) {
    // Map punctuation and special phonemes (matches Python's post_replace_ph)
    static const std::map<std::string, std::string> rep = {
        {":", ","}, {";", ","}, {"\n", "."}, {"v", "V"}, // MeloTTS uses uppercase V for the /v/ phoneme
    };
    auto it = rep.find(ph);
    if (it != rep.end()) {
        if (sym2id.count(it->second))
            return it->second;
    }
    if (sym2id.count(ph))
        return ph;
    return "UNK";
}

// Simple tokenizer: split on punctuation/spaces, keep punctuation as tokens
static std::vector<std::string> tokenize_english(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cur;
    for (char c : text) {
        if (c == ' ' || c == ',' || c == '.' || c == '!' || c == '?' || c == ';' || c == ':' || c == '-' || c == '\'' ||
            c == '\n') {
            if (!cur.empty()) {
                tokens.push_back(cur);
                cur.clear();
            }
            if (c != ' ') {
                tokens.push_back(std::string(1, c));
            }
        } else {
            cur += c;
        }
    }
    if (!cur.empty())
        tokens.push_back(cur);
    return tokens;
}

// ── Rule-based letter-to-sound for OOV words ─────────────────────
// Converts English letters → ARPAbet phonemes using digraph/trigraph
// rules. Not perfect, but produces intelligible speech for most words.

static void lts_fallback(const std::string& word, const std::map<std::string, int>& sym2id, int tone_start, int lang_id,
                         std::vector<int>& phone_ids, std::vector<int>& tone_ids, std::vector<int>& lang_ids) {
    // Emit a phoneme: look up in symbol table
    auto emit = [&](const char* ph, int tone) {
        std::string s = ph;
        std::string mapped = post_replace_ph(s, sym2id);
        auto it = sym2id.find(mapped);
        if (it != sym2id.end()) {
            phone_ids.push_back(it->second);
            tone_ids.push_back(tone + tone_start);
            lang_ids.push_back(lang_id);
        }
    };

    const std::string& w = word;
    int len = (int)w.size();
    bool first_vowel = true; // assign primary stress to first vowel

    for (int i = 0; i < len;) {
        char c = w[i];
        char c1 = (i + 1 < len) ? w[i + 1] : 0;
        char c2 = (i + 2 < len) ? w[i + 2] : 0;

        // Helper: is this position followed by silent-e at end?
        bool silent_e = (i + 2 < len && i + 3 == len && w[len - 1] == 'e');

        // --- Trigraphs ---
        if (c == 't' && c1 == 'c' && c2 == 'h') {
            emit("ch", 0);
            i += 3;
            continue;
        }
        if (c == 'i' && c1 == 'g' && c2 == 'h') {
            emit("ay", first_vowel ? 1 : 0);
            first_vowel = false;
            i += 3;
            continue;
        }
        if (c == 't' && c1 == 'i' && c2 == 'o') {
            emit("sh", 0);
            emit("ah", 0);
            i += 3;
            continue;
        }

        // --- Digraphs (consonant) ---
        if (c == 't' && c1 == 'h') {
            emit("th", 0);
            i += 2;
            continue;
        }
        if (c == 's' && c1 == 'h') {
            emit("sh", 0);
            i += 2;
            continue;
        }
        if (c == 'c' && c1 == 'h') {
            emit("ch", 0);
            i += 2;
            continue;
        }
        if (c == 'p' && c1 == 'h') {
            emit("f", 0);
            i += 2;
            continue;
        }
        if (c == 'w' && c1 == 'h') {
            emit("w", 0);
            i += 2;
            continue;
        }
        if (c == 'n' && c1 == 'g') {
            emit("ng", 0);
            i += 2;
            continue;
        }
        if (c == 'c' && c1 == 'k') {
            emit("k", 0);
            i += 2;
            continue;
        }
        if (c == 'g' && c1 == 'h') {
            i += 2;
            continue;
        } // silent gh
        if (c == 'k' && c1 == 'n') {
            emit("n", 0);
            i += 2;
            continue;
        } // kn
        if (c == 'w' && c1 == 'r') {
            emit("r", 0);
            i += 2;
            continue;
        } // wr

        // --- Digraphs (vowel) ---
        if (c == 'e' && c1 == 'a') {
            emit("iy", first_vowel ? 1 : 0);
            first_vowel = false;
            i += 2;
            continue;
        }
        if (c == 'e' && c1 == 'e') {
            emit("iy", first_vowel ? 1 : 0);
            first_vowel = false;
            i += 2;
            continue;
        }
        if (c == 'o' && c1 == 'o') {
            emit("uw", first_vowel ? 1 : 0);
            first_vowel = false;
            i += 2;
            continue;
        }
        if (c == 'o' && c1 == 'u') {
            emit("aw", first_vowel ? 1 : 0);
            first_vowel = false;
            i += 2;
            continue;
        }
        if (c == 'o' && c1 == 'w') {
            emit("ow", first_vowel ? 1 : 0);
            first_vowel = false;
            i += 2;
            continue;
        }
        if (c == 'a' && c1 == 'i') {
            emit("ey", first_vowel ? 1 : 0);
            first_vowel = false;
            i += 2;
            continue;
        }
        if (c == 'a' && c1 == 'y') {
            emit("ey", first_vowel ? 1 : 0);
            first_vowel = false;
            i += 2;
            continue;
        }
        if (c == 'o' && c1 == 'i') {
            emit("oy", first_vowel ? 1 : 0);
            first_vowel = false;
            i += 2;
            continue;
        }
        if (c == 'o' && c1 == 'y') {
            emit("oy", first_vowel ? 1 : 0);
            first_vowel = false;
            i += 2;
            continue;
        }
        if (c == 'a' && c1 == 'w') {
            emit("ao", first_vowel ? 1 : 0);
            first_vowel = false;
            i += 2;
            continue;
        }
        if (c == 'e' && c1 == 'w') {
            emit("uw", first_vowel ? 1 : 0);
            first_vowel = false;
            i += 2;
            continue;
        }
        if (c == 'e' && c1 == 'r') {
            emit("er", first_vowel ? 1 : 0);
            first_vowel = false;
            i += 2;
            continue;
        }

        // --- Silent final e ---
        if (c == 'e' && i == len - 1 && i > 0) {
            i++;
            continue;
        }

        // --- Consonants ---
        if (c == 'b') {
            emit("b", 0);
            i++;
            continue;
        }
        if (c == 'd') {
            emit("d", 0);
            i++;
            continue;
        }
        if (c == 'f') {
            emit("f", 0);
            i++;
            continue;
        }
        if (c == 'g') {
            emit("g", 0);
            i++;
            continue;
        }
        if (c == 'h') {
            emit("hh", 0);
            i++;
            continue;
        }
        if (c == 'j') {
            emit("jh", 0);
            i++;
            continue;
        }
        if (c == 'k') {
            emit("k", 0);
            i++;
            continue;
        }
        if (c == 'l') {
            emit("l", 0);
            i++;
            continue;
        }
        if (c == 'm') {
            emit("m", 0);
            i++;
            continue;
        }
        if (c == 'n') {
            emit("n", 0);
            i++;
            continue;
        }
        if (c == 'p') {
            emit("p", 0);
            i++;
            continue;
        }
        if (c == 'r') {
            emit("r", 0);
            i++;
            continue;
        }
        if (c == 's') {
            // s between vowels → z
            if (i > 0 && i + 1 < len && strchr("aeiou", w[i - 1]) && strchr("aeiou", c1)) {
                emit("z", 0);
            } else {
                emit("s", 0);
            }
            i++;
            continue;
        }
        if (c == 't') {
            emit("t", 0);
            i++;
            continue;
        }
        if (c == 'w') {
            emit("w", 0);
            i++;
            continue;
        }
        if (c == 'x') {
            emit("k", 0);
            emit("s", 0);
            i++;
            continue;
        }
        if (c == 'z') {
            emit("z", 0);
            i++;
            continue;
        }
        if (c == 'q') {
            emit("k", 0);
            if (c1 == 'u')
                i++;
            i++;
            continue;
        }

        // --- C: before e/i/y → S, else → K ---
        if (c == 'c') {
            if (c1 == 'e' || c1 == 'i' || c1 == 'y')
                emit("s", 0);
            else
                emit("k", 0);
            i++;
            continue;
        }

        // --- Y: consonant at start, vowel otherwise ---
        if (c == 'y') {
            if (i == 0) {
                emit("y", 0);
            } else {
                emit("iy", first_vowel ? 1 : 0);
                first_vowel = false;
            }
            i++;
            continue;
        }

        // --- Vowels with context ---
        if (c == 'a') {
            if (silent_e) {
                emit("ey", first_vowel ? 1 : 0);
            } else {
                emit("ae", first_vowel ? 1 : 0);
            }
            first_vowel = false;
            i++;
            continue;
        }
        if (c == 'e') {
            if (silent_e) {
                emit("iy", first_vowel ? 1 : 0);
            } else {
                emit("eh", first_vowel ? 1 : 0);
            }
            first_vowel = false;
            i++;
            continue;
        }
        if (c == 'i') {
            if (silent_e) {
                emit("ay", first_vowel ? 1 : 0);
            } else {
                emit("ih", first_vowel ? 1 : 0);
            }
            first_vowel = false;
            i++;
            continue;
        }
        if (c == 'o') {
            if (silent_e) {
                emit("ow", first_vowel ? 1 : 0);
            } else {
                emit("aa", first_vowel ? 1 : 0);
            }
            first_vowel = false;
            i++;
            continue;
        }
        if (c == 'u') {
            if (silent_e) {
                emit("uw", first_vowel ? 1 : 0);
            } else {
                emit("ah", first_vowel ? 1 : 0);
            }
            first_vowel = false;
            i++;
            continue;
        }

        // Skip unknown characters
        i++;
    }
}

static void g2p_english(const melotts_g2p& g2p, const std::string& text, std::vector<int>& phone_ids,
                        std::vector<int>& tone_ids, std::vector<int>& lang_ids, std::vector<int>& word2ph) {
    phone_ids.clear();
    tone_ids.clear();
    lang_ids.clear();
    word2ph.clear();

    // Language ID for English = 2 (ZH=0, JP=1, EN=2)
    int lang_id = 2;
    int tone_start = g2p.tone_start_en;

    auto words = tokenize_english(to_lower(text));

    // Start with pad
    auto add_phone = [&](const std::string& ph, int tone) {
        auto it = g2p.symbol_to_id.find(ph);
        if (it != g2p.symbol_to_id.end()) {
            phone_ids.push_back(it->second);
            tone_ids.push_back(tone + tone_start);
            lang_ids.push_back(lang_id);
        }
    };

    // Leading pad
    add_phone("_", 0);
    word2ph.push_back(1); // pad token

    for (const auto& word : words) {
        int n_before = (int)phone_ids.size();

        // Check if it's punctuation
        if (word.size() == 1 && g2p.symbol_to_id.count(word)) {
            add_phone(word, 0);
        }
        // Lookup in CMU dict
        else {
            std::string upper_word = to_upper(word);
            auto dict_it = g2p.cmudict.find(upper_word);
            if (dict_it != g2p.cmudict.end()) {
                for (const auto& syllable : dict_it->second) {
                    for (const auto& ph : syllable) {
                        int tone = arpa_to_tone(ph);
                        std::string base = to_lower(arpa_strip_stress(ph));
                        std::string mapped = post_replace_ph(base, g2p.symbol_to_id);
                        add_phone(mapped, tone);
                    }
                }
            } else {
                // Try neural G2P first (g2p_en model), fall back to LTS rules
                bool used_neural = false;
                if (g2p.neural.loaded) {
                    auto preds = neural_g2p_predict(g2p.neural, word);
                    if (!preds.empty()) {
                        for (const auto& ph : preds) {
                            int tone = arpa_to_tone(ph);
                            std::string base = to_lower(arpa_strip_stress(ph));
                            std::string mapped = post_replace_ph(base, g2p.symbol_to_id);
                            add_phone(mapped, tone);
                        }
                        used_neural = true;
                    }
                }
                if (!used_neural)
                    lts_fallback(word, g2p.symbol_to_id, tone_start, lang_id, phone_ids, tone_ids, lang_ids);
            }
        }

        int n_phones = (int)phone_ids.size() - n_before;
        word2ph.push_back(n_phones);
    }

    // Trailing pad
    add_phone("_", 0);
    word2ph.push_back(1);

    // Intersperse with blanks: [0, e0, 0, e1, 0, ..., eN, 0]
    // Matches Python: result = [item] * (len*2+1); result[1::2] = lst
    // Blank positions get phone=0, tone=0, lang=0
    {
        int N = (int)phone_ids.size();
        std::vector<int> p2(2 * N + 1, 0);
        std::vector<int> t2(2 * N + 1, 0);
        std::vector<int> l2(2 * N + 1, 0); // blanks get lang=0
        for (int i = 0; i < N; i++) {
            p2[2 * i + 1] = phone_ids[i];
            t2[2 * i + 1] = tone_ids[i];
            l2[2 * i + 1] = lang_ids[i];
        }
        phone_ids = p2;
        tone_ids = t2;
        lang_ids = l2;

        // Double word2ph to account for interspersed blanks
        for (auto& w : word2ph)
            w *= 2;
        if (!word2ph.empty())
            word2ph[0] += 1; // leading pad gets +1
    }
}

// ── Hparams ───────────────────────────────────────────────────────

struct melotts_hparams {
    uint32_t hidden_channels = 192;
    uint32_t inter_channels = 192;
    uint32_t filter_channels = 768;
    uint32_t n_heads = 2;
    uint32_t head_dim = 96;
    uint32_t n_layers_enc = 6;
    uint32_t n_layers_trans_flow = 3;
    uint32_t n_flow_blocks = 4;
    uint32_t n_upsample_stages = 5;
    uint32_t upsample_initial_channel = 512;
    uint32_t num_symbols = 112;
    uint32_t num_tones = 11;
    uint32_t num_languages = 3;
    uint32_t n_speakers = 256;
    uint32_t gin_channels = 256;
    uint32_t sample_rate = 44100;
    uint32_t n_sdp_flows = 4;
    uint32_t sdp_num_bins = 10;
    uint32_t n_sdp_dds_layers = 3;

    float noise_scale = 0.667f;
    float length_scale = 1.0f;
    float noise_w = 0.8f;
    float sdp_ratio = 0.0f; // default 0 for disable-bert; 0.2 with BERT

    std::vector<uint32_t> upsample_rates;
    std::vector<uint32_t> upsample_kernels;
    std::vector<uint32_t> resblock_kernels;
    std::vector<std::vector<uint32_t>> resblock_dilations;
};

// ── Weight struct ─────────────────────────────────────────────────

struct melotts_enc_layer {
    ggml_tensor *conv_q_w, *conv_q_b;
    ggml_tensor *conv_k_w, *conv_k_b;
    ggml_tensor *conv_v_w, *conv_v_b;
    ggml_tensor *conv_o_w, *conv_o_b;
    ggml_tensor *emb_rel_k, *emb_rel_v;
    ggml_tensor *norm1_g, *norm1_b;
    ggml_tensor *ffn_c1_w, *ffn_c1_b;
    ggml_tensor *ffn_c2_w, *ffn_c2_b;
    ggml_tensor *norm2_g, *norm2_b;
};

struct melotts_dds_conv {
    struct layer {
        ggml_tensor *conv_sep_w, *conv_sep_b;
        ggml_tensor *conv_1x1_w, *conv_1x1_b;
        ggml_tensor *norm1_g, *norm1_b;
        ggml_tensor *norm2_g, *norm2_b;
    };
    std::vector<layer> layers;
};

struct melotts_sdp_convflow {
    ggml_tensor *pre_w, *pre_b;
    melotts_dds_conv dds;
    ggml_tensor *proj_w, *proj_b;
};

struct melotts_flow_coupling {
    // TransformerCouplingLayer: pre -> encoder(3-layer transformer) -> post
    ggml_tensor *pre_w, *pre_b;
    ggml_tensor *post_w, *post_b;
    // Speaker conditioning in flow encoder
    ggml_tensor *spk_emb_linear_w, *spk_emb_linear_b;
    // Encoder layers (n_layers_trans_flow)
    std::vector<melotts_enc_layer> enc_layers;
};

struct melotts_resblock {
    // ResBlock1: 3 conv pairs (convs1 + convs2) with dilations (1,3,5)
    ggml_tensor *conv1_0_w, *conv1_0_b;
    ggml_tensor *conv1_1_w, *conv1_1_b;
    ggml_tensor *conv1_2_w, *conv1_2_b;
    ggml_tensor *conv2_0_w, *conv2_0_b;
    ggml_tensor *conv2_1_w, *conv2_1_b;
    ggml_tensor *conv2_2_w, *conv2_2_b;
};

struct melotts_weights {
    // Text encoder
    ggml_tensor* emb;
    ggml_tensor* tone_emb;
    ggml_tensor* lang_emb;
    ggml_tensor *bert_proj_w, *bert_proj_b;
    ggml_tensor *ja_bert_proj_w, *ja_bert_proj_b;
    ggml_tensor *proj_w, *proj_b;
    // Speaker-conditioned encoder
    ggml_tensor *spk_emb_linear_w, *spk_emb_linear_b;
    std::vector<melotts_enc_layer> enc_layers;

    // Speaker embedding
    ggml_tensor* emb_g;

    // Stochastic duration predictor
    ggml_tensor *sdp_pre_w, *sdp_pre_b;
    ggml_tensor *sdp_proj_w, *sdp_proj_b;
    ggml_tensor *sdp_cond_w, *sdp_cond_b;
    melotts_dds_conv sdp_convs;
    ggml_tensor *sdp_flow0_m, *sdp_flow0_logs;
    std::vector<melotts_sdp_convflow> sdp_flows;

    // Duration predictor
    ggml_tensor *dp_conv1_w, *dp_conv1_b;
    ggml_tensor *dp_norm1_g, *dp_norm1_b;
    ggml_tensor *dp_conv2_w, *dp_conv2_b;
    ggml_tensor *dp_norm2_g, *dp_norm2_b;
    ggml_tensor *dp_proj_w, *dp_proj_b;
    ggml_tensor *dp_cond_w, *dp_cond_b;

    // TransformerCoupling flow
    std::vector<melotts_flow_coupling> flow_blocks;

    // HiFi-GAN decoder
    ggml_tensor *dec_conv_pre_w, *dec_conv_pre_b;
    ggml_tensor *dec_cond_w, *dec_cond_b;
    struct ups_stage {
        ggml_tensor *w, *b;
    };
    std::vector<ups_stage> dec_ups;
    std::vector<melotts_resblock> dec_resblocks;
    ggml_tensor* dec_conv_post_w;
};

// ── Context ───────────────────────────────────────────────────────

struct melotts_context {
    melotts_hparams hp;
    melotts_weights w;
    melotts_g2p g2p;

    // Optional BERT encoder for contextual conditioning
    struct bert_encoder_context* bert_ctx = nullptr;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    ggml_context* w_ctx = nullptr;
    ggml_backend_buffer_t w_buf = nullptr;

    float noise_scale, length_scale, noise_w, sdp_ratio;
    int speaker_id;
    int verbosity;
    int n_threads;
    uint32_t seed;
    std::string dump_dir;
};

// ── Diff harness ──────────────────────────────────────────────────

static void dump_stage(const melotts_context* ctx, const char* label, const float* data, size_t n) {
    if (ctx->dump_dir.empty())
        return;
    std::string path = ctx->dump_dir + "/" + label + ".bin";
    FILE* f = fopen(path.c_str(), "wb");
    if (f) {
        fwrite(data, sizeof(float), n, f);
        fclose(f);
    }
}

// ── Helper: mini_graph compute ────────────────────────────────────

namespace {
struct mini_graph {
    ggml_context* ctx = nullptr;
    ggml_backend_sched_t sched = nullptr;

    mini_graph(ggml_backend_sched_t sched_, size_t ctx_size = 16 * 1024 * 1024) : sched(sched_) {
        ggml_init_params params = {ctx_size, nullptr, true};
        ctx = ggml_init(params);
    }
    ~mini_graph() {
        if (ctx)
            ggml_free(ctx);
    }

    std::vector<float> compute(ggml_tensor* output, int) {
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, 16384, false);
        ggml_build_forward_expand(gf, output);
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            fprintf(stderr, "melotts: graph alloc failed\n");
            return {};
        }
        ggml_backend_sched_graph_compute(sched, gf);
        int n = (int)ggml_nelements(output);
        std::vector<float> result(n);
        ggml_backend_tensor_get(output, result.data(), 0, n * sizeof(float));
        return result;
    }
};
} // namespace

// ── Tensor read helper ────────────────────────────────────────────

static void read_tensor_f32(ggml_tensor* t, std::vector<float>& out) {
    const int64_t n = ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    } else {
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, ggml_nbytes(t));
        const auto to_float = ggml_get_type_traits(t->type)->to_float;
        if (to_float) {
            to_float(raw.data(), out.data(), n);
        } else {
            fprintf(stderr, "melotts: unsupported type %d for '%s'\n", (int)t->type, t->name);
            std::fill(out.begin(), out.end(), 0.0f);
        }
    }
}

// ── Layer norm (channels-first: C,T) ──────────────────────────────

static ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* gamma, ggml_tensor* beta) {
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_mul(ctx, x, gamma);
    x = ggml_add(ctx, x, beta);
    return x;
}

// ── Conv1d channels-first helper ──────────────────────────────────

static ggml_tensor* conv1d_cf(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride = 1,
                              int pad = 0, int dilation = 1) {
    ggml_tensor* xT = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor* y = ggml_conv_1d(ctx, w, xT, stride, pad, dilation);
    y = ggml_cont(ctx, ggml_transpose(ctx, y));
    if (b)
        y = ggml_add(ctx, y, b);
    return y;
}

// ── CPU attention with relative position bias ─────────────────────
// Identical to piper_tts.cpp but extracted for reuse.

static void cpu_multihead_attention_relpos(const std::vector<float>& x, // (C, T) row-major [t*C + c]
                                           const melotts_enc_layer& layer, int C, int T, int H, int D,
                                           int W, // W=window_size(4)
                                           const float* spk_cond, int spk_dim, int cond_layer_idx, int cur_layer,
                                           std::vector<float>& out) {
    // 1. Linear projections via 1x1 conv weights
    std::vector<float> wq, bq, wk, bk, wv, bv;
    read_tensor_f32(layer.conv_q_w, wq);
    read_tensor_f32(layer.conv_q_b, bq);
    read_tensor_f32(layer.conv_k_w, wk);
    read_tensor_f32(layer.conv_k_b, bk);
    read_tensor_f32(layer.conv_v_w, wv);
    read_tensor_f32(layer.conv_v_b, bv);

    // Working copy of x (may have speaker conditioning injected)
    std::vector<float> x_in = x;

    // Speaker injection at cond_layer_idx (default 2 for MeloTTS)
    if (spk_cond && cur_layer == cond_layer_idx && spk_dim > 0) {
        // spk_emb_linear: (hidden, gin_channels) linear, already computed
        // Add speaker embedding to each timestep
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                x_in[t * C + c] += spk_cond[c]; // broadcast (C,) over T
            }
        }
    }

    std::vector<float> Q(C * T), K(C * T), V(C * T);
    for (int t = 0; t < T; t++) {
        for (int co = 0; co < C; co++) {
            float sq = bq[co], sk = bk[co], sv = bv[co];
            for (int ci = 0; ci < C; ci++) {
                float xv = x_in[t * C + ci];
                sq += xv * wq[ci + co * C];
                sk += xv * wk[ci + co * C];
                sv += xv * wv[ci + co * C];
            }
            Q[t * C + co] = sq;
            K[t * C + co] = sk;
            V[t * C + co] = sv;
        }
    }

    // 2. Relative position embeddings
    int n_rel = 2 * W + 1;
    std::vector<float> rel_k, rel_v;
    read_tensor_f32(layer.emb_rel_k, rel_k);
    read_tensor_f32(layer.emb_rel_v, rel_v);

    float inv_sqrt_d = 1.0f / sqrtf((float)D);
    out.resize(C * T, 0.0f);

    for (int h = 0; h < H; h++) {
        int ch_off = h * D;

        std::vector<float> scores(T * T);
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                float s = 0;
                for (int d = 0; d < D; d++) {
                    s += Q[i * C + ch_off + d] * K[j * C + ch_off + d];
                }
                s *= inv_sqrt_d;
                int r = j - i + W;
                if (r >= 0 && r < n_rel) {
                    float rel_s = 0;
                    for (int d = 0; d < D; d++) {
                        rel_s += Q[i * C + ch_off + d] * rel_k[d + r * D];
                    }
                    s += rel_s * inv_sqrt_d;
                }
                scores[i * T + j] = s;
            }
        }

        // Softmax
        for (int i = 0; i < T; i++) {
            float max_s = scores[i * T];
            for (int j = 1; j < T; j++)
                if (scores[i * T + j] > max_s)
                    max_s = scores[i * T + j];
            float sum_e = 0;
            for (int j = 0; j < T; j++) {
                scores[i * T + j] = expf(scores[i * T + j] - max_s);
                sum_e += scores[i * T + j];
            }
            for (int j = 0; j < T; j++)
                scores[i * T + j] /= sum_e;
        }

        // Weighted sum with relative value bias
        for (int i = 0; i < T; i++) {
            for (int d = 0; d < D; d++) {
                float s = 0;
                for (int j = 0; j < T; j++) {
                    float w_ij = scores[i * T + j];
                    s += w_ij * V[j * C + ch_off + d];
                    int r = j - i + W;
                    if (r >= 0 && r < n_rel) {
                        s += w_ij * rel_v[d + r * D];
                    }
                }
                out[i * C + ch_off + d] = s;
            }
        }
    }
}

// ── CPU helpers ───────────────────────────────────────────────────

static void cpu_conv1x1(const std::vector<float>& x, ggml_tensor* w_t, ggml_tensor* b_t, int C_in, int C_out, int T,
                        std::vector<float>& out) {
    std::vector<float> w, b;
    read_tensor_f32(w_t, w);
    read_tensor_f32(b_t, b);
    out.resize(C_out * T);
    for (int t = 0; t < T; t++) {
        for (int co = 0; co < C_out; co++) {
            float s = b[co];
            for (int ci = 0; ci < C_in; ci++) {
                s += x[t * C_in + ci] * w[ci + co * C_in];
            }
            out[t * C_out + co] = s;
        }
    }
}

static void cpu_layer_norm(std::vector<float>& x, ggml_tensor* g_t, ggml_tensor* b_t, int C, int T) {
    std::vector<float> g, b;
    read_tensor_f32(g_t, g);
    read_tensor_f32(b_t, b);
    for (int t = 0; t < T; t++) {
        float mean = 0, var = 0;
        for (int c = 0; c < C; c++)
            mean += x[t * C + c];
        mean /= C;
        for (int c = 0; c < C; c++) {
            float d = x[t * C + c] - mean;
            var += d * d;
        }
        var /= C;
        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (int c = 0; c < C; c++) {
            x[t * C + c] = (x[t * C + c] - mean) * inv_std * g[c] + b[c];
        }
    }
}

// ── Text Encoder forward ──────────────────────────────────────────

// ja_bert_features: (768, T) row-major, or empty for disable-bert mode
static void text_encoder_forward(melotts_context* ctx, const std::vector<int>& phone_ids,
                                 const std::vector<int>& tone_ids, const std::vector<int>& lang_ids,
                                 const std::vector<float>& ja_bert_features, std::vector<float>& out_enc,
                                 std::vector<float>& out_mean, std::vector<float>& out_logvar) {
    const auto& hp = ctx->hp;
    const auto& w = ctx->w;
    const int T = (int)phone_ids.size();
    const int C = (int)hp.hidden_channels;
    const int half = (int)hp.inter_channels;
    const int H = (int)hp.n_heads;
    const int D = (int)hp.head_dim;
    const int W = 4; // window_size

    if (ctx->verbosity >= 2)
        fprintf(stderr, "melotts: text_encoder T=%d C=%d\n", T, C);

    // ── Embedding lookup: phone + tone + lang, scaled by sqrt(C) ──
    float sqrt_c = sqrtf((float)C);

    auto read_emb = [](ggml_tensor* t, std::vector<float>& table) {
        int64_t n = ggml_nelements(t);
        table.resize(n);
        size_t bytes = ggml_nbytes(t);
        std::vector<uint8_t> raw(bytes);
        ggml_backend_tensor_get(t, raw.data(), 0, bytes);
        if (t->type == GGML_TYPE_F16) {
            const ggml_fp16_t* src = (const ggml_fp16_t*)raw.data();
            for (int64_t i = 0; i < n; i++)
                table[i] = ggml_fp16_to_fp32(src[i]);
        } else {
            memcpy(table.data(), raw.data(), n * sizeof(float));
        }
    };

    std::vector<float> phone_table, tone_table, lang_table;
    read_emb(w.emb, phone_table);
    read_emb(w.tone_emb, tone_table);
    read_emb(w.lang_emb, lang_table);

    int emb_dim = (int)w.emb->ne[0]; // C

    std::vector<float> x(C * T);
    for (int t = 0; t < T; t++) {
        int pid = phone_ids[t];
        int tid = tone_ids[t];
        int lid = lang_ids[t];
        // Clamp
        if (pid < 0 || pid >= (int)hp.num_symbols)
            pid = 0;
        if (tid < 0 || tid >= (int)hp.num_tones)
            tid = 0;
        if (lid < 0 || lid >= (int)hp.num_languages)
            lid = 0;

        for (int c = 0; c < C; c++) {
            float v = phone_table[pid * emb_dim + c] + tone_table[tid * emb_dim + c] + lang_table[lid * emb_dim + c];
            x[t * C + c] = v * sqrt_c;
        }
    }

    // BERT conditioning: project ja_bert features (768→192) + bert_proj bias
    // For English: bert(1024) is always zeros, so only bias contributes.
    // ja_bert(768) has actual features when BERT model is loaded.
    {
        std::vector<float> bp_bias;
        if (w.bert_proj_b)
            read_tensor_f32(w.bert_proj_b, bp_bias);

        // bert_proj bias (1024-dim input is zero, only bias matters)
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C && c < (int)bp_bias.size(); c++)
                x[t * C + c] += bp_bias[c] * sqrt_c;

        // ja_bert_proj: Conv1d(768, hidden, 1) applied to ja_bert features
        if (!ja_bert_features.empty() && w.ja_bert_proj_w && w.ja_bert_proj_b) {
            // ja_bert_features is (768, T) row-major = [c * T + t]
            // ja_bert_proj is Conv1d(768, 192, 1) — 1x1 conv = linear
            std::vector<float> jbp_w, jbp_b;
            read_tensor_f32(w.ja_bert_proj_w, jbp_w);
            read_tensor_f32(w.ja_bert_proj_b, jbp_b);
            int bert_dim = 768;
            for (int t = 0; t < T; t++) {
                for (int oc = 0; oc < C; oc++) {
                    float sum = jbp_b[oc];
                    for (int ic = 0; ic < bert_dim; ic++)
                        sum += ja_bert_features[ic * T + t] * jbp_w[ic + oc * bert_dim];
                    x[t * C + oc] += sum * sqrt_c;
                }
            }
        } else {
            // No BERT features — just add ja_bert_proj bias
            std::vector<float> jbp_bias;
            if (w.ja_bert_proj_b)
                read_tensor_f32(w.ja_bert_proj_b, jbp_bias);
            for (int t = 0; t < T; t++)
                for (int c = 0; c < C && c < (int)jbp_bias.size(); c++)
                    x[t * C + c] += jbp_bias[c] * sqrt_c;
        }
    }

    dump_stage(ctx, "emb_output", x.data(), x.size());

    // ── Speaker embedding for encoder conditioning ──
    // g = emb_g[speaker_id] -> (gin_channels,)
    // spk_linear = spk_emb_linear(g) -> (hidden_channels,)
    std::vector<float> spk_cond;
    if (w.spk_emb_linear_w) {
        std::vector<float> g_table;
        read_emb(w.emb_g, g_table);
        int gin = (int)hp.gin_channels;

        std::vector<float> g_vec(gin);
        for (int c = 0; c < gin; c++) {
            g_vec[c] = g_table[ctx->speaker_id * gin + c];
        }

        // spk_emb_linear: (hidden, gin) matmul
        std::vector<float> slw, slb;
        read_tensor_f32(w.spk_emb_linear_w, slw);
        read_tensor_f32(w.spk_emb_linear_b, slb);

        spk_cond.resize(C);
        for (int co = 0; co < C; co++) {
            float s = slb[co];
            for (int ci = 0; ci < gin; ci++) {
                // Linear weight (out, in) -> flat[ci + co * gin]
                s += g_vec[ci] * slw[ci + co * gin];
            }
            spk_cond[co] = s;
        }
    }

    // ── Encoder layers ──
    int cond_layer_idx = 2; // MeloTTS uses use_spk_conditioned_encoder with cond_layer_idx=2

    for (uint32_t il = 0; il < hp.n_layers_enc; il++) {
        const auto& layer = w.enc_layers[il];

        // Speaker conditioning injection at cond_layer_idx
        if ((int)il == cond_layer_idx && !spk_cond.empty()) {
            for (int t = 0; t < T; t++)
                for (int c = 0; c < C; c++)
                    x[t * C + c] += spk_cond[c];
        }

        // Attention with relative position bias (CPU)
        std::vector<float> attn_out;
        cpu_multihead_attention_relpos(x, layer, C, T, H, D, W, nullptr, 0, -1, (int)il, attn_out);

        // Output projection
        std::vector<float> o;
        cpu_conv1x1(attn_out, layer.conv_o_w, layer.conv_o_b, C, C, T, o);

        // Residual + norm
        for (int i = 0; i < C * T; i++)
            x[i] += o[i];
        cpu_layer_norm(x, layer.norm1_g, layer.norm1_b, C, T);

        // FFN via ggml graph (detect kernel size from weight shape)
        {
            int ffn_k = (int)layer.ffn_c1_w->ne[0]; // ne[0] = K for 3D conv weight
            int ffn_pad = (ffn_k - 1) / 2;          // same-padding

            mini_graph mg(ctx->sched, 4 * 1024 * 1024);
            auto* gc = mg.ctx;

            ggml_tensor* x_in = ggml_new_tensor_2d(gc, GGML_TYPE_F32, C, T);
            ggml_set_name(x_in, "ffn_in");
            ggml_set_input(x_in);

            ggml_tensor* ff = conv1d_cf(gc, x_in, layer.ffn_c1_w, layer.ffn_c1_b, 1, ffn_pad, 1);
            ff = ggml_relu(gc, ff);
            ff = conv1d_cf(gc, ff, layer.ffn_c2_w, layer.ffn_c2_b, 1, ffn_pad, 1);

            ggml_cgraph* gf = ggml_new_graph_custom(gc, 1024, false);
            ggml_build_forward_expand(gf, ff);
            ggml_backend_sched_reset(mg.sched);
            if (!ggml_backend_sched_alloc_graph(mg.sched, gf)) {
                fprintf(stderr, "melotts: FFN graph alloc failed\n");
                return;
            }
            ggml_backend_tensor_set(x_in, x.data(), 0, C * T * sizeof(float));
            ggml_backend_sched_graph_compute(mg.sched, gf);

            std::vector<float> ff_out(C * T);
            ggml_backend_tensor_get(ff, ff_out.data(), 0, C * T * sizeof(float));

            for (int i = 0; i < C * T; i++)
                x[i] += ff_out[i];
            cpu_layer_norm(x, layer.norm2_g, layer.norm2_b, C, T);
        }
    }

    out_enc = x;
    dump_stage(ctx, "enc_output", x.data(), x.size());

    // ── Final projection for mean/logvar ──
    {
        mini_graph mg(ctx->sched, 4 * 1024 * 1024);
        auto* gc = mg.ctx;

        ggml_tensor* x_in = ggml_new_tensor_2d(gc, GGML_TYPE_F32, C, T);
        ggml_set_name(x_in, "proj_in");
        ggml_set_input(x_in);

        ggml_tensor* proj = conv1d_cf(gc, x_in, w.proj_w, w.proj_b);

        ggml_cgraph* gf = ggml_new_graph_custom(gc, 256, false);
        ggml_build_forward_expand(gf, proj);
        ggml_backend_sched_reset(mg.sched);
        if (!ggml_backend_sched_alloc_graph(mg.sched, gf))
            return;
        ggml_backend_tensor_set(x_in, x.data(), 0, C * T * sizeof(float));
        ggml_backend_sched_graph_compute(mg.sched, gf);

        int proj_size = 2 * half * T;
        std::vector<float> proj_data(proj_size);
        ggml_backend_tensor_get(proj, proj_data.data(), 0, proj_size * sizeof(float));

        out_mean.resize(half * T);
        out_logvar.resize(half * T);
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < half; c++) {
                out_mean[c + t * half] = proj_data[c + t * 2 * half];
                out_logvar[c + t * half] = proj_data[c + half + t * 2 * half];
            }
        }
    }
    dump_stage(ctx, "enc_mean", out_mean.data(), out_mean.size());
    dump_stage(ctx, "enc_logvar", out_logvar.data(), out_logvar.size());
}

// ── DDSConv forward ───────────────────────────────────────────────
// Reused by SDP, identical to piper_tts.cpp

static void dds_conv_forward(const melotts_dds_conv& dds, const std::vector<float>& x_in, int C, int T,
                             std::vector<float>& out) {
    out = x_in;
    for (size_t il = 0; il < dds.layers.size(); il++) {
        const auto& layer = dds.layers[il];
        int K = 3;
        int dilation = 1;
        for (size_t p = 0; p < il; p++)
            dilation *= K;
        int pad = dilation;

        std::vector<float> w_sep, b_sep;
        read_tensor_f32(layer.conv_sep_w, w_sep);
        read_tensor_f32(layer.conv_sep_b, b_sep);

        std::vector<float> padded(C * (T + 2 * pad), 0.0f);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++)
                padded[(t + pad) * C + c] = out[t * C + c];

        std::vector<float> dw_out(C * T, 0.0f);
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                float sum = b_sep[c];
                for (int k = 0; k < K; k++) {
                    int ti = t + k * dilation;
                    sum += padded[ti * C + c] * w_sep[k + c * K];
                }
                dw_out[t * C + c] = sum;
            }
        }

        // Norm1 + GELU
        std::vector<float> n1g, n1b;
        read_tensor_f32(layer.norm1_g, n1g);
        read_tensor_f32(layer.norm1_b, n1b);
        for (int t = 0; t < T; t++) {
            float mean = 0, var = 0;
            for (int c = 0; c < C; c++)
                mean += dw_out[t * C + c];
            mean /= C;
            for (int c = 0; c < C; c++) {
                float d = dw_out[t * C + c] - mean;
                var += d * d;
            }
            var /= C;
            for (int c = 0; c < C; c++)
                dw_out[t * C + c] = (dw_out[t * C + c] - mean) / sqrtf(var + 1e-5f) * n1g[c] + n1b[c];
        }
        for (int i = 0; i < C * T; i++) {
            float v = dw_out[i];
            dw_out[i] = v * 0.5f * (1.0f + erff(v * 0.7071067811865476f));
        }

        // 1x1 conv
        std::vector<float> w_1x1, b_1x1;
        read_tensor_f32(layer.conv_1x1_w, w_1x1);
        read_tensor_f32(layer.conv_1x1_b, b_1x1);

        std::vector<float> lin_out(C * T, 0.0f);
        for (int t = 0; t < T; t++) {
            for (int co = 0; co < C; co++) {
                float sum = b_1x1[co];
                for (int ci = 0; ci < C; ci++)
                    sum += dw_out[t * C + ci] * w_1x1[ci + co * C];
                lin_out[t * C + co] = sum;
            }
        }

        // Norm2 + GELU + residual
        std::vector<float> n2g, n2b;
        read_tensor_f32(layer.norm2_g, n2g);
        read_tensor_f32(layer.norm2_b, n2b);
        for (int t = 0; t < T; t++) {
            float mean = 0, var = 0;
            for (int c = 0; c < C; c++)
                mean += lin_out[t * C + c];
            mean /= C;
            for (int c = 0; c < C; c++) {
                float d = lin_out[t * C + c] - mean;
                var += d * d;
            }
            var /= C;
            for (int c = 0; c < C; c++)
                lin_out[t * C + c] = (lin_out[t * C + c] - mean) / sqrtf(var + 1e-5f) * n2g[c] + n2b[c];
        }
        for (int i = 0; i < C * T; i++) {
            float v = lin_out[i];
            float g = v * 0.5f * (1.0f + erff(v * 0.7071067811865476f));
            out[i] += g;
        }
    }
}

// ── RQS spline transforms ────────────────────────────────────────
// Identical to piper_tts.cpp

static float rqs_inverse(float y_in, const float* w_bins, const float* h_bins, const float* d_knots, int num_bins,
                         float range_min, float range_max) {
    std::vector<float> widths(num_bins), heights(num_bins), derivatives(num_bins + 1);

    float max_w = *std::max_element(w_bins, w_bins + num_bins);
    float sum_w = 0;
    for (int i = 0; i < num_bins; i++) {
        widths[i] = expf(w_bins[i] - max_w);
        sum_w += widths[i];
    }
    for (int i = 0; i < num_bins; i++)
        widths[i] = widths[i] / sum_w * (range_max - range_min) + 1e-5f;

    float max_h = *std::max_element(h_bins, h_bins + num_bins);
    float sum_h = 0;
    for (int i = 0; i < num_bins; i++) {
        heights[i] = expf(h_bins[i] - max_h);
        sum_h += heights[i];
    }
    for (int i = 0; i < num_bins; i++)
        heights[i] = heights[i] / sum_h * (range_max - range_min) + 1e-5f;

    derivatives[0] = 1.0f;
    derivatives[num_bins] = 1.0f;
    for (int i = 0; i < num_bins - 1; i++)
        derivatives[i + 1] = logf(1.0f + expf(d_knots[i])) + 1e-5f;

    std::vector<float> cum_w(num_bins + 1), cum_h(num_bins + 1);
    cum_w[0] = range_min;
    cum_h[0] = range_min;
    for (int i = 0; i < num_bins; i++) {
        cum_w[i + 1] = cum_w[i] + widths[i];
        cum_h[i + 1] = cum_h[i] + heights[i];
    }

    int bin_idx = num_bins - 1;
    for (int i = 0; i < num_bins; i++) {
        if (y_in < cum_h[i + 1]) {
            bin_idx = i;
            break;
        }
    }

    float w_k = widths[bin_idx], h_k = heights[bin_idx];
    float d_k = derivatives[bin_idx], d_k1 = derivatives[bin_idx + 1];
    float s_k = h_k / (w_k + 1e-10f);
    float y_yk = y_in - cum_h[bin_idx];
    float t2 = d_k1 + d_k - 2.0f * s_k;

    float a = h_k * (s_k - d_k) + y_yk * t2;
    float b = h_k * d_k - y_yk * t2;
    float c = -s_k * y_yk;
    float disc = b * b - 4.0f * a * c;
    if (disc < 0)
        disc = 0;
    float xi = (2.0f * c) / (-b - sqrtf(disc + 1e-10f));
    xi = std::clamp(xi, 0.0f, 1.0f);

    return xi * w_k + cum_w[bin_idx];
}

// ── SDP ConvFlow inverse ──────────────────────────────────────────

static void sdp_convflow_inverse(melotts_context* /*ctx*/, const melotts_sdp_convflow& cf,
                                 std::vector<float>& z,       // (2, T)
                                 const std::vector<float>& h, // (C, T)
                                 int C, int T, int num_bins) {
    std::vector<float> z0(T);
    for (int t = 0; t < T; t++)
        z0[t] = z[t * 2 + 0];

    // pre conv (1→C) + conditioning
    std::vector<float> w_pre, b_pre;
    read_tensor_f32(cf.pre_w, w_pre);
    read_tensor_f32(cf.pre_b, b_pre);

    std::vector<float> h_cond(C * T);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            h_cond[t * C + c] = z0[t] * w_pre[c] + b_pre[c] + h[t * C + c];

    // DDSConv
    std::vector<float> h_out;
    dds_conv_forward(cf.dds, h_cond, C, T, h_out);

    // Project to spline params
    int n_params = 3 * num_bins - 1;
    std::vector<float> w_proj, b_proj;
    read_tensor_f32(cf.proj_w, w_proj);
    read_tensor_f32(cf.proj_b, b_proj);

    std::vector<float> params(n_params * T);
    for (int t = 0; t < T; t++) {
        for (int p = 0; p < n_params; p++) {
            float sum = b_proj[p];
            for (int c = 0; c < C; c++)
                sum += h_out[t * C + c] * w_proj[c + p * C];
            params[t * n_params + p] = sum;
        }
    }

    // Scale widths/heights by 1/sqrt(C)
    float inv_sqrt_fc = 1.0f / sqrtf((float)C);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < 2 * num_bins; i++)
            params[t * n_params + i] *= inv_sqrt_fc;

    // Apply spline inverse to z1
    float range_min = -5.0f, range_max = 5.0f;
    for (int t = 0; t < T; t++) {
        float z1 = z[t * 2 + 1];
        if (z1 <= range_min || z1 >= range_max)
            continue;
        const float* p = &params[t * n_params];
        z[t * 2 + 1] = rqs_inverse(z1, p, p + num_bins, p + 2 * num_bins, num_bins, range_min, range_max);
    }
}

// ── SDP forward (reverse mode for inference) ──────────────────────

static void sdp_forward(melotts_context* ctx,
                        const std::vector<float>& h_enc, // (hidden, T)
                        const std::vector<float>& g_vec, // (gin,) speaker
                        int T, float noise_w, std::vector<float>& log_durations) {
    const auto& hp = ctx->hp;
    const auto& w = ctx->w;
    int C = (int)hp.hidden_channels;
    int num_bins = (int)hp.sdp_num_bins;
    int gin = (int)hp.gin_channels;

    // pre conv + speaker conditioning
    std::vector<float> w_pre, b_pre;
    read_tensor_f32(w.sdp_pre_w, w_pre);
    read_tensor_f32(w.sdp_pre_b, b_pre);

    std::vector<float> h(C * T, 0.0f);
    for (int t = 0; t < T; t++) {
        for (int co = 0; co < C; co++) {
            float s = b_pre[co];
            for (int ci = 0; ci < C; ci++)
                s += h_enc[t * C + ci] * w_pre[ci + co * C];
            h[t * C + co] = s;
        }
    }

    // Add speaker conditioning: sdp.cond(g)
    if (w.sdp_cond_w && !g_vec.empty()) {
        std::vector<float> wc, bc;
        read_tensor_f32(w.sdp_cond_w, wc);
        read_tensor_f32(w.sdp_cond_b, bc);
        std::vector<float> g_proj(C);
        for (int co = 0; co < C; co++) {
            float s = bc[co];
            for (int ci = 0; ci < gin; ci++)
                s += g_vec[ci] * wc[ci + co * gin];
            g_proj[co] = s;
        }
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++)
                h[t * C + c] += g_proj[c];
    }

    // DDSConv
    std::vector<float> h_dds;
    dds_conv_forward(w.sdp_convs, h, C, T, h_dds);

    // proj
    std::vector<float> w_proj, b_proj;
    read_tensor_f32(w.sdp_proj_w, w_proj);
    read_tensor_f32(w.sdp_proj_b, b_proj);

    std::vector<float> h_cond(C * T, 0.0f);
    for (int t = 0; t < T; t++) {
        for (int co = 0; co < C; co++) {
            float s = b_proj[co];
            for (int ci = 0; ci < C; ci++)
                s += h_dds[t * C + ci] * w_proj[ci + co * C];
            h_cond[t * C + co] = s;
        }
    }

    // Generate noise
    std::vector<float> z(2 * T, 0.0f);
    if (noise_w > 0) {
        std::mt19937 gen(ctx->seed + 1);
        std::normal_distribution<float> dist(0.0f, noise_w);
        for (int i = 0; i < 2 * T; i++)
            z[i] = dist(gen);
    }

    // Run SDP flows in reverse
    for (int fi = (int)w.sdp_flows.size() - 1; fi >= 0; fi--) {
        sdp_convflow_inverse(ctx, w.sdp_flows[fi], z, h_cond, C, T, num_bins);
        // Flip
        for (int t = 0; t < T; t++)
            std::swap(z[t * 2 + 0], z[t * 2 + 1]);
    }

    // ElementwiseAffine inverse
    if (w.sdp_flow0_m) {
        std::vector<float> ea_m, ea_logs;
        read_tensor_f32(w.sdp_flow0_m, ea_m);
        if (w.sdp_flow0_logs)
            read_tensor_f32(w.sdp_flow0_logs, ea_logs);

        for (int t = 0; t < T; t++) {
            for (int c = 0; c < 2; c++) {
                float m_val = (c < (int)ea_m.size()) ? ea_m[c] : 0.0f;
                float neg_logs = (w.sdp_flow0_logs && c < (int)ea_logs.size()) ? -ea_logs[c] : 0.0f;
                z[t * 2 + c] = (z[t * 2 + c] - m_val) * expf(neg_logs);
            }
        }
    }

    log_durations.resize(T);
    for (int t = 0; t < T; t++)
        log_durations[t] = z[t * 2 + 0];

    dump_stage(ctx, "sdp_logw", log_durations.data(), T);
}

// ── Duration Predictor forward ────────────────────────────────────

static void dp_forward(melotts_context* ctx,
                       const std::vector<float>& h_enc, // (hidden, T)
                       const std::vector<float>& g_vec, // (gin,)
                       int T,
                       std::vector<float>& log_durations) // (T,)
{
    const auto& hp = ctx->hp;
    const auto& w = ctx->w;
    int C = (int)hp.hidden_channels;
    (void)hp; // DP filter_channels inferred from weights
    int gin = (int)hp.gin_channels;

    // Add speaker conditioning: dp.cond(g)
    std::vector<float> x = h_enc;
    if (w.dp_cond_w && !g_vec.empty()) {
        std::vector<float> wc, bc;
        read_tensor_f32(w.dp_cond_w, wc);
        read_tensor_f32(w.dp_cond_b, bc);
        std::vector<float> g_proj(C);
        for (int co = 0; co < C; co++) {
            float s = bc[co];
            for (int ci = 0; ci < gin; ci++)
                s += g_vec[ci] * wc[ci + co * gin];
            g_proj[co] = s;
        }
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++)
                x[t * C + c] += g_proj[c];
    }

    // DP via ggml graph: conv1(C->FC, k=3, pad=1) -> ReLU -> norm1
    //                     -> conv2(FC->FC, k=3, pad=1) -> ReLU -> norm2
    //                     -> proj(FC->1, k=1)
    {
        mini_graph mg(ctx->sched, 4 * 1024 * 1024);
        auto* gc = mg.ctx;

        ggml_tensor* x_in = ggml_new_tensor_2d(gc, GGML_TYPE_F32, C, T);
        ggml_set_name(x_in, "dp_in");
        ggml_set_input(x_in);

        ggml_tensor* h = conv1d_cf(gc, x_in, w.dp_conv1_w, w.dp_conv1_b, 1, 1, 1);
        h = ggml_relu(gc, h);
        h = layer_norm(gc, h, w.dp_norm1_g, w.dp_norm1_b);
        h = conv1d_cf(gc, h, w.dp_conv2_w, w.dp_conv2_b, 1, 1, 1);
        h = ggml_relu(gc, h);
        h = layer_norm(gc, h, w.dp_norm2_g, w.dp_norm2_b);
        h = conv1d_cf(gc, h, w.dp_proj_w, w.dp_proj_b);

        ggml_cgraph* gf = ggml_new_graph_custom(gc, 1024, false);
        ggml_build_forward_expand(gf, h);
        ggml_backend_sched_reset(mg.sched);
        if (!ggml_backend_sched_alloc_graph(mg.sched, gf)) {
            fprintf(stderr, "melotts: DP graph alloc failed\n");
            log_durations.assign(T, 0.0f);
            return;
        }
        ggml_backend_tensor_set(x_in, x.data(), 0, C * T * sizeof(float));
        ggml_backend_sched_graph_compute(mg.sched, gf);

        log_durations.resize(T);
        ggml_backend_tensor_get(h, log_durations.data(), 0, T * sizeof(float));
    }
    dump_stage(ctx, "dp_logw", log_durations.data(), T);
}

// ── Duration alignment ────────────────────────────────────────────

static void duration_align(const std::vector<float>& enc_out, const std::vector<int>& durations, int C, int T_text,
                           std::vector<float>& out) {
    int T_audio = 0;
    for (int d : durations)
        T_audio += d;
    out.resize(C * T_audio);
    int pos = 0;
    for (int t = 0; t < T_text; t++) {
        for (int d = 0; d < durations[t]; d++) {
            for (int c = 0; c < C; c++)
                out[pos * C + c] = enc_out[t * C + c];
            pos++;
        }
    }
}

// ── TransformerCoupling Flow (inverse) ────────────────────────────
// MeloTTS uses Transformer attention (not WaveNet) in the coupling layers.

static void transformer_coupling_forward(melotts_context* ctx, const melotts_flow_coupling& fb,
                                         const std::vector<float>& x_in,  // (hidden, T) after pre conv
                                         const std::vector<float>& g_vec, // (gin,) speaker
                                         int hidden, int T, int n_layers_tf, std::vector<float>& out) {
    const auto& hp = ctx->hp;
    int H = (int)hp.n_heads;
    int D = (int)hp.head_dim;
    int W = 4;

    // Run transformer encoder layers
    std::vector<float> x = x_in;
    int gin = (int)ctx->hp.gin_channels;

    // Compute speaker conditioning for this flow block's encoder
    std::vector<float> spk_cond_flow;
    if (fb.spk_emb_linear_w && !g_vec.empty()) {
        std::vector<float> slw, slb;
        read_tensor_f32(fb.spk_emb_linear_w, slw);
        read_tensor_f32(fb.spk_emb_linear_b, slb);
        spk_cond_flow.resize(hidden);
        for (int co = 0; co < hidden; co++) {
            float s = slb[co];
            for (int ci = 0; ci < gin; ci++)
                s += g_vec[ci] * slw[ci + co * gin];
            spk_cond_flow[co] = s;
        }
    }

    int cond_layer_idx = 2; // MeloTTS flow encoder also injects at layer 2

    // Determine FFN kernel size from weight shape
    // ggml ne[0] = K for 3D conv weights
    int ffn_k = 1;
    if (fb.enc_layers.size() > 0 && fb.enc_layers[0].ffn_c1_w) {
        ffn_k = (int)fb.enc_layers[0].ffn_c1_w->ne[0];
    }
    int ffn_pad = (ffn_k - 1) / 2; // same padding

    for (int il = 0; il < n_layers_tf; il++) {
        const auto& layer = fb.enc_layers[il];

        // Speaker conditioning at cond_layer_idx
        if (il == cond_layer_idx && !spk_cond_flow.empty()) {
            for (int t = 0; t < T; t++)
                for (int c = 0; c < hidden; c++)
                    x[t * hidden + c] += spk_cond_flow[c];
        }

        // Attention
        std::vector<float> attn_out;
        cpu_multihead_attention_relpos(x, layer, hidden, T, H, D, W, nullptr, 0, -1, il, attn_out);

        std::vector<float> o;
        cpu_conv1x1(attn_out, layer.conv_o_w, layer.conv_o_b, hidden, hidden, T, o);

        for (int i = 0; i < hidden * T; i++)
            x[i] += o[i];
        cpu_layer_norm(x, layer.norm1_g, layer.norm1_b, hidden, T);

        // FFN (kernel size may differ: text encoder uses k=3, flow uses k=5)
        {
            mini_graph mg(ctx->sched, 4 * 1024 * 1024);
            auto* gc = mg.ctx;

            ggml_tensor* x_t = ggml_new_tensor_2d(gc, GGML_TYPE_F32, hidden, T);
            ggml_set_name(x_t, "tf_ffn_in");
            ggml_set_input(x_t);

            ggml_tensor* ff = conv1d_cf(gc, x_t, layer.ffn_c1_w, layer.ffn_c1_b, 1, ffn_pad, 1);
            ff = ggml_relu(gc, ff);
            ff = conv1d_cf(gc, ff, layer.ffn_c2_w, layer.ffn_c2_b, 1, ffn_pad, 1);

            ggml_cgraph* gf = ggml_new_graph_custom(gc, 1024, false);
            ggml_build_forward_expand(gf, ff);
            ggml_backend_sched_reset(mg.sched);
            if (!ggml_backend_sched_alloc_graph(mg.sched, gf))
                return;
            ggml_backend_tensor_set(x_t, x.data(), 0, hidden * T * sizeof(float));
            ggml_backend_sched_graph_compute(mg.sched, gf);

            std::vector<float> ff_out(hidden * T);
            ggml_backend_tensor_get(ff, ff_out.data(), 0, hidden * T * sizeof(float));

            for (int i = 0; i < hidden * T; i++)
                x[i] += ff_out[i];
            cpu_layer_norm(x, layer.norm2_g, layer.norm2_b, hidden, T);
        }
    }

    out = x;
}

static void flow_inverse(melotts_context* ctx,
                         std::vector<float>& z, // (inter, T)
                         const std::vector<float>& g_vec, int T) {
    const auto& hp = ctx->hp;
    int C = (int)hp.inter_channels;
    int half = C / 2;
    int hidden = C; // flow encoder hidden = inter_channels
    int n_tf_layers = (int)hp.n_layers_trans_flow;

    for (int fi = (int)ctx->w.flow_blocks.size() - 1; fi >= 0; fi--) {
        const auto& fb = ctx->w.flow_blocks[fi];

        if (ctx->verbosity >= 2)
            fprintf(stderr, "melotts: flow block %d (reverse)\n", fi);

        // Flip
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C / 2; c++)
                std::swap(z[t * C + c], z[t * C + C - 1 - c]);

        // Split
        std::vector<float> z0(half * T), z1(half * T);
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < half; c++) {
                z0[t * half + c] = z[t * C + c];
                z1[t * half + c] = z[t * C + half + c];
            }
        }

        // Pre conv (half → hidden)
        std::vector<float> w_pre, b_pre;
        read_tensor_f32(fb.pre_w, w_pre);
        read_tensor_f32(fb.pre_b, b_pre);

        std::vector<float> h(hidden * T, 0.0f);
        for (int t = 0; t < T; t++) {
            for (int oc = 0; oc < hidden; oc++) {
                float sum = b_pre[oc];
                for (int ic = 0; ic < half; ic++)
                    sum += z0[t * half + ic] * w_pre[ic + oc * half];
                h[t * hidden + oc] = sum;
            }
        }

        // Transformer encoder (instead of WaveNet)
        std::vector<float> enc_out;
        transformer_coupling_forward(ctx, fb, h, g_vec, hidden, T, n_tf_layers, enc_out);

        // Post conv (hidden → half)
        std::vector<float> w_post, b_post;
        read_tensor_f32(fb.post_w, w_post);
        read_tensor_f32(fb.post_b, b_post);

        std::vector<float> m(half * T, 0.0f);
        for (int t = 0; t < T; t++) {
            for (int oc = 0; oc < half; oc++) {
                float sum = b_post[oc];
                for (int ic = 0; ic < hidden; ic++)
                    sum += enc_out[t * hidden + ic] * w_post[ic + oc * hidden];
                m[t * half + oc] = sum;
            }
        }

        // Inverse affine (mean_only): z1 = z1 - m
        for (int t = 0; t < T; t++)
            for (int c = 0; c < half; c++)
                z1[t * half + c] -= m[t * half + c];

        // Recombine
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < half; c++) {
                z[t * C + c] = z0[t * half + c];
                z[t * C + half + c] = z1[t * half + c];
            }
        }
    }
}

// ── HiFi-GAN Decoder ──────────────────────────────────────────────

static bool hifigan_decode(melotts_context* ctx, const std::vector<float>& z, const std::vector<float>& g_vec,
                           int T_latent, std::vector<float>& pcm_out) {
    const auto& hp = ctx->hp;
    const auto& w = ctx->w;
    int C_in = (int)hp.inter_channels;
    int gin = (int)hp.gin_channels;

    mini_graph mg(ctx->sched, 64 * 1024 * 1024);
    auto* gc = mg.ctx;

    // Input
    ggml_tensor* x_input = ggml_new_tensor_2d(gc, GGML_TYPE_F32, C_in, T_latent);
    ggml_set_name(x_input, "dec_input");
    ggml_set_input(x_input);

    // Speaker conditioning input
    ggml_tensor* g_input = nullptr;
    if (w.dec_cond_w && !g_vec.empty()) {
        g_input = ggml_new_tensor_1d(gc, GGML_TYPE_F32, gin);
        ggml_set_name(g_input, "dec_g");
        ggml_set_input(g_input);
    }

    // conv_pre
    ggml_tensor* x = conv1d_cf(gc, x_input, w.dec_conv_pre_w, w.dec_conv_pre_b, 1, 3, 1);

    // Speaker conditioning: x += cond(g)
    // dec.cond is Conv1d(gin, upsample_initial_ch, 1) — 1x1 conv on (gin,1)
    if (g_input && w.dec_cond_w) {
        // Reshape g to (gin, 1) for conv1d_cf
        ggml_tensor* g_2d = ggml_reshape_2d(gc, g_input, gin, 1);
        ggml_tensor* g_proj = conv1d_cf(gc, g_2d, w.dec_cond_w, w.dec_cond_b);
        // g_proj is (upsample_initial_channel, 1) — broadcast add over T
        x = ggml_add(gc, x, g_proj);
    }

    // Upsample stages
    int n_resblocks_per_stage = (int)hp.resblock_kernels.size();
    int rb_idx = 0;

    for (uint32_t us = 0; us < hp.n_upsample_stages; us++) {
        x = ggml_leaky_relu(gc, x, 0.1f, false);

        int stride = (int)hp.upsample_rates[us];
        int kernel = (int)hp.upsample_kernels[us];
        int crop_each = (kernel - stride) / 2;

        x = core_convt::convt1d_crop(gc, x, w.dec_ups[us].w, w.dec_ups[us].b, stride, crop_each, crop_each);

        // MRF: average of resblocks
        ggml_tensor* sum_rb = nullptr;

        for (int ri = 0; ri < n_resblocks_per_stage; ri++) {
            const auto& rb = w.dec_resblocks[rb_idx + ri];
            int rk = (int)hp.resblock_kernels[ri];

            // ResBlock1: 3 conv pairs with dilations (1,3,5)
            ggml_tensor* y = x;
            // Pair 0: dilation=1
            {
                int d = (int)hp.resblock_dilations[ri][0];
                int p = (rk * d - d) / 2;
                ggml_tensor* yt = ggml_leaky_relu(gc, y, 0.1f, false);
                yt = conv1d_cf(gc, yt, rb.conv1_0_w, rb.conv1_0_b, 1, p, d);
                yt = ggml_leaky_relu(gc, yt, 0.1f, false);
                yt = conv1d_cf(gc, yt, rb.conv2_0_w, rb.conv2_0_b, 1, (rk - 1) / 2, 1);
                y = ggml_add(gc, y, yt);
            }
            // Pair 1: dilation=3
            {
                int d = (int)hp.resblock_dilations[ri][1];
                int p = (rk * d - d) / 2;
                ggml_tensor* yt = ggml_leaky_relu(gc, y, 0.1f, false);
                yt = conv1d_cf(gc, yt, rb.conv1_1_w, rb.conv1_1_b, 1, p, d);
                yt = ggml_leaky_relu(gc, yt, 0.1f, false);
                yt = conv1d_cf(gc, yt, rb.conv2_1_w, rb.conv2_1_b, 1, (rk - 1) / 2, 1);
                y = ggml_add(gc, y, yt);
            }
            // Pair 2: dilation=5
            {
                int d = (int)hp.resblock_dilations[ri][2];
                int p = (rk * d - d) / 2;
                ggml_tensor* yt = ggml_leaky_relu(gc, y, 0.1f, false);
                yt = conv1d_cf(gc, yt, rb.conv1_2_w, rb.conv1_2_b, 1, p, d);
                yt = ggml_leaky_relu(gc, yt, 0.1f, false);
                yt = conv1d_cf(gc, yt, rb.conv2_2_w, rb.conv2_2_b, 1, (rk - 1) / 2, 1);
                y = ggml_add(gc, y, yt);
            }

            if (sum_rb == nullptr)
                sum_rb = y;
            else
                sum_rb = ggml_add(gc, sum_rb, y);
        }

        x = ggml_scale(gc, sum_rb, 1.0f / (float)n_resblocks_per_stage);
        rb_idx += n_resblocks_per_stage;
    }

    // Final: LeakyReLU → conv_post → tanh
    x = ggml_leaky_relu(gc, x, 0.1f, false);
    x = conv1d_cf(gc, x, w.dec_conv_post_w, nullptr, 1, 3, 1);
    x = ggml_tanh(gc, x);

    // Build and compute
    ggml_cgraph* gf = ggml_new_graph_custom(gc, 32768, false);
    ggml_build_forward_expand(gf, x);

    ggml_backend_sched_reset(mg.sched);
    if (!ggml_backend_sched_alloc_graph(mg.sched, gf)) {
        fprintf(stderr, "melotts: HiFi-GAN graph alloc failed\n");
        return false;
    }

    ggml_backend_tensor_set(x_input, z.data(), 0, z.size() * sizeof(float));
    if (g_input && !g_vec.empty())
        ggml_backend_tensor_set(g_input, g_vec.data(), 0, gin * sizeof(float));

    ggml_backend_sched_graph_compute(mg.sched, gf);

    int T_audio = (int)ggml_nelements(x);
    pcm_out.resize(T_audio);
    ggml_backend_tensor_get(x, pcm_out.data(), 0, T_audio * sizeof(float));
    return true;
}

// ── Weight loading ────────────────────────────────────────────────

static ggml_tensor* require_tensor(const std::map<std::string, ggml_tensor*>& tensors, const std::string& name) {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        fprintf(stderr, "melotts: missing tensor '%s'\n", name.c_str());
        return nullptr;
    }
    return it->second;
}

static ggml_tensor* try_tensor(const std::map<std::string, ggml_tensor*>& tensors, const std::string& name) {
    auto it = tensors.find(name);
    return (it != tensors.end()) ? it->second : nullptr;
}

static bool load_dds_conv(const std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix, int n_layers,
                          melotts_dds_conv& dds) {
    dds.layers.resize(n_layers);
    for (int i = 0; i < n_layers; i++) {
        auto& l = dds.layers[i];
        std::string si = std::to_string(i);
        l.conv_sep_w = require_tensor(tensors, prefix + ".convs_sep." + si + ".weight");
        l.conv_sep_b = require_tensor(tensors, prefix + ".convs_sep." + si + ".bias");
        l.conv_1x1_w = require_tensor(tensors, prefix + ".convs_1x1." + si + ".weight");
        l.conv_1x1_b = require_tensor(tensors, prefix + ".convs_1x1." + si + ".bias");
        l.norm1_g = require_tensor(tensors, prefix + ".norms_1." + si + ".gamma");
        l.norm1_b = require_tensor(tensors, prefix + ".norms_1." + si + ".beta");
        l.norm2_g = require_tensor(tensors, prefix + ".norms_2." + si + ".gamma");
        l.norm2_b = require_tensor(tensors, prefix + ".norms_2." + si + ".beta");
    }
    return true;
}

static void load_enc_layer(const std::map<std::string, ggml_tensor*>& tensors,
                           const std::string& prefix, // e.g. "enc_p.encoder"
                           int idx, melotts_enc_layer& layer) {
    std::string p = prefix + ".attn_layers." + std::to_string(idx);
    layer.conv_q_w = require_tensor(tensors, p + ".conv_q.weight");
    layer.conv_q_b = require_tensor(tensors, p + ".conv_q.bias");
    layer.conv_k_w = require_tensor(tensors, p + ".conv_k.weight");
    layer.conv_k_b = require_tensor(tensors, p + ".conv_k.bias");
    layer.conv_v_w = require_tensor(tensors, p + ".conv_v.weight");
    layer.conv_v_b = require_tensor(tensors, p + ".conv_v.bias");
    layer.conv_o_w = require_tensor(tensors, p + ".conv_o.weight");
    layer.conv_o_b = require_tensor(tensors, p + ".conv_o.bias");
    layer.emb_rel_k = require_tensor(tensors, p + ".emb_rel_k");
    layer.emb_rel_v = require_tensor(tensors, p + ".emb_rel_v");

    std::string np = prefix + ".norm_layers_1." + std::to_string(idx);
    layer.norm1_g = require_tensor(tensors, np + ".gamma");
    layer.norm1_b = require_tensor(tensors, np + ".beta");

    std::string fp = prefix + ".ffn_layers." + std::to_string(idx);
    layer.ffn_c1_w = require_tensor(tensors, fp + ".conv_1.weight");
    layer.ffn_c1_b = require_tensor(tensors, fp + ".conv_1.bias");
    layer.ffn_c2_w = require_tensor(tensors, fp + ".conv_2.weight");
    layer.ffn_c2_b = require_tensor(tensors, fp + ".conv_2.bias");

    std::string n2p = prefix + ".norm_layers_2." + std::to_string(idx);
    layer.norm2_g = require_tensor(tensors, n2p + ".gamma");
    layer.norm2_b = require_tensor(tensors, n2p + ".beta");
}

static bool load_weights(melotts_context* ctx, const char* path) {
    auto& hp = ctx->hp;
    auto& w = ctx->w;

    // Pass 1: metadata
    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta)
        return false;

    hp.hidden_channels = core_gguf::kv_u32(meta, "melotts.hidden_channels", hp.hidden_channels);
    hp.inter_channels = core_gguf::kv_u32(meta, "melotts.inter_channels", hp.inter_channels);
    hp.filter_channels = core_gguf::kv_u32(meta, "melotts.filter_channels", hp.filter_channels);
    hp.n_heads = core_gguf::kv_u32(meta, "melotts.n_heads", hp.n_heads);
    hp.head_dim = core_gguf::kv_u32(meta, "melotts.head_dim", hp.head_dim);
    hp.n_layers_enc = core_gguf::kv_u32(meta, "melotts.n_layers_enc", hp.n_layers_enc);
    hp.n_layers_trans_flow = core_gguf::kv_u32(meta, "melotts.n_layers_trans_flow", hp.n_layers_trans_flow);
    hp.n_flow_blocks = core_gguf::kv_u32(meta, "melotts.n_flow_blocks", hp.n_flow_blocks);
    hp.n_upsample_stages = core_gguf::kv_u32(meta, "melotts.n_upsample_stages", hp.n_upsample_stages);
    hp.upsample_initial_channel =
        core_gguf::kv_u32(meta, "melotts.upsample_initial_channel", hp.upsample_initial_channel);
    hp.num_symbols = core_gguf::kv_u32(meta, "melotts.num_symbols", hp.num_symbols);
    hp.num_tones = core_gguf::kv_u32(meta, "melotts.num_tones", hp.num_tones);
    hp.num_languages = core_gguf::kv_u32(meta, "melotts.num_languages", hp.num_languages);
    hp.n_speakers = core_gguf::kv_u32(meta, "melotts.n_speakers", hp.n_speakers);
    hp.gin_channels = core_gguf::kv_u32(meta, "melotts.gin_channels", hp.gin_channels);
    hp.sample_rate = core_gguf::kv_u32(meta, "melotts.sample_rate", hp.sample_rate);
    hp.n_sdp_flows = core_gguf::kv_u32(meta, "melotts.n_sdp_flows", hp.n_sdp_flows);
    hp.sdp_num_bins = core_gguf::kv_u32(meta, "melotts.sdp_num_bins", hp.sdp_num_bins);
    hp.n_sdp_dds_layers = core_gguf::kv_u32(meta, "melotts.n_sdp_dds_layers", hp.n_sdp_dds_layers);
    hp.noise_scale = core_gguf::kv_f32(meta, "melotts.noise_scale", hp.noise_scale);
    hp.length_scale = core_gguf::kv_f32(meta, "melotts.length_scale", hp.length_scale);
    hp.noise_w = core_gguf::kv_f32(meta, "melotts.noise_w", hp.noise_w);
    hp.sdp_ratio = core_gguf::kv_f32(meta, "melotts.sdp_ratio", hp.sdp_ratio);

    // Upsample/resblock params
    hp.upsample_rates.resize(hp.n_upsample_stages);
    hp.upsample_kernels.resize(hp.n_upsample_stages);
    for (uint32_t i = 0; i < hp.n_upsample_stages; i++) {
        char key[64];
        snprintf(key, sizeof(key), "melotts.upsample_rate.%u", i);
        hp.upsample_rates[i] = core_gguf::kv_u32(meta, key, 8);
        snprintf(key, sizeof(key), "melotts.upsample_kernel.%u", i);
        hp.upsample_kernels[i] = core_gguf::kv_u32(meta, key, 16);
    }

    hp.resblock_kernels.clear();
    hp.resblock_dilations.clear();
    for (int i = 0; i < 10; i++) {
        char key[64];
        snprintf(key, sizeof(key), "melotts.resblock_kernel.%d", i);
        uint32_t k = core_gguf::kv_u32(meta, key, 0);
        if (k == 0)
            break;
        hp.resblock_kernels.push_back(k);

        std::vector<uint32_t> dils;
        for (int j = 0; j < 10; j++) {
            snprintf(key, sizeof(key), "melotts.resblock_dilation.%d.%d", i, j);
            uint32_t d = core_gguf::kv_u32(meta, key, 0);
            if (d == 0)
                break;
            dils.push_back(d);
        }
        if (dils.empty())
            dils = {1, 3, 5};
        hp.resblock_dilations.push_back(dils);
    }
    if (hp.resblock_kernels.empty()) {
        hp.resblock_kernels = {3, 7, 11};
        hp.resblock_dilations = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    }

    // G2P data
    std::string symbols_json = core_gguf::kv_str(meta, "melotts.symbols_json", "[]");
    std::string cmudict_json = core_gguf::kv_str(meta, "melotts.cmudict_json", "{}");
    std::string g2p_en_json = core_gguf::kv_str(meta, "melotts.g2p_en_json", "");

    core_gguf::free_metadata(meta);

    // Parse symbols
    ctx->g2p.symbols = json_parse_string_array(symbols_json);
    for (size_t i = 0; i < ctx->g2p.symbols.size(); i++)
        ctx->g2p.symbol_to_id[ctx->g2p.symbols[i]] = (int)i;

    // English tone start = ZH(6) + JP(1) = 7
    ctx->g2p.tone_start_en = 7;

    // CMU dict parsing (simple JSON object of "WORD": [["syl1", ...], ...])
    // This is heavyweight; just store raw JSON and parse lazily or skip for now.
    // For the initial implementation, we parse on load.
    if (cmudict_json.size() > 2) {
        // Simple JSON parser for {"WORD": [["ph1", "ph2"], ["ph3"]], ...}
        // This is slow but only runs once.
        size_t p = cmudict_json.find('{');
        if (p != std::string::npos) {
            p++;
            while (p < cmudict_json.size()) {
                // Skip whitespace/commas
                while (p < cmudict_json.size() &&
                       (cmudict_json[p] == ' ' || cmudict_json[p] == '\n' || cmudict_json[p] == ',' ||
                        cmudict_json[p] == '\r' || cmudict_json[p] == '\t'))
                    p++;
                if (p >= cmudict_json.size() || cmudict_json[p] == '}')
                    break;

                // Parse key (with escape handling)
                if (cmudict_json[p] != '"')
                    break;
                p++;
                std::string word;
                while (p < cmudict_json.size() && cmudict_json[p] != '"') {
                    if (cmudict_json[p] == '\\' && p + 1 < cmudict_json.size()) {
                        p++;
                        if (cmudict_json[p] == '"')
                            word += '"';
                        else if (cmudict_json[p] == '\\')
                            word += '\\';
                        else
                            word += cmudict_json[p];
                    } else {
                        word += cmudict_json[p];
                    }
                    p++;
                }
                if (p < cmudict_json.size())
                    p++; // skip "

                // Skip :
                while (p < cmudict_json.size() && (cmudict_json[p] == ':' || cmudict_json[p] == ' '))
                    p++;

                // Parse array of arrays: [[...], [...], ...]
                if (p >= cmudict_json.size() || cmudict_json[p] != '[')
                    break;
                p++; // skip outer [

                std::vector<std::vector<std::string>> syllables;
                while (p < cmudict_json.size() && cmudict_json[p] != ']') {
                    while (p < cmudict_json.size() && (cmudict_json[p] == ' ' || cmudict_json[p] == ','))
                        p++;
                    if (p >= cmudict_json.size() || cmudict_json[p] == ']')
                        break;
                    if (cmudict_json[p] != '[')
                        break;
                    p++; // skip inner [

                    std::vector<std::string> syl;
                    while (p < cmudict_json.size() && cmudict_json[p] != ']') {
                        while (p < cmudict_json.size() && (cmudict_json[p] == ' ' || cmudict_json[p] == ','))
                            p++;
                        if (p >= cmudict_json.size() || cmudict_json[p] == ']')
                            break;
                        if (cmudict_json[p] == '"') {
                            p++;
                            std::string ph;
                            while (p < cmudict_json.size() && cmudict_json[p] != '"') {
                                if (cmudict_json[p] == '\\' && p + 1 < cmudict_json.size()) {
                                    p++;
                                    if (cmudict_json[p] == '"')
                                        ph += '"';
                                    else if (cmudict_json[p] == '\\')
                                        ph += '\\';
                                    else
                                        ph += cmudict_json[p];
                                } else {
                                    ph += cmudict_json[p];
                                }
                                p++;
                            }
                            if (p < cmudict_json.size())
                                p++;
                            syl.push_back(ph);
                        } else
                            break;
                    }
                    if (p < cmudict_json.size())
                        p++; // skip ]
                    syllables.push_back(syl);
                }
                if (p < cmudict_json.size())
                    p++; // skip outer ]

                ctx->g2p.cmudict[word] = syllables;
            }
        }
        if (ctx->verbosity >= 1)
            fprintf(stderr, "melotts: CMU dict: %zu entries\n", ctx->g2p.cmudict.size());
    }

    // Parse neural G2P weights (g2p_en model, base64 JSON)
    if (!g2p_en_json.empty()) {
        // Minimal JSON parse: extract meta.graphemes, meta.phonemes, weights.*
        // The JSON structure: {"meta":{"graphemes":[...],"phonemes":[...]},"weights":{"enc_emb":{"shape":[29,256],"data":"base64..."},...}}
        auto& nm = ctx->g2p.neural;

        // Extract graphemes and phonemes arrays
        auto extract_array = [&](const std::string& key) -> std::vector<std::string> {
            std::string pat = "\"" + key + "\"";
            size_t pos = g2p_en_json.find(pat);
            if (pos == std::string::npos)
                return {};
            pos = g2p_en_json.find('[', pos);
            if (pos == std::string::npos)
                return {};
            return json_parse_string_array(g2p_en_json.substr(pos, g2p_en_json.find(']', pos) - pos + 1));
        };
        nm.graphemes = extract_array("graphemes");
        nm.phonemes = extract_array("phonemes");
        for (size_t i = 0; i < nm.graphemes.size(); i++)
            nm.g2idx[nm.graphemes[i]] = (int)i;

        // Extract base64-encoded weight arrays
        auto extract_weight = [&](const std::string& key) -> std::vector<float> {
            std::string pat = "\"" + key + "\"";
            size_t pos = g2p_en_json.find(pat);
            if (pos == std::string::npos)
                return {};
            // Find "data":"base64..."
            size_t dpos = g2p_en_json.find("\"data\"", pos);
            if (dpos == std::string::npos)
                return {};
            size_t qstart = g2p_en_json.find('"', dpos + 6);
            if (qstart == std::string::npos)
                return {};
            qstart++;
            size_t qend = g2p_en_json.find('"', qstart);
            if (qend == std::string::npos)
                return {};
            std::string b64 = g2p_en_json.substr(qstart, qend - qstart);

            // Base64 decode
            static const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            std::vector<uint8_t> raw;
            int val = 0, bits = 0;
            for (char c : b64) {
                if (c == '=' || c == '\n' || c == '\r')
                    continue;
                size_t idx = chars.find(c);
                if (idx == std::string::npos)
                    continue;
                val = (val << 6) | (int)idx;
                bits += 6;
                if (bits >= 8) {
                    bits -= 8;
                    raw.push_back((uint8_t)(val >> bits));
                }
            }
            std::vector<float> out(raw.size() / sizeof(float));
            if (!raw.empty())
                memcpy(out.data(), raw.data(), out.size() * sizeof(float));
            return out;
        };

        nm.enc_emb = extract_weight("enc_emb");
        nm.dec_emb = extract_weight("dec_emb");
        nm.enc_w_ih = extract_weight("enc_w_ih");
        nm.enc_w_hh = extract_weight("enc_w_hh");
        nm.enc_b_ih = extract_weight("enc_b_ih");
        nm.enc_b_hh = extract_weight("enc_b_hh");
        nm.dec_w_ih = extract_weight("dec_w_ih");
        nm.dec_w_hh = extract_weight("dec_w_hh");
        nm.dec_b_ih = extract_weight("dec_b_ih");
        nm.dec_b_hh = extract_weight("dec_b_hh");
        nm.fc_w = extract_weight("fc_w");
        nm.fc_b = extract_weight("fc_b");

        nm.loaded = !nm.enc_emb.empty() && !nm.dec_emb.empty() && !nm.fc_w.empty();
        if (nm.loaded && ctx->verbosity >= 1)
            fprintf(stderr, "melotts: neural G2P loaded (%zu graphemes, %zu phonemes)\n", nm.graphemes.size(),
                    nm.phonemes.size());
    }

    // Pass 2: load weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "melotts", wl))
        return false;
    ctx->w_ctx = wl.ctx;
    ctx->w_buf = wl.buf;
    auto& tensors = wl.tensors;

    // ── Text encoder ──
    w.emb = require_tensor(tensors, "enc_p.emb.weight");
    w.tone_emb = require_tensor(tensors, "enc_p.tone_emb.weight");
    w.lang_emb = require_tensor(tensors, "enc_p.language_emb.weight");
    w.bert_proj_w = try_tensor(tensors, "enc_p.bert_proj.weight");
    w.bert_proj_b = try_tensor(tensors, "enc_p.bert_proj.bias");
    w.ja_bert_proj_w = try_tensor(tensors, "enc_p.ja_bert_proj.weight");
    w.ja_bert_proj_b = try_tensor(tensors, "enc_p.ja_bert_proj.bias");
    w.proj_w = require_tensor(tensors, "enc_p.proj.weight");
    w.proj_b = require_tensor(tensors, "enc_p.proj.bias");

    // Speaker-conditioned encoder
    w.spk_emb_linear_w = try_tensor(tensors, "enc_p.encoder.spk_emb_linear.weight");
    w.spk_emb_linear_b = try_tensor(tensors, "enc_p.encoder.spk_emb_linear.bias");

    w.enc_layers.resize(hp.n_layers_enc);
    for (uint32_t i = 0; i < hp.n_layers_enc; i++)
        load_enc_layer(tensors, "enc_p.encoder", (int)i, w.enc_layers[i]);

    // ── Speaker embedding ──
    w.emb_g = require_tensor(tensors, "emb_g.weight");

    // ── SDP ──
    w.sdp_pre_w = require_tensor(tensors, "sdp.pre.weight");
    w.sdp_pre_b = require_tensor(tensors, "sdp.pre.bias");
    w.sdp_proj_w = require_tensor(tensors, "sdp.proj.weight");
    w.sdp_proj_b = require_tensor(tensors, "sdp.proj.bias");
    w.sdp_cond_w = try_tensor(tensors, "sdp.cond.weight");
    w.sdp_cond_b = try_tensor(tensors, "sdp.cond.bias");
    w.sdp_flow0_m = require_tensor(tensors, "sdp.flows.0.m");
    w.sdp_flow0_logs = try_tensor(tensors, "sdp.flows.0.logs");

    load_dds_conv(tensors, "sdp.convs", hp.n_sdp_dds_layers, w.sdp_convs);

    w.sdp_flows.resize(hp.n_sdp_flows);
    int sdp_flow_idx[] = {3, 5, 7, 9, 11};
    for (uint32_t fi = 0; fi < hp.n_sdp_flows; fi++) {
        int idx = sdp_flow_idx[fi];
        auto& cf = w.sdp_flows[fi];
        std::string p = "sdp.flows." + std::to_string(idx);
        cf.pre_w = require_tensor(tensors, p + ".pre.weight");
        cf.pre_b = require_tensor(tensors, p + ".pre.bias");
        cf.proj_w = require_tensor(tensors, p + ".proj.weight");
        cf.proj_b = require_tensor(tensors, p + ".proj.bias");

        cf.dds.layers.resize(hp.n_sdp_dds_layers);
        for (uint32_t di = 0; di < hp.n_sdp_dds_layers; di++) {
            auto& dl = cf.dds.layers[di];
            std::string dp = p + ".convs";
            std::string si = std::to_string(di);
            dl.conv_sep_w = require_tensor(tensors, dp + ".convs_sep." + si + ".weight");
            dl.conv_sep_b = require_tensor(tensors, dp + ".convs_sep." + si + ".bias");
            dl.conv_1x1_w = require_tensor(tensors, dp + ".convs_1x1." + si + ".weight");
            dl.conv_1x1_b = require_tensor(tensors, dp + ".convs_1x1." + si + ".bias");
            dl.norm1_g = require_tensor(tensors, dp + ".norms_1." + si + ".gamma");
            dl.norm1_b = require_tensor(tensors, dp + ".norms_1." + si + ".beta");
            dl.norm2_g = require_tensor(tensors, dp + ".norms_2." + si + ".gamma");
            dl.norm2_b = require_tensor(tensors, dp + ".norms_2." + si + ".beta");
        }
    }

    // ── Duration Predictor ──
    w.dp_conv1_w = require_tensor(tensors, "dp.conv_1.weight");
    w.dp_conv1_b = require_tensor(tensors, "dp.conv_1.bias");
    w.dp_norm1_g = require_tensor(tensors, "dp.norm_1.gamma");
    w.dp_norm1_b = require_tensor(tensors, "dp.norm_1.beta");
    w.dp_conv2_w = require_tensor(tensors, "dp.conv_2.weight");
    w.dp_conv2_b = require_tensor(tensors, "dp.conv_2.bias");
    w.dp_norm2_g = require_tensor(tensors, "dp.norm_2.gamma");
    w.dp_norm2_b = require_tensor(tensors, "dp.norm_2.beta");
    w.dp_proj_w = require_tensor(tensors, "dp.proj.weight");
    w.dp_proj_b = require_tensor(tensors, "dp.proj.bias");
    w.dp_cond_w = try_tensor(tensors, "dp.cond.weight");
    w.dp_cond_b = try_tensor(tensors, "dp.cond.bias");

    // ── TransformerCoupling Flow ──
    w.flow_blocks.resize(hp.n_flow_blocks);
    int flow_idx[] = {0, 2, 4, 6, 8, 10};
    for (uint32_t fi = 0; fi < hp.n_flow_blocks; fi++) {
        int idx = flow_idx[fi];
        auto& fb = w.flow_blocks[fi];
        std::string p = "flow.flows." + std::to_string(idx);
        fb.pre_w = require_tensor(tensors, p + ".pre.weight");
        fb.pre_b = require_tensor(tensors, p + ".pre.bias");
        fb.post_w = require_tensor(tensors, p + ".post.weight");
        fb.post_b = require_tensor(tensors, p + ".post.bias");

        // Speaker conditioning in flow encoder
        fb.spk_emb_linear_w = try_tensor(tensors, p + ".enc.spk_emb_linear.weight");
        fb.spk_emb_linear_b = try_tensor(tensors, p + ".enc.spk_emb_linear.bias");

        // Encoder layers within this coupling block
        fb.enc_layers.resize(hp.n_layers_trans_flow);
        for (uint32_t li = 0; li < hp.n_layers_trans_flow; li++) {
            load_enc_layer(tensors, p + ".enc", (int)li, fb.enc_layers[li]);
        }
    }

    // ── HiFi-GAN Decoder ──
    w.dec_conv_pre_w = require_tensor(tensors, "dec.conv_pre.weight");
    w.dec_conv_pre_b = require_tensor(tensors, "dec.conv_pre.bias");
    w.dec_conv_post_w = require_tensor(tensors, "dec.conv_post.weight");
    w.dec_cond_w = try_tensor(tensors, "dec.cond.weight");
    w.dec_cond_b = try_tensor(tensors, "dec.cond.bias");

    w.dec_ups.resize(hp.n_upsample_stages);
    for (uint32_t i = 0; i < hp.n_upsample_stages; i++) {
        std::string p = "dec.ups." + std::to_string(i);
        w.dec_ups[i].w = require_tensor(tensors, p + ".weight");
        w.dec_ups[i].b = require_tensor(tensors, p + ".bias");
    }

    int n_resblocks = (int)hp.n_upsample_stages * (int)hp.resblock_kernels.size();
    w.dec_resblocks.resize(n_resblocks);
    for (int i = 0; i < n_resblocks; i++) {
        std::string p = "dec.resblocks." + std::to_string(i);
        auto& rb = w.dec_resblocks[i];
        rb.conv1_0_w = require_tensor(tensors, p + ".convs1.0.weight");
        rb.conv1_0_b = require_tensor(tensors, p + ".convs1.0.bias");
        rb.conv1_1_w = require_tensor(tensors, p + ".convs1.1.weight");
        rb.conv1_1_b = require_tensor(tensors, p + ".convs1.1.bias");
        rb.conv1_2_w = require_tensor(tensors, p + ".convs1.2.weight");
        rb.conv1_2_b = require_tensor(tensors, p + ".convs1.2.bias");
        rb.conv2_0_w = require_tensor(tensors, p + ".convs2.0.weight");
        rb.conv2_0_b = require_tensor(tensors, p + ".convs2.0.bias");
        rb.conv2_1_w = require_tensor(tensors, p + ".convs2.1.weight");
        rb.conv2_1_b = require_tensor(tensors, p + ".convs2.1.bias");
        rb.conv2_2_w = require_tensor(tensors, p + ".convs2.2.weight");
        rb.conv2_2_b = require_tensor(tensors, p + ".convs2.2.bias");
    }

    return true;
}

// ── Public API ────────────────────────────────────────────────────

struct melotts_params melotts_default_params(void) {
    struct melotts_params p;
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.noise_scale = -1.0f; // -1 = use model default
    p.length_scale = -1.0f;
    p.noise_w = -1.0f;
    p.sdp_ratio = -1.0f;
    p.speaker_id = 0;
    p.seed = 0;
    return p;
}

struct melotts_context* melotts_init_from_file(const char* path_model, struct melotts_params params) {
    auto* ctx = new melotts_context();
    ctx->verbosity = params.verbosity;
    ctx->n_threads = params.n_threads;

    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "melotts: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, params.n_threads);

    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ctx->backend_cpu;
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;

    if (!load_weights(ctx, path_model)) {
        melotts_free(ctx);
        return nullptr;
    }

    // Create scheduler
    {
        ggml_backend_t backends[2];
        int n_be = 0;
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 32768, false, false);
    }

    ctx->noise_scale = params.noise_scale >= 0 ? params.noise_scale : ctx->hp.noise_scale;
    ctx->length_scale = params.length_scale >= 0 ? params.length_scale : ctx->hp.length_scale;
    ctx->noise_w = params.noise_w >= 0 ? params.noise_w : ctx->hp.noise_w;
    ctx->sdp_ratio = params.sdp_ratio >= 0 ? params.sdp_ratio : ctx->hp.sdp_ratio;
    ctx->speaker_id = params.speaker_id;
    ctx->seed = params.seed;

    if (ctx->verbosity >= 1) {
        fprintf(stderr,
                "melotts: loaded %s (hidden=%u, enc=%u layers, "
                "flow=%u blocks (tf=%u layers), ups=%u, sr=%u, speakers=%u)\n",
                path_model, ctx->hp.hidden_channels, ctx->hp.n_layers_enc, ctx->hp.n_flow_blocks,
                ctx->hp.n_layers_trans_flow, ctx->hp.n_upsample_stages, ctx->hp.sample_rate, ctx->hp.n_speakers);
    }

    return ctx;
}

bool melotts_load_bert(struct melotts_context* ctx, const char* bert_gguf_path) {
    if (!ctx || !bert_gguf_path)
        return false;
    if (ctx->bert_ctx) {
        bert_encoder_free(ctx->bert_ctx);
        ctx->bert_ctx = nullptr;
    }
    ctx->bert_ctx = bert_encoder_init(bert_gguf_path, ctx->n_threads);
    if (!ctx->bert_ctx) {
        fprintf(stderr, "melotts: failed to load BERT model '%s'\n", bert_gguf_path);
        return false;
    }
    if (ctx->verbosity >= 1)
        fprintf(stderr, "melotts: BERT conditioning enabled (%s)\n", bert_gguf_path);
    return true;
}

void melotts_free(struct melotts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->bert_ctx)
        bert_encoder_free(ctx->bert_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->w_buf)
        ggml_backend_buffer_free(ctx->w_buf);
    if (ctx->w_ctx)
        ggml_free(ctx->w_ctx);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

int melotts_synthesize(struct melotts_context* ctx, const char* text, float** pcm_out, int* sample_rate_out) {
    if (!ctx || !text || !pcm_out)
        return 0;

    // 1. Text processing (G2P)
    std::vector<int> phone_ids, tone_ids, lang_ids;
    std::vector<int> word2ph;
    g2p_english(ctx->g2p, text, phone_ids, tone_ids, lang_ids, word2ph);
    int T = (int)phone_ids.size();

    if (ctx->verbosity >= 2) {
        fprintf(stderr, "melotts: %d phoneme IDs (after intersperse)\n", T);
    }

    // 2. Speaker embedding
    int gin = (int)ctx->hp.gin_channels;
    std::vector<float> g_vec(gin);
    {
        std::vector<float> g_table;
        int64_t n = ggml_nelements(ctx->w.emb_g);
        g_table.resize(n);
        size_t bytes = ggml_nbytes(ctx->w.emb_g);
        std::vector<uint8_t> raw(bytes);
        ggml_backend_tensor_get(ctx->w.emb_g, raw.data(), 0, bytes);
        if (ctx->w.emb_g->type == GGML_TYPE_F16) {
            const ggml_fp16_t* src = (const ggml_fp16_t*)raw.data();
            for (int64_t i = 0; i < n; i++)
                g_table[i] = ggml_fp16_to_fp32(src[i]);
        } else {
            memcpy(g_table.data(), raw.data(), n * sizeof(float));
        }
        for (int c = 0; c < gin; c++)
            g_vec[c] = g_table[ctx->speaker_id * gin + c];
    }
    dump_stage(ctx, "speaker_emb", g_vec.data(), g_vec.size());

    // 3. BERT conditioning (if loaded)
    std::vector<float> ja_bert_features;
    if (ctx->bert_ctx) {
        float* bert_out = nullptr;
        int n_bert_tokens = 0;
        if (bert_encoder_forward(ctx->bert_ctx, text, &bert_out, &n_bert_tokens)) {
            int bert_dim = bert_encoder_hidden_size(ctx->bert_ctx);

            // Build BERT-aligned word2ph: for each whitespace word,
            // get how many BERT subwords it has, then distribute the
            // word's phonemes across those subwords (matching Python's
            // distribute_phone from japanese.py).
            int* subtokens = nullptr;
            int n_words = bert_encoder_word_subtokens(ctx->bert_ctx, text, &subtokens);

            // Build per-BERT-token phone counts
            // word2ph has: [pad] [word0_phones] [word1_phones] ... [pad]
            // subtokens has: [n_sub_word0] [n_sub_word1] ...
            // BERT tokens: [CLS] [sub00 sub01..] [sub10..] ... [SEP]
            std::vector<int> bert_word2ph;
            bert_word2ph.push_back(word2ph[0]); // leading pad → [CLS]

            int w2p_idx = 1; // skip leading pad
            for (int wi = 0; wi < n_words && w2p_idx < (int)word2ph.size() - 1; wi++) {
                int n_phones_word = word2ph[w2p_idx];
                int n_sub = subtokens[wi];
                // distribute_phone: split n_phones evenly across n_sub tokens
                if (n_sub <= 1) {
                    bert_word2ph.push_back(n_phones_word);
                } else {
                    int base = n_phones_word / n_sub;
                    int remainder = n_phones_word % n_sub;
                    for (int s = 0; s < n_sub; s++) {
                        bert_word2ph.push_back(base + (s < remainder ? 1 : 0));
                    }
                }
                w2p_idx++;
            }

            // trailing pad → [SEP]
            if (w2p_idx < (int)word2ph.size())
                bert_word2ph.push_back(word2ph.back());
            free(subtokens);

            // Expand BERT features using bert_word2ph
            ja_bert_features.resize(bert_dim * T, 0.0f);
            int phone_pos = 0;
            for (int bi = 0; bi < (int)bert_word2ph.size() && phone_pos < T; bi++) {
                int n_phones = bert_word2ph[bi];
                int bt = std::clamp(bi, 0, n_bert_tokens - 1);
                for (int p = 0; p < n_phones && phone_pos < T; p++) {
                    for (int c = 0; c < bert_dim; c++)
                        ja_bert_features[c * T + phone_pos] = bert_out[bt * bert_dim + c];
                    phone_pos++;
                }
            }
            while (phone_pos < T) {
                int bt = n_bert_tokens - 1;
                for (int c = 0; c < bert_dim; c++)
                    ja_bert_features[c * T + phone_pos] = bert_out[bt * bert_dim + c];
                phone_pos++;
            }
            free(bert_out);
            if (ctx->verbosity >= 2)
                fprintf(stderr, "melotts: BERT %d tokens, %d word2ph entries → %d phones\n", n_bert_tokens,
                        (int)bert_word2ph.size(), T);
        }
    }

    // 4. Text encoder
    std::vector<float> enc_out, enc_mean, enc_logvar;
    text_encoder_forward(ctx, phone_ids, tone_ids, lang_ids, ja_bert_features, enc_out, enc_mean, enc_logvar);

    int C = (int)ctx->hp.inter_channels;
    if (enc_mean.empty()) {
        fprintf(stderr, "melotts: text encoder produced empty output\n");
        return 0;
    }

    // 4. Duration prediction (SDP + DP blend)
    std::vector<float> sdp_logw, dp_logw;
    sdp_forward(ctx, enc_out, g_vec, T, ctx->noise_w, sdp_logw);
    dp_forward(ctx, enc_out, g_vec, T, dp_logw);

    // Blend
    std::vector<float> logw(T);
    for (int t = 0; t < T; t++)
        logw[t] = sdp_logw[t] * ctx->sdp_ratio + dp_logw[t] * (1.0f - ctx->sdp_ratio);

    // Convert to integer durations
    std::vector<int> durations(T);
    int total_dur = 0;
    for (int t = 0; t < T; t++) {
        int d = (int)ceilf(expf(logw[t]) * ctx->length_scale);
        if (d < 1)
            d = 1;
        durations[t] = d;
        total_dur += d;
    }

    if (ctx->verbosity >= 2)
        fprintf(stderr, "melotts: total duration = %d frames\n", total_dur);

    // 5. Expand encoder outputs
    std::vector<float> z_mean, z_logvar;
    duration_align(enc_mean, durations, C, T, z_mean);
    duration_align(enc_logvar, durations, C, T, z_logvar);

    // 6. Sample latent
    int T_latent = total_dur;
    std::vector<float> z(C * T_latent);
    {
        uint32_t s = ctx->seed ? ctx->seed + 2 : (uint32_t)std::random_device()();
        std::mt19937 gen(s);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < C * T_latent; i++)
            z[i] = z_mean[i] + dist(gen) * ctx->noise_scale * expf(z_logvar[i]);
    }
    dump_stage(ctx, "z_p", z.data(), z.size());

    // 7. Flow inverse
    flow_inverse(ctx, z, g_vec, T_latent);
    dump_stage(ctx, "z_dec", z.data(), z.size());

    // 8. HiFi-GAN decode
    std::vector<float> pcm;
    if (!hifigan_decode(ctx, z, g_vec, T_latent, pcm))
        return 0;
    dump_stage(ctx, "audio", pcm.data(), pcm.size());

    // 9. Return
    int n_samples = (int)pcm.size();
    *pcm_out = (float*)malloc(n_samples * sizeof(float));
    memcpy(*pcm_out, pcm.data(), n_samples * sizeof(float));
    if (sample_rate_out)
        *sample_rate_out = (int)ctx->hp.sample_rate;

    if (ctx->verbosity >= 1) {
        float dur_sec = (float)n_samples / (float)ctx->hp.sample_rate;
        fprintf(stderr, "melotts: synthesized %.2f s (%d samples @ %u Hz)\n", dur_sec, n_samples, ctx->hp.sample_rate);
    }

    return n_samples;
}

void melotts_pcm_free(float* pcm) {
    free(pcm);
}

void melotts_set_noise_scale(struct melotts_context* ctx, float v) {
    if (ctx)
        ctx->noise_scale = v;
}
void melotts_set_length_scale(struct melotts_context* ctx, float v) {
    if (ctx)
        ctx->length_scale = v;
}
void melotts_set_noise_w(struct melotts_context* ctx, float v) {
    if (ctx)
        ctx->noise_w = v;
}
void melotts_set_sdp_ratio(struct melotts_context* ctx, float v) {
    if (ctx)
        ctx->sdp_ratio = v;
}
void melotts_set_speaker_id(struct melotts_context* ctx, int id) {
    if (ctx)
        ctx->speaker_id = id;
}
void melotts_set_seed(struct melotts_context* ctx, uint32_t seed) {
    if (ctx)
        ctx->seed = seed;
}
int melotts_sample_rate(const struct melotts_context* ctx) {
    return ctx ? (int)ctx->hp.sample_rate : 44100;
}
int melotts_num_speakers(const struct melotts_context* ctx) {
    return ctx ? (int)ctx->hp.n_speakers : 1;
}
void melotts_set_dump_dir(struct melotts_context* ctx, const char* dir) {
    if (ctx && dir)
        ctx->dump_dir = dir;
}
