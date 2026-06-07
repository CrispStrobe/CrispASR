// core/g2p_en.h — English grapheme-to-phoneme (text → IPA).
//
// Three-tier pipeline, all permissively licensed (MIT):
//   1. CMUdict lookup (~134K words, ARPAbet → IPA)
//   2. Neural G2P fallback (GRU seq2seq, ARPAbet → IPA, for OOV words)
//   3. Rule-based LTS (letter-to-sound digraph/trigraph rules, for fallback)
//
// The CMUdict and neural G2P weights can be loaded from a GGUF model
// (reusing MeloTTS's embedded data) or from standalone files.
//
// Usage:
//   g2p_en::context ctx;
//   g2p_en::load_cmudict_json(ctx, json_str);   // or load from GGUF
//   g2p_en::load_neural_g2p_json(ctx, json_str); // optional
//   std::string ipa = g2p_en::text_to_ipa(ctx, "Hello world");
//   // ipa = "hʌlˈoʊ wˈɜːld" (approximate)

#pragma once

#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace g2p_en {

// ── ARPAbet → IPA conversion table ──────────────────────────────────
// Standard 39-phoneme ARPAbet set (CMU Pronouncing Dictionary).
// Stress markers (0/1/2) stripped before lookup; stress is re-applied
// as IPA ˈ (primary) or ˌ (secondary) before the syllable.

inline const std::map<std::string, std::string>& arpabet_to_ipa() {
    static const std::map<std::string, std::string> table = {
        // Vowels
        {"AA", "ɑː"},  {"AE", "æ"},   {"AH", "ʌ"},   {"AO", "ɔː"},
        {"AW", "aʊ"},  {"AX", "ə"},   {"AY", "aɪ"},  {"EH", "ɛ"},
        {"ER", "ɜːɹ"}, {"EY", "eɪ"},  {"IH", "ɪ"},   {"IX", "ɨ"},
        {"IY", "iː"},  {"OW", "oʊ"},  {"OY", "ɔɪ"},  {"UH", "ʊ"},
        {"UW", "uː"},  {"UX", "ʉ"},
        // Consonants
        {"B",  "b"},    {"CH", "tʃ"},  {"D",  "d"},   {"DH", "ð"},
        {"DX", "ɾ"},    {"EL", "l̩"},   {"EM", "m̩"},   {"EN", "n̩"},
        {"F",  "f"},    {"G",  "ɡ"},   {"HH", "h"},   {"JH", "dʒ"},
        {"K",  "k"},    {"L",  "l"},   {"M",  "m"},   {"N",  "n"},
        {"NG", "ŋ"},    {"NX", "ɾ̃"},   {"P",  "p"},   {"Q",  "ʔ"},
        {"R",  "ɹ"},    {"S",  "s"},   {"SH", "ʃ"},   {"T",  "t"},
        {"TH", "θ"},    {"V",  "v"},   {"W",  "w"},   {"WH", "ʍ"},
        {"Y",  "j"},    {"Z",  "z"},   {"ZH", "ʒ"},
    };
    return table;
}

// Convert an ARPAbet phoneme (e.g. "AH0", "EY1") to IPA.
inline std::string arpa_to_ipa(const std::string& arpa) {
    // Strip stress digit
    std::string base = arpa;
    int stress = 0;
    if (!base.empty() && base.back() >= '0' && base.back() <= '2') {
        stress = base.back() - '0';
        base.pop_back();
    }
    // Uppercase for lookup
    for (auto& c : base) c = (char)toupper((unsigned char)c);
    auto& table = arpabet_to_ipa();
    auto it = table.find(base);
    if (it == table.end()) return "";
    std::string ipa;
    if (stress == 1) ipa = "ˈ";
    else if (stress == 2) ipa = "ˌ";
    ipa += it->second;
    return ipa;
}

// ── GRU cell (shared by neural G2P) ─────────────────────────────────

inline void gru_cell(const float* x, const float* h_prev, int input_dim,
                     int hidden_dim, const float* w_ih, const float* w_hh,
                     const float* b_ih, const float* b_hh, float* h_out) {
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
    auto sigmoid = [](float x) { return 1.0f / (1.0f + expf(-x)); };
    for (int i = 0; i < hidden_dim; i++) {
        float r = sigmoid(g_ih[i] + g_hh[i]);
        float z = sigmoid(g_ih[hidden_dim + i] + g_hh[hidden_dim + i]);
        float n = tanhf(g_ih[2 * hidden_dim + i] + r * g_hh[2 * hidden_dim + i]);
        h_out[i] = (1.0f - z) * n + z * h_prev[i];
    }
}

// ── Neural G2P model ────────────────────────────────────────────────

struct neural_model {
    bool loaded = false;
    int hidden_dim = 256;
    std::vector<std::string> graphemes; // 29: <pad> <unk> </s> a-z
    std::vector<std::string> phonemes;  // 74: <pad> <unk> <s> </s> AA0..ZH
    std::map<std::string, int> g2idx;
    std::vector<float> enc_emb, dec_emb;
    std::vector<float> enc_w_ih, enc_w_hh, enc_b_ih, enc_b_hh;
    std::vector<float> dec_w_ih, dec_w_hh, dec_b_ih, dec_b_hh;
    std::vector<float> fc_w, fc_b;
};

// Predict ARPAbet phonemes for a single word.
inline std::vector<std::string> neural_predict(const neural_model& m, const std::string& word) {
    if (!m.loaded) return {};
    int D = m.hidden_dim;
    std::string lower;
    for (char c : word) lower += (char)tolower((unsigned char)c);
    std::vector<int> char_ids;
    for (char c : lower) {
        std::string cs(1, c);
        auto it = m.g2idx.find(cs);
        char_ids.push_back(it != m.g2idx.end() ? it->second : 1);
    }
    char_ids.push_back(2); // </s>
    std::vector<float> h(D, 0.0f);
    for (int cid : char_ids) {
        std::vector<float> h_new(D);
        gru_cell(&m.enc_emb[cid * D], h.data(), D, D,
                 m.enc_w_ih.data(), m.enc_w_hh.data(),
                 m.enc_b_ih.data(), m.enc_b_hh.data(), h_new.data());
        h = h_new;
    }
    std::vector<std::string> preds;
    int dec_id = 2;
    for (int step = 0; step < 20; step++) {
        std::vector<float> h_new(D);
        gru_cell(&m.dec_emb[dec_id * D], h.data(), D, D,
                 m.dec_w_ih.data(), m.dec_w_hh.data(),
                 m.dec_b_ih.data(), m.dec_b_hh.data(), h_new.data());
        h = h_new;
        int n_ph = (int)m.phonemes.size();
        float best_val = -1e30f;
        int best_id = 0;
        for (int p = 0; p < n_ph; p++) {
            float s = m.fc_b[p];
            for (int d = 0; d < D; d++) s += h[d] * m.fc_w[p * D + d];
            if (s > best_val) { best_val = s; best_id = p; }
        }
        if (best_id == 3) break;
        if (best_id >= 4 && best_id < n_ph)
            preds.push_back(m.phonemes[best_id]);
        dec_id = best_id;
    }
    return preds;
}

// ── Neural G2P weight loading ────────────────────────────────────────

// Decode base64 to raw bytes.
inline std::vector<uint8_t> base64_decode(const std::string& b64) {
    static const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<uint8_t> raw;
    int val = 0, bits = 0;
    for (char c : b64) {
        if (c == '=' || c == '\n' || c == '\r') continue;
        size_t idx = chars.find(c);
        if (idx == std::string::npos) continue;
        val = (val << 6) | (int)idx;
        bits += 6;
        if (bits >= 8) { bits -= 8; raw.push_back((uint8_t)(val >> bits)); }
    }
    return raw;
}

// Load neural G2P weights from MeloTTS-format JSON.
// Format: {"meta":{"graphemes":[...],"phonemes":[...]},
//          "weights":{"enc_emb":{"shape":[29,256],"data":"base64..."},...}}
// Returns true if loaded successfully.
inline bool load_neural_g2p_json(neural_model& nm, const std::string& json) {
    if (json.empty()) return false;

    // Simple JSON string array extractor
    auto extract_array = [&](const std::string& key) -> std::vector<std::string> {
        std::string pat = "\"" + key + "\"";
        size_t pos = json.find(pat);
        if (pos == std::string::npos) return {};
        pos = json.find('[', pos);
        if (pos == std::string::npos) return {};
        size_t end = json.find(']', pos);
        if (end == std::string::npos) return {};
        std::vector<std::string> out;
        size_t p = pos + 1;
        while (p < end) {
            while (p < end && json[p] != '"') p++;
            if (p >= end) break;
            p++; // skip opening "
            std::string s;
            while (p < end && json[p] != '"') { s += json[p]; p++; }
            p++; // skip closing "
            out.push_back(s);
        }
        return out;
    };

    // Weight extractor: find "key":{"shape":...,"data":"base64..."}
    auto extract_weight = [&](const std::string& key) -> std::vector<float> {
        std::string pat = "\"" + key + "\"";
        size_t pos = json.find(pat);
        if (pos == std::string::npos) return {};
        size_t dpos = json.find("\"data\"", pos);
        if (dpos == std::string::npos) return {};
        size_t qstart = json.find('"', dpos + 6);
        if (qstart == std::string::npos) return {};
        qstart++;
        size_t qend = json.find('"', qstart);
        if (qend == std::string::npos) return {};
        auto raw = base64_decode(json.substr(qstart, qend - qstart));
        std::vector<float> out(raw.size() / sizeof(float));
        if (!raw.empty()) memcpy(out.data(), raw.data(), out.size() * sizeof(float));
        return out;
    };

    nm.graphemes = extract_array("graphemes");
    nm.phonemes = extract_array("phonemes");
    for (size_t i = 0; i < nm.graphemes.size(); i++)
        nm.g2idx[nm.graphemes[i]] = (int)i;

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
    return nm.loaded;
}

// Load neural G2P weights from a standalone JSON file (same format).
inline bool load_neural_g2p_file(neural_model& nm, const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string json(sz, '\0');
    size_t rd = fread(&json[0], 1, sz, f);
    fclose(f);
    json.resize(rd);
    return load_neural_g2p_json(nm, json);
}

// ── LTS rules (letter-to-sound for OOV) ─────────────────────────────
// Returns ARPAbet phonemes (lowercase) for an unknown word.

inline std::vector<std::string> lts_predict(const std::string& word) {
    std::vector<std::string> out;
    int len = (int)word.size();
    bool first_vowel = true;

    auto emit = [&](const char* ph, int stress) {
        std::string s = ph;
        if (stress > 0) s += (char)('0' + stress);
        out.push_back(s);
    };

    for (int i = 0; i < len;) {
        char c = (char)tolower((unsigned char)word[i]);
        char c1 = (i + 1 < len) ? (char)tolower((unsigned char)word[i + 1]) : 0;
        char c2 = (i + 2 < len) ? (char)tolower((unsigned char)word[i + 2]) : 0;

        // Trigraphs
        if (c == 't' && c1 == 'c' && c2 == 'h') { emit("CH", 0); i += 3; continue; }
        if (c == 'i' && c1 == 'g' && c2 == 'h') { emit("AY", first_vowel ? 1 : 0); first_vowel = false; i += 3; continue; }
        if (c == 't' && c1 == 'i' && c2 == 'o') { emit("SH", 0); emit("AH", 0); i += 3; continue; }

        // Digraphs (consonant)
        if (c == 't' && c1 == 'h') { emit("TH", 0); i += 2; continue; }
        if (c == 's' && c1 == 'h') { emit("SH", 0); i += 2; continue; }
        if (c == 'c' && c1 == 'h') { emit("CH", 0); i += 2; continue; }
        if (c == 'p' && c1 == 'h') { emit("F", 0); i += 2; continue; }
        if (c == 'w' && c1 == 'h') { emit("W", 0); i += 2; continue; }
        if (c == 'n' && c1 == 'g') { emit("NG", 0); i += 2; continue; }
        if (c == 'c' && c1 == 'k') { emit("K", 0); i += 2; continue; }
        if (c == 'g' && c1 == 'h') { i += 2; continue; }
        if (c == 'k' && c1 == 'n') { emit("N", 0); i += 2; continue; }
        if (c == 'w' && c1 == 'r') { emit("R", 0); i += 2; continue; }

        // Digraphs (vowel)
        if (c == 'e' && c1 == 'a') { emit("IY", first_vowel ? 1 : 0); first_vowel = false; i += 2; continue; }
        if (c == 'e' && c1 == 'e') { emit("IY", first_vowel ? 1 : 0); first_vowel = false; i += 2; continue; }
        if (c == 'o' && c1 == 'o') { emit("UW", first_vowel ? 1 : 0); first_vowel = false; i += 2; continue; }
        if (c == 'o' && c1 == 'u') { emit("AW", first_vowel ? 1 : 0); first_vowel = false; i += 2; continue; }
        if (c == 'o' && c1 == 'w') { emit("OW", first_vowel ? 1 : 0); first_vowel = false; i += 2; continue; }
        if (c == 'a' && c1 == 'i') { emit("EY", first_vowel ? 1 : 0); first_vowel = false; i += 2; continue; }
        if (c == 'a' && c1 == 'y') { emit("EY", first_vowel ? 1 : 0); first_vowel = false; i += 2; continue; }
        if (c == 'o' && c1 == 'i') { emit("OY", first_vowel ? 1 : 0); first_vowel = false; i += 2; continue; }
        if (c == 'o' && c1 == 'y') { emit("OY", first_vowel ? 1 : 0); first_vowel = false; i += 2; continue; }
        if (c == 'a' && c1 == 'w') { emit("AO", first_vowel ? 1 : 0); first_vowel = false; i += 2; continue; }
        if (c == 'e' && c1 == 'w') { emit("UW", first_vowel ? 1 : 0); first_vowel = false; i += 2; continue; }
        if (c == 'e' && c1 == 'r') { emit("ER", first_vowel ? 1 : 0); first_vowel = false; i += 2; continue; }

        // Silent final e
        if (c == 'e' && i == len - 1 && i > 0) { i++; continue; }

        // Single consonants
        if (c == 'b') { emit("B", 0); i++; continue; }
        if (c == 'd') { emit("D", 0); i++; continue; }
        if (c == 'f') { emit("F", 0); i++; continue; }
        if (c == 'g') { emit("G", 0); i++; continue; }
        if (c == 'h') { emit("HH", 0); i++; continue; }
        if (c == 'j') { emit("JH", 0); i++; continue; }
        if (c == 'k') { emit("K", 0); i++; continue; }
        if (c == 'l') { emit("L", 0); i++; continue; }
        if (c == 'm') { emit("M", 0); i++; continue; }
        if (c == 'n') { emit("N", 0); i++; continue; }
        if (c == 'p') { emit("P", 0); i++; continue; }
        if (c == 'q') { emit("K", 0); i++; continue; }
        if (c == 'r') { emit("R", 0); i++; continue; }
        if (c == 's') { emit("S", 0); i++; continue; }
        if (c == 't') { emit("T", 0); i++; continue; }
        if (c == 'v') { emit("V", 0); i++; continue; }
        if (c == 'w') { emit("W", 0); i++; continue; }
        if (c == 'x') { emit("K", 0); emit("S", 0); i++; continue; }
        if (c == 'y') { emit("Y", 0); i++; continue; }
        if (c == 'z') { emit("Z", 0); i++; continue; }

        // Single vowels
        if (c == 'a') { emit("AE", first_vowel ? 1 : 0); first_vowel = false; i++; continue; }
        if (c == 'e') { emit("EH", first_vowel ? 1 : 0); first_vowel = false; i++; continue; }
        if (c == 'i') { emit("IH", first_vowel ? 1 : 0); first_vowel = false; i++; continue; }
        if (c == 'o') { emit("AA", first_vowel ? 1 : 0); first_vowel = false; i++; continue; }
        if (c == 'u') { emit("AH", first_vowel ? 1 : 0); first_vowel = false; i++; continue; }
        if (c == 'c') {
            // c before e/i/y = /s/, otherwise /k/
            if (c1 == 'e' || c1 == 'i' || c1 == 'y') emit("S", 0);
            else emit("K", 0);
            i++; continue;
        }
        i++; // skip unknown
    }
    return out;
}

// ── CMUdict ─────────────────────────────────────────────────────────

struct cmudict {
    // word (UPPERCASE) → list of ARPAbet phonemes with stress (e.g. "HH AH0 L OW1")
    std::map<std::string, std::vector<std::string>> entries;
    bool loaded = false;
};

// Load CMUdict from a file in CMU format: "WORD PH1 PH2 PH3\n"
// Lines starting with ;;; are comments. Variant pronunciations
// (e.g. "HELLO(2)") are skipped (first pronunciation kept).
// Returns number of entries loaded.
inline int load_cmudict_file(cmudict& dict, const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return 0;
    char line[512];
    int count = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == ';' || line[0] == '\n' || line[0] == '\r') continue;
        // Strip newline
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
        if (len == 0) continue;
        // Split: first token is word, rest are phonemes
        char* p = line;
        // Extract word (up to first space or '(')
        std::string word;
        while (*p && *p != ' ' && *p != '\t' && *p != '(') {
            word += (char)toupper((unsigned char)*p);
            p++;
        }
        // Skip variant markers like (2), (3)
        if (*p == '(') {
            // Only keep first pronunciation
            if (dict.entries.count(word)) continue;
            while (*p && *p != ')') p++;
            if (*p == ')') p++;
        }
        // Skip whitespace
        while (*p == ' ' || *p == '\t') p++;
        // Parse phonemes
        std::vector<std::string> phones;
        while (*p) {
            while (*p == ' ' || *p == '\t') p++;
            if (!*p) break;
            std::string ph;
            while (*p && *p != ' ' && *p != '\t') ph += *p++;
            if (!ph.empty()) phones.push_back(ph);
        }
        if (!phones.empty() && !word.empty()) {
            dict.entries[word] = phones;
            count++;
        }
    }
    fclose(f);
    dict.loaded = count > 0;
    return count;
}

// ── Context ─────────────────────────────────────────────────────────

struct context {
    cmudict dict;
    neural_model neural;
};

// ── Tokenizer ───────────────────────────────────────────────────────

inline std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cur;
    for (char c : text) {
        if (c == ' ' || c == ',' || c == '.' || c == '!' || c == '?' ||
            c == ';' || c == ':' || c == '-' || c == '\n') {
            if (!cur.empty()) { tokens.push_back(cur); cur.clear(); }
            if (c != ' ') tokens.push_back(std::string(1, c));
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

// ── Main API: text → IPA ────────────────────────────────────────────

// Convert a single word to IPA using the three-tier pipeline.
inline std::string word_to_ipa(const context& ctx, const std::string& word) {
    // Uppercase for CMUdict lookup
    std::string upper;
    for (char c : word) upper += (char)toupper((unsigned char)c);

    std::vector<std::string> arpa_phones;

    // Tier 1: CMUdict
    if (ctx.dict.loaded) {
        auto it = ctx.dict.entries.find(upper);
        if (it != ctx.dict.entries.end()) {
            arpa_phones = it->second;
        }
    }

    // Tier 2: Neural G2P
    if (arpa_phones.empty() && ctx.neural.loaded) {
        arpa_phones = neural_predict(ctx.neural, word);
    }

    // Tier 3: LTS rules
    if (arpa_phones.empty()) {
        arpa_phones = lts_predict(word);
    }

    // Convert ARPAbet → IPA
    std::string ipa;
    for (const auto& ph : arpa_phones) {
        std::string p = arpa_to_ipa(ph);
        if (!p.empty()) ipa += p;
    }
    return ipa;
}

// Convert full text to IPA string.
inline std::string text_to_ipa(const context& ctx, const std::string& text) {
    auto words = tokenize(text);
    std::string ipa;
    for (const auto& w : words) {
        if (w.size() == 1 && (w[0] == ',' || w[0] == '.' || w[0] == '!' ||
            w[0] == '?' || w[0] == ';' || w[0] == ':' || w[0] == '-')) {
            // Keep punctuation as-is (piper's phoneme map includes them)
            if (!ipa.empty() && ipa.back() != ' ') ipa += ' ';
            continue;
        }
        if (!ipa.empty()) ipa += ' ';
        ipa += word_to_ipa(ctx, w);
    }
    return ipa;
}

} // namespace g2p_en
