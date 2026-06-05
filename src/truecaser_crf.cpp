// truecaser_crf.cpp — CRF-based truecaser with context features.
//
// Binary model format (little-endian):
//   "CRF1" magic (4 bytes)
//   uint16 n_labels
//   for each label: uint16 len + UTF-8 bytes
//   float32[n_labels * n_labels]  transition matrix (row-major)
//   uint32 n_features
//   for each feature:
//     uint16 key_len + UTF-8 key
//     float32[n_labels]  weights per label
//
// Labels: "lc" (lowercase), "u1" (capitalise first), "uc" (all upper)
// Features: word identity, suffix, prev/next word, article context, etc.
// Decode: Viterbi over the linear-chain CRF.

#include "truecaser_crf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

// German articles for the "after_article" feature
static const char* k_articles[] = {"der",    "die",    "das",    "ein",    "eine", "einem", "einen",
                                   "einer",  "eines",  "dem",    "den",    "des",  "kein",  "keine",
                                   "keinem", "keinen", "keiner", "keines", nullptr};

static bool is_article(const std::string& w) {
    for (const char** p = k_articles; *p; ++p)
        if (w == *p)
            return true;
    return false;
}

struct truecaser_crf_context {
    int n_labels = 0;
    std::vector<std::string> labels;
    std::vector<float> transitions; // [n_labels * n_labels]

    // Feature weights: feature_string → float[n_labels]
    std::unordered_map<std::string, std::vector<float>> feat_weights;
};

// ---------------------------------------------------------------------------
// Feature extraction (must match train-truecaser-crf.py exactly)
// ---------------------------------------------------------------------------

// Feature names MUST match the training script exactly.
static std::vector<std::string> extract_features(const std::vector<std::string>& words, int i, bool is_first) {
    const std::string& w = words[i];
    std::vector<std::string> feats;

    feats.push_back("w=" + w);
    if (w.size() >= 3)
        feats.push_back("s=" + w.substr(w.size() - 3));

    // German noun suffixes
    static const char* suffixes[] = {"ung", "heit", "keit", "schaft", "tion", "ment", "nis", nullptr};
    for (const char** s = suffixes; *s; ++s) {
        size_t slen = strlen(*s);
        if (w.size() >= slen && w.substr(w.size() - slen) == *s)
            feats.push_back(std::string("ns=") + *s);
    }

    // Word shape
    if (std::any_of(w.begin(), w.end(), [](char c) { return c >= '0' && c <= '9'; }))
        feats.push_back("has_digit");
    if (w.find('-') != std::string::npos)
        feats.push_back("has_hyphen");

    // Context
    if (i > 0) {
        feats.push_back("p=" + words[i - 1]);
        if (is_article(words[i - 1]))
            feats.push_back("art");
    } else {
        feats.push_back("BOS");
    }

    if (i < (int)words.size() - 1) {
        feats.push_back("n=" + words[i + 1]);
    } else {
        feats.push_back("EOS");
    }

    if (is_first)
        feats.push_back("S");

    return feats;
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

truecaser_crf_context* truecaser_crf_init(const char* model_path) {
    FILE* f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "truecaser_crf: cannot open '%s'\n", model_path);
        return nullptr;
    }

    // Magic
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "CRF1", 4) != 0) {
        fprintf(stderr, "truecaser_crf: bad magic in '%s'\n", model_path);
        fclose(f);
        return nullptr;
    }

    auto* ctx = new truecaser_crf_context();

    // Labels
    uint16_t n_labels = 0;
    fread(&n_labels, 2, 1, f);
    ctx->n_labels = n_labels;
    ctx->labels.resize(n_labels);
    for (int i = 0; i < n_labels; i++) {
        uint16_t len = 0;
        fread(&len, 2, 1, f);
        ctx->labels[i].resize(len);
        fread(&ctx->labels[i][0], 1, len, f);
    }

    // Transition matrix
    ctx->transitions.resize(n_labels * n_labels);
    fread(ctx->transitions.data(), sizeof(float), n_labels * n_labels, f);

    // Features
    uint32_t n_features = 0;
    fread(&n_features, 4, 1, f);
    ctx->feat_weights.reserve(n_features);
    for (uint32_t i = 0; i < n_features; i++) {
        uint16_t klen = 0;
        if (fread(&klen, 2, 1, f) != 1)
            break;
        std::string key(klen, '\0');
        if (fread(&key[0], 1, klen, f) != klen)
            break;
        std::vector<float> weights(n_labels);
        fread(weights.data(), sizeof(float), n_labels, f);
        ctx->feat_weights[key] = std::move(weights);
    }

    fclose(f);
    fprintf(stderr, "truecaser_crf: loaded %d labels, %zu features from '%s'\n", ctx->n_labels,
            ctx->feat_weights.size(), model_path);
    return ctx;
}

// ---------------------------------------------------------------------------
// Viterbi decode
// ---------------------------------------------------------------------------

static std::vector<int> viterbi(const truecaser_crf_context& ctx,
                                const std::vector<std::vector<std::string>>& feats_seq) {
    const int T = (int)feats_seq.size();
    const int L = ctx.n_labels;
    if (T == 0)
        return {};

    // score[t][l] = log-score of best path ending at label l at position t
    std::vector<std::vector<float>> score(T, std::vector<float>(L, 0.0f));
    std::vector<std::vector<int>> back(T, std::vector<int>(L, 0));

    // Compute emission scores for each position
    auto emission = [&](int t, int l) -> float {
        float s = 0.0f;
        for (const auto& feat : feats_seq[t]) {
            auto it = ctx.feat_weights.find(feat);
            if (it != ctx.feat_weights.end())
                s += it->second[l];
        }
        return s;
    };

    // Init: t=0
    for (int l = 0; l < L; l++)
        score[0][l] = emission(0, l);

    // Forward pass
    for (int t = 1; t < T; t++) {
        for (int l = 0; l < L; l++) {
            float em = emission(t, l);
            float best = -1e30f;
            int best_prev = 0;
            for (int p = 0; p < L; p++) {
                float s = score[t - 1][p] + ctx.transitions[p * L + l] + em;
                if (s > best) {
                    best = s;
                    best_prev = p;
                }
            }
            score[t][l] = best;
            back[t][l] = best_prev;
        }
    }

    // Backtrace
    std::vector<int> path(T);
    path[T - 1] = 0;
    float best = score[T - 1][0];
    for (int l = 1; l < L; l++) {
        if (score[T - 1][l] > best) {
            best = score[T - 1][l];
            path[T - 1] = l;
        }
    }
    for (int t = T - 2; t >= 0; t--)
        path[t] = back[t + 1][path[t + 1]];

    return path;
}

// ---------------------------------------------------------------------------
// ASCII case helpers
// ---------------------------------------------------------------------------

static std::string lowercase(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        out += (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
    return out;
}

static std::string capitalise(const std::string& s) {
    std::string out = s;
    for (size_t i = 0; i < out.size(); i++) {
        unsigned char c = (unsigned char)out[i];
        if (c >= 'a' && c <= 'z') {
            out[i] = (char)(c - 'a' + 'A');
            break;
        }
        if (c >= 'A' && c <= 'Z')
            break;
        if (c >= 0x80) {
            // skip multibyte
            if (c >= 0xF0)
                i += 3;
            else if (c >= 0xE0)
                i += 2;
            else
                i += 1;
        }
    }
    return out;
}

static std::string to_upper(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        out += (c >= 'a' && c <= 'z') ? (char)(c - 'a' + 'A') : c;
    return out;
}

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------

char* truecaser_crf_process(truecaser_crf_context* ctx, const char* text) {
    if (!ctx || !text)
        return nullptr;

    std::string input(text);

    // Split into words, preserving trailing punctuation
    struct Token {
        std::string word;  // the alphabetic part (lowercased for features)
        std::string trail; // trailing punctuation
        std::string orig;  // original word (for unknown-word passthrough)
    };
    std::vector<Token> tokens;
    {
        size_t i = 0;
        while (i < input.size()) {
            while (i < input.size() && (input[i] == ' ' || input[i] == '\t' || input[i] == '\n'))
                i++;
            if (i >= input.size())
                break;
            size_t start = i;
            while (i < input.size() && input[i] != ' ' && input[i] != '\t' && input[i] != '\n')
                i++;
            std::string tok = input.substr(start, i - start);

            // Split trailing punctuation
            size_t alpha_end = tok.size();
            while (alpha_end > 0) {
                char c = tok[alpha_end - 1];
                if (c == '.' || c == ',' || c == '?' || c == '!' || c == ':' || c == ';' || c == '-')
                    alpha_end--;
                else
                    break;
            }
            tokens.push_back({lowercase(tok.substr(0, alpha_end)), tok.substr(alpha_end), tok.substr(0, alpha_end)});
        }
    }

    if (tokens.empty()) {
        char* out = (char*)malloc(strlen(text) + 1);
        strcpy(out, text);
        return out;
    }

    // Extract features
    std::vector<std::string> words_lower;
    for (const auto& t : tokens)
        words_lower.push_back(t.word);

    // Detect sentence boundaries for is_first
    std::vector<bool> is_first(tokens.size(), false);
    is_first[0] = true;
    for (size_t i = 1; i < tokens.size(); i++) {
        const std::string& prev_trail = tokens[i - 1].trail;
        for (char c : prev_trail) {
            if (c == '.' || c == '?' || c == '!') {
                is_first[i] = true;
                break;
            }
        }
    }

    std::vector<std::vector<std::string>> feats_seq;
    for (size_t i = 0; i < tokens.size(); i++)
        feats_seq.push_back(extract_features(words_lower, (int)i, is_first[i]));

    // Viterbi decode
    std::vector<int> path = viterbi(*ctx, feats_seq);

    if (getenv("TRUECASER_DEBUG")) {
        for (size_t j = 0; j < tokens.size(); j++) {
            fprintf(stderr, "  tc[%zu] w='%s' label=%d (%s) feats=%zu\n", j, tokens[j].word.c_str(), path[j],
                    (path[j] < ctx->n_labels) ? ctx->labels[path[j]].c_str() : "?", feats_seq[j].size());
        }
    }

    // Apply labels
    std::string result;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i > 0)
            result += ' ';

        const std::string& word = tokens[i].word;
        int label = path[i];

        if (word.empty()) {
            result += tokens[i].trail;
            continue;
        }

        std::string cased;
        if (label < ctx->n_labels && ctx->labels[label] == "u1") {
            cased = capitalise(lowercase(word));
        } else if (label < ctx->n_labels && ctx->labels[label] == "uc") {
            cased = to_upper(word);
        } else {
            cased = lowercase(word);
        }

        // Force capitalise after sentence boundary
        if (is_first[i])
            cased = capitalise(cased);

        result += cased;
        result += tokens[i].trail;
    }

    char* out = (char*)malloc(result.size() + 1);
    memcpy(out, result.c_str(), result.size() + 1);
    return out;
}

// ---------------------------------------------------------------------------
// Free
// ---------------------------------------------------------------------------

void truecaser_crf_free(truecaser_crf_context* ctx) {
    delete ctx;
}
