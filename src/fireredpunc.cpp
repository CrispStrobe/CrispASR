// fireredpunc.cpp — FireRedPunc BERT-based punctuation restoration.
//
// Architecture: BERT-base (12L, d=768, 12 heads, d_ffn=3072, GELU)
//               + Linear(768, 5) classifier
//               5 classes: space(0), ，(1), 。(2), ？(3), ！(4)
//
// Input:  unpunctuated text (tokenised with BERT WordPiece vocab)
// Output: text with punctuation inserted

#include "fireredpunc.h"

#include "core/bert_norm.h"
#include "core/bert_pretok.h"
#include "core/punct_marks.h"
#include "core/gguf_loader.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/gpu_backend_pref.h"
#include "gguf.h"
#include "core/env_gate.h"
#include "core/ggml_cpu_backend.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ===========================================================================
// Bench instrumentation — `FIREREDPUNC_BENCH=1` for per-stage timings.
// ===========================================================================

static bool fireredpunc_bench_enabled() {
    static const bool v = core_env::on("FIREREDPUNC_BENCH");
    return v;
}

struct fireredpunc_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit fireredpunc_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~fireredpunc_bench_stage() {
        if (!fireredpunc_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  fireredpunc_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ---------------------------------------------------------------------------
// WordPiece tokenizer (minimal, matching BERT chinese-bert-wwm-ext)
// ---------------------------------------------------------------------------

namespace {

struct WordPieceTokenizer {
    std::vector<std::string> id_to_token;
    std::vector<float> scores; // SP Unigram log-probs (empty => greedy fallback)
    std::map<std::string, int> token_to_id;
    size_t max_piece_len = 0;
    int unk_id = 100;              // [UNK]
    int cls_id = 101;              // [CLS]
    int sep_id = 102;              // [SEP]
    int pad_id = 0;                // [PAD]
    bool is_sentencepiece = false; // true for XLM-RoBERTa

    void build_map() {
        token_to_id.clear();
        max_piece_len = 0;
        for (int i = 0; i < (int)id_to_token.size(); i++) {
            token_to_id[id_to_token[i]] = i;
            max_piece_len = std::max(max_piece_len, id_to_token[i].size());
        }
    }

    // SentencePiece Unigram is NOT greedy — the correct segmentation maximises the
    // sum of piece log-probs (Viterbi). Greedy longest-match mis-splits multi-subword
    // words (e.g. "delayed"), corrupting the XLM-R embeddings. Needs tokenizer.ggml.scores.
    std::vector<int> tokenize_sp_viterbi(const std::string& text) const {
        std::string s; // SP-normalised: whitespace-split words each ▁-prefixed
        {
            std::string cur;
            auto flush = [&]() {
                if (!cur.empty()) {
                    s += "\xE2\x96\x81";
                    s += cur;
                    cur.clear();
                }
            };
            for (char c : text) {
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
                    flush();
                else
                    cur += c;
            }
            flush();
        }
        const int n = (int)s.size();
        if (n == 0)
            return {};
        auto is_boundary = [&](int i) { return i == 0 || i == n || ((unsigned char)s[i] & 0xC0) != 0x80; };
        const double NEG = -1e30;
        std::vector<double> dp(n + 1, NEG);
        std::vector<int> bp(n + 1, -1), bid(n + 1, -1);
        dp[0] = 0.0;
        for (int i = 1; i <= n; i++) {
            if (!is_boundary(i))
                continue;
            int jmin = (max_piece_len && i > (int)max_piece_len) ? i - (int)max_piece_len : 0;
            for (int j = i - 1; j >= jmin; j--) {
                if (!is_boundary(j) || dp[j] <= NEG / 2)
                    continue;
                auto it = token_to_id.find(s.substr(j, i - j));
                if (it == token_to_id.end())
                    continue;
                double sc = dp[j] + scores[it->second];
                if (sc > dp[i]) {
                    dp[i] = sc;
                    bp[i] = j;
                    bid[i] = it->second;
                }
            }
            if (dp[i] <= NEG / 2) { // one-char <unk> fallback keeps the lattice connected
                int j = i - 1;
                while (j > 0 && !is_boundary(j))
                    j--;
                if (dp[j] > NEG / 2) {
                    dp[i] = dp[j] - 1e4;
                    bp[i] = j;
                    bid[i] = unk_id;
                }
            }
        }
        std::vector<int> ids;
        for (int i = n; i > 0;) {
            if (bp[i] < 0)
                return {};
            ids.push_back(bid[i]);
            i = bp[i];
        }
        std::reverse(ids.begin(), ids.end());
        return ids;
    }

    int lookup(const std::string& tok) const {
        auto it = token_to_id.find(tok);
        return it != token_to_id.end() ? it->second : unk_id;
    }

    // SentencePiece tokenization. Uses Unigram Viterbi when scores are present
    // (correct for XLM-R); falls back to greedy longest-match otherwise.
    std::vector<int> tokenize_sp(const std::string& text) const {
        if (!scores.empty())
            return tokenize_sp_viterbi(text);
        std::vector<int> ids;
        // Split on whitespace
        std::vector<std::string> words;
        std::string cur;
        for (char c : text) {
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                if (!cur.empty()) {
                    words.push_back(cur);
                    cur.clear();
                }
            } else {
                cur += c;
            }
        }
        if (!cur.empty())
            words.push_back(cur);

        for (size_t wi = 0; wi < words.size(); wi++) {
            // Prefix with ▁ (U+2581, 0xE2 0x96 0x81)
            std::string word = "\xE2\x96\x81" + words[wi];
            size_t start = 0;
            while (start < word.size()) {
                size_t end = word.size();
                int best_id = -1;
                while (end > start) {
                    std::string sub = word.substr(start, end - start);
                    auto it = token_to_id.find(sub);
                    if (it != token_to_id.end()) {
                        best_id = it->second;
                        break;
                    }
                    end--;
                    while (end > start && (word[end] & 0xC0) == 0x80)
                        end--;
                }
                if (best_id < 0) {
                    ids.push_back(unk_id);
                    break;
                }
                ids.push_back(best_id);
                start = end;
            }
        }
        return ids;
    }

    // BERT tokenization for chinese-bert-wwm-ext (uncased, 21128).
    //
    // This used to split on ASCII whitespace ONLY and lowercase ASCII A-Z
    // only, which is catastrophic for the CHINESE vocab it serves: Chinese
    // text has no spaces, so a whole sentence became ONE word and every
    // character after the first was looked up as a `##` continuation.
    // Measured against hfl/chinese-bert-wwm-ext:
    //
    //   今天天气很好...  HF:   今  天  天  气  ...
    //                    ours: 今 ##天 ##天 ##气 ...
    //
    // i.e. a different token id for essentially every character of every
    // Chinese sentence — on the model's primary input language. Latin was
    // wrong too (`café` -> `ca` `##f` `[UNK]` vs HF's `cafe`). Only pure
    // ASCII English agreed: 2/7 fixtures exact.
    //
    // The HF-correct stack (core/bert_norm.h + core/bert_pretok.h, measured
    // HF-exact for the embedders) runs by DEFAULT, behind
    // CRISPEMBED_FIREREDPUNC_HF_TOK=0 as the opt-out. Token ids went 2/9 -> 9/9
    // exact vs HF's BertTokenizer on the same vocab.
    //
    // ⚠ This block used to say the gate was OFF by default because the HF arm
    // made the decoded output WORSE:
    //
    //   in:   café Müller ist hier und arbeitet gut
    //   off:  Café Müller ist hier und arbeitet gut.     <- correct
    //   on:   Café Müller ist hier und arbeitet. Gut     <- period misplaced
    //
    // That was true, and it was NOT this function: `fireredpunc_process` used
    // to re-derive the subtoken COUNT per word ("Must match the tokenizer's
    // splitting exactly") in a SECOND copy of the WordPiece loop with its own
    // word list, so changing the splitting here alone desynchronised the
    // alignment and the labels landed on the wrong positions.
    //
    // That unification HAS since been done — the counts now come from the
    // tokenizer itself (`out_word_ntok`), the second loop is gone, and the
    // default was flipped ON. The comment simply never got updated, so for a
    // while the file documented the opposite of what it did. If you are here
    // because the two disagree again, the CODE is the answer: read the
    // `explicitly_off` call below, not this paragraph.
    std::vector<int> tokenize(const std::string& text) const { return tokenize_ex(text, nullptr); }

    // `out_tokens`, when non-null, receives the token SURFACE FORMS aligned
    // 1:1 with the returned ids — `##` prefixes preserved, and an
    // unsegmentable word recovered to its original characters (upstream's
    // `recover_unk`). fireredpunc_process needs them because the blueprint
    // rebuilds the punctuated text FROM THE TOKENS, not from the input.
    std::vector<int> tokenize_ex(const std::string& text, std::vector<std::string>* out_tokens,
                                 std::vector<std::string>* out_words = nullptr,
                                 std::vector<int>* out_word_ntok = nullptr) const {
        if (is_sentencepiece)
            return tokenize_sp(text);
        static const bool hf_tok = !core_env::explicitly_off("CRISPEMBED_FIREREDPUNC_HF_TOK");
        std::vector<int> ids;
        auto emit = [&](int id, const std::string& tok) {
            ids.push_back(id);
            if (out_tokens)
                out_tokens->push_back(tok);
        };

        std::vector<std::string> words;
        // ⚠ Pre-tokenize the ORIGINAL text, then normalize each word for the
        // vocabulary lookup — not the other way round. Normalization is
        // per-codepoint and cannot create or destroy a word boundary, so the
        // two word lists correspond 1:1 by construction, and keeping the
        // original spellings is what lets fireredpunc_process rebuild the
        // user's text instead of a lowercased, accent-stripped copy of it.
        std::vector<std::string> words_orig;
        if (hf_tok) {
            words_orig = core_bert::pretokenize(text);
            words.reserve(words_orig.size());
            for (const auto& w : words_orig)
                words.push_back(core_bert::lower_strip_accents(w));
        } else {
            // Historical: split on ASCII whitespace, lowercase ASCII only.
            std::string cur;
            for (char c : text) {
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                    if (!cur.empty()) {
                        words.push_back(cur);
                        cur.clear();
                    }
                } else {
                    if (c >= 'A' && c <= 'Z')
                        c = c - 'A' + 'a';
                    cur += c;
                }
            }
            if (!cur.empty())
                words.push_back(cur);
        }

        // WordPiece each word
        for (size_t wi = 0; wi < words.size(); wi++) {
            const std::string& word = words[wi];
            const size_t n_before = ids.size();
            struct WordTally {
                std::vector<int>* ntok;
                std::vector<std::string>* ws;
                const std::vector<std::string>* src;
                size_t wi, n_before;
                const std::vector<int>* ids;
                ~WordTally() {
                    if (!ntok)
                        return;
                    ntok->push_back((int)(ids->size() - n_before));
                    if (ws && wi < src->size())
                        ws->push_back((*src)[wi]);
                }
            } tally{out_word_ntok, out_words, &words_orig, wi, n_before, &ids};

            // Try to find the word as-is first
            auto it = token_to_id.find(word);
            if (it != token_to_id.end()) {
                emit(it->second, word);
                continue;
            }

            // WordPiece: greedily match longest subword. HF emits ONE [UNK]
            // for a word it cannot fully segment and DISCARDS the pieces it
            // already matched (`is_bad` in WordpieceTokenizer.tokenize); the
            // historical loop kept the prefix. Collect into a scratch vector
            // so the whole word can be dropped.
            std::vector<int> sub_ids;
            std::vector<std::string> sub_toks;
            bool is_bad = false;
            size_t start = 0;
            while (start < word.size()) {
                size_t end = word.size();
                int best_id = -1;
                std::string best_tok;
                while (end > start) {
                    std::string sub = (start == 0) ? word.substr(0, end) : ("##" + word.substr(start, end - start));
                    auto sit = token_to_id.find(sub);
                    if (sit != token_to_id.end()) {
                        best_id = sit->second;
                        best_tok = sub;
                        break;
                    }
                    // Handle multi-byte UTF-8: don't split mid-character
                    end--;
                    while (end > start && (word[end] & 0xC0) == 0x80)
                        end--;
                }
                if (best_id < 0) {
                    is_bad = true;
                    break;
                }
                sub_ids.push_back(best_id);
                sub_toks.push_back(best_tok);
                start = end;
            }
            if (is_bad && hf_tok) {
                // HF emits ONE [UNK] for the word. Upstream then runs
                // `recover_unk`, which puts the ORIGINAL characters back in
                // place of the [UNK] so the text can still be rebuilt from the
                // tokens — each recovered character becoming its own token,
                // looked up in the vocab and falling back to [UNK].
                size_t i = 0;
                while (i < word.size()) {
                    size_t len = 1;
                    while (i + len < word.size() && (word[i + len] & 0xC0) == 0x80)
                        len++;
                    const std::string ch = word.substr(i, len);
                    auto cit = token_to_id.find(ch);
                    emit(cit != token_to_id.end() ? cit->second : unk_id, ch);
                    i += len;
                }
            } else if (is_bad) {
                for (size_t k = 0; k < sub_ids.size(); k++)
                    emit(sub_ids[k], sub_toks[k]);
                emit(unk_id, "[UNK]");
            } else {
                for (size_t k = 0; k < sub_ids.size(); k++)
                    emit(sub_ids[k], sub_toks[k]);
            }
        }

        return ids;
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Model structure
// ---------------------------------------------------------------------------

struct BertLayer {
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_k_b = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_v_b = nullptr;
    ggml_tensor* attn_out_w = nullptr;
    ggml_tensor* attn_out_b = nullptr;
    ggml_tensor* attn_ln_w = nullptr;
    ggml_tensor* attn_ln_b = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_up_b = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
    ggml_tensor* ffn_down_b = nullptr;
    ggml_tensor* ffn_ln_w = nullptr;
    ggml_tensor* ffn_ln_b = nullptr;
};

struct fireredpunc_context {
    std::vector<int> debug_ids; // fireredpunc_debug_token_ids scratch
    // Hyperparams
    int d_model = 768;
    int d_ffn = 3072;
    int n_heads = 12;
    int n_layers = 12;
    int vocab_size = 21128;
    int max_pos = 512;
    int n_classes = 5;
    int cls_id = 101;
    int pad_id = 0;

    // Labels
    std::vector<std::string> labels;

    // Tokenizer
    WordPieceTokenizer tokenizer;

    // Weights
    ggml_tensor* tok_emb_w = nullptr;
    ggml_tensor* pos_emb_w = nullptr;
    ggml_tensor* type_emb_w = nullptr;
    ggml_tensor* emb_ln_w = nullptr;
    ggml_tensor* emb_ln_b = nullptr;

    std::vector<BertLayer> layers;

    ggml_tensor* cls_w = nullptr;
    ggml_tensor* cls_b = nullptr;

    // Backend
    ggml_backend_t backend = nullptr;
    // GH issue #68: ggml_backend_sched_new asserts that the last
    // backend in its list is CPU when a GPU backend is present —
    // otherwise host-side fallbacks have nowhere to land. We always
    // create a CPU backend alongside `backend` and append it to the
    // sched list (when distinct), even if the model is light enough
    // to fit fully on GPU. Mirror of voxtral4b / mimo_asr / etc.
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_context* w_ctx = nullptr;
    ggml_backend_sched_t sched = nullptr;
};

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

static bool fireredpunc_load(fireredpunc_context& ctx, const char* path) {
    // Pass 1: metadata
    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta)
        return false;

    ctx.d_model = (int)core_gguf::kv_u32(meta, "fireredpunc.d_model", 768);
    ctx.d_ffn = (int)core_gguf::kv_u32(meta, "fireredpunc.d_ffn", 3072);
    ctx.n_heads = (int)core_gguf::kv_u32(meta, "fireredpunc.n_heads", 12);
    ctx.n_layers = (int)core_gguf::kv_u32(meta, "fireredpunc.n_layers", 12);
    ctx.vocab_size = (int)core_gguf::kv_u32(meta, "fireredpunc.vocab_size", 21128);
    ctx.max_pos = (int)core_gguf::kv_u32(meta, "fireredpunc.max_pos", 512);
    ctx.n_classes = (int)core_gguf::kv_u32(meta, "fireredpunc.n_classes", 5);
    ctx.cls_id = (int)core_gguf::kv_u32(meta, "fireredpunc.cls_id", 101);
    ctx.pad_id = (int)core_gguf::kv_u32(meta, "fireredpunc.pad_id", 0);

    // Vocab
    ctx.tokenizer.id_to_token = core_gguf::kv_str_array(meta, "tokenizer.ggml.tokens");
    ctx.tokenizer.scores = core_gguf::kv_f32_array(meta, "tokenizer.ggml.scores");
    ctx.tokenizer.cls_id = ctx.cls_id;
    ctx.tokenizer.pad_id = ctx.pad_id;
    std::string tok_type = core_gguf::kv_str(meta, "fireredpunc.tokenizer_type", "wordpiece");
    ctx.tokenizer.is_sentencepiece = (tok_type == "sentencepiece");
    if (ctx.tokenizer.is_sentencepiece) {
        ctx.tokenizer.unk_id = 3; // XLM-R <unk> = 3
    }
    ctx.tokenizer.build_map();

    // Labels
    ctx.labels = core_gguf::kv_str_array(meta, "fireredpunc.labels");
    if (ctx.labels.empty()) {
        ctx.labels = {" ", "\xef\xbc\x8c", "\xe3\x80\x82", "\xef\xbc\x9f", "\xef\xbc\x81"};
    }

    core_gguf::free_metadata(meta);

    fprintf(stderr, "fireredpunc: %dL, d=%d, ffn=%d, heads=%d, vocab=%d, labels=%d\n", ctx.n_layers, ctx.d_model,
            ctx.d_ffn, ctx.n_heads, ctx.vocab_size, ctx.n_classes);

    // Pass 2: weights
    // FireRedPunc normally uses ggml's best backend. For investigation or a
    // local workaround, `FIREREDPUNC_BACKEND=cpu` forces CPU and
    // `FIREREDPUNC_BACKEND=gpu` forces `init_best()`.
    const char* punc_backend = getenv("FIREREDPUNC_BACKEND");
    const bool force_cpu = punc_backend && strcmp(punc_backend, "cpu") == 0;
    const bool force_gpu = punc_backend && strcmp(punc_backend, "gpu") == 0;
    ctx.backend = (force_cpu && !force_gpu) ? core_cpu_backend::init() : crispasr_init_gpu_backend();
    if (!ctx.backend)
        ctx.backend = core_cpu_backend::init();
    // Always have a separate CPU backend on hand for ggml_backend_sched
    // to fall back to (issue #68). Even though the primary backend is
    // CPU here, we keep the two-backend shape uniform.
    ctx.backend_cpu = core_cpu_backend::init();
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx.backend, "fireredpunc", wl))
        return false;
    ctx.w_ctx = wl.ctx;
    ctx.buf = wl.buf;

    auto& T = wl.tensors;
    // A GGUF of another model family loads its metadata fine and then misses
    // these tensors; binding nullptr and reporting success made the first
    // process() call segfault in ggml_mul_mat (#460). Fail the load instead.
    bool missing = false;
    auto req = [&](const char* n) {
        ggml_tensor* t = core_gguf::require(T, n, "fireredpunc");
        missing |= t == nullptr;
        return t;
    };

    ctx.tok_emb_w = req("emb.tok_emb.weight");
    ctx.pos_emb_w = req("emb.pos_emb.weight");
    ctx.type_emb_w = req("emb.type_emb.weight");
    ctx.emb_ln_w = req("emb.ln.weight");
    ctx.emb_ln_b = req("emb.ln.bias");

    ctx.layers.resize(ctx.n_layers);
    for (int i = 0; i < ctx.n_layers; i++) {
        auto ln = [&](const char* fmt) { return core_gguf::format_layer_name(fmt, i); };
        auto& L = ctx.layers[i];
        L.attn_q_w = req(ln("enc.%d.attn.q.weight").c_str());
        L.attn_q_b = req(ln("enc.%d.attn.q.bias").c_str());
        L.attn_k_w = req(ln("enc.%d.attn.k.weight").c_str());
        L.attn_k_b = req(ln("enc.%d.attn.k.bias").c_str());
        L.attn_v_w = req(ln("enc.%d.attn.v.weight").c_str());
        L.attn_v_b = req(ln("enc.%d.attn.v.bias").c_str());
        L.attn_out_w = req(ln("enc.%d.attn.out.weight").c_str());
        L.attn_out_b = req(ln("enc.%d.attn.out.bias").c_str());
        L.attn_ln_w = req(ln("enc.%d.attn.ln.weight").c_str());
        L.attn_ln_b = req(ln("enc.%d.attn.ln.bias").c_str());
        L.ffn_up_w = req(ln("enc.%d.ffn.up.weight").c_str());
        L.ffn_up_b = req(ln("enc.%d.ffn.up.bias").c_str());
        L.ffn_down_w = req(ln("enc.%d.ffn.down.weight").c_str());
        L.ffn_down_b = req(ln("enc.%d.ffn.down.bias").c_str());
        L.ffn_ln_w = req(ln("enc.%d.ffn.ln.weight").c_str());
        L.ffn_ln_b = req(ln("enc.%d.ffn.ln.bias").c_str());
    }

    ctx.cls_w = req("cls.weight");
    ctx.cls_b = req("cls.bias");
    if (missing) {
        fprintf(stderr, "fireredpunc: '%s' is not a FireRedPunc-family GGUF (required tensors missing)\n", path);
        return false;
    }

    // Scheduler. Issue #68: ggml_backend_sched_new asserts the last
    // backend is CPU when a GPU backend is present, otherwise the
    // process aborts with a stack trace. Pass the CPU backend last so
    // CUDA / Vulkan / Metal hosts don't crash.
    ggml_backend_t backends[2] = {ctx.backend, nullptr};
    int n_backends = 1;
    if (ctx.backend_cpu && ctx.backend_cpu != ctx.backend) {
        backends[n_backends++] = ctx.backend_cpu;
    }
    ctx.sched = ggml_backend_sched_new(backends, nullptr, n_backends, 8192, false, false);

    // NOTE (cross-repo): CrispEmbed's copy of this file installs its imatrix
    // collector on the sched here (crispembed_imatrix_install) so this model can
    // produce imatrix-calibrated quants. CrispASR has no imatrix collector, so
    // that call is the one intentional difference between the two copies, along
    // with core_cpu_backend below. Everything else is meant to stay identical.

    return true;
}

// ---------------------------------------------------------------------------
// Graph build: BERT encoder + classifier
// ---------------------------------------------------------------------------

static std::vector<int> fireredpunc_run(fireredpunc_context& ctx, const std::vector<int>& token_ids) {
    const int N = (int)token_ids.size();
    // Blueprint: [CLS] + tokens, and NO [SEP].
    //
    // fireredpunc_bert.py::_forward is `add_cls()` then BertModel then
    // `outputs[0][:, 1:]` — it prepends [CLS], drops its output, and never
    // appends a separator. This port used to append one anyway. BERT is
    // bidirectional, so that is not a harmless trailing token: every real token
    // attends to it, shifting the whole distribution.
    //
    // Measured against the blueprint (CrispEmbed
    // tools/dump_fireredpunc_reference.py) on the f16 GGUF, 119 tokens: with
    // [SEP] cos_min 0.931090 / max_abs 1.8431; without it cos_min 1.000000 /
    // max_abs 0.0021 and 119/119 argmax agreement. An order of magnitude past
    // the f16 numerical floor, i.e. structural, not precision.
    //
    // ⚠ SCOPED TO THE BERT PATH ONLY. This file also serves an XLM-RoBERTa
    // variant (`fireredpunc.tokenizer_type == "sentencepiece"`, the
    // fullstop-punc / punctuate-all models, 250002 vocab, 6 labels), and there
    // the trailing `</s>` is CORRECT — RoBERTa token classification is trained
    // with `<s> … </s>`. The blueprint above is FireRedPunc's and says nothing
    // about that model, so its shape stays exactly as shipped; flipping it too
    // would be changing a default on no evidence. Verified: that model's output
    // is identical with and without the gate.
    //
    // CRISPEMBED_FIREREDPUNC_SEP=1 restores the old shape for bisection. The
    // gate keeps CrispEmbed's spelling deliberately: this file is shared, and
    // one name that works in both trees beats two that drift.
    static const bool force_sep = core_env::on("CRISPEMBED_FIREREDPUNC_SEP");
    const bool append_sep = force_sep || ctx.tokenizer.is_sentencepiece;
    const int seq_len = N + (append_sep ? 2 : 1);

    // ggml context for compute graph
    size_t mem = ggml_tensor_overhead() * (ctx.n_layers * 40 + 50) + 1024 * 1024;
    struct ggml_init_params gp = {mem, nullptr, true};
    ggml_context* ctx0 = ggml_init(gp);

    // LayerNorm epsilon: BERT (chinese-bert-wwm) = 1e-12; XLM-RoBERTa = 1e-5.
    const float ln_eps = ctx.tokenizer.is_sentencepiece ? 1e-5f : 1e-12f;

    // Input: token IDs [seq_len]
    ggml_tensor* inp_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, seq_len);
    ggml_set_name(inp_ids, "inp_ids");
    ggml_set_input(inp_ids);

    // Position IDs [seq_len]
    ggml_tensor* pos_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, seq_len);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);

    // Token type IDs [seq_len] — all zeros
    ggml_tensor* type_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, seq_len);
    ggml_set_name(type_ids, "type_ids");
    ggml_set_input(type_ids);

    // Embeddings: tok + pos + type
    ggml_tensor* tok_emb = ggml_get_rows(ctx0, ctx.tok_emb_w, inp_ids);    // [seq_len, d]
    ggml_tensor* pos_emb = ggml_get_rows(ctx0, ctx.pos_emb_w, pos_ids);    // [seq_len, d]
    ggml_tensor* type_emb = ggml_get_rows(ctx0, ctx.type_emb_w, type_ids); // [seq_len, d]

    ggml_tensor* emb = ggml_add(ctx0, ggml_add(ctx0, tok_emb, pos_emb), type_emb);
    emb = ggml_norm(ctx0, emb, ln_eps);
    emb = ggml_add(ctx0, ggml_mul(ctx0, emb, ctx.emb_ln_w), ctx.emb_ln_b);
    // emb: [d_model, seq_len] (ggml column-major)

    // Dump embedding output for diff-testing
    if (getenv("FIREREDPUNC_DEBUG")) {
        ggml_set_name(emb, "emb_out");
        ggml_set_output(emb);
    }

    const int d = ctx.d_model;
    const int head_dim = d / ctx.n_heads;
    const int nh = ctx.n_heads;

    ggml_tensor* cur = emb;

    for (int i = 0; i < ctx.n_layers; i++) {
        const auto& L = ctx.layers[i];
        ggml_tensor* residual = cur;

        // Self-attention
        ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, L.attn_q_w, cur), L.attn_q_b);
        ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, L.attn_k_w, cur), L.attn_k_b);
        ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, L.attn_v_w, cur), L.attn_v_b);

        // Reshape for multi-head attention: [d, seq] -> [head_dim, nh, seq]
        Q = ggml_reshape_3d(ctx0, Q, head_dim, nh, seq_len);
        K = ggml_reshape_3d(ctx0, K, head_dim, nh, seq_len);
        V = ggml_reshape_3d(ctx0, V, head_dim, nh, seq_len);

        // Permute to [head_dim, seq_len, nh] for flash_attn_ext
        Q = ggml_permute(ctx0, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);

        // Flash attention: handles QK^T, scale, softmax, V matmul
        const float scale = 1.0f / sqrtf((float)head_dim);
        ggml_tensor* KQV = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        KQV = ggml_reshape_2d(ctx0, KQV, d, seq_len);

        // Output projection
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, L.attn_out_w, KQV), L.attn_out_b);

        // Post-norm (residual + LN)
        cur = ggml_add(ctx0, cur, residual);
        cur = ggml_norm(ctx0, cur, ln_eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, L.attn_ln_w), L.attn_ln_b);

        // FFN
        residual = cur;
        ggml_tensor* ffn = ggml_add(ctx0, ggml_mul_mat(ctx0, L.ffn_up_w, cur), L.ffn_up_b);
        ffn = ggml_gelu_erf(ctx0, ffn);
        ffn = ggml_add(ctx0, ggml_mul_mat(ctx0, L.ffn_down_w, ffn), L.ffn_down_b);

        // Post-norm
        cur = ggml_add(ctx0, ffn, residual);
        cur = ggml_norm(ctx0, cur, ln_eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, L.ffn_ln_w), L.ffn_ln_b);
    }

    // Remove CLS token: cur is [d, seq_len], we want [d, N] starting from position 1
    cur = ggml_view_2d(ctx0, cur, d, N, cur->nb[1], cur->nb[1]); // skip first column

    // Classifier: [n_classes, d] x [d, N] -> [n_classes, N]
    ggml_tensor* logits = ggml_add(ctx0, ggml_mul_mat(ctx0, ctx.cls_w, cur), ctx.cls_b);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    // Build & compute graph
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, logits);

    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, gf)) {
        fprintf(stderr, "fireredpunc: failed to allocate graph\n");
        ggml_free(ctx0);
        return {};
    }

    // Set input data
    {
        std::vector<int32_t> ids(seq_len);
        ids[0] = ctx.cls_id;
        for (int i = 0; i < N; i++)
            ids[i + 1] = token_ids[i];
        // SEP token: 102 for BERT, 2 for RoBERTa. Off by default — see above.
        if (append_sep)
            ids[N + 1] = ctx.tokenizer.is_sentencepiece ? 2 : 102;
        ggml_backend_tensor_set(inp_ids, ids.data(), 0, seq_len * sizeof(int32_t));

        std::vector<int32_t> pos(seq_len);
        // RoBERTa: position IDs start at padding_idx+1 (=2 for XLM-R)
        // BERT: position IDs start at 0
        const int pos_offset = ctx.tokenizer.is_sentencepiece ? (ctx.pad_id + 1) : 0;
        for (int i = 0; i < seq_len; i++)
            pos[i] = i + pos_offset;
        ggml_backend_tensor_set(pos_ids, pos.data(), 0, seq_len * sizeof(int32_t));

        std::vector<int32_t> types(seq_len, 0);
        ggml_backend_tensor_set(type_ids, types.data(), 0, seq_len * sizeof(int32_t));
    }

    ggml_backend_sched_graph_compute(ctx.sched, gf);

    // Dump embedding for diff-testing
    if (getenv("FIREREDPUNC_DEBUG")) {
        ggml_tensor* emb_t = ggml_get_tensor(ctx0, "emb_out");
        if (emb_t) {
            // emb is [d, seq_len], read values at position 15 (first "you")
            const int dump_pos = std::min(15, seq_len - 1);
            std::vector<float> emb_vals(d);
            ggml_backend_tensor_get(emb_t, emb_vals.data(), dump_pos * d * sizeof(float), d * sizeof(float));
            fprintf(stderr, "fireredpunc: emb[%d][:8] = [", dump_pos);
            for (int j = 0; j < 8; j++)
                fprintf(stderr, "%s%.6f", j ? ", " : "", emb_vals[j]);
            fprintf(stderr, "]\n");
            float norm = 0;
            for (int j = 0; j < d; j++)
                norm += emb_vals[j] * emb_vals[j];
            fprintf(stderr, "fireredpunc: emb[%d] norm = %.6f\n", dump_pos, sqrtf(norm));
        }
    }

    // Read logits: [n_classes, N]
    std::vector<float> logits_buf(ctx.n_classes * N);
    ggml_backend_tensor_get(logits, logits_buf.data(), 0, logits_buf.size() * sizeof(float));

    // Diff harness: append per-token raw class logits to $FIREREDPUNC_DUMP_LOGITS
    // (one line per token, n_classes space-separated floats). Lets an A/B measure
    // the pre-argmax distribution — where quantization/imatrix effects actually
    // live — instead of the thresholded restored-string exact-match.
    if (const char* dump_path = getenv("FIREREDPUNC_DUMP_LOGITS")) {
        if (FILE* fp = fopen(dump_path, "a")) {
            for (int t = 0; t < N; t++) {
                for (int c = 0; c < ctx.n_classes; c++) {
                    fprintf(fp, "%s%.7g", c ? " " : "", logits_buf[t * ctx.n_classes + c]);
                }
                fputc('\n', fp);
            }
            fclose(fp);
        }
    }

    // Argmax per token
    std::vector<int> preds(N);
    for (int t = 0; t < N; t++) {
        int best = 0;
        float best_val = logits_buf[t * ctx.n_classes];
        for (int c = 1; c < ctx.n_classes; c++) {
            float v = logits_buf[t * ctx.n_classes + c];
            if (v > best_val) {
                best = c;
                best_val = v;
            }
        }
        preds[t] = best;
    }

    // Debug: print first 5 tokens' logits + IDs for diff-testing
    if (getenv("FIREREDPUNC_DEBUG")) {
        fprintf(stderr, "fireredpunc: %d tokens, %d classes\n", N, ctx.n_classes);
        for (int t = 0; t < N; t++) {
            fprintf(stderr, "  [%d] id=%d pred=%d logits=[", t, token_ids[t], preds[t]);
            for (int c = 0; c < ctx.n_classes; c++)
                fprintf(stderr, "%s%.4f", c ? ", " : "", logits_buf[t * ctx.n_classes + c]);
            fprintf(stderr, "]\n");
        }
    }

    ggml_free(ctx0);
    return preds;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

fireredpunc_context* fireredpunc_init(const char* model_path) {
    auto* ctx = new fireredpunc_context();
    if (!fireredpunc_load(*ctx, model_path)) {
        fireredpunc_free(ctx); // releases whatever the partial load allocated
        return nullptr;
    }
    return ctx;
}

const int* fireredpunc_debug_token_ids(fireredpunc_context* ctx, const char* text, int* out_n) {
    static const int empty = 0;
    if (!ctx || !text) {
        if (out_n)
            *out_n = 0;
        return &empty;
    }
    ctx->debug_ids = ctx->tokenizer.tokenize(std::string(text));
    if (out_n)
        *out_n = (int)ctx->debug_ids.size();
    return ctx->debug_ids.empty() ? &empty : ctx->debug_ids.data();
}

char* fireredpunc_process(fireredpunc_context* ctx, const char* text) {
    if (!ctx || !text)
        return nullptr;
    fireredpunc_bench_stage _bs_total("process_total");

    // Tokenize
    std::string input(text);
    static const bool hf_tok = !core_env::explicitly_off("CRISPEMBED_FIREREDPUNC_HF_TOK");
    std::vector<std::string> tok_strs, words_orig;
    std::vector<int> word_ntok;
    std::vector<int> token_ids = ctx->tokenizer.tokenize_ex(
        input, hf_tok ? &tok_strs : nullptr, hf_tok ? &words_orig : nullptr, hf_tok ? &word_ntok : nullptr);
    if (token_ids.empty()) {
        char* out = (char*)malloc(strlen(text) + 1);
        strcpy(out, text);
        return out;
    }

    // Chunk if longer than max_pos - 2 (CLS + room)
    const int max_chunk = ctx->max_pos - 2;
    std::vector<int> all_preds;

    for (int offset = 0; offset < (int)token_ids.size(); offset += max_chunk) {
        int end = std::min(offset + max_chunk, (int)token_ids.size());
        std::vector<int> chunk(token_ids.begin() + offset, token_ids.begin() + end);
        std::vector<int> preds = fireredpunc_run(*ctx, chunk);
        all_preds.insert(all_preds.end(), preds.begin(), preds.end());
    }

    std::string result;

    // Blueprint reconstruction (fireredpunc/punc.py ModelIO.add_punc_to_txt).
    // It walks TOKENS, not words: one prediction per token, `##` stripped, and
    // a space inserted only between two adjacent alphanumeric tokens whose
    // predecessor predicted "no punctuation".
    //
    //     for i, token in enumerate(token_seq):
    //         tag = out_dict[pred_seq[i]]
    //         if token.startswith("##"): token = token.replace("##", "")
    //         elif re.search("[a-zA-Z0-9#]+", token) and i > 0 and \
    //              re.search("[a-zA-Z0-9#]+", token_seq[i-1]):
    //             if out_dict[pred_seq[i-1]] == DEFAULT_OUT: token = " " + token
    //         txt += token + ("" if tag == DEFAULT_OUT else tag)
    //
    // This is what removes the "must match the tokenizer's splitting exactly"
    // second WordPiece loop below: there is nothing to re-derive, because the
    // predictions already line up with the tokens the model was fed.
    //
    // ⚠ Gate on the ALIGNMENT DATA, not on hf_tok alone. `tokenize_ex` returns
    // early for a SentencePiece tokenizer (`if (is_sentencepiece) return
    // tokenize_sp(text);`) and never fills out_words/out_word_ntok, so for the
    // XLM-RoBERTa punctuation models — fullstop-punc / punctuate-all — these
    // vectors come back EMPTY. Branching on hf_tok alone then ran a loop bounded
    // by `words_orig.size()` zero times and returned an empty string: those
    // models silently produced NOTHING on the default path, while
    // CRISPEMBED_FIREREDPUNC_HF_TOK=0 worked fine.
    //
    // Testing for the data rather than for `is_sentencepiece` is deliberate: it
    // fails safe for any future tokenizer that also skips the alignment, which
    // a hard-coded arch test would not.
    const bool have_word_alignment = !words_orig.empty() && words_orig.size() == word_ntok.size();
    if (hf_tok && have_word_alignment) {
        // ONE DELIBERATE DEVIATION FROM THE BLUEPRINT, and why.
        //
        // Upstream emits the TOKEN surface forms, so its output is lowercased
        // and accent-stripped: `Café` -> `cafe`, and on out-of-domain Japanese
        // `ナイーブ` -> `ナイーフ` (the dakuten is a combining mark the
        // normalizer drops). That is fine for FireRedASR, which is restoring
        // punctuation onto its own ASR output — but CrispEmbed exposes this as
        // `--punct-model`, a post-processor over the user's OCR text, where
        // silently rewriting characters is a regression, not a fix.
        //
        // So predictions follow the blueprint exactly (per token, off the
        // HF-correct tokenization) while the emitted TEXT is the original
        // word. Aggregating to the word is safe here precisely because the
        // HF pre-tokenizer already splits every CJK ideograph into its own
        // word — which is where the per-token granularity actually mattered.
        //
        // The subtoken counts come from the tokenizer itself (word_ntok), so
        // the "must match the tokenizer's splitting exactly" second WordPiece
        // loop is gone rather than kept in sync.
        auto is_alnumish = [](const std::string& s) {
            for (unsigned char c : s)
                if (std::isalnum(c) || c == '#')
                    return true;
            return false;
        };
        int tok_i = 0, prev_pred = 0;
        for (size_t w = 0; w < words_orig.size() && w < word_ntok.size(); w++) {
            const int n_sub = word_ntok[w];
            const int last = tok_i + n_sub - 1;
            tok_i += n_sub;
            const int pred = (n_sub > 0 && last < (int)all_preds.size()) ? all_preds[last] : 0;

            // Blueprint spacing: a space only between two adjacent
            // alphanumeric units whose predecessor predicted "no punctuation".
            // CJK gets none, which is correct.
            if (w > 0 && is_alnumish(words_orig[w]) && is_alnumish(words_orig[w - 1]) && prev_pred == 0) {
                result += ' ';
            }
            result += words_orig[w];
            if (pred > 0 && pred < (int)ctx->labels.size() && !core_punct::ends_in_mark(result)) {
                result += ctx->labels[pred];
            }
            prev_pred = pred;
        }
        // punc.py: txt.replace("  ", " ")
        for (size_t i = 0; i + 1 < result.size();) {
            if (result[i] == ' ' && result[i + 1] == ' ') {
                result.erase(i, 1);
            } else {
                i++;
            }
        }
    } else {
        // Historical reconstruction: re-split the input on whitespace and take the
        // LAST subtoken's prediction per word, re-deriving the subtoken count with
        // a second copy of the WordPiece loop. Kept as the bit-exact comparison arm.
        std::vector<std::string> words;
        // Split input on whitespace to match tokenizer
        {
            std::string cur;
            for (char c : input) {
                if (c == ' ' || c == '\t' || c == '\n') {
                    if (!cur.empty()) {
                        words.push_back(cur);
                        cur.clear();
                    }
                } else {
                    cur += c;
                }
            }
            if (!cur.empty())
                words.push_back(cur);
        }

        // Map predictions back to words. Multiple token predictions per word
        // (from WordPiece splits) — take the prediction of the last subtoken.
        int tok_idx = 0;
        for (size_t w = 0; w < words.size(); w++) {
            if (w > 0)
                result += ' ';
            result += words[w];

            // Count how many subtokens this word consumed
            // Re-tokenize just this word to count
            std::string lword;
            for (char c : words[w]) {
                if (c >= 'A' && c <= 'Z')
                    lword += (char)(c - 'A' + 'a');
                else
                    lword += c;
            }

            // Find how many tokens this word produced.
            // Must match the tokenizer's splitting exactly.
            int n_subtokens;
            if (ctx->tokenizer.is_sentencepiece) {
                // Use the SAME segmentation the model saw (Viterbi when scores present) —
                // re-counting greedily here drifts on multi-subword words.
                n_subtokens = (int)ctx->tokenizer.tokenize_sp(lword).size();
            } else {
                // WordPiece: try whole word, then greedy ## splitting
                n_subtokens = 0;
                auto it = ctx->tokenizer.token_to_id.find(lword);
                if (it != ctx->tokenizer.token_to_id.end()) {
                    n_subtokens = 1;
                } else {
                    size_t start = 0;
                    while (start < lword.size()) {
                        size_t end = lword.size();
                        bool found = false;
                        while (end > start) {
                            std::string sub =
                                (start == 0) ? lword.substr(0, end) : ("##" + lword.substr(start, end - start));
                            if (ctx->tokenizer.token_to_id.count(sub)) {
                                found = true;
                                start = end;
                                n_subtokens++;
                                break;
                            }
                            end--;
                            while (end > start && (lword[end] & 0xC0) == 0x80)
                                end--;
                        }
                        if (!found) {
                            n_subtokens++;
                            break;
                        }
                    }
                }
            }

            // Take the last subtoken's prediction for this word
            int pred_idx = tok_idx + n_subtokens - 1;
            tok_idx += n_subtokens;

            if (pred_idx < (int)all_preds.size()) {
                int pred = all_preds[pred_idx];
                if (pred > 0 && pred < (int)ctx->labels.size() && !core_punct::ends_in_mark(result)) {
                    result += ctx->labels[pred];
                }
            }
        }
    } // end historical reconstruction

    // Auto-detect Latin script → map Chinese full-width punctuation to ASCII.
    // Count Latin letters vs CJK characters to decide.
    {
        int latin = 0, cjk = 0;
        for (size_t i = 0; i < result.size(); i++) {
            unsigned char c = (unsigned char)result[i];
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
                latin++;
            else if (c >= 0xE0)
                cjk++; // rough: 3-byte+ UTF-8 = CJK/fullwidth
        }
        if (latin > cjk) {
            // Replace full-width punctuation with ASCII equivalents.
            //
            // Upstream's RuleBaedTxtFix does this with regexes that also
            // insert a SPACE when the mark sits between two Latin letters:
            //     re.sub(r"([a-z])。([a-z])", r"\1. \2", txt)   (and , ? !)
            // while the start-of-string and end-of-string variants emit a bare
            // mark. Without that space the blueprint's token-wise
            // reconstruction yields "world.this", since it only spaces
            // adjacent alphanumeric tokens when the previous tag was "no
            // punctuation". Applied on the HF path only, so the historical arm
            // stays byte-identical.
            auto is_latin_alpha = [](unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
            std::string mapped;
            mapped.reserve(result.size());
            for (size_t i = 0; i < result.size();) {
                unsigned char b0 = (unsigned char)result[i];
                if (b0 >= 0xE0 && i + 2 < result.size()) {
                    unsigned char b1 = (unsigned char)result[i + 1];
                    unsigned char b2 = (unsigned char)result[i + 2];
                    const char* ascii = nullptr;
                    if (b0 == 0xEF && b1 == 0xBC && b2 == 0x8C)
                        ascii = ","; // ，
                    else if (b0 == 0xE3 && b1 == 0x80 && b2 == 0x82)
                        ascii = "."; // 。
                    else if (b0 == 0xEF && b1 == 0xBC && b2 == 0x9F)
                        ascii = "?"; // ？
                    else if (b0 == 0xEF && b1 == 0xBC && b2 == 0x81)
                        ascii = "!"; // ！
                    if (ascii) {
                        mapped += ascii;
                        const bool prev_alpha =
                            !mapped.empty() && is_latin_alpha((unsigned char)mapped[mapped.size() - 2]);
                        const bool next_alpha = (i + 3 < result.size()) && is_latin_alpha((unsigned char)result[i + 3]);
                        if (hf_tok && prev_alpha && next_alpha)
                            mapped += ' ';
                        i += 3;
                        continue;
                    }
                }
                mapped += result[i++];
            }
            result = mapped;
        }
    }

    // Apply rule-based fixes
    // 1. Capitalize after sentence-ending punctuation
    // 2. Capitalize first letter
    // 3. Capitalize "i" when standalone
    std::string fixed;
    bool cap_next = true;
    for (size_t i = 0; i < result.size(); i++) {
        char c = result[i];
        if (cap_next && c >= 'a' && c <= 'z') {
            fixed += (char)(c - 'a' + 'A');
            cap_next = false;
        } else {
            fixed += c;
            // #308: an ALREADY-uppercase letter must disarm cap_next, or the
            // pending capitalisation lands on the NEXT character and "And"
            // becomes "ANd". This applies on BOTH arms — it is a bug fix, not a
            // behaviour preference, and it is the one that "lived in a dead
            // copy" for months (762d9e27). Keep the literal form:
            // tests/test-copies-in-sync.cpp greps for it in both copies.
            if (c >= 'A' && c <= 'Z') {
                cap_next = false;
            }
            // The HF path goes further. `cap_next` used to survive every
            // character that was not a lowercase ASCII letter, so it stayed
            // armed across CJK too: `我在Google...` came out as `GOogle`. Any
            // non-space character ends the "start of sentence" position, which
            // is also closer to upstream's RuleBaedTxtFix, where only txt[0]
            // and a post-`.!?` position are capitalised.
            if (hf_tok && c != ' ')
                cap_next = false;
            // Check for sentence enders (. ? ! and their full-width versions)
            if (c == '.' || c == '?' || c == '!') {
                cap_next = true;
            }
            // Full-width: 。(E3 80 82), ？(EF BC 9F), ！(EF BC 81)
            if (i >= 2) {
                unsigned char b0 = (unsigned char)result[i - 2];
                unsigned char b1 = (unsigned char)result[i - 1];
                unsigned char b2 = (unsigned char)result[i];
                if ((b0 == 0xE3 && b1 == 0x80 && b2 == 0x82) || (b0 == 0xEF && b1 == 0xBC && b2 == 0x9F) ||
                    (b0 == 0xEF && b1 == 0xBC && b2 == 0x81)) {
                    cap_next = true;
                }
            }
        }
    }

    // Capitalize standalone "i" -> "I"
    for (size_t i = 0; i < fixed.size(); i++) {
        if (fixed[i] == 'i') {
            bool start = (i == 0 || fixed[i - 1] == ' ');
            bool end = (i + 1 >= fixed.size() || fixed[i + 1] == ' ' || fixed[i + 1] == ',' || fixed[i + 1] == '.' ||
                        fixed[i + 1] == '?' || fixed[i + 1] == '!');
            // Also check full-width punctuation
            if (start && end) {
                fixed[i] = 'I';
            }
        }
    }

    char* out = (char*)malloc(fixed.size() + 1);
    memcpy(out, fixed.c_str(), fixed.size() + 1);
    return out;
}

void fireredpunc_free(fireredpunc_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    core_gguf::release_weight_buffer(ctx->buf);
    if (ctx->w_ctx)
        ggml_free(ctx->w_ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}
