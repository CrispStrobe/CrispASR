#include <cmath>
// lid_fasttext.cpp — see lid_fasttext.h.
//
// fastText supervised LID forward pass, manual F32/F16 path. The model
// has only two weight tensors (embedding, output) and the compute is
// ~1 MFLOP per call, so a ggml graph would be pure overhead. Quant
// support (Phase 5) will need a small graph for the matmul.
//
// The forward, in pseudocode:
//
//   input_ids = []
//   for word in whitespace_split(text):
//     wid = vocab.get(word, -1)
//     if wid >= 0:
//       input_ids.append(wid)                              // word row
//     for ngram in char_ngrams(word, minn..maxn):
//       input_ids.append(n_words + fnv1a(ngram) % bucket)  // subword bucket row
//   input_ids.append(0)                                    // </s> EOS row (the trap!)
//   embedding_bag = mean(embedding[input_ids])
//   logits = output @ embedding_bag
//   softmax = stable_softmax(logits)
//   return labels[argmax(softmax)]

#include "lid_fasttext.h"

#include "core/gguf_loader.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// fastText reserves row 0 for the </s> pseudo-token in supervised mode.
// initNgrams() does not expand subwords for it, so its precomputed
// row list is just [0]. getLine() injects it at end-of-stream.
constexpr int32_t kEosRowId = 0;

// fastText's FNV-1a 32-bit variant. The int8_t cast is load-bearing —
// for UTF-8 bytes ≥ 0x80 the sign-extension flips the high 24 bits to 1
// before the XOR. Drop it and Cyrillic / Han / Devanagari hashes drift.
uint32_t fnv1a_fasttext(const char* data, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) {
        h = h ^ uint32_t(static_cast<int8_t>(data[i]));
        h = h * 16777619u;
    }
    return h;
}

// fastText's whitespace tokenizer, matching ``Dictionary::readWord``:
// split on space / tab / vertical-tab / form-feed / 0x00 / 0x0b / EOS
// chars. Newline becomes EOS in supervised mode but we don't expose
// pre-EOS handling here — getLine's EOS injection is implemented as a
// trailing ``kEosRowId`` push, not a token split.
std::vector<std::string> tokenize_fasttext(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : text) {
        bool is_ws = (c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r' || c == '\0');
        if (is_ws) {
            if (!cur.empty()) {
                out.push_back(std::move(cur));
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty())
        out.push_back(std::move(cur));
    return out;
}

// Mirror of fastText ``Dictionary::computeSubwords``: emit
// ``nwords + (fnv1a(ngram) % bucket)`` for every n ∈ [minn, maxn] of
// the wrapped form ``<word>``, iterating over UTF-8 codepoints (skip
// 10xxxxxx continuation bytes), with the same length-1 boundary skip
// the upstream applies.
void compute_subwords(const std::string& boundary_word, int minn, int maxn, uint32_t bucket, int n_words,
                      std::vector<int32_t>& out) {
    const size_t L = boundary_word.size();
    for (size_t i = 0; i < L; i++) {
        // skip if i is a UTF-8 continuation byte
        if ((static_cast<uint8_t>(boundary_word[i]) & 0xC0) == 0x80)
            continue;
        std::string ngram;
        size_t j = i;
        for (int n = 1; j < L && n <= maxn; n++) {
            ngram.push_back(boundary_word[j++]);
            while (j < L && (static_cast<uint8_t>(boundary_word[j]) & 0xC0) == 0x80) {
                ngram.push_back(boundary_word[j++]);
            }
            // Skip length-1 ngram if it's the leading '<' or trailing '>'.
            const bool is_boundary_alone = (n == 1) && (i == 0 || j == L);
            if (n >= minn && !is_boundary_alone) {
                uint32_t h = fnv1a_fasttext(ngram.data(), ngram.size());
                int32_t bucket_idx = static_cast<int32_t>(h % bucket);
                out.push_back(n_words + bucket_idx);
            }
        }
    }
}

// Tensor row-reader supporting F32 / F16 / any quantized type via the
// ggml type-traits dequant callback. For F32/F16 we read the row in
// place; for quantized types we dequant a full row at a time into the
// caller's scratch buffer using `ggml_get_type_traits(type)->to_float`.
//
// ggml stores `ne[0]` as the innermost (fastest-varying) dimension. The
// converter writes numpy `(n_rows, dim)` which gguf flips to ggml
// `(dim, n_rows)` — i.e. ne0=dim, ne1=n_rows. Element (row=r, dim=k) is
// at byte offset `r * row_size_bytes + (per-element offset)`. For
// quant types the per-element offset is non-trivial, so we always
// fetch a full row at a time.
struct TensorView {
    const uint8_t* data = nullptr; // raw bytes
    int64_t ne0 = 0;               // dim (innermost)
    int64_t ne1 = 0;               // n_rows (outer)
    enum ggml_type type = GGML_TYPE_F32;
    size_t row_bytes = 0;               // bytes per logical row of `ne0` elements
    ggml_to_float_t to_float = nullptr; // dequant callback when not F32
    bool ok = false;
};

TensorView view_tensor(ggml_tensor* t) {
    TensorView v;
    if (!t)
        return v;
    v.data = static_cast<const uint8_t*>(t->data);
    v.ne0 = t->ne[0];
    v.ne1 = t->ne[1];
    v.type = t->type;
    v.row_bytes = ggml_row_size(t->type, t->ne[0]);
    if (t->type == GGML_TYPE_F32) {
        v.to_float = nullptr; // handled inline
    } else {
        const auto* tr = ggml_get_type_traits(t->type);
        if (!tr || !tr->to_float) {
            return v; // ok stays false — unsupported type
        }
        v.to_float = tr->to_float;
    }
    v.ok = true;
    return v;
}

// Dequantize logical row `row` into the scratch buffer `dst[ne0]`.
// For F32 this is a memcpy; for F16/quant types it goes through the
// type-traits dequant callback.
inline void load_row(const TensorView& v, int64_t row, float* dst) {
    const uint8_t* row_ptr = v.data + static_cast<size_t>(row) * v.row_bytes;
    if (v.type == GGML_TYPE_F32) {
        std::memcpy(dst, row_ptr, static_cast<size_t>(v.ne0) * sizeof(float));
    } else {
        v.to_float(row_ptr, dst, v.ne0);
    }
}

} // namespace

// ===========================================================================
// Context
// ===========================================================================

struct lid_fasttext_context {
    // Loaded tensors / weights
    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr;

    // Tensor views
    TensorView embedding;
    TensorView output;

    // Hyperparams
    std::string variant;
    std::string loss; // "softmax" (flat) or "hs" (hierarchical)
    int dim = 0;
    int n_labels = 0;
    int n_words = 0;
    uint32_t bucket = 0;
    int minn = 0;
    int maxn = 0;

    // Hierarchical-softmax tree (only populated when loss == "hs"):
    //   path for label i is hs_paths[hs_path_offsets[i] .. hs_path_offsets[i+1])
    //   parallel codes in hs_codes
    std::vector<int32_t> hs_path_offsets; // size n_labels+1
    std::vector<int32_t> hs_paths;
    std::vector<int8_t> hs_codes;

    // Vocab + labels (kept on the C++ side so we never go back to GGUF)
    std::vector<std::string> labels; // size n_labels — short form, no __label__ prefix
    std::vector<std::string> words;  // size n_words
    std::unordered_map<std::string, int32_t> word_to_id;

    // Scratch buffers reused across calls
    std::vector<int32_t> input_ids;
    std::vector<float> embedding_bag;
    std::vector<float> logits;
    std::vector<float> softmax;
    std::vector<float> row_scratch; // size dim — dequant buffer for one row

    int n_threads = 1;
};

// ===========================================================================
// Forward stages
// ===========================================================================

// Build the input-id list for `text`. Mirrors ``Dictionary::getLine``
// for supervised mode: word_id (if known) + subword bucket IDs per
// word, then a trailing EOS row.
static void build_input_ids(lid_fasttext_context* ctx, const std::string& text) {
    ctx->input_ids.clear();
    auto words = tokenize_fasttext(text);
    for (const auto& w : words) {
        auto it = ctx->word_to_id.find(w);
        const int32_t wid = (it == ctx->word_to_id.end()) ? -1 : it->second;
        if (wid >= 0) {
            ctx->input_ids.push_back(wid);
        }
        // Subword n-grams over <w> wrapped form.
        std::string boundary = "<" + w + ">";
        compute_subwords(boundary, ctx->minn, ctx->maxn, ctx->bucket, ctx->n_words, ctx->input_ids);
    }
    // The </s> end-of-line row — fastText injects this in supervised mode.
    ctx->input_ids.push_back(kEosRowId);
}

static void compute_embedding_bag(lid_fasttext_context* ctx) {
    ctx->embedding_bag.assign(ctx->dim, 0.0f);
    if (ctx->input_ids.empty())
        return;
    if (static_cast<int>(ctx->row_scratch.size()) < ctx->dim)
        ctx->row_scratch.resize(ctx->dim);
    for (int32_t row : ctx->input_ids) {
        if (row < 0 || row >= ctx->embedding.ne1)
            continue;
        load_row(ctx->embedding, row, ctx->row_scratch.data());
        for (int k = 0; k < ctx->dim; k++) {
            ctx->embedding_bag[k] += ctx->row_scratch[k];
        }
    }
    const float inv = 1.0f / static_cast<float>(ctx->input_ids.size());
    for (int k = 0; k < ctx->dim; k++) {
        ctx->embedding_bag[k] *= inv;
    }
}

// Numerically-stable log(sigmoid(x)).
static inline float log_sigmoid_stable(float x) {
    if (x >= 0.0f)
        return -std::log1p(std::exp(-x));
    return x - std::log1p(std::exp(x));
}

static void compute_logits_softmax_flat(lid_fasttext_context* ctx) {
    // Plain output_matrix @ embedding_bag, one row at a time (allows
    // dequant of quantized weights without a graph).
    if (static_cast<int>(ctx->row_scratch.size()) < ctx->dim)
        ctx->row_scratch.resize(ctx->dim);
    ctx->logits.assign(ctx->n_labels, 0.0f);
    for (int j = 0; j < ctx->n_labels; j++) {
        load_row(ctx->output, j, ctx->row_scratch.data());
        float acc = 0.0f;
        for (int k = 0; k < ctx->dim; k++) {
            acc += ctx->row_scratch[k] * ctx->embedding_bag[k];
        }
        ctx->logits[j] = acc;
    }
}

// Hierarchical-softmax forward: log P(label i | hidden) =
//   sum_{(node, code) in path[i]} log_sigmoid((2*code - 1) * (output[node] · hidden))
// Each internal-node row of `output` is dotted at most once per call by
// caching node→dot-product results on the fly.
static void compute_logits_hs(lid_fasttext_context* ctx) {
    if (static_cast<int>(ctx->row_scratch.size()) < ctx->dim)
        ctx->row_scratch.resize(ctx->dim);
    ctx->logits.assign(ctx->n_labels, 0.0f);

    // Cache node-row dot products so we don't redo them for every label
    // sharing internal nodes near the root. Tree has ne1 internal-node
    // rows in the output matrix (≤ n_labels - 1 used; allocate generously).
    const int n_nodes = static_cast<int>(ctx->output.ne1);
    std::vector<float> node_score(n_nodes, 0.0f);
    std::vector<uint8_t> node_done(n_nodes, 0);

    for (int i = 0; i < ctx->n_labels; i++) {
        const int beg = ctx->hs_path_offsets[i];
        const int end = ctx->hs_path_offsets[i + 1];
        float s = 0.0f;
        for (int p = beg; p < end; p++) {
            const int32_t node = ctx->hs_paths[p];
            const int8_t code = ctx->hs_codes[p];
            if (node < 0 || node >= n_nodes) {
                fprintf(stderr, "lid_fasttext[hs]: path step %d out of range [0,%d)\n", node, n_nodes);
                continue;
            }
            if (!node_done[node]) {
                load_row(ctx->output, node, ctx->row_scratch.data());
                float acc = 0.0f;
                for (int k = 0; k < ctx->dim; k++)
                    acc += ctx->row_scratch[k] * ctx->embedding_bag[k];
                node_score[node] = acc;
                node_done[node] = 1;
            }
            s += log_sigmoid_stable((2.0f * code - 1.0f) * node_score[node]);
        }
        ctx->logits[i] = s; // log-prob — semantic difference from flat-softmax case
    }
}

static void compute_softmax(lid_fasttext_context* ctx) {
    ctx->softmax.assign(ctx->n_labels, 0.0f);
    if (ctx->loss == "hs") {
        // logits already are log-probs that sum to ~1 after exp.
        // No max-subtract needed (exp of log-prob is bounded by 1),
        // but we keep one as a numerical guard.
        float max_logit = ctx->logits[0];
        for (int j = 1; j < ctx->n_labels; j++)
            if (ctx->logits[j] > max_logit)
                max_logit = ctx->logits[j];
        double sum = 0.0;
        for (int j = 0; j < ctx->n_labels; j++) {
            const float e = std::exp(ctx->logits[j] - max_logit);
            ctx->softmax[j] = e;
            sum += e;
        }
        // For HS, sum should already be ≈1.0 (modulo float drift from
        // tree-internal log-sigmoid round-off); renormalise to be safe.
        const float inv = static_cast<float>(1.0 / sum);
        for (int j = 0; j < ctx->n_labels; j++)
            ctx->softmax[j] *= inv;
        return;
    }
    // flat softmax
    float max_logit = ctx->logits[0];
    for (int j = 1; j < ctx->n_labels; j++) {
        if (ctx->logits[j] > max_logit)
            max_logit = ctx->logits[j];
    }
    double sum = 0.0;
    for (int j = 0; j < ctx->n_labels; j++) {
        const float e = std::exp(ctx->logits[j] - max_logit);
        ctx->softmax[j] = e;
        sum += e;
    }
    const float inv = static_cast<float>(1.0 / sum);
    for (int j = 0; j < ctx->n_labels; j++)
        ctx->softmax[j] *= inv;
}

static void compute_logits(lid_fasttext_context* ctx) {
    if (ctx->loss == "hs")
        compute_logits_hs(ctx);
    else
        compute_logits_softmax_flat(ctx);
}

static int forward(lid_fasttext_context* ctx, const std::string& text) {
    build_input_ids(ctx, text);
    if (ctx->input_ids.empty())
        return -1;
    compute_embedding_bag(ctx);
    compute_logits(ctx);
    compute_softmax(ctx);
    int top1 = 0;
    for (int j = 1; j < ctx->n_labels; j++) {
        if (ctx->softmax[j] > ctx->softmax[top1])
            top1 = j;
    }
    return top1;
}

// ===========================================================================
// Public C ABI
// ===========================================================================

extern "C" struct lid_fasttext_context* lid_fasttext_init_from_file(const char* gguf_path, int n_threads) {
    if (!gguf_path) {
        fprintf(stderr, "lid_fasttext: NULL gguf_path\n");
        return nullptr;
    }

    // Pass 1: metadata
    gguf_context* meta = core_gguf::open_metadata(gguf_path);
    if (!meta) {
        fprintf(stderr, "lid_fasttext: cannot open '%s' for metadata\n", gguf_path);
        return nullptr;
    }

    auto* ctx = new lid_fasttext_context();
    ctx->n_threads = (n_threads > 0) ? n_threads : 1;
    ctx->variant = core_gguf::kv_str(meta, "lid_fasttext.variant", "");
    ctx->dim = static_cast<int>(core_gguf::kv_u32(meta, "lid_fasttext.dim", 0));
    ctx->n_labels = static_cast<int>(core_gguf::kv_u32(meta, "lid_fasttext.n_labels", 0));
    ctx->n_words = static_cast<int>(core_gguf::kv_u32(meta, "lid_fasttext.n_words", 0));
    ctx->bucket = core_gguf::kv_u32(meta, "lid_fasttext.bucket", 0);
    ctx->minn = static_cast<int>(core_gguf::kv_u32(meta, "lid_fasttext.minn", 0));
    ctx->maxn = static_cast<int>(core_gguf::kv_u32(meta, "lid_fasttext.maxn", 0));
    ctx->labels = core_gguf::kv_str_array(meta, "lid_fasttext.labels");
    ctx->words = core_gguf::kv_str_array(meta, "lid_fasttext.words");
    ctx->loss = core_gguf::kv_str(meta, "lid_fasttext.loss", "softmax");
    core_gguf::free_metadata(meta);
    if (ctx->loss != "softmax" && ctx->loss != "hs") {
        fprintf(stderr, "lid_fasttext: unsupported loss '%s' (only 'softmax' and 'hs' are wired up)\n",
                ctx->loss.c_str());
        delete ctx;
        return nullptr;
    }

    // Validation
    if (ctx->dim <= 0 || ctx->n_labels <= 0 || ctx->bucket == 0 || ctx->minn <= 0 || ctx->maxn < ctx->minn) {
        fprintf(stderr,
                "lid_fasttext: invalid metadata (dim=%d n_labels=%d bucket=%u minn=%d maxn=%d) — "
                "expected output of convert-glotlid-to-gguf.py\n",
                ctx->dim, ctx->n_labels, ctx->bucket, ctx->minn, ctx->maxn);
        delete ctx;
        return nullptr;
    }
    if (static_cast<int>(ctx->labels.size()) != ctx->n_labels) {
        fprintf(stderr, "lid_fasttext: labels array size %zu != n_labels %d\n", ctx->labels.size(), ctx->n_labels);
        delete ctx;
        return nullptr;
    }
    if (static_cast<int>(ctx->words.size()) != ctx->n_words) {
        fprintf(stderr, "lid_fasttext: words array size %zu != n_words %d\n", ctx->words.size(), ctx->n_words);
        delete ctx;
        return nullptr;
    }

    // Build word→id lookup
    ctx->word_to_id.reserve(ctx->n_words * 2);
    for (int i = 0; i < ctx->n_words; i++) {
        ctx->word_to_id[ctx->words[i]] = i;
    }

    // Pass 2: weights — CPU backend (small enough that GPU offload is overhead).
    ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) {
        fprintf(stderr, "lid_fasttext: ggml_backend_cpu_init failed\n");
        delete ctx;
        return nullptr;
    }
    if (!core_gguf::load_weights(gguf_path, ctx->backend, "lid_fasttext", ctx->wl)) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }

    // Tensor names use the .weight suffix (kokoro/parakeet convention)
    // so crispasr-quantize's is_weight gate picks them up for K-quant
    // re-quantization. Older converters wrote bare "lid_fasttext.embedding"
    // / "lid_fasttext.output"; fall back to those for backward compat.
    auto* emb_t = core_gguf::try_get(ctx->wl.tensors, "lid_fasttext.embedding.weight");
    if (!emb_t)
        emb_t = core_gguf::require(ctx->wl.tensors, "lid_fasttext.embedding", "lid_fasttext");
    auto* out_t = core_gguf::try_get(ctx->wl.tensors, "lid_fasttext.output.weight");
    if (!out_t)
        out_t = core_gguf::require(ctx->wl.tensors, "lid_fasttext.output", "lid_fasttext");
    if (!emb_t || !out_t) {
        core_gguf::free_weights(ctx->wl);
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    ctx->embedding = view_tensor(emb_t);
    ctx->output = view_tensor(out_t);
    if (!ctx->embedding.ok || !ctx->output.ok) {
        fprintf(stderr,
                "lid_fasttext: weight type unsupported (embedding type=%s, output type=%s). "
                "Need F32/F16 or any type with a registered to_float dequant.\n",
                ggml_type_name(emb_t->type), ggml_type_name(out_t->type));
        core_gguf::free_weights(ctx->wl);
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    if (ctx->embedding.ne0 != ctx->dim || ctx->embedding.ne1 != ctx->n_words + (int64_t)ctx->bucket) {
        fprintf(stderr,
                "lid_fasttext: embedding shape mismatch: tensor [%lld, %lld] vs metadata "
                "[n_words+bucket=%lld, dim=%d]\n",
                (long long)ctx->embedding.ne0, (long long)ctx->embedding.ne1,
                (long long)(ctx->n_words + (int64_t)ctx->bucket), ctx->dim);
        core_gguf::free_weights(ctx->wl);
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    if (ctx->output.ne0 != ctx->dim || ctx->output.ne1 != ctx->n_labels) {
        fprintf(stderr, "lid_fasttext: output shape mismatch: tensor [%lld, %lld] vs metadata [n_labels=%d, dim=%d]\n",
                (long long)ctx->output.ne0, (long long)ctx->output.ne1, ctx->n_labels, ctx->dim);
        core_gguf::free_weights(ctx->wl);
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }

    // Load HS tree if loss == "hs". Stored as I32 / I8 tensors so we
    // can read them directly without bit-unpacking metadata arrays.
    if (ctx->loss == "hs") {
        auto* off_t = core_gguf::require(ctx->wl.tensors, "lid_fasttext.hs_path_offsets", "lid_fasttext");
        auto* paths_t = core_gguf::require(ctx->wl.tensors, "lid_fasttext.hs_paths", "lid_fasttext");
        auto* codes_t = core_gguf::require(ctx->wl.tensors, "lid_fasttext.hs_codes", "lid_fasttext");
        if (!off_t || !paths_t || !codes_t) {
            core_gguf::free_weights(ctx->wl);
            ggml_backend_free(ctx->backend);
            delete ctx;
            return nullptr;
        }
        if (off_t->type != GGML_TYPE_I32 || paths_t->type != GGML_TYPE_I32 || codes_t->type != GGML_TYPE_I8) {
            fprintf(stderr,
                    "lid_fasttext[hs]: tree tensors have wrong types "
                    "(offsets=%s paths=%s codes=%s; expected I32 I32 I8)\n",
                    ggml_type_name(off_t->type), ggml_type_name(paths_t->type), ggml_type_name(codes_t->type));
            core_gguf::free_weights(ctx->wl);
            ggml_backend_free(ctx->backend);
            delete ctx;
            return nullptr;
        }
        const int64_t n_off = off_t->ne[0];
        const int64_t n_steps = paths_t->ne[0];
        if (n_off != ctx->n_labels + 1 || codes_t->ne[0] != n_steps) {
            fprintf(stderr,
                    "lid_fasttext[hs]: shape mismatch (offsets=%lld expected %d; "
                    "paths=%lld codes=%lld)\n",
                    (long long)n_off, ctx->n_labels + 1, (long long)n_steps, (long long)codes_t->ne[0]);
            core_gguf::free_weights(ctx->wl);
            ggml_backend_free(ctx->backend);
            delete ctx;
            return nullptr;
        }
        ctx->hs_path_offsets.assign(static_cast<const int32_t*>(off_t->data),
                                    static_cast<const int32_t*>(off_t->data) + n_off);
        ctx->hs_paths.assign(static_cast<const int32_t*>(paths_t->data),
                             static_cast<const int32_t*>(paths_t->data) + n_steps);
        ctx->hs_codes.assign(static_cast<const int8_t*>(codes_t->data),
                             static_cast<const int8_t*>(codes_t->data) + n_steps);
    }

    return ctx;
}

extern "C" void lid_fasttext_free(struct lid_fasttext_context* ctx) {
    if (!ctx)
        return;
    core_gguf::free_weights(ctx->wl);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" const char* lid_fasttext_predict(struct lid_fasttext_context* ctx, const char* utf8_text,
                                            float* confidence) {
    if (!ctx || !utf8_text)
        return nullptr;
    int top1 = forward(ctx, std::string(utf8_text));
    if (top1 < 0)
        return nullptr;
    if (confidence)
        *confidence = ctx->softmax[top1];
    return ctx->labels[top1].c_str();
}

extern "C" int lid_fasttext_predict_topk(struct lid_fasttext_context* ctx, const char* utf8_text, int k,
                                         const char** out_labels, float* out_scores) {
    if (!ctx || !utf8_text || k <= 0 || !out_labels || !out_scores)
        return 0;
    if (forward(ctx, std::string(utf8_text)) < 0)
        return 0;
    // Partial sort: pull the k largest by softmax score.
    std::vector<int> idx(ctx->n_labels);
    for (int j = 0; j < ctx->n_labels; j++)
        idx[j] = j;
    const int kk = std::min(k, ctx->n_labels);
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [ctx](int a, int b) { return ctx->softmax[a] > ctx->softmax[b]; });
    for (int i = 0; i < kk; i++) {
        out_labels[i] = ctx->labels[idx[i]].c_str();
        out_scores[i] = ctx->softmax[idx[i]];
    }
    return kk;
}

extern "C" float* lid_fasttext_extract_stage(struct lid_fasttext_context* ctx, const char* utf8_text,
                                             const char* stage_name, int* out_n) {
    if (!ctx || !utf8_text || !stage_name || !out_n)
        return nullptr;
    if (forward(ctx, std::string(utf8_text)) < 0)
        return nullptr;

    auto alloc_copy = [out_n](const float* src, int n) -> float* {
        float* dst = static_cast<float*>(std::malloc(static_cast<size_t>(n) * sizeof(float)));
        if (!dst)
            return nullptr;
        std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
        *out_n = n;
        return dst;
    };

    if (std::strcmp(stage_name, "input_ids") == 0) {
        // Reference dump stores int32 — but the diff-harness compare()
        // path normalises to float32, so emit floats here too.
        const int n = static_cast<int>(ctx->input_ids.size());
        float* dst = static_cast<float*>(std::malloc(static_cast<size_t>(n) * sizeof(float)));
        if (!dst)
            return nullptr;
        for (int i = 0; i < n; i++)
            dst[i] = static_cast<float>(ctx->input_ids[i]);
        *out_n = n;
        return dst;
    }
    if (std::strcmp(stage_name, "embedding_bag_out") == 0)
        return alloc_copy(ctx->embedding_bag.data(), ctx->dim);
    if (std::strcmp(stage_name, "logits") == 0)
        return alloc_copy(ctx->logits.data(), ctx->n_labels);
    if (std::strcmp(stage_name, "softmax") == 0)
        return alloc_copy(ctx->softmax.data(), ctx->n_labels);
    if (std::strcmp(stage_name, "top1_score") == 0) {
        int top1 = 0;
        for (int j = 1; j < ctx->n_labels; j++) {
            if (ctx->softmax[j] > ctx->softmax[top1])
                top1 = j;
        }
        float* dst = static_cast<float*>(std::malloc(sizeof(float)));
        if (!dst)
            return nullptr;
        dst[0] = ctx->softmax[top1];
        *out_n = 1;
        return dst;
    }

    fprintf(stderr, "lid_fasttext: unknown stage '%s'\n", stage_name);
    return nullptr;
}

extern "C" const char* lid_fasttext_variant(const struct lid_fasttext_context* ctx) {
    return ctx ? ctx->variant.c_str() : "";
}

extern "C" int lid_fasttext_n_labels(const struct lid_fasttext_context* ctx) {
    return ctx ? ctx->n_labels : 0;
}

extern "C" int lid_fasttext_dim(const struct lid_fasttext_context* ctx) {
    return ctx ? ctx->dim : 0;
}
