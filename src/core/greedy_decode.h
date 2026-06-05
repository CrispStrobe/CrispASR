// src/core/greedy_decode.h — shared autoregressive greedy decode loop.
//
// Every LLM-based backend (voxtral, voxtral4b, qwen3, granite) ends its
// transcribe pipeline with the same pattern:
//
//     first_token = argmax(prefill_logits[last_position])
//     gen         = [first_token]
//     n_past      = T_prompt
//     while (gen.size() < max_new_tokens && gen.back() != eos_id) {
//         emb    = model_embed_tokens(ctx, &gen.back(), 1)
//         logits = model_run_llm_kv(ctx, emb, 1, n_past, ..., &vocab)
//         n_past += 1
//         gen.push_back(argmax(logits, vocab))
//     }
//
// This helper captures that loop exactly once. Callers supply their
// model's C-API embed and forward function pointers and an optional
// per-step "pre-forward hook" that can mutate the embedding buffer and/or
// signal early termination — that's how voxtral4b's realtime streaming
// path (which ADDS the next adapter frame to every tail embedding) fits
// into the same skeleton as the straightforward voxtral/qwen3/granite
// path.
//
// Header-only so the compiler inlines each caller's concrete function
// pointers at the call site, keeping the ggml graph structure identical
// to the original hand-rolled loop (regression-gated bit-identity
// preserved).
//
// Usage (voxtral / qwen3 / granite — no hook needed):
//
//     core_greedy_decode::Config cfg;
//     cfg.max_new_tokens = params.max_new_tokens;
//     cfg.eos_id         = 2;            // Mistral Tekken EOS for voxtral
//     cfg.vocab_size     = vocab;
//
//     auto ids = core_greedy_decode::run(
//         ctx,
//         /*first_token=*/argmax_on_prefill_logits,
//         /*n_past=*/T_prompt,
//         voxtral_embed_tokens,
//         voxtral_run_llm_kv,
//         cfg);
//
// Usage (voxtral4b — streaming, pre-forward hook):
//
//     int adapter_pos = T_prompt;
//     auto pre_hook = [&](int /*step*/, float * tail) -> bool {
//         if (adapter_pos >= N_enc) return false;   // audio ran out
//         for (int j = 0; j < pdim; j++) {
//             tail[j] += audio_embeds[(size_t)adapter_pos * pdim + j];
//         }
//         adapter_pos++;
//         return true;
//     };
//     auto ids = core_greedy_decode::run(ctx, first, T_prompt,
//         voxtral4b_embed_tokens, voxtral4b_run_llm_kv, pre_hook, cfg);

#pragma once

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace core_greedy_decode {

struct Config {
    int max_new_tokens = 512; // hard cap on generated tokens
    int eos_id = 2;           // stop as soon as this token is produced
    int vocab_size = 0;       // required — use the value from prefill

    // Sampling knobs. temperature <= 0 gives pure argmax (the historical
    // greedy path). temperature > 0 draws from softmax(logits / temperature);
    // callers should pass whisper_params.temperature through. seed controls
    // the RNG; pass 0 for "non-deterministic" (time-based) or any non-zero
    // value for reproducibility.
    float temperature = 0.0f;
    float frequency_penalty = 0.0f; // 0 = disabled; subtracts penalty * generated count
    uint64_t seed = 0;
};

// Greedy argmax over a vocab-sized float logit vector.
static inline int argmax(const float* logits, int vocab) {
    int best = 0;
    float mx = logits[0];
    for (int k = 1; k < vocab; k++) {
        if (logits[k] > mx) {
            mx = logits[k];
            best = k;
        }
    }
    return best;
}

// Softmax probability of a specific token id under a logit vector.
// Uses the same numerically-stable trick as sample_temp (subtract
// the argmax logit before exp) but without the temperature division.
// Caller passes `best` == the argmax to avoid computing it twice.
static inline float softmax_of(const float* logits, int vocab, int best, float best_lp) {
    (void)best; // computed by caller; keeps the interface symmetric
    double sum = 0.0;
    for (int k = 0; k < vocab; k++) {
        sum += std::exp((double)(logits[k] - best_lp));
    }
    return sum > 0.0 ? (float)(1.0 / sum) : 0.0f;
}

// Temperature sampling over a vocab-sized float logit vector. Computes the
// numerically stable softmax of logits/temperature and draws one token via
// std::discrete_distribution. Caller owns the rng state.
static inline int sample_temp(const float* logits, int vocab, float temperature, std::mt19937_64& rng) {
    const float inv_t = 1.0f / temperature;
    float mx = logits[0] * inv_t;
    for (int k = 1; k < vocab; k++) {
        const float s = logits[k] * inv_t;
        if (s > mx)
            mx = s;
    }
    std::vector<double> probs((size_t)vocab);
    double sum = 0.0;
    for (int k = 0; k < vocab; k++) {
        const double e = std::exp((double)(logits[k] * inv_t - mx));
        probs[(size_t)k] = e;
        sum += e;
    }
    if (sum <= 0.0)
        return argmax(logits, vocab);
    // Inverse-CDF sampling — faster than discrete_distribution when vocab
    // is large because we avoid an extra normalize step inside the STL impl.
    std::uniform_real_distribution<double> unif(0.0, sum);
    const double r = unif(rng);
    double acc = 0.0;
    for (int k = 0; k < vocab; k++) {
        acc += probs[(size_t)k];
        if (r <= acc)
            return k;
    }
    return vocab - 1;
}

static inline const float* penalized_logits(const float* logits, int vocab, float frequency_penalty,
                                            const std::vector<int>& token_counts, std::vector<float>& scratch) {
    if (frequency_penalty <= 0.0f || token_counts.empty())
        return logits;
    scratch.resize((size_t)vocab);
    std::memcpy(scratch.data(), logits, (size_t)vocab * sizeof(float));
    const int n = std::min(vocab, (int)token_counts.size());
    for (int i = 0; i < n; ++i) {
        if (token_counts[(size_t)i] > 0)
            scratch[(size_t)i] -= frequency_penalty * (float)token_counts[(size_t)i];
    }
    return scratch.data();
}

static inline void count_generated_token(std::vector<int>& token_counts, int token_id) {
    if (token_id >= 0 && token_id < (int)token_counts.size())
        token_counts[(size_t)token_id]++;
}

// Default "no-op" pre-forward hook. The compiler inlines and prunes
// the body at call sites that don't need a hook.
struct NoHook {
    inline bool operator()(int /*step*/, float* /*embed*/) const { return true; }
};

// Optional per-token confidence output, aligned with the returned token
// vector. Callers that don't need confidences can pass nullptr and the
// helper skips the extra softmax pass entirely.
struct Result {
    std::vector<int32_t> tokens; // generated token ids (incl. first, incl. eos if hit)
    std::vector<float> probs;    // softmax prob per token in [0,1]; empty when skipped
};

// Run the greedy decode loop.
//
// Template parameters:
//   Ctx      : the model's opaque context type (voxtral_context*, etc.)
//   EmbedFn  : signature float * (Ctx *, const int32_t * ids, int n_ids)
//              returning a malloc'd embedding buffer (caller owns).
//   ForwardFn: signature float * (Ctx *, const float * embeds, int n_tokens,
//              int n_past, int * out_n_tokens, int * out_vocab_size)
//              returning a malloc'd logits buffer.
//   PreHook  : callable (int step, float * emb) -> bool. Return false to
//              terminate the loop (used by voxtral4b's audio-streaming
//              stop condition).
//
// The helper owns the embed/logits heap buffers inside the loop and
// free()s them before moving on. It does not free the `ctx`.
//
// Returns a vector of generated token IDs INCLUDING `first_token` at
// index 0 and possibly the final `eos_id` if the loop hit it (matching
// the convention of the existing backends). The caller is responsible
// for filtering out non-printable / control tokens before detokenising.
template <typename Ctx, typename EmbedFn, typename ForwardFn, typename PreHook>
std::vector<int32_t> run(Ctx* ctx, int32_t first_token, int initial_n_past, EmbedFn embed_fn, ForwardFn forward_fn,
                         PreHook pre_hook, const Config& cfg) {
    std::vector<int32_t> gen;
    gen.reserve((size_t)cfg.max_new_tokens);
    gen.push_back(first_token);

    // Early-exit when the prefill already predicted EOS.
    if (first_token == cfg.eos_id)
        return gen;

    // RNG state is only touched on the sampling path. Seeding is cheap
    // compared to a single vocab-sized softmax, so we always seed even
    // when we won't sample.
    std::mt19937_64 rng(cfg.seed != 0 ? cfg.seed : (uint64_t)std::random_device{}());
    const bool sampling = cfg.temperature > 0.0f;
    std::vector<int> token_counts(cfg.frequency_penalty > 0.0f ? (size_t)cfg.vocab_size : 0);
    std::vector<float> adjusted_logits;
    count_generated_token(token_counts, first_token);

    int n_past = initial_n_past;
    while ((int)gen.size() < cfg.max_new_tokens && gen.back() != cfg.eos_id) {
        const int step = (int)gen.size() - 1; // 0 = first decoded step
        int32_t last = gen.back();

        float* emb = embed_fn(ctx, &last, 1);
        if (!emb)
            break;

        // Let the caller mutate the embedding (e.g. add adapter frame)
        // and/or terminate the loop early (e.g. audio exhausted).
        if (!pre_hook(step, emb)) {
            std::free(emb);
            break;
        }

        float* lg = forward_fn(ctx, emb, 1, n_past, nullptr, nullptr);
        std::free(emb);
        if (!lg)
            break;
        n_past++;

        const float* pick_logits =
            penalized_logits(lg, cfg.vocab_size, cfg.frequency_penalty, token_counts, adjusted_logits);
        const int nx = sampling ? sample_temp(pick_logits, cfg.vocab_size, cfg.temperature, rng)
                                : argmax(pick_logits, cfg.vocab_size);
        std::free(lg);
        gen.push_back(nx);
        count_generated_token(token_counts, nx);
    }

    return gen;
}

// Convenience overload: no pre-forward hook (the common case).
template <typename Ctx, typename EmbedFn, typename ForwardFn>
inline std::vector<int32_t> run(Ctx* ctx, int32_t first_token, int initial_n_past, EmbedFn embed_fn,
                                ForwardFn forward_fn, const Config& cfg) {
    return run(ctx, first_token, initial_n_past, embed_fn, forward_fn, NoHook{}, cfg);
}

// --- run_with_probs: same loop as run() but also records the softmax
// probability of each emitted token so the caller can surface
// per-token confidence to the crispasr_segment vector. The first_prob
// parameter is the pre-computed softmax probability of `first_token`
// under the prefill logits — the caller already computed it when
// picking first_token so we don't redo the softmax here.
template <typename Ctx, typename EmbedFn, typename ForwardFn>
inline Result run_with_probs(Ctx* ctx, int32_t first_token, float first_prob, int initial_n_past, EmbedFn embed_fn,
                             ForwardFn forward_fn, const Config& cfg) {
    Result r;
    r.tokens.reserve((size_t)cfg.max_new_tokens);
    r.probs.reserve((size_t)cfg.max_new_tokens);
    r.tokens.push_back(first_token);
    r.probs.push_back(first_prob);

    if (first_token == cfg.eos_id)
        return r;

    std::mt19937_64 rng(cfg.seed != 0 ? cfg.seed : (uint64_t)std::random_device{}());
    const bool sampling = cfg.temperature > 0.0f;
    std::vector<int> token_counts(cfg.frequency_penalty > 0.0f ? (size_t)cfg.vocab_size : 0);
    std::vector<float> adjusted_logits;
    count_generated_token(token_counts, first_token);

    int n_past = initial_n_past;
    while ((int)r.tokens.size() < cfg.max_new_tokens && r.tokens.back() != cfg.eos_id) {
        int32_t last = r.tokens.back();
        float* emb = embed_fn(ctx, &last, 1);
        if (!emb)
            break;

        float* lg = forward_fn(ctx, emb, 1, n_past, nullptr, nullptr);
        std::free(emb);
        if (!lg)
            break;
        n_past++;

        // Pick next token + compute its softmax probability.
        int nx;
        float nx_lp;
        const float* pick_logits =
            penalized_logits(lg, cfg.vocab_size, cfg.frequency_penalty, token_counts, adjusted_logits);
        if (sampling) {
            nx = sample_temp(pick_logits, cfg.vocab_size, cfg.temperature, rng);
            nx_lp = pick_logits[nx];
        } else {
            nx = argmax(pick_logits, cfg.vocab_size);
            nx_lp = pick_logits[nx];
        }
        const float nx_p = softmax_of(pick_logits, cfg.vocab_size, nx, nx_lp);
        std::free(lg);

        r.tokens.push_back(nx);
        r.probs.push_back(nx_p);
        count_generated_token(token_counts, nx);
    }
    return r;
}

// run_with_probs + pre-forward hook (for voxtral4b streaming audio injection).
template <typename Ctx, typename EmbedFn, typename ForwardFn, typename PreHook>
inline Result run_with_probs(Ctx* ctx, int32_t first_token, float first_prob, int initial_n_past, EmbedFn embed_fn,
                             ForwardFn forward_fn, PreHook pre_hook, const Config& cfg) {
    Result r;
    r.tokens.reserve((size_t)cfg.max_new_tokens);
    r.probs.reserve((size_t)cfg.max_new_tokens);
    r.tokens.push_back(first_token);
    r.probs.push_back(first_prob);

    if (first_token == cfg.eos_id)
        return r;

    std::mt19937_64 rng(cfg.seed != 0 ? cfg.seed : (uint64_t)std::random_device{}());
    const bool sampling = cfg.temperature > 0.0f;
    std::vector<int> token_counts(cfg.frequency_penalty > 0.0f ? (size_t)cfg.vocab_size : 0);
    std::vector<float> adjusted_logits;
    count_generated_token(token_counts, first_token);

    int n_past = initial_n_past;
    while ((int)r.tokens.size() < cfg.max_new_tokens && r.tokens.back() != cfg.eos_id) {
        const int step = (int)r.tokens.size() - 1;
        int32_t last = r.tokens.back();
        float* emb = embed_fn(ctx, &last, 1);
        if (!emb)
            break;

        if (!pre_hook(step, emb)) {
            std::free(emb);
            break;
        }

        float* lg = forward_fn(ctx, emb, 1, n_past, nullptr, nullptr);
        std::free(emb);
        if (!lg)
            break;
        n_past++;

        int nx;
        float nx_lp;
        const float* pick_logits =
            penalized_logits(lg, cfg.vocab_size, cfg.frequency_penalty, token_counts, adjusted_logits);
        if (sampling) {
            nx = sample_temp(pick_logits, cfg.vocab_size, cfg.temperature, rng);
            nx_lp = pick_logits[nx];
        } else {
            nx = argmax(pick_logits, cfg.vocab_size);
            nx_lp = pick_logits[nx];
        }
        const float nx_p = softmax_of(pick_logits, cfg.vocab_size, nx, nx_lp);
        std::free(lg);

        r.tokens.push_back(nx);
        r.probs.push_back(nx_p);
        count_generated_token(token_counts, nx);
    }
    return r;
}

} // namespace core_greedy_decode
