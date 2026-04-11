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

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace core_greedy_decode {

struct Config {
    int max_new_tokens = 512;  // hard cap on generated tokens
    int eos_id         = 2;    // stop as soon as this token is produced
    int vocab_size     = 0;    // required — use the value from prefill
};

// Greedy argmax over a vocab-sized float logit vector.
static inline int argmax(const float * logits, int vocab) {
    int best = 0;
    float mx = logits[0];
    for (int k = 1; k < vocab; k++) {
        if (logits[k] > mx) { mx = logits[k]; best = k; }
    }
    return best;
}

// Default "no-op" pre-forward hook. The compiler inlines and prunes
// the body at call sites that don't need a hook.
struct NoHook {
    inline bool operator()(int /*step*/, float * /*embed*/) const { return true; }
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
std::vector<int32_t> run(
    Ctx        * ctx,
    int32_t      first_token,
    int          initial_n_past,
    EmbedFn      embed_fn,
    ForwardFn    forward_fn,
    PreHook      pre_hook,
    const Config & cfg)
{
    std::vector<int32_t> gen;
    gen.reserve((size_t)cfg.max_new_tokens);
    gen.push_back(first_token);

    // Early-exit when the prefill already predicted EOS.
    if (first_token == cfg.eos_id) return gen;

    int n_past = initial_n_past;
    while ((int)gen.size() < cfg.max_new_tokens && gen.back() != cfg.eos_id) {
        const int step = (int)gen.size() - 1;     // 0 = first decoded step
        int32_t last = gen.back();

        float * emb = embed_fn(ctx, &last, 1);
        if (!emb) break;

        // Let the caller mutate the embedding (e.g. add adapter frame)
        // and/or terminate the loop early (e.g. audio exhausted).
        if (!pre_hook(step, emb)) {
            std::free(emb);
            break;
        }

        float * lg = forward_fn(ctx, emb, 1, n_past, nullptr, nullptr);
        std::free(emb);
        if (!lg) break;
        n_past++;

        const int nx = argmax(lg, cfg.vocab_size);
        std::free(lg);
        gen.push_back(nx);
    }

    return gen;
}

// Convenience overload: no pre-forward hook (the common case).
template <typename Ctx, typename EmbedFn, typename ForwardFn>
inline std::vector<int32_t> run(
    Ctx        * ctx,
    int32_t      first_token,
    int          initial_n_past,
    EmbedFn      embed_fn,
    ForwardFn    forward_fn,
    const Config & cfg)
{
    return run(ctx, first_token, initial_n_past,
               embed_fn, forward_fn, NoHook{}, cfg);
}

} // namespace core_greedy_decode
