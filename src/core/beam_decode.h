// src/core/beam_decode.h — shared autoregressive beam-search decode loop.
//
// Companion to `core_greedy_decode::run_with_probs`. Same callback shape
// at the call site (a single `replay_fn`), but instead of picking one
// argmax / sampled token per step, this helper expands the top
// `beam_size` hypotheses in parallel and prunes globally to keep the
// highest-cumulative-logprob beams alive. Returns the winning beam's
// tokens + per-token softmax probabilities.
//
// KV strategy — replay-from-prefix
// --------------------------------
// Beam search needs each beam to have its own KV state, but the LLM-style
// runtime backends (glm-asr, kyutai-stt, omniasr-llm, moonshine via
// decode step) keep KV in a single context-owned buffer. To avoid adding
// a per-backend `kv_save` / `kv_restore` API, we exploit the fact that
// every call to "advance the LM by these tokens at this n_past" lets us
// write KV slots at any logical position. Each step rebuilds each beam's
// KV by replaying its full generated suffix from the post-prompt anchor.
//
// Cost: O(beam_size × T²/2) extra forward work for T generated tokens vs
// greedy's O(T). Acceptable for beam_size = 2-4 since the audio encoder
// typically dominates wall time on these backends. If perf is bad on
// long generations, the right next step is `*_kv_save` / `*_kv_restore`
// per backend, not making this helper smarter.
//
// Caller contract
// ---------------
// Caller is responsible for:
//   * Running the prompt prefill so KV slots [0, prompt_len) hold the
//     prompt's K/V.
//   * Capturing the prefill logits at the last prompt position.
//   * Providing a `replay_fn(ctx, tokens, n_tokens, prompt_len)` that
//     overwrites KV slots [prompt_len, prompt_len + n_tokens) with the
//     given suffix's K/V and returns the last-position logits as a
//     malloc'd `float*` of size `vocab_size` (or nullptr on failure).
//     The helper free()s the returned buffer.
//   * Resetting KV after `run_with_probs` returns (state is undefined
//     since each beam-step overwrites slots).
//
// For backends that natively expose a batched
// `forward(ctx, embeds, n_tokens, n_past)` — like glm-asr's
// `glm_asr_run_llm_kv` — the replay_fn is a one-liner that embeds +
// forwards. For backends with a per-token API (like omniasr-llm's
// `omniasr_run_dec_token`), the replay_fn loops over tokens internally.
//
// Usage (glm-asr):
//
//     core_beam_decode::Config cfg;
//     cfg.max_new_tokens = 512;
//     cfg.eos_id         = hp.eos_token_ids[0];
//     cfg.vocab_size     = hp.llm_vocab;
//     cfg.beam_size      = params.beam_size;
//     cfg.prompt_len     = (int)prompt_ids.size();
//
//     auto replay = [ctx](const int32_t* toks, int n, int prompt_len) -> float* {
//         float* emb = glm_asr_embed_tokens(ctx, toks, n);
//         if (!emb) return nullptr;
//         float* lg = glm_asr_run_llm_kv(ctx, emb, n, prompt_len, nullptr, nullptr);
//         std::free(emb);
//         return lg;
//     };
//
//     auto r = core_beam_decode::run_with_probs(ctx, prefill_lg, replay, cfg);
//
// Header-only so the compiler inlines each caller's concrete callable
// at the call site (matching the greedy helper's pattern).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace core_beam_decode {

struct Config {
    int max_new_tokens = 512; // hard cap on generated tokens
    int eos_id = 2;           // legacy single-EOS field; used iff eos_ids is empty
    std::vector<int> eos_ids; // multi-EOS set (e.g. glm-asr's 3 stop tokens). Any match terminates the beam.
    int vocab_size = 0;       // required
    int beam_size = 1;        // 1 = degenerate to greedy-via-beam (still works, just expensive)
    int prompt_len = 0;       // n_past after prompt prefill (replay anchor)
};

struct Result {
    std::vector<int32_t> tokens; // generated tokens of the winning beam
    std::vector<float> probs;    // softmax prob of each picked token in [0,1]
};

namespace detail {

// Internal beam state. Tokens are the generated suffix only (post-prompt).
struct Beam {
    std::vector<int32_t> tokens;
    std::vector<float> probs;
    double cum_logprob = 0.0;
    bool finished = false;
};

// Numerically-stable log-Z (log normaliser of softmax over `logits`).
inline double compute_logZ(const float* logits, int vocab) {
    float mx = logits[0];
    for (int k = 1; k < vocab; k++)
        if (logits[k] > mx)
            mx = logits[k];
    double sum = 0.0;
    for (int k = 0; k < vocab; k++)
        sum += std::exp((double)(logits[k] - mx));
    return (double)mx + std::log(sum);
}

// Pick top-K tokens from `logits` and return their log-softmax values.
// `out_ids` and `out_lps` are sized to K, sorted descending by logit.
inline void top_k_log_softmax(const float* logits, int vocab, int K, std::vector<int>& out_ids,
                              std::vector<double>& out_lps) {
    if (K > vocab)
        K = vocab;
    out_ids.resize((size_t)K);
    out_lps.resize((size_t)K);

    const double logZ = compute_logZ(logits, vocab);

    std::vector<int> idx((size_t)vocab);
    for (int i = 0; i < vocab; i++)
        idx[(size_t)i] = i;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                      [logits](int a, int b) { return logits[a] > logits[b]; });
    for (int i = 0; i < K; i++) {
        out_ids[(size_t)i] = idx[(size_t)i];
        out_lps[(size_t)i] = (double)logits[idx[(size_t)i]] - logZ;
    }
}

} // namespace detail

// Run the beam decode loop.
//
// Template parameters:
//   Ctx      : the model's opaque context type.
//   ReplayFn : callable with signature
//                `float * (Ctx*, const int32_t* tokens, int n_tokens, int prompt_len)`.
//              Must overwrite KV slots [prompt_len, prompt_len + n_tokens)
//              and return a malloc'd `float*` containing the last-position
//              logits ([vocab_size, 1]). Returning nullptr terminates that
//              beam.
//
// Pre-condition: caller has populated KV slots [0, prompt_len) with the
// prompt's K/V; `prefill_logits` is the [V, 1] logits at the last prompt
// position (caller-owned, not freed by this helper).
//
// Post-condition: KV is left in an undefined state — caller must reset
// it before the next transcription.
template <typename Ctx, typename ReplayFn>
inline Result run_with_probs(Ctx* ctx, const float* prefill_logits, ReplayFn replay_fn, const Config& cfg) {
    using detail::Beam;

    Result result;
    if (!prefill_logits || cfg.vocab_size <= 0)
        return result;

    const int B = (cfg.beam_size > 0) ? cfg.beam_size : 1;
    const int V = cfg.vocab_size;

    auto is_eos = [&](int id) {
        if (!cfg.eos_ids.empty()) {
            for (int e : cfg.eos_ids)
                if (id == e)
                    return true;
            return false;
        }
        return id == cfg.eos_id;
    };

    // 1. Initial beams: top-B from prefill logits.
    std::vector<int> first_ids;
    std::vector<double> first_lps;
    detail::top_k_log_softmax(prefill_logits, V, B, first_ids, first_lps);

    std::vector<Beam> beams((size_t)first_ids.size());
    for (size_t i = 0; i < first_ids.size(); i++) {
        beams[i].tokens.push_back(first_ids[i]);
        beams[i].probs.push_back((float)std::exp(first_lps[i]));
        beams[i].cum_logprob = first_lps[i];
        if (is_eos(first_ids[i]))
            beams[i].finished = true;
    }
    // top_k_log_softmax already returns descending; beams are sorted.

    if (beams.empty())
        return result;

    // 2. Per-step expand-and-prune.
    while ((int)beams[0].tokens.size() < cfg.max_new_tokens && !beams[0].finished) {
        struct Cand {
            int beam_idx;
            int token;
            double cum_logprob;
            float token_prob;
            bool from_finished; // carry-forward of an already-finished beam
        };
        std::vector<Cand> cands;
        cands.reserve((size_t)B * (size_t)B + (size_t)B);

        for (size_t bi = 0; bi < beams.size(); bi++) {
            auto& b = beams[bi];
            if (b.finished) {
                // Carry forward; the token field is only used by the
                // detokeniser, which skips finished beams.
                const int sentinel = cfg.eos_ids.empty() ? cfg.eos_id : cfg.eos_ids[0];
                cands.push_back({(int)bi, sentinel, b.cum_logprob, 1.0f, true});
                continue;
            }

            float* lg = replay_fn(ctx, b.tokens.data(), (int)b.tokens.size(), cfg.prompt_len);
            if (!lg) {
                b.finished = true;
                continue;
            }

            std::vector<int> ids;
            std::vector<double> lps;
            detail::top_k_log_softmax(lg, V, B, ids, lps);
            std::free(lg);

            for (size_t j = 0; j < ids.size(); j++) {
                Cand c;
                c.beam_idx = (int)bi;
                c.token = ids[j];
                c.cum_logprob = b.cum_logprob + lps[j];
                c.token_prob = (float)std::exp(lps[j]);
                c.from_finished = false;
                cands.push_back(c);
            }
        }

        if (cands.empty())
            break;

        const size_t keep = std::min<size_t>((size_t)B, cands.size());
        std::partial_sort(cands.begin(), cands.begin() + keep, cands.end(),
                          [](const Cand& a, const Cand& b) { return a.cum_logprob > b.cum_logprob; });
        cands.resize(keep);

        std::vector<Beam> next_beams;
        next_beams.reserve(keep);
        for (auto& c : cands) {
            Beam nb = beams[(size_t)c.beam_idx]; // copy parent
            if (c.from_finished) {
                next_beams.push_back(std::move(nb));
                continue;
            }
            nb.tokens.push_back(c.token);
            nb.probs.push_back(c.token_prob);
            nb.cum_logprob = c.cum_logprob;
            if (is_eos(c.token))
                nb.finished = true;
            next_beams.push_back(std::move(nb));
        }
        beams = std::move(next_beams);
    }

    // 3. Best beam is beams[0] (top-of-cands ordering preserved by partial_sort).
    result.tokens = std::move(beams[0].tokens);
    result.probs = std::move(beams[0].probs);
    return result;
}

// ---------------------------------------------------------------------------
// Branched variant: per-beam KV snapshots (O(B × T) single-token forwards).
// ---------------------------------------------------------------------------
//
// Use this when the backend's per-step decode is one-token-at-a-time and the
// caller can cheaply snapshot/restore the model's KV state. Gives true
// O(B × T) single-token forwards instead of replay-from-prefix's O(B × T²).
//
// Caller provides four callbacks plus the post-prefill prompt logits:
//   save_fn(ctx)            -> Snap   (snapshot of current KV state)
//   restore_fn(ctx, snap)   -> void   (writes snap back into ctx KV; snap
//                                       remains valid for further restores)
//   snap_free_fn(snap)      -> void   (called once when the snap is dead)
//   step_fn(ctx, tok, n_past)-> float* (writes KV slot at n_past for `tok`,
//                                       returns malloc'd [vocab_size] logits;
//                                       helper free()s it. nullptr on failure.)
//
// The Snap type is whatever save_fn returns — typically a heap pointer to a
// caller-defined struct. The helper wraps it in a refcounted holder so
// siblings can share a parent's snapshot without double-free. Restore is
// read-only on the snap (it copies snap into ctx); only step_fn / save_fn
// mutate ctx KV.
//
// Pre-condition: KV slots [0, prompt_len) hold the prompt's K/V; the helper
// snapshots that state once at entry. `prefill_logits` is the [V, 1] logits
// at the last prompt position (caller-owned, not freed).
template <typename Ctx, typename SaveFn, typename RestoreFn, typename SnapFreeFn, typename StepFn>
inline Result run_with_probs_branched(Ctx* ctx, const float* prefill_logits, SaveFn save_fn, RestoreFn restore_fn,
                                      SnapFreeFn snap_free_fn, StepFn step_fn, const Config& cfg) {
    using Snap = decltype(save_fn(ctx));

    Result result;
    if (!prefill_logits || cfg.vocab_size <= 0)
        return result;

    const int B = (cfg.beam_size > 0) ? cfg.beam_size : 1;
    const int V = cfg.vocab_size;

    auto is_eos = [&](int id) {
        if (!cfg.eos_ids.empty()) {
            for (int e : cfg.eos_ids)
                if (id == e)
                    return true;
            return false;
        }
        return id == cfg.eos_id;
    };

    // RAII wrapper so siblings can share a parent's snap by shared_ptr.
    // The destructor calls snap_free_fn exactly once when the last ref dies.
    struct Holder {
        Snap snap;
        SnapFreeFn* free_cb;
        Holder(Snap s, SnapFreeFn* f) : snap(s), free_cb(f) {}
        ~Holder() { (*free_cb)(snap); }
        Holder(const Holder&) = delete;
        Holder& operator=(const Holder&) = delete;
    };
    auto wrap = [&snap_free_fn](Snap s) { return std::shared_ptr<Holder>(new Holder(s, &snap_free_fn)); };

    struct BeamS {
        std::vector<int32_t> tokens;
        std::vector<float> probs;
        double cum_logprob = 0.0;
        bool finished = false;
        // KV state right BEFORE feeding tokens.back() to step_fn. All initial
        // beams share the prompt snap; siblings share their parent's
        // post-step snap. Restored at the start of each per-beam expand.
        std::shared_ptr<Holder> snap;
    };

    // 1. Snapshot the post-prefill prompt KV; seed initial beams from
    // top-K of prefill_logits. No step_fn calls happen during seeding —
    // the first round of the loop below will do that for each beam.
    auto prompt_snap = wrap(save_fn(ctx));

    std::vector<int> first_ids;
    std::vector<double> first_lps;
    detail::top_k_log_softmax(prefill_logits, V, B, first_ids, first_lps);

    std::vector<BeamS> beams((size_t)first_ids.size());
    for (size_t i = 0; i < first_ids.size(); i++) {
        beams[i].tokens.push_back(first_ids[i]);
        beams[i].probs.push_back((float)std::exp(first_lps[i]));
        beams[i].cum_logprob = first_lps[i];
        if (is_eos(first_ids[i]))
            beams[i].finished = true;
        beams[i].snap = prompt_snap;
    }

    if (beams.empty())
        return result;

    // 2. Per-round expand + prune.
    //
    // For each unfinished beam: restore beam.snap (which is the KV state
    // RIGHT BEFORE feeding beam.tokens.back()), call step_fn for that token
    // — which writes its KV slot and returns logits for the NEXT token —
    // then capture a fresh snap (state right AFTER that token). Take top-K
    // from the logits as expansion candidates; siblings inherit the same
    // post-step snap shared_ptr.
    while ((int)beams[0].tokens.size() < cfg.max_new_tokens && !beams[0].finished) {
        struct Cand {
            int beam_idx;
            int token;
            double cum_logprob;
            float token_prob;
            bool from_finished;
        };
        std::vector<Cand> cands;
        cands.reserve((size_t)B * (size_t)B + (size_t)B);

        // Per-beam post-step snap — shared by all surviving children.
        std::vector<std::shared_ptr<Holder>> step_snaps(beams.size());

        for (size_t bi = 0; bi < beams.size(); bi++) {
            auto& b = beams[bi];
            if (b.finished) {
                const int sentinel = cfg.eos_ids.empty() ? cfg.eos_id : cfg.eos_ids[0];
                cands.push_back({(int)bi, sentinel, b.cum_logprob, 1.0f, true});
                continue;
            }
            // n_past = number of tokens preceding the one we're feeding.
            // Beam currently holds (b.tokens.size()) chosen tokens; the
            // first (b.tokens.size() - 1) of them have already been stepped
            // through the LM in prior rounds. Now we step the most recent
            // one at slot prompt_len + (b.tokens.size() - 1).
            const int n_past = cfg.prompt_len + (int)b.tokens.size() - 1;
            restore_fn(ctx, b.snap->snap);
            float* lg = step_fn(ctx, b.tokens.back(), n_past);
            if (!lg) {
                b.finished = true;
                continue;
            }
            step_snaps[bi] = wrap(save_fn(ctx));

            std::vector<int> ids;
            std::vector<double> lps;
            detail::top_k_log_softmax(lg, V, B, ids, lps);
            std::free(lg);

            for (size_t j = 0; j < ids.size(); j++) {
                cands.push_back({(int)bi, ids[j], b.cum_logprob + lps[j], (float)std::exp(lps[j]), false});
            }
        }

        if (cands.empty())
            break;

        const size_t keep = std::min<size_t>((size_t)B, cands.size());
        std::partial_sort(cands.begin(), cands.begin() + keep, cands.end(),
                          [](const Cand& a, const Cand& b) { return a.cum_logprob > b.cum_logprob; });
        cands.resize(keep);

        std::vector<BeamS> next_beams;
        next_beams.reserve(keep);
        for (auto& c : cands) {
            const auto& parent = beams[(size_t)c.beam_idx];
            BeamS nb;
            nb.tokens = parent.tokens;
            nb.probs = parent.probs;
            nb.cum_logprob = parent.cum_logprob;
            nb.finished = parent.finished;
            if (c.from_finished) {
                // Carry forward unchanged; same snap stays valid since no
                // step_fn ran for this beam this round.
                nb.snap = parent.snap;
                next_beams.push_back(std::move(nb));
                continue;
            }
            nb.tokens.push_back(c.token);
            nb.probs.push_back(c.token_prob);
            nb.cum_logprob = c.cum_logprob;
            if (is_eos(c.token))
                nb.finished = true;
            nb.snap = step_snaps[(size_t)c.beam_idx];
            next_beams.push_back(std::move(nb));
        }

        beams = std::move(next_beams);
        // Old `beams` shared_ptrs and unselected `step_snaps` go out of scope
        // here — Holder destructors fire snap_free_fn exactly once each.
    }

    result.tokens = std::move(beams[0].tokens);
    result.probs = std::move(beams[0].probs);
    return result;
}

} // namespace core_beam_decode
