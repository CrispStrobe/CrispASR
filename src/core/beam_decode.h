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
//     WHAT THAT ASSUMPTION ACTUALLY SAYS — IT IS NARROWER THAN IT READS.
// "The audio encoder dominates" is not a claim about ASR. It is a claim about
// SMALL-DECODER ASR, and it fails from both directions:
//   * no encoder at all — m2m100 (incl. wmt21) and t5_translate/madlad are
//     text-to-text, so the quadratic decode term is the entire cost;
//   * an encoder dwarfed by its decoder — hojo-asr has a real audio encoder in
//     front of a 4.4B LM, and beam 4 over a 9 s clip costs ~248 min against
//     ~3.5 min greedy, because the decoder is where the time is.
// Before relying on this helper, ask whether the DECODER is cheap relative to
// whatever precedes it — not whether the backend has an encoder.
//
//     CONCRETELY, FOR THE TEXT-TO-TEXT CALLERS.
// m2m100 (incl. wmt21) and t5_translate/madlad have NO audio encoder to
// dominate, and their generation length is bounded by max_length (200), not by
// how long someone spoke. So the quadratic term is the whole cost there.
// Measured on m2m100-418m q8_0, 4 threads, a one-sentence input:
//
//     beam 1   8.95 s   15 tokens        (load + encode dominate)
//     beam 5  13.74 s   17 tokens        1.53x — fine
//
// ~6.7 ms per decoder forward, so the TAIL is what matters: a generation that
// reaches max_length costs beam 5 x 200^2/2 = 100,000 forwards, ~11 minutes on
// the 418M and proportionally worse on the 4.7B wmt21 checkpoints. Typical
// sentences are nowhere near that; runaway generations — exactly the failure
// #439 reported — are. Bounding max_length is therefore not only an output-
// quality fix, it is what keeps beam search affordable at all.
//
// The real fix for these two callers is kv_save/kv_restore (O(beam_size × T)),
// not lowering the beam count back to something that decodes badly.
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
#include <cstring>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace core_beam_decode {

// Search semantics.
//   Legacy: the original loop - per-beam top-B, rank by raw cumulative
//           log-prob, stop when the best beam has finished.
//   HF:     transformers GenerationMixin._beam_search (4.57): top 2B over
//           beams x vocab, only the top-B candidates may finish, finished
//           hypotheses scored sum / len^length_penalty, B unfinished beams
//           keep running, early_stopping heuristic, best finished wins.
// Default Legacy while the per-backend Kaggle A/B runs; CRISPASR_BEAM_SEMANTICS
// = hf | legacy overrides every caller (the A/B switch).
enum class Semantics { Legacy, HF };
enum class EarlyStopping { False, True, Never }; // transformers' early_stopping = False | True | "never"

struct Config {
    int max_new_tokens = 512; // hard cap on generated tokens
    int eos_id = 2;           // legacy single-EOS field; used iff eos_ids is empty
    std::vector<int> eos_ids; // multi-EOS set (e.g. glm-asr's 3 stop tokens). Any match terminates the beam.
    int vocab_size = 0;       // required
    int beam_size = 1;        // 1 = degenerate to greedy-via-beam (still works, just expensive)
    int prompt_len = 0;       // n_past after prompt prefill (replay anchor)
    // --- HF semantics only (generation_config.json of the upstream checkpoint) ---
    Semantics semantics = Semantics::Legacy;
    float length_penalty = 1.0f;                         // transformers default
    EarlyStopping early_stopping = EarlyStopping::False; // transformers default
    int min_new_tokens = 0; // EOS banned while fewer tokens were generated (MinNewTokensLengthLogitsProcessor)
    // Tokens transformers counts as GENERATED that the caller's prompt already
    // holds - e.g. m2m100's forced target-language BOS, which HF emits at step 1
    // via ForcedBOSTokenLogitsProcessor. Shifts every length in the finished
    // score and the early-stop heuristic, exactly as HF sees them.
    int length_offset = 0;
    // Logits processors that run AFTER log_softmax in beam search (e.g. the
    // repetition penalty): log_probs[vocab], the beam's generated tokens.
    std::function<void(float*, const int32_t*, int)> logprob_processor;
};

inline Semantics resolve_semantics(const Config& cfg) {
    if (const char* e = std::getenv("CRISPASR_BEAM_SEMANTICS")) {
        if (!std::strcmp(e, "hf"))
            return Semantics::HF;
        if (!std::strcmp(e, "legacy"))
            return Semantics::Legacy;
    }
    return cfg.semantics;
}

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
//
// Uses a min-heap of size K: O(V log K) instead of the previous
// O(V + V log K) partial_sort that allocated a full vocab-sized index
// vector per call (§176r). For beam_size=4 and vocab=32K-150K this
// eliminates a ~128-600 KB alloc per beam expansion step.
inline void top_k_log_softmax(const float* logits, int vocab, int K, std::vector<int>& out_ids,
                              std::vector<double>& out_lps) {
    if (K > vocab)
        K = vocab;
    out_ids.resize((size_t)K);
    out_lps.resize((size_t)K);

    const double logZ = compute_logZ(logits, vocab);

    // Min-heap: stores (logit, token_id) pairs; the smallest logit is
    // at the top so we can cheaply evict it when a larger one arrives.
    // Reused across calls (thread-local) so the K-element heap isn't
    // re-allocated on every beam-expansion step (B×T per transcribe). clear()
    // keeps capacity; the heap is rebuilt from scratch below, so the result is
    // bit-identical to a fresh vector. (§176r follow-up)
    using Pair = std::pair<float, int>;
    static thread_local std::vector<Pair> heap;
    heap.clear();
    heap.reserve((size_t)K);
    for (int i = 0; i < vocab; i++) {
        if ((int)heap.size() < K) {
            heap.push_back({logits[i], i});
            if ((int)heap.size() == K)
                std::make_heap(heap.begin(), heap.end(),
                               [](const Pair& a, const Pair& b) { return a.first > b.first; });
        } else if (logits[i] > heap[0].first) {
            std::pop_heap(heap.begin(), heap.end(), [](const Pair& a, const Pair& b) { return a.first > b.first; });
            heap.back() = {logits[i], i};
            std::push_heap(heap.begin(), heap.end(), [](const Pair& a, const Pair& b) { return a.first > b.first; });
        }
    }
    // Sort descending by logit.
    std::sort(heap.begin(), heap.end(), [](const Pair& a, const Pair& b) { return a.first > b.first; });
    for (int i = 0; i < K; i++) {
        out_ids[(size_t)i] = heap[(size_t)i].second;
        out_lps[(size_t)i] = (double)heap[(size_t)i].first - logZ;
    }
}

// transformers 4.57 GenerationMixin._beam_search for batch 1 (see Semantics).
// Generic over the KV strategy: `expand(beam)` returns the malloc'd next-token
// logits of a running beam (nullptr = drop it) and may set beam.child, the
// per-beam state its children inherit as their `state`. The seed beam gets
// `seed_child` and `prefill_logits` without an expand call.
template <typename State> struct HfBeam {
    std::vector<int32_t> tokens;
    std::vector<float> probs; // softmax prob of each token (unprocessed), for the callers' callbacks
    float score = 0.0f;
    State state{}; // e.g. the KV snapshot right BEFORE feeding tokens.back()
    State child{}; // the state after feeding it: inherited by this beam's children
};

template <typename State, typename ExpandFn>
inline Result hf_search(const float* prefill_logits, const Config& cfg, State seed_child, ExpandFn expand) {
    Result result;
    const int B = std::max(1, cfg.beam_size), V = cfg.vocab_size;
    std::vector<int> eos = cfg.eos_ids;
    if (eos.empty())
        eos.push_back(cfg.eos_id);
    auto is_eos = [&](int id) { return std::find(eos.begin(), eos.end(), id) != eos.end(); };
    const int K = std::max(2, 1 + (int)eos.size()) * B;
    const float NEG = -1.0e9f;
    auto lp_div = [&](int len) { return (float)std::pow((double)len, (double)cfg.length_penalty); };
    const int max_len = cfg.max_new_tokens;

    std::vector<HfBeam<State>> running(1);
    running[0].child = seed_child;
    std::vector<std::vector<float>> logits(1, std::vector<float>(prefill_logits, prefill_logits + V));
    std::vector<HfBeam<State>> fin(B); // HF `sequences` / `beam_scores`: score -1e9, not finished
    for (auto& f : fin)
        f.score = NEG;
    std::vector<bool> fin_done(B, false);
    bool heur_unsat = true;
    struct Cand {
        float s;
        int beam, tok;
        float prob;
    };
    for (int cur = 0; cur < max_len; cur++) {
        // 1. log_softmax, processors, + running score; top-K over beams x vocab
        std::vector<Cand> top;
        top.reserve((size_t)K + 1);
        std::vector<float> lsm;
        for (size_t b = 0; b < running.size(); b++) {
            if (logits[b].empty())
                continue; // dropped / at -1e9: can never reach the top K (see below)
            const std::vector<float>& lg = logits[b];
            const double logZ = compute_logZ(lg.data(), V);
            lsm.resize((size_t)V);
            for (int v = 0; v < V; v++)
                lsm[(size_t)v] = (float)((double)lg[(size_t)v] - logZ);
            std::vector<float> raw = lsm; // for the per-token probs
            if (cur < cfg.min_new_tokens)
                for (int e : eos)
                    if (e >= 0 && e < V)
                        lsm[(size_t)e] = -INFINITY;
            if (cfg.logprob_processor)
                cfg.logprob_processor(lsm.data(), running[b].tokens.data(), (int)running[b].tokens.size());
            const float base = running[b].score;
            for (int v = 0; v < V; v++) {
                const float sc = lsm[(size_t)v] + base;
                if ((int)top.size() == K && !(sc > top.back().s))
                    continue; // equal scores keep the lower flat index first
                Cand c{sc, (int)b, v, std::exp(raw[(size_t)v])};
                auto it =
                    std::upper_bound(top.begin(), top.end(), c, [](const Cand& a, const Cand& x) { return a.s > x.s; });
                top.insert(it, c);
                if ((int)top.size() > K)
                    top.pop_back();
            }
        }
        const int nk = (int)top.size();
        if (nk == 0)
            break;
        std::vector<char> hits((size_t)nk);
        bool all_hit = true;
        for (int k = 0; k < nk; k++) {
            hits[(size_t)k] = is_eos(top[(size_t)k].tok) || cur + 1 >= max_len;
            all_hit = all_hit && hits[(size_t)k];
        }
        auto make = [&](const Cand& c) {
            HfBeam<State> h;
            const HfBeam<State>& par = running[(size_t)c.beam];
            h.tokens = par.tokens;
            h.tokens.push_back(c.tok);
            h.probs = par.probs;
            h.probs.push_back(c.prob);
            h.state = par.child;
            return h;
        };
        // 2. finished: only the top-B candidates may finish
        {
            const bool full = cfg.early_stopping == EarlyStopping::True &&
                              std::all_of(fin_done.begin(), fin_done.end(), [](bool d) { return d; });
            std::vector<std::pair<float, int>> merged; // (score, index): < B = old finished, >= B = candidate
            for (int i = 0; i < B; i++)
                merged.push_back({fin[(size_t)i].score, i});
            for (int k = 0; k < nk; k++) {
                float sc = top[(size_t)k].s / lp_div(cur + 1 + cfg.length_offset);
                if (full)
                    sc += NEG;
                if (!heur_unsat)
                    sc += NEG;
                if (!(k < B && hits[(size_t)k]))
                    sc += NEG;
                merged.push_back({sc, B + k});
            }
            std::stable_sort(
                merged.begin(), merged.end(),
                [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });
            std::vector<HfBeam<State>> nf(B);
            std::vector<bool> nd(B);
            for (int i = 0; i < B; i++) {
                const int idx = merged[(size_t)i].second;
                if (idx < B) {
                    nf[(size_t)i] = fin[(size_t)idx];
                    nd[(size_t)i] = fin_done[(size_t)idx];
                } else {
                    const int k = idx - B;
                    nf[(size_t)i] = make(top[(size_t)k]);
                    nd[(size_t)i] = k < B && hits[(size_t)k];
                }
                nf[(size_t)i].score = merged[(size_t)i].first;
            }
            fin = std::move(nf);
            fin_done = std::move(nd);
        }
        // 3. running: the best B candidates that did not hit
        std::vector<int> order((size_t)nk);
        for (int k = 0; k < nk; k++)
            order[(size_t)k] = k;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return top[(size_t)a].s + (hits[(size_t)a] ? NEG : 0.0f) >
                   top[(size_t)b].s + (hits[(size_t)b] ? NEG : 0.0f);
        });
        std::vector<HfBeam<State>> next;
        for (int i = 0; i < std::min(B, nk); i++) {
            const int k = order[(size_t)i];
            HfBeam<State> h = make(top[(size_t)k]);
            h.score = top[(size_t)k].s + (hits[(size_t)k] ? NEG : 0.0f);
            next.push_back(std::move(h));
        }
        running = std::move(next);
        // 4. early-stop heuristic (sticky) and the loop conditions
        const int cur_len = cur + 1 + cfg.length_offset;
        const int best_len = (cfg.early_stopping == EarlyStopping::Never && cfg.length_penalty > 0.0f)
                                 ? max_len + cfg.length_offset
                                 : cur_len;
        const float best_running = running.empty() ? NEG : running[0].score / lp_div(best_len);
        float worst = fin[0].score;
        for (const auto& f : fin)
            worst = std::min(worst, f.score);
        bool any = false;
        for (int i = 0; i < B; i++)
            any = any || best_running > (fin_done[(size_t)i] ? worst : NEG);
        heur_unsat = heur_unsat && any;
        const bool open = !(cfg.early_stopping == EarlyStopping::True &&
                            std::all_of(fin_done.begin(), fin_done.end(), [](bool d) { return d; }));
        if (!heur_unsat || !open || all_hit)
            break;
        // 5. next-token logits for every running beam
        logits.assign(running.size(), {});
        for (size_t b = 0; b < running.size(); b++) {
            if (running[b].score <= 0.5f * NEG)
                continue; // a hit beam kept only to fill B: never selectable again
            float* lg = expand(running[b]);
            if (lg) {
                logits[b].assign(lg, lg + V);
                std::free(lg);
            }
        }
    }
    const HfBeam<State>& best = fin[0];
    result.tokens = best.tokens;
    result.probs = best.probs;
    return result;
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

    if (resolve_semantics(cfg) == Semantics::HF) {
        struct NoState {};
        return detail::hf_search<NoState>(prefill_logits, cfg, NoState{}, [&](detail::HfBeam<NoState>& b) -> float* {
            return replay_fn(ctx, b.tokens.data(), (int)b.tokens.size(), cfg.prompt_len);
        });
    }

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

    if (resolve_semantics(cfg) == Semantics::HF) {
        // state = KV snapshot right before feeding tokens.back(); a beam's
        // expand restores it, steps that token and snapshots the result for
        // its children. The seed's children start from the prompt snapshot.
        using SP = std::shared_ptr<Holder>;
        SP prompt = wrap(save_fn(ctx));
        return detail::hf_search<SP>(prefill_logits, cfg, prompt, [&](detail::HfBeam<SP>& b) -> float* {
            restore_fn(ctx, b.state->snap);
            float* lg = step_fn(ctx, b.tokens.back(), cfg.prompt_len + (int)b.tokens.size() - 1);
            if (lg)
                b.child = wrap(save_fn(ctx));
            return lg;
        });
    }

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
