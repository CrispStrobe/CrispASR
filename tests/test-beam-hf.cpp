// test-beam-hf.cpp - core_beam_decode's HF semantics vs REAL transformers
// generate(num_beams=...), token for token.
//
// tests/fixtures/beam_hf_cases.h comes from tools/gen_beam_hf_fixtures.py: a toy
// decoder whose logits are a fixed table row picked by an integer hash of the
// sequence, run through transformers 4.57.3 beam search over beam width,
// length_penalty, early_stopping (False/True/"never"), max/min new tokens, one
// or two EOS ids and repetition_penalty (a post-log_softmax processor in beam
// search). The same toy runs here through both KV strategies - replay-from-
// prefix and branched snapshots - and must give transformers' exact sequence.
#include <catch2/catch_test_macros.hpp>

#include "core/beam_decode.h"
#include "fixtures/beam_hf_cases.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using namespace beam_hf_fixture;

uint64_t seq_hash(const std::vector<int>& ids) {
    uint64_t h = 1469598103ull;
    for (int t : ids)
        h = (h * 1000003ull + (uint64_t)t + 7ull) % 2147483648ull;
    return h;
}

float* toy_logits(const std::vector<int>& seq) {
    float* lg = (float*)std::malloc(sizeof(float) * V);
    std::memcpy(lg, TABLE[seq_hash(seq) % R], sizeof(float) * V);
    return lg;
}

// transformers RepetitionPenaltyLogitsProcessor on log-probs: once per distinct token
void rep_penalty(float* lp, const int32_t* toks, int n, const std::vector<int>& prompt, float p) {
    std::vector<int> ids(prompt.begin(), prompt.end());
    ids.insert(ids.end(), toks, toks + n);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    for (int t : ids)
        lp[t] = lp[t] < 0.0f ? lp[t] * p : lp[t] / p;
}

core_beam_decode::Config make_cfg(const Case& c) {
    core_beam_decode::Config cfg;
    cfg.semantics = core_beam_decode::Semantics::HF;
    cfg.vocab_size = V;
    cfg.beam_size = c.beams;
    cfg.max_new_tokens = c.max_new;
    cfg.min_new_tokens = c.min_new;
    cfg.eos_ids = c.eos;
    cfg.prompt_len = (int)c.prompt.size();
    cfg.length_penalty = c.lp;
    cfg.early_stopping = (core_beam_decode::EarlyStopping)c.early;
    cfg.length_offset = c.forced >= 0 ? 1 : 0; // the prompt's last token is the forced BOS HF generated
    if (c.rep != 1.0f) {
        const std::vector<int> prompt = c.prompt;
        const float p = c.rep;
        cfg.logprob_processor = [prompt, p](float* lp, const int32_t* t, int n) { rep_penalty(lp, t, n, prompt, p); };
    }
    return cfg;
}

} // namespace

TEST_CASE("HF beam semantics reproduce transformers generate(): replay strategy", "[beam]") {
    int checked = 0;
    for (const Case& c : CASES) {
        if (c.beams < 2)
            continue; // num_beams=1 is transformers' greedy path, not beam search
        const auto cfg = make_cfg(c);
        float* prefill = toy_logits(c.prompt);
        struct Ctx {
        } ctx;
        auto replay = [&](Ctx*, const int32_t* toks, int n, int) {
            std::vector<int> seq = c.prompt;
            seq.insert(seq.end(), toks, toks + n);
            return toy_logits(seq);
        };
        auto r = core_beam_decode::run_with_probs(&ctx, prefill, replay, cfg);
        std::free(prefill);
        const std::vector<int> got(r.tokens.begin(), r.tokens.end());
        INFO("beams=" << c.beams << " lp=" << c.lp << " early=" << c.early << " max_new=" << c.max_new);
        CHECK(got == c.expect);
        checked++;
    }
    REQUIRE(checked > 100);
}

TEST_CASE("HF beam semantics reproduce transformers generate(): branched strategy", "[beam]") {
    int checked = 0;
    for (const Case& c : CASES) {
        if (c.beams < 2)
            continue;
        const auto cfg = make_cfg(c);
        float* prefill = toy_logits(c.prompt);
        // "KV state" = the sequence fed so far
        struct Ctx {
            std::vector<int> seq;
        } ctx{c.prompt};
        auto save = [](Ctx* x) { return new std::vector<int>(x->seq); };
        auto restore = [](Ctx* x, std::vector<int>* s) { x->seq = *s; };
        auto snap_free = [](std::vector<int>* s) { delete s; };
        auto step = [](Ctx* x, int32_t tok, int n_past) {
            REQUIRE((int)x->seq.size() == n_past);
            x->seq.push_back(tok);
            return toy_logits(x->seq);
        };
        auto r = core_beam_decode::run_with_probs_branched(&ctx, prefill, save, restore, snap_free, step, cfg);
        std::free(prefill);
        const std::vector<int> got(r.tokens.begin(), r.tokens.end());
        INFO("beams=" << c.beams << " lp=" << c.lp << " early=" << c.early << " max_new=" << c.max_new);
        CHECK(got == c.expect);
        checked++;
    }
    REQUIRE(checked > 100);
}
