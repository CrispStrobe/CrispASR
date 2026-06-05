// test-core-decode.cpp — unit tests for core_greedy_decode + core_beam_decode.
//
// These test the shared decode helpers in isolation using a trivial mock
// "LLM" that always returns a fixed logit distribution. No models loaded,
// no GPU needed — pure CPU, sub-millisecond.

#include <catch2/catch_test_macros.hpp>

#include "core/greedy_decode.h"
#include "core/beam_decode.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Mock LLM: vocab_size=8, always returns logits where token 3 is highest,
// token 5 is EOS. After generating 3 tokens of id=3, switches to EOS.
// ---------------------------------------------------------------------------

struct MockCtx {
    int call_count = 0;
    int vocab = 8;
    int eos = 5;
    int hot_tok = 3;
};

// Greedy-decode embed callback (unused — greedy helper doesn't call it)
static float* mock_embed(MockCtx* ctx, const int32_t* ids, int n) {
    (void)ctx;
    (void)ids;
    float* e = (float*)malloc(sizeof(float) * n);
    memset(e, 0, sizeof(float) * n);
    return e;
}

// Greedy-decode LLM callback: returns logits with hot_tok dominant,
// switches to EOS after 3 calls.
static float* mock_llm(MockCtx* ctx, const float* embeds, int n_tokens, int n_past, int* out_n, int* out_vocab) {
    (void)embeds;
    (void)n_tokens;
    (void)n_past;
    ctx->call_count++;
    if (out_n)
        *out_n = 1;
    if (out_vocab)
        *out_vocab = ctx->vocab;
    float* logits = (float*)malloc(sizeof(float) * ctx->vocab);
    for (int i = 0; i < ctx->vocab; i++)
        logits[i] = -10.0f;
    if (ctx->call_count <= 3)
        logits[ctx->hot_tok] = 5.0f;
    else
        logits[ctx->eos] = 5.0f;
    return logits;
}

// Beam-decode replay callback (takes ctx as first arg from template)
static float* mock_replay(MockCtx* ctx, const int32_t* toks, int n_toks, int prompt_len) {
    (void)toks;
    (void)prompt_len;
    ctx->call_count++;
    float* logits = (float*)malloc(sizeof(float) * ctx->vocab);
    for (int i = 0; i < ctx->vocab; i++)
        logits[i] = -10.0f;
    // Generate hot_tok for first 3 tokens, then EOS
    if (n_toks < 3)
        logits[ctx->hot_tok] = 5.0f;
    else
        logits[ctx->eos] = 5.0f;
    return logits;
}

// ---------------------------------------------------------------------------
// core_greedy_decode tests
// ---------------------------------------------------------------------------

TEST_CASE("greedy_decode: argmax picks highest logit", "[unit][decode]") {
    float logits[] = {1.0f, 3.0f, 2.0f, 5.0f, 0.5f};
    REQUIRE(core_greedy_decode::argmax(logits, 5) == 3);
}

TEST_CASE("greedy_decode: argmax with negative logits", "[unit][decode]") {
    float logits[] = {-5.0f, -1.0f, -3.0f, -2.0f};
    REQUIRE(core_greedy_decode::argmax(logits, 4) == 1);
}

TEST_CASE("greedy_decode: softmax_of returns valid probability", "[unit][decode]") {
    float logits[] = {1.0f, 2.0f, 3.0f};
    float p = core_greedy_decode::softmax_of(logits, 3, 2, logits[2]);
    REQUIRE(p > 0.5f);
    REQUIRE(p <= 1.0f);
}

TEST_CASE("greedy_decode: run_with_probs produces correct sequence", "[unit][decode]") {
    MockCtx ctx;
    core_greedy_decode::Config cfg;
    cfg.max_new_tokens = 10;
    cfg.eos_id = 5;
    cfg.vocab_size = 8;

    // Simulate prefill: first token is hot_tok=3
    float prefill_logits[8];
    for (int i = 0; i < 8; i++)
        prefill_logits[i] = -10.0f;
    prefill_logits[3] = 5.0f;

    int first_tok = core_greedy_decode::argmax(prefill_logits, 8);
    float first_p = core_greedy_decode::softmax_of(prefill_logits, 8, first_tok, prefill_logits[first_tok]);

    auto result = core_greedy_decode::run_with_probs(&ctx, first_tok, first_p, 0, mock_embed, mock_llm, cfg);

    // Should produce [3, 3, 3, 5] (3 hot tokens + EOS)
    REQUIRE(result.tokens.size() >= 3);
    REQUIRE(result.tokens[0] == 3);
    REQUIRE(result.tokens[1] == 3);
    REQUIRE(result.tokens[2] == 3);
    // EOS should terminate
    bool found_eos = false;
    for (auto t : result.tokens)
        if (t == 5)
            found_eos = true;
    REQUIRE(found_eos);
}

static float* mock_llm_repeat_pair(MockCtx* ctx, const float* embeds, int n_tokens, int n_past, int* out_n,
                                   int* out_vocab) {
    (void)embeds;
    (void)n_tokens;
    (void)n_past;
    ctx->call_count++;
    if (out_n)
        *out_n = 1;
    if (out_vocab)
        *out_vocab = ctx->vocab;
    float* logits = (float*)malloc(sizeof(float) * ctx->vocab);
    for (int i = 0; i < ctx->vocab; i++)
        logits[i] = -10.0f;
    logits[3] = 5.0f;
    logits[4] = 4.0f;
    return logits;
}

TEST_CASE("greedy_decode: frequency_penalty penalizes generated token ids", "[unit][decode]") {
    MockCtx ctx;
    core_greedy_decode::Config cfg;
    cfg.max_new_tokens = 2;
    cfg.eos_id = 5;
    cfg.vocab_size = 8;
    cfg.frequency_penalty = 2.0f;

    auto result = core_greedy_decode::run_with_probs(&ctx, 3, 1.0f, 0, mock_embed, mock_llm_repeat_pair, cfg);

    REQUIRE(result.tokens.size() == 2);
    REQUIRE(result.tokens[0] == 3);
    REQUIRE(result.tokens[1] == 4);
}

TEST_CASE("greedy_decode: sampling seed is reproducible", "[unit][decode]") {
    float logits[] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f};

    std::mt19937_64 rng_a(123);
    std::mt19937_64 rng_b(123);
    std::mt19937_64 rng_c(456);

    std::vector<int> a;
    std::vector<int> b;
    std::vector<int> c;
    for (int i = 0; i < 32; ++i) {
        a.push_back(core_greedy_decode::sample_temp(logits, 5, 1.0f, rng_a));
        b.push_back(core_greedy_decode::sample_temp(logits, 5, 1.0f, rng_b));
        c.push_back(core_greedy_decode::sample_temp(logits, 5, 1.0f, rng_c));
    }

    REQUIRE(a == b);
    REQUIRE(a != c);
}

// ---------------------------------------------------------------------------
// core_beam_decode tests
// ---------------------------------------------------------------------------

TEST_CASE("beam_decode: beam_size=1 behaves like greedy", "[unit][decode]") {
    MockCtx ctx;
    float prefill[8];
    for (int i = 0; i < 8; i++)
        prefill[i] = -10.0f;
    prefill[3] = 5.0f;

    core_beam_decode::Config cfg;
    cfg.max_new_tokens = 10;
    cfg.eos_id = 5;
    cfg.vocab_size = 8;
    cfg.beam_size = 1;
    cfg.prompt_len = 0;

    auto result = core_beam_decode::run_with_probs(&ctx, prefill, mock_replay, cfg);

    REQUIRE(result.tokens.size() >= 3);
    REQUIRE(result.tokens[0] == 3);
    REQUIRE(result.tokens[1] == 3);
    REQUIRE(result.tokens[2] == 3);
}

TEST_CASE("beam_decode: beam_size=4 produces valid output", "[unit][decode]") {
    MockCtx ctx;
    float prefill[8];
    for (int i = 0; i < 8; i++)
        prefill[i] = -10.0f;
    prefill[3] = 5.0f;

    core_beam_decode::Config cfg;
    cfg.max_new_tokens = 10;
    cfg.eos_id = 5;
    cfg.vocab_size = 8;
    cfg.beam_size = 4;
    cfg.prompt_len = 0;

    auto result = core_beam_decode::run_with_probs(&ctx, prefill, mock_replay, cfg);

    // With a strongly peaked distribution, beam search should agree with greedy
    REQUIRE(!result.tokens.empty());
    REQUIRE(result.tokens[0] == 3);
    REQUIRE(result.probs.size() == result.tokens.size());
    for (float p : result.probs) {
        REQUIRE(p >= 0.0f);
        REQUIRE(p <= 1.0f);
    }
}

TEST_CASE("beam_decode: respects max_new_tokens", "[unit][decode]") {
    // Mock that never produces EOS
    struct InfCtx {
        int vocab = 8;
    };
    InfCtx ictx;

    auto inf_replay = [](InfCtx* c, const int32_t*, int, int) -> float* {
        float* lg = (float*)malloc(sizeof(float) * c->vocab);
        for (int i = 0; i < c->vocab; i++)
            lg[i] = -10.0f;
        lg[3] = 5.0f; // always token 3, never EOS
        return lg;
    };

    float prefill[8];
    for (int i = 0; i < 8; i++)
        prefill[i] = -10.0f;
    prefill[3] = 5.0f;

    core_beam_decode::Config cfg;
    cfg.max_new_tokens = 5;
    cfg.eos_id = 7; // never generated
    cfg.vocab_size = 8;
    cfg.beam_size = 2;
    cfg.prompt_len = 0;

    auto result = core_beam_decode::run_with_probs(&ictx, prefill, inf_replay, cfg);

    REQUIRE(result.tokens.size() == 5);
}

TEST_CASE("beam_decode: multi-EOS stops on any", "[unit][decode]") {
    MockCtx ctx;
    float prefill[8];
    for (int i = 0; i < 8; i++)
        prefill[i] = -10.0f;
    prefill[3] = 5.0f;

    core_beam_decode::Config cfg;
    cfg.max_new_tokens = 10;
    cfg.eos_ids = {5, 6, 7}; // any of these is EOS
    cfg.vocab_size = 8;
    cfg.beam_size = 2;
    cfg.prompt_len = 0;

    auto result = core_beam_decode::run_with_probs(&ctx, prefill, mock_replay, cfg);

    REQUIRE(!result.tokens.empty());
    // Last token should be one of the EOS ids
    int last = result.tokens.back();
    bool is_eos = (last == 5 || last == 6 || last == 7);
    REQUIRE(is_eos);
}

// ---------------------------------------------------------------------------
// Token-ID invariant — pins the contract that PR #97 fixes at the consumer
// (qwen3 backend) level.
//
// core_beam_decode::run_with_probs returns result.tokens as raw int32_t ids.
// Backends MUST copy those ids into crispasr_token::id (default -1) when
// assembling the output token list. The qwen3 beam-search path was missing
// this assignment; leaving id=-1 broke downstream consumers (native JSON
// "tokens" array, whisper-compat "id" field in -ojf output).
//
// These tests verify:
//   1. Beam result tokens are non-negative (they are valid vocab ids, not -1).
//   2. The values match what the mock LLM emits (hot_tok=3 for first 3 calls,
//      then EOS=5) — so a consumer that copies result.tokens[i] → crispasr_token.id
//      gets a meaningful id, not the sentinel -1.
// ---------------------------------------------------------------------------

TEST_CASE("beam_decode: returned token ids are non-negative", "[unit][decode]") {
    MockCtx ctx;
    float prefill[8];
    for (int i = 0; i < 8; i++)
        prefill[i] = -10.0f;
    prefill[3] = 5.0f;

    core_beam_decode::Config cfg;
    cfg.max_new_tokens = 10;
    cfg.eos_id = 5;
    cfg.vocab_size = 8;
    cfg.beam_size = 4;
    cfg.prompt_len = 0;

    auto result = core_beam_decode::run_with_probs(&ctx, prefill, mock_replay, cfg);

    REQUIRE(!result.tokens.empty());
    for (int32_t id : result.tokens)
        REQUIRE(id >= 0);
}

TEST_CASE("beam_decode: token ids match expected hot_tok sequence", "[unit][decode]") {
    // The mock emits hot_tok=3 for the first 3 steps, then EOS=5.
    // Verifying the exact ids guarantees that a backend copying
    // result.tokens[i] → crispasr_token.id gets the correct vocabulary
    // index (not the -1 sentinel left when the assignment is absent).
    MockCtx ctx;
    float prefill[8];
    for (int i = 0; i < 8; i++)
        prefill[i] = -10.0f;
    prefill[3] = 5.0f;

    core_beam_decode::Config cfg;
    cfg.max_new_tokens = 10;
    cfg.eos_id = 5;
    cfg.vocab_size = 8;
    cfg.beam_size = 1; // beam_size=1 → deterministic, same as greedy
    cfg.prompt_len = 0;

    auto result = core_beam_decode::run_with_probs(&ctx, prefill, mock_replay, cfg);

    REQUIRE(result.tokens.size() >= 3);
    REQUIRE(result.tokens[0] == 3);
    REQUIRE(result.tokens[1] == 3);
    REQUIRE(result.tokens[2] == 3);
    // EOS terminates the sequence
    REQUIRE(result.tokens.back() == 5);
}
