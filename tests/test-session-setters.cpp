// test-session-setters.cpp — unit tests for the generation-control session
// setters added in the full-parity sweep (PLAN §88 / commit c88306fa).
//
// Strategy: call every setter with a null session handle and verify the
// null-guard path returns -1.  No model is loaded — sub-millisecond, no
// network, safe on every CI tier.  A regression here means a setter lost its
// null guard (silent crash risk for callers that open sessions lazily).
//
// Coverage: all setters declared in include/crispasr.h under the
// "Unified session decode / sampling controls" block.

#include <catch2/catch_test_macros.hpp>

#include "crispasr.h"
#include "crispasr_session.h" // crispasr_session_open_explicit (#282 load-path test)

// ─── single-float setters ──────────────────────────────────────────────────

TEST_CASE("session setter: set_temperature null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_temperature(nullptr, 0.8f, 42) == -1);
}

TEST_CASE("session setter: set_top_p null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_top_p(nullptr, 0.9f) == -1);
}

TEST_CASE("session setter: set_min_p null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_min_p(nullptr, 0.05f) == -1);
}

TEST_CASE("session setter: set_repetition_penalty null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_repetition_penalty(nullptr, 1.2f) == -1);
}

TEST_CASE("session setter: set_top_k null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_top_k(nullptr, 5) == -1);
}

TEST_CASE("session setter: set_do_sample null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_do_sample(nullptr, 1) == -1);
}

TEST_CASE("session setter: set_tts_num_candidates null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_tts_num_candidates(nullptr, 4) == -1);
}

TEST_CASE("session setter: set_cfg_weight null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_cfg_weight(nullptr, 0.5f) == -1);
}

TEST_CASE("session setter: set_tts_noise_temp null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_tts_noise_temp(nullptr, 0.9f) == -1);
}

TEST_CASE("session setter: set_exaggeration null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_exaggeration(nullptr, 0.5f) == -1);
}

TEST_CASE("session setter: set_length_scale null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_length_scale(nullptr, 1.0f) == -1);
}

TEST_CASE("session setter: set_frequency_penalty null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_frequency_penalty(nullptr, 0.4f) == -1);
}

// ─── single-int / uint64 setters ──────────────────────────────────────────

TEST_CASE("session setter: set_tts_seed null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_tts_seed(nullptr, 12345) == -1);
}

TEST_CASE("session setter: set_tts_steps null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_tts_steps(nullptr, 20) == -1);
}

TEST_CASE("session setter: set_max_new_tokens null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_max_new_tokens(nullptr, 256) == -1);
}

TEST_CASE("session setter: set_max_speech_tokens null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_max_speech_tokens(nullptr, 1000) == -1);
}

TEST_CASE("session setter: set_best_of null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_best_of(nullptr, 5) == -1);
}

TEST_CASE("session setter: set_beam_size null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_beam_size(nullptr, 4) == -1);
}

TEST_CASE("session setter: set_return_logits null-handle -> -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_return_logits(nullptr, 1) == -1);
}

TEST_CASE("session setter: set_alt_n null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_alt_n(nullptr, 3) == -1);
}

// ─── string / multi-param setters ─────────────────────────────────────────

TEST_CASE("session setter: set_ask null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_ask(nullptr, "hello") == -1);
}

TEST_CASE("session setter: set_grammar_text null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_grammar_text(nullptr, nullptr, nullptr, 100.0f) == -1);
}

TEST_CASE("session setter: set_grammar_strict null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_grammar_strict(nullptr, 1) == -1);
}

TEST_CASE("whisper params: grammar_strict is off by default", "[unit][setters]") {
    // Existing grammar users must see no change: strict end-of-text is opt-in.
    REQUIRE_FALSE(whisper_full_default_params(CRISPASR_SAMPLING_GREEDY).grammar_strict);
    REQUIRE_FALSE(whisper_full_default_params(CRISPASR_SAMPLING_BEAM_SEARCH).grammar_strict);
}

TEST_CASE("score_texts: bad arguments → -1", "[unit][setters]") {
    const float pcm[16] = {};
    const char* texts[] = {"e4"};
    float lp[1];
    int nt[1];
    REQUIRE(crispasr_session_score_texts(nullptr, pcm, 16, "en", nullptr, texts, 1, lp, nt) == -1);
    REQUIRE(whisper_score_texts(nullptr, pcm, 16, "en", nullptr, texts, 1, lp, nt, 1) == -1);
}

TEST_CASE("session setter: set_fallback_thresholds null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_fallback_thresholds(nullptr, 2.4f, -1.0f, 0.6f, 0.2f) == -1);
}

TEST_CASE("session setter: set_whisper_decode_extras null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_whisper_decode_extras(nullptr, 0, nullptr, 0) == -1);
}

// ─── last_synth_error ─────────────────────────────────────────────────────

TEST_CASE("session: last_synth_error null-handle → empty string", "[unit][setters]") {
    const char* err = crispasr_session_last_synth_error(nullptr);
    REQUIRE(err != nullptr);
    REQUIRE(err[0] == '\0');
}

// ─── #282: lazy dynamic-backend load path ─────────────────────────────────
// g_open_use_gpu_tls defaults to true, so crispasr_session_open_explicit runs
// ensure_dynamic_backends_loaded() (→ ggml_backend_load_all()) before the model
// is loaded. On a CPU-only CI box no GPU plugins are found; the load must be a
// safe no-op and a missing model must fail cleanly to nullptr, not crash. The
// second call proves the std::call_once guard makes repeat opens safe.
TEST_CASE("session open: GPU-default open of missing model loads plugins and returns null safely", "[unit][setters]") {
    crispasr_session* s1 = crispasr_session_open_explicit("/nonexistent/crispasr-282.gguf", "whisper", 1);
    REQUIRE(s1 == nullptr);
    crispasr_session* s2 = crispasr_session_open_explicit("/nonexistent/crispasr-282.gguf", "whisper", 1);
    REQUIRE(s2 == nullptr);
}

// ─── standalone punctuation API: wrong GGUFs fail at init (#460) ──────────
// A GGUF that is not a complete punctuation model used to load as
// FireRedPunc, log "required tensor ... not found", report success and then
// segfault in ggml_mul_mat on the first process() call.

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <filesystem>
#include <string>

static std::string write_tiny_gguf(const char* arch, const char* tensor_name, const char* file) {
    const std::string path = (std::filesystem::temp_directory_path() / file).string();
    ggml_init_params ip = {1024 * 1024, nullptr, false};
    ggml_context* ctx = ggml_init(ip);
    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", arch);
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(t, tensor_name);
    for (int i = 0; i < 4; i++)
        ((float*)t->data)[i] = 0.5f * i;
    gguf_add_tensor(g, t);
    gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

TEST_CASE("punc init: FireRedPunc-arch GGUF missing tensors -> NULL, not a later crash", "[unit][punc]") {
    const std::string p = write_tiny_gguf("fireredpunc", "emb.tok_emb.weight", "crispasr-460-incomplete.gguf");
    void* h = crispasr_punc_init(p.c_str());
    REQUIRE(h == nullptr);
    std::remove(p.c_str());
}

TEST_CASE("punc init: a GGUF of another architecture -> NULL", "[unit][punc]") {
    const std::string p = write_tiny_gguf("whisper", "encoder.conv1.weight", "crispasr-460-foreign.gguf");
    REQUIRE(crispasr_punc_init(p.c_str()) == nullptr);
    std::remove(p.c_str());
}

TEST_CASE("punc init: missing file / empty / null handle are safe", "[unit][punc]") {
    REQUIRE(crispasr_punc_init("/nonexistent/crispasr-460.gguf") == nullptr);
    REQUIRE(crispasr_punc_init("") == nullptr);
    REQUIRE(crispasr_punc_init(nullptr) == nullptr);
    REQUIRE(crispasr_punc_process(nullptr, "hello") == nullptr);
    crispasr_punc_free(nullptr); // must not crash
}
