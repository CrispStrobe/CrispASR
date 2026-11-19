// test-breeze-cfg.cpp — unit tests for core/breeze_cfg.h, the classifier-free
// guidance plan and logits combine behind bt2-tts Voice Design / Direction.
//
// Pinned to upstream breeze_infer/templates.py::prepare_inputs (which prompt
// branches exist for a given scale) and models/generation_breeze.py (the
// combine). No model load, pure CPU.

#include <catch2/catch_test_macros.hpp>

#include "core/breeze_cfg.h"

#include <vector>

TEST_CASE("breeze cfg — no instruction is always one plain branch", "[unit][breeze-cfg]") {
    for (float scale : {0.0f, 1.0f, 4.0f}) {
        const auto plan = breeze_cfg::plan(false, scale);
        REQUIRE(plan.size() == 1);
        REQUIRE_FALSE(plan[0].instruction);
    }
}

TEST_CASE("breeze cfg — scale 1 runs the instruction prompt alone", "[unit][breeze-cfg]") {
    const auto plan = breeze_cfg::plan(true, 1.0f);
    REQUIRE(plan.size() == 1);
    REQUIRE(plan[0].instruction);
}

TEST_CASE("breeze cfg — scale 0 is the negative prompt alone", "[unit][breeze-cfg]") {
    const auto plan = breeze_cfg::plan(true, 0.0f);
    REQUIRE(plan.size() == 1);
    REQUIRE_FALSE(plan[0].instruction);
}

TEST_CASE("breeze cfg — any other scale guides against the negative prompt", "[unit][breeze-cfg]") {
    const auto plan = breeze_cfg::plan(true, 4.0f);
    REQUIRE(plan.size() == 2);
    REQUIRE_FALSE(plan[0].instruction);
    REQUIRE(plan[1].instruction);
    REQUIRE(plan[1].scale == 4.0f);
}

TEST_CASE("breeze cfg — combine is negative + scale * (positive - negative)", "[unit][breeze-cfg]") {
    const std::vector<float> rows = {1.0f, -2.0f, 0.5f, /* positive */ 2.0f, -1.0f, 0.5f};
    const auto plan = breeze_cfg::plan(true, 4.0f);
    std::vector<float> out(3);
    breeze_cfg::combine(rows.data(), 3, plan, out.data());
    REQUIRE(out[0] == 5.0f);
    REQUIRE(out[1] == 2.0f);
    REQUIRE(out[2] == 0.5f);
}

TEST_CASE("breeze cfg — a single branch combines to itself", "[unit][breeze-cfg]") {
    const std::vector<float> rows = {1.0f, -2.0f, 0.5f};
    std::vector<float> out(3);
    breeze_cfg::combine(rows.data(), 3, breeze_cfg::plan(true, 1.0f), out.data());
    REQUIRE(out == rows);
}

// #594: upstream's GeneratedTokenRepetitionPenaltyLogitsProcessor penalises
// each distinct generated token ONCE (gather/scatter). A token emitted 300
// times must be exactly as penalised as one emitted once; compounding it to
// 1.1^300 suppressed every common code and made the model read its
// instruction aloud on long chunks.
TEST_CASE("breeze cfg — repetition penalty applies once per distinct token", "[unit][breeze-cfg]") {
    std::vector<float> logits = {2.2f, -1.0f, 3.0f, 0.0f};
    std::vector<int32_t> history(300, 0);
    history.push_back(1);
    history.push_back(1);
    history.push_back(-5);  // out of range: ignored
    history.push_back(999); // out of range: ignored
    breeze_cfg::apply_repetition_penalty(logits.data(), 4, history, 1.1f);
    REQUIRE(logits[0] == 2.2f / 1.1f);
    REQUIRE(logits[1] == -1.0f * 1.1f);
    REQUIRE(logits[2] == 3.0f); // never generated
    REQUIRE(logits[3] == 0.0f);
}

TEST_CASE("breeze cfg — repetition penalty 1 and empty history are no-ops", "[unit][breeze-cfg]") {
    std::vector<float> logits = {2.0f, -1.0f};
    const std::vector<float> orig = logits;
    breeze_cfg::apply_repetition_penalty(logits.data(), 2, {0, 1}, 1.0f);
    REQUIRE(logits == orig);
    breeze_cfg::apply_repetition_penalty(logits.data(), 2, {}, 1.3f);
    REQUIRE(logits == orig);
}
