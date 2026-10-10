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
