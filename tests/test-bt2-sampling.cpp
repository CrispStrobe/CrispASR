// test-bt2-sampling.cpp — per-call bt2-tts sampling knobs (#609 follow-up).
#include <cstdio>
#include <string>
//
// The adapter used to apply --temperature once at init, so a server request's
// `temperature` never reached the model. These pin the per-call resolution,
// including that an unset field returns to the STARTUP value rather than
// keeping the previous request's.

#include "crispasr_bt2_sampling.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("a request's temperature is used as given", "[unit]") {
    const auto s = crispasr_bt2_resolve_sampling(0.4f, -1, -1.0f, -1.0f, 0.9f);
    REQUIRE(s.temperature == 0.4f);
}

TEST_CASE("an unset temperature falls back to the startup value", "[unit]") {
    REQUIRE(crispasr_bt2_resolve_sampling(0.0f, -1, -1.0f, -1.0f, 0.9f).temperature == 0.9f);
    REQUIRE(crispasr_bt2_resolve_sampling(-1.0f, -1, -1.0f, -1.0f, 0.7f).temperature == 0.7f);
}

TEST_CASE("a request without temperature does not inherit the previous request's", "[unit]") {
    // The resolver is stateless: the adapter applies its result on every call,
    // so request 2 cannot see request 1's value.
    const float startup = 0.9f;
    const auto r1 = crispasr_bt2_resolve_sampling(0.3f, 10, 0.8f, 1.3f, startup);
    const auto r2 = crispasr_bt2_resolve_sampling(0.0f, -1, -1.0f, -1.0f, startup);
    REQUIRE(r1.temperature == 0.3f);
    REQUIRE(r2.temperature == startup);
    REQUIRE(r2.top_k == 0);
    REQUIRE(r2.top_p == 0.0f);
    REQUIRE(r2.repetition_penalty == 0.0f);
}

TEST_CASE("top_k, top_p and repetition_penalty pass through or map to checkpoint", "[unit]") {
    const auto s = crispasr_bt2_resolve_sampling(0.0f, 20, 0.95f, 1.2f, 0.9f);
    REQUIRE(s.top_k == 20);
    REQUIRE(s.top_p == 0.95f);
    REQUIRE(s.repetition_penalty == 1.2f);
    // The server's "unset" sentinel is -1; 0 is the native "checkpoint value".
    const auto d = crispasr_bt2_resolve_sampling(0.0f, -1, -1.0f, -1.0f, 0.9f);
    REQUIRE(d.top_k == 0);
    REQUIRE(d.top_p == 0.0f);
    REQUIRE(d.repetition_penalty == 0.0f);
}

TEST_CASE("the adapter applies the knobs per call, not only at init", "[unit]") {
    // Guards the join: the resolver could be right while the adapter still
    // only set temperature in init().
#ifndef CRISPASR_SOURCE_DIR
    SUCCEED("source dir not provided");
#else
    FILE* f = fopen(CRISPASR_SOURCE_DIR "/examples/cli/crispasr_backend_bt2_tts.cpp", "rb");
    REQUIRE(f != nullptr);
    std::string src;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0)
        src.append(buf, n);
    fclose(f);
    const size_t synth = src.find("std::vector<float> synthesize(");
    REQUIRE(synth != std::string::npos);
    const std::string body = src.substr(synth);
    for (const char* call :
         {"crispasr_bt2_resolve_sampling(", "breeze_tts_2_set_temperature(", "breeze_tts_2_set_topk(",
          "breeze_tts_2_set_topp(", "breeze_tts_2_set_repetition_penalty("}) {
        INFO(call);
        REQUIRE(body.find(call) != std::string::npos);
    }
#endif
}
