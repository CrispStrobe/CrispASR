#include "core/vibevoice_tts_generation.h"
#include <catch2/catch_test_macros.hpp>
#include <climits>

TEST_CASE("VibeVoice base generation follows the serving prompt budget", "[unit][vibevoice]") {
    using core_vibevoice_tts::generation_budget;
    // Full prompt, not just nine text tokens: allow speech after frame 27.
    CHECK(generation_budget(0, 120, 32768) == 240);
    CHECK(generation_budget(0, 300, 65536) == 600);
    CHECK(generation_budget(0, 160, 32768) > generation_budget(0, 120, 32768));
    CHECK(generation_budget(-1, 120, 32768) == 240);
    CHECK(generation_budget(20, 120, 32768) == 20);
    CHECK(generation_budget(1000, 120, 32768) == 240);
    CHECK(generation_budget(0, 100, 105) == 5);
    CHECK(generation_budget(INT_MAX, 100, 105) == 5);
    CHECK(generation_budget(0, 105, 105) == 0);
    CHECK(generation_budget(0, 106, 105) == 0);
    CHECK(generation_budget(0, 0, 105) == 0);
    CHECK(generation_budget(0, -1, 105) == 0);
    CHECK(generation_budget(0, INT_MAX / 2 + 1, INT_MAX) == INT_MAX / 2);
}

TEST_CASE("Legacy VibeVoice base models retain their source context lengths", "[unit][vibevoice]") {
    CHECK(core_vibevoice_tts::legacy_context_length(1536) == 65536);
    CHECK(core_vibevoice_tts::legacy_context_length(3584) == 32768);
}
