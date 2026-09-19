// test-m2m100-params.cpp — unit tests for m2m100_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "m2m100.h"

TEST_CASE("m2m100_params: default values are sensible", "[unit][m2m100]") {
    struct m2m100_context_params p = m2m100_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("issue #439: M2M100 library default follows checkpoint beam width", "[unit][m2m100][issue439]") {
    // All supported M2M100/WMT21 checkpoints declare num_beams=5. The original
    // fix changed only the CLI adapter, leaving direct and session API callers
    // on greedy decoding -- the surface where the reporter first saw the loop.
    REQUIRE(m2m100_default_beam_size() == 5);
}

TEST_CASE("m2m100_init_from_file: null path returns nullptr", "[unit][m2m100]") {
    struct m2m100_context_params p = m2m100_context_default_params();
    struct m2m100_context* ctx = m2m100_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("m2m100_init_from_file: empty path returns nullptr", "[unit][m2m100]") {
    struct m2m100_context_params p = m2m100_context_default_params();
    struct m2m100_context* ctx = m2m100_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("m2m100_free: NULL context is a no-op", "[unit][m2m100]") {
    m2m100_free(nullptr);
    SUCCEED("m2m100_free tolerated a NULL ctx.");
}
