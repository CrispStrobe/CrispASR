// test-link-odr.cpp — verify that linking multiple backends does not cause ODR violations.

#include "chatterbox.h"
#include "vibevoice.h"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("link: chatterbox and vibevoice can be linked together", "[unit]") {
    // This test primarily checks for linker errors (ODR violations).
    // If it compiles and links, the ODR issue is resolved.
    REQUIRE(true);
}
