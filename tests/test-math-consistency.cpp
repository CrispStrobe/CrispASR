// test-math-consistency.cpp — verify that M_PI and other math constants are available.

#define _USE_MATH_DEFINES
#include <cmath>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

TEST_CASE("math: M_PI is available and accurate", "[unit]") {
#ifdef M_PI
    REQUIRE(M_PI == Catch::Approx(3.14159265358979323846));
#else
    FAIL("M_PI is not defined even with _USE_MATH_DEFINES");
#endif
}

TEST_CASE("math: common trigonometric functions work", "[unit]") {
    REQUIRE(std::sin(0.0) == Catch::Approx(0.0));
    REQUIRE(std::cos(0.0) == Catch::Approx(1.0));
    REQUIRE(std::sin(M_PI / 2.0) == Catch::Approx(1.0));
}
