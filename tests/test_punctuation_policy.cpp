// test_punctuation_policy.cpp — regression tests for punctuation auto-enable.

#include "crispasr_punctuation_policy.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("auto punctuation is enabled only for backends that need it", "[unit][punctuation]") {
    whisper_params p;

    p.punctuation = true;
    p.punc_model.clear();
    REQUIRE(crispasr_should_auto_enable_punctuation(0, p));

    p.punc_model = "firered";
    REQUIRE_FALSE(crispasr_should_auto_enable_punctuation(0, p));

    p.punc_model.clear();
    p.punctuation = false;
    REQUIRE_FALSE(crispasr_should_auto_enable_punctuation(0, p));

    p.punctuation = true;
    REQUIRE_FALSE(crispasr_should_auto_enable_punctuation(CAP_PUNCTUATION_TOGGLE, p));
    REQUIRE_FALSE(crispasr_should_auto_enable_punctuation(CAP_PUNCTUATION_NATIVE, p));
}
