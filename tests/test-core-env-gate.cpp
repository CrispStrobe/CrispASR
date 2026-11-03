// test-core-env-gate.cpp — hermetic guard for src/core/env_gate.h, the
// value-parsed boolean gate shared with CrispEmbed (issue #50 mirror). No
// models, no network.
//
// Pins the contract both repos rely on: unset/empty are off for on() and NOT
// explicitly off; the off spellings are exactly "0", "false", "off", "no"
// (ASCII case-insensitive, untrimmed); every other non-empty value is on, so a
// typo never silently disables an instrument.

#include <catch2/catch_test_macros.hpp>

#include "core/env_gate.h"
#include "scoped_test_env.h"

static const char* const kVar = "CRISPASR_TEST_CORE_ENV_GATE";

TEST_CASE("core_env: unset is off and not explicitly off", "[unit][env-gate]") {
    ScopedTestEnv env(kVar);
    REQUIRE_FALSE(core_env::on(kVar));
    REQUIRE_FALSE(core_env::explicitly_off(kVar));
    REQUIRE_FALSE(core_env::on(nullptr));
    REQUIRE_FALSE(core_env::explicitly_off(nullptr));
}

TEST_CASE("core_env: off spellings", "[unit][env-gate]") {
    for (const char* w : {"0", "false", "off", "no", "False", "FALSE", "Off", "OFF", "No", "NO", "nO"}) {
        INFO(w);
        ScopedTestEnv env(kVar, w);
        REQUIRE_FALSE(core_env::on(kVar));
        REQUIRE(core_env::explicitly_off(kVar));
        REQUIRE(core_env::spells_off(w));
    }
}

TEST_CASE("core_env: other non-empty values are on", "[unit][env-gate]") {
    for (const char* w :
         {"1", "2", "yes", "true", "on", "00", "0 ", "nope", "n", "of", "offf", "fals", "false ", " no", "none"}) {
        INFO(w);
        ScopedTestEnv env(kVar, w);
        REQUIRE(core_env::on(kVar));
        REQUIRE_FALSE(core_env::explicitly_off(kVar));
    }
    REQUIRE_FALSE(core_env::spells_off(nullptr));
    REQUIRE_FALSE(core_env::spells_off(""));
}

#ifndef _WIN32
TEST_CASE("core_env: empty is off and not explicitly off", "[unit][env-gate]") {
    // The Windows CRT cannot hold a set-but-empty variable.
    ScopedTestEnv env(kVar, "");
    REQUIRE_FALSE(core_env::on(kVar));
    REQUIRE_FALSE(core_env::explicitly_off(kVar));
}
#endif
