// test-vad-keywords.cpp — every --vad-model keyword the resolver accepts must
// be advertised in --help, and vice versa.
//
// These were two hand-maintained lists that had already drifted: the help line
// named only 'firered' and 'silero' while crispasr_resolve_vad_model() also
// accepted 'whisper-vad', 'marblenet', 'webrtc' and 'auto'/'default'. Four
// working options were undiscoverable for anyone who did not read the source.
// Found while verifying a command before putting it in a public issue reply.
//
// The keyword string is now single-sourced in crispasr_vad_cli.h; this pins
// that the RESOLVER still agrees with it, which a shared constant alone cannot
// guarantee (the resolver has its own literal comparisons).

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef CRISPASR_SOURCE_DIR
#error "CRISPASR_SOURCE_DIR must be defined by the build"
#endif

namespace {

std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    REQUIRE(f.good());
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Every keyword the resolver special-cases or exempts from the "treat as a
// path" test. Kept here deliberately: if someone adds a keyword to the
// resolver without advertising it, this list is what they must also touch,
// and the assertions below then force the help string to match.
const char* kResolverKeywords[] = {"auto",    "default",     "silero",    "silero-v6.3",
                                   "firered", "whisper-vad", "marblenet", "webrtc"};

} // namespace

TEST_CASE("every --vad-model keyword the resolver accepts is advertised", "[vad-keywords]") {
    const std::string hdr = slurp(std::string(CRISPASR_SOURCE_DIR) + "/examples/cli/crispasr_vad_cli.h");
    const size_t fn = hdr.find("crispasr_vad_model_keywords()");
    REQUIRE(fn != std::string::npos);
    const size_t q0 = hdr.find('"', hdr.find("return", fn));
    REQUIRE(q0 != std::string::npos);
    const size_t q1 = hdr.find('"', q0 + 1);
    REQUIRE(q1 != std::string::npos);
    const std::string advertised = hdr.substr(q0 + 1, q1 - q0 - 1);
    INFO("advertised: " << advertised);

    for (const char* kw : kResolverKeywords) {
        // "default" is an alias of "auto" and need not be listed separately.
        if (std::string(kw) == "default")
            continue;
        INFO("keyword: " << kw);
        REQUIRE(advertised.find(kw) != std::string::npos);
    }
}

TEST_CASE("the resolver still accepts every advertised keyword", "[vad-keywords]") {
    // The other direction: an entry could be advertised and then dropped from
    // the resolver, leaving --help promising an option that is treated as a
    // file path and fails with a confusing "cannot open" error.
    const std::string src = slurp(std::string(CRISPASR_SOURCE_DIR) + "/examples/cli/crispasr_vad_cli.cpp");
    for (const char* kw : kResolverKeywords) {
        INFO("keyword: " << kw);
        REQUIRE(src.find(std::string("\"") + kw + "\"") != std::string::npos);
    }
}

TEST_CASE("the help line consumes the shared keyword string", "[vad-keywords]") {
    // Guards the join, not the predicate: the constant could exist and be
    // correct while --help still printed its own stale literal.
    const std::string cli = slurp(std::string(CRISPASR_SOURCE_DIR) + "/examples/cli/cli.cpp");
    REQUIRE(cli.find("crispasr_vad_model_keywords()") != std::string::npos);
    // And the old hardcoded wording is gone.
    REQUIRE(cli.find("VAD model (path, 'firered', or 'silero')") == std::string::npos);
}
