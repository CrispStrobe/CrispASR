// test-session-output-rate-parity.cpp — every backend the session can
// synthesize with must also answer crispasr_session_output_sample_rate().
//
// The two are separate hand-maintained if-chains over the same `s-><x>_ctx`
// members in src/crispasr_c_api.cpp, and nothing tied them together. Breeze-
// TTS-2 (#412) gained a synthesize arm and no rate arm, so the session reported
// 0 Hz for it: a binding that plays at the reported rate has nothing to play
// at, and one that falls back to a default is right only by luck. The CLI never
// sees this — its adapters answer tts_sample_rate() themselves — so it only
// surfaced through a downstream app.
//
// The check is structural on purpose: it needs no weights, runs in the unit
// suite, and fails the moment a new backend repeats the omission.

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>

#ifndef CRISPASR_SOURCE_DIR
#error "CRISPASR_SOURCE_DIR must be defined by the build"
#endif

namespace {

std::string slurp(const std::string& rel) {
    std::ifstream f(std::string(CRISPASR_SOURCE_DIR) + "/" + rel, std::ios::binary);
    REQUIRE(f.good());
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Brace-matched body of the first function whose signature contains `sig`.
std::string function_body(const std::string& src, const std::string& sig) {
    const size_t at = src.find(sig);
    REQUIRE(at != std::string::npos);
    const size_t open = src.find('{', at);
    REQUIRE(open != std::string::npos);
    int depth = 0;
    for (size_t i = open; i < src.size(); i++) {
        if (src[i] == '{')
            depth++;
        else if (src[i] == '}' && --depth == 0)
            return src.substr(open, i - open + 1);
    }
    FAIL("unbalanced braces after " << sig);
    return {};
}

std::set<std::string> ctx_members(const std::string& body) {
    static const std::regex re(R"(s->(\w+_ctx\w*)\b)");
    std::set<std::string> out;
    for (auto it = std::sregex_iterator(body.begin(), body.end(), re); it != std::sregex_iterator(); ++it)
        out.insert((*it)[1].str());
    return out;
}

} // namespace

TEST_CASE("every synthesizing backend reports an output sample rate", "[output-rate-parity]") {
    const std::string src = slurp("src/crispasr_c_api.cpp");
    const auto synth = ctx_members(function_body(src, "crispasr_session_synthesize_raw_impl("));
    const auto rate = ctx_members(function_body(src, "CA_EXPORT int crispasr_session_output_sample_rate("));

    // Guard the guard: if either extraction silently matched nothing, the
    // set difference below would pass vacuously.
    REQUIRE(synth.size() >= 30);
    REQUIRE(rate.size() >= 30);

    for (const auto& ctx : synth) {
        INFO("s->" << ctx << " has a synthesize arm but no crispasr_session_output_sample_rate arm");
        CHECK(rate.count(ctx) == 1);
    }
}
