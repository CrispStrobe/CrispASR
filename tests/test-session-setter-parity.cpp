// test-session-setter-parity.cpp — every session setter must exist in EVERY
// wrapper (docs/contributing.md §"Bindings — adding a new session setter").
//
// That checklist says these entry points are "kept at full parity" across nine
// surfaces, and nothing enforced it. `crispasr_session_set_sensitivity` shipped
// in the C ABI and Python only, and stayed that way through two commits and a
// release-notes pass because a hand-maintained list of nine files has no
// machine check — the exact failure the copies-in-sync guard was built for
// (it covered 1 of 14 files for months).
//
// This is intentionally narrow: it pins the setters this repo has already been
// burned on rather than trying to enumerate all ~60. Add a row when you add a
// setter; the point is that ADDING the row forces you to visit every file.

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

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

// The nine surfaces from docs/contributing.md. The C ABI is the source of
// truth; the rest must mirror it.
const char* kSurfaces[] = {
    "src/crispasr_c_api.cpp",
    "python/crispasr/_binding.py",
    "bindings/go/crispasr_session.go",
    "crispasr-sys/src/lib.rs",
    "crispasr/src/lib.rs",
    "flutter/crispasr/lib/src/crispasr.dart",
    "bindings/java/src/main/java/io/github/ggerganov/whispercpp/CrispasrSession.java",
    "bindings/ruby/ext/ruby_crispasr_session.c",
    "bindings/csharp/CrispASR/NativeMethods.cs",
    "bindings/javascript/emscripten.cpp",
};

} // namespace

TEST_CASE("crispasr_session_set_sensitivity reaches every wrapper", "[setter-parity]") {
    for (const char* rel : kSurfaces) {
        INFO("surface: " << rel);
        const std::string src = slurp(rel);
        REQUIRE(src.find("crispasr_session_set_sensitivity") != std::string::npos);
    }
}

TEST_CASE("the sibling it bundles is present everywhere too", "[setter-parity]") {
    // set_fallback_thresholds is the API set_sensitivity is sugar for. If it
    // ever loses a surface, set_sensitivity there becomes unoverridable.
    for (const char* rel : kSurfaces) {
        INFO("surface: " << rel);
        const std::string src = slurp(rel);
        REQUIRE(src.find("crispasr_session_set_fallback_thresholds") != std::string::npos);
    }
}

TEST_CASE("issue #360: set_min_speech_tokens reaches every wrapper", "[setter-parity]") {
    // Reported as "set_min_speech_tokens not exposed in libraries". It was
    // worse than that: the floor was reachable from exactly ONE surface,
    // /v1/audio/speech, which parsed "min_speech_tokens" into a whisper_params
    // field the CLI had no flag for and the session ABI never read. Its `max`
    // sibling was on all ten wrappers, which is the asymmetry the reporter saw.
    for (const char* rel : kSurfaces) {
        INFO("surface: " << rel);
        const std::string src = slurp(rel);
        REQUIRE(src.find("crispasr_session_set_min_speech_tokens") != std::string::npos);
    }
}

TEST_CASE("issue #360: the max sibling stays on every wrapper too", "[setter-parity]") {
    // The pair is only comprehensible together; pin both so neither can drift
    // back out. This list was "intentionally narrow" and neither was in it.
    for (const char* rel : kSurfaces) {
        INFO("surface: " << rel);
        const std::string src = slurp(rel);
        REQUIRE(src.find("crispasr_session_set_max_speech_tokens") != std::string::npos);
    }
}

TEST_CASE("issue #360: the CLI and the server can both set the floor", "[setter-parity]") {
    const std::string cli = slurp("examples/cli/cli.cpp");
    REQUIRE(cli.find("--tts-min-speech-tokens") != std::string::npos);
    const std::string srv = slurp("examples/cli/crispasr_server.cpp");
    REQUIRE(srv.find("min_speech_tokens") != std::string::npos);
}

TEST_CASE("C# does not swallow the unknown-preset return code", "[setter-parity]") {
    // Session.cs's Check() helper treats rc == -2 as SUCCESS, because for most
    // setters -2 means "this backend doesn't support the knob". For
    // set_sensitivity, -2 means "unknown preset" — routing it through Check()
    // would silently decode at the default thresholds after a typo, which is
    // precisely what the API exists to prevent.
    const std::string src = slurp("bindings/csharp/CrispASR/Session.cs");
    const size_t fn = src.find("public void SetSensitivity(");
    REQUIRE(fn != std::string::npos);
    const std::string body = src.substr(fn, 1200);
    INFO(body);
    REQUIRE(body.find("rc == -2") != std::string::npos);
    REQUIRE(body.find("ArgumentException") != std::string::npos);
}

TEST_CASE("the HTTP server exposes it and applies it before the raw fields", "[setter-parity]") {
    // Order matters: applied first so an explicit entropy_thold in the same
    // request still wins, matching the CLI's last-flag-wins rule.
    const std::string src = slurp("examples/cli/crispasr_server.cpp");
    const size_t sens = src.find("form_string(req, \"sensitivity\"");
    REQUIRE(sens != std::string::npos);
    const size_t ent = src.find("form_float(req, \"entropy_thold\"");
    REQUIRE(ent != std::string::npos);
    REQUIRE(sens < ent);
}

TEST_CASE("issue #574: set_autochunk reaches every wrapper", "[setter-parity]") {
    for (const char* rel : kSurfaces) {
        INFO("surface: " << rel);
        const std::string src = slurp(rel);
        REQUIRE(src.find("crispasr_session_set_autochunk") != std::string::npos);
    }
}

TEST_CASE("issue #574: set_max_new_tokens_per_second reaches every wrapper", "[setter-parity]") {
    for (const char* rel : kSurfaces) {
        INFO("surface: " << rel);
        const std::string src = slurp(rel);
        REQUIRE(src.find("crispasr_session_set_max_new_tokens_per_second") != std::string::npos);
    }
}
