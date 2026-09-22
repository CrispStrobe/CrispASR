// test-subprocess.cpp — core_subprocess::ReadPipe must pass argv to the child
// verbatim and never through a shell. Strings that would run a command under
// popen()/system() are handed to printf(1) and must come back unchanged, and
// the command they try to run must not have run.

#include <catch2/catch_test_macros.hpp>

#include "core/subprocess.h"

#include <cstdio>
#include <string>
#include <vector>

#if !defined(_WIN32) && !defined(CORE_SUBPROCESS_UNAVAILABLE)
#include <unistd.h>

static std::string run_printf(const std::string& arg, int* rc) {
    core_subprocess::ReadPipe p;
    REQUIRE(p.open({"printf", "%s", arg}));
    std::string out;
    char buf[256];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), p.out)) > 0) {
        out.append(buf, n);
    }
    *rc = p.close();
    return out;
}

TEST_CASE("subprocess: shell metacharacters reach the child verbatim", "[subprocess]") {
    char dir_tmpl[] = "/tmp/crispasr-subprocess-XXXXXX";
    const char* dir = mkdtemp(dir_tmpl);
    REQUIRE(dir != nullptr);
    const std::string sentinel = std::string(dir) + "/ran";

    const std::vector<std::string> hostile = {
        "\"; touch " + sentinel + "; \"",
        "'; touch " + sentinel + "; '",
        "$(touch " + sentinel + ")",
        "`touch " + sentinel + "`",
        "a\ntouch " + sentinel,
        "a && touch " + sentinel,
        "-v",
        "",
        "plain.wav",
    };
    for (const auto& arg : hostile) {
        int rc = -2;
        const std::string out = run_printf(arg, &rc);
        CHECK(rc == 0);
        CHECK(out == arg);
    }
    CHECK(access(sentinel.c_str(), F_OK) != 0);
    rmdir(dir);
}

TEST_CASE("subprocess: a missing program is reported, not run through a shell", "[subprocess]") {
    core_subprocess::ReadPipe p;
    CHECK_FALSE(p.open({"crispasr-no-such-program-xyz"}));
    CHECK(p.close() == -1);
}

TEST_CASE("subprocess: the child's exit code is returned", "[subprocess]") {
    core_subprocess::ReadPipe p;
    REQUIRE(p.open({"sh", "-c", "exit 3"}));
    CHECK(p.close() == 3);
}

TEST_CASE("subprocess: embedded NUL is refused", "[subprocess]") {
    core_subprocess::ReadPipe p;
    CHECK_FALSE(p.open({"printf", std::string("a\0b", 3)}));
}
#endif
