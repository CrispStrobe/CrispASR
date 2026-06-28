// test-speaker-db.cpp — unit tests for the named-voiceprint profile DB
// (src/speaker_db.{h,cpp}). Drives enroll / load / count / match with
// synthetic L2-normalized embeddings, so the suite is deterministic and
// needs no model load, no audio, and no network.
//
// This DB underpins the deliberately opt-in --speaker-db biometric path
// (1:N identification). The session-scoped *clustering* path that produces
// anonymous (speaker N) labels is covered separately by
// test-speaker-cluster.cpp.

#include "../src/speaker_db.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#include <process.h>
#include <windows.h>
static std::string make_temp_dir() {
    char buf[MAX_PATH];
    GetTempPathA(MAX_PATH, buf);
    std::string base = buf;
    if (!base.empty() && (base.back() == '\\' || base.back() == '/'))
        base.pop_back();
    std::string dir = base + "/crispasr_spkdb_" + std::to_string(_getpid());
    _mkdir(dir.c_str());
    return dir;
}
#else
#include <sys/stat.h>
#include <unistd.h>
static std::string make_temp_dir() {
    const char* env = std::getenv("CRISPASR_SCRATCH_DIR");
    std::string base = (env && *env) ? env : ".scratch";
    mkdir(base.c_str(), 0755);
    std::string pattern = base + "/crispasr_spkdb_XXXXXX";
    std::string writable = pattern;
    char* buf = writable.data();
    return mkdtemp(buf) ? std::string(buf) : base;
}
#endif

namespace {

// L2-normalize a copy so cosine == dot, matching how real embedders feed
// the DB (speaker_db_match assumes normalized inputs).
std::vector<float> norm(std::vector<float> v) {
    double s = 0.0;
    for (float x : v)
        s += (double)x * x;
    if (s > 0.0) {
        const float inv = (float)(1.0 / std::sqrt(s));
        for (float& x : v)
            x *= inv;
    }
    return v;
}

constexpr int D = 8;

} // namespace

TEST_CASE("speaker_db: empty / missing directory yields zero speakers and no match", "[unit]") {
    const std::string dir = make_temp_dir() + "/does_not_exist_yet";
    speaker_db* db = speaker_db_load(dir.c_str());
    REQUIRE(db != nullptr); // missing dir is valid (0 speakers), not an error
    REQUIRE(speaker_db_count(db) == 0);

    const std::vector<float> q = norm({1, 0, 0, 0, 0, 0, 0, 0});
    float score = 123.0f;
    REQUIRE(speaker_db_match(db, q.data(), D, 0.5f, &score) == nullptr);
    speaker_db_free(db);
}

TEST_CASE("speaker_db: enroll then match returns the nearest enrolled name", "[unit]") {
    const std::string dir = make_temp_dir();
    const std::vector<float> alice = norm({1, 0, 0, 0, 0, 0, 0, 0});
    const std::vector<float> bob = norm({0, 1, 0, 0, 0, 0, 0, 0});

    REQUIRE(speaker_db_enroll(dir.c_str(), "alice", alice.data(), D));
    REQUIRE(speaker_db_enroll(dir.c_str(), "bob", bob.data(), D));

    speaker_db* db = speaker_db_load(dir.c_str());
    REQUIRE(db != nullptr);
    REQUIRE(speaker_db_count(db) == 2);

    // A query close to alice (cosine ~0.995) should resolve to "alice".
    const std::vector<float> near_alice = norm({0.9f, 0.1f, 0, 0, 0, 0, 0, 0});
    float score = 0.0f;
    const char* name = speaker_db_match(db, near_alice.data(), D, 0.7f, &score);
    REQUIRE(name != nullptr);
    REQUIRE(std::string(name) == "alice");
    REQUIRE(score > 0.9f);

    // ...and a query close to bob resolves to "bob".
    const std::vector<float> near_bob = norm({0.1f, 0.9f, 0, 0, 0, 0, 0, 0});
    name = speaker_db_match(db, near_bob.data(), D, 0.7f, &score);
    REQUIRE(name != nullptr);
    REQUIRE(std::string(name) == "bob");
    speaker_db_free(db);
}

TEST_CASE("speaker_db: a dissimilar voice is rejected (below threshold => no name)", "[unit]") {
    const std::string dir = make_temp_dir();
    REQUIRE(speaker_db_enroll(dir.c_str(), "alice", norm({1, 0, 0, 0, 0, 0, 0, 0}).data(), D));

    speaker_db* db = speaker_db_load(dir.c_str());
    REQUIRE(db != nullptr);

    // Orthogonal to alice => cosine ~0 => below any sane threshold.
    const std::vector<float> stranger = norm({0, 0, 1, 0, 0, 0, 0, 0});
    float score = 1.0f;
    REQUIRE(speaker_db_match(db, stranger.data(), D, 0.5f, &score) == nullptr);
    REQUIRE(score < 0.5f); // out_score still reports the best (rejected) similarity
    speaker_db_free(db);
}

TEST_CASE("speaker_db: threshold gates an otherwise-close match", "[unit]") {
    const std::string dir = make_temp_dir();
    REQUIRE(speaker_db_enroll(dir.c_str(), "alice", norm({1, 0, 0, 0, 0, 0, 0, 0}).data(), D));
    speaker_db* db = speaker_db_load(dir.c_str());

    // cosine ~0.707 with alice: matches at 0.7 but is rejected at 0.8.
    const std::vector<float> q = norm({1, 1, 0, 0, 0, 0, 0, 0});
    REQUIRE(speaker_db_match(db, q.data(), D, 0.7f, nullptr) != nullptr);
    REQUIRE(speaker_db_match(db, q.data(), D, 0.8f, nullptr) == nullptr);
    speaker_db_free(db);
}

TEST_CASE("speaker_db: dimension mismatch never matches", "[unit]") {
    const std::string dir = make_temp_dir();
    REQUIRE(speaker_db_enroll(dir.c_str(), "alice", norm({1, 0, 0, 0, 0, 0, 0, 0}).data(), D));
    speaker_db* db = speaker_db_load(dir.c_str());

    // Querying with a different dimensionality (e.g. wrong embedder) must
    // not produce a false identification.
    const std::vector<float> q4 = norm({1, 0, 0, 0});
    REQUIRE(speaker_db_match(db, q4.data(), 4, 0.0f, nullptr) == nullptr);
    speaker_db_free(db);
}
