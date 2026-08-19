// #365 — canary duplicated phrases and ran its timestamps backwards.
//
// canary_transcribe_streamed decodes long audio in overlapping chunks (8 s
// chunk, 2 s overlap) and deduped the seam with an LCS over token ids. An LCS
// compares what the decoder SAID, so it only works when the decoder says the
// same thing twice. Given a different amount of right-context an attention
// decoder routinely re-words the overlap instead, and then the LCS matches
// nothing, the whole re-transcription is appended, and the timestamps run
// backwards because the new chunk's tokens carry their own chunk offset:
//
//   00:20:27,060 --> 00:20:28,340  J'arrive sur le...
//   00:20:26,820 --> 00:20:28,020  rendu compte qu'elle n'avait pas dormi.
//
// Time is the signal that survives re-wording: the overlap is a property of the
// chunking, not of the words. Anything ending at or before the last accepted
// token's end is audio already spoken for.
#include "core/asr_overlap_trim.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace {

struct Tok {
    std::string text;
    int64_t t0, t1; // centiseconds
};

int trim(const std::vector<Tok>& toks, int64_t accepted_end) {
    return core_overlap_trim::leading_covered((int)toks.size(), accepted_end,
                                              [&](int i) { return toks[(size_t)i].t1; });
}

} // namespace

TEST_CASE("overlap trim: the re-worded seam from #365 is dropped", "[unit][overlap][issue-365]") {
    // Chunk N ended at 122834 cs (00:20:28,340 — "J'arrive sur le...").
    // Chunk N+1 re-transcribes the tail of that audio with DIFFERENT words, so
    // no LCS can match it, and its first tokens start BEFORE where we already
    // are.
    const std::vector<Tok> next_chunk = {
        {" rendu", 122682, 122760},   // inside accepted audio
        {" compte", 122760, 122830},  // inside accepted audio
        {" qu'elle", 122830, 122900}, // extends past — first real content
        {" n'avait", 122900, 123000},
    };
    const int skip = trim(next_chunk, /*accepted_end=*/122834);
    REQUIRE(skip == 2);

    // What survives must not reach back before the boundary.
    REQUIRE(next_chunk[(size_t)skip].t1 > 122834);
}

TEST_CASE("overlap trim: nothing is dropped when the chunk is genuinely new", "[unit][overlap][issue-365]") {
    // The common case: chunk N+1 starts after everything accepted. Trimming
    // here would delete real speech, which is worse than the duplication.
    const std::vector<Tok> next_chunk = {
        {" Bonjour", 5000, 5100},
        {" monsieur", 5100, 5250},
    };
    REQUIRE(trim(next_chunk, /*accepted_end=*/4900) == 0);
}

TEST_CASE("overlap trim: a token ending exactly on the boundary is covered", "[unit][overlap][issue-365]") {
    // t1 == accepted_end means the token occupies audio already emitted; the
    // boundary is inclusive, otherwise a zero-length re-emission survives.
    const std::vector<Tok> next_chunk = {{" a", 100, 200}, {" b", 200, 300}};
    REQUIRE(trim(next_chunk, 200) == 1);
    REQUIRE(trim(next_chunk, 199) == 0);
}

TEST_CASE("overlap trim: an entirely covered chunk reports every token", "[unit][overlap][issue-365]") {
    // The helper reports what it sees; the CALLER clamps so a chunk cannot
    // vanish. Pinning the raw contract here so the clamp stays visible at the
    // call site rather than being silently baked in.
    const std::vector<Tok> next_chunk = {{" a", 100, 200}, {" b", 200, 300}};
    REQUIRE(trim(next_chunk, 99999) == 2);
}

TEST_CASE("overlap trim: conservative seam rule requires two covered tokens", "[unit][overlap][issue-375]") {
    const std::vector<Tok> one_early = {{" Many", 100, 200}, {" people", 200, 400}};
    REQUIRE(core_overlap_trim::leading_covered_multi(
                (int)one_early.size(), 250, [&](int i) { return one_early[(size_t)i].t1; }) == 0);

    const std::vector<Tok> repeated = {{" rendu", 100, 200}, {" compte", 200, 250}, {" real", 250, 400}};
    REQUIRE(core_overlap_trim::leading_covered_multi(
                (int)repeated.size(), 250, [&](int i) { return repeated[(size_t)i].t1; }) == 2);
}

TEST_CASE("overlap trim: scanning stops at the first uncovered token", "[unit][overlap][issue-365]") {
    // A decoder can emit a stray token whose timing sits behind its neighbours.
    // Scanning the whole chunk and counting every covered token would drop real
    // speech that follows it, so the scan stops at the first token that extends
    // past the boundary.
    const std::vector<Tok> next_chunk = {
        {" dup", 100, 200},   // covered
        {" real", 200, 400},  // NOT covered — stop here
        {" stray", 150, 180}, // covered, but after real content
    };
    REQUIRE(trim(next_chunk, 200) == 1);
}

TEST_CASE("overlap trim: empty and degenerate inputs", "[unit][overlap][issue-365]") {
    const std::vector<Tok> empty;
    REQUIRE(trim(empty, 1000) == 0);
    REQUIRE(core_overlap_trim::leading_covered(0, 0, [](int) { return (int64_t)0; }) == 0);
    // A negative count must not underflow into a scan.
    REQUIRE(core_overlap_trim::leading_covered(-3, 100, [](int) { return (int64_t)0; }) == 0);
}

TEST_CASE("overlap trim: applied repeatedly, output stays monotone", "[unit][overlap][issue-365]") {
    // The property the SRT actually needs. Feed three overlapping chunks whose
    // decoders disagree about the seam wording, and check that accepted token
    // ends never move backwards — which is the visible symptom in #365.
    const std::vector<std::vector<Tok>> chunks = {
        {{" one", 0, 100}, {" two", 100, 200}, {" three", 200, 300}},
        {{" THREE", 200, 300}, {" four", 300, 400}, {" five", 400, 500}}, // re-worded seam
        {{" FIVE", 400, 500}, {" six", 500, 600}},                        // re-worded seam
    };
    std::vector<Tok> accepted;
    for (const auto& c : chunks) {
        int skip = 0;
        if (!accepted.empty()) {
            skip = trim(c, accepted.back().t1);
            if (skip >= (int)c.size())
                skip = (int)c.size() - 1; // the caller's clamp
        }
        for (size_t i = (size_t)skip; i < c.size(); i++)
            accepted.push_back(c[i]);
    }
    REQUIRE(accepted.size() == 6);
    for (size_t i = 1; i < accepted.size(); i++) {
        INFO("token " << i << " t0=" << accepted[i].t0 << " prev t1=" << accepted[i - 1].t1);
        REQUIRE(accepted[i].t1 >= accepted[i - 1].t1);
    }
    // And no duplicated words survived the seams.
    std::string joined;
    for (const auto& t : accepted)
        joined += t.text;
    REQUIRE(joined == " one two three four five six");
}
