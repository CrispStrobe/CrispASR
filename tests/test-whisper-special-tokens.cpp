// test-whisper-special-tokens.cpp — serialized-vs-legacy special-token layout (#322).
//
// Nothing pinned this before. Whisper is absent from tests/regression/manifest.json
// (45 backends, none of them whisper) and no test anywhere asserted a token id, so
// the loader resolving `token_eot` to the wrong value was invisible to CI — and
// would have been invisible again after the fix.
//
// The bug: the old probe was `set_token_id(vocab.token_eot, "<|endoftext|>") &&
// set_token_id(vocab.token_sot, "<|startoftranscript|>")`. `set_token_id()` ASSIGNS
// its destination and then returns true, and it sat on the left of a short-circuit
// `&&` — so a vocab carrying `<|endoftext|>` but not `<|startoftranscript|>` came out
// with the flag correctly false and `token_eot` already overwritten. The legacy
// fixup (`token_eot++`) then incremented an id that was already resolved.
//
// The ids below are REAL, measured across eight whisper `.bin` files and
// openai/whisper-tiny's vocab.json — not invented for the test.

#include <catch2/catch_test_macros.hpp>

#include "core/whisper_special_tokens.h"

using core_whisper_specials::Serialized;
using core_whisper_specials::use_serialized;
using core_whisper_specials::valid_logits_indices;

TEST_CASE("required special indices fit legacy, serialized and Tiron vocabularies", "[unit][whisper]") {
    REQUIRE(valid_logits_indices(51864, 50256, 50257, 50363, false, 0, 0));
    REQUIRE(valid_logits_indices(51865, 50257, 50258, 50364, false, 0, 0));
    REQUIRE(valid_logits_indices(10240, 8631, 8632, 8739, false, 0, 0));
    REQUIRE(valid_logits_indices(51904, 50257, 50258, 50365, true, 51866, 8));
    REQUIRE_FALSE(valid_logits_indices(8739, 8631, 8632, 8739, false, 0, 0));
    REQUIRE_FALSE(valid_logits_indices(100, 100, 1, 2, false, 0, 0));
    REQUIRE_FALSE(valid_logits_indices(100, 0, 100, 2, false, 0, 0));
    REQUIRE_FALSE(valid_logits_indices(51873, 50257, 50258, 50365, true, 51866, 8));
    REQUIRE_FALSE(valid_logits_indices(100, -1, 1, 2, false, 0, 0));
}

TEST_CASE("prebuilt multilingual models use the legacy layout", "[unit][whisper]") {
    // ggerganov's ggml-tiny/base/large-v3 — what download-ggml-model.sh fetches —
    // serialize 50257 entries and contain no specials by name at all. Measured on
    // tiny, base and large-v3-turbo. These were never affected by #322.
    Serialized s;
    REQUIRE_FALSE(use_serialized(s));
}

TEST_CASE("English-only (.en) vocabs use the legacy layout", "[unit][whisper]") {
    // tiny.en / base.en DO serialize `<|endoftext|>`, at 50256. Legacy nonetheless.
    // Note this is why `.en` never exhibited the bug: 50256 is also the compiled-in
    // default, so the old probe's stray assignment happened to be a no-op, and the
    // fixup is skipped anyway because .en is not multilingual.
    Serialized s;
    s.eot = 50256;
    REQUIRE_FALSE(use_serialized(s));
}

TEST_CASE("HF-converted multilingual vocabs use the legacy layout", "[unit][whisper]") {
    // THE #322 TRAP. openai/whisper-tiny's vocab.json has 50258 entries with
    // `<|endoftext|>` at 50257 and no `<|startoftranscript|>`, and upstream's
    // convert-h5-to-ggml.py serializes vocab.json — so eot is present, sot is not.
    //
    // Reporting "legacy" here was never the bug; the bug was that saying so had
    // already written token_eot = 50257, which the fixup then pushed to 50258.
    Serialized s;
    s.eot = 50257;
    REQUIRE_FALSE(use_serialized(s));
}

TEST_CASE("fully serialized vocabs use the serialized layout", "[unit][whisper]") {
    // CrispASR-converted (#258, whisper-ja-760M): every special present. Real ids.
    Serialized s;
    s.eot = 8631;
    s.sot = 8632;
    s.beg = 8739;
    REQUIRE(use_serialized(s));
}

TEST_CASE("<|0.00|> is required for the serialized layout", "[unit][whisper]") {
    // The serialized branch resolves token_beg from `<|0.00|>`, and token_beg is the
    // pivot for every timestamp rule in whisper_process_logits(). A vocab with eot+sot
    // but no `<|0.00|>` taking that branch would leave token_beg at the English default
    // 50363 with no `dt` shift applied — worse than either path taken whole.
    Serialized s;
    s.eot = 8631;
    s.sot = 8632;
    REQUIRE_FALSE(use_serialized(s));
}

TEST_CASE("any missing special falls back to legacy", "[unit][whisper]") {
    Serialized no_eot;
    no_eot.sot = 8632;
    no_eot.beg = 8739;
    REQUIRE_FALSE(use_serialized(no_eot));

    Serialized no_sot;
    no_sot.eot = 8631;
    no_sot.beg = 8739;
    REQUIRE_FALSE(use_serialized(no_sot));
}

TEST_CASE("token id 0 counts as present", "[unit][whisper]") {
    // Only kAbsent means "not there". A predicate written as a truthiness test
    // instead of an explicit sentinel would wrongly reject this vocab.
    Serialized s;
    s.eot = 0;
    s.sot = 1;
    s.beg = 2;
    REQUIRE(use_serialized(s));
}

TEST_CASE("the legacy fixup arithmetic that #322 corrupted", "[unit][whisper]") {
    // Replays the shift so the numbers quoted in the comments are checked rather
    // than merely asserted in prose. Defaults are the English layout; a multilingual
    // model shifts both by one.
    const int default_eot = 50256, default_sot = 50257;

    SECTION("correct path: nothing pre-assigned") {
        int eot = default_eot, sot = default_sot;
        eot++;
        sot++;
        REQUIRE(eot == 50257);
        REQUIRE(sot == 50258);
        REQUIRE(eot != sot);
    }

    SECTION("old code: the probe had already written eot") {
        int eot = 50257; // <-- the stray assignment from the serialized <|endoftext|>
        int sot = default_sot;
        eot++;
        sot++;
        REQUIRE(eot == 50258);
        REQUIRE(sot == 50258);
        // eot aliasing sot is what broke EOT detection: sot is force-suppressed by
        // `logits[token_sot] = -INFINITY`, so the id read as "EOT" can never be
        // sampled, and the timestamp rule's `[0, token_eot)` mask now covers the
        // real EOT at 50257.
        REQUIRE(eot == sot);
    }
}
