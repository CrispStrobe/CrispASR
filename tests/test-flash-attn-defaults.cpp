// test-flash-attn-defaults.cpp — verifies every backend's
// *_context_default_params() returns flash_attn=true after the
// PLAN #89 field-migration sweep (May 2026).
//
// Two reasons for this test:
//   1. catch accidental drops of the `p.flash_attn = true;` line
//      in default_params() impls — easy to miss in a clang-format
//      sweep or merge conflict, and the rest of the build won't
//      notice because uninitialised `bool` reads as `false` which
//      is also a valid configuration.
//   2. document the per-backend coverage. Each section is guarded
//      by `__has_include("backend.h")` so the test auto-skips
//      backends that aren't built into the current libcrispasr
//      slice. A backend that gets disabled in CMake silently drops
//      its section rather than failing.
//
// Cross-platform note: every backend listed here has `use_gpu` in
// its context_params on every supported platform — Linux / Windows
// / macOS / iOS all see the same field layout, so the test runs
// uniformly across CI. The MTLBinaryArchive-specific smoke test
// lives elsewhere (Apple-only; see test-metal-pipeline-cache.mm).

#include <catch2/catch_test_macros.hpp>

// NOTE on use_flash vs flash_attn naming:
//
// parakeet / canary / cohere shipped a pre-existing `use_flash`
// field in their context_params before the PLAN #89 sweep. Those
// structs default `use_flash` to FALSE (per the upstream-derived
// defaults) — the session-API runtime override
// (`g_open_flash_attn_tls` → `p.use_flash = ...` in
// `crispasr_session_open_explicit`) is what enables flash-attn at
// session-open time. So for these three the per-backend default
// is correctly FALSE; the test pins that fact rather than
// imposing a uniform "always true" expectation that would mask
// an upstream behaviour change.
//
// Every backend without a pre-existing field got a fresh
// `flash_attn` field defaulting to TRUE (see below).

#if __has_include("parakeet.h")
#include "parakeet.h"
TEST_CASE("flash-attn defaults: parakeet use_flash=false (upstream)", "[unit][flash_attn]") {
    auto p = parakeet_context_default_params();
    REQUIRE(p.use_flash == false);
}
#endif

#if __has_include("canary.h")
#include "canary.h"
TEST_CASE("flash-attn defaults: canary use_flash=false (upstream)", "[unit][flash_attn]") {
    auto p = canary_context_default_params();
    REQUIRE(p.use_flash == false);
}
#endif

#if __has_include("cohere.h")
#include "cohere.h"
TEST_CASE("flash-attn defaults: cohere use_flash=false (upstream)", "[unit][flash_attn]") {
    auto p = cohere_context_default_params();
    REQUIRE(p.use_flash == false);
}
#endif

#if __has_include("qwen3_asr.h")
#include "qwen3_asr.h"
TEST_CASE("flash-attn defaults: qwen3-asr flash_attn=true", "[unit][flash_attn]") {
    auto p = qwen3_asr_context_default_params();
    REQUIRE(p.flash_attn == true);
}
#endif

#if __has_include("voxtral.h")
#include "voxtral.h"
TEST_CASE("flash-attn defaults: voxtral flash_attn=true", "[unit][flash_attn]") {
    auto p = voxtral_context_default_params();
    REQUIRE(p.flash_attn == true);
}
#endif

#if __has_include("voxtral4b.h")
#include "voxtral4b.h"
TEST_CASE("flash-attn defaults: voxtral4b flash_attn=true", "[unit][flash_attn]") {
    auto p = voxtral4b_context_default_params();
    REQUIRE(p.flash_attn == true);
}
#endif

#if __has_include("granite_speech.h")
#include "granite_speech.h"
TEST_CASE("flash-attn defaults: granite_speech flash_attn=true", "[unit][flash_attn]") {
    auto p = granite_speech_context_default_params();
    REQUIRE(p.flash_attn == true);
}
#endif

#if __has_include("vibevoice.h")
#include "vibevoice.h"
TEST_CASE("flash-attn defaults: vibevoice flash_attn=true + tts_steps=20", "[unit][flash_attn]") {
    auto p = vibevoice_context_default_params();
    REQUIRE(p.flash_attn == true);
    // PLAN #88 add-on: confirm tts_steps default survived the
    // round-6 wiring of the runtime setter.
    REQUIRE(p.tts_steps == 20);
}
#endif

#if __has_include("qwen3_tts.h")
#include "qwen3_tts.h"
TEST_CASE("flash-attn defaults: qwen3-tts flash_attn=true + temperature=0", "[unit][flash_attn]") {
    auto p = qwen3_tts_context_default_params();
    REQUIRE(p.flash_attn == true);
    // PLAN round 5: the code-predictor's sampler now reads
    // cparams.temperature instead of the hardcoded 0.9f. Default
    // is 0.0 = "use upstream qwen-tts reference value of 0.9",
    // honoured by the > 0 fallback in code_pred_generate_15.
    REQUIRE(p.temperature == 0.0f);
}
#endif

#if __has_include("orpheus.h")
#include "orpheus.h"
TEST_CASE("flash-attn defaults: orpheus flash_attn=true", "[unit][flash_attn]") {
    auto p = orpheus_context_default_params();
    REQUIRE(p.flash_attn == true);
}
#endif

#if __has_include("kokoro.h")
#include "kokoro.h"
TEST_CASE("flash-attn defaults: kokoro flash_attn=true + length_scale=1.0", "[unit][flash_attn]") {
    auto p = kokoro_context_default_params();
    REQUIRE(p.flash_attn == true);
    // PLAN #88: new length_scale field defaults to 1.0 (no-op).
    REQUIRE(p.length_scale == 1.0f);
}
#endif

#if __has_include("chatterbox.h")
#include "chatterbox.h"
TEST_CASE("flash-attn defaults: chatterbox flash_attn=true", "[unit][flash_attn]") {
    auto p = chatterbox_context_default_params();
    REQUIRE(p.flash_attn == true);
}
#endif
