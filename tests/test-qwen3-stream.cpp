// test-qwen3-stream.cpp — core/qwen3_stream.h against the upstream Python.
//
// tests/fixtures/qwen3_stream_cases.json is produced by
// tools/gen_qwen3_stream_cases.py, which runs the real qwen_asr and
// Confucius4-R2T2 helper functions. Every ported helper must reproduce them
// exactly, and the streaming state machine must follow the reference's
// control flow on a scripted model.

#include <catch2/catch_test_macros.hpp>

#include "core/qwen3_stream.h"

#include "json.hpp"

#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>

using namespace core_qwen3_stream;

static std::u32string U32(const std::string& s) {
    std::u32string o;
    REQUIRE(utf8_decode(s, o));
    return o;
}

static nlohmann::json load_cases() {
    const char* dir = std::getenv("CRISPASR_TEST_FIXTURES");
    const std::string path = std::string(dir ? dir : CRISPASR_TEST_FIXTURES_DIR) + "/qwen3_stream_cases.json";
    std::ifstream f(path);
    REQUIRE(f.good());
    return nlohmann::json::parse(f);
}

TEST_CASE("qwen3_stream: helpers match qwen_asr / R2T2 Python", "[qwen3-stream]") {
    const auto cases = load_cases();
    REQUIRE(cases.size() >= 10);
    for (const auto& c : cases) {
        const std::string in = c["in"].get<std::string>();
        INFO("input: " << in);
        const std::u32string s = U32(in);
        CHECK(utf8_encode(detect_and_fix_repetitions(s)) == c["repfix"].get<std::string>());
        {
            auto r = parse_asr_output(s, U"");
            CHECK(utf8_encode(r.language) == c["parse_asr"][0].get<std::string>());
            CHECK(utf8_encode(r.text) == c["parse_asr"][1].get<std::string>());
        }
        {
            auto r = parse_asr_output(s, U"English");
            CHECK(utf8_encode(r.language) == c["parse_asr_forced_en"][0].get<std::string>());
            CHECK(utf8_encode(r.text) == c["parse_asr_forced_en"][1].get<std::string>());
        }
        {
            auto r = parse_language_output(s, U"");
            CHECK(utf8_encode(r.language) == c["parse_lang"][0].get<std::string>());
            CHECK(utf8_encode(r.text) == c["parse_lang"][1].get<std::string>());
        }
        {
            auto r = parse_language_output(s, U"English");
            CHECK(utf8_encode(r.language) == c["parse_lang_forced_en"][0].get<std::string>());
            CHECK(utf8_encode(r.text) == c["parse_lang_forced_en"][1].get<std::string>());
        }
        CHECK(utf8_encode(normalize_punct_by_context(s)) == c["punct"].get<std::string>());
        CHECK(utf8_encode(strip_cjk_inner_spaces(s)) == c["cjk"].get<std::string>());
    }
}

TEST_CASE("qwen3_stream: utf8 replace decoding mirrors errors='replace'", "[qwen3-stream]") {
    // "中" is E4 B8 AD; a token boundary can cut it.
    CHECK(utf8_decode_replace("a\xE4\xB8") == U"a\uFFFD\uFFFD");
    CHECK(utf8_decode_replace("a\xE4\xB8\xAD") == U"a\u4E2D");
}

// A character-level fake tokenizer: one token per UTF-8 byte, so rollback of k
// tokens can land inside a multi-byte character — the case the reference's
// U+FFFD loop exists for.
static Hooks byte_hooks(std::vector<std::string> scripted, std::vector<std::string>* prefixes) {
    Hooks h;
    h.encode = [](const std::string& s) {
        std::vector<int32_t> ids;
        for (unsigned char c : s)
            ids.push_back(c);
        return ids;
    };
    h.decode_bytes = [](const std::vector<int32_t>& ids) {
        std::string s;
        for (int32_t id : ids)
            s.push_back((char)id);
        return s;
    };
    auto idx = std::make_shared<size_t>(0);
    h.generate = [scripted, prefixes, idx](const std::vector<float>&, const std::string& prefix, int) {
        prefixes->push_back(prefix);
        return (*idx < scripted.size()) ? scripted[(*idx)++] : std::string();
    };
    return h;
}

TEST_CASE("qwen3_stream: prefix rollback and chunk counting", "[qwen3-stream]") {
    std::vector<std::string> prefixes;
    // chunk 1: language tag + "Hel"; chunk 2 continues; chunk 3 is CJK.
    Hooks h = byte_hooks({"language English<asr_text>Hel", "llo world", "\xE4\xB8\xAD"}, &prefixes);
    State st;
    st.cfg.unfixed_chunk_num = 0;
    st.cfg.unfixed_token_num = 1;
    float audio[4] = {0, 0, 0, 0};

    step(st, h, audio, 4, 4);
    CHECK(prefixes[0].empty()); // nothing decoded yet
    CHECK(st.text == U"Hel");
    CHECK(st.chunk_id == 1);

    step(st, h, audio, 4, 4);
    // previous raw "language English<asr_text>Hel" minus one byte-token
    CHECK(prefixes[1] == "language English<asr_text>He");
    CHECK(st.text == U"Hello world");

    step(st, h, audio, 4, 4);
    CHECK(prefixes[2] == "language English<asr_text>Hello worl");
    CHECK(st.audio_accum.size() == 12);
}

TEST_CASE("qwen3_stream: untagged output without forced language does not advance", "[qwen3-stream]") {
    std::vector<std::string> prefixes;
    Hooks h = byte_hooks({"no tag"}, &prefixes);
    State st;
    float a[2] = {0, 0};
    auto r = step(st, h, a, 2, 4);
    CHECK_FALSE(r.counted);
    CHECK(st.chunk_id == 0);
    CHECK(st.text.empty());
}

TEST_CASE("qwen3_stream: rollback never leaves a split character", "[qwen3-stream]") {
    std::vector<std::string> prefixes;
    // raw ends in a 3-byte CJK char; rolling back 1 byte-token would split it,
    // so the loop must roll back all three bytes.
    Hooks h = byte_hooks({"language Chinese<asr_text>\xE4\xB8\xAD\xE6\x96\x87", "x"}, &prefixes);
    State st;
    float a[1] = {0};
    step(st, h, a, 1, 4);
    step(st, h, a, 1, 4);
    CHECK(prefixes[1] == "language Chinese<asr_text>\xE4\xB8\xAD");
}

TEST_CASE("qwen3_stream: example.py max_new_tokens schedule", "[qwen3-stream]") {
    Schedule s;
    s.init(2560, 2560); // 160 ms step + 160 ms lookahead
    CHECK(s.next_chunk_samples() == 5120);
    CHECK(s.max_new == 4);
    CHECK(s.first_max_new == 4);
    s.update(U"ab"); // grew: reset to step/1280 = 2
    CHECK(s.max_new == 2);
    CHECK(s.next_chunk_samples() == 2560);
    s.update(U"ab"); // no growth: +1
    CHECK(s.max_new == 3);
    s.update(U"ab");
    s.update(U"ab"); // capped at min(32, max(4, 2*2)) = 4
    CHECK(s.max_new == 4);
}
