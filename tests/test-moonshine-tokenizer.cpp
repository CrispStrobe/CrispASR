#include <catch2/catch_test_macros.hpp>
#include "moonshine-tokenizer.h"
#include <filesystem>
#include <fstream>
#include <chrono>

struct TokenizerFile {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("moonshine-tokenizer-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ~TokenizerFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};
TEST_CASE("Moonshine German JSON preserves byte fallback and added tokens", "[unit][moonshine-tokenizer]") {
    TokenizerFile file;
    std::ofstream(file.path)
        << R"({"model":{"vocab":{"<unk>":0,"<s>":1,"</s>":2,"▁Gr":3,"<0xC3>":4,"<0xBC>":5,"ße":6}},"added_tokens":[{"id":7,"content":"!"}]})";
    moonshine_tokenizer t;
    REQUIRE(t.load_json(file.path.string().c_str()));
    REQUIRE(t.vocab_size() == 8);
    REQUIRE(t.tokens_to_text({1, 3, 4, 5, 6, 7, 2}) == "Grüße!");
    REQUIRE(t.token_to_piece(1).empty());
    REQUIRE(t.token_to_piece(999).empty());
}
TEST_CASE("Moonshine binary tokenizer preserves byte fallback", "[unit][moonshine-tokenizer]") {
    TokenizerFile file;
    std::ofstream f(file.path, std::ios::binary);
    for (const auto& piece : {"<s>", "<0xC3>", "<0xA4>", "</s>"}) {
        const auto n = static_cast<char>(std::char_traits<char>::length(piece));
        f.write(&n, 1);
        f.write(piece, n);
    }
    f.close();
    moonshine_tokenizer t;
    REQUIRE(t.load(file.path.string().c_str()));
    REQUIRE(t.tokens_to_text({0, 1, 2, 3}) == "ä");
}
TEST_CASE("Moonshine tokenizer rejects malformed ids and truncated input", "[unit][moonshine-tokenizer]") {
    TokenizerFile file;
    std::ofstream(file.path) << R"({"model":{"vocab":{"bad":-1}}})";
    moonshine_tokenizer t;
    REQUIRE_FALSE(t.load_json(file.path.string().c_str()));
    std::ofstream(file.path, std::ios::binary) << '\x80';
    REQUIRE_FALSE(t.load(file.path.string().c_str()));
}
