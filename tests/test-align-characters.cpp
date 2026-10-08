// Known CTC paths, independent of model weights: timings come from label states.
#include "align.h"
#include "core/align_json.h"
#include "core/align_labels.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static int checks = 0;
static void check(bool ok, const char* message) {
    ++checks;
    if (!ok)
        throw std::runtime_error(message);
}
static void near(float actual, float expected) {
    check(std::abs(actual - expected) < 1e-6f, "frame timestamp mismatch");
}
static std::vector<ctc_word_stamp> align(const std::vector<int>& path, const std::vector<std::string>& vocab,
                                         const std::vector<std::string>& words) {
    std::vector<float> logits(path.size() * vocab.size(), -20.f);
    for (size_t t = 0; t < path.size(); ++t)
        logits[t * vocab.size() + path[t]] = 20.f;
    return ctc_forced_align(logits.data(), path.size(), vocab.size(), words, vocab, 0, .02f);
}
int main() {
    try {
        auto words = align({0, 1, 1, 0, 2, 0, 3, 3, 0, 3, 4, 4, 4, 0}, {"<pad>", "h", "e", "l", "o", "|"}, {"Hello!"});
        check(words.size() == 1, "one word expected");
        check(words[0].word == "Hello!", "original display text lost");
        const auto& chars = words[0].characters;
        check(chars.size() == 5, "punctuation must not create a timing");
        const char* labels[] = {"H", "e", "l", "l", "o"};
        const int starts[] = {1, 4, 6, 9, 10}, ends[] = {3, 5, 8, 10, 13};
        for (size_t i = 0; i < chars.size(); ++i) {
            check(chars[i].character == labels[i], "repeated label/order/case mismatch");
            near(chars[i].t0, starts[i] * .02f);
            near(chars[i].t1, ends[i] * .02f);
        }
        near(words[0].t0, .02f);
        near(words[0].t1, .26f);
        const auto separated = align({1, 0, 5, 0, 2}, {"<pad>", "h", "e", "l", "o", "|"}, {"h", "e"});
        check(separated.size() == 2, "word boundary lost");
        check(separated[0].characters.size() == 1 && separated[1].characters.size() == 1,
              "separator must not appear as a character");
        near(separated[1].characters[0].t0, .08f);
        const float valid[] = {0, 1};
        check(ctc_forced_align(nullptr, 1, 2, {"a"}, {"<pad>", "a"}, 0, .02f).empty(), "null logits accepted");
        check(ctc_forced_align(valid, 1, 2, {"a"}, {"<pad>", "a"}, 2, .02f).empty(), "invalid blank accepted");
        auto ar = align({0, 1, 1, 0, 2, 0, 1, 0, 1, 0}, {"<pad>", "ب", "َ", "|"}, {"بَبب!"});
        check(ar.size() == 1 && ar[0].characters.size() == 4, "Arabic codepoints/diacritic lost");
        check(ar[0].characters[1].character == "َ", "supported diacritic must be its own label");
        near(ar[0].characters[2].t0, .12f);
        near(ar[0].characters[3].t0, .16f);
        auto oov = align({1, 0, 2}, {"<pad>", "h", "e"}, {"h☃e"});
        check(oov[0].characters.size() == 2, "OOV codepoint must not get invented time");
        auto none = align({0, 0}, {"<pad>", "h"}, {"☃"});
        check(none.size() == 1 && none[0].characters.empty(), "OOV word contract changed");
        near(none[0].t0, 0);
        near(none[0].t1, 0);
        check(align({1, 1}, {"<pad>", "a"}, {"aa"}).empty(), "impossible repeated-label path accepted");
        check(core_align_labels::for_vocab("بب", {"<pad>", "ب"}, true) == "بب", "Arabic vocabulary romanized");
        check(core_align_labels::for_vocab("викинги", {"<pad>", "v", "i"}, true) == "vikingi", "Latin fallback lost");
        check(core_align_labels::for_vocab("بب", {"<pad>", "a"}, false) == "بب", "romanization override ignored");
        CrispasrAlignedWord word{"H\"\n", 120, 165, {{"H", 120, 130}, {"\"", 130, 147}}};
        const auto json = nlohmann::json::parse(core_align_json::word(word).dump());
        check(json["word"] == word.text, "JSON escaping broken");
        check(json["characters"][0]["char"] == "H", "JSON character missing");
        check(json["characters"][0]["start"] == 1.2, "offset/unit conversion broken");
        check(!core_align_json::word({"word", 0, 0, {}}).contains("characters"), "unmeasured spans invented");
        std::printf("CTC_CHARACTER_ALIGNMENT_PASS: %d assertions\n", checks);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
