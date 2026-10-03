#include <catch2/catch_test_macros.hpp>
#include "core/mimo_prompt.h"
#include "core/qwen3_prompt.h"

TEST_CASE("Qwen3 hotwords preserve forced-language transcription", "[unit][asr-prompt]") {
    const auto prompt = core_qwen3_prompt::build(2, "", "", "language English<asr_text>", "Hello, CrispASR");
    REQUIRE(prompt == "<|im_start|>system\nThe following words may appear in the audio: Hello, CrispASR."
                      "<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_pad|>"
                      "<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>");
    const auto cleared = core_qwen3_prompt::build(2, "", "", "language English<asr_text>", "");
    REQUIRE(cleared.find("The following words") == std::string::npos);
    REQUIRE(cleared.substr(cleared.size() - std::string("language English<asr_text>").size()) ==
            "language English<asr_text>");
}

TEST_CASE("Qwen3 hotwords leave explicit questions and auto language intact", "[unit][asr-prompt]") {
    const auto question = core_qwen3_prompt::build(1, "", "What was said?", "", "CrispASR");
    REQUIRE(question.find("<|audio_end|>\nWhat was said?<|im_end|>") != std::string::npos);
    REQUIRE(question.substr(question.size() - std::string("<|im_start|>assistant\n").size()) ==
            "<|im_start|>assistant\n");
    const auto automatic = core_qwen3_prompt::build(1, "", "", "", "CrispASR");
    REQUIRE(automatic.substr(automatic.size() - std::string("<|im_start|>assistant\n").size()) ==
            "<|im_start|>assistant\n");
    const auto legacy = core_qwen3_prompt::build(1, "Transcribe the speech in English.", "", "", "CrispASR");
    REQUIRE(legacy.find("English. The following words") != std::string::npos);
}

TEST_CASE("MiMo language bias and user instruction are independent", "[unit][asr-prompt]") {
    REQUIRE(core_mimo_prompt::instruction("en", "") == "Please transcribe this audio file");
    REQUIRE(core_mimo_prompt::instruction("zh", "") == "请将这段语音转换为文字");
    REQUIRE(core_mimo_prompt::instruction("auto", "") == "Please transcribe this audio file");
    REQUIRE(core_mimo_prompt::instruction("zh", "Custom instruction") == "Custom instruction");
    REQUIRE(std::string(core_mimo_prompt::language_tag("en")) == "<english>");
    REQUIRE(std::string(core_mimo_prompt::language_tag("zh")) == "<chinese>");
    for (const char* value : {static_cast<const char*>(nullptr), "", "auto", "de"})
        REQUIRE(std::string(core_mimo_prompt::language_tag(value)).empty());
}
