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

TEST_CASE("Qwen3 context preserves raw bytes and forced-language prefill", "[unit][asr-prompt]") {
    const auto prompt =
        core_qwen3_prompt::build(2, "", "", "language English<asr_text>", "", "  产品: CrispASR\nSpeaker: Zoë\t \n");
    REQUIRE(prompt == "<|im_start|>system\n  产品: CrispASR\nSpeaker: Zoë\t \n<|im_end|>\n"
                      "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_pad|><|audio_end|>"
                      "<|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>");
}

TEST_CASE("Qwen3 context preserves automatic language selection", "[unit][asr-prompt]") {
    const auto prompt = core_qwen3_prompt::build(1, "", "", "", "", "The speaker works on CrispASR.");
    REQUIRE(prompt == "<|im_start|>system\nThe speaker works on CrispASR.<|im_end|>\n"
                      "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
                      "<|im_end|>\n<|im_start|>assistant\n");
}

TEST_CASE("Qwen3 context leaves explicit user questions intact", "[unit][asr-prompt]") {
    const auto prompt = core_qwen3_prompt::build(1, "", "What was said?", "", "", "The speaker works on CrispASR.");
    REQUIRE(prompt == "<|im_start|>system\nThe speaker works on CrispASR.<|im_end|>\n"
                      "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>\nWhat was said?"
                      "<|im_end|>\n<|im_start|>assistant\n");
}

TEST_CASE("Qwen3 empty context preserves default and legacy prompts", "[unit][asr-prompt]") {
    const std::string expected = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
                                 "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
                                 "<|im_start|>assistant\nlanguage English<asr_text>";
    REQUIRE(core_qwen3_prompt::build(1, "", "", "language English<asr_text>", "") == expected);
    REQUIRE(core_qwen3_prompt::build(1, "", "", "language English<asr_text>", "", "") == expected);
    const std::string legacy =
        "<|im_start|>system\nTranscribe the speech in English. The following words may appear in the audio: CrispASR."
        "<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
        "<|im_end|>\n<|im_start|>assistant\n";
    REQUIRE(core_qwen3_prompt::build(1, "Transcribe the speech in English.", "", "", "CrispASR") == legacy);
    REQUIRE(core_qwen3_prompt::build(1, "Transcribe the speech in English.", "", "", "CrispASR", "") == legacy);
}

TEST_CASE("Qwen3 context composes with legacy instructions and wrapped hotwords", "[unit][asr-prompt]") {
    const std::string expected = "<|im_start|>system\nTranscribe the speech in English. The speaker is Zoë. "
                                 "The following words may appear in the audio: CrispASR, ggml.<|im_end|>\n"
                                 "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
                                 "<|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>";
    for (const char* system : {"Transcribe the speech in English.", "Transcribe the speech in English. "}) {
        REQUIRE(core_qwen3_prompt::build(1, system, "", "language English<asr_text>", "CrispASR, ggml",
                                         "The speaker is Zoë.") == expected);
    }
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
