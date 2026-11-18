// #589: accepting a speech buffer or an empty string is not an ASR success.
// Require recognizable JFK words from an actual ASR checkpoint.
#include "vibevoice.h"
#include "core/audio_resample.h"
#include "core/wav_reader.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const char* model = std::getenv("CRISPASR_MODEL_VIBEVOICE_ASR");
    if (!model || !*model) {
        std::fprintf(stderr, "SKIP: set CRISPASR_MODEL_VIBEVOICE_ASR to a VibeVoice-ASR GGUF\n");
        return 77;
    }
    std::error_code ec;
    if (!std::filesystem::is_regular_file(model, ec)) {
        std::fprintf(stderr, "FAIL: configured VibeVoice-ASR model does not exist: %s\n", model);
        return 1;
    }
    if (argc != 2)
        return 1;
    std::vector<float> samples;
    int sr = 0;
    if (!crispasr::core::read_wav_mono_pcm16(argv[1], samples, sr))
        return 1;
    samples = core_audio::resample_polyphase(samples.data(), (int)samples.size(), sr, 24000);
    if (samples.empty())
        return 1;
    auto params = vibevoice_context_default_params();
    const char* gpu = std::getenv("CRISPASR_TEST_GPU");
    params.use_gpu = gpu && std::string(gpu) == "1";
    params.max_new_tokens = 192;
    std::unique_ptr<vibevoice_context, decltype(&vibevoice_free)> ctx(vibevoice_init_from_file(model, params),
                                                                      vibevoice_free);
    if (!ctx || !vibevoice_has_asr(ctx.get()))
        return 1;
    std::unique_ptr<vibevoice_result, decltype(&vibevoice_result_free)> result(
        vibevoice_transcribe_with_probs(ctx.get(), samples.data(), (int)samples.size()), vibevoice_result_free);
    if (!result || !result->text || !*result->text || result->n_tokens <= 0)
        return 1;
    std::printf("JFK transcript: %s\n", result->text);
    std::string words;
    for (const unsigned char c : std::string(result->text)) {
        if (std::isalnum(c))
            words += (char)std::tolower(c);
        else if (!words.empty() && words.back() != ' ')
            words += ' ';
    }
    if (words.find("ask not what your country can do for you") == std::string::npos ||
        words.find("ask what you can do for your country") == std::string::npos) {
        std::fprintf(stderr, "FAIL: JFK speech was empty, special tokens, or unrecognizable\n");
        return 1;
    }
    std::printf("PASS: nonempty recognizable JFK (%d generated tokens)\n", result->n_tokens);
    return 0;
}
