#include "../src/qwen3_asr.h"
#include "../src/core/wav_reader.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    const char* one[] = {"test"};
    int64_t start = 0, end = 0;
    float hidden = 0;
    if (qwen3_asr_align_words_with_hidden(nullptr, nullptr, 0, one, 1, &start, &end, &hidden, 1) != -1 ||
        qwen3_asr_align_words_with_hidden(nullptr, nullptr, 0, one, 1, &start, &end, nullptr, 0) != -1)
        return 1;
    if (argc == 1) {
        std::puts("Ultra hidden-state input controls passed");
        return 0;
    }
    if (argc != 4)
        return 2;
    std::vector<float> audio;
    int sample_rate = 0;
    if (!crispasr::core::read_wav_mono_pcm16(argv[2], audio, sample_rate) || sample_rate != 16000)
        return 3;
    auto params = qwen3_asr_context_default_params();
    params.n_threads = 2;
    params.use_gpu = false;
    params.verbosity = 0;
    auto* ctx = qwen3_asr_init_from_file(argv[1], params);
    if (!ctx)
        return 4;
    const char* words[] = {"And",  "so",      "my",  "fellow", "Americans", "ask",    "not", "what",
                           "your", "country", "can", "do",     "for",       "you",    "ask", "what",
                           "you",  "can",     "do",  "for",    "your",      "country"};
    constexpr int count = sizeof(words) / sizeof(words[0]);
    std::vector<int64_t> base_start(count), base_end(count), capture_start(count), capture_end(count);
    std::vector<float> states(count * 2 * 1024);
    const int small_rc =
        qwen3_asr_align_words_with_hidden(ctx, audio.data(), static_cast<int>(audio.size()), words, count,
                                          capture_start.data(), capture_end.data(), states.data(), states.size() - 1);
    const int base_rc = qwen3_asr_align_words(ctx, audio.data(), static_cast<int>(audio.size()), words, count,
                                              base_start.data(), base_end.data());
    const int capture_rc =
        qwen3_asr_align_words_with_hidden(ctx, audio.data(), static_cast<int>(audio.size()), words, count,
                                          capture_start.data(), capture_end.data(), states.data(), states.size());
    qwen3_asr_free(ctx);
    if (small_rc != -1 || base_rc != 0 || capture_rc != 0 || base_start != capture_start || base_end != capture_end)
        return 5;
    FILE* output = std::fopen(argv[3], "wb");
    if (!output)
        return 6;
    const bool ok = std::fwrite(states.data(), sizeof(float), states.size(), output) == states.size();
    std::fclose(output);
    if (!ok)
        return 7;
    std::printf("ULTRA_HIDDEN_CAPTURE_PASS %zu values; uncaptured timestamps unchanged\n", states.size());
    return 0;
}
