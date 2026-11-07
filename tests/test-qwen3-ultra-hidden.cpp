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
        qwen3_asr_align_words_with_hidden(nullptr, nullptr, 0, one, 1, &start, &end, nullptr, 0) != -1 ||
        qwen3_asr_align_words_ultra(nullptr, nullptr, 0, one, 1, nullptr, nullptr, nullptr) != -1)
        return 1;
    if (argc == 1) {
        std::puts("Ultra hidden-state input controls passed");
        return 0;
    }
    if (argc != 4 && argc != 6)
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
    std::vector<double> refined_start(count), refined_end(count);
    int ultra_rc = 0;
    if (argc == 6)
        ultra_rc = qwen3_asr_align_words_ultra(ctx, audio.data(), static_cast<int>(audio.size()), words, count, argv[4],
                                               refined_start.data(), refined_end.data());
    qwen3_asr_free(ctx);
    if (ultra_rc != 0)
        return 8;
    if (argc == 6) {
        FILE* refined = std::fopen(argv[5], "wb");
        if (!refined)
            return 9;
        for (int i = 0; i < count; ++i) {
            if (std::fwrite(&refined_start[i], sizeof(double), 1, refined) != 1 ||
                std::fwrite(&refined_end[i], sizeof(double), 1, refined) != 1) {
                std::fclose(refined);
                return 10;
            }
        }
        std::fclose(refined);
        std::printf("ULTRA_REFINED_CAPTURE_PASS %d words\n", count);
    }
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
