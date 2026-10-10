#include "core/omnivoice_audio.h"
#include <cstring>
extern "C" {
int clean(const float* input, int n, int mid, int lead, int trail, float* output, int capacity) {
    auto result = core_omnivoice_audio::remove_silence(std::vector<float>(input, input + n), mid, lead, trail);
    if ((int)result.size() > capacity)
        return -1;
    std::copy(result.begin(), result.end(), output);
    return (int)result.size();
}
int fade(const float* input, int n, float pad, float duration, float* output, int capacity) {
    std::vector<float> result(input, input + n);
    core_omnivoice_audio::fade_and_pad(result, pad, duration);
    if ((int)result.size() > capacity)
        return -1;
    std::copy(result.begin(), result.end(), output);
    return (int)result.size();
}
int punctuate(const char* input, char* output, int capacity) {
    auto text = core_omnivoice_audio::add_punctuation(input);
    if ((int)text.size() + 1 > capacity)
        return -1;
    std::memcpy(output, text.c_str(), text.size() + 1);
    return (int)text.size();
}
}
