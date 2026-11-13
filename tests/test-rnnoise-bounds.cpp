// Compile the production entry point with small deterministic link stubs.
// Boundary probes stop at resampler init, before any large buffer allocation.
#include "../src/crispasr_enhance.cpp"

#include <climits>
#include <cstdlib>

namespace {
int checks = 0;
int init_calls = 0;
int frame_calls = 0;
bool fail_init = true;

void check(bool condition) {
    ++checks;
    if (!condition) {
        std::fprintf(stderr, "RNNoise bounds check %d failed\n", checks);
        std::exit(1);
    }
}
} // namespace

ma_resampler_config ma_resampler_config_init(ma_format format, ma_uint32 channels, ma_uint32 input_rate,
                                             ma_uint32 output_rate, ma_resample_algorithm algorithm) {
    ma_resampler_config result = {};
    result.format = format;
    result.channels = channels;
    result.sampleRateIn = input_rate;
    result.sampleRateOut = output_rate;
    result.algorithm = algorithm;
    return result;
}

ma_result ma_resampler_init(const ma_resampler_config* config, const ma_allocation_callbacks*, ma_resampler* state) {
    ++init_calls;
    state->sampleRateIn = config->sampleRateIn;
    return fail_init ? MA_ERROR : MA_SUCCESS;
}

void ma_resampler_uninit(ma_resampler*, const ma_allocation_callbacks*) {}

ma_result ma_resampler_process_pcm_frames(ma_resampler* state, const void* input, ma_uint64* input_count, void* output,
                                          ma_uint64* output_count) {
    const auto* in = static_cast<const float*>(input);
    auto* out = static_cast<float*>(output);
    const bool up = state->sampleRateIn == 16000;
    const ma_uint64 count = std::min(*output_count, up ? *input_count * 3 : *input_count / 3);
    for (ma_uint64 i = 0; i < count; ++i)
        out[i] = in[up ? i / 3 : i * 3];
    *output_count = count;
    return MA_SUCCESS;
}

struct DenoiseState {};
DenoiseState* rnnoise_create(RNNModel*) {
    static DenoiseState state;
    return &state;
}
void rnnoise_destroy(DenoiseState*) {}
float rnnoise_process_frame(DenoiseState*, float* output, const float* input) {
    ++frame_calls;
    std::memmove(output, input, 480 * sizeof(float));
    return 0;
}

int main() {
    float input = 0.25f;
    float output = 123.0f;
    CrispasrEnhanceOptions options;
    for (int count : {INT_MAX, INT_MAX - 1, INT_MAX / 3}) {
        init_calls = 0;
        check(!crispasr_enhance_audio(&input, count, &output, options));
        check(init_calls == 0);
        check(output == 123.0f);
    }
    const int maximum = rnnoise_max_input_samples(std::vector<float>().max_size());
    for (int count : {maximum, maximum + 1}) {
        init_calls = 0;
        check(!crispasr_enhance_audio(&input, count, &output, options));
        check(init_calls == (count == maximum ? 1 : 0));
        check(output == 123.0f);
    }
    // Exercise the production capacity calculation with 32-bit byte and
    // ptrdiff_t limits as well as the host's actual vector max_size.
    for (size_t capacity :
         {size_t(UINT32_MAX / sizeof(float)), size_t(INT32_MAX / sizeof(float)), std::vector<float>().max_size()}) {
        const int limit = rnnoise_max_input_samples(capacity);
        const uint64_t worst_count = uint64_t(limit) * 3 + 480;
        const uint64_t padding_sum = worst_count + 479;
        const uint64_t padded = (padding_sum / 480) * 480;
        check(worst_count <= INT_MAX);
        check(padding_sum <= INT_MAX);
        check(padded <= capacity);
        check(padded <= INT_MAX);
        check(uint64_t(limit) + 64 <= capacity);
        check(padding_sum + 3 > std::min(uint64_t(capacity), uint64_t(INT_MAX)));
    }
    check(rnnoise_max_input_samples(0) == 0);
    check(rnnoise_max_input_samples(958) == 0);
    for (int count : {0, -1}) {
        init_calls = 0;
        check(!crispasr_enhance_audio(&input, count, &output, options));
        check(init_calls == 0);
    }
    fail_init = false;
    for (int count : {1, 159, 160, 161, 321}) {
        std::vector<float> in(count, input), out(count + 1, 123.0f);
        frame_calls = 0;
        check(crispasr_enhance_audio(in.data(), count, out.data(), options));
        check(frame_calls == (3 * count + 479) / 480);
        check(std::equal(in.begin(), in.end(), out.begin()));
        check(out[count] == 123.0f);
    }
    std::printf("RNNoise bounds: %d checks passed\n", checks);
}
