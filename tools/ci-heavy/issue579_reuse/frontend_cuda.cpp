// Optional research STFT using NVIDIA's public cuFFT/cudart host APIs.
#include "../../../src/core/reuse_frontend_cuda.h"
#include "../../../src/core/reuse_frontend_mkl.h"
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cstdio>
#include <stdexcept>

namespace reuse_frontend_cuda {
namespace {
constexpr int fft_size = 640, hop = 80, bins = 321;
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
void check(cufftResult status, const char* operation) {
    if (status != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuFFT status " + std::to_string(status));
}
struct Buffer {
    void* data = nullptr;
    explicit Buffer(size_t bytes) { check(cudaMalloc(&data, bytes), "cudaMalloc"); }
    ~Buffer() {
        if (data)
            cudaFree(data);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void release() {
        check(cudaFree(data), "cudaFree");
        data = nullptr;
    }
};
struct Plan {
    cufftHandle handle = 0;
    Plan() = default;
    ~Plan() {
        if (handle)
            cufftDestroy(handle);
    }
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;
    void release() {
        check(cufftDestroy(handle), "cufftDestroy");
        handle = 0;
    }
};
} // namespace

bool available(std::string& reason) {
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0) {
        reason = "CUDA device unavailable";
        if (status != cudaSuccess)
            reason += std::string(": ") + cudaGetErrorString(status);
        return false;
    }
    reason.clear();
    return true;
}

std::string version() {
    int runtime = 0, driver = 0, fft = 0, device = 0;
    check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
    check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    check(cufftGetVersion(&fft), "cufftGetVersion");
    check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties;
    check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    return "cuFFT " + std::to_string(fft) + "; CUDA runtime " + std::to_string(runtime) + "; driver " +
           std::to_string(driver) + "; device " + properties.name + "; compute " + std::to_string(properties.major) +
           "." + std::to_string(properties.minor) + "; CPU Hann " + reuse_frontend_mkl::version();
}

Analysis transform(const std::vector<float>& audio) {
    if (audio.size() < 400 || audio.size() > 16000 || audio.size() % hop)
        throw std::runtime_error("input requires 400..16000 samples, divisible by 80");
    for (float value : audio)
        if (!std::isfinite(value))
            throw std::runtime_error("input contains nonfinite audio");
    std::string reason;
    if (!available(reason))
        throw std::runtime_error(reason);
    const size_t frames = 1 + audio.size() / hop, count = frames * bins;
    Analysis result;
    // Upstream creates the float32 Hann on the CPU before copying it to CUDA.
    result.window = reuse_frontend_mkl::periodic_window();
    std::vector<float> input(frames * fft_size);
    for (size_t frame = 0; frame < frames; ++frame) {
        for (int i = 0; i < fft_size; ++i) {
            int64_t index = static_cast<int64_t>(frame * hop) + i - fft_size / 2;
            if (index < 0)
                index = -index;
            else if (index >= static_cast<int64_t>(audio.size()))
                index = 2 * static_cast<int64_t>(audio.size()) - 2 - index;
            input[frame * fft_size + i] = audio[index] * result.window[i];
        }
    }
    Buffer device_input(input.size() * sizeof(float)), device_output(count * sizeof(cufftComplex));
    check(cudaMemcpy(device_input.data, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice),
          "cudaMemcpy input");
    Plan plan;
    int shape[] = {fft_size};
    // Explicit embeds select contiguous time-major frames and Hermitian bins.
    int input_embed[] = {fft_size}, output_embed[] = {bins};
    check(cufftPlanMany(&plan.handle, 1, shape, input_embed, 1, fft_size, output_embed, 1, bins, CUFFT_R2C,
                        static_cast<int>(frames)),
          "cufftPlanMany");
    check(cufftExecR2C(plan.handle, static_cast<cufftReal*>(device_input.data),
                       static_cast<cufftComplex*>(device_output.data)),
          "cufftExecR2C");
    std::vector<cufftComplex> output(count);
    // The blocking device-to-host copy also observes asynchronous FFT errors.
    check(cudaMemcpy(output.data(), device_output.data, output.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost),
          "cudaMemcpy output");
    plan.release();
    device_output.release();
    device_input.release();
    result.real.resize(count);
    result.imaginary.resize(count);
    result.magnitude.resize(count);
    result.phase.resize(count);
    for (size_t i = 0; i < count; ++i) {
        result.real[i] = output[i].x;
        result.imaginary[i] = output[i].y;
        result.magnitude[i] = std::log1p(std::hypot(output[i].x, output[i].y));
        result.phase[i] = std::atan2(output[i].y, output[i].x);
    }
    return result;
}

bool analyze(const std::vector<float>& audio, int sample_rate, reuse_frontend::Spectrum& output,
             reuse_frontend::Diagnostics* diagnostics) {
    if (sample_rate != 16000)
        return false;
    try {
        auto values = transform(audio);
        reuse_frontend::Spectrum result;
        result.sample_rate = sample_rate;
        result.n_fft = fft_size;
        result.hop = hop;
        result.frames = 1 + audio.size() / hop;
        result.bins = bins;
        result.magnitude = std::move(values.magnitude);
        result.phase = std::move(values.phase);
        if (diagnostics) {
            reuse_frontend::Diagnostics observations;
            observations.window = std::move(values.window);
            observations.real = std::move(values.real);
            observations.imaginary = std::move(values.imaginary);
            *diagnostics = std::move(observations);
        }
        output = std::move(result);
        return true;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "reuse: cuFFT frontend failed: %s\n", error.what());
        return false;
    }
}
} // namespace reuse_frontend_cuda
