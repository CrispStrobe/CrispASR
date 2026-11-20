// Opt-in reproduction of #598 through the production session C ABI.
#include "crispasr_session.h"
#include "core/gpu_backend_pref.h"
#include "core/wav_reader.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Wire layout of the opaque session-open options (ABI v2).
struct crispasr_open_params_v1 {
    int abi_version, n_threads, use_gpu, verbosity, flash_attn, n_gpu_layers;
    int reserved[6];
};

int main() {
    const char* model = std::getenv("CRISPASR_MODEL_XASR_Q8_0");
    if (!model || !*model || ggml_backend_cuda_get_device_count() == 0) {
        std::puts("SKIP: requires CRISPASR_MODEL_XASR_Q8_0 and a CUDA device; no runtime proof");
        return 4;
    }
    crispasr_set_gpu_backend_pref("cuda");
    auto* probe = crispasr_init_gpu_backend();
    if (!probe || !ggml_backend_is_cuda(probe)) {
        std::fputs("FAIL: production backend selection did not select CUDA\n", stderr);
        if (probe)
            ggml_backend_free(probe);
        return 1;
    }
    std::printf("CUDA device: %s\n", ggml_backend_name(probe));
    ggml_backend_free(probe);
    gguf_init_params gp{true, nullptr};
    auto* metadata = gguf_init_from_file(model, gp);
    if (!metadata)
        return 1;
    bool q8 = false;
    for (int64_t i = 0; i < gguf_get_n_tensors(metadata); ++i)
        q8 = q8 || gguf_get_tensor_type(metadata, i) == GGML_TYPE_Q8_0;
    gguf_free(metadata);
    if (!q8) {
        std::fputs("FAIL: xASR reproduction requires Q8_0 weights\n", stderr);
        return 1;
    }
    std::vector<float> pcm;
    int rate = 0;
    if (!crispasr::core::read_wav_mono_pcm16("samples/jfk.wav", pcm, rate) || rate != 16000)
        return 1;
    bool ok = true;
    for (int gpu : {0, 1}) {
        crispasr_open_params_v1 opts{2, 2, gpu, 1, 1, -1, {}};
        auto* session = crispasr_session_open_with_params(model, "xasr", &opts);
        if (!session)
            return 1;
        auto* result = crispasr_session_transcribe(session, pcm.data(), (int)pcm.size());
        std::string text;
        if (result) {
            for (int i = 0; i < crispasr_session_result_n_segments(result); ++i) {
                const char* segment = crispasr_session_result_segment_text(result, i);
                if (segment)
                    text += std::string(segment) + " ";
            }
            crispasr_session_result_free(result);
        }
        crispasr_session_close(session);
        std::printf("use_gpu=%d transcript: %s\n", gpu, text.c_str());
        std::string normalized;
        for (unsigned char c : text) {
            if (std::isalnum(c))
                normalized += (char)std::tolower(c);
        }
        // "ask"/"asked" is a genuine near-tie of this model at encoder frame 108:
        // on x86 CPU the joiner logits differ by only 0.19 (Q8_0) / 0.25 (F16) in
        // favour of "asked", so CPU builds, ISAs and CUDA legitimately differ.
        // Accept exactly these two variants; anything else is a regression (#598).
        static const char* accepted[] = {
            "andsomyfellowamericansasknotwhatyourcountrycandoforyouaskwhatyoucandoforyourcountry",
            "andsomyfellowamericansaskednotwhatyourcountrycandoforyouaskwhatyoucandoforyourcountry"};
        ok = ok && (normalized == accepted[0] || normalized == accepted[1]);
    }
    std::puts(ok ? "PASS xASR Q8_0 JFK CPU and CUDA session transcription" : "FAIL xASR JFK transcript");
    return ok ? 0 : 1;
}
