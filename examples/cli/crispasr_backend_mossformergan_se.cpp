// ClearerVoice-Studio MossFormerGAN_SE_16K is a speech-enhancement (denoise)
// model. It runs as the selectable enhancement pre-pass (src/crispasr_enhancer.h),
// not as a transcription backend; this shim only redirects `-m <gguf>`.
#include "crispasr_backend.h"
#include "whisper_params.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace {
class MossFormerGanSeBackend : public CrispasrBackend {
public:
    const char* name() const override { return "mossformergan-se"; }
    uint32_t capabilities() const override { return CAP_ENHANCE; }

    bool init(const whisper_params&) override {
        fprintf(stderr, "crispasr: mossformergan-se is a speech-enhancement model; run it with --enhance <gguf>\n"
                        "  crispasr --enhance mossformergan-se-16k-f16.gguf --enhance-output clean.wav -f noisy.wav\n"
                        "  crispasr --enhance mossformergan-se-16k-f16.gguf -m ggml-base.en.bin -f noisy.wav\n");
        return false;
    }
    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }
    void shutdown() override {}
};
} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_mossformergan_se_backend() {
    return std::make_unique<MossFormerGanSeBackend>();
}
