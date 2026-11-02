// ClearerVoice-Studio FRCRN_SE_16K is a speech-enhancement (denoise) model; it
// runs through the selectable enhancer (--enhance), not as a transcriber.
#include "crispasr_backend.h"
#include "whisper_params.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace {
class FrcrnSeBackend : public CrispasrBackend {
public:
    const char* name() const override { return "frcrn-se"; }
    uint32_t capabilities() const override { return CAP_ENHANCE; }

    bool init(const whisper_params&) override {
        fprintf(stderr, "crispasr: frcrn-se is a speech-enhancement model; run it with --enhance <gguf>\n"
                        "  crispasr --enhance frcrn-se-16k-f16.gguf --enhance-output clean.wav -f noisy.wav\n"
                        "  crispasr --enhance frcrn-se-16k-f16.gguf -m <asr-model> -f noisy.wav\n");
        return false;
    }
    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }
    void shutdown() override {}
};
} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_frcrn_se_backend() {
    return std::make_unique<FrcrnSeBackend>();
}
