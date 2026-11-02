// ClearerVoice-Studio MossFormer2_SR_48K is a 48 kHz speech super-resolution
// (bandwidth extension) model. It runs as the selectable enhancement pre-pass
// (src/crispasr_enhancer.h), not as a transcription backend; this shim only
// redirects `-m <gguf>`.
#include "crispasr_backend.h"
#include "whisper_params.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace {
class MossFormer2SrBackend : public CrispasrBackend {
public:
    const char* name() const override { return "mossformer2-sr"; }
    uint32_t capabilities() const override { return CAP_ENHANCE; }

    bool init(const whisper_params&) override {
        fprintf(stderr,
                "crispasr: mossformer2-sr is a speech super-resolution model; run it with --enhance <gguf>\n"
                "  crispasr --enhance mossformer2-sr-48k-f16.gguf --enhance-output out48k.wav -f narrowband.wav\n"
                "  (--enhance-output writes 48 kHz; as an ASR pre-pass the result is resampled back)\n");
        return false;
    }
    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }
    void shutdown() override {}
};
} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_mossformer2_sr_backend() {
    return std::make_unique<MossFormer2SrBackend>();
}
