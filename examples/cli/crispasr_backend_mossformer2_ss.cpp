// ClearerVoice-Studio MossFormer2_SS_16K uses the shared source-separation surface.
#include "crispasr_backend.h"
#include "whisper_params.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace {
class Mossformer2SsBackend : public CrispasrBackend {
public:
    const char* name() const override { return "mossformer2-ss"; }
    uint32_t capabilities() const override { return CAP_SEPARATE | CAP_INTERNAL_CHUNKING; }

    bool init(const whisper_params&) override {
        fprintf(stderr, "crispasr: mossformer2-ss is a two-speaker separation model; run it with --separate\n"
                        "  crispasr --separate -m mossformer2-ss-16k-f16.gguf -f mixture.wav\n");
        return false;
    }
    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }
    void shutdown() override {}
};
} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_mossformer2_ss_backend() {
    return std::make_unique<Mossformer2SsBackend>();
}
