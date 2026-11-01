// HiDolen Mini-BS-RoFormer V2.5 uses the shared source-separation surface.
#include "crispasr_backend.h"
#include "whisper_params.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace {
class MiniBsRoformerBackend : public CrispasrBackend {
public:
    const char* name() const override { return "mini-bs-roformer"; }
    uint32_t capabilities() const override { return CAP_SEPARATE | CAP_AUTO_DOWNLOAD | CAP_INTERNAL_CHUNKING; }

    bool init(const whisper_params&) override {
        fprintf(stderr, "crispasr: mini-bs-roformer is a source-separation model; run it with --separate\n"
                        "  crispasr --separate -m mini-bs-roformer-f16.gguf -f song.wav\n");
        return false;
    }
    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }
    void shutdown() override {}
};
} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_mini_bs_roformer_backend() {
    return std::make_unique<MiniBsRoformerBackend>();
}
