// Inworld Voice Isolator (#602) uses the shared source-separation surface:
// `--separate` writes <input>_voice.wav and <input>_background.wav.
#include "crispasr_backend.h"
#include "whisper_params.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace {
class VoiceIsolatorBackend : public CrispasrBackend {
public:
    const char* name() const override { return "voice-isolator"; }
    uint32_t capabilities() const override { return CAP_SEPARATE | CAP_AUTO_DOWNLOAD | CAP_INTERNAL_CHUNKING; }

    bool init(const whisper_params&) override {
        fprintf(stderr, "crispasr: voice-isolator is a voice-isolation (separation) model; run it with --separate\n"
                        "  crispasr --separate -m voice-isolator-f16.gguf -f noisy.wav   # -> _voice.wav + "
                        "_background.wav\n");
        return false;
    }
    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }
    void shutdown() override {}
};
} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_voice_isolator_backend() {
    return std::make_unique<VoiceIsolatorBackend>();
}
