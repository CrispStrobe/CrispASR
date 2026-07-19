// crispasr_backend_crepe.cpp — CREPE pitch (F0) backend shim.
//
// Pitch estimation is its OWN task (pitch frames out, not text), so it is not
// layered onto transcribe() — see docs/source-separation-surface.md for the
// precedent that settled this for stems. It is exposed through the `--pitch`
// dispatcher (examples/cli/crispasr_pitch_cli.cpp), which resolves the model,
// decodes the audio to CREPE's 16 kHz mono, runs crepe_compute_f0 and prints
// one line per frame.
//
// This shim exists only so `crepe` appears in --list-backends with the
// CAP_PITCH capability and so `--backend crepe` without `--pitch` gives a clear
// redirect instead of trying to force a pitch track through the transcribe()
// contract (which carries text segments and cannot represent it).

#include "crispasr_backend.h"
#include "whisper_params.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace {

class CrepeBackend : public CrispasrBackend {
public:
    const char* name() const override { return "crepe"; }

    uint32_t capabilities() const override { return CAP_PITCH | CAP_AUTO_DOWNLOAD | CAP_INTERNAL_CHUNKING; }

    bool init(const whisper_params&) override {
        fprintf(stderr, "crispasr: crepe is a pitch (F0) model — run it with --pitch\n"
                        "  (e.g. `crispasr --pitch -m <crepe-tiny-f16.gguf> -f audio.wav`), not as a\n"
                        "  transcribe backend. --pitch prints time_ms/f0_hz/voiced_prob per frame.\n");
        return false;
    }

    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }

    void shutdown() override {}
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_crepe_backend() {
    return std::make_unique<CrepeBackend>();
}
