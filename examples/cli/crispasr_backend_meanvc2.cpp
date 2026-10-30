// crispasr_backend_meanvc2.cpp — MeanVC2 zero-shot voice conversion backend shim.
//
// MeanVC2 is a low-latency streaming zero-shot voice conversion model
// (Fast-U2++ ASR BNF + UTTE + ChunkDiT + Vocos).
// It converts source speech audio to a target speaker's voice.
//
// This shim registers MeanVC2 in the CLI backend table and describes its
// capabilities (CAP_S2S | CAP_AUTO_DOWNLOAD | CAP_INTERNAL_CHUNKING).
// The full C ABI surface is exposed in meanvc2.h (meanvc2_convert, etc.).

#include "crispasr_backend.h"
#include "whisper_params.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace {

class Meanvc2Backend : public CrispasrBackend {
public:
    const char* name() const override { return "meanvc2"; }

    uint32_t capabilities() const override { return CAP_S2S | CAP_AUTO_DOWNLOAD | CAP_INTERNAL_CHUNKING; }

    bool init(const whisper_params&) override {
        fprintf(stderr, "crispasr: meanvc2 is a ZERO-SHOT VOICE CONVERSION model.\n"
                        "  It converts 16 kHz speech audio to a target voice timbre.\n"
                        "  Use the C ABI surface in meanvc2.h (meanvc2_convert) or the\n"
                        "  standalone test-meanvc2-convert harness:\n"
                        "    test-meanvc2-convert <meanvc2.gguf> <input.wav> [output.wav]\n");
        return false;
    }

    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }

    void shutdown() override {}
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_meanvc2_backend() {
    return std::make_unique<Meanvc2Backend>();
}
