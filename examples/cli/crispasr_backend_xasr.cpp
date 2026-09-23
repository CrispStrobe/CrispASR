// crispasr_backend_xasr.cpp — CLI adapter for X-ASR (streaming Zipformer2
// transducer, zh + en), #436.
//
// Decodes exactly as sherpa-onnx's OnlineRecognizer greedy search: the
// language is whatever the model hears, so -l is ignored. Offline
// transcription and the realtime WebSocket session run the same stream. Knobs:
//   CRISPASR_XASR_CHUNK_MS     160 / 480 (default) / 960 / 1920 — the upstream export's chunk size
//   CRISPASR_XASR_TAIL_PAD_MS  trailing silence before end of input (default T*10+1000 ms)

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "core/crispasr_env.h"
#include "xasr.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

// Realtime WebSocket session: the transducer's output is append-only by
// construction, so every partial is simply the transcript so far.
class XasrRealtimeSession final : public CrispasrRealtimeSession {
public:
    explicit XasrRealtimeSession(xasr_context* ctx) : stream_(xasr_stream_init(ctx)) {}
    ~XasrRealtimeSession() override { xasr_stream_free(stream_); }

    bool append(const float* samples, int n_samples, bool flush, callback on_text) override {
        char* text = xasr_stream_accept(stream_, samples, n_samples, flush);
        if (!text)
            return false;
        const std::string t(text);
        free(text);
        if (!t.empty() || flush)
            on_text(t, flush);
        return true;
    }

    void reset() override { xasr_stream_reset(stream_); }

private:
    xasr_stream* stream_ = nullptr;
};

class XasrBackend : public CrispasrBackend {
public:
    XasrBackend() = default;
    ~XasrBackend() override { XasrBackend::shutdown(); }

    const char* name() const override { return "xasr"; }

    uint32_t capabilities() const override { return CAP_AUTO_DOWNLOAD | CAP_PUNCTUATION_NATIVE; }

    bool init(const whisper_params& p) override {
        xasr_context_params cp = xasr_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        if (const char* v = crispasr_env::get("CRISPASR_XASR_CHUNK_MS"))
            cp.chunk_ms = std::atoi(v);
        if (const char* v = crispasr_env::get("CRISPASR_XASR_TAIL_PAD_MS"))
            cp.tail_pad_ms = std::atoi(v);
        ctx_ = xasr_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[xasr]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& /*params*/) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;
        char* text = xasr_transcribe(ctx_, samples, n_samples);
        if (!text)
            return out;
        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)n_samples * 100 / 16000;
        seg.text = text;
        free(text);
        out.push_back(std::move(seg));
        return out;
    }

    std::unique_ptr<CrispasrRealtimeSession> create_realtime_session(const whisper_params&) override {
        if (!ctx_)
            return nullptr;
        return std::unique_ptr<CrispasrRealtimeSession>(new XasrRealtimeSession(ctx_));
    }

    void shutdown() override {
        if (ctx_) {
            xasr_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    xasr_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_xasr_backend() {
    return std::unique_ptr<CrispasrBackend>(new XasrBackend());
}
