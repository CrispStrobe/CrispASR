#include "crispasr_backend.h"
#include "moonshine_onnx.h"
#include "whisper_params.h"
#include <cstdio>
#include <algorithm>

class MoonshineOnnxRealtime : public CrispasrRealtimeSession {
    moonshine_onnx_context* ctx_;
    moonshine_onnx_stream* stream_;
    int step_ms_;
    int64_t counter_ = 0;

public:
    MoonshineOnnxRealtime(moonshine_onnx_context* ctx, int step)
        : ctx_(ctx), stream_(moonshine_onnx_stream_open(ctx, step)), step_ms_(step) {}
    ~MoonshineOnnxRealtime() override { moonshine_onnx_stream_close(stream_); }
    bool valid() const { return stream_ != nullptr; }
    bool append(const float* pcm, int n, bool flush, callback on_text) override {
        if (!stream_ || (n && moonshine_onnx_stream_feed(stream_, pcm, n) < 0))
            return false;
        if (flush && moonshine_onnx_stream_flush(stream_) < 0)
            return false;
        std::vector<char> text(4096);
        int64_t revision = 0;
        int length = moonshine_onnx_stream_get_text(stream_, text.data(), static_cast<int>(text.size()), nullptr,
                                                    nullptr, &revision);
        if (length < 0)
            return false;
        if (length >= static_cast<int>(text.size())) {
            text.resize(length + 1);
            moonshine_onnx_stream_get_text(stream_, text.data(), static_cast<int>(text.size()), nullptr, nullptr,
                                           &revision);
        }
        if (flush || revision != counter_) {
            counter_ = revision;
            on_text(std::string(text.data()), flush);
        }
        return true;
    }
    void reset() override {
        moonshine_onnx_stream_close(stream_);
        stream_ = moonshine_onnx_stream_open(ctx_, step_ms_);
        counter_ = 0;
    }
};

class MoonshineOnnxBackend : public CrispasrBackend {
    moonshine_onnx_context* ctx_ = nullptr;

public:
    const char* name() const override { return "moonshine-onnx"; }
    const char* sole_language() const override { return "de"; }
    uint32_t capabilities() const override {
        return CAP_AUTO_DOWNLOAD | CAP_PUNCTUATION_NATIVE | (moonshine_onnx_incremental(ctx_) ? CAP_STREAMING : 0);
    }
    bool init(const whisper_params& p) override {
        ctx_ = moonshine_onnx_open(p.model.c_str(), p.n_threads);
        moonshine_onnx_set_max_new_tokens(ctx_, p.max_new_tokens_explicit ? p.max_new_tokens : 0);
        return ctx_ != nullptr;
    }
    std::vector<crispasr_segment> transcribe(const float* pcm, int n, int64_t offset,
                                             const whisper_params& p) override {
        try {
            moonshine_onnx_set_max_new_tokens(ctx_, p.max_new_tokens_explicit ? p.max_new_tokens : 0);
            crispasr_segment s;
            s.text = moonshine_onnx_transcribe(ctx_, pcm, n);
            s.t0 = offset;
            s.t1 = offset + n / 160;
            if (!s.text.empty())
                return {s};
        } catch (const std::exception& e) {
            fprintf(stderr, "moonshine-onnx: %s\n", e.what());
        }
        return {};
    }
    bool prefers_realtime_session() const override { return moonshine_onnx_incremental(ctx_); }
    std::unique_ptr<CrispasrRealtimeSession> create_realtime_session(const whisper_params& p) override {
        if (!moonshine_onnx_incremental(ctx_))
            return nullptr;
        moonshine_onnx_set_max_new_tokens(ctx_, p.max_new_tokens_explicit ? p.max_new_tokens : 0);
        auto stream = std::make_unique<MoonshineOnnxRealtime>(ctx_, p.stream_step_ms);
        if (!stream->valid())
            return nullptr;
        return stream;
    }
    void transcribe_streaming(const float* pcm, int n, int64_t offset, const whisper_params& p,
                              crispasr_stream_callback on_text) override {
        auto stream = create_realtime_session(p);
        if (!stream) {
            CrispasrBackend::transcribe_streaming(pcm, n, offset, p, on_text);
            return;
        }
        for (int i = 0; i < n; i += 5120)
            if (!stream->append(pcm + i, std::min(5120, n - i), false, on_text))
                return;
        stream->append(nullptr, 0, true, on_text);
    }
    void shutdown() override {
        moonshine_onnx_close(ctx_);
        ctx_ = nullptr;
    }
    ~MoonshineOnnxBackend() override { shutdown(); }
};
std::unique_ptr<CrispasrBackend> crispasr_make_moonshine_onnx_backend() {
    return std::make_unique<MoonshineOnnxBackend>();
}
