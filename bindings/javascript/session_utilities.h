#pragma once
#include "crispasr.h"
#include "crispasr_session.h"
#include <memory>
#include <string>
#include <vector>
#include <emscripten/bind.h>

class BindingTextProcessor {
    void* handle = nullptr;
    int kind;
    void free_text(const char* text) {
        if (kind == 0)
            crispasr_punc_free_text(text);
        else if (kind == 1)
            crispasr_truecase_free_text(text);
        else
            crispasr_pcs_free_text(text);
    }

public:
    BindingTextProcessor(const std::string& model, int kind) : kind(kind) {
        binding_check_reentry();
        if (kind < 0 || kind > 2)
            throw std::invalid_argument("text processor kind must be 0=punc, 1=truecase, 2=PCS");
        handle = kind == 0   ? crispasr_punc_init(model.c_str())
                 : kind == 1 ? crispasr_truecase_init(model.c_str())
                             : crispasr_pcs_init(model.c_str());
        if (!handle)
            throw std::runtime_error("text processor could not open");
    }
    BindingTextProcessor(const BindingTextProcessor&) = delete;
    BindingTextProcessor& operator=(const BindingTextProcessor&) = delete;
    std::string process(const std::string& text) {
        binding_check_reentry();
        if (!handle)
            throw std::runtime_error("text processor is closed");
        const char* p = kind == 0   ? crispasr_punc_process(handle, text.c_str())
                        : kind == 1 ? crispasr_truecase_process(handle, text.c_str())
                                    : crispasr_pcs_process(handle, text.c_str());
        if (!p)
            throw std::runtime_error("text processor failed");
        auto release = [this](const char* text) { free_text(text); };
        std::unique_ptr<const char, decltype(release)> owned(p, release);
        return std::string(p);
    }
    void close() {
        binding_check_reentry();
        if (!handle)
            return;
        if (kind == 0)
            crispasr_punc_free(handle);
        else if (kind == 1)
            crispasr_truecase_free(handle);
        else
            crispasr_pcs_free(handle);
        handle = nullptr;
    }
    ~BindingTextProcessor() {
        if (handle) {
            if (kind == 0)
                crispasr_punc_free(handle);
            else if (kind == 1)
                crispasr_truecase_free(handle);
            else
                crispasr_pcs_free(handle);
        }
    }
};
static void binding_register_utilities() {
    using emscripten::val;
    emscripten::class_<BindingTextProcessor>("TextProcessor")
        .constructor<const std::string&, int>()
        .function("process", &BindingTextProcessor::process)
        .function("close", &BindingTextProcessor::close);
    emscripten::function("audioLoad", emscripten::optional_override([](const std::string& path, int rate) {
                             binding_check_reentry();
                             if (rate <= 0)
                                 throw std::invalid_argument("positive sample rate required");
                             float* p = nullptr;
                             int n = 0, sr = 0;
                             int rc = rate == 16000 ? crispasr_audio_load(path.c_str(), &p, &n, &sr)
                                                    : crispasr_audio_load_at_rate(path.c_str(), rate, &p, &n, &sr);
                             std::unique_ptr<float, decltype(&crispasr_audio_free)> owned(p, crispasr_audio_free);
                             binding_status(rc);
                             auto out = val::object();
                             out.set("samples", binding_floats(p, binding_count(n, 1)));
                             out.set("sampleRate", sr);
                             return out;
                         }));
    emscripten::function("audioLoadStereo", emscripten::optional_override([](const std::string& path) {
                             binding_check_reentry();
                             float* left = nullptr;
                             float* right = nullptr;
                             int n = 0, rate = 0, channels = 0;
                             int rc = crispasr_audio_load_stereo(path.c_str(), &left, &right, &n, &rate, &channels);
                             std::unique_ptr<float, decltype(&crispasr_audio_free)> l(left, crispasr_audio_free),
                                 r(left == right ? nullptr : right, crispasr_audio_free);
                             binding_status(rc);
                             auto out = val::object();
                             out.set("left", binding_floats(left, binding_count(n, 1)));
                             out.set("right", binding_floats(right, binding_count(n, 1)));
                             out.set("sampleRate", rate);
                             out.set("sourceChannels", channels);
                             return out;
                         }));
    emscripten::function("audioResample", emscripten::optional_override([](const val& audio, int source, int target) {
                             binding_check_reentry();
                             if (source <= 0 || target <= 0)
                                 throw std::invalid_argument("positive sample rates required");
                             if (audio["length"].as<int>() == 0)
                                 return binding_floats(nullptr, 0);
                             auto pcm = binding_pcm(audio);
                             float* p = nullptr;
                             int n = 0;
                             int rc = crispasr_audio_resample(pcm.data(), pcm.size(), source, target, &p, &n);
                             std::unique_ptr<float, decltype(&crispasr_audio_free)> owned(p, crispasr_audio_free);
                             binding_status(rc);
                             return binding_floats(p, binding_count(n, 1));
                         }));
    emscripten::function("cApiVersion",
                         emscripten::optional_override([]() { return binding_string(crispasr_c_api_version()); }));
    emscripten::function("helperVersion", emscripten::optional_override(
                                              []() { return binding_string(crispasr_dart_helpers_version()); }));
    emscripten::function("setGpuBackend", emscripten::optional_override([](const std::string& name) {
                             binding_check_reentry();
                             crispasr_set_gpu_backend(name.c_str());
                         }));
    emscripten::function("backendCapabilities", emscripten::optional_override([](const std::string& name) {
                             std::vector<char> data(65536);
                             binding_status(crispasr_backend_caps_abi(name.c_str(), data.data(), data.size()));
                             return std::string(data.data());
                         }));
    emscripten::function("allBackendCapabilities", emscripten::optional_override([]() {
                             int n = crispasr_backend_caps_list_abi(nullptr, 0);
                             if (n == INT_MIN)
                                 throw std::runtime_error("invalid native size");
                             std::vector<char> data(std::max(1, n < 0 ? -n : n));
                             binding_status(crispasr_backend_caps_list_abi(data.data(), data.size()));
                             return std::string(data.data());
                         }));
    emscripten::function("detectBackendsFromGguf", emscripten::optional_override([](const std::string& path) {
                             std::vector<char> data(65536);
                             binding_status(crispasr_detect_backends_from_gguf(path.c_str(), data.data(), data.size()));
                             return std::string(data.data());
                         }));
    emscripten::function("loadWatermarkModel", emscripten::optional_override([](const std::string& path) {
                             binding_status(crispasr_watermark_load_model(path.c_str()));
                         }));
    emscripten::function("watermarkEmbed", emscripten::optional_override([](const val& audio, float alpha) {
                             binding_check_reentry();
                             auto pcm = binding_pcm(audio);
                             crispasr_watermark_embed(pcm.data(), pcm.size(), alpha);
                             return binding_floats(pcm.data(), pcm.size());
                         }));
    emscripten::function("watermarkDetect", emscripten::optional_override([](const val& audio) {
                             auto pcm = binding_pcm(audio);
                             return crispasr_watermark_detect(pcm.data(), pcm.size());
                         }));
    emscripten::function("freeAlignerCache", emscripten::optional_override([]() {
                             binding_check_reentry();
                             crispasr_free_aligner_cache();
                         }));
    emscripten::function("sessionBackend", emscripten::optional_override([]() {
                             binding_require_session();
                             return binding_string(crispasr_session_backend(g_tts_session));
                         }));
    emscripten::function("sessionSetPcmSampleRate", emscripten::optional_override([](int rate) {
                             binding_require_session();
                             return crispasr_session_set_pcm_sample_rate(g_tts_session, rate);
                         }));
    emscripten::function("sessionSetParakeetAttContext", emscripten::optional_override([](int left, int right) {
                             binding_require_session();
                             return crispasr_session_set_parakeet_att_context(g_tts_session, left, right);
                         }));
    emscripten::function("sessionSetAudioCtx", emscripten::optional_override([](int audio_ctx) {
                             binding_require_session();
                             return crispasr_session_set_audio_ctx(g_tts_session, audio_ctx);
                         }));
    emscripten::function("sessionSetHikariRefreshStride", emscripten::optional_override([](int stride) {
                             binding_require_session();
                             return crispasr_session_set_hikari_refresh_stride(g_tts_session, stride);
                         }));
    emscripten::function("sessionSetGrammarStrict", emscripten::optional_override([](bool enabled) {
                             binding_require_session();
                             return crispasr_session_set_grammar_strict(g_tts_session, enabled);
                         }));
    emscripten::function(
        "sessionScoreTexts", emscripten::optional_override([](const val& audio, const val& candidates,
                                                              const std::string& language, const std::string& prompt) {
            binding_require_session();
            auto pcm = binding_pcm(audio);
            int n = candidates["length"].as<int>();
            if (n <= 0)
                throw std::invalid_argument("at least one candidate required");
            std::vector<std::string> strings;
            strings.reserve(n);
            for (int i = 0; i < n; ++i)
                strings.push_back(candidates[i].as<std::string>());
            std::vector<const char*> texts;
            for (const auto& s : strings)
                texts.push_back(s.c_str());
            std::vector<float> scores(n);
            std::vector<int> tokens(n);
            binding_status(crispasr_session_score_texts(
                g_tts_session, pcm.data(), pcm.size(), language.empty() ? nullptr : language.c_str(),
                prompt.empty() ? nullptr : prompt.c_str(), texts.data(), n, scores.data(), tokens.data()));
            auto out = val::object();
            out.set("logProbabilities", binding_floats(scores.data(), scores.size()));
            auto counts = val::global("Int32Array").new_(n);
            counts.call<void>("set", val(emscripten::typed_memory_view(tokens.size(), tokens.data())));
            out.set("tokenCounts", counts);
            return out;
        }));
    emscripten::function("getStreamedSegmentCount", emscripten::optional_override([]() {
                             binding_check_reentry();
                             return crispasr_get_streamed_segment_count();
                         }));
    emscripten::function("drainStreamedSegments", emscripten::optional_override([]() {
                             binding_check_reentry();
                             auto r = crispasr_drain_streamed_segments();
                             if (!r)
                                 return val::array();
                             std::unique_ptr<crispasr_session_result, decltype(&crispasr_session_result_free)> owned(
                                 r, crispasr_session_result_free);
                             return binding_encode_segments(binding_copy_segments(r), 100.0);
                         }));
    emscripten::function("drainStreamedTokenBytes", emscripten::optional_override([]() {
                             binding_check_reentry();
                             int n = 0;
                             auto p = crispasr_drain_streamed_tokens(&n);
                             if (n < 0 || (n && !p))
                                 throw std::runtime_error("Invalid native token drain");
                             // JS constructors/setters can re-enter the module. Finish
                             // reading the borrowed native buffer before any JS call.
                             std::vector<std::string> pieces;
                             pieces.reserve(n);
                             for (int i = 0; i < n; ++i) {
                                 pieces.emplace_back(p);
                                 p += pieces.back().size() + 1;
                             }
                             auto out = val::array();
                             for (const auto& piece : pieces)
                                 out.call<void>("push", binding_bytes(piece));
                             return out;
                         }));
    emscripten::function("resetStreamedSegments", emscripten::optional_override([]() {
                             binding_check_reentry();
                             crispasr_reset_streamed_segments();
                         }));
    emscripten::function("getStreamedTokenCount", emscripten::optional_override([]() {
                             binding_check_reentry();
                             return crispasr_get_streamed_token_count();
                         }));
    emscripten::function("resetStreamedTokens", emscripten::optional_override([]() {
                             binding_check_reentry();
                             crispasr_reset_streamed_tokens();
                         }));
    emscripten::function("resetProgress", emscripten::optional_override([]() {
                             binding_check_reentry();
                             crispasr_reset_progress();
                         }));
    emscripten::function("sessionSeparate", emscripten::optional_override([](const val& audio) {
                             binding_require_session();
                             auto pcm = binding_pcm(audio);
                             if (pcm.size() % 2)
                                 throw std::invalid_argument("interleaved stereo required");
                             int count = crispasr_session_separate(g_tts_session, pcm.data(), pcm.size() / 2);
                             binding_status(count);
                             auto out = val::array();
                             for (int i = 0; i < count; ++i) {
                                 int n = 0;
                                 auto p = crispasr_session_separate_stem(g_tts_session, i, &n);
                                 auto stem = val::object();
                                 stem.set("name",
                                          binding_string(crispasr_session_separate_stem_name(g_tts_session, i)));
                                 stem.set("samples", binding_floats(p, binding_count(n, 2)));
                                 stem.set("sampleRate", crispasr_session_separate_sample_rate(g_tts_session));
                                 out.call<void>("push", stem);
                             }
                             return out;
                         }));
}
