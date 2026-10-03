// crispasr_backend_mimo_asr.cpp — XiaomiMiMo/MiMo-V2.5-ASR backend.
//
// MiMo-V2.5-ASR pairs a 6-layer "input_local_transformer" audio-token
// processor + 36-layer Qwen2 LM (the LM weights live in the GGUF passed
// via -m) with a SEPARATE audio tokenizer GGUF (the MiMo-Audio-Tokenizer
// encoder, ~395 MB Q4_K). The tokenizer path is resolved from
// --codec-model PATH or auto-discovered from candidate filenames in the
// same directory as the LM. PLAN #51 step 8/10.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "mimo_asr.h"
#include "whisper_params.h"
#include "core/bpe.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

// Thin alias — delegates to core_bpe::token_bytes_to_utf8().
static std::string mimo_decode_bpe_piece(const std::string& s) {
    return core_bpe::token_bytes_to_utf8(s);
}

bool file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

std::string discover_audio_tokenizer(const std::string& model_path) {
    auto dir_of = [](const std::string& p) -> std::string {
        auto sep = p.find_last_of("/\\");
        return (sep == std::string::npos) ? std::string(".") : p.substr(0, sep);
    };
    const std::string dir = dir_of(model_path);
    static const char* candidates[] = {
        "mimo-tokenizer-q4_k.gguf",
        "mimo-tokenizer.gguf",
        "mimo-audio-tokenizer.gguf",
    };
    for (const char* name : candidates) {
        std::string p = dir + "/" + name;
        if (file_exists(p))
            return p;
    }
    return "";
}

void configure_prompt(mimo_asr_context* ctx, const whisper_params& params) {
    mimo_asr_set_language(ctx, params.language.c_str());
    mimo_asr_set_ask(ctx, params.ask.c_str());
}

class MimoAsrBackend : public CrispasrBackend {
public:
    MimoAsrBackend() = default;
    ~MimoAsrBackend() override { MimoAsrBackend::shutdown(); }

    const char* name() const override { return "mimo-asr"; }
    uint32_t capabilities() const override {
        // #300/#308 audit: the model emits punctuated, sentence-cased text (verified
        // on jfk: "And so, my fellow Americans, ... for you. Ask what ..."), so the
        // CLI's automatic FireRedPunc pass must not run over it — it doubled every
        // sentence-final stop ("for you..", "country..").
        // Verified against src/mimo_asr.cpp as of 2026-05-04:
        //   CAP_AUTO_DOWNLOAD     registry entry exists
        //   CAP_TIMESTAMPS_CTC    framework -am post-step works on segments
        //   CAP_TOKEN_CONFIDENCE  emits per-token probs
        //   CAP_FLASH_ATTN        uses ggml_flash_attn_ext (×2 in src)
        //   CAP_TEMPERATURE       init() below plumbs params.temperature
        //                         into cp.temperature → decode cfg
        //   CAP_DIARIZE           framework post-step on segment list
        //   CAP_LANGUAGE_DETECT   empty audio tag lets MiMo detect language
        return CAP_AUTO_DOWNLOAD | CAP_TOKEN_CONFIDENCE | CAP_TIMESTAMPS_CTC | CAP_FLASH_ATTN | CAP_TEMPERATURE |
               CAP_DIARIZE | CAP_BEAM_SEARCH | CAP_PUNCTUATION_NATIVE | CAP_LANGUAGE_DETECT;
    }

    bool init(const whisper_params& params) override {
        auto cp = mimo_asr_context_default_params();
        cp.n_threads = params.n_threads;
        cp.verbosity = params.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(params);
        cp.temperature = params.temperature;

        ctx_ = mimo_asr_init_from_file(params.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[mimo-asr]: failed to load LM model '%s'\n", params.model.c_str());
            return false;
        }

        std::string tok_path = params.tts_codec_model;
        if (tok_path.empty())
            tok_path = discover_audio_tokenizer(params.model);
        if (tok_path.empty()) {
            fprintf(stderr, "crispasr[mimo-asr]: audio tokenizer GGUF not found next to LM. Options:\n"
                            "  1. Rerun with --auto-download (the tokenizer is in the manifest as of 2026-05-26).\n"
                            "  2. Pass --codec-model PATH/mimo-tokenizer-q4_k.gguf explicitly.\n"
                            "  3. Download manually:\n"
                            "       hf download cstr/mimo-tokenizer-GGUF mimo-tokenizer-q4_k.gguf "
                            "--local-dir <dir-of-LM>\n");
            return false;
        }
        if (mimo_asr_set_tokenizer_path(ctx_, tok_path.c_str()) != 0) {
            fprintf(stderr, "crispasr[mimo-asr]: failed to register tokenizer path '%s'\n", tok_path.c_str());
            return false;
        }
        if (!params.no_prints)
            fprintf(stderr, "crispasr[mimo-asr]: tokenizer at '%s'\n", tok_path.c_str());
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;
        mimo_asr_set_beam_size(ctx_, params.beam_size > 0 ? params.beam_size : 1);
        // #292: forward --max-new-tokens only when explicit; 0 keeps the backend default.
        mimo_asr_set_max_new_tokens(ctx_, params.max_new_tokens_explicit ? params.max_new_tokens : 0);
        configure_prompt(ctx_, params);
        char* text = mimo_asr_transcribe(ctx_, samples, n_samples);
        if (text) {
            crispasr_segment seg;
            seg.text = text;
            seg.t0 = t_offset_cs;
            seg.t1 = t_offset_cs + (int64_t)((int64_t)n_samples * 100 / 16000);
            out.push_back(std::move(seg));
            free(text);
        }
        return out;
    }

    void transcribe_streaming(const float* samples, int n_samples, int64_t /*t_offset_cs*/,
                              const whisper_params& params, crispasr_stream_callback on_text) override {
        if (!ctx_) {
            CrispasrBackend::transcribe_streaming(samples, n_samples, 0, params, on_text);
            return;
        }
        configure_prompt(ctx_, params);
        std::string accumulated;
        bool first_tok = true;
        auto cb = [&](int tok_id, float /*prob*/, void* /*ud*/) {
            const char* raw = mimo_asr_token_text(ctx_, tok_id);
            if (!raw || !*raw)
                return;
            std::string piece = mimo_decode_bpe_piece(std::string(raw));
            if (first_tok) {
                size_t sp = 0;
                while (sp < piece.size() && (piece[sp] == ' ' || piece[sp] == '\n'))
                    sp++;
                piece = piece.substr(sp);
                if (!piece.empty())
                    first_tok = false;
            }
            accumulated += piece;
            if (!accumulated.empty())
                on_text(accumulated.c_str(), false);
        };
        auto cb_fn = [](int id, float p, void* ud) { (*static_cast<decltype(cb)*>(ud))(id, p, nullptr); };
        mimo_asr_transcribe_cb(ctx_, samples, n_samples, cb_fn, &cb);
        on_text(accumulated.c_str(), true);
    }

    void shutdown() override {
        if (ctx_) {
            mimo_asr_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    mimo_asr_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_mimo_asr_backend() {
    return std::make_unique<MimoAsrBackend>();
}
