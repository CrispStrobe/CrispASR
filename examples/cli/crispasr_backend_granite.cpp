// crispasr_backend_granite.cpp — adapter for ibm-granite/granite-4.0-1b-speech.
//
// The library's granite_speech_transcribe() is currently a stub, so this
// backend implements the full pipeline manually using the stage-level API:
// mel -> encoder -> Q-Former projector -> LLM prompt splice -> KV prefill
// -> greedy decode -> GPT-2 byte detokenize.
//
// This code is a direct port of examples/granite-main/main.cpp's pipeline,
// wrapped in the CrispasrBackend interface. Once granite_speech_transcribe
// gains a real implementation, or once the shared LLM decode loop lands in
// src/core/, this file should shrink dramatically.
//
// Granite does not expose native timestamps. Word-level timestamps via a
// CTC aligner second pass will be wired up once crispasr_aligner lands.

#include "crispasr_backend.h"
#include "whisper_params.h"
#include "core/greedy_decode.h"

#include "granite_speech.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

// Granite chat-template prefix / suffix used around the audio placeholder.
// For granite-4.0-1b-speech the constants below are the exact tokens
// captured from the HF chat template under the GPT-NeoX tokenizer; for
// granite-speech-3.x (which uses a different 49160-token tokenizer) we
// re-tokenize the same strings at runtime via granite_speech_tokenize().
//
// Keeping both forms lets older granite-4.0 GGUFs that predate the
// merges-table conversion still work unchanged (they fall through to
// the hardcoded kPrefix4/kSuffix4 arrays), while granite-3.x / any
// future release dispatches through the tokenizer path.
constexpr int32_t kPrefix4[]   = { 6584, 25, 220 };                       // "USER: "
constexpr int     kNumPrefix4  = 3;
constexpr int32_t kSuffix4[]   = {                                        // "can you transcribe..."
    4919, 499, 1380, 3191, 279, 8982, 1139, 264, 5439, 3645, 30, 198,
    36660, 3931, 2891, 25
};
constexpr int     kNumSuffix4  = 16;

// Legacy token ids for granite-4.0-1b. Only used when the GGUF doesn't
// export granite_speech.llm.audio_token_index / eos_token_id (i.e. it
// was produced before that key landed). Both accessors below fall back
// to these when the runtime returns -1.
constexpr int kLegacyAudioTok4 = 100352;
constexpr int kLegacyEos4      = 100257;

class GraniteBackend : public CrispasrBackend {
public:
    GraniteBackend() = default;
    ~GraniteBackend() override { shutdown(); }

    const char * name() const override { return "granite"; }

    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_CTC | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE
             | CAP_PUNCTUATION_TOGGLE | CAP_FLASH_ATTN | CAP_TOKEN_CONFIDENCE
             | CAP_TRANSLATE | CAP_SRC_TGT_LANGUAGE | CAP_DIARIZE
             | CAP_PARALLEL_PROCESSORS;
    }

    bool init(const whisper_params & p) override {
        granite_speech_context_params cp = granite_speech_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;

        ctx_ = granite_speech_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[granite]: failed to load model '%s'\n",
                    p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(
        const float * samples, int n_samples,
        int64_t t_offset_cs,
        const whisper_params & params) override
    {
        std::vector<crispasr_segment> out;
        if (!ctx_) return out;

        // ---- Mel spectrogram ----
        int n_mels = 0, T_mel = 0;
        float * mel = granite_speech_compute_mel(ctx_, samples, n_samples, &n_mels, &T_mel);
        if (!mel) {
            fprintf(stderr, "crispasr[granite]: mel failed\n");
            return out;
        }

        // ---- Encoder ----
        int N_enc = 0, enc_dim = 0;
        float * enc = granite_speech_run_encoder(ctx_, mel, n_mels, T_mel, &N_enc, &enc_dim);
        free(mel);
        if (!enc) {
            fprintf(stderr, "crispasr[granite]: encoder failed\n");
            return out;
        }

        // ---- Q-Former projector ----
        int N_proj = 0, proj_dim = 0;
        float * proj = granite_speech_run_projector(ctx_, enc, N_enc, enc_dim, &N_proj, &proj_dim);
        free(enc);
        if (!proj) {
            fprintf(stderr, "crispasr[granite]: projector failed\n");
            return out;
        }

        // ---- Build prompt: [prefix] + [audio placeholders] + [suffix] ----
        //
        // Strategy:
        //   1. Ask the runtime for audio_token_index / eos_token_id / vocab.
        //      Newer GGUFs export them; older granite-4.0 GGUFs return
        //      -1 and we fall back to the hardcoded granite-4.0 values.
        //   2. If the model's audio_token_index is within the granite-4.0
        //      100k-vocab range AND the GGUF didn't ship a merges table
        //      (i.e. granite_speech_tokenize returns nothing for a simple
        //      probe), reuse the hardcoded kPrefix4/kSuffix4 arrays — they
        //      were captured from the granite-4.0 tokenizer and still
        //      tokenize correctly under it.
        //   3. Otherwise (granite-3.x, or granite-4.0 re-converted with
        //      merges) re-tokenize the chat-template prefix + suffix at
        //      runtime so the resulting ids match the model's vocab.
        int audio_tok = granite_speech_audio_token_id(ctx_);
        int eos_tok   = granite_speech_eos_token_id(ctx_);
        const int vocab_sz = granite_speech_vocab_size(ctx_);
        if (audio_tok < 0) audio_tok = kLegacyAudioTok4;
        if (eos_tok   < 0) eos_tok   = kLegacyEos4;
        (void)vocab_sz;

        auto iso_to_eng = [](const std::string & c) -> std::string {
            if (c == "en") return "English";
            if (c == "de") return "German";
            if (c == "fr") return "French";
            if (c == "es") return "Spanish";
            if (c == "it") return "Italian";
            if (c == "pt") return "Portuguese";
            if (c == "ru") return "Russian";
            if (c == "ja") return "Japanese";
            if (c == "zh") return "Chinese";
            return c;
        };

        const std::string prefix_str = "USER: ";
        std::string suffix_str =
            "can you transcribe the speech into a written format?\n ASSISTANT:";
        if (params.translate) {
            const std::string tgt = params.target_lang.empty()
                                  ? std::string("English")
                                  : iso_to_eng(params.target_lang);
            suffix_str = "can you translate the speech to " + tgt +
                         "?\n ASSISTANT:";
        }

        // Try runtime tokenization first.
        std::vector<int32_t> prefix_ids, suffix_ids;
        {
            int n = 0;
            int32_t * a = granite_speech_tokenize(ctx_, prefix_str.c_str(), &n);
            if (a && n > 0) { prefix_ids.assign(a, a + n); free(a); }
        }
        {
            int n = 0;
            int32_t * a = granite_speech_tokenize(ctx_, suffix_str.c_str(), &n);
            if (a && n > 0) { suffix_ids.assign(a, a + n); free(a); }
        }

        // Fall back to the hardcoded granite-4.0 arrays when runtime
        // tokenization failed AND the model looks like granite-4.0 (its
        // audio_token_index lives in the 100k+ range that matches the
        // GPT-NeoX table the old arrays were captured under).
        if (prefix_ids.empty() || suffix_ids.empty()) {
            if (audio_tok == kLegacyAudioTok4) {
                if (prefix_ids.empty())
                    prefix_ids.assign(kPrefix4, kPrefix4 + kNumPrefix4);
                if (suffix_ids.empty() && !params.translate)
                    suffix_ids.assign(kSuffix4, kSuffix4 + kNumSuffix4);
            }
            if (prefix_ids.empty() || suffix_ids.empty()) {
                fprintf(stderr,
                        "crispasr[granite]: tokenize failed — re-convert the GGUF "
                        "with the newer models/convert-granite-speech-to-gguf.py "
                        "to pick up the merges table\n");
                free(proj);
                return out;
            }
        }

        const int n_prefix = (int)prefix_ids.size();
        const int n_suffix = (int)suffix_ids.size();
        const int total_prompt = n_prefix + N_proj + n_suffix;
        std::vector<int32_t> prompt_ids;
        prompt_ids.reserve(total_prompt);
        for (int id : prefix_ids) prompt_ids.push_back(id);
        for (int i = 0; i < N_proj; i++) prompt_ids.push_back(audio_tok);
        for (int id : suffix_ids) prompt_ids.push_back(id);

        float * all_embeds = granite_speech_embed_tokens(ctx_, prompt_ids.data(), total_prompt);
        if (!all_embeds) {
            free(proj);
            fprintf(stderr, "crispasr[granite]: embed failed\n");
            return out;
        }

        // Splice projector output into the audio positions (skip the prefix).
        for (int i = 0; i < N_proj; i++) {
            std::memcpy(all_embeds + (size_t)(n_prefix + i) * proj_dim,
                        proj + (size_t)i * proj_dim,
                        proj_dim * sizeof(float));
        }
        free(proj);

        // ---- KV cache + prefill ----
        if (!granite_speech_kv_init(ctx_, 4096)) {
            free(all_embeds);
            fprintf(stderr, "crispasr[granite]: kv init failed\n");
            return out;
        }
        granite_speech_kv_reset(ctx_);

        int vocab = 0;
        float * logits = granite_speech_run_llm_kv(ctx_, all_embeds, total_prompt, 0,
                                                    nullptr, &vocab);
        free(all_embeds);
        if (!logits) {
            fprintf(stderr, "crispasr[granite]: prefill failed\n");
            return out;
        }

        // ---- Decode loop via src/core/greedy_decode.h ----
        // Temperature sampling kicks in when --temperature > 0; otherwise
        // we stay on the historical bit-identical greedy path.
        core_greedy_decode::Config dec_cfg;
        dec_cfg.max_new_tokens = params.max_new_tokens > 0 ? params.max_new_tokens : 200;
        dec_cfg.eos_id         = eos_tok;
        dec_cfg.vocab_size     = vocab;
        dec_cfg.temperature    = params.temperature;

        int   next   = 0;
        float next_p = 1.0f;
        if (dec_cfg.temperature > 0.0f) {
            std::mt19937_64 seed_rng(dec_cfg.seed != 0 ? dec_cfg.seed
                                       : (uint64_t)std::random_device{}());
            next = core_greedy_decode::sample_temp(
                logits, vocab, dec_cfg.temperature, seed_rng);
        } else {
            next = core_greedy_decode::argmax(logits, vocab);
        }
        next_p = core_greedy_decode::softmax_of(
            logits, vocab, next, logits[next]);
        free(logits);

        auto dec = core_greedy_decode::run_with_probs(
            ctx_,
            /*first_token=*/next,
            /*first_prob=*/next_p,
            /*initial_n_past=*/total_prompt,
            granite_speech_embed_tokens,
            granite_speech_run_llm_kv,
            dec_cfg);
        const std::vector<int32_t> & gen_ids = dec.tokens;
        const std::vector<float>   & probs   = dec.probs;

        // Strip EOS from generated IDs before detokenizing.
        std::vector<int32_t> text_ids;
        text_ids.reserve(gen_ids.size());
        for (int32_t id : gen_ids) if (id != eos_tok) text_ids.push_back(id);

        char * text = granite_speech_decode_tokens(ctx_, text_ids.data(), (int)text_ids.size());

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = text ? text : "";
        if (text) free(text);

        // Trim leading whitespace emitted by the chat template.
        while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n')) {
            seg.text.erase(seg.text.begin());
        }

        // Per-token entries with decode-loop confidences. granite uses
        // its own batch detokenizer (granite_speech_decode_tokens) for
        // the segment text, but we still surface per-token id + prob so
        // downstream consumers (JSON full, confidence filters) have the
        // raw signal. Token text is intentionally left empty here to
        // avoid duplicating the batch detokenizer's merging logic.
        seg.tokens.reserve(gen_ids.size());
        for (size_t i = 0; i < gen_ids.size(); i++) {
            if (gen_ids[i] == eos_tok) break;
            crispasr_token ct;
            ct.id         = gen_ids[i];
            ct.confidence = (i < probs.size()) ? probs[i] : -1.0f;
            seg.tokens.push_back(std::move(ct));
        }

        out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            granite_speech_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    granite_speech_context * ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_granite_backend() {
    return std::unique_ptr<CrispasrBackend>(new GraniteBackend());
}
