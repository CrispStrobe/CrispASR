// crispasr_backend_qwen3.cpp — adapter for Qwen/Qwen3-ASR-0.6B.
//
// Pipeline: mel -> encoder -> ChatML prompt (tokenized via BPE) with
// <|audio_pad|> placeholders -> embed + splice encoder frames -> KV
// prefill -> greedy decode -> GPT-2 byte-encoded detokenize.
//
// Qwen3's token_text() returns GPT-2 byte-encoded strings rather than
// raw bytes — decoded via core_bpe::token_bytes_to_utf8().
//
// Direct port of examples/qwen3-asr-main/main.cpp wrapped in the
// CrispasrBackend interface.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"
#include "core/bpe.h"
#include "core/greedy_decode.h"
#include "core/beam_decode.h"

#include "qwen3_asr.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

// Thin alias — delegates to core_bpe::token_bytes_to_utf8().
std::string decode_token(const std::string& s) {
    return core_bpe::token_bytes_to_utf8(s);
}

class Qwen3Backend : public CrispasrBackend {
public:
    Qwen3Backend() = default;
    ~Qwen3Backend() override { Qwen3Backend::shutdown(); }

    const char* name() const override { return "qwen3"; }

    uint32_t capabilities() const override {
        // CAP_LANGUAGE_DETECT intentionally NOT declared. The transcribe
        // path scrapes a "language <name>" prefix off the model output
        // when the system prompt asks for translation, but the default
        // ASR system prompt is empty — so qwen3 emits no language tag
        // for plain `-dl` and the cap would be dishonest. With it
        // absent, `-dl` correctly routes through the framework's
        // whisper-tiny pre-step LID.
        return CAP_TIMESTAMPS_CTC | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_PUNCTUATION_TOGGLE | CAP_FLASH_ATTN |
               CAP_TOKEN_CONFIDENCE | CAP_TRANSLATE | CAP_SRC_TGT_LANGUAGE | CAP_DIARIZE | CAP_PARALLEL_PROCESSORS |
               CAP_BEAM_SEARCH;
    }

    bool init(const whisper_params& p) override {
        auto cp = qwen3_asr_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        ctx_ = qwen3_asr_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[qwen3]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        // ---- Mel ----
        int n_mels = 0, T_mel = 0;
        float* mel = qwen3_asr_compute_mel(ctx_, samples, n_samples, &n_mels, &T_mel);
        if (!mel) {
            fprintf(stderr, "crispasr[qwen3]: mel failed\n");
            return out;
        }

        // ---- Encoder ----
        int N_enc = 0, pdim = 0;
        float* audio_embeds = qwen3_asr_run_encoder(ctx_, mel, n_mels, T_mel, &N_enc, &pdim);
        free(mel);
        if (!audio_embeds) {
            fprintf(stderr, "crispasr[qwen3]: encoder failed\n");
            return out;
        }

        // ---- ChatML prompt: <|im_start|>system\n[SYS]<|im_end|>\n
        //                     <|im_start|>user\n<|audio_start|>
        //                     <|audio_pad|> x N
        //                     <|audio_end|><|im_end|>\n
        //                     <|im_start|>assistant\n
        // For --translate we put a plain English instruction in the
        // SYSTEM turn ("Translate the audio to TGT.") because qwen3-asr
        // is a specialized ASR fine-tune that ignores user-turn
        // instructions but does honour system-prompt control. The
        // default transcribe path keeps the system turn empty, which
        // is the bit-identical historical behaviour. ISO-639-1 codes are
        // mapped to plain English names via crispasr_iso_to_english_lang —
        // qwen3 reads the system prompt literally and a bare "de" gets
        // interpreted as Spanish ("de" = "of"); the full name avoids that.
        std::string sys_instruction;
        if (!params.ask.empty()) {
            // Note: qwen3-asr is an ASR-specific fine-tune that may
            // ignore arbitrary instructions and transcribe anyway.
            // voxtral 3B is better suited for audio Q&A.
            sys_instruction = params.ask;
        } else if (params.translate) {
            const std::string tgt =
                params.target_lang.empty() ? std::string("English") : crispasr_iso_to_english_lang(params.target_lang);
            sys_instruction = "Translate the speech to " + tgt + ".";
        } else if (!params.language.empty() && params.language != "auto") {
            sys_instruction = "Transcribe the speech in " + crispasr_iso_to_english_lang(params.language) + ".";
        }
        // PLAN #98 Phase B: hotword prompt injection
        if (!params.hotwords.empty()) {
            if (!sys_instruction.empty() && sys_instruction.back() != ' ')
                sys_instruction += ' ';
            sys_instruction += "The following words may appear in the audio: " + params.hotwords + ".";
        }

        std::string text = "<|im_start|>system\n" + sys_instruction +
                           "<|im_end|>\n"
                           "<|im_start|>user\n"
                           "<|audio_start|>";
        text.reserve(text.size() + (size_t)N_enc * 13 + 64);
        for (int i = 0; i < N_enc; i++)
            text += "<|audio_pad|>";
        text += "<|audio_end|><|im_end|>\n"
                "<|im_start|>assistant\n";

        int n_prompt = 0;
        int32_t* raw_ids = qwen3_asr_tokenize(ctx_, text.c_str(), &n_prompt);
        if (!raw_ids) {
            fprintf(stderr, "crispasr[qwen3]: tokenize failed\n");
            free(audio_embeds);
            return out;
        }
        std::vector<int32_t> ids(raw_ids, raw_ids + n_prompt);
        free(raw_ids);

        // Look up the audio_pad token id by tokenizing just the special token.
        int n_pad_id = 0;
        int32_t* pad_id_arr = qwen3_asr_tokenize(ctx_, "<|audio_pad|>", &n_pad_id);
        int audio_pad_id = -1;
        if (pad_id_arr && n_pad_id >= 1)
            audio_pad_id = pad_id_arr[0];
        free(pad_id_arr);
        if (audio_pad_id < 0) {
            fprintf(stderr, "crispasr[qwen3]: could not resolve <|audio_pad|> id\n");
            free(audio_embeds);
            return out;
        }

        // ---- Embed + splice ----
        float* text_embeds = qwen3_asr_embed_tokens(ctx_, ids.data(), (int)ids.size());
        if (!text_embeds) {
            fprintf(stderr, "crispasr[qwen3]: embed failed\n");
            free(audio_embeds);
            return out;
        }
        int spliced = 0;
        for (size_t i = 0; i < ids.size() && spliced < N_enc; i++) {
            if (ids[i] == audio_pad_id) {
                std::memcpy(text_embeds + i * pdim, audio_embeds + (size_t)spliced * pdim, pdim * sizeof(float));
                spliced++;
            }
        }
        free(audio_embeds);

        // ---- KV cache + best-of-N decode ----
        if (!qwen3_asr_kv_init(ctx_, 4096)) {
            free(text_embeds);
            fprintf(stderr, "crispasr[qwen3]: kv_init failed\n");
            return out;
        }

        // Qwen3 EOS tokens: <|im_end|> (id unknown — look up via tokenize).
        int eos_id = -1;
        int n_eos = 0;
        int32_t* eos_arr = qwen3_asr_tokenize(ctx_, "<|im_end|>", &n_eos);
        if (eos_arr && n_eos >= 1)
            eos_id = eos_arr[0];
        free(eos_arr);

        const int prompt_len = (int)ids.size();
        const int max_new = params.max_new_tokens > 0 ? params.max_new_tokens : 256;

        // ---- Beam search path ----
        if (params.beam_size > 1) {
            qwen3_asr_kv_reset(ctx_);
            int n_t = 0, vocab = 0;
            float* logits = qwen3_asr_run_llm_kv(ctx_, text_embeds, prompt_len, 0, &n_t, &vocab);
            free(text_embeds);
            if (!logits) {
                fprintf(stderr, "crispasr[qwen3]: prefill failed\n");
                return out;
            }
            const float* last_logits = logits + (n_t - 1) * vocab;

            auto replay = [this](qwen3_asr_context* /*ctx*/, const int32_t* toks, int n, int pl) -> float* {
                float* emb = qwen3_asr_embed_tokens(ctx_, toks, n);
                if (!emb)
                    return nullptr;
                int nt2 = 0, v2 = 0;
                float* lg = qwen3_asr_run_llm_kv(ctx_, emb, n, pl, &nt2, &v2);
                std::free(emb);
                return lg;
            };

            core_beam_decode::Config cfg;
            cfg.max_new_tokens = max_new;
            cfg.eos_id = eos_id;
            cfg.vocab_size = vocab;
            cfg.beam_size = params.beam_size;
            cfg.prompt_len = prompt_len;

            auto beam_r = core_beam_decode::run_with_probs(ctx_, last_logits, replay, cfg);
            free(logits);

            // Feed beam result into the shared detokenize path below
            core_greedy_decode::Result best_dec;
            best_dec.tokens.assign(beam_r.tokens.begin(), beam_r.tokens.end());
            best_dec.probs = std::move(beam_r.probs);

            const std::vector<int32_t>& gen = best_dec.tokens;
            const std::vector<float>& probs = best_dec.probs;

            std::string transcript;
            std::string detected_language;
            bool capture_language = false;
            std::vector<crispasr_token> out_tokens;
            out_tokens.reserve(gen.size());
            for (size_t i = 0; i < gen.size(); i++) {
                const int32_t id = gen[i];
                if (id == eos_id)
                    break;
                const char* raw_piece = qwen3_asr_token_text(ctx_, id);
                if (!raw_piece || !*raw_piece)
                    continue;
                std::string raw = raw_piece;
                if (raw.size() >= 2 && raw[0] == '<' && raw[1] == '|')
                    continue;
                if (raw.size() >= 2 && raw[0] == '<' && raw.back() == '>')
                    continue;
                if (raw.size() >= 5 && raw[0] == '[' && raw[1] == 'P' && raw[2] == 'A' && raw[3] == 'D')
                    continue;
                std::string txt = decode_token(raw);
                if (raw == "language") {
                    capture_language = true;
                    continue;
                }
                if (capture_language) {
                    detected_language = txt;
                    capture_language = false;
                    continue;
                }
                transcript += txt;
                crispasr_token tk;
                tk.id = id;
                tk.text = txt;
                tk.confidence = (i < probs.size()) ? probs[i] : 1.0f;
                out_tokens.push_back(std::move(tk));
            }

            crispasr_segment seg;
            size_t start = 0;
            while (start < transcript.size() && transcript[start] == ' ')
                start++;
            seg.text = transcript.substr(start);
            crispasr_apply_ngram_loop_fix(seg.text, name(), params.no_prints);
            seg.tokens = std::move(out_tokens);
            if (!params.punctuation) {
                crispasr_strip_ascii_punctuation(seg.text);
                crispasr_lowercase_ascii(seg.text);
                for (auto& tk : seg.tokens) {
                    crispasr_strip_ascii_punctuation(tk.text);
                    crispasr_lowercase_ascii(tk.text);
                }
            }
            if (!seg.text.empty())
                out.push_back(std::move(seg));
            return out;
        }

        // ---- Greedy / best-of-N path ----
        core_greedy_decode::Config dec_cfg;
        dec_cfg.max_new_tokens = max_new;
        dec_cfg.eos_id = eos_id;
        dec_cfg.temperature = params.temperature;
        dec_cfg.frequency_penalty = params.frequency_penalty;
        dec_cfg.seed = params.seed;

        const int n_runs = (params.temperature > 0.0f && params.best_of > 1) ? params.best_of : 1;
        core_greedy_decode::Result best_dec;
        double best_score = -1.0;

        for (int run = 0; run < n_runs; run++) {
            qwen3_asr_kv_reset(ctx_);

            int n_t = 0, vocab = 0;
            float* logits = qwen3_asr_run_llm_kv(ctx_, text_embeds, prompt_len, 0, &n_t, &vocab);
            if (!logits) {
                fprintf(stderr, "crispasr[qwen3]: prefill failed (run %d/%d)\n", run + 1, n_runs);
                free(text_embeds);
                return out;
            }
            if (run == 0)
                dec_cfg.vocab_size = vocab;

            const int last_off = (n_t - 1) * vocab;
            int next = 0;
            float next_p = 1.0f;
            if (dec_cfg.temperature > 0.0f) {
                std::mt19937_64 seed_rng((dec_cfg.seed != 0 ? dec_cfg.seed : (uint64_t)std::random_device{}()) ^
                                         (uint64_t)(run * 0x9E3779B97F4A7C15ull));
                next = core_greedy_decode::sample_temp(logits + last_off, vocab, dec_cfg.temperature, seed_rng);
            } else {
                next = core_greedy_decode::argmax(logits + last_off, vocab);
            }
            next_p = core_greedy_decode::softmax_of(logits + last_off, vocab, next, logits[last_off + next]);

            free(logits);

            auto dec = core_greedy_decode::run_with_probs(ctx_,
                                                          /*first_token=*/next,
                                                          /*first_prob=*/next_p,
                                                          /*initial_n_past=*/(int)ids.size(), qwen3_asr_embed_tokens,
                                                          qwen3_asr_run_llm_kv, dec_cfg);

            double sum = 0.0;
            int cnt = 0;
            for (size_t i = 0; i < dec.probs.size(); i++) {
                if ((int32_t)dec.tokens[i] == eos_id)
                    break;
                sum += (double)dec.probs[i];
                cnt++;
            }
            const double score = (cnt > 0) ? (sum / cnt) : 0.0;
            if (run == 0 || score > best_score) {
                best_score = score;
                best_dec = std::move(dec);
            }
        }
        free(text_embeds);

        if (!params.no_prints && n_runs > 1)
            fprintf(stderr, "crispasr[qwen3]: best-of-%d picked score=%.4f\n", n_runs, best_score);

        const std::vector<int32_t>& gen = best_dec.tokens;
        const std::vector<float>& probs = best_dec.probs;

        // ---- Detokenize via GPT-2 byte decoder ----
        // Qwen3-ASR emits structured metadata tokens before the transcript:
        // special tokens like <|im_start|>, bracketed tags like <asr_text>,
        // and a "language <name>" prefix. Filter all of that out and keep
        // only the transcript itself.
        std::string transcript;
        std::string detected_language;
        bool capture_language = false;
        std::vector<crispasr_token> out_tokens;
        out_tokens.reserve(gen.size());
        for (size_t i = 0; i < gen.size(); i++) {
            const int32_t id = gen[i];
            if (id == eos_id)
                break;
            const char* raw_piece = qwen3_asr_token_text(ctx_, id);
            if (!raw_piece || !*raw_piece)
                continue;
            std::string raw = raw_piece;

            // Skip Qwen3 special tokens: <|im_start|>, <|audio_pad|>, ...
            if (raw.size() >= 2 && raw[0] == '<' && raw[1] == '|')
                continue;
            // Skip structured tags like <asr_text>, <punc>, ...
            if (raw.size() >= 2 && raw[0] == '<' && raw.back() == '>')
                continue;
            // Skip [PAD...] style placeholders if any leaked through.
            if (raw.size() >= 5 && raw[0] == '[' && raw[1] == 'P' && raw[2] == 'A' && raw[3] == 'D')
                continue;

            std::string txt = decode_token(raw);
            if (txt == "language") {
                capture_language = true;
                continue;
            }
            if (capture_language) {
                size_t s = 0;
                while (s < txt.size() && (txt[s] == ' ' || txt[s] == '\t'))
                    s++;
                detected_language = txt.substr(s);
                capture_language = false;
                continue;
            }
            transcript += txt;

            crispasr_token ct;
            ct.id = id;
            ct.text = std::move(txt);
            ct.confidence = (i < probs.size()) ? probs[i] : -1.0f;
            out_tokens.push_back(std::move(ct));
        }

        // Trim leading whitespace left over from the prompt template.
        while (!transcript.empty() && (transcript.front() == ' ' || transcript.front() == '\n')) {
            transcript.erase(transcript.begin());
        }

        if (!params.no_prints && !detected_language.empty()) {
            // Map qwen3's English-name back to ISO-639-1 so downstream
            // tooling (test_lid regex, JSON output) sees a stable code.
            // p=1.000 because qwen3's LID is a deterministic LLM-output
            // capture, not a probabilistic classifier — there's no real
            // confidence to report.
            auto english_to_iso = [](const std::string& n) -> std::string {
                std::string s;
                s.reserve(n.size());
                for (char c : n)
                    s += (char)std::tolower((unsigned char)c);
                if (s == "english")
                    return "en";
                if (s == "german")
                    return "de";
                if (s == "french")
                    return "fr";
                if (s == "spanish")
                    return "es";
                if (s == "italian")
                    return "it";
                if (s == "portuguese")
                    return "pt";
                if (s == "russian")
                    return "ru";
                if (s == "japanese")
                    return "ja";
                if (s == "korean")
                    return "ko";
                if (s == "chinese")
                    return "zh";
                if (s == "dutch")
                    return "nl";
                if (s == "polish")
                    return "pl";
                return s; // fall through — caller tolerates unknown
            };
            const std::string code = english_to_iso(detected_language);
            fprintf(stderr, "crispasr[qwen3]: detected '%s' (p=1.000) via model output\n", code.c_str());
        }

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = transcript;
        crispasr_apply_ngram_loop_fix(seg.text, name(), params.no_prints);
        seg.tokens = std::move(out_tokens);
        out.push_back(std::move(seg));
        return out;
    }

    void transcribe_streaming(const float* samples, int n_samples, int64_t t_offset_cs, const whisper_params& params,
                              crispasr_stream_callback on_text) override {
        (void)t_offset_cs; // For Qwen3 streaming we just stream text output
        if (!ctx_)
            return;

        // ---- Mel ----
        int n_mels = 0, T_mel = 0;
        float* mel = qwen3_asr_compute_mel(ctx_, samples, n_samples, &n_mels, &T_mel);
        if (!mel)
            return;

        // ---- Encoder ----
        int N_enc = 0, pdim = 0;
        float* audio_embeds = qwen3_asr_run_encoder(ctx_, mel, n_mels, T_mel, &N_enc, &pdim);
        free(mel);
        if (!audio_embeds)
            return;

        // ---- Prompt ----
        std::string sys_instruction;
        if (!params.ask.empty()) {
            sys_instruction = params.ask;
        } else if (params.translate) {
            const std::string tgt =
                params.target_lang.empty() ? std::string("English") : crispasr_iso_to_english_lang(params.target_lang);
            sys_instruction = "Translate the speech to " + tgt + ".";
        } else if (!params.language.empty() && params.language != "auto") {
            sys_instruction = "Transcribe the speech in " + crispasr_iso_to_english_lang(params.language) + ".";
        }
        if (!params.hotwords.empty()) {
            if (!sys_instruction.empty() && sys_instruction.back() != ' ')
                sys_instruction += ' ';
            sys_instruction += "The following words may appear in the audio: " + params.hotwords + ".";
        }

        std::string text = "<|im_start|>system\n" + sys_instruction +
                           "<|im_end|>\n"
                           "<|im_start|>user\n"
                           "<|audio_start|>";
        for (int i = 0; i < N_enc; i++)
            text += "<|audio_pad|>";
        text += "<|audio_end|><|im_end|>\n<|im_start|>assistant\n";

        int n_prompt = 0;
        int32_t* raw_ids = qwen3_asr_tokenize(ctx_, text.c_str(), &n_prompt);
        if (!raw_ids) {
            free(audio_embeds);
            return;
        }
        std::vector<int32_t> ids(raw_ids, raw_ids + n_prompt);
        free(raw_ids);

        // Look up the audio_pad token id by tokenizing just the special token.
        int n_pad_id = 0;
        int32_t* pad_id_arr = qwen3_asr_tokenize(ctx_, "<|audio_pad|>", &n_pad_id);
        int audio_pad_id = -1;
        if (pad_id_arr && n_pad_id >= 1)
            audio_pad_id = pad_id_arr[0];
        free(pad_id_arr);
        if (audio_pad_id < 0) {
            free(audio_embeds);
            return;
        }

        // ---- Embed + splice ----
        float* text_embeds = qwen3_asr_embed_tokens(ctx_, ids.data(), (int)ids.size());
        if (!text_embeds) {
            free(audio_embeds);
            return;
        }
        int spliced = 0;
        for (size_t i = 0; i < ids.size() && spliced < N_enc; i++) {
            if (ids[i] == audio_pad_id) {
                std::memcpy(text_embeds + i * pdim, audio_embeds + (size_t)spliced * pdim, pdim * sizeof(float));
                spliced++;
            }
        }
        free(audio_embeds);

        const int prompt_len = (int)ids.size();

        // ---- KV init ----
        if (!qwen3_asr_kv_init(ctx_, 4096)) {
            free(text_embeds);
            return;
        }

        int n_t = 0, vocab = 0;
        float* logits = qwen3_asr_run_llm_kv(ctx_, text_embeds, prompt_len, 0, &n_t, &vocab);
        if (!logits) {
            free(text_embeds);
            return;
        }

        int eos_id = -1;
        int n_eos = 0;
        int32_t* eos_arr = qwen3_asr_tokenize(ctx_, "<|im_end|>", &n_eos);
        if (eos_arr && n_eos >= 1)
            eos_id = eos_arr[0];
        free(eos_arr);
        if (eos_id < 0)
            eos_id = 151645; // Fallback

        core_greedy_decode::Config dec_cfg;
        dec_cfg.max_new_tokens = params.max_new_tokens;
        dec_cfg.eos_id = eos_id;
        dec_cfg.vocab_size = vocab;
        dec_cfg.temperature = params.temperature;
        dec_cfg.frequency_penalty = params.frequency_penalty;
        dec_cfg.seed = params.seed;

        int first_token = 0;
        float first_prob = 1.0f;
        const int last_off = (n_t - 1) * vocab;
        if (params.temperature > 0.0f) {
            std::mt19937_64 seed_rng((params.seed != 0 ? params.seed : (uint64_t)std::random_device{}()) ^
                                     (uint64_t)0x9E3779B97F4A7C15ull);
            first_token = core_greedy_decode::sample_temp(logits + last_off, vocab, params.temperature, seed_rng);
        } else {
            first_token = core_greedy_decode::argmax(logits + last_off, vocab);
        }
        first_prob =
            core_greedy_decode::softmax_of(logits + last_off, vocab, first_token, logits[last_off + first_token]);
        free(logits);

        std::string accumulated_text;
        bool capture_language = false;

        auto token_cb = [&](int32_t id, float prob) {
            (void)prob;
            if (id == eos_id)
                return;
            const char* raw_piece = qwen3_asr_token_text(ctx_, id);
            if (!raw_piece || !*raw_piece)
                return;
            std::string raw = raw_piece;
            if (raw.size() >= 2 && raw[0] == '<' && raw[1] == '|')
                return;
            if (raw.size() >= 2 && raw[0] == '<' && raw.back() == '>')
                return;
            if (raw.size() >= 5 && raw[0] == '[' && raw[1] == 'P' && raw[2] == 'A' && raw[3] == 'D')
                return;
            std::string txt = decode_token(raw);
            if (txt == "language") {
                capture_language = true;
                return;
            }
            if (capture_language) {
                capture_language = false;
                return;
            }
            accumulated_text += txt;
            if (!accumulated_text.empty()) {
                on_text(accumulated_text, false);
            }
        };

        core_greedy_decode::run_with_probs_cb(ctx_, first_token, first_prob, prompt_len, qwen3_asr_embed_tokens,
                                              qwen3_asr_run_llm_kv, token_cb, dec_cfg);

        // Emit final
        if (!accumulated_text.empty()) {
            on_text(accumulated_text, true);
        } else {
            on_text("", true);
        }
    }

    void shutdown() override {
        if (ctx_) {
            qwen3_asr_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    qwen3_asr_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_qwen3_backend() {
    return std::unique_ptr<CrispasrBackend>(new Qwen3Backend());
}
