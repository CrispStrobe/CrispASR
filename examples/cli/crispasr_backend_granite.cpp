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
// The granite-speech-4.1-2b-PLUS variant exposes native word-level timestamps
// ([T:N] tags) and speaker attribution ([Speaker N]: tags) via mode-specific
// instructions — but ONLY through its real control-token chat template (see the
// prompt-construction block below). Base / non-plus variants offer plain ASR +
// translation and use the legacy "USER:/ASSISTANT:" template.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"
#include "core/greedy_decode.h"
#include "core/beam_decode.h"

#include "granite_speech.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <regex>
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
constexpr int32_t kPrefix4[] = {6584, 25, 220}; // "USER: "
constexpr int kNumPrefix4 = 3;
constexpr int32_t kSuffix4[] = { // "can you transcribe..."
    4919, 499, 1380, 3191, 279, 8982, 1139, 264, 5439, 3645, 30, 198, 36660, 3931, 2891, 25};
constexpr int kNumSuffix4 = 16;

// Legacy token ids for granite-4.0-1b. Only used when the GGUF doesn't
// export granite_speech.llm.audio_token_index / eos_token_id (i.e. it
// was produced before that key landed). Both accessors below fall back
// to these when the runtime returns -1.
constexpr int kLegacyAudioTok4 = 100352;
constexpr int kLegacyEos4 = 100257;

class GraniteBackend : public CrispasrBackend {
public:
    GraniteBackend() = default;
    ~GraniteBackend() override { GraniteBackend::shutdown(); }

    const char* name() const override { return "granite"; }

    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_CTC | CAP_WORD_TIMESTAMPS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_PUNCTUATION_TOGGLE |
               CAP_FLASH_ATTN | CAP_TOKEN_CONFIDENCE | CAP_TRANSLATE | CAP_SRC_TGT_LANGUAGE | CAP_DIARIZE |
               CAP_PARALLEL_PROCESSORS | CAP_BEAM_SEARCH;
    }

    bool init(const whisper_params& p) override {
        granite_speech_context_params cp = granite_speech_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        use_gpu_ = cp.use_gpu;

        ctx_ = granite_speech_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[granite]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        // ---- Mel spectrogram ----
        int n_mels = 0, T_mel = 0;
        float* mel = granite_speech_compute_mel(ctx_, samples, n_samples, &n_mels, &T_mel);
        if (!mel) {
            fprintf(stderr, "crispasr[granite]: mel failed\n");
            return out;
        }

        // ---- Encoder ----
        int N_enc = 0, enc_dim = 0;
        float* enc = granite_speech_run_encoder(ctx_, mel, n_mels, T_mel, &N_enc, &enc_dim);
        free(mel);
        if (!enc) {
            fprintf(stderr, "crispasr[granite]: encoder failed\n");
            return out;
        }

        // ---- Q-Former projector ----
        int N_proj = 0, proj_dim = 0;
        float* proj = granite_speech_run_projector(ctx_, enc, N_enc, enc_dim, &N_proj, &proj_dim);
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
        int eos_tok = granite_speech_eos_token_id(ctx_);
        const int vocab_sz = granite_speech_vocab_size(ctx_);
        if (audio_tok < 0)
            audio_tok = kLegacyAudioTok4;
        if (eos_tok < 0)
            eos_tok = kLegacyEos4;
        (void)vocab_sz;

        auto iso_to_eng = [](const std::string& c) -> std::string { return crispasr_iso_to_english_lang(c); };

        // ---- User instruction (mode-specific), shared by both chat templates ----
        //
        // The PLUS variant emits native word timestamps ([T:N] tags, N =
        // end-time centiseconds mod 1000) and speaker tags when asked, using the
        // exact instruction strings from the model card
        // (ibm-granite/granite-speech-4.1-2b-plus).
        const bool is_plus = granite_speech_is_plus(ctx_);
        const bool want_saa = is_plus && params.diarize;
        const bool want_ts = is_plus && (params.output_wts || params.output_jsn_full || params.max_len > 0 ||
                                         params.output_srt || params.output_vtt || params.split_on_punct);
        std::string user_content;
        if (want_saa && want_ts) {
            user_content = "Speaker attribution: Transcribe and denote who is speaking "
                           "by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns. "
                           "After each word, add a timestamp tag showing the end time in "
                           "centiseconds, e.g. hello [T:45] world [T:82]";
        } else if (want_saa) {
            user_content = "Speaker attribution: Transcribe and denote who is speaking "
                           "by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns.";
        } else if (want_ts) {
            user_content = "Timestamps: Transcribe the speech. After each word, add a timestamp "
                           "tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]";
        } else if (!params.ask.empty()) {
            user_content = params.ask;
        } else if (params.translate) {
            const std::string tgt =
                params.target_lang.empty() ? std::string("English") : iso_to_eng(params.target_lang);
            user_content = "can you translate the speech to " + tgt + "?";
        } else if (!params.language.empty() && params.language != "auto") {
            user_content = "can you transcribe the speech into " + iso_to_eng(params.language) + "?";
        }
        // Empty user_content ⇒ default plain ASR; each template supplies its own.

        // #205: keyword/biasing list (KWB). Appending the keywords to an
        // ASR-family instruction biases recognition toward them (model card:
        // "... Keywords: ..."). Skip for translation and fully-custom (--ask)
        // prompts, which are not ASR.
        if (!params.hotwords.empty() && params.ask.empty() && !params.translate) {
            if (user_content.empty())
                user_content = "can you transcribe the speech into a written format?";
            user_content += " Keywords: " + params.hotwords;
        }

        // Chat-template selection.
        //
        // granite-4.0-1b was trained with a simple "USER: …\n ASSISTANT:"
        // format. granite-3.x AND granite-speech-4.1 (incl. the plus variant)
        // use the IBM Granite control-token scheme:
        //   <|start_of_role|>user<|end_of_role|>…<|end_of_text|>
        //   <|start_of_role|>assistant<|end_of_role|>
        //
        // #205: the plus model's timestamp / SAA instructions only work through
        // the control-token template — the legacy "USER:/ASSISTANT:" wrapper
        // makes it ignore the audio and loop ("thank you thank you ..."), even
        // though plain transcription survives the wrong wrapper. The
        // audio_token_index < 50000 heuristic catches granite-3.x; OR-in is_plus
        // so granite-speech-4.1-2b-plus (audio_token_index == 100352, same as
        // granite-4.0) also takes the control-token path.
        const bool use_v3_template = (audio_tok < 50000) || is_plus;

        std::vector<int32_t> prefix_ids, suffix_ids;

        if (use_v3_template) {
            // Runtime-tokenize the control-token template.
            // granite_speech_tokenize() detects <|...|> markers and emits their
            // vocab id directly.
            // Full control-token chat template, byte-for-byte what
            // transformers' apply_chat_template() emits for these models — the
            // default system turn is ALWAYS present (verified against
            // ibm-granite/granite-speech-4.1-2b-plus). Omitting it leaves the
            // model in an undefined state on the demanding timestamp / SAA
            // instructions (#205).
            const std::string prefix_str = "<|start_of_role|>system<|end_of_role|>You are a helpful assistant. Please "
                                           "ensure responses are professional, accurate, and safe.<|end_of_text|>\n"
                                           "<|start_of_role|>user<|end_of_role|>";
            const std::string instr = user_content.empty()
                                          ? std::string("can you transcribe the speech into a written format?")
                                          : user_content;
            // #205: incremental decoding — params.prefix_text seeds the start of
            // the assistant turn so the model continues from it (output is the
            // continuation only). Model card: the SAA `prefix_text` field.
            const std::string suffix_str =
                instr + "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>" + params.prefix_text;
            int n = 0;
            int32_t* a = granite_speech_tokenize(ctx_, prefix_str.c_str(), &n);
            if (a && n > 0) {
                prefix_ids.assign(a, a + n);
                free(a);
            } else if (a)
                free(a);
            a = granite_speech_tokenize(ctx_, suffix_str.c_str(), &n);
            if (a && n > 0) {
                suffix_ids.assign(a, a + n);
                free(a);
            } else if (a)
                free(a);
        } else {
            // granite-4.0/4.1-1b prompt: "USER: …\n ASSISTANT:"
            prefix_ids.assign(kPrefix4, kPrefix4 + kNumPrefix4);
            if (!user_content.empty() || !params.prefix_text.empty()) {
                const std::string base = user_content.empty()
                                             ? std::string("can you transcribe the speech into a written format?")
                                             : user_content;
                // prefix_text (#205) seeds the assistant turn for incremental decoding.
                const std::string instr = base + "\n ASSISTANT:" + params.prefix_text;
                int n = 0;
                int32_t* a = granite_speech_tokenize(ctx_, instr.c_str(), &n);
                if (a && n > 0) {
                    suffix_ids.assign(a, a + n);
                    free(a);
                } else if (a)
                    free(a);
            } else {
                // Default plain ASR — use hardcoded kSuffix4 for granite-4.0
                // compatibility (avoids whitespace-skip tokenization bug).
                suffix_ids.assign(kSuffix4, kSuffix4 + kNumSuffix4);
            }
        }

        if (prefix_ids.empty() || suffix_ids.empty()) {
            fprintf(stderr, "crispasr[granite]: tokenize failed — re-convert the GGUF "
                            "with the newer models/convert-granite-speech-to-gguf.py "
                            "to pick up the merges table\n");
            free(proj);
            return out;
        }

        const int n_prefix = (int)prefix_ids.size();
        const int n_suffix = (int)suffix_ids.size();
        const int total_prompt = n_prefix + N_proj + n_suffix;
        std::vector<int32_t> prompt_ids;
        prompt_ids.reserve(total_prompt);
        for (int id : prefix_ids)
            prompt_ids.push_back(id);
        for (int i = 0; i < N_proj; i++)
            prompt_ids.push_back(audio_tok);
        for (int id : suffix_ids)
            prompt_ids.push_back(id);

        float* all_embeds = granite_speech_embed_tokens(ctx_, prompt_ids.data(), total_prompt);
        if (!all_embeds) {
            free(proj);
            fprintf(stderr, "crispasr[granite]: embed failed\n");
            return out;
        }

        // Splice projector output into the audio positions (skip the prefix).
        for (int i = 0; i < N_proj; i++) {
            std::memcpy(all_embeds + (size_t)(n_prefix + i) * proj_dim, proj + (size_t)i * proj_dim,
                        proj_dim * sizeof(float));
        }
        free(proj);

        // ---- KV cache + best-of-N decode ----
        if (!granite_speech_kv_init(ctx_, 4096)) {
            free(all_embeds);
            fprintf(stderr, "crispasr[granite]: kv init failed\n");
            return out;
        }

        const int max_new = params.max_new_tokens > 0 ? params.max_new_tokens : (want_ts ? 4096 : 200);

        // ---- Beam search path ----
        if (params.beam_size > 1) {
            granite_speech_kv_reset(ctx_);
            int vocab = 0;
            float* logits = granite_speech_run_llm_kv(ctx_, all_embeds, total_prompt, 0, nullptr, &vocab);
            if (!logits) {
                fprintf(stderr, "crispasr[granite]: prefill failed\n");
                free(all_embeds);
                return out;
            }
            free(all_embeds);

            auto replay = [this](granite_speech_context* /*ctx*/, const int32_t* toks, int n,
                                 int prompt_len) -> float* {
                float* emb = granite_speech_embed_tokens(ctx_, toks, n);
                if (!emb)
                    return nullptr;
                int v = 0;
                float* lg = granite_speech_run_llm_kv(ctx_, emb, n, prompt_len, nullptr, &v);
                std::free(emb);
                return lg;
            };

            core_beam_decode::Config cfg;
            cfg.max_new_tokens = max_new;
            cfg.eos_id = eos_tok;
            cfg.vocab_size = vocab;
            cfg.beam_size = params.beam_size;
            cfg.prompt_len = total_prompt;

            auto beam_r = core_beam_decode::run_with_probs(ctx_, logits, replay, cfg);
            free(logits);

            // Convert beam result to greedy result format for shared detokenize path
            core_greedy_decode::Result best_dec;
            best_dec.tokens.assign(beam_r.tokens.begin(), beam_r.tokens.end());
            best_dec.probs = std::move(beam_r.probs);

            const std::vector<int32_t>& gen_ids = best_dec.tokens;
            const std::vector<float>& probs = best_dec.probs;
            // fall through to detokenize below via goto-equivalent scope
            // (duplicated for clarity — the alternative is a lambda)
            std::vector<int32_t> text_ids;
            text_ids.reserve(gen_ids.size());
            for (int32_t id : gen_ids)
                if (id != eos_tok)
                    text_ids.push_back(id);

            char* text = granite_speech_decode_tokens(ctx_, text_ids.data(), (int)text_ids.size());
            std::string transcript = text ? text : "";
            if (text)
                free(text);

            crispasr_segment seg;
            seg.t0 = t_offset_cs;
            seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
            // Strip leading space
            size_t start = 0;
            while (start < transcript.size() && transcript[start] == ' ')
                start++;
            seg.text = transcript.substr(start);
            seg.tokens.reserve(gen_ids.size());
            for (size_t i = 0; i < gen_ids.size(); i++) {
                if (gen_ids[i] == eos_tok)
                    break;
                crispasr_token ct;
                ct.id = gen_ids[i];
                ct.confidence = (i < probs.size()) ? probs[i] : -1.0f;
                seg.tokens.push_back(std::move(ct));
            }
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
        dec_cfg.eos_id = eos_tok;
        dec_cfg.temperature = params.temperature;
        dec_cfg.frequency_penalty = params.frequency_penalty;
        dec_cfg.seed = params.seed;

        // GPU only: enable the bucketed CUDA-graph-capture decode. Beam
        // search and CPU keep the legacy per-step path.
        const bool use_bucket = use_gpu_;
        if (use_bucket) {
            const int bucket = std::min(total_prompt + max_new + 1, 4096);
            granite_speech_set_decode_bucket(ctx_, bucket);
        }

        const int n_runs = (params.temperature > 0.0f && params.best_of > 1) ? params.best_of : 1;
        core_greedy_decode::Result best_dec;
        double best_score = -1.0;

        for (int run = 0; run < n_runs; run++) {
            granite_speech_kv_reset(ctx_);

            int vocab = 0;
            float* logits = granite_speech_run_llm_kv(ctx_, all_embeds, total_prompt, 0, nullptr, &vocab);
            if (!logits) {
                fprintf(stderr, "crispasr[granite]: prefill failed (run %d/%d)\n", run + 1, n_runs);
                free(all_embeds);
                return out;
            }
            if (run == 0)
                dec_cfg.vocab_size = vocab;

            int next = 0;
            float next_p = 1.0f;
            if (dec_cfg.temperature > 0.0f) {
                std::mt19937_64 seed_rng((dec_cfg.seed != 0 ? dec_cfg.seed : (uint64_t)std::random_device{}()) ^
                                         (uint64_t)(run * 0x9E3779B97F4A7C15ull));
                next = core_greedy_decode::sample_temp(logits, vocab, dec_cfg.temperature, seed_rng);
            } else {
                next = core_greedy_decode::argmax(logits, vocab);
            }
            next_p = core_greedy_decode::softmax_of(logits, vocab, next, logits[next]);
            free(logits);

            // Greedy + single-run: use the argmax-fused decode graph
            // (bit-identical token selection). Sampling/best-of-N fall through
            // to the shared loop (they need full logits + per-token probs).
            const bool fast_greedy = (n_runs == 1) && (dec_cfg.temperature == 0.0f) && use_bucket;
            if (fast_greedy) {
                int n_out = 0;
                int32_t* ids = granite_speech_greedy_decode(ctx_, /*first_token=*/next, /*initial_n_past=*/total_prompt,
                                                            /*max_new_tokens=*/max_new, eos_tok, &n_out);
                if (ids && n_out > 0) {
                    core_greedy_decode::Result r;
                    r.tokens.assign(ids, ids + n_out);
                    r.probs.assign(n_out, 1.0f); // probs unused when n_runs == 1
                    free(ids);
                    best_score = 1.0;
                    best_dec = std::move(r);
                    continue; // skip the shared loop this run
                }
                free(ids); // NULL on failure -> fall through to shared loop
            }

            auto dec = core_greedy_decode::run_with_probs(ctx_,
                                                          /*first_token=*/next,
                                                          /*first_prob=*/next_p,
                                                          /*initial_n_past=*/total_prompt, granite_speech_embed_tokens,
                                                          granite_speech_run_llm_kv, dec_cfg);

            double sum = 0.0;
            int cnt = 0;
            for (size_t i = 0; i < dec.probs.size(); i++) {
                if ((int32_t)dec.tokens[i] == eos_tok)
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
        free(all_embeds);

        if (!params.no_prints && n_runs > 1)
            fprintf(stderr, "crispasr[granite]: best-of-%d picked score=%.4f\n", n_runs, best_score);

        const std::vector<int32_t>& gen_ids = best_dec.tokens;
        const std::vector<float>& probs = best_dec.probs;

        // Strip EOS from generated IDs before detokenizing.
        std::vector<int32_t> text_ids;
        text_ids.reserve(gen_ids.size());
        for (int32_t id : gen_ids)
            if (id != eos_tok)
                text_ids.push_back(id);

        char* text = granite_speech_decode_tokens(ctx_, text_ids.data(), (int)text_ids.size());

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = text ? text : "";
        if (text)
            free(text);

        // Trim leading whitespace emitted by the chat template.
        while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n')) {
            seg.text.erase(seg.text.begin());
        }

        // ---- PLUS post-processing: parse structured output tags ----
        // Parse [T:N] end-time tags (N = centiseconds, emitted mod 1000) out of
        // `text` into words + clean text. The mod-1000 rollover state
        // (ts_last_t / ts_offset_cs) is threaded across calls so absolute times
        // stay monotonic even when parsing per-speaker chunks (#205: combined
        // SAA + timestamps used to leak the raw [T:N] tags into the text).
        int64_t ts_last_t = t_offset_cs;
        int64_t ts_offset_cs = 0;
        auto parse_ts = [&](const std::string& text, std::vector<crispasr_word>& words) -> std::string {
            // The word is any run of non-space, non-'[' characters. \S+ here is
            // greedy across the bracket: when the model emits run-on word/tag
            // pairs with no spaces ("if[T:7962]you[T:7970]got"), it collapses
            // the whole run into one token, producing the spaceless "ifyougot"
            // text (#205). Stopping at '[' splits each word correctly whether or
            // not the model spaces the pairs; the spaced form ("hello [T:45]
            // world") is unchanged.
            static const std::regex ts_re(R"(([^\[\s]+)\s*\[T:(\d+)\])");
            std::sregex_iterator it(text.begin(), text.end(), ts_re);
            std::sregex_iterator end;
            std::string clean;
            for (; it != end; ++it) {
                std::string word = (*it)[1].str();
                int64_t raw_cs = std::stoll((*it)[2].str());
                int64_t abs_cs = raw_cs + ts_offset_cs + t_offset_cs;
                while (abs_cs < ts_last_t)
                    abs_cs += 1000, ts_offset_cs += 1000;
                if (word != "_") { // skip silence tokens
                    crispasr_word w;
                    w.text = word;
                    w.t0 = ts_last_t;
                    w.t1 = abs_cs;
                    words.push_back(std::move(w));
                    if (!clean.empty())
                        clean += " ";
                    clean += word;
                }
                ts_last_t = abs_cs;
            }
            return clean;
        };

        if (want_saa && !seg.text.empty()) {
            // Split on [Speaker N]: markers → one segment per speaker turn,
            // parsing any [T:N] tags inside each turn when timestamps are also on.
            static const std::regex spk_re(R"(\[Speaker\s+(\d+)\]\s*:\s*)");
            std::sregex_iterator it(seg.text.begin(), seg.text.end(), spk_re);
            std::sregex_iterator end;
            std::string current_spk;
            size_t last_pos = 0;
            std::vector<crispasr_segment> saa_segs;

            auto emit = [&](std::string chunk) {
                while (!chunk.empty() && chunk.back() == ' ')
                    chunk.pop_back();
                if (chunk.empty() || current_spk.empty())
                    return;
                crispasr_segment s;
                s.speaker = current_spk;
                if (want_ts) {
                    std::vector<crispasr_word> words;
                    std::string clean = parse_ts(chunk, words);
                    if (!words.empty()) {
                        s.t0 = words.front().t0;
                        s.t1 = words.back().t1;
                        s.text = std::move(clean);
                        s.words = std::move(words);
                        saa_segs.push_back(std::move(s));
                        return;
                    }
                }
                s.t0 = seg.t0;
                s.t1 = seg.t1;
                s.text = std::move(chunk);
                saa_segs.push_back(std::move(s));
            };

            for (; it != end; ++it) {
                // Text before this speaker tag belongs to the previous speaker
                size_t match_start = (size_t)it->position();
                if (match_start > last_pos && !current_spk.empty())
                    emit(seg.text.substr(last_pos, match_start - last_pos));
                current_spk = "Speaker " + (*it)[1].str();
                last_pos = (size_t)(it->position() + it->length());
            }
            // Remaining text after last speaker tag
            if (last_pos < seg.text.size() && !current_spk.empty())
                emit(seg.text.substr(last_pos));
            if (!saa_segs.empty()) {
                out = std::move(saa_segs);
                return out;
            }
        }

        if (want_ts && !seg.text.empty()) {
            std::vector<crispasr_word> words;
            std::string clean = parse_ts(seg.text, words);
            if (!words.empty()) {
                seg.words = std::move(words);
                seg.text = std::move(clean);
            } else if (seg.text.find("[T:") != std::string::npos) {
                // Near-empty chunk that emitted only silence markers
                // ("_ [T:179]"). Strip the residual [T:N] tags + bare "_" so the
                // junk doesn't leak into the transcript (#205).
                static const std::regex tag_re(R"(\[T:\d+\])");
                seg.text = std::regex_replace(seg.text, tag_re, "");
                while (!seg.text.empty() && (seg.text.back() == ' ' || seg.text.back() == '\t'))
                    seg.text.pop_back();
                while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\t'))
                    seg.text.erase(seg.text.begin());
                if (seg.text == "_")
                    seg.text.clear();
            }
        }

        // Per-token entries with decode-loop confidences. granite uses
        // its own batch detokenizer (granite_speech_decode_tokens) for
        // the segment text, but we still surface per-token id + prob so
        // downstream consumers (JSON full, confidence filters) have the
        // raw signal. Token text is intentionally left empty here to
        // avoid duplicating the batch detokenizer's merging logic.
        seg.tokens.reserve(gen_ids.size());
        for (size_t i = 0; i < gen_ids.size(); i++) {
            if (gen_ids[i] == eos_tok)
                break;
            crispasr_token ct;
            ct.id = gen_ids[i];
            ct.confidence = (i < probs.size()) ? probs[i] : -1.0f;
            seg.tokens.push_back(std::move(ct));
        }

        out.push_back(std::move(seg));
        return out;
    }

    void transcribe_streaming(const float* samples, int n_samples, int64_t t_offset_cs, const whisper_params& params,
                              crispasr_stream_callback on_text) override {
        if (!ctx_)
            return;

        // Beam search, SAA, and timestamp post-processing require all tokens before
        // output can be structured — fall back to the batch transcribe() base path.
        const bool is_plus = granite_speech_is_plus(ctx_);
        const bool want_saa = is_plus && params.diarize;
        const bool want_ts = is_plus && (params.output_wts || params.output_jsn_full || params.max_len > 0 ||
                                         params.output_srt || params.output_vtt || params.split_on_punct);
        if (params.beam_size > 1 || want_saa || want_ts) {
            CrispasrBackend::transcribe_streaming(samples, n_samples, t_offset_cs, params, on_text);
            return;
        }

        // ---- Mel → encoder → projector ----
        int n_mels = 0, T_mel = 0;
        float* mel = granite_speech_compute_mel(ctx_, samples, n_samples, &n_mels, &T_mel);
        if (!mel)
            return;
        int N_enc = 0, enc_dim = 0;
        float* enc = granite_speech_run_encoder(ctx_, mel, n_mels, T_mel, &N_enc, &enc_dim);
        free(mel);
        if (!enc)
            return;
        int N_proj = 0, proj_dim = 0;
        float* proj = granite_speech_run_projector(ctx_, enc, N_enc, enc_dim, &N_proj, &proj_dim);
        free(enc);
        if (!proj)
            return;

        // ---- Prompt (mirrors transcribe() but without SAA/TS branches) ----
        int audio_tok = granite_speech_audio_token_id(ctx_);
        int eos_tok = granite_speech_eos_token_id(ctx_);
        if (audio_tok < 0)
            audio_tok = kLegacyAudioTok4;
        if (eos_tok < 0)
            eos_tok = kLegacyEos4;

        // Streaming reaches here only for plain ASR (SAA / TS / beam fall back
        // to the batch path above), so build the ASR-family instruction —
        // language / translate / --ask, plus KWB and incremental decoding —
        // exactly as transcribe() does, including the full control-token chat
        // template with system turn (#205).
        auto iso_to_eng = [](const std::string& c) -> std::string { return crispasr_iso_to_english_lang(c); };
        std::string user_content;
        if (!params.ask.empty()) {
            user_content = params.ask;
        } else if (params.translate) {
            const std::string tgt =
                params.target_lang.empty() ? std::string("English") : iso_to_eng(params.target_lang);
            user_content = "can you translate the speech to " + tgt + "?";
        } else if (!params.language.empty() && params.language != "auto") {
            user_content = "can you transcribe the speech into " + iso_to_eng(params.language) + "?";
        }
        if (!params.hotwords.empty() && params.ask.empty() && !params.translate) {
            if (user_content.empty())
                user_content = "can you transcribe the speech into a written format?";
            user_content += " Keywords: " + params.hotwords;
        }

        const bool use_v3_template = (audio_tok < 50000) || is_plus;
        std::vector<int32_t> prefix_ids, suffix_ids;
        if (use_v3_template) {
            const std::string prefix_str = "<|start_of_role|>system<|end_of_role|>You are a helpful assistant. Please "
                                           "ensure responses are professional, accurate, and safe.<|end_of_text|>\n"
                                           "<|start_of_role|>user<|end_of_role|>";
            const std::string instr = user_content.empty()
                                          ? std::string("can you transcribe the speech into a written format?")
                                          : user_content;
            const std::string suffix_str =
                instr + "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>" + params.prefix_text;
            int n = 0;
            int32_t* a = granite_speech_tokenize(ctx_, prefix_str.c_str(), &n);
            if (a && n > 0) {
                prefix_ids.assign(a, a + n);
                free(a);
            } else if (a)
                free(a);
            a = granite_speech_tokenize(ctx_, suffix_str.c_str(), &n);
            if (a && n > 0) {
                suffix_ids.assign(a, a + n);
                free(a);
            } else if (a)
                free(a);
        } else {
            prefix_ids.assign(kPrefix4, kPrefix4 + kNumPrefix4);
            if (!user_content.empty() || !params.prefix_text.empty()) {
                const std::string base = user_content.empty()
                                             ? std::string("can you transcribe the speech into a written format?")
                                             : user_content;
                const std::string instr = base + "\n ASSISTANT:" + params.prefix_text;
                int n = 0;
                int32_t* a = granite_speech_tokenize(ctx_, instr.c_str(), &n);
                if (a && n > 0) {
                    suffix_ids.assign(a, a + n);
                    free(a);
                } else if (a)
                    free(a);
            } else {
                suffix_ids.assign(kSuffix4, kSuffix4 + kNumSuffix4);
            }
        }
        if (prefix_ids.empty() || suffix_ids.empty()) {
            free(proj);
            return;
        }

        const int n_prefix = (int)prefix_ids.size();
        const int n_suffix = (int)suffix_ids.size();
        const int total_prompt = n_prefix + N_proj + n_suffix;
        std::vector<int32_t> prompt_ids;
        prompt_ids.reserve(total_prompt);
        for (int id : prefix_ids)
            prompt_ids.push_back(id);
        for (int i = 0; i < N_proj; i++)
            prompt_ids.push_back(audio_tok);
        for (int id : suffix_ids)
            prompt_ids.push_back(id);

        float* all_embeds = granite_speech_embed_tokens(ctx_, prompt_ids.data(), total_prompt);
        if (!all_embeds) {
            free(proj);
            return;
        }
        for (int i = 0; i < N_proj; i++)
            std::memcpy(all_embeds + (size_t)(n_prefix + i) * proj_dim, proj + (size_t)i * proj_dim,
                        (size_t)proj_dim * sizeof(float));
        free(proj);

        // ---- Prefill ----
        if (!granite_speech_kv_init(ctx_, 4096)) {
            free(all_embeds);
            return;
        }
        int vocab = 0;
        float* logits = granite_speech_run_llm_kv(ctx_, all_embeds, total_prompt, 0, nullptr, &vocab);
        free(all_embeds);
        if (!logits)
            return;

        core_greedy_decode::Config dec_cfg;
        dec_cfg.max_new_tokens = params.max_new_tokens > 0 ? params.max_new_tokens : 200;
        dec_cfg.eos_id = eos_tok;
        dec_cfg.vocab_size = vocab;
        dec_cfg.temperature = params.temperature;
        dec_cfg.frequency_penalty = params.frequency_penalty;
        dec_cfg.seed = params.seed;

        int first_token = 0;
        float first_prob = 1.0f;
        if (params.temperature > 0.0f) {
            std::mt19937_64 rng((params.seed != 0 ? params.seed : (uint64_t)std::random_device{}()));
            first_token = core_greedy_decode::sample_temp(logits, vocab, params.temperature, rng);
        } else {
            first_token = core_greedy_decode::argmax(logits, vocab);
        }
        first_prob = core_greedy_decode::softmax_of(logits, vocab, first_token, logits[first_token]);
        free(logits);

        // ---- Streaming decode ----
        std::string accumulated;
        bool leading_space_trimmed = false;

        // Enable bucketed CUDA-graph-capture decode for the streaming loop too.
        if (use_gpu_) {
            const int bucket = std::min(total_prompt + dec_cfg.max_new_tokens + 1, 4096);
            granite_speech_set_decode_bucket(ctx_, bucket);
        }

        auto token_cb = [&](int32_t id, float /*prob*/) {
            if (id == eos_tok)
                return;
            const char* piece = granite_speech_token_text(ctx_, id);
            if (!piece || !*piece)
                return;
            std::string txt(piece);
            // Trim leading whitespace from the very first emitted text only
            if (!leading_space_trimmed) {
                while (!txt.empty() && (txt.front() == ' ' || txt.front() == '\n'))
                    txt.erase(txt.begin());
                if (!txt.empty())
                    leading_space_trimmed = true;
            }
            if (txt.empty())
                return;
            accumulated += txt;
            on_text(accumulated, false);
        };

        core_greedy_decode::run_with_probs_cb(ctx_, first_token, first_prob, total_prompt, granite_speech_embed_tokens,
                                              granite_speech_run_llm_kv, token_cb, dec_cfg);

        on_text(accumulated, true);
    }

    void shutdown() override {
        if (ctx_) {
            granite_speech_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    granite_speech_context* ctx_ = nullptr;
    bool use_gpu_ = false;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_granite_backend() {
    return std::unique_ptr<CrispasrBackend>(new GraniteBackend());
}
