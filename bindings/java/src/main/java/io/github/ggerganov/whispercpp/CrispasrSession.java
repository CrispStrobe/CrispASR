package io.github.ggerganov.whispercpp;

import com.sun.jna.Callback;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.DoubleByReference;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.LongByReference;

/**
 * Minimal TTS surface for the Java binding. Exposes the unified
 * CrispASR Session API for TTS-capable backends (kokoro, vibevoice,
 * qwen3-tts, orpheus) plus the kokoro per-language model + voice
 * resolver (PLAN #56 opt 2b).
 *
 * <p>Usage:
 * <pre>{@code
 * CrispasrSession.Resolved r = CrispasrSession.kokoroResolveForLang(
 *     "/models/kokoro-82m-q8_0.gguf", "de");
 * try (CrispasrSession s = CrispasrSession.open(r.modelPath, 4)) {
 *     if (r.voicePath != null) s.setVoice(r.voicePath, null);
 *     float[] pcm = s.synthesize("Guten Tag");
 *     // ... write WAV ...
 * }
 * }</pre>
 */
public final class CrispasrSession implements AutoCloseable {

    public interface Lib extends Library {
        Lib INSTANCE = Native.load("crispasr", Lib.class);

        Pointer crispasr_session_open(String modelPath, int nThreads);
        void    crispasr_session_close(Pointer session);
        int     crispasr_session_set_codec_path(Pointer session, String path);
        int     crispasr_session_set_voice(Pointer session, String path, String refTextOrNull);
        int     crispasr_session_set_speaker_name(Pointer session, String name);
        int     crispasr_session_n_speakers(Pointer session);
        String  crispasr_session_get_speaker_name(Pointer session, int i);
        int     crispasr_session_set_instruct(Pointer session, String instruct);
        int     crispasr_session_is_custom_voice(Pointer session);
        int     crispasr_session_is_voice_design(Pointer session);
        Pointer crispasr_session_synthesize(Pointer session, String text, IntByReference outNSamples);
        void    crispasr_pcm_free(Pointer pcm);
        int     crispasr_session_kokoro_clear_phoneme_cache(Pointer session);
        int     crispasr_session_set_source_language(Pointer session, String lang);
        int     crispasr_session_set_target_language(Pointer session, String lang);
        int     crispasr_session_set_punctuation(Pointer session, int enable);
        int     crispasr_session_set_translate(Pointer session, int enable);
        int     crispasr_session_set_temperature(Pointer session, float temperature, long seed);
        int     crispasr_session_set_tts_seed(Pointer session, long seed);
        int     crispasr_session_set_max_new_tokens(Pointer session, int maxNewTokens);
        int     crispasr_session_set_frequency_penalty(Pointer session, float penalty);
        int     crispasr_session_set_tts_steps(Pointer session, int steps);
        int     crispasr_session_set_top_p(Pointer session, float topP);
        int     crispasr_session_set_min_p(Pointer session, float minP);
        int     crispasr_session_set_repetition_penalty(Pointer session, float r);
        int     crispasr_session_set_cfg_weight(Pointer session, float cfgWeight);
        int     crispasr_session_set_exaggeration(Pointer session, float exaggeration);
        int     crispasr_session_set_max_speech_tokens(Pointer session, int n);
        int     crispasr_session_set_length_scale(Pointer session, float scale);
        int     crispasr_session_set_best_of(Pointer session, int n);
        int     crispasr_session_set_beam_size(Pointer session, int n);
        int     crispasr_session_set_grammar_text(Pointer session, String gbnfText, String rootRule, float penalty);
        int     crispasr_session_set_fallback_thresholds(Pointer session, float entropyThold,
                                                          float logprobThold, float noSpeechThold,
                                                          float temperatureInc);
        int     crispasr_session_set_alt_n(Pointer session, int n);
        int     crispasr_session_set_whisper_decode_extras(Pointer session, int suppressNst,
                                                            String suppressRegex, int carryInitialPrompt);
        int     crispasr_session_set_ask(Pointer session, String prompt);
        int     crispasr_session_detect_language(Pointer session, float[] pcm, int n_samples,
                                                  String lid_model_path, int method,
                                                  byte[] out_lang, int out_lang_cap, float[] out_prob);

        int crispasr_kokoro_resolve_model_for_lang_abi(
                String modelPath, String lang, byte[] outPath, int outPathLen);
        int crispasr_kokoro_resolve_fallback_voice_abi(
                String modelPath, String lang,
                byte[] outPath, int outPathLen,
                byte[] outPicked, int outPickedLen);

        // --- ASR transcription (PLAN #59) ---
        Pointer crispasr_session_transcribe(Pointer session, float[] pcm, int nSamples);
        Pointer crispasr_session_transcribe_lang(Pointer session, float[] pcm, int nSamples, String language);
        Pointer crispasr_session_transcribe_vad(Pointer session, float[] pcm, int nSamples,
                                                 int sampleRate, String vadModelPath, Pointer opts);
        int          crispasr_session_result_n_segments(Pointer result);
        String       crispasr_session_result_segment_text(Pointer result, int i);
        long         crispasr_session_result_segment_t0(Pointer result, int i);
        long         crispasr_session_result_segment_t1(Pointer result, int i);
        int          crispasr_session_result_n_words(Pointer result, int iSeg);
        String       crispasr_session_result_word_text(Pointer result, int iSeg, int iWord);
        long         crispasr_session_result_word_t0(Pointer result, int iSeg, int iWord);
        long         crispasr_session_result_word_t1(Pointer result, int iSeg, int iWord);
        float        crispasr_session_result_word_p(Pointer result, int iSeg, int iWord);
        void         crispasr_session_result_free(Pointer result);

        // --- Alignment (PLAN #59) ---
        Pointer crispasr_align_words_abi(String alignerModel, String transcript,
                                         float[] samples, int nSamples, long tOffsetCs, int nThreads);
        int    crispasr_align_result_n_words(Pointer result);
        String crispasr_align_result_word_text(Pointer result, int i);
        long   crispasr_align_result_word_t0(Pointer result, int i);
        long   crispasr_align_result_word_t1(Pointer result, int i);
        void   crispasr_align_result_free(Pointer result);

        // --- Standalone LID (PLAN #59) ---
        int crispasr_detect_language_pcm(float[] samples, int nSamples, int method,
                                         String modelPath, int nThreads, int useGpu,
                                         int gpuDevice, int flashAttn,
                                         byte[] outLang, int outLangCap, float[] outProb);

        // --- VAD (PLAN #59) ---
        int crispasr_vad_segments(String vadModelPath, float[] pcm, int nSamples,
                                  int sampleRate, float threshold, int minSpeechMs, int minSilenceMs,
                                  int nThreads, int useGpu, Pointer[] outSpans);
        void crispasr_vad_free(Pointer spans);

        // --- Punctuation (PLAN #59) ---
        Pointer crispasr_punc_init(String modelPath);
        String  crispasr_punc_process(Pointer ctx, String text);
        void    crispasr_punc_free_text(String text);
        void    crispasr_punc_free(Pointer ctx);

        // --- Registry + cache (PLAN #59) ---
        int crispasr_registry_lookup_abi(String backend, byte[] outFilename, int filenameCap,
                                         byte[] outUrl, int urlCap, byte[] outSize, int sizeCap);
        int crispasr_cache_ensure_file_abi(String filename, String url, int quiet,
                                           String cacheDirOverride, byte[] outBuf, int outCap);
        int crispasr_cache_dir_abi(String cacheDirOverride, byte[] outBuf, int outCap);

        // --- Streaming (PLAN #62b) — rolling-window decoder, whisper-only today.
        Pointer crispasr_session_stream_open(Pointer session, int nThreads, int stepMs,
                                             int lengthMs, int keepMs, String language, int translate);
        int     crispasr_stream_feed(Pointer stream, float[] pcm, int nSamples);
        int     crispasr_stream_get_text(Pointer stream, byte[] outText, int outCap,
                                         DoubleByReference outT0, DoubleByReference outT1,
                                         LongByReference outCounter);
        int     crispasr_stream_flush(Pointer stream);
        void    crispasr_stream_close(Pointer stream);

        // --- Microphone capture (PLAN #62d) — miniaudio-backed cross-platform.
        Pointer crispasr_mic_open(int sampleRate, int channels, MicCallback cb, Pointer userdata);
        int     crispasr_mic_start(Pointer mic);
        int     crispasr_mic_stop(Pointer mic);
        void    crispasr_mic_close(Pointer mic);
        String  crispasr_mic_default_device_name();

        // --- Full C-ABI parity additions ---
        void crispasr_stream_set_live_decode(Pointer stream, int enabled);

        // params_set_* on whisper_full_params
        void crispasr_params_set_language(Pointer p, String lang);
        void crispasr_params_set_translate(Pointer p, int v);
        void crispasr_params_set_detect_language(Pointer p, int v);
        void crispasr_params_set_token_timestamps(Pointer p, int v);
        void crispasr_params_set_n_threads(Pointer p, int n);
        void crispasr_params_set_max_len(Pointer p, int n);
        void crispasr_params_set_best_of(Pointer p, int n);
        void crispasr_params_set_split_on_word(Pointer p, int v);
        void crispasr_params_set_no_context(Pointer p, int v);
        void crispasr_params_set_single_segment(Pointer p, int v);
        void crispasr_params_set_print_realtime(Pointer p, int v);
        void crispasr_params_set_print_progress(Pointer p, int v);
        void crispasr_params_set_print_timestamps(Pointer p, int v);
        void crispasr_params_set_print_special(Pointer p, int v);
        void crispasr_params_set_suppress_blank(Pointer p, int v);
        void crispasr_params_set_temperature(Pointer p, float t);
        void crispasr_params_set_max_tokens(Pointer p, int n);
        void crispasr_params_set_initial_prompt(Pointer p, String prompt);
        void crispasr_params_set_alt_n(Pointer p, int n);
        void crispasr_params_set_vad(Pointer p, int v);
        void crispasr_params_set_vad_model_path(Pointer p, String path);
        void crispasr_params_set_vad_threshold(Pointer p, float t);
        void crispasr_params_set_vad_min_speech_ms(Pointer p, int ms);
        void crispasr_params_set_vad_min_silence_ms(Pointer p, int ms);
        void crispasr_params_set_tdrz(Pointer p, int v);

        // Token-level accessors
        long  crispasr_token_t0(Pointer ctx, int iSeg, int iTok);
        long  crispasr_token_t1(Pointer ctx, int iSeg, int iTok);
        float crispasr_token_p(Pointer ctx, int iSeg, int iTok);
        int   crispasr_token_n_alts(Pointer ctx, int iSeg, int iTok);
        int   crispasr_token_alt_id(Pointer ctx, int iSeg, int iTok, int iAlt);
        float crispasr_token_alt_p(Pointer ctx, int iSeg, int iTok, int iAlt);
        int   crispasr_token_alt_text(Pointer ctx, int iSeg, int iTok, int iAlt, byte[] out, int outCap);

        // Language detection (whisper context)
        float crispasr_detect_language(Pointer ctx, float[] pcm, int nSamples,
                                       int nThreads, byte[] outCode, int outCap);

        // VAD slices
        int crispasr_vad_slices(String vadModelPath, float[] pcm, int nSamples,
                                int sampleRate, float threshold, int minSpeechMs, int minSilenceMs,
                                int speechPadMs, float maxChunkDurationS, int nThreads,
                                Pointer[] outSpans);

        // LCS dedup
        int crispasr_lcs_dedup_prefix_count(int[] prevTailTokens, int nPrev,
                                            int[] currTokens, int nCurr, int minLcsLength);

        // Direct Parakeet API
        Pointer crispasr_parakeet_init(String modelPath, int nThreads, int useFlash);
        void    crispasr_parakeet_free(Pointer ctx);
        Pointer crispasr_parakeet_transcribe(Pointer ctx, float[] pcm, int nSamples, String language);
        String  crispasr_parakeet_result_text(Pointer r);
        int     crispasr_parakeet_result_n_words(Pointer r);
        String  crispasr_parakeet_result_word_text(Pointer r, int i);
        long    crispasr_parakeet_result_word_t0(Pointer r, int i);
        long    crispasr_parakeet_result_word_t1(Pointer r, int i);
        int     crispasr_parakeet_result_n_tokens(Pointer r);
        String  crispasr_parakeet_result_token_text(Pointer r, int i);
        long    crispasr_parakeet_result_token_t0(Pointer r, int i);
        long    crispasr_parakeet_result_token_t1(Pointer r, int i);
        float   crispasr_parakeet_result_token_p(Pointer r, int i);
        void    crispasr_parakeet_result_free(Pointer r);

        // TitaNet
        Pointer crispasr_titanet_init(String modelPath, int nThreads);
        void    crispasr_titanet_free(Pointer ctx);
        int     crispasr_titanet_embed(Pointer ctx, float[] pcm16k, int nSamples, float[] out);
        float   crispasr_titanet_cosine_sim(float[] a, float[] b, int dim);

        // Speaker database
        Pointer crispasr_speaker_db_load(String dirPath);
        void    crispasr_speaker_db_free(Pointer db);
        int     crispasr_speaker_db_count(Pointer db);
        float   crispasr_speaker_db_match(Pointer db, float[] embedding, int dim,
                                          float threshold, byte[] outName, int outCap);
        int     crispasr_speaker_db_enroll(String dirPath, String name, float[] embedding, int dim);

        // Pluggable speaker embedder + clustering + pyannote cache
        Pointer crispasr_speaker_embedder_make_abi(String modelSpec, int nThreads, String cacheDir);
        void    crispasr_speaker_embedder_free_abi(Pointer embedder);
        int     crispasr_speaker_embedder_dim_abi(Pointer embedder);
        int     crispasr_speaker_embedder_embed_abi(Pointer embedder, float[] pcm16k, int nSamples, float[] out);
        String  crispasr_speaker_embedder_name_abi(Pointer embedder);
        int     crispasr_speaker_cluster_abi(float[] embeddings, int n, int dim,
                                            float mergeThreshold, int maxSpeakers, int[] labelsOut);
        Pointer crispasr_pyannote_cache_compute_abi(float[] fullAudio, int nSamples,
                                                    String modelPath, int nThreads);
        void    crispasr_pyannote_cache_free_abi(Pointer cache);
        int     crispasr_pyannote_cache_apply_abi(Pointer cache, long sliceT0Cs,
                                                  Pointer segs, int nSegs);

        // Kokoro lang helpers
        boolean crispasr_kokoro_lang_is_german_abi(String lang);
        boolean crispasr_kokoro_lang_has_native_voice_abi(String lang);

        // Backend detection
        int crispasr_detect_backend_from_gguf(String path, byte[] outName, int outCap);

        // RNNoise audio enhancement
        int crispasr_enhance_audio_rnnoise(float[] inPcm, int nSamples, float[] outPcm, int outCap);

        // Diarization
        int crispasr_diarize_segments_abi(float[] leftPcm, float[] rightPcm, int nSamples,
                                          int isStereo, Pointer segs, int nSegs, Pointer opts);

        // Text-LID
        int crispasr_text_detect_language(String text, String modelPath, int nThreads,
                                          byte[] outLabel, int outLabelCap, float[] outConfidence);

        // Registry extras
        int crispasr_registry_lookup_by_filename_abi(String filename, byte[] outFilename, int filenameCap,
                                                     byte[] outUrl, int urlCap, byte[] outSize, int sizeCap);
        int crispasr_registry_list_backends_abi(byte[] outCsv, int outCap);

        // Session extras
        int     crispasr_session_available_backends(byte[] outCsv, int outCap);
        Pointer crispasr_session_open_explicit(String modelPath, String backendName, int nThreads);
        Pointer crispasr_session_open_with_params(String modelPath, String backendName, Pointer params);
        Pointer crispasr_session_transcribe_vad_lang(Pointer session, float[] pcm, int nSamples,
                                                     int sampleRate, String vadModelPath, Pointer opts,
                                                     String language);
        Pointer crispasr_session_translate_text(Pointer session, String text, String srcLang,
                                                String tgtLang, int maxTokens);
        void    crispasr_session_translate_text_free(Pointer text);
        int     crispasr_session_result_word_n_alts(Pointer result, int iSeg, int iWord);
        String  crispasr_session_result_word_alt_text(Pointer result, int iSeg, int iWord, int iAlt);
        float   crispasr_session_result_word_alt_p(Pointer result, int iSeg, int iWord, int iAlt);

        // Streaming (whisper context)
        Pointer crispasr_stream_open(Pointer ctx, int nThreads, int stepMs,
                                     int lengthMs, int keepMs, String language, int translate);
    }

    /**
     * JNA callback for {@code crispasr_mic_callback}. Fired on
     * miniaudio's audio thread. Keep it short and non-blocking — the
     * driver reuses {@code pcm} after this returns. Copy out before
     * any heavy work.
     *
     * <p>Important: a strong reference to the {@code MicCallback}
     * instance must be held by the caller (i.e. the {@link Mic}
     * class) for as long as the device is open. JNA's GC has no idea
     * the C side is holding it.
     */
    public interface MicCallback extends Callback {
        void invoke(Pointer pcm, int nSamples, Pointer userdata);
    }

    private Pointer handle;

    private CrispasrSession(Pointer handle) {
        this.handle = handle;
    }

    /**
     * Open a backend session for the given model file. The backend is
     * detected automatically from the GGUF metadata.
     *
     * @throws IllegalStateException if the model can't be loaded.
     */
    public static CrispasrSession open(String modelPath, int nThreads) {
        Pointer p = Lib.INSTANCE.crispasr_session_open(modelPath, nThreads);
        if (p == null) {
            throw new IllegalStateException("crispasr_session_open: failed to open " + modelPath);
        }
        return new CrispasrSession(p);
    }

    /**
     * Drop the kokoro per-session phoneme cache. No-op for non-kokoro
     * backends. Useful for long-running daemons that resynthesize across
     * many speakers and want bounded memory. (PLAN #56 #5)
     */
    public void clearPhonemeCache() {
        int rc = Lib.INSTANCE.crispasr_session_kokoro_clear_phoneme_cache(handle);
        if (rc != 0) throw new IllegalStateException("clear_phoneme_cache failed (rc=" + rc + ")");
    }

    // ----- Sticky session-state setters (PLAN #59 partial unblock) -----

    /** Sticky source-language hint (canary/cohere/voxtral/whisper). Empty clears. */
    public void setSourceLanguage(String lang) {
        int rc = Lib.INSTANCE.crispasr_session_set_source_language(handle, lang == null ? "" : lang);
        if (rc != 0) throw new IllegalStateException("set_source_language failed (rc=" + rc + ")");
    }

    /** Sticky target-language. ≠ source on canary/cohere ⇒ translation. */
    public void setTargetLanguage(String lang) {
        int rc = Lib.INSTANCE.crispasr_session_set_target_language(handle, lang == null ? "" : lang);
        if (rc != 0) throw new IllegalStateException("set_target_language failed (rc=" + rc + ")");
    }

    /** Toggle punctuation + capitalisation. Default true. */
    public void setPunctuation(boolean enable) {
        int rc = Lib.INSTANCE.crispasr_session_set_punctuation(handle, enable ? 1 : 0);
        if (rc != 0) throw new IllegalStateException("set_punctuation failed (rc=" + rc + ")");
    }

    /** Whisper sticky --translate. */
    public void setTranslate(boolean enable) {
        int rc = Lib.INSTANCE.crispasr_session_set_translate(handle, enable ? 1 : 0);
        if (rc != 0) throw new IllegalStateException("set_translate failed (rc=" + rc + ")");
    }

    /** Decoder temperature on backends with runtime control (canary/cohere/parakeet/moonshine). */
    public void setTemperature(float temperature, long seed) {
        int rc = Lib.INSTANCE.crispasr_session_set_temperature(handle, temperature, seed);
        if (rc != 0 && rc != -2) throw new IllegalStateException("set_temperature failed (rc=" + rc + ")");
    }

    /** Reseed TTS backends that support runtime seed control (soft no-op otherwise). */
    public void setTtsSeed(long seed) {
        int rc = Lib.INSTANCE.crispasr_session_set_tts_seed(handle, seed);
        if (rc != 0 && rc != -2) throw new IllegalStateException("set_tts_seed failed (rc=" + rc + ")");
    }

    /** Generated-token cap for autoregressive session backends. Pass <= 0 to clear. */
    public void setMaxNewTokens(int maxNewTokens) {
        int rc = Lib.INSTANCE.crispasr_session_set_max_new_tokens(handle, maxNewTokens);
        if (rc != 0) throw new IllegalStateException("set_max_new_tokens failed (rc=" + rc + ")");
    }

    /** Opt-in repeated generated-token penalty for autoregressive session backends. Pass <= 0 to disable. */
    public void setFrequencyPenalty(float penalty) {
        int rc = Lib.INSTANCE.crispasr_session_set_frequency_penalty(handle, penalty);
        if (rc != 0) throw new IllegalStateException("set_frequency_penalty failed (rc=" + rc + ")");
    }

    /** Diffusion / CFM step count for diffusion-based TTS backends (chatterbox). Soft no-op otherwise. */
    public void setTtsSteps(int steps) {
        int rc = Lib.INSTANCE.crispasr_session_set_tts_steps(handle, steps);
        if (rc != 0 && rc != -2) throw new IllegalStateException("set_tts_steps failed (rc=" + rc + ")");
    }

    /** Top-p nucleus-sampling threshold. Honoured by chatterbox; other backends no-op. */
    public void setTopP(float topP) {
        int rc = Lib.INSTANCE.crispasr_session_set_top_p(handle, topP);
        if (rc != 0 && rc != -2) throw new IllegalStateException("set_top_p failed (rc=" + rc + ")");
    }

    /** Min-p sampling threshold. Honoured by chatterbox; other backends no-op. */
    public void setMinP(float minP) {
        int rc = Lib.INSTANCE.crispasr_session_set_min_p(handle, minP);
        if (rc != 0 && rc != -2) throw new IllegalStateException("set_min_p failed (rc=" + rc + ")");
    }

    /** Repetition penalty (1.0 = no penalty). Honoured by chatterbox. */
    public void setRepetitionPenalty(float r) {
        int rc = Lib.INSTANCE.crispasr_session_set_repetition_penalty(handle, r);
        if (rc != 0 && rc != -2) throw new IllegalStateException("set_repetition_penalty failed (rc=" + rc + ")");
    }

    /** Classifier-free-guidance weight (chatterbox). 0 disables CFG; 0.5 is the upstream default. */
    public void setCfgWeight(float cfgWeight) {
        int rc = Lib.INSTANCE.crispasr_session_set_cfg_weight(handle, cfgWeight);
        if (rc != 0 && rc != -2) throw new IllegalStateException("set_cfg_weight failed (rc=" + rc + ")");
    }

    /** Emotion-exaggeration scalar (chatterbox). 0.5 is the upstream default. */
    public void setExaggeration(float exaggeration) {
        int rc = Lib.INSTANCE.crispasr_session_set_exaggeration(handle, exaggeration);
        if (rc != 0 && rc != -2) throw new IllegalStateException("set_exaggeration failed (rc=" + rc + ")");
    }

    /** Upper bound on speech tokens per synthesize call (chatterbox). Default 1000 ≈ 20 s. */
    public void setMaxSpeechTokens(int n) {
        int rc = Lib.INSTANCE.crispasr_session_set_max_speech_tokens(handle, n);
        if (rc != 0 && rc != -2) throw new IllegalStateException("set_max_speech_tokens failed (rc=" + rc + ")");
    }

    /** Per-phoneme length-scale / speaking-rate scalar. Honoured by kokoro today; other backends no-op. */
    public void setLengthScale(float scale) {
        int rc = Lib.INSTANCE.crispasr_session_set_length_scale(handle, scale);
        if (rc != 0 && rc != -2) throw new IllegalStateException("set_length_scale failed (rc=" + rc + ")");
    }

    /** Best-of-N sampling count for ASR backends. */
    public void setBestOf(int n) {
        int rc = Lib.INSTANCE.crispasr_session_set_best_of(handle, n);
        if (rc != 0) throw new IllegalStateException("set_best_of failed (rc=" + rc + ")");
    }

    /** Beam-search width for ASR backends that support it. */
    public void setBeamSize(int n) {
        int rc = Lib.INSTANCE.crispasr_session_set_beam_size(handle, n);
        if (rc != 0) throw new IllegalStateException("set_beam_size failed (rc=" + rc + ")");
    }

    /** Set a GBNF grammar for constrained whisper decoding. Pass null or "" to clear. */
    public void setGrammarText(String gbnfText, String rootRule, float penalty) {
        int rc = Lib.INSTANCE.crispasr_session_set_grammar_text(handle, gbnfText, rootRule, penalty);
        if (rc == -2) throw new IllegalArgumentException("set_grammar_text: invalid GBNF or root rule not found");
        if (rc != 0) throw new IllegalStateException("set_grammar_text failed (rc=" + rc + ")");
    }

    /** Set whisper decoder fallback thresholds. temperatureInc=0 disables fallback. */
    public void setFallbackThresholds(float entropyThold, float logprobThold, float noSpeechThold, float temperatureInc) {
        int rc = Lib.INSTANCE.crispasr_session_set_fallback_thresholds(handle,
                entropyThold, logprobThold, noSpeechThold, temperatureInc);
        if (rc != 0) throw new IllegalStateException("set_fallback_thresholds failed (rc=" + rc + ")");
    }

    /** Per-token top-N alternative-candidate capture for whisper greedy decode. 0 = off. */
    public void setAltN(int n) {
        int rc = Lib.INSTANCE.crispasr_session_set_alt_n(handle, n);
        if (rc != 0) throw new IllegalStateException("set_alt_n failed (rc=" + rc + ")");
    }

    /** Whisper-only text-suppression and prompt-carry extras. suppressRegex may be null or "". */
    public void setWhisperDecodeExtras(boolean suppressNst, String suppressRegex, boolean carryInitialPrompt) {
        int rc = Lib.INSTANCE.crispasr_session_set_whisper_decode_extras(handle,
                suppressNst ? 1 : 0, suppressRegex != null ? suppressRegex : "", carryInitialPrompt ? 1 : 0);
        if (rc != 0) throw new IllegalStateException("set_whisper_decode_extras failed (rc=" + rc + ")");
    }

    /** Free-form prompt passed to the backend on the next transcribe/synthesize call. */
    public void setAsk(String prompt) {
        int rc = Lib.INSTANCE.crispasr_session_set_ask(handle, prompt);
        if (rc != 0) throw new IllegalStateException("set_ask failed (rc=" + rc + ")");
    }

    /** Auto-detect spoken language on raw 16 kHz mono PCM. method:
     *  0=Whisper, 1=Silero, 2=Firered, 3=Ecapa. Returns the ISO code; the
     *  confidence is written to {@code outConfidence[0]} (length 1 array).
     */
    public String detectLanguage(float[] pcm, String lidModelPath, int method, float[] outConfidence) {
        byte[] outLang = new byte[16];
        float[] prob = new float[]{ 0.0f };
        int rc = Lib.INSTANCE.crispasr_session_detect_language(
                handle, pcm, pcm.length, lidModelPath, method, outLang, outLang.length, prob);
        if (rc != 0) throw new IllegalStateException("detect_language failed (rc=" + rc + ")");
        if (outConfidence != null && outConfidence.length >= 1) outConfidence[0] = prob[0];
        int n = 0;
        while (n < outLang.length && outLang[n] != 0) n++;
        return new String(outLang, 0, n, java.nio.charset.StandardCharsets.UTF_8);
    }

    // PLAN #59: translateText, enhanceAudioRnnoise, textDetectLanguage,
    // detectBackendFromGguf — deferred to a separate PR after JNA ABI
    // testing. The JNA interface eagerly resolves all declared symbols,
    // and type mismatches cause "Invalid memory access" at load time.

    /**
     * Load a separate codec GGUF. Required for qwen3-tts (12 Hz tokenizer)
     * and orpheus (SNAC codec); no-op for other backends.
     */
    public void setCodecPath(String path) {
        int rc = Lib.INSTANCE.crispasr_session_set_codec_path(handle, path);
        if (rc != 0) throw new IllegalStateException("set_codec_path failed (rc=" + rc + ")");
    }

    /**
     * Load a voice prompt: a baked GGUF voice pack OR a *.wav reference.
     * {@code refText} is required for qwen3-tts when {@code path} is a WAV;
     * pass {@code null} otherwise.
     *
     * <p>For orpheus voice selection is BY NAME — use
     * {@link #setSpeakerName(String)} instead.
     */
    public void setVoice(String path, String refText) {
        int rc = Lib.INSTANCE.crispasr_session_set_voice(handle, path, refText);
        if (rc != 0) throw new IllegalStateException("set_voice failed (rc=" + rc + ")");
    }

    /**
     * Select a fixed/preset speaker by name (orpheus). Names are e.g.
     * {@code "tara"}, {@code "leo"}, {@code "leah"} for canopylabs;
     * {@code "Anton"}, {@code "Sophie"} for the Kartoffel_Orpheus DE
     * finetunes. Use {@link #speakers()} to enumerate.
     *
     * @throws IllegalArgumentException if {@code name} is not in the GGUF metadata
     * @throws IllegalStateException if the active backend has no preset-speaker contract
     */
    public void setSpeakerName(String name) {
        int rc = Lib.INSTANCE.crispasr_session_set_speaker_name(handle, name);
        if (rc == -2) throw new IllegalArgumentException("unknown speaker: " + name + "; call speakers() to enumerate");
        if (rc == -3) throw new IllegalStateException("backend has no preset speakers; use setVoice() instead");
        if (rc != 0) throw new IllegalStateException("set_speaker_name failed (rc=" + rc + ")");
    }

    /**
     * Return the list of preset speaker names for the active backend.
     * Empty if the backend has no preset-speaker contract.
     */
    public String[] speakers() {
        int n = Lib.INSTANCE.crispasr_session_n_speakers(handle);
        String[] out = new String[n];
        for (int i = 0; i < n; i++) {
            String s = Lib.INSTANCE.crispasr_session_get_speaker_name(handle, i);
            out[i] = (s == null) ? "" : s;
        }
        return out;
    }

    /**
     * Set the natural-language voice description for instruct-tuned TTS
     * backends (qwen3-tts VoiceDesign today). Required before
     * {@link #synthesize(String)} when the loaded backend is VoiceDesign.
     * Detect via {@link #isVoiceDesign()}.
     *
     * @throws IllegalStateException if the active backend isn't a VoiceDesign variant
     */
    public void setInstruct(String instruct) {
        int rc = Lib.INSTANCE.crispasr_session_set_instruct(handle, instruct);
        if (rc == -3) throw new IllegalStateException(
                "backend is not a VoiceDesign variant; setInstruct only applies to qwen3-tts VoiceDesign models");
        if (rc != 0) throw new IllegalStateException("set_instruct failed (rc=" + rc + ")");
    }

    /**
     * Whether the loaded model is a qwen3-tts CustomVoice variant
     * (use {@link #setSpeakerName(String)} for it).
     */
    public boolean isCustomVoice() {
        return Lib.INSTANCE.crispasr_session_is_custom_voice(handle) != 0;
    }

    /**
     * Whether the loaded model is a qwen3-tts VoiceDesign variant
     * (use {@link #setInstruct(String)} for it).
     */
    public boolean isVoiceDesign() {
        return Lib.INSTANCE.crispasr_session_is_voice_design(handle) != 0;
    }

    /**
     * Synthesise {@code text} to 24 kHz mono PCM. Requires a TTS-capable
     * backend (kokoro / vibevoice / qwen3-tts / orpheus).
     */
    public float[] synthesize(String text) {
        IntByReference n = new IntByReference(0);
        Pointer pcm = Lib.INSTANCE.crispasr_session_synthesize(handle, text, n);
        if (pcm == null || n.getValue() <= 0) {
            throw new IllegalStateException("synthesize returned no audio");
        }
        try {
            return pcm.getFloatArray(0, n.getValue());
        } finally {
            Lib.INSTANCE.crispasr_pcm_free(pcm);
        }
    }

    /**
     * Open a rolling-window streaming decoder for this session. Mirrors
     * Go's {@code Session.StreamOpen} and the Python {@code Session.stream_open}.
     * Currently whisper-only at the C-ABI level (PLAN #62b).
     *
     * @param stepMs   commit interval — how often to emit a partial transcript (default 3000)
     * @param lengthMs rolling window in ms (default 10000)
     * @param keepMs   trailing audio carried between commits (default 200)
     * @param language ISO code, "" for auto-detect
     * @param translate whisper {@code --translate} flag
     * @throws IllegalStateException if the active backend doesn't support streaming
     */
    public Stream streamOpen(int stepMs, int lengthMs, int keepMs, String language, boolean translate) {
        Pointer p = Lib.INSTANCE.crispasr_session_stream_open(
                handle, 4, stepMs, lengthMs, keepMs, language == null ? "" : language, translate ? 1 : 0);
        if (p == null) {
            throw new IllegalStateException(
                    "crispasr_session_stream_open failed (whisper-only today)");
        }
        return new Stream(p);
    }

    /**
     * Per-commit update from a streaming session — concatenated text +
     * absolute audio-time bounds. {@code counter} increments per commit;
     * same value = no new text.
     */
    public static final class StreamingUpdate {
        public final String text;
        public final double t0;
        public final double t1;
        public final long counter;

        StreamingUpdate(String text, double t0, double t1, long counter) {
            this.text = text;
            this.t0 = t0;
            this.t1 = t1;
            this.counter = counter;
        }
    }

    /**
     * Streaming-decoder handle. Feed PCM, pull text. Whisper-only at
     * the C-ABI level today; future backends (moonshine-streaming,
     * kyutai-stt) plug in here when 62c lands.
     */
    public static final class Stream implements AutoCloseable {
        private Pointer handle;

        Stream(Pointer handle) {
            this.handle = handle;
        }

        /**
         * Push 16 kHz mono float32 PCM. Returns {@code 0} if still
         * buffering, {@code 1} if a new partial transcript is ready
         * (call {@link #getText()} to fetch it).
         *
         * @throws IllegalStateException on decode failure (negative rc from C-ABI)
         */
        public int feed(float[] pcm) {
            if (handle == null) throw new IllegalStateException("stream is closed");
            if (pcm == null || pcm.length == 0) return 0;
            int rc = Lib.INSTANCE.crispasr_stream_feed(handle, pcm, pcm.length);
            if (rc < 0) throw new IllegalStateException("crispasr_stream_feed failed (rc=" + rc + ")");
            return rc;
        }

        /** Latest committed transcript + absolute audio-time bounds. */
        public StreamingUpdate getText() {
            if (handle == null) throw new IllegalStateException("stream is closed");
            byte[] buf = new byte[8192];
            DoubleByReference t0 = new DoubleByReference(0.0);
            DoubleByReference t1 = new DoubleByReference(0.0);
            LongByReference counter = new LongByReference(0L);
            int rc = Lib.INSTANCE.crispasr_stream_get_text(handle, buf, buf.length, t0, t1, counter);
            if (rc < 0) throw new IllegalStateException("crispasr_stream_get_text failed (rc=" + rc + ")");
            int n = 0;
            while (n < buf.length && buf[n] != 0) n++;
            String text = new String(buf, 0, n, java.nio.charset.StandardCharsets.UTF_8);
            return new StreamingUpdate(text, t0.getValue(), t1.getValue(), counter.getValue());
        }

        /** Force a decode on whatever is buffered. Useful when the audio has ended. */
        public void flush() {
            if (handle == null) throw new IllegalStateException("stream is closed");
            int rc = Lib.INSTANCE.crispasr_stream_flush(handle);
            if (rc < 0) throw new IllegalStateException("crispasr_stream_flush failed (rc=" + rc + ")");
        }

        @Override
        public void close() {
            if (handle != null) {
                Lib.INSTANCE.crispasr_stream_close(handle);
                handle = null;
            }
        }
    }

    @Override
    public void close() {
        if (handle != null) {
            Lib.INSTANCE.crispasr_session_close(handle);
            handle = null;
        }
    }

    // -----------------------------------------------------------------
    // -----------------------------------------------------------------
    // ASR Transcription (PLAN #59)
    // -----------------------------------------------------------------

    /** One word with timing and confidence from a transcription result. */
    public static final class Word {
        public final String text;
        public final long t0, t1; // centiseconds
        public final float p;     // confidence
        Word(String text, long t0, long t1, float p) {
            this.text = text; this.t0 = t0; this.t1 = t1; this.p = p;
        }
    }

    /** One segment from a transcription result. */
    public static final class Segment {
        public final String text;
        public final long t0, t1; // centiseconds
        public final Word[] words;
        Segment(String text, long t0, long t1, Word[] words) {
            this.text = text; this.t0 = t0; this.t1 = t1; this.words = words;
        }
    }

    /** Transcribe 16 kHz mono float32 PCM. */
    public Segment[] transcribe(float[] pcm) {
        return transcribeLang(pcm, null);
    }

    /** Transcribe with explicit language hint. */
    public Segment[] transcribeLang(float[] pcm, String lang) {
        Pointer r = Lib.INSTANCE.crispasr_session_transcribe_lang(handle, pcm, pcm.length, lang);
        if (r == null) throw new RuntimeException("transcription failed");
        try { return extractSegments(r); } finally { Lib.INSTANCE.crispasr_session_result_free(r); }
    }

    /** Transcribe with VAD segmentation. */
    public Segment[] transcribeVad(float[] pcm, int sampleRate, String vadModelPath) {
        Pointer r = Lib.INSTANCE.crispasr_session_transcribe_vad(
                handle, pcm, pcm.length, sampleRate, vadModelPath, null);
        if (r == null) throw new RuntimeException("transcription with VAD failed");
        try { return extractSegments(r); } finally { Lib.INSTANCE.crispasr_session_result_free(r); }
    }

    private static Segment[] extractSegments(Pointer r) {
        int nSegs = Lib.INSTANCE.crispasr_session_result_n_segments(r);
        Segment[] segs = new Segment[nSegs];
        for (int i = 0; i < nSegs; i++) {
            String text = Lib.INSTANCE.crispasr_session_result_segment_text(r, i);
            long t0 = Lib.INSTANCE.crispasr_session_result_segment_t0(r, i);
            long t1 = Lib.INSTANCE.crispasr_session_result_segment_t1(r, i);
            int nWords = Lib.INSTANCE.crispasr_session_result_n_words(r, i);
            Word[] words = new Word[nWords];
            for (int j = 0; j < nWords; j++) {
                words[j] = new Word(
                    Lib.INSTANCE.crispasr_session_result_word_text(r, i, j),
                    Lib.INSTANCE.crispasr_session_result_word_t0(r, i, j),
                    Lib.INSTANCE.crispasr_session_result_word_t1(r, i, j),
                    Lib.INSTANCE.crispasr_session_result_word_p(r, i, j));
            }
            segs[i] = new Segment(text, t0, t1, words);
        }
        return segs;
    }

    // -----------------------------------------------------------------
    // Forced alignment (PLAN #59)
    // -----------------------------------------------------------------

    /** One aligned word with timing. */
    public static final class AlignedWord {
        public final String text;
        public final long t0, t1; // centiseconds
        AlignedWord(String text, long t0, long t1) {
            this.text = text; this.t0 = t0; this.t1 = t1;
        }
    }

    /** Run CTC forced alignment on transcript + audio. */
    public static AlignedWord[] alignWords(String alignerModel, String transcript,
                                            float[] pcm, long tOffsetCs, int nThreads) {
        Pointer r = Lib.INSTANCE.crispasr_align_words_abi(alignerModel, transcript,
                pcm, pcm.length, tOffsetCs, nThreads);
        if (r == null) throw new RuntimeException("alignment failed");
        try {
            int n = Lib.INSTANCE.crispasr_align_result_n_words(r);
            AlignedWord[] words = new AlignedWord[n];
            for (int i = 0; i < n; i++) {
                words[i] = new AlignedWord(
                    Lib.INSTANCE.crispasr_align_result_word_text(r, i),
                    Lib.INSTANCE.crispasr_align_result_word_t0(r, i),
                    Lib.INSTANCE.crispasr_align_result_word_t1(r, i));
            }
            return words;
        } finally {
            Lib.INSTANCE.crispasr_align_result_free(r);
        }
    }

    // -----------------------------------------------------------------
    // Standalone VAD (PLAN #59)
    // -----------------------------------------------------------------

    /** One speech segment from VAD. */
    public static final class VADSpan {
        public final double t0, t1; // seconds
        VADSpan(double t0, double t1) { this.t0 = t0; this.t1 = t1; }
    }

    /** Run standalone VAD. vadModelPath can be empty for auto-download. */
    public static VADSpan[] vadSegments(String vadModelPath, float[] pcm, int sampleRate,
                                         float threshold, int minSpeechMs, int minSilenceMs, int nThreads) {
        Pointer[] outSpans = new Pointer[1];
        int n = Lib.INSTANCE.crispasr_vad_segments(vadModelPath, pcm, pcm.length,
                sampleRate, threshold, minSpeechMs, minSilenceMs, nThreads, 0, outSpans);
        if (n < 0) throw new RuntimeException("VAD failed (rc=" + n + ")");
        if (n == 0 || outSpans[0] == null) return new VADSpan[0];
        try {
            float[] raw = outSpans[0].getFloatArray(0, n * 2);
            VADSpan[] spans = new VADSpan[n];
            for (int i = 0; i < n; i++)
                spans[i] = new VADSpan(raw[i * 2], raw[i * 2 + 1]);
            return spans;
        } finally {
            Lib.INSTANCE.crispasr_vad_free(outSpans[0]);
        }
    }

    // -----------------------------------------------------------------
    // Standalone language detection (PLAN #59)
    // -----------------------------------------------------------------

    /** Detect spoken language from raw 16 kHz mono PCM.
     *  method: 0=Whisper, 1=Silero, 2=Firered, 3=Ecapa. */
    public static String[] detectLanguagePcm(float[] pcm, int method, String modelPath, int nThreads) {
        byte[] outLang = new byte[16];
        float[] outProb = new float[1];
        int rc = Lib.INSTANCE.crispasr_detect_language_pcm(pcm, pcm.length, method,
                modelPath, nThreads, 0, 0, 0, outLang, outLang.length, outProb);
        if (rc != 0) throw new RuntimeException("language detection failed");
        String lang = new String(outLang).trim().replaceAll("\0", "");
        return new String[]{lang, String.valueOf(outProb[0])};
    }

    // -----------------------------------------------------------------
    // Kokoro per-language routing (PLAN #56 opt 2b)
    // -----------------------------------------------------------------

    /** Result of {@link #kokoroResolveForLang(String, String)}. */
    public static final class Resolved {
        /** Path to actually load (may differ from input). */
        public final String modelPath;
        /** Fallback voice path; {@code null} if not applicable. */
        public final String voicePath;
        /** Basename of the picked voice (e.g. "df_victoria"); {@code null} otherwise. */
        public final String voiceName;
        /** True iff the model path was rewritten to the German backbone. */
        public final boolean backboneSwapped;

        Resolved(String modelPath, String voicePath, String voiceName, boolean backboneSwapped) {
            this.modelPath = modelPath;
            this.voicePath = voicePath;
            this.voiceName = voiceName;
            this.backboneSwapped = backboneSwapped;
        }
    }

    /**
     * Resolve the kokoro model + fallback voice for {@code lang}. Mirrors
     * what the CrispASR CLI does for {@code --backend kokoro -l <lang>}
     * (PLAN #56 opt 2b). Wrappers should call this <em>before</em>
     * {@link #open(String, int)} so the routing kicks in even outside
     * the CLI entry point.
     */
    public static Resolved kokoroResolveForLang(String modelPath, String lang) {
        byte[] outModel = new byte[1024];
        byte[] outVoice = new byte[1024];
        byte[] outPicked = new byte[64];

        int rc = Lib.INSTANCE.crispasr_kokoro_resolve_model_for_lang_abi(
                modelPath, lang == null ? "" : lang, outModel, outModel.length);
        if (rc < 0) throw new IllegalStateException("kokoro_resolve_model_for_lang: buffer too small");
        boolean swapped = (rc == 0);
        String resolvedModel = nullTerminated(outModel);
        if (resolvedModel.isEmpty()) resolvedModel = modelPath;

        rc = Lib.INSTANCE.crispasr_kokoro_resolve_fallback_voice_abi(
                modelPath, lang == null ? "" : lang,
                outVoice, outVoice.length, outPicked, outPicked.length);
        if (rc < 0) throw new IllegalStateException("kokoro_resolve_fallback_voice: buffer too small");
        if (rc == 0) {
            return new Resolved(resolvedModel, nullTerminated(outVoice), nullTerminated(outPicked), swapped);
        }
        return new Resolved(resolvedModel, null, null, swapped);
    }

    private static String nullTerminated(byte[] buf) {
        int n = 0;
        while (n < buf.length && buf[n] != 0) n++;
        return new String(buf, 0, n, java.nio.charset.StandardCharsets.UTF_8);
    }
}
