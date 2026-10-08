package io.github.ggerganov.whispercpp;

import com.sun.jna.Callback;
import com.sun.jna.DefaultTypeMapper;
import com.sun.jna.Memory;
import com.sun.jna.ToNativeContext;
import com.sun.jna.ToNativeConverter;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.DoubleByReference;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.LongByReference;
import com.sun.jna.ptr.PointerByReference;

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

    private static java.util.Map<String, Object> nativeOptions() {
        DefaultTypeMapper mapper = new DefaultTypeMapper();
        // JNA 5.13 applies OPTION_STRING_ENCODING to returned strings, but
        // scalar String arguments still use the global jna.encoding default.
        // Convert inputs explicitly without changing the host's global setting.
        mapper.addToNativeConverter(String.class, new ToNativeConverter() {
            @Override public Object toNative(Object value, ToNativeContext context) {
                if (value == null) return null;
                byte[] utf8 = ((String) value).getBytes(java.nio.charset.StandardCharsets.UTF_8);
                Memory memory = new Memory((long) utf8.length + 1);
                memory.write(0, utf8, 0, utf8.length);
                memory.setByte(utf8.length, (byte) 0);
                return memory;
            }
            @Override public Class<?> nativeType() { return Pointer.class; }
        });
        java.util.Map<String, Object> options = new java.util.HashMap<>();
        options.put(Library.OPTION_STRING_ENCODING, "UTF-8");
        options.put(Library.OPTION_TYPE_MAPPER, mapper);
        return options;
    }

    public interface Lib extends Library {
        // The session C ABI uses UTF-8 regardless of the host/JNA default.
        Lib INSTANCE = Native.load("crispasr", Lib.class, nativeOptions());

        Pointer crispasr_session_open(String modelPath, int nThreads);
        void    crispasr_session_close(Pointer session);
        int     crispasr_session_set_codec_path(Pointer session, String path);
        int     crispasr_session_set_voice(Pointer session, String path, String refTextOrNull);
        int     crispasr_session_set_speaker_name(Pointer session, String name);
        int     crispasr_session_set_speaker_id(Pointer session, int id);
        int     crispasr_session_set_punc_model(Pointer session, String puncModel);
        int     crispasr_session_set_hotwords(Pointer session, String hotwords, float boost);
        int     crispasr_session_set_sensitivity(Pointer session, String preset);
        int     crispasr_session_set_g2p_dict(Pointer session, String source);
        int     crispasr_session_n_speakers(Pointer session);
        String  crispasr_session_get_speaker_name(Pointer session, int i);
        int     crispasr_session_set_instruct(Pointer session, String instruct);
        // #316: synthesize these phonemes verbatim, skipping the G2P. Empty clears. kokoro and piper only (rc=-2 otherwise).
        int          crispasr_session_set_tts_phonemes(Pointer session, String phonemes);
        void         crispasr_session_set_tts_pad_silence_ms(Pointer session, int ms);
        int     crispasr_session_is_custom_voice(Pointer session);
        int     crispasr_session_is_voice_design(Pointer session);
        Pointer crispasr_session_synthesize(Pointer session, String text, IntByReference outNSamples);
        // UNMARKED synthesis — refused unless crispasr_session_accept_marking_responsibility() was called first.
        Pointer crispasr_session_synthesize_raw(Pointer session, String text, IntByReference outNSamples);
        int     crispasr_session_accept_marking_responsibility(Pointer session, String attestation);
        // Whose voice a PRESET voice is (EU AI Act Art. 50(4)); -2 on a bad value.
        int     crispasr_session_set_speaker_identity(Pointer session, String identity);
        // Speech-to-speech (lfm2-audio, mini-omni2, sidon, voxcpm2-vae). outText receives the intermediate transcript.
        Pointer crispasr_session_speech_to_speech(Pointer session, float[] inSamples, int nInSamples,
                                                  PointerByReference outText, IntByReference outNSamples);
        int     crispasr_session_input_sample_rate(Pointer session);
        // #332: output-side counterparts (rate of synthesize/s2s PCM; mono channels).
        int     crispasr_session_output_sample_rate(Pointer session);
        int     crispasr_session_input_channels(Pointer session);
        int     crispasr_session_output_channels(Pointer session);
        void    crispasr_pcm_free(Pointer pcm);
        // AI-content marking (EU AI Act Art. 50(2)) — the other half of synthesize_raw.
        void    crispasr_watermark_embed(float[] pcm, int nSamples, float alpha);
        float   crispasr_watermark_detect(float[] pcm, int nSamples);
        int     crispasr_session_kokoro_clear_phoneme_cache(Pointer session);
        int     crispasr_session_set_source_language(Pointer session, String lang);
        int     crispasr_session_set_target_language(Pointer session, String lang);
        int     crispasr_session_set_tts_reference_language(Pointer session, String lang);
        int     crispasr_session_set_punctuation(Pointer session, int enable);
        int     crispasr_session_set_translate(Pointer session, int enable);
        int     crispasr_session_set_temperature(Pointer session, float temperature, long seed);
        int     crispasr_session_set_tts_seed(Pointer session, long seed);
        int     crispasr_session_set_max_new_tokens(Pointer session, int maxNewTokens);
        int     crispasr_session_set_frequency_penalty(Pointer session, float penalty);
        int     crispasr_session_set_tts_steps(Pointer session, int steps);
        int     crispasr_session_set_tts_num_candidates(Pointer session, int n);
        int     crispasr_session_set_top_p(Pointer session, float topP);
        int     crispasr_session_set_top_k(Pointer session, int topK);
        int     crispasr_session_set_do_sample(Pointer session, int enable);
        int     crispasr_session_set_min_p(Pointer session, float minP);
        int     crispasr_session_set_repetition_penalty(Pointer session, float r);
        int     crispasr_session_set_cfg_weight(Pointer session, float cfgWeight);
        int     crispasr_session_set_tts_noise_temp(Pointer session, float noiseTemp);
        int     crispasr_session_set_exaggeration(Pointer session, float exaggeration);
        int     crispasr_session_set_max_speech_tokens(Pointer session, int n);
        int     crispasr_session_set_min_speech_tokens(Pointer session, int n);
        int     crispasr_session_set_length_scale(Pointer session, float scale);
        int     crispasr_session_set_best_of(Pointer session, int n);
        int     crispasr_session_set_beam_size(Pointer session, int n);
        int     crispasr_session_set_return_logits(Pointer session, int enable);
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
        Pointer crispasr_session_transcribe_chunked_lang(Pointer session, float[] pcm, int nSamples,
                                                         int chunkSeconds, int overlapSeconds, String language);
        int     crispasr_get_progress();
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
        float        crispasr_session_result_segment_no_speech_prob(Pointer result, int iSeg);
        // #300: native per-segment speaker label ("(Speaker N) "), or "" when the
        // backend does not diarize natively. Never NULL.
        String       crispasr_session_result_segment_speaker(Pointer result, int i);
        // Per-frame CTC logits (opted in via crispasr_session_set_return_logits)
        // for backends with a dense CTC grid (Omni CTC, wav2vec2/hubert/data2vec,
        // canary-ctc). _logits returns a const float* (frame-major;
        // logits[t * nVocab + v]) or NULL when none. Raw pre-softmax for Omni &
        // wav2vec2; log-probabilities for canary-ctc.
        int          crispasr_session_result_n_logit_frames(Pointer result);
        int          crispasr_session_result_n_logit_vocab(Pointer result);
        Pointer      crispasr_session_result_logits(Pointer result);
        void         crispasr_session_result_free(Pointer result);

        // --- Alignment (PLAN #59) ---
        Pointer crispasr_align_words_abi(String alignerModel, String transcript,
                                         float[] samples, int nSamples, long tOffsetCs, int nThreads);
        int    crispasr_align_result_n_words(Pointer result);
        String crispasr_align_result_word_text(Pointer result, int i);
        long   crispasr_align_result_word_t0(Pointer result, int i);
        long   crispasr_align_result_word_t1(Pointer result, int i);
        int crispasr_align_result_n_characters(Pointer result, int word);
        String crispasr_align_result_character_text(Pointer result, int word, int i);
        long crispasr_align_result_character_t0(Pointer result, int word, int i);
        long crispasr_align_result_character_t1(Pointer result, int word, int i);
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
        int crispasr_registry_default_bundle_info_abi(String backend, byte[] outBackend, int backendCap,
                                                       byte[] outLicense, int licenseCap,
                                                       int[] outRequiresAcceptance);
        int crispasr_registry_default_bundle_artifact_abi(String backend, int index, int[] outKind,
                                                           byte[] outFilename, int filenameCap,
                                                           byte[] outUrl, int urlCap,
                                                           byte[] outSize, int sizeCap);
        int crispasr_cache_ensure_file_abi(String filename, String url, int quiet,
                                           String cacheDirOverride, byte[] outBuf, int outCap);
        int crispasr_cache_dir_abi(String cacheDirOverride, byte[] outBuf, int outCap);

        // --- Streaming (PLAN #62b) — rolling-window decoder, whisper-only today.
        int crispasr_session_stream_kind(Pointer session);
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

        // Speaker database (closed-roster, consent-gated — issue #266).
        // expectedNamesCsv names the enrolled participants asserted present
        // (e.g. "Alice,Bob"); consentAttested affirms GDPR Art. 9 consent.
        // Open 1:N identification is unsupported.
        Pointer crispasr_speaker_db_open(String dirPath, String expectedNamesCsv, int consentAttested);
        void    crispasr_speaker_db_free(Pointer db);
        int     crispasr_speaker_db_count(Pointer db);
        float   crispasr_speaker_db_match(Pointer db, float[] embedding, int dim,
                                          float threshold, byte[] outName, int outCap);
        int     crispasr_speaker_db_enroll2(String dirPath, String name, float[] embedding, int dim,
                                            int consentAttested);

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
        int crispasr_diarize_segments_turns_abi(float[] leftPcm, float[] rightPcm, int nSamples,
                                                int isStereo, Pointer segs, int nSegs, Pointer opts,
                                                Pointer outTurns, int turnsCap, IntByReference outNTurns);

        // Text-LID
        int crispasr_text_detect_language(String text, String modelPath, int nThreads,
                                          byte[] outLabel, int outLabelCap, float[] outConfidence);

        // Registry extras
        int crispasr_registry_lookup_by_filename_abi(String filename, byte[] outFilename, int filenameCap,
                                                     byte[] outUrl, int urlCap, byte[] outSize, int sizeCap);
        int crispasr_registry_list_backends_abi(byte[] outCsv, int outCap);

        // Session extras
        int     crispasr_session_available_backends(byte[] outCsv, int outCap);
        int     crispasr_session_detected_language(Pointer session, byte[] outBuf, int outCap);
        // CTC vocabulary access (Omni CTC backend): n_vocab piece count,
        // token_text maps an id to its raw piece (JNA copies the model-owned
        // const char*; do not free) or "" when out of range / unsupported.
        int     crispasr_session_n_vocab(Pointer session);
        String  crispasr_session_token_text(Pointer session, int id);
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

    /**
     * Language a voice-cloning REFERENCE clip is spoken in (#329). Cross-lingual TTS backends
     * (cosyvoice3) drop the reference transcript when it differs from the requested output
     * language, so the clone speaks that language rather than carrying the reference's accent.
     * Optional — otherwise inferred from the voice bank or the reference transcript, and that
     * inference declines rather than guesses on a short transcript.
     */
    public void setTtsReferenceLanguage(String lang) {
        int rc = Lib.INSTANCE.crispasr_session_set_tts_reference_language(handle, lang == null ? "" : lang);
        if (rc != 0) throw new IllegalStateException("set_tts_reference_language failed (rc=" + rc + ")");
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

    /** Flow-matching timing candidates ranked per token (TADA). Soft no-op on other backends. */
    public void setTtsNumCandidates(int n) {
        int rc = Lib.INSTANCE.crispasr_session_set_tts_num_candidates(handle, n);
        if (rc != 0 && rc != -2) throw new IllegalStateException("set_tts_num_candidates failed (rc=" + rc + ")");
    }

    /** Top-p nucleus-sampling threshold. Honoured by chatterbox; other backends no-op. */
    public void setTopP(float topP) {
        int rc = Lib.INSTANCE.crispasr_session_set_top_p(handle, topP);
        if (rc != 0 && rc != -2) throw new IllegalStateException("set_top_p failed (rc=" + rc + ")");
    }

    /** Top-k sampling cutoff (0 = disabled). Honoured by TADA; other backends no-op. */
    public void setTopK(int topK) {
        int rc = Lib.INSTANCE.crispasr_session_set_top_k(handle, topK);
        if (rc != 0 && rc != -2) throw new IllegalStateException("set_top_k failed (rc=" + rc + ")");
    }

    /** Enable/disable sampling (false = greedy). Honoured by TADA; other backends no-op. */
    public void setDoSample(boolean enable) {
        int rc = Lib.INSTANCE.crispasr_session_set_do_sample(handle, enable ? 1 : 0);
        if (rc != 0 && rc != -2) throw new IllegalStateException("set_do_sample failed (rc=" + rc + ")");
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

    /** TADA flow-matching noise temperature (Python noise_temp, default 0.9). */
    public void setTtsNoiseTemp(float noiseTemp) {
        int rc = Lib.INSTANCE.crispasr_session_set_tts_noise_temp(handle, noiseTemp);
        if (rc != 0 && rc != -2) throw new IllegalStateException("set_tts_noise_temp failed (rc=" + rc + ")");
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

    /** Floor on generated audio length (MOSS TTS). Units are codec frames at 12.5 Hz (80 ms each), so n=25 floors at ~2 s. Other backends no-op (rc=-2). */
    public void setMinSpeechTokens(int n) {
        int rc = Lib.INSTANCE.crispasr_session_set_min_speech_tokens(handle, n);
        if (rc != 0 && rc != -2) throw new IllegalStateException("set_min_speech_tokens failed (rc=" + rc + ")");
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

    /**
     * Opt in to capturing the per-frame CTC logits (backends with a dense CTC
     * grid: Omni CTC, wav2vec2/hubert/data2vec, canary-ctc) so a following
     * transcribe attaches the dense grid read back via
     * {@link #transcribeWithLogits(float[], String)}. Off by default so the
     * normal path pays no {@code [vocab × frames]} copy.
     */
    public void setReturnLogits(boolean enable) {
        int rc = Lib.INSTANCE.crispasr_session_set_return_logits(handle, enable ? 1 : 0);
        if (rc != 0) throw new IllegalStateException("set_return_logits failed (rc=" + rc + ")");
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

    /** Select a multi-speaker backend's speaker by index. */
    public void setSpeakerId(int id) {
        int rc = Lib.INSTANCE.crispasr_session_set_speaker_id(handle, id);
        if (rc != 0 && rc != -2) throw new IllegalStateException("set_speaker_id failed (rc=" + rc + ")");
    }

    /**
     * Select + load a punctuation-restoration model on the session
     * ({@code auto}/{@code firered}/{@code fullstop}/{@code punctuate-all}/{@code pcs}/path;
     * {@code "none"}/{@code ""} unloads). Auto-downloads on first use. Restores
     * punctuation on backends that emit none (parakeet RNNT/CTC, …).
     */
    public void setPuncModel(String puncModel) {
        int rc = Lib.INSTANCE.crispasr_session_set_punc_model(handle, puncModel);
        if (rc != 0) throw new IllegalStateException("set_punc_model failed (rc=" + rc + ")");
    }

    /** Comma-separated hotwords for contextual biasing, boosted by {@code boost} per token match. */
    public void setHotwords(String hotwords, float boost) {
        int rc = Lib.INSTANCE.crispasr_session_set_hotwords(handle, hotwords, boost);
        if (rc != 0) throw new IllegalStateException("set_hotwords failed (rc=" + rc + ")");
    }

    /**
     * Apply a named bundle of the four decoder fallback thresholds:
     * {@code "conservative"}, {@code "balanced"} (the shipped defaults, a no-op)
     * or {@code "aggressive"}. {@code "strict"}/{@code "default"}/{@code "loose"}
     * are aliases. Mirrors the CLI's {@code --sensitivity}.
     *
     * <p>The four thresholds interact — a decode is only retried when the logprob
     * <em>and</em> no-speech bars are both crossed — so they move as a set. A later
     * {@link #setFallbackThresholds} overrides this.
     *
     * @throws IllegalArgumentException if the preset is unrecognised; a typo is
     *     never silently treated as {@code "balanced"}.
     */
    public void setSensitivity(String preset) {
        int rc = Lib.INSTANCE.crispasr_session_set_sensitivity(handle, preset);
        if (rc == -2) {
            throw new IllegalArgumentException(
                "unknown sensitivity preset '" + preset + "' (expected: conservative, balanced, aggressive)");
        }
        if (rc != 0) throw new IllegalStateException("set_sensitivity failed (rc=" + rc + ")");
    }

    /** Select the G2P pronunciation dictionary for TTS ({@code olaph}/{@code open-dict}/path). */
    public void setG2pDict(String source) {
        int rc = Lib.INSTANCE.crispasr_session_set_g2p_dict(handle, source);
        if (rc != 0) throw new IllegalStateException("set_g2p_dict failed (rc=" + rc + ")");
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

    /** #316: synthesize these phonemes verbatim, skipping the G2P. Empty clears. kokoro and piper only (rc=-2 otherwise). */
    public void setTtsPhonemes(String phonemes) {
        int rc = Lib.INSTANCE.crispasr_session_set_tts_phonemes(handle, phonemes == null ? "" : phonemes);
        if (rc == -2) throw new RuntimeException("backend has no phonemes-in entry point (kokoro and piper do)");
        if (rc != 0) throw new RuntimeException("set_tts_phonemes failed (rc=" + rc + ")");
    }

    /** Pad N ms of silence at the beginning of TTS output. Useful to bypass VLC playback bugs where it drops the first ~1.5s of audio while parsing a large C2PA chunk. */
    public void setTtsPadSilenceMs(int ms) {
        Lib.INSTANCE.crispasr_session_set_tts_pad_silence_ms(handle, ms);
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
     * UNMARKED synthesis — identical to {@link #synthesize(String)} but skips
     * the audible/inaudible watermark. Hard-refused (returns no audio) unless
     * {@link #acceptMarkingResponsibility(String)} was called first. Under the
     * EU AI Act (Art. 50) the integrator becomes responsible for marking the
     * output as AI-generated.
     */
    public float[] synthesizeRaw(String text) {
        IntByReference n = new IntByReference(0);
        Pointer pcm = Lib.INSTANCE.crispasr_session_synthesize_raw(handle, text, n);
        if (pcm == null || n.getValue() <= 0) {
            throw new IllegalStateException("synthesizeRaw returned no audio "
                    + "(did you call acceptMarkingResponsibility()?)");
        }
        try {
            return pcm.getFloatArray(0, n.getValue());
        } finally {
            Lib.INSTANCE.crispasr_pcm_free(pcm);
        }
    }

    /**
     * Attest that the integrator takes responsibility for marking generated
     * audio as AI-generated (EU AI Act Art. 50). REQUIRED before
     * {@link #synthesizeRaw(String)} will produce audio.
     */
    public void acceptMarkingResponsibility(String attestation) {
        int rc = Lib.INSTANCE.crispasr_session_accept_marking_responsibility(
                handle, attestation == null ? "" : attestation);
        if (rc != 0) {
            throw new IllegalStateException(
                    "accept_marking_responsibility failed (rc=" + rc + ")");
        }
    }

    /**
     * Declare whose voice a PRESET voice is: {@code "real_person"},
     * {@code "synthetic"} or {@code "unknown"}.
     *
     * <p>Cloning is not the only way to produce a deep fake: a preset voice
     * shipped inside a model can be an identifiable individual — a named donor,
     * or a corpus speaker such as VCTK's {@code p225} — and EU AI Act
     * Art. 3(60) attaches to the audio resembling that person, not to which
     * pipeline produced it. Setting {@code real_person} makes the Art. 50(4)
     * reminder fire for a non-cloned voice.
     *
     * <p>It does <b>not</b> require a consent attestation: whether that donor
     * agreed to the model being trained is a licensing matter settled upstream
     * that you cannot attest to.
     *
     * @throws IllegalArgumentException on an unrecognised value, rather than
     *         silently downgrading it to {@code unknown}.
     */
    public void setSpeakerIdentity(String identity) {
        int rc = Lib.INSTANCE.crispasr_session_set_speaker_identity(
                handle, identity == null ? "" : identity);
        if (rc == -2) {
            throw new IllegalArgumentException(
                    "unrecognised speaker_identity '" + identity
                            + "' (expected real_person, synthetic or unknown)");
        }
        if (rc != 0) {
            throw new IllegalStateException("set_speaker_identity failed (rc=" + rc + ")");
        }
    }

    /**
     * Embed the AI-content watermark into mono float32 PCM, in place.
     *
     * <p>The other half of {@link #synthesizeRaw(String)}: opting out of
     * automatic marking makes marking the result the integrator's duty under
     * EU AI Act Art. 50(2), and this is what discharges it. Do the
     * post-processing you opted out for — resample, mix, concatenate — then
     * mark the finished buffer.
     *
     * <p>Uses the robust, reliably detectable default strength; AudioSeal
     * instead when a model has been loaded. Static because marking is a
     * property of the samples, not of the session that produced them.
     */
    public static void watermarkEmbed(float[] pcm) {
        if (pcm == null || pcm.length == 0) {
            return;
        }
        Lib.INSTANCE.crispasr_watermark_embed(pcm, pcm.length, -1.0f);
    }

    /**
     * Confidence in [0, 1] that {@code pcm} carries the watermark.
     *
     * <p>A weak diagnostic, not proof: the spread-spectrum detector's null mean
     * is 0.5, not 0, and a negative result on a short clip is mostly evidence
     * that the clip was short. See {@code docs/eu-ai-act.md} §6.7.
     */
    public static float watermarkDetect(float[] pcm) {
        if (pcm == null || pcm.length == 0) {
            return 0.0f;
        }
        return Lib.INSTANCE.crispasr_watermark_detect(pcm, pcm.length);
    }

    /** Result of {@link #speechToSpeech(float[])}: output PCM plus the intermediate transcript. */
    public static final class SpeechToSpeechResult {
        /** 24 kHz mono float32 PCM produced by the backend. */
        public final float[] pcm;
        /** Intermediate transcript of the input speech (may be {@code null}). */
        public final String text;

        SpeechToSpeechResult(float[] pcm, String text) {
            this.pcm = pcm;
            this.text = text;
        }
    }

    /**
     * Speech-to-speech: transform input PCM to output PCM, returning the
     * intermediate transcript alongside. Requires an S2S-capable backend
     * (lfm2-audio / mini-omni2 / sidon / voxcpm2-vae).
     *
     * @param inSamples input float32 PCM at {@link #inputSampleRate()}
     */
    public SpeechToSpeechResult speechToSpeech(float[] inSamples) {
        IntByReference n = new IntByReference(0);
        PointerByReference outText = new PointerByReference();
        Pointer pcm = Lib.INSTANCE.crispasr_session_speech_to_speech(
                handle, inSamples, inSamples.length, outText, n);
        if (pcm == null || n.getValue() <= 0) {
            throw new IllegalStateException(
                    "speechToSpeech returned no audio (backend may not support S2S)");
        }
        String text = null;
        Pointer tp = outText.getValue();
        if (tp != null) {
            text = tp.getString(0);
            Lib.INSTANCE.crispasr_session_translate_text_free(tp);
        }
        try {
            return new SpeechToSpeechResult(pcm.getFloatArray(0, n.getValue()), text);
        } finally {
            Lib.INSTANCE.crispasr_pcm_free(pcm);
        }
    }

    /**
     * Sample rate the backend expects for input PCM (e.g. 16000 for
     * Whisper-family backends). Returns 0 on error.
     */
    public int inputSampleRate() {
        return Lib.INSTANCE.crispasr_session_input_sample_rate(handle);
    }

    /**
     * Sample rate of the PCM synthesize/speechToSpeech produce for this
     * backend. Returns 0 when the backend has no audio output (ASR-only). (#332)
     */
    public int outputSampleRate() {
        return Lib.INSTANCE.crispasr_session_output_sample_rate(handle);
    }

    /**
     * Channel count for audio input: 1 (mono) for every current backend,
     * 0 on error. Source separation is the stereo exception and has its own
     * surface. (#332)
     */
    public int inputChannels() {
        return Lib.INSTANCE.crispasr_session_input_channels(handle);
    }

    /**
     * Channel count for synthesized / s2s output audio: 1 (mono), or 0 when
     * the backend has no audio output. (#332)
     */
    public int outputChannels() {
        return Lib.INSTANCE.crispasr_session_output_channels(handle);
    }

    /** Streaming mode: 0 unavailable, 1 rolling windows, 2 persistent
     * model caches, 3 text-prefix streaming. */
    public int streamKind() {
        return Lib.INSTANCE.crispasr_session_stream_kind(handle);
    }

    /**
     * Open a rolling-window streaming decoder for this session. Mirrors
     * Go's {@code Session.StreamOpen} and the Python {@code Session.stream_open}.
     * Call streamKind() to query the loaded model's streaming implementation.
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
                    "crispasr_session_stream_open failed");
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
        /** Whisper's per-segment no-speech probability (the {@code <|nospeech|>}
         *  posterior) in [0, 1]. Whisper-only; other backends leave -1.0 ("no data"). */
        public final float noSpeechProb;
        /** Native per-segment speaker label from a backend that diarizes on its own,
         *  in the {@code "(Speaker N) "} form the CLI prefixes into text/srt/vtt output,
         *  or {@code ""} when the backend produced none. Populated today by vibevoice.
         *  The ordinals are CHUNK-LOCAL: {@code Speaker 1} from one transcribe call is
         *  not necessarily the same voice as {@code Speaker 1} from the next. */
        public final String speaker;
        Segment(String text, long t0, long t1, Word[] words, float noSpeechProb, String speaker) {
            this.text = text; this.t0 = t0; this.t1 = t1; this.words = words; this.noSpeechProb = noSpeechProb;
            this.speaker = speaker == null ? "" : speaker;
        }
    }

    /**
     * Per-frame CTC logits from a CTC backend (Omni CTC, wav2vec2/hubert/
     * data2vec, or canary-ctc), captured by
     * {@link #transcribeWithLogits(float[], String)}. {@code data} is
     * frame-major: {@code data[t * nVocab + v]} is the score for vocabulary
     * entry {@code v} at encoder frame {@code t}, so its length is
     * {@code nFrames * nVocab}. The Omni and wav2vec2 grids are raw logits
     * (pre-softmax); the canary-ctc grid is log-probabilities.
     */
    public static final class CtcLogits {
        /** Vocabulary size — the number of CTC output classes scored per frame. */
        public final int nVocab;
        /** Number of encoder frames (the time axis). */
        public final int nFrames;
        /** Frame-major CTC grid of length {@code nFrames * nVocab} (raw logits for Omni &amp; wav2vec2, log-probabilities for canary-ctc). */
        public final float[] data;
        CtcLogits(int nVocab, int nFrames, float[] data) {
            this.nVocab = nVocab; this.nFrames = nFrames; this.data = data;
        }
    }

    /**
     * Segments plus the optional CTC logit grid from
     * {@link #transcribeWithLogits(float[], String)}. {@code logits} is
     * {@code null} for backends that produce no dense CTC grid.
     */
    public static final class TranscriptionWithLogits {
        public final Segment[] segments;
        public final CtcLogits logits;
        TranscriptionWithLogits(Segment[] segments, CtcLogits logits) {
            this.segments = segments; this.logits = logits;
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

    /**
     * Transcribe and also return the per-frame CTC logits captured for this
     * call (backends with a dense CTC grid: Omni CTC, wav2vec2/hubert/data2vec,
     * canary-ctc). Opts in for the duration of the call — no need to call
     * {@link #setReturnLogits(boolean)} first — then returns the segments plus a
     * {@link CtcLogits} grid, or a {@code null} {@code logits} field for
     * backends that produce no dense CTC grid.
     */
    public TranscriptionWithLogits transcribeWithLogits(float[] pcm, String lang) {
        setReturnLogits(true);
        Pointer r = Lib.INSTANCE.crispasr_session_transcribe_lang(handle, pcm, pcm.length, lang);
        if (r == null) {
            setReturnLogits(false);
            throw new RuntimeException("transcription failed");
        }
        try {
            return new TranscriptionWithLogits(extractSegments(r), extractLogits(r));
        } finally {
            setReturnLogits(false);
            Lib.INSTANCE.crispasr_session_result_free(r);
        }
    }

    /**
     * Chunked-encode transcribe (issue #208): forces the Parakeet backend
     * through its bounded overlapping-window long-form path so long audio
     * transcribes in bounded time without dropping sections. Inert
     * (== {@link #transcribeLang}) on non-Parakeet backends. {@code chunkSeconds
     * <= 0} keeps the per-model default window; {@code overlapSeconds < 0} keeps
     * the default overlap. Poll {@link #getProgress()} (0..100) to render a
     * progress bar.
     */
    public Segment[] transcribeChunked(float[] pcm, int chunkSeconds, int overlapSeconds, String lang) {
        Pointer r = Lib.INSTANCE.crispasr_session_transcribe_chunked_lang(handle, pcm, pcm.length,
                chunkSeconds, overlapSeconds, lang);
        if (r == null) throw new RuntimeException("chunked transcription failed");
        try { return extractSegments(r); } finally { Lib.INSTANCE.crispasr_session_result_free(r); }
    }

    /**
     * Poll long-form (chunked) transcription progress: 0..100, or -1 when idle.
     * Updated in lockstep with {@link #transcribeChunked} windows (issue #208).
     */
    public static int getProgress() {
        return Lib.INSTANCE.crispasr_get_progress();
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
            float noSpeechProb = Lib.INSTANCE.crispasr_session_result_segment_no_speech_prob(r, i);
            String speaker = Lib.INSTANCE.crispasr_session_result_segment_speaker(r, i);
            segs[i] = new Segment(text, t0, t1, words, noSpeechProb, speaker);
        }
        return segs;
    }

    /**
     * The acoustic language Whisper detected on the last transcribe, as an
     * ISO-639-1 code ("en"). Whisper-only; other backends return the session's
     * source-language hint, or "unknown".
     */
    public String detectedLanguage() {
        if (handle == null) throw new IllegalStateException("session is closed");
        byte[] buf = new byte[32];
        Lib.INSTANCE.crispasr_session_detected_language(handle, buf, buf.length);
        return nullTerminated(buf);
    }

    // Copy the result-owned float* logit grid into a Java float[] before the
    // result handle is freed (same getFloatArray copy-out idiom as synthesize).
    private static CtcLogits extractLogits(Pointer r) {
        int nFrames = Lib.INSTANCE.crispasr_session_result_n_logit_frames(r);
        int nVocab = Lib.INSTANCE.crispasr_session_result_n_logit_vocab(r);
        Pointer ptr = Lib.INSTANCE.crispasr_session_result_logits(r);
        if (nFrames <= 0 || nVocab <= 0 || ptr == null) return null;
        float[] data = ptr.getFloatArray(0, nFrames * nVocab);
        return new CtcLogits(nVocab, nFrames, data);
    }

    /**
     * The Omni CTC vocabulary as raw pieces, indexed by token id
     * ({@code vocab[id]}). Pieces keep their word-boundary marker intact (the
     * v2 Omni vocab uses a literal space, v1 uses U+2581), so a consumer can
     * detokenize a greedy CTC decode over the grid from
     * {@link #transcribeWithLogits(float[], String)}. Returns {@code null} for
     * backends that don't expose a CTC vocab.
     */
    public String[] ctcVocab() {
        int n = Lib.INSTANCE.crispasr_session_n_vocab(handle);
        if (n <= 0) return null;
        String[] vocab = new String[n];
        for (int i = 0; i < n; i++) {
            String p = Lib.INSTANCE.crispasr_session_token_text(handle, i);
            vocab[i] = p != null ? p : "";
        }
        return vocab;
    }

    // -----------------------------------------------------------------
    // Forced alignment (PLAN #59)
    // -----------------------------------------------------------------

    /** One measured original codepoint; times are centiseconds. */
    public static final class AlignedCharacter {
        public final String text;
        public final long t0, t1;
        AlignedCharacter(String text, long t0, long t1) {
            this.text = text; this.t0 = t0; this.t1 = t1;
        }
    }

    /** One aligned word with timing. */
    public static final class AlignedWord {
        public final String text;
        public final long t0, t1; // centiseconds
        public final AlignedCharacter[] characters;
        AlignedWord(String text, long t0, long t1) { this(text, t0, t1, new AlignedCharacter[0]); }
        AlignedWord(String text, long t0, long t1, AlignedCharacter[] characters) {
            this.text = text; this.t0 = t0; this.t1 = t1; this.characters = characters;
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
                int nc = Lib.INSTANCE.crispasr_align_result_n_characters(r, i);
                AlignedCharacter[] characters = new AlignedCharacter[nc];
                for (int j = 0; j < nc; j++)
                    characters[j] = new AlignedCharacter(
                        Lib.INSTANCE.crispasr_align_result_character_text(r, i, j),
                        Lib.INSTANCE.crispasr_align_result_character_t0(r, i, j),
                        Lib.INSTANCE.crispasr_align_result_character_t1(r, i, j));
                words[i] = new AlignedWord(
                    Lib.INSTANCE.crispasr_align_result_word_text(r, i),
                    Lib.INSTANCE.crispasr_align_result_word_t0(r, i),
                    Lib.INSTANCE.crispasr_align_result_word_t1(r, i), characters);
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
