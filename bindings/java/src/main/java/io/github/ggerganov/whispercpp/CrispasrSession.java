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
        int     crispasr_session_detect_language(Pointer session, float[] pcm, int n_samples,
                                                  String lid_model_path, int method,
                                                  byte[] out_lang, int out_lang_cap, float[] out_prob);

        int crispasr_kokoro_resolve_model_for_lang_abi(
                String modelPath, String lang, byte[] outPath, int outPathLen);
        int crispasr_kokoro_resolve_fallback_voice_abi(
                String modelPath, String lang,
                byte[] outPath, int outPathLen,
                byte[] outPicked, int outPickedLen);

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
