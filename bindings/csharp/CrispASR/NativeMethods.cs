using System;
using System.Runtime.InteropServices;
using System.Text;

namespace CrispASR
{
    /// <summary>
    /// Raw P/Invoke declarations for the crispasr shared library.
    /// The public <see cref="Session"/> class wraps these into safe,
    /// idiomatic C# types.
    /// </summary>
    internal static class NativeMethods
    {
        // One name, shared with NativeLibraryResolver — the resolver only
        // intercepts loads of exactly this name, so the two must not drift.
        private const string Lib = NativeLibraryResolver.LibraryName;

        // An explicit type initializer clears `beforefieldinit`, so the CLR is
        // required to run this before the first static member of the class is
        // touched — i.e. before any P/Invoke below can attempt a load. That is
        // what makes the resolver's search order authoritative rather than a
        // race against default probing.
        static NativeMethods() => NativeLibraryResolver.Install();

        // ---- Session lifecycle ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_open(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath, int nThreads);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_open_explicit(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string backendName,
            int nThreads);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void crispasr_session_close(IntPtr session);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_available_backends(
            byte[] outCsv, int outCap);

        // Acoustic detected language (Whisper) as an ISO-639-1 code into outBuf;
        // other backends fall back to the source-language hint, then "unknown".
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_detected_language(
            IntPtr session, byte[] outBuf, int outCap);

        // CTC vocabulary access (Omni CTC backend). n_vocab is the piece count;
        // token_text returns a model-owned const char* (do not free) or "" when
        // out of range / unsupported.
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_n_vocab(IntPtr s);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_token_text(IntPtr s, int id);

        // ---- Session setters ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_codec_path(
            IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string path);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_voice(
            IntPtr s,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string path,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? refText);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_speaker_name(
            IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string name);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_speaker_id(IntPtr s, int id);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_n_speakers(IntPtr s);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_get_speaker_name(IntPtr s, int i);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_instruct(
            IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string instruct);

        // #316: synthesize these phonemes verbatim, skipping the G2P. Empty clears. kokoro and piper only (rc=-2).
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_tts_phonemes(
            IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string phonemes);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void crispasr_session_set_tts_pad_silence_ms(
            IntPtr s, int ms);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_is_custom_voice(IntPtr s);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_is_voice_design(IntPtr s);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_punc_model(
            IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string puncModel);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_hotwords(
            IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string hotwords, float boost);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_sensitivity(
            IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string preset);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_g2p_dict(
            IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string source);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_source_language(
            IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string lang);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_target_language(
            IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string lang);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_tts_reference_language(
            IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string lang);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_punctuation(IntPtr s, int enable);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_translate(IntPtr s, int enable);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_temperature(IntPtr s, float temperature, ulong seed);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_tts_seed(IntPtr s, ulong seed);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_max_new_tokens(IntPtr s, int n);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_frequency_penalty(IntPtr s, float penalty);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_tts_steps(IntPtr s, int steps);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_tts_num_candidates(IntPtr s, int n);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_top_p(IntPtr s, float topP);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_top_k(IntPtr s, int topK);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_do_sample(IntPtr s, int enable);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_min_p(IntPtr s, float minP);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_repetition_penalty(IntPtr s, float r);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_cfg_weight(IntPtr s, float cfgWeight);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_tts_noise_temp(IntPtr s, float noiseTemp);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_exaggeration(IntPtr s, float exaggeration);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_max_speech_tokens(IntPtr s, int n);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_min_speech_tokens(IntPtr s, int n);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_length_scale(IntPtr s, float scale);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_best_of(IntPtr s, int n);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_beam_size(IntPtr s, int n);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_return_logits(IntPtr s, int enable);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_grammar_text(
            IntPtr s,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string gbnfText,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string rootRule,
            float penalty);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_fallback_thresholds(
            IntPtr s, float entropyThold, float logprobThold,
            float noSpeechThold, float temperatureInc);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_alt_n(IntPtr s, int n);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_whisper_decode_extras(
            IntPtr s, int suppressNst,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string suppressRegex,
            int carryInitialPrompt);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_ask(
            IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string prompt);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_kokoro_clear_phoneme_cache(IntPtr s);

        // ---- TTS synthesize ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_synthesize(
            IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string text,
            out int outNSamples);

        // UNMARKED synthesis — hard-refused unless accept_marking_responsibility was called first.
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_synthesize_raw(
            IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string text,
            out int outNSamples);

        // Attest acceptance of AI-content marking/disclosure duty (EU AI Act Art. 50).
        // Whose voice a PRESET voice is (EU AI Act Art. 50(4)). 0 ok, -1 bad
        // session, -2 unrecognised value.
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_speaker_identity(
            IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string identity);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_accept_marking_responsibility(
            IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string attestation);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void crispasr_pcm_free(IntPtr pcm);

        // Speech-to-speech — audio in → audio out via a single model pass
        // (lfm2-audio, mini-omni2, sidon, voxcpm2-vae). Returns malloc'd float32
        // PCM (free via crispasr_pcm_free); outText, when non-null, receives the
        // intermediate transcript (malloc'd, free via
        // crispasr_session_translate_text_free).
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_speech_to_speech(
            IntPtr s, float[] inSamples, int nInSamples,
            out IntPtr outText, out int outNSamples);

        // Sample rate the backend expects for input PCM (16000 for
        // Whisper-family backends, 0 on error).
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_input_sample_rate(IntPtr s);

        // #332: output-side counterparts. output_sample_rate is the rate of
        // the PCM synthesize / speech_to_speech return (0 = no audio output);
        // the channel getters are 1 (mono) for every current backend.
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_output_sample_rate(IntPtr s);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_input_channels(IntPtr s);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_output_channels(IntPtr s);

        // AI-content marking (EU AI Act Art. 50(2)) — the other half of
        // synthesize_raw. alpha <= 0 selects the reliably detectable default.
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void crispasr_watermark_embed(float[] pcm, int nSamples, float alpha);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern float crispasr_watermark_detect(float[] pcm, int nSamples);

        // ---- ASR transcription ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_transcribe(
            IntPtr s, float[] pcm, int nSamples);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_transcribe_lang(
            IntPtr s, float[] pcm, int nSamples,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? language);

        // Chunked-encode transcribe (issue #208): forces the Parakeet backend
        // through its bounded overlapping-window long-form path. chunkSeconds<=0
        // keeps the per-model default window; overlapSeconds<0 keeps the default
        // overlap. Inert (== transcribe[_lang]) on non-Parakeet backends.
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_transcribe_chunked(
            IntPtr s, float[] pcm, int nSamples, int chunkSeconds, int overlapSeconds);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_transcribe_chunked_lang(
            IntPtr s, float[] pcm, int nSamples, int chunkSeconds, int overlapSeconds,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? language);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_transcribe_vad(
            IntPtr s, float[] pcm, int nSamples, int sampleRate,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string vadModelPath,
            IntPtr optsOrNull);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_transcribe_vad_lang(
            IntPtr s, float[] pcm, int nSamples, int sampleRate,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string vadModelPath,
            IntPtr optsOrNull,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? language);

        // ---- Result accessors ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_result_n_segments(IntPtr result);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_result_segment_text(IntPtr result, int i);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern long crispasr_session_result_segment_t0(IntPtr result, int i);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern long crispasr_session_result_segment_t1(IntPtr result, int i);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_result_n_words(IntPtr result, int iSeg);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_result_word_text(IntPtr result, int iSeg, int iWord);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern long crispasr_session_result_word_t0(IntPtr result, int iSeg, int iWord);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern long crispasr_session_result_word_t1(IntPtr result, int iSeg, int iWord);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern float crispasr_session_result_word_p(IntPtr result, int iSeg, int iWord);

        // Whisper's per-segment no-speech probability in [0, 1]; -1.0 sentinel
        // ("no data") for other backends and out-of-range indices.
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern float crispasr_session_result_segment_no_speech_prob(IntPtr result, int iSeg);

        // #300: native per-segment speaker label ("(Speaker N) "), or "" when the
        // backend does not diarize natively. Never NULL. Returned as IntPtr and
        // marshalled via PtrToUtf8 like the other const char* getters — a
        // [return: MarshalAs(LPUTF8Str)] would make the CLR try to free it.
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_result_segment_speaker(IntPtr result, int i);

        // Per-frame CTC logits (opted in via crispasr_session_set_return_logits)
        // for backends with a dense CTC grid (Omni CTC, wav2vec2/hubert/data2vec,
        // canary-ctc). _logits returns a const float* (frame-major;
        // logits[t * nVocab + v]) or NULL when none. Raw pre-softmax for Omni &
        // wav2vec2; log-probabilities for canary-ctc.
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_result_n_logit_frames(IntPtr result);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_result_n_logit_vocab(IntPtr result);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_result_logits(IntPtr result);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_result_word_n_alts(IntPtr result, int iSeg, int iWord);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_result_word_alt_text(IntPtr result, int iSeg, int iWord, int iAlt);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern float crispasr_session_result_word_alt_p(IntPtr result, int iSeg, int iWord, int iAlt);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void crispasr_session_result_free(IntPtr result);

        // ---- Language detection ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_detect_language(
            IntPtr s, float[] pcm, int nSamples,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string lidModelPath,
            int method, byte[] outLang, int outLangCap, float[] outProb);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_detect_language_pcm(
            float[] samples, int nSamples, int method,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath,
            int nThreads, int useGpu, int gpuDevice, int flashAttn,
            byte[] outLang, int outLangCap, float[] outProb);

        // ---- Alignment ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_align_words_abi(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string alignerModel,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string transcript,
            float[] samples, int nSamples, long tOffsetCs, int nThreads);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_align_result_n_words(IntPtr result);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_align_result_word_text(IntPtr result, int i);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern long crispasr_align_result_word_t0(IntPtr result, int i);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern long crispasr_align_result_word_t1(IntPtr result, int i);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void crispasr_align_result_free(IntPtr result);

        // ---- VAD ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_vad_segments(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string vadModelPath,
            float[] pcm, int nSamples, int sampleRate,
            float threshold, int minSpeechMs, int minSilenceMs,
            int nThreads, int useGpu, IntPtr[] outSpans);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void crispasr_vad_free(IntPtr spans);

        // ---- Logging ----
        // ggml_log_callback: void (*)(ggml_log_level level, const char* text, void* user_data)
        // Cdecl is mandatory: native calls this back with the C calling convention,
        // and the delegate default (StdCall) would corrupt the stack on every log
        // line on the platforms CrispASR actually ships to.
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void GgmlLogCallback(int level, [MarshalAs(UnmanagedType.LPUTF8Str)] string? text, IntPtr userData);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void whisper_log_set(GgmlLogCallback? logCallback, IntPtr userData);

        // ---- Music / task backends (tab, beats, chords, piano, pitch, separate, convert) ----
        // Every `_events`/`_spans`/`_notes`/`_frames`/`_emissions`/`_stem`/`_audio`
        // getter returns a SESSION-OWNED float* valid only until the next run call
        // or session close — copy out immediately, never free it (unlike synthesize,
        // which owns its buffer and needs crispasr_pcm_free).

        // tab (guitar tablature)
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_tab(IntPtr s, float[] pcm, int nSamples, int sampleRate);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_tab_n_frames(IntPtr s);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_tab_emissions(IntPtr s, out int nFrames, out int nStrings, out int nClasses);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_tab_silent_class(IntPtr s);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern float crispasr_session_tab_frame_period(IntPtr s);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_tab_string_open_midi(IntPtr s, int stringIndex);

        // beats (beat / downbeat tracking)
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_beats(IntPtr s, float[] pcm, int nSamples, int sampleRate);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_beats_n_events(IntPtr s);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_beats_events(IntPtr s, out int nEvents);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern float crispasr_session_beats_tempo_bpm(IntPtr s);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_beats_sample_rate(IntPtr s);

        // chords
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_chords(IntPtr s, float[] pcm, int nSamples, int sampleRate);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_chords_n_spans(IntPtr s);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_chords_spans(IntPtr s, out int nSpans);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_chords_span_name(IntPtr s, int idx);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_chords_vocab_size(IntPtr s);

        // piano transcription
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_piano(IntPtr s, float[] pcm16k, int nSamples);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_piano_n_notes(IntPtr s);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_piano_notes(IntPtr s, out int nNotes);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_piano_note_programs(IntPtr s, out int nNotes);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_piano_sample_rate(IntPtr s);

        // pitch (F0)
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_pitch(IntPtr s, float[] pcm16k, int nSamples, float hopMs);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_pitch_n_frames(IntPtr s);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_pitch_frames(IntPtr s, out int nFrames);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_pitch_sample_rate(IntPtr s);

        // source separation
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_separate(IntPtr s, float[] pcmStereo, int nSamples);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_separate_n_stems(IntPtr s);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_separate_stem_name(IntPtr s, int stemIdx);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_separate_stem(IntPtr s, int stemIdx, out int nSamples);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_separate_sample_rate(IntPtr s);

        // voice conversion (RVC content -> speaker)
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_convert(IntPtr s, float[] content, int nFrames,
            float[] f0Hz, int speakerId, float[]? noiseZp, float[]? noiseSine);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_convert_audio(IntPtr s, out int nSamples);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_convert_content_dim(IntPtr s);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_convert_n_speakers(IntPtr s);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_convert_sample_rate(IntPtr s);

        // ---- Streaming ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_stream_open(
            IntPtr s, int nThreads, int stepMs, int lengthMs,
            int keepMs, [MarshalAs(UnmanagedType.LPUTF8Str)] string language, int translate);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_stream_feed(IntPtr stream, float[] pcm, int nSamples);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_stream_get_text(
            IntPtr stream, byte[] outText, int outCap,
            out double outT0, out double outT1, out long outCounter);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_stream_flush(IntPtr stream);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void crispasr_stream_close(IntPtr stream);

        // ---- Kokoro lang helpers ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_kokoro_resolve_model_for_lang_abi(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string lang,
            byte[] outPath, int outPathLen);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_kokoro_resolve_fallback_voice_abi(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string lang,
            byte[] outPath, int outPathLen,
            byte[] outPicked, int outPickedLen);

        // ---- Backend detection ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_detect_backend_from_gguf(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string path,
            byte[] outName, int outCap);

        // ---- Audio enhancement ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_enhance_audio_rnnoise(
            float[] inPcm, int nSamples, float[] outPcm, int outCap);

        // ---- Speaker embedding (TitaNet) ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_titanet_init(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath, int nThreads);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void crispasr_titanet_free(IntPtr ctx);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_titanet_embed(IntPtr ctx, float[] pcm16k, int nSamples, float[] outEmb);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern float crispasr_titanet_cosine_sim(float[] a, float[] b, int dim);

        // ---- Speaker database (closed-roster, consent-gated — issue #266) ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_speaker_db_open(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string dirPath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string expectedNamesCsv,
            int consentAttested);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void crispasr_speaker_db_free(IntPtr db);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_speaker_db_count(IntPtr db);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern float crispasr_speaker_db_match(
            IntPtr db, float[] embedding, int dim, float threshold,
            byte[] outName, int outCap);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_speaker_db_enroll2(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string dirPath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string name,
            float[] embedding, int dim, int consentAttested);

        // ---- Text translation ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_translate_text(
            IntPtr s,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string text,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string srcLang,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string tgtLang,
            int maxTokens);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void crispasr_session_translate_text_free(IntPtr text);

        // ---- Registry + cache ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_registry_lookup_abi(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string backend,
            byte[] outFilename, int filenameCap,
            byte[] outUrl, int urlCap,
            byte[] outSize, int sizeCap);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_registry_default_bundle_info_abi(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string backend,
            byte[] outBackend, int backendCap,
            byte[] outLicense, int licenseCap,
            out int outRequiresAcceptance);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_registry_default_bundle_artifact_abi(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string backend,
            int index, out int outKind,
            byte[] outFilename, int filenameCap,
            byte[] outUrl, int urlCap,
            byte[] outSize, int sizeCap);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_cache_ensure_file_abi(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string filename,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string url,
            int quiet,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? cacheDirOverride,
            byte[] outBuf, int outCap);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_cache_dir_abi(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string? cacheDirOverride,
            byte[] outBuf, int outCap);

        // ---- Audio decode (issue #291: the binding had no way to read a file) ----
        // Decode any supported format to mono float32 PCM. *out_pcm is
        // malloc-owned by the native side — release it with crispasr_audio_free.
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_audio_load(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string path,
            out IntPtr outPcm, out int outSamples, out int outSampleRate);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_audio_load_at_rate(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string path, int targetRate,
            out IntPtr outPcm, out int outSamples, out int outSampleRate);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void crispasr_audio_free(IntPtr pcm);

        // Returns malloc'd WAV bytes; released with crispasr_c2pa_free (the
        // shared free for the C2PA/provenance byte-buffer family, per crispasr.h).
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_pcm_to_wav(
            float[] pcm, int nSamples, int sampleRate, out UIntPtr outLen);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void crispasr_c2pa_free(IntPtr p);

        // ---- Helpers ----
        internal static string NullTerminated(byte[] buf)
        {
            int n = 0;
            while (n < buf.Length && buf[n] != 0) n++;
            return Encoding.UTF8.GetString(buf, 0, n);
        }

        internal static string? PtrToUtf8(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero) return null;
            return Marshal.PtrToStringUTF8(ptr);
        }
    }
}
