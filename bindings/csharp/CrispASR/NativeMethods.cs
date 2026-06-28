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
        private const string Lib = "crispasr";

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
        internal static extern int crispasr_session_set_g2p_dict(
            IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string source);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_source_language(
            IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string lang);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_target_language(
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
        internal static extern int crispasr_session_set_length_scale(IntPtr s, float scale);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_best_of(IntPtr s, int n);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_session_set_beam_size(IntPtr s, int n);

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

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void crispasr_pcm_free(IntPtr pcm);

        // ---- ASR transcription ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_transcribe(
            IntPtr s, float[] pcm, int nSamples);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_session_transcribe_lang(
            IntPtr s, float[] pcm, int nSamples,
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

        // ---- Speaker database ----
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_speaker_db_load(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string dirPath);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void crispasr_speaker_db_free(IntPtr db);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_speaker_db_count(IntPtr db);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern float crispasr_speaker_db_match(
            IntPtr db, float[] embedding, int dim, float threshold,
            byte[] outName, int outCap);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_speaker_db_enroll(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string dirPath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string name,
            float[] embedding, int dim);

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
