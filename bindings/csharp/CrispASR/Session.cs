using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace CrispASR
{
    /// <summary>
    /// Unified CrispASR Session — wraps the C ABI for ASR transcription,
    /// TTS synthesis, streaming, and language detection across all backends
    /// (whisper, parakeet, kokoro, vibevoice, qwen3-tts, orpheus, chatterbox, ...).
    /// <para>
    /// Usage:
    /// <code>
    /// using var session = Session.Open("model.gguf", nThreads: 4);
    /// var segments = session.Transcribe(pcm16kMono);
    /// // or for TTS:
    /// float[] audio = session.Synthesize("Hello, world!");
    /// </code>
    /// </para>
    /// </summary>
    public sealed class Session : IDisposable
    {
        private IntPtr _handle;

        private Session(IntPtr handle) => _handle = handle;

        private IntPtr Handle
        {
            get
            {
                if (_handle == IntPtr.Zero)
                    throw new ObjectDisposedException(nameof(Session));
                return _handle;
            }
        }

        /// <summary>
        /// Open a session with automatic backend detection from GGUF metadata.
        /// </summary>
        public static Session Open(string modelPath, int nThreads = 4)
        {
            var p = NativeMethods.crispasr_session_open(modelPath, nThreads);
            if (p == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to open session for {modelPath}");
            return new Session(p);
        }

        /// <summary>
        /// Open a session with an explicit backend name.
        /// </summary>
        public static Session Open(string modelPath, string backend, int nThreads = 4)
        {
            var p = NativeMethods.crispasr_session_open_explicit(modelPath, backend, nThreads);
            if (p == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to open session for {modelPath} (backend={backend})");
            return new Session(p);
        }

        /// <summary>
        /// Returns the comma-separated list of compiled-in backends.
        /// </summary>
        public static string[] AvailableBackends()
        {
            var buf = new byte[2048];
            NativeMethods.crispasr_session_available_backends(buf, buf.Length);
            var csv = NativeMethods.NullTerminated(buf);
            if (string.IsNullOrEmpty(csv)) return Array.Empty<string>();
            return csv.Split(',', StringSplitOptions.RemoveEmptyEntries);
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                NativeMethods.crispasr_session_close(_handle);
                _handle = IntPtr.Zero;
            }
        }

        // ----------------------------------------------------------------
        // Session-state setters
        // ----------------------------------------------------------------

        private static void Check(int rc, string name)
        {
            if (rc != 0 && rc != -2)
                throw new InvalidOperationException($"{name} failed (rc={rc})");
        }

        /// <summary>Load a separate codec GGUF (qwen3-tts, orpheus).</summary>
        public void SetCodecPath(string path)
            => Check(NativeMethods.crispasr_session_set_codec_path(Handle, path), "set_codec_path");

        /// <summary>Load a voice prompt GGUF or WAV reference. refText is for qwen3-tts WAV refs.</summary>
        public void SetVoice(string path, string? refText = null)
            => Check(NativeMethods.crispasr_session_set_voice(Handle, path, refText), "set_voice");

        /// <summary>Select a preset speaker by name (orpheus).</summary>
        public void SetSpeakerName(string name)
        {
            int rc = NativeMethods.crispasr_session_set_speaker_name(Handle, name);
            if (rc == -2) throw new ArgumentException($"Unknown speaker: {name}; call Speakers() to enumerate");
            if (rc == -3) throw new InvalidOperationException("Backend has no preset speakers; use SetVoice() instead");
            if (rc != 0) throw new InvalidOperationException($"set_speaker_name failed (rc={rc})");
        }

        /// <summary>Select a multi-speaker backend's speaker by index.</summary>
        public void SetSpeakerId(int id)
            => Check(NativeMethods.crispasr_session_set_speaker_id(Handle, id), "set_speaker_id");

        /// <summary>Return the list of preset speaker names for the active backend.</summary>
        public string[] Speakers()
        {
            int n = NativeMethods.crispasr_session_n_speakers(Handle);
            var result = new string[n];
            for (int i = 0; i < n; i++)
                result[i] = NativeMethods.PtrToUtf8(NativeMethods.crispasr_session_get_speaker_name(Handle, i)) ?? "";
            return result;
        }

        /// <summary>Set VoiceDesign instruction (qwen3-tts VoiceDesign).</summary>
        public void SetInstruct(string instruct)
        {
            int rc = NativeMethods.crispasr_session_set_instruct(Handle, instruct);
            if (rc == -3) throw new InvalidOperationException("Backend is not a VoiceDesign variant");
            if (rc != 0) throw new InvalidOperationException($"set_instruct failed (rc={rc})");
        }

        /// <summary>Whether the loaded model is a qwen3-tts CustomVoice variant.</summary>
        public bool IsCustomVoice => NativeMethods.crispasr_session_is_custom_voice(Handle) != 0;

        /// <summary>Whether the loaded model is a qwen3-tts VoiceDesign variant.</summary>
        public bool IsVoiceDesign => NativeMethods.crispasr_session_is_voice_design(Handle) != 0;

        /// <summary>Select + load a punctuation-restoration model.</summary>
        public void SetPuncModel(string puncModel)
            => Check(NativeMethods.crispasr_session_set_punc_model(Handle, puncModel), "set_punc_model");

        /// <summary>Comma-separated hotwords for contextual biasing.</summary>
        public void SetHotwords(string hotwords, float boost)
            => Check(NativeMethods.crispasr_session_set_hotwords(Handle, hotwords, boost), "set_hotwords");

        /// <summary>Select the G2P pronunciation dictionary for TTS.</summary>
        public void SetG2pDict(string source)
            => Check(NativeMethods.crispasr_session_set_g2p_dict(Handle, source), "set_g2p_dict");

        /// <summary>Sticky source-language hint.</summary>
        public void SetSourceLanguage(string? lang)
            => Check(NativeMethods.crispasr_session_set_source_language(Handle, lang ?? ""), "set_source_language");

        /// <summary>Sticky target-language. Different from source triggers translation on canary/cohere.</summary>
        public void SetTargetLanguage(string? lang)
            => Check(NativeMethods.crispasr_session_set_target_language(Handle, lang ?? ""), "set_target_language");

        /// <summary>Toggle punctuation + capitalisation. Default true.</summary>
        public void SetPunctuation(bool enable)
            => Check(NativeMethods.crispasr_session_set_punctuation(Handle, enable ? 1 : 0), "set_punctuation");

        /// <summary>Whisper sticky --translate.</summary>
        public void SetTranslate(bool enable)
            => Check(NativeMethods.crispasr_session_set_translate(Handle, enable ? 1 : 0), "set_translate");

        /// <summary>Decoder temperature on backends with runtime control.</summary>
        public void SetTemperature(float temperature, ulong seed = 0)
            => Check(NativeMethods.crispasr_session_set_temperature(Handle, temperature, seed), "set_temperature");

        /// <summary>Reseed TTS backends that support runtime seed control.</summary>
        public void SetTtsSeed(ulong seed)
            => Check(NativeMethods.crispasr_session_set_tts_seed(Handle, seed), "set_tts_seed");

        /// <summary>Generated-token cap for autoregressive session backends.</summary>
        public void SetMaxNewTokens(int n)
            => Check(NativeMethods.crispasr_session_set_max_new_tokens(Handle, n), "set_max_new_tokens");

        /// <summary>Repeated generated-token penalty for autoregressive backends.</summary>
        public void SetFrequencyPenalty(float penalty)
            => Check(NativeMethods.crispasr_session_set_frequency_penalty(Handle, penalty), "set_frequency_penalty");

        /// <summary>Diffusion / CFM step count (chatterbox).</summary>
        public void SetTtsSteps(int steps)
            => Check(NativeMethods.crispasr_session_set_tts_steps(Handle, steps), "set_tts_steps");

        /// <summary>Flow-matching timing candidates ranked per token (TADA).</summary>
        public void SetTtsNumCandidates(int n)
            => Check(NativeMethods.crispasr_session_set_tts_num_candidates(Handle, n), "set_tts_num_candidates");

        /// <summary>Top-p nucleus-sampling threshold.</summary>
        public void SetTopP(float topP)
            => Check(NativeMethods.crispasr_session_set_top_p(Handle, topP), "set_top_p");

        /// <summary>Top-k sampling cutoff (0 = disabled). Honoured by TADA.</summary>
        public void SetTopK(int topK)
            => Check(NativeMethods.crispasr_session_set_top_k(Handle, topK), "set_top_k");

        /// <summary>Enable/disable sampling (false = greedy). Honoured by TADA.</summary>
        public void SetDoSample(bool enable)
            => Check(NativeMethods.crispasr_session_set_do_sample(Handle, enable ? 1 : 0), "set_do_sample");

        /// <summary>Min-p sampling threshold.</summary>
        public void SetMinP(float minP)
            => Check(NativeMethods.crispasr_session_set_min_p(Handle, minP), "set_min_p");

        /// <summary>Repetition penalty (1.0 = no penalty).</summary>
        public void SetRepetitionPenalty(float r)
            => Check(NativeMethods.crispasr_session_set_repetition_penalty(Handle, r), "set_repetition_penalty");

        /// <summary>Classifier-free-guidance weight (chatterbox).</summary>
        public void SetCfgWeight(float cfgWeight)
            => Check(NativeMethods.crispasr_session_set_cfg_weight(Handle, cfgWeight), "set_cfg_weight");

        /// <summary>TADA flow-matching noise temperature (Python noise_temp, default 0.9).</summary>
        public void SetTtsNoiseTemp(float noiseTemp)
            => Check(NativeMethods.crispasr_session_set_tts_noise_temp(Handle, noiseTemp), "set_tts_noise_temp");

        /// <summary>Emotion-exaggeration scalar (chatterbox).</summary>
        public void SetExaggeration(float exaggeration)
            => Check(NativeMethods.crispasr_session_set_exaggeration(Handle, exaggeration), "set_exaggeration");

        /// <summary>Upper bound on speech tokens per synthesize call.</summary>
        public void SetMaxSpeechTokens(int n)
            => Check(NativeMethods.crispasr_session_set_max_speech_tokens(Handle, n), "set_max_speech_tokens");

        /// <summary>Per-phoneme length-scale / speaking-rate scalar (kokoro).</summary>
        public void SetLengthScale(float scale)
            => Check(NativeMethods.crispasr_session_set_length_scale(Handle, scale), "set_length_scale");

        /// <summary>Best-of-N sampling count for ASR backends.</summary>
        public void SetBestOf(int n)
            => Check(NativeMethods.crispasr_session_set_best_of(Handle, n), "set_best_of");

        /// <summary>Beam-search width for ASR backends that support it.</summary>
        public void SetBeamSize(int n)
            => Check(NativeMethods.crispasr_session_set_beam_size(Handle, n), "set_beam_size");

        /// <summary>Set a GBNF grammar for constrained whisper decoding.</summary>
        public void SetGrammarText(string gbnfText, string rootRule, float penalty)
        {
            int rc = NativeMethods.crispasr_session_set_grammar_text(Handle, gbnfText, rootRule, penalty);
            if (rc == -2) throw new ArgumentException("Invalid GBNF or root rule not found");
            if (rc != 0) throw new InvalidOperationException($"set_grammar_text failed (rc={rc})");
        }

        /// <summary>Set whisper decoder fallback thresholds. temperatureInc=0 disables fallback.</summary>
        public void SetFallbackThresholds(float entropyThold, float logprobThold, float noSpeechThold, float temperatureInc)
            => Check(NativeMethods.crispasr_session_set_fallback_thresholds(Handle,
                entropyThold, logprobThold, noSpeechThold, temperatureInc), "set_fallback_thresholds");

        /// <summary>Per-token top-N alternative-candidate capture for whisper greedy decode. 0 = off.</summary>
        public void SetAltN(int n)
            => Check(NativeMethods.crispasr_session_set_alt_n(Handle, n), "set_alt_n");

        /// <summary>Whisper-only text-suppression and prompt-carry extras.</summary>
        public void SetWhisperDecodeExtras(bool suppressNst, string? suppressRegex = null, bool carryInitialPrompt = false)
            => Check(NativeMethods.crispasr_session_set_whisper_decode_extras(Handle,
                suppressNst ? 1 : 0, suppressRegex ?? "", carryInitialPrompt ? 1 : 0), "set_whisper_decode_extras");

        /// <summary>Free-form prompt passed to the backend on the next transcribe/synthesize call.</summary>
        public void SetAsk(string prompt)
            => Check(NativeMethods.crispasr_session_set_ask(Handle, prompt), "set_ask");

        /// <summary>Drop the kokoro per-session phoneme cache.</summary>
        public void ClearPhonemeCache()
            => Check(NativeMethods.crispasr_session_kokoro_clear_phoneme_cache(Handle), "clear_phoneme_cache");

        // ----------------------------------------------------------------
        // TTS
        // ----------------------------------------------------------------

        /// <summary>
        /// Synthesize text to 24 kHz mono float32 PCM.
        /// Requires a TTS-capable backend (kokoro, vibevoice, qwen3-tts, orpheus, chatterbox).
        /// </summary>
        public float[] Synthesize(string text)
        {
            var ptr = NativeMethods.crispasr_session_synthesize(Handle, text, out int nSamples);
            if (ptr == IntPtr.Zero || nSamples <= 0)
                throw new InvalidOperationException("Synthesize returned no audio");
            try
            {
                var pcm = new float[nSamples];
                Marshal.Copy(ptr, pcm, 0, nSamples);
                return pcm;
            }
            finally
            {
                NativeMethods.crispasr_pcm_free(ptr);
            }
        }

        // ----------------------------------------------------------------
        // ASR Transcription
        // ----------------------------------------------------------------

        /// <summary>Transcribe 16 kHz mono float32 PCM.</summary>
        public Segment[] Transcribe(float[] pcm)
            => TranscribeLang(pcm, null);

        /// <summary>Transcribe with explicit language hint.</summary>
        public Segment[] TranscribeLang(float[] pcm, string? language)
        {
            var r = NativeMethods.crispasr_session_transcribe_lang(Handle, pcm, pcm.Length, language);
            if (r == IntPtr.Zero) throw new InvalidOperationException("Transcription failed");
            try { return ExtractSegments(r); }
            finally { NativeMethods.crispasr_session_result_free(r); }
        }

        /// <summary>Transcribe with VAD segmentation.</summary>
        public Segment[] TranscribeVad(float[] pcm, int sampleRate, string vadModelPath)
        {
            var r = NativeMethods.crispasr_session_transcribe_vad(
                Handle, pcm, pcm.Length, sampleRate, vadModelPath, IntPtr.Zero);
            if (r == IntPtr.Zero) throw new InvalidOperationException("Transcription with VAD failed");
            try { return ExtractSegments(r); }
            finally { NativeMethods.crispasr_session_result_free(r); }
        }

        private static Segment[] ExtractSegments(IntPtr r)
        {
            int nSegs = NativeMethods.crispasr_session_result_n_segments(r);
            var segs = new Segment[nSegs];
            for (int i = 0; i < nSegs; i++)
            {
                string text = NativeMethods.PtrToUtf8(
                    NativeMethods.crispasr_session_result_segment_text(r, i)) ?? "";
                long t0 = NativeMethods.crispasr_session_result_segment_t0(r, i);
                long t1 = NativeMethods.crispasr_session_result_segment_t1(r, i);

                int nWords = NativeMethods.crispasr_session_result_n_words(r, i);
                var words = new Word[nWords];
                for (int j = 0; j < nWords; j++)
                {
                    int nAlts = NativeMethods.crispasr_session_result_word_n_alts(r, i, j);
                    var alts = new AltToken[nAlts];
                    for (int k = 0; k < nAlts; k++)
                    {
                        alts[k] = new AltToken(
                            NativeMethods.PtrToUtf8(NativeMethods.crispasr_session_result_word_alt_text(r, i, j, k)) ?? "",
                            NativeMethods.crispasr_session_result_word_alt_p(r, i, j, k));
                    }
                    words[j] = new Word(
                        NativeMethods.PtrToUtf8(NativeMethods.crispasr_session_result_word_text(r, i, j)) ?? "",
                        NativeMethods.crispasr_session_result_word_t0(r, i, j),
                        NativeMethods.crispasr_session_result_word_t1(r, i, j),
                        NativeMethods.crispasr_session_result_word_p(r, i, j),
                        alts);
                }
                segs[i] = new Segment(text, t0, t1, words);
            }
            return segs;
        }

        // ----------------------------------------------------------------
        // Language detection
        // ----------------------------------------------------------------

        /// <summary>
        /// Detect spoken language on raw 16 kHz mono PCM.
        /// method: 0=Whisper, 1=Silero, 2=Firered, 3=Ecapa.
        /// </summary>
        public LanguageDetection DetectLanguage(float[] pcm, string lidModelPath, int method = 0)
        {
            var outLang = new byte[16];
            var outProb = new float[1];
            int rc = NativeMethods.crispasr_session_detect_language(
                Handle, pcm, pcm.Length, lidModelPath, method, outLang, outLang.Length, outProb);
            if (rc != 0)
                throw new InvalidOperationException($"detect_language failed (rc={rc})");
            return new LanguageDetection(NativeMethods.NullTerminated(outLang), outProb[0]);
        }

        /// <summary>
        /// Standalone language detection (no session needed).
        /// method: 0=Whisper, 1=Silero, 2=Firered, 3=Ecapa.
        /// </summary>
        public static LanguageDetection DetectLanguagePcm(float[] pcm, int method, string modelPath, int nThreads = 4)
        {
            var outLang = new byte[16];
            var outProb = new float[1];
            int rc = NativeMethods.crispasr_detect_language_pcm(
                pcm, pcm.Length, method, modelPath, nThreads, 0, 0, 0,
                outLang, outLang.Length, outProb);
            if (rc != 0)
                throw new InvalidOperationException($"detect_language_pcm failed (rc={rc})");
            return new LanguageDetection(NativeMethods.NullTerminated(outLang), outProb[0]);
        }

        // ----------------------------------------------------------------
        // Text translation
        // ----------------------------------------------------------------

        /// <summary>Translate text using the active backend's LLM. Returns translated text.</summary>
        public string TranslateText(string text, string srcLang, string tgtLang, int maxTokens = 256)
        {
            var ptr = NativeMethods.crispasr_session_translate_text(Handle, text, srcLang, tgtLang, maxTokens);
            if (ptr == IntPtr.Zero)
                throw new InvalidOperationException("translate_text failed");
            try
            {
                return NativeMethods.PtrToUtf8(ptr) ?? "";
            }
            finally
            {
                NativeMethods.crispasr_session_translate_text_free(ptr);
            }
        }

        // ----------------------------------------------------------------
        // Streaming
        // ----------------------------------------------------------------

        /// <summary>
        /// Open a rolling-window streaming decoder. Whisper-only at the C-ABI level today.
        /// </summary>
        public StreamDecoder StreamOpen(int stepMs = 3000, int lengthMs = 10000, int keepMs = 200,
                                        string? language = null, bool translate = false, int nThreads = 4)
        {
            var p = NativeMethods.crispasr_session_stream_open(
                Handle, nThreads, stepMs, lengthMs, keepMs, language ?? "", translate ? 1 : 0);
            if (p == IntPtr.Zero)
                throw new InvalidOperationException("stream_open failed (whisper-only today)");
            return new StreamDecoder(p);
        }

        // ----------------------------------------------------------------
        // Forced alignment (static)
        // ----------------------------------------------------------------

        /// <summary>Run CTC forced alignment on transcript + audio.</summary>
        public static AlignedWord[] AlignWords(string alignerModel, string transcript,
                                                float[] pcm, long tOffsetCs = 0, int nThreads = 4)
        {
            var r = NativeMethods.crispasr_align_words_abi(alignerModel, transcript, pcm, pcm.Length, tOffsetCs, nThreads);
            if (r == IntPtr.Zero) throw new InvalidOperationException("Alignment failed");
            try
            {
                int n = NativeMethods.crispasr_align_result_n_words(r);
                var words = new AlignedWord[n];
                for (int i = 0; i < n; i++)
                {
                    words[i] = new AlignedWord(
                        NativeMethods.PtrToUtf8(NativeMethods.crispasr_align_result_word_text(r, i)) ?? "",
                        NativeMethods.crispasr_align_result_word_t0(r, i),
                        NativeMethods.crispasr_align_result_word_t1(r, i));
                }
                return words;
            }
            finally
            {
                NativeMethods.crispasr_align_result_free(r);
            }
        }

        // ----------------------------------------------------------------
        // Standalone VAD (static)
        // ----------------------------------------------------------------

        /// <summary>Run standalone VAD. Returns speech spans in seconds.</summary>
        public static VadSpan[] VadSegments(string vadModelPath, float[] pcm, int sampleRate,
                                             float threshold = 0.5f, int minSpeechMs = 250,
                                             int minSilenceMs = 100, int nThreads = 4)
        {
            var outSpans = new IntPtr[1];
            int n = NativeMethods.crispasr_vad_segments(vadModelPath, pcm, pcm.Length,
                sampleRate, threshold, minSpeechMs, minSilenceMs, nThreads, 0, outSpans);
            if (n < 0) throw new InvalidOperationException($"VAD failed (rc={n})");
            if (n == 0 || outSpans[0] == IntPtr.Zero) return Array.Empty<VadSpan>();
            try
            {
                var raw = new float[n * 2];
                Marshal.Copy(outSpans[0], raw, 0, n * 2);
                var spans = new VadSpan[n];
                for (int i = 0; i < n; i++)
                    spans[i] = new VadSpan(raw[i * 2], raw[i * 2 + 1]);
                return spans;
            }
            finally
            {
                NativeMethods.crispasr_vad_free(outSpans[0]);
            }
        }

        // ----------------------------------------------------------------
        // Backend detection (static)
        // ----------------------------------------------------------------

        /// <summary>Detect the backend name from a GGUF file's metadata.</summary>
        public static string? DetectBackendFromGguf(string path)
        {
            var buf = new byte[128];
            int rc = NativeMethods.crispasr_detect_backend_from_gguf(path, buf, buf.Length);
            if (rc != 0) return null;
            return NativeMethods.NullTerminated(buf);
        }

        // ----------------------------------------------------------------
        // Audio enhancement (static)
        // ----------------------------------------------------------------

        /// <summary>Enhance audio using RNNoise (48 kHz mono).</summary>
        public static float[] EnhanceAudioRnnoise(float[] pcm)
        {
            var outPcm = new float[pcm.Length];
            int rc = NativeMethods.crispasr_enhance_audio_rnnoise(pcm, pcm.Length, outPcm, outPcm.Length);
            if (rc != 0) throw new InvalidOperationException($"enhance_audio_rnnoise failed (rc={rc})");
            return outPcm;
        }

        // ----------------------------------------------------------------
        // Kokoro per-language routing (static)
        // ----------------------------------------------------------------

        /// <summary>
        /// Resolve the kokoro model + fallback voice for a language.
        /// Call before <see cref="Open(string, int)"/> to get the correct model path.
        /// </summary>
        public static KokoroResolved KokoroResolveForLang(string modelPath, string lang)
        {
            var outModel = new byte[1024];
            var outVoice = new byte[1024];
            var outPicked = new byte[64];

            int rc = NativeMethods.crispasr_kokoro_resolve_model_for_lang_abi(
                modelPath, lang ?? "", outModel, outModel.Length);
            if (rc < 0) throw new InvalidOperationException("kokoro_resolve_model_for_lang: buffer too small");
            bool swapped = (rc == 0);
            string resolvedModel = NativeMethods.NullTerminated(outModel);
            if (string.IsNullOrEmpty(resolvedModel)) resolvedModel = modelPath;

            rc = NativeMethods.crispasr_kokoro_resolve_fallback_voice_abi(
                modelPath, lang ?? "", outVoice, outVoice.Length, outPicked, outPicked.Length);
            if (rc < 0) throw new InvalidOperationException("kokoro_resolve_fallback_voice: buffer too small");
            if (rc == 0)
            {
                return new KokoroResolved(resolvedModel,
                    NativeMethods.NullTerminated(outVoice),
                    NativeMethods.NullTerminated(outPicked),
                    swapped);
            }
            return new KokoroResolved(resolvedModel, null, null, swapped);
        }
    }

    // ====================================================================
    // Data types
    // ====================================================================

    /// <summary>One word with timing and confidence from a transcription result.</summary>
    public readonly struct Word
    {
        public string Text { get; }
        public long T0 { get; }
        public long T1 { get; }
        public float P { get; }
        public AltToken[] Alts { get; }

        public Word(string text, long t0, long t1, float p, AltToken[]? alts = null)
        {
            Text = text; T0 = t0; T1 = t1; P = p;
            Alts = alts ?? Array.Empty<AltToken>();
        }

        public override string ToString() => $"{T0}-{T1} {Text}";
    }

    /// <summary>Alternative token candidate.</summary>
    public readonly struct AltToken
    {
        public string Text { get; }
        public float P { get; }

        public AltToken(string text, float p) { Text = text; P = p; }

        public override string ToString() => $"{Text}({P * 100:F1}%)";
    }

    /// <summary>One segment from a transcription result.</summary>
    public readonly struct Segment
    {
        public string Text { get; }
        public long T0 { get; }
        public long T1 { get; }
        public Word[] Words { get; }

        public Segment(string text, long t0, long t1, Word[] words)
        {
            Text = text; T0 = t0; T1 = t1; Words = words;
        }

        public override string ToString() => $"[{T0}-{T1}] {Text}";
    }

    /// <summary>One aligned word from forced alignment.</summary>
    public readonly struct AlignedWord
    {
        public string Text { get; }
        public long T0 { get; }
        public long T1 { get; }

        public AlignedWord(string text, long t0, long t1) { Text = text; T0 = t0; T1 = t1; }
    }

    /// <summary>One speech span from VAD (seconds).</summary>
    public readonly struct VadSpan
    {
        public double T0 { get; }
        public double T1 { get; }

        public VadSpan(double t0, double t1) { T0 = t0; T1 = t1; }
    }

    /// <summary>Language detection result.</summary>
    public readonly struct LanguageDetection
    {
        public string Code { get; }
        public float Probability { get; }

        public LanguageDetection(string code, float probability) { Code = code; Probability = probability; }

        public bool Ok => !string.IsNullOrEmpty(Code) && Probability >= 0f;

        public override string ToString() => $"LanguageDetection({Code}, {Probability * 100:F1}%)";
    }

    /// <summary>Result of <see cref="Session.KokoroResolveForLang"/>.</summary>
    public readonly struct KokoroResolved
    {
        public string ModelPath { get; }
        public string? VoicePath { get; }
        public string? VoiceName { get; }
        public bool BackboneSwapped { get; }

        public KokoroResolved(string modelPath, string? voicePath, string? voiceName, bool backboneSwapped)
        {
            ModelPath = modelPath; VoicePath = voicePath; VoiceName = voiceName; BackboneSwapped = backboneSwapped;
        }
    }

    // ====================================================================
    // Streaming decoder
    // ====================================================================

    /// <summary>
    /// Rolling-window streaming decoder handle. Feed PCM, pull text.
    /// Whisper-only at the C-ABI level today.
    /// </summary>
    public sealed class StreamDecoder : IDisposable
    {
        private IntPtr _handle;

        internal StreamDecoder(IntPtr handle) => _handle = handle;

        private IntPtr Handle
        {
            get
            {
                if (_handle == IntPtr.Zero) throw new ObjectDisposedException(nameof(StreamDecoder));
                return _handle;
            }
        }

        /// <summary>
        /// Push 16 kHz mono float32 PCM. Returns 0 if still buffering,
        /// 1 if a new partial transcript is ready.
        /// </summary>
        public int Feed(float[] pcm)
        {
            if (pcm == null || pcm.Length == 0) return 0;
            int rc = NativeMethods.crispasr_stream_feed(Handle, pcm, pcm.Length);
            if (rc < 0) throw new InvalidOperationException($"stream_feed failed (rc={rc})");
            return rc;
        }

        /// <summary>Latest committed transcript + absolute audio-time bounds.</summary>
        public StreamingUpdate GetText()
        {
            var buf = new byte[8192];
            int rc = NativeMethods.crispasr_stream_get_text(Handle, buf, buf.Length,
                out double t0, out double t1, out long counter);
            if (rc < 0) throw new InvalidOperationException($"stream_get_text failed (rc={rc})");
            string text = NativeMethods.NullTerminated(buf);
            return new StreamingUpdate(text, t0, t1, counter);
        }

        /// <summary>Force a decode on whatever is buffered.</summary>
        public void Flush()
        {
            int rc = NativeMethods.crispasr_stream_flush(Handle);
            if (rc < 0) throw new InvalidOperationException($"stream_flush failed (rc={rc})");
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                NativeMethods.crispasr_stream_close(_handle);
                _handle = IntPtr.Zero;
            }
        }
    }

    /// <summary>Per-commit update from a streaming session.</summary>
    public readonly struct StreamingUpdate
    {
        public string Text { get; }
        public double T0 { get; }
        public double T1 { get; }
        public long Counter { get; }

        public StreamingUpdate(string text, double t0, double t1, long counter)
        {
            Text = text; T0 = t0; T1 = t1; Counter = counter;
        }
    }

    // ====================================================================
    // TitaNet speaker embedding
    // ====================================================================

    /// <summary>TitaNet speaker embedding model.</summary>
    public sealed class TitaNet : IDisposable
    {
        private IntPtr _handle;
        private const int EmbeddingDim = 192;

        private TitaNet(IntPtr handle) => _handle = handle;

        public static TitaNet Open(string modelPath, int nThreads = 4)
        {
            var p = NativeMethods.crispasr_titanet_init(modelPath, nThreads);
            if (p == IntPtr.Zero) throw new InvalidOperationException($"Failed to init TitaNet from {modelPath}");
            return new TitaNet(p);
        }

        /// <summary>Compute a 192-dim speaker embedding from 16 kHz mono PCM.</summary>
        public float[] Embed(float[] pcm16k)
        {
            var emb = new float[EmbeddingDim];
            int rc = NativeMethods.crispasr_titanet_embed(_handle, pcm16k, pcm16k.Length, emb);
            if (rc != 0) throw new InvalidOperationException($"titanet_embed failed (rc={rc})");
            return emb;
        }

        /// <summary>Cosine similarity between two embeddings.</summary>
        public static float CosineSim(float[] a, float[] b)
        {
            if (a.Length != b.Length) throw new ArgumentException("Embedding dimensions must match");
            return NativeMethods.crispasr_titanet_cosine_sim(a, b, a.Length);
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                NativeMethods.crispasr_titanet_free(_handle);
                _handle = IntPtr.Zero;
            }
        }
    }

    // ====================================================================
    // Speaker database
    // ====================================================================

    /// <summary>On-disk speaker embedding database.</summary>
    public sealed class SpeakerDb : IDisposable
    {
        private IntPtr _handle;

        private SpeakerDb(IntPtr handle) => _handle = handle;

        public static SpeakerDb Load(string dirPath)
        {
            var p = NativeMethods.crispasr_speaker_db_load(dirPath);
            if (p == IntPtr.Zero) throw new InvalidOperationException($"Failed to load speaker db from {dirPath}");
            return new SpeakerDb(p);
        }

        public int Count => NativeMethods.crispasr_speaker_db_count(_handle);

        /// <summary>Match an embedding against the database. Returns (name, score) or (null, 0) if below threshold.</summary>
        public (string? Name, float Score) Match(float[] embedding, float threshold = 0.5f)
        {
            var outName = new byte[256];
            float score = NativeMethods.crispasr_speaker_db_match(
                _handle, embedding, embedding.Length, threshold, outName, outName.Length);
            if (score < threshold) return (null, score);
            return (NativeMethods.NullTerminated(outName), score);
        }

        /// <summary>Enroll a new speaker embedding.</summary>
        public static void Enroll(string dirPath, string name, float[] embedding)
        {
            int rc = NativeMethods.crispasr_speaker_db_enroll(dirPath, name, embedding, embedding.Length);
            if (rc != 0) throw new InvalidOperationException($"speaker_db_enroll failed (rc={rc})");
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                NativeMethods.crispasr_speaker_db_free(_handle);
                _handle = IntPtr.Zero;
            }
        }
    }
}
