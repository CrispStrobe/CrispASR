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
    public sealed partial class Session : IDisposable
    {
        private IntPtr _handle;

        // The session C ABI reports every time in centiseconds; this binding
        // exposes seconds throughout (issue #291). ONE constant, used by every
        // conversion site, so a future time-bearing accessor cannot pick a
        // different unit by accident.
        internal const double CentisecondsPerSecond = 100.0;

        // The C ABI uses -1 for "this backend produced no timing for this
        // unit" — moonshine sets every token t0/t1 to -1, exactly as p = -1
        // means "no per-word confidence". Dividing that sentinel by 100 would
        // hand the caller -0.01, which reads like a real timestamp; keep it
        // recognisable instead.
        internal static double Seconds(long centiseconds)
            => centiseconds < 0 ? -1.0 : centiseconds / CentisecondsPerSecond;

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
        /// <remarks>
        /// The model is loaded once and stays resident for the life of the
        /// <see cref="Session"/>. To transcribe many clips WITHOUT reloading the
        /// model (issue #291), open one session and call
        /// <see cref="Transcribe(float[])"/> repeatedly:
        /// <code>
        /// using var asr = Session.Open("parakeet.gguf");
        /// foreach (var clip in clips)
        ///     Process(asr.Transcribe(clip));   // model loaded once, reused
        /// </code>
        /// A <see cref="Session"/> is not thread-safe; use one per thread, or
        /// serialize calls. Standalone <see cref="VadSegments"/> is the exception —
        /// it loads and frees its VAD model on every call, so a hot VAD-only loop
        /// still reloads. Prefer session-integrated VAD (the vad transcribe path)
        /// when you also transcribe, so the one session covers both.
        /// </remarks>
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

        /// <summary>
        /// The acoustic language Whisper detected on the last transcribe, as an
        /// ISO-639-1 code ("en"). Whisper-only; other backends return the
        /// session's source-language hint, or "unknown".
        /// </summary>
        public string DetectedLanguage()
        {
            var buf = new byte[32];
            NativeMethods.crispasr_session_detected_language(Handle, buf, buf.Length);
            return NativeMethods.NullTerminated(buf);
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

        /// <summary>#316: synthesize these phonemes verbatim, skipping the G2P. Empty clears. kokoro and piper only (rc=-2).</summary>
        public void SetTtsPhonemes(string phonemes)
        {
            int rc = NativeMethods.crispasr_session_set_tts_phonemes(Handle, phonemes ?? "");
            if (rc == -2) throw new InvalidOperationException("Backend has no phonemes-in entry point (kokoro and piper do)");
            if (rc != 0) throw new InvalidOperationException($"set_tts_phonemes failed (rc={rc})");
        }

        /// <summary>Pad N ms of silence at the beginning of TTS output. Useful to bypass VLC playback bugs where it drops the first ~1.5s of audio while parsing a large C2PA chunk.</summary>
        public void SetTtsPadSilenceMs(int ms)
        {
            NativeMethods.crispasr_session_set_tts_pad_silence_ms(Handle, ms);
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

        /// <summary>
        /// Apply a named bundle of the four decoder fallback thresholds:
        /// "conservative", "balanced" (the shipped defaults, a no-op) or
        /// "aggressive". "strict"/"default"/"loose" are aliases. Mirrors the
        /// CLI's --sensitivity.
        /// </summary>
        /// <exception cref="ArgumentException">The preset is unrecognised.</exception>
        public void SetSensitivity(string preset)
        {
            // Deliberately NOT routed through Check(): that helper treats
            // rc == -2 as success, because for most setters -2 means "this
            // backend does not support the knob". Here -2 means "unknown
            // preset", and swallowing it would silently decode at the default
            // thresholds after a typo — exactly what this API exists to prevent.
            int rc = NativeMethods.crispasr_session_set_sensitivity(Handle, preset);
            if (rc == -2)
                throw new ArgumentException(
                    $"unknown sensitivity preset '{preset}' (expected: conservative, balanced, aggressive)",
                    nameof(preset));
            if (rc != 0)
                throw new InvalidOperationException($"set_sensitivity failed (rc={rc})");
        }

        /// <summary>Select the G2P pronunciation dictionary for TTS.</summary>
        public void SetG2pDict(string source)
            => Check(NativeMethods.crispasr_session_set_g2p_dict(Handle, source), "set_g2p_dict");

        /// <summary>Sticky source-language hint.</summary>
        public void SetSourceLanguage(string? lang)
            => Check(NativeMethods.crispasr_session_set_source_language(Handle, lang ?? ""), "set_source_language");

        /// <summary>Sticky target-language. Different from source triggers translation on canary/cohere.</summary>
        public void SetTargetLanguage(string? lang)
            => Check(NativeMethods.crispasr_session_set_target_language(Handle, lang ?? ""), "set_target_language");

        /// <summary>Language a voice-cloning reference clip is spoken in (#329). Cross-lingual TTS
        /// backends (cosyvoice3) drop the reference transcript when it differs from the requested
        /// output language, so the clone speaks that language instead of carrying the reference's
        /// accent. Optional — inferred from the voice bank or reference transcript otherwise, and
        /// that inference declines rather than guesses on a short transcript.</summary>
        public void SetTtsReferenceLanguage(string? lang)
            => Check(NativeMethods.crispasr_session_set_tts_reference_language(Handle, lang ?? ""),
                     "set_tts_reference_language");

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

        /// <summary>Floor on generated audio length (MOSS TTS): codec frames at 12.5 Hz (80 ms each); other backends no-op.</summary>
        public void SetMinSpeechTokens(int n)
            => Check(NativeMethods.crispasr_session_set_min_speech_tokens(Handle, n), "set_min_speech_tokens");

        /// <summary>Per-phoneme length-scale / speaking-rate scalar (kokoro).</summary>
        public void SetLengthScale(float scale)
            => Check(NativeMethods.crispasr_session_set_length_scale(Handle, scale), "set_length_scale");

        /// <summary>Best-of-N sampling count for ASR backends.</summary>
        public void SetBestOf(int n)
            => Check(NativeMethods.crispasr_session_set_best_of(Handle, n), "set_best_of");

        /// <summary>Beam-search width for ASR backends that support it.</summary>
        public void SetBeamSize(int n)
            => Check(NativeMethods.crispasr_session_set_beam_size(Handle, n), "set_beam_size");

        /// <summary>
        /// Opt in to capturing the per-frame CTC logits (backends with a dense
        /// CTC grid: Omni CTC, wav2vec2/hubert/data2vec, canary-ctc) so a
        /// following transcribe attaches the dense grid read back
        /// via <see cref="TranscribeWithLogits"/>. Off by default so the normal
        /// path pays no <c>[vocab × frames]</c> copy.
        /// </summary>
        public void SetReturnLogits(bool enable)
            => Check(NativeMethods.crispasr_session_set_return_logits(Handle, enable ? 1 : 0), "set_return_logits");

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
        /// Synthesize text to mono float32 PCM at OutputSampleRate().
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

        /// <summary>
        /// Attest acceptance of AI-content marking/disclosure responsibility (EU
        /// AI Act Art. 50). REQUIRED before <see cref="SynthesizeRaw"/> will return
        /// unmarked audio; the default <see cref="Synthesize"/> is watermarked and
        /// needs no attestation. <paramref name="attestation"/> is recorded for audit.
        /// </summary>
        public void AcceptMarkingResponsibility(string attestation = "")
            => Check(NativeMethods.crispasr_session_accept_marking_responsibility(Handle, attestation ?? ""),
                     "accept_marking_responsibility");

        /// <summary>
        /// Declare whose voice a PRESET voice is: <c>real_person</c>,
        /// <c>synthetic</c> or <c>unknown</c>.
        /// <para>
        /// Cloning is not the only way to produce a deep fake: a preset voice
        /// shipped inside a model can be an identifiable individual — a named
        /// donor, or a corpus speaker such as VCTK's <c>p225</c> — and EU AI Act
        /// Art. 3(60) attaches to the audio resembling that person, not to which
        /// pipeline produced it. Setting <c>real_person</c> makes the Art. 50(4)
        /// reminder fire for a non-cloned voice.
        /// </para>
        /// <para>
        /// It does <b>not</b> require a consent attestation: whether that donor
        /// agreed to the model being trained is a licensing matter settled
        /// upstream that you cannot attest to.
        /// </para>
        /// </summary>
        /// <exception cref="ArgumentException">
        /// Thrown on an unrecognised value, rather than silently downgrading it
        /// to <c>unknown</c>.
        /// </exception>
        public void SetSpeakerIdentity(string identity)
        {
            int rc = NativeMethods.crispasr_session_set_speaker_identity(Handle, identity ?? "");
            if (rc == -2)
                throw new ArgumentException(
                    $"unrecognised speaker_identity '{identity}' " +
                    "(expected real_person, synthetic or unknown)", nameof(identity));
            Check(rc, "set_speaker_identity");
        }

        /// <summary>
        /// UNMARKED synthesis (no watermark), for callers that post-process before
        /// embedding the mark themselves. Hard-refused (throws) unless
        /// <see cref="AcceptMarkingResponsibility"/> was called first. Prefer
        /// <see cref="Synthesize"/> for the default watermarked output.
        /// </summary>
        public float[] SynthesizeRaw(string text)
        {
            var ptr = NativeMethods.crispasr_session_synthesize_raw(Handle, text, out int nSamples);
            if (ptr == IntPtr.Zero || nSamples <= 0)
                throw new InvalidOperationException(
                    "SynthesizeRaw returned no audio (attestation required? call AcceptMarkingResponsibility first)");
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

        /// <summary>
        /// Embed the AI-content watermark into mono float32 PCM, in place.
        /// <para>
        /// The other half of <see cref="SynthesizeRaw"/>: opting out of automatic
        /// marking makes marking the result your duty under EU AI Act Art. 50(2),
        /// and this is what discharges it. Do the post-processing you opted out
        /// for — resample, mix, concatenate — then mark the finished buffer.
        /// </para>
        /// <para>
        /// Uses the robust, reliably detectable default strength; AudioSeal
        /// instead when a model has been loaded. Static because marking is a
        /// property of the samples, not of the session that produced them.
        /// </para>
        /// </summary>
        public static void WatermarkEmbed(float[] pcm)
        {
            if (pcm == null || pcm.Length == 0)
                return;
            NativeMethods.crispasr_watermark_embed(pcm, pcm.Length, -1.0f);
        }

        /// <summary>
        /// Confidence in [0, 1] that <paramref name="pcm"/> carries the watermark.
        /// A weak diagnostic, not proof: the spread-spectrum detector's null mean
        /// is 0.5, not 0, and a negative result on a short clip is mostly evidence
        /// that the clip was short. See <c>docs/eu-ai-act.md</c> §6.7.
        /// </summary>
        public static float WatermarkDetect(float[] pcm)
        {
            if (pcm == null || pcm.Length == 0)
                return 0.0f;
            return NativeMethods.crispasr_watermark_detect(pcm, pcm.Length);
        }

        /// <summary>
        /// Speech-to-speech: audio in → audio out through a single model pass,
        /// on backends with S2S capability (lfm2-audio, mini-omni2, sidon,
        /// voxcpm2-vae). Input is 16 kHz mono float32 PCM. Returns the output
        /// PCM plus the intermediate ASR transcript (may be <c>null</c>).
        /// </summary>
        public (float[] pcm, string? transcript) SpeechToSpeech(float[] input)
        {
            var ptr = NativeMethods.crispasr_session_speech_to_speech(
                Handle, input, input.Length, out IntPtr textPtr, out int nSamples);
            if (ptr == IntPtr.Zero || nSamples <= 0)
                throw new InvalidOperationException(
                    "SpeechToSpeech returned no audio (backend may not support S2S)");
            try
            {
                var pcm = new float[nSamples];
                Marshal.Copy(ptr, pcm, 0, nSamples);
                string? transcript = null;
                if (textPtr != IntPtr.Zero)
                {
                    transcript = Marshal.PtrToStringUTF8(textPtr);
                    NativeMethods.crispasr_session_translate_text_free(textPtr);
                }
                return (pcm, transcript);
            }
            finally
            {
                NativeMethods.crispasr_pcm_free(ptr);
            }
        }

        /// <summary>
        /// The sample rate the backend expects for input PCM (16000 for
        /// Whisper-family backends, 0 on error).
        /// </summary>
        public int InputSampleRate() => NativeMethods.crispasr_session_input_sample_rate(Handle);

        /// <summary>
        /// Sample rate of the PCM Synthesize/SpeechToSpeech produce for this
        /// backend; 0 when the backend has no audio output (ASR-only). (#332)
        /// </summary>
        public int OutputSampleRate() => NativeMethods.crispasr_session_output_sample_rate(Handle);

        /// <summary>
        /// Channel count for audio input: 1 (mono) for every current backend,
        /// 0 on error. Source separation is the stereo exception. (#332)
        /// </summary>
        public int InputChannels() => NativeMethods.crispasr_session_input_channels(Handle);

        /// <summary>
        /// Channel count for synthesized / s2s output audio: 1 (mono), or 0
        /// when the backend has no audio output. (#332)
        /// </summary>
        public int OutputChannels() => NativeMethods.crispasr_session_output_channels(Handle);

        // ----------------------------------------------------------------
        // ASR Transcription
        // ----------------------------------------------------------------

        /// <summary>Transcribe 16 kHz mono float32 PCM.</summary>
        /// <remarks>
        /// Long audio is auto-chunked natively (30 s windows), so a full
        /// recording can be passed in one call.
        /// </remarks>
        public Segment[] Transcribe(float[] pcm)
            => TranscribeLang(pcm, null);

        /// <summary>
        /// Decode an audio file and transcribe it — the binding equivalent of
        /// <c>crispasr -m model.gguf -f audio.wav</c> (issue #291). Handles any
        /// format and sample rate the CLI handles; see <see cref="Audio.Load(string)"/>.
        /// </summary>
        /// <example>
        /// <code>
        /// using var s = Session.Open("moonshine-base-de-fidoriel-q4_k.gguf");
        /// foreach (var seg in s.TranscribeFile("speech.wav"))
        ///     Console.WriteLine($"[{seg.T0:F2}-{seg.T1:F2}] {seg.Text}");
        /// </code>
        /// </example>
        public Segment[] TranscribeFile(string path, string? language = null)
            => TranscribeLang(Audio.Load(path), language);

        /// <summary>Transcribe with explicit language hint.</summary>
        public Segment[] TranscribeLang(float[] pcm, string? language)
        {
            var r = NativeMethods.crispasr_session_transcribe_lang(Handle, pcm, pcm.Length, language);
            if (r == IntPtr.Zero) throw new InvalidOperationException("Transcription failed");
            try { return ExtractSegments(r); }
            finally { NativeMethods.crispasr_session_result_free(r); }
        }

        /// <summary>
        /// Chunked-encode transcribe (issue #208): forces the Parakeet backend
        /// through its bounded overlapping-window long-form path so long audio
        /// transcribes in bounded time without dropping sections. Inert
        /// (== <see cref="TranscribeLang"/>) on non-Parakeet backends.
        /// <paramref name="chunkSeconds"/> &lt;= 0 keeps the per-model default
        /// window; <paramref name="overlapSeconds"/> &lt; 0 keeps the default overlap.
        /// </summary>
        public Segment[] TranscribeChunked(float[] pcm, int chunkSeconds = 0, int overlapSeconds = -1,
                                           string? language = null)
        {
            var r = NativeMethods.crispasr_session_transcribe_chunked_lang(
                Handle, pcm, pcm.Length, chunkSeconds, overlapSeconds, language);
            if (r == IntPtr.Zero) throw new InvalidOperationException("Chunked transcription failed");
            try { return ExtractSegments(r); }
            finally { NativeMethods.crispasr_session_result_free(r); }
        }

        /// <summary>
        /// Transcribe and also return the per-frame CTC logits captured for
        /// this call (backends with a dense CTC grid: Omni CTC, wav2vec2/hubert/
        /// data2vec, canary-ctc). Opts in for the duration of the call — no need
        /// to call <see cref="SetReturnLogits"/> first — then returns the
        /// segments plus a <see cref="CtcLogits"/> grid, or <c>null</c> for
        /// backends that produce no dense CTC grid.
        /// </summary>
        public (Segment[] Segments, CtcLogits? Logits) TranscribeWithLogits(float[] pcm, string? language = null)
        {
            SetReturnLogits(true);
            var r = NativeMethods.crispasr_session_transcribe_lang(Handle, pcm, pcm.Length, language);
            if (r == IntPtr.Zero) { SetReturnLogits(false); throw new InvalidOperationException("Transcription failed"); }
            try { return (ExtractSegments(r), ExtractLogits(r)); }
            finally { SetReturnLogits(false); NativeMethods.crispasr_session_result_free(r); }
        }

        // Lift the result-owned float* logit grid into a managed float[] before
        // the result handle is freed (same copy-out idiom as Synthesize).
        private static CtcLogits? ExtractLogits(IntPtr r)
        {
            int nFrames = NativeMethods.crispasr_session_result_n_logit_frames(r);
            int nVocab = NativeMethods.crispasr_session_result_n_logit_vocab(r);
            var ptr = NativeMethods.crispasr_session_result_logits(r);
            if (nFrames <= 0 || nVocab <= 0 || ptr == IntPtr.Zero) return null;
            var data = new float[nFrames * nVocab];
            Marshal.Copy(ptr, data, 0, nFrames * nVocab);
            return new CtcLogits(nVocab, nFrames, data);
        }

        /// <summary>
        /// The Omni CTC vocabulary as raw pieces, indexed by token id
        /// (<c>vocab[id]</c>). Pieces keep their word-boundary marker intact
        /// (the v2 Omni vocab uses a literal space, v1 uses U+2581), so a
        /// consumer can detokenize a greedy CTC decode over the grid from
        /// <see cref="TranscribeWithLogits"/>. Returns <c>null</c> for backends
        /// that don't expose a CTC vocab.
        /// </summary>
        public string[]? CtcVocab()
        {
            int n = NativeMethods.crispasr_session_n_vocab(Handle);
            if (n <= 0) return null;
            var vocab = new string[n];
            for (int i = 0; i < n; i++)
                vocab[i] = NativeMethods.PtrToUtf8(NativeMethods.crispasr_session_token_text(Handle, i)) ?? "";
            return vocab;
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
                // The C ABI reports centiseconds; every public time value in this
                // binding is seconds (issue #291), so convert once, here.
                double t0 = Seconds(NativeMethods.crispasr_session_result_segment_t0(r, i));
                double t1 = Seconds(NativeMethods.crispasr_session_result_segment_t1(r, i));

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
                        Seconds(NativeMethods.crispasr_session_result_word_t0(r, i, j)),
                        Seconds(NativeMethods.crispasr_session_result_word_t1(r, i, j)),
                        NativeMethods.crispasr_session_result_word_p(r, i, j),
                        alts);
                }
                float noSpeechProb = NativeMethods.crispasr_session_result_segment_no_speech_prob(r, i);
                string speaker = NativeMethods.PtrToUtf8(NativeMethods.crispasr_session_result_segment_speaker(r, i)) ?? "";
                segs[i] = new Segment(text, t0, t1, words, noSpeechProb, speaker);
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
        /// Streaming mode: 0 unavailable, 1 rolling windows, 2 persistent
        /// model caches, 3 text-prefix streaming.
        /// </summary>
        public int StreamKind() => NativeMethods.crispasr_session_stream_kind(Handle);

        /// <summary>Open a streaming decoder for this session's backend.</summary>
        public StreamDecoder StreamOpen(int stepMs = 3000, int lengthMs = 10000, int keepMs = 200,
                                        string? language = null, bool translate = false, int nThreads = 4)
        {
            var p = NativeMethods.crispasr_session_stream_open(
                Handle, nThreads, stepMs, lengthMs, keepMs, language ?? "", translate ? 1 : 0);
            if (p == IntPtr.Zero)
                throw new InvalidOperationException("stream_open failed");
            return new StreamDecoder(p);
        }

        // ----------------------------------------------------------------
        // Forced alignment (static)
        // ----------------------------------------------------------------

        /// <summary>Run CTC forced alignment on transcript + audio.</summary>
        public static AlignedWord[] AlignWords(string alignerModel, string transcript,
                                                float[] pcm, double tOffsetSeconds = 0.0, int nThreads = 4)
        {
            long tOffsetCs = (long)Math.Round(tOffsetSeconds * CentisecondsPerSecond);
            var r = NativeMethods.crispasr_align_words_abi(alignerModel, transcript, pcm, pcm.Length, tOffsetCs, nThreads);
            if (r == IntPtr.Zero) throw new InvalidOperationException("Alignment failed");
            try
            {
                int n = NativeMethods.crispasr_align_result_n_words(r);
                var words = new AlignedWord[n];
                for (int i = 0; i < n; i++)
                {
                    int nc = NativeMethods.crispasr_align_result_n_characters(r, i);
                    var characters = new AlignedCharacter[nc];
                    for (int j = 0; j < nc; j++)
                        characters[j] = new AlignedCharacter(
                            NativeMethods.PtrToUtf8(NativeMethods.crispasr_align_result_character_text(r, i, j)) ?? "",
                            Seconds(NativeMethods.crispasr_align_result_character_t0(r, i, j)),
                            Seconds(NativeMethods.crispasr_align_result_character_t1(r, i, j)));
                    words[i] = new AlignedWord(
                        NativeMethods.PtrToUtf8(NativeMethods.crispasr_align_result_word_text(r, i)) ?? "",
                        Seconds(NativeMethods.crispasr_align_result_word_t0(r, i)),
                        Seconds(NativeMethods.crispasr_align_result_word_t1(r, i)), characters);
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
                // The native crispasr_vad_segments ABI returns CENTISECONDS
                // (start_cs, end_cs — see crispasr.h), the raw whisper.cpp VAD
                // unit. Every other time value in this binding (Segment/Word T0/T1)
                // is seconds, and this method's own doc-comment promises seconds,
                // so convert here. Reported as issue #291: without this the spans
                // came back as ms/10 and silently disagreed with Session times.
                for (int i = 0; i < n; i++)
                    spans[i] = new VadSpan(raw[i * 2] / 100.0, raw[i * 2 + 1] / 100.0);
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
            // rc > 0 = detected (strlen of name); rc == 0 = valid GGUF but no backend
            // mapping; rc < 0 = error. The prior `rc != 0` returned null on every
            // successful detection.
            if (rc <= 0) return null;
            var name = NativeMethods.NullTerminated(buf);
            return string.IsNullOrEmpty(name) ? null : name;
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
        /// <summary>Word start, in SECONDS, or -1 when the backend produced no
        /// word timing (moonshine, and any other token-only decoder). (Issue #291:
        /// was raw centiseconds through v0.8.29 while the binding documented
        /// seconds everywhere else.)</summary>
        public double T0 { get; }
        /// <summary>Word end, in SECONDS, or -1 when the backend produced no word timing.</summary>
        public double T1 { get; }
        public float P { get; }
        public AltToken[] Alts { get; }

        public Word(string text, double t0, double t1, float p, AltToken[]? alts = null)
        {
            Text = text; T0 = t0; T1 = t1; P = p;
            Alts = alts ?? Array.Empty<AltToken>();
        }

        public override string ToString() => FormattableString.Invariant($"{T0:F2}-{T1:F2} {Text}");
    }

    /// <summary>Alternative token candidate.</summary>
    public readonly struct AltToken
    {
        public string Text { get; }
        public float P { get; }

        public AltToken(string text, float p) { Text = text; P = p; }

        public override string ToString() => FormattableString.Invariant($"{Text}({P * 100:F1}%)");
    }

    /// <summary>One segment from a transcription result.</summary>
    public readonly struct Segment
    {
        public string Text { get; }
        /// <summary>Segment start, in SECONDS, or -1 when the backend produced no
        /// timing. (Issue #291: was raw centiseconds through v0.8.29 while the
        /// binding documented seconds everywhere else.)</summary>
        public double T0 { get; }
        /// <summary>Segment end, in SECONDS, or -1 when the backend produced no timing.</summary>
        public double T1 { get; }
        public Word[] Words { get; }
        /// <summary>Whisper's per-segment no-speech probability (the &lt;|nospeech|&gt;
        /// posterior) in [0, 1]. Whisper-only; other backends leave -1.0 ("no data").</summary>
        public float NoSpeechProb { get; }
        /// <summary>Native per-segment speaker label from a backend that diarizes on
        /// its own, in the "(Speaker N) " form the CLI prefixes into text/srt/vtt
        /// output, or "" when the backend produced none. Populated today by vibevoice.
        /// The ordinals are CHUNK-LOCAL: "Speaker 1" from one transcribe call is not
        /// necessarily the same voice as "Speaker 1" from the next.</summary>
        public string Speaker { get; }

        public Segment(string text, double t0, double t1, Word[] words, float noSpeechProb = -1.0f,
                       string speaker = "")
        {
            Text = text; T0 = t0; T1 = t1; Words = words; NoSpeechProb = noSpeechProb; Speaker = speaker ?? "";
        }

        public override string ToString() => FormattableString.Invariant($"[{T0:F2}-{T1:F2}] {Text}");
    }

    /// <summary>
    /// Per-frame CTC logits from a CTC backend (Omni CTC, wav2vec2/hubert/
    /// data2vec, or canary-ctc), captured by
    /// <see cref="Session.TranscribeWithLogits"/>. <see cref="Data"/> is
    /// frame-major: <c>Data[t * NVocab + v]</c> is the score for vocabulary
    /// entry <c>v</c> at encoder frame <c>t</c>, so its length is
    /// <c>NFrames * NVocab</c>. The Omni and wav2vec2 grids are raw logits
    /// (pre-softmax); the canary-ctc grid is log-probabilities.
    /// </summary>
    public readonly struct CtcLogits
    {
        /// <summary>Vocabulary size — the number of CTC output classes scored per frame.</summary>
        public int NVocab { get; }

        /// <summary>Number of encoder frames (the time axis).</summary>
        public int NFrames { get; }

        /// <summary>Frame-major CTC grid of length <c>NFrames * NVocab</c> (raw logits for Omni &amp; wav2vec2, log-probabilities for canary-ctc).</summary>
        public float[] Data { get; }

        public CtcLogits(int nVocab, int nFrames, float[] data)
        {
            NVocab = nVocab; NFrames = nFrames; Data = data;
        }
    }

    /// <summary>A measured original codepoint; times are seconds.</summary>
    public readonly struct AlignedCharacter
    {
        public string Text { get; }
        public double T0 { get; }
        public double T1 { get; }
        public AlignedCharacter(string text, double t0, double t1) { Text = text; T0 = t0; T1 = t1; }
    }

    /// <summary>One aligned word from forced alignment.</summary>
    public readonly struct AlignedWord
    {
        public string Text { get; }
        /// <summary>Word start, in SECONDS (issue #291 — was raw centiseconds).</summary>
        public double T0 { get; }
        /// <summary>Word end, in SECONDS.</summary>
        public double T1 { get; }

        public AlignedCharacter[] Characters { get; }
        public AlignedWord(string text, double t0, double t1, AlignedCharacter[]? characters = null)
        { Text = text; T0 = t0; T1 = t1; Characters = characters ?? Array.Empty<AlignedCharacter>(); }
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

        public override string ToString() => FormattableString.Invariant($"LanguageDetection({Code}, {Probability * 100:F1}%)");
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

    /// <summary>
    /// On-disk speaker embedding database (closed-roster, consent-gated —
    /// issue #266). Matching is a claimed-participant confirmation, never
    /// an open 1:N search.
    /// </summary>
    public sealed class SpeakerDb : IDisposable
    {
        private IntPtr _handle;

        private SpeakerDb(IntPtr handle) => _handle = handle;

        /// <summary>
        /// Open a db narrowed to the claimed roster. <paramref name="expectedNames"/>
        /// is the comma-separated list of enrolled participants asserted present
        /// (e.g. "Alice,Bob"). <paramref name="consentAttested"/> affirms a lawful
        /// basis + explicit consent from every enrolled person (GDPR Art. 9);
        /// the call refuses without both.
        /// </summary>
        public static SpeakerDb Open(string dirPath, string expectedNames, bool consentAttested)
        {
            if (!consentAttested)
                throw new InvalidOperationException("SpeakerDb requires an explicit consent attestation (GDPR Art. 9)");
            var p = NativeMethods.crispasr_speaker_db_open(dirPath, expectedNames, 1);
            if (p == IntPtr.Zero) throw new InvalidOperationException($"Failed to open speaker db from {dirPath}");
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

        /// <summary>
        /// Enroll a new speaker embedding. <paramref name="consentAttested"/> records
        /// the enrolled person's explicit consent (GDPR Art. 9) in the profile;
        /// enrollment refuses without it.
        /// </summary>
        public static void Enroll(string dirPath, string name, float[] embedding, bool consentAttested)
        {
            if (!consentAttested)
                throw new InvalidOperationException("Enrollment requires an explicit consent attestation (GDPR Art. 9)");
            int rc = NativeMethods.crispasr_speaker_db_enroll2(dirPath, name, embedding, embedding.Length, 1);
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
