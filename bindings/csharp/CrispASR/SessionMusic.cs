using System;
using System.Runtime.InteropServices;

namespace CrispASR
{
    // Music / audio task backends: guitar tablature, beat tracking, chord
    // recognition, piano transcription, pitch (F0), source separation, and RVC
    // voice conversion. These were exposed in the C ABI but never bound in C#.
    //
    // UNITS. Every time value here is SECONDS, matching Segment/Word/VadSpan.
    // Several of these native calls report milliseconds (chords, piano, pitch)
    // or seconds (beats); the wrappers normalise to seconds so a C# caller never
    // has to remember which backend used which unit — the exact trap #291 was.

    /// <summary>One beat, from <see cref="Session.Beats"/>.</summary>
    public readonly struct BeatEvent
    {
        public double TimeSeconds { get; }
        /// <summary>True if this beat is also a downbeat (bar start).</summary>
        public bool IsDownbeat { get; }
        public BeatEvent(double timeSeconds, bool isDownbeat) { TimeSeconds = timeSeconds; IsDownbeat = isDownbeat; }
    }

    /// <summary>Beat-tracking result: the grid plus an estimated tempo.</summary>
    public readonly struct BeatResult
    {
        public BeatEvent[] Beats { get; }
        /// <summary>Median-interval tempo in BPM, or 0 with fewer than two beats.</summary>
        public double TempoBpm { get; }
        public BeatResult(BeatEvent[] beats, double tempoBpm) { Beats = beats; TempoBpm = tempoBpm; }
    }

    /// <summary>One chord span, from <see cref="Session.Chords"/>.</summary>
    public readonly struct ChordSpan
    {
        public double StartSeconds { get; }
        public double EndSeconds { get; }
        /// <summary>Vocabulary index; resolve to text via <see cref="Name"/>.</summary>
        public int Label { get; }
        public float Confidence { get; }
        /// <summary>Chord name, e.g. "C", "Am", "G:7", or "N" for no-chord.</summary>
        public string Name { get; }
        public ChordSpan(double startSeconds, double endSeconds, int label, float confidence, string name)
        { StartSeconds = startSeconds; EndSeconds = endSeconds; Label = label; Confidence = confidence; Name = name; }
    }

    /// <summary>One piano note, from <see cref="Session.Piano"/>.</summary>
    public readonly struct PianoNote
    {
        public double OnsetSeconds { get; }
        public double OffsetSeconds { get; }
        /// <summary>MIDI note 21-108 (A0-C8).</summary>
        public int MidiNote { get; }
        /// <summary>MIDI velocity 0-127.</summary>
        public int Velocity { get; }
        /// <summary>
        /// General MIDI program 0-127 — which instrument played the note;
        /// 128 for percussion (GM channel 10, which carries no meaningful
        /// program); -1 when the model does not identify an instrument.
        /// </summary>
        /// <remarks>
        /// Only MT3 fills this in. piano-transcription and basic-pitch report
        /// -1 throughout — deliberately -1 and not 0, because 0 is "Acoustic
        /// Grand Piano" and would be indistinguishable from a real answer. It
        /// is also -1 against a native library that predates
        /// crispasr_session_piano_note_programs, so callers can read the
        /// sentinel rather than probing for the export.
        /// </remarks>
        public int Program { get; }
        public PianoNote(double onsetSeconds, double offsetSeconds, int midiNote, int velocity,
            int program = -1)
        { OnsetSeconds = onsetSeconds; OffsetSeconds = offsetSeconds; MidiNote = midiNote; Velocity = velocity; Program = program; }
    }

    /// <summary>One pitch frame, from <see cref="Session.Pitch"/>.</summary>
    public readonly struct PitchFrame
    {
        public double TimeSeconds { get; }
        public float F0Hz { get; }
        /// <summary>Voicing probability 0-1; F0 is meaningful only when this is high.</summary>
        public float VoicedProb { get; }
        public PitchFrame(double timeSeconds, float f0Hz, float voicedProb)
        { TimeSeconds = timeSeconds; F0Hz = f0Hz; VoicedProb = voicedProb; }
    }

    /// <summary>One separated stem, from <see cref="Session.Separate"/>.</summary>
    public readonly struct Stem
    {
        public string Name { get; }
        /// <summary>Interleaved stereo PCM (L,R,L,R,...). Length is 2 * <see cref="PerChannelSamples"/>.</summary>
        public float[] Interleaved { get; }
        public int PerChannelSamples { get; }
        public Stem(string name, float[] interleaved, int perChannelSamples)
        { Name = name; Interleaved = interleaved; PerChannelSamples = perChannelSamples; }
    }

    /// <summary>
    /// Guitar-tablature emission scores, from <see cref="Session.Tab"/>.
    /// <see cref="LogProbs"/> is a flat [frame][string][class] grid of
    /// log-probabilities; index it as <c>LogProbs[(frame * Strings + str) * Classes + cls]</c>.
    /// </summary>
    public readonly struct TabEmissions
    {
        public int Frames { get; }
        public int Strings { get; }
        public int Classes { get; }
        public float[] LogProbs { get; }
        /// <summary>Class index meaning "string not played". Read it, never assume it.</summary>
        public int SilentClass { get; }
        /// <summary>Seconds between consecutive frames.</summary>
        public float FramePeriodSeconds { get; }
        /// <summary>Open-string MIDI pitch per string (index 0 = lowest).</summary>
        public int[] StringOpenMidi { get; }
        public TabEmissions(int frames, int strings, int classes, float[] logProbs,
                            int silentClass, float framePeriodSeconds, int[] stringOpenMidi)
        { Frames = frames; Strings = strings; Classes = classes; LogProbs = logProbs;
          SilentClass = silentClass; FramePeriodSeconds = framePeriodSeconds; StringOpenMidi = stringOpenMidi; }
    }

    public sealed partial class Session
    {
        // Copy a session-owned float* view into a managed array. The pointer is
        // valid only until the next run call, so this must happen immediately.
        private static float[] CopyView(IntPtr ptr, int count)
        {
            if (ptr == IntPtr.Zero || count <= 0) return Array.Empty<float>();
            var buf = new float[count];
            Marshal.Copy(ptr, buf, 0, count);
            return buf;
        }

        /// <summary>
        /// Guitar tablature: per-frame, per-string emission scores. Feed mono PCM;
        /// the backend resamples to its own rate. The scores are emission-only —
        /// a constrained decoder on your side picks the playable fingering.
        /// </summary>
        public TabEmissions Tab(float[] pcm, int sampleRate)
        {
            int n = NativeMethods.crispasr_session_tab(Handle, pcm, pcm.Length, sampleRate);
            if (n < 0) throw new InvalidOperationException($"tab failed (rc={n})");
            IntPtr p = NativeMethods.crispasr_session_tab_emissions(Handle, out int nf, out int nStr, out int nCls);
            var logits = CopyView(p, nf * nStr * nCls);
            var open = new int[nStr];
            for (int i = 0; i < nStr; i++)
                open[i] = NativeMethods.crispasr_session_tab_string_open_midi(Handle, i);
            return new TabEmissions(nf, nStr, nCls, logits,
                NativeMethods.crispasr_session_tab_silent_class(Handle),
                NativeMethods.crispasr_session_tab_frame_period(Handle), open);
        }

        /// <summary>Native input rate the loaded beat model expects, or 0 if the backend has no beat arm.</summary>
        public int BeatsSampleRate => NativeMethods.crispasr_session_beats_sample_rate(Handle);

        /// <summary>
        /// Beat and downbeat tracking. <paramref name="sampleRate"/> must equal
        /// <see cref="BeatsSampleRate"/> — the backend rejects a mismatch rather
        /// than resampling, because resampling would move every beat time.
        /// </summary>
        public BeatResult Beats(float[] pcm, int sampleRate)
        {
            int n = NativeMethods.crispasr_session_beats(Handle, pcm, pcm.Length, sampleRate);
            if (n < 0) throw new InvalidOperationException($"beats failed (rc={n})");
            IntPtr p = NativeMethods.crispasr_session_beats_events(Handle, out int ne);
            var raw = CopyView(p, ne * 2);          // {time_s, is_downbeat}
            var beats = new BeatEvent[ne];
            for (int i = 0; i < ne; i++)
                beats[i] = new BeatEvent(raw[i * 2], raw[i * 2 + 1] != 0.0f);
            return new BeatResult(beats, NativeMethods.crispasr_session_beats_tempo_bpm(Handle));
        }

        /// <summary>Chord-vocabulary size: 25 (maj/min + N) or 170 (full quality set).</summary>
        public int ChordsVocabSize => NativeMethods.crispasr_session_chords_vocab_size(Handle);

        /// <summary>Chord recognition: labelled time spans over the audio.</summary>
        public ChordSpan[] Chords(float[] pcm, int sampleRate)
        {
            int n = NativeMethods.crispasr_session_chords(Handle, pcm, pcm.Length, sampleRate);
            if (n < 0) throw new InvalidOperationException($"chords failed (rc={n})");
            IntPtr p = NativeMethods.crispasr_session_chords_spans(Handle, out int ns);
            var raw = CopyView(p, ns * 4);          // {start_ms, end_ms, label, confidence}
            var spans = new ChordSpan[ns];
            for (int i = 0; i < ns; i++)
            {
                int label = (int)raw[i * 4 + 2];
                IntPtr namePtr = NativeMethods.crispasr_session_chords_span_name(Handle, i);
                string name = namePtr == IntPtr.Zero ? "" : (Marshal.PtrToStringUTF8(namePtr) ?? "");
                spans[i] = new ChordSpan(raw[i * 4] / 1000.0, raw[i * 4 + 1] / 1000.0, label, raw[i * 4 + 3], name);
            }
            return spans;
        }

        /// <summary>Native input rate the loaded piano model expects (16000 for the current model).</summary>
        public int PianoSampleRate => NativeMethods.crispasr_session_piano_sample_rate(Handle);

        /// <summary>
        /// Piano transcription: onset/offset note events. Feed mono PCM at
        /// <see cref="PianoSampleRate"/> (16 kHz).
        /// </summary>
        public PianoNote[] Piano(float[] pcm16k)
        {
            int n = NativeMethods.crispasr_session_piano(Handle, pcm16k, pcm16k.Length);
            if (n < 0) throw new InvalidOperationException($"piano failed (rc={n})");
            IntPtr p = NativeMethods.crispasr_session_piano_notes(Handle, out int nn);
            var raw = CopyView(p, nn * 4);          // {onset_ms, offset_ms, midi_note, velocity}

            // Programs travel in a PARALLEL int array rather than as a fifth
            // float, because widening the note record would break every
            // existing reader of that layout. Absent (older native library,
            // or a model that identifies no instrument) leaves -1, so this
            // reads the sentinel instead of probing for the export.
            var programs = new int[nn];
            for (int i = 0; i < nn; i++) programs[i] = -1;
            IntPtr gp = NativeMethods.crispasr_session_piano_note_programs(Handle, out int gn);
            if (gp != IntPtr.Zero && gn >= nn)
                Marshal.Copy(gp, programs, 0, nn);

            var notes = new PianoNote[nn];
            for (int i = 0; i < nn; i++)
                notes[i] = new PianoNote(raw[i * 4] / 1000.0, raw[i * 4 + 1] / 1000.0,
                    (int)raw[i * 4 + 2], (int)raw[i * 4 + 3], programs[i]);
            return notes;
        }

        /// <summary>Native input rate the loaded pitch model expects (16000 for crepe).</summary>
        public int PitchSampleRate => NativeMethods.crispasr_session_pitch_sample_rate(Handle);

        /// <summary>
        /// Monophonic pitch (F0) track. Feed mono PCM at <see cref="PitchSampleRate"/>.
        /// <paramref name="hopMs"/> &lt;= 0 uses the model default (10 ms).
        /// </summary>
        public PitchFrame[] Pitch(float[] pcm16k, float hopMs = 0f)
        {
            int n = NativeMethods.crispasr_session_pitch(Handle, pcm16k, pcm16k.Length, hopMs);
            if (n < 0) throw new InvalidOperationException($"pitch failed (rc={n})");
            IntPtr p = NativeMethods.crispasr_session_pitch_frames(Handle, out int nf);
            var raw = CopyView(p, nf * 3);          // {time_ms, f0_hz, voiced_prob}
            var frames = new PitchFrame[nf];
            for (int i = 0; i < nf; i++)
                frames[i] = new PitchFrame(raw[i * 3] / 1000.0, raw[i * 3 + 1], raw[i * 3 + 2]);
            return frames;
        }

        /// <summary>Native output rate of the separation model (44100 for htdemucs).</summary>
        public int SeparateSampleRate => NativeMethods.crispasr_session_separate_sample_rate(Handle);

        /// <summary>
        /// Source separation into stems (e.g. drums, bass, other, vocals). Input is
        /// interleaved stereo PCM at the model's native rate (44.1 kHz for htdemucs).
        /// </summary>
        public Stem[] Separate(float[] pcmStereoInterleaved)
        {
            int n = NativeMethods.crispasr_session_separate(Handle, pcmStereoInterleaved, pcmStereoInterleaved.Length);
            if (n < 0) throw new InvalidOperationException($"separate failed (rc={n})");
            var stems = new Stem[n];
            for (int i = 0; i < n; i++)
            {
                IntPtr namePtr = NativeMethods.crispasr_session_separate_stem_name(Handle, i);
                string name = namePtr == IntPtr.Zero ? $"stem{i}" : (Marshal.PtrToStringUTF8(namePtr) ?? $"stem{i}");
                IntPtr pcmPtr = NativeMethods.crispasr_session_separate_stem(Handle, i, out int perChan);
                // per-channel count; the buffer is interleaved stereo → 2x floats.
                stems[i] = new Stem(name, CopyView(pcmPtr, perChan * 2), perChan);
            }
            return stems;
        }

        /// <summary>Content-embedding dimension the loaded RVC model expects: 256 (v1) or 768 (v2).</summary>
        public int ConvertContentDim => NativeMethods.crispasr_session_convert_content_dim(Handle);
        /// <summary>Number of target speakers in the loaded RVC model.</summary>
        public int ConvertSpeakerCount => NativeMethods.crispasr_session_convert_n_speakers(Handle);
        /// <summary>Native output rate of the RVC model (32k/40k/48k — not a constant).</summary>
        public int ConvertSampleRate => NativeMethods.crispasr_session_convert_sample_rate(Handle);

        /// <summary>
        /// RVC voice conversion. <paramref name="content"/> is a frame-major content
        /// embedding of shape [nFrames * <see cref="ConvertContentDim"/>] from your
        /// own content encoder; <paramref name="f0Hz"/> is one value per frame (0 = unvoiced).
        /// Output is mono PCM at <see cref="ConvertSampleRate"/>.
        ///
        /// This is stochastic by design — production callers leave the noise buffers
        /// null. Explicit noise only exists to replay a specific draw for A/B against
        /// another implementation; do not expect waveform-identical output otherwise.
        /// </summary>
        public float[] Convert(float[] content, float[] f0Hz, int speakerId = 0,
                               float[]? noiseZp = null, float[]? noiseSine = null)
        {
            int dim = ConvertContentDim;
            if (dim <= 0) throw new InvalidOperationException("loaded model has no voice-conversion arm");
            int nFrames = content.Length / dim;
            if (nFrames <= 0 || nFrames * dim != content.Length)
                throw new ArgumentException($"content length {content.Length} is not a multiple of content_dim {dim}", nameof(content));
            if (f0Hz.Length != nFrames)
                throw new ArgumentException($"f0Hz length {f0Hz.Length} must equal frame count {nFrames}", nameof(f0Hz));
            int n = NativeMethods.crispasr_session_convert(Handle, content, nFrames, f0Hz, speakerId, noiseZp, noiseSine);
            if (n < 0) throw new InvalidOperationException($"convert failed (rc={n})");
            IntPtr p = NativeMethods.crispasr_session_convert_audio(Handle, out int nSamples);
            return CopyView(p, nSamples);
        }
    }
}
