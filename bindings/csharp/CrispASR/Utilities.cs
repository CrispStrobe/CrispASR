using System;
using System.Runtime.InteropServices;

namespace CrispASR
{
    internal static partial class NativeMethods
    {
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_audio_load_stereo([MarshalAs(UnmanagedType.LPUTF8Str)] string path, out IntPtr left, out IntPtr right, out int n, out int rate, out int channels);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_audio_resample(float[] pcm, int n, int source, int target, out IntPtr output, out int count);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_c2pa_sign(byte[] data, UIntPtr size, [MarshalAs(UnmanagedType.LPUTF8Str)] string format, [MarshalAs(UnmanagedType.LPUTF8Str)] string? cert, [MarshalAs(UnmanagedType.LPUTF8Str)] string? key, out UIntPtr sizeOut);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_c_api_version();
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_dart_helpers_version();
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_set_gpu_backend([MarshalAs(UnmanagedType.LPUTF8Str)] string name);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_backend_caps_abi([MarshalAs(UnmanagedType.LPUTF8Str)] string backend, byte[] output, int cap);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_backend_caps_list_abi(byte[]? output, int cap);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_detect_backends_from_gguf([MarshalAs(UnmanagedType.LPUTF8Str)] string path, byte[] output, int cap);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_watermark_load_model([MarshalAs(UnmanagedType.LPUTF8Str)] string path);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_session_backend(IntPtr s);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_session_set_pcm_sample_rate(IntPtr s, int rate);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_session_set_parakeet_att_context(IntPtr s, int left, int right);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_session_set_audio_ctx(IntPtr s, int audioCtx);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_session_set_hikari_refresh_stride(IntPtr s, int stride);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_session_set_grammar_strict(IntPtr s, int enabled);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_free_aligner_cache();
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_session_score_texts(IntPtr session, float[] pcm, int n, [MarshalAs(UnmanagedType.LPUTF8Str)] string? language, [MarshalAs(UnmanagedType.LPUTF8Str)] string? prompt, IntPtr texts, int count, [Out] float[] scores, [Out] int[] tokens);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_punc_init([MarshalAs(UnmanagedType.LPUTF8Str)] string path);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_punc_process(IntPtr ctx, [MarshalAs(UnmanagedType.LPUTF8Str)] string text);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_punc_free_text(IntPtr text);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_punc_free(IntPtr ctx);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_truecase_init([MarshalAs(UnmanagedType.LPUTF8Str)] string path);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_truecase_process(IntPtr ctx, [MarshalAs(UnmanagedType.LPUTF8Str)] string text);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_truecase_free_text(IntPtr text);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_truecase_free(IntPtr ctx);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_pcs_init([MarshalAs(UnmanagedType.LPUTF8Str)] string path);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_pcs_process(IntPtr ctx, [MarshalAs(UnmanagedType.LPUTF8Str)] string text);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_pcs_free_text(IntPtr text);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_pcs_free(IntPtr ctx);
    }

    public static partial class Audio
    {
        internal static float[] CopyNative(IntPtr p, int n)
        {
            if (n < 0 || (n > 0 && p == IntPtr.Zero)) throw new InvalidOperationException("Invalid native PCM buffer");
            var result = new float[n]; if (n > 0) Marshal.Copy(p, result, 0, n); return result;
        }
        public readonly struct StereoData
        {
            public float[] Samples { get; }
            public int SampleRate { get; }
            public int SourceChannels { get; }
            internal StereoData(float[] pcm, int rate, int channels) { Samples = pcm; SampleRate = rate; SourceChannels = channels; }
        }
        /// <summary>Decode to interleaved stereo at 16 kHz, retaining original mono/stereo flag.</summary>
        public static StereoData LoadStereo(string path)
        {
            ArgumentNullException.ThrowIfNull(path);
            int rc = NativeMethods.crispasr_audio_load_stereo(path, out var left, out var right, out int n, out int rate, out int channels);
            try
            {
                if (rc != 0) throw new InvalidOperationException($"Stereo audio load failed (rc={rc})");
                var l = CopyNative(left, n); var r = CopyNative(right, n); var pcm = new float[checked(n * 2)];
                for (int i = 0; i < n; ++i) { pcm[i * 2] = l[i]; pcm[i * 2 + 1] = r[i]; }
                return new StereoData(pcm, rate, channels);
            }
            finally { if (left != IntPtr.Zero) NativeMethods.crispasr_audio_free(left); if (right != IntPtr.Zero && right != left) NativeMethods.crispasr_audio_free(right); }
        }
        public static float[] Resample(float[] pcm, int sourceRate, int targetRate)
        {
            ArgumentNullException.ThrowIfNull(pcm); if (sourceRate <= 0 || targetRate <= 0) throw new ArgumentOutOfRangeException(nameof(sourceRate));
            int rc = NativeMethods.crispasr_audio_resample(pcm, pcm.Length, sourceRate, targetRate, out var p, out int n);
            try { if (rc != 0) throw new InvalidOperationException($"Audio resample failed (rc={rc})"); return CopyNative(p, n); }
            finally { if (p != IntPtr.Zero) NativeMethods.crispasr_audio_free(p); }
        }
        /// <summary>Sign a container. Throws when native signing is unavailable; binary NUL bytes are preserved.</summary>
        public static byte[] SignC2pa(byte[] data, string format = "audio/wav", string? certificatePath = null, string? keyPath = null)
        {
            ArgumentNullException.ThrowIfNull(data); ArgumentNullException.ThrowIfNull(format);
            if ((certificatePath == null) != (keyPath == null)) throw new ArgumentException("Certificate and key must be paired");
            var p = NativeMethods.crispasr_c2pa_sign(data, (UIntPtr)data.Length, format, certificatePath, keyPath, out var size);
            try { if (p == IntPtr.Zero) throw new InvalidOperationException("Native C2PA signing unavailable or failed"); var output = new byte[checked((int)size.ToUInt64())]; Marshal.Copy(p, output, 0, output.Length); return output; }
            finally { if (p != IntPtr.Zero) NativeMethods.crispasr_c2pa_free(p); }
        }
    }
    public sealed partial class Session
    {
        public string Backend => NativeMethods.PtrToUtf8(NativeMethods.crispasr_session_backend(Handle)) ?? "";
        public void SetPcmSampleRate(int rate) { if (rate <= 0) throw new ArgumentOutOfRangeException(nameof(rate)); StrictCheck(NativeMethods.crispasr_session_set_pcm_sample_rate(Handle, rate), "set_pcm_sample_rate"); }
        public void SetParakeetAttentionContext(int left, int right) => StrictCheck(NativeMethods.crispasr_session_set_parakeet_att_context(Handle, left, right), "set_parakeet_att_context");
        public void SetAudioCtx(int audioCtx) => StrictCheck(NativeMethods.crispasr_session_set_audio_ctx(Handle, audioCtx), "set_audio_ctx");
        public void SetHikariRefreshStride(int stride) => StrictCheck(NativeMethods.crispasr_session_set_hikari_refresh_stride(Handle, stride), "set_hikari_refresh_stride");
        public void SetGrammarStrict(bool enabled) => StrictCheck(NativeMethods.crispasr_session_set_grammar_strict(Handle, enabled ? 1 : 0), "set_grammar_strict");
        public static void FreeAlignerCache() => NativeMethods.crispasr_free_aligner_cache();
        public (float[] LogProbabilities, int[] TokenCounts) ScoreTexts(float[] pcm, string[] candidates, string? language = null, string? prompt = null)
        {
            ArgumentNullException.ThrowIfNull(pcm); ArgumentNullException.ThrowIfNull(candidates);
            if (candidates.Length == 0) throw new ArgumentException("At least one candidate required", nameof(candidates));
            foreach (var text in candidates) ArgumentNullException.ThrowIfNull(text);
            var strings = new IntPtr[candidates.Length]; var pointers = Marshal.AllocHGlobal(checked(strings.Length * IntPtr.Size));
            try
            {
                for (int i = 0; i < strings.Length; ++i) { strings[i] = Marshal.StringToCoTaskMemUTF8(candidates[i]); Marshal.WriteIntPtr(pointers, i * IntPtr.Size, strings[i]); }
                var scores = new float[strings.Length]; var tokens = new int[strings.Length];
                StrictCheck(NativeMethods.crispasr_session_score_texts(Handle, pcm, pcm.Length, language, prompt, pointers, strings.Length, scores, tokens), "score_texts");
                return (scores, tokens);
            }
            finally { foreach (var p in strings) if (p != IntPtr.Zero) Marshal.FreeCoTaskMem(p); Marshal.FreeHGlobal(pointers); }
        }
    }
    public static class RuntimeInfo
    {
        private static void Check(int rc) { if (rc < 0) throw new InvalidOperationException($"Native utility failed (rc={rc})"); }
        public static string Version => NativeMethods.PtrToUtf8(NativeMethods.crispasr_c_api_version()) ?? "";
        public static string HelperVersion => NativeMethods.PtrToUtf8(NativeMethods.crispasr_dart_helpers_version()) ?? "";
        public static void SetGpuBackend(string name) { ArgumentNullException.ThrowIfNull(name); NativeMethods.crispasr_set_gpu_backend(name); }
        public static void LoadWatermarkModel(string path) { ArgumentNullException.ThrowIfNull(path); Check(NativeMethods.crispasr_watermark_load_model(path)); }
        public static string[] DetectBackends(string path) { ArgumentNullException.ThrowIfNull(path); var data = new byte[65536]; Check(NativeMethods.crispasr_detect_backends_from_gguf(path, data, data.Length)); return NativeMethods.NullTerminated(data).Split('\n', StringSplitOptions.RemoveEmptyEntries); }
        public static string[] BackendCapabilities(string backend) { ArgumentNullException.ThrowIfNull(backend); var data = new byte[65536]; Check(NativeMethods.crispasr_backend_caps_abi(backend, data, data.Length)); return NativeMethods.NullTerminated(data).Split(',', StringSplitOptions.RemoveEmptyEntries); }
        public static string AllBackendCapabilities() { int size = NativeMethods.crispasr_backend_caps_list_abi(null, 0); var data = new byte[Math.Max(1, checked(Math.Abs(size)))]; Check(NativeMethods.crispasr_backend_caps_list_abi(data, data.Length)); return NativeMethods.NullTerminated(data); }
    }
    public enum TextProcessorKind { Punctuation, Truecase, PunctuationCapitalizationSegmentation }
    public sealed class TextProcessor : IDisposable
    {
        private IntPtr _handle;
        private readonly TextProcessorKind _kind;
        public TextProcessor(string model, TextProcessorKind kind = TextProcessorKind.Truecase)
        {
            ArgumentNullException.ThrowIfNull(model); if (!Enum.IsDefined(kind)) throw new ArgumentOutOfRangeException(nameof(kind)); _kind = kind;
            _handle = kind switch { TextProcessorKind.Punctuation => NativeMethods.crispasr_punc_init(model), TextProcessorKind.Truecase => NativeMethods.crispasr_truecase_init(model), _ => NativeMethods.crispasr_pcs_init(model) };
            if (_handle == IntPtr.Zero) throw new InvalidOperationException("Text processor failed to open");
        }
        public string Process(string text)
        {
            if (_handle == IntPtr.Zero) throw new ObjectDisposedException(nameof(TextProcessor)); ArgumentNullException.ThrowIfNull(text);
            var p = _kind switch { TextProcessorKind.Punctuation => NativeMethods.crispasr_punc_process(_handle, text), TextProcessorKind.Truecase => NativeMethods.crispasr_truecase_process(_handle, text), _ => NativeMethods.crispasr_pcs_process(_handle, text) };
            try { if (p == IntPtr.Zero) throw new InvalidOperationException("Text processing failed"); return NativeMethods.PtrToUtf8(p) ?? ""; }
            finally { if (p != IntPtr.Zero) { switch (_kind) { case TextProcessorKind.Punctuation: NativeMethods.crispasr_punc_free_text(p); break; case TextProcessorKind.Truecase: NativeMethods.crispasr_truecase_free_text(p); break; default: NativeMethods.crispasr_pcs_free_text(p); break; } } }
        }
        public void Dispose() { if (_handle != IntPtr.Zero) { switch (_kind) { case TextProcessorKind.Punctuation: NativeMethods.crispasr_punc_free(_handle); break; case TextProcessorKind.Truecase: NativeMethods.crispasr_truecase_free(_handle); break; default: NativeMethods.crispasr_pcs_free(_handle); break; } _handle = IntPtr.Zero; } }
    }
}
