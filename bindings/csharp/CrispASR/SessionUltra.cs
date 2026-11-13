using System;
using System.Runtime.InteropServices;

namespace CrispASR
{
    public sealed partial class Session
    {
        /// <summary>
        /// Experimental Ultra alignment with a separately converted local-head GGUF.
        /// Audio must be mono 16 kHz and at most 120 seconds. Word boundaries are
        /// returned in seconds, preserving the head's fractional milliseconds.
        /// </summary>
        /// <remarks>
        /// Requires a native library exporting the experimental Ultra C ABI.
        /// Character timings are unavailable. Broader language and GPU qualification
        /// remain open; see docs/qwen3-ultra-local-head-port.md.
        /// </remarks>
        public static AlignedWord[] AlignWordsUltra(string alignerModel, string headModel,
                                                     string transcript, float[] pcm, int nThreads = 4)
        {
            ValidateUltraString(alignerModel, nameof(alignerModel));
            ValidateUltraString(headModel, nameof(headModel));
            ValidateUltraString(transcript, nameof(transcript));
            if (pcm == null) throw new ArgumentNullException(nameof(pcm));
            if (pcm.Length == 0 || pcm.Length > 120 * 16000)
                throw new ArgumentException("Audio must contain 1 to 1920000 mono 16 kHz samples", nameof(pcm));

            var result = NativeMethods.crispasr_align_words_ultra_abi(
                alignerModel, headModel, transcript, pcm, pcm.Length, nThreads);
            if (result == IntPtr.Zero) throw new InvalidOperationException("Ultra alignment failed");
            try
            {
                int count = NativeMethods.crispasr_ultra_align_result_n_words(result);
                if (count < 0) throw new InvalidOperationException("Ultra alignment returned an invalid word count");
                var words = new AlignedWord[count];
                for (int i = 0; i < count; i++)
                    words[i] = new AlignedWord(
                        NativeMethods.PtrToUtf8(NativeMethods.crispasr_ultra_align_result_word_text(result, i)) ?? "",
                        NativeMethods.crispasr_ultra_align_result_word_start_ms(result, i) / 1000.0,
                        NativeMethods.crispasr_ultra_align_result_word_end_ms(result, i) / 1000.0);
                return words;
            }
            finally
            {
                NativeMethods.crispasr_ultra_align_result_free(result);
            }
        }

        private static void ValidateUltraString(string value, string name)
        {
            if (value == null) throw new ArgumentNullException(name);
            if (value.Length == 0 || value.IndexOf('\0') >= 0)
                throw new ArgumentException("Value must be nonempty and contain no NUL characters", name);
        }
    }

    internal static partial class NativeMethods
    {
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_align_words_ultra_abi(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string alignerModel,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string headModel,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string transcript,
            [In] float[] samples, int nSamples, int nThreads);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int crispasr_ultra_align_result_n_words(IntPtr result);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr crispasr_ultra_align_result_word_text(IntPtr result, int index);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern double crispasr_ultra_align_result_word_start_ms(IntPtr result, int index);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern double crispasr_ultra_align_result_word_end_ms(IntPtr result, int index);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void crispasr_ultra_align_result_free(IntPtr result);
    }
}
