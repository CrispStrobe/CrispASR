using System;
using System.Runtime.InteropServices;
using System.Text;
using Xunit;
using CrispASR;

namespace CrispASR.Tests
{
    /// <summary>
    /// Tests for the NativeMethods helper functions and P/Invoke
    /// signature consistency checks.
    /// </summary>
    public class NativeMethodsHelperTests
    {
        // ---- NullTerminated ----

        [Fact]
        public void NullTerminated_FullBuffer()
        {
            var buf = Encoding.UTF8.GetBytes("hello\0world");
            Assert.Equal("hello", NativeMethods.NullTerminated(buf));
        }

        [Fact]
        public void NullTerminated_NoNull()
        {
            var buf = Encoding.UTF8.GetBytes("hello");
            Assert.Equal("hello", NativeMethods.NullTerminated(buf));
        }

        [Fact]
        public void NullTerminated_EmptyBuffer()
        {
            Assert.Equal("", NativeMethods.NullTerminated(Array.Empty<byte>()));
        }

        [Fact]
        public void NullTerminated_AllZeros()
        {
            Assert.Equal("", NativeMethods.NullTerminated(new byte[10]));
        }

        [Fact]
        public void NullTerminated_Utf8()
        {
            // "über" in UTF-8
            var buf = new byte[] { 0xC3, 0xBC, 0x62, 0x65, 0x72, 0x00 };
            Assert.Equal("\u00fcber", NativeMethods.NullTerminated(buf));
        }

        [Fact]
        public void NullTerminated_SingleChar()
        {
            var buf = new byte[] { (byte)'x', 0 };
            Assert.Equal("x", NativeMethods.NullTerminated(buf));
        }

        [Fact]
        public void NullTerminated_LeadingNull()
        {
            var buf = new byte[] { 0, (byte)'a', (byte)'b' };
            Assert.Equal("", NativeMethods.NullTerminated(buf));
        }

        // ---- PtrToUtf8 ----

        [Fact]
        public void PtrToUtf8_NullReturnsNull()
        {
            Assert.Null(NativeMethods.PtrToUtf8(IntPtr.Zero));
        }

        [Fact]
        public void PtrToUtf8_ValidPointer()
        {
            var str = "hello";
            var ptr = Marshal.StringToCoTaskMemUTF8(str);
            try
            {
                Assert.Equal("hello", NativeMethods.PtrToUtf8(ptr));
            }
            finally
            {
                Marshal.FreeCoTaskMem(ptr);
            }
        }

        [Fact]
        public void PtrToUtf8_EmptyString()
        {
            var ptr = Marshal.StringToCoTaskMemUTF8("");
            try
            {
                Assert.Equal("", NativeMethods.PtrToUtf8(ptr));
            }
            finally
            {
                Marshal.FreeCoTaskMem(ptr);
            }
        }

        [Fact]
        public void PtrToUtf8_Unicode()
        {
            var str = "Sch\u00f6n"; // Schön
            var ptr = Marshal.StringToCoTaskMemUTF8(str);
            try
            {
                Assert.Equal("Sch\u00f6n", NativeMethods.PtrToUtf8(ptr));
            }
            finally
            {
                Marshal.FreeCoTaskMem(ptr);
            }
        }
    }

    /// <summary>
    /// Validate that all P/Invoke entry point names match the C header.
    /// This is a compile-time/reflection check — it verifies method
    /// signatures exist and DllImport attributes are well-formed.
    /// </summary>
    public class PInvokeSignatureTests
    {
        [Theory]
        [InlineData("crispasr_session_open")]
        [InlineData("crispasr_session_open_explicit")]
        [InlineData("crispasr_session_close")]
        [InlineData("crispasr_session_available_backends")]
        [InlineData("crispasr_session_set_codec_path")]
        [InlineData("crispasr_session_set_voice")]
        [InlineData("crispasr_session_set_speaker_name")]
        [InlineData("crispasr_session_set_speaker_id")]
        [InlineData("crispasr_session_n_speakers")]
        [InlineData("crispasr_session_get_speaker_name")]
        [InlineData("crispasr_session_set_instruct")]
        [InlineData("crispasr_session_is_custom_voice")]
        [InlineData("crispasr_session_is_voice_design")]
        [InlineData("crispasr_session_set_punc_model")]
        [InlineData("crispasr_session_set_hotwords")]
        [InlineData("crispasr_session_set_g2p_dict")]
        [InlineData("crispasr_session_set_source_language")]
        [InlineData("crispasr_session_set_target_language")]
        [InlineData("crispasr_session_set_punctuation")]
        [InlineData("crispasr_session_set_translate")]
        [InlineData("crispasr_session_set_temperature")]
        [InlineData("crispasr_session_set_tts_seed")]
        [InlineData("crispasr_session_set_max_new_tokens")]
        [InlineData("crispasr_session_set_frequency_penalty")]
        [InlineData("crispasr_session_set_tts_steps")]
        [InlineData("crispasr_session_set_tts_num_candidates")]
        [InlineData("crispasr_session_set_top_p")]
        [InlineData("crispasr_session_set_min_p")]
        [InlineData("crispasr_session_set_repetition_penalty")]
        [InlineData("crispasr_session_set_cfg_weight")]
        [InlineData("crispasr_session_set_tts_noise_temp")]
        [InlineData("crispasr_session_set_exaggeration")]
        [InlineData("crispasr_session_set_max_speech_tokens")]
        [InlineData("crispasr_session_set_length_scale")]
        [InlineData("crispasr_session_set_best_of")]
        [InlineData("crispasr_session_set_beam_size")]
        [InlineData("crispasr_session_set_grammar_text")]
        [InlineData("crispasr_session_set_fallback_thresholds")]
        [InlineData("crispasr_session_set_alt_n")]
        [InlineData("crispasr_session_set_whisper_decode_extras")]
        [InlineData("crispasr_session_set_ask")]
        [InlineData("crispasr_session_kokoro_clear_phoneme_cache")]
        [InlineData("crispasr_session_synthesize")]
        [InlineData("crispasr_pcm_free")]
        [InlineData("crispasr_session_transcribe")]
        [InlineData("crispasr_session_transcribe_lang")]
        [InlineData("crispasr_session_transcribe_vad")]
        [InlineData("crispasr_session_transcribe_vad_lang")]
        [InlineData("crispasr_session_result_n_segments")]
        [InlineData("crispasr_session_result_segment_text")]
        [InlineData("crispasr_session_result_segment_t0")]
        [InlineData("crispasr_session_result_segment_t1")]
        [InlineData("crispasr_session_result_n_words")]
        [InlineData("crispasr_session_result_word_text")]
        [InlineData("crispasr_session_result_word_t0")]
        [InlineData("crispasr_session_result_word_t1")]
        [InlineData("crispasr_session_result_word_p")]
        [InlineData("crispasr_session_result_word_n_alts")]
        [InlineData("crispasr_session_result_word_alt_text")]
        [InlineData("crispasr_session_result_word_alt_p")]
        [InlineData("crispasr_session_result_free")]
        [InlineData("crispasr_session_detect_language")]
        [InlineData("crispasr_detect_language_pcm")]
        [InlineData("crispasr_align_words_abi")]
        [InlineData("crispasr_align_result_n_words")]
        [InlineData("crispasr_align_result_word_text")]
        [InlineData("crispasr_align_result_word_t0")]
        [InlineData("crispasr_align_result_word_t1")]
        [InlineData("crispasr_align_result_free")]
        [InlineData("crispasr_vad_segments")]
        [InlineData("crispasr_vad_free")]
        [InlineData("crispasr_session_stream_open")]
        [InlineData("crispasr_stream_feed")]
        [InlineData("crispasr_stream_get_text")]
        [InlineData("crispasr_stream_flush")]
        [InlineData("crispasr_stream_close")]
        [InlineData("crispasr_kokoro_resolve_model_for_lang_abi")]
        [InlineData("crispasr_kokoro_resolve_fallback_voice_abi")]
        [InlineData("crispasr_detect_backend_from_gguf")]
        [InlineData("crispasr_enhance_audio_rnnoise")]
        [InlineData("crispasr_titanet_init")]
        [InlineData("crispasr_titanet_free")]
        [InlineData("crispasr_titanet_embed")]
        [InlineData("crispasr_titanet_cosine_sim")]
        [InlineData("crispasr_speaker_db_load")]
        [InlineData("crispasr_speaker_db_free")]
        [InlineData("crispasr_speaker_db_count")]
        [InlineData("crispasr_speaker_db_match")]
        [InlineData("crispasr_speaker_db_enroll")]
        [InlineData("crispasr_session_translate_text")]
        [InlineData("crispasr_session_translate_text_free")]
        [InlineData("crispasr_registry_lookup_abi")]
        [InlineData("crispasr_cache_ensure_file_abi")]
        [InlineData("crispasr_cache_dir_abi")]
        public void PInvoke_MethodExists(string entryPoint)
        {
            // Verify that a method with this exact name exists in NativeMethods
            var method = typeof(NativeMethods).GetMethod(entryPoint,
                System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic);
            Assert.NotNull(method);

            // Verify DllImport attribute is present
            var attr = method!.GetCustomAttributes(typeof(DllImportAttribute), false);
            Assert.Single(attr);

            // Verify calling convention is Cdecl
            var dllImport = (DllImportAttribute)attr[0];
            Assert.Equal(CallingConvention.Cdecl, dllImport.CallingConvention);
        }

        [Fact]
        public void PInvoke_AllMethodsUseCdecl()
        {
            var methods = typeof(NativeMethods).GetMethods(
                System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic);
            foreach (var m in methods)
            {
                var attrs = m.GetCustomAttributes(typeof(DllImportAttribute), false);
                foreach (DllImportAttribute attr in attrs)
                {
                    Assert.Equal(CallingConvention.Cdecl, attr.CallingConvention);
                }
            }
        }

        [Fact]
        public void PInvoke_AllMethodsTargetCrispasrLib()
        {
            var methods = typeof(NativeMethods).GetMethods(
                System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic);
            foreach (var m in methods)
            {
                var attrs = m.GetCustomAttributes(typeof(DllImportAttribute), false);
                foreach (DllImportAttribute attr in attrs)
                {
                    Assert.Equal("crispasr", attr.Value);
                }
            }
        }
    }
}
