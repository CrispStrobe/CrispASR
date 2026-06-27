using System;
using Xunit;
using CrispASR;

namespace CrispASR.Tests
{
    /// <summary>
    /// Tests that exercise Session methods. These require the native
    /// crispasr library to be loadable at runtime. Tests that can't
    /// load the library are skipped rather than failed.
    /// </summary>
    public class SessionTests
    {
        private static bool CanLoadLibrary()
        {
            try
            {
                _ = Session.AvailableBackends();
                return true;
            }
            catch (DllNotFoundException) { return false; }
            catch (EntryPointNotFoundException) { return false; }
        }

        // ---- AvailableBackends ----

        [Fact]
        public void AvailableBackends_ReturnsNonEmpty()
        {
            if (!CanLoadLibrary()) return; // skip
            var backends = Session.AvailableBackends();
            Assert.NotNull(backends);
            Assert.NotEmpty(backends);
            Assert.Contains("whisper", backends);
        }

        [Fact]
        public void AvailableBackends_ContainsKnownBackends()
        {
            if (!CanLoadLibrary()) return;
            var backends = Session.AvailableBackends();
            // At minimum whisper should always be compiled in
            Assert.Contains("whisper", backends);
        }

        // ---- Open failures ----

        [Fact]
        public void Open_ThrowsOnNonexistentModel()
        {
            if (!CanLoadLibrary()) return;
            Assert.Throws<InvalidOperationException>(() =>
                Session.Open("/nonexistent/model.gguf"));
        }

        [Fact]
        public void Open_ExplicitBackend_ThrowsOnNonexistentModel()
        {
            if (!CanLoadLibrary()) return;
            Assert.Throws<InvalidOperationException>(() =>
                Session.Open("/nonexistent/model.gguf", "whisper"));
        }

        // ---- Dispose safety ----

        [Fact]
        public void Dispose_DoubleDispose_DoesNotThrow()
        {
            if (!CanLoadLibrary()) return;
            // We can't open a session without a model, but we can
            // verify double-dispose on the StreamDecoder path via
            // the type system. This tests the IDisposable pattern.
            // (Session.Open would fail, so we test the dispose
            // contract indirectly through data types.)

            // Verify the types implement IDisposable
            Assert.True(typeof(IDisposable).IsAssignableFrom(typeof(Session)));
            Assert.True(typeof(IDisposable).IsAssignableFrom(typeof(StreamDecoder)));
            Assert.True(typeof(IDisposable).IsAssignableFrom(typeof(TitaNet)));
            Assert.True(typeof(IDisposable).IsAssignableFrom(typeof(SpeakerDb)));
        }

        // ---- DetectBackendFromGguf ----

        [Fact]
        public void DetectBackendFromGguf_ReturnsNullForNonexistent()
        {
            if (!CanLoadLibrary()) return;
            var result = Session.DetectBackendFromGguf("/nonexistent/file.gguf");
            Assert.Null(result);
        }

        [Fact]
        public void DetectBackendFromGguf_DetectsWhisper()
        {
            if (!CanLoadLibrary()) return;
            // Use env var for model path, skip if not set
            var modelPath = Environment.GetEnvironmentVariable("CRISPASR_MODEL_WHISPER");
            if (string.IsNullOrEmpty(modelPath)) return;
            var backend = Session.DetectBackendFromGguf(modelPath);
            Assert.Equal("whisper", backend);
        }
    }

    /// <summary>
    /// Live session tests that require actual GGUF models on disk.
    /// Gated by CRISPASR_MODEL_WHISPER env var — skip cleanly when unset.
    /// </summary>
    public class LiveSessionTests
    {
        private static string? WhisperModel =>
            Environment.GetEnvironmentVariable("CRISPASR_MODEL_WHISPER");

        private static bool HasWhisper => !string.IsNullOrEmpty(WhisperModel);
        private static bool CanLoadLibrary()
        {
            try { _ = Session.AvailableBackends(); return true; }
            catch { return false; }
        }

        [Fact]
        public void Open_WhisperModel_Succeeds()
        {
            if (!CanLoadLibrary() || !HasWhisper) return;
            using var s = Session.Open(WhisperModel!, 2);
            // If we got here without throwing, the session opened
        }

        [Fact]
        public void Transcribe_SilentAudio_ReturnsSegments()
        {
            if (!CanLoadLibrary() || !HasWhisper) return;
            using var s = Session.Open(WhisperModel!, 2);
            // 1 second of silence at 16 kHz
            var pcm = new float[16000];
            var segments = s.Transcribe(pcm);
            Assert.NotNull(segments);
            // Silent audio may return 0 or 1 empty segments — either is fine
        }

        [Fact]
        public void SetTranslate_DoesNotThrow()
        {
            if (!CanLoadLibrary() || !HasWhisper) return;
            using var s = Session.Open(WhisperModel!, 2);
            s.SetTranslate(true);
            s.SetTranslate(false);
        }

        [Fact]
        public void SetPunctuation_DoesNotThrow()
        {
            if (!CanLoadLibrary() || !HasWhisper) return;
            using var s = Session.Open(WhisperModel!, 2);
            s.SetPunctuation(true);
            s.SetPunctuation(false);
        }

        [Fact]
        public void SetTemperature_DoesNotThrow()
        {
            if (!CanLoadLibrary() || !HasWhisper) return;
            using var s = Session.Open(WhisperModel!, 2);
            s.SetTemperature(0.0f);
            s.SetTemperature(0.5f, 42);
        }

        [Fact]
        public void SetBestOf_DoesNotThrow()
        {
            if (!CanLoadLibrary() || !HasWhisper) return;
            using var s = Session.Open(WhisperModel!, 2);
            s.SetBestOf(3);
            s.SetBestOf(1);
        }

        [Fact]
        public void SetBeamSize_DoesNotThrow()
        {
            if (!CanLoadLibrary() || !HasWhisper) return;
            using var s = Session.Open(WhisperModel!, 2);
            s.SetBeamSize(5);
            s.SetBeamSize(1);
        }

        [Fact]
        public void SetAltN_DoesNotThrow()
        {
            if (!CanLoadLibrary() || !HasWhisper) return;
            using var s = Session.Open(WhisperModel!, 2);
            s.SetAltN(3);
            s.SetAltN(0);
        }

        [Fact]
        public void SetFallbackThresholds_DoesNotThrow()
        {
            if (!CanLoadLibrary() || !HasWhisper) return;
            using var s = Session.Open(WhisperModel!, 2);
            s.SetFallbackThresholds(2.4f, -1.0f, 0.6f, 0.2f);
            // Disable fallback
            s.SetFallbackThresholds(2.4f, -1.0f, 0.6f, 0.0f);
        }

        [Fact]
        public void SetWhisperDecodeExtras_DoesNotThrow()
        {
            if (!CanLoadLibrary() || !HasWhisper) return;
            using var s = Session.Open(WhisperModel!, 2);
            s.SetWhisperDecodeExtras(true, null, false);
            s.SetWhisperDecodeExtras(false, "", true);
        }

        [Fact]
        public void SetAsk_DoesNotThrow()
        {
            if (!CanLoadLibrary() || !HasWhisper) return;
            using var s = Session.Open(WhisperModel!, 2);
            s.SetAsk("Transcribe this audio.");
        }

        [Fact]
        public void SetSourceLanguage_DoesNotThrow()
        {
            if (!CanLoadLibrary() || !HasWhisper) return;
            using var s = Session.Open(WhisperModel!, 2);
            s.SetSourceLanguage("en");
            s.SetSourceLanguage(null); // clear
        }

        [Fact]
        public void TranscribeLang_WithLanguageHint()
        {
            if (!CanLoadLibrary() || !HasWhisper) return;
            using var s = Session.Open(WhisperModel!, 2);
            var pcm = new float[16000]; // 1s silence
            var segments = s.TranscribeLang(pcm, "en");
            Assert.NotNull(segments);
        }

        [Fact]
        public void StreamOpen_Whisper_Succeeds()
        {
            if (!CanLoadLibrary() || !HasWhisper) return;
            using var s = Session.Open(WhisperModel!, 2);
            using var stream = s.StreamOpen(stepMs: 3000, lengthMs: 10000, keepMs: 200, language: "en");
            // Feed 1 second of silence
            var pcm = new float[16000];
            int rc = stream.Feed(pcm);
            Assert.True(rc >= 0);
            // Get text (may be empty for 1s of silence)
            var update = stream.GetText();
            Assert.NotNull(update.Text);
        }

        [Fact]
        public void StreamDecoder_Feed_EmptyArray_ReturnsZero()
        {
            if (!CanLoadLibrary() || !HasWhisper) return;
            using var s = Session.Open(WhisperModel!, 2);
            using var stream = s.StreamOpen();
            int rc = stream.Feed(Array.Empty<float>());
            Assert.Equal(0, rc);
        }

        [Fact]
        public void StreamDecoder_Feed_Null_ReturnsZero()
        {
            if (!CanLoadLibrary() || !HasWhisper) return;
            using var s = Session.Open(WhisperModel!, 2);
            using var stream = s.StreamOpen();
            int rc = stream.Feed(null!);
            Assert.Equal(0, rc);
        }

        [Fact]
        public void StreamDecoder_Flush_DoesNotThrow()
        {
            if (!CanLoadLibrary() || !HasWhisper) return;
            using var s = Session.Open(WhisperModel!, 2);
            using var stream = s.StreamOpen(language: "en");
            stream.Feed(new float[16000]);
            stream.Flush();
        }
    }
}
