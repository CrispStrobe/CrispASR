using System;
using Xunit;
using CrispASR;

namespace CrispASR.Tests
{
    /// <summary>
    /// Tests for the readonly-struct data types — these run without
    /// the native library and validate the public API surface.
    /// </summary>
    public class DataTypeTests
    {
        // ---- Word ----

        [Fact]
        public void Word_DefaultAlts_IsEmptyArray()
        {
            var w = new Word("hello", 0, 100, 0.95f);
            Assert.NotNull(w.Alts);
            Assert.Empty(w.Alts);
        }

        [Fact]
        public void Word_WithAlts_PreservesAll()
        {
            var alts = new[] { new AltToken("helo", 0.03f), new AltToken("hullo", 0.02f) };
            var w = new Word("hello", 10, 50, 0.95f, alts);
            Assert.Equal("hello", w.Text);
            Assert.Equal(10, w.T0);
            Assert.Equal(50, w.T1);
            Assert.Equal(0.95f, w.P);
            Assert.Equal(2, w.Alts.Length);
            Assert.Equal("helo", w.Alts[0].Text);
            Assert.Equal(0.03f, w.Alts[0].P);
        }

        [Fact]
        public void Word_ToString_ShowsRange()
        {
            var w = new Word("test", 100, 200, 0.9f);
            Assert.Equal("100-200 test", w.ToString());
        }

        // ---- AltToken ----

        [Fact]
        public void AltToken_ToString_ShowsPercentage()
        {
            var a = new AltToken("foo", 0.123f);
            Assert.Contains("foo", a.ToString());
            Assert.Contains("12.3%", a.ToString());
        }

        [Fact]
        public void AltToken_ZeroProbability()
        {
            var a = new AltToken("x", 0f);
            Assert.Equal(0f, a.P);
        }

        // ---- Segment ----

        [Fact]
        public void Segment_Properties()
        {
            var words = new[] { new Word("a", 0, 10, 1.0f), new Word("b", 10, 20, 0.9f) };
            var seg = new Segment("a b", 0, 20, words);
            Assert.Equal("a b", seg.Text);
            Assert.Equal(0, seg.T0);
            Assert.Equal(20, seg.T1);
            Assert.Equal(2, seg.Words.Length);
        }

        [Fact]
        public void Segment_ToString_ShowsBracketedRange()
        {
            var seg = new Segment("hello world", 50, 300, Array.Empty<Word>());
            Assert.Equal("[50-300] hello world", seg.ToString());
        }

        [Fact]
        public void Segment_EmptyWords()
        {
            var seg = new Segment("text", 0, 100, Array.Empty<Word>());
            Assert.Empty(seg.Words);
        }

        // ---- AlignedWord ----

        [Fact]
        public void AlignedWord_Properties()
        {
            var aw = new AlignedWord("hello", 100, 250);
            Assert.Equal("hello", aw.Text);
            Assert.Equal(100, aw.T0);
            Assert.Equal(250, aw.T1);
        }

        // ---- VadSpan ----

        [Fact]
        public void VadSpan_Properties()
        {
            var span = new VadSpan(1.5, 3.7);
            Assert.Equal(1.5, span.T0);
            Assert.Equal(3.7, span.T1);
        }

        [Fact]
        public void VadSpan_ZeroDuration()
        {
            var span = new VadSpan(2.0, 2.0);
            Assert.Equal(span.T0, span.T1);
        }

        // ---- LanguageDetection ----

        [Fact]
        public void LanguageDetection_Ok_WhenValid()
        {
            var ld = new LanguageDetection("en", 0.98f);
            Assert.True(ld.Ok);
            Assert.Equal("en", ld.Code);
            Assert.Equal(0.98f, ld.Probability);
        }

        [Fact]
        public void LanguageDetection_NotOk_WhenEmpty()
        {
            var ld = new LanguageDetection("", 0.5f);
            Assert.False(ld.Ok);
        }

        [Fact]
        public void LanguageDetection_NotOk_WhenNegativeProb()
        {
            var ld = new LanguageDetection("de", -1.0f);
            Assert.False(ld.Ok);
        }

        [Fact]
        public void LanguageDetection_ToString()
        {
            var ld = new LanguageDetection("fr", 0.876f);
            Assert.Contains("fr", ld.ToString());
            Assert.Contains("87.6%", ld.ToString());
        }

        // ---- KokoroResolved ----

        [Fact]
        public void KokoroResolved_WithVoice()
        {
            var r = new KokoroResolved("/models/kokoro.gguf", "/voices/de.gguf", "df_victoria", true);
            Assert.Equal("/models/kokoro.gguf", r.ModelPath);
            Assert.Equal("/voices/de.gguf", r.VoicePath);
            Assert.Equal("df_victoria", r.VoiceName);
            Assert.True(r.BackboneSwapped);
        }

        [Fact]
        public void KokoroResolved_WithoutVoice()
        {
            var r = new KokoroResolved("/models/kokoro.gguf", null, null, false);
            Assert.Null(r.VoicePath);
            Assert.Null(r.VoiceName);
            Assert.False(r.BackboneSwapped);
        }

        // ---- StreamingUpdate ----

        [Fact]
        public void StreamingUpdate_Properties()
        {
            var su = new StreamingUpdate("hello world", 1.5, 3.0, 42);
            Assert.Equal("hello world", su.Text);
            Assert.Equal(1.5, su.T0);
            Assert.Equal(3.0, su.T1);
            Assert.Equal(42, su.Counter);
        }

        [Fact]
        public void StreamingUpdate_EmptyText()
        {
            var su = new StreamingUpdate("", 0, 0, 0);
            Assert.Equal("", su.Text);
            Assert.Equal(0, su.Counter);
        }
    }
}
