package io.github.ggerganov.whispercpp;

import com.sun.jna.*;
import com.sun.jna.ptr.*;
import java.util.*;
import java.util.function.*;

/** Additional typed session tasks. Returned borrowed buffers are always copied. */
public abstract class CrispasrSessionTasks extends CrispasrUtilities {
  protected abstract Pointer nativeHandle();
  protected abstract CrispasrSession.Segment[] readNativeResult(Pointer result);
  protected boolean inCallback;
  private Throwable callbackError;
  private PcmCallback pcmCallback;
  private ProgressCallback progressCallback;
  private SegmentCallback segmentCallback;
  private TokenCallback tokenCallback;

  interface PcmCallback extends Callback {
    void invoke(Pointer pcm, int n, int last, Pointer data);
  }
  interface ProgressCallback extends Callback {
    void invoke(int done, int total, Pointer data);
  }
  interface SegmentCallback extends Callback {
    void invoke(Pointer text, long t0, long t1, int index, Pointer data);
  }
  interface TokenCallback extends Callback {
    void invoke(Pointer text, int index, Pointer data);
  }
  interface Tasks extends Library {
    Tasks INSTANCE = Native.load("crispasr", Tasks.class, CrispasrSession.nativeOptions());
    int crispasr_session_piano(Pointer s, float[] pcm, int n);
    Pointer crispasr_session_piano_notes(Pointer s, IntByReference n);
    Pointer crispasr_session_piano_note_programs(Pointer s, IntByReference n);
    int crispasr_session_piano_sample_rate(Pointer s);
    int crispasr_session_piano_n_notes(Pointer s);
    int crispasr_session_chords(Pointer s, float[] pcm, int n, int rate);
    Pointer crispasr_session_chords_spans(Pointer s, IntByReference n);
    String crispasr_session_chords_span_name(Pointer s, int i);
    int crispasr_session_chords_vocab_size(Pointer s);
    int crispasr_session_chords_n_spans(Pointer s);
    int crispasr_session_beats(Pointer s, float[] pcm, int n, int rate);
    Pointer crispasr_session_beats_events(Pointer s, IntByReference n);
    float crispasr_session_beats_tempo_bpm(Pointer s);
    int crispasr_session_beats_sample_rate(Pointer s);
    int crispasr_session_beats_n_events(Pointer s);
    int crispasr_session_tab(Pointer s, float[] pcm, int n, int rate);
    Pointer crispasr_session_tab_emissions(
        Pointer s, IntByReference n, IntByReference strings, IntByReference classes);
    int crispasr_session_tab_silent_class(Pointer s);
    float crispasr_session_tab_frame_period(Pointer s);
    int crispasr_session_tab_string_open_midi(Pointer s, int string);
    int crispasr_session_tab_n_frames(Pointer s);
    int crispasr_session_pitch(Pointer s, float[] pcm, int n, float hop);
    Pointer crispasr_session_pitch_frames(Pointer s, IntByReference n);
    int crispasr_session_pitch_sample_rate(Pointer s);
    int crispasr_session_pitch_n_frames(Pointer s);
    int crispasr_session_separate(Pointer s, float[] pcm, int frames);
    String crispasr_session_separate_stem_name(Pointer s, int index);
    Pointer crispasr_session_separate_stem(Pointer s, int index, IntByReference frames);
    int crispasr_session_separate_n_stems(Pointer s);
    int crispasr_session_separate_sample_rate(Pointer s);
    int crispasr_session_convert(
        Pointer s, float[] content, int frames, float[] f0, int speaker, float[] zp, float[] sine);
    Pointer crispasr_session_convert_audio(Pointer s, IntByReference n);
    int crispasr_session_convert_content_dim(Pointer s);
    int crispasr_session_convert_n_speakers(Pointer s);
    int crispasr_session_convert_sample_rate(Pointer s);
    int crispasr_session_convert_inter_channels(Pointer s);
    int crispasr_session_convert_upsample_product(Pointer s);
    int crispasr_session_set_voice_samples(Pointer s, float[] pcm, int n, int rate, String text);
    int crispasr_session_set_tts_cfg_scale(Pointer s, float scale);
    int crispasr_session_tada_set_makeref_models(Pointer s, String encoder, String aligner);
    String crispasr_session_last_synth_error(Pointer s);
    String crispasr_session_disclaimer_text();
    Pointer crispasr_session_get_disclaimer_pcm(Pointer s, IntByReference n);
    int crispasr_session_synthesize_streaming(Pointer s, String text, PcmCallback cb, Pointer data);
    void crispasr_session_set_progress_callback(Pointer s, ProgressCallback cb, Pointer data);
    void crispasr_session_set_segment_callback(Pointer s, SegmentCallback cb, Pointer data);
    void crispasr_session_set_token_callback(Pointer s, TokenCallback cb, Pointer data);
    void crispasr_reset_progress();
    int crispasr_get_streamed_segment_count();
    int crispasr_get_streamed_token_count();
    Pointer crispasr_drain_streamed_segments();
    Pointer crispasr_drain_streamed_tokens(IntByReference n);
    void crispasr_reset_streamed_segments();
    void crispasr_reset_streamed_tokens();
    int crispasr_session_set_pcm_sample_rate(Pointer s, int rate);
    int crispasr_session_set_parakeet_att_context(Pointer s, int left, int right);
    int crispasr_session_set_audio_ctx(Pointer s, int audio_ctx);
    int crispasr_session_set_hikari_refresh_stride(Pointer s, int stride);
    int crispasr_session_set_grammar_strict(Pointer s, int strict);
    void crispasr_free_aligner_cache();
  }
  static void check(int rc, String name) {
    if (rc < 0)
      throw new IllegalStateException(name + " failed (rc=" + rc + ")");
  }
  static void checkSetter(int rc, String name) {
    if (rc != 0)
      throw new IllegalStateException(name + " failed (rc=" + rc + ")");
  }
  static float[] copy(Pointer p, int n) {
    if (n < 0 || (n > 0 && p == null))
      throw new IllegalStateException("invalid native result buffer");
    return n == 0 ? new float[0] : p.getFloatArray(0, n);
  }
  public static final class PianoResult {
    /** Four floats per note: onset_ms, offset_ms, MIDI pitch, velocity. */
    public final float[] notes;
    public final int[] programs;
    public final int sampleRate;
    PianoResult(float[] n, int[] p, int sr) {
      notes = n;
      programs = p;
      sampleRate = sr;
    }
  }
  public PianoResult piano(float[] pcm) {
    Pointer s = nativeHandle();
    check(Tasks.INSTANCE.crispasr_session_piano(s, pcm, pcm.length), "piano");
    IntByReference n = new IntByReference();
    Pointer p = Tasks.INSTANCE.crispasr_session_piano_notes(s, n);
    float[] notes = copy(p, Math.multiplyExact(n.getValue(), 4));
    int[] programs = new int[n.getValue()];
    try {
      IntByReference count = new IntByReference();
      Pointer q = Tasks.INSTANCE.crispasr_session_piano_note_programs(s, count);
      if (count.getValue() != programs.length || (programs.length > 0 && q == null))
        throw new IllegalStateException("piano program count mismatch");
      if (programs.length > 0)
        programs = q.getIntArray(0, programs.length);
    } catch (UnsatisfiedLinkError absent) {
      Arrays.fill(programs, -1);
    }
    return new PianoResult(notes, programs, pianoSampleRate());
  }
  public int pianoSampleRate() {
    return Tasks.INSTANCE.crispasr_session_piano_sample_rate(nativeHandle());
  }
  public int pianoNoteCount() {
    return Tasks.INSTANCE.crispasr_session_piano_n_notes(nativeHandle());
  }
  public static final class ChordResult {
    /** Four floats per span: start_ms, end_ms, label, confidence. */
    public final float[] spans;
    public final String[] names;
    public final int vocabularySize;
    ChordResult(float[] s, String[] n, int v) {
      spans = s;
      names = n;
      vocabularySize = v;
    }
  }
  public ChordResult chords(float[] pcm, int rate) {
    if (rate <= 0)
      throw new IllegalArgumentException("positive sample rate required");
    Pointer s = nativeHandle();
    check(Tasks.INSTANCE.crispasr_session_chords(s, pcm, pcm.length, rate), "chords");
    IntByReference n = new IntByReference();
    Pointer p = Tasks.INSTANCE.crispasr_session_chords_spans(s, n);
    float[] spans = copy(p, Math.multiplyExact(n.getValue(), 4));
    String[] names = new String[n.getValue()];
    for (int i = 0; i < names.length; i++) names[i] = Tasks.INSTANCE.crispasr_session_chords_span_name(s, i);
    return new ChordResult(spans, names, Tasks.INSTANCE.crispasr_session_chords_vocab_size(s));
  }
  public int chordSpanCount() {
    return Tasks.INSTANCE.crispasr_session_chords_n_spans(nativeHandle());
  }
  public int chordsVocabSize() {
    return Tasks.INSTANCE.crispasr_session_chords_vocab_size(nativeHandle());
  }
  public static final class BeatResult {
    /** Two floats per event: time_s, is_downbeat. */
    public final float[] events;
    public final float tempoBpm;
    public final int sampleRate;
    BeatResult(float[] e, float t, int r) {
      events = e;
      tempoBpm = t;
      sampleRate = r;
    }
  }
  public BeatResult beats(float[] pcm) {
    return beats(pcm, beatsSampleRate());
  }
  public BeatResult beats(float[] pcm, int rate) {
    if (rate <= 0)
      throw new IllegalArgumentException("positive native sample rate required");
    Pointer s = nativeHandle();
    check(Tasks.INSTANCE.crispasr_session_beats(s, pcm, pcm.length, rate), "beats");
    IntByReference n = new IntByReference();
    Pointer p = Tasks.INSTANCE.crispasr_session_beats_events(s, n);
    return new BeatResult(copy(p, Math.multiplyExact(n.getValue(), 2)),
        Tasks.INSTANCE.crispasr_session_beats_tempo_bpm(s), beatsSampleRate());
  }
  public int beatsSampleRate() {
    return Tasks.INSTANCE.crispasr_session_beats_sample_rate(nativeHandle());
  }
  public int beatEventCount() {
    return Tasks.INSTANCE.crispasr_session_beats_n_events(nativeHandle());
  }
  public static final class TabResult {
    public final float[] emissions;
    public final int frames, strings, classes, silentClass;
    public final float framePeriod;
    public final int[] openMidi;
    TabResult(float[] e, int f, int s, int c, int silent, float period, int[] midi) {
      emissions = e;
      frames = f;
      strings = s;
      classes = c;
      silentClass = silent;
      framePeriod = period;
      openMidi = midi;
    }
  }
  /** Emission log-probabilities, not a decoded playable fingering. */
  public TabResult tab(float[] pcm, int rate) {
    if (rate <= 0)
      throw new IllegalArgumentException("positive sample rate required");
    Pointer s = nativeHandle();
    check(Tasks.INSTANCE.crispasr_session_tab(s, pcm, pcm.length, rate), "tab");
    IntByReference f = new IntByReference(), strings = new IntByReference(), classes = new IntByReference();
    Pointer p = Tasks.INSTANCE.crispasr_session_tab_emissions(s, f, strings, classes);
    float[] emissions =
        copy(p, Math.multiplyExact(Math.multiplyExact(f.getValue(), strings.getValue()), classes.getValue()));
    int[] midi = new int[strings.getValue()];
    for (int i = 0; i < midi.length; i++)
      midi[i] = Tasks.INSTANCE.crispasr_session_tab_string_open_midi(s, i);
    return new TabResult(emissions, f.getValue(), strings.getValue(), classes.getValue(),
        Tasks.INSTANCE.crispasr_session_tab_silent_class(s),
        Tasks.INSTANCE.crispasr_session_tab_frame_period(s), midi);
  }
  public int tabFrameCount() {
    return Tasks.INSTANCE.crispasr_session_tab_n_frames(nativeHandle());
  }
  public float[] pitch(float[] pcm, float hopMs) {
    Pointer s = nativeHandle();
    check(Tasks.INSTANCE.crispasr_session_pitch(s, pcm, pcm.length, hopMs), "pitch");
    IntByReference n = new IntByReference();
    Pointer p = Tasks.INSTANCE.crispasr_session_pitch_frames(s, n);
    return copy(p, Math.multiplyExact(n.getValue(), 3));
  }
  public int pitchSampleRate() {
    return Tasks.INSTANCE.crispasr_session_pitch_sample_rate(nativeHandle());
  }
  public int pitchFrameCount() {
    return Tasks.INSTANCE.crispasr_session_pitch_n_frames(nativeHandle());
  }
  public Map<String, float[]> separate(float[] stereo) {
    if (stereo.length % 2 != 0)
      throw new IllegalArgumentException("interleaved stereo required");
    Pointer s = nativeHandle();
    int n = Tasks.INSTANCE.crispasr_session_separate(s, stereo, stereo.length / 2);
    check(n, "separate");
    Map<String, float[]> out = new LinkedHashMap<>();
    for (int i = 0; i < n; i++) {
      IntByReference frames = new IntByReference();
      Pointer p = Tasks.INSTANCE.crispasr_session_separate_stem(s, i, frames);
      out.put(Tasks.INSTANCE.crispasr_session_separate_stem_name(s, i),
          copy(p, Math.multiplyExact(frames.getValue(), 2)));
    }
    return out;
  }
  public int separationSampleRate() {
    return Tasks.INSTANCE.crispasr_session_separate_sample_rate(nativeHandle());
  }
  public int separationStemCount() {
    return Tasks.INSTANCE.crispasr_session_separate_n_stems(nativeHandle());
  }
  public int convertContentDim() {
    return Tasks.INSTANCE.crispasr_session_convert_content_dim(nativeHandle());
  }
  public int convertSpeakerCount() {
    return Tasks.INSTANCE.crispasr_session_convert_n_speakers(nativeHandle());
  }
  public int convertSampleRate() {
    return Tasks.INSTANCE.crispasr_session_convert_sample_rate(nativeHandle());
  }
  public int convertInterChannels() {
    return Tasks.INSTANCE.crispasr_session_convert_inter_channels(nativeHandle());
  }
  public int convertUpsampleProduct() {
    return Tasks.INSTANCE.crispasr_session_convert_upsample_product(nativeHandle());
  }
  public float[] convert(float[] content, float[] f0, int speaker) {
    return convert(content, f0, speaker, null, null);
  }
  public float[] convert(float[] content, float[] f0, int speaker, float[] noiseZp, float[] noiseSine) {
    Pointer s = nativeHandle();
    int dim = convertContentDim();
    if (dim <= 0)
      throw new UnsupportedOperationException("no conversion arm");
    if (f0.length == 0 || content.length != Math.multiplyExact(dim, f0.length))
      throw new IllegalArgumentException("content/F0 shapes do not match model");
    if (speaker < 0 || speaker >= convertSpeakerCount())
      throw new IllegalArgumentException("speaker out of range");
    try {
      if (noiseZp != null && noiseZp.length != Math.multiplyExact(f0.length, convertInterChannels()))
        throw new IllegalArgumentException("noise_zp shape mismatch");
      if (noiseSine != null && noiseSine.length != Math.multiplyExact(f0.length, convertUpsampleProduct()))
        throw new IllegalArgumentException("noise_sine shape mismatch");
    } catch (UnsatisfiedLinkError absent) {
      throw new UnsupportedOperationException("replay noise needs a newer library", absent);
    }
    int rc = Tasks.INSTANCE.crispasr_session_convert(s, content, f0.length, f0, speaker, noiseZp, noiseSine);
    if (rc <= 0)
      throw new IllegalStateException("convert failed (rc=" + rc + ")");
    IntByReference n = new IntByReference();
    Pointer p = Tasks.INSTANCE.crispasr_session_convert_audio(s, n);
    return copy(p, n.getValue());
  }
  public void setVoiceSamples(float[] pcm, int rate, String referenceText) {
    if (pcm.length == 0 || rate <= 0)
      throw new IllegalArgumentException("reference PCM and positive rate required");
    checkSetter(Tasks.INSTANCE.crispasr_session_set_voice_samples(
                    nativeHandle(), pcm, pcm.length, rate, referenceText),
        "set_voice_samples");
  }
  public void setTtsCfgScale(float scale) {
    checkSetter(
        Tasks.INSTANCE.crispasr_session_set_tts_cfg_scale(nativeHandle(), scale), "set_tts_cfg_scale");
  }
  public void tadaSetMakerefModels(String encoder, String aligner) {
    checkSetter(Tasks.INSTANCE.crispasr_session_tada_set_makeref_models(nativeHandle(), encoder, aligner),
        "tada_set_makeref_models");
  }
  public String lastSynthError() {
    return Tasks.INSTANCE.crispasr_session_last_synth_error(nativeHandle());
  }
  public static String disclaimerText() {
    return Tasks.INSTANCE.crispasr_session_disclaimer_text();
  }
  public float[] getDisclaimerPcm() {
    IntByReference n = new IntByReference();
    Pointer p = Tasks.INSTANCE.crispasr_session_get_disclaimer_pcm(nativeHandle(), n);
    try {
      if (p == null || n.getValue() <= 0)
        throw new IllegalStateException(lastSynthError());
      return copy(p, n.getValue());
    } finally {
      if (p != null)
        CrispasrSession.Lib.INSTANCE.crispasr_pcm_free(p);
    }
  }
  protected void throwCallbackError() {
    Throwable error = callbackError;
    callbackError = null;
    if (error instanceof Error)
      throw (Error) error;
    if (error instanceof RuntimeException)
      throw (RuntimeException) error;
    if (error != null)
      throw new RuntimeException(error);
  }
  private void invokeCallback(Runnable action) {
    if (callbackError != null)
      return;
    inCallback = true;
    try {
      action.run();
    } catch (Throwable error) {
      callbackError = error;
    } finally {
      inCallback = false;
    }
  }
  /**
   * Synchronous sentence chunks; callbacks receive independent arrays. Do not re-enter/close this session.
   */
  public void synthesizeStreaming(String text, BiConsumer<float[], Boolean> callback) {
    Objects.requireNonNull(callback);
    Pointer s = nativeHandle();
    pcmCallback = (pcm, n, last, ignored) -> invokeCallback(() -> {
      if (pcm == null || n <= 0)
        throw new IllegalStateException("streaming synthesis returned no audio");
      callback.accept(copy(pcm, n), last != 0);
    });
    try {
      int rc = Tasks.INSTANCE.crispasr_session_synthesize_streaming(s, text, pcmCallback, null);
      throwCallbackError();
      checkSetter(rc, "synthesize_streaming");
    } finally {
      pcmCallback = null;
    }
  }
  public void setProgressCallback(BiConsumer<Integer, Integer> callback) {
    ProgressCallback cb = callback == null
        ? null
        : (done, total, ignored) -> invokeCallback(() -> callback.accept(done, total));
    Tasks.INSTANCE.crispasr_session_set_progress_callback(nativeHandle(), cb, null);
    progressCallback = cb;
  }
  @FunctionalInterface
  public interface SegmentConsumer {
    void accept(String text, long t0, long t1, int index);
  }
  public void setSegmentCallback(SegmentConsumer callback) {
    SegmentCallback cb = callback == null ? null : (text, t0, t1, index, ignored)
        -> invokeCallback(
            () -> callback.accept(text == null ? "" : text.getString(0, "UTF-8"), t0, t1, index));
    Tasks.INSTANCE.crispasr_session_set_segment_callback(nativeHandle(), cb, null);
    segmentCallback = cb;
  }
  public void setTokenCallback(BiConsumer<byte[], Integer> callback) {
    TokenCallback cb = callback == null
        ? null
        : (text, index,
              ignored) -> invokeCallback(() -> callback.accept(CrispasrSession.copyUtf8Bytes(text), index));
    Tasks.INSTANCE.crispasr_session_set_token_callback(nativeHandle(), cb, null);
    tokenCallback = cb;
  }
  public static void resetProgress() {
    Tasks.INSTANCE.crispasr_reset_progress();
  }
  public static int streamedSegmentCount() {
    return Tasks.INSTANCE.crispasr_get_streamed_segment_count();
  }
  public static int streamedTokenCount() {
    return Tasks.INSTANCE.crispasr_get_streamed_token_count();
  }
  public static void resetStreamedSegments() {
    Tasks.INSTANCE.crispasr_reset_streamed_segments();
  }
  public static void resetStreamedTokens() {
    Tasks.INSTANCE.crispasr_reset_streamed_tokens();
  }
  /** Global polling selects the last streaming native session, which may differ from this receiver. */
  public CrispasrSession.Segment[] drainStreamedSegments() {
    nativeHandle();
    Pointer r = Tasks.INSTANCE.crispasr_drain_streamed_segments();
    if (r == null)
      return new CrispasrSession.Segment[0];
    try {
      return readNativeResult(r);
    } finally {
      CrispasrSession.Lib.INSTANCE.crispasr_session_result_free(r);
    }
  }
  public static byte[][] drainStreamedTokens() {
    IntByReference n = new IntByReference();
    Pointer p = Tasks.INSTANCE.crispasr_drain_streamed_tokens(n);
    if (n.getValue() < 0 || (n.getValue() > 0 && p == null))
      throw new IllegalStateException("invalid token buffer");
    byte[][] out = new byte[n.getValue()][];
    for (int i = 0; i < out.length; i++) {
      out[i] = CrispasrSession.copyUtf8Bytes(p);
      p = p.share(out[i].length + 1);
    }
    return out;
  }
  public void setPcmSampleRate(int rate) {
    checkSetter(
        Tasks.INSTANCE.crispasr_session_set_pcm_sample_rate(nativeHandle(), rate), "set_pcm_sample_rate");
  }
  public void setParakeetAttContext(int left, int right) {
    checkSetter(Tasks.INSTANCE.crispasr_session_set_parakeet_att_context(nativeHandle(), left, right),
        "set_parakeet_att_context");
  }
  public void setAudioCtx(int audioCtx) {
    checkSetter(Tasks.INSTANCE.crispasr_session_set_audio_ctx(nativeHandle(), audioCtx),
        "set_audio_ctx");
  }
  public void setHikariRefreshStride(int stride) {
    checkSetter(Tasks.INSTANCE.crispasr_session_set_hikari_refresh_stride(nativeHandle(), stride),
        "set_hikari_refresh_stride");
  }
  public void setGrammarStrict(boolean strict) {
    checkSetter(Tasks.INSTANCE.crispasr_session_set_grammar_strict(nativeHandle(), strict ? 1 : 0),
        "set_grammar_strict");
  }
  public static void freeAlignerCache() {
    Tasks.INSTANCE.crispasr_free_aligner_cache();
  }
}
