import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

/// A transcription segment with timing.
class Segment {
  final String text;
  final double start; // seconds
  final double end;   // seconds
  final double noSpeechProb;
  final List<Word> words;

  Segment({
    required this.text,
    required this.start,
    required this.end,
    this.noSpeechProb = 0.0,
    this.words = const [],
  });

  @override
  String toString() =>
      '[${start.toStringAsFixed(1)}s - ${end.toStringAsFixed(1)}s] $text';
}

/// Word- / token-level timing, populated when
/// [TranscribeOptions.wordTimestamps] is true.
class Word {
  final String text;
  final double start; // seconds
  final double end;   // seconds
  final double p;     // token probability [0, 1]

  const Word({
    required this.text,
    required this.start,
    required this.end,
    required this.p,
  });

  @override
  String toString() =>
      '${start.toStringAsFixed(2)}-${end.toStringAsFixed(2)} $text';
}

/// Result of [CrispASR.detectLanguage].
class LanguageDetection {
  /// ISO-639 code (e.g. `en`, `de`). Empty if detection failed.
  final String code;
  /// Posterior probability of the detected language, in [0, 1].
  /// Negative when the underlying call failed.
  final double probability;

  const LanguageDetection({required this.code, required this.probability});

  bool get ok => code.isNotEmpty && probability >= 0.0;
  @override
  String toString() =>
      'LanguageDetection($code, ${(probability * 100).toStringAsFixed(1)}%)';
}

/// PCM returned by [decodeAudioFile] — always 16 kHz mono float32.
class DecodedAudio {
  final Float32List samples;
  final int sampleRate;
  const DecodedAudio({required this.samples, required this.sampleRate});

  double get durationSeconds => samples.length / sampleRate;
}

/// Decode any WAV / MP3 / FLAC file to 16 kHz mono float32 PCM using
/// miniaudio shipped inside libwhisper. Cross-platform, no ffmpeg needed.
///
/// Throws when the loaded dylib is pre-0.4.1 (no `crispasr_audio_load`
/// symbol) or when the decoder can't handle the file (returns an error
/// code from the C helper).
DecodedAudio decodeAudioFile(String path, {String? libPath}) {
  final lib = DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
  if (!lib.providesSymbol('crispasr_audio_load')) {
    throw UnsupportedError(
        'Audio decoder not available in the loaded CrispASR library — '
        'rebuild with 0.4.1+ helpers.');
  }

  final load = lib.lookupFunction<
      Int32 Function(Pointer<Utf8>, Pointer<Pointer<Float>>, Pointer<Int32>, Pointer<Int32>),
      int Function(Pointer<Utf8>, Pointer<Pointer<Float>>, Pointer<Int32>, Pointer<Int32>)>(
    'crispasr_audio_load',
  );
  final free = lib.lookupFunction<
      Void Function(Pointer<Float>),
      void Function(Pointer<Float>)>('crispasr_audio_free');

  final pathPtr = path.toNativeUtf8();
  final pcmOut  = calloc<Pointer<Float>>();
  final nOut    = calloc<Int32>();
  final srOut   = calloc<Int32>();

  try {
    final rc = load(pathPtr, pcmOut, nOut, srOut);
    if (rc != 0) {
      throw Exception('crispasr_audio_load failed (code $rc) for $path');
    }
    final ptr = pcmOut.value;
    final n = nOut.value;
    final sr = srOut.value;
    if (ptr == nullptr || n <= 0) {
      throw Exception('Audio decoded to empty buffer: $path');
    }
    // Copy the native float* into a Dart-owned Float32List so we can
    // free the native buffer now and not worry about lifetime.
    final copy = Float32List(n);
    final srcView = ptr.asTypedList(n);
    copy.setAll(0, srcView);
    free(ptr);
    return DecodedAudio(samples: copy, sampleRate: sr > 0 ? sr : 16000);
  } finally {
    calloc.free(pathPtr);
    calloc.free(pcmOut);
    calloc.free(nOut);
    calloc.free(srOut);
  }
}

/// One ASR segment passed in to [diarizeSegments]. Callers fill [t0] and
/// [t1] (seconds) from the upstream transcribe result; the diarizer
/// writes the zero-based speaker index into [speaker] (`-1` means the
/// method had no info to pick).
class DiarizeSegment {
  final double t0;
  final double t1;
  int speaker;
  DiarizeSegment({required this.t0, required this.t1, this.speaker = -1});
}

enum DiarizeMethod {
  /// Stereo only. |L| vs |R| energy per segment, 1.1× margin.
  energy,
  /// Stereo only. TDOA via cross-correlation, ±5 ms search window.
  xcorr,
  /// Mono-friendly. Alternates 0/1 every time the inter-segment gap
  /// exceeds 600 ms.
  vadTurns,
  /// Mono-friendly, ML-based. Runs the GGUF pyannote segmentation net.
  /// Requires [pyannoteModelPath].
  pyannote,
}

/// Assign a speaker index to each of [segs], based on the selected
/// [method], mutating each [DiarizeSegment.speaker] in place.
///
/// `left` is mono PCM for mono-only methods, otherwise the left channel
/// of a stereo pair. When `isStereo` is true, `right` must point at the
/// right channel. All PCM is 16 kHz float32.
///
/// Returns `true` on success. The only failure case is
/// [DiarizeMethod.pyannote] when the GGUF model can't be loaded — all
/// other methods always succeed (they may leave individual segments
/// with `speaker = -1` when they had no information to decide).
///
/// `sliceT0` is the absolute start time (seconds) of the PCM buffer
/// within the original audio, so absolute segment times in [DiarizeSegment]
/// can be mapped back to sample indices.
bool diarizeSegments({
  required List<DiarizeSegment> segs,
  required Float32List left,
  Float32List? right,
  bool isStereo = false,
  DiarizeMethod method = DiarizeMethod.vadTurns,
  String? pyannoteModelPath,
  int nThreads = 4,
  double sliceT0 = 0.0,
  DynamicLibrary? lib,
}) {
  if (segs.isEmpty || left.isEmpty) return true;
  lib ??= DynamicLibrary.open(CrispASR.defaultLibName());

  final n = left.length;
  final leftPtr = calloc<Float>(n);
  for (var i = 0; i < n; i++) leftPtr[i] = left[i];

  Pointer<Float> rightPtr = leftPtr;
  if (isStereo && right != null) {
    rightPtr = calloc<Float>(right.length);
    for (var i = 0; i < right.length; i++) rightPtr[i] = right[i];
  }

  // ABI struct layout must match `crispasr_diarize_seg_abi`:
  // int64 t0_cs, int64 t1_cs, int32 speaker, int32 _pad. 24 bytes each.
  final segsPtr = calloc<Uint8>(24 * segs.length);
  for (var i = 0; i < segs.length; i++) {
    final base = segsPtr.elementAt(i * 24);
    base.cast<Int64>().value = (segs[i].t0 * 100).round();
    base.elementAt(8).cast<Int64>().value = (segs[i].t1 * 100).round();
    base.elementAt(16).cast<Int32>().value = segs[i].speaker;
    base.elementAt(20).cast<Int32>().value = 0;
  }

  // ABI opts: int32 method, int32 n_threads, int64 slice_t0_cs, const char*.
  // 4+4+8+8 = 24 bytes on 64-bit. We write each field explicitly.
  final optsPtr = calloc<Uint8>(24);
  optsPtr.cast<Int32>().value = method.index;
  optsPtr.elementAt(4).cast<Int32>().value = nThreads;
  optsPtr.elementAt(8).cast<Int64>().value = (sliceT0 * 100).round();
  final pathPtr = (method == DiarizeMethod.pyannote &&
          pyannoteModelPath != null &&
          pyannoteModelPath.isNotEmpty)
      ? pyannoteModelPath.toNativeUtf8()
      : nullptr;
  optsPtr.elementAt(16).cast<IntPtr>().value = pathPtr.address;

  final fn = lib.lookupFunction<
      Int32 Function(Pointer<Float>, Pointer<Float>, Int32, Int32,
          Pointer<Uint8>, Int32, Pointer<Uint8>),
      int Function(Pointer<Float>, Pointer<Float>, int, int, Pointer<Uint8>,
          int, Pointer<Uint8>)>('crispasr_diarize_segments_abi');
  final rc = fn(leftPtr, rightPtr, n, isStereo ? 1 : 0, segsPtr, segs.length, optsPtr);

  if (rc == 0) {
    for (var i = 0; i < segs.length; i++) {
      segs[i].speaker = segsPtr.elementAt(i * 24 + 16).cast<Int32>().value;
    }
  }

  calloc.free(leftPtr);
  if (rightPtr != leftPtr) calloc.free(rightPtr);
  calloc.free(segsPtr);
  calloc.free(optsPtr);
  if (pathPtr != nullptr) calloc.free(pathPtr);

  return rc == 0;
}

/// A registry entry returned by [registryLookup] / [registryLookupByFilename].
class RegistryEntry {
  final String filename;
  final String url;
  final String approxSize;
  const RegistryEntry({
    required this.filename,
    required this.url,
    required this.approxSize,
  });
}

/// Look up the canonical GGUF for a backend (whisper, parakeet, canary,
/// voxtral, voxtral4b, granite, qwen3, cohere, wav2vec2). Returns null
/// on miss.
RegistryEntry? registryLookup(String backend, {DynamicLibrary? lib}) =>
    _registryCall('crispasr_registry_lookup_abi', backend, lib);

/// Look up the canonical GGUF by filename (exact match or substring).
/// Useful when a user-supplied filename isn't cached yet and we want to
/// suggest a download URL.
RegistryEntry? registryLookupByFilename(String filename, {DynamicLibrary? lib}) =>
    _registryCall('crispasr_registry_lookup_by_filename_abi', filename, lib);

RegistryEntry? _registryCall(String sym, String key, DynamicLibrary? lib) {
  if (key.isEmpty) return null;
  lib ??= DynamicLibrary.open(CrispASR.defaultLibName());
  final keyPtr = key.toNativeUtf8();
  final fnBuf = calloc<Uint8>(256);
  final urlBuf = calloc<Uint8>(512);
  final sizeBuf = calloc<Uint8>(32);
  final fn = lib.lookupFunction<
      Int32 Function(Pointer<Utf8>, Pointer<Uint8>, Int32, Pointer<Uint8>, Int32,
          Pointer<Uint8>, Int32),
      int Function(Pointer<Utf8>, Pointer<Uint8>, int, Pointer<Uint8>, int,
          Pointer<Uint8>, int)>(sym);
  final rc = fn(keyPtr, fnBuf, 256, urlBuf, 512, sizeBuf, 32);
  RegistryEntry? out;
  if (rc == 0) {
    out = RegistryEntry(
      filename: fnBuf.cast<Utf8>().toDartString(),
      url: urlBuf.cast<Utf8>().toDartString(),
      approxSize: sizeBuf.cast<Utf8>().toDartString(),
    );
  }
  calloc.free(keyPtr);
  calloc.free(fnBuf);
  calloc.free(urlBuf);
  calloc.free(sizeBuf);
  return out;
}

/// Download [filename] from [url] into the CrispASR cache (or return
/// the cached path if it's already present). Returns the absolute path
/// or null on failure.
///
/// [cacheDirOverride] defaults to the platform cache dir
/// (`~/.cache/crispasr` on POSIX). Flutter / mobile callers that want
/// a sandbox-friendly directory should pass their own.
String? cacheEnsureFile(
  String filename,
  String url, {
  bool quiet = false,
  String? cacheDirOverride,
  DynamicLibrary? lib,
}) {
  if (filename.isEmpty || url.isEmpty) return null;
  lib ??= DynamicLibrary.open(CrispASR.defaultLibName());
  final fnPtr = filename.toNativeUtf8();
  final urlPtr = url.toNativeUtf8();
  final ovPtr = (cacheDirOverride ?? '').toNativeUtf8();
  final outBuf = calloc<Uint8>(2048);
  final fn = lib.lookupFunction<
      Int32 Function(Pointer<Utf8>, Pointer<Utf8>, Int32, Pointer<Utf8>,
          Pointer<Uint8>, Int32),
      int Function(Pointer<Utf8>, Pointer<Utf8>, int, Pointer<Utf8>,
          Pointer<Uint8>, int)>('crispasr_cache_ensure_file_abi');
  final rc = fn(fnPtr, urlPtr, quiet ? 1 : 0, ovPtr, outBuf, 2048);
  final path = rc == 0 ? outBuf.cast<Utf8>().toDartString() : null;
  calloc.free(fnPtr);
  calloc.free(urlPtr);
  calloc.free(ovPtr);
  calloc.free(outBuf);
  return path;
}

/// Return the CrispASR cache directory (creating it if missing).
String? cacheDir({String? override, DynamicLibrary? lib}) {
  lib ??= DynamicLibrary.open(CrispASR.defaultLibName());
  final ovPtr = (override ?? '').toNativeUtf8();
  final outBuf = calloc<Uint8>(2048);
  final fn = lib.lookupFunction<
      Int32 Function(Pointer<Utf8>, Pointer<Uint8>, Int32),
      int Function(Pointer<Utf8>, Pointer<Uint8>, int)>('crispasr_cache_dir_abi');
  final rc = fn(ovPtr, outBuf, 2048);
  final dir = rc == 0 ? outBuf.cast<Utf8>().toDartString() : null;
  calloc.free(ovPtr);
  calloc.free(outBuf);
  return dir;
}

/// One word + centisecond timings returned by [alignWords].
class AlignedWord {
  final String text;
  final double start; // seconds
  final double end;
  const AlignedWord({required this.text, required this.start, required this.end});
}

/// CTC / forced-alignment word timings for a transcript + audio pair.
///
/// `alignerModel` picks the backend by filename convention: any path
/// containing "forced-aligner" / "qwen3-fa" / "qwen3-forced" routes to
/// the Qwen3-ForcedAligner path; everything else goes through
/// canary-ctc-aligner.
///
/// `tOffset` (seconds) is added to every word's start/end so the
/// returned timings are absolute against the original audio.
List<AlignedWord> alignWords({
  required String alignerModel,
  required String transcript,
  required Float32List pcm,
  double tOffset = 0.0,
  int nThreads = 4,
  DynamicLibrary? lib,
}) {
  if (alignerModel.isEmpty || transcript.isEmpty || pcm.isEmpty) {
    return const [];
  }
  lib ??= DynamicLibrary.open(CrispASR.defaultLibName());

  final samples = calloc<Float>(pcm.length);
  for (var i = 0; i < pcm.length; i++) samples[i] = pcm[i];
  final modelPtr = alignerModel.toNativeUtf8();
  final txtPtr = transcript.toNativeUtf8();

  final fn = lib.lookupFunction<
      Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>, Pointer<Float>,
          Int32, Int64, Int32),
      Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>, Pointer<Float>,
          int, int, int)>('crispasr_align_words_abi');
  final res = fn(modelPtr, txtPtr, samples, pcm.length,
      (tOffset * 100).round(), nThreads);

  calloc.free(samples);
  calloc.free(modelPtr);
  calloc.free(txtPtr);

  if (res == nullptr) return const [];

  final nFn = lib.lookupFunction<Int32 Function(Pointer<Void>),
      int Function(Pointer<Void>)>('crispasr_align_result_n_words');
  final textFn = lib.lookupFunction<
      Pointer<Utf8> Function(Pointer<Void>, Int32),
      Pointer<Utf8> Function(Pointer<Void>, int)>('crispasr_align_result_word_text');
  final t0Fn = lib.lookupFunction<Int64 Function(Pointer<Void>, Int32),
      int Function(Pointer<Void>, int)>('crispasr_align_result_word_t0');
  final t1Fn = lib.lookupFunction<Int64 Function(Pointer<Void>, Int32),
      int Function(Pointer<Void>, int)>('crispasr_align_result_word_t1');
  final freeFn = lib.lookupFunction<Void Function(Pointer<Void>),
      void Function(Pointer<Void>)>('crispasr_align_result_free');

  final n = nFn(res);
  final out = <AlignedWord>[];
  for (var i = 0; i < n; i++) {
    final tp = textFn(res, i);
    final t = tp == nullptr ? '' : tp.toDartString();
    out.add(AlignedWord(
      text: t,
      start: t0Fn(res, i) / 100.0,
      end: t1Fn(res, i) / 100.0,
    ));
  }
  freeFn(res);
  return out;
}

/// Language identification result from [detectLanguagePcm]. `langCode`
/// is an ISO 639-1 code ("en", "de", …) or empty on failure.
class LidResult {
  final String langCode;
  final double confidence;
  const LidResult({required this.langCode, required this.confidence});
  bool get isEmpty => langCode.isEmpty;
}

enum LidMethod {
  /// Whisper encoder + language head on a multilingual ggml-*.bin model.
  whisper,
  /// GGUF-packed Silero 95-language classifier.
  silero,
}

/// Run LID on a 16 kHz mono [pcm] buffer using the method in
/// [method]. [modelPath] must point to a concrete file on disk (the
/// whisper ggml-*.bin or the Silero GGUF); auto-download is the
/// caller's responsibility.
///
/// Returns an empty [LidResult] on failure.
LidResult detectLanguagePcm({
  required Float32List pcm,
  required LidMethod method,
  required String modelPath,
  int nThreads = 4,
  bool useGpu = false,
  int gpuDevice = 0,
  bool flashAttn = true,
  DynamicLibrary? lib,
}) {
  if (pcm.isEmpty || modelPath.isEmpty) {
    return const LidResult(langCode: '', confidence: -1);
  }
  lib ??= DynamicLibrary.open(CrispASR.defaultLibName());

  final samples = calloc<Float>(pcm.length);
  for (var i = 0; i < pcm.length; i++) samples[i] = pcm[i];
  final pathPtr = modelPath.toNativeUtf8();
  final outBuf = calloc<Uint8>(16);
  final outConf = calloc<Float>();

  final fn = lib.lookupFunction<
      Int32 Function(Pointer<Float>, Int32, Int32, Pointer<Utf8>, Int32, Int32,
          Int32, Int32, Pointer<Uint8>, Int32, Pointer<Float>),
      int Function(Pointer<Float>, int, int, Pointer<Utf8>, int, int, int, int,
          Pointer<Uint8>, int, Pointer<Float>)>('crispasr_detect_language_pcm');
  final rc = fn(samples, pcm.length, method.index, pathPtr, nThreads,
      useGpu ? 1 : 0, gpuDevice, flashAttn ? 1 : 0, outBuf, 16, outConf);

  String code = '';
  double conf = -1;
  if (rc == 0) {
    code = outBuf.cast<Utf8>().toDartString();
    conf = outConf.value;
  }

  calloc.free(samples);
  calloc.free(pathPtr);
  calloc.free(outBuf);
  calloc.free(outConf);

  return LidResult(langCode: code, confidence: conf);
}

/// Tunables for [CrispasrSession.transcribeVad]. Field names and defaults
/// mirror whisper.cpp's `whisper_vad_params` plus the max-chunk fallback
/// the shared library uses to bound encoder cost on long audio.
class SessionVadOptions {
  /// Silero VAD decision threshold (0..1). Higher = fewer / shorter
  /// speech regions. whisper.cpp ships 0.5.
  final double threshold;
  /// Shortest run of voiced frames (ms) kept as a speech segment.
  final int minSpeechDurationMs;
  /// Shortest silence (ms) needed to split one segment from the next.
  final int minSilenceDurationMs;
  /// Extra context padding (ms) added on each side of every segment.
  final int speechPadMs;
  /// Maximum merged-segment length (seconds). Any speech slice longer
  /// than this is split into roughly equal sub-slices so O(T²) backends
  /// don't blow up on a 10-minute continuous lecture. 0 disables the
  /// split.
  final int chunkSeconds;
  /// Threads used for Silero VAD inference. The ASR backend uses its
  /// own thread count configured at session open time.
  final int nThreads;

  const SessionVadOptions({
    this.threshold = 0.5,
    this.minSpeechDurationMs = 250,
    this.minSilenceDurationMs = 100,
    this.speechPadMs = 30,
    this.chunkSeconds = 30,
    this.nThreads = 4,
  });
}

/// One decoded segment from [CrispasrSession.transcribe]. Similar to the
/// Whisper-specific [Segment] but produced by a backend-agnostic code path.
class SessionSegment {
  final String text;
  final double start; // seconds (centiseconds / 100 on the C side)
  final double end;
  final List<Word> words;
  const SessionSegment({
    required this.text,
    required this.start,
    required this.end,
    this.words = const [],
  });
  @override
  String toString() =>
      '[${start.toStringAsFixed(1)}-${end.toStringAsFixed(1)}s] $text';
}

/// One "commit" from a streaming session — the latest concatenated text
/// that whisper produced for the current rolling window, plus its absolute
/// start/end time in the live audio stream.
class StreamingUpdate {
  /// Concatenated text of the last decode. Overwritten on every new
  /// [StreamingSession.feed] / [StreamingSession.flush] cycle that produces
  /// output, so caller diffs against previous text if they want an
  /// append-only stream.
  final String text;
  /// Start of the decoded window, in seconds from the beginning of the
  /// live stream.
  final double start;
  /// End of the decoded window, in seconds from the beginning of the
  /// live stream.
  final double end;
  /// Monotonic decode counter — useful to distinguish "new decode, same
  /// text" from "stale text replayed".
  final int counter;

  const StreamingUpdate({
    required this.text,
    required this.start,
    required this.end,
    required this.counter,
  });

  @override
  String toString() =>
      '[${start.toStringAsFixed(1)}-${end.toStringAsFixed(1)}s] $text';
}

/// A speech span returned by [CrispASR.vad].
class VadSpan {
  final double start; // seconds
  final double end;   // seconds

  const VadSpan({required this.start, required this.end});

  double get duration => end - start;

  @override
  String toString() =>
      'VadSpan(${start.toStringAsFixed(2)}s → ${end.toStringAsFixed(2)}s)';
}

/// Options controlling a call to [CrispASR.transcribePcm].
///
/// Maps to the most commonly-set fields of `whisper_full_params`. Anything
/// not listed here keeps whisper's default.
class TranscribeOptions {
  /// Sampling strategy: 0 = GREEDY, 1 = BEAM_SEARCH.
  final int strategy;
  /// ISO-639 code, or null to keep the default (usually "auto").
  final String? language;
  /// If true, whisper translates the audio into English.
  final bool translate;
  /// Let whisper auto-detect the language before decoding. Ignored when
  /// [language] is set and not "auto".
  final bool detectLanguage;
  /// Populate [Segment.words] with per-token timing.
  final bool wordTimestamps;
  /// Maximum tokens per segment. 0 = whisper default.
  final int maxLen;
  /// Split segments on word boundaries when [maxLen] is set.
  final bool splitOnWord;
  /// Thread count. 0 = whisper default (usually 4).
  final int nThreads;
  /// An initial text prompt to condition the decoder on.
  final String? initialPrompt;
  /// Silence the library's own stdout output.
  final bool silent;

  // --- VAD (Silero, built into whisper.cpp). Set [vad] + [vadModelPath]
  // to have whisper skip silent regions automatically; the rest are
  // fine-tuning knobs. ---
  final bool vad;
  final String? vadModelPath;
  final double vadThreshold;
  final int vadMinSpeechMs;
  final int vadMinSilenceMs;

  /// tinydiarize speaker-turn markers. Requires a whisper .en.tdrz
  /// finetune; output will contain `[SPEAKER_TURN]` tokens the host can
  /// split segments on.
  final bool tdrz;

  const TranscribeOptions({
    this.strategy = 0,
    this.language,
    this.translate = false,
    this.detectLanguage = false,
    this.wordTimestamps = false,
    this.maxLen = 0,
    this.splitOnWord = false,
    this.nThreads = 0,
    this.initialPrompt,
    this.silent = true,
    this.vad = false,
    this.vadModelPath,
    this.vadThreshold = 0.5,
    this.vadMinSpeechMs = 250,
    this.vadMinSilenceMs = 100,
    this.tdrz = false,
  });
}

// =====================================================================
// FFI typedefs — originals from 0.1.0 ...
// =====================================================================
typedef _WhisperInitNative = Pointer<Void> Function(Pointer<Utf8>, Pointer<Void>);
typedef _WhisperInit       = Pointer<Void> Function(Pointer<Utf8>, Pointer<Void>);

typedef _VoidPtr_C = Void Function(Pointer<Void>);
typedef _VoidPtr   = void Function(Pointer<Void>);

typedef _WhisperFullNative = Int32 Function(Pointer<Void>, Pointer<Void>, Pointer<Float>, Int32);
typedef _WhisperFull       = int  Function(Pointer<Void>, Pointer<Void>, Pointer<Float>, int);

typedef _DefaultParamsNative = Pointer<Void> Function(Int32);
typedef _DefaultParams       = Pointer<Void> Function(int);

typedef _DefaultCtxParamsNative = Pointer<Void> Function();
typedef _DefaultCtxParams       = Pointer<Void> Function();

typedef _IntPtr_C = Int32 Function(Pointer<Void>);
typedef _IntPtr   = int   Function(Pointer<Void>);

typedef _GetTextNative = Pointer<Utf8> Function(Pointer<Void>, Int32);
typedef _GetText       = Pointer<Utf8> Function(Pointer<Void>, int);

typedef _GetT0Native = Int64 Function(Pointer<Void>, Int32);
typedef _GetT0       = int   Function(Pointer<Void>, int);

typedef _GetNSPNative = Float  Function(Pointer<Void>, Int32);
typedef _GetNSP       = double Function(Pointer<Void>, int);

// =====================================================================
// ... new in 0.2.0: token / lang-detect / VAD / param setters
// =====================================================================
typedef _ParamsSetBoolNative = Void Function(Pointer<Void>, Int32);
typedef _ParamsSetBool       = void Function(Pointer<Void>, int);

typedef _ParamsSetStringNative = Void Function(Pointer<Void>, Pointer<Utf8>);
typedef _ParamsSetString       = void Function(Pointer<Void>, Pointer<Utf8>);

typedef _ParamsSetIntNative = Void Function(Pointer<Void>, Int32);
typedef _ParamsSetInt       = void Function(Pointer<Void>, int);

typedef _ParamsSetFloatNative = Void Function(Pointer<Void>, Float);
typedef _ParamsSetFloat       = void Function(Pointer<Void>, double);

typedef _FullNTokensNative = Int32 Function(Pointer<Void>, Int32);
typedef _FullNTokens       = int   Function(Pointer<Void>, int);

typedef _TokenTextNative = Pointer<Utf8> Function(Pointer<Void>, Int32, Int32);
typedef _TokenText       = Pointer<Utf8> Function(Pointer<Void>, int, int);

typedef _TokenT0Native = Int64 Function(Pointer<Void>, Int32, Int32);
typedef _TokenT0       = int   Function(Pointer<Void>, int, int);

typedef _TokenPNative = Float  Function(Pointer<Void>, Int32, Int32);
typedef _TokenP       = double Function(Pointer<Void>, int, int);

typedef _DetectLangNative = Float Function(
    Pointer<Void>, Pointer<Float>, Int32, Int32, Pointer<Utf8>, Int32);
typedef _DetectLang = double Function(
    Pointer<Void>, Pointer<Float>, int, int, Pointer<Utf8>, int);

typedef _VadSegmentsNative = Int32 Function(
    Pointer<Utf8>, Pointer<Float>, Int32, Int32, Float, Int32, Int32, Int32,
    Uint8, Pointer<Pointer<Float>>);
typedef _VadSegments = int Function(
    Pointer<Utf8>, Pointer<Float>, int, int, double, int, int, int,
    int, Pointer<Pointer<Float>>);

typedef _VadFreeNative = Void Function(Pointer<Float>);
typedef _VadFree       = void Function(Pointer<Float>);

typedef _LangStrNative = Pointer<Utf8> Function(Int32);
typedef _LangStr       = Pointer<Utf8> Function(int);

typedef _LangIdNative = Int32 Function(Pointer<Utf8>);
typedef _LangId       = int   Function(Pointer<Utf8>);

typedef _IntNative  = Int32 Function();
typedef _IntFn      = int   Function();

// Streaming helpers (0.3.0).
typedef _StreamOpenNative = Pointer<Void> Function(
    Pointer<Void>, Int32, Int32, Int32, Int32, Pointer<Utf8>, Int32);
typedef _StreamOpen = Pointer<Void> Function(
    Pointer<Void>, int, int, int, int, Pointer<Utf8>, int);

typedef _StreamFeedNative = Int32 Function(Pointer<Void>, Pointer<Float>, Int32);
typedef _StreamFeed       = int   Function(Pointer<Void>, Pointer<Float>, int);

typedef _StreamFlushNative = Int32 Function(Pointer<Void>);
typedef _StreamFlush       = int   Function(Pointer<Void>);

typedef _StreamGetTextNative = Int32 Function(
    Pointer<Void>, Pointer<Utf8>, Int32, Pointer<Double>, Pointer<Double>, Pointer<Int64>);
typedef _StreamGetText = int Function(
    Pointer<Void>, Pointer<Utf8>, int, Pointer<Double>, Pointer<Double>, Pointer<Int64>);

typedef _StreamCloseNative = Void Function(Pointer<Void>);
typedef _StreamClose       = void Function(Pointer<Void>);

/// On-device speech recognition model.
///
/// ```dart
/// final model = CrispASR('ggml-base.en.bin');
/// final segments = model.transcribePcm(pcmFloat32);
/// for (final seg in segments) {
///   print(seg);
/// }
/// model.dispose();
/// ```
class CrispASR {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  // 0.1.0 FFI handles
  late final _WhisperFull     _full;
  late final _VoidPtr         _free;
  late final _DefaultParams   _defaultParams;
  late final _IntPtr          _nSegments;
  late final _GetText         _getText;
  late final _GetT0           _getT0;
  late final _GetT0           _getT1;
  late final _GetNSP          _getNSP;
  late final _VoidPtr         _freeParams;

  // 0.2.0 additions — looked up lazily / tolerantly so a v0.1.0 dylib
  // loaded at runtime still works (minus the new features).
  _ParamsSetString? _paramsSetLanguage;
  _ParamsSetString? _paramsSetInitialPrompt;
  _ParamsSetBool?   _paramsSetTranslate;
  _ParamsSetBool?   _paramsSetDetectLanguage;
  _ParamsSetBool?   _paramsSetTokenTimestamps;
  _ParamsSetInt?    _paramsSetNThreads;
  _ParamsSetInt?    _paramsSetMaxLen;
  _ParamsSetBool?   _paramsSetSplitOnWord;
  _ParamsSetBool?   _paramsSetPrintRealtime;
  _ParamsSetBool?   _paramsSetPrintProgress;
  _ParamsSetBool?   _paramsSetPrintTimestamps;
  _ParamsSetBool?   _paramsSetPrintSpecial;

  // 0.4.2 additions — VAD + tinydiarize setters on whisper_full_params.
  _ParamsSetBool?   _paramsSetVad;
  _ParamsSetString? _paramsSetVadModelPath;
  _ParamsSetFloat?  _paramsSetVadThreshold;
  _ParamsSetInt?    _paramsSetVadMinSpeechMs;
  _ParamsSetInt?    _paramsSetVadMinSilenceMs;
  _ParamsSetBool?   _paramsSetTdrz;

  _FullNTokens? _fullNTokens;
  _TokenText?   _tokenText;
  _TokenT0?     _tokenT0;
  _TokenT0?     _tokenT1;
  _TokenP?      _tokenP;

  _DetectLang?  _detectLang;
  _VadSegments? _vadSegments;
  _VadFree?     _vadFree;

  _LangStr?     _langStr;
  _LangId?      _langId;
  _IntFn?       _langMaxId;

  _StreamOpen?    _streamOpen;
  _StreamFeed?    _streamFeed;
  _StreamFlush?   _streamFlush;
  _StreamGetText? _streamGetText;
  _StreamClose?   _streamClose;

  bool get supportsExtended => _detectLang != null;
  bool get supportsStreaming => _streamOpen != null;

  CrispASR(String modelPath, {String? libPath}) {
    _lib = DynamicLibrary.open(libPath ?? _findLib());

    final init = _lib.lookupFunction<_WhisperInitNative, _WhisperInit>(
        'whisper_init_from_file_with_params');
    _free          = _lib.lookupFunction<_VoidPtr_C, _VoidPtr>('whisper_free');
    _full          = _lib.lookupFunction<_WhisperFullNative, _WhisperFull>('whisper_full');
    _defaultParams = _lib.lookupFunction<_DefaultParamsNative, _DefaultParams>(
        'whisper_full_default_params_by_ref');
    _nSegments = _lib.lookupFunction<_IntPtr_C, _IntPtr>('whisper_full_n_segments');
    _getText   = _lib.lookupFunction<_GetTextNative, _GetText>('whisper_full_get_segment_text');
    _getT0     = _lib.lookupFunction<_GetT0Native, _GetT0>('whisper_full_get_segment_t0');
    _getT1     = _lib.lookupFunction<_GetT0Native, _GetT0>('whisper_full_get_segment_t1');
    _getNSP    = _lib.lookupFunction<_GetNSPNative, _GetNSP>('whisper_full_get_segment_no_speech_prob');
    _freeParams = _lib.lookupFunction<_VoidPtr_C, _VoidPtr>('whisper_free_params');

    final ctxDefault = _lib.lookupFunction<_DefaultCtxParamsNative, _DefaultCtxParams>(
        'whisper_context_default_params_by_ref')();
    final pathPtr = modelPath.toNativeUtf8();
    _ctx = init(pathPtr, ctxDefault);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load model: $modelPath');
    }

    _tryBindExtended();
  }

  /// Lookup the 0.2.0 additions. Any missing symbol leaves the matching
  /// feature off, so a susurrus-flutter build using a v0.1.0 dylib keeps
  /// transcribing.
  ///
  /// Note: `lookupFunction<T, R>` needs its type arguments at the call site
  /// to verify `NativeFunction<T>` — routing them through a generic
  /// helper blows Dart FFI's type check. We use `providesSymbol` instead
  /// so a missing symbol silently skips the feature.
  void _tryBindExtended() {
    if (_lib.providesSymbol('crispasr_params_set_language')) {
      _paramsSetLanguage = _lib.lookupFunction<_ParamsSetStringNative, _ParamsSetString>('crispasr_params_set_language');
    }
    if (_lib.providesSymbol('crispasr_params_set_initial_prompt')) {
      _paramsSetInitialPrompt = _lib.lookupFunction<_ParamsSetStringNative, _ParamsSetString>('crispasr_params_set_initial_prompt');
    }
    if (_lib.providesSymbol('crispasr_params_set_translate')) {
      _paramsSetTranslate = _lib.lookupFunction<_ParamsSetBoolNative, _ParamsSetBool>('crispasr_params_set_translate');
    }
    if (_lib.providesSymbol('crispasr_params_set_detect_language')) {
      _paramsSetDetectLanguage = _lib.lookupFunction<_ParamsSetBoolNative, _ParamsSetBool>('crispasr_params_set_detect_language');
    }
    if (_lib.providesSymbol('crispasr_params_set_token_timestamps')) {
      _paramsSetTokenTimestamps = _lib.lookupFunction<_ParamsSetBoolNative, _ParamsSetBool>('crispasr_params_set_token_timestamps');
    }
    if (_lib.providesSymbol('crispasr_params_set_n_threads')) {
      _paramsSetNThreads = _lib.lookupFunction<_ParamsSetIntNative, _ParamsSetInt>('crispasr_params_set_n_threads');
    }
    if (_lib.providesSymbol('crispasr_params_set_max_len')) {
      _paramsSetMaxLen = _lib.lookupFunction<_ParamsSetIntNative, _ParamsSetInt>('crispasr_params_set_max_len');
    }
    if (_lib.providesSymbol('crispasr_params_set_split_on_word')) {
      _paramsSetSplitOnWord = _lib.lookupFunction<_ParamsSetBoolNative, _ParamsSetBool>('crispasr_params_set_split_on_word');
    }
    if (_lib.providesSymbol('crispasr_params_set_print_realtime')) {
      _paramsSetPrintRealtime = _lib.lookupFunction<_ParamsSetBoolNative, _ParamsSetBool>('crispasr_params_set_print_realtime');
    }
    if (_lib.providesSymbol('crispasr_params_set_print_progress')) {
      _paramsSetPrintProgress = _lib.lookupFunction<_ParamsSetBoolNative, _ParamsSetBool>('crispasr_params_set_print_progress');
    }
    if (_lib.providesSymbol('crispasr_params_set_print_timestamps')) {
      _paramsSetPrintTimestamps = _lib.lookupFunction<_ParamsSetBoolNative, _ParamsSetBool>('crispasr_params_set_print_timestamps');
    }
    if (_lib.providesSymbol('crispasr_params_set_print_special')) {
      _paramsSetPrintSpecial = _lib.lookupFunction<_ParamsSetBoolNative, _ParamsSetBool>('crispasr_params_set_print_special');
    }

    // 0.4.2 — VAD + tdrz.
    if (_lib.providesSymbol('crispasr_params_set_vad')) {
      _paramsSetVad = _lib.lookupFunction<_ParamsSetBoolNative, _ParamsSetBool>('crispasr_params_set_vad');
    }
    if (_lib.providesSymbol('crispasr_params_set_vad_model_path')) {
      _paramsSetVadModelPath = _lib.lookupFunction<_ParamsSetStringNative, _ParamsSetString>('crispasr_params_set_vad_model_path');
    }
    if (_lib.providesSymbol('crispasr_params_set_vad_threshold')) {
      _paramsSetVadThreshold = _lib.lookupFunction<_ParamsSetFloatNative, _ParamsSetFloat>('crispasr_params_set_vad_threshold');
    }
    if (_lib.providesSymbol('crispasr_params_set_vad_min_speech_ms')) {
      _paramsSetVadMinSpeechMs = _lib.lookupFunction<_ParamsSetIntNative, _ParamsSetInt>('crispasr_params_set_vad_min_speech_ms');
    }
    if (_lib.providesSymbol('crispasr_params_set_vad_min_silence_ms')) {
      _paramsSetVadMinSilenceMs = _lib.lookupFunction<_ParamsSetIntNative, _ParamsSetInt>('crispasr_params_set_vad_min_silence_ms');
    }
    if (_lib.providesSymbol('crispasr_params_set_tdrz')) {
      _paramsSetTdrz = _lib.lookupFunction<_ParamsSetBoolNative, _ParamsSetBool>('crispasr_params_set_tdrz');
    }

    if (_lib.providesSymbol('whisper_full_n_tokens')) {
      _fullNTokens = _lib.lookupFunction<_FullNTokensNative, _FullNTokens>('whisper_full_n_tokens');
    }
    if (_lib.providesSymbol('whisper_full_get_token_text')) {
      _tokenText = _lib.lookupFunction<_TokenTextNative, _TokenText>('whisper_full_get_token_text');
    }
    if (_lib.providesSymbol('crispasr_token_t0')) {
      _tokenT0 = _lib.lookupFunction<_TokenT0Native, _TokenT0>('crispasr_token_t0');
    }
    if (_lib.providesSymbol('crispasr_token_t1')) {
      _tokenT1 = _lib.lookupFunction<_TokenT0Native, _TokenT0>('crispasr_token_t1');
    }
    if (_lib.providesSymbol('crispasr_token_p')) {
      _tokenP = _lib.lookupFunction<_TokenPNative, _TokenP>('crispasr_token_p');
    }

    if (_lib.providesSymbol('crispasr_detect_language')) {
      _detectLang = _lib.lookupFunction<_DetectLangNative, _DetectLang>('crispasr_detect_language');
    }
    if (_lib.providesSymbol('crispasr_vad_segments')) {
      _vadSegments = _lib.lookupFunction<_VadSegmentsNative, _VadSegments>('crispasr_vad_segments');
    }
    if (_lib.providesSymbol('crispasr_vad_free')) {
      _vadFree = _lib.lookupFunction<_VadFreeNative, _VadFree>('crispasr_vad_free');
    }

    if (_lib.providesSymbol('whisper_lang_str')) {
      _langStr = _lib.lookupFunction<_LangStrNative, _LangStr>('whisper_lang_str');
    }
    if (_lib.providesSymbol('whisper_lang_id')) {
      _langId = _lib.lookupFunction<_LangIdNative, _LangId>('whisper_lang_id');
    }
    if (_lib.providesSymbol('whisper_lang_max_id')) {
      _langMaxId = _lib.lookupFunction<_IntNative, _IntFn>('whisper_lang_max_id');
    }

    if (_lib.providesSymbol('crispasr_stream_open')) {
      _streamOpen = _lib.lookupFunction<_StreamOpenNative, _StreamOpen>('crispasr_stream_open');
    }
    if (_lib.providesSymbol('crispasr_stream_feed')) {
      _streamFeed = _lib.lookupFunction<_StreamFeedNative, _StreamFeed>('crispasr_stream_feed');
    }
    if (_lib.providesSymbol('crispasr_stream_flush')) {
      _streamFlush = _lib.lookupFunction<_StreamFlushNative, _StreamFlush>('crispasr_stream_flush');
    }
    if (_lib.providesSymbol('crispasr_stream_get_text')) {
      _streamGetText = _lib.lookupFunction<_StreamGetTextNative, _StreamGetText>('crispasr_stream_get_text');
    }
    if (_lib.providesSymbol('crispasr_stream_close')) {
      _streamClose = _lib.lookupFunction<_StreamCloseNative, _StreamClose>('crispasr_stream_close');
    }
  }

  /// Transcribe raw PCM audio (float32, mono, 16 kHz).
  ///
  /// Accepts either the legacy [strategy] int (backward compatible with
  /// 0.1.0) or a full [TranscribeOptions] object.
  List<Segment> transcribePcm(
    Float32List pcm, {
    int strategy = 0,
    TranscribeOptions? options,
  }) {
    _checkDisposed();

    final opts = options ?? TranscribeOptions(strategy: strategy);

    // Copy the PCM into a C-visible buffer.
    final samples = calloc<Float>(pcm.length);
    for (var i = 0; i < pcm.length; i++) {
      samples[i] = pcm[i];
    }

    final params = _defaultParams(opts.strategy);
    Pointer<Utf8>? langPtr;
    Pointer<Utf8>? promptPtr;
    Pointer<Utf8>? vadPathPtr;

    try {
      // Apply every option we have a setter for. Older dylibs skip these.
      if (opts.silent) {
        _paramsSetPrintRealtime?.call(params, 0);
        _paramsSetPrintProgress?.call(params, 0);
        _paramsSetPrintTimestamps?.call(params, 0);
        _paramsSetPrintSpecial?.call(params, 0);
      }
      if (opts.nThreads > 0) _paramsSetNThreads?.call(params, opts.nThreads);
      if (opts.language != null) {
        langPtr = opts.language!.toNativeUtf8();
        _paramsSetLanguage?.call(params, langPtr);
      }
      if (opts.translate) _paramsSetTranslate?.call(params, 1);
      if (opts.detectLanguage) _paramsSetDetectLanguage?.call(params, 1);
      if (opts.wordTimestamps) _paramsSetTokenTimestamps?.call(params, 1);
      if (opts.maxLen > 0) _paramsSetMaxLen?.call(params, opts.maxLen);
      if (opts.splitOnWord) _paramsSetSplitOnWord?.call(params, 1);
      if (opts.initialPrompt != null) {
        promptPtr = opts.initialPrompt!.toNativeUtf8();
        _paramsSetInitialPrompt?.call(params, promptPtr);
      }
      if (opts.vad) {
        _paramsSetVad?.call(params, 1);
        _paramsSetVadThreshold?.call(params, opts.vadThreshold);
        _paramsSetVadMinSpeechMs?.call(params, opts.vadMinSpeechMs);
        _paramsSetVadMinSilenceMs?.call(params, opts.vadMinSilenceMs);
        if (opts.vadModelPath != null && opts.vadModelPath!.isNotEmpty) {
          vadPathPtr = opts.vadModelPath!.toNativeUtf8();
          _paramsSetVadModelPath?.call(params, vadPathPtr);
        }
      }
      if (opts.tdrz) _paramsSetTdrz?.call(params, 1);

      final ret = _full(_ctx, params, samples, pcm.length);
      if (ret != 0) throw Exception('Transcription failed (error $ret)');

      return _collectSegments(wantWords: opts.wordTimestamps);
    } finally {
      _freeParams(params);
      calloc.free(samples);
      if (langPtr != null) calloc.free(langPtr);
      if (promptPtr != null) calloc.free(promptPtr);
      if (vadPathPtr != null) calloc.free(vadPathPtr);
    }
  }

  List<Segment> _collectSegments({required bool wantWords}) {
    final n = _nSegments(_ctx);
    final segments = <Segment>[];
    for (var i = 0; i < n; i++) {
      final textPtr = _getText(_ctx, i);
      final text = textPtr == nullptr ? '' : textPtr.toDartString();
      final t0 = _getT0(_ctx, i) / 100.0;
      final t1 = _getT1(_ctx, i) / 100.0;
      final nsp = _getNSP(_ctx, i);

      final words = <Word>[];
      if (wantWords &&
          _fullNTokens != null &&
          _tokenText != null &&
          _tokenT0 != null &&
          _tokenT1 != null &&
          _tokenP != null) {
        final nTokens = _fullNTokens!(_ctx, i);
        for (var k = 0; k < nTokens; k++) {
          final tp = _tokenText!(_ctx, i, k);
          final tok = tp == nullptr ? '' : tp.toDartString();
          if (tok.isEmpty) continue;
          // Skip the special-token brackets whisper emits inline.
          if (tok.startsWith('[_') || tok.startsWith('<|')) continue;
          words.add(Word(
            text: tok,
            start: _tokenT0!(_ctx, i, k) / 100.0,
            end:   _tokenT1!(_ctx, i, k) / 100.0,
            p:     _tokenP!(_ctx, i, k),
          ));
        }
      }

      segments.add(Segment(
        text: text,
        start: t0,
        end: t1,
        noSpeechProb: nsp,
        words: words,
      ));
    }
    return segments;
  }

  /// Auto-detect the spoken language of [pcm] without running a full
  /// decode. Returns an empty [LanguageDetection.code] when the extended
  /// helpers aren't available (i.e. the loaded dylib is < 0.2.0).
  LanguageDetection detectLanguage(
    Float32List pcm, {
    int nThreads = 4,
  }) {
    _checkDisposed();
    if (_detectLang == null) {
      return const LanguageDetection(code: '', probability: -1.0);
    }

    final samples = calloc<Float>(pcm.length);
    for (var i = 0; i < pcm.length; i++) {
      samples[i] = pcm[i];
    }
    // `Utf8` isn't a SizedNativeType in newer Dart FFI — allocate the raw
    // byte buffer and cast when we hand it to the C helper.
    final outBuf = calloc<Uint8>(16);
    final outCode = outBuf.cast<Utf8>();

    try {
      final p = _detectLang!(_ctx, samples, pcm.length, nThreads, outCode, 16);
      final code = p >= 0 ? outCode.toDartString() : '';
      return LanguageDetection(code: code, probability: p);
    } finally {
      calloc.free(samples);
      calloc.free(outBuf);
    }
  }

  /// Run Silero VAD (or whichever VAD model lives at [modelPath]) on [pcm]
  /// and return the detected speech spans.
  ///
  /// Requires a separate VAD GGML model — the usual Silero model bundled
  /// with CrispASR is ~885 KB.
  List<VadSpan> vad(
    Float32List pcm, {
    required String modelPath,
    int sampleRate = 16000,
    double threshold = 0.5,
    int minSpeechMs = 250,
    int minSilenceMs = 100,
    int nThreads = 4,
    bool useGpu = false,
  }) {
    if (_vadSegments == null || _vadFree == null) {
      throw UnsupportedError(
          'VAD helpers not available — rebuild CrispASR with 0.2.0+ helpers.');
    }

    final samples = calloc<Float>(pcm.length);
    for (var i = 0; i < pcm.length; i++) {
      samples[i] = pcm[i];
    }
    final modelPtr = modelPath.toNativeUtf8();
    final outPtr = calloc<Pointer<Float>>();

    try {
      final n = _vadSegments!(
        modelPtr,
        samples,
        pcm.length,
        sampleRate,
        threshold,
        minSpeechMs,
        minSilenceMs,
        nThreads,
        useGpu ? 1 : 0,
        outPtr,
      );
      if (n < 0) {
        throw Exception('VAD failed (error $n)');
      }
      final spans = <VadSpan>[];
      if (n > 0) {
        final data = outPtr.value;
        for (var i = 0; i < n; i++) {
          spans.add(VadSpan(start: data[2 * i], end: data[2 * i + 1]));
        }
        _vadFree!(data);
      }
      return spans;
    } finally {
      calloc.free(samples);
      calloc.free(modelPtr);
      calloc.free(outPtr);
    }
  }

  /// Supported language codes, e.g. `['en', 'de', ...]`. Returns `[]` when
  /// the loaded dylib doesn't export language-iteration helpers.
  List<String> supportedLanguageCodes() {
    if (_langStr == null || _langMaxId == null) return const [];
    final out = <String>[];
    final max = _langMaxId!();
    for (var i = 0; i <= max; i++) {
      final p = _langStr!(i);
      if (p == nullptr) continue;
      final s = p.toDartString();
      if (s.isNotEmpty) out.add(s);
    }
    return out;
  }

  /// Open a streaming session over this model. Feed PCM chunks as they
  /// arrive and poll each [StreamingSession.feed] return value for new
  /// text.
  ///
  /// Uses whisper.cpp's sliding-window trick: every [stepMs] of fresh
  /// audio triggers a decode over the last [lengthMs], carrying
  /// [keepMs] of context from the previous window. Good first defaults
  /// are the CLI's own (3000 / 10000 / 200 ms). No threads are spawned —
  /// every decode happens synchronously inside `feed`.
  ///
  /// Throws [UnsupportedError] if the loaded dylib is pre-0.3.0.
  StreamingSession openStream({
    int stepMs = 3000,
    int lengthMs = 10000,
    int keepMs = 200,
    int nThreads = 4,
    String? language,
    bool translate = false,
  }) {
    _checkDisposed();
    if (_streamOpen == null ||
        _streamFeed == null ||
        _streamGetText == null ||
        _streamClose == null) {
      throw UnsupportedError(
          'Streaming helpers not available — rebuild CrispASR with 0.3.0+.');
    }

    final langPtr = (language == null || language.isEmpty || language == 'auto')
        ? nullptr
        : language.toNativeUtf8();
    final handle = _streamOpen!(
      _ctx,
      nThreads,
      stepMs,
      lengthMs,
      keepMs,
      langPtr.cast<Utf8>(),
      translate ? 1 : 0,
    );
    if (langPtr != nullptr) calloc.free(langPtr);
    if (handle == nullptr) {
      throw Exception('crispasr_stream_open returned null');
    }

    return StreamingSession._(
      handle: handle,
      feed: _streamFeed!,
      flush: _streamFlush,
      getText: _streamGetText!,
      close: _streamClose!,
    );
  }

  void dispose() {
    if (!_disposed) {
      _free(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispASR has been disposed');
  }

  static String _findLib() => defaultLibName();

  /// Platform-default filename for the CrispASR shared library.
  ///
  /// As of CrispASR 0.4.0 the build produces both `libcrispasr.*`
  /// (preferred) and the historical `libwhisper.*` (alias). We open the
  /// new name first, fall back to the old one if the user's bundle
  /// predates the rename.
  static String defaultLibName() {
    for (final name in _libCandidates()) {
      try {
        DynamicLibrary.open(name); // probe
        return name;
      } catch (_) {/* try next */}
    }
    // Give the caller a sensible default to produce an error message
    // against; opening it will throw and the exception text points at
    // the name they can bundle.
    return _libCandidates().first;
  }

  static List<String> _libCandidates() {
    if (Platform.isAndroid || Platform.isLinux) {
      return ['libcrispasr.so', 'libwhisper.so'];
    }
    if (Platform.isIOS || Platform.isMacOS) {
      return [
        'libcrispasr.dylib',
        'crispasr.framework/crispasr',
        'libwhisper.dylib',
        'whisper.framework/whisper',
      ];
    }
    if (Platform.isWindows) {
      return ['crispasr.dll', 'whisper.dll'];
    }
    return ['libcrispasr.so', 'libwhisper.so'];
  }
}

/// A live streaming decode session, created via [CrispASR.openStream].
///
/// Feed PCM chunks as they arrive; every chunk whose accumulation crosses
/// the configured `stepMs` triggers a decode over the rolling window and
/// returns a [StreamingUpdate]. Chunks that don't trigger a decode return
/// `null` — the caller is still buffering.
///
/// Close the session explicitly with [close] to free the native state —
/// there is no Dart finalizer hooking libwhisper.
class StreamingSession {
  StreamingSession._({
    required Pointer<Void> handle,
    required _StreamFeed feed,
    required _StreamFlush? flush,
    required _StreamGetText getText,
    required _StreamClose close,
  })  : _handle = handle,
        _feedFn = feed,
        _flushFn = flush,
        _getTextFn = getText,
        _closeFn = close;

  final Pointer<Void> _handle;
  final _StreamFeed    _feedFn;
  final _StreamFlush?  _flushFn;
  final _StreamGetText _getTextFn;
  final _StreamClose   _closeFn;

  bool _closed = false;
  int _lastCounter = -1;

  bool get isClosed => _closed;

  /// Feed a chunk of 16 kHz mono float32 PCM. Returns a [StreamingUpdate]
  /// when this chunk's arrival triggered a new decode, otherwise `null`.
  StreamingUpdate? feed(Float32List pcm) {
    if (_closed) throw StateError('StreamingSession is closed');
    if (pcm.isEmpty) return null;

    final buf = calloc<Float>(pcm.length);
    for (var i = 0; i < pcm.length; i++) {
      buf[i] = pcm[i];
    }
    try {
      final r = _feedFn(_handle, buf, pcm.length);
      if (r < 0) throw Exception('crispasr_stream_feed error $r');
      if (r == 0) return null; // still buffering
      return _readOutput();
    } finally {
      calloc.free(buf);
    }
  }

  /// Force a final decode on whatever audio is currently buffered.
  ///
  /// Useful when the caller's audio source has ended (e.g. user stopped
  /// recording) and they want the last partial flushed out. Returns the
  /// resulting update, or `null` if nothing was buffered.
  StreamingUpdate? flush() {
    if (_closed) throw StateError('StreamingSession is closed');
    if (_flushFn == null) return _readOutput();
    final r = _flushFn!(_handle);
    if (r <= 0) return null;
    return _readOutput();
  }

  StreamingUpdate? _readOutput() {
    final outCap = 4096;
    final outBuf = calloc<Uint8>(outCap);
    final out    = outBuf.cast<Utf8>();
    final t0Ptr  = calloc<Double>();
    final t1Ptr  = calloc<Double>();
    final cntPtr = calloc<Int64>();

    try {
      final n = _getTextFn(_handle, out, outCap, t0Ptr, t1Ptr, cntPtr);
      if (n <= 0) return null;
      final counter = cntPtr.value;
      if (counter == _lastCounter) return null; // same decode we already saw
      _lastCounter = counter;
      return StreamingUpdate(
        text: out.toDartString(),
        start: t0Ptr.value,
        end: t1Ptr.value,
        counter: counter,
      );
    } finally {
      calloc.free(outBuf);
      calloc.free(t0Ptr);
      calloc.free(t1Ptr);
      calloc.free(cntPtr);
    }
  }

  /// Release the native session. Safe to call more than once.
  void close() {
    if (_closed) return;
    _closed = true;
    _closeFn(_handle);
  }
}

// =====================================================================
// Unified backend-agnostic session (0.4.0+)
//
// The [CrispASR] class is Whisper-only — it exists for backward
// compatibility and low-overhead direct access to whisper.h. For any new
// client, prefer [CrispasrSession]: one constructor, one `transcribe`
// method, auto-dispatched to whichever backend (Whisper, Parakeet, …)
// the GGUF metadata identifies. A backend the loaded libwhisper wasn't
// linked with will cause the open call to throw — [availableBackends]
// lists what's supported at runtime.
// =====================================================================

/// Unified session over any CrispASR-supported GGUF model.
class CrispasrSession {
  CrispasrSession._(
    this._lib,
    this._handle,
    this._backend,
  );

  final DynamicLibrary _lib;
  Pointer<Void> _handle;
  final String _backend;
  bool _closed = false;

  /// Open a model file. Backend is auto-detected from the GGUF
  /// `general.architecture` metadata key.
  ///
  /// Throws when the loaded dylib is pre-0.4.0 (no `crispasr_session_*`
  /// symbols) or when the backend identified in the file wasn't compiled
  /// into that dylib (`availableBackends` to introspect).
  factory CrispasrSession.open(
    String modelPath, {
    int nThreads = 4,
    String? libPath,
    String? backend,
  }) {
    final lib = DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
    if (!lib.providesSymbol('crispasr_session_open')) {
      throw UnsupportedError(
          'Unified session API not available — rebuild CrispASR with 0.4.0+ helpers.');
    }

    final pathPtr = modelPath.toNativeUtf8();
    Pointer<Utf8> bePtr = nullptr;
    try {
      Pointer<Void> handle;
      if (backend != null && backend.isNotEmpty) {
        bePtr = backend.toNativeUtf8();
        final openExpl = lib.lookupFunction<
            Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>, Int32),
            Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>, int)>(
          'crispasr_session_open_explicit',
        );
        handle = openExpl(pathPtr, bePtr, nThreads);
      } else {
        final open = lib.lookupFunction<
            Pointer<Void> Function(Pointer<Utf8>, Int32),
            Pointer<Void> Function(Pointer<Utf8>, int)>(
          'crispasr_session_open',
        );
        handle = open(pathPtr, nThreads);
      }
      if (handle == nullptr) {
        throw Exception(
            'crispasr_session_open returned null — either the GGUF backend '
            'isn\'t one of ${_availableBackends(lib).join(", ")} or the '
            'file is unreadable.');
      }
      final backendFn = lib.lookupFunction<
          Pointer<Utf8> Function(Pointer<Void>),
          Pointer<Utf8> Function(Pointer<Void>)>('crispasr_session_backend');
      final bp = backendFn(handle);
      final be = bp == nullptr ? '' : bp.toDartString();
      return CrispasrSession._(lib, handle, be);
    } finally {
      calloc.free(pathPtr);
      if (bePtr != nullptr) calloc.free(bePtr);
    }
  }

  /// List of backend names compiled into the loaded libwhisper.
  /// Always includes 'whisper'. Non-Whisper backends are added as they
  /// get linked in (parakeet, canary, qwen3, …).
  static List<String> availableBackends({String? libPath}) {
    try {
      final lib = DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
      return _availableBackends(lib);
    } catch (_) {
      return const [];
    }
  }

  static List<String> _availableBackends(DynamicLibrary lib) {
    if (!lib.providesSymbol('crispasr_session_available_backends')) {
      return const [];
    }
    final fn = lib.lookupFunction<
        Int32 Function(Pointer<Utf8>, Int32),
        int Function(Pointer<Utf8>, int)>('crispasr_session_available_backends');
    final buf = calloc<Uint8>(256);
    try {
      final ptr = buf.cast<Utf8>();
      fn(ptr, 256);
      final csv = ptr.toDartString();
      return csv.isEmpty
          ? const <String>[]
          : csv.split(',').map((s) => s.trim()).toList();
    } finally {
      calloc.free(buf);
    }
  }

  /// Name of the backend this session ended up using.
  String get backend => _backend;
  bool get isClosed => _closed;

  /// Transcribe 16 kHz mono float32 PCM. Returns a list of segments
  /// with word-level timings when the backend supports them.
  ///
  /// Pass [language] as an ISO 639-1 code ("en", "de", "ja", …) to steer
  /// backends that accept a source-language hint (whisper / canary /
  /// cohere / voxtral / voxtral4b). Backends that auto-detect
  /// (parakeet / qwen3) or that don't expose a language input
  /// (granite / wav2vec2 / fastconformer-ctc) ignore the hint silently.
  /// `null` or empty preserves each backend's historical default.
  List<SessionSegment> transcribe(Float32List pcm, {String? language}) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (pcm.isEmpty) return const [];

    final samples = calloc<Float>(pcm.length);
    for (var i = 0; i < pcm.length; i++) {
      samples[i] = pcm[i];
    }

    Pointer<Void> res;
    Pointer<Utf8>? langPtr;
    if (language != null && language.isNotEmpty) {
      langPtr = language.toNativeUtf8();
      final fn = _lib.lookupFunction<
          Pointer<Void> Function(Pointer<Void>, Pointer<Float>, Int32, Pointer<Utf8>),
          Pointer<Void> Function(Pointer<Void>, Pointer<Float>, int, Pointer<Utf8>)>(
        'crispasr_session_transcribe_lang',
      );
      res = fn(_handle, samples, pcm.length, langPtr);
    } else {
      final fn = _lib.lookupFunction<
          Pointer<Void> Function(Pointer<Void>, Pointer<Float>, Int32),
          Pointer<Void> Function(Pointer<Void>, Pointer<Float>, int)>(
        'crispasr_session_transcribe',
      );
      res = fn(_handle, samples, pcm.length);
    }
    calloc.free(samples);
    if (langPtr != null) calloc.free(langPtr);
    if (res == nullptr) {
      throw Exception('crispasr_session_transcribe returned null');
    }

    try {
      return _readSegments(res);
    } finally {
      final freeFn =
          _lib.lookupFunction<Void Function(Pointer<Void>), void Function(Pointer<Void>)>(
        'crispasr_session_result_free',
      );
      freeFn(res);
    }
  }

  /// Transcribe with Silero VAD segmentation + whisper.cpp-style stitching.
  ///
  /// Runs VAD on [pcm], merges short / overlong speech slices into usable
  /// chunks, stitches them into a single buffer with 0.1s silence gaps,
  /// calls the backend once on the stitched buffer, then remaps segment
  /// and word timestamps back to original-audio positions.
  ///
  /// [vadModelPath] must point to a Silero GGUF on disk (e.g. the one
  /// bundled as a Flutter asset). If it's empty or the model fails to
  /// load, this falls back to a plain [transcribe] call so callers always
  /// get a result when audio exists.
  ///
  /// Compared to calling [transcribe] on the raw buffer, this:
  /// * skips silence, cutting encoder cost substantially for sparse audio;
  /// * preserves cross-segment decoder context (one call, not N), which
  ///   matters for O(T²) backends such as parakeet / cohere / canary.
  List<SessionSegment> transcribeVad(
    Float32List pcm,
    String vadModelPath, {
    int sampleRate = 16000,
    SessionVadOptions options = const SessionVadOptions(),
    String? language,
  }) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (pcm.isEmpty) return const [];

    final samples = calloc<Float>(pcm.length);
    for (var i = 0; i < pcm.length; i++) {
      samples[i] = pcm[i];
    }
    final vadPathPtr = vadModelPath.toNativeUtf8();

    // ABI struct layout must match crispasr_vad_abi_opts in crispasr_c_api.cpp.
    // float threshold + 5 x int32 = 24 bytes.
    final optsPtr = calloc<Uint8>(24);
    optsPtr.cast<Float>().value = options.threshold;
    final intView = optsPtr.elementAt(4).cast<Int32>();
    intView[0] = options.minSpeechDurationMs;
    intView[1] = options.minSilenceDurationMs;
    intView[2] = options.speechPadMs;
    intView[3] = options.chunkSeconds;
    intView[4] = options.nThreads;

    Pointer<Void> res;
    Pointer<Utf8>? langPtr;
    if (language != null && language.isNotEmpty) {
      langPtr = language.toNativeUtf8();
      final fn = _lib.lookupFunction<
          Pointer<Void> Function(Pointer<Void>, Pointer<Float>, Int32, Int32,
              Pointer<Utf8>, Pointer<Uint8>, Pointer<Utf8>),
          Pointer<Void> Function(Pointer<Void>, Pointer<Float>, int, int,
              Pointer<Utf8>, Pointer<Uint8>, Pointer<Utf8>)>(
        'crispasr_session_transcribe_vad_lang',
      );
      res = fn(_handle, samples, pcm.length, sampleRate, vadPathPtr, optsPtr, langPtr);
    } else {
      final fn = _lib.lookupFunction<
          Pointer<Void> Function(Pointer<Void>, Pointer<Float>, Int32, Int32,
              Pointer<Utf8>, Pointer<Uint8>),
          Pointer<Void> Function(Pointer<Void>, Pointer<Float>, int, int,
              Pointer<Utf8>, Pointer<Uint8>)>(
        'crispasr_session_transcribe_vad',
      );
      res = fn(_handle, samples, pcm.length, sampleRate, vadPathPtr, optsPtr);
    }
    calloc.free(samples);
    calloc.free(vadPathPtr);
    calloc.free(optsPtr);
    if (langPtr != null) calloc.free(langPtr);
    if (res == nullptr) {
      throw Exception('crispasr_session_transcribe_vad returned null');
    }

    try {
      return _readSegments(res);
    } finally {
      final freeFn = _lib.lookupFunction<
          Void Function(Pointer<Void>),
          void Function(Pointer<Void>)>('crispasr_session_result_free');
      freeFn(res);
    }
  }

  List<SessionSegment> _readSegments(Pointer<Void> res) {
    final nSegs = _lib.lookupFunction<Int32 Function(Pointer<Void>), int Function(Pointer<Void>)>(
        'crispasr_session_result_n_segments')(res);
    final segText = _lib.lookupFunction<
        Pointer<Utf8> Function(Pointer<Void>, Int32),
        Pointer<Utf8> Function(Pointer<Void>, int)>(
      'crispasr_session_result_segment_text',
    );
    final segT0 = _lib.lookupFunction<
        Int64 Function(Pointer<Void>, Int32),
        int Function(Pointer<Void>, int)>('crispasr_session_result_segment_t0');
    final segT1 = _lib.lookupFunction<
        Int64 Function(Pointer<Void>, Int32),
        int Function(Pointer<Void>, int)>('crispasr_session_result_segment_t1');
    final nWords = _lib.lookupFunction<
        Int32 Function(Pointer<Void>, Int32),
        int Function(Pointer<Void>, int)>('crispasr_session_result_n_words');
    final wordText = _lib.lookupFunction<
        Pointer<Utf8> Function(Pointer<Void>, Int32, Int32),
        Pointer<Utf8> Function(Pointer<Void>, int, int)>(
      'crispasr_session_result_word_text',
    );
    final wordT0 = _lib.lookupFunction<
        Int64 Function(Pointer<Void>, Int32, Int32),
        int Function(Pointer<Void>, int, int)>('crispasr_session_result_word_t0');
    final wordT1 = _lib.lookupFunction<
        Int64 Function(Pointer<Void>, Int32, Int32),
        int Function(Pointer<Void>, int, int)>('crispasr_session_result_word_t1');

    final out = <SessionSegment>[];
    for (var i = 0; i < nSegs; i++) {
      final tp = segText(res, i);
      final text = tp == nullptr ? '' : tp.toDartString();
      final t0 = segT0(res, i) / 100.0;
      final t1 = segT1(res, i) / 100.0;
      final wc = nWords(res, i);
      final words = <Word>[];
      for (var k = 0; k < wc; k++) {
        final wp = wordText(res, i, k);
        final wt = wp == nullptr ? '' : wp.toDartString();
        words.add(Word(
          text: wt,
          start: wordT0(res, i, k) / 100.0,
          end:   wordT1(res, i, k) / 100.0,
          p: 1.0,
        ));
      }
      out.add(SessionSegment(text: text.trim(), start: t0, end: t1, words: words));
    }
    return out;
  }

  void close() {
    if (_closed) return;
    _closed = true;
    final closeFn =
        _lib.lookupFunction<Void Function(Pointer<Void>), void Function(Pointer<Void>)>(
      'crispasr_session_close',
    );
    closeFn(_handle);
    _handle = nullptr;
  }
}

// =========================================================================
// FireRedPunc — punctuation restoration post-processor
// =========================================================================

/// BERT-based punctuation restoration model (FireRedPunc).
///
/// Adds punctuation and capitalization to unpunctuated ASR output.
/// Particularly useful for CTC-based backends (wav2vec2, omniasr,
/// fastconformer-ctc, firered-asr) that output lowercase text.
///
/// ```dart
/// final punc = PuncModel.open('fireredpunc-q8_0.gguf');
/// final text = punc.process('and so my fellow americans ask not');
/// print(text); // "And so my fellow americans, ask not..."
/// punc.close();
/// ```
class PuncModel {
  final DynamicLibrary _lib;
  Pointer<Void> _handle;
  bool _closed = false;

  PuncModel._(this._lib, this._handle);

  /// Load a FireRedPunc GGUF model.
  static PuncModel open(String modelPath, {String? libPath}) {
    final lib = DynamicLibrary.open(libPath ?? _defaultLibPath());
    final initFn = lib.lookupFunction<
        Pointer<Void> Function(Pointer<Utf8>),
        Pointer<Void> Function(Pointer<Utf8>)>('crispasr_punc_init');
    final pathPtr = modelPath.toNativeUtf8();
    final handle = initFn(pathPtr);
    calloc.free(pathPtr);
    if (handle == nullptr) {
      throw Exception('Failed to load punc model: $modelPath');
    }
    return PuncModel._(lib, handle);
  }

  /// Add punctuation to unpunctuated text.
  String process(String text) {
    if (_closed) throw StateError('PuncModel is closed');
    final processFn = _lib.lookupFunction<
        Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>),
        Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>)>('crispasr_punc_process');
    final freeFn = _lib.lookupFunction<
        Void Function(Pointer<Utf8>),
        void Function(Pointer<Utf8>)>('crispasr_punc_free_text');
    final textPtr = text.toNativeUtf8();
    final resultPtr = processFn(_handle, textPtr);
    calloc.free(textPtr);
    if (resultPtr == nullptr) return text;
    final result = resultPtr.toDartString();
    freeFn(resultPtr);
    return result;
  }

  void close() {
    if (_closed) return;
    _closed = true;
    final freeFn = _lib.lookupFunction<
        Void Function(Pointer<Void>),
        void Function(Pointer<Void>)>('crispasr_punc_free');
    freeFn(_handle);
    _handle = nullptr;
  }
}
