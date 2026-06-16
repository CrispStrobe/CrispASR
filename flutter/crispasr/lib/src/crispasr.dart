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
  /// Top-N runner-up candidates for this token. Populated only when
  /// the caller asked for alt-token capture (`altN` > 0 on the
  /// session or `crispasr_params_set_alt_n` on the low-level path)
  /// AND the loaded libcrispasr is ≥ 0.5.13. Empty otherwise. Ordered
  /// descending by probability and the chosen token is not present.
  final List<AltToken> alts;

  const Word({
    required this.text,
    required this.start,
    required this.end,
    required this.p,
    this.alts = const [],
  });

  @override
  String toString() =>
      '${start.toStringAsFixed(2)}-${end.toStringAsFixed(2)} $text';
}

/// A single alternative-candidate suggestion. Surfaced for Whisper
/// greedy decode when `altN` > 0; lets transcript-editor UIs offer
/// tap-to-pick over ambiguous proper nouns / technical jargon.
class AltToken {
  /// The candidate's display text (already passed through the
  /// tokenizer's id→string mapping; for Whisper sub-word BPE this
  /// includes the leading-space marker when present).
  final String text;

  /// Softmax probability at the same decode step in `[0, 1]`.
  final double p;

  const AltToken({required this.text, required this.p});

  @override
  String toString() => '$text(${(p * 100).toStringAsFixed(1)}%)';
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

/// Result of [detectTextLanguage]: the model's native language label
/// (ISO 639-1 for CLD3; GlotLID emits `xxx_Script` codes) + a confidence
/// in [0, 1].
class TextLanguage {
  final String code;
  final double confidence;
  const TextLanguage(this.code, this.confidence);
  @override
  String toString() => 'TextLanguage($code, ${confidence.toStringAsFixed(3)})';
}

/// Identify the language of [text] in-process via CrispASR's text-LID
/// C-ABI (`crispasr_text_detect_language`, wrapping the CLD3 / GlotLID
/// `text_lid_dispatch` façade). One-shot: loads [modelPath] (e.g. a
/// `cld3-*.gguf`), predicts, frees — fine for the occasional
/// "what language is this transcript?" call; open a persistent handle
/// on the C side if you ever need high-throughput batch LID.
///
/// Returns null when the loaded library predates the symbol or detection
/// fails (the C side returns a non-zero status). Distinct from
/// [CrispASR.detectLanguage] / [LidService], which run AUDIO LID on PCM.
TextLanguage? detectTextLanguage(
  String text,
  String modelPath, {
  int nThreads = 1,
  String? libPath,
}) {
  final lib = DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
  if (!lib.providesSymbol('crispasr_text_detect_language')) return null;
  final fn = lib.lookupFunction<
      Int32 Function(
          Pointer<Utf8>, Pointer<Utf8>, Int32, Pointer<Utf8>, Int32, Pointer<Float>),
      int Function(Pointer<Utf8>, Pointer<Utf8>, int, Pointer<Utf8>, int,
          Pointer<Float>)>('crispasr_text_detect_language');

  final textPtr = text.toNativeUtf8();
  final modelPtr = modelPath.toNativeUtf8();
  const cap = 64; // a language label is a handful of bytes; 64 is ample
  final labelBuf = calloc<Uint8>(cap);
  final confOut = calloc<Float>();
  try {
    final rc = fn(textPtr, modelPtr, nThreads, labelBuf.cast<Utf8>(), cap, confOut);
    if (rc != 0) return null;
    final label = labelBuf.cast<Utf8>().toDartString();
    if (label.isEmpty) return null;
    return TextLanguage(label, confOut.value);
  } finally {
    calloc.free(textPtr);
    calloc.free(modelPtr);
    calloc.free(labelBuf);
    calloc.free(confOut);
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
    final base = segsPtr + i * 24;
    base.cast<Int64>().value = (segs[i].t0 * 100).round();
    (base + 8).cast<Int64>().value = (segs[i].t1 * 100).round();
    (base + 16).cast<Int32>().value = segs[i].speaker;
    (base + 20).cast<Int32>().value = 0;
  }

  // ABI opts: int32 method, int32 n_threads, int64 slice_t0_cs, const char*.
  // 4+4+8+8 = 24 bytes on 64-bit. We write each field explicitly.
  final optsPtr = calloc<Uint8>(24);
  optsPtr.cast<Int32>().value = method.index;
  (optsPtr + 4).cast<Int32>().value = nThreads;
  (optsPtr + 8).cast<Int64>().value = (sliceT0 * 100).round();
  final pathPtr = (method == DiarizeMethod.pyannote &&
          pyannoteModelPath != null &&
          pyannoteModelPath.isNotEmpty)
      ? pyannoteModelPath.toNativeUtf8()
      : nullptr;
  (optsPtr + 16).cast<IntPtr>().value = pathPtr.address;

  final fn = lib.lookupFunction<
      Int32 Function(Pointer<Float>, Pointer<Float>, Int32, Int32,
          Pointer<Uint8>, Int32, Pointer<Uint8>),
      int Function(Pointer<Float>, Pointer<Float>, int, int, Pointer<Uint8>,
          int, Pointer<Uint8>)>('crispasr_diarize_segments_abi');
  final rc = fn(leftPtr, rightPtr, n, isStereo ? 1 : 0, segsPtr, segs.length, optsPtr);

  if (rc == 0) {
    for (var i = 0; i < segs.length; i++) {
      segs[i].speaker = (segsPtr + i * 24 + 16).cast<Int32>().value;
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
/// voxtral, voxtral4b, granite, qwen3, cohere, nemotron, wav2vec2). Returns null
/// on miss.
RegistryEntry? registryLookup(String backend, {DynamicLibrary? lib}) =>
    _registryCall('crispasr_registry_lookup_abi', backend, lib);

/// Look up the canonical GGUF by filename (exact match or substring).
/// Useful when a user-supplied filename isn't cached yet and we want to
/// suggest a download URL.
RegistryEntry? registryLookupByFilename(String filename, {DynamicLibrary? lib}) =>
    _registryCall('crispasr_registry_lookup_by_filename_abi', filename, lib);

/// Every backend name in the registry, in declaration order. Each name
/// can be passed back to [registryLookup] for full details.
List<String> listKnownModels({DynamicLibrary? lib}) {
  lib ??= DynamicLibrary.open(CrispASR.defaultLibName());
  if (!lib.providesSymbol('crispasr_registry_list_backends_abi')) return const [];
  final fn = lib.lookupFunction<Int32 Function(Pointer<Uint8>, Int32), int Function(Pointer<Uint8>, int)>(
      'crispasr_registry_list_backends_abi');
  final buf = calloc<Uint8>(8192);
  try {
    final n = fn(buf, 8192);
    if (n < 0) return const [];
    return buf.cast<Utf8>().toDartString().split(',').where((s) => s.isNotEmpty).toList(growable: false);
  } finally {
    calloc.free(buf);
  }
}

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

// ===========================================================================
// Microphone capture (PLAN #62d) — cross-platform live PCM via miniaudio.
// ===========================================================================

typedef _MicCallbackNative = Void Function(Pointer<Float> pcm, Int32 nSamples, Pointer<Void> userdata);

/// Library-level microphone handle. The user-supplied callback is
/// invoked from miniaudio's audio thread with mono float32 PCM in
/// [-1, 1]. Keep the callback short and non-blocking.
class Mic {
  final DynamicLibrary _lib;
  Pointer<Void> _handle;
  // Hold the ctypes-style native callback alive for the lifetime of
  // the Mic; if it's GC'd, miniaudio will crash.
  late final NativeCallable<_MicCallbackNative> _trampoline;
  bool _started = false;

  Mic._(this._lib, this._handle, this._trampoline);

  /// Open the default capture device. `sampleRate=16000` matches every
  /// ASR backend. `channels=1` for mono (recommended).
  static Mic open({
    required void Function(Float32List pcm) callback,
    int sampleRate = 16000,
    int channels = 1,
    DynamicLibrary? lib,
  }) {
    lib ??= DynamicLibrary.open(CrispASR.defaultLibName());
    if (!lib.providesSymbol('crispasr_mic_open')) {
      throw UnsupportedError('mic API not present in this libcrispasr build');
    }
    final trampoline = NativeCallable<_MicCallbackNative>.listener((Pointer<Float> pcm, int n, Pointer<Void> _) {
      // Audio thread → main isolate via NativeCallable.listener.
      final view = pcm.asTypedList(n);
      final copy = Float32List.fromList(view); // detach from miniaudio's buffer
      try {
        callback(copy);
      } catch (e) {
        // ignore — audio thread mustn't propagate exceptions
      }
    });
    final fn = lib.lookupFunction<
        Pointer<Void> Function(Int32, Int32, Pointer<NativeFunction<_MicCallbackNative>>, Pointer<Void>),
        Pointer<Void> Function(int, int, Pointer<NativeFunction<_MicCallbackNative>>,
            Pointer<Void>)>('crispasr_mic_open');
    final h = fn(sampleRate, channels, trampoline.nativeFunction, nullptr);
    if (h == nullptr) {
      trampoline.close();
      throw Exception('crispasr_mic_open failed');
    }
    return Mic._(lib, h, trampoline);
  }

  void start() {
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>), int Function(Pointer<Void>)>('crispasr_mic_start');
    final rc = fn(_handle);
    if (rc != 0) throw Exception('mic_start failed (rc=$rc)');
    _started = true;
  }

  void stop() {
    if (!_started) return;
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>), int Function(Pointer<Void>)>('crispasr_mic_stop');
    fn(_handle);
    _started = false;
  }

  void close() {
    if (_handle == nullptr) return;
    stop();
    final fn = _lib.lookupFunction<Void Function(Pointer<Void>), void Function(Pointer<Void>)>('crispasr_mic_close');
    fn(_handle);
    _handle = nullptr;
    _trampoline.close();
  }
}

/// Human-readable name of the default capture device, or empty string
/// if no input device is available.
String micDefaultDeviceName({DynamicLibrary? lib}) {
  lib ??= DynamicLibrary.open(CrispASR.defaultLibName());
  if (!lib.providesSymbol('crispasr_mic_default_device_name')) return '';
  final fn = lib.lookupFunction<Pointer<Utf8> Function(), Pointer<Utf8> Function()>(
      'crispasr_mic_default_device_name');
  final p = fn();
  if (p == nullptr) return '';
  return p.toDartString();
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
  /// Reuses any multilingual ggml-tiny / base / small / medium / large
  /// file already on disk — no separate LID model to download.
  whisper,
  /// GGUF-packed Silero 95-language classifier (~16 MB). Fast, no GPU
  /// required. Recommended default when the user has the
  /// `silero-lid-95-f16.gguf` (or legacy `silero-lang95-v1-f16.gguf`)
  /// on disk.
  silero,
  /// FireRed-LID 120-language Transformer (~300 MB). Higher coverage
  /// than Silero, especially on low-resource languages. Routes through
  /// the same `crispasr_detect_language_pcm` C ABI; needs
  /// `firered-lid-f16.gguf` on disk.
  firered,
  /// ECAPA-TDNN 107-language LID (~42 MB, speechbrain/lang-id-voxlingua107).
  /// Strong on noisy / accented speech; faster than FireRed.
  /// Needs `ecapa-lid-107-f16.gguf` on disk.
  ecapa,
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

/// RNNoise-based audio enhancement (transcribe pre-step).
///
/// Returns a fresh [Float32List] of the same length as [pcm] with the
/// denoiser applied. [pcm] must be 16 kHz mono float32 in `[-1, 1]`;
/// the wrapper upsamples to 48 kHz internally to run RNNoise's
/// 480-sample frame loop and downsamples back. State is allocated
/// per call inside libcrispasr, so this is safe to invoke from
/// concurrent worker isolates.
///
/// Throws [UnsupportedError] when the loaded dylib predates 0.5.12
/// (i.e. doesn't expose `crispasr_enhance_audio_rnnoise`), letting
/// callers graceful-degrade to the un-enhanced PCM.
Float32List enhanceAudioRnnoise(
  Float32List pcm, {
  DynamicLibrary? lib,
}) {
  if (pcm.isEmpty) return Float32List(0);

  lib ??= DynamicLibrary.open(CrispASR.defaultLibName());
  if (!lib.providesSymbol('crispasr_enhance_audio_rnnoise')) {
    throw UnsupportedError(
        'crispasr_enhance_audio_rnnoise not present in this libcrispasr build — '
        'rebuild against CrispASR 0.5.12+');
  }

  final fn = lib.lookupFunction<
      Int32 Function(Pointer<Float>, Int32, Pointer<Float>, Int32),
      int Function(Pointer<Float>, int, Pointer<Float>, int)>(
      'crispasr_enhance_audio_rnnoise');

  final inBuf = calloc<Float>(pcm.length);
  final outBuf = calloc<Float>(pcm.length);
  try {
    for (var i = 0; i < pcm.length; i++) inBuf[i] = pcm[i];
    final rc = fn(inBuf, pcm.length, outBuf, pcm.length);
    if (rc != 0) {
      throw StateError('crispasr_enhance_audio_rnnoise failed (rc=$rc)');
    }
    final out = Float32List(pcm.length);
    for (var i = 0; i < pcm.length; i++) out[i] = outBuf[i];
    return out;
  } finally {
    calloc.free(inBuf);
    calloc.free(outBuf);
  }
}

/// Tunables for [CrispasrSession.transcribeVad]. Field names and defaults
/// mirror crispasr's `whisper_vad_params` plus the max-chunk fallback
/// the shared library uses to bound encoder cost on long audio.
class SessionVadOptions {
  /// Silero VAD decision threshold (0..1). Higher = fewer / shorter
  /// speech regions. crispasr ships 0.5.
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
  /// Best-of-N greedy sampling for whisper. 0 or 1 = disabled (single pass).
  /// Whisper runs N internal greedy decodes and picks the highest-scoring
  /// one — quality goes up at ~Nx CPU cost. Maps to
  /// `whisper_full_params.greedy.best_of`.
  final int bestOf;
  /// Thread count. 0 = whisper default (usually 4).
  final int nThreads;
  /// An initial text prompt to condition the decoder on.
  final String? initialPrompt;
  /// Silence the library's own stdout output.
  final bool silent;

  // --- VAD (Silero, built into crispasr). Set [vad] + [vadModelPath]
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

  /// Capture top-N alternative-candidate tokens per greedy-sampled
  /// step. 0 (default) = off. Whisper-only and meaningful only for
  /// greedy decoding (beam search candidates aren't comparable). UI
  /// layers should cap at 5 — beyond that the memory cost grows
  /// linearly with no real benefit.
  final int altN;

  /// Max tokens per segment. 0 = whisper default. Maps to
  /// `whisper_full_params.max_tokens`.
  final int maxTokens;
  /// Do not use past transcription as initial prompt for the decoder.
  final bool noContext;
  /// Force single-segment output (useful for streaming).
  final bool singleSegment;
  /// Suppress blank tokens at the beginning of each segment.
  final bool suppressBlank;
  /// Initial decoding temperature. 0.0 = whisper default.
  final double temperature;

  const TranscribeOptions({
    this.strategy = 0,
    this.language,
    this.translate = false,
    this.detectLanguage = false,
    this.wordTimestamps = false,
    this.maxLen = 0,
    this.splitOnWord = false,
    this.bestOf = 0,
    this.nThreads = 0,
    this.initialPrompt,
    this.silent = true,
    this.vad = false,
    this.vadModelPath,
    this.vadThreshold = 0.5,
    this.vadMinSpeechMs = 250,
    this.vadMinSilenceMs = 100,
    this.tdrz = false,
    this.altN = 0,
    this.maxTokens = 0,
    this.noContext = false,
    this.singleSegment = false,
    this.suppressBlank = true,
    this.temperature = 0.0,
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

// Alt-token accessors (0.5.13). All take (ctx, i_seg, i_tok) +
// optionally (i_alt) and return small POD values, so Dart FFI binds
// directly.
typedef _TokenNAltsNative = Int32 Function(Pointer<Void>, Int32, Int32);
typedef _TokenNAlts       = int   Function(Pointer<Void>, int, int);

typedef _TokenAltIdNative = Int32 Function(Pointer<Void>, Int32, Int32, Int32);
typedef _TokenAltId       = int   Function(Pointer<Void>, int, int, int);

typedef _TokenAltPNative = Float  Function(Pointer<Void>, Int32, Int32, Int32);
typedef _TokenAltP       = double Function(Pointer<Void>, int, int, int);

typedef _WhisperTokenToStrNative = Pointer<Utf8> Function(Pointer<Void>, Int32);
typedef _WhisperTokenToStr       = Pointer<Utf8> Function(Pointer<Void>, int);

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

typedef _StreamSetLiveDecodeNative = Void Function(Pointer<Void>, Int32);
typedef _StreamSetLiveDecode       = void Function(Pointer<Void>, int);

// token_alt_text: fills a buffer with the alt-candidate text.
typedef _TokenAltTextNative = Int32 Function(Pointer<Void>, Int32, Int32, Int32, Pointer<Utf8>, Int32);
typedef _TokenAltText       = int   Function(Pointer<Void>, int, int, int, Pointer<Utf8>, int);

// VAD slices (0.6.0+): unified VAD dispatcher returning seconds.
typedef _VadSlicesNative = Int32 Function(
    Pointer<Utf8>, Pointer<Float>, Int32, Int32, Float, Int32, Int32, Int32,
    Float, Int32, Pointer<Pointer<Float>>);
typedef _VadSlices = int Function(
    Pointer<Utf8>, Pointer<Float>, int, int, double, int, int, int,
    double, int, Pointer<Pointer<Float>>);

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
  late final _WhisperInit     _initByRef;
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
  _ParamsSetInt?    _paramsSetBestOf;
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

  // 0.5.13 alt-token accessors. Null when loaded dylib is pre-0.5.13
  // — callers fall back to empty alt lists and the UI hides the
  // tap-to-pick affordance.
  _TokenNAlts?       _tokenNAlts;
  _TokenAltId?       _tokenAltId;
  _TokenAltP?        _tokenAltP;
  _WhisperTokenToStr? _whisperTokenToStr;
  _ParamsSetInt?     _paramsSetAltN;

  _DetectLang?  _detectLang;
  _VadSegments? _vadSegments;
  _VadFree?     _vadFree;

  _LangStr?     _langStr;
  _IntFn?       _langMaxId;

  _StreamOpen?    _streamOpen;
  _StreamFeed?    _streamFeed;
  _StreamFlush?   _streamFlush;
  _StreamGetText? _streamGetText;
  _StreamClose?   _streamClose;

  // params_set additions for full parity.
  _ParamsSetInt?    _paramsSetMaxTokens;
  _ParamsSetBool?   _paramsSetNoContext;
  _ParamsSetBool?   _paramsSetSingleSegment;
  _ParamsSetBool?   _paramsSetSuppressBlank;
  _ParamsSetFloat?  _paramsSetTemperature;

  // token_alt_text: fills a char* buffer with the alternate text.
  _TokenAltText?    _tokenAltText;

  // VAD slices (unified dispatcher, returns seconds).
  _VadSlices?       _vadSlices;

  // Stream live-decode toggle.
  _StreamSetLiveDecode? _streamSetLiveDecode;

  bool get supportsExtended => _detectLang != null;
  bool get supportsStreaming => _streamOpen != null;

  CrispASR(String modelPath, {String? libPath}) {
    _lib = DynamicLibrary.open(libPath ?? _findLib());

    _free          = _lib.lookupFunction<_VoidPtr_C, _VoidPtr>('whisper_free');
    _defaultParams = _lib.lookupFunction<_DefaultParamsNative, _DefaultParams>(
        'whisper_full_default_params_by_ref');
    _nSegments = _lib.lookupFunction<_IntPtr_C, _IntPtr>('whisper_full_n_segments');
    _getText   = _lib.lookupFunction<_GetTextNative, _GetText>('whisper_full_get_segment_text');
    _getT0     = _lib.lookupFunction<_GetT0Native, _GetT0>('whisper_full_get_segment_t0');
    _getT1     = _lib.lookupFunction<_GetT0Native, _GetT0>('whisper_full_get_segment_t1');
    _getNSP    = _lib.lookupFunction<_GetNSPNative, _GetNSP>('whisper_full_get_segment_no_speech_prob');
    _freeParams = _lib.lookupFunction<_VoidPtr_C, _VoidPtr>('whisper_free_params');

    // Prefer the *_by_ref wrappers (CrispASR ≥0.6.11) that take struct
    // pointers — calling the canonical `whisper_init_from_file_with_params` /
    // `whisper_full` directly fails on platforms where the C ABI passes
    // large structs differently from how Dart marshals a `Pointer<Void>`.
    // The historical (x86_64 Linux) symptom was a corrupt
    // `whisper_full_params` reaching `whisper_full`, e.g. VAD enabled
    // with a garbage `vad_model_path`, producing
    //   `whisper_vad_init_from_file_with_params: failed to open VAD model '…'`
    // even though the caller passed `vad: false`. Fall back to the
    // by-value entry points when running against an older libwhisper so
    // existing macOS / ARM64 builds keep loading.
    final hasInitByRef =
        _lib.providesSymbol('whisper_init_from_file_with_params_by_ref');
    final hasFullByRef = _lib.providesSymbol('whisper_full_by_ref');
    if (hasInitByRef) {
      _initByRef = _lib.lookupFunction<_WhisperInitNative, _WhisperInit>(
          'whisper_init_from_file_with_params_by_ref');
    } else {
      _initByRef = _lib.lookupFunction<_WhisperInitNative, _WhisperInit>(
          'whisper_init_from_file_with_params');
    }
    if (hasFullByRef) {
      _full = _lib
          .lookupFunction<_WhisperFullNative, _WhisperFull>('whisper_full_by_ref');
    } else {
      _full = _lib.lookupFunction<_WhisperFullNative, _WhisperFull>('whisper_full');
    }

    final ctxDefault = _lib.lookupFunction<_DefaultCtxParamsNative, _DefaultCtxParams>(
        'whisper_context_default_params_by_ref')();
    final pathPtr = modelPath.toNativeUtf8();
    _ctx = _initByRef(pathPtr, ctxDefault);
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
    if (_lib.providesSymbol('crispasr_params_set_best_of')) {
      _paramsSetBestOf = _lib.lookupFunction<_ParamsSetIntNative, _ParamsSetInt>('crispasr_params_set_best_of');
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
    if (_lib.providesSymbol('crispasr_token_n_alts')) {
      _tokenNAlts = _lib.lookupFunction<_TokenNAltsNative, _TokenNAlts>('crispasr_token_n_alts');
    }
    if (_lib.providesSymbol('crispasr_token_alt_id')) {
      _tokenAltId = _lib.lookupFunction<_TokenAltIdNative, _TokenAltId>('crispasr_token_alt_id');
    }
    if (_lib.providesSymbol('crispasr_token_alt_p')) {
      _tokenAltP = _lib.lookupFunction<_TokenAltPNative, _TokenAltP>('crispasr_token_alt_p');
    }
    if (_lib.providesSymbol('whisper_token_to_str')) {
      _whisperTokenToStr =
          _lib.lookupFunction<_WhisperTokenToStrNative, _WhisperTokenToStr>('whisper_token_to_str');
    }
    if (_lib.providesSymbol('crispasr_params_set_alt_n')) {
      _paramsSetAltN = _lib.lookupFunction<_ParamsSetIntNative, _ParamsSetInt>('crispasr_params_set_alt_n');
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
    if (_lib.providesSymbol('crispasr_stream_set_live_decode')) {
      _streamSetLiveDecode = _lib.lookupFunction<_StreamSetLiveDecodeNative, _StreamSetLiveDecode>('crispasr_stream_set_live_decode');
    }

    // params_set additions for full C-ABI parity.
    if (_lib.providesSymbol('crispasr_params_set_max_tokens')) {
      _paramsSetMaxTokens = _lib.lookupFunction<_ParamsSetIntNative, _ParamsSetInt>('crispasr_params_set_max_tokens');
    }
    if (_lib.providesSymbol('crispasr_params_set_no_context')) {
      _paramsSetNoContext = _lib.lookupFunction<_ParamsSetBoolNative, _ParamsSetBool>('crispasr_params_set_no_context');
    }
    if (_lib.providesSymbol('crispasr_params_set_single_segment')) {
      _paramsSetSingleSegment = _lib.lookupFunction<_ParamsSetBoolNative, _ParamsSetBool>('crispasr_params_set_single_segment');
    }
    if (_lib.providesSymbol('crispasr_params_set_suppress_blank')) {
      _paramsSetSuppressBlank = _lib.lookupFunction<_ParamsSetBoolNative, _ParamsSetBool>('crispasr_params_set_suppress_blank');
    }
    if (_lib.providesSymbol('crispasr_params_set_temperature')) {
      _paramsSetTemperature = _lib.lookupFunction<_ParamsSetFloatNative, _ParamsSetFloat>('crispasr_params_set_temperature');
    }

    // token_alt_text: fills a buffer with the alt-candidate text string.
    if (_lib.providesSymbol('crispasr_token_alt_text')) {
      _tokenAltText = _lib.lookupFunction<_TokenAltTextNative, _TokenAltText>('crispasr_token_alt_text');
    }

    // VAD slices (unified dispatcher).
    if (_lib.providesSymbol('crispasr_vad_slices')) {
      _vadSlices = _lib.lookupFunction<_VadSlicesNative, _VadSlices>('crispasr_vad_slices');
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
      if (opts.bestOf > 1) _paramsSetBestOf?.call(params, opts.bestOf);
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
      if (opts.maxTokens > 0) _paramsSetMaxTokens?.call(params, opts.maxTokens);
      if (opts.noContext) _paramsSetNoContext?.call(params, 1);
      if (opts.singleSegment) _paramsSetSingleSegment?.call(params, 1);
      if (!opts.suppressBlank) _paramsSetSuppressBlank?.call(params, 0);
      if (opts.temperature > 0.0) _paramsSetTemperature?.call(params, opts.temperature);
      // Alt-token capture (0 = off, default). Pre-0.5.13 dylibs lack
      // the setter — silently skip so callers stay forward-compatible.
      if (opts.altN > 0) _paramsSetAltN?.call(params, opts.altN);

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
          // Pull alt-token candidates when the loaded dylib supports
          // them AND alt_n was actually set; n_alts returns 0 in both
          // unsupported and "not captured" cases so a single guard
          // covers both branches.
          var alts = const <AltToken>[];
          if (_tokenNAlts != null &&
              _tokenAltId != null &&
              _tokenAltP != null &&
              _whisperTokenToStr != null) {
            final nA = _tokenNAlts!(_ctx, i, k);
            if (nA > 0) {
              final list = <AltToken>[];
              for (var a = 0; a < nA; a++) {
                final aid = _tokenAltId!(_ctx, i, k, a);
                final ap = _tokenAltP!(_ctx, i, k, a);
                final ts = _whisperTokenToStr!(_ctx, aid);
                final atext = ts == nullptr ? '' : ts.toDartString();
                if (atext.isEmpty) continue;
                list.add(AltToken(text: atext, p: ap));
              }
              alts = list;
            }
          }
          words.add(Word(
            text: tok,
            start: _tokenT0!(_ctx, i, k) / 100.0,
            end:   _tokenT1!(_ctx, i, k) / 100.0,
            p:     _tokenP!(_ctx, i, k),
            alts: alts,
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

  /// Run CrispASR's unified VAD dispatcher on [pcm]. Returns speech spans
  /// in seconds. Uses Silero, FireRedVAD, MarbleNet, or Whisper-VAD-EncDec
  /// depending on the concrete model at [modelPath].
  ///
  /// Compared to [vad], this routes through the shared VAD dispatcher and
  /// returns float pairs in seconds (not centiseconds).
  List<VadSpan> vadSlices(
    Float32List pcm, {
    required String modelPath,
    int sampleRate = 16000,
    double threshold = 0.0,
    int minSpeechMs = 250,
    int minSilenceMs = 100,
    int speechPadMs = 30,
    double maxChunkDurationS = 30.0,
    int nThreads = 4,
  }) {
    if (_vadSlices == null || _vadFree == null) {
      throw UnsupportedError(
          'VAD slices helper not available — rebuild CrispASR with 0.6.0+ helpers.');
    }
    final samples = calloc<Float>(pcm.length);
    for (var i = 0; i < pcm.length; i++) {
      samples[i] = pcm[i];
    }
    final modelPtr = modelPath.toNativeUtf8();
    final outPtr = calloc<Pointer<Float>>();
    try {
      final n = _vadSlices!(
        modelPtr, samples, pcm.length, sampleRate, threshold,
        minSpeechMs, minSilenceMs, speechPadMs,
        maxChunkDurationS, nThreads, outPtr,
      );
      if (n < 0) throw Exception('VAD slices failed (error $n)');
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
  /// Uses crispasr's sliding-window trick: every [stepMs] of fresh
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
      setLiveDecode: _streamSetLiveDecode,
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
/// there is no Dart finalizer hooking the native library.
class StreamingSession {
  StreamingSession._({
    required Pointer<Void> handle,
    required _StreamFeed feed,
    required _StreamFlush? flush,
    required _StreamGetText getText,
    required _StreamClose close,
    _StreamSetLiveDecode? setLiveDecode,
  })  : _handle = handle,
        _feedFn = feed,
        _flushFn = flush,
        _getTextFn = getText,
        _closeFn = close,
        _setLiveDecodeFn = setLiveDecode;

  final Pointer<Void> _handle;
  final _StreamFeed    _feedFn;
  final _StreamFlush?  _flushFn;
  final _StreamGetText _getTextFn;
  final _StreamClose   _closeFn;
  final _StreamSetLiveDecode? _setLiveDecodeFn;

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

  /// Toggle live-decode mode. When enabled (1), the streaming session
  /// decodes every feed() call immediately instead of waiting for the
  /// step threshold. Useful for ultra-low-latency UIs at higher CPU cost.
  void setLiveDecode(bool enabled) {
    if (_closed) throw StateError('StreamingSession is closed');
    if (_setLiveDecodeFn == null) {
      throw UnsupportedError(
          'crispasr_stream_set_live_decode not available in this libcrispasr build');
    }
    _setLiveDecodeFn!(_handle, enabled ? 1 : 0);
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

  /// Open a session with explicit runtime knobs (CrispASR 0.6.1+).
  ///
  /// Wraps `crispasr_session_open_with_params`. Use this when the host
  /// app wants to toggle GPU offload or verbosity per session — the
  /// historical [open] factory always defaulted to GPU-on / silent.
  ///
  /// `useGpu = false` forces every backend that has a `use_gpu` field
  /// in its context_params to take the CPU path. Backends without that
  /// field (whisper-via-context, kyutai-stt, firered-asr, glm-asr,
  /// moonshine-streaming, gemma4-e2b, m2m100, mimo-asr, omniasr,
  /// canary-ctc, voxtral4b, wav2vec2) ignore the toggle silently —
  /// their GPU path is decided at compile time.
  ///
  /// Throws [UnsupportedError] when the loaded dylib predates 0.6.1
  /// (no `crispasr_session_open_with_params` symbol). Falls back to
  /// the regular [open] in that case via the binding's caller, not
  /// here.
  factory CrispasrSession.openWithParams(
    String modelPath, {
    int nThreads = 4,
    bool useGpu = true,
    int verbosity = 0,
    /// CrispASR 0.6.2+: enable flash-attention on backends that have
    /// a flash-attn path (whisper today; LLM backends incrementally).
    /// Defaults true to match the ggml convention.
    bool flashAttn = true,
    /// CrispASR 0.6.2+: cap on GPU-offloaded transformer layers for
    /// LLM-based backends. -1 = "as many as possible" (the C-side
    /// sentinel); 0 = CPU-only LLM inference; >0 = bounded.
    int nGpuLayers = -1,
    String? backend,
    String? libPath,
  }) {
    final lib = DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
    if (!lib.providesSymbol('crispasr_session_open_with_params')) {
      throw UnsupportedError(
          'crispasr_session_open_with_params missing — needs CrispASR 0.6.1+. '
          'Use CrispasrSession.open() for older builds.');
    }
    // ABI struct layout (see crispasr_open_params_v1 in
    // src/crispasr_c_api.cpp): int32 abi_version, n_threads, use_gpu,
    // verbosity, flash_attn, n_gpu_layers + 6 reserved int32. Total
    // 48 bytes — same size as the v1 layout (we used 2 of the 8
    // reserved slots).
    final paramsPtr = calloc<Uint8>(48);
    final ints = paramsPtr.cast<Int32>();
    ints[0] = 2;          // abi_version (v2 — opt into flash_attn / n_gpu_layers)
    ints[1] = nThreads;   // n_threads
    ints[2] = useGpu ? 1 : 0;
    ints[3] = verbosity;
    ints[4] = flashAttn ? 1 : 0;
    ints[5] = nGpuLayers;
    final pathPtr = modelPath.toNativeUtf8();
    final bePtr = backend != null && backend.isNotEmpty
        ? backend.toNativeUtf8()
        : nullptr;
    try {
      final fn = lib.lookupFunction<
          Pointer<Void> Function(
              Pointer<Utf8>, Pointer<Utf8>, Pointer<Uint8>),
          Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>,
              Pointer<Uint8>)>('crispasr_session_open_with_params');
      final handle = fn(pathPtr, bePtr.cast<Utf8>(), paramsPtr);
      if (handle == nullptr) {
        throw Exception(
            'crispasr_session_open_with_params returned null — either the '
            'GGUF backend isn\'t one of ${_availableBackends(lib).join(", ")} '
            'or the file is unreadable.');
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
      calloc.free(paramsPtr);
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
    // Two-call protocol: first call with a probe buffer reads the
    // full list length from the return value, then re-allocate at
    // (length+1) bytes so trailing entries don't get cut off. The
    // hard-coded 256-byte buffer that used to live here truncated
    // the list at "omn" once CrispASR crossed ~30 backends — every
    // entry past that point silently disappeared, and CrisperWeaver's
    // front-door check rejected models whose backend happened to fall
    // off the cliff (#7 second wave: moonshine / mimo-asr /
    // omniasr-llm-unlimited all reported as missing).
    const probeCap = 256;
    final probe = calloc<Uint8>(probeCap);
    int needed;
    try {
      needed = fn(probe.cast<Utf8>(), probeCap);
    } finally {
      calloc.free(probe);
    }
    if (needed <= 0) return const <String>[];
    // Round up generously — `needed` is the list size at the moment
    // of the probe; another thread might extend the list before the
    // second call (unlikely in practice but cheap to guard).
    final cap = needed + 64;
    final buf = calloc<Uint8>(cap);
    try {
      final ptr = buf.cast<Utf8>();
      fn(ptr, cap);
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

  /// Transcribe with Silero VAD segmentation + crispasr-style stitching.
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
    final intView = (optsPtr + 4).cast<Int32>();
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
    // crispasr_session_result_word_p was added 2026-05-02 to surface
    // per-word probabilities for backends that emit them. Older
    // libwhisper builds don't have the symbol — probe and fall back to
    // 1.0 (the documented "uniform confidence" default) so this
    // binding stays loadable against pre-2026-05 dylibs.
    final wordPFn = _lib.providesSymbol('crispasr_session_result_word_p')
        ? _lib.lookupFunction<
            Float Function(Pointer<Void>, Int32, Int32),
            double Function(Pointer<Void>, int, int)>(
            'crispasr_session_result_word_p')
        : null;
    // 0.5.13: per-word top-N alternative candidates. All three symbols
    // landed together so a single guard covers presence; pre-0.5.13
    // dylibs report no alts and the UI hides the affordance.
    final wordNAltsFn = _lib.providesSymbol('crispasr_session_result_word_n_alts')
        ? _lib.lookupFunction<
            Int32 Function(Pointer<Void>, Int32, Int32),
            int Function(Pointer<Void>, int, int)>(
            'crispasr_session_result_word_n_alts')
        : null;
    final wordAltTextFn = _lib.providesSymbol('crispasr_session_result_word_alt_text')
        ? _lib.lookupFunction<
            Pointer<Utf8> Function(Pointer<Void>, Int32, Int32, Int32),
            Pointer<Utf8> Function(Pointer<Void>, int, int, int)>(
            'crispasr_session_result_word_alt_text')
        : null;
    final wordAltPFn = _lib.providesSymbol('crispasr_session_result_word_alt_p')
        ? _lib.lookupFunction<
            Float Function(Pointer<Void>, Int32, Int32, Int32),
            double Function(Pointer<Void>, int, int, int)>(
            'crispasr_session_result_word_alt_p')
        : null;

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
        // -1.0 from the C side means "no per-word probability for this
        // backend" — clamp to 1.0 so the UI renders neutrally rather
        // than treating it as zero confidence.
        var p = wordPFn == null ? 1.0 : wordPFn(res, i, k);
        if (p < 0) p = 1.0;
        var alts = const <AltToken>[];
        if (wordNAltsFn != null && wordAltTextFn != null && wordAltPFn != null) {
          final nA = wordNAltsFn(res, i, k);
          if (nA > 0) {
            final list = <AltToken>[];
            for (var a = 0; a < nA; a++) {
              final tp2 = wordAltTextFn(res, i, k, a);
              final atext = tp2 == nullptr ? '' : tp2.toDartString();
              if (atext.isEmpty) continue;
              list.add(AltToken(text: atext, p: wordAltPFn(res, i, k, a)));
            }
            alts = list;
          }
        }
        words.add(Word(
          text: wt,
          start: wordT0(res, i, k) / 100.0,
          end:   wordT1(res, i, k) / 100.0,
          p: p,
          alts: alts,
        ));
      }
      out.add(SessionSegment(text: text.trim(), start: t0, end: t1, words: words));
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // TTS synthesis (vibevoice, qwen3-tts, kokoro, orpheus, chatterbox, zonos-tts, lfm2-audio, and others)
  // ---------------------------------------------------------------------------

  /// Load a separate codec GGUF (qwen3-tts only; no-op for other backends).
  void setCodecPath(String path) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_codec_path')) {
      throw UnsupportedError('TTS codec API not available in this libcrispasr build');
    }
    final fn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>, Pointer<Utf8>),
        int Function(Pointer<Void>, Pointer<Utf8>)>('crispasr_session_set_codec_path');
    final p = path.toNativeUtf8();
    try {
      final rc = fn(_handle, p);
      if (rc != 0) throw Exception('setCodecPath failed (rc=$rc) for backend $_backend');
    } finally {
      calloc.free(p);
    }
  }

  /// Drop the kokoro per-session phoneme cache. No-op for non-kokoro backends.
  /// Useful for long-running daemons that resynthesize across many speakers
  /// and want bounded memory. (PLAN #56 #5)
  void clearPhonemeCache() {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_kokoro_clear_phoneme_cache')) {
      return; // older libcrispasr build — no-op
    }
    final fn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>),
        int Function(Pointer<Void>)>('crispasr_session_kokoro_clear_phoneme_cache');
    final rc = fn(_handle);
    if (rc != 0) throw Exception('clearPhonemeCache failed (rc=$rc)');
  }

  // ---------------------------------------------------------------------------
  // Sticky session-state setters (PLAN #59 partial unblock).
  // ---------------------------------------------------------------------------

  /// Sticky source-language hint (canary, cohere, voxtral, whisper).
  /// Empty string clears. Per-call language arg passed to transcribe
  /// methods still wins.
  void setSourceLanguage(String lang) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_source_language')) {
      throw UnsupportedError('session-state API not present in this libcrispasr build');
    }
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Utf8>),
        int Function(Pointer<Void>, Pointer<Utf8>)>('crispasr_session_set_source_language');
    final p = lang.toNativeUtf8();
    try {
      final rc = fn(_handle, p);
      if (rc != 0) throw Exception('setSourceLanguage failed (rc=$rc)');
    } finally {
      calloc.free(p);
    }
  }

  /// Sticky target-language. When set ≠ source on canary/cohere, the backend
  /// emits a translation. For whisper, pair with [setTranslate]`(true)`.
  void setTargetLanguage(String lang) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_target_language')) {
      throw UnsupportedError('session-state API not present in this libcrispasr build');
    }
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Utf8>),
        int Function(Pointer<Void>, Pointer<Utf8>)>('crispasr_session_set_target_language');
    final p = lang.toNativeUtf8();
    try {
      final rc = fn(_handle, p);
      if (rc != 0) throw Exception('setTargetLanguage failed (rc=$rc)');
    } finally {
      calloc.free(p);
    }
  }

  /// Toggle punctuation + capitalisation in the output (canary/cohere
  /// natively; LLM backends via post-process strip). Default true.
  void setPunctuation(bool enable) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_punctuation')) {
      throw UnsupportedError('session-state API not present in this libcrispasr build');
    }
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Int32),
        int Function(Pointer<Void>, int)>('crispasr_session_set_punctuation');
    final rc = fn(_handle, enable ? 1 : 0);
    if (rc != 0) throw Exception('setPunctuation failed (rc=$rc)');
  }

  /// Select + load a punctuation-restoration model on the session.
  ///
  /// [model] is an alias (`auto` / `firered` / `fullstop` / `punctuate-all` /
  /// `pcs`) or a path to a `.gguf`; `'none'` or `''` unloads. Auto-downloads on
  /// first use. Restores punctuation on backends that emit none (parakeet
  /// RNNT/CTC, etc.) — the same post-processor the CLI `--punc-model` applies.
  void setPuncModel(String model) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_punc_model')) {
      throw UnsupportedError('punc-model API not present in this libcrispasr build');
    }
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Utf8>),
        int Function(Pointer<Void>, Pointer<Utf8>)>('crispasr_session_set_punc_model');
    final p = model.toNativeUtf8();
    try {
      final rc = fn(_handle, p);
      if (rc != 0) throw Exception('setPuncModel failed (rc=$rc)');
    } finally {
      calloc.free(p);
    }
  }

  /// Select the G2P pronunciation dictionary for TTS phonemization
  /// (`olaph` / `open-dict` or a path). Empty string keeps the default.
  void setG2pDict(String source) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_g2p_dict')) {
      throw UnsupportedError('crispasr_session_set_g2p_dict not present in this libcrispasr build');
    }
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Utf8>),
        int Function(Pointer<Void>, Pointer<Utf8>)>('crispasr_session_set_g2p_dict');
    final p = source.toNativeUtf8();
    try {
      final rc = fn(_handle, p);
      if (rc != 0) throw Exception('setG2pDict failed (rc=$rc)');
    } finally {
      calloc.free(p);
    }
  }

  /// Whisper sticky `--translate`. For canary/cohere/voxtral the equivalent
  /// is [setTargetLanguage] ≠ source.
  void setTranslate(bool enable) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_translate')) {
      throw UnsupportedError('session-state API not present in this libcrispasr build');
    }
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Int32),
        int Function(Pointer<Void>, int)>('crispasr_session_set_translate');
    final rc = fn(_handle, enable ? 1 : 0);
    if (rc != 0) throw Exception('setTranslate failed (rc=$rc)');
  }

  /// Sticky audio Q&A prompt for instruct-tuned audio-LLM backends
  /// (voxtral / voxtral4b / qwen3-asr). When set, the backend answers
  /// the question instead of producing a verbatim transcript:
  ///
  ///     session.setAsk("What is the speaker's tone?");
  ///     final segs = session.transcribe(pcm);
  ///     // segs[0].text == "The speaker sounds calm and measured..."
  ///
  /// Pass an empty string to clear and resume verbatim transcription.
  /// Backends without an instruct-tuned LLM head ignore the call —
  /// it's a no-op rather than an error so callers can set it
  /// unconditionally.
  void setAsk(String prompt) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_ask')) {
      throw UnsupportedError('setAsk API not present in this libcrispasr build');
    }
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Utf8>),
        int Function(Pointer<Void>, Pointer<Utf8>)>('crispasr_session_set_ask');
    final p = prompt.toNativeUtf8();
    try {
      final rc = fn(_handle, p);
      if (rc != 0) throw Exception('setAsk failed (rc=$rc)');
    } finally {
      calloc.free(p);
    }
  }

  /// Set best-of-N decoding on the session. For Whisper this maps to
  /// `wparams.greedy.best_of` (Whisper picks the highest-scoring of N
  /// internal decodes). For every other backend the C-side runs N
  /// independent transcribes per call and returns the one with the
  /// highest average per-token confidence — so this slider works
  /// across the full backend set, not just sampling-capable ones.
  /// `n <= 1` is a no-op (single decode, the historical default).
  void setBestOf(int n) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_best_of')) {
      throw UnsupportedError(
          'crispasr_session_set_best_of not present in this libcrispasr build');
    }
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Int32),
        int Function(Pointer<Void>, int)>('crispasr_session_set_best_of');
    final rc = fn(_handle, n);
    if (rc != 0) throw Exception('setBestOf failed (rc=$rc)');
  }

  /// §90 Set beam-search width for the next transcription.
  ///
  /// `n` >= 2 activates beam search for backends that support it:
  ///   whisper        — native BEAM_SEARCH strategy
  ///   qwen3-asr      — via core_beam_decode::run_with_probs (replay)
  ///   granite*        — via core_beam_decode::run_with_probs
  ///   voxtral         — via core_beam_decode::run_with_probs
  ///   glm-asr         — glm_asr_set_beam_size (per-backend setter)
  ///   kyutai-stt      — kyutai_stt_set_beam_size
  ///   firered         — firered_asr_set_beam_size
  ///   moonshine       — moonshine_set_beam_size
  ///   omniasr-llm     — omniasr_set_beam_size
  ///   canary          — canary_set_beam_size (branched-KV AED beam)
  ///   cohere          — cohere_set_beam_size (branched-KV AED beam)
  /// Silent no-op for voxtral4b, CTC/NAR backends.
  /// `n` <= 0 or 1 reverts to greedy (default).
  ///
  /// Only available when built with beam-search support (symbol presence
  /// is checked at runtime; throws on older native builds).
  void setBeamSize(int n) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_beam_size')) {
      throw UnsupportedError(
          'crispasr_session_set_beam_size not present in this libcrispasr build');
    }
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Int32),
        int Function(Pointer<Void>, int)>('crispasr_session_set_beam_size');
    final rc = fn(_handle, n);
    if (rc != 0) throw Exception('setBeamSize failed (rc=$rc)');
  }

  /// GBNF grammar-constrained sampling (whisper only — other backends
  /// silently ignore; the whisper transcribe path auto-switches to
  /// beam search when grammar is active because the constrained sampler
  /// requires beam ≥ 2).
  ///
  /// Pass an empty `text` to disable the grammar and resume verbatim
  /// decoding. `rootRule` is the symbol name to start parsing from
  /// (the GBNF convention is "root"). `penalty` is whisper's
  /// `grammar_penalty` scalar — the upstream default is 100.0.
  ///
  /// Throws [ArgumentError] when the GBNF source is invalid or the
  /// root rule isn't present, and [UnsupportedError] when the loaded
  /// dylib predates 0.5.9 (no `crispasr_session_set_grammar_text`
  /// symbol). Catch the latter for graceful fallback to unconstrained
  /// decoding on older builds.
  ///
  /// Example — force a JSON-shaped output:
  /// ```dart
  /// session.setGrammar(
  ///   'root ::= "{" key ":" value "}"\n'
  ///   'key   ::= "\\"" [a-zA-Z]+ "\\""\n'
  ///   'value ::= [0-9]+\n',
  ///   rootRule: 'root',
  /// );
  /// ```
  void setGrammar(String text,
      {String rootRule = 'root', double penalty = 100.0}) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_grammar_text')) {
      throw UnsupportedError(
          'crispasr_session_set_grammar_text not present in this libcrispasr build — '
          'rebuild against CrispASR 0.5.9+');
    }
    final fn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, Float),
        int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>,
            double)>('crispasr_session_set_grammar_text');
    final textPtr =
        text.isEmpty ? Pointer<Utf8>.fromAddress(0) : text.toNativeUtf8();
    final rulePtr =
        rootRule.isEmpty ? Pointer<Utf8>.fromAddress(0) : rootRule.toNativeUtf8();
    try {
      final rc = fn(_handle, textPtr, rulePtr, penalty);
      if (rc == -1) throw StateError('session is null');
      if (rc == -2) {
        throw ArgumentError(
            'invalid GBNF source or root rule "$rootRule" not found in grammar');
      }
      if (rc != 0) throw Exception('setGrammar failed (rc=$rc)');
    } finally {
      if (textPtr != Pointer<Utf8>.fromAddress(0)) calloc.free(textPtr);
      if (rulePtr != Pointer<Utf8>.fromAddress(0)) calloc.free(rulePtr);
    }
  }

  /// Convenience: clear any previously-set grammar so the next
  /// transcribe call decodes unconstrained again.
  void clearGrammar() => setGrammar('');

  /// Whisper text-suppression + prompt-carry extras (whisper-only;
  /// other backends silently ignore). Effective on CrispASR
  /// 0.5.11+.
  ///
  /// All three map directly onto `whisper_full_params` fields:
  ///
  /// * [suppressNonSpeechTokens] — when true, whisper drops
  ///   `[LAUGHTER]` / `[MUSIC]` / `[NOISE]` markers from the
  ///   output. Maps to `wparams.suppress_nst`. Default false
  ///   (= keep the markers, matches stock whisper.cpp).
  /// * [suppressRegex] — Posix regex; tokens whose text matches
  ///   are dropped during decoding. Empty string disables.
  ///   Useful for purging frequent hallucinated tokens or
  ///   speaker-tag patterns the model leaks. Maps to
  ///   `wparams.suppress_regex`.
  /// * [carryInitialPrompt] — when true, whisper prepends the
  ///   initial prompt to every decode window (not just the
  ///   first). Useful for vocabulary biasing on long audio at
  ///   the cost of weakening context conditioning. Maps to
  ///   `wparams.carry_initial_prompt`. Default false.
  ///
  /// Throws [UnsupportedError] when the loaded dylib predates
  /// 0.5.11 (no `crispasr_session_set_whisper_decode_extras`
  /// symbol). Callers should catch + graceful-degrade.
  void setWhisperDecodeExtras({
    bool suppressNonSpeechTokens = false,
    String suppressRegex = '',
    bool carryInitialPrompt = false,
  }) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol(
        'crispasr_session_set_whisper_decode_extras')) {
      throw UnsupportedError(
          'crispasr_session_set_whisper_decode_extras not present in this libcrispasr build — '
          'rebuild against CrispASR 0.5.11+');
    }
    final fn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>, Int32, Pointer<Utf8>, Int32),
        int Function(Pointer<Void>, int, Pointer<Utf8>,
            int)>('crispasr_session_set_whisper_decode_extras');
    final regexPtr = suppressRegex.isEmpty
        ? Pointer<Utf8>.fromAddress(0)
        : suppressRegex.toNativeUtf8();
    try {
      final rc = fn(_handle, suppressNonSpeechTokens ? 1 : 0, regexPtr,
          carryInitialPrompt ? 1 : 0);
      if (rc != 0) {
        throw Exception('setWhisperDecodeExtras failed (rc=$rc)');
      }
    } finally {
      if (regexPtr != Pointer<Utf8>.fromAddress(0)) {
        calloc.free(regexPtr);
      }
    }
  }

  /// Whisper decoder-fallback thresholds (whisper-only). Each
  /// value is written into `whisper_full_params` on every
  /// transcribe dispatch; non-whisper backends silently ignore
  /// because their wparams have no analog. Effective on
  /// CrispASR 0.5.10+.
  ///
  /// Parameters (all optional — omit to keep the session's
  /// current value):
  ///
  /// * [entropyThold] — per-token entropy that triggers a
  ///   fallback pass. Default 2.4. Lower = stricter
  ///   (fallback fires more often); raise for hard audio to
  ///   suppress repeated fallbacks.
  /// * [logprobThold] — avg log-probability cutoff that
  ///   triggers a fallback pass. Default -1.0 (= "any
  ///   decoding worse than -1 logprob retries"). Set more
  ///   negative to be tolerant of noisy decoding.
  /// * [noSpeechThold] — silence detector cutoff. Default
  ///   0.6. Higher = more conservative (less likely to drop
  ///   real speech as silence); lower = aggressive silence
  ///   gating.
  /// * [temperatureInc] — temperature step per fallback
  ///   pass. Default 0.2. Set to 0.0 to disable the
  ///   fallback loop entirely (= the CLI's `--no-fallback`).
  ///
  /// Throws [UnsupportedError] when the loaded dylib
  /// predates 0.5.10 (no `crispasr_session_set_fallback_thresholds`
  /// symbol). Callers should catch and downgrade gracefully —
  /// the C side wouldn't have honoured the values anyway.
  void setFallbackThresholds({
    double entropyThold = 2.4,
    double logprobThold = -1.0,
    double noSpeechThold = 0.6,
    double temperatureInc = 0.2,
  }) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_fallback_thresholds')) {
      throw UnsupportedError(
          'crispasr_session_set_fallback_thresholds not present in this libcrispasr build — '
          'rebuild against CrispASR 0.5.10+');
    }
    final fn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>, Float, Float, Float, Float),
        int Function(Pointer<Void>, double, double, double,
            double)>('crispasr_session_set_fallback_thresholds');
    final rc = fn(_handle, entropyThold, logprobThold, noSpeechThold,
        temperatureInc);
    if (rc != 0) {
      throw Exception('setFallbackThresholds failed (rc=$rc)');
    }
  }

  /// Sticky setter for per-token top-N alternative-candidate capture
  /// (whisper greedy decode only). 0 (default) = off.
  ///
  /// When `n > 0`, every greedy-sampled token also retains its top-N
  /// runner-up candidates, which the session-result accessors surface
  /// via [Word.alts]. Useful for "tap an ambiguous word in the
  /// transcript editor and pick a competing token" UIs. The model
  /// emits these only when it was genuinely uncertain — for confident
  /// tokens the alternates will all be tiny-probability filler.
  ///
  /// Beam search is intentionally excluded: its siblings are
  /// beam-conditional rather than greedy alternatives, so the
  /// semantics are different and conflating them would mislead users.
  ///
  /// Throws [UnsupportedError] when the loaded dylib predates 0.5.13
  /// (no `crispasr_session_set_alt_n` symbol). Callers should catch
  /// and downgrade gracefully — the C side wouldn't have honoured
  /// the value anyway.
  void setAltN(int n) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_alt_n')) {
      throw UnsupportedError(
          'crispasr_session_set_alt_n not present in this libcrispasr build — '
          'rebuild against CrispASR 0.5.13+');
    }
    final fn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>, Int32),
        int Function(Pointer<Void>, int)>('crispasr_session_set_alt_n');
    final rc = fn(_handle, n);
    if (rc != 0) {
      throw Exception('setAltN failed (rc=$rc)');
    }
  }

  /// Set decoder temperature on backends that support runtime control
  /// (canary, cohere, parakeet, moonshine). Other backends silently no-op.
  /// `seed` is the RNG seed; pass 0 for time-based.
  void setTemperature(double temperature, {int seed = 0}) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_temperature')) {
      throw UnsupportedError('session-state API not present in this libcrispasr build');
    }
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Float, Uint64),
        int Function(Pointer<Void>, double, int)>('crispasr_session_set_temperature');
    final rc = fn(_handle, temperature, seed);
    // rc == -2 means no backend in this session supports temperature — soft no-op.
    if (rc != 0 && rc != -2) {
      throw Exception('setTemperature failed (rc=$rc)');
    }
  }

  /// Set the diffusion / CFM step count for diffusion-based TTS
  /// backends (chatterbox today). Higher = better fidelity, slower.
  /// Returns silently when the active backend has no diffusion stage
  /// (rc=-2 from the C side maps to a soft no-op here).
  void setTtsSteps(int steps) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_tts_steps')) {
      // Older libcrispasr without the symbol — silent no-op so the
      // UI slider still works when bound against pre-0.6.1 dylibs.
      return;
    }
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Int32),
        int Function(Pointer<Void>, int)>('crispasr_session_set_tts_steps');
    final rc = fn(_handle, steps);
    if (rc != 0 && rc != -2) {
      throw Exception('setTtsSteps failed (rc=$rc)');
    }
  }

  /// Top-p nucleus sampling threshold (0.0..1.0). Honoured by
  /// chatterbox; other backends no-op.
  void setTopP(double topP) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_top_p')) return;
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Float),
        int Function(Pointer<Void>, double)>('crispasr_session_set_top_p');
    final rc = fn(_handle, topP);
    if (rc != 0 && rc != -2) throw Exception('setTopP failed (rc=$rc)');
  }

  /// Min-p sampling threshold (0.0..1.0). Honoured by chatterbox.
  void setMinP(double minP) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_min_p')) return;
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Float),
        int Function(Pointer<Void>, double)>('crispasr_session_set_min_p');
    final rc = fn(_handle, minP);
    if (rc != 0 && rc != -2) throw Exception('setMinP failed (rc=$rc)');
  }

  /// Repetition penalty (1.0 = no penalty; >1 discourages repeats).
  void setRepetitionPenalty(double r) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_repetition_penalty')) return;
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Float),
            int Function(Pointer<Void>, double)>(
        'crispasr_session_set_repetition_penalty');
    final rc = fn(_handle, r);
    if (rc != 0 && rc != -2) {
      throw Exception('setRepetitionPenalty failed (rc=$rc)');
    }
  }

  /// Classifier-free-guidance weight (chatterbox). 0 disables CFG;
  /// 0.5 is the upstream default.
  void setCfgWeight(double cfg) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_cfg_weight')) return;
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Float),
        int Function(Pointer<Void>, double)>('crispasr_session_set_cfg_weight');
    final rc = fn(_handle, cfg);
    if (rc != 0 && rc != -2) throw Exception('setCfgWeight failed (rc=$rc)');
  }

  /// Emotion-exaggeration scalar (chatterbox). 0.5 default.
  void setExaggeration(double exaggeration) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_exaggeration')) return;
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Float),
            int Function(Pointer<Void>, double)>(
        'crispasr_session_set_exaggeration');
    final rc = fn(_handle, exaggeration);
    if (rc != 0 && rc != -2) {
      throw Exception('setExaggeration failed (rc=$rc)');
    }
  }

  /// Upper bound on speech tokens generated per synthesize call
  /// (chatterbox). Default 1000 ≈ 20 s.
  void setMaxSpeechTokens(int n) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_max_speech_tokens')) return;
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Int32),
            int Function(Pointer<Void>, int)>(
        'crispasr_session_set_max_speech_tokens');
    final rc = fn(_handle, n);
    if (rc != 0 && rc != -2) {
      throw Exception('setMaxSpeechTokens failed (rc=$rc)');
    }
  }

  /// Per-phoneme length-scale / speaking-rate scalar for TTS
  /// backends with a duration model. Honoured by kokoro today
  /// (PLAN #88); other backends silently no-op. 1.0 = upstream
  /// default; >1.0 = slower / longer; <1.0 = faster / shorter.
  /// Clamped to [0.25, 4.0] on the C side.
  void setLengthScale(double scale) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_length_scale')) return;
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Float),
            int Function(Pointer<Void>, double)>(
        'crispasr_session_set_length_scale');
    final rc = fn(_handle, scale);
    if (rc != 0 && rc != -2) {
      throw Exception('setLengthScale failed (rc=$rc)');
    }
  }

  /// Reseed TTS backends that support runtime seed control (chatterbox,
  /// vibevoice, qwen3-tts, orpheus). Other backends silently no-op.
  void setTtsSeed(int seed) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_tts_seed')) return;
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Uint64),
        int Function(Pointer<Void>, int)>('crispasr_session_set_tts_seed');
    final rc = fn(_handle, seed);
    if (rc != 0 && rc != -2) throw Exception('setTtsSeed failed (rc=$rc)');
  }

  /// Set a generated-token cap for autoregressive session backends.
  /// Pass <= 0 to clear the override and use the backend default.
  void setMaxNewTokens(int n) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_max_new_tokens')) return;
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Int32),
        int Function(Pointer<Void>, int)>('crispasr_session_set_max_new_tokens');
    final rc = fn(_handle, n);
    if (rc != 0) throw Exception('setMaxNewTokens failed (rc=$rc)');
  }

  /// Opt-in repeated-token penalty for autoregressive session backends.
  /// Pass <= 0 to disable.
  void setFrequencyPenalty(double penalty) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_frequency_penalty')) return;
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>, Float),
            int Function(Pointer<Void>, double)>(
        'crispasr_session_set_frequency_penalty');
    final rc = fn(_handle, penalty);
    if (rc != 0) throw Exception('setFrequencyPenalty failed (rc=$rc)');
  }

  /// Text-to-text translation via this session's backend.
  ///
  /// Routes through the C-side `crispasr_session_translate_text`, which
  /// dispatches to whichever translation-capable backend the session
  /// loaded — `m2m100` (and `m2m-100` / `translate` aliases) today, plus
  /// `gemma4-e2b` for backends that include a translation head. The
  /// session must have been opened against a model that supports
  /// translation; calling this on an ASR-only session returns `null`
  /// (or an empty string on the C side).
  ///
  /// [srcLang] and [tgtLang] are ISO 639-1 codes (`en`, `de`, `fr`, …).
  /// [maxTokens] caps the output length; pass 0 to use the C-side
  /// default of 200.
  ///
  /// Throws [UnsupportedError] when the loaded dylib is pre-0.6.0 and
  /// doesn't ship the symbol. Returns `null` when the C side rejects
  /// the request (no translation-capable backend in this session).
  String? translateText(String text, String srcLang, String tgtLang,
      {int maxTokens = 0}) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_translate_text')) {
      throw UnsupportedError(
          'translateText not in this libcrispasr — needs CrispASR 0.6.0+');
    }
    final fn = _lib.lookupFunction<
        Pointer<Utf8> Function(
            Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Int32),
        Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>,
            Pointer<Utf8>, int)>('crispasr_session_translate_text');
    final freeFn = _lib.providesSymbol('crispasr_session_translate_text_free')
        ? _lib.lookupFunction<Void Function(Pointer<Utf8>),
            void Function(Pointer<Utf8>)>('crispasr_session_translate_text_free')
        : null;
    final textPtr = text.toNativeUtf8();
    final srcPtr = srcLang.toNativeUtf8();
    final tgtPtr = tgtLang.toNativeUtf8();
    try {
      final res = fn(_handle, textPtr, srcPtr, tgtPtr, maxTokens);
      if (res == nullptr) return null;
      final out = res.toDartString();
      // Older builds free with the same allocator that allocated; on
      // builds that don't expose the symbol we fall back to `free` via
      // calloc.free, which on glibc/macOS libc is byte-compatible with
      // strdup-style malloc'd output.
      if (freeFn != null) {
        freeFn(res);
      } else {
        calloc.free(res);
      }
      return out;
    } finally {
      calloc.free(textPtr);
      calloc.free(srcPtr);
      calloc.free(tgtPtr);
    }
  }

  /// Auto-detect spoken language on raw 16 kHz mono PCM.
  ///
  /// `method`: 0=Whisper, 1=Silero (default), 2=Firered, 3=Ecapa.
  /// Returns a record `(lang, confidence)`.
  ({String lang, double confidence}) detectLanguage(
    Float32List pcm,
    String lidModelPath, {
    int method = 1,
  }) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_detect_language')) {
      throw UnsupportedError('session-state API not present in this libcrispasr build');
    }
    final fn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>, Pointer<Float>, Int32, Pointer<Utf8>, Int32, Pointer<Utf8>, Int32, Pointer<Float>),
        int Function(Pointer<Void>, Pointer<Float>, int, Pointer<Utf8>, int, Pointer<Utf8>, int,
            Pointer<Float>)>('crispasr_session_detect_language');
    final pcmPtr = calloc<Float>(pcm.length);
    final pathPtr = lidModelPath.toNativeUtf8();
    final outBuf = calloc<Uint8>(16);
    final probPtr = calloc<Float>(1);
    try {
      for (var i = 0; i < pcm.length; i++) {
        pcmPtr[i] = pcm[i];
      }
      final rc = fn(_handle, pcmPtr, pcm.length, pathPtr, method, outBuf.cast<Utf8>(), 16, probPtr);
      if (rc != 0) throw Exception('detectLanguage failed (rc=$rc)');
      return (lang: outBuf.cast<Utf8>().toDartString(), confidence: probPtr[0]);
    } finally {
      calloc.free(pcmPtr);
      calloc.free(pathPtr);
      calloc.free(outBuf);
      calloc.free(probPtr);
    }
  }

  /// Load a voice prompt: a baked GGUF voice pack OR a *.wav reference audio.
  ///
  /// For qwen3-tts a WAV reference requires [refText] (the transcription of
  /// the reference audio). For vibevoice only GGUF voice packs are supported.
  /// For orpheus voice selection is BY NAME — use [setSpeakerName] instead.
  void setVoice(String path, {String? refText}) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_voice')) {
      throw UnsupportedError('TTS voice API not available in this libcrispasr build');
    }
    final fn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>),
        int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>)>('crispasr_session_set_voice');
    final pathPtr = path.toNativeUtf8();
    final refPtr = refText != null ? refText.toNativeUtf8() : nullptr;
    try {
      final rc = fn(_handle, pathPtr, refPtr.cast());
      if (rc != 0) throw Exception('setVoice failed (rc=$rc) for backend $_backend');
    } finally {
      calloc.free(pathPtr);
      if (refPtr != nullptr) calloc.free(refPtr);
    }
  }

  /// Select a fixed/preset speaker by NAME (orpheus).
  ///
  /// Names are e.g. `"tara"`, `"leo"`, `"leah"` for the canopylabs English
  /// finetune; `"Anton"`, `"Sophie"`, etc. for the Kartoffel_Orpheus DE
  /// finetunes. Use [speakers] to enumerate names baked into the GGUF.
  void setSpeakerName(String name) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_speaker_name')) {
      throw UnsupportedError('setSpeakerName API not available in this libcrispasr build');
    }
    final fn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>, Pointer<Utf8>),
        int Function(Pointer<Void>, Pointer<Utf8>)>('crispasr_session_set_speaker_name');
    final namePtr = name.toNativeUtf8();
    try {
      final rc = fn(_handle, namePtr);
      if (rc == -2) {
        throw ArgumentError('unknown speaker: $name (call speakers() to enumerate)');
      }
      if (rc == -3) {
        throw StateError('backend $_backend has no preset speakers; use setVoice() instead');
      }
      if (rc != 0) throw Exception('setSpeakerName failed (rc=$rc) for backend $_backend');
    } finally {
      calloc.free(namePtr);
    }
  }

  /// Return the list of preset speaker names for the active backend.
  /// Empty if the backend has no preset-speaker contract.
  List<String> speakers() {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_n_speakers')) return const [];
    final nFn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>),
        int Function(Pointer<Void>)>('crispasr_session_n_speakers');
    final getFn = _lib.lookupFunction<
        Pointer<Utf8> Function(Pointer<Void>, Int32),
        Pointer<Utf8> Function(Pointer<Void>, int)>('crispasr_session_get_speaker_name');
    final n = nFn(_handle);
    final out = <String>[];
    for (var i = 0; i < n; i++) {
      final ptr = getFn(_handle, i);
      if (ptr != nullptr) out.add(ptr.toDartString());
    }
    return out;
  }

  /// Select a speaker by integer index for multi-speaker TTS backends
  /// (melotts, piper, fastpitch). Index is 0-based; valid range is
  /// `[0, nSpeakers - 1]`.
  ///
  /// For melotts: 0=EN-US, 1=EN-BR, etc. For name-based backends like
  /// orpheus, use [setSpeakerName] instead.
  void setSpeakerID(int id) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_speaker_id')) {
      throw UnsupportedError('setSpeakerID API not available in this libcrispasr build');
    }
    final fn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>, Int32),
        int Function(Pointer<Void>, int)>('crispasr_session_set_speaker_id');
    final rc = fn(_handle, id);
    if (rc == -2) {
      throw RangeError('speaker id $id out of range for backend $_backend');
    }
    if (rc == -3) {
      throw StateError('backend $_backend has no integer-speaker contract; use setSpeakerName() instead');
    }
    if (rc != 0) throw Exception('setSpeakerID failed (rc=$rc) for backend $_backend');
  }

  /// Number of preset speakers for the active backend.
  /// Works for both name-based (orpheus, qwen3-tts) and index-based
  /// (melotts, piper, fastpitch) backends.
  int get nSpeakers {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_n_speakers')) return 0;
    final fn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>),
        int Function(Pointer<Void>)>('crispasr_session_n_speakers');
    return fn(_handle);
  }

  /// Set the natural-language voice description for instruct-tuned TTS
  /// backends (qwen3-tts VoiceDesign today).
  ///
  /// VoiceDesign generates speech in a voice **described by a
  /// natural-language instruction** — no reference WAV, no preset
  /// speaker. The instruct text is wrapped as
  /// `<|im_start|>user\n{instruct}<|im_end|>\n` and prepended to the
  /// talker prefill; the codec bridge omits the speaker frame.
  ///
  /// Required for qwen3-tts VoiceDesign before [synthesize]. Detect
  /// VoiceDesign via [isVoiceDesign].
  void setInstruct(String instruct) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_instruct')) {
      throw UnsupportedError('setInstruct API not available in this libcrispasr build');
    }
    final fn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>, Pointer<Utf8>),
        int Function(Pointer<Void>, Pointer<Utf8>)>('crispasr_session_set_instruct');
    final p = instruct.toNativeUtf8();
    try {
      final rc = fn(_handle, p);
      if (rc == -3) {
        throw StateError(
            'backend $_backend is not a VoiceDesign variant; setInstruct only applies to qwen3-tts VoiceDesign');
      }
      if (rc != 0) throw Exception('setInstruct failed (rc=$rc) for backend $_backend');
    } finally {
      calloc.free(p);
    }
  }

  /// Whether the loaded model is a qwen3-tts CustomVoice variant
  /// (use [setSpeakerName] for it).
  bool isCustomVoice() {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_is_custom_voice')) return false;
    final fn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>),
        int Function(Pointer<Void>)>('crispasr_session_is_custom_voice');
    return fn(_handle) != 0;
  }

  /// Whether the loaded model is a qwen3-tts VoiceDesign variant
  /// (use [setInstruct] for it).
  bool isVoiceDesign() {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_is_voice_design')) return false;
    final fn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>),
        int Function(Pointer<Void>)>('crispasr_session_is_voice_design');
    return fn(_handle) != 0;
  }

  /// Synthesise [text] to 24 kHz mono float32 PCM.
  ///
  /// Requires a TTS-capable backend (vibevoice, qwen3-tts, kokoro, orpheus).
  /// For qwen3-tts call [setCodecPath] and one of: [setVoice] (Base),
  /// [setSpeakerName] (CustomVoice), [setInstruct] (VoiceDesign). Branch
  /// via [isVoiceDesign] / [isCustomVoice]. For orpheus call
  /// [setCodecPath] (SNAC GGUF) and [setSpeakerName].
  Float32List synthesize(String text) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_synthesize')) {
      throw UnsupportedError('TTS synthesize API not available in this libcrispasr build');
    }
    final synFn = _lib.lookupFunction<
        Pointer<Float> Function(Pointer<Void>, Pointer<Utf8>, Pointer<Int32>),
        Pointer<Float> Function(Pointer<Void>, Pointer<Utf8>, Pointer<Int32>)>(
      'crispasr_session_synthesize',
    );
    final freeFn = _lib.lookupFunction<
        Void Function(Pointer<Float>),
        void Function(Pointer<Float>)>('crispasr_pcm_free');
    final textPtr = text.toNativeUtf8();
    final nPtr = calloc<Int32>();
    try {
      final pcmPtr = synFn(_handle, textPtr, nPtr);
      final n = nPtr.value;
      if (pcmPtr == nullptr || n <= 0) {
        // Query the C-side last_synth_error for a specific reason.
        String reason = '';
        if (_lib.providesSymbol('crispasr_session_last_synth_error')) {
          final errFn = _lib.lookupFunction<
              Pointer<Utf8> Function(Pointer<Void>),
              Pointer<Utf8> Function(Pointer<Void>)>(
            'crispasr_session_last_synth_error',
          );
          final errPtr = errFn(_handle);
          if (errPtr != nullptr) {
            final msg = errPtr.toDartString();
            if (msg.isNotEmpty) reason = msg;
          }
        }
        throw Exception(reason.isNotEmpty
            ? reason
            : 'synthesize returned no audio for backend $_backend');
      }
      try {
        // `asTypedList(n)` views the native buffer as a Float32List
        // without copying — then `Float32List.fromList(...)` copies
        // into Dart-owned memory before the native buffer is freed.
        //
        // The previous `List.generate(n, (i) => pcmPtr[i])` pattern
        // tripped on kokoro (and possibly other TTS backends): the
        // resulting buffer was 100% NaN despite the same C-side
        // `kokoro_synthesize` producing valid audio when invoked
        // through the CLI. Most likely an FFI optimisation interacting
        // badly with element-by-element reads on the malloc'd buffer;
        // the bulk view-then-copy path doesn't hit it.
        final view = pcmPtr.asTypedList(n);
        return Float32List.fromList(view);
      } finally {
        freeFn(pcmPtr);
      }
    } finally {
      calloc.free(textPtr);
      calloc.free(nPtr);
    }
  }

  /// Speech-to-Speech: audio in → audio out via a single model pass.
  ///
  /// Supported on backends with S2S capability (lfm2-audio, mini-omni2).
  /// Returns a record of (pcm, transcript) where `pcm` is the output
  /// audio and `transcript` is the intermediate text (may be empty if
  /// the backend doesn't produce one).
  ///
  /// Throws [UnsupportedError] when the loaded dylib predates the S2S
  /// API, [StateError] when the session is closed, and [Exception]
  /// when the backend doesn't support S2S.
  ({Float32List pcm, String transcript}) speechToSpeech(Float32List inputPcm) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_speech_to_speech')) {
      throw UnsupportedError(
          'speechToSpeech API not available in this libcrispasr build');
    }
    final s2sFn = _lib.lookupFunction<
        Pointer<Float> Function(Pointer<Void>, Pointer<Float>, Int32,
            Pointer<Pointer<Utf8>>, Pointer<Int32>),
        Pointer<Float> Function(Pointer<Void>, Pointer<Float>, int,
            Pointer<Pointer<Utf8>>, Pointer<Int32>)>(
      'crispasr_session_speech_to_speech',
    );
    final freeFn = _lib.lookupFunction<Void Function(Pointer<Float>),
        void Function(Pointer<Float>)>('crispasr_pcm_free');
    // Allocate native buffers for input PCM and output pointers.
    final inPtr = calloc<Float>(inputPcm.length);
    final nOutPtr = calloc<Int32>();
    final textOutPtr = calloc<Pointer<Utf8>>();
    try {
      // Copy input PCM to native memory.
      for (var i = 0; i < inputPcm.length; i++) {
        inPtr[i] = inputPcm[i];
      }
      final pcmOut =
          s2sFn(_handle, inPtr, inputPcm.length, textOutPtr, nOutPtr);
      final n = nOutPtr.value;
      if (pcmOut == nullptr || n <= 0) {
        String reason = '';
        if (_lib.providesSymbol('crispasr_session_last_synth_error')) {
          final errFn = _lib.lookupFunction<
              Pointer<Utf8> Function(Pointer<Void>),
              Pointer<Utf8> Function(Pointer<Void>)>(
            'crispasr_session_last_synth_error',
          );
          final errPtr = errFn(_handle);
          if (errPtr != nullptr) {
            final msg = errPtr.toDartString();
            if (msg.isNotEmpty) reason = msg;
          }
        }
        throw Exception(reason.isNotEmpty
            ? reason
            : 'speechToSpeech returned no audio for backend $_backend');
      }
      String transcript = '';
      final txtPtr = textOutPtr.value;
      if (txtPtr != nullptr) {
        transcript = txtPtr.toDartString();
        calloc.free(txtPtr);
      }
      try {
        final view = pcmOut.asTypedList(n);
        return (pcm: Float32List.fromList(view), transcript: transcript);
      } finally {
        freeFn(pcmOut);
      }
    } finally {
      calloc.free(inPtr);
      calloc.free(nOutPtr);
      calloc.free(textOutPtr);
    }
  }

  /// Set hotwords for contextual biasing.
  ///
  /// [hotwords] is a comma-separated list of words/phrases. For CTC/TDT
  /// backends (parakeet), this configures the Aho-Corasick trie with the
  /// given [boost] factor. For LLM backends, the hotwords are prepended
  /// to the ask prompt on the next transcribe call.
  ///
  /// Pass an empty string to clear. Gracefully degrades on older dylibs
  /// that don't have the symbol.
  void setHotwords(String hotwords, {double boost = 1.5}) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_set_hotwords')) {
      // Graceful degradation: older dylibs don't have the symbol.
      // Hotwords are still delivered via the ask-prompt merge path
      // in the Dart layer for LLM backends.
      return;
    }
    final fn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>, Pointer<Utf8>, Float),
        int Function(
            Pointer<Void>, Pointer<Utf8>, double)>('crispasr_session_set_hotwords');
    final p = hotwords.toNativeUtf8();
    try {
      final rc = fn(_handle, p, boost);
      if (rc != 0) throw Exception('setHotwords failed (rc=$rc)');
    } finally {
      calloc.free(p);
    }
  }

  /// Open a streaming decode session against this session's backend.
  ///
  /// Backed by `crispasr_session_stream_open` on the C side, which
  /// dispatches the rolling-window protocol to whichever backend the
  /// session loaded (whisper, kyutai-stt, moonshine-streaming, …).
  /// Same shape as [CrispASR.openStream] (Whisper-only) but works for
  /// every backend that publishes a streaming entry point.
  ///
  /// Throws [UnsupportedError] when the loaded dylib predates
  /// `crispasr_session_stream_open`, [StateError] when the session
  /// is already closed, and [Exception] when the backend has no
  /// streaming arm (the C side returns null in that case).
  StreamingSession openStream({
    int stepMs = 3000,
    int lengthMs = 10000,
    int keepMs = 200,
    int nThreads = 4,
    String? language,
    bool translate = false,
  }) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (!_lib.providesSymbol('crispasr_session_stream_open')) {
      throw UnsupportedError(
          'Session-level streaming not in this libcrispasr — needs CrispASR 0.6+.');
    }
    final fn = _lib.lookupFunction<
        Pointer<Void> Function(
            Pointer<Void>, Int32, Int32, Int32, Int32, Pointer<Utf8>, Int32),
        Pointer<Void> Function(Pointer<Void>, int, int, int, int,
            Pointer<Utf8>, int)>('crispasr_session_stream_open');
    final langPtr = (language == null || language.isEmpty || language == 'auto')
        ? nullptr
        : language.toNativeUtf8();
    final handle = fn(_handle, nThreads, stepMs, lengthMs, keepMs,
        langPtr.cast<Utf8>(), translate ? 1 : 0);
    if (langPtr != nullptr) calloc.free(langPtr);
    if (handle == nullptr) {
      throw Exception(
          'crispasr_session_stream_open returned null — backend "$_backend" '
          'has no streaming entry point.');
    }
    final feedFn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>, Pointer<Float>, Int32),
        int Function(Pointer<Void>, Pointer<Float>, int)>('crispasr_stream_feed');
    final flushFn = _lib.providesSymbol('crispasr_stream_flush')
        ? _lib.lookupFunction<Int32 Function(Pointer<Void>),
            int Function(Pointer<Void>)>('crispasr_stream_flush')
        : null;
    final getTextFn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>, Pointer<Utf8>, Int32, Pointer<Double>,
            Pointer<Double>, Pointer<Int64>),
        int Function(Pointer<Void>, Pointer<Utf8>, int, Pointer<Double>,
            Pointer<Double>, Pointer<Int64>)>('crispasr_stream_get_text');
    final closeFn = _lib.lookupFunction<Void Function(Pointer<Void>),
        void Function(Pointer<Void>)>('crispasr_stream_close');
    final setLiveDecodeFn = _lib.providesSymbol('crispasr_stream_set_live_decode')
        ? _lib.lookupFunction<_StreamSetLiveDecodeNative, _StreamSetLiveDecode>(
            'crispasr_stream_set_live_decode')
        : null;

    return StreamingSession._(
      handle: handle,
      feed: feedFn,
      flush: flushFn,
      getText: getTextFn,
      close: closeFn,
      setLiveDecode: setLiveDecodeFn,
    );
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
    final lib = DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
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

// =========================================================================
// Truecasing — standalone text post-processing (C-ABI 0.5.3+)
// =========================================================================

/// Standalone truecaser model. Restores letter casing on lowercased text.
/// Particularly useful for German (truecaser-de-lstm achieves 97.9% F1).
///
/// ```dart
/// final tc = TruecaseModel.open('truecaser-de-lstm.gguf');
/// final text = tc.process('hallo mein name ist max');
/// print(text); // "Hallo mein Name ist Max"
/// tc.close();
/// ```
class TruecaseModel {
  final DynamicLibrary _lib;
  Pointer<Void> _handle;
  bool _closed = false;

  TruecaseModel._(this._lib, this._handle);

  /// Load a truecaser GGUF model (statistical, BiLSTM, or CRF).
  static TruecaseModel open(String modelPath, {String? libPath}) {
    final lib = DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
    final initFn = lib.lookupFunction<
        Pointer<Void> Function(Pointer<Utf8>),
        Pointer<Void> Function(Pointer<Utf8>)>('crispasr_truecase_init');
    final pathPtr = modelPath.toNativeUtf8();
    final handle = initFn(pathPtr);
    calloc.free(pathPtr);
    if (handle == nullptr) {
      throw Exception('Failed to load truecase model: $modelPath');
    }
    return TruecaseModel._(lib, handle);
  }

  /// Apply truecasing to text.
  String process(String text) {
    if (_closed) throw StateError('TruecaseModel is closed');
    final processFn = _lib.lookupFunction<
        Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>),
        Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>)>('crispasr_truecase_process');
    final freeFn = _lib.lookupFunction<
        Void Function(Pointer<Utf8>),
        void Function(Pointer<Utf8>)>('crispasr_truecase_free_text');
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
        void Function(Pointer<Void>)>('crispasr_truecase_free');
    freeFn(_handle);
    _handle = nullptr;
  }
}

// =========================================================================
// PCS — Punctuation + Capitalization + Sentence-boundary (C-ABI 0.5.3+)
// =========================================================================

/// Standalone PCS model. Applies punctuation, truecasing, and sentence
/// boundary detection in a single pass. Supports 47 languages.
///
/// ```dart
/// final pcs = PcsModel.open('pcs-47-q8_0.gguf');
/// final text = pcs.process('hello how are you doing today');
/// print(text); // "Hello, how are you doing today?"
/// pcs.close();
/// ```
class PcsModel {
  final DynamicLibrary _lib;
  Pointer<Void> _handle;
  bool _closed = false;

  PcsModel._(this._lib, this._handle);

  /// Load a PCS GGUF model.
  static PcsModel open(String modelPath, {String? libPath}) {
    final lib = DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
    final initFn = lib.lookupFunction<
        Pointer<Void> Function(Pointer<Utf8>),
        Pointer<Void> Function(Pointer<Utf8>)>('crispasr_pcs_init');
    final pathPtr = modelPath.toNativeUtf8();
    final handle = initFn(pathPtr);
    calloc.free(pathPtr);
    if (handle == nullptr) {
      throw Exception('Failed to load PCS model: $modelPath');
    }
    return PcsModel._(lib, handle);
  }

  /// Apply punctuation, truecasing, and sentence boundaries to text.
  String process(String text) {
    if (_closed) throw StateError('PcsModel is closed');
    final processFn = _lib.lookupFunction<
        Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>),
        Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>)>('crispasr_pcs_process');
    final freeFn = _lib.lookupFunction<
        Void Function(Pointer<Utf8>),
        void Function(Pointer<Utf8>)>('crispasr_pcs_free_text');
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
        void Function(Pointer<Void>)>('crispasr_pcs_free');
    freeFn(_handle);
    _handle = nullptr;
  }
}

// =========================================================================
// TitaNet speaker verification
// =========================================================================

/// TitaNet-Large speaker embedding extractor (192-d, L2-normalized).
class CrispasrTitaNet {
  late final DynamicLibrary _lib;
  Pointer<Void> _handle = nullptr;

  CrispasrTitaNet(DynamicLibrary lib, String modelPath, {int nThreads = 4})
      : _lib = lib {
    final initFn = lib.lookupFunction<
        Pointer<Void> Function(Pointer<Utf8>, Int32),
        Pointer<Void> Function(Pointer<Utf8>, int)>('crispasr_titanet_init');
    final mp = modelPath.toNativeUtf8();
    _handle = initFn(mp, nThreads);
    malloc.free(mp);
    if (_handle == nullptr) {
      throw Exception('Failed to load TitaNet model: $modelPath');
    }
  }

  /// Extract 192-d speaker embedding from 16 kHz mono float32 PCM.
  Float32List embed(Float32List pcm16k) {
    final embedFn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>, Pointer<Float>, Int32, Pointer<Float>),
        int Function(Pointer<Void>, Pointer<Float>, int, Pointer<Float>)>(
        'crispasr_titanet_embed');
    final pcmPtr = malloc<Float>(pcm16k.length);
    pcmPtr.asTypedList(pcm16k.length).setAll(0, pcm16k);
    final outPtr = malloc<Float>(192);
    final dim = embedFn(_handle, pcmPtr, pcm16k.length, outPtr);
    malloc.free(pcmPtr);
    if (dim <= 0) {
      malloc.free(outPtr);
      throw Exception('TitaNet embedding failed');
    }
    final result = Float32List.fromList(outPtr.asTypedList(dim));
    malloc.free(outPtr);
    return result;
  }

  /// Cosine similarity between two embeddings.
  static double cosineSim(DynamicLibrary lib, Float32List a, Float32List b) {
    final fn = lib.lookupFunction<
        Float Function(Pointer<Float>, Pointer<Float>, Int32),
        double Function(Pointer<Float>, Pointer<Float>, int)>(
        'crispasr_titanet_cosine_sim');
    final dim = a.length < b.length ? a.length : b.length;
    final aPtr = malloc<Float>(dim);
    aPtr.asTypedList(dim).setAll(0, a.sublist(0, dim));
    final bPtr = malloc<Float>(dim);
    bPtr.asTypedList(dim).setAll(0, b.sublist(0, dim));
    final sim = fn(aPtr, bPtr, dim);
    malloc.free(aPtr);
    malloc.free(bPtr);
    return sim;
  }

  void close() {
    if (_handle == nullptr) return;
    final freeFn = _lib.lookupFunction<
        Void Function(Pointer<Void>),
        void Function(Pointer<Void>)>('crispasr_titanet_free');
    freeFn(_handle);
    _handle = nullptr;
  }
}

/// File-based speaker profile database.
class CrispasrSpeakerDB {
  late final DynamicLibrary _lib;
  Pointer<Void> _handle = nullptr;
  final String dirPath;

  CrispasrSpeakerDB(DynamicLibrary lib, this.dirPath) : _lib = lib {
    final loadFn = lib.lookupFunction<
        Pointer<Void> Function(Pointer<Utf8>),
        Pointer<Void> Function(Pointer<Utf8>)>('crispasr_speaker_db_load');
    final dp = dirPath.toNativeUtf8();
    _handle = loadFn(dp);
    malloc.free(dp);
  }

  int get count {
    final countFn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>),
        int Function(Pointer<Void>)>('crispasr_speaker_db_count');
    return countFn(_handle);
  }

  /// Match embedding against DB. Returns (name, score) or (null, score).
  (String?, double) match(Float32List embedding, {double threshold = 0.7}) {
    final matchFn = _lib.lookupFunction<
        Float Function(Pointer<Void>, Pointer<Float>, Int32, Float,
            Pointer<Utf8>, Int32),
        double Function(Pointer<Void>, Pointer<Float>, int, double,
            Pointer<Utf8>, int)>('crispasr_speaker_db_match');
    final embPtr = malloc<Float>(embedding.length);
    embPtr.asTypedList(embedding.length).setAll(0, embedding);
    final nameBuf = malloc<Uint8>(256);
    final score = matchFn(
        _handle, embPtr, embedding.length, threshold, nameBuf.cast(), 256);
    malloc.free(embPtr);
    String? name;
    if (score >= threshold) {
      name = nameBuf.cast<Utf8>().toDartString();
    }
    malloc.free(nameBuf);
    return (name, score);
  }

  /// Enroll a speaker with the given name and embedding.
  bool enroll(String name, Float32List embedding) {
    final enrollFn = _lib.lookupFunction<
        Int32 Function(Pointer<Utf8>, Pointer<Utf8>, Pointer<Float>, Int32),
        int Function(Pointer<Utf8>, Pointer<Utf8>, Pointer<Float>, int)>(
        'crispasr_speaker_db_enroll');
    final dp = dirPath.toNativeUtf8();
    final np = name.toNativeUtf8();
    final embPtr = malloc<Float>(embedding.length);
    embPtr.asTypedList(embedding.length).setAll(0, embedding);
    final rc = enrollFn(dp, np, embPtr, embedding.length);
    malloc.free(dp);
    malloc.free(np);
    malloc.free(embPtr);
    return rc == 0;
  }

  void close() {
    if (_handle == nullptr) return;
    final freeFn = _lib.lookupFunction<
        Void Function(Pointer<Void>),
        void Function(Pointer<Void>)>('crispasr_speaker_db_free');
    freeFn(_handle);
    _handle = nullptr;
  }
}

// =====================================================================
// Diarization pipeline primitives (issue #107 P6).
// Pluggable speaker embedder + agglomerative cosine clustering +
// pyannote-seg cache — the same building blocks the CLI's
// --diarize-embedder path uses.
// =====================================================================

/// Pluggable speaker-embedding model. Dispatch:
///   - `auto` / `titanet`           -> TitaNet-Large (192-d)
///   - `indextts` / `indextts-bigvgan` / `ecapa`
///                                  -> IndexTTS-BigVGAN ECAPA-TDNN (512-d)
///   - any `.gguf` path             -> TitaNet (default; IndexTTS if the
///                                     path contains "indextts")
class CrispasrSpeakerEmbedder {
  late final DynamicLibrary _lib;
  Pointer<Void> _handle = nullptr;

  CrispasrSpeakerEmbedder(DynamicLibrary lib, String modelSpec,
      {int nThreads = 4, String cacheDir = ''})
      : _lib = lib {
    final makeFn = lib.lookupFunction<
        Pointer<Void> Function(Pointer<Utf8>, Int32, Pointer<Utf8>),
        Pointer<Void> Function(Pointer<Utf8>, int, Pointer<Utf8>)>(
        'crispasr_speaker_embedder_make_abi');
    final specPtr = modelSpec.toNativeUtf8();
    final cachePtr = cacheDir.toNativeUtf8();
    _handle = makeFn(specPtr, nThreads, cachePtr);
    malloc.free(specPtr);
    malloc.free(cachePtr);
    if (_handle == nullptr) {
      throw Exception('Failed to build speaker embedder: $modelSpec');
    }
  }

  int get dim {
    final dimFn = _lib.lookupFunction<Int32 Function(Pointer<Void>),
        int Function(Pointer<Void>)>('crispasr_speaker_embedder_dim_abi');
    return dimFn(_handle);
  }

  String get name {
    final nameFn = _lib.lookupFunction<
        Pointer<Utf8> Function(Pointer<Void>),
        Pointer<Utf8> Function(Pointer<Void>)>(
        'crispasr_speaker_embedder_name_abi');
    final p = nameFn(_handle);
    return p == nullptr ? '' : p.toDartString();
  }

  /// Extract one embedding from mono 16 kHz float32 PCM. Returns null
  /// when the underlying model rejected the input.
  Float32List? embed(Float32List pcm16k) {
    final embedFn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>, Pointer<Float>, Int32, Pointer<Float>),
        int Function(Pointer<Void>, Pointer<Float>, int, Pointer<Float>)>(
        'crispasr_speaker_embedder_embed_abi');
    final d = dim;
    if (d <= 0) return null;
    final pcmPtr = malloc<Float>(pcm16k.length);
    pcmPtr.asTypedList(pcm16k.length).setAll(0, pcm16k);
    final outPtr = malloc<Float>(d);
    final ok = embedFn(_handle, pcmPtr, pcm16k.length, outPtr);
    malloc.free(pcmPtr);
    if (ok == 0) {
      malloc.free(outPtr);
      return null;
    }
    final result = Float32List.fromList(outPtr.asTypedList(d));
    malloc.free(outPtr);
    return result;
  }

  void close() {
    if (_handle == nullptr) return;
    final freeFn = _lib.lookupFunction<
        Void Function(Pointer<Void>),
        void Function(Pointer<Void>)>('crispasr_speaker_embedder_free_abi');
    freeFn(_handle);
    _handle = nullptr;
  }
}

/// Agglomerative single-linkage cosine clustering on (ideally
/// L2-normalized) speaker embeddings. `embeddings` is a flat row-major
/// `n × dim` buffer; returns one cluster ID per input in `[0, k)`.
List<int> crispasrAgglomerativeCluster(
  DynamicLibrary lib,
  Float32List embeddings, {
  required int n,
  required int dim,
  double mergeThreshold = 0.5,
  int maxSpeakers = 32,
}) {
  if (n <= 0 || dim <= 0 || embeddings.length < n * dim) {
    return List<int>.filled(n.clamp(0, 1 << 30), -1);
  }
  final fn = lib.lookupFunction<
      Int32 Function(Pointer<Float>, Int32, Int32, Float, Int32, Pointer<Int32>),
      int Function(Pointer<Float>, int, int, double, int, Pointer<Int32>)>(
      'crispasr_speaker_cluster_abi');
  final embPtr = malloc<Float>(embeddings.length);
  embPtr.asTypedList(embeddings.length).setAll(0, embeddings);
  final outPtr = malloc<Int32>(n);
  final rc = fn(embPtr, n, dim, mergeThreshold, maxSpeakers, outPtr);
  final labels = List<int>.from(outPtr.asTypedList(n));
  malloc.free(embPtr);
  malloc.free(outPtr);
  if (rc < 0) {
    throw Exception('crispasr_speaker_cluster_abi: invalid arguments');
  }
  return labels;
}

/// Pre-computed pyannote-seg posteriors over a full audio buffer.
class CrispasrPyannoteCache {
  late final DynamicLibrary _lib;
  Pointer<Void> _handle = nullptr;

  CrispasrPyannoteCache(DynamicLibrary lib, Float32List pcm16k, String modelPath,
      {int nThreads = 4})
      : _lib = lib {
    final computeFn = lib.lookupFunction<
        Pointer<Void> Function(Pointer<Float>, Int32, Pointer<Utf8>, Int32),
        Pointer<Void> Function(Pointer<Float>, int, Pointer<Utf8>, int)>(
        'crispasr_pyannote_cache_compute_abi');
    final pcmPtr = malloc<Float>(pcm16k.length);
    pcmPtr.asTypedList(pcm16k.length).setAll(0, pcm16k);
    final mp = modelPath.toNativeUtf8();
    _handle = computeFn(pcmPtr, pcm16k.length, mp, nThreads);
    malloc.free(pcmPtr);
    malloc.free(mp);
    if (_handle == nullptr) {
      throw Exception('Failed to compute pyannote cache from $modelPath');
    }
  }

  /// Score `segs` against the cached posteriors. Each segment's
  /// `speaker` is set to 0/1/2 or -1 for silence.
  void apply(List<DiarizeSegment> segs, {double sliceT0 = 0.0}) {
    if (segs.isEmpty) return;
    // ABI segment layout: t0_cs i64, t1_cs i64, speaker i32, _pad i32.
    final segPtr = malloc<Uint8>(segs.length * 24);
    final segBytes = segPtr.asTypedList(segs.length * 24);
    final bd = segBytes.buffer.asByteData();
    for (var i = 0; i < segs.length; i++) {
      bd.setInt64(i * 24, (segs[i].t0 * 100).round(), Endian.host);
      bd.setInt64(i * 24 + 8, (segs[i].t1 * 100).round(), Endian.host);
      bd.setInt32(i * 24 + 16, segs[i].speaker, Endian.host);
      bd.setInt32(i * 24 + 20, 0, Endian.host);
    }
    final applyFn = _lib.lookupFunction<
        Int32 Function(Pointer<Void>, Int64, Pointer<Uint8>, Int32),
        int Function(Pointer<Void>, int, Pointer<Uint8>, int)>(
        'crispasr_pyannote_cache_apply_abi');
    final rc = applyFn(
        _handle, (sliceT0 * 100).round(), segPtr, segs.length);
    if (rc != 0) {
      malloc.free(segPtr);
      throw Exception('crispasr_pyannote_cache_apply_abi returned $rc');
    }
    for (var i = 0; i < segs.length; i++) {
      segs[i] = DiarizeSegment(
        t0: segs[i].t0,
        t1: segs[i].t1,
        speaker: bd.getInt32(i * 24 + 16, Endian.host),
      );
    }
    malloc.free(segPtr);
  }

  void close() {
    if (_handle == nullptr) return;
    final freeFn = _lib.lookupFunction<
        Void Function(Pointer<Void>),
        void Function(Pointer<Void>)>('crispasr_pyannote_cache_free_abi');
    freeFn(_handle);
    _handle = nullptr;
  }
}

// =====================================================================
// Direct Parakeet API (bypasses unified session)
// =====================================================================

/// Parakeet ASR result with word- and token-level timestamps.
class ParakeetResult {
  final DynamicLibrary _lib;
  Pointer<Void> _handle;

  ParakeetResult._(this._lib, this._handle);

  String get text {
    final fn = _lib.lookupFunction<
        Pointer<Utf8> Function(Pointer<Void>),
        Pointer<Utf8> Function(Pointer<Void>)>('crispasr_parakeet_result_text');
    final p = fn(_handle);
    return p == nullptr ? '' : p.toDartString();
  }

  int get nWords {
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>),
        int Function(Pointer<Void>)>('crispasr_parakeet_result_n_words');
    return fn(_handle);
  }

  String wordText(int i) {
    final fn = _lib.lookupFunction<
        Pointer<Utf8> Function(Pointer<Void>, Int32),
        Pointer<Utf8> Function(Pointer<Void>, int)>('crispasr_parakeet_result_word_text');
    final p = fn(_handle, i);
    return p == nullptr ? '' : p.toDartString();
  }

  int wordT0(int i) {
    final fn = _lib.lookupFunction<Int64 Function(Pointer<Void>, Int32),
        int Function(Pointer<Void>, int)>('crispasr_parakeet_result_word_t0');
    return fn(_handle, i);
  }

  int wordT1(int i) {
    final fn = _lib.lookupFunction<Int64 Function(Pointer<Void>, Int32),
        int Function(Pointer<Void>, int)>('crispasr_parakeet_result_word_t1');
    return fn(_handle, i);
  }

  int get nTokens {
    final fn = _lib.lookupFunction<Int32 Function(Pointer<Void>),
        int Function(Pointer<Void>)>('crispasr_parakeet_result_n_tokens');
    return fn(_handle);
  }

  String tokenText(int i) {
    final fn = _lib.lookupFunction<
        Pointer<Utf8> Function(Pointer<Void>, Int32),
        Pointer<Utf8> Function(Pointer<Void>, int)>('crispasr_parakeet_result_token_text');
    final p = fn(_handle, i);
    return p == nullptr ? '' : p.toDartString();
  }

  int tokenT0(int i) {
    final fn = _lib.lookupFunction<Int64 Function(Pointer<Void>, Int32),
        int Function(Pointer<Void>, int)>('crispasr_parakeet_result_token_t0');
    return fn(_handle, i);
  }

  int tokenT1(int i) {
    final fn = _lib.lookupFunction<Int64 Function(Pointer<Void>, Int32),
        int Function(Pointer<Void>, int)>('crispasr_parakeet_result_token_t1');
    return fn(_handle, i);
  }

  double tokenP(int i) {
    final fn = _lib.lookupFunction<Float Function(Pointer<Void>, Int32),
        double Function(Pointer<Void>, int)>('crispasr_parakeet_result_token_p');
    return fn(_handle, i);
  }

  void free() {
    if (_handle == nullptr) return;
    final fn = _lib.lookupFunction<Void Function(Pointer<Void>),
        void Function(Pointer<Void>)>('crispasr_parakeet_result_free');
    fn(_handle);
    _handle = nullptr;
  }
}

/// Direct Parakeet ASR context. For most use cases prefer [CrispasrSession]
/// which auto-dispatches to Parakeet when the GGUF metadata indicates it.
class CrispasrParakeet {
  late final DynamicLibrary _lib;
  Pointer<Void> _handle = nullptr;

  CrispasrParakeet(DynamicLibrary lib, String modelPath,
      {int nThreads = 4, bool useFlash = true})
      : _lib = lib {
    final initFn = lib.lookupFunction<
        Pointer<Void> Function(Pointer<Utf8>, Int32, Int32),
        Pointer<Void> Function(Pointer<Utf8>, int, int)>('crispasr_parakeet_init');
    final mp = modelPath.toNativeUtf8();
    _handle = initFn(mp, nThreads, useFlash ? 1 : 0);
    malloc.free(mp);
    if (_handle == nullptr) {
      throw Exception('Failed to load Parakeet model: $modelPath');
    }
  }

  /// Transcribe mono 16 kHz float32 PCM. The returned [ParakeetResult]
  /// must be freed by the caller via [ParakeetResult.free].
  ParakeetResult transcribe(Float32List pcm16k, {String? language}) {
    final pcmPtr = malloc<Float>(pcm16k.length);
    pcmPtr.asTypedList(pcm16k.length).setAll(0, pcm16k);
    final langPtr = language != null ? language.toNativeUtf8() : nullptr;
    try {
      final fn = _lib.lookupFunction<
          Pointer<Void> Function(Pointer<Void>, Pointer<Float>, Int32, Pointer<Utf8>),
          Pointer<Void> Function(Pointer<Void>, Pointer<Float>, int, Pointer<Utf8>)>(
          'crispasr_parakeet_transcribe');
      final res = fn(_handle, pcmPtr, pcm16k.length, langPtr.cast<Utf8>());
      if (res == nullptr) {
        throw Exception('crispasr_parakeet_transcribe returned null');
      }
      return ParakeetResult._(_lib, res);
    } finally {
      malloc.free(pcmPtr);
      if (langPtr != nullptr) malloc.free(langPtr);
    }
  }

  void close() {
    if (_handle == nullptr) return;
    final fn = _lib.lookupFunction<Void Function(Pointer<Void>),
        void Function(Pointer<Void>)>('crispasr_parakeet_free');
    fn(_handle);
    _handle = nullptr;
  }
}

// =====================================================================
// Standalone helpers — full C-ABI parity
// =====================================================================

/// Detect the backend name from a GGUF file's metadata.
/// Returns the backend name (e.g. 'whisper', 'parakeet', 'canary') or
/// null if detection failed.
String? detectBackendFromGguf(String path, {String? libPath}) {
  final lib = DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
  if (!lib.providesSymbol('crispasr_detect_backend_from_gguf')) return null;
  final fn = lib.lookupFunction<
      Int32 Function(Pointer<Utf8>, Pointer<Utf8>, Int32),
      int Function(Pointer<Utf8>, Pointer<Utf8>, int)>(
      'crispasr_detect_backend_from_gguf');
  final pathPtr = path.toNativeUtf8();
  const cap = 128;
  final outBuf = calloc<Uint8>(cap);
  try {
    final rc = fn(pathPtr, outBuf.cast<Utf8>(), cap);
    if (rc != 0) return null;
    return outBuf.cast<Utf8>().toDartString();
  } finally {
    calloc.free(pathPtr);
    calloc.free(outBuf);
  }
}

/// Chunk-boundary LCS dedup: returns the number of leading tokens
/// of [currTokens] to drop to remove overlap with [prevTailTokens].
int lcsDedup(DynamicLibrary lib, List<int> prevTailTokens,
    List<int> currTokens, {int minLcsLength = 1}) {
  final fn = lib.lookupFunction<
      Int32 Function(Pointer<Int32>, Int32, Pointer<Int32>, Int32, Int32),
      int Function(Pointer<Int32>, int, Pointer<Int32>, int, int)>(
      'crispasr_lcs_dedup_prefix_count');
  final prevPtr = malloc<Int32>(prevTailTokens.length);
  for (var i = 0; i < prevTailTokens.length; i++) {
    prevPtr[i] = prevTailTokens[i];
  }
  final currPtr = malloc<Int32>(currTokens.length);
  for (var i = 0; i < currTokens.length; i++) {
    currPtr[i] = currTokens[i];
  }
  final result = fn(prevPtr, prevTailTokens.length, currPtr,
      currTokens.length, minLcsLength);
  malloc.free(prevPtr);
  malloc.free(currPtr);
  return result;
}

/// Whether [lang] is German (Kokoro phoneme selection).
bool kokoroLangIsGerman(String lang, {String? libPath}) {
  final lib = DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
  if (!lib.providesSymbol('crispasr_kokoro_lang_is_german_abi')) return false;
  final fn = lib.lookupFunction<
      Bool Function(Pointer<Utf8>),
      bool Function(Pointer<Utf8>)>('crispasr_kokoro_lang_is_german_abi');
  final p = lang.toNativeUtf8();
  try {
    return fn(p);
  } finally {
    calloc.free(p);
  }
}

/// Whether [lang] has a native Kokoro voice (vs. cross-lingual fallback).
bool kokoroLangHasNativeVoice(String lang, {String? libPath}) {
  final lib = DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
  if (!lib.providesSymbol('crispasr_kokoro_lang_has_native_voice_abi')) return false;
  final fn = lib.lookupFunction<
      Bool Function(Pointer<Utf8>),
      bool Function(Pointer<Utf8>)>('crispasr_kokoro_lang_has_native_voice_abi');
  final p = lang.toNativeUtf8();
  try {
    return fn(p);
  } finally {
    calloc.free(p);
  }
}

/// Resolve the appropriate Kokoro model path for [lang].
/// Returns the resolved path or null on failure.
String? kokoroResolveModelForLang(String modelPath, String lang,
    {String? libPath}) {
  final lib = DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
  if (!lib.providesSymbol('crispasr_kokoro_resolve_model_for_lang_abi')) {
    return null;
  }
  final fn = lib.lookupFunction<
      Int32 Function(Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Int32),
      int Function(Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, int)>(
      'crispasr_kokoro_resolve_model_for_lang_abi');
  final mp = modelPath.toNativeUtf8();
  final lp = lang.toNativeUtf8();
  const cap = 512;
  final outBuf = calloc<Uint8>(cap);
  try {
    final rc = fn(mp, lp, outBuf.cast<Utf8>(), cap);
    if (rc != 0) return null;
    return outBuf.cast<Utf8>().toDartString();
  } finally {
    calloc.free(mp);
    calloc.free(lp);
    calloc.free(outBuf);
  }
}

/// Resolve the Kokoro fallback voice for [lang].
/// Returns `(voicePath, voiceName)` or null on failure.
({String voicePath, String voiceName})? kokoroResolveFallbackVoice(
    String modelPath, String lang,
    {String? libPath}) {
  final lib = DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
  if (!lib.providesSymbol('crispasr_kokoro_resolve_fallback_voice_abi')) {
    return null;
  }
  final fn = lib.lookupFunction<
      Int32 Function(
          Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Int32, Pointer<Utf8>, Int32),
      int Function(
          Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, int, Pointer<Utf8>, int)>(
      'crispasr_kokoro_resolve_fallback_voice_abi');
  final mp = modelPath.toNativeUtf8();
  final lp = lang.toNativeUtf8();
  const cap = 512;
  final pathBuf = calloc<Uint8>(cap);
  final nameBuf = calloc<Uint8>(cap);
  try {
    final rc = fn(mp, lp, pathBuf.cast<Utf8>(), cap, nameBuf.cast<Utf8>(), cap);
    if (rc != 0) return null;
    return (
      voicePath: pathBuf.cast<Utf8>().toDartString(),
      voiceName: nameBuf.cast<Utf8>().toDartString(),
    );
  } finally {
    calloc.free(mp);
    calloc.free(lp);
    calloc.free(pathBuf);
    calloc.free(nameBuf);
  }
}

// =========================================================================
// Audio watermark — CrispASR native spread-spectrum + optional AudioSeal
// =========================================================================

/// Native audio watermarking via CrispASR's `crispasr_watermark_*` C API.
///
/// Provides two tiers:
///   1. **Built-in spread-spectrum** (always available when the symbols are
///      exported): frequency-domain pattern that survives re-encoding and
///      moderate compression.
///   2. **AudioSeal neural** (optional): Meta's SEANet-based watermark,
///      activated by calling [loadModel] with an AudioSeal GGUF.
///
/// When an AudioSeal model is loaded, [embed] and [detect] dispatch to it
/// automatically; otherwise they use the spread-spectrum fallback.
///
/// All operations work on float32 mono PCM. AudioSeal expects 16 kHz;
/// spread-spectrum works at any sample rate.
class CrispasrWatermark {
  CrispasrWatermark._();

  /// Check whether the loaded dylib exports the watermark symbols.
  /// Returns false on older CrispASR builds that predate the watermark API.
  static bool isAvailable({DynamicLibrary? lib}) {
    lib ??= DynamicLibrary.open(CrispASR.defaultLibName());
    return lib.providesSymbol('crispasr_watermark_embed') &&
        lib.providesSymbol('crispasr_watermark_detect');
  }

  /// Load an AudioSeal GGUF model for neural watermarking. Call once at
  /// startup. Returns true on success. On failure the API falls back to
  /// the built-in spread-spectrum watermark.
  static bool loadModel(String ggufPath, {DynamicLibrary? lib}) {
    lib ??= DynamicLibrary.open(CrispASR.defaultLibName());
    if (!lib.providesSymbol('crispasr_watermark_load_model')) return false;
    final fn = lib.lookupFunction<
        Int32 Function(Pointer<Utf8>),
        int Function(Pointer<Utf8>)>('crispasr_watermark_load_model');
    final p = ggufPath.toNativeUtf8();
    final rc = fn(p);
    malloc.free(p);
    return rc == 0;
  }

  /// Embed a watermark into float32 mono PCM (in-place).
  ///
  /// [alpha] controls spread-spectrum strength (0.005 default); ignored
  /// when AudioSeal is loaded.
  ///
  /// Returns a new [Float32List] with the watermark applied.
  static Float32List embed(Float32List pcm, {double alpha = 0.005, DynamicLibrary? lib}) {
    lib ??= DynamicLibrary.open(CrispASR.defaultLibName());
    final fn = lib.lookupFunction<
        Void Function(Pointer<Float>, Int32, Float),
        void Function(Pointer<Float>, int, double)>('crispasr_watermark_embed');
    final ptr = malloc<Float>(pcm.length);
    ptr.asTypedList(pcm.length).setAll(0, pcm);
    fn(ptr, pcm.length, alpha);
    final result = Float32List.fromList(ptr.asTypedList(pcm.length));
    malloc.free(ptr);
    return result;
  }

  /// Detect watermark presence in float32 mono PCM.
  ///
  /// Returns a confidence score in [0, 1]:
  ///   - > 0.65 — watermark present (AI-generated)
  ///   - < 0.40 — no watermark detected
  static double detect(Float32List pcm, {DynamicLibrary? lib}) {
    lib ??= DynamicLibrary.open(CrispASR.defaultLibName());
    final fn = lib.lookupFunction<
        Float Function(Pointer<Float>, Int32),
        double Function(Pointer<Float>, int)>('crispasr_watermark_detect');
    final ptr = malloc<Float>(pcm.length);
    ptr.asTypedList(pcm.length).setAll(0, pcm);
    final score = fn(ptr, pcm.length);
    malloc.free(ptr);
    return score;
  }
}

// ---------------------------------------------------------------------------
// C1: Transcription progress polling
// ---------------------------------------------------------------------------

/// Poll the global transcription progress (0–100). Returns -1 when no
/// transcription is active. The C layer updates this via an atomic int
/// from the whisper_progress_callback — no function pointers needed on
/// the Dart side.
int getTranscriptionProgress({String? libPath}) {
  final lib = DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
  if (!lib.providesSymbol('crispasr_get_progress')) return -1;
  final fn = lib.lookupFunction<Int32 Function(), int Function()>(
      'crispasr_get_progress');
  return fn();
}

/// Reset the progress counter to -1 (idle). Call before starting a new
/// transcription to clear stale values from the previous run.
void resetTranscriptionProgress({String? libPath}) {
  final lib = DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
  if (!lib.providesSymbol('crispasr_reset_progress')) return;
  final fn = lib.lookupFunction<Void Function(), void Function()>(
      'crispasr_reset_progress');
  fn();
}

// ---------------------------------------------------------------------------
// C2: Stereo audio decode
// ---------------------------------------------------------------------------

/// Result of [decodeAudioFileStereo].
class DecodedAudioStereo {
  /// Left channel (or the only channel for mono files).
  final Float32List left;

  /// Right channel. Identical to [left] for mono source files.
  final Float32List right;

  /// Sample rate (always 16000 for CrispASR).
  final int sampleRate;

  /// Actual channel count in the source file (1 or 2).
  final int sourceChannels;

  const DecodedAudioStereo({
    required this.left,
    required this.right,
    required this.sampleRate,
    required this.sourceChannels,
  });

  /// True when the source file was stereo.
  bool get isStereo => sourceChannels >= 2;
}

/// Decode an audio file preserving stereo channels. Falls back to
/// [decodeAudioFile] (mono) when the C symbol isn't available.
DecodedAudioStereo decodeAudioFileStereo(String path, {String? libPath}) {
  final lib = DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
  if (!lib.providesSymbol('crispasr_audio_load_stereo')) {
    // Fallback: decode mono and duplicate to both channels.
    final mono = decodeAudioFile(path, libPath: libPath);
    return DecodedAudioStereo(
      left: mono.samples,
      right: mono.samples,
      sampleRate: mono.sampleRate,
      sourceChannels: 1,
    );
  }

  final load = lib.lookupFunction<
      Int32 Function(Pointer<Utf8>, Pointer<Pointer<Float>>,
          Pointer<Pointer<Float>>, Pointer<Int32>, Pointer<Int32>,
          Pointer<Int32>),
      int Function(Pointer<Utf8>, Pointer<Pointer<Float>>,
          Pointer<Pointer<Float>>, Pointer<Int32>, Pointer<Int32>,
          Pointer<Int32>)>('crispasr_audio_load_stereo');
  final free = lib.lookupFunction<Void Function(Pointer<Float>),
      void Function(Pointer<Float>)>('crispasr_audio_free');

  final pathPtr = path.toNativeUtf8();
  final leftOut = calloc<Pointer<Float>>();
  final rightOut = calloc<Pointer<Float>>();
  final nOut = calloc<Int32>();
  final srOut = calloc<Int32>();
  final chOut = calloc<Int32>();

  try {
    final rc = load(pathPtr, leftOut, rightOut, nOut, srOut, chOut);
    if (rc != 0) {
      throw Exception(
          'crispasr_audio_load_stereo failed (code $rc) for $path');
    }
    final n = nOut.value;
    final sr = srOut.value;
    final ch = chOut.value;
    if (n <= 0) {
      throw Exception('Stereo audio decoded to empty buffer: $path');
    }

    final leftPtr = leftOut.value;
    final leftCopy = Float32List(n);
    leftCopy.setAll(0, leftPtr.asTypedList(n));
    free(leftPtr);

    final rightPtr = rightOut.value;
    final rightCopy = Float32List(n);
    rightCopy.setAll(0, rightPtr.asTypedList(n));
    free(rightPtr);

    return DecodedAudioStereo(
      left: leftCopy,
      right: rightCopy,
      sampleRate: sr > 0 ? sr : 16000,
      sourceChannels: ch,
    );
  } finally {
    calloc.free(pathPtr);
    calloc.free(leftOut);
    calloc.free(rightOut);
    calloc.free(nOut);
    calloc.free(srOut);
    calloc.free(chOut);
  }
}
