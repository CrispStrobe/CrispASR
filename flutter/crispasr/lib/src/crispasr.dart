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

  bool get supportsExtended => _detectLang != null;

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

      final ret = _full(_ctx, params, samples, pcm.length);
      if (ret != 0) throw Exception('Transcription failed (error $ret)');

      return _collectSegments(wantWords: opts.wordTimestamps);
    } finally {
      _freeParams(params);
      calloc.free(samples);
      if (langPtr != null) calloc.free(langPtr);
      if (promptPtr != null) calloc.free(promptPtr);
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

  void dispose() {
    if (!_disposed) {
      _free(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispASR has been disposed');
  }

  static String _findLib() {
    if (Platform.isAndroid || Platform.isLinux) return 'libwhisper.so';
    if (Platform.isIOS || Platform.isMacOS) return 'whisper.framework/whisper';
    if (Platform.isWindows) return 'whisper.dll';
    return 'libwhisper.so';
  }
}
