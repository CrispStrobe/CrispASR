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
  List<SessionSegment> transcribe(Float32List pcm) {
    if (_closed) throw StateError('CrispasrSession is closed');
    if (pcm.isEmpty) return const [];

    final samples = calloc<Float>(pcm.length);
    for (var i = 0; i < pcm.length; i++) {
      samples[i] = pcm[i];
    }
    final transcribeFn = _lib.lookupFunction<
        Pointer<Void> Function(Pointer<Void>, Pointer<Float>, Int32),
        Pointer<Void> Function(Pointer<Void>, Pointer<Float>, int)>(
      'crispasr_session_transcribe',
    );
    final res = transcribeFn(_handle, samples, pcm.length);
    calloc.free(samples);
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
