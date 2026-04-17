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

  Segment({
    required this.text,
    required this.start,
    required this.end,
    this.noSpeechProb = 0.0,
  });

  @override
  String toString() => '[${start.toStringAsFixed(1)}s - ${end.toStringAsFixed(1)}s] $text';
}

// FFI typedefs
typedef _WhisperInitNative = Pointer<Void> Function(Pointer<Utf8>, Pointer<Void>);
typedef _WhisperInit = Pointer<Void> Function(Pointer<Utf8>, Pointer<Void>);

typedef _WhisperFreeNative = Void Function(Pointer<Void>);
typedef _WhisperFree = void Function(Pointer<Void>);

typedef _WhisperFullNative = Int32 Function(Pointer<Void>, Pointer<Void>, Pointer<Float>, Int32);
typedef _WhisperFull = int Function(Pointer<Void>, Pointer<Void>, Pointer<Float>, int);

typedef _DefaultParamsNative = Pointer<Void> Function(Int32);
typedef _DefaultParams = Pointer<Void> Function(int);

typedef _DefaultCtxParamsNative = Pointer<Void> Function();
typedef _DefaultCtxParams = Pointer<Void> Function();

typedef _NSegmentsNative = Int32 Function(Pointer<Void>);
typedef _NSegments = int Function(Pointer<Void>);

typedef _GetTextNative = Pointer<Utf8> Function(Pointer<Void>, Int32);
typedef _GetText = Pointer<Utf8> Function(Pointer<Void>, int);

typedef _GetT0Native = Int64 Function(Pointer<Void>, Int32);
typedef _GetT0 = int Function(Pointer<Void>, int);

typedef _GetNSPNative = Float Function(Pointer<Void>, Int32);
typedef _GetNSP = double Function(Pointer<Void>, int);

typedef _FreeParamsNative = Void Function(Pointer<Void>);
typedef _FreeParams = void Function(Pointer<Void>);

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

  late final _WhisperFull _full;
  late final _WhisperFree _free;
  late final _DefaultParams _defaultParams;
  late final _NSegments _nSegments;
  late final _GetText _getText;
  late final _GetT0 _getT0;
  late final _GetT0 _getT1;
  late final _GetNSP _getNSP;
  late final _FreeParams _freeParams;

  CrispASR(String modelPath, {String? libPath}) {
    _lib = DynamicLibrary.open(libPath ?? _findLib());

    final init = _lib.lookupFunction<_WhisperInitNative, _WhisperInit>(
        'whisper_init_from_file_with_params');
    _free = _lib.lookupFunction<_WhisperFreeNative, _WhisperFree>('whisper_free');
    _full = _lib.lookupFunction<_WhisperFullNative, _WhisperFull>('whisper_full');
    _defaultParams = _lib.lookupFunction<_DefaultParamsNative, _DefaultParams>(
        'whisper_full_default_params_by_ref');
    _nSegments = _lib.lookupFunction<_NSegmentsNative, _NSegments>(
        'whisper_full_n_segments');
    _getText = _lib.lookupFunction<_GetTextNative, _GetText>(
        'whisper_full_get_segment_text');
    _getT0 = _lib.lookupFunction<_GetT0Native, _GetT0>(
        'whisper_full_get_segment_t0');
    _getT1 = _lib.lookupFunction<_GetT0Native, _GetT0>(
        'whisper_full_get_segment_t1');
    _getNSP = _lib.lookupFunction<_GetNSPNative, _GetNSP>(
        'whisper_full_get_segment_no_speech_prob');
    _freeParams = _lib.lookupFunction<_FreeParamsNative, _FreeParams>(
        'whisper_free_params');

    final ctxDefault = _lib.lookupFunction<_DefaultCtxParamsNative, _DefaultCtxParams>(
        'whisper_context_default_params_by_ref')();
    final pathPtr = modelPath.toNativeUtf8();
    _ctx = init(pathPtr, ctxDefault);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load model: $modelPath');
    }
  }

  /// Transcribe raw PCM audio (float32, mono, 16kHz).
  List<Segment> transcribePcm(Float32List pcm, {int strategy = 0}) {
    _checkDisposed();

    final samples = calloc<Float>(pcm.length);
    for (var i = 0; i < pcm.length; i++) {
      samples[i] = pcm[i];
    }

    final params = _defaultParams(strategy);
    final ret = _full(_ctx, params, samples, pcm.length);
    _freeParams(params);
    calloc.free(samples);

    if (ret != 0) throw Exception('Transcription failed (error $ret)');

    final n = _nSegments(_ctx);
    final segments = <Segment>[];
    for (var i = 0; i < n; i++) {
      final textPtr = _getText(_ctx, i);
      final text = textPtr == nullptr ? '' : textPtr.toDartString();
      final t0 = _getT0(_ctx, i) / 100.0;
      final t1 = _getT1(_ctx, i) / 100.0;
      final nsp = _getNSP(_ctx, i);
      segments.add(Segment(text: text, start: t0, end: t1, noSpeechProb: nsp));
    }
    return segments;
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
