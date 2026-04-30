// Integration test — real transcription via the unified Session API.
//
// Requires CRISPASR_LIB pointing at the built libwhisper and model files:
//   CRISPASR_LIB=../../../build-shared/src/libwhisper.so \
//   CRISPASR_MODEL=../../../models/ggml-tiny.en.bin \
//   dart test/transcription_test.dart

import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:test/test.dart';

DynamicLibrary _openLib() {
  final path = Platform.environment['CRISPASR_LIB'];
  if (path != null && path.isNotEmpty) return DynamicLibrary.open(path);
  if (Platform.isMacOS) return DynamicLibrary.open('../../build/src/libwhisper.dylib');
  return DynamicLibrary.open('../../build/src/libwhisper.so');
}

Float32List _loadJfkPcm() {
  final wavPath = '${Directory.current.parent.parent.path}/samples/jfk.wav';
  final file = File(wavPath);
  if (!file.existsSync()) {
    throw StateError('jfk.wav not found at $wavPath');
  }
  final bytes = file.readAsBytesSync();
  // Skip 44-byte WAV header, read as int16, convert to float32
  final int16 = bytes.buffer.asInt16List(44);
  return Float32List.fromList(int16.map((s) => s / 32768.0).toList());
}

void main() {
  late DynamicLibrary lib;

  setUpAll(() {
    lib = _openLib();
  });

  test('session open + transcribe whisper-tiny', () {
    final modelPath = Platform.environment['CRISPASR_MODEL'] ??
        '${Directory.current.parent.parent.path}/models/ggml-tiny.en.bin';
    if (!File(modelPath).existsSync()) {
      print('SKIP: whisper model not found at $modelPath');
      return;
    }

    // Open session
    final openFn = lib.lookupFunction<
        Pointer Function(Pointer<Utf8>, Int32),
        Pointer Function(Pointer<Utf8>, int)>('crispasr_session_open');
    final closeFn = lib.lookupFunction<
        Void Function(Pointer),
        void Function(Pointer)>('crispasr_session_close');
    final backendFn = lib.lookupFunction<
        Pointer<Utf8> Function(Pointer),
        Pointer<Utf8> Function(Pointer)>('crispasr_session_backend');
    final transcribeFn = lib.lookupFunction<
        Pointer Function(Pointer, Pointer<Float>, Int32),
        Pointer Function(Pointer, Pointer<Float>, int)>('crispasr_session_transcribe');
    final nSegFn = lib.lookupFunction<
        Int32 Function(Pointer),
        int Function(Pointer)>('crispasr_session_result_n_segments');
    final segTextFn = lib.lookupFunction<
        Pointer<Utf8> Function(Pointer, Int32),
        Pointer<Utf8> Function(Pointer, int)>('crispasr_session_result_segment_text');
    final segT0Fn = lib.lookupFunction<
        Int64 Function(Pointer, Int32),
        int Function(Pointer, int)>('crispasr_session_result_segment_t0');
    final segT1Fn = lib.lookupFunction<
        Int64 Function(Pointer, Int32),
        int Function(Pointer, int)>('crispasr_session_result_segment_t1');
    final resultFreeFn = lib.lookupFunction<
        Void Function(Pointer),
        void Function(Pointer)>('crispasr_session_result_free');

    final pathPtr = modelPath.toNativeUtf8();
    final session = openFn(pathPtr, 2);
    malloc.free(pathPtr);
    expect(session.address, isNot(0), reason: 'session should open');

    final be = backendFn(session).toDartString();
    expect(be, 'whisper');

    // Transcribe jfk.wav
    final pcm = _loadJfkPcm();
    final pcmPtr = malloc<Float>(pcm.length);
    for (int i = 0; i < pcm.length; i++) {
      pcmPtr[i] = pcm[i];
    }

    final result = transcribeFn(session, pcmPtr, pcm.length);
    malloc.free(pcmPtr);
    expect(result.address, isNot(0), reason: 'transcribe should succeed');

    final nSeg = nSegFn(result);
    expect(nSeg, greaterThan(0));

    final texts = <String>[];
    for (int i = 0; i < nSeg; i++) {
      final tp = segTextFn(result, i);
      texts.add(tp.toDartString().trim());
      final t0 = segT0Fn(result, i) / 100.0;
      final t1 = segT1Fn(result, i) / 100.0;
      expect(t0, greaterThanOrEqualTo(0.0));
      expect(t1, greaterThan(t0));
      expect(t1, lessThan(15.0));
    }

    resultFreeFn(result);
    closeFn(session);

    final full = texts.join(' ').toLowerCase();
    expect(full, contains('fellow americans'));
    expect(full, contains('country'));
  });

  test('available backends includes whisper and parakeet', () {
    final fn = lib.lookupFunction<
        Int32 Function(Pointer<Utf8>, Int32),
        int Function(Pointer<Utf8>, int)>('crispasr_session_available_backends');
    final buf = malloc<Uint8>(256);
    final n = fn(buf.cast<Utf8>(), 256);
    expect(n, greaterThan(0));
    final backends = buf.cast<Utf8>().toDartString();
    malloc.free(buf);
    expect(backends, contains('whisper'));
    expect(backends, contains('parakeet'));
  });
}
