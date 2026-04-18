// Smoke test — confirms every 0.2.0 FFI symbol resolves against the freshly
// built libwhisper. Does NOT run real transcription (that needs a model
// download); purely checks the binding surface.
//
// Requires CRISPASR_LIB pointing at the built libwhisper:
//   CRISPASR_LIB=../../../build/src/libwhisper.dylib dart test/bindings_smoke_test.dart

import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:test/test.dart';

DynamicLibrary _openLib() {
  final path = Platform.environment['CRISPASR_LIB'];
  if (path != null && path.isNotEmpty) {
    return DynamicLibrary.open(path);
  }
  if (Platform.isMacOS || Platform.isIOS) {
    return DynamicLibrary.open('../../build/src/libwhisper.dylib');
  }
  return DynamicLibrary.open('../../build/src/libwhisper.so');
}

void main() {
  late DynamicLibrary lib;

  setUpAll(() {
    lib = _openLib();
  });

  test('0.1.0 whisper symbols resolve', () {
    for (final s in [
      'whisper_init_from_file_with_params',
      'whisper_free',
      'whisper_full',
      'whisper_full_default_params_by_ref',
      'whisper_context_default_params_by_ref',
      'whisper_full_n_segments',
      'whisper_full_get_segment_text',
      'whisper_full_get_segment_t0',
      'whisper_full_get_segment_t1',
      'whisper_full_get_segment_no_speech_prob',
      'whisper_free_params',
    ]) {
      expect(() => lib.lookup(s), returnsNormally, reason: s);
    }
  });

  test('0.2.0 crispasr_ helpers resolve', () {
    for (final s in [
      'crispasr_params_set_language',
      'crispasr_params_set_translate',
      'crispasr_params_set_detect_language',
      'crispasr_params_set_token_timestamps',
      'crispasr_params_set_n_threads',
      'crispasr_params_set_max_len',
      'crispasr_params_set_split_on_word',
      'crispasr_params_set_no_context',
      'crispasr_params_set_single_segment',
      'crispasr_params_set_print_realtime',
      'crispasr_params_set_print_progress',
      'crispasr_params_set_print_timestamps',
      'crispasr_params_set_print_special',
      'crispasr_params_set_suppress_blank',
      'crispasr_params_set_temperature',
      'crispasr_params_set_initial_prompt',
      'crispasr_token_t0',
      'crispasr_token_t1',
      'crispasr_token_p',
      'crispasr_detect_language',
      'crispasr_vad_segments',
      'crispasr_vad_free',
      // Canonical C-ABI version symbol (was `crispasr_dart_helpers_version`
      // before the file moved to `src/crispasr_c_api.cpp`).
      'crispasr_c_api_version',
      // Back-compat alias, kept for one release cycle.
      'crispasr_dart_helpers_version',
    ]) {
      expect(() => lib.lookup(s), returnsNormally, reason: s);
    }
  });

  test('0.2.0 whisper language helpers resolve', () {
    for (final s in [
      'whisper_lang_max_id',
      'whisper_lang_id',
      'whisper_lang_str',
      'whisper_full_n_tokens',
      'whisper_full_get_token_text',
      'whisper_full_get_token_p',
    ]) {
      expect(() => lib.lookup(s), returnsNormally, reason: s);
    }
  });

  test('c_api_version reports 0.4.0 (canonical + alias agree)', () {
    for (final sym in const [
      'crispasr_c_api_version',
      'crispasr_dart_helpers_version', // deprecated alias
    ]) {
      final fn = lib.lookupFunction<Pointer<Utf8> Function(),
          Pointer<Utf8> Function()>(sym);
      final ptr = fn();
      expect(ptr.cast<Uint8>().address, isNot(0), reason: sym);
      expect(ptr.toDartString(), '0.4.0', reason: sym);
    }
  });

  test('0.3.0 streaming helpers resolve', () {
    for (final s in [
      'crispasr_stream_open',
      'crispasr_stream_feed',
      'crispasr_stream_flush',
      'crispasr_stream_get_text',
      'crispasr_stream_close',
    ]) {
      expect(() => lib.lookup(s), returnsNormally, reason: s);
    }
  });

  test('0.4.1 audio decoder helpers resolve', () {
    for (final s in [
      'crispasr_audio_load',
      'crispasr_audio_free',
    ]) {
      expect(() => lib.lookup(s), returnsNormally, reason: s);
    }
  });

  test('0.4.0 unified session helpers resolve', () {
    for (final s in [
      'crispasr_session_open',
      'crispasr_session_open_explicit',
      'crispasr_session_backend',
      'crispasr_session_available_backends',
      'crispasr_session_transcribe',
      'crispasr_session_result_n_segments',
      'crispasr_session_result_segment_text',
      'crispasr_session_result_segment_t0',
      'crispasr_session_result_segment_t1',
      'crispasr_session_result_n_words',
      'crispasr_session_result_word_text',
      'crispasr_session_result_word_t0',
      'crispasr_session_result_word_t1',
      'crispasr_session_result_free',
      'crispasr_session_close',
      'crispasr_detect_backend_from_gguf',
    ]) {
      expect(() => lib.lookup(s), returnsNormally, reason: s);
    }
  });
}
