// Smoke test — confirms every 0.2.0 FFI symbol resolves against the freshly
// built libwhisper. Does NOT run real transcription (that needs a model
// download); purely checks the binding surface.
//
// Requires CRISPASR_LIB pointing at the built libwhisper:
//   CRISPASR_LIB=../../../build/src/libwhisper.dylib dart test/bindings_smoke_test.dart

import 'dart:ffi';
import 'dart:io';

import 'package:crispasr/crispasr.dart' show LidMethod;
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
      // Shared VAD-driven session transcribe (0.4.3+). Merges slices,
      // stitches with 0.1s gaps, remaps timestamps — same algorithm
      // as the CLI's --vad path, reachable from every binding.
      'crispasr_session_transcribe_vad',
      // Language-aware session transcribe (0.4.9+). Accepts an ISO
      // 639-1 code so caller-supplied LID results can feed into the
      // backend's source-language hint.
      'crispasr_session_transcribe_lang',
      'crispasr_session_transcribe_vad_lang',
      // Shared diarization (0.4.5+). Assigns a speaker index to each
      // caller-supplied segment; 4 methods (energy, xcorr, vad-turns,
      // pyannote) share the library path with the CLI.
      'crispasr_diarize_segments_abi',
      // Shared language identification (0.4.6+). Two methods (whisper,
      // silero) share the library path with the CLI.
      'crispasr_detect_language_pcm',
      // Shared CTC / forced-aligner word timings (0.4.7+). Canary-CTC
      // and Qwen3-ForcedAligner paths share one entry point.
      'crispasr_align_words_abi',
      'crispasr_align_result_n_words',
      'crispasr_align_result_word_text',
      'crispasr_align_result_word_t0',
      'crispasr_align_result_word_t1',
      'crispasr_align_result_free',
      // Shared HF download + cache + model registry (0.4.8+).
      'crispasr_cache_ensure_file_abi',
      'crispasr_cache_dir_abi',
      'crispasr_registry_lookup_abi',
      'crispasr_registry_lookup_by_filename_abi',
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

  test('c_api_version: canonical + alias agree on the same string', () {
    // Don't pin a specific version — the C side bumps independently
    // of the Dart binding. The test's real intent is to verify that
    // the deprecated alias `crispasr_dart_helpers_version` keeps
    // returning the same string as the canonical
    // `crispasr_c_api_version` until the alias is removed.
    final versions = <String>[];
    for (final sym in const [
      'crispasr_c_api_version',
      'crispasr_dart_helpers_version', // deprecated alias
    ]) {
      final fn = lib.lookupFunction<Pointer<Utf8> Function(),
          Pointer<Utf8> Function()>(sym);
      final ptr = fn();
      expect(ptr.cast<Uint8>().address, isNot(0), reason: sym);
      final s = ptr.toDartString();
      expect(s, matches(RegExp(r'^\d+\.\d+\.\d+')),
          reason: '$sym should return a semver-shaped string');
      versions.add(s);
    }
    expect(versions[0], versions[1],
        reason: 'canonical + alias must report the same version string');
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

  test('0.5.9 grammar-constrained sampling helpers resolve', () {
    // crispasr_session_set_grammar_text was added in 0.5.9 — it
    // threads a parsed GBNF graph through wparams.grammar_rules
    // on the whisper transcribe dispatch. Symbol-presence test;
    // the real parse-and-bind path is exercised by the
    // CrispasrSession.setGrammar() Dart smoke below.
    expect(() => lib.lookup('crispasr_session_set_grammar_text'),
        returnsNormally,
        reason: 'rebuild libcrispasr — 0.5.9 grammar setter is missing');
  });

  test('0.5.13 whisper alt-token capture symbols resolve', () {
    // 0.5.13 adds top-N alternative-candidate capture for whisper
    // greedy decode. The C side exposes them at three layers:
    //   * params-level (`crispasr_params_set_alt_n`) for the
    //     low-level transcribePcm path
    //   * session-level sticky setter
    //     (`crispasr_session_set_alt_n`)
    //   * per-token + per-word accessors so consumers can surface
    //     runner-up candidates in tap-to-pick UIs.
    // Pre-0.5.13 dylibs don't have any of these; the Dart wrapper
    // raises UnsupportedError so apps can graceful-degrade.
    for (final s in [
      'crispasr_params_set_alt_n',
      'crispasr_session_set_alt_n',
      'crispasr_token_n_alts',
      'crispasr_token_alt_id',
      'crispasr_token_alt_p',
      'crispasr_token_alt_text',
      'crispasr_session_result_word_n_alts',
      'crispasr_session_result_word_alt_text',
      'crispasr_session_result_word_alt_p',
    ]) {
      expect(() => lib.lookup(s), returnsNormally,
          reason: 'rebuild libcrispasr — 0.5.13 alt-token symbol '
              '$s is missing');
    }
  });

  test('0.5.12 audio enhancement helper resolves', () {
    // crispasr_enhance_audio_rnnoise runs RNNoise on a 16 kHz mono
    // float32 buffer (upsample → denoise frames → downsample) as a
    // transcribe pre-step. Pre-0.5.12 dylibs don't have the symbol;
    // the Dart wrapper raises UnsupportedError so apps can
    // graceful-degrade to the un-enhanced PCM.
    expect(() => lib.lookup('crispasr_enhance_audio_rnnoise'),
        returnsNormally,
        reason: 'rebuild libcrispasr — 0.5.12 audio enhancement '
            'helper is missing');
  });

  test('0.5.11 whisper decode-extras setter resolves', () {
    // crispasr_session_set_whisper_decode_extras writes
    // wparams.suppress_nst + suppress_regex + carry_initial_prompt
    // on every whisper transcribe. Pre-0.5.11 dylibs don't have
    // the symbol; the Dart wrapper raises UnsupportedError so
    // apps can graceful-degrade.
    expect(
        () => lib
            .lookup('crispasr_session_set_whisper_decode_extras'),
        returnsNormally,
        reason: 'rebuild libcrispasr — 0.5.11 whisper decode-extras '
            'setter is missing');
  });

  test('0.5.10 whisper decoder-fallback thresholds setter resolves', () {
    // crispasr_session_set_fallback_thresholds writes the four
    // wparams.*_thold fields + wparams.temperature_inc on every
    // whisper transcribe. Pre-0.5.10 dylibs don't have the
    // symbol; the Dart wrapper raises UnsupportedError so apps
    // can graceful-degrade.
    expect(
        () => lib.lookup('crispasr_session_set_fallback_thresholds'),
        returnsNormally,
        reason: 'rebuild libcrispasr — 0.5.10 fallback-threshold '
            'setter is missing');
  });

  test('C-ABI parity: parakeet + kokoro + vad_slices + lcs symbols resolve', () {
    for (final s in [
      'crispasr_parakeet_init',
      'crispasr_parakeet_free',
      'crispasr_parakeet_transcribe',
      'crispasr_parakeet_result_text',
      'crispasr_parakeet_result_n_words',
      'crispasr_parakeet_result_word_text',
      'crispasr_parakeet_result_word_t0',
      'crispasr_parakeet_result_word_t1',
      'crispasr_parakeet_result_n_tokens',
      'crispasr_parakeet_result_token_text',
      'crispasr_parakeet_result_token_t0',
      'crispasr_parakeet_result_token_t1',
      'crispasr_parakeet_result_token_p',
      'crispasr_parakeet_result_free',
      'crispasr_kokoro_lang_is_german_abi',
      'crispasr_kokoro_lang_has_native_voice_abi',
      'crispasr_kokoro_resolve_model_for_lang_abi',
      'crispasr_kokoro_resolve_fallback_voice_abi',
      'crispasr_lcs_dedup_prefix_count',
      'crispasr_vad_slices',
      'crispasr_stream_set_live_decode',
      'crispasr_titanet_cosine_sim',
      'crispasr_session_open_with_params',
      'crispasr_session_translate_text',
      'crispasr_session_translate_text_free',
      'crispasr_session_result_word_p',
      'crispasr_params_set_max_tokens',
      'crispasr_text_detect_language',
      'crispasr_enhance_audio_rnnoise',
    ]) {
      expect(() => lib.lookup(s), returnsNormally,
          reason: 'missing C-ABI symbol: $s');
    }
  });

  test('LidMethod enum indexes match the C-side CrispasrLidMethod', () {
    // crispasr_detect_language_pcm dispatches on the int value of
    // `method.index`; the C side's `enum class CrispasrLidMethod`
    // hard-codes Whisper=0, Silero=1, Firered=2, Ecapa=3 (see
    // src/crispasr_lid.h). If somebody reorders the Dart enum or
    // inserts a new variant in the middle, every Firered/Ecapa
    // call silently routes to the wrong backend with no compile
    // error. Pin the indexes here so a reorder shows up as a red
    // test, not a runtime regression.
    expect(LidMethod.whisper.index, 0);
    expect(LidMethod.silero.index, 1);
    expect(LidMethod.firered.index, 2);
    expect(LidMethod.ecapa.index, 3);
    expect(LidMethod.values.length, 4,
        reason: 'extending LidMethod without bumping the C-side enum '
            'will silently drop the new variant');
  });
}
