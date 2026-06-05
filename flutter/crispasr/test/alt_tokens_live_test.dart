// Live end-to-end test for §5.1.11 / 0.5.13 alt-token capture.
//
// Opens a CrispasrSession against ggml-tiny.en.bin, sets altN=3,
// transcribes samples/jfk.wav, and asserts:
//   (a) At least one returned word carries a non-empty `alts` list.
//   (b) Every alt's probability is in [0, 1] and the list is
//       descending by p.
//   (c) The chosen word's text never duplicates one of its own
//       alts (the C side excludes the picked token from the alt
//       list by design).
//
// Skips silently when CRISPASR_LIB / CRISPASR_MODEL are absent —
// same pattern as transcription_test.dart + grammar_test.dart so
// `dart test` in this package's CI step stays green even on
// model-less runners.
//
// Run locally with:
//   CRISPASR_LIB=../../build/src/libcrispasr.dylib \
//     dart test test/alt_tokens_live_test.dart
//
// On the dev box this also prints a small summary so you can
// eyeball the actual alt candidates whisper-tiny produces on JFK
// (helpful when the chip-row UX feels off — the alts are exactly
// what that UI will render).

import 'dart:io';
import 'dart:typed_data';

import 'package:crispasr/crispasr.dart';
import 'package:test/test.dart';

Float32List _loadJfkPcm() {
  // jfk.wav is a 16 kHz mono 16-bit PCM file — header is exactly
  // 44 bytes (the canonical PCM-WAV header), no LIST / INFO chunks,
  // no extension. Skip past it and reinterpret the rest as int16.
  final wavPath =
      '${Directory.current.parent.parent.path}/samples/jfk.wav';
  final file = File(wavPath);
  if (!file.existsSync()) {
    throw StateError('jfk.wav not found at $wavPath');
  }
  final bytes = file.readAsBytesSync();
  final int16 = bytes.buffer.asInt16List(44);
  return Float32List.fromList(int16.map((s) => s / 32768.0).toList());
}

void main() {
  final libEnv = Platform.environment['CRISPASR_LIB'];
  final modelEnv = Platform.environment['CRISPASR_MODEL'];
  final libPath = (libEnv != null &&
          libEnv.isNotEmpty &&
          File(libEnv).existsSync())
      ? libEnv
      : null;
  final modelPath = (modelEnv != null &&
          modelEnv.isNotEmpty &&
          File(modelEnv).existsSync())
      ? modelEnv
      : '${Directory.current.parent.parent.path}/models/ggml-tiny.en.bin';
  final canRun = libPath != null && File(modelPath).existsSync();
  final skipReason = canRun
      ? null
      : 'set CRISPASR_LIB + CRISPASR_MODEL (or drop ggml-tiny.en.bin '
          'into ../../models/) to run alt-token live test';

  group('§5.1.11 alt-token capture — live whisper-tiny', () {
    test('setAltN(3) populates Word.alts on JFK clip', () {
      final s = CrispasrSession.open(modelPath, libPath: libPath);
      try {
        // 3 alts per token — matches the CrisperWeaver UI's
        // recommended setting. Beam search is intentionally
        // left off (the default greedy path is where alt
        // capture fires).
        s.setAltN(3);

        final pcm = _loadJfkPcm();
        final segs = s.transcribe(pcm, language: 'en');
        expect(segs, isNotEmpty,
            reason: 'whisper should produce ≥1 segment on JFK');

        // Collect every word across every segment. Whisper
        // tiny gives ~1 segment for JFK so this is mostly
        // a single list.
        final allWords = segs.expand((seg) => seg.words).toList();
        expect(allWords, isNotEmpty,
            reason: 'session API should populate seg.words on '
                'whisper (token_timestamps gets forced on inside '
                'the session transcribe path when we need word '
                'data — see 0.5.13 release note)');

        // (a) At least one word has non-empty alts.
        final withAlts =
            allWords.where((w) => w.alts.isNotEmpty).toList();
        expect(withAlts, isNotEmpty,
            reason: 'altN=3 should leave at least one word '
                'with runner-up candidates — if this is empty '
                'the capture path isn\'t firing on the greedy '
                'sampling site');

        // (b) Every alt's p in [0, 1] and the list descends.
        for (final w in allWords) {
          double prev = 1.000001; // small epsilon for == ties
          for (final a in w.alts) {
            expect(a.p, inInclusiveRange(0.0, 1.0),
                reason: 'alt probabilities must be valid '
                    'softmax outputs (word: "${w.text}", '
                    'alt: "${a.text}")');
            expect(a.p, lessThanOrEqualTo(prev),
                reason: 'alts must come back descending by p '
                    '(word: "${w.text}")');
            prev = a.p;
          }
        }

        // (c) Chosen word never appears verbatim in its own
        // alts list — the C side filters the chosen token out.
        for (final w in withAlts) {
          final chosen = w.text.trim();
          for (final a in w.alts) {
            final altText = a.text.trim();
            expect(altText, isNot(equals(chosen)),
                reason: 'chosen token must not appear in alt '
                    'list (word: "${w.text}", alt: "${a.text}")');
          }
        }

        // Print a small summary — comes out under `dart test
        // --reporter=expanded` and is genuinely useful when
        // tuning the chip-row UX.
        print(
            '\n[alt-tokens] altN=3 on JFK: ${withAlts.length}/'
            '${allWords.length} words have runner-ups.');
        for (final w in withAlts.take(5)) {
          final altSummary = w.alts
              .map((a) => '${a.text.trim()}'
                  '(${(a.p * 100).toStringAsFixed(2)}%)')
              .join(', ');
          print('  ${w.text.trim().padRight(20)} → $altSummary');
        }
      } finally {
        s.close();
      }
    }, skip: skipReason, tags: ['live']);

    test('setAltN(0) clears alts on subsequent decode', () {
      final s = CrispasrSession.open(modelPath, libPath: libPath);
      try {
        // First decode with capture on — establish baseline.
        s.setAltN(3);
        final pcm = _loadJfkPcm();
        final withCapture = s.transcribe(pcm, language: 'en');
        final hasAlts = withCapture
            .expand((seg) => seg.words)
            .any((w) => w.alts.isNotEmpty);
        expect(hasAlts, isTrue,
            reason: 'sanity — altN=3 baseline should have alts');

        // Flip to off and re-decode the same audio. No alts
        // should come back this time. This pins the
        // "slider drag back to 0 actually disables capture"
        // behaviour — a regression there would leak compute
        // and confuse downstream consumers.
        s.setAltN(0);
        final withoutCapture = s.transcribe(pcm, language: 'en');
        final stillHasAlts = withoutCapture
            .expand((seg) => seg.words)
            .any((w) => w.alts.isNotEmpty);
        expect(stillHasAlts, isFalse,
            reason: 'altN=0 must not produce alts — a slider '
                'drag back to 0 has to actually disable capture');
      } finally {
        s.close();
      }
    }, skip: skipReason, tags: ['live']);
  });
}
