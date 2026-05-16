// Tests for CrispasrSession.setGrammar — GBNF parser + symbol-
// resolution path. Requires CRISPASR_LIB + CRISPASR_MODEL like
// transcription_test.dart; skips silently when either is absent.

import 'dart:io';

import 'package:crispasr/crispasr.dart';
import 'package:test/test.dart';

void main() {
  final libEnv = Platform.environment['CRISPASR_LIB'];
  final modelEnv = Platform.environment['CRISPASR_MODEL'];
  final libPath = (libEnv != null && libEnv.isNotEmpty && File(libEnv).existsSync())
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
          'into ../../models/) to run grammar smoke tests';

  group('CrispasrSession.setGrammar', () {
    test('parses + binds a simple GBNF, then clears via empty string',
        () {
      // Open against a tiny whisper model — grammar is whisper-only
      // and this is the smallest model that exercises the dispatch.
      final s = CrispasrSession.open(modelPath, libPath: libPath);
      try {
        const gbnf = 'root ::= "yes" | "no"\n';
        s.setGrammar(gbnf, rootRule: 'root', penalty: 100.0);
        // Idempotent re-set with the same source should also succeed —
        // the previous parse_state gets replaced cleanly.
        s.setGrammar(gbnf, rootRule: 'root');
        // Clearing via empty string returns 0 from the C side and
        // leaves the session in unconstrained mode.
        s.setGrammar('');
        s.clearGrammar(); // double-clear is a no-op, not an error
      } finally {
        s.close();
      }
    }, skip: skipReason);

    test('invalid GBNF source throws ArgumentError', () {
      final s = CrispasrSession.open(modelPath, libPath: libPath);
      try {
        // Garbage that the parser will reject — no valid rule
        // structure. The parser's failure mode is "empty rules
        // vector", which the C ABI maps to rc=-2 → ArgumentError.
        expect(() => s.setGrammar('@@@ not a grammar @@@'),
            throwsA(isA<ArgumentError>()));
      } finally {
        s.close();
      }
    }, skip: skipReason);

    test('unknown root rule throws ArgumentError', () {
      final s = CrispasrSession.open(modelPath, libPath: libPath);
      try {
        const gbnf = 'root ::= "hello"\n';
        // Valid grammar but root rule name doesn't exist — same
        // failure mode (rc=-2) as a parse failure.
        expect(
            () => s.setGrammar(gbnf, rootRule: 'nonexistent'),
            throwsA(isA<ArgumentError>()));
      } finally {
        s.close();
      }
    }, skip: skipReason);
  });
}
