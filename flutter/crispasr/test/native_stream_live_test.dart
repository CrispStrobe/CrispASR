// Public Dart -> session C ABI integration, independent of the CLI adapter.
import 'dart:io';
import 'dart:typed_data';
import 'package:crispasr/crispasr.dart';
import 'package:test/test.dart';

Float32List readPcm(String path) {
  final bytes = File(path).readAsBytesSync();
  final b = ByteData.sublistView(bytes);
  for (var off = 12; off + 8 <= bytes.length;) {
    final n = b.getUint32(off + 4, Endian.little);
    if (String.fromCharCodes(bytes.sublist(off, off + 4)) == 'data') {
      return Float32List.fromList([
        for (var i = 0; i < n ~/ 2; i++)
          b.getInt16(off + 8 + i * 2, Endian.little) / 32768,
      ]);
    }
    off += 8 + n + (n & 1);
  }
  throw StateError('Missing PCM WAV data');
}

void main() {
  final modelPath = Platform.environment['CRISPASR_STREAM_MODEL'];
  final wavPath = Platform.environment['CRISPASR_STREAM_WAV'];
  final kind = int.parse(Platform.environment['CRISPASR_STREAM_KIND'] ?? '2');
  test('native German stream preserves state across chunks and reopening', () {
    final session = CrispasrSession.openWithParams(modelPath!,
        libPath: Platform.environment['CRISPASR_LIB'], useGpu: false, nThreads: 3);
    try {
      expect(session.streamingKind, kind);
      final pcm = readPcm(wavPath!);
      String run(int chunk) {
        final stream = session.openStream(language: 'de');
        var text = '';
        var updates = 0;
        try {
          for (var off = 0; off < pcm.length; off += chunk) {
            final update = stream.feed(Float32List.sublistView(
                pcm, off, (off + chunk).clamp(0, pcm.length)));
            if (update != null) {
              if (kind == 2) expect(update.text, startsWith(text));
              text = update.text;
              updates++;
            }
          }
          text = stream.flush()?.text ?? text;
          expect(stream.flush(), isNull);
          expect(() => stream.feed(Float32List(1600)), throwsException);
          expect(updates, greaterThan(2), reason: 'must emit before final flush');
          expect(text, isNot(contains('<de-DE>')));
          // Stream protocol test, not an exact-word accuracy gate: Nemotron
          // has opening-word errors on this fixture; retain substantive German checks.
          expect(text.toLowerCase(), contains('wasserhosen'));
          expect(text.toLowerCase(), contains('niederschläge'));
          stdout.writeln('chunk=$chunk updates=$updates: $text');
          return text;
        } finally {
          stream.close();
        }
      }
      expect(run(1777), run(5120),
          reason: 'feed partition must not reset model caches');
    } finally {
      session.close();
    }
  },
      skip: modelPath == null || wavPath == null
          ? 'set CRISPASR_STREAM_MODEL + CRISPASR_STREAM_WAV'
          : null,
      timeout: const Timeout(Duration(minutes: 8)));
}
