import 'dart:io';
import 'dart:typed_data';

import 'package:crispasr/crispasr.dart';
import 'package:test/test.dart';

void main() {
  late Directory temp;
  late String lib;
  setUpAll(() async {
    temp = Directory.systemTemp.createTempSync('crispasr-stream-test-');
    lib = '${temp.path}/stream.${Platform.isMacOS ? 'dylib' : 'so'}';
    final result = await Process.run('cc', [
      '-shared',
      '-fPIC',
      'test/fixtures/stream_protocol.c',
      '-o',
      lib,
    ]);
    expect(result.exitCode, 0, reason: '${result.stderr}');
  });
  tearDownAll(() => temp.deleteSync(recursive: true));

  test('stream polls async output, preserves long UTF-8 and deduplicates', () {
    final model = CrispasrSession.open('normal', libPath: lib);
    expect(model.streamingKind, 2);
    final stream = model.openStream(language: 'de');
    try {
      stream.setLiveDecode(true);
      expect(stream.feed(Float32List(1600)), isNull);
      final update = stream.feed(Float32List(1600));
      expect(update?.text, List.filled(3000, 'ä').join());
      expect(update?.end, .2);
      expect(stream.feed(Float32List(1600)), isNull);
      expect(stream.flush()?.counter, 2);
      stream.close();
      stream.close();
      expect(() => stream.feed(Float32List(1)), throwsStateError);
      expect(() => stream.flush(), throwsStateError);
    } finally {
      stream.close();
      model.close();
    }
  });

  test('native flush failures reach the caller', () {
    final model = CrispasrSession.open('errors', libPath: lib);
    final stream = model.openStream(language: 'de');
    try {
      expect(
        () => stream.flush(),
        throwsA(predicate((e) => '$e'.contains('-7'))),
      );
    } finally {
      stream.close();
      model.close();
    }
  });
}
