// Model-free FFI behavior test using checked-in probe at tests/fixtures/dart-wrapper-abi.c
// Compile on demand to TMPDIR or use CRISPASR_DART_PROBE env.
// Tests: bytes, copy-before-free mutation, empty, unavailable/old builds, errors/disposal/callback lifetime
// Covers shared tokens/raw bytes, chords/tab/convert with validation, TTS extras, sentence streaming, callbacks.

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:crispasr/crispasr.dart';

final _probeTemporaryDirectories = <Directory>[];
String? _cachedProbePath;
String? _cachedOldProbePath;

String _repoRoot() {
  // Try to locate repo root by searching upwards for include/crispasr_session.h
  var dir = Directory.current;
  try {
    final script = Platform.script.toFilePath();
    dir = File(script).parent;
  } catch (_) {}
  for (var i = 0; i < 8; i++) {
    final candidate = File('${dir.path}/include/crispasr_session.h');
    if (candidate.existsSync()) return dir.path;
    final alt = File('${dir.path}/include/crispasr.h');
    if (alt.existsSync()) return dir.path;
    final parent = dir.parent;
    if (parent.path == dir.path) break;
    dir = parent;
  }
  // Fallback: try common relative locations
  final fallbacks = [
    Directory.current.path,
    '${Directory.current.path}/../..',
    '${Directory.current.path}/../../..',
  ];
  for (final f in fallbacks) {
    if (File('$f/include/crispasr_session.h').existsSync()) {
      return Directory(f).resolveSymbolicLinksSync();
    }
  }
  return Directory.current.path;
}

String _tmpDir() {
  final env = Platform.environment['TMPDIR'];
  if (env != null && env.isNotEmpty) return env;
  return Directory.systemTemp.path;
}

String _compileProbe({bool oldTokens = false}) {
  if (!oldTokens && _cachedProbePath != null) return _cachedProbePath!;
  if (oldTokens && _cachedOldProbePath != null) return _cachedOldProbePath!;

  final envProbe = Platform.environment['CRISPASR_DART_PROBE'];
  if (!oldTokens &&
      envProbe != null &&
      envProbe.isNotEmpty &&
      File(envProbe).existsSync()) {
    _cachedProbePath = envProbe;
    return envProbe;
  }
  final envOld = Platform.environment['CRISPASR_DART_PROBE_OLD'];
  if (oldTokens &&
      envOld != null &&
      envOld.isNotEmpty &&
      File(envOld).existsSync()) {
    _cachedOldProbePath = envOld;
    return envOld;
  }

  final repo = _repoRoot();
  final probeSrc = File('$repo/tests/fixtures/dart-wrapper-abi.c');
  if (!probeSrc.existsSync()) {
    // Try alternative locations
    final alts = [
      File('tests/fixtures/dart-wrapper-abi.c'),
      File('../../tests/fixtures/dart-wrapper-abi.c'),
      File('../../../tests/fixtures/dart-wrapper-abi.c'),
    ];
    File? found;
    for (final a in alts) {
      if (a.existsSync()) {
        found = a;
        break;
      }
    }
    if (found == null) {
      throw StateError(
          'Probe source not found at $repo/tests/fixtures/dart-wrapper-abi.c');
    }
  }
  final srcPath = probeSrc.existsSync()
      ? probeSrc.path
      : 'tests/fixtures/dart-wrapper-abi.c';
  final tmp = Directory(_tmpDir()).createTempSync('crispasr-dart-probe-');
  final outPath = '${tmp.path}/${oldTokens ? "old" : "current"}.so';
  _probeTemporaryDirectories.add(tmp);

  final nativeSource = File('$repo/src/crispasr_c_api.cpp').readAsStringSync();
  final layouts = [
    'crispasr_diarize_seg_abi',
    'crispasr_diarize_turn_abi',
    'crispasr_diarize_opts_abi'
  ].map((name) {
    final start = nativeSource.indexOf('struct $name {');
    return nativeSource.substring(start, nativeSource.indexOf('};', start) + 2);
  }).join('\n');
  File('${tmp.path}/runtime-abi-layout.h').writeAsStringSync(layouts);
  final cc = Platform.environment['CC'] ?? 'cc';
  final includeArgs = [
    '-I',
    '$repo/include',
    '-I',
    '$repo/ggml/include',
    '-I',
    tmp.path,
  ];
  final args = [
    '-std=c11',
    '-shared',
    '-fPIC',
    '-O2',
    ...includeArgs,
    if (oldTokens) '-DCRISPASR_PROBE_OLD_TOKENS',
    srcPath,
    '$repo/tests/fixtures/wrapper-diarize-arguments.c',
    '-o',
    outPath,
  ];
  final result = Process.runSync(cc, args);
  if (result.exitCode != 0) {
    throw StateError(
        'Failed to compile probe: ${result.stdout}\n${result.stderr}\nargs: $cc ${args.join(' ')}');
  }
  if (oldTokens) {
    _cachedOldProbePath = outPath;
  } else {
    _cachedProbePath = outPath;
  }
  return outPath;
}

String get probePath => _compileProbe(oldTokens: false);
String get oldProbePath => _compileProbe(oldTokens: true);

DynamicLibrary _openProbe({bool old = false}) =>
    DynamicLibrary.open(old ? oldProbePath : probePath);

void _setMode(int m, {bool old = false}) {
  final lib = _openProbe(old: old);
  final fn = lib
      .lookupFunction<Int32 Function(Int32), int Function(int)>('probe_mode');
  fn(m);
}

int _getResultFrees({bool old = false}) {
  final lib = _openProbe(old: old);
  final fn = lib
      .lookupFunction<Int32 Function(), int Function()>('probe_result_frees');
  return fn();
}

int _getLive({bool old = false}) {
  final lib = _openProbe(old: old);
  final fn = lib.lookupFunction<Int32 Function(), int Function()>('probe_live');
  return fn();
}

void _probeMutate({bool old = false}) {
  final lib = _openProbe(old: old);
  final fn =
      lib.lookupFunction<Void Function(), void Function()>('probe_mutate');
  fn();
}

Float32List _pcm123() => Float32List.fromList([1, 3, 5]);

void main() {
  tearDownAll(() {
    for (final dir in _probeTemporaryDirectories) {
      if (dir.existsSync()) dir.deleteSync(recursive: true);
    }
  });
  test('generic speaker exact success and resource lifetime', () {
    final lib = _openProbe();
    final mode = lib.lookupFunction<Void Function(Int32), void Function(int)>(
        'probe_speaker_mode');
    final live = lib
        .lookupFunction<Int32 Function(), int Function()>('probe_speaker_live');
    mode(0);
    final embedder =
        CrispasrSpeakerEmbedder(lib, 'embed', nThreads: 2, cacheDir: 'cache');
    try {
      expect(embedder.dim, 3);
      expect(embedder.name, 'generic');
      final owned = embedder.embed(_pcm123());
      expect(owned, [.25, .5, .75]);
      for (final invalid in [2, 3, 4, 8]) {
        mode(invalid);
        expect(embedder.embed(_pcm123()), isNull);
      }
      mode(0);
      embedder.close();
      embedder.close();
      expect(owned, [.25, .5, .75]);
      expect(() => embedder.dim, throwsStateError);
      expect(() => embedder.name, throwsStateError);
      expect(() => embedder.embed(_pcm123()), throwsStateError);
    } finally {
      embedder.close();
      mode(0);
    }
    expect(live(), 0);
    expect(
        () => CrispasrSpeakerEmbedder(lib, 'embed\u0000bad',
            nThreads: 2, cacheDir: 'cache'),
        throwsArgumentError);
    expect(
        () => CrispasrSpeakerEmbedder(lib, 'embed',
            nThreads: 0x100000002, cacheDir: 'cache'),
        throwsArgumentError);
    expect(live(), 0);
  });
  test('generic cluster count labels input bounds and unassigned', () {
    final lib = _openProbe();
    final mode = lib.lookupFunction<Void Function(Int32), void Function(int)>(
        'probe_speaker_mode');
    final input = Float32List.fromList([1, 2, 3, 4, 5, 6]);
    List<int> run() => crispasrAgglomerativeCluster(lib, input,
        n: 2, dim: 3, mergeThreshold: .6, maxSpeakers: 3);
    mode(0);
    expect(run(), [0, 1]);
    for (final invalid in [2, 6, 7]) {
      mode(invalid);
      expect(run, throwsException);
    }
    mode(9);
    expect(run(), [-1, -1]);
    mode(0);
    expect(() => crispasrAgglomerativeCluster(lib, input, n: 2, dim: 1 << 62),
        throwsArgumentError);
    expect(
        () => crispasrAgglomerativeCluster(lib, input,
            n: 2, dim: 3, mergeThreshold: double.nan),
        throwsArgumentError);
    expect(
        () => crispasrAgglomerativeCluster(lib, input,
            n: 2, dim: 3, maxSpeakers: 0x100000003),
        throwsArgumentError);
  });
  test('generic pyannote cache transport failures and closed guards', () {
    final lib = _openProbe();
    final mode = lib.lookupFunction<Void Function(Int32), void Function(int)>(
        'probe_speaker_mode');
    final live = lib
        .lookupFunction<Int32 Function(), int Function()>('probe_speaker_live');
    mode(0);
    final cache = CrispasrPyannoteCache(lib, _pcm123(), 'seg', nThreads: 2);
    final segs = [
      DiarizeSegment(t0: 1.25, t1: 1.75, speaker: -1),
      DiarizeSegment(t0: 2, t1: 2.5, speaker: -1)
    ];
    try {
      cache.apply(segs, sliceT0: 1.25);
      expect(segs.map((s) => s.speaker), [2, 1]);
      for (final invalid in [2, 5]) {
        mode(invalid);
        expect(() => cache.apply(segs, sliceT0: 1.25), throwsException);
        expect(segs.map((s) => s.speaker), [2, 1]);
      }
      mode(0);
      expect(() => cache.apply(segs, sliceT0: double.nan), throwsArgumentError);
      cache.close();
      cache.close();
      expect(() => cache.apply(segs, sliceT0: 1.25), throwsStateError);
    } finally {
      cache.close();
      mode(0);
    }
    expect(live(), 0);
  });
  test('speaker database roster, negative scores, enrollment and closed guards',
      () {
    final lib = _openProbe();
    final mode = lib.lookupFunction<Void Function(Int32), void Function(int)>(
        'probe_db_mode');
    final live =
        lib.lookupFunction<Int32 Function(), int Function()>('probe_db_live');
    final calls =
        lib.lookupFunction<Int32 Function(), int Function()>('probe_db_calls');
    final emb = _pcm123();
    mode(0);
    final before = calls();
    expect(
        () => CrispasrSpeakerDB(lib, 'db',
            expectedNames: 'Alice,Bob', consentAttested: false),
        throwsStateError);
    expect(
        () => CrispasrSpeakerDB(lib, 'db',
            expectedNames: '', consentAttested: true),
        throwsArgumentError);
    expect(calls(), before);
    final db = CrispasrSpeakerDB(lib, 'db',
        expectedNames: 'Alice,Bob', consentAttested: true);
    expect(db.count, 2);
    expect(live(), 1);
    final match = db.match(emb);
    expect(match, ('アリス', .875));
    mode(1);
    expect(db.match(emb, threshold: -1), (null, -1.0));
    expect(match.$1, 'アリス');
    mode(0);
    expect(() => db.match(Float32List(0)), throwsArgumentError);
    expect(() => db.match(emb, threshold: double.nan), throwsArgumentError);
    expect(() => db.enroll('Alice', Float32List(0)), throwsArgumentError);
    expect(db.enroll('Alice', emb), true);
    mode(3);
    expect(db.enroll('Alice', emb), false);
    mode(2);
    expect(
        () => CrispasrSpeakerDB(lib, 'db',
            expectedNames: 'Alice,Bob', consentAttested: true),
        throwsStateError);
    db.close();
    db.close();
    expect(live(), 0);
    final closed = calls();
    expect(() => db.count, throwsStateError);
    expect(() => db.match(emb), throwsStateError);
    expect(() => db.enroll('Alice', emb), throwsStateError);
    expect(calls(), closed);
    mode(0);
  });
  test('diarization options, turn sizing, native counts and stereo bounds', () {
    final lib = _openProbe();
    final mode = lib.lookupFunction<Void Function(Int32), void Function(int)>(
        'probe_diarize_mode');
    final calls = lib.lookupFunction<Int32 Function(), int Function()>(
        'probe_diarize_calls');
    final segs = [
      DiarizeSegment(t0: 1.25, t1: 1.75),
      DiarizeSegment(t0: 2, t1: 2.5)
    ];
    final turns = <DiarizeTurn>[];
    bool run(
            {Float32List? right,
            bool stereo = false,
            List<DiarizeTurn>? output}) =>
        diarizeSegments(
            segs: segs,
            left: _pcm123(),
            right: right,
            isStereo: stereo,
            method: DiarizeMethod.foxNose,
            foxnoseEmbedderPath: 'embed',
            nThreads: 2,
            sliceT0: 1.25,
            minSpeakers: 2,
            maxSpeakers: 4,
            numSpeakers: 3,
            outTurns: output,
            lib: lib);
    mode(0);
    expect(run(), true);
    expect(segs.map((s) => s.speaker), [1, 0]);
    mode(1);
    final before = calls();
    expect(run(output: turns), true);
    expect(calls(), before + 2);
    expect(turns.length, 2);
    expect(turns.first.t0, 1.25);
    expect(turns.last.t1, 2.5);
    expect(turns.last.speaker, 0);
    mode(2);
    expect(() => run(output: turns), throwsStateError);
    mode(3);
    expect(() => run(output: turns), throwsStateError);
    mode(4);
    expect(run(), false);
    expect(run(output: turns), false);
    expect(turns, isEmpty);
    mode(0);
    final validCalls = calls();
    expect(() => run(stereo: true, right: Float32List(1)), throwsArgumentError);
    expect(() => run(stereo: true), throwsArgumentError);
    expect(calls(), validCalls);
    expect(run(stereo: true, right: _pcm123()), true);
  });
  test('legacy native pointer parameters, parallel decoding, DTW and cleanup',
      () {
    final tmp = Directory(_tmpDir()).createTempSync('dart-legacy-');
    addTearDown(() => tmp.deleteSync(recursive: true));
    final path = '${tmp.path}/legacy.so';
    final result = Process.runSync('python3',
        ['${_repoRoot()}/tests/build_legacy_wrapper_probe.py', path]);
    expect(result.exitCode, 0, reason: '${result.stdout}\n${result.stderr}');
    final lib = DynamicLibrary.open(path);
    final live = lib
        .lookupFunction<Int32 Function(), int Function()>('probe_legacy_live');
    final contexts = lib.lookupFunction<Int32 Function(), int Function()>(
        'probe_legacy_contexts');
    final mode = lib.lookupFunction<Void Function(Int32), void Function(int)>(
        'probe_legacy_mode');
    final processors = lib.lookupFunction<Int32 Function(), int Function()>(
        'probe_legacy_processors');
    final dtw = lib
        .lookupFunction<Int32 Function(), int Function()>('probe_legacy_dtw');
    final zero = lib.lookupFunction<Int32 Function(), int Function()>(
        'probe_legacy_progress_zero');
    expect(() => CrispASR('probe', libPath: path, dtw: true, dtwPreset: 2),
        throwsArgumentError);
    expect(() => CrispASR('fail', libPath: path), throwsException);
    expect(live(), 0);
    final model = CrispASR('probe', libPath: path, dtw: true, dtwTop: 2);
    expect(dtw(), 102);
    expect(live(), 0);
    expect(contexts(), 1);
    final opts = TranscribeOptions(
        language: 'de', initialPrompt: 'Grüße', wordTimestamps: true);
    final resultSegments = model.transcribePcm(Float32List.fromList([1, 3, 5]),
        options: opts, nProcessors: 2);
    expect(processors(), 2);
    expect(zero(), 1);
    expect(live(), 0);
    expect(resultSegments.single.text, '日');
    expect(resultSegments.single.start, 1.0);
    expect(resultSegments.single.words.first.start, -1);
    expect(resultSegments.single.words.first.end, -1);
    expect(resultSegments.single.words.first.text, contains('�'));
    expect(resultSegments.single.words.first.alts.single.text, 'alternative');
    expect(model.tokenDtwTime(0, 0), 1.4);
    expect(() => model.tokenDtwTime(1, 0), throwsRangeError);
    expect(() => model.tokenDtwTime(0, 2), throwsRangeError);
    mode(2);
    expect(
        () =>
            model.transcribePcm(Float32List.fromList([1, 3, 5]), options: opts),
        throwsException);
    expect(live(), 0);
    mode(3);
    expect(
        () =>
            model.transcribePcm(Float32List.fromList([1, 3, 5]), options: opts),
        throwsStateError);
    expect(live(), 0);
    mode(0);
    model.dispose();
    expect(contexts(), 0);
    expect(() => model.tokenDtwTime(0, 0), throwsStateError);
  });

  test('direct Parakeet passes integer offsets and owns independent results',
      () {
    final tmp = Directory(_tmpDir()).createTempSync('dart-parakeet-');
    addTearDown(() => tmp.deleteSync(recursive: true));
    final path = '${tmp.path}/parakeet.so';
    final root = _repoRoot();
    final cc = Process.runSync('cc', [
      '-shared',
      '-fPIC',
      '-Werror',
      '-I$root/include',
      '$root/tests/fixtures/wrapper-parakeet-legacy.c',
      '-o',
      path
    ]);
    expect(cc.exitCode, 0, reason: '${cc.stderr}');
    final lib = DynamicLibrary.open(path);
    final ctx = CrispasrParakeet(lib, 'probe');
    final result =
        ctx.transcribe(Float32List.fromList([1, 3, 5]), offsetCs: 123);
    expect(result.text, '日本語');
    expect(result.wordT0(0), 223);
    expect(result.tokenT0(0), -1);
    expect(
        utf8.decode([...result.tokenBytes(0), ...result.tokenBytes(1)]), '日');
    expect(result.tokenText(0), contains('�'));
    expect(
        () => ctx.transcribe(Float32List.fromList([1, 3, 5]), language: 'de'),
        throwsArgumentError);
    result.free();
    result.free();
    expect(() => result.text, throwsStateError);
    ctx.close();
    ctx.close();
    expect(() => ctx.transcribe(Float32List.fromList([1])), throwsStateError);
    final contexts = lib.lookupFunction<Int32 Function(), int Function()>(
        'probe_parakeet_contexts');
    final results = lib.lookupFunction<Int32 Function(), int Function()>(
        'probe_parakeet_results');
    expect(contexts(), 0);
    expect(results(), 0);
  });

  test('TitaNet float count, error counts and closed-context guards', () {
    final tmp = Directory(_tmpDir()).createTempSync('dart-titanet-');
    addTearDown(() => tmp.deleteSync(recursive: true));
    final path = '${tmp.path}/titanet.so';
    final root = _repoRoot();
    final cc = Process.runSync('cc', [
      '-shared',
      '-fPIC',
      '-Werror',
      '-I$root/include',
      '$root/tests/fixtures/wrapper-titanet-legacy.c',
      '-o',
      path
    ]);
    expect(cc.exitCode, 0, reason: '${cc.stderr}');
    final lib = DynamicLibrary.open(path);
    final mode = lib.lookupFunction<Void Function(Int32), void Function(int)>(
        'probe_titanet_mode');
    final live = lib
        .lookupFunction<Int32 Function(), int Function()>('probe_titanet_live');
    final ctx = CrispasrTitaNet(lib, 'probe');
    final pcm = Float32List.fromList([1, 3, 5]);
    final result = ctx.embed(pcm);
    expect(result.length, 192);
    expect(result[0], 1);
    for (final m in [2, 3, 4]) {
      mode(m);
      expect(() => ctx.embed(pcm), throwsStateError);
    }
    mode(0);
    expect(() => ctx.embed(Float32List(0)), throwsArgumentError);
    expect(() => CrispasrTitaNet.cosineSim(lib, Float32List(1), Float32List(2)),
        throwsArgumentError);
    expect(CrispasrTitaNet.cosineSim(lib, Float32List(0), Float32List(0)), 0);
    ctx.close();
    ctx.close();
    expect(live(), 0);
    expect(() => ctx.embed(pcm), throwsStateError);
  });

  // Ensure probes compile
  test('probe compiles and loads', () {
    final path = probePath;
    expect(File(path).existsSync(), isTrue,
        reason: 'probe at $path should exist');
    final lib = _openProbe();
    expect(lib.providesSymbol('probe_mode'), isTrue);
    expect(lib.providesSymbol('crispasr_session_transcribe'), isTrue);
    expect(lib.providesSymbol('crispasr_session_piano'), isTrue);
    expect(lib.providesSymbol('crispasr_session_chords'), isTrue);
    expect(lib.providesSymbol('crispasr_session_tab'), isTrue);
    expect(lib.providesSymbol('crispasr_session_convert'), isTrue);
    // Token symbols present in current probe
    expect(lib.providesSymbol('crispasr_session_result_n_tokens'), isTrue);
    expect(lib.providesSymbol('crispasr_session_result_token_text'), isTrue);
  });

  group('transcribe tokens - shared session', () {
    test('bytes lossless and UTF8 fragments', () {
      _setMode(0);
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      final segs = s.transcribe(_pcm123());
      expect(segs.length, 1);
      final seg = segs.first;
      expect(seg.text, '世');
      expect(seg.speaker, '(Speaker 1) ');
      expect(seg.words.length, 1);
      final w = seg.words.first;
      expect(w.text, '世');
      expect(w.start, -1);
      expect(w.end, -1);
      expect(w.hasTimestamps, isFalse);
      expect(w.p, closeTo(0.9, 1e-6));
      expect(w.alts.length, 1);
      expect(w.alts.first.text, 'alternative');
      // Tokens: raw bytes must reconstruct original character losslessly
      expect(seg.tokens.length, 2);
      final t0 = seg.tokens[0];
      final t1 = seg.tokens[1];
      expect(t0.rawBytes, [0xe4]);
      expect(t1.rawBytes, [0xb8, 0x96]);
      // Joining raw bytes should decode to original
      final joined = Uint8List.fromList([...t0.rawBytes, ...t1.rawBytes]);
      // raw bytes decoded as UTF8 should give original character
      expect(utf8.decode(joined), '世');
      // t.text decoded with replacement for fragment: first token is invalid alone, so text is replacement char
      expect(t0.text, '\uFFFD');
      // But raw bytes are preserved
      expect(t0.start, -1);
      expect(t0.hasTimestamps, isFalse);
      expect(t1.start, -1);
      expect(t0.p, closeTo(0.5, 1e-6));
      expect(t1.p, closeTo(0.75, 1e-6));
    });

    test('text cut inside a UTF-8 character decodes to U+FFFD, not a throw',
        () {
      _setMode(3);
      addTearDown(() => _setMode(0));
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      final seg = s.transcribe(_pcm123()).single;
      expect(seg.text, '�');
      expect(seg.words.single.text, '�');
      expect(seg.words.single.alts.single.text, '�');
    });

    test(
        'copy before free - mutate after transcribe retains Dart copy and frees count increments',
        () {
      _setMode(0);
      final beforeFrees = _getResultFrees();
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      final segs = s.transcribe(_pcm123());
      final afterFrees = _getResultFrees();
      // Each transcribe should free exactly one result
      expect(afterFrees, beforeFrees + 1);
      final originalBytes =
          Uint8List.fromList(segs.first.tokens.first.rawBytes);
      expect(originalBytes, [0xe4]);
      // Mutate native buffer - should NOT affect Dart copy
      _probeMutate();
      expect(segs.first.tokens.first.rawBytes, [0xe4]);
      expect(originalBytes, [0xe4]);
      // Verify piano buffer copy-before-free as well
      final s2 = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s2.close());
      final notes = s2.pianoNotes(Float32List.fromList([1, 3, 5]));
      final firstNoteCopy = notes.first.onMs;
      _probeMutate();
      expect(notes.first.onMs, firstNoteCopy);
      expect(notes.first.onMs, isNot(999));
    });

    test('empty results default', () {
      _setMode(1);
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      final segs = s.transcribe(_pcm123());
      expect(segs, isEmpty);
      final segs2 = s.transcribeChunked(_pcm123());
      expect(segs2, isEmpty);
      final segs3 = s.transcribeVad(_pcm123(), 'dummy-vad.gguf');
      expect(segs3, isEmpty);
      final (segs4, logits) = s.transcribeWithLogits(_pcm123());
      expect(segs4, isEmpty);
      expect(logits, isNull);
      // Piano/tab/chords/convert also empty on mode 1
      expect(s.pianoNotes(Float32List.fromList([1, 3, 5])), isEmpty);
      expect(s.chords(Float32List.fromList([1, 3, 5])), isEmpty);
      expect(s.tab(Float32List.fromList([1, 3, 5])).nFrames, 0);
    });

    test('all transcribe routes copy correctly and free', () {
      _setMode(0);
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      final before = _getResultFrees();
      for (final fn in [
        () => s.transcribe(_pcm123()),
        () => s.transcribeChunked(_pcm123()),
        () => s.transcribeVad(_pcm123(), 'vad.gguf'),
        () => s.transcribeWithLogits(_pcm123()).$1,
      ]) {
        final segs = fn();
        expect(segs.length, 1);
        expect(segs.first.tokens.length, 2);
        expect(segs.first.tokens.first.rawBytes, [0xe4]);
      }
      final after = _getResultFrees();
      expect(after, before + 4);
    });

    test('missing timing preserved as -1 not -0.01', () {
      _setMode(0);
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      final seg = s.transcribe(_pcm123()).first;
      // Word timing is -1 => should be -1 not -0.01
      expect(seg.words.first.start, -1);
      expect(seg.words.first.end, -1);
      expect(seg.words.first.start, isNot(closeTo(-0.01, 1e-9)));
      // Token timing also -1
      expect(seg.tokens.first.start, -1);
      expect(seg.tokens.first.end, -1);
    });

    test(
        'old library fallback - missing token symbols returns empty tokens via separate process',
        () async {
      // Build old probe and verify in subprocess that tokens are empty
      final oldPath = oldProbePath;
      final oldLib = DynamicLibrary.open(oldPath);
      expect(
          oldLib.providesSymbol('crispasr_session_result_n_tokens'), isFalse);
      expect(
          oldLib.providesSymbol('crispasr_session_result_token_text'), isFalse);

      // Write subprocess Dart file inside flutter/crispasr so package:crispasr resolves
      final repo = _repoRoot();
      final flutterDir = Directory('$repo/flutter/crispasr');
      final workDir = flutterDir.existsSync() ? flutterDir.path : repo;
      final subPath = '$workDir/test/_tmp_old_probe_test.dart';
      final subCode = '''
import 'dart:ffi';
import 'dart:typed_data';
import 'package:crispasr/crispasr.dart';
void main() {
  final s = CrispasrSession.open('dummy.gguf', libPath: r'$oldPath');
  final segs = s.transcribe(Float32List.fromList([1,3,5]));
  if (segs.isEmpty) throw StateError('expected segments');
  if (segs.first.tokens.isNotEmpty) {
    throw StateError('old library should give empty tokens but got \${segs.first.tokens.length}');
  }
  print('PASS old tokens empty');
  s.close();
}
''';
      File(subPath).writeAsStringSync(subCode);
      try {
        final runResult = await Process.run(
          Platform.executable,
          [subPath],
          workingDirectory: workDir,
        );
        expect(runResult.exitCode, 0,
            reason: '${runResult.stdout}\n${runResult.stderr}');
        expect(runResult.stdout.toString(), contains('PASS old tokens empty'));
      } finally {
        try {
          File(subPath).deleteSync();
        } catch (_) {}
      }
    });

    test('errors thrown on invalid pcm and closed session', () {
      _setMode(0);
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      // Invalid pcm should return null and throw
      expect(
          () => s.transcribe(Float32List.fromList([0, 0, 0])), throwsException);
      s.close();
      expect(() => s.transcribe(_pcm123()), throwsStateError);
      expect(() => s.transcribeChunked(_pcm123()), throwsStateError);
      expect(() => s.transcribeVad(_pcm123(), 'x'), throwsStateError);
    });

    test('callback exception always deferred and result_free still runs', () {
      _setMode(0);
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      // Set a token callback that throws
      s.setTokenCallback((raw, text, idx) {
        throw ArgumentError('sentinel');
      });
      final beforeFrees = _getResultFrees();
      // Transcribe fires token callback during native call; exception should be deferred
      // and result_free must still happen (frees count increments)
      // The Dart wrapper should capture callback error and rethrow after freeing
      expect(
          () => s.transcribe(_pcm123()),
          throwsA(isA<ArgumentError>()
              .having((e) => e.message, 'message', contains('sentinel'))));
      final afterFrees = _getResultFrees();
      expect(afterFrees, beforeFrees + 1,
          reason: 'result must be freed even when callback throws');
      s.setTokenCallback(null);
      // After clearing callback, transcribe should succeed
      final segs = s.transcribe(_pcm123());
      expect(segs.length, 1);
    });

    test('callback reentry is rejected and frees still runs', () {
      _setMode(0);
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      // Callback tries to re-enter session (e.g., call transcribe from inside callback)
      s.setProgressCallback((processed, total) {
        // This should not deadlock; should throw and be deferred
        expect(() => s.transcribe(_pcm123()), throwsStateError);
        expect(() => s.inputSampleRate, throwsStateError);
        expect(() => s.setTtsCfgScale(1.5), throwsStateError);
        // Alternative: try to close inside callback
      });
      // Transcribe will fire progress callback; reentry attempt should be handled
      final beforeFrees = _getResultFrees();
      final segs = s.transcribe(_pcm123());
      expect(segs.length, 1);
      expect(_getResultFrees(), beforeFrees + 1);
      s.setProgressCallback(null);
    });
  });

  test('drained segments and tokens share owned metadata and raw bytes', () {
    _setMode(0);
    final before = _getResultFrees();
    final segments = drainStreamedSegments(libPath: probePath);
    expect(segments.first.tokens.first.rawBytes, [0xe4]);
    expect(segments.first.words.first.alts.first.text, 'alternative');
    expect(segments.first.hasTimestamps, isTrue);
    expect(_getResultFrees(), before + 1);
    final pieces = drainStreamedTokenBytes(libPath: probePath);
    expect(utf8.decode([...pieces[0], ...pieces[1]]), '世');
    expect(drainStreamedTokens(libPath: probePath), isNotEmpty);
  });

  test('parent closes native streams before releasing model', () {
    final lib = _openProbe();
    final count = lib.lookupFunction<Int32 Function(), int Function()>(
        'probe_stream_closes');
    final bad = lib.lookupFunction<Int32 Function(), int Function()>(
        'probe_bad_parent_closes');
    final before = count();
    final session = CrispasrSession.open('probe', libPath: probePath);
    final stream = session.openStream();
    stream.feed(_pcm123());
    session.close();
    expect(stream.isClosed, isTrue);
    expect(count(), before + 1);
    expect(bad(), 0);
    expect(() => stream.feed(_pcm123()), throwsStateError);
    stream.close();
    expect(count(), before + 1);
  });

  group('chords', () {
    test('success returns spans with correct units', () {
      _setMode(0);
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      final spans = s.chords(Float32List.fromList([1, 3, 5]));
      expect(spans.length, 2);
      expect(spans.first.name, 'C');
      expect(spans.last.name, 'Am');
      expect(s.chordsVocabSize, 170);
    });
    test('empty on mode 1', () {
      _setMode(1);
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      expect(s.chords(Float32List.fromList([1, 3, 5])), isEmpty);
    });
    test('error on invalid rate or pcm', () {
      _setMode(0);
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      expect(() => s.chords(Float32List.fromList([0, 0, 0])), throwsException);
      expect(() => s.chords(Float32List.fromList([1, 3, 5]), sampleRate: -1),
          throwsArgumentError);
    });
  });

  group('tab', () {
    test('success returns emissions', () {
      _setMode(0);
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      final tab = s.tab(Float32List.fromList([1, 3, 5]));
      expect(tab.nFrames, 2);
      expect(tab.nStrings, 2);
      expect(tab.nClasses, 3);
      expect(tab.emissions.length, 2 * 2 * 3);
      expect(tab.silentClass, 2);
      expect(tab.framePeriod, closeTo(0.02, 1e-6));
      expect(tab.openMidi, [40, 45]);
    });
    test('error on empty', () {
      _setMode(0);
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      expect(() => s.tab(Float32List(0)), throwsArgumentError);
    });
  });

  group('convert', () {
    test('success with validation', () {
      _setMode(0);
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      expect(s.convertContentDim, 2);
      expect(s.convertNSpeakers, 1);
      expect(s.convertSampleRate, 40000);
      expect(s.convertInterChannels, 2);
      expect(s.convertUpsampleProduct, 4);
      final content = Float32List.fromList(
          [0, 1, 2, 3, 4, 5]); // 3 frames *2 dim, second element =1
      final f0 = Float32List.fromList([0, 0, 5]); // last =5
      final pcm = s.convert(content, f0, 0);
      expect(pcm.length, 6);
    });
    test('validation rejects wrong dim', () {
      _setMode(0);
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      final badContent =
          Float32List(6); // correct length but wrong second element
      final f0 = Float32List.fromList([0, 0, 5]);
      expect(() => s.convert(badContent, f0, 0), throwsException);
      expect(() => s.convert(Float32List(2), f0, 0),
          throwsArgumentError); // wrong length
      expect(
          () => s.convert(
              Float32List.fromList([0, 1, 2, 3, 4, 5]), Float32List(2), 0),
          throwsArgumentError);
      expect(() => s.convert(Float32List.fromList([0, 1, 2, 3, 4, 5]), f0, 5),
          throwsRangeError);
    });
    test('optional replay noise validated', () {
      _setMode(0);
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      final content = Float32List.fromList([0, 1, 2, 3, 4, 5]);
      final f0 = Float32List.fromList([0, 0, 5]);
      // correct noise
      final zp = Float32List(6)..[5] = 5; // 2*3, index 5 =5
      final sine = Float32List(12)..[11] = 11;
      expect(s.convert(content, f0, 0, noiseZp: zp, noiseSine: sine).length, 6);
      // wrong shape
      expect(() => s.convert(content, f0, 0, noiseZp: Float32List(2)),
          throwsArgumentError);
      expect(() => s.convert(content, f0, 0, noiseSine: Float32List(2)),
          throwsArgumentError);
    });
  });

  group('TTS extras', () {
    test('voice samples and cfg scale and tada', () {
      _setMode(0);
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      expect(
          () => s.setVoiceSamples(Float32List.fromList([1, 3, 5]), 24000,
              refText: 'hello'),
          returnsNormally);
      expect(
          () => s.setVoiceSamples(Float32List(0), 24000), throwsArgumentError);
      expect(() => s.setVoiceSamples(Float32List.fromList([1, 3, 5]), 0),
          throwsArgumentError);
      expect(
          () => s.setVoiceSamples(Float32List.fromList([1, 3, 5]), 24000,
              refText: 'bad'),
          throwsException);
      s.setTtsCfgScale(1.5);
      expect(() => s.setTtsCfgScale(9.9), returnsNormally);
      s.tadaSetMakerefModels();
      expect(() => s.tadaSetMakerefModels(encoderGguf: 'x'), throwsException);
    });
    test('disclaimer and error details', () {
      _setMode(0);
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      expect(
          CrispasrSession.disclaimerText(libPath: probePath), 'AI generated');
      final pcm = s.getDisclaimerPcm();
      expect(pcm.length, 2);
      expect(s.lastSynthError, 'probe synthesis failure');
      expect(_getLive(), 0);
      // After getting disclaimer, live count should be 0 after copy (we freed)
      _setMode(1);
      final s2 = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s2.close());
      expect(() => s2.getDisclaimerPcm(), throwsException);
    });
    test('sentence streaming isolate-safe', () {
      _setMode(0);
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      final chunks = <Float32List>[];
      final finals = <bool>[];
      s.synthesizeStreaming('hello', (pcm, isFinal) {
        chunks.add(pcm);
        finals.add(isFinal);
      });
      expect(chunks.length, 2);
      expect(finals, [false, true]);
      expect(chunks.first.length, 2);
      // After close, streaming should throw
      s.close();
      expect(() => s.synthesizeStreaming('x', (_, __) {}), throwsStateError);
    });
    test('callback lifetime - close clears holders and double close safe', () {
      _setMode(0);
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      s.setProgressCallback((a, b) {});
      s.setSegmentCallback((a, b, c, d) {});
      s.setTokenCallback((a, b, c) {});
      expect(() => s.close(), returnsNormally);
      // double close is safe
      expect(() => s.close(), returnsNormally);
    });
    test('disposal errors', () {
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      s.close();
      expect(() => s.transcribe(_pcm123()), throwsStateError);
      expect(() => s.pianoNotes(_pcm123()), throwsStateError);
      expect(() => s.chords(_pcm123()), throwsStateError);
      expect(() => s.tab(_pcm123()), throwsStateError);
      expect(
          () => s.convert(Float32List(6), Float32List(3), 0), throwsStateError);
      expect(() => s.setVoiceSamples(_pcm123(), 24000), throwsStateError);
      expect(() => s.synthesize('hi'), throwsStateError);
      expect(() => s.setTtsCfgScale(1.5), throwsStateError);
    });
  });

  test('actual audio allocations, aliased stereo, NUL bytes and error frees',
      () {
    final repo = _repoRoot();
    final tmp = Directory(_tmpDir()).createTempSync('crispasr-dart-audio-');
    addTearDown(() => tmp.deleteSync(recursive: true));
    final libPath = '${tmp.path}/audio.so';
    final build = Process.runSync(Platform.environment['CC'] ?? 'cc', [
      '-shared',
      '-fPIC',
      '-Werror',
      '-I$repo/include',
      '-I$repo/ggml/include',
      '$repo/tests/fixtures/wrapper-audio-utilities.c',
      '-o',
      libPath
    ]);
    expect(build.exitCode, 0, reason: '${build.stdout} ${build.stderr}');
    final lib = DynamicLibrary.open(libPath);
    final mode = lib.lookupFunction<Void Function(Int32), void Function(int)>(
        'probe_utility_mode');
    final live = lib
        .lookupFunction<Int32 Function(), int Function()>('probe_utility_live');
    expect(decodeAudioFile('x', libPath: libPath).samples, [.25, .5, .75]);
    expect(
        decodeAudioFileAtRate('x', 24000, libPath: libPath).sampleRate, 24000);
    expect(decodeAudioFileStereo('x', libPath: libPath).right, [.25, .5, .75]);
    mode(3);
    expect(decodeAudioFileStereo('x', libPath: libPath).sourceChannels, 1);
    expect(audioResample(_pcm123(), 16000, 24000, libPath: libPath),
        [.25, .5, .75]);
    expect(CrispasrC2pa.pcmToWav(_pcm123(), lib: lib), [87, 0, 65, 86]);
    expect(
        CrispasrC2pa.sign(Uint8List.fromList([1, 0, 2]),
            format: 'audio/wav', lib: lib),
        [1, 0, 2]);
    mode(2);
    expect(() => decodeAudioFile('x', libPath: libPath), throwsException);
    expect(() => decodeAudioFileAtRate('x', 24000, libPath: libPath),
        throwsException);
    expect(() => decodeAudioFileStereo('x', libPath: libPath), throwsException);
    expect(() => audioResample(_pcm123(), 16000, 24000, libPath: libPath),
        throwsException);
    expect(
        CrispasrC2pa.sign(Uint8List(1), format: 'audio/wav', lib: lib), isNull);
    expect(live(), 0);
  });

  group('top-level helpers', () {
    test('cApiVersion and backend caps', () {
      expect(cApiVersion(libPath: probePath), '0.8.99-probe');
      expect(backendCaps('probe', libPath: probePath), 'asr,tts');
      expect(backendCaps('unknown', libPath: probePath), isNull);
      expect(backendCapsList(libPath: probePath), contains('probe'));
      expect(detectBackendFromGguf('dummy', libPath: probePath), 'probe');
      final all = detectBackendsFromGguf('dummy', libPath: probePath);
      expect(all, ['probe']);
    });
    test('audio helpers availability', () {
      expect(() => decodeAudioFileAtRate('x', 16000, libPath: probePath),
          throwsException);
      expect(
          () =>
              audioResample(Float32List(10), 16000, 24000, libPath: probePath),
          throwsException);
    });
    test('freeAlignerCache', () {
      expect(() => freeAlignerCache(libPath: probePath), returnsNormally);
      expect(() => setGpuBackend('cuda', libPath: probePath), returnsNormally);
    });
    test('input/output channels', () {
      final s = CrispasrSession.open('dummy.gguf', libPath: probePath);
      addTearDown(() => s.close());
      expect(s.inputChannels, 1);
      expect(s.outputChannels, 1);
      expect(s.inputSampleRate, 16000);
      expect(s.outputSampleRate, 24000);
      s.setPcmSampleRate(16000);
      s.setParakeetAttContext(-1, -1);
      s.setTtsPadSilenceMs(100);
    });
    test('stereo audio decode via probe', () {
      // Probe returns -1 for stereo, should throw
      expect(() => decodeAudioFileStereo('x', libPath: probePath),
          throwsException);
    });
    test('watermark and c2pa containers', () {
      // These are top-level helpers, should be callable even if probe returns stub
      final pcm = Float32List.fromList([0.1, 0.2, 0.3]);
      final lib = DynamicLibrary.open(probePath);
      final watermarked = CrispasrWatermark.embed(pcm, lib: lib);
      expect(watermarked, isA<Float32List>());
      expect(CrispasrWatermark.detect(pcm, lib: lib), isA<double>());
      final wav = CrispasrC2pa.pcmToWav(pcm, sampleRate: 16000, lib: lib);
      // Probe returns null for pcmToWav, so expect null
      expect(wav, isNull);
    });
  });
}
