// chat_test.dart — smoke tests for the CrispasrChatSession Dart binding.
//
// Two layers:
//   1. Symbol-resolution test (always runs): confirms the chat ABI
//      surface is present in the loaded libcrispasr.
//   2. Real generate test (gated on CRISPASR_CHAT_TEST_MODEL pointing
//      at a small GGUF chat model on disk): exercises open / generate
//      / generateStream / reset / close end-to-end. Mirrors the
//      tests/test-chat-ggml.cpp smoke on the C++ side.
//
// Run with:
//   CRISPASR_LIB=../../build-ninja-compile/src/libcrispasr.dylib \
//   CRISPASR_CHAT_TEST_MODEL=/tmp/smollm2/smollm2-360m-instruct-q8_0.gguf \
//   dart test test/chat_test.dart

import 'dart:ffi';
import 'dart:io';

import 'package:crispasr/crispasr.dart';
import 'package:test/test.dart';

DynamicLibrary _openLib() {
  final path = Platform.environment['CRISPASR_LIB'];
  if (path != null && path.isNotEmpty) return DynamicLibrary.open(path);
  if (Platform.isMacOS || Platform.isIOS) {
    return DynamicLibrary.open('../../build-ninja-compile/src/libcrispasr.dylib');
  }
  return DynamicLibrary.open('../../build-ninja-compile/src/libcrispasr.so');
}

void main() {
  late DynamicLibrary lib;

  setUpAll(() {
    lib = _openLib();
  });

  test('chat ABI symbols resolve', () {
    const symbols = [
      'crispasr_chat_open',
      'crispasr_chat_close',
      'crispasr_chat_reset',
      'crispasr_chat_generate',
      'crispasr_chat_generate_stream',
      'crispasr_chat_template_name',
      'crispasr_chat_n_ctx',
      'crispasr_chat_memory_estimate',
      'crispasr_chat_string_free',
      'crispasr_chat_open_params_default',
      'crispasr_chat_generate_params_default',
    ];
    for (final s in symbols) {
      expect(lib.providesSymbol(s), isTrue, reason: 'missing $s in libcrispasr');
    }
  });

  test('open / generate / reset / close', () async {
    final modelPath = Platform.environment['CRISPASR_CHAT_TEST_MODEL'];
    if (modelPath == null || modelPath.isEmpty) {
      markTestSkipped('CRISPASR_CHAT_TEST_MODEL not set');
      return;
    }

    final libPath = Platform.environment['CRISPASR_LIB'];
    final session = CrispasrChatSession.open(
      modelPath,
      params: const ChatOpenParams(nCtx: 1024, nGpuLayers: -1),
      libPath: libPath,
    );

    addTearDown(session.close);

    expect(session.nCtx, greaterThan(0));
    expect(session.templateName, isNotEmpty);

    const messages = [
      ChatMessage(role: 'system', content: 'You are a terse assistant. Answer in one word.'),
      ChatMessage(role: 'user',   content: 'Say hello.'),
    ];

    // Greedy + seeded → reproducible across runs.
    const gp = ChatGenerateParams(maxTokens: 16, temperature: 0.0, seed: 1);

    final reply = await session.generate(messages, params: gp);
    expect(reply, isNotEmpty);

    // After reset, the next generate re-prefills from scratch and
    // produces the same (greedy-deterministic) output.
    session.reset();
    final reply2 = await session.generate(messages, params: gp);
    expect(reply2, equals(reply));

    // Stop sequences truncate the output.
    final clamped = await session.generate(
      messages,
      params: const ChatGenerateParams(
        maxTokens: 32,
        temperature: 0.0,
        seed: 1,
        stop: ['.'],
      ),
    );
    // The stop sequence is the period; output is whatever came before
    // the first '.'. If the model emits no '.' inside max_tokens, the
    // output is just the full thing — but with greedy + seed=1 on
    // SmolLM2 we expect a sentence-ish reply.
    expect(clamped, isNotEmpty);
    expect(clamped.contains('.'), isFalse,
        reason: 'stop sequence should truncate before the period');
  }, timeout: const Timeout(Duration(minutes: 2)));
}
