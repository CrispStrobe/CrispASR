// chat.dart — Dart binding for the crispasr_chat_* C ABI (text → text LLM).
//
// Mirrors the session-handle shape used by CrispasrSession (transcribe)
// and Mic — opaque Pointer<Void> wrapped in a Dart class, NativeFinalizer
// for free-on-GC, plus an explicit `close()` for deterministic cleanup.
//
// Two generation modes:
//   • generate(messages)        — Future<String>, returns the whole reply
//   • generateStream(messages)  — Stream<String> of UTF-8 deltas (yields
//                                 from the same isolate; the FFI call
//                                 blocks until the model finishes, but
//                                 the stream surface lets consumer code
//                                 await chunks idiomatically)
//
// The header lives in include/crispasr_chat.h; see docs/prompts/chat-abi.md
// for the design rationale.

import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'crispasr.dart' show CrispASR;

// ---------------------------------------------------------------------------
// FFI shapes — kept private to this file so consumers stay on the
// high-level Dart classes below.
// ---------------------------------------------------------------------------

final class _ChatOpenParams extends Struct {
  @Int32() external int nThreads;
  @Int32() external int nThreadsBatch;
  @Int32() external int nCtx;
  @Int32() external int nBatch;
  @Int32() external int nUbatch;
  @Int32() external int nGpuLayers;
  @Bool()  external bool useMmap;
  @Bool()  external bool useMlock;
  @Bool()  external bool embeddings;
  external Pointer<Utf8> chatTemplate;
}

final class _ChatGenerateParams extends Struct {
  @Int32() external int maxTokens;
  @Float() external double temperature;
  @Int32() external int topK;
  @Float() external double topP;
  @Float() external double minP;
  @Float() external double repeatPenalty;
  @Int32() external int repeatLastN;
  @Uint32() external int seed;
  external Pointer<Pointer<Utf8>> stop;
  @Size()  external int nStop;
  @Bool()  external bool prefillOnly;
}

final class _ChatMessage extends Struct {
  external Pointer<Utf8> role;
  external Pointer<Utf8> content;
}

// Error struct: int32 code + 256-byte message buffer. Total 260 bytes.
// `Array<Int8>` was stabilised in Dart 3.1 (see pubspec.yaml SDK
// constraint); pre-3.1 hosts can't bind this ABI.
final class _ChatError extends Struct {
  @Int32() external int code;
  @Array.multi([256])
  external Array<Int8> message;
}

/// Parameters describing how a chat session is opened on top of a GGUF.
class ChatOpenParams {
  /// Generation threads. Defaults to physical-cores cap via the C ABI.
  final int? nThreads;
  /// Batch / prefill threads. Defaults to `nThreads`.
  final int? nThreadsBatch;
  /// Context window in tokens. `null` = model default.
  final int? nCtx;
  /// Logical batch size.
  final int? nBatch;
  /// Physical micro-batch.
  final int? nUbatch;
  /// `-1` = all layers on GPU (default), `0` = CPU only.
  final int? nGpuLayers;
  final bool useMmap;
  final bool useMlock;
  /// Override the chat template baked into the GGUF.
  /// `null` → read `tokenizer.chat_template`, falling back to `chatml`.
  final String? chatTemplate;

  const ChatOpenParams({
    this.nThreads,
    this.nThreadsBatch,
    this.nCtx,
    this.nBatch,
    this.nUbatch,
    this.nGpuLayers,
    this.useMmap = true,
    this.useMlock = false,
    this.chatTemplate,
  });
}

/// Per-call sampling configuration. Defaults match the C ABI's
/// `crispasr_chat_generate_params_default`.
class ChatGenerateParams {
  final int maxTokens;
  final double temperature;
  final int topK;
  final double topP;
  final double minP;
  final double repeatPenalty;
  final int repeatLastN;
  final int seed;
  /// Stop substrings — generation halts (output truncated before the match)
  /// the first time any of these appears in the accumulated decode.
  final List<String> stop;

  const ChatGenerateParams({
    this.maxTokens = 256,
    this.temperature = 0.8,
    this.topK = 40,
    this.topP = 0.95,
    this.minP = 0.05,
    this.repeatPenalty = 1.1,
    this.repeatLastN = 64,
    this.seed = 0,
    this.stop = const [],
  });
}

/// One message in a chat conversation. Same shape as the OpenAI
/// chat-completions schema; the chat-template layer translates
/// `role` into whatever the model's GGUF template expects.
class ChatMessage {
  final String role;
  final String content;
  const ChatMessage({required this.role, required this.content});

  factory ChatMessage.system(String content) => ChatMessage(role: 'system', content: content);
  factory ChatMessage.user(String content)   => ChatMessage(role: 'user',   content: content);
  factory ChatMessage.assistant(String c)    => ChatMessage(role: 'assistant', content: c);
}

class ChatException implements Exception {
  final int code;
  final String message;
  const ChatException(this.code, this.message);
  @override
  String toString() => 'ChatException($code: $message)';
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

class CrispasrChatSession {
  CrispasrChatSession._(this._lib, this._handle, this._templateName, this._nCtx) {
    _finalizer.attach(this, _handle.cast<Void>(), detach: this);
  }

  final DynamicLibrary _lib;
  Pointer<Void> _handle;
  final String _templateName;
  final int _nCtx;
  bool _closed = false;

  /// Name of the chat template the session resolved against
  /// (e.g. `chatml`, `llama3`, `gemma`).
  String get templateName => _templateName;

  /// Context window in tokens.
  int get nCtx => _nCtx;

  // Free-on-GC. We can't call _lib.lookupFunction inside the finalizer
  // (it's invoked by the GC, not Dart code), so we hand-roll a static
  // NativeFinalizer keyed on the crispasr_chat_close symbol address.
  static final Finalizer<Pointer<Void>> _finalizer = Finalizer<Pointer<Void>>((handle) {
    // Only fires for sessions whose owner was GC'd without close().
    // Look the symbol up lazily; we can't capture the DynamicLibrary
    // safely here, so re-open the default name. Matches Mic's pattern.
    try {
      final lib = DynamicLibrary.open(CrispASR.defaultLibName());
      final close = lib.lookupFunction<Void Function(Pointer<Void>), void Function(Pointer<Void>)>(
        'crispasr_chat_close',
      );
      close(handle);
    } catch (_) {
      // Library can't be re-opened in some teardown paths — leak in
      // that case; the OS will reclaim on process exit.
    }
  });

  /// Open a chat session from a GGUF chat model on disk.
  ///
  /// Throws [UnsupportedError] when the loaded dylib doesn't expose
  /// `crispasr_chat_open` (predates the chat ABI). Throws
  /// [ChatException] when the underlying load fails (missing file,
  /// unsupported architecture, …).
  factory CrispasrChatSession.open(
    String modelPath, {
    ChatOpenParams params = const ChatOpenParams(),
    String? libPath,
  }) {
    final lib = DynamicLibrary.open(libPath ?? CrispASR.defaultLibName());
    if (!lib.providesSymbol('crispasr_chat_open')) {
      throw UnsupportedError(
        'crispasr_chat_open not found in this libcrispasr — '
        'needs CrispASR 0.7.0+ with the chat ABI.',
      );
    }

    final defaults = lib.lookupFunction<
        Void Function(Pointer<_ChatOpenParams>),
        void Function(Pointer<_ChatOpenParams>)>('crispasr_chat_open_params_default');
    final open = lib.lookupFunction<
        Pointer<Void> Function(Pointer<Utf8>, Pointer<_ChatOpenParams>, Pointer<_ChatError>),
        Pointer<Void> Function(Pointer<Utf8>, Pointer<_ChatOpenParams>, Pointer<_ChatError>)>(
      'crispasr_chat_open',
    );
    final tmplName = lib.lookupFunction<
        Pointer<Utf8> Function(Pointer<Void>),
        Pointer<Utf8> Function(Pointer<Void>)>('crispasr_chat_template_name');
    final nCtxFn = lib.lookupFunction<
        Int32 Function(Pointer<Void>),
        int Function(Pointer<Void>)>('crispasr_chat_n_ctx');

    final paramsPtr = calloc<_ChatOpenParams>();
    defaults(paramsPtr);
    final pp = paramsPtr.ref;
    if (params.nThreads != null)       pp.nThreads = params.nThreads!;
    if (params.nThreadsBatch != null)  pp.nThreadsBatch = params.nThreadsBatch!;
    if (params.nCtx != null)           pp.nCtx = params.nCtx!;
    if (params.nBatch != null)         pp.nBatch = params.nBatch!;
    if (params.nUbatch != null)        pp.nUbatch = params.nUbatch!;
    if (params.nGpuLayers != null)     pp.nGpuLayers = params.nGpuLayers!;
    pp.useMmap = params.useMmap;
    pp.useMlock = params.useMlock;

    final tmplPtr = params.chatTemplate != null ? params.chatTemplate!.toNativeUtf8() : nullptr;
    pp.chatTemplate = tmplPtr.cast();

    final pathPtr = modelPath.toNativeUtf8();
    final errPtr  = calloc<_ChatError>();
    try {
      final handle = open(pathPtr, paramsPtr, errPtr);
      if (handle == nullptr) {
        throw ChatException(errPtr.ref.code, _readErrorMessage(errPtr));
      }
      final tp = tmplName(handle);
      final tmpl = tp == nullptr ? '' : tp.cast<Utf8>().toDartString();
      final ctx = nCtxFn(handle);
      return CrispasrChatSession._(lib, handle, tmpl, ctx);
    } finally {
      calloc.free(pathPtr);
      calloc.free(errPtr);
      calloc.free(paramsPtr);
      if (tmplPtr != nullptr) calloc.free(tmplPtr);
    }
  }

  /// Clear the KV cache so the next generate call re-prefills from
  /// scratch. Call when starting a new conversation in a reused session.
  void reset() {
    _ensureOpen();
    final reset = _lib.lookupFunction<
        Int32 Function(Pointer<Void>, Pointer<_ChatError>),
        int Function(Pointer<Void>, Pointer<_ChatError>)>('crispasr_chat_reset');
    final errPtr = calloc<_ChatError>();
    try {
      final rc = reset(_handle, errPtr);
      if (rc != 0) {
        throw ChatException(errPtr.ref.code, _readErrorMessage(errPtr));
      }
    } finally {
      calloc.free(errPtr);
    }
  }

  /// One-shot generate. Applies the chat template, prefills, runs to
  /// `maxTokens` or a stop sequence, returns the assistant reply.
  ///
  /// Blocks the calling isolate for the duration of generation — wrap
  /// in `Isolate.run` if the host app needs the UI isolate free.
  Future<String> generate(
    List<ChatMessage> messages, {
    ChatGenerateParams params = const ChatGenerateParams(),
  }) async {
    _ensureOpen();
    final stringFree = _lib.lookupFunction<Void Function(Pointer<Utf8>), void Function(Pointer<Utf8>)>(
      'crispasr_chat_string_free',
    );
    final generate = _lib.lookupFunction<
        Pointer<Utf8> Function(Pointer<Void>, Pointer<_ChatMessage>, Size, Pointer<_ChatGenerateParams>,
                               Pointer<_ChatError>),
        Pointer<Utf8> Function(Pointer<Void>, Pointer<_ChatMessage>, int, Pointer<_ChatGenerateParams>,
                               Pointer<_ChatError>)>('crispasr_chat_generate');

    final marshalled = _marshalMessages(messages);
    final paramsPtr = _marshalParams(params);
    final errPtr    = calloc<_ChatError>();
    try {
      final out = generate(_handle, marshalled.messagesPtr, messages.length, paramsPtr, errPtr);
      if (out == nullptr) {
        throw ChatException(errPtr.ref.code, _readErrorMessage(errPtr));
      }
      try {
        return out.toDartString();
      } finally {
        stringFree(out);
      }
    } finally {
      calloc.free(errPtr);
      _freeParams(paramsPtr, params);
      marshalled.dispose();
    }
  }

  // generateStream is intentionally NOT exposed on this binding.
  //
  // The chat C ABI passes `const char* utf8` to its on_token callback —
  // a pointer that is only valid for the duration of the synchronous
  // C-side call. NativeCallable.listener posts callbacks via SendPort
  // for asynchronous delivery on the owning isolate's event loop; by
  // the time the Dart closure runs, that pointer is already dangling
  // (the C++ std::string piece has gone out of scope, or worse — been
  // re-used by a later iteration of the generate loop).
  // NativeCallable.isolateLocal would deliver synchronously, but is
  // disallowed for closure callbacks in Dart's JIT-mode `dart test`
  // host, where the binding's smoke tests run.
  //
  // The recommended Dart streaming path is the HTTP endpoint exposed by
  // `crispasr --server --chat-model …` (POST /v1/chat/completions
  // with `stream: true`) — that emits SSE deltas a regular HTTP client
  // can subscribe to. The FFI path stays one-shot.

  /// Close the session and free its KV cache. Idempotent.
  void close() {
    if (_closed || _handle == nullptr) return;
    _closed = true;
    _finalizer.detach(this);
    final closeFn = _lib.lookupFunction<Void Function(Pointer<Void>), void Function(Pointer<Void>)>(
      'crispasr_chat_close',
    );
    closeFn(_handle);
    _handle = nullptr;
  }

  void _ensureOpen() {
    if (_closed || _handle == nullptr) {
      throw StateError('CrispasrChatSession is closed');
    }
  }

  // -------------------------------------------------------------------------
  // Internal marshalling helpers
  // -------------------------------------------------------------------------

  Pointer<_ChatGenerateParams> _marshalParams(ChatGenerateParams p) {
    final ptr = calloc<_ChatGenerateParams>();
    final ref = ptr.ref;
    ref.maxTokens     = p.maxTokens;
    ref.temperature   = p.temperature;
    ref.topK          = p.topK;
    ref.topP          = p.topP;
    ref.minP          = p.minP;
    ref.repeatPenalty = p.repeatPenalty;
    ref.repeatLastN   = p.repeatLastN;
    ref.seed          = p.seed;
    ref.prefillOnly   = false;
    if (p.stop.isEmpty) {
      ref.stop = nullptr;
      ref.nStop = 0;
    } else {
      final arr = calloc<Pointer<Utf8>>(p.stop.length);
      for (var i = 0; i < p.stop.length; i++) {
        arr[i] = p.stop[i].toNativeUtf8();
      }
      ref.stop = arr;
      ref.nStop = p.stop.length;
    }
    return ptr;
  }

  void _freeParams(Pointer<_ChatGenerateParams> ptr, ChatGenerateParams p) {
    if (p.stop.isNotEmpty) {
      final arr = ptr.ref.stop;
      for (var i = 0; i < p.stop.length; i++) {
        calloc.free(arr[i]);
      }
      calloc.free(arr);
    }
    calloc.free(ptr);
  }

  _MarshalledMessages _marshalMessages(List<ChatMessage> messages) {
    if (messages.isEmpty) {
      return _MarshalledMessages._(nullptr, const []);
    }
    final arr = calloc<_ChatMessage>(messages.length);
    final owned = <Pointer<Utf8>>[];
    for (var i = 0; i < messages.length; i++) {
      final r = messages[i].role.toNativeUtf8();
      final c = messages[i].content.toNativeUtf8();
      owned.add(r);
      owned.add(c);
      arr[i].role = r;
      arr[i].content = c;
    }
    return _MarshalledMessages._(arr, owned);
  }
}

class _MarshalledMessages {
  _MarshalledMessages._(this.messagesPtr, this._owned);
  final Pointer<_ChatMessage> messagesPtr;
  final List<Pointer<Utf8>> _owned;
  void dispose() {
    if (messagesPtr != nullptr) {
      calloc.free(messagesPtr);
    }
    for (final p in _owned) {
      calloc.free(p);
    }
  }
}

String _readErrorMessage(Pointer<_ChatError> errPtr) {
  // Read the inline char[256] until the first NUL.
  final buf = StringBuffer();
  final ref = errPtr.ref;
  for (var i = 0; i < 256; i++) {
    final c = ref.message[i];
    if (c == 0) break;
    buf.writeCharCode(c & 0xff);
  }
  return buf.toString();
}

