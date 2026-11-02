"""#549: decoded native fixtures through production packaging, C ABI and Python.

Compiles small literal production functions, substituting decoded model output
only. This tests metadata preservation, not model accuracy. Set CRISPASR_LIB to
also exercise exported getters/signature setup in a complete shared library.
"""
import ctypes
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import crispasr
from crispasr import _binding as binding
from test_session_utf8 import definition

ROOT = Path(__file__).resolve().parents[1]


class SessionMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / 'src/crispasr_c_api.cpp').read_text()
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name)
        code = '''
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "core/ngram_loop_fix.h"
#include "core/asr_segment_group.h"
#include "core/moss_diarize_segments.h"
#include "cohere.h"
#include "crispasr_vad.h"
#include "parakeet_orchestrate.h"
#include "core/parakeet_word_ranges.h"
#define CA_EXPORT extern "C"
'''
        for marker in ('struct crispasr_session_seg {', 'struct crispasr_session_result {'):
            code += definition(source, marker) + ';\n'
        code += 'using ca_token_record = crispasr_session_seg::token;\n'
        code += definition((ROOT / 'src/parakeet.cpp').read_text(), 'static bool parakeet_group_words(') + '\n'
        for marker in ('static std::vector<crispasr_session_seg::word> emit_words_from_tokens(',
                       'static void filter_words_by_ngram_collapse(',
                       'static void filter_tokens_by_ngram_collapse('):
            code += definition(source, marker) + '\n'
        code += '''
static int cache_frees = 0;
static void crispasr_aligner_free_cache() { ++cache_frees; }
extern "C" int cache_free_count() { return cache_frees; }
struct crispasr_session {
    std::string backend = "cohere", source_language, ask;
    cohere_context* cohere_ctx = reinterpret_cast<cohere_context*>(1);
    moss_diarize_context* moss_diarize_ctx = reinterpret_cast<moss_diarize_context*>(1);
    parakeet_context* parakeet_ctx = nullptr;
    int max_new_tokens = 512, beam_size = 1;
    float frequency_penalty = 0, temperature = 0, top_p = -1.0f;
    int do_sample = -1, top_k = -1;
    uint64_t seed = 0;
};
static void _fire_token_callbacks(crispasr_session*, const std::vector<ca_token_record>&) {}
extern "C" void cohere_set_max_new_tokens(cohere_context*, int) {}
extern "C" void cohere_set_frequency_penalty(cohere_context*, float) {}
extern "C" void cohere_set_beam_size(cohere_context*, int) {}
static std::string transcript;
static std::vector<cohere_token_data> decoded_tokens;
extern "C" cohere_result* cohere_transcribe_ex(cohere_context*, const float*, int, const char*, int64_t) {
    static cohere_result r;
    r.text = const_cast<char*>(transcript.c_str()); r.tokens = decoded_tokens.data();
    r.n_tokens = (int)decoded_tokens.size(); return &r;
}
extern "C" void cohere_result_free(cohere_result*) {}
extern "C" void moss_diarize_set_ask(moss_diarize_context*, const char*) {}
extern "C" void moss_diarize_set_max_new_tokens(moss_diarize_context*, int) {}
extern "C" void moss_diarize_set_beam_size(moss_diarize_context*, int) {}
extern "C" void moss_diarize_set_sampling(moss_diarize_context*, float, int, float, uint64_t) {}
extern "C" char* moss_diarize_transcribe(moss_diarize_context*, const float*, int) { return strdup(transcript.c_str()); }
static crispasr_session_result* cohere_package() {
    crispasr_session session; auto* s = &session;
    auto* r = new crispasr_session_result;
    bool lang_set = false; std::string lang;
    const float* pcm = nullptr; int n_samples = 16000;
'''
        code += definition(source, 'if (s->backend == "cohere" && s->cohere_ctx)')
        code += '\nreturn r; }\n'
        code += '''
static crispasr_session_result* moss_package() {
    crispasr_session session; auto* s = &session;
    auto* r = new crispasr_session_result;
    char* text = nullptr; bool need_free = false;
    const float* pcm = nullptr; int n_samples = 16000;
'''
        code += definition(source, 'if (!text && s->moss_diarize_ctx)')
        code += '''
    if (text) { crispasr_session_seg seg; seg.text = text; r->segments.push_back(seg); }
    if (need_free) free(text);
    return r;
}
extern "C" crispasr_session_result* inject_cohere(const char* text) {
    transcript = text; decoded_tokens.clear();
    int t = 10;
    for (const auto& piece : std::vector<std::string>{" Hello", ",", " world", "!"}) {
        cohere_token_data token = {}; strcpy(token.text, piece.c_str());
        token.t0 = t; token.t1 = t + 5; token.p = .75f; t += 5;
        decoded_tokens.push_back(token);
    }
    return cohere_package();
}
extern "C" crispasr_session_result* inject_moss(const char* text) {
    transcript = text; return moss_package();
}
extern "C" crispasr_session_result* inject_untimed() {
    auto* r = new crispasr_session_result; crispasr_session_seg seg;
    seg.text = "世"; seg.t0 = 100; seg.t1 = 200; seg.speaker = "(Speaker 1) ";
    std::vector<ca_token_record> toks = {{"\\xe4", -1, -1, .5f, {}}, {"\\xb8\\x96", -1, -1, .75f, {}}};
    seg.words = emit_words_from_tokens(toks); seg.tokens = std::move(toks);
    r->segments.push_back(seg); return r;
}
'''
        # The real orchestrator -> session adapter, not a duplicate conversion.
        code += '''
static std::vector<parakeet_seg> parakeet_transcribe_segments(parakeet_context*, const float*, int, int, bool, int) {
    parakeet_seg seg; seg.text = "Hello"; seg.t0 = 50; seg.t1 = 75;
    seg.words.push_back({"Hello", 50, 75, .625f});
    seg.tokens.push_back({" Hel", 1, 50, 60, .5f});
    seg.tokens.push_back({"lo", 2, 60, 75, .75f}); return {seg};
}
extern "C" crispasr_session_result* inject_parakeet() {
    crispasr_session session; auto* s = &session; auto* r = new crispasr_session_result;
    const float* pcm = nullptr; int n_samples = 16000; bool is_ja = false; int oo = 0;
'''
        code += definition(source, 'for (auto& ps : parakeet_transcribe_segments(')
        code += '\nreturn r; }\n'
        orchestrate = (ROOT / 'src/parakeet_orchestrate.cpp').read_text()
        code += 'namespace cli_split {\n'
        code += definition(orchestrate, 'parakeet_seg result_to_seg(') + '\n'
        code += definition(orchestrate, 'std::vector<parakeet_seg> split_result_to_segs(') + '\n}\n'
        code += definition(source, 'static void parakeet_result_to_session_segs(') + '\n'
        code += r'''
extern "C" crispasr_session_result* inject_split(int fixture, int seg_seconds, bool shared) {
    parakeet_word_data words[] = {
        {"Hello", 10, 20, .5f}, {"世界", 110, 115, .75f},
        {"again", 110, 130, .625f}, {"last", 240, 260, .875f}};
    parakeet_token_data tokens[] = {
        {1, " Hel", 10, 15, .5f}, {2, "lo", 15, 20, .75f},
        {3, " 世界", 110, 115, .625f}, {4, " again", 110, 130, .875f},
        {5, " last", 240, 260, .5f}, {6, "!", INT64_MAX, INT64_MAX, .25f}};
    char text[] = "Hello 世界 again last!";
    char empty[] = "";
    parakeet_result native = {text, tokens, 6, words, 4};
    if (fixture == 1) { native.words = nullptr; native.n_words = 0; }
    if (fixture == 2) { native.text = empty; native.words = nullptr; native.n_words = 0; }
    if (fixture == 3) { native.text = empty; native.words = nullptr; native.n_words = 0;
                        native.tokens = nullptr; native.n_tokens = 0; }
    if (fixture == 4) { native.tokens = nullptr; native.n_tokens = 0; }
    if (fixture == 6) {
        tokens[1].t0 = 110; tokens[1].t1 = 118;
        native.n_tokens = 4; native.words = nullptr; native.n_words = 0;
        parakeet_group_words(&native, 8);
    }
    struct WordOwner { parakeet_result* r; bool own; ~WordOwner() { if (own) free(r->words); } } owner{&native,fixture==6};
    auto* pr = fixture == 5 ? nullptr : &native;
    auto* r = new crispasr_session_result;
    if (!shared) {
        parakeet_result_to_session_segs(pr, seg_seconds, r->segments);
        return r;
    }
'''
        adapter = definition(source, 'for (auto& ps : parakeet_transcribe_segments(')
        adapter = adapter.replace('parakeet_transcribe_segments(s->parakeet_ctx, pcm, n_samples, 0, is_ja, oo)',
                                  'cli_split::split_result_to_segs(pr, 0, seg_seconds)')
        code += adapter + '\nreturn r; }\n'
        code += r'''
extern "C" crispasr_session_result* inject_shifted() {
    auto* part = inject_untimed(); auto* merged = new crispasr_session_result;
    part->segments[0].tokens.push_back({" timed", 110, 120, .25f, {}});
    const int64_t off_cs = 300;
'''
        code += definition(source, 'for (auto& seg : part->segments)')
        code += '\ndelete part; return merged; }\n'
        vad_source = (ROOT / 'src/crispasr_vad.cpp').read_text()
        code += definition(vad_source, 'int64_t crispasr_vad_remap_timestamp(') + '\n'
        code += r'''
extern "C" crispasr_session_result* inject_remapped() {
    auto* r = inject_untimed();
    r->segments[0].tokens.push_back({" timed", 110, 120, .25f, {}});
    crispasr_stitched_audio stitched;
    stitched.mapping = {{0, 300}, {200, 500}};
'''
        vad_start = source.index('CA_EXPORT crispasr_session_result* crispasr_session_transcribe_vad_lang(')
        code += definition(source[vad_start:], 'for (auto& seg : r->segments)')
        code += '\nreturn r; }\n'
        for name in ('n_segments', 'segment_text', 'segment_t0', 'segment_t1', 'segment_speaker',
                     'segment_no_speech_prob', 'n_words', 'word_text', 'word_t0', 'word_t1', 'word_p', 'word_n_alts', 'word_alt_text', 'word_alt_p',
                     'n_tokens', 'token_text', 'token_t0', 'token_t1', 'token_p', 'free'):
            name = 'crispasr_session_result_' + name
            marker = re.search(r'^CA_EXPORT [^\n]*\b' + name + r'\(', source, re.MULTILINE).group()
            code += definition(source, marker) + '\n'
        code += 'extern "C" void inject_word_alt(crispasr_session_result* r) { if (!r->segments.empty() && !r->segments[0].words.empty()) r->segments[0].words[0].alts.push_back({"alternative 日本語", 0.25f}); }\n'
        code += definition(source, 'CA_EXPORT void crispasr_free_aligner_cache(') + '\n'
        (path / 'test.cpp').write_text(code)
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-shared', '-fPIC',
                        '-I', str(ROOT / 'src'), str(path / 'test.cpp'), '-o', str(path / 'test.so')], check=True)
        cls.fixture = ctypes.CDLL(str(path / 'test.so'))
        for name, args in [('cohere', [ctypes.c_char_p]), ('moss', [ctypes.c_char_p]),
                           ('untimed', []), ('parakeet', []), ('shifted', []), ('remapped', []),
                           ('split', [ctypes.c_int, ctypes.c_int, ctypes.c_bool])]:
            fn = getattr(cls.fixture, 'inject_' + name)
            fn.argtypes = args
            fn.restype = ctypes.c_void_p
        cls.lib = ctypes.CDLL(os.environ['CRISPASR_LIB']) if os.environ.get('CRISPASR_LIB') else cls.fixture
        cls.session = binding.Session.__new__(binding.Session)
        cls.session._lib = cls.lib
        cls.session._handle = None
        if os.environ.get('CRISPASR_LIB'):
            cls.session._setup_session_signatures()
        else:
            for name, count, restype in [('n_segments', 0, ctypes.c_int), ('n_words', 1, ctypes.c_int),
                                         ('n_tokens', 1, ctypes.c_int), ('segment_text', 1, ctypes.c_char_p),
                                         ('segment_speaker', 1, ctypes.c_char_p),
                                         ('segment_no_speech_prob', 1, ctypes.c_float)]:
                fn = getattr(cls.lib, 'crispasr_session_result_' + name)
                fn.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * count
                fn.restype = restype
            for kind, count in [('segment', 1), ('word', 2), ('token', 2)]:
                for field, restype in [('t0', ctypes.c_int64), ('t1', ctypes.c_int64),
                                      ('text', ctypes.c_char_p), ('p', ctypes.c_float)]:
                    if kind == 'segment' and field == 'p':
                        continue
                    fn = getattr(cls.lib, 'crispasr_session_result_' + kind + '_' + field)
                    fn.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * count
                    fn.restype = restype
        cls.fixture.inject_word_alt.argtypes = [ctypes.c_void_p]
        for name, args, restype in [('word_n_alts', 2, ctypes.c_int),
                                    ('word_alt_text', 3, ctypes.c_char_p),
                                    ('word_alt_p', 3, ctypes.c_float)]:
            fn = getattr(cls.lib, 'crispasr_session_result_' + name)
            fn.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * args
            fn.restype = restype
        cls.fixture.crispasr_session_result_free.argtypes = [ctypes.c_void_p]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def read(self, handle):
        try:
            return self.session._read_result(handle)
        finally:
            self.fixture.crispasr_session_result_free(handle)

    def test_word_alternatives_copied_before_native_free(self):
        handle = self.fixture.inject_cohere(b'Hello world')
        self.fixture.inject_word_alt(handle)
        segments = self.read(handle)
        self.assertEqual(segments[0].words[0].alternatives, [('alternative 日本語', 0.25)])

    def test_cohere_native_token_metadata_survives(self):
        seg, = self.read(self.fixture.inject_cohere(b'Hello, world!'))
        self.assertEqual([t.raw_text for t in seg.tokens], [b' Hello', b',', b' world', b'!'])
        self.assertEqual([t.start for t in seg.tokens], [.1, .15, .2, .25])
        self.assertEqual([t.end for t in seg.tokens], [.15, .2, .25, .3])
        self.assertTrue(all(t.confidence == .75 for t in seg.tokens))
        self.assertEqual([w.text for w in seg.words], ['Hello,', 'world!'])

    def test_parakeet_shared_orchestrator_tokens_survive(self):
        seg, = self.read(self.fixture.inject_parakeet())
        self.assertEqual([(t.text, t.start, t.end, t.confidence) for t in seg.tokens],
                         [(' Hel', .5, .6, .5), ('lo', .6, .75, .75)])

    def test_legacy_split_matches_shared_cli_for_all_metadata(self):
        for fixture in range(6):
            for seconds in (-1, 0, 1, 2, 20):
                with self.subTest(fixture=fixture, seconds=seconds):
                    expected = self.read(self.fixture.inject_split(fixture, seconds, True))
                    actual = self.read(self.fixture.inject_split(fixture, seconds, False))
                    self.assertEqual(actual, expected)
                    self.assertEqual(sum(len(s.tokens) for s in actual), 0 if fixture >= 3 else 6)

    def test_moss_native_segments_and_speakers(self):
        segs = self.read(self.fixture.inject_moss('[0.12][S01] Hello [0.87][1.0][S02] 世界[2.0]'.encode()))
        self.assertEqual([(s.text, s.start, s.end, s.speaker) for s in segs],
                         [('Hello', .12, .87, '(Speaker 1) '), ('世界', 1., 2., '(Speaker 2) ')])
        self.assertEqual(segs[0].words, [])
        self.assertEqual(segs[0].tokens, [])

    def test_moss_more_than_cli_fixed_capacity_and_plain_text_fallback(self):
        raw = '[0][S01] hello[1]' * 300
        self.assertEqual(len(self.read(self.fixture.inject_moss(raw.encode()))), 300)
        seg, = self.read(self.fixture.inject_moss(b'plain text'))
        self.assertEqual(seg.text, 'plain text')
        self.assertEqual(seg.speaker, '')

    def test_chunk_boundary_keeps_continuation_with_its_native_word(self):
        for shared in (False, True):
            with self.subTest(shared=shared):
                segs = self.read(self.fixture.inject_split(6, 1, shared))
                self.assertEqual(segs[0].text, 'Hello')
                self.assertEqual([t.raw_text for t in segs[0].tokens], [b' Hel', b'lo'])
                self.assertEqual([t.raw_text for s in segs for t in s.tokens],
                                 [b' Hel', b'lo', ' 世界'.encode(), b' again'])

    def test_missing_word_times_and_split_utf8_bytes(self):
        seg, = self.read(self.fixture.inject_untimed())
        self.assertEqual((seg.start, seg.end), (1., 2.))
        self.assertEqual(seg.words[0].text, '世')
        self.assertIsNone(seg.words[0].start)
        self.assertIsNone(seg.words[0].end)
        self.assertFalse(seg.words[0].has_timestamps)
        self.assertFalse(seg.tokens[0].has_timestamps)
        self.assertEqual(b''.join(t.raw_text for t in seg.tokens).decode(), '世')

    def test_chunk_and_vad_offsets_preserve_unavailable_times(self):
        for name in ('shifted', 'remapped'):
            seg, = self.read(getattr(self.fixture, 'inject_' + name)())
            self.assertEqual((seg.start, seg.end), (4., 5.))
            self.assertIsNone(seg.words[0].start)
            self.assertIsNone(seg.words[0].end)
            self.assertFalse(seg.tokens[0].has_timestamps)
            self.assertEqual((seg.tokens[-1].start, seg.tokens[-1].end), (4.1, 4.2))

    def test_all_python_transcription_entry_points_copy_metadata(self):
        owner = self

        class Frontend:
            freed = 0

            def __getattr__(self, name):
                if name in ('crispasr_session_transcribe', 'crispasr_session_transcribe_lang',
                            'crispasr_session_transcribe_chunked_lang',
                            'crispasr_session_transcribe_vad', 'crispasr_session_transcribe_vad_lang'):
                    return lambda *args: owner.fixture.inject_untimed()
                if name == 'crispasr_session_result_free':
                    def free(handle):
                        self.freed += 1
                        owner.fixture.crispasr_session_result_free(handle)
                    return free
                if name in ('crispasr_session_result_n_logit_frames', 'crispasr_session_result_n_logit_vocab'):
                    return lambda *args: 0
                if name == 'crispasr_session_result_logits':
                    return lambda *args: None
                if name == 'crispasr_session_set_return_logits':
                    raise AttributeError(name)
                return getattr(owner.lib, name)

        frontend = Frontend()
        pcm = np.zeros(16, dtype=np.float32)
        with patch.object(self.session, '_lib', frontend), patch.object(self.session, '_handle', 1):
            results = [self.session.transcribe(pcm, language='ja'),
                       self.session.transcribe_chunked(pcm),
                       self.session.transcribe_vad(pcm, 'fixture-vad.gguf'),
                       self.session.transcribe_with_logits(pcm)[0]]
        self.assertEqual(frontend.freed, 4)
        for segs in results:
            seg, = segs
            self.assertEqual(len(seg.tokens), 2)
            self.assertFalse(seg.words[0].has_timestamps)
            self.assertEqual(seg.speaker, '(Speaker 1) ')

    def test_token_getters_null_and_bounds(self):
        for handle in (None, self.fixture.inject_untimed()):
            try:
                for i in (-1, 5):
                    self.assertEqual(self.lib.crispasr_session_result_n_tokens(handle, i), 0)
                    self.assertEqual(self.lib.crispasr_session_result_token_text(handle, i, 0), b'')
                    self.assertEqual(self.lib.crispasr_session_result_token_t0(handle, i, 0), -1)
                    self.assertEqual(self.lib.crispasr_session_result_token_t1(handle, i, 0), -1)
                    self.assertEqual(self.lib.crispasr_session_result_token_p(handle, i, 0), -1)
                self.assertEqual(self.lib.crispasr_session_result_token_text(handle, 0, -1), b'')
                self.assertEqual(self.lib.crispasr_session_result_token_t0(handle, 0, 9), -1)
            finally:
                self.fixture.crispasr_session_result_free(handle)

    def test_old_library_and_positional_constructor_compatibility(self):
        class OldLibrary:
            def __getattr__(_, name):
                if name.endswith(('n_tokens', 'token_text', 'token_t0', 'token_t1', 'token_p')):
                    raise AttributeError(name)
                return getattr(self.lib, name)
        with patch.object(self.session, '_lib', OldLibrary()):
            seg, = self.read(self.fixture.inject_untimed())
        self.assertEqual(seg.tokens, [])
        old = crispasr.SessionSegment('hi', 0., 1., [], .1, 'speaker')
        self.assertEqual(old.speaker, 'speaker')
        self.assertEqual(old.tokens, [])
        old.tokens.append('independent')
        self.assertEqual(crispasr.SessionSegment('hi', 0., 1., []).tokens, [])

    def test_aligner_cache_release_session_module_and_older_library(self):
        # Use the literal wrapper + counter, avoiding any model allocation.
        with patch.object(self.session, '_lib', self.fixture):
            before = self.fixture.cache_free_count()
            self.session.free_aligner_cache()
            with patch.object(binding.ctypes, 'CDLL', return_value=self.fixture):
                crispasr.free_aligner_cache(lib_path='same-library')
            self.assertEqual(self.fixture.cache_free_count(), before + 2)
        with patch.object(self.session, '_lib', object()):
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                self.session.free_aligner_cache()


if __name__ == '__main__':
    unittest.main()
