"""Literal native grouping/window/repair + production result getters + Python.

Decoded tensors are tiny controlled fixtures, not model-quality evidence.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from crispasr import _binding as binding
from test_session_utf8 import definition


BASELINE_SHA256 = '41ff8c7749210e1349df150de1fef1fe77ad2f18e1114d7536da90137cf550ac'


class WindowMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name)
        api = (ROOT / 'src/crispasr_c_api.cpp').read_text()
        native = (ROOT / 'src/parakeet.cpp').read_text()
        shared = (ROOT / 'src/parakeet_orchestrate.cpp').read_text()
        code = '''
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "parakeet_orchestrate.h"
#include "crispasr_vad.h"
#define CA_EXPORT extern "C"
'''
        if (ROOT / 'src/core/parakeet_word_ranges.h').exists():
            code += '#include "core/parakeet_word_ranges.h"\n'
        code += definition(native, 'static bool parakeet_group_words(') + '\n'
        baseline = ROOT / 'tests/fixtures/parakeet-group-08aea3537.inc'
        actual_hash = hashlib.sha256(baseline.read_bytes()).hexdigest()
        if actual_hash != BASELINE_SHA256:
            raise RuntimeError('Immutable native grouping reference changed')
        code += baseline.read_text() + '\n'
        for marker in ('struct crispasr_session_seg {', 'struct crispasr_session_result {'):
            code += definition(api, marker) + ';\n'
        code += '''
struct crispasr_session { parakeet_context* parakeet_ctx = nullptr; };
static int scenario = 0, decodes = 0;
static void session_report_progress(crispasr_session*, int, int) {}
static void add(parakeet_result* r, const char* text, int t0, int t1, float p) {
    auto& token = r->tokens[r->n_tokens++];
    strncpy(token.text, text, sizeof(token.text)-1);
    token.t0=t0; token.t1=t1; token.p=p; token.id=r->n_tokens;
}
extern "C" void parakeet_result_free(parakeet_result* r) {
    if (!r) return; free(r->tokens); free(r->words); free(r->text); delete r;
}
extern "C" parakeet_result* parakeet_transcribe_ex(parakeet_context*, const float*, int, int64_t offset) {
    ++decodes;
    if ((scenario==2 || scenario==4) && decodes>1) return nullptr;
    if (scenario==1 && offset%400!=0 && offset%400!=68) return nullptr;
    auto* r = new parakeet_result{}; r->tokens = (parakeet_token_data*)calloc(16,sizeof(parakeet_token_data));
    if (scenario==4) {
        r->n_words=2; r->words=(parakeet_word_data*)calloc(2,sizeof(parakeet_word_data));
        strcpy(r->words[0].text,"known"); r->words[0].t0=40; r->words[0].t1=80; r->words[0].p=.25f;
        strcpy(r->words[1].text,"words"); r->words[1].t0=100; r->words[1].t1=120; r->words[1].p=.5f;
        r->text=strdup("known words"); return r;
    }
    if (scenario==0 || scenario==3) {
        if (scenario==3) { add(r,"",0,0,.01f); add(r," ",0,0,.02f); }
        if (offset==0) {
            add(r," \\xe4",40,40,.1f); add(r,"\\xb8\\x96",40,48,.2f);
            add(r," beta",200,220,.3f);
            if(scenario==3) add(r,"",200,200,.05f);
            add(r," seam",350,370,.4f);
        } else { add(r," beta",205,225,.5f); add(r," seam",350,370,.6f); add(r," tail",450,470,.7f); if(scenario==3) add(r," ",470,470,.09f); }
    } else if (scenario==1) {
        int base=(int)(offset/400)*400;
        if (offset%400==0) { add(r,"猫",base+50,base+80,.2f); add(r,"。",base+80,base+88,.3f); add(r,"犬",base+300,base+330,.4f); }
        else { add(r,"猫",base+70,base+90,.5f); add(r,"新",base+160,base+180,.6f); add(r,"物",base+160,base+180,.7f); add(r,"犬",base+300,base+330,.8f); }
    } else {
        add(r," old",0,20,.2f); add(r," Alpha",100,120,.3f); add(r," Beta",100,120,.4f); add(r," end",400,420,.5f);
    }
    parakeet_group_words(r,8); r->text=strdup("native fixture"); return r;
}
std::vector<crispasr_audio_slice> crispasr_energy_chunk_slices(const float*,int n,int sr,int,float) {
    std::vector<crispasr_audio_slice> out;
    for(int start=0;start<n;start+=4*sr) { crispasr_audio_slice s{}; s.start=start; s.end=std::min(n,start+4*sr);
        s.t0_cs=start/sr*100; s.t1_cs=s.end/sr*100; out.push_back(s); }
    return out;
}
'''
        for marker in ('struct parakeet_session_owned_word :',
                       'static std::vector<parakeet_session_owned_word> parakeet_session_copy_words('):
            if marker in api:
                code += definition(api, marker) + (';\n' if marker.startswith('struct') else '\n')
        code += definition(api, 'static std::string parakeet_norm_word(') + '\n'
        code += definition(api, 'static void parakeet_session_chunked_merge(') + '\n'
        code += definition(shared, 'parakeet_seg result_to_seg(') + '\n'
        code += definition(shared, 'std::string norm_word(') + '\n'
        code += definition(shared, 'int gap_fill_segments(') + '\n'
        code += '''
extern "C" const char* standalone_boundary_words(int kind) {
    const std::vector<std::vector<const char*>> fixtures = {
        {" on", " the", " ", "3090", ",", " because"},
        {" but", " we", " could", " ", "util", "ize", ".", " that"},
        {" ", " on", " the", " ", "", " ", "3090", ",", " because", " "},
        {" a", " b", " ", ".", "c", " d"},
        {" ", ".", "a", " b", " c"},
    };
    parakeet_result r{};
    r.tokens=(parakeet_token_data*)calloc(32,sizeof(parakeet_token_data));
    for(auto text:fixtures.at(kind)) add(&r,text,r.n_tokens*8,r.n_tokens*8+8,.5f);
    parakeet_group_words(&r,8);
    auto seg=result_to_seg(&r,0);
    static std::string text;
    text.clear();
    for(const auto& word:seg.words) { if(!text.empty()) text += "|"; text+=word.text; }
    free(r.words); free(r.tokens);
    return text.c_str();
}
'''
        code += '''
extern "C" crispasr_session_result* inject(int kind) {
    scenario=kind; decodes=0;
    std::vector<float> audio(8*16000);
    auto* r=new crispasr_session_result;
    crispasr_session session; auto* s=&session;
    if(kind==0 || kind==3 || kind==4) { parakeet_session_chunked_merge(nullptr,audio.data(),5*16000,4*16000,2*16000,r->segments,s); return r; }
    if(kind==2) {
        parakeet_seg seg; seg.text="old end"; seg.words={{"old",0,20,.2f},{"end",400,420,.5f}};
        seg.tokens={{" old",0,0,20,.2f},{" end",1,400,420,.5f}};
        std::vector<parakeet_seg> segs{seg}; gap_fill_segments(nullptr,audio.data(),5*16000,0,segs,10,100,true);
        for(auto& ps:segs) { crispasr_session_seg out; out.text=ps.text; out.t0=ps.t0; out.t1=ps.t1;
            for(auto& w:ps.words) out.words.push_back({w.text,w.t0,w.t1,w.p,{}});
            for(auto& t:ps.tokens) out.tokens.push_back({t.text,t.t0,t.t1,t.p,{}});
            r->segments.push_back(std::move(out)); }
        return r;
    }
'''
        code += definition(api, 'auto transcribe_ja_sliced =') + ';\n'
        code += 'r->segments.push_back(transcribe_ja_sliced(audio.data(),8*16000,16000,4)); return r; }\n'
        code += '''
extern "C" int compare_grouping() {
    std::mt19937 rng(549);
    const char* texts[]={"", " ", " a", "b", " c", ".", "猫", "。", "!", " \\xe4", "\\xb8\\x96"};
    for(int c=0;c<2000;++c) {
        parakeet_result a{},b{}; std::vector<parakeet_token_data> tokens(rng()%40);
        for(auto& t:tokens) { strcpy(t.text,texts[rng()%11]); t.t0=(rng()%6)*8; t.t1=t.t0+8;
            t.p=(rng()%100)/100.f; }
        // Historical grouping discarded interior standalone boundaries (#558).
        // Keep its immutable oracle for the unchanged domain; the explicit
        // standalone-boundary fixtures below assert the corrected behavior.
        for(size_t i=1;i+1<tokens.size();++i)
            if(strcmp(tokens[i].text," ")==0) tokens[i].text[0]='\\0';
        a.tokens=b.tokens=tokens.data(); a.n_tokens=b.n_tokens=(int)tokens.size();
        parakeet_group_words(&a,8); parakeet_group_words_reference(&b,8);
        bool equal=a.n_words==b.n_words;
        for(int i=0;equal && i<a.n_words;++i) equal=strcmp(a.words[i].text,b.words[i].text)==0 &&
            a.words[i].t0==b.words[i].t0 && a.words[i].t1==b.words[i].t1 &&
            memcmp(&a.words[i].p,&b.words[i].p,sizeof(float))==0;
        free(a.words); free(b.words); if(!equal) return c+1;
    }
    return 0;
}
'''
        for name in ('n_segments', 'segment_text', 'segment_t0', 'segment_t1', 'segment_speaker',
                     'segment_no_speech_prob', 'n_words', 'word_text', 'word_t0', 'word_t1', 'word_p',
                     'n_tokens', 'token_text', 'token_t0', 'token_t1', 'token_p',
                     'word_n_alts', 'word_alt_text', 'word_alt_p', 'free'):
            marker = re.search(r'^CA_EXPORT [^\n]*\bcrispasr_session_result_' + name + r'\(', api, re.MULTILINE).group()
            code += definition(api, marker) + '\n'
        (path / 'probe.cpp').write_text(code)
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-shared', '-fPIC',
                        '-I', str(ROOT / 'src'), str(path / 'probe.cpp'), '-o', str(path / 'probe.so')], check=True)
        cls.lib = ctypes.CDLL(str(path / 'probe.so'))
        cls.lib.inject.argtypes, cls.lib.inject.restype = [ctypes.c_int], ctypes.c_void_p
        cls.lib.compare_grouping.restype = ctypes.c_int
        cls.lib.standalone_boundary_words.argtypes = [ctypes.c_int]
        cls.lib.standalone_boundary_words.restype = ctypes.c_char_p
        cls.result_lib = ctypes.CDLL(os.environ['CRISPASR_LIB']) if os.environ.get('CRISPASR_LIB') else cls.lib
        cls.lib.crispasr_session_result_free.argtypes = [ctypes.c_void_p]
        cls.session = binding.Session.__new__(binding.Session)
        cls.session._lib = cls.result_lib
        cls.session._handle = None
        # Actual exported getters when CRISPASR_LIB is supplied; decoded fixtures
        # use the same opaque result layout. No model/session open is needed.
        for name, count, restype in [('n_segments', 0, ctypes.c_int), ('n_words', 1, ctypes.c_int),
                ('n_tokens', 1, ctypes.c_int), ('segment_text', 1, ctypes.c_char_p),
                ('segment_speaker', 1, ctypes.c_char_p), ('segment_no_speech_prob', 1, ctypes.c_float)]:
            fn = getattr(cls.result_lib, 'crispasr_session_result_' + name)
            fn.argtypes, fn.restype = [ctypes.c_void_p] + [ctypes.c_int] * count, restype
        for kind, count in [('segment', 1), ('word', 2), ('token', 2)]:
            for field, restype in [('t0', ctypes.c_int64), ('t1', ctypes.c_int64), ('text', ctypes.c_char_p), ('p', ctypes.c_float)]:
                if kind == 'segment' and field == 'p': continue
                fn = getattr(cls.result_lib, 'crispasr_session_result_' + kind + '_' + field)
                fn.argtypes, fn.restype = [ctypes.c_void_p] + [ctypes.c_int] * count, restype
        # 0.5.13 per-word alts getters: register like the binding's own
        # signature setup does. Without argtypes, ctypes converts the fixture
        # handle (a plain int) to c_int — silently truncating the pointer — and
        # the reads segfault. Caught by test_empty_and_bos… segfaulting on CI.
        if hasattr(cls.result_lib, 'crispasr_session_result_word_n_alts'):
            cls.result_lib.crispasr_session_result_word_n_alts.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            cls.result_lib.crispasr_session_result_word_n_alts.restype = ctypes.c_int
            cls.result_lib.crispasr_session_result_word_alt_text.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            cls.result_lib.crispasr_session_result_word_alt_text.restype = ctypes.c_char_p
            cls.result_lib.crispasr_session_result_word_alt_p.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            cls.result_lib.crispasr_session_result_word_alt_p.restype = ctypes.c_float

        # _read_result also discovers alternatives when present in a full library.
        # Leaving these untyped truncates the opaque result pointer to C int.
        # Export them in the tiny fixture too, so local runs exercise this path.
        for name, count, restype in [('word_n_alts', 2, ctypes.c_int),
                                     ('word_alt_text', 3, ctypes.c_char_p),
                                     ('word_alt_p', 3, ctypes.c_float)]:
            fn = getattr(cls.result_lib, 'crispasr_session_result_' + name)
            fn.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * count
            fn.restype = restype

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def read(self, scenario):
        with patch.dict(os.environ, {'CRISPASR_GAP_FILL': '1', 'CRISPASR_GAP_FILL_MIN_CS': '100'}):
            handle = self.lib.inject(scenario)
        try:
            return self.session._read_result(handle)[0]
        finally:
            self.lib.crispasr_session_result_free(handle)

    def test_grouping_matches_frozen_native_without_interior_standalone_spaces(self):
        self.assertEqual(self.lib.compare_grouping(), 0)

    def test_standalone_word_boundary_survives_grouping(self):
        for kind, expected in enumerate([
                'on|the|3090,|because', 'but|we|could|utilize.|that',
                'on|the|3090,|because', 'a|b.|c|d', '.a|b|c']):
            with self.subTest(kind=kind):
                self.assertEqual(self.lib.standalone_boundary_words(kind).decode(), expected)

    def test_overlap_splice_and_dedup_keep_owner_bytes_and_probabilities(self):
        seg = self.read(0)
        self.assertEqual([w.text for w in seg.words], ['世', 'beta', 'seam', 'tail'])
        self.assertEqual([t.raw_text for t in seg.tokens], [b' \xe4', b'\xb8\x96', b' beta', b' seam', b' tail'])
        self.assertEqual([t.confidence for t in seg.tokens], np.array([.1, .2, .3, .6, .7], dtype=np.float32).tolist())
        self.assertEqual([t.start for t in seg.tokens], [.4, .4, 2., 3.5, 4.5])

    def test_empty_and_bos_tokens_follow_the_selected_owner(self):
        seg = self.read(3)
        self.assertEqual([w.text for w in seg.words], ['世', 'beta', 'seam', 'tail'])
        self.assertEqual([t.raw_text for t in seg.tokens],
                         [b'', b' ', b' \xe4', b'\xb8\x96', b' beta', b'', b' seam', b' tail', b' '])
        self.assertEqual([t.confidence for t in seg.tokens],
                         np.array([.01, .02, .1, .2, .3, .05, .6, .7, .09], dtype=np.float32).tolist())

    def test_japanese_two_slices_gap_fill_keeps_only_recovered_word_tokens(self):
        seg = self.read(1)
        self.assertEqual([w.text for w in seg.words], ['猫。', '新', '物', '犬'] * 2)
        self.assertEqual([t.raw_text for t in seg.tokens], [t.encode() for t in ['猫', '。', '新', '物', '犬'] * 2])
        self.assertEqual([t.confidence for t in seg.tokens], np.array([.2, .3, .6, .7, .4] * 2, dtype=np.float32).tolist())
        self.assertEqual([t.start for t in seg.tokens], [.5, .8, 1.6, 1.6, 3., 4.5, 4.8, 5.6, 5.6, 7.])

    def test_tokenless_native_words_are_kept_without_fabricated_tokens(self):
        seg = self.read(4)
        self.assertEqual([w.text for w in seg.words], ['known', 'words'])
        self.assertEqual([w.confidence for w in seg.words], [.25, .5])
        self.assertEqual(seg.tokens, [])

    def test_shared_repair_equal_frame_words_have_no_duplicate_tokens(self):
        seg = self.read(2)
        self.assertEqual([w.text for w in seg.words], ['old', 'Alpha', 'Beta', 'end'])
        self.assertEqual([t.raw_text for t in seg.tokens], [b' old', b' Alpha', b' Beta', b' end'])
        self.assertEqual([t.confidence for t in seg.tokens], np.array([.2, .3, .4, .5], dtype=np.float32).tolist())


if __name__ == '__main__':
    unittest.main()
