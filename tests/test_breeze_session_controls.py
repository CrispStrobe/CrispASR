#!/usr/bin/env python3
"""Compile actual session controls with recording stubs; no model numerics.

Replay old code with CRISPASR_CONTRACT_SOURCE and CRISPASR_BREEZE_SOURCE.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(source, signature):
    start = source.index(signature)
    opening = source.index('{', start)
    end, depth = opening + 1, 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


PREAMBLE = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#define CA_EXPORT extern "C"
#define CA_HAVE_BT2_TTS
struct breeze_tts_2_context {
    int max_new_tokens=17;
    struct { struct { unsigned s_max_new_tokens=999; } hp; } model;
};
struct crispasr_session {
    std::string backend="bt2-tts";
    breeze_tts_2_context* bt2_ctx=nullptr;
    int n_threads=4,max_new_tokens=0;
    float temperature=1;
    uint64_t seed=0;
};
struct params {
    int n_threads=0,verbosity=0;
    bool use_gpu=false;
    float temperature=1;
    uint64_t seed=0;
    const char* codec_path=nullptr;
};
static params received;
static breeze_tts_2_context context;
static float received_temperature=-1;
static uint64_t received_seed=0;
static int temperature_calls=0,seed_calls=0;
static int g_open_verbosity_tls=2;
static bool g_open_use_gpu_tls=false;
static float g_open_temperature_tls=.7f;
static uint64_t g_open_seed_tls=1234567890123ULL;
static params breeze_tts_2_context_default_params() { return {}; }
static breeze_tts_2_context* breeze_tts_2_init_from_file(const char*,params p) {
    received=p;return &context;
}
static void breeze_tts_2_set_temperature(breeze_tts_2_context*,float t) {
    received_temperature=t;temperature_calls++;
}
static void breeze_tts_2_set_seed(breeze_tts_2_context*,uint64_t s) {
    received_seed=s;seed_calls++;
}
'''

MAIN = r'''
int main(int argc,char** argv) {
    assert(argc==2);std::string scenario=argv[1];crispasr_session s;s.bt2_ctx=&context;
    if(scenario=="open" || scenario=="open-seed") {
        auto* result=open_breeze("/nonexistent/model.gguf");assert(result);
        if(scenario=="open-seed") assert(received.seed==g_open_seed_tls);
        else assert(received.n_threads==4 && received.verbosity==2 && !received.use_gpu && received.temperature==.7f);
        delete result;
    } else if(scenario=="temperature") {
        assert(crispasr_session_set_temperature(&s,0.f,77)==0);
        assert(temperature_calls==1 && received_temperature==0.f && seed_calls==1 && received_seed==77);
        assert(s.temperature==0.f && s.seed==77);
    } else if(scenario=="seed") {
        assert(crispasr_session_set_tts_seed(&s,0x123456789ABCULL)==0);
        assert(seed_calls==1 && received_seed==0x123456789ABCULL);
    } else if(scenario=="limit") {
        assert(crispasr_session_set_max_new_tokens(&s,128)==0);
        assert(s.max_new_tokens==128 && context.max_new_tokens==128);
    } else if(scenario=="reset") {
        assert(crispasr_session_set_max_new_tokens(&s,128)==0);
        assert(crispasr_session_set_max_new_tokens(&s,-3)==0);
        assert(s.max_new_tokens==0 && context.max_new_tokens==999);
    } else if(scenario=="null") {
        assert(crispasr_session_set_temperature(nullptr,0.f,1)==-1);
        assert(crispasr_session_set_tts_seed(nullptr,1)==-1);
        assert(crispasr_session_set_max_new_tokens(nullptr,1)==-1);
        breeze_tts_2_set_max_new_tokens(nullptr,1);
    } else if(scenario=="unsupported") {
        s.bt2_ctx=nullptr;
        assert(crispasr_session_set_temperature(&s,0.f,1)==-2);
        assert(crispasr_session_set_tts_seed(&s,1)==-2);
        assert(temperature_calls==0 && seed_calls==0);
    }
}
'''


class BreezeSessionControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ.get('CRISPASR_CONTRACT_SOURCE', ROOT / 'src/crispasr_c_api.cpp')).read_text()
        native = Path(os.environ.get('CRISPASR_BREEZE_SOURCE', ROOT / 'src/breeze_tts_2.cpp')).read_text()
        branch = source.index('    if (s->backend == "bt2-tts"')
        start = source.rfind('#endif\n', 0, branch) + len('#endif\n')
        end = source.index('#ifdef CA_HAVE_POCKET', branch)
        text = PREAMBLE + function(native, 'extern "C" void breeze_tts_2_set_max_new_tokens(')
        text += '\n' + function(source, 'static void session_apply_max_new_tokens(crispasr_session* s, int n) {')
        for signature in ['CA_EXPORT int crispasr_session_set_temperature(',
                          'CA_EXPORT int crispasr_session_set_tts_seed(',
                          'CA_EXPORT int crispasr_session_set_max_new_tokens(']:
            text += '\n' + function(source, signature)
        text += '\nstatic crispasr_session* open_breeze(const char* model_path) {\n'
        text += 'auto* s=new crispasr_session;\n' + source[start:end] + '\ndelete s;return nullptr;\n}\n' + MAIN
        cls.tmp = tempfile.TemporaryDirectory(prefix='crispasr-breeze-controls-')
        folder = Path(cls.tmp.name)
        (folder / 'controls.cpp').write_text(text)
        cls.binary = folder / 'controls'
        subprocess.run(['c++', '-std=c++17', str(folder / 'controls.cpp'), '-o', str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, scenario):
        subprocess.run([str(self.binary), scenario], check=True)

    def test_independent_open_guard(self):
        self.check('open')

    def test_open_forwards_seed(self):
        self.check('open-seed')

    def test_temperature_and_seed(self):
        self.check('temperature')

    def test_runtime_seed(self):
        self.check('seed')

    def test_runtime_limit(self):
        self.check('limit')

    def test_clear_limit_restores_model_default(self):
        self.check('reset')

    def test_null_arguments(self):
        self.check('null')

    def test_unsupported_preserved(self):
        self.check('unsupported')


if __name__ == '__main__':
    unittest.main()
