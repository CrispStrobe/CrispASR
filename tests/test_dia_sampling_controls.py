#!/usr/bin/env python3
"""Execute literal Dia setters, C ABI dispatch, CLI adapter and Python methods.

Only model loading/synthesis and unrelated backends are stubs. No model or
numerical parity claim. CRISPASR_DIA_CONTROL_ROOT can replay an old checkout.
"""
import ast
import ctypes
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest

ROOT = Path(os.environ.get('CRISPASR_DIA_CONTROL_ROOT', Path(__file__).resolve().parents[1]))


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
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <sys/stat.h>
#include "dia_tts.h"
#define CA_EXPORT extern "C"
#ifndef WITHOUT_DIA
#define CA_HAVE_DIA
#endif
struct dia_tts_context { dia_tts_context_params params; };
struct crispasr_session { dia_tts_context* dia_tts_ctx=nullptr; };
static dia_tts_context context;
static dia_tts_context_params synthesized;
extern "C" dia_tts_context* dia_tts_init_from_file(const char*, dia_tts_context_params p) {
    context.params=p;return &context;
}
extern "C" int dia_tts_set_codec_path(dia_tts_context*, const char*) {return 0;}
extern "C" void dia_tts_set_seed(dia_tts_context* c, uint64_t seed) {c->params.seed=seed;}
extern "C" float* dia_tts_synthesize(dia_tts_context* c, const char*, int* n) {
    synthesized=c->params;*n=1;
    auto* p=static_cast<float*>(std::malloc(sizeof(float)));*p=.25f;return p;
}
extern "C" void dia_tts_pcm_free(float* p) {std::free(p);}
extern "C" void dia_tts_free(dia_tts_context*) {}
struct whisper_params {
    int n_threads=4,max_new_tokens=512,tts_top_k=-1;
    bool no_prints=true,max_new_tokens_explicit=false,auto_download=false;
    uint64_t seed=42;
    float temperature=0,tts_top_p=-1;
    std::string model="/nonexistent/dia.gguf",tts_codec_model="auto",backend="dia",cache_dir,tts_codec_quant;
};
struct crispasr_segment {};
enum { CAP_TTS=1,CAP_AUTO_DOWNLOAD=2,CAP_TEMPERATURE=4 };
struct CrispasrBackend {
    virtual ~CrispasrBackend()=default;
    virtual const char* name() const=0;
    virtual uint32_t capabilities() const=0;
    virtual int tts_sample_rate() const=0;
    virtual std::vector<crispasr_segment> transcribe(const float*,int,int64_t,const whisper_params&)=0;
    virtual bool init(const whisper_params&)=0;
    virtual std::vector<float> synthesize(const std::string&,const whisper_params&)=0;
    virtual void shutdown()=0;
};
static bool crispasr_backend_should_use_gpu(const whisper_params&) {return false;}
static std::string crispasr_resolve_model_cli(const std::string& p,const std::string&,bool,
    const std::string&,bool,const std::string&) {return p;}
struct CrispasrRegistryEntry {std::string companion_filename;};
static bool crispasr_registry_lookup(const std::string&,CrispasrRegistryEntry&,const std::string&) {return false;}
extern "C" void* test_session() {
    static crispasr_session session;
    context.params=dia_tts_context_default_params();session.dia_tts_ctx=&context;return &session;
}
extern "C" int test_top_k() {return context.params.top_k;}
extern "C" float test_top_p() {return context.params.top_p;}
'''

MAIN = r'''
int main(int argc,char** argv) {
    assert(argc==2);std::string mode=argv[1];context.params=dia_tts_context_default_params();
    if(mode=="default") {
        assert(context.params.temperature==1.2f && context.params.top_k==45 && context.params.top_p==.95f);
    } else if(mode=="null") {
        dia_tts_set_top_k(nullptr,35);dia_tts_set_top_p(nullptr,.8f);
        assert(crispasr_session_set_top_k(nullptr,35)==-1);
        assert(crispasr_session_set_top_p(nullptr,.8f)==-1);
    } else if(mode=="unsupported") {
        crispasr_session s;
        assert(crispasr_session_set_top_k(&s,35)==-2);
        assert(crispasr_session_set_top_p(&s,.8f)==-2);
        assert(context.params.top_k==45 && context.params.top_p==.95f);
    } else if(mode=="session") {
        crispasr_session s;s.dia_tts_ctx=&context;
#ifdef WITHOUT_DIA
        assert(crispasr_session_set_top_k(&s,35)==-2);
        assert(crispasr_session_set_top_p(&s,.8f)==-2);
#else
        assert(crispasr_session_set_top_k(&s,35)==0);
        assert(crispasr_session_set_top_p(&s,.8f)==0);
        assert(context.params.top_k==35 && context.params.top_p==.8f);
        assert(crispasr_session_set_top_k(&s,0)==0);
        assert(crispasr_session_set_top_p(&s,1.f)==0);
        assert(context.params.top_k==0 && context.params.top_p==1.f);
        assert(crispasr_session_set_top_k(&s,45)==0);
        assert(crispasr_session_set_top_p(&s,.95f)==0);
        assert(context.params.top_k==45 && context.params.top_p==.95f);
#endif
    } else if(mode=="cli-reset-default" || mode=="cli-reset-configured") {
        auto backend=crispasr_make_dia_backend();whisper_params configured;
        if(mode=="cli-reset-configured") {configured.tts_top_k=35;configured.tts_top_p=.8f;}
        assert(backend->init(configured));
        whisper_params request;request.tts_top_k=0;request.tts_top_p=1.f;
        backend->synthesize("[S1] hello",request);
        assert(synthesized.top_k==0 && synthesized.top_p==1.f);
        whisper_params omitted;
        backend->synthesize("[S1] hello",omitted);
        assert(synthesized.top_k==(mode=="cli-reset-configured"?35:45));
        assert(synthesized.top_p==(mode=="cli-reset-configured"?.8f:.95f));
    } else if(mode=="cli-default" || mode=="cli-explicit" || mode=="cli-repeated") {
        auto backend=crispasr_make_dia_backend();whisper_params p;
        if(mode!="cli-default") {
            char a[]="crispasr",b[]="--tts-top-k",c[]="35",d[]="--tts-top-p",e[]=".8";
            char* args[]={a,b,c,d,e};int i=1;
            assert(whisper_params_parse_arg_tts_sampling(5,args,i,p) && i==2);
            ++i;assert(whisper_params_parse_arg_tts_sampling(5,args,i,p) && i==4);
        }
        assert(backend->init(p));
        assert(context.params.top_k==(mode=="cli-default"?45:35));
        assert(context.params.top_p==(mode=="cli-default"?.95f:.8f));
        assert(backend->synthesize("[S1] hello",p)==std::vector<float>{.25f});
        assert(synthesized.top_k==(mode=="cli-default"?45:35));
        assert(synthesized.top_p==(mode=="cli-default"?.95f:.8f));
        if(mode=="cli-repeated") {
            p.tts_top_k=0;p.tts_top_p=1.f;
            backend->synthesize("[S1] hello",p);
            assert(synthesized.top_k==0 && synthesized.top_p==1.f);
            p.tts_top_k=45;p.tts_top_p=.95f;
            backend->synthesize("[S1] hello",p);
            assert(synthesized.top_k==45 && synthesized.top_p==.95f);
        }
    } else {return 2;}
}
'''


class DiaSamplingControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native = (ROOT / 'src/dia_tts.cpp').read_text()
        dispatch = Path(os.environ.get('CRISPASR_DIA_DISPATCH_SOURCE', ROOT / 'src/crispasr_c_api.cpp')).read_text()
        adapter = Path(os.environ.get('CRISPASR_DIA_ADAPTER_SOURCE', ROOT / 'examples/cli/crispasr_backend_dia.cpp')).read_text()
        cli = (ROOT / 'examples/cli/cli.cpp').read_text()
        # Entire real adapter, only includes replaced by the recording boundary.
        adapter = '\n'.join(line for line in adapter.splitlines() if not line.startswith('#include'))
        text = PREAMBLE
        for signature in ['struct dia_tts_context_params dia_tts_context_default_params(',
                          'void dia_tts_set_temperature(', 'void dia_tts_set_top_k(', 'void dia_tts_set_top_p(']:
            text += '\n' + function(native, signature)
        for name in ['top_k', 'top_p']:
            text += '\n' + function(dispatch, 'CA_EXPORT int crispasr_session_set_' + name + '(')
        text += '\n' + function(cli, 'static char* requires_value_error(')
        text += '\n' + function(cli, 'static bool whisper_params_parse_arg_tts_sampling(')
        cls.tmp = tempfile.TemporaryDirectory(prefix='crispasr-dia-controls-')
        folder = Path(cls.tmp.name)
        source = folder / 'controls.cpp'
        source.write_text(text + '\n' + adapter + '\n' + MAIN)
        cls.enabled, cls.disabled = folder / 'enabled', folder / 'disabled'
        common = ['c++', '-std=c++17', '-I', str(ROOT / 'src'), str(source)]
        subprocess.run(common + ['-o', str(cls.enabled)], check=True)
        subprocess.run(common + ['-DWITHOUT_DIA', '-o', str(cls.disabled)], check=True)
        shared = folder / 'controls.so'
        subprocess.run(common + ['-shared', '-fPIC', '-o', str(shared)], check=True)
        cls.lib = ctypes.CDLL(str(shared))
        cls.lib.test_session.restype = ctypes.c_void_p
        cls.lib.test_top_p.restype = ctypes.c_float
        tree = ast.parse((ROOT / 'python/crispasr/_binding.py').read_text())
        session = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Session')
        nodes = [node for node in session.body if isinstance(node, ast.FunctionDef) and node.name in ['set_top_k', 'set_top_p']]
        assert len(nodes) == 2
        namespace = {'ctypes': ctypes}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual-python-session-methods', 'exec'), namespace)
        cls.python_methods = namespace

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_native_session_cli_controls(self):
        for mode in ['default', 'null', 'unsupported', 'session', 'cli-default', 'cli-explicit',
                     'cli-repeated', 'cli-reset-default', 'cli-reset-configured']:
            with self.subTest(mode=mode):
                subprocess.run([str(self.enabled), mode], check=True)

    def test_backend_compiled_out_remains_unsupported(self):
        subprocess.run([str(self.disabled), 'session'], check=True)

    def test_actual_python_methods_reach_actual_c_abi(self):
        session = types.SimpleNamespace(_lib=self.lib, _handle=self.lib.test_session())
        for k, p in [(35, .8), (0, 1.), (45, .95)]:
            self.python_methods['set_top_k'](session, k)
            self.python_methods['set_top_p'](session, p)
            self.assertEqual(self.lib.test_top_k(), k)
            self.assertAlmostEqual(self.lib.test_top_p(), p, places=6)


if __name__ == '__main__':
    unittest.main()
