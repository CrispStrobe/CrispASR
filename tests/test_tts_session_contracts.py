#!/usr/bin/env python3
"""Compile real session dispatch with stub backends to test failure/ownership contracts.

No models or numerical acceptance: the production setter/synthesis bodies are
compiled unchanged, while loader/backend stubs inject failures and track every
caller-owned allocation. Set CRISPASR_CONTRACT_SOURCE to replay a prior source.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def block(source, signature):
    start = source.index(signature)
    opening = source.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


PREAMBLE = r'''
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <unordered_set>
#include "moss_tts_local.h"
#define CA_EXPORT
#define CA_HAVE_BT2_TTS
#define CA_HAVE_MOSS_TTS_LOCAL
#define CA_HAVE_MINI_OMNI2
static std::unordered_set<void*> allocations;
static int until_failure=-1;
static bool encode_failure=false, codec_failure=false, conversion_failure=false, load_failure=false;
static void* tracked_malloc(size_t n) {
    if (until_failure==0) { until_failure=-1; return nullptr; }
    if (until_failure>0) --until_failure;
    void* p=std::malloc(n); if(p) allocations.insert(p); return p;
}
static void tracked_free(void* p) {
    if(p) { assert(allocations.erase(p)==1); std::free(p); }
}
struct crispasr_session {
    void* bt2_ctx=nullptr; void* mini_omni2_ctx=nullptr;
    moss_tts_local_context* moss_tts_local_ctx=nullptr;
    moss_tts_local_synth_params* moss_tts_local_params=nullptr;
    std::vector<float> bt2_ref_pcm, disclaimer_pcm, ov2_ref_pcm;
    std::string bt2_ref_text, cosyvoice3_voices_path, voice_pack_identity, voice_path;
    bool voice_is_clone=false, logged_clone_consent=false;
    void* melotts_ctx=nullptr; void* ov2_ctx=nullptr;
};
namespace crispasr_voice {
struct CloneDecision { bool is_clone=false; std::string pack_identity; };
static CloneDecision classify_voice(const char*,const std::string&,bool,const std::string&) {return {};}
}
static int crispasr_session_input_sample_rate(crispasr_session*) {return 22050;}
static int crispasr_session_output_sample_rate(crispasr_session*);
static int crispasr_audio_load_at_rate(const char* path,int target,float** pcm,int* n,int* sr) {
    if(load_failure) return -1;
    *n=4;*sr=target;*pcm=static_cast<float*>(tracked_malloc(4*sizeof(float)));assert(*pcm);
    for(int i=0;i<4;++i) (*pcm)[i]=path[0]=='a'?1.f:2.f;
    return 0;
}
extern "C" bool moss_tts_local_can_clone(const moss_tts_local_context*) {return true;}
extern "C" int32_t* moss_tts_local_encode_reference(moss_tts_local_context*,const float* pcm,int,int,int* nv,int* nt) {
    if(encode_failure) return nullptr;
    *nv=2;*nt=2;auto* p=static_cast<int32_t*>(tracked_malloc(4*sizeof(int32_t)));assert(p);
    for(int i=0;i<4;++i) p[i]=int(pcm[0]);return p;
}
extern "C" moss_tts_local_synth_params moss_tts_local_synth_default_params() {return {};}
extern "C" bool moss_tts_local_set_codec_path(moss_tts_local_context*,const char*) {return true;}
static bool mini_omni2_load_snac(void*,const char*) {return !codec_failure;}
extern "C" void moss_tts_local_free(moss_tts_local_context*) {}
extern "C" int moss_tts_local_sampling_rate(const moss_tts_local_context*) {return 48000;}
static int melotts_sample_rate(void*) {return 44100;}
static int melotts_synthesize(void*,const char*,float** p,int* sr) {
    *sr=44100;*p=static_cast<float*>(tracked_malloc(4*sizeof(float)));assert(*p);return 4;
}
static bool openvoice2_convert(void*,const float*,int,int sr,const float*,int,int,float** p,int* n) {
    assert(sr==44100);*n=2;*p=static_cast<float*>(tracked_malloc(2*sizeof(float)));assert(*p);
    (*p)[0]=.1f;(*p)[1]=.2f;return !conversion_failure;
}
static int openvoice2_sample_rate(void*) {return 22050;}
namespace core_audio {
static std::vector<float> resample_polyphase(const float*,int n,int from,int to) {
    assert(from==22050 && to==44100);return std::vector<float>(n*to/from,.1f);
}
}
'''

MAIN = r'''
int main(int argc,char** argv) {
 assert(argc==2);std::string scenario=argv[1];crispasr_session s;
 if(scenario=="breeze") {
  s.bt2_ctx=&s;
  assert(crispasr_session_set_voice(&s,"b.wav",nullptr)==-2);
  assert(s.bt2_ref_pcm.empty() && s.bt2_ref_text.empty() && s.voice_path.empty());
  assert(crispasr_session_set_voice(&s,"a.wav","text A")==0);
  s.disclaimer_pcm={.5f};
  assert(crispasr_session_set_voice(&s,"b.wav","")==-2);
  assert(s.bt2_ref_pcm==std::vector<float>(4,1.f) && s.bt2_ref_text=="text A");
  assert(s.voice_path=="a.wav" && s.disclaimer_pcm==std::vector<float>{.5f});
  load_failure=true;assert(crispasr_session_set_voice(&s,"b.wav","text B")==-1);load_failure=false;
  assert(s.bt2_ref_pcm==std::vector<float>(4,1.f) && s.bt2_ref_text=="text A" && s.voice_path=="a.wav");
  assert(crispasr_session_set_voice(&s,"b.wav","text B")==0);
  assert(s.bt2_ref_pcm==std::vector<float>(4,2.f) && s.bt2_ref_text=="text B");
 } else if(scenario=="moss") {
  s.moss_tts_local_ctx=reinterpret_cast<moss_tts_local_context*>(&s);
  assert(crispasr_session_set_voice(&s,"a.wav",nullptr)==0);assert(allocations.size()==2);
  auto* prior=s.moss_tts_local_params;
  encode_failure=true;assert(crispasr_session_set_voice(&s,"b.wav",nullptr)==-1);encode_failure=false;
  assert(s.moss_tts_local_params==prior && allocations.size()==2 && prior->ref_codes[0]==1);
  until_failure=2;assert(crispasr_session_set_voice(&s,"b.wav",nullptr)==-1);
  assert(s.moss_tts_local_params==prior && allocations.size()==2);
  for(int i=0;i<100;++i) {assert(crispasr_session_set_voice(&s,"b.wav",nullptr)==0);assert(allocations.size()==2);}
  assert(s.moss_tts_local_params->ref_codes[0]==2);close_moss(&s);
 } else if(scenario=="codec") {
  s.mini_omni2_ctx=&s;codec_failure=true;assert(crispasr_session_set_codec_path(&s,"bad.gguf")==-1);
  codec_failure=false;assert(crispasr_session_set_codec_path(&s,"good.gguf")==0);
 } else if(scenario=="openvoice") {
  s.melotts_ctx=&s;assert(crispasr_session_output_sample_rate(&s)==44100);
  s.ov2_ctx=&s;s.ov2_ref_pcm={.1f};assert(crispasr_session_output_sample_rate(&s)==22050);int n=0;
  for(int i=0;i<100;++i) {float* p=synthesize_melo(&s,"hello",&n);assert(p && n==2);tracked_free(p);assert(allocations.empty());}
  conversion_failure=true;assert(!synthesize_melo(&s,"hello",&n));assert(allocations.empty());
 } else {return 2;}
 assert(allocations.empty());
}
'''


class TtsSessionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ.get('CRISPASR_CONTRACT_SOURCE', ROOT / 'src/crispasr_c_api.cpp')).read_text()
        voice = block(source, 'CA_EXPORT int crispasr_session_set_voice(')
        codec = block(source, 'CA_EXPORT int crispasr_session_set_codec_path(')
        # Compile the actual model dispatch arm, including its feature guards.
        synth_start = source.index('#ifdef CA_HAVE_MELOTTS', source.index('crispasr_session_synthesize_raw_impl('))
        melo = block(source[synth_start:], '    if (s->melotts_ctx) {')
        close_start = source.index('#ifdef CA_HAVE_MOSS_TTS_LOCAL', source.index('CA_EXPORT void crispasr_session_close('))
        close_end = source.index('#endif', close_start)
        close = source[close_start:close_end]
        text = PREAMBLE + voice + '\n' + codec + '\n#define CA_HAVE_OPEN_VOICE2\n'
        text += 'static float* synthesize_melo(crispasr_session* s,const char* text,int* out_n_samples) {\n' + melo + '\nreturn nullptr;\n}\n'
        text += '\n#define CA_HAVE_MELOTTS\n' + block(source, 'CA_EXPORT int crispasr_session_output_sample_rate(') + '\n'
        text += 'static void close_moss(crispasr_session* s) {\n' + close + '\n#endif\n}\n' + MAIN
        # Instrument only explicit caller-owned allocations in production code.
        import re
        begin = len(PREAMBLE)
        text = text[:begin] + re.sub(r'(?:std::)?\b(malloc|free)\(', r'tracked_\1(', text[begin:])
        cls.tmp = tempfile.TemporaryDirectory(prefix='crispasr-tts-contract-')
        path = Path(cls.tmp.name)
        (path / 'contract.cpp').write_text(text)
        cls.binary = path / 'contract'
        subprocess.run(['c++', '-std=c++17', '-I', str(ROOT / 'src'), str(path / 'contract.cpp'), '-o', str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_breeze_failure_preserves_reference_and_provenance(self):
        subprocess.run([str(self.binary), 'breeze'], check=True)

    def test_moss_replace_failure_and_close_release_buffers(self):
        subprocess.run([str(self.binary), 'moss'], check=True)

    def test_snac_loader_failure_is_reported(self):
        subprocess.run([str(self.binary), 'codec'], check=True)

    def test_openvoice_rates_and_owned_outputs(self):
        subprocess.run([str(self.binary), 'openvoice'], check=True)


if __name__ == '__main__':
    unittest.main()
