#!/usr/bin/env python3
"""Compile actual Breeze session dispatch and verify all four request modes.

Backend stubs inspect received parameters; this does not test model numerics.
CRISPASR_CONTRACT_SOURCE can replay either pre-composition source.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def dispatch(source):
    start = source.index('    if (s->bt2_ctx) {', source.index('crispasr_session_synthesize_raw_impl('))
    opening = source.index('{', start)
    end, depth = opening + 1, 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


PREAMBLE = r'''
#include <cassert>
#include <string>
#include <vector>
struct crispasr_session {
    void* bt2_ctx=nullptr;
    std::vector<float> bt2_ref_pcm;
    std::string bt2_ref_text, bt2_instruct;
    float bt2_cfg_scale=1.f;
};
static float output[]={.25f,.5f};
static void* received_ctx;
static std::string received_text, received_instruction, received_ref_text;
static const float* received_ref;
static int received_ref_samples;
static float received_scale;
static bool synthesis_failure=false;
static float* record(void* ctx,const char* text,const char* ins,const float* ref,
                     int n,const char* ref_text,float scale,int* out_n) {
    received_ctx=ctx;received_text=text;
    received_instruction=ins?ins:"";received_ref_text=ref_text?ref_text:"";
    received_ref=ref;received_ref_samples=n;received_scale=scale;
    if(out_n) *out_n=synthesis_failure?999:2;
    return synthesis_failure?nullptr:output;
}
static float* breeze_tts_2_synthesize(void* ctx,const char* text,int* out_n) {
    return record(ctx,text,nullptr,nullptr,0,nullptr,1.f,out_n);
}
static float* breeze_tts_2_synthesize_with_reference(void* ctx,const char* text,
    const float* ref,int n,const char* ref_text,int* out_n) {
    return record(ctx,text,nullptr,ref,n,ref_text,1.f,out_n);
}
static float* breeze_tts_2_synthesize_instructed(void* ctx,const char* text,const char* ins,
    const float* ref,int n,const char* ref_text,float scale,int* out_n) {
    return record(ctx,text,ins,ref,n,ref_text,scale,out_n);
}
'''

MAIN = r'''
static void check(crispasr_session& s,bool reference,bool instruction) {
    int n=-1;
    float* p=synthesize(&s,"New sentence.",&n);
    assert(p==output && n==2 && received_ctx==s.bt2_ctx && received_text=="New sentence.");
    assert(received_ref_samples==(reference?3:0));
    assert(received_ref==(reference?s.bt2_ref_pcm.data():nullptr));
    assert(received_ref_text==(reference?"Reference transcript.":""));
    assert(received_instruction==(instruction?"Speak calmly.":""));
    if(instruction) assert(received_scale==4.f);
    assert(s.bt2_ref_pcm==(reference?std::vector<float>{.1f,.2f,.3f}:std::vector<float>{}));
}
static void setup(crispasr_session& s,bool reference,bool instruction) {
    s.bt2_ref_pcm=reference?std::vector<float>{.1f,.2f,.3f}:std::vector<float>{};
    s.bt2_ref_text=reference?"Reference transcript.":"";
    s.bt2_instruct=instruction?"Speak calmly.":"";
    s.bt2_cfg_scale=instruction?4.f:1.f;
}
int main(int argc,char** argv) {
    assert(argc==2);std::string scenario=argv[1];crispasr_session s;s.bt2_ctx=&s;
    if(scenario=="failure") {
        setup(s,true,true);synthesis_failure=true;int n=0;
        assert(synthesize(&s,"New sentence.",&n)==nullptr && n==0);
        assert(s.bt2_ref_pcm==std::vector<float>({.1f,.2f,.3f}));
        assert(s.bt2_ref_text=="Reference transcript." && s.bt2_instruct=="Speak calmly.");
    } else if(scenario=="no-count") {
        setup(s,true,true);assert(synthesize(&s,"New sentence.",nullptr)==output);
        assert(received_ref_samples==3 && received_instruction=="Speak calmly.");
    } else if(scenario=="reused") {
        for(int i=0;i<20;++i) for(auto flags:{0,3,1,2}) {
            bool ref=flags&1,ins=flags&2;setup(s,ref,ins);check(s,ref,ins);
        }
    } else {
        bool ref=scenario=="clone"||scenario=="direction";
        bool ins=scenario=="design"||scenario=="direction";
        setup(s,ref,ins);check(s,ref,ins);
    }
}
'''


class BreezeSessionModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ.get('CRISPASR_CONTRACT_SOURCE', ROOT / 'src/crispasr_c_api.cpp')).read_text()
        text = PREAMBLE + '\nstatic float* synthesize(crispasr_session* s,const char* text,int* out_n_samples) {\n'
        text += dispatch(source) + '\nreturn nullptr;\n}\n' + MAIN
        cls.tmp = tempfile.TemporaryDirectory(prefix='crispasr-breeze-modes-')
        folder = Path(cls.tmp.name)
        (folder / 'modes.cpp').write_text(text)
        cls.binary = folder / 'modes'
        subprocess.run(['c++', '-std=c++17', str(folder / 'modes.cpp'), '-o', str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_plain(self):
        subprocess.run([str(self.binary), 'plain'], check=True)

    def test_design(self):
        subprocess.run([str(self.binary), 'design'], check=True)

    def test_clone(self):
        subprocess.run([str(self.binary), 'clone'], check=True)

    def test_direction(self):
        subprocess.run([str(self.binary), 'direction'], check=True)

    def test_reused_context_modes(self):
        subprocess.run([str(self.binary), 'reused'], check=True)

    def test_failure_keeps_zero_output_count_and_request(self):
        subprocess.run([str(self.binary), 'failure'], check=True)

    def test_optional_output_count(self):
        subprocess.run([str(self.binary), 'no-count'], check=True)


if __name__ == '__main__':
    unittest.main()
