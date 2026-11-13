"""Additional public-header utilities: native transport and literal WAV/sign wrappers."""
import ctypes as C
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
import crispasr
from test_session_utf8 import definition


class AudioUtilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        path=Path(cls.tmp.name)
        code=r'''
#include "crispasr.h"
#include "crispasr_session.h"
#include "core/crispasr_wav_writer.h"
#include <cstdlib>
#include <cstring>
static int live,mode;
extern "C" int probe_live() { return live; }
extern "C" void probe_mode(int m) { mode=m; }
static void* probe_malloc(size_t n) { ++live;return std::malloc(n ? n : 1); }
static void probe_free(void* p) { if(p){ --live;std::free(p); } }
static bool crispasr_c2pa_sign_auto(std::string& b,const std::string& f,const std::string& cert,
                                  const std::string& key,const std::string&) {
    if(mode==2 || f!="audio/wav" || (cert.empty()!=key.empty()))return false;
    b.append("signed\0",7);return true;
}
#define CA_EXPORT extern "C"
#define malloc probe_malloc
#define free probe_free
'''
        source=(ROOT/'src/crispasr_c_api.cpp').read_text()
        for marker in ('CA_EXPORT unsigned char* crispasr_c2pa_sign(',
                       'CA_EXPORT void crispasr_c2pa_free(',
                       'CA_EXPORT unsigned char* crispasr_pcm_to_wav('):
            code+=definition(source,marker)+'\n'
        code+=r'''
#undef malloc
#undef free
extern "C" void crispasr_audio_free(float* p) { probe_free(p); }
extern "C" int crispasr_audio_load_at_rate(const char*,int rate,float** pcm,int* n,int* sr) {
    *pcm=(float*)probe_malloc(12);(*pcm)[0]=1;(*pcm)[1]=2;(*pcm)[2]=3;
    *n=mode==1 ? 0:3;*sr=rate;return mode==2 ? -1:0;
}
extern "C" int crispasr_audio_load(const char* p,float** pcm,int* n,int* sr) {
    return crispasr_audio_load_at_rate(p,16000,pcm,n,sr);
}
extern "C" int crispasr_audio_load_stereo(const char* p,float** l,float** r,int* n,int* sr,int* channels) {
    crispasr_audio_load(p,l,n,sr);crispasr_audio_load(p,r,n,sr);
    for(int i=0;i<3;i++)(*r)[i]*=-1;
    *channels=2;return mode==2 ? -1:0;
}
static int vad_calls;
extern "C" int probe_vad_calls() { return vad_calls; }
static int vad_result(const float* pcm, int n, float** spans, float scale) {
    ++vad_calls;
    if(n!=3 || pcm[0]!=3 || pcm[1]!=2 || pcm[2]!=1) return -9;
    if(mode==3 || mode==4) { *spans=nullptr;return mode==3 ? 1:0; }
    *spans=(float*)probe_malloc(4*sizeof(float));
    (*spans)[0]=1.25f*scale;(*spans)[1]=2.5f*scale;
    (*spans)[2]=3.75f*scale;(*spans)[3]=4.5f*scale;
    return mode==1 ? 0 : mode==2 ? -3 : 2;
}
extern "C" int crispasr_vad_segments(const char*,const float* pcm,int n,int,float,int,int,int,bool,float** spans) {
    return vad_result(pcm,n,spans,100);
}
extern "C" int crispasr_vad_slices(const char*,const float* pcm,int n,int,float,int,int,int,float,int,float** spans) {
    return vad_result(pcm,n,spans,1);
}
extern "C" void crispasr_vad_free(float* spans) { probe_free(spans); }
extern "C" int crispasr_session_input_channels(crispasr_session*) { return 1; }
extern "C" int crispasr_session_output_channels(crispasr_session*) { return 2; }
'''
        (path/'probe.cpp').write_text(code)
        cls.lib_path=str(path/'probe.so')
        subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-shared','-fPIC',
                        '-I'+str(ROOT/'include'),'-I'+str(ROOT/'ggml/include'),'-I'+str(ROOT/'src'),
                        str(path/'probe.cpp'),'-o',cls.lib_path],check=True)
        cls.lib=C.CDLL(cls.lib_path)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()
    def setUp(self): self.lib.probe_mode(0)
    def tearDown(self): self.assertEqual(self.lib.probe_live(),0)

    def test_decode_layout_rates_empty_and_error_cleanup(self):
        result=crispasr.load_audio('probe',lib_path=self.lib_path)
        np.testing.assert_array_equal(result.samples,[1,2,3])
        self.assertEqual(result.sample_rate,16000)
        result=crispasr.load_audio('probe',sample_rate=44100,lib_path=self.lib_path)
        self.assertEqual(result.sample_rate,44100)
        stereo=crispasr.load_audio_stereo('probe',lib_path=self.lib_path)
        np.testing.assert_array_equal(stereo.samples,[[1,-1],[2,-2],[3,-3]])
        self.assertEqual(stereo.source_channels,2)
        self.lib.probe_mode(1)
        self.assertEqual(crispasr.load_audio('probe',lib_path=self.lib_path).samples.shape,(0,))
        self.assertEqual(crispasr.load_audio_stereo('probe',lib_path=self.lib_path).samples.shape,(0,2))
        self.lib.probe_mode(2)
        for fn in (crispasr.load_audio,crispasr.load_audio_stereo):
            with self.assertRaises(RuntimeError):fn('probe',lib_path=self.lib_path)
            self.assertEqual(self.lib.probe_live(),0)

    def test_real_wav_container_and_binary_signing_transport(self):
        pcm=np.array([0,.25,0,-.25],dtype=np.float32)[::2]
        wav=crispasr.pcm_to_wav(pcm,24000,lib_path=self.lib_path)
        self.assertEqual(wav[:4],b'RIFF');self.assertIn(b'WAVE',wav)
        self.assertIn(b'AI-generated',wav)
        self.assertEqual(crispasr.c2pa_sign(b'\0abc',lib_path=self.lib_path),b'\0abcsigned\0')
        self.assertEqual(crispasr.c2pa_sign(b'abc',cert_path='cert.pem',key_path='key.pem',lib_path=self.lib_path),b'abcsigned\0')
        with self.assertRaises(ValueError):crispasr.c2pa_sign(b'abc',cert_path='cert.pem',lib_path=self.lib_path)
        self.lib.probe_mode(2)
        with self.assertRaises(RuntimeError):crispasr.c2pa_sign(wav,lib_path=self.lib_path)

    def test_vad_units_strided_transport_and_allocator(self):
        # Native legacy centiseconds and dispatcher seconds describe the same spans.
        pcm=np.array([1,2,3],dtype=np.float32)[::-1]
        for fn in (crispasr.vad_segments,crispasr.vad_slices):
            with self.subTest(fn=fn.__name__):
                spans=fn(pcm,'probe',lib_path=self.lib_path)
                self.assertEqual([(v.start,v.end) for v in spans],[(1.25,2.5),(3.75,4.5)])
                self.assertEqual(self.lib.probe_live(),0)

    def test_vad_cleanup_on_empty_error_and_python_exception(self):
        pcm=np.array([3,2,1],dtype=np.float32)
        for fn in (crispasr.vad_segments,crispasr.vad_slices):
            for mode in (1,2,3,4):
                with self.subTest(fn=fn.__name__,mode=mode):
                    self.lib.probe_mode(mode)
                    if mode in (1,4):
                        self.assertEqual(fn(pcm,'probe',lib_path=self.lib_path),[])
                    else:
                        with self.assertRaises(RuntimeError):fn(pcm,'probe',lib_path=self.lib_path)
                    self.assertEqual(self.lib.probe_live(),0)
            self.lib.probe_mode(0)
            with mock.patch('crispasr._binding.VadSpan',side_effect=ValueError('conversion failed')):
                with self.assertRaisesRegex(ValueError,'conversion failed'):
                    fn(pcm,'probe',lib_path=self.lib_path)
            self.assertEqual(self.lib.probe_live(),0)

    def test_vad_rejects_non_mono_before_native_dispatch(self):
        for fn in (crispasr.vad_segments,crispasr.vad_slices):
            for pcm in (np.ones((3,2)),np.ones((1,3)),1.0):
                with self.subTest(fn=fn.__name__,shape=np.shape(pcm)):
                    calls=self.lib.probe_vad_calls()
                    with self.assertRaises(ValueError):fn(pcm,'probe',lib_path=self.lib_path)
                    self.assertEqual(self.lib.probe_vad_calls(),calls)

    def test_channel_getters(self):
        s=crispasr.Session.__new__(crispasr.Session)
        s._lib=self.lib;s._handle=1
        self.assertEqual(s.input_channels(),1);self.assertEqual(s.output_channels(),2)
        s._handle=None
        with self.assertRaises(RuntimeError):s.input_channels()


if __name__=='__main__': unittest.main()
