"""Python operations through a compiled, public-header-checked C ABI probe.

No model inference: these tests establish transport/ownership/shape semantics.
"""
import ctypes as C
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
import crispasr
from crispasr import _binding


@unittest.skipIf(sys.platform == 'win32', 'probe uses POSIX shared compiler flags')
class WrapperOperations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = str(Path(cls.tmp.name) / 'probe.so')
        cc = os.environ.get('CC') or shutil.which('cc')
        if not cc:
            raise unittest.SkipTest('no C compiler')
        subprocess.run([cc, '-shared', '-fPIC', '-std=c11', '-Werror', '-DCRISPASR_PROBE_PYTHON_AUDIO',
                        '-I'+str(ROOT/'include'), str(ROOT/'tests/fixtures/python-wrapper-abi.c'),
                        '-o', cls.path], check=True)
        cls.lib = C.CDLL(cls.path)
        cls.lib.probe_mode.argtypes = [C.c_int]
        cls.lib.probe_live.restype = C.c_int
        cls.lib.probe_calls.restype = C.c_int

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.lib.probe_mode(0)
        self.session = crispasr.Session.__new__(crispasr.Session)
        self.session._handle = 1
        self.session._lib = self.lib
        self.session.backend = 'probe'
        self.pcm = np.arange(1, 7, dtype=np.float32)[::2]

    def tearDown(self):
        self.session.set_progress_callback(None)
        self.session.set_segment_callback(None)
        self.session.set_token_callback(None)
        self.session.close()
        self.assertEqual(self.lib.probe_live(), 0)
        self.assertEqual(self.lib.probe_invalid_frees(), 0)

    def test_separation_strides_and_owned_output(self):
        padded = np.zeros((3, 4), dtype=np.float32)
        padded[:, ::2] = np.arange(1, 7).reshape(3, 2)
        reversed_storage = np.arange(6, 0, -1, dtype=np.float32)
        for pcm in (padded[:, ::2], padded.reshape(-1)[::2], reversed_storage[::-1]):
            result = self.session.separate(pcm)['vocals']
            np.testing.assert_array_equal(result, np.arange(1, 7))
            self.lib.probe_mutate()
            self.assertEqual(result[0], 1)
        self.lib.probe_mode(2)
        with self.assertRaises(RuntimeError):
            self.session.separate(np.arange(1, 7))

    def test_audio_shape_and_oversized_views_rejected_before_dispatch(self):
        bad_stereo = [1.0, np.zeros((2, 3)), np.zeros(3)]
        bad_mono = [1.0, np.zeros((2, 2))]
        for dtype in (np.float32, np.float64):
            one = np.zeros(1, dtype=dtype)
            bad_stereo.extend([
                np.lib.stride_tricks.as_strided(one, shape=(2147483648, 2), strides=(0, 0)),
                np.lib.stride_tricks.as_strided(one, shape=(4294967296,), strides=(0,)),
            ])
            bad_mono.append(np.lib.stride_tricks.as_strided(one, shape=(2147483648,), strides=(0,)))
        for fn, inputs in ((self.session.separate, bad_stereo),
                           (lambda pcm: crispasr.enhance_audio_rnnoise(pcm, lib_path=self.path), bad_mono),
                           (self.session.piano, bad_mono)):
            for pcm in inputs:
                calls = self.lib.probe_calls()
                # A rejected virtual view must never be materialized.
                with patch.object(np, 'ascontiguousarray', side_effect=AssertionError('unexpected copy')):
                    with self.assertRaises(ValueError):
                        fn(pcm)
                self.assertEqual(self.lib.probe_calls(), calls)

    def test_rnnoise_strided_mono_output_and_errors(self):
        for pcm in (self.pcm, np.array([5, 3, 1], dtype=np.float64)[::-1]):
            result = crispasr.enhance_audio_rnnoise(pcm, lib_path=self.path)
            np.testing.assert_array_equal(result, [.5, 1.5, 2.5])
            self.assertEqual(result.dtype, np.float32)
            result[0] = 42
            self.assertEqual(pcm[0], 1)
        self.lib.probe_mode(2)
        with self.assertRaises(RuntimeError):
            crispasr.enhance_audio_rnnoise(self.pcm, lib_path=self.path)

    def test_music_layout_units_and_borrowed_copies(self):
        piano = self.session.piano(self.pcm)
        np.testing.assert_array_equal(piano.notes, np.arange(8).reshape(2,4))
        np.testing.assert_array_equal(piano.programs, [24,128])
        self.assertEqual(piano.sample_rate,16000)
        chords = self.session.chords(self.pcm,16000)
        self.assertEqual(chords.names,['C','Am'])
        self.assertEqual(chords.vocabulary_size,170)
        beats = self.session.beats(self.pcm)
        self.assertEqual(beats.tempo_bpm,123.5)
        self.assertEqual(beats.events.shape,(2,2))
        self.assertEqual(beats.sample_rate,22050)
        tab = self.session.tab(self.pcm)
        self.assertEqual(tab.emissions.shape,(2,2,3))
        self.assertEqual(tab.open_midi,(40,45))
        self.assertEqual(tab.silent_class,2)
        self.assertAlmostEqual(tab.frame_period,.02)
        self.lib.probe_mutate()
        self.assertEqual(piano.notes[0,0],0)
        self.assertEqual(chords.spans[0,0],0)
        self.assertEqual(beats.events[0,0],0)
        self.assertEqual(tab.emissions[0,0,0],0)

    def test_empty_music_and_native_failures(self):
        self.lib.probe_mode(1)
        self.assertEqual(self.session.piano(self.pcm).notes.shape,(0,4))
        self.assertEqual(self.session.chords(self.pcm).spans.shape,(0,4))
        self.assertEqual(self.session.beats(self.pcm).events.shape,(0,2))
        self.assertEqual(self.session.tab(self.pcm).emissions.shape,(0,2,3))
        self.lib.probe_mode(2)
        for method in ('piano','chords','beats','tab'):
            with self.assertRaises(RuntimeError):
                getattr(self.session,method)(self.pcm)
        with self.assertRaises(RuntimeError):
            self.session.beats(self.pcm,16000)

    def test_shape_validation_prevents_native_reads(self):
        before=self.lib.probe_calls()
        for method in ('piano','chords','beats','tab'):
            with self.assertRaises(ValueError):
                getattr(self.session,method)(np.zeros((2,2)))
        self.assertEqual(self.lib.probe_calls(),before)

    def test_conversion_and_voice_controls(self):
        audio = self.session.convert(np.arange(6).reshape(3,2),self.pcm)
        self.lib.probe_mutate()
        np.testing.assert_array_equal(audio,np.arange(6))
        replay = self.session.convert(np.arange(6).reshape(3,2), self.pcm,
                                      noise_zp=np.arange(6).reshape(2,3),
                                      noise_sine=np.arange(12).reshape(3,4))
        np.testing.assert_array_equal(replay,np.arange(6))
        with self.assertRaises(ValueError):
            self.session.convert(np.arange(6).reshape(3,2),self.pcm,noise_sine=np.zeros((3,3)))
        self.assertEqual(self.session.convert_sample_rate(),40000)
        with self.assertRaises(ValueError):
            self.session.convert(np.zeros((3,3)),self.pcm)
        with self.assertRaises(ValueError):
            self.session.convert(np.zeros((3,2)),self.pcm,speaker_id=1)
        self.session.set_voice_samples(self.pcm,24000,'hello')
        self.session.tada_set_makeref_models()
        self.session.set_tts_cfg_scale(1.5)
        with self.assertRaises(RuntimeError):
            self.session.set_tts_cfg_scale(2)
        self.lib.probe_mode(2)
        with self.assertRaises(RuntimeError):
            self.session.accept_marking_responsibility()

    def test_allocated_utf8_text_and_empty_strings_are_freed(self):
        for cls, expected in ((crispasr.PuncModel,'Hello!'),(crispasr.TruecaseModel,'Grüße'),(crispasr.PCSModel,'Hello.')):
            with cls('probe',lib_path=self.path) as model:
                self.assertEqual(model.process('hello'),expected)
            with self.assertRaises(RuntimeError):
                model.process('closed')
        self.assertEqual(self.session.translate_text('hello','en','ja'),'こんにちは')
        self.lib.probe_mode(1)
        self.assertEqual(self.session.translate_text('','en','ja'),'')
        with crispasr.TruecaseModel('probe',lib_path=self.path) as model:
            self.assertEqual(model.process(''),'')

    def test_pcm_ownership_on_success_and_empty_allocations(self):
        for method in ('get_disclaimer_pcm','synthesize','synthesize_raw'):
            args=() if method=='get_disclaimer_pcm' else ('test',)
            arr=getattr(self.session,method)(*args)
            np.testing.assert_array_equal(arr,[.25,.5])
            self.assertEqual(self.lib.probe_live(),0)
            self.lib.probe_mode(1)
            with self.assertRaises(RuntimeError):
                getattr(self.session,method)(*args)
            self.assertEqual(self.lib.probe_live(),0)
            self.lib.probe_mode(0)
        self.assertEqual(self.session.disclaimer_text(),'AI generated')

    def test_speech_to_speech_frees_both_outputs_on_all_exits(self):
        audio,text=self.session.speech_to_speech(self.pcm,language="de")
        np.testing.assert_array_equal(audio,[.25,.5])
        self.assertEqual(text,'hello')
        self.assertEqual(self.lib.probe_live(),0)
        self.lib.probe_mode(1)
        with self.assertRaises(RuntimeError): self.session.speech_to_speech(self.pcm)
        self.assertEqual(self.lib.probe_live(),0)
        self.lib.probe_mode(2)
        with self.assertRaises(UnicodeDecodeError): self.session.speech_to_speech(self.pcm)
        self.assertEqual(self.lib.probe_live(),0)

    def test_streaming_pcm_copies_and_exception_transport(self):
        chunks=[]
        self.session.synthesize_streaming('one. two.',lambda pcm,final:chunks.append((pcm,final)))
        self.assertEqual([final for _,final in chunks],[False,True])
        self.assertEqual(chunks[0][0][0],.25)
        def bad(*args): raise ValueError('callback sentinel')
        with self.assertRaisesRegex(ValueError,'callback sentinel'):
            self.session.synthesize_streaming('test',bad)
        self.lib.probe_mode(2)
        with self.assertRaisesRegex(RuntimeError,'probe synthesis failure'):
            self.session.synthesize_streaming('test',lambda *_:None)

    def test_registered_callbacks_raw_bytes_and_exception_transport(self):
        segments,tokens,progress=[],[],[]
        self.session.set_progress_callback(lambda *a:progress.append(a))
        self.session.set_segment_callback(lambda *a:segments.append(a))
        self.session.set_token_callback(lambda *a:tokens.append(a))
        gc.collect()
        self.lib.probe_callbacks()
        self.assertEqual(segments,[('日本語',1.23,4.56,7)])
        self.assertEqual(progress,[(3,6)])
        self.assertEqual(tokens,[(b'\xe6\x97',9)])
        self.assertEqual(self.session.drain_streamed_tokens(),[b'\xe6\x97',b'\xa5',b'',b'end'])
        def bad(*args): raise ValueError('registered sentinel')
        self.session.set_segment_callback(bad)
        self.lib.probe_callbacks()
        with self.assertRaisesRegex(ValueError,'registered sentinel'):
            self.session.raise_callback_error()
        self.session.raise_callback_error()
        self.session.set_segment_callback(None)
        self.lib.probe_callbacks()

    def test_capability_queries_and_public_exports(self):
        self.assertEqual(crispasr.backend_capabilities(lib_path=self.path),{'foo':('asr','tts'),'bar':()})
        self.assertEqual(crispasr.backend_capabilities('foo',lib_path=self.path),('asr','tts'))
        with self.assertRaises(ValueError): crispasr.backend_capabilities('unknown',lib_path=self.path)
        self.assertEqual(crispasr.c_api_version(lib_path=self.path),'probe')
        self.assertEqual(crispasr.dart_helpers_version(lib_path=self.path),'probe-helpers')
        crispasr.set_gpu_backend(None,lib_path=self.path)
        for name in crispasr.__all__: self.assertTrue(hasattr(crispasr,name),name)

    def test_load_options_preserve_native_struct_layout_and_pointer_width(self):
        from crispasr._session_extra import function
        def setup(session):
            function(session._lib, 'crispasr_session_open_with_params', [C.c_char_p,C.c_char_p,C.c_void_p], C.c_void_p)
            function(session._lib, 'crispasr_session_backend', [C.c_void_p], C.c_char_p)
            function(session._lib, 'crispasr_session_close', [C.c_void_p], None)
        with patch.object(crispasr.Session,'_setup_session_signatures',setup):
            with crispasr.Session('probe',lib_path=self.path,n_threads=7,use_gpu=False,
                                  verbosity=2,flash_attn=False,n_gpu_layers=3) as s:
                self.assertEqual(s._handle,0x100000001)
                self.assertEqual(s.backend,'probe')

    def test_callback_reentry_and_closed_stream_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'close a session'):
            self.session.synthesize_streaming('test',lambda *_: self.session.close())
        with self.assertRaisesRegex(RuntimeError,'re-enter'):
            self.session.synthesize_streaming('test',lambda *_: self.session.piano(self.pcm))
        stream=crispasr.Session._Stream(self.lib,1)
        self.assertEqual(stream.feed(self.pcm),1)
        with self.assertRaises(ValueError): stream.feed(np.zeros((2,2)))
        stream.close()
        for method in ('feed','flush','get_text'):
            with self.assertRaisesRegex(RuntimeError,'closed'):
                getattr(stream,method)(*([self.pcm] if method=='feed' else []))
        stream.close()

    @unittest.skipUnless(os.environ.get('CRISPASR_LIB'),'complete native library supplied by CI')
    def test_real_library_exports_and_null_dimension_contracts(self):
        from crispasr._session_extra import function
        lib=C.CDLL(os.environ['CRISPASR_LIB'])
        for name in ('content_dim','n_speakers','sample_rate','inter_channels','upsample_product'):
            fn=function(lib,'crispasr_session_convert_'+name,[C.c_void_p],C.c_int)
            self.assertEqual(fn(None),0)
        self.assertTrue(crispasr.c_api_version(lib_path=os.environ['CRISPASR_LIB']))
        self.assertIn('whisper',crispasr.backend_capabilities(lib_path=os.environ['CRISPASR_LIB']))

    def test_legacy_compiled_struct_adapters_tokens_and_parameter_lifetime(self):
        path=Path(self.tmp.name)
        (path/'crispasr.h').write_text((ROOT/'tests/fixtures/python-wrapper-legacy.h').read_text())
        legacy=str(path/'legacy.so')
        subprocess.run([os.environ.get('CC') or 'cc','-shared','-fPIC','-std=c11','-Werror',
                        '-I'+str(path),'-I'+str(ROOT/'include'),
                        str(ROOT/'tests/fixtures/python-wrapper-legacy.c'),
                        str(ROOT/'python/crispasr/_helpers.c'),'-o',legacy],check=True)
        options={'initial_prompt':'Grüße','token_timestamps':True}
        with crispasr.CrispASR('probe',lib_path=legacy,helpers_lib_path=legacy,dtw=True,dtw_n_top=2) as model:
            self.assertEqual(model._ctx,0x100000001)
            for workers in (1,3):
                segments=model.transcribe_pcm(self.pcm,language='de',n_processors=workers,decode_options=options)
                self.assertEqual(segments[0].text,'日本語')
                self.assertEqual(segments[0].start,1.23)
                self.assertEqual(segments[0].no_speech_prob,.125)
                tokens=segments[0].tokens
                self.assertEqual(b''.join(t.raw_text for t in tokens),'日'.encode())
                self.assertIsNone(tokens[0].start)
                self.assertEqual(tokens[1].dtw_time,.42)
                self.assertEqual(tokens[0].alternatives,[crispasr.TokenAlternative('x'*300,.25,1234)])
            with self.assertRaises(ValueError):
                model.transcribe_pcm(self.pcm,decode_options={'unknown':True})
            with self.assertRaises(IndexError): model.tokens(-1)
            with self.assertRaises(IndexError): model.token_dtw_time(0,2)
            model._helpers=None
            with self.assertRaisesRegex(RuntimeError,'requires native pointer wrappers or libcrispasr_helpers'):
                model.transcribe_pcm(self.pcm)
            self.assertEqual(model._lib.probe_live(),0)

    def test_replay_dimensions_from_literal_native_getters(self):
        from test_session_utf8 import definition
        native=(ROOT/'src/rvc_svc.cpp').read_text()
        abi=(ROOT/'src/crispasr_c_api.cpp').read_text()
        code='#include <vector>\n#define CA_HAVE_RVC_SVC\n#define CA_EXPORT extern "C"\n'
        code+=definition(native,'struct rvc_hparams {')+';\n'
        code+='struct rvc_svc_context { rvc_hparams hp; };\nstruct crispasr_session { rvc_svc_context* rvc_ctx; };\n'
        for name in ('inter_channels','upsample_product'):
            code+=definition(native,'int rvc_svc_'+name+'(')+'\n'
            code+=definition(abi,'CA_EXPORT int crispasr_session_convert_'+name+'(')+'\n'
        code+='''
extern "C" int probe_dimensions() {
    rvc_svc_context model;
    model.hp.inter=37;
    model.hp.upsample_rates={3,5,7};
    crispasr_session session{&model}, empty{nullptr};
    return crispasr_session_convert_inter_channels(&session)==37 &&
           crispasr_session_convert_upsample_product(&session)==105 &&
           crispasr_session_convert_inter_channels(nullptr)==0 &&
           crispasr_session_convert_upsample_product(&empty)==0;
}
'''
        path=Path(self.tmp.name)
        (path/'dimensions.cpp').write_text(code)
        subprocess.run([os.environ.get('CXX') or 'c++','-std=c++17','-shared','-fPIC',
                        str(path/'dimensions.cpp'),'-o',str(path/'dimensions.so')],check=True)
        self.assertEqual(C.CDLL(str(path/'dimensions.so')).probe_dimensions(),1)

    def test_partial_auxiliary_construction_cleanup_is_idempotent(self):
        for cls in (crispasr.TitaNet,crispasr.SpeakerEmbedder,crispasr.PyannoteCache,crispasr.Parakeet):
            model=cls.__new__(cls)
            model.close()
            model.close()

    def test_missing_and_closed_library_errors(self):
        original=self.session._lib
        self.session._lib=object()
        with self.assertRaisesRegex(RuntimeError,'unavailable'): self.session.piano_sample_rate()
        self.session._lib=original
        self.session._handle=None
        with self.assertRaisesRegex(RuntimeError,'closed'): self.session.piano_sample_rate()
        self.session._handle=1


if __name__ == '__main__':
    unittest.main()
