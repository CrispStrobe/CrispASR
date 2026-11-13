"""Real opaque Whisper parameter adapters with a model-free decoder."""
import ctypes as C
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch
import numpy as np
from build_legacy_wrapper_probe import build
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from crispasr import CrispASR, Parakeet, TitaNet, SpeakerDB

class LegacyPointerContract(unittest.TestCase):
    def test_native_parameters_parallel_dtw_and_cleanup(self):
        with tempfile.TemporaryDirectory(prefix='python-legacy-') as tmp:
            path=build(Path(tmp)/'libcrispasr.so');lib=C.CDLL(str(path))
            with self.assertRaises(ValueError):CrispASR('probe',lib_path=str(path),dtw=True,dtw_aheads_preset=2)
            with self.assertRaises(RuntimeError):CrispASR('fail',lib_path=str(path))
            self.assertEqual(lib.probe_legacy_live(),0)
            model=CrispASR('probe',lib_path=str(path),dtw=True,dtw_n_top=2)
            self.assertEqual(lib.probe_legacy_dtw(),102)
            self.assertEqual(lib.probe_legacy_flash(),0)
            self.assertEqual(lib.probe_legacy_live(),0)
            # Exercise native pointer inference even on clones with a built helper.
            model._helpers = None
            pcm=np.array([1,3,5],dtype=np.float32)
            result=model.transcribe_pcm(pcm,language='de',n_processors=2,decode_options={'initial_prompt':'Grüße','token_timestamps':True})
            self.assertEqual(lib.probe_legacy_processors(),2);self.assertEqual(lib.probe_legacy_progress_zero(),1)
            self.assertEqual(result[0].text,'日');self.assertEqual(result[0].start,1)
            self.assertEqual(b''.join(t.raw_text for t in result[0].tokens),'日'.encode())
            self.assertIsNone(result[0].tokens[0].start)
            self.assertEqual(model.token_dtw_time(0,0),1.4)
            self.assertEqual(result[0].tokens[0].alternatives[0].token_id,42)
            self.assertEqual(lib.probe_legacy_live(),0)
            lib.probe_legacy_mode(2)
            with self.assertRaises(RuntimeError):model.transcribe_pcm(pcm,language='de')
            self.assertEqual(lib.probe_legacy_live(),0)
            lib.probe_legacy_mode(3)
            with self.assertRaises(RuntimeError):model.transcribe_pcm(pcm,language='de')
            self.assertEqual(lib.probe_legacy_live(),0)
            lib.probe_legacy_mode(0)
            with self.assertRaises(ValueError):model.transcribe_pcm(pcm,n_processors=0)
            self.assertEqual(lib.probe_legacy_live(),0)
            model.close();self.assertEqual(lib.probe_legacy_contexts(),0)

    def test_literal_dtw_helper_attention_policy(self):
        with tempfile.TemporaryDirectory(prefix='native-dtw-policy-') as tmp:
            lib=C.CDLL(str(build(Path(tmp)/'libcrispasr.so')))
            lib.whisper_context_default_params_by_ref.restype=C.c_void_p
            lib.whisper_free_context_params.argtypes=[C.c_void_p]
            lib.whisper_init_from_file_with_params_by_ref.argtypes=[C.c_char_p,C.c_void_p]
            lib.whisper_init_from_file_with_params_by_ref.restype=C.c_void_p
            lib.whisper_free.argtypes=[C.c_void_p]
            setter=lib.crispasr_ctx_params_set_dtw
            setter.argtypes=[C.c_void_p,C.c_bool,C.c_int,C.c_int];setter.restype=None
            setter(None,True,3,0)  # Null remains harmless.
            for sequence, expected_flash in [([False],1),([True],0),([True,False],0)]:
                params=lib.whisper_context_default_params_by_ref()
                self.assertTrue(params)
                try:
                    for enable in sequence:setter(params,enable,3,0)
                    ctx=lib.whisper_init_from_file_with_params_by_ref(b'probe',params)
                    self.assertTrue(ctx)
                    try:self.assertEqual(lib.probe_legacy_flash(),expected_flash)
                    finally:lib.whisper_free(ctx)
                finally:lib.whisper_free_context_params(params)
                self.assertEqual(lib.probe_legacy_live(),0)
                self.assertEqual(lib.probe_legacy_contexts(),0)

    def test_parakeet_uses_int64_offset_and_copies_partial_tokens(self):
        with tempfile.TemporaryDirectory(prefix='python-parakeet-') as tmp:
            path=Path(tmp)/'libcrispasr.so'
            subprocess.run(['cc','-shared','-fPIC','-Werror','-I'+str(ROOT/'include'),str(ROOT/'tests/fixtures/wrapper-parakeet-legacy.c'),'-o',str(path)],check=True)
            lib=C.CDLL(str(path));model=Parakeet('probe',lib_path=str(path));pcm=np.array([1,3,5],dtype=np.float32)
            result=model.transcribe(pcm,t_offset_cs=123)
            self.assertEqual(result['text'],'日本語');self.assertEqual(result['words'][0][1],223)
            self.assertEqual(result['tokens'][0][1],-1)
            self.assertEqual(b''.join(result['token_bytes']),'日'.encode());self.assertEqual(lib.probe_parakeet_results(),0)
            with self.assertRaises(ValueError):model.transcribe(pcm,language='de')
            with self.assertRaises(ValueError):model.transcribe(np.array([],dtype=np.float32))
            model.close();model.close();self.assertEqual(lib.probe_parakeet_contexts(),0)
            with self.assertRaises(RuntimeError):model.transcribe(pcm)

    def test_titanet_requires_192_float_success_and_rejects_closed_context(self):
        with tempfile.TemporaryDirectory(prefix='python-titanet-') as tmp:
            path=Path(tmp)/'libcrispasr.so'
            subprocess.run(['cc','-shared','-fPIC','-Werror','-I'+str(ROOT/'include'),str(ROOT/'tests/fixtures/wrapper-titanet-legacy.c'),'-o',str(path)],check=True)
            lib=C.CDLL(str(path));model=TitaNet('probe',lib_path=str(path));pcm=np.array([1,3,5],dtype=np.float32)
            result=model.embed(pcm);self.assertEqual(result.shape,(192,));self.assertEqual(result[0],1)
            for mode in (2,3,4):
                lib.probe_titanet_mode(mode)
                with self.assertRaises(RuntimeError):model.embed(pcm)
            lib.probe_titanet_mode(0)
            with self.assertRaises(ValueError):model.embed([])
            model.close();model.close();self.assertEqual(lib.probe_titanet_live(),0)
            calls=lib.probe_titanet_calls()
            with self.assertRaises(RuntimeError):model.embed(pcm)
            self.assertEqual(lib.probe_titanet_calls(),calls)

    def test_speaker_database_roster_scores_enrollment_and_lifetime(self):
        with tempfile.TemporaryDirectory(prefix='python-speaker-db-') as tmp:
            path=Path(tmp)/'libcrispasr.so'
            subprocess.run(['cc','-shared','-fPIC','-Werror','-I'+str(ROOT/'include'),str(ROOT/'tests/fixtures/wrapper-speaker-db.c'),'-o',str(path)],check=True)
            lib=C.CDLL(str(path));emb=np.array([1,3,5],dtype=np.float32)
            before=lib.probe_db_calls()
            with self.assertRaises(ValueError):SpeakerDB('db','Alice,Bob',False,lib_path=str(path))
            with self.assertRaises(ValueError):SpeakerDB('db','',True,lib_path=str(path))
            self.assertEqual(lib.probe_db_calls(),before)
            db=SpeakerDB('db','Alice,Bob',True,lib_path=str(path));self.assertEqual(db.count,2)
            matched=db.match(emb);self.assertEqual(matched,('アリス',.875));self.assertEqual(lib.probe_db_live(),1)
            lib.probe_db_mode(1);self.assertEqual(db.match(emb,threshold=-1),(None,-1.0))
            self.assertEqual(matched[0],'アリス');lib.probe_db_mode(0)
            for invalid in ([],[[1,2]]):
                with self.assertRaises(ValueError):db.match(invalid)
                with self.assertRaises(ValueError):db.enroll('Alice',invalid)
            with self.assertRaises(ValueError):db.match(emb,threshold=float('nan'))
            self.assertTrue(db.enroll('Alice',emb));lib.probe_db_mode(3);self.assertFalse(db.enroll('Alice',emb))
            lib.probe_db_mode(2)
            with self.assertRaises(RuntimeError):SpeakerDB('db','Alice,Bob',True,lib_path=str(path))
            db.close();db.close();self.assertEqual(lib.probe_db_live(),0);before=lib.probe_db_calls()
            with self.assertRaises(RuntimeError):db.match(emb)
            with self.assertRaises(RuntimeError):db.enroll('Alice',emb)
            self.assertEqual(lib.probe_db_calls(),before)

    def test_speaker_vectors_reject_large_views_before_copy_and_dispatch(self):
        with tempfile.TemporaryDirectory(prefix='python-speaker-bounds-') as tmp:
            titan_path = Path(tmp) / 'titan.so'
            db_path = Path(tmp) / 'db.so'
            for path, fixture in ((titan_path, 'wrapper-titanet-legacy.c'),
                                  (db_path, 'wrapper-speaker-db.c')):
                subprocess.run(['cc', '-shared', '-fPIC', '-Werror', '-I' + str(ROOT / 'include'),
                                str(ROOT / 'tests/fixtures' / fixture), '-o', str(path)], check=True)
            titan = TitaNet('probe', lib_path=str(titan_path))
            db = SpeakerDB('db', 'Alice,Bob', True, lib_path=str(db_path))
            try:
                original_asarray = np.asarray
                def no_dtype_copy(value, *args, **kwargs):
                    if args or kwargs.get('dtype') is not None:
                        raise AssertionError('unexpected dtype copy')
                    return original_asarray(value)
                for dtype in (np.float32, np.float64):
                    oversized = np.broadcast_to(np.array([1], dtype=dtype), (2**31,))
                    for operation in (titan.embed, db.match, lambda v: db.enroll('Alice', v)):
                        with self.subTest(dtype=dtype, operation=operation):
                            with patch.object(np, 'asarray', side_effect=no_dtype_copy):
                                with patch.object(np, 'ascontiguousarray', side_effect=AssertionError('unexpected copy')):
                                    with self.assertRaises(ValueError):
                                        operation(oversized)
                for invalid in (np.array(1., dtype=np.float64), np.ones((2, 3), dtype=np.float64)):
                    for operation in (titan.embed, db.match, lambda v: db.enroll('Alice', v)):
                        with patch.object(np, 'ascontiguousarray', side_effect=AssertionError('unexpected copy')):
                            with self.assertRaises(ValueError):
                                operation(invalid)
                self.assertEqual(titan._lib.probe_titanet_calls(), 0)
                self.assertEqual(db._lib.probe_db_calls(), 1)  # constructor only
                # Both dtype conversion and logical ordering must survive valid views.
                for pcm in (np.arange(1, 7, dtype=np.float64)[::2],
                            np.array([5, 3, 1], dtype=np.float32)[::-1]):
                    self.assertEqual(titan.embed(pcm)[0], 1)
                    self.assertEqual(db.match(pcm), ('アリス', .875))
                    self.assertTrue(db.enroll('Alice', pcm))
            finally:
                titan.close()
                db.close()
            self.assertEqual(titan._lib.probe_titanet_live(), 0)
            self.assertEqual(db._lib.probe_db_live(), 0)

    def test_speaker_array_coercion_cannot_dispatch_after_close(self):
        with tempfile.TemporaryDirectory(prefix='python-speaker-reentry-') as tmp:
            titan_path = Path(tmp) / 'titan.so'
            db_path = Path(tmp) / 'db.so'
            for path, fixture in ((titan_path, 'wrapper-titanet-legacy.c'),
                                  (db_path, 'wrapper-speaker-db.c')):
                subprocess.run(['cc', '-shared', '-fPIC', '-Werror', '-I' + str(ROOT / 'include'),
                                str(ROOT / 'tests/fixtures' / fixture), '-o', str(path)], check=True)
            for name in ('embed', 'match', 'enroll', 'threshold', 'name'):
                with self.subTest(operation=name):
                    owner = (TitaNet('probe', lib_path=str(titan_path)) if name == 'embed' else
                             SpeakerDB('db', 'Alice,Bob', True, lib_path=str(db_path)))
                    counter = (owner._lib.probe_titanet_calls if name == 'embed' else owner._lib.probe_db_calls)
                    before = counter()
                    class ClosesOwner:
                        def __array__(self, dtype=None, copy=None):
                            owner.close()
                            return np.array([1, 3, 5], dtype=np.float64)
                    class ClosingThreshold(float):
                        def __float__(self):
                            owner.close()
                            return .7
                    class ClosingName(str):
                        def encode(self, *args, **kwargs):
                            owner.close()
                            return b'Alice'
                    try:
                        with self.assertRaisesRegex(RuntimeError, 'closed'):
                            if name == 'threshold':
                                owner.match([1, 3, 5], threshold=ClosingThreshold(.7))
                            elif name == 'name':
                                owner.enroll(ClosingName('Alice'), [1, 3, 5])
                            elif name == 'enroll':
                                owner.enroll('Alice', ClosesOwner())
                            else:
                                getattr(owner, name)(ClosesOwner())
                    finally:
                        owner.close()
                    self.assertEqual(counter(), before)
                    live = owner._lib.probe_titanet_live if name == 'embed' else owner._lib.probe_db_live
                    self.assertEqual(live(), 0)

if __name__=='__main__':unittest.main()
