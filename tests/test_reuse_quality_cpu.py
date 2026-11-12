"""Model-free controls for offline speech-quality evidence collection."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

SOURCE = Path(__file__).resolve().parents[1]/'tools/kaggle/reuse-quality-cpu-20261112/quality_cpu.py'
spec = importlib.util.spec_from_file_location('quality_cpu', SOURCE)
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


class QualityCPUControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clean = (.2*np.sin(np.arange(16000)*.071)).astype('float32')

    def corpus(self):
        # Deliberately shuffled metadata and a too-short earlier male utterance.
        (self.root/'SPEAKERS.TXT').write_text('; header\n260 | M | test-clean\n237 | F | test-clean\n61 | M | test-clean\n121 | F | test-clean\n1 | F | train-clean-100\n')
        expected = {}
        for rel in quality.CORPUS_FILES:
            p = self.root/'test-clean'/rel
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, np.tile(self.clean, 5), 16000, subtype='PCM_16')
            expected[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
        short = self.root/'test-clean/61/70968/61-70968-0000.flac'
        sf.write(short, self.clean, 16000, subtype='PCM_16')
        return expected

    def test_official_selection_is_stable_and_ordered(self):
        expected = self.corpus()
        a = quality.select_corpus(self.root, expected)
        b = quality.select_corpus(self.root, dict(reversed(list(expected.items()))))
        self.assertEqual([x['speaker'] for x in a], [121,237,61,260])
        self.assertEqual(a, b)
        self.assertEqual({str(x['path'].relative_to(self.root/'test-clean')) for x in a}, set(expected))

    def test_pinned_corpus_rejects_tamper_and_missing_file(self):
        expected = self.corpus()
        p = self.root/'test-clean'/next(iter(expected))
        p.write_bytes(p.read_bytes()+b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            quality.select_corpus(self.root, expected)
        p.unlink()
        with self.assertRaises(FileNotFoundError):
            quality.select_corpus(self.root, expected)

    def test_selection_rejects_wrong_sample_rate_even_with_matching_hash(self):
        expected = self.corpus()
        rel = next(iter(expected))
        p = self.root/'test-clean'/rel
        sf.write(p, np.tile(self.clean, 5), 8000, subtype='PCM_16')
        expected[rel] = quality.sha(p)
        with self.assertRaisesRegex(ValueError, 'could not select'):
            quality.select_corpus(self.root, expected)

    def test_noise_scaling_determinism_and_pink_spectrum(self):
        for kind in ('white','pink'):
            for snr in (0,5,10):
                y = quality.make_noisy(self.clean, kind, snr, 579)
                np.testing.assert_array_equal(y, quality.make_noisy(self.clean, kind, snr, 579))
                self.assertTrue(np.isfinite(y).all())
                self.assertAlmostEqual(quality.metrics(self.clean,y)['snr_db'], snr, places=5)
            self.assertFalse(np.array_equal(y, quality.make_noisy(self.clean,kind,10,580)))
        noise = quality.make_noisy(self.clean,'pink',5,579)-self.clean
        power = abs(np.fft.rfft(noise))**2
        self.assertGreater(power[20:500].mean(), 5*power[4000:7000].mean())
        copy = quality.make_noisy(self.clean,None,None,579)
        np.testing.assert_array_equal(copy,self.clean)
        copy[0]=1
        self.assertNotEqual(copy[0],self.clean[0])
        for x,kind,snr in [(np.zeros(16000),'white',0),(self.clean,'unknown',5),(self.clean,'pink',np.nan)]:
            with self.assertRaises(ValueError):quality.make_noisy(x,kind,snr,579)

    def test_stage_fingerprints_check_shape_scale_layout_and_finiteness(self):
        a = np.arange(12,dtype='float32').reshape(3,4)
        f = quality.stage_fingerprint(a)
        self.assertEqual(f['shape'],[3,4])
        self.assertEqual(f['sha256'],hashlib.sha256(a.tobytes()).hexdigest())
        self.assertEqual(f,quality.stage_fingerprint(np.asfortranarray(a)))
        self.assertAlmostEqual(quality.stage_fingerprint(2*a)['norm'],2*f['norm'])
        for bad in (np.array([]),np.array([np.nan]),np.array([np.inf])):
            with self.assertRaises(ValueError):quality.stage_fingerprint(bad)
        names = quality.expected_stages()
        self.assertEqual(len(names),72)
        self.assertTrue({'raw_audio','frontend_magnitude','encoder_layer_5','block_time_29','block_29','amplitude','output_phase','waveform'} <= names)
        stages = {n:f for n in names}
        quality.validate_stages(stages)
        for changed in ({k:v for k,v in stages.items() if k!='block_29'}, {**stages,'unexpected':f}):
            with self.assertRaises(ValueError):quality.validate_stages(changed)

    def test_waveform_consistency_and_float_output_roundtrip(self):
        noisy = quality.make_noisy(self.clean,'white',5,579)
        quality.validate_waveforms(self.clean,noisy,noisy)
        p = self.root/'enhanced.wav'
        sf.write(p,noisy,16000,subtype='FLOAT')
        read,rate = sf.read(p,dtype='float32')
        self.assertEqual(rate,16000)
        np.testing.assert_array_equal(read,noisy)
        self.assertEqual(quality.metrics(self.clean,read),quality.metrics(self.clean,noisy))
        for bad in (noisy[:-1],noisy[:,None],np.full_like(noisy,np.nan)):
            with self.assertRaises(ValueError):quality.validate_waveforms(self.clean,noisy,bad)
        with self.assertRaises(ValueError):quality.metrics(np.full_like(noisy,np.nan),noisy)

    def test_failed_aggregate_checkpoint_preserves_completed_cases(self):
        report = {'status':'running','cases':[{'case_id':'earlier'}]}
        quality.save(self.root/'earlier.json',{'case_id':'earlier','complete':True})
        quality.save(self.root/'quality-report.json',report)
        original = (self.root/'quality-report.json').read_bytes()
        real_replace = Path.replace
        def fail_report(path,target):
            if Path(target).name == 'quality-report.json':raise OSError('simulated interruption')
            return real_replace(path,target)
        case = {'case_id':'next','stage_fingerprints':{'waveform':{'sha256':'test'}},'output_sha256':'output'}
        with mock.patch.object(Path,'replace',fail_report):
            with self.assertRaisesRegex(OSError,'simulated interruption'):
                quality.checkpoint_case(self.root,case,report)
        quality.record_failure(self.root,OSError('simulated interruption'))
        self.assertEqual((self.root/'quality-report.json').read_bytes(),original)
        self.assertTrue(json.loads((self.root/'earlier.json').read_text())['complete'])
        self.assertEqual(json.loads((self.root/'next.json').read_text()),case)
        self.assertEqual(json.loads((self.root/'failure.json').read_text())['error_type'],'OSError')
        self.assertEqual(len(report['cases']),1)
        updated = quality.checkpoint_case(self.root,case,report)
        self.assertEqual(len(updated['cases']),2)
        self.assertNotIn('stage_fingerprints',updated['cases'][-1])
        self.assertEqual(json.loads((self.root/'next.json').read_text()),case)

    def test_metrics_detect_scale_delay_and_zero_output(self):
        quality.metric_controls()


if __name__ == '__main__':
    unittest.main()
