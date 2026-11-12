"""Numerical and evidence-integrity tests without models or GGUF archives."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

SOURCE = Path(__file__).resolve().parents[1]/'tools/kaggle/reuse-quality-cpu-20261112/audit_saved_waveforms.py'
with mock.patch.object(sys, 'argv', ['unrelated-program', '--unknown-option']):
    spec = importlib.util.spec_from_file_location('saved_waveform_audit', SOURCE)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)


class SavedWaveformAuditTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root/'output'
        self.reference = np.random.default_rng(41).normal(size=16000)

    def fixture(self):
        rng = np.random.default_rng(579)
        self.clean = (.2*rng.standard_normal(audit.SAMPLES)).astype('float32')
        self.noisy = (self.clean+.03*rng.standard_normal(audit.SAMPLES)).astype('float32')
        self.enhanced = (.9*self.clean).astype('float32')
        for name, audio in zip(audit.INPUT_NAMES, (self.clean,self.noisy,self.enhanced)):
            sf.write(self.root/name,audio,16000,subtype='FLOAT')
        return {name:hashlib.sha256((self.root/name).read_bytes()).hexdigest() for name in audit.INPUT_NAMES}

    def loader(self, path):
        index = int(path.name.split('-')[1])
        start = audit.STARTS[index]
        return self.noisy[start:start+audit.WINDOW].copy(), self.enhanced[start:start+audit.WINDOW].copy()

    def test_import_has_no_cli_or_output_side_effects(self):
        before = set(self.root.iterdir())
        with mock.patch.object(sys,'argv',['foreign','--not-an-audit-option']), contextlib.redirect_stdout(io.StringIO()) as stdout:
            spec.loader.exec_module(audit)
        self.assertEqual(stdout.getvalue(),'')
        self.assertEqual(set(self.root.iterdir()),before)

    def test_scale_zero_and_constant_estimates_have_defined_metrics(self):
        result = audit.metrics(self.reference,.5*self.reference)
        self.assertAlmostEqual(result['snr_db'],6.020599913279624)
        self.assertAlmostEqual(result['gain_projection'],.5)
        self.assertAlmostEqual(result['rms_ratio'],.5)
        self.assertAlmostEqual(result['spectral_convergence'],.5)
        self.assertGreater(result['si_sdr_db'],250)
        zero = audit.metrics(self.reference,np.zeros_like(self.reference))
        self.assertIsNone(zero['si_sdr_db'])
        self.assertAlmostEqual(zero['snr_db'],0)
        self.assertAlmostEqual(zero['spectral_convergence'],1)
        constant = audit.metrics(self.reference,np.ones_like(self.reference))
        self.assertIsNone(constant['si_sdr_db'])
        self.assertEqual(constant['gain_projection'],0)
        # A nonconstant short signal remains numerically valid without STOI.
        self.assertIsNone(audit.metrics([-1,1],[0,0])['si_sdr_db'])

    def test_malformed_and_undefined_reference_metrics_rejected(self):
        for ref, est in (([],[]),([1,1],[1,2]),([0,0],[1,2]),([1,np.nan],[1,2]),
                         ([1,2],[1,np.inf]),([[1,2]],[[1,2]]),([1,2],[1])):
            with self.subTest(reference=ref,estimate=est), self.assertRaises(ValueError):
                audit.metrics(ref,est)

    def test_stoi_seed_is_deterministic_without_changing_caller_rng(self):
        def scorer(ref,est,rate,extended=False):
            self.assertEqual(rate,16000)
            return np.random.random()
        np.random.seed(17)
        expected_next = np.random.random()
        np.random.seed(17)
        first = audit.metrics(self.reference,self.reference,stoi_scorer=scorer)
        self.assertEqual(np.random.random(),expected_next)
        np.random.seed(91)
        second = audit.metrics(self.reference,self.reference,stoi_scorer=scorer)
        self.assertEqual(first,second)
        def broken(*args,**kwargs):
            np.random.random()
            raise RuntimeError('scorer failed')
        np.random.seed(17)
        with self.assertRaisesRegex(RuntimeError,'scorer failed'):
            audit.metrics(self.reference,self.reference,stoi_scorer=broken)
        self.assertEqual(np.random.random(),expected_next)
        with self.assertRaisesRegex(ValueError,'nonfinite metrics'):
            audit.metrics(self.reference,self.reference,stoi_scorer=lambda *a,**k:np.nan)

    def test_audit_checks_hashes_and_matching_window_inputs(self):
        hashes = self.fixture()
        report = audit.audit(self.root,self.output,expected_input_hashes=hashes,
                             reference_loader=self.loader,metric_function=audit.metrics)
        self.assertEqual(report['input_hashes'],hashes)
        self.assertEqual(len(report['segments']),7)
        self.assertEqual(len(report['windows']),3)
        self.assertEqual(len(report['overlaps']),2)
        self.assertEqual(json.loads((self.output/'saved-waveform-audit.json').read_text()),report)
        for index, entry in enumerate(report['windows']):
            raw, wave = self.loader(Path(f'window-{index}-reference.gguf'))
            self.assertTrue(entry['raw_input_matches'])
            self.assertEqual(entry['raw_audio_sha256'],hashlib.sha256(raw.tobytes()).hexdigest())
            self.assertEqual(entry['waveform_sha256'],hashlib.sha256(wave.tobytes()).hexdigest())
            saved, rate = sf.read(self.output/f'window-{index}-reference.wav',dtype='float32')
            self.assertEqual(rate,16000)
            np.testing.assert_array_equal(saved,wave)

    def test_hash_mismatch_fails_before_any_reference_load(self):
        hashes = self.fixture()
        with (self.root/audit.INPUT_NAMES[0]).open('ab') as stream:
            stream.write(b'changed trailing bytes')
        loader = mock.Mock(side_effect=AssertionError('must not load reference'))
        with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):
            audit.audit(self.root,self.output,expected_input_hashes=hashes,reference_loader=loader)
        loader.assert_not_called()
        with self.assertRaisesRegex(ValueError,'exactly the three input WAVs'):
            audit.audit(self.root,self.output,expected_input_hashes={})

    def test_bad_wav_shape_rate_or_samples_rejected_before_reference_load(self):
        self.fixture()
        for samples,rate in ((self.clean[:-1],16000),(self.clean[:,None].repeat(2,axis=1),16000),
                             (self.clean,8000),(np.full_like(self.clean,np.nan),16000)):
            sf.write(self.root/audit.INPUT_NAMES[0],samples,rate,subtype='FLOAT')
            loader = mock.Mock()
            with self.assertRaisesRegex(ValueError,'finite mono samples'):
                audit.audit(self.root,self.output,reference_loader=loader)
            loader.assert_not_called()

    def test_reference_mismatch_and_shape_are_explicit_errors(self):
        self.fixture()
        raw, wave = self.loader(Path('window-0-reference.gguf'))
        changed = raw.copy()
        changed[0] += 1
        for invalid_raw,invalid_wave in ((changed,wave),(raw[:-1],wave),(raw,wave[:-1]),
                                        (raw.astype('float64'),wave),(raw,np.full_like(wave,np.inf))):
            with self.assertRaises(ValueError):
                audit.validate_reference_window(invalid_raw,invalid_wave,raw)
        with self.assertRaisesRegex(ValueError,'differs'):
            audit.audit(self.root,self.output,reference_loader=lambda path:(changed,wave),metric_function=audit.metrics)
        self.assertFalse((self.output/'saved-waveform-audit.json').exists())

    def test_atomic_output_preserves_existing_receipt_on_failures(self):
        target = self.root/'receipt.json'
        audit.write_json(target,{'accepted':'prior'})
        original = target.read_bytes()
        with self.assertRaises(ValueError):
            audit.write_json(target,{'invalid':np.nan})
        with mock.patch.object(Path,'replace',side_effect=OSError('interrupted')):
            with self.assertRaisesRegex(OSError,'interrupted'):
                audit.write_json(target,{'candidate':'new'})
        self.assertEqual(target.read_bytes(),original)
        self.assertEqual(list(self.root.iterdir()),[target])


if __name__ == '__main__':
    unittest.main()
