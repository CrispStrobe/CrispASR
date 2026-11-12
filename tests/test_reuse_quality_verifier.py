"""Model-free verification controls using small synthetic waveform artifacts."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import soundfile as sf

SOURCE = Path(__file__).resolve().parents[1]/'tools/kaggle/reuse-quality-cpu-20261112/verify_completed.py'
spec = importlib.util.spec_from_file_location('quality_verifier',SOURCE)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
kernel = verifier.kernel


def write(path,value):
    path.write_text(json.dumps(value,allow_nan=False))


def fake_stoi(*args,**kwargs):
    return .5


class QualityVerifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = tempfile.TemporaryDirectory()
        root = Path(cls.base.name)
        cls.terminal = {'kernel_slug':verifier.KERNEL_SLUG,'version':1,'status':'COMPLETE',
                        'script_sha256':verifier.RUNNING_SCRIPT_SHA256}
        report = {'status':'complete','completed_cases':16,'cases':[],
                  'script_sha256':verifier.RUNNING_SCRIPT_SHA256,'script_version':kernel.VERSION,
                  'bundled_source_revision':kernel.BUNDLED_SOURCE_REVISION,
                  'sample_rate':16000,'window_samples':80000,'model_files':kernel.MODEL_HASHES,
                  'bundle':{name:item['sha256'] for name,item in kernel.BUNDLE.items()},
                  'official_flac_sha256':kernel.CORPUS_FILES,
                  'official_openslr_test_clean_archive_md5':kernel.OFFICIAL_ARCHIVE_MD5,
                  'estoi_normalization_seed':579,'stoi_available':False,'selected':[]}
        for relative,digest in kernel.CORPUS_FILES.items():
            speaker = int(relative.split('/')[0])
            sex = 'F' if speaker in (121,237) else 'M'
            report['selected'].append({'speaker':speaker,'sex':sex,'path':'test-clean/'+relative,'source_sha256':digest})
            clean = (.1*np.random.default_rng(speaker).standard_normal(80000)).astype('float32')
            sf.write(root/f'speaker-{speaker}-clean.wav',clean,16000,subtype='FLOAT')
            clean_sha = verifier.auditor.sha256_file(root/f'speaker-{speaker}-clean.wav')
            for condition,kind,snr in [('clean',None,None),('white-0db','white',0),('white-10db','white',10),('pink-5db','pink',5)]:
                case_id = f'speaker-{speaker}-{condition}'
                raw = kernel.make_noisy(clean,kind,snr,579000+speaker)
                enhanced = (.9*clean+.01*raw).astype('float32')
                stages = {name:{'shape':verifier.expected_shape(name),'dtype':'float32','sha256':'0'*64,'norm':0,'max_abs':0}
                          for name in kernel.expected_stages()}
                stages['raw_audio'] = kernel.stage_fingerprint(raw)
                stages['waveform'] = kernel.stage_fingerprint(enhanced)
                receipt = {'case_id':case_id,'speaker':speaker,'sex':sex,'condition':condition,
                           'noise_seed':579000+speaker,'target_snr_db':snr,'stage_count':72,
                           'comparison_stage_fingerprints':71,'stage_fingerprints':stages,'clean_sha256':clean_sha,
                           'input_metrics':kernel.metrics(clean,raw),'output_metrics':kernel.metrics(clean,enhanced)}
                for label,wave in [('input',raw),('output',enhanced)]:
                    path = root/f'{case_id}-{label}.wav'
                    sf.write(path,wave,16000,subtype='FLOAT')
                    receipt[label+'_sha256'] = verifier.auditor.sha256_file(path)
                write(root/f'{case_id}.json',receipt)
                report['cases'].append({k:v for k,v in receipt.items() if k!='stage_fingerprints'})
        write(root/'quality-report.json',report)

    @classmethod
    def tearDownClass(cls):
        cls.base.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'artifacts'
        shutil.copytree(self.base.name,self.root)

    def verify(self):
        return verifier.verify(self.root,scorer=fake_stoi,terminal_receipt=self.terminal)

    def alter_case(self,edit,aggregate=False):
        path = self.root/'speaker-121-clean.json'
        receipt = json.loads(path.read_text())
        edit(receipt)
        write(path,receipt)
        if aggregate:
            path = self.root/'quality-report.json'
            report = json.loads(path.read_text())
            report['cases'][0] = {k:v for k,v in receipt.items() if k!='stage_fingerprints'}
            write(path,report)

    def test_complete_synthetic_artifacts_validate_and_group(self):
        result = self.verify()
        self.assertFalse(result['conclusive'])  # synthetic fixture has no official crop comparison
        self.assertEqual(result['verified_cases'],16)
        self.assertEqual(len(result['speakers']),4)
        self.assertEqual(len(result['conditions']),4)
        self.assertEqual(result['conditions']['white-0db']['metrics']['snr_db']['valid_pairs'],4)
        self.assertAlmostEqual(result['conditions']['white-0db']['metrics']['snr_db']['mean_input'],0,places=6)
        self.assertFalse(result['official_clean_crop_comparison'])

    def test_partial_error_or_wrong_terminal_version_never_pass(self):
        for changed in ({'status':'RUNNING'},{'status':'ERROR'},{'version':2},{'kernel_slug':'wrong'},
                        {'script_sha256':'0'*64}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                verifier.verify(self.root,terminal_receipt={**self.terminal,**changed})
        write(self.root/'failure.json',{'error':'interruption'})
        with self.assertRaisesRegex(ValueError,'failure.json'):
            self.verify()
        (self.root/'failure.json').unlink()
        p = self.root/'quality-report.json'
        report = json.loads(p.read_text());report['status']='partial-wall-budget';write(p,report)
        with self.assertRaisesRegex(ValueError,'complete 16-case'):
            self.verify()

    def test_duplicate_case_or_wrong_source_is_rejected(self):
        p = self.root/'quality-report.json'
        original = json.loads(p.read_text())
        for field,value in [('script_sha256','0'*64),('bundled_source_revision','wrong'),('official_flac_sha256',{})]:
            report = copy.deepcopy(original);report[field]=value;write(p,report)
            with self.subTest(field=field),self.assertRaises(ValueError):self.verify()
        report = copy.deepcopy(original);report['cases'][1]=report['cases'][0];write(p,report)
        with self.assertRaisesRegex(ValueError,'duplicate cases'):self.verify()

    def test_wav_tamper_is_detected(self):
        with (self.root/'speaker-121-clean-output.wav').open('ab') as stream:stream.write(b'tampered')
        with self.assertRaisesRegex(ValueError,'file SHA-256 mismatch'):self.verify()

    def test_wrong_stage_shape_or_missing_stage_is_detected(self):
        self.alter_case(lambda case:case['stage_fingerprints']['block_29'].update(shape=[1]))
        with self.assertRaisesRegex(ValueError,'stage shape mismatch'):self.verify()
        self.alter_case(lambda case:case['stage_fingerprints'].pop('block_29'))
        with self.assertRaisesRegex(ValueError,'incomplete original-graph'):self.verify()

    def test_raw_tensor_hash_must_match_saved_input(self):
        self.alter_case(lambda case:case['stage_fingerprints']['raw_audio'].update(sha256='0'*64))
        with self.assertRaisesRegex(ValueError,'waveform tensor hash mismatch'):self.verify()

    def test_metrics_recomputed_instead_of_trusting_consistent_receipts(self):
        self.alter_case(lambda case:case['output_metrics'].update(snr_db=100),aggregate=True)
        with self.assertRaisesRegex(ValueError,'numerical metric mismatch'):self.verify()

    def test_cli_failure_replaces_a_stale_passing_verdict(self):
        receipt = self.root/'terminal.json'
        write(receipt,{**self.terminal,'status':'RUNNING'})
        output = self.root/'verification.json'
        write(output,{'conclusive':True})
        with self.assertRaisesRegex(ValueError,'not terminal COMPLETE'):
            verifier.main(['--artifacts',str(self.root),'--official-corpus',str(self.root),
                           '--terminal-receipt',str(receipt),'--output',str(output)])
        self.assertFalse(json.loads(output.read_text())['conclusive'])

    def test_aggregate_and_individual_receipts_must_agree(self):
        self.alter_case(lambda case:case['output_metrics'].update(snr_db=100))
        with self.assertRaisesRegex(ValueError,'aggregate/case disagreement'):self.verify()


if __name__ == '__main__':
    unittest.main()
