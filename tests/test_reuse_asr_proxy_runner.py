"""Run only tiny fake executables; no ASR model or real model binary is used."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

SOURCE = Path(__file__).resolve().parents[1]/'tools/kaggle/reuse-quality-cpu-20261112/asr_proxy_runner.py'
spec = importlib.util.spec_from_file_location('asr_proxy_runner',SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ASRProxyRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.artifacts = self.root/'artifacts'
        self.artifacts.mkdir()
        self.output = self.root/'proxy'
        self.binary = self.root/'fake-asr'
        self.model = self.root/'dummy-model.bin'
        self.model.write_bytes(b'not a real model')
        report = {'status':'complete','completed_cases':16,'cases':[]}
        for speaker in runner.SPEAKERS:
            clean = (.1*np.random.default_rng(speaker).standard_normal(80000)).astype('float32')
            clean_path = self.artifacts/f'speaker-{speaker}-clean.wav'
            sf.write(clean_path,clean,16000,subtype='FLOAT')
            for i,condition in enumerate(runner.CONDITIONS):
                case_id = f'speaker-{speaker}-{condition}'
                record = {'case_id':case_id,'speaker':speaker,'condition':condition,
                          'clean_sha256':runner.auditor.sha256_file(clean_path)}
                raw = clean if condition == 'clean' else clean+i*.01
                # Output identity deliberately permits safe content deduplication.
                for label,wave in [('input',raw),('output',clean)]:
                    path = self.artifacts/f'{case_id}-{label}.wav'
                    sf.write(path,wave,16000,subtype='FLOAT')
                    record[label+'_sha256'] = runner.auditor.sha256_file(path)
                (self.artifacts/(case_id+'.json')).write_text(json.dumps({**record,'stage_fingerprints':{}}))
                report['cases'].append(record)
        self.report = report
        self.write_report()
        self.fake_binary()

    def write_report(self):
        (self.artifacts/'quality-report.json').write_text(json.dumps(self.report))

    def fake_binary(self,mode='ok'):
        source = f'''#!{sys.executable} -S
from pathlib import Path
import sys,time
mode = {mode!r}
args = sys.argv[1:]
wav = Path(args[-1])
prefix = Path(args[args.index('-of')+1])
speaker = wav.name.split('-')[1]
if 'white-10db-input' in wav.name:
    if mode == 'fail':
        print('intentional fake error',file=sys.stderr)
        raise SystemExit(7)
    if mode == 'timeout':time.sleep(2)
text = 'speaker '+speaker+' reference\\n'
if mode == 'nondeterministic':
    counter_path = Path(__file__).with_suffix('.counter')
    count = int(counter_path.read_text())+1 if counter_path.exists() else 1
    counter_path.write_text(str(count))
    text = text.strip()+' iteration '+str(count)+'\\n'
if mode == 'empty-output' and wav.name.endswith('-output.wav'):text = ''
prefix.with_suffix('.txt').write_text(text)
print(text.strip())
print('output_txt: saving output to '+str(prefix.with_suffix('.txt')))
'''
        self.binary.write_text(source)
        self.binary.chmod(0o755)

    def execute(self,**kwargs):
        return runner.run_proxy(self.artifacts,self.binary,self.model,self.output,
                                binary_sha256=runner.auditor.sha256_file(self.binary),
                                model_sha256=runner.auditor.sha256_file(self.model),
                                dependency_inspector=lambda binary:{'fake-runtime.so':'test-digest'},
                                **kwargs)

    def test_sequential_mapping_dedup_and_clean_identity_reexecution(self):
        result = self.execute()
        self.assertEqual(result['status'],'complete')
        self.assertEqual(len(result['cases']),16)
        # Four clean references + four forced identity repeats + twelve noisy inputs.
        self.assertEqual(len(result['invocations']),20)
        for case in result['cases']:
            if case['condition'] == 'clean':
                self.assertFalse(case['input_mapping']['deduplicated'])
                self.assertNotEqual(case['input_mapping']['invocation_id'],case['reference_invocation_id'])
                self.assertTrue(case['clean_input_audio_identity'])
                self.assertTrue(case['clean_input_transcript_identity'])
            self.assertTrue(case['output_mapping']['deduplicated'])
            self.assertEqual(case['output_agreement']['wer'],0)
        self.assertEqual(list((self.output/'.pending').iterdir()),[])
        for path in (self.output/'logs').glob('*.txt'):
            self.assertNotIn(str(self.root),path.read_text())
        self.assertNotIn(str(self.root),json.dumps(result))
        self.assertTrue(all(x['exit_code']==0 for x in result['invocations']))
        first = result['invocations'][0]
        self.assertIn('-ng',first['command'])
        self.assertEqual(runner.auditor.sha256_file(self.output/first['transcript_file']),first['transcript_sha256'])

    def test_later_nonzero_exit_preserves_completed_case_receipts(self):
        self.fake_binary('fail')
        with self.assertRaisesRegex(runner.ProxyFailure,'nonzero-exit'):
            self.execute()
        report = json.loads((self.output/'proxy-report.json').read_text())
        self.assertEqual(report['status'],'failed')
        self.assertEqual(len(report['cases']),2)
        self.assertTrue((self.output/'cases/speaker-121-white-0db.json').exists())
        failed = report['invocations'][-1]
        self.assertEqual(failed['exit_code'],7)
        self.assertIn('intentional fake error',(self.output/failed['stderr_raw_file']).read_text())
        self.assertEqual(list((self.output/'.pending').iterdir()),[])

    def test_timeout_is_nonpassing_and_cleans_temporary_files(self):
        self.fake_binary()
        def controlled(command,**kwargs):
            self.assertEqual(kwargs['timeout'],.3)
            if 'white-10db-input' in command[-1]:
                raise runner.subprocess.TimeoutExpired(command,.3,output=b'partial output',stderr=b'timeout detail')
            return runner.subprocess.run(command,**{**kwargs,'timeout':10})
        with self.assertRaisesRegex(runner.ProxyFailure,'timeout'):
            self.execute(timeout=.3,executor=controlled)
        report = json.loads((self.output/'proxy-report.json').read_text())
        self.assertEqual(report['status'],'failed')
        self.assertEqual(len(report['cases']),2)
        self.assertEqual(report['invocations'][-1]['status'],'timeout')
        self.assertIsNone(report['invocations'][-1]['exit_code'])
        self.assertEqual(list((self.output/'.pending').iterdir()),[])

    def test_identical_clean_audio_with_different_transcript_fails(self):
        self.fake_binary('nondeterministic')
        with self.assertRaisesRegex(runner.ProxyFailure,'nondeterministic ASR transcript'):
            self.execute()
        report = json.loads((self.output/'proxy-report.json').read_text())
        self.assertEqual(report['status'],'failed')
        self.assertEqual(len(report['clean_references']),4)
        self.assertEqual(len(report['cases']),0)
        self.assertEqual(len(report['invocations']),5)
        self.assertEqual(list((self.output/'.pending').iterdir()),[])

    def test_partial_or_tampered_corpus_never_starts_asr(self):
        self.report['status']='partial-wall-budget'
        self.write_report()
        executor = mock.Mock()
        with self.assertRaisesRegex(runner.ProxyFailure,'incomplete'):
            self.execute(executor=executor)
        executor.assert_not_called()
        self.assertEqual(json.loads((self.output/'proxy-report.json').read_text())['status'],'failed')
        shutil.rmtree(self.output)
        self.report['status']='complete'
        self.write_report()
        with (self.artifacts/'speaker-121-white-0db-input.wav').open('ab') as stream:stream.write(b'tampered')
        with self.assertRaisesRegex(runner.ProxyFailure,'hash mismatch'):
            self.execute(executor=executor)
        executor.assert_not_called()

    def test_clean_audio_mismatch_fails_before_asr(self):
        path = self.artifacts/'speaker-121-clean-input.wav'
        sf.write(path,np.zeros(80000,dtype='float32'),16000,subtype='FLOAT')
        digest = runner.auditor.sha256_file(path)
        self.report['cases'][0]['input_sha256'] = digest
        self.write_report()
        receipt_path = self.artifacts/'speaker-121-clean.json'
        receipt = json.loads(receipt_path.read_text());receipt['input_sha256']=digest
        receipt_path.write_text(json.dumps(receipt))
        executor = mock.Mock()
        with self.assertRaisesRegex(runner.ProxyFailure,'differs from speaker reference audio'):
            self.execute(executor=executor)
        executor.assert_not_called()

    def test_preexisting_report_is_not_overwritten(self):
        self.output.mkdir()
        path = self.output/'proxy-report.json';path.write_text('{"prior":"preserve"}')
        with self.assertRaisesRegex(runner.ProxyFailure,'fresh directory'):
            self.execute()
        self.assertEqual(path.read_text(),'{"prior":"preserve"}')

    def test_missing_transcript_is_nonpassing_and_preserves_invocation(self):
        def no_file(command,**kwargs):
            return runner.subprocess.CompletedProcess(command,0,b'',b'')
        with self.assertRaisesRegex(runner.ProxyFailure,'missing-transcript'):
            self.execute(executor=no_file)
        report = json.loads((self.output/'proxy-report.json').read_text())
        self.assertEqual(report['invocations'][0]['status'],'missing-transcript')
        self.assertTrue((self.output/'invocations/000-clean-reference-121.json').exists())
        self.assertEqual(list((self.output/'.pending').iterdir()),[])


if __name__ == '__main__':
    unittest.main()
