"""Weight-free controls for offline delivery and strict corpus gate accounting."""
import base64
import gzip
import lzma
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import os

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT/'tools/kaggle/reuse-corpus-native-20261113'
spec = importlib.util.spec_from_file_location('corpus_native', KERNEL/'qualification.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
prepare_spec = importlib.util.spec_from_file_location('corpus_prepare', KERNEL/'prepare.py')
prepare = importlib.util.module_from_spec(prepare_spec)
prepare_spec.loader.exec_module(prepare)


def log():
    return '\n'.join(f'mkl-avx2{arm}/{name} PASS rel_rms=0 max_abs=0 cos_min=1 norm_ratio=1'
                     for arm in ('', '-conditioned') for name in module.EXPECTED_STAGES)


def payload(name, data=b'checked', link=False):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w') as archive:
        item = tarfile.TarInfo(name)
        item.size = len(data)
        if link:
            item.type, item.linkname = tarfile.SYMTYPE, '/outside'
        archive.addfile(item, io.BytesIO(data))
    data = output.getvalue()
    return dict(data=base64.b85encode(lzma.compress(data)).decode(),
                sha256=hashlib.sha256(data).hexdigest(), manifest={name: hashlib.sha256(b'checked').hexdigest()})


class OfflineCorpusControls(unittest.TestCase):
    def test_all_142_unique_stage_rows_required(self):
        self.assertEqual(module.parse_rows(log()), dict(unconditioned=71, conditioned=71))
        for bad in (log().replace('PASS','FAIL',1), '\n'.join(log().splitlines()[1:]),
                    log()+'\n'+log().splitlines()[0], log().replace('waveform','unknown',1)):
            with self.subTest(log=bad[:50]), self.assertRaises(ValueError):
                module.parse_rows(bad)

    def test_conditioned_cannot_replace_unconditioned(self):
        conditioned = '\n'.join(x for x in log().splitlines() if '-conditioned/' in x)
        with self.assertRaisesRegex(ValueError, 'unconditioned'):
            module.parse_rows(conditioned)

    def test_payload_rejects_traversal_absolute_and_links_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, link in (('../escape',False),('/absolute',False),('safe',True)):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'unsafe'):
                    module.unpack(payload(name,link=link), root)
            self.assertEqual(list(root.iterdir()), [])

    def test_payload_corruption_rejected(self):
        value = payload('safe')
        value['sha256'] = '0'*64
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'hash'):
            module.unpack(value, Path(directory))

    def test_payload_manifest_requires_exact_members_and_member_hashes(self):
        for mode in ('unlisted', 'missing', 'hash'):
            value = payload('safe')
            if mode == 'unlisted': value['manifest'].clear()
            if mode == 'missing': value['manifest']['absent'] = '0'*64
            if mode == 'hash': value['manifest']['safe'] = '0'*64
            with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                module.unpack(value, Path(directory))

    def test_duplicate_tar_members_rejected(self):
        value = payload('safe')
        stream = io.BytesIO(lzma.decompress(base64.b85decode(value['data'])))
        with tarfile.open(fileobj=stream, mode='a') as archive:
            item = tarfile.TarInfo('safe'); item.size = 7
            archive.addfile(item, io.BytesIO(b'checked'))
        data = stream.getvalue()
        value.update(data=base64.b85encode(lzma.compress(data)).decode(), sha256=hashlib.sha256(data).hexdigest())
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError,'duplicate'):
            module.unpack(value, Path(directory))

    def test_child_timeout_reaps_process_and_keeps_log(self):
        import sys
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/'child.log'
            with self.assertRaises(TimeoutError):
                module.run(['/bin/sh','-c','printf started; sleep 30'], log, timeout=.5)
            self.assertIn('started',log.read_text())
            self.assertIsNone(module.ACTIVE_CHILD)

    def test_gguf_unpinned_wheel_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory)/'wrong.whl'; wheel.write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError,'exact official'):
                prepare.gguf_files(wheel)

    def test_gguf_metadata_and_license_required(self):
        for version,license_text in (('9.0','MIT License'),('0.18.0','wrong')):
            with tempfile.TemporaryDirectory() as directory:
                wheel = Path(directory)/'test.whl'
                with zipfile.ZipFile(wheel,'w') as archive:
                    archive.writestr('gguf-0.18.0.dist-info/METADATA',f'Version: {version}\n')
                    archive.writestr('gguf-0.18.0.dist-info/licenses/LICENSE',license_text)
                with patch.object(prepare,'GGUF_WHEEL_SHA256',hashlib.sha256(wheel.read_bytes()).hexdigest()), self.assertRaisesRegex(ValueError,'version/license'):
                    prepare.gguf_files(wheel)

    @unittest.skipUnless(os.environ.get('REUSE_TEST_GGUF_WHEEL'), 'set REUSE_TEST_GGUF_WHEEL to the official wheel')
    def test_converter_child_imports_actual_bundled_gguf(self):
        import sys
        wheel = Path(os.environ['REUSE_TEST_GGUF_WHEEL'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name,data in prepare.gguf_files(wheel).items():
                destination=root/name; destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
            with patch.dict(os.environ, {'PYTHONPATH':''}):
                module.inherit_vendor(root)
                log=root/'import.log'
                module.run([os.environ.get('REUSE_TEST_CHILD_PYTHON',sys.executable),'-c','import gguf; print(gguf.__file__)'],log)
            self.assertIn(str(root/'vendor/gguf/__init__.py'),log.read_text())

    def test_pinned_file_selection_rejects_missing_mismatch_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'a').mkdir(); (root/'b').mkdir()
            digest = hashlib.sha256(b'exact').hexdigest()
            with self.assertRaises(ValueError):
                module.unique_file(root,'input.wav',digest)
            (root/'a/input.wav').write_bytes(b'exact')
            self.assertEqual(module.unique_file(root,'input.wav',digest),root/'a/input.wav')
            (root/'b/input.wav').write_bytes(b'exact')
            with self.assertRaises(ValueError):
                module.unique_file(root,'input.wav',digest)

    def test_metadata_requires_cpu_offline_and_existing_inputs_only(self):
        data = json.loads((KERNEL/'kernel-metadata.json').read_text())
        self.assertEqual(data['enable_gpu'],'false')
        self.assertEqual(data['enable_internet'],'false')
        self.assertEqual(len(data['dataset_sources']),1)
        self.assertEqual(len(data['kernel_sources']),1)

    def test_unprepared_source_refuses_execution(self):
        with self.assertRaisesRegex(ValueError,'prepare.py'):
            module.main()


if __name__ == '__main__':
    unittest.main()
