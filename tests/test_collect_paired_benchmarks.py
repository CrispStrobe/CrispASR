"""Model-free checks of benchmark ordering, guards and failure retention."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from collect_paired_benchmarks import collect


class CollectorTest(unittest.TestCase):
    def config(self, root):
        source = root / 'input'; source.write_bytes(b'input')
        return {'input_file': str(source), 'identity': {'model_revision': 'r', 'precision': 'f16',
            'input_sha256': hashlib.sha256(b'input').hexdigest(), 'settings': {}, 'device': 'T4'},
            'commands': {'left': ['left'], 'right': ['right']}, 'warmups': 1, 'rounds': 4}

    def test_order_and_retained_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); calls = []
            def run(argv, **kwargs):
                calls.append(argv[0]); kwargs['stdout'].write(b'output')
                return Mock(returncode=0)
            result = collect(self.config(root), root/'out', runner=run, resource_guard=lambda _: {})
            self.assertEqual(calls, ['left','right','left','right','right','left','left','right','right','left'])
            self.assertEqual(result['left']['samples'], 4)
            ledger = json.loads((root/'out/ledger.json').read_text())
            self.assertTrue(ledger['complete']); self.assertEqual(len(ledger['events']), 10)
            self.assertTrue((root/'out/measured-0-left.stdout').is_file())

    def test_failure_is_retained_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = Mock(return_value=Mock(returncode=7))
            with self.assertRaises(RuntimeError):
                collect(self.config(root), root/'out', runner=run, resource_guard=lambda _: {})
            ledger = json.loads((root/'out/ledger.json').read_text())
            self.assertFalse(ledger['complete']); self.assertEqual(ledger['events'][0]['returncode'], 7)
            self.assertEqual(run.call_count, 1)

    def test_bad_input_and_guard_stop_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = self.config(root); run = Mock()
            config['identity']['input_sha256'] = 'a' * 64
            with self.assertRaises(ValueError):
                collect(config, root/'bad', runner=run, resource_guard=lambda _: {})
            config = self.config(root)
            def deny(_): raise RuntimeError('headroom')
            with self.assertRaises(RuntimeError):
                collect(config, root/'guarded', runner=run, resource_guard=deny)
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
