"""No-network controls for the one-shot post-refresh RE-USE launcher."""
import copy
import contextlib
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools/kaggle'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('deferred_launch', TOOLS / 'launch_reuse_followup.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
NOW = datetime(2026, 11, 14, 0, 5, tzinfo=timezone.utc)
QUOTA = {'allowed': 108000, 'used': 0, 'reserved': 0,
         'refresh': datetime(2026, 11, 21, tzinfo=timezone.utc)}
SNAPSHOT = {'target_exists': False, 'files': list(launcher.EVIDENCE),
            'producer': {'id': 123, 'version': 1, 'status': 'COMPLETE', 'last_run_time': 'fixed'}}


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.receipt = self.root / 'receipt.jsonl'
        self.remote = Mock()
        self.remote.identity.return_value = 'synthetic-owner'
        self.remote.quota.return_value = copy.deepcopy(QUOTA)
        self.remote.snapshot.return_value = copy.deepcopy(SNAPSHOT)
        self.remote.submit.return_value = 1
        self.script = b'frozen script'
        self.metadata = json.dumps({'id': '${KAGGLE_ACCOUNT}/' + launcher.TARGET}).encode()
        self.addCleanup(patch.stopall)
        patch.object(launcher, 'pinned_candidate', return_value=(self.script, self.metadata)).start()
        patch.object(launcher, 'git', return_value=str(self.root).encode()).start()

    def launch(self, now=NOW, receipt=None):
        launcher.launch(self.root, self.root, receipt or self.receipt, self.remote, 'synthetic-owner', now)

    def test_success_publishes_exact_script_once_with_requested_timeout(self):
        def submit(folder, seconds):
            self.assertEqual(seconds, 3600)
            self.assertEqual((folder / 'qualification.py').read_bytes(), self.script)
            meta = json.loads((folder / 'kernel-metadata.json').read_text())
            self.assertEqual(meta['id'], 'synthetic-owner/' + launcher.TARGET)
            return 1
        self.remote.submit.side_effect = submit
        self.launch()
        self.remote.submit.assert_called_once()
        self.assertEqual(self.remote.snapshot.call_count, 2)
        events = [json.loads(line) for line in self.receipt.read_text().splitlines()]
        self.assertEqual(events[-1], {'state': 'submitted', 'version': 1})
        self.assertNotIn('synthetic-owner', self.receipt.read_text())

    def test_before_refresh_and_expired_do_not_touch_remote_or_claim(self):
        for now in (NOW - timedelta(days=1), NOW + timedelta(days=1)):
            with self.subTest(now=now), self.assertRaisesRegex(launcher.Refusal, 'authorized UTC'):
                self.launch(now)
        self.assertEqual(self.remote.mock_calls, [])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_second_invocation_with_different_receipt_cannot_submit_again(self):
        self.launch()
        with self.assertRaisesRegex(launcher.Refusal, 'claim already exists'):
            self.launch(receipt=self.root / 'different.jsonl')
        self.remote.submit.assert_called_once()

    def test_identity_duplicate_wrong_version_running_and_missing_evidence_refuse(self):
        cases = ['identity', 'duplicate', 'version', 'running', 'missing-file']
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                remote = Mock()
                remote.identity.return_value = 'wrong' if case == 'identity' else 'synthetic-owner'
                snapshot = copy.deepcopy(SNAPSHOT)
                if case == 'duplicate': snapshot['target_exists'] = True
                if case == 'version': snapshot['producer']['version'] = 2
                if case == 'running': snapshot['producer']['status'] = 'RUNNING'
                if case == 'missing-file': snapshot['files'].pop()
                remote.snapshot.return_value = snapshot
                remote.quota.return_value = QUOTA
                with patch.object(launcher, 'git', return_value=str(root).encode()):
                    with self.assertRaises(launcher.Refusal):
                        launcher.launch(root, root, root/'receipt', remote, 'synthetic-owner', NOW)
                remote.submit.assert_not_called()

    def test_changing_producer_refuses_before_submission(self):
        changed = copy.deepcopy(SNAPSHOT)
        changed['producer']['last_run_time'] = 'changed'
        self.remote.snapshot.side_effect = [SNAPSHOT, changed]
        with self.assertRaisesRegex(launcher.Refusal, 'metadata changed'):
            self.launch()
        self.remote.submit.assert_not_called()

    def test_insufficient_reserved_or_stale_quota_and_api_failure_fail_closed(self):
        for change in ({'used': 108000}, {'reserved': 104000},
                       {'refresh': launcher.NOT_BEFORE}, {'allowed': float('nan')}, {'used': None}):
            with self.subTest(change=change), self.assertRaises(launcher.Refusal):
                launcher.quota_guard({**QUOTA, **change}, NOW)
        self.remote.quota.side_effect = RuntimeError('private signed URL/account/token')
        with self.assertRaises(launcher.Refusal) as error:
            self.launch()
        self.assertNotIn('signed', str(error.exception))
        self.assertNotIn('signed', self.receipt.read_text())
        self.remote.submit.assert_not_called()
        self.remote.quota.assert_called_once()

    def test_uncertain_submission_consumes_claim_without_retry(self):
        self.remote.submit.side_effect = RuntimeError('private remote body')
        with self.assertRaisesRegex(launcher.Refusal, 'outcome is uncertain'):
            self.launch()
        with self.assertRaisesRegex(launcher.Refusal, 'claim already exists'):
            self.launch()
        self.remote.submit.assert_called_once()
        self.assertEqual(json.loads(self.receipt.read_text().splitlines()[-1])['stage'], 'submission')
        self.assertNotIn('private remote body', self.receipt.read_text())

    def test_bad_local_pin_does_not_consume_claim_or_call_remote(self):
        launcher.pinned_candidate.side_effect = launcher.Refusal('candidate script hash mismatch')
        with self.assertRaises(launcher.Refusal):
            self.launch()
        self.assertEqual(self.remote.mock_calls, [])
        self.assertEqual(list(self.root.iterdir()), [])


class AdapterTests(unittest.TestCase):
    def test_structured_quota_preserves_days_in_timedelta(self):
        remote = launcher.KaggleRemote()
        remote.api = Mock()
        remote.api.quota_view.return_value = SimpleNamespace(
            gpu_quota=SimpleNamespace(total_time_allowed=timedelta(days=1, seconds=21600),
                                      time_used=timedelta(seconds=68629), time_reserved=timedelta(0)),
            quota_refresh_time=QUOTA['refresh'])
        quota = remote.quota()
        self.assertEqual(quota['allowed'], 108000)
        self.assertEqual(quota['allowed']-quota['used']-quota['reserved'], 39371)
        launcher.quota_guard(quota, NOW)

    def test_exact_live_naive_sdk_refresh_restores_utc_and_stale_reset_still_refuses(self):
        remote = launcher.KaggleRemote()
        remote.api = Mock()
        response = SimpleNamespace(
            gpu_quota=SimpleNamespace(total_time_allowed=timedelta(days=1, seconds=21600),
                                      time_used=timedelta(seconds=68629), time_reserved=timedelta(0)),
            quota_refresh_time=datetime(2026, 11, 14, 0, 0, 0))
        remote.api.quota_view.return_value = response
        quota = remote.quota()
        self.assertEqual(quota['refresh'], datetime(2026, 11, 14, tzinfo=timezone.utc))
        self.assertIs(quota['refresh'].tzinfo, timezone.utc)
        with self.assertRaisesRegex(launcher.Refusal, 'has not refreshed'):
            launcher.quota_guard(quota, NOW)
        response.quota_refresh_time = datetime(2026, 11, 21, 0, 0, 0)
        launcher.quota_guard(remote.quota(), NOW)
        # Unknown callers cannot pass ambiguous naive timestamps to the guard.
        with self.assertRaisesRegex(launcher.Refusal, 'uncertain quota refresh time'):
            launcher.quota_guard({**QUOTA, 'refresh': datetime(2026, 11, 21)}, NOW)

    def test_unreviewed_sdk_refuses_before_authentication(self):
        with patch.object(launcher, 'package_version', return_value='new-version'), \
             patch.object(launcher, 'authenticated_api') as authenticate:
            with self.assertRaisesRegex(launcher.Refusal, 'SDK changed'):
                launcher.KaggleRemote().identity()
        authenticate.assert_not_called()

    def test_submission_requests_timeout_and_checks_returned_sources_identity_version(self):
        remote = launcher.KaggleRemote()
        remote.owner = 'synthetic-owner'
        remote.api = Mock()
        response = SimpleNamespace(error='', invalid_dataset_sources=[], invalid_kernel_sources=[],
                                   invalid_model_sources=[], ref='synthetic-owner/' + launcher.TARGET,
                                   kernel_id=5, version_number=1)
        remote.api.kernels_push.return_value = response
        self.assertEqual(remote.submit(Path('private-stage'), 3600), 1)
        remote.api.kernels_push.assert_called_once_with('private-stage', timeout='3600')
        response.invalid_kernel_sources = ['unavailable']
        with self.assertRaisesRegex(launcher.Refusal, 'attached source'):
            remote.submit(Path('private-stage'), 3600)
        response.invalid_kernel_sources = []
        response.version_number = 2
        with self.assertRaisesRegex(launcher.Refusal, 'unexpected'):
            remote.submit(Path('private-stage'), 3600)

    def test_snapshot_requires_complete_target_search_and_exact_current_producer(self):
        endpoint = Mock()
        item = SimpleNamespace(ref='synthetic-owner/' + launcher.PRODUCER, id=123,
                               current_version_number=1, last_run_time=NOW)
        endpoint.get_kernel.return_value = SimpleNamespace(metadata=item)
        endpoint.list_kernel_session_output.return_value = SimpleNamespace(
            files=[SimpleNamespace(file_name=name) for name in launcher.EVIDENCE], next_page_token='')
        client = SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=endpoint))
        remote = launcher.KaggleRemote()
        remote.api = Mock()
        remote.api.build_kaggle_client.side_effect = lambda: contextlib.nullcontext(client)
        remote.api.kernels_status.return_value = SimpleNamespace(status=SimpleNamespace(name='COMPLETE'))
        remote.api.kernels_list.return_value = []
        remote.request_factory = SimpleNamespace
        module = SimpleNamespace(ApiGetKernelRequest=SimpleNamespace)
        with patch.dict(sys.modules, {'kagglesdk.kernels.types.kernels_api_service': module}):
            snapshot = remote.snapshot('synthetic-owner')
            launcher.snapshot_guard(snapshot)
            self.assertEqual([call.args[0].kernel_slug for call in endpoint.get_kernel.call_args_list],
                             [launcher.PRODUCER])
            remote.api.kernels_list.assert_called_once_with(mine=True, search=launcher.TARGET, page=1, page_size=100)
            remote.api.kernels_list.return_value = [SimpleNamespace(ref='synthetic-owner/other')] * 100
            with self.assertRaisesRegex(launcher.Refusal, 'unavailable or truncated'):
                remote.snapshot('synthetic-owner')

    def test_missing_or_non_duration_quota_fails(self):
        remote = launcher.KaggleRemote()
        remote.api = Mock()
        remote.api.quota_view.return_value = SimpleNamespace(
            gpu_quota=SimpleNamespace(total_time_allowed='21600s', time_used=timedelta(0),
                                      time_reserved=timedelta(0)), quota_refresh_time=QUOTA['refresh'])
        with self.assertRaisesRegex(launcher.Refusal, 'quota schema'):
            remote.quota()


if __name__ == '__main__':
    unittest.main()
