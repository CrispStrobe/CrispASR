"""Keep MUSA SDK use and unpublished S5000 packaging within the reviewed scope."""
import copy
import json
from pathlib import Path
import re
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]


def check_policy(dockerfile, publisher, sdk_workflows):
    image = re.search(r'^ARG MUSA_SDK_IMAGE=(.+)$', dockerfile, re.M)
    assert image and re.fullmatch(
        r'registry\.mthreads\.com/mcconline/musa_sdk:5\.2\.0-devel-ubuntu22\.04-s5000@sha256:[0-9a-f]{64}',
        image[1]), 'SDK must be pinned to the tested S5000 profile'
    receipt = json.loads((ROOT / 'docs/historical/musa-sdk52-2026-10-26.json').read_text())
    assert image[1] == receipt['sdk_image'], 'SDK differs from the successful compiler probe'
    assert '-DMUSA_ARCHITECTURES=31' in dockerfile
    assert not re.search(r'-DMUSA_ARCHITECTURES=31[;0-9]', dockerfile)
    assert 'FROM ${MUSA_SDK_IMAGE} AS sdk' in dockerfile
    assert 'FROM sdk AS build' in dockerfile and 'FROM sdk AS runtime' in dockerfile
    for job in publisher['jobs'].values():
        configs = job.get('strategy', {}).get('matrix', {}).get('config', [])
        assert all('musa' not in str(c).lower() for c in configs), 'MUSA publication is not cleared'
    for workflow in sdk_workflows:
        triggers = workflow.get('on', workflow.get(True))
        assert set(triggers) == {'workflow_dispatch'}, 'SDK work must be manual'
        assert workflow['permissions'] == {'contents': 'read'}
        for job in workflow['jobs'].values():
            assert {'self-hosted', 'musa-s5000'} <= set(job['runs-on']), 'Requires MUSA platform'
            steps = str(job['steps'])
            assert '0x1ed5' in steps, 'Check actual platform before SDK use'
            assert 'sudo rm -rf' not in steps and 'docker system prune' not in steps
            for step in job['steps']:
                if step.get('uses', '').startswith('docker/build-push-action@'):
                    assert step['with']['push'] is False, 'Image publication is not cleared'


class MusaBuildPolicy(unittest.TestCase):
    def setUp(self):
        self.dockerfile = (ROOT / '.devops/main-musa.Dockerfile').read_text()
        self.publisher = yaml.safe_load((ROOT / '.github/workflows/docker.yml').read_text())
        self.workflows = [yaml.safe_load((ROOT / '.github/workflows' / n).read_text())
                          for n in ('probe-musa-sdk52.yml', 'validate-musa-image.yml')]

    def check(self):
        check_policy(self.dockerfile, self.publisher, self.workflows)

    def test_current_policy(self):
        self.check()

    def test_unpinned_sdk_rejected(self):
        self.dockerfile = re.sub(r'@sha256:[0-9a-f]{64}', '', self.dockerfile)
        with self.assertRaises(AssertionError):
            self.check()

    def test_unqualified_targets_rejected(self):
        self.dockerfile = self.dockerfile.replace('-DMUSA_ARCHITECTURES=31', '-DMUSA_ARCHITECTURES=31;22')
        with self.assertRaises(AssertionError):
            self.check()

    def test_mismatched_runtime_base_rejected(self):
        self.dockerfile = self.dockerfile.replace('FROM sdk AS runtime', 'FROM ubuntu:22.04 AS runtime')
        with self.assertRaises(AssertionError):
            self.check()

    def test_automatic_publication_rejected(self):
        job = next(iter(self.publisher['jobs'].values()))
        job['strategy']['matrix']['config'].append({'tag': 'main-musa'})
        with self.assertRaises(AssertionError):
            self.check()

    def test_hosted_sdk_builder_rejected(self):
        for workflow in self.workflows:
            bad = copy.deepcopy(workflow)
            next(iter(bad['jobs'].values()))['runs-on'] = 'ubuntu-22.04'
            with self.assertRaises(AssertionError):
                check_policy(self.dockerfile, self.publisher, [bad])

    def test_automatic_sdk_trigger_rejected(self):
        self.workflows[0][True]['push'] = {}
        with self.assertRaises(AssertionError):
            self.check()

    def test_self_hosted_image_publication_rejected(self):
        steps = next(iter(self.workflows[1]['jobs'].values()))['steps']
        next(s for s in steps if s.get('uses', '').startswith('docker/build-push-action@'))['with']['push'] = True
        with self.assertRaises(AssertionError):
            self.check()


if __name__ == '__main__':
    unittest.main()
