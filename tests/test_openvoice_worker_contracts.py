#!/usr/bin/env python3
"""Bounded loader/environment checks; never start Kaggle or import Torch."""
import ast
import importlib.util
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
WORKER=ROOT/'tools/kaggle/openvoice-bake/openvoice_bake.py'


class WorkerContracts(unittest.TestCase):
    def test_host_loader_path_survives_native_bundle_prefix(self):
        spec=importlib.util.spec_from_file_location('openvoice_worker',WORKER)
        worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
        with patch.dict(os.environ,{'LD_LIBRARY_PATH':'/bundle:/driver',
                                   'CRISPASR_OV2_HOST_LD_LIBRARY_PATH':'/driver'},clear=True):
            self.assertEqual(worker.host_environment()['LD_LIBRARY_PATH'],'/driver')
        with patch.dict(os.environ,{'LD_LIBRARY_PATH':'/driver'},clear=True):
            self.assertEqual(worker.host_environment()['LD_LIBRARY_PATH'],'/driver')

    def test_resource_failure_stops_before_child_work(self):
        spec=importlib.util.spec_from_file_location('openvoice_worker',WORKER)
        worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
        healthy=dict(available_bytes=4*1024**3, scratch_free_bytes=4*1024**3,
                     load=[0.5,0.5,0.5],cpus=4,gpu_free_mib=[4096])
        for changed in [dict(available_bytes=2*1024**3),dict(scratch_free_bytes=1024**3),
                        dict(load=[9,1,1]),dict(gpu_free_mib=[1023]),dict(gpu_free_mib=[])]:
            with self.subTest(changed=changed), \
                    patch.object(worker,'resource_snapshot',return_value={**healthy,**changed}), \
                    patch.object(worker,'save') as save,patch.object(worker.subprocess,'run') as child:
                with self.assertRaisesRegex(RuntimeError,'headroom'):
                    worker.run(['unused-compiler'],'compile-model-delta')
                child.assert_not_called();save.assert_called_once()
                self.assertEqual(worker.RECEIPT['resource_checks'][-1]['phase'],'compile-model-delta')
        with patch.object(worker,'resource_snapshot',return_value=healthy),patch.object(worker,'save'):
            worker.guard_resources('healthy')

    def test_conditioning_bootstrap_matches_prepared_worker(self):
        import hashlib
        bootstrap=ROOT/'tools/kaggle/openvoice-conditioning/conditioning_bake.py'
        constants={node.targets[0].id:node.value.value for node in ast.parse(bootstrap.read_text()).body
                   if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
                   and isinstance(node.value,ast.Constant)}
        self.assertRegex(constants['WORKER_SOURCE'],r'^[0-9a-f]{40}$')
        self.assertEqual(constants['WORKER_SHA256'],hashlib.sha256(WORKER.read_bytes()).hexdigest())

    def test_read_only_validation_retains_checkpoints_without_hub_writes(self):
        spec=importlib.util.spec_from_file_location('openvoice_worker',WORKER)
        worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);model=root/'model.gguf';ref=root/'reference.gguf';license_file=root/'LICENSE'
            model.write_bytes(b'GGUF model probe');ref.write_bytes(b'GGUF original probe');license_file.write_text('MIT notice')
            hub=Mock()
            with patch.object(worker,'OUT',root),patch.object(worker,'PUBLISH',False):
                worker.checkpoint_model(hub,model,'original-model-revision',license_file)
                record=worker.checkpoint_reference(hub,ref,0,'unused/namespace','source attribution',license_file)
                self.assertEqual(hub.mock_calls,[])
                self.assertEqual(worker.RECEIPT['model_checkpoint']['sha256'],worker.sha(model))
                self.assertEqual(record['sha256'],worker.sha(ref))
                self.assertNotIn('published_revision',record)
                self.assertEqual((root/'reference-OpenVoice-LICENSE.txt').read_text(),'MIT notice')
                self.assertEqual((root/'reference-ATTRIBUTION.txt').read_text(),'source attribution')
                self.assertFalse(worker.RECEIPT['model_acceptance'])

    def test_requested_publication_denial_preserves_local_evidence_and_fails(self):
        spec=importlib.util.spec_from_file_location('openvoice_worker',WORKER)
        worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);model=root/'model.gguf';license_file=root/'LICENSE'
            model.write_bytes(b'GGUF probe');license_file.write_text('MIT notice')
            hub=Mock();hub.create_repo.side_effect=PermissionError('write denied')
            hub.upload_file.side_effect=PermissionError('write denied')
            with patch.object(worker,'OUT',root),patch.object(worker,'PUBLISH',True):
                with self.assertRaises(PermissionError):
                    worker.checkpoint_model(hub,model,'original-model-revision',license_file)
                with self.assertRaisesRegex(RuntimeError,'publication failed'):
                    worker.checkpoint_reference(hub,model,0,'unused/namespace','attribution',license_file)
                self.assertEqual(worker.RECEIPT['model_checkpoint']['sha256'],worker.sha(model))
                self.assertEqual(worker.RECEIPT['reference_checkpoints'][0]['publication_error'],'PermissionError')
                self.assertFalse(worker.RECEIPT['model_acceptance'])

    def test_embedded_python_phases_compile(self):
        tree=ast.parse(WORKER.read_text())
        phases=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.startswith('import sys,'):
                compile(node.value,'embedded-phase','exec');phases.append(node.value)
        self.assertEqual(len(phases),2)
        outer_imports=[alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names]
        self.assertNotIn('torch',outer_imports)
        # Modules may load Torch in these isolated child processes; never put
        # CUDA12.4's LD_LIBRARY_PATH into the child environment.


if __name__=='__main__':unittest.main()
