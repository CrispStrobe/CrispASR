#!/usr/bin/env python3
"""Bounded loader/environment checks; never start Kaggle or import Torch."""
import ast
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

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
