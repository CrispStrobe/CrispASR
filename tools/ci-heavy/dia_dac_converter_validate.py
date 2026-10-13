#!/usr/bin/env python3
"""Validate isolated Dia converter integration without neural runtime changes."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(SCRATCH)
sys.path.insert(0, str(ROOT / 'tools'))
from dia_dac_export import digest

prior = json.loads((ROOT / 'docs/dia-dac-standalone-export-2026-10-12.json').read_text())
assert prior['source'] == 'a0b702d0d26b543a82bee689c69ac296d319e5b2'
assert prior['conclusion'] == 'success' and prior['worker_receipt']['codec_export_validation_pass']
assert len(prior['stages']['startup']) == 48 and len(prior['stages']['full']) == 40
assert all(row['status'] == 'PASS' for rows in prior['stages'].values() for row in rows)
report = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              source_worktree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
              source_files={name: digest(ROOT / name) for name in (
                  'tools/dia_dac_export.py', 'models/convert-dia-dac-to-gguf.py',
                  'tests/test_dia_codec_parameters.py', 'tests/test_dia_dac_export.py',
                  'tools/ci-heavy/dia_dac_converter_validate.py')},
              numerical_proof_run=38306172119, numerical_proof_receipt_sha256=digest(ROOT / 'docs/dia-dac-standalone-export-2026-10-12.json'),
              runs=[], production_defaults_changed=False, native_speech_acceptance=False,
              scope='Isolated converter integration: real original export must reproduce the byte-identical prior numerical candidate; no new neural, speech or GPU claim.')


def run(command, label):
    path = OUT / (label + '.log')
    with path.open('w') as log:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    report['runs'].append(dict(name=label, exit=result.returncode, log_sha256=digest(path)))
    (OUT / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(label, result.returncode, path.read_text()[-1000:], flush=True)
    assert result.returncode == 0, label


run([sys.executable, ROOT / 'tests/test_dia_dac_export.py', '-v'], 'original-module-and-cli-contracts')
from huggingface_hub import hf_hub_download
revision = '3233fbcb32be47761d2e736857b6d1a075b9ba7e'
template = hf_hub_download('cstr/dia-1.6b-GGUF', 'dac-44khz.gguf', revision=revision)
expected = prior['worker_receipt']['physical_export']
assert digest(template) == expected['template_sha256']
candidate = OUT / 'dac-output-f32.gguf'
run([sys.executable, ROOT / 'models/convert-dia-dac-to-gguf.py', '--template', template,
     '--output', candidate, '--precision', 'output-f32', '--receipt', OUT / 'physical-tensor-audit.json'], 'pinned-original-conversion')
audit = json.loads((OUT / 'physical-tensor-audit.json').read_text())
assert digest(candidate) == expected['export_sha256'] == audit['export_sha256']
assert candidate.stat().st_size == expected['export_bytes'] == 108697184
assert audit['byte_delta'] == 1312 and audit['tensors'] == expected['tensors']
report.update(byte_identical_validated_candidate=True, all_134_physical_tensors_exact=True,
              physical_export=audit, converter_integration_pass=True)
(OUT / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
print('DIA_DAC_CONVERTER_INTEGRATION_PASS; no native free-speech acceptance', flush=True)
