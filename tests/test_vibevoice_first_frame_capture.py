#!/usr/bin/env python3
"""Weight-free runner controls; fake producer does not prove native parity."""
import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class CaptureControl(unittest.TestCase):
    def test_diagnostic_nonzero_exit_and_isolation(self):
        spec = importlib.util.spec_from_file_location('capture', ROOT / 'tools/vibevoice_first_frame_capture.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix='vv-capture-control-') as tmp:
            root = Path(tmp)
            model = root/'dummy.gguf'; model.write_bytes(b'tiny weight-free runner control')
            module.EXPECTED = hashlib.sha256(model.read_bytes()).hexdigest()
            text = root/'text'; text.write_text('Hello')
            binary = root/'producer'
            binary.write_text('''#!/usr/bin/env python
import os,struct,sys
assert sys.argv[sys.argv.index("--backend")+1] == "vibevoice-1.5b"
from pathlib import Path
p=Path(os.environ['CRISPASR_VIBEVOICE_TTS_CAPTURE_FIRST_FRAME'])
(p/'capture_complete.bin').write_bytes(struct.pack('<iii',4,8,20))
sizes={'tts_prompt_ids':2,'tts_prompt_embeds':16,'tts_prefill_logits':16,'tts_frame0_selected_tokens':1,'tts_prefill_hidden':8,'tts_pos_condition_frame0':8,'tts_neg_condition_frame0':8,'tts_negative_input_frame0':8,'tts_noise_frame0':4,'tts_latent_frame0':4,'tts_frame0_positions':3,'tts_frame0_timesteps':20}
for i in range(20):
 for key,n in [('v_both',8),('v_cfg',4),('x0',4),('z',4)]: sizes[f'tts_frame0_{key}_s{i:03}']=n
for key,n in sizes.items(): (p/(key+'.bin')).write_bytes(b'\\0'*(n*4))
(p/'tts_prefill_dimensions.bin').write_bytes(struct.pack('<ii',2,16))
(p/'tts_frame0_positions.bin').write_bytes(struct.pack('<iii',0,2,1))
raise SystemExit(4)
''')
            binary.chmod(0o755)
            args = ['capture','--binary',str(binary),'--gguf',str(model),'--text-file',str(text),'--output',str(root/'result')]
            with patch.object(sys,'argv',args), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            import json
            receipt = json.loads((root/'result/capture.json').read_text())
            self.assertEqual(receipt['cli_returncode'],4)
            self.assertEqual(receipt['token_constraint'],0)
            self.assertEqual(receipt['base_source'],'a8151a2e8d7914571139dff1fd0a03620eaca051')
            with patch.object(sys,'argv',args), self.assertRaises(FileExistsError):
                module.main()
            producer_source = binary.read_text()
            # Essential prefill payloads must be present and exactly sized.
            for stage in ('tts_prompt_ids', 'tts_prompt_embeds', 'tts_prefill_logits', 'tts_frame0_selected_tokens'):
                for failure in ('missing', 'truncated'):
                    with self.subTest(stage=stage, failure=failure):
                        mutation = (f"(p/'{stage}.bin').unlink()" if failure == 'missing' else
                                    f"(p/'{stage}.bin').write_bytes(b'\\0')")
                        binary.write_text(producer_source.replace('raise SystemExit(4)', mutation + '\nraise SystemExit(4)'))
                        args[-1] = str(root / (stage + '-' + failure))
                        with patch.object(sys, 'argv', args), self.assertRaisesRegex(RuntimeError, 'missing/truncated'):
                            module.main()
            binary.write_text(producer_source)
            # Incomplete stage must not be accepted even with a completion marker.
            binary.write_text(binary.read_text().replace("raise SystemExit(4)","(p/'tts_latent_frame0.bin').write_bytes(b'')\nraise SystemExit(4)"))
            args[-1] = str(root/'bad')
            with patch.object(sys,'argv',args), self.assertRaisesRegex(RuntimeError,'truncated'):
                module.main()


if __name__ == '__main__':
    unittest.main()
