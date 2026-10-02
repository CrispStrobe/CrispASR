"""Reject incomplete source controls and hidden gaps in 9B acceptance."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import index_echo_9b_acceptance as audit


class NineBOracleGuards(unittest.TestCase):
    def setUp(self):
        self.source = dict(complete=True, context_audio='zh-pause',
            parameter_dtypes={name:dict(parameter_elements={'torch.float32':count})
                              for name,count in audit.PARAMETERS.items()},
            cases={name:dict(target=lang, segments=[dict(start=0,end=1,text='原文\nTranslation')],
                             rows=[dict(parse_warn=0)]) for name,lang in audit.REQUIRED.items()})
        self.source['parameter_dtypes']['embedding_dtype']='torch.float32'
        self.source['cases']['multi-en']['rows']=[dict(parse_warn=0),dict(parse_warn=0,has_ctx=True),dict(__summary__={})]

    def test_partial_capture_cannot_become_gold(self):
        audit.source_cases(self.source)
        self.source['complete']=False
        with self.assertRaises(ValueError):audit.source_cases(self.source)
        self.source.pop('complete')
        with self.assertRaises(ValueError):audit.source_cases(self.source)

    def test_rejected_stress_and_missing_conditioning_are_not_alternatives(self):
        self.source['context_audio']='jfk-repeat'
        with self.assertRaises(ValueError):audit.source_cases(self.source)
        self.source['context_audio']='zh-pause'
        self.source['cases']['multi-en']['rows'][1]['has_ctx']=False
        with self.assertRaises(ValueError):audit.source_cases(self.source)

    def test_requested_dtype_is_not_an_effective_dtype_audit(self):
        self.source['parameter_dtypes']['llm']['parameter_elements']={'torch.bfloat16':8953803264}
        with self.assertRaises(ValueError):audit.source_cases(self.source)

    def test_missing_language_and_parse_warning_fail(self):
        bad=copy.deepcopy(self.source);bad['cases'].pop('zh-es')
        with self.assertRaises(ValueError):audit.source_cases(bad)
        self.source['cases']['zh-ja']['rows'][0]['parse_warn']=1
        with self.assertRaises(ValueError):audit.source_cases(self.source)

    def test_24_layer_or_skipped_stage_report_cannot_validate_32_layers(self):
        names=[f'encoder_layer_{i}' for i in range(32)]+[f'llm_block_{i}' for i in range(32)]+[f'aux_{i}' for i in range(11)]
        def report(selected):
            return ''.join(f'[PASS] {n} shape=[1] cos_min=1.000000\n relative_l2=0.000000\n' for n in selected)+'[PASS] first greedy token parity (1/1)\n[PASS] prompt_ids byte parity\n[PASS] cached greedy token parity (16/16)\n'
        audit.stage_report(report(names))
        audit.stage_report(report(names).replace('shape=[1] cos_min=', 'shape=[1]~llama_context: buffer stats\n cos_min='))
        audit.stage_report(report(names).replace('shape=[1]', 'shape~llama_context: buffer stats\n=[1]', 1))
        with self.assertRaises(ValueError):audit.stage_report(report([n for n in names if n!='llm_block_31']))
        with self.assertRaises(ValueError):audit.stage_report(report(names)+'[SKIP] missing activation\n')
        with self.assertRaises(ValueError):audit.stage_report(report(names).replace('(16/16)','(15/16)'))
        with self.assertRaises(ValueError):audit.stage_report(report(names).replace('cos_min=1.000000','cos_min=0.997000',1))
        with self.assertRaises(ValueError):audit.stage_report(report(names).replace('relative_l2=0.000000','relative_l2=0.021000',1))


if __name__=='__main__':unittest.main()
