#!/usr/bin/env python3
"""Regenerate tests/fixtures/qwen3_stream_cases.json from the upstream Python.

Runs the real qwen_asr / Confucius4-R2T2 helper functions (extracted by AST,
so torch and vllm are never imported) that src/core/qwen3_stream.h ports, and
records their outputs for tests/test-qwen3-stream.cpp.
"""
import ast, json, re
from typing import Optional, Tuple
def grab(path, names):
    src = open(path, encoding='utf-8').read(); tree = ast.parse(src); out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.Assign)):
            nm = node.name if isinstance(node, ast.FunctionDef) else getattr(node.targets[0], 'id', None)
            if nm in names: out.append(ast.get_source_segment(src, node))
    return '\n\n'.join(out)
ns = {'re': re, 'Optional': Optional, 'Tuple': Tuple}
import os, sys
# usage: python tools/gen_qwen3_stream_cases.py <qwen_asr-0.0.6 src dir> <Confucius4-R2T2 checkout>
U = os.path.join(sys.argv[1], 'qwen_asr/inference/utils.py')
R = os.path.join(sys.argv[2], 'r2t2/r2t2_asr.py')
exec(grab(U, {'_ASR_TEXT_TAG','_LANG_PREFIX','normalize_language_name','detect_and_fix_repetitions','parse_asr_output'}), ns)
ra = {}; ra.update(ns)
exec(grab(R, {'_EN2ZH_PUNCT','_ZH2EN_PUNCT','_ALL_PUNCT_PAT','_normalize_punct_by_context','parse_language_output'}), ra)
cjk = lambda s: re.sub(r'(?<=[一-鿿])\s+(?=[一-鿿])', '', s)
inputs = [
  "", "   ", "language English<asr_text>Hello, world.", "language Chinese<asr_text>你好 世界 ,好 .",
  "language None<asr_text>", "language None<asr_text> stray", "no tag here  ", "　language german\n<asr_text> Guten Tag ",
  "language English<asr_text>" + "ha" * 30 + " end", "a" * 45 + "b", "x" + "abc" * 25 + "tail",
  "中文,测试.English, test!  (括号)", "价格是 3.5 元,好吗?", "\"quoted\",yes.", "é,ü.", "language French\r\n<asr_text>Bonjour, à tous.",
  "language English<asr_text>one|two", "  mixed 中 文 字 and  words ",
]
cases = []
for s in inputs:
    c = {"in": s}
    c["repfix"] = ns['detect_and_fix_repetitions'](s)
    c["parse_asr"] = list(ns['parse_asr_output'](s, None))
    c["parse_asr_forced_en"] = list(ns['parse_asr_output'](s, "English"))
    c["parse_lang"] = list(ra['parse_language_output'](s, None))
    c["parse_lang_forced_en"] = list(ra['parse_language_output'](s, "English"))
    c["punct"] = ra['_normalize_punct_by_context'](s)
    c["cjk"] = cjk(s)
    cases.append(c)
json.dump(cases, open('tests/fixtures/qwen3_stream_cases.json','w'), ensure_ascii=False, indent=0)
print(len(cases), "cases"); print(json.dumps(cases[3], ensure_ascii=False))
