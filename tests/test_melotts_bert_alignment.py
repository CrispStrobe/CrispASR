"""Execute actual conditioning callers against original tokenizer/phone goldens.

No model or numerical graph. Melo 209145371cff8fc3bd60d7be902ea69cbdb7965a
distributes raw phones over WordPieces, then inserts blanks. Tokenizer goldens
follow HF BertNormalizer/BertPreTokenizer and atomic whole-word [UNK].
"""
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import unittest

from test_openvoice_cli_routing import function

ROOT = Path(__file__).resolve().parents[1]
PREAMBLE = r'''
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "core/bert_norm.h"
#include "core/bert_pretok.h"
'''
if (ROOT/'src/core/bert_words.h').exists():
    PREAMBLE += '#include "core/bert_words.h"\n'


def execute(code):
    if Path('/proc/meminfo').exists():
        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(memory['MemAvailable'].split()[0])*1024 < 3*1024**3 or os.getloadavg()[0] > 2*(os.cpu_count() or 1):
            raise RuntimeError('Insufficient resources before conditioning fixture compile')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with tempfile.TemporaryDirectory(prefix='melotts-conditioning-') as temp:
        root = Path(temp); (root/'probe.cpp').write_text(code)
        subprocess.run(['c++', '-std=c++17', '-I', str(ROOT/'src'),
                        str(root/'probe.cpp'), '-o', str(root/'probe')], check=True)
        subprocess.run([str(root/'probe')], check=True)


class Conditioning(unittest.TestCase):
    def test_actual_tokenizer_and_subword_counts(self):
        source = Path(os.environ.get('BERT_ENCODER_SOURCE', ROOT/'src/bert_encoder.cpp')).read_text()
        start = source.index('struct bert_tokenizer {')
        end = source.index('// ── BERT layer weights', start)
        tokenization = source[start:end]
        counts = function(source, 'extern "C" int bert_encoder_word_subtokens(')
        code = PREAMBLE + tokenization + '\nstruct bert_encoder_context{bert_tokenizer tok;};\n' + counts
        execute(code + r'''
int main(){
 bert_encoder_context ctx;
 ctx.tok.vocab={{"red",1},{"blue",2},{".",3},{"-",4},{"sea",5},{"##shells",6},{"cafe",7}};
 assert(bert_tokenize(ctx.tok,"Red blue.")==std::vector<int>({101,1,2,3,102}));
 assert(bert_tokenize(ctx.tok,"red-blue")==std::vector<int>({101,1,4,2,102}));
 assert(bert_tokenize(ctx.tok,"seashells")==std::vector<int>({101,5,6,102}));
 assert(bert_tokenize(ctx.tok,"redq")==std::vector<int>({101,100,102}));
 assert(bert_tokenize(ctx.tok,std::string("CAFÉ")+char(9)+"blue.")==std::vector<int>({101,7,2,3,102}));
 assert(bert_tokenize(ctx.tok,"")==std::vector<int>({101,102}));
 ctx.tok.vocab[std::string(101,'a')]=8;
 assert(bert_tokenize(ctx.tok,std::string(101,'a'))==std::vector<int>({101,100,102}));
 int* sub=nullptr;int n=bert_encoder_word_subtokens(&ctx,"seashells blue.",&sub);
 assert(n==3 && std::vector<int>(sub,sub+n)==std::vector<int>({2,1,1}));free(sub);
}
''')

    def test_actual_caller_distributes_before_blank_insertion(self):
        source = Path(os.environ.get('MELOTTS_SOURCE', ROOT/'src/melotts.cpp')).read_text()
        start = source.index('// Build per-BERT-token phone counts')
        end = source.index('// Expand BERT features using bert_word2ph', start)
        dump = source.find('dump_input_ids(ctx, "bert_word2ph"', start, end)
        if dump != -1:
            end = dump
        region = source[start:end]
        alignment = ROOT/'src/core/melotts_bert_alignment.h'
        include = '#include "core/melotts_bert_alignment.h"\n' if alignment.exists() else ''
        code = PREAMBLE + include + r'''
static void* g_melotts_ctx=nullptr;
int build(const std::vector<int>& word2ph,const std::vector<int>& splits,int n_bert_tokens,int T,
          std::vector<int>& result){
 int n_words=(int)splits.size();int* subtokens=(int*)malloc(sizeof(int)*n_words);
 if(n_words)memcpy(subtokens,splits.data(),sizeof(int)*n_words);
 float* bert_out=(float*)malloc(sizeof(float));
''' + region + r'''
 result=bert_word2ph;free(bert_out);return 1;
}
int main(){std::vector<int> result;
 // Original distribute_phone(5,2)=[3,2], then *2; punctuation has its own token.
 assert(build({3,10,2,2},{2,1},5,17,result));
 assert(result==std::vector<int>({3,6,4,2,2}));
 assert(build({3,6,2},{2},4,11,result));
 assert(result==std::vector<int>({3,4,2,2}));
 assert(build({3,2,2},{3},5,7,result));
 assert(result==std::vector<int>({3,2,0,0,2}));
 assert(!build({3,10,2},{2},3,15,result)); // Token count must match BERT.
 assert(!build({3,10,2},{2},4,99,result)); // No silent SEP padding of lost phones.
 assert(!build({3,10,2},{0},2,15,result));
}
'''
        execute(code)


if __name__ == '__main__':
    unittest.main()
