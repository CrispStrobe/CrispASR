"""Compile actual rotary op dispatch; parameter routing only, not model math."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RotaryDispatchTest(unittest.TestCase):
    def test_existing_portable_op_and_unchanged_default(self):
        text = (ROOT / 'src/dia_tts.cpp').read_text(); start = text.index('static ggml_tensor* dia_rotary(')
        end = text.index('\nstatic ggml_tensor* build_dia_encoder', start)
        driver = r'''
struct ggml_context {};
struct ggml_tensor {};
struct dia_model { ggml_tensor* rope_timescale; };
int classic=0, explicit_factors=0;
ggml_tensor *observed_factors=nullptr, *observed_value=nullptr, *observed_positions=nullptr;
ggml_tensor classic_result, factor_result;
ggml_tensor* ggml_rope(ggml_context*,ggml_tensor* value,ggml_tensor* positions,int dims,int mode) {
 if(dims!=128||mode!=2)return nullptr;
 ++classic; observed_value=value; observed_positions=positions;return &classic_result;
}
ggml_tensor* ggml_rope_ext(ggml_context*,ggml_tensor* value,ggml_tensor* positions,ggml_tensor* factors,
 int dims,int mode,int original_context,float base,float scale,float extension,float amplitude,float fast,float slow) {
 if(dims!=128||mode!=2||original_context!=0||base!=1.f||scale!=1.f||extension!=0.f||amplitude!=1.f||fast!=32.f||slow!=1.f)return nullptr;
 ++explicit_factors;observed_value=value;observed_positions=positions;observed_factors=factors;return &factor_result;
}
''' + text[start:end] + r'''
int main() {
 ggml_context ctx;ggml_tensor value,positions,factors;dia_model model{&factors};
 if(dia_rotary(&ctx,&value,&positions,128,2,model,false)!=&classic_result)return 1;
 if(classic!=1||explicit_factors!=0||observed_value!=&value||observed_positions!=&positions)return 2;
 if(dia_rotary(&ctx,&value,&positions,128,2,model,true)!=&factor_result)return 3;
 if(classic!=1||explicit_factors!=1||observed_factors!=&factors||observed_value!=&value||observed_positions!=&positions)return 4;
 return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='dia-rotary-dispatch-') as temp:
            root = Path(temp);cpp=root/'test.cpp';binary=root/'test';cpp.write_text(driver)
            subprocess.run(['c++','-std=c++17',str(cpp),'-o',str(binary)],check=True)
            subprocess.run([str(binary)],check=True)


if __name__ == '__main__':
    unittest.main()
