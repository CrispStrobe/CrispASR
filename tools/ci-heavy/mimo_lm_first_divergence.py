#!/usr/bin/env python3
"""Locate Q4 flash/eager drift within the LM; diagnostic BLAS control, no promotion.

Temporary native captures preserve tensors explicitly and stop after each selected
layer. Full canonical stages independently check that the patch's disabled path
still agrees with the pinned Python reference. Native A/B captures are not a new
original-checkpoint oracle or speech acceptance. Production sources are restored.
"""
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import gguf
import numpy as np
from huggingface_hub import hf_hub_download
from pr492_acceptance import ROOT, PINS, STAGES, digest, metrics, run

PARTS = ['input', 'attn_norm', 'attn_out', 'post_attn', 'ffn_norm', 'mlp_out', 'output']
LAYERS = [0, 1, 4, 8, 16, 24, 35]


def patch_source(original):
    changes = [
        ('struct mimo_asr_context {', '''static void mimo_lm_diag_blas_threads(ggml_backend_t be, int threads) {
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(be));
    auto fn = reinterpret_cast<void (*)(ggml_backend_t, int)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
    GGML_ASSERT(fn);
    fn(be, threads);
}
struct mimo_asr_context {'''),
        ('ggml_backend_t backend_cpu = nullptr;', 'ggml_backend_t backend_blas = nullptr;\n    ggml_backend_t backend_cpu = nullptr;'),
        ('        ggml_backend_t backends[2];\n        backends[n_be++] = ctx->backend;', '''        ggml_backend_t backends[3];
        if (std::getenv("MIMO_LM_DIAG_BLAS") && ctx->backend == ctx->backend_cpu) {
            ctx->backend_blas = ggml_backend_init_by_name("BLAS", nullptr);
            GGML_ASSERT(ctx->backend_blas);
            mimo_lm_diag_blas_threads(ctx->backend_blas, ctx->n_threads);
            fprintf(stderr, "MIMO_LM_DIAG_BLAS_ACTIVE backend=%s\\n", ggml_backend_name(ctx->backend_blas));
            backends[n_be++] = ctx->backend_blas;
        }
        backends[n_be++] = ctx->backend;'''),
        ('    if (ctx->backend_cpu)\n        ggml_backend_free(ctx->backend_cpu);', '    if (ctx->backend_blas)\n        ggml_backend_free(ctx->backend_blas);\n    if (ctx->backend_cpu)\n        ggml_backend_free(ctx->backend_cpu);'),
        ('    ggml_tensor* cur = inputs_embeds;\n    for (uint32_t il = 0; il < hp.llm_layers; il++) {', '''    const char* diag_layer_env = diag_captures ? std::getenv("MIMO_LM_DIAG_LAYER") : nullptr;
    const int diag_layer = diag_layer_env ? std::atoi(diag_layer_env) : -1;
    auto capture = [&](ggml_tensor* value, const char* part) {
        ggml_tensor* copy = ggml_cont(ctx0, value);
        char name[64];
        snprintf(name, sizeof(name), "lm_layer_%d_%s", diag_layer, part);
        ggml_set_name(copy, name);
        ggml_set_output(copy);
        ggml_build_forward_expand(gf, copy);
    };
    ggml_tensor* cur = inputs_embeds;
    for (uint32_t il = 0; il < hp.llm_layers; il++) {
        if ((int)il == diag_layer) capture(cur, "input");'''),
        ('        h = ggml_mul(ctx0, h, b.attn_norm_w);\n\n        // KV-cached', '        h = ggml_mul(ctx0, h, b.attn_norm_w);\n        if ((int)il == diag_layer) capture(h, "attn_norm");\n\n        // KV-cached'),
        ('            /*o_b*/ nullptr, /*qkv_b*/ b.attn_qkv_b);\n        cur = ggml_add(ctx0, residual, attn);', '''            /*o_b*/ nullptr, /*qkv_b*/ b.attn_qkv_b);
        if ((int)il == diag_layer) capture(attn, "attn_out");
        cur = ggml_add(ctx0, residual, attn);
        if ((int)il == diag_layer) capture(cur, "post_attn");'''),
        ('        h = ggml_mul(ctx0, h, b.ffn_norm_w);\n        ggml_tensor* mlp = core_ffn::swiglu(ctx0, h, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);\n        cur = ggml_add(ctx0, residual, mlp);\n    }\n\n    // Final norm', '''        h = ggml_mul(ctx0, h, b.ffn_norm_w);
        if ((int)il == diag_layer) capture(h, "ffn_norm");
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, h, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        if ((int)il == diag_layer) capture(mlp, "mlp_out");
        cur = ggml_add(ctx0, residual, mlp);
        if ((int)il == diag_layer) {
            capture(cur, "output");
            ggml_free(ctx0);
            return gf;
        }
    }

    // Final norm'''),
        ('    // Pull the requested stage tensor.\n', '''    // Dump all explicitly retained captures from this single computation.
    const char* directory = std::getenv("MIMO_LM_DIAG_DUMP");
    const char* layer = std::getenv("MIMO_LM_DIAG_LAYER");
    if (directory && layer) {
        const char* parts[] = {"input", "attn_norm", "attn_out", "post_attn", "ffn_norm", "mlp_out", "output"};
        for (const char* part : parts) {
            char name[64]; snprintf(name, sizeof(name), "lm_layer_%d_%s", std::atoi(layer), part);
            auto* tensor = ggml_graph_get_tensor(gf, name);
            GGML_ASSERT(tensor && tensor->type == GGML_TYPE_F32);
            std::vector<float> values((size_t)ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
            const std::string path = std::string(directory) + "/" + part + ".f32";
            FILE* file = fopen(path.c_str(), "wb"); GGML_ASSERT(file);
            const size_t written = fwrite(values.data(), sizeof(float), values.size(), file);
            GGML_ASSERT(written == values.size());
            GGML_ASSERT(fclose(file) == 0);
        }
    }
    // Pull the requested stage tensor.
'''),
    ]
    # Layer studies start from the SAME independently frozen fused LM input.
    changes.extend([
        ('    ggml_tensor* inputs_embeds = ggml_add(ctx0, text_embeds, x); // [d, T]',
         '''    ggml_tensor* inputs_embeds = ggml_add(ctx0, text_embeds, x); // [d, T]
    if (diag_captures && std::getenv("MIMO_LM_DIAG_LAYER")) {
        inputs_embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
        ggml_set_input(inputs_embeds);
        ggml_set_name(inputs_embeds, "lm_diag_input");
    }'''),
        ('    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {\n        fprintf(stderr, "mimo_asr_extract_stage: graph compute failed',
         '''    if (std::getenv("MIMO_LM_DIAG_LAYER")) {
        const char* path = std::getenv("MIMO_LM_DIAG_INPUT"); GGML_ASSERT(path);
        auto* tensor = ggml_graph_get_tensor(gf, "prefill_inputs_embeds");
        GGML_ASSERT(tensor && tensor->type == GGML_TYPE_F32);
        std::vector<float> values((size_t)ggml_nelements(tensor));
        FILE* file = fopen(path, "rb"); GGML_ASSERT(file);
        const size_t count = fread(values.data(), sizeof(float), values.size(), file);
        GGML_ASSERT(count == values.size() && fgetc(file) == EOF);
        GGML_ASSERT(fclose(file) == 0);
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    }
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "mimo_asr_extract_stage: graph compute failed'''),
    ])
    patched = original
    for index, (before, after) in enumerate(changes):
        if 4 <= index <= 7:
            start = patched.index('static ggml_cgraph* mimo_asr_build_prefill_graph(')
            end = patched.index('// 51b — Step decode graph', start)
            region = patched[start:end]
            assert region.count(before) == 1, (before[:100], region.count(before))
            patched = patched[:start] + region.replace(before, after) + patched[end:]
        else:
            assert patched.count(before) == 1, (before[:100], patched.count(before))
            patched = patched.replace(before, after)
    return patched


def worker(library, model, reference, destination, flash):
    out = Path(destination); out.mkdir(parents=True, exist_ok=True)
    lib = C.CDLL(library)
    class Params(C.Structure):
        _fields_ = [('n_threads', C.c_int), ('verbosity', C.c_int), ('use_gpu', C.c_bool),
                    ('temperature', C.c_float), ('flash_attn', C.c_bool)]
    lib.mimo_asr_context_default_params.restype = Params
    lib.mimo_asr_init_from_file.argtypes = [C.c_char_p, Params]
    lib.mimo_asr_init_from_file.restype = C.c_void_p
    lib.mimo_asr_extract_stage.argtypes = [C.c_void_p, C.POINTER(C.c_int32), C.c_int, C.c_char_p, C.POINTER(C.c_int)]
    lib.mimo_asr_extract_stage.restype = C.POINTER(C.c_float)
    lib.mimo_asr_free.argtypes = [C.c_void_p]
    libc = C.CDLL(None); libc.free.argtypes = [C.c_void_p]
    ref = {t.name: t.data for t in gguf.GGUFReader(reference).tensors}
    ids = np.ascontiguousarray(ref['prefill_input_ids'], dtype=np.int32)
    assert ids.ndim == 2 and ids.shape[0] == 9
    frozen = out/'frozen-input.f32'
    np.asarray(ref['prefill_inputs_embeds'],dtype=np.float32).tofile(frozen)
    params = lib.mimo_asr_context_default_params()
    params.n_threads, params.verbosity, params.use_gpu, params.flash_attn = 4, 1, False, flash == 'flash'
    ctx = lib.mimo_asr_init_from_file(os.fsencode(model), params); assert ctx
    rows = {}
    try:
        for layer in [-1] + LAYERS:
            stages = STAGES if layer == -1 else [f'lm_layer_{layer}_output']
            dest = out if layer == -1 else out / str(layer); dest.mkdir(exist_ok=True)
            if layer == -1:
                os.environ.pop('MIMO_LM_DIAG_LAYER', None); os.environ.pop('MIMO_LM_DIAG_DUMP', None)
            else:
                os.environ.update(MIMO_LM_DIAG_LAYER=str(layer), MIMO_LM_DIAG_DUMP=str(dest), MIMO_LM_DIAG_INPUT=str(frozen))
            for name in stages:
                started = time.perf_counter(); n = C.c_int()
                ptr = lib.mimo_asr_extract_stage(ctx, ids.ctypes.data_as(C.POINTER(C.c_int32)), ids.shape[1], name.encode(), C.byref(n))
                assert ptr and n.value > 0, name
                try: values = np.ctypeslib.as_array(ptr, (n.value,)).copy()
                finally: libc.free(ptr)
                assert np.isfinite(values).all()
                rows[name] = dict(seconds=time.perf_counter()-started, elements=n.value)
                if layer == -1:
                    np.save(out/(name+'.npy'), values)
                    rows[name]['python_reference'] = metrics(values, ref[name].reshape(-1))
                else:
                    assert np.array_equal(values, np.fromfile(dest/'output.f32',dtype=np.float32))
                    if layer == 0:
                        assert np.array_equal(np.fromfile(dest/'input.f32',dtype=np.float32),ref['prefill_inputs_embeds'].reshape(-1))
                print(name, rows[name], flush=True)
    finally: lib.mimo_asr_free(ctx)
    (out/'execution.json').write_text(json.dumps(rows,indent=2)+'\n')


def main():
    out = Path(os.environ['HEAVY_OUT']); out.mkdir(parents=True,exist_ok=True)
    scratch = Path(os.environ['HEAVY_SCRATCH'])/'mimo-lm-divergence'; scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(TMPDIR=str(scratch), HF_HOME=str(scratch/'hf'), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                      CRISPASR_GGUF_MMAP='1', CRISPASR_MIMO_FORCE_CPU='1')
    for key in ['MIMO_LM_DIAG_BLAS','MIMO_LM_DIAG_LAYER','MIMO_LM_DIAG_DUMP','CRISPASR_CORE_ATTN_EAGER_F32']:
        os.environ.pop(key,None)
    receipt = dict(source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),scope=__doc__,passed=False,pins=PINS,layers=LAYERS,comparisons={})
    def save(): (out/'lm-first-divergence.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    run(['bash',ROOT/'tools/ci-apt.sh','update'],out/'apt-update.log')
    run(['bash',ROOT/'tools/ci-apt.sh','install','-y','libopenblas-dev'],out/'apt-install.log')
    source=ROOT/'src/mimo_asr.cpp'; original=source.read_text(); patched=patch_source(original)
    (out/'mimo_asr.cpp.diagnostic').write_text(patched); receipt['patched_sha256']=digest(out/'mimo_asr.cpp.diagnostic'); save()
    try:
        source.write_text(patched)
        wrapper=scratch/'wrapper';wrapper.mkdir(exist_ok=True)
        (wrapper/'entry.cpp').write_text('#include "mimo_asr.h"\nextern "C" int lm_probe_threads() { return mimo_asr_context_default_params().n_threads; }\n')
        (wrapper/'add.cmake').write_text(f'function(add_lm_probe)\nadd_library(mimo_lm_probe SHARED "{wrapper / "entry.cpp"}")\ntarget_link_libraries(mimo_lm_probe PRIVATE mimo_asr)\nendfunction()\ncmake_language(DEFER CALL add_lm_probe)\n')
        build=scratch/'build'
        run(['cmake','-S',ROOT,'-B',build,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DBUILD_SHARED_LIBS=ON',
             '-DGGML_NATIVE=OFF','-DGGML_BLAS=ON','-DGGML_BLAS_VENDOR=OpenBLAS','-DCRISPASR_MEL_BLAS=OFF',
             '-DCRISPASR_BUILD_TESTS=OFF','-DCRISPASR_BUILD_SERVER=OFF','-DCRISPASR_BUILD_EXAMPLES=OFF',
             f'-DCMAKE_PROJECT_crispasr_INCLUDE={wrapper / "add.cmake"}'],out/'configure.log')
        run(['cmake','--build',build,'--target','mimo_lm_probe','-j4'],out/'build.log')
        library=build/'libmimo_lm_probe.so';assert library.is_file()
        (out/'CMakeCache.txt').write_bytes((build/'CMakeCache.txt').read_bytes())
        def pinned(key):
            repo,rev,name,sha=PINS[key]
            p=Path(hf_hub_download(repo,name,revision=rev,local_dir=scratch/'models'));assert digest(p)==sha;return p
        reference=pinned('reference')
        for quant in ['q4_k','f16']:
            model=pinned(quant)
            for blas in [False,True]:
                for flash in ['flash','eager']:
                    arm=('blas-' if blas else 'cpu-')+flash; dest=out/quant/arm
                    env=dict(os.environ)
                    if blas:env['MIMO_LM_DIAG_BLAS']='1'
                    run([sys.executable,__file__,'--worker',library,model,reference,dest,flash],out/(quant+'-'+arm+'.log'),env=env)
                    trace=(out/(quant+'-'+arm+'.log')).read_text()
                    assert ('MIMO_LM_DIAG_BLAS_ACTIVE backend=BLAS' in trace)==blas
            comparisons={}
            for layer in LAYERS:
                comparisons[str(layer)]={}
                for part in PARTS:
                    arrays={arm:np.fromfile(out/quant/arm/str(layer)/(part+'.f32'),dtype=np.float32) for arm in ['cpu-flash','cpu-eager','blas-flash','blas-eager']}
                    base=arrays['cpu-flash']
                    comparisons[str(layer)][part]={arm:metrics(a,base) for arm,a in arrays.items()}
                    comparisons[str(layer)][part]['blas_eager_vs_flash']=metrics(arrays['blas-eager'],arrays['blas-flash'])
            receipt['comparisons'][quant]=comparisons
            receipt.setdefault('canonical',{})[quant]={arm:json.loads((out/quant/arm/'execution.json').read_text()) for arm in ['cpu-flash','cpu-eager','blas-flash','blas-eager']}
            save()
            model.unlink()  # source pins remain remote; do not retain F16 and Q4 together
        control=metrics(np.array([2.,4.,6.]),np.array([1.,2.,3.])); assert control['relative_l2']==1
        receipt['scale_negative_control']=control
        receipt['passed']=True;receipt['acceptance_scope']='Capture invariants only; not model or PR acceptance';save()
    finally:
        source.write_text(original)
        assert source.read_text()==original


if __name__=='__main__':
    if '--worker' in sys.argv:worker(*sys.argv[2:])
    else:main()
