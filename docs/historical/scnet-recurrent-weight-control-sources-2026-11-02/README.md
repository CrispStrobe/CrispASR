# Historical recurrent-weight control sources

These are the exact private control source and driver executed on 2026-11-02.
They support the [public recurrent-weight control and fidelity receipt](../scnet-recurrent-weight-controls-2026-11-02.json)
(SHA256 `68b89a337b3b7ebff73368b2cf40308824de9aef3f970fd9ddd7dd73d2e4d704`).
The `.txt` endings keep ordinary C++ formatting and Python test discovery from
modifying or executing the historical bytes. These files are not installed test
targets or a performance benchmark.

## Exact copies

The original directory is identified here by the portable alias
`{control} = scnet-local-validation/recurrent-weight-cache-controls-v1`.
Neither file was reformatted, redacted, or otherwise changed. Inspection before
copying found no credentials requiring redaction. The C++ file deliberately
retains the original absolute include paths because its byte identity is part
of the evidence.

| Original | Historical copy | Bytes | SHA256 |
| --- | --- | ---: | --- |
| `{control}/control-v2.cpp` | [control-v2.cpp.txt](control-v2.cpp.txt) | 16741 | `1f90a2f1df3a2a46d1c5db616cba24becafaed9bb3947111422382a1e855d1ef` |
| `{control}/run.py` | [run.py.txt](run.py.txt) | 9438 | `6debb7119d5e93e251f23e72a375458f8cfa211d6110cb32a41995b26a35ee82` |

## Tested source and runtime

The working tree's parent commit was
`99588fb01986f1ebfb7dad27e072fd424ccdc967`; the recurrent-weight change was
present as a source-bound working-tree change. The parent commit alone does
not identify the tested native source. Its exact `src/scnet.cpp` SHA256 is
`eeb6e583d499b73e28d3fe2b1bb3f55f7be2a9edde8fe81140608b446661fc1c`.
The tested ggml commit was `2a7982f40c97b84fcc2c1f802ebeead0cccde7b3`,
including the allocator failure repair. Subsequent merged ggml revisions are
not the identity of this execution.

The provider was the unmodified LP64/pthread OpenBLAS 0.3.34 build, with two
threads and automatic CPU dispatch. Its shared-library SHA256 was
`f90baa4930522d4f716cce1b6704a99e35e88231306497c12f0258cbb114306e`.

| Original evidence | SHA256 |
| --- | --- |
| `{control}/build.json`: canonical native build and source/object bindings | `7f752d1462171aac969f6133d4a8ed34b913429c3a08679f410666190da9a47b` |
| `{control}/control-build-v2.json`: executed control compile/link argument arrays | `ea94d9100324f8fce280793e04fc806f4045b8108bd98cd5200b697147ea2223` |
| `{control}/receipt.json`: six-case control execution | `d79ec52aac09b334e4ea221f5cd5b0576396b4a63374846fe07868e6971436e8` |
| Executed `control-v2` binary | `a5419f335de0ee4d86e0ec66c1e6895a82b933d16c651d2335aadaa17740389c` |

The native build receipt binds the compiled translation units and objects.
Its broad before/after source snapshot brackets a recorded no-op confirmation
build after the initial rebuild; it must not be described as a snapshot taken
before that initial compilation. The public receipt records this boundary.

## Compile/link command source and reproduction

The canonical executed commands are the `compile_argv` and `link_argv` arrays
in the identified `control-build-v2.json`. The compile flags match the native
`src/scnet.cpp` translation unit. The control includes that source directly,
so its link command uses the same core, ggml and provider libraries without
also linking `libscnet.a`.

The following is a portable rendering of those argument arrays. `REPO`,
`BUILD`, `CONTROL` and `OPENBLAS_PREFIX` denote the matching checkout, its
`build-scnet-openblas034-profile` directory, a separate writable control
directory and the matching provider installation. The original linker ran
with `BUILD` as its working directory. The explicit GCC/OpenMP and pthread
paths below identify the original Linux layout; adapt them to a reproduction
toolchain without claiming the resulting binary has the historical hash.

```sh
/usr/bin/c++ \
  -DCRISPASR_HAVE_BLAS=1 -DGGML_BACKEND_SHARED -DGGML_MAX_NAME=128 \
  -DGGML_SHARED -DGGML_USE_BLAS -DGGML_USE_CPU -DHAVE_BLAS \
  -I"${REPO}/src/." -I"${REPO}/src/../include" \
  -I"${REPO}/ggml/src/../include" -I"${OPENBLAS_PREFIX}/include/openblas" \
  -O3 -DNDEBUG -fPIC -Wmissing-declarations -Wmissing-noreturn \
  -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -O3 -fopenmp \
  -o "${CONTROL}/control-v2.o" -c "${CONTROL}/control-v2.cpp"

/usr/bin/c++ -O3 -DNDEBUG "${CONTROL}/control-v2.o" \
  -o "${CONTROL}/control-v2" \
  -Wl,-rpath,"${OPENBLAS_PREFIX}/lib:${BUILD}/ggml/src:${BUILD}/ggml/src/ggml-blas" \
  src/libcrispasr-core.a -ldl /usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so \
  /usr/lib/x86_64-linux-gnu/libpthread.a ggml/src/libggml.so.0.26.0 \
  ggml/src/ggml-blas/libggml-blas.so.0.26.0 ggml/src/libggml-cpu.so.0.26.0 \
  ggml/src/libggml-base.so.0.26.0 "${OPENBLAS_PREFIX}/lib/libopenblas.so"
```

Reproduction requires working copies with the `.txt` endings removed, adapted
absolute C++ includes and Python directory constants, and the corresponding
source/build receipts, six genuine GGUF models and exact input fixture. Leave
these archived files unchanged and record hashes of any adapted copies. The
driver expects the original evidence layout, rejects existing results and
loader/provider overrides, and verifies the source, model and dependency
identities before accepting results. Simply running it from this documentation
directory is not a supported reproduction procedure.

## What the execution establishes

Across SCNet and SCNet-large in F16, Q8_0 and Q4_K, the control compared all 144
cached matrices byte-for-byte with genuine CPU `ggml_cast` results. Under
`BENCH=0`, it observed 72 actual recurrence callbacks running once with
`ith=0, nth=1`; their CPU splits planned one worker while the requested CPU
and BLAS budgets remained two. Ordinary and observed short-call outputs were
byte-identical. The original and cached weights and input bytes were preserved.

This was a private include-translation-unit diagnostic, with scoped backend
and callback forwarding restored after observation. Mapped-library evidence
describes endpoint file identities, not continuous loader monitoring. The
fixture was the first 0.25 seconds of the recorded music input. The control
does not establish full-profile performance, GPU behavior, or execution of the
no-BLAS/scalar-recurrence fallback. Independent original-F32 stage checks are
separately identified in the linked public receipt.
