# Ascend operator validation

`tools/validate_backend_ops.py` runs the existing ggml correctness tests against
CPU reference math, preserving upstream tolerances. It requires the requested
backend to complete and each requested operator to run nonzero passing cases.
Unsupported cases remain in the complete logs; they are not counted as passes.
Receipts include binary/script-checkout provenance, log hashes and actual counts.

[Hosted ARM CPU validation](https://github.com/CrispStrobe/CrispASR/actions/runs/38281476089)
executes one real passing ADD case and three real upstream false-success cases:
missing backend, empty parameter selection and unknown operator. The upstream
binary exits zero for all three, while the wrapper rejects each. See
[the complete receipt](cann-hardware-wrapper-2026-10-12.json).

For the contributor hardware experiment, use a CANN-enabled `test-backend-ops`
binary built against this repository's ggml revision
`21acf3162beeb4e51260f0f86deda80995423712`. This is the physically validated
310P BF16 repair. The wrapper requires only Python's
standard library. Start with the same bounded case exercised on CPU:

```bash
python3 tools/validate_backend_ops.py \
  --binary /path/to/CANN/test-backend-ops --backend CANN0 --ops ADD \
  --params '^type=f32,ne=\[1,1,1,1\],nr=\[32,1,1,1\],nf=1,perm1=0,src_overlap=0$' \
  --out cann-add-proof
```

Then request the wider existing operator suite, in a fresh output directory:

```bash
python3 tools/validate_backend_ops.py \
  --binary /path/to/CANN/test-backend-ops --backend CANN0 \
  --ops ADD,RMS_NORM,SOFT_MAX,ROPE,MUL_MAT --out cann-operators-proof
```

Return the complete output directories, device/chip model, SDK version and
whether ACL graphs were enabled in the binary. Operation failures/timeouts
retain logs and receipts and produce a failing process exit. The command
interface and empty-test guards are verified on CPU; the contributor returned physical Ascend 310P3 execution on 2026-10-13.
The original broad suite failed; the exact repair rerun passes all 924 executed
cases. Both original failures and the passing rerun are retained below. Passing operator tests alone
cannot establish full-model inference, quantized-model speech quality or speed.


## Physical Ascend 310P3 results — 2026-10-13

The contributor returned [the complete archive](https://github.com/user-attachments/files/33372242/crispasr-cann-310p3-validation-2026-10-13.tar.gz)
in [PR #544 comment 6140622935](https://github.com/CrispStrobe/CrispASR/pull/544#issuecomment-6140622935).
Its SHA-256 matches the reported
`83cc620570c9345153545c02418a6171f5b5914d5dc43f743dbc4b12a162feb7`.
All 17 member hashes are retained and all five receipt assessments were
recomputed against their complete logs with the unchanged wrapper. The
[verified receipt](cann-310p3-hardware-2026-10-13.json) preserves failures too.

Hardware: two Ascend 310P3 devices installed, `CANN0` tested on ARM64;
SDK 9.1.0, ACL graphs disabled, ggml
`c36dab89b662838f0f5d4826c399198c0b90bbfc`.

| Requested control | Physical result |
|---|---|
| Bounded F32 ADD | 1/1 pass |
| All F32 ADD cases | 56/56 pass |
| F32 weights / F32 input MUL_MAT | 272/272 pass |
| F16 weights / F32 input MUL_MAT | 282/282 pass |
| RMS_NORM | 51/51 pass |
| SOFT_MAX | 107/107 pass; 107 unsupported cases retained |
| ROPE | 109/109 pass; 361 unsupported cases retained |
| Broad ADD | SIGABRT after 92 observed passing cases; no completed suite |
| Broad MUL_MAT | SIGABRT after 257 observed passing cases; 1,488 unsupported cases retained |

Both aborts explicitly report unsupported `DT_BFLOAT16`: `aclnnAdd` rejects
its input and `aclnnMm` rejects `matA`. In the pinned source, ADD's capability
check accepts BF16 unconditionally; MUL_MAT excludes BF16 weights on 310P but
misses BF16 activations. A scoped capability repair is being prepared and must
be compiled and rerun on the device. Production ggml remains pinned to the
reported version. Partial case counts and successful restricted suites do not
turn the broad suite into a pass. Full-model diff/ASR, quantized quality and
performance remain unvalidated on this NPU.


## BF16 capability candidate: physical rerun

Diagnostic CrispASR source
`bfb4943ac1e4488e377257a47163a0edcea7d6cd` pins shared ggml candidate
`21acf3162beeb4e51260f0f86deda80995423712`. Only 310P capability routing
changes: BF16 ADD input/output and BF16 MUL_MAT activation inputs are rejected
so the scheduler can use a supported backend. The 910B predicate is unchanged.
Thirteen isolated routing cases pass under each chip configuration; the old
310P predicate fails six. This does not prove a running model's CPU fallback.
[SDK matrix 38365667723](https://github.com/CrispStrobe/CrispASR/actions/runs/38365667723)
passes ARM64/310P (ACL graphs off) and x86_64/910B (ACL graphs on), including
CLI, shared ABI, diff, quantizer and `test-backend-ops`. See
[the complete compile receipt](cann-310p-bf16-compile-2026-10-13.json).
The physical candidate rerun now passes 924/924 executed cases; source and
compiler evidence are verified below. The exact repair is integrated.
See [the candidate receipt](cann-310p-bf16-candidate-2026-10-13.json).

Use a fresh directory and the same activated CANN SDK 9.1.0 environment as the
reported run. The following builds only the upstream operator test binary:

```bash
git clone --no-checkout https://github.com/CrispStrobe/CrispASR.git CrispASR-cann-bf16-proof
cd CrispASR-cann-bf16-proof
git checkout --detach bfb4943ac1e4488e377257a47163a0edcea7d6cd
git submodule update --init --recursive
python3 tests/test_cann_bf16_routing.py -v
cmake -S ggml -B build-cann-ops \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DGGML_CANN=ON -DCANN_INSTALL_DIR="$ASCEND_TOOLKIT_HOME" \
  -DSOC_TYPE=Ascend310P -DUSE_ACL_GRAPH=OFF \
  -DGGML_BUILD_TESTS=ON -DGGML_BUILD_EXAMPLES=OFF
cmake --build build-cann-ops --target test-backend-ops -j4
python3 tools/validate_backend_ops.py \
  --binary build-cann-ops/bin/test-backend-ops --backend CANN0 \
  --ops ADD,RMS_NORM,SOFT_MAX,ROPE,MUL_MAT --out cann-bf16-candidate-proof
```

Return the complete fresh proof directory and configure/build stdout/stderr,
`build-cann-ops/CMakeCache.txt`, `build-cann-ops/compile_commands.json`, device
information and SDK versions. This tests the broad suite that previously
aborted, retaining unsupported BF16 cases explicitly. Keep the original failed
archive; do not overwrite it or filter BF16 out through `--params` to obtain a
pass. A passing candidate must finish both broad operators under the unchanged
upstream tolerances. Full-model diff, decoded ASR and scheduler fallback still
need separate validation before any NPU model or speed claim.


## Verified physical candidate rerun

[Contributor follow-up 6143426863](https://github.com/CrispStrobe/CrispASR/pull/544#issuecomment-6143426863)
returns the exact immutable candidate on the same 310P3/SDK9.1 environment.
The [complete verified receipt](cann-310p3-bf16-physical-2026-10-13.json) retains
all member/log hashes, source revisions, actual compiler flags and unsupported
cases. All five broad operators finish: ADD 103/103, RMS_NORM 51/51,
SOFT_MAX 107/107, ROPE 109/109, MUL_MAT 554/554. Thus 924 executed cases pass;
2,241 unsupported cases remain visible. Both prior BF16 aborts are resolved.
Both SDK compilation architectures and all 13 composed general CI jobs pass;
the exact candidate is integrated. [Integration ledger](cann-310p-bf16-integration-2026-10-13.json).
Full-model inference, quantized speech and performance still require separate proof.


Final-main SDK revalidation at `cfceeb752` passes both architecture jobs in
run `38401026881`, including the actual operator build target and routing
guards. [Complete source/job/log receipt](cann-final-main-sdk-2026-10-13.json).
Full model and quantized speech/speed still require separate physical proof.

## Physical Ascend 910B3 follow-up — 2026-10-14

[Contributor comment 6162885815](https://github.com/CrispStrobe/CrispASR/pull/544#issuecomment-6162885815)
returns a [complete 25-member archive](https://github.com/user-attachments/files/33421861/crispasr-cann-910b3-official-validation-2026-10-14.tar.gz),
33,052 bytes, SHA256
`d191c69b564ad8b47d8e92e15b0c993bc589359ffa39d1683b20143a45d94ef5`.
All member/log hashes and all seven assessments are recomputed using the
unchanged original wrapper. [Complete verified receipt](cann-910b3-hardware-2026-10-14.json).

Native Git receipts record clean tracked main `cacd697e37d3aa002d996509ba4de60865003a6a`
and ggml `21acf3162beeb4e51260f0f86deda80995423712`. Device output identifies
910B3; the actual compiler command defines `ASCEND_910B` and `USE_ACL_GRAPH`.
SDK runtime is 9.1.0, ARM64. The binary itself is not included: its SHA256
is a producer receipt, not an independently rehashed executable.

| Broad operator | Executed cases | Unsupported cases retained |
| --- | ---: | ---: |
| ADD | 154/154 pass | 0 |
| RMS_NORM | 51/51 pass | 0 |
| SOFT_MAX | 107/107 pass | 107 |
| ROPE | **450/452 fail** | 18 |
| MUL_MAT | 927/927 pass | 1,349 |

Bounded ADD 1/1 and routing controls pass. A separate complete RoPE rerun
again fails the same two cases under the unchanged 1e-7 error tolerance:
F32 NEOX `[256,8,512,1]`, rotary width 256, no frequency factors; and
F32 NEOX `[512,8,512,1]`, rotary width 128, frequency factors enabled.
Reported errors are about .5109/.0975, repeated at .5103/.0981. Thus the
910B3 broad suite is not a pass. The earlier passing 310P3 evidence remains
scoped to that chip and its supported cases. No full-model/quantized speech
or performance acceptance follows.

Source review identifies a candidate allocation defect: sin/cos cache resize
is gated only by position-length growth, while required bytes also grow
with rotary width. The final four observed cases keep 512 positions and
change rotary width 64,64,256,128. Required bytes per sin/cos buffer change
131,072,131,072,524,288,262,144. The last two fail. The reviewed operator
source matches the exact tested revision (Git blob
`902d2eda693837c7ea636840facb6725ab0f7746`). This is a hypothesis awaiting
a pre-fix regression and physical candidate rerun, not a claimed repair.
Per the requested task order, runtime work resumes after the active Dia
free-running capture-transparency control passes.

### Pre-fix allocation regression now reproduced

[Complete method and proof](cann-rope-capacity-2026-10-14.md) compiles the
literal original allocation block and cache class. The two observed failing
widths retain 131,072-byte requests where 524,288/262,144 bytes are needed;
width growth with shrinking positions also fails. Four original-source
capacity tests yield two expected failures and two passing controls. This
confirms the requested-capacity defect, not a physical device repair.
The failing test is retained in the diagnostic branch; production is unchanged.

### Scoped capacity candidate and physical rerun

Candidate ggml0a8620765 now passes all four unchanged CPU byte-capacity
regressions; the old failures remain public. Both chip routing controls
pass. [Exact candidate/proof/device commands](cann-rope-capacity-candidate-2026-10-14.md)
pin CrispASRca4c12c4 and ggml0a8620765 in fresh directories. SDK
run38536737288 is submitted on both compilation architectures. Its result
and the physical 910B3 rerun are pending; production pin remains21acf3162.

## Rotary-capacity repair integrated — October14

Exact physically tested ggml0a8620765 is now the production pin after all
thirteen composed CI jobs and both SDK builds pass. Physical910B3/SDK9.1
passes1691/1691 broad cases and452/452 full RoPE repeat; unsupported cases
and original failures remain retained. [Final integration proof](cann-rope-integration-2026-10-14.md).
No further rotary-candidate rerun is requested. [Remaining tasks](../cann-todo.md)
are maintainer-owned; future full-model/device help is optional.
