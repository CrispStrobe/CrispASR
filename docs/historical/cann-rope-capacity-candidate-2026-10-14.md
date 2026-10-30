# CANN rotary cache-capacity candidate

Shared ggml candidate `0a862076567afd8468822ccc512c97ac6a69b340` is based on
the exact physically tested `21acf3162beeb4e51260f0f86deda80995423712`.
It records allocated sin/cos bytes separately from last-call dimensions and
grows both buffers when the required byte count exceeds that capacity.
Rotary arithmetic, upstream tolerances and capability routing do not change.
The existing graph-preload path uses the same allocation function; device
execution and graph compatibility still need the physical rerun.

The four unchanged original-source capacity regressions now pass. Both
previously failing transitions fit their buffers; ordinary position growth
and equal-byte shape changes still pass. The [pre-fix failure proof](cann-rope-capacity-2026-10-14.md)
is retained. Thirteen actual BF16 predicate cases per chip also pass under
310P and 910B compilation. These are bounded CPU allocation/routing tests,
not RoPE numerical execution or model acceptance.

CrispASR diagnostic source `ca4c12c4cfec475c0009f9718a18ffb42d2cda6a` pins
the candidate and adds the capacity regression to both manual SDK jobs.
[SDK run38536737288](https://github.com/CrispStrobe/CrispASR/actions/runs/38536737288)
is submitted: x86_64/910B ACL-graph-on and ARM64/310P ACL-graph-off, including
CLI, shared ABI, diff, quantizer and test-backend-ops. Its result remains
pending. Production main still pins21acf3162, and no new release follows.

[Complete 15-payload candidate proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/4658a96c2b67c22c793f5f9be22a4c18234e0473/cann-20261014-rope-capacity/complete-candidate-capacity.zip)
is public and fully anonymously verified: 69,611 bytes, SHA256
`b4f7cfc5273fffd4702d89fbbab79edd46c2a54fcd728907e9ece1bb7e606929`.
It includes actual candidate sources, patch, original test, probe/executable,
all result rows and receipts. [Structured result](cann-rope-capacity-candidate-2026-10-14.json).

## Physical 910B3 rerun

Use the same activated SDK9.1 environment as the reported run. Start with a
fresh clone/build/output, rather than altering the old failed evidence.
These are the existing verified build/operator commands with immutable
candidate source pins and the locally executed capacity test added.

```bash
git clone --no-checkout https://github.com/CrispStrobe/CrispASR.git CrispASR-cann-rope-proof
cd CrispASR-cann-rope-proof
git checkout --detach ca4c12c4cfec475c0009f9718a18ffb42d2cda6a
git submodule update --init --recursive
python3 tests/test_cann_rope_capacity.py -v
python3 tests/test_cann_bf16_routing.py -v
cmake -S ggml -B build-cann-ops \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DGGML_CANN=ON -DCANN_INSTALL_DIR="$ASCEND_TOOLKIT_HOME" \
  -DSOC_TYPE=Ascend910B -DUSE_ACL_GRAPH=ON \
  -DGGML_BUILD_TESTS=ON -DGGML_BUILD_EXAMPLES=OFF
cmake --build build-cann-ops --target test-backend-ops -j4
python3 tools/validate_backend_ops.py \
  --binary build-cann-ops/bin/test-backend-ops --backend CANN0 \
  --ops ADD,RMS_NORM,SOFT_MAX,ROPE,MUL_MAT --out cann-rope-candidate-proof
```

Return the complete proof directory, configure/build logs, CMakeCache,
compile_commands, native Git revision/status receipts, device information
and SDK versions. Keep unsupported and failing cases and upstream tolerances
unchanged. Do not filter out the two failing widths. A fresh-process complete
RoPE repeat should confirm the original repeated failure is resolved.
Full-model diff, quantized speech, actual scheduler fallback and performance
remain follow-ups after the operator gate, not claims of this candidate.
