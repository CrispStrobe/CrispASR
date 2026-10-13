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
`c36dab89b662838f0f5d4826c399198c0b90bbfc`. The wrapper requires only Python's
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
interface and empty-test guards are verified on CPU; actual Ascend execution
is the requested experiment and remains pending. Passing operator tests alone
cannot establish full-model inference, quantized-model speech quality or speed.
