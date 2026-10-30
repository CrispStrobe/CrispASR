# OmniVoice standalone utility proof and integration

The utility header, standalone probe and original-function oracle have been
split from [draft PR521](https://github.com/CrispStrobe/CrispASR/pull/521).
First split commit `a140b3d7` preserves the original author and exact three
source files from original commit `a1ae00df`, as present at PR head `381f3a3a`.
Candidate `69d9eac4b10e1cc857e4e2e90b1101fd40983b6b` adds full evidence retention
and upstream license attribution. Its tested source bytes are unchanged from
`a479139977210766cf93860f77fa8fdf99118d38`.

No production speech funnel includes the helper. Reference preprocessing,
output processing, generation, params/setter ABI and defaults are unchanged.
The full PR remains a draft: both original and native fixed clone controls
fail the unchanged 20% WER gate at 2/9 words. Upstream issue256 attributes the
reported onset to an abruptly truncated reference; utility parity does not
prove that the onset or cloned speech has been repaired.

Corrected2026-10-22: the2/9 was the reference clip itself. `fox.wav` says
"fex/dig" under both an RNNT and a CTC recognizer, and every clone arm inherited
those two words; the no-reference control says "fox/dog". The speech gate now
transcribes each fixture before cloning and scores clones cross-sentence on a
vetted reference, where seven arms pass the unchanged0.2 gate at maximum0.111.
See [reference integrity](omnivoice-clone-reference-integrity-2026-10-22.json).
Utility parity alone still does not prove a speech repair, and no gate was
relaxed.

## Complete local original/native replay

On isolated Python 3.12, NumPy 1.26.4 and pydub 0.25.1, the inherited oracle
passes 738 waveform calls and 18 multilingual text cases. An independent
auditor then loads the actual compiled native probe and re-executes the pinned
original Python functions against every stored input:

| Check | Result |
| --- | --- |
| PCM16 silence-cleanup calls | 369, byte-exact |
| Fade/pad calls | 369, maximum absolute error 0 |
| Text punctuation calls | 18, exact |
| Full stored arrays checked | 1,601 |
| Rejecting witnesses | Unprocessed input, doubled amplitude, reversed samples |

The fade gate remains equal lengths and maximum absolute error at most 2e-7;
it was not relaxed. Original upstream audio/text source hashes are pinned at
`08be0b4ccbac3e13e374e86fbfead4b4cac343e2`. Every raw array, source and actual
probe hash is checked. The native helper/probe are byte-identical to the PR's
previously audited files. Compiler command/version, actual ELF and linked-library
hashes are retained. Actual probe SHA256:
`123675aa7f009292c1eac38eee9c4f545c4d8025bbc1c5c1c2c05c796935f583`.

## Public proof and remaining gates

The [complete immutable proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/1ac18a4e22a6b23ca7769849e68608cba0eae26a/omnivoice/utility-split-20261015/complete-utility-split-v1.zip) retains
1620 payloads and 37100271 compressed bytes, SHA256
`c825004f1e04378d7b7fa99dc7b98bafeabc1d06fcf1e18b9ec42865052f2d1f`. An anonymous full download matches that hash. It contains all
inputs and both complete output arrays, original/native source, actual probe,
compiler/link receipts, auditor, manifests and license notices. No model weights
or private speech are required for this utility diagnostic.

[Hosted utility run38739872601](https://github.com/CrispStrobe/CrispASR/actions/runs/38739872601)
and [lint38739877120](https://github.com/CrispStrobe/CrispASR/actions/runs/38739877120)
both pass on the exact tested source, including all 10 lint jobs. The full raw
artifact11980620391 is 36,669,267 bytes, SHA256
`135b5b6681c32260414305a6eef8d7cabfc4f1b61b83ecfa64cb39d041c4ed65`.
Independent replay of its actual ELF and original Python functions verifies all
1,601 arrays, 369 exact cleanup calls, 369 zero-error fade/pad calls and 18 exact
text cases. Source, array, original-function and actual ELF hashes all match.

The [complete hosted proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/71ef6bdbea06e77f26b2fb2951ea8f371fb5b63f/omnivoice/utility-split-20261015/complete-hosted-utility-split-v1.zip) retains the raw artifact,
terminal API/logs, independent auditor/result, manifest and license notices:
32488123 bytes, SHA256 `3a60926485cb4354ec0763e829a92fc2ea54558db64ae37cc4fbaef220a30139`.
An anonymous full download matches that digest.

Integration `5d582d75b71f5e7a29f6728dcdc93aa091ba6dd8` preserves all tested helper/probe/oracle bytes and
original commit authorship after rebase/fast-forward. This accepts the standalone
utilities only. The complete runtime PR, clone quality, onset repair, GPU and
production defaults remain excluded.

The [candidate method](https://github.com/CrispStrobe/CrispASR/blob/69d9eac4b10e1cc857e4e2e90b1101fd40983b6b/docs/historical/omnivoice-utility-split-2026-10-15.md)
records reproduction details. All large arrays, isolated dependencies and proof
are under `${CRISPASR_STORAGE}/crispasr/omnivoice-utility-split-20261015`.
