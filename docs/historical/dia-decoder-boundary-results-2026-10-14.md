# Dia decoder boundary capture: independent CPU results

[GH 38472367538](https://github.com/CrispStrobe/CrispASR/actions/runs/38472367538)
succeeded at source `8d0ad8862d2cad5d305a48a57f2a99b5024bc39a`. Complete
raw ZIP SHA, per-member CRC and payload hashes verify. All nine delivered
source files match immutable Git objects; ggml is pinned to c36dab89.
All sixteen actual-original hook tests pass. Original ASR is WER 0.
Original OFF/ON/OFF common arrays and canonical code/input histories are
exact; native complete logits/feedback/PCM are exact across its three arms.
All 120 ordinary stages pass. Exactly sixteen ON-only boundary snapshots
exist, and doubled-hidden/invalid-index/out-of-history negatives reject.
Sixteen scale-invisible negatives expose magnitude error independently.
Neural/DAC receipts contain 381/134 audited tensors and match pinned original
weights/checkpoint and exported model hashes. Full model/library payloads
are not delivered here; their SHA values are producer receipts, rather than
new local whole-model measurements.

At block zero and fixed step 978, input and pre-SA norm are byte-exact.
The remaining differences are small. These rows compare original and native
pipeline captures, including upstream input drift; they do not isolate an
operator on matching inputs. Metrics use 128-value rows.

| Boundary | Minimum row cosine | Maximum row norm error | Relative L2 |
| --- | ---: | ---: | ---: |
| input | 1.000000000000000 | 0.000e+00 | 0.000e+00 |
| pre_sa_norm | 1.000000000000000 | 0.000e+00 | 0.000e+00 |
| sa_attention | 0.999999999943487 | 2.204e-06 | 4.711e-06 |
| sa_output | 0.999999999994273 | 6.788e-07 | 2.790e-06 |
| sa_residual | 0.999999999996923 | 4.333e-07 | 2.140e-06 |
| pre_ca_norm | 0.999999999994897 | 1.113e-06 | 1.889e-06 |
| cross_q | 0.999999999855428 | 5.349e-07 | 5.204e-06 |
| cross_attention | 0.999999999970525 | 2.881e-06 | 1.264e-06 |
| cross_output | 0.999999999999314 | 7.940e-07 | 7.605e-07 |
| ca_residual | 0.999999999997027 | 5.935e-07 | 1.947e-06 |
| pre_mlp_norm | 0.999999999997303 | 5.518e-07 | 1.560e-06 |
| mlp_gate | 0.999999999999707 | 3.529e-07 | 6.572e-07 |
| mlp_up | 0.999999999994747 | 9.310e-07 | 1.831e-06 |
| mlp_hidden | 0.999999999991833 | 3.066e-06 | 2.024e-06 |
| mlp_output | 0.999999999997143 | 1.404e-06 | 1.614e-06 |
| output | 0.999999999997396 | 1.313e-06 | 1.436e-06 |

No large block-zero projection/residual/MLP defect is demonstrated by these
observations. Next replay the actual original fused MLP and literal native
split graph on each owned pre-MLP input. Compare local-vs-hosted controls
explicitly before attributing error to the operator. Native free-running
speech, Q4, GPU changes, performance and release remain unaccepted.

The complete raw GH payload and audit/method/log archive are public and
fully anonymously SHA256-verified at revision `4a625f2472c3208a355691a1fda171c70042c4e6`:

- [artifact-11858752399.zip](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/4a625f2472c3208a355691a1fda171c70042c4e6/dia-20261014-decoder-boundaries/artifact-11858752399.zip) — 890,528,559 bytes; SHA256 `de7187841db0a43da0d3462e4d72f50b7283d3876e366d50745ee0be99058e26`.
- [complete-decoder-boundary-audit.zip](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/4a625f2472c3208a355691a1fda171c70042c4e6/dia-20261014-decoder-boundaries/complete-decoder-boundary-audit.zip) — 115,968 bytes; SHA256 `6223a7f15e9143e0f6595818a1df66cb959b5344190eaee04044a77682eeae19`.

Full source/trajectory/physical/metric ledgers and scope are in
`dia-decoder-boundary-results-2026-10-14.json`. Raw producer inputs remain
unchanged. Initial and final local audit ledgers are retained: immutable
delivery inputs are separated from mutable analysis-helper source files.
The latter are bound separately by the public proof manifest.

Transfer note: default small writes slowed the CIFS download. The previous
writer was terminated before a strictly validated range resume with 4 MiB
buffering; the complete final ZIP matches GitHub's original digest.
