# Dia late-attention capture and same-input replay

CPU capture [38438623888](https://github.com/CrispStrobe/CrispASR/actions/runs/38438623888) and CUDA compile package [38446095504](https://github.com/CrispStrobe/CrispASR/actions/runs/38446095504) completed. Complete logs and raw artifact ZIPs are retained under storage; both ZIPs match GitHub digests. Independent audits verify original/capture/source/trajectory/payload bytes. Native free speech remains rejected; compilation and teacher replay accept neither GPU execution nor Q4.

The original Python dumper runs actual pinned original generation, rather than a reconstructed network. Its complete input/code grids remain canonical, and capture OFF/ON/OFF preserves all common arrays. Original audio is recognized as “We are committed to making it secure.” with WER 0. Native capture OFF/ON/OFF preserves complete logits, feedback and decoded audio. All three normal `crispasr-diff` replays pass 40/40 stages, including magnitude checks; scaled-hidden and invalid/out-of-history negatives still fail.

The 72 new tensors cover Q, full K/V histories and attention before output projection at step 978 in every decoder layer. Relative L2 errors on original versus native inputs are:

| Layer | Q | K history | V history | Attention |
|---:|---:|---:|---:|---:|
| 0 | 5.37441e-05 | 1.57306e-05 | 6.78769e-07 | 4.71123e-06 |
| 1 | 6.28442e-05 | 2.07014e-05 | 1.42957e-06 | 1.7431e-06 |
| 2 | 4.34084e-05 | 1.80044e-05 | 2.12624e-06 | 5.97857e-06 |
| 3 | 4.31465e-05 | 1.69621e-05 | 4.65075e-06 | 1.06224e-05 |
| 4 | 3.27069e-05 | 1.25961e-05 | 6.4445e-06 | 3.53297e-06 |
| 5 | 3.39442e-05 | 1.24653e-05 | 4.56807e-06 | 2.73219e-06 |
| 6 | 3.77909e-05 | 1.92182e-05 | 3.42258e-06 | 9.12035e-06 |
| 7 | 4.05899e-05 | 1.86026e-05 | 3.73795e-06 | 6.06786e-06 |
| 8 | 5.39549e-05 | 1.47367e-05 | 2.54812e-06 | 4.16051e-06 |
| 9 | 5.28009e-05 | 1.89351e-05 | 3.03061e-06 | 6.06309e-06 |
| 10 | 5.43131e-05 | 1.72458e-05 | 5.82443e-06 | 8.39187e-06 |
| 11 | 5.66657e-05 | 1.96033e-05 | 3.37646e-06 | 3.726e-06 |
| 12 | 5.50655e-05 | 1.82281e-05 | 2.82603e-06 | 4.52621e-06 |
| 13 | 4.91591e-05 | 1.65653e-05 | 3.76017e-06 | 4.81186e-06 |
| 14 | 5.51546e-05 | 1.82109e-05 | 2.73591e-06 | 3.05145e-06 |
| 15 | 5.93762e-05 | 2.7124e-05 | 2.54858e-06 | 3.16752e-06 |
| 16 | 5.48657e-05 | 1.87237e-05 | 2.57556e-06 | 2.61529e-06 |
| 17 | 5.81513e-05 | 1.83055e-05 | 2.36648e-06 | 3.26552e-06 |

Minimum attention row cosine is 0.9999999994017428; maximum attention row norm error is 2.0642567031603498e-5. Exact cosine, magnitudes, maximum errors, shapes and raw-file hashes for all 72 tensors, plus all 120 normal stage rows, are retained in `dia-attention-history-results-2026-10-13.json`.

The weight-free consumer runs the actual original SDPA/transpose expression and literal native attention graph on the same captured inputs. All 18 native attention outputs reproduce byte-for-byte at BOTH one and four native CPU threads (36 comparisons). Compact and 3072-slot original caches produce identical outputs in all 36 original/native input cases. Same-input native versus actual-original attention has minimum row cosine 0.99999999999744, maximum row norm error 6.882499823390731e-7 and maximum relative L2 4.850572647357973e-7. All 216 comparison rows and exact repeats are retained.

The separately packaged CPU bridge and hosted original hardware/build differ: original-input Torch replay versus saved original attention has maximum relative L2 9.09885957456676e-7, rather than byte identity. Native outputs do reproduce exactly from native inputs. This excludes a large CPU cache-layout/GQA/attention defect at this observed boundary; it does not establish every graph boundary, free-running speech or CUDA arithmetic. Do not promote the already-rejected original-RoPE path on the basis of these diagnostic numbers.

The CUDA package independently passes archive/member/alias/provenance checks for all 27 members. Its seven source files match immutable package source, and all six runtime files match CPU capture source. Prepared GPU worker bf895ecd6 passes 16 guards, reuses these inputs and keeps runtime bytes unchanged. Inputs are published and anonymously verified at public fixture revision `61eafbd6901d5f8fe72025acf8c86eb078ff5bf9`. Kaggle accepted script version `2026-10-13.dia-cuda-attention.4` (kernel version 4), with pinned rendered script SHA `b4817824f86d779f0521c6a6b15d9e25ed32260468fd3202594800d707c40aac`. All previous 61 terminal outputs and complete raw/decoded logs remain verified; the returned kernel slug is tracked. GPU results are pending. Actual GPU observation, then remaining decoder boundaries, precede any runtime/default/model/release promotion.

GPU follow-up: version 4 failed before Dia capture because the worker hid
the host driver library path. CPU ASR fallback was correctly rejected by
the CUDA witness. The repair preserves that path, adds preflight and passes
a real ELF dependency regression; actual version 5 is submitted. See
`dia-attention-cuda-driver-repair-2026-10-13.json`.
