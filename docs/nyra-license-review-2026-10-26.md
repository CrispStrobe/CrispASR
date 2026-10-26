# Nyra #456 license review — 2026-10-26

Disposition: keep integration deferred. No conversion, backend, registry entry,
model publication or generated alignment dataset is authorized by this review.
The whole [issue and both comments](https://github.com/CrispStrobe/CrispASR/issues/456)
were read. The October 20 clarification supersedes the September gated-GGUF
proposal.

The complete [model license at immutable revision
`9c8e724072261c30641a91ca365ce4f7b4ae7f83`](https://huggingface.co/nyralabs/nyra_forced_aligner_en/blob/9c8e724072261c30641a91ca365ce4f7b4ae7f83/LICENSE.md)
was read, including Parts A–C and every section of Part B. This is version 1.0,
effective September 21, 2026. The [upstream license statement](https://github.com/nyrahealth/nyra-forced-aligner#license)
independently distinguishes model/output rights from code rights. Its source
revision is `e8235c3a213af2e63449b814969b029dc8b6f78a`. Anonymous model
metadata still resolves to the pinned revision. [Hashes and review receipt](nyra-license-review-2026-10-26.json).

| Component or action | Published terms and consequence |
|---|---|
| Inference, preprocessing and postprocessing software | Part A MIT; retain its copyright and permission notice when copying substantial code. This grant does not cover the model or its outputs. |
| Acoustic/GMM parameters, projections, trees, lexicon, phone maps and configuration | Part B, sections 1–2. Restricted materials include these small assets, not just neural weights. Rewriting the runtime or changing file format does not clear their use. |
| GGUF conversion, quantization and other changed models | Part B sections 1 and 3. Derivatives retain the agreement; redistribution requires the complete agreement, attribution and contractually bound recipients before or upon access. Permissive relicensing is prohibited. |
| Non-commercial activity | The definition permits research and other activity without commercial gain, but expressly excludes production or operational deployment. Calling an unpaid deployment non-commercial does not supply permission. |
| Production, products, hosted service or API | Section 2.3 requires a separate signed written commercial license; section 5(b) separately prohibits hosted/API offerings under the public license. A download acceptance switch cannot supply that grant. |
| Timestamps, alignments, segmentations and intermediate representations | Sections 1 and 4 cover these outputs and derived annotation datasets. Output use and redistribution have their own restrictions, recipient-contract and generated-content marking obligations. |
| Training or improving a commercially intended model using outputs | Section 4 prohibits this under the public license. Models trained partly on outputs inherit the agreement under sections 1 and 3–4, regardless of their proportion in a dataset. Do not quietly reuse outputs as general regression/training fixtures. |
| Bundled Microsoft WavLM-large | Part C excludes unchanged upstream WavLM files from Part B and points to Microsoft's upstream terms. That statement alone does not establish exact file provenance or complete notices for a future converted bundle. Audit those separately before distribution. |
| Optional espeak-ng/OOV fallback | Separate GPL/dependency review. Omitting it would not resolve the model/output restrictions; GPL is not a blanket prohibition on all forms of distribution. No bundling or linkage decision is made here. |

No applicable signed commercial license or other written grant was provided or
obtained. Do not contact the licensor or accept terms on the user's behalf.
A future proposal must identify the intended use and actual distribution,
provide rights covering that scope (including output use and any downstream
training), resolve notices/provenance and dependency obligations, and pass the
repository's separate model-license review. Research-only experimentation would
also require its own scoped review of recipient and output obligations; it is
not queued by this report. The existing permissive software license cannot
supply the missing model/output rights.
