# Model license support policy

Permissive licenses (including MIT, Apache-2.0 and BSD), CC-BY, and CC-BY-SA
are eligible for support with their attribution and redistribution obligations.
CC-BY-SA is a share-alike license, not a non-commercial license.

Ordinary CC-BY-NC and CC-BY-NC-SA models remain eligible **only with explicit
license acceptance**, clear non-commercial labeling, and no silent download via
`-m auto` or `--auto-download`. Managed download acceptance uses the exact license
tag through `--accept-license` or `CRISPASR_ACCEPT_LICENSE`; bindings expose the
same bundle metadata. These controls do not grant commercial rights. Local-file
loading is not a rights-verification system.

Research-only terms and additional use, output, revenue, user-count or hosted
service restrictions need a separate review before support or promotion. An
acceptance switch records the user's acknowledgment; it is not project approval
of every possible use. For eligible custom-license support, supplying the switch
affirms that the user accepts the full terms, is eligible, and will fulfill the
conditions applicable to them, including obtaining separately required rights.
This is the user's obligation; CrispASR does not waive those conditions or grant
extra rights. We retain responsibility for our own distribution/notices.
Existing custom-license entries are not automatically certified
by this policy. Do not add new restricted models under a permissive or ordinary
NC label. Review the main weights, tokenizer/codec, other companions and source
code separately, retaining immutable revisions and the actual terms. Unknown
terms are not evidence of permission.

## Follow-up audit, 2026-10-20

- Drama-3: the Fish Audio Community License includes revenue and hosted-service
  conditions. Keep issues545/546 outside implementation/release scope.
- Nyra: the [full pinned review](historical/nyra-license-review-2026-10-26.md) keeps issue456
  deferred. Terms exclude production/operational and hosted/API use, restrict
  generated outputs and impose derivative/recipient obligations. A normal NC
  acceptance switch does not clear integration.
- OmniVoice: the original-generation model revision
  `c5fdb5ccb189668d56333f77ba2629f4cd7535f4` already distinguishes Apache-2.0 code
  from CC-BY-NC weights. This is not merely a later change to today's card. The
  converted-model card's blanket Apache label needs reconciliation. Its named
  HiggsAudioV2 companion has Boson community terms in that same revision under
  `audio_tokenizer/LICENSE`; its text is independently downloaded and hashed.
  The converted-model card is corrected at HF revision
  `3425b20adc97e495b2710cb3a97aac577460f21f`, with anonymous readback verification
  and no weight changes. The registry uses the
  distinct `omnivoice-terms` gate for the bundle, not plain NC acceptance. PR521
  remains subject to both this review and its unchanged speech-quality gates.
- Supertonic-3: OpenRAIL permits some commercial uses but adds use restrictions.
  It now requires explicit `openrail-m` acceptance rather than merely printing a
  notice. Further support/promotion needs the separate review above.

Sources: [Fish terms](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE),
[Nyra terms](https://github.com/nyrahealth/nyra-forced-aligner#license),
[pinned OmniVoice card](https://huggingface.co/k2-fsa/OmniVoice/blob/c5fdb5ccb189668d56333f77ba2629f4cd7535f4/README.md),
[codec terms](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/blob/main/LICENSE),
[Supertonic](https://huggingface.co/Supertone/supertonic-3),
[CC-BY-SA](https://creativecommons.org/licenses/by-sa/4.0/).

## NVIDIA Unified and Boson companion review, 2026-10-22

This is an engineering disposition of the published terms, not a determination
that every downstream use is permitted. A managed-download acceptance switch
records acknowledgment; it cannot supply missing rights or satisfy our own
redistribution obligations. Independently written runtime code and the weights
it loads have separate licenses.

### Parakeet Unified: NVIDIA Open Model License

The original and referenced third-party Unified cards name the NVIDIA Open Model
License, not CC-BY or Apache-2.0. The agreement dated October 24, 2025 permits
commercial use and redistribution, with license/NOTICE requirements, and claims
no ownership of outputs. It is not research-only or NC.

Additional conditions matter: section 2.1 terminates rights for removing a
guardrail without a substantially similar replacement, and requires compliance
with later regulatory license updates or cessation of use/distribution.
Section 2.3 incorporates NVIDIA's Trustworthy AI terms; section 8 includes an
indemnity. Whether a particular conversion removes any guardrail must be checked
against that checkpoint, not presumed from the GGUF format. Cosmos branding
conditions apply to Cosmos models; do not present them as a Parakeet requirement.

Disposition: fixing issue551's erroneous routing/rejection does not add model
support or a registry entry. Before managed Unified support, review the exact
checkpoint/conversion, retain its terms and NOTICE, and use a distinct explicit
`nvidia-open-model-license` acceptance tag. The gate now recognizes this tag;
there is no Unified registry entry or inference support yet. Acceptance affirms
the user's eligibility and compliance, independently of ordinary NC acceptance.

Sources: [original model](https://huggingface.co/nvidia/parakeet-unified-en-0.6b),
[converted model terms](https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/blob/main/LICENSE),
[NVIDIA agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/),
[incorporated terms](https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/).

### OmniVoice's Boson Higgs Audio 2 tokenizer

The pinned companion terms incorporate Meta Llama3's license and acceptable-use
policy. Their requirements extend beyond attribution:

- Section 2 requires Boson's expanded permission above 100,000 annual active users
  in the preceding year. Its wording covers the licensee's/affiliates' products
  and services, not just users of this tokenizer; small model usage does not
  establish eligibility for a large organization.
- Section 1.b.v prohibits using the materials or their outputs/results to improve
  another LLM, except Higgs Audio2/derivatives. Generated training data can
  therefore raise a separate issue.
- Section 1.b.i requires license copies and prominent Boson/Meta attribution for
  covered distributions/products/services; its model/software naming clause
  needs review for the particular integration. Sections 1.b.iv and 5.c add the
  acceptable-use policy and indemnity.

Disposition: keep the existing distinct `omnivoice-terms` gate. Supplying it
affirms acceptance of both the NC main weights and companion terms, eligibility
and compliance, including any separately required Boson permission. These remain
the user's obligations. Our distribution/notices and any naming obligations that
apply to our integration still need to be fulfilled; a switch does not transfer
them to users. PR521's speech-quality acceptance gate remains unchanged.

Sources: [exact bundled Boson terms](https://huggingface.co/k2-fsa/OmniVoice/blob/c5fdb5ccb189668d56333f77ba2629f4cd7535f4/audio_tokenizer/LICENSE),
[incorporated Llama3 agreement](https://www.llama.com/llama3/license/),
[incorporated use policy](https://www.llama.com/llama3/use-policy/).

The converted bundle now includes the full Boson/Meta texts, acceptable-use policy
and NOTICE, plus corrected card attribution/provenance, at immutable HF revision
[`a2ad263`](https://huggingface.co/cstr/omnivoice-GGUF/tree/a2ad2630338fc80d1a0565c8e58aa8eff615db85).
All five text files were anonymously downloaded and hashed; no weights changed.
[Publication receipt](historical/omnivoice-license-notices-2026-10-22.json). These notices do
not certify every downstream use or resolve every possible naming question.
