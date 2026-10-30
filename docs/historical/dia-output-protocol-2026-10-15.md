# Dia output protocol: original witnesses and candidate repair

The pinned original Python generation/output methods demonstrate three defects
in the current native protocol: the final return slice includes an extra row,
the EOS countdown is not restarted at the capacity boundary, and invalid codec
indices cause complete frames to be dropped rather than individually zeroed.
The repair is integrated as6503408cb95d42e80ad84bee513ae58a67e616c5. Full
speech acceptance remains open.

The reference executes actual `_prepare_audio_prompt`, `generate`,
`_generate_output`, `DecoderOutput` and delay-reversal methods from original
Dia `4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8`, using Torch2.6.0 CPU.
Only the neural prediction and final DAC entry are replaced: predictions are
synthetic known rows, and the DAC entry retains its exact codec inputs. No
neural model or codec weights are loaded. Source hashes and literal extracted
methods are retained; the configurable reference generator reproduces the
committed fixture byte for byte, SHA256
`bb3db2bf3d52f419ecad359f5f44fbbd1dff9cb88a3f095c38f979362d0e246f`.
The original model capacity remains3072 even for small generation caps; a first
reference setup using the cap as model capacity failed before inference, and
that raw failure is preserved separately from the corrected run.

The native guard compiles the actual production stopping, BOS initialization,
emission/stop block and delay reversal bodies without changing them. Sixteen
cases cover caps1/7/14/15/16/17/32/64, EOS at steps0/1/2/20, EOS followed by the
capacity boundary, invalid code indices, and forced input/output diagnostics.
The old native source fails12 cases. The repaired source passes all16, with
exact original feedback rows, return slices and complete codec inputs.
Restoring each individual defect is rejected: stopping fails1 case, reversal
fails4, and retaining the final row fails10. These are original-reference
comparisons, rather than tests comparing the patch with itself.

The patch excludes the last written row only on a real stop; a short generation
that exhausts its loop and forced diagnostic modes retain all their rows.
Invalid entries become zero without removing other channels or changing frame
timing. The original capacity condition is reproduced with signed arithmetic.
Both CLI and session C ABI call the repaired shared `dia_tts_synthesize`.

[Complete50-payload proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/062b2bed6188680bf3c45677354cf8cd5f22367f/dia-20261015-output-protocol/complete-output-protocol-v1.zip)
contains the original sources/methods/license, fixture and generator, old and
candidate runtime sources, generated C++ harness and actual executable,
all16 returned arrays, failed old-source and mutation logs, compiler metadata,
and payload hashes. The183,210-byte ZIP is anonymously downloaded and SHA256
verified: `442d1a2c6f1d6085892cecac124617acb58f72c74634ddc7db46a1255677cc09`.
The actual extracted-protocol ELF SHA256 is
`2e3559eeda1d7c934d7759326770b8091884e6cf18bc827a9fcc90c8247d7350`.
It is a protocol executable, not a full model executable. The receipt binds its
actual production source hash independently of commit and compiler identity.

Candidate `fc1b05c922edfeece675a32f1b7855f588276968` is pushed.
[Fresh general CI](https://github.com/CrispStrobe/CrispASR/actions/runs/38713471171)
and [lint](https://github.com/CrispStrobe/CrispASR/actions/runs/38713474686)
target that exact source. All13 general CI jobs pass, including execution of
the new original-protocol guard; all10 lint jobs also pass. The guard is wired into general CI. The
[complete-code original-DAC diagnosis](dia-full-codec-preparation-2026-10-15.md)
is independently complete locally: original WER0, both native-code variants
WER1. The same repaired protocol also matches all1115 feedback/1114 returned
rows/1099 codec frames on both real CLI/C-ABI captures. That complete proof is
linked in the codec receipt. The repair is rebased/ff integrated; all five
tested source/workflow/fixture files remain byte-identical tofc1b05c9. The
accepted feature branch is removed with its source worktree preserved. Main-tip
CI runs separately.
No neural arithmetic, sampler or default is changed. Full free-generation,
CLI/C-ABI speech, original intermediate diff, CUDA, Q4 and speed remain separate
gates. Current-source output must be measured rather than inferring its PCM
length from the older retained producer: their invalid-code reversal differs.

[Structured receipt](dia-output-protocol-2026-10-15.json).
