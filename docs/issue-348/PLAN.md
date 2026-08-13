# Issue #348 — Chatterbox Multilingual V3 parity port

## NOW

- [x] Read the full issue and audit the current official GitHub, model repo,
  and production V3 Space.
- [x] Pin source model revision
  `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` and production pairing
  `t3_mtl23ls_v3.safetensors` + original `s3gen.pt` / `s3gen.safetensors`.
- [x] Correct converter checkpoint selection and make the V3 pair explicit.
- [x] Make the Python diff blueprint multilingual and drive all clone
  conditionals from the fixture voice rather than unrelated built-in conds.
- [x] Verify `.safetensors` converter inputs tensor-for-tensor against the
  upstream `.pt` states; convert explicit V3 F16 T3 and S3Gen GGUFs.
- [x] Quantize with `crispasr-quantize`; use staged F16/Q8/Q4 parity to set
  precision carve-outs for sampling, CFM trajectory, and vocoder tensors.
- [ ] Dump a multilingual voice-clone `-ref.gguf`, run `crispasr-diff` from
  front ends through AR tokens, CFM mel, vocoder stages, and PCM, and publish
  it under `chatterbox-v3/<fixture>/ref.gguf` in regression fixtures. (The
  archive and 32-pass local run are complete; publication remains part of the
  final artifact checkpoint.)
- [ ] Run local unit/ABI/CLI tests plus multilingual live synthesis and closed
  loop R/C/B TTS→ASR clone roundtrips with speaker-similarity evidence.
- [ ] Add the exact V3 artifacts to the registry and verify autodownload into
  `/Volumes/backups/ai/crispasr-gguf`.
- [ ] Repeat conversion/parity/live roundtrips with CUDA in a dedicated Kaggle
  kernel, preserving `progress.txt` and downloadable result artifacts.
- [ ] Rebase, merge only with green CI, then answer #348 with linked evidence.

## Acceptance gates

- The Python oracle imports the actual multilingual V3 driver and passes an
  explicit ISO language ID through the multilingual tokenizer.
- Every GGUF records the selected upstream checkpoint; generic fallback cannot
  silently replace V3 with V2 or pair V3 T3 with the retired V3 vocoder.
- `crispasr-diff` reports both cosine and norm/scale checks. Quant acceptance is
  based on decoded output and roundtrip results, not cosine alone.
- Cross-language cloning uses a real generated reference R, clone C, and
  baseline B: C is non-silent, ASR(C) preserves the target text, and speaker
  similarity satisfies `cos(C,R) > cos(B,R)`.
- CPU/Metal success is not CUDA evidence; the same pinned commit and artifacts
  must pass on a real Kaggle NVIDIA GPU.

## Local evidence checkpoint

- Exact upstream state audit: S3Gen safetensors has 2,489/2,489 shared tensors
  byte-identical to `s3gen.pt`; its only omitted key is the upstream-declared
  non-persistent tokenizer window. VoiceEncoder has 16/16 exact tensors.
- `crispasr-diff`, German/JFK reference, canonical Q4: **32 pass, 0 fail,
  2 intentional skips**. T3 condition cosine/norm = 0.995116/0.9995; S3Gen
  encoder = 0.988826/0.9963; CFM mel = 0.994839/0.9775; isolated vocoder and
  PCM stages are effectively 1.0.
- Q4 policy: S3Tokenizer proj-down cosine improves 0.999477 -> 0.999929 and
  downstream T3 condition cosine 0.9855 -> 0.9951 with the Q8 floor. The
  built-in quantizer reproduces the manually tested artifacts byte-for-byte.
- Closed-loop generated R/C/B: Kokoro-generated English R -> Chatterbox German
  C roundtrips the complete target through Parakeet V3. TitaNet speaker cosine
  is `C,R = 0.792725` vs un-cloned `B,R = 0.431873`.
- Hermetic Chatterbox/registry/Parakeet language-routing set: 76/76 pass;
  public CLI capability/dispatch tests: 12/12 pass; converter tests: 5/5 pass.
