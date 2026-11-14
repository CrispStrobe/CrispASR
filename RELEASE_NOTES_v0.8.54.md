# v0.8.54 — VibeVoice task detection

VibeVoice base TTS and ASR are different model tasks. The runtime now detects legacy TTS checkpoints that were mislabeled as ASR and refuses to transcribe with them. It also refuses to synthesize speech with an ASR-only checkpoint, with a task-specific diagnostic. The converter requires the source `config.architectures` and records the model architecture and ASR capability in new GGUFs.

An independent Mac M1 test of the broader, unreleased PR #590 candidate (`dc49a24`) reported an exact JFK transcript from VibeVoice-ASR Q4 on Metal and CPU. For 1.5B TTS Q4 on Metal, it reported 0% WER in default-disclosure and requested-only clone tests, with clone/reference speaker cosine 0.785 and 0.797 versus 0.532 for the baseline. Those results describe that experimental candidate, not a general quality claim for this release.

7B long-text and two-speaker cloning, and physical Vulkan inference, remain under investigation in #589. This release does not qualify those cases.
