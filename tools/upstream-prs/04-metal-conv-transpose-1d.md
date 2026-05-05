**Title:** `metal : tighten input-position loop in kernel_conv_transpose_1d`

---

`kernel_conv_transpose_1d` iterates the full `IL` input range per output
position and filters with an `if`. For a given output position `j`,
only input positions `i` with `i*s0 <= j < i*s0 + K` contribute — i.e.
`i ∈ [⌈(j - K + 1) / s0⌉, ⌊j / s0⌋] ∩ [0, IL-1]`, at most `⌈K/s0⌉`
values (typically 2 for stride==K/2 transposed convs).

For a representative codec-decoder shape (IL=320, K=10, s0=5) that's
`IL / ⌈K/s0⌉ = 160×` more iterations than necessary, all already
filtered out by the `if`. On Apple M1 the wasted work trips the macOS
GPU watchdog (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`)
on long graphs.

Fix: compute `i_min, i_max` analytically before the inner loop, iterate
only `[i_min, i_max]`. Bit-identical output (same multiplies and adds
in the same order). Loop bound shrinks by `IL / ⌈K/s0⌉`.

Patch: `04-metal-conv-transpose-1d.patch` (1 file, +29/-7).

**Verification.** Tested on M1 / M3 Pro / M4 Pro via downstream
consumer (qwen3-tts codec at full T_codec). End-to-end codec decode
~3-4× faster on M1; zero watchdog hits across long synthesis runs vs.
~30% pre-patch. Recommend running existing `test-backend-ops`
conv_transpose_1d cases under Metal to confirm bit-identical output.
