# Accepted-source Linux package preflight

The bounded preflight freezes source
`998f8d18e536bf503fbd4c655caa01a1d461836f` on
`diag/release-preflight-source-20261015`. This accepted source excludes the
unmerged Dia intervention and OmniVoice runtime changes. It still identifies
as version 0.8.43; these are not v0.8.44 assets or the final release freeze.

[CLI run38742749435](https://github.com/CrispStrobe/CrispASR/actions/runs/38742749435)
and [library run38742753167](https://github.com/CrispStrobe/CrispASR/actions/runs/38742753167)
use the existing Release workflow's empty-tag dry run, selecting only
`build-linux` and `build-libs-linux-x86_64`. Both event metadata records match
the frozen SHA and validation branch. Both runs are terminal-successful and the bounded packaging preflight passes. No tag or
publication was requested. The v0.8.44 tag is absent.

Existing local guards pass against byte-identical frozen scripts:

- All 11 release workflow tests, with no skips.
- Runtime bundler: relocated execution, unresolved dependency rejection,
  host-runtime handling and already-staged dependency controls.
- Shared-library bundler: dependency closure, actual relocated loading/call,
  preserved directory and SOVERSION symlinks, and host-toolkit RUNPATH.
- All 21 redistributable x86 jobs pin portable CPU flags.
- Glint's sync marker exactly matches upstream main at
  `77738f3ed9b15f627196cc5bbd7f6406814ba2fb`.

The JSON receipt retains full guard output, nine exact source-file hashes and
four terminal-cancelled superseded secret scans. Current scans and required
science/lint jobs remain intact. All evidence is under
`${CRISPASR_STORAGE}/crispasr/release-preflight-20261015`; tiny symlink-dependent test
fixtures used local temporary storage and were removed by their own traps.

The CLI artifact11980941526 passes its complete official SHA256, CRC and ten
unique safe tar-member checks. Relocated startup reports0.8.43 and998f8d18.
The quantizer has no version switch; its documented usage path executes with
exit1. Actual ldd resolves bundled libraries beside both executables, and the
dependency checker accepts all seven ELF files with `$ORIGIN` RUNPATHs.
All release attachment, creation and notes steps are terminal-skipped.

The library artifact11981646466 passes full SHA256/CRC, 49 unique safe tar
members and every original symlink target. Its raw SHA256 is
`be88c50b50e8e9b7906f1ed9b93cd11c92b50640dc82050d751932376d1d67ce`. The actual hosted symlink bundle passes
relocated dependency closure/dlopen and session-ABI checks. Locally, CIFS cannot
create those symlinks: the immutable tar remains untouched, its two directory
aliases are audited, and nine file aliases are explicitly materialized as the
wheel stager expects. That materialized bundle also passes relocated loading.
This is not a local native-symlink preservation execution claim.

The wheel uses the exact frozen Python tree and actual shared bundle, with the
existing `stage_libs.py`, compiled legacy helper and Linux wheel tagging. The
built wheel is `crispasr-0.8.43-py3-none-linux_x86_64.whl`, 53352316 bytes,
SHA256 `b26d6d1b496e07c318c5bb7171c6da7018f7a2c1550ee8ccda8534c18dcf90c2`. Every Python member matches the frozen commit;
CRC, unique safe paths and actual installed-library bytes pass. An isolated
Python3.13 environment installs it with NumPy2.3.5. From an unrelated directory,
without loader or Python path overrides, library discovery resolves inside the
installed package, both native/helper libraries load, and all116 backend names
remain stable across repeated C-ABI calls and missing-model error paths. All18
wheel ELF files pass dependency closure with `$ORIGIN` RUNPATHs. Large packages,
source, installed site-packages and logs remain under ${CRISPASR_STORAGE}.

The two initial smoke-harness assumptions (a quantizer version switch and a
`dia-tts` C-ABI name) were corrected against frozen source; original failed
scripts/logs remain retained. They required no runtime or package change.

Both workflows' release attachment/creation/notes steps are terminal-skipped.
No tag, publication or version bump occurred. This bounded preflight completes
the accepted-source packaging task; it does not replace model/GPU acceptance,
the full platform matrix, a final source freeze, applicable final-tip CI or the
later version-bump process. The JSON receipt retains the complete audit,
original archive link table, wheel identity and verification log hashes.
