# MUSA S5000 build candidate

`.devops/main-musa.Dockerfile` uses the official SDK 5.2 S5000 image pinned by
digest and builds only architecture 31. SDK 4.2 and 4.3 failed WKV compilation;
the [SDK 5.2 probe](historical/musa-sdk52-2026-10-26.json) compiled the unchanged WKV source
and linked the complete CLI/backend. This proves the tested build configuration,
without qualifying older architectures or GPU output.

The Docker recipe now uses that configuration, with official muAlg/muThrust
1.17.2 packages, matching SDK build/runtime bases, two compiler workers and
retained shared libraries, entrypoint and notice metadata. The runtime currently
retains the full SDK base. Its packaging and model-free runtime checks still
require validation on a MUSA machine; no new image is published.

## Platform and redistribution terms

Moore Threads' [general software agreement](https://www.mthreads.com/software_license_agreement)
expressly covers MUSA SDK. Section II.2(7) restricts use on non-Moore-Threads
platforms without written authorization; section II.1 makes binary
redistribution conditional, including preservation of binaries and delivery of
the agreement. Third-party component terms also apply. Copyright metadata that
only names a proprietary license does not establish these permissions.

Until applicable SDK-specific exceptions or written authorization are
established, the repository conservatively restricts further SDK execution to
dedicated Moore Threads machines. The earlier hosted compiler result remains
historical technical evidence, not licensing clearance. No performance benchmark
or hardware/speech acceptance is claimed by that experiment.

## Validation and publication status

The manual **Validate MUSA S5000 image** workflow requires a self-hosted Linux
x64 runner labelled `musa-s5000`. Before SDK use it checks for the vendor's PCI
identity, load, available memory and storage. The PCI vendor identity follows
the [vendor's installation guide](https://docs.mthreads.com/musa-sdk/version-5.1.0/install_guide/).
Assign this label only to an appropriately configured S5000 host.

The workflow builds the actual Dockerfile, checks CLI version, entrypoint,
project notices and shared-library dependency closure, and retains text/metadata
evidence. It has read-only repository permissions and does not push an image.
It is prepared but has not yet run on a self-hosted MUSA machine.

MUSA is excluded from the automatic Docker publisher. Other flavours continue
through the existing publisher. The old SDK 4.3 recovery workflow was removed;
the original failed run receipts remain available. The SDK probe is also manual
and restricted to the MUSA runner, without destructive host cleanup commands.

Before enabling MUSA image publication, complete the applicable license and
notice review, validate the actual runtime image, and run backend operation and
speech checks on the target hardware. Generic `main-musa` tags must not imply
support for untested S80/S4000 architectures.

## October 26 permission and runner recheck

The complete current agreement was reread, including section II.2(6): public
performance/benchmark disclosure needs the vendor's written permission. A
hardware runner alone does not resolve that separate condition. No applicable
written exception has been supplied, and the repository runner API currently
lists zero self-hosted runners. No new SDK execution, image build or inference
was dispatched in this review. [Review receipt](historical/musa-permissions-review-2026-10-26.json).

To resume, provide an appropriately configured and authorized S5000 host, plus
applicable SDK-specific terms or written authorization covering any proposed
exception. Before distributing the full SDK-based runtime, identify every vendor
and third-party component, establish permission for that actual payload, and
include the complete applicable agreements and notices. The earlier fourteen
package copyright metadata records do not clear the full image. Before publishing
performance numbers, separately establish permission for their disclosure.
The manual image validator is a packaging check; operation and speech parity
still require target-hardware tests. Neither successful compilation nor a PCI
vendor match proves those checks passed.

The hosted policy/CPU Docker smoke run [40402386515](https://github.com/CrispStrobe/CrispASR/actions/runs/40402386515)
passes, including all eight MUSA policy controls. It does not execute the MUSA
SDK or build/qualify the MUSA runtime image. The user confirmed that neither
the target host nor applicable new permissions are available yet.
