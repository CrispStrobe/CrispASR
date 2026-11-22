#!/bin/bash
# run-licensed-live.sh LICENCE CMD [ARGS...]
#
# Live tests that auto-download a model under a restricted licence must not
# accept that licence on the user's behalf. The CLI refuses the download unless
# CRISPASR_ACCEPT_LICENSE names the licence; in that case report the test as
# SKIPPED (exit 77) with the reason, instead of failing it.
licence="$1"
shift
out=$("$@" 2>&1)
rc=$?
printf '%s\n' "$out"
if printf '%s' "$out" | grep -q "refusing to download without explicit licence acceptance"; then
    echo "SKIP: needs CRISPASR_ACCEPT_LICENSE=$licence (restricted model licence; not accepted implicitly)"
    exit 77
fi
exit $rc
