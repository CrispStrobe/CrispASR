#!/usr/bin/env bash
# Fetch a pinned official CPU SDK. Prints its root for CMake; no Python needed.
# Source: Microsoft's official Microsoft.ML.OnnxRuntime NuGet package (one
# archive for every platform). The github.com/microsoft/onnxruntime release
# tarballs previously used here started returning 404 in 2026-11.
set -euo pipefail
version=1.30.0
digest=30df119bd9164eaa187666d93147448fdd4a9a1a3caf83f1505daf7af2d1c5d2
case "$(uname -s)/$(uname -m)" in
  Darwin/arm64) rid=osx-arm64 ;;
  Linux/x86_64) rid=linux-x64 ;;
  Linux/aarch64) rid=linux-arm64 ;;
  *) echo 'No pinned CPU SDK for this platform; set CRISPASR_ONNXRUNTIME_ROOT to your SDK.' >&2; exit 1 ;;
esac
package="microsoft.ml.onnxruntime.$version-$rid"
root="${1:-$(cd "$(dirname "$0")/.." && pwd)/build/onnxruntime-sdk}"
if [[ -f "$root/.crispasr-sdk" && "$(cat "$root/.crispasr-sdk")" == "$package $digest" && -f "$root/include/onnxruntime_cxx_api.h" ]]; then
  echo "$root"; exit 0
fi
archive="$(mktemp -t crispasr-ort.XXXXXXXX)"
staging="$(mktemp -d -t crispasr-ort-unpack.XXXXXXXX)"
trap 'rm -f "$archive"; rm -rf "$staging"' EXIT
curl -fsSL --retry 3 "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$version/microsoft.ml.onnxruntime.$version.nupkg" -o "$archive"
if command -v sha256sum >/dev/null; then actual="$(sha256sum "$archive" | cut -d' ' -f1)"; else actual="$(shasum -a 256 "$archive" | cut -d' ' -f1)"; fi
if [[ "$actual" != "$digest" ]]; then echo 'ONNX Runtime SDK checksum mismatch' >&2; exit 1; fi
unzip -q "$archive" 'build/native/include/*' "runtimes/$rid/native/*" -d "$staging"
if [[ ! -f "$staging/build/native/include/onnxruntime_cxx_api.h" ]] || ! ls "$staging/runtimes/$rid/native/"libonnxruntime.* >/dev/null 2>&1; then
  echo 'ONNX Runtime package is missing the expected SDK headers or library' >&2
  exit 1
fi
rm -rf "${root:?}/include" "${root:?}/lib"
mkdir -p "$root/include" "$root/lib"
cp -R "$staging/build/native/include/." "$root/include/"
cp -R "$staging/runtimes/$rid/native/." "$root/lib/"
# The Linux library's SONAME is libonnxruntime.so.1; the release tarballs shipped
# that name, the NuGet package does not (copied: some filesystems refuse symlinks).
if [[ -f "$root/lib/libonnxruntime.so" ]]; then cp -p "$root/lib/libonnxruntime.so" "$root/lib/libonnxruntime.so.1"; fi
printf '%s %s\n' "$package" "$digest" > "$root/.crispasr-sdk"
echo "$root"
