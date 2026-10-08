#!/usr/bin/env bash
# Fetch a pinned official CPU SDK. Prints its root for CMake; no Python needed.
set -euo pipefail
case "$(uname -s)/$(uname -m)" in
  Darwin/arm64) package=onnxruntime-osx-arm64-1.30.0; digest=6ebb5062a934537c352937821f9fe9718e7de1a2db1122a93dd363ffd53a7012 ;;
  Linux/x86_64) package=onnxruntime-linux-x64-1.30.0; digest=a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd ;;
  Linux/aarch64) package=onnxruntime-linux-aarch64-1.30.0; digest=e16a27a8ed330bbc698df7330b0cf56e722f354e3bcc92118682c74ef3c3e3da ;;
  *) echo 'No pinned CPU SDK for this platform; set CRISPASR_ONNXRUNTIME_ROOT to your SDK.' >&2; exit 1 ;;
esac
root="${1:-$(cd "$(dirname "$0")/.." && pwd)/build/onnxruntime-sdk}"
if [[ -f "$root/.crispasr-sdk" && "$(cat "$root/.crispasr-sdk")" == "$package $digest" && -f "$root/include/onnxruntime_cxx_api.h" ]]; then
  echo "$root"; exit 0
fi
archive="$(mktemp -t crispasr-ort.XXXXXXXX)"
staging="$(mktemp -d -t crispasr-ort-unpack.XXXXXXXX)"
trap 'rm -f "$archive"; rm -rf "$staging"' EXIT
curl -fsSL --retry 3 "https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/$package.tgz" -o "$archive"
if command -v sha256sum >/dev/null; then actual="$(sha256sum "$archive" | cut -d' ' -f1)"; else actual="$(shasum -a 256 "$archive" | cut -d' ' -f1)"; fi
if [[ "$actual" != "$digest" ]]; then echo 'ONNX Runtime SDK checksum mismatch' >&2; exit 1; fi
mkdir -p "$root"
# Some official archives prefix the package directory with './'. Stripping
# one component leaves an extra directory on macOS. Extract first so both
# archive layouts resolve to the same SDK root.
tar -xzf "$archive" -C "$staging"
sdk="$staging/$package"
if [[ ! -f "$sdk/include/onnxruntime_cxx_api.h" ]]; then
  echo 'ONNX Runtime archive is missing the expected SDK headers' >&2
  exit 1
fi
cp -R "$sdk/." "$root/"
printf '%s %s\n' "$package" "$digest" > "$root/.crispasr-sdk"
echo "$root"
