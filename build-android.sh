#!/bin/bash
# Build CrispASR (whisper.cpp) for Android via NDK.
#
# Usage:
#   ./build-android.sh                      # All ABIs
#   ./build-android.sh --abi arm64-v8a      # Single ABI
#   ./build-android.sh --vulkan             # With Vulkan GPU

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build-android"
API_LEVEL=24
VULKAN=OFF
ABIS=("arm64-v8a" "armeabi-v7a" "x86_64")

NDK="${ANDROID_NDK_HOME:-${NDK_HOME:-${ANDROID_HOME}/ndk-bundle}}"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --abi)     ABIS=("$2"); shift ;;
        --vulkan)  VULKAN=ON ;;
        --api)     API_LEVEL="$2"; shift ;;
        --ndk)     NDK="$2"; shift ;;
        --clean)   rm -rf "$BUILD_DIR"; echo "Cleaned."; exit 0 ;;
    esac
    shift
done

TOOLCHAIN="$NDK/build/cmake/android.toolchain.cmake"
if [ ! -f "$TOOLCHAIN" ]; then
    echo "[ERROR] NDK toolchain not found. Set ANDROID_NDK_HOME."
    exit 1
fi

for ABI in "${ABIS[@]}"; do
    echo "=== Building for $ABI ==="
    cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR/$ABI" \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
        -DANDROID_ABI="$ABI" \
        -DANDROID_NATIVE_API_LEVEL="$API_LEVEL" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DWHISPER_BUILD_EXAMPLES=OFF \
        -DWHISPER_BUILD_TESTS=OFF \
        -DGGML_VULKAN="$VULKAN"

    cmake --build "$BUILD_DIR/$ABI" -j"$(nproc 2>/dev/null || echo 4)"

    ls -lh "$BUILD_DIR/$ABI/src/libwhisper.so" 2>/dev/null || true
done

echo "=== Android build complete ==="
