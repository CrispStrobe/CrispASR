#!/bin/bash
# Build CrispASR (whisper.cpp) for iOS (xcframework with Metal GPU).
#
# Usage:
#   ./build-ios.sh              # arm64 device + simulator
#   ./build-ios.sh --device     # device only

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build-ios"

CMAKE_COMMON=(
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=OFF
    -DWHISPER_BUILD_EXAMPLES=OFF
    -DWHISPER_BUILD_TESTS=OFF
    -DGGML_METAL=ON
    -DGGML_METAL_EMBED_LIBRARY=ON
)

DEVICE_ONLY=0
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --device)    DEVICE_ONLY=1 ;;
        --clean)     rm -rf "$BUILD_DIR"; echo "Cleaned."; exit 0 ;;
    esac
    shift
done

# Device (arm64)
echo "=== Building for iOS device (arm64) ==="
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR/device" \
    -G Xcode \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    "${CMAKE_COMMON[@]}"
cmake --build "$BUILD_DIR/device" --config Release -- -quiet

# Simulator
if [ $DEVICE_ONLY -eq 0 ]; then
    echo "=== Building for iOS simulator (arm64) ==="
    cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR/simulator" \
        -G Xcode \
        -DCMAKE_SYSTEM_NAME=iOS \
        -DCMAKE_OSX_SYSROOT=iphonesimulator \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        "${CMAKE_COMMON[@]}"
    cmake --build "$BUILD_DIR/simulator" --config Release -- -quiet

    echo "=== Creating xcframework ==="
    rm -rf "$BUILD_DIR/CrispASR.xcframework"
    xcodebuild -create-xcframework \
        -library "$BUILD_DIR/device/Release-iphoneos/libwhisper.a" \
        -headers "$SCRIPT_DIR/include/whisper.h" \
        -library "$BUILD_DIR/simulator/Release-iphonesimulator/libwhisper.a" \
        -headers "$SCRIPT_DIR/include/whisper.h" \
        -output "$BUILD_DIR/CrispASR.xcframework"
    echo "=== Built: $BUILD_DIR/CrispASR.xcframework ==="
fi

echo "Done."
