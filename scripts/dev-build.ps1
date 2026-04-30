# dev-build.ps1 — Windows analogue of dev-build.sh.
# Picks Ninja, ccache (when present), and a Release config.
# Usage:
#   .\scripts\dev-build.ps1                           # build crispasr
#   .\scripts\dev-build.ps1 -Reconfigure              # nuke build/ first
#   .\scripts\dev-build.ps1 -Target crispasr-quantize # build different target
#   .\scripts\dev-build.ps1 -DGGML_VULKAN=ON          # extra cmake args
#
# Run from a Visual Studio "x64 Native Tools" prompt or any shell where
# `cmake` and `ninja` are on PATH and a C++ compiler is selectable.

param(
    [switch]$Reconfigure,
    [string]$Target = "crispasr",
    [Parameter(ValueFromRemainingArguments)] [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path $PSScriptRoot "..")

$args = @(
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCRISPASR_BUILD_TESTS=OFF"
)

# ccache integration is auto-detected by ggml's CMakeLists when the
# binary is on PATH. No extra flag needed.
if (-not (Get-Command "ninja" -ErrorAction SilentlyContinue)) {
    Write-Host "WARNING: ninja not on PATH. Install via 'choco install ninja' or 'scoop install ninja'."
}

if ($Reconfigure -or -not (Test-Path "build") -or $ExtraArgs) {
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    Write-Host "Configuring with: $($args + $ExtraArgs)"
    & cmake -S . -B build @args @ExtraArgs
}

& cmake --build build --target $Target -j
