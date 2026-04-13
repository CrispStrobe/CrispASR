@echo off
setlocal enabledelayedexpansion

:: Find vswhere.exe
set "vswhere=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "!vswhere!" (
    echo [ERROR] vswhere.exe not found. Please install Visual Studio 2022.
    exit /b 1
)

:: Find VS Installation Path
for /f "usebackq tokens=*" %%i in (`"!vswhere!" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "vs_path=%%i"
)

if not defined vs_path (
    echo [ERROR] Visual Studio C++ build tools not found.
    exit /b 1
)

:: Check for Vulkan SDK
if not defined VULKAN_SDK (
    if exist "C:\VulkanSDK" (
        for /f "delims=" %%a in ('dir /b /ad /on "C:\VulkanSDK"') do set "VULKAN_VER=%%a"
        set "VULKAN_SDK=C:\VulkanSDK\!VULKAN_VER!"
        echo [INFO] Found Vulkan SDK at !VULKAN_SDK!
    ) else (
        echo [ERROR] VULKAN_SDK not defined and C:\VulkanSDK not found.
        exit /b 1
    )
)

:: Initialize MSVC Environment
set "vcvars=!vs_path!\VC\Auxiliary\Build\vcvars64.bat"
if not exist "!vcvars!" (
    echo [ERROR] vcvars64.bat not found at !vcvars!
    exit /b 1
)

echo [INFO] Initializing MSVC environment...
call "!vcvars!"

:: Run CMake
echo [INFO] Configuring project with Ninja into build-vulkan...
cmake -G Ninja -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_CUDA=OFF %*

if %ERRORLEVEL% neq 0 (
    echo [ERROR] CMake configuration failed.
    exit /b 1
)

:: Build
echo [INFO] Building whisper-cli...
cmake --build build-vulkan --target whisper-cli

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Build failed.
    exit /b 1
)

echo [SUCCESS] Build complete. Binary is at build-vulkan\bin\crispasr.exe
