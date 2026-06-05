@echo off
setlocal enabledelayedexpansion

:: Find vswhere.exe
set "vswhere=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "!vswhere!" (
    echo [ERROR] vswhere.exe not found. Please install Visual Studio 2022.
    exit /b 1
)

:: Find VS Installation Path (-products * includes Build Tools installations)
for /f "usebackq tokens=*" %%i in (`"!vswhere!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "vs_path=%%i"
)

if not defined vs_path (
    echo [ERROR] Visual Studio C++ build tools not found.
    exit /b 1
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
echo [INFO] Configuring project with Ninja...
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release %*

if %ERRORLEVEL% neq 0 (
    echo [ERROR] CMake configuration failed.
    exit /b 1
)

:: Build
echo [INFO] Building crispasr...
cmake --build build --target crispasr-cli

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Build failed.
    exit /b 1
)

if not exist "build\bin\crispasr.exe" (
    echo [ERROR] Build succeeded but build\bin\crispasr.exe not found.
    exit /b 1
)

echo [SUCCESS] Build complete. Binary is at build\bin\crispasr.exe
