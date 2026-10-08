param([ValidateSet('OFF', 'ON')][string]$Mode)
$ErrorActionPreference = 'Stop'

function Cache-Value([string]$key) {
    $line = Get-Content build/CMakeCache.txt | Where-Object { $_ -match "^${key}:[^=]+=" }
    if (@($line).Count -ne 1) { throw "Missing/ambiguous cache key $key" }
    return ($line -split '=', 2)[1]
}
$expected = @{
    GGML_CUDA = 'ON'; GGML_CUDA_FORCE_MMQ = $Mode
    CMAKE_CUDA_ARCHITECTURES = '61-virtual;80-virtual'
    GGML_NATIVE = 'OFF'; GGML_AVX2 = 'ON'; GGML_FMA = 'ON'; GGML_F16C = 'ON'
}
foreach ($key in $expected.Keys) {
    if ((Cache-Value $key) -ne $expected[$key]) { throw "Wrong configuration: $key" }
}
$commands = Get-Content -Raw build/compile_commands.json | ConvertFrom-Json
$cudaCommands = @($commands | Where-Object { $_.file -like '*.cu' })
if ($cudaCommands.Count -lt 2) { throw 'No CUDA kernel compile commands' }
foreach ($command in $cudaCommands) {
    $hasFlag = $command.command -match '(?:^|\s)(?:-D|/D)GGML_CUDA_FORCE_MMQ(?:[=\s]|$)'
    if ($hasFlag -ne ($Mode -eq 'ON')) { throw "Wrong MMQ compiler definition: $($command.file)" }
}

$name = "crispasr-windows-cuda126-ptx-mmq-$Mode"
$stage = Join-Path (Get-Location) "release/$name"
New-Item -ItemType Directory -Force $stage | Out-Null
Copy-Item build/bin/crispasr.exe, build/bin/crispasr-quantize.exe $stage
Get-ChildItem build/bin/*.dll | Where-Object { $_.Name -notlike 'Catch2*' } | Copy-Item -Destination $stage
Copy-Item LICENSE, THIRD_PARTY_NOTICES.txt $stage
$runtime = @(
    Get-ChildItem "$env:CUDA_PATH/bin/cudart64_*.dll"
    Get-ChildItem "$env:CUDA_PATH/bin/cublas64_*.dll"
    Get-ChildItem "$env:CUDA_PATH/bin/cublasLt64_*.dll"
)
if ($runtime.Count -ne 3) { throw 'Expected exactly three matching CUDA runtime DLLs' }
$runtime | Copy-Item -Destination $stage
$hashes = [ordered]@{}
foreach ($dll in $runtime) {
    $hashes[$dll.Name] = (Get-FileHash (Join-Path $stage $dll.Name) -Algorithm SHA256).Hash.ToLowerInvariant()
}
$cudart = Join-Path $stage ($runtime | Where-Object Name -like 'cudart64_*').Name
$signature = '[DllImport(@"' + $cudart + '")] public static extern int cudaRuntimeGetVersion(out int version);'
Add-Type -MemberDefinition $signature -Name RuntimeProbe -Namespace CrispASR
$runtimeVersion = 0
$rc = [CrispASR.RuntimeProbe]::cudaRuntimeGetVersion([ref]$runtimeVersion)
if ($rc -ne 0 -or $runtimeVersion -ne 12060) { throw "Wrong staged CUDA runtime: rc=$rc version=$runtimeVersion" }

# Remove build/toolkit paths: the staged package must find its own DLLs.
$originalPath = $env:PATH
$env:PATH = ($originalPath -split ';' | Where-Object { $_ -notmatch '(?i)CUDA|\\build\\bin' }) -join ';'
try {
    foreach ($argument in @('--version', '--diagnostics')) {
        $stem = $argument.TrimStart('-')
        $stdout = Join-Path (Get-Location) "release/$stem.stdout.txt"
        $stderr = Join-Path (Get-Location) "release/$stem.stderr.txt"
        $process = Start-Process -FilePath "$stage/crispasr.exe" -WorkingDirectory $stage -ArgumentList $argument `
            -Wait -PassThru -NoNewWindow -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $output = (Get-Content -Raw $stdout) + (Get-Content -Raw $stderr)
        if ($process.ExitCode -ne 0 -or $output -notmatch 'cuda toolkit\s*:\s*12\.6\b' -or
            $output -notmatch 'cuda runtime ABI\s*:\s*12\b') { throw "Staged $argument failed: $output" }
        Write-Host $output
    }
} finally {
    $env:PATH = $originalPath
}
$manifest = [ordered]@{
    commit = (git rev-parse HEAD).Trim()
    ggml_commit = (git -C ggml rev-parse HEAD).Trim()
    toolkit = '12.6.3'; runtime_version = $runtimeVersion
    architectures = $expected.CMAKE_CUDA_ARCHITECTURES
    force_mmq = $Mode
    cpu_floor = @{ native = 'OFF'; avx2 = 'ON'; fma = 'ON'; f16c = 'ON' }
    runtime_sha256 = $hashes
    cuda_compile_commands_verified = $cudaCommands.Count
    gpu_execution = $false
    scope = 'Compile, staged DLL/runtime and driverless startup checks; no GPU performance acceptance'
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content release/manifest.json -Encoding utf8
Copy-Item release/manifest.json $stage
@"
Experimental #483 package: CUDA 12.6.3, PTX 61/80, FORCE_MMQ=$Mode.
Compare with the other ON/OFF package from the same workflow run.
All source, architecture, CPU-floor and runtime settings are matched.
Hosted checks cover compilation and driverless startup. Actual GTX1660/MX150
GPU transcript correctness and performance still need to be measured.
"@ | Set-Content "$stage/EXPERIMENT.txt"
Compress-Archive -Path $stage -DestinationPath "release/$name.zip" -CompressionLevel Optimal
Write-Host "MMQ_PACKAGE_OK mode=$Mode cuda_compile_commands=$($cudaCommands.Count) runtime=$runtimeVersion"
