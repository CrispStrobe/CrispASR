param(
    [string]$Destination = ".hf-space-dist"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$sourceDir = Join-Path $repoRoot "hf-space"
$destDir = if ([System.IO.Path]::IsPathRooted($Destination)) { $Destination } else { Join-Path $repoRoot $Destination }

if (-not (Test-Path $sourceDir)) {
    throw "hf-space directory not found at $sourceDir"
}

if (Test-Path $destDir) {
    Remove-Item -LiteralPath $destDir -Recurse -Force
}

New-Item -ItemType Directory -Path $destDir | Out-Null

Copy-Item -LiteralPath (Join-Path $sourceDir "README.md") -Destination (Join-Path $destDir "README.md")
Copy-Item -LiteralPath (Join-Path $sourceDir "Dockerfile") -Destination (Join-Path $destDir "Dockerfile")
Copy-Item -LiteralPath (Join-Path $sourceDir "requirements.txt") -Destination (Join-Path $destDir "requirements.txt")
Copy-Item -LiteralPath (Join-Path $sourceDir "app.py") -Destination (Join-Path $destDir "app.py")
Copy-Item -LiteralPath (Join-Path $sourceDir "start.sh") -Destination (Join-Path $destDir "start.sh")

$dockerfilePath = Join-Path $destDir "Dockerfile"
$dockerfile = Get-Content -LiteralPath $dockerfilePath -Raw
$dockerfile = $dockerfile -replace 'COPY \. \.', 'COPY . /src/CrispASR'
$dockerfile = $dockerfile -replace 'COPY hf-space/requirements\.txt /space/requirements\.txt', 'COPY requirements.txt /space/requirements.txt'
$dockerfile = $dockerfile -replace 'COPY hf-space/app\.py hf-space/start\.sh /space/', 'COPY app.py start.sh /space/'
Set-Content -LiteralPath $dockerfilePath -Value $dockerfile -NoNewline

Write-Host "Exported Hugging Face Space files to $destDir"
