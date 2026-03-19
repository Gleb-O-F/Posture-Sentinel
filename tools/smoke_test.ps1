param(
    [string]$VenvDir = '.venv',
    [int]$CameraId = 0,
    [switch]$Headless = $true,
    [switch]$NoTray = $true
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

Write-Host 'Step 1: environment diagnostics'
powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'run_tests.ps1') -VenvDir $VenvDir

Write-Host 'Step 2: launch smoke run'
$runArgs = @('-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'run.ps1'), '-VenvDir', $VenvDir, '-CameraId', $CameraId)
if ($Headless) {
    $runArgs += '-Headless'
}
if ($NoTray) {
    $runArgs += '-NoTray'
}
$runArgs += '-CheckCamera'

powershell @runArgs
exit $LASTEXITCODE
