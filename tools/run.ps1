param(
    [switch]$Headless,
    [switch]$NoTray,
    [switch]$CheckCamera,
    [int]$CameraId = 0,
    [string]$VenvDir = '.venv'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot "$VenvDir\Scripts\python.exe"

function Find-PythonPath {
    if (Test-Path $venvPython) {
        return $venvPython
    }

    $candidates = @('python', 'py', 'python3')
    foreach ($candidate in $candidates) {
        try {
            if ($candidate -eq 'py') {
                $null = & $candidate -3 --version 2>$null
                if ($LASTEXITCODE -eq 0) {
                    return "$candidate -3"
                }
            }
            else {
                $null = & $candidate --version 2>$null
                if ($LASTEXITCODE -eq 0) {
                    return $candidate
                }
            }
        }
        catch {
        }
    }

    throw 'No working Python interpreter found. Run tools/bootstrap.ps1 after installing Python 3.10+.'
}

function Invoke-Python {
    param(
        [string]$PythonRef,
        [string[]]$Arguments
    )

    if ($PythonRef -eq 'py -3') {
        & py -3 @Arguments
    }
    else {
        & $PythonRef @Arguments
    }
}

$pythonRef = Find-PythonPath
Write-Host "Using Python: $pythonRef"

$checkArgs = @('tools/check_env.py', '--camera-id', $CameraId.ToString())
if ($CheckCamera) {
    $checkArgs += '--check-camera'
}
$checkArgs += '--print'

Write-Host 'Running environment check'
Invoke-Python -PythonRef $pythonRef -Arguments $checkArgs
if ($LASTEXITCODE -ne 0) {
    throw 'Environment check failed. Fix the reported issues before running the app.'
}

$appArgs = @('main.py')
if ($Headless) {
    $appArgs += '--headless'
}
if ($NoTray) {
    $appArgs += '--no-tray'
}

Write-Host 'Starting Posture Sentinel'
Invoke-Python -PythonRef $pythonRef -Arguments $appArgs
exit $LASTEXITCODE
