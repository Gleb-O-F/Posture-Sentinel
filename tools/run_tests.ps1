param(
    [string]$VenvDir = '.venv',
    [switch]$VerboseTests
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot "$VenvDir\Scripts\python.exe"

function Find-PythonPath {
    if (Test-Path $venvPython) {
        return $venvPython
    }

    try {
        $null = & python --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            return 'python'
        }
    }
    catch {
    }

    try {
        $null = & py -3 --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            return 'py -3'
        }
    }
    catch {
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
$verbosity = if ($VerboseTests) { '2' } else { '1' }

Write-Host 'Running environment diagnostics'
Invoke-Python -PythonRef $pythonRef -Arguments @('tools/check_env.py', '--print')
if ($LASTEXITCODE -ne 0) {
    throw 'Environment diagnostics failed.'
}

Write-Host 'Running unit tests'
Invoke-Python -PythonRef $pythonRef -Arguments @('-m', 'unittest', 'discover', '-s', 'tests', '-v')
if ($LASTEXITCODE -ne 0) {
    throw 'Unit tests failed.'
}

Write-Host 'Running CLI smoke checks'
Invoke-Python -PythonRef $pythonRef -Arguments @('tools/report.py', '--date', '2026-03-19')
if ($LASTEXITCODE -ne 1) {
    throw 'Expected tools/report.py to return 1 when logs are absent.'
}

Invoke-Python -PythonRef $pythonRef -Arguments @('tools/perf_report.py', '--date', '2026-03-19')
if ($LASTEXITCODE -ne 1) {
    throw 'Expected tools/perf_report.py to return 1 when perf logs are absent.'
}

Invoke-Python -PythonRef $pythonRef -Arguments @('tools/tune.py', '--dry-run', '--print')
if (($LASTEXITCODE -ne 0) -and ($LASTEXITCODE -ne 1)) {
    throw 'Unexpected tools/tune.py exit code.'
}

Write-Host 'Automated test pass complete.'
