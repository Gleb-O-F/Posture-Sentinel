param(
    [string]$VenvDir = '.venv',
    [switch]$UpgradePip,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot $VenvDir
$requirementsPath = Join-Path $repoRoot 'requirements.txt'

function Find-PythonCommand {
    $candidates = @(
        @('python'),
        @('py', '-3'),
        @('python3')
    )

    foreach ($candidate in $candidates) {
        $command = $candidate[0]
        $args = @()
        if ($candidate.Length -gt 1) {
            $args = $candidate[1..($candidate.Length - 1)]
        }

        try {
            $null = & $command @args --version 2>$null
            if ($LASTEXITCODE -eq 0) {
                return @{ Command = $command; Args = $args }
            }
        }
        catch {
        }
    }

    throw 'No working Python interpreter found. Install Python 3.10+ and retry.'
}

$python = Find-PythonCommand
Write-Host "Using Python launcher: $($python.Command) $($python.Args -join ' ')"

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment at $venvPath"
    & $python.Command @($python.Args + @('-m', 'venv', $venvPath))
}
else {
    Write-Host "Virtual environment already exists: $venvPath"
}

$venvPython = Join-Path $venvPath 'Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment python not found: $venvPython"
}

if ($UpgradePip) {
    Write-Host 'Upgrading pip'
    & $venvPython -m pip install --upgrade pip
}

if (-not $SkipInstall) {
    Write-Host "Installing dependencies from $requirementsPath"
    & $venvPython -m pip install -r $requirementsPath
}

Write-Host 'Bootstrap complete.'
Write-Host "Virtual environment python: $venvPython"
