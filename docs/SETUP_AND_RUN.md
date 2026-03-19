# Setup And Run

## Bootstrap

Create a virtual environment and install dependencies:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\bootstrap.ps1
```

Optional flags:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\bootstrap.ps1 -UpgradePip
powershell -ExecutionPolicy Bypass -File .\tools\bootstrap.ps1 -SkipInstall
```

## Run

Run the app with an environment check first:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run.ps1
```

Headless mode:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run.ps1 -Headless
```

Headless without tray:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run.ps1 -Headless -NoTray
```

Probe camera during the environment check:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run.ps1 -CheckCamera -CameraId 0
```

## Diagnostics

Environment report:

```powershell
python .\tools\check_env.py --print
```

Performance summary:

```powershell
python .\tools\perf_report.py --date 2026-03-19 --print
```
