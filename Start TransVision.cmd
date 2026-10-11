@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run Setup TransVision.cmd first to install the local environment.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m src.app --app-window %*
if errorlevel 1 pause
