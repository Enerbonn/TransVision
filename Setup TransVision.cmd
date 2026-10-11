@echo off
setlocal
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --upgrade "pip>=26.2"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r setup/requirements.txt
if errorlevel 1 goto failed
echo.
echo Setup complete. Open Start TransVision.cmd.
pause
exit /b 0
:failed
echo Setup failed. Install Python 3.12 or newer and check your internet connection.
pause
exit /b 1
