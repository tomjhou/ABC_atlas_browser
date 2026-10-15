@echo off
REM Starts the TIFF to NeuroInfo JP2 converter (double-click this file in Windows Explorer).
REM Looks for Python as "py", then "python", then "python3", and uses the first that works.
REM Missing libraries are installed automatically on first run.
setlocal
cd /d "%~dp0"

set "PYTHON_CMD="
for %%P in (py python python3) do (
  if not defined PYTHON_CMD (
    %%P --version >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=%%P"
  )
)

if not defined PYTHON_CMD (
  echo Could not find Python: tried "py", "python" and "python3".
  echo Install Python from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^).
  echo.
  pause
  exit /b 1
)

"%PYTHON_CMD%" tifs_to_jp2.py %*
if errorlevel 1 (
  echo.
  echo The converter stopped with an error ^(see above^).
  pause
)
