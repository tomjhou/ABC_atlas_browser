@echo off
REM Starts the ABC atlas browser (double-click this file in Windows Explorer).
REM Looks for Python as "py", then "python", then "python3", and uses the first that works.
setlocal
cd /d "%~dp0"
title ABC atlas browser
echo ==== ABC atlas browser ====
echo.

set "PYTHON_CMD="
for %%P in (py python python3) do (
  if not defined PYTHON_CMD (
    %%P --version >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=%%P"
  )
)

if not defined PYTHON_CMD (
  echo Could not find Python: tried "py", "python" and "python3".
  echo Install Python from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^),
  echo then run python-install.bat to install the libraries this program uses.
  echo.
  pause
  exit /b 1
)

"%PYTHON_CMD%" ABC_atlas_browser.py %*
if errorlevel 1 (
  echo.
  echo the ABC atlas browser stopped with an error ^(see above^). If a library is missing, run python-install.bat.
  pause
)
