@echo off
REM Starts the CCF browser (double-click this file in Windows Explorer).
REM Finds Python (installing Python 3.13 first if there is none): see ensure_python.bat.
setlocal
cd /d "%~dp0"
title CCF browser
echo ==== CCF browser ====
echo.

call "%~dp0ensure_python.bat"

if not defined PYTHON_CMD (
  echo Python could not be found or installed automatically.
  echo Please install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^)
  echo and run this file again.
  echo.
  pause
  exit /b 1
)

"%PYTHON_CMD%" CCF_browser.py %*
if errorlevel 1 (
  echo.
  echo The CCF browser stopped with an error ^(see above^). If a library is missing, run python-install.bat.
  pause
)
