@echo off
REM Starts the ABC atlas browser (double-click this file in Windows Explorer).
REM Finds Python (installing Python 3.13 first if there is none): see ensure_python.bat.
setlocal
cd /d "%~dp0"
title ABC atlas browser
echo ==== ABC atlas browser ====
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

"%PYTHON_CMD%" ABC_atlas_browser.py %*
if errorlevel 1 (
  echo.
  echo the ABC atlas browser stopped with an error ^(see above^). If a library is missing, run python-install.bat.
  pause
)
