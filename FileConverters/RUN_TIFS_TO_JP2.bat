@echo off
REM Starts the TIFF to NeuroInfo JP2 converter (double-click this file in Windows Explorer).
REM Finds Python (installing Python 3.13 first if there is none): see ensure_python.bat.
REM Missing libraries are installed automatically on first run.
setlocal
cd /d "%~dp0"
title TIFF to NeuroInfo JP2 converter
echo ==== TIFF to NeuroInfo JP2 converter ====
echo.

call "%~dp0..\ensure_python.bat"

if not defined PYTHON_CMD (
  echo Python could not be found or installed automatically.
  echo Please install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^)
  echo and run this file again.
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
