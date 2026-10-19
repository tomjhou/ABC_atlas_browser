@echo off
REM Starts the section workflow (step 1: find the sections on .czi slide scans; step 2: the JPX aligner for NeuroInfo section stacks) (double-click this file in Windows Explorer).
REM Finds Python (installing Python 3.13 first if there is none): see ensure_python.bat.
REM Missing libraries are installed automatically on first run.
setlocal
cd /d "%~dp0"
title Section alignment
echo ==== Section alignment: step 1 load/detect, step 2 align ====
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

"%PYTHON_CMD%" workflow.py %*
if errorlevel 1 (
  echo.
  echo The aligner stopped with an error ^(see above^).
  pause
)
