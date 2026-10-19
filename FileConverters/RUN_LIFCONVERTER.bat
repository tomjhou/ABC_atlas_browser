@echo off
REM Starts the Leica .lif converter (double-click this file in Windows Explorer).
REM Finds Python (installing Python 3.13 first if there is none): see ensure_python.bat.
REM Missing libraries (numpy, opencv-python, pillow) are installed automatically on first run.
setlocal
cd /d "%~dp0"
title Leica .lif converter
echo ==== Leica .lif converter ====
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

"%PYTHON_CMD%" -c "import numpy, cv2, PIL" >nul 2>&1
if errorlevel 1 (
  echo Installing the libraries the converter needs ^(numpy, opencv-python, pillow^)...
  echo.
  "%PYTHON_CMD%" -m pip install numpy opencv-python pillow
  if errorlevel 1 (
    echo.
    echo The libraries could not be installed ^(see above^).
    pause
    exit /b 1
  )
  echo.
)

"%PYTHON_CMD%" LifConverterGUI.py %*
if errorlevel 1 (
  echo.
  echo The converter stopped with an error ^(see above^).
  pause
)
