@echo off
REM Finds Python for the RUN_*.bat files, installing Python 3.13 first when there is none.
REM Usage:  call "path\to\ensure_python.bat"   (sets PYTHON_CMD; if it is not set afterwards, Python could not be found or installed)
REM It looks for "py", "python" and "python3" (ignoring Windows' "Microsoft Store" stand-ins, which fail --version), then for a
REM per-user install that is not on PATH yet. The installation is for the current user only and needs no administrator rights.

set "PYTHON_CMD="
for %%P in (py python python3) do (
  if not defined PYTHON_CMD (
    %%P --version >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=%%P"
  )
)
if defined PYTHON_CMD exit /b 0

REM Installed earlier but not on PATH (or just installed by this script in an earlier run)?
for /d %%D in ("%LocalAppData%\Programs\Python\Python3*") do (
  if exist "%%D\python.exe" set "PYTHON_CMD=%%D\python.exe"
)
if defined PYTHON_CMD exit /b 0

echo Python is not installed on this computer, so it will be downloaded from python.org and installed now.
echo ^(This happens once; it needs an internet connection and takes a few minutes. Please wait.^)
echo.
if not exist "%~dp0install_python.ps1" (
  echo install_python.ps1 is missing from %~dp0
  exit /b 1
)
set "PYTHON_RESULT=%TEMP%\python_install_result.txt"
if exist "%PYTHON_RESULT%" del "%PYTHON_RESULT%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_python.ps1" -ResultFile "%PYTHON_RESULT%"
if exist "%PYTHON_RESULT%" set /p PYTHON_CMD=<"%PYTHON_RESULT%"
if exist "%PYTHON_RESULT%" del "%PYTHON_RESULT%"
if defined PYTHON_CMD (
  echo.
  echo Python installed.
  echo.
)
exit /b 0
