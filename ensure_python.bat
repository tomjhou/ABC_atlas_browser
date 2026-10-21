@echo off
REM Finds Python 3.10 to 3.13 (64-bit) for the RUN_*.bat files, installing Python 3.13 first when there is none.
REM Usage:  call "path\to\ensure_python.bat"   (sets PYTHON_CMD; if it is not set afterwards, Python could not be found or installed)
REM It looks through the py launcher, then "python" / "python3", then for a per-user install that is not on PATH yet. The installation is for the current user only and needs no administrator rights.

set "PYTHON_CMD="

REM Which versions to use: the newest Python often has no prebuilt libraries (numpy, h5py, ...) for a while, and the py launcher
REM and PATH prefer the newest. So only 3.10 to 3.13 (64-bit) are used; with none of those, 3.13 is installed (next to any other).
REM 1. The py launcher, asking for each acceptable version by name (this stores python.exe's full path).
for %%V in (3.13 3.12 3.11 3.10) do (
  if not defined PYTHON_CMD (
    for /f "delims=" %%I in ('py -%%V -c "import sys;print(sys.executable*(sys.maxsize>2**32))" 2^>nul') do set "PYTHON_CMD=%%I"
  )
)
if defined PYTHON_CMD exit /b 0

REM 2. "python" / "python3" on PATH (ignoring Windows' "Microsoft Store" stand-ins, which fail), if it is an acceptable version.
for %%P in (python python3) do (
  if not defined PYTHON_CMD (
    %%P -c "import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) and sys.maxsize>2**32 else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=%%P"
  )
)
if defined PYTHON_CMD exit /b 0

REM 3. Installed earlier but not on PATH (or just installed by this script in an earlier run)?
for %%V in (313 312 311 310) do (
  if not defined PYTHON_CMD (
    if exist "%LocalAppData%\Programs\Python\Python%%V\python.exe" set "PYTHON_CMD=%LocalAppData%\Programs\Python\Python%%V\python.exe"
  )
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
