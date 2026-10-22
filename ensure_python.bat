@echo off
REM Finds Python 3.10 to 3.13 (64-bit) for the RUN_*.bat files, installing Python 3.13 first when there is none.
REM Usage:  call "path\to\ensure_python.bat"   (sets PYTHON_CMD; if it is not set afterwards, Python could not be found or installed)
REM It looks through the py launcher, then "python" / "python3", then for a per-user install that is not on PATH yet. The installation is for the current user only and needs no administrator rights.
REM PYTHON_CMD ends up as the python.exe of a virtual environment, the ".venv" folder next to this file, built from the Python found here. All the RUN_*.bat
REM files share it, and missing libraries are installed into it rather than into the computer's Python. Deleting the .venv folder starts it afresh.

set "PYTHON_CMD="

REM Which versions to use: the newest Python often has no prebuilt libraries (numpy, h5py, ...) for a while, and the py launcher
REM and PATH prefer the newest. So only 3.10 to 3.13 (64-bit) are used; with none of those, 3.13 is installed (next to any other).
REM 1. The py launcher, asking for each acceptable version by name (this stores python.exe's full path).
for %%V in (3.13 3.12 3.11 3.10) do (
  if not defined PYTHON_CMD (
    for /f "delims=" %%I in ('py -%%V -c "import sys;print(sys.executable*(sys.maxsize>2**32))" 2^>nul') do set "PYTHON_CMD=%%I"
  )
)
if defined PYTHON_CMD goto :venv

REM 2. "python" / "python3" on PATH (ignoring Windows' "Microsoft Store" stand-ins, which fail), if it is an acceptable version.
for %%P in (python python3) do (
  if not defined PYTHON_CMD (
    %%P -c "import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) and sys.maxsize>2**32 else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=%%P"
  )
)
if defined PYTHON_CMD goto :venv

REM 3. Installed earlier but not on PATH (or just installed by this script in an earlier run)?
for %%V in (313 312 311 310) do (
  if not defined PYTHON_CMD (
    if exist "%LocalAppData%\Programs\Python\Python%%V\python.exe" set "PYTHON_CMD=%LocalAppData%\Programs\Python\Python%%V\python.exe"
  )
)
if defined PYTHON_CMD goto :venv

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
if not defined PYTHON_CMD exit /b 1

:venv
REM The virtual environment, built from the Python found above. A broken one (its Python was uninstalled or replaced) is rebuilt.
set "VENV_DIR=%~dp0.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_CHECK=import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) and sys.maxsize>2**32 else 1)"
if exist "%VENV_PYTHON%" (
  "%VENV_PYTHON%" -c "%VENV_CHECK%" >nul 2>&1
  if errorlevel 1 (
    echo The virtual environment in %VENV_DIR% no longer works, so it will be rebuilt.
    echo.
    rmdir /s /q "%VENV_DIR%"
  )
)
if not exist "%VENV_PYTHON%" (
  echo Setting up a private Python environment in %VENV_DIR% ^(this happens once and takes a minute^)...
  "%PYTHON_CMD%" -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo.
    echo The virtual environment could not be created, so the Python of the computer will be used instead.
    echo.
    if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
    exit /b 0
  )
  "%VENV_PYTHON%" -m pip install --quiet --upgrade pip >nul 2>&1
)
set "PYTHON_CMD=%VENV_PYTHON%"
exit /b 0
