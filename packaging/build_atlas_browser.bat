@echo off
REM Builds the packaged ABC atlas browser on this computer, as the release workflow does:
REM   packaging\dist\ABCAtlasBrowser\ABCAtlasBrowser.exe   (Python and every library inside; it keeps its console window)
REM   packaging\dist\ABCAtlasBrowser.zip                   (what is handed to users)
REM It uses the repository's virtual environment (the .venv folder: see ensure_python.bat) and installs the pinned libraries of
REM packaging\requirements-atlas.txt into it (about a gigabyte the first time).
setlocal
title Building the packaged ABC atlas browser
cd /d "%~dp0.."
echo ==== Building the packaged ABC atlas browser ====
echo.

call "%~dp0..\ensure_python.bat"
if not defined PYTHON_CMD (
  echo Python could not be found or installed. Install Python 3.13 from https://www.python.org/downloads/ and run this again.
  goto :fail
)

REM Nothing may be running from the environment: Windows keeps the libraries of a running program locked, and pip could not replace them.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0venv_in_use.ps1" -VenvDir "%~dp0..\.venv"
if errorlevel 1 goto :fail

echo Installing the libraries (the first time takes a while) ...
"%PYTHON_CMD%" -m pip install --quiet --only-binary :all: -r packaging\requirements-atlas.txt
if errorlevel 1 (
  echo The libraries could not be installed: see above.
  goto :fail
)

echo.
echo Checking the libraries from the source (the UMAP part compiles for a minute the first time) ...
"%PYTHON_CMD%" packaging\atlas_entry.py --selftest
if errorlevel 1 (
  echo Something the program needs does not work: see above.
  goto :fail
)

echo.
echo Building the package (several minutes) ...
"%PYTHON_CMD%" -m PyInstaller packaging\atlas_browser.spec --noconfirm --distpath packaging\dist --workpath packaging\build
if errorlevel 1 (
  echo The build failed: see above.
  goto :fail
)

echo.
echo Checking the package ...
if exist packaging\dist\ABCAtlasBrowser\selftest.txt del packaging\dist\ABCAtlasBrowser\selftest.txt
packaging\dist\ABCAtlasBrowser\ABCAtlasBrowser.exe --selftest
if not exist packaging\dist\ABCAtlasBrowser\selftest.txt (
  echo The package did not run its self-test.
  goto :fail
)
findstr /c:"ALL FINE" packaging\dist\ABCAtlasBrowser\selftest.txt >nul
if errorlevel 1 (
  echo The package is missing something: see above.
  goto :fail
)
del packaging\dist\ABCAtlasBrowser\selftest.txt

powershell -NoProfile -Command "Compress-Archive -Path packaging\dist\ABCAtlasBrowser -DestinationPath packaging\dist\ABCAtlasBrowser.zip -Force"
if errorlevel 1 (
  echo The zip could not be made.
  goto :fail
)
echo.
echo Built packaging\dist\ABCAtlasBrowser.zip
echo (the program itself: packaging\dist\ABCAtlasBrowser\ABCAtlasBrowser.exe)
echo.
pause
exit /b 0

:fail
echo.
echo The build did not finish.
echo.
pause
exit /b 1
