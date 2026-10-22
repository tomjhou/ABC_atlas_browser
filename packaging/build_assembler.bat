@echo off
REM Builds the packaged section assembler and aligner on this computer, as the release workflow does:
REM   packaging\dist\SectionAssembler\SectionAssembler.exe   (Python and every library inside)
REM   packaging\dist\SectionAssembler.zip                    (what is handed to users)
REM It uses the same virtual environment as the RUN_*.bat files (the .venv folder: see ensure_python.bat), in which the program's own libraries
REM are what the program installed there; only PyInstaller is added to it.
setlocal
title Building the packaged section assembler
cd /d "%~dp0.."
echo ==== Building the packaged section assembler and aligner ====
echo.

call "%~dp0..\ensure_python.bat"
if not defined PYTHON_CMD (
  echo Python could not be found or installed. Install Python 3.13 from https://www.python.org/downloads/ and run this again.
  goto :fail
)

echo Adding PyInstaller to the environment ...
"%PYTHON_CMD%" -m pip install --quiet --only-binary :all: pyinstaller==6.22.3
if errorlevel 1 (
  echo PyInstaller could not be installed: see above.
  goto :fail
)

REM The program installs the libraries it is missing into the environment when it starts; running its self-test from the source makes it do that
REM now (and checks them), so that they are there to be packaged.
echo.
echo Checking the libraries the program needs ...
"%PYTHON_CMD%" packaging\entry.py --selftest
if errorlevel 1 (
  echo The libraries the program needs could not all be installed: see above.
  goto :fail
)

echo.
echo Building the package (a few minutes) ...
"%PYTHON_CMD%" -m PyInstaller packaging\assembler.spec --noconfirm --distpath packaging\dist --workpath packaging\build
if errorlevel 1 (
  echo The build failed: see above.
  goto :fail
)

echo.
echo Checking the package ...
if exist packaging\dist\SectionAssembler\selftest.txt del packaging\dist\SectionAssembler\selftest.txt
packaging\dist\SectionAssembler\SectionAssembler.exe --selftest
if not exist packaging\dist\SectionAssembler\selftest.txt (
  echo The package did not run its self-test.
  goto :fail
)
type packaging\dist\SectionAssembler\selftest.txt
findstr /c:"ALL FINE" packaging\dist\SectionAssembler\selftest.txt >nul
if errorlevel 1 (
  echo The package is missing something: see above.
  goto :fail
)
del packaging\dist\SectionAssembler\selftest.txt

powershell -NoProfile -Command "Compress-Archive -Path packaging\dist\SectionAssembler -DestinationPath packaging\dist\SectionAssembler.zip -Force"
if errorlevel 1 (
  echo The zip could not be made.
  goto :fail
)
echo.
echo Built packaging\dist\SectionAssembler.zip
echo (the program itself: packaging\dist\SectionAssembler\SectionAssembler.exe)
echo.
pause
exit /b 0

:fail
echo.
echo The build did not finish.
echo.
pause
exit /b 1
