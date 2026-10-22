@echo off
REM Installs the Python libraries used by ABC_atlas_browser.py and CCF_browser.py into the private virtual environment (the .venv folder)
REM that the RUN_*.bat files use; see ensure_python.bat, which finds Python and makes the environment first if needed.
setlocal
cd /d "%~dp0"
call "%~dp0ensure_python.bat"
if not defined PYTHON_CMD (
  echo Python could not be found or installed automatically.
  echo Please install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^)
  echo and run this file again.
  echo.
  pause
  exit /b 1
)
echo Installing into %PYTHON_CMD%
echo.

REM Required
"%PYTHON_CMD%" -m pip install numpy
"%PYTHON_CMD%" -m pip install pandas
"%PYTHON_CMD%" -m pip install matplotlib
"%PYTHON_CMD%" -m pip install pillow
"%PYTHON_CMD%" -m pip install anndata
"%PYTHON_CMD%" -m pip install scanpy
"%PYTHON_CMD%" -m pip install "abc_atlas_access[notebooks] @ git+https://github.com/alleninstitute/abc_atlas_access.git"

REM Optional: Leiden clustering of new UMAP runs
"%PYTHON_CMD%" -m pip install igraph

REM Parquet file support
"%PYTHON_CMD%" -m pip install pyarrow

REM Interactive 3D view in the browser
"%PYTHON_CMD%" -m pip install plotly

REM Drag-and-drop of CSV files into the 3D viewer and CCF browser windows
"%PYTHON_CMD%" -m pip install tkinterdnd2

REM CCF browser "Generate meshes" (region surfaces for MBF / NeuroInfo atlases)
"%PYTHON_CMD%" -m pip install vtk

echo.
echo Done.
pause
