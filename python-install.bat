@echo off
REM Installs the Python libraries used by ABC_atlas_browser.py.
REM Run it from the same Python environment you launch the app with
REM (e.g. after "conda activate <env>" if you use conda).

REM Required
python -m pip install numpy
python -m pip install pandas
python -m pip install matplotlib
python -m pip install pillow
python -m pip install anndata
python -m pip install scanpy
python -m pip install "abc_atlas_access[notebooks] @ git+https://github.com/alleninstitute/abc_atlas_access.git"

REM Optional: Leiden clustering of new UMAP runs
python -m pip install igraph

REM Parquet file support
python -m pip install pyarrow

pause
