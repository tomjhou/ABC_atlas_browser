# ABC browser

Interactive browser for the Allen Brain Cell (ABC) atlas MERFISH dataset
(C57BL6J-638850): pick brain sections and regions of interest, compute a UMAP
of the selected cells, and explore it alongside spatial section maps, colored by
taxonomy level or gene expression (including the imputed gene dataset).

## Section alignment program for Windows (no Python needed)

The section assembler and aligner (`FileConverters/`: find sections on Zeiss .czi slide scans, put them in order, and align NeuroInfo
serial-section stacks) is also available as a Windows program with everything inside it, so nothing has to be installed:

1. Download [SectionAssembler-windows.zip](https://github.com/tomjhou/Brain-anatomy-tools/releases/latest/download/SectionAssembler-windows.zip)
   (always the newest version; older ones and their notes are on the [Releases](https://github.com/tomjhou/Brain-anatomy-tools/releases) page).
2. Unzip it anywhere and run `SectionAssembler\SectionAssembler.exe`.

Windows may warn "Windows protected your PC" the first time, because the program is not code-signed: choose More info, then Run anyway.
To update, download the zip again. `SHA256SUMS.txt` on the release page lets you check the download. How it is built: `packaging/README.md`.

## Setup

1. Install Python.
2. Launch the app (see Running) from the Python environment you want to use
   (e.g. after `conda activate <env>` if you use conda). At startup it checks
   for missing libraries and pip-installs them automatically into that
   environment. `abc_atlas_access` installs from GitHub, so it needs git on
   your PATH. If anything fails to install, the app lists the commands to run
   by hand. To install everything up front instead, run `python-install.bat`
   from the same environment, or set `AUTO_INSTALL_LIBRARIES = False` in
   `ABC_atlas_browser.py` to turn automatic installation off.
   The `RUN_*.bat` files do this inside a private virtual environment, the
   `.venv` folder next to them (shared by all of them, made on first run; delete
   it to start afresh), so the computer's own Python is left alone.

## Running

    python ABC_atlas_browser.py

## Data locations

- **ABC atlas download:** by default `abc_atlas_cache`, in the folder *containing*
  this repository (e.g. `D:/repos/JhouLab/abc_atlas_cache` when this repository
  is `D:/repos/JhouLab/ABC_atlas_browser`). It can reach about 100 GB. If it isn't
  there at startup, the app lists your local drives and their free space, and
  asks once where to put it (or where an existing download is), suggesting a
  drive with enough room. The choice is saved in
  `cache_local/abc_atlas_cache_location.json`; delete that file to be asked again.
  Atlas files are downloaded on first use.
- **Local cache:** `cache_local` inside this repository (git-ignored). Section
  images and processed files are regenerated as needed.
- **Output folder:** `ABC_atlas_browser_output`, next to this repository, on first
  launch. It can be changed in the startup panel, and the most recently used
  folder is remembered.

These locations are relative to the script itself, so the app can be launched
from any working directory.
