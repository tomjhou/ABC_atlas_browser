# Brain Anatomy Tools

This repository has several tools for analyzing brain anatomy data.

Although written in Python, you do NOT need to install anything yourself. Just run the .exe or .bat files according to the
instructions below, and the computer sets up what it needs on its own. The .exe has Python and every library inside it. Currently
the section assembler (section 1) and the ABC atlas browser (section 2) have .exe releases; the other tools need to be run from .bat
files. The first time you run a .bat file, it finds Python (installing Python 3.13 if there is none), builds a private `.venv` folder next
to it, and installs the libraries it needs into that; this needs an internet connection, and it leaves any Python you already
have untouched. All the .bat files share the one `.venv`.


## 1. Section alignment program for Windows

The section assembler and aligner (in `FileConverters/`) can read NeuroInfo reconstructions (.xml or .jpx) or raw image
files (Zeiss .czi, or jp2 images), detect sections, sort them in order, and adjust alignment and section outlines to
create serial-section stacks.

This program is available as a Windows .exe program with everything inside it, so nothing has to be installed:

1. Download [SectionAssembler-windows.zip](https://github.com/tomjhou/Brain-anatomy-tools/releases/latest/download/SectionAssembler-windows.zip)
   (always the newest version; older ones and their notes are on the [Releases](https://github.com/tomjhou/Brain-anatomy-tools/releases) page).
2. Unzip it anywhere and run `SectionAssembler\SectionAssembler.exe`.

Windows may warn "Windows protected your PC" the first time, because the program is not code-signed: choose More info, then Run anyway.
To update, download the zip again. How it is built: `packaging/README.md`.


## 2. Allen ABC atlas browser

Interactive browser for the Allen Brain Cell (ABC) atlas MERFISH dataset
(C57BL6J-638850): pick brain sections and regions of interest, compute a UMAP
of the selected cells, and explore it alongside spatial section maps, colored by
taxonomy level or gene expression (including the imputed gene dataset).

### Windows standalone .exe program (easiest way to run)

1. Download [ABCAtlasBrowser-1.0.0-windows.zip](https://github.com/tomjhou/Brain-anatomy-tools/releases/download/atlas-v1.0.0/ABCAtlasBrowser-1.0.0-windows.zip)
   (about 290 MB; the [release page](https://github.com/tomjhou/Brain-anatomy-tools/releases/tag/atlas-v1.0.0) has its notes;
   newer versions will be on the [Releases](https://github.com/tomjhou/Brain-anatomy-tools/releases) page, as `atlas-v...`).
2. Unzip it anywhere and run `ABCAtlasBrowser\ABCAtlasBrowser.exe`. A console window opens and stays open: the program prints its progress
   there and occasionally asks a yes/no question there (for example, to confirm a run with no sections selected), so keep it visible.

Windows may warn "Windows protected your PC" the first time, because the program is not code-signed: choose More info, then Run anyway.
The first time, it asks where to keep the atlas download and suggests a drive with enough free space.
Since the atlas cache can reach about 100 GB, make sure the drive you choose has room.


### Running from the source

If you plan to edit or debug code, you'll need to run from source. The easiest way on Windows is to
double click RUN_ABC_ATLAS_BROWSER.bat. The first time you run it, it creates and installs a
shared `.venv` (see above), which takes about 30 seconds for the section assembler, and longer for the
ABC browser. The `.venv` will be about 300MB with the assembler, and grow to about 1.1 GB with the ABC
browser. This only happens once, and it does not alter existing Python environments.

On other systems, or to use your own Python environment, install Python and run `python ABC_atlas_browser.py` from that
environment (e.g. after `conda activate <env>`): at startup it checks for missing libraries and pip-installs them into it.


### Data locations

- **ABC atlas download:** The packaged Windows ABC browser program keeps the
  ABC atlas cache in locations described under "Windows standalone .exe program" above.
  When run from source (using `.bat` files or `python ABC_atlas_browser.py`)
  for the first time, the app looks for a folder abc_atlas_cache in the PARENT
  folder of the repository. If not found, the app lists your local drives and their
  free space, and asks once where to put it (or where an existing download is),
  suggesting a drive with enough room. The choice is saved in
  `cache_local/abc_atlas_cache_location.json`; delete that file to be asked again.
  Atlas files are downloaded on first use.
- **Local cache:** `cache_local` inside this repository (git-ignored). Section
  images and processed files are regenerated as needed. The packaged Windows program has no repository
  folder of its own to write in (a new version replaces its files), so its `cache_local` is
  `%LOCALAPPDATA%\JhouLab\ABCAtlasBrowser\program\cache_local`, and the Settings file
  `viewer_settings.json` is in the `program` folder beside it.

These default locations are relative to the script itself, so the app can be launched
from any working directory. If you choose your own cache location, this may not apply.


## 3. CCF browser

Scroll through the Allen CCF annotation volume (coronal, sagittal or horizontal slices) and plot cells from CSV files on it:
the coordinates exported by the ABC browser's "Export coordinates" button, or a marker export from NeuroInfo. On Windows, double
click RUN_CCF_BROWSER.bat (it uses the same `.venv`). The 25 um annotation volume (3.6 MB) comes from the ABC atlas cache, and is
downloaded on first use, after asking, if it is not there yet. Making surface meshes of regions needs the `vtk` library, which the
program offers to install the first time (a download of about 100 MB).
