# Brain Anatomy Tools

This repository has several tools for analyzing brain anatomy data.

Although written in Python, you do NOT need to install anything yourself. Just run the .exe or .bat files according to the
instructions below, and the computer sets up what it needs on its own. The .exe has Python and every library inside it. Currently
only the assembler program has a .exe release; the other tools need to be run from .bat files (but I'm in the process of adding
.exe files). The first time you run a .bat file, it finds Python (installing Python 3.13 if there is none), builds a private `.venv` folder next
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
To update, download the zip again. `SHA256SUMS.txt` on the release page lets you check the download. How it is built: `packaging/README.md`.


## 2. Allen ABC atlas browser

Interactive browser for the Allen Brain Cell (ABC) atlas MERFISH dataset
(C57BL6J-638850): pick brain sections and regions of interest, compute a UMAP
of the selected cells, and explore it alongside spatial section maps, colored by
taxonomy level or gene expression (including the imputed gene dataset).

### Running

On Windows, double click RUN_ABC_ATLAS_BROWSER.bat. The first time you run it, it installs everything it needs into the
shared `.venv` (see above). For the section assembler that takes about 30 seconds; the ABC browser's libraries (scanpy,
anndata and others) are much larger, so expect it to take longer, and the `.venv` grows to about 1.1 GB with them. This only
happens once, and it does not alter any Python environment you might already have. The ABC browser also needs `git`
installed, to fetch the Allen Institute's `abc_atlas_access` library; if git is missing, the program says so.

On other systems, or to use your own Python environment, install Python and run `python ABC_atlas_browser.py` from that
environment (e.g. after `conda activate <env>`): at startup it checks for missing libraries and pip-installs them into it.


### Data locations

- **ABC atlas download:** by default `abc_atlas_cache`, in the folder *containing*
  this repository (e.g. `D:/repos/JhouLab/abc_atlas_cache` when this repository
  is `D:/repos/JhouLab/Brain-anatomy-tools`). It can reach about 100 GB. If it isn't
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


## 3. CCF browser

Scroll through the Allen CCF annotation volume (coronal, sagittal or horizontal slices) and plot cells from CSV files on it:
the coordinates exported by the ABC browser's "Export coordinates" button, or a marker export from NeuroInfo. On Windows, double
click RUN_CCF_BROWSER.bat (it uses the same `.venv`). The 25 um annotation volume (3.6 MB) comes from the ABC atlas cache, and is
downloaded on first use, after asking, if it is not there yet. Making surface meshes of regions needs the `vtk` library, which the
program offers to install the first time (a download of about 100 MB).
