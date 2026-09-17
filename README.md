# ABC browser

Interactive browser for the Allen Brain Cell (ABC) atlas MERFISH dataset
(C57BL6J-638850): pick brain sections and regions of interest, compute a UMAP
of the selected cells, and explore it alongside spatial section maps, colored by
taxonomy level or gene expression (including the imputed gene dataset).

## Setup

1. Install Python.
2. Install the required libraries by running `python-install.bat` from the same
   Python environment you will launch the app with (e.g. after
   `conda activate <env>` if you use conda). The app checks for missing
   libraries at startup and lists the install commands if any are missing.

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
