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

- **ABC atlas download:** `abc_atlas_cache`, in the folder *containing* this
  repository (e.g. `D:/repos/JhouLab/abc_atlas_cache` when this repository is
  `D:/repos/JhouLab/ABC_atlas_browser`). Downloaded on first use if missing.
- **Local cache:** `cache_local` inside this repository (git-ignored). Section
  images and processed files are regenerated as needed.
- **Output folder:** chosen in the startup panel.

Both locations are relative to the script itself, so the app can be launched
from any working directory.
