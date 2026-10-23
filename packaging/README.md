# Packaging: the section assembler and aligner as a Windows program

`SectionAssembler.exe` is `FileConverters/SerialAssembler.py` (the start window, steps 1-4 and the aligner) with Python and every library inside,
so a user needs nothing installed: they unzip `SectionAssembler-<version>-windows.zip` and run `SectionAssembler\SectionAssembler.exe`
(or give it a `.jpx` or project file to open it straight in the aligner).

## Publishing a version

    git tag v1.2.0
    git push origin v1.2.0

`.github/workflows/release-assembler.yml` then builds the program on a Windows runner, checks that every library the program needs is inside
(`SectionAssembler.exe --selftest`), zips it and attaches the zip to a GitHub release for that tag. The same workflow can be run by hand
(Actions > Release section assembler > Run workflow) to get the zip as a build artifact without making a release.

Each release carries three files: `SectionAssembler-<version>-windows.zip`, the same zip under the fixed name `SectionAssembler-windows.zip`, and
`SHA256SUMS.txt`. The fixed name is what makes this link always give the newest version (the root README uses it):

    https://github.com/tomjhou/Brain-anatomy-tools/releases/latest/download/SectionAssembler-windows.zip

(A release marked "pre-release" is not what `latest` points to: mark test builds that way. The repository must be public for others to download.)
A user updates by downloading the zip again.

## Building it on this computer

`packaging\build_assembler.bat` does what the workflow does, in the shared `.venv` of the `RUN_*.bat` files (see `ensure_python.bat`);
the result is `packaging\dist\SectionAssembler.zip`.

## What is where

- `entry.py`: what PyInstaller starts. Without a console the program's output goes to `%LOCALAPPDATA%\JhouLab\section_assembler.log`;
  `--selftest` checks the package.
- `assembler.spec`: the PyInstaller recipe. imagecodecs, rasterio, czifile and h5py are taken whole: they load compiled parts and data by names
  that PyInstaller cannot see from the imports (the first build, without that, failed its own self-test).
- `requirements-assembler.txt`: the libraries (pinned) the workflow builds with. When the program's libraries are updated for the source,
  update the pins here and let the workflow's self-test confirm the package.
- `FileConverters/app_version.py`: `"dev"` in the source; the workflow writes the tag's version into it.
- In the packaged program `neuroinfo_stack.ensure_packages` does not run pip (the libraries are inside; there is nothing to install into).

## Known limits

- The program is not code-signed, so Windows SmartScreen shows "Windows protected your PC" on first run (More info > Run anyway), and some
  antivirus programs may object to an unsigned program. Signing needs a certificate.
- The ABC atlas browser and the CCF browser are not part of this package (they need far larger libraries).

# Packaging: the ABC atlas browser as a Windows program

`ABCAtlasBrowser.exe` is `ABC_atlas_browser.py` (the launch screen, the section and ROI pickers, the UMAP viewer) with Python and every library
inside, so a user needs nothing installed: they unzip `ABCAtlasBrowser-<version>-windows.zip` and run `ABCAtlasBrowser\ABCAtlasBrowser.exe`.
**It keeps its console window**: the browser asks a few things there (for example the label of an ROI run).

## Publishing a version

    git tag atlas-v1.0.0
    git push origin atlas-v1.0.0

`.github/workflows/release-atlas-browser.yml` builds it on a Windows runner, checks that every library is inside (`ABCAtlasBrowser.exe --selftest`,
which includes a small neighbors / UMAP / Leiden run), zips it and attaches `ABCAtlasBrowser-<version>-windows.zip` and `SHA256SUMS-atlas.txt` to
a release for that tag. (The assembler's tags are `v1.2.0`; the two products do not trigger each other's workflow.) Because both products publish
releases in this repository, `.../releases/latest` points to whichever was published last: link to a specific release's page for the atlas browser.

## Building it on this computer

`packaging\build_atlas_browser.bat` does what the workflow does, in the shared `.venv` (see `ensure_python.bat`): it installs the pinned libraries
of `requirements-atlas.txt` (about a gigabyte; git is needed for the Allen Institute's access library), runs the self-test from the source,
builds, runs the self-test on the package and zips it into `packaging\dist\ABCAtlasBrowser.zip`.

## What is where

- `atlas_entry.py`: what PyInstaller starts; `--selftest` checks the package; it points numba's cache at `%LOCALAPPDATA%\JhouLab\ABCAtlasBrowser\numba_cache`
  (the UMAP's compiled parts: the first UMAP run on a computer compiles for about a minute) and holds the console open on a crash.
- `atlas_browser.spec`: the PyInstaller recipe (the libraries that load things by name are taken whole).
- `requirements-atlas.txt`: the pinned libraries.
- `atlas_version.py` (repository root): `"dev"` in the source; the workflow writes the tag's version into it.
- Where the packaged program keeps things, because its own folder is replaced by every new version: caches, settings and the Tk settings under
  `%LOCALAPPDATA%\JhouLab\ABCAtlasBrowser\program`; the default output folder is `Documents\ABC_atlas_browser_output`; the atlas download location
  is asked on the first run (the default, `%LOCALAPPDATA%\JhouLab\ABCAtlasBrowser\abc_atlas_cache`, is only a suggestion; the atlas is several gigabytes).
- The packaged program does not install libraries (`AUTO_INSTALL_LIBRARIES` and the start-up library check are skipped when it runs frozen).

## Known limits

- Not code-signed (SmartScreen: More info > Run anyway), as for the assembler.
- The package is large (several hundred megabytes), because scanpy, numba and scikit-learn come with it. The CCF browser is not part of it.
- The `abc_atlas_access` library (inside the package) is under the Allen Institute's license: BSD 2-clause plus a clause that prohibits redistribution
  and use for commercial purposes without their permission. Fine for sharing among research colleagues; check it before the zip goes anywhere else.
