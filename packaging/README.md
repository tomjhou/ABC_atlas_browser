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
