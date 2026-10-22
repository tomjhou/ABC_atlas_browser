# PyInstaller spec for the section assembler and aligner: one folder, SectionAssembler\SectionAssembler.exe, with Python and every library inside.
# Build it from the repository's root:   pyinstaller packaging/assembler.spec --noconfirm --distpath packaging/dist --workpath packaging/build
# (packaging/build_assembler.bat does that in a virtual environment of its own; the release workflow in .github/workflows does it on a tag.)
import os

from PyInstaller.utils.hooks import collect_all

root = os.path.abspath(os.path.join(SPECPATH, ".."))
source = os.path.join(root, "FileConverters")

# These libraries load their compiled parts and data files by names that PyInstaller cannot see from the imports (imagecodecs imports its
# codecs lazily; rasterio needs GDAL's data folder and modules it names as strings), so everything of them is taken.
datas, binaries, hiddenimports = [], [], ["PIL.ImageTk", "psutil"]
for package in ("imagecodecs", "rasterio", "czifile", "h5py"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

a = Analysis(
    [os.path.join(SPECPATH, "entry.py")],
    pathex=[source],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
    excludes=["matplotlib", "scipy", "pandas", "sklearn", "numba", "IPython", "pytest", "setuptools", "pip", "tkinter.test", "test", "unittest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="SectionAssembler",
    console=False,          # (what the program prints goes to %LOCALAPPDATA%\JhouLab\section_assembler.log: see entry.py)
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="SectionAssembler", upx=False)
