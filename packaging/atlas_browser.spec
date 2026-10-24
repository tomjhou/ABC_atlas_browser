# PyInstaller spec for the ABC atlas browser: one folder, ABCAtlasBrowser\ABCAtlasBrowser.exe, with Python and every library inside.
# Build it from the repository's root:   pyinstaller packaging/atlas_browser.spec --noconfirm --distpath packaging/dist --workpath packaging/build
# (packaging/build_atlas_browser.bat does that in the repository's virtual environment; the release workflow in .github/workflows does it on a tag.)
# The program keeps its console window (console=True): the browser asks things there now and then.
import os

from PyInstaller.utils.hooks import collect_all

root = os.path.abspath(os.path.join(SPECPATH, ".."))

# These libraries load compiled parts, data files or modules by names that PyInstaller cannot see from the imports (scanpy and umap import a
# great deal lazily, numba and llvmlite carry their own compiler, tkinterdnd2 has its tkdnd binaries, plotly its JavaScript, ...), so everything
# of them is taken.
datas, binaries, hiddenimports = [], [], [
    "abc_core", "abc_tk_ccf", "abc_tk_grid", "abc_tk_launcher", "abc_tk_picker", "abc_tk_umap", "abc_tk_umap_export", "atlas_meshes", "atlas_version",
    "matplotlib.backends.backend_tkagg", "PIL.ImageTk",
]
for package in ("scanpy", "anndata", "umap", "pynndescent", "numba", "llvmlite", "igraph", "leidenalg", "plotly", "skimage", "tkinterdnd2",
                "h5py", "pyarrow", "sklearn", "abc_atlas_access", "session_info2", "legacy_api_wrap", "fast_array_utils", "lazy_loader",
                "natsort", "boto3", "botocore"):
    try:
        package_datas, package_binaries, package_hidden = collect_all(package)
    except Exception as e:
        print(f"collect_all({package}) failed: {e}")
        continue
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

a = Analysis(
    [os.path.join(SPECPATH, "atlas_entry.py")],
    pathex=[root],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
    excludes=["IPython", "pytest", "notebook", "jupyter", "sphinx", "vtk", "tkinter.test", "test"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="ABCAtlasBrowser",
    console=True,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="ABCAtlasBrowser", upx=False)
