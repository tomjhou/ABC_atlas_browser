"""The start of the packaged ABC atlas browser (what PyInstaller runs; see atlas_browser.spec).

  ABCAtlasBrowser.exe                 the launch screen (the browser keeps its console window: it asks things there now and then)
  ABCAtlasBrowser.exe --selftest      checks that everything the program needs is inside the package, prints the result and exits
                                      (the release build runs this on what it has just built)
"""
import multiprocessing
import os
import sys
import traceback

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if not getattr(sys, "frozen", False):      # (run from the source, as build_atlas_browser.bat does for the self-test)
    sys.path.insert(0, ROOT)


def numba_cache():
    """numba (the UMAP's compiled parts) keeps what it has compiled in a cache folder; the program's own folder is not one it may write in."""
    if os.environ.get("NUMBA_CACHE_DIR"):
        return
    folder = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "JhouLab", "ABCAtlasBrowser", "numba_cache")
    try:
        os.makedirs(folder, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = folder
    except OSError:
        pass


def selftest():
    """Import and use each library the program needs, so that one left out of the package is found now and not when a user needs it.
    The result is printed, and also written to selftest.txt beside the program."""
    lines, problems = [], []

    def say(text):
        lines.append(text)
        print(text, flush=True)

    def check(name, fn):
        try:
            fn()
            say(f"ok       {name}")
        except Exception as e:
            problems.append(name)
            say(f"MISSING  {name}: {type(e).__name__}: {e}")

    import numpy as np

    def imports():
        import abc_core, abc_tk_ccf, abc_tk_grid, abc_tk_launcher, abc_tk_picker, abc_tk_umap, abc_tk_umap_export  # noqa: F401,E401

    def numpy_pandas():
        import pandas as pd
        assert pd.DataFrame({"a": [1, 2]}).a.sum() == 3

    def tables_and_figures():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure()
        fig.add_subplot(111).plot([0, 1])
        fig.canvas.draw()
        plt.close(fig)
        import matplotlib.backends.backend_tkagg  # noqa: F401

    def images():
        import cv2
        from PIL import Image, ImageTk  # noqa: F401
        assert cv2.resize(np.zeros((8, 8, 3), np.uint8), (4, 4)).shape == (4, 4, 3)

    def hdf5_and_anndata():
        import anndata
        import h5py  # noqa: F401
        import tempfile
        a = anndata.AnnData(np.random.default_rng(0).random((20, 5)).astype(np.float32))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.h5ad")
            a.write_h5ad(p)
            assert anndata.read_h5ad(p).shape == (20, 5)

    def scanpy_umap_leiden():
        """The chain a new run uses: neighbors, UMAP (numba compiles it: a minute the first time) and Leiden (igraph, leidenalg)."""
        import anndata
        import scanpy as sc
        a = anndata.AnnData(np.random.default_rng(0).random((300, 20)).astype(np.float32))
        sc.pp.pca(a, n_comps=10)
        sc.pp.neighbors(a, n_neighbors=10)
        sc.tl.umap(a)
        sc.tl.leiden(a, resolution=1.0, flavor="igraph", n_iterations=2, directed=False)
        assert a.obsm["X_umap"].shape == (300, 2) and "leiden" in a.obs

    def scipy_sklearn():
        from scipy import ndimage, sparse  # noqa: F401
        from scipy.spatial import cKDTree
        from sklearn.neighbors import NearestNeighbors  # noqa: F401
        cKDTree(np.random.default_rng(0).random((10, 2)))

    def surfaces():
        from skimage import measure
        vol = np.zeros((10, 10, 10), np.float32)
        vol[3:7, 3:7, 3:7] = 1
        verts, faces, _n, _v = measure.marching_cubes(vol, 0.5)
        assert len(verts) and len(faces)

    def parquet_and_plotly():
        import pyarrow  # noqa: F401
        import plotly.graph_objects as go
        go.Figure(go.Scatter3d(x=[0], y=[0], z=[0])).to_html()

    def atlas_access():
        from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache  # noqa: F401

    def tk_window():
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        try:
            from tkinterdnd2 import TkinterDnD
            TkinterDnD._require(root)
        finally:
            root.destroy()

    for name, fn in (("the program's own modules", imports), ("numpy, pandas", numpy_pandas), ("matplotlib", tables_and_figures),
                     ("OpenCV, Pillow", images), ("h5py, anndata", hdf5_and_anndata), ("scipy, scikit-learn", scipy_sklearn),
                     ("scikit-image (surfaces)", surfaces), ("pyarrow, plotly", parquet_and_plotly), ("the Allen atlas access library", atlas_access),
                     ("Tk, drag and drop", tk_window), ("scanpy: neighbors, UMAP, Leiden", scanpy_umap_leiden)):
        check(name, fn)
    from atlas_version import VERSION
    say(f"version {VERSION}; " + ("ALL FINE" if not problems else "PROBLEMS: " + ", ".join(problems)))
    if getattr(sys, "frozen", False):          # (from the source the result is only printed)
        try:
            with open(os.path.join(os.path.dirname(sys.executable), "selftest.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError:
            pass
    return 1 if problems else 0


def main():
    numba_cache()
    from atlas_version import VERSION
    print(f"==== ABC atlas browser {VERSION} ====", flush=True)
    try:
        import ABC_atlas_browser  # noqa: F401   (it runs when imported: the launch screen, then the browser, until it is closed)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        print("\nThe ABC atlas browser stopped with an error (above).")
        try:
            input("Press Enter to close this window. ")
        except EOFError:
            pass
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if "--selftest" in sys.argv[1:]:
        numba_cache()
        sys.exit(selftest())
    main()
