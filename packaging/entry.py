"""The start of the packaged section assembler and aligner (what PyInstaller runs; see assembler.spec).

  SectionAssembler.exe                 the start window
  SectionAssembler.exe file.jpx        straight to the alignment tool with that file (or a project saved by step 1)
  SectionAssembler.exe --selftest      checks that everything the program needs is inside the package, prints the result and exits
                                       (the release build runs this on what it has just built)
"""
import multiprocessing
import os
import sys
import time

if not getattr(sys, "frozen", False):      # (run from the source, as build_assembler.bat does for the self-test: the program's folder is not this one)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FileConverters"))


def log_to_file():
    """A packaged program with no console has no stdout or stderr: what the program prints goes to a log file instead (the newest
    one kept beside the older one), as it would to the console."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    folder = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "JhouLab")
    try:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "section_assembler.log")
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
            os.replace(path, path + ".old")
        stream = open(path, "a", buffering=1, encoding="utf-8")
        stream.write(f"\n===== started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    except OSError:
        stream = open(os.devnull, "w")
    sys.stdout = sys.stdout or stream
    sys.stderr = sys.stderr or stream


def selftest():
    """Import and use each library the program needs, so that one left out of the package is found now and not when a user needs it.
    The result is printed, and also written to selftest.txt beside the program (a program with no console has nowhere to print it)."""
    lines = []
    problems = []

    def say(text):
        lines.append(text)
        if sys.stdout is not None:
            print(text, flush=True)

    def check(name, fn):
        try:
            fn()
            say(f"ok       {name}")
        except Exception as e:
            problems.append(name)
            say(f"MISSING  {name}: {type(e).__name__}: {e}")

    try:
        import neuroinfo_stack as ns      # (from the source this installs numpy, imagecodecs and OpenCV if they are missing, as when the program starts)
        ns.ensure_packages({"PIL": "pillow"}, exit_on_failure=False)
        import numpy as np
    except BaseException as e:
        say(f"MISSING  the program's own modules and numpy: {type(e).__name__}: {e}")
        say("version ?; PROBLEMS: the program could not start")
        return 1

    def jpeg2000():
        import imagecodecs
        a = (np.arange(64 * 64 * 3).reshape(64, 64, 3) % 65535).astype(np.uint16)
        b = imagecodecs.jpeg2k_decode(imagecodecs.jpeg2k_encode(a, 60.0, codecformat="j2k", mct=False, resolutions=3))
        assert b.shape == a.shape

    def opencv():
        import cv2
        assert cv2.warpAffine(np.zeros((8, 8), np.uint8), np.eye(2, 3), (8, 8)).shape == (8, 8)

    def pillow():
        from PIL import Image, ImageTk  # noqa: F401
        Image.fromarray(np.zeros((4, 4), np.uint8))

    def hdf5():
        import neuroinfo_stack as ns
        ns.check_h5py()                      # (from the source, installs it if it is missing)
        import h5py
        import tempfile
        with tempfile.TemporaryDirectory() as d, h5py.File(os.path.join(d, "t.h5"), "w") as f:
            f.create_dataset("x", data=[1, 2, 3])

    def czi():
        import neuroinfo_stack as ns
        ns.ensure_packages(ns.CZI_PACKAGES, exit_on_failure=False)     # (from the source, installs it if it is missing)
        import czifile  # noqa: F401

    def rasterio_window():
        import neuroinfo_stack as ns
        assert ns.rasterio_modules() is not None       # (the program's own way of getting them: from the source it installs rasterio first)
        import rasterio  # noqa: F401
        from rasterio.io import MemoryFile  # noqa: F401
        from rasterio.windows import Window  # noqa: F401

    def tk_window():
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.destroy()

    def program():
        import neuroinfo_stack, section_loader, section_project, jpx_aligner, SerialAssembler, neuroinfo_dialogs  # noqa: F401,E401

    for name, fn in (("the program's own modules", program), ("JPEG 2000 (imagecodecs)", jpeg2000), ("OpenCV", opencv), ("Pillow", pillow),
                     ("HDF5 (h5py)", hdf5), ("Zeiss .czi reader (czifile)", czi), ("rasterio / GDAL windows", rasterio_window), ("Tk", tk_window)):
        check(name, fn)
    from app_version import VERSION
    say(f"version {VERSION}; " + ("ALL FINE" if not problems else "PROBLEMS: " + ", ".join(problems)))
    if getattr(sys, "frozen", False):          # (from the source the result is only printed)
        try:
            with open(os.path.join(os.path.dirname(sys.executable), "selftest.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError:
            pass
    return 1 if problems else 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if "--selftest" in sys.argv[1:]:
        sys.exit(selftest())
    log_to_file()
    import SerialAssembler
    SerialAssembler.main()
