"""
Generate a UMAP embedding from the Allen Institute MERFISH (C57BL6J-638850) dataset.

Pipeline:
1. Load the raw MERFISH AnnData in backed mode (as in the user's snippet).
2. Prompt for a cell type subset (Neurons / NonNeurons / All) and a brain
   section subset. Section selection first tries a clickable GUI (a grid of
   per-section spatial thumbnails, with non-neurons drawn gray and neurons
   colored by class on top), falling back to a console prompt if no
   interactive display is available. In the GUI, double-clicking a thumbnail
   opens a single-section ROI picker (after a confirmation, since it takes a
   few seconds) to hand-draw one or more rectangular regions of interest;
   ROIs can be collected across multiple sections in one session and, if
   any are drawn, take priority over the whole-section selection.
3. Bring counts into memory (backed='r' data must be loaded before most scanpy ops).
4. Merge in cell metadata (class/subclass/cluster/section labels) from the ABC cache.
5. Filter to the selected cell type, section(s), and ROIs (if any).
6. Subsample if needed, then standard scanpy preprocessing: filter, normalize, log-transform.
7. PCA -> neighbors -> UMAP.
8. Save the UMAP coordinates and a plot to disk (plus, if ROIs were drawn, a
   CSV of their coordinates and a PNG map showing them on their sections).
9. Highlight subclass 268 on the full UMAP.

Every output file for a run is saved into its own subfolder of `out_folder`,
named 'umap_{run_suffix}' (the cell type/section/ROI selection encoded as a
filename-safe string), with short fixed names within it (umap_coords.csv,
class_plot.png, subclass268.png, and, if ROIs were drawn, roi_coords.csv and
roi_map.png; if the optional subclass/supertype plots are generated, also,
for each of 'subclass'/'supertype', a 'umap_by_{subclass,supertype}'
subfolder of UMAP group*.png files and a 'spatial_maps_{subclass,supertype}'
subfolder of spatial group*.png files) — so distinct runs don't clutter
`out_folder` with many similarly-prefixed files.
Every plot PNG is saved alongside a same-named .svg twin.

Adjust `abc_cache` setup at the top to match however you're already initializing it
(e.g. via `abc_atlas_access.abc_atlas_cache.AbcProjectCache`).
"""

import os
import re
import sys
import gc
import json
import math
import time
import socket
import hashlib
import shutil
import warnings
import platform
import threading
import subprocess
from pathlib import Path
import importlib.util


def _make_windows_dpi_aware():
    """On Windows, declare this process DPI-aware before any Tk window gets
    created. Without this, under Windows display scaling (125%/150%/...), Tk
    reports screen dimensions (winfo_screenheight() etc.) in a different
    pixel space than what actually ends up on screen, so sizing a window
    from those numbers (see compute_figsize_for_screen_height) produces a
    window visibly smaller than intended — the two spaces disagree by
    roughly the scaling factor. Must run before the first Tk root/figure is
    created, so this is called at import time, right here."""
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
    except Exception:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


# Before check_libraries_or_exit(), which can open a dialog: if any Tk window
# exists before this call, Tk keeps reporting the scaled screen size for the
# rest of the process (e.g. 3413x1440 instead of 5120x2160 at 150%), and every
# window opens that much too large.
_make_windows_dpi_aware()


# Checked before the imports below, at every launch. Import name -> pip
# package spec (they differ for some).
REQUIRED_LIBRARIES = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'PIL': 'pillow',
    'anndata': 'anndata',
    'scanpy': 'scanpy',
    'abc_atlas_access': '"abc_atlas_access[notebooks] @ git+https://github.com/alleninstitute/abc_atlas_access.git"',
}
# The app runs without these; only the named feature is unavailable.
OPTIONAL_LIBRARIES = {
    'igraph': ('igraph', 'Leiden clustering of new UMAP runs'),
}


def check_libraries_or_exit():
    """Look for every library this script needs, without importing them
    (importlib.util.find_spec only locates them, so this costs milliseconds).
    Missing required libraries: show the install commands and quit, since the
    imports below would fail anyway. Missing optional ones only: ask whether
    to continue. Commands are plain pip; we can't tell whether the user runs
    conda or a base Python, so the message says to run them in whichever
    environment launches this script, and names that interpreter. Everything
    is also printed to the console, where it can be copied."""
    def is_missing(name):
        try:
            return importlib.util.find_spec(name) is None
        except (ImportError, ValueError):
            return True

    missing_required = [name for name in REQUIRED_LIBRARIES if is_missing(name)]
    missing_optional = [name for name in OPTIONAL_LIBRARIES if is_missing(name)]
    if not missing_required and not missing_optional:
        return

    lines = []
    if missing_required:
        lines += ["These required Python libraries are not installed:", ""]
        lines += [f"    {name}" for name in missing_required]
        lines += [""]
    if missing_optional:
        lines += ["These optional Python libraries are not installed:", ""]
        lines += [f"    {name} (needed for {OPTIONAL_LIBRARIES[name][1]})" for name in missing_optional]
        lines += [""]
    lines += ["Install them with:", ""]
    lines += [f"    python -m pip install {REQUIRED_LIBRARIES[name]}" for name in missing_required]
    lines += [f"    python -m pip install {OPTIONAL_LIBRARIES[name][0]}" for name in missing_optional]
    lines += ["", "Run these in the same Python environment you use to launch this app",
              "(e.g. after 'conda activate <env>' if you use conda). This app is running from:",
              f"    {sys.executable}"]
    if missing_required:
        lines += ["", "The app will now quit."]
    else:
        lines += ["", "Continue without them?"]
    message = "\n".join(lines)
    print("\n" + message + "\n")

    title = "Missing Python libraries"
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        if missing_required:
            messagebox.showerror(title, message, parent=root)
            proceed = False
        else:
            proceed = messagebox.askyesno(title, message, parent=root)
        root.destroy()
    except Exception:
        # No GUI available: fall back to the console.
        if missing_required:
            proceed = False
        else:
            try:
                proceed = input("Continue? [y/N] ").strip().lower() in ('y', 'yes')
            except EOFError:
                proceed = False
    if not proceed:
        sys.exit(1)


check_libraries_or_exit()

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from PIL import Image
import matplotlib
# Every window in this app is built on Tk: its sizing (compute_figsize_for_
# screen_height, normalize_tk_scaling), blitting, cursors and dialogs all
# assume the TkAgg backend. Without this, matplotlib picks whichever backend
# it finds first, e.g. Qt when PyQt is installed, and Qt's own display
# scaling then makes every window open too large. Must come before pyplot.
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Button, TextBox
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_tools import Cursors
from matplotlib.transforms import Bbox
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, TextArea, VPacker

from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache



# Every interactive window in this file has its own hand-rolled pan/zoom/
# hover controls (right-mouse-drag panning, scroll-to-zoom, ROI dragging,
# hover-for-cell-info, ...) — matplotlib's default NavigationToolbar2Tk
# (home/pan/zoom/save icons + coordinate readout, packed as a separate Tk
# widget below the canvas) is both redundant and, once a window is sized
# large relative to its original layout (see maximize_figure_window), prone
# to visually crowding/overlapping the custom button row at the bottom of
# the figure — those buttons are positioned in the *figure's* own fractional
# coordinate space, which has no notion of the toolbar's separate,
# fixed-pixel-height Tk strip. Disabling it removes that whole class of
# layout conflict. Must be set before any figure is created.
matplotlib.rcParams['toolbar'] = 'None'


def _has_internet_connection(host='s3.amazonaws.com', port=443, timeout=2.0):
    """Best-effort, fast (few-second) connectivity probe — a plain TCP
    connect attempt, not an HTTP/S3 request. AbcProjectCache.from_cache_dir
    (below) picks an S3-backed cache over a local-only one any time the
    cache directory is merely *writable* — true for basically every normal
    setup, even when every dataset this session will ever touch is already
    fully downloaded — so it doesn't matter whether the data is cached: the
    S3 client still gets constructed and still calls out to list/verify
    the manifest. Without internet, that call doesn't fail fast: boto3's
    own default connect/read timeouts (60s each) plus its default retry
    count meant the app sat there for minutes, apparently frozen, before
    any window ever appeared — this whole check exists to detect that
    upfront, in seconds, and skip straight to the local-only path instead."""
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


# Resolved from this script's own folder, not the working directory, so the app
# finds the same data wherever it's launched from.
SCRIPT_DIR = Path(__file__).resolve().parent

# Local cache dir for anything expensive to regenerate but cheap to keep on
# disk (the section-picker grid image, processed .h5ad files, ...). Defined up
# here because the atlas location below is remembered in it.
CACHE_DIR = SCRIPT_DIR / 'cache_local'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Where the ABC atlas download lives. By default it's next to this repository,
# in the parent folder. If it isn't there, the user picks a location once (see
# choose_atlas_cache_location) and the choice is remembered in this file.
DEFAULT_ATLAS_CACHE_DIR = SCRIPT_DIR.parent / 'abc_atlas_cache'
ATLAS_CACHE_LOCATION_FILE = CACHE_DIR / 'abc_atlas_cache_location.json'
# Free space a drive should have to hold the atlas download. A typical cache for
# this app, including the imputed gene dataset, is around 75 GB.
ATLAS_CACHE_SPACE_NEEDED_GB = 100


def format_gb(n_bytes):
    return f"{n_bytes / 1e9:,.0f} GB"


def list_local_drives():
    """[(root, free_bytes, total_bytes)] for every local drive, sorted by root.
    On Windows, fixed and removable drives (network, CD and RAM drives are
    left out); elsewhere, just the filesystem root. Drives that can't be read,
    like an empty card reader, are skipped."""
    roots = []
    if sys.platform == 'win32':
        import ctypes
        import string
        DRIVE_REMOVABLE, DRIVE_FIXED = 2, 3
        bitmask = ctypes.windll.kernel32.GetLogicalDrives()
        for i, letter in enumerate(string.ascii_uppercase):
            if bitmask & (1 << i):
                root = f'{letter}:\\'
                if ctypes.windll.kernel32.GetDriveTypeW(root) in (DRIVE_FIXED, DRIVE_REMOVABLE):
                    roots.append(root)
    else:
        roots.append('/')
    drives = []
    for root in sorted(roots):
        try:
            usage = shutil.disk_usage(root)
        except OSError:
            continue
        drives.append((root, usage.free, usage.total))
    return drives


def free_bytes_at(path):
    """Free space on the drive holding `path` (which may not exist yet), or None."""
    path = Path(path)
    for candidate in [path, *path.parents]:
        if candidate.exists():
            try:
                return shutil.disk_usage(candidate).free
            except OSError:
                return None
    return None


def suggest_atlas_cache_dir(drives):
    """The default location if its drive has room, otherwise abc_atlas_cache at
    the root of the local drive with the most free space."""
    needed = ATLAS_CACHE_SPACE_NEEDED_GB * 1e9
    free_default = free_bytes_at(DEFAULT_ATLAS_CACHE_DIR)
    if free_default is not None and free_default >= needed:
        return DEFAULT_ATLAS_CACHE_DIR
    if drives:
        roomiest_root = max(drives, key=lambda d: d[1])[0]
        return Path(roomiest_root) / 'abc_atlas_cache'
    return DEFAULT_ATLAS_CACHE_DIR


def looks_like_atlas_cache(folder):
    folder = Path(folder)
    return (folder / '_downloaded_data.json').exists() or (folder / 'metadata').is_dir()


def normalize_chosen_atlas_dir(folder):
    """A folder picked with Browse is used as-is if it already holds an atlas
    download or is named abc_atlas_cache; otherwise abc_atlas_cache is created
    inside it, so picking e.g. E:\\ gives E:\\abc_atlas_cache."""
    folder = Path(folder)
    if folder.name.lower() == 'abc_atlas_cache' or looks_like_atlas_cache(folder):
        return folder
    return folder / 'abc_atlas_cache'


def prompt_atlas_cache_location(suggested, drives):
    """Small Tk dialog: an editable path pre-filled with `suggested`, a Browse
    button, and the free space on the chosen path's drive. Not tkinter's own
    askdirectory, which can't pre-select a folder that doesn't exist yet.
    Returns the chosen Path, or None if cancelled."""
    import tkinter as tk
    from tkinter import filedialog

    needed = ATLAS_CACHE_SPACE_NEEDED_GB * 1e9
    result = {'path': None}
    root = tk.Tk()
    root.title("Choose ABC atlas download location")
    root.attributes('-topmost', True)
    root.resizable(False, False)
    frame = tk.Frame(root, padx=16, pady=12)
    frame.pack(fill='both', expand=True)

    tk.Label(frame, justify='left', wraplength=620, text=(
        "No ABC atlas download was found at the default location:\n"
        f"    {DEFAULT_ATLAS_CACHE_DIR}\n\n"
        "Choose where it should go. If this is a new location, the atlas files are "
        f"downloaded there as needed, which can take about {ATLAS_CACHE_SPACE_NEEDED_GB} GB, "
        "so pick a drive with plenty of free space. If you already have a download "
        "elsewhere, choose that folder instead. This is asked only once. The local "
        "drives and their free space are also listed in the console."
    )).pack(anchor='w')

    path_var = tk.StringVar(value=str(suggested))
    row = tk.Frame(frame, pady=10)
    row.pack(fill='x')
    entry = tk.Entry(row, textvariable=path_var, width=60)
    entry.pack(side='left', fill='x', expand=True)

    def browse():
        current = Path(path_var.get().strip() or str(suggested))
        start = next((str(p) for p in [current, *current.parents] if p.exists()), None)
        chosen = filedialog.askdirectory(parent=root, initialdir=start,
                                         title="Choose a folder for the ABC atlas download")
        if chosen:
            path_var.set(str(normalize_chosen_atlas_dir(chosen)))

    tk.Button(row, text="Browse...", command=browse).pack(side='left', padx=(8, 0))

    space_label = tk.Label(frame, justify='left', anchor='w')
    space_label.pack(fill='x')

    def update_space(*_args):
        text = path_var.get().strip()
        free = free_bytes_at(text) if text else None
        if free is None:
            space_label.config(text="Free space: unknown (drive not found)", fg='firebrick')
        elif looks_like_atlas_cache(text):
            space_label.config(text=f"Existing atlas download found here. Free space: {format_gb(free)}", fg='darkgreen')
        elif free < needed:
            space_label.config(text=f"Free space: {format_gb(free)} (less than the recommended "
                                    f"{ATLAS_CACHE_SPACE_NEEDED_GB} GB)", fg='firebrick')
        else:
            space_label.config(text=f"Free space: {format_gb(free)}", fg='darkgreen')

    path_var.trace_add('write', update_space)
    update_space()

    buttons = tk.Frame(frame, pady=(8))
    buttons.pack(fill='x')

    def ok(_event=None):
        text = path_var.get().strip()
        if not text:
            return
        if free_bytes_at(text) is None:
            # Keep the dialog open rather than failing to create the folder.
            from tkinter import messagebox
            messagebox.showerror("Drive not found", f"Can't find the drive for:\n{text}", parent=root)
            return
        result['path'] = Path(text)
        root.destroy()

    def cancel(_event=None):
        root.destroy()

    tk.Button(buttons, text="Cancel", width=10, command=cancel).pack(side='right')
    tk.Button(buttons, text="OK", width=10, command=ok).pack(side='right', padx=(0, 8))
    root.bind('<Return>', ok)
    root.bind('<Escape>', cancel)
    root.protocol('WM_DELETE_WINDOW', cancel)
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 3
    root.geometry(f'+{max(x, 0)}+{max(y, 0)}')
    entry.focus_set()
    entry.icursor('end')
    root.mainloop()
    return result['path']


def choose_atlas_cache_location():
    """Where the ABC atlas download lives, in this order: the location saved by
    an earlier choice (if that folder still exists), the default location next
    to this repository (if it exists), or else ask the user once. Asking lists
    every local drive and its free space on the console, then opens
    prompt_atlas_cache_location pre-filled with suggest_atlas_cache_dir. The
    answer is saved to ATLAS_CACHE_LOCATION_FILE. Exits if cancelled."""
    try:
        saved = json.loads(ATLAS_CACHE_LOCATION_FILE.read_text(encoding='utf-8')).get('path')
    except (OSError, ValueError, AttributeError):
        saved = None
    if saved:
        if Path(saved).is_dir():
            return Path(saved)
        print(f"The saved ABC atlas location {saved} no longer exists.")
    if DEFAULT_ATLAS_CACHE_DIR.is_dir():
        return DEFAULT_ATLAS_CACHE_DIR

    drives = list_local_drives()
    print(f"No ABC atlas download found at {DEFAULT_ATLAS_CACHE_DIR}.")
    print(f"It can need about {ATLAS_CACHE_SPACE_NEEDED_GB} GB. Local drives:")
    for drive_root, free, total in drives:
        print(f"    {drive_root:6s} {format_gb(free):>10s} free of {format_gb(total)}")
    suggested = suggest_atlas_cache_dir(drives)
    print(f"Suggested location: {suggested}")

    try:
        chosen = prompt_atlas_cache_location(suggested, drives)
    except Exception as e:
        print(f"Could not show the location dialog ({e}).")
        try:
            typed = input(f"ABC atlas location [{suggested}]: ").strip()
        except EOFError:
            typed = ''
        chosen = Path(typed) if typed else suggested
    if chosen is None:
        print("No ABC atlas location chosen; exiting.")
        sys.exit(0)

    try:
        chosen.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Could not create {chosen} ({e}); exiting.")
        sys.exit(1)
    ATLAS_CACHE_LOCATION_FILE.write_text(json.dumps({'path': str(chosen)}, indent=2), encoding='utf-8')
    print(f"Using {chosen} for the ABC atlas download (saved to {ATLAS_CACHE_LOCATION_FILE}).")
    return chosen


download_base = choose_atlas_cache_location()
if _has_internet_connection():
    abc_cache = AbcProjectCache.from_cache_dir(download_base)
else:
    print("No internet connection detected — using the local ABC atlas cache only "
          f"(skipping the S3 manifest check that would otherwise stall startup for minutes).")
    # NotImplementedError (not a hang) if this session ends up needing a
    # file that was never downloaded — see LocalCache's own _download_
    # file/_download_manifest, both of which just raise instead of trying
    # to reach the network — so a genuinely offline machine with no cache
    # at all still fails fast here, just with a real Python traceback
    # instead of a silent freeze.
    abc_cache = AbcProjectCache.from_local_cache(download_base)

# ---------------------------------------------------------------------------
# Cell type subset selection
# ---------------------------------------------------------------------------
# Non-neuron classes in the ABC atlas taxonomy are numbered 30-34 (e.g.
# "30 Astro-Epen", ..., "34 Immune"); everything else is a neuron class.
NON_NEURON_CLASS_IDS = {30, 31, 32, 33, 34}
CELL_TYPE_OPTIONS = {'1': 'Neurons', '2': 'NonNeurons', '3': 'All'}
CELL_TYPE_SUFFIXES = {'Neurons': 'neurons', 'NonNeurons': 'nonneurons', 'All': 'all'}

# Column identifying which physical tissue section (coronal slice) each cell
# came from, e.g. "C57BL6J-638850.36".
SECTION_COL = 'brain_section_label'

# obs column holding scanpy's own Leiden clustering (Step 6), computed from
# this run's neighbor graph. Deliberately *not* called 'cluster': that name
# is already taken by the Allen ABC taxonomy's finest annotation level (see
# metadata_cols), which is a published label, not something derived here.
LEIDEN_KEY = 'leiden'
# Passed straight through to sc.tl.leiden's own `resolution` argument —
# scanpy's own default (1.0) if left at 1.0 here. Roughly, cluster *count*
# scales with resolution: higher values favor more, smaller clusters; lower
# values favor fewer, larger ones. There's no formula for "resolution X
# gives Y clusters" (it depends on the actual neighbor graph), so tuning
# this is trial and error — rerun Step 6 (or offer_leiden_backfill for a
# cached run) after changing it and check adata.obs[LEIDEN_KEY].nunique().
# Typical exploratory range is roughly 0.3 (coarse) to 2.0 (fine); scanpy's
# docs suggest 0.4-1.2 for a "usual" range, but that's dataset-dependent.
LEIDEN_RESOLUTION = 1.2

# ===========================================================================
# TUNABLE CONSTANTS — INTERACTIVE UMAP VIEWER
# ===========================================================================
# Everything here belongs to the "Interactive UMAP viewer" window opened at
# Step 9 (show_interactive_umap_window) — the one with the brain-section
# panels on the left, the UMAP scatter in the middle, and the sidebar
# controls on the right. Edit freely; all are read live, so a change takes
# effect on the next launch with no other edits needed.
#
# NOT collected here, deliberately:
#   * The section/ROI picker windows' own constants, which live in
#     prompt_section_selection_gui / prompt_subregion_selection. Two names
#     appear in both places with *different* values, so the viewer's copies
#     are prefixed VIEWER_ below to keep them distinct.
#   * This window's layout fractions (GRID_LEFT, GAP, CBAR_WIDTH,
#     QUERY_TOP_Y, the STATUS_PANEL_* family, ...) — they only make sense
#     next to the layout code that consumes them.
#   * Sizes derived from the ones here (UMAP_HIGHLIGHT_BASE_SIZE and
#     friends) and anything derived from the screen-dependent font size;
#     both stay put, and pick up changes made here automatically.
#   * AREA_BOTTOM / GRID_RIGHT / UMAP_LEFT / UMAP_RIGHT / SIDEBAR_LEFT /
#     SIDEBAR_WIDTH / HANDLE_BOTTOM: despite the naming these are mutable
#     layout *state*, reassigned when the resize handles are dragged.

# --- UMAP scatter appearance ---
UMAP_POINT_SIZE = 10       # matplotlib `s` (an AREA) for each cell, at full zoom-out
UMAP_LABEL_FONTSIZE = 10   # centroid ID labels, at full zoom-out; grows with the dots
UMAP_GROUP_DIAMETER_RATIO = 1.25  # family-highlight '+' size vs the cell's own CURRENT diameter

# When True, categorical modes ('All'/'Single Subclass' etc.) use
# category_rank_shape's marker shapes on the *live* UMAP too, not just
# the saved PNG/SVG (which always uses shapes — see render_export_figure).
# Costs one scatter() call per (shape, filled/open) combination actually
# in use instead of one call total, but only at redraw time (a mode/level/
# query change) — the far more frequent zoom/pan ticks already redraw
# from a cached bitmap regardless of how many artists made up the last
# real one, so the expected performance cost of this is minor.
UMAP_USE_SHAPES_ON_SCREEN = True

# On-screen ID legend (draw_id_legend, in cbar_ax) for 'All <level>s' mode,
# at whichever level is currently selected (class/subclass/supertype/
# cluster/leiden). How many rows fit is computed dynamically from the
# window's own current height (see legend_max_rows_for_current_size,
# inside show_interactive_umap_window) rather than a fixed count here —
# a 4k monitor can comfortably show far more rows than a laptop screen.
# Rows beyond that count are dropped smallest-cell-count first (the
# ranking compute_ranked_category_colors already produces), not shrunk to
# fit — past a certain row count the text stops being legible regardless
# of how small it's squeezed; the export-only legend (build_export_
# legend_entries) is what covers the rest for a level with far more
# categories than could ever fit here (subclass/supertype/cluster
# routinely do) — see redraw_all_subclasses' own comment.

# Per-cell opacity. Below 1.0, overlapping cells accumulate into visibly
# darker regions, so a dense cluster reads as dense rather than as one flat
# patch of color — the trade is that an isolated cell is fainter. 1.0
# disables the effect. Applies to the cells only: the hover ring and the
# family-highlight '+' markers stay fully opaque, so they still read clearly
# against whatever is underneath them.
UMAP_POINT_ALPHA = 0.75

# Baseline (every gene at its low end) color for the multi-gene (2 or 3
# genes, red/green/blue) overlay — see multi_gene_rgb — shared by the UMAP
# and the section panels, so a cell's color means the same thing (0 =
# black, rising toward red/green/blue as expression increases) regardless
# of which plot it's read off. Both plots' own *axes* background is set
# separately, to MULTI_GENE_UMAP_FACECOLOR (a dark grey, not this pure
# black) — a zero-expression cell (this color) needs to stay visibly a
# cell, not melt into the space around it the way it would against an
# identically-black background. Applied in redraw_multi_genes directly for
# the UMAP's own `ax`, and via set_section_panel_facecolor for every
# section panel (whose normal, every-other-mode background is pure black —
# SECTION_PANEL_FACECOLOR — restored by clear_colorbar on leaving
# multi-gene mode).
MULTI_GENE_LOW_EXPRESSION_COLOR = (0.0, 0.0, 0.0)
MULTI_GENE_UMAP_FACECOLOR = (0.1, 0.1, 0.1)
# Genes 4-6 (UMAP only, so far): a second red/green/blue overlay, drawn as
# small filled circles centered on top of the circle layer above, instead of
# its own separate plot — see redraw_multi_genes. Opaque (alpha=1) and
# smaller than the base circle, so the base layer's own color still shows
# around its edges. Diameter (not area — matplotlib `s` is area, so this
# gets squared before use) multiplier on the circle layer's own current
# size.
MULTI_GENE_PLUS_SIZE_DIAMETER_MULTIPLIER = 0.5
# Cells with exactly zero expression across all of genes 4-6 simply don't
# get an overlay dot drawn at all (rather than an opaque black one) — see
# redraw_multi_genes' own visibility mask.
MULTI_GENE_PLUS_ZORDER = 1.5  # above the circle layer's default zorder (1), below hover highlights (5.2+)
# The UMAP axes' own background for every *other* mode (categorical
# taxonomy levels, single gene) — restored explicitly by clear_colorbar
# whenever leaving multi-gene mode. Needed because Axes.clear() does *not*
# reset facecolor on its own (confirmed directly — it's one of the few
# Artist properties clear() leaves alone), so without this, a dark
# MULTI_GENE_UMAP_FACECOLOR set while looking at two or three genes stayed
# on screen indefinitely after switching to 'All <level>s'/'Specified
# <level>(s)', where there's no gradient-of-brightness cell color to need
# a dark background for in the first place.
UMAP_AXES_FACECOLOR = 'white'

# How fast dots (and the centroid labels) grow as you zoom in.
#   0.0 = constant on-screen size at every zoom level
#   1.0 = grows in step with the zoom, so a cluster looks identical at any zoom
# Applied to the DIAMETER: at zoom ratio Z, diameter is 1 + RATE*(Z-1).
# matplotlib's `s` is an area, so the dots square this; the label font, being
# a linear dimension, takes it as-is — which is what keeps the two in
# constant proportion.
UMAP_ZOOM_DOT_GROWTH_RATE = 0.75

# How many of a level's categories — ranked by cell count, largest first —
# get a real, distinct color (and a centroid ID label) in 'All <level>s'
# mode. Below this cut, categories are gray with no label on the UMAP
# itself. class/subclass rarely exceed this; supertype/cluster/leiden
# routinely have hundreds to thousands of categories, and matplotlib has no
# way to make that many colors mutually distinguishable regardless of what
# palette generates them — this keeps the ones with the most cells (and
# therefore the most visual weight) identifiable, rather than making every
# category equally hard to tell apart.
#
# The section panels use the *same* ranking but don't gray out the excess —
# they cycle back through the same N colors instead (see
# set_section_colors_categorical), since a flat gray region there reads as
# "nothing here" rather than "many small categories here". A recycled color
# no longer uniquely identifies one category, but hovering any cell still
# reports its exact one regardless of color.
UMAP_MAX_COLORED_CATEGORIES = 400  # category_rank_color/category_rank_shape's own combined cycle length

# Experimental: when True, each centroid ID label's font size scales with
# the 1/4 POWER of that category's own cell count relative to the median
# among labeled categories (see cell_weighted_median_count), instead of
# every label sharing one flat UMAP_LABEL_FONTSIZE — a category at the
# median gets scale 1 (unchanged), one with 10x the median gets scale
# 10**0.25 ≈ 1.78, one with 100x gets ≈3.16 (clamped down to MAX_SCALE
# below), one with 1/10th gets ≈0.56, one with 1/100th gets ≈0.32 (clamped
# up to MIN_SCALE below). Gentler than a straight log10 (which would put
# 10x at scale 2 and 1/10th at scale 0 exactly) — cell counts across
# categories routinely span several orders of magnitude, and the 1/4
# power keeps the largest few categories' labels from dwarfing everything
# else while still keeping tiny categories legible without hitting the
# floor immediately.
# Clamped by the two _SCALE constants below so a very rare or very dominant
# category can't shrink to unreadable or balloon over its neighbors, and
# further adjusted by zoom level — see umap_label_fontsize, the only place
# that clamping actually happens; MIN/MAX_SCALE are just its inputs. Toggle
# UMAP_LABEL_SIZE_BY_CELL_COUNT off if the whole effect reads as more
# distracting than informative.
UMAP_LABEL_SIZE_BY_CELL_COUNT = True
UMAP_LABEL_SIZE_BY_CELL_COUNT_MIN_SCALE = 0.5
UMAP_LABEL_SIZE_BY_CELL_COUNT_MAX_SCALE = 2.5

# --- Brain-section panels (left) ---
SECTION_BACKGROUND_BASE_SIZE = 3     # `s` for each background cell, at full zoom-out
SECTION_POINT_ALPHA = 0.75           # as UMAP_POINT_ALPHA, for the panels' cells
SECTION_ZOOM_DOT_GROWTH_RATE = 0.5   # as UMAP_ZOOM_DOT_GROWTH_RATE, for the panels' dots
SECTION_UNKNOWN_COLOR = 'dimgray'    # cells with no value at the current level
# Panel axes background for every mode except multi-gene (which swaps it to
# MULTI_GENE_UMAP_FACECOLOR — see that constant's own comment; a genuinely
# black background there would make a zero-expression cell, itself black,
# disappear into it). set at panel-build time and restored by clear_
# colorbar whenever leaving multi-gene mode — see set_section_panel_facecolor.
SECTION_PANEL_FACECOLOR = 'black'
VIEWER_SPAN_PERCENTILE = 90          # percentile of section extents setting the shared panel scale
PANEL_PROGRESS_INTERVAL = 10         # log a progress line every N panels while building the grid

# --- Section-panel home-view disk cache (interactive viewer only) ---
# One rendered PNG per (run, section, level) — the "All <level>s" mode's
# categorical background coloring at each panel's fully-zoomed-out (home)
# extent, persisted into the run's own folder (see show_interactive_umap_
# window's own run_folder) so re-showing it — even in a brand new session,
# on a different machine sharing that folder — is instant instead of a
# real ~80-panel vector re-render (which is what was taking ~30s). Screen-
# size independent by construction: rendered off-screen at this fixed
# size/DPI (render_section_home_view_png), never tied to whatever window
# happened to generate it, then displayed via imshow — which resamples to
# fit whatever panel size actually needs it, the same way the live zoom-
# preview bitmaps already work — regardless of which machine's screen
# first produced the cached file.
SECTION_HOME_CACHE_DPI = 150
# Long edge of the cached bitmap, in inches at SECTION_HOME_CACHE_DPI — a
# fixed size regardless of the live window's own panel count/size (see the
# comment above), which means it has to be generous enough to still look
# sharp for the *largest* a panel ever gets: a run with few sections (each
# panel then gets a much bigger on-screen box than one of 60+ sections
# sharing the same grid area) rather than tuned for the common many-
# sections case. 3.0in (450px) was tuned for the latter and visibly
# pixelated once blown up to fill a large panel in a small-section-count
# run — bumped to 8.0in (1200px) to stay sharp across that whole range.
SECTION_HOME_CACHE_LONG_EDGE_IN = 8.0
# Bump this whenever render_section_home_view_png's own output would
# change (colors, dot size/style, point selection, ...) in a way that
# makes an already-cached PNG show something subtly wrong — there's no
# way to detect that automatically, so a stale cache would otherwise just
# keep being served as if still valid. Bumped 1 -> 2 alongside the
# LONG_EDGE_IN increase above, so existing low-resolution caches on disk
# are treated as a different (missing) cache key and regenerated at the
# new size automatically, rather than requiring every run folder's
# section_home_cache to be deleted by hand.
SECTION_HOME_CACHE_VERSION = 2

# --- Section-panel scale bar (first panel only; see build_section_scalebar) ---
SCALEBAR_TARGET_FRACTION = 0.22      # of the panel's current view width
SCALEBAR_MARGIN_FRACTION = 0.06      # inset from the panel's own edges, as a fraction of its view
SCALEBAR_TEXT_GAP_FRACTION = 0.02    # extra gap above the bar, for its length label
# A 1-2-5 sequence in micrometers, the ABC atlas's own native unit divided by
# 1000 (see HOVER_CELL_RADIUS_DATA_UNITS's comment on why the atlas's 'x'/'y'
# columns are actually millimeters). Covers a single MERFISH cell's own
# width (~10 um) up to several whole sections (~50 mm) side by side.
SCALEBAR_NICE_LENGTHS_UM = (
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000,
)
SECTION_MAPS_SAVE_DPI = 200           # matches save_roi_map/save_group_spatial_maps's own PNGs

# --- Interaction timing (milliseconds unless noted) ---
# How long a scroll burst must go quiet before the cheap stand-in bitmaps are
# swapped back for a real, full-quality redraw. Longer = fewer expensive
# redraws while scrolling, at the cost of the sharp view arriving later.
ZOOM_PREVIEW_SETTLE_MS_UMAP = 800
ZOOM_PREVIEW_SETTLE_MS_PANEL = 800
# Below this zoom ratio (home_span / current_span — 1.0 at fully zoomed
# out, 3.0 means the current view spans a third of the home extent), the
# UMAP's own zoom/pan settle (end_zoom_previews) skips its real, full-
# resolution fig.canvas.draw() entirely and just stays on the cheap
# zoom-preview bitmap already on screen — a magnified crop, capped at
# whatever resolution it was captured at, rather than a fresh re-render.
# At or above it, the settle still does a real draw, but first filters the
# UMAP scatter down to only the points inside the current view (see
# filter_main_scatter_to_viewport) — every real draw used to reprocess the
# *entire* ~200k-cell point set through matplotlib's transform/clip
# pipeline regardless of zoom level, which is almost certainly why zooming
# stayed slow even with very little actually visible on screen. Also
# governs the section panels' own zoom/pan settle the same way, against
# section_zoom_ratio (the one shared zoom level all ~80 panels zoom
# together at) instead — see filter_all_section_scatters_to_viewport.
ZOOM_BITMAP_ONLY_MAX_MULTIPLIER = 2.0
# Prints a "[zoom-filter] ..." line every time filter_main_scatter_to_
# viewport (or its section-panel equivalent) runs, showing the cell count
# before/after filtering — useful while tuning ZOOM_BITMAP_ONLY_MAX_
# MULTIPLIER itself, noisy otherwise (it fires on every settle past the
# threshold), so off by default.
ZOOM_DEBUG_DIAGNOSTICS = True  # TODO: set back to False once the section-cache speed/orientation issue is confirmed fixed
VIEWER_HOVER_HOLD_MS = 250       # cursor must settle this long before the hovered cell is looked up
GROUP_HOVER_HOLD_MS = 500        # ...and this long before its whole family is highlighted
LAYOUT_REDRAW_MIN_INTERVAL = 0.03  # SECONDS; caps redraws while dragging the resize handles

# --- Saving ("Save UMAP" button) ---
# Resolution of the saved PNG. The SVG is vector and ignores this — but note
# an SVG of a large UMAP stores every cell as its own element, so for a
# several-hundred-thousand-cell run that file is big and slow to open; the
# PNG is the practical one at those sizes.
UMAP_SAVE_DPI = 450

# --- Rendering detail ---
# How far inside its own boundary a zoom/pan preview snapshot is cropped, to
# keep the axes' antialiased spine out of the bitmap — without this the
# baked-in border draws as a hard rectangle over the layer beneath it.
SPINE_INSET_PX = 2

# --- Status-bar text ---
HOVER_DEFAULT_MESSAGE = 'Hover over any cell to see its class/subclass/supertype/cluster.'
# Between the fields of the hover status line. A semicolon (not a comma)
# because several of these values contain commas of their own.
HOVER_FIELD_SEP = ';   '
# ===========================================================================

# Prefixed onto per-section/per-gene cache filenames so they don't collide if
# another Allen ABC dataset is added alongside this one later.
DATASET_NAME = 'merfish_c57bl6j_638850'

# The section-picker grid always shows every section, neuron-only, regardless
# of which cell type / section subset is chosen for the actual UMAP run — so
# there's exactly one canonical cached image, reused across every run.
GRID_CACHE_PNG = CACHE_DIR / f'{DATASET_NAME}_section_picker_grid.png'
GRID_CACHE_LAYOUT = CACHE_DIR / f'{DATASET_NAME}_section_picker_grid_layout.json'
GRID_CACHE_DPI = 375  # 300 * 1.25

# Persisted, most-recent-first list of output folders the user has run
# against, surfaced as a dropdown on the startup panel (prompt_startup_panel)
# so switching between projects is one click rather than a re-typed path.
RECENT_OUTPUT_FOLDERS_PATH = CACHE_DIR / 'recent_output_folders.json'
MAX_RECENT_OUTPUT_FOLDERS = 10


def _normalized_folder_key(folder):
    """Case- and separator-normalized absolute form of `folder`, so
    'M:/proj', 'M:\\proj\\' and 'm:\\PROJ' all de-duplicate to one entry in
    the recent-folders list on Windows. Falls back to the raw string if the
    path can't be normalized (e.g. an unmounted drive)."""
    try:
        return os.path.normcase(os.path.abspath(os.path.expanduser(str(folder))))
    except Exception:
        return str(folder)


def load_recent_output_folders():
    """The recently-used output folders, most-recent-first, at most
    MAX_RECENT_OUTPUT_FOLDERS and de-duplicated. Empty list on first run or
    any read/parse failure — this list is a convenience, never load-bearing,
    so a broken file just means "no history yet"."""
    try:
        with open(RECENT_OUTPUT_FOLDERS_PATH, encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out, seen = [], set()
    for entry in data:
        s = str(entry)
        key = _normalized_folder_key(s)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out[:MAX_RECENT_OUTPUT_FOLDERS]


def remember_recent_output_folder(folder):
    """Move `folder` to the front of the persisted recent-folders list
    (de-duplicated via _normalized_folder_key, capped at
    MAX_RECENT_OUTPUT_FOLDERS) and write it back. Best-effort: a write
    failure is swallowed. Returns the updated list."""
    folder = str(folder)
    key = _normalized_folder_key(folder)
    updated = [folder] + [p for p in load_recent_output_folders()
                          if _normalized_folder_key(p) != key]
    updated = updated[:MAX_RECENT_OUTPUT_FOLDERS]
    try:
        with open(RECENT_OUTPUT_FOLDERS_PATH, 'w', encoding='utf-8') as f:
            json.dump(updated, f, indent=2)
    except Exception:
        pass
    return updated


def prompt_cell_type_selection():
    print("Select cell type subset to compute UMAP for:")
    for key, label in CELL_TYPE_OPTIONS.items():
        print(f"  {key}. {label}")
    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice in CELL_TYPE_OPTIONS:
            return CELL_TYPE_OPTIONS[choice]
        print("Invalid selection; please enter 1, 2, or 3.")


def materialize_subset(adata, mask):
    """Subset adata by a boolean mask and bring the result into memory.
    Backed AnnData objects can't be .copy()'d directly (anndata raises,
    asking for a filename); .to_memory() on the subset view instead reads
    only the kept rows from disk, which is what keeps memory use down when
    filtering runs right after a backed load."""
    subset = adata[mask]
    return subset.to_memory() if adata.isbacked else subset.copy()


def filter_by_cell_type(adata, selection, class_col='class'):
    """Subset adata to the selected cell type, based on the leading numeric
    ID in the 'class' column (non-neuron classes are 30-34)."""
    if selection == 'All':
        return adata
    if class_col not in adata.obs.columns:
        print(f"Warning: '{class_col}' column not found; cannot filter by cell type. "
              "Proceeding with all cells.")
        return adata

    class_ids = adata.obs[class_col].astype(str).str.strip().str.extract(r'^(\d+)')[0]
    class_ids = pd.to_numeric(class_ids, errors='coerce')
    valid = class_ids.notna()
    is_non_neuron = valid & class_ids.isin(NON_NEURON_CLASS_IDS)

    if selection == 'NonNeurons':
        keep = is_non_neuron
    else:  # 'Neurons'
        keep = valid & ~is_non_neuron

    filtered = materialize_subset(adata, keep.to_numpy())
    print(f"Filtered to {selection}: {filtered.n_obs} of {adata.n_obs} cells kept.")
    return filtered


def get_section_labels(adata, abc_cache):
    """Return a pandas Series of per-cell section labels, aligned to
    adata.obs.index. Prefers the column already present on adata.obs;
    otherwise loads it from the ABC atlas cell_metadata table."""
    if SECTION_COL in adata.obs.columns:
        return adata.obs[SECTION_COL]
    try:
        cell_metadata_path = abc_cache.get_metadata_path(
            directory='MERFISH-C57BL6J-638850',
            file_name='cell_metadata_with_cluster_annotation'
        )
        header_cols = pd.read_csv(cell_metadata_path, nrows=0).columns.tolist()
        if SECTION_COL not in header_cols:
            print(f"Warning: '{SECTION_COL}' column not found in cell_metadata; "
                  "section selection unavailable.")
            return None
        index_col_name = header_cols[0]
        cell_meta = pd.read_csv(
            cell_metadata_path, index_col=0, converters={0: str},
            usecols=[index_col_name, SECTION_COL],
        )
        return cell_meta[SECTION_COL].reindex(adata.obs.index)
    except Exception as e:
        print(f"Warning: could not load '{SECTION_COL}' for section selection ({e}).")
        return None


def sanitize_section_token(label):
    """Turn a section label into a short filesystem-safe token for filenames."""
    match = re.search(r'(\d+)\s*$', str(label))
    return match.group(1) if match else re.sub(r'[^A-Za-z0-9]+', '', str(label))


def section_sort_key(label):
    """Sort key that orders section labels numerically by their trailing
    number (e.g. '...36' before '...100'), falling back to plain string
    comparison for labels without one."""
    match = re.search(r'(\d+)\s*$', str(label))
    if match:
        return (0, int(match.group(1)))
    return (1, str(label))


def sorted_sections_descending(unique_sections):
    return sorted(unique_sections, key=section_sort_key, reverse=True)


def parse_index_selection(raw):
    """Parse a comma-separated list of indices and/or dash-ranges, e.g.
    '28-33, 35' -> [28, 29, 30, 31, 32, 33, 35]. Raises ValueError on any
    malformed token. Preserves first-seen order and drops duplicates."""
    indices = []
    for token in raw.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            start_str, sep, end_str = token.partition('-')
            start, end = int(start_str.strip()), int(end_str.strip())
            if start > end:
                start, end = end, start
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(token))
    return list(dict.fromkeys(indices))


def prompt_section_selection(section_series):
    """Console-based section picker. Returns (selected_labels, sections_suffix,
    rois); selected_labels is None when all sections are selected. rois is
    always [] here — ROI drawing needs the GUI, so it's unavailable in the
    console fallback.

    A future GUI can bypass this prompt entirely and call filter_by_sections()
    directly with a list of section labels chosen by clicking.
    """
    if section_series is None:
        print("No section metadata available; proceeding with all sections.")
        return None, 'allsections', []

    counts = section_series.value_counts()
    unique_sections = sorted_sections_descending(counts.index.tolist())
    print(f"Available brain sections ({len(unique_sections)}):")
    for i, s in enumerate(unique_sections, start=1):
        print(f"  {i}. {s} ({counts[s]} cells)")
    print("Enter section number(s) to analyze — comma-separated, dash-ranges allowed "
          "(e.g. 28-33, 35), or 'all':")
    while True:
        raw = input("> ").strip()
        if raw.lower() == 'all':
            return None, 'allsections', []
        try:
            idxs = parse_index_selection(raw)
        except ValueError:
            idxs = []
        if idxs and all(1 <= i <= len(unique_sections) for i in idxs):
            selected = [unique_sections[i - 1] for i in idxs]
            tokens = '-'.join(sanitize_section_token(s) for s in selected)
            return selected, f'sections-{tokens}', []
        print(f"Invalid input; enter comma-separated numbers and/or dash-ranges between "
              f"1 and {len(unique_sections)}, or 'all'.")


SPATIAL_CSV_CHUNK_SIZE = 200_000  # see the chunked read in load_section_spatial_coords()
SPATIAL_CSV_CACHE_VERSION = 1


def _spatial_csv_cache_path(cell_metadata_path):
    """Shared (not run-specific) disk-cache path for the full, un-reindexed
    cell_metadata table read by load_section_spatial_coords(). Keyed on the
    source CSV's own mtime+size so a re-downloaded/updated atlas file is
    detected and the cache is rebuilt automatically."""
    st = os.stat(cell_metadata_path)
    key = f"{cell_metadata_path}|{st.st_mtime_ns}|{st.st_size}"
    digest = hashlib.md5(key.encode('utf-8')).hexdigest()[:16]
    return CACHE_DIR / f"spatial_coords_v{SPATIAL_CSV_CACHE_VERSION}_{digest}.pkl"


def load_section_spatial_coords(adata, abc_cache, progress_callback=None):
    """Load per-cell 'x'/'y' spatial coordinates (and 'class'/'subclass'/
    'supertype'/'cluster', if available) from the ABC atlas cell_metadata table (these
    aren't normally joined into adata), aligned to adata.obs.index. Used for
    the GUI thumbnail grid and the ROI/group spatial maps — not persisted
    onto adata. Returns None on failure.

    `progress_callback`, if given, is called as `progress_callback(rows_read,
    total_rows)` after each chunk of the underlying CSV is read — see
    SPATIAL_CSV_CHUNK_SIZE — so a caller polling from another thread (e.g.
    show_processing_dialog's Cancel-button loop) can show real progress.
    `total_rows` is None if it couldn't be determined up front."""
    try:
        cell_metadata_path = abc_cache.get_metadata_path(
            directory='MERFISH-C57BL6J-638850',
            file_name='cell_metadata_with_cluster_annotation'
        )
        header_cols = pd.read_csv(cell_metadata_path, nrows=0).columns.tolist()
        if 'x' not in header_cols or 'y' not in header_cols:
            print("Warning: 'x'/'y' spatial columns not found in cell_metadata; "
                  "GUI section picker unavailable.")
            return None
        index_col_name = header_cols[0]
        wanted_cols = [c for c in ['x', 'y', 'class', 'subclass', 'supertype', 'cluster'] if c in header_cols]

        cache_path = None
        try:
            cache_path = _spatial_csv_cache_path(cell_metadata_path)
            if cache_path.exists():
                t0 = time.perf_counter()
                cell_meta = pd.read_pickle(cache_path)
                if ZOOM_DEBUG_DIAGNOSTICS:
                    print(f"[spatial-cache] loaded {len(cell_meta)} rows from disk cache in "
                          f"{time.perf_counter() - t0:.3f}s ({cache_path.name})")
                if progress_callback is not None:
                    progress_callback(len(cell_meta), len(cell_meta))
                return cell_meta.reindex(adata.obs.index)
        except Exception as e:
            if ZOOM_DEBUG_DIAGNOSTICS:
                print(f"[spatial-cache] cache read failed, falling back to CSV ({e})")

        total_rows = None
        if progress_callback is not None:
            # A cheap-ish extra pass (raw line counting, not CSV parsing) so
            # progress can be reported as a real percentage. Only bothered
            # with when a caller actually wants progress.
            try:
                with open(cell_metadata_path, 'rb') as f:
                    total_rows = sum(1 for _ in f) - 1  # minus the header line
            except Exception:
                total_rows = None

        # Read in chunks rather than one single pd.read_csv() call. This
        # file is multi-million-row, and pandas' C parser doesn't release
        # the GIL during a single read — when this function runs in a
        # background thread (see handle_double_click's use of it) to keep
        # the "Processing..." dialog's Cancel button responsive, one giant
        # blocking call starves the main thread of any chance to run for
        # the *entire* read, making Cancel do nothing until it's already
        # too late. Chunked reads return control to Python between chunks,
        # giving the GIL — and thus the main thread's polling loop — real
        # opportunities to run throughout, not just at the very end; it also
        # gives us a natural point to report progress.
        reader = pd.read_csv(
            cell_metadata_path, index_col=0, converters={0: str},
            usecols=[index_col_name] + wanted_cols, chunksize=SPATIAL_CSV_CHUNK_SIZE,
        )
        chunks = []
        rows_read = 0
        for chunk in reader:
            chunks.append(chunk)
            rows_read += len(chunk)
            if progress_callback is not None:
                progress_callback(rows_read, total_rows)
        cell_meta = pd.concat(chunks, copy=False)
        if cache_path is not None:
            try:
                t0 = time.perf_counter()
                cell_meta.to_pickle(cache_path)
                if ZOOM_DEBUG_DIAGNOSTICS:
                    print(f"[spatial-cache] saved {len(cell_meta)} rows to disk cache in "
                          f"{time.perf_counter() - t0:.3f}s ({cache_path.name})")
            except Exception as e:
                if ZOOM_DEBUG_DIAGNOSTICS:
                    print(f"[spatial-cache] cache save failed ({e})")
        return cell_meta.reindex(adata.obs.index)
    except Exception as e:
        print(f"Warning: could not load spatial coordinates for GUI section picker ({e}).")
        return None


def ensure_spatial_load_started(spatial_cache, adata, abc_cache):
    """Starts a background thread loading the full spatial coordinates
    dataframe into spatial_cache['df'], unless one's already loaded or
    already in flight — idempotent, so it's safe to call from multiple
    places (the grid picker opening, a double-click, a single-section
    window's own hover/gene-view code) without ever starting more than one
    redundant load; later callers just find spatial_cache['load_thread']
    already set and wait on that instead.

    `spatial_cache` must be a dict with 'df'/'load_thread'/'load_progress'
    ({'rows':, 'total':})/'load_error' keys — see prompt_section_selection_
    gui's own spatial_cache, which callers are expected to share (by
    reference) rather than each keeping a separate one."""
    if spatial_cache['df'] is not None or spatial_cache['load_thread'] is not None:
        return

    def worker():
        try:
            spatial_cache['df'] = load_section_spatial_coords(
                adata, abc_cache,
                progress_callback=lambda rows, total: spatial_cache['load_progress'].update(rows=rows, total=total),
            )
        except Exception as e:
            spatial_cache['load_error'] = e
            # Cleared (not left pointing at this now-finished thread) so a
            # later call sees neither df nor load_thread set and retries,
            # instead of treating a failed load as permanent.
            spatial_cache['load_thread'] = None

    thread = threading.Thread(target=worker, daemon=True)
    spatial_cache['load_thread'] = thread
    thread.start()


IMPUTED_DATASET_DIRECTORY = 'MERFISH-C57BL6J-638850-imputed'
IMPUTED_DATASET_FILE_NAME = 'C57BL6J-638850-imputed/log2'


def is_data_file_cached(abc_cache, directory, file_name):
    """Best-effort check for whether `directory`/`file_name` (as would be
    passed to abc_cache.get_data_path) is already present in the local
    cache — unlike get_data_path itself, this never triggers a download for
    a file that isn't. Returns True/False, or None if the check itself
    failed (e.g. abc_atlas_access's internal cache/manifest shape changed
    in some future version) — callers should treat None as "unknown,
    assume not cached" rather than erroring, since there's no public
    abc_atlas_access API for this as of this writing; reaching into
    AbcProjectCache's underlying S3CloudCache/LocalCache and its manifest
    is the only way to answer this without paying for the download it's
    trying to avoid."""
    try:
        file_attributes = abc_cache.cache._manifest.get_file_attributes(
            directory=directory, file_name=file_name,
        )
        return abc_cache.cache._file_exists(file_attributes)
    except Exception:
        return None


def load_imputed_adata(abc_cache):
    """Opens the imputed-gene-expression MERFISH dataset (C57BL6J-638850-
    imputed/log2 — log2-scale imputed values covering far more genes than
    the standard 500-gene panel) in backed mode, same as the module-level
    `adata` load. Backed opening itself is fast (only the header, not X, is
    read); the slow part users actually wait on is abc_cache downloading
    the file to local cache the first time, which this doesn't (and can't
    easily) report progress for — callers should run this off the main
    thread and show an indeterminate-progress dialog around it, same as
    ensure_imputed_gene_dataset_loaded does."""
    h5ad_path = abc_cache.get_data_path(
        directory=IMPUTED_DATASET_DIRECTORY,
        file_name=IMPUTED_DATASET_FILE_NAME,
    )
    return anndata.read_h5ad(h5ad_path, backed='r')


def restore_cursor_after_pending_draw(fig, still_busy, cursor=Cursors.POINTER):
    """Put `cursor` back once the figure has finished redrawing, not right
    away. Operations usually end by marking themselves done and then drawing,
    either synchronously right after, or with draw_idle(), which runs only
    once the handler returns. Restoring immediately showed the arrow for that
    whole redraw. This queues the restore on Tk's idle queue behind any
    pending draw, re-queueing while a draw_idle() is still outstanding.
    `still_busy()` returning True means a new operation started in the
    meantime, and it owns the cursor now."""
    try:
        widget = fig.canvas.get_tk_widget()
    except Exception:
        fig.canvas.set_cursor(cursor)
        return

    def attempt():
        try:
            if still_busy():
                return
            if getattr(fig.canvas, '_idle_draw_id', None):
                widget.after_idle(attempt)
                return
            fig.canvas.set_cursor(cursor)
        except Exception:
            pass  # window closed in the meantime
    widget.after_idle(attempt)


def set_wait_cursor_on_open_figures():
    """Show the busy cursor over every open pyplot window. Returns a function
    that puts each window's previous cursor back, which may be a resize or
    other custom cursor rather than the arrow. Works on the Tk widgets
    directly, so a window whose own code tracks its cursor state isn't
    left out of sync. Windows that aren't Tk are skipped."""
    saved = []
    # Gcf rather than plt.figure(num), which would also change pyplot's
    # current figure.
    from matplotlib._pylab_helpers import Gcf
    for manager in Gcf.get_all_fig_managers():
        try:
            widget = manager.canvas.get_tk_widget()
            saved.append((widget, widget.cget('cursor')))
            widget.configure(cursor='watch')
            widget.update_idletasks()
        except Exception:
            pass

    def restore():
        for widget, cursor in saved:
            try:
                widget.configure(cursor=cursor)
            except Exception:
                pass  # window closed during the load
    return restore


def ensure_imputed_gene_dataset_loaded(imputed_state, abc_cache):
    """Loads the imputed gene dataset into imputed_state['adata'] if it
    isn't already (or already in flight). Warns the user first that this
    can take a long time — but only when it actually might: skipped when
    is_data_file_cached() can confirm the file is already sitting in the
    local cache, so re-selecting 'Imputed Gene' (once it's been downloaded
    once — this session, or an earlier run) doesn't nag about a download
    that isn't going to happen. When the cached-or-not check itself can't
    be answered (is_data_file_cached() returns None — see its own
    docstring), this still warns, since an unnecessary prompt is a much
    smaller cost than an unannounced multi-minute download.

    `imputed_state` must be a dict with 'adata'/'load_thread'/'load_error'
    keys — shared by reference across every caller that wants to reuse the
    same load rather than each paying for their own (see
    prompt_section_selection_gui's own imputed_state, which callers should
    pass through/reuse rather than create a fresh one, same reasoning as
    ensure_spatial_load_started's spatial_cache).

    Returns True once imputed_state['adata'] is ready to use, False if the
    user declined the warning, cancelled, or the load failed — callers
    should leave whatever was selected before unchanged in that case."""
    if imputed_state['adata'] is not None:
        return True
    if imputed_state['load_thread'] is None:
        already_cached = is_data_file_cached(
            abc_cache, IMPUTED_DATASET_DIRECTORY, IMPUTED_DATASET_FILE_NAME,
        )
        if already_cached is not True:
            try:
                import tkinter as tk
                from tkinter import messagebox
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                confirmed = messagebox.askyesno(
                    "Load imputed gene dataset?",
                    "The imputed gene dataset covers far more genes than the standard "
                    "500-gene MERFISH panel, but is much larger. The first time it's "
                    "loaded this session (and cached to disk, if not already), this can "
                    "take a long time. Continue?",
                )
                root.destroy()
            except Exception as e:
                print(f"Could not show confirmation dialog ({e}); not loading the imputed dataset.")
                return False
            if not confirmed:
                return False

        def worker():
            try:
                imputed_state['adata'] = load_imputed_adata(abc_cache)
            except Exception as e:
                imputed_state['load_error'] = e
                # Cleared (not left pointing at this now-finished thread) so a
                # later attempt retries fresh instead of treating this as
                # permanent — same reasoning as ensure_spatial_load_started.
                imputed_state['load_thread'] = None

        thread = threading.Thread(target=worker, daemon=True)
        imputed_state['load_thread'] = thread
        thread.start()

    thread = imputed_state['load_thread']
    # Hourglass over the app's other windows (e.g. the section window that
    # asked for the dataset) for the whole load. Opened before the dialog, so
    # the dialog keeps its normal arrow and Cancel still looks clickable.
    restore_cursors = set_wait_cursor_on_open_figures()
    proc_fig, cancel_flag, set_progress = show_processing_dialog(
        "Loading imputed gene dataset (first use this session; this can take a while)..."
    )
    try:
        set_progress(None)  # indeterminate — no fine-grained progress available for this load
        while thread.is_alive() and not cancel_flag['cancelled']:
            proc_fig.canvas.flush_events()
            time.sleep(0.05)
    finally:
        plt.close(proc_fig)
        restore_cursors()

    if cancel_flag['cancelled']:
        # As with spatial_cache's own cancel handling, the thread keeps
        # running in the background rather than being killed outright — a
        # later selection just finds imputed_state['adata'] already there
        # (or retries, if it had failed by then) instead of paying for a
        # second full load.
        print("Cancelled — the imputed dataset load continues in the background; try again shortly.")
        return False
    if imputed_state['adata'] is None:
        print(f"Could not load imputed gene dataset: {imputed_state['load_error']}")
        return False
    print(f"Loaded imputed gene dataset with {imputed_state['adata'].n_obs} cells x "
          f"{imputed_state['adata'].n_vars} genes.")
    return True


def extract_leading_numeric_id(series):
    """Pull the leading integer ID off ABC atlas category strings like
    '30 Astro-Epen' -> 30. Returns a float Series (NaN where no match).

    Regex-extracts only the *unique* strings in `series`, then maps that
    back onto every row via a dict lookup — class/subclass/supertype/
    cluster/leiden labels each have only a few hundred to a few thousand
    distinct values even when `series` itself is millions of rows (one row
    per cell), so this does orders of magnitude less regex work than
    running str.extract over every row directly."""
    s = series.astype(str).str.strip()
    uniques = s.unique()
    extracted = pd.Series(uniques).str.extract(r'^(\d+)')[0]
    id_by_value = dict(zip(uniques, pd.to_numeric(extracted, errors='coerce')))
    return s.map(id_by_value)


DEFAULT_GREY_RGBA = (0.8, 0.8, 0.8, 1.0)

# 10 mutually distinguishable colors (Kelly/Boynton-style max-contrast set,
# avoiding low-saturation/near-white/near-black entries) for
# save_group_spatial_maps(): assigned by a category's *position* within
# its group of up to 10, so the same 10 colors — in the same order — are
# reused across every group's figure, unlike class_id_to_color()'s
# golden-angle hue stepping (great for spreading dozens of IDs apart on
# average, but with only 10 in play at once, some pairs land close enough
# in hue to look similar).
SPATIAL_MAP_COLORS = [
    '#e6194b',  # red
    '#3cb44b',  # green
    '#4363d8',  # blue
    '#f58231',  # orange
    '#911eb4',  # purple
    '#42d4f4',  # cyan
    '#f032e6',  # magenta
    '#9a6324',  # brown
    '#000075',  # navy
    '#808000',  # olive
]


def class_id_to_color(cid):
    """Deterministic, bright color for a given ABC atlas class ID — a
    function of the ID's own value (golden-angle-stepped hue), not of which
    other classes happen to be present in whatever's currently being
    plotted. That statelessness is what keeps a given class the same color
    everywhere it's drawn (grid, single-section picker, ROI map, ...);
    computing a palette from only the locally-visible classes (the previous
    approach) assigns colors by each class's *rank* within that local set,
    which differs from one view to another and made the same class look
    like a different color depending on what else was in the view."""
    golden_angle = 0.6180339887498949
    hue = (cid * golden_angle) % 1.0
    r, g, b = mcolors.hsv_to_rgb((hue, 0.95, 0.95))
    return (r, g, b, 1.0)


GOLDEN_ANGLE = 0.6180339887498949  # shared with class_id_to_color's own local copy above

# Circle first, then nine more of matplotlib's filled markers — paired
# with CATEGORY_COLORS_PER_CYCLE (20) and the filled/open doubling in
# category_rank_shape below, 10 shapes x 20 colors x 2 fill styles = 400
# combinations before the whole cycle repeats.
CATEGORY_MARKER_SHAPES = ('o', 's', '^', '<', 'v', '>', 'D', 'P', 'X', '*')
CATEGORY_COLORS_PER_CYCLE = 20  # 10 hues x 2 saturation/value families — see category_rank_color


def category_rank_color(rank):
    """Deterministic color for a category's rank (0 = most common, by
    whatever ordering assigned that rank — see compute_ranked_category_
    colors/redraw_subclass, its own two callers) within a 20-color cycle:
    10 hues stepped by the golden angle (same mechanism as class_id_to_
    color), generated twice — once vivid, once muted.

    A single hue-only ring (class_id_to_color's own approach, used here
    until this was added) assigns well over 100 *numerically* distinct
    hues at fixed saturation/value, but human hue discrimination at fixed
    saturation/value tops out around a dozen-ish steps — past that,
    hues that are numerically well-separated still read as "the same
    color". Adding a second saturation/value family roughly doubles how
    many colors stay individually recognizable before the cycle repeats.
    Paired with category_rank_shape (below): once the 20-color cycle
    itself repeats, the shape changes too, so two categories that land on
    the same color again still read as different at a glance."""
    hue_index = rank % (CATEGORY_COLORS_PER_CYCLE // 2)
    family = (rank // (CATEGORY_COLORS_PER_CYCLE // 2)) % 2  # 0 = vivid, 1 = muted
    hue = (hue_index * GOLDEN_ANGLE) % 1.0
    saturation, value = (0.90, 0.95) if family == 0 else (0.45, 0.85)
    r, g, b = mcolors.hsv_to_rgb((hue, saturation, value))
    return mcolors.to_hex((r, g, b))


def category_rank_shape(rank):
    """(marker, is_open) for a category's rank — paired with category_
    rank_color (above). Every full color cycle (CATEGORY_COLORS_PER_CYCLE
    = 20 ranks) advances to the next marker in CATEGORY_MARKER_SHAPES (10
    of them); once every shape has been used with every color (20 * 10 =
    200 combinations), the same 200 repeat again as *open* (hollow)
    markers before the whole sequence truly repeats at rank 400
    (20 * 10 * 2) — see CATEGORY_MARKER_SHAPES's own comment. Derived from
    len(CATEGORY_MARKER_SHAPES) rather than a hardcoded count, so this
    stays correct if that tuple's length ever changes again."""
    combos_per_fill = CATEGORY_COLORS_PER_CYCLE * len(CATEGORY_MARKER_SHAPES)  # 200
    wrapped = rank % (combos_per_fill * 2)  # 400
    shape_index = (wrapped // CATEGORY_COLORS_PER_CYCLE) % len(CATEGORY_MARKER_SHAPES)
    is_open = wrapped >= combos_per_fill
    return CATEGORY_MARKER_SHAPES[shape_index], is_open


def build_neuron_class_colors(class_ids_all, grey=DEFAULT_GREY_RGBA, color_ids_all=None):
    """Given a float array of per-cell class IDs (NaN allowed), return
    (point_colors_all, neuron_mask_all): an Nx4 RGBA array coloring each
    neuron by class (bright, evenly-spaced hues, stable per class ID — see
    class_id_to_color) and gray for everything else (non-neurons and
    unknown/NaN classes), plus the boolean neuron mask. Returns (None, None)
    if class_ids_all is None.

    `color_ids_all`, if given, is a parallel array of IDs at another
    taxonomy level (subclass/supertype/cluster) to color neurons by instead
    of class. Which cells count as neurons is still decided by class, so
    non-neurons stay gray at every level; a neuron whose ID at that level is
    unknown (NaN) is gray too."""
    if class_ids_all is None:
        return None, None
    neuron_mask_all = ~np.isin(class_ids_all, list(NON_NEURON_CLASS_IDS))
    point_colors_all = np.tile(np.array(grey), (len(class_ids_all), 1))

    ids_all = class_ids_all if color_ids_all is None else color_ids_all
    neuron_ids = ids_all[neuron_mask_all]
    unique_ids = sorted(pd.unique(neuron_ids[~np.isnan(neuron_ids)]))
    for cid in unique_ids:
        point_colors_all[neuron_mask_all & (ids_all == cid)] = class_id_to_color(cid)

    return point_colors_all, neuron_mask_all


def scatter_gray_then_colored(ax, xs, ys, is_neuron, colors, grey=DEFAULT_GREY_RGBA,
                               gray_size=0.1, colored_size=0.3, linewidths=0):
    """Plot non-neurons in gray first (background layer), then colored
    neurons on top, so the tissue outline stays visible without competing
    for attention."""
    ax.scatter(xs[~is_neuron], ys[~is_neuron], s=gray_size,
               c=np.array(grey).reshape(1, -1), linewidths=linewidths, zorder=1)
    ax.scatter(xs[is_neuron], ys[is_neuron], s=colored_size,
               c=colors[is_neuron], linewidths=linewidths, zorder=2, alpha=0.5)


def raise_figure_window(fig):
    """Best-effort attempt to bring a figure's window to the front and give
    it input focus. Matters most for figures opened from inside another
    window's event handler (nested plt.show(), e.g. a confirmation dialog
    popped up mid double-click) — some backends don't automatically focus
    the new window, so clicks can silently land on the window underneath
    instead, making the new window's buttons look unresponsive."""
    try:
        window = fig.canvas.manager.window
    except AttributeError:
        return
    try:
        # Force the window to actually be created/mapped now, before the
        # focus-stealing calls below — on some backends the window object
        # exists but isn't visible yet until plt.show() runs its own show(),
        # which would be too late for lift()/raise_() to have any effect.
        fig.canvas.manager.show()
    except Exception:
        pass
    try:  # Tk (TkAgg)
        window.attributes('-topmost', 1)
        window.attributes('-topmost', 0)
        window.lift()
        window.focus_force()
    except Exception:
        pass
    try:  # Qt (QtAgg/Qt5Agg)
        window.raise_()
        window.activateWindow()
    except Exception:
        pass


def make_textbox_blit_fast(textbox, blit_func):
    """Patch `textbox` (a matplotlib.widgets.TextBox) so every keystroke
    blits via `blit_func()` instead of triggering a full, synchronous
    `fig.canvas.draw()` — which is what TextBox's own internal
    `_rendercursor()` does on *every single keystroke*, entirely on its
    own, regardless of what any `on_text_change` observer does. For a
    large cached image (particularly the section-grid picker's
    multi-section grid), that one call was the actual dominant cost behind
    sluggish typing — no amount of optimizing an application's own
    on_text_change callback touches it, since it happens inside the widget
    itself, not in response to that callback.

    This re-implements _rendercursor's cursor-positioning logic exactly
    (copied from matplotlib's own source, matplotlib 3.10) but swaps its
    final fig.canvas.draw() for blit_func() — fragile in the sense that
    it'll silently stop taking effect (falling back to TextBox's own slow
    default, not breaking outright) if a future matplotlib version changes
    that internal implementation; acceptable for a pinned local script."""
    def fast_rendercursor(self):
        # .figure rather than get_figure(root=True), which needs matplotlib 3.10+.
        # Same result here, since the textbox's axes are never in a subfigure.
        fig = self.ax.figure
        if fig._get_renderer() is None:
            fig.canvas.draw()

        text = self.text_disp.get_text()
        widthtext = text[:self.cursor_index]

        bb_text = self.text_disp.get_window_extent()
        self.text_disp.set_text(widthtext or ",")
        bb_widthtext = self.text_disp.get_window_extent()

        if bb_text.y0 == bb_text.y1:
            bb_text.y0 -= bb_widthtext.height / 2
            bb_text.y1 += bb_widthtext.height / 2
        elif not widthtext:
            bb_text.x1 = bb_text.x0
        else:
            bb_text.x1 = bb_text.x0 + bb_widthtext.width

        self.cursor.set(
            segments=[[(bb_text.x1, bb_text.y0), (bb_text.x1, bb_text.y1)]],
            visible=True)
        self.text_disp.set_text(text)

        blit_func()

    textbox._rendercursor = fast_rendercursor.__get__(textbox, type(textbox))


def make_textbox_stop_typing_blit_fast(textbox, blit_func):
    """Patch `textbox` so stop_typing() blits via `blit_func()` instead of
    a full, synchronous `fig.canvas.draw()`. TextBox._click() — connected
    to 'button_press_event' globally, the same as every other widget, not
    scoped to the textbox's own axes — calls stop_typing() on *any* click
    that lands outside the box, unconditionally, regardless of whether the
    box was ever actually focused/typed into. In a figure with an
    expensive background (many section-panel scatters, in particular),
    that means nearly every click anywhere in the window pays for one full
    synchronous re-render, entirely inside matplotlib's own widget
    internals — invisible to and unpreventable by any of *our* code's own
    draw_idle()/blit routing, since it isn't triggered through any
    callback we control.

    Reimplements stop_typing()'s logic exactly (matplotlib 3.9's own
    source) but swaps its final fig.canvas.draw() for blit_func() — same
    fragility caveat as make_textbox_blit_fast above."""
    def fast_stop_typing(self):
        if self.capturekeystrokes:
            self._on_stop_typing()
            self._on_stop_typing = None
            notifysubmit = True
        else:
            notifysubmit = False
        self.capturekeystrokes = False
        self.cursor.set_visible(False)
        blit_func()
        if notifysubmit and self.eventson:
            # Because process() might throw an error in the user's code, only
            # call it once we've already done our cleanup.
            self._observers.process('submit', self.text)

    textbox.stop_typing = fast_stop_typing.__get__(textbox, type(textbox))


def make_textbox_motion_blit_fast(textbox, blit_func):
    """Patch `textbox` so its hover-color redraw blits via `blit_func()`
    instead of a full, synchronous `fig.canvas.draw()`. TextBox._motion is
    connected to 'motion_notify_event' *globally* on the whole canvas, not
    scoped to the box's own axes (AxesWidget.ignore() only checks self.
    active, never whether the event actually landed inside self.ax) — so it
    runs on *every* mouse move anywhere in the figure, and fires that full
    draw every time the cursor crosses into or out of the box's own hover
    region. With dozens of section panels each carrying a real, several-
    thousand-point scatter (any mode where the query/gene box stays active
    — Gene, Imputed Gene, Specified IDs — categorical "All <level>s" modes
    leave it inactive and are unaffected), simply moving the mouse near the
    box was enough to trigger this repeatedly, each one queued behind the
    last, reading as the whole UI going unresponsive for as long as the
    mouse kept moving — with no visible content change to explain it, since
    the hover-color change itself is barely noticeable next to the redraw
    it triggers.

    Unlike _rendercursor (called *indirectly*, via self._rendercursor()
    from within _keypress — see make_textbox_blit_fast, where simply
    overwriting the instance attribute is enough), _motion is connected
    *directly* as the event callback itself in TextBox.__init__, so
    matplotlib's event system already holds a reference to the original
    bound method; reassigning textbox._motion afterward wouldn't change
    what that existing connection actually calls. The original connection
    has to be torn down and replaced with one pointing at the patched
    version instead.

    Reimplements _motion's logic exactly (matplotlib 3.9's own source) but
    swaps its final fig.canvas.draw() for blit_func() — same fragility
    caveat as make_textbox_blit_fast above, and if the disconnect below
    ever fails (a future matplotlib reordering its own __init__), this
    simply leaves the original slow _motion in place rather than raising."""
    def fast_motion(self, event):
        if self.ignore(event):
            return
        c = self.hovercolor if self.ax.contains(event)[0] else self.color
        if not mcolors.same_color(c, self.ax.get_facecolor()):
            self.ax.set_facecolor(c)
            if self.drawon:
                blit_func()

    try:
        # Index 2 = 'motion_notify_event', per TextBox.__init__'s own fixed
        # connect_event() call order: button_press, button_release, motion,
        # key_press, resize.
        textbox.canvas.mpl_disconnect(textbox._cids[2])
        del textbox._cids[2]
    except Exception:
        return
    textbox._motion = fast_motion.__get__(textbox, type(textbox))
    textbox.connect_event('motion_notify_event', textbox._motion)


def enable_textbox_clipboard_shortcuts(textbox):
    """Ctrl+A/C/X/V for `textbox` (a matplotlib.widgets.TextBox) -- entirely
    missing from matplotlib's own _keypress, which only recognizes single
    characters and a handful of named keys (left/right/home/end/backspace/
    delete); a modifier combo like 'ctrl+c' matches none of its branches and
    is silently dropped. There's no selection concept in TextBox at all (no
    click-drag or shift-arrow range, nothing to highlight), so this doesn't
    add one -- it treats Ctrl+A as "select the whole field": Ctrl+C/X
    always act on the *whole* current text (there's nothing else to act
    on), and Ctrl+A only changes what Ctrl+V
    (or typing a plain character, handled by TextBox's own _keypress) does
    next -- replace everything instead of inserting at the cursor -- exactly
    like a normal OS text field after selecting all then typing over it.
    That pending replace is cleared by any other key, including a plain
    character (observed here, not suppressed -- _keypress's own separate
    handling of it is untouched).

    A real per-character range selection (click-drag, shift-arrows) would
    need matplotlib to track and render a selection at all, which it simply
    doesn't; getting that would mean either implementing it from scratch or
    replacing TextBox with a native Tk Entry/Text widget, which is a
    separate, larger change.

    Connects to the widget's own figure's 'key_press_event' -- the same
    event TextBox._keypress is already separately connected to; since
    Ctrl+A/C/X/V match none of that method's own branches, the two don't
    need to coordinate. Only acts while `textbox` is the one actually being
    typed into (capturekeystrokes), so this is safe to enable on more than
    one textbox in the same figure."""
    # .figure rather than get_figure(root=True), which needs matplotlib
    # 3.10+ -- same compatibility reasoning as make_textbox_blit_fast above.
    fig = textbox.ax.figure
    pending_replace = {'active': False}

    def get_tk_widget():
        # The actual Tk widget (not fig.canvas.manager.window, a wrapper)
        # -- its clipboard_get/clipboard_clear/clipboard_append, inherited
        # from tkinter.Misc, reach the one process-wide Tk clipboard
        # regardless of which widget in this interpreter calls them.
        return fig.canvas.get_tk_widget()

    def replace_text(new_text, cursor_at):
        # Mirrors TextBox._keypress's own update sequence for a plain
        # character, so this is indistinguishable from normal typing to
        # every existing on_text_change observer (autocomplete, clearing a
        # bad-name mark, ...) -- only 'change' fires, not 'submit', same as
        # any other keystroke that isn't literally Enter.
        textbox.text_disp.set_text(new_text)
        textbox.cursor_index = cursor_at
        textbox._rendercursor()
        if textbox.eventson:
            textbox._observers.process('change', textbox.text)

    def on_key(event):
        if not textbox.capturekeystrokes:
            return
        key = event.key
        if key == 'ctrl+a':
            pending_replace['active'] = True
            return
        replace_on_next, pending_replace['active'] = pending_replace['active'], False
        if key == 'ctrl+c':
            try:
                widget = get_tk_widget()
                widget.clipboard_clear()
                widget.clipboard_append(textbox.text)
            except Exception:
                pass  # clipboard unavailable (e.g. headless) -- nothing to copy to
        elif key == 'ctrl+x':
            try:
                widget = get_tk_widget()
                widget.clipboard_clear()
                widget.clipboard_append(textbox.text)
            except Exception:
                pass
            replace_text('', 0)
        elif key == 'ctrl+v':
            try:
                pasted = get_tk_widget().clipboard_get()
            except Exception:
                return  # nothing text-like on the clipboard (empty, an image, ...)
            # Collapses a multi-line clipboard source (e.g. a spreadsheet
            # cell) into this single-line box instead of pasting literal
            # line breaks into it.
            pasted = ' '.join(pasted.splitlines())
            if replace_on_next:
                replace_text(pasted, len(pasted))
            else:
                text = textbox.text
                idx = textbox.cursor_index
                replace_text(text[:idx] + pasted + text[idx:], idx + len(pasted))

    fig.canvas.mpl_connect('key_press_event', on_key)


# Rough allowance for OS window chrome (title bar + borders) that sits
# outside the matplotlib canvas but still counts toward the window's actual
# on-screen footprint — without this, a window whose *content* is exactly
# height_frac of the screen ends up slightly taller than that once its
# title bar is added, and can't fully fit; and centering math based only on
# the content size places it slightly off from the window's true center.
WINDOW_CHROME_HEIGHT_PX = 80
WINDOW_CHROME_WIDTH_PX = 20
# Bare desktop kept on every side of a newly-opened window (see
# fit_figure_window_to_work_area/center_figure_window) — enough that the
# window border itself stays grabbable for resizing rather than sitting
# flush against, or past, the screen edge.
WINDOW_EDGE_MARGIN_PX = 40

def compute_figsize_for_screen_height(aspect_wh, height_frac=0.8, default=(10, 7), width_frac=0.95):
    """Compute a (width, height) figsize in inches so the resulting
    *window* (content plus estimated chrome) is `height_frac` of the screen
    height tall at matplotlib's default figure DPI, keeping the given
    width/height aspect ratio. Sizing the figure *before* creating it this
    way (rather than resizing the window after plt.subplots()) avoids a
    real bug that approach had: resizing the Tk window post-creation
    doesn't reliably force matplotlib to relay out its axes at the new size
    before the window is shown, which clipped the right/bottom of the plot
    and pushed the button axes below the visible area. Falls back to
    `default` if screen dimensions can't be determined.

    The height-driven width is also capped at `width_frac` of the screen's
    own width (height reduced to match, preserving aspect_wh) — a wide
    aspect_wh (e.g. the UMAP window's 2.4:1) combined with a tall/narrow
    screen could otherwise compute a width wider than the screen itself,
    since height_frac alone says nothing about how wide the screen is."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        screen_w_px, screen_h_px = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
    except Exception:
        return default
    dpi = matplotlib.rcParams.get('figure.dpi', 100)
    content_h_px = max(screen_h_px * height_frac - WINDOW_CHROME_HEIGHT_PX, 1)
    height_in = content_h_px / dpi
    width_in = height_in * aspect_wh
    max_width_px = max(screen_w_px * width_frac - WINDOW_CHROME_WIDTH_PX, 1)
    max_width_in = max_width_px / dpi
    if width_in > max_width_in:
        width_in = max_width_in
        height_in = width_in / aspect_wh
    return width_in, height_in


def compute_ui_fontsize(button_height_frac=0.05):
    """Font size (points) shared by every button/text across the section
    grid picker, the single-section ROI picker, and the 'Processing...'
    dialog. Computed once, here, from a representative on-screen button
    height, rather than separately per window from that window's own
    figure size and button-height fraction — those two things used to
    differ enough between windows (the grid and single-section pickers use
    different button-height fractions of their own figure; the processing
    dialog uses a fixed, non-screen-relative figsize) to produce visibly
    different sizes, even though the two pickers' figures actually end up
    the same height in inches (compute_figsize_for_screen_height's height
    only depends on screen height and height_frac, not aspect ratio)."""
    ref_height_in = compute_figsize_for_screen_height(1.0, default=(10, 7))[1]
    button_height_pt = button_height_frac * ref_height_in * 72
    return max(8, min(16, button_height_pt * 0.45))


UI_BUTTON_FONTSIZE = compute_ui_fontsize()


def measured_text_width_pt(text, fontsize):
    """Width, in points, of `text` as matplotlib would actually render it at
    `fontsize` — measured via a throwaway off-screen Agg figure rather than
    estimated from character count, so it's exact regardless of font
    metrics. Used by compute_horizontal_radio_layout() to lay out hand-rolled
    radio widgets without needing to force a real draw of whatever
    (possibly large/expensive) figure they'll actually end up on —
    FigureCanvasAgg.get_renderer() lazily creates a renderer on its own,
    unlike the interactive TkAgg canvas's _get_renderer(), which returns None
    until that figure has actually been drawn once."""
    probe_fig = Figure()
    FigureCanvasAgg(probe_fig)
    probe_text = probe_fig.text(0, 0, text, fontsize=fontsize)
    bbox = probe_text.get_window_extent(renderer=probe_fig.canvas.get_renderer())
    return bbox.width * 72 / probe_fig.dpi


_TEXT_WIDTH_CACHE = {}


def cached_text_width_pt(text, fontsize):
    """measured_text_width_pt(), memoized. For relaying out widgets on every
    window resize, where re-measuring through a throwaway figure each time
    would make drag-resizing sluggish. Callers that need a range of font sizes
    should measure at one fixed size and scale linearly (text width is
    proportional to font size) rather than passing each size here, so the
    cache stays small."""
    key = (text, fontsize)
    if key not in _TEXT_WIDTH_CACHE:
        _TEXT_WIDTH_CACHE[key] = measured_text_width_pt(text, fontsize)
    return _TEXT_WIDTH_CACHE[key]


def compute_horizontal_radio_layout(options, fontsize, left_margin_pt=4.0, dot_label_gap_pt=4.0,
                                     option_gap_pt=14.0, right_margin_pt=4.0):
    """Lay out a hand-rolled horizontal radio widget (dot, label, dot, label,
    ... left to right) using each label's actually-measured rendered width,
    rather than fixed axes-fraction positions for the dots/labels (the
    previous approach in each of this file's 3 radio widgets — grid picker's
    Classes/Gene, single-section picker's Groups/Gene and Linear/Log).

    A fixed axes-fraction layout only ever fits the one screen size/font it
    happened to be tuned against: the dots are a fixed size in *points* (see
    each call site's own dot_size comment) and the labels' rendered width is
    also fixed in points (font metrics don't scale with window size), while
    a fixed *fraction* of the axes shrinks right along with the axes on a
    smaller window or narrower figure — so the same fractional gap that
    looked fine on one screen can end up smaller than the dot's own radius,
    or smaller than a label's own rendered width, on another. That's what
    let a label overlap its own dot, or (on a narrow enough window) overlap
    the *next* dot: the second dot's fixed 0.62-fraction position had no
    relationship to how wide the first label actually rendered.

    Returns (dot_x_pt, label_x_pt, total_width_pt): the first two are tuples
    of point offsets from the axes' left edge — not yet axes-fraction, since
    that depends on how wide the caller ends up making the axes (it should
    be at least `total_width_pt`, converted via the figure's width in
    points, to guarantee no overlap; the caller may make it wider still, in
    which case these positions are simply left-packed with extra room to
    their right)."""
    dot_radius_pt = fontsize / 2
    dot_x_pt = []
    label_x_pt = []
    cursor_pt = left_margin_pt
    for i, opt in enumerate(options):
        dot_center_pt = cursor_pt + dot_radius_pt
        dot_x_pt.append(dot_center_pt)
        label_start_pt = dot_center_pt + dot_radius_pt + dot_label_gap_pt
        label_x_pt.append(label_start_pt)
        cursor_pt = label_start_pt + measured_text_width_pt(opt, fontsize)
        if i < len(options) - 1:
            cursor_pt += option_gap_pt
    total_width_pt = cursor_pt + right_margin_pt
    return tuple(dot_x_pt), tuple(label_x_pt), total_width_pt


def radio_layout_as_axes_fractions(options, fontsize, default_width_frac, fig_width_in, **layout_kwargs):
    """compute_horizontal_radio_layout(), converted to axes-fraction
    positions plus the axes width (as a figure-fraction) the caller should
    actually use — `default_width_frac` if that's already wide enough to fit
    the widget without overlap, otherwise grown to whatever
    compute_horizontal_radio_layout() says is actually needed. When
    `default_width_frac` turns out wider than the content actually needs,
    the dots/labels are centered within that extra width rather than left
    -packed against the axes' left edge with empty space on the right —
    most noticeable for scale_radio_ax, which draws a bordered box behind
    it, where left-packed content read as visibly off-center in that box.
    Extra keyword arguments (e.g. left_margin_pt/right_margin_pt) are passed
    through to compute_horizontal_radio_layout().
    Returns (dot_x_frac, label_x_frac, radio_width_frac)."""
    dot_x_pt, label_x_pt, total_width_pt = compute_horizontal_radio_layout(options, fontsize, **layout_kwargs)
    default_width_pt = default_width_frac * fig_width_in * 72
    axes_width_pt = max(default_width_pt, total_width_pt)
    extra_pt = axes_width_pt - total_width_pt
    if extra_pt > 0:
        offset_pt = extra_pt / 2
        dot_x_pt = tuple(x + offset_pt for x in dot_x_pt)
        label_x_pt = tuple(x + offset_pt for x in label_x_pt)
    radio_width_frac = axes_width_pt / (fig_width_in * 72)
    dot_x_frac = tuple(x / axes_width_pt for x in dot_x_pt)
    label_x_frac = tuple(x / axes_width_pt for x in label_x_pt)
    return dot_x_frac, label_x_frac, radio_width_frac


def scaled_radio_layout_pt(options, base_fontsize, scale, margin_pt):
    """compute_horizontal_radio_layout()'s geometry at font size
    base_fontsize * scale, cheap enough to run on every window resize: label
    widths come from cached_text_width_pt() at base_fontsize and are scaled
    linearly, and its fixed 4pt dot-to-label and 14pt option-to-option gaps
    scale by the same factor so spacing stays proportional to the text.
    margin_pt (already at the target size) is used at both ends.
    Returns (dot_x_pt, label_x_pt, total_width_pt), offsets from the
    widget's left edge."""
    dot_radius_pt = base_fontsize * scale / 2
    dot_x_pt, label_x_pt = [], []
    cursor_pt = margin_pt
    for i, opt in enumerate(options):
        dot_center_pt = cursor_pt + dot_radius_pt
        dot_x_pt.append(dot_center_pt)
        label_start_pt = dot_center_pt + dot_radius_pt + 4.0 * scale
        label_x_pt.append(label_start_pt)
        cursor_pt = label_start_pt + cached_text_width_pt(opt, base_fontsize) * scale
        if i < len(options) - 1:
            cursor_pt += 14.0 * scale
    return dot_x_pt, label_x_pt, cursor_pt + margin_pt


def normalize_tk_scaling(fig):
    """Force Tk's own internal pixel-per-point scaling to match matplotlib's
    dpi assumption (dpi pixels/inch = dpi/72 pixels/point). Without this,
    even with the process declared DPI-aware (see _make_windows_dpi_aware),
    Tk independently applies its own DPI-derived scaling (observed ~2x on a
    150%-scaled 4K display) when it actually realizes the canvas widget's
    size — and TkAgg then auto-adjusts the figure to match whatever size Tk
    actually rendered, silently inflating the window well past the size
    compute_figsize_for_screen_height intended. That's what was pushing the
    bottom button row off-screen: not a missing height allowance, but the
    whole window ending up bigger than requested. Must run right after the
    figure/window is created, before anything else is laid out."""
    try:
        window = fig.canvas.manager.window
        dpi = fig.dpi
        window.tk.call('tk', 'scaling', dpi / 72.0)
    except Exception:
        pass


def _windows_monitor_work_area(window):
    """The work area (screen bounds minus the taskbar, whichever edge it's
    docked to) of whichever monitor `window` is actually on, as (left, top,
    width, height) in physical pixels — or None on failure/non-Windows.

    Windows' own 'zoomed' window state (ShowWindow(SW_MAXIMIZE) — what
    window.state('zoomed') triggers) is *supposed* to already respect this
    automatically, but doing it explicitly avoids relying on that: a window
    that's SYSTEM (not per-monitor) DPI-aware, as this process declares
    itself (see _make_windows_dpi_aware), can end up with 'zoomed' computed
    against stale or wrong-monitor metrics, letting the bottom of the window
    land under the taskbar. Querying the monitor's rcWork rectangle directly
    via MonitorFromWindow/GetMonitorInfoW and setting geometry to exactly
    that is unambiguous regardless of DPI-awareness quirks, and — using
    MonitorFromWindow rather than the primary screen's metrics — correct on
    whichever monitor the window actually ends up on, not just the primary
    one."""
    if sys.platform != 'win32':
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class MONITORINFO(ctypes.Structure):
            _fields_ = [
                ('cbSize', wintypes.DWORD),
                ('rcMonitor', wintypes.RECT),
                ('rcWork', wintypes.RECT),
                ('dwFlags', wintypes.DWORD),
            ]

        user32 = ctypes.windll.user32
        # HMONITOR is a pointer-sized handle — on 64-bit Windows that's 8
        # bytes. ctypes defaults an undeclared restype to c_int (4 bytes,
        # signed), which silently truncates/corrupts a real 64-bit handle
        # value rather than raising — GetMonitorInfoW then either fails
        # outright on the corrupted handle, or (worse) succeeds against
        # whatever monitor that garbage handle happens to coincidentally
        # resolve to, producing a plausible-looking but wrong rectangle.
        # Declaring these explicitly is what makes the handle round-trip
        # correctly.
        user32.MonitorFromWindow.restype = ctypes.c_void_p
        user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        user32.GetMonitorInfoW.restype = wintypes.BOOL
        user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(MONITORINFO)]

        hwnd = window.winfo_id()
        MONITOR_DEFAULTTONEAREST = 2
        monitor = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
        if not monitor:
            return None

        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return None
        r = info.rcWork
        left, top, right, bottom = r.left, r.top, r.right, r.bottom
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            return None
        return left, top, width, height
    except Exception:
        return None


def maximize_figure_window(fig):
    """Best-effort: maximize a figure's window to fill the current monitor's
    work area (screen bounds minus the taskbar) — explicitly, via
    _windows_monitor_work_area, rather than Windows' own 'zoomed' window
    state, which is supposed to already exclude the taskbar but doesn't
    reliably here (see that function's docstring). Falls back to 'zoomed'
    (or Tk's '-zoomed' attribute, for X11/some window managers) if the work
    area can't be determined — e.g. non-Windows, or the ctypes calls fail.

    Maximizing programmatically like this doesn't reliably deliver a
    correctly-sized resize_event to matplotlib the way an interactive drag-
    resize does — see on_configure in prompt_subregion_selection, which
    this relies on the caller having already wired up (via the toplevel's
    own <Configure> binding, not just matplotlib's resize_event) to actually
    pick up the resulting size once Tk finishes applying it."""
    try:
        window = fig.canvas.manager.window
    except AttributeError:
        return
    work_area = _windows_monitor_work_area(window)
    if work_area is not None:
        left, top, width, height = work_area
        try:
            # Pass 1: content-sized to the *entire* work area — this alone
            # overshoots the window's true *outer* (decorated) bounds past
            # the work area's bottom edge, under the taskbar, by however
            # tall the title bar is: Tk's own docs are explicit that wm
            # geometry's WxH excludes window decorations (border/title bar)
            # entirely, unlike most other toolkits' "maximize", which sizes
            # the outer window to fit the work area. This is exactly what
            # let the button row end up under the taskbar even though the
            # work area itself (confirmed via _windows_monitor_work_area)
            # was already correct — real OS maximize doesn't have this
            # problem since it operates on the outer window directly.
            window.geometry(f"{width}x{height}+{left}+{top}")
            # Needs the window actually realized before its decoration size
            # can be measured (pass 2, below) — deiconify+update_idletasks
            # mirrors center_figure_window's own two-pass approach, just
            # done eagerly here instead of deferred via after_idle, since
            # this function is expected to complete synchronously.
            window.deiconify()
            window.update_idletasks()
            # Pass 2: same trick as center_figure_window's own two-pass
            # centering — winfo_rootx()/winfo_rooty() give the *content*
            # area's screen position, winfo_x()/winfo_y() give the *outer*
            # (decorated) window's, so the gap between them is the real
            # title bar/border chrome. Shrinking the requested content size
            # by that amount is what keeps the window's true outer bounds
            # within the work area instead of extending past it.
            chrome_w = max(0, window.winfo_rootx() - window.winfo_x()) * 2
            chrome_h = max(0, window.winfo_rooty() - window.winfo_y())
            if chrome_w > 0 or chrome_h > 0:
                adj_width = max(1, width - chrome_w)
                adj_height = max(1, height - chrome_h)
                window.geometry(f"{adj_width}x{adj_height}+{left}+{top}")
            return
        except Exception:
            pass
    try:
        window.state('zoomed')
    except Exception:
        try:
            window.attributes('-zoomed', True)
        except Exception:
            pass


def assert_figure_content_size(fig, fig_w_in, fig_h_in):
    """Best-effort: explicitly (re)assert the *canvas widget's* own pixel
    size, computed from (fig_w_in, fig_h_in) at the figure's own dpi — same
    reasoning as maximize_figure_window (whose own target is instead the
    monitor's work area, not a specific figsize): normalize_tk_scaling's fix
    for Tk's own DPI-derived auto-scaling only reliably takes effect once
    the window is actually realized, which otherwise happens lazily
    whenever show_figure_blocking() finally maps it — deiconifying/
    realizing it eagerly here instead (same early timing normalize_tk_
    scaling itself already requires) and reasserting the intended size right
    after catches and corrects any residual mismatch, instead of leaving the
    window to visibly snap to a different (usually smaller) size right
    after it first appears, with no later maximize_figure_window call
    around to paper over it the way the other pickers' own windows get.

    Deliberately resizes the *canvas* (via its own Tk widget's width/height
    options — the same thing FigureCanvasTkAgg sets once at construction)
    rather than calling `window.geometry("WxH")` directly on the toplevel:
    an explicit `wm geometry WxH` call turns off Tk's automatic geometry
    propagation for that window, so if the canvas subsequently resized
    itself again on its own (exactly the behavior this is working around),
    the *window* would no longer follow it — leaving the window pinned at
    this call's size while the canvas shrinks inside it, i.e. a visible gap
    below/right of the actual content. Resizing the canvas widget directly
    leaves propagation on, so the (packed, fill+expand) toplevel keeps
    tracking the canvas's true size the same way it already does for every
    later resize, including whatever internal one this is meant to catch."""
    try:
        canvas_widget = fig.canvas.get_tk_widget()
        dpi = fig.dpi
        w = max(1, int(round(fig_w_in * dpi)))
        h = max(1, int(round(fig_h_in * dpi)))
        canvas_widget.configure(width=w, height=h)
        window = fig.canvas.manager.window
        window.deiconify()
        window.update_idletasks()
    except Exception:
        pass


def _centered_position(area_left, area_top, area_w, area_h, win_w, win_h,
                        margin_px=WINDOW_EDGE_MARGIN_PX):
    """Top-left corner that centers a win_w x win_h window in the given work
    area, clamped so it never extends past either edge.

    The clamp is the point: a plain centered `area_left + (area_w - win_w)
    // 2` goes *negative* for a window wider than the work area, and the
    previous max(0, ...) guard only stopped it from running off the left —
    the overflow simply moved to the right edge instead, which is what put
    the right side of the UMAP window off-screen. Here an oversized window
    is pinned to the margin on the top/left, so at least its own controls
    and the resize border on that side stay reachable."""
    x = area_left + (area_w - win_w) // 2
    y = area_top + (area_h - win_h) // 2
    x = max(area_left + margin_px, min(x, area_left + area_w - win_w - margin_px))
    y = max(area_top + margin_px, min(y, area_top + area_h - win_h - margin_px))
    return int(x), int(y)


def fit_figure_window_to_work_area(fig, margin_px=WINDOW_EDGE_MARGIN_PX):
    """Shrink `fig` if its window wouldn't fit on the monitor it actually
    opened on, leaving at least `margin_px` of bare desktop on every side so
    the window border stays grabbable for resizing.

    compute_figsize_for_screen_height (which picked this figure's size
    before it existed) can only consult a throwaway Tk root's
    winfo_screenwidth()/height() — the *primary* monitor's full bounds,
    taskbar included. That's the wrong number in two common cases: the
    window opens on a different, smaller monitor, or the taskbar/DPI
    scaling eats enough that the nominally-fitting size doesn't. Either way
    the result was a window whose right edge ran off the screen. Once the
    window really exists, MonitorFromWindow gives the true work area of the
    monitor it's on, so the size can be corrected before anything is laid
    out against it.

    Aspect ratio is deliberately *not* preserved: each dimension is clamped
    independently, since the goal is to fit the available desktop, and this
    window's own layout is fully responsive to whatever size it ends up
    with (see its on_figure_resize/apply_region_layout)."""
    try:
        window = fig.canvas.manager.window
        work_area = _windows_monitor_work_area(window)
        if work_area is not None:
            _area_left, _area_top, area_w, area_h = work_area
        else:
            area_w, area_h = window.winfo_screenwidth(), window.winfo_screenheight()
        dpi = fig.dpi
        # Budget is for the *outer* (decorated) window, so the chrome has to
        # come out before comparing against the figure's own content size.
        max_content_w = area_w - 2 * margin_px - WINDOW_CHROME_WIDTH_PX
        max_content_h = area_h - 2 * margin_px - WINDOW_CHROME_HEIGHT_PX
        cur_w_in, cur_h_in = fig.get_size_inches()
        new_w_in = min(cur_w_in, max_content_w / dpi)
        new_h_in = min(cur_h_in, max_content_h / dpi)
        if new_w_in <= 0 or new_h_in <= 0:
            return  # implausible work-area reading — leave the figure alone
        if new_w_in < cur_w_in or new_h_in < cur_h_in:
            fig.set_size_inches(new_w_in, new_h_in)
            assert_figure_content_size(fig, new_w_in, new_h_in)
    except Exception:
        pass


def center_figure_window(fig):
    """Best-effort: center a figure's window on screen (position only,
    doesn't touch its size). Positions in two passes.

    Pass 1 (immediate): uses the figure's own size in inches (known
    precisely — we just computed it via compute_figsize_for_screen_height)
    converted to pixels via its DPI, plus the flat WINDOW_CHROME_* guess for
    title bar/border overhead, rather than querying the Tk window's
    winfo_width()/winfo_height(): those can still reflect a stale,
    not-yet-realized size right after creation, which previously centered
    large windows as if they were tiny — anchoring near the screen's center
    point and then letting the window's real, much larger size overflow
    past it toward the bottom right. This pass just gets the window roughly
    in place immediately, instead of at Tk's default spawn point, while it
    is being realized.

    Pass 2 (deferred, via after_idle): WINDOW_CHROME_* is a single flat
    estimate and doesn't scale with OS DPI scaling, so on a scaled display
    the real chrome can be taller than assumed, leaving the window visibly
    off-center (low). Once Tk has actually mapped the window, its true
    decorated size can be measured instead of guessed: winfo_rootx()/
    winfo_rooty() give the screen position of the window's *content* area,
    while winfo_x()/winfo_y() give the screen position of the *outer*
    window (as tracked by Tk) — the gap between them is the real title
    bar/border chrome, a standard Tkinter trick for recovering WM
    decoration size. This has to be deferred (scheduled instead of read
    immediately) because right after creation those values are still stale
    placeholders, same reason pass 1 doesn't use them directly — after_idle
    guarantees Tk has processed the map/configure events first. Both
    show_figure_blocking()'s and show_processing_dialog()'s event pumping
    (flush_events()/update()) give this callback a chance to run.

    Both passes center within the monitor's *work area* (via
    _windows_monitor_work_area — same query maximize_figure_window uses),
    not its full screen bounds: centering in the full screen height put the
    window's midpoint below the work area's actual visual center by half
    the taskbar's height, since the taskbar eats into the bottom (or
    whichever edge it's docked to) without the centering math accounting
    for it. Falls back to winfo_screenwidth()/height() (full screen, taskbar
    included) if the work-area query fails — e.g. non-Windows."""
    try:
        window = fig.canvas.manager.window
        fig_w_in, fig_h_in = fig.get_size_inches()
        dpi = fig.dpi
        w = int(fig_w_in * dpi) + WINDOW_CHROME_WIDTH_PX
        h = int(fig_h_in * dpi) + WINDOW_CHROME_HEIGHT_PX
        work_area = _windows_monitor_work_area(window)
        if work_area is not None:
            area_left, area_top, screen_w, screen_h = work_area
        else:
            area_left, area_top = 0, 0
            screen_w, screen_h = window.winfo_screenwidth(), window.winfo_screenheight()
        x, y = _centered_position(area_left, area_top, screen_w, screen_h, w, h)
        window.geometry(f"+{x}+{y}")

        def _recenter_with_real_geometry():
            try:
                window.update_idletasks()
                content_w, content_h = window.winfo_width(), window.winfo_height()
                if content_w <= 1 or content_h <= 1:
                    return  # window still not realized; keep the pass-1 estimate
                chrome_w = max(0, window.winfo_rootx() - window.winfo_x()) * 2
                chrome_h = max(0, window.winfo_rooty() - window.winfo_y())
                real_w = content_w + chrome_w
                real_h = content_h + chrome_h
                # Shrink-to-fit, from *measured* geometry rather than the
                # predicted size fit_figure_window_to_work_area works from.
                # That prediction multiplies figsize by fig.dpi and adds the
                # flat WINDOW_CHROME_* estimate — both of which can be wrong
                # (Tk's own DPI-derived scaling, real WM decoration size), and
                # when they are, a window that "fits" on paper still opens
                # with its right edge past the screen. Here the true decorated
                # size is already known, so any overflow is exact and can be
                # taken straight back off the canvas.
                over_w = max(0, real_w - (screen_w - 2 * WINDOW_EDGE_MARGIN_PX))
                over_h = max(0, real_h - (screen_h - 2 * WINDOW_EDGE_MARGIN_PX))
                if over_w > 0 or over_h > 0:
                    new_content_w = max(1, content_w - over_w)
                    new_content_h = max(1, content_h - over_h)
                    # Canvas widget (not window.geometry) for the same reason
                    # assert_figure_content_size documents: an explicit
                    # `wm geometry WxH` turns off Tk's geometry propagation,
                    # leaving the window pinned while the canvas resizes
                    # inside it. set_size_inches keeps matplotlib's own idea
                    # of the size in step, so any resize_event handler
                    # relayouts against the corrected size.
                    fig.set_size_inches(new_content_w / fig.dpi, new_content_h / fig.dpi)
                    fig.canvas.get_tk_widget().configure(width=new_content_w, height=new_content_h)
                    window.update_idletasks()
                    real_w = new_content_w + chrome_w
                    real_h = new_content_h + chrome_h
                x2, y2 = _centered_position(area_left, area_top, screen_w, screen_h, real_w, real_h)
                window.geometry(f"+{x2}+{y2}")
            except Exception:
                pass

        window.after_idle(_recenter_with_real_geometry)
    except Exception:
        pass


def show_figure_blocking(fig):
    """Show `fig` and block until *that specific figure* is closed — not
    plt.show()'s usual behavior of blocking until every open figure is
    closed. This matters whenever a figure is opened while another one
    (e.g. the section grid picker) is still open: with plain plt.show(),
    closing the nested window wouldn't return control, since the call is
    still waiting on the outer figure too — the nested window's buttons
    would appear to do nothing (they do work; the call just never returns)."""
    raise_figure_window(fig)
    fig.canvas.manager.show()
    fig.canvas.draw_idle()
    reraise_timer_id = None
    tk_widget = None
    try:
        # A window manager's own re-stacking in response to another window
        # closing right as this one opens (e.g. a "Processing..." dialog
        # being dismissed just before this call) can lag behind that close
        # by a moment — landing *after* the raise_figure_window() call
        # above and silently undoing it, which read as this window
        # intermittently dropping behind another one shortly after
        # appearing. Raising again a beat later catches that.
        tk_widget = fig.canvas.get_tk_widget()
        reraise_timer_id = tk_widget.after(200, lambda: raise_figure_window(fig))
    except Exception:
        pass
    while plt.fignum_exists(fig.number):
        fig.canvas.flush_events()
        time.sleep(0.02)
    if reraise_timer_id is not None:
        # Without this, closing the window within that 200ms window left
        # this timer to fire after Tk had already destroyed it — not
        # something raise_figure_window's own try/except can catch, since
        # the failure happens in Tcl's own "after" dispatch, before any of
        # our Python code (including that except) gets a chance to run:
        # 'invalid command name "...<lambda>" ("after" script)'.
        try:
            tk_widget.after_cancel(reraise_timer_id)
        except Exception:
            pass


def show_processing_dialog(message):
    """Non-blocking 'Processing...' popup with a progress bar and Cancel
    button, for covering genuinely slow work (not just asking permission up
    front, which just makes the user wait through two delays instead of
    one). Does NOT start its own blocking event loop — the caller must
    periodically call fig.canvas.flush_events() (e.g. while polling a
    worker thread) to keep the window responsive and let a Cancel click
    actually register, and is responsible for calling plt.close(fig) when
    the work finishes or is cancelled.

    Returns (fig, cancel_flag, set_progress): cancel_flag['cancelled']
    becomes True once Cancel is clicked; set_progress(fraction) fills the
    bar to `fraction` (0.0-1.0, clamped) — call it from the same place
    that's already calling flush_events() in a loop, since set_progress
    itself only updates the bar's geometry (via draw_idle()) rather than
    forcing a repaint. set_progress(None) resets the bar to empty, for
    work whose progress isn't (yet) known."""
    # Sized to the message: wrapped here (by an estimate of how many characters
    # fit per line) and the window made tall enough for the resulting lines,
    # instead of a fixed size that let longer messages spill over the top edge
    # and into the progress bar. Everything is laid out in inches from the
    # bottom up: Cancel button, progress bar, then the text.
    import textwrap
    fig_w_in = 6.5
    line_h_in = UI_BUTTON_FONTSIZE * 1.35 / 72
    chars_per_line = max(20, int(fig_w_in * 0.9 * 72 / (UI_BUTTON_FONTSIZE * 0.55)))
    lines = [wrapped for paragraph in message.split('\n')
             for wrapped in (textwrap.wrap(paragraph, chars_per_line) or [''])]
    button_bottom_in, button_h_in = 0.15, 0.45
    bar_bottom_in, bar_h_in = button_bottom_in + button_h_in + 0.3, 0.26
    text_bottom_in = bar_bottom_in + bar_h_in + 0.2
    fig_h_in = text_bottom_in + len(lines) * line_h_in + 0.25

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    ax.set_position([0, 0, 1, 1])
    ax.axis('off')
    ax.text(0.5, (text_bottom_in + len(lines) * line_h_in / 2) / fig_h_in, '\n'.join(lines),
            ha='center', va='center', fontsize=UI_BUTTON_FONTSIZE, linespacing=1.2,
            transform=fig.transFigure)

    progress_ax = fig.add_axes([0.1, bar_bottom_in / fig_h_in, 0.8, bar_h_in / fig_h_in])
    progress_ax.set_xlim(0, 1)
    progress_ax.set_ylim(0, 1)
    progress_ax.axis('off')
    progress_ax.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='gray', linewidth=1))
    progress_fill = Rectangle((0, 0), 0, 1, facecolor='steelblue', edgecolor='none')
    progress_ax.add_patch(progress_fill)

    def set_progress(fraction):
        progress_fill.set_width(0.0 if fraction is None else max(0.0, min(1.0, fraction)))
        fig.canvas.draw_idle()

    cancel_flag = {'cancelled': False}

    def on_cancel(event):
        cancel_flag['cancelled'] = True

    cancel_ax = fig.add_axes([0.35, button_bottom_in / fig_h_in, 0.3, button_h_in / fig_h_in])
    cancel_button = Button(cancel_ax, 'Cancel')
    cancel_button.label.set_fontsize(UI_BUTTON_FONTSIZE)
    cancel_button.on_clicked(on_cancel)
    # Keep the Button alive beyond this function's scope (see the note on
    # this same bug elsewhere in this file) by hanging it off the figure,
    # which the caller holds a reference to until it closes it.
    fig._cancel_button_ref = cancel_button

    fig.canvas.manager.set_window_title("Processing")
    center_figure_window(fig)
    raise_figure_window(fig)
    fig.canvas.manager.show()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    return fig, cancel_flag, set_progress


def map_data_to_pixel(section_layout, x, y):
    """Map a data-space (x, y) point for a section into pixel coordinates
    within its thumbnail box in the cached grid image, using that section's
    stored pixel bounding box and the data x/y limits its subplot was
    plotted with. Works regardless of axis direction (the y axis is
    inverted) since it's pure proportional interpolation, not dependent on
    which of each limit pair is numerically larger."""
    px_x0, px_x1, px_y0, px_y1 = section_layout['pixel_bbox']
    dx0, dx1 = section_layout['data_xlim']
    dy0, dy1 = section_layout['data_ylim']
    frac_x = 0.5 if dx1 == dx0 else (x - dx0) / (dx1 - dx0)
    frac_y = 0.5 if dy1 == dy0 else (y - dy0) / (dy1 - dy0)
    return px_x0 + frac_x * (px_x1 - px_x0), px_y0 + frac_y * (px_y1 - px_y0)


def map_pixel_to_data(section_layout, px, py):
    """Inverse of map_data_to_pixel(): given pixel coordinates within a
    section's box, recover the original data-space (x, y)."""
    px_x0, px_x1, px_y0, px_y1 = section_layout['pixel_bbox']
    dx0, dx1 = section_layout['data_xlim']
    dy0, dy1 = section_layout['data_ylim']
    frac_x = 0.5 if px_x1 == px_x0 else (px - px_x0) / (px_x1 - px_x0)
    frac_y = 0.5 if px_y1 == px_y0 else (py - px_y0) / (px_y1 - px_y0)
    return dx0 + frac_x * (dx1 - dx0), dy0 + frac_y * (dy1 - dy0)


def section_roi_cache_paths(section_label):
    """Per-section cached background image + its data x/y limits, so
    reopening the ROI picker for the same section is fast the second time
    (skips reloading spatial data and re-scattering every point)."""
    token = sanitize_section_token(section_label)
    stem = f'{DATASET_NAME}_section_roi_{token}'
    return CACHE_DIR / f'{stem}.png', CACHE_DIR / f'{stem}.json'


def gene_expression_cache_paths(section_label, resolved_gene, cell_type_selection='All', color_scale='log',
                                 dataset='standard'):
    """Per-section, per-gene cached expression-color PNG + its max
    expression value, so re-showing a gene already rendered — even in an
    earlier session — is instant instead of re-pulling and re-rasterizing.
    `resolved_gene` should be the canonical-case gene symbol (as returned by
    find_gene_index()), not the raw user-typed query, so differently-cased
    queries for the same gene share one cache entry.

    `cell_type_selection` is part of the key too — 'Neurons'/'NonNeurons'
    grey out the excluded type (see render_gene_expression_array), so a
    cached image from one selection would show the wrong thing reused under
    another. `color_scale` ('linear'/'log') likewise — they're colored
    differently, not just relabeled. `dataset` ('standard'/'imputed')
    likewise — the imputed panel's expression values for a given gene name
    aren't the same numbers as the standard panel's, so they can't share a
    cache entry; left out of the filename for 'standard' so this doesn't
    invalidate every cache built before the imputed dataset existed."""
    section_token = sanitize_section_token(section_label)
    gene_token = re.sub(r'[^A-Za-z0-9]+', '', str(resolved_gene))
    dataset_token = '' if dataset == 'standard' else f'_{dataset}'
    stem = (
        f'{DATASET_NAME}_gene_expr_{section_token}_{gene_token}_'
        f'{cell_type_selection.lower()}_{color_scale}{dataset_token}'
    )
    return CACHE_DIR / f'{stem}.png', CACHE_DIR / f'{stem}.json'


def load_valid_section_roi_cache(section_label):
    """Return the cached layout dict for section_label if a valid,
    current-format cache exists on disk, else None (this covers both 'no
    cache yet' and 'cache exists but is an older format missing pixel_bbox,
    needs regenerating'). Used by both handle_double_click() — to decide
    whether the slow regeneration path (and its processing dialog) is about
    to run — and prompt_subregion_selection() — to decide whether to trust
    it. Checking file existence alone isn't enough: an old-format cache
    passes that check but still needs regenerating, which previously meant
    the processing dialog got skipped for a section that was, in fact,
    about to take several seconds to redraw."""
    cache_png, cache_json = section_roi_cache_paths(section_label)
    if not (cache_png.exists() and cache_json.exists()):
        return None
    try:
        with open(cache_json) as f:
            layout = json.load(f)
    except Exception:
        return None
    if 'pixel_bbox' not in layout:
        return None
    return layout


class UserCancelledSelection(Exception):
    """Raised when the user explicitly cancels a top-level GUI picker
    (closes the window, clicks Cancel/Exit, declines a whole-brain
    confirmation) — distinct from the window failing to open/run at all
    (a real error), so the main script's session loop can return to the
    startup panel for another session instead of falling back to a
    console prompt, which is what any *other* exception from these same
    call sites means."""


class _RenderCancelled(Exception):
    """Raised by handle_double_click's progress_callback (from inside
    generate_and_cache_section_image, between rendering stages) to unwind
    out early once the user clicks Cancel."""


def format_scalebar_length(length_um):
    """'500 \u00b5m' below 1 mm, otherwise 'N mm' (e.g. '2 mm', '1.5 mm')."""
    if length_um < 1000:
        return f'{length_um:g} \u00b5m'
    return f'{length_um / 1000:g} mm'


def nice_scalebar_length_um(view_width_mm, target_fraction=SCALEBAR_TARGET_FRACTION,
                             nice_lengths_um=SCALEBAR_NICE_LENGTHS_UM):
    """The largest length (in micrometers, from `nice_lengths_um`) that's no
    more than `target_fraction` of a view this wide (in millimeters — the
    ABC atlas's own native data units; see HOVER_CELL_RADIUS_DATA_UNITS's
    comment elsewhere in this file). Used for a scale bar that reads a
    clean, round physical length at any zoom level: it shrinks as the view
    narrows and grows as it widens, always snapping to one of these fixed
    lengths, so it can never overflow the view or shrink away to nothing."""
    target_um = view_width_mm * 1000 * target_fraction
    candidates = [um for um in nice_lengths_um if um <= target_um]
    return candidates[-1] if candidates else nice_lengths_um[0]


def compute_scalebar_geometry(ax, margin_fraction=SCALEBAR_MARGIN_FRACTION,
                               text_gap_fraction=SCALEBAR_TEXT_GAP_FRACTION):
    """Where a scale bar belongs in `ax`'s lower-left corner right now: a
    dict with the bar's two endpoints, the label's position, and its text —
    all in `ax`'s own data coordinates, so the result is a real vector
    object under savefig too, not a fixed-pixel overlay drawn after the
    fact. Positioned via transAxes -> transData (screen-relative margins,
    not a data-space offset) rather than assuming which data direction is
    'right'/'up' on screen, so this works whether or not the axes' y-axis
    is inverted — e.g. every section panel's tissue-orientation flip (see
    invert_yaxis() elsewhere in this file). Returns None if the axes has no
    usable width yet (e.g. mid-setup, before its limits are set)."""
    x0, x1 = ax.get_xlim()
    view_width_mm = abs(x1 - x0)
    if not np.isfinite(view_width_mm) or view_width_mm <= 0:
        return None
    length_um = nice_scalebar_length_um(view_width_mm)
    length_mm = length_um / 1000
    inv = ax.transData.inverted()
    corner_x, corner_y = inv.transform(ax.transAxes.transform((margin_fraction, margin_fraction)))
    # A second point just to the right of the corner in axes-fraction terms
    # (screen-right, always) — comparing its data-x against corner_x gives
    # the sign that means 'rightward' in *this* axes' own data coordinates,
    # without assuming x is never inverted.
    probe_x, _probe_y = inv.transform(ax.transAxes.transform((margin_fraction + 0.01, margin_fraction)))
    x_sign = 1 if probe_x >= corner_x else -1
    end_x = corner_x + x_sign * length_mm
    text_y = inv.transform(ax.transAxes.transform((margin_fraction, margin_fraction + text_gap_fraction)))[1]
    return {
        'bar_x': (corner_x, end_x), 'bar_y': (corner_y, corner_y),
        'text_xy': (corner_x + x_sign * length_mm / 2, text_y),
        'label': format_scalebar_length(length_um),
    }


def apply_scalebar_geometry(line, text, geometry):
    """Push `geometry` (see compute_scalebar_geometry; None hides the bar)
    onto an existing scale bar's Line2D/Text — the update half of
    build_section_scalebar, for a bar that has to keep tracking a live,
    still-changing view rather than being drawn once for a static export."""
    if geometry is None:
        line.set_visible(False)
        text.set_visible(False)
        return
    line.set_visible(True)
    text.set_visible(True)
    line.set_data(geometry['bar_x'], geometry['bar_y'])
    text.set_position(geometry['text_xy'])
    text.set_text(geometry['label'])


def build_section_scalebar(ax, color='white', fontsize=8, geometry=None, zorder=6, animated=False):
    """Add a scale bar (a Line2D + Text, both real vector artists in `ax`'s
    own data coordinates — not a fixed-pixel overlay) to `ax`, sized and
    positioned from `geometry` (see compute_scalebar_geometry), or freshly
    computed from `ax`'s current view if omitted. Returns (line, text): a
    caller that will keep tracking a live, changing view hangs onto them
    and re-applies new geometry via apply_scalebar_geometry; a one-off
    static export can just discard the return value.

    animated=True (the interactive viewer's own live scale bar; left False
    — the default — for a one-off static export, which has no zoom-preview
    bitmaps or cached images to sit on top of) marks both artists animated
    and gives them a very high zorder, so draw_animated_overlays can
    re-stamp them on top of *any* current panel content — a live scatter,
    a zoom-preview bitmap, or a cached home-view PNG — instead of the bar
    only ever being correct when the panel happens to be showing its real,
    freshly-drawn scatter. Without this, whatever the bar looked like at
    the moment a bitmap snapshot was taken got baked into that bitmap's
    own pixels, stretching/shifting as the bitmap was zoomed, and
    sometimes doubling up with the live bar (see snapshot_axes_region's
    own note on why it hides this pair before capturing)."""
    if geometry is None:
        geometry = compute_scalebar_geometry(ax)
    line = Line2D([0, 0], [0, 0], color=color, linewidth=2.5, solid_capstyle='butt', zorder=zorder)
    ax.add_line(line)
    text = ax.text(0, 0, '', color=color, ha='center', va='bottom', fontsize=fontsize, zorder=zorder)
    if animated:
        line.set_animated(True)
        text.set_animated(True)
    apply_scalebar_geometry(line, text, geometry)
    return line, text


def visible_points_only(offsets, facecolors, sizes, xlim, ylim):
    """`(offsets, facecolors, sizes)` filtered down to just the points inside
    `xlim`/`ylim` (each may be given in either order — a decreasing ylim,
    e.g. every section panel's own inverted y-axis, is handled the same as
    an increasing one). `facecolors`/`sizes` are matplotlib scatter's own
    get_facecolors()/get_sizes() — each is either one row that broadcasts to
    every point (left untouched) or one row per point (filtered the same as
    `offsets`).

    Exists for render_section_maps_export_figure: a background_artist
    always holds a *whole section's* points — many more than are inside the
    current, possibly zoomed-in view — and relying on the new axes' own
    clip-path to hide the rest at render/save time turned out not to be
    enough. It works in every renderer this was actually drawn in (Agg for
    the PNG, a browser for the SVG), but at least one common SVG viewer the
    result also gets opened in doesn't apply a matplotlib clip-path to a
    large scatter's individual points, which showed as this panel's own
    off-screen cells bleeding into whatever sits next to it in the grid.
    Filtering the *data* here removes the dependence on any renderer
    honoring the clip at all, and, incidentally, makes for a much smaller
    file than shipping a whole section's worth of points."""
    x0, x1 = sorted(xlim)
    y0, y1 = sorted(ylim)
    visible = ((offsets[:, 0] >= x0) & (offsets[:, 0] <= x1)
               & (offsets[:, 1] >= y0) & (offsets[:, 1] <= y1))
    facecolors = facecolors if len(facecolors) == 1 else facecolors[visible]
    sizes = sizes if len(sizes) == 1 else sizes[visible]
    return offsets[visible], facecolors, sizes


def unique_export_paths(target_dir, stem, extensions):
    """{ext: path} for a new export named `stem` (no leading '.' on the
    extensions), without overwriting an earlier one: the first save uses
    `stem` bare; every save after that appends the lowest unused '_N'
    suffix, checked across every extension in `extensions` together so a
    PNG and SVG saved in the same call always share one suffix. A suffix
    the user appended themselves after the number (e.g. renaming a copy to
    '..._2_final.png') doesn't confuse the numbering — only the digits
    immediately after '_N' are read; anything past that is ignored."""
    base_exists = any((target_dir / f'{stem}{ext}').exists() for ext in extensions)
    if not base_exists:
        return {ext: target_dir / f'{stem}{ext}' for ext in extensions}
    pattern = re.compile(rf'^{re.escape(stem)}_(\d+)(?:_.*)?$')
    highest = 0
    for ext in extensions:
        for existing in target_dir.glob(f'{stem}_*{ext}'):
            m = pattern.match(existing.stem)
            if m:
                highest = max(highest, int(m.group(1)))
    n = highest + 1
    return {ext: target_dir / f'{stem}_{n}{ext}' for ext in extensions}


def generate_and_cache_section_image(adata, abc_cache, section_series, section_label, spatial,
                                      figsize, progress_callback=None):
    """Render `section_label`'s background scatter (points only — no ROI
    overlays; those get drawn fresh every time by prompt_subregion_selection)
    to its cached PNG/JSON (see section_roi_cache_paths()).

    `figsize` must be computed by the caller (e.g. via
    compute_figsize_for_screen_height()) and passed in rather than computed
    in here, since that function briefly creates its own tk.Tk() root to
    query the screen — harmless on its own, but see the note on threading
    below.

    Uses a plain, off-screen Agg-backed Figure (matplotlib.figure.Figure +
    FigureCanvasAgg) rather than pyplot's plt.subplots(), which ties into
    whatever interactive backend (TkAgg here) is currently active. This one
    never creates a window or talks to Tkinter at all. That was originally
    meant to make it safe to call from a background thread — it turned out
    not to be enough: even Tk-free work running concurrently with another
    thread's flush_events() calls still produced "main thread is not in
    main loop" errors, apparently from Tcl/Tk's own reentrancy being more
    fragile than "don't touch Tk from the other thread" accounts for. So
    this now expects to be called synchronously, from the *same* thread
    that's driving the event loop — see `progress_callback` below, which is
    how that caller stays responsive instead.

    `progress_callback`, if given, is called as `progress_callback(fraction)`
    after each of this function's four main steps (computing the section's
    x/y coordinates, plotting the scatter, saving the figure, and reading it
    back to record its layout) — coarse compared to a true percentage (none
    of these steps has a natural finer-grained equivalent), but still gives
    the caller a chance to pump the event loop and check for cancellation
    between stages, instead of a bar (and an unresponsive Cancel button)
    frozen for however long the whole thing takes. May raise whatever
    `progress_callback` raises (e.g. to signal the user cancelled).

    Raises RuntimeError if spatial x/y coordinates aren't available, or if
    no cells fall in this section."""
    def report(fraction):
        if progress_callback is not None:
            progress_callback(fraction)

    if spatial is None or 'x' not in spatial.columns or 'y' not in spatial.columns:
        raise RuntimeError("spatial x/y coordinates not available")

    mask = (section_series == section_label).to_numpy()
    # Filter down to this section's rows *before* any per-row work, not
    # after — extract_leading_numeric_id() in particular runs a regex
    # extract, which (unlike pd.to_numeric) isn't vectorized in C, so
    # running it across all of `spatial` (every cell in the dataset) before
    # masking down to the ~thousands belonging to this one section was by
    # far the most expensive part of this function.
    section_spatial = spatial[mask]
    report(0.1)

    xs = pd.to_numeric(section_spatial['x'], errors='coerce').to_numpy()
    ys = pd.to_numeric(section_spatial['y'], errors='coerce').to_numpy()
    report(0.3)
    class_ids = None
    if 'class' in section_spatial.columns:
        class_ids = extract_leading_numeric_id(section_spatial['class']).to_numpy()

    report(0.4)
    valid = ~(np.isnan(xs) | np.isnan(ys))
    report(0.5)
    xs, ys = xs[valid], ys[valid]
    report(0.6)
    if class_ids is not None:
        class_ids = class_ids[valid]
    if len(xs) == 0:
        raise RuntimeError(f"no spatial coordinates for section {section_label}")
    report(0.7)

    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)  # attaches itself as fig.canvas; never touches Tk
    ax = fig.add_subplot(111)
    # Same gray-non-neurons/colored-neurons scheme as the thumbnail grid.
    point_colors, neuron_mask = build_neuron_class_colors(class_ids)
    if neuron_mask is not None:
        scatter_gray_then_colored(ax, xs, ys, neuron_mask, point_colors, gray_size=3, colored_size=10)
    else:
        ax.scatter(xs, ys, s=3, c='steelblue', linewidths=0)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    # Without this, matplotlib's generous *default* margins (left≈0.125,
    # right≈0.9, top≈0.88, ...) get baked directly into the saved PNG's own
    # pixel data as blank white space around the plot — no amount of
    # adjusting the *viewer* window's axes/aspect afterward can remove that,
    # since by then it's just part of the image being displayed, not a
    # layout choice anymore. pixel_bbox (below) already correctly records
    # wherever the axes actually lands, so this isn't needed for
    # correctness — but leaving it at the default just means the saved
    # image is mostly blank margin around a small plot.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    report(0.8)

    cache_png, cache_json = section_roi_cache_paths(section_label)
    fig.savefig(cache_png, dpi=150)

    # Read the actual saved image back to get its exact pixel dimensions
    # (matching generate_section_grid_image's approach), then record where
    # the axes actually landed within it.
    saved_img = plt.imread(cache_png)
    img_h, img_w = saved_img.shape[0], saved_img.shape[1]
    frac = ax.get_position()  # figure-fraction Bbox, y-up
    pixel_bbox = [frac.x0 * img_w, frac.x1 * img_w, frac.y0 * img_h, frac.y1 * img_h]

    with open(cache_json, 'w') as f:
        json.dump({
            'pixel_bbox': pixel_bbox,
            'data_xlim': list(ax.get_xlim()),
            'data_ylim': list(ax.get_ylim()),
        }, f)
    print(f"Cached section image to {cache_png} for faster reloads.")
    report(1.0)


def section_home_cache_path(run_folder, section_label, level):
    """Cached PNG of one section panel's home-extent categorical coloring
    for the interactive viewer's own 'All <level>s' mode — see SECTION_
    HOME_CACHE_VERSION's own comment. Lives inside `run_folder` itself
    (not the shared CACHE_DIR the picker-stage/gene-expression caches
    above use) since, unlike those, the color *assignment* here depends
    on which cells are actually in this run (compute_ranked_category_
    colors ranks categories by count within the run's own selection) —
    a different ROI/cell-type selection could color the same section+
    level differently, so this can't be shared across runs the way a
    property of the raw dataset (like raw expression) can."""
    section_token = sanitize_section_token(section_label)
    return (Path(run_folder) / 'section_home_cache' /
            f'{DATASET_NAME}_v{SECTION_HOME_CACHE_VERSION}_{section_token}_{level}.png')


def render_section_home_view_png(xs, ys, colors, is_gray, home_xlim, home_ylim):
    """Off-screen render of one section panel's *complete* (unfiltered)
    background point set at its home extent, colored per `colors` (gray
    cells drawn first/underneath, same convention as apply_panel_colors_
    with_gray_behind — `is_gray` decides that draw order the same way).

    Deliberately does *not* invert the y-axis the way the live section
    panels do (see sec_ax.invert_yaxis()'s own comment on the atlas's y-
    increases-downward convention) — this produces a plain, naturally-
    oriented image (like the raw xs/ys, not pre-flipped), the same way a
    real scatter's own raw data isn't pre-flipped either; the *live*
    axes' own inversion is what correctly orients it once displayed via
    imshow (see show_section_home_cache_or_scatter). Inverting here too
    would flip it twice — this image would come out correctly oriented as
    its own standalone picture, but upside down once shown on the already-
    inverted live axes, which is exactly what happened before this
    comment existed.

    Uses a fresh, throwaway Figure+FigureCanvasAgg at a fixed size/DPI
    (SECTION_HOME_CACHE_DPI/_LONG_EDGE_IN) — entirely independent of the
    live interactive window, so the result looks identical regardless of
    which machine's screen happened to generate it (see section_home_
    cache_path's own docstring). Returns an RGBA uint8 array, ready for
    Image.fromarray(...).save(...) or direct imshow use."""
    x0, x1 = home_xlim
    y0, y1 = home_ylim
    width_data, height_data = abs(x1 - x0), abs(y1 - y0)
    if width_data <= 0 or height_data <= 0:
        figsize = (SECTION_HOME_CACHE_LONG_EDGE_IN, SECTION_HOME_CACHE_LONG_EDGE_IN)
    elif width_data >= height_data:
        figsize = (SECTION_HOME_CACHE_LONG_EDGE_IN, SECTION_HOME_CACHE_LONG_EDGE_IN * height_data / width_data)
    else:
        figsize = (SECTION_HOME_CACHE_LONG_EDGE_IN * width_data / height_data, SECTION_HOME_CACHE_LONG_EDGE_IN)
    fig = Figure(figsize=figsize, dpi=SECTION_HOME_CACHE_DPI)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(SECTION_PANEL_FACECOLOR)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(SECTION_PANEL_FACECOLOR)
    order = np.argsort(~np.asarray(is_gray), kind='stable')
    ax.scatter(xs[order], ys[order], c=np.asarray(colors, dtype=object)[order],
               s=SECTION_BACKGROUND_BASE_SIZE, alpha=SECTION_POINT_ALPHA, linewidths=0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


# Taxonomy levels a single-section window's Groups view can be colored by.
# 'class' is the image generate_and_cache_section_image() already produces;
# the others are rendered on demand by render_section_level_array().
SECTION_COLOR_LEVELS = ('class', 'subclass', 'supertype', 'cluster')


def section_level_image_cache_path(section_label, level):
    """Cached PNG of `section_label` colored by a non-class taxonomy level,
    next to (and sharing a name stem with) its class-colored image. The
    layout (JSON) is shared with the class image, so there's no JSON here."""
    cache_png, _cache_json = section_roi_cache_paths(section_label)
    return cache_png.with_name(f'{cache_png.stem}_{level}.png')


def render_section_level_array(section_series, section_label, spatial, level, cached_layout,
                                img_w, img_h, dpi=150, on_progress=None):
    """Render `section_label` colored by taxonomy `level` (subclass,
    supertype or cluster) as an RGBA uint8 array, drawn to match the
    class-colored image from generate_and_cache_section_image(): white
    background, non-neurons gray underneath, neurons on top colored by their
    ID at `level` (class_id_to_color, stable per ID, so a given subclass is
    the same color in every section).

    Placed with the class image's own layout rather than re-deriving one:
    the axes go exactly at cached_layout['pixel_bbox'] with its recorded data
    limits, the same technique render_gene_expression_array() uses, so the
    result can be swapped in for the class image pixel-for-pixel, and the
    ROI overlays and hover lookup (which use the same layout) stay aligned.

    Cached to disk (section_level_image_cache_path); a cached file whose size
    no longer matches img_w x img_h (the class image was regenerated at a
    different size) is ignored and re-rendered. `on_progress(fraction)`, if
    given, is called between stages. Raises ValueError if the level or
    spatial coordinates aren't available."""
    def report(fraction):
        if on_progress is not None:
            on_progress(fraction)

    cache_png = section_level_image_cache_path(section_label, level)
    if cache_png.exists():
        try:
            rgba = np.asarray(Image.open(cache_png).convert('RGBA'), dtype=np.uint8)
            if rgba.shape[:2] == (img_h, img_w):
                return rgba
        except Exception:
            pass  # unreadable; just re-render

    if spatial is None or 'x' not in spatial.columns or 'y' not in spatial.columns:
        raise ValueError("spatial x/y coordinates not available")
    if level not in spatial.columns or 'class' not in spatial.columns:
        raise ValueError(f"'{level}' labels not available in the cell metadata")
    report(0.0)

    # Filter to this section before any per-row work (see generate_and_cache_
    # section_image: the ID regex isn't vectorized, so run it on this
    # section's cells only).
    section_spatial = spatial[(section_series == section_label).to_numpy()]
    xs = pd.to_numeric(section_spatial['x'], errors='coerce').to_numpy()
    ys = pd.to_numeric(section_spatial['y'], errors='coerce').to_numpy()
    class_ids = extract_leading_numeric_id(section_spatial['class']).to_numpy()
    level_ids = extract_leading_numeric_id(section_spatial[level]).to_numpy()
    valid = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys, class_ids, level_ids = xs[valid], ys[valid], class_ids[valid], level_ids[valid]
    if len(xs) == 0:
        raise ValueError(f"no spatial coordinates for section {section_label}")
    report(0.3)

    point_colors, neuron_mask = build_neuron_class_colors(class_ids, color_ids_all=level_ids)
    fig = Figure(figsize=(img_w / dpi, img_h / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    px_x0, px_x1, px_y0, px_y1 = cached_layout['pixel_bbox']
    ax = fig.add_axes([px_x0 / img_w, px_y0 / img_h, (px_x1 - px_x0) / img_w, (px_y1 - px_y0) / img_h])
    # Same point sizes and layering as the class image.
    scatter_gray_then_colored(ax, xs, ys, neuron_mask, point_colors, gray_size=3, colored_size=10)
    ax.set_xlim(cached_layout['data_xlim'])
    ax.set_ylim(cached_layout['data_ylim'])
    ax.set_xticks([])
    ax.set_yticks([])
    report(0.6)
    fig.canvas.draw()  # rasterizing the scatter — the slow step for large sections
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    report(0.9)

    try:
        Image.fromarray(rgba, mode='RGBA').save(cache_png)
    except Exception as e:
        print(f"Warning: could not cache {level}-colored image for section {section_label}: {e}")
    report(1.0)
    return rgba


def prompt_load_rois_file(initial_dir=None):
    """Native 'Open File' dialog for picking a previously saved roi_coords.csv.
    Returns the chosen path (str) or None if cancelled/unavailable. Uses
    Tkinter directly (rather than a custom matplotlib dialog) since the
    interactive matplotlib backend in use is TkAgg in practice, so Tkinter
    is already available; falls back to printing an error if not.

    `initial_dir`, if given and it exists, is where the dialog opens —
    typically out_folder, the parent of the per-run subfolders that ROI
    CSVs actually get saved into (roi_coords.csv within each run's own
    'umap_...' subfolder), so the user still has to navigate one level
    down to the run they want."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        kwargs = {}
        if initial_dir and Path(initial_dir).is_dir():
            kwargs['initialdir'] = str(initial_dir)
        path = filedialog.askopenfilename(
            title="Select a previously saved ROI CSV",
            filetypes=[("ROI CSV files", "roi_coords.csv"), ("CSV files", "*.csv"), ("All files", "*.*")],
            **kwargs,
        )
        root.destroy()
        return path or None
    except Exception as e:
        print(f"Could not open file picker ({e}).")
        return None


def load_rois_csv(path):
    """Load a previously saved roi_coords.csv file. Returns (rois,
    whole_sections): rois is a list of ROI dicts ({'section','x_min',
    'x_max','y_min','y_max'}); whole_sections is a list of section labels
    for rows whose bounds were blank, meaning the entire section (not a
    sub-region) was selected — see the comment above the CSV-writing code
    in the main script. Raises ValueError if the file doesn't look like an
    ROI CSV."""
    df = pd.read_csv(path)
    required = {'section', 'x_min', 'x_max', 'y_min', 'y_max'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing expected column(s) {sorted(missing)}")
    df = df.copy()
    df['section'] = df['section'].astype(str)
    bounds_cols = ['x_min', 'x_max', 'y_min', 'y_max']
    is_whole_section = df[bounds_cols].isna().all(axis=1)
    rois = df.loc[~is_whole_section, ['section'] + bounds_cols].to_dict('records')
    whole_sections = df.loc[is_whole_section, 'section'].tolist()
    return rois, whole_sections


def roi_selection_unchanged(path, rois, whole_sections):
    """True if the roi_coords.csv at `path` already records exactly this
    session's `rois` and `whole_sections` (order-independent) — lets the
    caller skip re-prompting/rewriting roi_coords.csv and roi_map.png when
    nothing about the ROI/whole-section selection actually changed since
    they were last saved (e.g. reusing a cached run's exact selection).
    False if `path` doesn't exist or can't be parsed as an ROI CSV."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        existing_rois, existing_whole_sections = load_rois_csv(path)
    except Exception:
        return False

    def roi_key(roi):
        return (
            str(roi['section']),
            round(float(roi['x_min']), 6), round(float(roi['x_max']), 6),
            round(float(roi['y_min']), 6), round(float(roi['y_max']), 6),
        )

    try:
        rois_match = {roi_key(r) for r in rois} == {roi_key(r) for r in existing_rois}
    except (TypeError, ValueError):
        return False
    whole_sections_match = set(map(str, whole_sections)) == set(map(str, existing_whole_sections))
    return rois_match and whole_sections_match


def generate_section_grid_image(adata, abc_cache, section_series):
    """Render the per-section spatial thumbnail grid (neuron-only) to a PNG,
    plus a JSON file mapping each section label to its pixel bounding box in
    that PNG. This is the expensive, one-time step; prompt_section_selection_gui()
    loads the cached result instead of re-rendering on every run."""
    spatial = load_section_spatial_coords(adata, abc_cache)
    if spatial is None:
        raise RuntimeError("spatial x/y coordinates not available")

    counts = section_series.value_counts()
    unique_sections = sorted_sections_descending(counts.index.tolist())
    n = len(unique_sections)
    if n == 0:
        raise RuntimeError("no sections found")

    ncols = min(10, max(1, math.ceil(math.sqrt(n))))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.3, nrows * 1.5 + 0.5), squeeze=False)
    axes_flat = axes.ravel()

    section_values = section_series.to_numpy()
    xs_all = pd.to_numeric(spatial['x'], errors='coerce').to_numpy()
    ys_all = pd.to_numeric(spatial['y'], errors='coerce').to_numpy()

    # All cells are kept and plotted (not just neurons) — non-neurons are
    # drawn in gray underneath, with neurons colored by class on top, so the
    # tissue outline stays visible without competing for attention.
    class_ids_all = None
    if 'class' in spatial.columns:
        class_ids_all = extract_leading_numeric_id(spatial['class']).to_numpy()
    else:
        print("Warning: 'class' column not available; cannot distinguish neurons from non-neurons.")
    point_colors_all, neuron_mask_all = build_neuron_class_colors(class_ids_all)

    # A shared axis span (data units) — separate for x and y — and a
    # per-section centroid, so every thumbnail is drawn at the same
    # physical scale instead of each autoscaling to fill its panel with
    # just its own data — otherwise an anatomically small section (e.g. the
    # olfactory bulb) gets zoomed to look just as large as a full coronal
    # section. Same approach as save_group_spatial_maps's fix for the same
    # underlying issue. Width and height are sized independently (rather
    # than one isotropic span reused for both) because sections are
    # typically wider than tall — a single shared span would end up sized
    # to the width (the larger of the two for most sections), leaving a lot
    # of unnecessary blank margin above and below every panel even though
    # its horizontal fit was already tight. Each is sized to the
    # SPAN_PERCENTILE-th percentile of section extents (not the strict max)
    # plus 5% padding: using the single largest section would leave most
    # panels surrounded by a lot of blank margin. A percentile trades that
    # off deliberately: the few sections above it get their edges clipped
    # in this thumbnail grid only (the full data is still used everywhere
    # else — filtering, ROI picking, actual processing), in exchange for
    # noticeably less wasted space in the common case.
    SPAN_PERCENTILE = 100  # temporarily testing without percentile clipping — was 90
    section_centroids = {}
    half_widths, half_heights = [], []
    for section in unique_sections:
        sec_mask = section_values == section
        xs_sec, ys_sec = xs_all[sec_mask], ys_all[sec_mask]
        valid_sec = ~(np.isnan(xs_sec) | np.isnan(ys_sec))
        xs_sec, ys_sec = xs_sec[valid_sec], ys_sec[valid_sec]
        if len(xs_sec) == 0:
            continue
        x_min, x_max = xs_sec.min(), xs_sec.max()
        y_min, y_max = ys_sec.min(), ys_sec.max()
        section_centroids[section] = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        half_widths.append((x_max - x_min) / 2)
        half_heights.append((y_max - y_min) / 2)
    shared_half_width = np.percentile(half_widths, SPAN_PERCENTILE) * 1.05 if half_widths else 1.0
    shared_half_height = np.percentile(half_heights, SPAN_PERCENTILE) * 1.05 if half_heights else 1.0

    for i, section in enumerate(unique_sections):
        ax = axes_flat[i]
        mask = section_values == section
        xs, ys = xs_all[mask], ys_all[mask]
        valid = ~(np.isnan(xs) | np.isnan(ys))
        xs, ys = xs[valid], ys[valid]

        if neuron_mask_all is not None:
            is_neuron = neuron_mask_all[mask][valid]
            colors = point_colors_all[mask][valid]
            scatter_gray_then_colored(ax, xs, ys, is_neuron, colors, gray_size=0.1, colored_size=0.3)
        else:
            ax.scatter(xs, ys, s=0.4, c='steelblue', linewidths=0)
        cx, cy = section_centroids.get(section, (0.0, 0.0))
        ax.set_xlim(cx - shared_half_width, cx + shared_half_width)
        ax.set_ylim(cy - shared_half_height, cy + shared_half_height)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # match tissue orientation (image top/bottom vs. y-coordinate)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(sanitize_section_token(section), fontsize=12, pad=1)
        for spine in ax.spines.values():
            spine.set_edgecolor('lightgray')
            spine.set_linewidth(1)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    fig.subplots_adjust(top=0.98, bottom=0.02, hspace=0.05, wspace=0.05)
    fig.savefig(GRID_CACHE_PNG, dpi=GRID_CACHE_DPI)

    # Read the actual saved image back to get its exact pixel dimensions, so
    # the bounding boxes we store line up exactly with what gets displayed.
    img = plt.imread(GRID_CACHE_PNG)
    img_h, img_w = img.shape[0], img.shape[1]

    # Store both the pixel bounding box (for click-hit-testing and drawing
    # the whole-section selection highlight) and the data x/y limits each
    # thumbnail's axes actually ended up autoscaled to (for mapping an ROI's
    # data-space bounds into this section's little box in the grid image —
    # see map_data_to_pixel()).
    layout = {}
    for i, section in enumerate(unique_sections):
        ax = axes_flat[i]
        frac = ax.get_position()  # figure-fraction Bbox, y-up
        layout[str(section)] = {
            'pixel_bbox': [frac.x0 * img_w, frac.x1 * img_w, frac.y0 * img_h, frac.y1 * img_h],
            'data_xlim': list(ax.get_xlim()),
            'data_ylim': list(ax.get_ylim()),
        }
    with open(GRID_CACHE_LAYOUT, 'w') as f:
        json.dump(layout, f)

    plt.close(fig)
    print(f"Cached section grid image to {GRID_CACHE_PNG} ({img_w}x{img_h}px, {n} sections).")
    return img, layout


def required_cached_run_files(run_folder):
    """The 3 files a previously computed run can draw on for the "load
    existing" startup path to skip straight to the interactive UMAP viewer:
    the run's own umap_coords.csv (embedding + metadata) and roi_coords.csv
    (ROI/whole-section rectangles), plus the full processed AnnData Step 8
    wrote to CACHE_DIR (named after the run folder itself, not stored inside
    it). Returns a dict of label -> Path; callers check .exists() on each.

    Only 'UMAP coordinates' is actually required. 'ROI coordinates' is never
    written at all for a whole-brain run, and 'Processed AnnData' can be
    rebuilt from the other two (see rebuild_processed_adata) — so the
    startup panel treats both as recoverable rather than blocking on them."""
    run_folder = Path(run_folder)
    return {
        'UMAP coordinates': run_folder / 'umap_coords.csv',
        'ROI coordinates': run_folder / 'roi_coords.csv',
        'Processed AnnData': CACHE_DIR / f'{run_folder.name}_processed.h5ad',
    }


def leiden_labels_from_csv(series):
    """Leiden labels read back out of umap_coords.csv, normalized to plain
    integer-like strings ('0', '1', ...).

    A CSV round trip doesn't preserve these as written: pandas re-infers the
    column, so '0'/'1' come back as int64 — or, if any row is blank (a cell
    present in the CSV but not in the subset being loaded), as float64,
    whose str() is '0.0'. Left alone that quietly produces a second set of
    category names that don't match the '0'/'1' a freshly computed run
    yields, so the same cluster would look like two different ones
    depending on whether the run came from cache. Blanks stay None
    (unknown) rather than becoming the string 'nan'."""
    numeric = pd.to_numeric(series, errors='coerce')
    out = series.astype(object).where(series.notna(), None)
    is_num = numeric.notna()
    out[is_num] = numeric[is_num].astype('int64').astype(str)
    return out.to_numpy()


def rebuild_processed_adata(run_folder, adata_backed, processed_h5ad_path):
    """Rebuilds (and re-caches) a run's processed AnnData from its saved
    umap_coords.csv plus the raw backed h5ad, for when the .h5ad Step 8
    wrote has been deleted but the run folder itself is intact.

    Everything needed is already in the CSV — the exact cell IDs the run
    settled on, their ABC taxonomy columns, the Leiden clustering, and the
    UMAP embedding itself — so none of the expensive work (subsample,
    preprocessing, PCA, neighbors, UMAP, Leiden) has to happen again, and
    the section/ROI picker doesn't need to be reopened just to re-derive a
    selection the CSV already records. Mirrors what the main pipeline's own
    found_existing_umap branch does with a cached CSV; kept separate rather
    than shared with it because that branch is threaded through the full
    Step 0-8 flow (subsample caps, cell-type filters, section selection)
    that this path deliberately skips.

    Returns the rebuilt AnnData, having also written it back to
    `processed_h5ad_path` so the next load is a plain read again."""
    csv_path = Path(run_folder) / 'umap_coords.csv'
    print(f"Rebuilding processed AnnData from {csv_path}...")
    # converters={0: str} keeps the ~19-digit cell-ID index as text — as
    # float64 pandas silently rounds off its trailing digits.
    umap_coords = pd.read_csv(csv_path, index_col=0, converters={0: str})
    keep_mask = adata_backed.obs.index.isin(umap_coords.index)
    n_found = int(keep_mask.sum())
    if n_found == 0:
        raise RuntimeError(
            f"None of the {len(umap_coords)} cell IDs in {csv_path} were found in the raw dataset."
        )
    if n_found < len(umap_coords):
        print(f"Warning: only {n_found} of {len(umap_coords)} cells from {csv_path} "
              "were found in the raw dataset; rebuilding with those.")
    # Reads only the matched cells off disk, rather than materializing the
    # full multi-million-cell dataset just to discard most of it.
    adata = materialize_subset(adata_backed, keep_mask)
    for meta_col in ('class', 'subclass', 'supertype', 'cluster', SECTION_COL):
        if meta_col in umap_coords.columns and meta_col not in adata.obs.columns:
            adata.obs[meta_col] = umap_coords.loc[adata.obs.index, meta_col].values
    if LEIDEN_KEY in umap_coords.columns:
        adata.obs[LEIDEN_KEY] = pd.Categorical(
            leiden_labels_from_csv(umap_coords.loc[adata.obs.index, LEIDEN_KEY])
        )
    adata.obsm['X_umap'] = umap_coords.loc[adata.obs.index, ['UMAP1', 'UMAP2']].to_numpy()
    # Same as the main pipeline's cache-hit branch: adata.X here is still
    # the untouched raw counts as read from disk (materialize_subset never
    # normalizes or transforms), which is exactly what the viewer's Gene
    # mode wants from layers['counts'].
    adata.layers['counts'] = adata.X.copy()
    processed_h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing rebuilt AnnData to {processed_h5ad_path}...")
    adata.write_h5ad(processed_h5ad_path)
    print(f"Rebuilt processed AnnData for {adata.n_obs} cells.")
    return adata


def show_error_dialog(title, message):
    """Best-effort modal error dialog, falling back to the console if Tk
    isn't usable. Same withdrawn-root/topmost pattern as this file's other
    tkinter prompts (offer_leiden_backfill's own askyesno, the ROI file
    pickers), with one addition: an already-running Tk root is reused rather
    than a second one created alongside it, since these can fire while a
    matplotlib window (itself a Tk app) is open."""
    print(f"{title}: {message}")
    try:
        import tkinter as tk
        from tkinter import messagebox
        existing_root = getattr(tk, '_default_root', None)
        root = existing_root
        if root is None:
            root = tk.Tk()
            root.withdraw()
        root.attributes('-topmost', True)
        messagebox.showerror(title, message)
        if existing_root is None:
            root.destroy()
    except Exception as e:
        print(f"(Could not show the error dialog: {e})")


def leiden_failure_message(exc):
    """User-facing explanation for a failed Leiden run, with an install hint
    when the cause is a missing optional package.

    scanpy's Leiden support needs igraph (and, for the non-'igraph' flavors,
    leidenalg) — neither is a hard dependency of scanpy itself, so a missing
    one is by far the most likely reason this fails, and it's fixable in one
    command. ModuleNotFoundError carries the module in .name; a plain
    ImportError raised from inside scanpy may not, so the message text is
    checked as a fallback."""
    lines = [f"Leiden clustering could not run:", "", f"{type(exc).__name__}: {exc}", ""]
    if isinstance(exc, ImportError):
        missing = getattr(exc, 'name', None)
        if not missing:
            text = str(exc).lower()
            missing = next((pkg for pkg in ('leidenalg', 'igraph') if pkg in text), None)
        if missing:
            lines += [f"This needs the optional '{missing}' package. Install it with:",
                       "", f"    pip install {missing}", ""]
        else:
            lines += ["This needs scanpy's optional clustering packages. Install them with:",
                       "", "    pip install igraph leidenalg", ""]
    lines.append("Everything else is unaffected: the UMAP and all ABC taxonomy levels "
                  "(class/subclass/supertype/cluster) still work. Only the Leiden level "
                  "will be unavailable for this run.")
    return "\n".join(lines)


def prompt_deg_comparison_mode(parent_window, title, level_name):
    """Modal dialog for the interactive UMAP viewer's 'Export DEGs' button,
    offering two comparison modes: each specified ID versus every other
    cell (export_de_genes' original, still-default behavior — 'others'),
    or each specified ID versus one particular reference ID's own cells
    only ('reference').

    Returns {'mode': 'others' or 'reference', 'reference': int or None}
    once OK is clicked with valid input, or None if cancelled (Cancel, the
    window's own close button, or Escape).

    A real Toplevel, not tkinter.messagebox — this needs radio buttons and
    a text entry, which messagebox doesn't offer — parented to and modal
    over `parent_window` (the viewer's own Tk window, already running its
    own mainloop via show_figure_blocking) via transient()/grab_set()/
    wait_window(), the standard pattern for a modal child dialog. This
    deliberately does *not* create a second, competing Tk() root the way
    this file's earlier tkinter.messagebox-based prompts do (those run
    before any window/mainloop exists yet; this one has to coexist with
    one that's already running)."""
    import tkinter as tk
    from tkinter import messagebox

    result = {'value': None}
    dialog = tk.Toplevel(parent_window)
    dialog.title(title)
    dialog.transient(parent_window)
    dialog.resizable(False, False)

    mode_var = tk.StringVar(value='others')
    level_word = level_name.lower()
    # 'es' for names ending in a sibilant ('class'/'subclass' ->
    # classes/subclasses, the standard English rule for words ending in
    # s/x/z/ch/sh), plain 's' otherwise (supertype, cluster, leiden) — same
    # rule mode_display_label's own 'Single Subclass' branch uses, so this
    # reads correctly for every level without a per-level special case.
    level_word_plural = level_word + ('es' if level_word.endswith('s') else 's')
    tk.Radiobutton(dialog, text=f"...selected {level_word_plural} vs others", variable=mode_var,
                   value='others', anchor='w').pack(fill='x', padx=12, pady=(12, 2))
    tk.Radiobutton(dialog, text=f"...selected {level_word_plural} versus reference", variable=mode_var,
                   value='reference', anchor='w').pack(fill='x', padx=12, pady=(2, 2))

    ref_frame = tk.Frame(dialog)
    tk.Label(ref_frame, text=f"Reference {level_word} ID:").pack(side='left')
    ref_entry = tk.Entry(ref_frame, width=10)
    ref_entry.pack(side='left', padx=(6, 0))
    ref_frame.pack(fill='x', padx=32, pady=(0, 12))

    def update_ref_entry_state(*_args):
        ref_entry.configure(state='normal' if mode_var.get() == 'reference' else 'disabled')

    mode_var.trace_add('write', update_ref_entry_state)
    update_ref_entry_state()  # starts disabled — 'others' is the default selection above

    button_frame = tk.Frame(dialog)
    button_frame.pack(pady=(0, 12))

    def on_ok():
        mode = mode_var.get()
        if mode == 'reference':
            raw = ref_entry.get().strip()
            try:
                reference = int(raw)
            except ValueError:
                # Left open rather than destroyed — same "fix it and try
                # again" flow a real form validation error implies, instead
                # of silently discarding what was already typed elsewhere
                # in the dialog.
                messagebox.showerror("Invalid reference ID", f"'{raw}' is not a valid integer ID.", parent=dialog)
                return
            result['value'] = {'mode': 'reference', 'reference': reference}
        else:
            result['value'] = {'mode': 'others', 'reference': None}
        dialog.destroy()

    def on_cancel():
        result['value'] = None
        dialog.destroy()

    tk.Button(button_frame, text="OK", command=on_ok, width=10, default='active').pack(side='left', padx=6)
    tk.Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(side='left', padx=6)

    dialog.protocol("WM_DELETE_WINDOW", on_cancel)
    dialog.bind('<Return>', lambda event: on_ok())
    dialog.bind('<Escape>', lambda event: on_cancel())

    # Centered over the parent window rather than wherever Tk happens to
    # place a fresh Toplevel (typically the screen's own top-left corner).
    # update_idletasks() first — the dialog's own winfo_width()/height()
    # only reflect its real, laid-out size once the window manager has
    # actually sized the widgets above, not immediately after creating them.
    dialog.update_idletasks()
    px, py = parent_window.winfo_rootx(), parent_window.winfo_rooty()
    pw, ph = parent_window.winfo_width(), parent_window.winfo_height()
    dw, dh = dialog.winfo_width(), dialog.winfo_height()
    dialog.geometry(f"+{px + max(0, (pw - dw) // 2)}+{py + max(0, (ph - dh) // 2)}")

    dialog.grab_set()
    dialog.focus_set()
    dialog.wait_window()  # blocks here (this window's own local event loop) until destroy() above
    return result['value']


def offer_leiden_backfill(adata, csv_path, processed_h5ad_path=None):
    """For a cached run whose umap_coords.csv predates Leiden clustering:
    offer to compute it now and save it back, so the viewer's Leiden level
    works without recomputing the whole run.

    Only the clustering is recomputed — the saved UMAP embedding is left
    exactly as it was, so the layout the user already knows doesn't shift
    underneath them. That still means redoing normalize/log1p/PCA/neighbors
    (Leiden needs a neighbor graph, and cached runs don't store one), which
    is why this asks rather than just doing it; it's a fraction of a full
    run's cost, but not free.

    A no-op returning False if the clustering is already present, if the
    user declines, or if anything fails — every caller carries on with an
    adata that simply has no Leiden column, which the level selector
    already reports as unavailable."""
    if LEIDEN_KEY in adata.obs.columns:
        return False
    message = (
        f"This run's saved data has no Leiden clustering — it was computed before "
        f"Leiden was added (or Leiden could not run at the time).\n\n"
        f"Compute it now for these {adata.n_obs:,} cells and save it back?\n\n"
        f"The existing UMAP layout is kept unchanged; only the clustering is "
        f"computed, which still requires redoing PCA and the neighbor graph and "
        f"may take a few minutes.\n\n"
        f"Choosing No just leaves the Leiden level unavailable for this run."
    )
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        confirmed = messagebox.askyesno("Compute Leiden clustering?", message)
        root.destroy()
    except Exception as e:
        print(f"Could not show the Leiden prompt ({e}); asking on the console instead.")
        confirmed = prompt_yes_no("Compute Leiden clustering for this cached run? (y/n): ")
    if not confirmed:
        print("Skipping Leiden clustering; the Leiden level will be unavailable for this run.")
        return False

    try:
        print("Computing Leiden clustering for this cached run...")
        leiden_start = time.perf_counter()
        # A copy, so nothing here touches the caller's own `adata` — only
        # the resulting label column is copied back at the end.
        work = adata.copy()
        # `adata.X` is NOT reliably raw counts at this point — this is
        # exactly what was producing the reported "Input X contains NaN"
        # failure. For a run loaded from an *existing* processed .h5ad
        # (as opposed to one rebuilt fresh by rebuild_processed_adata,
        # which leaves .X alone), .X is Step 5's own already-normalized/
        # log1p'd/scaled matrix — the main pipeline scales adata.X in
        # place and only *afterward* is it ever written to disk, so a
        # freshly-loaded run's .X is that scaled (z-scored, hence
        # negative-valued) matrix, not raw counts. Re-running
        # normalize_total/log1p (below) on already-scaled data is where
        # the NaN actually came from: log1p(x) for x <= -1 is NaN
        # outright, and scaled data routinely dips well below -1. The
        # *actual* raw counts survive separately in layers['counts'] —
        # both this loaded-h5ad case (see the main pipeline's own
        # `adata.layers['counts'] = adata.X.copy()`, saved before scaling
        # overwrites .X) and the rebuild_processed_adata case (which
        # populates the same layer, redundantly but harmlessly, since
        # its own .X already *is* raw) — so it's always present here.
        if 'counts' in work.layers:
            work.X = work.layers['counts'].copy()
        # Still defensive from here, unlike the main pipeline's own
        # preprocessing (Step 5), which never reaches PCA with either of
        # these problems because it runs sc.pp.filter_cells(min_counts=20)/
        # filter_genes(min_cells=5) first. This path skips that (the whole
        # point is reusing the already-saved UMAP embedding/cell selection
        # unchanged — see the docstring), so `work.X` (now the raw counts
        # from layers['counts'], per above) is unfiltered, with no gene
        # filtering applied at all. Two ways that reaches PCA broken:
        #  - An actual NaN entry in the raw counts (e.g. a corrupted/missing
        #    MERFISH measurement) — sklearn's PCA rejects any NaN outright
        #    ("Input X contains NaN"), which is the exact failure this was
        #    written for.
        #  - A cell with zero total counts — turns normalize_total's
        #    per-cell division into 0/0 = NaN, which is the same failure
        #    one step removed.
        # Both are rare (a handful of cells at most), so zeroing stray NaNs
        # and dropping any now-all-zero cells costs this recompute nothing
        # meaningful, instead of failing Leiden for every cell over a
        # problem in a few.
        from scipy.sparse import issparse
        if issparse(work.X):
            if work.X.nnz and np.isnan(work.X.data).any():
                n_nan = int(np.isnan(work.X.data).sum())
                print(f"Warning: {n_nan} NaN count value(s) found; treating as zero.")
                work.X.data = np.nan_to_num(work.X.data, nan=0.0)
        else:
            n_nan = int(np.isnan(work.X).sum())
            if n_nan:
                print(f"Warning: {n_nan} NaN count value(s) found; treating as zero.")
                work.X = np.nan_to_num(work.X, nan=0.0)
        n_before = work.n_obs
        sc.pp.filter_cells(work, min_counts=1)
        if work.n_obs < n_before:
            print(f"Warning: dropping {n_before - work.n_obs} cell(s) with zero total counts "
                  "(would otherwise turn into NaN during normalization).")
        if work.n_obs == 0:
            raise RuntimeError("every cell has zero total counts; nothing left to cluster")
        sc.pp.normalize_total(work)
        sc.pp.log1p(work)
        n_comps = min(50, work.n_vars - 1, work.n_obs - 1)
        sc.tl.pca(work, n_comps=n_comps, svd_solver='arpack')
        sc.pp.neighbors(work, n_neighbors=15, n_pcs=n_comps)
        sc.tl.leiden(work, key_added=LEIDEN_KEY, flavor='igraph', n_iterations=2, resolution=LEIDEN_RESOLUTION)
        # Assigned by index, not by position (work.obs[...].values) — work
        # can now be a strict subset of adata (see the zero-count filtering
        # above), so a positional assignment would silently misalign every
        # row after the first dropped cell. Cells not in work (dropped
        # above) are left as NaN, same as any other cell Leiden couldn't
        # place.
        adata.obs[LEIDEN_KEY] = pd.Series(np.nan, index=adata.obs.index, dtype=object)
        adata.obs.loc[work.obs.index, LEIDEN_KEY] = work.obs[LEIDEN_KEY].astype(str).values
        del work
        gc.collect()
        print(f"Leiden found {adata.obs[LEIDEN_KEY].nunique()} clusters in "
              f"{format_duration(time.perf_counter() - leiden_start)}.")
    except Exception as e:
        show_error_dialog("Leiden clustering failed", leiden_failure_message(e))
        adata.obs.drop(columns=[LEIDEN_KEY], inplace=True, errors='ignore')
        return False

    # Written back so this is a one-time cost per run, not per session.
    # Best-effort: a failure to save doesn't invalidate the clustering
    # that's already on `adata` and usable in this session.
    try:
        csv_path = Path(csv_path)
        if csv_path.exists():
            umap_coords = pd.read_csv(csv_path, index_col=0, converters={0: str})
            umap_coords[LEIDEN_KEY] = adata.obs[LEIDEN_KEY].astype(str).reindex(umap_coords.index).values
            umap_coords.to_csv(csv_path)
            print(f"Saved Leiden clustering to {csv_path}.")
    except Exception as e:
        print(f"Warning: could not update {csv_path} with the Leiden column ({e}).")
    try:
        if processed_h5ad_path is not None and Path(processed_h5ad_path).exists():
            adata.write_h5ad(processed_h5ad_path)
            print(f"Saved Leiden clustering to {processed_h5ad_path}.")
    except Exception as e:
        print(f"Warning: could not update {processed_h5ad_path} with the Leiden column ({e}).")
    return True


def get_cached_raw_backed_adata(raw_backed_cache, abc_cache):
    """The raw, unfiltered MERFISH h5ad, opened once in backed mode and
    reused across every session in this process — both the 'new_run' path
    (Step 0) and the 'load_existing' path need this exact same read for
    adata_backed (each section's full spatial background in the
    interactive UMAP viewer), and re-opening it fresh every session the
    user starts (see the startup panel's own session loop) was pure
    redundant work: backed mode only reads the header, not X, but it's
    still a real file-system round trip and h5py setup cost each time.
    `raw_backed_cache` is a plain {} the caller keeps alive across
    sessions; this populates 'adata_backed' in it on first use."""
    if 'adata_backed' not in raw_backed_cache:
        print("Step 0: Locating raw MERFISH h5ad file...")
        h5ad_path = abc_cache.get_data_path(
            directory='MERFISH-C57BL6J-638850',
            file_name='C57BL6J-638850/raw'
        )
        print(f"Step 0: Opening {h5ad_path} in backed mode...")
        raw_backed_cache['adata_backed'] = anndata.read_h5ad(h5ad_path, backed='r')
    else:
        print("Step 0: Reusing already-opened raw MERFISH h5ad from an earlier session this run.")
    return raw_backed_cache['adata_backed']


def prompt_startup_panel(out_folder):
    """First screen shown at launch: pick a cell-type subset, then either
    open the section/ROI picker (today's full pipeline, Step 0 onward) or
    load a previously computed run's cached files directly into the
    interactive UMAP viewer, skipping section/ROI selection and UMAP
    computation entirely.

    Returns (action, cell_type_selection, load_folder, chosen_out_folder):
    action is 'new_run' or 'load_existing'; cell_type_selection is always
    one of CELL_TYPE_OPTIONS' values (used by the 'new_run' path; harmless
    but unused for 'load_existing', kept for a uniform return shape);
    load_folder is the chosen run folder (a Path) when action is
    'load_existing', else None; chosen_out_folder is the output folder the
    user settled on via the folder box (a Path, already created) — the
    caller should adopt it as the working directory for this session.

    Blocks until a choice is made (see show_figure_blocking). Exits the
    process if the window is closed without choosing an action (mirrors
    resolve_out_folder's own "cancelled -> exit" behavior for this same
    startup phase, rather than falling through into a pipeline with no
    selection made)."""
    RADIO_OPTIONS = ('Neurons', 'NonNeurons', 'All')

    fig = plt.figure(figsize=compute_figsize_for_screen_height(1.55, height_frac=0.5, default=(10, 6)))
    normalize_tk_scaling(fig)
    fig.canvas.manager.set_window_title("AllenABC UMAP Explorer")

    fig.text(0.5, 0.93, "AllenABC UMAP Explorer", ha='center', va='center', fontsize=UI_BUTTON_FONTSIZE * 1.4,
              fontweight='bold')
    fig.text(0.5, 0.85, "Cell type when choosing new section/ROI set", ha='center', va='center',
             fontsize=UI_BUTTON_FONTSIZE)

    fig_w_in = fig.get_size_inches()[0]
    dot_x, label_x, radio_width = radio_layout_as_axes_fractions(RADIO_OPTIONS, UI_BUTTON_FONTSIZE, 0.6, fig_w_in)
    radio_ax = fig.add_axes([(1 - radio_width) / 2, 0.745, radio_width, 0.09])
    radio_ax.set_xlim(0, 1)
    radio_ax.set_ylim(0, 1)
    radio_ax.set_xticks([])
    radio_ax.set_yticks([])
    for spine in radio_ax.spines.values():
        spine.set_visible(False)
    # Same fontsize**2 dot-size convention as this file's other hand-rolled
    # radios (e.g. the interactive UMAP viewer's mode_dots).
    radio_dots = radio_ax.scatter(
        dot_x, [0.5] * len(RADIO_OPTIONS), s=[UI_BUTTON_FONTSIZE ** 2] * len(RADIO_OPTIONS),
        marker='o', edgecolor='black', facecolor=['tab:blue'] + ['none'] * (len(RADIO_OPTIONS) - 1), zorder=3,
    )
    for x, label in zip(label_x, RADIO_OPTIONS):
        radio_ax.text(x, 0.5, label, fontsize=UI_BUTTON_FONTSIZE, va='center', ha='left')

    radio_state = {'index': 0}  # 'Neurons' preselected

    def set_radio_selection(idx):
        radio_state['index'] = idx
        facecolors = ['none'] * len(RADIO_OPTIONS)
        facecolors[idx] = 'tab:blue'
        radio_dots.set_facecolor(facecolors)
        fig.canvas.draw_idle()

    def on_radio_click(event):
        if event.inaxes is not radio_ax or event.xdata is None:
            return
        idx = min(range(len(dot_x)), key=lambda i: abs(dot_x[i] - event.xdata))
        set_radio_selection(idx)

    fig.canvas.mpl_connect('button_press_event', on_radio_click)

    # ------------------------------------------------------------------
    # Output-folder chooser: an editable path box, a Browse… dialog, and a
    # dropdown of recently-used folders (persisted — see
    # load_recent_output_folders / remember_recent_output_folder) so
    # switching between projects is a click, not a re-typed path.
    # ------------------------------------------------------------------
    fig.text(0.06, 0.675, "Output folder (where runs are saved and loaded from):",
             ha='left', va='center', fontsize=UI_BUTTON_FONTSIZE)

    folder_box_ax = fig.add_axes([0.06, 0.585, 0.55, 0.075])
    folder_textbox = TextBox(folder_box_ax, '', initial=str(out_folder))
    folder_textbox.text_disp.set_fontsize(UI_BUTTON_FONTSIZE)

    recent_toggle_ax = fig.add_axes([0.615, 0.585, 0.05, 0.075])
    recent_toggle_button = Button(recent_toggle_ax, '▾', hovercolor='0.85')
    recent_toggle_button.label.set_fontsize(UI_BUTTON_FONTSIZE)

    browse_ax = fig.add_axes([0.68, 0.585, 0.27, 0.075])
    browse_button = Button(browse_ax, 'Browse…', hovercolor='0.85')
    browse_button.label.set_fontsize(UI_BUTTON_FONTSIZE)

    # Opaque popup drawn on top of the action buttons below (high zorder) —
    # closed by picking an entry or clicking away, same pattern as the
    # gene-name autocomplete dropdowns elsewhere in this file. Its exact
    # position/size is set each time it's shown, sized to the number of
    # recent folders (see show_recent_dropdown).
    recent_dropdown_ax = fig.add_axes([0.06, 0.20, 0.89, 0.38], zorder=30)
    recent_dropdown_ax.set_xlim(0, 1)
    recent_dropdown_ax.set_ylim(0, 1)
    recent_dropdown_ax.set_xticks([])
    recent_dropdown_ax.set_yticks([])
    for spine in recent_dropdown_ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    recent_dropdown_ax.patch.set_facecolor('#f0f0f0')
    recent_dropdown_ax.set_visible(False)
    recent_dropdown_texts = [
        recent_dropdown_ax.text(0.03, 0, '', fontsize=UI_BUTTON_FONTSIZE * 0.85, va='center', ha='left',
                                 family='monospace')
        for _ in range(MAX_RECENT_OUTPUT_FOLDERS)
    ]
    recent_state = {'items': []}

    def current_recent_items():
        typed = folder_textbox.text.strip().strip('"')
        items = load_recent_output_folders()
        if typed and _normalized_folder_key(typed) not in {_normalized_folder_key(p) for p in items}:
            items = [typed] + items
        return items[:MAX_RECENT_OUTPUT_FOLDERS]

    def hide_recent_dropdown():
        if recent_dropdown_ax.get_visible():
            recent_dropdown_ax.set_visible(False)
            fig.canvas.draw_idle()

    def show_recent_dropdown():
        items = current_recent_items()
        recent_state['items'] = items
        if not items:
            return
        n = len(items)
        row_h = min(0.05, 0.40 / n)
        box_top = 0.575
        recent_dropdown_ax.set_position([0.06, box_top - n * row_h, 0.89, n * row_h])
        for i, t in enumerate(recent_dropdown_texts):
            if i < n:
                label = items[i]
                if len(label) > 92:  # ellipsize from the left — the tail is what differs between projects
                    label = '…' + label[-91:]
                t.set_text(label)
                t.set_position((0.03, 1 - (i + 0.5) / n))
            else:
                t.set_text('')
        recent_dropdown_ax.set_visible(True)
        fig.canvas.draw_idle()

    def on_recent_toggle(event):
        if recent_dropdown_ax.get_visible():
            hide_recent_dropdown()
        else:
            show_recent_dropdown()

    recent_toggle_button.on_clicked(on_recent_toggle)

    def on_dropdown_click(event):
        if not recent_dropdown_ax.get_visible():
            return
        if event.inaxes is recent_dropdown_ax and event.ydata is not None:
            n = len(recent_state['items'])
            if n > 0:
                row = min(n - 1, max(0, int((1 - event.ydata) * n)))
                folder_textbox.set_val(recent_state['items'][row])
            hide_recent_dropdown()
        elif event.inaxes is not recent_toggle_ax:  # the toggle has its own handler
            hide_recent_dropdown()

    fig.canvas.mpl_connect('button_press_event', on_dropdown_click)

    def show_folder_error(message):
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            messagebox.showerror("Output folder", message)
            root.destroy()
        except Exception:
            print(message)

    def committed_out_folder():
        """The folder currently in the box, created if it doesn't exist yet.
        Returns a Path on success, or None (after showing an error and
        leaving the panel open) if it's blank or can't be created."""
        raw = folder_textbox.text.strip().strip('"')
        if not raw:
            show_folder_error("Enter an output folder, or click Browse… to pick one.")
            return None
        path = Path(os.path.expanduser(raw))
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            show_folder_error(f"Can't create or access that folder:\n\n{path}\n\n{e}")
            return None
        return path

    def on_browse(event):
        hide_recent_dropdown()
        raw = folder_textbox.text.strip().strip('"')
        initial = raw if raw and Path(os.path.expanduser(raw)).is_dir() else str(out_folder)
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            chosen = filedialog.askdirectory(title="Choose the output folder", initialdir=initial)
            root.destroy()
        except Exception as e:
            print(f"Could not open folder picker: {e}")
            return
        if chosen:
            folder_textbox.set_val(os.path.normpath(chosen))  # askdirectory returns forward slashes on Windows

    browse_button.on_clicked(on_browse)

    result = {'action': None, 'load_folder': None, 'out_folder': None}
    button_width, button_height = 0.55, 0.08
    button_left = (1 - button_width) / 2

    new_run_ax = fig.add_axes([button_left, 0.45, button_width, button_height])
    new_run_button = Button(new_run_ax, 'Open Brain Browser/ROI Chooser', hovercolor='0.85')
    new_run_button.label.set_fontsize(UI_BUTTON_FONTSIZE)

    load_existing_ax = fig.add_axes([button_left, 0.335, button_width, button_height])
    load_existing_button = Button(load_existing_ax, 'Load Existing ROI Dataset…', hovercolor='0.85')
    load_existing_button.label.set_fontsize(UI_BUTTON_FONTSIZE)

    cancel_ax = fig.add_axes([button_left, 0.115, button_width, button_height])
    cancel_button = Button(cancel_ax, 'Exit', hovercolor='0.85')
    cancel_button.label.set_fontsize(UI_BUTTON_FONTSIZE)

    def on_new_run(event):
        hide_recent_dropdown()
        folder = committed_out_folder()
        if folder is None:
            return
        result['action'] = 'new_run'
        result['out_folder'] = folder
        remember_recent_output_folder(folder)
        plt.close(fig)

    def on_cancel(event):
        plt.close(fig)

    def on_load_existing(event):
        hide_recent_dropdown()
        folder = committed_out_folder()
        if folder is None:
            return
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            chosen = filedialog.askdirectory(
                title="Select an existing run folder to load", initialdir=str(folder),
            )
            root.destroy()
        except Exception as e:
            print(f"Could not open folder picker: {e}")
            return
        if not chosen:
            return
        chosen_path = Path(chosen)
        required = required_cached_run_files(chosen_path)
        # Only umap_coords.csv is genuinely required. The other two are
        # recoverable:
        #   ROI coordinates — a run over the whole brain (no ROIs, no
        #     explicit whole-section picks) never gets a roi_coords.csv
        #     written in the first place (see the main script's own
        #     roi_csv_path-writing block); warned about, not blocked, once
        #     this run folder is actually loaded.
        #   Processed AnnData — rebuilt on demand from the raw backed h5ad
        #     plus this CSV's own cell IDs/metadata, no section/ROI picker
        #     needed (see rebuild_processed_adata).
        recoverable = {'ROI coordinates', 'Processed AnnData'}
        missing = [f'{label}: {path}' for label, path in required.items()
                   if label not in recoverable and not path.exists()]
        if missing:
            message = "This folder is missing required cached file(s):\n\n" + "\n".join(missing)
            try:
                messagebox.showerror("Missing cached files", message)
            except Exception:
                print(message)
            return  # stay on the panel so the user can pick a different folder
        result['action'] = 'load_existing'
        result['load_folder'] = chosen_path
        result['out_folder'] = folder
        remember_recent_output_folder(folder)
        plt.close(fig)

    new_run_button.on_clicked(on_new_run)
    load_existing_button.on_clicked(on_load_existing)
    cancel_button.on_clicked(on_cancel)

    center_figure_window(fig)
    show_figure_blocking(fig)

    if result['action'] is None:
        print("No selection made; exiting.")
        sys.exit(0)

    cell_type_selection = RADIO_OPTIONS[radio_state['index']]
    return result['action'], cell_type_selection, result['load_folder'], result['out_folder']


def prompt_section_selection_gui(adata, abc_cache, section_series, out_folder=None, cell_type_selection='All',
                                 imputed_state=None):
    """Clickable, zoomable/pannable grid of per-section spatial thumbnails
    (a single cached bitmap, not one Axes per section — see
    generate_section_grid_image). Click a thumbnail to toggle it
    (multi-select), scroll to zoom/pan the whole grid, then press 'Confirm
    Selection'. Double-click a thumbnail to open a single-section ROI picker
    for that section (after a confirmation, since it takes a few seconds to
    load) — this lets you collect ROIs across multiple sections in one
    session. 'Load ROIs' opens a file picker (defaulting to `out_folder`,
    where ROI CSVs get saved) to merge in a previously saved set. Returns
    (selected_labels, sections_suffix, rois, imputed_adata): the first two
    have the same contract as prompt_section_selection(); rois is a list of
    dicts ({'section','x_min','x_max','y_min','y_max'}), empty if none were
    drawn; imputed_adata is the imputed-gene-expression AnnData if 'Imputed
    Gene' was ever selected this session (see ensure_imputed_gene_dataset_
    loaded) — already loaded and ready to reuse, e.g. by a later interactive
    UMAP viewer — or None if it never was.
    Raises RuntimeError if a GUI can't be shown (caller should fall back to
    the console prompt in that case).

    `cell_type_selection` is the same 'Neurons'/'NonNeurons'/'All' choice
    prompt_cell_type_selection() collects before this is even called — note
    that `adata` here is called with, at this point in the pipeline, still
    unfiltered by it (filter_by_cell_type() doesn't run until after this
    whole picker session ends), so every cell type is still actually
    present in every section shown here regardless of this choice; it's
    only threaded through to each single-section window's gene view, where
    it controls which type (if any) gets greyed out rather than colored by
    expression — see render_gene_expression_array's own docstring."""
    backend = matplotlib.get_backend().lower()
    if backend in ('agg', 'pdf', 'svg', 'ps', 'template', 'cairo'):
        raise RuntimeError(f"non-interactive matplotlib backend '{backend}'")

    img, layout = None, None
    if GRID_CACHE_PNG.exists() and GRID_CACHE_LAYOUT.exists():
        with open(GRID_CACHE_LAYOUT) as f:
            layout = json.load(f)
        # Older cache format stored a flat [x0,x1,y0,y1] list per section
        # (pixel bbox only, no data x/y limits, needed for mapping ROI
        # rectangles onto the grid) — treat that as a cache miss.
        if layout and not isinstance(next(iter(layout.values())), dict):
            print("Cached section grid uses an older format; regenerating...")
            layout = None
        else:
            print(f"Loading cached section grid image from {GRID_CACHE_PNG}...")
            img = plt.imread(GRID_CACHE_PNG)

    if layout is None:
        if section_series is None:
            raise RuntimeError("no section metadata available")
        print("No cached section grid image found; generating one (this happens once)...")
        img, layout = generate_section_grid_image(adata, abc_cache, section_series)

    unique_sections = list(layout.keys())
    img_h, img_w = img.shape[0], img.shape[1]

    # Margins for the main (image) axes, as figure fractions — bottom is
    # taller than the others to leave room for the button row. Reused below
    # for both the figsize calculation and the actual subplots_adjust call,
    # so they can't drift out of sync with each other.
    AXES_LEFT, AXES_RIGHT, AXES_BOTTOM, AXES_TOP = 0.005, 0.995, 0.08, 0.995
    axes_width_frac = AXES_RIGHT - AXES_LEFT
    axes_height_frac = AXES_TOP - AXES_BOTTOM

    # imshow keeps the image's own aspect ratio and pads the rest of its
    # axes box with blank space if the box's aspect doesn't match — that
    # padding, plus the axes spine drawn around the (larger, padded) box
    # rather than snug against the image, is what read as "two borders with
    # a lot of space between them and around them." Picking the figure's
    # aspect ratio so the axes box's aspect exactly matches the image's
    # means there's no padding to begin with.
    fig_aspect = (img_w / img_h) * (axes_height_frac / axes_width_frac)

    fig, ax = plt.subplots(figsize=compute_figsize_for_screen_height(fig_aspect, default=(10, 7)))
    normalize_tk_scaling(fig)
    # 'nearest' skips the antialiasing/resampling filter matplotlib would
    # otherwise re-run on every pan/zoom redraw — that filter, not the source
    # image's resolution per se, is what makes interaction feel laggy on a
    # large cached bitmap.
    ax.imshow(img, extent=(0, img_w, 0, img_h), origin='upper', interpolation='nearest')
    # 'auto' (rather than imshow's default 'equal') fills the axes box
    # exactly regardless of any small mismatch between the box's aspect
    # ratio and the image's — 'equal' instead expands the *data* range to
    # preserve aspect when the two don't match exactly, which left the
    # (fixed-size) image visibly smaller than its box despite fig_aspect
    # above already targeting a close match.
    ax.set_aspect('auto')
    # Anywhere within the axes' current xlim/ylim that imshow's own extent
    # doesn't cover (the padding added by compute_padded_canvas_bounds
    # below) shows this facecolor — white to match the grid thumbnails' own
    # background (unlike the single-section picker's black), so the padding
    # reads as more of the same canvas rather than a visible seam.
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, img_w)
    ax.set_ylim(0, img_h)

    # Tracks the current *un-zoomed* full view: the image itself plus
    # whatever white padding compute_padded_canvas_bounds below has added to
    # match the axes box's aspect ratio — see the single-section ROI
    # picker's identical mechanism (prompt_subregion_selection) for the
    # full rationale. clamp_view uses this (not img_w/img_h directly) as the
    # zoom-out/pan limit, which is what makes that padding real,
    # pannable/zoomable data-space canvas rather than a fixed backdrop the
    # view can never reach past. Recomputed on every resize (see on_resize)
    # since the box's aspect can genuinely change after the window's already
    # open — e.g. maximizing, zooming in, then un-maximizing back to a
    # narrower window.
    canvas_state = {'xlim': (0, img_w), 'ylim': (0, img_h)}

    def compute_padded_canvas_bounds(box_aspect):
        """(xlim, ylim) for the full, un-zoomed canvas: the image, centered,
        padded with extra data-space on whichever axis has less room so the
        canvas's own aspect ratio exactly matches `box_aspect` (the axes
        box's on-screen width/height ratio) — letting 'auto' aspect fill the
        box with zero distortion, since the data range's ratio already
        matches the box's, instead of the previous approach of shrinking the
        *box* itself to match the image (which is what left blank figure
        background on the sides in the first place)."""
        image_aspect = img_w / img_h
        if box_aspect > image_aspect:
            canvas_h = img_h
            canvas_w = img_h * box_aspect
        else:
            canvas_w = img_w
            canvas_h = img_w / box_aspect
        cx, cy = img_w / 2, img_h / 2
        return (cx - canvas_w / 2, cx + canvas_w / 2), (cy - canvas_h / 2, cy + canvas_h / 2)

    def rescale_view_to_aspect(x0, x1, y0, y1, box_aspect):
        """Adjust (x0,x1,y0,y1) — preserving its center — so its own aspect
        ratio matches `box_aspect`, by expanding (never shrinking) whichever
        dimension has too little room. Used to correct the *current* view
        (zoomed in, panned, or the full canvas) after a real resize changes
        the box's own aspect, instead of only ever setting the view once (an
        earlier version of this fix): that avoided disrupting an in-progress
        zoom during a resize, but also meant a later genuine resize (e.g.
        maximize, zoom in, then un-maximize) never re-corrected the now-
        stale view — the image rendered stretched into whatever aspect no
        longer matched the window's new shape."""
        width, height = x1 - x0, y1 - y0
        if width <= 0 or height <= 0 or box_aspect <= 0:
            return x0, x1, y0, y1
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if (width / height) < box_aspect:
            new_width = height * box_aspect
            return cx - new_width / 2, cx + new_width / 2, y0, y1
        else:
            new_height = width / box_aspect
            return x0, x1, cy - new_height / 2, cy + new_height / 2

    def on_resize(event):
        # The axes box now always fills the whole available region (unlike
        # the previous approach, which shrunk *this box* to match the
        # image's own aspect ratio, leaving blank space beside it) —
        # aspect-correctness instead comes from padding the *data* range to
        # match the box's aspect (see compute_padded_canvas_bounds), so
        # 'auto' aspect still displays with zero distortion.
        fig_w_px, fig_h_px = event.width, event.height
        if fig_w_px <= 0 or fig_h_px <= 0:
            return
        avail_w_px = fig_w_px * axes_width_frac
        avail_h_px = fig_h_px * axes_height_frac
        if avail_w_px <= 0 or avail_h_px <= 0:
            return
        ax.set_position([AXES_LEFT, AXES_BOTTOM, axes_width_frac, axes_height_frac])
        box_aspect = avail_w_px / avail_h_px
        # The zoom-out/pan limit always reflects the *current* box shape...
        canvas_state['xlim'], canvas_state['ylim'] = compute_padded_canvas_bounds(box_aspect)
        # ...and the *current* view (whatever it was — possibly zoomed/
        # panned) gets corrected to that same new shape, preserving its
        # center/zoom level rather than snapping back to the full canvas.
        rescaled = rescale_view_to_aspect(*ax.get_xlim(), *ax.get_ylim(), box_aspect)
        # Re-clamped in case rescaling (e.g. a drastic aspect change) pushed
        # the view outside the — also just-updated — canvas bounds.
        new_x0, new_x1, new_y0, new_y1 = clamp_view(*rescaled)
        ax.set_xlim(new_x0, new_x1)
        ax.set_ylim(new_y0, new_y1)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('resize_event', on_resize)

    selected = set()
    highlight_patches = {}
    rois = []
    roi_border_patches = {}
    roi_rect_patches = {}
    spatial_cache = {'df': None, 'load_thread': None, 'load_progress': {'rows': 0, 'total': None}, 'load_error': None}

    def start_spatial_load_if_needed():
        ensure_spatial_load_started(spatial_cache, adata, abc_cache)

    # Deliberately *not* started eagerly here (it used to be, "since most
    # sessions need this sooner or later") — it's only actually needed by
    # handle_double_click's own single-section view (which already calls
    # start_spatial_load_if_needed() itself, lazily, right when it's about
    # to wait on it — see below). A session that only ever uses "Load ROIs"
    # (never double-clicking a section) has no use for this at all, but the
    # eager version started it anyway: a daemon thread that doesn't die
    # with this window, quietly chewing through a multi-million-row CSV via
    # pandas' own C parser — which holds the GIL for the duration of each
    # chunk it parses — for the rest of the script's run, including deep
    # into the *later*, unrelated interactive UMAP viewer. That's what was
    # behind "everything feels sluggish" there despite every render path
    # itself measuring fast: the main thread's Tk event loop was being
    # starved of the GIL by this leftover background thread, not by
    # anything the interactive viewer was actually doing.

    # Unlike spatial_cache above, this is *not* started eagerly — the
    # imputed dataset is much larger than the standard 500-gene panel and
    # most sessions never touch it, so it's only ever loaded the first time
    # 'Imputed Gene' is actually selected (see the module-level
    # ensure_imputed_gene_dataset_loaded), behind an explicit warning/
    # confirmation. Returned from this function (see its own return
    # statements below) so the post-UMAP interactive viewer can reuse
    # whatever got loaded here instead of paying for a second load. A caller
    # that keeps one alive across sessions (run_session's, from the
    # module-level session_imputed_state) passes it in, so a dataset loaded in
    # an earlier session is reused here too.
    if imputed_state is None:
        imputed_state = {'adata': None, 'load_thread': None, 'load_error': None}

    # matplotlib delivers a double-click as two separate button_press_events
    # (the first with dblclick=False, the second with dblclick=True) — there
    # is no way to know the first press is part of a double-click until the
    # second one arrives. The single-click toggle used to wait out a timer
    # for that, which made every click visibly lag. Now the first press
    # toggles immediately, and if the second press of a double-click lands
    # on the same section, it undoes that toggle before opening the ROI
    # picker, so a double-click still leaves the selection unchanged (at the
    # cost of the outline flickering briefly).
    click_state = {'last_toggle': None}  # (section, perf_counter time) of the latest single-click toggle
    DOUBLE_CLICK_WINDOW_MS = 450  # only a double-click's second press this soon after a toggle undoes it

    def redraw_selection(section, is_selected):
        # The red whole-section outline only makes sense when the whole
        # section is actually what will be used — once any ROI has been
        # drawn on it, the orange dashed outline (redraw_roi_indicators)
        # takes over as the only indicator, since ROIs take priority over
        # the whole-section selection at confirm time (see run_suffix
        # construction) and showing both would misleadingly suggest the
        # whole section is still in play.
        has_rois = any(roi['section'] == section for roi in rois)
        if is_selected and not has_rois:
            if section not in highlight_patches:
                x0, x1, y0, y1 = layout[section]['pixel_bbox']
                patch = Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    linewidth=3, edgecolor='red', facecolor='none',
                    animated=True,  # drawn by blit_highlights/on_grid_draw, not full redraws
                )
                ax.add_patch(patch)
                highlight_patches[section] = patch
        elif section in highlight_patches:
            highlight_patches.pop(section).remove()

    def redraw_roi_indicators(section):
        # Two layers: a dashed border around the whole thumbnail (easy to
        # spot at a glance which sections have any ROIs, even zoomed out)
        # plus a small solid outline per ROI, mapped from its own data-space
        # bounds into this section's pixel box, showing where within the
        # section each ROI actually is.
        old_border = roi_border_patches.pop(section, None)
        if old_border is not None:
            old_border.remove()
        for patch in roi_rect_patches.pop(section, []):
            patch.remove()

        section_rois = [roi for roi in rois if roi['section'] == section]
        if not section_rois:
            return

        x0, x1, y0, y1 = layout[section]['pixel_bbox']
        border = Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2, edgecolor='orange', facecolor='none', linestyle='--',
        )
        ax.add_patch(border)
        roi_border_patches[section] = border

        section_layout = layout[section]
        new_rect_patches = []
        for roi in section_rois:
            px_x0, px_y0 = map_data_to_pixel(section_layout, roi['x_min'], roi['y_min'])
            px_x1, px_y1 = map_data_to_pixel(section_layout, roi['x_max'], roi['y_max'])
            x_lo, x_hi = sorted((px_x0, px_x1))
            y_lo, y_hi = sorted((px_y0, px_y1))
            patch = Rectangle(
                (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                linewidth=1.2, edgecolor='orange', facecolor='none',
            )
            ax.add_patch(patch)
            new_rect_patches.append(patch)
        roi_rect_patches[section] = new_rect_patches

    def handle_double_click(section):
        has_cache = load_valid_section_roi_cache(section) is not None
        going_into_gene_mode = session_view_settings['mode'] in ('gene', 'imputed_gene')
        using_imputed = session_view_settings['mode'] == 'imputed_gene'
        # A saved Imputed Gene setting implies the dataset was loaded (it's
        # shared across windows and sessions), but if it somehow isn't, open
        # in standard view rather than rendering with no imputed data.
        if using_imputed and imputed_state['adata'] is None:
            going_into_gene_mode = using_imputed = False
        # Captured regardless of mode — even when opening in standard view,
        # the single-section window's gene box gets pre-filled with the saved
        # gene (see initial_view below), so it's ready to go the moment the
        # user picks 'Gene' over there instead of arriving empty.
        saved_gene_text = (session_view_settings['gene'] or '').strip()
        requested_gene = saved_gene_text if going_into_gene_mode else None
        # Taken whenever going into gene mode with a gene name entered,
        # *regardless* of has_cache — deliberately not gated on it the way
        # the class-colored render below still is. Gating this on has_cache
        # too was exactly what made the dialog-then-window sequence
        # inconsistent: whether a given section happened to already have a
        # class-colored cache (from any earlier visit, this session or a
        # past run — e.g. from once clicking 'Groups' on it) decided
        # whether double-clicking it skipped the dialog entirely and opened
        # straight into a blank window that filled in via the status bar
        # instead. render_gene_expression_array() has its own disk cache
        # regardless (see gene_expression_cache_paths) — an already-cached
        # gene render still shows this dialog, just very briefly, which is
        # the point: dialog first, then a fully-formed window, every time.
        # Single gene only: render_gene_expression_array draws one gene. A
        # saved multi-gene setting ("Sox14, Foxp1") skips this and lets the
        # section window render its red/green/blue overlay itself, rather than
        # failing here first and printing a spurious error.
        take_fast_path = going_into_gene_mode and bool(requested_gene) and ',' not in requested_gene
        proc_fig = None
        precomputed_gene_open = None
        # Whenever the fast path might run, even if a class cache already
        # exists (it isn't needed at all when going into gene mode).
        need_dialog = take_fast_path or not has_cache

        if need_dialog:
            # Two genuinely slow steps can be needed here — loading the
            # (multi-million-row, one-time-only) spatial CSV, and rendering
            # +caching *this* section's background image — and both now run
            # in a background thread with the same dialog, so Cancel and
            # the progress bar stay responsive throughout either one, not
            # just the CSV load. Without this, a second (or third, ...)
            # double-click on a different not-yet-cached section — where
            # the CSV is already warm, so only the render step runs — did
            # that work with no responsive dialog at all, making the
            # window look like it had frozen and Cancel like it did nothing.
            proc_fig, cancel_flag, set_progress = show_processing_dialog(
                (f"Generating '{requested_gene}' expression view for section "
                 f"{sanitize_section_token(section)}...\nThis can take a few seconds. Click Cancel to stop.")
                if take_fast_path else
                (f"Generating view for section {sanitize_section_token(section)}...\n"
                 "This can take a few seconds; image will be cached to speed up subsequent loads. "
                 "Click Cancel to stop.")
            )

            if spatial_cache['df'] is None:
                # Usually already running (or done) by now — see
                # start_spatial_load_if_needed()'s call right when this
                # window opened; this only actually starts a fresh load if
                # that one somehow hasn't (e.g. it already failed and
                # left load_thread cleared — see below). Either way, what
                # follows just waits on whatever thread is running.
                start_spatial_load_if_needed()
                thread = spatial_cache['load_thread']
                while thread.is_alive() and not cancel_flag['cancelled']:
                    total = spatial_cache['load_progress']['total']
                    if total:
                        set_progress(spatial_cache['load_progress']['rows'] / total)
                    proc_fig.canvas.flush_events()
                    time.sleep(0.05)

                if cancel_flag['cancelled']:
                    # Unlike before this used a shared background load,
                    # cancelling here doesn't discard it — the thread keeps
                    # running (still can't be safely interrupted mid-
                    # flight) and still writes its result into
                    # spatial_cache['df'] once done, same as the eager
                    # load always would have; a future double-click just
                    # finds it already there instead of paying for a full
                    # reload.
                    plt.close(proc_fig)
                    print(f"Cancelled — not opening the ROI picker for section {section}.")
                    return
                if spatial_cache['df'] is None:
                    # Failed — the worker already cleared load_thread on its
                    # way out, so the next double-click's
                    # start_spatial_load_if_needed() call retries fresh
                    # rather than treating this as permanent.
                    plt.close(proc_fig)
                    print(f"Could not load spatial data for section {section}: {spatial_cache['load_error']}")
                    return

            # Render + cache this section's background image — deliberately
            # on the *main* thread, unlike the CSV load above. Even though
            # generate_and_cache_section_image() only ever touches an
            # off-screen Agg-backed Figure, never Tkinter directly, actually
            # running it in a background thread still produced "main thread
            # is not in main loop" errors — Tcl/Tk's reentrancy guarantees
            # are apparently fragile enough that *any* second thread doing
            # work concurrently with this loop's flush_events() calls is
            # asking for trouble, whether or not that thread's own code
            # looks Tk-free. So instead, progress_callback below runs
            # synchronously, once per stage, right here on the main thread —
            # pumping events and checking Cancel between stages rather than
            # continuously during them. Coarser (Cancel can only take effect
            # at a stage boundary, not instantly), but safe.
            section_figsize = compute_figsize_for_screen_height(15 / 10, default=(15, 10))
            set_progress(None)

            def on_render_progress(fraction):
                set_progress(fraction)
                proc_fig.canvas.flush_events()
                if cancel_flag['cancelled']:
                    raise _RenderCancelled()

            if take_fast_path:
                try:
                    layout, img_w, img_h = compute_section_layout(
                        section_series, section, spatial_cache['df'], section_figsize,
                    )
                    rgba, resolved_gene, max_expr, color_vmin, color_vmax = render_gene_expression_array(
                        adata, section_series, section, lambda: spatial_cache['df'], requested_gene,
                        layout, img_w, img_h, on_progress=on_render_progress,
                        cell_type_selection=cell_type_selection, color_scale=session_color_scale['mode'],
                        expr_adata=imputed_state['adata'] if using_imputed else None,
                        dataset='imputed' if using_imputed else 'standard',
                    )
                    precomputed_gene_open = {
                        'layout': layout, 'img_w': img_w, 'img_h': img_h,
                        'rgba': rgba, 'resolved_gene': resolved_gene, 'max_expr': max_expr,
                        'color_vmin': color_vmin, 'color_vmax': color_vmax,
                        'color_scale': session_color_scale['mode'],
                        # Read by prompt_subregion_selection to seed its own
                        # Groups/Gene/Imputed Gene radio correctly — this
                        # render already picked a dataset (see expr_adata
                        # above), but that choice isn't otherwise recorded
                        # anywhere in this dict.
                        'dataset': 'imputed' if using_imputed else 'standard',
                    }
                except _RenderCancelled:
                    plt.close(proc_fig)
                    print(f"Cancelled — not opening the ROI picker for section {section}.")
                    return
                except Exception as e:
                    # Falls back to the classic class-colored render below
                    # instead of failing outright — e.g. a typo'd gene name
                    # shouldn't block opening the section, just the
                    # shortcut for skipping straight to its (nonexistent)
                    # expression view.
                    print(f"Could not render '{requested_gene}' expression for section {section} "
                          f"({e}); falling back to the standard view.")
                    take_fast_path = False

            if not take_fast_path and not has_cache:
                try:
                    generate_and_cache_section_image(
                        adata, abc_cache, section_series, section, spatial_cache['df'],
                        section_figsize, progress_callback=on_render_progress,
                    )
                except _RenderCancelled:
                    plt.close(proc_fig)
                    print(f"Cancelled — not opening the ROI picker for section {section}.")
                    return
                except Exception as e:
                    plt.close(proc_fig)
                    print(f"Could not render section {section}: {e}")
                    return
            # proc_fig deliberately stays open here — prompt_subregion_selection
            # still has to read the cache back and set up its window before
            # it's ready, so closing now would leave a visible gap. It gets
            # closed via on_ready, right as that window is about to appear
            # (or in the except branch above, as a fallback if setup fails
            # first).

        def close_processing_dialog():
            if proc_fig is not None:
                plt.close(proc_fig)

        existing_rois = [
            (i, (roi['x_min'], roi['x_max'], roi['y_min'], roi['y_max']))
            for i, roi in enumerate(rois) if roi['section'] == section
        ]
        try:
            actions = prompt_subregion_selection(
                adata, abc_cache, section_series, section,
                existing_rois=existing_rois,
                on_ready=close_processing_dialog,
                spatial=spatial_cache['df'],
                initial_view=None if precomputed_gene_open is not None else {
                    'mode': session_view_settings['mode'],
                    'gene': saved_gene_text or None,
                },
                # The window writes its final radio selection and gene box text
                # back here when it closes, for the next section opened.
                session_view_settings=session_view_settings,
                precomputed_gene_open=precomputed_gene_open,
                cell_type_selection=cell_type_selection,
                session_color_scale=session_color_scale,
                shared_spatial_cache=spatial_cache,
                # Passed through (not resolved to using_imputed's snapshot
                # at double-click time) so the window's own Imputed Gene
                # radio can load it on demand and switch independently of
                # whatever's currently selected back on the grid.
                imputed_state=imputed_state,
            )
        except Exception as e:
            close_processing_dialog()
            print(f"Could not open ROI picker for section {section}: {e}")
            return
        if not actions:
            return
        # 'edit'/'add' first, while every 'delete' action's index still
        # refers to its original position in `rois` — applying deletes
        # first (or interleaved) would shift later indices out from under
        # any edit/add actions still waiting to use them. Deletes themselves
        # are applied afterward in descending index order so removing one
        # doesn't invalidate the *other* pending delete indices either.
        delete_indices = []
        for action in actions:
            if action['action'] == 'delete':
                delete_indices.append(action['index'])
                continue
            x_min, x_max, y_min, y_max = action['bounds']
            if action['action'] == 'edit':
                rois[action['index']].update(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
                verb = f"Updated ROI #{action['index'] + 1}"
            else:
                rois.append({'section': section, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max})
                verb = "Added ROI"
            print(f"{verb} on section {section}: x[{x_min:.1f}, {x_max:.1f}] "
                  f"y[{y_min:.1f}, {y_max:.1f}] (total ROIs: {len(rois)}).")
        for index in sorted(delete_indices, reverse=True):
            removed = rois.pop(index)
            print(f"Deleted ROI #{index + 1} on section {removed['section']} (total ROIs: {len(rois)}).")
        redraw_roi_indicators(section)
        redraw_selection(section, section in selected)  # drop the red outline now that this section has an ROI
        fig.canvas.draw_idle()

    def section_at(xdata, ydata):
        for section, section_layout in layout.items():
            x0, x1, y0, y1 = section_layout['pixel_bbox']
            if x0 <= xdata <= x1 and y0 <= ydata <= y1:
                return section
        return None

    def clamp_view(x0, x1, y0, y1):
        """Never zoom out past the full padded canvas view, and never let
        the visible window extend past the canvas bounds — against
        canvas_state (the image plus its white aspect-matching padding —
        see compute_padded_canvas_bounds), not img_w/img_h directly: that
        padding is real, pannable/zoomable data-space canvas, not a fixed
        backdrop, so both the zoom-out limit and the pan bounds need to
        extend into it too, not stop exactly at the image's own edges."""
        canvas_x0, canvas_x1 = canvas_state['xlim']
        canvas_y0, canvas_y1 = canvas_state['ylim']
        canvas_w, canvas_h = canvas_x1 - canvas_x0, canvas_y1 - canvas_y0
        width, height = x1 - x0, y1 - y0
        if width >= canvas_w:
            x0, x1, width = canvas_x0, canvas_x1, canvas_w
        if height >= canvas_h:
            y0, y1, height = canvas_y0, canvas_y1, canvas_h
        if x0 < canvas_x0:
            x0, x1 = canvas_x0, canvas_x0 + width
        elif x1 > canvas_x1:
            x0, x1 = canvas_x1 - width, canvas_x1
        if y0 < canvas_y0:
            y0, y1 = canvas_y0, canvas_y0 + height
        elif y1 > canvas_y1:
            y0, y1 = canvas_y1 - height, canvas_y1
        return x0, x1, y0, y1

    # Redrawing this large cached bitmap is the expensive part of every
    # pan/zoom step (not the coordinate math), so cap how often we actually
    # trigger a redraw during a fast drag/scroll burst — otherwise redraw
    # requests queue up faster than the canvas can render them and input
    # feels laggy/delayed. REDRAW_MIN_INTERVAL caps this at ~33 redraws/sec.
    # A skipped redraw always gets a trailing timer so the view still
    # settles to its true final state even if no further event arrives
    # (e.g. the very last scroll notch or mouse-move of a burst).
    REDRAW_MIN_INTERVAL = 0.03
    redraw_state = {'last_time': 0.0, 'timer': None}

    def force_draw():
        redraw_state['last_time'] = time.perf_counter()
        fig.canvas.draw_idle()

    def throttled_draw_idle():
        if redraw_state['timer'] is not None:
            redraw_state['timer'].stop()
            redraw_state['timer'] = None
        elapsed = time.perf_counter() - redraw_state['last_time']
        if elapsed >= REDRAW_MIN_INTERVAL:
            force_draw()
            return
        timer = fig.canvas.new_timer(interval=max((REDRAW_MIN_INTERVAL - elapsed) * 1000, 1))
        timer.single_shot = True
        timer.add_callback(force_draw)
        redraw_state['timer'] = timer
        timer.start()

    pan_state = {'active': False}

    def apply_pan(x_px, y_px):
        # Pixel deltas divided by the (unchanging, since panning doesn't
        # rescale) axes size in pixels give a data-space shift that keeps
        # the point originally under the cursor fixed under the cursor.
        bbox = ax.get_window_extent()
        x0, x1 = pan_state['xlim0']
        y0, y1 = pan_state['ylim0']
        dx = -(x_px - pan_state['x0_px']) / bbox.width * (x1 - x0)
        dy = -(y_px - pan_state['y0_px']) / bbox.height * (y1 - y0)
        new_x0, new_x1, new_y0, new_y1 = clamp_view(x0 + dx, x1 + dx, y0 + dy, y1 + dy)
        ax.set_xlim(new_x0, new_x1)
        ax.set_ylim(new_y0, new_y1)

    # A full redraw of the grid takes ~0.3s (the whole-dataset grid image is
    # ~6900x4700 px and gets resampled on every draw), which made each
    # click's red outline visibly lag. The outlines are animated artists
    # instead: every full draw saves the axes without them, and a toggle
    # just restores that and paints the outlines on top.
    highlight_blit_state = {'background': None}

    def draw_highlight_patches():
        for patch in highlight_patches.values():
            ax.draw_artist(patch)

    def on_grid_draw(event):
        highlight_blit_state['background'] = fig.canvas.copy_from_bbox(ax.bbox)
        draw_highlight_patches()

    fig.canvas.mpl_connect('draw_event', on_grid_draw)

    def blit_highlights():
        background = highlight_blit_state['background']
        if background is None:
            fig.canvas.draw_idle()
            return
        fig.canvas.restore_region(background)
        draw_highlight_patches()
        fig.canvas.blit(ax.bbox)

    def toggle_selection(section):
        if section in selected:
            selected.discard(section)
        else:
            selected.add(section)
        redraw_selection(section, section in selected)
        blit_highlights()

    RIGHT_CLICK_DRAG_THRESHOLD_PX = 5  # below this, a right-button press/release is a click, not a pan

    def show_grid_context_menu(event):
        try:
            import tkinter as tk
            menu = tk.Menu(fig.canvas.manager.window, tearoff=0)
            menu.add_command(label="Open local cache folder", command=lambda: open_with_default_viewer(CACHE_DIR))
            if out_folder is not None:
                menu.add_command(label="Open output data folder", command=lambda: open_with_default_viewer(out_folder))
            gui_event = event.guiEvent
            try:
                menu.tk_popup(gui_event.x_root, gui_event.y_root)
            finally:
                menu.grab_release()
        except Exception as e:
            print(f"Could not show context menu ({e}).")

    def on_press(event):
        if event.inaxes is not ax:
            return
        if event.button == 1:  # left click: toggle selection, double-click: ROI picker
            if event.xdata is None or event.ydata is None:
                return
            section = section_at(event.xdata, event.ydata)
            if section is None:
                return
            if event.dblclick:
                # Second press of a double-click: revert the first press's
                # toggle (if it was on this section) and open the picker.
                last = click_state['last_toggle']
                click_state['last_toggle'] = None
                if (last is not None and last[0] == section
                        and time.perf_counter() - last[1] <= DOUBLE_CLICK_WINDOW_MS / 1000):
                    toggle_selection(section)
                handle_double_click(section)
                return
            toggle_selection(section)
            click_state['last_toggle'] = (section, time.perf_counter())
        elif event.button == 3:  # right click: start pan
            pan_state['active'] = True
            pan_state['x0_px'] = event.x
            pan_state['y0_px'] = event.y
            pan_state['xlim0'] = ax.get_xlim()
            pan_state['ylim0'] = ax.get_ylim()

    def on_release(event):
        if event.button == 3 and pan_state['active']:
            pan_state['active'] = False
            # Throttled motion events can leave the view slightly behind
            # where the mouse actually stopped; snap to the exact final
            # position with one unconditional redraw, superseding any
            # pending trailing-timer redraw.
            if redraw_state['timer'] is not None:
                redraw_state['timer'].stop()
                redraw_state['timer'] = None
            moved_px = 0.0
            if 'last_px' in pan_state:
                moved_px = math.hypot(
                    pan_state['last_px'][0] - pan_state['x0_px'],
                    pan_state['last_px'][1] - pan_state['y0_px'],
                )
                apply_pan(*pan_state['last_px'])
                force_draw()
                del pan_state['last_px']
            # A right *click* (negligible movement between press and
            # release) opens the context menu instead of just ending a pan
            # — right-drag still pans as before.
            if moved_px < RIGHT_CLICK_DRAG_THRESHOLD_PX:
                show_grid_context_menu(event)

    def on_motion(event):
        if not pan_state['active'] or event.x is None or event.y is None:
            return
        pan_state['last_px'] = (event.x, event.y)
        apply_pan(event.x, event.y)
        throttled_draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    def on_scroll(event):
        # A single Axes holding the whole grid image, so zooming it zooms
        # every thumbnail together — thumbnails near the edges pan out of
        # view rather than each zooming independently.
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        scale = 1.25 if event.button == 'down' else 0.8  # 'up' zooms in
        x_left, x_right = ax.get_xlim()
        y_bottom, y_top = ax.get_ylim()
        new_x0 = event.xdata - (event.xdata - x_left) * scale
        new_x1 = event.xdata + (x_right - event.xdata) * scale
        new_y0 = event.ydata - (event.ydata - y_bottom) * scale
        new_y1 = event.ydata + (y_top - event.ydata) * scale
        new_x0, new_x1, new_y0, new_y1 = clamp_view(new_x0, new_x1, new_y0, new_y1)
        ax.set_xlim(new_x0, new_x1)
        ax.set_ylim(new_y0, new_y1)
        throttled_draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    result = {}

    def on_confirm(event):
        # Close immediately on click rather than prompting for whole-brain
        # confirmation here — asking via console input() while this window
        # stays open leaves it sitting on top of (or behind) the terminal
        # for however long the user takes to answer, which is exactly the
        # overlap/clutter this avoids. The whole-brain check instead happens
        # after this window is already gone, once show_figure_blocking()
        # returns below.
        result['selected'] = [s for s in unique_sections if s in selected]
        result['rois'] = list(rois)
        plt.close(fig)

    def on_cancel(event):
        # Don't call sys.exit() here — SystemExit raised inside a
        # Tkinter-dispatched button callback gets caught and swallowed by
        # Tkinter's own callback exception handler (it prints a traceback
        # and keeps the event loop running) rather than propagating out of
        # show_figure_blocking() below. Just flag it and exit after we're
        # back in plain script code.
        result['cancelled'] = True
        plt.close(fig)

    def on_load_rois(event):
        path = prompt_load_rois_file(initial_dir=out_folder)
        if not path:
            return
        try:
            loaded_rois, loaded_whole_sections = load_rois_csv(path)
        except Exception as e:
            print(f"Could not load ROIs from '{path}': {e}")
            return

        existing_keys = {(r['section'], r['x_min'], r['x_max'], r['y_min'], r['y_max']) for r in rois}
        added_sections = set()
        n_added = skipped_unknown = skipped_duplicate = 0
        for roi in loaded_rois:
            if roi['section'] not in layout:
                skipped_unknown += 1
                continue
            key = (roi['section'], roi['x_min'], roi['x_max'], roi['y_min'], roi['y_max'])
            if key in existing_keys:
                skipped_duplicate += 1
                continue
            rois.append(roi)
            existing_keys.add(key)
            added_sections.add(roi['section'])
            n_added += 1

        n_whole_added = 0
        for section in loaded_whole_sections:
            if section not in layout:
                skipped_unknown += 1
                continue
            if section not in selected:
                selected.add(section)
                added_sections.add(section)
                n_whole_added += 1

        for section in added_sections:
            redraw_roi_indicators(section)
            redraw_selection(section, section in selected)  # drop the red outline now that this section has an ROI
        if added_sections:
            fig.canvas.draw_idle()

        msg = (f"Loaded {n_added} ROI(s) and {n_whole_added} whole section(s) from {path} "
               f"across {len(added_sections)} section(s).")
        if skipped_unknown:
            msg += f" Skipped {skipped_unknown} for section(s) not in this grid."
        if skipped_duplicate:
            msg += f" Skipped {skipped_duplicate} already-present duplicate(s)."
        print(msg)

    def on_clear_rois(event):
        if not rois and not selected:
            print("No ROIs or whole-section picks to clear.")
            return
        n_rois, n_whole = len(rois), len(selected)
        # A modal Tkinter dialog rather than a console y/n prompt — unlike
        # the whole-brain confirmation (deliberately moved to after this
        # window closes, see on_confirm's comment), this action doesn't
        # close the window, so there's no good later point to ask in the
        # console instead; a popup on top of the still-open window is the
        # only sensible place, and confirming this specific irreversible
        # action deserves a bit more visual weight than a console prompt.
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            confirmed = messagebox.askyesno(
                "Clear all ROIs?",
                f"Clear {n_rois} ROI(s) and {n_whole} whole-section pick(s)? "
                "This cannot be undone.",
            )
            root.destroy()
        except Exception as e:
            print(f"Could not show confirmation dialog ({e}); not clearing.")
            return
        if not confirmed:
            return
        affected_sections = set(selected) | {roi['section'] for roi in rois}
        rois.clear()
        selected.clear()
        for section in affected_sections:
            redraw_roi_indicators(section)
            redraw_selection(section, False)
        fig.canvas.draw_idle()
        print(f"Cleared {n_rois} ROI(s) and {n_whole} whole-section pick(s).")

    button_height = 0.045
    # Shared with the single-section picker and the processing dialog — see
    # compute_ui_fontsize() — rather than recomputed per-window.
    button_fontsize = UI_BUTTON_FONTSIZE

    ROW1_Y = 0.01

    fig.subplots_adjust(top=AXES_TOP, bottom=AXES_BOTTOM, left=AXES_LEFT, right=AXES_RIGHT)
    # Four equal-width buttons with equal gaps between and around them,
    # spanning the same [AXES_LEFT, AXES_RIGHT] range as the axes margins above.
    button_width = 0.19
    gap = (axes_width_frac - 4 * button_width) / 5
    x1 = AXES_LEFT + gap
    x2 = x1 + button_width + gap
    x3 = x2 + button_width + gap
    x4 = x3 + button_width + gap

    confirm_ax = fig.add_axes([x1, ROW1_Y, button_width, button_height])
    confirm_button = Button(confirm_ax, 'Confirm Selection')
    confirm_button.label.set_fontsize(button_fontsize)
    confirm_button.on_clicked(on_confirm)

    load_rois_ax = fig.add_axes([x2, ROW1_Y, button_width, button_height])
    load_rois_button = Button(load_rois_ax, 'Load ROIs')
    load_rois_button.label.set_fontsize(button_fontsize)
    load_rois_button.on_clicked(on_load_rois)

    clear_rois_ax = fig.add_axes([x3, ROW1_Y, button_width, button_height])
    clear_rois_button = Button(clear_rois_ax, 'Clear ROIs')
    clear_rois_button.label.set_fontsize(button_fontsize)
    clear_rois_button.on_clicked(on_clear_rois)

    cancel_ax = fig.add_axes([x4, ROW1_Y, button_width, button_height])
    cancel_button = Button(cancel_ax, 'Exit')
    cancel_button.label.set_fontsize(button_fontsize)
    cancel_button.on_clicked(on_cancel)

    # --- Starting view for single-section windows. There are no controls for
    # this on the grid itself: each single-section window's own Groups/Gene/
    # Imputed Gene radio and gene box are the controls. Whatever a window is
    # left showing when it closes (see prompt_subregion_selection's
    # session_view_settings) becomes the starting point for the next section
    # opened, so switching to e.g. Gene 'Gad2' in one section carries over to
    # the next. Starts as Groups, with Sox14 pre-filled in the gene box.
    session_view_settings = {'mode': 'standard', 'gene': 'Sox14'}
    # Shared by reference with every single-section window this session
    # opens (see prompt_subregion_selection's session_color_scale param) —
    # toggling Linear/Log in one window updates this dict directly, so it's
    # remembered as the starting scale for the next section opened, rather
    # than resetting to 'log' every time. Defaults to 'log' (not 'linear')
    # since raw counts are heavily right-skewed/zero-inflated — log1p keeps
    # both dim and bright cells visible from the first render, not just
    # after manually switching.
    session_color_scale = {'mode': 'log'}

    fig.canvas.manager.set_window_title(
        "Click to select sections, double-click for the ROI picker, scroll to zoom/pan, "
        "then Confirm Selection"
    )
    center_figure_window(fig)
    show_figure_blocking(fig)

    if result.get('cancelled'):
        raise UserCancelledSelection("Section picker cancelled.")

    if 'selected' not in result:
        raise RuntimeError("section picker window was closed without confirming a selection")

    chosen = result['selected']
    chosen_rois = result['rois']

    if not chosen and not chosen_rois and not confirm_whole_brain():
        raise UserCancelledSelection("Whole-brain run not confirmed.")

    if not chosen:
        return None, 'allsections', chosen_rois, imputed_state['adata']
    if len(chosen) == len(unique_sections):
        return None, 'allsections', chosen_rois, imputed_state['adata']
    tokens = '-'.join(sanitize_section_token(s) for s in chosen)
    return chosen, f'sections-{tokens}', chosen_rois, imputed_state['adata']


def filter_by_sections(adata, section_series, selected_sections, section_col=SECTION_COL):
    """Subset adata to cells whose section label is in `selected_sections`
    (a list of labels), also stashing the section label onto adata.obs for
    downstream use. `selected_sections=None` means "all sections" (no-op)."""
    if section_series is not None and section_col not in adata.obs.columns:
        adata.obs[section_col] = section_series.reindex(adata.obs.index)
    if selected_sections is None:
        return adata
    if section_col not in adata.obs.columns:
        print(f"Warning: '{section_col}' column not available; cannot filter by section. "
              "Proceeding with all sections.")
        return adata
    keep = adata.obs[section_col].isin(selected_sections)
    filtered = materialize_subset(adata, keep.to_numpy())
    print(f"Filtered to {len(selected_sections)} section(s): {filtered.n_obs} of {adata.n_obs} cells kept.")
    return filtered


_expr_column_cache = {}


def _get_full_gene_column(expr_adata, gene_col, layer=None):
    """The full expression column for var-index `gene_col`, across every
    cell in `expr_adata`, read once and cached in memory (keyed by the
    AnnData object's identity + gene index + layer — safe for the process
    lifetime, since callers always pass a persistent, already-loaded
    AnnData like imputed_state['adata'] or adata_backed, never a fresh one
    per call). `layer=None` reads .X (imputed values); `layer='counts'`
    reads that layer instead (raw counts, for adata_backed).

    Exists because backed-mode random-row access — `expr_adata[positions,
    gene_col].X` for an arbitrary, scattered `positions` array (as needed
    to line cells up with a particular section's cells by ID) — is
    drastically slower than one bulk read: each scattered row triggers its
    own separate HDF5 round trip, however small. A single `[:, gene_col]`
    slice reads every row in one sequential-ish pass instead, and once
    that's cached here, every section's (and every later section's) lookup
    for the same gene becomes plain in-memory numpy indexing — instant."""
    key = (id(expr_adata), gene_col, layer)
    cached = _expr_column_cache.get(key)
    if cached is not None:
        return cached
    sliced = expr_adata[:, gene_col]
    col = sliced.X if layer is None else sliced.layers[layer]
    if hasattr(col, 'toarray'):
        col = col.toarray()
    col = np.asarray(col).ravel().astype(float)
    _expr_column_cache[key] = col
    return col


def find_gene_index(adata, query):
    """Case-insensitive lookup of `query` in adata.var['gene_symbol']. Returns the
    index of the matching var or None if there's no match."""
    query = (query or '').strip()
    if not query:
        return None

    gene_name_list = adata.var['gene_symbol'].tolist()
    query = query.casefold()
    return next((i for i, item in enumerate(gene_name_list) if item.casefold() == query), None)


# ==============================================================================
# Gene entry, shared by the single-section ROI picker and the interactive UMAP
# viewer. Both have a text box taking comma-separated gene names (1-3 here,
# via MAX_GENE_NAMES below), with autocomplete for the name being typed;
# everything about *entering* genes lives here so the two behave identically.
# Rendering stays per-window — the interactive UMAP viewer's own redraw_gene
# uses a higher, viewer-local cap (INTERACTIVE_MULTI_GENE_MAX_NAMES, up to 6)
# instead of MAX_GENE_NAMES, since it's the only renderer that does anything
# with genes 4-6 (drawn as a '+' overlay — see redraw_multi_genes).
# ==============================================================================

MAX_GENE_NAMES = 3
TOO_MANY_GENES_MESSAGE = f"Enter at most {MAX_GENE_NAMES} gene names, separated by commas."
# Color of a gene name that didn't resolve, shown in the box itself until the
# next edit.
GENE_NAME_ERROR_COLOR = 'red'
# Faded text/outline color for controls disabled while a window is busy. Only
# text and outlines fade, not backgrounds: a paler fill reads as highlighted
# rather than disabled.
DISABLED_CONTROL_COLOR = '0.65'


def split_gene_query(text):
    """Split a comma-separated gene query into (prefix, current_token):
    `prefix` is everything through the last comma plus a ', ' separator,
    ready to have a completed name appended; `current_token` is what's being
    typed after it, stripped. No comma: prefix is '' and the whole text is
    the token. Used for autocompleting only the name being typed, and for
    picking a suggestion without erasing names typed before it."""
    if ',' in text:
        prefix, _, current = text.rpartition(',')
        return prefix + ', ', current.strip()
    return '', text.strip()


def completed_gene_query(text, suggestion):
    """The box text after picking `suggestion` for the name being typed."""
    prefix, _current_token = split_gene_query(text)
    return prefix + suggestion


def parse_gene_names(text):
    """The non-empty, stripped, comma-separated names in `text`, in order."""
    return [g.strip() for g in (text or '').split(',') if g.strip()]


def resolve_gene_names(gene_source, names):
    """Look up each of `names` (case-insensitively) in gene_source.var.
    Returns (cols, canonical_names, missing): cols[i] is names[i]'s var index
    or None, canonical_names[i] its spelling from var['gene_symbol'] (or the
    name as typed if not found), and missing the names that weren't found, in
    order. A None gene_source (dataset not loaded) finds nothing."""
    if gene_source is None:
        return [None] * len(names), list(names), list(names)
    cols = [find_gene_index(gene_source, name) for name in names]
    canonical = [
        str(gene_source.var['gene_symbol'].iloc[col]) if col is not None else name
        for name, col in zip(names, cols)
    ]
    missing = [name for name, col in zip(names, cols) if col is None]
    return cols, canonical, missing


def rank_gene_suggestions(symbol_list, query, limit):
    """Autocomplete matches for `query` (case-insensitive): names starting
    with it first, then names merely containing it, capped at `limit`."""
    query = (query or '').casefold()
    if not query:
        return []
    starts = [g for g in symbol_list if g.casefold().startswith(query)]
    contains = [g for g in symbol_list if query in g.casefold() and not g.casefold().startswith(query)]
    return (starts + contains)[:limit]


def gene_symbol_list(adata, imputed_state, use_imputed):
    """The gene names to autocomplete against: the imputed dataset's (much
    larger) panel when that view is active and loaded, else `adata`'s."""
    if use_imputed and imputed_state is not None and imputed_state['adata'] is not None:
        return imputed_state['adata'].var['gene_symbol'].tolist()
    return adata.var['gene_symbol'].tolist()


def is_enter_submit(textbox):
    """Whether a TextBox 'submit' event came from pressing Enter.

    'submit' fires on two different paths: Enter (capturekeystrokes still
    True), and any click outside the box, which stops typing first
    (capturekeystrokes already False). A click on an autocomplete suggestion
    is one of those clicks, and each window's dropdown click handler selects
    the suggestion itself. So a submit handler should only act on Enter;
    acting on the click too would empty the dropdown before that handler
    could read which row was clicked."""
    return textbox.capturekeystrokes


def _gene_name_segments(text, bad_names):
    """Split `text` into drawable (segment, is_bad) pieces for per-name
    coloring. A segment never ends in whitespace: text extents are measured
    from ink, so a trailing space has no width and chaining the next piece
    from that edge would drop it. Whitespace is carried onto the start of
    the following piece instead, where it measures correctly."""
    bad = {b.casefold() for b in bad_names}
    pieces = []
    for piece in re.split(r'(,)', text):
        if not piece:
            continue
        if piece == ',':
            pieces.append((',', False))
            continue
        body = piece.rstrip()
        if body:
            pieces.append((body, body.strip().casefold() in bad))
        trailing = piece[len(body):]
        if trailing:
            pieces.append((trailing, False))
    segments, carry = [], ''
    for seg, is_bad in pieces:
        if not seg.strip():
            carry += seg
            continue
        segments.append((carry + seg, is_bad))
        carry = ''
    return segments


def mark_invalid_gene_names(textbox, bad_names):
    """Show the names in `bad_names` in GENE_NAME_ERROR_COLOR inside the box,
    leaving the rest in the box's normal text color. Clear with
    clear_gene_name_marks().

    A TextBox draws its text as one Text artist, in one color, so this makes
    that artist transparent (it still positions the typing cursor) and draws
    the text as a chain of pieces, each anchored at the right edge of the
    one before and colored per name. Measured against the normal single-color
    rendering, every glyph lands within 1 px at 200 dpi. The pieces are
    children of the box's axes, so they're drawn wherever the box is, blitted
    or not. Call again after changing the box's font size to rebuild them."""
    _remove_gene_name_mark_artists(textbox)
    bad_names = list(bad_names)
    textbox._gene_name_marks = {'bad': bad_names, 'artists': [], 'override_color': None}
    if not bad_names:
        textbox.text_disp.set_alpha(None)
        return
    text_disp = textbox.text_disp
    base_color = text_disp.get_color()
    text_disp.set_alpha(0.0)
    previous = None
    for segment, is_bad in _gene_name_segments(text_disp.get_text(), bad_names):
        style = dict(fontsize=text_disp.get_fontsize(), fontfamily=text_disp.get_fontfamily(),
                     color=GENE_NAME_ERROR_COLOR if is_bad else base_color, annotation_clip=False)
        if previous is None:
            artist = textbox.ax.annotate(segment, xy=text_disp.get_position(), xycoords=text_disp.get_transform(),
                                         va=text_disp.get_va(), ha='left', **style)
        else:
            artist = textbox.ax.annotate(segment, xy=(1, 0), xycoords=previous, va='bottom', ha='left', **style)
        artist._gene_name_is_bad = is_bad
        textbox._gene_name_marks['artists'].append(artist)
        previous = artist


def _remove_gene_name_mark_artists(textbox):
    marks = getattr(textbox, '_gene_name_marks', None)
    if marks:
        for artist in marks['artists']:
            artist.remove()
        marks['artists'] = []


def clear_gene_name_marks(textbox):
    """Undo mark_invalid_gene_names(): remove the colored pieces and show the
    box's own text again. Safe to call when nothing is marked."""
    _remove_gene_name_mark_artists(textbox)
    textbox._gene_name_marks = None
    textbox.text_disp.set_alpha(None)


def refresh_gene_name_marks(textbox):
    """Rebuild any current marks to match the box's current font size/text
    color (e.g. after a resize changed its font size). No-op if unmarked."""
    marks = getattr(textbox, '_gene_name_marks', None)
    if marks and marks['bad']:
        override = marks['override_color']
        mark_invalid_gene_names(textbox, marks['bad'])
        if override is not None:
            set_gene_name_marks_color(textbox, override)


def set_gene_name_marks_color(textbox, override_color):
    """Recolor any marked pieces: every piece `override_color` (e.g. faded
    while disabled), or None to restore normal/red coloring."""
    marks = getattr(textbox, '_gene_name_marks', None)
    if not marks:
        return
    marks['override_color'] = override_color
    base_color = textbox.text_disp.get_color()
    for artist in marks['artists']:
        if override_color is not None:
            artist.set_color(override_color)
        else:
            artist.set_color(GENE_NAME_ERROR_COLOR if artist._gene_name_is_bad else base_color)


def set_textbox_text_silent(textbox, value):
    """TextBox.set_val without firing its 'change'/'submit' observers, so
    restoring or canonicalizing text doesn't trigger a redraw or reopen
    autocomplete. Clears any invalid-name marks, since they describe the old
    text."""
    clear_gene_name_marks(textbox)
    was_on = textbox.eventson
    textbox.eventson = False
    try:
        textbox.set_val(value)
    finally:
        textbox.eventson = was_on


def render_gene_expression_array(adata, section_series, section_label, get_spatial, gene_name,
                                  cached_layout, img_w, img_h, dpi=150, on_progress=None,
                                  cell_type_selection='All', color_scale='log',
                                  expr_adata=None, dataset='standard'):
    """Render `gene_name`'s expression across `section_label`'s cells as an
    RGBA array, transparently cached to disk per section+gene+cell-type
    +color-scale (see gene_expression_cache_paths()) so re-showing one
    already rendered in an earlier session is instant. Positioned and sized
    to align pixel-for-pixel with that cached background: rather than
    letting matplotlib re-derive an axes box from figsize/aspect/data-
    limits (and hoping it reproduces the original exactly), the new axes'
    position is set directly from cached_layout['pixel_bbox'] (converted to
    figure fractions) and its data limits directly from
    cached_layout['data_xlim'/'data_ylim'] — so this image can be swapped
    in via the same imshow artist, over the same ROI overlays, with no
    separate coordinate mapping needed. Returns (rgba_array,
    resolved_gene_name, max_expression, color_vmin, color_vmax) —
    color_vmin/color_vmax are the actual range the colormap was scaled
    to (post color_scale transform, see below), for building a matching
    colorbar; max_expression is always in raw units regardless.

    `get_spatial` is a zero-arg callable returning the spatial coordinates
    dataframe (e.g. a lazy loader around load_section_spatial_coords()),
    not the dataframe itself — it's only actually called on a disk-cache
    miss, so a cache hit never pays for loading it (that load can take
    several seconds the first time, and callers may already have it cached
    from an earlier call anyway).

    `cell_type_selection`: the picker always has every cell type available
    at this stage (see prompt_section_selection_gui's own note — the
    Neurons/NonNeurons/All choice at the very start of the pipeline isn't
    applied to `adata` until well after the picker session ends), so this
    doesn't change *which* cells are plotted, only how — 'Neurons' greys
    out non-neuron cells (dimgray, ignoring their actual expression) rather
    than excluding them, so section outlines stay visible via the excluded
    type the way NON_NEURON_CLASS_IDS classification defines them;
    'NonNeurons' greys out neurons the same way; 'All' (the default)
    colors every cell by expression as before.

    `color_scale`: meaning depends on `dataset`. For 'standard' (the
    default `adata`/raw-count path), 'linear' colors cells by their raw
    expression value — not log-transformed here (log1p/normalize_total
    don't run until well after the picker session, on the filtered/
    subsampled adata — see around Step 3); 'log' colors by log1p
    (expression) instead, compressing the (often heavily right-skewed,
    zero-inflated) raw count range so both dim and bright cells stay
    visible — chosen over a LogNorm on the raw values specifically to
    sidestep LogNorm's requirement that every value be strictly positive,
    which raw counts (mostly zeros) aren't. For any other `dataset` (e.g.
    'imputed'), `expr_adata`'s values are already log2-transformed on disk
    — log1p'ing them again would double-transform (and can even yield NaN,
    since log1p needs x > -1 while log2 values can be negative) — so the
    toggle flips instead: 'log' shows those on-disk values as-is, 'linear'
    undoes the presumed log2(x + 1) transform (2**x - 1) back to
    approximate linear-scale expression.

    `on_progress`, if given, is called with a fraction in [0, 1] at each
    stage boundary — coarse (fixed checkpoints, not time-weighted) but
    enough for a live status readout during the slower steps (pulling the
    expression column and rasterizing the scatter).

    `expr_adata`, if given (e.g. the imputed-gene-expression dataset — see
    load_imputed_adata), is used instead of `adata` for the gene lookup and
    expression values themselves, while `adata`/`section_series`/`spatial`
    still supply which cells belong to this section and where they sit —
    `expr_adata` is a separate AnnData that isn't guaranteed to share
    `adata`'s cell order (or even its full cell set), so cells are matched
    between the two by ID (obs_names), not by position; any of this
    section's cells missing from `expr_adata` get NaN expression (plotted,
    but effectively invisible at viridis's low end) rather than raising.
    `dataset` should describe `expr_adata` (e.g. 'imputed') for the on-disk
    cache key (see gene_expression_cache_paths) — a given gene name means
    different values in a different dataset, so they can't share a cache
    entry. Omit both (the defaults) to look up and plot straight from
    `adata`, as before.

    Raises ValueError if the gene isn't found, spatial coordinates aren't
    available, or there's no data to plot."""
    def report(frac):
        if on_progress is not None:
            on_progress(frac)

    report(0.0)
    gene_source = adata if expr_adata is None else expr_adata
    gene_col = find_gene_index(gene_source, gene_name)
    if gene_col is None:
        raise ValueError(f"gene '{gene_name}' not found in this dataset")

    resolved_gene = gene_source.var['gene_symbol'].iloc[gene_col]

    cache_png, cache_json = gene_expression_cache_paths(
        section_label, resolved_gene, cell_type_selection, color_scale, dataset,
    )
    if cache_png.exists() and cache_json.exists():
        try:
            rgba = np.asarray(Image.open(cache_png).convert('RGBA'), dtype=np.uint8)
            with open(cache_json) as f:
                cached_json = json.load(f)
            report(1.0)
            return (rgba, resolved_gene, cached_json['max_expr'],
                    cached_json['color_vmin'], cached_json['color_vmax'])
        except Exception:
            pass  # corrupt/partial cache — fall through and regenerate

    mask = (section_series == section_label).to_numpy()
    if not mask.any():
        raise ValueError(f"no cells found for section {section_label}")

    report(0.15)
    spatial = get_spatial()
    if spatial is None:
        raise ValueError("spatial coordinates not available")
    xs = pd.to_numeric(spatial['x'], errors='coerce').to_numpy()[mask]
    ys = pd.to_numeric(spatial['y'], errors='coerce').to_numpy()[mask]
    valid = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[valid], ys[valid]
    if len(xs) == 0:
        raise ValueError(f"no spatial coordinates for section {section_label}")

    row_indices = np.where(mask)[0][valid]
    report(0.3)
    if expr_adata is None:
        expr = adata[row_indices, gene_col].X  # slowest step for backed data — a disk read
        if hasattr(expr, 'toarray'):
            expr = expr.toarray()
        expr = np.asarray(expr).ravel().astype(float)
    else:
        # expr_adata doesn't necessarily share adata's cell order/set (see
        # this function's own docstring) — matched here by cell ID rather
        # than by position. Cells adata has but expr_adata doesn't get NaN
        # expression rather than raising.
        cell_ids = adata.obs_names[row_indices]
        expr_positions = expr_adata.obs_names.get_indexer(cell_ids)
        found = expr_positions >= 0
        expr = np.full(len(cell_ids), np.nan, dtype=float)
        if found.any():
            # See _get_full_gene_column's own docstring: this reads (and
            # caches) the whole gene's column once, up front, instead of
            # the scattered per-cell backed reads that made this line
            # extremely slow — expr_positions[found] then just indexes an
            # in-memory numpy array.
            expr[found] = _get_full_gene_column(expr_adata, gene_col)[expr_positions[found]]
    report(0.6)

    # See this function's own docstring: the picker still has every cell
    # type at this point, so 'Neurons'/'NonNeurons' don't exclude anything
    # here — they grey out the *other* type instead, ignoring its actual
    # expression, while leaving it in place (as opposed to actually
    # excluding it, which was the original ask — but non-neurons are what
    # keep section outlines filled in where neurons alone leave gaps, so
    # dropping them here would have undone that).
    grey_mask = np.zeros(len(xs), dtype=bool)
    if cell_type_selection in ('Neurons', 'NonNeurons') and 'class' in spatial.columns:
        class_ids = extract_leading_numeric_id(spatial['class']).to_numpy()[row_indices]
        is_non_neuron = np.isin(class_ids, list(NON_NEURON_CLASS_IDS))
        grey_mask = is_non_neuron if cell_type_selection == 'Neurons' else ~is_non_neuron

    fig = Figure(figsize=(img_w / dpi, img_h / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    # Black rather than the usual white — viridis's low end is a dark
    # purple, which all but disappears against white, making low-expressing
    # cells look like background rather than data. Both the figure's own
    # patch and the axes' facecolor need setting: the axes box only covers
    # pixel_bbox (a couple percent smaller than the full canvas, matching
    # the cached background image's own small margin), so leaving the
    # figure white would show through as a thin white border around an
    # otherwise-black plot.
    fig.patch.set_facecolor('black')
    px_x0, px_x1, px_y0, px_y1 = cached_layout['pixel_bbox']
    ax = fig.add_axes([px_x0 / img_w, px_y0 / img_h, (px_x1 - px_x0) / img_w, (px_y1 - px_y0) / img_h])
    ax.set_facecolor('black')
    if grey_mask.any():
        # Drawn first/underneath — greyed-out cells are context, not data.
        ax.scatter(xs[grey_mask], ys[grey_mask], c=[[0.15, 0.15, 0.15]], s=6, linewidths=0)
    colored = ~grey_mask
    color_values = expr[colored]
    if dataset == 'standard':
        if color_scale == 'log':
            color_values = np.log1p(color_values)
        # else 'linear': raw counts as-is.
    else:
        # expr_adata's values (e.g. the imputed dataset) are already log2-
        # transformed on disk — see this function's docstring on
        # `color_scale` for why the toggle's meaning flips here instead of
        # just log1p'ing on top of that (which would double-transform).
        if color_scale == 'linear':
            color_values = np.exp2(color_values) - 1.0
        # else 'log': on-disk log2 values as-is.
    order = np.argsort(color_values)  # plot highest-expressing cells last/on top
    color_vmin = float(np.nanmin(color_values)) if len(color_values) else 0.0
    color_vmax = float(np.nanmax(color_values)) if len(color_values) else 1.0
    ax.scatter(xs[colored][order], ys[colored][order], c=color_values[order],
               s=6, cmap='viridis', vmin=color_vmin, vmax=color_vmax, linewidths=0)
    ax.set_xlim(cached_layout['data_xlim'])
    ax.set_ylim(cached_layout['data_ylim'])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    report(0.75)
    fig.canvas.draw()  # rasterizing the scatter — the other slow step, for large point counts
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    max_expr = float(np.nanmax(expr)) if len(expr) else 0.0
    report(1.0)

    try:
        Image.fromarray(rgba, mode='RGBA').save(cache_png)
        with open(cache_json, 'w') as f:
            json.dump({'max_expr': max_expr, 'color_vmin': color_vmin, 'color_vmax': color_vmax}, f)
    except Exception as e:
        print(f"Warning: could not cache gene expression image for '{resolved_gene}': {e}")

    return rgba, resolved_gene, max_expr, color_vmin, color_vmax


def render_multi_gene_expression_array(adata, section_series, section_label, get_spatial, gene_names,
                                        cached_layout, img_w, img_h, dpi=150, on_progress=None,
                                        cell_type_selection='All', color_scale='log',
                                        expr_adata=None, dataset='standard'):
    """Multi-gene (2 or 3) overlay variant of render_gene_expression_array:
    gene_names[0] drives red, [1] green, [2] (if given) blue, blended via
    multi_gene_rgb (module-level, shared with the interactive UMAP viewer's
    own multi-gene mode — same MULTI_GENE_BRIGHT_BLUE convention) from a
    black baseline toward each channel's own full saturation as that gene's
    own normalized expression rises. Otherwise mirrors render_gene_
    expression_array's structure closely — same disk-cache pattern (see
    gene_expression_cache_paths, keyed on the genes joined together), same
    section/background/cell-type-grey handling — kept as its own function
    rather than folded into that one since there's no single scalar value/
    colormap to reuse once colors come from up to three values apiece.

    Returns (rgba_array, resolved_gene_names, max_expressions, color_vmins,
    color_vmaxes) — the last three are lists, one entry per gene, in the
    same order as gene_names (already-normalized [0,1] vmin/vmax, i.e. the
    actual data range each channel was scaled from — same meaning as
    normalize_expression's own return in the interactive UMAP viewer).

    Raises ValueError if any gene isn't found, spatial coordinates aren't
    available, or there's no data to plot."""
    def report(frac):
        if on_progress is not None:
            on_progress(frac)

    report(0.0)
    gene_source = adata if expr_adata is None else expr_adata
    gene_cols = [find_gene_index(gene_source, name) for name in gene_names]
    missing = [name for name, col in zip(gene_names, gene_cols) if col is None]
    if missing:
        raise ValueError(f"gene(s) not found in this dataset: {', '.join(missing)}")
    resolved_genes = [str(gene_source.var['gene_symbol'].iloc[c]) for c in gene_cols]

    cache_png, cache_json = gene_expression_cache_paths(
        section_label, '+'.join(resolved_genes), cell_type_selection, color_scale, dataset,
    )
    if cache_png.exists() and cache_json.exists():
        try:
            rgba = np.asarray(Image.open(cache_png).convert('RGBA'), dtype=np.uint8)
            with open(cache_json) as f:
                cached_json = json.load(f)
            report(1.0)
            return (rgba, resolved_genes, cached_json['max_exprs'],
                    cached_json['color_vmins'], cached_json['color_vmaxes'])
        except Exception:
            pass  # corrupt/partial cache — fall through and regenerate

    mask = (section_series == section_label).to_numpy()
    if not mask.any():
        raise ValueError(f"no cells found for section {section_label}")

    report(0.15)
    spatial = get_spatial()
    if spatial is None:
        raise ValueError("spatial coordinates not available")
    xs = pd.to_numeric(spatial['x'], errors='coerce').to_numpy()[mask]
    ys = pd.to_numeric(spatial['y'], errors='coerce').to_numpy()[mask]
    valid = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[valid], ys[valid]
    if len(xs) == 0:
        raise ValueError(f"no spatial coordinates for section {section_label}")

    row_indices = np.where(mask)[0][valid]
    report(0.3)

    def gene_expr(gene_col):
        # Same per-gene extraction as render_gene_expression_array's own
        # single-gene branch, just factored out so it can run once per gene.
        if expr_adata is None:
            e = adata[row_indices, gene_col].X
            if hasattr(e, 'toarray'):
                e = e.toarray()
            return np.asarray(e).ravel().astype(float)
        cell_ids = adata.obs_names[row_indices]
        expr_positions = expr_adata.obs_names.get_indexer(cell_ids)
        found = expr_positions >= 0
        e = np.full(len(cell_ids), np.nan, dtype=float)
        if found.any():
            e[found] = _get_full_gene_column(expr_adata, gene_col)[expr_positions[found]]
        return e

    exprs = [gene_expr(c) for c in gene_cols]
    report(0.5)

    grey_mask = np.zeros(len(xs), dtype=bool)
    if cell_type_selection in ('Neurons', 'NonNeurons') and 'class' in spatial.columns:
        class_ids = extract_leading_numeric_id(spatial['class']).to_numpy()[row_indices]
        is_non_neuron = np.isin(class_ids, list(NON_NEURON_CLASS_IDS))
        grey_mask = is_non_neuron if cell_type_selection == 'Neurons' else ~is_non_neuron
    colored = ~grey_mask

    def transform(values):
        # Same dataset-dependent scale-toggle meaning as render_gene_
        # expression_array's own docstring explains for `color_scale`.
        if dataset == 'standard':
            return np.log1p(values) if color_scale == 'log' else values
        return (np.exp2(values) - 1.0) if color_scale == 'linear' else values

    def normalize(values):
        # Same min-max-to-[0,1] convention as the interactive UMAP viewer's
        # own normalize_expression (NaN preserved — multi_gene_rgb treats
        # that as "0 expression" for that channel).
        vmin = float(np.nanmin(values)) if len(values) else 0.0
        vmax = float(np.nanmax(values)) if len(values) else 1.0
        if vmax > vmin:
            return (values - vmin) / (vmax - vmin), vmin, vmax
        return np.zeros_like(values), vmin, vmax

    normalized = [normalize(transform(e[colored])) for e in exprs]
    norms = [n[0] for n in normalized]
    color_vmins = [n[1] for n in normalized]
    color_vmaxes = [n[2] for n in normalized]
    report(0.65)

    rgb = np.clip(np.nan_to_num(multi_gene_rgb(norms, (0.0, 0.0, 0.0)), nan=0.0), 0.0, 1.0)

    fig = Figure(figsize=(img_w / dpi, img_h / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    # Black, same reasoning as render_gene_expression_array's own — and the
    # same baseline multi_gene_rgb was given above, so a genuinely
    # zero-expression cell (all three channels at their baseline) actually
    # disappears into it rather than reading as a dim but visible dot.
    fig.patch.set_facecolor('black')
    px_x0, px_x1, px_y0, px_y1 = cached_layout['pixel_bbox']
    ax = fig.add_axes([px_x0 / img_w, px_y0 / img_h, (px_x1 - px_x0) / img_w, (px_y1 - px_y0) / img_h])
    ax.set_facecolor('black')
    if grey_mask.any():
        ax.scatter(xs[grey_mask], ys[grey_mask], c=[[0.15, 0.15, 0.15]], s=6, linewidths=0)
    # Brightest-overall-last/on-top (sum of channels), same "don't let a dim
    # cell drawn later hide a bright one drawn earlier" reasoning as the
    # single-gene version's own `order` — there's no single value to sort
    # by here, so overall brightness stands in for it.
    order = np.argsort(rgb.sum(axis=1))
    ax.scatter(xs[colored][order], ys[colored][order], c=rgb[order], s=6, linewidths=0)
    ax.set_xlim(cached_layout['data_xlim'])
    ax.set_ylim(cached_layout['data_ylim'])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    report(0.85)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    max_exprs = [float(np.nanmax(e)) if len(e) else 0.0 for e in exprs]
    report(1.0)

    try:
        Image.fromarray(rgba, mode='RGBA').save(cache_png)
        with open(cache_json, 'w') as f:
            json.dump({'max_exprs': max_exprs, 'color_vmins': color_vmins, 'color_vmaxes': color_vmaxes}, f)
    except Exception as e:
        print(f"Warning: could not cache multi-gene expression image for {resolved_genes}: {e}")

    return rgba, resolved_genes, max_exprs, color_vmins, color_vmaxes


def compute_section_layout(section_series, section_label, spatial, figsize, dpi=150):
    """Computes the same pixel_bbox/data_xlim/data_ylim (plus img_w/img_h)
    that generate_and_cache_section_image() would write to its cache JSON
    for this section — without actually rendering or saving anything.
    Used when only *where things go* is needed (so a gene-expression image
    can be positioned/sized identically), not the class-colored pixel
    content itself — e.g. skipping that PNG entirely when a picker is about
    to open straight into the gene view and would never show it anyway.

    Cheap by construction, not just in effect: matplotlib's autoscale and
    'equal'-aspect handling only need each point's *position* registered
    (a plain scatter call updates the axes' bookkeeping without rendering
    any pixels) plus ax.apply_aspect() — confirmed empirically to produce
    pixel-identical pixel_bbox/data_xlim/data_ylim to a full
    fig.canvas.draw() (which is what actually costs something, scaling
    with point count), without paying for one. img_w/img_h likewise come
    straight from fig.canvas.get_width_height() rather than a save+reload
    round trip through disk.

    Raises RuntimeError if spatial x/y coordinates aren't available, or if
    no cells fall in this section."""
    if spatial is None or 'x' not in spatial.columns or 'y' not in spatial.columns:
        raise RuntimeError("spatial x/y coordinates not available")

    mask = (section_series == section_label).to_numpy()
    section_spatial = spatial[mask]
    xs = pd.to_numeric(section_spatial['x'], errors='coerce').to_numpy()
    ys = pd.to_numeric(section_spatial['y'], errors='coerce').to_numpy()
    valid = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[valid], ys[valid]
    if len(xs) == 0:
        raise RuntimeError(f"no spatial coordinates for section {section_label}")

    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.scatter(xs, ys, s=3)  # content is irrelevant here — only the resulting axes limits/position matter
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)  # must match generate_and_cache_section_image
    ax.apply_aspect()
    img_w, img_h = fig.canvas.get_width_height()
    frac = ax.get_position()
    pixel_bbox = [frac.x0 * img_w, frac.x1 * img_w, frac.y0 * img_h, frac.y1 * img_h]
    layout = {
        'pixel_bbox': pixel_bbox,
        'data_xlim': list(ax.get_xlim()),
        'data_ylim': list(ax.get_ylim()),
    }
    return layout, img_w, img_h


def prompt_subregion_selection(adata, abc_cache, section_series, section_label,
                                existing_rois=None, on_ready=None, spatial=None, initial_view=None,
                                cell_type_selection='All', precomputed_gene_open=None,
                                session_color_scale=None, shared_spatial_cache=None, imputed_state=None,
                                session_view_settings=None):
    """Single-section ROI picker: draw, resize, move, and delete one or more
    rectangular subregions of `section_label`'s cells without closing the
    window between them. Every ROI — pre-existing or added this session —
    is shown at all times with 8 pink handles (4 corners + 4 edge
    midpoints); drag any handle to resize, drag its interior to move it, or
    drag empty space to draw a new one. Right-click an ROI for a context
    menu to delete it. Nothing is finalized until Done is clicked (Cancel
    discards everything from this session, including edits/deletes of
    pre-existing ROIs). Bounds are (x_min, x_max, y_min, y_max) in the same
    'x'/'y' coordinate space as the ABC atlas metadata. Raises RuntimeError
    if a GUI can't be shown, or if `section_label`'s background image
    hasn't already been cached (see generate_and_cache_section_image() —
    the caller is responsible for calling that first; this function only
    ever reads the cache, so its own setup work stays fast and safe to run
    on the main thread).

    The bottom row also has a Groups/Gene/Imputed Gene radio plus a gene-
    name text box — picking Gene or Imputed Gene (or pressing Enter/picking
    an autocomplete suggestion while one of those is already selected)
    renders that gene's expression across this section in place of the
    standard class-colored view (see render_gene_expression_array()).
    Entering up to three comma-separated gene names instead shows them as a
    red/green/blue(-lightened) overlay (see render_multi_gene_expression_
    array() and the interactive UMAP viewer's own identical convention).
    This is unrelated to ROI editing and doesn't affect what gets returned;
    it's purely a different way to look at the same section while deciding
    where to draw ROIs.

    `existing_rois`, if given, is a list of (index, (x_min, x_max, y_min,
    y_max)) pairs already collected for this section (index = that ROI's
    position in the caller's full rois list); the caller should apply the
    returned actions (see below) against that same list.

    `on_ready`, if given, is called with no arguments right before the
    picker window is actually shown — e.g. to close a still-open
    "Processing..." dialog exactly when there's something to replace it
    with, instead of leaving a visible gap between the dialog closing and
    this window appearing (reading the cached image back and setting up
    the window isn't instant either).

    `spatial`, if given, is a pre-loaded load_section_spatial_coords()
    result (as already held by the caller's spatial_cache) — passed through
    just so 'Show Gene' doesn't have to re-read the multi-million-row
    cell_metadata CSV; only loaded fresh here (once, on first use) if not
    given, since most ROI-picking sessions never touch the gene view at all.

    `initial_view`, if given, is a {'mode': 'standard'/'gene', 'gene': name
    or None} dict (e.g. the grid picker's own gene box/buttons, which just
    set this as a default rather than re-rendering every thumbnail) — when
    mode is 'gene', this window opens already showing that gene's
    expression instead of the standard class-colored view, exactly as if
    'Show Gene' had been clicked with that name typed in.

    `cell_type_selection` — 'Neurons'/'NonNeurons'/'All', the same choice
    prompt_cell_type_selection() collects at the very start of the pipeline
    — is passed straight through to render_gene_expression_array (see its
    own docstring for what it actually changes about the gene view).

    `precomputed_gene_open`, if given, takes priority over `initial_view`
    and skips reading a cached class-colored background PNG from disk
    entirely — {'layout': ..., 'img_w': ..., 'img_h': ..., 'rgba': ...,
    'resolved_gene': ..., 'max_expr': ...} (layout/img_w/img_h from
    compute_section_layout(); rgba/resolved_gene/max_expr from
    render_gene_expression_array()), all computed by the caller *before*
    this window is ever created. Meant for the case where the caller
    already knows this window is going to open straight into the gene
    view: rendering the section's class-colored background first, only to
    immediately replace it, would otherwise cost roughly the same as the
    actual gene render for no visible benefit. The window opens already
    showing that gene's expression, radio set to 'Gene' — 'Standard View'
    still works afterward, it just renders (and disk-caches, same as
    always) the real class-colored background on first use instead of
    already having it.

    `session_color_scale`, if given, is a {'mode': 'linear'/'log'} dict
    used *by reference*, not copied — toggling the Linear/Log radio in this
    window mutates it directly, so a caller reusing the same dict across
    multiple windows (e.g. the section-grid picker, across double-clicks)
    gets it remembered as the starting scale for the next one, instead of
    resetting to the default every time. Omit for a one-off, non-persisted
    'log' default.

    `shared_spatial_cache`, if given, is the caller's own spatial_cache
    dict (see ensure_spatial_load_started) — used, not copied, so this
    window's hover-cell-info and gene-expression-view code wait on
    whatever background load is already in flight (started when the grid
    picker opened, or by an earlier double-click) instead of each starting
    its own redundant load the first time either feature is actually used.
    Omit to fall back to loading independently, synchronously, on first
    use — the original behavior.

    `imputed_state`, if given, is the shared {'adata':, 'load_thread':,
    'load_error':} dict (see ensure_imputed_gene_dataset_loaded) that the
    caller keeps across windows (and sessions) — passed through, not copied,
    so a load already in flight (or already done) from an earlier window is
    reused here instead of triggered again. This window's Show radio is
    3-way (Groups/Gene/Imputed Gene), switchable at any time: picking
    'Imputed Gene' here loads the dataset on demand (via
    ensure_imputed_gene_dataset_loaded's warn-then-load flow) if it isn't
    already, then pulls
    expression from `imputed_state['adata']` (matched to this window's
    cells by cell ID, not position — see render_gene_expression_array's own
    `expr_adata` param) instead of from `adata`, with the gene textbox's
    autocomplete list switching to its (far larger) gene panel too. Omit to
    disable the option entirely (selecting it just fails, as if the load
    itself had failed).

    `session_view_settings`, if given, is a {'mode': 'standard'/'gene'/
    'imputed_gene', 'gene': text} dict used *by reference*: when this window
    closes (Done, Cancel, or the window's close button), the view actually
    showing and the gene box's text are written back into it. The section-
    grid picker opens every section from the same dict, so the next section
    starts with whatever the previous one was left on. Only the view that
    actually displayed is saved, so a switch that failed (e.g. a gene not
    found) doesn't carry over. Omit to save nothing.

    Return value: a list of action dicts, built once when Done is clicked
    from whatever's on screen at that point (empty if Cancel was clicked,
    or if Done was clicked with nothing ever added) —
    {'action': 'add', 'bounds': (...)} for a brand new ROI,
    {'action': 'edit', 'index': i, 'bounds': (...)} for an edited existing
    one, or {'action': 'delete', 'index': i} for a deleted existing one (no
    'bounds' — nothing to apply, the caller should just remove rois[i])."""
    backend = matplotlib.get_backend().lower()
    if backend in ('agg', 'pdf', 'svg', 'ps', 'template', 'cairo'):
        raise RuntimeError(f"non-interactive matplotlib backend '{backend}'")

    # class_image_ready False means img is *not* a real class-colored
    # render — see precomputed_gene_open above — so on_standard_view has to
    # actually generate (and disk-cache) one on first use instead of
    # already having it.
    class_image_ready = precomputed_gene_open is None
    if precomputed_gene_open is not None:
        cached_layout = precomputed_gene_open['layout']
        img_w, img_h = precomputed_gene_open['img_w'], precomputed_gene_open['img_h']
        img = precomputed_gene_open['rgba']
    else:
        # The cached background image is always expected to already exist
        # by this point — generate_and_cache_section_image() is
        # responsible for creating it (in a background thread, so the
        # caller's 'Processing...' dialog can stay responsive), not this
        # function. Splitting the two apart is what makes that dialog's
        # Cancel button and progress bar actually work for section
        # rendering, not just for the one-time spatial-CSV load.
        cache_png, cache_json = section_roi_cache_paths(section_label)
        cached_layout = load_valid_section_roi_cache(section_label)
        if cached_layout is None:
            raise RuntimeError(f"no cached image found for section {section_label}")

        print(f"Loading cached section image for {section_label}...")
        img = plt.imread(cache_png)
        img_h, img_w = img.shape[0], img.shape[1]

    # Margins for the main (image) axes, as figure fractions — bottom is
    # taller than the others to leave room for the Done/Cancel row. Reused
    # below for both the figsize calculation and the actual subplots_adjust
    # call, so they can't drift out of sync with each other. Explicitly
    # setting all four (not just bottom, as before) matters: matplotlib's
    # *defaults* for the other three are individually much more generous
    # (roughly left=0.125, right=0.9, top=0.88) than anything intended here,
    # which is what produced a large blank margin around the plotted image.
    # Bottom is taller than before (0.14 vs. 0.1) to fit the "Show:" label,
    # Groups/Gene radio, and gene-name text box alongside Group/Done/Cancel
    # in one row without crowding.
    AXES_LEFT, AXES_RIGHT, AXES_BOTTOM, AXES_TOP = 0.005, 0.995, 0.14, 0.995
    axes_width_frac = AXES_RIGHT - AXES_LEFT
    axes_height_frac = AXES_TOP - AXES_BOTTOM

    # imshow keeps the image's own aspect ratio and pads the rest of its
    # axes box with blank space if the box's aspect doesn't match (which a
    # fixed 15:10 figsize, ignoring the image's actual aspect ratio, did) —
    # that padding, plus the axes spine drawn around the (larger, padded)
    # box rather than snug against the image, is what read as two borders
    # with a lot of space between them and around them. Picking the
    # figure's aspect ratio so the axes box's aspect exactly matches the
    # image's means there's no padding to begin with.
    fig_aspect = (img_w / img_h) * (axes_height_frac / axes_width_frac)
    figsize = compute_figsize_for_screen_height(fig_aspect, default=(15, 10))

    fig, ax = plt.subplots(figsize=figsize)
    normalize_tk_scaling(fig)
    # Opens full-screen rather than sized via compute_figsize_for_screen_
    # height's height_frac like the other pickers — this is the window
    # people spend the most time in (drawing/adjusting ROIs), where more
    # screen space directly helps. figsize above still matters: it's what
    # the image axes' aspect-ratio math (recompute_image_axes_position
    # below) starts from before Tk finishes applying 'zoomed'.
    maximize_figure_window(fig)
    # Forced early (rather than waiting for show_figure_blocking's own
    # show() call much later) so the window's *real*, full-screen size can
    # actually be read back below — an unmapped/withdrawn Tk window doesn't
    # reliably report its real geometry from update_idletasks() alone.
    try:
        fig.canvas.manager.show()
        fig.canvas.manager.window.update_idletasks()
    except Exception:
        pass
    # AXES_BOTTOM (and button_height, set further below) were tuned as
    # figure-*fractions* against the modest window figsize computed just
    # above — sensible when this window was sized via compute_figsize_for_
    # screen_height like the other pickers, but now that it opens full-
    # screen, the same fractions reserve a proportionally much bigger
    # *absolute* pixel margin at the bottom than the (font-size-capped)
    # button row actually needs — which also means recompute_image_axes_
    # position gets correspondingly less vertical room to work with, and
    # (since it derives the image's width from that available height to
    # preserve aspect ratio) unnecessarily shrinks the image's width too,
    # padding the left/right margins wider than they need to be. Scaling
    # AXES_BOTTOM down by how much taller this window actually ended up
    # than `figsize` keeps its *absolute* on-screen size the same as
    # originally tuned, instead of ballooning with the window.
    full_screen_shrink_scale = 1.0
    try:
        real_h_in = fig.get_size_inches()[1]
        if real_h_in > 0:
            full_screen_shrink_scale = min(1.0, figsize[1] / real_h_in)
            # AXES_BOTTOM measures the *margin itself* (from y=0 up to where
            # the image starts) — scaling it directly down is what keeps its
            # absolute size constant while giving the image the freed-up
            # room. Scaling the gap between AXES_BOTTOM and AXES_TOP instead
            # (an earlier version of this fix) does the opposite: shrinking
            # that gap pushes AXES_BOTTOM *up*, handing the image *less*
            # room, not more.
            AXES_BOTTOM = AXES_BOTTOM * full_screen_shrink_scale
            axes_height_frac = AXES_TOP - AXES_BOTTOM
    except Exception:
        pass


    # Kept so on_show_gene/on_standard_view (now triggered by the Groups/
    # Gene radio) can swap the displayed array in-place
    # (img_artist.set_data(...)) instead of clearing and redrawing the
    # whole axes (which would also have to redraw every ROI overlay).
    img_artist = ax.imshow(img, extent=(0, img_w, 0, img_h), origin='upper', interpolation='nearest')
    # 'auto' (rather than imshow's default 'equal') fills the axes box
    # exactly regardless of any small mismatch between the box's aspect
    # ratio and the image's — 'equal' instead expands the *data* range to
    # preserve aspect when the two don't match exactly, which left the
    # (fixed-size) image visibly smaller than its box despite fig_aspect
    # above already targeting a close match.
    ax.set_aspect('auto')
    # Anywhere within the axes' current xlim/ylim that imshow's own extent
    # doesn't cover (the padding added by compute_padded_canvas_bounds
    # below) shows this facecolor, not the figure's own background — same
    # black as the section images themselves, so the padding reads as part
    # of the canvas rather than as an empty gap.
    ax.set_facecolor('black')
    ax.set_xlim(0, img_w)
    ax.set_ylim(0, img_h)
    ax.set_xticks([])
    ax.set_yticks([])

    # Tracks the current *un-zoomed* full view: the image itself plus
    # whatever black padding compute_padded_canvas_bounds below has added
    # to match the axes box's aspect ratio. clamp_zoom_view uses this (not
    # img_w/img_h directly) as the zoom-out/pan limit, which is what makes
    # that padding real, pannable/zoomable data-space canvas rather than a
    # fixed backdrop the view can never reach past. Recomputed on every
    # resize (see recompute_image_axes_position) since the box's aspect can
    # genuinely change after the window's already open — e.g. maximizing,
    # zooming in, then un-maximizing back to a narrower window.
    canvas_state = {'xlim': (0, img_w), 'ylim': (0, img_h)}

    def compute_padded_canvas_bounds(box_aspect):
        """(xlim, ylim) for the full, un-zoomed canvas: the image, centered,
        padded with extra data-space on whichever axis has less room so the
        canvas's own aspect ratio exactly matches `box_aspect` (the axes
        box's on-screen width/height ratio) — letting 'auto' aspect fill the
        box with zero distortion, since the data range's ratio already
        matches the box's, instead of the previous approach of shrinking the
        *box* itself to match the image (which is what left blank figure
        background — now black canvas — on the sides in the first place)."""
        image_aspect = img_w / img_h
        if box_aspect > image_aspect:
            canvas_h = img_h
            canvas_w = img_h * box_aspect
        else:
            canvas_w = img_w
            canvas_h = img_w / box_aspect
        cx, cy = img_w / 2, img_h / 2
        return (cx - canvas_w / 2, cx + canvas_w / 2), (cy - canvas_h / 2, cy + canvas_h / 2)

    def rescale_view_to_aspect(x0, x1, y0, y1, box_aspect):
        """Adjust (x0,x1,y0,y1) — preserving its center and whichever of its
        two dimensions is the *limiting* one — so its own aspect ratio
        matches `box_aspect`, by expanding (never shrinking) the other
        dimension. Used to correct the *current* view (whatever it is —
        zoomed in, panned, or the full canvas) after a real resize changes
        the box's own aspect, rather than resetting the view back to the
        full canvas every time: an earlier version of this fix only ever
        set the view once (on a "haven't zoomed yet" flag) and then left it
        alone, which correctly avoided disrupting an in-progress zoom during
        this window's own initial multi-step maximize sequence, but also
        meant a later *genuine* resize (e.g. maximize, zoom in, then
        un-maximize) never re-corrected the now-stale view — the image
        rendered stretched into whatever the old aspect ratio no longer
        matched the window's new shape."""
        width, height = x1 - x0, y1 - y0
        if width <= 0 or height <= 0 or box_aspect <= 0:
            return x0, x1, y0, y1
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if (width / height) < box_aspect:
            new_width = height * box_aspect
            return cx - new_width / 2, cx + new_width / 2, y0, y1
        else:
            new_height = width / box_aspect
            return x0, x1, cy - new_height / 2, cy + new_height / 2

    def recompute_image_axes_position(fig_w_px, fig_h_px):
        # The axes box now always fills the whole available region (unlike
        # the previous approach, which shrunk *this box* to match the
        # image's own aspect ratio, leaving blank space beside it) —
        # aspect-correctness instead comes from padding the *data* range to
        # match the box's aspect (see compute_padded_canvas_bounds), so
        # 'auto' aspect still displays with zero distortion. ROI overlays
        # are unaffected: they're plotted in the same pixel-space data
        # coordinates as the image itself, not the axes' on-screen position.
        if fig_w_px <= 0 or fig_h_px <= 0:
            return
        avail_w_px = fig_w_px * axes_width_frac
        avail_h_px = fig_h_px * axes_height_frac
        if avail_w_px <= 0 or avail_h_px <= 0:
            return
        ax.set_position([AXES_LEFT, AXES_BOTTOM, axes_width_frac, axes_height_frac])
        box_aspect = avail_w_px / avail_h_px
        # The zoom-out/pan limit always reflects the *current* box shape...
        canvas_state['xlim'], canvas_state['ylim'] = compute_padded_canvas_bounds(box_aspect)
        # ...and the *current* view (whatever it was — possibly zoomed/
        # panned) gets corrected to that same new shape, preserving its
        # center/zoom level rather than snapping back to the full canvas.
        rescaled = rescale_view_to_aspect(*ax.get_xlim(), *ax.get_ylim(), box_aspect)
        # Re-clamped in case rescaling (e.g. a drastic aspect change) pushed
        # the view outside the — also just-updated — canvas bounds.
        new_x0, new_x1, new_y0, new_y1 = clamp_zoom_view(*rescaled)
        ax.set_xlim(new_x0, new_x1)
        ax.set_ylim(new_y0, new_y1)
        fig.canvas.draw_idle()

    # Widgets further down (the bottom button row, the gene-view overlays)
    # register a relayout here once they exist. Their positions are figure
    # fractions but their text and radio dots are sized in points, so without
    # re-measuring on every resize a narrower window squeezes them into
    # overlapping each other. Run after the image axes are recomputed.
    post_resize_hooks = []

    def run_post_resize_hooks():
        for hook in post_resize_hooks:
            hook()

    def on_resize(event):
        recompute_image_axes_position(event.width, event.height)
        run_post_resize_hooks()

    fig.canvas.mpl_connect('resize_event', on_resize)

    def on_configure(tk_event=None):
        # Supplementary to on_resize/resize_event above — matplotlib's own
        # resize_event only fires from the canvas widget's *own* <Configure>
        # callback, which (depending on window manager/platform) doesn't
        # always fire with the *final* correct size for every way a window's
        # size can change — notably, maximizing the toplevel programmatically
        # (see maximize_figure_window, called for this window so it opens
        # full-screen) and dragging an already-maximized/full-size window to
        # a differently-sized monitor. Both left the image pinned at
        # whatever size it last had a *real* resize_event for, while every
        # other axes in this figure (buttons, labels, ...) kept looking
        # right regardless, since their positions are plain figure-fractions
        # that matplotlib re-derives from the actual current canvas size on
        # every draw — no special handling (or event) needed for those.
        # Binding directly to the Tk toplevel's own <Configure> and reading
        # the canvas's actual current rendered size (fig.canvas.get_width_
        # height(), which matplotlib itself keeps correct across backends)
        # catches those cases too, instead of trusting a possibly-stale/
        # never-delivered synthetic event.
        w_px, h_px = fig.canvas.get_width_height()
        recompute_image_axes_position(w_px, h_px)
        run_post_resize_hooks()

    try:
        fig.canvas.manager.window.bind('<Configure>', on_configure)
    except Exception:
        pass

    # The axes' "data" coordinates are actually image pixel coordinates, so
    # ROI bounds (always real x/y) and mouse clicks need converting via
    # cached_layout.
    to_pixel = lambda x, y: map_data_to_pixel(cached_layout, x, y)
    to_data = lambda px, py: map_pixel_to_data(cached_layout, px, py)

    default_window_title = (
        f"Section {sanitize_section_token(section_label)}: drag a pink handle to resize an ROI, "
        "its interior to move it, or empty space to draw a new one (right-click to delete), then Done"
    )
    if precomputed_gene_open is not None:
        fig.canvas.manager.set_window_title(
            f"Section {sanitize_section_token(section_label)} — {precomputed_gene_open['resolved_gene']} "
            f"expression (max {precomputed_gene_open['max_expr']:.2f}); drag handles to edit ROIs, then Done"
        )
    else:
        fig.canvas.manager.set_window_title(default_window_title)

    # Lazily loaded on the first 'Show Gene' click if the caller didn't
    # already have it (most ROI-picking sessions never touch the gene view).
    spatial_state = {'df': spatial}
    # img is *not* a real class-colored render when this is False — see
    # precomputed_gene_open/class_image_ready above.
    class_image_state = {'ready': class_image_ready}
    if precomputed_gene_open is not None:
        # 'dataset' (set by handle_double_click when it builds this dict)
        # is what tells the initial view radio apart as Gene vs. Imputed
        # Gene — the render itself doesn't otherwise carry that.
        initial_mode = 'imputed_gene' if precomputed_gene_open.get('dataset') == 'imputed' else 'gene'
        current_view = {'mode': initial_mode, 'gene': precomputed_gene_open['resolved_gene']}
    else:
        current_view = {'mode': 'standard'}
    if session_color_scale is not None:
        # Shared by reference, not copied — toggling Linear/Log in this
        # window updates session_color_scale directly, so the caller's next
        # section opens with whatever was last chosen here.
        color_scale_state = session_color_scale
    elif precomputed_gene_open is not None:
        color_scale_state = {'mode': precomputed_gene_open['color_scale']}
    else:
        color_scale_state = {'mode': 'log'}
    # Per-gene render cache for this picker session, keyed on (gene, color
    # scale) — case-insensitively on the gene — since the two scales are
    # colored differently, not just relabeled (see render_gene_expression_
    # array's own docstring); re-showing a (gene, scale) combination
    # already viewed this session is then instant instead of re-pulling and
    # re-rasterizing its expression.
    gene_render_cache = {}
    if precomputed_gene_open is not None:
        gene_render_cache[(precomputed_gene_open['resolved_gene'].casefold(), color_scale_state['mode'])] = (
            precomputed_gene_open['rgba'], precomputed_gene_open['resolved_gene'], precomputed_gene_open['max_expr'],
            precomputed_gene_open['color_vmin'], precomputed_gene_open['color_vmax'],
        )

    # --- Hover-to-identify-cell: after the cursor sits still over the image
    # for HOVER_HOLD_MS, find the nearest cell (if any) within
    # HOVER_CELL_RADIUS_MICRONS and show its class/subclass/supertype in the
    # status bar with a ring around it. Distance is computed in real data
    # space, against this section's cells only.
    #
    # The ABC atlas's 'x'/'y' spatial columns are in millimeters, not
    # microns (confirmed empirically: nearest-cell distances came back as
    # ~0.00-0.02 "units", matching ~0-20 microns of real spacing between
    # MERFISH-detected cells — not 0.00-0.02 literal microns, which would
    # put cells sub-micron apart). HOVER_CELL_RADIUS_DATA_UNITS converts the
    # human-meaningful 30-micron threshold into that native unit once, for
    # both the distance comparison and the ring's on-screen radius — get
    # this wrong and the ring's radius comes out ~1000x too large (thirty
    # *millimeters* worth of pixels), which is what was happening before:
    # a circle far larger than the whole image, invisible/off-canvas rather
    # than a ring around the cell.
    HOVER_CELL_RADIUS_MICRONS = 10
    HOVER_CELL_RADIUS_DATA_UNITS = HOVER_CELL_RADIUS_MICRONS / 1000
    HOVER_HOLD_MS = 250
    # A real hand can't hold a cursor at pixel-perfect stillness — without
    # this tolerance, the ~1px jitter every mouse/trackpad produces keeps
    # re-triggering "the mouse moved" on every single frame, which
    # cancels the hold timer before it ever reaches HOVER_HOLD_MS and hides
    # the ring the instant it appears. Measured in on-screen display pixels
    # (event.x/event.y), not data/image pixels, so it means the same thing
    # at any zoom level.
    HOVER_MOVE_TOLERANCE_PX = 3
    tk_widget = fig.canvas.get_tk_widget()
    hover_state = {
        'timer_id': None, 'anchor_screen_px': None, 'was_in_ax': False, 'reraise_timer_id': None,
    }
    # Set while on_show_gene's report_progress is driving status_text
    # itself (via its own flush_events() calls, which pump the Tk event
    # queue and so can dispatch a queued mouse-motion event *during* that
    # synchronous rendering call) — without this, on_motion firing mid-render
    # overwrites the just-set "Rendering... NN%" text with plain coordinates
    # a frame later, before the user ever sees the percentage.
    status_updates_suppressed = {'active': False}
    # Populated lazily (shares spatial_state['df'] with the gene-expression
    # view — loading it once serves both) on the first hover-hold, not
    # eagerly, since most ROI-picking sessions never need it.
    hover_cells = {
        'ready': False, 'xs': None, 'ys': None,
        'class': None, 'subclass': None, 'supertype': None, 'cluster': None,
    }

    def wait_for_shared_spatial_load(progress_prefix):
        """Waits on shared_spatial_cache's background thread (starting it if
        it hasn't already been, e.g. by the grid picker at launch) and
        copies the result into spatial_state['df'] once done, instead of
        this window loading its own independent copy. Only called when
        shared_spatial_cache is not None — see prompt_subregion_selection's
        docstring."""
        ensure_spatial_load_started(shared_spatial_cache, adata, abc_cache)
        thread = shared_spatial_cache['load_thread']
        while thread is not None and thread.is_alive():
            total = shared_spatial_cache['load_progress']['total']
            rows = shared_spatial_cache['load_progress']['rows']
            frac = rows / total if total else 0.0
            status_text.set_text(f'{progress_prefix} {frac * 100:.0f}%')
            blit_hover_overlays()
            fig.canvas.flush_events()
            time.sleep(0.05)
        spatial_state['df'] = shared_spatial_cache['df']

    def ensure_hover_cell_data():
        if hover_cells['ready']:
            return hover_cells['xs'] is not None
        if spatial_state['df'] is None and shared_spatial_cache is not None:
            wait_for_shared_spatial_load('Loading cell metadata...')
        if spatial_state['df'] is None:
            # Can take several seconds the first time (same load the gene
            # view needs) — flagged in the status bar too, not just the
            # console, since this runs from inside a hover-hold callback
            # where a silent multi-second pause could otherwise look like
            # the whole UI just froze.
            print("Loading spatial data for cell info (first request; will be reused after)...")
            status_text.set_text('Loading cell metadata (first request; will be reused after)...')
            blit_hover_overlays()
            fig.canvas.flush_events()
            spatial_state['df'] = load_section_spatial_coords(adata, abc_cache)
            # An immediate raise_figure_window() here isn't enough: clicking
            # into the gene text box while this load is running (its click
            # gets queued and only actually processed once we next pump
            # events) leaves a focus change landing right in the middle of
            # this, and an immediate re-raise right as that settles loses
            # the race — same fix as show_figure_blocking's delayed
            # re-raise, applied here for the same reason. Tracked (not
            # fire-and-forget) so on_done/on_cancel can cancel it if the
            # window closes before it fires — see their own comment on
            # hover_state['timer_id'] for why that matters.
            hover_state['reraise_timer_id'] = tk_widget.after(200, lambda: raise_figure_window(fig))
        if spatial_state['df'] is None:
            hover_cells['ready'] = True  # tried and failed; don't keep retrying every hover
            return False
        df = spatial_state['df']
        section_mask = (section_series == section_label).to_numpy()
        section_df = df[section_mask]
        xs = pd.to_numeric(section_df['x'], errors='coerce').to_numpy()
        ys = pd.to_numeric(section_df['y'], errors='coerce').to_numpy()
        valid = ~(np.isnan(xs) | np.isnan(ys))
        hover_cells['xs'] = xs[valid]
        hover_cells['ys'] = ys[valid]
        for col in ('class', 'subclass', 'supertype', 'cluster'):
            hover_cells[col] = section_df[col].to_numpy()[valid] if col in section_df.columns else None
        hover_cells['ready'] = True
        return True

    # cached_layout's pixel_bbox/data_xlim/data_ylim give the (near-uniform)
    # pixel-per-micron scale needed to draw the ring — sized to the actual
    # search radius above — in this axes' pixel-space coordinate system.
    _pb_x0, _pb_x1, _pb_y0, _pb_y1 = cached_layout['pixel_bbox']
    _dx0, _dx1 = cached_layout['data_xlim']
    _dy0, _dy1 = cached_layout['data_ylim']
    _scale_x = abs(_pb_x1 - _pb_x0) / abs(_dx1 - _dx0) if _dx1 != _dx0 else 1.0
    _scale_y = abs(_pb_y1 - _pb_y0) / abs(_dy1 - _dy0) if _dy1 != _dy0 else 1.0
    hover_ring_radius_px = HOVER_CELL_RADIUS_DATA_UNITS * (_scale_x + _scale_y) / 2

    hover_ring = Circle((0, 0), hover_ring_radius_px, edgecolor='deeppink', facecolor='none',
                         linewidth=2, zorder=7, visible=False)
    # Animated (like status_text): skipped during normal full draws, so the
    # blitted background cache never has a stale ring baked into it.
    hover_ring.set_animated(True)
    ax.add_patch(hover_ring)

    # --- Group-marker overlay: once cell info has been showing for another
    # GROUP_MARKER_DELAY_MS with no further movement, mark every other cell
    # in this section sharing the hovered cell's class/subclass/supertype/
    # cluster (chosen via the "Group:" dropdown, built below with the rest
    # of the bottom button row) with a red '+'.
    GROUP_BY_OPTIONS = ('class', 'subclass', 'supertype', 'cluster')
    GROUP_MARKER_DELAY_MS = 500
    group_by_state = {'field': 'class'}
    group_marker_state = {'timer_id': None}
    group_marker_artist = ax.scatter([], [], marker='+', color='red', s=50, linewidths=1.5, zorder=8)
    # Animated for the same reason as hover_ring — shown/hidden purely via
    # blitting, never baked into the cached background.
    group_marker_artist.set_animated(True)
    group_marker_artist.set_visible(False)

    def show_group_markers(idx):
        # Scheduled from on_hover_settled only once a nearby cell was
        # actually found, so hover_cells is already populated by now.
        group_marker_state['timer_id'] = None
        field = group_by_state['field']
        values = hover_cells.get(field)
        if values is None:
            return
        target = values[idx]
        if pd.isna(target):
            return
        mask = values == target
        px_x, px_y = to_pixel(hover_cells['xs'][mask], hover_cells['ys'][mask])
        group_marker_artist.set_offsets(np.column_stack([px_x, px_y]))
        group_marker_artist.set_visible(True)
        print(f"Highlighting {int(mask.sum())} cells sharing {field} '{target}'.")
        blit_hover_overlays()

    # --- Every ROI — pre-existing or added this session — is always shown
    # with its handles and can be dragged at any time; nothing reverts to a
    # plain static outline mid-session anymore. `rois_display` holds one
    # entry per ROI, keyed by a stable id (not a list position, so deleting
    # one doesn't invalidate others): {'kind': 'existing'/'new',
    # 'orig_index': i or None, 'bounds': axes-space (x_lo, x_hi, y_lo, y_hi),
    # 'patch': Rectangle, 'handles': {(xkind, ykind): scatter artist}}.
    # Nothing is finalized into the returned action list until Done is
    # clicked — see on_done — so dragging/moving/deleting mid-session is
    # all just local display state until then.
    #
    # This is handled by our own press/motion/release logic rather than
    # matplotlib's RectangleSelector, which only ever manages one live
    # rectangle and doesn't distinguish "grabbed a handle" from "clicked
    # elsewhere" until well after it's already started rendering a live
    # drag in response — so any after-the-fact check bolted onto its
    # onselect callback is always just guessing at, and sometimes
    # disagreeing with, a decision it already visibly acted on. Owning the
    # whole interaction ourselves avoids that, and lets every ROI be
    # simultaneously editable instead of just one at a time.
    HANDLE_HIT_MARGIN_PX = 10
    MIN_DRAG_PX = 5  # below this, treat a drawing drag as an accidental click and ignore it
    # Right-clicking exactly on the boundary line of a thin/small ROI is
    # otherwise easy to miss by a pixel or two, especially at a low zoom
    # level where a whole ROI can be only a few pixels tall/wide — this
    # grows hit_test_interior's own hit zone by this many screen pixels on
    # every side (only for right-click delete, not the left-click move
    # tested at the same call site).
    INTERIOR_HIT_MARGIN_PX = 5

    rois_display = {}
    next_roi_id = [0]
    deleted_existing_indices = []  # orig_index of any pre-existing ROI deleted this session
    drag = {'mode': None, 'roi_id': None, 'handle': None, 'anchor': None}
    drawing = {'active': False, 'start': None, 'patch': None}  # brand-new ROI being drawn from scratch
    pan_state = {'active': False}  # right-drag over empty space (not an ROI — that opens the delete menu instead)

    # Redrawing this section's (potentially large) cached bitmap is the
    # expensive part of every pan step, not the coordinate math — same
    # throttle as the section-grid picker's own pan/zoom, capping actual
    # redraws at ~33/sec during a fast drag so requests can't queue up
    # faster than the canvas can render them. A skipped redraw always gets
    # a trailing timer so the view still settles to its exact final
    # position even if no further motion event arrives.
    REDRAW_MIN_INTERVAL = 0.03
    redraw_state = {'last_time': 0.0, 'timer': None}

    def force_draw():
        redraw_state['last_time'] = time.perf_counter()
        fig.canvas.draw_idle()

    def throttled_draw_idle():
        if redraw_state['timer'] is not None:
            redraw_state['timer'].stop()
            redraw_state['timer'] = None
        elapsed = time.perf_counter() - redraw_state['last_time']
        if elapsed >= REDRAW_MIN_INTERVAL:
            force_draw()
            return
        timer = fig.canvas.new_timer(interval=max((REDRAW_MIN_INTERVAL - elapsed) * 1000, 1))
        timer.single_shot = True
        timer.add_callback(force_draw)
        redraw_state['timer'] = timer
        timer.start()

    def apply_pan(x_px, y_px):
        # Pixel deltas divided by the (unchanging, since panning doesn't
        # rescale) axes size in pixels give a data-space shift that keeps
        # the point originally under the cursor fixed under the cursor —
        # same approach as the section-grid picker's own apply_pan.
        bbox = ax.get_window_extent()
        x0, x1 = pan_state['xlim0']
        y0, y1 = pan_state['ylim0']
        dx = -(x_px - pan_state['x0_px']) / bbox.width * (x1 - x0)
        dy = -(y_px - pan_state['y0_px']) / bbox.height * (y1 - y0)
        new_x0, new_x1, new_y0, new_y1 = clamp_zoom_view(x0 + dx, x1 + dx, y0 + dy, y1 + dy)
        ax.set_xlim(new_x0, new_x1)
        ax.set_ylim(new_y0, new_y1)

    def handle_points(bounds):
        x_lo, x_hi, y_lo, y_hi = bounds
        x_mid, y_mid = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
        xs = {'lo': x_lo, 'mid': x_mid, 'hi': x_hi}
        ys = {'lo': y_lo, 'mid': y_mid, 'hi': y_hi}
        return {
            (xk, yk): (xs[xk], ys[yk])
            for xk in ('lo', 'mid', 'hi') for yk in ('lo', 'mid', 'hi')
            if not (xk == 'mid' and yk == 'mid')  # 8 points: 4 corners + 4 edge midpoints
        }

    def add_display_entry(kind, orig_index, bounds):
        x_lo, x_hi, y_lo, y_hi = bounds
        patch = Rectangle(
            (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
            linewidth=2, edgecolor='deeppink', facecolor='none', zorder=6,
        )
        ax.add_patch(patch)
        handles = {
            key: ax.scatter([hx], [hy], s=36, c='deeppink', marker='s', zorder=7)
            for key, (hx, hy) in handle_points(bounds).items()
        }
        next_roi_id[0] += 1
        roi_id = next_roi_id[0]
        rois_display[roi_id] = {
            'kind': kind, 'orig_index': orig_index, 'bounds': bounds,
            'patch': patch, 'handles': handles,
        }
        return roi_id

    def redraw_display_geometry(roi_id):
        rec = rois_display[roi_id]
        x_lo, x_hi, y_lo, y_hi = rec['bounds']
        rec['patch'].set_xy((x_lo, y_lo))
        rec['patch'].set_width(x_hi - x_lo)
        rec['patch'].set_height(y_hi - y_lo)
        for key, (hx, hy) in handle_points(rec['bounds']).items():
            rec['handles'][key].set_offsets([[hx, hy]])

    def remove_display_entry(roi_id):
        rec = rois_display.pop(roi_id)
        rec['patch'].remove()
        for h in rec['handles'].values():
            h.remove()

    for idx, (x_min, x_max, y_min, y_max) in (existing_rois or []):
        px0, py0 = to_pixel(x_min, y_min)
        px1, py1 = to_pixel(x_max, y_max)
        x_lo, x_hi = sorted((px0, px1))
        y_lo, y_hi = sorted((py0, py1))
        add_display_entry('existing', idx, (x_lo, x_hi, y_lo, y_hi))

    def hit_test_handles(event):
        """Nearest handle within HANDLE_HIT_MARGIN_PX across *all* ROIs, or
        None. Returns (roi_id, handle_key)."""
        best, best_dist = None, HANDLE_HIT_MARGIN_PX
        for roi_id, rec in rois_display.items():
            for key, (hx, hy) in handle_points(rec['bounds']).items():
                hx_px, hy_px = ax.transData.transform((hx, hy))
                dist = math.hypot(hx_px - event.x, hy_px - event.y)
                if dist <= best_dist:
                    best, best_dist = (roi_id, key), dist
        return best

    def hit_test_interior(event, margin_px=0):
        """roi_id whose bounds contain event's data position, inflated by
        `margin_px` screen pixels on every side. Converted to data-space
        deltas via the axes' own current transform (not a flat data-unit
        margin) so the hit zone grows by the same number of *screen*
        pixels regardless of the current zoom/pan, and separately for x/y
        since they don't necessarily share one data-per-pixel scale."""
        if margin_px:
            inv = ax.transData.inverted()
            x0, y0 = inv.transform((0, 0))
            dx, dy = inv.transform((margin_px, margin_px))
            margin_x, margin_y = abs(dx - x0), abs(dy - y0)
        else:
            margin_x = margin_y = 0
        for roi_id, rec in rois_display.items():
            x_lo, x_hi, y_lo, y_hi = rec['bounds']
            if x_lo - margin_x <= event.xdata <= x_hi + margin_x and y_lo - margin_y <= event.ydata <= y_hi + margin_y:
                return roi_id
        return None

    def delete_roi(roi_id):
        if roi_id not in rois_display:
            return
        rec = rois_display[roi_id]
        if rec['kind'] == 'existing':
            # Nothing further to do for a 'new' one — it just never gets
            # added to `pending` in on_done since it's no longer in
            # rois_display. A pre-existing ROI needs an explicit 'delete'
            # action so the caller actually removes it from its own list;
            # simply omitting it wouldn't do anything, since the caller
            # only ever adds/updates entries in response to a returned
            # action, never removes them on its own.
            deleted_existing_indices.append(rec['orig_index'])
        remove_display_entry(roi_id)
        fig.canvas.draw_idle()
        print("Deleted that ROI.")

    def show_delete_menu(event, roi_id):
        try:
            import tkinter as tk
            menu = tk.Menu(fig.canvas.manager.window, tearoff=0)
            menu.add_command(label="Delete ROI", command=lambda: delete_roi(roi_id))
            gui_event = event.guiEvent
            try:
                menu.tk_popup(gui_event.x_root, gui_event.y_root)
            finally:
                menu.grab_release()
        except Exception as e:
            print(f"Could not show ROI context menu ({e}).")

    def on_press(event):
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return

        if event.button == 3:  # right-click on an ROI: delete menu; right-drag elsewhere: pan
            roi_id = hit_test_interior(event, margin_px=INTERIOR_HIT_MARGIN_PX)
            if roi_id is None:
                hit = hit_test_handles(event)
                roi_id = hit[0] if hit is not None else None
            if roi_id is not None:
                show_delete_menu(event, roi_id)
                return
            pan_state['active'] = True
            pan_state['x0_px'] = event.x
            pan_state['y0_px'] = event.y
            pan_state['xlim0'] = ax.get_xlim()
            pan_state['ylim0'] = ax.get_ylim()
            return

        if event.button != 1:  # left-click only for everything else below
            return

        hit = hit_test_handles(event)
        if hit is not None:
            drag['mode'] = 'handle'
            drag['roi_id'], drag['handle'] = hit
            return

        roi_id = hit_test_interior(event)
        if roi_id is not None:
            drag['mode'] = 'move'
            drag['roi_id'] = roi_id
            drag['anchor'] = (event.xdata, event.ydata, rois_display[roi_id]['bounds'])
            return

        # Otherwise: start drawing a brand-new ROI from scratch.
        drawing['active'] = True
        drawing['start'] = (event.xdata, event.ydata)
        drawing['patch'] = Rectangle(
            (event.xdata, event.ydata), 0, 0,
            linewidth=2, edgecolor='deeppink', facecolor='none', zorder=6,
        )
        ax.add_patch(drawing['patch'])

    def blit_hover_overlays():
        if blit_bg['data'] is not None:
            fig.canvas.restore_region(blit_bg['data'])
            fig.draw_artist(status_text)
            fig.draw_artist(hover_ring)
            fig.draw_artist(group_marker_artist)
            fig.draw_artist(gene_ax)
            fig.draw_artist(view_radio_ax)
            fig.draw_artist(scale_radio_ax)
            fig.canvas.blit(fig.bbox)
        else:
            fig.canvas.draw_idle()

    def on_hover_settled(pixel_pos):
        # Fires HOVER_HOLD_MS after the cursor last moved — on_motion
        # cancels this via after_cancel() on every subsequent move, so
        # reaching this line at all already means the mouse has been still
        # for the full hold time (there's no separate build-up-then-abort
        # mid-computation path to worry about: nothing else runs on this
        # thread between on_motion scheduling this and Tk invoking it).
        hover_state['timer_id'] = None
        if not ensure_hover_cell_data():
            return
        xs, ys = hover_cells['xs'], hover_cells['ys']
        if xs is None or len(xs) == 0:
            return
        data_x, data_y = to_data(*pixel_pos)
        dist2 = (xs - data_x) ** 2 + (ys - data_y) ** 2
        idx = int(np.argmin(dist2))
        if dist2[idx] > HOVER_CELL_RADIUS_DATA_UNITS ** 2:
            return  # nearest cell is still too far away

        hover_ring.center = to_pixel(xs[idx], ys[idx])
        hover_ring.set_visible(True)
        info_parts = [
            f'{label}: {hover_cells[label][idx]}'
            for label in ('class', 'subclass', 'supertype') if hover_cells[label] is not None
        ]
        info_line = ', '.join(info_parts) if info_parts else '(no class/subclass/supertype metadata)'
        status_text.set_text(f'x = {data_x:.3f}, y = {data_y:.3f}\n{info_line}')
        blit_hover_overlays()
        # A second, separate hold — the group markers are a bigger visual
        # change than the ring/text, so they only appear once the cursor
        # has stayed on this exact cell for a while longer, not the instant
        # its info shows up.
        group_marker_state['timer_id'] = tk_widget.after(GROUP_MARKER_DELAY_MS, lambda: show_group_markers(idx))

    def on_motion(event):
        if status_updates_suppressed['active']:
            return  # a gene render's progress readout owns the status bar right now

        if pan_state['active']:
            # Takes over entirely while active — bypasses the hover/status
            # -text machinery below altogether, same as it would for any
            # other exclusive drag. event.x/event.y (unlike xdata/ydata)
            # stay valid even once the drag continues past the image's
            # edge, so panning doesn't stall out there.
            if event.x is not None and event.y is not None:
                apply_pan(event.x, event.y)
                throttled_draw_idle()
            return

        # Cursor readout: independent of the drag/draw handling below, and
        # updated (or blanked, once the cursor leaves the image) on every
        # move rather than only while a button is held.
        if event.inaxes is ax and event.xdata is not None and event.ydata is not None:
            data_x, data_y = to_data(event.xdata, event.ydata)
            status_text.set_text(f'x = {data_x:.3f}, y = {data_y:.3f}')
        else:
            status_text.set_text('x = —, y = —')

        # A move beyond HOVER_MOVE_TOLERANCE_PX invalidates whatever
        # hover-cell lookup was pending or already shown: cancel the
        # pending timer (this — and the fact nothing gets scheduled again
        # until this handler sees a real move — is the "cancel if I move
        # while it's calculating" behavior) and hide the ring. A move
        # within tolerance (ordinary hand/mouse jitter) changes nothing,
        # so a hold in progress keeps counting down and a ring already
        # shown stays put.
        anchor = hover_state['anchor_screen_px']
        if anchor is None:
            moved_far_enough = True
        elif event.x is None or event.y is None:
            # Can't tell how far this event moved (no pixel coords) —
            # assume it didn't, rather than treating every such event as a
            # cancel; a real move triggers plenty of properly-coordinated
            # events on its own.
            moved_far_enough = False
        else:
            moved_far_enough = (
                abs(event.x - anchor[0]) > HOVER_MOVE_TOLERANCE_PX
                or abs(event.y - anchor[1]) > HOVER_MOVE_TOLERANCE_PX
            )
        hovering_only = drag['mode'] is None and not drawing['active']
        in_axes = event.inaxes is ax and event.xdata is not None and event.ydata is not None
        if moved_far_enough:
            if hover_state['timer_id'] is not None:
                tk_widget.after_cancel(hover_state['timer_id'])
                hover_state['timer_id'] = None
            if hover_ring.get_visible():
                hover_ring.set_visible(False)
            if group_marker_state['timer_id'] is not None:
                tk_widget.after_cancel(group_marker_state['timer_id'])
                group_marker_state['timer_id'] = None
            if group_marker_artist.get_visible():
                group_marker_artist.set_visible(False)
            if hovering_only and in_axes:
                hover_state['anchor_screen_px'] = (event.x, event.y)
                pixel_pos = (event.xdata, event.ydata)
                hover_state['timer_id'] = tk_widget.after(HOVER_HOLD_MS, lambda: on_hover_settled(pixel_pos))
            else:
                hover_state['anchor_screen_px'] = None

        # Plain hover (no drag/draw in progress) is by far the most common
        # case and the one where lag is most noticeable, so it's blitted:
        # restore the cached snapshot of everything else and redraw just
        # the status text/ring, instead of a full draw_idle() re-rendering
        # the image and every ROI patch on every single mouse move. When a
        # drag or draw IS in progress, skip this and let the branches below
        # do their own draw_idle() — it already needs to run for the
        # updated ROI geometry, and will pick up this text change too.
        #
        # Only blitted while actually over the image (or on the one move
        # that just left it, to clear the readout/ring rather than leaving
        # them stuck) — not on every move anywhere in the window. Blitting
        # restores blit_bg, which predates whatever matplotlib's own Button
        # widgets just drew for their hover-highlight; doing that on every
        # single mouse move, including ones over a button, was overwriting
        # that highlight right after it appeared — buttons visibly
        # flickered but never looked properly highlighted while hovered.
        if hovering_only and (in_axes or hover_state['was_in_ax']):
            blit_hover_overlays()
        hover_state['was_in_ax'] = in_axes

        if event.xdata is None or event.ydata is None:
            return

        if drag['mode'] == 'handle':
            roi_id = drag['roi_id']
            xkind, ykind = drag['handle']
            x_lo, x_hi, y_lo, y_hi = rois_display[roi_id]['bounds']
            min_size = 1e-9  # keeps the box from inverting if dragged past its opposite edge
            if xkind == 'lo':
                x_lo = min(event.xdata, x_hi - min_size)
            elif xkind == 'hi':
                x_hi = max(event.xdata, x_lo + min_size)
            if ykind == 'lo':
                y_lo = min(event.ydata, y_hi - min_size)
            elif ykind == 'hi':
                y_hi = max(event.ydata, y_lo + min_size)
            rois_display[roi_id]['bounds'] = (x_lo, x_hi, y_lo, y_hi)
            redraw_display_geometry(roi_id)
            fig.canvas.draw_idle()
        elif drag['mode'] == 'move':
            roi_id = drag['roi_id']
            press_x, press_y, (ox_lo, ox_hi, oy_lo, oy_hi) = drag['anchor']
            dx, dy = event.xdata - press_x, event.ydata - press_y
            rois_display[roi_id]['bounds'] = (ox_lo + dx, ox_hi + dx, oy_lo + dy, oy_hi + dy)
            redraw_display_geometry(roi_id)
            fig.canvas.draw_idle()
        elif drawing['active']:
            x0, y0 = drawing['start']
            x_lo, x_hi = sorted((x0, event.xdata))
            y_lo, y_hi = sorted((y0, event.ydata))
            drawing['patch'].set_xy((x_lo, y_lo))
            drawing['patch'].set_width(x_hi - x_lo)
            drawing['patch'].set_height(y_hi - y_lo)
            fig.canvas.draw_idle()

    def on_release(event):
        if event.button == 3 and pan_state['active']:
            pan_state['active'] = False
            # Throttled motion events can leave the view slightly behind
            # where the mouse actually stopped; snap to the exact final
            # position with one unconditional redraw, superseding any
            # pending trailing-timer redraw — same as the section-grid
            # picker's own pan release.
            if redraw_state['timer'] is not None:
                redraw_state['timer'].stop()
                redraw_state['timer'] = None
            if event.x is not None and event.y is not None:
                apply_pan(event.x, event.y)
            force_draw()
            return

        if drag['mode'] is not None:
            drag['mode'] = None
            drag['roi_id'] = None
            drag['handle'] = None
            drag['anchor'] = None
            fig.canvas.draw_idle()
        elif drawing['active']:
            x0, y0 = drawing['start']
            # event.xdata/ydata are None if the release happened outside the
            # axes (e.g. dragged off the edge) — fall back to the last
            # known position (wherever the live rectangle was last drawn to)
            # rather than discarding the drag.
            x1 = event.xdata if event.xdata is not None else drawing['patch'].get_x() + drawing['patch'].get_width()
            y1 = event.ydata if event.ydata is not None else drawing['patch'].get_y() + drawing['patch'].get_height()
            x_lo, x_hi = sorted((x0, x1))
            y_lo, y_hi = sorted((y0, y1))
            drawing['patch'].remove()
            drawing['active'] = False
            drawing['patch'] = None
            drawing['start'] = None

            (sx0, sy0) = ax.transData.transform((x_lo, y_lo))
            (sx1, sy1) = ax.transData.transform((x_hi, y_hi))
            if abs(sx1 - sx0) < MIN_DRAG_PX or abs(sy1 - sy0) < MIN_DRAG_PX:
                # Too small to be a deliberate drag — likely an accidental
                # click; discard it silently, same as the old minspanx/
                # minspany behavior.
                fig.canvas.draw_idle()
                return

            add_display_entry('new', None, (x_lo, x_hi, y_lo, y_hi))
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    def clamp_zoom_view(x0, x1, y0, y1):
        """Never zoom out past the full padded canvas view, and never let
        the visible window extend past the canvas bounds — same approach as
        the section grid picker's clamp_view, but against canvas_state (the
        image plus its black aspect-matching padding — see
        compute_padded_canvas_bounds) rather than img_w/img_h directly: that
        padding is real, pannable/zoomable data-space canvas, not a fixed
        backdrop, so both the zoom-out limit and the pan bounds need to
        extend into it too, not stop exactly at the image's own edges."""
        canvas_x0, canvas_x1 = canvas_state['xlim']
        canvas_y0, canvas_y1 = canvas_state['ylim']
        canvas_w, canvas_h = canvas_x1 - canvas_x0, canvas_y1 - canvas_y0
        width, height = x1 - x0, y1 - y0
        if width >= canvas_w:
            x0, x1, width = canvas_x0, canvas_x1, canvas_w
        if height >= canvas_h:
            y0, y1, height = canvas_y0, canvas_y1, canvas_h
        if x0 < canvas_x0:
            x0, x1 = canvas_x0, canvas_x0 + width
        elif x1 > canvas_x1:
            x0, x1 = canvas_x1 - width, canvas_x1
        if y0 < canvas_y0:
            y0, y1 = canvas_y0, canvas_y0 + height
        elif y1 > canvas_y1:
            y0, y1 = canvas_y1 - height, canvas_y1
        return x0, x1, y0, y1

    def on_scroll(event):
        # Scrolling the wheel doesn't move the cursor, so it wouldn't
        # otherwise cancel a pending hover-hold timer — without this, one
        # could fire mid-zoom (its first-ever run possibly blocking for
        # several seconds to load spatial data), making a zoom gesture
        # feel like it randomly freezes.
        if hover_state['timer_id'] is not None:
            tk_widget.after_cancel(hover_state['timer_id'])
            hover_state['timer_id'] = None
        hover_state['anchor_screen_px'] = None
        if hover_ring.get_visible():
            hover_ring.set_visible(False)
        if group_marker_state['timer_id'] is not None:
            tk_widget.after_cancel(group_marker_state['timer_id'])
            group_marker_state['timer_id'] = None
        if group_marker_artist.get_visible():
            group_marker_artist.set_visible(False)

        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        scale = 1.25 if event.button == 'down' else 0.8  # 'up' zooms in
        x_left, x_right = ax.get_xlim()
        y_bottom, y_top = ax.get_ylim()
        new_x0 = event.xdata - (event.xdata - x_left) * scale
        new_x1 = event.xdata + (x_right - event.xdata) * scale
        new_y0 = event.ydata - (event.ydata - y_bottom) * scale
        new_y1 = event.ydata + (y_top - event.ydata) * scale
        new_x0, new_x1, new_y0, new_y1 = clamp_zoom_view(new_x0, new_x1, new_y0, new_y1)
        ax.set_xlim(new_x0, new_x1)
        ax.set_ylim(new_y0, new_y1)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    pending = []  # built once, from whatever's on screen, when Done is clicked

    def on_done(event):
        for rec in rois_display.values():
            x_lo, x_hi, y_lo, y_hi = rec['bounds']
            x0_data, y0_data = to_data(x_lo, y_lo)
            x1_data, y1_data = to_data(x_hi, y_hi)
            bounds = (
                min(x0_data, x1_data), max(x0_data, x1_data),
                min(y0_data, y1_data), max(y0_data, y1_data),
            )
            if rec['kind'] == 'existing':
                pending.append({'action': 'edit', 'index': rec['orig_index'], 'bounds': bounds})
            else:
                pending.append({'action': 'add', 'bounds': bounds})
        for orig_index in deleted_existing_indices:
            pending.append({'action': 'delete', 'index': orig_index})
        if hover_state['timer_id'] is not None:  # else it could fire after the window's gone
            tk_widget.after_cancel(hover_state['timer_id'])
        if hover_state['reraise_timer_id'] is not None:
            tk_widget.after_cancel(hover_state['reraise_timer_id'])
        if group_marker_state['timer_id'] is not None:
            tk_widget.after_cancel(group_marker_state['timer_id'])
        plt.close(fig)

    def on_cancel(event):
        if hover_state['timer_id'] is not None:
            tk_widget.after_cancel(hover_state['timer_id'])
        if hover_state['reraise_timer_id'] is not None:
            tk_widget.after_cancel(hover_state['reraise_timer_id'])
        if group_marker_state['timer_id'] is not None:
            tk_widget.after_cancel(group_marker_state['timer_id'])
        plt.close(fig)  # pending stays empty — nothing from this session is applied

    # on_show_gene's return value when it couldn't even start because of the
    # gene names typed (none, too many, or not found), as opposed to True
    # (shown) or False (a render that failed for another reason). Lets
    # set_view_radio_selection keep the gene box usable so the names can be
    # fixed; see gene_entry_state.
    GENE_NAME_PROBLEM = 'gene_name_problem'

    def on_show_gene(event, use_imputed=None):
        # None (the default, for callers like rerender_gene_if_showing/
        # set_color_scale_selection that just want to re-render whatever's
        # already showing) means "whatever the current view already is" —
        # only set_view_radio_selection ever passes an explicit True/False,
        # since it's the only caller actually *switching* to a specific one.
        if use_imputed is None:
            use_imputed = current_view['mode'] == 'imputed_gene'
        gene_names = parse_gene_names(gene_textbox.text)
        if not gene_names:
            print("Enter a gene name first.")
            return GENE_NAME_PROBLEM
        if len(gene_names) > MAX_GENE_NAMES:
            print(TOO_MANY_GENES_MESSAGE)
            # blit_hover_overlays(), not draw_idle() — status_text is an
            # animated artist (see its own set_animated(True)), which a
            # plain full draw skips entirely; this is what actually makes
            # it show up right away.
            status_text.set_text(TOO_MANY_GENES_MESSAGE)
            blit_hover_overlays()
            return GENE_NAME_PROBLEM
        is_multi = len(gene_names) >= 2

        # Resolved *before* anything slow — a plain in-memory var-column
        # lookup, not the render itself — so the box can be corrected to
        # the canonical spelling, and the autocomplete dropdown closed,
        # immediately: the same "Enter commits right away" feel as the
        # interactive UMAP viewer's own query box, rather than waiting
        # however long the actual render takes.
        gene_source = imputed_state['adata'] if use_imputed else adata
        gene_cols, resolved_names, missing = resolve_gene_names(gene_source, gene_names)
        if missing:
            msg = (f"Gene(s) not found in the {'imputed' if use_imputed else 'standard'} dataset: "
                   f"{', '.join(missing)}.")
            print(msg)
            status_text.set_text(msg)
            # Only the names that didn't resolve turn red, in the box itself,
            # until the next edit (see update_suggestions).
            mark_invalid_gene_names(gene_textbox, missing)
            blit_hover_overlays()
            return GENE_NAME_PROBLEM
        clear_gene_name_marks(gene_textbox)
        combined_query = ', '.join(resolved_names)
        if gene_textbox.text != combined_query:
            set_textbox_text_silent(gene_textbox, combined_query)
        dropdown_ax.set_visible(False)
        suggestion_state['matches'] = []
        # Shown right now — before the render below, which for a cache miss
        # can take several seconds — via the same blit path report_progress
        # uses further down, so the corrected text/closed dropdown are
        # visible immediately rather than sitting behind whatever was on
        # screen until the render's own eventual draw_idle().
        blit_hover_overlays()
        fig.canvas.flush_events()

        def get_spatial():
            # Lazy on purpose: loading this can take several seconds the
            # first time, and both the in-memory gene_render_cache above and
            # render_gene_expression_array's own on-disk cache can make a
            # render need it not at all — eagerly loading it here regardless
            # was exactly the "several seconds of nothing happening before
            # any progress shows" stall.
            if spatial_state['df'] is None and shared_spatial_cache is not None:
                # Usually already running (or done) by now — see
                # start_spatial_load_if_needed()'s call when the grid
                # picker opened, or an earlier double-click's own wait on
                # it. Either way, waiting on that shared thread instead of
                # starting an independent load avoids paying for the same
                # multi-million-row CSV read twice.
                wait_for_shared_spatial_load('Loading spatial data...')
            if spatial_state['df'] is None:
                print("Loading spatial data for gene expression (first request; will be reused after)...")

                # Without its own feedback, this multi-second, multi-million
                # -row CSV read just freezes the status bar at whatever
                # percentage render_gene_expression_array's own checkpoints
                # last showed (report(0.15), reached right before this is
                # called) for however long it takes — indistinguishable
                # from the app having hung. load_section_spatial_coords
                # already reports real progress (rows read / total rows);
                # this just needs to actually show it.
                def on_spatial_progress(rows, total):
                    frac = rows / total if total else 0.0
                    status_text.set_text(f'Loading spatial data... {frac * 100:.0f}%')
                    if blit_bg['data'] is not None:
                        fig.canvas.restore_region(blit_bg['data'])
                        fig.draw_artist(status_text)
                        fig.canvas.blit(fig.bbox)
                    else:
                        fig.canvas.draw_idle()
                    fig.canvas.flush_events()

                spatial_state['df'] = load_section_spatial_coords(
                    adata, abc_cache, progress_callback=on_spatial_progress,
                )
            return spatial_state['df']

        # Dataset (and gene count/order) included — the same gene name(s)
        # can mean different renders now that Imputed Gene is independently
        # toggleable and 2-3 genes render as a red/green/blue overlay
        # instead of one viridis-colored gene, and a stale cache hit from
        # any of those would otherwise silently show the wrong thing. A
        # tuple of names (multi) vs. a single string (one gene) can never
        # collide with each other as dict keys.
        cache_key = (
            tuple(g.casefold() for g in resolved_names) if is_multi else resolved_names[0].casefold(),
            color_scale_state['mode'], 'imputed' if use_imputed else 'standard',
        )
        cached = gene_render_cache.get(cache_key)
        if cached is not None:
            new_img, resolved_out, extra1, extra2, extra3 = cached
        else:
            # For a fast render (a small section, or most of the work
            # already cached in memory/on disk), consecutive report_progress
            # calls can land within a couple milliseconds of each other —
            # faster than the screen's own refresh interval, so only the
            # very last one is ever actually presented, no matter how
            # promptly it's blitted. progress_timing enforces a floor on
            # how soon after the previous step's *screen time* the next one
            # is allowed to be shown, so each percentage gets a real chance
            # to be seen; for genuinely slow steps (where the real work
            # between checkpoints already exceeds this) it adds no delay.
            progress_timing = {'last_shown': None}
            MIN_STEP_DISPLAY_S = 0.08

            def report_progress(frac):
                # status_text is an animated artist (see the blitting setup
                # below on_motion) — a plain draw_idle() would skip it
                # entirely, not just draw it late. Blitting it directly, the
                # same way on_motion does, is what actually makes it appear;
                # flush_events() then forces Tk to repaint immediately
                # instead of batching every update until on_show_gene
                # returns (at which point there'd be nothing left to show).
                # Uses blit_hover_overlays() (not just status_text on its
                # own) so the *other* animated overlays — gene_ax,
                # view_radio_ax, scale_radio_ax — don't blink out for the
                # whole render: restoring blit_bg alone (which excludes all
                # animated artists, that's the point of them being animated)
                # and only re-drawing status_text on top was erasing them
                # from the screen on every single progress update, only for
                # them to reappear once the render finished.
                status_text.set_text(f'Rendering {combined_query} expression... {frac * 100:.0f}%')
                blit_hover_overlays()
                fig.canvas.flush_events()

                now = time.time()
                if progress_timing['last_shown'] is not None:
                    remaining = MIN_STEP_DISPLAY_S - (now - progress_timing['last_shown'])
                    if remaining > 0:
                        time.sleep(remaining)
                progress_timing['last_shown'] = time.time()

            # See status_updates_suppressed's definition: without this,
            # on_motion firing during one of report_progress's own
            # flush_events() calls below would immediately overwrite the
            # percentage text it just set.
            status_updates_suppressed['active'] = True
            # Only on a cache miss: a cache hit is instant, and greying the box
            # for it would just flicker.
            set_render_controls_busy(True)
            try:
                report_progress(0.0)
                render_fn = render_multi_gene_expression_array if is_multi else render_gene_expression_array
                try:
                    new_img, resolved_out, extra1, extra2, extra3 = render_fn(
                        adata, section_series, section_label, get_spatial,
                        resolved_names if is_multi else resolved_names[0],
                        cached_layout, img_w, img_h, on_progress=report_progress,
                        cell_type_selection=cell_type_selection, color_scale=color_scale_state['mode'],
                        expr_adata=imputed_state['adata'] if use_imputed else None,
                        dataset='imputed' if use_imputed else 'standard',
                    )
                except Exception as e:
                    print(f"Could not show expression for '{combined_query}': {e}")
                    status_text.set_text('x = —, y = —')
                    fig.canvas.draw_idle()
                    return False
            finally:
                status_updates_suppressed['active'] = False
                set_render_controls_busy(False)  # also on a failed render
            gene_render_cache[cache_key] = (new_img, resolved_out, extra1, extra2, extra3)
            status_text.set_text('x = —, y = —')

        # Single-gene: (resolved_out, extra1, extra2, extra3) are (resolved_
        # gene, max_expr, color_vmin, color_vmax). Multi-gene: (resolved_
        # genes, max_exprs, color_vmins, color_vmaxes) — all lists, one
        # entry per gene, same order as resolved_names (see render_multi_
        # gene_expression_array's own return-value docstring).
        img_artist.set_data(new_img)
        current_view['mode'] = 'imputed_gene' if use_imputed else 'gene'
        current_view['gene'] = combined_query
        update_mode_dependent_controls()
        imputed_suffix = ' [imputed]' if use_imputed else ''
        if is_multi:
            fig.canvas.manager.set_window_title(
                f"Section {sanitize_section_token(section_label)} — {combined_query} expression{imputed_suffix} "
                f"(red/green/blue); drag handles to edit ROIs, then Done"
            )
            update_multi_gene_colorbar(resolved_out, extra2, extra3)
            print(f"Showing '{combined_query}'{imputed_suffix} expression (red/green/blue) "
                  f"for section {section_label}.")
        else:
            max_expr, color_vmin, color_vmax = extra1, extra2, extra3
            fig.canvas.manager.set_window_title(
                f"Section {sanitize_section_token(section_label)} — {resolved_out} expression{imputed_suffix} "
                f"(max {max_expr:.2f}); drag handles to edit ROIs, then Done"
            )
            update_colorbar(resolved_out, color_vmin, color_vmax)
            print(f"Showing '{resolved_out}'{imputed_suffix} expression for section {section_label} "
                  f"(max value {max_expr:.3f}).")
        scale_radio_ax.set_visible(True)
        fig.canvas.draw_idle()
        raise_figure_window(fig)
        return True

    def ensure_class_image_ready():
        # Only actually does anything the first time this section's
        # standard view is needed after opening straight into the gene
        # view (see precomputed_gene_open/class_image_state) — renders and
        # disk-caches the real class-colored background exactly like
        # handle_double_click's own (skipped, in that scenario) call to
        # generate_and_cache_section_image would have, just deferred to
        # whenever 'Groups' first actually gets clicked instead of paid
        # for up front regardless of whether it's ever needed.
        nonlocal img
        if class_image_state['ready']:
            return True
        status_text.set_text('Rendering standard view (first time for this section)...')
        blit_hover_overlays()
        fig.canvas.flush_events()
        if spatial_state['df'] is None:
            spatial_state['df'] = load_section_spatial_coords(adata, abc_cache)
        if spatial_state['df'] is None:
            print(f"Could not load spatial data; standard view unavailable for section {section_label}.")
            status_text.set_text('x = —, y = —')
            fig.canvas.draw_idle()
            return False
        section_figsize = compute_figsize_for_screen_height(15 / 10, default=(15, 10))
        try:
            generate_and_cache_section_image(
                adata, abc_cache, section_series, section_label, spatial_state['df'], section_figsize,
            )
        except Exception as e:
            print(f"Could not render standard view for section {section_label}: {e}")
            status_text.set_text('x = —, y = —')
            fig.canvas.draw_idle()
            return False
        cache_png, _ = section_roi_cache_paths(section_label)
        img = plt.imread(cache_png)
        class_image_state['ready'] = True
        status_text.set_text('x = —, y = —')
        return True

    # --- Groups view coloring level ---------------------------------------
    # The Groups view is colored by whichever level the Group dropdown has
    # selected (group_by_state['field'], which also sets what the hover's red
    # '+' markers group by). Not remembered across windows: each window starts
    # at group_by_state's own default, 'class', matching the section grid. Images rendered in this
    # window are kept here; render_section_level_array also caches to disk.
    class_level_images = {}

    def ensure_picker_spatial(progress_prefix):
        """This section's spatial metadata, loading it on first use (waiting
        on the shared background load when there is one)."""
        if spatial_state['df'] is None and shared_spatial_cache is not None:
            wait_for_shared_spatial_load(progress_prefix)
        if spatial_state['df'] is None:
            status_text.set_text(f'{progress_prefix} (first request; will be reused after)')
            blit_hover_overlays()
            fig.canvas.flush_events()
            spatial_state['df'] = load_section_spatial_coords(adata, abc_cache)
        return spatial_state['df']

    def show_class_level(level):
        """Show the Groups-view image colored by `level`. Returns True if it's
        now showing. The class level is the section's cached class image;
        other levels come from render_section_level_array. While a render
        runs, the controls are disabled and the status bar shows progress,
        same as a gene render. On failure the current image is left as-is."""
        if level == 'class':
            if not class_image_state['ready'] and not ensure_class_image_ready():
                return False
            img_artist.set_data(img)
            img_artist.set_visible(True)
            fig.canvas.draw_idle()
            return True

        level_img = class_level_images.get(level)
        if level_img is None:
            def report_progress(frac):
                status_text.set_text(f'Coloring by {level}... {frac * 100:.0f}%')
                blit_hover_overlays()
                fig.canvas.flush_events()

            status_updates_suppressed['active'] = True
            set_render_controls_busy(True)
            try:
                spatial_df = ensure_picker_spatial('Loading cell metadata...')
                report_progress(0.0)
                level_img = render_section_level_array(
                    section_series, section_label, spatial_df, level, cached_layout, img_w, img_h,
                    on_progress=report_progress,
                )
                class_level_images[level] = level_img
            except Exception as e:
                print(f"Could not color section {section_label} by {level}: {e}")
                level_img = None
            finally:
                status_updates_suppressed['active'] = False
                set_render_controls_busy(False)
                status_text.set_text('x = —, y = —')
            if level_img is None:
                blit_hover_overlays()
                return False
        img_artist.set_data(level_img)
        img_artist.set_visible(True)
        fig.canvas.draw_idle()
        return True

    # Hourglass cursor while this window computes a new map: gene renders
    # (typing a gene, switching Gene/Imputed Gene, Linear/Log) and Group-level
    # recoloring. Both go through on_show_gene / show_class_level, so wrapping
    # those two covers every trigger. A depth count keeps nested calls from
    # restoring the arrow early. The arrow comes back only after the redraw
    # that shows the result (restore_cursor_after_pending_draw), so even a
    # cache hit shows the hourglass for that redraw.
    wait_cursor_state = {'depth': 0}

    def with_wait_cursor(func):
        def wrapper(*args, **kwargs):
            if wait_cursor_state['depth'] == 0:
                fig.canvas.set_cursor(Cursors.WAIT)
                try:
                    # Applies the cursor now, without processing queued input
                    # the way flush_events() would.
                    fig.canvas.manager.window.update_idletasks()
                except Exception:
                    pass
            wait_cursor_state['depth'] += 1
            try:
                return func(*args, **kwargs)
            finally:
                wait_cursor_state['depth'] -= 1
                if wait_cursor_state['depth'] == 0:
                    # Renders end with draw_idle(), which paints after this returns.
                    restore_cursor_after_pending_draw(fig, lambda: wait_cursor_state['depth'] > 0)
        return wrapper

    on_show_gene = with_wait_cursor(on_show_gene)
    show_class_level = with_wait_cursor(show_class_level)

    def on_standard_view(event):
        if current_view['mode'] == 'standard':
            return
        if not show_class_level(group_by_state['field']):
            return  # couldn't render; stay on the current (gene) view
        current_view['mode'] = 'standard'
        update_mode_dependent_controls()
        fig.canvas.manager.set_window_title(default_window_title)
        colorbar_ax.set_visible(False)
        scale_radio_ax.set_visible(False)
        # The multi-gene legend is a separate artist from colorbar_ax, so it
        # has to be cleared on its own or it stays over the class image.
        clear_multi_gene_legend()
        fig.canvas.draw_idle()
        raise_figure_window(fig)  # same reasoning as on_show_gene's — a real render here can take a moment too

    fig.subplots_adjust(top=AXES_TOP, bottom=AXES_BOTTOM, left=AXES_LEFT, right=AXES_RIGHT)
    # Shorter than before (0.06) so the freed vertical space can hold the
    # cursor-position readout above the button row, without growing
    # AXES_BOTTOM and eating into the plot area. Scaled by
    # full_screen_shrink_scale (see its own comment, above) so the row's
    # absolute on-screen height doesn't balloon just because this window
    # opens full-screen.
    button_height = 0.045 * full_screen_shrink_scale
    # Shared with the section grid picker and the processing dialog — see
    # compute_ui_fontsize() — rather than recomputed per-window.
    button_fontsize = UI_BUTTON_FONTSIZE

    # --- Gene-view colorbar + Linear/Log color-scale toggle, floating over
    # the top-right of the image (only relevant/shown while a gene is
    # displayed) — fixed figure-fraction overlays, like the hover ring/
    # group markers, rather than carving a permanent margin out of
    # AXES_RIGHT: that would mean re-deriving the image axes' own aspect
    # math (fig_aspect/on_resize) around a smaller box even for sessions
    # that never open the gene view at all.
    COLOR_SCALE_OPTIONS = ('Linear', 'Log')
    # Geometry is set by layout_gene_overlays() (below), on creation and on
    # every resize, not here: the box is exactly as wide as its buttons plus
    # OVERLAY_PAD_EM of padding each side, measured in points at the current
    # UI font scale, so it can't squeeze its labels into overlapping when the
    # window shrinks, nor leave most of itself empty when the window is large
    # (the old 10%-of-window minimum width did the latter). Its right edge is
    # at OVERLAY_RIGHT_EDGE unless the multi-gene legend below it is wider, in
    # which case both shift left together so their left edges stay aligned
    # and the legend doesn't run past the window's right side.
    OVERLAY_PAD_EM = 0.5
    OVERLAY_RIGHT_EDGE = 0.98
    OVERLAY_BOTTOM = 0.92
    # Font scale shared by the bottom button row and these overlays (1.0 =
    # button_fontsize). Set by layout_bottom_row() further down, which is
    # what shrinks text once the window is too small for it at full size.
    ui_scale_state = {'scale': 1.0}
    scale_radio_ax = fig.add_axes([OVERLAY_RIGHT_EDGE - 0.05, OVERLAY_BOTTOM, 0.05, button_height])
    scale_radio_ax.set_xlim(0, 1)
    scale_radio_ax.set_ylim(0, 1)
    scale_radio_ax.set_xticks([])
    scale_radio_ax.set_yticks([])
    for spine in scale_radio_ax.spines.values():
        spine.set_visible(False)
    # A solid background (same grey-box convention as the autocomplete
    # dropdown/group list elsewhere) rather than transparent — against a
    # busy viridis scatter, transparent left the dots' black text labels
    # essentially unreadable, so the "Log" dot in particular just read as
    # a stray unlabeled circle floating next to the colorbar.
    scale_radio_ax.patch.set_facecolor('#dddddd')
    scale_radio_ax.patch.set_edgecolor('black')
    scale_radio_ax.patch.set_linewidth(1)
    # Animated for the same reason as view_radio_ax — blitted so a click
    # shows immediately, before the re-render it triggers.
    scale_radio_ax.set_animated(True)

    scale_radio_dot_size = button_fontsize ** 2  # same 2x-diameter sizing as view_radio_dots
    scale_radio_initial_facecolor = (
        ['none', 'tab:blue'] if color_scale_state['mode'] == 'log' else ['tab:blue', 'none']
    )
    # Dot/label x positions are placeholders until layout_gene_overlays().
    scale_radio_state = {'dot_x': (0.25, 0.75)}
    scale_radio_dots = scale_radio_ax.scatter(
        scale_radio_state['dot_x'], [0.5, 0.5], s=[scale_radio_dot_size, scale_radio_dot_size],
        marker='o', edgecolor='black', facecolor=scale_radio_initial_facecolor, zorder=3,
    )
    scale_radio_label_texts = [
        scale_radio_ax.text(0.0, 0.5, slabel, fontsize=button_fontsize, va='center', ha='left')
        for slabel in COLOR_SCALE_OPTIONS
    ]
    scale_radio_ax.set_visible(precomputed_gene_open is not None)  # only meaningful once a gene is shown

    def set_color_scale_selection(idx):
        new_mode = COLOR_SCALE_OPTIONS[idx].lower()
        facecolors = ['none', 'none']
        facecolors[idx] = 'tab:blue'
        scale_radio_dots.set_facecolor(facecolors)
        blit_hover_overlays()
        if new_mode == color_scale_state['mode']:
            return
        color_scale_state['mode'] = new_mode
        if current_view['mode'] in ('gene', 'imputed_gene'):
            on_show_gene(None)  # re-render the currently-shown gene under the new scale

    # True while on_show_gene is rendering; see set_render_controls_busy.
    render_busy_state = {'active': False}

    def on_scale_radio_click(event):
        if not scale_radio_ax.get_visible() or event.inaxes is not scale_radio_ax or event.xdata is None:
            return
        if render_busy_state['active']:
            return  # greyed out mid-render; see set_render_controls_busy
        dot_x = scale_radio_state['dot_x']
        idx = 0 if event.xdata < (dot_x[0] + dot_x[1]) / 2 else 1
        set_color_scale_selection(idx)

    fig.canvas.mpl_connect('button_press_event', on_scale_radio_click)

    # Not animated/blitted, unlike the widgets above — it only ever changes
    # right alongside a gene render, which already ends in its own full
    # draw_idle() (for the image swap), so there's no separate fast-update
    # path worth building for it the way there is for click feedback.
    colorbar_ax = fig.add_axes([0.90, 0.55, 0.025, 0.3])
    colorbar_gradient = np.linspace(1, 0, 256).reshape(-1, 1)
    colorbar_ax.imshow(colorbar_gradient, aspect='auto', cmap='viridis', extent=(0, 1, 0, 1))
    colorbar_ax.set_xticks([])
    colorbar_ax.set_yticks([])
    # White, not the default black — the colorbar is only ever visible
    # while a gene render (black background, see render_gene_expression_
    # array's own comment on why) is showing behind it, against which
    # black ticks/labels/border were unreadable.
    for spine in colorbar_ax.spines.values():
        spine.set_edgecolor('white')
    colorbar_max_text = colorbar_ax.text(
        1.4, 1.0, '', fontsize=button_fontsize, va='top', ha='left', color='white',
        transform=colorbar_ax.transAxes,
    )
    colorbar_mid_text = colorbar_ax.text(
        1.4, 0.5, '', fontsize=button_fontsize, va='center', ha='left', color='white',
        transform=colorbar_ax.transAxes,
    )
    colorbar_min_text = colorbar_ax.text(
        1.4, 0.0, '', fontsize=button_fontsize, va='bottom', ha='left', color='white',
        transform=colorbar_ax.transAxes,
    )
    colorbar_title_text = colorbar_ax.text(
        0.5, 1.06, '', fontsize=button_fontsize, va='bottom', ha='center', color='white',
        transform=colorbar_ax.transAxes,
    )
    colorbar_ax.set_visible(precomputed_gene_open is not None)

    # Multi-gene legend: a *separate* box from the viridis colorbar above,
    # not a repurposed version of it — that narrow bar has no room to
    # actually contain gene-name text within its own bounds (the single-
    # gene labels above are positioned outside it, at x=1.4, relying on the
    # black image behind showing through); reusing it for multi-gene turned
    # into an empty-looking white bar once its gradient was hidden, with the
    # legend text floating outside it with no backdrop of its own. This is
    # wide enough to hold the text itself, with its own explicit white
    # background, so it reads as a real legend regardless of what's under
    # it. Same red/green/MULTI_GENE_BRIGHT_BLUE convention as the
    # interactive UMAP viewer's own multi-gene legend (draw_multi_gene_
    # legend). Mutually exclusive with colorbar_ax — see update_colorbar/
    # update_multi_gene_colorbar, which each hide the other.
    #
    # Built as an AnchoredOffsetbox rather than a fixed-size axes: a fixed
    # axes box (previously 16% x 30% of the figure) stays that size whatever
    # it holds, covering far more of the image than two or three short lines
    # need. The offsetbox's white frame wraps its contents, one single-line
    # TextArea per gene ("Gene  min–max") so each keeps its own color. Its
    # top-left corner sits MULTI_GENE_LEGEND_GAP_PT below the Linear/Log box,
    # aligned with that box's left edge, with the same OVERLAY_PAD_EM padding.
    # layout_gene_overlays() rebuilds it (at the current font scale and
    # position) whenever its contents or the window size change, so unused
    # gene slots take no room.
    MULTI_GENE_LEGEND_COLORS = ('red', 'green', MULTI_GENE_BRIGHT_BLUE)
    MULTI_GENE_LEGEND_GAP_PT = 4
    # 'rows': [(text, color), ...] while the legend should show, else None.
    multi_gene_legend = {'box': None, 'rows': None}

    def layout_gene_overlays():
        """Size and place the Linear/Log box and the multi-gene legend for the
        current window size and ui_scale_state, and rebuild the legend from
        multi_gene_legend['rows']. Everything is measured in points, then
        converted to figure fractions for the *current* figure size."""
        fig_w_in, fig_h_in = fig.get_size_inches()
        if fig_w_in <= 0 or fig_h_in <= 0:
            return
        fig_w_pt, fig_h_pt = fig_w_in * 72, fig_h_in * 72
        scale = ui_scale_state['scale']
        fontsize = button_fontsize * scale
        pad_pt = OVERLAY_PAD_EM * fontsize

        dot_x_pt, label_x_pt, radio_w_pt = scaled_radio_layout_pt(
            COLOR_SCALE_OPTIONS, button_fontsize, scale, pad_pt)
        rows = multi_gene_legend['rows']
        legend_w_pt = 0.0
        if rows:
            legend_w_pt = max(cached_text_width_pt(text, button_fontsize) for text, _ in rows) * scale \
                + 2 * pad_pt
        right_pt = OVERLAY_RIGHT_EDGE * fig_w_pt
        # With a legend showing, both boxes take the wider one's width.
        overlay_w_pt = max(radio_w_pt, legend_w_pt)
        left_pt = max(0.005 * fig_w_pt, right_pt - overlay_w_pt)
        box_w_pt = overlay_w_pt if rows else radio_w_pt
        # Buttons centered in a box widened to match the legend.
        offset_pt = (box_w_pt - radio_w_pt) / 2

        scale_radio_ax.set_position(
            [left_pt / fig_w_pt, OVERLAY_BOTTOM, box_w_pt / fig_w_pt, button_height])
        dot_x = tuple((x + offset_pt) / box_w_pt for x in dot_x_pt)
        scale_radio_state['dot_x'] = dot_x
        scale_radio_dots.set_offsets(np.column_stack([dot_x, [0.5] * len(dot_x)]))
        scale_radio_dots.set_sizes([fontsize ** 2] * len(dot_x))
        for text_artist, lx_pt in zip(scale_radio_label_texts, label_x_pt):
            text_artist.set_x((lx_pt + offset_pt) / box_w_pt)
            text_artist.set_fontsize(fontsize)

        if multi_gene_legend['box'] is not None:
            multi_gene_legend['box'].remove()
            multi_gene_legend['box'] = None
        if not rows:
            return
        gap_frac = MULTI_GENE_LEGEND_GAP_PT * scale / fig_h_pt
        text_rows = VPacker(
            children=[TextArea(text, textprops=dict(fontsize=fontsize, color=color))
                      for text, color in rows],
            align='left', pad=0, sep=fontsize * 0.25)
        # A zero-height spacer sets the legend's inner width, so the whole box
        # (inner width + pad_pt each side) comes out exactly box_w_pt wide,
        # matching the Linear/Log box. Stacked with sep=0 so it adds no height.
        width_spacer = DrawingArea(max(0.0, box_w_pt - 2 * pad_pt), 0)
        box = AnchoredOffsetbox(
            loc='upper left',
            child=VPacker(children=[width_spacer, text_rows], align='left', pad=0, sep=0),
            # pad is in units of prop's font size, so this equals pad_pt.
            pad=OVERLAY_PAD_EM, prop=dict(size=fontsize),
            borderpad=0, frameon=True,
            bbox_to_anchor=(left_pt / fig_w_pt, OVERLAY_BOTTOM - gap_frac), bbox_transform=fig.transFigure,
        )
        box.patch.set_facecolor('white')
        box.patch.set_edgecolor('black')
        box.patch.set_linewidth(scale_radio_ax.patch.get_linewidth())
        fig.add_artist(box)
        multi_gene_legend['box'] = box

    def clear_multi_gene_legend():
        multi_gene_legend['rows'] = None
        layout_gene_overlays()

    def update_colorbar(resolved_gene_name, color_vmin, color_vmax):
        colorbar_max_text.set_text(f'{color_vmax:.2f}')
        colorbar_mid_text.set_text(f'{(color_vmin + color_vmax) / 2:.2f}')
        colorbar_min_text.set_text(f'{color_vmin:.2f}')
        scale_label = 'log1p(expr)' if color_scale_state['mode'] == 'log' else 'expr (raw)'
        colorbar_title_text.set_text(f'{resolved_gene_name}\n{scale_label}')
        colorbar_ax.set_visible(True)
        clear_multi_gene_legend()

    def update_multi_gene_colorbar(resolved_genes, color_vmins, color_vmaxes):
        colorbar_ax.set_visible(False)
        multi_gene_legend['rows'] = [
            (f'{gene}  {vmin:.1f}–{vmax:.1f}', color)
            for gene, vmin, vmax, color in zip(
                resolved_genes, color_vmins, color_vmaxes, MULTI_GENE_LEGEND_COLORS)
        ] or None
        layout_gene_overlays()

    layout_gene_overlays()
    post_resize_hooks.append(layout_gene_overlays)

    if precomputed_gene_open is not None:
        update_colorbar(
            precomputed_gene_open['resolved_gene'],
            precomputed_gene_open['color_vmin'], precomputed_gene_open['color_vmax'],
        )

    status_text = fig.text(
        AXES_LEFT, button_height + 0.03, 'x = —, y = —',
        fontsize=button_fontsize, family='monospace', ha='left', va='bottom',
    )
    # 'animated' artists are skipped during matplotlib's normal full draws,
    # so the blit background cached below never has any of this text baked
    # into its pixels — without this, restoring that snapshot and drawing
    # new (often narrower) text on top left stale edges of the old text
    # peeking out around it.
    status_text.set_animated(True)

    # Cached snapshot of the whole canvas (image, ROI overlays, buttons —
    # everything except status_text's latest content) for on_motion's
    # blitting above. Re-snapshotted after every full draw, so it can never
    # go stale relative to zooming, panning, dragging, or switching between
    # standard/gene views — all of which already trigger a full draw_idle()
    # on their own.
    blit_bg = {'data': None}

    def cache_blit_background(event=None):
        blit_bg['data'] = fig.canvas.copy_from_bbox(fig.bbox)
        # Redraw the animated overlays on top immediately — otherwise a
        # full draw triggered elsewhere (zoom, drag, Standard View/Show
        # Gene, ...) would leave them invisible (gene_ax in particular,
        # once mouse movement or typing next redraws it, is fine, but
        # there'd be a stretch of "the box just vanished") until that next
        # interaction, since being animated means they're excluded from
        # normal full draws.
        fig.draw_artist(status_text)
        fig.draw_artist(hover_ring)
        fig.draw_artist(group_marker_artist)
        fig.draw_artist(gene_ax)
        fig.draw_artist(view_radio_ax)
        fig.draw_artist(scale_radio_ax)
        fig.canvas.blit(fig.bbox)

    fig.canvas.mpl_connect('draw_event', cache_blit_background)

    # Left to right: "Show:" label, Groups/Gene radio (horizontal — see
    # the section-grid picker's own hand-rolled version, matplotlib's own
    # RadioButtons only ever stacks vertically), gene text box, then the
    # Group/Done/Cancel buttons — same layout math as the section-grid
    # picker's rows (just with more widgets sharing this one).
    label_width = 0.07
    VIEW_RADIO_OPTIONS = ('Groups', 'Gene', 'Imputed Gene')
    # Computed from the labels' actual rendered width (see
    # radio_layout_as_axes_fractions) rather than a bare 0.20 constant — kept
    # as the floor for the common case where it's already enough room, grown
    # when it isn't, so no label ever runs into the next dot over.
    view_radio_dot_x, view_radio_label_x, radio_width = radio_layout_as_axes_fractions(
        VIEW_RADIO_OPTIONS, button_fontsize, 0.20, fig.get_size_inches()[0])
    textbox_width = 0.15
    button_width = 0.12
    # TextBox draws its label ('Gene(s): ') to the left of its own axes, not
    # inside it — with nothing reserved there it would be flush against the
    # radio widget just to its left. Measured the same way as
    # grid_label_width/gene_row_label_reserve above rather than a bare 0.06
    # constant — see their comments for why a fixed fraction isn't safe on a
    # narrow figure.
    label_reserve = (measured_text_width_pt('Gene(s): ', button_fontsize) + 4.0) \
        / (fig.get_size_inches()[0] * 72)
    n_gaps = 7  # before the label, label->radio, radio->text box, then between/after the 3 buttons
    gap = (
        axes_width_frac - label_width - radio_width - label_reserve - textbox_width - 3 * button_width
    ) / n_gaps
    x_label = AXES_LEFT + gap
    x_radio = x_label + label_width + gap
    x0 = x_radio + radio_width + gap + label_reserve  # gene text box — kept as x0, referenced widely below
    x3 = x0 + textbox_width + gap  # group button
    x4 = x3 + button_width + gap  # done
    x5 = x4 + button_width + gap  # cancel

    # Prefer the already-rendered gene's canonical name; otherwise fall back
    # to whatever the grid's own gene box had typed in it (initial_view's
    # 'gene', set regardless of the grid's Groups/Gene radio — see
    # handle_double_click) — so the box arrives ready to go even when this
    # window opens in standard view.
    if precomputed_gene_open is not None:
        gene_textbox_initial = precomputed_gene_open['resolved_gene']
    elif initial_view is not None and initial_view.get('gene'):
        gene_textbox_initial = initial_view['gene']
    else:
        gene_textbox_initial = ''

    gene_ax = fig.add_axes([x0, 0.02, textbox_width, button_height])
    gene_textbox = TextBox(gene_ax, 'Gene(s): ', initial=gene_textbox_initial)
    gene_textbox.label.set_fontsize(button_fontsize)
    gene_textbox.text_disp.set_fontsize(button_fontsize)  # the typed text itself, separate from the label
    # Animated for the same reason as status_text/hover_ring — drawn purely
    # via blitting, never through a normal full draw.
    gene_ax.set_animated(True)
    # See make_textbox_blit_fast's own docstring: TextBox forces a full,
    # synchronous fig.canvas.draw() on every keystroke entirely on its own
    # (inside _rendercursor), independent of on_text_change below — the
    # same fix already applied to the section-grid picker's gene box.
    make_textbox_blit_fast(gene_textbox, blit_hover_overlays)
    # See make_textbox_stop_typing_blit_fast's own docstring: separately
    # from typing, TextBox._click() forces the same kind of full,
    # synchronous draw on *any* click that lands outside this box — i.e.
    # nearly every click anywhere in the window.
    make_textbox_stop_typing_blit_fast(gene_textbox, blit_hover_overlays)
    # See make_textbox_motion_blit_fast's own docstring: TextBox._motion is
    # connected globally to every mouse move in the figure (not scoped to
    # this box), and forces the same kind of full, synchronous draw every
    # time the cursor crosses into or out of the box's own hover region.
    make_textbox_motion_blit_fast(gene_textbox, blit_hover_overlays)
    enable_textbox_clipboard_shortcuts(gene_textbox)

    # Normal text/outline color, restored when leaving the busy state.
    ENABLED_CONTROL_COLOR = 'black'

    # 'pending_mode': the Gene view ('gene'/'imputed_gene') the user last tried
    # to switch to, if that failed only because of the gene names typed, else
    # None. While set, the gene box stays enabled even though Groups is still
    # showing, so the names can be fixed; Enter then retries the switch (see
    # rerender_gene_if_showing). Without this, a misspelled name would leave
    # the box disabled in Groups with no way to correct it.
    gene_entry_state = {'pending_mode': None}

    def update_gene_box_enabled():
        """Enable the gene box only when it's usable: never mid-render, and
        otherwise only while a Gene view is showing or a switch to one is
        pending (gene_entry_state). Same disabled look as the other controls:
        text, label, outline and any red names fade to DISABLED_CONTROL_COLOR,
        background unchanged; set_active(False) makes TextBox ignore input.
        Closes the autocomplete list when disabling."""
        enabled = (not render_busy_state['active']
                   and (current_view['mode'] in ('gene', 'imputed_gene')
                        or gene_entry_state['pending_mode'] is not None))
        color = ENABLED_CONTROL_COLOR if enabled else DISABLED_CONTROL_COLOR
        gene_textbox.set_active(enabled)
        gene_textbox.text_disp.set_color(color)
        gene_textbox.label.set_color(color)
        for spine in gene_ax.spines.values():
            spine.set_edgecolor(color)
        set_gene_name_marks_color(gene_textbox, None if enabled else DISABLED_CONTROL_COLOR)
        if not enabled and dropdown_ax.get_visible():
            dropdown_ax.set_visible(False)
            suggestion_state['matches'] = []

    def set_render_controls_busy(busy):
        """Disable the gene box and the Linear/Log buttons while a gene render
        runs, and re-enable them afterwards. Same look as the UMAP viewer's
        set_sidebar_controls_busy: text, outlines and radio-dot edges fade to
        DISABLED_CONTROL_COLOR; backgrounds and the selected dot's fill stay
        as they are.

        Input has to actually be blocked, not just styled: the render's
        progress updates call flush_events(), which processes clicks and
        keystrokes mid-render. An edit could otherwise land in the box while
        it renders the previous text, and a Linear/Log click would start a
        second render inside this one. set_active(False) makes TextBox ignore
        events; the Linear/Log buttons are hand-rolled, so on_scale_radio_click
        checks render_busy_state itself."""
        render_busy_state['active'] = busy
        color = DISABLED_CONTROL_COLOR if busy else ENABLED_CONTROL_COLOR
        # The gene box also depends on the view (disabled in Groups), so its
        # state is decided in one place rather than simply re-enabled here.
        update_gene_box_enabled()
        for text_artist in scale_radio_label_texts:
            text_artist.set_color(color)
        scale_radio_dots.set_edgecolor(color)
        scale_radio_ax.patch.set_edgecolor(color)
        # gene_ax is animated, so it only reaches the screen via blitting. No
        # flush_events() here: on restore it would process queued clicks/keys
        # while on_show_gene is still finishing, and the render's own progress
        # updates already flush right after the box goes grey.
        blit_hover_overlays()

    # Autocomplete dropdown for the gene box. Opens upward — there's no room
    # below the bottom row — overlaying the lower part of the image; it's
    # hidden by default and only appears once there's a query to show
    # matches for.
    max_suggestions = 6
    suggestion_line_height = 0.035
    dropdown_ax = fig.add_axes([
        x0, 0.02 + button_height, textbox_width, max_suggestions * suggestion_line_height,
    ])
    dropdown_ax.set_xlim(0, 1)
    dropdown_ax.set_ylim(0, 1)
    # NOT axis('off') — that sets axison=False, which makes matplotlib skip
    # drawing the axes' own background patch entirely (not just ticks), so
    # the facecolor below was silently never actually rendered. Hiding
    # ticks/spines individually instead leaves the patch itself drawable.
    dropdown_ax.set_xticks([])
    dropdown_ax.set_yticks([])
    for spine in dropdown_ax.spines.values():
        spine.set_visible(False)
    dropdown_ax.patch.set_facecolor('#dddddd')  # distinct from the plain-white rest of the window
    dropdown_ax.patch.set_edgecolor('black')
    dropdown_ax.patch.set_linewidth(1)
    dropdown_ax.set_visible(False)
    suggestion_texts = [
        dropdown_ax.text(0.03, 1 - (i + 0.5) / max_suggestions, '', fontsize=button_fontsize, va='center', ha='left')
        for i in range(max_suggestions)
    ]
    suggestion_state = {'matches': []}
    # Set around a programmatic gene_textbox.set_val() (picking a gene from
    # the dropdown, or on_show_gene's own canonical-case correction) so it
    # doesn't re-trigger update_suggestions on the text it just set — same
    # reasoning, and same name, as the interactive UMAP viewer's own flag.
    suppress_autocomplete = {'value': False}

    def update_suggestions(text):
        # Any edit clears the red "not found" names; they only come back if
        # the edited text still doesn't resolve when next shown.
        clear_gene_name_marks(gene_textbox)
        if suppress_autocomplete['value']:
            return
        _prefix, current_token = split_gene_query(text)
        if not current_token:
            dropdown_ax.set_visible(False)
            suggestion_state['matches'] = []
            fig.canvas.draw_idle()
            return
        matches = rank_gene_suggestions(
            gene_symbol_list(adata, imputed_state, current_view['mode'] == 'imputed_gene'),
            current_token, max_suggestions)
        suggestion_state['matches'] = matches
        n = len(matches)
        if n == 0:
            dropdown_ax.set_visible(False)
            fig.canvas.draw_idle()
            return
        # Resized to fit exactly n rows, bottom edge anchored to the text
        # box's top — otherwise, with a fixed-size box always showing
        # max_suggestions worth of rows, matches (drawn from the top down)
        # leave a growing blank gap between the last real one and the text
        # box as the list narrows.
        # Follows the gene box's *current* x/width, which layout_bottom_row()
        # changes on resize.
        gene_box_pos = gene_ax.get_position()
        dropdown_ax.set_position(
            [gene_box_pos.x0, 0.02 + button_height, gene_box_pos.width, n * suggestion_line_height])
        for i, t in enumerate(suggestion_texts):
            if i < n:
                t.set_text(matches[i])
                t.set_position((0.03, 1 - (i + 0.5) / n))
            else:
                t.set_text('')
        dropdown_ax.set_visible(True)
        fig.canvas.draw_idle()

    gene_textbox.on_text_change(update_suggestions)

    def rerender_gene_if_showing(_text):
        # Enter only; a click-away submit (including a click on a dropdown
        # suggestion, which on_gene_dropdown_click handles) is ignored. See
        # is_enter_submit.
        if not is_enter_submit(gene_textbox):
            return
        dropdown_ax.set_visible(False)
        suggestion_state['matches'] = []
        # Only while already viewing Gene or Imputed Gene, or retrying a switch
        # to one that failed on its gene names (gene_entry_state) — the box is
        # disabled in Groups otherwise, so nothing here switches views
        # unprompted. on_show_gene's own use_imputed=None default picks
        # whichever of the two is actually current, and does its own
        # (redundant here, but harmless) case-correction/dropdown-closing
        # before rendering.
        if current_view['mode'] in ('gene', 'imputed_gene'):
            on_show_gene(None)
        elif gene_entry_state['pending_mode'] is not None:
            set_view_radio_selection(VIEW_INDEX_BY_MODE[gene_entry_state['pending_mode']])

    gene_textbox.on_submit(rerender_gene_if_showing)

    def on_gene_dropdown_click(event):
        if not dropdown_ax.get_visible():
            return
        if event.inaxes is gene_ax:
            return  # let the text box handle its own focus/cursor click
        if event.inaxes is dropdown_ax and event.ydata is not None:
            # Rows are spaced across n = len(matches), not max_suggestions
            # (see update_suggestions) — the box is resized to fit exactly
            # the current match count, so this must match that spacing.
            n = len(suggestion_state['matches'])
            if n > 0:
                row = min(n - 1, max(0, int((1 - event.ydata) * n)))
                # completed_gene_query keeps names typed before the one being
                # completed (e.g. "Snap25, " while picking the second gene).
                # A *plain* set_val (not set_textbox_text_silent) — this is
                # what actually fires 'submit', running rerender_gene_if_
                # showing above exactly once; suppress_autocomplete only
                # blocks the dropdown from reopening on its own 'change'
                # event, not the render.
                suppress_autocomplete['value'] = True
                try:
                    gene_textbox.set_val(
                        completed_gene_query(gene_textbox.text, suggestion_state['matches'][row]))
                finally:
                    suppress_autocomplete['value'] = False
        dropdown_ax.set_visible(False)
        suggestion_state['matches'] = []
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_gene_dropdown_click)

    show_label_text = fig.text(x_label, 0.02 + button_height / 2, 'Color cells by:',
                                fontsize=button_fontsize, va='center', ha='left')

    # Hand-rolled horizontal 2-option radio (Groups | Gene) — same
    # approach as the section-grid picker's own (matplotlib's RadioButtons
    # only ever stacks options vertically, regardless of the axes' aspect
    # ratio). Selecting an option here immediately switches the view
    # (unlike the grid's, which only records a default) by calling straight
    # into on_show_gene/on_standard_view — including re-clicking "Gene"
    # while it's already selected, which is how a newly-typed/picked gene
    # name actually gets applied now that there's no separate 'Show Gene'
    # button to click.
    view_radio_ax = fig.add_axes([x_radio, 0.02, radio_width, button_height])
    view_radio_ax.set_xlim(0, 1)
    view_radio_ax.set_ylim(0, 1)
    view_radio_ax.set_xticks([])
    view_radio_ax.set_yticks([])
    for spine in view_radio_ax.spines.values():
        spine.set_visible(False)
    view_radio_ax.set_facecolor('none')
    # Animated for the same reason as gene_ax etc. — drawn purely via
    # blitting (see blit_hover_overlays), so set_view_radio_selection can
    # show the click landing immediately, before the (potentially
    # multi-second) render it triggers, instead of only updating once that
    # render's own full draw_idle() finally runs.
    view_radio_ax.set_animated(True)

    # matplotlib's own RadioButtons sizes its dots as (fontsize/2)**2 —
    # this is 2x the linear size (diameter), tied to the same font-scaling
    # (UI_BUTTON_FONTSIZE/button_fontsize) as everything else, same as the
    # grid picker's. view_radio_dot_x/view_radio_label_x were computed above
    # (with radio_width) via radio_layout_as_axes_fractions.
    view_radio_dot_size = button_fontsize ** 2
    VIEW_MODE_BY_LABEL = {'Groups': 'standard', 'Gene': 'gene', 'Imputed Gene': 'imputed_gene'}
    VIEW_INDEX_BY_MODE = {mode: i for i, mode in enumerate(VIEW_MODE_BY_LABEL[label] for label in VIEW_RADIO_OPTIONS)}
    initial_view_idx = VIEW_INDEX_BY_MODE.get(current_view['mode'], 0)
    view_radio_initial_facecolor = ['none'] * len(VIEW_RADIO_OPTIONS)
    view_radio_initial_facecolor[initial_view_idx] = 'tab:blue'
    view_radio_dots = view_radio_ax.scatter(
        view_radio_dot_x, [0.5] * len(VIEW_RADIO_OPTIONS),
        s=[view_radio_dot_size] * len(VIEW_RADIO_OPTIONS),
        marker='o', edgecolor='black', facecolor=view_radio_initial_facecolor, zorder=3,
    )
    view_radio_label_texts = [
        view_radio_ax.text(vx, 0.5, vlabel, fontsize=button_fontsize, va='center', ha='left')
        for vx, vlabel in zip(view_radio_label_x, VIEW_RADIO_OPTIONS)
    ]
    # Updated by layout_bottom_row() on resize; read by on_view_radio_click.
    view_radio_state = {'dot_x': tuple(view_radio_dot_x)}

    def set_view_radio_selection(idx):
        # Shown immediately — before on_show_gene/on_standard_view, which
        # can take a while (a real gene render is not fast, and Imputed
        # Gene can also mean loading that whole dataset first) — so the dot
        # reacts to the click right away rather than sitting on the old
        # selection for the whole render, which read as the click not
        # having registered. Cheap since view_radio_ax is animated: this
        # blits just it (plus the other animated overlays), not a full
        # draw_idle() of the whole window.
        optimistic_facecolors = ['none'] * len(VIEW_RADIO_OPTIONS)
        optimistic_facecolors[idx] = 'tab:blue'
        view_radio_dots.set_facecolor(optimistic_facecolors)
        blit_hover_overlays()

        label = VIEW_RADIO_OPTIONS[idx]
        gene_entry_state['pending_mode'] = None
        if label == 'Groups':
            on_standard_view(None)
        else:
            use_imputed = label == 'Imputed Gene'
            # Loading the imputed dataset is slow and needs an explicit
            # up-front warning (see ensure_imputed_gene_dataset_loaded) —
            # same gate the section-grid picker's own radio uses. If it's
            # unavailable (declined, cancelled, failed, or this window was
            # never given an imputed_state to begin with), skip straight to
            # the actual_idx correction below rather than rendering with no
            # data to render from.
            if not use_imputed or (imputed_state is not None
                                    and ensure_imputed_gene_dataset_loaded(imputed_state, abc_cache)):
                result = on_show_gene(None, use_imputed=use_imputed)
                if result == GENE_NAME_PROBLEM:
                    # Couldn't switch only because of the names typed: keep the
                    # gene box usable (it's otherwise disabled in Groups) so
                    # they can be fixed, then Enter retries this switch.
                    gene_entry_state['pending_mode'] = 'imputed_gene' if use_imputed else 'gene'
        update_mode_dependent_controls()
        blit_hover_overlays()
        # Reflects what actually happened, not just what was clicked —
        # on_show_gene can fail (bad gene name, no spatial data, an
        # unavailable imputed dataset, ...) and leave the view unchanged, in
        # which case the dot should revert to whatever's actually showing
        # rather than keep showing a switch that didn't happen.
        actual_idx = VIEW_INDEX_BY_MODE.get(current_view['mode'], 0)
        if actual_idx != idx:
            # on_show_gene/on_standard_view already ran their own
            # draw_idle() above (for the image swap) — a full draw_idle()
            # here would just skip view_radio_ax again, same as it did
            # then, since it's animated; blitting is what actually shows
            # this correction.
            facecolors = ['none'] * len(VIEW_RADIO_OPTIONS)
            facecolors[actual_idx] = 'tab:blue'
            view_radio_dots.set_facecolor(facecolors)
            blit_hover_overlays()

    def on_view_radio_click(event):
        if event.inaxes is not view_radio_ax or event.xdata is None:
            return
        # A render's progress updates process pending clicks, so without this
        # a click here mid-render would start a second render inside it.
        if render_busy_state['active']:
            return
        dot_x = view_radio_state['dot_x']
        idx = min(range(len(dot_x)), key=lambda i: abs(dot_x[i] - event.xdata))
        set_view_radio_selection(idx)

    fig.canvas.mpl_connect('button_press_event', on_view_radio_click)

    # "Group: <field> ▾" — picks which of hover_cells' fields
    # show_group_markers() groups by. Opens upward, same as the gene box's
    # autocomplete (no room below the bottom row); same click-to-open/
    # click-to-select/click-elsewhere-to-close pattern, just with a fixed
    # 4-row list instead of one filtered as you type.
    group_button_ax = fig.add_axes([x3, 0.02, button_width, button_height])
    group_button = Button(group_button_ax, f'Group: {group_by_state["field"].capitalize()} ▾')
    group_button.label.set_fontsize(button_fontsize)

    group_option_row_height = 0.035
    group_list_ax = fig.add_axes([
        x3, 0.02 + button_height, button_width, len(GROUP_BY_OPTIONS) * group_option_row_height,
    ])
    group_list_ax.set_xlim(0, 1)
    group_list_ax.set_ylim(0, 1)
    group_list_ax.set_xticks([])
    group_list_ax.set_yticks([])
    for spine in group_list_ax.spines.values():
        spine.set_visible(False)
    group_list_ax.patch.set_facecolor('#dddddd')
    group_list_ax.patch.set_edgecolor('black')
    group_list_ax.patch.set_linewidth(1)
    group_list_ax.set_visible(False)
    group_list_texts = [
        group_list_ax.text(0.08, 1 - (i + 0.5) / len(GROUP_BY_OPTIONS), opt.capitalize(),
                            fontsize=button_fontsize, va='center', ha='left')
        for i, opt in enumerate(GROUP_BY_OPTIONS)
    ]

    def on_group_button_clicked(event):
        if render_busy_state['active']:
            return
        group_list_ax.set_visible(not group_list_ax.get_visible())
        fig.canvas.draw_idle()

    group_button.on_clicked(on_group_button_clicked)

    def on_group_list_click(event):
        if not group_list_ax.get_visible() or render_busy_state['active']:
            return
        if event.inaxes is group_button_ax:
            return  # the button's own on_clicked (fires on release) handles this click
        chosen = None
        if event.inaxes is group_list_ax and event.ydata is not None:
            n = len(GROUP_BY_OPTIONS)
            row = min(n - 1, max(0, int((1 - event.ydata) * n)))
            chosen = GROUP_BY_OPTIONS[row]
        group_list_ax.set_visible(False)
        fig.canvas.draw_idle()
        if chosen is None or chosen == group_by_state['field']:
            return
        group_by_state['field'] = chosen
        group_button.label.set_text(f'Group: {chosen.capitalize()} ▾')
        # In the Groups view the level also sets the coloring. In a Gene view
        # it only changes the hover grouping; the coloring is applied the next
        # time Groups is shown (on_standard_view).
        if current_view['mode'] == 'standard':
            show_class_level(chosen)

    fig.canvas.mpl_connect('button_press_event', on_group_list_click)

    def update_mode_dependent_controls():
        """Refresh every control whose enabled state depends on the view: the
        gene box (Gene views only). Called wherever current_view['mode'] or
        gene_entry_state changes. The Group dropdown is deliberately *not*
        view-dependent: it picks what the hover highlights (red '+' markers)
        group by, which applies in every view, including Gene views."""
        update_gene_box_enabled()

    update_mode_dependent_controls()

    done_ax = fig.add_axes([x4, 0.02, button_width, button_height])
    done_button = Button(done_ax, 'Done')
    done_button.label.set_fontsize(button_fontsize)
    done_button.on_clicked(on_done)

    cancel_ax = fig.add_axes([x5, 0.02, button_width, button_height])
    cancel_button = Button(cancel_ax, 'Cancel')
    cancel_button.label.set_fontsize(button_fontsize)
    cancel_button.on_clicked(on_cancel)

    GROUP_BUTTON_LABELS = [f'Group: {opt.capitalize()} ▾' for opt in GROUP_BY_OPTIONS]
    MIN_UI_FONTSIZE = 6.0
    BOTTOM_ROW_GAPS = 7  # same count as n_gaps above

    def layout_bottom_row():
        """Re-lay out the bottom row ('Show:', Groups/Gene/Imputed Gene,
        'Gene:' box, Group/Done/Cancel) for the current window size, and set
        ui_scale_state['scale'], which the gene-view overlays also use.

        At a comfortable size this reproduces the fixed-fraction layout the
        row is created with (same preferred widths, same evenly-spread gaps),
        so the window looks the same when it first opens. As the window
        narrows, the widgets first shrink from those preferred widths toward
        the minimum their own text needs; only once even that no longer fits
        does the text itself shrink (one shared factor, floored at
        MIN_UI_FONTSIZE). The text also shrinks if the window gets short
        enough that it would no longer fit the row's height. All widths are
        in points, since that's what text and radio dots are sized in."""
        fig_w_in, fig_h_in = fig.get_size_inches()
        if fig_w_in <= 0 or fig_h_in <= 0:
            return
        fig_w_pt = fig_w_in * 72
        avail_pt = axes_width_frac * fig_w_pt
        base = button_fontsize

        def tw(text):
            return cached_text_width_pt(text, base)

        # Minimum widths at full font size. Every term is proportional to the
        # font size, so at scale s the whole row's minimum is s times this.
        _, _, radio_min_full = scaled_radio_layout_pt(VIEW_RADIO_OPTIONS, base, 1.0, 4.0)
        mins_full = {
            'label': tw('Show:') + 0.4 * base,
            'radio': radio_min_full,
            'textbox': 6.0 * base,
            'group': max(tw(t) for t in GROUP_BUTTON_LABELS) + 1.2 * base,
            'done': tw('Done') + 1.2 * base,
            'cancel': tw('Cancel') + 1.2 * base,
        }
        reserve_full = tw('Gene(s): ') + 4.0  # the TextBox's label, drawn left of its axes
        gap_min_full = 0.4 * base
        min_row_full = sum(mins_full.values()) + reserve_full + BOTTOM_ROW_GAPS * gap_min_full

        button_height_pt = button_height * fig_h_in * 72
        scale = min(1.0, avail_pt / min_row_full, (button_height_pt / 1.6) / base)
        scale = max(scale, MIN_UI_FONTSIZE / base)
        ui_scale_state['scale'] = scale
        fontsize = base * scale

        mins = {k: v * scale for k, v in mins_full.items()}
        reserve = reserve_full * scale
        fixed_pt = reserve + BOTTOM_ROW_GAPS * gap_min_full * scale
        prefs = {
            'label': max(label_width * fig_w_pt, mins['label']),
            'radio': max(0.20 * fig_w_pt, mins['radio']),
            'textbox': max(textbox_width * fig_w_pt, mins['textbox']),
            'group': max(button_width * fig_w_pt, mins['group']),
            'done': max(button_width * fig_w_pt, mins['done']),
            'cancel': max(button_width * fig_w_pt, mins['cancel']),
        }
        sum_pref, sum_min = sum(prefs.values()), sum(mins.values())
        if sum_pref + fixed_pt <= avail_pt:
            widths = prefs
        else:
            # Shrink every widget the same fraction of the way from its
            # preferred width toward its minimum, just enough to fit.
            span = sum_pref - sum_min
            t = (avail_pt - fixed_pt - sum_min) / span if span > 0 else 0.0
            t = min(1.0, max(0.0, t))
            widths = {k: mins[k] + (prefs[k] - mins[k]) * t for k in prefs}
        gap = max(0.0, (avail_pt - sum(widths.values()) - reserve) / BOTTOM_ROW_GAPS)

        x_label_pt = AXES_LEFT * fig_w_pt + gap
        x_radio_pt = x_label_pt + widths['label'] + gap
        x_gene_pt = x_radio_pt + widths['radio'] + gap + reserve
        x_group_pt = x_gene_pt + widths['textbox'] + gap
        x_done_pt = x_group_pt + widths['group'] + gap
        x_cancel_pt = x_done_pt + widths['done'] + gap

        def frac(pt):
            return pt / fig_w_pt

        radio_w_pt = widths['radio']
        dot_x_pt, label_x_pt, content_pt = scaled_radio_layout_pt(
            VIEW_RADIO_OPTIONS, base, scale, 4.0 * scale)
        offset_pt = max(0.0, (radio_w_pt - content_pt) / 2)  # centered, like radio_layout_as_axes_fractions

        # 'Show:' hugs the first radio dot (right-aligned, half an em to its
        # left) rather than sitting at the left end of its own slot, where the
        # radio's centering left a wide gap between it and the options. The
        # slot's minimum width still keeps it clear of the window edge.
        first_dot_left_pt = x_radio_pt + offset_pt + dot_x_pt[0] - fontsize / 2
        show_label_text.set_horizontalalignment('right')
        show_label_text.set_x(frac(max(x_label_pt + mins['label'] - 0.4 * fontsize,
                                       first_dot_left_pt - 0.5 * fontsize)))
        show_label_text.set_fontsize(fontsize)
        view_radio_ax.set_position([frac(x_radio_pt), 0.02, frac(radio_w_pt), button_height])
        dot_x = tuple((x + offset_pt) / radio_w_pt for x in dot_x_pt)
        view_radio_state['dot_x'] = dot_x
        view_radio_dots.set_offsets(np.column_stack([dot_x, [0.5] * len(dot_x)]))
        view_radio_dots.set_sizes([fontsize ** 2] * len(dot_x))
        for text_artist, lx_pt in zip(view_radio_label_texts, label_x_pt):
            text_artist.set_x((lx_pt + offset_pt) / radio_w_pt)
            text_artist.set_fontsize(fontsize)

        gene_ax.set_position([frac(x_gene_pt), 0.02, frac(widths['textbox']), button_height])
        gene_textbox.label.set_fontsize(fontsize)
        gene_textbox.text_disp.set_fontsize(fontsize)
        refresh_gene_name_marks(gene_textbox)  # red names follow the new font size
        for text_artist in suggestion_texts:
            text_artist.set_fontsize(fontsize)
        dropdown_pos = dropdown_ax.get_position()
        dropdown_ax.set_position([frac(x_gene_pt), dropdown_pos.y0, frac(widths['textbox']), dropdown_pos.height])

        for button_ax, key, x_pt in ((group_button_ax, 'group', x_group_pt),
                                     (done_ax, 'done', x_done_pt),
                                     (cancel_ax, 'cancel', x_cancel_pt)):
            button_ax.set_position([frac(x_pt), 0.02, frac(widths[key]), button_height])
        for button in (group_button, done_button, cancel_button):
            button.label.set_fontsize(fontsize)
        group_list_pos = group_list_ax.get_position()
        group_list_ax.set_position([frac(x_group_pt), group_list_pos.y0, frac(widths['group']), group_list_pos.height])
        for text_artist in group_list_texts:
            text_artist.set_fontsize(fontsize)

    # The bottom row sets the shared font scale, so it has to run before the
    # overlays (already registered above) on every resize.
    post_resize_hooks.insert(0, layout_bottom_row)
    run_post_resize_hooks()

    if on_ready is not None:
        on_ready()

    going_into_gene_mode = (
        precomputed_gene_open is None and initial_view is not None
        and initial_view.get('mode') in ('gene', 'imputed_gene') and initial_view.get('gene')
    )
    if going_into_gene_mode:
        # Skip painting the class-colored image in the first draw below —
        # it's about to be replaced by the gene render anyway, so fully
        # rasterizing it first just to immediately discard it roughly
        # doubled total time to get to the image actually wanted. Made
        # visible again right after set_view_radio_selection below,
        # whether or not that render actually succeeds.
        img_artist.set_visible(False)
    # In the Groups view the coloring follows the Group level. Every new
    # window starts at 'class', which is the cached image, so this only
    # renders if that default changes (or opening into Gene falls back).
    recolor_classes_on_open = (
        not going_into_gene_mode and current_view['mode'] == 'standard'
        and group_by_state['field'] != 'class'
    )
    if recolor_classes_on_open:
        img_artist.set_visible(False)

    # Actually show the window *before* the initial_view render below, not
    # after (show_figure_blocking normally does this) — the render's
    # progress readout is only visible on screen if the window is already
    # mapped and has had at least one real draw (so blit_bg is populated);
    # otherwise the whole thing runs against an invisible window and only
    # the finished image ever appears, once show_figure_blocking finally
    # shows it — which read as "the window shows up on time, just without
    # any progress bar."
    # No center_figure_window() here — this window opens maximized (see
    # maximize_figure_window above), so there's no position left to center.
    raise_figure_window(fig)
    fig.canvas.manager.show()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    # A redundant-but-cheap re-confirmation of the same recompute already
    # forced right after maximize_figure_window() above (before AXES_BOTTOM/
    # button_height/the button row were even laid out) — kept here too in
    # case anything between there and here changed the canvas's real size.
    # Without *some* explicit recompute, the image axes would stay sized for
    # whatever it last had a real resize_event for (the bug this whole
    # on_configure/recompute_image_axes_position mechanism exists for) until
    # the user happened to trigger another resize.
    try:
        fig.canvas.manager.window.update_idletasks()
    except Exception:
        pass
    recompute_image_axes_position(*fig.canvas.get_width_height())
    run_post_resize_hooks()  # widgets too, for the same real canvas size

    if going_into_gene_mode:
        # Pre-fill the box (so it's visible/correct if the user later wants
        # to tweak it), then go through set_view_radio_selection rather
        # than calling on_show_gene directly, so the radio dot ends up
        # showing 'Gene'/'Imputed Gene' selected (or reverts to 'Groups' if
        # this fails) instead of silently disagreeing with what's actually
        # displayed.
        gene_textbox.set_val(initial_view['gene'])
        dropdown_ax.set_visible(False)  # set_val() re-triggers autocomplete filtering; not wanted here
        initial_idx = 2 if initial_view.get('mode') == 'imputed_gene' else 1
        set_view_radio_selection(initial_idx)
        if current_view['mode'] == 'standard':
            # The gene render failed and it fell back to Groups; color by level.
            recolor_classes_on_open = group_by_state['field'] != 'class'
        img_artist.set_visible(True)  # covers both a successful gene render and a fallback to standard view
    if recolor_classes_on_open:
        if not show_class_level(group_by_state['field']):
            img_artist.set_visible(True)  # fall back to the class-colored image

    show_figure_blocking(fig)

    # Carry this window's view over to the next section opened (see the
    # session_view_settings docstring entry). Runs however the window was
    # closed, since show_figure_blocking only returns once it's gone.
    if session_view_settings is not None:
        session_view_settings['mode'] = current_view['mode']
        session_view_settings['gene'] = gene_textbox.text.strip()

    return pending


def filter_by_rois(adata, abc_cache, rois, section_col=SECTION_COL):
    """Subset adata to cells falling within any of `rois` (each a dict with
    'section', 'x_min', 'x_max', 'y_min', 'y_max'). A cell must match both
    the ROI's section and its spatial bounds. No-op if rois is empty/None.
    Assumes adata.obs[section_col] is already populated (e.g. by a prior
    filter_by_sections() call)."""
    if not rois:
        return adata
    if section_col not in adata.obs.columns:
        print(f"Warning: '{section_col}' column not available; cannot filter by ROIs.")
        return adata
    spatial = load_section_spatial_coords(adata, abc_cache)
    if spatial is None or 'x' not in spatial.columns or 'y' not in spatial.columns:
        print("Warning: spatial x/y coordinates not available; cannot filter by ROIs.")
        return adata

    xs = pd.to_numeric(spatial['x'], errors='coerce').to_numpy()
    ys = pd.to_numeric(spatial['y'], errors='coerce').to_numpy()
    sections = adata.obs[section_col].to_numpy()

    keep = np.zeros(len(xs), dtype=bool)
    for roi in rois:
        section_mask = sections == roi['section']
        bounds_mask = (
            (xs >= roi['x_min']) & (xs <= roi['x_max'])
            & (ys >= roi['y_min']) & (ys <= roi['y_max'])
        )
        keep |= section_mask & bounds_mask

    filtered = materialize_subset(adata, keep)
    n_sections = len({roi['section'] for roi in rois})
    print(f"Filtered to {len(rois)} ROI(s) across {n_sections} section(s): "
          f"{filtered.n_obs} of {adata.n_obs} cells kept.")
    return filtered


def save_roi_map(adata, abc_cache, section_series, rois, output_path):
    """Save a grid of per-section scatter plots — one panel per section that
    has at least one ROI, each showing all of that section's cells (gray
    non-neurons, colored neurons) plus its ROI rectangle(s) outlined in red —
    as a visual reference for where the ROIs were drawn."""
    spatial = load_section_spatial_coords(adata, abc_cache)
    if spatial is None or 'x' not in spatial.columns or 'y' not in spatial.columns:
        print("Warning: spatial coordinates not available; cannot save ROI map.")
        return

    print(f"Generating ROI map ...")
    rois_by_section = {}
    for roi in rois:
        rois_by_section.setdefault(roi['section'], []).append(roi)
    sections = sorted_sections_descending(list(rois_by_section.keys()))
    n = len(sections)

    ncols = max(1, math.ceil(math.sqrt(n)))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4), squeeze=False)
    axes_flat = axes.ravel()

    section_values = section_series.to_numpy()
    xs_all = pd.to_numeric(spatial['x'], errors='coerce').to_numpy()
    ys_all = pd.to_numeric(spatial['y'], errors='coerce').to_numpy()
    class_ids_all = None
    if 'class' in spatial.columns:
        class_ids_all = extract_leading_numeric_id(spatial['class']).to_numpy()
    point_colors_all, neuron_mask_all = build_neuron_class_colors(class_ids_all)

    for i, section in enumerate(sections):
        ax = axes_flat[i]
        mask = section_values == section
        xs, ys = xs_all[mask], ys_all[mask]
        valid = ~(np.isnan(xs) | np.isnan(ys))
        xs, ys = xs[valid], ys[valid]
        if neuron_mask_all is not None:
            is_neuron = neuron_mask_all[mask][valid]
            colors = point_colors_all[mask][valid]
            scatter_gray_then_colored(ax, xs, ys, is_neuron, colors, gray_size=1, colored_size=2)
        else:
            ax.scatter(xs, ys, s=1, c='steelblue', linewidths=0)

        section_rois = rois_by_section[section]
        for roi in section_rois:
            rect = Rectangle(
                (roi['x_min'], roi['y_min']),
                roi['x_max'] - roi['x_min'], roi['y_max'] - roi['y_min'],
                # Explicit zorder so the outline always draws over the
                # scatter dots (their default zorder ties with a patch's,
                # which would otherwise leave the ordering to insertion
                # order and risk the line looking broken where dots overlap it).
                linewidth=2, edgecolor='red', facecolor='none', zorder=5,
            )
            ax.add_patch(rect)

        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        roi_word = 'ROI' if len(section_rois) == 1 else 'ROIs'
        ax.set_title(f"{sanitize_section_token(section)} ({len(section_rois)} {roi_word})", fontsize=9)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    print(f"Saving ROI map to {output_path} ...", end='', flush=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    fig.savefig(output_path.with_suffix('.svg'))
    plt.close(fig)
    print(f" Done saving")


def save_group_spatial_maps(adata, abc_cache, section_series, rois, whole_sections, output_dir,
                             group_col='subclass', cell_type_selection='All'):
    """For each section with at least one ROI or a whole-section pick,
    find which `group_col` categories (e.g. 'subclass' or 'supertype')
    actually occur there among cells of `cell_type_selection` ('Neurons',
    'NonNeurons', or 'All') — within the ROI bounds specifically for a
    section with ROI(s) (a category elsewhere in that section but outside
    every ROI doesn't count), or anywhere in the section for a
    whole-section pick (no ROI to restrict to). Take the union of
    qualifying categories across all such sections, split it into groups of
    up to 10, and save one spatial-scatter grid figure per group into
    `output_dir` (one panel per relevant section, same grid layout as
    save_roi_map). Every cell not in the current group's *qualifying*
    categories — including any cell of the excluded cell type — is drawn
    gray so the section's outline stays visible; each section's ROI
    rectangle(s), if any, are outlined in red on top. Each group's up to 10
    categories get one of SPATIAL_MAP_COLORS each, assigned by position
    within the group, so the same 10 colors are reused (in the same order)
    across every group's figure rather than each group inventing its own
    palette. No-op (prints a message) if there are no relevant sections or
    no qualifying categories."""
    roi_sections = sorted_sections_descending(list({roi['section'] for roi in rois})) if rois else []
    relevant_sections = sorted_sections_descending(list(set(roi_sections) | set(whole_sections or [])))
    if not relevant_sections:
        relevant_sections = sorted_sections_descending(list(section_series.dropna().unique()))
        print(f"No ROI/whole-section picks; defaulting to all {len(relevant_sections)} section(s) for {group_col} spatial maps.")

    spatial = load_section_spatial_coords(adata, abc_cache)
    if spatial is None or 'x' not in spatial.columns or 'y' not in spatial.columns or group_col not in spatial.columns:
        print(f"Warning: spatial x/y/{group_col} columns not available; cannot save {group_col} spatial maps.")
        return

    rois_by_section = {}
    for roi in rois:
        rois_by_section.setdefault(roi['section'], []).append(roi)

    section_values = section_series.reindex(spatial.index).to_numpy()
    xs_all = pd.to_numeric(spatial['x'], errors='coerce').to_numpy()
    ys_all = pd.to_numeric(spatial['y'], errors='coerce').to_numpy()
    group_ids_all = extract_leading_numeric_id(spatial[group_col]).to_numpy()
    group_labels_all = spatial[group_col].astype(str).to_numpy()

    # Restrict everything below to the selected cell type, same rule as
    # filter_by_cell_type(): non-neuron classes are 30-34, everything else
    # valid counts as neuron. Cells excluded here are treated exactly like
    # cells with no `group_col` category at all — always gray, never
    # counted toward a section's qualifying categories.
    if cell_type_selection != 'All' and 'class' in spatial.columns:
        class_ids_all = extract_leading_numeric_id(spatial['class']).to_numpy()
        valid_class = ~np.isnan(class_ids_all)
        is_non_neuron_all = valid_class & np.isin(class_ids_all, list(NON_NEURON_CLASS_IDS))
        if cell_type_selection == 'NonNeurons':
            cell_type_mask_all = valid_class & is_non_neuron_all
        else:  # 'Neurons'
            cell_type_mask_all = valid_class & ~is_non_neuron_all
    else:
        cell_type_mask_all = np.ones(len(xs_all), dtype=bool)

    # For each relevant section, the set of category IDs that qualify there
    # (within-ROI only, if the section has ROIs) — plus a global id->label
    # lookup and the union of all qualifying IDs across every section.
    section_qualifying = {}
    qualifying_ids = set()
    id_to_label = {}
    for section in relevant_sections:
        sec_mask = section_values == section
        if section in rois_by_section:
            target_mask = np.zeros(len(xs_all), dtype=bool)
            for roi in rois_by_section[section]:
                target_mask |= (
                    sec_mask
                    & (xs_all >= roi['x_min']) & (xs_all <= roi['x_max'])
                    & (ys_all >= roi['y_min']) & (ys_all <= roi['y_max'])
                )
        else:
            target_mask = sec_mask
        target_mask = target_mask & cell_type_mask_all
        ids_here = group_ids_all[target_mask]
        valid_ids = set(ids_here[~np.isnan(ids_here)].tolist())
        section_qualifying[section] = valid_ids
        qualifying_ids.update(valid_ids)
        for cid, label in zip(group_ids_all[sec_mask & cell_type_mask_all], group_labels_all[sec_mask & cell_type_mask_all]):
            if not np.isnan(cid) and cid not in id_to_label:
                id_to_label[cid] = label

    if not qualifying_ids:
        print(f"No {group_col} categories found within the selected ROIs/whole sections; skipping spatial maps.")
        return

    sorted_ids = sorted(qualifying_ids)
    groups = [sorted_ids[i:i + 10] for i in range(0, len(sorted_ids), 10)]

    ncols = max(1, math.ceil(math.sqrt(len(relevant_sections))))
    nrows = math.ceil(len(relevant_sections) / ncols)

    # A shared axis span (data units) — separate for x and y — and a
    # per-section centroid, so every panel is drawn at the same physical
    # scale instead of each autoscaling to fill its cell with just its own
    # data — otherwise an anatomically small section (e.g. the olfactory
    # bulb) gets zoomed to look just as large on the page as a full coronal
    # section. Width and height are sized independently (rather than one
    # isotropic span reused for both) because sections are typically wider
    # than tall — a single shared span would end up sized to the width (the
    # larger of the two for most sections), leaving a lot of unnecessary
    # blank margin above and below every panel even though its horizontal
    # fit was already tight. Each is sized to the SPAN_PERCENTILE-th
    # percentile of section extents (not the strict max) plus 5% padding —
    # using the single largest section would leave most panels surrounded
    # by a lot of blank margin; a percentile trades that off deliberately,
    # clipping the few sections above it in this figure only (the full data
    # is still used for actual processing) for noticeably less wasted space
    # in the common case. Every panel still shares the exact same (width,
    # height) pair, so set_aspect('equal') still shrinks every panel by the
    # same amount within its grid cell — which is what keeps row gaps even.
    SPAN_PERCENTILE = 100  # temporarily testing without percentile clipping — was 90
    section_centroids = {}
    half_widths, half_heights = [], []
    for section in relevant_sections:
        sec_mask = section_values == section
        xs_sec, ys_sec = xs_all[sec_mask], ys_all[sec_mask]
        valid_sec = ~(np.isnan(xs_sec) | np.isnan(ys_sec))
        xs_sec, ys_sec = xs_sec[valid_sec], ys_sec[valid_sec]
        if len(xs_sec) == 0:
            continue
        x_min, x_max = xs_sec.min(), xs_sec.max()
        y_min, y_max = ys_sec.min(), ys_sec.max()
        section_centroids[section] = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        half_widths.append((x_max - x_min) / 2)
        half_heights.append((y_max - y_min) / 2)
    shared_half_width = np.percentile(half_widths, SPAN_PERCENTILE) * 1.05 if half_widths else 1.0
    shared_half_height = np.percentile(half_heights, SPAN_PERCENTILE) * 1.05 if half_heights else 1.0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clear out any group*.png/.svg files from a previous run in this folder
    # first — otherwise a run that produces fewer groups than a prior one
    # (e.g. fewer qualifying subclasses this time) would leave a stale
    # leftover file that looks like part of the current output.
    for pattern in ('group*.png', 'group*.svg'):
        for stale in output_dir.glob(pattern):
            stale.unlink()

    for group_idx, group_ids in enumerate(groups, start=1):
        color_by_cid = {cid: SPATIAL_MAP_COLORS[i % len(SPATIAL_MAP_COLORS)] for i, cid in enumerate(group_ids)}

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4), squeeze=False)
        axes_flat = axes.ravel()

        for i, section in enumerate(relevant_sections):
            ax = axes_flat[i]
            sec_mask = section_values == section
            xs, ys = xs_all[sec_mask], ys_all[sec_mask]
            ids_here = group_ids_all[sec_mask]
            in_cell_type = cell_type_mask_all[sec_mask]
            valid = ~(np.isnan(xs) | np.isnan(ys))
            xs, ys, ids_here, in_cell_type = xs[valid], ys[valid], ids_here[valid], in_cell_type[valid]

            # Only this section's qualifying categories (within its own ROI,
            # or anywhere if it's a whole-section pick), and only cells of
            # the selected cell type, get colored here — everything else
            # (a group category that qualified via a *different* section,
            # the wrong cell type, or no category at all) falls back to gray.
            active_group = [cid for cid in group_ids if cid in section_qualifying.get(section, set())]
            colored_mask = in_cell_type & np.isin(ids_here, active_group)

            ax.scatter(xs[~colored_mask], ys[~colored_mask], s=1, color=DEFAULT_GREY_RGBA, linewidths=0)
            for cid in active_group:
                cid_mask = colored_mask & (ids_here == cid)
                if not cid_mask.any():
                    continue
                ax.scatter(xs[cid_mask], ys[cid_mask], s=2, color=color_by_cid[cid], linewidths=0, alpha=0.5)

            for roi in rois_by_section.get(section, []):
                rect = Rectangle(
                    (roi['x_min'], roi['y_min']),
                    roi['x_max'] - roi['x_min'], roi['y_max'] - roi['y_min'],
                    linewidth=2, edgecolor='red', facecolor='none', zorder=5,
                )
                ax.add_patch(rect)

            cx, cy = section_centroids.get(section, (0.0, 0.0))
            ax.set_xlim(cx - shared_half_width, cx + shared_half_width)
            ax.set_ylim(cy - shared_half_height, cy + shared_half_height)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(section, fontsize=9)

        for j in range(len(relevant_sections), len(axes_flat)):
            axes_flat[j].axis('off')

        # One shared legend for the whole figure — every panel uses the same
        # cid -> color mapping, so a per-panel legend would just repeat itself.
        legend_handles = [
            Line2D([0], [0], marker='o', linestyle='none', markersize=6,
                   color=color_by_cid[cid], label=id_to_label.get(cid, str(cid)))
            for cid in group_ids
        ]
        fig.legend(handles=legend_handles, loc='lower center', ncol=min(5, len(legend_handles)),
                   fontsize=7, bbox_to_anchor=(0.5, -0.02))

        group_path = output_dir / f'group{group_idx:02d}.png'
        fig.tight_layout(rect=(0, 0.06, 1, 1))  # leave room at the bottom for the legend
        fig.savefig(group_path, dpi=200, bbox_inches='tight')
        fig.savefig(group_path.with_suffix('.svg'), bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {group_col} spatial map group {group_idx}/{len(groups)} "
              f"({len(group_ids)} {group_col} categories) to {group_path}.")


def format_duration(seconds):
    """Format a duration in seconds as 'Hh MMm SS.Ss', dropping leading
    all-zero units (e.g. 45.3s stays as '45.3s', 125s becomes '2m 5.0s')."""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {secs:.1f}s"
    return f"{secs:.1f}s"


def resolve_out_folder(default_folder):
    """Ensure `default_folder` exists (creating it if needed) and return it
    for use as the output directory. If it can't be created or accessed —
    e.g. a network drive that isn't currently mounted on this machine —
    falls back to a native folder-picker dialog so the user can choose an
    alternate location instead of the script just crashing with an OSError.
    Exits the process if the user cancels that dialog."""
    try:
        Path(default_folder).mkdir(parents=True, exist_ok=True)
        return default_folder
    except OSError as e:
        print(f"Could not create/access the default output folder '{default_folder}' ({e}).")

    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        chosen = filedialog.askdirectory(
            title="Default output folder unavailable — choose an alternate output folder",
        )
        root.destroy()
    except Exception as e:
        print(f"Could not show folder picker ({e}).")
        chosen = None

    if not chosen:
        print("No output folder chosen; exiting.")
        sys.exit(0)
    Path(chosen).mkdir(parents=True, exist_ok=True)
    return chosen


def prompt_yes_no(question, default=None):
    """Loop until the user answers y/n to `question` (which should include
    its own trailing '(y/n): ' cue). Returns True for yes, False for no.
    If `default` is given (True/False) and the user just hits enter with no
    input, returns `default` instead of re-prompting."""
    while True:
        choice = input(question).strip().lower()
        if choice == '' and default is not None:
            return default
        if choice in ('y', 'yes'):
            return True
        if choice in ('n', 'no'):
            return False
        print("Please enter 'y' or 'n'.")


def confirm_whole_brain():
    """Ask for confirmation before proceeding with no sections or ROIs
    selected, which means processing every cell in the whole brain — a
    much slower run than a typical subset. Returns True if the caller
    should proceed."""
    return prompt_yes_no(
        "No sections or ROIs selected: this will process the entire brain, "
        "which is time-consuming. Continue? (y/n): "
    )


def prompt_subsample_choice(n_obs, default_cap):
    """Console prompt shown when the filtered cell count `n_obs` exceeds
    `default_cap`: let the user choose to use the full filtered dataset (no
    subsampling), restrict to `default_cap`, or subsample to a custom
    value. Returns the chosen subsample target (an int less than n_obs), or
    None to mean 'use every filtered cell, don't subsample'."""
    print(f"Filtered dataset has {n_obs} cells, which exceeds the default "
          f"subsample cap of {default_cap}.")
    while True:
        choice = input(
            f"Use the (f)ull dataset, (r)estrict to {default_cap} (default), "
            "or enter a (c)ustom number to subsample to? [f/r/c]: "
        ).strip().lower()
        if choice in ('f', 'full'):
            return None
        if choice in ('r', 'restrict', ''):
            return default_cap
        if choice in ('c', 'custom'):
            while True:
                raw = input(f"Enter number of cells to subsample to (1-{n_obs - 1}): ").strip()
                try:
                    n = int(raw)
                except ValueError:
                    print("Please enter a valid integer.")
                    continue
                if 1 <= n < n_obs:
                    return n
                print(f"Enter a value between 1 and {n_obs - 1}, "
                      "or choose 'f' to use the full dataset instead.")
        print("Please enter 'f', 'r', or 'c'.")


def prompt_roi_run_label():
    """Optional short, freeform label the user can add to an ROI run's
    folder name — purely a memory aid (e.g. 'hippocampus') alongside the
    auto-generated ROI hash, which is unique but not remotely readable.
    Blank input (just pressing Enter) skips it entirely, leaving the folder
    named exactly as it always was.

    Appended to region_suffix *before* run_suffix/session_prefix/run_folder
    are built from it, so the label becomes an ordinary part of this run's
    identity, the same way subsample_target already is — no changes needed
    anywhere else in the pipeline (early_cache_hit's glob, the later exact
    csv_path check, 'Load existing run', etc. all just see a longer folder
    name and work unmodified). The one consequence, also already true of
    subsample_target: re-running the identical ROI selection later only
    hits this same cached folder if the *same* label (or none) is given
    again — a different or missing label is a different folder name, hence
    a fresh run. Reusing a labeled run on purpose is what the startup
    panel's 'Load existing run' picker is for."""
    raw = input("Optional label for this ROI run's folder name (blank to skip): ").strip()
    if not raw:
        return ''
    # Alphanumerics kept as-is; every run of anything else (spaces,
    # punctuation) collapsed to a single hyphen, and any leading/trailing
    # hyphen trimmed — readable word separators instead of sanitize_
    # section_token's own all-punctuation-stripped style, which is fine for
    # a bare trailing section number but would squash a multi-word label
    # like "left hippocampus" into one illegible run of letters.
    label = re.sub(r'[^A-Za-z0-9]+', '-', raw).strip('-')
    return label


def confirm_overwrite(path):
    """If `path` already exists, ask the user whether to overwrite it.
    Returns True if the caller should proceed with writing the file."""
    path = Path(path)
    if not path.exists():
        return True
    return prompt_yes_no(f"File '{path}' already exists. Overwrite? (y/n): ")


def log_status(message):
    """print() prefixed with the elapsed time, in seconds (3 decimals, i.e.
    millisecond resolution), since `selection_confirmed_at` (a
    time.perf_counter() value set once the user confirms their section/ROI
    selection in the main script below) — so the sequence of pipeline steps
    that follow can be timed against each other at a glance. Only meant for
    status updates printed after that point; earlier messages (data
    loading, the picker itself) use plain print()."""
    elapsed_s = time.perf_counter() - selection_confirmed_at
    print(f"[+{elapsed_s:9.3f} s] {message}")


def open_with_default_viewer(path):
    """Open `path` with the OS's default application for its file type
    (e.g. Photos on Windows for a .png) — a separate process entirely from
    this script, so the image stays open (and interactive: zoom, close
    whenever the user wants) regardless of what this script does afterward
    or whether it's still running. Best-effort; prints a warning instead of
    raising if the platform isn't recognized or the launch fails."""
    try:
        system = platform.system()
        if system == 'Windows':
            os.startfile(path)
        elif system == 'Darwin':
            subprocess.Popen(['open', str(path)])
        else:
            subprocess.Popen(['xdg-open', str(path)])
    except Exception as e:
        print(f"Could not open {path} in the default viewer ({e}).")


def darken_color(color, factor=0.55):
    """Return an RGB tuple that is a darker version of `color` (any matplotlib
    color spec), by scaling its RGB channels toward black."""
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def add_category_id_labels(adata, ax, color_key, fontsize=6, darken_factor=0.55):
    """Label each category of adata.obs[color_key] at its UMAP centroid with
    its leading numeric ID (e.g. '30 Astro-Epen' -> '30'), in a font colored
    as a darker version of that category's plotted color. Each label is
    clipped to `ax`'s own box (clip_on=True) — a no-op for this function's
    original static, full-view callers (every centroid is always within a
    full-view axes' own bounds), but essential for a zoomed-in interactive
    view (see show_interactive_umap_window's 'All Subclasses' mode): without
    it, a label whose centroid falls outside the current xlim/ylim still
    renders at its mapped screen position regardless, which — for a window
    where another panel sits immediately outside that axes' box — can bleed
    text into whatever's next door instead of just vanishing off-screen.

    Returns the list of Text artists created (empty if there was nothing to
    label), so an interactive caller can keep hold of them — the interactive
    viewer rescales their font size on zoom, in step with its dots."""
    if color_key not in adata.obs.columns or 'X_umap' not in adata.obsm:
        return []
    obs_col = adata.obs[color_key]
    if not isinstance(obs_col.dtype, pd.CategoricalDtype):
        obs_col = obs_col.astype('category')
    categories = obs_col.cat.categories
    colors = adata.uns.get(f'{color_key}_colors')
    if colors is None or len(colors) != len(categories):
        print(f"No stored colors found for '{color_key}'; skipping ID labels.")
        return []

    coords = adata.obsm['X_umap']
    obs_values = obs_col.to_numpy()
    labels = []
    for category, color in zip(categories, colors):
        mask = obs_values == category
        if not mask.any():
            continue
        centroid = coords[mask].mean(axis=0)
        match = re.match(r'^\s*(\d+)', str(category))
        label = match.group(1) if match else str(category)
        labels.append(ax.text(
            centroid[0], centroid[1], label,
            fontsize=fontsize, color=darken_color(color, darken_factor),
            fontweight='bold', ha='center', va='center', clip_on=True,
        ))
    return labels


def save_umap_by_group(adata, color_key, save_dpi, output_dir, group_size=20):
    """Split adata.obs[color_key]'s categories into groups of up to
    `group_size`, saving one UMAP plot per group (group01.png/.svg,
    group02.png/.svg, ... into `output_dir`), each with only that group's
    cells colored and every other cell left at scanpy's na_color gray.

    This exists because scanpy silently colors *every* cell uniform gray —
    no per-category color at all — once a categorical column has more
    categories than it's willing to assign distinct colors to (observed
    around 103; well past the ~300+ subclasses or ~1000+ supertypes this
    atlas has). Restricting each individual plot's color column to just one
    group of categories (everything else set to NaN via the Categorical
    constructor's `categories=` argument, which scanpy then renders as
    na_color) keeps each plot's category count comfortably under that
    limit — it's also a lot less cluttered than trying to tell 300+ colors
    apart on a single plot regardless of whether scanpy can render them.

    No-op (prints a message) if `color_key` isn't in adata.obs. `save_dpi`
    should match the resolution the rest of Step 7's plots are saved at.

    When more than one group is needed (i.e. more than `group_size`
    categories), and every expected group*.png already exists in
    `output_dir`, asks whether to skip regenerating them entirely — with a
    single group this is fast enough not to bother asking."""
    if color_key not in adata.obs.columns:
        print(f"'{color_key}' not available on adata.obs; skipping grouped UMAP plots.")
        return

    obs_col = adata.obs[color_key]
    if not isinstance(obs_col.dtype, pd.CategoricalDtype):
        obs_col = obs_col.astype('category')

    # Sort categories by their leading numeric ID (same convention used by
    # save_group_spatial_maps) so groups are stable/meaningful rather than
    # whatever order pandas happened to assign.
    ids = extract_leading_numeric_id(pd.Series(obs_col.cat.categories))
    sorted_categories = [obs_col.cat.categories[i] for i in ids.sort_values(kind='stable').index]
    groups = [sorted_categories[i:i + group_size] for i in range(0, len(sorted_categories), group_size)]

    output_dir = Path(output_dir)
    # Regenerating is cheap with a single group (<= group_size categories),
    # but with many groups it means one sc.pl.umap() call per group, which
    # adds up — so when there's more than one, offer to skip entirely if
    # every expected output file is already sitting there from a previous
    # run, rather than silently redoing all of it every time.
    if len(groups) > 1:
        expected_files = [output_dir / f'group{i:02d}.png' for i in range(1, len(groups) + 1)]
        if all(f.exists() for f in expected_files):
            if prompt_yes_no(
                f"Found {len(expected_files)} existing {color_key} UMAP group plot(s) in "
                f"{output_dir}; skip regenerating them? (y/n): "
            ):
                print(f"Skipping {color_key} UMAP group plots; using the existing files in {output_dir}.")
                return

    output_dir.mkdir(parents=True, exist_ok=True)
    # Same reasoning as save_group_spatial_maps: clear stale files from a
    # previous run before writing, in case this run produces fewer groups.
    for pattern in ('group*.png', 'group*.svg'):
        for stale in output_dir.glob(pattern):
            stale.unlink()

    temp_col = f'__{color_key}_group'
    for group_idx, group_categories in enumerate(groups, start=1):
        # Categorical(..., categories=group_categories) maps any value not
        # in that list to NaN — exactly the "everything else gray" split
        # this function is for.
        adata.obs[temp_col] = pd.Categorical(obs_col, categories=group_categories)
        # temp_col is the same *name* on every iteration, but a different
        # set of categories each time (a different-sized group, generally)
        # — scanpy's sc.pl.umap doesn't always notice that and recompute:
        # if adata.uns[f'{temp_col}_colors'] is still sitting there from
        # the *previous* group's call, it can get reused as-is rather than
        # regenerated for this group's own (often different) category
        # count. Harmless when consecutive groups both have the full
        # group_size categories, but the last group of a level is usually
        # smaller — e.g. this file's subclass level ends in a 2-category
        # final group right after four 20-category ones — and reusing a
        # stale 20-color array there is exactly what produced add_
        # category_id_labels' "No stored colors found ... skipping ID
        # labels" warning: colors existed, just for the wrong-sized
        # category list. Clearing it first forces a fresh, correctly-sized
        # array every time.
        adata.uns.pop(f'{temp_col}_colors', None)
        ax = sc.pl.umap(adata, color=temp_col, size=2, show=False)
        add_category_id_labels(adata, ax, temp_col)
        group_path = output_dir / f'group{group_idx:02d}.png'
        ax.figure.savefig(group_path, dpi=save_dpi, bbox_inches='tight')
        ax.figure.savefig(group_path.with_suffix('.svg'), bbox_inches='tight')
        plt.close(ax.figure)
        print(f"Saved {color_key} UMAP group {group_idx}/{len(groups)} "
              f"({len(group_categories)} categories) to {group_path}.")
    del adata.obs[temp_col]


# Thresholds shared by every "Export DEGs" call — a gene only makes the
# exported list if its nominal (not multiple-testing-corrected — see
# compute_de_genes' own docstring for why) p-value is below DE_P_VALUE_MAX
# *and* its linear fold change clears one of the two DE_FOLD_CHANGE_*
# bounds (up or down).
DE_P_VALUE_MAX = 0.05
DE_FOLD_CHANGE_MIN_UP = 1.1
DE_FOLD_CHANGE_MAX_DOWN = 0.9


# A third gene's fully-saturated color, instead of pure (0, 0, 1) — plain
# blue reads noticeably darker than saturated red or green on most
# monitors (perceived luminance of blue is much lower than red/green at the
# same numeric intensity), so the "blue" channel targets this lighter,
# slightly desaturated blue instead. Also used by draw_multi_gene_legend's
# swatch/label for the 3rd gene, so the legend matches what's actually
# on screen.
MULTI_GENE_BRIGHT_BLUE = (64 / 255, 64 / 255, 1.0)


def multi_gene_rgb(norms, baseline_rgb):
    """RGB color(s) for simultaneous multi-gene expression display: each
    array in `norms` (2 or 3 already-normalized-to-[0, 1] values — a
    min-max scaling of that gene's own log-expression values, see callers)
    drives one color channel directly, in order — red, green, and (for
    three genes) blue. Each channel is a plain per-channel blend from
    `baseline_rgb`'s own component up to fully saturated:

        channel_c = baseline_c * (1 - norm) + 1 * norm

    With this window's own black baseline (MULTI_GENE_LOW_EXPRESSION_COLOR /
    MULTI_GENE_UMAP_FACECOLOR) that reduces to channel_c == norm directly —
    each gene's own normalized value *is* its channel's brightness — and
    genes mix additively: two genes both near 1 read as bright yellow,
    three as white, the standard multi-channel-fluorescence-overlay
    convention. (`baseline_rgb` need not be black in principle — a 2-gene
    call's unused blue channel, for one, still fades toward baseline_rgb's
    own blue component rather than always toward 0 — but a non-black
    baseline does mean a single highly-expressed gene not saturate to a
    *clean* red/green the way it would against black, since every other
    channel would sit at its own baseline value instead of dropping to 0.)

    The third gene (blue channel) is the one exception to "fully saturated
    == (0, 0, 1)": it instead blends toward MULTI_GENE_BRIGHT_BLUE (a
    lighter, slightly desaturated blue — see its own comment), combined via
    max() with red/green's own independent contribution so it never dims
    out whatever genes 1/2 already put into those channels — a cell bright
    in gene 1 (red) and gene 3 (blue) alone still saturates red fully; it
    just also gets a little lighter/pinker from blue's own brightening,
    rather than turning a harsh, pure magenta.

    Broadcasts over arrays of any shape. NaN in any input (a cell missing
    from that gene's own source — e.g. present in the MERFISH panel but
    not the imputed dataset, or vice versa) is treated as 0 expression for
    that gene, same as it simply not having been detected, rather than
    propagating to a NaN (and hence invisible/erroring) color."""
    channels = [np.nan_to_num(np.asarray(n, dtype=float), nan=0.0) for n in norms]
    while len(channels) < 3:  # a 2-gene call leaves blue undriven by any gene
        channels.append(np.zeros_like(channels[0]))
    stacked = np.stack(channels[:3], axis=-1)
    baseline = np.asarray(baseline_rgb, dtype=float)
    plain = baseline * (1.0 - stacked) + stacked
    blue_norm = stacked[..., 2]
    bright_blue_target = np.asarray(MULTI_GENE_BRIGHT_BLUE, dtype=float)
    brightened = baseline * (1.0 - blue_norm[..., None]) + bright_blue_target * blue_norm[..., None]
    return np.maximum(plain, brightened)


def mean_log2_expression(work_adata, mask, log1p_base=None):
    """Mean log2 expression — the mean *of the already-log-transformed
    values* (not the log of the mean) for the cells where `mask` is True,
    one value per gene, as a Series indexed by work_adata.var_names.

    `work_adata.X` must already be log-transformed (natural-log log1p if
    `log1p_base` is None, matching scanpy's own sc.pp.log1p default, or
    already-log2 if `log1p_base` is 2) — this only reads it, it doesn't
    normalize or log anything itself, since callers need different
    treatment before this point (raw counts need normalizing first; the
    imputed dataset is already log2 on disk and must NOT be logged again).
    Converted to log2 units here regardless of the source base, so the
    result always means what its name says no matter which caller it came
    from. Shared by compute_de_genes (its own mean_log2_group/mean_log2_
    rest columns) and export_expression's plain per-ID average, which
    needs the same number with no statistical test alongside it."""
    X = work_adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X)
    to_log2 = np.log2(log1p_base if log1p_base is not None else np.e)
    mask = np.asarray(mask)
    return pd.Series(X[mask].mean(axis=0) * to_log2, index=work_adata.var_names)


def compute_de_genes(work_adata, group_mask, log1p_base=None):
    """Differentially expressed genes for the cells where `group_mask` is
    True versus the rest of `work_adata`'s own cells, via scanpy's own
    sc.tl.rank_genes_groups(method='wilcoxon') — the Wilcoxon rank-sum
    test, matching what Seurat's FindMarkers uses by default, since this
    lab's earlier DE work was done there (comparable results, not a switch
    to a different statistical convention just because the tool changed).

    `work_adata.X` must already be log-transformed — this only runs the
    test, it doesn't normalize or log anything itself, since the two
    callers need different treatment before this point (raw counts need
    normalizing first; the imputed dataset is already log2 on disk and
    must NOT be normalized/logged again). `log1p_base` tells scanpy what
    base that log used: None for natural-log log1p (scanpy's own
    sc.pp.log1p default), or 2 for already-log2 data. This matters — it's
    what scanpy's own fold-change math inverts the log with internally
    (adata.uns['log1p']['base']) — get it wrong and every reported fold
    change is quietly computed against the wrong base, still self-
    consistent-looking but not actually the linear-scale ratio it claims
    to be.

    Returns a DataFrame with columns gene_symbol/ensembl_id/mean_log2_group/
    mean_log2_rest/log2_fold_change/fold_change/p_value/p_value_adj,
    already filtered to p_value < DE_P_VALUE_MAX and fold_change outside
    [DE_FOLD_CHANGE_MAX_DOWN, DE_FOLD_CHANGE_MIN_UP], sorted by fold_change
    descending — most upregulated first, most downregulated last, not
    grouped by effect size regardless of direction. p_value_adj
    (Benjamini-Hochberg) is reported for reference alongside the nominal
    p_value the filter actually uses, since with hundreds to thousands of
    genes tested at once, a flat p<0.05 on the nominal value alone will
    let some fraction through by chance — worth checking that column
    before treating a borderline hit as real.

    Filtering on the *nominal* p-value (not p_value_adj) is a deliberate
    choice to match the caller's exact spec, not an oversight — the
    multiple-testing-corrected value is included specifically so it's not
    lost, not to quietly override that choice."""
    work_adata = work_adata.copy()  # rank_genes_groups writes into .uns; never mutate the caller's own object
    if work_adata.X.dtype == np.float16:
        # The imputed dataset (and possibly a counts layer) can be stored
        # as float16 on disk to save space — scipy.sparse won't even
        # construct a float16 matrix at all, and scanpy's own rank_genes_
        # groups rejects a *dense* float16 array too (raises
        # NotImplementedError('float16'), which is genuinely all the info
        # that exception carries — no context, just the dtype's own str()).
        # float32 is plenty of precision for a log-expression value; this
        # only ever needs to be a working copy for this one DE run, not a
        # permanent upcast of anything on disk.
        work_adata.X = work_adata.X.astype(np.float32)
    if log1p_base is not None:
        work_adata.uns['log1p'] = {'base': log1p_base}
    else:
        work_adata.uns.pop('log1p', None)  # ensures scanpy assumes natural log, not a stale base carried over from elsewhere
    group_mask = np.asarray(group_mask)
    work_adata.obs['_de_group'] = pd.Categorical(np.where(group_mask, 'specified', 'rest'))

    sc.tl.rank_genes_groups(work_adata, groupby='_de_group', groups=['specified'], reference='rest',
                             method='wilcoxon')
    result = sc.get.rank_genes_groups_df(work_adata, group='specified')

    # (result['logfoldchanges'] needs no such conversion — scanpy's own
    # rank_genes_groups already reports that in log2 unconditionally, using
    # log1p_base internally only to invert the log correctly before
    # recomputing the ratio; see this function's own docstring.)
    mean_group = mean_log2_expression(work_adata, group_mask, log1p_base)
    mean_rest = mean_log2_expression(work_adata, ~group_mask, log1p_base)

    # result['names'] comes from work_adata.var_names — in this atlas's own
    # convention that's the Ensembl gene ID (ENSMUSG...), not the common
    # name (Th, Drd2, ...), which instead lives in the separate
    # var['gene_symbol'] column (same column find_gene_index already
    # searches elsewhere in this file). Both are exported, clearly
    # distinguished by name, rather than the exported column being called
    # 'gene_symbol' while actually holding the Ensembl ID.
    gene_symbol = work_adata.var['gene_symbol'].reindex(result['names']).values

    out = pd.DataFrame({
        'gene_symbol': gene_symbol,
        'ensembl_id': result['names'].values,
        'mean_log2_group': mean_group.loc[result['names']].values,
        'mean_log2_rest': mean_rest.loc[result['names']].values,
        'log2_fold_change': result['logfoldchanges'].values,
        'fold_change': 2.0 ** result['logfoldchanges'].values,
        'p_value': result['pvals'].values,
        'p_value_adj': result['pvals_adj'].values,
    })
    out = out[
        (out['p_value'] < DE_P_VALUE_MAX)
        & ((out['fold_change'] > DE_FOLD_CHANGE_MIN_UP) | (out['fold_change'] < DE_FOLD_CHANGE_MAX_DOWN))
    ]
    return out.sort_values('fold_change', ascending=False).reset_index(drop=True)


def show_interactive_umap_window(adata, abc_cache, imputed_state=None, adata_backed=None, rois=None,
                                  run_folder=None):
    """Interactive UMAP viewer, opened once the embedding (adata.obsm[
    'X_umap']) is computed. A vertical hand-rolled radio on the right picks
    what colors the scatter:
      - All Subclasses: every cell colored by its 'subclass' category (via
        scanpy's own categorical coloring — same convention, including its
        color-count cap, as the old optional group plots) with each
        category's ID number labeled at its centroid (add_category_id_
        labels) instead of a legend, which wouldn't fit hundreds of
        categories anyway. No query needed.
      - Single Subclass: one subclass ID, typed below, highlighted red
        against gray — the old hardcoded-268 static plot, now with any ID.
      - Gene: raw counts from adata.layers['counts'], log1p'd — not
        adata.X itself, which by this point has been normalized/scaled/
        clipped for PCA and no longer holds interpretable expression
        values.
      - Imputed Gene: the imputed dataset's log2 values, matched to these
        cells by ID, same convention as the single-section picker's gene
        view.
    A text box below the radio (hidden for All Subclasses, which has
    nothing to type) takes the corresponding query and Enter (or the Show
    button) re-colors the plot; the gene box autocompletes against
    whichever gene panel the current mode implies. Gene/Imputed Gene's
    colorbar draws into its own permanently-reserved, fixed-size axes
    (cax=, not ax=) — so the main plot's size never changes switching
    between color-by modes, colorbar or not.

    Left of the UMAP scatter is a brain-section grid: one real Axes per
    section actually present in `adata` (post filter/subsample — i.e.
    exactly the sections that ended up in this run's ROI/whole-section
    selection), each showing that section's cells in gray, all sharing one
    physical scale (so an anatomically small section isn't zoomed to look
    as large as a full coronal one — see the SPAN_PERCENTILE block below)
    but with an independent pan/zoom per panel. Hovering a UMAP point
    (after a short hold, like prompt_subregion_selection's own hover-to-
    identify-cell — deferred, not looked up on every raw mouse-move event,
    so a ~200k-cell nearest-point search never runs more often than the
    mouse actually settles) highlights that cell in both the UMAP itself
    and, if its section is currently in the grid, that one section panel
    (every other panel's highlight is hidden) — and fills in a status line
    below with its class/subclass/supertype/cluster. Both the UMAP scatter
    and every section panel support mouse-wheel zoom and left-drag pan,
    each independently (see pannable_axes/on_scroll_zoom/on_press_pan
    below) — zooming out is clamped per-axes to that axes' own "home"
    (fully-zoomed-out) extent, computed once at setup.

    `adata_backed`, if given, is the original unfiltered backed AnnData
    from Step 1 (the same one save_roi_map/save_group_spatial_maps already
    use, for the same reason) — each section panel's background shows
    *every* cell of that section (via get_section_labels/load_section_
    spatial_coords on `adata_backed`, not `adata`), not just the ones that
    survived this run's own filtering/subsampling. Omit for a panel
    background made of only this run's own cells, same as before this was
    added. `rois`, if given, is the same list of {'section','x_min',
    'x_max','y_min','y_max'} dicts prompt_section_selection_gui collects —
    drawn as dashed orange rectangles on the matching section panel(s),
    same convention as that picker's own ROI borders; entries for whole-
    section picks (blank x_min/x_max/y_min/y_max) are skipped, since
    there's no meaningful rectangle to draw for "the whole section".

    `imputed_state`, if given, is the same {'adata':, 'load_thread':,
    'load_error':} dict prompt_section_selection_gui uses (see
    ensure_imputed_gene_dataset_loaded) — reused here so 'Imputed Gene'
    doesn't pay for a second load if the picker session already loaded it.
    Pass None (a fresh dict is created) if it wasn't, or the picker's GUI
    wasn't used at all (console fallback).

    Blocks until the window is closed, same as every other window in this
    file (see show_figure_blocking)."""
    if imputed_state is None:
        imputed_state = {'adata': None, 'load_thread': None, 'load_error': None}

    if 'X_umap' not in adata.obsm:
        print("No UMAP embedding available; skipping the interactive UMAP viewer.")
        return

    # This window builds one Axes (plus its own scatter/text/spine
    # artists) per brain section, on top of the UMAP's own scatter — many
    # thousands of Artist objects in total, and matplotlib Artists are
    # notorious for participating in reference cycles (Axes <-> Figure <->
    # Artist back-references), which CPython's cyclic GC can only reclaim
    # via a full generation-2 scan, not plain refcounting. With this many
    # objects, that scan is real (if not, it turned out, the dominant cost
    # behind the sluggishness this was originally chasing — see
    # make_textbox_stop_typing_blit_fast for that) — cheap insurance to
    # keep it from triggering mid-interaction regardless. Python's
    # automatic GC triggers it based on allocation count, which correlates
    # with UI activity here (each hover/redraw allocates
    # more artists), so it fires periodically *during* interaction: the
    # whole interpreter (and therefore Tk's event loop, which is just
    # Python callbacks) pauses mid-scan when it does. Non-cyclic garbage
    # (the vast majority) still gets freed immediately via refcounting
    # regardless —
    # this only defers the *cyclic* collector, at the cost of temporarily
    # higher memory use, for the lifetime of this one window.
    gc_was_enabled = gc.isenabled()
    gc.disable()

    log_status("Step 9: Preparing interactive UMAP viewer data...")
    coords = adata.obsm['X_umap']
    has_subclass = 'subclass' in adata.obs.columns
    subclass_ids = extract_leading_numeric_id(adata.obs['subclass']) if has_subclass else None
    has_counts = 'counts' in adata.layers

    # Per-cell spatial x/y (for the section-grid panel on the left) and
    # section labels — x/y are never merged onto adata.obs itself (see
    # load_section_spatial_coords's own docstring), so this is a full
    # metadata-CSV re-read, aligned 1:1 with `coords`/adata.obs.index via
    # that function's own reindex(). Done synchronously here, like every
    # other one-time setup step in this pipeline (Step 2's own metadata
    # read included) — only the *interactive* part of this window (hover
    # lookups, further down) needs to stay responsive, not this one-time
    # load before the window even opens.
    has_section_col = SECTION_COL in adata.obs.columns
    section_spatial = load_section_spatial_coords(adata, abc_cache) if has_section_col else None
    has_spatial = (
        section_spatial is not None
        and 'x' in section_spatial.columns and 'y' in section_spatial.columns
    )
    if has_section_col and not has_spatial:
        print("Warning: spatial coordinates unavailable; the section grid will be empty.")

    # Per-cell arrays, all aligned by position with `coords`/adata.obs —
    # used both to build the section grid below and to answer "what is
    # this hovered cell" during hover cross-linking.
    cell_section = adata.obs[SECTION_COL].to_numpy() if has_section_col else None
    cell_x = pd.to_numeric(section_spatial['x'], errors='coerce').to_numpy() if has_spatial else None
    cell_y = pd.to_numeric(section_spatial['y'], errors='coerce').to_numpy() if has_spatial else None
    cell_class = adata.obs['class'].to_numpy() if 'class' in adata.obs.columns else None
    cell_subclass = adata.obs['subclass'].to_numpy() if has_subclass else None
    cell_supertype = adata.obs['supertype'].to_numpy() if 'supertype' in adata.obs.columns else None
    cell_cluster = adata.obs['cluster'].to_numpy() if 'cluster' in adata.obs.columns else None
    # scanpy's own Leiden clustering (Step 6), rather than an ABC annotation
    # — absent for runs computed before it was added, or whose Step 6
    # couldn't run it, in which case the level selector reports it as
    # unavailable the same way it would any missing column.
    cell_leiden = adata.obs[LEIDEN_KEY].astype(str).to_numpy() if LEIDEN_KEY in adata.obs.columns else None
    # Cell ID -> position in `coords`/adata — the join key for the reverse
    # direction of hover cross-linking (section panel -> UMAP): a section
    # panel's own background dots are drawn from adata_backed's full cell
    # set (see background_x/y/section below), a superset of adata's own,
    # so "which UMAP point (if any) is this hovered section-panel point"
    # has to be answered by matching cell IDs, not position.
    cell_id_to_umap_idx = {cid: i for i, cid in enumerate(adata.obs_names)}
    sections_present = (
        sorted_sections_descending(adata.obs[SECTION_COL].dropna().unique().tolist())
        if has_section_col else []
    )

    # All cells of each section (not just the ones that survived filtering/
    # subsampling into `adata`) — for background context in the section
    # grid, so each panel shows the whole section, not just the sparse
    # subset that happened to end up in this run. `adata_backed` is the
    # original, unfiltered backed AnnData from Step 1 (same one save_roi_
    # map/save_group_spatial_maps already use for this exact reason) —
    # get_section_labels() is used here (not adata_backed.obs[SECTION_COL]
    # directly) since that column isn't guaranteed to be merged onto
    # adata_backed.obs's own metadata in every code path (e.g. an early
    # cache-hit run only merges it onto the already-filtered `adata`).
    background_section = get_section_labels(adata_backed, abc_cache) if adata_backed is not None else None
    if adata_backed is not None:
        log_status("Step 9: Loading background spatial data for section panels...")
    background_spatial = load_section_spatial_coords(adata_backed, abc_cache) if adata_backed is not None else None
    has_background = (
        background_section is not None and background_spatial is not None
        and 'x' in background_spatial.columns and 'y' in background_spatial.columns
    )
    if adata_backed is not None and not has_background:
        print("Warning: could not load all-cell background context for the section grid; "
              "showing only this run's own cells instead.")
    background_section_arr = background_section.to_numpy() if has_background else None
    background_x = pd.to_numeric(background_spatial['x'], errors='coerce').to_numpy() if has_background else None
    background_y = pd.to_numeric(background_spatial['y'], errors='coerce').to_numpy() if has_background else None
    # The four annotation granularities the level selector (further down)
    # switches between — coarsest (class) to finest (cluster). Both the
    # per-cell (adata-aligned) and background (all-cells-in-section)
    # versions are precomputed for all four up front, not just whichever
    # is initially selected, so switching levels later is an instant
    # dict lookup rather than a fresh column extraction.
    # The first four are the ABC taxonomy's own granularities (coarsest to
    # finest); LEIDEN_KEY is appended as a fifth, distinct in kind — this
    # run's own de-novo clustering rather than a published annotation, which
    # is why it sorts after them rather than among them.
    LEVEL_OPTIONS = ('class', 'subclass', 'supertype', 'cluster', LEIDEN_KEY)
    cell_level_arrays = {
        'class': cell_class, 'subclass': cell_subclass, 'supertype': cell_supertype, 'cluster': cell_cluster,
        LEIDEN_KEY: cell_leiden,
    }
    # load_section_spatial_coords also pulls these same four columns
    # straight from the metadata CSV when present — used below for the
    # same-level "family" highlight on the section panels, covering every
    # matching cell in the section (not just the ones that survived into
    # this run's own `adata`), same as the grey background dots already
    # do.
    background_level_arrays = {
        level: (
            background_spatial[level].to_numpy()
            if has_background and level in background_spatial.columns else None
        )
        for level in LEVEL_OPTIONS
    }
    # Leading numeric ID per level (e.g. '268 L5 PT CTX Glut' -> 268), used
    # by "Single <Level>" mode to match a typed ID number — precomputed for
    # all four the same way, and only ever for adata's own cells (not
    # background_*), since typed-ID matching is a UMAP-space concept.
    level_ids_arrays = {
        level: (extract_leading_numeric_id(adata.obs[level]) if level in adata.obs.columns else None)
        for level in LEVEL_OPTIONS
    }
    # Same leading-numeric-ID extraction, but for background_level_arrays
    # (every cell of every section, not just this run's own adata) —
    # precomputed once per level here, over the whole (multi-million-row)
    # background array, rather than once per *panel* per level inside the
    # section-panel build loop below (arr[sec_mask][valid] is just numpy
    # indexing into this). Doing the regex-based extraction 295 times (59
    # panels x 5 levels) on ~66k-row slices instead of ~5 times on the
    # full array was previously the single largest cost in opening this
    # window (~11.5s of the section-panel build loop's ~12.8s).
    background_level_ids_arrays = {
        level: (extract_leading_numeric_id(pd.Series(arr)).to_numpy() if arr is not None else None)
        for level, arr in background_level_arrays.items()
    }

    # UMAP_POINT_SIZE / UMAP_LABEL_FONTSIZE and the rest of this window's
    # tunables now live in the TUNABLE CONSTANTS block at the top of the file.
    # SIDEBAR_FONTSIZE stays here: it's the radio/textbox/button/status font,
    # matched to UI_BUTTON_FONTSIZE (itself measured from the screen at
    # import) so this window's sidebar text reads the same size as every
    # other window's — not something to set by hand.
    SIDEBAR_FONTSIZE = UI_BUTTON_FONTSIZE

    umap_window_figsize = compute_figsize_for_screen_height(24 / 10, default=(24, 10))
    fig = plt.figure(figsize=umap_window_figsize)
    normalize_tk_scaling(fig)
    # run_folder's own name (not the full path) identifies which run this
    # viewer belongs to — the per-run subfolder of out_folder every output
    # file for the run lands in, so it's the same token the user sees on
    # disk. Omitted entirely when the caller didn't pass one, rather than
    # showing an empty separator.
    umap_window_title = "Interactive UMAP viewer"
    if run_folder is not None:
        umap_window_title = f"{Path(run_folder).name} — {umap_window_title}"
    fig.canvas.manager.set_window_title(umap_window_title)
    # Forced early (rather than waiting for show_figure_blocking's own
    # show() call much later, after the rest of this function has built out
    # every axes/widget below) — without this, the window opened at its
    # intended size, then visibly snapped smaller once Tk actually realized
    # it (whatever residual DPI-scaling mismatch normalize_tk_scaling alone
    # didn't fully correct for), and manually dragging it bigger afterward
    # just triggered the same snap-back on the next <Configure> event — only
    # an OS-level maximize (a single, final geometry Tk doesn't fight)
    # actually stuck. Reasserting the real intended geometry right after an
    # eager show(), the same early-realization trick prompt_subregion_
    # selection's own maximize_figure_window call already relies on, catches
    # and corrects that mismatch before the user ever sees it.
    try:
        fig.canvas.manager.show()
        assert_figure_content_size(fig, *umap_window_figsize)
    except Exception:
        pass

    # Done here, right after the window is realized but *before* any of the
    # layout below is computed, so everything that reads fig.get_size_inches()
    # while building (the section grid's own rows/cols, compute_area_bottom,
    # the status panel) sees the final size and lays out against it once,
    # rather than being built at an oversized figure and then relaid out by
    # a resize event afterward.
    fit_figure_window_to_work_area(fig)

    # Three side-by-side regions, left to right: the brain-section grid,
    # the UMAP scatter, and the sidebar (mode radio/query/buttons) — fixed
    # figure-fraction regions, not gridspec, so widget positions further
    # down can be figured relative to these same margins without also
    # depending on gridspec's own coordinate quirks. Derived from two
    # draggable boundary positions (boundary1 between grid/UMAP, boundary2
    # between UMAP/sidebar — see the resize-handle wiring near the end of
    # this function, which is the only thing that ever changes boundary1/
    # boundary2 after this point) rather than hardcoded independently, so
    # there's exactly one source of truth for where each region starts/
    # ends, whether it's this initial layout or a live drag later.
    # GRID_LEFT and SIDEBAR_RIGHT are the two outer edges — never dragged.
    # GAP is the fixed gap kept on both sides of each boundary; only
    # boundary1/boundary2 themselves move. Initially, GRID and UMAP come
    # out the same width (0.382 each); SIDEBAR_WIDTH is ~30% narrower than
    # its original 0.18 (0.18 * 0.7 = 0.126) — the width it gave up is
    # split evenly between GRID and UMAP, which is why they're wider than
    # in the first version of this window.
    GRID_LEFT = 0.03
    SIDEBAR_RIGHT = 0.98
    GAP = 0.03
    boundary1 = 0.427
    # Sidebar ~20% narrower than its previous 0.126 (0.126 * 0.8 ≈ 0.101) —
    # boundary2 moved right by the same amount, so the reclaimed width
    # goes to the UMAP region right next to it rather than just shrinking
    # the window. SIDEBAR_RIGHT stays fixed, so SIDEBAR_WIDTH = SIDEBAR_
    # RIGHT - SIDEBAR_LEFT still lands at the intended ~0.101 below.
    boundary2 = 0.864
    GRID_RIGHT = boundary1 - GAP / 2
    UMAP_LEFT = boundary1 + GAP / 2
    UMAP_RIGHT = boundary2 - GAP / 2
    SIDEBAR_LEFT = boundary2 + GAP / 2
    SIDEBAR_WIDTH = SIDEBAR_RIGHT - SIDEBAR_LEFT
    # The status panel below (status_panel_ax etc., further down) is sized
    # in *inches*, not a flat figure-fraction like everything else here —
    # its own font size is fixed in points (a fixed physical/inches size),
    # so a flat-fraction box for it would grow/shrink in inches right along
    # with the window while the text inside stayed the same physical size:
    # excess empty space in a bigger window, text overflowing (into the
    # UMAP/grid area above it) in a smaller one. AREA_BOTTOM is recomputed
    # from these on every resize (see on_figure_resize below) so the plot
    # area above always starts exactly where the status panel's own
    # (also-recomputed) top edge actually ends up, in both directions.
    # Derived from SIDEBAR_FONTSIZE itself (not a flat guessed inches
    # value) — a flat guess is only ever right for one specific font size;
    # SIDEBAR_FONTSIZE varies with screen height (see UI_BUTTON_FONTSIZE),
    # so a box sized for a small-screen font was too short for a larger
    # one, which is what let the two text rows overlap each other and
    # overflow the panel's own bottom/top edges. 1.4x a font's own point
    # size (pt -> inches via /72) is a standard comfortable single-line
    # height; STATUS_PANEL_LINE_GAP_IN is the gap kept between the two
    # lines, separate from the top/bottom margins around both of them.
    STATUS_PANEL_LINE_HEIGHT_IN = SIDEBAR_FONTSIZE * 1.4 / 72
    STATUS_PANEL_LINE_GAP_IN = 0.05
    STATUS_PANEL_HEIGHT_IN = 2 * STATUS_PANEL_LINE_HEIGHT_IN + STATUS_PANEL_LINE_GAP_IN
    STATUS_PANEL_BOTTOM_MARGIN_IN = 0.06
    # Reserves room for the UMAP axes' own xlabel ("UMAP1"), which renders
    # *below* its bottom spine — i.e. into this exact margin, between the
    # axes and the status panel below it. This used to be a flat 0.08in
    # guess, exactly the "only right for one specific font size" trap the
    # comment above already warns about for its sibling constants here —
    # at SIDEBAR_FONTSIZE up to 16pt, the label alone (plus matplotlib's
    # own default 4pt axes.labelpad) needs upwards of 0.3in, so 0.08in
    # left it consistently clipped by the status panel drawn beneath it,
    # regardless of zoom level or anything else — the "few pixels of the
    # label peeking out from behind the status text" was this margin being
    # too small from the very start, not anything zoom-triggered. Same
    # 1.4x-fontsize comfortable-line-height convention as STATUS_PANEL_
    # LINE_HEIGHT_IN above.
    UMAP_XLABEL_PAD_PT = 4.0  # matplotlib's own rcParams['axes.labelpad'] default
    STATUS_PANEL_TOP_MARGIN_IN = (UMAP_XLABEL_PAD_PT + SIDEBAR_FONTSIZE * 1.4) / 72
    # Fixed ratios of the panel's own height (not the figure's) — by
    # construction (STATUS_PANEL_HEIGHT_IN's own definition above),
    # 2*LINE_FRAC + GAP_FRAC == 1.0 exactly, so the two rows plus the gap
    # between them exactly fill the panel with no leftover slack, whatever
    # the actual current inches-to-fraction conversion happens to be.
    STATUS_PANEL_LINE_FRAC = STATUS_PANEL_LINE_HEIGHT_IN / STATUS_PANEL_HEIGHT_IN
    STATUS_PANEL_GAP_FRAC = STATUS_PANEL_LINE_GAP_IN / STATUS_PANEL_HEIGHT_IN

    def compute_area_bottom():
        fig_h_in = fig.get_size_inches()[1]
        return (STATUS_PANEL_BOTTOM_MARGIN_IN + STATUS_PANEL_HEIGHT_IN + STATUS_PANEL_TOP_MARGIN_IN) / fig_h_in

    def status_panel_geometry():
        fig_h_in = fig.get_size_inches()[1]
        return STATUS_PANEL_BOTTOM_MARGIN_IN / fig_h_in, STATUS_PANEL_HEIGHT_IN / fig_h_in

    AREA_BOTTOM = compute_area_bottom()
    AREA_TOP = 0.96

    # A fixed-width strip reserved for the colorbar *and* (see draw_id_
    # legend, defined alongside clear_colorbar further down) the multi-ID
    # legend for 'Specified <level>(s)' mode — the two never appear at once,
    # so they share this one axes rather than each needing their own.
    # Permanently carved out of the UMAP area's own width rather than
    # borrowed from it on demand (see redraw_gene's cax= colorbar below) —
    # the earlier ax=ax approach (fig.colorbar(scatter, ax=ax)) resized ax
    # itself to make room, so the plot visibly changed size switching
    # between Gene and Subclass views; an axes that's simply hidden/shown,
    # never created/destroyed or borrowing space from ax, keeps ax exactly
    # the same size regardless of which mode (or how many legend rows) is
    # currently using this strip.
    #
    # Wide enough for a swatch + "1234 (5678)" ID-and-count text at
    # SIDEBAR_FONTSIZE, not just a colorbar's own tick numbers — the
    # legend's actual requirement, sized generously since the tradeoff is
    # against the UMAP plot's own width, not against much else.
    CBAR_WIDTH = 0.026
    CBAR_GAP = 0.006
    # Room reserved to the right of the bar/legend strip itself, before
    # UMAP_RIGHT (and the draggable boundary2 resize handle right past it)
    # — a plain vertical colorbar's own label renders further right than
    # the bar, and with cbar_ax's right edge sitting flush against
    # UMAP_RIGHT there was nowhere for a longer gene-name label to go
    # except under the resize handle itself.
    CBAR_LABEL_MARGIN = 0.011

    # Widens the whole figure (all fractional positions above stay valid —
    # only the absolute inches they scale to changes) so ax's own
    # allocated box — the UMAP scatter's plotting area, after CBAR_WIDTH/
    # CBAR_GAP/CBAR_LABEL_MARGIN are carved out of its width — comes out
    # exactly square in real inches, not just in figure-fraction terms.
    # Those three eat into ax's *width* without touching its *height*, so
    # without this correction the plotting area came out slightly narrower
    # than tall. AREA_BOTTOM already reflects the figure's real (screen-
    # derived) height at this point, so this reproduces the same
    # correction on every screen rather than a single fixed ratio that
    # would only happen to be right for the screen it was tuned on.
    ax_width_fraction = (UMAP_RIGHT - UMAP_LEFT) - CBAR_WIDTH - CBAR_GAP - CBAR_LABEL_MARGIN
    ax_height_fraction = AREA_TOP - AREA_BOTTOM
    _fig_w_in, _fig_h_in = fig.get_size_inches()
    _squared_fig_w_in = _fig_h_in * (ax_height_fraction / ax_width_fraction)
    fig.set_size_inches(_squared_fig_w_in, _fig_h_in, forward=True)
    # forward=True (above) asks Tk to resize the window too, but the same
    # residual-DPI-scaling snap-back the early fig.canvas.manager.show()/
    # assert_figure_content_size() call (right after fig's own creation,
    # above) was already guarding against can happen again on *this*
    # resize — reasserted here for the same reason, now at the corrected
    # size (assert_figure_content_size is already best-effort/self-
    # guarded, see its own docstring).
    assert_figure_content_size(fig, _squared_fig_w_in, _fig_h_in)

    ax = fig.add_axes([UMAP_LEFT, AREA_BOTTOM,
                        (UMAP_RIGHT - UMAP_LEFT) - CBAR_WIDTH - CBAR_GAP - CBAR_LABEL_MARGIN,
                        AREA_TOP - AREA_BOTTOM])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('UMAP1', fontsize=SIDEBAR_FONTSIZE)
    ax.set_ylabel('UMAP2', fontsize=SIDEBAR_FONTSIZE)
    # 'box' (not the default 'datalim') re-letterboxes the *box* itself to
    # stay 1:1 whenever ax's own allocated box changes shape (e.g. dragging
    # the resize handles below) — ax.clear() resets this to matplotlib's
    # default ('auto', no enforced aspect) on every mode/redraw, so each
    # redraw_* function below re-applies it right after its own clear().
    ax.set_aspect('equal', adjustable='box')

    cbar_ax = fig.add_axes([UMAP_RIGHT - CBAR_WIDTH - CBAR_LABEL_MARGIN, AREA_BOTTOM, CBAR_WIDTH,
                             AREA_TOP - AREA_BOTTOM])
    cbar_ax.set_visible(False)

    # Every zoomable/pannable axes (the UMAP scatter, plus every section
    # panel below) registers itself here with its own "home" (fully-
    # zoomed-out) extent — on_scroll_zoom uses this per-axes to clamp how
    # far out that *specific* axes can zoom, and it's what makes pan/zoom
    # independent per section panel (each just has its own entry) while
    # still letting every section start at the same shared physical scale
    # (see the section-grid loop below, which is what actually keeps that
    # scale in sync across panels — this dict itself has no notion of
    # "sync", it just remembers one axes' own limits).
    pannable_axes = {}
    umap_x_pad = ((coords[:, 0].max() - coords[:, 0].min()) * 0.05) or 1.0
    umap_y_pad = ((coords[:, 1].max() - coords[:, 1].min()) * 0.05) or 1.0
    pannable_axes[ax] = {
        'home_xlim': (coords[:, 0].min() - umap_x_pad, coords[:, 0].max() + umap_x_pad),
        'home_ylim': (coords[:, 1].min() - umap_y_pad, coords[:, 1].max() + umap_y_pad),
    }

    # The UMAP scatter's own dot size grows on zoom-in (rate set by
    # UMAP_ZOOM_DOT_GROWTH_RATE, top of file) — the artist is tracked here
    # (each redraw_* function below sets this to whatever it just plotted)
    # so on_scroll_zoom can rescale it live via set_sizes() without needing
    # a full redraw. 'artists' holds *every* scatter call that made up the
    # current view — a categorical mode with UMAP_USE_SHAPES_ON_SCREEN on
    # draws one call per (shape, filled/open) combination in use, not one
    # — 'artist' is kept too, as artists[0], for the many callers (hover/
    # export/etc.) that only need *a* representative artist (e.g. for its
    # shared size/alpha), not every one of them.
    main_scatter_state = {'artist': None, 'artists': [], 'size_multipliers': []}
    # A short, filename-safe token describing whatever the UMAP is currently
    # showing — the gene name(s) in Gene mode, the taxonomy ID(s) in
    # 'Single <level>' mode. Each redraw_* sets it; 'Save UMAP' turns it into
    # the saved file's name, so what's on screen and what lands on disk stay
    # described the same way.
    current_view_name = {'token': 'view'}

    def sanitized_view_token():
        """current_view_name['token'], made filename-safe — shared by every
        saved-file naming site (save_current_umap, save_section_maps) so
        they describe the same on-screen state the same way. Only the
        characters safe in a filename on every platform; gene symbols and
        ID lists are already tame, but a stray '/' or ':' from a category
        name would otherwise fail the write outright."""
        return re.sub(r'[^A-Za-z0-9._-]+', '_', str(current_view_name['token'])).strip('_') or 'view'
    # Derived from UMAP_POINT_SIZE (top of file) rather than set here, so it
    # keeps its proportion to the cells automatically when that's tuned.
    # The extra *2.25 is a 50% bump to the marker's *diameter*, not its area
    # — matplotlib's scatter `s` is area, so diameter scales as sqrt(s); to
    # make the highlight 1.5x wider, the area needs to grow by 1.5**2.
    UMAP_HIGHLIGHT_BASE_SIZE = UMAP_POINT_SIZE * 4 * 2.25

    def set_main_scatter_artists(artists, size_multipliers=None):
        """`size_multipliers`, if given, is one AREA multiplier per artist
        (applied to UMAP_POINT_SIZE before the zoom multiplier itself — see
        update_umap_dot_size) — for the rare case (currently just the
        multi-gene '+' overlay layer, drawn deliberately larger so its arms
        extend past the circle layer beneath it) where every artist on this
        axes shouldn't be the same size. Defaults to 1.0 for every artist,
        matching every other mode's single uniformly-sized scatter."""
        main_scatter_state['artists'] = artists
        main_scatter_state['artist'] = artists[0] if artists else None
        main_scatter_state['size_multipliers'] = (
            list(size_multipliers) if size_multipliers is not None else [1.0] * len(artists)
        )
        # A snapshot of each artist's own data, taken immediately after
        # redraw_* just built it from the *complete* point set (this is
        # always called right after a fresh ax.scatter(), never after any
        # filtering) — not a live reference, since filter_main_scatter_to_
        # viewport mutates the artist's own offsets/colors in place below,
        # and repeatedly re-filtering *that* instead of this untouched
        # snapshot would compound/lose points across zoom ticks. Read back
        # generically from whatever the artist actually ended up holding,
        # rather than needing its own copy of each mode's own color logic
        # (single gene, multi-gene RGB blend, categorical with or without
        # shapes all end up here the same way).
        main_scatter_state['full'] = [
            {
                'offsets': np.asarray(a.get_offsets()),
                'array': a.get_array(),  # None unless this is a continuous (Gene-mode) scatter
                'facecolors': np.array(a.get_facecolors(), copy=True),
                'edgecolors': np.array(a.get_edgecolors(), copy=True),
            }
            for a in artists
        ]

    def filter_main_scatter_to_viewport():
        """Replaces each current UMAP scatter artist's own data with just
        the subset of main_scatter_state['full'] (the true, complete point
        set snapshotted once in set_main_scatter_artists) that falls
        within ax's *current* view — so a real draw at high zoom only ever
        asks matplotlib to transform/clip the handful of points actually
        on screen, not the full ~200k. Always re-filters from that
        untouched full snapshot, never from whatever the artist currently
        holds, so zooming in/out past ZOOM_BITMAP_ONLY_MAX_MULTIPLIER
        repeatedly never compounds or loses points. Called by end_zoom_
        previews right before its own real fig.canvas.draw(); a uniform
        (single-color, e.g. the gray "rest" scatter) collection's face/
        edgecolors are left alone rather than indexed, since matplotlib
        already broadcasts a single row across however many offsets are
        set — trying to index it would raise."""
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xlo, xhi = min(x0, x1), max(x0, x1)
        ylo, yhi = min(y0, y1), max(y0, y1)
        total_full, total_shown = 0, 0
        for artist, full in zip(main_scatter_state['artists'], main_scatter_state['full']):
            offsets = full['offsets']
            if len(offsets) == 0:
                continue
            in_view = ((offsets[:, 0] >= xlo) & (offsets[:, 0] <= xhi)
                       & (offsets[:, 1] >= ylo) & (offsets[:, 1] <= yhi))
            artist.set_offsets(offsets[in_view])
            if full['array'] is not None:
                artist.set_array(full['array'][in_view])
            else:
                if len(full['facecolors']) == len(offsets):
                    artist.set_facecolors(full['facecolors'][in_view])
                if len(full['edgecolors']) == len(offsets):
                    artist.set_edgecolors(full['edgecolors'][in_view])
            total_full += len(offsets)
            total_shown += int(in_view.sum())
        if ZOOM_DEBUG_DIAGNOSTICS:
            print(f"[zoom-filter] umap view=({xlo:.3f},{xhi:.3f})x({ylo:.3f},{yhi:.3f}) "
                  f"ratio={umap_zoom_ratio():.2f}x  cells: {total_full} -> {total_shown} "
                  f"across {len(main_scatter_state['artists'])} artist(s)")

    def filter_all_section_scatters_to_viewport():
        """Section-panel equivalent of filter_main_scatter_to_viewport:
        for every panel, replaces its single background_artist's data with
        just the subset of panel['full_offsets']/['full_colors'] (the
        complete, unfiltered set snapshotted in apply_panel_colors_with_
        gray_behind — the one choke point every color assignment already
        goes through) that falls within *that panel's own* current view.
        Each panel has its own xlim/ylim (they don't pan in lockstep, only
        zoom by the same shared ratio — see apply_section_zoom_ratio), so
        this can't reuse one shared view rectangle the way the UMAP's own
        single axes can."""
        total_full, total_shown = 0, 0
        for panel in section_panels.values():
            full_offsets = panel['full_offsets']
            if full_offsets is None or len(full_offsets) == 0:
                continue
            # A panel currently showing its cached home-view image (see
            # show_section_home_cache_or_scatter) has its real background_
            # artist hidden — switch back to it now, since we're about to
            # filter/draw the real, zoomed-in content and the cached image
            # (a fixed snapshot of the *home* extent) has nothing useful to
            # show once zoomed past it.
            if panel['cached_home_image'] is not None and panel['cached_home_image'].get_visible():
                panel['cached_home_image'].set_visible(False)
                panel['background_artist'].set_visible(True)
            x0, x1 = panel['ax'].get_xlim()
            y0, y1 = panel['ax'].get_ylim()
            xlo, xhi = min(x0, x1), max(x0, x1)
            ylo, yhi = min(y0, y1), max(y0, y1)
            in_view = ((full_offsets[:, 0] >= xlo) & (full_offsets[:, 0] <= xhi)
                       & (full_offsets[:, 1] >= ylo) & (full_offsets[:, 1] <= yhi))
            panel['background_artist'].set_offsets(full_offsets[in_view])
            panel['background_artist'].set_facecolor(panel['full_colors'][in_view])
            total_full += len(full_offsets)
            total_shown += int(in_view.sum())
        if ZOOM_DEBUG_DIAGNOSTICS:
            print(f"[zoom-filter] sections ratio={section_zoom_ratio['value']:.2f}x  "
                  f"cells: {total_full} -> {total_shown} across {len(section_panels)} panel(s)")

    def category_of_point_from_mapping(per_cell_keys, color_by_key):
        """(category_of_point, rank_color): category_of_point gives every
        cell (in coords' own row order — per_cell_keys must be too) the
        rank of its own category within color_by_key's own iteration
        order (rank 0 = color_by_key's first entry), or -1 if its key
        isn't in color_by_key at all (e.g. a category past UMAP_MAX_
        COLORED_CATEGORIES, or a typed ID that matched no cells).
        rank_color maps each rank back to its color. Shared by redraw_all_
        subclasses, redraw_subclass, and render_export_figure's own
        categorical branch (each passing in whatever mapping — umap_
        color_map or target_colors — it already had to build anyway), so
        all three assign ranks/colors identically."""
        category_of_point = np.full(len(per_cell_keys), -1)
        rank_color = {}
        for rank, (key, color) in enumerate(color_by_key.items()):
            category_of_point[per_cell_keys == key] = rank
            rank_color[rank] = color
        return category_of_point, rank_color

    def plot_categorical_umap(target_ax, category_of_point, rank_color, point_size, point_alpha, use_shapes):
        """Draws a categorical UMAP view onto target_ax: uncategorized
        cells (category_of_point < 0) in plain gray first/underneath,
        then colored cells on top — same "meaningful color on top"
        convention as apply_panel_colors_with_gray_behind. use_shapes=
        False (a single mode/level/query redraw's usual on-screen case
        unless UMAP_USE_SHAPES_ON_SCREEN is on) draws every colored cell
        in one scatter() call; True (always, for the exported file — see
        render_export_figure) draws one call per (marker shape, filled/
        open) combination actually in use instead, since scatter() only
        ever takes one marker per call. Returns every scatter artist
        created, in draw order (for main_scatter_state — see set_main_
        scatter_artists)."""
        artists = []
        gray_mask = category_of_point < 0
        if gray_mask.any():
            artists.append(target_ax.scatter(coords[gray_mask, 0], coords[gray_mask, 1], c='lightgray',
                                              s=point_size, alpha=point_alpha, linewidths=0))
        if rank_color:
            if use_shapes:
                shape_groups = {}
                for rank in rank_color:
                    shape_groups.setdefault(category_rank_shape(rank), []).append(rank)
                for (marker, is_open), ranks_in_group in shape_groups.items():
                    group_mask = np.isin(category_of_point, ranks_in_group)
                    group_colors = [rank_color[r] for r in category_of_point[group_mask]]
                    if is_open:
                        artists.append(target_ax.scatter(
                            coords[group_mask, 0], coords[group_mask, 1], marker=marker,
                            facecolors='none', edgecolors=group_colors, linewidths=0.8,
                            s=point_size, alpha=point_alpha))
                    else:
                        artists.append(target_ax.scatter(
                            coords[group_mask, 0], coords[group_mask, 1], marker=marker,
                            c=group_colors, s=point_size, alpha=point_alpha, linewidths=0))
            else:
                colored_mask = ~gray_mask
                point_colors = [rank_color[r] for r in category_of_point[colored_mask]]
                artists.append(target_ax.scatter(
                    coords[colored_mask, 0], coords[colored_mask, 1],
                    c=point_colors, s=point_size, alpha=point_alpha, linewidths=0))
        return artists

    def umap_zoom_ratio():
        """home_span / current_span — 1.0 at fully zoomed out, growing as
        the view narrows. Shared by umap_zoom_diameter_multiplier (below,
        the dot-growth curve) and end_zoom_previews' own ZOOM_BITMAP_ONLY_
        MAX_MULTIPLIER check, so both ever compute this the same way."""
        home = pannable_axes[ax]
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        cur_span = max(abs(x1 - x0), abs(y1 - y0))
        home_span = max(abs(home['home_xlim'][1] - home['home_xlim'][0]),
                         abs(home['home_ylim'][1] - home['home_ylim'][0]))
        return (home_span / cur_span) if cur_span > 0 else 1.0

    def umap_zoom_diameter_multiplier():
        return 1.0 + UMAP_ZOOM_DOT_GROWTH_RATE * max(0.0, umap_zoom_ratio() - 1.0)

    def umap_group_marker_size(multiplier):
        # matplotlib's `s` is area — squaring the diameter ratio (on top of
        # the already-squared zoom multiplier) is what actually makes the
        # marker 1.25x the *diameter*, not 1.25x the area, of the main dot.
        return UMAP_POINT_SIZE * multiplier ** 2 * UMAP_GROUP_DIAMETER_RATIO ** 2

    # The centroid ID labels currently on the UMAP axes ('All <level>s' mode
    # only — the other modes don't draw any), kept so update_umap_dot_size
    # can rescale their font on zoom. Reset on every ax.clear(), which
    # removes them along with everything else.
    umap_label_state = {'artists': []}

    def umap_label_fontsize(raw_scale, multiplier):
        """Font size for a centroid ID label, given its own raw (not yet
        clamped) UMAP_LABEL_SIZE_BY_CELL_COUNT scale and the current zoom
        multiplier — the one place both add_umap_centroid_labels (creation,
        always at multiplier=1.0 since redraw_* always resets to home
        extent first) and update_umap_dot_size (every zoom tick thereafter,
        at whatever multiplier is current) get a label's fontsize, so the
        clamping logic itself lives in exactly one place.

        The min-scale floor shrinks as 1/multiplier while zooming in from
        1x to 5x, then holds flat from 5x on. For a label already pinned to
        that floor (raw_scale below it at every zoom level), the two
        factors of `multiplier` below exactly cancel while it's shrinking
        (min_scale * multiplier == UMAP_LABEL_SIZE_BY_CELL_COUNT_MIN_SCALE,
        constant) — so its on-screen size holds flat through that whole 1x-
        5x range rather than growing with the dots the way an unclamped
        label does, which is what actually fixes the "small clusters'
        labels overrun everything at high zoom" problem this feature exists
        for. Past 5x the floor stops shrinking (fixed at MIN_SCALE/5), so
        that cancellation stops too and the label resumes growing with zoom
        like any unclamped one, just from a smaller starting point. Smooth
        (no jump) at both the 1x and 5x boundaries — verified numerically,
        not just by inspection, before this doc was written."""
        min_scale = UMAP_LABEL_SIZE_BY_CELL_COUNT_MIN_SCALE
        if multiplier > 1:
            min_scale /= min(5.0, multiplier)
        scale = float(np.clip(raw_scale, min_scale, UMAP_LABEL_SIZE_BY_CELL_COUNT_MAX_SCALE))
        return UMAP_LABEL_FONTSIZE * scale * multiplier

    def update_umap_dot_size():
        # Applies to every dot-like artist on the UMAP axes — the main
        # scatter, the single-cell ring, and the same-subclass family
        # highlight — all scaled by the same multiplier, so the highlight
        # artists stay proportionate to the underlying points as you zoom,
        # not left behind at their fully-zoomed-out size.
        multiplier = umap_zoom_diameter_multiplier()
        for artist, size_mult in zip(main_scatter_state['artists'], main_scatter_state['size_multipliers']):
            artist.set_sizes([UMAP_POINT_SIZE * size_mult * multiplier ** 2])
        if umap_highlight_state['artist'] is not None:
            umap_highlight_state['artist'].set_sizes([UMAP_HIGHLIGHT_BASE_SIZE * multiplier ** 2])
        if group_highlight_state['artist'] is not None:
            group_highlight_state['artist'].set_sizes([umap_group_marker_size(multiplier)])
        # Centroid ID labels track the dots exactly. `multiplier` is a
        # *diameter* ratio (the dot sizes above square it, since matplotlib's
        # `s` is an area), and font size is likewise a linear dimension — so
        # it takes the multiplier as-is, and the labels stay in constant
        # proportion to the dots at every zoom level rather than being the
        # one thing on the axes that doesn't move with the rest.
        #
        # Each label's own raw (unclamped) count-based scale (_umap_raw_
        # scale, stamped on it when created — see add_umap_centroid_labels)
        # is what actually varies per category when UMAP_LABEL_SIZE_BY_
        # CELL_COUNT is on; re-clamped through umap_label_fontsize on every
        # tick (not just once at creation) since its own min-scale floor
        # depends on the *current* multiplier. A missing attribute
        # (shouldn't happen, but cheap to guard) falls back to a flat
        # unscaled label rather than raising mid-zoom.
        for label in umap_label_state['artists']:
            raw_scale = getattr(label, '_umap_raw_scale', 1.0)
            label.set_fontsize(umap_label_fontsize(raw_scale, multiplier))

    def cell_weighted_median_count(counts):
        """The cell count of whichever category the *middle cell* falls
        into, if every cell were laid out in one line ordered by category
        size, largest category first — as opposed to plain median(), which
        is the middle of the category *sizes themselves*, treating a
        10-cell category and a 10,000-cell category as equally "one data
        point" toward the middle.

        A handful of oversized categories can hold a disproportionate share
        of all cells, which pulls this well above the plain median (see
        UMAP_LABEL_SIZE_BY_CELL_COUNT's own comment for why that mattered):
        anchoring the label-size scale here instead means "same size as
        default" lines up with where most *cells* actually are, at the cost
        of compressing more of the smaller (by definition, less cell-heavy,
        hence less important) categories together at the size floor."""
        ranked = counts.sort_values(ascending=False)
        half = ranked.sum() / 2
        return float(ranked.iloc[(ranked.cumsum() >= half).argmax()])

    def add_umap_centroid_labels(level, palette_info):
        """Centroid ID labels for 'All <level>s' mode — the interactive
        viewer's own version of the module-level add_category_id_labels, not
        a call to it: that function labels every category using
        adata.uns[f'{color_key}_colors'], the same all-or-nothing scanpy
        assignment compute_ranked_category_colors exists to replace (see its
        own docstring), and has no notion of the optional per-category font
        scaling below. Kept separate so Step 7's static plot and
        save_umap_by_group — the module-level function's other two callers
        — are completely unaffected by either change.

        Only labels the categories in palette_info['umap_color_map'] (the
        top UMAP_MAX_COLORED_CATEGORIES by cell count) — the same ones
        actually colored on the UMAP; a label pointing at a gray,
        color-unidentified dot would have nothing to anchor it to."""
        color_map = palette_info['umap_color_map']
        if not color_map:
            return []
        counts = palette_info['counts']
        obs_values = adata.obs[level].to_numpy()
        if UMAP_LABEL_SIZE_BY_CELL_COUNT:
            median_count = cell_weighted_median_count(counts.loc[list(color_map)])
        labels = []
        for category, color in color_map.items():
            mask = obs_values == category
            if not mask.any():  # shouldn't happen — color_map only has categories with count > 0
                continue
            centroid = coords[mask].mean(axis=0)
            match = re.match(r'^\s*(\d+)', str(category))
            label_text = match.group(1) if match else str(category)
            # Unclamped here — umap_label_fontsize (shared with update_umap_
            # dot_size's own per-tick rescaling) does the clamping, since its
            # min-scale floor depends on the current zoom multiplier, which
            # is always 1.0 at creation time (redraw_* always resets to home
            # extent first) but not necessarily whenever this label is next
            # rescaled.
            raw_scale = np.power(counts[category] / median_count, 0.25) if UMAP_LABEL_SIZE_BY_CELL_COUNT else 1.0
            text = ax.text(
                centroid[0], centroid[1], label_text, fontsize=umap_label_fontsize(raw_scale, 1.0),
                color=darken_color(color, 0.55), fontweight='bold', ha='center', va='center', clip_on=True,
            )
            # Remembered so update_umap_dot_size can re-derive THIS
            # category's fontsize at any zoom level, not just the one it had
            # when created.
            text._umap_raw_scale = raw_scale
            labels.append(text)
        return labels

    # Highlight ring for whichever UMAP cell is currently hovered — recreated
    # (not just hidden) after every ax.clear() in the redraw_* functions
    # below, since clear() removes every artist including this one; kept in
    # a mutable container so show_highlight_for_cell/hide_all_highlights
    # (defined once, used across every redraw) always reach the current one.
    umap_highlight_state = {'artist': None}

    def recreate_umap_highlight():
        artist = ax.scatter([], [], s=UMAP_HIGHLIGHT_BASE_SIZE * umap_zoom_diameter_multiplier() ** 2,
                             facecolor='none', edgecolor='red', linewidths=1.5, zorder=6)
        artist.set_visible(False)
        # Animated: updated on nearly every hover-settle via blit_hover_
        # overlays() (a cheap restore-and-stamp of the cached background),
        # not a full fig.canvas.draw_idle() — with dozens of section panels
        # each carrying a per-point-colored, several-thousand-point
        # scatter (see update_section_dot_size/set_section_colors_*), a
        # full redraw on every hover tick was the actual source of the
        # sluggishness, not anything about hovering itself.
        artist.set_animated(True)
        umap_highlight_state['artist'] = artist
        # Veil over the UMAP while a hover highlight is engaged, the UMAP's
        # counterpart to the section panels' dim_veil (see its comment there
        # for why it's an animated overlay). Recreated here for the same
        # reason as the ring: ax.clear() removes it. Its color is picked when
        # shown (see set_section_dimming), since the background changes with
        # the mode. zorder is below the '+' markers (5.5) and the ring (6).
        veil = Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes,
            facecolor='white', edgecolor='none', alpha=UMAP_VEIL_ALPHA, zorder=5.2,
        )
        veil.set_visible(False)
        veil.set_animated(True)
        ax.add_patch(veil)
        umap_highlight_state['veil'] = veil

    # How strongly the UMAP veil washes out the dots toward the background.
    UMAP_VEIL_ALPHA = 0.5
    recreate_umap_highlight()

    # Same-subclass "family" highlight — every other cell sharing the
    # hovered cell's subclass, filled light red (not the hollow, solid-red
    # ring above, which stays reserved for the one actual cell under the
    # cursor) — shown only after a longer hold (GROUP_HOVER_HOLD_MS, well
    # past VIEWER_HOVER_HOLD_MS's single-cell lookup) so briefly passing over a
    # cell doesn't immediately light up its whole subclass. zorder sits
    # between the base scatter and the single-cell ring, so the ring still
    # reads as the one exact cell even against its own now-highlighted
    # family. Recreated alongside umap_highlight_state after every ax.
    # clear() in the redraw_* functions below, for the same reason.
    group_highlight_state = {'artist': None}

    def recreate_group_highlight():
        # Just resets the reference — unlike umap_highlight (a single
        # fixed-size point that only ever moves), the group highlight is a
        # variable number of points that changes with every hovered cell,
        # so it's simplest (and, empirically, actually reliable — see
        # show_group_highlight_for_cell's own comment) to create it fresh
        # with real data each time rather than pre-allocate an empty one
        # and grow it via set_offsets.
        group_highlight_state['artist'] = None

    recreate_group_highlight()

    # --- Brain-section grid (left) ------------------------------------
    # One real Axes per section (not a single flattened bitmap the way the
    # section-thumbnail grid picker does it) — sharing one physical scale
    # (SPAN_PERCENTILE below, same convention as generate_section_grid_
    # image's own) so an anatomically small section isn't zoomed to look
    # as large as a full coronal one; only each panel's own center
    # (pan position) differs — zoom *level*, unlike pan, is kept in sync
    # across every panel (see section_zoom_ratio/apply_section_zoom_ratio
    # further down): scrolling on any one panel zooms all of them by the
    # same amount, each around its own current center.
    # Both derived from UMAP_POINT_SIZE (top of file), so they keep their
    # proportion to the cells automatically when that's tuned;
    # SECTION_BACKGROUND_BASE_SIZE is up there too.
    # *2.25 here too — see UMAP_HIGHLIGHT_BASE_SIZE's own comment on why
    # that's a 50%-diameter (not 50%-area) increase.
    SECTION_HIGHLIGHT_BASE_SIZE = UMAP_POINT_SIZE * 3 * 2.25
    SECTION_GROUP_BASE_SIZE = UMAP_POINT_SIZE * 2.5
    section_panels = {}
    # Set inside the panel-build loop below, on whichever panel is actually
    # built first (see its own comment) — None if there turn out to be no
    # panels at all (e.g. no spatial data for this run).
    section_scalebar = None
    if sections_present and has_spatial:
        n_sections = len(sections_present)
        fig_width_in, fig_height_in = fig.get_size_inches()
        region_w_in = (GRID_RIGHT - GRID_LEFT) * fig_width_in
        region_h_in = (AREA_TOP - AREA_BOTTOM) * fig_height_in
        # One shared value for both row and column spacing — halved from
        # its previous 0.012, since the gap between panels (most
        # noticeable scanning across a row of columns) felt too wide.
        panel_gap = 0.006
        gap_w_in = panel_gap * fig_width_in
        gap_h_in = panel_gap * fig_height_in

        # Background (all cells of the section, for context) uses
        # background_x/y/section when available (see their own comment
        # above) — falls back to this run's own cells (cell_x/y/section)
        # if adata_backed wasn't given, same as before this was added.
        bg_x = background_x if has_background else cell_x
        bg_y = background_y if has_background else cell_y
        bg_section = background_section_arr if has_background else cell_section
        # Cell IDs aligned 1:1 with bg_x/bg_y/bg_section — the join key for
        # hovering a section panel's own point back to a UMAP index (via
        # cell_id_to_umap_idx above), same reasoning as that dict's own
        # comment: background_spatial isn't necessarily adata's own cells
        # in the same order (or even the same set).
        bg_ids = background_spatial.index.to_numpy() if has_background else adata.obs_names.to_numpy()

        # Computed before grid_ncols/grid_nrows below (moved up from where
        # this used to sit, after that calc) — the layout search needs
        # shared_half_w/shared_half_h's own ratio (a section's actual
        # spatial aspect ratio, typically wider than tall) to pick a grid
        # shape that matches it, not just the region's own aspect ratio.
        centroids, half_widths, half_heights = {}, [], []
        # sec_mask (bg_section == sec) is a full pass over the whole
        # (multi-million-row) background array — the single costliest step
        # per section here and, again, in the panel-build loop below, which
        # used to redo the identical comparison for the same 59 sections.
        # Cached per section (mask + the resulting valid/xs/ys) so the panel
        # loop can reuse it instead of recomputing.
        sec_filtered = {}
        for sec in sections_present:
            sec_mask = bg_section == sec
            xs_sec = bg_x[sec_mask]
            ys_sec = bg_y[sec_mask]
            valid = ~(np.isnan(xs_sec) | np.isnan(ys_sec))
            xs_sec, ys_sec = xs_sec[valid], ys_sec[valid]
            if len(xs_sec) == 0:
                continue
            sec_filtered[sec] = (sec_mask, valid, xs_sec, ys_sec)
            centroids[sec] = ((xs_sec.min() + xs_sec.max()) / 2, (ys_sec.min() + ys_sec.max()) / 2)
            half_widths.append((xs_sec.max() - xs_sec.min()) / 2)
            half_heights.append((ys_sec.max() - ys_sec.min()) / 2)
        # VIEWER_SPAN_PERCENTILE (top of file) — prefixed because the two
        # picker windows have their own SPAN_PERCENTILE, set to a different
        # value, that this must not be confused with.
        shared_half_w = (np.percentile(half_widths, VIEWER_SPAN_PERCENTILE) * 1.05) if half_widths else 1.0
        shared_half_h = (np.percentile(half_heights, VIEWER_SPAN_PERCENTILE) * 1.05) if half_heights else 1.0

        # Grid shape chosen to maximize each panel's own rendered content
        # size, not to make each panel's *box* square. Those aren't the
        # same thing: every panel plots at 'equal'/'box' aspect (see
        # sec_ax.set_aspect below), so if a section's own content is wider
        # than tall (the ABC atlas' coronal sections typically are, ~3:2)
        # but its box is square, matplotlib letterboxes the content inside
        # that box — shrinking it to fit — rather than filling the box,
        # which is exactly the "lots of white space between panels" this
        # was reported as. The previous approach picked ncols from sqrt(n *
        # region_aspect), which only ever aimed for square boxes.
        #
        # Instead, this searches every ncols from 1..n_sections (nrows is
        # always ceil(n_sections / ncols) — for a fixed ncols, more rows
        # than that only shrinks panels further for no benefit, so it's
        # never worth considering separately) and scores each candidate
        # grid by the resulting panel's rendered content height once
        # letterboxed to content_aspect — content_w and content_h scale
        # together at a fixed aspect, so maximizing either one maximizes
        # the actual rendered area. n_sections is at most a few hundred, so
        # this brute-force search (versus solving the two-variable
        # optimization directly) is negligible next to actually building
        # that many panels.
        content_aspect = shared_half_w / shared_half_h if shared_half_h > 0 else 1.0
        best_ncols, best_nrows, best_content_h = 1, n_sections, -1.0
        for ncols in range(1, n_sections + 1):
            nrows = math.ceil(n_sections / ncols)
            cand_w_in = (region_w_in - (ncols - 1) * gap_w_in) / ncols
            cand_h_in = (region_h_in - (nrows - 1) * gap_h_in) / nrows
            if cand_w_in <= 0 or cand_h_in <= 0:
                continue
            content_h_in = min(cand_h_in, cand_w_in / content_aspect)
            if content_h_in > best_content_h:
                best_ncols, best_nrows, best_content_h = ncols, nrows, content_h_in
        grid_ncols, grid_nrows = best_ncols, best_nrows
        panel_w = (GRID_RIGHT - GRID_LEFT - (grid_ncols - 1) * panel_gap) / grid_ncols
        panel_h = (AREA_TOP - AREA_BOTTOM - (grid_nrows - 1) * panel_gap) / grid_nrows

        log_status(f"Step 9: Building {n_sections} section panel(s)...")
        _panel_timing = {'axes': 0.0, 'mask': 0.0, 'umap_idx': 0.0, 'levels': 0.0, 'scatter': 0.0, 'other': 0.0}
        # Printed every 10th panel (not every one — n_sections can be large
        # enough that per-panel prints would themselves add clutter/latency)
        # purely so the console shows *some* progress during what's often
        # the single slowest stretch of opening this window (this is where
        # each section's own scatter, previously masked/loaded, actually
        # gets rendered for the first time) — the same "show something is
        # happening" reasoning as the sidebar's own Working... indicator,
        # just for the console during setup, before the window exists yet.
        # (Interval set by PANEL_PROGRESS_INTERVAL, top of file.) The window
        # is already on screen by this point (created earlier, above) but
        # has never been drawn to — without an explicit draw here, it just
        # sits blank for the whole build loop and then the entire grid
        # (still gray placeholders — real per-cell colors land later, in
        # redraw_all_subclasses) pops in all at once at the first real
        # fig.canvas.draw(). A synchronous draw+flush every Nth panel instead
        # shows the grid filling in as it's actually built, same "something
        # is happening" reasoning as the console progress line right below.
        for i, sec in enumerate(sections_present):
            if sec not in centroids:
                continue
            if i % PANEL_PROGRESS_INTERVAL == 0:
                log_status(f"Step 9: Building section panel {i + 1}/{n_sections} ({sanitize_section_token(sec)})...")
                fig.canvas.draw()
                fig.canvas.flush_events()
            _t0 = time.perf_counter()
            row, col = divmod(i, grid_ncols)
            x0 = GRID_LEFT + col * (panel_w + panel_gap)
            y0 = AREA_TOP - (row + 1) * panel_h - row * panel_gap
            sec_ax = fig.add_axes([x0, y0, panel_w, panel_h])
            sec_ax.set_xticks([])
            sec_ax.set_yticks([])
            sec_ax.set_facecolor(SECTION_PANEL_FACECOLOR)
            # Belt-and-suspenders alongside the aspect-aware ncols/nrows
            # above: even if a panel's box isn't perfectly square (rounding,
            # or a very uneven section count), 'equal' keeps a data-unit
            # circle looking circular instead of stretched to fill whatever
            # box shape it got — 'box' (not the default 'datalim') shrinks
            # the *box* to match, rather than changing the data limits.
            sec_ax.set_aspect('equal', adjustable='box')
            for spine in sec_ax.spines.values():
                spine.set_linewidth(0.5)
            _panel_timing['axes'] += time.perf_counter() - _t0
            _t0 = time.perf_counter()
            # Reuses the mask/valid/xs/ys already computed once for this
            # section in the centroid loop above, instead of redoing the
            # expensive full-array `bg_section == sec` comparison here too.
            sec_mask, valid, xs_sec, ys_sec = sec_filtered[sec]
            ids_sec = bg_ids[sec_mask][valid]
            _panel_timing['mask'] += time.perf_counter() - _t0
            # Per-cell class/subclass/supertype/cluster labels (both the raw
            # category string and its leading numeric ID) and, where the
            # cell is also part of this run's own `adata` (not just the
            # unfiltered background), its row index there — precomputed once
            # here so recoloring the panel to match the UMAP's current
            # "Color by" mode (update_section_background_colors) doesn't
            # need to re-filter the whole-brain background arrays on every
            # redraw.
            _t0 = time.perf_counter()
            umap_idx_sec = np.array([cell_id_to_umap_idx.get(cid, -1) for cid in ids_sec], dtype=int)
            _panel_timing['umap_idx'] += time.perf_counter() - _t0
            _t0 = time.perf_counter()
            level_values_sec = {}
            level_ids_sec = {}
            for level in LEVEL_OPTIONS:
                arr = background_level_arrays[level]
                if arr is not None:
                    # The whole section's own labels, straight from the
                    # metadata CSV — covers every background cell, including
                    # ones this run filtered out.
                    vals = arr[sec_mask][valid]
                    # Leading numeric ID, sliced from the full-background
                    # precomputed array (background_level_ids_arrays, just
                    # above LEVEL_OPTIONS) with the same mask/valid indexing
                    # as vals itself — avoids re-running regex extraction on
                    # this panel's own slice.
                    level_values_sec[level] = vals
                    level_ids_sec[level] = background_level_ids_arrays[level][sec_mask][valid]
                    continue
                else:
                    # No background column for this level. True for Leiden by
                    # construction: it's computed from *this run's* neighbor
                    # graph, so it simply doesn't exist for cells outside
                    # `adata`, and no metadata CSV carries it. Rather than
                    # leave every panel uniformly gray on that level, fill in
                    # the cells that *are* in this run (via umap_idx) and
                    # leave the rest unknown — which the existing "gray
                    # behind, colored in front" convention already renders
                    # sensibly (see apply_panel_colors_with_gray_behind).
                    cells_arr = cell_level_arrays.get(level)
                    cell_ids_arr = level_ids_arrays.get(level)
                    if cells_arr is None:
                        level_values_sec[level] = None
                        level_ids_sec[level] = None
                        continue
                    in_run = umap_idx_sec >= 0
                    vals = np.full(umap_idx_sec.shape, None, dtype=object)
                    vals[in_run] = cells_arr[umap_idx_sec[in_run]]
                    ids = np.full(umap_idx_sec.shape, np.nan)
                    # level_ids_arrays (precomputed above LEVEL_OPTIONS,
                    # adata's own cells only) sliced by umap_idx_sec — same
                    # "index once, slice per panel" avoidance of re-running
                    # extract_leading_numeric_id per panel as the background
                    # branch above.
                    if cell_ids_arr is not None:
                        ids[in_run] = np.asarray(cell_ids_arr)[umap_idx_sec[in_run]]
                    level_values_sec[level] = vals
                    level_ids_sec[level] = ids
            _panel_timing['levels'] += time.perf_counter() - _t0
            # Antialiasing left on (matplotlib's default). It genuinely costs
            # something here — once these dots carry per-point color
            # (update_section_background_colors) rather than one uniform
            # fill, matplotlib can't use its fast single-color marker path
            # and rasterizes each point's antialiased edge individually,
            # across a whole section's worth of cells per panel. It was
            # switched off for that reason back when every zoom/pan tick
            # re-rendered all of them; now that a drag or scroll shows
            # cached bitmaps instead (see begin_zoom_preview), that cost is
            # paid once per settle rather than per tick, which buys back
            # smooth-edged dots — and with SECTION_POINT_ALPHA below 1.0,
            # aliased edges composite noticeably harsher where dots overlap.
            # alpha set once here and never again: matplotlib re-applies the
            # artist's stored alpha every time set_facecolor runs, so it
            # survives all the recoloring below (categorical, binary, and
            # colormapped alike) without each of those having to know.
            _t0 = time.perf_counter()
            background_artist = sec_ax.scatter(xs_sec, ys_sec, c='dimgray',
                                                s=SECTION_BACKGROUND_BASE_SIZE, linewidths=0,
                                                alpha=SECTION_POINT_ALPHA)
            _panel_timing['scatter'] += time.perf_counter() - _t0
            _t0 = time.perf_counter()
            # ROI rectangles drawn for this run (if any) — same dashed-
            # orange convention as the section-grid picker's own ROI
            # borders. Whole-section picks (the entire section selected,
            # no drawn rectangle) have blank x_min/x_max/y_min/y_max in
            # `rois` (see roi_csv_rows in the main pipeline) and are
            # skipped here — there's no meaningful rectangle to draw for
            # "the whole thing was selected".
            for roi in (rois or []):
                if roi['section'] != sec:
                    continue
                if any(pd.isna(roi[k]) for k in ('x_min', 'x_max', 'y_min', 'y_max')):
                    continue
                sec_ax.add_patch(Rectangle(
                    (roi['x_min'], roi['y_min']), roi['x_max'] - roi['x_min'], roi['y_max'] - roi['y_min'],
                    edgecolor='orange', facecolor='none', linestyle='--', linewidth=1.5, zorder=4,
                ))
            cx, cy = centroids[sec]
            home_xlim = (cx - shared_half_w, cx + shared_half_w)
            home_ylim = (cy - shared_half_h, cy + shared_half_h)
            sec_ax.set_xlim(home_xlim)
            sec_ax.set_ylim(home_ylim)
            # Same tissue-orientation fix as every other spatial (not UMAP-
            # embedding) plot in this file — y increases downward in this
            # atlas's own coordinate convention, opposite matplotlib's
            # default, so left uninverted a section renders top/bottom
            # flipped.
            sec_ax.invert_yaxis()
            sec_ax.set_title(sanitize_section_token(sec), fontsize=max(6, SIDEBAR_FONTSIZE * 0.55))
            # Hollow ring, not a filled dot — now that the background dots
            # themselves carry meaningful color (see update_section_
            # background_colors), a filled highlight would obscure it. Same
            # convention as umap_highlight_state's ring on the UMAP.
            highlight = sec_ax.scatter([], [], s=SECTION_HIGHLIGHT_BASE_SIZE,
                                        facecolor='none', edgecolor='red', linewidths=1.5, zorder=5)
            highlight.set_visible(False)
            # Animated for the same reason as umap_highlight_state's own
            # ring — see recreate_umap_highlight's comment.
            highlight.set_animated(True)
            # Half-brightness veil, shown only while a hover highlight is
            # engaged (see set_section_dimming), so the red ring/'+' markers
            # stand out against a panel that may already be full of red
            # cells at the current coloring.
            #
            # Deliberately an *overlay* rather than turning down the
            # background scatter's own alpha: the scatter lives in the
            # cached blit background (blit_bg), so changing it would force a
            # full re-render of every panel's scatter on every hover — the
            # exact cost the whole blit path exists to avoid. As an animated
            # artist it instead gets stamped on top of the restored
            # background each blit, costing one rectangle per panel.
            # transform=sec_ax.transAxes so it always covers the whole panel
            # regardless of the current zoom/pan data limits, and zorder
            # sits above the background dots but below both highlight
            # layers, so those draw at full strength over the dimmed field.
            dim_veil = Rectangle(
                (0, 0), 1, 1, transform=sec_ax.transAxes,
                facecolor='black', edgecolor='none', alpha=0.5, zorder=4.2,
            )
            dim_veil.set_visible(False)
            dim_veil.set_animated(True)
            sec_ax.add_patch(dim_veil)
            # One scale bar total, on the first panel actually built (not
            # necessarily sections_present[0] — a section can be skipped
            # above for missing centroid data). The other panels all share
            # this one's physical scale (see section_zoom_ratio), so one bar
            # already describes every one of them; kept in the enclosing
            # scope (section_scalebar, just below the panel loop) rather
            # than in this panel's own dict, since nothing else about it is
            # per-panel.
            if section_scalebar is None:
                # zorder above the zoom-preview stand-in images' own 1e6/
                # 1e6+1 (see begin_zoom_preview) and the cached home-view
                # image's 1e6+1 (see set_section_colors_categorical) — see
                # build_section_scalebar's own animated=True docstring for
                # why both matter.
                scalebar_line, scalebar_text = build_section_scalebar(
                    sec_ax, geometry=compute_scalebar_geometry(sec_ax), zorder=1e6 + 10, animated=True)
                section_scalebar = {'ax': sec_ax, 'line': scalebar_line, 'text': scalebar_text}
            section_panels[sec] = {
                'ax': sec_ax, 'highlight': highlight, 'group_highlight': None,
                'dim_veil': dim_veil,
                'background_artist': background_artist,
                # Filled in by apply_panel_colors_with_gray_behind on the
                # first real color assignment — None until then, which
                # filter_section_scatter_to_viewport treats as "nothing to
                # filter yet" (matches background_artist's own initial,
                # already-complete point set, so there's nothing to do
                # regardless).
                'full_offsets': None, 'full_colors': None,
                # A stand-in imshow for this panel's disk-cached home-view
                # PNG (see section_home_cache_path/show_section_home_
                # cache_or_scatter) — None until first used, created once
                # then just toggled visible/hidden afterward, same "create
                # once, reuse" convention as background_artist itself.
                'cached_home_image': None,
                # Precomputed once here (not re-filtered from the full bg_*
                # arrays on every hover settle) — same x/y/id triples the
                # background dots themselves were drawn from, just kept
                # around for resolve_section_hover_target's own nearest-
                # point search further down.
                'hover_x': xs_sec, 'hover_y': ys_sec, 'hover_ids': ids_sec,
                # Same-order companions to hover_x/y/ids, used by
                # update_section_background_colors to recolor this panel's
                # background dots to match whatever's currently driving the
                # UMAP's own coloring.
                'level_values': level_values_sec, 'level_ids': level_ids_sec, 'umap_idx': umap_idx_sec,
            }
            pannable_axes[sec_ax] = {'home_xlim': home_xlim, 'home_ylim': home_ylim}
            _panel_timing['other'] += time.perf_counter() - _t0
        if ZOOM_DEBUG_DIAGNOSTICS:
            _timing_str = ', '.join(f"{k}={v:.3f}s" for k, v in _panel_timing.items())
            print(f"[panel-build] {n_sections} panels — {_timing_str}")
        # Final draw+flush so every panel built after the last progress
        # checkpoint above (up to PANEL_PROGRESS_INTERVAL - 1 of them) is
        # also visible during the coloring phase that follows, rather than
        # only appearing at the very first real (colored) draw.
        fig.canvas.draw()
        fig.canvas.flush_events()
        log_status(f"Step 9: Built {n_sections} section panel(s) (axes + placeholder scatter only — "
                   f"real coloring happens in redraw_all_subclasses, timed separately).")
    else:
        empty_grid_ax = fig.add_axes([GRID_LEFT, AREA_BOTTOM, GRID_RIGHT - GRID_LEFT, AREA_TOP - AREA_BOTTOM])
        empty_grid_ax.axis('off')
        empty_grid_ax.text(0.5, 0.5, 'No section/spatial data available', ha='center', va='center',
                            fontsize=SIDEBAR_FONTSIZE, wrap=True)

    def update_section_scalebar():
        """Recompute the live scale bar's length/position from its panel's
        *current* view (zoom and/or pan) — a no-op if there are no section
        panels at all. Called after anything that changes that view."""
        if section_scalebar is None:
            return
        apply_scalebar_geometry(
            section_scalebar['line'], section_scalebar['text'],
            compute_scalebar_geometry(section_scalebar['ax']),
        )

    # --- Generic scroll-to-zoom / drag-to-pan ---------------------------
    # The UMAP scatter zooms independently, cursor-anchored, same as
    # before. Section panels are different: scrolling on *any* one of them
    # changes a single shared zoom ratio applied to *all* of them at once
    # (see section_zoom_ratio/apply_section_zoom_ratio below) — each panel
    # keeps its own current center (pan position), only the zoomed-in
    # *amount* is synced. Panning (below) stays independent per panel
    # either way — only zoom is special-cased here.
    def clamped_zoom_span(cur0, cur1, home0, home1, factor):
        # Sign-preserving: section panels are y-inverted (see sec_ax.
        # invert_yaxis() above, to match tissue orientation), so their
        # ylim is a *decreasing* pair (cur1 < cur0) — a plain (cur1-cur0)
        # there is negative, and comparing that against home_span (always
        # taken as a positive magnitude below) with min() would silently
        # pick the wrong one and never actually clamp. Magnitude is clamped
        # against the home span, then the original direction (increasing
        # or decreasing) is re-applied, so this works the same for the
        # UMAP axes (normal orientation) and every section panel (inverted)
        # without needing two separate code paths.
        cur_span = cur1 - cur0
        home_span = abs(home1 - home0)
        sign = 1 if cur_span >= 0 else -1
        return sign * min(abs(cur_span) * factor, home_span)

    def clamp_pan_center(center, half_span, home_lo, home_hi):
        """Keeps a [center-half_span, center+half_span] viewport from ever
        showing anything outside [home_lo, home_hi] — the section-panel
        equivalent of an image viewer not letting you pan past an image's
        own edges, so there's never empty canvas beyond it. When the
        viewport is as big as (or bigger than) the home extent itself
        (span >= home span — i.e. fully zoomed out), the *only* position
        satisfying that is dead center, which is exactly why this alone is
        enough to also reset panning to zero automatically at 1.0 zoom,
        with no separate "am I at 1.0, if so recenter" check needed."""
        home_lo, home_hi = min(home_lo, home_hi), max(home_lo, home_hi)
        if 2 * half_span >= (home_hi - home_lo):
            return (home_lo + home_hi) / 2
        return min(max(center, home_lo + half_span), home_hi - half_span)

    # 1.0 == every panel at its home (fully-zoomed-out) extent — grows as
    # any panel is scrolled in, shared by all of them (see
    # apply_section_zoom_ratio). All panels start at the *same* home span
    # (shared_half_w/shared_half_h above), so one shared ratio is always
    # enough to describe every panel's current zoom level, regardless of
    # how differently they've each been panned.
    section_zoom_ratio = {'value': 1.0}

    def section_zoom_diameter_multiplier():
        return 1.0 + SECTION_ZOOM_DOT_GROWTH_RATE * max(0.0, section_zoom_ratio['value'] - 1.0)

    def update_section_dot_size():
        multiplier = section_zoom_diameter_multiplier()
        for panel in section_panels.values():
            panel['background_artist'].set_sizes([SECTION_BACKGROUND_BASE_SIZE * multiplier ** 2])
            panel['highlight'].set_sizes([SECTION_HIGHLIGHT_BASE_SIZE * multiplier ** 2])
            if panel['group_highlight'] is not None:
                panel['group_highlight'].set_sizes([SECTION_GROUP_BASE_SIZE * multiplier ** 2])

    def apply_section_zoom_ratio(anchor_ax=None, anchor_x=None, anchor_y=None):
        """Recomputes every panel's view at the current section_zoom_ratio.

        `anchor_ax`/`anchor_x`/`anchor_y` (the panel actually under the
        cursor, and the cursor's own data-space position in it, from
        on_scroll_zoom) give a single shared *fractional* anchor point —
        how far across/down anchor_ax's own current view the cursor sat —
        computed once, up front, then applied to *every* panel using that
        panel's own current view, not just anchor_ax. The effect: every
        panel zooms toward the same relative on-screen spot the user is
        actually pointing at (e.g. "35% across, 60% down"), the same way a
        single axes' own cursor-anchored zoom keeps one data point fixed
        under the cursor — mirroring the UMAP's own frac_x/frac_y math in
        on_scroll_zoom, just shared across every section panel instead of
        confined to one axes. Previously only anchor_ax itself got this
        treatment; every other panel just kept its own existing center,
        which is what read as "the other panels zoom from dead center"
        regardless of where the cursor actually was.

        Omitted entirely (frac_x/frac_y fall back to 0.5, i.e. dead center)
        by the couple of other callers that just need to re-apply the
        existing ratio without any particular cursor position — e.g. after
        a resize — which reduces to each panel keeping its own center,
        exactly like before this shared-anchor behavior existed."""
        if not section_panels:
            return
        new_half_w = shared_half_w / section_zoom_ratio['value']
        new_half_h = shared_half_h / section_zoom_ratio['value']
        frac_x = frac_y = 0.5
        if anchor_ax is not None and anchor_x is not None and anchor_y is not None:
            ax0, ax1 = anchor_ax.get_xlim()
            ay0, ay1 = anchor_ax.get_ylim()
            frac_x = (anchor_x - ax0) / (ax1 - ax0) if ax1 != ax0 else 0.5
            frac_y = (anchor_y - ay0) / (ay1 - ay0) if ay1 != ay0 else 0.5
        for panel in section_panels.values():
            sec_ax_ = panel['ax']
            home = pannable_axes[sec_ax_]
            x0, x1 = sec_ax_.get_xlim()
            y0, y1 = sec_ax_.get_ylim()
            # Preserves each axis's current direction (section panels are
            # y-inverted — see sec_ax.invert_yaxis() above) rather than
            # assuming increasing order, same reasoning as clamped_zoom_
            # span's own sign handling.
            x_dir = 1 if (x1 - x0) >= 0 else -1
            y_dir = 1 if (y1 - y0) >= 0 else -1
            # This panel's own data point at the shared fraction above —
            # for anchor_ax itself this recovers anchor_x/anchor_y exactly
            # (frac_x/frac_y were derived from them via the same linear
            # relationship), so the panel actually being scrolled behaves
            # identically to before; every other panel gets its own
            # equivalent point instead of just its current center.
            anchor_data_x = x0 + frac_x * (x1 - x0)
            anchor_data_y = y0 + frac_y * (y1 - y0)
            new_x0 = anchor_data_x - frac_x * (x_dir * 2 * new_half_w)
            new_y0 = anchor_data_y - frac_y * (y_dir * 2 * new_half_h)
            cx, cy = new_x0 + x_dir * new_half_w, new_y0 + y_dir * new_half_h
            # Re-clamped at the *new* span on every zoom change, not just
            # panned — zooming back out grows new_half_w/h, and without
            # re-clamping here, a center that was a valid pan position at
            # the old (smaller) span could leave the new, larger viewport
            # hanging off the edge of home_xlim/home_ylim. At exactly 1.0
            # zoom this also forces cx/cy back to dead center (see clamp_
            # pan_center's own docstring), which is what actually resets
            # a panel's panning the moment it's fully zoomed back out. Also
            # what keeps the anchored point above from landing a viewport
            # partly outside home_xlim/home_ylim near an edge — same
            # tradeoff the UMAP's own cursor-anchored zoom makes.
            cx = clamp_pan_center(cx, new_half_w, home['home_xlim'][0], home['home_xlim'][1])
            cy = clamp_pan_center(cy, new_half_h, home['home_ylim'][0], home['home_ylim'][1])
            sec_ax_.set_xlim(cx - x_dir * new_half_w, cx + x_dir * new_half_w)
            sec_ax_.set_ylim(cy - y_dir * new_half_h, cy + y_dir * new_half_h)
        update_section_dot_size()
        # Same shared zoom ratio drove every panel's view above, so the one
        # scale bar (see build_section_scalebar) only ever needs recomputing
        # once here, not per panel.
        update_section_scalebar()

    def suspend_hover_during_zoom():
        """Cancels any pending hover/family-highlight timer and hides
        whatever's currently shown. Referenced here but defined in terms of
        hover_state/hide_all_highlights/tk_widget, which are only assigned
        further down this same function — fine, since this (like every
        other handler here) isn't actually called until the user scrolls,
        long after the rest of show_interactive_umap_window has finished
        running. Without this, rescaling the view out from under an
        already-shown (or about-to-appear) highlight made cells flash on
        and off while scroll-zooming, which read as broken rather than
        just busy."""
        if tk_widget is not None:
            if hover_state['timer_id'] is not None:
                tk_widget.after_cancel(hover_state['timer_id'])
                hover_state['timer_id'] = None
            if hover_state['group_timer_id'] is not None:
                tk_widget.after_cancel(hover_state['group_timer_id'])
                hover_state['group_timer_id'] = None
        hover_state['group_key'] = None
        # blit=False: called on *every* scroll tick, and the caller
        # (on_scroll_zoom) always does its own, more-current blit/draw
        # right after anyway (blit_zoomed_axes, or the full draw_idle() on
        # a burst's first tick) — blitting here too used to just get
        # silently overwritten within the same tick before Tk ever
        # actually painted it, which was harmless right up until
        # force_repaint() started making every blit hit the screen for
        # real: then this call's own restore_region(blit_bg) (whatever was
        # cached as of the *previous* tick) became a real, visible,
        # one-tick-stale frame that flashed on screen before the current
        # tick's real content replaced it a moment later — the zoom
        # "flickering backward".
        hide_all_highlights(blit=False)

    # --- Scroll-zoom preview: a cheap scaled bitmap stand-in while a scroll
    # burst is in progress, real full-quality redraw once it settles ------
    # Re-rendering every real artist (a section panel's own per-point-
    # colored scatter, or the full UMAP scatter) on *every single* scroll
    # notch is what made rapid zooming feel slow — each notch only needs to
    # look approximately right until the user stops scrolling, not be
    # pixel-perfect. active_zoom_previews tracks, per axes currently in a
    # preview, the stand-in image artist and every other child artist's
    # visibility (so it can be restored exactly, whatever it was) — hidden
    # for the duration, since the stand-in image (opaque, on top) covers
    # them anyway and there's no point paying to keep them in sync.
    active_zoom_previews = {}
    # Per-axes cache of a *full home-extent* snapshot (bitmap + the
    # home_xlim/home_ylim it was captured at), refreshed opportunistically
    # by cache_blit_background whenever that axes is drawn for real while
    # already sitting exactly at its own home view (e.g. right after any
    # mode/color redraw, since those now reset to home first). Zooming OUT
    # uses this instead of a fresh snapshot of the current, still-zoomed-in
    # view — imshow naturally crops a fixed-extent image to whatever the
    # axes' current (narrower or wider) view window is, so the same cached
    # bitmap can serve every tick of an out-zoom burst with no per-tick
    # updates, showing progressively more of the real picture exactly the
    # way a real zoom-out would — instead of the same fixed-extent crop of
    # the *zoomed-in* view shrinking into a hard-edged box as the viewport
    # widens past what that snapshot ever covered.
    home_view_cache = {}
    zoom_settle_timer = {'timer': None}
    # Which axes the *pending* settle is actually for — 'ax' (the literal
    # UMAP Axes object) for a UMAP scroll-zoom or a UMAP pan-release,
    # 'panel' for a section-panel scroll-zoom, set right before each of
    # on_scroll_zoom's two schedule_zoom_preview_settle(...) calls and in
    # on_release_pan. end_zoom_previews checks this (is ax) before applying
    # ZOOM_BITMAP_ONLY_MAX_MULTIPLIER — section-panel bursts are untouched
    # for now, and always fall through to the existing behavior below.
    zoom_settle_source = {'ax': None}
    # Debounce lengths (ZOOM_PREVIEW_SETTLE_MS_UMAP/_PANEL) and the snapshot
    # crop inset (SPINE_INSET_PX) are at the top of the file. The two
    # debounces differ even though a burst on either source freezes *all*
    # heavy axes together (see all_zoom_preview_axes) — the UMAP scatter's
    # own real redraw is worth waiting a bit longer to settle than a section
    # panel's.

    def snapshot_axes_region(target_ax):
        """A copy of `target_ax`'s own on-screen pixels, plus the imshow
        `extent` that places them back exactly where they came from.

        The extent is derived from the *actual* pixel rectangle that got
        copied, not from the axes' own xlim/ylim: the crop snaps to whole
        pixels, so it never lines up with the axes bounds exactly, and
        labelling the bitmap with those bounds instead shifted it by up to
        a pixel per side — enough to leave a hairline of whatever was
        underneath showing around the edge.

        The rectangle is also inset by SPINE_INSET_PX, so the axes' own
        spine — an antialiased dark line drawn *on* the boundary, bleeding
        a couple of pixels inward — never gets baked into the bitmap.
        Composited over the base layer, that baked-in border drew itself as
        a hard rectangle outline wherever the top layer ended: the black
        seams on both the UMAP and the section panels. The few pixels given
        up at the edge cost nothing, since the layer below (or the axes'
        own background) is showing the same content there anyway.
        """
        renderer = fig.canvas.get_renderer()
        if renderer is None:
            return None
        # The scale bar (see build_section_scalebar's own animated=True
        # docstring) is drawn separately, always on top, via draw_
        # animated_overlays — it must never end up baked into a snapshot
        # bitmap itself, or it gets magnified/shifted right along with the
        # rest of the image as that bitmap is zoomed, and can end up
        # doubled up with the live one once both are on screen at once (a
        # snapshot from one moment layered under home_view_cache's own
        # snapshot from another — see capture_home_view_if_at_home —
        # showing two bars at two different lengths/positions). Hidden and
        # redrawn into just the *renderer's* own buffer (fig.draw_artist,
        # no blit — nothing needs to reach the screen for this, only the
        # pixels read back below), then restored to visible immediately
        # after: draw_animated_overlays re-stamps it correctly on screen
        # the next time anything blits, which every real caller of this
        # function does soon afterward regardless.
        is_scalebar_ax = section_scalebar is not None and target_ax is section_scalebar['ax']
        if is_scalebar_ax:
            section_scalebar['line'].set_visible(False)
            section_scalebar['text'].set_visible(False)
            fig.draw_artist(target_ax)
        try:
            buf = np.asarray(renderer.buffer_rgba())
            h = buf.shape[0]
            bbox = target_ax.bbox
            x0 = max(0, int(np.ceil(bbox.x0)) + SPINE_INSET_PX)
            x1 = min(buf.shape[1], int(np.floor(bbox.x1)) - SPINE_INSET_PX)
            y0 = max(0, int(np.ceil(bbox.y0)) + SPINE_INSET_PX)
            y1 = min(h, int(np.floor(bbox.y1)) - SPINE_INSET_PX)
            if x1 <= x0 or y1 <= y0:
                return None
            # Row 0 of buffer_rgba() is the *top* of the canvas; matplotlib's
            # own y-pixel coordinates (bbox) increase upward — hence the flip.
            snapshot = buf[h - y1:h - y0, x0:x1, :].copy()
            inv = target_ax.transData.inverted()
            (ex0, ey0), (ex1, ey1) = inv.transform([(x0, y0), (x1, y1)])
            # (left, right, bottom, top) in the axes' own coordinate directions —
            # taken straight from the inverse transform, so an inverted axes
            # (section panels) gets a correctly-flipped extent with no special
            # case, exactly as origin='upper' expects.
            return snapshot, (ex0, ex1, ey0, ey1)
        finally:
            if is_scalebar_ax:
                section_scalebar['line'].set_visible(True)
                section_scalebar['text'].set_visible(True)

    def begin_zoom_preview(target_ax, use_home_cache=False):
        if target_ax in active_zoom_previews:
            return  # already mid-burst for this axes
        captured = snapshot_axes_region(target_ax)
        if captured is None:
            return
        snapshot, snapshot_extent = captured
        cached_home = home_view_cache.get(target_ax) if use_home_cache else None
        # Excludes xaxis/yaxis/spines — these own the axis labels ("UMAP1"/
        # "UMAP2") and the box border, none of which are expensive to
        # redraw (there are no tick labels to speak of; both are turned
        # off), so there was never a performance reason to touch them here.
        # Toggling ax.xaxis/ax.yaxis off and back on around a zoom burst
        # was also what caused the axis label to go missing and the
        # effective plot box to visibly shift — 'equal'/'box' aspect
        # adjustment recomputes against whatever's currently governing the
        # axes' layout, and an Axis flipping invisible then visible again
        # median-burst is exactly the kind of state change that shouldn't
        # have been happening for a purely cosmetic, cheap artist anyway.
        # target_ax.patch (the axes' own background) is structural too: with
        # it hidden, the few pixels a preview bitmap doesn't cover fell
        # through to the *figure's* white background rather than this axes'
        # own — which is where the white seams around the section panels
        # (black backgrounds) came from.
        structural = {target_ax.xaxis, target_ax.yaxis, target_ax.patch,
                      *target_ax.spines.values()}
        if section_scalebar is not None and target_ax is section_scalebar['ax']:
            # Animated (see build_section_scalebar's own animated=True
            # docstring) — never hidden for a preview burst the way every
            # other real artist on this axes is; it's drawn separately,
            # always on top of whatever the preview ends up showing, via
            # draw_animated_overlays.
            structural = structural | {section_scalebar['line'], section_scalebar['text']}
        hidden = [(artist, artist.get_visible()) for artist in target_ax.get_children()
                  if hasattr(artist, 'get_visible') and artist not in structural]
        for artist, _ in hidden:
            artist.set_visible(False)
        # imshow(aspect=...) sets the *axes'* own aspect, not just the
        # image's — passing 'auto' silently dropped this axes' 'equal'/'box'
        # setting for the whole burst (and beyond: end_zoom_previews only
        # restores artist visibility, so it stayed 'auto' until the next
        # full redraw re-applied it). That's what let the axes box grow
        # past its letterboxed size mid-zoom, and it also stretched the
        # preview itself, since the bitmap was captured at the *letterboxed*
        # box size and then drawn into a taller one. Saved and put back
        # immediately below, so the preview renders in exactly the geometry
        # its snapshot was taken in.
        saved_aspect = target_ax.get_aspect()
        saved_adjustable = target_ax.get_adjustable()
        images = []
        if cached_home is not None:
            # Underneath the crisp snapshot: the cached home-extent bitmap,
            # which is the only thing that covers the parts of the view that
            # come *into* frame as the viewport widens. On its own it isn't
            # enough — for a sparse plot like the UMAP (mostly white space
            # at home extent), magnifying a small crop of it while zoomed in
            # reads as a blank screen — hence the pairing rather than a
            # straight swap.
            images.append(target_ax.imshow(
                cached_home['buf'], extent=cached_home['extent'],
                aspect='auto', origin='upper', zorder=1e6, interpolation='nearest',
            ))
        # On top: the full-resolution snapshot of the view as it was when
        # this burst started, so the middle of the frame — the part the user
        # is actually looking at — stays sharp instead of being replaced by
        # a magnified crop of the home bitmap.
        images.append(target_ax.imshow(
            snapshot, extent=snapshot_extent, aspect='auto',
            origin='upper', zorder=1e6 + 1, interpolation='nearest',
        ))
        target_ax.set_aspect(saved_aspect, adjustable=saved_adjustable)
        active_zoom_previews[target_ax] = {'images': images, 'hidden': hidden,
                                           'has_home_base': cached_home is not None,
                                           'aspect': (saved_aspect, saved_adjustable)}

    def swap_preview_to_home_cache(target_ax):
        """Adds the cached home-extent bitmap *underneath* an already-running
        preview. begin_zoom_preview only decides whether to include that base
        layer once, at the start of a burst — so a burst that begins by
        zooming *in* (which has no need for it) and then reverses direction
        without pausing long enough to settle would otherwise spend the rest
        of the burst with nothing covering the area coming into frame, which
        is exactly the shrinking-boxes artifact the home cache exists to
        avoid."""
        state = active_zoom_previews.get(target_ax)
        if state is None or state['has_home_base']:
            return
        cached_home = home_view_cache.get(target_ax)
        if cached_home is None:
            return
        saved_aspect, saved_adjustable = state['aspect']
        base = target_ax.imshow(
            cached_home['buf'], extent=cached_home['extent'],
            aspect='auto', origin='upper', zorder=1e6, interpolation='nearest',
        )
        target_ax.set_aspect(saved_aspect, adjustable=saved_adjustable)
        state['images'].insert(0, base)
        state['has_home_base'] = True

    def teardown_zoom_previews():
        """Drop every active preview's stand-in image and restore the real
        artists' visibility/aspect. No drawing — callers own that.

        Every step is individually guarded because the preview images can
        legitimately be *already gone* by the time this runs: anything that
        calls ax.clear() (any redraw_* — reachable mid-burst via the resize
        settle's own maybe_redraw_for_legend_resize) removes them and, in
        doing so, clears the _remove_method matplotlib's Artist.remove()
        needs, so a second remove() raises NotImplementedError('cannot
        remove artist'). That exception used to escape end_zoom_previews
        before it could clear active_zoom_previews or re-enable the sidebar,
        which wedged the whole window: blit_hover_overlays early-returns
        while a preview is registered, so hover and every other blit-driven
        interaction went dead while pan/zoom (which don't use that path)
        kept working. The state reset in `finally` is what makes that
        unwedgeable regardless of what else goes wrong here."""
        try:
            for target_ax, state in active_zoom_previews.items():
                for image in state['images']:
                    try:
                        image.remove()
                    except Exception:
                        pass  # already removed by an ax.clear() — see above
                for artist, was_visible in state['hidden']:
                    try:
                        artist.set_visible(was_visible)
                    except Exception:
                        pass
                # Belt and braces alongside begin_zoom_preview's own
                # immediate restore — every imshow() there re-set the axes'
                # aspect, so put back what this axes actually had before any
                # of it happened.
                try:
                    saved_aspect, saved_adjustable = state['aspect']
                    target_ax.set_aspect(saved_aspect, adjustable=saved_adjustable)
                except Exception:
                    pass
        finally:
            active_zoom_previews.clear()

    def discard_zoom_previews():
        """Tear down previews *and* cancel the pending settle, for callers
        that are about to rebuild the axes from scratch anyway (redraw_*).
        Without cancelling, the settle timer would still fire afterward and
        run end_zoom_previews against artists that no longer exist."""
        if zoom_settle_timer['timer'] is not None:
            try:
                zoom_settle_timer['timer'].stop()
            except Exception:
                pass
            zoom_settle_timer['timer'] = None
        if active_zoom_previews:
            teardown_zoom_previews()

    def end_zoom_previews():
        if not active_zoom_previews:
            return
        _settle_t0 = time.perf_counter()
        umap_settle = zoom_settle_source['ax'] is ax
        if ZOOM_DEBUG_DIAGNOSTICS:
            print(f"[zoom-settle] source={zoom_settle_source['ax']!r} "
                  f"umap_ratio={umap_zoom_ratio():.2f}x panel_ratio={section_zoom_ratio['value']:.2f}x")
        if umap_settle and umap_zoom_ratio() <= ZOOM_BITMAP_ONLY_MAX_MULTIPLIER:
            # Below the threshold, stay on the zoom-preview bitmap already
            # on screen indefinitely rather than paying for a real, full-
            # resolution fig.canvas.draw() — see ZOOM_BITMAP_ONLY_MAX_
            # MULTIPLIER's own comment for why that draw is expensive
            # regardless of how little ends up visible. teardown_zoom_
            # previews() (not the heavier finish below) does only the
            # *in-memory* bookkeeping — restoring the real (possibly
            # stale/viewport-filtered — see the >threshold branch below)
            # artists' visibility and clearing active_zoom_previews —
            # without ever actually drawing them, so that staleness is
            # never painted. Re-caching blit_bg first (same as the real
            # path below) matters on its own: several other things (hover's
            # own blit path, capture_home_view_if_at_home, the resize-
            # settle handler) treat a non-empty active_zoom_previews as "a
            # preview owns the screen right now" and suppress themselves
            # accordingly — correct for the few hundred milliseconds a real
            # settle used to take, but hover in particular would otherwise
            # stay dead for as long as the user's zoom stays in this range,
            # which could be indefinitely — so this clears it every time,
            # never leaves it set.
            blit_bg['data'] = fig.canvas.copy_from_bbox(fig.bbox)
            teardown_zoom_previews()
            if ZOOM_DEBUG_DIAGNOSTICS:
                print(f"[zoom-settle] umap<=threshold (bitmap-only) in {time.perf_counter() - _settle_t0:.3f}s")
            return
        if umap_settle and umap_zoom_ratio() > ZOOM_BITMAP_ONLY_MAX_MULTIPLIER:
            # Re-filters from main_scatter_state['full']'s own untouched
            # snapshot every time (never from whatever the artists
            # currently hold), so this is correct however many times the
            # user has already crossed the threshold this session.
            filter_main_scatter_to_viewport()
            # Targeted finish, not the generic full-figure one below: a
            # plain fig.canvas.draw() redraws *every* visible Axes, which
            # includes every section panel — all ~80 of them, each with
            # their own few-thousand-point real scatter — even though a
            # UMAP-only zoom never touched any of them. teardown_zoom_
            # previews() still restores every axes' real-artist visibility
            # (in memory only, same as the skip branch above — see its own
            # comment on why that has to stay a *full*, not per-axes,
            # teardown: several other guards elsewhere key off active_zoom_
            # previews being empty, not per-axes), but only `ax` itself
            # actually gets *drawn* here (fig.draw_artist, same low-level
            # call blit_zoomed_axes already uses for every mid-burst tick —
            # it doesn't fire 'draw_event', so cache_blit_background never
            # runs and blit_bg has to be re-cached by hand afterward, same
            # as blit_zoomed_axes's own callers already rely on during a
            # burst). Every section panel's own screen pixels are simply
            # never touched: their stand-in bitmap was already a pixel-
            # accurate snapshot of their real, unchanged content, so the
            # canvas buffer is already correct there without redrawing
            # anything — blitting the *whole* canvas (not just ax.bbox,
            # same as blit_zoomed_axes) just re-pushes those already-
            # correct pixels alongside the UMAP's freshly drawn ones.
            teardown_zoom_previews()
            fig.draw_artist(ax)
            fig.canvas.blit(fig.bbox)
            force_repaint()
            blit_bg['data'] = fig.canvas.copy_from_bbox(fig.bbox)
            if ZOOM_DEBUG_DIAGNOSTICS:
                print(f"[zoom-settle] umap>threshold (filtered real draw) in {time.perf_counter() - _settle_t0:.3f}s")
            return
        panel_settle = zoom_settle_source['ax'] == 'panel'
        if panel_settle and section_zoom_ratio['value'] <= ZOOM_BITMAP_ONLY_MAX_MULTIPLIER:
            # Mirror image of the UMAP skip branch above — same reasoning,
            # same guarantees (blit_bg re-cached, active_zoom_previews
            # actually cleared rather than left "stuck", so hover etc.
            # don't go dead for the whole time zoom stays under threshold).
            blit_bg['data'] = fig.canvas.copy_from_bbox(fig.bbox)
            teardown_zoom_previews()
            if ZOOM_DEBUG_DIAGNOSTICS:
                print(f"[zoom-settle] panel<=threshold (bitmap-only) in {time.perf_counter() - _settle_t0:.3f}s")
            return
        if panel_settle and section_zoom_ratio['value'] > ZOOM_BITMAP_ONLY_MAX_MULTIPLIER:
            # Mirror image of the UMAP >threshold branch above: filter
            # every panel to its own current viewport, then draw+blit only
            # the section panels — never the UMAP, which this zoom never
            # touched. section_panels.values() is a lot of axes to draw
            # individually, but each is now cheap (only the in-view subset
            # of that one panel's own points), and it's still strictly less
            # work than the full fig.canvas.draw() this replaces, which
            # drew all of them *and* the UMAP.
            # teardown_zoom_previews() *first*, not after: it restores every
            # artist's pre-burst visibility (see its own state['hidden']),
            # which for a panel that was showing its cached home-view image
            # at burst start means re-hiding background_artist and re-
            # showing cached_home_image — exactly undoing filter_all_
            # section_scatters_to_viewport()'s own swap to the real,
            # zoomed-in scatter if that ran first. This ordering bug was
            # why zooming a section panel in past ZOOM_BITMAP_ONLY_MAX_
            # MULTIPLIER never actually improved its resolution: the swap
            # happened, then was immediately clobbered back to the cached
            # bitmap, every single settle, regardless of how far past the
            # threshold the zoom went.
            teardown_zoom_previews()
            filter_all_section_scatters_to_viewport()
            for panel in section_panels.values():
                fig.draw_artist(panel['ax'])
            fig.canvas.blit(fig.bbox)
            force_repaint()
            blit_bg['data'] = fig.canvas.copy_from_bbox(fig.bbox)
            if ZOOM_DEBUG_DIAGNOSTICS:
                print(f"[zoom-settle] panel>threshold (filtered real draw) in {time.perf_counter() - _settle_t0:.3f}s")
            return
        # Re-cache the full-figure background from what's *currently* in the
        # Agg buffer before anything below restores it. Only a burst's first
        # tick does a real draw (the one thing that fires draw_event ->
        # cache_blit_background); every tick after that goes through
        # blit_zoomed_axes, which draws and blits without a draw_event — so
        # by settle time blit_bg still holds tick 1's pixels, i.e. very
        # nearly the pre-zoom view. show_working_indicator() immediately
        # below restores that region (blit_sidebar_overlays' restore_region
        # repaints the whole in-memory buffer, not just the sidebar bbox it
        # pushes), which put that stale, near-original-zoom frame on screen
        # for the split second before the real draw() landed — the zoom
        # appearing to briefly revert itself. Same stale-blit_bg hazard
        # suspend_hover_during_zoom's own blit=False guards against; one
        # copy per settle is negligible next to the full draw that follows.
        if ZOOM_DEBUG_DIAGNOSTICS:
            print(f"[zoom-settle] FALLBACK (neither umap_settle nor panel_settle matched) — "
                  f"about to do a full fig.canvas.draw()")
        blit_bg['data'] = fig.canvas.copy_from_bbox(fig.bbox)
        # Shown *before* any of the (comparatively cheap) cleanup below,
        # so it's on screen as early as possible ahead of the genuinely
        # slow, blocking fig.canvas.draw() further down — matplotlib/Tk
        # rendering is single-threaded, so nothing else can run once that
        # call actually starts; this at least gives the user something to
        # look at instead of the window just going unresponsive.
        show_working_indicator()
        teardown_zoom_previews()
        # Synchronous (fig.canvas.draw(), not draw_idle()) so cleanup and
        # the real redraw are one atomic step. draw_idle() only *schedules*
        # the real render for later — if a new scroll tick landed in the
        # gap between active_zoom_previews.clear() (already run, above)
        # and that deferred render actually executing, on_scroll_zoom would
        # see no active preview, start a *second* fresh burst (its own
        # real render) on top of the one already pending, and the two
        # would have to be processed back to back — which is what actually
        # made zooming feel like it was "waiting for the redraw", not
        # anything inherent to a single real draw's own cost.
        # Cleared *before* the real draw, not after: fig.canvas.draw() is
        # synchronous, and Figure.draw() fires 'draw_event' (triggering our
        # own cache_blit_background, which re-stamps every animated artist
        # — including working_text — back on top of whatever it just
        # rendered) *before draw() itself returns*. Clearing the text after
        # draw() meant cache_blit_background's own re-stamp used the
        # still-stale "Working…" content, which is exactly what kept
        # showing until some later, unrelated interaction happened to
        # re-stamp it correctly (by then empty).
        # try/finally so a failure in the real draw can't leave the sidebar
        # stuck disabled with "Working…" still showing — show_working_
        # indicator() above dimmed the controls, and this is the only thing
        # that undoes that.
        try:
            hide_working_indicator()
            fig.canvas.draw()
        finally:
            force_repaint()  # see its own docstring — draw()'s own internal blit needs this too
            if ZOOM_DEBUG_DIAGNOSTICS:
                print(f"[zoom-settle] FALLBACK full draw complete in {time.perf_counter() - _settle_t0:.3f}s")

    def schedule_zoom_preview_settle(interval_ms):
        if zoom_settle_timer['timer'] is not None:
            zoom_settle_timer['timer'].stop()
        timer = fig.canvas.new_timer(interval=interval_ms)
        timer.single_shot = True
        timer.add_callback(end_zoom_previews)
        zoom_settle_timer['timer'] = timer
        timer.start()

    def all_zoom_preview_axes():
        # Every axes with an expensive real background — the UMAP scatter
        # *and* every section panel's own scatter — regardless of which one
        # is actually being scrolled: a plain fig.canvas.draw_idle() redraws
        # the *whole* figure either way, so zooming the UMAP alone still
        # paid to fully re-rasterize all ~80 section panels' real scatters
        # (never hidden, since only the UMAP's own view was changing) on
        # every single scroll tick — that, not anything about the UMAP
        # itself, was why UMAP zoom stayed slow even with its own preview
        # image visibly active. Freezing everything up front avoids that
        # regardless of which axes the zoom started on.
        return [ax] + [panel['ax'] for panel in section_panels.values()]

    def blit_zoomed_axes(target_axes):
        # Cheap redraw+push of just `target_axes`' own screen regions —
        # used for every zoom-preview tick after the first, once every
        # heavy axes is already frozen as a stand-in image: only the
        # image(s) actually being zoomed need to be *drawn* (a single cheap
        # image resample apiece, not thousands of real points), and only
        # their own screen pixels need to be *pushed*, not the whole
        # canvas's — a full fig.canvas.draw_idle() (and the draw_event-
        # triggered full-canvas copy_from_bbox it schedules) touches every
        # section panel's Axes regardless of whether its own image content
        # actually changed this tick.
        for target_ax in target_axes:
            fig.draw_artist(target_ax)
        fig.canvas.blit(fig.bbox)

    def on_scroll_zoom(event):
        ax_obj = event.inaxes
        if ax_obj is ax:
            if ax not in pannable_axes or event.xdata is None or event.ydata is None:
                return
            suspend_hover_during_zoom()
            # Pushes the now-hidden highlight to the actual screen buffer —
            # same reasoning, and same fix, as on_press_pan's own call right
            # after its own suspend_hover_during_zoom(). suspend_hover_
            # during_zoom() uses blit=False (deliberately — see its own
            # comment), so on its own it only updates the highlight
            # artists' visibility in memory; nothing pushes that to screen
            # here. begin_zoom_preview's own snapshot_axes_region, just
            # below, reads pixels straight from the renderer's *current*
            # buffer (no draw of its own) — so without this, a highlight
            # that was on screen a moment ago got baked into the zoom
            # preview bitmap and sat there, frozen, for the whole burst,
            # only actually disappearing once the burst settled into a real
            # draw. Safe to call unconditionally on every tick, not just the
            # first: blit_hover_overlays() itself already no-ops while a
            # zoom burst is already active (see its own guard), so this
            # only ever does real work on the tick that starts a fresh one.
            blit_hover_overlays()
            burst_already_active = bool(active_zoom_previews)
            zooming_out = event.button != 'up'
            # Home cache only for the axes actually being zoomed. Every
            # *other* axes is frozen here purely so it doesn't get re-
            # rendered mid-burst (see all_zoom_preview_axes) — its view
            # isn't changing, so it must keep showing exactly what's on
            # screen now. Handing a zoomed-in section panel its *home*
            # bitmap while the UMAP is what's being scrolled would make
            # every panel visibly jump back to full extent for the duration.
            for target_ax in all_zoom_preview_axes():
                begin_zoom_preview(target_ax, use_home_cache=(zooming_out and target_ax is ax))
            if zooming_out:
                swap_preview_to_home_cache(ax)
            home = pannable_axes[ax]
            factor = 0.85 if event.button == 'up' else (1 / 0.85)
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            new_w = clamped_zoom_span(x0, x1, home['home_xlim'][0], home['home_xlim'][1], factor)
            new_h = clamped_zoom_span(y0, y1, home['home_ylim'][0], home['home_ylim'][1], factor)
            frac_x = (event.xdata - x0) / (x1 - x0) if x1 != x0 else 0.5
            frac_y = (event.ydata - y0) / (y1 - y0) if y1 != y0 else 0.5
            new_x0 = event.xdata - frac_x * new_w
            new_y0 = event.ydata - frac_y * new_h
            # Cursor-anchored zoom (above) can still land a viewport partly
            # outside home_xlim/home_ylim near an edge — clamped the same
            # way as the section panels now, so scrolling near the UMAP's
            # own boundary can't reveal empty canvas past it either. This
            # re-centers rather than staying perfectly cursor-anchored in
            # that edge case, which is the same tradeoff clamp_pan_center
            # already makes for panels.
            half_w, half_h = abs(new_w) / 2, abs(new_h) / 2
            cx = clamp_pan_center(new_x0 + new_w / 2, half_w, home['home_xlim'][0], home['home_xlim'][1])
            cy = clamp_pan_center(new_y0 + new_h / 2, half_h, home['home_ylim'][0], home['home_ylim'][1])
            x_dir = 1 if new_w >= 0 else -1
            y_dir = 1 if new_h >= 0 else -1
            ax.set_xlim(cx - x_dir * half_w, cx + x_dir * half_w)
            ax.set_ylim(cy - y_dir * half_h, cy + y_dir * half_h)
            update_umap_dot_size()
            if burst_already_active:
                blit_zoomed_axes([ax])
            else:
                # Targeted (fig.draw_artist(ax), same as blit_zoomed_axes'
                # own mid-burst call just above), not a full fig.canvas.
                # draw() — begin_zoom_preview (just above) already hid
                # every real artist on *every* axes (UMAP and all section
                # panels alike), so a full draw's own section-panel work
                # should in principle be near-free with nothing real left
                # visible to draw there — but it still means Figure.draw()
                # walking every one of those ~80 axes' own structural
                # bits (patch/spines/labels) and however many now-hidden
                # children each one has, purely to confirm there's nothing
                # to do. Drawing only `ax` skips that walk entirely; the
                # sections' own stand-in images (just added, never drawn
                # yet) don't need drawing either — each one's pixel content
                # is a snapshot of whatever was already correctly on
                # screen a moment before begin_zoom_preview hid the real
                # artists, so the canvas buffer is already right there
                # without painting anything new.
                #
                # Still synchronous, still not draw_idle(): a deferred
                # draw here left a window where the preview image (already
                # showing the correct zoomed view, right above) could sit
                # on screen for one or more idle-loop turns before this
                # draw actually ran — and if anything else serviced the Tk
                # event queue in that gap (a queued resize/configure from
                # the window only just having been mapped, in particular
                # right after the window first opens), the canvas could
                # repaint with the *old* home-extent content in between,
                # reading as a flicker back to zoomed-out before this
                # draw's own result finally landed. blit_zoomed_axes is
                # just as synchronous as fig.canvas.draw() was — same
                # guarantee, by the time this call returns the correct
                # zoomed view is already the only thing that's been drawn.
                blit_zoomed_axes([ax])
                force_repaint()
            zoom_settle_source['ax'] = ax
            schedule_zoom_preview_settle(ZOOM_PREVIEW_SETTLE_MS_UMAP)
            return
        if any(panel['ax'] is ax_obj for panel in section_panels.values()):
            suspend_hover_during_zoom()
            # See the mirror-image comment in the UMAP branch above — same
            # fix, same reason: without this, a highlight active on a
            # section panel a moment ago gets baked into that burst's own
            # zoom-preview bitmap and stays frozen on screen for the whole
            # zoom, instead of disappearing the instant the scroll starts.
            blit_hover_overlays()
            burst_already_active = bool(active_zoom_previews)
            zooming_out = event.button != 'up'
            # Section panels all zoom together (one shared section_zoom_
            # ratio), so they're the ones getting the home cache here — the
            # UMAP is the bystander this time and keeps its current view.
            # See the mirror-image comment in the UMAP branch above.
            section_axes = [panel['ax'] for panel in section_panels.values()]
            for target_ax in all_zoom_preview_axes():
                begin_zoom_preview(target_ax, use_home_cache=(zooming_out and target_ax is not ax))
            if zooming_out:
                for target_ax in section_axes:
                    swap_preview_to_home_cache(target_ax)
            factor = 0.85 if event.button == 'up' else (1 / 0.85)
            # factor < 1 (scrolling in) shrinks the span, so the ratio
            # (home_span / span) grows by dividing by factor instead of
            # multiplying — same relationship clamped_zoom_span expresses
            # the other way around (as a span, not a ratio). Never lets the
            # ratio drop below 1.0 (home/fully zoomed out).
            section_zoom_ratio['value'] = max(1.0, section_zoom_ratio['value'] / factor)
            apply_section_zoom_ratio(ax_obj, event.xdata, event.ydata)
            if burst_already_active:
                blit_zoomed_axes(section_axes)
            else:
                # Targeted (blit_zoomed_axes), not a full fig.canvas.draw()
                # — same reasoning as the mirror-image fix in the UMAP
                # branch above: begin_zoom_preview already hid every real
                # artist on every axes (including the UMAP's own, which
                # this zoom never touches), so a full draw's UMAP-side work
                # is theoretically near-free but still means walking its
                # own (large) child list to confirm there's nothing to do.
                # Still synchronous — same flicker risk from a deferred
                # draw_idle() leaving a gap before the burst-starting real
                # draw lands.
                blit_zoomed_axes(section_axes)
                force_repaint()
            zoom_settle_source['ax'] = 'panel'
            schedule_zoom_preview_settle(ZOOM_PREVIEW_SETTLE_MS_PANEL)

    fig.canvas.mpl_connect('scroll_event', on_scroll_zoom)

    pan_state = {'active': False, 'ax': None, 'last_pixel': None, 'preview_started': False}

    def on_press_pan(event):
        if event.inaxes not in pannable_axes or event.button != 1:
            return
        pan_state['active'] = True
        pan_state['ax'] = event.inaxes
        pan_state['last_pixel'] = (event.x, event.y)
        # Previews are *not* started here — a plain click (press with no
        # drag) would then pay for building and tearing them down, including
        # the full redraw that ending them forces, for no visible benefit.
        # Deferred to the first actual motion instead.
        pan_state['preview_started'] = False
        # Hover suppression, however, *is* done right here at press time —
        # not deferred to the first drag motion the way the preview is.
        # on_umap_motion/on_section_motion already skip scheduling a *new*
        # hover timer while pan_state['active'] is True, but a timer
        # scheduled from mouse movement a moment *before* this press isn't
        # touched by that check, and would otherwise still fire mid-drag —
        # exactly the "clicking near a cell triggers a highlight" case,
        # since starting a pan almost always means pressing down right on
        # top of a cell that was just hovered. Cancelling and hiding
        # immediately here closes that gap instead of racing it.
        suspend_hover_during_zoom()
        blit_hover_overlays()

    def on_motion_pan(event):
        if not pan_state['active'] or event.inaxes is not pan_state['ax']:
            return
        ax_obj = pan_state['ax']
        preview_just_started = not pan_state['preview_started']
        if preview_just_started:
            # Same stand-in-bitmap treatment a scroll burst gets, for the
            # same reason: a drag fires motion events far faster than every
            # section panel's real scatter can be re-rendered, and a plain
            # draw_idle() re-renders *all* of them on every one of those
            # events even though only the dragged axes' view is changing.
            #
            # Panning is the ideal case for the cached home bitmap: the drag
            # is clamped so the viewport can never leave the axes' home
            # extent (see clamp_pan_center), and that bitmap covers exactly
            # that extent — so whatever scrolls into frame is always real
            # content, with the crisp snapshot of where the drag started
            # riding on top.
            suspend_hover_during_zoom()
            # A scroll burst just before this drag may still have its settle
            # timer pending; letting that fire mid-drag would tear the
            # stand-ins down underneath the pan and put every real artist
            # back, which is exactly the per-motion full redraw this is
            # avoiding. The release handler ends the previews instead.
            if zoom_settle_timer['timer'] is not None:
                zoom_settle_timer['timer'].stop()
                zoom_settle_timer['timer'] = None
            for target_ax in all_zoom_preview_axes():
                begin_zoom_preview(target_ax, use_home_cache=(target_ax is ax_obj))
            swap_preview_to_home_cache(ax_obj)
            pan_state['preview_started'] = True
        inv = ax_obj.transData.inverted()
        x0_data, y0_data = inv.transform(pan_state['last_pixel'])
        x1_data, y1_data = inv.transform((event.x, event.y))
        dx, dy = x0_data - x1_data, y0_data - y1_data
        xlim, ylim = ax_obj.get_xlim(), ax_obj.get_ylim()
        new_x0, new_x1 = xlim[0] + dx, xlim[1] + dx
        new_y0, new_y1 = ylim[0] + dy, ylim[1] + dy
        # Clamped so dragging can never reveal empty canvas past that
        # axes' own home extent (see clamp_pan_center's own docstring) —
        # the UMAP scatter and every section panel alike.
        home = pannable_axes.get(ax_obj)
        if home is not None:
            half_w = abs(new_x1 - new_x0) / 2
            half_h = abs(new_y1 - new_y0) / 2
            cx = clamp_pan_center((new_x0 + new_x1) / 2, half_w, home['home_xlim'][0], home['home_xlim'][1])
            cy = clamp_pan_center((new_y0 + new_y1) / 2, half_h, home['home_ylim'][0], home['home_ylim'][1])
            x_dir = 1 if (new_x1 - new_x0) >= 0 else -1
            y_dir = 1 if (new_y1 - new_y0) >= 0 else -1
            new_x0, new_x1 = cx - x_dir * half_w, cx + x_dir * half_w
            new_y0, new_y1 = cy - y_dir * half_h, cy + y_dir * half_h
        ax_obj.set_xlim(new_x0, new_x1)
        ax_obj.set_ylim(new_y0, new_y1)
        # Panning doesn't change the zoom (so not a job for update_section_
        # scalebar's *other* caller, apply_section_zoom_ratio) but it does
        # move this panel's own corner, which is where the bar is anchored —
        # only relevant if the panel being dragged is the one carrying it.
        if section_scalebar is not None and ax_obj is section_scalebar['ax']:
            update_section_scalebar()
        pan_state['last_pixel'] = (event.x, event.y)
        if preview_just_started:
            # One full draw to get the newly-created stand-ins (and the
            # now-hidden real artists) onto the canvas properly; every
            # motion after this can then just blit the dragged axes. Same
            # first-tick/rest-of-burst split as on_scroll_zoom — without it
            # the first blit would leave the real artists' already-rendered
            # pixels sitting underneath the preview.
            fig.canvas.draw_idle()
        else:
            blit_zoomed_axes([ax_obj])

    def on_release_pan(event):
        was_previewing = pan_state['preview_started']
        panned_ax = pan_state['ax']
        pan_state['active'] = False
        pan_state['ax'] = None
        pan_state['last_pixel'] = None
        pan_state['preview_started'] = False
        if was_previewing:
            # Straight to the real redraw, with none of scrolling's debounce
            # wait: a mouse release is an unambiguous end to the gesture,
            # unlike a gap between scroll notches that may or may not mean
            # the user is finished. zoom_settle_source captured *before*
            # pan_state['ax'] was cleared above — panned_ax is the UMAP
            # Axes for a UMAP drag, or a section panel's for one of those
            # (untouched by ZOOM_BITMAP_ONLY_MAX_MULTIPLIER for now, same as
            # a section scroll-zoom — see zoom_settle_source's own comment).
            zoom_settle_source['ax'] = ax if panned_ax is ax else 'panel'
            end_zoom_previews()

    fig.canvas.mpl_connect('button_press_event', on_press_pan)
    fig.canvas.mpl_connect('motion_notify_event', on_motion_pan)
    fig.canvas.mpl_connect('button_release_event', on_release_pan)

    # --- Hover cross-linking: UMAP -> section grid + status text --------
    try:
        tk_widget = fig.canvas.get_tk_widget()
    except Exception:
        tk_widget = None

    # Hold times (VIEWER_HOVER_HOLD_MS, GROUP_HOVER_HOLD_MS) and the status-
    # line text (HOVER_DEFAULT_MESSAGE, HOVER_FIELD_SEP) are at the top of
    # the file. VIEWER_HOVER_HOLD_MS is prefixed because the ROI picker has
    # its own, longer HOVER_HOLD_MS that this must not be merged with. The
    # second, longer GROUP_HOVER_HOLD_MS exists so that briefly passing over
    # a cell shows *that* cell without also lighting up its whole family:
    # only stopping on it for a while does that, so a fast pass across the
    # plot doesn't flash a different subclass on every cell it crosses.
    hover_state = {'timer_id': None, 'group_timer_id': None, 'group_key': None}

    # Cached snapshot of the whole canvas (every section panel's own
    # per-point-colored scatter, the UMAP scatter, sidebar — everything
    # except the animated hover overlays below) — re-snapshotted after
    # every full draw via matplotlib's own 'draw_event' hook, so zooming,
    # panning, and switching Color-by mode/level all keep it current
    # automatically (same pattern prompt_subregion_selection's own
    # blit_hover_overlays uses). Hovering only ever needs to move a ring or
    # add/remove a handful of '+' markers, so restoring this cached bitmap
    # and stamping just those on top (blit_hover_overlays) skips
    # re-rendering every section panel's own scatter on every hover tick —
    # that redundant full redraw, not anything about hovering itself, was
    # what made this window feel excruciatingly slow while moving the
    # mouse around, since it has one Axes (and one expensive scatter) per
    # brain section rather than just one.
    blit_bg = {'data': None}

    def draw_animated_overlays():
        # Skips get_visible()==False artists in Python rather than letting
        # fig.draw_artist() discover that internally — with one panel per
        # brain section, this loop runs over dozens of them on *every* blit
        # (a dropdown open, a mode click, ...), and almost all of their
        # per-panel highlight/group_highlight are invisible/absent at any
        # given moment; skipping the call entirely avoids paying its
        # (Python + Agg) overhead dozens of times over for artists that
        # would draw nothing anyway.
        # Group (family '+' markers) before the single-cell highlight, not
        # after — zorder only orders artists within a single Axes draw, and
        # these are stamped one at a time here, so call order is what
        # actually decides what ends up on top. Drawing the single cell
        # *last* is what makes it sit visibly above the family markers
        # instead of getting painted over by one — which, for section
        # panels specifically, used to happen literally every time: the
        # section-panel group highlight (further down) doesn't exclude the
        # hovered cell's own position the way the UMAP one does (see
        # show_group_highlight_for_cell's exclude_idx), so that panel's own
        # '+' for the hovered cell's own family sits at the *exact* same
        # spot as its highlight dot — previously drawn first, then fully
        # painted over.
        # UMAP veil first, so the '+' markers and ring are stamped over it.
        if umap_highlight_state.get('veil') is not None and umap_highlight_state['veil'].get_visible():
            fig.draw_artist(umap_highlight_state['veil'])
        if group_highlight_state['artist'] is not None:
            fig.draw_artist(group_highlight_state['artist'])
        if umap_highlight_state['artist'] is not None and umap_highlight_state['artist'].get_visible():
            fig.draw_artist(umap_highlight_state['artist'])
        fig.draw_artist(hover_status_text)
        for panel in section_panels.values():
            if panel['dim_veil'].get_visible():
                fig.draw_artist(panel['dim_veil'])
            if panel['group_highlight'] is not None:
                fig.draw_artist(panel['group_highlight'])
            if panel['highlight'].get_visible():
                fig.draw_artist(panel['highlight'])
        # Always on top of whatever the scalebar's own panel is currently
        # showing — real scatter, zoom-preview bitmap, or cached home-view
        # image — since it's animated (see build_section_scalebar's own
        # animated=True docstring) and excluded from all of those; this is
        # the one place it's ever actually drawn.
        if section_scalebar is not None:
            fig.draw_artist(section_scalebar['line'])
            fig.draw_artist(section_scalebar['text'])
        # query_ax/gene_dropdown_ax/level_dropdown_ax are defined further
        # down (query/level box setup) — fine, same forward-reference-via-
        # closure reasoning as everything else here: this function isn't
        # actually called until later, once they already exist.
        fig.draw_artist(query_ax)
        fig.draw_artist(query_label)
        fig.draw_artist(gene_dropdown_ax)
        fig.draw_artist(level_dropdown_ax)
        fig.draw_artist(status_text)
        fig.draw_artist(working_text)
        # color_by_label/level_label ("Color by:"/"Level:") aren't touched
        # by anything else that would redraw them — drawn here purely so
        # set_sidebar_controls_busy's dimming of these two (alongside
        # query_label, already above) shows up on a blit-only path too.
        fig.draw_artist(color_by_label)
        fig.draw_artist(level_label)
        # radio_ax/level_selector_ax: their own spines/text aren't animated
        # (only mode_dots and level_dropdown_ax are — see those artists'
        # own comments), so a *real* full draw always renders them
        # correctly on its own; drawn here too only so set_sidebar_
        # controls_busy's text/border dimming (below) is visible on a
        # blit-only path as well, same reasoning as the buttons just below.
        # radio_ax *must* be drawn before mode_dots, not after: mode_dots
        # is animated, so radio_ax's own draw() call skips it (same as any
        # other animated artist) and instead redraws radio_ax's own
        # background patch — which, drawn *after* mode_dots, painted
        # straight over the already-drawn dots and made them disappear
        # entirely, functional but invisible.
        fig.draw_artist(radio_ax)
        fig.draw_artist(mode_dots)
        fig.draw_artist(level_selector_ax)
        # show_button/save_button/export_degs_button/close_button (defined
        # further down, same forward-reference reasoning as query_ax etc.
        # above): re-stamped here so set_sidebar_controls_busy's dimmed/
        # normal facecolor is what every blit_hover_overlays()/blit_
        # sidebar_overlays() call actually shows, instead of whatever
        # stale color happened to be in the cached background snapshot
        # (restore_region's own restore only reaches back to the last real
        # draw — this is what makes anything *after* that draw visible
        # without needing a whole new one).
        for button in BUSY_GATED_BUTTONS:
            fig.draw_artist(button.ax)

    def axes_has_plotted_content(target_ax):
        """True if `target_ax` currently has at least one visible scatter
        with points in it — i.e. a draw right now would actually render
        this axes' real content.

        Guards the home-view cache against being filled from a draw that
        renders an *empty* axes, which produced a cache that was nothing
        but the axes border. Two ways that happened: the UMAP scatter takes
        a moment to build, so the earliest draws after the window appears
        show the axes already sitting at its home limits (set before the
        scatter is added) with nothing in it yet; and mid-zoom-burst draws
        render with every real artist deliberately hidden (see
        begin_zoom_preview), which looks equally empty. Section panels never
        hit the first case — their background scatter exists from
        panel-build time — which is why only the UMAP showed a blank
        preview."""
        for artist in target_ax.get_children():
            get_visible = getattr(artist, 'get_visible', None)
            get_offsets = getattr(artist, 'get_offsets', None)
            if get_visible is None or get_offsets is None or not get_visible():
                continue
            try:
                if len(get_offsets()) > 0:
                    return True
            except Exception:
                continue
        return False

    def capture_home_view_if_at_home(target_ax):
        # Opportunistic — captures only when this axes happens to already be
        # sitting at its own home extent on a real draw (the common case
        # right after any mode/color redraw, since those reset to home
        # first). Never forces an extra render just to keep the cache warm.
        if active_zoom_previews:
            # Every real artist is hidden behind a stand-in bitmap right
            # now — nothing worth caching, and cheaper to bail here than to
            # let axes_has_plotted_content walk the children to find that
            # out for each axes in turn.
            return
        if target_ax in home_view_cache:
            # Lazy/once, not refreshed on every draw. The copy below is a
            # real full-resolution bitmap copy per axes (the UMAP's alone is
            # ~9MB) — re-taking it on *every* draw meant that any burst
            # touching some *other* axes still paid to re-copy every axes
            # that happened to be sitting at home throughout (zooming a
            # section panel re-copied the whole never-moving UMAP on every
            # burst), which is what made section-panel zoom slow and jerky.
            # Explicitly invalidated instead, wherever content actually
            # changes — see invalidate_home_view_cache's callers.
            return
        home = pannable_axes.get(target_ax)
        if home is None:
            return
        xlim, ylim = target_ax.get_xlim(), target_ax.get_ylim()
        # Compared as sorted pairs, not elementwise: section panels are
        # y-inverted (sec_ax.invert_yaxis(), right after their home_ylim is
        # recorded), so a panel sitting exactly at home reports get_ylim()
        # as home_ylim *reversed*. An elementwise comparison therefore never
        # matched for any section panel — the cache silently stayed empty
        # for all of them, which is why zoom-out kept falling back to the
        # plain current-view snapshot ("stacked boxes") no matter what,
        # while the UMAP (never inverted) worked fine.
        if not (np.allclose(sorted(xlim), sorted(home['home_xlim']))
                and np.allclose(sorted(ylim), sorted(home['home_ylim']))):
            return
        if not axes_has_plotted_content(target_ax):
            return  # nothing rendered yet — see that function
        captured = snapshot_axes_region(target_ax)
        if captured is None:
            return
        snapshot, extent = captured
        home_view_cache[target_ax] = {'buf': snapshot, 'extent': extent}

    def invalidate_home_view_cache(target_ax=None):
        """Drops the cached home-extent bitmap for `target_ax` (or every
        axes, if None) — call whenever what that axes *renders* changes
        (recolor, mode/level switch), since the cached bitmap is only a
        valid stand-in for content that still looks the same."""
        if target_ax is None:
            home_view_cache.clear()
        else:
            home_view_cache.pop(target_ax, None)

    # Declared here, ahead of cache_blit_background below (which reads it on
    # every draw event), rather than down beside enter/exit_resize_fast_mode
    # where the rest of that feature lives — a draw firing before the
    # assignment would otherwise be a NameError on the closure lookup.
    resize_fast_mode = {'active': False, 'saved': []}

    def cache_blit_background(event=None):
        # 'draw_event' — this function's own trigger — fires from more than
        # just real interactive draws: save_current_umap's own fig.savefig()
        # calls also draw the figure internally (to actually render the
        # output file), and this hook doesn't distinguish those from a
        # normal one. Two distinct problems came from that, both traced by
        # hand rather than guessed:
        #
        #  1. fig.savefig(..., bbox_inches=<a Bbox>) — used to crop the
        #     saved file to just the UMAP panel — works by temporarily
        #     *shrinking the live figure itself* down to that crop's own
        #     size (confirmed directly: fig.get_size_inches() measured
        #     inside this very callback, mid-save, briefly reports the
        #     crop's dimensions instead of the window's real ones) for the
        #     duration of that one render. For PNG, nothing here fails, so
        #     this ran to completion and blitted that shrunk, UMAP-only
        #     render straight onto the *live* Tk canvas — which is exactly
        #     what "the UMAP suddenly fills the whole window, sidebar and
        #     all" was: a real render of the tiny cropped figure, stretched
        #     across the window's actual (unchanged) on-screen size.
        #  2. For SVG specifically, matplotlib also swaps fig.canvas to a
        #     non-Agg FigureCanvasSVG for that one render (needed to
        #     produce that format at all) — which has no copy_from_bbox,
        #     turning the very first line below into a hard crash:
        #     'FigureCanvasSVG' object has no attribute 'copy_from_bbox'.
        #
        # fig.canvas._is_saving is set by matplotlib itself for exactly
        # this window (true for every format, not just SVG) — skipping
        # entirely whenever it's set sidesteps both: there's nothing here
        # worth doing for a save's own internal render anyway, since none
        # of it is what ends up in the saved file, only what's cached for
        # the *next real* interactive blit.
        if getattr(fig.canvas, '_is_saving', False):
            return
        # Mid-resize, everything below is both wasted and expensive: the
        # full-figure copy_from_bbox plus a per-axes home_view_cache
        # snapshot (the UMAP's alone is ~9MB) would run on every one of the
        # many draws a live drag produces, and none of those intermediate
        # sizes is worth caching — the drag's final size is. exit_resize_
        # fast_mode's own trailing draw repopulates all of it once.
        if resize_fast_mode['active']:
            return
        blit_bg['data'] = fig.canvas.copy_from_bbox(fig.bbox)
        # Home-view snapshots taken *before* the overlays are stamped on,
        # not after: these read back the rendered buffer, and this cache is
        # a stand-in for an axes' own plotted content during a zoom-out
        # preview — so it has to hold the clean full-draw output. Captured
        # after draw_animated_overlays() below, whatever happened to be
        # showing at the time (a hover ring, the family '+' markers, or the
        # section panels' dim_veil) got baked into the bitmap and would
        # reappear, frozen, in every later preview that used it.
        if ZOOM_DEBUG_DIAGNOSTICS:
            _cbb_t0 = time.perf_counter()
            _cbb_misses = sum(1 for a in [ax] + [p['ax'] for p in section_panels.values()]
                               if a not in home_view_cache)
        capture_home_view_if_at_home(ax)
        for panel in section_panels.values():
            capture_home_view_if_at_home(panel['ax'])
        if ZOOM_DEBUG_DIAGNOSTICS:
            _cbb_elapsed = time.perf_counter() - _cbb_t0
            if _cbb_elapsed > 0.05:
                print(f"[cache-blit] capture_home_view_if_at_home over {1 + len(section_panels)} axes "
                      f"({_cbb_misses} not yet cached) took {_cbb_elapsed:.3f}s")
        # Stamp the overlays back on immediately — being animated means a
        # normal full draw skips them, so without this they'd flash away
        # (only reappearing on the next hover tick) right after every zoom,
        # pan, or mode/level switch.
        draw_animated_overlays()
        fig.canvas.blit(fig.bbox)

    fig.canvas.mpl_connect('draw_event', cache_blit_background)

    def force_repaint():
        # fig.canvas.blit() (and fig.canvas.draw(), which also ends with
        # its own internal blit) only updates the Tk PhotoImage's pixel
        # *data* — actually compositing that to the visible window is
        # handled by Tk's own idle/redraw queue, which needs the event
        # loop serviced to run at all. Without forcing that now, a blit
        # immediately followed by more synchronous work (as every caller
        # here does) just sat queued until *something* eventually serviced
        # Tk's event loop again — which could be as late as the next real
        # user input, reading as "the update didn't happen until I moved
        # the mouse". update_idletasks() (not update()) forces the pending
        # repaint through without risking processing a new user input
        # event (another scroll tick, say) reentrantly mid-handler.
        if tk_widget is not None:
            tk_widget.update_idletasks()

    def blit_hover_overlays():
        # Never paint from blit_bg mid zoom-burst. Only a burst's first tick
        # does a real draw (the sole thing that fires draw_event ->
        # cache_blit_background); every tick after goes through
        # blit_zoomed_axes, which draws and blits without a draw_event, so
        # blit_bg stays pinned at tick 1 — very nearly the pre-zoom view —
        # for the whole ZOOM_PREVIEW_SETTLE_MS settle window. Any
        # restore_region from it in that window puts that stale frame on
        # screen, which is the zoom briefly appearing to revert itself.
        # That settle debounce (800ms) is far longer than the hover hold
        # (VIEWER_HOVER_HOLD_MS), so a mouse move mid-burst reliably lands a
        # hover settle inside it: whether it resolves to a cell
        # (show_highlight_for_cell), to nothing (hide_all_highlights' own
        # default blit=True), or to the cursor leaving the plot area
        # entirely (on_hover_region_leave's direct call here), every path
        # ends up in this function. Skipping outright is correct, not merely
        # safe — the preview images own the screen for the duration, and
        # hover highlights are already deliberately suppressed while zooming
        # (see suspend_hover_during_zoom).
        if active_zoom_previews:
            return
        if blit_bg['data'] is not None:
            fig.canvas.restore_region(blit_bg['data'])
            draw_animated_overlays()
            fig.canvas.blit(fig.bbox)
            force_repaint()
        else:
            fig.canvas.draw_idle()

    def figure_fraction_bbox(x0, y0, x1, y1):
        """A pixel-space Bbox for fig.canvas.blit(), from a rectangle given
        in figure-fraction coordinates."""
        px0, py0 = fig.transFigure.transform((x0, y0))
        px1, py1 = fig.transFigure.transform((x1, y1))
        return Bbox.from_extents(min(px0, px1), min(py0, py1), max(px0, px1), max(py0, py1))

    def blit_sidebar_overlays():
        """Same as blit_hover_overlays, but only actually pushes the
        sidebar's own pixels (mode radio, level selector/dropdown, query
        box, gene dropdown) to the screen, via fig.canvas.blit()'s own bbox
        parameter — restore_region/draw_artist still touch the full cached
        Agg buffer in memory either way (that part's fast: an in-RAM
        buffer copy), but the *screen* update on TkAgg goes through Tk's
        own PhotoImage transfer, whose cost scales with how many pixels
        get pushed, not with how little actually changed. For a sidebar-
        only interaction (a dropdown opening, a mode dot flipping), that
        transfer was, empirically, most of the remaining "still feels
        slow" cost, since blit_hover_overlays() always pushed the *entire*
        canvas regardless of what changed — this canvas has one Axes per
        brain section plus the full UMAP scatter, so 'entire canvas' is a
        lot of pixels even when none of them actually differ from what's
        already on screen. Only valid for interactions that are provably
        confined to the sidebar column — anything that might also touch
        status_text (outside the sidebar, in the grid/UMAP status strip)
        or a section panel/UMAP highlight still needs blit_hover_overlays's
        full-canvas version."""
        if blit_bg['data'] is None:
            fig.canvas.draw_idle()
            return
        fig.canvas.restore_region(blit_bg['data'])
        draw_animated_overlays()
        fig.canvas.blit(figure_fraction_bbox(SIDEBAR_LEFT, 0.0, SIDEBAR_RIGHT, 1.0))
        force_repaint()

    def hide_group_highlight():
        if group_highlight_state['artist'] is not None:
            group_highlight_state['artist'].remove()
            group_highlight_state['artist'] = None
        for panel in section_panels.values():
            if panel['group_highlight'] is not None:
                panel['group_highlight'].remove()
                panel['group_highlight'] = None

    def set_section_dimming(on):
        """Half-brightness veil over every section panel while a hover
        highlight is engaged — see the dim_veil artists' own comment at
        panel-build time for why this is an overlay rather than an alpha
        change on the background scatter."""
        for panel in section_panels.values():
            panel['dim_veil'].set_visible(on)
        # The UMAP gets the same treatment, but toward its own background:
        # lighter on a light background (the default white), darker on a
        # dark one (e.g. multi-gene mode's dark grey), so the dots fade into
        # the background either way and the red markers stand out.
        veil = umap_highlight_state.get('veil')
        if veil is not None:
            if on:
                r, g, b, _a = mcolors.to_rgba(ax.get_facecolor())
                is_light = 0.299 * r + 0.587 * g + 0.114 * b >= 0.5
                veil.set_facecolor('white' if is_light else 'black')
            veil.set_visible(on)

    def hide_all_highlights(blit=True):
        if umap_highlight_state['artist'] is not None:
            umap_highlight_state['artist'].set_visible(False)
        hide_group_highlight()
        for panel in section_panels.values():
            panel['highlight'].set_visible(False)
        set_section_dimming(False)
        hover_status_text.set_text(HOVER_DEFAULT_MESSAGE)
        if blit:
            blit_hover_overlays()

    # Style for a single-cell highlight in whichever panel *type* is
    # actually being hovered (the cursor's right there already, so the
    # original hollow red ring is easy enough to find) versus the *other*
    # panel type, showing that same cell's mirrored position with no cursor
    # to help — that one used to get the same hollow red ring, which could
    # vanish among the same-colored '+' family markers once those appear.
    # Given a solid white fill instead (still with a thin dark edge, for
    # contrast against a pale background) so it reads as a distinct "you are
    # here" marker rather than one more red mark. See build_hover_info_from_
    # umap's 'source' field for how each hover-info dict records which panel
    # type it actually came from.
    HOVER_SOURCE_RING_STYLE = {'facecolor': 'none', 'edgecolor': 'red'}
    HOVER_MIRROR_DOT_STYLE = {'facecolor': 'white', 'edgecolor': 'black'}

    def show_highlight_for_cell(info):
        """`info` is a hover-target dict — see build_hover_info_from_umap/
        build_hover_info_from_section — not a bare UMAP index, since a
        section-panel hover can resolve to a background cell with no UMAP
        counterpart at all (info['umap_idx'] is None in that case)."""
        idx = info['umap_idx']
        umap_style = HOVER_SOURCE_RING_STYLE if info.get('source') == 'umap' else HOVER_MIRROR_DOT_STYLE
        section_style = HOVER_SOURCE_RING_STYLE if info.get('source') == 'section' else HOVER_MIRROR_DOT_STYLE
        if umap_highlight_state['artist'] is not None:
            if idx is not None:
                artist = umap_highlight_state['artist']
                artist.set_offsets([coords[idx]])
                artist.set_facecolor(umap_style['facecolor'])
                artist.set_edgecolor(umap_style['edgecolor'])
                artist.set_visible(True)
            else:
                # No UMAP position to ring — this cell isn't part of this
                # run's `adata`/embedding at all.
                umap_highlight_state['artist'].set_visible(False)
        # Does *not* touch the family highlight — this runs on every
        # settle, including repeat settles on the very same cell (tiny
        # jitter within its hit-radius still cancels/reschedules the
        # single-cell timer, which fires again for the same cell); hiding
        # the family highlight unconditionally here used to flicker it off
        # on every one of those, even though the hovered cell never
        # actually changed. on_any_hover_settled (the only caller) now owns
        # deciding whether the cell actually changed and hiding/restarting
        # the family highlight accordingly — see its own comment.
        sec, x, y = info['sec'], info['x'], info['y']
        cell_has_xy = x is not None and y is not None and not (np.isnan(x) or np.isnan(y))
        for s, panel in section_panels.items():
            if s == sec and cell_has_xy:
                highlight = panel['highlight']
                highlight.set_offsets([[x, y]])
                highlight.set_facecolor(section_style['facecolor'])
                highlight.set_edgecolor(section_style['edgecolor'])
                highlight.set_visible(True)
            else:
                panel['highlight'].set_visible(False)
        # Engaged on *every* highlight, not only the ones that put a ring in
        # some panel — the family '+' markers that follow (and, for a cell
        # with no spatial coords, the UMAP ring alone) are just as much a
        # highlight being engaged, and the veil is what makes them legible.
        # Turned back off in hide_all_highlights, the single path every
        # disengage goes through.
        set_section_dimming(True)
        if idx is not None:
            info_parts = []
            if cell_class is not None:
                info_parts.append(f"class: {cell_class[idx]}")
            if cell_subclass is not None:
                info_parts.append(f"subclass: {cell_subclass[idx]}")
            if cell_supertype is not None:
                info_parts.append(f"supertype: {cell_supertype[idx]}")
            if cell_cluster is not None:
                info_parts.append(f"cluster: {cell_cluster[idx]}")
            # After the four ABC taxonomy levels, since it's a different
            # kind of thing — this run's own computed clustering rather than
            # a published annotation.
            if cell_leiden is not None:
                info_parts.append(f"leiden: {cell_leiden[idx]}")
            hover_status_text.set_text(
                HOVER_FIELD_SEP.join(info_parts) if info_parts else 'No metadata available for this cell.'
            )
        else:
            # Background-only cell (from adata_backed, not this run's own
            # `adata`) — same four-level summary, pulled from the section
            # panel's own precomputed per-level arrays instead of the
            # cell_class/cell_subclass/etc. arrays (those are only aligned
            # with adata's own cells), plus an explicit note that it has no
            # UMAP counterpart to highlight.
            panel = section_panels.get(sec)
            local_idx = info.get('panel_local_idx')
            info_parts = []
            if panel is not None and local_idx is not None:
                for level in LEVEL_OPTIONS:
                    vals = panel['level_values'].get(level)
                    if vals is None:
                        continue
                    value = vals[local_idx]
                    # Skipped rather than printed as "leiden: None": this is
                    # a background-only cell, and Leiden exists only for
                    # cells that were part of this run, so it's genuinely
                    # absent here rather than unknown.
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        continue
                    info_parts.append(f"{level}: {value}")
            text = HOVER_FIELD_SEP.join(info_parts) if info_parts else 'No metadata available for this cell.'
            hover_status_text.set_text(f"{text}{HOVER_FIELD_SEP}(not in UMAP)")
        blit_hover_overlays()

    def show_group_highlight_for_cell(target_value, exclude_idx=None):
        """`target_value` is the raw category string at the active level
        (level_state['value']) — passed in directly rather than an index,
        since the "family" being highlighted doesn't require the hovered
        cell itself to have a UMAP counterpart (see on_group_hover_settled:
        a background-only cell's own category can still match plenty of
        cells that *are* in the UMAP). `exclude_idx`, when given, is the
        hovered cell's own UMAP row — skipped so it keeps its solid ring
        instead of also getting a '+' drawn on top of it."""
        hide_group_highlight()  # drop the previous cell's family highlight, if any, before drawing a new one
        if target_value is None or pd.isna(target_value):
            blit_hover_overlays()
            return
        level_cells = cell_level_arrays[level_state['value']]
        if level_cells is None:
            mask = None
            group_in_umap = False
        else:
            mask = level_cells == target_value
            # Checked *before* excluding the hovered cell's own row — a
            # group whose only UMAP member is the hovered cell itself still
            # counts as "in the UMAP" for section-panel purposes below,
            # even though exclude_idx will zero it out of `mask` right after.
            group_in_umap = bool(mask.any())
            if exclude_idx is not None:
                mask[exclude_idx] = False  # that one cell already has its own ring — no need to double-mark it
        if mask is not None and mask.any():
            # Created fresh with real data each call, not via set_offsets()
            # on a pre-allocated empty scatter (recreate_group_highlight
            # used to make one upfront) — that path never actually rendered
            # anything after the offsets were updated, which was the root
            # cause of the group highlight silently not showing up at all.
            # A '+' marker (not a filled dot) so the cell's own UMAP color
            # still shows through underneath — the single hovered cell
            # keeps its own solid-red hollow ring (umap_highlight_state),
            # unchanged; this is only for the rest of its subclass.
            group_highlight_state['artist'] = ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=umap_group_marker_size(umap_zoom_diameter_multiplier()),
                marker='+', c='red', linewidths=1.5, zorder=5.5,
            )
            # Animated, like every other hover overlay — see recreate_umap_
            # highlight's comment. Set fresh each call since this artist is
            # recreated (not reused) every time the highlighted group changes.
            group_highlight_state['artist'].set_animated(True)
        # Section panels: same level value, matched against each panel's own
        # precomputed per-cell labels (level_values, built at panel-build
        # time from whichever spatial source that panel's gray background
        # dots used) rather than re-masking the whole-brain arrays here.
        # These are already section-filtered and NaN-filtered, and share one
        # index space with the panel's own hover_x/hover_y — a *different*
        # space than `mask` above (which is over adata's own cells only,
        # aligned with `coords`), hence its own comparison per panel.
        #
        # Using level_values (not background_level_arrays) is also what makes
        # this work for Leiden: that level has no whole-brain array at all —
        # it's computed from this run's own neighbor graph, so cells outside
        # `adata` simply have no value — and reading the missing array
        # skipped the section highlight entirely for it, while the panels
        # were meanwhile coloring by Leiden perfectly well from exactly
        # these precomputed values.
        #
        # Gated on group_in_umap: if none of this group's cells made it
        # into this run's UMAP at all, skip the section-panel highlight too
        # — otherwise hovering a background-only cell (adata_backed) would
        # light up its whole cluster across every section even though
        # nothing about that cluster is actually visible/selectable in the
        # UMAP, which reads as a highlight for a cluster the UMAP has no
        # relationship to. The status text (built elsewhere) still reports
        # that cell's info either way.
        if group_in_umap and target_value is not None:
            level = level_state['value']
            for panel in section_panels.values():
                vals = panel['level_values'].get(level)
                if vals is None:
                    continue
                sec_mask = vals == target_value
                if not sec_mask.any():
                    continue
                panel['group_highlight'] = panel['ax'].scatter(
                    panel['hover_x'][sec_mask], panel['hover_y'][sec_mask],
                    s=SECTION_GROUP_BASE_SIZE * section_zoom_diameter_multiplier() ** 2,
                    marker='+', c='red', linewidths=1.5, zorder=4.5,
                )
                panel['group_highlight'].set_animated(True)
        blit_hover_overlays()

    def nearest_umap_cell(data_x, data_y):
        """Index of the UMAP point closest to (data_x, data_y), or None if
        nothing's within ~2% of the current view span (scales with zoom
        level, so this stays a "hovering directly over a point" threshold
        whether zoomed all the way out or in)."""
        if data_x is None or data_y is None:
            return None
        dist2 = (coords[:, 0] - data_x) ** 2 + (coords[:, 1] - data_y) ** 2
        idx = int(np.argmin(dist2))
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        view_span = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]))
        threshold2 = (view_span * 0.02) ** 2
        return idx if dist2[idx] <= threshold2 else None

    def resolve_section_hover_target(sec, data_x, data_y):
        """Local index (into that panel's own hover_x/y/ids/level_values
        arrays) of the point closest to (data_x, data_y) within section
        `sec`'s own panel, or None if nothing's close enough (same ~2%-of-
        that-panel's-own-view threshold as nearest_umap_cell)."""
        panel = section_panels.get(sec)
        if panel is None or data_x is None or data_y is None:
            return None
        xs, ys = panel['hover_x'], panel['hover_y']
        if len(xs) == 0:
            return None
        dist2 = (xs - data_x) ** 2 + (ys - data_y) ** 2
        nearest = int(np.argmin(dist2))
        xlim, ylim = panel['ax'].get_xlim(), panel['ax'].get_ylim()
        view_span = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]))
        threshold2 = (view_span * 0.02) ** 2
        if dist2[nearest] > threshold2:
            return None
        return nearest

    def build_hover_info_from_umap(idx, source='umap'):
        """`source` records which panel *type* this hover actually
        originated from ('umap' or 'section') — build_hover_info_from_section
        below reuses this same builder (when the section-hovered cell has a
        UMAP counterpart) but passes source='section', since the cell's own
        UMAP index doesn't tell you which panel the mouse is actually over.
        show_highlight_for_cell uses this to tell the panel you're pointing
        at (whose highlight sits right under the cursor, one style) apart
        from the *other* panel type showing that same cell's mirrored
        position (a different style, since there's no cursor there to help
        it stand out)."""
        return {
            'cell_id': adata.obs_names[idx], 'umap_idx': idx,
            'sec': cell_section[idx] if cell_section is not None else None,
            'x': cell_x[idx] if cell_x is not None else None,
            'y': cell_y[idx] if cell_y is not None else None,
            'source': source,
        }

    def build_hover_info_from_section(sec, local_idx):
        """Builds the same hover-target shape as build_hover_info_from_umap
        — the section panel's own background dots can include cells from
        adata_backed that never made it into this run's `adata`/UMAP (see
        cell_id_to_umap_idx's own comment); when that's the case, 'umap_idx'
        comes back None and 'panel_local_idx' is kept around so callers can
        still pull that cell's own class/subclass/supertype/cluster values
        straight from the panel's precomputed level_values."""
        panel = section_panels[sec]
        cid = panel['hover_ids'][local_idx]
        umap_idx = cell_id_to_umap_idx.get(cid)
        if umap_idx is not None:
            return build_hover_info_from_umap(umap_idx, source='section')
        return {
            'cell_id': cid, 'umap_idx': None, 'sec': sec,
            'x': panel['hover_x'][local_idx], 'y': panel['hover_y'][local_idx],
            'panel_local_idx': local_idx,
            'source': 'section',
        }

    def on_any_hover_settled(info):
        """Shared by hover-settle from the UMAP scatter *and* every section
        panel (see on_umap_hover_settled/on_section_hover_settled's own
        thin wrappers, which only differ in how they resolve `info`) — one
        state machine either way, so switching which one you're hovering
        (or hovering nothing) behaves identically regardless of source,
        including which panel(s) light up and the family-highlight timing/
        de-dup logic below. Keyed by `info['cell_id']` rather than a UMAP
        index — a background-only cell (info['umap_idx'] is None) still has
        a stable identity to key the "did the hovered cell actually change"
        check on."""
        hover_state['timer_id'] = None
        if info is None:
            hide_all_highlights()
            if hover_state['group_timer_id'] is not None:
                tk_widget.after_cancel(hover_state['group_timer_id'])
                hover_state['group_timer_id'] = None
            hover_state['group_key'] = None
            return
        # The family highlight is only ever hidden/restarted here, on an
        # *actual* cell change — never inside show_highlight_for_cell
        # itself, which also runs on repeat settles for the very same cell
        # (jitter). Without this being the single place that decides "did
        # the cell actually change", hide-on-every-settle and show-once-
        # per-new-cell used to disagree, which is what let the family
        # highlight flicker off while the single-cell ring stayed on the
        # same cell the whole time.
        key = info['cell_id']
        if key != hover_state.get('group_key'):
            if hover_state['group_timer_id'] is not None:
                tk_widget.after_cancel(hover_state['group_timer_id'])
            hide_group_highlight()
            hover_state['group_key'] = key
            # Deferred via tk's own `after` — keyed on the *resolved* cell
            # rather than raw mouse position/pixel motion, since
            # rescheduling on every motion_notify_event instead (as it
            # originally was) meant the tiniest jitter during the hold
            # (extremely common with a trackpad, and not unheard of even
            # with a mouse) kept cancelling and restarting it before
            # GROUP_HOVER_HOLD_MS ever actually elapsed uninterrupted —
            # which is why it never seemed to fire in practice. Re-hovering
            # the *same* cell (key unchanged, this whole branch skipped)
            # leaves its already-running countdown (or already-shown
            # highlight) alone entirely.
            hover_state['group_timer_id'] = tk_widget.after(
                max(0, GROUP_HOVER_HOLD_MS - VIEWER_HOVER_HOLD_MS), lambda: on_group_hover_settled(info))
        show_highlight_for_cell(info)

    def on_umap_hover_settled(data_pos):
        idx = nearest_umap_cell(*data_pos)
        on_any_hover_settled(build_hover_info_from_umap(idx) if idx is not None else None)

    def on_section_hover_settled(data_pos):
        sec, data_x, data_y = data_pos
        local_idx = resolve_section_hover_target(sec, data_x, data_y)
        on_any_hover_settled(build_hover_info_from_section(sec, local_idx) if local_idx is not None else None)

    def on_group_hover_settled(info):
        hover_state['group_timer_id'] = None
        level = level_state['value']
        idx = info['umap_idx']
        if idx is not None:
            level_cells = cell_level_arrays[level]
            target_value = level_cells[idx] if level_cells is not None else None
            show_group_highlight_for_cell(target_value, exclude_idx=idx)
        else:
            panel = section_panels.get(info['sec'])
            vals = panel['level_values'].get(level) if panel is not None else None
            target_value = vals[info['panel_local_idx']] if vals is not None else None
            show_group_highlight_for_cell(target_value)

    def on_umap_motion(event):
        # Deferred via tk's own `after` (not looked up on every raw motion
        # event) — a nearest-point search over up to ~200k cells is fast
        # enough to run synchronously once the mouse actually settles, but
        # doing it on *every* pixel of movement would still add up and make
        # dragging the mouse across the plot feel sluggish; this is the
        # same "wait for the mouse to stop, then look" pattern prompt_
        # subregion_selection's own hover-to-identify-cell feature uses.
        # Only the single-cell timer is (re)started directly from raw
        # motion — the group-highlight one is managed from within
        # on_any_hover_settled instead, keyed on the resolved cell (see its
        # own comment on why).
        if event.inaxes is not ax or pan_state['active'] or tk_widget is None:
            return
        if hover_state['timer_id'] is not None:
            tk_widget.after_cancel(hover_state['timer_id'])
        data_pos = (event.xdata, event.ydata)
        hover_state['timer_id'] = tk_widget.after(VIEWER_HOVER_HOLD_MS, lambda: on_umap_hover_settled(data_pos))

    # section_axes_to_label is built once, after every panel already
    # exists, so on_section_motion can cheaply tell *which* section a
    # motion event landed in without scanning section_panels.values() on
    # every single mouse move.
    section_axes_to_label = {panel['ax']: sec for sec, panel in section_panels.items()}

    def on_section_motion(event):
        # Same hold-then-look reasoning as on_umap_motion, and shares its
        # own timer slot (hover_state['timer_id']) — only one of the two
        # can ever be "the thing currently hovered" at a time, so moving
        # from the UMAP to a section panel (or between panels) correctly
        # cancels whichever was pending, regardless of which of these two
        # handlers had scheduled it.
        sec = section_axes_to_label.get(event.inaxes)
        if sec is None or pan_state['active'] or tk_widget is None:
            return
        if hover_state['timer_id'] is not None:
            tk_widget.after_cancel(hover_state['timer_id'])
        data_pos = (sec, event.xdata, event.ydata)
        hover_state['timer_id'] = tk_widget.after(VIEWER_HOVER_HOLD_MS, lambda: on_section_hover_settled(data_pos))

    fig.canvas.mpl_connect('motion_notify_event', on_section_motion)

    fig.canvas.mpl_connect('motion_notify_event', on_umap_motion)

    def on_hover_region_leave(event):
        # on_umap_motion/on_section_motion only ever *schedule new* hover
        # work while the cursor is over their own axes — neither one runs
        # at all once the mouse moves off the UMAP/a panel into the
        # sidebar, the grid labels, or any other gap in the window, so
        # nothing was ever noticing that move and clearing whatever was
        # already showing. figure_leave_event (below) only covers the
        # cursor leaving the *whole window* — moving from the plot area
        # into, say, the sidebar without ever crossing the window's own
        # edge left a highlight stuck on screen indefinitely. Runs on every
        # motion tick outside a hover-relevant axes, not just the first,
        # but the early-return below skips the actual work (timer cancel +
        # a full blit) unless there's something to clear, so drifting the
        # mouse across the sidebar doesn't pay for a blit on every tick.
        if event.inaxes is ax or event.inaxes in section_axes_to_label:
            return  # still over a hover-relevant axes — its own motion handler owns this
        if pan_state['active']:
            return  # already suspended at press time — see on_press_pan
        if (hover_state['timer_id'] is None and hover_state['group_timer_id'] is None
                and hover_state['group_key'] is None):
            return  # nothing pending or shown — nothing to clear
        suspend_hover_during_zoom()
        blit_hover_overlays()

    fig.canvas.mpl_connect('motion_notify_event', on_hover_region_leave)

    def on_figure_leave(event):
        # A fast sweep of the mouse across a section/UMAP can settle a
        # hover right as the cursor crosses out of the window entirely —
        # there's no further motion event once it's outside to ever notice
        # that and clear it, so without this the highlight just stays on
        # screen until the mouse re-enters and happens to land on a
        # different cell (or nothing at all). Same cancel-and-hide used at
        # pan-press time, just triggered by leaving instead of clicking.
        suspend_hover_during_zoom()
        blit_hover_overlays()
        # Same reasoning for the resize-handle hover cursor (set_resize_
        # cursor, defined later alongside the handles themselves — a
        # forward reference, fine since this isn't actually called until
        # the user moves the mouse, long after the rest of this window has
        # finished being built): leaving the figure mid-hover over a
        # divider has no further motion event to notice and revert it.
        set_resize_cursor(False)

    fig.canvas.mpl_connect('figure_leave_event', on_figure_leave)

    # --- Status bar: its own panel within `fig`, not a separate window ---
    # A bordered/shaded strip along the bottom, spanning the grid+UMAP
    # width — visually its own panel (a background rectangle behind both
    # lines of text), but still part of the one main window, not a second
    # Tk toplevel to manage/dock/keep in sync. Sized via status_panel_
    # geometry() (inches-based — see its own comment above, by AREA_BOTTOM)
    # rather than flat figure-fractions, and repositioned on every resize
    # (on_figure_resize, near the end of this function) for the same
    # reason — otherwise this box (tightly sized around exactly two lines
    # of fixed-physical-size text, unlike the more generously-padded
    # buttons/radios elsewhere) would visibly overflow into the UMAP axes
    # above it in a shrunk window, or sit in a growing sea of empty space
    # in an enlarged one.
    panel_bottom, panel_height = status_panel_geometry()
    status_panel_ax = fig.add_axes([GRID_LEFT, panel_bottom, UMAP_RIGHT - GRID_LEFT, panel_height])
    status_panel_ax.set_xticks([])
    status_panel_ax.set_yticks([])
    status_panel_ax.set_facecolor('#eeeeee')
    for spine in status_panel_ax.spines.values():
        spine.set_edgecolor('#999999')
        spine.set_linewidth(0.8)

    hover_status_ax = fig.add_axes([
        GRID_LEFT + 0.005, panel_bottom + panel_height * (STATUS_PANEL_LINE_FRAC + STATUS_PANEL_GAP_FRAC),
        UMAP_RIGHT - GRID_LEFT - 0.01, panel_height * STATUS_PANEL_LINE_FRAC,
    ])
    hover_status_ax.axis('off')
    hover_status_text = hover_status_ax.text(0, 0.5, HOVER_DEFAULT_MESSAGE,
                                              fontsize=SIDEBAR_FONTSIZE, va='center', ha='left')
    # Animated — updated via blit_hover_overlays() on nearly every hover
    # settle instead of a full draw_idle(); see blit_bg's own comment above.
    hover_status_text.set_animated(True)

    status_ax = fig.add_axes([
        GRID_LEFT + 0.005, panel_bottom, UMAP_RIGHT - GRID_LEFT - 0.01, panel_height * STATUS_PANEL_LINE_FRAC,
    ])
    status_ax.axis('off')
    status_text = status_ax.text(0, 0.5, '', fontsize=SIDEBAR_FONTSIZE, va='center', ha='left')
    # Animated, drawn via draw_animated_overlays/blit_hover_overlays — some
    # callers (redraw_gene's early-return branches in particular: empty
    # query, gene not found, ...) only ever update this text and touch
    # nothing else, so they blit instead of paying for a full fig.canvas.
    # draw_idle() (every section panel's own scatter re-rendered) just to
    # show a one-line message. Callers that *do* change real content (a
    # new UMAP scatter, recolored section panels) still do a full draw —
    # this text just rides along for free either way, same as hover_status_
    # text already does.
    status_text.set_animated(True)

    LEVEL_DISPLAY_NAMES = {'class': 'Class', 'subclass': 'Subclass', 'supertype': 'Supertype',
                           'cluster': 'Cluster', LEIDEN_KEY: 'Leiden'}
    level_state = {'value': 'subclass'}

    def mode_display_label(option):
        level_name = LEVEL_DISPLAY_NAMES[level_state['value']]
        if option == 'All Subclasses':
            return f'All {level_name}s'
        if option == 'Single Subclass':
            # 'es' for names ending in a sibilant ('Class', 'Subclass' ->
            # Classes/Subclasses, the standard English rule for words ending
            # in s/x/z/ch/sh), plain 's' otherwise (Supertype, Cluster,
            # Leiden) — so this reads correctly for every level without a
            # per-level special case. 'Single Subclass' stays the *internal*
            # mode identifier throughout the rest of this function (mode_
            # state, query_by_mode, ...); only the displayed text changes
            # here, same separation LEVEL_OPTIONS/LEVEL_DISPLAY_NAMES already
            # uses for the level names themselves.
            plural = 'es' if level_name.endswith('s') else 's'
            return f'Specified {level_name}({plural})'
        return option

    MODE_OPTIONS = ('All Subclasses', 'Single Subclass', 'Gene', 'Imputed Gene')
    color_by_label = fig.text(SIDEBAR_LEFT, 0.965, 'Color by:', fontsize=SIDEBAR_FONTSIZE, va='bottom', ha='left')
    radio_ax = fig.add_axes([SIDEBAR_LEFT, 0.80, SIDEBAR_WIDTH, 0.16])
    radio_ax.set_xlim(0, 1)
    radio_ax.set_ylim(0, 1)
    radio_ax.set_xticks([])
    radio_ax.set_yticks([])
    radio_ax.set_facecolor('#dddddd')
    for spine in radio_ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
    # Hand-rolled (not matplotlib's own RadioButtons) for the same reason
    # every other radio in this file is: RadioButtons sizes its dots once,
    # at construction, from whatever rcParams['font.size'] was at that
    # moment — calling label.set_fontsize() afterward (the only way to
    # match UI_BUTTON_FONTSIZE, computed after import) resizes the *text*
    # but leaves the dots at their original, un-scaled size, so the whole
    # widget reads as smaller/inconsistent with the rest of the app's
    # explicitly font-scaled dots elsewhere. Vertical here (unlike this
    # file's other hand-rolled radios, which are horizontal) since that's
    # what actually fits this narrow, tall sidebar.
    mode_dot_x, mode_label_x = 0.14, 0.26
    mode_dot_y = np.linspace(1, 0, len(MODE_OPTIONS) + 2)[1:-1]
    mode_dot_size = SIDEBAR_FONTSIZE ** 2  # same formula as this file's other hand-rolled radios
    mode_dots = radio_ax.scatter(
        [mode_dot_x] * len(MODE_OPTIONS), mode_dot_y, s=[mode_dot_size] * len(MODE_OPTIONS),
        marker='o', edgecolor='black', facecolor=(['tab:blue'] + ['none'] * (len(MODE_OPTIONS) - 1)), zorder=3,
    )
    # Animated, drawn via draw_animated_overlays/blit_hover_overlays —
    # set_mode_dot_visual (below) runs *before* on_mode_change's own
    # redraw(), so without this it always paid for one full fig.canvas.
    # draw_idle() of its own regardless of whether the mode switch that
    # follows turns out to need a real one (e.g. switching to Gene/Imputed
    # Gene with an empty query — see redraw_gene's early returns — doesn't).
    mode_dots.set_animated(True)
    mode_label_texts = [
        radio_ax.text(mode_label_x, y, mode_display_label(opt), fontsize=SIDEBAR_FONTSIZE, va='center', ha='left')
        for y, opt in zip(mode_dot_y, MODE_OPTIONS)
    ]

    def update_mode_labels_for_level():
        for text_obj, opt in zip(mode_label_texts, MODE_OPTIONS):
            text_obj.set_text(mode_display_label(opt))

    def set_mode_dot_visual(idx):
        facecolors = ['none'] * len(MODE_OPTIONS)
        facecolors[idx] = 'tab:blue'
        mode_dots.set_facecolor(facecolors)
        blit_sidebar_overlays()

    # A single shared TextBox, not one-per-mode — two TextBox widgets
    # stacked at the same axes position (one hidden via set_visible(False)
    # while the other's shown) still both get 'button_press_event's for
    # clicks in that region, since matplotlib's own widget hit-testing
    # doesn't check axes visibility. Each one's _click handler tries to
    # grab the mouse for itself, and the second grab attempt on a
    # different (if invisible) Axes is exactly what raised "Another Axes
    # already grabs mouse input". One widget, relabeled per mode, sidesteps
    # this entirely instead of trying to keep two widgets' active states in
    # sync. What's actually been typed per mode is remembered separately
    # (query_by_mode below) so switching modes and back doesn't lose it.
    # Not shown at all for 'All Subclasses' — there's no query for it.
    QUERY_TOP_Y = 0.50
    query_ax = fig.add_axes([SIDEBAR_LEFT, QUERY_TOP_Y, SIDEBAR_WIDTH, 0.04])
    query_textbox = TextBox(query_ax, '', initial='268' if has_subclass else '')
    query_textbox.text_disp.set_fontsize(SIDEBAR_FONTSIZE)
    # Animated, drawn via draw_animated_overlays/blit_hover_overlays, same
    # as gene_dropdown_ax — make_textbox_blit_fast (wired up below) routes
    # every keystroke's cursor/text redraw through blit_hover_overlays()
    # instead of a full draw_idle(); without query_ax itself also being in
    # that blit set, each keystroke's blit restored the *stale* cached
    # background over the box (wiping out whatever was just typed) and then
    # only stamped the *other* animated artists back on top — the textbox's
    # own updated text/cursor never made it into that stamp, so the box
    # looked permanently blank (or stuck on old text) while the separate
    # autocomplete dropdown (already in the blit set) kept updating fine.
    query_ax.set_animated(True)
    query_label = fig.text(SIDEBAR_LEFT, 0.545, 'Subclass ID:', fontsize=SIDEBAR_FONTSIZE, va='bottom', ha='left')
    # Animated, like query_ax — on_mode_change (below) changes its
    # visibility/text every mode switch, but redraw()/redraw_gene() doesn't
    # always follow up with a *real* full draw (e.g. switching to Gene with
    # an empty query is blit-only, see redraw_gene's own comment) — without
    # this, the label's new state sat correct in memory but never actually
    # got painted until some unrelated later interaction forced a full
    # draw, which is why it looked like it "never showed" after picking
    # Gene/Imputed Gene from a blank query.
    query_label.set_animated(True)
    # Keyed by query_state_key() below, not by mode alone: 'Single Subclass'
    # gets a separate slot per taxonomy level, since an ID typed while
    # 'Subclass' is selected means nothing at 'Supertype' — switching levels
    # used to carry the old number across, where it either silently matched
    # some unrelated category or matched nothing at all.
    query_by_mode = {('Single Subclass', 'subclass'): '268' if has_subclass else '',
                     'Gene': '', 'Imputed Gene': ''}

    def query_state_key(mode=None, level=None):
        mode = mode_state['mode'] if mode is None else mode
        if mode != 'Single Subclass':
            return mode
        return (mode, level_state['value'] if level is None else level)

    def set_query_text(value):
        """query_textbox.set_val without triggering its on_submit.

        TextBox.set_val fires both 'change' and 'submit' observers, so a
        plain call here would kick off a full redraw() as a side effect of
        merely *restoring* remembered text — and every caller below goes on
        to redraw deliberately straight afterwards, so that redraw would
        run twice, doubling the most expensive work in this window."""
        set_textbox_text_silent(query_textbox, value)

    def parse_id_query(query):
        """The ID box's comma-separated content as (ids, bad_tokens).

        Accepts '268', '268, 270', '268 270' — commas and/or whitespace —
        keeping the order typed and dropping exact duplicates, so the color
        assigned to each ID below is stable and predictable. Anything that
        isn't an integer comes back in `bad_tokens` for the caller to
        report, rather than silently vanishing."""
        ids, bad = [], []
        for token in re.split(r'[,\s]+', (query or '').strip()):
            if not token:
                continue
            try:
                value = int(token)
            except ValueError:
                bad.append(token)
                continue
            if value not in ids:
                ids.append(value)
        return ids, bad
    query_ax.set_visible(False)  # MODE_OPTIONS[0] ('All Subclasses') has no query box
    query_label.set_visible(False)

    # Level-of-granularity selector (class/subclass/supertype/cluster) —
    # same hand-rolled click-to-open/select dropdown pattern as the gene
    # autocomplete box below, placed just above it. Governs the category
    # column used by 'All Subclasses'/'Single Subclass' coloring and by the
    # family/group highlight shown on hover (show_group_highlight_for_cell).
    # LEVEL_OPTIONS itself is defined earlier, alongside cell_level_arrays.
    LEVEL_TOP_Y = 0.73
    level_label = fig.text(SIDEBAR_LEFT, 0.775, 'Level:', fontsize=SIDEBAR_FONTSIZE, va='bottom', ha='left')
    level_selector_ax = fig.add_axes([SIDEBAR_LEFT, LEVEL_TOP_Y, SIDEBAR_WIDTH, 0.04])
    level_selector_ax.set_xlim(0, 1)
    level_selector_ax.set_ylim(0, 1)
    level_selector_ax.set_xticks([])
    level_selector_ax.set_yticks([])
    level_selector_ax.patch.set_facecolor('white')
    for spine in level_selector_ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    level_selector_text = level_selector_ax.text(
        0.04, 0.5, LEVEL_DISPLAY_NAMES[level_state['value']],
        fontsize=SIDEBAR_FONTSIZE, va='center', ha='left',
    )
    # A drawn triangle marker, not a "▾" text glyph — the glyph isn't in
    # every font matplotlib might fall back to, and renders as a tofu box
    # (a literal square) when missing.
    level_selector_triangle = level_selector_ax.scatter(
        [0.93], [0.5], s=SIDEBAR_FONTSIZE ** 1.5, marker='v', c='black', zorder=3)

    n_levels = len(LEVEL_OPTIONS)
    level_dropdown_line_height = 0.035
    level_dropdown_ax = fig.add_axes([
        SIDEBAR_LEFT, LEVEL_TOP_Y - n_levels * level_dropdown_line_height,
        SIDEBAR_WIDTH, n_levels * level_dropdown_line_height,
    ])
    level_dropdown_ax.set_xlim(0, 1)
    level_dropdown_ax.set_ylim(0, 1)
    level_dropdown_ax.set_xticks([])
    level_dropdown_ax.set_yticks([])
    for spine in level_dropdown_ax.spines.values():
        spine.set_visible(False)
    level_dropdown_ax.patch.set_facecolor('#dddddd')
    level_dropdown_ax.patch.set_edgecolor('black')
    level_dropdown_ax.patch.set_linewidth(1)
    level_dropdown_ax.set_visible(False)
    # Animated, drawn via draw_animated_overlays/blit_hover_overlays, same
    # as gene_dropdown_ax — opening/closing it is a pure visibility toggle,
    # not a content change, so it shouldn't need a full fig.canvas.
    # draw_idle() (which re-renders every section panel's own scatter) just
    # to appear.
    level_dropdown_ax.set_animated(True)
    level_option_texts = [
        level_dropdown_ax.text(0.04, 1 - (i + 0.5) / n_levels, LEVEL_DISPLAY_NAMES[lvl],
                                fontsize=SIDEBAR_FONTSIZE, va='center', ha='left')
        for i, lvl in enumerate(LEVEL_OPTIONS)
    ]

    def on_level_selector_click(event):
        if busy_state['active']:
            return
        # Opening/closing the dropdown, or clicking it without actually
        # changing the level, is a pure visibility toggle — blit_hover_
        # overlays() handles it (level_dropdown_ax is animated, see its own
        # comment above) without the full fig.canvas.draw_idle() every
        # branch here used to do unconditionally, which re-rendered every
        # section panel's own scatter just to show/hide a 4-row dropdown.
        # A *real* level change still needs a full draw further down —
        # level_selector_text's own updated content lives in the (non-
        # animated) cached background, not something blitting alone can
        # refresh.
        if event.inaxes is level_selector_ax:
            level_dropdown_ax.set_visible(not level_dropdown_ax.get_visible())
            blit_sidebar_overlays()
            return
        if not level_dropdown_ax.get_visible():
            return
        if event.inaxes is level_dropdown_ax and event.ydata is not None:
            row = min(n_levels - 1, max(0, int((1 - event.ydata) * n_levels)))
            new_level = LEVEL_OPTIONS[row]
            if new_level != level_state['value']:
                # IDs are per-level, so the box's content belongs to the
                # level being left — stash it there, then bring back
                # whatever was last typed for the level being entered.
                in_id_mode = mode_state['mode'] == 'Single Subclass'
                if in_id_mode:
                    query_by_mode[query_state_key()] = query_textbox.text
                level_state['value'] = new_level
                level_selector_text.set_text(LEVEL_DISPLAY_NAMES[new_level])
                update_mode_labels_for_level()
                if in_id_mode:
                    query_label.set_text(f'{LEVEL_DISPLAY_NAMES[new_level]} ID:')
                    set_query_text(query_by_mode.get(query_state_key(), ''))
                level_dropdown_ax.set_visible(False)
                if mode_state['mode'] in ('All Subclasses', 'Single Subclass'):
                    redraw()  # shows/hides its own working indicator around its own heavy work
                else:
                    show_working_indicator()
                    fig.canvas.draw_idle()
                    hide_working_indicator()
                return
        level_dropdown_ax.set_visible(False)
        blit_sidebar_overlays()

    fig.canvas.mpl_connect('button_press_event', on_level_selector_click)

    # Autocomplete dropdown for the gene box — same click-to-select pattern
    # as the grid/single-section pickers' own gene boxes, including the same
    # fast-blit typing optimization (make_textbox_blit_fast, wired up right
    # after query_textbox is created below) — this window has its own
    # per-section-panel background scatter (one Axes per brain section, each
    # a few-thousand-point collection), so TextBox's own per-keystroke
    # fig.canvas.draw() was, if anything, more costly here than in the
    # other two gene boxes this same fix already covers.
    # Capped at 4 (not the other gene boxes' 6) — this sidebar has less
    # vertical room to spare above the Show/Close buttons below, and the
    # dropdown opens *downward* from the text box here (there's no room
    # above it, unlike the other two gene boxes, which sit at the very
    # bottom of their window) — so a taller dropdown would risk overlapping
    # them instead of just extending harmlessly off the top of the figure.
    gene_max_suggestions = 4
    gene_suggestion_line_height = 0.035
    gene_dropdown_ax = fig.add_axes([SIDEBAR_LEFT, QUERY_TOP_Y - gene_max_suggestions * gene_suggestion_line_height,
                                      SIDEBAR_WIDTH, gene_max_suggestions * gene_suggestion_line_height])
    gene_dropdown_ax.set_xlim(0, 1)
    gene_dropdown_ax.set_ylim(0, 1)
    gene_dropdown_ax.set_xticks([])
    gene_dropdown_ax.set_yticks([])
    for spine in gene_dropdown_ax.spines.values():
        spine.set_visible(False)
    gene_dropdown_ax.patch.set_facecolor('#dddddd')
    gene_dropdown_ax.patch.set_edgecolor('black')
    gene_dropdown_ax.patch.set_linewidth(1)
    gene_dropdown_ax.set_visible(False)
    # Animated, drawn via draw_animated_overlays/blit_hover_overlays
    # (defined above, under "Hover cross-linking") — same reasoning as
    # every other overlay there: updating it on each keystroke shouldn't
    # force a full re-render of every section panel's own scatter.
    gene_dropdown_ax.set_animated(True)
    gene_suggestion_texts = [
        gene_dropdown_ax.text(0.03, 1 - (i + 0.5) / gene_max_suggestions, '',
                               fontsize=SIDEBAR_FONTSIZE, va='center', ha='left')
        for i in range(gene_max_suggestions)
    ]
    gene_suggestion_state = {'matches': []}
    # Set around a programmatic query_textbox.set_val() (picking a gene
    # from the dropdown) so update_gene_suggestions' own 'change'-event
    # handling doesn't reopen the dropdown it was just told to close — see
    # on_gene_dropdown_click's own comment for why a *reactive* "close it
    # again afterward" doesn't work here.
    suppress_autocomplete = {'value': False}

    def update_gene_suggestions(text):
        # Any edit clears the red "not found" names (see the single-/multi-
        # gene redraw functions' "not found" branches); they only come back
        # if the edited text still doesn't resolve when next submitted.
        clear_gene_name_marks(query_textbox)
        # Shared query box (see its own comment above) — autocomplete only
        # makes sense while a gene-based mode is active, not while typing a
        # subclass number.
        if mode_state['mode'] not in ('Gene', 'Imputed Gene'):
            return
        if suppress_autocomplete['value']:
            return
        _prefix, current_token = split_gene_query(text)
        if not current_token:
            gene_dropdown_ax.set_visible(False)
            gene_suggestion_state['matches'] = []
            blit_sidebar_overlays()
            return
        matches = rank_gene_suggestions(
            gene_symbol_list(adata, imputed_state, mode_state['mode'] == 'Imputed Gene'),
            current_token, gene_max_suggestions)
        gene_suggestion_state['matches'] = matches
        n = len(matches)
        if n == 0:
            gene_dropdown_ax.set_visible(False)
            blit_sidebar_overlays()
            return
        gene_dropdown_ax.set_position([SIDEBAR_LEFT, QUERY_TOP_Y - n * gene_suggestion_line_height, SIDEBAR_WIDTH,
                                        n * gene_suggestion_line_height])
        for i, t in enumerate(gene_suggestion_texts):
            if i < n:
                t.set_text(matches[i])
                t.set_position((0.03, 1 - (i + 0.5) / n))
            else:
                t.set_text('')
        gene_dropdown_ax.set_visible(True)
        blit_sidebar_overlays()

    query_textbox.on_text_change(update_gene_suggestions)
    # TextBox's own internal _rendercursor() does a full, synchronous
    # fig.canvas.draw() on *every* keystroke entirely on its own — separate
    # from, and in addition to, whatever on_text_change (above) does. With
    # dozens of section panels each carrying a several-thousand-point
    # scatter, that per-keystroke full draw (not update_gene_suggestions
    # itself, already fixed above) was what made typing gene names feel
    # laggy. Same fix as the grid/single-section pickers' own gene boxes.
    make_textbox_blit_fast(query_textbox, blit_sidebar_overlays)
    # This is the one that actually mattered here: TextBox._click() forces
    # a full, synchronous fig.canvas.draw() on *any* click that lands
    # outside this box (see make_textbox_stop_typing_blit_fast's own
    # docstring) — with one Axes per brain section in this window, that
    # unconditional full render, firing on nearly every click anywhere in
    # the figure, was the actual root cause of "everything feels sluggish
    # regardless of what I click" — entirely inside matplotlib's own
    # widget internals, invisible to and unpreventable by any of the
    # draw_idle()/blit routing done elsewhere in this window.
    make_textbox_stop_typing_blit_fast(query_textbox, blit_sidebar_overlays)
    # This is the one that actually explains the "freezes as soon as I
    # switch to Gene/Imputed Gene/Specified IDs, and it's not about
    # zooming" report: TextBox._motion is connected globally to every
    # mouse move in the whole figure (not scoped to this box's own axes —
    # see make_textbox_motion_blit_fast's own docstring), and forces the
    # same kind of full, synchronous fig.canvas.draw() every time the
    # cursor crosses into or out of the box's own hover region — which,
    # with dozens of real section-panel scatters, is exactly the kind of
    # per-crossing full render that reads as "the UI goes unresponsive for
    # 5-10 seconds and there's no 'Working' message and nothing visibly
    # changed". query_ax.set_visible(False) in every categorical "All
    # <level>s" mode (below) doesn't disable this widget-internal path —
    # _motion's own self.ax.contains(event) check doesn't care whether the
    # axes is visible — it's just that the mouse rarely crosses a hidden
    # box's own (now visually irrelevant) region during normal use of a
    # mode that doesn't need it, whereas the box sits front-and-center in
    # the sidebar, right where the mouse naturally travels, in every mode
    # that does.
    make_textbox_motion_blit_fast(query_textbox, blit_sidebar_overlays)
    enable_textbox_clipboard_shortcuts(query_textbox)

    def on_gene_dropdown_click(event):
        if busy_state['active']:
            return
        if not gene_dropdown_ax.get_visible():
            return
        if event.inaxes is query_ax:
            return
        selected = None
        if event.inaxes is gene_dropdown_ax and event.ydata is not None:
            n = len(gene_suggestion_state['matches'])
            if n > 0:
                row = min(n - 1, max(0, int((1 - event.ydata) * n)))
                selected = gene_suggestion_state['matches'][row]
        gene_dropdown_ax.set_visible(False)
        gene_suggestion_state['matches'] = []
        # Blit the dropdown closing immediately, *before* set_val below can
        # kick off a real gene render via TextBox's own 'submit' event
        # (query_textbox.on_submit(lambda text: redraw()), further down) —
        # picking a gene is a genuinely heavy operation (recolors every
        # section panel), and without this the dropdown stayed visibly open
        # for that entire render, reading as "a delay before it closes"
        # even though the click itself had already registered.
        blit_sidebar_overlays()
        if selected is not None:
            # completed_gene_query keeps names typed before the one being
            # completed (e.g. "Snap25, " while picking the second gene),
            # re-derived from the box's live text rather than anything cached
            # from update_gene_suggestions' last run, so it's always current
            # even if this click follows keystrokes that changed which name is
            # being typed.
            full_value = completed_gene_query(query_textbox.text, selected)
            # set_val() below fires its own 'change' event synchronously,
            # which would otherwise re-run update_gene_suggestions() on the
            # text it just set — and since the selected gene trivially
            # matches itself, that reopened the dropdown right back up.
            # Fixing that *reactively* (closing it again right after
            # set_val()) doesn't actually work: set_val() also fires
            # 'submit', which runs the full heavy gene render synchronously
            # before set_val() itself returns — so "close it again after
            # set_val()" only ever took effect *after* that entire render
            # had already finished, not before. Suppressing the reopen at
            # the source instead means the close blitted just above stays
            # correct the whole time.
            suppress_autocomplete['value'] = True
            try:
                query_textbox.set_val(full_value)
            finally:
                suppress_autocomplete['value'] = False
            # set_val()'s own 'submit' does fire here too, but on_query_
            # submit now deliberately ignores it — query_textbox.
            # capturekeystrokes was already cleared to False by the click
            # that got us into this handler in the first place (TextBox.
            # _click -> stop_typing, before on_gene_dropdown_click ever
            # runs), and on_query_submit's own capturekeystrokes check
            # (added to stop *that* stray, pre-selection submit from
            # closing the dropdown before this function got a chance to
            # read it) can't tell that submit apart from this one — both
            # see the same False. So the actual render has to be triggered
            # explicitly here instead of relying on that event.
            redraw()

    fig.canvas.mpl_connect('button_press_event', on_gene_dropdown_click)

    # Kept clear below the gene dropdown's lowest possible extent
    # (QUERY_TOP_Y - gene_max_suggestions*gene_suggestion_line_height = 0.36)
    # with a real gap, not flush against it.
    # hovercolor deliberately matches Button's own default `color` ('0.85')
    # — Button._motion does a full, *synchronous* fig.canvas.draw() (not
    # even draw_idle()) whenever the mouse enters/leaves its axes, but only
    # if the hover color actually differs from the current one. With dozens
    # of expensive section-panel scatters in this same figure, that single
    # call was a real, noticeable stall right as the mouse approached either
    # button — matching hovercolor to color means the check never sees a
    # change, so that redraw never fires, at the cost of losing the (purely
    # cosmetic) hover highlight.
    # 'Refresh', not 'Show': every control here now redraws the moment it
    # changes, so this is only needed to force a re-render if something
    # didn't take.
    # Seven buttons now share this block (it started as four, at 0.065
    # apart/0.05 tall each, then five at 0.055/0.045, then six at
    # 0.048/0.04) — shrunk again (0.035 tall, 0.042 apart) to fit the new
    # one. The block's own top stays at 0.34 (still a real, deliberate 0.02
    # gap below the gene-dropdown suggestions' own lowest possible extent
    # at 0.36 — see gene_dropdown_ax's own comment — not flush against it).
    # Three rounds of shrinking is getting genuinely tight — an eighth
    # button would be a good prompt to restructure this as a 2-column grid
    # instead of shrinking a fourth time.
    BUTTON_BLOCK_HEIGHT = 0.035
    BUTTON_BLOCK_STEP = 0.042
    BUTTON_BLOCK_TOP_Y0 = 0.34 - BUTTON_BLOCK_HEIGHT  # 'Open data folder' button's own y0 (bottom edge)

    open_data_folder_ax = fig.add_axes([SIDEBAR_LEFT, BUTTON_BLOCK_TOP_Y0, SIDEBAR_WIDTH, BUTTON_BLOCK_HEIGHT])
    open_data_folder_button = Button(open_data_folder_ax, 'Open data folder', hovercolor='0.85')
    open_data_folder_button.label.set_fontsize(SIDEBAR_FONTSIZE)

    show_ax = fig.add_axes(
        [SIDEBAR_LEFT, BUTTON_BLOCK_TOP_Y0 - BUTTON_BLOCK_STEP, SIDEBAR_WIDTH, BUTTON_BLOCK_HEIGHT])
    show_button = Button(show_ax, 'Refresh', hovercolor='0.85')
    show_button.label.set_fontsize(SIDEBAR_FONTSIZE)

    save_section_maps_ax = fig.add_axes(
        [SIDEBAR_LEFT, BUTTON_BLOCK_TOP_Y0 - 2 * BUTTON_BLOCK_STEP, SIDEBAR_WIDTH, BUTTON_BLOCK_HEIGHT])
    save_section_maps_button = Button(save_section_maps_ax, 'Save section maps', hovercolor='0.85')
    save_section_maps_button.label.set_fontsize(SIDEBAR_FONTSIZE)

    save_ax = fig.add_axes(
        [SIDEBAR_LEFT, BUTTON_BLOCK_TOP_Y0 - 3 * BUTTON_BLOCK_STEP, SIDEBAR_WIDTH, BUTTON_BLOCK_HEIGHT])
    save_button = Button(save_ax, 'Save UMAP', hovercolor='0.85')
    save_button.label.set_fontsize(SIDEBAR_FONTSIZE)

    # export_degs_ax/export_expr_ax: only meaningful in 'Specified
    # <level>(s)' mode — hidden the rest of the time by on_mode_change,
    # same show/hide convention as query_ax.
    export_degs_ax = fig.add_axes(
        [SIDEBAR_LEFT, BUTTON_BLOCK_TOP_Y0 - 4 * BUTTON_BLOCK_STEP, SIDEBAR_WIDTH, BUTTON_BLOCK_HEIGHT])
    export_degs_button = Button(export_degs_ax, 'Export DEGs', hovercolor='0.85')
    export_degs_button.label.set_fontsize(SIDEBAR_FONTSIZE)
    export_degs_ax.set_visible(False)

    export_expr_ax = fig.add_axes(
        [SIDEBAR_LEFT, BUTTON_BLOCK_TOP_Y0 - 5 * BUTTON_BLOCK_STEP, SIDEBAR_WIDTH, BUTTON_BLOCK_HEIGHT])
    export_expr_button = Button(export_expr_ax, 'Export expression', hovercolor='0.85')
    export_expr_button.label.set_fontsize(SIDEBAR_FONTSIZE)
    export_expr_ax.set_visible(False)

    close_ax = fig.add_axes(
        [SIDEBAR_LEFT, BUTTON_BLOCK_TOP_Y0 - 6 * BUTTON_BLOCK_STEP, SIDEBAR_WIDTH, BUTTON_BLOCK_HEIGHT])
    close_button = Button(close_ax, 'Close', hovercolor='0.85')
    close_button.label.set_fontsize(SIDEBAR_FONTSIZE)
    close_button.on_clicked(lambda event: plt.close(fig))

    # "Working..." indicator — shown right before any operation that's
    # about to block the UI with a real, synchronous, non-interruptible
    # render (the zoom-preview settle draw, a gene expression render, ...),
    # so the user sees *something* happening instead of the window just
    # going unresponsive for however long that takes. Animated, like
    # everything else drawn via draw_animated_overlays/blit_sidebar_
    # overlays — the trick that makes this self-hiding: show_working_
    # indicator() blits it on screen immediately (cheap, fast), then the
    # slow operation's own eventual real draw naturally *excludes*
    # animated artists, so the moment that real draw finally lands on
    # screen, this disappears on its own — no separate "hide" call needed.
    working_text = fig.text(SIDEBAR_LEFT, 0.025, '', fontsize=SIDEBAR_FONTSIZE, va='center', ha='left',
                             color='firebrick', fontweight='bold')
    working_text.set_animated(True)

    # Set by redraw_subclass/redraw_gene whenever the current mode has no
    # valid parameter to show (no gene typed, an ID that doesn't parse,
    # ...) and working_text is showing a brief explanation instead of
    # "Working…" — see those functions' own comments. Checked by
    # hide_working_indicator (below) rather than by each of its 9-odd call
    # sites individually: an unrelated hide_working_indicator() call from,
    # say, a zoom settling or a hover timing out, would otherwise blank the
    # explanation before the actual problem (still unfixed) had anything to
    # do with that call at all.
    pending_param_warning = {'active': False}

    # Set for the duration of any operation guarded by show_working_indicator/
    # hide_working_indicator, so a rapid second click (e.g. double-clicking
    # Save UMAP) can't queue a redundant overlapping operation on top of one
    # already running — this is a single-threaded Tk/matplotlib event loop,
    # so there's no real concurrency risk, just wasted duplicate work. Only
    # sidebar controls are gated; panel zoom/pan/hover stay live throughout.
    busy_state = {'active': False}
    # Whether the divider-resize cursor is showing (see set_resize_cursor).
    # Defined up here because set_sidebar_controls_busy resets it, and that can
    # run before the resize handlers below are set up.
    resize_cursor_state = {'active': False}
    # Greying text/borders rather than any control's own face or fill: a
    # lighter face against still-black text actually read as *more*
    # prominent, not less — the dark text became the dominant visual
    # element against a paler background, the opposite of "inactive".
    # Fading the text and outline themselves, against each control's
    # normal, unchanged fill, is what actually reads as disabled. Shared
    # with the single-section ROI picker via DISABLED_CONTROL_COLOR.
    DISABLED_TEXT_COLOR = DISABLED_CONTROL_COLOR
    ENABLED_TEXT_COLOR = 'black'
    BUSY_GATED_BUTTONS = (open_data_folder_button, show_button, save_section_maps_button, save_button,
                          export_degs_button, export_expr_button, close_button)

    def set_sidebar_controls_busy(is_busy):
        # Only touches Python-side widget state (Button.active gates its
        # own click handling; the label color is picked up by draw_
        # animated_overlays() below, same as every other sidebar artist) —
        # no drawing or blitting here. An earlier version of this function
        # drew+blit the button axes directly, on the theory that a plain
        # color change (unlike an *animated* artist) can't be made visible
        # through the restore_region-based blit machinery alone. True, but
        # that direct blit was itself immediately undone: the very next
        # blit_hover_overlays()/blit_sidebar_overlays() call (there's
        # always one close by) restores the *stale* cached background over
        # the whole canvas before re-stamping only the artists draw_
        # animated_overlays() knows about — silently wiping the change
        # right back out since the buttons weren't among them. Making the
        # buttons genuine animated overlays (below) fixes that the same
        # way it already works for working_text/query_ax/etc.: whatever
        # their *current* label color is gets re-stamped on every such
        # blit, automatically, with no separate push needed here.
        was_busy = busy_state['active']
        busy_state['active'] = is_busy
        if is_busy != was_busy:
            # Hourglass for the whole operation. Returning to idle clears
            # resize_cursor_state too, so the divider hover check sets the
            # resize cursor again on the next mouse move near a boundary.
            resize_cursor_state['active'] = False
            if is_busy:
                fig.canvas.set_cursor(Cursors.WAIT)
                try:
                    # Applies it before the synchronous work starts, without
                    # processing queued input the way flush_events() would.
                    fig.canvas.manager.window.update_idletasks()
                except Exception:
                    pass
            else:
                # Callers re-enable the controls *before* their final redraw
                # (see end_zoom_previews), so the arrow waits for that.
                restore_cursor_after_pending_draw(fig, lambda: busy_state['active'])
        color = DISABLED_TEXT_COLOR if is_busy else ENABLED_TEXT_COLOR
        for button in BUSY_GATED_BUTTONS:
            button.set_active(not is_busy)
            button.label.set_color(color)
            for spine in button.ax.spines.values():
                spine.set_edgecolor(color)
        query_textbox.set_active(not is_busy)
        query_textbox.text_disp.set_color(color)
        set_gene_name_marks_color(query_textbox, DISABLED_TEXT_COLOR if is_busy else None)
        for spine in query_ax.spines.values():
            spine.set_edgecolor(color)
        query_label.set_color(color)  # "Subclass ID:"/"Gene name:" etc — query_ax's own caption
        # Mode radio and level picker are hand-rolled (see their own
        # comments above), not real Widget subclasses — on_mode_click/
        # on_level_selector_click already check busy_state themselves to
        # reject clicks, but nothing else about them responds to .active,
        # so their own text/border dimming has to happen by hand here too.
        for text_obj in mode_label_texts:
            text_obj.set_color(color)
        for spine in radio_ax.spines.values():
            spine.set_edgecolor(color)
        mode_dots.set_edgecolor(color)
        color_by_label.set_color(color)  # "Color by:" — radio_ax's own caption
        level_selector_text.set_color(color)
        for spine in level_selector_ax.spines.values():
            spine.set_edgecolor(color)
        level_selector_triangle.set_color(color)
        level_label.set_color(color)  # "Level:" — level_selector_ax's own caption

    def show_working_indicator(message='Working…'):
        # Any real, valid-parameter operation starting — the only time this
        # is called — means whatever explanation was pending is moot now;
        # the upcoming hide_working_indicator() at the end of that same
        # operation is what actually clears working_text, but only because
        # this already cleared the flag that would otherwise stop it.
        pending_param_warning['active'] = False
        set_sidebar_controls_busy(True)
        working_text.set_text(message)
        blit_sidebar_overlays()  # picks up both the message and the now-dimmed buttons
        # blit_sidebar_overlays' own fig.canvas.blit() only stages pixel
        # data — actually compositing it to the visible window needs Tk's
        # idle queue serviced (see force_repaint's own docstring). Without
        # this, every caller of show_working_indicator() that follows it
        # with a blocking synchronous call (a real fig.canvas.draw(), a
        # slow gene/dataset load, ...) never actually got the message onto
        # screen before freezing the UI — nothing serviced Tk's event loop
        # in between the blit and the freeze, so the "Working…" text (and
        # the dimmed/disabled controls) simply never appeared, reading as
        # the whole window silently hanging with no explanation.
        force_repaint()

    def hide_working_indicator():
        # Text cleared *before* re-enabling controls, not after: re-
        # enabling used to also force a repaint (see set_sidebar_controls_
        # busy's old docstring, above), and force_repaint() services Tk's
        # idle queue — which could include a still-*pending* draw_idle()
        # from the operation that just finished. Flushing that early, while
        # working_text still held the "Working…" message, baked the stale
        # text into that real draw's own pixels; nothing ever repainted
        # that region again afterward, so it stuck around until some
        # unrelated later interaction happened to blit over it. Clearing
        # the text first (when there's nothing pending to preserve it for)
        # means any draw that gets flushed by what follows already has the
        # right content. set_sidebar_controls_busy no longer forces a
        # repaint itself, but keeping this order is still the safer one.
        if not pending_param_warning['active']:
            working_text.set_text('')
        # Always re-enable controls, independent of pending_param_warning
        # above — busy_state tracks whether an operation is *running*, not
        # whether its result is a parameter explanation the user should
        # still see; either way, the operation itself is done and the user
        # should be able to interact again (e.g. to fix the bad parameter).
        set_sidebar_controls_busy(False)

    mode_state = {'mode': MODE_OPTIONS[0]}
    colorbar_state = {'cbar': None}
    # The current 'Specified <level>(s)' selection, updated at the end of
    # every redraw_subclass() call (targets=[] whenever there's nothing
    # valid to highlight) — export_de_genes (the "Export DEGs" button) reads
    # this rather than re-parsing query_textbox.text itself, so it always
    # exports exactly what's currently on screen, including the same
    # "which IDs were actually found" resolution redraw_subclass already did.
    last_id_selection = {'level': None, 'targets': [], 'colors': {}, 'counts': {}}
    # Set at the end of redraw_multi_genes (its own scatter has no scalar
    # array/cmap the way a single gene's does, and isn't a taxonomy
    # category either, so render_export_figure needs an explicit flag —
    # not scatter.get_array() is None, which alone can't distinguish
    # "multi-gene" from "categorical" — to know which of its export paths
    # applies). 'genes' is the list of 2 or 3 resolved gene names, in
    # red/green/blue order. Reset to inactive by clear_colorbar, the one
    # function every other redraw_* already calls first, so switching to
    # any other mode correctly clears it without needing its own explicit
    # reset too.
    multi_gene_state = {'active': False, 'genes': []}

    def set_section_panel_facecolor(color):
        """Sets every section panel's own axes background — panels are
        never ax.clear()'d the way the UMAP `ax` is on every redraw (only
        their scatter data/colors change in place), so this has to be its
        own explicit step rather than something a clear() naturally resets.
        Cheap regardless of how many panels there are: a patch color
        change, not a re-render — no need to skip it when `color` already
        matches. Invalidates each panel's own cached home-extent bitmap too
        (see invalidate_home_view_cache's own comment) — that bitmap is a
        stand-in for this axes' *rendered* content, background included, so
        a stale one would show the old background color during the next
        zoom-out preview until some unrelated interaction happened to
        recapture it."""
        for panel in section_panels.values():
            panel['ax'].set_facecolor(color)
            invalidate_home_view_cache(panel['ax'])

    def clear_colorbar():
        # cbar_ax is a permanent, pre-made axes (see its own comment above)
        # — cleared and hidden here, never removed from the figure, so
        # there's nothing to recreate on the next Gene/Imputed Gene view.
        # Also what clears a leftover draw_id_legend() (below), since the
        # two share this one axes and every redraw_* function calls this
        # unconditionally before deciding whether it has something new to
        # put there.
        cbar_ax.clear()
        cbar_ax.set_visible(False)
        colorbar_state['cbar'] = None
        # Reset unconditionally too — only redraw_all_subclasses sets this
        # back to a real row count, and only when it ends up actually
        # drawing a legend; every other redraw_* leaving it cleared is what
        # tells maybe_redraw_for_legend_resize (near the resize-handle
        # wiring) there's nothing for a resize to rebuild right now.
        legend_row_cap_state['rows'] = None
        # Same reasoning, for render_export_figure's own multi-gene branch —
        # only redraw_multi_genes sets this back to active.
        multi_gene_state['active'] = False
        # And for the section panels' own background color — only redraw_
        # multi_genes sets this to MULTI_GENE_UMAP_FACECOLOR instead, so
        # leaving multi-gene mode for any other redraw_* correctly restores
        # the normal black background via this one shared reset point.
        set_section_panel_facecolor(SECTION_PANEL_FACECOLOR)
        # Same for the UMAP axes itself — ax.clear() (called by every
        # redraw_*, right after this) does *not* reset facecolor on its
        # own, so without this a dark MULTI_GENE_UMAP_FACECOLOR set while
        # looking at multiple genes would otherwise persist indefinitely
        # into every other mode. redraw_multi_genes sets it back to
        # MULTI_GENE_UMAP_FACECOLOR itself, same pattern as the line above.
        ax.set_facecolor(UMAP_AXES_FACECOLOR)

    def draw_id_legend(level_name, target_colors, counts, target_shapes=None, extra_note=None):
        """The multi-ID legend for 'Specified <level>(s)' mode, into the
        same cbar_ax slot redraw_gene's real colorbar uses — see CBAR_
        WIDTH's own comment for why sharing one axes is fine (the two modes
        never show at the same time) and why that width is sized for this
        specifically, not just a colorbar's tick numbers.

        One swatch-and-label row per entry in `target_colors` (dict
        iteration order == the order the IDs were typed, since that dict
        was itself built by iterating the parsed query in order — so the
        legend lists IDs in the same order the user entered them, not
        re-sorted by ID number or match count). A "not found" entry still
        gets a row, with its own (0 cells) — the same information status_
        text already reports in prose, just also here at the swatch that'll
        actually appear, or not, on the plot itself.

        target_shapes: optional {key: (marker, is_open)} — category_rank_
        shape's own return shape — so a swatch shows the *actual* marker/
        fill that key is plotted with when UMAP_USE_SHAPES_ON_SCREEN is
        on, rather than always a plain filled dot. Any key missing from
        this dict (or the dict itself being None, the default) falls back
        to a plain filled circle, matching how this legend always looked
        before shapes existed.

        extra_note: optional plain-text final line (no swatch) — used by
        redraw_all_subclasses to report how many categories didn't make
        the row cap, since color_map itself can hold more entries than
        legend_max_rows_for_current_size() decided would fit; without
        this, a dropped category was silently absent with no indication
        anything had been left out at all."""
        cbar_ax.set_visible(True)
        cbar_ax.set_xlim(0, 1)
        cbar_ax.set_ylim(0, 1)
        cbar_ax.set_xticks([])
        cbar_ax.set_yticks([])
        for spine in cbar_ax.spines.values():
            spine.set_visible(False)
        # extra_note counts as one more row for sizing/centering purposes
        # (below) even though it has no swatch of its own — otherwise it
        # either overflowed the block or sat oddly close to the last real
        # row instead of reading as part of the same list.
        n = len(target_colors) + (1 if extra_note else 0)
        # Capped, not just 1/n: cbar_ax spans this whole mode's full plot
        # height, which is a lot of room for two or three rows — 1/n alone
        # stretched a short legend across the entire strip, wildly
        # oversized and floating oddly with only a couple of dots in it.
        # Still falls back to 1/n once there are enough rows that the cap
        # would otherwise make them overlap, so a long list still fits.
        MAX_ROW_HEIGHT_FRAC = 0.08
        row_h = min(MAX_ROW_HEIGHT_FRAC, 1.0 / n)
        block_h = n * row_h
        top_y = 0.5 + block_h / 2  # vertically centers the whole block in the strip
        # SWATCH_X/TEXT_X as axes-fraction positions (cbar_ax's own xlim is
        # 0-1, same convention as radio_ax and every other small hand-built
        # UI axes in this file) — not figure fractions, so this doesn't need
        # to know CBAR_WIDTH's actual value at all.
        SWATCH_X, TEXT_X = 0.22, 0.42
        legend_fontsize = SIDEBAR_FONTSIZE * 0.85
        for i, (target, color) in enumerate(target_colors.items()):
            y = top_y - (i + 0.5) * row_h  # centers of each row, top row first
            marker, is_open = (target_shapes or {}).get(target, ('o', False))
            if is_open:
                cbar_ax.scatter([SWATCH_X], [y], s=legend_fontsize * 4, marker=marker,
                                 facecolors='none', edgecolors=[color], linewidths=1.2, clip_on=False)
            else:
                cbar_ax.scatter([SWATCH_X], [y], s=legend_fontsize * 4, marker=marker,
                                 c=[color], linewidths=0, clip_on=False)
            cbar_ax.text(TEXT_X, y, f'{target} ({counts.get(target, 0)})',
                         fontsize=legend_fontsize, va='center', ha='left')
        if extra_note:
            y = top_y - (len(target_colors) + 0.5) * row_h  # one row below the last real entry
            cbar_ax.text(TEXT_X, y, extra_note, fontsize=legend_fontsize * 0.9,
                         va='center', ha='left', style='italic', color='0.4')

    # Fixed red/green/blue order — matches multi_gene_rgb's own channel
    # order and redraw_multi_genes' own resolved-gene ordering (first gene
    # typed = red, second = green, third = blue; 4th/5th/6th repeat the same
    # three colors for the '+' overlay layer). The 3rd/6th entries are
    # MULTI_GENE_BRIGHT_BLUE, not the named color 'blue', so the legend
    # swatch/label matches the actual (brighter) blue multi_gene_rgb uses
    # on screen.
    MULTI_GENE_CHANNEL_COLORS = ('red', 'green', MULTI_GENE_BRIGHT_BLUE, 'red', 'green', MULTI_GENE_BRIGHT_BLUE)
    # Legend swatch size per gene slot — full-size circle for genes 1-3 (the
    # base layer), a smaller one (matching MULTI_GENE_PLUS_SIZE_DIAMETER_
    # MULTIPLIER, squared since this scales an area) for genes 4-6 (the
    # small overlay dot drawn on top of it in redraw_multi_genes) — so the
    # legend swatch size reads as which layer each gene actually shows up
    # in on screen.
    MULTI_GENE_CHANNEL_SWATCH_SCALE = (1.0, 1.0, 1.0) + (MULTI_GENE_PLUS_SIZE_DIAMETER_MULTIPLIER ** 2,) * 3

    def draw_multi_gene_legend(genes, vmins, vmaxes):
        """The multi-gene mode's own legend, into the same cbar_ax slot a
        real colorbar or draw_id_legend's swatches use — there's no single
        colorbar to show here (color is a function of 2 or 3 values, not
        one scalar + a Colormap/Normalize pair), so this is a compact
        swatch-and-label row per gene instead (red/green/blue, in typed
        order), plus each gene's own log-expression range so the "how much
        more saturated does this much more expression make it" question
        has a concrete answer alongside the plot.

        A small 2D (or, for three genes, cube-sliced) gradient swatch would
        show the actual blend more completely than isolated per-gene rows
        do, but would need real room this narrow (CBAR_WIDTH-sized) strip
        doesn't have to spare without also shrinking the text rows below
        illegibly — left as a possible follow-up rather than attempted
        here."""
        cbar_ax.set_visible(True)
        cbar_ax.set_xlim(0, 1)
        cbar_ax.set_ylim(0, 1)
        cbar_ax.set_xticks([])
        cbar_ax.set_yticks([])
        for spine in cbar_ax.spines.values():
            spine.set_visible(False)
        SWATCH_X, TEXT_X = 0.22, 0.42
        legend_fontsize = SIDEBAR_FONTSIZE * 0.85
        n = len(genes)
        row_h = 0.14
        top_y = 0.5 + n * row_h / 2  # centers the n-row block in the strip
        for i, (gene, color, swatch_scale, vmin, vmax) in enumerate(
                zip(genes, MULTI_GENE_CHANNEL_COLORS, MULTI_GENE_CHANNEL_SWATCH_SCALE, vmins, vmaxes)):
            y = top_y - (i + 0.5) * row_h
            cbar_ax.scatter([SWATCH_X], [y], s=legend_fontsize * 4 * swatch_scale, marker='o', c=color,
                             linewidths=0, clip_on=False)
            cbar_ax.text(TEXT_X, y, gene, fontsize=legend_fontsize, va='center', ha='left', color=color)
            cbar_ax.text(TEXT_X, y - row_h * 0.42, f'log2: {vmin:.1f}–{vmax:.1f}',
                         fontsize=legend_fontsize * 0.75, va='center', ha='left', color='0.4')

    # Set by redraw_all_subclasses to the row count its own 'All <level>s'
    # legend was actually built with, or None whenever no such legend is
    # showing (Gene mode, or a level whose column is missing) — checked on
    # every resize (see maybe_redraw_for_legend_resize, near the resize-
    # handle wiring further down) so a window resize that changes how many
    # rows now fit triggers a real redraw to rebuild the legend at the new
    # count, not just a silent mismatch that lingers until the next
    # unrelated mode/level/query change.
    legend_row_cap_state = {'rows': None}

    def legend_max_rows_for_current_size():
        """How many 'All <level>s' legend rows fit in cbar_ax at a still-
        legible size, given the figure's *current* height — recomputed
        fresh each call (not a fixed constant) so a taller window (a 4k
        monitor, say) can show more rows before any get dropped, and a
        shorter one shows fewer, rather than one fixed count picked for
        whatever screen it happened to be tuned on. MIN_ROW_HEIGHT_IN is a
        hair under draw_id_legend's own MAX_ROW_HEIGHT_FRAC-driven row
        height at a typical window size, so *this* cap is normally what
        limits row count — draw_id_legend's own shrink-to-fit fallback is
        just cheap insurance against ever being handed more entries than
        this said would fit."""
        fig_h_in = fig.get_size_inches()[1]
        cbar_h_in = fig_h_in * (AREA_TOP - AREA_BOTTOM)
        legend_fontsize = SIDEBAR_FONTSIZE * 0.85
        min_row_height_in = legend_fontsize * 1.6 / 72
        return max(1, int(cbar_h_in / min_row_height_in))

    # Per-level cache for compute_ranked_category_colors, below — category
    # cell counts never change during a session, so once computed for a
    # level, the ranking/palette stays valid for the rest of it.
    category_color_cache = {}

    def compute_ranked_category_colors(level):
        """Colors for 'All <level>s' mode: categories ranked by cell count
        (largest first), the top UMAP_MAX_COLORED_CATEGORIES given distinct
        colors from category_rank_color's own rank-based cycle (index-based
        here, not the atlas-ID-based class_id_to_color used elsewhere in
        this file — there's no "this category's own stable ID color" to
        preserve, only a need for N mutually well-separated colors, which
        is exactly what that cycle guarantees, up to its own 20-color
        period). Export legends/plots additionally use category_rank_
        shape to pair each rank with a marker shape/fill style, so once
        the cycle repeats (rank 20, 40, ...) colors alone don't have to
        keep telling categories apart — see that function's own docstring.

        Replaces the previous approach of borrowing scanpy's own
        adata.uns[f'{level}_colors'] (via a throwaway sc.pl.umap render):
        scanpy assigns colors or not for the *entire* column at once — every
        category or none — so past its own ~103-category cutoff it fell
        back to uniform gray with no way to ask for "color the biggest
        ones". Computing it here directly also drops that throwaway render
        entirely, which was itself a non-trivial cost paid on every mode
        switch into 'All <level>s'.

        Returns a dict with:
          'umap_color_map'    category -> color, top N only (rest: not a UMAP color, i.e. gray)
          'section_color_map' category -> color, every category, recycling through the same N
          'counts'             the full category -> cell count Series (for the label-sizing feature)
        """
        if level in category_color_cache:
            return category_color_cache[level]
        obs_col = adata.obs[level]
        if not isinstance(obs_col.dtype, pd.CategoricalDtype):
            obs_col = obs_col.astype('category')
        counts = obs_col.value_counts()  # descending by count; only categories actually present
        ranked = counts.index.tolist()
        n_colored = min(UMAP_MAX_COLORED_CATEGORIES, len(ranked))
        # category_rank_color already returns hex strings (unlike class_id_
        # to_color's raw (r, g, b, 1.0) tuples, which numpy's uniform-
        # sequence-length shape inference used to trip over when mixed with
        # the 'lightgray'/SECTION_UNKNOWN_COLOR string fallback — an (N, 4)
        # array instead of the (N,) scatter expects, the "'c' argument has
        # 4x as many elements as x/y" failure) — nothing further to convert.
        palette = [category_rank_color(i) for i in range(n_colored)]
        result = {
            'umap_color_map': {cat: palette[r] for r, cat in enumerate(ranked[:n_colored])},
            'section_color_map': ({cat: palette[r % n_colored] for r, cat in enumerate(ranked)}
                                   if n_colored else {}),
            'counts': counts,
        }
        category_color_cache[level] = result
        return result

    # --- Section-panel background coloring, kept in sync with the UMAP's
    # own "Color by" mode/level -----------------------------------------
    # The section panels show *every* cell of each section (from
    # adata_backed, a superset of this run's own `adata`), so recoloring
    # them can't just reuse the UMAP's own point_colors/color_values arrays
    # directly — each panel's per-cell class/subclass/supertype/cluster
    # labels (level_values/level_ids) and, where available, row index into
    # `adata` (umap_idx, -1 if that background cell isn't part of this run)
    # were precomputed once at panel-build time for exactly this purpose.
    # (SECTION_UNKNOWN_COLOR, used throughout below, is at the top of the file.)

    def apply_panel_colors_with_gray_behind(panel, colors, is_gray):
        """Sets background_artist's positions *and* colors together so gray
        (unknown/unmatched) cells draw first — behind — and colored cells
        draw last — in front — same "gray background, colored on top"
        convention scatter_gray_then_colored already uses for the section-
        grid picker and single-section ROI picker's own spatial panels,
        rather than whatever incidental order these cells happened to be
        precomputed in at panel-build time.

        `colors` and `is_gray` must already be in panel['hover_x']/
        ['hover_y']'s own row order (all three, and every panel field
        derived from it — hover_ids/level_values/level_ids/umap_idx — share
        one fixed index space). Only the scatter artist's own offsets/
        facecolor get reordered here, a copy local to this call — the
        stored hover_x/hover_y/etc. arrays themselves are left untouched,
        since resolve_section_hover_target's nearest-point lookup indexes
        into *those*, independently of whatever order the artist actually
        draws in."""
        xs, ys = panel['hover_x'], panel['hover_y']
        order = np.argsort(~np.asarray(is_gray), kind='stable')
        full_offsets = np.column_stack([xs[order], ys[order]])
        full_colors = np.asarray(colors)[order]
        panel['background_artist'].set_offsets(full_offsets)
        panel['background_artist'].set_facecolor(full_colors)
        # Snapshot of the *complete* (unfiltered) point set this call just
        # set on the artist — filter_section_scatter_to_viewport (see its
        # own docstring) always re-filters from this, never from whatever
        # the artist currently holds, so zooming in/out past ZOOM_BITMAP_
        # ONLY_MAX_MULTIPLIER repeatedly never compounds/loses points. This
        # function is the single choke point every set_section_colors_*
        # path already goes through (see the comment below), so it's the
        # one place that's always right after a fresh, complete color
        # assignment — same reasoning as set_main_scatter_artists' own
        # snapshot for the UMAP.
        panel['full_offsets'] = full_offsets
        panel['full_colors'] = full_colors
        # Defaults every caller back to "show the real scatter" — the only
        # thing that ever hides it instead is show_section_home_cache_or_
        # scatter, called by set_section_colors_categorical right after
        # this, for categorical ('All <level>s') mode specifically. Without
        # this, switching from a cached categorical view to Gene mode or a
        # 'Specified <level>(s)' highlight (both of which call this
        # function too, but never show_section_home_cache_or_scatter) left
        # background_artist hidden and the stale cached image still
        # showing on top of it, no matter what fresh data was just set.
        panel['background_artist'].set_visible(True)
        if panel['cached_home_image'] is not None:
            panel['cached_home_image'].set_visible(False)
        # This panel no longer looks the way its cached home bitmap does —
        # the single choke point every set_section_colors_* path goes
        # through, so invalidating here covers all of them. Re-captured by
        # cache_blit_background on the next real draw that finds this panel
        # back at its home extent; until then a zoom-out on it falls back to
        # the plain current-view snapshot.
        invalidate_home_view_cache(panel['ax'])

    # --- Resize fast mode ------------------------------------------------
    # While a divider drag or an OS window-edge drag is in progress, every
    # *data* artist (the UMAP scatter and its centroid labels, plus every
    # section panel's own scatter/highlights) is hidden outright, leaving
    # only the cheap frame: axes boxes, titles, sidebar widgets, status
    # panel. Each resize frame then costs almost nothing, so the window
    # tracks the drag live; the real content comes back the moment the drag
    # settles (exit_resize_fast_mode).
    #
    # Recoloring the points instead of hiding them (the previous attempt at
    # this) didn't help: the dominant per-frame costs are rasterizing the
    # UMAP's own up-to-~200k-point scatter — which recoloring the *section*
    # panels never touched at all — and cache_blit_background's full-figure
    # copy_from_bbox plus its per-axes home_view_cache snapshots, which run
    # on every real draw regardless of what any artist is colored. Hiding
    # skips the rasterization entirely, and skip_expensive_draw_caching
    # below suppresses the caching work for the duration.
    # (resize_fast_mode itself is declared earlier — see its own comment.)

    def iter_heavy_artists():
        for artist in main_scatter_state['artists']:
            yield artist
        for artist in umap_label_state['artists']:
            yield artist
        if umap_highlight_state['artist'] is not None:
            yield umap_highlight_state['artist']
        if group_highlight_state['artist'] is not None:
            yield group_highlight_state['artist']
        for panel in section_panels.values():
            yield panel['background_artist']
            yield panel['highlight']
            if panel['group_highlight'] is not None:
                yield panel['group_highlight']

    def enter_resize_fast_mode():
        if resize_fast_mode['active']:
            return
        resize_fast_mode['active'] = True
        # Each artist's own pre-drag visibility is remembered rather than
        # assumed True — the single-cell/family highlights in particular are
        # routinely already hidden (nothing hovered), and unhiding those on
        # exit would light up stale highlights that were never showing.
        resize_fast_mode['saved'] = [(a, a.get_visible()) for a in iter_heavy_artists()]
        for artist, _was_visible in resize_fast_mode['saved']:
            artist.set_visible(False)

    def exit_resize_fast_mode():
        if not resize_fast_mode['active']:
            return
        resize_fast_mode['active'] = False
        for artist, was_visible in resize_fast_mode.get('saved', []):
            artist.set_visible(was_visible)
        resize_fast_mode['saved'] = []
        # Everything moved/resized while hidden, so no axes' cached home
        # bitmap is a valid stand-in any more.
        invalidate_home_view_cache()

    def show_section_home_cache_or_scatter(panel, sec, level):
        """Called right after apply_panel_colors_with_gray_behind has set
        panel['background_artist']'s own data (so full_offsets/full_colors
        — needed for zoom-in filtering regardless of what ends up actually
        displayed — are always correct), this decides whether the panel's
        *visible* content should be that real scatter or a cached home-
        view PNG (see section_home_cache_path's own docstring for why this
        only applies to categorical coloring, not Gene/Imputed Gene or a
        'Specified <level>(s)' highlight).

        Only relevant when the panel is currently sitting exactly at its
        own home extent — a cached home-view image is meaningless once
        zoomed/panned; end_zoom_previews' own zoom-in handling already
        takes over there — and run_folder is known (no folder to cache
        into for a session not tied to a saved run). On a cache hit, swaps
        to the cached image and hides the real scatter: cheap, since a
        hidden artist costs ~nothing to draw (same reasoning already
        established for the UMAP/zoom-preview work this session). On a
        miss, renders one now — off-screen, independent of this window,
        see render_section_home_view_png — saves it, and uses it the same
        way, so *this* session also gets the speedup, not just later ones.

        Also seeds home_view_cache directly from whatever image ends up
        showing, so zoom-out backfill (swap_preview_to_home_cache) works
        immediately rather than waiting on capture_home_view_if_at_home's
        own later, opportunistic capture.

        Returns one of 'hit' (loaded an existing PNG), 'rendered' (no
        usable PNG existed — rendered and saved one just now), or
        'skipped' (not at home, no run_folder, or something failed —
        real scatter shown instead) — purely for set_section_colors_
        categorical's own ZOOM_DEBUG_DIAGNOSTICS accounting."""
        ax_ = panel['ax']
        image_artist = panel['cached_home_image']

        def use_real_scatter():
            if image_artist is not None:
                image_artist.set_visible(False)
            panel['background_artist'].set_visible(True)

        if run_folder is None:
            use_real_scatter()
            return 'skipped'
        home = pannable_axes.get(ax_)
        if home is None:
            use_real_scatter()
            return 'skipped'
        xlim, ylim = ax_.get_xlim(), ax_.get_ylim()
        # sorted() comparison, not elementwise — see capture_home_view_if_
        # at_home's own comment: section panels are y-inverted, so a panel
        # sitting exactly at home reports get_ylim() as home_ylim reversed.
        at_home = (np.allclose(sorted(xlim), sorted(home['home_xlim']))
                   and np.allclose(sorted(ylim), sorted(home['home_ylim'])))
        if not at_home:
            use_real_scatter()
            return 'skipped'
        cache_path = section_home_cache_path(run_folder, sec, level)
        rgba = None
        outcome = 'rendered'
        if cache_path.exists():
            try:
                rgba = np.asarray(Image.open(cache_path).convert('RGBA'), dtype=np.uint8)
                outcome = 'hit'
            except Exception:
                rgba = None  # unreadable/corrupt — just re-render below
        if rgba is None:
            outcome = 'rendered'
            try:
                # full_offsets/full_colors (set just above, in apply_panel_
                # colors_with_gray_behind) are already gray-first ordered
                # and row-aligned with each other — is_gray=all-False here
                # just means "nothing further to reorder", not "no cell is
                # actually gray" (their colors already reflect that).
                full_offsets, full_colors = panel['full_offsets'], panel['full_colors']
                rgba = render_section_home_view_png(
                    full_offsets[:, 0], full_offsets[:, 1], full_colors,
                    np.zeros(len(full_colors), dtype=bool),
                    home['home_xlim'], home['home_ylim'],
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgba, mode='RGBA').save(cache_path)
            except Exception as e:
                print(f"Warning: could not cache home-view image for section {sec} ({level}): {e}")
                rgba = None
        if rgba is None:
            use_real_scatter()
            return 'skipped'
        extent = (min(home['home_xlim']), max(home['home_xlim']),
                  min(home['home_ylim']), max(home['home_ylim']))
        if image_artist is None:
            # Axes.imshow(aspect=...), if not None, calls self.ax.set_
            # aspect(aspect) internally — it resets the *axes'* own aspect
            # setting, not just how this one image gets drawn. Passing
            # 'auto' here silently overwrote this panel's 'equal'/'box'
            # aspect (set once at panel construction) the first time its
            # cached home-view image was ever shown, and nothing ever set
            # it back — every subsequent redraw of this axes (this cached
            # image on later runs, and the real scatter too, once zoomed
            # in past the bitmap-only threshold) then stretched to fill
            # whatever box shape the section grid happened to give this
            # panel, unless that shape already happened to match the data's
            # own aspect ratio. Same bug, same fix, as begin_zoom_preview's
            # own imshow calls already have to save/restore around (see its
            # own comment) — just a separate code path that was missing it.
            saved_aspect = ax_.get_aspect()
            saved_adjustable = ax_.get_adjustable()
            image_artist = ax_.imshow(rgba, extent=extent, aspect='auto', origin='upper', zorder=2)
            ax_.set_aspect(saved_aspect, adjustable=saved_adjustable)
            panel['cached_home_image'] = image_artist
        else:
            image_artist.set_data(rgba)
            image_artist.set_extent(extent)
            image_artist.set_visible(True)
        panel['background_artist'].set_visible(False)
        home_view_cache[ax_] = {'buf': rgba, 'extent': extent}
        return outcome

    def set_section_colors_categorical(level):
        # section_color_map covers *every* category (recycled past the top
        # N — see UMAP_MAX_COLORED_CATEGORIES), unlike the UMAP's own
        # umap_color_map, so this only falls back to SECTION_UNKNOWN_COLOR
        # for a level with no data at all, not for being a low-ranked
        # category the way the UMAP scatter does.
        t_start = time.perf_counter()
        outcome_counts = {'hit': 0, 'rendered': 0, 'skipped': 0}
        color_map = (compute_ranked_category_colors(level)['section_color_map']
                     if level in adata.obs.columns else {})
        for sec, panel in section_panels.items():
            vals = panel['level_values'].get(level)
            if not color_map or vals is None:
                panel['background_artist'].set_facecolor(SECTION_UNKNOWN_COLOR)
                if panel['cached_home_image'] is not None:
                    panel['cached_home_image'].set_visible(False)
                panel['background_artist'].set_visible(True)
            else:
                colors = np.array([color_map.get(v, SECTION_UNKNOWN_COLOR) for v in vals], dtype=object)
                apply_panel_colors_with_gray_behind(panel, colors, colors == SECTION_UNKNOWN_COLOR)
                outcome = show_section_home_cache_or_scatter(panel, sec, level)
                outcome_counts[outcome] += 1
        if ZOOM_DEBUG_DIAGNOSTICS:
            elapsed = time.perf_counter() - t_start
            print(f"[section-cache] set_section_colors_categorical('{level}'): {elapsed:.3f}s — "
                  f"{outcome_counts['hit']} from cache, {outcome_counts['rendered']} rendered+saved fresh, "
                  f"{outcome_counts['skipped']} skipped (not at home / no run_folder / failed) "
                  f"(run_folder={'set' if run_folder is not None else 'None — caching disabled'})")

    def set_section_colors_for_ids(level, target_colors):
        """Colors each panel's cells to match the UMAP's 'Single <level>'
        highlight. `target_colors` maps ID -> color (as chosen by
        redraw_subclass, so the same ID is the same color in both places);
        an empty mapping greys every panel out."""
        for panel in section_panels.values():
            ids_arr = panel['level_ids'].get(level)
            if ids_arr is None or not target_colors:
                panel['background_artist'].set_facecolor(SECTION_UNKNOWN_COLOR)
                continue
            colors = np.full(len(ids_arr), SECTION_UNKNOWN_COLOR, dtype=object)
            for target, color in target_colors.items():
                colors[ids_arr == target] = color
            apply_panel_colors_with_gray_behind(panel, colors, colors == SECTION_UNKNOWN_COLOR)

    def set_section_colors_by_value(color_values, cmap, norm, full_source=None, full_gene_col=None,
                                     full_layer=None, full_transform=None):
        # color_values is aligned with adata's own row order/coords (same
        # indexing space as each panel's precomputed 'umap_idx') — covers
        # only the ROI's own cells. Cells that are only in the unfiltered
        # background (the rest of that section, outside the ROI) have no
        # entry in color_values at all — if `full_source` is given (an
        # AnnData covering the *whole* section, e.g. adata_backed), those
        # cells' values are looked up there instead, by cell ID (not
        # assumed row order — full_source's own row order has no reason to
        # match adata's), via the same bulk-read-and-cache approach
        # _get_full_gene_column already uses for the imputed dataset.
        # Falls back to SECTION_UNKNOWN_COLOR only for cells `full_source`
        # doesn't have either (or when full_source isn't given at all,
        # same as before this parameter existed).
        unknown_rgba = mcolors.to_rgba(SECTION_UNKNOWN_COLOR)
        full_col = None
        if full_source is not None and full_gene_col is not None:
            try:
                full_col = _get_full_gene_column(full_source, full_gene_col, layer=full_layer)
            except Exception:
                full_col = None  # e.g. full_source has no such layer/gene — just skip the extension
        for panel in section_panels.values():
            idx = panel['umap_idx']
            colors = np.tile(unknown_rgba, (len(idx), 1))
            is_gray = np.ones(len(idx), dtype=bool)
            in_adata = idx >= 0
            if in_adata.any():
                colors[in_adata] = cmap(norm(color_values[idx[in_adata]]))
                is_gray[in_adata] = False
            missing = ~in_adata
            if full_col is not None and missing.any():
                missing_rows = np.flatnonzero(missing)
                positions = full_source.obs_names.get_indexer(panel['hover_ids'][missing_rows])
                found = positions >= 0
                if found.any():
                    vals = full_col[positions[found]]
                    if full_transform is not None:
                        vals = full_transform(vals)
                    colors[missing_rows[found]] = cmap(norm(vals))
                    is_gray[missing_rows[found]] = False
            apply_panel_colors_with_gray_behind(panel, colors, is_gray)

    def set_section_colors_multi_gene(norms, baseline_rgb, vmins, vmaxes,
                                       full_sources, full_gene_cols, full_layers, full_transforms):
        """Multi-gene analogue of set_section_colors_by_value — same
        in_adata + background-extension structure (see that function's own
        comment for the general shape), generalized to a *list* of genes
        (2 or 3): `norms`/`vmins`/`vmaxes`/`full_sources`/`full_gene_cols`/
        `full_layers`/`full_transforms` are all parallel lists, one entry
        per gene, since each gene can come from a different background
        source (e.g. one is on the MERFISH panel and gets the adata_backed
        extension, another's imputed and doesn't need one at all — its own
        full_sources entry is simply None in that case, same as redraw_
        single_gene's own existing convention).

        Cells found via a background source are normalized with vmin/vmax
        already computed from adata's own cells (same ones the UMAP
        scatter used), not a separately-scaled background-only range — so
        a background cell's color sits on the same red/green/blue scale as
        an in-ROI one instead of being stretched differently. A cell with
        no data for a given gene at all (neither in adata nor found via
        that gene's own background source) contributes 0 to that gene's
        own channel via multi_gene_rgb's own NaN handling, not a missing-
        data marker — 'missing' here just reads as 'not expressed', the
        same convention a single-gene view already uses for a cell absent
        from the imputed dataset."""
        def normalize_bg(values, vmin, vmax):
            if vmax > vmin:
                return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
            return np.zeros_like(values)

        n_genes = len(norms)
        full_cols = []
        for src, col, layer in zip(full_sources, full_gene_cols, full_layers):
            full_col = None
            if src is not None and col is not None:
                try:
                    full_col = _get_full_gene_column(src, col, layer=layer)
                except Exception:
                    full_col = None
            full_cols.append(full_col)

        unknown_rgba = mcolors.to_rgba(SECTION_UNKNOWN_COLOR)
        for panel in section_panels.values():
            idx = panel['umap_idx']
            colors = np.tile(unknown_rgba, (len(idx), 1))
            is_gray = np.ones(len(idx), dtype=bool)
            in_adata = idx >= 0
            if in_adata.any():
                colors[in_adata, :3] = multi_gene_rgb([n[idx[in_adata]] for n in norms], baseline_rgb)
                is_gray[in_adata] = False
            missing = ~in_adata
            if any(c is not None for c in full_cols) and missing.any():
                missing_rows = np.flatnonzero(missing)
                cell_ids = panel['hover_ids'][missing_rows]
                bg_norms = [np.zeros(len(missing_rows)) for _ in range(n_genes)]
                found_any = np.zeros(len(missing_rows), dtype=bool)
                for gi, (full_col, src, transform, vmin, vmax) in enumerate(
                        zip(full_cols, full_sources, full_transforms, vmins, vmaxes)):
                    if full_col is None:
                        continue
                    positions = src.obs_names.get_indexer(cell_ids)
                    found = positions >= 0
                    if found.any():
                        vals = full_col[positions[found]]
                        if transform is not None:
                            vals = transform(vals)
                        bg_norms[gi][found] = normalize_bg(vals, vmin, vmax)
                        found_any |= found
                if found_any.any():
                    colors[missing_rows[found_any], :3] = multi_gene_rgb(
                        [n[found_any] for n in bg_norms], baseline_rgb)
                    is_gray[missing_rows[found_any]] = False
            apply_panel_colors_with_gray_behind(panel, colors, is_gray)

    def redraw_all_subclasses():
        show_working_indicator()
        level = level_state['value']
        clear_colorbar()
        ax.clear()
        invalidate_home_view_cache(ax)  # about to render different content — see that function
        umap_label_state['artists'] = []  # ax.clear() just removed any centroid labels
        recreate_umap_highlight()
        recreate_group_highlight()
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(pannable_axes[ax]['home_xlim'])
        ax.set_ylim(pannable_axes[ax]['home_ylim'])
        ax.set_xticks([])
        ax.set_yticks([])
        if cell_level_arrays[level] is None:
            set_main_scatter_artists([ax.scatter(coords[:, 0], coords[:, 1], c='lightgray',
                                                  s=UMAP_POINT_SIZE, alpha=UMAP_POINT_ALPHA,
                                                  linewidths=0)])
            update_umap_dot_size()
            ax.set_xlabel('UMAP1', fontsize=SIDEBAR_FONTSIZE)
            ax.set_ylabel('UMAP2', fontsize=SIDEBAR_FONTSIZE)
            status_text.set_text(f"No '{level}' column available on this dataset.")
            set_section_colors_categorical(level)
            fig.canvas.draw_idle()
            hide_working_indicator()
            return
        palette_info = compute_ranked_category_colors(level)
        obs_col = adata.obs[level]
        if not isinstance(obs_col.dtype, pd.CategoricalDtype):
            obs_col = obs_col.astype('category')
        # umap_color_map only has the top UMAP_MAX_COLORED_CATEGORIES by
        # cell count (see compute_ranked_category_colors); everything else
        # falls through to gray (category_of_point_from_mapping's own -1),
        # same convention as every other mode's own "not highlighted" cells.
        color_map = palette_info['umap_color_map']
        obs_values = obs_col.to_numpy()
        category_of_point, rank_color = category_of_point_from_mapping(obs_values, color_map)
        set_main_scatter_artists(plot_categorical_umap(
            ax, category_of_point, rank_color, UMAP_POINT_SIZE, UMAP_POINT_ALPHA, UMAP_USE_SHAPES_ON_SCREEN))
        set_section_colors_categorical(level)
        umap_label_state['artists'] = add_umap_centroid_labels(level, palette_info)
        # Reuses draw_id_legend as-is (same "swatch + label (count)" row
        # format 'Specified <level>(s)' mode already uses), keyed by
        # leading numeric ID rather than the full category string — same
        # convention add_umap_centroid_labels' own centroid dots already
        # use, so a legend row and its dot are trivially cross-referenced.
        # color_map is already ranked by cell count, largest first (see
        # compute_ranked_category_colors), so taking its first max_rows
        # entries drops the smallest-count categories first when there
        # isn't room for all of them — but that's only for *which*
        # categories make the cut; the rows are then re-sorted by
        # taxonomic (leading numeric ID) order for display, since that's a
        # far more useful reading order than "biggest cluster first" once
        # they're all on screen together. max_rows itself is recomputed
        # from the figure's *current* height (legend_max_rows_for_current_
        # size), not a fixed count — remembered in legend_row_cap_state so
        # maybe_redraw_for_legend_resize (near the resize-handle wiring)
        # can tell whether a later resize actually changed how many rows
        # now fit, and only then pay for rebuilding this.
        max_rows = legend_max_rows_for_current_size()
        legend_row_cap_state['rows'] = max_rows
        legend_entries = list(color_map.items())[:max_rows]

        def _legend_id_key(entry):
            match = re.match(r'^\s*(\d+)', str(entry[0]))
            return int(match.group(1)) if match else math.inf

        legend_entries.sort(key=_legend_id_key)
        # Shapes keyed by each category's *original* color-cycle rank
        # (its position in color_map, before the ID re-sort above) — the
        # same rank plot_categorical_umap just used to pick its own
        # marker/fill, so a legend swatch always matches what's actually
        # on the plot regardless of the legend's own (different) display
        # order.
        rank_of_category = {cat: rank for rank, cat in enumerate(color_map)}
        legend_colors, legend_counts, legend_shapes = {}, {}, {}
        for cat, color in legend_entries:
            match = re.match(r'^\s*(\d+)', str(cat))
            label = match.group(1) if match else str(cat)
            legend_colors[label] = color
            legend_counts[label] = int(palette_info['counts'][cat])
            legend_shapes[label] = (category_rank_shape(rank_of_category[cat])
                                     if UMAP_USE_SHAPES_ON_SCREEN else ('o', False))
        dropped_count = len(color_map) - len(legend_entries)
        extra_note = f"+{dropped_count} more" if dropped_count > 0 else None
        draw_id_legend(LEVEL_DISPLAY_NAMES[level], legend_colors, legend_counts, legend_shapes, extra_note)
        # Deliberately after the labels are created, so this single call
        # sizes them and the dots together — called before, it would set the
        # dots correctly and leave the labels at their unscaled base size
        # until the next zoom tick.
        update_umap_dot_size()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('UMAP1', fontsize=SIDEBAR_FONTSIZE)
        ax.set_ylabel('UMAP2', fontsize=SIDEBAR_FONTSIZE)
        ax.set_title(f'All {LEVEL_DISPLAY_NAMES[level].lower()}s', fontsize=SIDEBAR_FONTSIZE)
        current_view_name['token'] = f'all_{level}'
        n_categories = obs_col.nunique()  # categories actually present, not just declared
        n_colored = len(color_map)
        level_word = LEVEL_DISPLAY_NAMES[level].lower()
        if n_colored < n_categories:
            status_text.set_text(
                f"Showing all {n_categories} {level_word}s{HOVER_FIELD_SEP}"
                f"{n_colored} largest colored & labeled, rest gray (recycled colors on sections)."
            )
        else:
            status_text.set_text(f"Showing all {n_categories} {level_word}s (ID labels at each centroid).")
        fig.canvas.draw_idle()
        hide_working_indicator()

    def redraw_subclass(query):
        show_working_indicator()
        level = level_state['value']
        level_name = LEVEL_DISPLAY_NAMES[level]
        clear_colorbar()
        ax.clear()
        invalidate_home_view_cache(ax)  # about to render different content — see that function
        umap_label_state['artists'] = []  # ax.clear() just removed any centroid labels
        recreate_umap_highlight()
        recreate_group_highlight()
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(pannable_axes[ax]['home_xlim'])
        ax.set_ylim(pannable_axes[ax]['home_ylim'])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('UMAP1', fontsize=SIDEBAR_FONTSIZE)
        ax.set_ylabel('UMAP2', fontsize=SIDEBAR_FONTSIZE)
        targets, target_colors, counts = [], {}, {}
        if level_ids_arrays[level] is None:
            set_main_scatter_artists([ax.scatter(coords[:, 0], coords[:, 1], c='lightgray',
                                                  s=UMAP_POINT_SIZE, alpha=UMAP_POINT_ALPHA,
                                                  linewidths=0)])
            status_text.set_text(f"No '{level}' column available on this dataset.")
            working_text.set_text(f"No '{level}' data")
            pending_param_warning['active'] = True
        else:
            targets, bad_tokens = parse_id_query(query)
            if not targets:
                set_main_scatter_artists([ax.scatter(coords[:, 0], coords[:, 1], c='lightgray',
                                                      s=UMAP_POINT_SIZE, alpha=UMAP_POINT_ALPHA,
                                                      linewidths=0)])
                status_text.set_text(
                    f"No valid {level_name.lower()} number in '{query.strip()}'." if bad_tokens else
                    f"Enter one or more {level_name.lower()} numbers, separated by commas."
                )
                working_text.set_text("Invalid ID number" if bad_tokens else "Enter ID number(s)")
                pending_param_warning['active'] = True
            else:
                # One color per requested ID (a single ID keeps the original
                # plain red — an explicit, unambiguous choice for the
                # common case, not just rank 0 of the cycle below), so
                # several can be compared at once instead of merging into
                # one indistinguishable highlight. Same rank-based color
                # cycle as 'All <level>s' mode (category_rank_color) —
                # each typed ID's position in the query is its rank, kept
                # in last_id_selection so save_current_umap's export can
                # also assign matching marker shapes (category_rank_shape)
                # by that same rank.
                if len(targets) == 1:
                    target_colors = {targets[0]: 'red'}
                else:
                    target_colors = {t: category_rank_color(i) for i, t in enumerate(targets)}
                ids_arr = level_ids_arrays[level].to_numpy()
                counts = {t: int((ids_arr == t).sum()) for t in targets}
                matched = [t for t in targets if counts[t]]
                missing = [t for t in targets if not counts[t]]
                # category_of_point_from_mapping/plot_categorical_umap draw
                # the gray "rest" first/underneath and each highlighted ID
                # on top — same reasoning as the section panels' own gray-
                # behind ordering, just via separate scatter() calls now
                # rather than one call over a stable-sorted point order.
                category_of_point, rank_color = category_of_point_from_mapping(ids_arr, target_colors)
                set_main_scatter_artists(plot_categorical_umap(
                    ax, category_of_point, rank_color, UMAP_POINT_SIZE, UMAP_POINT_ALPHA,
                    UMAP_USE_SHAPES_ON_SCREEN))
                ax.set_title(f"{level_name} {', '.join(str(t) for t in targets)}", fontsize=SIDEBAR_FONTSIZE)
                current_view_name['token'] = f"{level}_{'-'.join(str(t) for t in targets)}"
                parts = []
                if matched:
                    parts.append('Highlighting ' + ', '.join(
                        f"{level_name.lower()} {t} ({counts[t]} cells)" for t in matched))
                if missing:
                    parts.append('not found: ' + ', '.join(str(t) for t in missing))
                if bad_tokens:
                    parts.append('ignored: ' + ', '.join(bad_tokens))
                status_text.set_text(HOVER_FIELD_SEP.join(parts) + '.' if parts else '')
                # Only once there's more than one color to explain — a
                # single ID is already unambiguous (plain red), and
                # clear_colorbar() at the top of this function already left
                # cbar_ax hidden for that case, same as it does for every
                # mode that doesn't need this slot.
                if len(targets) > 1:
                    # Keyed by the same enumerate(targets) rank target_colors
                    # itself was built from — target_colors' own dict order
                    # (typed order, never re-sorted) is what plot_categorical_
                    # umap actually used to pick markers, so this always
                    # matches what's on the plot.
                    target_shapes = ({t: category_rank_shape(i) for i, t in enumerate(targets)}
                                      if UMAP_USE_SHAPES_ON_SCREEN else None)
                    draw_id_legend(level_name, target_colors, counts, target_shapes)
        last_id_selection['level'] = level
        last_id_selection['targets'] = targets
        # Remembered alongside level/targets (rather than recomputed later)
        # so save_current_umap's export legend can show exactly the colors
        # and counts actually drawn, without duplicating the color-cycling/
        # counting logic above.
        last_id_selection['colors'] = target_colors
        last_id_selection['counts'] = counts
        set_section_colors_for_ids(level, target_colors)
        update_umap_dot_size()
        fig.canvas.draw_idle()
        # Unconditional — hide_working_indicator() itself checks pending_
        # param_warning now, so it already no-ops correctly on its own
        # whenever one of the branches above just set that flag, leaving
        # this branch's own explanation on screen instead of wiping it back
        # to blank. See that flag's own comment for why this couldn't just
        # rely on the fig.canvas.draw_idle() above the way redraw_gene's
        # early returns do — every branch here (valid or not) reaches this
        # same full draw, which excludes animated artists outright.
        hide_working_indicator()

    # This viewer's own gene-count cap, used below instead of the shared
    # MAX_GENE_NAMES (3) — this is the one window that knows what to do
    # with 4-6 genes (the '+' overlay layer in redraw_multi_genes); every
    # other gene-entry box in the app still only renders up to 3 (see
    # render_multi_gene_expression_array/multi_gene_rgb's own 3-channel
    # cap), so bumping the *shared* constant would have let those boxes
    # silently accept and then drop names 4-6 with no indication.
    INTERACTIVE_MULTI_GENE_MAX_NAMES = 6
    TOO_MANY_GENES_MESSAGE_INTERACTIVE = (
        f"Enter at most {INTERACTIVE_MULTI_GENE_MAX_NAMES} gene names, separated by commas."
    )

    def redraw_gene(query, imputed):
        """Dispatches to redraw_single_gene (one gene name) or redraw_
        multi_genes (2-6, comma-separated — 'Gene1, Gene2[, ...Gene6]'; the
        first up to 3 show as red/green/blue circles, the next up to 3 as a
        red/green/blue '+' overlay on top — see redraw_multi_genes) based
        on how many comma-separated names are in `query`. An empty query
        falls through to redraw_single_gene('', imputed), which already has
        the right "enter a gene name" messaging for that case."""
        gene_names = parse_gene_names(query)
        if 2 <= len(gene_names) <= INTERACTIVE_MULTI_GENE_MAX_NAMES:
            redraw_multi_genes(gene_names, imputed)
            return
        if len(gene_names) > INTERACTIVE_MULTI_GENE_MAX_NAMES:
            status_text.set_text(TOO_MANY_GENES_MESSAGE_INTERACTIVE)
            working_text.set_text(f"At most {INTERACTIVE_MULTI_GENE_MAX_NAMES} genes")
            pending_param_warning['active'] = True
            blit_hover_overlays()
            return
        redraw_single_gene(gene_names[0] if gene_names else '', imputed)

    def redraw_single_gene(query, imputed):
        # Every early return below only ever updates status_text and
        # working_text (both animated — see their own comments) and touches
        # nothing else, so it blits instead of paying for a full fig.canvas.
        # draw_idle(). Only the real success path further down (a new UMAP
        # scatter, recolored section panels) needs — and does — a full draw.
        #
        # working_text's short message (also set in each branch below) is
        # the same "why nothing changed" explanation as status_text's own,
        # just brief enough for the narrow sidebar slot it otherwise shows
        # "Working…" in — the two roles never overlap in time, since
        # show_working_indicator() is never reached until *past* every one
        # of these early returns. It persists (survives being re-stamped by
        # cache_blit_background after later draws) until the next redraw_
        # gene() call overwrites it — either with a fresh "Working…" once
        # the problem's fixed, or a new explanation if it isn't yet.
        query = query.strip()
        if not query:
            status_text.set_text("Enter a gene name.")
            working_text.set_text("Enter a gene name")
            pending_param_warning['active'] = True
            blit_hover_overlays()
            return
        if imputed:
            if not ensure_imputed_gene_dataset_loaded(imputed_state, abc_cache):
                status_text.set_text("Imputed dataset not available.")
                working_text.set_text("Imputed data unavailable")
                pending_param_warning['active'] = True
                blit_hover_overlays()
                return
            gene_source = imputed_state['adata']
        elif not has_counts:
            status_text.set_text("No raw counts available on this dataset (layers['counts'] missing).")
            working_text.set_text("No raw counts available")
            pending_param_warning['active'] = True
            blit_hover_overlays()
            return
        else:
            gene_source = adata

        (gene_col,), (resolved_gene,), missing = resolve_gene_names(gene_source, [query])
        if missing:
            status_text.set_text(
                f"Gene '{query}' not found in the {'imputed' if imputed else 'standard'} dataset."
            )
            working_text.set_text(f"Gene '{query}' not found")
            # Right in the box itself, not just in status_text/working_text
            # below the plot — covers both a misspelling and a gene that's
            # only valid in the *other* Gene/Imputed Gene dataset (e.g.
            # typed while in 'Gene' mode but it's actually an imputed-only
            # gene). Cleared on the next edit (update_gene_suggestions) or
            # the next successful resolution.
            mark_invalid_gene_names(query_textbox, missing)
            pending_param_warning['active'] = True
            blit_hover_overlays()
            return
        clear_gene_name_marks(query_textbox)
        # Query was matched case-insensitively (find_gene_index) — replace
        # whatever casing the user typed with the canonical form from the
        # var table, the same spelling shown on the colorbar label/axes
        # title below, so the box always reflects what's actually on
        # screen. set_query_text (not a plain set_val) so this doesn't fire
        # a second on_submit/redraw on top of the one already in progress.
        # Also updates query_by_mode directly, not just the textbox, so
        # switching modes/levels away and back remembers the canonical
        # spelling rather than reverting to whatever was originally typed.
        if query_textbox.text != resolved_gene:
            set_query_text(resolved_gene)
        query_by_mode[query_state_key()] = resolved_gene
        # From here on this is the genuinely heavy path — resolving the
        # full gene column (a real disk read the first time, for the
        # imputed dataset), recoloring every section panel, redrawing the
        # UMAP scatter — same "give some feedback before a non-interruptible
        # block" reasoning as end_zoom_previews's own show_working_
        # indicator() call.
        show_working_indicator()

        # full_source/full_gene_col/full_layer/full_transform: what
        # set_section_colors_by_value (below) should use to color section-
        # panel cells *outside* the ROI, which color_values itself never
        # covers (it's aligned with adata's own ROI-filtered cells only) —
        # see that function's own comment.
        if imputed:
            # Imputed dataset isn't guaranteed to share adata's cell
            # order/set — matched by cell ID, same as render_gene_
            # expression_array's own expr_adata path. Already log2 on
            # disk (see that function's docstring on color_scale) — shown
            # as-is, no further transform. It's genome-wide by construction
            # (not ROI-filtered), so it already covers the whole section on
            # its own — no separate background source needed.
            full_col = _get_full_gene_column(gene_source, gene_col)
            positions = gene_source.obs_names.get_indexer(adata.obs_names)
            found = positions >= 0
            color_values = np.full(adata.n_obs, np.nan, dtype=float)
            color_values[found] = full_col[positions[found]]
            background_full_source, background_full_gene_col = gene_source, gene_col
            background_full_layer = background_full_transform = None
        else:
            expr = adata.layers['counts'][:, gene_col]
            if hasattr(expr, 'toarray'):
                expr = expr.toarray()
            color_values = np.log1p(np.asarray(expr).ravel().astype(float))
            # adata (ROI-filtered) has no counts for cells outside the ROI
            # at all — adata_backed (the whole, unfiltered section) is the
            # only place to get them, if it was even given (see this
            # window's own adata_backed=None default). Read from .X, not a
            # 'counts' layer: adata.layers['counts'] only exists on the
            # ROI-filtered `adata` (copied there once, after materializing
            # the subset — see materialize_subset's own callers), never on
            # adata_backed itself, whose raw counts live directly in .X
            # (backed mode, never normalized/transformed in place). Passing
            # layer='counts' here always raised inside _get_full_gene_column
            # (silently caught, background falling back to unknown-color for
            # every non-ROI cell), which is why gene-expression coloring
            # outside the ROI never worked for MERFISH genes even though the
            # exact same mechanism worked fine for imputed genes (whose
            # source AnnData is read from .X already, no layer involved).
            # Re-resolved by gene *symbol* here, not assumed to share
            # adata's own var index — adata_backed isn't guaranteed to have
            # been derived the same way.
            background_full_source = background_full_gene_col = None
            background_full_layer, background_full_transform = None, np.log1p
            if has_background and adata_backed is not None:
                bg_gene_col = find_gene_index(adata_backed, query)
                if bg_gene_col is not None:
                    background_full_source, background_full_gene_col = adata_backed, bg_gene_col

        clear_colorbar()
        ax.clear()
        invalidate_home_view_cache(ax)  # about to render different content — see that function
        umap_label_state['artists'] = []  # ax.clear() just removed any centroid labels
        recreate_umap_highlight()
        recreate_group_highlight()
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(pannable_axes[ax]['home_xlim'])
        ax.set_ylim(pannable_axes[ax]['home_ylim'])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('UMAP1', fontsize=SIDEBAR_FONTSIZE)
        ax.set_ylabel('UMAP2', fontsize=SIDEBAR_FONTSIZE)
        order = np.argsort(color_values)  # highest-expressing cells drawn last/on top
        scatter = ax.scatter(coords[order, 0], coords[order, 1], c=color_values[order],
                              cmap='viridis', s=UMAP_POINT_SIZE, alpha=UMAP_POINT_ALPHA,
                              linewidths=0)
        set_main_scatter_artists([scatter])
        set_section_colors_by_value(
            color_values, scatter.cmap, scatter.norm,
            full_source=background_full_source, full_gene_col=background_full_gene_col,
            full_layer=background_full_layer, full_transform=background_full_transform,
        )
        update_umap_dot_size()
        # cax=cbar_ax draws into that pre-made, fixed-size axes directly,
        # instead of ax=ax (which carves a new colorbar axes *out of* ax,
        # shrinking it) — ax's own size and position are never touched by
        # this, in either direction.
        cbar_ax.set_visible(True)
        colorbar_state['cbar'] = fig.colorbar(scatter, cax=cbar_ax)
        colorbar_state['cbar'].set_label(resolved_gene, fontsize=SIDEBAR_FONTSIZE)
        colorbar_state['cbar'].ax.tick_params(labelsize=SIDEBAR_FONTSIZE)
        ax.set_title(f"{resolved_gene}{' [imputed]' if imputed else ''}", fontsize=SIDEBAR_FONTSIZE)
        current_view_name['token'] = f"{resolved_gene}{'_imputed' if imputed else ''}"
        status_text.set_text(f"Showing '{resolved_gene}'{' [imputed]' if imputed else ' (raw counts, log1p)'}.")
        fig.canvas.draw_idle()
        hide_working_indicator()

    def resolve_gene_expression(gene_source, gene_col, imputed):
        """The same per-cell log-expression extraction redraw_single_gene's
        own MERFISH/imputed branches use, factored out so redraw_multi_
        genes can call it once per gene (2 or 3) without duplicating it.
        Returns (color_values, full_source, full_gene_col, full_layer,
        full_transform) — the last four are exactly what set_section_
        colors_by_value/set_section_colors_multi_gene need to also color
        panel cells *outside* the ROI, which color_values itself never
        covers (aligned with adata's own ROI-filtered cells only) — see
        that function's own comment."""
        if imputed:
            full_col = _get_full_gene_column(gene_source, gene_col)
            positions = gene_source.obs_names.get_indexer(adata.obs_names)
            found = positions >= 0
            color_values = np.full(adata.n_obs, np.nan, dtype=float)
            color_values[found] = full_col[positions[found]]
            return color_values, gene_source, gene_col, None, None
        expr = adata.layers['counts'][:, gene_col]
        if hasattr(expr, 'toarray'):
            expr = expr.toarray()
        color_values = np.log1p(np.asarray(expr).ravel().astype(float))
        full_source = full_gene_col = None
        if has_background and adata_backed is not None:
            resolved_symbol = str(gene_source.var['gene_symbol'].iloc[gene_col])
            bg_gene_col = find_gene_index(adata_backed, resolved_symbol)
            if bg_gene_col is not None:
                full_source, full_gene_col = adata_backed, bg_gene_col
        return color_values, full_source, full_gene_col, None, np.log1p

    def normalize_expression(values):
        """Min-max scales `values` to [0, 1] (NaN preserved — multi_gene_
        rgb is what turns those into 'treat as 0 expression'), for the
        multi-gene overlay's own red/green/blue channel inputs. A flat gene
        (every cell the same value, including the degenerate all-zero
        case) has no meaningful spread to show — comes back all zeros
        rather than dividing by a zero range."""
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        if vmax > vmin:
            return (values - vmin) / (vmax - vmin), vmin, vmax
        return np.zeros_like(values), vmin, vmax

    def redraw_multi_genes(gene_queries, imputed):
        """Multi-gene overlay (2-6 genes): `gene_queries[0]` drives the red
        channel, `gene_queries[1]` green, `gene_queries[2]` blue, all drawn
        as circles (the base layer) — blended via multi_gene_rgb from
        MULTI_GENE_LOW_EXPRESSION_COLOR (black, shared by the UMAP and the
        section panels) toward each channel's own full saturation as that
        gene's own normalized expression rises. `gene_queries[3:6]`, if
        given, repeat the same red/green/blue blend as a second '+'-marker
        layer drawn on top (UMAP only, for now — see the '+' overlay layer's
        own comment below for why zero-expression cells there simply aren't
        plotted). The UMAP's own axes background is set to MULTI_GENE_UMAP_
        FACECOLOR (dark grey, not black) so a genuinely zero-expression
        circle still reads as a dot — see that constant's own comment.
        Mirrors redraw_single_gene's own structure (same early-return
        messaging conventions, same show_working_indicator timing, same
        background-extension handling via resolve_gene_expression), but
        can't reuse its scatter(cmap=...)/set_section_colors_by_value calls
        — those are built around one scalar value plus a matplotlib
        Colormap/Normalize pair, and this has no such single scale (colors
        are computed directly by multi_gene_rgb from up to 3 values apiece)."""
        if imputed:
            if not ensure_imputed_gene_dataset_loaded(imputed_state, abc_cache):
                status_text.set_text("Imputed dataset not available.")
                working_text.set_text("Imputed data unavailable")
                pending_param_warning['active'] = True
                blit_hover_overlays()
                return
            gene_source = imputed_state['adata']
        elif not has_counts:
            status_text.set_text("No raw counts available on this dataset (layers['counts'] missing).")
            working_text.set_text("No raw counts available")
            pending_param_warning['active'] = True
            blit_hover_overlays()
            return
        else:
            gene_source = adata

        gene_cols, resolved, missing = resolve_gene_names(gene_source, gene_queries)
        if missing:
            status_text.set_text(
                f"Gene(s) not found in the {'imputed' if imputed else 'standard'} dataset: {', '.join(missing)}."
            )
            working_text.set_text("Gene not found")
            # See the single-gene redraw's identical comment, above. Only the
            # names that didn't resolve turn red, not the whole box.
            mark_invalid_gene_names(query_textbox, missing)
            pending_param_warning['active'] = True
            blit_hover_overlays()
            return
        clear_gene_name_marks(query_textbox)
        combined_query = ', '.join(resolved)
        if query_textbox.text != combined_query:
            set_query_text(combined_query)
        query_by_mode[query_state_key()] = combined_query

        show_working_indicator()

        # One resolve_gene_expression/normalize_expression call per gene —
        # each returns (values, bg_source, bg_col, bg_layer, bg_transform)
        # / (norm, vmin, vmax); unzipped into parallel lists (one entry per
        # gene) below, since set_section_colors_multi_gene and multi_gene_
        # rgb both want the whole set at once, not one gene at a time.
        per_gene = [resolve_gene_expression(gene_source, c, imputed) for c in gene_cols]
        bg_sources = [g[1] for g in per_gene]
        bg_cols = [g[2] for g in per_gene]
        bg_layers = [g[3] for g in per_gene]
        bg_transforms = [g[4] for g in per_gene]
        normalized = [normalize_expression(g[0]) for g in per_gene]
        norms = [n[0] for n in normalized]
        vmins = [n[1] for n in normalized]
        vmaxes = [n[2] for n in normalized]

        clear_colorbar()
        ax.clear()
        invalidate_home_view_cache(ax)  # about to render different content — see that function
        umap_label_state['artists'] = []  # ax.clear() just removed any centroid labels
        recreate_umap_highlight()
        recreate_group_highlight()
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor(MULTI_GENE_UMAP_FACECOLOR)
        # Same not-quite-black background as the UMAP, on every section
        # panel too — clear_colorbar() (already run above) just reset them
        # to the normal SECTION_PANEL_FACECOLOR (pure black); overridden
        # back to the shared dark grey here so a genuinely zero-expression
        # cell (itself black, via MULTI_GENE_LOW_EXPRESSION_COLOR) stays
        # visibly a cell instead of disappearing into an identically-black
        # background.
        set_section_panel_facecolor(MULTI_GENE_UMAP_FACECOLOR)
        ax.set_xlim(pannable_axes[ax]['home_xlim'])
        ax.set_ylim(pannable_axes[ax]['home_ylim'])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('UMAP1', fontsize=SIDEBAR_FONTSIZE)
        ax.set_ylabel('UMAP2', fontsize=SIDEBAR_FONTSIZE)
        # Circle layer: the first up-to-3 genes (red/green/blue), same as
        # before genes 4-6 existed. Highest-*combined*-expression drawn
        # last/on top — same "most interesting cells visible over
        # everything else" convention as redraw_single_gene's own
        # np.argsort(color_values), generalized to however many of the
        # up-to-3 circle genes are actually present.
        circle_norms = norms[:3]
        circle_intensity = np.maximum.reduce([np.nan_to_num(n, nan=0.0) for n in circle_norms])
        circle_order = np.argsort(circle_intensity)
        umap_rgb = multi_gene_rgb([n[circle_order] for n in circle_norms], MULTI_GENE_LOW_EXPRESSION_COLOR)
        scatter = ax.scatter(coords[circle_order, 0], coords[circle_order, 1], c=umap_rgb,
                              s=UMAP_POINT_SIZE, alpha=UMAP_POINT_ALPHA, linewidths=0, zorder=1)
        main_artists = [scatter]
        main_size_multipliers = [1.0]

        # '+' overlay layer: genes 4-6 (red/green/blue again), UMAP only for
        # now — section panels below still only ever see the circle-layer
        # genes (see the sliced norms[:3]/vmins[:3]/... a few lines down).
        # Drawn as a second, separate scatter on top of the circle layer
        # (not blended into umap_rgb above) — a small, opaque (alpha=1),
        # filled circle centered on each cell, half the base circle's own
        # diameter, so the base layer's own color still shows around its
        # edges. A cell with exactly zero expression across all three of
        # these genes is simply left out of this scatter entirely, rather
        # than getting an opaque black dot stamped over an otherwise-
        # untouched circle — see MULTI_GENE_PLUS_ZORDER's own comment for
        # the zero-expression-baseline reasoning.
        plus_norms = norms[3:6]
        if plus_norms:
            plus_intensity = np.maximum.reduce([np.nan_to_num(n, nan=0.0) for n in plus_norms])
            visible = plus_intensity > 0
            plus_order = np.argsort(plus_intensity[visible])
            visible_idx = np.flatnonzero(visible)[plus_order]
            plus_rgb = multi_gene_rgb([n[visible_idx] for n in plus_norms], MULTI_GENE_LOW_EXPRESSION_COLOR)
            plus_scatter = ax.scatter(
                coords[visible_idx, 0], coords[visible_idx, 1], c=plus_rgb,
                marker='o', s=UMAP_POINT_SIZE, alpha=1.0,
                linewidths=0, zorder=MULTI_GENE_PLUS_ZORDER,
            )
            main_artists.append(plus_scatter)
            main_size_multipliers.append(MULTI_GENE_PLUS_SIZE_DIAMETER_MULTIPLIER ** 2)

        set_main_scatter_artists(main_artists, size_multipliers=main_size_multipliers)
        set_section_colors_multi_gene(
            norms[:3], MULTI_GENE_LOW_EXPRESSION_COLOR, vmins[:3], vmaxes[:3],
            full_sources=bg_sources[:3], full_gene_cols=bg_cols[:3],
            full_layers=bg_layers[:3], full_transforms=bg_transforms[:3],
        )
        update_umap_dot_size()
        # vmins/vmaxes are each in whatever base resolve_gene_expression
        # happened to log-transform that gene with — natural-log log1p for
        # MERFISH, already-log2 for imputed (see that function's own
        # comment) — converted to log2 units here purely for display, same
        # convention mean_log2_expression already established for export_
        # expression's own reported ranges, so the legend's numbers mean
        # what they say regardless of which source each gene came from.
        # Purely a display-side conversion: normalize_expression's own
        # min-max scaling is invariant to this positive linear factor, so
        # the actual on-screen colors are unaffected either way.
        to_log2 = 1.0 if imputed else np.log2(np.e)
        draw_multi_gene_legend(resolved, [v * to_log2 for v in vmins], [v * to_log2 for v in vmaxes])
        # Set *after* clear_colorbar (already run above) reset it to
        # inactive — this is what tells render_export_figure (Save UMAP)
        # to rebuild its own copy from this scatter's actual facecolors
        # instead of assuming a scalar cmap or a taxonomy category, neither
        # of which apply to a multi-gene view.
        multi_gene_state['active'] = True
        multi_gene_state['genes'] = resolved
        channel_names = MULTI_GENE_CHANNEL_COLORS[:len(resolved)]
        title_parts = ' / '.join(f'{g} ({c})' for g, c in zip(resolved, channel_names))
        ax.set_title(f"{title_parts}{' [imputed]' if imputed else ''}", fontsize=SIDEBAR_FONTSIZE)
        current_view_name['token'] = f"{'_'.join(resolved)}{'_imputed' if imputed else ''}"
        status_and_channels = ' and '.join(f"'{g}' ({c})" for g, c in zip(resolved, channel_names))
        status_text.set_text(
            f"Showing {status_and_channels}{' [imputed]' if imputed else ' (raw counts, log1p)'}."
        )
        fig.canvas.draw_idle()
        hide_working_indicator()

    def redraw():
        # Every redraw_* below starts with ax.clear(), which silently
        # orphans any zoom-preview stand-in image still registered for that
        # axes — the settle timer would then fire against artists that no
        # longer exist. Reachable in practice because the resize settle
        # (on_resize_settled -> maybe_redraw_for_legend_resize) can land
        # inside a zoom burst's own, longer settle window. Torn down here
        # instead, before anything clears.
        discard_zoom_previews()
        mode = mode_state['mode']
        if mode == 'All Subclasses':
            redraw_all_subclasses()
            return
        query = query_textbox.text
        query_by_mode[query_state_key()] = query  # remembered across mode (and level) switches
        if mode == 'Single Subclass':
            redraw_subclass(query)
        elif mode == 'Gene':
            redraw_gene(query, imputed=False)
        else:
            redraw_gene(query, imputed=True)

    def full_taxonomy_label(level, target_id):
        """The full descriptive string (e.g. '268 L5 PT CTX Glut') behind a
        typed leading-numeric ID at `level` — the same value shown in the
        hover info panel (cell_class/cell_subclass/etc.), not the bare
        number the user types into the query box. None if no cell in this
        run actually carries that ID."""
        ids_arr = level_ids_arrays[level]
        if ids_arr is None:
            return None
        match = np.nonzero(ids_arr.to_numpy() == target_id)[0]
        if len(match) == 0:
            return None
        return str(cell_level_arrays[level][match[0]])

    # Only exceeds the on-screen legend's own row cap (draw_id_legend's
    # MAX_ROW_HEIGHT_FRAC) when saving to file — save_current_umap builds a
    # fresh legend axes each time rather than reusing cbar_ax, so it isn't
    # bound by that fixed on-screen slot's width/height at all.
    EXPORT_LEGEND_MAX_ROWS_PER_COL = 35
    # Fixed, in points — deliberately *not* SIDEBAR_FONTSIZE (itself
    # derived from compute_ui_fontsize(), which reads the screen's own
    # pixel height at import time). A saved file's text should look the
    # same regardless of what screen it was saved from; only render_
    # export_figure (below) uses this, never anything in the live window.
    EXPORT_FONTSIZE = 14
    # The UMAP scatter's own plotting area (not the whole saved file, which
    # ends up a bit larger once axis labels/title/colorbar/legend are
    # added around it — see render_export_figure) comes out this many
    # pixels square at UMAP_SAVE_DPI, tying the two together so the target
    # stays correct even if UMAP_SAVE_DPI itself is ever changed.
    EXPORT_AXES_TARGET_PX = 6400
    EXPORT_AXES_SIZE_IN = EXPORT_AXES_TARGET_PX / UMAP_SAVE_DPI
    # Room around the core square for axis labels/title on every side, and
    # the colorbar/legend on the right — generous rather than exact, since
    # bbox_inches='tight' expands to whatever's actually drawn regardless
    # of this figure's own nominal size (a wide multi-column legend, in
    # particular, routinely needs more than this).
    EXPORT_MARGIN_IN = 2.5

    def legend_entry_sort_key(entry):
        # Sorts by the same leading numeric ID extract_leading_numeric_id
        # pulls from a category string ('30 Astro-Epen' -> 30) — a plain
        # string sort would put '10' before '2', which reads as random
        # once the IDs cross into double/triple digits. Entries with no
        # leading number at all (shouldn't normally happen — every ABC
        # taxonomy category string starts with one) sort last rather than
        # erroring.
        match = re.match(r'^(\d+)', entry[1])
        return int(match.group(1)) if match else math.inf

    def build_export_legend_entries(sort_by='id'):
        """(title, [(color, label, count, marker, is_open), ...]) describing
        the current mode's categorical color+shape assignment, full
        descriptive names rather than the bare IDs/short labels the
        on-screen legend uses (that one has to fit cbar_ax's narrow slot —
        see draw_id_legend's own comment; a saved file has no such
        constraint). marker/is_open come from category_rank_shape, keyed
        by each entry's rank in *assignment* order (typed order for Single
        Subclass, cell-count order for All Subclasses — the same order
        category_rank_color/target_colors/umap_color_map were built in,
        which is what the actual color+shape on the plot depend on) —
        computed before the sort_by-driven sort below, since re-sorting
        for display afterward must not disturb which shape belongs to
        which entry.

        sort_by='id' (the combined plot's own embedded legend) sorts by
        leading numeric ID; 'count' (the standalone legend-only file — see
        render_export_legend_figure) sorts by cell count, largest first.

        None for a mode with nothing categorical to show (Gene/Imputed
        Gene already has its own colorbar label, which already is the
        full gene name)."""
        mode = mode_state['mode']
        if mode == 'Single Subclass':
            level = last_id_selection['level']
            targets = last_id_selection['targets']
            colors = last_id_selection['colors']
            counts = last_id_selection['counts']
            if level is None or not targets or not colors:
                return None
            entries = []
            for rank, t in enumerate(targets):
                color = colors.get(t)
                if color is None:
                    continue  # not found among this run's cells — no color was ever assigned
                label = full_taxonomy_label(level, t) or f"{LEVEL_DISPLAY_NAMES[level]} {t}"
                marker, is_open = category_rank_shape(rank)
                entries.append((color, label, counts.get(t, 0), marker, is_open))
            title = LEVEL_DISPLAY_NAMES[level]
        elif mode == 'All Subclasses':
            level = level_state['value']
            info = compute_ranked_category_colors(level)
            color_map = info['umap_color_map']  # dict order == rank order, largest first
            counts = info['counts']
            entries = []
            for rank, (cat, color) in enumerate(color_map.items()):
                marker, is_open = category_rank_shape(rank)
                entries.append((color, str(cat), int(counts[cat]), marker, is_open))
            title = LEVEL_DISPLAY_NAMES[level]
        else:
            return None  # Gene / Imputed Gene — the colorbar's own label already names the color scale
        if not entries:
            return None
        if sort_by == 'count':
            entries.sort(key=lambda entry: entry[2], reverse=True)
        else:
            entries.sort(key=legend_entry_sort_key)
        return title, entries

    def build_legend_handles(entries):
        """Line2D proxy handles (empty data, marker only — the standard
        way to get a legend handle with an arbitrary marker) for
        build_export_legend_entries' own (color, label, count, marker,
        is_open) tuples. Shared by render_export_figure's embedded legend
        and render_export_legend_figure's standalone one, so the two
        always render entries identically."""
        return [
            Line2D(
                [], [], marker=marker, linestyle='none', markersize=9,
                markerfacecolor=('none' if is_open else color),
                markeredgecolor=color, markeredgewidth=(1.2 if is_open else 0.5),
                label=f"{label}  ({count} cells)",
            )
            for color, label, count, marker, is_open in entries
        ]

    def render_export_legend_figure():
        """A brand-new, off-screen Figure containing *only* the
        categorical legend — nothing else on it, sorted by cell count
        (largest first) rather than the combined plot's own embedded
        legend (sorted by leading numeric ID) — for save_current_umap's
        additional standalone legend file. None for a mode with no
        categorical legend to show (see build_export_legend_entries)."""
        legend_info = build_export_legend_entries(sort_by='count')
        if legend_info is None:
            return None
        title, entries = legend_info
        handles = build_legend_handles(entries)
        ncol = max(1, math.ceil(len(handles) / EXPORT_LEGEND_MAX_ROWS_PER_COL))
        legend_fig = Figure()
        FigureCanvasAgg(legend_fig)  # attaches itself as legend_fig.canvas; never touches Tk
        # loc='center'/bbox_to_anchor omitted (unlike render_export_
        # figure's own legend): with no axes on this figure at all, the
        # legend just centers in the whole (otherwise-empty) figure —
        # bbox_inches='tight' at save time crops down to exactly its own
        # content regardless of this figure's nominal starting size.
        legend_fig.legend(
            handles=handles, title=title, loc='center',
            fontsize=EXPORT_FONTSIZE * 0.85, title_fontsize=EXPORT_FONTSIZE,
            ncol=ncol, frameon=True, handlelength=1.0, handleheight=1.0,
            labelspacing=0.4, columnspacing=1.5,
        )
        return legend_fig

    def render_export_figure():
        """A brand-new, off-screen Figure reproducing exactly what's
        currently shown in `ax` — scatter, limits, labels, title, plus its
        colorbar (Gene/Imputed Gene) or legend (categorical modes) — for
        save_current_umap to write straight to disk.

        Built via Figure()+FigureCanvasAgg (same headless pattern already
        used elsewhere in this file, e.g. render_gene_expression_array),
        never plt.figure()/pyplot — this never touches Tk and is never
        shown, so nothing about it (or the saved file that comes from it)
        depends on this window's own screen-relative size the way cropping
        the *live* `fig` did (see compute_figsize_for_screen_height, which
        sizes the on-screen window/figure from the screen's own pixel
        height — the actual root cause: the live figure, and therefore
        everything cropped out of it, came out a different physical size
        in inches depending on what screen the window happened to open on,
        even at the same fixed DPI). Also sidesteps the sidebar/resize-bar
        overlap problem an earlier version of this function had to
        explicitly hide-then-restore for: this figure never has sidebar
        artists on it in the first place, so there's nothing to hide.

        Reads the live scatter's own already-computed offsets/colors/sizes
        rather than recomputing them, so this is a faithful copy of what's
        on screen right now, not a fresh derivation that could drift from
        it. Every font here uses the fixed EXPORT_FONTSIZE, not SIDEBAR_
        FONTSIZE — the same screen-dependence that motivated the figure
        rebuild in the first place would otherwise have leaked right back
        in through the text."""
        scatter = main_scatter_state['artist']
        offsets = scatter.get_offsets()
        # export_ax is given an *exact* physical size (EXPORT_AXES_SIZE_IN
        # square) rather than a fraction of some arbitrary default figure
        # — sizing it as a fraction of a fixed total figsize would make
        # the actual pixel count depend on that arbitrary total, which is
        # exactly the kind of incidental dependence this function exists
        # to avoid (see its own docstring). export_total_size_in only sets
        # where the core square sits within the figure, not its own size —
        # bbox_inches='tight' at save time crops to the real content
        # regardless.
        export_total_size_in = EXPORT_AXES_SIZE_IN + 2 * EXPORT_MARGIN_IN
        export_fig = Figure(figsize=(export_total_size_in, export_total_size_in))
        FigureCanvasAgg(export_fig)  # attaches itself as export_fig.canvas; never touches Tk
        axes_frac = EXPORT_AXES_SIZE_IN / export_total_size_in
        margin_frac = EXPORT_MARGIN_IN / export_total_size_in
        export_ax = export_fig.add_axes([margin_frac, margin_frac, axes_frac, axes_frac])
        export_ax.set_aspect('equal', adjustable='box')
        export_ax.set_xlim(ax.get_xlim())
        export_ax.set_ylim(ax.get_ylim())
        export_ax.set_xticks([])
        export_ax.set_yticks([])
        export_ax.set_xlabel('UMAP1', fontsize=EXPORT_FONTSIZE)
        export_ax.set_ylabel('UMAP2', fontsize=EXPORT_FONTSIZE)
        export_ax.set_title(ax.get_title(), fontsize=EXPORT_FONTSIZE)
        values = scatter.get_array()
        if multi_gene_state['active']:
            # Multi-gene overlay — the live scatter's `c=` was already an
            # explicit per-point RGB(A) array (multi_gene_rgb's own
            # output), not a scalar + cmap/norm, so there's no
            # ScalarMappable to rebuild a colorbar from the way the
            # continuous branch below does. get_facecolor() reads back
            # exactly what was actually rendered — including UMAP_POINT_
            # ALPHA already baked into its own 4th channel by the live
            # scatter's own alpha= (matplotlib sets, not multiplies, a
            # collection-wide alpha onto each point's stored color) — so
            # alpha=None here leaves that as-is instead of re-overriding it
            # a second time. The live axes' own dark-grey background
            # (MULTI_GENE_UMAP_FACECOLOR) is copied too, not left at
            # export_ax's own default white — a zero-expression dot is
            # deliberately near-black against that background, not white.
            export_ax.set_facecolor(ax.get_facecolor())
            export_ax.scatter(offsets[:, 0], offsets[:, 1], c=scatter.get_facecolor(),
                               s=scatter.get_sizes(), alpha=None, linewidths=0)
            # A plain per-gene legend, not draw_multi_gene_legend's own
            # cbar_ax-based one — export_fig has no such axes (this figure
            # never has sidebar artists at all, see this function's own
            # docstring), so a real matplotlib legend (Line2D proxies, the
            # standard way to hand it markers/colors with no actual plotted
            # artist to point to) fills the equivalent role here.
            legend_handles = [
                Line2D([0], [0], marker='o', linestyle='', color=color, label=f'{gene} ({color})')
                for gene, color in zip(multi_gene_state['genes'], MULTI_GENE_CHANNEL_COLORS)
            ]
            export_fig.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
                               bbox_transform=export_ax.transAxes, fontsize=EXPORT_FONTSIZE, frameon=True)
        elif values is not None:
            # Continuous (Gene/Imputed Gene) — re-plotted from the same
            # scalar values + cmap/norm the live scatter used, not its
            # already-resolved RGBA facecolors, so a real colorbar can be
            # attached below (a colorbar needs an actual ScalarMappable;
            # per-point RGBA alone isn't one — see the categorical branch,
            # which has the opposite problem: no scalar values to speak
            # of, only a legend to explain the colors).
            export_scatter = export_ax.scatter(
                offsets[:, 0], offsets[:, 1], c=values, cmap=scatter.cmap, norm=scatter.norm,
                s=scatter.get_sizes(), alpha=scatter.get_alpha(), linewidths=0,
            )
            export_cbar = export_fig.colorbar(export_scatter, ax=export_ax)
            export_cbar.set_label(cbar_ax.get_ylabel(), fontsize=EXPORT_FONTSIZE)
            export_cbar.ax.tick_params(labelsize=EXPORT_FONTSIZE)
        else:
            # Categorical (Single/All Subclasses) — recomputed per-cell
            # here via the same category_of_point_from_mapping/plot_
            # categorical_umap helpers redraw_subclass/redraw_all_
            # subclasses themselves use, rather than read back from the
            # live scatter's own facecolors: the live scatter's own point
            # order is permuted (colored cells drawn last/on top), with no
            # way to recover per-point category membership from the
            # scatter object alone. The color *mappings* passed in
            # (last_id_selection['colors'] / compute_ranked_category_
            # colors's own umap_color_map) are the exact same ones those
            # two functions used to color the live scatter — not
            # recomputed via category_rank_color here — so a single ID's
            # special-cased plain red (see redraw_subclass) still comes
            # out red, not rank 0's actual cycle color. use_shapes=True
            # unconditionally: the export always uses shapes regardless of
            # UMAP_USE_SHAPES_ON_SCREEN, which only affects the live view.
            mode = mode_state['mode']
            if mode == 'Single Subclass':
                level = last_id_selection['level']
                ids_arr = (level_ids_arrays[level].to_numpy()
                           if level is not None and level_ids_arrays[level] is not None else None)
                category_of_point, rank_color = (
                    category_of_point_from_mapping(ids_arr, last_id_selection['colors'])
                    if ids_arr is not None else (np.full(len(coords), -1), {})
                )
            else:  # 'All Subclasses'
                level = level_state['value']
                if cell_level_arrays[level] is not None:
                    obs_values = adata.obs[level].to_numpy()
                    color_map = compute_ranked_category_colors(level)['umap_color_map']
                    category_of_point, rank_color = category_of_point_from_mapping(obs_values, color_map)
                else:
                    category_of_point, rank_color = np.full(len(coords), -1), {}
            plot_categorical_umap(export_ax, category_of_point, rank_color,
                                   scatter.get_sizes()[0], scatter.get_alpha(), True)
        # 'All Subclasses' mode's per-centroid ID number labels
        # (add_umap_centroid_labels) — separate Text artists on the live
        # `ax`, not part of the scatter itself, so they need their own
        # copy here too. Each one's *current* fontsize (already rescaled
        # for whatever zoom level is showing right now — see update_umap_
        # dot_size) is read back directly rather than recomputed, same
        # "faithful copy of what's on screen" reasoning as the scatter.
        for label in umap_label_state['artists']:
            lx, ly = label.get_position()
            export_ax.text(
                lx, ly, label.get_text(), fontsize=label.get_fontsize(),
                color=label.get_color(), fontweight=label.get_fontweight(),
                ha=label.get_ha(), va=label.get_va(), clip_on=True,
            )
        legend_info = build_export_legend_entries()
        if legend_info is not None:
            title, entries = legend_info
            # Each handle's own marker/fill mirrors exactly what's actually
            # plotted for that category (see the categorical branch above).
            handles = build_legend_handles(entries)
            ncol = max(1, math.ceil(len(handles) / EXPORT_LEGEND_MAX_ROWS_PER_COL))
            export_fig.legend(
                handles=handles, title=title, loc='center left',
                bbox_to_anchor=(1.02, 0.5), bbox_transform=export_ax.transAxes,
                fontsize=EXPORT_FONTSIZE * 0.85, title_fontsize=EXPORT_FONTSIZE,
                ncol=ncol, frameon=True, handlelength=1.0, handleheight=1.0,
                labelspacing=0.4, columnspacing=1.5,
            )
        return export_fig

    def open_data_folder(event=None):
        """Opens this run's own output folder (or CACHE_DIR, same fallback
        every other save/export button here uses when run_folder wasn't
        given) in the OS's file browser — Windows Explorer, via
        open_with_default_viewer's os.startfile branch. A plain folder
        launch, not a save/export — nothing to compute, so no working
        indicator or status_text update, just the OS call and a console
        line for a record of what happened if it silently fails."""
        target_dir = Path(run_folder) if run_folder is not None else CACHE_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        open_with_default_viewer(target_dir)
        print(f"Opened {target_dir} in the file browser.")

    def save_current_umap(event=None):
        """Write the UMAP panel to PNG and SVG in this run's own folder,
        named after whatever it's currently showing (see
        current_view_name) — rendered via render_export_figure (above),
        not by cropping the live, on-screen `fig` directly, so the saved
        file's pixel dimensions depend only on fixed constants
        (EXPORT_FONTSIZE, UMAP_SAVE_DPI) and the content itself, never on
        this window's own screen-relative size.

        For a categorical mode, also writes a third, legend-only PNG
        (render_export_legend_figure) — sorted by cell count rather than
        the combined plot's own embedded legend (by leading numeric ID),
        since "which categories are actually the big ones" is usually the
        more useful ordering to skim on its own, separate from the plot."""
        target_dir = Path(run_folder) if run_folder is not None else CACHE_DIR
        token = sanitized_view_token()
        show_working_indicator('Saving…')
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            export_fig = render_export_figure()
            written = []
            for suffix in ('.png', '.svg'):
                path = target_dir / f'umap_{token}{suffix}'
                export_fig.savefig(path, bbox_inches='tight',
                                    dpi=UMAP_SAVE_DPI if suffix == '.png' else None)
                written.append(path)
            legend_fig = render_export_legend_figure()
            if legend_fig is not None:
                legend_path = target_dir / f'umap_{token}_legend.png'
                legend_fig.savefig(legend_path, bbox_inches='tight', dpi=UMAP_SAVE_DPI)
                written.append(legend_path)
            names = ', '.join(p.name for p in written)
            status_text.set_text(f"Saved {names} to {target_dir}.")
            print(f"Saved UMAP to {', '.join(str(p) for p in written)}.")
        except Exception as e:
            status_text.set_text(f"Could not save the UMAP: {e}")
            print(f"Could not save the UMAP: {e}")
        hide_working_indicator()
        # status_text is animated, so a blit is what actually puts the
        # message on screen — no full redraw needed just to report this.
        blit_hover_overlays()

    def render_section_maps_export_figure():
        """A brand-new, off-screen Figure reproducing the section-panel grid
        exactly as currently displayed — each panel's own pan/zoom, current
        coloring, and ROI rectangles, but not the hover ring/family
        highlight/dimming veil, same exclusions save_current_umap's own
        render_export_figure makes for the UMAP (a hover artifact isn't
        part of "the section maps" as a deliverable). Unlike that function,
        this deliberately *keeps* the live window's own current physical
        size and shape in the result — every panel comes out at exactly the
        size it currently has on screen, in inches — rather than a fixed,
        screen-independent size: since every panel can be independently
        panned/zoomed, there's no single "current view" to define a
        canonical size from the way the one UMAP scatter has, so matching
        what's actually on screen right now (still true to a real,
        physical scale via the scale bar) is the closest thing to a
        faithful "as the user currently sees it" export — the same goal
        the screenshot this replaces was already going for.

        Built via Figure()+FigureCanvasAgg (same headless pattern as
        render_export_figure — see its own docstring), so this never
        touches Tk and is never shown."""
        crop_x0, crop_y0, crop_x1, crop_y1 = GRID_LEFT, AREA_BOTTOM, GRID_RIGHT, 1.0
        crop_w_frac, crop_h_frac = crop_x1 - crop_x0, crop_y1 - crop_y0
        live_w_in, live_h_in = fig.get_size_inches()
        export_fig = Figure(figsize=(crop_w_frac * live_w_in, crop_h_frac * live_h_in))
        FigureCanvasAgg(export_fig)  # attaches itself as export_fig.canvas; never touches Tk

        for panel in section_panels.values():
            live_ax = panel['ax']
            pos = live_ax.get_position()
            # This panel's live box, remapped from whole-figure fractions
            # into fractions of just the crop region — since export_fig's
            # own size *is* that crop region's own physical size (above),
            # this reproduces the panel at the exact same absolute size (in
            # inches) it currently has on screen, not merely the same
            # proportions.
            new_ax = export_fig.add_axes([
                (pos.x0 - crop_x0) / crop_w_frac, (pos.y0 - crop_y0) / crop_h_frac,
                pos.width / crop_w_frac, pos.height / crop_h_frac,
            ])
            new_ax.set_facecolor(live_ax.get_facecolor())
            new_ax.set_aspect('equal', adjustable='box')
            new_ax.set_xlim(live_ax.get_xlim())
            new_ax.set_ylim(live_ax.get_ylim())
            new_ax.set_xticks([])
            new_ax.set_yticks([])
            for live_spine, new_spine in zip(live_ax.spines.values(), new_ax.spines.values()):
                new_spine.set_linewidth(live_spine.get_linewidth())
            new_ax.set_title(live_ax.get_title(), fontsize=live_ax.title.get_fontsize())

            bg = panel['background_artist']
            offsets, facecolors, sizes = visible_points_only(
                bg.get_offsets(), bg.get_facecolors(), bg.get_sizes(),
                live_ax.get_xlim(), live_ax.get_ylim(),
            )
            new_ax.scatter(offsets[:, 0], offsets[:, 1], c=facecolors, s=sizes, linewidths=0)

            # ROI rectangles only — panel['ax'].patches also holds dim_veil
            # (the hover-dimming overlay), deliberately excluded here along
            # with the highlight/group_highlight scatters (never copied at
            # all, since nothing above reads them).
            for roi_patch in live_ax.patches:
                if roi_patch is panel['dim_veil']:
                    continue
                new_ax.add_patch(Rectangle(
                    roi_patch.get_xy(), roi_patch.get_width(), roi_patch.get_height(),
                    edgecolor=roi_patch.get_edgecolor(), facecolor=roi_patch.get_facecolor(),
                    linestyle=roi_patch.get_linestyle(), linewidth=roi_patch.get_linewidth(),
                ))

            # One scale bar total (see build_section_scalebar) — added here,
            # not copied, since a static export never needs the live one's
            # ability to track a still-changing view.
            if section_scalebar is not None and live_ax is section_scalebar['ax']:
                build_section_scalebar(new_ax, fontsize=section_scalebar['text'].get_fontsize())
        return export_fig

    def save_section_maps(event=None):
        """Write the section-panel grid to PNG and SVG in this run's own
        folder, named after whatever it's currently showing (see
        sanitized_view_token) — rendered via render_section_maps_export_
        figure (above), so both are real vector files (a scale bar included
        as one, not a rasterized overlay), not a screenshot.

        Since every panel's pan/zoom is independent and there's no
        canonical "reset" view to fall back on, re-saving after changing
        the view doesn't overwrite the previous save — see
        unique_export_paths — so a set of earlier views already saved this
        session stays on disk alongside the new one."""
        if not section_panels:
            status_text.set_text("No section panels to save.")
            blit_hover_overlays()
            return
        target_dir = Path(run_folder) if run_folder is not None else CACHE_DIR
        show_working_indicator('Saving…')
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            export_fig = render_section_maps_export_figure()
            paths = unique_export_paths(target_dir, f'{sanitized_view_token()}_section_maps', ('.png', '.svg'))
            export_fig.savefig(paths['.png'], dpi=SECTION_MAPS_SAVE_DPI)
            export_fig.savefig(paths['.svg'])
            names = ', '.join(path.name for path in paths.values())
            status_text.set_text(f"Saved {names} to {target_dir}.")
            print(f"Saved section maps to {', '.join(str(path) for path in paths.values())}.")
        except Exception as e:
            status_text.set_text(f"Could not save section maps: {e}")
            print(f"Could not save section maps: {e}")
        hide_working_indicator()
        # status_text is animated, so a blit is what actually puts the
        # message on screen — no full redraw needed just to report this.
        blit_hover_overlays()

    def export_de_genes(event=None):
        """Writes two DEG CSVs to this run's own folder — one for the
        standard ~500-gene MERFISH panel, one for the (much larger) imputed
        gene set. The two aren't mutually exclusive: a panel gene appears in
        both, each with its own independently computed p-value/fold-change
        (actually measured vs. imputed) — deliberately not deduplicated,
        since having both is useful, not redundant. For the current
        'Specified <level>(s)' selection (last_id_selection).

        First pops up prompt_deg_comparison_mode to choose between two
        comparison modes:
          'others'    — each typed ID vs. every *other* cell in this run's
                        own adata (including cells belonging to any other
                        typed ID — from any one ID's perspective, those are
                        still just "other cells"), the same one-cluster-
                        vs-rest convention Seurat's FindAllMarkers uses.
                        This is the original, still-default behavior.
          'reference' — each typed ID vs. one particular reference ID's own
                        cells *only* (every other cell, including any other
                        typed ID's, is excluded from that comparison
                        entirely — not just left uncompared, actually
                        removed from the pool compute_de_genes sees, so its
                        own "rest" is exactly the reference and nothing
                        else). A typed ID matching the reference itself is
                        skipped — comparing a group against itself is
                        meaningless.
        Cancelling the dialog aborts the whole export with no files
        written, same as if the button had never been clicked.

        Each file is long-format: one row per (gene, ID) combination, with
        a 'group_id' column saying which ID that row's comparison was
        against (plus 'reference_id' in 'reference' mode, since the "what
        was this compared to" the column name alone implies in 'others'
        mode is no longer a fixed, implicit "everything else") — rather
        than either a separate file or a separate pair of columns per ID —
        the natural shape for filtering/pivoting in Excel or pandas
        regardless of how many IDs were typed.

        See compute_de_genes for the actual statistics; this is just
        group-mask bookkeeping (once per ID), file naming, and reporting
        what happened.

        Only ever reachable while in ID mode with the Export DEGs button
        actually visible (see on_mode_change), but still checks
        last_id_selection itself — a redraw can leave it stale (e.g. a
        query that resolved to zero valid IDs) even while the mode itself
        hasn't changed."""
        level = last_id_selection['level']
        targets = last_id_selection['targets']
        if not targets:
            status_text.set_text("Specify at least one ID before exporting DE genes.")
            blit_hover_overlays()
            return
        level_name = LEVEL_DISPLAY_NAMES[level]

        parent_window = tk_widget.winfo_toplevel() if tk_widget is not None else None
        if parent_window is None:
            status_text.set_text("Could not open the DEG comparison dialog (no window handle available).")
            blit_hover_overlays()
            return
        title = f"DEGs for {level_name} {', '.join(str(t) for t in targets)}"
        choice = prompt_deg_comparison_mode(parent_window, title, level_name)
        if choice is None:
            status_text.set_text("DEG export cancelled.")
            blit_hover_overlays()
            return
        comparison_mode = choice['mode']
        reference = choice['reference']

        if comparison_mode == 'reference':
            effective_targets = [t for t in targets if t != reference]
            if not effective_targets:
                status_text.set_text(
                    f"Every entered {level_name.lower()} ID equals the reference ({reference}); "
                    "nothing to compare."
                )
                blit_hover_overlays()
                return
        else:
            effective_targets = targets

        ids_arr = level_ids_arrays[level].to_numpy()
        target_dir = Path(run_folder) if run_folder is not None else CACHE_DIR
        token = re.sub(
            r'[^A-Za-z0-9._-]+', '_',
            f"{level}_{'-'.join(str(t) for t in effective_targets)}"
            + (f'_vsRef{reference}' if comparison_mode == 'reference' else '')
        ).strip('_') or level
        # Only the "no group/rest to compare" skip message's wording
        # actually differs between modes — everything else about the two
        # loops below is identical bookkeeping, just over a possibly-
        # restricted cell pool.
        if comparison_mode == 'reference':
            merfish_skip_reason = f"had both its own cells and the reference {level_name.lower()}'s cells present"
            imputed_skip_reason = (f"had both its own cells and the reference {level_name.lower()}'s cells present "
                                    "among this run's cells found in the imputed dataset")
        else:
            merfish_skip_reason = 'had both matching cells and a rest to compare against'
            imputed_skip_reason = ('resolved to both a group and a rest among this run\'s cells found in the '
                                    'imputed dataset')
        show_working_indicator('Computing DE genes…')
        written, skipped = [], []
        try:
            target_dir.mkdir(parents=True, exist_ok=True)

            if has_counts:
                # Normalized/logged once, up front — reused for every ID's
                # own comparison below, since none of that depends on which
                # ID is currently being compared. A copy, not a live
                # reference: normalize_total/log1p must never touch this
                # window's own adata.X (Gene mode's layers['counts'] source,
                # and whatever redraw_* last drew, both depend on it staying
                # untouched).
                working_text.set_text('Computing DE genes (MERFISH)…')
                blit_sidebar_overlays()
                merfish_adata = adata.copy()
                merfish_adata.X = merfish_adata.layers['counts'].copy()
                sc.pp.normalize_total(merfish_adata)
                sc.pp.log1p(merfish_adata)  # natural log — scanpy's own default, matches log1p_base=None below
                merfish_frames = []
                for target in effective_targets:
                    if comparison_mode == 'reference':
                        # Restricts the comparison pool itself to just this
                        # pair — not merely which cells are flagged
                        # 'specified' vs 'rest' — so compute_de_genes' own
                        # "rest" (~group_mask, over whatever it's handed)
                        # ends up being exactly the reference's cells, not
                        # every other cell in the run.
                        pair_mask = (ids_arr == target) | (ids_arr == reference)
                        compare_adata = merfish_adata[pair_mask]
                        group_mask = ids_arr[pair_mask] == target
                    else:
                        compare_adata = merfish_adata
                        group_mask = ids_arr == target
                    if not group_mask.any() or group_mask.all():
                        continue  # no matching cells, or no rest to compare against — nothing to report for this one
                    df = compute_de_genes(compare_adata, group_mask, log1p_base=None)
                    df.insert(0, 'group_id', target)
                    if comparison_mode == 'reference':
                        df.insert(1, 'reference_id', reference)
                    merfish_frames.append(df)
                if merfish_frames:
                    merfish_df = pd.concat(merfish_frames, ignore_index=True)
                    merfish_path = target_dir / f'DEG_{token}_merfish.csv'
                    merfish_df.to_csv(merfish_path, index=False)
                    written.append((merfish_path, len(merfish_df), len(merfish_frames)))
                else:
                    skipped.append(f'MERFISH (no specified ID {merfish_skip_reason})')
            else:
                skipped.append('MERFISH (no counts layer on this dataset)')

            working_text.set_text('Computing DE genes (imputed)…')
            blit_sidebar_overlays()
            if ensure_imputed_gene_dataset_loaded(imputed_state, abc_cache):
                imputed_full = imputed_state['adata']
                # Matched by cell ID, not assumed to share adata's own row
                # order — same reasoning as redraw_gene's own imputed
                # branch (find_gene_index is a *gene* lookup; this is the
                # equivalent for cells).
                positions = imputed_full.obs_names.get_indexer(adata.obs_names)
                found = positions >= 0
                if found.any():
                    # materialize_subset pulls only these rows off disk —
                    # not the whole (genome-wide, atlas-scale) backed
                    # dataset — and does it as one bulk row-read rather
                    # than per-gene column reads, the fast direction for a
                    # CSR-backed AnnData (see materialize_subset's own
                    # docstring; this is exactly the kind of subsetting it
                    # exists for). Also done once, up front, same reasoning
                    # as merfish_adata above.
                    imputed_subset = materialize_subset(imputed_full, positions[found])
                    # Deliberately not excluding genes already in the MERFISH
                    # panel — a gene appearing in both files, each with its
                    # own p-value/fold-change computed from a different
                    # source (actually measured vs. imputed), is useful
                    # information on its own, not a duplicate to filter out.
                    # ids_arr restricted to the cells that actually matched
                    # — 'found' selects rows out of ids_arr in the identical
                    # order materialize_subset just used to select rows out
                    # of imputed_full, so the two stay aligned.
                    imputed_ids_arr = ids_arr[found]
                    imputed_frames = []
                    for target in effective_targets:
                        if comparison_mode == 'reference':
                            imputed_pair_mask = (imputed_ids_arr == target) | (imputed_ids_arr == reference)
                            imputed_compare = imputed_subset[imputed_pair_mask]
                            imputed_group_mask = imputed_ids_arr[imputed_pair_mask] == target
                        else:
                            imputed_compare = imputed_subset
                            imputed_group_mask = imputed_ids_arr == target
                        if not imputed_group_mask.any() or imputed_group_mask.all():
                            continue
                        # Already log2 on disk (see load_imputed_adata's own
                        # docstring) — never normalized or logged again,
                        # unlike the MERFISH panel above.
                        df = compute_de_genes(imputed_compare, imputed_group_mask, log1p_base=2)
                        df.insert(0, 'group_id', target)
                        if comparison_mode == 'reference':
                            df.insert(1, 'reference_id', reference)
                        imputed_frames.append(df)
                    if imputed_frames:
                        imputed_df = pd.concat(imputed_frames, ignore_index=True)
                        imputed_path = target_dir / f'DEG_{token}_imputed.csv'
                        imputed_df.to_csv(imputed_path, index=False)
                        written.append((imputed_path, len(imputed_df), len(imputed_frames)))
                    else:
                        skipped.append(f'imputed (no specified ID {imputed_skip_reason})')
                else:
                    skipped.append("imputed (none of this run's cells found in the imputed dataset)")
            else:
                skipped.append('imputed (dataset not loaded)')

            if written:
                summary = '; '.join(
                    f"{p.name} ({n} rows, {k} of {len(effective_targets)} IDs)" for p, n, k in written)
                mode_note = f" vs. reference {level_name.lower()} {reference}" if comparison_mode == 'reference' else ''
                msg = f"Exported {summary}{mode_note} to {target_dir}."
                if comparison_mode == 'reference' and len(effective_targets) < len(targets):
                    msg += f" (Excluded reference {level_name.lower()} {reference} from its own comparison.)"
                if skipped:
                    msg += f" Skipped: {'; '.join(skipped)}."
                status_text.set_text(msg)
                print(f"Exported DE genes for {level_name} {', '.join(str(t) for t in effective_targets)}: {msg}")
            else:
                status_text.set_text(f"Could not export DE genes: {'; '.join(skipped) or 'nothing to export'}.")
        except Exception as e:
            status_text.set_text(f"Could not export DE genes: {e}")
            print(f"Could not export DE genes: {e}")
        hide_working_indicator()
        blit_hover_overlays()

    def export_expression(event=None):
        """Writes one or two 'mean log2 expression' CSVs (MERFISH panel /
        imputed genome-wide — same has_counts/imputed-availability split as
        export_de_genes) for the current 'Specified <level>(s)' selection:
        one row per gene (gene_symbol in column A, ensembl_id in column B),
        sorted alphabetically by gene_symbol, with one 'mean_log2_expr_<ID>'
        column per specified ID holding that ID's own cells' mean log2
        expression (mean_log2_expression — the same math export_de_genes'
        own mean_log2_group column uses).

        No comparison, fold-change, or significance test against anything
        else — just each ID's own raw average, unlike export_de_genes'
        group-vs-something output. An ID with no matching cells at all is
        skipped (there's nothing to average) and noted in the status
        message; unlike export_de_genes, an ID doesn't also need a 'rest'
        to compare against, so this only ever drops an ID for being
        entirely absent, never for being the *only* thing present.

        Only ever reachable while in ID mode with the Export expression
        button actually visible (see on_mode_change), but still checks
        last_id_selection itself — same reasoning as export_de_genes' own
        check."""
        level = last_id_selection['level']
        targets = last_id_selection['targets']
        if not targets:
            status_text.set_text("Specify at least one ID before exporting expression.")
            blit_hover_overlays()
            return
        level_name = LEVEL_DISPLAY_NAMES[level]
        ids_arr = level_ids_arrays[level].to_numpy()
        target_dir = Path(run_folder) if run_folder is not None else CACHE_DIR
        token = re.sub(r'[^A-Za-z0-9._-]+', '_',
                       f"{level}_{'-'.join(str(t) for t in targets)}").strip('_') or level
        show_working_indicator('Computing expression…')
        written, skipped = [], []
        try:
            target_dir.mkdir(parents=True, exist_ok=True)

            if has_counts:
                # Same normalize/log1p-once-up-front reasoning as export_
                # de_genes' own merfish_adata — a copy, so this window's own
                # adata.X is never touched.
                working_text.set_text('Computing expression (MERFISH)…')
                blit_sidebar_overlays()
                merfish_adata = adata.copy()
                merfish_adata.X = merfish_adata.layers['counts'].copy()
                sc.pp.normalize_total(merfish_adata)
                sc.pp.log1p(merfish_adata)  # natural log — matches log1p_base=None below
                merfish_df = pd.DataFrame({
                    'gene_symbol': merfish_adata.var['gene_symbol'].values,
                    'ensembl_id': merfish_adata.var_names,
                })
                included = []
                for target in targets:
                    group_mask = ids_arr == target
                    if not group_mask.any():
                        continue
                    merfish_df[f'mean_log2_expr_{target}'] = mean_log2_expression(
                        merfish_adata, group_mask, log1p_base=None).values
                    included.append(target)
                if included:
                    merfish_df = merfish_df.sort_values('gene_symbol', kind='stable').reset_index(drop=True)
                    merfish_path = target_dir / f'EXPR_{token}_merfish.csv'
                    merfish_df.to_csv(merfish_path, index=False)
                    written.append((merfish_path, len(merfish_df), len(included)))
                else:
                    skipped.append('MERFISH (no specified ID had any matching cells)')
            else:
                skipped.append('MERFISH (no counts layer on this dataset)')

            working_text.set_text('Computing expression (imputed)…')
            blit_sidebar_overlays()
            if ensure_imputed_gene_dataset_loaded(imputed_state, abc_cache):
                imputed_full = imputed_state['adata']
                # Same cell-ID matching as export_de_genes' own imputed
                # branch — imputed_full isn't assumed to share adata's row
                # order.
                positions = imputed_full.obs_names.get_indexer(adata.obs_names)
                found = positions >= 0
                if found.any():
                    imputed_subset = materialize_subset(imputed_full, positions[found])
                    imputed_ids_arr = ids_arr[found]
                    imputed_df = pd.DataFrame({
                        'gene_symbol': imputed_subset.var['gene_symbol'].values,
                        'ensembl_id': imputed_subset.var_names,
                    })
                    included = []
                    for target in targets:
                        imputed_group_mask = imputed_ids_arr == target
                        if not imputed_group_mask.any():
                            continue
                        # Already log2 on disk — never normalized/logged
                        # again, unlike the MERFISH panel above.
                        imputed_df[f'mean_log2_expr_{target}'] = mean_log2_expression(
                            imputed_subset, imputed_group_mask, log1p_base=2).values
                        included.append(target)
                    if included:
                        imputed_df = imputed_df.sort_values('gene_symbol', kind='stable').reset_index(drop=True)
                        imputed_path = target_dir / f'EXPR_{token}_imputed.csv'
                        imputed_df.to_csv(imputed_path, index=False)
                        written.append((imputed_path, len(imputed_df), len(included)))
                    else:
                        skipped.append("imputed (no specified ID had any matching cells among this run's "
                                       "cells found in the imputed dataset)")
                else:
                    skipped.append("imputed (none of this run's cells found in the imputed dataset)")
            else:
                skipped.append('imputed (dataset not loaded)')

            if written:
                summary = '; '.join(f"{p.name} ({n} rows, {k} of {len(targets)} IDs)" for p, n, k in written)
                msg = f"Exported {summary} to {target_dir}."
                if skipped:
                    msg += f" Skipped: {'; '.join(skipped)}."
                status_text.set_text(msg)
                print(f"Exported expression for {level_name} {', '.join(str(t) for t in targets)}: {msg}")
            else:
                status_text.set_text(f"Could not export expression: {'; '.join(skipped) or 'nothing to export'}.")
        except Exception as e:
            status_text.set_text(f"Could not export expression: {e}")
            print(f"Could not export expression: {e}")
        hide_working_indicator()
        blit_hover_overlays()

    def on_mode_change(label):
        previous_mode = mode_state['mode']
        if previous_mode != 'All Subclasses':
            # Saved under the *outgoing* mode's key — for 'Single Subclass'
            # that includes the level, which hasn't changed here.
            query_by_mode[query_state_key(previous_mode)] = query_textbox.text
        mode_state['mode'] = label
        show_query = label != 'All Subclasses'  # no query box at all for 'All Subclasses'
        query_ax.set_visible(show_query)
        query_label.set_visible(show_query)
        export_degs_ax.set_visible(label == 'Single Subclass')
        export_expr_ax.set_visible(label == 'Single Subclass')
        gene_dropdown_ax.set_visible(False)
        gene_suggestion_state['matches'] = []
        if label == 'Imputed Gene':
            # Loaded here — as soon as the mode is picked — rather than
            # only inside redraw_gene: redraw_gene only runs its own
            # ensure_imputed_gene_dataset_loaded() call once a non-empty
            # query is submitted, but the gene box's autocomplete
            # (gene_symbol_list) needs imputed_state['adata']
            # ready well before that, the moment the user starts typing —
            # without this, autocomplete kept showing the standard 500-gene
            # panel until the first full gene name was actually submitted.
            ensure_imputed_gene_dataset_loaded(imputed_state, abc_cache)
        if show_query:
            query_label.set_text(
                f'{LEVEL_DISPLAY_NAMES[level_state["value"]]} ID:' if label == 'Single Subclass'
                else 'Gene name(s):'
            )
            # Restores this mode's last-typed text (per level, for
            # 'Single Subclass'), or '' if nothing's been typed there yet.
            set_query_text(query_by_mode.get(query_state_key(label), ''))
            # Moves keyboard focus into the query box the moment its mode is
            # picked, so the user can start typing immediately without an
            # extra click — begin_typing() is the exact same call a real
            # click on the box makes (see TextBox._click); TextBox's own
            # _keypress only ever checks capturekeystrokes, never the
            # mouse's current position, so this doesn't depend on where the
            # cursor happens to be after clicking the radio button.
            query_textbox.begin_typing()
            query_textbox.cursor_index = len(query_textbox.text)
            query_textbox._rendercursor()
        redraw()

    def on_mode_click(event):
        if busy_state['active']:
            return
        if event.inaxes is not radio_ax or event.ydata is None:
            return
        idx = min(range(len(MODE_OPTIONS)), key=lambda i: abs(mode_dot_y[i] - event.ydata))
        label = MODE_OPTIONS[idx]
        if label == mode_state['mode']:
            return
        set_mode_dot_visual(idx)
        on_mode_change(label)

    fig.canvas.mpl_connect('button_press_event', on_mode_click)

    def on_query_submit(text):
        # 'submit' fires from two genuinely different matplotlib code
        # paths, and only one of them belongs here. A real Enter press
        # (TextBox._keypress) calls self._observers.process('submit', ...)
        # directly, leaving capturekeystrokes untouched (still True) —
        # that's the case this function is for: Enter is just as much a
        # "selection made" as a click on a suggestion, but nothing else
        # closes the dropdown on this path, so it used to sit open (showing
        # stale suggestions for whatever was typed) until some later,
        # unrelated interaction happened to touch it again.
        #
        # But 'submit' *also* fires whenever the user clicks anywhere
        # outside the box at all (TextBox._click -> stop_typing, patched by
        # make_textbox_stop_typing_blit_fast as fast_stop_typing here) —
        # including a click squarely on a dropdown suggestion, since that's
        # just as much "outside the box" as anywhere else. fast_stop_typing
        # sets capturekeystrokes = False *before* firing that submit, which
        # is what distinguishes it from a real Enter below. That click is
        # also its own separate 'button_press_event', so on_gene_dropdown_
        # click (registered later, so it runs after this one for the same
        # click) is *also* about to fire for it — and already handles
        # "close the dropdown, and select a row if the click landed on one"
        # correctly on its own, for every click, not just ones aimed at a
        # suggestion. This function clearing gene_suggestion_state and
        # hiding the dropdown *first* raced that: on_gene_dropdown_click's
        # own row lookup ran second, against a dropdown that had already
        # been emptied, and so ends up finding nothing to select — the
        # dropdown still closed (looked like it "worked"), but the click
        # never reached query_textbox.set_val(). Bailing out here for a
        # click-triggered submit leaves on_gene_dropdown_click as the one
        # and only thing reacting to it. (See is_enter_submit.)
        if not is_enter_submit(query_textbox):
            return
        gene_dropdown_ax.set_visible(False)
        gene_suggestion_state['matches'] = []
        redraw()

    query_textbox.on_submit(on_query_submit)
    open_data_folder_button.on_clicked(open_data_folder)
    show_button.on_clicked(lambda event: redraw())
    save_section_maps_button.on_clicked(save_section_maps)
    save_button.on_clicked(save_current_umap)
    export_degs_button.on_clicked(export_de_genes)
    export_expr_button.on_clicked(export_expression)

    # --- Resize handles between the three panels -----------------------
    # Two thin draggable bars, at boundary1 (grid/UMAP) and boundary2
    # (UMAP/sidebar) — dragging one only ever trades width between the two
    # regions it separates (the outer edges, GRID_LEFT and SIDEBAR_RIGHT,
    # never move), and doesn't touch the other boundary at all, so the
    # "equal widths" of the initial layout is just that — an initial
    # layout — not an invariant re-enforced during a drag.
    HANDLE_WIDTH = 0.0042  # 30% narrower than its previous 0.006
    MIN_REGION_WIDTH = 0.10
    # Stops just above the status panel (y=[0.005, 0.08], see status_panel_
    # ax above) instead of running the full 0.0-1.0 figure height — a full
    # -height bar drawn after (so on top of, in both z-order and mouse hit-
    # testing) the status panel was cutting through/obscuring it. Tracks
    # AREA_BOTTOM (not a separate fixed constant) so it stays just above
    # the plot area's own bottom edge even as that moves on resize — see
    # on_figure_resize.
    HANDLE_BOTTOM = AREA_BOTTOM

    handle1_ax = fig.add_axes([boundary1 - HANDLE_WIDTH / 2, HANDLE_BOTTOM, HANDLE_WIDTH, 1.0 - HANDLE_BOTTOM])
    handle1_ax.set_facecolor('#999999')
    handle1_ax.set_xticks([])
    handle1_ax.set_yticks([])
    for spine in handle1_ax.spines.values():
        spine.set_visible(False)

    handle2_ax = fig.add_axes([boundary2 - HANDLE_WIDTH / 2, HANDLE_BOTTOM, HANDLE_WIDTH, 1.0 - HANDLE_BOTTOM])
    handle2_ax.set_facecolor('#999999')
    handle2_ax.set_xticks([])
    handle2_ax.set_yticks([])
    for spine in handle2_ax.spines.values():
        spine.set_visible(False)

    def reposition_grid(grid_right):
        # section_panels is empty when there's no spatial/section data (see
        # its own build block above) — grid_ncols/panel_gap/panel_h are
        # only ever defined in that same case, so this has to bail before
        # touching them.
        nonlocal panel_h
        if not section_panels:
            return
        # Recomputed here (not just panel_w below) since AREA_BOTTOM can
        # also change now — see on_figure_resize — which changes how much
        # vertical room (AREA_TOP - AREA_BOTTOM) the grid has to work with,
        # same as grid_right changing how much horizontal room it has.
        panel_h = (AREA_TOP - AREA_BOTTOM - (grid_nrows - 1) * panel_gap) / grid_nrows
        panel_w_new = (grid_right - GRID_LEFT - (grid_ncols - 1) * panel_gap) / grid_ncols
        for i, (_sec, panel) in enumerate(section_panels.items()):
            row, col = divmod(i, grid_ncols)
            x0 = GRID_LEFT + col * (panel_w_new + panel_gap)
            y0 = AREA_TOP - (row + 1) * panel_h - row * panel_gap
            panel['ax'].set_position([x0, y0, panel_w_new, panel_h])

    def reposition_sidebar(new_left, new_width):
        color_by_label.set_x(new_left)
        radio_ax.set_position([new_left, 0.80, new_width, 0.16])
        level_label.set_x(new_left)
        level_selector_ax.set_position([new_left, LEVEL_TOP_Y, new_width, 0.04])
        # level_dropdown_ax's own y0/height are fixed (always n_levels rows),
        # unlike gene_dropdown_ax below — only x0/width change here.
        cur_level_pos = level_dropdown_ax.get_position()
        level_dropdown_ax.set_position([new_left, cur_level_pos.y0, new_width, cur_level_pos.height])
        query_ax.set_position([new_left, QUERY_TOP_Y, new_width, 0.04])
        query_label.set_x(new_left)
        # gene_dropdown_ax's own y0/height depend on how many suggestions
        # are currently shown (see update_gene_suggestions) — only x0/width
        # change here.
        cur_pos = gene_dropdown_ax.get_position()
        gene_dropdown_ax.set_position([new_left, cur_pos.y0, new_width, cur_pos.height])
        open_data_folder_ax.set_position([new_left, BUTTON_BLOCK_TOP_Y0, new_width, BUTTON_BLOCK_HEIGHT])
        show_ax.set_position(
            [new_left, BUTTON_BLOCK_TOP_Y0 - BUTTON_BLOCK_STEP, new_width, BUTTON_BLOCK_HEIGHT])
        save_section_maps_ax.set_position(
            [new_left, BUTTON_BLOCK_TOP_Y0 - 2 * BUTTON_BLOCK_STEP, new_width, BUTTON_BLOCK_HEIGHT])
        save_ax.set_position(
            [new_left, BUTTON_BLOCK_TOP_Y0 - 3 * BUTTON_BLOCK_STEP, new_width, BUTTON_BLOCK_HEIGHT])
        export_degs_ax.set_position(
            [new_left, BUTTON_BLOCK_TOP_Y0 - 4 * BUTTON_BLOCK_STEP, new_width, BUTTON_BLOCK_HEIGHT])
        export_expr_ax.set_position(
            [new_left, BUTTON_BLOCK_TOP_Y0 - 5 * BUTTON_BLOCK_STEP, new_width, BUTTON_BLOCK_HEIGHT])
        close_ax.set_position(
            [new_left, BUTTON_BLOCK_TOP_Y0 - 6 * BUTTON_BLOCK_STEP, new_width, BUTTON_BLOCK_HEIGHT])
        working_text.set_x(new_left)

    def apply_region_layout():
        nonlocal GRID_RIGHT, UMAP_LEFT, UMAP_RIGHT, SIDEBAR_LEFT, SIDEBAR_WIDTH, AREA_BOTTOM, HANDLE_BOTTOM
        GRID_RIGHT = boundary1 - GAP / 2
        UMAP_LEFT = boundary1 + GAP / 2
        UMAP_RIGHT = boundary2 - GAP / 2
        SIDEBAR_LEFT = boundary2 + GAP / 2
        SIDEBAR_WIDTH = SIDEBAR_RIGHT - SIDEBAR_LEFT
        # Recomputed every call (not just on the horizontal boundary drags
        # this function was originally written for) so it also tracks the
        # current window/figure size correctly when called from
        # on_figure_resize — see compute_area_bottom's own comment.
        AREA_BOTTOM = compute_area_bottom()
        HANDLE_BOTTOM = AREA_BOTTOM
        ax.set_position([UMAP_LEFT, AREA_BOTTOM,
                          (UMAP_RIGHT - UMAP_LEFT) - CBAR_WIDTH - CBAR_GAP - CBAR_LABEL_MARGIN,
                          AREA_TOP - AREA_BOTTOM])
        cbar_ax.set_position([UMAP_RIGHT - CBAR_WIDTH - CBAR_LABEL_MARGIN, AREA_BOTTOM, CBAR_WIDTH,
                               AREA_TOP - AREA_BOTTOM])
        reposition_grid(GRID_RIGHT)
        reposition_sidebar(SIDEBAR_LEFT, SIDEBAR_WIDTH)
        panel_bottom, panel_height = status_panel_geometry()
        status_panel_ax.set_position([GRID_LEFT, panel_bottom, UMAP_RIGHT - GRID_LEFT, panel_height])
        hover_status_ax.set_position([
            GRID_LEFT + 0.005, panel_bottom + panel_height * (STATUS_PANEL_LINE_FRAC + STATUS_PANEL_GAP_FRAC),
            UMAP_RIGHT - GRID_LEFT - 0.01, panel_height * STATUS_PANEL_LINE_FRAC,
        ])
        status_ax.set_position([
            GRID_LEFT + 0.005, panel_bottom, UMAP_RIGHT - GRID_LEFT - 0.01, panel_height * STATUS_PANEL_LINE_FRAC,
        ])
        handle1_ax.set_position([boundary1 - HANDLE_WIDTH / 2, HANDLE_BOTTOM, HANDLE_WIDTH, 1.0 - HANDLE_BOTTOM])
        handle2_ax.set_position([boundary2 - HANDLE_WIDTH / 2, HANDLE_BOTTOM, HANDLE_WIDTH, 1.0 - HANDLE_BOTTOM])

    # Redrawing every section panel's own per-point-colored scatter is the
    # expensive part of a layout change here (apply_region_layout()'s own
    # set_position() calls are cheap; the full fig.canvas.draw_idle() that
    # follows is not) — both a divider drag (on_motion_resize) and an
    # actual OS window resize (on_figure_resize) fire many events per
    # second while the drag is in progress, so triggering a full redraw on
    # *every* one queued them up faster than the canvas could render them,
    # reading as "can't resize — it just freezes/snaps back", especially
    # for an OS-driven window-edge drag, where a laggy app can miss enough
    # live-resize frames that Windows itself reverts the in-progress drag.
    # Same throttle-and-settle pattern as prompt_section_selection_gui's own
    # throttled_draw_idle: caps redraws during a fast burst, but always
    # leaves a trailing timer so the layout still settles to its true final
    # state even if no further event arrives. (Cap set by
    # LAYOUT_REDRAW_MIN_INTERVAL, top of file.)
    layout_redraw_state = {'last_time': 0.0, 'timer': None}

    def force_layout_draw():
        layout_redraw_state['last_time'] = time.perf_counter()
        apply_region_layout()
        fig.canvas.draw_idle()

    # maybe_redraw_for_legend_resize() calls the *heavy* redraw() (rebuilds
    # the UMAP scatter and every section panel's per-cell colors) whenever
    # the legend row count would change — calling that on every throttled
    # layout tick (as often as every LAYOUT_REDRAW_MIN_INTERVAL, i.e. up to
    # ~30x/second) during a live window-edge drag is what made resizing
    # freeze: each drag tick queued a full, expensive redraw behind the
    # previous one faster than they could finish. Debounced separately here,
    # on its own longer timer that only fires once resize activity has
    # actually paused — the cheap apply_region_layout()/draw_idle() in
    # force_layout_draw() above still runs at the fast/responsive cadence so
    # the window itself keeps tracking the drag smoothly.
    # Also the point at which resize fast mode ends and the real content
    # comes back, so this interval has to comfortably exceed the gap between
    # consecutive resize events — otherwise it fires *between* two frames of
    # a still-in-progress drag, which is what made the panels visibly blink
    # between gray and colored on every drag frame during the first attempt
    # at this: each frame cost ~1s, far longer than the 250ms it was set to.
    # Fast mode makes frames cheap enough that events now arrive far closer
    # together than this, so it only elapses on a genuine pause.
    RESIZE_SETTLE_INTERVAL = 0.35  # seconds of no further resize activity
    resize_settle_state = {'timer': None}

    def on_resize_settled():
        resize_settle_state['timer'] = None
        exit_resize_fast_mode()  # unhide the real data artists
        if active_zoom_previews:
            # A zoom burst is still running (its settle window is much
            # longer than this one, so a resize during a zoom lands here
            # first). Redrawing now would tear that burst down mid-flight
            # *and* reset the view to home, since every redraw_* re-applies
            # the axes' home limits — the user's in-progress zoom would
            # silently snap back out. Wait for the zoom to settle instead;
            # end_zoom_previews does its own full draw, and this check runs
            # again on the next resize event if there is one.
            schedule_resize_settle_check()
            return
        maybe_redraw_for_legend_resize()
        fig.canvas.draw_idle()

    def schedule_resize_settle_check():
        if resize_settle_state['timer'] is not None:
            resize_settle_state['timer'].stop()
        timer = fig.canvas.new_timer(interval=RESIZE_SETTLE_INTERVAL * 1000)
        timer.single_shot = True
        timer.add_callback(on_resize_settled)
        resize_settle_state['timer'] = timer
        timer.start()

    def maybe_redraw_for_legend_resize():
        # legend_row_cap_state['rows'] is only ever non-None right after
        # redraw_all_subclasses actually drew an 'All <level>s' legend
        # (cleared by clear_colorbar at the top of every redraw_*
        # otherwise) — so this is a no-op the rest of the time (Gene mode,
        # 'Specified <level>(s)' mode, or a level with no legend at all).
        # apply_region_layout() (just above) already updated AREA_BOTTOM
        # for this new size, so legend_max_rows_for_current_size() here
        # reflects it; redraw() is the heavier full mode redraw (rebuilds
        # the scatter, section colors, everything) — worth its cost since
        # it only actually runs when the row count would really change,
        # not on every resize tick.
        if legend_row_cap_state['rows'] is None:
            return
        if legend_max_rows_for_current_size() != legend_row_cap_state['rows']:
            redraw()

    def throttled_layout_redraw():
        enter_resize_fast_mode()  # hide the data artists for the duration of the drag/resize
        schedule_resize_settle_check()  # (re)start the debounce on every resize tick, not just the throttled ones
        if layout_redraw_state['timer'] is not None:
            layout_redraw_state['timer'].stop()
            layout_redraw_state['timer'] = None
        elapsed = time.perf_counter() - layout_redraw_state['last_time']
        if elapsed >= LAYOUT_REDRAW_MIN_INTERVAL:
            force_layout_draw()
            return
        timer = fig.canvas.new_timer(interval=max((LAYOUT_REDRAW_MIN_INTERVAL - elapsed) * 1000, 1))
        timer.single_shot = True
        timer.add_callback(force_layout_draw)
        layout_redraw_state['timer'] = timer
        timer.start()

    resize_state = {'active': None}  # None, 'b1', or 'b2'

    # A slightly bigger hit target than the bar's own drawn width —
    # HANDLE_WIDTH alone (0.0042 of the figure) is a tiny, hard-to-hit
    # target for a mouse. Shared by on_press_resize (does a click here
    # start a drag?) and the cursor-hover check below (should hovering
    # here, whether or not the mouse is pressed, hint that it's
    # draggable?) — one threshold, so the cursor never shows the resize
    # shape somewhere a click wouldn't actually grab the divider.
    HANDLE_HIT_MARGIN = HANDLE_WIDTH * 3


    def set_resize_cursor(active):
        # Guarded on an actual state change — set_cursor() ultimately does
        # a Tk widget .configure(cursor=...) call on every invocation
        # regardless of whether the cursor would even change, and this
        # runs on every mouse-move near a boundary.
        if active == resize_cursor_state['active']:
            return
        if busy_state['active']:
            return  # keep the hourglass; mouse moves are still processed mid-operation
        resize_cursor_state['active'] = active
        fig.canvas.set_cursor(Cursors.RESIZE_HORIZONTAL if active else Cursors.POINTER)

    def on_press_resize(event):
        if event.x is None or event.y is None:
            return
        fx, _fy = fig.transFigure.inverted().transform((event.x, event.y))
        if abs(fx - boundary1) < HANDLE_HIT_MARGIN:
            resize_state['active'] = 'b1'
        elif abs(fx - boundary2) < HANDLE_HIT_MARGIN:
            resize_state['active'] = 'b2'

    def on_motion_resize(event):
        nonlocal boundary1, boundary2
        if event.x is None or event.y is None:
            return
        fx, _fy = fig.transFigure.inverted().transform((event.x, event.y))
        if resize_state['active'] is None:
            # Not dragging — just hint, via the cursor, that a divider is
            # draggable here (same hit target on_press_resize itself uses
            # to decide whether a click would actually grab one).
            set_resize_cursor(abs(fx - boundary1) < HANDLE_HIT_MARGIN or abs(fx - boundary2) < HANDLE_HIT_MARGIN)
            return
        if resize_state['active'] == 'b1':
            fx = max(GRID_LEFT + MIN_REGION_WIDTH + GAP / 2, min(fx, boundary2 - MIN_REGION_WIDTH - GAP))
            boundary1 = fx
        else:
            fx = max(boundary1 + MIN_REGION_WIDTH + GAP, min(fx, SIDEBAR_RIGHT - MIN_REGION_WIDTH - GAP / 2))
            boundary2 = fx
        throttled_layout_redraw()

    def on_release_resize(event):
        resize_state['active'] = None

    fig.canvas.mpl_connect('button_press_event', on_press_resize)
    fig.canvas.mpl_connect('motion_notify_event', on_motion_resize)
    fig.canvas.mpl_connect('button_release_event', on_release_resize)

    def on_figure_resize(event):
        # Fired whenever the window itself (not a resize *handle* drag,
        # which calls apply_region_layout() directly) is resized — the
        # status panel's own inches-based sizing (see compute_area_bottom)
        # is the one thing here that actually depends on the figure's
        # current size, but it's simplest to just recompute the whole
        # layout the same way a horizontal handle drag already does,
        # rather than duplicating a status-panel-only code path.
        throttled_layout_redraw()

    fig.canvas.mpl_connect('resize_event', on_figure_resize)

    def on_umap_window_close(event):
        # Any of these still-pending .after()/new_timer() callbacks firing
        # once this window (and its Tk widget) is gone raises Tcl's own
        # 'invalid command name "...<lambda>" ("after" script)' error — not
        # catchable from Python, since the failure happens inside Tcl's
        # "after" dispatch on a later event-loop tick (e.g. servicing the
        # *next* window opened by the session loop), before any of our code
        # runs. Cancelling everything here, while the widget still exists,
        # avoids that.
        if tk_widget is None:
            return
        for key in ('timer_id', 'group_timer_id', 'reraise_timer_id'):
            timer_id = hover_state.get(key)
            if timer_id is not None:
                try:
                    tk_widget.after_cancel(timer_id)
                except Exception:
                    pass
                hover_state[key] = None
        if zoom_settle_timer['timer'] is not None:
            try:
                zoom_settle_timer['timer'].stop()
            except Exception:
                pass
            zoom_settle_timer['timer'] = None

    fig.canvas.mpl_connect('close_event', on_umap_window_close)

    log_status("Step 9: Rendering initial view...")
    try:
        redraw()  # initial view — MODE_OPTIONS[0] ('All Subclasses')
        log_status("Step 9: Initial view built (colors/data assigned) — starting first real draw...")
        # Forced synchronous (not the draw_idle() redraw() already scheduled)
        # so the first real draw — which is what populates blit_bg for every
        # blit_hover_overlays()/blit_sidebar_overlays() call, and also warms
        # matplotlib's own font/glyph caches for every text label in this
        # window — happens *before* show_figure_blocking()'s event loop starts
        # accepting input, not whenever that pending idle draw happens to get
        # its turn. Without this, typing (or any other blit-only interaction)
        # right as the window appears could still land before blit_bg was
        # populated, falling back to the slow full-draw path for those first
        # few keystrokes — which is why this was "more noticeable right after
        # launch" specifically.
        fig.canvas.draw()
        log_status("Step 9: First real draw complete.")
        center_figure_window(fig)
    except Exception:
        # A failure anywhere in this block happens *after* the window is
        # already visible on screen, but before show_figure_blocking()'s own
        # event-pump loop (below) ever starts — so it would otherwise
        # propagate straight past this whole function, leaving that Tk
        # window on screen with nothing servicing its event loop: open, but
        # completely unresponsive, regardless of what the caller's own
        # exception handler logs immediately afterward ("...viewer closed").
        # Closing it here makes a setup failure behave the same as any other
        # close, from the caller's perspective.
        plt.close(fig)
        raise
    try:
        show_figure_blocking(fig)
    finally:
        # Re-enable and run one real collection now that the window (and
        # its many thousands of cyclic-referencing Artists) is closed and
        # can actually be reclaimed, rather than leaving gc permanently
        # disabled for the rest of the script.
        if gc_was_enabled:
            gc.enable()
        gc.collect()


def run_session(abc_cache, session_state, raw_backed_cache, imputed_state=None):
    """One full user session: show the startup panel, then either open
    the section/ROI picker plus the full UMAP-computation pipeline, or
    load a previously computed run directly — ending, either way, with
    the interactive UMAP viewer. Returns normally once that viewer
    window closes, or the user cancels partway through (a
    UserCancelledSelection from the section/ROI picker) — never calls
    sys.exit() itself; only prompt_startup_panel's own "closed without
    choosing" case does that, which is deliberately the one place that
    ends the whole program (see the session loop below, which just
    calls this repeatedly). raw_backed_cache is a plain {} the caller
    keeps alive across calls, so the raw MERFISH h5ad (see
    get_cached_raw_backed_adata) is only ever opened once per process,
    not once per session. imputed_state is likewise kept alive by the
    caller, so the imputed gene dataset (see
    ensure_imputed_gene_dataset_loaded) is loaded at most once per process;
    None gives this session its own, as before.

    session_state is a mutable dict carrying {'out_folder': Path} across
    calls: the startup panel lets the user change the output folder, and
    writing it back here (rather than returning it) means every one of this
    function's several early-return paths picks up the new folder for the
    next session without each having to thread it through."""
    # log_status (defined at module level, outside this function) reads
    # selection_confirmed_at as a plain global — it used to be a real
    # module-level variable, set once by this same top-level script code;
    # now that the assignments below run inside this function, they'd
    # otherwise create a *local* invisible to log_status instead of
    # updating the global it actually reads.
    global selection_confirmed_at
    if imputed_state is None:
        imputed_state = {'adata': None, 'load_thread': None, 'load_error': None}
    out_folder = session_state['out_folder']
    try:
        startup_action, cell_type_selection, load_existing_folder, chosen_out_folder = prompt_startup_panel(out_folder)
        if chosen_out_folder is not None:
            out_folder = session_state['out_folder'] = chosen_out_folder
    except Exception as e:
        print(f"GUI startup panel unavailable ({e}); falling back to console prompt.")
        cell_type_selection = prompt_cell_type_selection()
        startup_action, load_existing_folder = 'new_run', None

    subset_suffix = CELL_TYPE_SUFFIXES[cell_type_selection]
    print(f"Selected cell type subset: {cell_type_selection}")

    if startup_action == 'load_existing':
        # Skips section/ROI selection and UMAP computation (Steps 1-8) entirely
        # — the chosen folder's own cached files (verified by prompt_startup_
        # panel before it let this choice through) already have everything
        # Step 9 needs. Still needs the raw backed h5ad (same read Step 0 below
        # does for the 'new_run' path) since that's what the viewer's
        # adata_backed argument uses for each section's full spatial
        # background, independent of any specific run's own selection —
        # get_cached_raw_backed_adata reuses it across sessions instead of
        # re-opening the file every time.
        print(f"Loading cached run from {load_existing_folder}...")
        adata_backed = get_cached_raw_backed_adata(raw_backed_cache, abc_cache)

        required_files = required_cached_run_files(load_existing_folder)
        processed_h5ad_path = required_files['Processed AnnData']
        roi_csv_path = required_files['ROI coordinates']
        if processed_h5ad_path.exists():
            print(f"Loading processed AnnData from {processed_h5ad_path}...")
            adata = anndata.read_h5ad(processed_h5ad_path)
        else:
            # Deleted (or never kept) — rebuildable from this run's own
            # umap_coords.csv without reopening the section/ROI picker, since
            # that CSV already records exactly which cells the run covered.
            print(f"{processed_h5ad_path} not found; rebuilding it from this run's saved UMAP coordinates.")
            try:
                adata = rebuild_processed_adata(load_existing_folder, adata_backed, processed_h5ad_path)
            except Exception as e:
                print(f"Could not rebuild the processed AnnData ({e}); returning to the startup panel.")
                return
        if roi_csv_path.exists():
            rois, _whole_sections = load_rois_csv(roi_csv_path)
            print(f"Loaded {len(rois)} ROI(s) from {roi_csv_path}.")
        else:
            # Not written at all for a whole-brain run (no ROIs, no explicit
            # whole-section picks) — see the main script's own roi_csv_path-
            # writing block further down, guarded by "if rois or whole_only_
            # sections". Not an error; just means this run covers everything.
            rois = []
            print(f"Warning: {roi_csv_path} not found; assuming this run covers the whole brain (no ROIs).")

        # Runs computed before Leiden was added have no clustering saved —
        # offer to backfill it rather than silently showing the Leiden level
        # as unavailable. Declining is fine; nothing below depends on it.
        offer_leiden_backfill(adata, Path(load_existing_folder) / 'umap_coords.csv', processed_h5ad_path)

        # Reference point for log_status()'s elapsed-time prefix, same as the
        # normal flow sets right after its own section/ROI selection step.
        selection_confirmed_at = time.perf_counter()
        log_status("Step 9: Opening the interactive UMAP viewer...")
        try:
            show_interactive_umap_window(
                adata, abc_cache,
                imputed_state=imputed_state,
                adata_backed=adata_backed, rois=rois, run_folder=Path(load_existing_folder),
            )
        except Exception as e:
            log_status(f"Step 9: Could not open the interactive UMAP viewer: {e}")
        log_status("Step 9: Interactive UMAP viewer closed.")
        return

    # ---------------------------------------------------------------------------
    # 0. Load data (as provided)
    # ---------------------------------------------------------------------------
    # abc_cache = ...  # assumed already created/configured elsewhere in your workflow

    adata = get_cached_raw_backed_adata(raw_backed_cache, abc_cache)
    print(f"Step 0: Loaded backed AnnData with {adata.n_obs} cells x {adata.n_vars} genes.")

    # ---------------------------------------------------------------------------
    # Section subset selection
    # ---------------------------------------------------------------------------
    section_series = get_section_labels(adata, abc_cache)
    try:
        print("Attempting to open the clickable section-thumbnail picker...")
        selected_sections, sections_suffix, rois, imputed_adata_from_picker = prompt_section_selection_gui(
            adata, abc_cache, section_series, out_folder=out_folder, cell_type_selection=cell_type_selection,
            imputed_state=imputed_state,
        )
    except UserCancelledSelection:
        print("Section/ROI selection cancelled; returning to the startup panel.")
        return
    except Exception as e:
        print(f"GUI section picker unavailable ({e}); falling back to console prompt.")
        selected_sections, sections_suffix = prompt_section_selection(section_series)
        rois = []
        imputed_adata_from_picker = None
    # Reference point for log_status()'s elapsed-time prefix, used from here on
    # for every status update in the processing pipeline below.
    selection_confirmed_at = time.perf_counter()
    if selected_sections is None:
        log_status("Selected sections: all")
    else:
        log_status(f"Selected sections: {selected_sections}")
    # Saved off before the ROI override below can replace selected_sections, so
    # the ROI CSV (further down) can still record whole-section picks even when
    # ROIs were also drawn on other sections in the same picker session.
    whole_sections_selected = selected_sections

    # ---------------------------------------------------------------------------
    # If any ROIs were drawn (double-click on a thumbnail in the grid picker),
    # they take priority over the whole-section selection above: the run
    # processes exactly the ROI cells, across however many sections they span.
    # ---------------------------------------------------------------------------
    region_suffix = ''
    if rois:
        roi_sections = sorted_sections_descending(list({roi['section'] for roi in rois}))
        selected_sections = roi_sections
        sections_suffix = 'sections-' + '-'.join(sanitize_section_token(s) for s in roi_sections)
        # A short hash of the exact ROI bounds keeps the cache filename unique
        # per distinct ROI configuration without making it unreasonably long.
        roi_key = json.dumps(
            sorted(
                [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in roi.items()} for roi in rois],
                key=lambda r: (r['section'], r['x_min'], r['y_min']),
            )
        )
        roi_hash = hashlib.md5(roi_key.encode()).hexdigest()[:8]
        region_suffix = f'_rois{len(rois)}-{roi_hash}'
        log_status(f"Using {len(rois)} ROI(s) across {len(roi_sections)} section(s) "
                   "(overrides whole-section selection above).")
        run_label = prompt_roi_run_label()
        if run_label:
            region_suffix += f'_{run_label}'
            log_status(f"Adding label '{run_label}' to this run's folder name.")

    # ---------------------------------------------------------------------------
    # Early cache-hit check: look for an existing UMAP run matching this exact
    # section/cell-type/ROI selection *before* paying for Step 2's metadata
    # merge (a full multi-million-row CSV read+join) or Step 3's filtering
    # (which reads the matched cells off disk) — both become pure waste if
    # we're about to just reload a previously computed embedding anyway.
    # ---------------------------------------------------------------------------
    # subsample_target isn't known yet at this point — it's normally decided by
    # Step 4, from the post-filter cell count this is specifically trying to
    # avoid computing — so this can't look for one exact folder name the way
    # csv_path does further down (once subsample_target *is* known). Instead it
    # globs for any run folder for this selection, with any subsample cap or
    # none, and only acts when that's unambiguous (exactly one match); with
    # zero or multiple matches this just falls through to the unchanged Step
    # 2/3/4 pipeline, which finds the *exact* match (if any) the normal way,
    # once subsample_target is actually known.
    base_run_suffix = f'{subset_suffix}_{sections_suffix}{region_suffix}'
    run_dir_pattern = re.compile(rf'^umap_{re.escape(base_run_suffix)}(?:_sub(\d+))?$')
    early_cache_candidates = []
    for candidate in Path(out_folder).glob(f'umap_{base_run_suffix}*'):
        match = run_dir_pattern.match(candidate.name)
        if match is not None and candidate.is_dir() and (candidate / 'umap_coords.csv').exists():
            early_cache_candidates.append((candidate, int(match.group(1)) if match.group(1) else None))

    early_cache_hit = len(early_cache_candidates) == 1
    subsample_target = None
    if early_cache_hit:
        early_cache_dir, subsample_target = early_cache_candidates[0]
        log_status(f"Found a single existing UMAP run matching this selection at {early_cache_dir} "
                   "before filtering — skipping Step 2's metadata merge and Step 3's filtering "
                   "entirely (see Step 2/3/4 below).")

    # ---------------------------------------------------------------------------
    # 1. Keep the backed AnnData for now — defer the full disk read
    # ---------------------------------------------------------------------------
    # adata stays backed (X is not read from disk yet) through metadata merging
    # and filtering below. Both only touch adata.obs, which is a real in-memory
    # DataFrame even in backed mode, so they don't need X materialized. This
    # means the actual disk read (in materialize_subset(), used by the filters
    # below and by subsampling further down) only ever pulls in the cells that
    # survive filtering/subsampling, instead of loading the entire multi-million
    # -cell dataset into memory just to filter or subsample most of it away
    # immediately after.
    adata_backed = adata  # keep the original, unfiltered backed AnnData for the ROI map below

    # ---------------------------------------------------------------------------
    # 2. Attach cell type metadata (class/subclass/cluster/section), if available
    # ---------------------------------------------------------------------------
    # The ABC atlas ships separate metadata tables (cell_metadata.csv) with
    # cluster/class/subclass/section annotations keyed by cell_label. If you
    # already have this loaded elsewhere, merge it in here.
    if early_cache_hit:
        log_status("Step 2: Skipped — reusing the cached run found above; its own saved "
                   "umap_coords.csv already carries whichever metadata columns it needs "
                   "(see Step 2/3/4's reuse branch further down).")
        metadata_cols = []
        color_key = None
    else:
        log_status("Step 2: Attempting to load and merge cell type metadata...")
        try:
            cell_metadata_path = abc_cache.get_metadata_path(
                directory='MERFISH-C57BL6J-638850',
                file_name='cell_metadata_with_cluster_annotation'
            )
            # converters={0: str} keeps the cell-ID index as text (see note above
            # about float64 precision loss on long numeric-looking IDs).
            cell_meta = pd.read_csv(cell_metadata_path, index_col=0, converters={0: str})
            metadata_cols = [
                c for c in ['class', 'subclass', 'supertype', 'cluster', SECTION_COL]
                if c in cell_meta.columns and c not in adata.obs.columns
            ]
            # A left join onto adata.obs (rather than subsetting adata to the
            # intersection first) keeps adata backed — subsetting a backed AnnData
            # with .copy() isn't allowed (see materialize_subset's note), and would
            # force a full disk read here regardless, before any filtering has had
            # a chance to shrink what needs to be read.
            n_matched = adata.obs.index.isin(cell_meta.index).sum()
            adata.obs = adata.obs.join(cell_meta[metadata_cols])
            color_key = 'class' if 'class' in adata.obs.columns else None
            log_status(f"Step 2: Merged metadata for {n_matched} of {adata.n_obs} cells. Coloring by '{color_key}'.")
        except Exception as e:
            log_status(f"Step 2: Could not load/merge cell type metadata ({e}); proceeding without it.")
            metadata_cols = []
            color_key = None

    # ---------------------------------------------------------------------------
    # 3. Filter to the selected cell type, section(s), and ROIs (if any)
    # ---------------------------------------------------------------------------
    if early_cache_hit:
        log_status("Step 3: Skipped — the cached run found above already implies this exact "
                   "cell-type/section/ROI filter; materializing straight from its saved "
                   "umap_coords.csv further down picks out the same cells directly.")
    else:
        section_desc = 'all sections' if selected_sections is None else f'{len(selected_sections)} selected section(s)'
        log_status(f"Step 3: Filtering to cell type subset '{cell_type_selection}' and {section_desc}...")
        adata = filter_by_cell_type(adata, cell_type_selection)
        adata = filter_by_sections(adata, section_series, selected_sections)
        if rois:
            adata = filter_by_rois(adata, abc_cache, rois)
        log_status(f"Step 3: {adata.n_obs} cells remain after filtering.")

    # ---------------------------------------------------------------------------
    # 4. Decide whether/how much to subsample
    # ---------------------------------------------------------------------------
    # The chosen cap becomes part of the output folder name below, so it has to
    # be decided before that folder is created — a run capped at a different
    # number of cells is a different run, not something to merge into a folder
    # an earlier run with a different cap already claimed.
    N_SUBSAMPLE_DEFAULT = 200_000
    if early_cache_hit:
        # subsample_target was already set (from the matched folder's own
        # '_sub{N}' suffix, or left None if it had none) by the early cache-hit
        # check above — recomputing it here from adata.n_obs would be wrong
        # anyway, since adata is still the full, unfiltered dataset at this
        # point (Step 3 was skipped) rather than the post-filter count this
        # decision is normally based on.
        log_status(f"Step 4: Skipped — reusing subsample_target={subsample_target} from the matched run.")
    else:
        subsample_target = None  # None means "use every filtered cell, no subsampling"
        if adata.n_obs > N_SUBSAMPLE_DEFAULT:
            subsample_target = prompt_subsample_choice(adata.n_obs, N_SUBSAMPLE_DEFAULT)

    run_suffix = f'{subset_suffix}_{sections_suffix}{region_suffix}'
    if subsample_target is not None:
        run_suffix += f'_sub{subsample_target}'
    # Every output file for this run is grouped into its own subfolder, named
    # after what used to be their shared filename prefix, so a directory
    # listing of out_folder shows one entry per run instead of a handful of
    # similarly-prefixed files per run.
    session_prefix = f'umap_{run_suffix}'
    run_folder = Path(out_folder) / session_prefix
    run_folder.mkdir(parents=True, exist_ok=True)
    csv_path = run_folder / 'umap_coords.csv'

    # Whole-section picks (clicked to select the entire section, no ROI drawn
    # on it) are only worth recording here if the run also has actual ROIs —
    # otherwise every section in the run is already a whole-section pick, and
    # that's already fully captured by the run's folder name.
    roi_section_set = {roi['section'] for roi in rois}
    whole_only_sections = [s for s in (whole_sections_selected or []) if s not in roi_section_set] if rois else []

    roi_csv_path = run_folder / 'roi_coords.csv'
    # Skip the confirm-overwrite prompt (and the write itself) entirely when
    # roi_coords.csv already records exactly this session's ROI/whole-section
    # selection — most relevantly when reusing a cached run's selection
    # unchanged, where re-prompting to overwrite an identical file is just
    # noise.
    roi_selection_is_unchanged = roi_selection_unchanged(roi_csv_path, rois, whole_only_sections)

    if rois or whole_only_sections:
        if roi_selection_is_unchanged:
            log_status(f"ROI/whole-section selection unchanged since last save; skipping {roi_csv_path}.")
        elif confirm_overwrite(roi_csv_path):
            # Whole-section rows get blank x_min/x_max/y_min/y_max, marking
            # "every cell in this section" as distinct from an actual
            # sub-region — load_rois_csv() reads that blankness back out to
            # route them to the right place (selected sections vs. rois) when
            # this file is reloaded via the picker's 'Load ROIs' button.
            whole_section_rows = [
                {'section': s, 'x_min': np.nan, 'x_max': np.nan, 'y_min': np.nan, 'y_max': np.nan}
                for s in whole_only_sections
            ]
            roi_csv_rows = pd.DataFrame(list(rois) + whole_section_rows)
            roi_csv_rows[['section', 'x_min', 'x_max', 'y_min', 'y_max']].to_csv(roi_csv_path, index=False)
            log_status(f"Saved {len(rois)} ROI(s) and {len(whole_only_sections)} whole section(s) to {roi_csv_path}.")
        else:
            log_status(f"Skipped overwriting {roi_csv_path}.")

    if rois:
        roi_map_path = run_folder / 'roi_map.png'
        if roi_selection_is_unchanged and roi_map_path.exists() and roi_map_path.with_suffix('.svg').exists():
            log_status(f"ROI selection unchanged since last save; skipping {roi_map_path}.")
        elif confirm_overwrite(roi_map_path):
            # Uses adata_backed, the reference to the original, unfiltered
            # AnnData saved off in Step 1 before `adata` gets reassigned by
            # filtering/subsampling below, so the map shows each ROI's full
            # section for context, not just the cells that end up surviving
            # downstream.
            save_roi_map(adata_backed, abc_cache, section_series, rois, roi_map_path)
        else:
            log_status(f"Skipped overwriting {roi_map_path}.")

    # Check for a previously computed UMAP coordinates CSV for this exact cell
    # type + section + subsample-cap selection. If found, skip the heavy
    # subsample/preprocessing/PCA/neighbors/UMAP steps and just reuse it. Also
    # true already if early_cache_hit (Steps 2/3/4 above were skipped on the
    # strength of this same check, run early) — csv_path.exists() is
    # necessarily also true then (same file, found via a different route), but
    # checking early_cache_hit first avoids a redundant stat() call.
    log_status("Looking for existing UMAP for current selection")
    found_existing_umap = early_cache_hit or csv_path.exists()

    if found_existing_umap:
        # -----------------------------------------------------------------
        # 4-7 (skipped). Reuse a previously saved UMAP coordinates CSV.
        # -----------------------------------------------------------------
        log_status(f"Steps 4-7: Found existing UMAP coordinates at {csv_path}; "
                   "skipping subsample/preprocessing/PCA/neighbors/UMAP and reusing it.")
        # converters={0: str} keeps the cell-ID index column as text; otherwise
        # pandas infers it as float64 and silently rounds off the trailing
        # digits of these ~19-digit cell IDs (float64 only has ~15-17 sig figs).
        umap_coords = pd.read_csv(csv_path, index_col=0, converters={0: str})
        keep_mask = adata.obs.index.isin(umap_coords.index)
        # materialize_subset() reads only the matched cells from disk if adata
        # is still backed (e.g. the whole-brain/all-cell-types case, where none
        # of the filters in Step 3 triggered a load — or early_cache_hit, where
        # Step 3 never ran at all) instead of materializing the full dataset
        # just to throw most of it away right here.
        adata = materialize_subset(adata, keep_mask)
        # Only actually adds anything when Step 2's merge was skipped
        # (early_cache_hit) — umap_coords.csv already carries whichever of
        # these Step 8 saved, so there's no need to re-read/re-join the full
        # multi-million-row metadata CSV just to get them back. A no-op
        # (columns already present) in the normal found-late case, where Step
        # 2's own merge already put them there.
        for meta_col in ('class', 'subclass', 'supertype', 'cluster', SECTION_COL):
            if meta_col in umap_coords.columns and meta_col not in adata.obs.columns:
                adata.obs[meta_col] = umap_coords.loc[adata.obs.index, meta_col].values
        # Restored from the CSV rather than recomputed: this branch skips
        # sc.pp.neighbors entirely, so the graph Leiden needs doesn't exist
        # here. Simply absent for runs whose CSV predates this column (or
        # whose Step 6 couldn't run it) — nothing downstream requires it.
        if LEIDEN_KEY in umap_coords.columns and LEIDEN_KEY not in adata.obs.columns:
            adata.obs[LEIDEN_KEY] = pd.Categorical(
                umap_coords.loc[adata.obs.index, LEIDEN_KEY].astype(str).values
            )
        metadata_cols = [c for c in ('class', 'subclass', 'supertype', 'cluster', SECTION_COL)
                          if c in adata.obs.columns]
        color_key = 'class' if 'class' in adata.obs.columns else None
        adata.obsm['X_umap'] = umap_coords.loc[adata.obs.index, ['UMAP1', 'UMAP2']].to_numpy()
        # The interactive UMAP viewer's Gene mode (Step 9) needs raw counts —
        # normally saved off in Step 5, which every cache-hit path (early or
        # late) skips entirely; adata.X is still the untouched raw counts read
        # from disk here (materialize_subset above never normalizes/transforms
        # it), so this just claims that as layers['counts'] rather than
        # re-deriving it.
        adata.layers['counts'] = adata.X.copy()
        # Same backfill offer as the 'load_existing' startup path — this
        # branch reuses an equally old CSV, just reached from the new-run
        # flow instead. Done after layers['counts'] is claimed above, so the
        # copy it works on already carries everything a fresh run would.
        offer_leiden_backfill(adata, csv_path, CACHE_DIR / f'{session_prefix}_processed.h5ad')
        log_status(f"Aligned {adata.n_obs} cells to the saved UMAP embedding. "
                   f"Coloring by '{color_key}'.")
    else:
        if subsample_target is not None:
            log_status(f"Step 4: Subsampling from {adata.n_obs} to {subsample_target} cells...")
            rng = np.random.RandomState(0)
            keep_mask = np.zeros(adata.n_obs, dtype=bool)
            keep_mask[rng.choice(adata.n_obs, size=subsample_target, replace=False)] = True
            # Same reasoning as the cache-hit branch above: if nothing in Step 3
            # triggered a load (e.g. the whole-brain/all-cell-types case), adata
            # is still backed here, so this is where the actual disk read
            # happens — and it only reads the subsampled cells, not the full
            # unfiltered dataset first.
            adata = materialize_subset(adata, keep_mask)
            log_status(f"Step 4: Subsampled to {adata.n_obs} cells.")
        elif adata.isbacked:
            log_status(f"Step 4: Using all {adata.n_obs} filtered cells (no subsampling); loading into memory...")
            adata = adata.to_memory()
        else:
            log_status(f"Step 4: Using all {adata.n_obs} filtered cells (no subsampling).")

        # ---------------------------------------------------------------------------
        # 5. Preprocessing
        # ---------------------------------------------------------------------------
        # Basic QC filtering
        log_status("Step 5: Filtering cells (min_counts=20) and genes (min_cells=5)...")
        sc.pp.filter_cells(adata, min_counts=20)
        sc.pp.filter_genes(adata, min_cells=5)
        log_status(f"Step 5: After filtering, {adata.n_obs} cells x {adata.n_vars} genes remain.")

        # Save raw counts
        adata.layers['counts'] = adata.X.copy()

        # Normalize + log-transform (MERFISH panels are small, ~500 genes,
        # so we skip highly-variable-gene selection and use all genes for PCA)
        log_status("Step 5: Normalizing, log-transforming, and scaling...")
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        # sc.pp.scale zero-centers the data, which densifies the sparse matrix.
        # Harmless here since the MERFISH panel is small (~500 genes vs. tens of
        # thousands for full transcriptome data), so just silence the warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message='zero-centering a sparse array/matrix densifies it',
                category=UserWarning,
            )
            sc.pp.scale(adata, max_value=10)
        log_status("Step 5: Preprocessing complete.")

        # ---------------------------------------------------------------------------
        # 6. PCA -> neighbors -> UMAP
        # ---------------------------------------------------------------------------
        log_status("Step 6: Running PCA (n_comps=50)...")
        pca_start = time.perf_counter()
        sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
        log_status(f"Step 6: PCA complete in {format_duration(time.perf_counter() - pca_start)}.")
        log_status("Step 6: Computing neighbor graph (n_neighbors=15, n_pcs=50)...")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
        log_status("Step 6: Running UMAP...")
        umap_start = time.perf_counter()
        sc.tl.umap(adata)
        log_status(f"Step 6: UMAP complete in {format_duration(time.perf_counter() - umap_start)}.")

        # Leiden clustering — scanpy's own de-novo grouping of these cells,
        # entirely independent of the Allen ABC taxonomy columns (class/
        # subclass/supertype/cluster) merged in Step 2. Stored under
        # LEIDEN_KEY rather than 'leiden' straight into obs so it can never
        # be confused with ABC's own 'cluster' column, which is an
        # annotation, not a computed clustering.
        #
        # Reuses the neighbor graph built just above, so this is cheap
        # relative to everything preceding it. Non-fatal if it can't run:
        # leidenalg/igraph are optional extras, and a missing one shouldn't
        # cost the user the whole (already-computed) UMAP.
        log_status("Step 6: Running Leiden clustering...")
        leiden_start = time.perf_counter()
        try:
            # flavor/n_iterations pinned explicitly: leaving flavor unset
            # emits a FutureWarning in scanpy 1.12 about the default
            # changing, and 'igraph' with n_iterations=2 is what that
            # warning steers callers toward.
            sc.tl.leiden(adata, key_added=LEIDEN_KEY, flavor='igraph', n_iterations=2,
                         resolution=LEIDEN_RESOLUTION)
            log_status(f"Step 6: Leiden found {adata.obs[LEIDEN_KEY].nunique()} clusters in "
                       f"{format_duration(time.perf_counter() - leiden_start)}.")
        except Exception as e:
            log_status(f"Step 6: Leiden clustering unavailable ({e}); continuing without it.")
            # Surfaced as a dialog too, not just in the console: the run
            # otherwise carries on to a perfectly normal-looking viewer that
            # silently has no Leiden level, with the only clue several
            # screens back in the terminal.
            show_error_dialog("Leiden clustering failed", leiden_failure_message(e))

    # ---------------------------------------------------------------------------
    # 7. Plot
    # ---------------------------------------------------------------------------
    log_status(f"Step 7: Saving UMAP plot to {run_folder}...")
    sc.settings.figdir = run_folder
    DEFAULT_DPI_SAVE = 150  # scanpy's built-in default
    SAVE_DPI = 3 * DEFAULT_DPI_SAVE  # triple saved-PNG resolution
    # sc.set_figure_params (scanpy=True, its own default) doesn't just apply
    # dpi_save — it also overwrites several *on-screen* rcParams (figure.dpi
    # defaults to 80, among others scanpy styles) for the rest of the process,
    # not just for the plots this step is about to save. Step 9's own window
    # below targets a fixed *pixel* size on screen regardless of dpi, but its
    # text is sized in points (rendered as points * dpi/72 pixels) — so a
    # lingering rcParams['figure.dpi'] of 80 instead of matplotlib's true
    # default of 100 made that same point size render visibly smaller relative
    # to the window there, only in a run that actually went through this step.
    # Snapshotting rcParams now and restoring them once this step's own
    # plotting is done (right before Step 8) keeps that styling scoped to just
    # these saved files, so Step 9 always sees the same rcParams regardless of
    # whether this step ran at all — matching "load existing", which skips it
    # entirely.
    rc_params_before_step7 = dict(matplotlib.rcParams)
    sc.set_figure_params(dpi_save=SAVE_DPI)

    main_plot_path = run_folder / 'class_plot.png'
    if color_key is not None:
        ax = sc.pl.umap(adata, color=color_key, size=2, show=False)
        log_status(f"Step 7: Labeling each '{color_key}' group with its ID number...")
        add_category_id_labels(adata, ax, color_key)
    else:
        ax = sc.pl.umap(adata, size=2, show=False)

    ax.figure.savefig(main_plot_path, dpi=SAVE_DPI, bbox_inches='tight')
    ax.figure.savefig(main_plot_path.with_suffix('.svg'), bbox_inches='tight')
    log_status(f"Step 7: Plot saved to {main_plot_path}.")
    plt.close(ax.figure)
    # Not auto-opened in a viewer window (see Step 9's interactive viewer for
    # the on-screen equivalent) — this just exists as a file on disk, same as
    # the optional subclass/supertype group plots below, until/unless someone
    # opens it themselves.

    # Optional: a separate set of plots of every subclass/supertype (not just
    # whichever single subclass gets highlighted interactively in Step 9 below)
    # colored on the UMAP, plus their spatial location.
    if 'subclass' in adata.obs.columns:
        if prompt_yes_no(
            "Step 7: Generate optional plots of UMAP subclasses, and spatial plot of subclass/supertype? (y/N): ",
            default=False,
        ):
            # Still generated and saved to disk for the record even though
            # Step 9's interactive viewer now covers the same coloring on
            # demand — never auto-opened in a window (same as main_plot_path
            # above, as of this turn); just a file until someone opens it.
            # One UMAP plot per group isn't optional here — scanpy renders a
            # categorical column entirely gray, with no per-category colors at
            # all, once it has more categories than colors it's willing to
            # assign (subclass/supertype both have far more than that); see
            # save_umap_by_group's docstring. Groups of 20 also just reads
            # better than trying to tell hundreds of colors apart at once.
            for group_col in ('subclass', 'supertype'):
                if group_col not in adata.obs.columns:
                    log_status(f"Step 7: No '{group_col}' column available; skipping {group_col} UMAP plots.")
                    continue
                umap_group_dir = run_folder / f'umap_by_{group_col}'
                log_status(f"Step 7: Plotting {group_col} UMAP groups to {umap_group_dir}...")
                save_umap_by_group(adata, group_col, SAVE_DPI, umap_group_dir)

            # Also show where these subclasses — and, separately, supertypes —
            # actually sit in space, restricted to the sections that were part
            # of this run's ROI/whole-section selection, not the whole,
            # unfiltered dataset. Uses adata_backed (the original, unfiltered
            # backed AnnData from Step 1) rather than the processed/filtered/
            # subsampled `adata`, so each section's full cell population is
            # available for spatial context, same as save_roi_map above.
            roi_section_set = {roi['section'] for roi in rois} if rois else set()
            whole_sections_for_maps = [s for s in (whole_sections_selected or []) if s not in roi_section_set]
            for group_col in ('subclass', 'supertype'):
                if group_col not in adata.obs.columns:
                    log_status(f"Step 7: No '{group_col}' column available; skipping {group_col} spatial maps.")
                    continue
                group_maps_dir = run_folder / f'spatial_maps_{group_col}'
                log_status(f"Step 7: Generating {group_col} spatial maps in {group_maps_dir}...")
                save_group_spatial_maps(
                    adata_backed, abc_cache, section_series, rois, whole_sections_for_maps, group_maps_dir,
                    group_col=group_col, cell_type_selection=cell_type_selection,
                )
        else:
            log_status("Step 7: Skipped the optional subclass/supertype plots.")

    # Restore whatever sc.set_figure_params changed above — see its own comment.
    matplotlib.rcParams.update(rc_params_before_step7)

    # ---------------------------------------------------------------------------
    # 8. Save results
    # ---------------------------------------------------------------------------
    log_status("Step 8: Assembling UMAP coordinates table...")
    umap_coords = pd.DataFrame(
        adata.obsm['X_umap'],
        index=adata.obs.index,
        columns=['UMAP1', 'UMAP2']
    )
    for c in metadata_cols:
        if c in adata.obs.columns:
            umap_coords[c] = adata.obs[c].values
    # Saved alongside the ABC taxonomy columns above, so a cached run can be
    # reloaded with its clustering intact. Absent whenever Step 6 skipped or
    # failed (leidenalg/igraph missing), and absent from every CSV written
    # before this column existed — hence a membership check here and on
    # every read, rather than assuming it's present.
    if LEIDEN_KEY in adata.obs.columns:
        umap_coords[LEIDEN_KEY] = adata.obs[LEIDEN_KEY].values

    if found_existing_umap:
        log_status(f"Step 8: Reused existing UMAP coordinates from {csv_path}; not rewriting it.")
    else:
        log_status(f"Step 8: Writing UMAP coordinates to {csv_path}...")
        umap_coords.to_csv(csv_path)

    # Optionally save the processed AnnData for reuse (kept in the local cache dir, not out_folder)
    h5ad_out_path = CACHE_DIR / f'{session_prefix}_processed.h5ad'
    log_status(f"Step 8: Writing processed AnnData to {h5ad_out_path}...")
    adata.write_h5ad(h5ad_out_path)

    log_status(f"Done. UMAP computed for {adata.n_obs} cells ({cell_type_selection}, {sections_suffix}).")
    log_status(f"Saved: {main_plot_path}, "
               f"{csv_path}, "
               f"{h5ad_out_path}")

    # ---------------------------------------------------------------------------
    # 9. Interactive UMAP viewer (subclass highlight / gene / imputed gene)
    # ---------------------------------------------------------------------------
    log_status("Step 9: Opening the interactive UMAP viewer...")
    try:
        show_interactive_umap_window(
            adata, abc_cache,
            # The shared state already holds whatever the picker loaded
            # (imputed_adata_from_picker is that same object, when set).
            imputed_state=imputed_state,
            adata_backed=adata_backed, rois=rois, run_folder=run_folder,
        )
    except Exception as e:
        log_status(f"Step 9: Could not open the interactive UMAP viewer: {e}")
    log_status("Step 9: Interactive UMAP viewer closed.")


raw_backed_cache = {}
# The imputed gene dataset, once loaded, kept for the life of the process —
# same reasoning as raw_backed_cache. Passed down to every window that can
# load it (section picker, ROI picker, UMAP viewer), so a later session or a
# reloaded previous run reuses it instead of loading it again. Safe to share:
# it's opened read-only (backed='r'), and consumers only read it or copy
# subsets out of it.
session_imputed_state = {'adata': None, 'load_thread': None, 'load_error': None}
# Seeded with the last folder used (persisted across launches — see
# remember_recent_output_folder), or the historical default on a first run,
# then overridden per-session by whatever the startup panel's output-folder
# box is set to (kept in this dict so it persists across sessions — see
# run_session).
_recent_folders = load_recent_output_folders()
# First launch: a folder next to this repository (so, next to the default atlas
# location), which needs no drive letter or user name and isn't cloud-synced
# the way Documents often is. After that, the most recently used folder.
DEFAULT_OUTPUT_FOLDER = SCRIPT_DIR.parent / 'ABC_atlas_browser_output'
_initial_out_folder = resolve_out_folder(_recent_folders[0] if _recent_folders else str(DEFAULT_OUTPUT_FOLDER))
session_state = {'out_folder': _initial_out_folder}
while True:
    run_session(abc_cache, session_state, raw_backed_cache, session_imputed_state)
