"""The data and rendering layer of the ABC atlas browser, importable on its own.

Everything here is independent of any window: reading the cell metadata and the per-section spatial table, the section grid and
single-section pictures (colored by class, subclass, supertype, cluster, a chosen group, or by a gene's expression), gene lookup,
the per-section z-coordinate table. It was moved out of ABC_atlas_browser.py (which imports it back), so that other front ends (the
Tk pickers) can use the same code. Importing this module has no side effects beyond making the local cache folder: no windows,
no downloads, and no matplotlib backend is chosen (pictures are rendered off-screen on plain Agg figures).

The atlas cache object (`abc_cache`) is always passed in, never global.
"""

import os
import re
import json
import math
import time
import hashlib
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import anndata
from PIL import Image
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.image import imread


# Resolved from this script's own folder, not the working directory, so the app
# finds the same data wherever it's launched from.
SCRIPT_DIR = Path(__file__).resolve().parent


# Local cache dir for anything expensive to regenerate but cheap to keep on
# disk (the section-picker grid image, processed .h5ad files, ...). Defined up
# here because the atlas location below is remembered in it.
CACHE_DIR = SCRIPT_DIR / 'cache_local'


# The interactive UMAP viewer's own Settings dialog (hover-highlighting
# on/off, the multi-gene overlay's genes-4-6 marker shape) — a real user
# preference, not disposable/regenerable data, so it lives next to the
# script rather than inside CACHE_DIR.
SETTINGS_FILE = SCRIPT_DIR / 'viewer_settings.json'


# --- Which z coordinate sections are labelled with, everywhere in the app (section panels, hover readouts, the section grid). Set from
# the launcher's Settings button; always kept in viewer_settings.json, whether or not the viewer's own settings persist.
#   'section': the atlas's own section coordinate (the z column of the cell metadata, mm; sections are 0.2 mm apart)
#   'ccf_ap':  the section's position along the Allen CCF's anterior-posterior axis (mm from the front; x_ccf, the median of its cells)
#   'paxinos': that, relative to bregma and counted positive to the front as Paxinos does: bregma_ap_mm - x_ccf. Approximate: there is no
#              exact CCF-to-Paxinos mapping, only registrations such as Chon et al. 2019, and the bregma position used is a setting.
Z_MODE_OPTIONS = (
    ('section', "Section z: the atlas's own section coordinate (mm)"),
    ('ccf_ap', "CCF AP: position along the Allen CCF's front-to-back axis (mm)"),
    ('paxinos', "Paxinos AP: relative to bregma, positive toward the front (mm; approximate)"),
)


DEFAULT_BREGMA_CCF_AP_MM = 5.40


# (these settings are always kept in viewer_settings.json, whether or not the viewer's own settings persist; 'tk_section_window':
# open the single-section window as the Tk version, abc_tk_picker.py, instead of the matplotlib one)
COORD_SETTINGS = {'z_mode': 'section', 'bregma_ap_mm': DEFAULT_BREGMA_CCF_AP_MM, 'tk_section_window': False}


def load_coordinate_settings():
    """Read the z-coordinate setting from SETTINGS_FILE into COORD_SETTINGS (anything unreadable leaves the defaults)."""
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            on_disk = json.load(f)
        if not isinstance(on_disk, dict):
            return
        if on_disk.get('z_mode') in dict(Z_MODE_OPTIONS):
            COORD_SETTINGS['z_mode'] = on_disk['z_mode']
        COORD_SETTINGS['tk_section_window'] = bool(on_disk.get('tk_section_window', False))
        bregma = float(on_disk.get('bregma_ap_mm'))
        if 0.0 < bregma < 13.5:
            COORD_SETTINGS['bregma_ap_mm'] = bregma
    except Exception:
        pass


def save_coordinate_settings():
    """Write the z-coordinate setting into SETTINGS_FILE, keeping whatever else is in it. Best effort."""
    on_disk = {}
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            on_disk = loaded
    except Exception:
        pass
    on_disk['z_mode'] = COORD_SETTINGS['z_mode']
    on_disk['bregma_ap_mm'] = COORD_SETTINGS['bregma_ap_mm']
    on_disk['tk_section_window'] = bool(COORD_SETTINGS['tk_section_window'])
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(on_disk, f, indent=2)
    except Exception as e:
        print(f"Warning: could not save the coordinate setting ({e})")


# The per-section values behind those labels: {'z': {section label: mm}, 'ccf_ap': {section label: mm}}, each the median over the
# section's cells, read straight from the atlas's cell metadata (and CCF) tables, once, in a background thread, and kept in a small
# file next to the other caches (keyed on the source files) so later launches only read that.
SECTION_COORDS_CACHE_VERSION = 1


section_coords_state = {'table': None, 'thread': None, 'error': None, 'with_ccf': False}


section_coords_lock = threading.Lock()


def _read_section_coordinates(abc_cache, with_ccf):
    meta_path = abc_cache.get_metadata_path(directory='MERFISH-C57BL6J-638850', file_name='cell_metadata_with_cluster_annotation')
    ccf_path = abc_cache.get_metadata_path(directory=CCF_DIRECTORY, file_name=CCF_COORDS_FILE_NAME) if with_ccf else None
    key = '|'.join(f"{p}|{os.stat(p).st_mtime_ns}|{os.stat(p).st_size}" for p in (meta_path, ccf_path) if p is not None)
    cache_path = CACHE_DIR / f"section_coords_v{SECTION_COORDS_CACHE_VERSION}_{hashlib.md5(key.encode('utf-8')).hexdigest()[:16]}.json"
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            table = json.load(f)
        if isinstance(table, dict) and 'z' in table:
            return table
    except Exception:
        pass
    header = pd.read_csv(meta_path, nrows=0).columns.tolist()
    wanted = [header[0], SECTION_COL] + (['z'] if 'z' in header else [])
    meta = pd.read_csv(meta_path, index_col=0, converters={0: str}, usecols=wanted)
    table = {'z': {}, 'ccf_ap': {}}
    if 'z' in meta.columns:
        z = pd.to_numeric(meta['z'], errors='coerce').groupby(meta[SECTION_COL]).median().dropna()
        table['z'] = {str(label): float(value) for label, value in z.items()}
    if ccf_path is not None:
        ccf_header = pd.read_csv(ccf_path, nrows=0).columns.tolist()
        ccf = pd.read_csv(ccf_path, index_col=0, dtype={ccf_header[0]: str}, usecols=[ccf_header[0], 'x'])
        sections = meta[SECTION_COL].reindex(ccf.index)
        ap = pd.to_numeric(ccf['x'], errors='coerce').groupby(sections).median().dropna()
        table['ccf_ap'] = {str(label): float(value) for label, value in ap.items()}
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(table, f)
    except Exception as e:
        print(f"Could not save the section coordinate table ({e}).")
    return table


def start_section_coordinate_load(abc_cache):
    """Start reading the per-section coordinate table in the background, if that hasn't been done (or the setting now needs the CCF
    part that an earlier read left out). Never raises."""
    needs_ccf = COORD_SETTINGS['z_mode'] != 'section'
    try:
        with_ccf = needs_ccf and not ccf_files_missing(abc_cache)
    except Exception:
        with_ccf = False
    with section_coords_lock:
        state = section_coords_state
        if state['thread'] is not None and (state['with_ccf'] or not with_ccf):
            return
        state.update(table=None, error=None, with_ccf=with_ccf)

        def work():
            try:
                table = _read_section_coordinates(abc_cache, with_ccf)
                section_coords_state['table'] = table
            except Exception as e:
                section_coords_state['error'] = e
                print(f"Could not read the section coordinates ({e}).")

        state['thread'] = threading.Thread(target=work, daemon=True)
        state['thread'].start()


def section_z_text(section_label, abc_cache=None, wait=False, short=False):
    """The section's z in the coordinate chosen in Settings, as text ('' if it isn't known, or not yet read: `wait` makes this
    wait for the table, which is quick after the first time). `short` leaves out the unit, for tight places."""
    if abc_cache is not None:
        start_section_coordinate_load(abc_cache)
    thread = section_coords_state['thread']
    if wait and thread is not None and thread.is_alive():
        thread.join()
    table = section_coords_state['table']
    if not table:
        return ''
    mode = COORD_SETTINGS['z_mode']
    label = str(section_label)
    if mode == 'section':
        value = table['z'].get(label)
    else:
        value = table['ccf_ap'].get(label)
        if value is not None and mode == 'paxinos':
            value = COORD_SETTINGS['bregma_ap_mm'] - value
    if value is None:
        return ''
    unit = '' if short else ' mm'
    if mode == 'section':
        return f"z {value:.2f}{unit}" if short else f"z = {value:.2f} mm"
    if mode == 'ccf_ap':
        return f"CCF AP {value:.2f}{unit}" if short else f"CCF AP = {value:.2f} mm"
    return f"AP {value:+.2f}{unit}" if short else f"Paxinos AP = {value:+.2f} mm (approx.)"


# ---------------------------------------------------------------------------
# Cell type subset selection
# ---------------------------------------------------------------------------
# Non-neuron classes in the ABC atlas taxonomy are numbered 30-34 (e.g.
# "30 Astro-Epen", ..., "34 Immune"); everything else is a neuron class.
NON_NEURON_CLASS_IDS = {30, 31, 32, 33, 34}


# Column identifying which physical tissue section (coronal slice) each cell
# came from, e.g. "C57BL6J-638850.36".
SECTION_COL = 'brain_section_label'


# Prints a "[zoom-filter] ..." line every time filter_main_scatter_to_
# viewport (or its section-panel equivalent) runs, showing the cell count
# before/after filtering — useful while tuning ZOOM_BITMAP_ONLY_MAX_
# MULTIPLIER itself, noisy otherwise (it fires on every settle past the
# threshold), so off by default.
ZOOM_DEBUG_DIAGNOSTICS = True  # TODO: set back to False once the section-cache speed/orientation issue is confirmed fixed


# Prefixed onto per-section/per-gene cache filenames so they don't collide if
# another Allen ABC dataset is added alongside this one later.
DATASET_NAME = 'merfish_c57bl6j_638850'


# The section-picker grid always shows every section, neuron-only, regardless
# of which cell type / section subset is chosen for the actual UMAP run — so
# there's exactly one canonical cached image, reused across every run.
GRID_CACHE_PNG = CACHE_DIR / f'{DATASET_NAME}_section_picker_grid.png'


GRID_CACHE_LAYOUT = CACHE_DIR / f'{DATASET_NAME}_section_picker_grid_layout.json'


GRID_CACHE_DPI = 375  # 300 * 1.25


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


SPATIAL_CSV_CHUNK_SIZE = 200_000  # see the chunked read in load_section_spatial_coords()


SPATIAL_CSV_CACHE_VERSION = 2  # v2: adds the section 'z' column


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
        wanted_cols = [c for c in ['x', 'y', 'z', 'class', 'subclass', 'supertype', 'cluster'] if c in header_cols]

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


# ---------------------------------------------------------------------------
# CCF coordinates (Allen common coordinate framework positions) and the 3D view
# ---------------------------------------------------------------------------
# The cell metadata's own x/y/z are in each section's imaging frame: z is just the section's position
# (about 59 discrete planes, 0.2 mm apart), so a 3D plot of them looks like stacked slices. The CCF
# file registers every cell into the Allen reference brain: continuous x (anterior -> posterior),
# y (dorsal -> ventral) and z (left -> right), all in mm, plus a parcellation index naming the region.
# Not part of the normal cache; downloaded (with a confirmation) the first time it is needed.
CCF_DIRECTORY = 'MERFISH-C57BL6J-638850-CCF'


CCF_COORDS_FILE_NAME = 'ccf_coordinates'


CCF_PARCELLATION_DIRECTORY = 'Allen-CCF-2020'


CCF_PARCELLATION_FILE_NAME = 'parcellation_to_parcellation_term_membership_acronym'


CCF_FILES = ((CCF_DIRECTORY, CCF_COORDS_FILE_NAME), (CCF_PARCELLATION_DIRECTORY, CCF_PARCELLATION_FILE_NAME))


def metadata_file_status(abc_cache, directory, file_name):
    """(local_path, expected_size_bytes, is_present) for an atlas metadata file, without downloading
    it; (None, None, None) if abc_atlas_access's internals no longer answer this (same private-API
    reach as is_data_file_cached)."""
    try:
        attributes = abc_cache.cache._manifest.get_file_attributes(directory=directory, file_name=file_name)
        return Path(attributes.local_path), int(attributes.file_size), bool(abc_cache.cache._file_exists(attributes))
    except Exception:
        return None, None, None


def ccf_files_missing(abc_cache):
    """[(directory, file_name, expected_bytes, local_path)] of the CCF files not yet in the local cache."""
    missing = []
    for directory, file_name in CCF_FILES:
        local_path, size, present = metadata_file_status(abc_cache, directory, file_name)
        if present is not True:
            missing.append((directory, file_name, size, local_path))
    return missing


IMPUTED_DATASET_DIRECTORY = 'MERFISH-C57BL6J-638850-imputed'


IMPUTED_DATASET_FILE_NAME = 'C57BL6J-638850-imputed/log2'


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
    saved_img = imread(cache_png)
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


def render_group_highlight_array(section_series, section_label, spatial, level, number, cached_layout, img_w, img_h, dpi=150):
    """Render `section_label` with the cells of one taxonomy group marked: those whose `level` (class, subclass, supertype or cluster)
    ID, the number a label such as '05 IT-ET Glut' starts with, is `number` are drawn in gold, every other cell dim grey, on black.
    Placed like render_gene_expression_array's picture (the axes exactly at cached_layout's pixel_bbox and data limits), so it can be
    swapped in for the others. Not cached on disk. Returns (RGBA uint8 array, number of cells in the group, the group's label, '' if
    this section has none of it). Raises ValueError if the level or spatial coordinates aren't available."""
    if spatial is None or 'x' not in spatial.columns or 'y' not in spatial.columns:
        raise ValueError("spatial x/y coordinates not available")
    if level not in spatial.columns:
        raise ValueError(f"'{level}' labels not available in the cell metadata")
    section_spatial = spatial[(section_series == section_label).to_numpy()]
    xs = pd.to_numeric(section_spatial['x'], errors='coerce').to_numpy()
    ys = pd.to_numeric(section_spatial['y'], errors='coerce').to_numpy()
    ids = extract_leading_numeric_id(section_spatial[level]).to_numpy()
    names = section_spatial[level].astype(str).to_numpy()
    valid = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys, ids, names = xs[valid], ys[valid], ids[valid], names[valid]
    if len(xs) == 0:
        raise ValueError(f"no spatial coordinates for section {section_label}")
    in_group = ids == float(number)
    label = str(names[in_group][0]).strip() if in_group.any() else ''

    fig = Figure(figsize=(img_w / dpi, img_h / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor('black')
    px_x0, px_x1, px_y0, px_y1 = cached_layout['pixel_bbox']
    ax = fig.add_axes([px_x0 / img_w, px_y0 / img_h, (px_x1 - px_x0) / img_w, (px_y1 - px_y0) / img_h])
    ax.set_facecolor('black')
    ax.scatter(xs[~in_group], ys[~in_group], c=[[0.28, 0.28, 0.28]], s=3, linewidths=0)
    if in_group.any():
        ax.scatter(xs[in_group], ys[in_group], c=[[1.0, 0.84, 0.0]], s=10, linewidths=0)
    ax.set_xlim(cached_layout['data_xlim'])
    ax.set_ylim(cached_layout['data_ylim'])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy(), int(in_group.sum()), label


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

    fig = Figure(figsize=(ncols * 2.3, nrows * 1.5 + 0.5))     # (off-screen, not pyplot's: no window, no backend)
    FigureCanvasAgg(fig)
    axes = fig.subplots(nrows, ncols, squeeze=False)
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
    img = imread(GRID_CACHE_PNG)
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

    print(f"Cached section grid image to {GRID_CACHE_PNG} ({img_w}x{img_h}px, {n} sections).")
    return img, layout


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


def section_coordinates_ready():
    """True once the table has been read (or reading it failed, so nothing more will come)."""
    return section_coords_state['table'] is not None or section_coords_state['error'] is not None


CACHE_DIR.mkdir(parents=True, exist_ok=True)
