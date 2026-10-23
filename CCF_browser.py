"""CCF browser: scroll through the Allen CCF annotation volume (coronal, sagittal or horizontal slices) and plot cells
from CSVs on them.

The volume is the 25 um CCF annotation (annotation_25.nii.gz) from the ABC atlas download that ABC_atlas_browser.py
uses: it is looked up in the same atlas cache folder, and downloaded (3.6 MB, after asking) if it is not there yet.
Each coronal slice is shown with its regions coloured as in the atlas; the scroll bar along the bottom (also the mouse
wheel and the arrow keys) moves through the 528 slices, front to back.

Cells come from up to two CSVs, each chosen with its button, pasted as a path into its box, or dragged from Windows
Explorer into it (dropped anywhere else, a file goes into the next empty slot, else replaces the older of the two). Two layouts are recognised:
  * the coordinates_*.csv written by ABC_atlas_browser.py's "Export coordinates" button (CCF positions in mm);
  * the marker export of MBF NeuroInfo (an "Atlas" x/y/z in um, plus the region of each detection).
Each group of cells (the group_id, or the marker) gets its own colour and marker shape. A floating info window beside
the image shows the slice, the cursor's coordinates and region, and the cell nearest the cursor.

"Generate meshes" makes the surface meshes (.vtk, as in the meshes_coronal folder of MBF's Allen2017 atlas for NeuroInfo)
of regions added to an atlas, e.g. with the Paint panel; see atlas_meshes.py. It needs the vtk library (offered on first
use).

Run:  python CCF_browser.py
Mouse: shift+wheel = next/previous slice (ctrl+shift: 10 slices); wheel = zoom about the cursor (Settings can swap it
with ctrl+wheel); +/- also zoom; drag = pan; 0 = fit to window.
"""
import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


def ensure_libraries(required):
    """Install, with pip into this Python, the libraries ({import name: pip name}) that are missing, as the other tools do: the shared .venv of
    the RUN_*.bat files has only those of whichever tool was run first, so the browser must not assume the rest are there. A packaged
    program has its libraries inside."""
    missing = [pip for module, pip in required.items() if importlib.util.find_spec(module) is None]
    if not missing or getattr(sys, 'frozen', False):
        return
    print(f"Installing missing libraries ({', '.join(missing)}) into {sys.executable}...", flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary', ':all:', *missing])
    importlib.invalidate_caches()
    still = [pip for module, pip in required.items() if importlib.util.find_spec(module) is None]
    if still:
        sys.exit(f"Couldn't install {', '.join(still)} for Python {sys.version.split()[0]}; a very new Python often has no prebuilt libraries "
                 f"yet (use Python 3.13: see ensure_python.bat). To install them by hand:  python -m pip install {' '.join(still)}")


ensure_libraries({'numpy': 'numpy', 'pandas': 'pandas', 'PIL': 'pillow'})

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox, ttk  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / 'cache_local'
ATLAS_CACHE_LOCATION_FILE = CACHE_DIR / 'abc_atlas_cache_location.json'   # shared with ABC_atlas_browser.py
SETTINGS_FILE = CACHE_DIR / 'ccf_browser_settings.json'
DEFAULT_ATLAS_CACHE_DIR = SCRIPT_DIR.parent / 'abc_atlas_cache'

VOLUME_DIRECTORY = 'Allen-CCF-2020'
VOLUME_FILE_NAME = 'annotation_25'
REGION_TABLES = ('acronym', 'name', 'color')   # parcellation_to_parcellation_term_membership_<this>
VOXEL_UM = 25.0
N_AP, N_DV, N_ML = 528, 320, 456        # the volume as [anterior-posterior, dorsal-ventral, left-right]
VOLUME_SHAPE = (N_AP, N_DV, N_ML)
CCF_SHAPE, CCF_VOXEL_UM = VOLUME_SHAPE, VOXEL_UM
MIDLINE_UM = N_ML * VOXEL_UM / 2        # left-right positions at or beyond this are the left hemisphere (as in the atlas)
BACKGROUND = (24, 24, 24)
MAX_GROUPS = 12
HIT_RADIUS_PX = 9

GROUP_COLORS = ((230, 25, 75), (60, 180, 75), (67, 99, 216), (245, 130, 49), (145, 30, 180), (66, 212, 244),
                (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (255, 225, 25), (170, 110, 40))
GROUP_SHAPES = ('circle', 'plus', 'square', 'triangle', 'diamond', 'x')
SHAPE_SYMBOLS = {'circle': '●', 'plus': '+', 'square': '■', 'triangle': '▲', 'diamond': '◆',
                 'x': '×'}
ATLAS_MODES = ('Regions', 'Regions (dim)', 'Boundaries')
DIFF_MODE = 'Differences from CCF 2020'   # offered only while another atlas is loaded
DIFF_MODE_MERGED = 'Differences (X-unassigned = X)'   # the same, ignoring a trailing '-unassigned' on region names
DIFF_MODES = (DIFF_MODE, DIFF_MODE_MERGED)
UNASSIGNED_SUFFIX = '-unassigned'
PANE_GAP = 2                 # pixels between the two panes of the Differences view
# Bregma in CCF coordinates (mm: AP from the anterior edge, DV from the dorsal edge, ML from the left). The CCF does not
# define bregma, so this is a convention; the default is, as far as I know, the one used by the International Brain
# Laboratory's atlas tools, and sources differ by a few hundred um. Editable in Settings.
BREGMA_DEFAULT_MM = (5.400, 0.332, 5.739)
TIP_DELAY_MS = 350           # how long the pointer rests before the region tool tip appears
DIFF_DIM_PERCENT = 18        # brightness of the regions that do not differ, in the Differences view
VIEWS = ('Coronal', 'Sagittal', 'Horizontal')
VIEW_AXIS = {'Coronal': 0, 'Sagittal': 2, 'Horizontal': 1}        # volume axis [AP, DV, ML] stepped through
VIEW_CELL_AXES = {'Coronal': (2, 1), 'Sagittal': (0, 1), 'Horizontal': (2, 0)}   # axes drawn across (u) and down (v)
VIEW_IMAGE_SIZE = {'Coronal': (N_ML, N_DV), 'Sagittal': (N_AP, N_DV), 'Horizontal': (N_ML, N_AP)}   # (w, h) voxels
VIEW_EDGE_LABELS = {'Coronal': ('dorsal', 'ventral', 'right', 'left'),
                    'Sagittal': ('dorsal', 'ventral', 'anterior', 'posterior'),
                    'Horizontal': ('anterior', 'posterior', 'right', 'left')}   # top, bottom, left, right
AXIS_NAMES = ('AP', 'DV', 'ML')


def set_geometry(shape, voxel_um):
    """Make the module's volume constants those of the atlas now on screen (the CCF's, or another atlas's: a loaded
    volume is always [AP, DV, ML] with isotropic voxels of `voxel_um`)."""
    global VOXEL_UM, N_AP, N_DV, N_ML, VOLUME_SHAPE, MIDLINE_UM
    N_AP, N_DV, N_ML = (int(n) for n in shape)
    VOLUME_SHAPE = (N_AP, N_DV, N_ML)
    VOXEL_UM = float(voxel_um)
    MIDLINE_UM = N_ML * VOXEL_UM / 2
    VIEW_IMAGE_SIZE.update({'Coronal': (N_ML, N_DV), 'Sagittal': (N_AP, N_DV), 'Horizontal': (N_ML, N_AP)})


def view_slice(volume, view, index):
    """The (rows, columns) image of region labels of one slice in the given view."""
    if view == 'Coronal':
        return volume[index]                                   # (DV, ML)
    if view == 'Sagittal':
        return np.ascontiguousarray(volume[:, :, index].T)     # (DV, AP)
    return volume[:, index, :]                                 # (AP, ML)


# ---- optional drag-and-drop ---------------------------------------------------------------------------------------
def import_drag_and_drop():
    """(TkinterDnD, DND_FILES) from tkinterdnd2, installing it first if it is missing; (None, None) if that fails
    (everything else still works: choose the file with the button, or paste its path)."""
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
        return TkinterDnD, DND_FILES
    except ImportError:
        pass
    print("Installing tkinterdnd2 (drag-and-drop of a CSV into the box)...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'tkinterdnd2'], check=True)
        from tkinterdnd2 import DND_FILES, TkinterDnD
        return TkinterDnD, DND_FILES
    except Exception as e:
        print(f"Could not set up drag-and-drop ({e}); paste a path or use the button instead.")
        return None, None


# ---- the atlas ---------------------------------------------------------------------------------------------------
def read_nifti_volume(path, with_header=False):
    """The voxel array of a (gzipped) NIfTI-1 file as [i, j, k] (the file's own axis order), without nibabel."""
    import gzip
    import struct
    with open(path, 'rb') as plain:
        gzipped = plain.read(2) == bytes([0x1f, 0x8b])   # .nii.gz, or an uncompressed .nii
    with (gzip.open if gzipped else open)(path, 'rb') as handle:
        raw = handle.read()
    header = raw[:348]
    dims = struct.unpack('<8h', header[40:56])
    shape = tuple(dims[1:1 + dims[0]])
    code, = struct.unpack('<h', header[70:72])
    dtypes = {2: '<u1', 4: '<i2', 8: '<i4', 16: '<f4', 64: '<f8', 256: '<i1', 512: '<u2', 768: '<u4'}
    if code not in dtypes:
        raise ValueError(f"unsupported NIfTI data type {code}")
    offset = int(struct.unpack('<f', header[108:112])[0])
    count = int(np.prod(shape))
    array = np.frombuffer(raw, dtype=dtypes[code], offset=offset, count=count).reshape(shape[::-1]).T
    return (array, raw[:max(348, offset)]) if with_header else array


def find_atlas_cache_dir():
    """The ABC atlas download folder: the location ABC_atlas_browser.py saved, else the default next to the repository,
    else the user is asked (and the answer saved where ABC_atlas_browser.py will find it too). None if cancelled."""
    try:
        saved = json.loads(ATLAS_CACHE_LOCATION_FILE.read_text(encoding='utf-8')).get('path')
    except (OSError, ValueError, AttributeError):
        saved = None
    if saved and Path(saved).is_dir():
        return Path(saved)
    if DEFAULT_ATLAS_CACHE_DIR.is_dir():
        return DEFAULT_ATLAS_CACHE_DIR
    chosen = filedialog.askdirectory(
        title="Where is (or should go) the ABC atlas download folder (abc_atlas_cache)?",
        initialdir=str(DEFAULT_ATLAS_CACHE_DIR.parent))
    if not chosen:
        return None
    chosen = Path(chosen)
    chosen.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ATLAS_CACHE_LOCATION_FILE.write_text(json.dumps({'path': str(chosen)}, indent=2), encoding='utf-8')
    return chosen


ABC_ATLAS_ACCESS = 'abc_atlas_access @ https://github.com/alleninstitute/abc_atlas_access/archive/refs/heads/main.zip'   # (the zip: no git needed)
_abc_atlas_access_problem = None      # why it could not be installed, once that has been tried (it is not tried again in the same run)


def import_project_cache():
    """AbcProjectCache from the Allen Institute's abc_atlas_access, installing that first if it is missing (it comes from GitHub, not PyPI).
    Raises ImportError, saying why, if it cannot be had; only the first failure tries to install."""
    global _abc_atlas_access_problem
    try:
        from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
        return AbcProjectCache
    except ImportError:
        pass
    if _abc_atlas_access_problem is None and not getattr(sys, 'frozen', False):
        print(f"Installing abc_atlas_access (the Allen Institute's atlas download library) into {sys.executable}...", flush=True)
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', ABC_ATLAS_ACCESS])
        importlib.invalidate_caches()
        try:
            from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
            return AbcProjectCache
        except ImportError:
            _abc_atlas_access_problem = (f"abc_atlas_access could not be installed (pip said {result.returncode}); install it by hand with  "
                                         f"python -m pip install \"{ABC_ATLAS_ACCESS}\"")
    raise ImportError(_abc_atlas_access_problem or 'abc_atlas_access is not installed')


def volume_status(base):
    """(local path or None, expected size in bytes or None, is_present) of the annotation volume under `base`,
    without downloading anything."""
    try:
        from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
        cache = AbcProjectCache.from_local_cache(base)
        attributes = cache.cache._manifest.get_file_attributes(directory=VOLUME_DIRECTORY, file_name=VOLUME_FILE_NAME)
        path = Path(attributes.local_path)
        return path, int(attributes.file_size), path.exists()
    except Exception:
        matches = sorted(Path(base).glob(f'image_volumes/{VOLUME_DIRECTORY}/*/{VOLUME_FILE_NAME}.nii.gz'))
        return (matches[0] if matches else None), None, bool(matches)


def load_atlas(base, download):
    """(volume [AP, DV, ML] uint16, region table or None, colour lookup (n, 3) uint8). Blocking: run it on a thread.
    `download` says the volume may be fetched if it is missing."""
    path, _size, present = volume_status(base)
    cache = None
    if not present:
        if not download:
            raise FileNotFoundError("the CCF annotation volume is not in the atlas cache")
        AbcProjectCache = import_project_cache()
        cache = AbcProjectCache.from_cache_dir(base)
        path = Path(cache.get_data_path(directory=VOLUME_DIRECTORY, file_name=VOLUME_FILE_NAME))
    volume = np.ascontiguousarray(read_nifti_volume(path))
    if tuple(volume.shape) != CCF_SHAPE:   # not VOLUME_SHAPE: that follows the atlas on screen
        raise ValueError(f"unexpected annotation volume shape {volume.shape}")

    table = None
    try:
        AbcProjectCache = import_project_cache()
        if cache is None:
            try:
                cache = AbcProjectCache.from_local_cache(base)
                for kind in REGION_TABLES:
                    cache.get_metadata_path(directory=VOLUME_DIRECTORY,
                                            file_name=f'parcellation_to_parcellation_term_membership_{kind}')
            except Exception:
                cache = AbcProjectCache.from_cache_dir(base)  # a missing small table: fetch it
        frames = {}
        for kind in REGION_TABLES:
            frame = pd.read_csv(cache.get_metadata_path(
                directory=VOLUME_DIRECTORY, file_name=f'parcellation_to_parcellation_term_membership_{kind}'))
            frames[kind] = frame.set_index('parcellation_index')
        table = frames['acronym'].add_prefix('acronym_').join(frames['name'].add_prefix('name_')).join(
            frames['color'][['substructure_color']])
        table['detail'] = ('structure: ' + table['acronym_structure'].astype(str) + '   division: '
                           + table['acronym_division'].astype(str))
    except Exception as e:
        print(f"Region names and colours unavailable ({e}); regions are coloured by label number only.")

    return volume, table, region_colors(volume, table), {'voxel_um': CCF_VOXEL_UM, 'legacy': True}


def quiet_atlas_cache_dir():
    """Like find_atlas_cache_dir, but never asks: the saved or default ABC atlas folder, else None."""
    try:
        saved = json.loads(ATLAS_CACHE_LOCATION_FILE.read_text(encoding='utf-8')).get('path')
    except (OSError, ValueError, AttributeError):
        saved = None
    if saved and Path(saved).is_dir():
        return Path(saved)
    return DEFAULT_ATLAS_CACHE_DIR if DEFAULT_ATLAS_CACHE_DIR.is_dir() else None


def comparison_luts(alt_table, alt_size, alt_ids, ref_table, ref_size, merge_unassigned=False):
    """(alt_lut, ref_lut, basis): label -> comparison code for the alternate and the standard atlas, so two voxels have
    the same code exactly when they are the same region. Regions are matched by substructure acronym when both atlases
    name their regions that way (the label numbers of two atlases need not agree), else by region number: the real id
    of a remapped atlas, or the label itself. Code 0 is outside the brain. With `merge_unassigned`, a name ending in
    '-unassigned' counts as the same region as the name without it (e.g. MB-unassigned = MB): it only applies when
    regions are matched by name."""
    def acronyms(table):
        if table is None or 'acronym_substructure' not in table.columns:
            return None
        names = table['acronym_substructure'].fillna('').astype(str)
        if merge_unassigned:
            names = names.where(~names.str.endswith(UNASSIGNED_SUFFIX), names.str[:-len(UNASSIGNED_SUFFIX)])
        return names.to_dict() if names.str.len().gt(0).any() else None

    alt_names, ref_names = acronyms(alt_table), acronyms(ref_table)
    by_name = alt_names is not None and ref_names is not None
    codes = {}

    def build(names, size, ids):
        lut = np.zeros(size, dtype=np.int32)
        for label in range(size):
            real = int(ids[label]) if ids is not None and label < len(ids) else label
            if real == 0:
                continue
            key = (names.get(label) or f"#{real}") if by_name else real
            lut[label] = codes.setdefault(key, len(codes) + 1)
        return lut

    return (build(alt_names, alt_size, alt_ids), build(ref_names, ref_size, None),
            ("region acronym" + (", X-unassigned = X" if merge_unassigned else "")) if by_name else "region number")


def diff_panes(labels, lut, ref_labels, ref_lut, alt_codes, ref_codes):
    """(left_rgb, right_rgb, counts) for the Differences view: the standard atlas's slice and the alternate atlas's
    slice, both with their own region colours markedly dimmed except where the two differ, which keep their normal
    brightness. `counts` = voxels (both have tissue but different regions, tissue only in the alternate, tissue only
    in the standard)."""
    alt, ref = alt_codes[labels], ref_codes[ref_labels]
    changed = alt != ref

    def pane(slice_labels, colors):
        rgb = colors[slice_labels]
        dimmed = (rgb.astype(np.uint16) * DIFF_DIM_PERCENT // 100).astype(np.uint8)
        dimmed[changed] = rgb[changed]
        return dimmed

    counts = (int((changed & (alt > 0) & (ref > 0)).sum()), int((changed & (ref == 0)).sum()),
              int((changed & (alt == 0)).sum()))
    return pane(ref_labels, ref_lut), pane(labels, lut), counts


def region_colors(volume, table):
    """Colour lookup (label -> RGB uint8) for the labels present in the volume: the table's colours (hex strings in
    its `substructure_color` column) where known, otherwise a fixed random colour."""
    top = int(volume.max())
    lut = np.zeros((top + 1, 3), dtype=np.uint8)
    rng = np.random.default_rng(3)
    lut[1:] = rng.integers(60, 256, size=(top, 3))
    if table is not None and 'substructure_color' in table:
        for index, color in table['substructure_color'].items():
            if isinstance(color, str) and color.startswith('#') and len(color) == 7 and 0 <= index <= top:
                lut[index] = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    lut[0] = (0, 0, 0)
    return lut


NRRD_TYPES = {'unsigned short': 'u2', 'uint16': 'u2', 'ushort': 'u2', 'uint16_t': 'u2', 'unsigned char': 'u1',
              'uint8': 'u1', 'uchar': 'u1', 'uint8_t': 'u1', 'short': 'i2', 'int16': 'i2', 'signed short': 'i2',
              'int': 'i4', 'int32': 'i4', 'signed int': 'i4', 'unsigned int': 'u4', 'uint32': 'u4', 'float': 'f4',
              'double': 'f8'}


def read_nrrd_volume(path, details=False):
    """The voxel array of a NRRD file (raw or gzip encoded) as [slowest axis, ..., fastest axis], without pynrrd."""
    import zlib
    with open(path, 'rb') as handle:
        raw = handle.read()
    end = raw.find(b'\n\n')
    if not raw.startswith(b'NRRD') or end < 0:
        raise ValueError("not a NRRD file")
    fields = {}
    lines = raw[:end].decode('latin-1').splitlines()[1:]
    for line in lines:
        if line.startswith('#') or ':' not in line:
            continue
        key, value = line.split(':', 1)
        fields[key.strip().lower()] = value.strip()
    kind = NRRD_TYPES.get(fields.get('type', '').lower())
    if kind is None:
        raise ValueError(f"unsupported NRRD data type {fields.get('type')!r}")
    dtype = np.dtype(('>' if fields.get('endian', 'little').lower() == 'big' else '<') + kind)
    encoding = fields.get('encoding', 'raw').lower()
    body = raw[end + 2:]
    if encoding in ('gzip', 'gz'):
        body = zlib.decompress(body, 31)
    elif encoding != 'raw':
        raise ValueError(f"unsupported NRRD encoding {encoding!r}")
    sizes = tuple(int(n) for n in fields['sizes'].split())
    array = np.frombuffer(body, dtype=dtype, count=int(np.prod(sizes))).reshape(sizes[::-1])
    return (array, fields, lines) if details else array


SAVE_FORMATS = {'.nrrd': "NRRD (.nrrd)", '.nii.gz': "NIfTI-1, gzipped (.nii.gz)", '.nii': "NIfTI-1 (.nii)"}


def volume_format(path):
    """'.nrrd', '.nii.gz' or '.nii' for a file name, or None."""
    name = str(path).lower()
    return next((ext for ext in ('.nii.gz', '.nii', '.nrrd') if name.endswith(ext)), None)


def nifti_header(shape, dtype):
    """A 352-byte NIfTI-1 header (single file, millimetres) with the orientation of the ABC CCF annotation volume."""
    import struct
    code, bits = {'uint16': (512, 16), 'uint32': (768, 32)}[np.dtype(dtype).name]
    header = bytearray(352)
    struct.pack_into('<i', header, 0, 348)
    struct.pack_into('<8h', header, 40, 3, *shape, 1, 1, 1, 1)
    struct.pack_into('<2h', header, 70, code, bits)
    struct.pack_into('<8f', header, 76, 1.0, VOXEL_UM / 1000, VOXEL_UM / 1000, VOXEL_UM / 1000, 0, 0, 0, 0)
    struct.pack_into('<2f', header, 108, 352.0, 1.0)   # vox_offset, scl_slope
    header[123] = 2                                  # xyzt_units: millimetres
    struct.pack_into('<2h', header, 252, 1, 1)       # qform_code, sform_code
    struct.pack_into('<6f', header, 256, -0.5, 0.5, -0.5, -5.695, 5.35, 5.22)
    struct.pack_into('<12f', header, 280, 0, 0, 0.025, -5.695, -0.025, 0, 0, 5.35, 0, -0.025, 0, 5.22)
    header[344:348] = b'n+1\0'
    return bytes(header)


def write_volume(path, values, progress=None, source=None):
    """Write a [AP, DV, ML] array of uint16/uint32 region values as .nrrd, .nii or .nii.gz (chosen by the name). For an
    atlas that was not on the CCF grid, `source` (its info from oriented_volume) puts the array back in the file's own
    axis order and keeps the original header, so the file is written in its original format and orientation."""
    import gzip
    import zlib
    fmt = volume_format(path)
    if progress:
        progress("compressing" if fmt != '.nii' else "writing")
    if source is not None:
        if fmt != source['format']:
            raise ValueError(f"this atlas can only be saved in its own format ({source['format']})")
        flipped = np.flip(values, axis=[a for a in range(3) if source['flips'][a]])
        file_array = np.transpose(flipped, np.argsort(source['perm']))   # [i, j, k] as in the file
        slowest_first = file_array.T
        shape = file_array.shape
    else:
        shape = None
        slowest_first = values if fmt == '.nrrd' else values.T   # the CCF files: see below
    if fmt == '.nrrd':
        kind = {'uint16': 'unsigned short', 'uint32': 'unsigned int'}[values.dtype.name]
        if source is not None:
            skipped = ('type', 'endian', 'encoding', 'byteskip', 'line skip', 'lineskip', 'data file', 'datafile')
            kept = [line for line in source['nrrd_lines'] if line.split(':', 1)[0].strip().lower() not in skipped]
            header = ("NRRD0004\n" + "\n".join(kept) + f"\ntype: {kind}\nendian: little\nencoding: gzip\n\n")
        else:
            header = (f"NRRD0004\n# Written by CCF_browser.py\ntype: {kind}\ndimension: 3\n"
                      f"space: left-posterior-superior\n"
                      f"sizes: {values.shape[2]} {values.shape[1]} {values.shape[0]}\n"
                      f"space directions: ({VOXEL_UM:g},0,0) (0,{VOXEL_UM:g},0) (0,0,{VOXEL_UM:g})\n"
                      "kinds: domain domain domain\nendian: little\nencoding: gzip\nspace origin: (0,0,0)\n\n")
        compressor = zlib.compressobj(6, zlib.DEFLATED, 31)
        with open(path, 'wb') as handle:
            handle.write(header.encode('latin-1'))
            for start in range(0, slowest_first.shape[0], 16):   # in slabs, to keep the temporary copies small
                handle.write(compressor.compress(np.ascontiguousarray(slowest_first[start:start + 16]).tobytes()))
            handle.write(compressor.flush())
    elif fmt in ('.nii', '.nii.gz'):
        if source is not None:
            import struct
            header = bytearray(source['nifti_header'])
            code, bits = {'uint16': (512, 16), 'uint32': (768, 32)}[values.dtype.name]
            struct.pack_into('<2h', header, 70, code, bits)
            struct.pack_into('<2f', header, 112, 1.0, 0.0)   # scl_slope, scl_inter: the stored numbers are the labels
            header = bytes(header)
        else:
            header = nifti_header(values.shape, values.dtype)
        opener = gzip.open(path, 'wb', compresslevel=5) if fmt == '.nii.gz' else open(path, 'wb')
        with opener as handle:
            handle.write(header)
            for start in range(0, slowest_first.shape[0], 16):   # NIfTI stores the first axis fastest
                handle.write(np.ascontiguousarray(slowest_first[start:start + 16]).tobytes())
    else:
        raise ValueError("the file name must end in .nrrd, .nii or .nii.gz")


def read_ontology(path, top):
    """A region table from an Allen-style ontology CSV, for the ids up to `top`: same columns as the ABC table, with the
    lineage of acronyms as the detail line. Needs id, name and acronym columns; colours come from color_hex_triplet or
    color, and the lineage from structure_id_path or, failing that, a parent_id column."""
    frame = pd.read_csv(path, dtype={'color': str, 'color_hex_triplet': str})
    unique = frame.drop_duplicates('id').set_index('id')
    acronyms = unique['acronym']
    frame = unique[unique.index <= top]
    parents = unique['parent_id'].to_dict() if 'parent_id' in unique else {}

    def lineage(row_id):
        if 'structure_id_path' in frame:
            ids = [int(part) for part in str(frame.at[row_id, 'structure_id_path']).split('/') if part.strip().isdigit()]
            return ' > '.join(str(acronyms.get(i, i)) for i in ids if i not in (997, 8))
        chain, current = [], row_id
        while current in unique.index and len(chain) < 30:
            chain.append(str(acronyms.get(current, current)))
            current = parents.get(current)
            if current is None or pd.isna(current):
                break
            current = int(current)
        return ' > '.join(reversed(chain[:-1] if len(chain) > 1 else chain))   # without the root

    color_column = 'color_hex_triplet' if 'color_hex_triplet' in frame else 'color'
    return pd.DataFrame({'acronym_substructure': frame['acronym'], 'name_substructure': frame['name'],
                         'detail': 'lineage: ' + pd.Series([lineage(i) for i in frame.index], index=frame.index),
                         'substructure_color': '#' + frame[color_column].astype(str).str.strip().str.zfill(6)})


OPPOSITE = {'left': 'right', 'right': 'left', 'anterior': 'posterior', 'posterior': 'anterior',
            'superior': 'inferior', 'inferior': 'superior'}
LETTER_WORDS = {'l': 'left', 'r': 'right', 'a': 'anterior', 'p': 'posterior', 's': 'superior', 'i': 'inferior'}


def orientation_transform(vectors, words):
    """How to turn a volume stored as [i, j, k] into [AP, DV, ML] (AP growing posterior, DV growing ventral, ML growing
    toward the left, as the CCF is held here): returns (perm, flips) so that
    np.flip(np.transpose(volume, perm), axes where flips) is the result. `vectors[k]` is the direction in the world
    of file axis k, `words` the names of the world's +x, +y, +z directions (e.g. left, posterior, superior)."""
    roles = {'posterior': (0, False), 'anterior': (0, True), 'inferior': (1, False), 'superior': (1, True),
             'left': (2, False), 'right': (2, True)}
    perm, flips = [None] * 3, [False] * 3
    for k, vector in enumerate(vectors):
        world = int(np.argmax(np.abs(vector)))
        toward = words[world] if vector[world] > 0 else OPPOSITE[words[world]]
        axis, flip = roles[toward]
        if perm[axis] is not None:
            raise ValueError("cannot tell the orientation of this volume: two axes point the same way")
        perm[axis], flips[axis] = k, flip
    return tuple(perm), tuple(flips)


def qform_matrix(b, c, d, pixdim):
    """The 3 x 3 voxel-to-world matrix of a NIfTI qform (quaternion b, c, d; pixdim[0] is qfac, pixdim[1:4] the sizes)."""
    a = np.sqrt(max(0.0, 1.0 - b * b - c * c - d * d))
    rotation = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                         [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                         [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - c * c - b * b]])
    return rotation * np.array([pixdim[1], pixdim[2], pixdim[3] * (-1.0 if pixdim[0] < 0 else 1.0)])


def embedded_nifti_matrix(fields):
    """The voxel-to-world matrix (RAS) from NIfTI fields a converter left in a NRRD header (qform_code, quatern_b, ...,
    srow_x, ...), or None. These describe the original data, whose orientation the NRRD's own 'space directions' can
    contradict (the signs of x and y are wrong in some converted files)."""
    def field(name):   # 'key:=value' lines arrive as 'key' -> '=value'
        return fields[name].lstrip('=').strip()

    try:
        sform, qform = int(field('sform_code')), int(field('qform_code'))
        if sform > 0:
            return np.array([[float(n) for n in field(f'srow_{axis}').split()] for axis in 'xyz'])[:, :3]
        if qform > 0:
            pixdim = [float(field(f'pixdim[{i}]')) for i in range(4)]
            return qform_matrix(float(field('quatern_b')), float(field('quatern_c')), float(field('quatern_d')),
                                pixdim)
    except (KeyError, ValueError):
        pass
    return None


def nrrd_space(fields):
    """(directions, words, voxel um) from NRRD header fields: the world vector of each axis, the names of the world's
    +x/+y/+z, and the voxel size."""
    import re
    space = fields.get('space', '').strip().lower().replace('_', '-')
    words = space.split('-') if '-' in space else [LETTER_WORDS.get(c, '') for c in space]
    if len(words) != 3 or not all(w in OPPOSITE for w in words):
        raise ValueError(f"cannot tell the orientation of this volume: unsupported NRRD space {space!r}")
    vectors = [[float(n) for n in text.split(',')] for text in re.findall(r'\(([^)]*)\)', fields.get('space directions', ''))]
    if len(vectors) != 3 or any(len(v) != 3 for v in vectors):
        raise ValueError("cannot tell the orientation of this volume: no usable 'space directions' in the NRRD header")
    sizes = np.linalg.norm(np.array(vectors), axis=1)
    units = fields.get('space units', '').lower()
    if 'mm' in units or (not units and sizes.max() < 1):
        sizes = sizes * 1000
    embedded = embedded_nifti_matrix(fields)
    if embedded is not None:
        return embedded.T, ('right', 'anterior', 'superior'), sizes
    vectors = np.array(vectors)
    if tuple(words) == ('left', 'posterior', 'superior') and np.all(np.diag(vectors) > 0)             and np.count_nonzero(vectors) == 3:
        # Converted NIfTI files (MBF) label the space LPS but keep the RAS storage order: i to the right, j anterior, k
        # superior, with the signs of x and y lost. Read them as RAS (Settings can still flip an axis).
        return vectors, ('right', 'anterior', 'superior'), sizes
    return vectors, tuple(words), sizes


def nifti_space(header):
    """The same as nrrd_space for the 348+ byte header of a NIfTI-1 file (sform if set, else qform)."""
    import struct
    pixdim = struct.unpack('<8f', header[76:108])
    qform, sform = struct.unpack('<2h', header[252:256])
    if sform > 0:
        rows = np.array(struct.unpack('<12f', header[280:328])).reshape(3, 4)
        matrix = rows[:, :3]
    elif qform > 0:
        matrix = qform_matrix(*struct.unpack('<3f', header[256:268]), pixdim)
    else:
        raise ValueError("cannot tell the orientation of this volume: the NIfTI header has neither sform nor qform")
    unit = {1: 1e6, 2: 1e3, 3: 1.0}.get(header[123] & 7, 1e3)   # to um; mm if unspecified
    return matrix.T, ('right', 'anterior', 'superior'), np.linalg.norm(matrix, axis=0) * unit


def oriented_volume(path, extra_flips=(False, False, False)):
    """(volume [AP, DV, ML], info) from a .nrrd / .nii / .nii.gz volume of any size. A volume with the CCF's dimensions
    (in any axis order) is taken as the CCF grid, 25 um; any other is turned into [AP, DV, ML] using the orientation in
    its header. `info` has 'voxel_um', 'legacy' (CCF grid) and, for the others, what write_volume needs to save it back
    in the file's own layout. Raises ValueError for a volume it cannot orient or with non-cubic voxels."""
    path = Path(path)
    fmt = volume_format(path)
    if fmt == '.nrrd':
        array, fields, lines = read_nrrd_volume(path, details=True)
        file_array = array.T
    elif fmt in ('.nii', '.nii.gz'):
        file_array, header = read_nifti_volume(path, with_header=True)
    else:
        raise ValueError("the file name must end in .nrrd, .nii or .nii.gz")
    if file_array.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got {file_array.shape}")
    if sorted(file_array.shape) == sorted(CCF_SHAPE):
        return (np.transpose(file_array, [file_array.shape.index(n) for n in CCF_SHAPE]),
                {'voxel_um': CCF_VOXEL_UM, 'legacy': True})
    vectors, words, sizes = nrrd_space(fields) if fmt == '.nrrd' else nifti_space(header)
    if sizes.max() > sizes.min() * 1.02:
        raise ValueError(f"the voxels are not cubic ({', '.join(f'{x:.4g}' for x in sizes)} um): not supported")
    perm, flips = orientation_transform(vectors, words)
    flips = tuple(bool(f) != bool(e) for f, e in zip(flips, extra_flips))   # the user's corrections, from Settings
    volume = np.transpose(file_array, perm)
    flip_axes = [a for a in range(3) if flips[a]]
    if flip_axes:
        volume = np.flip(volume, axis=flip_axes)
    info = {'voxel_um': float(sizes.mean()), 'legacy': False, 'perm': perm, 'flips': flips, 'format': fmt}
    info.update({'nrrd_lines': lines} if fmt == '.nrrd' else {'nifti_header': header})
    return volume, info


IMAGE_MAX_NEIGHBOUR_SAME = 0.5   # below this share of equal neighbours, a volume is a greyscale image, not a map of regions


def neighbour_same_share(volume):
    """The share of voxels that equal the next voxel along the fastest axis, counting only pairs with at least one
    voxel that isn't background (the empty background is the same value in every kind of volume). The background is
    taken to be the most common value rather than 0: MBF's Allen2017 template (AllenCoronal_25um.nrrd) fills the space
    around the brain with 1, which counted as background-free made it look like a map of regions. A map of regions is
    made of large patches, so this is high (about 0.9 for the Allen CCF); in a greyscale image neighbouring voxels
    rarely hold exactly the same value (about 0.0005 for a Nissl template, 0.14 for that Allen2017 template, which has
    only 516 grey levels). Every second plane is used, which is plenty."""
    a, b = volume[::2, :, 1:], volume[::2, :, :-1]
    sample = a[::4, ::4, ::4].ravel()
    if sample.dtype.kind in 'iub' and sample.size:
        values, counts = np.unique(sample, return_counts=True)
        background = values[np.argmax(counts)]
    else:
        background = 0
    pairs = (a != background) | (b != background)
    return float((a == b)[pairs].mean()) if pairs.any() else 1.0


def intensity_volume(volume):
    """(uint16 volume, colour lookup) for a greyscale image volume (a Nissl or other microscopy template rather than a map
    of regions): values that already fit 16 bits are kept as they are, anything else (floats, negative or larger values) is
    stretched to 0-65535; the lookup is a grey ramp from the 0.5th to the 99.5th percentile of the non-zero values."""
    if volume.dtype != np.uint16:
        low, high = float(volume.min()), float(volume.max())
        volume = np.rint((volume.astype(np.float32) - low) / max(high - low, 1e-9) * 65535).astype(np.uint16)
    sample = volume[::3, ::3, ::3]
    sample = sample[sample > 0] if (sample > 0).any() else sample.ravel()
    low, high = np.percentile(sample, [0.5, 99.5])
    ramp = np.clip((np.arange(int(volume.max()) + 1, dtype=np.float32) - low) / max(float(high - low), 1.0), 0, 1)
    lut = np.repeat((ramp * 255).astype(np.uint8)[:, None], 3, axis=1)
    lut[0] = (0, 0, 0)
    return np.ascontiguousarray(volume), lut


def load_custom_atlas(path, extra_flips=(False, False, False)):
    """(volume [AP, DV, ML], region table or None, colour lookup, info) from a .nrrd / .nii / .nii.gz volume of any size
    (see oriented_volume). An annotation volume gets names and colours from an ontology*.csv beside it when there is one.
    A greyscale image volume (non-integer values, or too few equal neighbours: see neighbour_same_share) is shown in
    grey instead, with no region table, and info['intensity'] is True."""
    path = Path(path)
    volume, info = oriented_volume(path, extra_flips)
    whole_numbers = volume.dtype.kind != 'f' or np.array_equal(volume, np.round(volume))
    if not whole_numbers or neighbour_same_share(volume) < IMAGE_MAX_NEIGHBOUR_SAME:
        image, lut = intensity_volume(volume)
        return image, None, lut, dict(info, intensity=True)
    if volume.dtype.kind == 'f':
        volume = volume.astype(np.int64)
    if volume.min() < 0:
        raise ValueError("region labels cannot be negative")
    ids = None
    if volume.max() > 65535:
        # Ids too large to index a colour table (the full 2017 ontology has ids up to ~6e8): number the labels that
        # occur 0..n-1 for display, and keep the real ids to show and to look up in the ontology.
        ids, inverse = np.unique(volume, return_inverse=True)
        if len(ids) > 65535:
            raise ValueError("more than 65535 distinct labels: this is not an annotation volume")
        volume = inverse.reshape(volume.shape).astype(np.uint16)
    else:
        volume = volume.astype(np.uint16)
    volume = np.ascontiguousarray(volume)
    top = int(ids[-1]) if ids is not None else int(volume.max())
    ontology = None
    for candidate in sorted(path.parent.glob('*ontology*.csv')):
        try:
            ontology = read_ontology(candidate, top)
            break
        except Exception as e:
            print(f"Could not use {candidate.name} as an ontology ({e}).")
    table = ontology
    if ids is not None:
        table = (ontology.reindex(ids) if ontology is not None else pd.DataFrame(index=range(len(ids))))
        table.insert(0, 'region_id', ids)
        table.index = pd.RangeIndex(len(ids))
        for column in ('acronym_substructure', 'name_substructure', 'detail'):
            if column in table:
                table[column] = table[column].fillna('')
    return volume, table, region_colors(volume, table), info


def read_section_values(path, axis, index):
    """The region values (whole numbers, int64) of one plane of a .nrrd / .nii / .nii.gz annotation volume on the 25 um
    grid: the plane `index` along `axis` of [AP, DV, ML], whatever the file's own axis order. Raises ValueError for a
    file of the wrong size or one holding an intensity image."""
    volume, _info = oriented_volume(path)
    if volume.shape != tuple(VOLUME_SHAPE):
        raise ValueError(f"expected a volume of {N_AP} x {N_DV} x {N_ML} voxels like the atlas on screen, "
                         f"got {volume.shape}")
    plane = np.array(np.take(volume, index, axis=axis))
    if plane.dtype.kind == 'f':
        if not np.array_equal(plane, np.round(plane)):
            raise ValueError("this is an intensity image, not an annotation volume (region labels are whole numbers)")
    plane = plane.astype(np.int64)
    if plane.min() < 0:
        raise ValueError("region labels cannot be negative")
    return plane


# ---- cells from a CSV ------------------------------------------------------------------------------------------
COORDINATE_COLUMNS = {'ap_um', 'dv_um', 'ml_um', 'ccf_x_mm', 'ccf_y_mm', 'ccf_z_mm', 'section_x_mm', 'section_y_mm',
                      'section_z_mm', 'umap1', 'umap2'}


def count_lines(path):
    total = 0
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(1 << 24)
            if not chunk:
                return total
            total += chunk.count(b'\n')


def categorical_to_float(series):
    """A categorical column of numbers-as-text as float64 (NaN where missing or not a number)."""
    values = pd.to_numeric(pd.Series(series.cat.categories.astype(str)), errors='coerce').to_numpy()
    codes = series.cat.codes.to_numpy()
    return np.where(codes >= 0, values[np.clip(codes, 0, None)], np.nan)


def neuroinfo_channel_names(header_line):
    """The channel intensity columns of a NeuroInfo marker export, from its second header line
    (",x,y,z,strength,channel gad2,channel foxp1,x,y,z,name,acronym,id,hemisphere"): the fields between
    'strength' and the atlas 'x', without the "channel " prefix. Exports have as many as the image had channels."""
    fields = [f.strip() for f in header_line.rstrip('\r\n').split(',')]
    try:
        start = fields.index('strength') + 1
        end = fields.index('x', start)
    except ValueError:
        return ['DAPI', 'EGFP', 'AF594']   # the three-channel layout this reader first supported
    raw = [(f[len('channel '):].strip() if f.lower().startswith('channel ') else f) or f'channel {k + 1}'
           for k, f in enumerate(fields[start:end])]
    # Repeated names (e.g. "channel RGB" three times in exports of RGB images) are numbered: RGB 1, RGB 2, ...
    seen = {}
    names = []
    for name in raw:
        if raw.count(name) > 1:
            seen[name] = seen.get(name, 0) + 1
            name = f'{name} {seen[name]}'
        names.append(name)
    taken = {'Marker', 'strength', 'region', 'acronym', 'region_id', 'hemisphere', 'ex', 'ey', 'ez',
             'ml_um', 'dv_um', 'ap_um'}
    return [f'{n} (channel)' if n in taken else n for n in names]


def read_neuroinfo_markers(path, progress=None):
    """The marker export of MBF NeuroInfo: two header lines, then one row per detection with its experiment position,
    channel intensities (one column per image channel, named in the second header line), atlas position (um: x
    left-right, y dorsal-ventral, z anterior-posterior) and region. A region name containing a comma makes a row one
    field longer, which is repaired here rather than dropped."""
    with open(path, 'rb') as handle:
        handle.readline()
        channels = neuroinfo_channel_names(handle.readline().decode('utf-8-sig', errors='replace'))
    region_columns = ['c_name', 'c_acronym', 'c_id', 'c_hemisphere', 'c_extra']
    names = ['Marker', 'ex', 'ey', 'ez', 'strength'] + channels + ['ml_um', 'dv_um', 'ap_um'] + region_columns
    wanted = ['Marker', 'strength'] + channels + ['ml_um', 'dv_um', 'ap_um'] + region_columns
    kinds = {'Marker': 'category', 'strength': 'float32', 'ml_um': 'float32', 'dv_um': 'float32', 'ap_um': 'float32'}
    kinds.update({c: 'float32' for c in channels})
    kinds.update({c: 'category' for c in region_columns})
    from pandas.api.types import union_categoricals
    size = max(1, Path(path).stat().st_size)
    pieces = []
    import io
    dummy = b',' * (len(names) - 1) + bytes([10])   # a full-width first row, so every block is parsed full width
    with open(path, 'rb') as handle:
        handle.readline()
        handle.readline()
        while True:
            block = handle.readlines(48 << 20)
            if not block:
                break
            piece = pd.read_csv(io.BytesIO(dummy + b''.join(block)), header=None, names=names, usecols=wanted,
                                dtype=kinds, on_bad_lines='skip', low_memory=False).iloc[1:]
            pieces.append(piece.reset_index(drop=True))
            if progress:
                progress(f"{min(99, int(100 * handle.tell() / size))}%")
    if progress:
        progress("joining")
    columns = {}
    for column in pieces[0].columns:
        parts = [p[column] for p in pieces]
        if isinstance(parts[0].dtype, pd.CategoricalDtype):
            columns[column] = pd.Series(union_categoricals([p.array for p in parts]))
        else:
            columns[column] = pd.concat(parts, ignore_index=True)
    frame = pd.DataFrame(columns)
    del pieces
    long_rows = frame['c_extra'].notna().to_numpy()  # name had a comma: name, name-rest, acronym, id, hemisphere

    def repaired(normal, long_form):
        column = normal.copy()
        if long_rows.any():
            fixed = long_form[long_rows].astype(str)
            column = column.cat.add_categories([v for v in fixed.unique() if v not in column.cat.categories])
            column[long_rows] = fixed.to_numpy()
        return column

    name = repaired(frame['c_name'], frame['c_name'].astype(str) + ',' + frame['c_acronym'].astype(str)) \
        if long_rows.any() else frame['c_name']
    acronym = repaired(frame['c_acronym'], frame['c_id']) if long_rows.any() else frame['c_acronym']
    hemisphere = repaired(frame['c_hemisphere'], frame['c_extra']) if long_rows.any() else frame['c_hemisphere']
    region_id = categorical_to_float(frame['c_id'])
    if long_rows.any():
        region_id[long_rows] = categorical_to_float(frame['c_hemisphere'])[long_rows]
    columns = {'Marker': frame['Marker'], 'strength': frame['strength']}
    columns.update({c: frame[c] for c in channels})
    columns.update({'region': name, 'acronym': acronym, 'region_id': region_id, 'hemisphere': hemisphere})
    table = pd.DataFrame(columns)
    skipped = max(0, count_lines(path) - 2 - len(frame))
    return {'table': table, 'ap': frame['ap_um'].to_numpy(np.float32), 'dv': frame['dv_um'].to_numpy(np.float32),
            'ml': frame['ml_um'].to_numpy(np.float32), 'default_group': 'Marker', 'kind': 'NeuroInfo markers',
            'skipped': skipped}


def read_mbf_landmarks(path):
    """The points of an MBF Bioscience (NeuroInfo) XML file, such as the *_landmarks.xml next to an atlas template, as the
    same dict read_points_csv returns: one row per point, grouped by marker. The file's coordinates are in um in the
    image's own frame: x runs left to right, y is negative going down the image and z negative going through the sections,
    so the position in the atlas is ML = x, DV = -y and AP = -z (checked on the Gerfen 2024 Nissl landmarks, which come out
    on the anterior commissure, the endopiriform nucleus, the mammillothalamic tract and the fasciculus retroflexus,
    in left-right pairs about the midline)."""
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    rows = []
    for marker in root.iter():
        if not marker.tag.endswith('}marker') and marker.tag != 'marker':
            continue
        name = marker.get('name') or 'marker'
        number = 0
        for point in marker:
            if not (point.tag.endswith('}point') or point.tag == 'point'):
                continue
            try:
                x, y, z = (float(point.get(axis)) for axis in 'xyz')
            except (TypeError, ValueError):
                continue
            number += 1
            rows.append({'marker': name, 'landmark': number, 'x_um': x, 'y_um': y, 'z_um': z,
                         'diameter_um': float(point.get('d') or 0), 'ml_um': x, 'dv_um': -y, 'ap_um': -z})
    if not rows:
        raise ValueError("no marker points found in this XML file")
    table = pd.DataFrame(rows)
    return {'table': table, 'ap': table['ap_um'].to_numpy(dtype=np.float32), 'dv': table['dv_um'].to_numpy(dtype=np.float32),
            'ml': table['ml_um'].to_numpy(dtype=np.float32), 'default_group': 'marker', 'kind': 'MBF landmarks', 'skipped': 0}


def read_points_csv(path, progress=None):
    """The cells of a CSV as {'table', 'ap', 'dv', 'ml' (um), 'default_group', 'kind', 'skipped'}; raises ValueError
    when the layout is not one of the recognised ones (an MBF landmarks .xml is read too)."""
    if str(path).lower().endswith('.xml'):
        return read_mbf_landmarks(path)
    with open(path, 'r', encoding='utf-8-sig', errors='replace') as handle:
        first = handle.readline()
    if first.startswith('Marker,Experiment'):
        return read_neuroinfo_markers(path, progress)
    table = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    if {'ccf_x_mm', 'ccf_y_mm', 'ccf_z_mm'} <= set(table.columns):
        ap, dv, ml = (table[c].to_numpy(dtype=np.float64) * 1000 for c in ('ccf_x_mm', 'ccf_y_mm', 'ccf_z_mm'))
        keep = np.isfinite(ap) & np.isfinite(dv) & np.isfinite(ml)
        skipped = int((~keep).sum())
        table = table.loc[keep].reset_index(drop=True)
        return {'table': table, 'ap': ap[keep].astype(np.float32), 'dv': dv[keep].astype(np.float32),
                'ml': ml[keep].astype(np.float32), 'default_group': 'group_id' if 'group_id' in table else None,
                'kind': 'coordinates export', 'skipped': skipped}
    raise ValueError("not a layout this browser knows: expected the coordinates_*.csv from ABC_atlas_browser "
                     "(ccf_x_mm, ccf_y_mm, ccf_z_mm columns) or a NeuroInfo marker export")


class PointSet:
    """The cells, sorted by anterior-posterior position so one slice's cells are a contiguous block."""

    def __init__(self, loaded, path):
        self.path = Path(path)
        self.kind = loaded['kind']
        self.skipped = loaded['skipped']
        self.table = loaded['table']
        order = np.argsort(loaded['ap'], kind='stable')
        self.ap, self.dv, self.ml = loaded['ap'][order], loaded['dv'][order], loaded['ml'][order]
        self.row = order
        self.coords = (self.ap, self.dv, self.ml)   # um, indexed like the volume's axes [AP, DV, ML]
        self.orders = [None] + [np.argsort(c, kind='stable').astype(np.int32) for c in (self.dv, self.ml)]
        self.sorted_coords = [self.ap, self.dv[self.orders[1]], self.ml[self.orders[2]]]
        self.group_columns = []
        for column in self.table.columns:
            if column in COORDINATE_COLUMNS or self.table[column].dtype.kind == 'f':
                continue
            try:
                n = self.table[column].nunique()
            except TypeError:
                continue
            if 1 <= n <= MAX_GROUPS:
                self.group_columns.append(column)
        default = loaded['default_group']
        self.set_group_column(default if default in self.group_columns else
                              (self.group_columns[0] if self.group_columns else None))

    def set_group_column(self, column):
        self.group_column = column
        if column is None:
            self.group_labels = ['all cells']
            self.codes = np.zeros(len(self.ap), dtype=np.int16)
        else:
            series = self.table[column]
            if not isinstance(series.dtype, pd.CategoricalDtype):
                series = series.astype('category')
            self.group_labels = [str(c) for c in series.cat.categories]
            codes = series.cat.codes.to_numpy()[self.row]
            if (codes < 0).any():
                self.group_labels.append('(none)')
                codes = np.where(codes < 0, len(self.group_labels) - 1, codes)
            self.codes = codes.astype(np.int16)
        self.totals = np.bincount(self.codes, minlength=len(self.group_labels))

    def slab_indices(self, axis, center_um, half_um):
        """Indices of the cells within `half_um` of the plane at `center_um` along volume axis `axis`."""
        values = self.sorted_coords[axis]
        lo = int(np.searchsorted(values, center_um - half_um, side='left'))
        hi = int(np.searchsorted(values, center_um + half_um, side='right'))
        return np.arange(lo, hi) if axis == 0 else self.orders[axis][lo:hi]

    def describe(self, sorted_index):
        """[(label, value)] for one cell: its group, position and every column of its CSV row."""
        row = self.table.iloc[int(self.row[sorted_index])]
        lines = [('position (um)', f"AP {self.ap[sorted_index]:.0f}  DV {self.dv[sorted_index]:.0f}  "
                                   f"ML {self.ml[sorted_index]:.0f}")]
        for key, value in row.items():
            if isinstance(value, (float, np.floating)):
                if not np.isfinite(value):
                    continue
                value = f"{value:.4g}"
            lines.append((str(key), str(value)))
        return lines


# ---- drawing ---------------------------------------------------------------------------------------------------
def marker_offsets(shape, radius):
    """(dx, dy) pixel offsets making up a marker of the given shape and radius around its centre."""
    r = max(1, int(radius))
    span = np.arange(-r, r + 1)
    dx, dy = np.meshgrid(span, span)
    if shape == 'circle':
        mask = dx ** 2 + dy ** 2 <= r * r + 0.5
    elif shape == 'square':
        mask = (np.abs(dx) <= r) & (np.abs(dy) <= r)
    elif shape == 'plus':
        mask = ((dx == 0) | (dy == 0)) if r < 4 else ((np.abs(dx) <= 0) | (np.abs(dy) <= 0) | (np.abs(dx) == 1) & (np.abs(dy) <= r)
                                                       | (np.abs(dy) == 1) & (np.abs(dx) <= r)) & (
            (np.abs(dx) <= 1) | (np.abs(dy) <= 1))
    elif shape == 'x':
        mask = (np.abs(dx) == np.abs(dy))
    elif shape == 'triangle':
        mask = (np.abs(dx) <= (dy + r) / 2.0) & (dy >= -r) & (dy <= r)
    else:  # diamond
        mask = np.abs(dx) + np.abs(dy) <= r
    return dx[mask], dy[mask]


def plane_view(array, view, index):
    """A writable (rows, columns) view, not a copy, of one slice of a volume-shaped array."""
    if view == 'Coronal':
        return array[index]
    if view == 'Sagittal':
        return array[:, :, index].T
    return array[:, index, :]


def value_ranges(values):
    """'1-5, 9, 12-20' for a sorted array of distinct whole numbers."""
    values = np.asarray(values, dtype=np.int64)
    if not len(values):
        return 'none'
    breaks = np.flatnonzero(np.diff(values) != 1)
    starts, ends = np.concatenate(([0], breaks + 1)), np.concatenate((breaks, [len(values) - 1]))
    return ', '.join(str(values[a]) if a == b else f"{values[a]}\u2013{values[b]}" for a, b in zip(starts, ends))


def lowest_unused(values, start=1):
    """The smallest whole number >= `start` missing from a sorted array of distinct whole numbers."""
    values = np.asarray(values, dtype=np.int64)
    values = values[values >= start]
    expected = np.arange(start, start + len(values))
    mismatch = np.flatnonzero(values != expected)
    return int(expected[mismatch[0]]) if len(mismatch) else start + len(values)


def disk_offsets(radius):
    """(dx, dy) voxel offsets of a filled disk; radius 0 is the single voxel."""
    r = max(0, int(radius))
    span = np.arange(-r, r + 1)
    dx, dy = np.meshgrid(span, span)
    keep = dx * dx + dy * dy <= r * r + 0.25
    return dx[keep], dy[keep]


def desaturated(lut, saturation):
    """The colour table with its colours pulled toward grey: saturation 1.0 = as is, 0.0 = pure grey."""
    if saturation >= 1.0:
        return lut
    grey = (lut @ np.array([0.299, 0.587, 0.114]))[:, None]
    return np.clip(grey + (lut - grey) * saturation, 0, 255).astype(np.uint8)


def atlas_rgb(labels, lut, mode):
    """The (320, 456, 3) colour image of one coronal slice of region labels, in the chosen display mode."""
    if mode == 'Boundaries':
        edge = np.zeros(labels.shape, dtype=bool)
        edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
        edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
        image = np.where((labels > 0)[..., None], np.uint8(34), np.uint8(0)) * np.ones((1, 1, 3), dtype=np.uint8)
        image[edge & (labels > 0)] = (210, 210, 210)
        return image
    rgb = lut[labels]
    if mode == 'Regions (dim)':
        rgb = (rgb.astype(np.uint16) * 4 // 10).astype(np.uint8)
    return rgb


def outline_edges(labels):
    """Boolean mask of the pixels whose region differs from the pixel to their left or above: a one-pixel line along
    every region boundary."""
    edge = np.zeros(labels.shape, dtype=bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    return edge


def render_frame(labels, lut, mode, view, size, layers=(), radius=3, overlay=None, base_rgb=None, outlines=None):
    """The picture of one slice as an (H, W, 3) uint8 array. `view` = (scale, ox, oy) maps image coordinates (voxel
    edges) to canvas pixels as ox + u * scale. `layers` = [(u_um, v_um, group_codes, styles)], one
    per CSV, of the cells to stamp on it (later layers on top); `styles` = [(colour, shape)] per group."""
    width, height = size
    scale, ox, oy = view
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = BACKGROUND
    rgb = base_rgb if base_rgb is not None else atlas_rgb(labels, lut, mode)
    if overlay is not None and overlay.any():   # the painted voxels not yet applied, in red
        rgb = rgb.copy()
        rgb[overlay] = (rgb[overlay] * 0.35 + np.array([255, 0, 0]) * 0.65).astype(np.uint8)
    image_h, image_w = labels.shape
    u0, u1 = max(0.0, -ox / scale), min(float(image_w), (width - ox) / scale)
    v0, v1 = max(0.0, -oy / scale), min(float(image_h), (height - oy) / scale)
    if u1 > u0 and v1 > v0:
        dest_x, dest_y = int(round(ox + u0 * scale)), int(round(oy + v0 * scale))
        dest_w, dest_h = max(1, int(round((u1 - u0) * scale))), max(1, int(round((v1 - v0) * scale)))
        piece = np.asarray(Image.fromarray(rgb).resize((dest_w, dest_h), Image.NEAREST, box=(u0, v0, u1, v1)))
        if outlines is not None:
            # One screen pixel wide whatever the zoom: found on the picture as drawn (each screen pixel takes the region
            # of its nearest voxel, as the resize above does), not on the voxel grid.
            cols = np.clip(np.floor(u0 + (np.arange(dest_w) + 0.5) * (u1 - u0) / dest_w).astype(np.int64), 0, image_w - 1)
            rows = np.clip(np.floor(v0 + (np.arange(dest_h) + 0.5) * (v1 - v0) / dest_h).astype(np.int64), 0, image_h - 1)
            edge = outline_edges(labels[rows][:, cols])
            color, alpha = outlines
            piece = piece.copy()
            piece[edge] = (piece[edge] * (1.0 - alpha) + np.array(color) * alpha).astype(np.uint8)
        x0, y0 = max(0, dest_x), max(0, dest_y)
        x1, y1 = min(width, dest_x + dest_w), min(height, dest_y + dest_h)
        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] = piece[y0 - dest_y:y1 - dest_y, x0 - dest_x:x1 - dest_x]
    for u_um, v_um, codes, styles in layers:
        px = np.rint(ox + (u_um / VOXEL_UM + 0.5) * scale).astype(np.int64)
        py = np.rint(oy + (v_um / VOXEL_UM + 0.5) * scale).astype(np.int64)
        for group in np.unique(codes):
            selected = codes == group
            color, shape = styles[int(group) % len(styles)]
            gx, gy = px[selected], py[selected]
            for dx, dy in zip(*marker_offsets(shape, radius)):
                xx, yy = gx + dx, gy + dy
                inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
                canvas[yy[inside], xx[inside]] = color
    return canvas


# ---- settings -----------------------------------------------------------------------------------------------------
def load_settings():
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def save_settings(settings):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    except OSError:
        pass


# ---- the info window ----------------------------------------------------------------------------------------------
class InfoWindow:
    """The floating window beside the image: the slice, the cursor, the cell nearest it, and the legend."""

    def __init__(self, app):
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("CCF browser: info")
        self.window.transient(app.root)
        self.window.protocol('WM_DELETE_WINDOW', self.window.withdraw)  # closing only hides it; the Info button reopens
        tabs = ttk.Notebook(self.window)
        tabs.pack(fill='both', expand=True)
        self.text = tk.Text(tabs, width=62, height=36, font=('Consolas', 10), bg='#fbfbfb', wrap='none',
                            state='disabled', cursor='arrow')
        self.overall = tk.Text(tabs, width=62, height=36, font=('Consolas', 10), bg='#fbfbfb', wrap='word',
                               state='disabled', cursor='arrow')
        tabs.add(self.text, text="Cursor")
        tabs.add(self.overall, text="Overall")
        for widget in (self.text, self.overall):
            widget.tag_configure('head', font=('Consolas', 10, 'bold'), foreground='#1a3d6d', spacing1=6)
            widget.tag_configure('dim', foreground='#777777')
        self.slice_lines, self.cursor_lines, self.nearest_lines = [], [], []

    def set_overall(self, lines):
        """Rows of (text, tag) for the Overall tab."""
        self.overall.configure(state='normal')
        self.overall.delete('1.0', 'end')
        for line, tag in lines:
            self.overall.insert('end', line + '\n', tag or ())
        self.overall.configure(state='disabled')

    def place_beside(self):
        self.app.root.update_idletasks()
        x = self.app.root.winfo_x() + self.app.root.winfo_width() + 8
        self.window.geometry(f"+{x}+{self.app.root.winfo_y()}")
        self.app.last_root_pos = (self.app.root.winfo_x(), self.app.root.winfo_y())

    def show(self):
        if self.window.state() == 'withdrawn' and self.app.follow_var.get():
            self.place_beside()
        self.window.deiconify()
        self.window.lift()

    def set_sections(self, slice_lines=None, cursor_lines=None, nearest_lines=None):
        """Each section is a list of (text, tag) rows, or None to leave it as it is."""
        if slice_lines is not None:
            self.slice_lines = slice_lines
        if cursor_lines is not None:
            self.cursor_lines = cursor_lines
        if nearest_lines is not None:
            self.nearest_lines = nearest_lines
        self.text.configure(state='normal')
        self.text.delete('1.0', 'end')
        for title, lines in (('SLICE', self.slice_lines), ('CURSOR', self.cursor_lines),
                             ('NEAREST CELL', self.nearest_lines)):
            self.text.insert('end', title + '\n', 'head')
            for line, tag in lines:
                self.text.insert('end', line + '\n', tag or ())
        self.text.configure(state='disabled')


# ---- the application -------------------------------------------------------------------------------------------
def hex_color(color, factor=0.8):
    return '#%02x%02x%02x' % tuple(int(c * factor) for c in color)


class Slot:
    """One of the two CSV slots: its entry box, grouping, toggles, and the cells loaded into it."""

    def __init__(self, index):
        self.index = index
        self.points = None
        self.loading = False
        self.loaded_at = 0.0           # when its file was last loaded, to find the least recent
        self.visible = {}              # group label -> BooleanVar (shown on the slices)
        self.visible_mask = None       # bool per group code
        shapes = GROUP_SHAPES[3 * index:] + GROUP_SHAPES[:3 * index]   # the two slots use different shape orders
        self.styles = [(GROUP_COLORS[(i + 6 * index) % len(GROUP_COLORS)], shapes[i % len(shapes)])
                       for i in range(64)]
        self.path_var = tk.StringVar()
        self.group_var = tk.StringVar()


class CCFBrowser:
    SLOTS = 2

    @property
    def on_ccf_grid(self):
        """Whether the atlas on screen is on the CCF's grid (25 um, 528 x 320 x 456): bregma and the comparison with the
        standard CCF only make sense there."""
        return (self.volume is None or (tuple(self.volume.shape) == CCF_SHAPE
                                        and abs(self.atlas_info['voxel_um'] - CCF_VOXEL_UM) < 1e-6))

    @property
    def slice(self):
        return self.slices[self.view_name]

    @slice.setter
    def slice(self, value):
        self.slices[self.view_name] = int(value)

    @property
    def axis(self):
        return VIEW_AXIS[self.view_name]

    @property
    def n_slices(self):
        return VOLUME_SHAPE[self.axis]

    @property
    def slice_um(self):
        return self.slice * VOXEL_UM

    def __init__(self):
        self.dnd, self.dnd_files = import_drag_and_drop()
        self.root = self.dnd.Tk() if self.dnd is not None else tk.Tk()
        self.root.title("CCF browser")
        self.root.geometry("1100x800+60+40")
        self.settings = load_settings()
        self.volume = self.region_table = self.lut = None
        self.atlas_info = {'voxel_um': CCF_VOXEL_UM, 'legacy': True}   # geometry and orientation of the atlas on screen
        self.reference = None          # the standard CCF (volume, table, lut) while another atlas is on screen
        self.reference_loading = False
        self.diff_cache = None         # (key, alt_lut, ref_lut, basis, totals or None)
        self.ref_lut_view = None       # the standard atlas's colours at the chosen saturation
        self.panes_shown = 1           # 2 in the Differences view: standard atlas left, the alternate right
        self.tip_job = None            # pending after() that shows the region tool tip
        self.tip_active = False        # once a tool tip has appeared it stays up and follows the pointer
        self.slots = [Slot(i) for i in range(self.SLOTS)]
        self.photo = None
        self.view_name = self.settings.get('view', VIEWS[0]) if self.settings.get('view') in VIEWS else VIEWS[0]
        saved_slices = self.settings.get('slices', {})
        self.slices = {name: int(saved_slices.get(name, VOLUME_SHAPE[VIEW_AXIS[name]] // 2)) for name in VIEWS}
        self.lut_view = None           # the region colours at the chosen saturation
        self.follow_var = tk.BooleanVar(value=bool(self.settings.get('follow_info', True)))
        self.atlas_path = self.settings.get('atlas_path', '')   # '' = the CCF from the ABC atlas cache
        self.atlas_flips = {k: tuple(bool(x) for x in v) for k, v in self.settings.get('atlas_flips', {}).items()}
        self.loaded_flips = (False, False, False)   # orientation corrections the atlas on screen was loaded with
        self.loading_flips = (False, False, False)
        self.alt_atlas_path = self.settings.get('alt_atlas_path', '') or self.atlas_path   # remembered for switching back
        self.loaded_atlas_path = ''    # the atlas now on screen
        self.settings_status = tk.StringVar()
        self.atlas_loaded_once = False
        self.last_root_pos = None
        self.zoom_goal = None          # scale the smooth zoom is heading for, None when not zooming
        self.zoom_anchor = None        # (canvas x, canvas y, image u, image v) kept fixed during the zoom
        self.wheel_zoom_var = tk.BooleanVar(value=bool(self.settings.get('wheel_zoom', True)))
        try:
            self.bregma = [float(v) for v in self.settings['bregma_mm']][:3]
            assert len(self.bregma) == 3
        except Exception:
            self.bregma = list(BREGMA_DEFAULT_MM)
        self.settings_window = None
        self.label_ids = None          # real ids of a remapped atlas (volume holds 0..n-1), else None
        self.overlay = None            # bool volume of the red painted voxels, made on first use
        self.overlay_count = 0
        self.edit_count = 0            # voxels changed by Apply since the atlas was loaded
        self.edit_version = 0          # counts every Apply and undone Apply, to tell whether the volume was saved
        self.saved_version = 0
        self.undo_stack = []           # ('stroke', flat voxel indices, drawn?) / ('apply', flat indices, old values)
        self.tool = 'none'             # 'draw', 'erase' or 'none' (drag pans)
        self.stroke = None             # flat voxel indices changed by the stroke in progress
        self.last_paint = None
        self.value_cache = None        # voxels per label, until the volume changes
        self.paint_window = None
        self.paint_buttons = {}
        self.paint_value_var = tk.StringVar(value=self.settings.get('paint_value', ''))
        self.paint_warning = tk.StringVar()
        self.filter_var = tk.StringVar()   # Paint panel's "Filter by": blank = off (deliberately not remembered)
        self.paint_status = tk.StringVar()
        # The paint dialog's brush radius counts from 1 (1 = just the voxel under the pointer); an older settings file
        # stored it from 0 under 'brush'.
        self.brush_var = tk.IntVar(value=int(self.settings['brush_radius']) if 'brush_radius' in self.settings
                                   else int(self.settings.get('brush', 5)) + 1)
        self.view = [1.0, 0.0, 0.0]   # scale, ox, oy
        self.fit = True
        self.drag = None
        self.mouse = None              # last cursor position over the canvas
        self.render_pending = False
        self.work = queue.Queue()
        self.busy_label, self.progress_text = '', ''
        self.busy_overlay = ''         # "Loading CSV…" / "Loading Atlas…", drawn over the middle of the picture while it lasts
        self.autoload = []             # (slot, path) from the last session, loaded one after another
        self.build_ui()
        self.info = InfoWindow(self)
        self.refresh_info_slice()
        self.root.after(150, self.info.place_beside)
        self.root.bind('<Configure>', self.on_root_configure)
        self.root.after(100, self.poll_work)
        self.root.after(200, self.start_atlas)
        self.root.protocol('WM_DELETE_WINDOW', self.close)

    # ---- recent files of the CSV boxes
    RECENT_CSV_MAX = 12

    def recent_files(self, slot):
        """The files this CSV box loaded recently (newest first) that still exist."""
        return [path for path in self.recent_csv.get(str(slot.index), []) if Path(path).is_file()]

    def remember_recent(self, slot, path):
        """Put `path` first in this box's list (once), keep at most RECENT_CSV_MAX, and refresh the box's drop-down."""
        key = os.path.normcase(os.path.normpath(path))
        kept = [item for item in self.recent_csv.get(str(slot.index), [])
                if os.path.normcase(os.path.normpath(item)) != key]
        self.recent_csv[str(slot.index)] = ([path] + kept)[:self.RECENT_CSV_MAX]
        widgets = self.slot_widgets[slot.index] if slot.index < len(self.slot_widgets) else None
        if widgets is not None:
            widgets['entry'].configure(values=self.recent_files(slot))

    # ---- layout
    def build_ui(self):
        root = self.root
        try:
            self.recent_csv = {str(k): [str(p) for p in v] for k, v in self.settings.get('recent_csv', {}).items()}
        except (AttributeError, TypeError):
            self.recent_csv = {}
        self.slot_widgets = []
        for slot in self.slots:
            saved = self.settings.get(f'csv{slot.index}', self.settings.get('csv', '') if slot.index == 0 else '')
            slot.path_var.set(saved)
            block = ttk.Frame(root, padding=(6, 4 if slot.index == 0 else 0, 6, 0))
            block.pack(fill='x')
            row = ttk.Frame(block)
            row.pack(fill='x')
            ttk.Label(row, text=f"CSV {slot.index + 1}:").pack(side='left')
            # An editable drop-down: type, paste or drop a path, or pick one of this box's recently loaded files.
            entry = ttk.Combobox(row, textvariable=slot.path_var, values=self.recent_files(slot))
            entry.pack(side='left', fill='x', expand=True, padx=4)
            entry.bind('<<ComboboxSelected>>', lambda e, s=slot: self.load_slot(s, chosen=True))
            entry.bind('<Return>', lambda e, s=slot: self.load_slot(s))
            entry.bind('<<Paste>>', lambda e, s=slot: self.root.after(60, lambda: self.load_slot(s)))
            ttk.Button(row, text="Choose file…", command=lambda s=slot: self.choose_file(s)).pack(side='left')
            ttk.Button(row, text="Clear", width=6, command=lambda s=slot: self.clear_slot(s)).pack(side='left',
                                                                                                    padx=(4, 8))
            ttk.Label(row, text="Group by").pack(side='left')
            combo = ttk.Combobox(row, textvariable=slot.group_var, state='readonly', width=14, values=())
            combo.pack(side='left', padx=(3, 0))
            combo.bind('<<ComboboxSelected>>', lambda e, s=slot: self.change_group(s))
            toggles = ttk.Frame(block)
            toggles.pack(fill='x')
            self.slot_widgets.append({'entry': entry, 'combo': combo, 'toggles': toggles})
            slot.start_text = saved

        views = ttk.Frame(root, padding=(6, 4, 6, 0))
        views.pack(fill='x')
        ttk.Label(views, text="View:").pack(side='left')
        self.view_var = tk.StringVar(value=self.view_name)
        for name in VIEWS:
            ttk.Radiobutton(views, text=name, value=name, variable=self.view_var,
                            command=self.change_view).pack(side='left', padx=(6, 0))
        ttk.Button(views, text="\u2699 Settings", command=self.open_settings).pack(side='right')
        ttk.Label(views, text="Atlas colour saturation").pack(side='left', padx=(24, 4))
        self.sat_var = tk.IntVar(value=int(self.settings.get('saturation', 100)))
        self.sat_label = ttk.Label(views, text="", width=5)
        ttk.Scale(views, from_=0, to=100, orient='horizontal', length=160, variable=self.sat_var,
                  command=lambda v: self.change_saturation()).pack(side='left')
        self.sat_label.pack(side='left', padx=(4, 0))

        controls = ttk.Frame(root, padding=(6, 4, 6, 2))
        controls.pack(fill='x')
        ttk.Label(controls, text="Slab ± slices").pack(side='left')
        self.slab_var = tk.IntVar(value=int(self.settings.get('slab', 2)))
        ttk.Spinbox(controls, from_=0, to=40, width=4, textvariable=self.slab_var,
                    command=self.refresh_all).pack(side='left', padx=(3, 10))
        ttk.Label(controls, text="Dot radius").pack(side='left')
        self.dot_var = tk.IntVar(value=int(self.settings.get('dot', 3)))
        ttk.Spinbox(controls, from_=1, to=12, width=4, textvariable=self.dot_var,
                    command=self.refresh_all).pack(side='left', padx=(3, 10))
        ttk.Label(controls, text="Atlas").pack(side='left')
        self.mode_var = tk.StringVar(value=self.settings.get('mode', ATLAS_MODES[0]))
        mode_box = self.mode_box = ttk.Combobox(controls, textvariable=self.mode_var, state='readonly', width=28,
                                               values=ATLAS_MODES)
        mode_box.pack(side='left', padx=3)
        mode_box.bind('<<ComboboxSelected>>', lambda e: self.mode_changed())
        self.outline_var = tk.BooleanVar(value=bool(self.settings.get('outlines', True)))
        ttk.Checkbutton(controls, text="Region outlines", variable=self.outline_var,
                        command=self.request_render).pack(side='left', padx=(8, 0))
        ttk.Button(controls, text="Info window", command=lambda: self.info.show()).pack(side='left', padx=(10, 0))
        ttk.Button(controls, text="Paint panel", command=self.open_paint).pack(side='left', padx=(4, 0))
        ttk.Button(controls, text="Copy section from", command=self.restore_section).pack(side='left', padx=(4, 0))
        ttk.Button(controls, text="Generate meshes", command=self.open_mesh_dialog).pack(side='left', padx=(4, 0))
        for variable in (self.slab_var, self.dot_var):
            variable.trace_add('write', lambda *a: self.refresh_all())
        self.message = ttk.Label(controls, text="", foreground='#7a3b00')
        self.message.pack(side='right')

        bottom = ttk.Frame(root, padding=(6, 2, 6, 6))
        bottom.pack(side='bottom', fill='x')
        self.slice_label = ttk.Label(bottom, text="", width=52)
        self.slice_label.pack(side='left')
        self.scrollbar = tk.Scrollbar(bottom, orient='horizontal', command=self.on_scrollbar)
        self.scrollbar.pack(side='left', fill='x', expand=True)

        self.canvas = tk.Canvas(root, bg='#181818', highlightthickness=0, cursor='crosshair')
        self.canvas.pack(fill='both', expand=True)
        self.image_item = self.canvas.create_image(0, 0, anchor='nw')
        self.canvas.bind('<Configure>', lambda e: self.on_canvas_resize())
        self.canvas.bind('<Motion>', self.on_motion)
        self.canvas.bind('<Leave>', lambda e: self.on_leave())
        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        self.canvas.bind('<MouseWheel>', self.on_wheel)
        self.canvas.bind('<Button-3>', self.on_right_click)
        root.bind('<Left>', lambda e: self.step(-1))
        root.bind('<Right>', lambda e: self.step(1))
        root.bind('<Prior>', lambda e: self.step(-10))
        root.bind('<Next>', lambda e: self.step(10))
        root.bind('<Home>', lambda e: self.go_to(0))
        root.bind('<End>', lambda e: self.go_to(self.n_slices - 1))
        root.bind('<plus>', lambda e: self.zoom_at(1.25))
        root.bind('<equal>', lambda e: self.zoom_at(1.25))
        root.bind('<minus>', lambda e: self.zoom_at(0.8))
        root.bind('0', lambda e: self.fit_to_window())
        if self.dnd is not None:
            for slot, widgets in zip(self.slots, self.slot_widgets):   # onto an entry box: that slot
                widgets['entry'].drop_target_register(self.dnd_files)
                widgets['entry'].dnd_bind('<<Drop>>', lambda e, s=slot: self.on_drop(e, s))
            for widget in (self.canvas, root):   # anywhere else: the next empty slot, else the least recent
                widget.drop_target_register(self.dnd_files)
                widget.dnd_bind('<<Drop>>', lambda e: self.on_drop(e, None))

    # ---- painting
    def paint_filter_label(self):
        """None when the Paint panel's filter is off (blank box); otherwise the label that painting is limited to, or -1
        when the box holds something that matches no voxel (not a whole number, or a value the atlas does not have), so
        that nothing at all gets painted rather than everything."""
        text = self.filter_var.get().strip()
        if not text:
            return None
        if not text.isdigit():
            return -1
        label = self.stored_label(int(text))
        return -1 if label is None else label

    def paint_to(self, x, y):
        """Draw or erase the brush disk (and the stretch since the last mouse position) on the overlay."""
        scale, ox, oy = self.view
        u, v = int(np.floor((self.pane_x(x) - ox) / scale)), int(np.floor((y - oy) / scale))
        previous, self.last_paint = self.last_paint, (u, v)
        if previous is None:
            centers = np.array([[u, v]])
        else:
            steps = max(1, abs(u - previous[0]), abs(v - previous[1]))
            fractions = np.linspace(0, 1, steps + 1)[:, None]
            centers = np.rint(np.array(previous) + (np.array([u, v]) - np.array(previous)) * fractions).astype(np.int64)
        dx, dy = disk_offsets(self.brush_radius())
        us = (centers[:, 0, None] + dx[None, :]).ravel()
        vs = (centers[:, 1, None] + dy[None, :]).ravel()
        image_w, image_h = VIEW_IMAGE_SIZE[self.view_name]
        inside = (us >= 0) & (us < image_w) & (vs >= 0) & (vs < image_h)
        us, vs = us[inside], vs[inside]
        if not len(us):
            return
        plane = plane_view(self.overlay, self.view_name, self.slice)
        target = self.tool == 'draw'
        changed = plane[vs, us] != target
        wanted = self.paint_filter_label() if target else None   # erasing is never filtered
        if wanted is not None:   # "Filter by": only voxels that currently hold that region's value are drawn on
            changed &= plane_view(self.volume, self.view_name, self.slice)[vs, us] == wanted
        us, vs = us[changed], vs[changed]
        if not len(us):
            return
        plane[vs, us] = target
        across, down = VIEW_CELL_AXES[self.view_name]
        voxel = [None, None, None]
        voxel[self.axis], voxel[across], voxel[down] = np.full(len(us), self.slice), us, vs
        flat = np.unique(np.ravel_multi_index(voxel, VOLUME_SHAPE))
        self.stroke.append(flat)
        self.overlay_count += len(flat) if target else -len(flat)
        self.request_render()

    def on_right_click(self, event):
        """With the paint panel open: a menu to add the region under the cursor to the drawing, or remove it."""
        if self.volume is None or self.paint_window is None or not self.paint_window.winfo_exists():
            return
        if not self.in_paint_pane(event.x):
            return   # the standard atlas at the left is never edited
        scale, ox, oy = self.view
        u_idx, v_idx = int(np.floor((self.pane_x(event.x) - ox) / scale)), int(np.floor((event.y - oy) / scale))
        image_w, image_h = VIEW_IMAGE_SIZE[self.view_name]
        if not (0 <= u_idx < image_w and 0 <= v_idx < image_h):
            return
        across, down = VIEW_CELL_AXES[self.view_name]
        voxel = [0, 0, 0]
        voxel[self.axis], voxel[across], voxel[down] = self.slice, u_idx, v_idx
        label = int(self.volume[voxel[0], voxel[1], voxel[2]])
        value = int(self.label_ids[label]) if self.label_ids is not None else label
        name = self.region_name(value)
        shown = f"{name} ({value})" if name else str(value)
        menu = tk.Menu(self.canvas, tearoff=0)
        menu.add_command(label=f"Add region {shown} to drawing", command=lambda: self.region_to_drawing(label, True))
        menu.add_command(label=f"Remove region {shown} from drawing",
                         command=lambda: self.region_to_drawing(label, False))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def region_to_drawing(self, label, add):
        """Add all of a region's voxels to the red drawing, or remove them from it (one undo step)."""
        if self.overlay is None:
            self.overlay = np.zeros(VOLUME_SHAPE, dtype=bool)
        region = self.volume == label
        wanted = self.paint_filter_label() if add else None
        if wanted is not None and wanted != label:   # the Paint panel's filter limits what may be drawn
            region = np.zeros_like(region)
        changed = np.flatnonzero((region & ~self.overlay if add else region & self.overlay).reshape(-1))
        if not len(changed):
            self.paint_status.set("Nothing to " + ("add: the region is already all in the drawing."
                                                   if add else "remove: none of it is in the drawing."))
            return
        self.overlay.reshape(-1)[changed] = add
        self.undo_stack.append(('stroke', changed, add))
        self.overlay_count += len(changed) if add else -len(changed)
        self.refresh_all()
        self.update_paint_status()

    def brush_radius(self):
        """How many voxels the brush reaches beyond the one under the pointer: the dialog's one-based radius minus 1."""
        try:
            return max(0, int(self.brush_var.get()) - 1)
        except (tk.TclError, ValueError):   # the box is empty while it is being retyped
            return 0

    def draw_brush(self):
        """The brush outline following the cursor while a paint tool is on. In the Differences view it is shown on both
        maps at the same place: solid on the alternate atlas at the right (where painting happens), dashed on the
        standard atlas at the left, to show which voxels there the painting has to match."""
        self.canvas.delete('brush')
        if self.tool == 'none' or self.mouse is None:
            return
        paintable = self.in_paint_pane(self.mouse[0])
        if not self.busy_label:
            self.canvas.configure(cursor='pencil' if paintable else 'crosshair')
        radius = (self.brush_radius() + 0.5) * self.view[0]
        x, y = self.mouse
        if self.pane_count() == 2:
            x_in_pane = self.pane_x(x)
            right_start = self.pane_width() + PANE_GAP
            for origin, solid in ((0, False), (right_start, True)):
                cx = origin + x_in_pane
                self.canvas.create_oval(cx - radius, y - radius, cx + radius, y + radius, outline='#ff4040',
                                        dash=() if solid else (4, 3), tags='brush')
        else:
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline='#ff4040', tags='brush')

    def set_tool(self, tool):
        """Switch the paint tool on (draw / erase), or off (the same tool again, or 'none'): dragging pans when off."""
        self.tool = 'none' if tool == self.tool else tool
        self.deactivate_tip()
        self.update_cursors()
        for name, button in self.paint_buttons.items():
            if button.winfo_exists():
                button.configure(relief='sunken' if name == self.tool else 'raised')
        self.canvas.delete('brush')

    def value_counts(self):
        """Voxels per stored label (cached until the volume changes)."""
        if self.value_cache is None:
            counts = np.zeros(len(self.lut), dtype=np.int64)
            for start in range(0, N_AP, 16):   # in slabs: bincount would copy the whole volume as 64-bit
                counts += np.bincount(self.volume[start:start + 16].ravel(), minlength=len(self.lut))[:len(self.lut)]
            self.value_cache = counts
        return self.value_cache

    def used_values(self):
        """Sorted array of the region values (real ids) that occur in the volume."""
        present = np.flatnonzero(self.value_counts() > 0)
        if self.label_ids is not None:
            return np.sort(self.label_ids[present[present < len(self.label_ids)]])
        return present

    def stored_label(self, value):
        """The label stored in the volume for a region value, or None if the value is not in the atlas's list."""
        if self.label_ids is None:
            return value if value < len(self.lut) else None
        found = np.flatnonzero(self.label_ids == value)
        return int(found[0]) if len(found) else None

    def ensure_label(self, value):
        """The label to store for a region value, adding it (with a random colour and a table row) if it is new."""
        label = self.stored_label(value)
        if label is not None and (self.region_table is None or label in self.region_table.index):
            return label
        if label is None:
            if self.label_ids is None:
                if value > 65535:
                    raise ValueError("this atlas stores 16-bit labels: the value must be 65535 or less")
                label, new_size = value, value + 1
            else:
                label, new_size = len(self.label_ids), len(self.label_ids) + 1
                if label > 65535:
                    raise ValueError("no room for another label")
                self.label_ids = np.append(self.label_ids, value)
            rng = np.random.default_rng(value)
            extra = rng.integers(60, 256, size=(max(0, new_size - len(self.lut)), 3)).astype(np.uint8)
            self.lut = np.vstack([self.lut, extra])
            if self.value_cache is not None:
                self.value_cache = np.concatenate([self.value_cache, np.zeros(len(self.lut) - len(self.value_cache),
                                                                              dtype=np.int64)])
        if self.region_table is not None:
            row = {column: np.nan for column in self.region_table.columns}
            row.update({'acronym_substructure': '(painted)', 'name_substructure': f"painted region {value}",
                        'detail': 'painted in the CCF browser'})
            if 'region_id' in row:
                row['region_id'] = value
            self.region_table.loc[label] = pd.Series(row)
        return label

    def refresh_overall(self):
        """The Overall tab of the info window: the atlas, the pixel values in use, the edits, the loaded CSVs."""
        if self.volume is None:
            return
        used = self.used_values()
        lines = [("ATLAS", 'head'),
                 (Path(self.atlas_path).name if self.atlas_path else "Allen CCF 2020 (ABC atlas cache)", None),
                 (f"{N_AP} x {N_DV} x {N_ML} voxels of {VOXEL_UM:g} um", None),
                 ("PIXEL VALUES IN USE", 'head')]
        if len(used):
            lines += [(f"{len(used):,} distinct values, from {int(used.min())} to {int(used.max())}:", None),
                      (value_ranges(used), None), (f"Lowest unused value: {lowest_unused(used)}", None)]
        lines += [("EDITS", 'head'), (f"Red overlay (not applied): {self.overlay_count:,} voxels", None),
                  (f"Applied to the volume: {self.edit_count:,} voxel changes ({len(self.undo_stack)} undo steps)",
                   None), ("CSV FILES", 'head')]
        loaded = [slot for slot in self.slots if slot.points is not None]
        for slot in loaded:
            lines.append((f"CSV {slot.index + 1}: {slot.points.path.name}, {len(slot.points.ap):,} cells", None))
        if not loaded:
            lines.append(("none loaded", 'dim'))
        self.info.set_overall(lines)

    def region_name(self, value):
        label = self.stored_label(value)
        if label is None or self.region_table is None or label not in self.region_table.index:
            return ''
        return str(self.region_table.loc[label].get('acronym_substructure') or '')

    def update_paint_status(self):
        if self.paint_window is None or not self.paint_window.winfo_exists():
            return
        text = self.paint_value_var.get().strip()
        warning = ''
        if text:
            if not text.isdigit():
                warning = "Enter a whole number, 0 or more."
            else:
                value = int(text)
                label = self.stored_label(value)
                count = int(self.value_counts()[label]) if label is not None and label < len(self.value_counts()) else 0
                if count:
                    name = self.region_name(value)
                    warning = (f"Value {value} is already in use ({name + ', ' if name else ''}{count:,} voxels): "
                               f"applying will merge the overlay into it.")
        filter_text = self.filter_var.get().strip()
        if filter_text:
            if not filter_text.isdigit():
                warning += ("\n" if warning else "") + "Filter by: enter a whole number, or leave it blank. Nothing can be drawn."
            else:
                filter_label = self.stored_label(int(filter_text))
                counts = self.value_counts()
                in_use = filter_label is not None and filter_label < len(counts) and counts[filter_label] > 0
                name = self.region_name(int(filter_text))
                warning += ("\n" if warning else "") + (
                    f"Filter on: only voxels with value {filter_text}{' (' + name + ')' if name else ''} can be painted."
                    if in_use else f"Filter by {filter_text}: no voxel has that value, so nothing can be drawn.")
        self.paint_warning.set(warning)
        self.paint_status.set(f"Overlay: {self.overlay_count:,} voxels   Undo steps: {len(self.undo_stack)}")

    def open_paint(self):
        if self.volume is None:
            return
        if self.paint_window is not None and self.paint_window.winfo_exists():
            self.paint_window.lift()
            return
        window = self.paint_window = tk.Toplevel(self.root)
        window.title("CCF browser: paint")
        window.transient(self.root)
        window.protocol('WM_DELETE_WINDOW', self.close_paint)
        body = ttk.Frame(window, padding=12)
        body.pack(fill='both', expand=True)
        row = ttk.Frame(body)
        row.pack(fill='x')
        self.paint_buttons = {}
        for name, label in (('draw', "Draw"), ('erase', "Erase")):
            button = tk.Button(row, text=label, width=9, command=lambda n=name: self.set_tool(n))
            button.pack(side='left', padx=(0, 6))
            self.paint_buttons[name] = button
        ttk.Button(row, text="Undo", command=self.undo).pack(side='left')
        for name, button in self.paint_buttons.items():
            button.configure(relief='sunken' if name == self.tool else 'raised')
        ttk.Label(body, foreground='#555555', text="Draw or Erase on, drag on the image to paint in red; "
                                                   "press the tool again to pan.", wraplength=300).pack(
            anchor='w', pady=(6, 8))
        grid = ttk.Frame(body)
        grid.pack(fill='x')
        ttk.Label(grid, text="Brush radius (voxels)").grid(row=0, column=0, sticky='w')
        ttk.Spinbox(grid, from_=1, to=101, width=6, textvariable=self.brush_var).grid(row=0, column=1, padx=(8, 0),
                                                                                    sticky='w')
        ttk.Label(grid, text="Pixel value").grid(row=1, column=0, sticky='w', pady=(6, 0))
        entry = ttk.Entry(grid, textvariable=self.paint_value_var, width=14)
        entry.grid(row=1, column=1, padx=(8, 0), pady=(6, 0), sticky='w')
        if not self.paint_value_var.get().strip():   # first time: suggest the lowest unused value, else keep the last
            self.paint_value_var.set(str(lowest_unused(self.used_values())))
        self.paint_value_var.trace_add('write', lambda *a: self.update_paint_status())
        ttk.Label(grid, text="Filter by (pixel value)").grid(row=2, column=0, sticky='w', pady=(6, 0))
        ttk.Entry(grid, textvariable=self.filter_var, width=14).grid(row=2, column=1, padx=(8, 0), pady=(6, 0),
                                                                     sticky='w')
        ttk.Label(grid, foreground='#555555', text="blank = off; a number limits painting to voxels that "
                                                   "currently have that value").grid(row=3, column=0, columnspan=2,
                                                                                    sticky='w', pady=(2, 0))
        self.filter_var.trace_add('write', lambda *a: self.update_paint_status())
        ttk.Label(body, textvariable=self.paint_warning, foreground='#b00020', wraplength=320,
                  justify='left').pack(anchor='w', pady=(6, 0))
        ttk.Label(body, textvariable=self.paint_status, foreground='#555555').pack(anchor='w', pady=(6, 0))
        buttons = ttk.Frame(body)
        buttons.pack(fill='x', pady=(10, 0))
        ttk.Button(buttons, text="Close", command=self.close_paint).pack(side='right')
        ttk.Button(buttons, text="Apply", command=self.apply_paint).pack(side='right', padx=(0, 6))
        ttk.Button(buttons, text="Save volume as…", command=self.save_volume).pack(side='left')
        info = self.info.window
        info.update_idletasks()
        below = info.winfo_y() + info.winfo_height() + 8
        if below + 280 < self.root.winfo_screenheight():
            window.geometry(f"+{info.winfo_x()}+{below}")
        else:
            window.geometry(f"+{info.winfo_x() + info.winfo_width() + 8}+{info.winfo_y()}")
        self.update_paint_status()

    def save_volume(self):
        """Write the (edited) volume to a file; the format defaults to that of the atlas file it was loaded from."""
        if self.volume is None or self.busy_label:
            return
        parent = self.paint_window if self.paint_window is not None and self.paint_window.winfo_exists() else self.root
        if self.overlay_count and not messagebox.askyesno(
                "Unapplied overlay", f"{self.overlay_count:,} red voxels have not been applied and will not be saved. "
                "Save anyway?", parent=parent):
            return
        source = Path(self.atlas_path) if self.atlas_path else None
        default = volume_format(source) if source else '.nii.gz'
        default = default or '.nii.gz'
        generic = not self.atlas_info.get('legacy', True)   # not on the CCF grid: saved in its own format and layout
        order = [default] + ([] if generic else [ext for ext in SAVE_FORMATS if ext != default])
        stem = source.name[:-len(default)] if source and source.name.lower().endswith(default) else 'annotation_25'
        kind = tk.StringVar()
        chosen = filedialog.asksaveasfilename(
            title="Save the edited volume", parent=parent, initialdir=str((source.parent if source else SCRIPT_DIR.parent)),
            initialfile=f"{stem}_edited{default}", defaultextension=default, typevariable=kind,
            filetypes=[(SAVE_FORMATS[ext], f"*{ext}") for ext in order])
        if not chosen:
            return
        if volume_format(chosen) is None:   # no recognised extension typed: use the file type chosen in the dialog
            ext = next((e for e, label in SAVE_FORMATS.items() if label == kind.get()), default)
            chosen += ext
        path = Path(chosen)
        if generic and volume_format(path) != self.atlas_info['format']:
            messagebox.showerror("Save volume", f"This atlas can only be saved in its own format "
                                 f"({self.atlas_info['format']}), with its original orientation.", parent=parent)
            return
        ids = self.label_ids
        if ids is not None:
            values = ids.astype(np.uint32 if ids.max() > 65535 else np.uint16)[self.volume]
        else:
            values = self.volume.copy()
        if values.dtype != np.uint32 and values.max() > 65535:
            values = values.astype(np.uint32)

        version = self.edit_version

        def done(result, error):
            if error is not None:
                messagebox.showerror("Save volume", f"Could not save the volume:\n{error}", parent=parent)
                self.paint_status.set("Save failed.")
                return
            self.saved_version = version
            self.paint_status.set(f"Saved {path.name}")
            self.message.configure(text=f"Saved {path}")
        self.run_in_thread(f"Saving {path.name}…", lambda: write_volume(path, values, self.set_progress, self.atlas_info if generic else None), done)

    def close_paint(self):
        self.set_tool('none')
        if self.paint_window is not None:
            self.paint_window.destroy()
        self.paint_window = None

    def apply_paint(self):
        """Give the red overlay's voxels the typed pixel value."""
        if not self.overlay_count:
            self.paint_status.set("Nothing is drawn: use Draw first.")
            return
        text = self.paint_value_var.get().strip()
        if not text.isdigit():
            self.paint_status.set("Enter a whole number, 0 or more, as the pixel value.")
            return
        value = int(text)
        if value in set(self.used_values().tolist()) and not messagebox.askyesno(
                "Value already in use", f"Pixel value {value} is already in use"
                + (f" ({self.region_name(value)})" if self.region_name(value) else "")
                + ". The overlay will be merged into it.\n\nApply anyway?", parent=self.paint_window):
            return
        try:
            label = self.ensure_label(value)
        except ValueError as e:
            self.paint_status.set(str(e))
            return
        flat = np.flatnonzero(self.overlay.reshape(-1))
        wanted = self.paint_filter_label()
        left_over = 0
        if wanted is not None:   # filter set after drawing: only matching voxels are painted, the rest stay red
            matching = self.volume.reshape(-1)[flat] == wanted
            left_over = int((~matching).sum())
            flat = flat[matching]
            if not len(flat):
                self.paint_status.set("Nothing was applied: none of the red voxels matches the filter.")
                return
        old = self.volume.reshape(-1)[flat].copy()
        self.volume.reshape(-1)[flat] = label
        self.overlay.reshape(-1)[flat] = False
        self.undo_stack.append(('apply', flat, old))
        self.overlay_count = left_over
        self.edit_count += len(flat)
        self.edit_version += 1
        self.value_cache = None
        self.apply_saturation()
        self.refresh_all()
        self.update_paint_status()
        self.paint_status.set(f"Applied value {value} to {len(flat):,} voxels."
                              + (f" {left_over:,} red voxels outside the filter were left." if left_over else ""))

    def restore_section(self):
        """Reset the section on screen, and nothing else, to the same section of another annotation volume (.nrrd, .nii
        or .nii.gz on the same grid), whose pixel values are taken as region values like those typed in the Paint
        panel. One undo step; the red drawing is left alone."""
        if self.volume is None:
            return
        if self.busy_label:
            self.message.configure(text="Still busy; try again when it finishes.")
            return
        start = self.settings.get('restore_dir')
        chosen = filedialog.askopenfilename(
            title=f"Copy {self.view_name} section {self.slice + 1} from which volume?", parent=self.root,
            initialdir=start if start and Path(start).is_dir() else None,
            filetypes=[("Annotation volumes", "*.nrrd *.nii *.nii.gz"), ("All files", "*.*")])
        if not chosen:
            return
        self.settings['restore_dir'] = str(Path(chosen).parent)
        path, axis, index, view = Path(chosen), self.axis, self.slice, self.view_name

        def done(result, error):
            if error is not None:
                messagebox.showerror("Copy section from", f"Could not copy the section:\n{error}", parent=self.root)
                return
            self.apply_restored_section(result, axis, index, view, path)
        self.run_in_thread(f"Reading {path.name}…", lambda: read_section_values(path, axis, index), done)

    def apply_restored_section(self, values, axis, index, view, path):
        unique, inverse = np.unique(values, return_inverse=True)
        try:
            labels = np.array([self.ensure_label(int(v)) for v in unique], dtype=np.uint16)
        except ValueError as e:
            messagebox.showerror("Copy section from", str(e), parent=self.root)
            return
        new_plane = labels[inverse.reshape(values.shape)]
        selector = [slice(None)] * 3
        selector[axis] = index
        current = self.volume[tuple(selector)]
        rows, cols = np.nonzero(new_plane != current)
        if not len(rows):
            self.message.configure(text=f"The {view} section {index + 1} already matches {path.name}.")
            return
        coords, free = [None] * 3, [a for a in range(3) if a != axis]
        coords[axis] = np.full(len(rows), index)
        coords[free[0]], coords[free[1]] = rows, cols
        flat = np.ravel_multi_index(tuple(coords), VOLUME_SHAPE)
        old = self.volume.reshape(-1)[flat].copy()
        self.volume.reshape(-1)[flat] = new_plane[rows, cols]
        self.undo_stack.append(('restore', flat, old))
        self.edit_count += len(flat)
        self.edit_version += 1
        self.value_cache = None
        self.apply_saturation()
        self.refresh_all()
        self.update_paint_status()
        self.message.configure(text=f"Copied {len(flat):,} voxels of the {view} section {index + 1} from "
                                    f"{path.name} (Undo is in the Paint panel).")

    # ---- region meshes for an MBF (NeuroInfo) atlas
    def open_mesh_dialog(self):
        """Make the surface meshes (.vtk, as in an MBF atlas's meshes_coronal folder) of regions added to an atlas: the
        new regions are those in the annotation's ontology but not in the reference atlas's; the reference atlas's own
        meshes are copied across unchanged (see atlas_meshes.py)."""
        import importlib.util
        if importlib.util.find_spec('vtk') is None:
            if not messagebox.askyesno(
                    "Generate meshes",
                    "Making meshes needs the 'vtk' library, which isn't installed in this Python:\n"
                    f"{sys.executable}\n\nInstall it now (a download of about 100 MB)?", parent=self.root):
                return

            def install():
                subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary', ':all:', 'vtk'], check=True)
                importlib.invalidate_caches()
                import site
                user_site = site.getusersitepackages()
                if site.ENABLE_USER_SITE and os.path.isdir(user_site) and user_site not in sys.path:
                    site.addsitedir(user_site)

            def installed(_result, error):
                if error is not None or importlib.util.find_spec('vtk') is None:
                    messagebox.showerror("Generate meshes", f"Could not install vtk:\n{error}", parent=self.root)
                    return
                self.open_mesh_dialog()
            self.run_in_thread("Installing vtk…", install, installed)
            return
        if getattr(self, 'mesh_window', None) is not None and self.mesh_window.winfo_exists():
            self.mesh_window.lift()
            return
        import atlas_meshes
        win = self.mesh_window = tk.Toplevel(self.root)
        win.title("Generate meshes")
        win.transient(self.root)
        frame = ttk.Frame(win, padding=10)
        frame.pack(fill='both', expand=True)
        frame.columnconfigure(1, weight=1)

        loaded = self.loaded_atlas_path if str(self.loaded_atlas_path).lower().endswith('.nrrd') else ''
        annotation = tk.StringVar(value=self.settings.get('mesh_annotation') or loaded)
        ontology = tk.StringVar()
        reference = tk.StringVar(value=self.settings.get('mesh_reference', ''))
        output = tk.StringVar()
        copy_existing = tk.BooleanVar(value=bool(self.settings.get('mesh_copy_existing', True)))
        auto = {'ontology': True, 'output': True}   # follow the annotation until the user edits them

        def follow_annotation(*_args):
            folder = Path(annotation.get().strip().strip('"')).parent
            if auto['ontology']:
                found = atlas_meshes.find_ontology(folder) if folder.is_dir() else None
                ontology.set(str(found) if found else '')
            if auto['output']:
                output.set(str(folder / atlas_meshes.MESH_FOLDER_NAME) if annotation.get().strip() else '')
            invalidate()

        state = {'plan': None}

        def invalidate(*_args):
            state['plan'] = None
            generate_button.configure(state='disabled')

        def browse_file(var, title, types, key=None):
            start = Path(var.get()).parent if var.get() else None
            chosen = filedialog.askopenfilename(title=title, parent=win, filetypes=types,
                                                initialdir=str(start) if start and start.is_dir() else None)
            if chosen:
                if key:
                    auto[key] = False
                var.set(str(Path(chosen)))

        def browse_folder(var, title, key=None):
            start = var.get() if var.get() and Path(var.get()).is_dir() else None
            chosen = filedialog.askdirectory(title=title, parent=win, initialdir=start)
            if chosen:
                if key:
                    auto[key] = False
                var.set(str(Path(chosen)))

        rows = (("Annotation volume", annotation,
                 lambda: browse_file(annotation, "Annotation volume with the new regions",
                                     [("NRRD volumes", "*.nrrd"), ("All files", "*.*")])),
                ("Ontology", ontology,
                 lambda: browse_file(ontology, "Ontology CSV with the new regions",
                                     [("CSV files", "*.csv"), ("All files", "*.*")], 'ontology')),
                ("Reference atlas folder", reference,
                 lambda: browse_folder(reference, "Original atlas folder (with meshes_coronal and its ontology)")),
                ("Output folder", output,
                 lambda: browse_folder(output, "Folder to write the meshes into", 'output')))
        for r, (label, var, command) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=r, column=0, sticky='e', padx=(0, 6), pady=2)
            entry = ttk.Entry(frame, textvariable=var, width=80)
            entry.grid(row=r, column=1, sticky='ew', pady=2)
            if var is ontology:
                entry.bind('<Key>', lambda e: auto.update(ontology=False))
            if var is output:
                entry.bind('<Key>', lambda e: auto.update(output=False))
            ttk.Button(frame, text="Browse…", command=command).grid(row=r, column=2, padx=(6, 0), pady=2)
        ttk.Checkbutton(frame, text="Copy the reference atlas's existing meshes into the output folder too",
                        variable=copy_existing, command=invalidate).grid(row=4, column=1, sticky='w', pady=(4, 0))
        ttk.Label(frame, foreground='gray30', justify='left', wraplength=620, text=(
            "Meshes are made only for regions that are in the ontology but not in the reference atlas's ontology. "
            "A region's mesh covers its sub-regions too. Existing meshes are never rewritten.")).grid(
            row=5, column=0, columnspan=3, sticky='w', pady=(6, 4))
        report = tk.Text(frame, height=14, width=100, wrap='word', state='disabled', font=('Consolas', 9))
        report.grid(row=6, column=0, columnspan=3, sticky='nsew')
        frame.rowconfigure(6, weight=1)
        buttons = ttk.Frame(frame)
        buttons.grid(row=7, column=0, columnspan=3, sticky='e', pady=(8, 0))
        check_button = ttk.Button(buttons, text="Check")
        check_button.pack(side='left')
        generate_button = ttk.Button(buttons, text="Generate", state='disabled')
        generate_button.pack(side='left', padx=6)
        ttk.Button(buttons, text="Close", command=win.destroy).pack(side='left')

        def show(text):
            report.configure(state='normal')
            report.delete('1.0', 'end')
            report.insert('end', text)
            report.configure(state='disabled')

        def inputs():
            values = [v.get().strip().strip('"') for v in (annotation, ontology, reference, output)]
            names = ("annotation volume", "ontology", "reference atlas folder", "output folder")
            for value, name in zip(values, names):
                if not value:
                    raise ValueError(f"Choose the {name}.")
            if not Path(values[0]).is_file():
                raise ValueError(f"Annotation volume not found:\n{values[0]}")
            if not Path(values[1]).is_file():
                raise ValueError(f"Ontology not found:\n{values[1]}")
            if not Path(values[2]).is_dir():
                raise ValueError(f"Reference atlas folder not found:\n{values[2]}")
            return values

        def check():
            if self.busy_label:
                show("Still busy with something else; try again when it finishes.")
                return
            try:
                ann, ont, ref, out = inputs()
            except ValueError as e:
                show(str(e))
                return
            notes = []
            if (self.loaded_atlas_path and Path(ann) == Path(self.loaded_atlas_path)
                    and self.edit_version != self.saved_version):
                notes.append("NOTE: the atlas on screen has painted changes that haven't been saved; the meshes are "
                             "made from the file on disk. Use Save volume as… in the Paint panel first to include them.\n")
            self.settings.update(mesh_annotation=ann, mesh_reference=ref, mesh_copy_existing=copy_existing.get())
            show("Reading the volumes…")

            def done(result, error):
                if not win.winfo_exists():
                    return
                if error is not None:
                    show(f"Could not plan the meshes:\n{error}")
                    return
                state['plan'] = result
                text = "\n".join(notes) + atlas_meshes.describe_plan(result)
                existing = [atlas_meshes.mesh_file_name(a, r) for r, a, _n, _v in result['new']
                            if (result['output'] / atlas_meshes.mesh_file_name(a, r)).exists()]
                if existing:
                    text += "\n\nAlready in the output folder (will be replaced): " + ", ".join(existing)
                if result['new'] or result['copy']:
                    generate_button.configure(state='normal')
                else:
                    text += "\n\nNothing to do."
                show(text)
            self.run_in_thread("Planning meshes…",
                               lambda: atlas_meshes.plan(ann, ont, ref, out, copy_existing.get()), done)

        def generate():
            p = state['plan']
            if p is None or self.busy_label:
                return
            generate_button.configure(state='disabled')

            def progress(text):
                self.progress_text = text

            def done(result, error):
                if not win.winfo_exists():
                    return
                if error is not None:
                    show(f"Mesh generation failed:\n{error}")
                    return
                show(atlas_meshes.describe_plan(p) + "\n\nDone. Wrote:\n" + "\n".join(f"   {f}" for f in result)
                     + (f"\nand copied {p['copy']} existing meshes." if p['copy'] else ""))
                self.message.configure(text=f"Wrote {len(result)} new mesh(es) to {p['output']}")
                state['plan'] = None
            self.run_in_thread("Generating meshes…", lambda: atlas_meshes.generate(p, progress), done)

        check_button.configure(command=check)
        generate_button.configure(command=generate)
        annotation.trace_add('write', follow_annotation)
        for var in (ontology, reference, output):
            var.trace_add('write', invalidate)
        follow_annotation()
        show("Choose the files, then press Check to see what will be made.")

    def undo(self):
        if not self.undo_stack:
            self.paint_status.set("Nothing to undo.")
            return
        kind, flat, data = self.undo_stack.pop()
        if kind == 'restore':   # a restored section: put the old labels back, leaving the red drawing as it is
            self.volume.reshape(-1)[flat] = data
            self.edit_count -= len(flat)
            self.edit_version += 1
            self.value_cache = None
            self.refresh_all()
            self.update_paint_status()
            return
        overlay = self.overlay.reshape(-1)
        if kind == 'stroke':
            overlay[flat] = not data
            self.overlay_count += -len(flat) if data else len(flat)
        else:
            self.volume.reshape(-1)[flat] = data
            overlay[flat] = True
            self.overlay_count += len(flat)
            self.edit_count -= len(flat)
            self.edit_version += 1
            self.value_cache = None
        self.refresh_all()
        self.update_paint_status()

    # ---- settings
    def on_root_configure(self, event):
        """Move the info window by as much as the main window moved, when that option is on."""
        if event.widget is not self.root or self.last_root_pos is None:
            return
        position = (self.root.winfo_x(), self.root.winfo_y())
        if position == self.last_root_pos:
            return
        dx, dy = position[0] - self.last_root_pos[0], position[1] - self.last_root_pos[1]
        self.last_root_pos = position
        if self.follow_var.get():
            for window in (self.info.window, self.paint_window):
                if window is not None and window.winfo_exists() and window.winfo_viewable():
                    window.geometry(f"+{window.winfo_x() + dx}+{window.winfo_y() + dy}")

    def open_settings(self):
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return
        window = self.settings_window = tk.Toplevel(self.root)
        window.title("CCF browser: settings")
        window.transient(self.root)
        body = ttk.Frame(window, padding=12)
        body.pack(fill='both', expand=True)
        ttk.Checkbutton(body, text="Move the info window along when the main window is moved",
                        variable=self.follow_var).pack(anchor='w')
        ttk.Checkbutton(body, text="Mouse wheel zooms (Ctrl+wheel steps through slices; off: the other way round; Shift+wheel always steps)",
                        variable=self.wheel_zoom_var).pack(anchor='w', pady=(4, 0))

        box = ttk.LabelFrame(body, text="Atlas", padding=8)
        box.pack(fill='x', pady=(12, 0))
        mode = tk.StringVar(value='custom' if self.atlas_path else 'abc')
        path_var = tk.StringVar(value=self.alt_atlas_path)
        self.settings_status.set('')
        ttk.Radiobutton(box, text="Allen CCF 2020, from the ABC atlas cache (default)", value='abc',
                        variable=mode).pack(anchor='w')
        ttk.Radiobutton(box, text="Another annotation volume (.nrrd, .nii or .nii.gz, any size, e.g. Allen2017 or Waxholm rat):",
                        value='custom', variable=mode).pack(anchor='w', pady=(6, 0))
        row = ttk.Frame(box)
        row.pack(fill='x', padx=(20, 0))
        entry = ttk.Entry(row, textvariable=path_var, width=60)
        entry.pack(side='left', fill='x', expand=True)
        entry.bind('<Return>', lambda e: close_settings())   # Enter in the path box is OK

        def browse():
            current = path_var.get().strip().strip('"')
            start = Path(current).parent if current else None
            chosen = filedialog.askopenfilename(
                title="Choose an annotation volume", parent=window,
                initialdir=str(start) if start and start.is_dir() else None,
                filetypes=[("Annotation volumes", "*.nrrd *.nii *.nii.gz"), ("All files", "*.*")])
            if chosen:
                path_var.set(chosen)
                mode.set('custom')
        ttk.Button(row, text="Browse…", command=browse).pack(side='left', padx=(4, 0))

        def open_folder():
            """Show the chosen volume's folder in the system file browser (with the file selected, when it exists)."""
            text = path_var.get().strip().strip('"')
            if not text:
                self.settings_status.set("No file chosen yet.")
                return
            current = Path(text)
            folder = current if current.is_dir() else current.parent
            if not folder.is_dir():
                self.settings_status.set("That folder does not exist.")
                return
            try:
                if sys.platform == 'win32':
                    subprocess.Popen(['explorer', '/select,', str(current)] if current.is_file() else ['explorer', str(folder)])
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', str(folder)])
                else:
                    subprocess.Popen(['xdg-open', str(folder)])
            except OSError as e:
                self.settings_status.set(f"Could not open the folder: {e}")
        ttk.Button(row, text="Open folder", command=open_folder).pack(side='left', padx=(4, 0))
        flip_row = ttk.Frame(box)
        flip_row.pack(fill='x', padx=(20, 0), pady=(6, 0))
        ttk.Label(flip_row, text="If it looks reversed, flip:").pack(side='left')
        flip_vars = [tk.BooleanVar(value=False) for _ in range(3)]
        for var, text in zip(flip_vars, ("front ↔ back", "top ↔ bottom", "left ↔ right")):
            ttk.Checkbutton(flip_row, text=text, variable=var).pack(side='left', padx=(10, 0))

        def show_flips(*_args):
            for var, value in zip(flip_vars, self.atlas_flips.get(path_var.get().strip().strip('"'), (False,) * 3)):
                var.set(value)
        path_var.trace_add('write', show_flips)
        show_flips()
        ttk.Label(box, foreground='#555555', justify='left', wraplength=520, text=(
            "The orientation is read from the file's header, which is sometimes wrong; the flips (remembered per file, "
            "and kept when the volume is saved) correct it. They do not apply to volumes on the CCF grid. "
            "Region names and colours come from an ontology*.csv in the same folder, if there is one (the Allen2017 "
            "folder has ontology_api_2017.csv). Label numbers differ between the 2017 and ABC atlases.")).pack(
            anchor='w', pady=(6, 0))
        bregma_box = ttk.LabelFrame(body, text="Bregma (CCF mm)", padding=8)
        bregma_box.pack(fill='x', pady=(12, 0))
        bregma_vars = [tk.StringVar(value=f"{value:g}") for value in self.bregma]

        def bregma_edited(*_args):
            try:
                values = [float(v.get()) for v in bregma_vars]
            except ValueError:
                return   # half-typed: keep the last good position
            self.bregma = values
            if self.volume is not None:
                self.refresh_all()
                self.on_leave()

        for column, (name, var) in enumerate(zip(("AP", "DV", "ML"), bregma_vars)):
            ttk.Label(bregma_box, text=name).grid(row=0, column=2 * column, padx=(0 if column == 0 else 12, 4))
            ttk.Entry(bregma_box, textvariable=var, width=8).grid(row=0, column=2 * column + 1)
            var.trace_add('write', bregma_edited)

        def reset_bregma():
            for var, value in zip(bregma_vars, BREGMA_DEFAULT_MM):
                var.set(f"{value:g}")

        ttk.Button(bregma_box, text="Reset", command=reset_bregma).grid(row=0, column=6, padx=(14, 0))
        ttk.Label(bregma_box, foreground='#555555', justify='left', wraplength=520, text=(
            "Where bregma lies in the CCF (AP from the anterior edge, DV from the dorsal edge, ML from the left edge). "
            "The CCF does not define it; the default follows the IBL atlas convention, and sources differ by a few "
            "hundred um, so enter your lab's values. The info window then also gives positions relative to bregma "
            "(AP + anterior, DV + dorsal).")).grid(row=1, column=0, columnspan=7, sticky='w', pady=(6, 0))
        status = ttk.Label(body, textvariable=self.settings_status, foreground='#7a3b00')
        status.pack(anchor='w', pady=(8, 0))

        def wanted_atlas():
            return path_var.get().strip().strip('"') if mode.get() == 'custom' else ''

        def wanted_flips():
            return tuple(var.get() for var in flip_vars)

        def apply_atlas():
            """Switch to the atlas the box asks for; True if the switch started (or was not needed)."""
            wanted = wanted_atlas()
            if mode.get() == 'custom' and not Path(wanted).is_file():
                self.settings_status.set("Choose an existing atlas file first.")
                return False
            if wanted == self.atlas_path and (not wanted or wanted_flips() == self.loaded_flips):
                self.settings_status.set("That atlas is already in use.")
                return True
            if self.busy_label:
                self.settings_status.set("Busy loading; try again when it finishes.")
                return False
            if (self.edit_version != self.saved_version or self.overlay_count) and not messagebox.askyesno(
                    "Painted changes", "Switching atlas discards the applied edits (they are not saved anywhere). The painted red "
                    "overlay is kept if the new atlas has the same size and resolution as this one, and discarded if it has not. "
                    "Continue?", parent=window):
                return False
            self.atlas_path = wanted
            if wanted:
                self.alt_atlas_path = wanted
                self.atlas_flips[wanted] = wanted_flips()
            self.settings.update({'atlas_path': wanted, 'alt_atlas_path': self.alt_atlas_path,
                                  'atlas_flips': {k: list(v) for k, v in self.atlas_flips.items()}})
            save_settings(self.settings)
            self.start_atlas()
            self.settings_status.set("Loading the atlas…")
            self.update_settings_close()
            return True

        def close_settings():
            """OK: load the atlas chosen here if it is a different one, then close (the picture shows "Loading Atlas…")."""
            if self.busy_label:   # the OK button is greyed out while something loads; the window's own X is ignored too
                return
            if (wanted_atlas() != self.atlas_path or (wanted_atlas() and wanted_flips() != self.loaded_flips))                     and not apply_atlas():
                return   # the reason is in the status line; the window stays open
            window.destroy()

        def cancel_settings():
            """The window's own X: close without loading a newly chosen atlas."""
            if not self.busy_label:
                window.destroy()

        buttons = ttk.Frame(body)
        buttons.pack(fill='x', pady=(10, 0))
        self.settings_close_button = ttk.Button(buttons, text="OK", command=close_settings)
        self.settings_close_button.pack(side='right')
        window.protocol('WM_DELETE_WINDOW', cancel_settings)
        self.update_settings_close()
        x = self.root.winfo_x() + 80
        window.geometry(f"+{x}+{self.root.winfo_y() + 80}")

    def update_settings_close(self):
        """Grey out the Settings window's OK button while anything is loading (an atlas switch in particular)."""
        button = getattr(self, 'settings_close_button', None)
        try:
            if button is not None and button.winfo_exists():
                button.state(['disabled'] if self.busy_label else ['!disabled'])
        except tk.TclError:
            pass

    # ---- background work
    def draw_busy_overlay(self):
        """Put (or take away) the big "Loading ..." note over the middle of the main picture."""
        self.canvas.delete('busy')
        if not self.busy_overlay:
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 10 or height < 10:
            return
        item = self.canvas.create_text(width // 2, height // 2, text=self.busy_overlay, fill='white', tags='busy',
                                       font=('Segoe UI', 22, 'bold'))
        x0, y0, x1, y1 = self.canvas.bbox(item)
        box = self.canvas.create_rectangle(x0 - 24, y0 - 14, x1 + 24, y1 + 14, fill='#202020', outline='#808080',
                                           tags='busy')
        self.canvas.tag_raise(item, box)

    def run_in_thread(self, label, function, on_done, overlay=''):
        """Run `function()` on a thread; `on_done(result, error)` runs on the main thread when it finishes. `overlay`,
        if given, is shown over the middle of the picture until the work is done."""
        self.busy_label, self.progress_text = label, ''
        self.busy_overlay = overlay
        self.message.configure(text=label)
        self.update_cursors()
        self.update_settings_close()
        self.draw_busy_overlay()

        def runner():
            try:
                self.work.put((on_done, function(), None))
            except Exception as e:
                self.work.put((on_done, None, e))
        threading.Thread(target=runner, daemon=True).start()

    def poll_work(self):
        try:
            while True:
                on_done, result, error = self.work.get_nowait()
                self.busy_label = ''
                self.busy_overlay = ''
                self.update_cursors()
                self.message.configure(text="")
                on_done(result, error)
                self.update_settings_close()   # (on_done may itself have started another load)
                self.draw_busy_overlay()       # (and if it started one, the new note is drawn; else it is cleared)
        except queue.Empty:
            if self.busy_label:
                self.message.configure(text=f"{self.busy_label} {self.progress_text}")
        self.root.after(100, self.poll_work)

    # ---- the atlas
    def start_atlas(self):
        if self.atlas_path:
            name = Path(self.atlas_path).name
            flips = self.loading_flips = self.atlas_flips.get(self.atlas_path, (False, False, False))
            self.run_in_thread(f"Loading {name}…", lambda: load_custom_atlas(self.atlas_path, flips), self.atlas_ready,
                               overlay="Loading Atlas…")
            return
        base = find_atlas_cache_dir()
        if base is None:
            self.message.configure(text="No atlas folder chosen.")
            return
        path, size, present = volume_status(base)
        download = False
        if not present:
            size_text = f"{size / 1e6:.1f} MB" if size else "a few MB"
            download = messagebox.askyesno(
                "Download the CCF annotation volume?",
                f"The CCF annotation volume is not in the atlas cache ({base}). It is a one-time download of "
                f"{size_text}. Download it now?", parent=self.root)
            if not download:
                self.message.configure(text="The CCF volume is needed; restart to be asked again.")
                return
        self.run_in_thread("Loading the CCF atlas…" if present else "Downloading the CCF volume…",
                           lambda: load_atlas(base, download), self.atlas_ready, overlay="Loading Atlas…")

    def atlas_ready(self, result, error):
        if error is not None:
            messagebox.showerror("CCF atlas", f"Could not load the annotation volume:\n{error}", parent=self.root)
            self.settings_status.set("Could not load that atlas.")
            if self.volume is not None:   # a switch that failed: the previous atlas is still the one on screen
                self.atlas_path = self.loaded_atlas_path
                self.settings['atlas_path'] = self.atlas_path
            if self.atlas_path and self.volume is None:   # a bad custom file at startup: fall back to the default
                self.atlas_path = ''
                self.start_atlas()
            return
        self.volume, self.region_table, self.lut, self.atlas_info = result
        previous = tuple(VOLUME_SHAPE), VOXEL_UM
        set_geometry(self.volume.shape, self.atlas_info['voxel_um'])
        for name in VIEWS:   # another size of atlas: start in the middle; the same size: keep the places
            count = VOLUME_SHAPE[VIEW_AXIS[name]]
            self.slices[name] = count // 2 if previous != (tuple(VOLUME_SHAPE), VOXEL_UM) \
                else min(max(self.slices[name], 0), count - 1)
        self.loaded_atlas_path = self.atlas_path
        self.loaded_flips = self.loading_flips if self.atlas_path else (False, False, False)
        self.update_comparison_mode()
        self.label_ids = (self.region_table['region_id'].to_numpy()
                          if self.region_table is not None and 'region_id' in self.region_table else None)
        # The painted (red) overlay belongs to positions, not to an atlas: an atlas of the same size and resolution keeps it. The edits
        # already applied were made in the old volume, so they and their undo steps go with it.
        keep_overlay = (self.overlay is not None and self.overlay.shape == tuple(VOLUME_SHAPE)
                        and previous == (tuple(VOLUME_SHAPE), VOXEL_UM))
        if not keep_overlay:
            self.overlay, self.overlay_count = None, 0
        self.edit_count, self.undo_stack = 0, []
        self.edit_version = self.saved_version = 0
        self.value_cache, self.stroke = None, None
        self.set_tool('none')
        self.settings_status.set("Atlas loaded: " + (Path(self.atlas_path).name if self.atlas_path else "default CCF")
                                 + (" (a greyscale image: shown in grey, no regions)" if self.atlas_info.get('intensity') else ""))
        self.root.title("CCF browser" + (f" - {Path(self.atlas_path).name}" if self.atlas_path else ""))
        self.apply_saturation()
        self.fit_to_window()
        self.refresh_all()
        self.on_leave()
        if self.atlas_loaded_once:   # the atlas was swapped; the CSVs stay as they are
            return
        self.atlas_loaded_once = True
        self.autoload = [(s, s.start_text.strip().strip('"')) for s in self.slots
                         if s.start_text.strip() and Path(s.start_text.strip().strip('"')).is_file()]
        self.load_next_saved()

    def update_comparison_mode(self):
        """Offer the Differences mode (and load the standard CCF to compare with) while an alternate atlas is on
        screen; take both away when the standard atlas itself is."""
        alternate = (bool(self.loaded_atlas_path) and self.on_ccf_grid   # only the same grid can be compared voxel by voxel
                     and not self.atlas_info.get('intensity'))               # and only a map of regions with another
        self.diff_cache = None
        if alternate:
            self.mode_box.configure(values=ATLAS_MODES + DIFF_MODES)
            if self.reference is None and not self.reference_loading:
                base = quiet_atlas_cache_dir()
                _path, _size, present = volume_status(base) if base is not None else (None, None, False)
                if present:
                    self.reference_loading = True
                    self.run_in_thread("Loading the standard CCF to compare with…",
                                       lambda: load_atlas(base, False), self.reference_ready,
                                       overlay="Loading Atlas…")
                else:
                    self.message.configure(text="The standard CCF is not in the atlas cache, so differences cannot be shown.")
        else:
            self.reference = None
            self.mode_box.configure(values=ATLAS_MODES)
            if self.mode_var.get() in DIFF_MODES:
                self.mode_var.set(ATLAS_MODES[0])
        self.sync_panes()

    def reference_ready(self, result, error):
        self.reference_loading = False
        if error is not None:
            self.reference = None
            self.message.configure(text=f"Could not load the standard CCF for comparison: {error}")
            self.load_next_saved()
            return
        volume, table, lut = result[:3]
        self.reference = {'volume': volume, 'table': table, 'lut': lut}
        self.diff_cache = None
        self.apply_saturation()
        self.sync_panes()
        self.refresh_all()
        self.on_leave()
        self.load_next_saved()   # the remembered CSVs wait while this load is busy (see atlas_ready)

    def diff_state(self):
        """(alt_lut, ref_lut, basis) for comparing the atlas on screen with the standard one, or None while the
        standard one is not loaded. Rebuilt when the volume's regions change (painting adds labels)."""
        if self.reference is None or self.volume is None or not self.loaded_atlas_path:
            return None
        merge = self.mode_var.get() == DIFF_MODE_MERGED
        key = (id(self.region_table), len(self.lut), self.edit_version, id(self.reference), merge)
        if self.diff_cache is None or self.diff_cache[0] != key:
            alt_lut, ref_lut, basis = comparison_luts(
                self.region_table, len(self.lut), self.label_ids, self.reference['table'], len(self.reference['lut']),
                merge_unassigned=merge)
            self.diff_cache = (key, alt_lut, ref_lut, basis, None)
        return self.diff_cache[1:4]

    def diff_totals(self):
        """Whole-volume voxel counts (changed, only alternate, only standard), computed once per state of the volume."""
        if self.diff_state() is None:
            return None
        if self.diff_cache[4] is None:
            alt_lut, ref_lut = self.diff_cache[1], self.diff_cache[2]
            totals = np.zeros(3, dtype=np.int64)
            for start in range(0, self.volume.shape[0], 16):
                alt = alt_lut[self.volume[start:start + 16]]
                ref = ref_lut[self.reference['volume'][start:start + 16]]
                changed = alt != ref
                totals += (int((changed & (alt > 0) & (ref > 0)).sum()), int((changed & (ref == 0)).sum()),
                           int((changed & (alt == 0)).sum()))
            self.diff_cache = self.diff_cache[:4] + (tuple(int(t) for t in totals),)
        return self.diff_cache[4]

    def diff_slice(self):
        """(left_rgb, right_rgb, counts) of the Differences view of the current slice, or None while it cannot be made."""
        state = self.diff_state()
        if state is None or self.ref_lut_view is None:
            return None
        return diff_panes(view_slice(self.volume, self.view_name, self.slice), self.lut_view,
                          view_slice(self.reference['volume'], self.view_name, self.slice), self.ref_lut_view,
                          state[0], state[1])

    def apply_saturation(self):
        value = int(round(float(self.sat_var.get())))
        self.sat_label.configure(text=f"{value}%")
        if self.lut is not None:
            self.lut_view = desaturated(self.lut.astype(np.float32), value / 100.0) if value < 100 else self.lut
        if self.reference is not None:
            ref_lut = self.reference['lut']
            self.ref_lut_view = desaturated(ref_lut.astype(np.float32), value / 100.0) if value < 100 else ref_lut

    def change_saturation(self):
        self.apply_saturation()
        self.request_render()

    def change_view(self):
        if self.view_var.get() in VIEWS and self.view_var.get() != self.view_name:
            self.view_name = self.view_var.get()
            self.fit_to_window()
            self.refresh_all()
            self.on_leave()

    def load_next_saved(self):
        while self.autoload and not self.busy_label:
            slot, text = self.autoload.pop(0)
            slot.path_var.set(text)
            self.load_slot(slot)

    # ---- CSV
    def choose_file(self, slot):
        current = slot.path_var.get().strip().strip('"')
        start = Path(current).parent if current else None
        chosen = filedialog.askopenfilename(
            title="Choose a CSV of cells or an MBF landmarks XML", parent=self.root, initialdir=str(start) if start and start.is_dir() else None,
            filetypes=[("CSV files and landmarks", "*.csv *.xml"), ("All files", "*.*")])
        if chosen:
            slot.path_var.set(chosen)
            self.load_slot(slot, chosen=True)

    def slot_for_new_file(self):
        """The first empty slot, else the one whose file was loaded least recently."""
        for slot in self.slots:
            if slot.points is None and not slot.loading:
                return slot
        return min(self.slots, key=lambda s: s.loaded_at)

    def on_drop(self, event, slot):
        paths = self.root.tk.splitlist(event.data)
        if paths:
            slot = slot or self.slot_for_new_file()
            slot.path_var.set(paths[0])
            self.load_slot(slot, chosen=True)
        return event.action

    def restore_entry(self, slot):
        slot.path_var.set(str(slot.points.path) if slot.points is not None else '')

    def load_slot(self, slot, chosen=False):
        """Load the path in the slot's box. `chosen`: the user just picked or dropped the file, so even re-loading
        the same file into the same slot is worth a question (Enter or a paste in an unchanged box is not)."""
        text = slot.path_var.get().strip().strip('"')
        if not text:
            return
        path = Path(text)
        same_slot = slot.points is not None and slot.points.path == path
        if same_slot and not chosen:
            slot.path_var.set(str(path))
            return
        if self.busy_label:
            self.message.configure(text="Still busy loading; try again when it finishes.")
            self.restore_entry(slot)
            return
        if not path.is_file():
            self.message.configure(text="No such file.")
            self.restore_entry(slot)
            return
        for other in self.slots:
            if other.points is not None and other.points.path == path:
                where = "this slot" if other is slot else f"CSV {other.index + 1}"
                if not messagebox.askyesno(
                        "File already loaded",
                        f"{path.name} is already loaded in {where}.\n\nLoad it into CSV {slot.index + 1} anyway?",
                        parent=self.root):
                    self.restore_entry(slot)
                    return
                break
        slot.path_var.set(str(path))
        slot.loading = True
        self.run_in_thread(f"Reading {path.name}… (large files take a while)",
                           lambda: PointSet(read_points_csv(path, self.set_progress), path),
                           lambda result, error: self.points_ready(slot, result, error), overlay="Loading CSV…")

    def set_progress(self, text):
        self.progress_text = text

    def clear_slot(self, slot):
        if slot.loading:
            return
        slot.points = None
        slot.path_var.set('')
        slot.group_var.set('')
        self.slot_widgets[slot.index]['combo'].configure(values=())
        self.rebuild_toggles(slot)
        self.refresh_all()

    def rebuild_toggles(self, slot):
        """One checkbox per group of this slot, to show or hide that group's cells."""
        frame = self.slot_widgets[slot.index]['toggles']
        for child in frame.winfo_children():
            child.destroy()
        old = {k: v.get() for k, v in slot.visible.items()}
        slot.visible = {}
        points = slot.points
        if points is None:
            slot.visible_mask = None
            return
        ttk.Label(frame, text="Show:").pack(side='left')
        ttk.Button(frame, text="All", width=4, command=lambda s=slot: self.toggle_all(s)).pack(side='left', padx=(4, 2))
        for index, label in enumerate(points.group_labels):
            color, shape = slot.styles[index % len(slot.styles)]
            variable = tk.BooleanVar(value=old.get(label, True))
            slot.visible[label] = variable
            tk.Checkbutton(frame, text=f"{SHAPE_SYMBOLS[shape]} {label} ({points.totals[index]:,})",
                           variable=variable, fg=hex_color(color),
                           command=lambda s=slot: self.visibility_changed(s)).pack(side='left', padx=4)
        self.visibility_changed(slot, refresh=False)

    def toggle_all(self, slot):
        """Switch every group of the slot off if all are on, otherwise all on."""
        if slot.points is None:
            return
        turn_on = not all(variable.get() for variable in slot.visible.values())
        for variable in slot.visible.values():
            variable.set(turn_on)
        self.visibility_changed(slot)

    def update_cursors(self):
        """The busy cursor over the whole window, the image included (it has a cursor of its own), else the tool's."""
        busy = bool(self.busy_label)
        self.root.configure(cursor='watch' if busy else '')
        self.canvas.configure(cursor='watch' if busy else ('pencil' if self.tool != 'none' else 'crosshair'))

    def visibility_changed(self, slot, refresh=True):
        if slot.points is not None:
            slot.visible_mask = np.array([slot.visible[l].get() for l in slot.points.group_labels])
        if refresh:
            self.refresh_all()

    def points_ready(self, slot, result, error):
        slot.loading = False
        if error is not None:
            messagebox.showerror("CSV", f"Could not read that file:\n{error}", parent=self.root)
            self.restore_entry(slot)
            self.load_next_saved()
            return
        slot.points = result
        slot.loaded_at = time.time()
        points = result
        self.remember_recent(slot, str(points.path))
        self.slot_widgets[slot.index]['combo'].configure(values=points.group_columns or ['(all cells)'])
        slot.group_var.set(points.group_column or '(all cells)')
        slot.visible = {}
        self.rebuild_toggles(slot)
        if not self.autoload:   # a file the user just opened: jump to the slice holding most of its cells
            planes = np.rint(points.coords[self.axis] / VOXEL_UM).astype(np.int64)
            planes = planes[(planes >= 0) & (planes < self.n_slices)]   # cells outside the volume are on no slice
            if len(planes) and sum(s.points is not None for s in self.slots) == 1:
                self.slice = int(np.argmax(np.bincount(planes, minlength=self.n_slices)))
        note = f" ({points.skipped:,} rows skipped)" if points.skipped else ""
        self.message.configure(text=f"CSV {slot.index + 1}, {points.kind}: {len(points.ap):,} cells{note}")
        self.refresh_all()
        self.load_next_saved()

    def change_group(self, slot):
        if slot.points is not None and slot.group_var.get() in slot.points.group_columns:
            slot.points.set_group_column(slot.group_var.get())
            slot.visible = {}
            self.rebuild_toggles(slot)
            self.refresh_all()

    # ---- navigation
    def go_to(self, index):
        index = int(min(max(index, 0), self.n_slices - 1))
        if index != self.slice:
            self.slice = index
            self.refresh_all()

    def step(self, delta):
        self.go_to(self.slice + delta)

    def on_scrollbar(self, *args):
        if args[0] == 'moveto':
            self.go_to(round(float(args[1]) * self.n_slices))
        elif args[0] == 'scroll':
            self.step(int(args[1]) * (10 if args[2] == 'pages' else 1))

    def on_wheel(self, event):
        """Shift+wheel always steps through the slices one at a time (Ctrl+Shift+wheel: ten). Without Shift the wheel
        zooms about the cursor or steps through slices, as set in Settings; Ctrl swaps the two."""
        shift, ctrl = bool(event.state & 0x1), bool(event.state & 0x4)
        direction = -1 if event.delta > 0 else 1
        if shift:
            self.step(direction * (10 if ctrl else 1))
        elif ctrl == self.wheel_zoom_var.get():   # slices
            self.step(direction)
        else:
            self.zoom_at(1.2 ** (event.delta / 120.0), (self.pane_x(event.x), event.y))

    def pane_count(self):
        """2 in the Differences view once the standard atlas to compare with is loaded, else 1."""
        return 2 if (self.mode_var.get() in DIFF_MODES and self.reference is not None and self.loaded_atlas_path) else 1

    def pane_width(self):
        count = self.pane_count()
        return max(50, (self.canvas.winfo_width() - PANE_GAP * (count - 1)) // count)

    def pane_x(self, x):
        """A canvas x position as an x position within the pane it is over (both panes show the same view)."""
        if self.pane_count() == 2:
            right_start = self.pane_width() + PANE_GAP
            return x - right_start if x >= right_start else x
        return x

    def in_paint_pane(self, x):
        """Whether painting is allowed at canvas x: always with one map, and in the Differences view only on the right
        (the alternate atlas), never on the standard atlas at the left."""
        return self.pane_count() == 1 or x >= self.pane_width() + PANE_GAP

    def sync_panes(self):
        """Refit the picture when the number of panes just changed."""
        count = self.pane_count()
        if count != self.panes_shown:
            self.panes_shown = count
            if self.volume is not None:
                self.fit_to_window()

    def mode_changed(self):
        self.sync_panes()
        self.refresh_all()

    def canvas_size(self):
        """(width of one pane, height): the area the picture is fitted into."""
        return self.pane_width(), max(50, self.canvas.winfo_height())

    def fit_scale(self):
        """The scale at which the whole image just fits the window: the most zoomed-out view there is."""
        width, height = self.canvas_size()
        image_w, image_h = VIEW_IMAGE_SIZE[self.view_name]
        return min(width / image_w, height / image_h) * 0.97

    def clamp_view(self):
        """Keep the picture covering the window: along an axis where it is larger than the window it cannot be panned
        past its edge, and where it is smaller it stays centred. So no blank space beyond the image can be brought
        into view by panning or zooming."""
        scale, ox, oy = self.view
        width, height = self.canvas_size()
        image_w, image_h = VIEW_IMAGE_SIZE[self.view_name]

        def limited(offset, extent, image_extent):
            size = image_extent * scale
            return (extent - size) / 2 if size <= extent else min(0.0, max(extent - size, offset))

        self.view[1], self.view[2] = limited(ox, width, image_w), limited(oy, height, image_h)

    def fit_to_window(self):
        self.zoom_goal = None
        width, height = self.canvas_size()
        image_w, image_h = VIEW_IMAGE_SIZE[self.view_name]
        scale = self.fit_scale()
        self.view = [scale, (width - image_w * scale) / 2, (height - image_h * scale) / 2]
        self.fit = True
        self.request_render()

    def zoom_at(self, factor, anchor=None):
        """Zoom by `factor` about a canvas point (default the centre), easing toward the new scale over a few frames;
        wheel notches arriving meanwhile add up."""
        scale, ox, oy = self.view
        ax, ay = anchor if anchor else (self.canvas_size()[0] / 2, self.canvas_size()[1] / 2)
        base = self.zoom_goal if self.zoom_goal is not None else scale
        self.zoom_goal = min(max(base * factor, self.fit_scale()), 60.0)   # never further out than the full view
        self.zoom_anchor = (ax, ay, (ax - ox) / scale, (ay - oy) / scale)
        self.fit = False
        if base == scale or self.zoom_goal != base:
            self.animate_zoom()

    def animate_zoom(self):
        if self.zoom_goal is None:
            return
        scale = self.view[0]
        ratio = self.zoom_goal / scale
        done = abs(ratio - 1) < 0.003
        new_scale = self.zoom_goal if done else scale * ratio ** 0.22
        ax, ay, u, v = self.zoom_anchor
        self.view = [new_scale, ax - u * new_scale, ay - v * new_scale]
        self.clamp_view()
        self.request_render()
        if done:
            self.zoom_goal = None
        else:
            self.root.after(16, self.animate_zoom)

    def on_press(self, event):
        self.zoom_goal = None
        self.deactivate_tip()
        if self.tool != 'none' and self.volume is not None and self.in_paint_pane(event.x):
            if self.overlay is None:
                self.overlay = np.zeros(VOLUME_SHAPE, dtype=bool)
            self.stroke, self.last_paint = [], None
            self.paint_to(event.x, event.y)
            return
        self.drag = (event.x, event.y, self.view[1], self.view[2])

    def on_release(self, event):
        self.drag = None
        if self.stroke is not None:
            if self.stroke:
                self.undo_stack.append(('stroke', np.unique(np.concatenate(self.stroke)), self.tool == 'draw'))
            self.stroke, self.last_paint = None, None
            self.refresh_all()
            self.update_paint_status()

    def on_drag(self, event):
        if self.stroke is not None:
            if self.in_paint_pane(event.x):
                self.paint_to(event.x, event.y)
            else:
                self.last_paint = None   # the stroke wandered onto the standard atlas: nothing is painted there
            self.mouse = (event.x, event.y)
            self.draw_brush()
            return
        self.deactivate_tip()
        if self.drag is None:
            return
        x0, y0, ox, oy = self.drag
        self.view[1], self.view[2] = ox + event.x - x0, oy + event.y - y0
        self.clamp_view()
        self.fit = False
        self.request_render()

    def on_canvas_resize(self):
        if self.busy_overlay:
            self.draw_busy_overlay()
        if self.volume is not None and (self.fit or self.view[0] < self.fit_scale()):
            self.fit_to_window()   # also when the window shrank below the current zoom: never smaller than the full view
        else:
            self.clamp_view()
            self.request_render()

    # ---- drawing
    def refresh_all(self):
        if self.volume is None:
            return
        self.request_render()
        self.refresh_info_slice()
        self.refresh_overall()
        self.scrollbar.set(self.slice / self.n_slices, (self.slice + 1) / self.n_slices)
        self.slice_label.configure(text=f"{self.view_name} {self.slice + 1} / {self.n_slices}   "
                                        f"{AXIS_NAMES[self.axis]} {self.slice_um / 1000:.3f} mm ({self.slice_bregma_text()})")

    def request_render(self):
        if not self.render_pending:
            self.render_pending = True
            self.root.after_idle(self.render)

    def slab_cells(self, slot):
        """Indices of the slot's cells in view on this slice (all groups), or None if the slot is empty."""
        if slot.points is None:
            return None
        half = (max(0, int(self.slab_var.get() or 0)) + 0.5) * VOXEL_UM
        return slot.points.slab_indices(self.axis, self.slice_um, half)

    def shown_cells(self, slot):
        """Indices of the slot's cells in the slab whose group is switched on."""
        inside = self.slab_cells(slot)
        if inside is None or not len(inside) or slot.visible_mask is None:
            return np.zeros(0, dtype=np.int64)
        return inside[slot.visible_mask[slot.points.codes[inside]]]

    def outline_style(self, mode):
        """(colour, opacity) of the region outlines for this display mode, or None for none: dark lines over the bright
        regions, light ones over the dimmed views, and nothing in Boundaries (which is outlines already)."""
        if not self.outline_var.get() or mode == 'Boundaries' or self.atlas_info.get('intensity'):
            return None   # (a greyscale image has no regions to outline)
        return ((190, 190, 190), 0.35) if mode in ('Regions (dim)',) + DIFF_MODES else ((0, 0, 0), 0.6)

    def draw_edge_labels(self, origin, pane_width, height):
        top, bottom, left, right = VIEW_EDGE_LABELS[self.view_name]
        for text, x, y, anchor in ((top, origin + pane_width / 2, 10, 'n'), (bottom, origin + pane_width / 2, height - 8, 's'),
                                   (left, origin + 8, height / 2, 'w'), (right, origin + pane_width - 8, height / 2, 'e')):
            self.canvas.create_text(x, y, text=text, fill='#8a8a8a', anchor=anchor, tags='label', font=('Segoe UI', 9))

    def render(self):
        self.render_picture()
        if self.tip_active:
            self.refresh_tip()
        if self.busy_overlay:
            self.draw_busy_overlay()   # keeps the note on top of the freshly drawn picture

    def render_picture(self):
        self.render_pending = False
        self.sync_panes()
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if self.volume is None or width < 10 or height < 10:
            return
        layers = []
        across, down = VIEW_CELL_AXES[self.view_name]
        for slot in self.slots:
            shown = self.shown_cells(slot)
            if len(shown):
                layers.append((slot.points.coords[across][shown], slot.points.coords[down][shown],
                               slot.points.codes[shown], slot.styles))
        overlay = view_slice(self.overlay, self.view_name, self.slice) if self.overlay_count > 0 else None
        labels = view_slice(self.volume, self.view_name, self.slice)
        radius = int(self.dot_var.get() or 1)
        mode = self.mode_var.get()
        compared = self.diff_slice() if self.pane_count() == 2 else None
        self.canvas.delete('label')
        if compared is not None:
            # Differences: the standard atlas on the left, the alternate on the right, same view in both.
            pane_w = self.pane_width()
            ref_labels = view_slice(self.reference['volume'], self.view_name, self.slice)
            outlines = self.outline_style('Regions (dim)')
            left = render_frame(ref_labels, self.ref_lut_view, 'Regions', self.view, (pane_w, height), layers, radius,
                                None, compared[0], outlines)
            right = render_frame(labels, self.lut_view, 'Regions', self.view, (pane_w, height), layers, radius,
                                 overlay, compared[1], outlines)
            frame = np.empty((height, width, 3), dtype=np.uint8)
            frame[:] = (90, 90, 90)
            frame[:, :pane_w] = left
            frame[:, pane_w + PANE_GAP:pane_w + PANE_GAP + pane_w] = right
            self.photo = ImageTk.PhotoImage(Image.fromarray(frame))
            self.canvas.itemconfigure(self.image_item, image=self.photo)
            for origin, title in ((0, "CCF 2020"), (pane_w + PANE_GAP, Path(self.loaded_atlas_path).name)):
                self.canvas.create_text(origin + 8, 8, text=title, fill='#d0d0d0', anchor='nw', tags='label',
                                        font=('Segoe UI', 10, 'bold'))
                self.draw_edge_labels(origin, pane_w, height)
            return
        if mode in DIFF_MODES:   # the standard atlas is still loading: just the dimmed alternate
            mode = 'Regions (dim)'
        frame = render_frame(labels, self.lut_view, mode, self.view, (width, height), layers, radius, overlay,
                             None, self.outline_style(mode))
        self.photo = ImageTk.PhotoImage(Image.fromarray(frame))
        self.canvas.itemconfigure(self.image_item, image=self.photo)
        self.draw_edge_labels(0, width, height)

    # ---- bregma
    def bregma_mm(self, ap_um, dv_um, ml_um):
        """(AP, DV, ML) in mm relative to bregma from CCF positions in um: AP is positive anterior of bregma, DV positive
        dorsal (so negative below it), ML the distance from bregma's midline, positive toward the left hemisphere (the
        side the info window already calls left)."""
        return (self.bregma[0] - ap_um / 1000.0, self.bregma[1] - dv_um / 1000.0, ml_um / 1000.0 - self.bregma[2])

    def bregma_text(self, ap_um, dv_um, ml_um):
        if not self.on_ccf_grid:
            return "Bregma: defined for the CCF only"
        ap, dv, ml = self.bregma_mm(ap_um, dv_um, ml_um)
        return f"Bregma mm: AP {ap:+.3f}   DV {dv:+.3f}   ML {abs(ml):.3f} {'left' if ml >= 0 else 'right'}"

    def slice_bregma_text(self):
        """The current slice's position relative to bregma along its own axis, e.g. 'bregma -1.250 mm'."""
        if not self.on_ccf_grid:
            return "no bregma"
        value = self.slice_um / 1000.0
        relative = (self.bregma[0] - value, self.bregma[1] - value, value - self.bregma[2])[self.axis]
        return f"bregma {relative:+.3f} mm"

    # ---- region tool tips
    def tip_text(self, label, table):
        """Region number and name for a tool tip."""
        if label == 0:
            return "0\noutside the brain"
        if table is None or label not in table.index:
            return f"{label}"
        row = table.loc[label]
        number = str(int(row['region_id'])) if 'region_id' in row.index and pd.notna(row['region_id']) else str(label)
        acronym, name = row.get('acronym_substructure'), row.get('name_substructure')
        head = number + (f"  {acronym}" if isinstance(acronym, str) and acronym else '')
        return head + (f"\n{name}" if isinstance(name, str) and name else '')

    def hide_tip(self):
        if self.tip_job is not None:
            self.root.after_cancel(self.tip_job)
            self.tip_job = None
        self.canvas.delete('tip')

    def schedule_tip(self):
        if self.tip_job is None and self.mouse is not None and self.tool == 'none':
            self.tip_job = self.root.after(TIP_DELAY_MS, self.show_tip)

    def voxel_under_mouse(self):
        """(AP, DV, ML) voxel index under the pointer, or None when it is outside the image."""
        if self.mouse is None or self.volume is None:
            return None
        u, v = self.cursor_image_coords()
        image_w, image_h = VIEW_IMAGE_SIZE[self.view_name]
        u_idx, v_idx = int(np.floor(u)), int(np.floor(v))
        if not (0 <= u_idx < image_w and 0 <= v_idx < image_h):
            return None
        across, down = VIEW_CELL_AXES[self.view_name]
        voxel = [0, 0, 0]
        voxel[self.axis], voxel[across], voxel[down] = self.slice, u_idx, v_idx
        return tuple(voxel)

    def draw_tip(self, x, y, origin, pane_width, text, ring=False):
        """A note with `text` beside canvas point (x, y), kept inside the pane [origin, origin + pane_width)."""
        if ring:   # marks the same place on the other map
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline='#ffe000', width=2, tags='tip')
        item = self.canvas.create_text(x + 14, y + 18, text=text, anchor='nw', fill='black', tags='tip',
                                       font=('Segoe UI', 10))
        x0, y0, x1, y1 = self.canvas.bbox(item)
        pad = 4
        dx = min(0, origin + pane_width - 2 - (x1 + pad)) if x1 + pad > origin + pane_width - 2 else 0
        dy = min(0, self.canvas.winfo_height() - 2 - (y1 + pad)) if y1 + pad > self.canvas.winfo_height() - 2 else 0
        if dy:   # no room below: put it above the pointer
            dy = -(y1 - y0) - 36
        self.canvas.move(item, dx, dy)
        x0, y0, x1, y1 = self.canvas.bbox(item)
        box = self.canvas.create_rectangle(x0 - pad, y0 - pad, x1 + pad, y1 + pad, fill='#ffffe0', outline='#555555',
                                           tags='tip')
        self.canvas.tag_raise(item, box)

    def show_tip(self):
        """The first appearance, after the pointer has rested: from here on the tip follows the pointer."""
        self.tip_job = None
        self.tip_active = self.draw_tips()

    def refresh_tip(self):
        """Redraw the tip for wherever the pointer (or the picture under it) is now; nothing while it is off the image."""
        self.canvas.delete('tip')
        self.draw_tips()

    def deactivate_tip(self):
        self.tip_active = False
        self.hide_tip()

    def draw_tips(self):
        """Draw the region under the pointer (in the Differences view also the region at the same place of the other
        map). False if there is nothing to show: the pointer is off the image or outside the brain (in the Differences
        view: outside the brain in both atlases)."""
        voxel = self.voxel_under_mouse()
        if voxel is None or self.tool != 'none':
            return False
        x, y = self.mouse
        if self.pane_count() == 2 and self.reference is not None:
            standard, alternate = int(self.reference['volume'][voxel]), int(self.volume[voxel])
            if standard == 0 and alternate == 0:
                return False
            pane_w = self.pane_width()
            right_start = pane_w + PANE_GAP
            hovering_right = x >= right_start
            x_in_pane = x - right_start if hovering_right else x
            for is_right, origin, label, table in ((False, 0, standard, self.reference['table']),
                                                   (True, right_start, alternate, self.region_table)):
                self.draw_tip(origin + x_in_pane, y, origin, pane_w, self.tip_text(label, table),
                              ring=(is_right != hovering_right))
            return True
        label = int(self.volume[voxel])
        if label == 0:
            return False
        self.draw_tip(x, y, 0, self.canvas.winfo_width(), self.tip_text(label, self.region_table))
        return True

    # ---- info
    def region_text(self, label, table=None):
        table = self.region_table if table is None else table
        if label == 0:
            return "0  outside the brain"
        if table is None or label not in table.index:
            return f"{label}"
        row = table.loc[label]
        text = str(int(row['region_id'])) if 'region_id' in row.index else str(label)
        if row.get('acronym_substructure'):
            text += f"  {row['acronym_substructure']}  ({row.get('name_substructure', '')})"
        return text + (f"\n    {row['detail']}" if row.get('detail') else '')

    def refresh_info_slice(self):
        lines = [(f"{self.view_name} slice {self.slice + 1} of {self.n_slices}   "
                  f"{AXIS_NAMES[self.axis]} {self.slice_um / 1000:.3f} mm   ({self.slice_bregma_text()})", None),
                 ((f"Bregma at CCF mm AP {self.bregma[0]:.3f}, DV {self.bregma[1]:.3f}, ML {self.bregma[2]:.3f} "
                   "(change in Settings); AP + is anterior, DV + is dorsal") if self.on_ccf_grid else
                  f"Voxels of {VOXEL_UM:g} um; positions are measured from the volume's corner (AP from the front, DV "
                  "from the top, ML from the right)", 'dim')]
        if self.mode_var.get() in DIFF_MODES:
            compared = self.diff_slice()
            if compared is None:
                lines.append(("Differences from CCF 2020: loading the standard atlas to compare with…", 'dim'))
            else:
                totals = self.diff_totals()
                lines.append((f"Differences from CCF 2020 (regions compared by {self.diff_state()[2]}); left: CCF 2020, "
                              f"right: {Path(self.loaded_atlas_path).name}; regions that agree are dimmed:", None))
                lines.append((f"  different region (both have tissue)   {compared[2][0]:,} voxels in this slice, {totals[0]:,} in all", None))
                lines.append((f"  tissue only in this atlas   {compared[2][1]:,} here, {totals[1]:,} in all", None))
                lines.append((f"  tissue only in CCF 2020   {compared[2][2]:,} here, {totals[2]:,} in all", None))
        if all(s.points is None for s in self.slots):
            lines.append(("No CSV loaded: choose, paste or drop one above.", 'dim'))
        for slot in self.slots:
            points = slot.points
            if points is None:
                continue
            inside = self.slab_cells(slot)
            in_view = np.bincount(points.codes[inside], minlength=len(points.group_labels))
            lines.append((f"CSV {slot.index + 1}: {points.path.name}", None))
            lines.append((f"  slab ±{int(self.slab_var.get() or 0)}: {len(inside):,} of {len(points.ap):,} cells "
                          f"({points.kind}; by {points.group_column or 'nothing'})", None))
            for index, label in enumerate(points.group_labels):
                color, shape = slot.styles[index % len(slot.styles)]
                tag = f"s{slot.index}g{index}"
                self.info.text.tag_configure(tag, foreground=hex_color(color))
                off = '' if slot.visible_mask is None or slot.visible_mask[index] else '  (hidden)'
                lines.append((f"  {SHAPE_SYMBOLS[shape]} {label:<22} {in_view[index]:>9,} in view   "
                              f"{points.totals[index]:>10,} total{off}", tag))
        self.info.set_sections(slice_lines=lines)

    def cursor_image_coords(self):
        if self.mouse is None:
            return None
        scale, ox, oy = self.view
        return (self.pane_x(self.mouse[0]) - ox) / scale, (self.mouse[1] - oy) / scale

    def on_leave(self):
        self.mouse = None
        self.deactivate_tip()
        self.canvas.delete('brush')
        self.info.set_sections(cursor_lines=[("(cursor outside the image)", 'dim')],
                               nearest_lines=[("", 'dim')])

    def on_motion(self, event):
        if self.volume is None:
            return
        self.mouse = (event.x, event.y)
        self.draw_brush()
        if self.tip_active:
            self.refresh_tip()   # already showing: it follows the pointer and updates at once
        else:
            self.hide_tip()
            self.schedule_tip()
        u, v = self.cursor_image_coords()
        image_w, image_h = VIEW_IMAGE_SIZE[self.view_name]
        u_idx, v_idx = int(np.floor(u)), int(np.floor(v))
        cursor_lines, nearest_lines = [], []
        if 0 <= u_idx < image_w and 0 <= v_idx < image_h:
            across, down = VIEW_CELL_AXES[self.view_name]
            voxel = [0, 0, 0]
            voxel[self.axis], voxel[across], voxel[down] = self.slice, u_idx, v_idx
            position = [0.0, 0.0, 0.0]
            position[self.axis], position[across], position[down] = self.slice_um, (u - 0.5) * VOXEL_UM, \
                (v - 0.5) * VOXEL_UM
            label = int(self.volume[voxel[0], voxel[1], voxel[2]])
            side = 'left' if voxel[2] * VOXEL_UM >= MIDLINE_UM else 'right'
            cursor_lines = [
                (f"voxel (AP, DV, ML) = ({voxel[0]}, {voxel[1]}, {voxel[2]})   {side} hemisphere", None),
                (f"{'CCF' if self.on_ccf_grid else 'Atlas'} mm: AP {position[0] / 1000:.3f}   DV {position[1] / 1000:.3f}   "
                 f"ML {position[2] / 1000:.3f}", None),
                (self.bregma_text(*position), None),
                (f"Region {self.region_text(label)}", None)]
            if self.reference is not None and self.loaded_atlas_path:
                standard = int(self.reference['volume'][voxel[0], voxel[1], voxel[2]])
                state = self.diff_state()
                differs = state is not None and state[0][label] != state[1][standard]
                cursor_lines.append((f"CCF 2020: {self.region_text(standard, self.reference['table'])}"
                                     + ("   <- differs" if differs else ""), None))
            for slot in self.slots:
                nearest_lines += self.nearest_lines(slot, u, v)
        else:
            cursor_lines = [("(cursor outside the image)", 'dim')]
        self.info.set_sections(cursor_lines=cursor_lines, nearest_lines=nearest_lines or [("none within reach", 'dim')])

    def nearest_lines(self, slot, u, v):
        """Info rows for the slot's cell in view nearest the cursor, if any is within reach."""
        shown = self.shown_cells(slot)
        if not len(shown):
            return []
        points = slot.points
        scale = self.view[0]
        across, down = VIEW_CELL_AXES[self.view_name]
        du = (points.coords[across][shown] / VOXEL_UM + 0.5 - u) * scale
        dv = (points.coords[down][shown] / VOXEL_UM + 0.5 - v) * scale
        distance = du * du + dv * dv
        best = int(np.argmin(distance))
        reach = max(HIT_RADIUS_PX, int(self.dot_var.get() or 1) + 3)
        if distance[best] > reach * reach:
            return []
        index = int(shown[best])
        group = int(points.codes[index])
        color, shape = slot.styles[group % len(slot.styles)]
        tag = f"s{slot.index}g{group}"
        close = int((distance <= reach * reach).sum())
        lines = [(f"CSV {slot.index + 1}  {SHAPE_SYMBOLS[shape]} {points.group_labels[group]}   "
                  f"({close} cell{'s' if close != 1 else ''} within {reach} px)", tag),
                 (f"plane offset {(points.coords[self.axis][index] - self.slice_um):+.0f} um", None),
                 (f"  {self.bregma_text(points.coords[0][index], points.coords[1][index], points.coords[2][index])}", None)]
        lines += [(f"  {key}: {value}", None) for key, value in points.describe(index)]
        return lines

    # ---- lifecycle
    def close(self):
        if self.edit_version != self.saved_version and not messagebox.askyesno(
                "Unsaved changes", f"The volume has {self.edit_count:,} edited voxels that have not been saved "
                "(Paint panel > Save volume as).{0}{0}Close without saving?".format(chr(10)), icon='warning', default='no',
                parent=self.root):
            return
        self.settings['bregma_mm'] = list(self.bregma)
        self.settings['atlas_flips'] = {k: list(v) for k, v in self.atlas_flips.items()}
        self.settings.update({'follow_info': bool(self.follow_var.get()), 'wheel_zoom': bool(self.wheel_zoom_var.get()), 'atlas_path': self.atlas_path,
                              'alt_atlas_path': self.alt_atlas_path,
                              'brush_radius': self.brush_radius() + 1, 'paint_value': self.paint_value_var.get().strip(), 'view': self.view_name, 'slices': self.slices,
                              'saturation': int(round(float(self.sat_var.get()))), 'slab': int(self.slab_var.get() or 0),
                              'dot': int(self.dot_var.get() or 1), 'mode': self.mode_var.get(),
                              'outlines': bool(self.outline_var.get())})
        self.settings['recent_csv'] = self.recent_csv
        for slot in self.slots:
            self.settings[f'csv{slot.index}'] = str(slot.points.path) if slot.points is not None else ''
        self.settings.pop('csv', None)
        self.settings.pop('brush', None)
        self.settings.pop('slice', None)
        save_settings(self.settings)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    CCFBrowser().run()


if __name__ == '__main__':
    main()
