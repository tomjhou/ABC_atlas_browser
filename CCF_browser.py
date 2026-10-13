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

Run:  python CCF_browser.py
Mouse: wheel = next/previous slice (shift: 10 slices); ctrl+wheel or +/- = zoom; drag = pan; 0 = fit to window.
"""
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

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
VIEWS = ('Coronal', 'Sagittal', 'Horizontal')
VIEW_AXIS = {'Coronal': 0, 'Sagittal': 2, 'Horizontal': 1}        # volume axis [AP, DV, ML] stepped through
VIEW_CELL_AXES = {'Coronal': (2, 1), 'Sagittal': (0, 1), 'Horizontal': (2, 0)}   # axes drawn across (u) and down (v)
VIEW_IMAGE_SIZE = {'Coronal': (N_ML, N_DV), 'Sagittal': (N_AP, N_DV), 'Horizontal': (N_ML, N_AP)}   # (w, h) voxels
VIEW_EDGE_LABELS = {'Coronal': ('dorsal', 'ventral', 'right', 'left'),
                    'Sagittal': ('dorsal', 'ventral', 'anterior', 'posterior'),
                    'Horizontal': ('anterior', 'posterior', 'right', 'left')}   # top, bottom, left, right
AXIS_NAMES = ('AP', 'DV', 'ML')


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
def read_nifti_volume(path):
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
    return np.frombuffer(raw, dtype=dtypes[code], offset=offset, count=count).reshape(shape[::-1]).T


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
        from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
        cache = AbcProjectCache.from_cache_dir(base)
        path = Path(cache.get_data_path(directory=VOLUME_DIRECTORY, file_name=VOLUME_FILE_NAME))
    volume = np.ascontiguousarray(read_nifti_volume(path))
    if volume.shape != (N_AP, N_DV, N_ML):
        raise ValueError(f"unexpected annotation volume shape {volume.shape}")

    table = None
    try:
        from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
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

    return volume, table, region_colors(volume, table)


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


def read_nrrd_volume(path):
    """The voxel array of a NRRD file (raw or gzip encoded) as [slowest axis, ..., fastest axis], without pynrrd."""
    import zlib
    with open(path, 'rb') as handle:
        raw = handle.read()
    end = raw.find(b'\n\n')
    if not raw.startswith(b'NRRD') or end < 0:
        raise ValueError("not a NRRD file")
    fields = {}
    for line in raw[:end].decode('latin-1').splitlines()[1:]:
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
    return np.frombuffer(body, dtype=dtype, count=int(np.prod(sizes))).reshape(sizes[::-1])


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


def write_volume(path, values, progress=None):
    """Write a [AP, DV, ML] array of uint16/uint32 region values as .nrrd, .nii or .nii.gz (chosen by the name)."""
    import gzip
    import zlib
    fmt = volume_format(path)
    if progress:
        progress("compressing" if fmt != '.nii' else "writing")
    if fmt == '.nrrd':
        kind = {'uint16': 'unsigned short', 'uint32': 'unsigned int'}[values.dtype.name]
        header = (f"NRRD0004\n# Written by CCF_browser.py\ntype: {kind}\ndimension: 3\nspace: left-posterior-superior\n"
                  f"sizes: {values.shape[2]} {values.shape[1]} {values.shape[0]}\n"
                  f"space directions: ({VOXEL_UM:g},0,0) (0,{VOXEL_UM:g},0) (0,0,{VOXEL_UM:g})\n"
                  "kinds: domain domain domain\nendian: little\nencoding: gzip\nspace origin: (0,0,0)\n\n")
        compressor = zlib.compressobj(6, zlib.DEFLATED, 31)
        with open(path, 'wb') as handle:
            handle.write(header.encode('ascii'))
            for start in range(0, values.shape[0], 16):   # in slabs, to keep the temporary copies small
                handle.write(compressor.compress(np.ascontiguousarray(values[start:start + 16]).tobytes()))
            handle.write(compressor.flush())
    elif fmt in ('.nii', '.nii.gz'):
        opener = gzip.open(path, 'wb', compresslevel=5) if fmt == '.nii.gz' else open(path, 'wb')
        with opener as handle:
            handle.write(nifti_header(values.shape, values.dtype))
            transposed = values.T   # NIfTI stores the first axis (AP) fastest
            for start in range(0, transposed.shape[0], 16):
                handle.write(np.ascontiguousarray(transposed[start:start + 16]).tobytes())
    else:
        raise ValueError("the file name must end in .nrrd, .nii or .nii.gz")


def read_ontology(path, top):
    """A region table from an Allen ontology CSV (id, name, acronym, color_hex_triplet, structure_id_path), for the ids
    up to `top`: same columns as the ABC table, with the lineage of acronyms as the detail line."""
    frame = pd.read_csv(path)
    acronyms = frame.drop_duplicates('id').set_index('id')['acronym']
    frame = frame[frame['id'] <= top].drop_duplicates('id').set_index('id')

    def lineage(id_path):
        ids = [int(part) for part in str(id_path).split('/') if part.strip().isdigit()]
        return ' > '.join(str(acronyms.get(i, i)) for i in ids if i not in (997, 8))

    return pd.DataFrame({'acronym_substructure': frame['acronym'], 'name_substructure': frame['name'],
                         'detail': 'lineage: ' + frame['structure_id_path'].map(lineage),
                         'substructure_color': '#' + frame['color_hex_triplet'].astype(str).str.zfill(6)})


def load_custom_atlas(path):
    """(volume [AP, DV, ML], region table or None, colour lookup) from a .nrrd / .nii / .nii.gz annotation volume on
    the same 25 um grid, with names and colours from an ontology*.csv beside it when there is one."""
    path = Path(path)
    volume = read_nrrd_volume(path) if path.suffix.lower() == '.nrrd' else read_nifti_volume(path)
    if sorted(volume.shape) != sorted(VOLUME_SHAPE):
        raise ValueError(f"expected a {N_AP} x {N_DV} x {N_ML} volume (25 um), got {volume.shape}")
    volume = np.transpose(volume, [volume.shape.index(n) for n in VOLUME_SHAPE])   # to [AP, DV, ML]
    if volume.dtype.kind == 'f':
        if not np.array_equal(volume, np.round(volume)):
            raise ValueError("this is an intensity image, not an annotation volume (region labels are whole numbers)")
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
    return volume, table, region_colors(volume, table)


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


def read_neuroinfo_markers(path, progress=None):
    """The marker export of MBF NeuroInfo: two header lines, then one row per detection with its experiment position,
    channel intensities, atlas position (um: x left-right, y dorsal-ventral, z anterior-posterior) and region. A region
    name containing a comma makes a row one field longer, which is repaired here rather than dropped."""
    names = ['Marker', 'ex', 'ey', 'ez', 'strength', 'DAPI', 'EGFP', 'AF594', 'ml_um', 'dv_um', 'ap_um',
             'c11', 'c12', 'c13', 'c14', 'c15']
    wanted = ['Marker', 'strength', 'DAPI', 'EGFP', 'AF594', 'ml_um', 'dv_um', 'ap_um', 'c11', 'c12', 'c13', 'c14', 'c15']
    kinds = {'Marker': 'category', 'strength': 'float32', 'DAPI': 'float32', 'EGFP': 'float32', 'AF594': 'float32',
             'ml_um': 'float32', 'dv_um': 'float32', 'ap_um': 'float32'}
    kinds.update({c: 'category' for c in ('c11', 'c12', 'c13', 'c14', 'c15')})
    from pandas.api.types import union_categoricals
    size = max(1, Path(path).stat().st_size)
    pieces = []
    import io
    dummy = b',' * (len(names) - 1) + bytes([10])   # a full-width first row, so every block is parsed as 16 fields wide
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
    long_rows = frame['c15'].notna().to_numpy()  # name had a comma: name, name-rest, acronym, id, hemisphere

    def repaired(normal, long_form):
        column = normal.copy()
        if long_rows.any():
            fixed = long_form[long_rows].astype(str)
            column = column.cat.add_categories([v for v in fixed.unique() if v not in column.cat.categories])
            column[long_rows] = fixed.to_numpy()
        return column

    name = repaired(frame['c11'], frame['c11'].astype(str) + ',' + frame['c12'].astype(str)) if long_rows.any() \
        else frame['c11']
    acronym = repaired(frame['c12'], frame['c13']) if long_rows.any() else frame['c12']
    hemisphere = repaired(frame['c14'], frame['c15']) if long_rows.any() else frame['c14']
    region_id = categorical_to_float(frame['c13'])
    if long_rows.any():
        region_id[long_rows] = categorical_to_float(frame['c14'])[long_rows]
    table = pd.DataFrame({'Marker': frame['Marker'], 'strength': frame['strength'], 'DAPI': frame['DAPI'],
                          'EGFP': frame['EGFP'], 'AF594': frame['AF594'], 'region': name, 'acronym': acronym,
                          'region_id': region_id, 'hemisphere': hemisphere})
    skipped = max(0, count_lines(path) - 2 - len(frame))
    return {'table': table, 'ap': frame['ap_um'].to_numpy(np.float32), 'dv': frame['dv_um'].to_numpy(np.float32),
            'ml': frame['ml_um'].to_numpy(np.float32), 'default_group': 'Marker', 'kind': 'NeuroInfo markers',
            'skipped': skipped}


def read_points_csv(path, progress=None):
    """The cells of a CSV as {'table', 'ap', 'dv', 'ml' (um), 'default_group', 'kind', 'skipped'}; raises ValueError
    when the layout is not one of the two recognised."""
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


def render_frame(labels, lut, mode, view, size, layers=(), radius=3, overlay=None):
    """The picture of one slice as an (H, W, 3) uint8 array. `view` = (scale, ox, oy) maps image coordinates (voxel
    edges) to canvas pixels as ox + u * scale. `layers` = [(u_um, v_um, group_codes, styles)], one
    per CSV, of the cells to stamp on it (later layers on top); `styles` = [(colour, shape)] per group."""
    width, height = size
    scale, ox, oy = view
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = BACKGROUND
    rgb = atlas_rgb(labels, lut, mode)
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
        self.slots = [Slot(i) for i in range(self.SLOTS)]
        self.photo = None
        self.view_name = self.settings.get('view', VIEWS[0]) if self.settings.get('view') in VIEWS else VIEWS[0]
        saved_slices = self.settings.get('slices', {})
        self.slices = {name: int(saved_slices.get(name, VOLUME_SHAPE[VIEW_AXIS[name]] // 2)) for name in VIEWS}
        self.lut_view = None           # the region colours at the chosen saturation
        self.follow_var = tk.BooleanVar(value=bool(self.settings.get('follow_info', True)))
        self.atlas_path = self.settings.get('atlas_path', '')   # '' = the CCF from the ABC atlas cache
        self.alt_atlas_path = self.settings.get('alt_atlas_path', '') or self.atlas_path   # remembered for switching back
        self.loaded_atlas_path = ''    # the atlas now on screen
        self.settings_status = tk.StringVar()
        self.atlas_loaded_once = False
        self.last_root_pos = None
        self.zoom_goal = None          # scale the smooth zoom is heading for, None when not zooming
        self.zoom_anchor = None        # (canvas x, canvas y, image u, image v) kept fixed during the zoom
        self.wheel_zoom_var = tk.BooleanVar(value=bool(self.settings.get('wheel_zoom', True)))
        self.settings_window = None
        self.label_ids = None          # real ids of a remapped atlas (volume holds 0..n-1), else None
        self.overlay = None            # bool volume of the red painted voxels, made on first use
        self.overlay_count = 0
        self.edit_count = 0            # voxels changed by Apply since the atlas was loaded
        self.undo_stack = []           # ('stroke', flat voxel indices, drawn?) / ('apply', flat indices, old values)
        self.tool = 'none'             # 'draw', 'erase' or 'none' (drag pans)
        self.stroke = None             # flat voxel indices changed by the stroke in progress
        self.last_paint = None
        self.value_cache = None        # voxels per label, until the volume changes
        self.paint_window = None
        self.paint_buttons = {}
        self.paint_value_var = tk.StringVar(value=self.settings.get('paint_value', ''))
        self.paint_warning = tk.StringVar()
        self.paint_status = tk.StringVar()
        self.brush_var = tk.IntVar(value=int(self.settings.get('brush', 5)))
        self.view = [1.0, 0.0, 0.0]   # scale, ox, oy
        self.fit = True
        self.drag = None
        self.mouse = None              # last cursor position over the canvas
        self.render_pending = False
        self.work = queue.Queue()
        self.busy_label, self.progress_text = '', ''
        self.autoload = []             # (slot, path) from the last session, loaded one after another
        self.build_ui()
        self.info = InfoWindow(self)
        self.refresh_info_slice()
        self.root.after(150, self.info.place_beside)
        self.root.bind('<Configure>', self.on_root_configure)
        self.root.after(100, self.poll_work)
        self.root.after(200, self.start_atlas)
        self.root.protocol('WM_DELETE_WINDOW', self.close)

    # ---- layout
    def build_ui(self):
        root = self.root
        self.slot_widgets = []
        for slot in self.slots:
            saved = self.settings.get(f'csv{slot.index}', self.settings.get('csv', '') if slot.index == 0 else '')
            slot.path_var.set(saved)
            block = ttk.Frame(root, padding=(6, 4 if slot.index == 0 else 0, 6, 0))
            block.pack(fill='x')
            row = ttk.Frame(block)
            row.pack(fill='x')
            ttk.Label(row, text=f"CSV {slot.index + 1}:").pack(side='left')
            entry = ttk.Entry(row, textvariable=slot.path_var)
            entry.pack(side='left', fill='x', expand=True, padx=4)
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
        mode_box = ttk.Combobox(controls, textvariable=self.mode_var, state='readonly', width=14, values=ATLAS_MODES)
        mode_box.pack(side='left', padx=3)
        mode_box.bind('<<ComboboxSelected>>', lambda e: self.refresh_all())
        ttk.Button(controls, text="Info window", command=lambda: self.info.show()).pack(side='left', padx=(10, 0))
        ttk.Button(controls, text="Paint panel", command=self.open_paint).pack(side='left', padx=(4, 0))
        for variable in (self.slab_var, self.dot_var):
            variable.trace_add('write', lambda *a: self.refresh_all())
        self.message = ttk.Label(controls, text="", foreground='#7a3b00')
        self.message.pack(side='right')

        bottom = ttk.Frame(root, padding=(6, 2, 6, 6))
        bottom.pack(side='bottom', fill='x')
        self.slice_label = ttk.Label(bottom, text="", width=34)
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
    def paint_to(self, x, y):
        """Draw or erase the brush disk (and the stretch since the last mouse position) on the overlay."""
        scale, ox, oy = self.view
        u, v = int(np.floor((x - ox) / scale)), int(np.floor((y - oy) / scale))
        previous, self.last_paint = self.last_paint, (u, v)
        if previous is None:
            centers = np.array([[u, v]])
        else:
            steps = max(1, abs(u - previous[0]), abs(v - previous[1]))
            fractions = np.linspace(0, 1, steps + 1)[:, None]
            centers = np.rint(np.array(previous) + (np.array([u, v]) - np.array(previous)) * fractions).astype(np.int64)
        dx, dy = disk_offsets(self.brush_var.get() or 0)
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
        scale, ox, oy = self.view
        u_idx, v_idx = int(np.floor((event.x - ox) / scale)), int(np.floor((event.y - oy) / scale))
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

    def draw_brush(self):
        """The brush outline following the cursor while a paint tool is on."""
        self.canvas.delete('brush')
        if self.tool == 'none' or self.mouse is None:
            return
        radius = ((self.brush_var.get() or 0) + 0.5) * self.view[0]
        x, y = self.mouse
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline='#ff4040', tags='brush')

    def set_tool(self, tool):
        """Switch the paint tool on (draw / erase), or off (the same tool again, or 'none'): dragging pans when off."""
        self.tool = 'none' if tool == self.tool else tool
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
        ttk.Spinbox(grid, from_=0, to=100, width=6, textvariable=self.brush_var).grid(row=0, column=1, padx=(8, 0),
                                                                                    sticky='w')
        ttk.Label(grid, text="Pixel value").grid(row=1, column=0, sticky='w', pady=(6, 0))
        entry = ttk.Entry(grid, textvariable=self.paint_value_var, width=14)
        entry.grid(row=1, column=1, padx=(8, 0), pady=(6, 0), sticky='w')
        if not self.paint_value_var.get().strip():   # first time: suggest the lowest unused value, else keep the last
            self.paint_value_var.set(str(lowest_unused(self.used_values())))
        self.paint_value_var.trace_add('write', lambda *a: self.update_paint_status())
        ttk.Label(body, textvariable=self.paint_warning, foreground='#b00020', wraplength=320,
                  justify='left').pack(anchor='w', pady=(6, 0))
        ttk.Label(body, textvariable=self.paint_status, foreground='#555555').pack(anchor='w', pady=(6, 0))
        buttons = ttk.Frame(body)
        buttons.pack(fill='x', pady=(10, 0))
        ttk.Button(buttons, text="Close", command=self.close_paint).pack(side='right')
        ttk.Button(buttons, text="Apply", command=self.apply_paint).pack(side='right', padx=(0, 6))
        ttk.Button(buttons, text="Save volume…", command=self.save_volume).pack(side='left')
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
        order = [default] + [ext for ext in SAVE_FORMATS if ext != default]
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
        ids = self.label_ids
        if ids is not None:
            values = ids.astype(np.uint32 if ids.max() > 65535 else np.uint16)[self.volume]
        else:
            values = self.volume.copy()
        if values.dtype != np.uint32 and values.max() > 65535:
            values = values.astype(np.uint32)

        def done(result, error):
            if error is not None:
                messagebox.showerror("Save volume", f"Could not save the volume:\n{error}", parent=parent)
                self.paint_status.set("Save failed.")
                return
            self.paint_status.set(f"Saved {path.name}")
            self.message.configure(text=f"Saved {path}")
        self.run_in_thread(f"Saving {path.name}…", lambda: write_volume(path, values, self.set_progress), done)

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
        old = self.volume.reshape(-1)[flat].copy()
        self.volume.reshape(-1)[flat] = label
        self.overlay.reshape(-1)[flat] = False
        self.undo_stack.append(('apply', flat, old))
        self.overlay_count = 0
        self.edit_count += len(flat)
        self.value_cache = None
        self.apply_saturation()
        self.refresh_all()
        self.update_paint_status()
        self.paint_status.set(f"Applied value {value} to {len(flat):,} voxels.")

    def undo(self):
        if not self.undo_stack:
            self.paint_status.set("Nothing to undo.")
            return
        kind, flat, data = self.undo_stack.pop()
        overlay = self.overlay.reshape(-1)
        if kind == 'stroke':
            overlay[flat] = not data
            self.overlay_count += -len(flat) if data else len(flat)
        else:
            self.volume.reshape(-1)[flat] = data
            overlay[flat] = True
            self.overlay_count += len(flat)
            self.edit_count -= len(flat)
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
        ttk.Checkbutton(body, text="Mouse wheel zooms (Ctrl+wheel steps through slices; off: the other way round)",
                        variable=self.wheel_zoom_var).pack(anchor='w', pady=(4, 0))

        box = ttk.LabelFrame(body, text="Atlas", padding=8)
        box.pack(fill='x', pady=(12, 0))
        mode = tk.StringVar(value='custom' if self.atlas_path else 'abc')
        path_var = tk.StringVar(value=self.alt_atlas_path)
        self.settings_status.set('')
        ttk.Radiobutton(box, text="Allen CCF 2020, from the ABC atlas cache (default)", value='abc',
                        variable=mode).pack(anchor='w')
        ttk.Radiobutton(box, text="Another annotation volume (.nrrd, .nii or .nii.gz, 25 um, same grid):",
                        value='custom', variable=mode).pack(anchor='w', pady=(6, 0))
        row = ttk.Frame(box)
        row.pack(fill='x', padx=(20, 0))
        entry = ttk.Entry(row, textvariable=path_var, width=60)
        entry.pack(side='left', fill='x', expand=True)

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
        ttk.Label(box, foreground='#555555', justify='left', wraplength=520, text=(
            "Region names and colours come from an ontology*.csv in the same folder, if there is one (the Allen2017 "
            "folder has ontology_api_2017.csv). Label numbers differ between the 2017 and ABC atlases.")).pack(
            anchor='w', pady=(6, 0))
        status = ttk.Label(body, textvariable=self.settings_status, foreground='#7a3b00')
        status.pack(anchor='w', pady=(8, 0))

        def apply_atlas():
            wanted = path_var.get().strip().strip('"') if mode.get() == 'custom' else ''
            if mode.get() == 'custom' and not Path(wanted).is_file():
                self.settings_status.set("Choose an existing atlas file first.")
                return
            if wanted == self.atlas_path:
                self.settings_status.set("That atlas is already in use.")
                return
            if self.busy_label:
                self.settings_status.set("Busy loading; try again when it finishes.")
                return
            if (self.edit_count or self.overlay_count) and not messagebox.askyesno(
                    "Painted changes", "Switching atlas discards the painted overlay and the applied edits "
                    "(they are not saved anywhere). Continue?", parent=window):
                return
            self.atlas_path = wanted
            if wanted:
                self.alt_atlas_path = wanted
            self.settings.update({'atlas_path': wanted, 'alt_atlas_path': self.alt_atlas_path})
            save_settings(self.settings)
            self.start_atlas()
            self.settings_status.set("Loading the atlas…")

        buttons = ttk.Frame(body)
        buttons.pack(fill='x', pady=(10, 0))
        ttk.Button(buttons, text="Close", command=window.destroy).pack(side='right')
        ttk.Button(buttons, text="Apply atlas", command=apply_atlas).pack(side='right', padx=(0, 6))
        x = self.root.winfo_x() + 80
        window.geometry(f"+{x}+{self.root.winfo_y() + 80}")

    # ---- background work
    def run_in_thread(self, label, function, on_done):
        """Run `function()` on a thread; `on_done(result, error)` runs on the main thread when it finishes."""
        self.busy_label, self.progress_text = label, ''
        self.message.configure(text=label)
        self.update_cursors()

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
                self.update_cursors()
                self.message.configure(text="")
                on_done(result, error)
        except queue.Empty:
            if self.busy_label:
                self.message.configure(text=f"{self.busy_label} {self.progress_text}")
        self.root.after(100, self.poll_work)

    # ---- the atlas
    def start_atlas(self):
        if self.atlas_path:
            name = Path(self.atlas_path).name
            self.run_in_thread(f"Loading {name}…", lambda: load_custom_atlas(self.atlas_path), self.atlas_ready)
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
                           lambda: load_atlas(base, download), self.atlas_ready)

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
        self.volume, self.region_table, self.lut = result
        self.loaded_atlas_path = self.atlas_path
        self.label_ids = (self.region_table['region_id'].to_numpy()
                          if self.region_table is not None and 'region_id' in self.region_table else None)
        self.overlay, self.overlay_count, self.edit_count, self.undo_stack = None, 0, 0, []
        self.value_cache, self.stroke = None, None
        self.set_tool('none')
        self.settings_status.set("Atlas loaded: " + (Path(self.atlas_path).name if self.atlas_path else "default CCF"))
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

    def apply_saturation(self):
        value = int(round(float(self.sat_var.get())))
        self.sat_label.configure(text=f"{value}%")
        if self.lut is not None:
            self.lut_view = desaturated(self.lut.astype(np.float32), value / 100.0) if value < 100 else self.lut

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
            title="Choose a CSV of cells", parent=self.root, initialdir=str(start) if start and start.is_dir() else None,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
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
                           lambda result, error: self.points_ready(slot, result, error))

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
        """The wheel zooms about the cursor (or steps through slices, shift: 10); Ctrl swaps the two."""
        if bool(event.state & 0x4) == self.wheel_zoom_var.get():   # slices
            self.step((10 if event.state & 0x1 else 1) * (-1 if event.delta > 0 else 1))
        else:
            self.zoom_at(1.2 ** (event.delta / 120.0), (event.x, event.y))

    def fit_to_window(self):
        self.zoom_goal = None
        width, height = max(50, self.canvas.winfo_width()), max(50, self.canvas.winfo_height())
        image_w, image_h = VIEW_IMAGE_SIZE[self.view_name]
        scale = min(width / image_w, height / image_h) * 0.97
        self.view = [scale, (width - image_w * scale) / 2, (height - image_h * scale) / 2]
        self.fit = True
        self.request_render()

    def zoom_at(self, factor, anchor=None):
        """Zoom by `factor` about a canvas point (default the centre), easing toward the new scale over a few frames;
        wheel notches arriving meanwhile add up."""
        scale, ox, oy = self.view
        ax, ay = anchor if anchor else (self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)
        base = self.zoom_goal if self.zoom_goal is not None else scale
        self.zoom_goal = min(max(base * factor, 0.5), 60.0)
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
        self.request_render()
        if done:
            self.zoom_goal = None
        else:
            self.root.after(16, self.animate_zoom)

    def on_press(self, event):
        self.zoom_goal = None
        if self.tool != 'none' and self.volume is not None:
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
            self.paint_to(event.x, event.y)
            self.mouse = (event.x, event.y)
            self.draw_brush()
            return
        if self.drag is None:
            return
        x0, y0, ox, oy = self.drag
        self.view[1], self.view[2] = ox + event.x - x0, oy + event.y - y0
        self.fit = False
        self.request_render()

    def on_canvas_resize(self):
        if self.fit and self.volume is not None:
            self.fit_to_window()
        else:
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
                                        f"{AXIS_NAMES[self.axis]} {self.slice_um / 1000:.3f} mm")

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

    def render(self):
        self.render_pending = False
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
        frame = render_frame(view_slice(self.volume, self.view_name, self.slice), self.lut_view, self.mode_var.get(),
                             self.view, (width, height), layers, int(self.dot_var.get() or 1), overlay)
        self.photo = ImageTk.PhotoImage(Image.fromarray(frame))
        self.canvas.itemconfigure(self.image_item, image=self.photo)
        self.canvas.delete('label')
        top, bottom, left, right = VIEW_EDGE_LABELS[self.view_name]
        for text, x, y, anchor in ((top, width / 2, 10, 'n'), (bottom, width / 2, height - 8, 's'),
                                   (left, 8, height / 2, 'w'), (right, width - 8, height / 2, 'e')):
            self.canvas.create_text(x, y, text=text, fill='#8a8a8a', anchor=anchor, tags='label', font=('Segoe UI', 9))

    # ---- info
    def region_text(self, label):
        if label == 0:
            return "0  outside the brain"
        if self.region_table is None or label not in self.region_table.index:
            return f"{label}"
        row = self.region_table.loc[label]
        text = str(int(row['region_id'])) if 'region_id' in row.index else str(label)
        if row.get('acronym_substructure'):
            text += f"  {row['acronym_substructure']}  ({row.get('name_substructure', '')})"
        return text + (f"\n    {row['detail']}" if row.get('detail') else '')

    def refresh_info_slice(self):
        lines = [(f"{self.view_name} slice {self.slice + 1} of {self.n_slices}   "
                  f"{AXIS_NAMES[self.axis]} {self.slice_um / 1000:.3f} mm", None)]
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
        return (self.mouse[0] - ox) / scale, (self.mouse[1] - oy) / scale

    def on_leave(self):
        self.mouse = None
        self.canvas.delete('brush')
        self.info.set_sections(cursor_lines=[("(cursor outside the image)", 'dim')],
                               nearest_lines=[("", 'dim')])

    def on_motion(self, event):
        if self.volume is None:
            return
        self.mouse = (event.x, event.y)
        self.draw_brush()
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
                (f"CCF mm: AP {position[0] / 1000:.3f}   DV {position[1] / 1000:.3f}   ML {position[2] / 1000:.3f}",
                 None),
                (f"Region {self.region_text(label)}", None)]
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
                 (f"plane offset {(points.coords[self.axis][index] - self.slice_um):+.0f} um", None)]
        lines += [(f"  {key}: {value}", None) for key, value in points.describe(index)]
        return lines

    # ---- lifecycle
    def close(self):
        self.settings.update({'follow_info': bool(self.follow_var.get()), 'wheel_zoom': bool(self.wheel_zoom_var.get()), 'atlas_path': self.atlas_path,
                              'alt_atlas_path': self.alt_atlas_path,
                              'brush': int(self.brush_var.get() or 0), 'paint_value': self.paint_value_var.get().strip(), 'view': self.view_name, 'slices': self.slices,
                              'saturation': int(round(float(self.sat_var.get()))), 'slab': int(self.slab_var.get() or 0),
                              'dot': int(self.dot_var.get() or 1), 'mode': self.mode_var.get()})
        for slot in self.slots:
            self.settings[f'csv{slot.index}'] = str(slot.points.path) if slot.points is not None else ''
        self.settings.pop('csv', None)
        self.settings.pop('slice', None)
        save_settings(self.settings)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    CCFBrowser().run()


if __name__ == '__main__':
    main()
