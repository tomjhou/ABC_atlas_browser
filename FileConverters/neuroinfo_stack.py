#!/usr/bin/env python3
"""
neuroinfo_stack.py

Read, edit and re-render NeuroInfo "serial section assembler" stacks
(reconstruction.jpx + reconstruction.xml + CaseFile.H5), the first-stage
output that aligns serial sections before atlas registration. jpx_aligner.py
is the GUI on top of this module.

How a NeuroInfo stack is understood (worked out from NeuroInfo 2026.1.1):
  - CaseFile.H5 lists the sections in stack order (BrainMaker/OrderedSections),
    each with its source image (ImageFile.SlideFileName) and its outline in
    SOURCE coordinates (Points, um).
  - reconstruction.xml has the same outlines (matched by contour GUID, point for
    point) in STACK coordinates (um).
  - Fitting outline -> outline gives each section's exact rigid transform
    (rotation, shift, optional mirror), with residuals of ~0.004 um. The
    transform chain stored in CaseFile.H5 is NOT used: its "Reflection2D" steps
    don't correspond to the actual mirroring.
  - Coordinates: a source pixel (col, row) is at (px*col, -px*row) um; a stack
    canvas pixel (u, v) is at (px*u, -px*v) um (y up, origin at the top-left).
  - NeuroInfo only reads the .jpx.

Edits and re-editing: each section carries an Edit (rotation, shift, flip)
applied in stack coordinates about the center of its outline, on top of
NeuroInfo's alignment (the "base" transform). Exported JPX files contain a
private 'uuid' box (ALIGNER_UUID) at the end with a JSON description of the
whole stack (sources, base transforms, outlines, edits), so an exported file
can be reopened and edited again without the .xml/.h5. NeuroInfo ignores it.

Export re-renders only sections whose edit differs from what the opened file
already shows; the others' codestreams are copied byte for byte.

    python neuroinfo_stack.py info "<reconstruction.jpx or stack folder>"
    python neuroinfo_stack.py roundtrip "<stack>" "<out.jpx>" [--compare]

Requires numpy, h5py, imagecodecs, opencv-python (installed automatically if
missing, like tifs_to_jp2.py).
"""

import argparse
import contextlib
import copy
import datetime
import importlib.util
import itertools
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict

REQUIRED_PACKAGES = {"numpy": "numpy", "h5py": "h5py", "imagecodecs": "imagecodecs",
                     "cv2": "opencv-python"}


CZI_PACKAGES = {"czifile": "czifile"}   # the Zeiss .czi reader, installed only when a stack needs it


def lower_thread_priority():
    """Make the calling thread yield to the rest of the program (Windows: below-normal priority), so that long background work
    does not make the window sluggish. No effect elsewhere."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadPriority(ctypes.windll.kernel32.GetCurrentThread(), -1)
        except Exception:
            pass


def ensure_packages(required=None, exit_on_failure=True):
    """Install any missing packages ({import name: pip name}) into this
    interpreter's environment."""
    required = required or REQUIRED_PACKAGES
    missing = [pip for mod, pip in required.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    print(f"Installing missing libraries ({', '.join(missing)}) into {sys.executable}...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--only-binary", ":all:", *missing])
    importlib.invalidate_caches()
    import site
    user_site = site.getusersitepackages()
    if site.ENABLE_USER_SITE and os.path.isdir(user_site) and user_site not in sys.path:
        site.addsitedir(user_site)
    still = [pip for mod, pip in required.items() if importlib.util.find_spec(mod) is None]
    if still:
        message = f"Couldn't install {', '.join(still)}; install them with pip and try again."
        if exit_on_failure:
            sys.exit(message)
        raise ImportError(message)


ensure_packages()

import cv2  # noqa: E402
import h5py  # noqa: E402
import imagecodecs  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_QUALITY_DB = 60.0
XMP_UUID = bytes.fromhex("be7acfcb97a942e89c71999491e3afac")
# Private box holding this tool's JSON description of the stack
JP2_EXTENSIONS = (".jp2", ".j2k", ".j2c", ".jpx")   # the section images this tool can re-render (others, e.g. Zeiss .czi, cannot)
ALIGNER_UUID = bytes.fromhex("4a686f754c61624a5058416c69676e31")  # "JhouLabJPXAlign1"
JSON_FORMAT = "jhoulab-jpx-aligner"
JSON_VERSION = 2   # the newest alignment data this version can read (2: edited outlines, with a fixed edit pivot)


class UserCanceled(Exception):
    pass


##################################################
#
#    JPEG2000 boxes
#
##################################################

@dataclass
class Box:
    kind: bytes
    offset: int      # of the box header
    length: int      # whole box, header included
    header: int      # header size (8, or 16 for XLBox)

    @property
    def payload_offset(self):
        return self.offset + self.header

    @property
    def payload_length(self):
        return self.length - self.header


def read_boxes(path):
    """Top-level boxes of a JP2/JPX file."""
    boxes = []
    with open(path, "rb") as f:
        f.seek(0, 2)
        end = f.tell()
        pos = 0
        while pos + 8 <= end:
            f.seek(pos)
            length, kind = struct.unpack(">I4s", f.read(8))
            header = 8
            if length == 1:
                length = struct.unpack(">Q", f.read(8))[0]
                header = 16
            elif length == 0:
                length = end - pos
            boxes.append(Box(kind, pos, length, header))
            pos += length
    return boxes


def read_payload(f, box, n=None):
    f.seek(box.payload_offset)
    return f.read(box.payload_length if n is None else n)


def box_bytes(kind, payload):
    if 8 + len(payload) < 2 ** 32:
        return struct.pack(">I4s", 8 + len(payload), kind) + payload
    return struct.pack(">I4sQ", 1, kind, 16 + len(payload)) + payload


def add_com_markers(codestream, comments):
    """Insert COM marker segments just before the first tile-part (SOT)."""
    pos = 2
    while struct.unpack(">H", codestream[pos:pos + 2])[0] != 0xFF90:
        pos += 2 + struct.unpack(">H", codestream[pos + 2:pos + 4])[0]
    coms = b"".join(struct.pack(">HHH", 0xFF64, 4 + len(c), 1) + c.encode("latin-1")
                    for c in comments)
    return codestream[:pos] + coms + codestream[pos:]


def codestream_size(f, box):
    """(width, height, components) from a jp2c box's SIZ marker."""
    head = read_payload(f, box, 48)
    xs, ys, xo, yo = struct.unpack(">IIII", head[8:24])
    nc = struct.unpack(">H", head[40:42])[0]
    return xs - xo, ys - yo, nc


def is_online_only(path):
    """True for a OneDrive/SharePoint placeholder not stored on this computer
    (opening it would download all of it). Checking doesn't download."""
    try:
        return bool(getattr(os.stat(path), "st_file_attributes", 0) & (0x00400000 | 0x00040000 | 0x00001000))
    except OSError:
        return False


##################################################
#
#    Stack model
#
##################################################

def rot(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


FLIP_X = np.diag([-1.0, 1.0])


@dataclass
class Edit:
    """Change applied in stack coordinates (um, y up) about the section's
    outline center c:  p -> c + Rot(rotation) * F^flip * (p - c) + (dx, dy),
    with F mirroring left-right. Positive rotation is counterclockwise."""
    rotation_deg: float = 0.0
    dx_um: float = 0.0
    dy_um: float = 0.0
    flip: bool = False

    def __post_init__(self):
        self.rotation_deg, self.dx_um, self.dy_um = float(self.rotation_deg), float(self.dx_um), float(self.dy_um)
        self.flip = bool(self.flip)

    def matrix(self):
        return rot(self.rotation_deg) @ (FLIP_X if self.flip else np.eye(2))

    def is_identity(self):
        return self == Edit()

    def same_as(self, other, tol=1e-9):
        return (self.flip == other.flip and abs(self.rotation_deg - other.rotation_deg) < tol
                and abs(self.dx_um - other.dx_um) < tol and abs(self.dy_um - other.dy_um) < tol)


@dataclass
class Section:
    index: int               # position in the stack
    name: str                # source file name, e.g. "section 5.jp2"
    source_path: str         # resolved path on this computer ("" if not found)
    case_points: np.ndarray  # outline in source coordinates (um)
    R: np.ndarray            # base transform, source um -> stack um (det -1 = mirrored)
    t: np.ndarray
    edit: Edit = field(default_factory=Edit)
    saved_edit: Edit = field(default_factory=Edit)  # what the opened file's frame shows
    source_abs_hint: str = ""                        # path recorded in the file
    pivot_src: np.ndarray = None   # the edit pivot in source um (fixed when first opened, so editing the outline does not move it)
    saved_points: np.ndarray = None  # the outline as the opened file has it (to tell whether it was edited)
    czi_origin: tuple = None   # for a Zeiss .czi source: where its source pixel (0, 0) is in the slide's pixel grid (see calibrate_czi)
    czi_assumed: bool = False  # that origin is the one its scan's other sections share, not yet checked against the section itself
    raw_points: np.ndarray = None   # the outline exactly as the case file (CaseFile.H5) stores it (um); None for stacks without one
    xml_points: np.ndarray = None   # the outline exactly as reconstruction.xml has it (the stack's coordinates, um); None without one
    frame: int = -1            # which frame of the opened JPX is this section's (index is its position now, which the user can change)

    def __post_init__(self):
        if self.frame < 0:
            self.frame = self.index
        self.case_points = np.asarray(self.case_points, float).reshape(-1, 2)
        if self.pivot_src is None:
            self.pivot_src = self.case_points.mean(0) if len(self.case_points) else np.zeros(2)
        self.pivot_src = np.asarray(self.pivot_src, float)
        if self.saved_points is None:
            self.saved_points = self.case_points.copy()

    @property
    def base_stack_points(self):
        return (self.R @ self.case_points.T).T + self.t

    @property
    def renderable(self):
        """The source image was found and is one this tool can read, so the section can be re-rendered (and its alignment
        edited). A .czi source counts once its position on the slide is known (czi_origin, found by calibrate_czi). Otherwise the
        section only has its frame in the JPX, which is shown as it is; its outline can still be edited."""
        path = self.source_path.lower()
        return bool(path) and (path.endswith(JP2_EXTENSIONS) or (path.endswith(".czi") and self.czi_origin is not None))

    @property
    def center(self):
        """Edit pivot: the centroid of the outline as first opened, under the base transform (not recomputed when the outline
        is edited, or the image would jump)."""
        return self.R @ self.pivot_src + self.t

    def effective(self):
        """(R, t) including the edit: source um -> stack um."""
        E = self.edit.matrix()
        c = self.center
        d = np.array([self.edit.dx_um, self.edit.dy_um])
        return E @ self.R, c + E @ (self.t - c) + d

    @property
    def stack_points(self):
        R, t = self.effective()
        return (R @ self.case_points.T).T + t

    @property
    def mirrored(self):
        return np.linalg.det(self.effective()[0]) < 0

    @property
    def rotation_deg(self):
        R = self.effective()[0]
        return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    @property
    def dirty(self):
        """The placement was edited: the image has to be rendered again."""
        return not self.edit.same_as(self.saved_edit)

    @property
    def outline_dirty(self):
        """The outline was edited (vertices moved, added or removed, or the whole outline deleted)."""
        return not (self.case_points.shape == self.saved_points.shape and np.array_equal(self.case_points, self.saved_points))

    @property
    def changed(self):
        return self.dirty or self.outline_dirty


@dataclass
class ChannelInfo:
    name: str
    color: tuple             # display color (r, g, b) 0-255
    enabled: bool = True


@dataclass
class Stack:
    jpx_path: str            # the opened file (also the template for export)
    canvas_w: int
    canvas_h: int
    channels: int
    pixel_um: float
    sections: list = field(default_factory=list)
    channel_info: list = field(default_factory=list)
    origin: str = ""         # the original NeuroInfo jpx this descends from
    from_aligner: bool = False
    source_pixel_um: float = 0.0     # the source images' pixel size, when it differs from pixel_um (the stack's label)
    pixel_corrected: bool = False    # pixel_um was changed here: the next export writes it into the JPX's own metadata
    deleted: int = 0                 # how many sections were deleted since opening
    case_file: str = ""              # the NeuroInfo case file (.h5) the alignment was recovered from, when it was
    unsaved: bool = False            # made in memory from slide scans (no JPX behind it yet): everything is still to be exported

    def apply_pixel_size(self, new_um):
        """Relabel the stack with pixel size `new_um` (the source images' true pixel size). Every length in um follows: the
        outlines, the pivots, the base transforms' shifts and the edits' shifts are multiplied by new/old, so everything stays
        where it is in pixels. The next export writes the new size into the JPX's metadata."""
        f = float(new_um) / self.pixel_um
        for sec in self.sections:
            sec.case_points = sec.case_points * f
            sec.saved_points = sec.saved_points * f
            sec.pivot_src = sec.pivot_src * f
            sec.t = sec.t * f
            for name in ("edit", "saved_edit"):
                e = getattr(sec, name)
                setattr(sec, name, Edit(e.rotation_deg, e.dx_um * f, e.dy_um * f, e.flip))
        self.pixel_um = float(new_um)
        self.source_pixel_um = 0.0
        self.pixel_corrected = True

    @property
    def folder(self):
        if not self.jpx_path:      # (made in memory from slide scans: beside the first of them)
            first = next((s.source_path for s in self.sections if s.source_path), "")
            return os.path.dirname(os.path.abspath(first)) if first else os.getcwd()
        return os.path.dirname(os.path.abspath(self.jpx_path))

    @property
    def dirty(self):
        return any(s.dirty for s in self.sections)

    @property
    def reordered(self):
        """Sections were deleted or moved since opening."""
        return self.deleted > 0 or any(s.frame != s.index for s in self.sections)

    @property
    def changed(self):
        return self.unsaved or self.pixel_corrected or self.reordered or any(s.changed for s in self.sections)

    def out_of_canvas(self):
        """Sections whose outline extends past the canvas, with the overhang in um."""
        w_um, h_um = self.canvas_w * self.pixel_um, self.canvas_h * self.pixel_um
        bad = []
        for s in self.sections:
            P = s.stack_points
            if not len(P):
                continue
            over = max(-P[:, 0].min(), P[:, 0].max() - w_um, P[:, 1].max(), -h_um - P[:, 1].min(), 0)
            if over > 0:
                bad.append((s, over))
        return bad


def _s(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def fit_rigid(P, Q):
    """Least-squares Q ~ R P + t, points matched by index, mirror allowed
    (whichever fits better). Returns R, t, mean error."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    best = None
    for mirror in (False, True):
        D = np.eye(2)
        if (np.linalg.det(Vt.T @ U.T) < 0) != mirror:
            D[1, 1] = -1
        R = Vt.T @ D @ U.T
        t = qc - R @ pc
        err = float(np.sqrt((((R @ P.T).T + t - Q) ** 2).sum(1)).mean())
        if best is None or err < best[2]:
            best = (R, t, err)
    return best


def resolve_source(jpx_folder, name, relative="", absolute=""):
    """Find a section's source image on this computer."""
    candidates = []
    if relative:
        candidates.append(os.path.join(jpx_folder, relative))
    candidates += [os.path.join(jpx_folder, name), os.path.join(jpx_folder, "..", name)]
    if absolute:
        candidates.append(absolute)
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.normpath(c)
    # Not there under that name: a file whose name is the same with some extra characters before the extension (a label added
    # by hand, e.g. "...0033.czi" -> "...0033-LHb.czi") is taken for it, searched for where the exact name was looked for.
    stem, ext = os.path.splitext(name)
    folders = []
    for c in candidates:
        d = os.path.dirname(c)
        if d and d not in folders and os.path.isdir(d):
            folders.append(d)
    for d in folders:
        found = []
        for n in os.listdir(d):
            n_stem, n_ext = os.path.splitext(n)
            if n_ext.lower() == ext.lower() and len(n_stem) > len(stem) and n_stem.lower().startswith(stem.lower())                     and not n_stem[len(stem)].isdigit() and os.path.isfile(os.path.join(d, n)):   # (not "0033" for "003")
                found.append(n)
        found.sort(key=len)
        if found and (len(found) == 1 or len(found[0]) < len(found[1])):   # the one with the least added; a tie is left unresolved
            return os.path.normpath(os.path.join(d, found[0]))
    return ""


def read_xmp_channels(jpx_path):
    """Channel names, colors and visibility from the stack's MBF XMP."""
    boxes = read_boxes(jpx_path)
    with open(jpx_path, "rb") as f:
        for b in boxes:
            if b.kind == b"uuid":
                data = read_payload(f, b)
                if data[:16] == XMP_UUID:
                    xmp = data[16:].decode("utf-8", "replace")
                    break
        else:
            return []
    names = re.findall(r'xmp:GroupName="([^"]*)"', xmp)
    colors = [tuple(int(v) for v in m) for m in
              re.findall(r'<xmp:WhiteColor xmp:Red="(\d+)" xmp:Green="(\d+)" xmp:Blue="(\d+)"', xmp)]
    n = max(len(names), len(colors))
    # Shown in the editor regardless of the file's Enabled flags, which export keeps as they are
    return [ChannelInfo(names[i] if i < len(names) else f"channel {i}",
                        colors[i] if i < len(colors) else (255, 255, 255))
            for i in range(n)]


def find_aligner_json(jpx_path):
    boxes = read_boxes(jpx_path)
    with open(jpx_path, "rb") as f:
        for b in reversed(boxes):
            if b.kind == b"uuid" and b.payload_length > 16:
                data = read_payload(f, b)
                if data[:16] == ALIGNER_UUID:
                    return json.loads(data[16:].decode("utf-8"))
    return None


def _xml_guids(xml_path):
    """The contour GUIDs in a NeuroInfo reconstruction XML (upper case)."""
    with open(xml_path, encoding="latin-1") as f:
        return {g.upper() for g in re.findall(r'name="GUID"><s>([0-9A-Fa-f]+)</s>', f.read())}


def _h5_matches(h5_path, guids):
    """Whether the case file holds all of these contour GUIDs (None if it cannot be read)."""
    try:
        with h5py.File(h5_path, "r") as f:
            return guids <= {k.upper() for k in f["BrainMaker/Contours"].keys()}
    except Exception:
        return None


def _find_neuroinfo_files(jpx_path, locate=None):
    """(xml path, h5 path) of the NeuroInfo files that go with the JPX.

    The XML is <jpx name>.xml beside the JPX. Nothing assumes a name for the case file: the reconstruction's own GUIDs say which
    .h5 files fit it, and when exactly one does it is used without asking. The user is asked (and the likeliest offered first)
    when several fit, when none fits, and when there is none at all (to locate it); the same goes for the XML when no
    <jpx name>.xml is there. Asking is done through
    locate(kind, expected_name, folder, candidates, reason, default, notes) -> path or None, where kind is "xml" or "h5",
    candidates are the files of that type in the folder, reason is "missing" or "ambiguous", default is the one to offer first
    and notes maps a candidate to a remark about it. Without `locate` (or when it returns None for a missing file) the error
    names the file; without `locate`, an ambiguous choice falls back on the default."""
    folder = os.path.dirname(os.path.abspath(jpx_path))
    files = {n.lower(): n for n in os.listdir(folder)}
    jpx_stem = os.path.splitext(os.path.basename(jpx_path))[0]

    def listing(ext):
        return sorted(n for n in files.values() if n.lower().endswith(ext))

    # --- the XML
    xml_name = files.get(jpx_stem.lower() + ".xml")
    if xml_name:
        xml_path = os.path.join(folder, xml_name)
    else:
        candidates = listing(".xml")
        default = files.get("reconstruction.xml") or (candidates[0] if candidates else None)
        chosen = locate("xml", f"{jpx_stem}.xml", folder, candidates, "ambiguous" if candidates else "missing", default, {}) \
            if locate else None
        if not chosen and not locate and default:
            chosen = os.path.join(folder, default)
        if not chosen:
            raise FileNotFoundError(f"This JPX has no alignment data of its own, and {jpx_stem}.xml (needed to recover it) "
                                    f"wasn't found in\n{folder}")
        xml_path = chosen

    # --- the case file
    candidates = listing(".h5")
    guids = _xml_guids(xml_path)
    notes = {}
    for n in candidates:
        fits = _h5_matches(os.path.join(folder, n), guids)
        notes[n] = "fits this reconstruction" if fits else ("doesn't fit this reconstruction" if fits is False else "can't be read")
    default = next((n for n in candidates if notes[n] == "fits this reconstruction"), None) \
        or files.get("casefile.h5") or (candidates[0] if candidates else None)
    fitting = [n for n in candidates if notes[n] == "fits this reconstruction"]
    if len(fitting) == 1:
        h5_path = os.path.join(folder, fitting[0])
    else:
        chosen = locate("h5", "the case file (.h5)", folder, candidates, "ambiguous" if candidates else "missing", default, notes) \
            if locate else None
        if not chosen and not locate and default:
            chosen = os.path.join(folder, default)
        if not chosen:
            raise FileNotFoundError("This JPX has no alignment data of its own, and its NeuroInfo case file (.h5) "
                                    f"{'was not chosen' if candidates else 'wasn' + chr(39) + 't found'} in\n{folder}")
        h5_path = chosen
    return xml_path, h5_path


def _load_neuroinfo(jpx_path, locate=None):
    folder = os.path.dirname(os.path.abspath(jpx_path))
    xml_path, h5_path = _find_neuroinfo_files(jpx_path, locate)
    xml_name = os.path.basename(xml_path)
    xml = open(xml_path, encoding="latin-1").read()
    m = re.search(r'<scale x="([0-9.]+)"', xml)
    pixel_um = float(m.group(1)) if m else 0.575
    stack_outlines = {}
    for block in re.findall(r"<contour .*?</contour>", xml, re.S):
        guid = re.search(r'name="GUID"><s>([0-9A-Fa-f]+)</s>', block).group(1).upper()
        stack_outlines[guid] = np.array(
            [[float(x), float(y)] for x, y in re.findall(r'<point x="([-0-9.]+)" y="([-0-9.]+)"', block)])

    frames = [b for b in read_boxes(jpx_path) if b.kind == b"jp2c"]
    with open(jpx_path, "rb") as f:
        w, h, nc = codestream_size(f, frames[0])
    stack = Stack(jpx_path, w, h, nc, pixel_um, origin=os.path.abspath(jpx_path), case_file=os.path.basename(h5_path))

    with h5py.File(h5_path, "r") as f:
        bm = f["BrainMaker"]
        chain = {_s(r["Previous"]): _s(r["Name"]) for r in bm["OrderedSections/Contours"][()]}
        order, cur = [], ""
        while cur in chain:
            cur = chain[cur]
            order.append(cur)
        for i, guid in enumerate(order):
            c = bm["Contours"][guid]
            recorded = _s(c["ImageFile"].attrs["SlideFileName"])
            name = os.path.basename(recorded.replace("\\", "/"))
            pts = c["Points"][()]
            P = np.column_stack([pts["X"], pts["Y"]]).astype(float)
            P_raw = P.copy()
            Q = stack_outlines.get(guid.upper())
            if Q is None or len(Q) != len(P):
                raise ValueError(f"{name}: outline in {xml_name} missing or doesn't match {os.path.basename(h5_path)} "
                                 "(is that the case file of this reconstruction?)")
            R, t, err = fit_rigid(P, Q)
            if err > 1.0:
                # Some source images (e.g. Zeiss .czi slides) have a different pixel size from the stack's (0.65 um against
                # 0.575 in the example this was worked out from): NeuroInfo maps one source pixel to one stack pixel, so the
                # outline in um differs by the ratio of the pixel sizes. Expressing the source outline in um at the stack's
                # pixel size makes the fit rigid again (error ~0.004 um), and the section's source "um" then are the stack's.
                spacing = c["ImageFile"].attrs.get("SpacingXY")
                scale = pixel_um / float(np.mean(spacing)) if spacing is not None and float(np.mean(spacing)) > 0 else 1.0
                R2, t2, err2 = fit_rigid(P * scale, Q)
                if err2 <= 1.0:
                    P, R, t, err = P * scale, R2, t2, err2
                    stack.source_pixel_um = round(float(np.mean(spacing)), 6)   # (offered to the user as a correction, see the aligner)
            if err > 1.0:
                raise ValueError(f"{name}: outlines don't fit a rigid transform (error {err:.2f} um)")
            stack.sections.append(Section(i, name, resolve_source(folder, name, absolute=recorded),
                                          P, R, t, source_abs_hint=recorded, raw_points=P_raw, xml_points=np.array(Q, float)))
    if len(stack.sections) != len(frames):
        raise ValueError(f"{len(stack.sections)} sections in CaseFile.H5 but {len(frames)} frames in the JPX")
    return stack


def _load_aligner(jpx_path, doc):
    if doc.get("format") != JSON_FORMAT or doc.get("version", 0) > JSON_VERSION:
        raise ValueError("Unrecognized alignment data in this JPX (made by a newer version?)")
    folder = os.path.dirname(os.path.abspath(jpx_path))
    stack = Stack(jpx_path, doc["canvas"][0], doc["canvas"][1], doc["channels"], doc["pixel_um"],
                  origin=doc.get("origin", ""), from_aligner=True, source_pixel_um=float(doc.get("source_pixel_um", 0.0)))
    for i, d in enumerate(doc["sections"]):
        e = Edit(**d["edit"])
        stack.sections.append(Section(
            i, d["name"], resolve_source(folder, d["name"], d.get("source_relative", ""), d.get("source_absolute", "")),
            np.array(d["case_points"], float).reshape(-1, 2), np.array(d["base_R"], float), np.array(d["base_t"], float),
            edit=copy.copy(e), saved_edit=copy.copy(e), source_abs_hint=d.get("source_absolute", ""),
            pivot_src=d.get("pivot_src"), czi_origin=tuple(d["czi_origin"]) if d.get("czi_origin") else None))
    frames = [b for b in read_boxes(jpx_path) if b.kind == b"jp2c"]
    if len(stack.sections) != len(frames):
        raise ValueError("Alignment data doesn't match the number of images in this JPX")
    return stack


def open_stack(path, locate=None):
    """Open a JPX written by this tool, or a NeuroInfo reconstruction (a .jpx
    with reconstruction.xml and CaseFile.H5 beside it, or that folder). `locate` is asked for NeuroInfo files that are not
    beside the JPX (see _find_neuroinfo_files)."""
    if os.path.isdir(path):
        names = {n.lower(): n for n in os.listdir(path)}
        if "reconstruction.jpx" not in names:
            raise FileNotFoundError(f"No reconstruction.jpx in {path}")
        path = os.path.join(path, names["reconstruction.jpx"])
    doc = find_aligner_json(path)
    stack = _load_aligner(path, doc) if doc else _load_neuroinfo(path, locate)
    # The XMP repeats channel names once per image; the first `channels` apply
    info = read_xmp_channels(path)[:stack.channels]
    stack.channel_info = info + [ChannelInfo(f"channel {i}", (255, 255, 255))
                                 for i in range(len(info), stack.channels)]
    return stack


def stack_json(stack, out_path):
    out_folder = os.path.dirname(os.path.abspath(out_path))
    secs = []
    for s in stack.sections:
        src = s.source_path or s.source_abs_hint
        rel = ""
        if s.source_path:
            try:
                rel = os.path.relpath(s.source_path, out_folder)
            except ValueError:  # different drive
                rel = ""
        secs.append({
            "name": s.name,
            "source_relative": rel.replace("\\", "/"),
            "source_absolute": os.path.abspath(src).replace("\\", "/") if src else "",
            "base_R": np.round(s.R, 12).tolist(),
            "base_t": np.round(s.t, 6).tolist(),
            "case_points": np.round(s.case_points, 4).tolist(),
            "pivot_src": np.round(s.pivot_src, 6).tolist(),
            "edit": asdict(s.edit),
        })
        if s.czi_origin is not None:
            secs[-1]["czi_origin"] = [round(float(v), 4) for v in s.czi_origin]
    # Version 2 only when needed (an outline was edited or removed, so the pivot is no longer the outline's centroid);
    # otherwise the file stays readable by the first version of this tool.
    needs_v2 = any(not len(s.case_points) or not np.allclose(s.pivot_src, s.case_points.mean(0), atol=1e-4)
                   for s in stack.sections)
    return {
        "format": JSON_FORMAT, "version": 2 if needs_v2 else 1,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "origin": stack.origin, "pixel_um": stack.pixel_um,
        "source_pixel_um": stack.source_pixel_um,
        "canvas": [stack.canvas_w, stack.canvas_h], "channels": stack.channels,
        "sections": secs,
    }


##################################################
#
#    Rendering
#
##################################################

def canvas_to_source_matrix(R, t, pixel_um):
    """2x3 affine M with [col, row] = M @ [u, v, 1]: stack canvas pixel ->
    source pixel, for the transform (R, t) from source um to stack um."""
    to_um = np.diag([pixel_um, -pixel_um])          # pixel -> um (y up)
    to_px = np.diag([1 / pixel_um, -1 / pixel_um])  # um -> pixel
    Rinv = np.linalg.inv(R)
    lin = to_px @ Rinv @ to_um
    off = -to_px @ Rinv @ t
    return np.column_stack([lin, off])


##################################################
#
#    Zeiss .czi sources
#
# A slide scan holds several scenes (one per section, in a grid), each with its position (bbox) in one big pixel grid
# and a pyramid of resolutions. NeuroInfo keeps each section's outline in um relative to a slide origin it does not record,
# in a y-up frame: this tool's source pixel (col, row) of such a section is the slide pixel (col + X0, row + Y0), with
# one (X0, Y0) for the whole .czi file (czi_origin). Both are found by matching a frame of the JPX against the scans
# (calibrate_czi), after which sections are rendered from the scan like any other source image.
#
##################################################

try:
    import j2k_partial      # (reading only the parts of a JPEG 2000 frame that are needed)
except ImportError:
    j2k_partial = None

CZI_WORKERS = max(1, min(os.cpu_count() or 1, 16))   # threads for decoding a scan's tiles (the reader's default is one: 6 s a scene)
_czi_lock = threading.RLock()   # (installing the reader library, and making the locks below)
_czi_files = {}
_czi_locks = {}


def _czi_lock_for(path):
    """The lock of one .czi file. The reader seeks and reads in one file object, so reads of a file take turns; but different
    files are independent, and one lock for all of them made the scans wait for each other."""
    key = os.path.normcase(os.path.abspath(path))
    lock = _czi_locks.get(key)
    if lock is None:
        with _czi_lock:
            lock = _czi_locks.setdefault(key, threading.RLock())
    return lock


def czi_file(path):
    """The (cached) open czifile.CziFile for `path`; installs the reader library first if needed."""
    key = os.path.normcase(os.path.abspath(path))
    with _czi_lock_for(path):
        if key not in _czi_files:
            with _czi_lock:
                ensure_packages(CZI_PACKAGES, exit_on_failure=False)
                import czifile
            _czi_files[key] = czifile.CziFile(path)
        return _czi_files[key]


_czi_boxes = {}
_czi_stage = {}


def czi_scenes(path):
    """Position of each scene in the slide's pixel grid: [(x0, y0, width, height), ...]. Remembered, and answered without
    taking the reader's lock once known: the window asks for it on every redraw, and must not wait for a scene being read."""
    key = os.path.normcase(os.path.abspath(path))
    boxes = _czi_boxes.get(key)
    if boxes is None:
        with _czi_lock_for(path):
            cz = czi_file(path)
            boxes = _czi_boxes[key] = [tuple(int(v) for v in cz.scenes[i].bbox) for i in range(len(cz.scenes))]
    return boxes


def czi_scene_stage(path):
    """The scenes as the scan's metadata places them, in the stage's um: [(centre x, centre y, width, height), ...] by scene
    index (the <CenterPosition> and <ContourSize> of each <Scene>: where the operator put the scan region, and its size).
    These are not the scenes' pixel boxes (czi_scenes), which record where the tiles were actually acquired. []
    if the file has no such metadata."""
    key = os.path.normcase(os.path.abspath(path))
    found = _czi_stage.get(key)
    if found is None:
        with _czi_lock_for(path):
            cz = czi_file(path)
            md = cz.metadata() if callable(getattr(cz, "metadata", None)) else cz.metadata
        md = md if isinstance(md, str) else str(md)
        found = {}
        for idx, centre, size in re.findall(r'<Scene Index="(\d+)"[^>]*>.*?<CenterPosition>([^<]*)</CenterPosition>'
                                            r'.*?<ContourSize>([^<]*)</ContourSize>', md, re.S):
            try:
                (cx, cy), (w, h) = [[float(v) for v in a.split(",")] for a in (centre, size)]
                found[int(idx)] = (cx, cy, w, h)
            except ValueError:
                pass
        found = _czi_stage[key] = [found[i] for i in sorted(found)] if found else []
    return found


def czi_levels(path, scene=0):
    with _czi_lock_for(path):
        return len(czi_file(path).scenes[scene].levels)


def czi_read(path, scene, level=0):
    """A scene at pyramid `level` (each level halves the resolution), H x W x C."""
    with _czi_lock_for(path):
        arr = czi_file(path).scenes[scene].levels[level].asarray(maxworkers=CZI_WORKERS)
    if arr.ndim == 2:
        arr = arr[None]
    return np.moveaxis(arr, 0, -1)


def czi_read_region(path, scene, x, y, w, h):
    """The part (x, y, w, h) of a scene at full resolution, in slide pixels, as H x W x C (zeros where the scene does not
    reach). Only the tiles under it are read, which is far quicker than the whole scene."""
    with _czi_lock_for(path):
        level = czi_file(path).scenes[scene].levels[0]
        try:
            arr = level(roi=(int(x), int(y), int(w), int(h))).asarray(maxworkers=CZI_WORKERS)
        except ValueError:   # (no tile under the region at all: nothing there)
            arr = np.zeros((level.shape[0], int(h), int(w)), level.dtype)
    if arr.ndim == 2:
        arr = arr[None]
    return np.moveaxis(arr, 0, -1)


def czi_scene_of(section, stack):
    """Index of the scene holding the section's outline (the one with most of the outline's points inside)."""
    boxes = czi_scenes(section.source_path)
    P = section.saved_points if len(section.saved_points) else section.pivot_src.reshape(1, 2)
    X = P[:, 0] / stack.pixel_um + section.czi_origin[0]
    Y = -P[:, 1] / stack.pixel_um + section.czi_origin[1]
    inside = [int(np.sum((X >= x0) & (X < x0 + w) & (Y >= y0) & (Y < y0 + h))) for x0, y0, w, h in boxes]
    return int(np.argmax(inside))


def section_source_offset(section, stack):
    """(dx, dy): the section's source pixel (col, row) is at (col + dx, row + dy) in the array that is read for it (the scene
    of a .czi; zero for ordinary images)."""
    if section.czi_origin is None or not section.source_path.lower().endswith(".czi"):
        return 0.0, 0.0
    x0, y0, _, _ = czi_scenes(section.source_path)[czi_scene_of(section, stack)]
    return float(section.czi_origin[0] - x0), float(section.czi_origin[1] - y0)


def czi_slide_thumbnail(path, max_px=300, colors=None):
    """A small picture of the whole slide scan (the scenes put side by side where they lie): (8-bit image, (x0, y0, k)) where
    slide pixel (X, Y) is at ((X - x0) / k, (Y - y0) / k) in the image. With `colors` (an (r, g, b) for each channel) the picture
    of the channels in those colors follows as a third item: (gray, (x0, y0, k), 8-bit RGB image)."""
    boxes = czi_scenes(path)
    gx0, gy0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    gx1, gy1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    k = max(gx1 - gx0, gy1 - gy0) / max_px
    out = np.zeros((int(np.ceil((gy1 - gy0) / k)), int(np.ceil((gx1 - gx0) / k))), np.float32)
    chans = None
    for i, (x0, y0, w, h) in enumerate(boxes):
        level = int(min(max(np.floor(np.log2(max(k, 1))), 0), czi_levels(path, i) - 1))
        raw = czi_read(path, i, level).astype(np.float32)
        img = raw.mean(2)
        tw, th = max(1, round(w / k)), max(1, round(h / k))
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        px, py = round((x0 - gx0) / k), round((y0 - gy0) / k)
        if colors:
            if chans is None:
                chans = np.zeros(out.shape + (raw.shape[2],), np.float32)
            each = np.stack([cv2.resize(raw[:, :, c], (tw, th), interpolation=cv2.INTER_AREA) for c in range(raw.shape[2])], axis=2)
            rc = chans[py:py + th, px:px + tw]
            np.maximum(rc, each[:rc.shape[0], :rc.shape[1]], out=rc)
        region = out[py:py + th, px:px + tw]
        # (a scene's box has empty corners where its tile grid does not reach, and the box overlaps the neighbours' data: the
        # brighter of what is there and what the scene has, so that an empty corner does not wipe out the neighbour's picture)
        np.maximum(region, img[:region.shape[0], :region.shape[1]], out=region)
    hi = max(float(np.percentile(out[out > 0], 99.5)) if (out > 0).any() else 1.0, 1.0)
    gray = (np.clip(out / hi, 0, 1) ** 0.6 * 255).astype(np.uint8)
    if not colors:
        return gray, (gx0, gy0, k)
    rgb = np.zeros(out.shape + (3,), np.float32)
    for c in range(chans.shape[2]):     # (each channel stretched on its own, then shown in its color)
        v = chans[:, :, c]
        hi_c = max(float(np.percentile(v[v > 0], 99.5)) if (v > 0).any() else 1.0, 1.0)
        rgb += (np.clip(v / hi_c, 0, 1) ** 0.6)[:, :, None] * (np.array(colors[c % len(colors)], np.float32) / 255)
    return gray, (gx0, gy0, k), (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def czi_preview(section, stack, max_px):
    """(image H x W x C, k) for a section's scene at the coarsest pyramid level that is still at least max_px on its long
    side (k source pixels per image pixel), as (img, kx, ky) like the aligner's other previews."""
    path = section.source_path
    scene = czi_scene_of(section, stack)
    _, _, w, h = czi_scenes(path)[scene]
    level = 0
    while level + 1 < czi_levels(path, scene) and max(w, h) / 2 ** level > max_px:
        level += 1
    img = czi_read(path, scene, level)
    return img, w / img.shape[1], h / img.shape[0]


def _czi_template(frame_mean, section, stack, S, box, pre_reduced=False):
    """The frame resampled into source pixels (S source px per template px) over box = (cmin, rmin, cmax, rmax). With
    pre_reduced, frame_mean is already the frame at 1/S (a reduced decode), not the full one."""
    cmin, rmin, cmax, rmax = box
    R, t = section.effective()
    M = np.vstack([canvas_to_source_matrix(R, t, stack.pixel_um), [0, 0, 1]])   # canvas px -> source px
    small = frame_mean if (S == 1 or pre_reduced) else cv2.resize(
        frame_mean, (frame_mean.shape[1] // S, frame_mean.shape[0] // S), interpolation=cv2.INTER_AREA)
    B = np.array([[S, 0, cmin], [0, S, rmin], [0, 0, 1]], float)                 # template px -> source px
    D = np.array([[1 / S, 0, 0.5 / S - 0.5], [0, 1 / S, 0.5 / S - 0.5], [0, 0, 1]]) if S > 1 else np.eye(3)
    A = (D @ np.linalg.inv(M) @ B)[:2]                                          # template px -> (shrunk) frame px
    tw, th = int(np.ceil((cmax - cmin) / S)), int(np.ceil((rmax - rmin) / S))
    return cv2.warpAffine(small, A, (tw, th), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _best_match(image, template):
    """(score, (x, y) with sub-pixel refinement) of the best normalized-correlation position of template in image."""
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(res)
    fx, fy = float(x), float(y)
    if 0 < x < res.shape[1] - 1:
        a, b, c = res[y, x - 1], res[y, x], res[y, x + 1]
        if a - 2 * b + c < 0:
            fx += 0.5 * (a - c) / (a - 2 * b + c)
    if 0 < y < res.shape[0] - 1:
        a, b, c = res[y - 1, x], res[y, x], res[y + 1, x]
        if a - 2 * b + c < 0:
            fy += 0.5 * (a - c) / (a - 2 * b + c)
    return float(score), (fx, fy)


def _czi_outline_box(section, stack):
    """The section's outline in source pixels, padded: (cmin, rmin, cmax, rmax)."""
    P = section.saved_points
    cols, rows = P[:, 0] / stack.pixel_um, -P[:, 1] / stack.pixel_um
    pad = 0.08 * max(np.ptp(cols), np.ptp(rows))
    return cols.min() - pad, rows.min() - pad, cols.max() + pad, rows.max() + pad


def _crop_zero(arr, x0, y0, x1, y1):
    """arr[y0:y1, x0:x1] where the part outside arr (a section can reach past its scene) reads as zeros."""
    out = np.zeros((y1 - y0, x1 - x0) + arr.shape[2:], arr.dtype)
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, arr.shape[1]), min(y1, arr.shape[0])
    if cx1 > cx0 and cy1 > cy0:
        out[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = arr[cy0:cy1, cx0:cx1]
    return out


def czi_candidates(stack, czi_path, section, frame, coarse=None, hint=None, pre_reduced=0):
    """Coarse matches of the section's frame (H x W x C) against every scene of the scan: [(score, scene, X0, Y0), ...], best
    first, each with the origin it implies (to within a few pixels). `coarse` is a dict that keeps the scenes' coarse images
    from one call to the next (they are the same for every section of the scan). With hint = (X0, Y0, band_x, band_y), the
    origin is only looked for within that band around the expected one (far fewer places to try)."""
    boxes = czi_scenes(czi_path)
    level = min(3, min(czi_levels(czi_path, i) for i in range(len(boxes))) - 1)
    S = 2 ** level
    box = _czi_outline_box(section, stack)
    if pre_reduced:   # (frame is already a 1/pre_reduced picture, 2-D mean over the channels; S must equal it)
        S, level = pre_reduced, int(np.log2(pre_reduced))
        tmpl = _czi_template(frame, section, stack, S, box, pre_reduced=True)
    else:
        tmpl = _czi_template(frame.astype(np.float32).mean(2), section, stack, S, box)
    found = []
    for i, (x0, y0, w, h) in enumerate(boxes):
        key = (czi_path, i, level)
        if coarse is not None and key in coarse:
            arr = coarse[key]
        else:
            arr = czi_read(czi_path, i, level).astype(np.float32).mean(2)
            if coarse is not None:
                coarse[key] = arr
        ph, pw = tmpl.shape[0] // 2, tmpl.shape[1] // 2      # (the template's padding may reach past the scene)
        if hint is None:
            img = _crop_zero(arr, -pw, -ph, arr.shape[1] + pw, arr.shape[0] + ph)
            left, top = -pw, -ph
        else:
            hx, hy, bx, by = hint
            ex, ey = (hx + box[0] - x0) / S, (hy + box[1] - y0) / S      # where the template should lie in this scene
            wx, wy = int(np.ceil(bx / S)), int(np.ceil(by / S))
            if ex > arr.shape[1] + wx or ex + tmpl.shape[1] < -wx or ey > arr.shape[0] + wy or ey + tmpl.shape[0] < -wy:
                continue   # (the expected place is not in this scene)
            left, top = int(round(ex)) - wx, int(round(ey)) - wy
            img = _crop_zero(arr, left, top, left + tmpl.shape[1] + 2 * wx, top + tmpl.shape[0] + 2 * wy)
        score, (fx, fy) = _best_match(img, tmpl)
        found.append((score, i, x0 + S * (fx + left) - box[0], y0 + S * (fy + top) - box[1]))
    return sorted(found, reverse=True)


def czi_refine(stack, czi_path, section, frame, scene, X0, Y0, steps=((2, 16, None), (0, 8, 1500))):
    """(X0, Y0, score) improved to a fraction of a pixel, from a coarse estimate, by matching at finer pyramid levels."""
    x0, y0 = czi_scenes(czi_path)[scene][:2]
    fmean = frame.astype(np.float32).mean(2)
    box = _czi_outline_box(section, stack)
    score = 0.0
    for level, window, central in steps:
        S = 2 ** level
        b = box
        if central:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            b = (cx - central / 2, cy - central / 2, cx + central / 2, cy + central / 2)
        tmpl = _czi_template(fmean, section, stack, S, b)
        th, tw = tmpl.shape
        ex, ey = (X0 + b[0] - x0) / S, (Y0 + b[1] - y0) / S      # expected position of the template in the scene array
        ax0, ay0 = int(round(ex)) - window, int(round(ey)) - window
        if level == 0:   # (only the tiles under the window are read)
            img = czi_read_region(czi_path, scene, x0 + ax0, y0 + ay0, tw + 2 * window, th + 2 * window).astype(np.float32).mean(2)
        else:
            arr = czi_read(czi_path, scene, level)
            img = _crop_zero(arr, ax0, ay0, ax0 + tw + 2 * window, ay0 + th + 2 * window).astype(np.float32).mean(2)
            del arr
        score, (fx, fy) = _best_match(img, tmpl)
        X0, Y0 = x0 + S * (ax0 + fx) - b[0], y0 + S * (ay0 + fy) - b[1]
    return float(X0), float(Y0), float(score)


# --- locating a section from a reduced decode of its frame and a full-resolution window of it ---
#
# Decoding a whole frame costs seconds (10 s for a 16530 x 11900 one); a section can be placed just as well from a quick 1/8 decode
# (for the search over the scan) and one small window of the frame at full resolution, read without decoding the rest. The
# window is the busiest one (most edges) that lies well inside the outline: NeuroInfo blanks its frames outside the outline
# along a stair-step edge, so near the outline the frame and the scan do not hold the same pixels.

RASTERIO_PACKAGES = {"rasterio": "rasterio"}   # (GDAL's OpenJPEG: it can decode just a window of a frame)
DECODE_BLOCK_PX = 1024   # the block size of the JPEG 2000 reader (GDAL) for these frames
WINDOW_PX = 500          # the first try; if it does not match well, the larger one
WINDOW_PX_LARGE = 1600
WINDOW_MARGIN_PX = 80    # the window stays this far inside the outline
GOOD_MATCH = 0.95
_rasterio_mods = []
_rasterio_lock = threading.Lock()


def rasterio_modules():
    """(MemoryFile, Window) from rasterio (installed on first use), or None if it cannot be had (the older, slower way is used)."""
    with _rasterio_lock:
        if not _rasterio_mods:
            try:
                ensure_packages(RASTERIO_PACKAGES, exit_on_failure=False)
                import warnings
                import rasterio
                from rasterio.io import MemoryFile
                from rasterio.windows import Window
                warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
                _rasterio_mods.append((MemoryFile, Window))
            except Exception:
                _rasterio_mods.append(None)
        return _rasterio_mods[0]


class _Frame:
    """The compressed frame of a section (one JPEG 2000 codestream): decoded only as much as needed.

    Given as `data` (the whole codestream), or as `source` = (JPX path, the frame's box): then only the parts of the file that
    the picture or window asked for are read (see j2k_partial; about 7 MB of 59 for the 1/8 picture and a window), unless the
    codestream is not laid out for that, when the whole frame is read."""

    def __init__(self, data=None, source=None):
        self.data = data
        self.source = source
        self.layout = None
        self.partial = source is not None and j2k_partial is not None

    def _stream(self, keep_res=None, skip=None, window=None):
        """The codestream to decode: a reduced one when the file can give that, else the whole frame."""
        if self.partial:
            try:
                path, box = self.source
                with open(path, "rb") as f:
                    if self.layout is None:
                        self.layout = j2k_partial.parse_layout(f, box.payload_offset, box.payload_length)
                    keep = self.layout.levels - skip if skip is not None else -1
                    return j2k_partial.build(f, box.payload_offset, self.layout, keep, window)
            except (j2k_partial.Unsupported, KeyError):
                self.partial = False
        if self.data is None:
            path, box = self.source
            with open(path, "rb") as f:
                self.data = read_payload(f, box)
        return self.data

    def reduced_mean(self, skip):
        """The frame at 1/2**skip, averaged over its channels (float32)."""
        img = imagecodecs.jpeg2k_decode(self._stream(skip=skip), skipres=skip)
        return img.astype(np.float32).mean(2) if img.ndim == 3 else img.astype(np.float32)

    def window_mean(self, win):
        """The window (u0, v0, w, h) of the frame at full resolution, averaged over its channels (float32)."""
        MemoryFile, Window = rasterio_modules()
        with MemoryFile(self._stream(window=win)) as mf:
            with mf.open() as ds:
                return ds.read(window=Window(*win)).astype(np.float32).mean(0)


def _canvas_to_source(section, stack):
    R, t = section.effective()
    return np.vstack([canvas_to_source_matrix(R, t, stack.pixel_um), [0, 0, 1]])


def _outline_in_canvas(section, stack, M):
    """The outline as canvas pixels (n x 2), at full resolution."""
    P = section.saved_points
    pts = np.column_stack([P[:, 0] / stack.pixel_um, -P[:, 1] / stack.pixel_um, np.ones(len(P))])
    return (np.linalg.inv(M) @ pts.T).T[:, :2]


def _covered_by_other_scenes(shape, S, M, X0, Y0, boxes, scene, margin_px):
    """Mask (at 1/S, the shape of the 1/S picture) of the canvas pixels that, for a section with origin (X0, Y0) on `scene`, lie on
    slide area that another scene of the scan covers too (within margin_px). NeuroInfo draws such strips from the neighbouring
    scene, which is registered a few tens of pixels differently, so frame and scene do not agree there."""
    h, w = shape
    u, v = np.meshgrid((np.arange(w) + 0.5) * S, (np.arange(h) + 0.5) * S)
    X = M[0, 0] * u + M[0, 1] * v + M[0, 2] + X0
    Y = M[1, 0] * u + M[1, 1] * v + M[1, 2] + Y0
    covered = np.zeros(shape, bool)
    for j, (bx, by, bw, bh) in enumerate(boxes):
        if j != scene:
            covered |= (X > bx - margin_px) & (X < bx + bw + margin_px) & (Y > by - margin_px) & (Y < by + bh + margin_px)
    return covered


def _busiest_window(small_mean, S, win_px, section, stack, M, margin_px, avoid=None):
    """Canvas window (u0, v0, w, h) of win_px square where the 1/S picture has most edges, lying inside the outline with margin_px
    to spare, or None if there is no outline. `avoid(margin)` gives a mask of pixels to keep out of the window (the strips covered
    by a second scene); the margins are relaxed (the avoided area narrower, then not avoided, the outline margin dropped) when no
    window would fit."""
    if not len(section.saved_points):
        return None
    poly = _outline_in_canvas(section, stack, M) / S
    g = cv2.GaussianBlur(small_mean, (0, 0), 1.2)
    E = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2)
    inside = np.zeros(E.shape, np.uint8)
    cv2.fillPoly(inside, [poly.astype(np.int32)], 1)
    k = max(3, int(round(win_px / S)))
    energy = cv2.boxFilter(E * inside, -1, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)
    chosen = None
    avoid_margins = (200, 60, None) if avoid is not None else (None,)   # (px beyond the strip; the origin is only roughly known)
    for am in avoid_margins:
        allowed = inside if am is None else (inside * (1 - avoid(am).astype(np.uint8))).astype(np.uint8)
        for m in (margin_px, 0):
            half = int(np.ceil((win_px / 2 + m) / S))
            ok = cv2.erode(allowed, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * half + 1, 2 * half + 1)))
            if ok.any():
                chosen = np.where(ok > 0, energy, -1)
                break
        if chosen is not None:
            break
    if chosen is None:
        chosen = energy
    if win_px <= DECODE_BLOCK_PX:
        # The reader decodes the frame in blocks of 1024 px, and a window that straddles block borders pays for every block it
        # touches (4 blocks: 1.25 s; 1 block: 0.3 s). So the window is kept inside one block where there is a choice.
        u_ok = ((np.arange(chosen.shape[1]) - k // 2) * S) % DECODE_BLOCK_PX + win_px <= DECODE_BLOCK_PX
        v_ok = ((np.arange(chosen.shape[0]) - k // 2) * S) % DECODE_BLOCK_PX + win_px <= DECODE_BLOCK_PX
        inside_block = np.where(v_ok[:, None] & u_ok[None, :], chosen, -1)
        if (inside_block >= 0).any():
            chosen = inside_block
    ys, xs = np.unravel_index(int(np.argmax(chosen)), chosen.shape)
    u0 = int(round((xs - k // 2) * S))
    v0 = int(round((ys - k // 2) * S))
    return (max(0, min(stack.canvas_w - win_px, u0)), max(0, min(stack.canvas_h - win_px, v0)), win_px, win_px)


def _source_box_in_window(win, M):
    """The largest axis-aligned square, in source pixels, inside the (rotated) canvas window."""
    u0, v0, w, h = win
    centre = (M @ np.array([u0 + w / 2, v0 + h / 2, 1.0]))[:2]
    th = np.arctan2(M[1, 0], M[0, 0])
    half = (w / 2) / (abs(np.cos(th)) + abs(np.sin(th))) - 4
    return (centre[0] - half, centre[1] - half, centre[0] + half, centre[1] + half)


def _window_template(region, win, M, sbox):
    """The frame's window `region` resampled into source pixels (1:1) over sbox."""
    cmin, rmin, cmax, rmax = sbox
    B = np.array([[1, 0, cmin], [0, 1, rmin], [0, 0, 1]], float)
    A = (np.linalg.inv(M) @ B)[:2].copy()
    A[0, 2] -= win[0]
    A[1, 2] -= win[1]
    return cv2.warpAffine(region, A, (int(np.floor(cmax - cmin)), int(np.floor(rmax - rmin))),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _match_window(czi_path, scene, tmpl, sbox, X0e, Y0e, margin=40):
    """(X0, Y0, score): the template matched against the full-resolution scan around where X0e, Y0e put it."""
    rx, ry = int(round(X0e + sbox[0])) - margin, int(round(Y0e + sbox[1])) - margin
    th, tw = tmpl.shape
    reg = czi_read_region(czi_path, scene, rx, ry, tw + 2 * margin, th + 2 * margin).astype(np.float32).mean(2)
    score, (fx, fy) = _best_match(reg, tmpl)
    return rx + fx - sbox[0], ry + fy - sbox[1], score


def _scene_at(czi_path, section, stack, X0, Y0):
    """Index of the scene holding the section's outline if its origin were (X0, Y0)."""
    boxes = czi_scenes(czi_path)
    P = section.saved_points
    X = P[:, 0] / stack.pixel_um + X0
    Y = -P[:, 1] / stack.pixel_um + Y0
    inside = [int(np.sum((X >= x0) & (X < x0 + w) & (Y >= y0) & (Y < y0 + h))) for x0, y0, w, h in boxes]
    return int(np.argmax(inside))


FINE_MARGINS = (160, 400)   # how far (px) around the expected place the full-resolution match looks, tried in turn


def _fine_match(czi_path, scene, tmpl, sbox, X0, Y0, margins=FINE_MARGINS):
    """The best of _match_window over growing margins (stopping as soon as one matches well)."""
    best = None
    for m in margins:
        x, y, score = _match_window(czi_path, scene, tmpl, sbox, X0, Y0, m)
        if best is None or score > best[2]:
            best = (x, y, score)
        if score >= GOOD_MATCH:
            break
    return best


def czi_locate_fast(stack, czi_path, section, frame, coarse, guesses=(), candidate=None, strong=0.9, hint=None):
    """Place a section on the scan from `frame` (a _Frame). Returns (candidates, (X0, Y0, score) or None).

    `guesses` are origins (X0, Y0, margin) expected from the sections placed already; each is tried first (a match at the
    expected place needs no search over the scan). Otherwise the 1/8 picture is searched against the scan for the coarse
    candidates (best first; a `candidate` skips that), and the best of them, when clear (score >= strong), is matched at full
    resolution, looking further around it if the first look does not match well (the coarse place can be off by 100 px or
    more). Each full-resolution match uses the busiest window inside the outline, 500 px and then 1600 px."""
    # (a quicker, coarser decode (1/16) is enough to pick the window for a match at a guessed place; the 1/8 one is for the search)
    S = 16 if guesses else 8
    small = frame.reduced_mean(int(np.log2(S)))
    M = _canvas_to_source(section, stack)
    boxes = czi_scenes(czi_path)
    regions = {}   # the windows already read from the frame

    masks = {}   # the strips covered by a second scene, for each place tried (the same for both window sizes)

    def window_template(size, scene, X0, Y0):
        """The busiest window of `size` that no second scene covers (given where the section is expected to be)."""
        def avoid(m):
            key = (scene, round(X0 / 25), round(Y0 / 25), m)
            if key not in masks:
                masks[key] = _covered_by_other_scenes(small.shape, S, M, X0, Y0, boxes, scene, m)
            return masks[key]
        win = _busiest_window(small, S, size, section, stack, M, WINDOW_MARGIN_PX, avoid=avoid)
        if win is None:
            return None
        sbox = _source_box_in_window(win, M)
        if win not in regions:
            regions[win] = frame.window_mean(win)
        return sbox, _window_template(regions[win], win, M, sbox)

    def attempt(X0, Y0, scene, margins):
        best = None
        for size in (WINDOW_PX, WINDOW_PX_LARGE):
            wt = window_template(size, scene, X0, Y0)
            if wt is None:
                break
            x, y, score = _fine_match(czi_path, scene, wt[1], wt[0], X0, Y0, margins)
            if best is None or score > best[2]:
                best = (x, y, score)
            if score >= GOOD_MATCH or score < 0.5:   # (a larger window cannot rescue a match that is not there at all)
                break
        return best

    for gx, gy, gmargin in guesses:
        scene = _scene_at(czi_path, section, stack, gx, gy)
        best = attempt(gx, gy, scene, (gmargin,))
        if best is not None and best[2] >= GOOD_MATCH:
            return [(best[2], scene, best[0], best[1])], best
    found = [candidate] if candidate is not None else []
    if S != 8:   # (no guess matched: the search and the windows after it work from the 1/8 picture)
        S = 8
        small = frame.reduced_mean(3)
        masks.clear()
    if candidate is None:
        if hint is not None:   # (the coarse search only over the band where the origin is expected)
            found = czi_candidates(stack, czi_path, section, small, coarse, hint=hint, pre_reduced=8)
        if not found or found[0][0] < strong:
            found = czi_candidates(stack, czi_path, section, small, coarse, pre_reduced=8)
    if not found or (candidate is None and found[0][0] < strong):
        return found, None
    _, scene, X0, Y0 = found[0]
    return found, attempt(X0, Y0, scene, FINE_MARGINS)


def calibrate_stack_czi(stack, log=lambda m: None, strong=0.9, accept=0.6, progress=None, on_located=None, workers=2, share=True, scans_at_once=None):
    """Locate the sections of every .czi source of the stack that have no czi_origin yet, so that they can be rendered.

    NeuroInfo's slide coordinates relate to the scan by an offset (X0, Y0) that it does not record, found here by matching each
    section's frame in the JPX against the scan. X0 is the same for the whole scan; Y0 turns out to differ a little between
    sections, so each section gets its own. Ambiguous matches (a faint section can resemble the wrong scene) are resolved by
    taking the X0 the clear matches agree on. Returns {file name: error text or ""}. progress(done, total, text) is called as
    the sections are located, and on_located(section) as soon as each one has its czi_origin (possibly from another thread).
    With `share` (and 6 or more sections in the scan) only a few sections are located and the scan's other sections are given
    their origin (czi_assumed is set on those); see verify_czi_origins."""
    frames = [b for b in read_boxes(stack.jpx_path) if b.kind == b"jp2c"]

    def load_frame(sec):
        with open(stack.jpx_path, "rb") as f:
            img = imagecodecs.jpeg2k_decode(read_payload(f, frames[sec.frame]))
        return img[..., None] if img.ndim == 2 else img

    def payload(sec):
        return _Frame(source=(stack.jpx_path, frames[sec.frame]))

    problems = {}

    first_ready = threading.Event()   # set when the first section of the stack has been placed (or its scan has failed)
    first_path = []

    def do_scan(path):
        """One scan; the others wait for the scan of the stack's first section to place that section, so that it is ready (and
        can be shown) as soon as possible, with all the processors to itself."""
        if first_path and path != first_path[0]:
            first_ready.wait(timeout=120)
        try:
            scan(path)
        finally:
            if first_path and path == first_path[0]:
                first_ready.set()

    def scan(path):
        name = os.path.basename(path)
        todo = [s for s in stack.sections if s.source_path == path and s.czi_origin is None and len(s.saved_points)]
        if not todo:
            problems[name] = ""
            return
        try:
            # the quick way needs rasterio and scenes with at least 4 pyramid levels (for the 1/8 search)
            fast = rasterio_modules() is not None and min(czi_levels(path, i) for i in range(len(czi_scenes(path)))) >= 4
            coarse = {}
            cands, done = {}, [0]
            sizes = {sec.frame: max(np.ptp(sec.saved_points[:, 0]), np.ptp(sec.saved_points[:, 1])) for sec in todo}
            typical = float(np.median(list(sizes.values())))
            total = [len(todo)]
            lock = threading.Lock()

            active = set()   # the sections being looked at now (two at a time): the lowest is the one named

            def report(text=None):
                if progress:
                    if text is None:
                        text = (f"Locating section {min(active) + 1} of {len(stack.sections)} on the slide scan..."
                                if active else "Locating sections on the slide scan...")
                    progress(done[0], total[0], text)

            located_so_far = []   # (outline y in pixels, X0, Y0) of the sections placed so far
            failed = set()        # sections that did not match at their coarse place

            def hint_for(sec):
                """(X0, Y0, band_x, band_y) around which the coarse search looks first, from the sections placed so far."""
                with lock:
                    done_ = list(located_so_far)
                if len(done_) < 3 or not len(sec.saved_points):
                    return None
                ys = np.array([d[0] for d in done_]); xs = np.array([d[1] for d in done_]); y0s = np.array([d[2] for d in done_])
                y = float(sec.saved_points[:, 1].mean()) / stack.pixel_um
                if len(done_) >= 4 and np.ptp(ys) > 3000:
                    slope, icpt = np.polyfit(ys, y0s, 1)
                    resid = float(np.std(y0s - (slope * ys + icpt)))
                    return float(np.median(xs)), float(slope * y + icpt), 100, int(max(1100, 4 * resid + 300))
                return None

            def guesses_for(sec):
                """Origins (X0, Y0, margin) to try for the section first, from those placed already in this scan. X0 is the same
                for all of them. Y0 is the same too in some scans, and in others follows the outline's y in the case file; which
                it is, is seen from the sections placed so far (at least 3, or 4 spread out along y, are needed)."""
                with lock:
                    done_ = list(located_so_far)
                if not done_ or not len(sec.saved_points):
                    return []
                ys = np.array([d[0] for d in done_]); xs = np.array([d[1] for d in done_]); y0s = np.array([d[2] for d in done_])
                x0 = float(np.median(xs))
                if len(done_) < 3:
                    # (the origin of the section placed first, or of the two placed first when they agree: the sections of most
                    # scans share one, and a match at that place needs no search over the scan; if it does not match, the
                    # search follows as usual)
                    return [(x0, float(np.median(y0s)), 40)] if np.std(y0s) < 6 else []
                y = float(sec.saved_points[:, 1].mean()) / stack.pixel_um
                out = []
                if np.std(y0s) < 6:   # the same Y0 for all of them
                    out.append((x0, float(np.median(y0s)), 40))
                if len(done_) >= 4 and np.ptp(ys) > 3000:
                    slope, icpt = np.polyfit(ys, y0s, 1)
                    resid = float(np.std(y0s - (slope * ys + icpt)))
                    if 4 * resid <= 200:   # (with more scatter than that, a search over the scan is the quicker way)
                        out.append((x0, float(slope * y + icpt), int(max(4 * resid, 40))))
                return out

            def look(sec):
                """A quick look at the section; if it is clearly at one place, position it precisely right away (the frame
                is in memory now, and decoding it again would cost as much as everything else). Once some sections are placed,
                the others are looked for only near where they are expected."""
                active.add(sec.index)
                report()
                if fast:
                    found, placed = czi_locate_fast(stack, path, sec, payload(sec), coarse, guesses_for(sec), strong=strong,
                                                    hint=hint_for(sec))
                else:
                    frame = load_frame(sec)
                    found = czi_candidates(stack, path, sec, frame, coarse)
                    placed = None
                    if found and found[0][0] >= strong:
                        _, scene, X0, Y0 = found[0]
                        placed = czi_refine(stack, path, sec, frame, scene, X0, Y0)
                if fast and found and found[0][0] >= strong and (placed is None or placed[2] < accept):
                    failed.add(sec.frame)   # (not found at its coarse place: no point trying the same place again later)
                if placed is not None:
                    X0, Y0, score = placed
                    if score >= accept:
                        sec.czi_origin = (X0, Y0)
                        with lock:
                            located_so_far.append((float(sec.saved_points[:, 1].mean()) / stack.pixel_um, X0, Y0))
                        if on_located:
                            on_located(sec)
                active.discard(sec.index)
                report()
                return sec, found

            # All the sections of a scan share one origin (it is the scan's: the spread between the sections of a scan is at the
            # level of the matching's own noise). So a few of the largest sections are located properly, and the origin they
            # agree on is given to the others, which then need neither decoding nor searching. (Checking each of those against
            # its own frame is verify_czi_origins's job, which can run afterwards in the background.)
            # The first section of the list alone, so that it is ready (and can be shown) as soon as possible.
            sec, found = look(todo[0])
            cands[sec.frame] = found
            done[0] += 1
            if first_path and path == first_path[0]:
                first_ready.set()
            rest = todo[1:]
            with ThreadPoolExecutor(max_workers=workers, initializer=lower_thread_priority) as pool:
                if fast and share and len(todo) >= 6:
                    # The sections of a scan often share one origin (the spread between them is at the level of the matching's
                    # own noise). So a few of the largest sections are located properly, and if the origin they agree on is
                    # that of the first section too, it is given to the others, which then need neither decoding nor
                    # searching. (Checking each of those against its own frame is verify_czi_origins's job, and can run
                    # afterwards in the background.) When they do not agree, the others are located one by one as usual.
                    order = sorted(rest, key=lambda s: -sizes[s.frame])

                    def try_share(need):
                        """Give the origin that `need` located sections agree on to the others, if they do."""
                        got = np.array([s.czi_origin for s in todo if s.czi_origin is not None], float)
                        left = [s for s in rest if s.czi_origin is None]
                        if len(got) < need or not left:
                            return left, False
                        med = np.median(got, axis=0)
                        agree = got[np.hypot(got[:, 0] - med[0], got[:, 1] - med[1]) <= 3.0]
                        if len(agree) < need:
                            return left, False
                        shared = tuple(float(v) for v in np.median(agree, axis=0))
                        for s in left:
                            s.czi_origin, s.czi_assumed = shared, True
                            done[0] += 1
                            if on_located:
                                on_located(s)
                        log(f"{name}: {len(agree)} sections located, their origin {shared[0]:.1f}, {shared[1]:.1f} given to {len(left)} more")
                        return [], True

                    # One more section is located first, at the first one's origin if it is there (no search over the scan needed);
                    # if the two agree, that is enough. Otherwise two more are located, and three must agree.
                    for sec, found in pool.map(look, order[:1]):
                        cands[sec.frame] = found
                        with lock:
                            done[0] += 1
                    rest, shared_ok = try_share(2)
                    if not shared_ok and rest:
                        more = [s for s in order[1:] if s.czi_origin is None][:2]
                        for sec, found in pool.map(look, more):
                            cands[sec.frame] = found
                            with lock:
                                done[0] += 1
                        rest, shared_ok = try_share(3)
                    rest = [s for s in rest if s.czi_origin is None]
                for sec, found in pool.map(look, rest):
                    cands[sec.frame] = found
                    with lock:
                        done[0] += 1
            clear = [c[0][2] for c in cands.values() if c and c[0][0] >= strong]
            x_known = float(np.median(clear)) if clear else None
            missed, small = [], []
            later = [sec for sec in todo if sec.czi_origin is None]
            total[0] = len(todo) + len(later)
            for n, sec in enumerate(later, 1):
                c = [k for k in cands[sec.frame] if x_known is None or abs(k[2] - x_known) < 60]
                g = guesses_for(sec) if fast else []
                if sec.frame in failed and not g:
                    c = []
                if not c and not g:
                    (small if sizes[sec.frame] < 0.35 * typical else missed).append(str(sec.index + 1))
                    continue
                report(f"Positioning section {sec.index + 1} of {len(stack.sections)}...")
                if fast:
                    _, placed = czi_locate_fast(stack, path, sec, payload(sec), coarse, g, candidate=c[0] if c else None, strong=strong)
                    X0, Y0, score = placed if placed is not None else (0.0, 0.0, 0.0)
                else:
                    X0, Y0, score = czi_refine(stack, path, sec, load_frame(sec), c[0][1], c[0][2], c[0][3])
                done[0] += 1
                if score >= accept:
                    sec.czi_origin = (X0, Y0)
                    if on_located:
                        on_located(sec)
                else:
                    (small if sizes[sec.frame] < 0.35 * typical else missed).append(str(sec.index + 1))
            notes = []
            if missed:
                notes.append(f"section{'s' if len(missed) > 1 else ''} {', '.join(missed[:8])}{', ...' if len(missed) > 8 else ''} "
                             "couldn't be located on the scan")
            if small:   # (tiny pieces of tissue are often stray fragments, which are not worth the trouble)
                notes.append(f"section{'s' if len(small) > 1 else ''} {', '.join(small[:8])} "
                             f"{'are' if len(small) > 1 else 'is'} very small (a stray fragment?) and couldn't be located on the scan")
            problems[name] = "; ".join(notes)
        except Exception as e:
            problems[name] = f"the scan couldn't be read ({type(e).__name__}: {e})"

    paths = list(dict.fromkeys(s.source_path for s in stack.sections if s.source_path.lower().endswith(".czi")))   # (in stack order)
    todo_paths = [p for p in paths if any(s.source_path == p and s.czi_origin is None for s in stack.sections)]
    first_path[:] = todo_paths[:1]
    if scans_at_once is None:   # (a scan's own steps follow one another, so more scans at once is what keeps the processors busy: 45 s
        scans_at_once = max(4, min(12, int((os.cpu_count() or 8) * 0.6)))   # for 24 scans with 12, 59 s with 6, 50 s with 24 on 20 processors)
    with ThreadPoolExecutor(max_workers=scans_at_once) as scan_pool:   # (several scans at once)
        list(scan_pool.map(do_scan, paths))
    return problems


def verify_czi_origins(stack, log=lambda m: None, progress=None, tolerance=3.0, workers=4, stop=None):
    """Check each section whose origin was given (czi_assumed) against its own frame, and settle it:
      - the section's own origin is within `tolerance` px of the assumed one: confirmed (the assumed one is kept);
      - it differs by more, and the match is good: the section's own origin replaces it (reported as corrected);
      - no good match can be had: the assumed origin stays, and the section is reported as unverified.
    Returns {"checked": n, "corrected": [(section number, dx, dy)], "unverified": [section numbers]}; progress(done, total, text)
    is called along the way, and `stop` (a threading.Event) ends it early."""
    todo = [s for s in stack.sections if s.czi_assumed and s.czi_origin is not None]
    by_scan = {}
    for s in todo:
        by_scan.setdefault(s.source_path, []).append(s)
    todo = [s for group in itertools.zip_longest(*by_scan.values()) for s in group if s is not None]   # (one from each scan in turn)
    summary = {"checked": 0, "corrected": [], "unverified": []}
    if not todo or rasterio_modules() is None:
        return summary
    frames = [b for b in read_boxes(stack.jpx_path) if b.kind == b"jp2c"]
    lock = threading.Lock()
    done = [0]

    def check(sec):
        if stop is not None and stop.is_set():
            return
        frame = _Frame(source=(stack.jpx_path, frames[sec.frame]))
        X0, Y0 = sec.czi_origin
        _, best = czi_locate_fast(stack, sec.source_path, sec, frame, {}, guesses=[(X0, Y0, 40)])
        with lock:
            done[0] += 1
            summary["checked"] += 1
            if best is None or best[2] < GOOD_MATCH:
                summary["unverified"].append(sec.frame + 1)
            elif np.hypot(best[0] - X0, best[1] - Y0) > tolerance:
                summary["corrected"].append((sec.frame + 1, round(best[0] - X0, 1), round(best[1] - Y0, 1)))
                sec.czi_origin = (float(best[0]), float(best[1]))
            sec.czi_assumed = False
            if progress:
                progress(done[0], len(todo), f"Checking section positions against their frames ({done[0]} of {len(todo)})...")

    with ThreadPoolExecutor(max_workers=workers, initializer=lower_thread_priority) as pool:
        list(pool.map(check, todo))
    return summary


def render_section(section, stack, source=None, interpolation=cv2.INTER_LINEAR, use_edit=True):
    """Full-resolution section image on the stack canvas (H x W x C)."""
    offset = (0.0, 0.0)
    if source is None:
        if not section.source_path:
            raise FileNotFoundError(f"source image for {section.name} not found")
        if section.source_path.lower().endswith(".czi"):
            scene = czi_scene_of(section, stack)
            source = czi_read(section.source_path, scene, 0)
            offset = section_source_offset(section, stack)
        else:
            source = imagecodecs.imread(section.source_path)
    if source.ndim == 2:
        source = source[..., None]
    R, t = section.effective() if use_edit else (section.R, section.t)
    M = canvas_to_source_matrix(R, t, stack.pixel_um)
    M[:, 2] += offset
    out = np.zeros((stack.canvas_h, stack.canvas_w, source.shape[2]), source.dtype)
    for c in range(source.shape[2]):
        out[..., c] = cv2.warpAffine(source[..., c], M, (stack.canvas_w, stack.canvas_h),
                                     flags=interpolation | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if section.source_path.lower().endswith(".czi") and len(section.case_points) >= 3:
        # A slide scan holds more than one section (and debris); NeuroInfo's frames keep only what is inside the section's
        # outline, so everything outside the (possibly edited) outline is blanked here too.
        P = section.stack_points if use_edit else (R @ section.case_points.T).T + t
        mask = np.zeros(out.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.column_stack([P[:, 0] / stack.pixel_um, -P[:, 1] / stack.pixel_um]).round().astype(np.int32)], 1)
        for c in range(out.shape[2]):      # (a mask, not out[mask == 0] = 0, which is slow on a canvas this big)
            out[..., c] = cv2.bitwise_and(out[..., c], out[..., c], mask=mask)
    return out


def encode_frame(image, quality_db=DEFAULT_QUALITY_DB, threads=None):
    cs = imagecodecs.jpeg2k_encode(image, quality_db, codecformat="j2k", mct=False,
                                   resolutions=7, numthreads=threads or os.cpu_count() or 1)
    return add_com_markers(cs, ["MBF Bioscience"])


def patch_xmp_pixel_size(payload, pixel_um):
    """The MBF XMP (a uuid box payload, UUID first) with its images' PixelWidth/PixelHeight set to `pixel_um`."""
    text = payload[16:].decode("utf-8", "replace")
    value = f"{pixel_um:.6f}"
    text = re.sub(r'(\sxmp:PixelWidth=")[^"]*(")', lambda m: m.group(1) + value + m.group(2), text)
    text = re.sub(r'(\sxmp:PixelHeight=")[^"]*(")', lambda m: m.group(1) + value + m.group(2), text)
    return payload[:16] + text.encode("utf-8")


def write_jpx_like(template_path, codestreams, out_path, extra_payloads=(), pixel_um=None):
    """Copy template_path's boxes verbatim, replacing its jp2c codestreams in
    order with `codestreams` (bytes, or callables returning bytes), dropping
    any old aligner box and appending `extra_payloads` as uuid boxes. With fewer
    codestreams than the template has frames, the template's last frames (each
    with its own jpch/jplh header boxes) are left out."""
    boxes = read_boxes(template_path)
    frames = [b for b in boxes if b.kind == b"jp2c"]
    if len(codestreams) > len(frames) or not codestreams:
        raise ValueError(f"template has {len(frames)} frames, got {len(codestreams)} codestreams")
    tmp = out_path + ".partial"
    i = 0
    try:
        with open(template_path, "rb") as src, open(tmp, "wb") as dst:
            for b in boxes:
                if i >= len(codestreams) and b.kind in (b"jp2c", b"jpch", b"jplh"):
                    continue   # (the frames that were left out, and their header boxes)
                if b.kind == b"jp2c":
                    cs = codestreams[i]
                    dst.write(box_bytes(b"jp2c", cs() if callable(cs) else cs))
                    i += 1
                    continue
                if b.kind == b"uuid" and read_payload(src, b, 16) == ALIGNER_UUID:
                    continue
                if pixel_um is not None and b.kind == b"uuid" and read_payload(src, b, 16) == XMP_UUID:
                    dst.write(box_bytes(b"uuid", patch_xmp_pixel_size(read_payload(src, b), pixel_um)))   # corrected pixel size
                    continue
                src.seek(b.offset)
                dst.write(src.read(b.length))
            for payload in extra_payloads:
                dst.write(box_bytes(b"uuid", payload))
        os.replace(tmp, out_path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# The start of a JPX written by NeuroInfo (signature, file type, reader requirements): the same for every one.
JPX_PREAMBLE = bytes.fromhex("0000000c6a5020200d0a870a0000001c667479706a707820000000006a7078206a7032206a707862000000457272657103ffc000003ff0000a0005800000002d40000000022000000012100000001308000000010400000008020000000c010000001f00800000140040000000")


def make_xmp(n_frames, channel_info, pixel_um):
    """The XMP packet NeuroInfo keeps in its JPX files: the channels' colors, and for every frame its pixel size and channels."""
    def rgb(c):
        return f'xmp:Red="{c[0]}" xmp:Green="{c[1]}" xmp:Blue="{c[2]}"'
    px = f"{pixel_um:.6f}"
    t = ['<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?> <x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 6.0.0"> '
         '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"> <rdf:Description rdf:about="" '
         'xmlns:xmp="http://ns.adobe.com/xap/1.0/"> <xmp:Image_Headers> <rdf:Bag> <rdf:li xmp:Application="NeuroInfo" '
         'xmp:Version="2026.1.1" xmp:UseChannelData="True" xmp:CameraName="" xmp:MicroscopeName="" xmp:LensName="" '
         'xmp:CorrectionFactor="0.000000" xmp:CreateDate=""/> </rdf:Bag> </xmp:Image_Headers> <xmp:ColorMaps> <rdf:Bag>']
    for i, ch in enumerate(channel_info):
        t.append(f' <rdf:li> <rdf:Description xmp:Map="{i}" xmp:BitsUsed="16" xmp:MapType="0" xmp:BlackPoint="0" '
                 'xmp:WhitePoint="65535" xmp:Gamma="1.000000" xmp:GrayScale="False" xmp:Gain="1.000000" '
                 'xmp:WhiteBalance="1.000000" xmp:Offset="0"> <xmp:BlackColor xmp:Red="0" xmp:Green="0" xmp:Blue="0" xmp:Alpha="0"/> '
                 f'<xmp:WhiteColor {rgb(ch.color)} xmp:Alpha="0"/> </rdf:Description> </rdf:li>')
    t.append(' </rdf:Bag> </xmp:ColorMaps> <xmp:Images> <rdf:Bag>')
    for f in range(n_frames):
        t.append(f' <rdf:li> <rdf:Description xmp:Image="{f}" xmp:Rotation="0.000000" xmp:TranslationX="0.000000" '
                 f'xmp:TranslationY="0.000000" xmp:PixelWidth="{px}" xmp:PixelHeight="{px}" xmp:SizeZ="240.000000" '
                 'xmp:OrigPixelWidth="-1.000000" xmp:OrigPixelHeight="-1.000000" xmp:OrigSizeZ="-1.000000" xmp:OriginX="0.000000" '
                 'xmp:OriginY="0.000000" xmp:OriginZ="0.000000" xmp:CombinationMethod="0"> <xmp:Channels> <rdf:Bag>')
        for i, ch in enumerate(channel_info):
            t.append(f' <rdf:li xmp:Channel="{i}" xmp:ColorMapID="{i}" xmp:Name="" xmp:Enabled="True"/>')
        t.append(' </rdf:Bag> </xmp:Channels> <xmp:Groups> <rdf:Bag>')
        for i, ch in enumerate(channel_info):
            t.append(f' <rdf:li xmp:GroupName="{ch.name}" xmp:Red="{i}" xmp:Green="-1" xmp:Blue="-1" '
                     'xmp:ExposureSeconds="-1.000000" xmp:Gain="-1.000000"/>')
        t.append(' </rdf:Bag> </xmp:Groups> </rdf:Description> </rdf:li>')
    t.append(' </rdf:Bag> </xmp:Images> </rdf:Description> </rdf:RDF> </x:xmpmeta> <?xpacket end="w"?>')
    return "".join(t).encode("utf-8")


def write_jpx_new(codestreams, out_path, canvas_w, canvas_h, channel_info, pixel_um, extra_payloads=()):
    """A multi-frame JPX of the codestreams (bytes, or callables returning bytes), laid out like NeuroInfo's, written from
    scratch (write_jpx_like needs an existing JPX to copy from)."""
    n = len(codestreams)
    if not n:
        raise ValueError("no codestreams")
    ihdr = box_bytes(b"ihdr", struct.pack(">IIHBBBB", canvas_h, canvas_w, len(channel_info), 15, 7, 1, 0))
    colr = box_bytes(b"colr", struct.pack(">BBBI", 1, 0, 0, 16 if len(channel_info) >= 3 else 17))
    tmp = out_path + ".partial"
    try:
        with open(tmp, "wb") as dst:
            dst.write(JPX_PREAMBLE)
            dst.write(box_bytes(b"jp2h", ihdr + colr))
            for i, cs in enumerate(codestreams):
                dst.write(box_bytes(b"jpch", b""))
                dst.write(box_bytes(b"jplh", b""))
                if i == 0:
                    dst.write(box_bytes(b"uuid", XMP_UUID + make_xmp(n, channel_info, pixel_um)))
                dst.write(box_bytes(b"jp2c", cs() if callable(cs) else cs))
            for payload in extra_payloads:
                dst.write(box_bytes(b"uuid", payload))
        os.replace(tmp, out_path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


EXPORT_PARALLEL = 1   # sections rendered at once. Measured on 6 sections of a 17664 px canvas: 1 -> 12.8 s each, 2 -> 11.7 s,
                      # 3 -> 20.8 s (the encoder already uses every processor, and memory bandwidth is the limit)


def export_workers(stack, n_sections):
    """How many sections to render at once: EXPORT_PARALLEL, fewer if memory is short (a section takes about three canvases of
    the stack, plus the scene it is cut from)."""
    need = stack.canvas_w * stack.canvas_h * stack.channels * 2 * 3 + 2e9
    try:
        import psutil
        avail = psutil.virtual_memory().available
    except Exception:
        avail = 8e9
    return int(max(1, min(EXPORT_PARALLEL, n_sections, avail * 0.6 // need)))


def export_stack(stack, out_path, quality_db=DEFAULT_QUALITY_DB, rerender_all=False,
                 log=print, progress=None, stop_event=None):
    """Write `stack` with its edits to out_path. Sections whose edit matches
    the opened file are copied unchanged unless rerender_all (a stack made in memory from slide scans has no file to copy from:
    every section is rendered, and the JPX is written from scratch).
    progress(i, n, text) is called before each section."""
    t_export = time.time()
    fresh = not stack.jpx_path
    frames = [] if fresh else [b for b in read_boxes(stack.jpx_path) if b.kind == b"jp2c"]
    # (a section whose source image cannot be read, e.g. a .czi, is never re-rendered: its frame is copied as it is)
    todo = [s for s in stack.sections if s.renderable and (fresh or rerender_all or s.dirty)]
    missing = [s.name for s in todo if not s.source_path]
    if missing:
        raise FileNotFoundError("Source images not found: " + ", ".join(missing))
    if fresh and len(todo) != len(stack.sections):
        raise ValueError("Some sections can't be rendered from their source images.")
    codestreams = []
    n_render = 0
    workers = export_workers(stack, len(todo))
    threads = max(2, (os.cpu_count() or 4) // workers)   # (each encode gets its share of the processors)

    def render(s):
        """Render and encode one section (in a worker thread: several sections go at once, and the single-threaded parts of one
        overlap with the encoding of another)."""
        if stop_event is not None and stop_event.is_set():
            raise UserCanceled
        if progress is not None:
            progress(s.index, len(stack.sections), s.name)
        t0 = time.time()
        img = render_section(s, stack)
        cs = encode_frame(img, quality_db, threads)
        del img
        log(f"  {s.name}: re-rendered ({len(cs) / 1e6:.1f} MB, {time.time() - t0:.0f} s)")
        return cs

    with (contextlib.nullcontext() if fresh else open(stack.jpx_path, "rb")) as fj, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {id(s): pool.submit(render, s) for s in todo}
        try:
            for s in stack.sections:
                if stop_event is not None and stop_event.is_set():
                    raise UserCanceled
                if s not in todo:
                    if progress is not None:
                        progress(s.index, len(stack.sections), s.name)
                    codestreams.append(read_payload(fj, frames[s.frame]))
                    log(f"  {s.name}: unchanged, copied")
                    continue
                codestreams.append(jobs[id(s)].result())
                n_render += 1
        except BaseException:
            for j in jobs.values():
                j.cancel()
            raise
    doc = stack_json(stack, out_path)
    extra = [ALIGNER_UUID + json.dumps(doc, separators=(",", ":")).encode("utf-8")]
    if fresh:
        write_jpx_new(codestreams, out_path, stack.canvas_w, stack.canvas_h, stack.channel_info, stack.pixel_um, extra_payloads=extra)
    else:
        write_jpx_like(stack.jpx_path, codestreams, out_path, extra_payloads=extra,
                       pixel_um=stack.pixel_um if stack.pixel_corrected else None)
    elapsed = time.time() - t_export
    log(f"Wrote {out_path} ({os.path.getsize(out_path) / 1e6:.0f} MB; {n_render} re-rendered, "
        f"{len(stack.sections) - n_render} copied)")
    log(f"Export took {elapsed:.1f} s in total" + (f" ({elapsed / n_render:.1f} s per re-rendered section)" if n_render else "") + ".")
    return n_render


def next_free_name(folder, stem="reconstruction", ext=".jpx"):
    """reconstruction#.jpx with the lowest # that doesn't exist yet."""
    n = 1
    while os.path.exists(os.path.join(folder, f"{stem}{n}{ext}")):
        n += 1
    return f"{stem}{n}{ext}"


##################################################
#
#    Comparison with a reference frame
#
##################################################

def subpixel_shift(a, b):
    """Shift (dx, dy) in pixels that best aligns b onto a (phase correlation
    with a parabolic peak fit)."""
    a = a - a.mean()
    b = b - b.mean()
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    F = np.fft.fft2(a * win) * np.conj(np.fft.fft2(b * win))
    r = np.fft.ifft2(F / (np.abs(F) + 1e-9)).real
    py, px = np.unravel_index(np.argmax(r), r.shape)

    def refine(m1, c0, p1):
        d = m1 - 2 * c0 + p1
        return 0.0 if d == 0 else 0.5 * (m1 - p1) / d
    ny, nx = r.shape
    dy = py + refine(r[(py - 1) % ny, px], r[py, px], r[(py + 1) % ny, px])
    dx = px + refine(r[py, (px - 1) % nx], r[py, px], r[py, (px + 1) % nx])
    dy = dy - ny if dy > ny / 2 else dy
    dx = dx - nx if dx > nx / 2 else dx
    return dx, dy


def compare_frames(ours, theirs, crop=2048):
    """Agreement between two renders of the same section (H x W x C)."""
    a = ours.astype(np.float32).sum(-1)
    b = theirs.astype(np.float32).sum(-1)
    both = (a > 0) & (b > 0)
    corr = float(np.corrcoef(a[both][::7], b[both][::7])[0, 1])
    mad = float(np.abs(a[both] - b[both]).mean() / max(b[both].mean(), 1))
    h, w = a.shape
    step = crop // 2
    best, best_score = (0, 0), -1
    for y in range(0, max(h - crop, 1), step):
        for x in range(0, max(w - crop, 1), step):
            score = both[y:y + crop:16, x:x + crop:16].mean()
            if score > best_score:
                best, best_score = (y, x), score
    y, x = best
    dx, dy = subpixel_shift(b[y:y + crop, x:x + crop], a[y:y + crop, x:x + crop])
    return corr, mad, dx, dy


##################################################
#
#    Command line
#
##################################################

def cmd_info(args):
    st = open_stack(args.stack)
    kind = "aligner export" if st.from_aligner else "NeuroInfo reconstruction"
    print(f"{st.jpx_path} ({kind})\n  canvas {st.canvas_w}x{st.canvas_h}, {st.channels} channels "
          f"({', '.join(c.name for c in st.channel_info)}), {st.pixel_um} um/px, {len(st.sections)} sections")
    for s in st.sections:
        e = s.edit
        edit = "" if e.is_identity() else (f"  edit: rot {e.rotation_deg:+.2f}, shift ({e.dx_um:+.1f}, "
                                          f"{e.dy_um:+.1f}) um{', flipped' if e.flip else ''}")
        print(f"  {s.index}: {s.name:20s} rot {s.rotation_deg:+9.3f}  mirrored {str(s.mirrored):5s}  "
              f"source {'found' if s.source_path else 'MISSING'}{edit}")


def cmd_roundtrip(args):
    st = open_stack(args.stack)
    secs = st.sections if args.sections is None else [st.sections[i] for i in args.sections]
    missing = [s.name for s in secs if not s.source_path]
    if missing:
        sys.exit("Source images not found: " + ", ".join(missing))
    frames = [b for b in read_boxes(st.jpx_path) if b.kind == b"jp2c"]
    interp = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}[args.interp]
    print(f"Canvas {st.canvas_w}x{st.canvas_h}; rendering {len(secs)} section(s) with {args.interp} "
          f"interpolation, quality {args.quality} dB", flush=True)
    codestreams = []
    with open(st.jpx_path, "rb") as fj:
        for s in st.sections:
            if s not in secs:
                codestreams.append(read_payload(fj, frames[s.frame]))
                continue
            t0 = time.time()
            img = render_section(s, st, interpolation=interp)
            line = f"  {s.index}: {s.name:20s} rendered in {time.time() - t0:4.1f} s"
            if args.compare:
                theirs = imagecodecs.jpeg2k_decode(read_payload(fj, frames[s.frame]))
                corr, mad, dx, dy = compare_frames(img, theirs)
                del theirs
                line += (f" | vs original: corr {corr:.4f}, mean |diff| {100 * mad:4.1f}% of mean, "
                         f"offset ({dx:+.2f}, {dy:+.2f}) px")
            t1 = time.time()
            codestreams.append(encode_frame(img, args.quality))
            del img
            line += f" | encoded {len(codestreams[-1]) / 1e6:5.1f} MB in {time.time() - t1:4.1f} s"
            print(line, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    doc = stack_json(st, args.out)
    write_jpx_like(st.jpx_path, codestreams, args.out,
                   extra_payloads=[ALIGNER_UUID + json.dumps(doc, separators=(",", ":")).encode("utf-8")])
    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser(description="NeuroInfo serial section assembler stacks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="List the sections, their alignment and any edits")
    p.add_argument("stack", help="reconstruction.jpx (or its folder), or a JPX exported by jpx_aligner")
    p.set_defaults(func=cmd_info)
    p = sub.add_parser("roundtrip", help="Re-render the stack from the source images")
    p.add_argument("stack")
    p.add_argument("out", help="Output .jpx")
    p.add_argument("--compare", action="store_true", help="Compare each render with the original frame")
    p.add_argument("--sections", type=lambda x: [int(i) for i in x.split(",")],
                   help="Only re-render these stack indices (others are copied)")
    p.add_argument("--interp", choices=["linear", "cubic", "lanczos"], default="linear")
    p.add_argument("--quality", type=float, default=DEFAULT_QUALITY_DB,
                   help=f"Target PSNR in dB (default {DEFAULT_QUALITY_DB:g})")
    p.set_defaults(func=cmd_roundtrip)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
