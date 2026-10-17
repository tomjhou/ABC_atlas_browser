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
import copy
import datetime
import importlib.util
import json
import os
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict

REQUIRED_PACKAGES = {"numpy": "numpy", "h5py": "h5py", "imagecodecs": "imagecodecs",
                     "cv2": "opencv-python"}


def ensure_packages(required=None):
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
        sys.exit(f"Couldn't install {', '.join(still)}; install them with pip and try again.")


ensure_packages()

import cv2  # noqa: E402
import h5py  # noqa: E402
import imagecodecs  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_QUALITY_DB = 60.0
XMP_UUID = bytes.fromhex("be7acfcb97a942e89c71999491e3afac")
# Private box holding this tool's JSON description of the stack
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

    def __post_init__(self):
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

    @property
    def folder(self):
        return os.path.dirname(os.path.abspath(self.jpx_path))

    @property
    def dirty(self):
        return any(s.dirty for s in self.sections)

    @property
    def changed(self):
        return any(s.changed for s in self.sections)

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
        # NeuroInfo case files may record another computer's drive letter
        m = re.match(r"^[A-Za-z]:[\\/](.*)$", absolute)
        if m:
            for drive in "DCEFGHIJ":
                candidates.append(f"{drive}:/{m.group(1)}")
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.normpath(c)
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


def _load_neuroinfo(jpx_path):
    folder = os.path.dirname(os.path.abspath(jpx_path))
    files = {n.lower(): n for n in os.listdir(folder)}
    stem = os.path.splitext(os.path.basename(jpx_path))[0].lower()
    xml_name = files.get(stem + ".xml") or files.get("reconstruction.xml")
    h5_name = files.get("casefile.h5")
    if not xml_name or not h5_name:
        raise FileNotFoundError(
            "This JPX has no alignment data of its own, and the NeuroInfo files needed to recover it "
            f"(reconstruction.xml and CaseFile.H5) aren't next to it in\n{folder}")
    xml = open(os.path.join(folder, xml_name), encoding="latin-1").read()
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
    stack = Stack(jpx_path, w, h, nc, pixel_um, origin=os.path.abspath(jpx_path))

    with h5py.File(os.path.join(folder, h5_name), "r") as f:
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
            Q = stack_outlines.get(guid.upper())
            if Q is None or len(Q) != len(P):
                raise ValueError(f"{name}: outline in {xml_name} missing or doesn't match CaseFile.H5")
            R, t, err = fit_rigid(P, Q)
            if err > 1.0:
                raise ValueError(f"{name}: outlines don't fit a rigid transform (error {err:.2f} um)")
            stack.sections.append(Section(i, name, resolve_source(folder, name, absolute=recorded),
                                          P, R, t, source_abs_hint=recorded))
    if len(stack.sections) != len(frames):
        raise ValueError(f"{len(stack.sections)} sections in CaseFile.H5 but {len(frames)} frames in the JPX")
    return stack


def _load_aligner(jpx_path, doc):
    if doc.get("format") != JSON_FORMAT or doc.get("version", 0) > JSON_VERSION:
        raise ValueError("Unrecognized alignment data in this JPX (made by a newer version?)")
    folder = os.path.dirname(os.path.abspath(jpx_path))
    stack = Stack(jpx_path, doc["canvas"][0], doc["canvas"][1], doc["channels"], doc["pixel_um"],
                  origin=doc.get("origin", ""), from_aligner=True)
    for i, d in enumerate(doc["sections"]):
        e = Edit(**d["edit"])
        stack.sections.append(Section(
            i, d["name"], resolve_source(folder, d["name"], d.get("source_relative", ""), d.get("source_absolute", "")),
            np.array(d["case_points"], float).reshape(-1, 2), np.array(d["base_R"], float), np.array(d["base_t"], float),
            edit=copy.copy(e), saved_edit=copy.copy(e), source_abs_hint=d.get("source_absolute", ""),
            pivot_src=d.get("pivot_src")))
    frames = [b for b in read_boxes(jpx_path) if b.kind == b"jp2c"]
    if len(stack.sections) != len(frames):
        raise ValueError("Alignment data doesn't match the number of images in this JPX")
    return stack


def open_stack(path):
    """Open a JPX written by this tool, or a NeuroInfo reconstruction (a .jpx
    with reconstruction.xml and CaseFile.H5 beside it, or that folder)."""
    if os.path.isdir(path):
        names = {n.lower(): n for n in os.listdir(path)}
        if "reconstruction.jpx" not in names:
            raise FileNotFoundError(f"No reconstruction.jpx in {path}")
        path = os.path.join(path, names["reconstruction.jpx"])
    doc = find_aligner_json(path)
    stack = _load_aligner(path, doc) if doc else _load_neuroinfo(path)
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
    # Version 2 only when needed (an outline was edited or removed, so the pivot is no longer the outline's centroid);
    # otherwise the file stays readable by the first version of this tool.
    needs_v2 = any(not len(s.case_points) or not np.allclose(s.pivot_src, s.case_points.mean(0), atol=1e-4)
                   for s in stack.sections)
    return {
        "format": JSON_FORMAT, "version": 2 if needs_v2 else 1,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "origin": stack.origin, "pixel_um": stack.pixel_um,
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


def render_section(section, stack, source=None, interpolation=cv2.INTER_LINEAR, use_edit=True):
    """Full-resolution section image on the stack canvas (H x W x C)."""
    if source is None:
        if not section.source_path:
            raise FileNotFoundError(f"source image for {section.name} not found")
        source = imagecodecs.imread(section.source_path)
    if source.ndim == 2:
        source = source[..., None]
    R, t = section.effective() if use_edit else (section.R, section.t)
    M = canvas_to_source_matrix(R, t, stack.pixel_um)
    out = np.zeros((stack.canvas_h, stack.canvas_w, source.shape[2]), source.dtype)
    for c in range(source.shape[2]):
        out[..., c] = cv2.warpAffine(source[..., c], M, (stack.canvas_w, stack.canvas_h),
                                     flags=interpolation | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


def encode_frame(image, quality_db=DEFAULT_QUALITY_DB):
    cs = imagecodecs.jpeg2k_encode(image, quality_db, codecformat="j2k", mct=False,
                                   resolutions=7, numthreads=os.cpu_count() or 1)
    return add_com_markers(cs, ["MBF Bioscience"])


def write_jpx_like(template_path, codestreams, out_path, extra_payloads=()):
    """Copy template_path's boxes verbatim, replacing its jp2c codestreams in
    order with `codestreams` (bytes, or callables returning bytes), dropping
    any old aligner box and appending `extra_payloads` as uuid boxes."""
    boxes = read_boxes(template_path)
    frames = [b for b in boxes if b.kind == b"jp2c"]
    if len(frames) != len(codestreams):
        raise ValueError(f"template has {len(frames)} frames, got {len(codestreams)} codestreams")
    tmp = out_path + ".partial"
    i = 0
    try:
        with open(template_path, "rb") as src, open(tmp, "wb") as dst:
            for b in boxes:
                if b.kind == b"jp2c":
                    cs = codestreams[i]
                    dst.write(box_bytes(b"jp2c", cs() if callable(cs) else cs))
                    i += 1
                    continue
                if b.kind == b"uuid" and read_payload(src, b, 16) == ALIGNER_UUID:
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


def export_stack(stack, out_path, quality_db=DEFAULT_QUALITY_DB, rerender_all=False,
                 log=print, progress=None, stop_event=None):
    """Write `stack` with its edits to out_path. Sections whose edit matches
    the opened file are copied unchanged unless rerender_all.
    progress(i, n, text) is called before each section."""
    frames = [b for b in read_boxes(stack.jpx_path) if b.kind == b"jp2c"]
    todo = [s for s in stack.sections if rerender_all or s.dirty]
    missing = [s.name for s in todo if not s.source_path]
    if missing:
        raise FileNotFoundError("Source images not found: " + ", ".join(missing))
    codestreams = []
    n_render = 0
    with open(stack.jpx_path, "rb") as fj:
        for s in stack.sections:
            if stop_event is not None and stop_event.is_set():
                raise UserCanceled
            if progress is not None:
                progress(s.index, len(stack.sections), s.name)
            if s not in todo:
                codestreams.append(read_payload(fj, frames[s.index]))
                log(f"  {s.name}: unchanged, copied")
                continue
            t0 = time.time()
            img = render_section(s, stack)
            cs = encode_frame(img, quality_db)
            del img
            codestreams.append(cs)
            n_render += 1
            log(f"  {s.name}: re-rendered ({len(cs) / 1e6:.1f} MB, {time.time() - t0:.0f} s)")
    doc = stack_json(stack, out_path)
    write_jpx_like(stack.jpx_path, codestreams, out_path,
                   extra_payloads=[ALIGNER_UUID + json.dumps(doc, separators=(",", ":")).encode("utf-8")])
    log(f"Wrote {out_path} ({os.path.getsize(out_path) / 1e6:.0f} MB; {n_render} re-rendered, "
        f"{len(stack.sections) - n_render} copied)")
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
                codestreams.append(read_payload(fj, frames[s.index]))
                continue
            t0 = time.time()
            img = render_section(s, st, interpolation=interp)
            line = f"  {s.index}: {s.name:20s} rendered in {time.time() - t0:4.1f} s"
            if args.compare:
                theirs = imagecodecs.jpeg2k_decode(read_payload(fj, frames[s.index]))
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
