"""The data side of step 1 of the workflow (find the sections on Zeiss .czi slide scans): detecting section outlines from a
brightness threshold, merging outlines, the project (slides, outlines, order) with its file, and turning a project into the
in-memory Stack that the alignment tool (jpx_aligner) opens. No windows here; see section_loader for those."""
import hashlib
import json
import math
import os
import re
import struct
from dataclasses import dataclass, field

import cv2
import imagecodecs
import numpy as np

import neuroinfo_stack as ns

PROJECT_FORMAT = "section-project"
PROJECT_VERSION = 1
PROJECT_SUFFIX = "_project.json"             # the name of a project file ends like this, so that it is not taken for some other .json
OLD_PROJECT_SUFFIX = ".sections.json"       # what project files were called at first: still opened
CACHE_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "JhouLab", "jpx_aligner_cache")   # (beside the aligner's)
OVERVIEW_PX = 2400          # the longest side of the slide picture that outlines are detected on (a pixel is ~15 um on a big slide)
OUTWARD_CUTOFF = 0.35       # smoothing keeps a pixel when this much of its neighbourhood is in the region (below 0.5: outward)
MAX_VERTICES = 80           # NeuroInfo's own outlines have 60-80 points
DEFAULT_COLORS = ((0, 160, 255), (0, 255, 91), (255, 1, 0), (255, 255, 255))


SETTINGS_FILE = os.path.join(os.path.dirname(CACHE_DIR), "jpx_aligner_settings.json")     # (the aligner's settings file: shared)
RECENT_PROJECTS_MAX = 20


def recent_projects():
    """The recently used project files, newest first (the settings file's "recent_projects"); [] if there are none."""
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        recent = data.get("recent_projects", []) if isinstance(data, dict) else []
        return [p for p in recent if isinstance(p, str)]
    except (OSError, ValueError):
        return []


def _save_recent_projects(paths):
    try:
        try:
            with open(SETTINGS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            data = data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            data = {}
        data["recent_projects"] = paths[:RECENT_PROJECTS_MAX]
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
    except OSError:
        pass            # (the list is only a convenience)


def add_recent_project(path):
    """Put a project file first in the recent list (once, whatever the letter case)."""
    key = os.path.normcase(os.path.abspath(path))
    _save_recent_projects([os.path.abspath(path)] + [p for p in recent_projects() if os.path.normcase(os.path.abspath(p)) != key])


def remove_recent_project(path):
    key = os.path.normcase(os.path.abspath(path))
    _save_recent_projects([p for p in recent_projects() if os.path.normcase(os.path.abspath(p)) != key])


def _relative(path, folder):
    """`path` relative to `folder`, with forward slashes (the absolute path only if there is no relative one, another drive)."""
    try:
        rel = os.path.relpath(path, folder)
    except ValueError:
        rel = os.path.abspath(path)
    return rel.replace("\\", "/")


def is_project_path(path):
    """Whether a file name is a project file's (the current ending, or the one used at first)."""
    return path.lower().endswith((PROJECT_SUFFIX, OLD_PROJECT_SUFFIX))


##################################################
#
#    Slides
#
##################################################

@dataclass
class SlideInfo:
    path: str
    scenes: int
    pixel_um: float
    channels: list          # ns.ChannelInfo for each channel
    size_bytes: int


JP2_SUFFIXES = (".jp2", ".j2k", ".j2c")    # single-image files: one section each (a .czi holds many)


def is_jp2(path):
    return path.lower().endswith(JP2_SUFFIXES)


def jp2_header(path):
    """(width, height, channels, pixel size in um or 0 if the file does not say, the image's bytes' offset and length) of a JPEG 2000
    section image. The pixel size is the one in NeuroInfo's XMP packet (xmp:PixelWidth), when there is one."""
    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(0)
        if head[:2] == b"\xff\x4f":      # a bare codestream (.j2k): no boxes, so no pixel size
            h = f.read(48)
            xs, ys, xo, yo = struct.unpack(">IIII", h[8:24])
            return xs - xo, ys - yo, struct.unpack(">H", h[40:42])[0], 0.0
        boxes = ns.read_boxes(path)
        frames = [b for b in boxes if b.kind == b"jp2c"]
        if len(frames) != 1:
            raise ValueError(f"{os.path.basename(path)} has {len(frames)} images in it; this takes files with one image each.")
        w, h, nc = ns.codestream_size(f, frames[0])
        px = 0.0
        for b in boxes:
            if b.kind == b"uuid":
                data = ns.read_payload(f, b)
                if data[:16] == ns.XMP_UUID:
                    m = re.search(r'xmp:PixelWidth="([0-9.eE+-]+)"', data[16:].decode("utf-8", "replace"))
                    px = float(m.group(1)) if m else 0.0
                    if px < 0:
                        px = 0.0
                    break
    return w, h, nc, px


def scene_boxes(path):
    """The scenes of a slide as (x, y, width, height) in its pixel grid: a .czi's, or the whole of a single-image file."""
    if is_jp2(path):
        w, h = jp2_header(path)[:2]
        return [(0, 0, w, h)]
    return ns.czi_scenes(path)


def slide_info(path):
    """What step 1 needs to know about a slide (.czi) or section image (.jp2): its scenes, pixel size (um; 0 if a .jp2 does not
    say) and channels."""
    if is_jp2(path):
        w, h, nc, px = jp2_header(path)
        chans = [c for c in ns.read_xmp_channels(path)[:nc]] if not path.lower().endswith((".j2k", ".j2c")) else []
        chans += [ns.ChannelInfo(f"channel {i}", DEFAULT_COLORS[min(i, 3)] if nc > 1 else (255, 255, 255)) for i in range(len(chans), nc)]
        return SlideInfo(path, 1, px, chans, os.path.getsize(path))
    boxes = ns.czi_scenes(path)
    with ns._czi_lock_for(path):
        cz = ns.czi_file(path)
        md = cz.metadata() if callable(getattr(cz, "metadata", None)) else cz.metadata
    md = md if isinstance(md, str) else str(md)
    m = re.search(r'<Distance Id="X">\s*<Value>([^<]+)</Value>', md)
    pixel_um = float(m.group(1)) * 1e6 if m else 0.0
    channels = []
    for idx, name, body in re.findall(r'<Channel Id="Channel:(\d+)" Name="([^"]*)"[^>]*>(.*?)</Channel>', md, re.S):
        idx = int(idx)
        if idx < len(channels):
            continue                       # (the same channels are listed again further down the metadata)
        color = re.search(r"<Color>#?([0-9A-Fa-f]{6,8})</Color>", body)
        rgb = tuple(int(color.group(1)[-6:][i:i + 2], 16) for i in (0, 2, 4)) if color else DEFAULT_COLORS[min(idx, 3)]
        channels.append(ns.ChannelInfo(name or f"channel {idx}", rgb))
    if not channels:
        channels = [ns.ChannelInfo(f"channel {i}", c) for i, c in enumerate(DEFAULT_COLORS[:3])]
    return SlideInfo(path, len(boxes), pixel_um or 0.65, channels, os.path.getsize(path))


def compose(chans, colors):
    """(gray, RGB) 8-bit pictures from a float32 H x W x C stack of channels: the gray is the stretched mean, the RGB has each
    channel stretched on its own and shown in its color."""
    mean = chans.mean(2)
    hi = max(float(np.percentile(mean[mean > 0], 99.5)) if (mean > 0).any() else 1.0, 1.0)
    gray = (np.clip(mean / hi, 0, 1) ** 0.6 * 255).astype(np.uint8)
    rgb = np.zeros(mean.shape + (3,), np.float32)
    for c in range(chans.shape[2]):
        v = chans[:, :, c]
        hi_c = max(float(np.percentile(v[v > 0], 99.5)) if (v > 0).any() else 1.0, 1.0)
        rgb += (np.clip(v / hi_c, 0, 1) ** 0.6)[:, :, None] * (np.array(colors[c % len(colors)], np.float32) / 255)
    return gray, (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _overview_cache_file(path, max_px):
    st = os.stat(path)
    key = f"overview1|{os.path.abspath(path).lower()}|{st.st_size}|{st.st_mtime_ns}|{max_px}"
    return os.path.join(CACHE_DIR, "ov_" + hashlib.sha1(key.encode()).hexdigest() + ".npz")


def overview(path, max_px=OVERVIEW_PX, color=False):
    """A picture of the whole slide for finding sections on: (8-bit gray image, (x0, y0, k), RGB or None) where slide pixel (X, Y) is
    at ((X - x0) / k, (Y - y0) / k). The RGB picture (the channels in their colors) costs half as much again to make for a .czi, so
    it comes only when `color` asks for it (a .jp2 always gives it: it is made from the same decode). Pictures are kept on disk, so
    that the second time is quick."""
    cache = None
    try:
        cache = _overview_cache_file(path, max_px)
        with np.load(cache) as d:
            rgb = d["rgb"] if "rgb" in d.files else None
            if rgb is not None or not color:
                return d["gray"], tuple(float(v) for v in d["geo"]), rgb
            gray, geo = d["gray"], tuple(float(v) for v in d["geo"])
    except Exception:
        gray = geo = None            # (no cache file, or a damaged one)
    info = slide_info(path)
    colors = [c.color for c in info.channels]
    rgb = None
    if gray is not None:             # (the gray picture was cached; only the color is missing)
        rgb = ns.czi_slide_thumbnail(path, max_px, colors=colors)[2]
    elif is_jp2(path):
        w, h = jp2_header(path)[:2]
        r = max(0, int(math.ceil(math.log2(max(w, h) / max_px))))     # (the picture is decoded at 1/2**r: quick, whatever the size)
        with open(path, "rb") as f:
            img = imagecodecs.jpeg2k_decode(f.read(), skipres=r)
        chans = (img[:, :, None] if img.ndim == 2 else img).astype(np.float32)
        gray, rgb = compose(chans, colors)
        geo = (0.0, 0.0, float(2 ** r))
    elif color:
        gray, geo, rgb = ns.czi_slide_thumbnail(path, max_px, colors=colors)
    else:
        gray, geo = ns.czi_slide_thumbnail(path, max_px)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = cache + ".partial.npz"
        np.savez(tmp, gray=gray, geo=np.array(geo, float), **({"rgb": rgb} if rgb is not None else {}))
        os.replace(tmp, cache)
    except Exception:
        pass                          # (the cache is only a convenience)
    return gray, tuple(float(v) for v in geo), rgb


##################################################
#
#    Detecting outlines
#
##################################################

def auto_threshold(img):
    """A starting brightness threshold (Otsu's), for the user to adjust."""
    t, _ = cv2.threshold(cv2.GaussianBlur(img, (0, 0), 1.5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t)


def fill_holes(mask):
    """The mask with the holes inside its regions filled (as the outlines go round the outside of each region)."""
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, found, -1, 1, cv2.FILLED)
    return out


def detect_mask(img, threshold, min_area, smooth=0, pad=0):
    """Pixels brighter than `threshold` (after a slight blur), without the regions smaller than `min_area` pixels: 0/1, uint8.
    `smooth` (pixels) evens out the boundary, outward by preference: notches and inlets narrower than 2 * smooth are filled in
    first (a closing, which only adds pixels), then the boundary is rounded with a blur whose cut-off lies below the middle, so
    that the boundary settles outside of where it was rather than inside. `pad` (pixels) then grows each region outward by that much."""
    mask = (cv2.GaussianBlur(img, (0, 0), 1.5) > threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(min_area, 1)
    mask = keep[labels]
    if smooth > 0 or pad > 0:
        # Region by region, so that smoothing never joins two regions (the sections of a slide can lie only a few hundred um
        # apart): what a region gains stays nearer to it than to any other, by a margin.
        k = 2 * int(smooth) + 1
        margin = 2 * k + int(pad)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = mask.copy()
        h, w = mask.shape
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        for i in range(1, n):
            x, y, bw, bh, _a = stats[i]
            x0, y0, x1, y1 = max(x - margin, 0), max(y - margin, 0), min(x + bw + margin, w), min(y + bh + margin, h)
            mine = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
            others = ((labels[y0:y1, x0:x1] != i) & (labels[y0:y1, x0:x1] > 0)).astype(np.uint8)
            if smooth > 0:
                padded = cv2.copyMakeBorder(mine, k, k, k, k, cv2.BORDER_CONSTANT, value=0)
                closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)
                rounded = (cv2.GaussianBlur(closed.astype(np.float32), (0, 0), 0.7 * smooth) > OUTWARD_CUTOFF)[k:-k, k:-k]
            else:
                rounded = mine.astype(bool)
            if pad > 0:
                rounded = cv2.dilate(rounded.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(pad) + 1,) * 2)).astype(bool)
            if others.any():   # (only where this region is nearer than any other, by a margin: so grown regions stay apart)
                near = cv2.distanceTransform(1 - mine, cv2.DIST_L2, 3) < cv2.distanceTransform(1 - others, cv2.DIST_L2, 3) - 2
                grown = rounded.astype(np.uint8) & near.astype(np.uint8)
            else:
                grown = rounded.astype(np.uint8)
            out[y0:y1, x0:x1] |= grown | mine
        mask = out
    return mask


def mask_to_polygons(mask, max_vertices=MAX_VERTICES):
    """The outer boundary of each region of the mask, as polygons (n x 2, x then y, in mask pixels) of at most max_vertices
    points. Only outer boundaries: holes inside a region (ventricles) are not outlined."""
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in found:
        if len(c) < 3:
            continue
        eps = 0.5
        approx = c
        while len(approx) > max_vertices:
            approx = cv2.approxPolyDP(c, eps, True)
            eps *= 1.25
        pts = approx.reshape(-1, 2).astype(float)
        if len(pts) >= 3:
            out.append(pts)
    out.sort(key=lambda p: (round(p[:, 1].mean() / 400), p[:, 0].mean()))   # (reading order, roughly)
    return out


def scene_of(boxes, x, y):
    """Index of the scene a slide point (x, y) is in: the box that holds it (the nearest centre if several do), else the box
    with the nearest centre."""
    centres = np.array([(bx + bw / 2, by + bh / 2) for bx, by, bw, bh in boxes])
    inside = [i for i, (bx, by, bw, bh) in enumerate(boxes) if bx <= x < bx + bw and by <= y < by + bh]
    cand = inside or list(range(len(boxes)))
    return min(cand, key=lambda i: float(np.hypot(*(centres[i] - (x, y)))))


def sort_by_scene(polys, boxes, to_slide):
    """The outlines in the order of the scenes of the .czi they lie in (scene 1 first); outlines in one scene go top to bottom,
    left to right. `to_slide` turns an outline's points into slide pixels. Returns (sorted outlines, their scene indices)."""
    keyed = []
    for p in polys:
        c = polygon_centroid(to_slide(p))
        keyed.append((scene_of(boxes, c[0], c[1]), round(c[1] / 300), c[0], p))
    keyed.sort(key=lambda t: t[:3])
    return [t[3] for t in keyed], [t[0] for t in keyed]


def signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def merge_polygons(a, b):
    """One polygon from two: they are joined along their two closest vertices (the outline of the first runs up to its vertex,
    crosses to the other's, goes right round it, and comes back)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if signed_area(a) < 0:
        a = a[::-1]
    if signed_area(b) < 0:
        b = b[::-1]
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
    return np.vstack([a[:i + 1], np.roll(b, -j, axis=0), b[j:j + 1], a[i:]])


def polygon_centroid(p):
    """Area centroid (the mean of the vertices if the area is nil)."""
    x, y = p[:, 0], p[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cr.sum() / 2
    if abs(a) < 1e-9:
        return p.mean(0)
    return np.array([((x + np.roll(x, -1)) * cr).sum(), ((y + np.roll(y, -1)) * cr).sum()]) / (6 * a)


##################################################
#
#    The project
#
##################################################

@dataclass
class Slide:
    path: str
    threshold: int = -1               # brightness threshold used (-1: not set yet)
    min_area: int = -1                # smallest region kept, in overview pixels (-1: not set yet: 0.25 mm2 when first shown)
    smooth: int = 8                   # how much the boundary is smoothed (pixels; outward by preference)
    pad: int = 2                      # how far each outline is grown outward (pixels)
    own: dict = None                  # this image's own settings, kept while "same settings for all images" is on (to go back to)
    pixel_um: float = 0.0             # pixel size set by the user (0: the file's own)
    polygons: list = field(default_factory=list)   # outlines in the slide's pixel grid (n x 2, x then y), once accepted
    accepted: bool = False


@dataclass
class Project:
    slides: list = field(default_factory=list)
    order: list = field(default_factory=list)      # [(slide index, polygon index)] in section order (empty: not ordered yet)
    path: str = ""                                 # the project file ("" until saved)
    z_spacing_um: float = 0.0                      # distance between adjacent sections (um); NeuroInfo asks for it too
    alignment: dict = None                         # the aligner's state (stack_json): its edits, outline changes, order, canvas
    alignment_key: str = ""                        # fingerprint() of the outlines and order that alignment was made from
    shared: dict = None                            # the settings every image shares ("same settings for all images"), or None

    def polygons_key(self):
        """Something that changes when any slide's outlines do (the order is kept while this is the same)."""
        return [(s.path, [np.round(p, 1).tolist() for p in s.polygons]) for s in self.slides]

    def fingerprint(self):
        """A short code for the outlines and their order: the saved alignment belongs to these, and is dropped if they change."""
        text = json.dumps([self.polygons_key(), self.order or self.default_order()])
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def default_order(self):
        """Slide by slide, each in reading order (the order the outlines were found in)."""
        return [(i, j) for i, s in enumerate(self.slides) for j in range(len(s.polygons))]

    def save(self, path=None):
        path = path or self.path
        if not path:
            raise ValueError("no project file name")
        folder = os.path.dirname(os.path.abspath(path))
        alignment = self.alignment
        if alignment:      # (paths in the project file are relative to its folder only: drives may be mapped differently elsewhere)
            alignment = dict(alignment, origin="", sections=[{k: v for k, v in sec.items() if k != "source_absolute"}
                                                             for sec in alignment["sections"]])
        doc = {"format": PROJECT_FORMAT, "version": PROJECT_VERSION,
               "slides": [{"path": _relative(s.path, folder), "threshold": s.threshold, "min_area": s.min_area,
                           "smooth": s.smooth, "pad": s.pad, "own": s.own, "pixel_um": s.pixel_um, "accepted": s.accepted, "polygons": [np.round(p, 2).tolist() for p in s.polygons]}
                          for s in self.slides],
               "order": [list(o) for o in self.order], "z_spacing_um": self.z_spacing_um,
               "shared": self.shared, "alignment": alignment, "alignment_key": self.alignment_key}
        tmp = path + ".partial"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, separators=(",", ":"))
        os.replace(tmp, path)
        self.path = path

    def folder(self):
        """The folder the project lives in: its file's, or (not saved yet) its first image's; "" if there is neither."""
        if self.path:
            return os.path.dirname(os.path.abspath(self.path))
        return os.path.dirname(os.path.abspath(self.slides[0].path)) if self.slides else ""

    @staticmethod
    def load(path):
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("format") != PROJECT_FORMAT or doc.get("version", 0) > PROJECT_VERSION:
            raise ValueError("This is not a project file of this tool (or it is from a newer version).")
        slides = []
        for d in doc["slides"]:
            p = d["path"]
            here = os.path.dirname(os.path.abspath(path))
            if not os.path.isabs(p):       # (relative to the project file's folder, as projects are saved now)
                p = os.path.normpath(os.path.join(here, p))
            elif not os.path.isfile(p):    # (an older project's absolute path, from another machine or place: look beside the project file)
                alt = os.path.join(here, os.path.basename(p))
                p = alt if os.path.isfile(alt) else p
            slides.append(Slide(p, threshold=d.get("threshold", -1), min_area=d.get("min_area", -1), smooth=d.get("smooth", 8), pad=d.get("pad", 2), own=d.get("own"), pixel_um=d.get("pixel_um", 0.0),
                                polygons=[np.array(q, float).reshape(-1, 2) for q in d["polygons"]],
                                accepted=d.get("accepted", False)))
        proj = Project(slides, [tuple(o) for o in doc.get("order", [])], os.path.abspath(path), float(doc.get("z_spacing_um", 0.0)),
                       doc.get("alignment"), doc.get("alignment_key", ""))
        proj.shared = doc.get("shared")
        return proj

    def new_path(self):
        """The file name for a project not saved yet: <first image>_project.json, or ..._2_project.json, ..._3_project.json if that name
        is taken (by a project that was not opened here: one that was opened keeps its own name and is written to)."""
        base = self.default_path()
        stem = base[:-len(PROJECT_SUFFIX)]
        path, n = base, 1
        while os.path.exists(path):
            n += 1
            path = f"{stem}_{n}{PROJECT_SUFFIX}"
        return path

    def default_path(self):
        first = self.slides[0].path
        return os.path.join(os.path.dirname(os.path.abspath(first)), os.path.splitext(os.path.basename(first))[0] + PROJECT_SUFFIX)


##################################################
#
#    From a project to a stack
#
##################################################

def outline_key(poly):
    """A short code for an outline (in slide pixels), to tell whether it is the same one later."""
    return hashlib.sha1(np.round(np.asarray(poly, float), 1).tobytes()).hexdigest()[:12]


def _order_items(project):
    """[(slide, outline in slide pixels, name)] of the project's sections in their order."""
    items = []
    for si, pi in (project.order or project.default_order()):
        slide = project.slides[si]
        items.append((slide, slide.polygons[pi], f"{os.path.splitext(os.path.basename(slide.path))[0]} #{pi + 1}"))   # (# is the outline's number in step 1, whatever order it ends up in)
    return items


def match_alignment(project):
    """Which section of the project's saved alignment each of its sections now is. The alignment is of the images, so it still holds when
    an outline is redrawn (another threshold, padding, a merge), or the sections are put in another order: a transform of the section's
    image is as good for the new outline as for the old. Outlines are matched by how much they overlap on the same slide.
    Returns (assign, dropped, new): {position in the new order: index in the alignment's sections}, the indices of alignment sections
    with nothing to go with (their outline was deleted, or merged into another), and the positions of sections with no alignment yet."""
    old = project.alignment
    items = _order_items(project)
    if not old:
        return {}, [], list(range(len(items)))
    old_px = float(old.get("pixel_um") or 1.0)
    olds = []
    for o in old["sections"]:
        P = np.array(o["case_points"], float).reshape(-1, 2)
        base = os.path.basename((o.get("source_relative") or o.get("source_absolute") or o.get("name", "")).replace("\\", "/")).lower()
        olds.append((base, np.column_stack([P[:, 0] / old_px, -P[:, 1] / old_px]) if len(P) else None))
    assign, taken = {}, set()
    groups = {}
    for n, (slide, poly, _) in enumerate(items):
        groups.setdefault(os.path.basename(slide.path).lower(), []).append(n)
    for base, news in groups.items():
        cand = [i for i, (b, p) in enumerate(olds) if p is not None and (b == base or b.startswith(os.path.splitext(base)[0]))]
        if not cand:
            continue
        polys = [items[n][1] for n in news] + [olds[i][1] for i in cand]
        lo = np.min([p.min(0) for p in polys], axis=0)
        hi = np.max([p.max(0) for p in polys], axis=0)
        scale = min(1.0, 1200 / max(float((hi - lo).max()), 1.0))
        shape = (int(np.ceil((hi[1] - lo[1]) * scale)) + 2, int(np.ceil((hi[0] - lo[0]) * scale)) + 2)

        def raster(p):
            m = np.zeros(shape, np.uint8)
            cv2.fillPoly(m, [np.round((p - lo) * scale).astype(np.int32)], 1)
            return m.astype(bool)

        mn = [raster(items[n][1]) for n in news]
        mo = [raster(olds[i][1]) for i in cand]
        pairs = []
        for a, n in enumerate(news):
            for b, i in enumerate(cand):
                inter = int((mn[a] & mo[b]).sum())
                if inter and inter >= 0.5 * min(int(mn[a].sum()), int(mo[b].sum())):
                    pairs.append((inter, n, i))
        for inter, n, i in sorted(pairs, reverse=True):        # (the biggest overlaps first; each used once)
            if n not in assign and i not in taken:
                assign[n] = i
                taken.add(i)
    dropped = [i for i in range(len(olds)) if i not in taken]
    return assign, dropped, [n for n in range(len(items)) if n not in assign]


def build_stack(project, margin=0.2):
    """The in-memory Stack of a project, ready for the alignment tool: one section per outline, in order, each rendered from its
    slide scan (so no JPX is needed) and placed unrotated at the middle of a canvas big enough for the largest of them. If the project
    holds an alignment made from these same outlines and order, that is what comes back, exactly as it was left (including the order and
    deletions made in the aligner). If the outlines or order have changed since, every section that matches one of the alignment's sections
    (see match_alignment) comes back placed as it was, on the same canvas; the others start at the middle."""
    if project.alignment and project.alignment_key == project.fingerprint():
        stack = ns.stack_from_json(project.alignment, os.path.dirname(os.path.abspath(project.path or project.slides[0].path)))
        stack.channel_info = list(slide_info(project.slides[0].path).channels)
        stack.z_spacing_um = stack.z_spacing_um or project.z_spacing_um
        return stack
    infos = {}
    for s in project.slides:
        if s.path not in infos:
            infos[s.path] = slide_info(s.path)
    first = infos[project.slides[0].path]

    def px_of(slide):
        px = slide.pixel_um or infos[slide.path].pixel_um
        if px <= 0:
            raise ValueError(f"The pixel size of {os.path.basename(slide.path)} isn't known: set it in step 1.")
        return px

    px = px_of(project.slides[0])
    for s in project.slides:
        if abs(px_of(s) / px - 1) > 0.01:
            raise ValueError(f"{os.path.basename(s.path)} has a pixel size of {px_of(s):g} um, but {os.path.basename(project.slides[0].path)} "
                             f"has {px:g} um. Sections are aligned in one stack, so the pixel sizes must match.")
    if any(len(infos[s.path].channels) != len(first.channels) for s in project.slides):
        raise ValueError("The slides have different numbers of channels, so their sections can't go in one stack.")
    items = _order_items(project)
    if not items:
        raise ValueError("There are no section outlines.")
    # section outlines in the scan's own um (y up, as NeuroInfo keeps them), whatever the slide's pixel size
    pts = []
    for slide, poly, _ in items:
        k = px_of(slide)
        pts.append(np.column_stack([poly[:, 0] * k, -poly[:, 1] * k]))
    extent = max(max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) for p in pts)
    old = project.alignment
    assign = match_alignment(project)[0] if old else {}
    if old:
        cw, ch = old["canvas"]                              # (the canvas the alignment was made on)
    else:
        cw = ch = int(math.ceil(extent * (1 + 2 * margin) / px / 16) * 16)
    stack = ns.Stack("", cw, ch, len(first.channels), px)
    stack.channel_info = list(first.channels)
    stack.z_spacing_um = (old or {}).get("z_spacing_um") or project.z_spacing_um
    stack.unsaved = True
    centre = np.array([cw / 2 * px, -ch / 2 * px])
    for n, ((slide, poly, name), p) in enumerate(zip(items, pts)):
        c = polygon_centroid(p)
        sec = ns.Section(n, name, slide.path, p, np.eye(2), centre - c, pivot_src=c,
                         czi_origin=None if is_jp2(slide.path) else (0.0, 0.0), origin_key=outline_key(poly))
        if n in assign:
            o = old["sections"][assign[n]]
            # the image keeps the placement it was given: same base transform, same edit about the same pivot (a point of the image)
            sec.R, sec.t = np.array(o["base_R"], float), np.array(o["base_t"], float)
            sec.edit, sec.saved_edit = ns.Edit(**o["edit"]), ns.Edit(**o["edit"])
            if o.get("pivot_src") is not None:
                sec.pivot_src = np.array(o["pivot_src"], float)
            if o.get("origin_key") and o["origin_key"] == sec.origin_key:     # (the outline is the one step 1 gave it, as it was: keep it as it is
                sec.case_points = np.array(o["case_points"], float).reshape(-1, 2)    # now, with any changes made to it in the aligner)
                sec.saved_points = sec.case_points.copy()
        stack.sections.append(sec)
    return stack
