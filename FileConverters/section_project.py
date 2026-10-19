"""The data side of step 1 of the workflow (find the sections on Zeiss .czi slide scans): detecting section outlines from a
brightness threshold, merging outlines, the project (slides, outlines, order) with its file, and turning a project into the
in-memory Stack that the alignment tool (jpx_aligner) opens. No windows here; see section_loader for those."""
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
PROJECT_SUFFIX = ".sections.json"
OVERVIEW_PX = 2400          # the longest side of the slide picture that outlines are detected on (a pixel is ~15 um on a big slide)
OUTWARD_CUTOFF = 0.35       # smoothing keeps a pixel when this much of its neighbourhood is in the region (below 0.5: outward)
MAX_VERTICES = 80           # NeuroInfo's own outlines have 60-80 points
DEFAULT_COLORS = ((0, 160, 255), (0, 255, 91), (255, 1, 0), (255, 255, 255))


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


def overview(path, max_px=OVERVIEW_PX):
    """A picture of the whole slide for finding sections on: (8-bit gray image, (x0, y0, k), the same in the channels' colors
    as an 8-bit RGB image) where slide pixel (X, Y) is at ((X - x0) / k, (Y - y0) / k)."""
    info = slide_info(path)
    colors = [c.color for c in info.channels]
    if is_jp2(path):
        w, h = jp2_header(path)[:2]
        r = max(0, int(math.ceil(math.log2(max(w, h) / max_px))))     # (the picture is decoded at 1/2**r: quick, whatever the size)
        with open(path, "rb") as f:
            img = imagecodecs.jpeg2k_decode(f.read(), skipres=r)
        chans = (img[:, :, None] if img.ndim == 2 else img).astype(np.float32)
        gray, rgb = compose(chans, colors)
        return gray, (0, 0, float(2 ** r)), rgb
    return ns.czi_slide_thumbnail(path, max_px, colors=colors)


##################################################
#
#    Detecting outlines
#
##################################################

def auto_threshold(img):
    """A starting brightness threshold (Otsu's), for the user to adjust."""
    t, _ = cv2.threshold(cv2.GaussianBlur(img, (0, 0), 1.5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t)


def detect_mask(img, threshold, min_area, smooth=0):
    """Pixels brighter than `threshold` (after a slight blur), without the regions smaller than `min_area` pixels: 0/1, uint8.
    `smooth` (pixels) evens out the boundary, outward by preference: notches and inlets narrower than 2 * smooth are filled in
    first (a closing, which only adds pixels), then the boundary is rounded with a blur whose cut-off lies below the middle, so
    that the boundary settles outside of where it was rather than inside."""
    mask = (cv2.GaussianBlur(img, (0, 0), 1.5) > threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(min_area, 1)
    mask = keep[labels]
    if smooth > 0:
        # Region by region, so that smoothing never joins two regions (the sections of a slide can lie only a few hundred um
        # apart): what a region gains stays nearer to it than to any other, by a margin.
        k = 2 * int(smooth) + 1
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = mask.copy()
        h, w = mask.shape
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        for i in range(1, n):
            x, y, bw, bh, _a = stats[i]
            x0, y0, x1, y1 = max(x - 2 * k, 0), max(y - 2 * k, 0), min(x + bw + 2 * k, w), min(y + bh + 2 * k, h)
            mine = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
            others = ((labels[y0:y1, x0:x1] != i) & (labels[y0:y1, x0:x1] > 0)).astype(np.uint8)
            padded = cv2.copyMakeBorder(mine, k, k, k, k, cv2.BORDER_CONSTANT, value=0)
            closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)
            rounded = (cv2.GaussianBlur(closed.astype(np.float32), (0, 0), 0.7 * smooth) > OUTWARD_CUTOFF)[k:-k, k:-k]
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
    min_area: int = 200               # smallest region kept, in overview pixels
    smooth: int = 8                   # how much the boundary is smoothed (pixels; outward by preference)
    pixel_um: float = 0.0             # pixel size set by the user (0: the file's own)
    polygons: list = field(default_factory=list)   # outlines in the slide's pixel grid (n x 2, x then y), once accepted
    accepted: bool = False


@dataclass
class Project:
    slides: list = field(default_factory=list)
    order: list = field(default_factory=list)      # [(slide index, polygon index)] in section order (empty: not ordered yet)
    path: str = ""                                 # the project file ("" until saved)
    z_spacing_um: float = 0.0                      # distance between adjacent sections (um); NeuroInfo asks for it too

    def polygons_key(self):
        """Something that changes when any slide's outlines do (the order is kept while this is the same)."""
        return [(s.path, [np.round(p, 1).tolist() for p in s.polygons]) for s in self.slides]

    def default_order(self):
        """Slide by slide, each in reading order (the order the outlines were found in)."""
        return [(i, j) for i, s in enumerate(self.slides) for j in range(len(s.polygons))]

    def save(self, path=None):
        path = path or self.path
        if not path:
            raise ValueError("no project file name")
        doc = {"format": PROJECT_FORMAT, "version": PROJECT_VERSION,
               "slides": [{"path": s.path.replace("\\", "/"), "threshold": s.threshold, "min_area": s.min_area,
                           "smooth": s.smooth, "pixel_um": s.pixel_um, "accepted": s.accepted, "polygons": [np.round(p, 2).tolist() for p in s.polygons]}
                          for s in self.slides],
               "order": [list(o) for o in self.order], "z_spacing_um": self.z_spacing_um}
        tmp = path + ".partial"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, separators=(",", ":"))
        os.replace(tmp, path)
        self.path = path

    @staticmethod
    def load(path):
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("format") != PROJECT_FORMAT or doc.get("version", 0) > PROJECT_VERSION:
            raise ValueError("This is not a project file of this tool (or it is from a newer version).")
        slides = []
        for d in doc["slides"]:
            p = d["path"]
            if not os.path.isfile(p):      # (the slide may have moved with the project file: look beside it)
                alt = os.path.join(os.path.dirname(os.path.abspath(path)), os.path.basename(p))
                p = alt if os.path.isfile(alt) else p
            slides.append(Slide(p, threshold=d.get("threshold", -1), min_area=d.get("min_area", 200), smooth=d.get("smooth", 8), pixel_um=d.get("pixel_um", 0.0),
                                polygons=[np.array(q, float).reshape(-1, 2) for q in d["polygons"]],
                                accepted=d.get("accepted", False)))
        proj = Project(slides, [tuple(o) for o in doc.get("order", [])], os.path.abspath(path), float(doc.get("z_spacing_um", 0.0)))
        return proj

    def default_path(self):
        first = self.slides[0].path
        return os.path.join(os.path.dirname(os.path.abspath(first)), os.path.splitext(os.path.basename(first))[0] + PROJECT_SUFFIX)


##################################################
#
#    From a project to a stack
#
##################################################

def build_stack(project, margin=0.2):
    """The in-memory Stack of a project, ready for the alignment tool: one section per outline, in order, each rendered from its
    slide scan (so no JPX is needed) and placed unrotated at the middle of a canvas big enough for the largest of them."""
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
    order = project.order or project.default_order()
    items = []
    counts = {}
    for si, pi in order:
        slide = project.slides[si]
        poly = slide.polygons[pi]
        counts[si] = counts.get(si, 0) + 1
        items.append((slide, poly, f"{os.path.splitext(os.path.basename(slide.path))[0]} #{counts[si]}"))
    if not items:
        raise ValueError("There are no section outlines.")
    # section outlines in the scan's own um (y up, as NeuroInfo keeps them), whatever the slide's pixel size
    pts = []
    for slide, poly, _ in items:
        k = px_of(slide)
        pts.append(np.column_stack([poly[:, 0] * k, -poly[:, 1] * k]))
    extent = max(max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) for p in pts)
    side = int(math.ceil(extent * (1 + 2 * margin) / px / 16) * 16)
    stack = ns.Stack("", side, side, len(first.channels), px)
    stack.channel_info = list(first.channels)
    stack.z_spacing_um = project.z_spacing_um
    stack.unsaved = True
    centre = np.array([side / 2 * px, -side / 2 * px])
    for n, ((slide, poly, name), p) in enumerate(zip(items, pts)):
        c = polygon_centroid(p)
        sec = ns.Section(n, name, slide.path, p, np.eye(2), centre - c, czi_origin=None if is_jp2(slide.path) else (0.0, 0.0))
        stack.sections.append(sec)
    return stack
