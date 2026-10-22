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


def count_sections(path):
    """How many sections (outlines) a project file holds, or None if it can't be read."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if len(doc["slides"]) == 1 and is_jpx(doc["slides"][0]["path"]):       # (a JPX stack: its own sections)
            return doc.get("jpx_sections") or None
        return sum(len(s.get("polygons") or []) for s in doc["slides"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def sections_text(path):
    """What the recent-projects list shows as a project's number of sections: the number, or, for a JPX project that has no count yet (it was
    added but never got as far as the order), why not when its NeuroInfo files are missing ("No casefile", "No xml"), else blank."""
    n = count_sections(path)
    if n is not None:
        return n
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        slides = doc["slides"]
        if len(slides) != 1 or not is_jpx(slides[0]["path"]) or doc.get("alignment"):
            return ""
        jpx = slides[0]["path"]
        if not os.path.isabs(jpx):
            jpx = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), jpx))
        check = ns.neuroinfo_check(jpx)
        if check.get("state") != "missing":
            return ""
        return {("h5",): "No casefile", ("xml",): "No xml"}.get(tuple(check["missing"]), "No xml or casefile")
    except Exception:
        return ""


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
JPX_SUFFIXES = (".jpx",)                   # a stack of sections already (NeuroInfo's, or this tool's): a project of its own


def is_jp2(path):
    return path.lower().endswith(JP2_SUFFIXES)


def is_jpx(path):
    return path.lower().endswith(JPX_SUFFIXES)


def jpx_slide_info(path):
    """What step 1 shows for a JPX stack: its sections (frames), pixel size and channels."""
    boxes = ns.read_boxes(path)
    frames = [b for b in boxes if b.kind == b"jp2c"]
    if not frames:
        raise ValueError(f"{os.path.basename(path)} has no images in it.")
    px = 0.0
    with open(path, "rb") as f:
        _w, _h, nc = ns.codestream_size(f, frames[0])
        for b in boxes:
            if b.kind == b"uuid":
                data = ns.read_payload(f, b)
                if data[:16] == ns.XMP_UUID:
                    m = re.search(r'xmp:PixelWidth="([0-9.eE+-]+)"', data[16:].decode("utf-8", "replace"))
                    px = max(float(m.group(1)), 0.0) if m else 0.0
                    break
    chans = list(ns.read_xmp_channels(path)[:nc])
    chans += [ns.ChannelInfo(f"channel {i}", DEFAULT_COLORS[min(i, 3)]) for i in range(len(chans), nc)]
    return SlideInfo(path, len(frames), px, chans, os.path.getsize(path))


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
    if is_jpx(path):
        return jpx_slide_info(path)
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


NUMBERINGS = ("scene", "rows", "columns")      # how outlines are numbered: by scene number, across then down, down then across


def _grid_ids(boxes, axis):
    """For each scene, the number of its row (axis 1) or column (axis 0) in the grid the scenes lie in: scenes whose centres are within
    half a scene's size of each other along the axis share a row or column."""
    centres = [(b[0] + b[2] / 2, b[1] + b[3] / 2)[axis] for b in boxes]
    size = float(np.median([b[2 + axis] for b in boxes])) if boxes else 1.0
    ids, group, last = [0] * len(boxes), -1, None
    for i in sorted(range(len(boxes)), key=lambda i: centres[i]):
        if last is None or centres[i] - last > 0.5 * size:
            group += 1
        ids[i], last = group, centres[i]
    return ids


def sort_by_scene(polys, boxes, to_slide, mode="scene"):
    """The outlines in order, by `mode`: "scene" (the order of the scenes of the .czi they lie in, scene 1 first; outlines in one scene
    go top to bottom, left to right), "rows" (the scenes' rows from the top, and in a row from left to right) or "columns" (the
    columns from the left, each from top to bottom). `to_slide` turns an outline's points into slide pixels. Returns (sorted outlines,
    their scene indices)."""
    rows = _grid_ids(boxes, 1) if mode == "rows" else None
    cols = _grid_ids(boxes, 0) if mode == "columns" else None
    keyed = []
    for p in polys:
        c = polygon_centroid(to_slide(p))
        scene = scene_of(boxes, c[0], c[1])
        if mode == "rows":
            key = (rows[scene], c[0], c[1])
        elif mode == "columns":
            key = (cols[scene], c[1], c[0])
        else:
            key = (scene, round(c[1] / 300), c[0])
        keyed.append((key, scene, p))
    keyed.sort(key=lambda t: t[0])
    return [t[2] for t in keyed], [t[1] for t in keyed]


def split_polygon(p, min_steps=2, min_part=0.1):
    """Two outlines from one, cut along the chord between the pair of its vertices with the lowest ratio of straight-line distance to
    steps along the outline (the shorter way round). A neck, where the outline comes close to itself after a long way round, scores
    lowest; so does the bridge a merge leaves (a chord of no length), which this undoes. Pairs are tried from the lowest ratio up, and
    the first that leaves each part with at least `min_part` of the area (a bridge needs no such size) and `min_steps` + 1 vertices
    is used: without that the lowest ratios are of the tiny bumps where the outline's vertices crowd together. Returns (a, b), or None
    if no cut is possible."""
    p = np.asarray(p, float)
    n = len(p)
    if n < 2 * min_steps + 2:
        return None
    cands = []
    for i in range(n - 1):
        j = np.arange(i + min_steps, n)
        steps = np.minimum(j - i, n - (j - i))
        ok = steps >= min_steps
        if not ok.any():
            continue
        j, steps = j[ok], steps[ok]
        d = np.hypot(p[j, 0] - p[i, 0], p[j, 1] - p[i, 1])
        cands += [(float(dd / ss), float(dd), i, int(jj)) for dd, ss, jj in zip(d, steps, j)]
    cands.sort()
    total = abs(signed_area(p))
    for ratio, d, i, j in cands:
        a, b = p[i:j + 1], np.vstack([p[j:], p[:i + 1]])
        if d < 1e-6 or min(abs(signed_area(a)), abs(signed_area(b))) >= min_part * total:
            return a.copy(), b
    return None


def merge_group(polys):
    """One outline from several: it starts from the biggest and takes in, one at a time, the one nearest to what it has so far (each is
    joined along the two closest vertices, as merge_polygons does)."""
    rest = sorted(polys, key=lambda q: -abs(signed_area(q)))
    merged = rest.pop(0)
    while rest:
        near = min(range(len(rest)), key=lambda i: float(np.min(np.hypot(merged[:, None, 0] - rest[i][None, :, 0],
                                                                         merged[:, None, 1] - rest[i][None, :, 1]))))
        merged = merge_polygons(merged, rest.pop(near))
    return merged


def scene_groups(polys, boxes):
    """The scenes that hold more than one of the outlines: [(scene index, [positions in `polys`]), ...] in scene order."""
    by_scene = {}
    for i, q in enumerate(polys):
        c = polygon_centroid(q)
        by_scene.setdefault(scene_of(boxes, c[0], c[1]), []).append(i)
    return [(scene, idx) for scene, idx in sorted(by_scene.items()) if len(idx) > 1]


def merge_within_scenes(polys, boxes, mode="scene"):
    """The outlines (in slide pixels) with all those in one scene merged into one: (the new list in scene order, the number of scenes
    where that was done)."""
    by_scene = {}
    for q in polys:
        c = polygon_centroid(q)
        by_scene.setdefault(scene_of(boxes, c[0], c[1]), []).append(q)
    out, n = [], 0
    for scene in sorted(by_scene):
        group = by_scene[scene]
        if len(group) > 1:
            n += 1
        out.append(merge_group(group) if len(group) > 1 else group[0])
    return sort_by_scene(out, boxes, lambda q: q, mode)[0], n


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
    numbering: str = "rows"                        # how outlines are numbered: NUMBERINGS (new projects: across, then down)
    jpx_order: list = field(default_factory=list)  # a JPX project: the frames of its JPX in section order (empty: the JPX's own order)
    jpx_sections: int = 0                          # a JPX project: how many sections its JPX has (for the list of recent projects)

    @property
    def is_jpx(self):
        """A JPX project: one row, a JPX stack, no scans to find outlines on (its sections come with outlines, from NeuroInfo)."""
        return len(self.slides) == 1 and is_jpx(self.slides[0].path)

    @property
    def jpx_path(self):
        return self.slides[0].path if self.is_jpx else ""

    def polygons_key(self):
        """Something that changes when any slide's outlines do (the order is kept while this is the same)."""
        return [(s.path, [np.round(p, 1).tolist() for p in s.polygons]) for s in self.slides]

    def fingerprint(self):
        """A short code for the outlines and their order: the saved alignment belongs to these, and is dropped if they change."""
        if self.is_jpx:
            text = json.dumps(["jpx", os.path.basename(self.slides[0].path), self.jpx_order])
        else:
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
               "shared": self.shared, "numbering": self.numbering, "alignment": alignment, "alignment_key": self.alignment_key,
               "jpx_order": self.jpx_order, "jpx_sections": self.jpx_sections}
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
        proj.jpx_order = [int(v) for v in doc.get("jpx_order", [])]
        proj.jpx_sections = int(doc.get("jpx_sections", 0) or 0)
        # (a project saved before the choice existed was numbered by scene)
        proj.numbering = doc.get("numbering", "scene") if doc.get("numbering") in NUMBERINGS else "scene"
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


def projects_for_jpx(jpx_path):
    """The project files beside a JPX that are about it, most recently saved first. A project is named after its first image, so they are
    <name>_project.json, <name>_2_project.json ... for <name>.jpx (only those are read: a folder may hold many big project files)."""
    folder = os.path.dirname(os.path.abspath(jpx_path))
    stem = os.path.splitext(os.path.basename(jpx_path))[0]
    found = []
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    same = lambda a, b: os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
    for name in names:
        if not name.endswith(PROJECT_SUFFIX):
            continue
        rest = name[:-len(PROJECT_SUFFIX)]
        if rest != stem and not (rest.startswith(stem + "_") and rest[len(stem) + 1:].isdigit()):
            continue
        path = os.path.join(folder, name)
        try:
            proj = Project.load(path)
        except Exception:
            continue
        if proj.is_jpx and same(proj.jpx_path, jpx_path):
            found.append((os.path.getmtime(path), path))
    return [p for _, p in sorted(found, reverse=True)]


##################################################
#
#    From a project to a stack
#
##################################################

def outline_key(poly):
    """A short code for an outline (in slide pixels), to tell whether it is the same one later."""
    return hashlib.sha1(np.round(np.asarray(poly, float), 1).tobytes()).hexdigest()[:12]


def _norm(path):
    return os.path.normcase(os.path.abspath(path)) if path else ""


def polygon_index(project):
    """What polygon_of needs: ({(slide's path, code of the outline): (slide index, outline index)}, {code: [(slide index, outline index)]}).
    Outlines of different slides can be identical (images of one size, each outlined whole), so the slide takes part in telling them apart."""
    exact, loose = {}, {}
    for si, slide in enumerate(project.slides):
        for pi, poly in enumerate(slide.polygons):
            code = outline_key(poly)
            exact.setdefault((_norm(slide.path), code), (si, pi))
            loose.setdefault(code, []).append((si, pi))
    return exact, loose


def polygon_of(index, section):
    """(slide index, outline index) of the project's outline that a section was made from (by its source image and its origin_key), or None."""
    exact, loose = index
    hit = exact.get((_norm(section.source_path), section.origin_key))
    if hit is not None:
        return hit
    candidates = loose.get(section.origin_key, [])
    return candidates[0] if len(candidates) == 1 else None


PLACEHOLDER_SLIDE = -1     # in a project's order, an entry [-1, k] is a placeholder (a blank section, number k) instead of an outline [slide, outline]


def _order_items(project):
    """[(slide, outline in slide pixels, name)] of the project's sections in their order; a placeholder is (None, its number, "Placeholder")."""
    items = []
    for si, pi in (project.order or project.default_order()):
        if si == PLACEHOLDER_SLIDE:
            items.append((None, pi, "Placeholder"))
            continue
        slide = project.slides[si]
        items.append((slide, slide.polygons[pi], f"{os.path.splitext(os.path.basename(slide.path))[0]} #{pi + 1}"))   # (# is the outline's number in step 1, whatever order it ends up in)
    return items


def order_from_stack(project, stack):
    """The project's order (entries [slide, outline], [-1, k] for a placeholder) that the sections of `stack` are in now, so that what is done in
    the alignment tool (a placeholder inserted, sections moved or deleted) is what step 3 shows; None for a JPX project (which keeps frames),
    or when a section cannot be told (then the order stays as the project has it). Needs the outlines written back first
    (write_back_outlines), since a section is told by the code of the outline it came from."""
    if project.is_jpx:
        return None
    index = polygon_index(project)
    order = []
    for sec in stack.sections:
        if sec.placeholder:
            if sec.frame < ns.PLACEHOLDER_FRAME_BASE:
                return None
            order.append((PLACEHOLDER_SLIDE, sec.frame - ns.PLACEHOLDER_FRAME_BASE))
        else:
            hit = polygon_of(index, sec) if sec.origin_key else None
            if hit is None:
                return None
            order.append(hit)
    return order or None


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
        if slide is None:
            continue                                         # (a placeholder: nothing to match)
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
    return assign, dropped, [n for n in range(len(items)) if n not in assign and items[n][0] is not None]


def apply_jpx_order(stack, frames):
    """Put the sections of a JPX project's stack in the order of `frames` (frame numbers of the JPX). A section not named keeps its
    relative place, after the named ones."""
    if not frames:
        return
    named = set(frames)
    for f in frames:                 # (a placeholder inserted in step 3: a new, blank frame, not in the JPX yet)
        if f >= ns.PLACEHOLDER_FRAME_BASE and not any(s.frame == f for s in stack.sections):
            stack.sections.append(ns.placeholder_section(len(stack.sections), frame=f))
    by_frame = {s.frame: s for s in stack.sections}
    # The frames named are the sections: one not named was deleted (in step 3), and a placeholder not named is gone.
    stack.deleted += sum(1 for s in stack.sections if s.frame not in named and not ns.is_new_placeholder(s))
    stack.sections[:] = [by_frame[f] for f in frames if f in by_frame]
    for i, s in enumerate(stack.sections):
        s.index = i


def open_jpx_stack(project, locate=None):
    """The Stack of a JPX project, ready for the alignment tool: its JPX opened (with the project's saved alignment, if it has one),
    in the project's order."""
    stack = ns.open_stack(project.jpx_path, locate=locate, alignment=project.alignment)
    apply_jpx_order(stack, project.jpx_order)
    if project.z_spacing_um > 0:
        stack.z_spacing_um = project.z_spacing_um
    return stack


_pixel_um_of_file = {}     # slide path -> the file's own pixel size (read once)


def slide_pixel_um(slide):
    """The pixel size (um) step 1 uses for a slide: the one set for it, else the file's own (0 if unknown)."""
    if slide.pixel_um:
        return float(slide.pixel_um)
    if slide.path not in _pixel_um_of_file:
        _pixel_um_of_file[slide.path] = float(slide_info(slide.path).pixel_um or 0.0)
    return _pixel_um_of_file[slide.path]


def write_back_outlines(project, stack):
    """Outlines edited in the alignment tool become the project's own outlines, so that step 1 shows them too (there is then one version of
    each outline, the current one). Each section of a scan-based project came from one of the project's polygons, which its `origin_key`
    names; where the section's outline is no longer that polygon (converted to um as build_stack does), the polygon is replaced by the
    section's outline in slide pixels (rounded as the project file rounds them), and the section's origin_key becomes the new polygon's.
    A section whose outline was deleted keeps its polygon (a polygon is what makes a section). Returns how many polygons were replaced;
    the caller then saves the project (its alignment_key must be made again: the fingerprint changes with the outlines)."""
    if project.is_jpx:
        return 0
    index = polygon_index(project)
    replaced = 0
    for sec in stack.sections:
        hit = polygon_of(index, sec) if sec.origin_key else None
        P = np.asarray(sec.case_points, float).reshape(-1, 2)
        if hit is None or len(P) < 3:
            continue
        slide, pi = project.slides[hit[0]], hit[1]
        k = slide_pixel_um(slide)
        if k <= 0:
            continue
        old = np.asarray(slide.polygons[pi], float)
        was = np.column_stack([old[:, 0] * k, -old[:, 1] * k])           # (what build_stack made of the polygon)
        if was.shape == P.shape and np.allclose(was, P, atol=1e-3):
            continue
        new = np.round(np.column_stack([P[:, 0] / k, -P[:, 1] / k]), 2)
        slide.polygons[pi] = new
        sec.origin_key = outline_key(new)
        replaced += 1
    return replaced


def reuse_info(project, stack):
    """What the alignment window made for `project` (its `stack`) keeps, to tell later whether it can still stand for the project: see
    reuse_order."""
    if project.is_jpx:
        return {"jpx": os.path.abspath(project.jpx_path)}
    outlines = [(slide, np.asarray(p, float).copy()) for slide, p, _ in _order_items(project) if slide is not None]
    polys = [p for _, p in outlines]
    return {"snap": polys, "keys": [outline_key(p) for p in polys], "paths": [_norm(slide.path) for slide, _ in outlines],
            "pixel_um": stack.pixel_um}


def reuse_order(info, project, live_stack, new_stack):
    """The sections of the live alignment window's stack, in the order the project asks for now, if that stack still stands for the
    project (so the window can be kept as it is, with its pictures, edits, deleted sections and view); None if the project has changed in a
    way that needs the stack made anew: another JPX, outlines redrawn (compared on their points, to within the rounding of the project
    file), another pixel size. `new_stack` is the stack the loader just made from the project (its pixel size is compared)."""
    if project.is_jpx:
        if info.get("jpx") != os.path.abspath(project.jpx_path):
            return None
        by_frame = {s.frame: s for s in live_stack.sections}
        if any(f not in by_frame for f in project.jpx_order):
            return None                  # (a placeholder inserted since: this window's stack lacks it, so it is made again)
        return [by_frame[f] for f in project.jpx_order] or list(live_stack.sections)
    if "snap" not in info or abs(new_stack.pixel_um - info["pixel_um"]) > 1e-9:
        return None
    snap, keys = info["snap"], info["keys"]
    items = _order_items(project)
    outlines = [item for item in items if item[0] is not None]       # (the placeholders are found by their number below)
    if len(outlines) != len(snap):
        return None
    paths = info.get("paths") or [None] * len(snap)
    cents = np.array([p.mean(0) for p in snap]) if snap else np.zeros((0, 2))
    taken, chosen = set(), []
    for slide, poly, _ in outlines:
        poly = np.asarray(poly, float)
        match = None
        for j in np.flatnonzero(np.abs(cents - poly.mean(0)).max(axis=1) < 1.0):
            if int(j) not in taken and paths[j] in (None, _norm(slide.path)) and snap[j].shape == poly.shape \
                    and np.allclose(snap[j], poly, atol=0.02):
                match = int(j)
                break
        if match is None:
            return None
        taken.add(match)
        chosen.append(match)
    by_key = {(_norm(s.source_path), s.origin_key): s for s in live_stack.sections if s.origin_key}
    live_placeholders = {s.frame: s for s in live_stack.sections if s.placeholder}
    ordered, next_outline = [], 0
    for slide, number, _ in items:
        if slide is None:
            section = live_placeholders.get(ns.PLACEHOLDER_FRAME_BASE + number)
            if section is None:
                return None                  # (a placeholder inserted in step 3 since: this window's stack lacks it, so it is made again)
            ordered.append(section)
        else:
            j = chosen[next_outline]
            next_outline += 1
            if (paths[j] or _norm(slide.path), keys[j]) in by_key:
                ordered.append(by_key[(paths[j] or _norm(slide.path), keys[j])])
    return ordered or None


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
    if not any(slide is not None for slide, _, _ in items):
        raise ValueError("There are no section outlines.")
    # section outlines in the scan's own um (y up, as NeuroInfo keeps them), whatever the slide's pixel size
    pts = []
    for slide, poly, _ in items:
        if slide is None:
            pts.append(None)                         # (a placeholder has no outline)
            continue
        k = px_of(slide)
        pts.append(np.column_stack([poly[:, 0] * k, -poly[:, 1] * k]))
    extent = max(max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) for p in pts if p is not None)
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
        if slide is None:                            # (a placeholder: its blank frame is written when the stack is exported)
            stack.sections.append(ns.placeholder_section(n, frame=ns.PLACEHOLDER_FRAME_BASE + poly))
            continue
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
