"""A Tk version of the atlas browser's single-section window (first version: the picture, smooth zoom and pan, the hover readout and
a choice of what to color the cells by).

Why Tk: the matplotlib window re-renders its whole figure for every zoom, pan, hover and control change, and its widgets are
artists in that figure. Here the picture is one image on a Tk canvas (a crop of the section's picture, resized to the view: the cost
follows the number of pixels on screen, not the number of cells), the hover ring is a canvas item, and the controls are ordinary
Tk widgets in a side panel, so changing them redraws nothing but themselves.

The data and rendering come from abc_core.py (the same code the matplotlib window uses). Run it directly:

    python abc_tk_picker.py [section number or label]
"""
import json
import math
import os
import sys
import threading
import queue
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import abc_core as core

ACCENT = "#1f5fa8"
HOVER_RADIUS_MM = 0.010            # a cell is "under the pointer" within 10 um (the matplotlib window's rule too)
LEVELS = ("class", "subclass", "supertype", "cluster")


class ZoomView:
    """A picture on a tk.Canvas: wheel zoom about the pointer, right- or middle-drag to pan. It starts at the fit view (the whole
    picture in the window), never zooms out past that, and never pans the picture off the window's edges."""

    MAX_SCREEN_PX_PER_PX = 40.0        # most zoomed in: one picture pixel is this many screen pixels

    def __init__(self, parent, bg="black"):
        self.idle_cursor = "crosshair"       # the pointer when not panning (a window that picks whole things can choose another)
        self.canvas = tk.Canvas(parent, bg=bg, highlightthickness=0, cursor=self.idle_cursor)
        self.src = None                # H x W x 3 uint8
        self.levels = []               # the picture and, for a big one, halved copies of it (zoomed out, a smaller one is resized: faster)
        self.scale, self.ox, self.oy = 1.0, 0.0, 0.0     # screen px per picture px; where picture pixel (0, 0) is on the canvas
        self.at_fit = True             # still the starting view: it follows the window's size
        self.photo = None
        self.item = self.canvas.create_image(0, 0, anchor="nw", tags="picture")
        self.pending = None
        self.listeners = []            # called (no arguments) after the view changed: overlays reposition themselves
        self.post_draw = None          # fn(out, x0, y0, scale) -> out: draws on the resized picture (the cell marks) before it is shown
        self.last_draw_ms = 0.0
        self.pan_from = None
        c = self.canvas
        c.bind("<Configure>", lambda e: self.on_resize())
        c.bind("<MouseWheel>", self.on_wheel)
        c.bind("<Button-4>", lambda e: self.zoom_at(e.x, e.y, 1.25))        # (other platforms' wheels)
        c.bind("<Button-5>", lambda e: self.zoom_at(e.x, e.y, 0.8))
        for press, motion, release in (("<ButtonPress-3>", "<B3-Motion>", "<ButtonRelease-3>"),
                                       ("<ButtonPress-2>", "<B2-Motion>", "<ButtonRelease-2>")):
            c.bind(press, self.pan_start)
            c.bind(motion, self.pan_move)
            c.bind(release, self.pan_end)
        c.bind("<Key-Home>", lambda e: self.fit())

    # ---- the picture

    def set_image(self, rgb, keep_view=True):
        """Show a new picture (H x W x 3 or 4, uint8). A picture of the same size keeps the view as it is."""
        rgb = np.ascontiguousarray(rgb[..., :3])
        same = self.src is not None and self.src.shape == rgb.shape
        self.src = rgb
        self.levels = [rgb]
        while self.levels[-1].shape[0] * self.levels[-1].shape[1] > 4_000_000 and min(self.levels[-1].shape[:2]) > 800:
            last = self.levels[-1]
            self.levels.append(cv2.resize(last, (last.shape[1] // 2, last.shape[0] // 2), interpolation=cv2.INTER_AREA))
        if not (same and keep_view):
            self.fit()
        else:
            self.request_draw()

    def canvas_size(self):
        return max(self.canvas.winfo_width(), 2), max(self.canvas.winfo_height(), 2)

    def fit_scale(self):
        h, w = self.src.shape[:2]
        cw, ch = self.canvas_size()
        return min(cw / w, ch / h)

    def fit(self):
        """The starting view: the whole picture, centered."""
        if self.src is None:
            return
        self.at_fit = True
        self.scale = self.fit_scale()
        self.clamp()
        self.request_draw()

    def clamp(self):
        """The view limits: not out past the fit view, and the picture's edges never come inside the window (a picture smaller than the
        window along an axis is centered on it)."""
        h, w = self.src.shape[:2]
        cw, ch = self.canvas_size()
        self.scale = min(max(self.scale, self.fit_scale()), self.MAX_SCREEN_PX_PER_PX)
        for axis, size, extent in (("x", w, cw), ("y", h, ch)):
            shown = size * self.scale
            o = self.ox if axis == "x" else self.oy
            o = (extent - shown) / 2 if shown <= extent else min(0.0, max(extent - shown, o))
            if axis == "x":
                self.ox = o
            else:
                self.oy = o

    def on_resize(self):
        if self.src is None:
            return
        if self.at_fit:
            self.fit()
        else:
            self.clamp()
            self.request_draw()

    # ---- coordinates

    def to_canvas(self, ix, iy):
        return self.ox + ix * self.scale, self.oy + iy * self.scale

    def to_picture(self, cx, cy):
        return (cx - self.ox) / self.scale, (cy - self.oy) / self.scale

    # ---- zoom and pan

    def on_wheel(self, e):
        self.zoom_at(e.x, e.y, 1.25 if e.delta > 0 else 0.8)

    def zoom_at(self, cx, cy, factor):
        if self.src is None:
            return
        ix, iy = self.to_picture(cx, cy)
        before = self.scale
        self.scale = min(max(self.scale * factor, self.fit_scale()), self.MAX_SCREEN_PX_PER_PX)
        if self.scale == before:
            return
        self.at_fit = abs(self.scale - self.fit_scale()) < 1e-9
        self.ox, self.oy = cx - ix * self.scale, cy - iy * self.scale       # (the point under the pointer stays under it)
        self.clamp()
        self.request_draw()

    def pan_start(self, e):
        self.pan_from = (e.x, e.y, self.ox, self.oy)
        self.canvas.config(cursor="fleur")

    def pan_move(self, e):
        if self.pan_from is None or self.src is None:
            return
        x0, y0, ox, oy = self.pan_from
        self.ox, self.oy = ox + e.x - x0, oy + e.y - y0
        self.at_fit = False
        self.clamp()
        self.request_draw()

    def pan_end(self, e):
        self.pan_from = None
        self.canvas.config(cursor=self.idle_cursor)

    # ---- drawing

    def request_draw(self):
        """Redraws are coalesced: a burst of wheel or drag events draws once, when the window is idle."""
        if self.pending is None:
            self.pending = self.canvas.after_idle(self.draw)

    def draw(self):
        self.pending = None
        if self.src is None:
            return
        t0 = time.perf_counter()
        cw, ch = self.canvas_size()
        s = self.scale
        # zoomed out far, a halved copy of the picture (a quarter, an eighth...) is resized instead of the whole: the cost stays small
        k = 0 if s >= 1 else min(int(math.floor(math.log2(1.0 / s))), len(self.levels) - 1)
        f = 2 ** k                                    # picture pixels per pixel of this level
        lv = self.levels[k]
        hk, wk = lv.shape[:2]
        sk = s * f                                    # screen pixels per pixel of this level
        x0, y0 = max(int(math.floor(-self.ox / sk)), 0), max(int(math.floor(-self.oy / sk)), 0)
        x1, y1 = min(int(math.ceil((cw - self.ox) / sk)), wk), min(int(math.ceil((ch - self.oy) / sk)), hk)
        if x1 <= x0 or y1 <= y0:
            return
        crop = lv[y0:y1, x0:x1]
        size = (max(int(round((x1 - x0) * sk)), 1), max(int(round((y1 - y0) * sk)), 1))
        # zoomed out: area averaging (smooth, no shimmer); zoomed in: nearest, so each cell stays a crisp block of its own color
        if k > 0:
            interp = cv2.INTER_LINEAR if sk >= 1 else cv2.INTER_AREA
        else:
            interp = cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST
        out = cv2.resize(crop, size, interpolation=interp)
        if self.post_draw is not None:
            out = self.post_draw(out, x0 * f, y0 * f, s)
        self.photo = ImageTk.PhotoImage(Image.fromarray(out), master=self.canvas)
        self.canvas.itemconfig(self.item, image=self.photo)
        self.canvas.coords(self.item, round(self.ox + x0 * sk), round(self.oy + y0 * sk))
        self.canvas.tag_raise("overlay")
        self.last_draw_ms = (time.perf_counter() - t0) * 1000
        for fn in self.listeners:
            fn()


class SectionData:
    """What the window needs about one section: its pictures and layout, and its cells (positions, labels) for the hover."""

    def __init__(self, label, rgba, layout, cells):
        self.label, self.rgba, self.layout, self.cells = label, rgba, layout, cells
        self.img_h, self.img_w = rgba.shape[:2]
        self.xs = cells["x"].to_numpy(float)
        self.ys = cells["y"].to_numpy(float)
        self.rows = None                                 # each cell's row in the dataset (set by load: for reading its expression)
        px, py_up = core.map_data_to_pixel(layout, self.xs, self.ys)
        self.cell_px, self.cell_py = np.asarray(px, float), self.img_h - np.asarray(py_up, float)      # (each cell's place in the picture)
        self.pictures = {"class": rgba}                  # level -> RGBA picture, as they are made

    @staticmethod
    def load(adata, abc_cache, section_series, spatial, label, figsize=(15, 10)):
        layout = core.load_valid_section_roi_cache(label)
        if layout is None:
            core.generate_and_cache_section_image(adata, abc_cache, section_series, label, spatial, figsize)
            layout = core.load_valid_section_roi_cache(label)
        png, _json = core.section_roi_cache_paths(label)
        rgba = np.asarray(Image.open(png).convert("RGBA"))
        mask = (section_series == label).to_numpy()
        cells = spatial[mask]
        x = np.asarray(cells["x"], float)
        y = np.asarray(cells["y"], float)
        valid = ~(np.isnan(x) | np.isnan(y))
        data = SectionData(label, rgba, layout, cells[valid])
        data.rows = np.where(mask)[0][valid]
        return data


class RoiLayer:
    """The regions of interest on the picture: rectangles that are drawn with the left button, and then moved by their inside, resized by
    their handles, or deleted with a right-click. They are kept in the picture's own pixels (so they follow the view), and handed back in
    data units (mm) as the same list of actions the matplotlib window returns: {'action': 'add'|'edit'|'delete', ...}."""

    HANDLE = 5               # half the size of a handle square, in screen pixels
    HIT = 9                  # how close, in screen pixels, the pointer must be to a handle
    MIN_DRAG = 5             # a drag smaller than this is a stray click, not a new ROI
    COLOR = "#ff8c00"
    SELECTED_COLOR = "#e00000"   # the selected region (the last one clicked or drawn): the one the Delete key removes
    LINE_WIDTH = 4           # thickness of an ROI's dashed outline, in screen pixels (raise it to make the regions stand out more)
    DASH = (10, 5)
    HANDLES = [(hx, hy) for hy in (-1, 0, 1) for hx in (-1, 0, 1) if (hx, hy) != (0, 0)]   # (-1 = left/top edge, 0 = middle, 1 = right/bottom)
    CURSORS = {(-1, -1): "size_nw_se", (1, 1): "size_nw_se", (1, -1): "size_ne_sw", (-1, 1): "size_ne_sw",
               (-1, 0): "sb_h_double_arrow", (1, 0): "sb_h_double_arrow", (0, -1): "sb_v_double_arrow", (0, 1): "sb_v_double_arrow"}

    def __init__(self, window, existing=()):
        self.w, self.view, self.canvas = window, window.view, window.view.canvas
        self.rois = {}                 # id -> {"kind": "existing"|"new", "orig": index or None, "box": [x0, y0, x1, y1] in picture px, items}
        self.deleted = []              # indices of existing ROIs that were deleted
        self.selected = None           # id of the selected ROI
        self.next_id = 1
        self.drag = None
        self.right_press = None
        self.menu = tk.Menu(self.canvas, tearoff=0)
        self.on_change = None
        c = self.canvas
        c.bind("<ButtonPress-1>", self.press)
        c.bind("<B1-Motion>", self.move)
        c.bind("<ButtonRelease-1>", self.release)
        c.bind("<Delete>", lambda e: self.delete_selected())
        c.bind("<ButtonPress-3>", self.right_down, add="+")
        c.bind("<ButtonRelease-3>", self.right_up, add="+")
        c.bind("<Motion>", self.hover_cursor, add="+")
        self.view.listeners.append(self.layout)
        for orig, bounds in existing:
            self.add_from_data(bounds, "existing", orig)

    def reset(self, existing=()):
        """Another section is shown: its own regions replace these (the window has handed the actions of the old ones on already)."""
        for rid in list(self.rois):
            self.remove(rid, forget=False)
        self.deleted = []
        self.selected = None
        self.drag = self.right_press = None
        for orig, bounds in existing:
            self.add_from_data(bounds, "existing", orig)

    # ---- conversions: picture pixels <-> data (mm)

    def data_to_box(self, bounds):
        x_lo, x_hi, y_lo, y_hi = bounds
        d = self.w.data
        (ax, ay), (bx, by) = (core.map_data_to_pixel(d.layout, x_lo, y_lo), core.map_data_to_pixel(d.layout, x_hi, y_hi))
        ay, by = d.img_h - ay, d.img_h - by                    # (the layout counts y up from the bottom of the picture)
        return [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)]

    def box_to_data(self, box):
        d = self.w.data
        (x0, y0), (x1, y1) = (core.map_pixel_to_data(d.layout, box[0], d.img_h - box[1]), core.map_pixel_to_data(d.layout, box[2], d.img_h - box[3]))
        return (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))

    # ---- the rectangles

    def add_from_data(self, bounds, kind, orig=None):
        return self.add(self.data_to_box(bounds), kind, orig)

    def add(self, box, kind, orig=None):
        rid = self.next_id
        self.next_id += 1
        rect = self.canvas.create_rectangle(0, 0, 0, 0, outline=self.COLOR, width=self.LINE_WIDTH, dash=self.DASH, tags="overlay")
        handles = {h: self.canvas.create_rectangle(0, 0, 0, 0, outline="black", fill="white", tags="overlay") for h in self.HANDLES}
        self.rois[rid] = {"kind": kind, "orig": orig, "box": list(box), "rect": rect, "handles": handles}
        self.place(rid)
        self.changed()
        return rid

    def select(self, rid):
        """Make `rid` (or None) the selected ROI, shown in red."""
        if rid == self.selected:
            return
        for r, color in ((self.selected, self.COLOR), (rid, self.SELECTED_COLOR)):
            if r in self.rois:
                self.canvas.itemconfig(self.rois[r]["rect"], outline=color)
        self.selected = rid

    def delete_selected(self):
        if self.selected in self.rois:
            self.remove(self.selected)

    def remove(self, rid, forget=True):
        if rid == self.selected:
            self.selected = None
        roi = self.rois.pop(rid)
        self.canvas.delete(roi["rect"])
        for h in roi["handles"].values():
            self.canvas.delete(h)
        if forget and roi["kind"] == "existing":
            self.deleted.append(roi["orig"])
        self.changed()

    def changed(self):
        if self.on_change:
            self.on_change()

    def screen_box(self, rid):
        x0, y0, x1, y1 = self.rois[rid]["box"]
        (a, b), (c, d) = self.view.to_canvas(x0, y0), self.view.to_canvas(x1, y1)
        return a, b, c, d

    def handle_points(self, rid):
        a, b, c, d = self.screen_box(rid)
        xs, ys = {-1: a, 0: (a + c) / 2, 1: c}, {-1: b, 0: (b + d) / 2, 1: d}
        return {h: (xs[h[0]], ys[h[1]]) for h in self.HANDLES}

    def place(self, rid):
        roi = self.rois[rid]
        a, b, c, d = self.screen_box(rid)
        self.canvas.coords(roi["rect"], a, b, c, d)
        for h, (x, y) in self.handle_points(rid).items():
            self.canvas.coords(roi["handles"][h], x - self.HANDLE, y - self.HANDLE, x + self.HANDLE, y + self.HANDLE)
        self.canvas.tag_raise("overlay")

    def layout(self):
        for rid in self.rois:
            self.place(rid)

    # ---- hit testing

    def hit(self, cx, cy):
        """('handle', id, handle) for a handle within reach, else ('inside', id) for the topmost ROI under the pointer, else None."""
        for rid in reversed(list(self.rois)):
            for h, (x, y) in self.handle_points(rid).items():
                if abs(cx - x) <= self.HIT and abs(cy - y) <= self.HIT:
                    return ("handle", rid, h)
        for rid in reversed(list(self.rois)):
            a, b, c, d = self.screen_box(rid)
            if min(a, c) <= cx <= max(a, c) and min(b, d) <= cy <= max(b, d):
                return ("inside", rid)
        return None

    def hover_cursor(self, e):
        if self.drag is not None or self.view.pan_from is not None:
            return
        hit = self.hit(e.x, e.y)
        cursor = "crosshair" if hit is None else self.CURSORS[hit[2]] if hit[0] == "handle" else "fleur"
        if self.canvas.cget("cursor") != cursor:
            self.canvas.config(cursor=cursor)

    # ---- the left button: draw, move, resize

    def clamp_point(self, ix, iy):
        h, w = self.view.src.shape[:2]
        return min(max(ix, 0.0), float(w)), min(max(iy, 0.0), float(h))

    def press(self, e):
        self.canvas.focus_set()
        hit = self.hit(e.x, e.y)
        ix, iy = self.clamp_point(*self.view.to_picture(e.x, e.y))
        if hit and hit[0] == "handle":
            self.drag = {"mode": "handle", "id": hit[1], "handle": hit[2]}
        elif hit:
            self.drag = {"mode": "move", "id": hit[1], "from": (ix, iy), "box": list(self.rois[hit[1]]["box"])}
        else:
            self.drag = {"mode": "draw", "start": (ix, iy), "start_px": (e.x, e.y), "id": self.add([ix, iy, ix, iy], "new")}
        self.select(self.drag["id"])

    def move(self, e):
        d = self.drag
        if d is None:
            return
        ix, iy = self.clamp_point(*self.view.to_picture(e.x, e.y))
        roi = self.rois[d["id"]]
        box = roi["box"]
        if d["mode"] == "draw":
            sx, sy = d["start"]
            box[:] = [min(sx, ix), min(sy, iy), max(sx, ix), max(sy, iy)]
        elif d["mode"] == "move":
            fx, fy = d["from"]
            x0, y0, x1, y1 = d["box"]
            w, h = self.view.src.shape[1], self.view.src.shape[0]
            dx = min(max(ix - fx, -x0), w - x1)
            dy = min(max(iy - fy, -y0), h - y1)
            box[:] = [x0 + dx, y0 + dy, x1 + dx, y1 + dy]
        else:
            hx, hy = d["handle"]
            gap = 1.0 / self.view.scale                      # (an edge can't cross the opposite one: at least a screen pixel apart)
            if hx == -1:
                box[0] = min(ix, box[2] - gap)
            elif hx == 1:
                box[2] = max(ix, box[0] + gap)
            if hy == -1:
                box[1] = min(iy, box[3] - gap)
            elif hy == 1:
                box[3] = max(iy, box[1] + gap)
        self.place(d["id"])
        self.changed()

    def release(self, e):
        d, self.drag = self.drag, None
        if d is None:
            return
        if d["mode"] == "draw":
            x0, y0, x1, y1 = self.rois[d["id"]]["box"]
            if max(abs(x1 - x0), abs(y1 - y0)) * self.view.scale < self.MIN_DRAG:       # (too small to be meant: dropped)
                self.remove(d["id"], forget=False)
        self.changed()

    # ---- the right button: a click on an ROI offers Delete (a drag pans, see ZoomView)

    def right_down(self, e):
        self.right_press = (e.x, e.y)

    def right_up(self, e):
        start, self.right_press = self.right_press, None
        if start is None or abs(e.x - start[0]) + abs(e.y - start[1]) > 4:
            return
        hit = self.hit(e.x, e.y)
        if hit is None:
            return
        rid = hit[1]
        self.menu.delete(0, tk.END)
        self.menu.add_command(label="Delete ROI", command=lambda: self.remove(rid))
        try:
            self.menu.tk_popup(e.x_root, e.y_root)
        finally:
            self.menu.grab_release()

    # ---- the result

    def actions(self):
        """What the user did, in the matplotlib window's form: edits of the ROIs that were there, new ones, and deletions."""
        out = []
        for roi in self.rois.values():
            bounds = self.box_to_data(roi["box"])
            out.append({"action": "edit", "index": roi["orig"], "bounds": bounds} if roi["kind"] == "existing"
                       else {"action": "add", "bounds": bounds})
        out += [{"action": "delete", "index": i} for i in self.deleted]
        return out


class SectionWindow:
    """The single-section window: the picture, with its controls in a side panel.

    What the picture shows (the "Show" choice): Groups (the cells colored by the Group level), Gene (one gene's expression, or up to three
    as a red/green/blue overlay), Imputed Gene (the same, from the much larger imputed dataset), or Find group (one group's cells in
    gold, the rest dim). Linear/Log applies to the gene views. The Group level also says which cells get the red '+' markers when the
    pointer rests on a cell: those of the same group."""

    MARK_DELAY_MS = 500
    MARK_COLOR = (255, 0, 0)
    SCALE_UNITS = {("standard", "log"): "log1p(expr)", ("standard", "linear"): "expr (raw counts)",
                   ("imputed", "log"): "log2 (imputed)", ("imputed", "linear"): "linear (imputed)"}
    CHANNEL_COLORS = ("#e53935", "#43a047", "#4a6cff")        # the red, green and blue of a multi-gene overlay

    def __init__(self, data, abc_cache, spatial=None, section_series=None, adata=None, root=None, on_close=None, existing_rois=(),
                 imputed_state=None, ensure_imputed=None, session_color_scale=None, cell_type_selection="All", navigator=None):
        self.data, self.abc_cache, self.spatial, self.section_series, self.adata = data, abc_cache, spatial, section_series, adata
        # navigator: (neighbors, move) from the window that opened this one. neighbors(label) -> (the previous section's label or None, the next's);
        # move(label, direction, actions) -> (the new section's data, its existing ROIs) or None: it takes the ROI actions of the section being left.
        self.navigator = navigator
        self.generation = 0            # counts the sections shown: a read that was started for an earlier one is ignored when it ends
        self.imputed_state = imputed_state if imputed_state is not None else {"adata": None, "load_thread": None, "load_error": None}
        self.ensure_imputed = ensure_imputed
        self.color_scale = session_color_scale if session_color_scale is not None else {"mode": "log"}
        self.cell_type_selection = cell_type_selection
        self.own_root = root is None
        self.root = tk.Tk() if root is None else tk.Toplevel(root)
        self.root.title(f"Section {core.sanitize_section_token(data.label)}")
        self.root.geometry("1500x950")
        self.on_close = on_close
        self.msgs = queue.Queue()
        self.mode = "groups"           # "groups" | "gene" | "imputed" | "find"
        self.level = "class"           # the Group level
        self.busy = False
        self.pictures = data.pictures  # (groups level -> picture)
        self.renders = {}              # gene views already made: key -> result
        self.cell_expr = {}            # (gene, dataset) -> expression of each cell of this section (for the hover), or None while it is being read
        self.finds = {}                # (level, number) -> (picture, cells, name)
        self.shown_genes = []          # the genes the picture shows (canonical names), and which dataset
        self.shown_dataset = "standard"
        self.hover_job = self.mark_job = None
        self.hover_pos = None
        self.hover_idx = None
        self.mark = None               # (picture x, picture y) of the cells to mark with '+', or None
        self.mark_anchor = None

        self.pane = ttk.Frame(self.root)
        self.pane.pack(fill=tk.BOTH, expand=True)
        side = ttk.Frame(self.pane, padding=10, width=340)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)
        self.view = ZoomView(self.pane)
        self.view.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.view.post_draw = self.draw_marks
        self.build_side(side)
        self.view.listeners.append(self.place_overlays)
        self.result = None                 # the ROI actions, once Done is clicked (None: cancelled, or still open)
        self.roi = RoiLayer(self, existing_rois)
        self.roi.on_change = self.update_roi_count
        self.update_roi_count()
        c = self.view.canvas
        c.bind("<Motion>", self.on_motion, add="+")
        c.bind("<Leave>", lambda e: self.clear_hover())
        self.ring = c.create_oval(0, 0, 0, 0, outline="#ff1493", width=2, state="hidden", tags="overlay")
        self.view.set_image(data.rgba)
        from scipy.spatial import cKDTree
        self.tree = cKDTree(np.column_stack([data.xs, data.ys]))       # (the cells' positions: the hover finds the nearest within 10 um)
        pb = data.layout["pixel_bbox"]
        dx = data.layout["data_xlim"]
        self.px_per_mm = abs(pb[1] - pb[0]) / abs(dx[1] - dx[0]) if dx[1] != dx[0] else 1.0
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda e: self.close())
        self.update_mode_frames()
        self.update_nav_buttons()
        self.poll()

    # ---- the side panel

    def build_side(self, f):
        d = self.data
        if self.navigator is not None:
            nav = ttk.Frame(f)
            nav.pack(fill=tk.X, pady=(0, 8))
            self.btn_prev = ttk.Button(nav, text="\u25c0 Previous section", command=lambda: self.go(-1))
            self.btn_prev.pack(side=tk.LEFT, expand=True, fill=tk.X)
            self.btn_next = ttk.Button(nav, text="Next section \u25b6", command=lambda: self.go(1))
            self.btn_next.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))
        self.var_title = tk.StringVar(f, f"Section {core.sanitize_section_token(d.label)}")
        ttk.Label(f, textvariable=self.var_title, font=("Segoe UI", 15, "bold")).pack(anchor="w")
        self.var_z = tk.StringVar(f, core.section_z_text(d.label, self.abc_cache, wait=False))
        ttk.Label(f, textvariable=self.var_z, foreground="gray30", wraplength=310).pack(anchor="w")
        self.var_cells = tk.StringVar(f, f"{len(d.xs):,} cells")
        ttk.Label(f, textvariable=self.var_cells, foreground="gray30").pack(anchor="w", pady=(0, 6))

        show = ttk.LabelFrame(f, text="Show", padding=6)
        show.pack(fill=tk.X)
        self.var_mode = tk.StringVar(f, "groups")
        for k, (value, text) in enumerate((("groups", "Groups"), ("gene", "Gene"), ("imputed", "Imputed Gene"), ("find", "Find group"))):
            ttk.Radiobutton(show, text=text, value=value, variable=self.var_mode, command=self.on_mode).grid(row=k // 2, column=k % 2, sticky="w", padx=(0, 14))

        lv = ttk.LabelFrame(f, text="Group level", padding=6)
        lv.pack(fill=tk.X, pady=(6, 0))
        self.var_level = tk.StringVar(f, "class")
        for k, name in enumerate(LEVELS):
            ttk.Radiobutton(lv, text=name.capitalize(), value=name, variable=self.var_level, command=self.on_level).grid(row=k // 2, column=k % 2, sticky="w", padx=(0, 14))

        self.mode_box = ttk.Frame(f)
        self.mode_box.pack(fill=tk.X, pady=(6, 0))
        self.build_gene_controls(self.mode_box)
        self.build_find_controls(self.mode_box)

        self.var_rois = tk.StringVar(f, "")
        roi_box = ttk.LabelFrame(f, text="Regions of interest", padding=6)
        roi_box.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(roi_box, textvariable=self.var_rois, wraplength=300, justify=tk.LEFT).pack(anchor="w")
        ttk.Label(roi_box, text="drag: draw   inside: move   handles: resize   right-click or Delete key: delete the selected (red) one\n"
                                "wheel: zoom   right-drag: pan   Home: whole section", foreground="gray40",
                  wraplength=300, justify=tk.LEFT).pack(anchor="w", pady=(4, 0))

        hov = ttk.LabelFrame(f, text="Under the pointer", padding=6)
        hov.pack(fill=tk.X, pady=(6, 0))
        self.var_pos = tk.StringVar(f, "x = —, y = —")
        ttk.Label(hov, textvariable=self.var_pos, font=("Segoe UI", 10, "bold"), wraplength=310, justify=tk.LEFT).pack(anchor="w")
        self.var_cell = tk.StringVar(f, "Hover over a cell.")
        ttk.Label(hov, textvariable=self.var_cell, wraplength=310, justify=tk.LEFT).pack(anchor="w", pady=(4, 0))

        self.var_status = tk.StringVar(f, "")
        ttk.Label(f, textvariable=self.var_status, foreground="gray30", wraplength=310, justify=tk.LEFT).pack(anchor="w", side=tk.BOTTOM, pady=(6, 0))
        self.var_perf = tk.StringVar(f, "")
        ttk.Label(f, textvariable=self.var_perf, foreground="gray50", font=("Segoe UI", 8)).pack(anchor="w", side=tk.BOTTOM)
        bar = ttk.Frame(f)
        bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        ttk.Button(bar, text="Done", command=self.done).pack(side=tk.RIGHT)
        ttk.Button(bar, text="Cancel", command=self.close).pack(side=tk.RIGHT, padx=6)

    def build_gene_controls(self, parent):
        f = self.gene_frame = ttk.LabelFrame(parent, text="Gene(s)", padding=6)
        ttk.Label(f, text="Up to three, separated by commas (Enter shows them):", foreground="gray30", wraplength=310).pack(anchor="w")
        self.var_gene = tk.StringVar(f, "")
        self.gene_entry = ttk.Entry(f, textvariable=self.var_gene)
        self.gene_entry.pack(fill=tk.X, pady=(2, 0))
        self.gene_entry.bind("<KeyRelease>", self.on_gene_key)
        self.gene_entry.bind("<Return>", lambda e: self.apply_gene())
        self.gene_entry.bind("<Down>", lambda e: self.focus_suggestions())
        self.sugg = tk.Listbox(f, height=6, exportselection=False, activestyle="none")
        self.sugg.bind("<ButtonRelease-1>", lambda e: self.pick_suggestion())
        self.sugg.bind("<Return>", lambda e: self.pick_suggestion())
        scale = ttk.Frame(f)
        scale.pack(fill=tk.X, pady=(6, 0))
        self.var_scale = tk.StringVar(f, self.color_scale["mode"])
        ttk.Label(scale, text="Scale:").pack(side=tk.LEFT)
        for value, text in (("linear", "Linear"), ("log", "Log")):
            ttk.Radiobutton(scale, text=text, value=value, variable=self.var_scale, command=self.on_scale).pack(side=tk.LEFT, padx=(8, 0))
        self.cbar = tk.Canvas(f, width=310, height=86, highlightthickness=0)
        self.cbar.pack(fill=tk.X, pady=(6, 0))
        self.cbar_photo = None

    def build_find_controls(self, parent):
        f = self.find_frame = ttk.LabelFrame(parent, text="Find group", padding=6)
        ttk.Label(f, text="One or more groups' numbers, at the Group level above, e.g. 12, 15, 31 (Enter):", foreground="gray30",
                  wraplength=310).pack(anchor="w")
        self.var_find = tk.StringVar(f, "")
        self.find_entry = ttk.Entry(f, textvariable=self.var_find, width=28)
        self.find_entry.pack(anchor="w", pady=(2, 0))
        self.find_entry.bind("<Return>", lambda e: self.apply_find())
        self.var_find.trace_add("write", lambda *a: self.check_find_text())
        self.find_legend = ttk.Frame(f)
        self.find_legend.pack(fill=tk.X, pady=(4, 0))

    def clear_find_legend(self):
        for w in self.find_legend.winfo_children():
            w.destroy()

    def check_find_text(self):
        """The text turns red while it holds anything that isn't a whole number."""
        bad = core.parse_group_numbers(self.var_find.get())[1]
        self.find_entry.configure(foreground="red" if bad else "")

    def update_mode_frames(self):
        self.gene_frame.pack_forget()
        self.find_frame.pack_forget()
        if self.mode in ("gene", "imputed"):
            self.gene_frame.pack(fill=tk.X)
            self.gene_entry.focus_set()
        elif self.mode == "find":
            self.find_frame.pack(fill=tk.X)
            self.find_entry.focus_set()

    # ---- background work (one job at a time: the controls wait, the picture can still be moved)

    def job(self, text, work, done):
        if self.busy:
            return False
        self.busy = True
        self.var_status.set(text)
        self.root.config(cursor="watch")

        def target():
            try:
                res = work()
            except BaseException as e:
                res = e
            self.msgs.put(("job", (done, res)))

        threading.Thread(target=target, daemon=True).start()
        return True

    def poll(self):
        try:
            while True:
                kind, payload = self.msgs.get_nowait()
                if kind == "job":
                    done, res = payload
                    self.busy = False
                    self.root.config(cursor="")
                    done(res)
                elif kind == "expr":
                    key, values, generation = payload
                    if generation != self.generation:
                        continue                                 # (read for a section that is no longer shown)
                    self.cell_expr[key] = values
                    if self.hover_idx is not None:
                        self.update_hover()
        except queue.Empty:
            pass
        self.var_perf.set(f"view {self.view.scale / max(self.view.fit_scale(), 1e-9):.1f}x   redraw {self.view.last_draw_ms:.0f} ms"
                          if self.view.src is not None else "")
        if not self.var_z.get():               # (the table of the sections' z values is read in the background: it may arrive later)
            self.var_z.set(core.section_z_text(self.data.label, self.abc_cache, wait=False))
        self.poll_id = self.root.after(100, self.poll)

    # ---- the choice of what to show

    def start_in(self, mode, gene):
        """Open already showing a gene view ('gene' or 'imputed_gene') of `gene`."""
        self.var_mode.set("imputed" if mode == "imputed_gene" else "gene")
        self.var_gene.set(gene)
        self.on_mode()

    def on_mode(self):
        mode = self.var_mode.get()
        if self.busy or mode == self.mode:
            self.var_mode.set(self.mode)
            return
        if mode == "imputed" and not self.imputed_ready():
            self.var_mode.set(self.mode)
            return
        self.mode = mode
        self.update_mode_frames()
        self.refresh_picture()

    def on_level(self):
        if self.busy:
            self.var_level.set(self.level)
            return
        self.level = self.var_level.get()
        self.hide_marks()
        if self.mode in ("groups", "find"):
            self.refresh_picture()

    def on_scale(self):
        if self.busy:
            self.var_scale.set(self.color_scale["mode"])
            return
        self.color_scale["mode"] = self.var_scale.get()
        if self.mode in ("gene", "imputed") and self.var_gene.get().strip():
            self.apply_gene()

    def refresh_picture(self):
        """Show what the current choice calls for."""
        self.var_status.set("")
        if self.mode == "groups":
            self.show_groups()
        elif self.mode in ("gene", "imputed"):
            if self.var_gene.get().strip():
                self.apply_gene()
            else:
                self.show_groups()
                self.clear_colorbar()
        else:
            if self.var_find.get().strip():
                self.apply_find()
            else:
                self.show_groups()

    def show_picture(self, rgba):
        self.view.set_image(rgba, keep_view=True)

    # ---- Groups

    def show_groups(self):
        level = self.level
        if level in self.pictures:
            self.show_picture(self.pictures[level])
            return
        if self.spatial is None or self.section_series is None:
            self.var_status.set("No cell metadata to color by.")
            return
        d, spatial, series = self.data, self.spatial, self.section_series

        def work():
            return core.render_section_level_array(series, d.label, spatial, level, d.layout, d.img_w, d.img_h)

        def done(res):
            self.var_status.set("")
            if isinstance(res, BaseException):
                self.var_status.set(f"Could not color by {level}: {res}")
                return
            self.pictures[level] = res
            if self.mode == "groups" and self.level == level:
                self.show_picture(res)

        self.job(f"Coloring by {level}...", work, done)

    # ---- Gene and Imputed Gene

    def imputed_ready(self):
        if self.imputed_state["adata"] is not None:
            return True
        if self.ensure_imputed is not None:
            return bool(self.ensure_imputed(self.imputed_state, self.abc_cache))
        from tkinter import messagebox
        if not messagebox.askyesno("Load imputed gene dataset?", "The imputed gene dataset covers far more genes than the standard 500-gene "
                                   "MERFISH panel, but is much larger. The first time it is loaded (and cached to disk) this can take a long "
                                   "time, and this window will wait. Continue?", parent=self.root):
            return False
        self.var_status.set("Loading the imputed gene dataset...")
        self.root.update()
        try:
            self.imputed_state["adata"] = core.load_imputed_adata(self.abc_cache)
        except Exception as e:
            self.var_status.set(f"Could not load the imputed dataset: {e}")
            return False
        self.var_status.set("")
        return True

    def gene_source(self):
        return self.imputed_state["adata"] if self.mode == "imputed" else self.adata

    def symbols(self):
        return core.gene_symbol_list(self.adata, self.imputed_state, self.mode == "imputed")

    def on_gene_key(self, e):
        self.gene_entry.config(foreground="")
        if e.keysym in ("Return", "Up", "Down", "Escape", "Tab"):
            return
        token = self.var_gene.get().rsplit(",", 1)[-1].strip()
        matches = core.rank_gene_suggestions(self.symbols(), token, 6) if token and self.adata is not None else []
        self.sugg.delete(0, tk.END)
        for m in matches:
            self.sugg.insert(tk.END, m)
        if matches:
            self.sugg.config(height=len(matches))
            self.sugg.pack(fill=tk.X, after=self.gene_entry)
        else:
            self.sugg.pack_forget()

    def focus_suggestions(self):
        if self.sugg.winfo_ismapped() and self.sugg.size():
            self.sugg.focus_set()
            self.sugg.selection_clear(0, tk.END)
            self.sugg.selection_set(0)
        return "break"

    def pick_suggestion(self):
        sel = self.sugg.curselection()
        if not sel:
            return
        name = self.sugg.get(sel[0])
        text = self.var_gene.get()
        prefix = text.rsplit(",", 1)[0].strip() + ", " if "," in text else ""
        self.var_gene.set(prefix + name)
        self.sugg.pack_forget()
        self.gene_entry.focus_set()
        self.apply_gene()

    def apply_gene(self):
        self.sugg.pack_forget()
        if self.busy or self.adata is None:
            return
        names = core.parse_gene_names(self.var_gene.get())
        if not names:
            self.var_status.set("Enter a gene name first.")
            return
        if len(names) > 3:
            self.var_status.set("Enter at most three gene names, separated by commas.")
            return
        use_imputed = self.mode == "imputed"
        source = self.gene_source()
        _cols, resolved, missing = core.resolve_gene_names(source, names)
        if missing:
            self.gene_entry.config(foreground="red")
            self.var_status.set(f"Not found in the {'imputed' if use_imputed else 'standard'} dataset: {', '.join(missing)}.")
            return
        self.gene_entry.config(foreground="")
        self.var_gene.set(", ".join(resolved))
        dataset = "imputed" if use_imputed else "standard"
        scale = self.color_scale["mode"]
        key = (tuple(r.casefold() for r in resolved), scale, dataset)
        if key in self.renders:
            self.show_gene_result(key, resolved, dataset, self.renders[key])
            return
        d, spatial, series, adata = self.data, self.spatial, self.section_series, self.adata
        expr_adata = self.imputed_state["adata"] if use_imputed else None
        selection = self.cell_type_selection

        def work():
            kw = dict(cell_type_selection=selection, color_scale=scale, expr_adata=expr_adata, dataset=dataset)
            if len(resolved) >= 2:
                return core.render_multi_gene_expression_array(adata, series, d.label, lambda: spatial, resolved, d.layout, d.img_w, d.img_h, **kw)
            return core.render_gene_expression_array(adata, series, d.label, lambda: spatial, resolved[0], d.layout, d.img_w, d.img_h, **kw)

        def done(res):
            self.var_status.set("")
            if isinstance(res, BaseException):
                self.var_status.set(f"Could not show {', '.join(resolved)}: {res}")
                return
            self.renders[key] = res
            self.show_gene_result(key, resolved, dataset, res)

        self.job(f"Rendering {', '.join(resolved)}...", work, done)

    def show_gene_result(self, key, resolved, dataset, res):
        if self.mode not in ("gene", "imputed") or (self.mode == "imputed") != (dataset == "imputed"):
            return                                       # (the choice changed while it was being made)
        self.show_picture(res[0])
        self.shown_genes, self.shown_dataset = list(resolved), dataset
        scale = key[1]
        if len(resolved) >= 2:
            self.draw_colorbar_multi(res[1], res[3], res[4], scale, dataset)
            self.var_status.set(f"Showing {', '.join(resolved)} (red / green / blue).")
        else:
            self.draw_colorbar(res[3], res[4], scale, dataset)
            self.var_status.set(f"Showing {resolved[0]} (highest expression {res[2]:.2f}).")
        self.load_cell_expr(resolved, dataset)
        if self.hover_idx is not None:
            self.update_hover()

    # ---- the colour bar

    def clear_colorbar(self):
        self.cbar.delete("all")
        self.shown_genes = []

    def draw_colorbar(self, vmin, vmax, scale, dataset):
        from matplotlib import colormaps
        c = self.cbar
        c.delete("all")
        w = max(c.winfo_width(), 200)
        ramp = (colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)[None, :, :]
        img = cv2.resize(np.repeat(ramp, 16, axis=0), (w - 20, 16), interpolation=cv2.INTER_LINEAR)
        self.cbar_photo = ImageTk.PhotoImage(Image.fromarray(img), master=c)
        c.create_image(10, 8, anchor="nw", image=self.cbar_photo)
        c.create_text(10, 30, anchor="nw", text=f"{vmin:.2f}", font=("Segoe UI", 9))
        c.create_text(w - 10, 30, anchor="ne", text=f"{vmax:.2f}", font=("Segoe UI", 9))
        c.create_text(w / 2, 50, text=self.SCALE_UNITS[(dataset, scale)], font=("Segoe UI", 9), fill="gray30")

    def draw_colorbar_multi(self, genes, vmins, vmaxs, scale, dataset):
        c = self.cbar
        c.delete("all")
        self.cbar_photo = None
        for k, (g, lo, hi) in enumerate(zip(genes, vmins, vmaxs)):
            y = 8 + 18 * k
            c.create_rectangle(10, y, 24, y + 12, fill=self.CHANNEL_COLORS[k], outline="")
            c.create_text(32, y - 1, anchor="nw", text=f"{g}   {lo:.1f}–{hi:.1f}", font=("Segoe UI", 9))
        c.create_text(10, 8 + 18 * len(genes), anchor="nw", text=self.SCALE_UNITS[(dataset, scale)], font=("Segoe UI", 9), fill="gray30")

    # ---- the expression of each cell (for the hover), read in the background after a gene is shown

    def load_cell_expr(self, genes, dataset):
        for gene in genes:
            key = (gene, dataset)
            if key in self.cell_expr:
                continue
            self.cell_expr[key] = None                  # (being read)
            source = self.imputed_state["adata"] if dataset == "imputed" else self.adata
            rows, adata, generation = self.data.rows, self.adata, self.generation

            def work(gene=gene, key=key, source=source):
                try:
                    col = core.find_gene_index(source, gene)
                    if dataset == "imputed":
                        pos = source.obs_names.get_indexer(adata.obs_names[rows])
                        values = np.full(len(rows), np.nan)
                        found = pos >= 0
                        values[found] = core._get_full_gene_column(source, col)[pos[found]]
                    else:
                        cells = adata[rows, col].X
                        values = np.asarray(cells.toarray() if hasattr(cells, "toarray") else cells).ravel().astype(float)
                except Exception as e:
                    print(f"Could not read the expression of {gene}: {e}", flush=True)
                    values = False
                self.msgs.put(("expr", (key, values, generation)))

            threading.Thread(target=work, daemon=True).start()

    def expression_text(self, idx):
        """The hovered cell's value of each gene being shown, in the units of the Linear/Log choice (as the picture is colored)."""
        if self.mode not in ("gene", "imputed") or not self.shown_genes:
            return None
        log = self.color_scale["mode"] == "log"
        parts = []
        for gene in self.shown_genes:
            values = self.cell_expr.get((gene, self.shown_dataset))
            if values is None:
                parts.append(f"{gene} = ...")
            elif values is False:
                continue
            else:
                v = float(values[idx])
                if self.shown_dataset == "imputed":
                    v = v if log else 2.0 ** v - 1.0
                else:
                    v = float(np.log1p(v)) if log else v
                parts.append(f"{gene} = {v:.3g}" if np.isfinite(v) else f"{gene} = n/a")
        if not parts:
            return None
        return ", ".join(parts) + f" ({self.SCALE_UNITS[(self.shown_dataset, self.color_scale['mode'])]})"

    # ---- Find group

    def apply_find(self):
        if self.busy:
            return
        text = self.var_find.get().strip()
        if not text:
            self.clear_find_legend()
            self.show_groups()
            return
        numbers, bad = core.parse_group_numbers(text)
        if bad or not numbers:
            self.find_entry.configure(foreground="red")
            self.var_status.set(f"Not a number: {', '.join(bad)}. Type {self.level} numbers separated by commas, e.g. 12, 15.")
            return
        level = self.level
        key = (level, tuple(numbers))
        if key in self.finds:
            self.show_find(key, self.finds[key])
            return
        if self.spatial is None or self.section_series is None:
            self.var_status.set("No cell metadata to search.")
            return
        d, spatial, series = self.data, self.spatial, self.section_series

        def work():
            return core.render_groups_highlight_array(series, d.label, spatial, level, numbers, d.layout, d.img_w, d.img_h)

        def done(res):
            self.var_status.set("")
            if isinstance(res, BaseException):
                self.var_status.set(f"Could not mark {level} {', '.join(map(str, numbers))}: {res}")
                return
            if key[1] != tuple(core.parse_group_numbers(self.var_find.get())[0]) or level != self.level:
                return                      # (the text or the level changed while this was drawn)
            self.finds[key] = res
            self.show_find(key, res)

        self.job(f"Finding {level} {', '.join(map(str, numbers))}...", work, done)

    def show_find(self, key, res):
        if self.mode != "find":
            return
        rgba, found = res
        self.show_picture(rgba)
        level = key[0]
        self.clear_find_legend()
        for number, count, name, (r, g, b) in found:
            text = f"{level} {name}: {count:,} cells" if count else f"{level} {number}: none in this section"
            tk.Label(self.find_legend, text="● " + text, fg="#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)), bg="black",
                     anchor="w", padx=4).pack(fill=tk.X, pady=1)

    # ---- hover

    def pointer_to_data(self, cx, cy):
        """The data position (mm) under a canvas point, or None outside the picture."""
        ix, iy = self.view.to_picture(cx, cy)
        d = self.data
        if not (0 <= ix < d.img_w and 0 <= iy < d.img_h):
            return None
        return core.map_pixel_to_data(d.layout, ix, d.img_h - iy)       # (the layout's y counts up from the bottom of the picture)

    def on_motion(self, e):
        self.hover_pos = (e.x, e.y)
        if self.mark is not None and self.mark_anchor is not None and (abs(e.x - self.mark_anchor[0]) > 3 or abs(e.y - self.mark_anchor[1]) > 3):
            self.hide_marks()
        if self.hover_job is None:
            self.hover_job = self.view.canvas.after(25, self.update_hover)      # (at most about 40 lookups a second)

    def update_hover(self):
        self.hover_job = None
        if self.hover_pos is None:
            return
        pos = self.pointer_to_data(*self.hover_pos)
        if pos is None:
            self.clear_hover()
            return
        x, y = pos
        z_text = core.section_z_text(self.data.label, self.abc_cache)
        self.var_pos.set(f"x = {x:.3f}, y = {y:.3f}" + (f"\n{z_text}" if z_text else ""))
        dist, idx = self.tree.query([x, y], distance_upper_bound=HOVER_RADIUS_MM)
        if not np.isfinite(dist):
            self.hover_idx = None
            self.var_cell.set("No cell within 10 µm.")
            self.view.canvas.itemconfig(self.ring, state="hidden")
            self.cancel_marks()
            return
        idx = int(idx)
        changed = idx != self.hover_idx
        self.hover_idx = idx
        cells = self.data.cells
        parts = [f"{lv}: {cells[lv].iloc[idx]}" for lv in LEVELS if lv in cells.columns]
        expression = self.expression_text(idx)
        self.var_cell.set(("\n".join(parts) if parts else "(no class metadata)") + (f"\n{expression}" if expression else ""))
        self.place_overlays()
        if changed or self.mark_job is None and self.mark is None:
            self.cancel_marks()
            self.mark_job = self.view.canvas.after(self.MARK_DELAY_MS, lambda: self.show_marks(idx))

    def clear_hover(self):
        self.hover_idx = None
        self.view.canvas.itemconfig(self.ring, state="hidden")
        self.var_pos.set("x = —, y = —")
        self.var_cell.set("Hover over a cell.")
        self.cancel_marks()

    def place_overlays(self):
        """The hover ring follows its cell as the view changes."""
        if self.hover_idx is None:
            return
        d = self.data
        cx, cy = self.view.to_canvas(d.cell_px[self.hover_idx], d.cell_py[self.hover_idx])
        r = max(HOVER_RADIUS_MM * self.px_per_mm * self.view.scale, 4.0)
        self.view.canvas.coords(self.ring, cx - r, cy - r, cx + r, cy + r)
        self.view.canvas.itemconfig(self.ring, state="normal")
        self.view.canvas.tag_raise("overlay")

    # ---- the red '+' on the cells of the hovered cell's group, once the pointer has rested on it

    def show_marks(self, idx):
        self.mark_job = None
        if self.hover_idx != idx or self.hover_pos is None:
            return
        cells = self.data.cells
        if self.level not in cells.columns:
            return
        values = cells[self.level].to_numpy()
        target = values[idx]
        if target is None or (isinstance(target, float) and np.isnan(target)):
            return
        same = values == target
        self.mark = (self.data.cell_px[same], self.data.cell_py[same])
        self.mark_anchor = self.hover_pos
        self.view.request_draw()
        self.var_status.set(f"{int(same.sum()):,} cells share {self.level} {target}.")

    def cancel_marks(self):
        if self.mark_job is not None:
            self.view.canvas.after_cancel(self.mark_job)
            self.mark_job = None
        self.hide_marks()

    def hide_marks(self):
        if self.mark is not None:
            self.mark = None
            self.mark_anchor = None
            self.view.request_draw()

    def draw_marks(self, out, x0, y0, s):
        """(ZoomView's hook) the '+' marks, drawn on the resized picture that is about to be shown."""
        if self.mark is None:
            return out
        px, py = self.mark
        sx = ((px - x0) * s).astype(np.int32)
        sy = ((py - y0) * s).astype(np.int32)
        h, w = out.shape[:2]
        keep = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        sx, sy = sx[keep], sy[keep]
        if len(sx) > 30000:
            step = len(sx) // 30000 + 1
            sx, sy = sx[::step], sy[::step]
        for k in range(-3, 4):
            out[sy, np.clip(sx + k, 0, w - 1)] = self.MARK_COLOR
            out[np.clip(sy + k, 0, h - 1), sx] = self.MARK_COLOR
        return out

    # ---- the regions of interest

    def update_roi_count(self):
        n = len(self.roi.rois)
        if not n:
            self.var_rois.set("None yet.")
            return
        lines = [f"{n} region(s):"]
        for k, roi in enumerate(self.roi.rois.values(), 1):
            x_lo, x_hi, y_lo, y_hi = self.roi.box_to_data(roi["box"])
            lines.append(f"  {k}. {x_hi - x_lo:.2f} × {y_hi - y_lo:.2f} mm" + ("" if roi["kind"] == "new" else "  (from before)"))
        self.var_rois.set("\n".join(lines))

    def done(self):
        self.result = self.roi.actions()
        self.close()

    # ---- the next and previous section: the same window, everything as it is set, on another section

    def update_nav_buttons(self):
        if self.navigator is None:
            return
        previous, following = self.navigator[0](self.data.label)
        self.btn_prev.state(["!disabled" if previous is not None else "disabled"])
        self.btn_next.state(["!disabled" if following is not None else "disabled"])

    def go(self, direction):
        """Next (+1) or previous (-1) section, in the order of the section grid. This section's regions are kept, as with Done; what is shown
        (Groups, Gene, Find group, the level, the genes, the scale) stays as it is."""
        if self.navigator is None:
            return
        if self.busy:
            self.var_status.set("Wait for the picture to finish, then try again.")
            return
        self.root.config(cursor="watch")
        try:
            moved = self.navigator[1](self.data.label, direction, self.roi.actions())
        except Exception as e:
            moved = None
            self.var_status.set(f"Could not open the {'next' if direction > 0 else 'previous'} section: {e}")
        finally:
            self.root.config(cursor="")
        if moved is None:
            self.update_nav_buttons()
            return
        self.switch_section(*moved)

    def switch_section(self, data, existing_rois):
        """Show another section in this window: its picture, its cells, its regions; and the picture the current choice calls for."""
        self.generation += 1
        self.data = data
        self.pictures = data.pictures
        self.renders, self.cell_expr, self.finds = {}, {}, {}
        self.shown_genes, self.shown_dataset = [], "standard"
        self.clear_hover()
        self.hide_marks()
        self.clear_colorbar()
        self.clear_find_legend()
        self.root.title(f"Section {core.sanitize_section_token(data.label)}")
        self.var_title.set(f"Section {core.sanitize_section_token(data.label)}")
        self.var_z.set(core.section_z_text(data.label, self.abc_cache, wait=False))
        self.var_cells.set(f"{len(data.xs):,} cells")
        from scipy.spatial import cKDTree
        self.tree = cKDTree(np.column_stack([data.xs, data.ys]))
        pb, dx = data.layout["pixel_bbox"], data.layout["data_xlim"]
        self.px_per_mm = abs(pb[1] - pb[0]) / abs(dx[1] - dx[0]) if dx[1] != dx[0] else 1.0
        self.view.set_image(data.rgba, keep_view=False)
        self.roi.reset(existing_rois)
        self.update_nav_buttons()
        self.refresh_picture()

    # ---- closing

    def close(self):
        try:
            self.root.after_cancel(self.poll_id)
        except Exception:
            pass
        if self.own_root:
            try:
                self.root.quit()        # (the loop ends even while other windows, matplotlib's, are alive in the same process)
            except tk.TclError:
                pass
        self.root.destroy()
        if self.on_close:
            self.on_close()

    def run(self):
        if self.own_root:
            self.root.mainloop()
        else:
            self.root.grab_set()
            self.root.wait_window(self.root)         # (a child window: this returns when it is closed, the parent's loop goes on)


# ---------------------------------------------------------------------------------------------------------------------- the browser's call

def prompt_subregion_selection_tk(adata, abc_cache, section_series, section_label, existing_rois=None, on_ready=None, spatial=None,
                                  initial_view=None, cell_type_selection="All", precomputed_gene_open=None, session_color_scale=None,
                                  shared_spatial_cache=None, imputed_state=None, session_view_settings=None, ensure_imputed=None,
                                  parent=None, navigator=None, on_window=None):
    """The same call, and the same result, as ABC_atlas_browser.prompt_subregion_selection (the matplotlib window): the single-section ROI
    picker. Returns a list of {'action': 'add'|'edit'|'delete', ...} (empty if cancelled). The section's picture must already be cached
    (the caller makes it first, as for the matplotlib window). `ensure_imputed(imputed_state, abc_cache)`, if given, loads the imputed gene
    dataset when the Imputed Gene view is chosen (the browser's own loader, with its warning and progress window)."""
    if spatial is None:
        spatial = core.load_section_spatial_coords(adata, abc_cache)
    data = SectionData.load(adata, abc_cache, section_series, spatial, section_label)
    win = SectionWindow(data, abc_cache, spatial=spatial, section_series=section_series, adata=adata, root=parent, existing_rois=existing_rois or (),
                        imputed_state=imputed_state, ensure_imputed=ensure_imputed, session_color_scale=session_color_scale,
                        cell_type_selection=cell_type_selection, navigator=navigator)
    if on_window:
        on_window(win)                  # (the grid shows its waits over this window when it moves to another section)
    try:
        win.root.state("zoomed")        # (opens full-screen, as the matplotlib window does)
    except tk.TclError:
        pass
    # opens already showing what the grid (or the last window) was set to, as the matplotlib window does
    view = None
    if precomputed_gene_open is not None:
        view = {"mode": "imputed_gene" if precomputed_gene_open.get("dataset") == "imputed" else "gene", "gene": precomputed_gene_open["resolved_gene"]}
    elif initial_view and initial_view.get("mode") in ("gene", "imputed_gene") and initial_view.get("gene"):
        view = initial_view
    if view is not None:
        win.start_in(view["mode"], view["gene"])
    if on_ready:
        on_ready()
    win.root.lift()
    win.root.focus_force()
    win.run()
    if session_view_settings is not None:         # (the next section opens on the same view)
        session_view_settings["mode"] = {"groups": "standard", "find": "standard", "gene": "gene", "imputed": "imputed_gene"}[win.mode]
        session_view_settings["gene"] = win.var_gene.get().strip()
    return win.result or []


# ---------------------------------------------------------------------------------------------------------------------- running it

def main(argv=None):
    from pathlib import Path
    import anndata
    from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
    argv = sys.argv[1:] if argv is None else argv
    core.load_coordinate_settings()          # (the z coordinate chosen in the launcher's Settings)
    root = tk.Tk()
    root.title("Loading...")
    status = tk.StringVar(root, "Opening the atlas...")
    ttk.Label(root, textvariable=status, padding=30).pack()
    root.update()
    base = Path(os.environ.get("ABC_ATLAS_CACHE", str(core.SCRIPT_DIR.parent / "abc_atlas_cache")))
    cache = AbcProjectCache.from_local_cache(base)
    path = cache.get_data_path(directory="MERFISH-C57BL6J-638850", file_name="C57BL6J-638850/raw")
    adata = anndata.read_h5ad(path, backed="r")
    status.set("Reading the cell metadata...")
    root.update()
    sections = core.get_section_labels(adata, cache)
    spatial = core.load_section_spatial_coords(adata, cache)
    labels = sorted(sections.unique(), key=core.section_sort_key)
    want = argv[0] if argv else labels[len(labels) // 2]
    label = next((s for s in labels if s == want or core.sanitize_section_token(s) == str(want).lstrip("0")), labels[len(labels) // 2])
    status.set(f"Section {core.sanitize_section_token(label)}...")
    root.update()
    data = SectionData.load(adata, cache, sections, spatial, label)
    root.destroy()
    SectionWindow(data, cache, spatial=spatial, section_series=sections, adata=adata).run()


if __name__ == "__main__":
    main()
