"""A Tk version of the atlas browser's interactive UMAP viewer (first version: the UMAP, a grid of the run's brain sections, colouring
by group / single group(s) / gene / imputed gene, and the hover link between the two).

Why it is quicker than the matplotlib window: every plot here is one image on a Tk canvas that is drawn directly from the cells' positions
(a point is a few pixels written into an array), so a recolour, a zoom or a pan costs the same however many artists the figure would
have had, and each section panel is a plain canvas of its own with its own zoom and pan. The hover ring is a canvas item.

    show_umap_window_tk(adata, abc_cache, imputed_state=None, adata_backed=None, rois=None, run_folder=None, ensure_imputed=None)

blocks until the window is closed, like the matplotlib one. Not yet in this version: more than one gene at once, the UMAP variants,
the group (family) highlight on hover, and the exports.
"""
import colorsys
import gc
import queue
import re
from pathlib import Path
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from scipy.spatial import cKDTree

import abc_core as core
from abc_tk_umap_export import ExportsMixin, hexcolor

LEIDEN_KEY = "leiden"
GREY = np.array([205, 205, 205], np.uint8)
MULTI_BG = (26, 26, 26)           # the plots' background in the multi-gene view (a zero-expression cell is black, so it must not melt into it)
MAX_GENES = 6                      # the first three are red / green / blue circles; the next three the same colors as small dots on top
CHANNEL_HEX = ("#e53935", "#43a047", "#4a6cff", "#e53935", "#43a047", "#4a6cff")
CHANNEL_NAMES = ("red", "green", "blue", "red", "green", "blue")
LIGHT_GREY = np.array([222, 222, 222], np.uint8)
RING = "#ff1493"
MAX_COLORED_CATEGORIES = 400
LEVEL_NAMES = {"class": "Class", "subclass": "Subclass", "supertype": "Supertype", "cluster": "Cluster", "leiden": "Leiden"}
TILE_PX = 230
MIN_TILE_PX = 50
TILE_COLUMNS = (6, 5, 4, 3, 2, 1)         # the Section map size slider's stops: how many panels fit across the pane (1/6 of its width ... all of it)
DEFAULT_TILE_INDEX = 4                    # (two across)
DOT_SIZES = {"Small": 0.5, "Medium": 1.0, "Large": 1.5}      # the section panels' dot size choices: times DOT_MM (15, 30 and 60 um)
OTHER_CELLS_DOT_RATIO = 0.5       # the dots of the cells of the other kind (e.g. the non-neurons of a neuron run), relative to the others'
DEFAULT_DOT_INDEX = 1                     # (the middle of DOT_SIZES)
MARKER_SIZES = (0.5, 1.0, 2.0)            # the Marker size slider's stops: times the size of the hover ring and of the red + marks in the section panels
DEFAULT_MARKER_INDEX = 1
GROUP_SORT_DESCENDING_FIRST = {"count"}     # the group list's columns whose first click sorts largest first (the others: smallest / A first)
LABEL_SIZES = (0.7, 1.0, 1.4, 2.0)         # the UMAP's Label size slider's stops: times the group numbers' font size
LABEL_SIZE_NAMES = ("Small", "Medium", "Large", "Very large")
DEFAULT_LABEL_INDEX = 1
LABEL_ZOOM_CAP = 5.0               # the group numbers grow with the zoom up to this, then stay the same size
HOVER_HOLD_MS = 250                # the pointer must rest this long before the plots dim
GROUP_HOLD_MS = 500                # ...and this long before the cell's group is marked
DOT_MM = 0.04                      # a cell's dot in the section panels: 30 um across on the tissue
BIG_DOT_PX = 40                    # dots wider than this are drawn one by one as circles (the maximum filter gets slow)
SCALEBAR_UM = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)
HOVER_RADIUS_PX = 9
SCALE_UNITS = {("standard", "log"): "log1p(expr)", ("standard", "linear"): "expr (raw counts)",
               ("imputed", "log"): "log2 (imputed)", ("imputed", "linear"): "linear (imputed)"}


def rank_color(rank):
    """The browser's colour for a category of this rank (0 = commonest): ten hues stepped by the golden angle, in a vivid and a muted set."""
    hue = ((rank % 10) * 0.6180339887498949) % 1.0
    s, v = (0.90, 0.95) if (rank // 10) % 2 == 0 else (0.45, 0.85)
    return np.array([int(round(c * 255)) for c in colorsys.hsv_to_rgb(hue, s, v)], np.uint8)


def viridis_rgb(t):
    from matplotlib import colormaps
    return (colormaps["viridis"](np.clip(t, 0, 1))[:, :3] * 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------------------- the point canvas

def stamp_plus(img, px, py, rgb, arm):
    """A '+' (two pixels thick, `arm` pixels each way) at each point, in place. `rgb`: one color (3,) or one per point (n, 3)."""
    h, w = img.shape[:2]
    per_point = np.ndim(rgb) == 2
    for k in range(-arm, arm + 1):
        for dx, dy in ((k, 0), (k, 1), (0, k), (1, k)):
            xx, yy = px + dx, py + dy
            ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            img[yy[ok], xx[ok]] = rgb[ok] if per_point else rgb


class HoverTip:
    """A small tip beside the pointer while it is over `widget`; `text_fn()` gives the text (asked again as the pointer moves)."""

    def __init__(self, widget, text_fn):
        self.widget, self.text_fn, self.tip, self.label = widget, text_fn, None, None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Motion>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.show, add="+")

    def show(self, e):
        if self.tip is None:
            self.tip = tk.Toplevel(self.widget)
            self.tip.wm_overrideredirect(True)
            self.label = tk.Label(self.tip, bg="#ffffe0", relief="solid", borderwidth=1, font=("Segoe UI", 9), padx=4, pady=1)
            self.label.pack()
        self.label.config(text=self.text_fn())
        self.tip.wm_geometry(f"+{e.x_root + 12}+{e.y_root + 18}")

    def hide(self, e=None):
        if self.tip is not None:
            self.tip.destroy()
            self.tip = self.label = None


class PointCanvas:
    """Points drawn straight onto a Tk canvas: wheel zoom about the pointer, left- or right-drag pan, never out past the whole (`home`)
    view and never off its edges. `layers` are drawn in order; each is {'x','y','rgb','size'}. Markers (the hover ring), rectangles
    and text labels are canvas items given in data coordinates and repositioned after each draw."""

    MAX_ZOOM = 400.0

    def __init__(self, parent, home, flip_y, width=300, height=300):
        self.canvas = tk.Canvas(parent, width=width, height=height, bg="white", highlightthickness=1, highlightbackground="#bbbbbb")
        self.home = home                            # (xmin, xmax, ymin, ymax)
        self.flip_y = flip_y                        # True: data y grows upward on screen (the UMAP); False: downward (a brain section)
        self.layers = []
        self.cx = (home[0] + home[1]) / 2
        self.cy = (home[2] + home[3]) / 2
        self.scale = 1.0
        self.at_fit = True
        self.photo = None
        self.pending = None
        self.item = self.canvas.create_image(0, 0, anchor="nw")
        self.markers = {}                           # name -> [item, x, y] (x None: hidden)
        self.rects = []                             # [item, x0, y0, x1, y1]
        self.labels = []                            # [item, x, y]
        self.label_factor = 1.0                     # the Label size slider
        self.label_size = 8                         # the group labels' font size: grows with the zoom up to LABEL_ZOOM_CAP, then stops
        self.scalebar = None                        # (line item, text item) once enabled: data units are mm
        self.on_motion = None                       # fn(cx, cy) ; fn(None, None) on leave
        self.on_view = None                         # fn() after a redraw
        self.on_change = None                       # fn(zoom, dx, dy) after the user zoomed or panned (the view, relative to the whole view)
        self.pan_from = None
        self.marker_factor = 1.0                    # times the size of the hover ring and of the + marks
        self.dot_factor = 1.0                       # times the size of the dots that have a size on the tissue
        self.visible_fn = None                      # fn() -> whether this canvas is on screen; one that is not is only drawn once it is
        self.stale = False                          # needs drawing (it was not on screen when asked)
        self.base = None                            # the last picture of the points (H x W x 3)
        self.dim = False                            # a veil over the points (while a cell is highlighted)
        self.bgcolor = (255, 255, 255)
        self.plus_arm = 3                           # the '+' marks' arm length (pixels) at the whole view
        self.pluses = None                          # (x, y) of the red '+' marks of the highlighted group
        self.compose_pending = None
        c = self.canvas
        c.bind("<Configure>", lambda e: self.on_resize())
        c.bind("<MouseWheel>", lambda e: self.zoom_at(e.x, e.y, 1.25 if e.delta > 0 else 0.8))
        c.bind("<Button-4>", lambda e: self.zoom_at(e.x, e.y, 1.25))
        c.bind("<Button-5>", lambda e: self.zoom_at(e.x, e.y, 0.8))
        for press, motion, release in (("<ButtonPress-1>", "<B1-Motion>", "<ButtonRelease-1>"), ("<ButtonPress-3>", "<B3-Motion>", "<ButtonRelease-3>"),
                                       ("<ButtonPress-2>", "<B2-Motion>", "<ButtonRelease-2>")):
            c.bind(press, self.pan_start)
            c.bind(motion, self.pan_move)
            c.bind(release, self.pan_end)
        c.bind("<Motion>", lambda e: self.on_motion and self.on_motion(e.x, e.y), add="+")
        c.bind("<Leave>", lambda e: self.on_motion and self.on_motion(None, None))
        c.bind("<Double-Button-1>", lambda e: (self.fit(), self.announce()))

    # ---- geometry

    def size(self):
        return max(self.canvas.winfo_width(), 2), max(self.canvas.winfo_height(), 2)

    def fit_scale(self):
        w, h = self.size()
        x0, x1, y0, y1 = self.home
        return min(w / max(x1 - x0, 1e-9), h / max(y1 - y0, 1e-9))

    def announce(self):
        if self.on_change:
            self.on_change(self.scale / self.fit_scale(), self.cx - (self.home[0] + self.home[1]) / 2, self.cy - (self.home[2] + self.home[3]) / 2)

    def follow(self, zoom, dx, dy):
        """Take the same view (zoom, and offset from the whole view's center) as the panel that was moved."""
        self.scale = self.fit_scale() * zoom
        self.cx = (self.home[0] + self.home[1]) / 2 + dx
        self.cy = (self.home[2] + self.home[3]) / 2 + dy
        self.at_fit = zoom <= 1.0000001
        self.clamp()
        self.request_draw()

    def fit(self):
        self.at_fit = True
        self.cx, self.cy = (self.home[0] + self.home[1]) / 2, (self.home[2] + self.home[3]) / 2
        self.scale = self.fit_scale()
        self.request_draw()

    def clamp(self):
        w, h = self.size()
        fs = self.fit_scale()
        self.scale = min(max(self.scale, fs), fs * self.MAX_ZOOM)
        x0, x1, y0, y1 = self.home
        hw, hh = w / (2 * self.scale), h / (2 * self.scale)
        self.cx = (x0 + x1) / 2 if 2 * hw >= x1 - x0 else min(max(self.cx, x0 + hw), x1 - hw)
        self.cy = (y0 + y1) / 2 if 2 * hh >= y1 - y0 else min(max(self.cy, y0 + hh), y1 - hh)

    def on_resize(self):
        if self.at_fit:
            self.fit()
        else:
            self.clamp()
            self.request_draw()

    def to_canvas(self, x, y):
        w, h = self.size()
        sy = -(y - self.cy) if self.flip_y else (y - self.cy)
        return w / 2 + (x - self.cx) * self.scale, h / 2 + sy * self.scale

    def to_data(self, px, py):
        w, h = self.size()
        x = self.cx + (px - w / 2) / self.scale
        dy = (py - h / 2) / self.scale
        return x, self.cy + (-dy if self.flip_y else dy)

    def zoom_at(self, px, py, factor):
        x, y = self.to_data(px, py)
        before = self.scale
        fs = self.fit_scale()
        self.scale = min(max(self.scale * factor, fs), fs * self.MAX_ZOOM)
        if self.scale == before:
            return
        self.at_fit = abs(self.scale - fs) < 1e-9 * fs
        w, h = self.size()
        self.cx = x - (px - w / 2) / self.scale
        dy = (py - h / 2) / self.scale
        self.cy = y + dy if self.flip_y else y - dy
        self.clamp()
        self.request_draw()
        self.announce()

    def pan_start(self, e):
        self.pan_from = (e.x, e.y, self.cx, self.cy)

    def pan_move(self, e):
        if self.pan_from is None:
            return
        x0, y0, cx, cy = self.pan_from
        self.cx = cx - (e.x - x0) / self.scale
        dy = (e.y - y0) / self.scale
        self.cy = cy + dy if self.flip_y else cy - dy
        self.at_fit = False
        self.clamp()
        self.request_draw()
        self.announce()

    def pan_end(self, e):
        self.pan_from = None

    # ---- drawing

    def set_layers(self, layers):
        self.layers = layers
        self.request_draw()

    def request_draw(self):
        if self.pending is None:
            self.pending = self.canvas.after_idle(self.draw)

    def draw(self):
        self.pending = None
        w, h = self.size()
        if w <= 2 or h <= 2:
            return
        if self.visible_fn is not None and not self.visible_fn():
            self.stale = True
            return
        self.stale = False
        if self.at_fit:
            self.scale = self.fit_scale()
            self.cx, self.cy = (self.home[0] + self.home[1]) / 2, (self.home[2] + self.home[3]) / 2
        zoom = self.scale / self.fit_scale()
        img = np.empty((h, w, 3), np.uint8)
        img[:] = self.bgcolor
        hw, hh = w / (2 * self.scale), h / (2 * self.scale)
        for lay in self.layers:
            if "size_mm" in lay:                                              # (a size on the tissue: the same at any window size)
                d = int(max(round(lay["size_mm"] * self.dot_factor * self.scale), 1))
            else:
                d = int(max(round(lay["size"] * zoom ** 0.7), 1))          # (the dots grow as the view is zoomed in)
            x, y = lay["x"], lay["y"]
            pm = d / self.scale
            m = (x >= self.cx - hw - pm) & (x <= self.cx + hw + pm) & (y >= self.cy - hh - pm) & (y <= self.cy + hh + pm)
            idx = np.flatnonzero(m)
            if not len(idx):
                continue
            px = ((x[idx] - self.cx) * self.scale + w / 2).astype(np.int32)
            dy = (y[idx] - self.cy) * self.scale
            py = ((-dy if self.flip_y else dy) + h / 2).astype(np.int32)
            rgb = lay["rgb"][idx]
            if lay.get("shape") == "plus":
                stamp_plus(img, px, py, rgb, d)
                continue
            if d > BIG_DOT_PX:                                             # (very big dots, so few on screen: each drawn as a circle, later ones on top)
                r = d // 2
                for X, Y, c in zip(px.tolist(), py.tolist(), rgb.tolist()):
                    cv2.circle(img, (X, Y), r, c, -1)
                continue
            px, py = px + d, py + d
            # each point writes its number (later = on top) at its pixel; a bigger dot is that number grown by a maximum filter
            lab = np.zeros((h + 2 * d, w + 2 * d), np.float32)
            lab[py, px] = np.arange(1, len(idx) + 1, dtype=np.float32)
            if lay.get("shape") == "half" and d > 1:                     # (the right half of a disc, flat side on the cell)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
                k[:, :d // 2] = 0
                lab = cv2.dilate(lab, k[:, d // 2:], anchor=(0, d // 2))
            elif d > 1:
                lab = cv2.dilate(lab, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
            lab = lab[d:d + h, d:d + w]
            hit = lab > 0
            img[hit] = rgb[lab[hit].astype(np.int64) - 1]
        self.base = img
        self.compose()
        self.reposition()
        if self.on_view:
            self.on_view()

    def compose(self):
        """The picture shown: the points, veiled and with the highlighted group's '+' marks when those are on."""
        self.compose_pending = None
        if self.base is None:
            return
        if self.visible_fn is not None and not self.visible_fn():
            self.stale = True
            return
        img = self.base
        if self.dim:
            img = (img >> 1) + (np.array(self.bgcolor, np.uint8) >> 1)
        elif self.pluses is not None:
            img = img.copy()
        if self.pluses is not None and len(self.pluses[0]):
            h, w = img.shape[:2]
            x, y = self.pluses
            px = ((x - self.cx) * self.scale + w / 2).astype(np.int32)
            dy = (y - self.cy) * self.scale
            py = ((-dy if self.flip_y else dy) + h / 2).astype(np.int32)
            zoom = self.scale / self.fit_scale()
            stamp_plus(img, px, py, np.array([255, 0, 0], np.uint8), max(int(min((self.plus_arm + zoom ** 0.4) * self.marker_factor, 14 * self.marker_factor)), 1))
        self.photo = ImageTk.PhotoImage(Image.fromarray(np.ascontiguousarray(img)), master=self.canvas)
        self.canvas.itemconfig(self.item, image=self.photo)

    def request_compose(self):
        if self.pending is None and self.compose_pending is None:
            self.compose_pending = self.canvas.after_idle(self.compose)

    def set_background(self, rgb):
        if tuple(rgb) != tuple(self.bgcolor):
            self.bgcolor = tuple(rgb)
            self.canvas.config(bg="#%02x%02x%02x" % self.bgcolor)
            if self.scalebar:                                   # (white on a dark background)
                ink = "white" if sum(self.bgcolor) < 300 else "black"
                self.canvas.itemconfig(self.scalebar[0], fill=ink)
                self.canvas.itemconfig(self.scalebar[1], fill=ink)

    def set_dim(self, on):
        if on != self.dim:
            self.dim = on
            self.request_compose()

    def set_pluses(self, xy):
        if xy is None and self.pluses is None:
            return
        self.pluses = xy
        self.request_compose()

    def refresh(self):
        """Called when this canvas may have come on screen: draw it if it was waiting."""
        if self.stale and self.pending is None:
            self.request_draw()

    # ---- items in data coordinates

    def add_marker(self, name, radius=8):
        item = self.canvas.create_oval(0, 0, 0, 0, outline=RING, width=2, state="hidden")
        self.markers[name] = [item, None, None, radius]

    MARKER_STYLES = {"ring": ("", "#e00000"), "mirror": ("white", "black")}      # (fill, outline): the cell under the pointer; the same cell in the other plot

    def set_marker(self, name, x, y, style="ring"):
        m = self.markers[name]
        m[1], m[2] = x, y
        fill, outline = self.MARKER_STYLES[style]
        self.canvas.itemconfig(m[0], fill=fill, outline=outline)
        self.place_marker(m)

    def place_marker(self, m):
        if m[1] is None:
            self.canvas.itemconfig(m[0], state="hidden")
            return
        px, py = self.to_canvas(m[1], m[2])
        r = m[3] * self.marker_factor
        self.canvas.coords(m[0], px - r, py - r, px + r, py + r)
        self.canvas.itemconfig(m[0], state="normal")
        self.canvas.tag_raise(m[0])

    def add_rect(self, x0, y0, x1, y1):
        item = self.canvas.create_rectangle(0, 0, 0, 0, outline="#ff8c00", width=2, dash=(6, 3))
        self.rects.append([item, x0, y0, x1, y1])

    def enable_scalebar(self):
        self.scalebar = (self.canvas.create_line(0, 0, 0, 0, fill="black", width=3), self.canvas.create_text(0, 0, anchor="s", font=("Segoe UI", 9, "bold"), fill="black"))

    def place_scalebar(self):
        w, h = self.size()
        view_mm = w / self.scale
        cands = [u for u in SCALEBAR_UM if u <= view_mm * 1000 * 0.22]
        um = cands[-1] if cands else SCALEBAR_UM[0]
        x0, y = w * 0.06, h * 0.94
        x1 = x0 + um / 1000 * self.scale
        self.canvas.coords(self.scalebar[0], x0, y, x1, y)
        self.canvas.coords(self.scalebar[1], (x0 + x1) / 2, y - 4)
        self.canvas.itemconfig(self.scalebar[1], text=f"{um:g} µm" if um < 1000 else f"{um / 1000:g} mm")
        self.canvas.tag_raise(self.scalebar[0])
        self.canvas.tag_raise(self.scalebar[1])

    def clear_labels(self):
        for it in self.labels:
            self.canvas.delete(it[0])
        self.labels = []

    def add_label(self, text, x, y, color="black"):
        item = self.canvas.create_text(0, 0, text=text, font=("Segoe UI", self.label_size, "bold"), fill=color)
        self.labels.append([item, x, y])

    def set_label_factor(self, factor):
        self.label_factor = factor
        self.label_size = None                      # (so the next update applies the new size)
        self.update_label_size()

    def update_label_size(self):
        zoom = min(self.scale / self.fit_scale(), LABEL_ZOOM_CAP)
        size = max(int(round(8 * self.label_factor * (1 + 0.75 * (zoom - 1)))), 1)
        if size != self.label_size:
            self.label_size = size
            for it in self.labels:
                self.canvas.itemconfig(it[0], font=("Segoe UI", size, "bold"))

    def reposition(self):
        if self.labels:
            self.update_label_size()
        if self.scalebar:
            self.place_scalebar()
        for m in self.markers.values():
            self.place_marker(m)
        for it, x0, y0, x1, y1 in self.rects:
            a, b = self.to_canvas(x0, y0)
            c, d = self.to_canvas(x1, y1)
            self.canvas.coords(it, a, b, c, d)
            self.canvas.tag_raise(it)
        w, h = self.size()
        for it, x, y in self.labels:
            px, py = self.to_canvas(x, y)
            if 0 <= px <= w and 0 <= py <= h:
                self.canvas.coords(it, px, py)
                self.canvas.itemconfig(it, state="normal")
                self.canvas.tag_raise(it)
            else:
                self.canvas.itemconfig(it, state="hidden")

    def nearest(self, tree, px, py):
        """Index (into the tree's points) of the point within HOVER_RADIUS_PX of the canvas position, or None."""
        x, y = self.to_data(px, py)
        dist, k = tree.query([x, y], distance_upper_bound=HOVER_RADIUS_PX / self.scale)
        return None if not np.isfinite(dist) else int(k)


# ---------------------------------------------------------------------------------------------------------------------- the window

class UmapWindow(ExportsMixin):
    def __init__(self, adata, abc_cache, imputed_state, adata_backed, rois, run_folder, ensure_imputed, title, parent=None, host=None, params=None):
        self.adata, self.abc_cache = adata, abc_cache
        self.host = host or {}             # the browser's own functions this window uses (settings dialog, UMAP variants, the exports)
        self.run_folder = run_folder
        self.params = dict(params) if params else None
        self.viewer_settings = self.host["load_viewer_settings"]() if "load_viewer_settings" in self.host else {"hover_enabled": True, "gene456_shape": "small circle", "umap_circle_size": "Default", "persist_settings": False}
        self.base_title = title
        self.imputed_state = imputed_state if imputed_state is not None else {"adata": None, "load_thread": None, "load_error": None}
        self.ensure_imputed = ensure_imputed
        self.own_root = parent is None
        self.root = tk.Tk() if parent is None else tk.Toplevel(parent)
        self.root.title(title)
        self.root.geometry("1750x950")
        loading = ttk.Label(self.root, text="Preparing the UMAP viewer...", font=("Segoe UI", 14), padding=40)
        loading.pack()
        self.root.update()
        self.msgs = queue.Queue()
        self.busy = False
        self.gc_was_enabled = gc.isenabled()
        gc.disable()
        self.mode = "all"                  # "all" | "single" | "gene" | "imputed"
        self.level = "subclass" if "subclass" in adata.obs.columns else "class"
        self.shown_gene = None
        self.shown_dataset = "standard"
        self.values = None                 # the shown gene's value for each cell (as coloured), or None
        self.tree = None
        self.hover_job = self.group_job = None
        self.gene_cache = None             # ((genes, imputed, scale), values read) of the genes shown, so a legend click re-uses them
        self.hidden = set()                # the genes switched off by a legend click
        self.multi_rows = None
        self.last_group_click = None       # (group, whether it was switched to gray) of the last click on a legend row: where a shift-click starts
        self.hover_group = None            # the legend row the pointer is on
        self.single_names = {}             # Single group(s): number -> full name of each group
        self.off_groups = set()            # the groups switched to gray by a click on their legend swatch (All groups: their names; Single group(s): their numbers)
        self.last_ids = None               # the groups typed in Single group(s): level, targets, colors, cells
        self.export_cat = None             # what the categorical colours mean (for a saved legend)
        self.label_info = []
        self.view_kind, self.view_token, self.view_title = "all", "all", ""
        self.hover_key = None
        self.variants = {}                 # UMAP parameter key -> (coordinates, Leiden labels), the ones already shown
        self.load_data(adata_backed, rois)
        if self.host and self.params is not None:
            self.variants[self.host["umap_params_key"](self.params)] = (self.coords.copy(), self.info.get("leiden"))
        loading.destroy()
        self.build()
        self.recolor()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda e: self.close())
        self.poll()

    # ---- data

    def load_data(self, adata_backed, rois):
        a = self.adata
        self.coords = np.asarray(a.obsm["X_umap"], dtype=float)[:, :2]
        self.n = len(self.coords)
        self.has_counts = "counts" in a.layers
        sec_col = core.SECTION_COL
        spatial = core.load_section_spatial_coords(a, self.abc_cache) if sec_col in a.obs.columns else None
        self.section_of = a.obs[sec_col].to_numpy() if sec_col in a.obs.columns else None
        ok = spatial is not None and "x" in spatial.columns and "y" in spatial.columns
        self.cx = pd.to_numeric(spatial["x"], errors="coerce").to_numpy() if ok else None
        self.cy = pd.to_numeric(spatial["y"], errors="coerce").to_numpy() if ok else None
        self.sections = core.sorted_sections_descending(a.obs[sec_col].dropna().unique().tolist()) if ok else []
        self.info = {k: (a.obs[k].astype(str).to_numpy() if k in a.obs.columns else None) for k in LEVEL_NAMES}
        self.levels = [k for k in LEVEL_NAMES if self.info[k] is not None]
        # the section panels' background: every cell of each section (from the unfiltered data), when given
        self.bg = None
        self.adata_backed = adata_backed
        if ok and adata_backed is not None:
            try:
                bsec = core.get_section_labels(adata_backed, self.abc_cache)
                bsp = core.load_section_spatial_coords(adata_backed, self.abc_cache)
                if bsp is not None and "x" in bsp.columns:
                    codes, uniques = pd.factorize(bsec.to_numpy())
                    order = np.argsort(codes, kind="stable")
                    starts = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
                    where = {u: order[starts[i]:starts[i + 1]] for i, u in enumerate(uniques)}
                    self.bg = {"ids": adata_backed.obs_names, "sec_arr": bsec.to_numpy(), "z": (pd.to_numeric(bsp["z"], errors="coerce").to_numpy() if "z" in bsp.columns else None),
                               "x": pd.to_numeric(bsp["x"], errors="coerce").to_numpy(), "y": pd.to_numeric(bsp["y"], errors="coerce").to_numpy(),
                               "where": where, "levels": {k: (bsp[k].astype(str).to_numpy() if k in bsp.columns else None) for k in LEVEL_NAMES}}
            except Exception as e:
                print(f"Could not load the all-cell background for the section panels: {e}", flush=True)
        self.run_idx = {}
        if ok:
            codes, uniques = pd.factorize(self.section_of)
            for i, u in enumerate(uniques):
                if pd.notna(u):
                    self.run_idx[u] = np.flatnonzero(codes == i)
        self.id_of_cell = {}
        self.rois = []
        for r in rois or []:
            try:
                vals = [float(r[k]) for k in ("x_min", "x_max", "y_min", "y_max")]
            except (TypeError, ValueError, KeyError):
                continue
            if all(np.isfinite(vals)):
                self.rois.append((str(r["section"]), *vals))
        self.cache = {}
        self.run_index = pd.Index(a.obs_names)
        self.bg_under = None
        if self.info.get("class") is not None:
            cid = core.extract_leading_numeric_id(pd.Series(self.info["class"])).to_numpy()
            frac = float(np.isin(cid, list(core.NON_NEURON_CLASS_IDS)).mean())
            self.bg_under = "nonneuron" if frac < 0.5 else "neuron"           # (the minority kind is drawn underneath)
        # ...unless the run says which cell type it was made for: the launch screen's choice is in the run folder's name
        name = Path(self.run_folder).name.lower() if self.run_folder is not None else ""
        for prefix, under in (("umap_neurons_", "nonneuron"), ("umap_nonneurons_", "neuron"), ("umap_all_", None)):
            if name.startswith(prefix):
                self.bg_under = under

    def category_codes(self, level):
        """(codes of each run cell, uniques) of a level; and for the all-cell background (codes, uniques) over the same uniques."""
        key = ("codes", level)
        if key not in self.cache:
            codes, uniques = pd.factorize(self.info[level])
            self.cache[key] = (codes, uniques)
        return self.cache[key]

    def bg_codes(self, level):
        key = ("bgcodes", level)
        if key not in self.cache:
            arr = self.bg["levels"].get(level) if self.bg else None
            if arr is None:
                self.cache[key] = None
            else:
                codes, uniques = self.category_codes(level)
                self.cache[key] = pd.Index(uniques).get_indexer(arr)       # (-1: a group none of the run's cells is in)
        return self.cache[key]

    # ---- the window

    def build(self):
        self.pw = pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)
        # the section panels
        left = ttk.Frame(pw)
        bars = ttk.Frame(left, padding=(6, 4))
        bars.pack(side=tk.TOP, fill=tk.X)
        self.var_tile = tk.IntVar(bars, DEFAULT_TILE_INDEX)
        self.var_dot = tk.IntVar(bars, DEFAULT_DOT_INDEX)
        self.var_marker = tk.IntVar(bars, DEFAULT_MARKER_INDEX)
        dot_names = list(DOT_SIZES)
        self.dot_factors = list(DOT_SIZES.values())
        marker_names = ("Small", "Medium", "Large")
        dot_um = lambda i: f"{self.dot_factors[i] * DOT_MM * 1000:g} \u00b5m"
        self.sliders = [
            self.add_slider(bars, 0, "Section map size:", self.var_tile, len(TILE_COLUMNS), self.layout_tiles,
                            lambda i: "full width" if TILE_COLUMNS[i] == 1 else f"1/{TILE_COLUMNS[i]} width", None),
            self.add_slider(bars, 1, "Dot size:", self.var_dot, len(dot_names), self.apply_dot_size, lambda i: dot_names[i],
                            lambda i: f"Dots are {dot_um(i)} across"),
            self.add_slider(bars, 2, "Marker size:", self.var_marker, len(MARKER_SIZES), self.apply_marker_size, lambda i: marker_names[i], None)]
        ttk.Button(bars, text="Reset to default", command=self.reset_sliders).grid(row=0, column=0, rowspan=3, sticky="w")
        bars.columnconfigure(3, weight=1)
        self.tile_scroll = tk.Canvas(left, highlightthickness=0, bg="#f4f4f4")
        sb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tile_scroll.yview)
        self.tile_scroll.configure(yscrollcommand=lambda a, b: (sb.set(a, b), self.root.after_idle(self.refresh_visible)))
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        sb.bind("<MouseWheel>", lambda e: self.tile_scroll.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.tile_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tile_frame = tk.Frame(self.tile_scroll, bg="#f4f4f4")
        self.tile_window = self.tile_scroll.create_window(0, 0, window=self.tile_frame, anchor="nw")
        self.tile_frame.bind("<Configure>", lambda e: self.tile_scroll.configure(scrollregion=self.tile_scroll.bbox("all")))
        self.tile_scroll.bind("<Configure>", lambda e: self.layout_tiles())
        self.tile_scroll.bind("<MouseWheel>", lambda e: self.tile_scroll.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        pw.add(left, weight=2)
        # the UMAP
        mid = ttk.Frame(pw)
        umap_bar = ttk.Frame(mid, padding=(6, 4))
        umap_bar.pack(side=tk.TOP, fill=tk.X)
        self.var_label = tk.IntVar(umap_bar, DEFAULT_LABEL_INDEX)
        self.umap_sliders = [self.add_slider(umap_bar, 0, "Label size:", self.var_label, len(LABEL_SIZES), self.apply_label_size,
                                             lambda i: LABEL_SIZE_NAMES[i], None)]
        ttk.Button(umap_bar, text="Reset to default", command=self.reset_umap_sliders).grid(row=0, column=0, sticky="w")
        umap_bar.columnconfigure(3, weight=1)
        c = self.coords
        padx = (c[:, 0].max() - c[:, 0].min()) * 0.05 or 1.0
        pady = (c[:, 1].max() - c[:, 1].min()) * 0.05 or 1.0
        self.umap = PointCanvas(mid, (c[:, 0].min() - padx, c[:, 0].max() + padx, c[:, 1].min() - pady, c[:, 1].max() + pady), flip_y=True)
        self.umap.canvas.pack(fill=tk.BOTH, expand=True)
        self.umap.add_marker("hover", 9)
        self.umap.on_motion = self.umap_motion
        self.umap.on_view = None
        self.umap_tree = cKDTree(c)
        pw.add(mid, weight=3)
        side = ttk.Frame(pw, padding=10, width=420)
        side.pack_propagate(False)
        self.build_side(side)
        pw.add(side, weight=0)
        self.side_pane = side
        self.build_tiles()
        self.root.after(400, self.fit_left_pane)

    def fit_left_pane(self, again=True):
        """Once, at the start: the section pane gets about a third of the window (the panels fill whatever width it has)."""
        if not self.root.winfo_viewable():
            return
        total = self.pw.winfo_width()
        side_w = int(self.side_pane.cget("width")) or 420
        if not again and self.side_pane.winfo_width() >= side_w - 5:
            return                                                      # (the second look: nothing to put right)
        edge = total - side_w                                           # (the side panel keeps its own width, at the right)
        want = min(max(total * 0.33, 520), total * 0.5, edge - 300)     # (and the UMAP keeps at least 300)
        self.pw.sashpos(0, int(want))
        self.pw.sashpos(1, int(edge))
        if again:                                                       # (once more after the window has settled)
            self.root.after(600, lambda: self.fit_left_pane(False))

    def build_side(self, f):
        self.color_label = ttk.Label(f, text="Color by", font=("Segoe UI", 13, "bold"))
        self.color_label.pack(anchor="w")
        self.var_mode = tk.StringVar(f, "all")
        for value, text in (("all", "All groups"), ("single", "Specific group(s)"), ("gene", "Gene"), ("imputed", "Imputed Gene")):
            ttk.Radiobutton(f, text=text, value=value, variable=self.var_mode, command=self.on_mode).pack(anchor="w")
        self.level_frame = ttk.LabelFrame(f, text="Level", padding=6)
        self.level_frame.pack(fill=tk.X, pady=(8, 0))
        self.var_level = tk.StringVar(f, self.level)
        self.build_level_radios()
        self.box = ttk.Frame(f)
        self.box.pack(fill=tk.X, pady=(8, 0))
        # a group number box (Specific group(s))
        self.id_frame = ttk.LabelFrame(self.box, text="Group number(s)", padding=6)
        ttk.Label(self.id_frame, text="One or more, separated by commas (Enter):", foreground="gray30", wraplength=300).pack(anchor="w")
        self.var_ids = tk.StringVar(f, "")
        e = ttk.Entry(self.id_frame, textvariable=self.var_ids)
        e.pack(fill=tk.X, pady=(2, 0))
        e.bind("<Return>", lambda ev: self.recolor())
        self.id_entry = e
        # the gene box
        self.gene_frame = ttk.LabelFrame(self.box, text="Gene(s)", padding=6)
        ttk.Label(self.gene_frame, text="Up to six, separated by commas: the first three red / green / blue circles, the next three the same as small dots:", foreground="gray30", wraplength=300).pack(anchor="w")
        self.var_gene = tk.StringVar(f, "")
        self.gene_entry = ttk.Entry(self.gene_frame, textvariable=self.var_gene)
        self.gene_entry.pack(fill=tk.X)
        self.gene_entry.bind("<KeyRelease>", self.on_gene_key)
        self.gene_entry.bind("<Return>", lambda ev: self.recolor())
        self.gene_entry.bind("<Down>", lambda ev: self.focus_suggestions())
        self.sugg = tk.Listbox(self.gene_frame, height=6, exportselection=False, activestyle="none")
        self.sugg.bind("<ButtonRelease-1>", lambda ev: self.pick_suggestion())
        self.sugg.bind("<Return>", lambda ev: self.pick_suggestion())
        sc = ttk.Frame(self.gene_frame)
        sc.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(sc, text="Scale:").pack(side=tk.LEFT)
        self.var_scale = tk.StringVar(f, "log")
        for value, text in (("linear", "Linear"), ("log", "Log")):
            ttk.Radiobutton(sc, text=text, value=value, variable=self.var_scale, command=self.on_scale).pack(side=tk.LEFT, padx=(8, 0))
        self.cbar = tk.Canvas(self.gene_frame, width=300, height=70, highlightthickness=0)
        self.cbar.pack(fill=tk.X, pady=(6, 0))
        self.cbar_photo = None
        self.cbar.bind("<Button-1>", self.on_legend_click)
        # the legend for Single group(s)
        self.legend = tk.Canvas(f, width=300, height=10, highlightthickness=0)
        self.group_list = ttk.Frame(f)                      # the groups' numbers and colors (All groups), scrollable
        self.group_header = tk.Canvas(self.group_list, height=20, highlightthickness=0, cursor="hand2")      # the column headings: click to sort
        self.group_header.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
        self.group_header.bind("<Button-1>", self.on_group_header_click)
        self.group_sort = ("id", False)                    # (column, descending): kept when the level changes
        self.group_canvas = tk.Canvas(self.group_list, width=300, height=40, highlightthickness=0, bg="white")
        gsb = ttk.Scrollbar(self.group_list, orient=tk.VERTICAL, command=self.group_canvas.yview)
        self.group_canvas.configure(yscrollcommand=gsb.set)
        gsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.group_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.group_canvas.bind("<Button-1>", self.on_group_list_click)
        self.group_canvas.bind("<Motion>", self.on_group_list_motion)
        self.group_canvas.bind("<Leave>", lambda e: self.show_legend_group(None))
        self.group_canvas.config(cursor="hand2")
        self.legend.bind("<Button-1>", self.on_single_legend_click)
        self.legend.bind("<Motion>", self.on_single_legend_motion)
        self.legend.bind("<Leave>", lambda e: self.show_legend_group(None))
        self.legend.config(cursor="hand2")
        self.group_canvas.bind("<MouseWheel>", lambda e: self.group_canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.build_buttons(f)
        hov = ttk.LabelFrame(f, text="Under the pointer", padding=6)
        hov.pack(fill=tk.X, side=tk.BOTTOM, pady=(6, 0))
        self.var_cell = tk.StringVar(f, "Hover over a cell in the UMAP or a section.")
        self.cell_hint = ttk.Label(hov, textvariable=self.var_cell, wraplength=320, justify=tk.LEFT)
        self.cell_hint.pack(anchor="w")
        # the hovered cell as a table: what it is / its number / its name, each in a column of its own
        self.cell_grid = ttk.Frame(hov)
        self.cell_grid.columnconfigure(2, weight=1)
        self.cell_grid.bind("<Configure>", lambda e: self.fit_cell_names())
        self.cell_cells = []
        for r in range(12):
            a = ttk.Label(self.cell_grid, foreground="gray35", font=("Segoe UI", 9, "bold"))
            b = ttk.Label(self.cell_grid, width=6, anchor="e", font=("Segoe UI", 10, "bold"))
            c = ttk.Label(self.cell_grid, wraplength=190, justify=tk.LEFT)
            a.grid(row=r, column=0, sticky="nw", padx=(0, 8))
            b.grid(row=r, column=1, sticky="ne", padx=(0, 8))
            c.grid(row=r, column=2, sticky="nw")
            self.cell_cells.append((a, b, c))
        self.var_status = tk.StringVar(f, "")
        ttk.Label(f, textvariable=self.var_status, foreground="gray30", wraplength=300, justify=tk.LEFT).pack(anchor="w", side=tk.BOTTOM)
        ttk.Label(f, text="wheel: zoom   drag: pan   double-click: whole view", foreground="gray40", wraplength=300).pack(anchor="w", side=tk.BOTTOM, pady=(0, 6))
        self.update_frames()

    def build_level_radios(self):
        for w in self.level_frame.winfo_children():
            w.destroy()
        for k, name in enumerate(self.levels):
            ttk.Radiobutton(self.level_frame, text=LEVEL_NAMES[name], value=name, variable=self.var_level, command=self.on_level).grid(row=k // 2, column=k % 2, sticky="w", padx=(0, 14))

    def build_buttons(self, f):
        ttk.Style(self.root).configure("Side.TButton", font=("Segoe UI", 9))
        bar = ttk.Frame(f)
        bar.pack(fill=tk.X, side=tk.TOP, pady=(0, 10), before=self.color_label)
        bar.columnconfigure(0, weight=1)
        bar.columnconfigure(1, weight=1)
        self.group_buttons = []
        rows = ((("Settings...", self.open_settings, False), ("Open data folder", self.open_data_folder, False)),
                (("Save section maps", self.save_section_maps, False), ("Save UMAP", self.save_umap, False)),
                (("Export DEGs", self.export_degs, True), ("Export expression", self.export_expression, True)),
                (("Export coordinates", self.export_coordinates, True), ("3D view", self.show_3d, True)),
                (("3D in browser", self.show_3d_browser, True), ("Close", self.close, False)))
        for r, pair in enumerate(rows):
            for c, (text, cmd, grouped) in enumerate(pair):
                b = ttk.Button(bar, text=text, command=cmd, style="Side.TButton")
                b.grid(row=r, column=c, sticky="ew", padx=2, pady=2)
                if grouped:
                    self.group_buttons.append(b)
        self.update_group_buttons()

    def update_group_buttons(self):
        """The group exports and 3D views are for the groups typed in Single group(s)."""
        for b in self.group_buttons:
            b.state(["!disabled"] if self.mode == "single" else ["disabled"])

    def update_frames(self):
        for w in (self.id_frame, self.gene_frame, self.legend, self.group_list):
            w.pack_forget()
        if self.mode == "single":
            self.id_frame.pack(fill=tk.X)
            self.id_entry.focus_set()
        elif self.mode in ("gene", "imputed"):
            self.gene_frame.pack(fill=tk.X)
            self.gene_entry.focus_set()

    # ---- the section panels

    def build_tiles(self):
        self.tiles = []
        if not self.sections:
            return
        boxes = {}
        for s in self.sections:
            idx = self.run_idx[s]
            xs, ys = self.cx[idx], self.cy[idx]
            if self.bg is not None and s in self.bg["where"]:          # (the whole section, not only the run's cells in it)
                w = self.bg["where"][s]
                xs, ys = self.bg["x"][w], self.bg["y"][w]
            ok = np.isfinite(xs) & np.isfinite(ys)
            if ok.any():
                boxes[s] = (xs[ok].min(), xs[ok].max(), ys[ok].min(), ys[ok].max())
        span_x = max(b[1] - b[0] for b in boxes.values()) * 1.04 or 1.0          # (one size for every panel, so they share a scale)
        span_y = max(b[3] - b[2] for b in boxes.values()) * 1.04 or 1.0
        self.tile_aspect = span_x / span_y
        for s in self.sections:
            if s not in boxes:
                continue
            b = boxes[s]
            mx, my = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
            fr = tk.Frame(self.tile_frame, bg="#f4f4f4")
            token = core.sanitize_section_token(s)
            title = tk.StringVar(fr, token)
            tk.Label(fr, textvariable=title, bg="#f4f4f4", font=("Segoe UI", 9, "bold")).pack(anchor="w")
            pc = PointCanvas(fr, (mx - span_x / 2, mx + span_x / 2, my - span_y / 2, my + span_y / 2), flip_y=False, width=TILE_PX, height=int(TILE_PX / self.tile_aspect))
            pc.canvas.pack()
            pc.canvas.bind("<Shift-MouseWheel>", lambda e: self.tile_scroll.yview_scroll(-1 if e.delta > 0 else 1, "units"))     # (the plain wheel zooms)
            pc.add_marker("hover", 7)
            pc.plus_arm = 2
            pc.dot_factor = self.dot_factors[self.var_dot.get()]
            pc.marker_factor = MARKER_SIZES[self.var_marker.get()]
            pc.visible_fn = lambda fr=fr: self.frame_visible(fr)
            if not self.tiles:
                pc.enable_scalebar()                       # (the first section's panel carries the scale bar)
            tile = {"label": s, "frame": fr, "pc": pc, "title": title, "token": token, "tree": None}
            pc.on_motion = lambda px, py, t=tile: self.tile_motion(t, px, py)
            pc.on_change = lambda zoom, dx, dy, t=tile: self.link_tiles(t, zoom, dx, dy)
            for r in self.rois:
                if r[0] == str(s) or core.sanitize_section_token(r[0]) == token:
                    pc.add_rect(r[1], r[3], r[2], r[4])
            self.tiles.append(tile)
        self.tile_cols = self.tile_px = 0
        self.layout_tiles()

    def add_slider(self, parent, row, text, var, n, command, value_text, tip_text):
        """A slider with `n` stops (0 .. n-1): it jumps from stop to stop, shows `value_text(stop)` beside it, and calls `command()` when the
        stop changes. `tip_text(stop)`, if given, is shown in a tip while the pointer is over the slider. Returns the slider."""
        ttk.Label(parent, text=text).grid(row=row, column=1, sticky="w", padx=(12, 6))
        shown = ttk.Label(parent, width=12)
        shown.grid(row=row, column=3, sticky="w", padx=(6, 0))
        scale = ttk.Scale(parent, from_=0, to=n - 1, orient=tk.HORIZONTAL, length=130)
        scale.grid(row=row, column=2, sticky="w")

        def moved(v):
            i = int(round(float(v)))
            if abs(float(v) - i) > 1e-6:
                scale.set(i)                               # (snaps to the stop, which calls this again)
                return
            shown.config(text=value_text(i))
            if var.get() != i:
                var.set(i)
                command()

        scale.config(command=moved)
        scale.set(var.get())
        shown.config(text=value_text(var.get()))
        if tip_text is not None:
            HoverTip(scale, lambda: tip_text(var.get()))
        return scale

    def apply_label_size(self):
        """The UMAP's Label size slider: the group numbers drawn on the UMAP, bigger or smaller."""
        self.umap.set_label_factor(LABEL_SIZES[self.var_label.get()])

    def reset_umap_sliders(self):
        for scale, default in zip(self.umap_sliders, (DEFAULT_LABEL_INDEX,)):
            scale.set(default)

    def reset_sliders(self):
        for scale, default in zip(self.sliders, (DEFAULT_TILE_INDEX, DEFAULT_DOT_INDEX, DEFAULT_MARKER_INDEX)):
            scale.set(default)

    def layout_tiles(self):
        """The panels are a third, a half or all of the pane's width (the Section map size buttons), as many to a row as that gives; the
        pane scrolls to the rest."""
        if not getattr(self, "tiles", None):
            return
        cols = TILE_COLUMNS[self.var_tile.get()]
        w = max(self.tile_scroll.winfo_width(), 100)
        px = int(max((w - 4) // cols - 8 - 2, MIN_TILE_PX))          # (8: the gaps either side of a panel; 2: its canvas border)
        if cols == self.tile_cols and px == self.tile_px:
            return
        self.tile_cols, self.tile_px = cols, px
        for k, t in enumerate(self.tiles):
            t["pc"].canvas.config(width=px, height=int(px / self.tile_aspect))
            t["frame"].grid(row=k // cols, column=k % cols, padx=4, pady=3, sticky="nw")
        self.root.after_idle(self.refresh_visible)

    def apply_dot_size(self):
        """The Dot size buttons: redraw the section panels with bigger or smaller dots (the ones off screen when they come on screen)."""
        f = self.dot_factors[self.var_dot.get()]
        for t in self.tiles:
            t["pc"].dot_factor = f
            if self.frame_visible(t["frame"]):
                t["pc"].request_draw()
            else:
                t["pc"].stale = True

    def apply_marker_size(self):
        """The Marker size slider: the hover ring and the + marks of the section panels, bigger or smaller."""
        f = MARKER_SIZES[self.var_marker.get()]
        for t in self.tiles:
            pc = t["pc"]
            pc.marker_factor = f
            for m in pc.markers.values():
                pc.place_marker(m)
            if pc.pluses is not None:
                pc.request_compose()

    def link_tiles(self, source, zoom, dx, dy):
        """Zooming or panning one section does the same to all the others."""
        for t in self.tiles:
            if t is not source:
                t["pc"].follow(zoom, dx, dy)

    def section_tile(self, label):
        for t in self.tiles:
            if t["label"] == label:
                return t
        return None

    # ---- colouring

    def on_mode(self):
        if self.busy:
            self.var_mode.set(self.mode)
            return
        mode = self.var_mode.get()
        if mode == "imputed" and self.imputed_state["adata"] is None:
            ok = bool(self.ensure_imputed(self.imputed_state, self.abc_cache)) if self.ensure_imputed else False
            if not ok:
                self.var_mode.set(self.mode)
                return
        self.mode = mode
        self.update_frames()
        self.update_group_buttons()
        self.recolor()

    def on_level(self):
        if self.busy:
            self.var_level.set(self.level)
            return
        self.level = self.var_level.get()
        if self.mode in ("all", "single"):
            self.recolor()

    def on_scale(self):
        if self.mode in ("gene", "imputed") and self.var_gene.get().strip():
            self.recolor()

    def recolor(self, keep_off=False):
        if not keep_off:
            self.last_group_click = None
            self.off_groups = set()                      # (a new level, mode or list of groups starts with every group colored)
        self.var_status.set("")
        self.legend.pack_forget()
        self.group_list.pack_forget()
        if self.mode == "all":
            self.color_all()
        elif self.mode == "single":
            self.color_single()
        else:
            self.color_gene()

    def dot_scale(self):
        return self.host.get("umap_circle_sizes", {}).get(self.viewer_settings.get("umap_circle_size", "Default"), 1.0)

    def apply(self, rgb_run, order_run, bg_rgb_fn, labels=None, extra_umap_layers=(), dark=False):
        """Show the colouring. `rgb_run`: (n, 3) colour of each run cell; `order_run`: the order to draw them in (lowest first);
        `bg_rgb_fn(section label)` -> (x, y, rgb) of the section panel's background cells, or None."""
        self.rgb_run = rgb_run
        o = order_run
        bgc = MULTI_BG if dark else (255, 255, 255)
        self.umap.set_background(bgc)
        for t in self.tiles:
            t["pc"].set_background(bgc)
        self.umap.set_layers([{"x": self.coords[o, 0], "y": self.coords[o, 1], "rgb": rgb_run[o], "size": 4 * self.dot_scale()}] + list(extra_umap_layers))
        self.umap.clear_labels()
        self.label_info = list(labels or [])
        for text, x, y, color in labels or []:
            self.umap.add_label(text, x, y, color)
        self.umap.reposition()
        self.tile_queue = list(self.tiles)
        self.tile_fn = bg_rgb_fn
        if not getattr(self, "tile_job", None):
            self.tile_job = self.root.after(1, self.tile_step)

    def frame_visible(self, fr):
        """Whether a section panel is (at least partly) in the scrolled pane."""
        top, h = fr.winfo_y(), fr.winfo_height()
        vt, vh = self.tile_scroll.canvasy(0), self.tile_scroll.winfo_height()
        return top + h > vt and top < vt + vh

    def refresh_visible(self):
        """Panels that were off screen when they were recolored or moved are made now that they are on screen."""
        for t in self.tiles:
            if t.get("dirty") and self.frame_visible(t["frame"]):
                self.build_tile(t)
            t["pc"].refresh()

    def tile_step(self):
        self.tile_job = None
        t0 = time.perf_counter()
        while self.tile_queue and time.perf_counter() - t0 < 0.03:
            t = self.tile_queue.pop(0)
            t["dirty"] = True
            if self.frame_visible(t["frame"]):
                self.build_tile(t)
        if self.tile_queue:
            self.tile_job = self.root.after(1, self.tile_step)

    def build_tile(self, t):
        """The section's dots, back to front: the cells of the other kind (half size), then the gray cells of the run's kind (those of the
        whole section, then the run's own), and last all the colored ones, so a colored cell is never hidden by a gray one."""
        t["dirty"] = False
        idx = self.run_idx[t["label"]]
        layers = []
        bgl = self.tile_fn(t["label"])
        sources = []                     # (x, y, rgb) of the run's kind, the section's cells first
        if bgl is not None:
            x, y, rgb, w = bgl
            if self.bg_under is not None and self.bg["levels"].get("class") is not None:
                # the cells of the other kind (non-neurons in a neuron run) go underneath, so they never cover the ones being looked at
                ids = core.extract_leading_numeric_id(pd.Series(self.bg["levels"]["class"][w])).to_numpy()
                non_neuron = np.isin(ids, list(core.NON_NEURON_CLASS_IDS))
                other = non_neuron if self.bg_under == "nonneuron" else ~non_neuron
                layers.append({"x": x[other], "y": y[other], "rgb": rgb[other], "size_mm": DOT_MM * OTHER_CELLS_DOT_RATIO})
                x, y, rgb = x[~other], y[~other], rgb[~other]
            sources.append((x, y, rgb))
        # the run's own cells of this section, in drawing order (lowest first)
        sel = idx[np.argsort(self.rank_run[idx], kind="stable")]
        sources.append((self.cx[sel], self.cy[sel], self.rgb_run[sel]))
        grays = []
        for x, y, rgb in sources:
            gray = (rgb == GREY).all(axis=1) | (rgb == LIGHT_GREY).all(axis=1)
            grays.append((x, y, rgb, gray))
        for x, y, rgb, gray in grays:
            layers.append({"x": x[gray], "y": y[gray], "rgb": rgb[gray], "size_mm": DOT_MM})
        for x, y, rgb, gray in grays:
            layers.append({"x": x[~gray], "y": y[~gray], "rgb": rgb[~gray], "size_mm": DOT_MM})
        t["pc"].set_layers(layers)

    def color_all(self):
        level = self.level
        if self.info.get(level) is None:
            self.var_status.set(f"No '{level}' column on this dataset.")
            return
        codes, uniques = self.category_codes(level)
        counts = np.bincount(codes[codes >= 0], minlength=len(uniques))
        ranked = np.argsort(-counts, kind="stable")[:MAX_COLORED_CATEGORIES]
        lut = np.tile(GREY, (len(uniques), 1))
        for r, u in enumerate(ranked):
            lut[u] = GREY if str(uniques[u]) in self.off_groups else rank_color(r)
        rgb = lut[codes]
        self.rank_run = np.zeros(self.n, np.int32)
        bgc = self.bg_codes(level) if self.bg else None

        def bg_fn(label):
            if not self.bg:
                return None
            w = self.bg["where"].get(label)
            if w is None:
                return None
            c = bgc[w] if bgc is not None else np.full(len(w), -1)
            col = np.where((c >= 0)[:, None], lut[np.maximum(c, 0)], LIGHT_GREY)
            return self.bg["x"][w], self.bg["y"][w], col.astype(np.uint8), w

        # each coloured group's number at its centre
        cent = pd.DataFrame({"g": codes, "x": self.coords[:, 0], "y": self.coords[:, 1]})
        on = [u for u in ranked if str(uniques[u]) not in self.off_groups]
        cent = cent[cent["g"].isin(on)].groupby("g").median()
        labels = []
        for g, row in cent.iterrows():
            m = re.match(r"^\s*(\d+)", str(uniques[g]))
            dark = (lut[g].astype(float) * 0.55).astype(int)               # (the group's own color, darker)
            labels.append((m.group(1) if m else str(uniques[g]), row["x"], row["y"], "#%02x%02x%02x" % tuple(dark)))
        self.apply(rgb, np.arange(self.n), bg_fn, labels)
        self.view_kind, self.view_token = "all", f"all_{level}"
        self.view_title = f"All {LEVEL_NAMES[level].lower()}s"
        rank_of = np.full(len(uniques), -1)
        rank_of[ranked] = np.arange(len(ranked))
        self.export_cat = {"rank": np.where(codes >= 0, rank_of[np.maximum(codes, 0)], -1), "title": LEVEL_NAMES[level],
                           "entries": [(hexcolor(GREY if str(uniques[u]) in self.off_groups else rank_color(r)), str(uniques[u]), int(counts[u])) for r, u in enumerate(ranked)]}
        self.shown_values = None
        n_cat = int((counts > 0).sum())
        self.var_status.set(f"Showing all {n_cat} {LEVEL_NAMES[level].lower()}s" + (f"; the {len(ranked)} largest colored, the rest gray." if n_cat > len(ranked) else " (numbers at each center).") + " Click a group in the list to gray it or color it again.")
        self.values = None
        self.draw_group_list(ranked, uniques, counts)

    def draw_legend_rows(self, canvas, rows, x0):
        """The rows of a group legend, in columns: swatch, number, cell count (right-aligned) and the group's name (all names start at the same x).
        `rows`: (swatch color, number, count text, name, grayed out). Returns the width used."""
        font = tkfont.Font(root=canvas, family="Segoe UI", size=9)
        w_num = max((font.measure(r[1]) for r in rows), default=0)
        w_cnt = max((font.measure(r[2]) for r in rows), default=0)
        x_num, x_cnt, x_name = x0 + 22, x0 + 22 + w_num + 8 + w_cnt, x0 + 22 + w_num + 8 + w_cnt + 10
        self.legend_cols = (x_num, x_cnt - w_cnt, x_cnt, x_name)          # (where the number, count and name columns start; the count's right edge)
        for k, (col, num, cnt, name, off) in enumerate(rows):
            cy = 4 + 22 * k + 11
            fill = "gray55" if off else "black"
            canvas.create_rectangle(x0, cy - 6, x0 + 12, cy + 6, fill=col, outline="")
            canvas.create_text(x_num, cy, anchor="w", text=num, font=font, fill=fill)
            canvas.create_text(x_cnt, cy, anchor="e", text=cnt, font=font, fill=fill)
            canvas.create_text(x_name, cy, anchor="w", text=name, font=font, fill=fill)
        return x_name + max((font.measure(r[3]) for r in rows), default=0) + 8

    def draw_group_list(self, ranked, uniques, counts):
        """The colored groups, each with its swatch, number, cell count and name (the ones past the colored cap are gray and not listed),
        in the order of the Sort by buttons."""
        self.group_list_args = (ranked, uniques, counts)
        c = self.group_canvas
        c.delete("all")
        rows = []
        for r, u in enumerate(ranked):
            m = re.match(r"^\s*(\d+)", str(uniques[u]))
            off = str(uniques[u]) in self.off_groups
            name = re.sub(r"^\s*\d+\s*", "", str(uniques[u])) if m else ""
            rows.append({"id": int(m.group(1)) if m else 10 ** 9, "color": hexcolor(GREY if off else rank_color(r)),
                         "num": m.group(1) if m else str(uniques[u]), "count": int(counts[u]), "name": name, "off": off, "key": str(uniques[u])})
        column, descending = self.group_sort
        rows.sort(key=lambda r: (r["id"], r["key"]))                      # (ties stay in ID order)
        if column == "count":
            rows.sort(key=lambda r: r["count"], reverse=descending)
        elif column == "name":
            rows.sort(key=lambda r: (r["name"] or r["num"]).lower(), reverse=descending)
        elif descending:
            rows.reverse()
        self.group_rows = [r["key"] for r in rows]
        width = self.draw_legend_rows(c, [(r["color"], r["num"], f"({r['count']})", r["name"], r["off"]) for r in rows], 6)
        c.configure(scrollregion=(0, 0, max(300, width), 8 + 22 * len(rows)))
        self.draw_group_header()
        self.group_list.pack(fill=tk.BOTH, expand=True, pady=(6, 0), after=self.box)

    def draw_group_header(self):
        """The headings of the group list's columns, over the columns, each with an arrow: dark on the column it is sorted by (pointing the way it
        is sorted), light gray on the others (pointing the way a click on them would sort)."""
        h = self.group_header
        h.delete("all")
        x_num, x_cnt_left, x_cnt, x_name = self.legend_cols
        column, descending = self.group_sort
        font = tkfont.Font(root=h, family="Segoe UI", size=9, weight="bold")
        gap = font.measure(" ")
        for key, text, x, right in (("id", "#", x_num, False), ("count", "Cells", x_cnt, True), ("name", "Name", x_name, False)):
            active = key == column
            down = descending if active else key in GROUP_SORT_DESCENDING_FIRST
            arrow = "\u25bc" if down else "\u25b2"
            color = "gray20" if active else "gray75"
            w_arrow = font.measure(arrow)
            if right:                                     # (the text ends at the column's right edge, with the arrow there)
                h.create_text(x, 10, anchor="e", text=arrow, font=font, fill=color)
                h.create_text(x - w_arrow - gap, 10, anchor="e", text=text, font=font, fill="gray30")
                self.count_head_left = min(x_cnt_left, x - w_arrow - gap - font.measure(text)) - 2      # (where a click counts for Cells, not #)
            else:
                h.create_text(x, 10, anchor="w", text=text, font=font, fill="gray30")
                h.create_text(x + font.measure(text) + gap, 10, anchor="w", text=arrow, font=font, fill=color)

    def on_group_header_click(self, e):
        """Click a heading to sort by that column; click it again to reverse the order."""
        if self.mode != "all" or getattr(self, "group_list_args", None) is None:
            return
        x_num, x_cnt_left, x_cnt, x_name = self.legend_cols
        key = "id" if e.x < getattr(self, "count_head_left", x_cnt_left) else "count" if e.x < x_name else "name"
        column, descending = self.group_sort
        self.group_sort = (key, not descending) if key == column else (key, key in GROUP_SORT_DESCENDING_FIRST)
        self.draw_group_list(*self.group_list_args)
        self.group_canvas.yview_moveto(0)

    def parse_ids(self):
        ids, bad = [], []
        for tok in re.split(r"[,\s]+", self.var_ids.get().strip()):
            if not tok:
                continue
            try:
                v = int(tok)
            except ValueError:
                bad.append(tok)
                continue
            if v not in ids:
                ids.append(v)
        return ids, bad

    def color_single(self):
        level = self.level
        if self.info.get(level) is None:
            self.var_status.set(f"No '{level}' column on this dataset.")
            return
        ids, bad = self.parse_ids()
        codes, uniques = self.category_codes(level)
        uid = core.extract_leading_numeric_id(pd.Series(uniques)).to_numpy()
        colors = {t: (GREY if t in self.off_groups else np.array([255, 0, 0], np.uint8) if len(ids) == 1 else rank_color(i)) for i, t in enumerate(ids)}
        lut = np.tile(GREY, (len(uniques), 1))
        for t, col in colors.items():
            lut[uid == t] = col
        rgb = lut[codes]
        hit = np.zeros(len(uniques), np.int32)
        for i, t in enumerate(ids):
            hit[uid == t] = i + 1
        self.rank_run = hit[codes].astype(np.int32)
        order = np.argsort(self.rank_run, kind="stable")
        bgc = self.bg_codes(level) if self.bg else None

        def bg_fn(label):
            if not self.bg:
                return None
            w = self.bg["where"].get(label)
            if w is None:
                return None
            c = bgc[w] if bgc is not None else np.full(len(w), -1)
            h = np.where(c >= 0, hit[np.maximum(c, 0)], 0)
            col = np.where((h > 0)[:, None], lut[np.maximum(c, 0)], LIGHT_GREY).astype(np.uint8)
            o = np.argsort(h, kind="stable")
            return self.bg["x"][w][o], self.bg["y"][w][o], col[o], w[o]

        self.apply(rgb, order, bg_fn)
        counts = {t: int((uid[np.maximum(codes, 0)] == t)[codes >= 0].sum()) for t in ids}
        full = {t: (str(uniques[np.flatnonzero(uid == t)[0]]) if (uid == t).any() else f"{LEVEL_NAMES[level]} {t}") for t in ids}
        self.last_ids = {"level": level, "targets": ids, "colors": colors, "counts": counts}
        self.view_kind, self.view_token = "single", f"{level}_{'-'.join(str(t) for t in ids)}"
        self.view_title = f"{LEVEL_NAMES[level]} {', '.join(str(t) for t in ids)}"
        order_of = {t: i for i, t in enumerate(ids)}
        self.export_cat = {"rank": np.where((codes >= 0) & (hit[np.maximum(codes, 0)] > 0), hit[np.maximum(codes, 0)] - 1, -1), "title": LEVEL_NAMES[level],
                           "entries": [(hexcolor(colors[t]), full[t], counts[t]) for t in ids]}
        self.shown_values = None
        parts = []
        if ids:
            parts.append("Highlighting " + ", ".join(f"{t} ({counts[t]} cells)" for t in ids if counts[t]))
            if any(not counts[t] for t in ids):
                parts.append("not found: " + ", ".join(str(t) for t in ids if not counts[t]))
        else:
            parts.append(f"Enter one or more {LEVEL_NAMES[level].lower()} numbers.")
        if bad:
            parts.append("ignored: " + ", ".join(bad))
        self.var_status.set("; ".join(parts) + ".")
        self.single_names = full
        self.draw_legend(ids, colors, counts, full)
        self.values = None

    def draw_legend(self, ids, colors, counts, names=None):
        if len(ids) < 2:
            return
        self.legend.pack(fill=tk.X, pady=(6, 0), after=self.box)
        self.legend.delete("all")
        self.legend.config(height=22 * len(ids) + 8)
        rows = [("#%02x%02x%02x" % tuple(int(v) for v in colors[t]), str(t), f"({counts[t]})",
                 re.sub(r"^\s*\d+\s*", "", (names or {}).get(t, "")), t in self.off_groups) for t in ids]
        self.draw_legend_rows(self.legend, rows, 4)

    # ---- genes

    def symbols(self):
        return core.gene_symbol_list(self.adata, self.imputed_state, self.mode == "imputed")

    def on_gene_key(self, e):
        self.gene_entry.config(foreground="")
        if e.keysym in ("Return", "Up", "Down", "Escape", "Tab"):
            return
        token = self.var_gene.get().rsplit(",", 1)[-1].strip()
        matches = core.rank_gene_suggestions(self.symbols(), token, 6) if token else []
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
        text = self.var_gene.get()
        prefix = text.rsplit(",", 1)[0].strip() + ", " if "," in text else ""
        self.var_gene.set(prefix + self.sugg.get(sel[0]))
        self.sugg.pack_forget()
        self.gene_entry.focus_set()
        self.recolor()

    def color_gene(self):
        self.sugg.pack_forget()
        text = self.var_gene.get().strip()
        names = core.parse_gene_names(text)
        if not names:
            self.cbar.delete("all")
            self.var_status.set("Enter a gene name.")
            self.color_all_quiet()
            return
        if len(names) > MAX_GENES:
            self.var_status.set(f"Enter at most {MAX_GENES} gene names, separated by commas.")
            return
        imputed = self.mode == "imputed"
        if not imputed and not self.has_counts:
            self.var_status.set("No raw counts on this dataset (layers['counts'] missing).")
            return
        source = self.imputed_state["adata"] if imputed else self.adata
        _cols, resolved, missing = core.resolve_gene_names(source, names)
        if missing:
            self.gene_entry.config(foreground="red")
            self.var_status.set(f"Not found in the {'imputed' if imputed else 'standard'} dataset: {', '.join(missing)}.")
            return
        self.gene_entry.config(foreground="")
        self.var_gene.set(", ".join(resolved))
        scale = self.var_scale.get()
        key = (tuple(resolved), imputed, scale)
        if self.gene_cache is not None and self.gene_cache[0] == key:                   # (a legend click, or the same genes again)
            self.show_genes(resolved, imputed, scale, self.gene_cache[1])
            return
        adata = self.adata
        backed = self.adata_backed if self.bg else None
        bg_ids = self.bg["ids"] if self.bg else None

        def read_values(gene):
            col = core.find_gene_index(source, gene)
            if imputed:
                full = core._get_full_gene_column(source, col)
                pos = source.obs_names.get_indexer(adata.obs_names)
                vals = np.full(adata.n_obs, np.nan)
                found = pos >= 0
                vals[found] = full[pos[found]]
                if scale == "linear":
                    vals = np.exp2(vals) - 1.0
            else:
                expr = adata.layers["counts"][:, col]
                expr = expr.toarray() if hasattr(expr, "toarray") else expr
                vals = np.asarray(expr).ravel().astype(float)
                if scale == "log":
                    vals = np.log1p(vals)
            return vals

        def read_background(gene):
            """The gene's value in every cell of the sections (the whole unfiltered dataset), or None if it can't be had."""
            if backed is None:
                return None
            try:
                if imputed:
                    full = core._get_full_gene_column(source, core.find_gene_index(source, gene))
                    pos = source.obs_names.get_indexer(bg_ids)
                    out = np.full(len(bg_ids), np.nan)
                    out[pos >= 0] = full[pos[pos >= 0]]
                    return np.exp2(out) - 1.0 if scale == "linear" else out
                bcol = core.find_gene_index(backed, gene)
                if bcol is None:
                    return None
                out = core._get_full_gene_column(backed, bcol)
                return np.log1p(out) if scale == "log" else out
            except Exception as e:
                print(f"Could not read {gene} for the cells outside the selection: {e}", flush=True)
                return None

        def work():
            return [(read_values(g), read_background(g)) for g in resolved]

        def done(res):
            if isinstance(res, BaseException):
                self.var_status.set(f"Could not read {', '.join(resolved)}: {res}")
                return
            self.gene_cache = (key, res)
            self.hidden = set()                    # (a different query: every gene shown again)
            self.show_genes(resolved, imputed, scale, res)

        self.job(f"Reading {', '.join(resolved)}...", work, done)

    def color_all_quiet(self):
        mode = self.mode
        self.mode = "all"
        self.color_all()
        self.mode = mode

    def show_genes(self, genes, imputed, scale, per_gene):
        dataset = "imputed" if imputed else "standard"
        if self.mode not in ("gene", "imputed") or (self.mode == "imputed") != imputed:
            return
        self.values = [(g, pv[0]) for g, pv in zip(genes, per_gene)]
        self.view_token = "_".join(genes) + ("_imputed" if imputed else "")
        self.export_cat = None
        if len(genes) == 1:
            self.view_kind = "gene"
            self.view_title = f"{genes[0]}{' [imputed]' if imputed else ''}"
            self.shown_values = (genes[0], dataset, scale, per_gene[0][0])
            self.show_single(genes[0], dataset, scale, per_gene[0][0], per_gene[0][1])
        else:
            self.view_kind, self.multi_genes = "multi", list(genes)
            self.view_title = " / ".join(f"{g} ({CHANNEL_NAMES[i]})" for i, g in enumerate(genes)) + (" [imputed]" if imputed else "")
            self.show_multi(genes, dataset, scale, per_gene)

    def show_single(self, gene, dataset, scale, vals, bg_vals):
        fin = np.isfinite(vals)
        vmin, vmax = (float(np.nanmin(vals)), float(np.nanmax(vals))) if fin.any() else (0.0, 1.0)
        t = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)
        rgb = viridis_rgb(np.where(fin, t, 0))
        rgb[~fin] = GREY
        order = np.argsort(np.where(fin, vals, -np.inf), kind="stable")
        self.rank_run = np.argsort(order, kind="stable").astype(np.int64)       # the rank of each cell: highest expression on top

        def bg_fn(label):
            if not self.bg:
                return None
            w = self.bg["where"].get(label)
            if w is None:
                return None
            if bg_vals is None:
                return self.bg["x"][w], self.bg["y"][w], np.tile(LIGHT_GREY, (len(w), 1)), w
            v = bg_vals[w]
            ok = np.isfinite(v)
            tt = np.clip((v - vmin) / (vmax - vmin), 0, 1) if vmax > vmin else np.zeros_like(v)
            col = viridis_rgb(np.where(ok, tt, 0))
            col[~ok] = LIGHT_GREY
            o = np.argsort(np.where(ok, v, -np.inf), kind="stable")           # (highest on top)
            return self.bg["x"][w][o], self.bg["y"][w][o], col[o], w[o]

        self.apply(rgb, order, bg_fn)
        self.draw_colorbar(vmin, vmax, scale, dataset)
        self.var_status.set(f"Showing {gene}{' [imputed]' if dataset == 'imputed' else ''} (every cell of the sections, colored on the selection's scale)."
                            if bg_vals is not None else "(cells outside this run's selection are gray).")

    # ---- several genes: red / green / blue circles for the first three, the same three colors as small dots on top for the next three

    def show_multi(self, genes, dataset, scale, per_gene):
        n = len(genes)
        norm = []
        for vals, _bg in per_gene:
            fin = np.isfinite(vals)
            lo, hi = (float(np.nanmin(vals)), float(np.nanmax(vals))) if fin.any() else (0.0, 1.0)
            norm.append((np.clip((vals - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(vals), lo, hi))
        hidden = self.hidden
        zero = np.zeros(self.n)
        runn = [zero if i in hidden else norm[i][0] for i in range(n)]
        intensity = lambda arrs: np.maximum.reduce([np.nan_to_num(a, nan=0.0) for a in arrs])
        circ = runn[:3]
        rgb = (core.multi_gene_rgb(circ, (0.0, 0.0, 0.0)) * 255).astype(np.uint8)
        order = np.argsort(intensity(circ), kind="stable")                      # (the most expressing on top)
        self.rank_run = np.argsort(order, kind="stable").astype(np.int64)
        extra = []
        if n > 3:
            dots = runn[3:6]
            inten = intensity(dots)
            vis = np.flatnonzero(inten > 0)
            vis = vis[np.argsort(inten[vis], kind="stable")]
            drgb = (core.multi_gene_rgb([d[vis] for d in dots], (0.0, 0.0, 0.0)) * 255).astype(np.uint8)
            half = self.viewer_settings.get("gene456_shape") == "half circle"       # (the Settings choice: a half circle as big as the base one, or a small circle)
            extra = [{"x": self.coords[vis, 0], "y": self.coords[vis, 1], "rgb": drgb, "size": (4 if half else 2) * self.dot_scale(),
                      "shape": "half" if half else "circle"}]

        def bg_fn(label):
            if not self.bg:
                return None
            w = self.bg["where"].get(label)
            if w is None:
                return None
            chans = []
            for i in range(min(n, 3)):
                bgv = per_gene[i][1]
                if i in hidden or bgv is None:
                    chans.append(np.zeros(len(w)))
                    continue
                lo, hi = norm[i][1], norm[i][2]
                v = bgv[w]
                chans.append(np.clip((np.nan_to_num(v, nan=lo) - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros(len(w)))
            col = (core.multi_gene_rgb(chans, (0.0, 0.0, 0.0)) * 255).astype(np.uint8)
            o = np.argsort(intensity(chans), kind="stable")
            return self.bg["x"][w][o], self.bg["y"][w][o], col[o], w[o]

        self.apply(rgb, order, bg_fn, extra_umap_layers=extra, dark=True)
        to_log2 = 1.0 if dataset == "imputed" or scale == "linear" else float(np.log2(np.e))
        self.draw_multi_legend(genes, [(v[1] * to_log2, v[2] * to_log2) for v in norm], scale)
        parts = " and ".join(f"'{g}' ({CHANNEL_NAMES[i]})" for i, g in enumerate(genes))
        self.var_status.set(f"Showing {parts}{' [imputed]' if dataset == 'imputed' else ''}."
                            + (f" Hidden: {', '.join(genes[i] for i in sorted(hidden))}." if hidden else ""))

    def draw_multi_legend(self, genes, ranges, scale):
        c = self.cbar
        c.delete("all")
        self.cbar_photo = None
        row = 44
        c.config(height=8 + row * len(genes) + 22)
        for i, (g, (lo, hi)) in enumerate(zip(genes, ranges)):
            y = 8 + row * i
            off = i in self.hidden
            color = "#999999" if off else CHANNEL_HEX[i]
            r = 6 if i < 3 else 3                                  # (genes 4-6 are the small dots)
            if off:
                c.create_oval(18 - r, y + 11 - r, 18 + r, y + 11 + r, outline="#999999")
            else:
                c.create_oval(18 - r, y + 11 - r, 18 + r, y + 11 + r, fill=color, outline="")
            c.create_text(34, y + 11, anchor="w", text=g, fill=color, font=("Segoe UI", 10, "bold"), tags=f"row{i}")      # (centered on the circle)
            c.create_text(34, y + 22, anchor="nw", text="hidden" if off else f"{'' if scale == 'linear' else 'log2: '}{lo:.1f}–{hi:.1f}",
                          fill="#666666", font=("Segoe UI", 8))
        c.create_text(150, 8 + row * len(genes) + 2, anchor="n", text="click a gene to hide / show it", fill="#888888", font=("Segoe UI", 8, "italic"))
        self.multi_rows = (8, row, len(genes))

    def toggle_group(self, key, shift=False, order=()):
        """Switch a group between its color and gray (a click on its legend swatch); the cells, the swatch and the labels follow. With the shift
        key, every group in the legend from the one last clicked to this one is put in the state the last-clicked one was left in."""
        if self.busy:
            return
        order = list(order)
        last = self.last_group_click
        if shift and last is not None and last[0] in order and key in order:
            a, b = sorted((order.index(last[0]), order.index(key)))
            if last[1]:
                self.off_groups.update(order[a:b + 1])
            else:
                self.off_groups.difference_update(order[a:b + 1])
            self.last_group_click = (key, last[1])
        else:
            self.off_groups.symmetric_difference_update({key})
            self.last_group_click = (key, key in self.off_groups)
        self.recolor(keep_off=True)

    def on_group_list_click(self, e):
        rows = getattr(self, "group_rows", [])
        i = int((self.group_canvas.canvasy(e.y) - 4) // 22)
        if self.mode == "all" and 0 <= i < len(rows):
            self.toggle_group(rows[i], bool(e.state & 0x1), rows)

    def on_single_legend_click(self, e):
        ids = (self.last_ids or {}).get("targets", [])
        i = int((e.y - 4) // 22)
        if self.mode == "single" and len(ids) > 1 and 0 <= i < len(ids):
            self.toggle_group(ids[i], bool(e.state & 0x1), ids)

    # ---- the legend rows under the pointer

    def group_info_rows(self, level, label):
        """A group as table rows like a cell's: its place in the taxonomy (the levels above it, then its own), how many cells it has in the run and
        in how many sections. None if the run has no such group."""
        values = self.info.get(level)
        if values is None:
            return None
        mask = values == label
        if not mask.any():
            return None
        i0 = int(np.flatnonzero(mask)[0])
        levels = list(LEVEL_NAMES)
        chain = [level] if level == "leiden" else levels[:levels.index(level) + 1]
        rows = []
        for k in chain:
            v = self.info[k][i0] if self.info.get(k) is not None else None
            if v is None or str(v) == "nan":
                continue
            m = re.match(r"^\s*(\d+)\s*(.*)$", str(v))
            rows.append((LEVEL_NAMES[k], m.group(1) if m else "", m.group(2) if m else str(v)))
        rows.append(("Cells", f"{int(mask.sum())}", "in this run"))
        if self.section_of is not None:
            rows.append(("Sections", f"{len(np.unique(self.section_of[mask]))}", ""))
        return rows

    def show_legend_group(self, key):
        """Describe the legend row's group in the 'Under the pointer' box (key None: the pointer left the rows)."""
        if key == self.hover_group:
            return
        self.hover_group = key
        if key is None:
            self.show_info(None)
            return
        label = key if self.mode == "all" else self.single_names.get(key)
        rows = self.group_info_rows(self.level, label) if label is not None else None
        if rows is None:
            self.show_info(None)
        else:
            self.show_cell_rows(rows)

    def on_group_list_motion(self, e):
        rows = getattr(self, "group_rows", [])
        i = int((self.group_canvas.canvasy(e.y) - 4) // 22)
        self.show_legend_group(rows[i] if self.mode == "all" and 0 <= i < len(rows) else None)

    def on_single_legend_motion(self, e):
        ids = (self.last_ids or {}).get("targets", [])
        i = int((e.y - 4) // 22)
        self.show_legend_group(ids[i] if self.mode == "single" and len(ids) > 1 and 0 <= i < len(ids) else None)

    def on_legend_click(self, e):
        if self.mode not in ("gene", "imputed") or self.multi_rows is None or self.gene_cache is None:
            return
        top, row, n = self.multi_rows
        i = int((e.y - top) // row)
        if not 0 <= i < n or len(self.gene_cache[0][0]) < 2 or self.busy:
            return
        self.hidden.symmetric_difference_update({i})
        key = self.gene_cache[0]
        self.show_genes(list(key[0]), key[1], key[2], self.gene_cache[1])

    def draw_colorbar(self, vmin, vmax, scale, dataset):
        c = self.cbar
        c.delete("all")
        w = max(c.winfo_width(), 200)
        ramp = viridis_rgb(np.linspace(0, 1, w - 20))[None, :, :]
        img = np.repeat(ramp, 16, axis=0)
        self.cbar_photo = ImageTk.PhotoImage(Image.fromarray(img), master=c)
        c.create_image(10, 8, anchor="nw", image=self.cbar_photo)
        c.create_text(10, 28, anchor="nw", text=f"{vmin:.2f}", font=("Segoe UI", 9))
        c.create_text(w - 10, 28, anchor="ne", text=f"{vmax:.2f}", font=("Segoe UI", 9))
        c.create_text(w / 2, 50, text=SCALE_UNITS[(dataset, scale)], font=("Segoe UI", 9), fill="gray30")

    # ---- background work

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
            self.msgs.put((done, res))

        threading.Thread(target=target, daemon=True).start()
        return True

    def poll(self):
        try:
            while True:
                done, res = self.msgs.get_nowait()
                self.busy = False
                self.root.config(cursor="")
                done(res)
        except queue.Empty:
            pass
        for t in self.tiles:                  # (the table of the sections' z values is read in the background: it may arrive later)
            if t["title"].get() == t["token"]:
                z = core.section_z_text(t["label"], self.abc_cache, wait=False, short=True)
                if z:
                    t["title"].set(f"{t['token']}   {z}")
        self.ticks = getattr(self, "ticks", 0) + 1
        if self.ticks % 20 == 0 and not self.busy:          # (garbage is collected here, on the Tk thread: a worker thread freeing Tk images fails)
            gc.collect()
        self.poll_id = self.root.after(300, self.poll)

    # ---- hover and the link between the UMAP and the section panels

    def level_text(self, info):
        """The lines describing the cell under the pointer."""
        lines = []
        sec = info["sec"]
        if sec is not None:
            lines.append(f"Section {core.sanitize_section_token(sec)}")
        i = info["umap_idx"]
        for k in LEVEL_NAMES:
            if i is not None:
                v = self.info[k][i] if self.info.get(k) is not None else None
            else:
                arr = self.bg["levels"].get(k) if self.bg else None
                v = arr[info["bg_idx"]] if arr is not None else None
            if v is not None and str(v) != "nan":
                lines.append(f"{LEVEL_NAMES[k]}: {v}")
        z = core.section_z_text(sec, self.abc_cache, wait=False) if sec is not None else ""
        if z:
            lines.append(z)
        if i is not None and self.values is not None:
            for gene, vals in self.values:
                lines.append(f"{gene} = {vals[i]:.3g}" if np.isfinite(vals[i]) else f"{gene} = n/a")
        elif i is None:
            lines.append("(not in the UMAP)")
        return "\n".join(lines)

    def cell_rows(self, info):
        """The hovered cell as table rows: (what, number, name). A group's label '268 CS-PRNr-DR ...' is split into its number and its name."""
        rows = []
        sec = info["sec"]
        if sec is not None:
            rows.append(("Section", core.sanitize_section_token(sec), ""))
        i = info["umap_idx"]
        for k in LEVEL_NAMES:
            if i is not None:
                v = self.info[k][i] if self.info.get(k) is not None else None
            else:
                arr = self.bg["levels"].get(k) if self.bg else None
                v = arr[info["bg_idx"]] if arr is not None else None
            if v is None or str(v) == "nan":
                continue
            m = re.match(r"^\s*(\d+)\s*(.*)$", str(v))
            rows.append((LEVEL_NAMES[k], m.group(1) if m else "", m.group(2) if m else str(v)))
        z = core.section_z_text(sec, self.abc_cache, wait=False) if sec is not None else ""
        if z:
            rows.append(("", "", z))
        if i is not None and self.values is not None:
            for gene, vals in self.values:
                rows.append((gene, f"{vals[i]:.3g}" if np.isfinite(vals[i]) else "n/a", ""))
        elif i is None:
            rows.append(("", "", "(not in the UMAP)"))
        return rows

    def fit_cell_names(self):
        """The names wrap at the table's right edge, whatever the screen's scaling."""
        try:
            x2 = self.cell_grid.grid_bbox(column=2, row=0)[0]
        except tk.TclError:
            return
        wl = max(self.cell_grid.winfo_width() - x2 - 4, 60)
        for _a, _b, c in self.cell_cells:
            if int(str(c.cget("wraplength"))) != wl:
                c.config(wraplength=wl)

    def show_cell_rows(self, rows):
        self.cell_hint.pack_forget()
        self.cell_grid.pack(fill=tk.X, anchor="w")
        for k, (a, b, c) in enumerate(self.cell_cells):
            if k < len(rows):
                what, num, name = rows[k]
                a.config(text=what)
                b.config(text=num)
                c.config(text=name)
                a.grid()
                b.grid()
                c.grid()
            else:
                a.grid_remove()
                b.grid_remove()
                c.grid_remove()
        self.root.after_idle(self.fit_cell_names)

    def cell_info(self, i, source):
        return {"key": ("run", i), "umap_idx": i, "bg_idx": None, "sec": self.section_of[i] if self.section_of is not None else None,
                "x": self.cx[i] if self.cx is not None else np.nan, "y": self.cy[i] if self.cy is not None else np.nan, "source": source}

    def bg_cell_info(self, sec, g):
        """A cell of the section that is not in the run (so not in the UMAP)."""
        return {"key": ("bg", g), "umap_idx": None, "bg_idx": g, "sec": sec, "x": self.bg["x"][g], "y": self.bg["y"][g], "source": "section"}

    def show_info(self, info):
        """Ring the cell where the pointer is, mirror it in the other plot, and describe it."""
        if info is None:
            self.umap.set_marker("hover", None, None)
            for t in self.tiles:
                t["pc"].set_marker("hover", None, None)
            self.cell_grid.pack_forget()
            self.cell_hint.pack(anchor="w")
            return
        src = info["source"]
        i = info["umap_idx"]
        if i is None:
            self.umap.set_marker("hover", None, None)
        else:
            self.umap.set_marker("hover", *self.coords[i], style="ring" if src == "umap" else "mirror")
        finite = info["x"] is not None and np.isfinite(info["x"]) and np.isfinite(info["y"])
        for t in self.tiles:
            if t["label"] == info["sec"] and finite:
                t["pc"].set_marker("hover", info["x"], info["y"], style="ring" if src == "section" else "mirror")
            else:
                t["pc"].set_marker("hover", None, None)
        self.show_cell_rows(self.cell_rows(info))

    def hover(self, info):
        """The pointer is on a cell (info) or on none (None). The ring and description are immediate; once it rests, the plots dim and, a
        little later, the cell's group is marked with red '+' signs."""
        if info is None:
            self.cancel_hover_jobs()
            self.hover_key = None
            self.show_info(None)
            self.clear_highlights()
            return
        self.show_info(info)
        if info["key"] == self.hover_key:
            return
        self.cancel_hover_jobs()
        self.clear_highlights(keep_dim=True)
        self.hover_key = info["key"]
        self.hover_job = self.root.after(HOVER_HOLD_MS, lambda: self.hover_settled(info))
        self.group_job = self.root.after(GROUP_HOLD_MS, lambda: self.show_group(info))

    def cancel_hover_jobs(self):
        for name in ("hover_job", "group_job"):
            j = getattr(self, name, None)
            if j is not None:
                self.root.after_cancel(j)
                setattr(self, name, None)

    def hover_settled(self, info):
        self.hover_job = None
        self.set_dim(True)

    def set_dim(self, on):
        self.umap.set_dim(on)
        for t in self.tiles:
            t["pc"].set_dim(on)

    def clear_highlights(self, keep_dim=False):
        if not keep_dim:
            self.set_dim(False)
        self.umap.set_pluses(None)
        for t in self.tiles:
            t["pc"].set_pluses(None)

    def show_group(self, info):
        """Red '+' signs on every cell of the hovered cell's group (at the Group level), in the UMAP and in every section."""
        self.group_job = None
        level = self.level
        if self.info.get(level) is None:
            return
        i = info["umap_idx"]
        if i is not None:
            label = self.info[level][i]
        else:
            arr = self.bg["levels"].get(level) if self.bg else None
            if arr is None:
                return
            label = arr[info["bg_idx"]]
        if label is None or str(label) == "nan":
            return
        run_mask = self.info[level] == label
        if not run_mask.any():
            return                                   # (a group none of the run's cells is in: nothing in the UMAP to relate to)
        if i is not None:
            run_mask[i] = False                      # (the hovered cell keeps its own ring)
        self.umap.set_pluses((self.coords[run_mask, 0], self.coords[run_mask, 1]))
        barr = self.bg["levels"].get(level) if self.bg else None
        for t in self.tiles:
            if not self.frame_visible(t["frame"]):
                t["pc"].pluses = None
                continue
            if barr is not None:
                w = self.bg["where"][t["label"]]
                m = barr[w] == label
                xy = (self.bg["x"][w][m], self.bg["y"][w][m])
            else:
                idx = self.run_idx[t["label"]]
                m = self.info[level][idx] == label
                xy = (self.cx[idx][m], self.cy[idx][m])
            t["pc"].set_pluses(xy if len(xy[0]) else None)

    def umap_motion(self, px, py):
        if px is None or not self.viewer_settings.get("hover_enabled", True):
            self.hover(None)
            return
        i = self.umap.nearest(self.umap_tree, px, py)
        self.hover(None if i is None else self.cell_info(i, "umap"))

    def tile_motion(self, tile, px, py):
        if px is None or not self.viewer_settings.get("hover_enabled", True):
            self.hover(None)
            return
        if tile["tree"] is None:
            sec = tile["label"]
            if self.bg is not None and sec in self.bg["where"]:         # (any cell of the section, in the run or not)
                w = self.bg["where"][sec]
                x, y = self.bg["x"][w], self.bg["y"][w]
                good = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
                tile["good"] = w[good]
                tile["umap_of"] = self.run_index.get_indexer(self.bg["ids"][tile["good"]])
                pts = np.column_stack([x[good], y[good]])
            else:
                idx = self.run_idx[sec]
                good = np.flatnonzero(np.isfinite(self.cx[idx]) & np.isfinite(self.cy[idx]))
                tile["good"] = idx[good]
                tile["umap_of"] = tile["good"]
                pts = np.column_stack([self.cx[idx][good], self.cy[idx][good]])
            tile["tree"] = cKDTree(pts) if len(good) else False
        if tile["tree"] is False:
            return
        k = tile["pc"].nearest(tile["tree"], px, py)
        if k is None:
            self.hover(None)
            return
        u = int(tile["umap_of"][k])
        self.hover(self.cell_info(u, "section") if u >= 0 else self.bg_cell_info(tile["label"], int(tile["good"][k])))

    # ---- Settings and the UMAP parameters

    def open_settings(self):
        dialog = self.host.get("settings_dialog")
        if dialog is None or self.busy:
            return
        new = dialog(self.root, self.viewer_settings, umap_params=self.params, n_cells=self.n)
        if new is None:
            return
        new_params = new.pop("umap_params", None)        # (belong to this run, not to the user's preferences)
        shape_changed = new.get("gene456_shape") != self.viewer_settings.get("gene456_shape")
        size_changed = new.get("umap_circle_size") != self.viewer_settings.get("umap_circle_size")
        self.viewer_settings.update(new)
        self.host["save_viewer_settings"](self.viewer_settings)
        if not self.viewer_settings.get("hover_enabled", True):
            self.hover(None)
        if (size_changed or (shape_changed and self.mode in ("gene", "imputed"))) and not self.busy:
            self.recolor()
        if new_params is not None and self.params is not None:
            self.switch_variant(new_params)

    def switch_variant(self, new_params):
        """Show the UMAP for these parameters: from memory, else from the run's cache, else by calculating it (progress window)."""
        h = self.host
        key = h["umap_params_key"](new_params)
        if key == h["umap_params_key"](self.params):
            return
        entry = self.variants.get(key)
        message = None
        if entry is None:
            folder = self.run_folder if self.run_folder is not None else h["CACHE_DIR"]
            self.var_status.set("Working... recalculating the UMAP; the viewer is locked until it finishes.")
            self.root.config(cursor="watch")
            self.root.update()
            try:
                csv_path, message = h["ensure_umap_variant_csv"](self.adata, new_params, folder)
                if csv_path is None:
                    self.var_status.set(message or "The UMAP was not recalculated.")
                    return
                try:
                    new_coords, new_leiden = h["load_umap_variant"](csv_path, self.adata.obs_names)
                except Exception as e:
                    self.var_status.set(f"Could not read the recalculated UMAP: {e}")
                    return
            finally:
                self.root.config(cursor="")
            entry = (np.asarray(new_coords, dtype=float)[:, :2], None if new_leiden is None else np.asarray(new_leiden, dtype=object).astype(str))
            self.variants[key] = entry
        self.install_variant(entry[0], entry[1], new_params)
        self.var_status.set(message or f"UMAP shown with {h['umap_params_label'](new_params)}.")

    def install_variant(self, new_coords, new_leiden, new_params):
        """Swap the embedding and its Leiden clustering in place and redraw everything derived from either."""
        h = self.host
        self.hover(None)
        self.coords = np.asarray(new_coords, dtype=float)[:, :2].copy()
        if new_leiden is not None:
            self.adata.obs[LEIDEN_KEY] = pd.Categorical(new_leiden)
            self.info["leiden"] = np.asarray(new_leiden).astype(str)
        else:
            self.adata.obs.drop(columns=[LEIDEN_KEY], inplace=True, errors="ignore")
            self.info["leiden"] = None
        for k in [k for k in self.cache if isinstance(k, tuple) and k[1] == "leiden"]:
            del self.cache[k]
        self.levels = [k for k in LEVEL_NAMES if self.info[k] is not None]
        if self.level not in self.levels:
            self.level = self.levels[0]
            self.var_level.set(self.level)
        self.build_level_radios()
        c = self.coords
        padx = (c[:, 0].max() - c[:, 0].min()) * 0.05 or 1.0
        pady = (c[:, 1].max() - c[:, 1].min()) * 0.05 or 1.0
        self.umap.home = (c[:, 0].min() - padx, c[:, 0].max() + padx, c[:, 1].min() - pady, c[:, 1].max() + pady)
        self.umap.fit()
        self.umap_tree = cKDTree(c)
        self.params = h["normalize_umap_params"](new_params)
        self.root.title(self.window_title())
        if self.run_folder is not None:
            h["write_last_umap_params"](self.run_folder, self.params)
        self.recolor()

    def window_title(self):
        title = "Interactive UMAP viewer"
        h = self.host
        if self.params is not None and not h["umap_params_are_default"](self.params):
            title = f"{title} ({h['umap_params_label'](self.params)})"
        if self.run_folder is not None:
            title = f"{Path(self.run_folder).name} \u2014 {title}"
        return title

    def open_data_folder(self):
        folder = Path(self.run_folder) if self.run_folder is not None else Path(self.host["CACHE_DIR"])
        folder.mkdir(parents=True, exist_ok=True)
        self.host["open_with_default_viewer"](folder)

    # ---- closing

    def close(self):
        if self.gc_was_enabled:
            gc.enable()
        try:
            self.root.after_cancel(self.poll_id)
        except Exception:
            pass
        if self.own_root:
            try:
                self.root.quit()
            except tk.TclError:
                pass
        self.root.destroy()

    def run(self):
        try:
            self.root.state("zoomed")
        except tk.TclError:
            pass
        self.root.lift()
        self.root.focus_force()
        if self.own_root:
            self.root.mainloop()
        else:
            self.root.wait_window(self.root)


def show_umap_window_tk(adata, abc_cache, imputed_state=None, adata_backed=None, rois=None, run_folder=None, ensure_imputed=None, title=None,
                        parent=None, host=None, params=None):
    """Open the viewer and return when it is closed. `host`: the browser's own functions (see UmapWindow); `params`: the UMAP parameters
    the embedding in `adata` was made with."""
    if "X_umap" not in adata.obsm:
        print("No UMAP embedding available; skipping the interactive UMAP viewer.")
        return
    win = UmapWindow(adata, abc_cache, imputed_state, adata_backed, rois, run_folder, ensure_imputed, title or "Interactive UMAP viewer", parent,
                     host=host, params=params)
    win.run()
